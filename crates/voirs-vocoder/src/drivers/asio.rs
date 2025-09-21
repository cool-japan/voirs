//! Windows ASIO audio driver implementation
//!
//! This module provides Windows ASIO (Audio Stream Input/Output) support
//! for professional low-latency audio applications using the ASIO API through cpal.

use super::{
    AudioCallback, AudioDeviceInfo, AudioDriver, AudioDriverError, AudioStreamConfig,
    AudioStreamMetrics, DriverResult,
};
use async_trait::async_trait;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{
    BuildStreamError, Device, Host, SampleFormat, SampleRate, Stream, StreamConfig,
    SupportedStreamConfig,
};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Windows ASIO audio driver
pub struct AsioDriver {
    host: Host,
    stream: Option<Stream>,
    is_running: AtomicBool,
    metrics: Arc<Mutex<AudioStreamMetrics>>,
    start_time: Option<Instant>,
}

impl AsioDriver {
    /// Create a new ASIO driver instance
    pub fn new() -> DriverResult<Self> {
        // Try to get ASIO host
        let host = cpal::host_from_id(cpal::HostId::Asio).map_err(|e| {
            AudioDriverError::InternalError(format!("Failed to get ASIO host: {e}"))
        })?;

        Ok(Self {
            host,
            stream: None,
            is_running: AtomicBool::new(false),
            metrics: Arc::new(Mutex::new(AudioStreamMetrics::default())),
            start_time: None,
        })
    }

    /// Convert cpal device to our AudioDeviceInfo
    fn device_to_info(&self, device: &Device) -> DriverResult<AudioDeviceInfo> {
        let name = device.name().map_err(|e| {
            AudioDriverError::InternalError(format!("Failed to get device name: {e}"))
        })?;

        // Get supported output configurations
        let mut supported_configs = device
            .supported_output_configs()
            .map_err(|e| {
                AudioDriverError::InternalError(format!("Failed to get supported configs: {e}"))
            })?
            .collect::<Vec<_>>();

        if supported_configs.is_empty() {
            return Err(AudioDriverError::InternalError(
                "No supported output configurations".to_string(),
            ));
        }

        // Sort by sample rate for consistent ordering
        supported_configs.sort_by_key(|config| config.min_sample_rate().0);

        let config = &supported_configs[0];
        let sample_rate_range = config.min_sample_rate().0..=config.max_sample_rate().0;
        let supported_sample_rates = vec![
            16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 176400, 192000,
        ]
        .into_iter()
        .filter(|&rate| sample_rate_range.contains(&rate))
        .collect();

        // ASIO typically supports very low buffer sizes
        let min_buffer_size = 32;
        let max_buffer_size = 2048;
        let default_buffer_size = 128;

        Ok(AudioDeviceInfo {
            id: name.clone(),
            name,
            is_default: false, // Will be set by caller if this is default
            supported_sample_rates,
            max_output_channels: config.channels() as u32,
            min_buffer_size,
            max_buffer_size,
            default_buffer_size,
        })
    }

    /// Get the optimal stream configuration for given parameters
    fn get_stream_config(
        &self,
        device: &Device,
        config: &AudioStreamConfig,
    ) -> DriverResult<StreamConfig> {
        let supported_configs = device.supported_output_configs().map_err(|e| {
            AudioDriverError::InternalError(format!("Failed to get supported configs: {e}"))
        })?;

        // Find a configuration that matches our requirements
        for supported_config in supported_configs {
            let sample_rate_range =
                supported_config.min_sample_rate().0..=supported_config.max_sample_rate().0;

            if sample_rate_range.contains(&config.sample_rate)
                && supported_config.channels() >= config.channels as u16
            {
                let stream_config = StreamConfig {
                    channels: config.channels as u16,
                    sample_rate: SampleRate(config.sample_rate),
                    buffer_size: cpal::BufferSize::Fixed(config.buffer_size),
                };

                return Ok(stream_config);
            }
        }

        Err(AudioDriverError::UnsupportedSampleRate {
            rate: config.sample_rate,
        })
    }
}

#[async_trait(?Send)]
impl AudioDriver for AsioDriver {
    async fn enumerate_devices(&self) -> DriverResult<Vec<AudioDeviceInfo>> {
        let devices = self.host.output_devices().map_err(|e| {
            AudioDriverError::InternalError(format!("Failed to enumerate devices: {e}"))
        })?;

        let mut device_infos = Vec::new();

        for device in devices {
            match self.device_to_info(&device) {
                Ok(info) => device_infos.push(info),
                Err(e) => {
                    tracing::warn!("Failed to get info for device: {e}");
                    continue;
                }
            }
        }

        if device_infos.is_empty() {
            return Err(AudioDriverError::DeviceNotFound(
                "No ASIO output devices found".to_string(),
            ));
        }

        Ok(device_infos)
    }

    async fn default_device(&self) -> DriverResult<AudioDeviceInfo> {
        let device = self.host.default_output_device().ok_or_else(|| {
            AudioDriverError::DeviceNotFound("No default ASIO device".to_string())
        })?;

        let mut info = self.device_to_info(&device)?;
        info.is_default = true;
        Ok(info)
    }

    async fn initialize_stream(
        &mut self,
        device_id: Option<&str>,
        config: AudioStreamConfig,
        callback: AudioCallback,
    ) -> DriverResult<()> {
        // Get the device
        let device = if let Some(id) = device_id {
            // Find device by ID/name
            let devices = self.host.output_devices().map_err(|e| {
                AudioDriverError::InternalError(format!("Failed to enumerate devices: {e}"))
            })?;

            devices
                .find(|d| d.name().map(|name| name == id).unwrap_or(false))
                .ok_or_else(|| AudioDriverError::DeviceNotFound(id.to_string()))?
        } else {
            // Use default device
            self.host
                .default_output_device()
                .ok_or_else(|| AudioDriverError::DeviceNotFound("No default device".to_string()))?
        };

        // Get stream configuration
        let stream_config = self.get_stream_config(&device, &config)?;

        // Create metrics tracking
        let metrics = Arc::clone(&self.metrics);
        let frames_processed = Arc::new(AtomicU64::new(0));
        let underruns = Arc::new(AtomicU32::new(0));
        let overruns = Arc::new(AtomicU32::new(0));

        let frames_processed_clone = Arc::clone(&frames_processed);
        let underruns_clone = Arc::clone(&underruns);
        let _overruns_clone = Arc::clone(&overruns);

        // Create the audio callback wrapper
        let stream_callback = move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let start_time = Instant::now();

            // Call the user callback
            match callback(data) {
                Ok(()) => {
                    // Update frame count
                    frames_processed_clone.fetch_add(data.len() as u64, Ordering::Relaxed);
                }
                Err(_) => {
                    // Fill with silence on error
                    data.fill(0.0);
                    underruns_clone.fetch_add(1, Ordering::Relaxed);
                }
            }

            // Update metrics
            let processing_time = start_time.elapsed().as_secs_f32() * 1000.0;
            let buffer_duration = (data.len() as f32 / config.sample_rate as f32) * 1000.0;
            let cpu_load = (processing_time / buffer_duration) * 100.0;

            let mut m = metrics.lock();
            m.frames_processed = frames_processed_clone.load(Ordering::Relaxed);
            m.underruns = underruns_clone.load(Ordering::Relaxed);
            m.overruns = overruns.load(Ordering::Relaxed);
            m.current_latency_ms = config.target_latency_ms;
            m.cpu_load_percent = cpu_load;
            m.is_active = true;
        };

        // Create error callback
        let error_callback = move |err| {
            tracing::error!("ASIO stream error: {err}");
        };

        // Build the stream
        let stream = device
            .build_output_stream(
                &stream_config,
                stream_callback,
                error_callback,
                None, // timeout
            )
            .map_err(|e| match e {
                BuildStreamError::DeviceNotAvailable => {
                    AudioDriverError::DeviceNotFound("Device not available".to_string())
                }
                BuildStreamError::InvalidArgument => AudioDriverError::UnsupportedSampleRate {
                    rate: config.sample_rate,
                },
                _ => AudioDriverError::StreamInitFailed(format!("Failed to build stream: {e}")),
            })?;

        self.stream = Some(stream);
        self.start_time = Some(Instant::now());

        Ok(())
    }

    async fn start_stream(&mut self) -> DriverResult<()> {
        if let Some(stream) = &self.stream {
            stream.play().map_err(|e| {
                AudioDriverError::StreamInitFailed(format!("Failed to start stream: {e}"))
            })?;
            self.is_running.store(true, Ordering::Relaxed);
            Ok(())
        } else {
            Err(AudioDriverError::StreamNotInitialized)
        }
    }

    async fn stop_stream(&mut self) -> DriverResult<()> {
        if let Some(stream) = &self.stream {
            stream.pause().map_err(|e| {
                AudioDriverError::InternalError(format!("Failed to stop stream: {e}"))
            })?;
            self.is_running.store(false, Ordering::Relaxed);

            // Update metrics
            let mut metrics = self.metrics.lock();
            metrics.is_active = false;

            Ok(())
        } else {
            Err(AudioDriverError::StreamNotInitialized)
        }
    }

    fn is_stream_running(&self) -> bool {
        self.is_running.load(Ordering::Relaxed)
    }

    fn get_metrics(&self) -> AudioStreamMetrics {
        self.metrics.lock().clone()
    }

    fn driver_name(&self) -> &'static str {
        "ASIO"
    }

    fn is_available() -> bool {
        // Check if ASIO is available on this system
        #[cfg(target_os = "windows")]
        {
            cpal::host_from_id(cpal::HostId::Asio).is_ok()
        }
        #[cfg(not(target_os = "windows"))]
        {
            false
        }
    }
}

impl Drop for AsioDriver {
    fn drop(&mut self) {
        if self.is_running.load(Ordering::Relaxed) {
            if let Some(stream) = &self.stream {
                let _ = stream.pause();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_asio_driver_creation() {
        // Only test on Windows where ASIO might be available
        #[cfg(target_os = "windows")]
        {
            if AsioDriver::is_available() {
                let driver = AsioDriver::new();
                assert!(driver.is_ok());

                let driver = driver.unwrap();
                assert_eq!(driver.driver_name(), "ASIO");
                assert!(!driver.is_stream_running());
            }
        }
    }

    #[test]
    fn test_asio_availability() {
        // On Windows, ASIO might be available
        #[cfg(target_os = "windows")]
        {
            // Just test that the function runs without panicking
            let _available = AsioDriver::is_available();
        }

        // On non-Windows, ASIO should not be available
        #[cfg(not(target_os = "windows"))]
        {
            assert!(!AsioDriver::is_available());
        }
    }

    #[cfg(target_os = "windows")]
    #[tokio::test]
    async fn test_asio_device_enumeration() {
        if AsioDriver::is_available() {
            let driver = AsioDriver::new().expect("Failed to create ASIO driver");

            // Try to enumerate devices - might fail if no ASIO devices are available
            let result = driver.enumerate_devices().await;
            match result {
                Ok(devices) => {
                    // If we get devices, they should have valid properties
                    for device in devices {
                        assert!(!device.name.is_empty());
                        assert!(!device.supported_sample_rates.is_empty());
                        assert!(device.max_output_channels > 0);
                        assert!(device.min_buffer_size > 0);
                        assert!(device.max_buffer_size > device.min_buffer_size);
                    }
                }
                Err(AudioDriverError::DeviceNotFound(_)) => {
                    // This is acceptable - no ASIO devices might be installed
                }
                Err(e) => {
                    panic!("Unexpected error: {e}");
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    #[tokio::test]
    async fn test_asio_default_device() {
        if AsioDriver::is_available() {
            let driver = AsioDriver::new().expect("Failed to create ASIO driver");

            // Try to get default device - might fail if no ASIO devices are available
            let result = driver.default_device().await;
            match result {
                Ok(device) => {
                    assert!(device.is_default);
                    assert!(!device.name.is_empty());
                    assert!(!device.supported_sample_rates.is_empty());
                }
                Err(AudioDriverError::DeviceNotFound(_)) => {
                    // This is acceptable - no ASIO devices might be installed
                }
                Err(e) => {
                    panic!("Unexpected error: {e}");
                }
            }
        }
    }

    #[test]
    fn test_asio_metrics() {
        #[cfg(target_os = "windows")]
        {
            if AsioDriver::is_available() {
                let driver = AsioDriver::new().expect("Failed to create ASIO driver");
                let metrics = driver.get_metrics();

                assert_eq!(metrics.frames_processed, 0);
                assert_eq!(metrics.underruns, 0);
                assert_eq!(metrics.overruns, 0);
                assert!(!metrics.is_active);
            }
        }
    }
}
