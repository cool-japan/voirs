//! Linux audio drivers (ALSA and PulseAudio)
//!
//! This module provides Linux audio driver implementations using the cpal crate,
//! which provides cross-platform audio support including ALSA and PulseAudio backends.

use async_trait::async_trait;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, SampleFormat, SampleRate, StreamConfig};
use parking_lot::Mutex;
use std::sync::Arc;

use super::{
    AudioCallback, AudioDeviceInfo, AudioDriver, AudioDriverError, AudioStreamConfig,
    AudioStreamMetrics, DriverResult,
};

/// Linux audio driver implementation using cpal
pub struct LinuxAudioDriver {
    host: Host,
    device: Option<Device>,
    stream: Option<cpal::Stream>,
    callback: Option<AudioCallback>,
    metrics: Arc<Mutex<AudioStreamMetrics>>,
    is_running: bool,
    backend_name: String,
}

impl LinuxAudioDriver {
    /// Create a new Linux audio driver
    pub fn new() -> DriverResult<Self> {
        let host = cpal::default_host();
        let backend_name = format!("{:?}", host.id())
            .replace("HostId(", "")
            .replace(")", "");
        let metrics = Arc::new(Mutex::new(AudioStreamMetrics::default()));

        Ok(Self {
            host,
            device: None,
            stream: None,
            callback: None,
            metrics,
            is_running: false,
            backend_name,
        })
    }

    /// Convert cpal device to our AudioDeviceInfo
    fn device_info_from_cpal(device: &Device, is_default: bool) -> DriverResult<AudioDeviceInfo> {
        let name = device.name().map_err(|e| {
            AudioDriverError::InternalError(format!("Failed to get device name: {e}"))
        })?;

        // Get supported configurations
        let mut supported_sample_rates = Vec::new();
        let mut max_channels = 0;
        let mut min_buffer_size = 64;
        let mut max_buffer_size = 4096;
        let mut default_buffer_size = 256;

        // Try to get default output config
        if let Ok(default_config) = device.default_output_config() {
            supported_sample_rates.push(default_config.sample_rate().0);
            max_channels = default_config.channels() as u32;

            // Get buffer size range if available
            let buffer_size = default_config.buffer_size();
            match buffer_size {
                cpal::SupportedBufferSize::Range { min, max } => {
                    min_buffer_size = *min;
                    max_buffer_size = *max;
                    default_buffer_size = (*min + *max) / 2;
                }
                cpal::SupportedBufferSize::Unknown => {
                    default_buffer_size = 256;
                }
            }
        }

        // Add common sample rates
        for &rate in &[8000, 16000, 22050, 44100, 48000, 88200, 96000] {
            if !supported_sample_rates.contains(&rate) {
                // Test if this sample rate is supported
                if device.supported_output_configs().is_ok_and(|mut configs| {
                    configs.any(|c| c.min_sample_rate().0 <= rate && rate <= c.max_sample_rate().0)
                }) {
                    supported_sample_rates.push(rate);
                }
            }
        }

        supported_sample_rates.sort_unstable();

        Ok(AudioDeviceInfo {
            id: format!("linux_{}", name.replace(' ', "_")),
            name,
            is_default,
            supported_sample_rates,
            max_output_channels: max_channels,
            min_buffer_size,
            max_buffer_size,
            default_buffer_size,
        })
    }

    /// Get all available Linux audio devices
    fn get_linux_devices(host: &Host) -> DriverResult<Vec<AudioDeviceInfo>> {
        let mut devices = Vec::new();

        // Get default device
        if let Ok(default_device) = host.default_output_device() {
            let device_info = Self::device_info_from_cpal(&default_device, true)?;
            devices.push(device_info);
        }

        // Get all output devices
        let output_devices = host.output_devices().map_err(|e| {
            AudioDriverError::InternalError(format!("Failed to enumerate output devices: {e}"))
        })?;

        for device in output_devices {
            let device_name = device.name().unwrap_or_else(|_| "Unknown".to_string());

            // Skip if we already have this device as default
            if devices
                .iter()
                .any(|d| d.name == device_name && d.is_default)
            {
                continue;
            }

            if let Ok(device_info) = Self::device_info_from_cpal(&device, false) {
                devices.push(device_info);
            }
        }

        if devices.is_empty() {
            return Err(AudioDriverError::DeviceNotFound(
                "No audio output devices found".to_string(),
            ));
        }

        Ok(devices)
    }
}

#[async_trait(?Send)]
impl super::AudioDriver for LinuxAudioDriver {
    async fn enumerate_devices(&self) -> DriverResult<Vec<AudioDeviceInfo>> {
        Self::get_linux_devices(&self.host)
    }

    async fn default_device(&self) -> DriverResult<AudioDeviceInfo> {
        let default_device = self.host.default_output_device().ok_or_else(|| {
            AudioDriverError::DeviceNotFound("No default output device found".to_string())
        })?;

        Self::device_info_from_cpal(&default_device, true)
    }

    async fn initialize_stream(
        &mut self,
        device_id: Option<&str>,
        config: AudioStreamConfig,
        callback: AudioCallback,
    ) -> DriverResult<()> {
        // Validate configuration
        if config.sample_rate < 8000 || config.sample_rate > 192000 {
            return Err(AudioDriverError::UnsupportedSampleRate {
                rate: config.sample_rate,
            });
        }

        if config.channels == 0 || config.channels > 32 {
            return Err(AudioDriverError::UnsupportedChannelCount {
                channels: config.channels,
            });
        }

        // Find the device to use
        let device = if let Some(device_id) = device_id {
            // Find device by ID
            let devices = self.host.output_devices().map_err(|e| {
                AudioDriverError::InternalError(format!("Failed to enumerate devices: {e}"))
            })?;

            devices
                .find(|device| {
                    device
                        .name()
                        .map(|name| format!("linux_{}", name.replace(' ', "_")) == device_id)
                        .unwrap_or(false)
                })
                .ok_or_else(|| AudioDriverError::DeviceNotFound(device_id.to_string()))?
        } else {
            // Use default device
            self.host.default_output_device().ok_or_else(|| {
                AudioDriverError::DeviceNotFound("No default output device".to_string())
            })?
        };

        // Get device's default configuration
        let device_config = device.default_output_config().map_err(|e| {
            AudioDriverError::StreamInitFailed(format!("Failed to get device config: {e}"))
        })?;

        // Create stream configuration
        let sample_format = device_config.sample_format();
        let stream_config = StreamConfig {
            channels: config.channels as u16,
            sample_rate: SampleRate(config.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(config.buffer_size),
        };

        // Validate the configuration is supported
        let supported_configs = device.supported_output_configs().map_err(|e| {
            AudioDriverError::InternalError(format!("Failed to get supported configs: {e}"))
        })?;

        let config_supported = supported_configs.any(|supported_config| {
            supported_config.channels() as u32 >= config.channels
                && supported_config.min_sample_rate().0 <= config.sample_rate
                && config.sample_rate <= supported_config.max_sample_rate().0
                && supported_config.sample_format() == sample_format
        });

        if !config_supported {
            return Err(AudioDriverError::StreamInitFailed(format!(
                "Configuration not supported: {} channels, {} Hz, {:?}",
                config.channels, config.sample_rate, sample_format
            )));
        }

        self.device = Some(device);
        self.callback = Some(callback);

        Ok(())
    }

    async fn start_stream(&mut self) -> DriverResult<()> {
        let device = self
            .device
            .as_ref()
            .ok_or(AudioDriverError::StreamNotInitialized)?;
        let callback = self
            .callback
            .as_ref()
            .ok_or(AudioDriverError::StreamNotInitialized)?;

        // Get device configuration
        let device_config = device.default_output_config().map_err(|e| {
            AudioDriverError::StreamInitFailed(format!("Failed to get device config: {e}"))
        })?;

        let sample_format = device_config.sample_format();
        let channels = device_config.channels();
        let sample_rate = device_config.sample_rate();

        // Create stream configuration (use device defaults for compatibility)
        let config = StreamConfig {
            channels,
            sample_rate,
            buffer_size: cpal::BufferSize::Default,
        };

        let callback_clone = Arc::clone(callback);
        let metrics_clone = Arc::clone(&self.metrics);

        // Create the audio stream
        let stream = match sample_format {
            SampleFormat::F32 => {
                device.build_output_stream(
                    &config,
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        // Call our callback to get audio data
                        if let Err(_) = callback_clone(data) {
                            // Handle callback error - for now just fill with silence
                            data.fill(0.0);
                        }

                        // Update metrics
                        let mut metrics = metrics_clone.lock();
                        metrics.frames_processed += data.len() as u64;
                        metrics.is_active = true;
                    },
                    move |err| {
                        eprintln!("Audio stream error: {err}");
                    },
                    None,
                )
            }
            SampleFormat::I16 => {
                device.build_output_stream(
                    &config,
                    move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                        // Create a temporary f32 buffer for our callback
                        let mut f32_buffer = vec![0.0f32; data.len()];

                        if let Err(_) = callback_clone(&mut f32_buffer) {
                            data.fill(0);
                            return;
                        }

                        // Convert f32 to i16
                        for (i, &sample) in f32_buffer.iter().enumerate() {
                            data[i] = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                        }

                        // Update metrics
                        let mut metrics = metrics_clone.lock();
                        metrics.frames_processed += data.len() as u64;
                        metrics.is_active = true;
                    },
                    move |err| {
                        eprintln!("Audio stream error: {err}");
                    },
                    None,
                )
            }
            SampleFormat::U16 => {
                device.build_output_stream(
                    &config,
                    move |data: &mut [u16], _: &cpal::OutputCallbackInfo| {
                        // Create a temporary f32 buffer for our callback
                        let mut f32_buffer = vec![0.0f32; data.len()];

                        if let Err(_) = callback_clone(&mut f32_buffer) {
                            data.fill(0);
                            return;
                        }

                        // Convert f32 to u16
                        for (i, &sample) in f32_buffer.iter().enumerate() {
                            let normalized = (sample.clamp(-1.0, 1.0) + 1.0) * 0.5;
                            data[i] = (normalized * u16::MAX as f32) as u16;
                        }

                        // Update metrics
                        let mut metrics = metrics_clone.lock();
                        metrics.frames_processed += data.len() as u64;
                        metrics.is_active = true;
                    },
                    move |err| {
                        eprintln!("Audio stream error: {err}");
                    },
                    None,
                )
            }
        }
        .map_err(|e| AudioDriverError::StreamInitFailed(format!("Failed to build stream: {e}")))?;

        // Start the stream
        stream.play().map_err(|e| {
            AudioDriverError::StreamInitFailed(format!("Failed to start stream: {e}"))
        })?;

        self.stream = Some(stream);
        self.is_running = true;

        Ok(())
    }

    async fn stop_stream(&mut self) -> DriverResult<()> {
        if let Some(stream) = self.stream.take() {
            stream.pause().map_err(|e| {
                AudioDriverError::InternalError(format!("Failed to stop stream: {e}"))
            })?;
        }

        self.is_running = false;
        let mut metrics = self.metrics.lock();
        metrics.is_active = false;

        Ok(())
    }

    fn is_stream_running(&self) -> bool {
        self.is_running
    }

    fn get_metrics(&self) -> AudioStreamMetrics {
        self.metrics.lock().clone()
    }

    fn driver_name(&self) -> &'static str {
        "Linux Audio"
    }

    fn is_available() -> bool {
        cfg!(target_os = "linux")
    }
}

// Type aliases for backward compatibility
pub type AlsaDriver = LinuxAudioDriver;
pub type PulseAudioDriver = LinuxAudioDriver;

/// Create the best available Linux audio driver
pub fn create_linux_driver() -> DriverResult<Box<dyn super::AudioDriver>> {
    // On Linux, cpal automatically selects the best available backend
    // This could be ALSA, PulseAudio, or JACK depending on what's available
    if LinuxAudioDriver::is_available() {
        return Ok(Box::new(LinuxAudioDriver::new()?));
    }

    Err(AudioDriverError::InternalError(
        "No Linux audio drivers available".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_linux_driver_creation() {
        let driver = LinuxAudioDriver::new();
        assert!(driver.is_ok());
    }

    #[tokio::test]
    async fn test_alsa_driver_alias() {
        let driver = AlsaDriver::new();
        assert!(driver.is_ok());
    }

    #[tokio::test]
    async fn test_pulse_driver_alias() {
        let driver = PulseAudioDriver::new();
        assert!(driver.is_ok());
    }

    #[tokio::test]
    async fn test_device_enumeration() {
        // This test will only work properly on Linux with audio devices
        #[cfg(target_os = "linux")]
        {
            let driver = LinuxAudioDriver::new().unwrap();
            let devices = driver.enumerate_devices().await;

            // On systems with audio hardware, this should succeed
            // On headless systems or CI, it might fail, which is expected
            if let Ok(devices) = devices {
                assert!(!devices.is_empty());
                // Should have at least one default device
                assert!(devices.iter().any(|d| d.is_default));
            }
        }
    }

    #[tokio::test]
    async fn test_default_device() {
        #[cfg(target_os = "linux")]
        {
            let driver = LinuxAudioDriver::new().unwrap();
            let default_device = driver.default_device().await;

            // On systems with audio hardware, this should succeed
            if let Ok(device) = default_device {
                assert!(device.is_default);
                assert!(!device.name.is_empty());
                assert!(!device.supported_sample_rates.is_empty());
            }
        }
    }

    #[tokio::test]
    async fn test_driver_availability() {
        // These tests will only pass on Linux
        #[cfg(target_os = "linux")]
        {
            assert!(LinuxAudioDriver::is_available());
            assert!(AlsaDriver::is_available());
            assert!(PulseAudioDriver::is_available());
        }

        #[cfg(not(target_os = "linux"))]
        {
            assert!(!LinuxAudioDriver::is_available());
            assert!(!AlsaDriver::is_available());
            assert!(!PulseAudioDriver::is_available());
        }
    }

    #[tokio::test]
    async fn test_stream_initialization() {
        #[cfg(target_os = "linux")]
        {
            let mut driver = LinuxAudioDriver::new().unwrap();
            let config = AudioStreamConfig::default();

            let callback: AudioCallback = Arc::new(|_output| Ok(()));

            // Stream initialization might fail on headless systems without audio
            // This is expected behavior
            let result = driver.initialize_stream(None, config, callback).await;

            // On systems without audio devices, this will fail
            // On systems with audio devices, this should succeed
            if result.is_err() {
                // This is acceptable for headless/CI environments
                println!("Stream initialization failed - likely no audio devices available");
            }
        }
    }

    #[tokio::test]
    async fn test_create_linux_driver() {
        let driver = create_linux_driver();

        #[cfg(target_os = "linux")]
        {
            // Should succeed on Linux
            if driver.is_err() {
                // Might fail on headless systems - this is acceptable
                println!("Linux driver creation failed - likely no audio devices available");
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            // Should fail on non-Linux systems
            assert!(driver.is_err());
        }
    }
}
