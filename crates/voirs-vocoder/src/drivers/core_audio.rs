//! Core Audio driver implementation for macOS
//!
//! This module provides real-time audio output using Apple's Core Audio framework
//! through the cpal crate, optimized for low-latency audio applications.

use async_trait::async_trait;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BuildStreamError, Device, Host, SampleFormat, SampleRate, StreamConfig};
use parking_lot::Mutex;
use std::sync::Arc;

use super::{
    AudioCallback, AudioDeviceInfo, AudioDriver, AudioDriverError, AudioStreamConfig,
    AudioStreamMetrics, DriverResult,
};

/// Core Audio driver for macOS
pub struct CoreAudioDriver {
    host: Host,
    device: Option<Device>,
    stream: Option<cpal::Stream>,
    callback: Option<AudioCallback>,
    metrics: Arc<Mutex<AudioStreamMetrics>>,
    is_running: bool,
}

impl CoreAudioDriver {
    /// Create a new Core Audio driver instance
    pub fn new() -> DriverResult<Self> {
        let host = cpal::default_host();
        let metrics = Arc::new(Mutex::new(AudioStreamMetrics::default()));

        Ok(Self {
            host,
            device: None,
            stream: None,
            callback: None,
            metrics,
            is_running: false,
        })
    }

    /// Convert cpal device to our AudioDeviceInfo
    fn device_info_from_cpal(device: &Device) -> DriverResult<AudioDeviceInfo> {
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
                let _config = StreamConfig {
                    channels: 1,
                    sample_rate: SampleRate(rate),
                    buffer_size: cpal::BufferSize::Default,
                };

                if device.supported_output_configs().is_ok_and(|mut configs| {
                    configs.any(|c| c.min_sample_rate().0 <= rate && rate <= c.max_sample_rate().0)
                }) {
                    supported_sample_rates.push(rate);
                }
            }
        }

        supported_sample_rates.sort_unstable();

        Ok(AudioDeviceInfo {
            id: format!("core_audio_{name}"),
            name,
            is_default: false, // Will be set by caller if this is the default device
            supported_sample_rates,
            max_output_channels: max_channels,
            min_buffer_size,
            max_buffer_size,
            default_buffer_size,
        })
    }

    /// Convert our AudioStreamConfig to cpal StreamConfig
    fn to_cpal_config(config: &AudioStreamConfig) -> StreamConfig {
        StreamConfig {
            channels: config.channels as u16,
            sample_rate: SampleRate(config.sample_rate),
            buffer_size: if config.buffer_size > 0 {
                cpal::BufferSize::Fixed(config.buffer_size)
            } else {
                cpal::BufferSize::Default
            },
        }
    }
}

#[async_trait(?Send)]
impl AudioDriver for CoreAudioDriver {
    async fn enumerate_devices(&self) -> DriverResult<Vec<AudioDeviceInfo>> {
        let devices = self.host.output_devices().map_err(|e| {
            AudioDriverError::InternalError(format!("Failed to enumerate devices: {e}"))
        })?;

        let mut device_infos = Vec::new();

        for device in devices {
            match Self::device_info_from_cpal(&device) {
                Ok(info) => device_infos.push(info),
                Err(e) => {
                    // Log warning but continue with other devices
                    tracing::warn!("Failed to get info for audio device: {}", e);
                }
            }
        }

        Ok(device_infos)
    }

    async fn default_device(&self) -> DriverResult<AudioDeviceInfo> {
        let device = self.host.default_output_device().ok_or_else(|| {
            AudioDriverError::DeviceNotFound("No default output device found".to_string())
        })?;

        let mut info = Self::device_info_from_cpal(&device)?;
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
            // Find device by ID
            let mut devices = self.host.output_devices().map_err(|e| {
                AudioDriverError::InternalError(format!("Failed to enumerate devices: {e}"))
            })?;

            devices
                .find(|d| {
                    d.name()
                        .is_ok_and(|name| format!("core_audio_{name}") == id)
                })
                .ok_or_else(|| {
                    AudioDriverError::DeviceNotFound(format!("Device not found: {id}"))
                })?
        } else {
            // Use default device
            self.host.default_output_device().ok_or_else(|| {
                AudioDriverError::DeviceNotFound("No default output device found".to_string())
            })?
        };

        // Validate configuration
        let default_config = device.default_output_config().map_err(|e| {
            AudioDriverError::InternalError(format!("Failed to get default config: {e}"))
        })?;

        // Check if sample rate is supported
        let supported_configs: Vec<_> = device
            .supported_output_configs()
            .map_err(|e| {
                AudioDriverError::InternalError(format!("Failed to get supported configs: {e}"))
            })?
            .collect();

        let is_supported = supported_configs.iter().any(|c| {
            c.min_sample_rate().0 <= config.sample_rate
                && config.sample_rate <= c.max_sample_rate().0
                && c.channels() >= config.channels as u16
        });

        if !is_supported {
            return Err(AudioDriverError::UnsupportedSampleRate {
                rate: config.sample_rate,
            });
        }

        let cpal_config = Self::to_cpal_config(&config);
        let metrics = Arc::clone(&self.metrics);

        // Create the stream based on sample format
        let stream = match default_config.sample_format() {
            SampleFormat::F32 => {
                let callback_clone = Arc::clone(&callback);
                device.build_output_stream(
                    &cpal_config,
                    move |data: &mut [f32], _info: &cpal::OutputCallbackInfo| {
                        // Call our callback
                        if let Err(e) = callback_clone(data) {
                            tracing::error!("Audio callback error: {}", e);
                        }

                        // Update metrics
                        let mut m = metrics.lock();
                        m.frames_processed += data.len() as u64;
                        m.is_active = true;
                    },
                    move |err| {
                        tracing::error!("Core Audio stream error: {}", err);
                    },
                    None,
                )
            }
            SampleFormat::I16 => {
                let callback_clone = Arc::clone(&callback);
                device.build_output_stream(
                    &cpal_config,
                    move |data: &mut [i16], _info: &cpal::OutputCallbackInfo| {
                        // Convert to f32, call callback, then convert back
                        let mut f32_data: Vec<f32> = data
                            .iter()
                            .map(|&sample| sample as f32 / i16::MAX as f32)
                            .collect();

                        if let Err(e) = callback_clone(&mut f32_data) {
                            tracing::error!("Audio callback error: {}", e);
                        }

                        // Convert back to i16
                        for (out, &sample) in data.iter_mut().zip(f32_data.iter()) {
                            *out = (sample * i16::MAX as f32) as i16;
                        }

                        // Update metrics
                        let mut m = metrics.lock();
                        m.frames_processed += data.len() as u64;
                        m.is_active = true;
                    },
                    move |err| {
                        tracing::error!("Core Audio stream error: {}", err);
                    },
                    None,
                )
            }
            SampleFormat::U16 => {
                let callback_clone = Arc::clone(&callback);
                device.build_output_stream(
                    &cpal_config,
                    move |data: &mut [u16], _info: &cpal::OutputCallbackInfo| {
                        // Convert to f32, call callback, then convert back
                        let mut f32_data: Vec<f32> = data
                            .iter()
                            .map(|&sample| {
                                (sample as f32 - u16::MAX as f32 / 2.0) / (u16::MAX as f32 / 2.0)
                            })
                            .collect();

                        if let Err(e) = callback_clone(&mut f32_data) {
                            tracing::error!("Audio callback error: {}", e);
                        }

                        // Convert back to u16
                        for (out, &sample) in data.iter_mut().zip(f32_data.iter()) {
                            *out = (sample * u16::MAX as f32 / 2.0 + u16::MAX as f32 / 2.0) as u16;
                        }

                        // Update metrics
                        let mut m = metrics.lock();
                        m.frames_processed += data.len() as u64;
                        m.is_active = true;
                    },
                    move |err| {
                        tracing::error!("Core Audio stream error: {}", err);
                    },
                    None,
                )
            }
            _ => {
                return Err(AudioDriverError::InternalError(format!(
                    "Unsupported sample format: {:?}",
                    default_config.sample_format()
                )));
            }
        }
        .map_err(|e| match e {
            BuildStreamError::DeviceNotAvailable => {
                AudioDriverError::DeviceNotFound("Device not available".to_string())
            }
            BuildStreamError::InvalidArgument => {
                AudioDriverError::InternalError("Invalid stream configuration".to_string())
            }
            _ => AudioDriverError::StreamInitFailed(format!("Failed to build stream: {e}")),
        })?;

        self.device = Some(device);
        self.stream = Some(stream);
        self.callback = Some(callback);

        Ok(())
    }

    async fn start_stream(&mut self) -> DriverResult<()> {
        let stream = self
            .stream
            .as_ref()
            .ok_or(AudioDriverError::StreamNotInitialized)?;

        stream.play().map_err(|e| {
            AudioDriverError::StreamInitFailed(format!("Failed to start stream: {e}"))
        })?;

        self.is_running = true;

        // Update metrics
        let mut metrics = self.metrics.lock();
        metrics.is_active = true;

        tracing::info!("Core Audio stream started successfully");

        Ok(())
    }

    async fn stop_stream(&mut self) -> DriverResult<()> {
        if let Some(stream) = &self.stream {
            stream.pause().map_err(|e| {
                AudioDriverError::InternalError(format!("Failed to stop stream: {e}"))
            })?;
        }

        self.is_running = false;

        // Update metrics
        let mut metrics = self.metrics.lock();
        metrics.is_active = false;

        tracing::info!("Core Audio stream stopped");

        Ok(())
    }

    fn is_stream_running(&self) -> bool {
        self.is_running
    }

    fn get_metrics(&self) -> AudioStreamMetrics {
        self.metrics.lock().clone()
    }

    fn driver_name(&self) -> &'static str {
        "Core Audio"
    }

    fn is_available() -> bool {
        // Core Audio is always available on macOS
        cfg!(target_os = "macos")
    }
}

impl Drop for CoreAudioDriver {
    fn drop(&mut self) {
        if self.is_running {
            // Try to stop the stream gracefully
            if let Some(stream) = &self.stream {
                let _ = stream.pause();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_core_audio_driver_creation() {
        if !CoreAudioDriver::is_available() {
            return;
        }

        let driver = CoreAudioDriver::new();
        assert!(driver.is_ok());
    }

    #[tokio::test]
    async fn test_enumerate_devices() {
        if !CoreAudioDriver::is_available() {
            return;
        }

        let driver = CoreAudioDriver::new().unwrap();
        let devices = driver.enumerate_devices().await;

        match devices {
            Ok(device_list) => {
                println!("Found {} audio devices", device_list.len());
                for device in device_list {
                    println!("Device: {} ({})", device.name, device.id);
                    println!("  Channels: {}", device.max_output_channels);
                    println!("  Sample rates: {:?}", device.supported_sample_rates);
                }
            }
            Err(e) => {
                println!("Failed to enumerate devices: {e}");
            }
        }
    }

    #[tokio::test]
    async fn test_default_device() {
        if !CoreAudioDriver::is_available() {
            return;
        }

        let driver = CoreAudioDriver::new().unwrap();
        let default_device = driver.default_device().await;

        match default_device {
            Ok(device) => {
                println!("Default device: {} ({})", device.name, device.id);
                assert!(device.is_default);
            }
            Err(e) => {
                println!("Failed to get default device: {e}");
            }
        }
    }

    #[tokio::test]
    async fn test_stream_initialization() {
        if !CoreAudioDriver::is_available() {
            return;
        }

        let mut driver = CoreAudioDriver::new().unwrap();
        let config = AudioStreamConfig::default();

        // Create a simple sine wave callback
        let callback: AudioCallback = Arc::new(|output: &mut [f32]| {
            // Generate a 440 Hz sine wave
            static mut PHASE: f32 = 0.0;
            let freq = 440.0;
            let sample_rate = 22050.0;

            unsafe {
                for sample in output.iter_mut() {
                    *sample = (PHASE * 2.0 * std::f32::consts::PI).sin() * 0.1;
                    PHASE += freq / sample_rate;
                    if PHASE >= 1.0 {
                        PHASE -= 1.0;
                    }
                }
            }

            Ok(())
        });

        let result = driver.initialize_stream(None, config, callback).await;

        match result {
            Ok(()) => {
                println!("Stream initialized successfully");

                // Try to start the stream
                let start_result = driver.start_stream().await;
                assert!(start_result.is_ok());

                // Let it play for a short time
                sleep(Duration::from_millis(100)).await;

                // Stop the stream
                let stop_result = driver.stop_stream().await;
                assert!(stop_result.is_ok());
            }
            Err(e) => {
                println!("Failed to initialize stream: {e}");
            }
        }
    }
}
