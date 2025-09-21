//! Real-time audio drivers for voirs-vocoder
//!
//! This module provides real-time audio output capabilities through various
//! audio drivers and APIs, enabling low-latency audio playback for real-time
//! vocoding applications.

use crate::{AudioBuffer, VocoderError};
use async_trait::async_trait;
use std::sync::Arc;

pub mod core_audio;

#[cfg(target_os = "windows")]
pub mod asio;

#[cfg(target_os = "linux")]
pub mod linux;

/// Result type for audio driver operations
pub type DriverResult<T> = Result<T, AudioDriverError>;

/// Audio driver errors
#[derive(thiserror::Error, Debug)]
pub enum AudioDriverError {
    #[error("Audio device not found: {0}")]
    DeviceNotFound(String),

    #[error("Audio stream initialization failed: {0}")]
    StreamInitFailed(String),

    #[error("Audio buffer underrun occurred")]
    BufferUnderrun,

    #[error("Audio buffer overrun occurred")]
    BufferOverrun,

    #[error("Unsupported sample rate: {rate} Hz")]
    UnsupportedSampleRate { rate: u32 },

    #[error("Unsupported channel configuration: {channels} channels")]
    UnsupportedChannelCount { channels: u32 },

    #[error("Audio stream is not initialized")]
    StreamNotInitialized,

    #[error("Audio device disconnected")]
    DeviceDisconnected,

    #[error("Real-time constraint violation: {message}")]
    RealTimeViolation { message: String },

    #[error("Driver internal error: {0}")]
    InternalError(String),
}

impl From<AudioDriverError> for VocoderError {
    fn from(err: AudioDriverError) -> Self {
        VocoderError::RuntimeError(format!("Audio driver error: {err}"))
    }
}

/// Audio device information
#[derive(Debug, Clone)]
pub struct AudioDeviceInfo {
    /// Device identifier
    pub id: String,
    /// Human-readable device name
    pub name: String,
    /// Whether this is the default output device
    pub is_default: bool,
    /// Supported sample rates
    pub supported_sample_rates: Vec<u32>,
    /// Maximum number of output channels
    pub max_output_channels: u32,
    /// Minimum buffer size in frames
    pub min_buffer_size: u32,
    /// Maximum buffer size in frames
    pub max_buffer_size: u32,
    /// Default buffer size in frames
    pub default_buffer_size: u32,
}

/// Audio stream configuration
#[derive(Debug, Clone)]
pub struct AudioStreamConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u32,
    /// Buffer size in frames
    pub buffer_size: u32,
    /// Target latency in milliseconds
    pub target_latency_ms: f32,
}

impl Default for AudioStreamConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            channels: 1,
            buffer_size: 256,
            target_latency_ms: 10.0,
        }
    }
}

/// Audio stream metrics
#[derive(Debug, Clone, Default)]
pub struct AudioStreamMetrics {
    /// Number of audio frames processed
    pub frames_processed: u64,
    /// Number of buffer underruns
    pub underruns: u32,
    /// Number of buffer overruns
    pub overruns: u32,
    /// Current latency in milliseconds
    pub current_latency_ms: f32,
    /// Average CPU load percentage
    pub cpu_load_percent: f32,
    /// Whether the stream is currently active
    pub is_active: bool,
}

/// Callback function type for real-time audio processing
pub type AudioCallback = Arc<dyn Fn(&mut [f32]) -> DriverResult<()> + Send + Sync>;

/// Trait for real-time audio drivers
#[async_trait(?Send)]
pub trait AudioDriver {
    /// Get available audio output devices
    async fn enumerate_devices(&self) -> DriverResult<Vec<AudioDeviceInfo>>;

    /// Get the default output device
    async fn default_device(&self) -> DriverResult<AudioDeviceInfo>;

    /// Initialize an audio stream with the specified configuration
    async fn initialize_stream(
        &mut self,
        device_id: Option<&str>,
        config: AudioStreamConfig,
        callback: AudioCallback,
    ) -> DriverResult<()>;

    /// Start the audio stream
    async fn start_stream(&mut self) -> DriverResult<()>;

    /// Stop the audio stream
    async fn stop_stream(&mut self) -> DriverResult<()>;

    /// Check if the stream is currently running
    fn is_stream_running(&self) -> bool;

    /// Get current stream metrics
    fn get_metrics(&self) -> AudioStreamMetrics;

    /// Get the driver name
    fn driver_name(&self) -> &'static str;

    /// Check if the driver is available on the current platform
    fn is_available() -> bool
    where
        Self: Sized;
}

/// Factory for creating audio drivers
pub struct AudioDriverFactory;

impl AudioDriverFactory {
    /// Create the best available audio driver for the current platform
    pub fn create_default() -> DriverResult<Box<dyn AudioDriver>> {
        #[cfg(target_os = "macos")]
        {
            if core_audio::CoreAudioDriver::is_available() {
                return Ok(Box::new(core_audio::CoreAudioDriver::new()?));
            }
        }

        #[cfg(target_os = "windows")]
        {
            if asio::AsioDriver::is_available() {
                return Ok(Box::new(asio::AsioDriver::new()?));
            }
            return Err(AudioDriverError::InternalError(
                "ASIO audio driver not available on this system".to_string(),
            ));
        }

        #[cfg(target_os = "linux")]
        {
            return linux::create_linux_driver();
        }

        Err(AudioDriverError::InternalError(
            "No audio driver available for this platform".to_string(),
        ))
    }

    /// List all available drivers on the current platform
    pub fn available_drivers() -> Vec<&'static str> {
        let mut drivers = Vec::new();

        #[cfg(target_os = "macos")]
        {
            if core_audio::CoreAudioDriver::is_available() {
                drivers.push("Core Audio");
            }
        }

        #[cfg(target_os = "windows")]
        {
            if asio::AsioDriver::is_available() {
                drivers.push("ASIO");
            }
        }

        #[cfg(target_os = "linux")]
        {
            if linux::AlsaDriver::is_available() {
                drivers.push("ALSA");
            }
            if linux::PulseAudioDriver::is_available() {
                drivers.push("PulseAudio");
            }
        }

        drivers
    }
}

/// Real-time audio output manager
pub struct RealTimeAudioOutput {
    driver: Box<dyn AudioDriver>,
    config: AudioStreamConfig,
    buffer_queue: Arc<parking_lot::Mutex<std::collections::VecDeque<AudioBuffer>>>,
    metrics: Arc<parking_lot::Mutex<AudioStreamMetrics>>,
}

impl RealTimeAudioOutput {
    /// Create a new real-time audio output manager
    pub fn new(config: AudioStreamConfig) -> DriverResult<Self> {
        let driver = AudioDriverFactory::create_default()?;
        let buffer_queue = Arc::new(parking_lot::Mutex::new(std::collections::VecDeque::new()));
        let metrics = Arc::new(parking_lot::Mutex::new(AudioStreamMetrics::default()));

        Ok(Self {
            driver,
            config,
            buffer_queue,
            metrics,
        })
    }

    /// Initialize and start the audio stream
    pub async fn start(&mut self) -> DriverResult<()> {
        let metrics = Arc::clone(&self.metrics);
        let buffer_queue = Arc::clone(&self.buffer_queue);

        // Create the audio callback
        let callback: AudioCallback = Arc::new(move |output: &mut [f32]| {
            // Try to get the next audio buffer from the queue
            let mut queue = buffer_queue.lock();

            if let Some(audio_buffer) = queue.pop_front() {
                let samples = audio_buffer.samples();
                let frames_needed = output.len();

                if samples.len() >= frames_needed {
                    // Copy audio data to output
                    output.copy_from_slice(&samples[..frames_needed]);

                    // If there are remaining samples, put them back in the queue
                    if samples.len() > frames_needed {
                        let remaining_samples = samples[frames_needed..].to_vec();
                        let remaining_buffer = AudioBuffer::new(
                            remaining_samples,
                            audio_buffer.sample_rate(),
                            audio_buffer.channels(),
                        );
                        queue.push_front(remaining_buffer);
                    }
                } else {
                    // Zero-pad if not enough samples
                    output[..samples.len()].copy_from_slice(samples);
                    output[samples.len()..].fill(0.0);
                }

                drop(queue); // Release lock early

                // Update metrics
                let mut m = metrics.lock();
                m.frames_processed += frames_needed as u64;
                m.is_active = true;

                Ok(())
            } else {
                // No audio data available - output silence
                output.fill(0.0);
                drop(queue); // Release lock early

                let mut m = metrics.lock();
                m.underruns += 1;

                Ok(())
            }
        });

        // Initialize and start the audio stream
        self.driver
            .initialize_stream(None, self.config.clone(), callback)
            .await?;
        self.driver.start_stream().await?;

        Ok(())
    }

    /// Stop the audio stream
    pub async fn stop(&mut self) -> DriverResult<()> {
        self.driver.stop_stream().await
    }

    /// Get current audio metrics
    pub fn metrics(&self) -> AudioStreamMetrics {
        self.metrics.lock().clone()
    }

    /// Queue an audio buffer for playback
    pub fn queue_audio(&self, buffer: AudioBuffer) -> DriverResult<()> {
        let mut queue = self.buffer_queue.lock();
        queue.push_back(buffer);
        Ok(())
    }
}
