//! Cross-platform audio playback implementation.

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    ChannelCount, Device, Host, SampleFormat, SampleRate, Stream, StreamConfig, StreamError,
};
use hound::{WavReader, WavSpec};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use voirs::{Result, VoirsError};

/// Audio data representation
#[derive(Debug, Clone)]
pub struct AudioData {
    /// Raw audio samples as i16
    pub samples: Vec<i16>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
}

impl AudioData {
    /// Get duration in seconds
    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / (self.sample_rate as f32 * self.channels as f32)
    }

    /// Convert to f32 samples
    pub fn to_f32_samples(&self) -> Vec<f32> {
        self.samples
            .iter()
            .map(|&s| s as f32 / i16::MAX as f32)
            .collect()
    }
}

/// Audio playback configuration
#[derive(Debug, Clone)]
pub struct PlaybackConfig {
    /// Target sample rate for playback
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u16,
    /// Audio buffer size in frames
    pub buffer_size: u32,
    /// Target audio device (None for default)
    pub device_name: Option<String>,
    /// Master volume (0.0 to 1.0)
    pub volume: f32,
}

impl Default for PlaybackConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            channels: 1,
            buffer_size: 1024,
            device_name: None,
            volume: 1.0,
        }
    }
}

/// Audio device information
#[derive(Debug, Clone)]
pub struct AudioDevice {
    pub name: String,
    pub is_default: bool,
    pub max_output_channels: u16,
    pub sample_rates: Vec<u32>,
    pub supported_formats: Vec<SampleFormat>,
}

impl AudioDevice {
    /// Check if device supports the given configuration
    pub fn supports_config(&self, config: &PlaybackConfig) -> bool {
        self.max_output_channels >= config.channels
            && self.sample_rates.contains(&config.sample_rate)
    }
}

/// Audio playback queue item
#[derive(Debug, Clone)]
pub struct QueueItem {
    pub id: String,
    pub audio_data: AudioData,
    pub metadata: std::collections::HashMap<String, String>,
}

/// Audio playback queue
#[derive(Debug)]
pub struct PlaybackQueue {
    items: Arc<Mutex<VecDeque<QueueItem>>>,
    current_playing: Arc<Mutex<Option<String>>>,
}

impl PlaybackQueue {
    /// Create a new playback queue
    pub fn new() -> Self {
        Self {
            items: Arc::new(Mutex::new(VecDeque::new())),
            current_playing: Arc::new(Mutex::new(None)),
        }
    }

    /// Add audio to the queue
    pub fn enqueue(&self, item: QueueItem) -> Result<()> {
        let mut items = self
            .items
            .lock()
            .map_err(|_| VoirsError::device_error("audio_queue", "Failed to lock queue mutex"))?;
        items.push_back(item);
        Ok(())
    }

    /// Get the next item from the queue
    pub fn dequeue(&self) -> Result<Option<QueueItem>> {
        let mut items = self
            .items
            .lock()
            .map_err(|_| VoirsError::device_error("audio_queue", "Failed to lock queue mutex"))?;
        Ok(items.pop_front())
    }

    /// Get queue length
    pub fn len(&self) -> Result<usize> {
        let items = self
            .items
            .lock()
            .map_err(|_| VoirsError::device_error("audio_queue", "Failed to lock queue mutex"))?;
        Ok(items.len())
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Clear the queue
    pub fn clear(&self) -> Result<()> {
        let mut items = self
            .items
            .lock()
            .map_err(|_| VoirsError::device_error("audio_queue", "Failed to lock queue mutex"))?;
        items.clear();
        Ok(())
    }

    /// Set currently playing item
    pub fn set_current_playing(&self, id: Option<String>) -> Result<()> {
        let mut current = self.current_playing.lock().map_err(|_| {
            VoirsError::device_error("audio_queue", "Failed to lock current_playing mutex")
        })?;
        *current = id;
        Ok(())
    }

    /// Get currently playing item ID
    pub fn get_current_playing(&self) -> Result<Option<String>> {
        let current = self.current_playing.lock().map_err(|_| {
            VoirsError::device_error("audio_queue", "Failed to lock current_playing mutex")
        })?;
        Ok(current.clone())
    }
}

impl Default for PlaybackQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Cross-platform audio player
pub struct AudioPlayer {
    config: PlaybackConfig,
    device: Device,
    host: Host,
    stream: Option<Stream>,
    queue: PlaybackQueue,
    is_playing: Arc<Mutex<bool>>,
}

impl AudioPlayer {
    /// Create a new audio player with the given configuration
    pub fn new(config: PlaybackConfig) -> Result<Self> {
        let host = cpal::default_host();
        let device = if let Some(device_name) = &config.device_name {
            Self::find_device_by_name(&host, device_name)?.ok_or_else(|| {
                VoirsError::device_error(
                    "audio_device",
                    format!("Audio device '{}' not found", device_name),
                )
            })?
        } else {
            host.default_output_device().ok_or_else(|| {
                VoirsError::device_error("audio_device", "No default audio output device found")
            })?
        };

        Ok(Self {
            config,
            device,
            host,
            stream: None,
            queue: PlaybackQueue::new(),
            is_playing: Arc::new(Mutex::new(false)),
        })
    }

    /// Get available audio output devices
    pub fn get_output_devices() -> Result<Vec<AudioDevice>> {
        let host = cpal::default_host();
        let mut devices = Vec::new();

        let default_device = host.default_output_device();
        let default_device_name = default_device.as_ref().and_then(|d| d.name().ok());

        for device in host.output_devices().map_err(|e| {
            VoirsError::device_error(
                "audio_device",
                format!("Failed to enumerate devices: {}", e),
            )
        })? {
            if let Ok(name) = device.name() {
                let is_default = default_device_name
                    .as_ref()
                    .map(|default| default == &name)
                    .unwrap_or(false);

                // Get device capabilities
                let supported_configs = device.supported_output_configs().map_err(|e| {
                    VoirsError::device_error(
                        "audio_device",
                        format!("Failed to get device configs: {}", e),
                    )
                })?;

                let mut sample_rates = Vec::new();
                let mut supported_formats = Vec::new();
                let mut max_channels = 0;

                for config in supported_configs {
                    max_channels = max_channels.max(config.channels());
                    sample_rates.push(config.min_sample_rate().0);
                    sample_rates.push(config.max_sample_rate().0);
                    supported_formats.push(config.sample_format());
                }

                // Remove duplicates and sort
                sample_rates.sort_unstable();
                sample_rates.dedup();
                supported_formats.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
                supported_formats.dedup();

                devices.push(AudioDevice {
                    name,
                    is_default,
                    max_output_channels: max_channels,
                    sample_rates,
                    supported_formats,
                });
            }
        }

        Ok(devices)
    }

    /// Find device by name
    fn find_device_by_name(host: &Host, device_name: &str) -> Result<Option<Device>> {
        for device in host.output_devices().map_err(|e| {
            VoirsError::device_error(
                "audio_device",
                format!("Failed to enumerate devices: {}", e),
            )
        })? {
            if let Ok(name) = device.name() {
                if name == device_name {
                    return Ok(Some(device));
                }
            }
        }
        Ok(None)
    }

    /// Play audio data immediately
    pub async fn play(&mut self, audio_data: &AudioData) -> Result<()> {
        let item = QueueItem {
            id: format!("direct_{}", chrono::Utc::now().timestamp_millis()),
            audio_data: audio_data.clone(),
            metadata: std::collections::HashMap::new(),
        };

        self.queue.enqueue(item)?;
        self.start_playback().await
    }

    /// Play audio from file
    pub async fn play_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let audio_data = self.load_audio_file(path.as_ref())?;
        self.play(&audio_data).await
    }

    /// Add audio to playback queue
    pub fn enqueue(&self, audio_data: AudioData, id: Option<String>) -> Result<()> {
        let item = QueueItem {
            id: id.unwrap_or_else(|| format!("queued_{}", chrono::Utc::now().timestamp_millis())),
            audio_data,
            metadata: std::collections::HashMap::new(),
        };

        self.queue.enqueue(item)
    }

    /// Start playback from queue
    pub async fn start_playback(&mut self) -> Result<()> {
        if self.is_playing()? {
            return Ok(());
        }

        self.set_playing(true)?;

        let stream_config = StreamConfig {
            channels: self.config.channels as ChannelCount,
            sample_rate: SampleRate(self.config.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(self.config.buffer_size),
        };

        let queue = self.queue.items.clone();
        let volume = self.config.volume;
        let is_playing = self.is_playing.clone();
        let current_playing = self.queue.current_playing.clone();

        let mut current_audio: Option<Vec<f32>> = None;
        let mut audio_position = 0;

        let stream = self
            .device
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    // Fill buffer with audio data
                    for sample in data.iter_mut() {
                        *sample = 0.0;
                    }

                    // Check if we need new audio data
                    if current_audio.is_none()
                        || audio_position >= current_audio.as_ref().unwrap().len()
                    {
                        // Try to get next item from queue
                        if let Ok(mut queue_guard) = queue.lock() {
                            if let Some(item) = queue_guard.pop_front() {
                                // Set currently playing
                                if let Ok(mut current_guard) = current_playing.lock() {
                                    *current_guard = Some(item.id.clone());
                                }

                                // Convert audio data to f32 samples
                                current_audio = Some(
                                    item.audio_data
                                        .samples
                                        .iter()
                                        .map(|&s| s as f32 / i16::MAX as f32)
                                        .collect(),
                                );
                                audio_position = 0;
                            } else {
                                // No more audio, stop playback
                                if let Ok(mut playing_guard) = is_playing.lock() {
                                    *playing_guard = false;
                                }
                                if let Ok(mut current_guard) = current_playing.lock() {
                                    *current_guard = None;
                                }
                                return;
                            }
                        }
                    }

                    // Copy audio data to output buffer
                    if let Some(ref audio) = current_audio {
                        let samples_to_copy = (data.len()).min(audio.len() - audio_position);

                        for i in 0..samples_to_copy {
                            data[i] = audio[audio_position + i] * volume;
                        }

                        audio_position += samples_to_copy;
                    }
                },
                move |err| {
                    tracing::error!("Audio stream error: {}", err);
                },
                None, // No timeout
            )
            .map_err(|e| {
                VoirsError::device_error(
                    "audio_device",
                    format!("Failed to build output stream: {}", e),
                )
            })?;

        stream.play().map_err(|e| {
            VoirsError::device_error("audio_device", format!("Failed to start stream: {}", e))
        })?;

        self.stream = Some(stream);
        Ok(())
    }

    /// Stop playback
    pub fn stop(&mut self) -> Result<()> {
        self.set_playing(false)?;
        self.queue.set_current_playing(None)?;

        if let Some(stream) = self.stream.take() {
            stream.pause().map_err(|e| {
                VoirsError::device_error("audio_device", format!("Failed to stop stream: {}", e))
            })?;
        }

        Ok(())
    }

    /// Pause playback
    pub fn pause(&self) -> Result<()> {
        if let Some(stream) = &self.stream {
            stream.pause().map_err(|e| {
                VoirsError::device_error("audio_device", format!("Failed to pause stream: {}", e))
            })?;
        }
        self.set_playing(false)
    }

    /// Resume playback
    pub fn resume(&self) -> Result<()> {
        if let Some(stream) = &self.stream {
            stream.play().map_err(|e| {
                VoirsError::device_error("audio_device", format!("Failed to resume stream: {}", e))
            })?;
        }
        self.set_playing(true)
    }

    /// Check if currently playing
    pub fn is_playing(&self) -> Result<bool> {
        let playing = self.is_playing.lock().map_err(|_| {
            VoirsError::device_error("audio_player", "Failed to lock is_playing mutex")
        })?;
        Ok(*playing)
    }

    /// Set playing state
    fn set_playing(&self, playing: bool) -> Result<()> {
        let mut state = self.is_playing.lock().map_err(|_| {
            VoirsError::device_error("audio_player", "Failed to lock is_playing mutex")
        })?;
        *state = playing;
        Ok(())
    }

    /// Set volume (0.0 to 1.0)
    pub fn set_volume(&mut self, volume: f32) -> Result<()> {
        if volume < 0.0 || volume > 1.0 {
            return Err(VoirsError::config_error(
                "Volume must be between 0.0 and 1.0",
            ));
        }
        self.config.volume = volume;
        Ok(())
    }

    /// Get current volume
    pub fn get_volume(&self) -> f32 {
        self.config.volume
    }

    /// Get queue reference
    pub fn queue(&self) -> &PlaybackQueue {
        &self.queue
    }

    /// Load audio file
    fn load_audio_file(&self, path: &Path) -> Result<AudioData> {
        let mut reader = WavReader::open(path).map_err(|e| {
            VoirsError::device_error("audio_device", format!("Failed to open audio file: {}", e))
        })?;

        let spec = reader.spec();
        let samples: std::result::Result<Vec<i16>, hound::Error> =
            reader.samples::<i16>().collect();
        let samples = samples.map_err(|e| {
            VoirsError::device_error(
                "audio_device",
                format!("Failed to read audio samples: {}", e),
            )
        })?;

        Ok(AudioData {
            samples,
            sample_rate: spec.sample_rate,
            channels: spec.channels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::AudioData;
    use super::*;

    #[test]
    fn test_playback_config_default() {
        let config = PlaybackConfig::default();
        assert_eq!(config.sample_rate, 22050);
        assert_eq!(config.channels, 1);
        assert_eq!(config.volume, 1.0);
    }

    #[test]
    fn test_playback_queue() {
        let queue = PlaybackQueue::new();
        assert!(queue.is_empty().unwrap());

        let audio_data = AudioData {
            samples: vec![0, 1, 2, 3],
            sample_rate: 22050,
            channels: 1,
        };

        let item = QueueItem {
            id: "test".to_string(),
            audio_data,
            metadata: std::collections::HashMap::new(),
        };

        queue.enqueue(item).unwrap();
        assert_eq!(queue.len().unwrap(), 1);
        assert!(!queue.is_empty().unwrap());

        let dequeued = queue.dequeue().unwrap().unwrap();
        assert_eq!(dequeued.id, "test");
        assert!(queue.is_empty().unwrap());
    }

    #[tokio::test]
    async fn test_get_output_devices() {
        // This test might fail in CI environments without audio devices
        match AudioPlayer::get_output_devices() {
            Ok(devices) => {
                // If we have devices, at least one should be marked as default
                if !devices.is_empty() {
                    assert!(devices.iter().any(|d| d.is_default));
                }
            }
            Err(_) => {
                // It's okay if no audio devices are available in test environment
            }
        }
    }

    #[tokio::test]
    async fn test_audio_player_creation() {
        let config = PlaybackConfig::default();

        // This test might fail in CI environments without audio devices
        match AudioPlayer::new(config) {
            Ok(player) => {
                assert_eq!(player.get_volume(), 1.0);
                assert!(!player.is_playing().unwrap());
            }
            Err(_) => {
                // It's okay if no audio devices are available in test environment
            }
        }
    }

    #[test]
    fn test_audio_device_supports_config() {
        let device = AudioDevice {
            name: "Test Device".to_string(),
            is_default: true,
            max_output_channels: 2,
            sample_rates: vec![22050, 44100, 48000],
            supported_formats: vec![SampleFormat::F32],
        };

        let config = PlaybackConfig {
            sample_rate: 22050,
            channels: 1,
            buffer_size: 1024,
            device_name: None,
            volume: 1.0,
        };

        assert!(device.supports_config(&config));

        let unsupported_config = PlaybackConfig {
            sample_rate: 96000, // Not supported
            channels: 1,
            buffer_size: 1024,
            device_name: None,
            volume: 1.0,
        };

        assert!(!device.supports_config(&unsupported_config));
    }
}
