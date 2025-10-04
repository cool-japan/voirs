//! Real-time audio streaming implementation.

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    ChannelCount, Device, Host, SampleRate, Stream,
};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use voirs_sdk::{Result, VoirsError};

use super::AudioData;

/// Real-time streaming configuration
#[derive(Debug, Clone)]
pub struct RealTimeStreamConfig {
    /// Sample rate for streaming
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u16,
    /// Stream buffer size
    pub buffer_size: u32,
    /// Target latency in milliseconds
    pub target_latency_ms: u32,
    /// Device name (None for default)
    pub device_name: Option<String>,
}

impl Default for RealTimeStreamConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            channels: 1,
            buffer_size: 512, // Smaller buffer for lower latency
            target_latency_ms: 50,
            device_name: None,
        }
    }
}

/// Audio buffer configuration
#[derive(Debug, Clone)]
pub struct BufferConfig {
    /// Number of buffers in the ring buffer
    pub buffer_count: usize,
    /// Size of each buffer in frames
    pub buffer_size: usize,
    /// Underrun threshold (number of empty buffers before warning)
    pub underrun_threshold: usize,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            buffer_count: 8,
            buffer_size: 512,
            underrun_threshold: 2,
        }
    }
}

/// Audio buffer for real-time streaming
#[derive(Debug)]
struct AudioBuffer {
    data: Vec<f32>,
    is_ready: bool,
    timestamp: std::time::Instant,
}

impl AudioBuffer {
    fn new(size: usize) -> Self {
        Self {
            data: vec![0.0; size],
            is_ready: false,
            timestamp: std::time::Instant::now(),
        }
    }

    fn write_samples(&mut self, samples: &[f32]) {
        let copy_len = samples.len().min(self.data.len());
        self.data[..copy_len].copy_from_slice(&samples[..copy_len]);
        self.is_ready = true;
        self.timestamp = std::time::Instant::now();
    }

    fn read_samples(&mut self, output: &mut [f32]) -> usize {
        if !self.is_ready {
            // Fill with silence
            for sample in output.iter_mut() {
                *sample = 0.0;
            }
            return 0;
        }

        let copy_len = output.len().min(self.data.len());
        output[..copy_len].copy_from_slice(&self.data[..copy_len]);

        // Mark as consumed
        self.is_ready = false;

        copy_len
    }
}

/// Real-time audio stream statistics
#[derive(Debug, Clone)]
pub struct StreamStats {
    pub buffers_played: u64,
    pub buffers_dropped: u64,
    pub underruns: u64,
    pub average_latency_ms: f32,
    pub current_buffer_fill: f32,
}

impl Default for StreamStats {
    fn default() -> Self {
        Self {
            buffers_played: 0,
            buffers_dropped: 0,
            underruns: 0,
            average_latency_ms: 0.0,
            current_buffer_fill: 0.0,
        }
    }
}

/// Real-time audio streaming interface
pub struct RealTimeAudioStream {
    config: RealTimeStreamConfig,
    buffer_config: BufferConfig,
    device: Device,
    stream: Option<Stream>,
    buffers: Arc<Mutex<Vec<AudioBuffer>>>,
    write_index: Arc<Mutex<usize>>,
    read_index: Arc<Mutex<usize>>,
    stats: Arc<Mutex<StreamStats>>,
    is_active: Arc<Mutex<bool>>,
}

impl RealTimeAudioStream {
    /// Create a new real-time audio stream
    pub fn new(stream_config: RealTimeStreamConfig, buffer_config: BufferConfig) -> Result<Self> {
        let host = cpal::default_host();
        let device = if let Some(device_name) = &stream_config.device_name {
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

        // Initialize ring buffer
        let mut buffers = Vec::with_capacity(buffer_config.buffer_count);
        for _ in 0..buffer_config.buffer_count {
            buffers.push(AudioBuffer::new(buffer_config.buffer_size));
        }

        Ok(Self {
            config: stream_config,
            buffer_config,
            device,
            stream: None,
            buffers: Arc::new(Mutex::new(buffers)),
            write_index: Arc::new(Mutex::new(0)),
            read_index: Arc::new(Mutex::new(0)),
            stats: Arc::new(Mutex::new(StreamStats::default())),
            is_active: Arc::new(Mutex::new(false)),
        })
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

    /// Start the real-time audio stream
    pub async fn start(&mut self) -> Result<()> {
        if self.is_active()? {
            return Ok(());
        }

        let stream_config = cpal::StreamConfig {
            channels: self.config.channels as ChannelCount,
            sample_rate: SampleRate(self.config.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(self.config.buffer_size),
        };

        let buffers = self.buffers.clone();
        let read_index = self.read_index.clone();
        let stats = self.stats.clone();
        let is_active = self.is_active.clone();

        let stream = self
            .device
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    // Check if stream is active
                    let active = if let Ok(guard) = is_active.lock() {
                        *guard
                    } else {
                        false
                    };

                    if !active {
                        // Fill with silence
                        for sample in data.iter_mut() {
                            *sample = 0.0;
                        }
                        return;
                    }

                    // Get current read buffer
                    let mut read_idx = if let Ok(guard) = read_index.lock() {
                        *guard
                    } else {
                        0
                    };

                    let (samples_read, buffer_count) = if let Ok(mut buffers_guard) = buffers.lock()
                    {
                        let count = buffers_guard.len();
                        let read = if read_idx < count {
                            buffers_guard[read_idx].read_samples(data)
                        } else {
                            0
                        };
                        (read, count)
                    } else {
                        (0, 1)
                    };

                    if samples_read > 0 {
                        // Update read index
                        if let Ok(mut guard) = read_index.lock() {
                            *guard = (read_idx + 1) % buffer_count;
                        }

                        // Update statistics
                        if let Ok(mut stats_guard) = stats.lock() {
                            stats_guard.buffers_played += 1;
                        }
                    } else {
                        // Underrun - fill with silence
                        for sample in data.iter_mut() {
                            *sample = 0.0;
                        }

                        // Update statistics
                        if let Ok(mut stats_guard) = stats.lock() {
                            stats_guard.underruns += 1;
                        }
                    }
                },
                move |err| {
                    tracing::error!("Real-time audio stream error: {}", err);
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
        self.set_active(true)?;

        Ok(())
    }

    /// Stop the real-time audio stream
    pub fn stop(&mut self) -> Result<()> {
        self.set_active(false)?;

        if let Some(stream) = self.stream.take() {
            stream.pause().map_err(|e| {
                VoirsError::device_error("audio_device", format!("Failed to stop stream: {}", e))
            })?;
        }

        Ok(())
    }

    /// Write audio data to the stream buffer
    pub fn write_audio(&self, audio_data: &AudioData) -> Result<()> {
        let samples_f32: Vec<f32> = audio_data
            .samples
            .iter()
            .map(|&s| s as f32 / i16::MAX as f32)
            .collect();

        self.write_samples(&samples_f32)
    }

    /// Write raw audio samples to the stream buffer
    pub fn write_samples(&self, samples: &[f32]) -> Result<()> {
        let mut write_idx = self.write_index.lock().map_err(|_| {
            VoirsError::device_error("audio_stream", "Failed to lock write_index mutex")
        })?;

        let mut buffers = self.buffers.lock().map_err(|_| {
            VoirsError::device_error("audio_stream", "Failed to lock buffers mutex")
        })?;

        if *write_idx < buffers.len() {
            buffers[*write_idx].write_samples(samples);
            *write_idx = (*write_idx + 1) % buffers.len();
        }

        Ok(())
    }

    /// Check if the stream is active
    pub fn is_active(&self) -> Result<bool> {
        let active = self.is_active.lock().map_err(|_| {
            VoirsError::device_error("audio_stream", "Failed to lock is_active mutex")
        })?;
        Ok(*active)
    }

    /// Set active state
    fn set_active(&self, active: bool) -> Result<()> {
        let mut state = self.is_active.lock().map_err(|_| {
            VoirsError::device_error("audio_stream", "Failed to lock is_active mutex")
        })?;
        *state = active;
        Ok(())
    }

    /// Get buffer fill level (0.0 to 1.0)
    pub fn get_buffer_fill_level(&self) -> Result<f32> {
        let write_idx = self.write_index.lock().map_err(|_| {
            VoirsError::device_error("audio_stream", "Failed to lock write_index mutex")
        })?;
        let read_idx = self.read_index.lock().map_err(|_| {
            VoirsError::device_error("audio_stream", "Failed to lock read_index mutex")
        })?;

        let buffers = self.buffers.lock().map_err(|_| {
            VoirsError::device_error("audio_stream", "Failed to lock buffers mutex")
        })?;

        let ready_buffers = buffers.iter().filter(|b| b.is_ready).count();

        Ok(ready_buffers as f32 / self.buffer_config.buffer_count as f32)
    }

    /// Get stream statistics
    pub fn get_stats(&self) -> Result<StreamStats> {
        let stats = self
            .stats
            .lock()
            .map_err(|_| VoirsError::device_error("audio_stream", "Failed to lock stats mutex"))?;

        let mut stats_copy = stats.clone();
        stats_copy.current_buffer_fill = self.get_buffer_fill_level()?;

        Ok(stats_copy)
    }

    /// Reset statistics
    pub fn reset_stats(&self) -> Result<()> {
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| VoirsError::device_error("audio_stream", "Failed to lock stats mutex"))?;
        *stats = StreamStats::default();
        Ok(())
    }

    /// Get estimated latency in milliseconds
    pub fn get_estimated_latency_ms(&self) -> Result<f32> {
        let buffer_fill = self.get_buffer_fill_level()?;
        let buffer_duration_ms =
            (self.buffer_config.buffer_size as f32 / self.config.sample_rate as f32) * 1000.0;
        let total_buffer_latency =
            buffer_fill * buffer_duration_ms * self.buffer_config.buffer_count as f32;

        Ok(total_buffer_latency)
    }

    /// Check if there's enough buffer space for low-latency streaming
    pub fn has_sufficient_buffer_space(&self) -> Result<bool> {
        let buffer_fill = self.get_buffer_fill_level()?;
        Ok(buffer_fill < 0.8) // Keep 20% buffer space
    }

    /// Wait for buffer space to become available
    pub async fn wait_for_buffer_space(&self, timeout: Duration) -> Result<bool> {
        let start_time = std::time::Instant::now();

        while start_time.elapsed() < timeout {
            if self.has_sufficient_buffer_space()? {
                return Ok(true);
            }

            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::AudioData;
    use super::*;

    #[test]
    fn test_stream_config_default() {
        let config = RealTimeStreamConfig::default();
        assert_eq!(config.sample_rate, 22050);
        assert_eq!(config.channels, 1);
        assert_eq!(config.target_latency_ms, 50);
    }

    #[test]
    fn test_buffer_config_default() {
        let config = BufferConfig::default();
        assert_eq!(config.buffer_count, 8);
        assert_eq!(config.buffer_size, 512);
        assert_eq!(config.underrun_threshold, 2);
    }

    #[test]
    fn test_audio_buffer() {
        let mut buffer = AudioBuffer::new(4);
        assert!(!buffer.is_ready);

        let samples = vec![0.1, 0.2, 0.3, 0.4];
        buffer.write_samples(&samples);
        assert!(buffer.is_ready);

        let mut output = vec![0.0; 4];
        let samples_read = buffer.read_samples(&mut output);
        assert_eq!(samples_read, 4);
        assert_eq!(output, samples);
        assert!(!buffer.is_ready);
    }

    #[tokio::test]
    async fn test_realtime_stream_creation() {
        let stream_config = RealTimeStreamConfig::default();
        let buffer_config = BufferConfig::default();

        // This test might fail in CI environments without audio devices
        match RealTimeAudioStream::new(stream_config, buffer_config) {
            Ok(stream) => {
                assert!(!stream.is_active().unwrap());
                let fill_level = stream.get_buffer_fill_level().unwrap();
                assert!(fill_level >= 0.0 && fill_level <= 1.0);
            }
            Err(_) => {
                // It's okay if no audio devices are available in test environment
            }
        }
    }

    #[tokio::test]
    async fn test_stream_buffer_operations() {
        let stream_config = RealTimeStreamConfig::default();
        let buffer_config = BufferConfig::default();

        if let Ok(stream) = RealTimeAudioStream::new(stream_config, buffer_config) {
            let audio_data = AudioData {
                samples: vec![0, 1000, 2000, 3000],
                sample_rate: 22050,
                channels: 1,
            };

            // Write audio data
            stream.write_audio(&audio_data).unwrap();

            // Check buffer fill level increased
            let fill_level = stream.get_buffer_fill_level().unwrap();
            assert!(fill_level > 0.0);

            // Test buffer space check
            assert!(stream.has_sufficient_buffer_space().unwrap());
        }
    }

    #[tokio::test]
    async fn test_stream_stats() {
        let stream_config = RealTimeStreamConfig::default();
        let buffer_config = BufferConfig::default();

        if let Ok(stream) = RealTimeAudioStream::new(stream_config, buffer_config) {
            let stats = stream.get_stats().unwrap();
            assert_eq!(stats.buffers_played, 0);
            assert_eq!(stats.buffers_dropped, 0);
            assert_eq!(stats.underruns, 0);

            // Reset stats should work
            stream.reset_stats().unwrap();
            let stats_after_reset = stream.get_stats().unwrap();
            assert_eq!(stats_after_reset.buffers_played, 0);
        }
    }
}
