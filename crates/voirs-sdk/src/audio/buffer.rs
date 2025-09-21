//! Core audio buffer implementation and basic operations.

use crate::types::AudioFormat;
use serde::{Deserialize, Serialize};

/// Audio buffer containing synthesized speech
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioBuffer {
    /// Audio samples as f32 values in range [-1.0, 1.0]
    pub(super) samples: Vec<f32>,

    /// Sample rate in Hz
    pub(super) sample_rate: u32,

    /// Number of audio channels (1=mono, 2=stereo)
    pub(super) channels: u32,

    /// Audio metadata
    pub(super) metadata: AudioMetadata,
}

impl AudioBuffer {
    /// Create new audio buffer
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u32) -> Self {
        let metadata = AudioMetadata::from_samples(&samples, sample_rate, channels);

        Self {
            samples,
            sample_rate,
            channels,
            metadata,
        }
    }

    /// Create mono audio buffer
    pub fn mono(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self::new(samples, sample_rate, 1)
    }

    /// Create stereo audio buffer
    pub fn stereo(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self::new(samples, sample_rate, 2)
    }

    /// Get audio samples as slice
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Get mutable access to samples
    pub fn samples_mut(&mut self) -> &mut [f32] {
        &mut self.samples
    }

    /// Get sample rate in Hz
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get number of channels
    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// Get duration in seconds
    pub fn duration(&self) -> f32 {
        self.metadata.duration
    }

    /// Get audio metadata
    pub fn metadata(&self) -> &AudioMetadata {
        &self.metadata
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Create silent audio buffer
    pub fn silence(duration_seconds: f32, sample_rate: u32, channels: u32) -> Self {
        let sample_count = (duration_seconds * sample_rate as f32 * channels as f32) as usize;
        let samples = vec![0.0; sample_count];
        Self::new(samples, sample_rate, channels)
    }

    /// Create audio buffer with sine wave (for testing)
    pub fn sine_wave(
        frequency: f32,
        duration_seconds: f32,
        sample_rate: u32,
        amplitude: f32,
    ) -> Self {
        let sample_count = (duration_seconds * sample_rate as f32) as usize;
        let mut samples = Vec::with_capacity(sample_count);

        for i in 0..sample_count {
            let t = i as f32 / sample_rate as f32;
            let sample = amplitude * (2.0 * std::f32::consts::PI * frequency * t).sin();
            samples.push(sample);
        }

        Self::mono(samples, sample_rate)
    }

    /// Update metadata after modifications
    pub(super) fn update_metadata(&mut self) {
        self.metadata = AudioMetadata::from_samples(&self.samples, self.sample_rate, self.channels);
    }

    /// Create a new buffer with the same format but different samples
    pub fn with_samples(&self, samples: Vec<f32>) -> Self {
        Self::new(samples, self.sample_rate, self.channels)
    }

    /// Clone the buffer format without samples
    pub fn clone_format(&self) -> BufferFormat {
        BufferFormat {
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
    }

    /// Create buffer from format and samples
    pub fn from_format(format: &BufferFormat, samples: Vec<f32>) -> Self {
        Self::new(samples, format.sample_rate, format.channels)
    }
}

/// Audio metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMetadata {
    /// Duration in seconds
    pub duration: f32,

    /// Peak amplitude (0.0 - 1.0)
    pub peak_amplitude: f32,

    /// RMS amplitude (0.0 - 1.0)
    pub rms_amplitude: f32,

    /// Dynamic range in dB
    pub dynamic_range: f32,

    /// Audio format
    pub format: AudioFormat,
}

impl AudioMetadata {
    /// Create metadata from audio samples
    pub fn from_samples(samples: &[f32], sample_rate: u32, channels: u32) -> Self {
        let duration = samples.len() as f32 / (sample_rate * channels) as f32;
        let peak_amplitude = samples.iter().map(|&s| s.abs()).fold(0.0, f32::max);
        let sum_squares: f32 = samples.iter().map(|&s| s * s).sum();
        let rms_amplitude = if samples.is_empty() {
            0.0
        } else {
            (sum_squares / samples.len() as f32).sqrt()
        };

        // Calculate dynamic range (simplified)
        let dynamic_range = if rms_amplitude > 0.0 {
            20.0 * (peak_amplitude / rms_amplitude).log10()
        } else {
            0.0
        };

        Self {
            duration,
            peak_amplitude,
            rms_amplitude,
            dynamic_range,
            format: AudioFormat::Wav, // Default format
        }
    }
}

impl Default for AudioMetadata {
    fn default() -> Self {
        Self {
            duration: 0.0,
            peak_amplitude: 0.0,
            rms_amplitude: 0.0,
            dynamic_range: 0.0,
            format: AudioFormat::Wav,
        }
    }
}

/// Buffer format specification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferFormat {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u32,
}

impl BufferFormat {
    /// Create new buffer format
    pub fn new(sample_rate: u32, channels: u32) -> Self {
        Self {
            sample_rate,
            channels,
        }
    }

    /// Mono format
    pub fn mono(sample_rate: u32) -> Self {
        Self::new(sample_rate, 1)
    }

    /// Stereo format
    pub fn stereo(sample_rate: u32) -> Self {
        Self::new(sample_rate, 2)
    }

    /// Check if formats are compatible
    pub fn is_compatible(&self, other: &BufferFormat) -> bool {
        self.sample_rate == other.sample_rate && self.channels == other.channels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_buffer_creation() {
        let samples = vec![0.1, 0.2, 0.3, 0.4];
        let buffer = AudioBuffer::mono(samples.clone(), 44100);

        assert_eq!(buffer.samples(), &samples);
        assert_eq!(buffer.sample_rate(), 44100);
        assert_eq!(buffer.channels(), 1);
        assert!(buffer.duration() > 0.0);
    }

    #[test]
    fn test_sine_wave_generation() {
        let buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);

        assert_eq!(buffer.sample_rate(), 44100);
        assert_eq!(buffer.channels(), 1);
        assert_eq!(buffer.len(), 44100);
        assert!((buffer.duration() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_silence_generation() {
        let buffer = AudioBuffer::silence(2.0, 22050, 2);

        assert_eq!(buffer.sample_rate(), 22050);
        assert_eq!(buffer.channels(), 2);
        assert_eq!(buffer.len(), 88200); // 2 seconds * 22050 Hz * 2 channels
        assert!(buffer.samples().iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_metadata_calculation() {
        let samples = vec![0.5, -0.3, 0.8, -0.1];
        let buffer = AudioBuffer::mono(samples, 44100);

        let metadata = buffer.metadata();
        assert_eq!(metadata.peak_amplitude, 0.8);
        assert!(metadata.rms_amplitude > 0.0);
        assert!(metadata.duration > 0.0);
    }

    #[test]
    fn test_buffer_format() {
        let format = BufferFormat::mono(44100);
        assert_eq!(format.sample_rate, 44100);
        assert_eq!(format.channels, 1);

        let stereo_format = BufferFormat::stereo(22050);
        assert_eq!(stereo_format.sample_rate, 22050);
        assert_eq!(stereo_format.channels, 2);

        assert!(!format.is_compatible(&stereo_format));
    }

    #[test]
    fn test_buffer_cloning() {
        let samples = vec![0.1, 0.2, 0.3];
        let buffer = AudioBuffer::mono(samples.clone(), 44100);

        let format = buffer.clone_format();
        let new_buffer = AudioBuffer::from_format(&format, vec![0.4, 0.5, 0.6]);

        assert_eq!(new_buffer.sample_rate(), 44100);
        assert_eq!(new_buffer.channels(), 1);
        assert_eq!(new_buffer.samples(), &[0.4, 0.5, 0.6]);
    }
}
