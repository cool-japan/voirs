//! Time-stretching augmentation using phase vocoder
//!
//! This module provides high-quality time-stretching without pitch modification
//! using the phase vocoder algorithm. It's particularly useful for:
//! - Adjusting speech rate without changing speaker characteristics
//! - Data augmentation for varying speaking speeds
//! - Creating training data with different temporal characteristics

use crate::{AudioData, DatasetError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, Zero};
use scirs2_core::{Complex, Complex64};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Time-stretching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeStretchConfig {
    /// Time-stretching factors to apply (1.0 = no change, <1.0 = slower, >1.0 = faster)
    pub stretch_factors: Vec<f32>,
    /// FFT window size (should be power of 2)
    pub window_size: usize,
    /// Hop size as fraction of window size
    pub hop_ratio: f32,
    /// Enable phase locking for better transient preservation
    pub phase_locking: bool,
    /// Use Hann window for analysis/synthesis
    pub use_hann_window: bool,
}

impl Default for TimeStretchConfig {
    fn default() -> Self {
        Self {
            stretch_factors: vec![0.8, 0.9, 1.1, 1.2],
            window_size: 2048,
            hop_ratio: 0.25,
            phase_locking: true,
            use_hann_window: true,
        }
    }
}

impl TimeStretchConfig {
    /// Create configuration for speech (optimized for voice)
    pub fn for_speech() -> Self {
        Self {
            stretch_factors: vec![0.85, 0.9, 0.95, 1.05, 1.1, 1.15],
            window_size: 1024,
            hop_ratio: 0.25,
            phase_locking: true,
            use_hann_window: true,
        }
    }

    /// Create configuration for music (optimized for harmonic content)
    pub fn for_music() -> Self {
        Self {
            stretch_factors: vec![0.8, 0.9, 1.1, 1.2],
            window_size: 4096,
            hop_ratio: 0.125,
            phase_locking: true,
            use_hann_window: true,
        }
    }

    /// Create configuration for aggressive augmentation
    pub fn aggressive() -> Self {
        Self {
            stretch_factors: vec![0.7, 0.8, 0.9, 1.1, 1.2, 1.3],
            window_size: 2048,
            hop_ratio: 0.25,
            phase_locking: true,
            use_hann_window: true,
        }
    }
}

/// Phase vocoder time-stretching augmentor
pub struct TimeStretchAugmentor {
    config: TimeStretchConfig,
}

impl TimeStretchAugmentor {
    /// Create new time-stretching augmentor
    pub fn new(config: TimeStretchConfig) -> Self {
        Self { config }
    }

    /// Apply time-stretching to audio
    pub fn stretch(&mut self, audio: &AudioData, stretch_factor: f32) -> Result<AudioData> {
        if stretch_factor <= 0.0 {
            return Err(DatasetError::AudioError(
                "Stretch factor must be positive".to_string(),
            ));
        }

        if stretch_factor == 1.0 {
            return Ok(audio.clone());
        }

        let samples = audio.samples();
        if samples.is_empty() {
            return Ok(audio.clone());
        }

        // Process each channel separately
        let num_channels = audio.channels() as usize;
        let samples_per_channel = samples.len() / num_channels;

        let mut stretched_channels = Vec::new();

        for ch in 0..num_channels {
            // Extract channel data
            let channel_data: Vec<f32> = if num_channels == 1 {
                samples.to_vec()
            } else {
                samples
                    .iter()
                    .skip(ch)
                    .step_by(num_channels)
                    .copied()
                    .collect()
            };

            // Apply phase vocoder
            let stretched = self.phase_vocoder(&channel_data, stretch_factor)?;
            stretched_channels.push(stretched);
        }

        // Interleave channels if multi-channel
        let output_samples = if num_channels == 1 {
            stretched_channels[0].clone()
        } else {
            let output_len = stretched_channels[0].len() * num_channels;
            let mut interleaved = vec![0.0; output_len];

            for i in 0..stretched_channels[0].len() {
                for (ch, channel_data) in stretched_channels.iter().enumerate() {
                    interleaved[i * num_channels + ch] = channel_data[i];
                }
            }
            interleaved
        };

        Ok(AudioData::new(
            output_samples,
            audio.sample_rate(),
            audio.channels(),
        ))
    }

    /// Phase vocoder implementation
    fn phase_vocoder(&mut self, input: &[f32], stretch_factor: f32) -> Result<Vec<f32>> {
        let window_size = self.config.window_size;
        let hop_size = (window_size as f32 * self.config.hop_ratio) as usize;
        // Invert stretch_factor: <1.0 = slower (longer), >1.0 = faster (shorter)
        let synthesis_hop = (hop_size as f32 / stretch_factor) as usize;

        // Create analysis and synthesis windows
        let analysis_window = if self.config.use_hann_window {
            self.create_hann_window(window_size)
        } else {
            vec![1.0; window_size]
        };

        let synthesis_window = analysis_window.clone();

        // Calculate number of frames
        let num_frames = (input.len() - window_size) / hop_size + 1;
        let output_length = (num_frames - 1) * synthesis_hop + window_size;

        // Initialize output buffer
        let mut output = vec![0.0; output_length];
        let mut norm_window = vec![0.0; output_length];

        // Previous frame phase for phase unwrapping
        let mut prev_phase = vec![0.0; window_size / 2 + 1];
        let mut phase_cumulative = vec![0.0; window_size / 2 + 1];

        // Expected phase advance per bin
        let expected_phase_advance: Vec<f32> = (0..=window_size / 2)
            .map(|k| 2.0 * PI * hop_size as f32 * k as f32 / window_size as f32)
            .collect();

        // Process each frame
        for frame_idx in 0..num_frames {
            let input_pos = frame_idx * hop_size;
            let output_pos = frame_idx * synthesis_hop;

            // Extract and window the frame
            let windowed_frame: Vec<f32> = (0..window_size)
                .map(|i| {
                    if input_pos + i < input.len() {
                        input[input_pos + i] * analysis_window[i]
                    } else {
                        0.0
                    }
                })
                .collect();

            // Forward FFT using scirs2-fft
            let spectrum64 = scirs2_fft::fft(&windowed_frame, Some(window_size))
                .map_err(|e| DatasetError::AudioError(format!("FFT error: {}", e)))?;

            // Convert to f32 Complex for processing
            let spectrum: Vec<Complex<f32>> = spectrum64
                .iter()
                .map(|c| Complex::new(c.re as f32, c.im as f32))
                .collect();

            // Extract magnitude and phase
            let mut magnitude = vec![0.0; window_size / 2 + 1];
            let mut phase = vec![0.0; window_size / 2 + 1];

            for k in 0..=window_size / 2 {
                magnitude[k] = spectrum[k].norm();
                phase[k] = spectrum[k].arg();
            }

            // Phase unwrapping and accumulation
            if frame_idx > 0 {
                for k in 0..=window_size / 2 {
                    // Calculate phase difference
                    let mut phase_diff = phase[k] - prev_phase[k];

                    // Remove expected phase advance
                    phase_diff -= expected_phase_advance[k];

                    // Wrap to [-π, π]
                    while phase_diff > PI {
                        phase_diff -= 2.0 * PI;
                    }
                    while phase_diff < -PI {
                        phase_diff += 2.0 * PI;
                    }

                    // Add back expected phase advance
                    phase_diff += expected_phase_advance[k];

                    // Accumulate phase (inverted stretch_factor)
                    phase_cumulative[k] += phase_diff / stretch_factor;

                    // Apply phase locking for better transient preservation
                    if self.config.phase_locking && k > 0 {
                        let peak_bin = self.find_local_peak(&magnitude, k, 2);
                        if peak_bin != k {
                            let phase_offset = phase_cumulative[peak_bin] - phase_cumulative[k];
                            phase_cumulative[k] = phase_cumulative[peak_bin] - phase_offset;
                        }
                    }
                }
            }

            prev_phase.copy_from_slice(&phase);

            // Reconstruct spectrum with new phases
            let mut modified_spectrum = vec![Complex::zero(); window_size];
            for k in 0..=window_size / 2 {
                let mag = magnitude[k];
                let ph = if frame_idx == 0 {
                    phase[k]
                } else {
                    phase_cumulative[k]
                };
                modified_spectrum[k] = Complex::new(mag * ph.cos(), mag * ph.sin());
            }

            // Conjugate symmetry for real signal
            for k in 1..window_size / 2 {
                modified_spectrum[window_size - k] = modified_spectrum[k].conj();
            }

            // Inverse FFT using scirs2-fft
            // Convert back to f64 for scirs2-fft
            let modified_spectrum64: Vec<Complex64> = modified_spectrum
                .iter()
                .map(|c| Complex64::new(c.re as f64, c.im as f64))
                .collect();

            let time_domain64 = scirs2_fft::ifft(&modified_spectrum64, Some(window_size))
                .map_err(|e| DatasetError::AudioError(format!("IFFT error: {}", e)))?;

            // Overlap-add with synthesis window
            for i in 0..window_size {
                if output_pos + i < output.len() {
                    let sample = time_domain64[i].re as f32;
                    output[output_pos + i] += sample * synthesis_window[i];
                    norm_window[output_pos + i] += synthesis_window[i] * synthesis_window[i];
                }
            }
        }

        // Normalize by window overlap
        for i in 0..output.len() {
            if norm_window[i] > 1e-8 {
                output[i] /= norm_window[i];
            }
        }

        Ok(output)
    }

    /// Create Hann window
    fn create_hann_window(&self, size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                let phase = 2.0 * PI * i as f32 / (size - 1) as f32;
                0.5 * (1.0 - phase.cos())
            })
            .collect()
    }

    /// Find local peak in magnitude spectrum
    #[allow(clippy::needless_range_loop)]
    fn find_local_peak(&self, magnitude: &[f32], center: usize, range: usize) -> usize {
        let start = center.saturating_sub(range);
        let end = (center + range + 1).min(magnitude.len());

        let mut peak_idx = center;
        let mut peak_mag = magnitude[center];

        for i in start..end {
            if magnitude[i] > peak_mag {
                peak_mag = magnitude[i];
                peak_idx = i;
            }
        }

        peak_idx
    }

    /// Process multiple stretch factors
    pub fn stretch_multiple(&mut self, audio: &AudioData) -> Result<Vec<AudioData>> {
        let mut results = Vec::new();
        let stretch_factors = self.config.stretch_factors.clone();

        for factor in stretch_factors {
            let stretched = self.stretch(audio, factor)?;
            results.push(stretched);
        }

        Ok(results)
    }
}

/// Batch time-stretching processor
pub struct BatchTimeStretchAugmentor {
    augmentor: TimeStretchAugmentor,
}

impl BatchTimeStretchAugmentor {
    /// Create new batch processor
    pub fn new(config: TimeStretchConfig) -> Self {
        Self {
            augmentor: TimeStretchAugmentor::new(config),
        }
    }

    /// Process batch of audio samples
    pub fn stretch_batch(
        &mut self,
        audio_batch: &[AudioData],
        stretch_factor: f32,
    ) -> Result<Vec<AudioData>> {
        audio_batch
            .iter()
            .map(|audio| self.augmentor.stretch(audio, stretch_factor))
            .collect()
    }

    /// Process batch with multiple stretch factors
    pub fn stretch_batch_multiple(&mut self, audio_batch: &[AudioData]) -> Result<Vec<AudioData>> {
        let mut results = Vec::new();

        for audio in audio_batch {
            let stretched = self.augmentor.stretch_multiple(audio)?;
            results.extend(stretched);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_audio(sample_rate: u32, duration: f32) -> AudioData {
        let num_samples = (sample_rate as f32 * duration) as usize;
        let frequency = 440.0; // A4 note

        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * PI * frequency * t).sin() * 0.5
            })
            .collect();

        AudioData::new(samples, sample_rate, 1)
    }

    #[test]
    fn test_config_default() {
        let config = TimeStretchConfig::default();
        assert_eq!(config.stretch_factors, vec![0.8, 0.9, 1.1, 1.2]);
        assert_eq!(config.window_size, 2048);
        assert_eq!(config.hop_ratio, 0.25);
        assert!(config.phase_locking);
        assert!(config.use_hann_window);
    }

    #[test]
    fn test_config_for_speech() {
        let config = TimeStretchConfig::for_speech();
        assert_eq!(
            config.stretch_factors,
            vec![0.85, 0.9, 0.95, 1.05, 1.1, 1.15]
        );
        assert_eq!(config.window_size, 1024);
    }

    #[test]
    fn test_config_for_music() {
        let config = TimeStretchConfig::for_music();
        assert_eq!(config.window_size, 4096);
        assert_eq!(config.hop_ratio, 0.125);
    }

    #[test]
    fn test_config_aggressive() {
        let config = TimeStretchConfig::aggressive();
        assert_eq!(config.stretch_factors, vec![0.7, 0.8, 0.9, 1.1, 1.2, 1.3]);
    }

    #[test]
    fn test_stretch_factor_one() {
        let config = TimeStretchConfig::default();
        let mut augmentor = TimeStretchAugmentor::new(config);
        let audio = create_test_audio(16000, 0.5);

        let stretched = augmentor.stretch(&audio, 1.0).unwrap();

        // Should return identical audio
        assert_eq!(stretched.samples().len(), audio.samples().len());
        assert_eq!(stretched.sample_rate(), audio.sample_rate());
    }

    #[test]
    fn test_stretch_slower() {
        let config = TimeStretchConfig::default();
        let mut augmentor = TimeStretchAugmentor::new(config);
        let audio = create_test_audio(16000, 0.5);

        let stretched = augmentor.stretch(&audio, 0.8).unwrap();

        // Slower = longer audio
        assert!(stretched.samples().len() > audio.samples().len());
        assert_eq!(stretched.sample_rate(), audio.sample_rate());
    }

    #[test]
    fn test_stretch_faster() {
        let config = TimeStretchConfig::default();
        let mut augmentor = TimeStretchAugmentor::new(config);
        let audio = create_test_audio(16000, 0.5);

        let stretched = augmentor.stretch(&audio, 1.2).unwrap();

        // Faster = shorter audio
        assert!(stretched.samples().len() < audio.samples().len());
        assert_eq!(stretched.sample_rate(), audio.sample_rate());
    }

    #[test]
    fn test_stretch_invalid_factor() {
        let config = TimeStretchConfig::default();
        let mut augmentor = TimeStretchAugmentor::new(config);
        let audio = create_test_audio(16000, 0.5);

        let result = augmentor.stretch(&audio, 0.0);
        assert!(result.is_err());

        let result = augmentor.stretch(&audio, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_stretch_empty_audio() {
        let config = TimeStretchConfig::default();
        let mut augmentor = TimeStretchAugmentor::new(config);
        let audio = AudioData::new(vec![], 16000, 1);

        let stretched = augmentor.stretch(&audio, 0.9).unwrap();

        assert!(stretched.samples().is_empty());
    }

    #[test]
    fn test_stretch_multiple() {
        let config = TimeStretchConfig {
            stretch_factors: vec![0.8, 1.0, 1.2],
            ..Default::default()
        };
        let mut augmentor = TimeStretchAugmentor::new(config);
        let audio = create_test_audio(16000, 0.5);

        let results = augmentor.stretch_multiple(&audio).unwrap();

        assert_eq!(results.len(), 3);
        // First should be longer (slower)
        assert!(results[0].samples().len() > audio.samples().len());
        // Second should be same (factor 1.0)
        assert_eq!(results[1].samples().len(), audio.samples().len());
        // Third should be shorter (faster)
        assert!(results[2].samples().len() < audio.samples().len());
    }

    #[test]
    fn test_hann_window() {
        let config = TimeStretchConfig::default();
        let augmentor = TimeStretchAugmentor::new(config);
        let window = augmentor.create_hann_window(256);

        assert_eq!(window.len(), 256);
        // Hann window should start and end near zero
        assert!(window[0].abs() < 0.01);
        assert!(window[255].abs() < 0.01);
        // Peak should be near center
        let max_val = window.iter().copied().fold(0.0f32, f32::max);
        assert!((max_val - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_stretching() {
        let config = TimeStretchConfig::default();
        let mut batch_augmentor = BatchTimeStretchAugmentor::new(config);

        let audio1 = create_test_audio(16000, 0.5);
        let audio2 = create_test_audio(16000, 0.3);
        let batch = vec![audio1, audio2];

        let results = batch_augmentor.stretch_batch(&batch, 1.1).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].samples().len() < batch[0].samples().len());
        assert!(results[1].samples().len() < batch[1].samples().len());
    }

    #[test]
    fn test_batch_multiple_factors() {
        let config = TimeStretchConfig {
            stretch_factors: vec![0.9, 1.1],
            ..Default::default()
        };
        let mut batch_augmentor = BatchTimeStretchAugmentor::new(config);

        let audio = create_test_audio(16000, 0.5);
        let batch = vec![audio];

        let results = batch_augmentor.stretch_batch_multiple(&batch).unwrap();

        // Should have 2 results (2 stretch factors)
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_multi_channel_stretch() {
        let config = TimeStretchConfig::default();
        let mut augmentor = TimeStretchAugmentor::new(config);

        // Create stereo test audio
        let num_samples = 8000;
        let mut stereo_samples = Vec::new();
        for i in 0..num_samples {
            let t = i as f32 / 16000.0;
            let left = (2.0 * PI * 440.0 * t).sin() * 0.5;
            let right = (2.0 * PI * 550.0 * t).sin() * 0.5;
            stereo_samples.push(left);
            stereo_samples.push(right);
        }

        let stereo_audio = AudioData::new(stereo_samples, 16000, 2);

        let stretched = augmentor.stretch(&stereo_audio, 0.9).unwrap();

        // Should maintain stereo format
        assert_eq!(stretched.channels(), 2);
        // Should be longer (slower)
        assert!(stretched.samples().len() > stereo_audio.samples().len());
    }
}
