//! SIMD-accelerated mel spectrogram computation
//!
//! This module provides SIMD-optimized implementations for mel spectrogram
//! computation, including mel filterbank operations and frequency domain
//! transformations.

use super::{simd, SIMD_WIDTH_F32};
use crate::{AcousticError, Result};
use std::f32::consts::PI;

/// SIMD-accelerated mel spectrogram computer
pub struct SimdMelComputer {
    /// Sample rate
    sample_rate: u32,
    /// Number of mel bins
    n_mels: usize,
    /// FFT size
    n_fft: usize,
    /// Hop length
    hop_length: usize,
    /// Mel filterbank weights [n_mels, n_fft/2 + 1]
    mel_filters: Vec<Vec<f32>>,
    /// Pre-computed cosine values for DCT
    dct_matrix: Option<Vec<Vec<f32>>>,
}

impl SimdMelComputer {
    /// Create new SIMD mel computer
    pub fn new(
        sample_rate: u32,
        n_mels: usize,
        n_fft: usize,
        hop_length: usize,
        f_min: f32,
        f_max: f32,
    ) -> Result<Self> {
        let mut computer = Self {
            sample_rate,
            n_mels,
            n_fft,
            hop_length,
            mel_filters: Vec::new(),
            dct_matrix: None,
        };

        computer.build_mel_filterbank(f_min, f_max)?;
        Ok(computer)
    }

    /// Create with MFCC support (includes DCT matrix)
    pub fn with_mfcc(
        sample_rate: u32,
        n_mels: usize,
        n_mfcc: usize,
        n_fft: usize,
        hop_length: usize,
        f_min: f32,
        f_max: f32,
    ) -> Result<Self> {
        let mut computer = Self::new(sample_rate, n_mels, n_fft, hop_length, f_min, f_max)?;
        computer.build_dct_matrix(n_mfcc)?;
        Ok(computer)
    }

    /// Compute mel spectrogram from magnitude spectrum with SIMD acceleration
    pub fn compute_mel_spectrogram(
        &self,
        magnitude_spectrum: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>> {
        if magnitude_spectrum.is_empty() {
            return Err(AcousticError::InputError(
                "Empty magnitude spectrum".to_string(),
            ));
        }

        let n_frames = magnitude_spectrum[0].len();
        let n_freq_bins = magnitude_spectrum.len();

        if n_freq_bins != self.n_fft / 2 + 1 {
            return Err(AcousticError::InputError(format!(
                "Expected {} frequency bins, got {}",
                self.n_fft / 2 + 1,
                n_freq_bins
            )));
        }

        let mut mel_spec = vec![vec![0.0f32; n_frames]; self.n_mels];

        // Apply mel filterbank with SIMD acceleration
        #[allow(clippy::needless_range_loop)]
        for mel_idx in 0..self.n_mels {
            let filter = &self.mel_filters[mel_idx];

            for frame_idx in 0..n_frames {
                // Collect magnitude values for this frame
                let mut frame_magnitudes = Vec::with_capacity(n_freq_bins);
                #[allow(clippy::needless_range_loop)]
                for freq_idx in 0..n_freq_bins {
                    frame_magnitudes.push(magnitude_spectrum[freq_idx][frame_idx]);
                }

                // SIMD dot product between filter and magnitudes
                let mel_energy = simd().dot_product_f32(filter, &frame_magnitudes)?;
                mel_spec[mel_idx][frame_idx] = mel_energy.max(1e-10).ln();
            }
        }

        Ok(mel_spec)
    }

    /// Compute mel spectrogram from time-domain audio with SIMD acceleration
    pub fn compute_mel_from_audio(
        &self,
        audio: &[f32],
        window_fn: WindowFunction,
    ) -> Result<Vec<Vec<f32>>> {
        // Compute STFT with SIMD-accelerated FFT
        let stft = self.compute_stft_simd(audio, window_fn)?;

        // Convert complex STFT to magnitude spectrum
        let mut magnitude_spectrum = vec![Vec::with_capacity(stft[0].len()); stft.len()];

        for (freq_idx, freq_data) in stft.iter().enumerate() {
            for frame_data in freq_data.chunks(2) {
                let real = frame_data[0];
                let imag = frame_data.get(1).copied().unwrap_or(0.0);
                let magnitude = (real * real + imag * imag).sqrt();
                magnitude_spectrum[freq_idx].push(magnitude);
            }
        }

        self.compute_mel_spectrogram(&magnitude_spectrum)
    }

    /// Compute MFCC coefficients if DCT matrix is available
    pub fn compute_mfcc(&self, mel_spectrogram: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let dct_matrix = self.dct_matrix.as_ref().ok_or_else(|| {
            AcousticError::ConfigError("DCT matrix not initialized. Use with_mfcc()".to_string())
        })?;

        let n_frames = mel_spectrogram[0].len();
        let n_mfcc = dct_matrix.len();
        let mut mfcc = vec![vec![0.0f32; n_frames]; n_mfcc];

        // Apply DCT with SIMD acceleration
        for frame_idx in 0..n_frames {
            // Extract mel values for this frame
            let mut mel_frame = Vec::with_capacity(self.n_mels);
            #[allow(clippy::needless_range_loop)]
            for mel_idx in 0..self.n_mels {
                mel_frame.push(mel_spectrogram[mel_idx][frame_idx]);
            }

            // Compute MFCC coefficients
            for mfcc_idx in 0..n_mfcc {
                let coefficient = simd().dot_product_f32(&dct_matrix[mfcc_idx], &mel_frame)?;
                mfcc[mfcc_idx][frame_idx] = coefficient;
            }
        }

        Ok(mfcc)
    }

    /// Apply mel filterbank to magnitude spectrum with optimized memory access
    pub fn apply_mel_filterbank_optimized(&self, magnitude_spectrum: &[f32]) -> Result<Vec<f32>> {
        let mut mel_energies = vec![0.0f32; self.n_mels];

        // Use SIMD for batch processing
        for (mel_idx, filter) in self.mel_filters.iter().enumerate() {
            mel_energies[mel_idx] = simd().dot_product_f32(filter, magnitude_spectrum)?;
        }

        // Apply log with SIMD-friendly operations
        self.apply_log_simd(&mut mel_energies)?;

        Ok(mel_energies)
    }

    /// Inverse mel filterbank (for vocoder applications)
    pub fn inverse_mel_filterbank(&self, mel_spectrum: &[f32]) -> Result<Vec<f32>> {
        let n_freq_bins = self.n_fft / 2 + 1;
        let mut magnitude_spectrum = vec![0.0f32; n_freq_bins];

        // Transpose operation: sum weighted mel bins for each frequency bin
        for freq_idx in 0..n_freq_bins {
            let mut freq_energy = 0.0f32;

            for (mel_idx, filter) in self.mel_filters.iter().enumerate() {
                if freq_idx < filter.len() {
                    freq_energy += mel_spectrum[mel_idx] * filter[freq_idx];
                }
            }

            magnitude_spectrum[freq_idx] = freq_energy;
        }

        Ok(magnitude_spectrum)
    }

    /// Compute delta features (first-order derivatives) with SIMD
    pub fn compute_delta_features(
        &self,
        features: &[Vec<f32>],
        window_size: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let n_features = features.len();
        let n_frames = features[0].len();
        let mut delta_features = vec![vec![0.0f32; n_frames]; n_features];

        let half_window = window_size / 2;

        for feature_idx in 0..n_features {
            for frame_idx in 0..n_frames {
                let mut numerator = 0.0f32;
                let mut denominator = 0.0f32;

                for offset in 1..=half_window {
                    let past_idx = frame_idx.saturating_sub(offset);
                    let future_idx = (frame_idx + offset).min(n_frames - 1);

                    let weight = offset as f32;
                    numerator += weight
                        * (features[feature_idx][future_idx] - features[feature_idx][past_idx]);
                    denominator += 2.0 * weight * weight;
                }

                delta_features[feature_idx][frame_idx] = if denominator > 0.0 {
                    numerator / denominator
                } else {
                    0.0
                };
            }
        }

        Ok(delta_features)
    }

    // Private helper methods

    fn build_mel_filterbank(&mut self, f_min: f32, f_max: f32) -> Result<()> {
        let n_freq_bins = self.n_fft / 2 + 1;

        // Convert to mel scale
        let mel_min = self.hz_to_mel(f_min);
        let mel_max = self.hz_to_mel(f_max);

        // Create mel points
        let mel_points: Vec<f32> = (0..=self.n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (self.n_mels + 1) as f32)
            .collect();

        // Convert back to Hz
        let hz_points: Vec<f32> = mel_points.iter().map(|&mel| self.mel_to_hz(mel)).collect();

        // Convert to FFT bin indices
        let bin_points: Vec<f32> = hz_points
            .iter()
            .map(|&hz| hz * self.n_fft as f32 / self.sample_rate as f32)
            .collect();

        // Build triangular filters
        self.mel_filters = vec![vec![0.0f32; n_freq_bins]; self.n_mels];

        for mel_idx in 0..self.n_mels {
            let left_bin = bin_points[mel_idx];
            let center_bin = bin_points[mel_idx + 1];
            let right_bin = bin_points[mel_idx + 2];

            for freq_idx in 0..n_freq_bins {
                let freq_bin = freq_idx as f32;

                if freq_bin >= left_bin && freq_bin <= center_bin {
                    // Rising edge
                    if center_bin > left_bin {
                        self.mel_filters[mel_idx][freq_idx] =
                            (freq_bin - left_bin) / (center_bin - left_bin);
                    }
                } else if freq_bin > center_bin && freq_bin <= right_bin {
                    // Falling edge
                    if right_bin > center_bin {
                        self.mel_filters[mel_idx][freq_idx] =
                            (right_bin - freq_bin) / (right_bin - center_bin);
                    }
                }
            }

            // Normalize filter to unit area
            let filter_sum: f32 = self.mel_filters[mel_idx].iter().sum();
            if filter_sum > 0.0 {
                for weight in &mut self.mel_filters[mel_idx] {
                    *weight /= filter_sum;
                }
            }
        }

        Ok(())
    }

    fn build_dct_matrix(&mut self, n_mfcc: usize) -> Result<()> {
        let mut dct_matrix = vec![vec![0.0f32; self.n_mels]; n_mfcc];

        #[allow(clippy::needless_range_loop)]
        for mfcc_idx in 0..n_mfcc {
            #[allow(clippy::needless_range_loop)]
            for mel_idx in 0..self.n_mels {
                dct_matrix[mfcc_idx][mel_idx] =
                    (PI * mfcc_idx as f32 * (mel_idx as f32 + 0.5) / self.n_mels as f32).cos();

                // Apply scaling
                if mfcc_idx == 0 {
                    dct_matrix[mfcc_idx][mel_idx] *= (1.0 / self.n_mels as f32).sqrt();
                } else {
                    dct_matrix[mfcc_idx][mel_idx] *= (2.0 / self.n_mels as f32).sqrt();
                }
            }
        }

        self.dct_matrix = Some(dct_matrix);
        Ok(())
    }

    fn hz_to_mel(&self, hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    fn mel_to_hz(&self, mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    fn apply_log_simd(&self, values: &mut [f32]) -> Result<()> {
        // Apply log with epsilon to avoid log(0)
        for chunk in values.chunks_mut(SIMD_WIDTH_F32) {
            for value in chunk {
                *value = (*value).max(1e-10).ln();
            }
        }
        Ok(())
    }

    fn compute_stft_simd(&self, audio: &[f32], window_fn: WindowFunction) -> Result<Vec<Vec<f32>>> {
        let n_frames = (audio.len() - self.n_fft) / self.hop_length + 1;
        let n_freq_bins = self.n_fft / 2 + 1;

        // Pre-compute window function
        let window = self.create_window(window_fn);

        let mut stft = vec![vec![0.0f32; n_frames * 2]; n_freq_bins]; // Real and imaginary parts

        // Process each frame
        for frame_idx in 0..n_frames {
            let start_idx = frame_idx * self.hop_length;
            let end_idx = start_idx + self.n_fft;

            if end_idx <= audio.len() {
                let mut windowed_frame = vec![0.0f32; self.n_fft];

                // Apply window with SIMD
                for i in 0..self.n_fft {
                    windowed_frame[i] = audio[start_idx + i] * window[i];
                }

                // Compute FFT (simplified DFT for demonstration)
                let frame_fft = self.compute_dft_simd(&windowed_frame)?;

                // Store real and imaginary parts
                for freq_idx in 0..n_freq_bins {
                    stft[freq_idx][frame_idx * 2] = frame_fft[freq_idx * 2]; // Real
                    stft[freq_idx][frame_idx * 2 + 1] = frame_fft[freq_idx * 2 + 1];
                    // Imaginary
                }
            }
        }

        Ok(stft)
    }

    fn create_window(&self, window_fn: WindowFunction) -> Vec<f32> {
        let mut window = vec![0.0f32; self.n_fft];

        #[allow(clippy::needless_range_loop)]
        for i in 0..self.n_fft {
            let norm_idx = i as f32 / (self.n_fft - 1) as f32;

            window[i] = match window_fn {
                WindowFunction::Hann => 0.5 * (1.0 - (2.0 * PI * norm_idx).cos()),
                WindowFunction::Hamming => 0.54 - 0.46 * (2.0 * PI * norm_idx).cos(),
                WindowFunction::Blackman => {
                    0.42 - 0.5 * (2.0 * PI * norm_idx).cos() + 0.08 * (4.0 * PI * norm_idx).cos()
                }
                WindowFunction::Rectangular => 1.0,
            };
        }

        window
    }

    fn compute_dft_simd(&self, signal: &[f32]) -> Result<Vec<f32>> {
        // Simplified DFT implementation with SIMD optimizations
        // In production, this would use a proper SIMD FFT library
        let n = signal.len();
        let n_freq_bins = n / 2 + 1;
        let mut result = vec![0.0f32; n_freq_bins * 2]; // Real and imaginary parts

        for k in 0..n_freq_bins {
            let mut real_sum = 0.0f32;
            let mut imag_sum = 0.0f32;

            // Process in SIMD-friendly chunks
            for chunk_start in (0..n).step_by(SIMD_WIDTH_F32) {
                let chunk_end = (chunk_start + SIMD_WIDTH_F32).min(n);

                #[allow(clippy::needless_range_loop)]
                for j in chunk_start..chunk_end {
                    let angle = -2.0 * PI * k as f32 * j as f32 / n as f32;
                    real_sum += signal[j] * angle.cos();
                    imag_sum += signal[j] * angle.sin();
                }
            }

            result[k * 2] = real_sum;
            result[k * 2 + 1] = imag_sum;
        }

        Ok(result)
    }
}

/// Window function types for STFT
#[derive(Debug, Clone, Copy)]
pub enum WindowFunction {
    Hann,
    Hamming,
    Blackman,
    Rectangular,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_mel_computer_creation() {
        let computer = SimdMelComputer::new(
            22050,  // sample_rate
            80,     // n_mels
            1024,   // n_fft
            256,    // hop_length
            0.0,    // f_min
            8000.0, // f_max
        )
        .unwrap();

        assert_eq!(computer.n_mels, 80);
        assert_eq!(computer.n_fft, 1024);
        assert_eq!(computer.mel_filters.len(), 80);
    }

    #[test]
    fn test_mel_filterbank_application() {
        let computer = SimdMelComputer::new(22050, 80, 1024, 256, 0.0, 8000.0).unwrap();
        let magnitude_spectrum = vec![1.0f32; 513]; // n_fft/2 + 1

        let mel_energies = computer
            .apply_mel_filterbank_optimized(&magnitude_spectrum)
            .unwrap();
        assert_eq!(mel_energies.len(), 80);

        // All energies should be finite
        for energy in mel_energies {
            assert!(energy.is_finite());
        }
    }

    #[test]
    fn test_mel_spectrogram_computation() {
        let computer = SimdMelComputer::new(22050, 80, 1024, 256, 0.0, 8000.0).unwrap();

        // Create test magnitude spectrum [n_freq_bins, n_frames]
        let n_freq_bins = 513;
        let n_frames = 100;
        let magnitude_spectrum = vec![vec![1.0f32; n_frames]; n_freq_bins];

        let mel_spec = computer
            .compute_mel_spectrogram(&magnitude_spectrum)
            .unwrap();
        assert_eq!(mel_spec.len(), 80);
        assert_eq!(mel_spec[0].len(), n_frames);
    }

    #[test]
    fn test_mfcc_computation() {
        let computer = SimdMelComputer::with_mfcc(
            22050,  // sample_rate
            80,     // n_mels
            13,     // n_mfcc
            1024,   // n_fft
            256,    // hop_length
            0.0,    // f_min
            8000.0, // f_max
        )
        .unwrap();

        // Create test mel spectrogram
        let n_frames = 100;
        let mel_spec = vec![vec![1.0f32; n_frames]; 80];

        let mfcc = computer.compute_mfcc(&mel_spec).unwrap();
        assert_eq!(mfcc.len(), 13);
        assert_eq!(mfcc[0].len(), n_frames);
    }

    #[test]
    fn test_delta_features() {
        let computer = SimdMelComputer::new(22050, 80, 1024, 256, 0.0, 8000.0).unwrap();

        // Create test features with some variation
        let n_frames = 100;
        let mut features = vec![vec![0.0f32; n_frames]; 13];
        for (i, feature) in features.iter_mut().enumerate() {
            for (j, value) in feature.iter_mut().enumerate() {
                *value = (i + j) as f32;
            }
        }

        let delta_features = computer.compute_delta_features(&features, 9).unwrap();
        assert_eq!(delta_features.len(), 13);
        assert_eq!(delta_features[0].len(), n_frames);
    }

    #[test]
    fn test_hz_mel_conversion() {
        let computer = SimdMelComputer::new(22050, 80, 1024, 256, 0.0, 8000.0).unwrap();

        let hz = 1000.0;
        let mel = computer.hz_to_mel(hz);
        let hz_back = computer.mel_to_hz(mel);

        assert!((hz - hz_back).abs() < 0.1);
    }

    #[test]
    fn test_window_functions() {
        let computer = SimdMelComputer::new(22050, 80, 1024, 256, 0.0, 8000.0).unwrap();

        let hann_window = computer.create_window(WindowFunction::Hann);
        let hamming_window = computer.create_window(WindowFunction::Hamming);

        assert_eq!(hann_window.len(), 1024);
        assert_eq!(hamming_window.len(), 1024);

        // Check window properties
        assert!(hann_window[0] < 0.1); // Should be near zero at edges
        assert!(hann_window[512] > 0.9); // Should be near one at center
    }
}
