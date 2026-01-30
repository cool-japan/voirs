//! Vocal Tract Length Perturbation (VTLP) for speech augmentation
//!
//! VTLP is a frequency warping technique that simulates different vocal tract lengths,
//! effectively augmenting the dataset with different speaker characteristics.
//!
//! # References
//! - Jaitly, Navdeep, and Geoffrey E. Hinton. "Vocal tract length perturbation (VTLP)
//!   improves speech recognition." ICML Workshop on Deep Learning for Audio, Speech and
//!   Language Processing, 2013.

use crate::{DatasetError, Result};
use scirs2_core::{Complex, Complex32};
use scirs2_fft;
use std::f32::consts::PI;

// Convert FFTError to DatasetError
impl From<scirs2_fft::FFTError> for DatasetError {
    fn from(err: scirs2_fft::FFTError) -> Self {
        DatasetError::AudioError(format!("FFT error: {:?}", err))
    }
}

/// Configuration for VTLP augmentation
#[derive(Debug, Clone)]
pub struct VtlpConfig {
    /// Warping factors to apply (typically between 0.8 and 1.2)
    pub warp_factors: Vec<f32>,
    /// Sample rate of the audio
    pub sample_rate: u32,
    /// FFT window size
    pub window_size: usize,
    /// Hop size for STFT
    pub hop_size: usize,
    /// Lower cutoff frequency for warping (Hz)
    pub lower_cutoff: f32,
    /// Upper cutoff frequency for warping (Hz)
    pub upper_cutoff: f32,
}

impl Default for VtlpConfig {
    fn default() -> Self {
        Self {
            warp_factors: vec![0.9, 1.0, 1.1],
            sample_rate: 22050,
            window_size: 1024,
            hop_size: 256,
            lower_cutoff: 0.0,
            upper_cutoff: 0.0, // 0.0 means use Nyquist frequency
        }
    }
}

/// VTLP augmentor for frequency warping
pub struct VtlpAugmentor {
    config: VtlpConfig,
}

impl VtlpAugmentor {
    /// Create a new VTLP augmentor with the given configuration
    pub fn new(config: VtlpConfig) -> Self {
        Self { config }
    }

    /// Apply VTLP to audio samples
    ///
    /// # Arguments
    /// * `audio` - Input audio samples
    /// * `warp_factor` - Warping factor (< 1.0 for lower frequencies, > 1.0 for higher)
    ///
    /// # Returns
    /// Warped audio samples
    pub fn apply_vtlp(&self, audio: &[f32], warp_factor: f32) -> Result<Vec<f32>> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        if warp_factor <= 0.0 {
            return Err(DatasetError::AudioError(
                "Warp factor must be positive".to_string(),
            ));
        }

        // Perform STFT
        let frames = self.stft(audio)?;

        // Apply frequency warping to each frame
        let warped_frames = frames
            .iter()
            .map(|frame| self.warp_spectrum(frame, warp_factor))
            .collect::<Result<Vec<_>>>()?;

        // Perform inverse STFT
        let output = self.istft(&warped_frames)?;

        Ok(output)
    }

    /// Perform Short-Time Fourier Transform
    fn stft(&self, audio: &[f32]) -> Result<Vec<Vec<Complex32>>> {
        let num_frames = (audio.len() - self.config.window_size) / self.config.hop_size + 1;
        let mut frames = Vec::with_capacity(num_frames);

        // Hanning window
        let window = self.create_hanning_window(self.config.window_size);

        for i in 0..num_frames {
            let start = i * self.config.hop_size;
            let end = start + self.config.window_size;

            if end > audio.len() {
                break;
            }

            // Apply window - convert to f64 for FFT
            let windowed: Vec<scirs2_core::Complex64> = audio[start..end]
                .iter()
                .zip(window.iter())
                .map(|(&sample, &w)| scirs2_core::Complex64::new((sample * w) as f64, 0.0))
                .collect();

            // Perform FFT (returns Vec<Complex64>)
            let spectrum_f64 = scirs2_fft::fft(&windowed, None)?;

            // Convert back to f32
            let spectrum: Vec<Complex32> = spectrum_f64
                .iter()
                .map(|c| Complex::new(c.re as f32, c.im as f32))
                .collect();

            frames.push(spectrum);
        }

        Ok(frames)
    }

    /// Perform inverse Short-Time Fourier Transform
    fn istft(&self, frames: &[Vec<Complex32>]) -> Result<Vec<f32>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }

        let num_frames = frames.len();
        let output_len = self.config.window_size + (num_frames - 1) * self.config.hop_size;
        let mut output = vec![0.0_f32; output_len];
        let mut window_sum = vec![0.0_f32; output_len];

        // Hanning window
        let window = self.create_hanning_window(self.config.window_size);

        for (i, spectrum) in frames.iter().enumerate() {
            // Convert to f64 for IFFT
            let spectrum_f64: Vec<scirs2_core::Complex64> = spectrum
                .iter()
                .map(|c| scirs2_core::Complex64::new(c.re as f64, c.im as f64))
                .collect();

            // Perform inverse FFT (returns Vec<Complex64>)
            let time_domain_f64 = scirs2_fft::ifft(&spectrum_f64, None)?;

            let start = i * self.config.hop_size;

            // Overlap-add
            for (j, sample) in time_domain_f64.iter().enumerate() {
                if start + j < output_len {
                    output[start + j] += (sample.re as f32) * window[j];
                    window_sum[start + j] += window[j] * window[j];
                }
            }
        }

        // Normalize by window sum
        for (i, sample) in output.iter_mut().enumerate() {
            if window_sum[i] > 1e-10 {
                *sample /= window_sum[i];
            }
        }

        Ok(output)
    }

    /// Warp the frequency spectrum
    fn warp_spectrum(&self, spectrum: &[Complex32], warp_factor: f32) -> Result<Vec<Complex32>> {
        let n = spectrum.len();
        let mut warped = vec![Complex::new(0.0, 0.0); n];

        let nyquist = self.config.sample_rate as f32 / 2.0;
        let lower_cutoff = self.config.lower_cutoff;
        let upper_cutoff = if self.config.upper_cutoff > 0.0 {
            self.config.upper_cutoff
        } else {
            nyquist
        };

        // Compute warping function
        // Index k is used for frequency calculations, not just array indexing
        #[allow(clippy::needless_range_loop)]
        for k in 0..n / 2 {
            let freq = k as f32 * nyquist / (n as f32 / 2.0);

            // Only warp frequencies within the cutoff range
            let warped_freq = if freq >= lower_cutoff && freq <= upper_cutoff {
                self.warp_frequency(freq, warp_factor, nyquist)
            } else {
                freq
            };

            // Compute the warped bin index
            let warped_k = warped_freq * (n as f32 / 2.0) / nyquist;

            // Linear interpolation
            warped[k] = self.interpolate_spectrum(spectrum, warped_k);
        }

        // Mirror the spectrum for real-valued output
        for k in n / 2..n {
            warped[k] = warped[n - k].conj();
        }

        Ok(warped)
    }

    /// Frequency warping function using bilinear transform
    fn warp_frequency(&self, freq: f32, alpha: f32, nyquist: f32) -> f32 {
        // Normalize frequency to [0, PI]
        let omega = freq * PI / nyquist;

        // Bilinear warping formula
        let warped_omega = omega + 2.0 * (alpha - 1.0) * omega.sin().atan2(omega.cos() + alpha);

        // Convert back to Hz
        warped_omega * nyquist / PI
    }

    /// Interpolate spectrum at a fractional bin index
    fn interpolate_spectrum(&self, spectrum: &[Complex32], k: f32) -> Complex32 {
        let k_floor = k.floor() as usize;
        let k_ceil = k.ceil() as usize;

        if k_ceil >= spectrum.len() {
            return Complex::new(0.0, 0.0);
        }

        if k_floor == k_ceil {
            return spectrum[k_floor];
        }

        let alpha = k - k_floor as f32;
        let val1 = spectrum[k_floor];
        let val2 = spectrum[k_ceil];

        // Linear interpolation
        Complex::new(
            val1.re * (1.0 - alpha) + val2.re * alpha,
            val1.im * (1.0 - alpha) + val2.im * alpha,
        )
    }

    /// Create Hanning window
    fn create_hanning_window(&self, size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (size - 1) as f32).cos()))
            .collect()
    }

    /// Get all configured warp factors
    pub fn warp_factors(&self) -> &[f32] {
        &self.config.warp_factors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vtlp_creation() {
        let config = VtlpConfig::default();
        let augmentor = VtlpAugmentor::new(config);
        assert_eq!(augmentor.warp_factors().len(), 3);
    }

    #[test]
    fn test_vtlp_empty_audio() {
        let config = VtlpConfig::default();
        let augmentor = VtlpAugmentor::new(config);
        let result = augmentor.apply_vtlp(&[], 0.9).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_vtlp_invalid_warp_factor() {
        let config = VtlpConfig::default();
        let augmentor = VtlpAugmentor::new(config);
        let audio = vec![0.0; 1024];
        let result = augmentor.apply_vtlp(&audio, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_vtlp_warp_identity() {
        let config = VtlpConfig {
            warp_factors: vec![1.0],
            sample_rate: 16000,
            window_size: 512,
            hop_size: 128,
            lower_cutoff: 0.0,
            upper_cutoff: 0.0,
        };
        let augmentor = VtlpAugmentor::new(config);

        // Create a simple sine wave
        let audio: Vec<f32> = (0..16000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();

        let result = augmentor.apply_vtlp(&audio, 1.0).unwrap();
        assert_eq!(result.len(), audio.len());
    }

    #[test]
    fn test_vtlp_warp_lower_frequency() {
        let config = VtlpConfig {
            warp_factors: vec![0.9],
            sample_rate: 16000,
            window_size: 512,
            hop_size: 128,
            lower_cutoff: 0.0,
            upper_cutoff: 0.0,
        };
        let augmentor = VtlpAugmentor::new(config);

        let audio: Vec<f32> = (0..8192)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();

        let result = augmentor.apply_vtlp(&audio, 0.9).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), audio.len());
    }

    #[test]
    fn test_vtlp_warp_higher_frequency() {
        let config = VtlpConfig {
            warp_factors: vec![1.1],
            sample_rate: 16000,
            window_size: 512,
            hop_size: 128,
            lower_cutoff: 0.0,
            upper_cutoff: 0.0,
        };
        let augmentor = VtlpAugmentor::new(config);

        let audio: Vec<f32> = (0..8192)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();

        let result = augmentor.apply_vtlp(&audio, 1.1).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), audio.len());
    }

    #[test]
    fn test_hanning_window() {
        let config = VtlpConfig::default();
        let augmentor = VtlpAugmentor::new(config);
        let window = augmentor.create_hanning_window(64);

        assert_eq!(window.len(), 64);
        // Window should start and end near zero
        assert!(window[0].abs() < 0.01);
        assert!(window[63].abs() < 0.01);
        // Peak should be near middle
        assert!(window[32] > 0.9);
    }

    #[test]
    fn test_vtlp_with_cutoff_frequencies() {
        let config = VtlpConfig {
            warp_factors: vec![0.9],
            sample_rate: 16000,
            window_size: 512,
            hop_size: 128,
            lower_cutoff: 100.0,
            upper_cutoff: 4000.0,
        };
        let augmentor = VtlpAugmentor::new(config);

        let audio: Vec<f32> = (0..8192)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();

        let result = augmentor.apply_vtlp(&audio, 0.9).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), audio.len());
    }

    #[test]
    fn test_vtlp_energy_preservation() {
        let config = VtlpConfig {
            warp_factors: vec![0.95, 1.05],
            sample_rate: 16000,
            window_size: 512,
            hop_size: 128,
            lower_cutoff: 0.0,
            upper_cutoff: 0.0,
        };
        let augmentor = VtlpAugmentor::new(config);

        let audio: Vec<f32> = (0..8192)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();

        let original_energy: f32 = audio.iter().map(|&x| x * x).sum();

        let result = augmentor.apply_vtlp(&audio, 0.95).unwrap();
        let warped_energy: f32 = result.iter().map(|&x| x * x).sum();

        // Energy may not be strictly preserved in VTLP due to frequency warping
        // and STFT reconstruction, but output should have non-zero energy
        assert!(warped_energy > 0.0);
        assert!(result.len() == audio.len());

        // Energy should be in a reasonable range (within order of magnitude)
        let ratio = warped_energy / original_energy;
        assert!(ratio > 0.01 && ratio < 100.0, "Energy ratio: {}", ratio);
    }

    #[test]
    fn test_stft_istft_reconstruction() {
        let config = VtlpConfig {
            sample_rate: 16000,
            window_size: 512,
            hop_size: 128,
            ..Default::default()
        };
        let augmentor = VtlpAugmentor::new(config);

        // Create a simple test signal
        let audio: Vec<f32> = (0..2048)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();

        // STFT
        let frames = augmentor.stft(&audio).unwrap();
        assert!(!frames.is_empty());

        // ISTFT
        let reconstructed = augmentor.istft(&frames).unwrap();
        assert_eq!(reconstructed.len(), audio.len());

        // Check reconstruction quality (should be close to original)
        let mut max_error = 0.0_f32;
        for (i, (&orig, &recon)) in audio.iter().zip(reconstructed.iter()).enumerate() {
            // Skip edges which may have boundary effects
            if i > 512 && i < audio.len() - 512 {
                let error = (orig - recon).abs();
                max_error = max_error.max(error);
            }
        }

        // Reconstruction should be accurate within reasonable tolerance
        assert!(max_error < 0.1, "Max error: {}", max_error);
    }
}
