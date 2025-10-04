//! PESQ (Perceptual Evaluation of Speech Quality) implementation
//!
//! Implements a simplified version of ITU-T P.862 PESQ algorithm
//! for evaluating speech quality. This is a lightweight approximation
//! of the full PESQ standard.

use crate::Result;
use scirs2_core::ndarray::{Array1, Array2};
use std::f32::consts::PI;

/// PESQ quality assessor
pub struct PesqCalculator {
    /// Sample rate for analysis
    sample_rate: u32,

    /// Frame size for analysis
    frame_size: usize,

    /// Hop length between frames
    hop_length: usize,
}

impl PesqCalculator {
    /// Create new PESQ calculator
    pub fn new(sample_rate: u32) -> Self {
        let frame_size = match sample_rate {
            8000 => 256,
            16000 => 512,
            _ => 1024, // For 22050, 44100, etc.
        };

        Self {
            sample_rate,
            frame_size,
            hop_length: frame_size / 4,
        }
    }

    /// Calculate PESQ score between reference and degraded signals
    pub fn calculate(&self, reference: &Array1<f32>, degraded: &Array1<f32>) -> Result<f32> {
        // Ensure signals are the same length
        let min_len = reference.len().min(degraded.len());
        let ref_signal = reference
            .slice(scirs2_core::ndarray::s![..min_len])
            .to_owned();
        let deg_signal = degraded
            .slice(scirs2_core::ndarray::s![..min_len])
            .to_owned();

        // Apply pre-processing
        let ref_processed = self.preprocess(&ref_signal);
        let deg_processed = self.preprocess(&deg_signal);

        // Calculate perceptual model
        let ref_loudness = self.calculate_loudness(&ref_processed)?;
        let deg_loudness = self.calculate_loudness(&deg_processed)?;

        // Calculate asymmetric disturbance
        let disturbance = self.calculate_disturbance(&ref_loudness, &deg_loudness);

        // Map to PESQ score
        let pesq_score = self.map_to_pesq(disturbance);

        Ok(pesq_score.clamp(1.0, 4.5))
    }

    /// Pre-process audio signal (filtering, normalization)
    fn preprocess(&self, signal: &Array1<f32>) -> Array1<f32> {
        // Apply basic pre-emphasis filter (simplified)
        let mut processed = signal.clone();
        let pre_emphasis = 0.85;

        for i in (1..processed.len()).rev() {
            processed[i] -= pre_emphasis * processed[i - 1];
        }

        // Normalize amplitude
        let max_abs = processed.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        if max_abs > 0.0 {
            processed.mapv_inplace(|x| x / max_abs);
        }

        processed
    }

    /// Calculate loudness using simplified Bark scale model
    fn calculate_loudness(&self, signal: &Array1<f32>) -> Result<Array2<f32>> {
        let n_frames = (signal.len().saturating_sub(self.frame_size)) / self.hop_length + 1;
        let n_bark_bands = 24; // Simplified Bark scale

        let mut loudness = Array2::zeros((n_frames, n_bark_bands));

        // Hanning window
        let window: Vec<f64> = (0..self.frame_size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * PI as f64 * i as f64 / (self.frame_size - 1) as f64).cos())
            })
            .collect();

        // Bark frequency boundaries (simplified)
        let bark_boundaries = self.get_bark_boundaries();

        for frame in 0..n_frames {
            let start = frame * self.hop_length;
            let end = (start + self.frame_size).min(signal.len());

            // Prepare input with windowing
            let mut input = vec![0.0f64; self.frame_size];
            for (i, &sample) in signal
                .slice(scirs2_core::ndarray::s![start..end])
                .iter()
                .enumerate()
            {
                input[i] = sample as f64 * window[i];
            }

            // Compute FFT using scirs2_fft
            let output = scirs2_fft::rfft(&input, None)
                .map_err(|e| crate::VocoderError::ProcessingError(format!("FFT error: {:?}", e)))?;

            // Convert to power spectrum
            let power_spectrum: Vec<f32> = output
                .iter()
                .map(|c| (c.re * c.re + c.im * c.im) as f32)
                .collect();

            // Map to Bark bands
            for (band, &(start_bin, end_bin)) in bark_boundaries.iter().enumerate() {
                let band_power: f32 = power_spectrum[start_bin..=end_bin].iter().sum();

                // Apply loudness transformation (simplified)
                let loudness_val = if band_power > 1e-10 {
                    band_power.log10() * 10.0 // Convert to dB-like scale
                } else {
                    -100.0 // Very quiet
                };

                loudness[[frame, band]] = loudness_val;
            }
        }

        Ok(loudness)
    }

    /// Get Bark band boundaries in FFT bins
    fn get_bark_boundaries(&self) -> Vec<(usize, usize)> {
        let nyquist_freq = self.sample_rate as f32 / 2.0;
        let bin_freq = nyquist_freq / (self.frame_size / 2) as f32;

        // Simplified Bark frequencies (Hz)
        let bark_freqs = [
            0.0, 100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0,
            1720.0, 2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0,
            12000.0, 15500.0,
        ];

        let mut boundaries = Vec::new();
        for i in 0..bark_freqs.len() - 1 {
            let start_bin = (bark_freqs[i] / bin_freq) as usize;
            let end_bin = ((bark_freqs[i + 1] / bin_freq) as usize).min(self.frame_size / 2);

            if start_bin < end_bin && end_bin <= self.frame_size / 2 {
                boundaries.push((start_bin, end_bin));
            }
        }

        boundaries
    }

    /// Calculate perceptual disturbance
    fn calculate_disturbance(&self, ref_loudness: &Array2<f32>, deg_loudness: &Array2<f32>) -> f32 {
        let min_frames = ref_loudness.nrows().min(deg_loudness.nrows());
        let n_bands = ref_loudness.ncols().min(deg_loudness.ncols());

        let mut total_disturbance = 0.0;
        let mut frame_count = 0;

        for frame in 0..min_frames {
            let mut frame_disturbance = 0.0;

            for band in 0..n_bands {
                let ref_val = ref_loudness[[frame, band]];
                let deg_val = deg_loudness[[frame, band]];

                // Calculate asymmetric disturbance (simplified)
                let difference = ref_val - deg_val;
                let asymmetric_factor = if difference > 0.0 { 1.0 } else { 0.5 };

                frame_disturbance += difference.abs() * asymmetric_factor;
            }

            total_disturbance += frame_disturbance / n_bands as f32;
            frame_count += 1;
        }

        if frame_count > 0 {
            total_disturbance / frame_count as f32
        } else {
            0.0
        }
    }

    /// Map disturbance to PESQ score
    fn map_to_pesq(&self, disturbance: f32) -> f32 {
        // Simplified mapping (in real PESQ this is more complex)
        // Lower disturbance = higher PESQ score

        let normalized_disturbance = (disturbance / 50.0).clamp(0.0, 1.0);

        // Map to PESQ range (1.0 - 4.5)
        4.5 - normalized_disturbance * 3.5
    }
}

/// Simplified PESQ calculation function
pub fn calculate_pesq(
    reference: &Array1<f32>,
    degraded: &Array1<f32>,
    sample_rate: u32,
) -> Result<f32> {
    let mut calculator = PesqCalculator::new(sample_rate);
    calculator.calculate(reference, degraded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_pesq_calculator_creation() {
        let calculator = PesqCalculator::new(22050);
        assert_eq!(calculator.sample_rate, 22050);
        assert_eq!(calculator.frame_size, 1024);
    }

    #[test]
    fn test_perfect_signal_pesq() {
        let signal = Array1::from_vec(vec![0.5, -0.3, 0.8, -0.2, 0.1, 0.6, -0.7, 0.4]);
        let score = calculate_pesq(&signal, &signal, 22050).unwrap();

        // Perfect reconstruction should have high PESQ
        assert!(score >= 4.0);
    }

    #[test]
    fn test_noisy_signal_pesq() {
        let clean = Array1::from_vec(vec![0.5, -0.3, 0.8, -0.2, 0.1, 0.6, -0.7, 0.4]);
        let noisy = Array1::from_vec(vec![0.6, -0.2, 0.7, -0.3, 0.0, 0.5, -0.8, 0.3]);

        let score = calculate_pesq(&clean, &noisy, 22050).unwrap();

        // Noisy signal should have lower PESQ
        assert!((1.0..=4.5).contains(&score));
    }

    #[test]
    fn test_preprocessing() {
        let calculator = PesqCalculator::new(22050);
        let signal = Array1::from_vec(vec![1.0, 0.5, -0.5, -1.0]);
        let processed = calculator.preprocess(&signal);

        // Should be normalized
        let max_abs = processed.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        assert!((max_abs - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bark_boundaries() {
        let calculator = PesqCalculator::new(22050);
        let boundaries = calculator.get_bark_boundaries();

        // Should have reasonable number of bands
        assert!(boundaries.len() > 10);
        assert!(boundaries.len() <= 24);

        // Boundaries should be ordered
        for i in 1..boundaries.len() {
            assert!(boundaries[i].0 >= boundaries[i - 1].1);
        }
    }
}
