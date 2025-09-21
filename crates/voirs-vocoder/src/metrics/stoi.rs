//! STOI (Short-Time Objective Intelligibility) implementation
//!
//! Implements a simplified version of the STOI algorithm for measuring
//! speech intelligibility. STOI correlates well with subjective
//! intelligibility ratings.

use crate::Result;
use ndarray::{s, Array1, Array2};
use realfft::RealFftPlanner;
use std::f32::consts::PI;

/// STOI intelligibility assessor
pub struct StoiCalculator {
    /// Sample rate for analysis
    sample_rate: u32,

    /// FFT planner
    fft_planner: RealFftPlanner<f32>,

    /// Frame length in samples
    frame_length: usize,

    /// Overlap between frames
    overlap: usize,

    /// Number of one-third octave bands
    n_bands: usize,

    /// Band center frequencies
    center_frequencies: Vec<f32>,
}

impl StoiCalculator {
    /// Create new STOI calculator
    pub fn new(sample_rate: u32) -> Self {
        let frame_length = (0.032 * sample_rate as f32) as usize; // 32ms frames
        let overlap = frame_length / 2; // 50% overlap

        let (center_frequencies, n_bands) = Self::generate_center_frequencies(sample_rate);

        Self {
            sample_rate,
            fft_planner: RealFftPlanner::new(),
            frame_length,
            overlap,
            n_bands,
            center_frequencies,
        }
    }

    /// Calculate STOI score between clean and degraded signals
    pub fn calculate(&mut self, clean: &Array1<f32>, degraded: &Array1<f32>) -> Result<f32> {
        // Ensure signals are the same length
        let min_len = clean.len().min(degraded.len());
        let clean_signal = clean.slice(s![..min_len]).to_owned();
        let degraded_signal = degraded.slice(s![..min_len]).to_owned();

        // Remove DC component
        let clean_processed = self.remove_dc(&clean_signal);
        let degraded_processed = self.remove_dc(&degraded_signal);

        // Apply one-third octave band filterbank
        let clean_bands = self.apply_filterbank(&clean_processed)?;
        let degraded_bands = self.apply_filterbank(&degraded_processed)?;

        // Calculate intermediate intelligibility measure
        let d_values = self.calculate_d_values(&clean_bands, &degraded_bands)?;

        // Calculate final STOI score
        let stoi_score = d_values.mean().unwrap_or(0.0);

        Ok(stoi_score.clamp(0.0, 1.0))
    }

    /// Generate one-third octave band center frequencies
    fn generate_center_frequencies(sample_rate: u32) -> (Vec<f32>, usize) {
        let nyquist = sample_rate as f32 / 2.0;
        let mut frequencies = Vec::new();

        // One-third octave bands from ~150 Hz to ~4 kHz (intelligibility range)
        let start_freq = 150.0;
        let mut freq = start_freq;

        while freq < nyquist && freq < 4000.0 {
            frequencies.push(freq);
            freq *= 2.0_f32.powf(1.0 / 3.0); // One-third octave step
        }

        let n_bands = frequencies.len();
        (frequencies, n_bands)
    }

    /// Remove DC component from signal
    fn remove_dc(&self, signal: &Array1<f32>) -> Array1<f32> {
        let mean = signal.mean().unwrap_or(0.0);
        signal.mapv(|x| x - mean)
    }

    /// Apply one-third octave band filterbank
    fn apply_filterbank(&mut self, signal: &Array1<f32>) -> Result<Array2<f32>> {
        // Check if signal is long enough for analysis
        if signal.len() <= self.overlap {
            return Ok(Array2::zeros((self.n_bands, 0)));
        }

        let n_frames = (signal.len() - self.overlap) / (self.frame_length - self.overlap);
        let mut band_signals = Array2::zeros((self.n_bands, n_frames));

        // Clone center frequencies to avoid borrowing issues
        let center_frequencies = self.center_frequencies.clone();

        // For each band, apply bandpass filter and frame
        for (band_idx, center_freq) in center_frequencies.iter().enumerate() {
            let filtered = self.bandpass_filter(signal, *center_freq);
            let framed = self.frame_signal(&filtered);

            let min_frames = framed.len().min(n_frames);
            for frame in 0..min_frames {
                band_signals[[band_idx, frame]] = framed[frame];
            }
        }

        Ok(band_signals)
    }

    /// Apply simplified bandpass filter around center frequency
    fn bandpass_filter(&mut self, signal: &Array1<f32>, center_freq: f32) -> Array1<f32> {
        // Simplified bandpass using FFT filtering
        let n = signal.len();
        let next_power_of_2 = n.next_power_of_two();

        let fft = self.fft_planner.plan_fft_forward(next_power_of_2);
        let ifft = self.fft_planner.plan_fft_inverse(next_power_of_2);

        let mut input = vec![0.0; next_power_of_2];
        input[..n].copy_from_slice(signal.as_slice().unwrap());

        let mut spectrum = fft.make_output_vec();
        fft.process(&mut input, &mut spectrum).unwrap();

        // Apply bandpass filter in frequency domain
        let freq_resolution = self.sample_rate as f32 / next_power_of_2 as f32;
        let bandwidth = center_freq * 0.5; // One-third octave approximation

        for (i, spec_val) in spectrum.iter_mut().enumerate() {
            let freq = i as f32 * freq_resolution;
            let distance = (freq - center_freq).abs();

            if distance > bandwidth {
                *spec_val *= 0.1; // Attenuate out-of-band
            }
        }

        // Convert back to time domain
        let mut output = vec![0.0; next_power_of_2];
        ifft.process(&mut spectrum, &mut output).unwrap();

        // Normalize and extract original length
        let scale = 1.0 / next_power_of_2 as f32;
        Array1::from_vec(output[..n].iter().map(|&x| x * scale).collect())
    }

    /// Frame signal into overlapping windows with RMS calculation
    fn frame_signal(&self, signal: &Array1<f32>) -> Vec<f32> {
        let hop_length = self.frame_length - self.overlap;
        let n_frames = (signal.len() - self.overlap) / hop_length;
        let mut frames = Vec::with_capacity(n_frames);

        // Hanning window
        let window: Vec<f32> = (0..self.frame_length)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (self.frame_length - 1) as f32).cos()))
            .collect();

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;
            let end = start + self.frame_length;

            if end <= signal.len() {
                // Calculate RMS of windowed frame
                let mut rms = 0.0;
                for (i, &sample) in signal.slice(s![start..end]).iter().enumerate() {
                    let windowed = sample * window[i];
                    rms += windowed * windowed;
                }
                rms = (rms / self.frame_length as f32).sqrt();
                frames.push(rms);
            }
        }

        frames
    }

    /// Calculate intermediate intelligibility measure (d values)
    fn calculate_d_values(
        &self,
        clean_bands: &Array2<f32>,
        degraded_bands: &Array2<f32>,
    ) -> Result<Array1<f32>> {
        let n_bands = clean_bands.nrows().min(degraded_bands.nrows());
        let mut d_values = Array1::zeros(n_bands);

        for band in 0..n_bands {
            let clean_band = clean_bands.row(band);
            let degraded_band = degraded_bands.row(band);

            // Calculate correlation coefficient
            let correlation =
                self.calculate_correlation(&clean_band.to_owned(), &degraded_band.to_owned());

            // Apply clipping and normalization
            let d_value = correlation.max(0.0); // Negative correlations don't contribute
            d_values[band] = d_value;
        }

        Ok(d_values)
    }

    /// Calculate correlation coefficient between two signals
    fn calculate_correlation(&self, x: &Array1<f32>, y: &Array1<f32>) -> f32 {
        let min_len = x.len().min(y.len());
        if min_len == 0 {
            return 0.0;
        }

        let x_slice = x.slice(s![..min_len]);
        let y_slice = y.slice(s![..min_len]);

        // Calculate means
        let mean_x = x_slice.mean().unwrap_or(0.0);
        let mean_y = y_slice.mean().unwrap_or(0.0);

        // Calculate correlation coefficient
        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for (xi, yi) in x_slice.iter().zip(y_slice.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;

            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        }
    }
}

/// Simplified STOI calculation function
pub fn calculate_stoi(
    clean: &Array1<f32>,
    degraded: &Array1<f32>,
    sample_rate: u32,
) -> Result<f32> {
    let mut calculator = StoiCalculator::new(sample_rate);
    calculator.calculate(clean, degraded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_stoi_calculator_creation() {
        let calculator = StoiCalculator::new(22050);
        assert_eq!(calculator.sample_rate, 22050);
        assert!(calculator.n_bands > 0);
    }

    #[test]
    fn test_perfect_signal_stoi() {
        // Create a longer signal appropriate for STOI analysis (1 second)
        let signal = Array1::from_vec((0..22050).map(|i| (i as f32 * 0.01).sin()).collect());
        let score = calculate_stoi(&signal, &signal, 22050).unwrap();

        // Perfect reconstruction should have high STOI
        tracing::debug!("STOI score for perfect signal: {score}");
        assert!(
            score >= 0.8,
            "STOI score {score} is below expected threshold"
        );
    }

    #[test]
    fn test_noisy_signal_stoi() {
        // Create longer signals for proper STOI analysis
        let clean = Array1::from_vec((0..22050).map(|i| (i as f32 * 0.01).sin()).collect());
        let noisy = Array1::from_vec(
            (0..22050)
                .map(|i| (i as f32 * 0.01).sin() + 0.5 * (i as f32 * 0.05).sin())
                .collect(),
        );

        let score = calculate_stoi(&clean, &noisy, 22050).unwrap();

        // Noisy signal should have lower STOI than perfect
        assert!((0.0..=1.0).contains(&score));
        assert!(score < 0.99); // Should be less than perfect (relaxed threshold)
    }

    #[test]
    fn test_dc_removal() {
        let calculator = StoiCalculator::new(22050);
        let signal_with_dc = Array1::from_vec(vec![1.0, 1.5, 0.5, 1.2, 0.8]);
        let no_dc = calculator.remove_dc(&signal_with_dc);

        // Mean should be approximately zero
        let mean = no_dc.mean().unwrap();
        assert!(mean.abs() < 1e-6);
    }

    #[test]
    fn test_center_frequencies() {
        let (freqs, n_bands) = StoiCalculator::generate_center_frequencies(22050);

        // Should have reasonable number of bands
        assert!(n_bands > 5);
        assert!(n_bands < 50);

        // Frequencies should be increasing
        for i in 1..freqs.len() {
            assert!(freqs[i] > freqs[i - 1]);
        }

        // Should be in intelligibility range
        assert!(freqs[0] >= 100.0);
        assert!(freqs[freqs.len() - 1] <= 5000.0);
    }

    #[test]
    fn test_correlation_calculation() {
        let calculator = StoiCalculator::new(22050);

        // Perfect correlation
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let y = x.clone();
        let corr = calculator.calculate_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-6);

        // No correlation (orthogonal)
        let x = Array1::from_vec(vec![1.0, -1.0, 1.0, -1.0]);
        let y = Array1::from_vec(vec![1.0, 1.0, -1.0, -1.0]);
        let corr = calculator.calculate_correlation(&x, &y);
        assert!(corr.abs() < 1e-6);
    }
}
