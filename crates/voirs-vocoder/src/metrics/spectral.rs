//! Spectral analysis and distortion metrics
//!
//! Provides spectral-based quality metrics including mel-cepstral distortion,
//! log spectral distance, and other frequency-domain measures.

use crate::Result;
use ndarray::{s, Array1, Array2};
use realfft::RealFftPlanner;
use std::f32::consts::PI;

/// Spectral analyzer for quality metrics
pub struct SpectralAnalyzer {
    /// Sample rate
    sample_rate: u32,

    /// FFT planner
    fft_planner: RealFftPlanner<f32>,

    /// Frame size for analysis
    frame_size: usize,

    /// Hop length
    hop_length: usize,

    /// Number of mel filter banks
    n_mels: usize,

    /// Number of MFCC coefficients
    n_mfcc: usize,
}

impl SpectralAnalyzer {
    /// Create new spectral analyzer
    pub fn new(sample_rate: u32) -> Self {
        let frame_size = 1024;
        let hop_length = frame_size / 4;

        Self {
            sample_rate,
            fft_planner: RealFftPlanner::new(),
            frame_size,
            hop_length,
            n_mels: 80,
            n_mfcc: 13,
        }
    }

    /// Calculate mel-cepstral distortion (MCD) between two signals
    pub fn calculate_mcd(
        &mut self,
        reference: &Array1<f32>,
        degraded: &Array1<f32>,
    ) -> Result<f32> {
        // Extract MFCC features for both signals
        let ref_mfcc = self.extract_mfcc(reference)?;
        let deg_mfcc = self.extract_mfcc(degraded)?;

        // Calculate frame-wise MCD
        let min_frames = ref_mfcc.nrows().min(deg_mfcc.nrows());
        let mut total_distortion = 0.0;

        for frame in 0..min_frames {
            let ref_frame = ref_mfcc.row(frame);
            let deg_frame = deg_mfcc.row(frame);

            // Calculate Euclidean distance
            let mut frame_distortion = 0.0;
            for (r, d) in ref_frame.iter().zip(deg_frame.iter()) {
                frame_distortion += (r - d).powi(2);
            }

            total_distortion += frame_distortion.sqrt();
        }

        // Convert to MCD in dB
        let mcd = (10.0 / PI.ln()) * (total_distortion / min_frames as f32);

        Ok(mcd)
    }

    /// Calculate log spectral distance (LSD)
    pub fn calculate_lsd(
        &mut self,
        reference: &Array1<f32>,
        degraded: &Array1<f32>,
    ) -> Result<f32> {
        let ref_spectrum = self.compute_log_spectrum(reference)?;
        let deg_spectrum = self.compute_log_spectrum(degraded)?;

        let min_frames = ref_spectrum.nrows().min(deg_spectrum.nrows());
        let mut total_distance = 0.0;

        for frame in 0..min_frames {
            let ref_frame = ref_spectrum.row(frame);
            let deg_frame = deg_spectrum.row(frame);

            let mut frame_distance = 0.0;
            for (r, d) in ref_frame.iter().zip(deg_frame.iter()) {
                frame_distance += (r - d).powi(2);
            }

            total_distance += frame_distance.sqrt();
        }

        Ok(total_distance / min_frames as f32)
    }

    /// Calculate spectral convergence
    pub fn calculate_spectral_convergence(
        &mut self,
        reference: &Array1<f32>,
        degraded: &Array1<f32>,
    ) -> Result<f32> {
        let ref_spectrum = self.compute_power_spectrum(reference)?;
        let deg_spectrum = self.compute_power_spectrum(degraded)?;

        let min_frames = ref_spectrum.nrows().min(deg_spectrum.nrows());
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for frame in 0..min_frames {
            let ref_frame = ref_spectrum.row(frame);
            let deg_frame = deg_spectrum.row(frame);

            for (r, d) in ref_frame.iter().zip(deg_frame.iter()) {
                numerator += (r - d).powi(2);
                denominator += r.powi(2);
            }
        }

        if denominator > 1e-20 {
            Ok((numerator / denominator).sqrt())
        } else {
            Ok(0.0)
        }
    }

    /// Calculate spectral distortion in multiple bands
    pub fn calculate_band_spectral_distortion(
        &mut self,
        reference: &Array1<f32>,
        degraded: &Array1<f32>,
    ) -> Result<Vec<f32>> {
        let ref_spectrum = self.compute_power_spectrum(reference)?;
        let deg_spectrum = self.compute_power_spectrum(degraded)?;

        // Define frequency bands (in bins)
        let nyquist_bin = self.frame_size / 2;
        let bands = vec![
            (1, nyquist_bin / 8),               // Low: 0 - fs/16
            (nyquist_bin / 8, nyquist_bin / 4), // Low-mid: fs/16 - fs/8
            (nyquist_bin / 4, nyquist_bin / 2), // Mid: fs/8 - fs/4
            (nyquist_bin / 2, nyquist_bin),     // High: fs/4 - fs/2
        ];

        let min_frames = ref_spectrum.nrows().min(deg_spectrum.nrows());
        let mut band_distortions = Vec::new();

        for (start_bin, end_bin) in bands {
            let mut band_distortion = 0.0;
            let mut frame_count = 0;

            for frame in 0..min_frames {
                let ref_frame = ref_spectrum.row(frame);
                let deg_frame = deg_spectrum.row(frame);

                let mut frame_band_distortion = 0.0;
                for bin in start_bin..end_bin.min(ref_frame.len()) {
                    let ref_val = ref_frame[bin];
                    let deg_val = deg_frame[bin];

                    if ref_val > 1e-20 && deg_val > 1e-20 {
                        frame_band_distortion += (ref_val.log10() - deg_val.log10()).powi(2);
                    }
                }

                band_distortion += frame_band_distortion.sqrt();
                frame_count += 1;
            }

            if frame_count > 0 {
                band_distortions.push(band_distortion / frame_count as f32);
            } else {
                band_distortions.push(0.0);
            }
        }

        Ok(band_distortions)
    }

    /// Extract MFCC features
    fn extract_mfcc(&mut self, audio: &Array1<f32>) -> Result<Array2<f32>> {
        // First get mel spectrogram
        let mel_spec = self.compute_mel_spectrogram(audio)?;

        // Apply DCT to get cepstral coefficients
        let n_frames = mel_spec.nrows();
        let mut mfcc = Array2::zeros((n_frames, self.n_mfcc));

        for frame in 0..n_frames {
            let mel_frame = mel_spec.row(frame);

            // Apply DCT (simplified)
            for coeff in 0..self.n_mfcc {
                let mut dct_val = 0.0;
                for mel_bin in 0..self.n_mels.min(mel_frame.len()) {
                    let angle = PI * coeff as f32 * (mel_bin as f32 + 0.5) / self.n_mels as f32;
                    dct_val += mel_frame[mel_bin] * angle.cos();
                }

                // Apply normalization
                let norm = if coeff == 0 {
                    (1.0 / self.n_mels as f32).sqrt()
                } else {
                    (2.0 / self.n_mels as f32).sqrt()
                };

                mfcc[[frame, coeff]] = dct_val * norm;
            }
        }

        Ok(mfcc)
    }

    /// Compute mel spectrogram
    fn compute_mel_spectrogram(&mut self, audio: &Array1<f32>) -> Result<Array2<f32>> {
        let power_spectrum = self.compute_power_spectrum(audio)?;
        let mel_filterbank = self.create_mel_filterbank();

        let n_frames = power_spectrum.nrows();
        let mut mel_spec = Array2::zeros((n_frames, self.n_mels));

        for frame in 0..n_frames {
            let power_frame = power_spectrum.row(frame);

            for mel_bin in 0..self.n_mels {
                let mut mel_energy = 0.0;
                for (freq_bin, &filter_val) in mel_filterbank.row(mel_bin).iter().enumerate() {
                    if freq_bin < power_frame.len() {
                        mel_energy += power_frame[freq_bin] * filter_val;
                    }
                }

                // Convert to log scale
                mel_spec[[frame, mel_bin]] = if mel_energy > 1e-20 {
                    mel_energy.log10() * 10.0
                } else {
                    -200.0 // Very quiet
                };
            }
        }

        Ok(mel_spec)
    }

    /// Create mel filterbank
    fn create_mel_filterbank(&self) -> Array2<f32> {
        let n_fft_bins = self.frame_size / 2 + 1;
        let mut filterbank = Array2::zeros((self.n_mels, n_fft_bins));

        // Convert frequency to mel scale
        let mel_low = self.hz_to_mel(0.0);
        let mel_high = self.hz_to_mel(self.sample_rate as f32 / 2.0);

        // Create mel points
        let mel_points: Vec<f32> = (0..=self.n_mels + 1)
            .map(|i| mel_low + (mel_high - mel_low) * i as f32 / (self.n_mels + 1) as f32)
            .collect();

        // Convert mel points back to Hz
        let hz_points: Vec<f32> = mel_points.iter().map(|&mel| self.mel_to_hz(mel)).collect();

        // Convert Hz to FFT bin numbers
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&hz| ((n_fft_bins - 1) as f32 * hz / (self.sample_rate as f32 / 2.0)) as usize)
            .map(|bin| bin.min(n_fft_bins - 1))
            .collect();

        // Create triangular filters
        for mel_bin in 0..self.n_mels {
            let left = bin_points[mel_bin];
            let center = bin_points[mel_bin + 1];
            let right = bin_points[mel_bin + 2];

            // Left slope
            for bin in left..center {
                if center > left {
                    filterbank[[mel_bin, bin]] = (bin - left) as f32 / (center - left) as f32;
                }
            }

            // Right slope
            for bin in center..=right {
                if right > center {
                    filterbank[[mel_bin, bin]] = (right - bin) as f32 / (right - center) as f32;
                }
            }
        }

        filterbank
    }

    /// Convert Hz to mel scale
    fn hz_to_mel(&self, hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    /// Convert mel scale to Hz
    fn mel_to_hz(&self, mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    /// Compute power spectrum for all frames
    fn compute_power_spectrum(&mut self, audio: &Array1<f32>) -> Result<Array2<f32>> {
        let n_frames = (audio.len().saturating_sub(self.frame_size)) / self.hop_length + 1;
        let n_fft_bins = self.frame_size / 2 + 1;
        let mut spectrum = Array2::zeros((n_frames, n_fft_bins));

        let fft = self.fft_planner.plan_fft_forward(self.frame_size);
        let mut input = vec![0.0; self.frame_size];
        let mut output = fft.make_output_vec();

        // Hanning window
        let window: Vec<f32> = (0..self.frame_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (self.frame_size - 1) as f32).cos()))
            .collect();

        for frame in 0..n_frames {
            let start = frame * self.hop_length;
            let end = (start + self.frame_size).min(audio.len());

            // Clear input
            input.fill(0.0);

            // Copy audio with windowing
            for (i, &sample) in audio.slice(s![start..end]).iter().enumerate() {
                input[i] = sample * window[i];
            }

            // Compute FFT
            fft.process(&mut input, &mut output).unwrap();

            // Convert to power spectrum
            for (i, complex_val) in output.iter().enumerate() {
                spectrum[[frame, i]] = complex_val.norm_sqr();
            }
        }

        Ok(spectrum)
    }

    /// Compute log power spectrum
    fn compute_log_spectrum(&mut self, audio: &Array1<f32>) -> Result<Array2<f32>> {
        let power_spectrum = self.compute_power_spectrum(audio)?;

        // Convert to log scale
        let log_spectrum =
            power_spectrum.mapv(|x| if x > 1e-20 { x.log10() * 10.0 } else { -200.0 });

        Ok(log_spectrum)
    }
}

/// Calculate mel-cepstral distortion
pub fn calculate_mcd(
    reference: &Array1<f32>,
    degraded: &Array1<f32>,
    sample_rate: u32,
) -> Result<f32> {
    let mut analyzer = SpectralAnalyzer::new(sample_rate);
    analyzer.calculate_mcd(reference, degraded)
}

/// Calculate log spectral distance
pub fn calculate_lsd(
    reference: &Array1<f32>,
    degraded: &Array1<f32>,
    sample_rate: u32,
) -> Result<f32> {
    let mut analyzer = SpectralAnalyzer::new(sample_rate);
    analyzer.calculate_lsd(reference, degraded)
}

/// Calculate spectral convergence
pub fn calculate_spectral_convergence(
    reference: &Array1<f32>,
    degraded: &Array1<f32>,
    sample_rate: u32,
) -> Result<f32> {
    let mut analyzer = SpectralAnalyzer::new(sample_rate);
    analyzer.calculate_spectral_convergence(reference, degraded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use std::f32::consts::PI;

    #[test]
    fn test_spectral_analyzer_creation() {
        let analyzer = SpectralAnalyzer::new(22050);
        assert_eq!(analyzer.sample_rate, 22050);
        assert_eq!(analyzer.n_mels, 80);
        assert_eq!(analyzer.n_mfcc, 13);
    }

    #[test]
    fn test_mel_scale_conversion() {
        let analyzer = SpectralAnalyzer::new(22050);

        // Test known conversions
        let hz = 1000.0;
        let mel = analyzer.hz_to_mel(hz);
        let hz_back = analyzer.mel_to_hz(mel);

        assert!((hz - hz_back).abs() < 1e-3);
    }

    #[test]
    fn test_mcd_perfect_signal() {
        let mut analyzer = SpectralAnalyzer::new(22050);

        // Generate test signal
        let samples: Vec<f32> = (0..2048)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 22050.0).sin())
            .collect();
        let signal = Array1::from_vec(samples);

        let mcd = analyzer.calculate_mcd(&signal, &signal).unwrap();

        // Perfect reconstruction should have very low MCD
        assert!(mcd < 1.0);
    }

    #[test]
    fn test_lsd_calculation() {
        let mut analyzer = SpectralAnalyzer::new(22050);

        let samples1: Vec<f32> = (0..2048)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 22050.0).sin())
            .collect();
        let samples2: Vec<f32> = (0..2048)
            .map(|i| (2.0 * PI * 880.0 * i as f32 / 22050.0).sin())
            .collect();

        let signal1 = Array1::from_vec(samples1);
        let signal2 = Array1::from_vec(samples2);

        let lsd = analyzer.calculate_lsd(&signal1, &signal2).unwrap();

        // Different frequency signals should have non-zero LSD
        assert!(lsd > 0.0);
        assert!(lsd.is_finite());
    }

    #[test]
    fn test_spectral_convergence() {
        let mut analyzer = SpectralAnalyzer::new(22050);

        let reference: Vec<f32> = (0..2048)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 22050.0).sin())
            .collect();
        let degraded: Vec<f32> = (0..2048)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 22050.0).sin() * 0.8) // Scaled version
            .collect();

        let ref_signal = Array1::from_vec(reference);
        let deg_signal = Array1::from_vec(degraded);

        let convergence = analyzer
            .calculate_spectral_convergence(&ref_signal, &deg_signal)
            .unwrap();

        // Should have some convergence error for scaled signal
        assert!(convergence > 0.0);
        assert!(convergence < 1.0);
    }

    #[test]
    fn test_mel_filterbank_creation() {
        let analyzer = SpectralAnalyzer::new(22050);
        let filterbank = analyzer.create_mel_filterbank();

        assert_eq!(filterbank.nrows(), analyzer.n_mels);
        assert_eq!(filterbank.ncols(), analyzer.frame_size / 2 + 1);

        // Check that filters are non-negative
        for value in filterbank.iter() {
            assert!(*value >= 0.0);
        }

        // Check that filters have reasonable magnitude
        let max_val = filterbank.iter().fold(0.0f32, |acc, &x| acc.max(x));
        assert!(max_val <= 1.0);
    }

    #[test]
    fn test_band_spectral_distortion() {
        let mut analyzer = SpectralAnalyzer::new(22050);

        let samples1: Vec<f32> = (0..2048)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 22050.0).sin())
            .collect();
        let samples2: Vec<f32> = (0..2048)
            .map(|i| {
                (2.0 * PI * 440.0 * i as f32 / 22050.0).sin()
                    + 0.1 * (2.0 * PI * 1760.0 * i as f32 / 22050.0).sin()
            })
            .collect();

        let signal1 = Array1::from_vec(samples1);
        let signal2 = Array1::from_vec(samples2);

        let band_distortions = analyzer
            .calculate_band_spectral_distortion(&signal1, &signal2)
            .unwrap();

        // Should have 4 frequency bands
        assert_eq!(band_distortions.len(), 4);

        // All distortions should be finite and non-negative
        for distortion in &band_distortions {
            assert!(distortion.is_finite());
            assert!(*distortion >= 0.0);
        }
    }
}
