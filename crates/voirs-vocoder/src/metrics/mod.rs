//! Audio quality metrics for vocoder evaluation
//!
//! Provides objective and perceptual quality metrics including:
//! - PESQ (Perceptual Evaluation of Speech Quality)
//! - STOI (Short-Time Objective Intelligibility)
//! - SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
//! - MOS prediction models
//! - Traditional metrics (SNR, THD+N, spectral distortion)

use crate::{AudioBuffer, Result, VocoderError};
use scirs2_core::ndarray::{s, Array1, Array2};
use std::f32::consts::PI;

pub mod mos;
pub mod pesq;
pub mod si_sdr;
pub mod spectral;
pub mod stoi;

/// Comprehensive quality assessment results
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// PESQ score (1.0-4.5, higher is better)
    pub pesq: Option<f32>,

    /// STOI score (0.0-1.0, higher is better)
    pub stoi: Option<f32>,

    /// SI-SDR in dB (higher is better)
    pub si_sdr: Option<f32>,

    /// Predicted MOS score (1.0-5.0, higher is better)
    pub mos_prediction: Option<f32>,

    /// Signal-to-noise ratio in dB
    pub snr: f32,

    /// Total harmonic distortion + noise (%)
    pub thd_n: f32,

    /// Log spectral distance (lower is better)
    pub lsd: f32,

    /// Mel-cepstral distortion (lower is better)
    pub mcd: Option<f32>,

    /// Peak signal-to-noise ratio in dB
    pub psnr: f32,

    /// Spectral convergence (lower is better)
    pub spectral_convergence: f32,

    /// Mean opinion score estimate (1.0-5.0)
    pub mos_estimate: f32,
}

/// Quality metrics calculator
pub struct QualityCalculator {
    /// Configuration options
    config: QualityConfig,
}

/// Configuration for quality metric calculation
#[derive(Debug, Clone)]
pub struct QualityConfig {
    /// Calculate expensive metrics (PESQ, STOI)
    pub include_expensive: bool,

    /// Calculate MOS prediction
    pub include_mos_prediction: bool,

    /// Sample rate for analysis
    pub sample_rate: u32,

    /// Frame size for analysis (samples)
    pub frame_size: usize,

    /// Hop length between frames
    pub hop_length: usize,

    /// Frequency weighting for perceptual metrics
    pub frequency_weighting: bool,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            include_expensive: true,
            include_mos_prediction: true,
            sample_rate: 22050,
            frame_size: 1024,
            hop_length: 256,
            frequency_weighting: true,
        }
    }
}

impl QualityCalculator {
    /// Create new quality calculator
    pub fn new(config: QualityConfig) -> Self {
        Self { config }
    }

    /// Calculate comprehensive quality metrics between reference and degraded audio
    pub fn calculate_metrics(
        &mut self,
        reference: &AudioBuffer,
        degraded: &AudioBuffer,
    ) -> Result<QualityMetrics> {
        // Validate inputs
        self.validate_inputs(reference, degraded)?;

        // Ensure same length and convert to mono if needed
        let (ref_mono, deg_mono) = self.prepare_audio(reference, degraded)?;

        // Calculate basic metrics
        let snr = self.calculate_snr(&ref_mono, &deg_mono);
        let thd_n = self.calculate_thd_n(&deg_mono);
        let psnr = self.calculate_psnr(&ref_mono, &deg_mono);

        // Calculate spectral metrics
        let lsd = self.calculate_lsd(&ref_mono, &deg_mono)?;
        let spectral_convergence = self.calculate_spectral_convergence(&ref_mono, &deg_mono)?;

        // Calculate expensive metrics if enabled
        let pesq = if self.config.include_expensive {
            Some(pesq::calculate_pesq(
                &ref_mono,
                &deg_mono,
                self.config.sample_rate,
            )?)
        } else {
            None
        };

        let stoi = if self.config.include_expensive {
            Some(stoi::calculate_stoi(
                &ref_mono,
                &deg_mono,
                self.config.sample_rate,
            )?)
        } else {
            None
        };

        let si_sdr = if self.config.include_expensive {
            Some(si_sdr::calculate_si_sdr(&ref_mono, &deg_mono)?)
        } else {
            None
        };

        // Calculate MOS prediction if enabled
        let mos_prediction = if self.config.include_mos_prediction {
            Some(mos::predict_mos(&deg_mono, self.config.sample_rate)?)
        } else {
            None
        };

        // Calculate mel-cepstral distortion if requested
        let mcd = if self.config.include_expensive {
            Some(spectral::calculate_mcd(
                &ref_mono,
                &deg_mono,
                self.config.sample_rate,
            )?)
        } else {
            None
        };

        // Estimate overall MOS based on available metrics
        let mos_estimate = self.estimate_mos(snr, thd_n, lsd, pesq, stoi);

        Ok(QualityMetrics {
            pesq,
            stoi,
            si_sdr,
            mos_prediction,
            snr,
            thd_n,
            lsd,
            mcd,
            psnr,
            spectral_convergence,
            mos_estimate,
        })
    }

    /// Validate input audio buffers
    fn validate_inputs(&self, reference: &AudioBuffer, degraded: &AudioBuffer) -> Result<()> {
        if reference.sample_rate() != degraded.sample_rate() {
            return Err(VocoderError::InputError(
                "Sample rates must match between reference and degraded audio".to_string(),
            ));
        }

        if reference.samples().is_empty() || degraded.samples().is_empty() {
            return Err(VocoderError::InputError(
                "Audio buffers cannot be empty".to_string(),
            ));
        }

        Ok(())
    }

    /// Prepare audio for analysis (convert to mono, align lengths)
    fn prepare_audio(
        &self,
        reference: &AudioBuffer,
        degraded: &AudioBuffer,
    ) -> Result<(Array1<f32>, Array1<f32>)> {
        // Convert to mono
        let ref_mono = self.to_mono(reference);
        let deg_mono = self.to_mono(degraded);

        // Align lengths (truncate to shorter)
        let min_len = ref_mono.len().min(deg_mono.len());
        let ref_aligned = ref_mono.slice(s![..min_len]).to_owned();
        let deg_aligned = deg_mono.slice(s![..min_len]).to_owned();

        Ok((ref_aligned, deg_aligned))
    }

    /// Convert audio buffer to mono
    fn to_mono(&self, audio: &AudioBuffer) -> Array1<f32> {
        let samples = audio.samples();
        let channels = audio.channels() as usize;

        if channels == 1 {
            Array1::from_vec(samples.to_vec())
        } else {
            // Mix down to mono by averaging channels
            let mono_len = samples.len() / channels;
            let mut mono = Array1::zeros(mono_len);

            for i in 0..mono_len {
                let mut sum = 0.0;
                for ch in 0..channels {
                    sum += samples[i * channels + ch];
                }
                mono[i] = sum / channels as f32;
            }

            mono
        }
    }

    /// Calculate signal-to-noise ratio
    fn calculate_snr(&self, reference: &Array1<f32>, degraded: &Array1<f32>) -> f32 {
        let signal_power: f32 = reference.iter().map(|&x| x * x).sum();
        let noise_power: f32 = reference
            .iter()
            .zip(degraded.iter())
            .map(|(&r, &d)| (r - d).powi(2))
            .sum();

        if noise_power > 0.0 && signal_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else if noise_power == 0.0 {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        }
    }

    /// Calculate total harmonic distortion plus noise
    fn calculate_thd_n(&self, audio: &Array1<f32>) -> f32 {
        // Simplified THD+N calculation
        // In practice, this would require more sophisticated harmonic analysis

        // Calculate RMS of the signal
        let _rms: f32 = (audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32).sqrt();

        // Estimate distortion using high-frequency content as a proxy
        let fft_size = self.config.frame_size;
        if audio.len() < fft_size {
            return 0.0;
        }

        // Use first frame for analysis
        let audio_slice = audio.slice(s![..fft_size]);
        let input: Vec<f64> = audio_slice.iter().map(|&x| x as f64).collect();

        // Compute FFT using scirs2_fft
        let output = match scirs2_fft::rfft(&input, None) {
            Ok(spectrum) => spectrum,
            Err(_) => return 0.0,
        };

        // Calculate energy in high frequencies (rough distortion estimate)
        let nyquist_bin = fft_size / 2;
        let high_freq_start = nyquist_bin / 2; // Above 1/4 Nyquist

        let high_freq_energy: f32 = output[high_freq_start..nyquist_bin.min(output.len())]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im) as f32)
            .sum();

        let total_energy: f32 = output[1..nyquist_bin.min(output.len())]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im) as f32)
            .sum();

        if total_energy > 0.0 {
            (high_freq_energy / total_energy * 100.0).min(100.0)
        } else {
            0.0
        }
    }

    /// Calculate peak signal-to-noise ratio
    fn calculate_psnr(&self, reference: &Array1<f32>, degraded: &Array1<f32>) -> f32 {
        let max_val = reference.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        let mse: f32 = reference
            .iter()
            .zip(degraded.iter())
            .map(|(&r, &d)| (r - d).powi(2))
            .sum::<f32>()
            / reference.len() as f32;

        if mse > 0.0 && max_val > 0.0 {
            20.0 * (max_val / mse.sqrt()).log10()
        } else if mse == 0.0 {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        }
    }

    /// Calculate log spectral distance
    fn calculate_lsd(&mut self, reference: &Array1<f32>, degraded: &Array1<f32>) -> Result<f32> {
        let ref_spectrum = self.compute_spectrum(reference)?;
        let deg_spectrum = self.compute_spectrum(degraded)?;

        let min_frames = ref_spectrum.nrows().min(deg_spectrum.nrows());
        let mut lsd_sum = 0.0;

        for frame in 0..min_frames {
            let ref_frame = ref_spectrum.row(frame);
            let deg_frame = deg_spectrum.row(frame);

            let mut frame_lsd = 0.0;
            for (r, d) in ref_frame.iter().zip(deg_frame.iter()) {
                if *r > 1e-10 && *d > 1e-10 {
                    frame_lsd += (r.log10() - d.log10()).powi(2);
                }
            }

            lsd_sum += frame_lsd.sqrt();
        }

        Ok(lsd_sum / min_frames as f32)
    }

    /// Calculate spectral convergence
    fn calculate_spectral_convergence(
        &mut self,
        reference: &Array1<f32>,
        degraded: &Array1<f32>,
    ) -> Result<f32> {
        let ref_spectrum = self.compute_spectrum(reference)?;
        let deg_spectrum = self.compute_spectrum(degraded)?;

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

        if denominator > 0.0 {
            Ok((numerator / denominator).sqrt())
        } else {
            Ok(0.0)
        }
    }

    /// Compute power spectrum
    fn compute_spectrum(&self, audio: &Array1<f32>) -> Result<Array2<f32>> {
        let frame_size = self.config.frame_size;
        let hop_length = self.config.hop_length;
        let n_frames = (audio.len().saturating_sub(frame_size)) / hop_length + 1;

        let mut spectrum = Array2::zeros((n_frames, frame_size / 2 + 1));

        // Apply window function
        let window: Vec<f32> = (0..frame_size)
            .map(|i| {
                0.5 * (1.0
                    - (2.0 * std::f32::consts::PI * i as f32 / (frame_size - 1) as f32).cos())
            })
            .collect();

        for frame in 0..n_frames {
            let start = frame * hop_length;
            let end = (start + frame_size).min(audio.len());

            // Copy audio data with windowing
            let mut input = vec![0.0f64; frame_size];
            for (i, &sample) in audio.slice(s![start..end]).iter().enumerate() {
                input[i] = sample as f64 * window[i] as f64;
            }

            // Compute FFT using scirs2_fft
            let output = scirs2_fft::rfft(&input, None)
                .map_err(|e| VocoderError::ProcessingError(format!("FFT error: {:?}", e)))?;

            // Convert to power spectrum
            for (i, complex_val) in output.iter().enumerate() {
                spectrum[[frame, i]] =
                    (complex_val.re * complex_val.re + complex_val.im * complex_val.im) as f32;
            }
        }

        Ok(spectrum)
    }

    /// Estimate overall MOS score from available metrics
    fn estimate_mos(
        &self,
        snr: f32,
        thd_n: f32,
        lsd: f32,
        pesq: Option<f32>,
        stoi: Option<f32>,
    ) -> f32 {
        // If we have PESQ, use it as primary indicator
        if let Some(pesq_score) = pesq {
            return pesq_score.clamp(1.0, 5.0);
        }

        // Otherwise, estimate from available metrics
        let mut mos = 3.0; // Neutral starting point

        // SNR contribution (0-30 dB range)
        let snr_contrib = (snr.clamp(0.0, 30.0) / 30.0) * 2.0 - 1.0; // -1 to 1
        mos += snr_contrib * 0.8;

        // THD+N contribution (0-10% range, lower is better)
        let thd_contrib = (10.0 - thd_n.clamp(0.0, 10.0)) / 10.0; // 0 to 1
        mos += (thd_contrib * 2.0 - 1.0) * 0.5;

        // LSD contribution (0-2 range, lower is better)
        let lsd_contrib = (2.0 - lsd.clamp(0.0, 2.0)) / 2.0; // 0 to 1
        mos += (lsd_contrib * 2.0 - 1.0) * 0.7;

        // STOI contribution if available
        if let Some(stoi_score) = stoi {
            mos += (stoi_score * 2.0 - 1.0) * 0.6;
        }

        mos.clamp(1.0, 5.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioBuffer;

    #[test]
    fn test_quality_calculator_creation() {
        let config = QualityConfig::default();
        let calculator = QualityCalculator::new(config);
        assert!(calculator.config.include_expensive);
    }

    #[test]
    fn test_snr_calculation() {
        let config = QualityConfig::default();
        let calculator = QualityCalculator::new(config);

        let reference = Array1::from_vec(vec![1.0, 0.5, -0.5, -1.0]);
        let degraded = Array1::from_vec(vec![1.1, 0.4, -0.6, -0.9]);

        let snr = calculator.calculate_snr(&reference, &degraded);
        assert!(snr > 0.0); // Should be positive for small noise
    }

    #[test]
    fn test_perfect_reconstruction() {
        let config = QualityConfig::default();
        let mut calculator = QualityCalculator::new(config);

        // Create longer signal for proper analysis (>1024 samples for MOS prediction)
        let samples: Vec<f32> = (0..2048).map(|i| 0.5 * (i as f32 * 0.01).sin()).collect();
        let audio = AudioBuffer::new(samples.clone(), 22050, 1);

        let result = calculator.calculate_metrics(&audio, &audio).unwrap();

        // Perfect reconstruction should have very high SNR
        assert!(result.snr > 100.0 || result.snr == f32::INFINITY);
        assert!(result.mos_estimate >= 4.0);
    }

    #[test]
    fn test_to_mono_conversion() {
        let config = QualityConfig::default();
        let calculator = QualityCalculator::new(config);

        // Stereo audio: [L1, R1, L2, R2, L3, R3]
        let stereo_samples = vec![1.0, 0.0, 0.5, 0.5, -1.0, 0.0];
        let stereo_audio = AudioBuffer::new(stereo_samples, 22050, 2);

        let mono = calculator.to_mono(&stereo_audio);

        // Should be averaged: [(1.0+0.0)/2, (0.5+0.5)/2, (-1.0+0.0)/2]
        let expected = [0.5, 0.5, -0.5];
        assert_eq!(mono.len(), expected.len());

        for (actual, expected) in mono.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_input_validation() {
        let config = QualityConfig::default();
        let calculator = QualityCalculator::new(config);

        let audio1 = AudioBuffer::new(vec![1.0, 2.0], 22050, 1);
        let audio2 = AudioBuffer::new(vec![1.0, 2.0], 44100, 1); // Different sample rate

        let result = calculator.validate_inputs(&audio1, &audio2);
        assert!(result.is_err());
    }
}
