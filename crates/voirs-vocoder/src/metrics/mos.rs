//! MOS (Mean Opinion Score) prediction implementation
//!
//! Provides algorithmic prediction of subjective quality ratings
//! using various acoustic features and perceptual models.

use crate::{Result, VocoderError};
use scirs2_core::ndarray::{s, Array1};
use std::f32::consts::PI;

/// MOS prediction model
pub struct MosPredictor {
    /// Sample rate for analysis
    sample_rate: u32,

    /// Frame size for analysis
    frame_size: usize,

    /// Hop length
    #[allow(dead_code)]
    hop_length: usize,
}

/// Acoustic features for MOS prediction
#[derive(Debug, Clone)]
pub struct AcousticFeatures {
    /// Spectral centroid (Hz)
    pub spectral_centroid: f32,

    /// Spectral rolloff (Hz)
    pub spectral_rolloff: f32,

    /// Zero crossing rate
    pub zero_crossing_rate: f32,

    /// RMS energy
    pub rms_energy: f32,

    /// Spectral flatness
    pub spectral_flatness: f32,

    /// Harmonic-to-noise ratio (dB)
    pub hnr: f32,

    /// Jitter (pitch variation)
    pub jitter: f32,

    /// Shimmer (amplitude variation)
    pub shimmer: f32,

    /// Total harmonic distortion
    pub thd: f32,

    /// Signal-to-noise ratio
    pub snr: f32,
}

impl MosPredictor {
    /// Create new MOS predictor
    pub fn new(sample_rate: u32) -> Self {
        let frame_size = 1024;
        let hop_length = frame_size / 4;

        Self {
            sample_rate,
            frame_size,
            hop_length,
        }
    }

    /// Predict MOS score for audio signal
    pub fn predict(&self, audio: &Array1<f32>) -> Result<f32> {
        if audio.len() < self.frame_size {
            return Err(VocoderError::InputError(
                "Audio too short for MOS prediction".to_string(),
            ));
        }

        // Extract acoustic features
        let features = self.extract_features(audio)?;

        // Apply MOS prediction model
        let mos = self.features_to_mos(&features);

        Ok(mos.clamp(1.0, 5.0))
    }

    /// Extract comprehensive acoustic features
    fn extract_features(&self, audio: &Array1<f32>) -> Result<AcousticFeatures> {
        // Basic time-domain features
        let rms_energy = self.calculate_rms(audio);
        let zero_crossing_rate = self.calculate_zcr(audio);

        // Spectral features
        let spectrum = self.compute_spectrum(audio)?;
        let spectral_centroid = self.calculate_spectral_centroid(&spectrum);
        let spectral_rolloff = self.calculate_spectral_rolloff(&spectrum);
        let spectral_flatness = self.calculate_spectral_flatness(&spectrum);

        // Harmonic features
        let hnr = self.calculate_hnr(audio);
        let jitter = self.calculate_jitter(audio);
        let shimmer = self.calculate_shimmer(audio);
        let thd = self.calculate_thd(audio);

        // Noise features
        let snr = self.calculate_snr(audio);

        Ok(AcousticFeatures {
            spectral_centroid,
            spectral_rolloff,
            zero_crossing_rate,
            rms_energy,
            spectral_flatness,
            hnr,
            jitter,
            shimmer,
            thd,
            snr,
        })
    }

    /// Calculate RMS energy
    fn calculate_rms(&self, audio: &Array1<f32>) -> f32 {
        let sum_squares: f32 = audio.iter().map(|&x| x * x).sum();
        (sum_squares / audio.len() as f32).sqrt()
    }

    /// Calculate zero crossing rate
    fn calculate_zcr(&self, audio: &Array1<f32>) -> f32 {
        let mut zero_crossings = 0;
        for i in 1..audio.len() {
            if (audio[i] >= 0.0) != (audio[i - 1] >= 0.0) {
                zero_crossings += 1;
            }
        }
        zero_crossings as f32 / (audio.len() - 1) as f32
    }

    /// Compute power spectrum
    fn compute_spectrum(&self, audio: &Array1<f32>) -> Result<Array1<f32>> {
        // Use middle portion of audio
        let start = (audio.len() - self.frame_size) / 2;
        let end = start + self.frame_size;

        if end > audio.len() {
            return Err(VocoderError::InputError("Audio too short".to_string()));
        }

        let audio_slice = audio.slice(s![start..end]);

        // Apply Hanning window
        let mut input = vec![0.0f64; self.frame_size];
        for (i, &sample) in audio_slice.iter().enumerate() {
            let window =
                0.5 * (1.0 - (2.0 * PI as f64 * i as f64 / (self.frame_size - 1) as f64).cos());
            input[i] = sample as f64 * window;
        }

        // Compute FFT using scirs2_fft
        let output = scirs2_fft::rfft(&input, None)
            .map_err(|e| VocoderError::ProcessingError(format!("FFT error: {:?}", e)))?;

        // Convert to power spectrum
        let power_spectrum: Vec<f32> = output
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im) as f32)
            .collect();

        Ok(Array1::from_vec(power_spectrum))
    }

    /// Calculate spectral centroid
    fn calculate_spectral_centroid(&self, spectrum: &Array1<f32>) -> f32 {
        let freq_resolution = self.sample_rate as f32 / self.frame_size as f32;

        let mut weighted_sum = 0.0;
        let mut total_magnitude = 0.0;

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let freq = i as f32 * freq_resolution;
            weighted_sum += freq * magnitude;
            total_magnitude += magnitude;
        }

        if total_magnitude > 0.0 {
            weighted_sum / total_magnitude
        } else {
            0.0
        }
    }

    /// Calculate spectral rolloff (frequency below which 85% of energy is contained)
    fn calculate_spectral_rolloff(&self, spectrum: &Array1<f32>) -> f32 {
        let total_energy: f32 = spectrum.sum();
        let threshold = total_energy * 0.85;

        let mut cumulative_energy = 0.0;
        let freq_resolution = self.sample_rate as f32 / self.frame_size as f32;

        for (i, &magnitude) in spectrum.iter().enumerate() {
            cumulative_energy += magnitude;
            if cumulative_energy >= threshold {
                return i as f32 * freq_resolution;
            }
        }

        (spectrum.len() - 1) as f32 * freq_resolution
    }

    /// Calculate spectral flatness (measure of noise-like vs tonal quality)
    fn calculate_spectral_flatness(&self, spectrum: &Array1<f32>) -> f32 {
        // Skip DC component
        let relevant_spectrum = spectrum.slice(s![1..spectrum.len() / 2]);

        if relevant_spectrum.is_empty() {
            return 0.0;
        }

        // Geometric mean
        let log_sum: f32 = relevant_spectrum
            .iter()
            .map(|&x| if x > 1e-20 { x.ln() } else { -46.0 }) // -46 ≈ ln(1e-20)
            .sum();
        let geometric_mean = (log_sum / relevant_spectrum.len() as f32).exp();

        // Arithmetic mean
        let arithmetic_mean = relevant_spectrum.mean().unwrap_or(0.0);

        if arithmetic_mean > 1e-20 {
            geometric_mean / arithmetic_mean
        } else {
            0.0
        }
    }

    /// Calculate harmonic-to-noise ratio (simplified)
    fn calculate_hnr(&self, audio: &Array1<f32>) -> f32 {
        // Simplified HNR calculation using autocorrelation
        let max_lag = self.sample_rate as usize / 50; // For fundamental frequencies down to 50 Hz
        let min_lag = self.sample_rate as usize / 500; // Up to 500 Hz

        let mut max_correlation = 0.0;
        let mut best_lag = min_lag;

        // Find the lag with maximum autocorrelation (fundamental period)
        for lag in min_lag..max_lag.min(audio.len() / 2) {
            let mut correlation = 0.0;
            let mut count = 0;

            for i in 0..(audio.len() - lag) {
                correlation += audio[i] * audio[i + lag];
                count += 1;
            }

            if count > 0 {
                correlation /= count as f32;
                if correlation > max_correlation {
                    max_correlation = correlation;
                    best_lag = lag;
                }
            }
        }

        // Calculate HNR using the best lag
        if max_correlation > 0.0 {
            let noise_correlation = self.calculate_noise_correlation(audio, best_lag);
            let signal_power = max_correlation;
            let noise_power = (signal_power - noise_correlation).max(signal_power * 0.01);

            10.0 * (signal_power / noise_power).log10()
        } else {
            -10.0 // Low HNR for non-periodic signals
        }
    }

    /// Calculate noise correlation for HNR
    fn calculate_noise_correlation(&self, audio: &Array1<f32>, period: usize) -> f32 {
        let mut noise_energy = 0.0;
        let mut count = 0;

        for i in period..(audio.len() - period) {
            let predicted = audio[i - period];
            let actual = audio[i];
            let error = actual - predicted;
            noise_energy += error * error;
            count += 1;
        }

        if count > 0 {
            noise_energy / count as f32
        } else {
            0.0
        }
    }

    /// Calculate jitter (pitch variation)
    fn calculate_jitter(&self, audio: &Array1<f32>) -> f32 {
        // Simplified jitter calculation based on period variation
        let periods = self.extract_periods(audio);

        if periods.len() < 2 {
            return 0.0;
        }

        let mean_period: f32 = periods.iter().sum::<f32>() / periods.len() as f32;

        let period_variation: f32 = periods
            .iter()
            .map(|&p| (p - mean_period).abs())
            .sum::<f32>()
            / periods.len() as f32;

        if mean_period > 0.0 {
            period_variation / mean_period * 100.0 // Percentage
        } else {
            0.0
        }
    }

    /// Calculate shimmer (amplitude variation)
    fn calculate_shimmer(&self, audio: &Array1<f32>) -> f32 {
        // Calculate RMS amplitude for each period
        let periods = self.extract_periods(audio);
        let mut amplitudes = Vec::new();

        let mut start = 0;
        for &period_len in &periods {
            let end = (start + period_len as usize).min(audio.len());
            if end > start {
                let period_rms = self.calculate_rms(&audio.slice(s![start..end]).to_owned());
                amplitudes.push(period_rms);
                start = end;
            }
        }

        if amplitudes.len() < 2 {
            return 0.0;
        }

        let mean_amplitude: f32 = amplitudes.iter().sum::<f32>() / amplitudes.len() as f32;

        let amplitude_variation: f32 = amplitudes
            .iter()
            .map(|&a| (a - mean_amplitude).abs())
            .sum::<f32>()
            / amplitudes.len() as f32;

        if mean_amplitude > 0.0 {
            amplitude_variation / mean_amplitude * 100.0 // Percentage
        } else {
            0.0
        }
    }

    /// Extract fundamental periods from audio
    fn extract_periods(&self, audio: &Array1<f32>) -> Vec<f32> {
        // Simplified period extraction using zero crossings
        let mut periods = Vec::new();
        let mut last_crossing = 0;
        let mut positive = audio[0] >= 0.0;

        for (i, &sample) in audio.iter().enumerate().skip(1) {
            let current_positive = sample >= 0.0;

            if current_positive != positive && current_positive {
                // Found positive-going zero crossing
                let period_length = i - last_crossing;
                if period_length > 20 && period_length < 1000 {
                    // Reasonable period range
                    periods.push(period_length as f32);
                }
                last_crossing = i;
            }
            positive = current_positive;
        }

        periods
    }

    /// Calculate total harmonic distortion
    fn calculate_thd(&self, audio: &Array1<f32>) -> f32 {
        let spectrum = match self.compute_spectrum(audio) {
            Ok(s) => s,
            Err(_) => return 0.0,
        };

        // Find fundamental frequency (simplified)
        let mut max_bin = 1;
        let mut max_magnitude = spectrum[1];

        for i in 2..spectrum.len() / 4 {
            // Look in lower frequencies
            if spectrum[i] > max_magnitude {
                max_magnitude = spectrum[i];
                max_bin = i;
            }
        }

        // Calculate harmonic distortion
        let fundamental_power = spectrum[max_bin];
        let mut harmonic_power = 0.0;

        // Check harmonics (2f, 3f, 4f, 5f)
        for harmonic in 2..=5 {
            let harmonic_bin = max_bin * harmonic;
            if harmonic_bin < spectrum.len() {
                harmonic_power += spectrum[harmonic_bin];
            }
        }

        if fundamental_power > 1e-20 {
            (harmonic_power / fundamental_power * 100.0).min(100.0)
        } else {
            0.0
        }
    }

    /// Calculate signal-to-noise ratio
    fn calculate_snr(&self, audio: &Array1<f32>) -> f32 {
        let signal_power: f32 = audio.iter().map(|&x| x * x).sum();

        // Estimate noise from high-frequency content
        let mut noise_estimate = 0.0;
        for i in 1..audio.len() {
            let diff = audio[i] - audio[i - 1];
            noise_estimate += diff * diff;
        }

        if noise_estimate > 1e-20 && signal_power > 1e-20 {
            10.0 * (signal_power / noise_estimate).log10()
        } else if noise_estimate <= 1e-20 {
            60.0 // High SNR
        } else {
            -10.0 // Low SNR
        }
    }

    /// Convert acoustic features to MOS score using empirical model
    fn features_to_mos(&self, features: &AcousticFeatures) -> f32 {
        // Empirical model based on typical relationships
        let mut mos = 3.0; // Neutral starting point

        // SNR contribution (major factor)
        let snr_score = (features.snr.clamp(0.0, 40.0) / 40.0) * 2.0; // 0 to 2
        mos += snr_score * 0.4;

        // HNR contribution (harmonicity)
        let hnr_score = (features.hnr.clamp(0.0, 20.0) / 20.0) * 2.0; // 0 to 2
        mos += hnr_score * 0.3;

        // THD contribution (distortion, lower is better)
        let thd_score = (10.0 - features.thd.clamp(0.0, 10.0)) / 10.0 * 2.0; // 0 to 2
        mos += thd_score * 0.2;

        // Jitter and shimmer (stability, lower is better)
        let stability_score =
            (5.0 - (features.jitter + features.shimmer).clamp(0.0, 5.0)) / 5.0 * 2.0;
        mos += stability_score * 0.15;

        // Spectral quality
        let spectral_score = features.spectral_flatness.clamp(0.0, 1.0) * 2.0;
        mos += spectral_score * 0.1;

        // Energy normalization
        if features.rms_energy < 0.01 || features.rms_energy > 0.5 {
            mos -= 0.5; // Penalize very quiet or very loud signals
        }

        mos.clamp(1.0, 5.0)
    }
}

/// Simplified MOS prediction function
pub fn predict_mos(audio: &Array1<f32>, sample_rate: u32) -> Result<f32> {
    let predictor = MosPredictor::new(sample_rate);
    predictor.predict(audio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_mos_predictor_creation() {
        let predictor = MosPredictor::new(22050);
        assert_eq!(predictor.sample_rate, 22050);
        assert_eq!(predictor.frame_size, 1024);
    }

    #[test]
    fn test_mos_prediction() {
        // Generate test signal (sine wave)
        let duration = 2.0; // seconds
        let sample_rate = 22050;
        let freq = 440.0; // A4
        let samples: Vec<f32> = (0..(duration * sample_rate as f32) as usize)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin() * 0.5)
            .collect();

        let audio = Array1::from_vec(samples);
        let mos = predict_mos(&audio, sample_rate).unwrap();

        // Clean sine wave should have reasonable MOS
        assert!((1.0..=5.0).contains(&mos));
        assert!(mos > 3.0); // Should be above neutral for clean signal
    }

    #[test]
    fn test_rms_calculation() {
        let predictor = MosPredictor::new(22050);
        let signal = Array1::from_vec(vec![1.0, -1.0, 1.0, -1.0]);
        let rms = predictor.calculate_rms(&signal);

        assert!((rms - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_zcr_calculation() {
        let predictor = MosPredictor::new(22050);
        let signal = Array1::from_vec(vec![1.0, -1.0, 1.0, -1.0]);
        let zcr = predictor.calculate_zcr(&signal);

        // Should have high ZCR for alternating signal
        assert!(zcr > 0.5);
    }

    #[test]
    fn test_spectral_centroid() {
        let predictor = MosPredictor::new(22050);

        // Create a spectrum with energy concentrated at low frequencies
        let spectrum = Array1::from_vec(vec![0.0, 1.0, 0.5, 0.1, 0.05, 0.0, 0.0, 0.0]);
        let centroid = predictor.calculate_spectral_centroid(&spectrum);

        // Should be in the lower frequency range
        assert!(centroid > 0.0);
        assert!(centroid < 5000.0); // Reasonable for 22kHz sample rate
    }

    #[test]
    fn test_spectral_flatness() {
        let predictor = MosPredictor::new(22050);

        // Flat spectrum (white noise-like)
        let flat_spectrum = Array1::from_vec(vec![1.0; 10]);
        let flatness_flat = predictor.calculate_spectral_flatness(&flat_spectrum);

        // Peaked spectrum (tonal)
        let peaked_spectrum =
            Array1::from_vec(vec![0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);
        let flatness_peaked = predictor.calculate_spectral_flatness(&peaked_spectrum);

        // Flat spectrum should have higher flatness
        assert!(flatness_flat > flatness_peaked);
    }

    #[test]
    fn test_short_audio_error() {
        let short_audio = Array1::from_vec(vec![1.0, -1.0, 1.0]);
        let result = predict_mos(&short_audio, 22050);

        // Should return error for too short audio
        assert!(result.is_err());
    }

    #[test]
    fn test_feature_extraction() {
        let predictor = MosPredictor::new(22050);

        // Generate longer test signal
        let samples: Vec<f32> = (0..2048)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 22050.0).sin() * 0.5)
            .collect();

        let audio = Array1::from_vec(samples);
        let features = predictor.extract_features(&audio).unwrap();

        // Check that features are reasonable
        assert!(features.rms_energy > 0.0);
        assert!(features.spectral_centroid > 0.0);
        assert!(features.snr.is_finite());
    }
}
