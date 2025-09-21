//! Quality metrics system for objective quality measurement and assessment

use crate::{Error, Result};
use tracing::{debug, info};

/// Objective quality metrics system for conversion evaluation
#[derive(Debug, Clone)]
pub struct QualityMetricsSystem {
    /// Reference audio features for comparison
    reference_features: Option<QualityFeatures>,
    /// Perceptual quality model parameters
    perceptual_params: PerceptualParameters,
}

/// Features used for quality assessment
#[derive(Debug, Clone)]
pub struct QualityFeatures {
    /// Spectral features
    pub spectral: Vec<f32>,
    /// Temporal features  
    pub temporal: Vec<f32>,
    /// Prosodic features
    pub prosodic: Vec<f32>,
    /// Perceptual features
    pub perceptual: Vec<f32>,
}

/// Parameters for perceptual quality modeling
#[derive(Debug, Clone)]
pub struct PerceptualParameters {
    /// Weight for spectral similarity
    pub spectral_weight: f32,
    /// Weight for temporal consistency
    pub temporal_weight: f32,
    /// Weight for prosodic preservation
    pub prosodic_weight: f32,
    /// Weight for naturalness
    pub naturalness_weight: f32,
}

impl Default for PerceptualParameters {
    fn default() -> Self {
        Self {
            spectral_weight: 0.3,
            temporal_weight: 0.2,
            prosodic_weight: 0.3,
            naturalness_weight: 0.2,
        }
    }
}

/// Objective quality metrics results
#[derive(Debug, Clone)]
pub struct ObjectiveQualityMetrics {
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f32,
    /// Spectral similarity score
    pub spectral_similarity: f32,
    /// Temporal consistency score
    pub temporal_consistency: f32,
    /// Prosodic preservation score
    pub prosodic_preservation: f32,
    /// Naturalness score
    pub naturalness: f32,
    /// Perceptual quality score
    pub perceptual_quality: f32,
    /// Signal-to-noise ratio estimate
    pub snr_estimate: f32,
    /// Segmental SNR
    pub segmental_snr: f32,
}

impl QualityMetricsSystem {
    /// Create new quality metrics system
    pub fn new() -> Self {
        Self {
            reference_features: None,
            perceptual_params: PerceptualParameters::default(),
        }
    }

    /// Create with custom perceptual parameters
    pub fn with_perceptual_params(perceptual_params: PerceptualParameters) -> Self {
        Self {
            reference_features: None,
            perceptual_params,
        }
    }

    /// Set reference audio for quality comparison
    pub fn set_reference(&mut self, reference_audio: &[f32], sample_rate: u32) -> Result<()> {
        self.reference_features =
            Some(self.extract_quality_features(reference_audio, sample_rate)?);
        Ok(())
    }

    /// Evaluate objective quality metrics
    pub fn evaluate_quality(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<ObjectiveQualityMetrics> {
        debug!(
            "Evaluating objective quality metrics for {} samples",
            audio.len()
        );

        let features = self.extract_quality_features(audio, sample_rate)?;

        let spectral_similarity = if let Some(ref reference) = self.reference_features {
            self.calculate_feature_similarity(&features.spectral, &reference.spectral)
        } else {
            self.estimate_spectral_quality(&features.spectral)
        };

        let temporal_consistency = self.calculate_temporal_consistency(&features.temporal);
        let prosodic_preservation = self.calculate_prosodic_quality(&features.prosodic);
        let naturalness = self.calculate_naturalness(&features.perceptual);
        let perceptual_quality = self.calculate_perceptual_quality(&features);

        let snr_estimate = self.estimate_snr(audio);
        let segmental_snr = self.calculate_segmental_snr(audio);

        // Calculate weighted overall score
        let overall_score = spectral_similarity * self.perceptual_params.spectral_weight
            + temporal_consistency * self.perceptual_params.temporal_weight
            + prosodic_preservation * self.perceptual_params.prosodic_weight
            + naturalness * self.perceptual_params.naturalness_weight;

        info!(
            "Quality evaluation complete: overall_score={:.3}",
            overall_score
        );

        Ok(ObjectiveQualityMetrics {
            overall_score,
            spectral_similarity,
            temporal_consistency,
            prosodic_preservation,
            naturalness,
            perceptual_quality,
            snr_estimate,
            segmental_snr,
        })
    }

    /// Extract quality-relevant features from audio
    fn extract_quality_features(
        &self,
        audio: &[f32],
        _sample_rate: u32,
    ) -> Result<QualityFeatures> {
        // Extract spectral features (simplified MFCCs)
        let spectral = self.extract_spectral_features(audio);

        // Extract temporal features
        let temporal = self.extract_temporal_features(audio);

        // Extract prosodic features
        let prosodic = self.extract_prosodic_features(audio);

        // Extract perceptual features
        let perceptual = self.extract_perceptual_features(audio);

        Ok(QualityFeatures {
            spectral,
            temporal,
            prosodic,
            perceptual,
        })
    }

    fn extract_spectral_features(&self, audio: &[f32]) -> Vec<f32> {
        // Simplified spectral feature extraction
        let mut features = Vec::new();

        let spectrum = self.calculate_power_spectrum(audio);

        // Spectral centroid
        let spectral_centroid = self.calculate_spectral_centroid(&spectrum);
        features.push(spectral_centroid);

        // Spectral rolloff
        let spectral_rolloff = self.calculate_spectral_rolloff(&spectrum);
        features.push(spectral_rolloff);

        // Spectral flatness
        let spectral_flatness = self.calculate_spectral_flatness(&spectrum);
        features.push(spectral_flatness);

        // Add simplified MFCC-like features
        let num_bands = 13;
        let band_energies = self.calculate_mel_band_energies(&spectrum, num_bands);
        features.extend(band_energies);

        features
    }

    fn extract_temporal_features(&self, audio: &[f32]) -> Vec<f32> {
        let mut features = Vec::new();

        // RMS energy
        let rms = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        features.push(rms);

        // Zero crossing rate
        let zcr = self.calculate_zero_crossing_rate(audio);
        features.push(zcr);

        // Short-time energy variation
        let energy_variation = self.calculate_energy_variation(audio);
        features.push(energy_variation);

        // Spectral flux
        let spectral_flux = self.calculate_spectral_flux(audio);
        features.push(spectral_flux);

        features
    }

    fn extract_prosodic_features(&self, audio: &[f32]) -> Vec<f32> {
        let mut features = Vec::new();

        // F0 statistics (simplified)
        let f0_stats = self.calculate_f0_statistics(audio);
        features.extend(f0_stats);

        // Energy contour statistics
        let energy_stats = self.calculate_energy_statistics(audio);
        features.extend(energy_stats);

        // Duration features (simplified)
        let duration_features = self.calculate_duration_features(audio);
        features.extend(duration_features);

        features
    }

    fn extract_perceptual_features(&self, audio: &[f32]) -> Vec<f32> {
        let mut features = Vec::new();

        // Loudness estimate
        let loudness = self.estimate_loudness(audio);
        features.push(loudness);

        // Sharpness estimate
        let sharpness = self.estimate_sharpness(audio);
        features.push(sharpness);

        // Roughness estimate
        let roughness = self.estimate_roughness(audio);
        features.push(roughness);

        features
    }

    // Implementation of helper methods for feature extraction

    fn calculate_power_spectrum(&self, audio: &[f32]) -> Vec<f32> {
        // Simplified power spectrum calculation
        let window_size = 512;
        let mut spectrum = vec![0.0; window_size / 2];

        for i in 0..window_size.min(audio.len()) {
            let real = audio[i];
            let bin = i / 2; // Simplified frequency mapping
            if bin < spectrum.len() {
                spectrum[bin] += real * real;
            }
        }

        spectrum
    }

    fn calculate_spectral_centroid(&self, spectrum: &[f32]) -> f32 {
        let mut weighted_sum = 0.0;
        let mut total_energy = 0.0;

        for (i, &energy) in spectrum.iter().enumerate() {
            weighted_sum += (i as f32) * energy;
            total_energy += energy;
        }

        if total_energy > 0.0 {
            weighted_sum / total_energy
        } else {
            0.0
        }
    }

    fn calculate_spectral_rolloff(&self, spectrum: &[f32]) -> f32 {
        let total_energy: f32 = spectrum.iter().sum();
        let threshold = total_energy * 0.85; // 85% rolloff

        let mut cumulative_energy = 0.0;
        for (i, &energy) in spectrum.iter().enumerate() {
            cumulative_energy += energy;
            if cumulative_energy >= threshold {
                return i as f32 / spectrum.len() as f32;
            }
        }

        1.0
    }

    fn calculate_spectral_flatness(&self, spectrum: &[f32]) -> f32 {
        if spectrum.is_empty() {
            return 0.0;
        }

        let geometric_mean = spectrum
            .iter()
            .filter(|&&x| x > 0.0)
            .map(|&x| x.ln())
            .sum::<f32>()
            / spectrum.len() as f32;

        let arithmetic_mean = spectrum.iter().sum::<f32>() / spectrum.len() as f32;

        if arithmetic_mean > 0.0 {
            geometric_mean.exp() / arithmetic_mean
        } else {
            0.0
        }
    }

    fn calculate_mel_band_energies(&self, spectrum: &[f32], num_bands: usize) -> Vec<f32> {
        let mut band_energies = vec![0.0; num_bands];
        let band_size = spectrum.len() / num_bands;

        for band in 0..num_bands {
            let start = band * band_size;
            let end = ((band + 1) * band_size).min(spectrum.len());

            for i in start..end {
                band_energies[band] += spectrum[i];
            }

            if end > start {
                band_energies[band] /= (end - start) as f32;
            }
        }

        band_energies
    }

    fn calculate_zero_crossing_rate(&self, audio: &[f32]) -> f32 {
        if audio.len() < 2 {
            return 0.0;
        }

        let crossings = audio
            .windows(2)
            .filter(|w| (w[0] > 0.0) != (w[1] > 0.0))
            .count();

        crossings as f32 / (audio.len() - 1) as f32
    }

    fn calculate_energy_variation(&self, audio: &[f32]) -> f32 {
        let window_size = audio.len() / 10;
        if window_size < 10 {
            return 0.0;
        }

        let mut energies = Vec::new();
        for i in (0..audio.len()).step_by(window_size) {
            let end = (i + window_size).min(audio.len());
            let energy: f32 = audio[i..end].iter().map(|x| x * x).sum();
            energies.push(energy / (end - i) as f32);
        }

        if energies.len() < 2 {
            return 0.0;
        }

        let mean = energies.iter().sum::<f32>() / energies.len() as f32;
        let variance =
            energies.iter().map(|&e| (e - mean).powi(2)).sum::<f32>() / energies.len() as f32;

        variance.sqrt()
    }

    fn calculate_spectral_flux(&self, audio: &[f32]) -> f32 {
        let window_size = 256;
        let hop_size = window_size / 2;

        let mut prev_spectrum: Vec<f32> = Vec::new();
        let mut flux_values = Vec::new();

        for i in (0..audio.len()).step_by(hop_size) {
            if i + window_size >= audio.len() {
                break;
            }

            let window = &audio[i..i + window_size];
            let spectrum = self.calculate_power_spectrum(window);

            if !prev_spectrum.is_empty() {
                let mut flux = 0.0;
                for (curr, &prev) in spectrum.iter().zip(prev_spectrum.iter()) {
                    flux += (curr - prev).max(0.0);
                }
                flux_values.push(flux);
            }

            prev_spectrum = spectrum;
        }

        if flux_values.is_empty() {
            0.0
        } else {
            flux_values.iter().sum::<f32>() / flux_values.len() as f32
        }
    }

    fn calculate_f0_statistics(&self, audio: &[f32]) -> Vec<f32> {
        // Simplified F0 extraction and statistics
        let window_size = 1024;
        let hop_size = window_size / 2;
        let mut f0_values = Vec::new();

        for i in (0..audio.len()).step_by(hop_size) {
            if i + window_size >= audio.len() {
                break;
            }

            let window = &audio[i..i + window_size];
            let f0 = self.estimate_f0_simple(window);
            if f0 > 0.0 {
                f0_values.push(f0);
            }
        }

        if f0_values.is_empty() {
            return vec![0.0, 0.0, 0.0];
        }

        let mean_f0 = f0_values.iter().sum::<f32>() / f0_values.len() as f32;
        let min_f0 = f0_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_f0 = f0_values.iter().fold(0.0f32, |a, &b| a.max(b));

        vec![mean_f0, min_f0, max_f0]
    }

    fn estimate_f0_simple(&self, audio: &[f32]) -> f32 {
        // Very simplified F0 estimation
        let mut best_lag = 0;
        let mut best_correlation = -1.0;

        let min_lag = 20; // Assuming sample rate around 44100, this gives ~440 Hz max
        let max_lag = 400; // This gives ~110 Hz min

        for lag in min_lag..max_lag.min(audio.len() / 2) {
            let mut correlation = 0.0;
            let mut count = 0;

            for i in 0..audio.len() - lag {
                correlation += audio[i] * audio[i + lag];
                count += 1;
            }

            if count > 0 {
                correlation /= count as f32;
                if correlation > best_correlation {
                    best_correlation = correlation;
                    best_lag = lag;
                }
            }
        }

        if best_lag > 0 {
            44100.0 / best_lag as f32 // Assuming 44.1kHz sample rate
        } else {
            0.0
        }
    }

    fn calculate_energy_statistics(&self, audio: &[f32]) -> Vec<f32> {
        let window_size = 256;
        let mut energies = Vec::new();

        for i in (0..audio.len()).step_by(window_size / 2) {
            let end = (i + window_size).min(audio.len());
            let energy: f32 = audio[i..end].iter().map(|x| x * x).sum();
            energies.push(energy / (end - i) as f32);
        }

        if energies.is_empty() {
            return vec![0.0, 0.0];
        }

        let mean_energy = energies.iter().sum::<f32>() / energies.len() as f32;
        let energy_variance = energies
            .iter()
            .map(|&e| (e - mean_energy).powi(2))
            .sum::<f32>()
            / energies.len() as f32;

        vec![mean_energy, energy_variance.sqrt()]
    }

    fn calculate_duration_features(&self, audio: &[f32]) -> Vec<f32> {
        // Simplified duration features
        let total_duration = audio.len() as f32;
        let non_silent_samples = audio.iter().filter(|&&x| x.abs() > 0.01).count() as f32;

        let speech_rate = if total_duration > 0.0 {
            non_silent_samples / total_duration
        } else {
            0.0
        };

        vec![speech_rate]
    }

    fn estimate_loudness(&self, audio: &[f32]) -> f32 {
        // Simplified loudness estimation based on RMS
        let rms = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        (rms * 100.0).min(1.0) // Normalize to 0-1 range
    }

    fn estimate_sharpness(&self, audio: &[f32]) -> f32 {
        // Simplified sharpness based on high-frequency content
        let spectrum = self.calculate_power_spectrum(audio);
        let total_energy: f32 = spectrum.iter().sum();

        if total_energy > 0.0 {
            let hf_start = spectrum.len() * 2 / 3; // Upper third of spectrum
            let hf_energy: f32 = spectrum[hf_start..].iter().sum();
            hf_energy / total_energy
        } else {
            0.0
        }
    }

    fn estimate_roughness(&self, audio: &[f32]) -> f32 {
        // Simplified roughness based on amplitude modulation
        if audio.len() < 3 {
            return 0.0;
        }

        let mut modulation = 0.0;
        for i in 1..audio.len() - 1 {
            let local_variation = (audio[i + 1] - audio[i - 1]).abs();
            modulation += local_variation;
        }

        (modulation / (audio.len() - 2) as f32).min(1.0)
    }

    // Quality calculation methods

    fn calculate_feature_similarity(&self, features1: &[f32], features2: &[f32]) -> f32 {
        let min_len = features1.len().min(features2.len());
        if min_len == 0 {
            return 0.0;
        }

        let mut similarity = 0.0;
        for i in 0..min_len {
            let diff = (features1[i] - features2[i]).abs();
            similarity += 1.0 / (1.0 + diff);
        }

        similarity / min_len as f32
    }

    fn estimate_spectral_quality(&self, features: &[f32]) -> f32 {
        // Estimate quality without reference based on feature characteristics
        if features.is_empty() {
            return 0.0;
        }

        // Check for typical speech-like spectral characteristics
        let centroid = features[0];
        let rolloff = if features.len() > 1 { features[1] } else { 0.5 };

        // Ideal values for speech
        let centroid_quality =
            1.0 - ((centroid / features.len() as f32 - 0.3).abs() / 0.3).min(1.0);
        let rolloff_quality = 1.0 - ((rolloff - 0.8).abs() / 0.2).min(1.0);

        (centroid_quality + rolloff_quality) / 2.0
    }

    fn calculate_temporal_consistency(&self, temporal_features: &[f32]) -> f32 {
        if temporal_features.len() < 3 {
            return 1.0;
        }

        // Check energy variation (less variation = better consistency)
        let energy_variation = temporal_features[2];
        1.0 - energy_variation.min(1.0)
    }

    fn calculate_prosodic_quality(&self, prosodic_features: &[f32]) -> f32 {
        if prosodic_features.len() < 3 {
            return 0.5;
        }

        let mean_f0 = prosodic_features[0];
        let f0_range = prosodic_features[2] - prosodic_features[1];

        // Check if F0 is in typical speech range
        let f0_quality = if mean_f0 >= 80.0 && mean_f0 <= 400.0 {
            1.0 - ((mean_f0 - 150.0).abs() / 150.0).min(1.0)
        } else {
            0.0
        };

        // Check if F0 range is reasonable
        let range_quality = if f0_range >= 10.0 && f0_range <= 100.0 {
            1.0 - ((f0_range - 30.0).abs() / 30.0).min(1.0)
        } else {
            0.0
        };

        (f0_quality + range_quality) / 2.0
    }

    fn calculate_naturalness(&self, perceptual_features: &[f32]) -> f32 {
        if perceptual_features.is_empty() {
            return 0.5;
        }

        let loudness = perceptual_features[0];
        let roughness = if perceptual_features.len() > 2 {
            perceptual_features[2]
        } else {
            0.5
        };

        // Natural speech should have moderate loudness and low roughness
        let loudness_quality = 1.0 - ((loudness - 0.3).abs() / 0.3).min(1.0);
        let roughness_quality = 1.0 - roughness;

        (loudness_quality + roughness_quality) / 2.0
    }

    fn calculate_perceptual_quality(&self, features: &QualityFeatures) -> f32 {
        // Combine all feature types for overall perceptual quality
        let spectral_quality = self.estimate_spectral_quality(&features.spectral);
        let temporal_quality = self.calculate_temporal_consistency(&features.temporal);
        let prosodic_quality = self.calculate_prosodic_quality(&features.prosodic);
        let naturalness = self.calculate_naturalness(&features.perceptual);

        // Weighted combination
        spectral_quality * 0.3 + temporal_quality * 0.2 + prosodic_quality * 0.3 + naturalness * 0.2
    }

    fn estimate_snr(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        // Estimate signal power (simplified)
        let signal_power: f32 = audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32;

        // Estimate noise power from quiet segments (simplified)
        let sorted_samples: Vec<f32> = {
            let mut samples = audio.iter().map(|&x| x.abs()).collect::<Vec<f32>>();
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            samples
        };

        let noise_threshold_idx = sorted_samples.len() / 4; // Bottom quartile as noise
        let noise_power = if noise_threshold_idx < sorted_samples.len() {
            sorted_samples[noise_threshold_idx] * sorted_samples[noise_threshold_idx]
        } else {
            0.001 // Small noise floor
        };

        if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            60.0 // High SNR when no noise detected
        }
    }

    fn calculate_segmental_snr(&self, audio: &[f32]) -> f32 {
        let segment_size = 256;
        let mut snr_values = Vec::new();

        for i in (0..audio.len()).step_by(segment_size) {
            let end = (i + segment_size).min(audio.len());
            let segment = &audio[i..end];

            if !segment.is_empty() {
                let segment_snr = self.estimate_snr(segment);
                snr_values.push(segment_snr);
            }
        }

        if snr_values.is_empty() {
            0.0
        } else {
            snr_values.iter().sum::<f32>() / snr_values.len() as f32
        }
    }
}

impl Default for QualityMetricsSystem {
    fn default() -> Self {
        Self::new()
    }
}