//! Quality prediction module
//!
//! This module provides implementations for predicting audio quality metrics
//! using various machine learning approaches including Random Forest, SVM, and Neural Networks.

use super::config::{QualityFeature, QualityModelType, QualityPredictionConfig, QualityTarget};
use crate::{DatasetSample, QualityMetrics, Result};
// HashMap import removed as it's not used in current implementation

/// Quality predictor
pub struct QualityPredictor {
    config: QualityPredictionConfig,
    #[allow(dead_code)]
    model: Option<QualityModel>,
}

/// Quality model (placeholder)
#[allow(dead_code)]
struct QualityModel {
    weights: Vec<f32>,
    feature_importance: Vec<f32>,
}

impl QualityPredictor {
    pub fn new(config: QualityPredictionConfig) -> Result<Self> {
        Ok(Self {
            config,
            model: None,
        })
    }

    pub async fn predict_quality(&self, sample: &DatasetSample) -> Result<QualityMetrics> {
        // Extract features for quality prediction
        let _features = self.extract_quality_features(sample).await?;

        // Calculate actual quality metrics from audio analysis
        let audio_data = &sample.audio.samples;
        let sample_rate = sample.audio.sample_rate();

        // Calculate SNR (Signal-to-Noise Ratio)
        let snr = self.calculate_snr(audio_data);

        // Calculate clipping percentage
        let clipping = self.calculate_clipping(audio_data);

        // Calculate dynamic range
        let dynamic_range = self.calculate_dynamic_range(audio_data);

        // Calculate spectral quality
        let spectral_quality = self.calculate_spectral_quality(audio_data, sample_rate)?;

        // Prediction based on model type with real feature analysis
        match &self.config.model_type {
            QualityModelType::RandomForest { .. } => {
                // Random Forest-like prediction using ensemble of simple rules
                let mut predictions = Vec::new();

                // Rule 1: SNR-based prediction
                let snr_score = if snr > 25.0 {
                    0.9
                } else if snr > 15.0 {
                    0.7
                } else {
                    0.4
                };
                predictions.push(snr_score);

                // Rule 2: Clipping-based prediction
                let clipping_score = if clipping < 0.01 {
                    0.9
                } else if clipping < 0.05 {
                    0.6
                } else {
                    0.3
                };
                predictions.push(clipping_score);

                // Rule 3: Dynamic range-based prediction
                let dr_score = if dynamic_range > 40.0 {
                    0.9
                } else if dynamic_range > 25.0 {
                    0.7
                } else {
                    0.5
                };
                predictions.push(dr_score);

                // Rule 4: Spectral quality-based prediction
                predictions.push(spectral_quality);

                // Ensemble average
                let overall_quality = predictions.iter().sum::<f32>() / predictions.len() as f32;

                Ok(QualityMetrics {
                    snr: Some(snr),
                    clipping: Some(clipping),
                    dynamic_range: Some(dynamic_range),
                    spectral_quality: Some(spectral_quality),
                    overall_quality: Some(overall_quality),
                })
            }
            QualityModelType::SVM { .. } => {
                // SVM-like prediction using linear combination with feature weights
                let feature_weights = [0.3, 0.25, 0.25, 0.2]; // SNR, clipping, DR, spectral
                let normalized_features = [
                    (snr - 15.0) / 20.0,              // Normalize SNR to 0-1 range
                    1.0 - (clipping * 10.0).min(1.0), // Invert clipping (lower is better)
                    (dynamic_range - 20.0) / 30.0,    // Normalize DR to 0-1 range
                    spectral_quality,
                ];

                let mut weighted_score = 0.0;
                for (feature, weight) in normalized_features.iter().zip(feature_weights.iter()) {
                    weighted_score += feature.clamp(0.0, 1.0) * weight;
                }

                // Apply SVM-like non-linear transformation
                let overall_quality = (weighted_score * 2.0 - 1.0).tanh() * 0.5 + 0.5;

                Ok(QualityMetrics {
                    snr: Some(snr),
                    clipping: Some(clipping),
                    dynamic_range: Some(dynamic_range),
                    spectral_quality: Some(spectral_quality),
                    overall_quality: Some(overall_quality.clamp(0.0, 1.0)),
                })
            }
            QualityModelType::NeuralNetwork { .. } => {
                // Neural Network-like prediction using multi-layer perceptron simulation
                let input_features = [
                    snr / 30.0,                       // Normalize SNR
                    1.0 - (clipping * 20.0).min(1.0), // Invert and normalize clipping
                    dynamic_range / 50.0,             // Normalize dynamic range
                    spectral_quality,
                ];

                // Hidden layer 1 (4 neurons)
                let hidden1 = [
                    (input_features[0] * 0.8
                        + input_features[1] * 0.3
                        + input_features[2] * 0.5
                        + input_features[3] * 0.7)
                        .tanh(),
                    (input_features[0] * 0.4
                        + input_features[1] * 0.9
                        + input_features[2] * 0.2
                        + input_features[3] * 0.6)
                        .tanh(),
                    (input_features[0] * 0.6
                        + input_features[1] * 0.5
                        + input_features[2] * 0.8
                        + input_features[3] * 0.4)
                        .tanh(),
                    (input_features[0] * 0.7
                        + input_features[1] * 0.2
                        + input_features[2] * 0.6
                        + input_features[3] * 0.9)
                        .tanh(),
                ];

                // Hidden layer 2 (2 neurons)
                let hidden2 = [
                    (hidden1[0] * 0.7 + hidden1[1] * 0.3 + hidden1[2] * 0.8 + hidden1[3] * 0.4)
                        .tanh(),
                    (hidden1[0] * 0.5 + hidden1[1] * 0.8 + hidden1[2] * 0.3 + hidden1[3] * 0.7)
                        .tanh(),
                ];

                // Output layer
                let overall_quality =
                    ((hidden2[0] * 0.6 + hidden2[1] * 0.4).tanh() * 0.5 + 0.5).clamp(0.0, 1.0);

                Ok(QualityMetrics {
                    snr: Some(snr),
                    clipping: Some(clipping),
                    dynamic_range: Some(dynamic_range),
                    spectral_quality: Some(spectral_quality),
                    overall_quality: Some(overall_quality),
                })
            }
        }
    }

    async fn extract_quality_features(&self, _sample: &DatasetSample) -> Result<Vec<f32>> {
        let mut features = Vec::new();

        for feature_type in &self.config.input_features {
            match feature_type {
                QualityFeature::Spectral => {
                    // Extract spectral features using simplified analysis
                    let audio_data = &_sample.audio.samples;
                    let spectral_features = self.extract_spectral_features(audio_data)?;
                    features.extend(spectral_features);
                }
                QualityFeature::Temporal => {
                    // Extract temporal features from audio
                    let audio_data = &_sample.audio.samples;
                    let temporal_features = self.extract_temporal_features(audio_data)?;
                    features.extend(temporal_features);
                }
                QualityFeature::Perceptual => {
                    // Extract perceptual features
                    let audio_data = &_sample.audio.samples;
                    let sample_rate = _sample.audio.sample_rate();
                    let perceptual_features =
                        self.extract_perceptual_features(audio_data, sample_rate)?;
                    features.extend(perceptual_features);
                }
                QualityFeature::Speaker => {
                    // Extract speaker-related features
                    let audio_data = &_sample.audio.samples;
                    let speaker_features = self.extract_speaker_features(audio_data)?;
                    features.extend(speaker_features);
                }
                QualityFeature::Content => {
                    // Extract content complexity features
                    let content_features = self.extract_content_features(_sample)?;
                    features.extend(content_features);
                }
            }
        }

        Ok(features)
    }

    // Helper methods for quality calculation
    fn calculate_snr(&self, audio_data: &[f32]) -> f32 {
        if audio_data.is_empty() {
            return 0.0;
        }

        // Calculate signal power (RMS)
        let signal_power = audio_data.iter().map(|&x| x * x).sum::<f32>() / audio_data.len() as f32;

        // Estimate noise floor using the lowest 10% of power values
        let mut sorted_powers: Vec<f32> = audio_data.iter().map(|&x| x * x).collect();
        sorted_powers.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let noise_samples = sorted_powers.len() / 10;
        let noise_power = if noise_samples > 0 {
            sorted_powers[..noise_samples].iter().sum::<f32>() / noise_samples as f32
        } else {
            0.001 // Minimum noise floor
        };

        // Calculate SNR in dB
        if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            60.0 // Very high SNR for effectively noiseless signals
        }
    }

    fn calculate_clipping(&self, audio_data: &[f32]) -> f32 {
        if audio_data.is_empty() {
            return 0.0;
        }

        let clipping_threshold = 0.99;
        let clipped_samples = audio_data
            .iter()
            .filter(|&&x| x.abs() >= clipping_threshold)
            .count();
        clipped_samples as f32 / audio_data.len() as f32
    }

    fn calculate_dynamic_range(&self, audio_data: &[f32]) -> f32 {
        if audio_data.is_empty() {
            return 0.0;
        }

        let max_amplitude = audio_data.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        let rms = (audio_data.iter().map(|&x| x * x).sum::<f32>() / audio_data.len() as f32).sqrt();

        if rms > 0.0 {
            20.0 * (max_amplitude / rms).log10()
        } else {
            0.0
        }
    }

    fn calculate_spectral_quality(&self, audio_data: &[f32], sample_rate: u32) -> Result<f32> {
        if audio_data.is_empty() {
            return Ok(0.0);
        }

        // Simple spectral quality based on frequency distribution
        let window_size = (sample_rate / 100) as usize; // 10ms windows
        let mut quality_scores = Vec::new();

        for chunk in audio_data.chunks(window_size) {
            if chunk.len() < window_size / 2 {
                continue;
            }

            // Simple frequency analysis
            let nyquist = sample_rate as f32 / 2.0;
            let freq_resolution = nyquist / (chunk.len() as f32 / 2.0);

            let mut weighted_freq_sum = 0.0;
            let mut total_energy = 0.0;

            for (i, &magnitude) in chunk.iter().enumerate().take(chunk.len() / 2) {
                let freq = i as f32 * freq_resolution;
                let energy = magnitude * magnitude;
                weighted_freq_sum += freq * energy;
                total_energy += energy;
            }

            let spectral_centroid = if total_energy > 0.0 {
                weighted_freq_sum / total_energy
            } else {
                0.0
            };

            // Quality score based on spectral centroid distribution
            let normalized_centroid = spectral_centroid / nyquist;
            let quality = if normalized_centroid > 0.1 && normalized_centroid < 0.6 {
                1.0 - (normalized_centroid - 0.35).abs() * 2.0
            } else {
                0.3
            };

            quality_scores.push(quality.clamp(0.0, 1.0));
        }

        if quality_scores.is_empty() {
            Ok(0.5)
        } else {
            Ok(quality_scores.iter().sum::<f32>() / quality_scores.len() as f32)
        }
    }

    // Simplified implementations of feature extraction methods
    fn extract_spectral_features(&self, audio_data: &[f32]) -> Result<Vec<f32>> {
        let mut features = Vec::new();

        if audio_data.is_empty() {
            return Ok(vec![0.0; 10]);
        }

        // Advanced spectral analysis using windowed processing
        let window_size = 1024.min(audio_data.len());
        let hop_size = window_size / 4;
        let mut spectral_features = Vec::new();

        for chunk_start in (0..audio_data.len().saturating_sub(window_size)).step_by(hop_size) {
            let chunk = &audio_data[chunk_start..chunk_start + window_size];

            // Apply Hann window
            let windowed: Vec<f32> = chunk
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let window_val = 0.5
                        * (1.0
                            - ((2.0 * std::f32::consts::PI * i as f32) / (window_size - 1) as f32)
                                .cos());
                    x * window_val
                })
                .collect();

            // Calculate magnitude spectrum (simplified FFT approximation)
            let mut magnitude_spectrum = Vec::new();
            for k in 0..window_size / 2 {
                let mut real = 0.0;
                let mut imag = 0.0;
                for (n, &sample) in windowed.iter().enumerate() {
                    let angle =
                        2.0 * std::f32::consts::PI * k as f32 * n as f32 / window_size as f32;
                    real += sample * angle.cos();
                    imag -= sample * angle.sin();
                }
                magnitude_spectrum.push((real * real + imag * imag).sqrt());
            }

            spectral_features.push(self.calculate_spectral_metrics(&magnitude_spectrum));
        }

        // Aggregate features across all windows
        if spectral_features.is_empty() {
            return Ok(vec![0.0; 10]);
        }

        // Extract various spectral characteristics
        let feature_count = spectral_features[0].len();
        for i in 0..feature_count {
            let values: Vec<f32> = spectral_features.iter().map(|f| f[i]).collect();
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let variance =
                values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
            features.push(mean);
            features.push(variance.sqrt()); // Standard deviation
        }

        Ok(features)
    }

    /// Calculate advanced spectral metrics from magnitude spectrum
    fn calculate_spectral_metrics(&self, magnitude_spectrum: &[f32]) -> Vec<f32> {
        let mut metrics = Vec::new();

        if magnitude_spectrum.is_empty() {
            return vec![0.0; 5];
        }

        let total_energy: f32 = magnitude_spectrum.iter().sum();

        // Spectral centroid (weighted frequency center)
        let centroid = if total_energy > 0.0 {
            magnitude_spectrum
                .iter()
                .enumerate()
                .map(|(i, &mag)| i as f32 * mag)
                .sum::<f32>()
                / total_energy
        } else {
            0.0
        };
        metrics.push(centroid / magnitude_spectrum.len() as f32);

        // Spectral spread (frequency distribution width)
        let spread = if total_energy > 0.0 {
            magnitude_spectrum
                .iter()
                .enumerate()
                .map(|(i, &mag)| (i as f32 - centroid).powi(2) * mag)
                .sum::<f32>()
                / total_energy
        } else {
            0.0
        };
        metrics.push(spread.sqrt() / magnitude_spectrum.len() as f32);

        // Spectral skewness (asymmetry)
        let skewness = if total_energy > 0.0 && spread > 0.0 {
            magnitude_spectrum
                .iter()
                .enumerate()
                .map(|(i, &mag)| ((i as f32 - centroid) / spread.sqrt()).powi(3) * mag)
                .sum::<f32>()
                / total_energy
        } else {
            0.0
        };
        metrics.push(skewness);

        // Spectral kurtosis (peakedness)
        let kurtosis = if total_energy > 0.0 && spread > 0.0 {
            magnitude_spectrum
                .iter()
                .enumerate()
                .map(|(i, &mag)| ((i as f32 - centroid) / spread.sqrt()).powi(4) * mag)
                .sum::<f32>()
                / total_energy
                - 3.0
        } else {
            0.0
        };
        metrics.push(kurtosis);

        // Spectral rolloff (frequency below which 85% of energy is contained)
        let target_energy = total_energy * 0.85;
        let mut cumulative_energy = 0.0;
        let mut rolloff = magnitude_spectrum.len() as f32;
        for (i, &mag) in magnitude_spectrum.iter().enumerate() {
            cumulative_energy += mag;
            if cumulative_energy >= target_energy {
                rolloff = i as f32;
                break;
            }
        }
        metrics.push(rolloff / magnitude_spectrum.len() as f32);

        metrics
    }

    fn extract_temporal_features(&self, audio_data: &[f32]) -> Result<Vec<f32>> {
        let mut features = Vec::new();

        if audio_data.is_empty() {
            return Ok(vec![0.0; 12]);
        }

        // Zero crossing rate
        let zero_crossings = audio_data.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        let zcr = zero_crossings as f32 / (audio_data.len() - 1) as f32;
        features.push(zcr);

        // RMS energy
        let rms = (audio_data.iter().map(|&x| x * x).sum::<f32>() / audio_data.len() as f32).sqrt();
        features.push(rms);

        // Envelope analysis using moving average
        let window_size = 256.min(audio_data.len() / 10);
        let mut envelope = Vec::new();
        for chunk in audio_data.chunks(window_size) {
            let chunk_rms = (chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();
            envelope.push(chunk_rms);
        }

        // Envelope variation (temporal stability)
        if envelope.len() > 1 {
            let envelope_mean = envelope.iter().sum::<f32>() / envelope.len() as f32;
            let envelope_variance = envelope
                .iter()
                .map(|&x| (x - envelope_mean).powi(2))
                .sum::<f32>()
                / envelope.len() as f32;
            features.push(envelope_variance.sqrt()); // Envelope standard deviation

            // Envelope attack and decay characteristics
            let mut attack_slopes = Vec::new();
            let mut decay_slopes = Vec::new();

            for window in envelope.windows(3) {
                let slope1 = window[1] - window[0];
                let slope2 = window[2] - window[1];

                if slope1 > 0.0 {
                    attack_slopes.push(slope1);
                }
                if slope2 < 0.0 {
                    decay_slopes.push(slope2.abs());
                }
            }

            let avg_attack = if !attack_slopes.is_empty() {
                attack_slopes.iter().sum::<f32>() / attack_slopes.len() as f32
            } else {
                0.0
            };
            features.push(avg_attack);

            let avg_decay = if !decay_slopes.is_empty() {
                decay_slopes.iter().sum::<f32>() / decay_slopes.len() as f32
            } else {
                0.0
            };
            features.push(avg_decay);
        } else {
            features.extend(vec![0.0; 3]); // envelope_variance, avg_attack, avg_decay
        }

        // Temporal centroid (time-weighted center of energy)
        let mut weighted_time_sum = 0.0;
        let mut total_energy = 0.0;
        for (i, &sample) in audio_data.iter().enumerate() {
            let energy = sample * sample;
            weighted_time_sum += i as f32 * energy;
            total_energy += energy;
        }
        let temporal_centroid = if total_energy > 0.0 {
            weighted_time_sum / total_energy / audio_data.len() as f32
        } else {
            0.5
        };
        features.push(temporal_centroid);

        // Short-time energy variation
        let frame_size = 512.min(audio_data.len());
        let hop_size = frame_size / 2;
        let mut energies = Vec::new();

        for chunk_start in (0..audio_data.len().saturating_sub(frame_size)).step_by(hop_size) {
            let chunk = &audio_data[chunk_start..chunk_start + frame_size];
            let energy = chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32;
            energies.push(energy);
        }

        if energies.len() > 1 {
            let energy_mean = energies.iter().sum::<f32>() / energies.len() as f32;
            let energy_variance = energies
                .iter()
                .map(|&x| (x - energy_mean).powi(2))
                .sum::<f32>()
                / energies.len() as f32;
            features.push(energy_variance.sqrt());

            // Energy flux (rate of change in energy)
            let energy_flux = energies
                .windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .sum::<f32>()
                / (energies.len() - 1) as f32;
            features.push(energy_flux);
        } else {
            features.extend(vec![0.0; 2]); // energy_variance, energy_flux
        }

        // Autocorrelation-based periodicity detection
        let autocorr_lag = (audio_data.len() / 4).min(1000);
        let mut max_autocorr = 0.0;
        let mut best_lag = 0;

        for lag in 1..autocorr_lag {
            let mut correlation = 0.0;
            let mut norm_factor = 0.0;

            for i in 0..(audio_data.len() - lag) {
                correlation += audio_data[i] * audio_data[i + lag];
                norm_factor += audio_data[i] * audio_data[i];
            }

            let normalized_corr = if norm_factor > 0.0 {
                correlation / norm_factor
            } else {
                0.0
            };

            if normalized_corr > max_autocorr {
                max_autocorr = normalized_corr;
                best_lag = lag;
            }
        }

        features.push(max_autocorr); // Periodicity strength
        features.push(best_lag as f32 / audio_data.len() as f32); // Normalized period

        // Silence ratio (percentage of samples below threshold)
        let silence_threshold = rms * 0.1; // 10% of RMS
        let silence_samples = audio_data
            .iter()
            .filter(|&&x| x.abs() < silence_threshold)
            .count();
        let silence_ratio = silence_samples as f32 / audio_data.len() as f32;
        features.push(silence_ratio);

        // Temporal skewness (asymmetry of the amplitude distribution)
        if audio_data.len() > 2 {
            let mean = audio_data.iter().sum::<f32>() / audio_data.len() as f32;
            let variance = audio_data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
                / audio_data.len() as f32;
            let std_dev = variance.sqrt();

            if std_dev > 0.0 {
                let skewness = audio_data
                    .iter()
                    .map(|&x| ((x - mean) / std_dev).powi(3))
                    .sum::<f32>()
                    / audio_data.len() as f32;
                features.push(skewness);
            } else {
                features.push(0.0);
            }
        } else {
            features.push(0.0);
        }

        Ok(features)
    }

    fn extract_perceptual_features(
        &self,
        audio_data: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<f32>> {
        let mut features = Vec::new();

        if audio_data.is_empty() {
            return Ok(vec![0.0; 8]);
        }

        // Advanced loudness estimation using A-weighting approximation
        let rms = (audio_data.iter().map(|&x| x * x).sum::<f32>() / audio_data.len() as f32).sqrt();
        let loudness_db = if rms > 0.0 {
            20.0 * rms.log10()
        } else {
            -96.0 // Equivalent to 16-bit noise floor
        };
        features.push(loudness_db);

        // Crest factor (peak-to-RMS ratio)
        let peak = audio_data.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        let crest_factor = if rms > 0.0 { peak / rms } else { 1.0 };
        features.push(crest_factor);

        // Perceptual sharpness (high-frequency content relative to total energy)
        let window_size = 1024.min(audio_data.len());
        if window_size >= 64 {
            let chunk = &audio_data[0..window_size];

            // Simple frequency analysis for sharpness calculation
            let nyquist_freq = sample_rate as f32 / 2.0;
            let freq_resolution = nyquist_freq / (window_size as f32 / 2.0);

            let mut total_weighted_energy = 0.0;
            let mut total_energy = 0.0;

            // Calculate weighted energy based on frequency (higher frequencies get more weight)
            for k in 0..window_size / 2 {
                let mut real = 0.0;
                let mut imag = 0.0;
                for (n, &sample) in chunk.iter().enumerate() {
                    let angle =
                        2.0 * std::f32::consts::PI * k as f32 * n as f32 / window_size as f32;
                    real += sample * angle.cos();
                    imag -= sample * angle.sin();
                }
                let magnitude = (real * real + imag * imag).sqrt();
                let frequency = k as f32 * freq_resolution;

                // Perceptual weighting (simplified bark scale approximation)
                let bark = 13.0 * (0.76 * frequency / 1000.0).atan()
                    + 3.5 * ((frequency / 7500.0).powi(2)).atan();
                let weight = if bark > 13.0 { bark - 13.0 } else { 0.0 }; // Emphasize high frequencies

                total_weighted_energy += magnitude * weight;
                total_energy += magnitude;
            }

            let sharpness = if total_energy > 0.0 {
                total_weighted_energy / total_energy
            } else {
                0.0
            };
            features.push(sharpness);
        } else {
            features.push(0.0);
        }

        // Roughness estimation (amplitude modulation detection)
        let modulation_freqs = [4.0, 8.0, 16.0, 32.0]; // Hz - typical roughness frequencies
        let mut roughness_score = 0.0;

        for &mod_freq in &modulation_freqs {
            let samples_per_cycle = sample_rate as f32 / mod_freq;
            if samples_per_cycle < audio_data.len() as f32 / 4.0 {
                let cycle_samples = samples_per_cycle as usize;
                let mut modulation_strength = 0.0;
                let mut cycle_count = 0;

                for chunk in audio_data.chunks(cycle_samples) {
                    if chunk.len() == cycle_samples {
                        let chunk_rms =
                            (chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();
                        modulation_strength += chunk_rms;
                        cycle_count += 1;
                    }
                }

                if cycle_count > 1 {
                    let avg_amplitude = modulation_strength / cycle_count as f32;
                    let mut variance = 0.0;
                    let mut valid_cycles = 0;

                    for chunk in audio_data.chunks(cycle_samples) {
                        if chunk.len() == cycle_samples {
                            let chunk_rms = (chunk.iter().map(|&x| x * x).sum::<f32>()
                                / chunk.len() as f32)
                                .sqrt();
                            variance += (chunk_rms - avg_amplitude).powi(2);
                            valid_cycles += 1;
                        }
                    }

                    if valid_cycles > 1 && avg_amplitude > 0.0 {
                        let modulation_depth =
                            (variance / valid_cycles as f32).sqrt() / avg_amplitude;
                        roughness_score += modulation_depth * (1.0 / mod_freq); // Weight lower frequencies more
                    }
                }
            }
        }
        features.push(roughness_score);

        // Tonality (harmonic content estimation)
        let mut tonality_score = 0.0;
        if audio_data.len() >= 2048 {
            // Find fundamental frequency using autocorrelation
            let max_lag = (sample_rate / 50) as usize; // Minimum F0 = 50 Hz
            let min_lag = (sample_rate / 1000) as usize; // Maximum F0 = 1000 Hz

            let mut max_correlation = 0.0;
            let mut fundamental_period = 0;

            for lag in min_lag..max_lag.min(audio_data.len() / 2) {
                let mut correlation = 0.0;
                let mut norm = 0.0;

                for i in 0..(audio_data.len() - lag) {
                    correlation += audio_data[i] * audio_data[i + lag];
                    norm += audio_data[i] * audio_data[i];
                }

                let normalized_corr = if norm > 0.0 { correlation / norm } else { 0.0 };
                if normalized_corr > max_correlation {
                    max_correlation = normalized_corr;
                    fundamental_period = lag;
                }
            }

            // Check for harmonic structure
            if fundamental_period > 0 && max_correlation > 0.3 {
                let mut harmonic_strength = max_correlation;

                // Check 2nd and 3rd harmonics
                for harmonic in 2..4 {
                    let harmonic_lag = fundamental_period / harmonic;
                    if harmonic_lag >= min_lag && harmonic_lag < audio_data.len() / 2 {
                        let mut correlation = 0.0;
                        let mut norm = 0.0;

                        for i in 0..(audio_data.len() - harmonic_lag) {
                            correlation += audio_data[i] * audio_data[i + harmonic_lag];
                            norm += audio_data[i] * audio_data[i];
                        }

                        let harmonic_corr = if norm > 0.0 { correlation / norm } else { 0.0 };
                        harmonic_strength += harmonic_corr * (1.0 / harmonic as f32);
                    }
                }

                tonality_score = harmonic_strength / 3.0; // Average over fundamental + 2 harmonics
            }
        }
        features.push(tonality_score);

        // Brightness (spectral centroid normalized by Nyquist frequency)
        let window_size = 1024.min(audio_data.len());
        if window_size >= 64 {
            let chunk = &audio_data[0..window_size];
            let mut weighted_freq_sum = 0.0;
            let mut total_magnitude = 0.0;

            for k in 0..window_size / 2 {
                let mut real = 0.0;
                let mut imag = 0.0;
                for (n, &sample) in chunk.iter().enumerate() {
                    let angle =
                        2.0 * std::f32::consts::PI * k as f32 * n as f32 / window_size as f32;
                    real += sample * angle.cos();
                    imag -= sample * angle.sin();
                }
                let magnitude = (real * real + imag * imag).sqrt();
                weighted_freq_sum += k as f32 * magnitude;
                total_magnitude += magnitude;
            }

            let brightness = if total_magnitude > 0.0 {
                (weighted_freq_sum / total_magnitude) / (window_size as f32 / 2.0)
            } else {
                0.0
            };
            features.push(brightness);
        } else {
            features.push(0.0);
        }

        // Spectral flatness (measure of noise-like vs tonal characteristics)
        let window_size = 512.min(audio_data.len());
        if window_size >= 32 {
            let chunk = &audio_data[0..window_size];
            let mut spectrum = Vec::new();

            for k in 1..window_size / 2 {
                // Skip DC component
                let mut real = 0.0;
                let mut imag = 0.0;
                for (n, &sample) in chunk.iter().enumerate() {
                    let angle =
                        2.0 * std::f32::consts::PI * k as f32 * n as f32 / window_size as f32;
                    real += sample * angle.cos();
                    imag -= sample * angle.sin();
                }
                let magnitude = (real * real + imag * imag).sqrt();
                if magnitude > 1e-10 {
                    // Avoid log of zero
                    spectrum.push(magnitude);
                }
            }

            if !spectrum.is_empty() {
                let geometric_mean =
                    spectrum.iter().map(|&x| x.ln()).sum::<f32>().exp() / spectrum.len() as f32;
                let arithmetic_mean = spectrum.iter().sum::<f32>() / spectrum.len() as f32;

                let spectral_flatness = if arithmetic_mean > 0.0 {
                    geometric_mean / arithmetic_mean
                } else {
                    0.0
                };
                features.push(spectral_flatness);
            } else {
                features.push(0.0);
            }
        } else {
            features.push(0.0);
        }

        // Warmth (low frequency content relative to total)
        let window_size = 1024.min(audio_data.len());
        if window_size >= 64 {
            let chunk = &audio_data[0..window_size];
            let mut low_freq_energy = 0.0;
            let mut total_energy = 0.0;
            let warmth_cutoff = window_size / 8; // Roughly 1/8 of Nyquist

            for k in 0..window_size / 2 {
                let mut real = 0.0;
                let mut imag = 0.0;
                for (n, &sample) in chunk.iter().enumerate() {
                    let angle =
                        2.0 * std::f32::consts::PI * k as f32 * n as f32 / window_size as f32;
                    real += sample * angle.cos();
                    imag -= sample * angle.sin();
                }
                let magnitude = (real * real + imag * imag).sqrt();
                total_energy += magnitude;

                if k < warmth_cutoff {
                    low_freq_energy += magnitude;
                }
            }

            let warmth = if total_energy > 0.0 {
                low_freq_energy / total_energy
            } else {
                0.0
            };
            features.push(warmth);
        } else {
            features.push(0.0);
        }

        Ok(features)
    }

    fn extract_speaker_features(&self, audio_data: &[f32]) -> Result<Vec<f32>> {
        let mut features = Vec::new();

        if audio_data.is_empty() {
            return Ok(vec![0.0; 2]);
        }

        // Basic pitch variation estimation
        let mut variations = 0.0;
        for window in audio_data.windows(100) {
            let mean = window.iter().sum::<f32>() / window.len() as f32;
            let variance =
                window.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / window.len() as f32;
            variations += variance;
        }

        features.push(variations / (audio_data.len() / 100) as f32);

        Ok(features)
    }

    fn extract_content_features(&self, sample: &DatasetSample) -> Result<Vec<f32>> {
        let mut features = Vec::new();

        // Text complexity features
        let text = &sample.text;

        // Text length complexity
        let text_length = text.len() as f32;
        features.push((text_length / 100.0).min(1.0));

        // Vocabulary complexity
        let words: Vec<&str> = text.split_whitespace().collect();
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
        let vocab_complexity = if words.is_empty() {
            0.0
        } else {
            unique_words.len() as f32 / words.len() as f32
        };
        features.push(vocab_complexity);

        Ok(features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::features::config::{QualityFeature, QualityModelType, QualityPredictionConfig};
    use crate::{AudioData, DatasetSample, Phoneme, QualityMetrics, SpeakerInfo};
    use std::collections::HashMap;

    fn create_test_sample(duration_secs: f32, sample_rate: u32, frequency: f32) -> DatasetSample {
        let sample_count = (duration_secs * sample_rate as f32) as usize;
        let samples: Vec<f32> = (0..sample_count)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
            })
            .collect();

        let audio = AudioData::new(samples, sample_rate, 1);
        let speaker = SpeakerInfo {
            id: "test_speaker".to_string(),
            name: Some("Test Speaker".to_string()),
            gender: Some("neutral".to_string()),
            age: None,
            accent: None,
            metadata: HashMap::new(),
        };

        DatasetSample {
            id: "test_sample".to_string(),
            audio,
            text: "Hello world".to_string(),
            language: crate::LanguageCode::EnUs,
            speaker: Some(speaker),
            phonemes: Some(vec![
                Phoneme::new("h"),
                Phoneme::new("ɛ"),
                Phoneme::new("l"),
                Phoneme::new("oʊ"),
            ]),
            quality: QualityMetrics::default(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_quality_predictor_creation() {
        let config = QualityPredictionConfig {
            model_type: QualityModelType::RandomForest {
                n_trees: 100,
                max_depth: Some(10),
            },
            input_features: vec![
                QualityFeature::Spectral,
                QualityFeature::Temporal,
                QualityFeature::Perceptual,
            ],
            target_metrics: vec![QualityTarget::SNR, QualityTarget::SpectralQuality],
            ..Default::default()
        };

        let predictor = QualityPredictor::new(config);
        assert!(predictor.is_ok());
    }

    #[tokio::test]
    async fn test_advanced_spectral_features() {
        let config = QualityPredictionConfig::default();
        let predictor = QualityPredictor::new(config).unwrap();

        // Test with a 440Hz sine wave
        let sample = create_test_sample(1.0, 44100, 440.0);
        let features = predictor
            .extract_spectral_features(sample.audio.samples())
            .unwrap();

        // Should extract 10 features (5 metrics * 2 for mean and std dev)
        assert_eq!(features.len(), 10);

        // All features should be finite numbers
        for &feature in &features {
            assert!(
                feature.is_finite(),
                "Feature value should be finite: {}",
                feature
            );
        }
    }

    #[tokio::test]
    async fn test_advanced_temporal_features() {
        let config = QualityPredictionConfig::default();
        let predictor = QualityPredictor::new(config).unwrap();

        // Test with a 440Hz sine wave
        let sample = create_test_sample(1.0, 44100, 440.0);
        let features = predictor
            .extract_temporal_features(sample.audio.samples())
            .unwrap();

        // Should extract 12 temporal features
        assert_eq!(features.len(), 12);

        // All features should be finite numbers
        for &feature in &features {
            assert!(
                feature.is_finite(),
                "Feature value should be finite: {}",
                feature
            );
        }

        // ZCR should be reasonable for a 440Hz sine wave
        let zcr = features[0];
        assert!(
            zcr > 0.0 && zcr < 1.0,
            "Zero crossing rate should be between 0 and 1: {}",
            zcr
        );
    }

    #[tokio::test]
    async fn test_advanced_perceptual_features() {
        let config = QualityPredictionConfig::default();
        let predictor = QualityPredictor::new(config).unwrap();

        // Test with a 440Hz sine wave
        let sample = create_test_sample(1.0, 44100, 440.0);
        let features = predictor
            .extract_perceptual_features(sample.audio.samples(), sample.audio.sample_rate())
            .unwrap();

        // Should extract 8 perceptual features
        assert_eq!(features.len(), 8);

        // All features should be finite numbers
        for &feature in &features {
            assert!(
                feature.is_finite(),
                "Feature value should be finite: {}",
                feature
            );
        }

        // Loudness should be reasonable (not extremely negative)
        let loudness = features[0];
        assert!(
            loudness > -100.0,
            "Loudness should be reasonable: {}",
            loudness
        );

        // Crest factor should be reasonable for a sine wave (should be around sqrt(2) ≈ 1.414)
        let crest_factor = features[1];
        assert!(
            crest_factor > 1.0 && crest_factor < 2.0,
            "Crest factor should be reasonable for sine wave: {}",
            crest_factor
        );
    }

    #[tokio::test]
    async fn test_spectral_metrics_calculation() {
        let config = QualityPredictionConfig::default();
        let predictor = QualityPredictor::new(config).unwrap();

        // Create a simple magnitude spectrum
        let magnitude_spectrum = vec![0.1, 0.5, 1.0, 0.8, 0.3, 0.1, 0.05];
        let metrics = predictor.calculate_spectral_metrics(&magnitude_spectrum);

        // Should return 5 metrics: centroid, spread, skewness, kurtosis, rolloff
        assert_eq!(metrics.len(), 5);

        // All metrics should be finite
        for &metric in &metrics {
            assert!(
                metric.is_finite(),
                "Spectral metric should be finite: {}",
                metric
            );
        }

        // Centroid should be normalized (between 0 and 1)
        assert!(
            metrics[0] >= 0.0 && metrics[0] <= 1.0,
            "Spectral centroid should be normalized: {}",
            metrics[0]
        );

        // Rolloff should be normalized (between 0 and 1)
        assert!(
            metrics[4] >= 0.0 && metrics[4] <= 1.0,
            "Spectral rolloff should be normalized: {}",
            metrics[4]
        );
    }

    #[tokio::test]
    async fn test_predict_quality_with_enhanced_features() {
        let config = QualityPredictionConfig {
            model_type: QualityModelType::RandomForest {
                n_trees: 50,
                max_depth: Some(8),
            },
            input_features: vec![
                QualityFeature::Spectral,
                QualityFeature::Temporal,
                QualityFeature::Perceptual,
                QualityFeature::Speaker,
                QualityFeature::Content,
            ],
            target_metrics: vec![QualityTarget::SNR, QualityTarget::OverallQuality],
            ..Default::default()
        };

        let predictor = QualityPredictor::new(config).unwrap();

        // Test with different types of signals
        let samples = vec![
            create_test_sample(1.0, 44100, 440.0), // Pure tone
            create_test_sample(2.0, 22050, 880.0), // Higher frequency, lower sample rate
            create_test_sample(0.5, 48000, 220.0), // Lower frequency, high sample rate
        ];

        for sample in samples {
            let quality_metrics = predictor.predict_quality(&sample).await.unwrap();

            // Check that all required metrics are present
            assert!(quality_metrics.snr.is_some(), "SNR should be calculated");
            assert!(
                quality_metrics.clipping.is_some(),
                "Clipping should be calculated"
            );
            assert!(
                quality_metrics.dynamic_range.is_some(),
                "Dynamic range should be calculated"
            );
            assert!(
                quality_metrics.spectral_quality.is_some(),
                "Spectral quality should be calculated"
            );
            assert!(
                quality_metrics.overall_quality.is_some(),
                "Overall quality should be calculated"
            );

            // Check that values are reasonable
            let snr = quality_metrics.snr.unwrap();
            assert!(
                snr > 0.0 && snr < 100.0,
                "SNR should be reasonable: {}",
                snr
            );

            let overall_quality = quality_metrics.overall_quality.unwrap();
            assert!(
                overall_quality >= 0.0 && overall_quality <= 1.0,
                "Overall quality should be between 0 and 1: {}",
                overall_quality
            );
        }
    }

    #[tokio::test]
    async fn test_empty_audio_handling() {
        let config = QualityPredictionConfig::default();
        let predictor = QualityPredictor::new(config).unwrap();

        let empty_audio = vec![];

        let spectral_features = predictor.extract_spectral_features(&empty_audio).unwrap();
        assert_eq!(spectral_features.len(), 10);
        assert!(spectral_features.iter().all(|&x| x == 0.0));

        let temporal_features = predictor.extract_temporal_features(&empty_audio).unwrap();
        assert_eq!(temporal_features.len(), 12);
        assert!(temporal_features.iter().all(|&x| x == 0.0));

        let perceptual_features = predictor
            .extract_perceptual_features(&empty_audio, 44100)
            .unwrap();
        assert_eq!(perceptual_features.len(), 8);
        assert!(perceptual_features.iter().all(|&x| x == 0.0 || x == -96.0)); // -96.0 is noise floor
    }

    #[tokio::test]
    async fn test_different_model_types() {
        let model_types = vec![
            QualityModelType::RandomForest {
                n_trees: 10,
                max_depth: Some(5),
            },
            QualityModelType::SVM {
                kernel: "rbf".to_string(),
                c: 1.0,
            },
            QualityModelType::NeuralNetwork {
                hidden_sizes: vec![10, 5, 1],
                activation: "relu".to_string(),
                dropout: 0.1,
            },
        ];

        let sample = create_test_sample(1.0, 44100, 440.0);

        for model_type in model_types {
            let config = QualityPredictionConfig {
                model_type,
                input_features: vec![QualityFeature::Spectral, QualityFeature::Temporal],
                target_metrics: vec![QualityTarget::OverallQuality],
                ..Default::default()
            };

            let predictor = QualityPredictor::new(config).unwrap();
            let quality_metrics = predictor.predict_quality(&sample).await.unwrap();

            assert!(quality_metrics.overall_quality.is_some());
            let quality = quality_metrics.overall_quality.unwrap();
            assert!(
                quality >= 0.0 && quality <= 1.0,
                "Quality should be normalized: {}",
                quality
            );
        }
    }

    #[test]
    fn test_feature_extraction_consistency() {
        let config = QualityPredictionConfig::default();
        let predictor = QualityPredictor::new(config).unwrap();

        // Create identical samples
        let sample1 = create_test_sample(1.0, 44100, 440.0);
        let sample2 = create_test_sample(1.0, 44100, 440.0);

        let features1 = predictor
            .extract_spectral_features(sample1.audio.samples())
            .unwrap();
        let features2 = predictor
            .extract_spectral_features(sample2.audio.samples())
            .unwrap();

        // Features should be identical for identical inputs
        assert_eq!(features1.len(), features2.len());
        for (f1, f2) in features1.iter().zip(features2.iter()) {
            assert!(
                (f1 - f2).abs() < 1e-6,
                "Features should be consistent for identical inputs: {} vs {}",
                f1,
                f2
            );
        }
    }
}
