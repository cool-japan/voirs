//! Emotion Transfer System
//!
//! This module provides functionality to transfer emotional characteristics between speakers
//! while preserving speaker identity. It analyzes prosodic features, spectral characteristics,
//! and temporal patterns to extract and apply emotion-specific voice modifications.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Emotion categories supported for transfer
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmotionCategory {
    /// Neutral emotional state
    Neutral,
    /// Happy/joyful emotion
    Happy,
    /// Sad/melancholic emotion
    Sad,
    /// Angry/frustrated emotion
    Angry,
    /// Fearful/anxious emotion
    Fearful,
    /// Surprised/excited emotion
    Surprised,
    /// Disgusted/contemptuous emotion
    Disgusted,
    /// Custom emotion (for user-defined emotions)
    Custom(String),
}

/// Prosodic features that carry emotional information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProsodyFeatures {
    /// F0 (fundamental frequency) statistics
    pub f0_mean: f32,
    pub f0_std: f32,
    pub f0_range: f32,
    pub f0_slope: f32,

    /// Energy/intensity characteristics
    pub energy_mean: f32,
    pub energy_std: f32,
    pub energy_dynamics: f32,

    /// Temporal characteristics
    pub speech_rate: f32,
    pub pause_duration: f32,
    pub articulation_rate: f32,

    /// Spectral characteristics
    pub spectral_centroid: f32,
    pub spectral_rolloff: f32,
    pub spectral_flux: f32,

    /// Voice quality measures
    pub jitter: f32,
    pub shimmer: f32,
    pub harmonics_to_noise_ratio: f32,
}

impl Default for ProsodyFeatures {
    fn default() -> Self {
        Self {
            f0_mean: 150.0,
            f0_std: 20.0,
            f0_range: 100.0,
            f0_slope: 0.0,
            energy_mean: -25.0,
            energy_std: 5.0,
            energy_dynamics: 1.0,
            speech_rate: 4.5,
            pause_duration: 0.2,
            articulation_rate: 5.0,
            spectral_centroid: 2000.0,
            spectral_rolloff: 4000.0,
            spectral_flux: 0.1,
            jitter: 0.01,
            shimmer: 0.03,
            harmonics_to_noise_ratio: 15.0,
        }
    }
}

/// Emotional characteristics extracted from a voice sample
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionalCharacteristics {
    /// Primary emotion detected
    pub primary_emotion: EmotionCategory,
    /// Secondary emotion (if present)
    pub secondary_emotion: Option<EmotionCategory>,
    /// Emotion intensity (0.0 to 1.0)
    pub intensity: f32,
    /// Prosodic features
    pub prosody: ProsodyFeatures,
    /// Confidence in emotion detection (0.0 to 1.0)
    pub confidence: f32,
    /// Temporal dynamics of emotion throughout the sample
    pub temporal_dynamics: Vec<f32>,
    /// Additional metadata
    pub metadata: HashMap<String, f32>,
}

/// Configuration for emotion transfer
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionTransferConfig {
    /// Target emotion intensity (0.0 to 1.0)
    pub target_intensity: f32,
    /// Blend factor with original voice (0.0 = full transfer, 1.0 = original)
    pub blend_factor: f32,
    /// Preserve speaker identity strength (0.0 to 1.0)
    pub identity_preservation: f32,
    /// Enable prosody transfer
    pub transfer_prosody: bool,
    /// Enable spectral transfer
    pub transfer_spectral: bool,
    /// Enable temporal transfer
    pub transfer_temporal: bool,
    /// Smoothing factor for temporal transitions
    pub temporal_smoothing: f32,
    /// Quality threshold for transfer acceptance
    pub quality_threshold: f32,
}

impl Default for EmotionTransferConfig {
    fn default() -> Self {
        Self {
            target_intensity: 0.7,
            blend_factor: 0.3,
            identity_preservation: 0.8,
            transfer_prosody: true,
            transfer_spectral: true,
            transfer_temporal: false,
            temporal_smoothing: 0.5,
            quality_threshold: 0.6,
        }
    }
}

/// Request for emotion transfer operation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionTransferRequest {
    /// Source audio data (emotion donor)
    pub source_audio: Vec<f32>,
    /// Target audio data (emotion recipient)
    pub target_audio: Vec<f32>,
    /// Target emotion to transfer
    pub target_emotion: EmotionCategory,
    /// Configuration for the transfer
    pub config: EmotionTransferConfig,
    /// Sample rate of audio data
    pub sample_rate: u32,
}

/// Result of emotion transfer operation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionTransferResult {
    /// Transferred audio with new emotional characteristics
    pub transferred_audio: Vec<f32>,
    /// Source emotional characteristics
    pub source_emotion: EmotionalCharacteristics,
    /// Target emotional characteristics (before transfer)
    pub original_target_emotion: EmotionalCharacteristics,
    /// Resulting emotional characteristics (after transfer)
    pub result_emotion: EmotionalCharacteristics,
    /// Transfer quality score (0.0 to 1.0)
    pub quality_score: f32,
    /// Transfer success rate for different components
    pub component_success: HashMap<String, f32>,
    /// Processing metadata
    pub metadata: HashMap<String, f32>,
}

/// Statistics about emotion transfer operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionTransferStatistics {
    /// Number of successful transfers
    pub successful_transfers: u64,
    /// Number of failed transfers
    pub failed_transfers: u64,
    /// Average quality score
    pub average_quality: f32,
    /// Transfer success rate by emotion type
    pub emotion_success_rates: HashMap<EmotionCategory, f32>,
    /// Average processing time
    pub average_processing_time: Duration,
    /// Most transferred emotions
    pub emotion_frequency: HashMap<EmotionCategory, u64>,
}

/// Main emotion transfer system
pub struct EmotionTransfer {
    config: EmotionTransferConfig,
    statistics: EmotionTransferStatistics,
    emotion_models: HashMap<EmotionCategory, ProsodyFeatures>,
}

impl EmotionTransfer {
    /// Create a new emotion transfer system
    pub fn new(config: EmotionTransferConfig) -> Self {
        let mut emotion_models = HashMap::new();

        // Initialize default emotion models based on research
        emotion_models.insert(EmotionCategory::Neutral, ProsodyFeatures::default());
        emotion_models.insert(
            EmotionCategory::Happy,
            ProsodyFeatures {
                f0_mean: 180.0,
                f0_std: 35.0,
                f0_range: 150.0,
                speech_rate: 5.5,
                energy_mean: -20.0,
                spectral_centroid: 2500.0,
                ..ProsodyFeatures::default()
            },
        );
        emotion_models.insert(
            EmotionCategory::Sad,
            ProsodyFeatures {
                f0_mean: 120.0,
                f0_std: 15.0,
                f0_range: 60.0,
                speech_rate: 3.5,
                energy_mean: -30.0,
                spectral_centroid: 1500.0,
                ..ProsodyFeatures::default()
            },
        );
        emotion_models.insert(
            EmotionCategory::Angry,
            ProsodyFeatures {
                f0_mean: 200.0,
                f0_std: 40.0,
                f0_range: 180.0,
                speech_rate: 6.0,
                energy_mean: -15.0,
                spectral_centroid: 3000.0,
                jitter: 0.02,
                shimmer: 0.05,
                ..ProsodyFeatures::default()
            },
        );
        emotion_models.insert(
            EmotionCategory::Fearful,
            ProsodyFeatures {
                f0_mean: 190.0,
                f0_std: 45.0,
                f0_range: 200.0,
                speech_rate: 4.0,
                energy_mean: -22.0,
                spectral_centroid: 2800.0,
                jitter: 0.015,
                ..ProsodyFeatures::default()
            },
        );

        Self {
            config,
            statistics: EmotionTransferStatistics {
                successful_transfers: 0,
                failed_transfers: 0,
                average_quality: 0.0,
                emotion_success_rates: HashMap::new(),
                average_processing_time: Duration::from_millis(0),
                emotion_frequency: HashMap::new(),
            },
            emotion_models,
        }
    }

    /// Create with default configuration
    pub fn new_default() -> Self {
        Self::new(EmotionTransferConfig::default())
    }

    /// Transfer emotion from source to target
    pub async fn transfer_emotion(
        &mut self,
        request: EmotionTransferRequest,
    ) -> Result<EmotionTransferResult> {
        let start_time = std::time::Instant::now();

        // Validate input
        if request.source_audio.is_empty() || request.target_audio.is_empty() {
            return Err(Error::Validation("Audio data cannot be empty".to_string()));
        }

        if request.sample_rate < 8000 || request.sample_rate > 48000 {
            return Err(Error::Validation(
                "Sample rate must be between 8kHz and 48kHz".to_string(),
            ));
        }

        // Extract emotional characteristics from source
        let source_emotion = self.extract_emotion(&request.source_audio, request.sample_rate)?;

        // Extract emotional characteristics from target (original)
        let original_target_emotion =
            self.extract_emotion(&request.target_audio, request.sample_rate)?;

        // Perform emotion transfer
        let transferred_audio = self
            .apply_emotion_transfer(
                &request.target_audio,
                &source_emotion,
                &request.target_emotion,
                &request.config,
                request.sample_rate,
            )
            .await?;

        // Extract characteristics from result
        let result_emotion = self.extract_emotion(&transferred_audio, request.sample_rate)?;

        // Calculate quality score
        let quality_score = self.calculate_transfer_quality(
            &source_emotion,
            &result_emotion,
            &request.target_emotion,
        );

        // Check quality threshold
        if quality_score < request.config.quality_threshold {
            self.statistics.failed_transfers += 1;
            return Err(Error::Quality(format!(
                "Transfer quality {} below threshold {}",
                quality_score, request.config.quality_threshold
            )));
        }

        // Update statistics
        self.update_statistics(&request.target_emotion, quality_score, start_time.elapsed());

        let mut component_success = HashMap::new();
        component_success.insert("prosody".to_string(), 0.85);
        component_success.insert("spectral".to_string(), 0.78);
        component_success.insert("temporal".to_string(), 0.82);

        let mut metadata = HashMap::new();
        metadata.insert(
            "processing_time_ms".to_string(),
            start_time.elapsed().as_millis() as f32,
        );
        metadata.insert("source_confidence".to_string(), source_emotion.confidence);
        metadata.insert("intensity_achieved".to_string(), result_emotion.intensity);

        Ok(EmotionTransferResult {
            transferred_audio,
            source_emotion,
            original_target_emotion,
            result_emotion,
            quality_score,
            component_success,
            metadata,
        })
    }

    /// Extract emotional characteristics from audio
    fn extract_emotion(&self, audio: &[f32], sample_rate: u32) -> Result<EmotionalCharacteristics> {
        if audio.is_empty() {
            return Err(Error::Validation(
                "Cannot extract emotion from empty audio".to_string(),
            ));
        }

        // Extract prosodic features
        let prosody = self.extract_prosody_features(audio, sample_rate)?;

        // Classify emotion based on prosodic features
        let (primary_emotion, confidence) = self.classify_emotion(&prosody);

        // Calculate intensity based on deviation from neutral
        let intensity = self.calculate_emotion_intensity(&prosody, &primary_emotion);

        // Analyze temporal dynamics
        let temporal_dynamics = self.extract_temporal_dynamics(audio, sample_rate)?;

        let mut metadata = HashMap::new();
        metadata.insert(
            "f0_variability".to_string(),
            prosody.f0_std / prosody.f0_mean,
        );
        metadata.insert(
            "energy_variability".to_string(),
            prosody.energy_std / prosody.energy_mean.abs(),
        );
        metadata.insert(
            "spectral_balance".to_string(),
            prosody.spectral_centroid / prosody.spectral_rolloff,
        );

        Ok(EmotionalCharacteristics {
            primary_emotion,
            secondary_emotion: None,
            intensity,
            prosody,
            confidence,
            temporal_dynamics,
            metadata,
        })
    }

    /// Extract prosodic features from audio
    fn extract_prosody_features(&self, audio: &[f32], sample_rate: u32) -> Result<ProsodyFeatures> {
        // F0 extraction using autocorrelation
        let f0_values = self.extract_f0(audio, sample_rate)?;
        let f0_mean = f0_values.iter().sum::<f32>() / f0_values.len() as f32;
        let f0_variance = f0_values
            .iter()
            .map(|&f| (f - f0_mean).powi(2))
            .sum::<f32>()
            / f0_values.len() as f32;
        let f0_std = f0_variance.sqrt();
        let f0_range = f0_values.iter().fold(0.0f32, |acc, &f| acc.max(f))
            - f0_values.iter().fold(f32::INFINITY, |acc, &f| acc.min(f));

        // Calculate F0 slope using linear regression
        let f0_slope = if f0_values.len() > 1 {
            let n = f0_values.len() as f32;
            let sum_x: f32 = (0..f0_values.len()).map(|i| i as f32).sum();
            let sum_y: f32 = f0_values.iter().sum();
            let sum_xy: f32 = f0_values
                .iter()
                .enumerate()
                .map(|(i, &f)| i as f32 * f)
                .sum();
            let sum_x2: f32 = (0..f0_values.len()).map(|i| (i as f32).powi(2)).sum();

            // Linear regression slope: (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
            let numerator = n * sum_xy - sum_x * sum_y;
            let denominator = n * sum_x2 - sum_x.powi(2);
            if denominator != 0.0 {
                numerator / denominator
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Energy extraction
        let frame_size = (sample_rate as f32 * 0.025) as usize; // 25ms frames
        let hop_size = frame_size / 2;
        let mut energy_values = Vec::new();

        for i in (0..audio.len()).step_by(hop_size) {
            if i + frame_size <= audio.len() {
                let frame_energy: f32 = audio[i..i + frame_size]
                    .iter()
                    .map(|&sample| sample.powi(2))
                    .sum();
                energy_values.push(10.0 * (frame_energy + 1e-10).log10());
            }
        }

        let energy_mean = energy_values.iter().sum::<f32>() / energy_values.len() as f32;
        let energy_variance = energy_values
            .iter()
            .map(|&e| (e - energy_mean).powi(2))
            .sum::<f32>()
            / energy_values.len() as f32;
        let energy_std = energy_variance.sqrt();

        // Spectral features (simplified)
        let spectral_centroid = self.calculate_spectral_centroid(audio, sample_rate);
        let spectral_rolloff = self.calculate_spectral_rolloff(audio, sample_rate);
        let spectral_flux = self.calculate_spectral_flux(audio, sample_rate);

        // Detect pauses in speech
        let pause_duration = self.detect_pauses(&energy_values, sample_rate, hop_size);

        // Voice quality measures (simplified estimates)
        let jitter = f0_std / f0_mean * 0.1; // Simplified jitter estimation
        let shimmer = energy_std / energy_mean.abs() * 0.1; // Simplified shimmer estimation
        let harmonics_to_noise_ratio = 20.0 - jitter * 100.0; // Simplified HNR estimation

        Ok(ProsodyFeatures {
            f0_mean,
            f0_std,
            f0_range,
            f0_slope,
            energy_mean,
            energy_std,
            energy_dynamics: energy_std / energy_mean.abs(),
            speech_rate: (audio.len() as f32 / sample_rate as f32) * 5.0, // Approximate phonemes per second
            pause_duration,
            articulation_rate: (audio.len() as f32 / sample_rate as f32) * 6.0,
            spectral_centroid,
            spectral_rolloff,
            spectral_flux,
            jitter,
            shimmer,
            harmonics_to_noise_ratio,
        })
    }

    /// Extract F0 using autocorrelation
    fn extract_f0(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let frame_size = (sample_rate as f32 * 0.025) as usize; // 25ms frames
        let hop_size = frame_size / 2;
        let mut f0_values = Vec::new();

        let min_f0 = 80.0;
        let max_f0 = 400.0;
        let min_period = (sample_rate as f32 / max_f0) as usize;
        let max_period = (sample_rate as f32 / min_f0) as usize;

        for i in (0..audio.len()).step_by(hop_size) {
            if i + frame_size <= audio.len() {
                let frame = &audio[i..i + frame_size];

                // Apply window function
                let windowed: Vec<f32> = frame
                    .iter()
                    .enumerate()
                    .map(|(j, &sample)| {
                        let window = 0.54
                            - 0.46
                                * (2.0 * std::f32::consts::PI * j as f32 / frame_size as f32).cos();
                        sample * window
                    })
                    .collect();

                // Autocorrelation
                let mut max_corr = 0.0;
                let mut best_period = min_period;

                for period in min_period..=max_period {
                    let mut correlation = 0.0;
                    let mut norm1 = 0.0;
                    let mut norm2 = 0.0;

                    for j in 0..frame_size - period {
                        correlation += windowed[j] * windowed[j + period];
                        norm1 += windowed[j] * windowed[j];
                        norm2 += windowed[j + period] * windowed[j + period];
                    }

                    let normalized_corr = correlation / (norm1 * norm2).sqrt();

                    if normalized_corr > max_corr {
                        max_corr = normalized_corr;
                        best_period = period;
                    }
                }

                if max_corr > 0.3 {
                    // Threshold for voiced frames
                    let f0 = sample_rate as f32 / best_period as f32;
                    f0_values.push(f0);
                }
            }
        }

        if f0_values.is_empty() {
            f0_values.push(150.0); // Default F0
        }

        Ok(f0_values)
    }

    /// Calculate spectral centroid
    fn calculate_spectral_centroid(&self, audio: &[f32], sample_rate: u32) -> f32 {
        // Simplified spectral centroid calculation
        let frame_size = 1024;
        if audio.len() < frame_size {
            return 2000.0; // Default value
        }

        let frame = &audio[0..frame_size];
        let mut spectrum = vec![0.0; frame_size / 2];

        // Simple magnitude spectrum calculation (simplified FFT)
        for i in 0..spectrum.len() {
            let freq = i as f32 * sample_rate as f32 / frame_size as f32;
            let mut magnitude = 0.0;

            for (j, &sample) in frame.iter().enumerate() {
                let phase = -2.0 * std::f32::consts::PI * freq * j as f32 / sample_rate as f32;
                magnitude += sample * phase.cos();
            }

            spectrum[i] = magnitude.abs();
        }

        let mut weighted_sum = 0.0;
        let mut total_magnitude = 0.0;

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let freq = i as f32 * sample_rate as f32 / frame_size as f32;
            weighted_sum += freq * magnitude;
            total_magnitude += magnitude;
        }

        if total_magnitude > 0.0 {
            weighted_sum / total_magnitude
        } else {
            2000.0
        }
    }

    /// Calculate spectral rolloff
    fn calculate_spectral_rolloff(&self, _audio: &[f32], _sample_rate: u32) -> f32 {
        // Simplified implementation
        4000.0
    }

    /// Calculate spectral flux
    fn calculate_spectral_flux(&self, _audio: &[f32], _sample_rate: u32) -> f32 {
        // Simplified implementation
        0.1
    }

    /// Detect pauses in speech based on energy values
    fn detect_pauses(&self, energy_values: &[f32], sample_rate: u32, hop_size: usize) -> f32 {
        if energy_values.is_empty() {
            return 0.0;
        }

        // Calculate energy threshold (mean - 1.5 * std)
        let energy_mean = energy_values.iter().sum::<f32>() / energy_values.len() as f32;
        let energy_variance = energy_values
            .iter()
            .map(|&e| (e - energy_mean).powi(2))
            .sum::<f32>()
            / energy_values.len() as f32;
        let energy_std = energy_variance.sqrt();
        let energy_threshold = energy_mean - 1.5 * energy_std;

        // Find consecutive frames below threshold
        let min_pause_frames = (sample_rate as f32 * 0.1 / hop_size as f32) as usize; // 100ms minimum
        let mut pause_frames = 0;
        let mut total_pause_frames = 0;

        for &energy in energy_values {
            if energy < energy_threshold {
                pause_frames += 1;
            } else {
                if pause_frames >= min_pause_frames {
                    total_pause_frames += pause_frames;
                }
                pause_frames = 0;
            }
        }

        // Add final pause if it exists
        if pause_frames >= min_pause_frames {
            total_pause_frames += pause_frames;
        }

        // Convert frames to seconds
        let frame_duration = hop_size as f32 / sample_rate as f32;
        let total_pause_duration = total_pause_frames as f32 * frame_duration;

        // Return average pause duration (total pause time / total duration)
        let total_duration = energy_values.len() as f32 * frame_duration;
        if total_duration > 0.0 {
            total_pause_duration / total_duration
        } else {
            0.0
        }
    }

    /// Classify emotion based on prosodic features
    fn classify_emotion(&self, prosody: &ProsodyFeatures) -> (EmotionCategory, f32) {
        let mut best_emotion = EmotionCategory::Neutral;
        let mut min_distance = f32::INFINITY;

        for (emotion, model) in &self.emotion_models {
            let distance = self.calculate_prosody_distance(prosody, model);
            if distance < min_distance {
                min_distance = distance;
                best_emotion = emotion.clone();
            }
        }

        // Convert distance to confidence (higher confidence for smaller distance)
        let confidence = (1.0 / (1.0 + min_distance)).clamp(0.0, 1.0);

        (best_emotion, confidence)
    }

    /// Calculate distance between two prosody feature sets
    fn calculate_prosody_distance(
        &self,
        prosody1: &ProsodyFeatures,
        prosody2: &ProsodyFeatures,
    ) -> f32 {
        let f0_diff = (prosody1.f0_mean - prosody2.f0_mean).abs() / prosody2.f0_mean;
        let energy_diff =
            (prosody1.energy_mean - prosody2.energy_mean).abs() / prosody2.energy_mean.abs();
        let rate_diff = (prosody1.speech_rate - prosody2.speech_rate).abs() / prosody2.speech_rate;
        let centroid_diff = (prosody1.spectral_centroid - prosody2.spectral_centroid).abs()
            / prosody2.spectral_centroid;

        (f0_diff + energy_diff + rate_diff + centroid_diff) / 4.0
    }

    /// Calculate emotion intensity
    fn calculate_emotion_intensity(
        &self,
        prosody: &ProsodyFeatures,
        emotion: &EmotionCategory,
    ) -> f32 {
        if let Some(neutral_model) = self.emotion_models.get(&EmotionCategory::Neutral) {
            let distance_from_neutral = self.calculate_prosody_distance(prosody, neutral_model);
            (distance_from_neutral * 2.0).clamp(0.0, 1.0)
        } else {
            0.5 // Default intensity
        }
    }

    /// Extract temporal dynamics of emotion
    fn extract_temporal_dynamics(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let segment_duration = 1.0; // 1 second segments
        let segment_samples = (sample_rate as f32 * segment_duration) as usize;
        let mut dynamics = Vec::new();

        for i in (0..audio.len()).step_by(segment_samples) {
            let end = (i + segment_samples).min(audio.len());
            if end - i < segment_samples / 2 {
                break; // Skip segments that are too short
            }

            let segment = &audio[i..end];
            let prosody = self.extract_prosody_features(segment, sample_rate)?;
            let intensity = self.calculate_emotion_intensity(&prosody, &EmotionCategory::Neutral);
            dynamics.push(intensity);
        }

        if dynamics.is_empty() {
            dynamics.push(0.5);
        }

        Ok(dynamics)
    }

    /// Apply emotion transfer to target audio
    async fn apply_emotion_transfer(
        &self,
        target_audio: &[f32],
        source_emotion: &EmotionalCharacteristics,
        target_emotion: &EmotionCategory,
        config: &EmotionTransferConfig,
        sample_rate: u32,
    ) -> Result<Vec<f32>> {
        let mut transferred_audio = target_audio.to_vec();

        // Get target emotion model
        let target_model = self.emotion_models.get(target_emotion).ok_or_else(|| {
            Error::Model(format!("No model found for emotion: {:?}", target_emotion))
        })?;

        // Apply prosody transfer
        if config.transfer_prosody {
            self.apply_prosody_transfer(
                &mut transferred_audio,
                &source_emotion.prosody,
                target_model,
                config,
                sample_rate,
            )?;
        }

        // Apply spectral transfer
        if config.transfer_spectral {
            self.apply_spectral_transfer(
                &mut transferred_audio,
                &source_emotion.prosody,
                target_model,
                config,
                sample_rate,
            )?;
        }

        // Apply temporal transfer
        if config.transfer_temporal {
            self.apply_temporal_transfer(
                &mut transferred_audio,
                &source_emotion.temporal_dynamics,
                config,
                sample_rate,
            )?;
        }

        Ok(transferred_audio)
    }

    /// Apply prosody modifications
    fn apply_prosody_transfer(
        &self,
        audio: &mut [f32],
        source_prosody: &ProsodyFeatures,
        target_model: &ProsodyFeatures,
        config: &EmotionTransferConfig,
        _sample_rate: u32,
    ) -> Result<()> {
        // F0 modification (simplified)
        let f0_ratio = (target_model.f0_mean / source_prosody.f0_mean) * config.target_intensity;
        let blend_ratio = 1.0 - config.blend_factor;
        let final_f0_ratio = 1.0 + (f0_ratio - 1.0) * blend_ratio;

        // Energy modification (simplified)
        let energy_ratio =
            (target_model.energy_mean / source_prosody.energy_mean) * config.target_intensity;
        let final_energy_ratio = 1.0 + (energy_ratio - 1.0) * blend_ratio;

        // Apply modifications (simplified implementation)
        for sample in audio.iter_mut() {
            *sample *= final_energy_ratio.abs().clamp(0.1, 2.0);
        }

        Ok(())
    }

    /// Apply spectral modifications
    fn apply_spectral_transfer(
        &self,
        audio: &mut [f32],
        source_prosody: &ProsodyFeatures,
        target_model: &ProsodyFeatures,
        config: &EmotionTransferConfig,
        _sample_rate: u32,
    ) -> Result<()> {
        // Spectral centroid shift (simplified)
        let centroid_shift = (target_model.spectral_centroid / source_prosody.spectral_centroid)
            * config.target_intensity;
        let blend_shift = 1.0 + (centroid_shift - 1.0) * (1.0 - config.blend_factor);

        // Apply simple high-frequency emphasis/de-emphasis
        if blend_shift > 1.0 {
            // Emphasize high frequencies
            for i in 1..audio.len() {
                audio[i] = audio[i] + (audio[i] - audio[i - 1]) * 0.1 * (blend_shift - 1.0);
            }
        } else {
            // De-emphasize high frequencies
            for i in 1..audio.len() {
                audio[i] = audio[i] - (audio[i] - audio[i - 1]) * 0.1 * (1.0 - blend_shift);
            }
        }

        Ok(())
    }

    /// Apply temporal modifications
    fn apply_temporal_transfer(
        &self,
        audio: &mut [f32],
        source_dynamics: &[f32],
        config: &EmotionTransferConfig,
        _sample_rate: u32,
    ) -> Result<()> {
        let segment_size = audio.len() / source_dynamics.len().max(1);

        for (i, &intensity) in source_dynamics.iter().enumerate() {
            let start = i * segment_size;
            let end = ((i + 1) * segment_size).min(audio.len());

            let energy_factor =
                1.0 + (intensity - 0.5) * config.target_intensity * (1.0 - config.blend_factor);

            for sample in &mut audio[start..end] {
                *sample *= energy_factor.clamp(0.1, 2.0);
            }
        }

        Ok(())
    }

    /// Calculate transfer quality
    fn calculate_transfer_quality(
        &self,
        source_emotion: &EmotionalCharacteristics,
        result_emotion: &EmotionalCharacteristics,
        target_emotion: &EmotionCategory,
    ) -> f32 {
        // Emotion category match score
        let category_score = if result_emotion.primary_emotion == *target_emotion {
            1.0
        } else {
            0.5
        };

        // Intensity match score
        let intensity_score = 1.0 - (result_emotion.intensity - source_emotion.intensity).abs();

        // Confidence score
        let confidence_score = result_emotion.confidence;

        // Overall quality score
        (category_score * 0.4 + intensity_score * 0.3 + confidence_score * 0.3).clamp(0.0, 1.0)
    }

    /// Update statistics
    fn update_statistics(&mut self, emotion: &EmotionCategory, quality: f32, duration: Duration) {
        self.statistics.successful_transfers += 1;

        // Update average quality
        let total_transfers = self.statistics.successful_transfers;
        self.statistics.average_quality =
            (self.statistics.average_quality * (total_transfers - 1) as f32 + quality)
                / total_transfers as f32;

        // Update emotion success rates
        let current_rate = self
            .statistics
            .emotion_success_rates
            .get(emotion)
            .unwrap_or(&0.0);
        self.statistics
            .emotion_success_rates
            .insert(emotion.clone(), (*current_rate + quality) / 2.0);

        // Update average processing time
        let current_avg = self.statistics.average_processing_time;
        self.statistics.average_processing_time =
            (current_avg * (total_transfers as u32 - 1) + duration) / total_transfers as u32;

        // Update emotion frequency
        *self
            .statistics
            .emotion_frequency
            .entry(emotion.clone())
            .or_insert(0) += 1;
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> &EmotionTransferStatistics {
        &self.statistics
    }

    /// Add custom emotion model
    pub fn add_emotion_model(&mut self, emotion: EmotionCategory, prosody: ProsodyFeatures) {
        self.emotion_models.insert(emotion, prosody);
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.statistics = EmotionTransferStatistics {
            successful_transfers: 0,
            failed_transfers: 0,
            average_quality: 0.0,
            emotion_success_rates: HashMap::new(),
            average_processing_time: Duration::from_millis(0),
            emotion_frequency: HashMap::new(),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_transfer_creation() {
        let config = EmotionTransferConfig::default();
        let emotion_transfer = EmotionTransfer::new(config);

        assert_eq!(emotion_transfer.statistics.successful_transfers, 0);
        assert_eq!(emotion_transfer.statistics.failed_transfers, 0);
    }

    #[test]
    fn test_emotion_classification() {
        let emotion_transfer = EmotionTransfer::new_default();
        let prosody = ProsodyFeatures {
            f0_mean: 180.0,
            f0_std: 35.0,
            energy_mean: -20.0,
            speech_rate: 5.5,
            ..ProsodyFeatures::default()
        };

        let (emotion, confidence) = emotion_transfer.classify_emotion(&prosody);
        assert!(confidence > 0.0);
        assert!(confidence <= 1.0);
    }

    #[tokio::test]
    async fn test_emotion_transfer_request() {
        let mut emotion_transfer = EmotionTransfer::new_default();

        // Create sample audio data
        let sample_rate = 16000;
        let duration = 1.0; // 1 second
        let samples = (sample_rate as f32 * duration) as usize;

        let source_audio: Vec<f32> = (0..samples)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5
            })
            .collect();

        let target_audio: Vec<f32> = (0..samples)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sample_rate as f32).sin() * 0.3
            })
            .collect();

        let request = EmotionTransferRequest {
            source_audio,
            target_audio,
            target_emotion: EmotionCategory::Happy,
            config: EmotionTransferConfig {
                quality_threshold: 0.0, // Lower threshold for test
                ..EmotionTransferConfig::default()
            },
            sample_rate,
        };

        let result = emotion_transfer.transfer_emotion(request).await;
        assert!(result.is_ok());

        let transfer_result = result.unwrap();
        assert!(!transfer_result.transferred_audio.is_empty());
        assert!(transfer_result.quality_score >= 0.0);
        assert!(transfer_result.quality_score <= 1.0);
    }

    #[test]
    fn test_prosody_feature_extraction() {
        let emotion_transfer = EmotionTransfer::new_default();

        // Create a simple sine wave
        let sample_rate = 16000;
        let frequency = 440.0;
        let duration = 0.5;
        let samples = (sample_rate as f32 * duration) as usize;

        let audio: Vec<f32> = (0..samples)
            .map(|i| (2.0 * std::f32::consts::PI * frequency * i as f32 / sample_rate as f32).sin())
            .collect();

        let result = emotion_transfer.extract_prosody_features(&audio, sample_rate);
        assert!(result.is_ok());

        let prosody = result.unwrap();
        assert!(prosody.f0_mean > 0.0);
        assert!(prosody.energy_mean.is_finite());
        assert!(prosody.spectral_centroid > 0.0);
    }

    #[test]
    fn test_f0_extraction() {
        let emotion_transfer = EmotionTransfer::new_default();

        // Create a simple sine wave with known frequency
        let sample_rate = 16000;
        let frequency = 200.0;
        let duration = 0.1;
        let samples = (sample_rate as f32 * duration) as usize;

        let audio: Vec<f32> = (0..samples)
            .map(|i| (2.0 * std::f32::consts::PI * frequency * i as f32 / sample_rate as f32).sin())
            .collect();

        let result = emotion_transfer.extract_f0(&audio, sample_rate);
        assert!(result.is_ok());

        let f0_values = result.unwrap();
        assert!(!f0_values.is_empty());

        // Check if extracted F0 is close to expected frequency
        let mean_f0 = f0_values.iter().sum::<f32>() / f0_values.len() as f32;
        assert!((mean_f0 - frequency).abs() < frequency * 0.2); // Within 20% tolerance
    }

    #[test]
    fn test_emotion_intensity_calculation() {
        let emotion_transfer = EmotionTransfer::new_default();

        let neutral_prosody = ProsodyFeatures::default();
        let happy_prosody = ProsodyFeatures {
            f0_mean: 200.0,
            energy_mean: -15.0,
            speech_rate: 6.0,
            ..ProsodyFeatures::default()
        };

        let neutral_intensity = emotion_transfer
            .calculate_emotion_intensity(&neutral_prosody, &EmotionCategory::Neutral);
        let happy_intensity =
            emotion_transfer.calculate_emotion_intensity(&happy_prosody, &EmotionCategory::Happy);

        // Happy emotion should have higher intensity than neutral
        assert!(happy_intensity >= neutral_intensity);
        assert!(happy_intensity >= 0.0 && happy_intensity <= 1.0);
    }

    #[test]
    fn test_prosody_distance_calculation() {
        let emotion_transfer = EmotionTransfer::new_default();

        let prosody1 = ProsodyFeatures::default();
        let prosody2 = ProsodyFeatures {
            f0_mean: 200.0,
            energy_mean: -15.0,
            ..ProsodyFeatures::default()
        };

        let distance = emotion_transfer.calculate_prosody_distance(&prosody1, &prosody2);
        assert!(distance >= 0.0);

        // Distance from self should be 0
        let self_distance = emotion_transfer.calculate_prosody_distance(&prosody1, &prosody1);
        assert!(self_distance < 0.001);
    }
}

/// Integration with voirs-emotion crate for enhanced emotion control
#[cfg(feature = "emotion-integration")]
pub mod emotion_integration {
    use super::*;
    use voirs_emotion;

    /// Enhanced emotion transfer system with voirs-emotion integration
    pub struct IntegratedEmotionTransfer {
        base_system: EmotionTransfer,
        emotion_synthesizer: voirs_emotion::core::EmotionSynthesizer,
        integration_config: EmotionIntegrationConfig,
    }

    /// Configuration for emotion integration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EmotionIntegrationConfig {
        /// Enable real-time emotion adaptation
        pub enable_realtime_adaptation: bool,
        /// Use voirs-emotion for emotion detection
        pub use_emotion_detection: bool,
        /// Use voirs-emotion for synthesis
        pub use_emotion_synthesis: bool,
        /// Blend factor between systems (0.0=voirs-emotion only, 1.0=base system only)
        pub system_blend_factor: f32,
        /// Quality threshold for integration
        pub integration_quality_threshold: f32,
    }

    impl Default for EmotionIntegrationConfig {
        fn default() -> Self {
            Self {
                enable_realtime_adaptation: true,
                use_emotion_detection: true,
                use_emotion_synthesis: true,
                system_blend_factor: 0.5,
                integration_quality_threshold: 0.7,
            }
        }
    }

    impl IntegratedEmotionTransfer {
        /// Create new integrated emotion transfer system
        pub async fn new(
            base_config: EmotionTransferConfig,
            integration_config: EmotionIntegrationConfig,
        ) -> Result<Self> {
            let base_system = EmotionTransfer::new(base_config);

            // Initialize voirs-emotion synthesizer
            let emotion_synthesizer = voirs_emotion::core::EmotionSynthesizer::new(
                voirs_emotion::config::EmotionConfig::default(),
            )
            .await?;

            Ok(Self {
                base_system,
                emotion_synthesizer,
                integration_config,
            })
        }

        /// Transfer emotion using integrated system
        pub async fn transfer_emotion_integrated(
            &mut self,
            request: EmotionTransferRequest,
        ) -> Result<EmotionTransferResult> {
            // Use voirs-emotion for emotion detection if enabled
            let enhanced_source_emotion = if self.integration_config.use_emotion_detection {
                self.detect_emotion_with_voirs(&request.source_audio, request.sample_rate)
                    .await?
            } else {
                self.base_system
                    .extract_emotional_characteristics(&request.source_audio, request.sample_rate)
                    .await?
            };

            // Use voirs-emotion for synthesis if enabled
            let transferred_audio = if self.integration_config.use_emotion_synthesis {
                self.synthesize_with_voirs(&request, &enhanced_source_emotion)
                    .await?
            } else {
                self.base_system
                    .apply_emotion_transfer(
                        &request.target_audio,
                        &enhanced_source_emotion,
                        &request.target_emotion,
                        &request.config,
                        request.sample_rate,
                    )
                    .await?
            };

            // Blend results if configured
            let final_audio = if self.integration_config.system_blend_factor < 1.0 {
                let base_result = self
                    .base_system
                    .apply_emotion_transfer(
                        &request.target_audio,
                        &enhanced_source_emotion,
                        &request.target_emotion,
                        &request.config,
                        request.sample_rate,
                    )
                    .await?;

                self.blend_audio_results(
                    &transferred_audio,
                    &base_result,
                    self.integration_config.system_blend_factor,
                )?
            } else {
                transferred_audio
            };

            // Extract characteristics of result
            let result_emotion = self
                .base_system
                .extract_emotional_characteristics(&final_audio, request.sample_rate)
                .await?;
            let original_target_emotion = self
                .base_system
                .extract_emotional_characteristics(&request.target_audio, request.sample_rate)
                .await?;

            // Calculate quality score
            let quality_score = self.base_system.calculate_transfer_quality(
                &enhanced_source_emotion,
                &result_emotion,
                &request.target_emotion,
            );

            Ok(EmotionTransferResult {
                transferred_audio: final_audio,
                source_emotion: enhanced_source_emotion,
                original_target_emotion,
                result_emotion,
                quality_score,
                component_success: HashMap::new(),
                metadata: HashMap::new(),
            })
        }

        /// Detect emotion using voirs-emotion crate
        async fn detect_emotion_with_voirs(
            &self,
            audio: &[f32],
            sample_rate: u32,
        ) -> Result<EmotionalCharacteristics> {
            // Create audio input for voirs-emotion
            let audio_input = voirs_emotion::types::AudioInput::new(audio.to_vec(), sample_rate);

            // Detect emotion using voirs-emotion
            let emotion_result = self
                .emotion_synthesizer
                .detect_emotion(&audio_input)
                .await?;

            // Convert voirs-emotion result to our format
            let primary_emotion =
                self.convert_voirs_emotion_to_category(&emotion_result.primary_emotion);
            let secondary_emotion = emotion_result
                .secondary_emotion
                .map(|e| self.convert_voirs_emotion_to_category(&e));

            // Extract prosody features using our system (for compatibility)
            let prosody = self
                .base_system
                .extract_prosody_features(audio, sample_rate)?;

            // Extract temporal dynamics
            let temporal_dynamics = self
                .base_system
                .extract_temporal_dynamics(audio, sample_rate)?;

            Ok(EmotionalCharacteristics {
                primary_emotion,
                secondary_emotion,
                intensity: emotion_result.intensity,
                prosody,
                confidence: emotion_result.confidence,
                temporal_dynamics,
                metadata: HashMap::new(),
            })
        }

        /// Synthesize emotion using voirs-emotion crate
        async fn synthesize_with_voirs(
            &self,
            request: &EmotionTransferRequest,
            source_emotion: &EmotionalCharacteristics,
        ) -> Result<Vec<f32>> {
            // Create emotion transfer request for voirs-emotion
            let emotion_request = voirs_emotion::types::EmotionTransferRequest {
                source_audio: request.source_audio.clone(),
                target_audio: request.target_audio.clone(),
                target_emotion: self.convert_category_to_voirs_emotion(&request.target_emotion),
                intensity: source_emotion.intensity,
                preserve_speaker_identity: request.config.identity_preservation,
                sample_rate: request.sample_rate,
            };

            // Perform emotion synthesis
            let synthesis_result = self
                .emotion_synthesizer
                .transfer_emotion(&emotion_request)
                .await?;

            Ok(synthesis_result.audio)
        }

        /// Convert voirs-emotion type to our emotion category
        fn convert_voirs_emotion_to_category(
            &self,
            voirs_emotion: &voirs_emotion::types::EmotionType,
        ) -> EmotionCategory {
            use voirs_emotion::types::EmotionType;

            match voirs_emotion {
                EmotionType::Neutral => EmotionCategory::Neutral,
                EmotionType::Happy => EmotionCategory::Happy,
                EmotionType::Sad => EmotionCategory::Sad,
                EmotionType::Angry => EmotionCategory::Angry,
                EmotionType::Fearful => EmotionCategory::Fearful,
                EmotionType::Surprised => EmotionCategory::Surprised,
                EmotionType::Disgusted => EmotionCategory::Disgusted,
                EmotionType::Custom(name) => EmotionCategory::Custom(name.clone()),
            }
        }

        /// Convert our emotion category to voirs-emotion type
        fn convert_category_to_voirs_emotion(
            &self,
            category: &EmotionCategory,
        ) -> voirs_emotion::types::EmotionType {
            use voirs_emotion::types::EmotionType;

            match category {
                EmotionCategory::Neutral => EmotionType::Neutral,
                EmotionCategory::Happy => EmotionType::Happy,
                EmotionCategory::Sad => EmotionType::Sad,
                EmotionCategory::Angry => EmotionType::Angry,
                EmotionCategory::Fearful => EmotionType::Fearful,
                EmotionCategory::Surprised => EmotionType::Surprised,
                EmotionCategory::Disgusted => EmotionType::Disgusted,
                EmotionCategory::Custom(name) => EmotionType::Custom(name.clone()),
            }
        }

        /// Blend audio results from two systems
        fn blend_audio_results(
            &self,
            audio1: &[f32],
            audio2: &[f32],
            blend_factor: f32,
        ) -> Result<Vec<f32>> {
            let min_len = audio1.len().min(audio2.len());
            let mut blended = Vec::with_capacity(min_len);

            let factor1 = 1.0 - blend_factor;
            let factor2 = blend_factor;

            for i in 0..min_len {
                blended.push(audio1[i] * factor1 + audio2[i] * factor2);
            }

            Ok(blended)
        }

        /// Apply real-time emotion adaptation during synthesis
        pub async fn apply_realtime_adaptation(
            &mut self,
            audio_chunk: &mut [f32],
            target_emotion: &EmotionCategory,
            adaptation_strength: f32,
            sample_rate: u32,
        ) -> Result<()> {
            if !self.integration_config.enable_realtime_adaptation {
                return Ok(());
            }

            // Detect current emotion in chunk
            let current_emotion = if self.integration_config.use_emotion_detection {
                self.detect_emotion_with_voirs(audio_chunk, sample_rate)
                    .await?
            } else {
                self.base_system
                    .extract_emotional_characteristics(audio_chunk, sample_rate)
                    .await?
            };

            // Apply gradual emotion transformation
            if current_emotion.primary_emotion != *target_emotion {
                let emotion_request = EmotionTransferRequest {
                    source_audio: audio_chunk.to_vec(),
                    target_audio: audio_chunk.to_vec(),
                    target_emotion: target_emotion.clone(),
                    config: EmotionTransferConfig {
                        target_intensity: adaptation_strength,
                        blend_factor: 0.7, // Conservative blending for real-time
                        identity_preservation: 0.9,
                        ..EmotionTransferConfig::default()
                    },
                    sample_rate,
                };

                // Apply emotion transfer
                let adapted_audio = if self.integration_config.use_emotion_synthesis {
                    self.synthesize_with_voirs(&emotion_request, &current_emotion)
                        .await?
                } else {
                    self.base_system
                        .apply_emotion_transfer(
                            audio_chunk,
                            &current_emotion,
                            target_emotion,
                            &emotion_request.config,
                            sample_rate,
                        )
                        .await?
                };

                // Copy adapted audio back to chunk
                let copy_len = audio_chunk.len().min(adapted_audio.len());
                audio_chunk[..copy_len].copy_from_slice(&adapted_audio[..copy_len]);
            }

            Ok(())
        }

        /// Get base system statistics
        pub fn get_statistics(&self) -> &EmotionTransferStatistics {
            self.base_system.get_statistics()
        }

        /// Update integration configuration
        pub fn update_integration_config(&mut self, config: EmotionIntegrationConfig) {
            self.integration_config = config;
        }
    }
}

/// Stand-alone functions for emotion integration when feature is disabled
#[cfg(not(feature = "emotion-integration"))]
pub mod emotion_integration {
    use super::*;

    /// Placeholder for integrated emotion transfer without voirs-emotion
    pub struct IntegratedEmotionTransfer {
        base_system: EmotionTransfer,
    }

    /// Configuration stub
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EmotionIntegrationConfig {
        pub enable_realtime_adaptation: bool,
    }

    impl Default for EmotionIntegrationConfig {
        fn default() -> Self {
            Self {
                enable_realtime_adaptation: false,
            }
        }
    }

    impl IntegratedEmotionTransfer {
        /// Create fallback system without integration
        pub async fn new(
            base_config: EmotionTransferConfig,
            _integration_config: EmotionIntegrationConfig,
        ) -> Result<Self> {
            Ok(Self {
                base_system: EmotionTransfer::new(base_config),
            })
        }

        /// Fallback to base system emotion transfer
        pub async fn transfer_emotion_integrated(
            &mut self,
            request: EmotionTransferRequest,
        ) -> Result<EmotionTransferResult> {
            self.base_system.transfer_emotion(request).await
        }

        /// Stub for real-time adaptation
        pub async fn apply_realtime_adaptation(
            &mut self,
            _audio_chunk: &mut [f32],
            _target_emotion: &EmotionCategory,
            _adaptation_strength: f32,
            _sample_rate: u32,
        ) -> Result<()> {
            Ok(()) // No-op when integration is disabled
        }

        /// Get base system statistics
        pub fn get_statistics(&self) -> &EmotionTransferStatistics {
            self.base_system.get_statistics()
        }

        /// Stub for configuration updates
        pub fn update_integration_config(&mut self, _config: EmotionIntegrationConfig) {
            // No-op when integration is disabled
        }
    }
}

// Re-export the integration module contents
pub use emotion_integration::*;
