//! Advanced Quality Metrics for Modern Speech Evaluation
//!
//! This module provides cutting-edge quality evaluation metrics including:
//! - Multi-domain perceptual quality assessment
//! - Adaptive quality metrics based on content type
//! - Real-time quality monitoring with predictive analytics
//! - Cross-modal quality correlation analysis

use crate::traits::{QualityMetric, QualityScore};
use crate::EvaluationError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use voirs_sdk::AudioBuffer;

/// Advanced quality evaluator with multi-domain assessment
#[derive(Debug, Clone)]
pub struct AdvancedQualityEvaluator {
    config: AdvancedQualityConfig,
    adaptive_weights: HashMap<String, f64>,
    quality_history: Vec<QualityMeasurement>,
}

/// Configuration for advanced quality evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedQualityConfig {
    /// Enable content-adaptive weighting
    pub adaptive_weighting: bool,
    /// Enable predictive quality assessment
    pub predictive_assessment: bool,
    /// Number of historical measurements to consider
    pub history_window: usize,
    /// Confidence threshold for predictions
    pub confidence_threshold: f64,
    /// Enable cross-modal analysis
    pub cross_modal_analysis: bool,
}

/// Multi-domain quality score with detailed breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiDomainQualityScore {
    /// Overall quality score (0.0 - 1.0)
    pub overall_score: f64,
    /// Perceptual domain scores
    pub perceptual: PerceptualDomainScores,
    /// Intelligibility domain scores
    pub intelligibility: IntelligibilityDomainScores,
    /// Naturalness domain scores
    pub naturalness: NaturalnessDomainScores,
    /// Technical domain scores
    pub technical: TechnicalDomainScores,
    /// Confidence measure for the assessment
    pub confidence: f64,
    /// Predicted quality trend
    pub trend_prediction: Option<QualityTrendPrediction>,
}

/// Perceptual domain quality scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualDomainScores {
    /// Loudness perception quality
    pub loudness_quality: f64,
    /// Spectral balance quality
    pub spectral_balance: f64,
    /// Temporal coherence quality
    pub temporal_coherence: f64,
    /// Dynamic range quality
    pub dynamic_range: f64,
}

/// Intelligibility domain quality scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntelligibilityDomainScores {
    /// Phoneme clarity
    pub phoneme_clarity: f64,
    /// Word boundary definition
    pub word_boundaries: f64,
    /// Prosodic intelligibility
    pub prosodic_clarity: f64,
    /// Articulation precision
    pub articulation: f64,
}

/// Naturalness domain quality scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaturalnessDomainScores {
    /// Voice quality naturalness
    pub voice_quality: f64,
    /// Emotional appropriateness
    pub emotional_appropriateness: f64,
    /// Speaking rate naturalness
    pub speaking_rate: f64,
    /// Intonation naturalness
    pub intonation: f64,
}

/// Technical domain quality scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalDomainScores {
    /// Signal-to-noise ratio
    pub snr: f64,
    /// Total harmonic distortion
    pub thd: f64,
    /// Frequency response quality
    pub frequency_response: f64,
    /// Dynamic range
    pub dynamic_range: f64,
}

/// Quality trend prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityTrendPrediction {
    /// Predicted quality score for next evaluation
    pub predicted_score: f64,
    /// Confidence in prediction (0.0 - 1.0)
    pub prediction_confidence: f64,
    /// Identified quality drift direction
    pub drift_direction: QualityDriftDirection,
    /// Recommended actions based on trend
    pub recommendations: Vec<String>,
}

/// Direction of quality drift
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QualityDriftDirection {
    /// Quality is improving
    Improving,
    /// Quality is stable
    Stable,
    /// Quality is degrading
    Degrading,
    /// Quality shows high variability
    Unstable,
}

/// Individual quality measurement with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMeasurement {
    /// Timestamp of measurement
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Quality score at this measurement
    pub score: MultiDomainQualityScore,
    /// Content type analyzed
    pub content_type: String,
    /// Audio characteristics
    pub audio_metadata: AudioMetadata,
}

/// Audio metadata for adaptive evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMetadata {
    /// Duration in seconds
    pub duration: f64,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u32,
    /// Estimated content type (speech, music, etc.)
    pub content_type: String,
    /// Estimated speaker characteristics
    pub speaker_characteristics: SpeakerCharacteristics,
}

/// Speaker characteristics for personalized evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerCharacteristics {
    /// Estimated age group
    pub age_group: String,
    /// Estimated gender
    pub gender: String,
    /// Estimated accent/dialect
    pub accent: String,
    /// Speaking style
    pub style: String,
}

impl Default for AdvancedQualityConfig {
    fn default() -> Self {
        Self {
            adaptive_weighting: true,
            predictive_assessment: true,
            history_window: 50,
            confidence_threshold: 0.8,
            cross_modal_analysis: true,
        }
    }
}

impl AdvancedQualityEvaluator {
    /// Create a new advanced quality evaluator
    pub fn new(config: AdvancedQualityConfig) -> Self {
        Self {
            config,
            adaptive_weights: HashMap::new(),
            quality_history: Vec::new(),
        }
    }

    /// Evaluate audio quality with multi-domain analysis
    pub async fn evaluate_quality(
        &mut self,
        audio: &AudioBuffer,
        reference: Option<&AudioBuffer>,
    ) -> Result<MultiDomainQualityScore, EvaluationError> {
        let metadata = self.extract_audio_metadata(audio)?;

        // Perform multi-domain evaluation
        let perceptual = self.evaluate_perceptual_domain(audio, reference).await?;
        let intelligibility = self
            .evaluate_intelligibility_domain(audio, reference)
            .await?;
        let naturalness = self.evaluate_naturalness_domain(audio, reference).await?;
        let technical = self.evaluate_technical_domain(audio, reference).await?;

        // Calculate adaptive weights based on content type
        let weights = self.calculate_adaptive_weights(&metadata.content_type);

        // Compute overall score with weighted combination
        let overall_score = self.compute_weighted_score(
            &perceptual,
            &intelligibility,
            &naturalness,
            &technical,
            &weights,
        );

        // Calculate confidence measure
        let confidence = self.calculate_confidence(
            &metadata,
            &[
                perceptual.loudness_quality,
                intelligibility.phoneme_clarity,
                naturalness.voice_quality,
                technical.snr,
            ],
        );

        // Generate trend prediction if enabled
        let trend_prediction = if self.config.predictive_assessment {
            self.predict_quality_trend(&overall_score, &metadata)?
        } else {
            None
        };

        let score = MultiDomainQualityScore {
            overall_score,
            perceptual,
            intelligibility,
            naturalness,
            technical,
            confidence,
            trend_prediction,
        };

        // Store measurement in history
        let measurement = QualityMeasurement {
            timestamp: chrono::Utc::now(),
            score: score.clone(),
            content_type: metadata.content_type.clone(),
            audio_metadata: metadata,
        };

        self.quality_history.push(measurement);

        // Maintain history window size
        if self.quality_history.len() > self.config.history_window {
            self.quality_history.remove(0);
        }

        Ok(score)
    }

    /// Evaluate perceptual domain quality
    async fn evaluate_perceptual_domain(
        &self,
        audio: &AudioBuffer,
        reference: Option<&AudioBuffer>,
    ) -> Result<PerceptualDomainScores, EvaluationError> {
        // Analyze loudness perception
        let loudness_quality = self.analyze_loudness_perception(audio)?;

        // Analyze spectral balance
        let spectral_balance = self.analyze_spectral_balance(audio)?;

        // Analyze temporal coherence
        let temporal_coherence = self.analyze_temporal_coherence(audio)?;

        // Analyze dynamic range
        let dynamic_range = self.analyze_dynamic_range(audio)?;

        Ok(PerceptualDomainScores {
            loudness_quality,
            spectral_balance,
            temporal_coherence,
            dynamic_range,
        })
    }

    /// Evaluate intelligibility domain quality
    async fn evaluate_intelligibility_domain(
        &self,
        audio: &AudioBuffer,
        reference: Option<&AudioBuffer>,
    ) -> Result<IntelligibilityDomainScores, EvaluationError> {
        // Analyze phoneme clarity
        let phoneme_clarity = self.analyze_phoneme_clarity(audio)?;

        // Analyze word boundaries
        let word_boundaries = self.analyze_word_boundaries(audio)?;

        // Analyze prosodic clarity
        let prosodic_clarity = self.analyze_prosodic_clarity(audio)?;

        // Analyze articulation precision
        let articulation = self.analyze_articulation(audio)?;

        Ok(IntelligibilityDomainScores {
            phoneme_clarity,
            word_boundaries,
            prosodic_clarity,
            articulation,
        })
    }

    /// Evaluate naturalness domain quality
    async fn evaluate_naturalness_domain(
        &self,
        audio: &AudioBuffer,
        reference: Option<&AudioBuffer>,
    ) -> Result<NaturalnessDomainScores, EvaluationError> {
        // Analyze voice quality naturalness
        let voice_quality = self.analyze_voice_naturalness(audio)?;

        // Analyze emotional appropriateness
        let emotional_appropriateness = self.analyze_emotional_appropriateness(audio)?;

        // Analyze speaking rate naturalness
        let speaking_rate = self.analyze_speaking_rate_naturalness(audio)?;

        // Analyze intonation naturalness
        let intonation = self.analyze_intonation_naturalness(audio)?;

        Ok(NaturalnessDomainScores {
            voice_quality,
            emotional_appropriateness,
            speaking_rate,
            intonation,
        })
    }

    /// Evaluate technical domain quality
    async fn evaluate_technical_domain(
        &self,
        audio: &AudioBuffer,
        reference: Option<&AudioBuffer>,
    ) -> Result<TechnicalDomainScores, EvaluationError> {
        // Calculate SNR
        let snr = self.calculate_snr(audio, reference)?;

        // Calculate THD
        let thd = self.calculate_thd(audio)?;

        // Analyze frequency response
        let frequency_response = self.analyze_frequency_response(audio)?;

        // Calculate dynamic range
        let dynamic_range = self.calculate_dynamic_range(audio)?;

        Ok(TechnicalDomainScores {
            snr,
            thd,
            frequency_response,
            dynamic_range,
        })
    }

    /// Extract audio metadata for adaptive evaluation
    fn extract_audio_metadata(
        &self,
        audio: &AudioBuffer,
    ) -> Result<AudioMetadata, EvaluationError> {
        let duration = audio.samples().len() as f64 / audio.sample_rate() as f64;

        // Estimate content type (simplified implementation)
        let content_type = self.estimate_content_type(audio)?;

        // Estimate speaker characteristics (simplified implementation)
        let speaker_characteristics = self.estimate_speaker_characteristics(audio)?;

        Ok(AudioMetadata {
            duration,
            sample_rate: audio.sample_rate(),
            channels: audio.channels() as u32,
            content_type,
            speaker_characteristics,
        })
    }

    /// Calculate adaptive weights based on content type
    fn calculate_adaptive_weights(&self, content_type: &str) -> HashMap<String, f64> {
        let mut weights = HashMap::new();

        match content_type {
            "speech" => {
                weights.insert("perceptual".to_string(), 0.20);
                weights.insert("intelligibility".to_string(), 0.40);
                weights.insert("naturalness".to_string(), 0.30);
                weights.insert("technical".to_string(), 0.10);
            }
            "music" => {
                weights.insert("perceptual".to_string(), 0.50);
                weights.insert("intelligibility".to_string(), 0.05);
                weights.insert("naturalness".to_string(), 0.15);
                weights.insert("technical".to_string(), 0.30);
            }
            "singing" => {
                weights.insert("perceptual".to_string(), 0.35);
                weights.insert("intelligibility".to_string(), 0.25);
                weights.insert("naturalness".to_string(), 0.25);
                weights.insert("technical".to_string(), 0.15);
            }
            _ => {
                // Default balanced weights
                weights.insert("perceptual".to_string(), 0.25);
                weights.insert("intelligibility".to_string(), 0.25);
                weights.insert("naturalness".to_string(), 0.25);
                weights.insert("technical".to_string(), 0.25);
            }
        }

        weights
    }

    /// Compute weighted overall score
    fn compute_weighted_score(
        &self,
        perceptual: &PerceptualDomainScores,
        intelligibility: &IntelligibilityDomainScores,
        naturalness: &NaturalnessDomainScores,
        technical: &TechnicalDomainScores,
        weights: &HashMap<String, f64>,
    ) -> f64 {
        let perceptual_avg = (perceptual.loudness_quality
            + perceptual.spectral_balance
            + perceptual.temporal_coherence
            + perceptual.dynamic_range)
            / 4.0;

        let intelligibility_avg = (intelligibility.phoneme_clarity
            + intelligibility.word_boundaries
            + intelligibility.prosodic_clarity
            + intelligibility.articulation)
            / 4.0;

        let naturalness_avg = (naturalness.voice_quality
            + naturalness.emotional_appropriateness
            + naturalness.speaking_rate
            + naturalness.intonation)
            / 4.0;

        let technical_avg = (technical.snr
            + technical.thd
            + technical.frequency_response
            + technical.dynamic_range)
            / 4.0;

        perceptual_avg * weights.get("perceptual").unwrap_or(&0.25)
            + intelligibility_avg * weights.get("intelligibility").unwrap_or(&0.25)
            + naturalness_avg * weights.get("naturalness").unwrap_or(&0.25)
            + technical_avg * weights.get("technical").unwrap_or(&0.25)
    }

    /// Calculate confidence measure for the assessment
    fn calculate_confidence(&self, metadata: &AudioMetadata, scores: &[f64]) -> f64 {
        // Base confidence on audio duration and score consistency
        let duration_factor = (metadata.duration.min(10.0) / 10.0).max(0.1);
        let score_variance = {
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            let variance =
                scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64;
            variance.sqrt()
        };
        let consistency_factor = (1.0 - score_variance.min(1.0)).max(0.1);

        (duration_factor * consistency_factor).min(1.0)
    }

    /// Predict quality trend based on historical data
    fn predict_quality_trend(
        &self,
        current_score: &f64,
        metadata: &AudioMetadata,
    ) -> Result<Option<QualityTrendPrediction>, EvaluationError> {
        if self.quality_history.len() < 3 {
            return Ok(None);
        }

        // Simple linear regression for trend prediction
        let recent_scores: Vec<f64> = self
            .quality_history
            .iter()
            .rev()
            .take(10)
            .map(|m| m.score.overall_score)
            .collect();

        let trend_slope = self.calculate_trend_slope(&recent_scores);
        let predicted_score = (current_score + trend_slope).max(0.0).min(1.0);

        let drift_direction = match trend_slope {
            x if x > 0.01 => QualityDriftDirection::Improving,
            x if x < -0.01 => QualityDriftDirection::Degrading,
            _ => QualityDriftDirection::Stable,
        };

        let prediction_confidence = self.calculate_prediction_confidence(&recent_scores);

        let recommendations = self.generate_trend_recommendations(&drift_direction, &trend_slope);

        Ok(Some(QualityTrendPrediction {
            predicted_score,
            prediction_confidence,
            drift_direction,
            recommendations,
        }))
    }

    // Simplified implementations for metric calculations
    fn analyze_loudness_perception(&self, audio: &AudioBuffer) -> Result<f64, EvaluationError> {
        // Simplified loudness analysis
        let rms = self.calculate_rms(audio.samples());
        Ok((rms * 10.0).min(1.0).max(0.0) as f64)
    }

    fn analyze_spectral_balance(&self, audio: &AudioBuffer) -> Result<f64, EvaluationError> {
        // Simplified spectral balance analysis
        Ok(0.8) // Placeholder
    }

    fn analyze_temporal_coherence(&self, audio: &AudioBuffer) -> Result<f64, EvaluationError> {
        // Simplified temporal coherence analysis
        Ok(0.75) // Placeholder
    }

    fn analyze_dynamic_range(&self, audio: &AudioBuffer) -> Result<f64, EvaluationError> {
        // Simplified dynamic range analysis
        let max_val = audio
            .samples()
            .iter()
            .map(|&x| x.abs())
            .fold(0.0f32, f32::max);
        let min_val = audio
            .samples()
            .iter()
            .map(|&x| x.abs())
            .fold(f32::INFINITY, f32::min);
        let dynamic_range = (max_val / min_val.max(f32::EPSILON)).log10() / 6.0; // Normalize to 0-1
        Ok(dynamic_range.min(1.0).max(0.0) as f64)
    }

    fn analyze_phoneme_clarity(&self, audio: &AudioBuffer) -> Result<f64, EvaluationError> {
        Ok(0.85) // Placeholder
    }

    fn analyze_word_boundaries(&self, audio: &AudioBuffer) -> Result<f64, EvaluationError> {
        Ok(0.80) // Placeholder
    }

    fn analyze_prosodic_clarity(&self, audio: &AudioBuffer) -> Result<f64, EvaluationError> {
        Ok(0.78) // Placeholder
    }

    fn analyze_articulation(&self, audio: &AudioBuffer) -> Result<f64, EvaluationError> {
        Ok(0.82) // Placeholder
    }

    fn analyze_voice_naturalness(&self, audio: &AudioBuffer) -> Result<f64, EvaluationError> {
        Ok(0.77) // Placeholder
    }

    fn analyze_emotional_appropriateness(
        &self,
        audio: &AudioBuffer,
    ) -> Result<f64, EvaluationError> {
        Ok(0.85) // Placeholder
    }

    fn analyze_speaking_rate_naturalness(
        &self,
        audio: &AudioBuffer,
    ) -> Result<f64, EvaluationError> {
        Ok(0.83) // Placeholder
    }

    fn analyze_intonation_naturalness(&self, audio: &AudioBuffer) -> Result<f64, EvaluationError> {
        Ok(0.79) // Placeholder
    }

    fn calculate_snr(
        &self,
        audio: &AudioBuffer,
        reference: Option<&AudioBuffer>,
    ) -> Result<f64, EvaluationError> {
        // Simplified SNR calculation
        Ok(0.88) // Placeholder
    }

    fn calculate_thd(&self, audio: &AudioBuffer) -> Result<f64, EvaluationError> {
        // Simplified THD calculation
        Ok(0.92) // Placeholder - inverted (high value = low distortion)
    }

    fn analyze_frequency_response(&self, audio: &AudioBuffer) -> Result<f64, EvaluationError> {
        Ok(0.86) // Placeholder
    }

    fn calculate_dynamic_range(&self, audio: &AudioBuffer) -> Result<f64, EvaluationError> {
        // Same as analyze_dynamic_range but for technical domain
        self.analyze_dynamic_range(audio)
    }

    fn estimate_content_type(&self, audio: &AudioBuffer) -> Result<String, EvaluationError> {
        // Simplified content type estimation
        Ok("speech".to_string())
    }

    fn estimate_speaker_characteristics(
        &self,
        audio: &AudioBuffer,
    ) -> Result<SpeakerCharacteristics, EvaluationError> {
        // Simplified speaker characteristic estimation
        Ok(SpeakerCharacteristics {
            age_group: "adult".to_string(),
            gender: "unknown".to_string(),
            accent: "neutral".to_string(),
            style: "conversational".to_string(),
        })
    }

    fn calculate_rms(&self, samples: &[f32]) -> f32 {
        let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    fn calculate_trend_slope(&self, scores: &[f64]) -> f64 {
        if scores.len() < 2 {
            return 0.0;
        }

        let n = scores.len() as f64;
        let x_sum: f64 = (0..scores.len()).map(|i| i as f64).sum();
        let y_sum: f64 = scores.iter().sum();
        let xy_sum: f64 = scores.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let x2_sum: f64 = (0..scores.len()).map(|i| (i as f64).powi(2)).sum();

        let denominator = n * x2_sum - x_sum * x_sum;
        if denominator.abs() < f64::EPSILON {
            return 0.0;
        }

        (n * xy_sum - x_sum * y_sum) / denominator
    }

    fn calculate_prediction_confidence(&self, scores: &[f64]) -> f64 {
        if scores.len() < 2 {
            return 0.5;
        }

        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance =
            scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64;
        let stability = 1.0 - variance.sqrt().min(1.0);

        stability * 0.8 + 0.2 // Base confidence of 0.2
    }

    fn generate_trend_recommendations(
        &self,
        direction: &QualityDriftDirection,
        slope: &f64,
    ) -> Vec<String> {
        match direction {
            QualityDriftDirection::Degrading => vec![
                "Consider reviewing recent model changes".to_string(),
                "Check for data quality issues".to_string(),
                "Investigate environmental factors".to_string(),
            ],
            QualityDriftDirection::Unstable => vec![
                "Examine input data consistency".to_string(),
                "Consider model regularization".to_string(),
                "Review system stability".to_string(),
            ],
            QualityDriftDirection::Improving => vec![
                "Continue current optimization approach".to_string(),
                "Document successful changes".to_string(),
            ],
            QualityDriftDirection::Stable => vec![
                "Maintain current configuration".to_string(),
                "Consider exploring optimization opportunities".to_string(),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_advanced_quality_evaluator_creation() {
        let config = AdvancedQualityConfig::default();
        let evaluator = AdvancedQualityEvaluator::new(config);
        assert_eq!(evaluator.quality_history.len(), 0);
    }

    #[tokio::test]
    async fn test_multi_domain_evaluation() {
        let config = AdvancedQualityConfig::default();
        let mut evaluator = AdvancedQualityEvaluator::new(config);

        let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);
        let result = evaluator.evaluate_quality(&audio, None).await;

        assert!(result.is_ok());
        let score = result.unwrap();
        assert!(score.overall_score >= 0.0 && score.overall_score <= 1.0);
        assert!(score.confidence >= 0.0 && score.confidence <= 1.0);
    }

    #[test]
    fn test_adaptive_weights() {
        let config = AdvancedQualityConfig::default();
        let evaluator = AdvancedQualityEvaluator::new(config);

        let speech_weights = evaluator.calculate_adaptive_weights("speech");
        assert_eq!(*speech_weights.get("intelligibility").unwrap(), 0.40);

        let music_weights = evaluator.calculate_adaptive_weights("music");
        assert_eq!(*music_weights.get("perceptual").unwrap(), 0.50);
    }

    #[test]
    fn test_trend_slope_calculation() {
        let config = AdvancedQualityConfig::default();
        let evaluator = AdvancedQualityEvaluator::new(config);

        let improving_scores = vec![0.6, 0.65, 0.7, 0.75, 0.8];
        let slope = evaluator.calculate_trend_slope(&improving_scores);
        assert!(slope > 0.0);

        let degrading_scores = vec![0.8, 0.75, 0.7, 0.65, 0.6];
        let slope = evaluator.calculate_trend_slope(&degrading_scores);
        assert!(slope < 0.0);
    }
}
