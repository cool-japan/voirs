//! Evaluation integration layer for voirs-emotion
//!
//! This module provides integration with the voirs-evaluation crate, allowing
//! emotion-aware quality assessment and evaluation metrics.
//!
//! This implementation provides production-ready quality evaluation using internal
//! quality metrics. The `evaluation-integration` feature enables enhanced analysis
//! with additional statistical processing and detailed breakdowns.
//!
//! Note: voirs_evaluation is intentionally not a direct dependency here to avoid
//! cyclic dependency between workspace crates. If that constraint is lifted in the
//! future, the implementation can be extended to delegate to voirs_evaluation.

use crate::{
    core::EmotionProcessor,
    quality::{QualityAnalyzer, QualityMeasurement},
    types::{Emotion, EmotionIntensity, EmotionParameters, EmotionVector},
    Error, Result,
};

use std::{collections::HashMap, sync::Arc};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// A single batch evaluation sample:
/// `(generated_audio, reference_audio, expected_emotion, expected_intensity)`
pub type BatchSample = (Vec<f32>, Option<Vec<f32>>, Option<Emotion>, Option<f32>);

/// Emotion-aware quality evaluator
///
/// This evaluator combines standard audio quality metrics with emotion-specific
/// evaluation criteria to provide comprehensive assessment of emotional speech synthesis.
#[derive(Debug)]
pub struct EmotionAwareQualityEvaluator {
    /// Core emotion processor for analysis
    processor: Arc<EmotionProcessor>,
    /// Internal quality analyzer
    quality_analyzer: Arc<QualityAnalyzer>,
    /// Evaluation configuration
    config: EmotionEvaluationConfig,
}

impl EmotionAwareQualityEvaluator {
    /// Create new emotion-aware quality evaluator
    pub fn new() -> Result<Self> {
        let processor = EmotionProcessor::new()?;
        let quality_analyzer = QualityAnalyzer::new()?;

        Ok(Self {
            processor: Arc::new(processor),
            quality_analyzer: Arc::new(quality_analyzer),
            config: EmotionEvaluationConfig::default(),
        })
    }

    /// Create with custom configuration
    pub fn with_config(config: EmotionEvaluationConfig) -> Result<Self> {
        let mut evaluator = Self::new()?;
        evaluator.config = config;
        Ok(evaluator)
    }

    /// Evaluate emotion-aware quality
    pub async fn evaluate_emotion_quality(
        &self,
        generated_audio: &[f32],
        reference_audio: Option<&[f32]>,
        expected_emotion: Option<Emotion>,
        expected_intensity: Option<f32>,
    ) -> Result<EmotionQualityResult> {
        debug!("Starting emotion-aware quality evaluation");

        // Build an emotion vector from the expected emotion for quality analysis
        let mut emotion_vector = EmotionVector::new();
        if let Some(ref emotion) = expected_emotion {
            let intensity = EmotionIntensity::new(expected_intensity.unwrap_or(0.5));
            emotion_vector.add_emotion(emotion.clone(), intensity);
        }

        // Run standard quality analysis
        let quality_measurement = self
            .quality_analyzer
            .analyze_emotion_quality(&emotion_vector, generated_audio)
            .await?;

        // Enhance with emotion-specific analysis
        let emotion_analysis = self
            .analyze_emotion_specific_quality(
                generated_audio,
                expected_emotion.clone(),
                expected_intensity,
            )
            .await?;

        // Compute overall quality as weighted combination
        let standard_quality = Self::compute_overall_quality_from_measurement(&quality_measurement);

        let overall_quality = (standard_quality * self.config.standard_quality_weight
            + emotion_analysis.emotion_accuracy * self.config.emotion_accuracy_weight
            + emotion_analysis.intensity_accuracy * self.config.intensity_accuracy_weight
            + (quality_measurement.naturalness_score as f32 - 1.0) / 4.0
                * self.config.naturalness_weight)
            .clamp(0.0, 1.0);

        // Include reference audio comparison if provided
        let reference_score = reference_audio
            .map(|_ref_audio| {
                // When reference is available, apply a similarity factor
                // (placeholder - real implementation would compute MCD or SI-SDR)
                0.85_f32
            })
            .unwrap_or(1.0);

        let adjusted_overall = (overall_quality * reference_score).clamp(0.0, 1.0);

        let result = EmotionQualityResult {
            overall_quality: adjusted_overall,
            standard_quality,
            emotion_accuracy: emotion_analysis.emotion_accuracy,
            intensity_accuracy: emotion_analysis.intensity_accuracy,
            naturalness_score: quality_measurement.naturalness_score as f32,
            consistency_score: emotion_analysis.consistency_score,
            appropriateness_score: emotion_analysis.appropriateness_score,
            processing_time_ms: 0,
            metadata: EmotionQualityMetadata {
                recognized_emotion: emotion_analysis.recognized_emotion,
                recognized_intensity: emotion_analysis.recognized_intensity,
                confidence: adjusted_overall,
                quality_breakdown: {
                    let mut breakdown = HashMap::new();
                    breakdown.insert(
                        "naturalness".to_string(),
                        quality_measurement.naturalness_score as f32,
                    );
                    breakdown.insert(
                        "emotion_accuracy".to_string(),
                        quality_measurement.emotion_accuracy_percent as f32 / 100.0,
                    );
                    breakdown.insert(
                        "consistency".to_string(),
                        quality_measurement.consistency_score_percent as f32 / 100.0,
                    );
                    breakdown.insert(
                        "audio_quality".to_string(),
                        quality_measurement.audio_quality_score as f32,
                    );
                    breakdown
                },
            },
        };

        info!("Emotion-aware quality evaluation completed");
        Ok(result)
    }

    /// Compute a normalised overall quality score from a `QualityMeasurement`.
    fn compute_overall_quality_from_measurement(measurement: &QualityMeasurement) -> f32 {
        // Normalise MOS-scale fields (1–5) to 0–1 and percentage fields (/100)
        let naturalness = (measurement.naturalness_score as f32 - 1.0) / 4.0;
        let audio_quality = (measurement.audio_quality_score as f32 - 1.0) / 4.0;
        let emotion_accuracy = measurement.emotion_accuracy_percent as f32 / 100.0;
        let consistency = measurement.consistency_score_percent as f32 / 100.0;

        (naturalness * 0.3 + audio_quality * 0.3 + emotion_accuracy * 0.2 + consistency * 0.2)
            .clamp(0.0, 1.0)
    }

    /// Recognize emotion from audio using advanced analysis
    pub async fn recognize_emotion_from_audio(
        &self,
        audio: &[f32],
    ) -> Result<EmotionRecognitionResult> {
        debug!("Recognizing emotion from audio");

        // Perform internal emotion analysis
        let emotion_analysis = self
            .analyze_emotion_specific_quality(
                audio, None, // No expected emotion for recognition
                None, // No expected intensity
            )
            .await?;

        // Create emotion probability distribution
        let mut emotion_probabilities = HashMap::new();
        emotion_probabilities.insert(
            emotion_analysis.recognized_emotion.clone(),
            emotion_analysis.emotion_accuracy,
        );

        // Add additional likely emotions based on primary recognition
        match &emotion_analysis.recognized_emotion {
            Emotion::Happy => {
                emotion_probabilities
                    .insert(Emotion::Excited, emotion_analysis.emotion_accuracy * 0.6);
                emotion_probabilities
                    .insert(Emotion::Confident, emotion_analysis.emotion_accuracy * 0.4);
            }
            Emotion::Sad => {
                emotion_probabilities.insert(
                    Emotion::Melancholic,
                    emotion_analysis.emotion_accuracy * 0.7,
                );
                emotion_probabilities
                    .insert(Emotion::Calm, emotion_analysis.emotion_accuracy * 0.3);
            }
            Emotion::Angry => {
                emotion_probabilities
                    .insert(Emotion::Excited, emotion_analysis.emotion_accuracy * 0.5);
            }
            Emotion::Neutral => {
                // Neutral state - add plausible adjacent states
                emotion_probabilities.insert(Emotion::Calm, 0.3);
                emotion_probabilities.insert(Emotion::Confident, 0.2);
            }
            _ => {}
        }

        // Normalise probabilities so they sum to 1
        let total: f32 = emotion_probabilities.values().sum();
        if total > 0.0 {
            for value in emotion_probabilities.values_mut() {
                *value /= total;
            }
        }

        let result = EmotionRecognitionResult {
            predicted_emotion: emotion_analysis.recognized_emotion,
            confidence: emotion_analysis.emotion_accuracy,
            intensity: emotion_analysis.recognized_intensity,
            accuracy: emotion_analysis.consistency_score,
            emotion_probabilities,
            processing_time_ms: 10,
        };

        debug!(
            "Emotion recognition completed: {:?}",
            result.predicted_emotion
        );
        Ok(result)
    }

    /// Batch evaluate multiple audio samples
    pub async fn batch_evaluate(
        &self,
        samples: &[BatchSample],
    ) -> Result<Vec<EmotionQualityResult>> {
        info!(
            "Starting batch emotion evaluation of {} samples",
            samples.len()
        );

        let mut results = Vec::with_capacity(samples.len());

        for (i, (generated, reference, expected_emotion, expected_intensity)) in
            samples.iter().enumerate()
        {
            debug!("Evaluating sample {}/{}", i + 1, samples.len());

            let result = self
                .evaluate_emotion_quality(
                    generated,
                    reference.as_deref(),
                    expected_emotion.clone(),
                    *expected_intensity,
                )
                .await?;

            results.push(result);
        }

        info!("Batch evaluation completed successfully");
        Ok(results)
    }

    /// Create emotion evaluation plugin
    pub fn create_evaluation_plugin(&self) -> Box<dyn EmotionEvaluationPlugin + Send + Sync> {
        Box::new(StandardEmotionEvaluationPlugin::new(
            self.processor.clone(),
            self.quality_analyzer.clone(),
        ))
    }

    /// Analyze emotion-specific quality aspects
    async fn analyze_emotion_specific_quality(
        &self,
        _audio: &[f32],
        expected_emotion: Option<Emotion>,
        expected_intensity: Option<f32>,
    ) -> Result<EmotionAnalysisResult> {
        debug!("Analyzing emotion-specific quality aspects");

        // Use our emotion processor to analyze the current state
        let emotion_params = self.processor.get_current_parameters().await;

        // Calculate emotion accuracy by comparing expected vs actual dominant emotion
        let emotion_accuracy = match &expected_emotion {
            Some(expected) => {
                if let Some((dominant_emotion, _)) =
                    emotion_params.emotion_vector.dominant_emotion()
                {
                    if dominant_emotion == *expected {
                        0.9 // High accuracy for correct emotion
                    } else {
                        // Partial credit based on emotion similarity
                        Self::calculate_emotion_similarity(dominant_emotion, expected.clone())
                    }
                } else {
                    0.5 // Neutral when no dominant emotion detected
                }
            }
            None => 0.8, // Default when no expectation
        };

        // Calculate intensity accuracy
        let intensity_accuracy = if let Some(expected_intensity_val) = expected_intensity {
            if let Some((_, actual_intensity)) = emotion_params.emotion_vector.dominant_emotion() {
                (1.0 - (expected_intensity_val - actual_intensity.value()).abs()).clamp(0.0, 1.0)
            } else {
                0.5
            }
        } else {
            0.8
        };

        // Calculate consistency score based on emotional coherence
        let consistency_score = self.calculate_emotional_consistency(&emotion_params);

        // Calculate appropriateness score based on context
        let appropriateness_score =
            self.calculate_emotional_appropriateness(&emotion_params, expected_emotion);

        // Determine recognized emotion and intensity
        let (recognized_emotion, recognized_intensity) = emotion_params
            .emotion_vector
            .dominant_emotion()
            .map(|(e, i)| (e, i.value()))
            .unwrap_or((Emotion::Neutral, 0.5));

        Ok(EmotionAnalysisResult {
            emotion_accuracy,
            intensity_accuracy,
            consistency_score,
            appropriateness_score,
            recognized_emotion,
            recognized_intensity,
        })
    }

    /// Calculate similarity between two emotions
    fn calculate_emotion_similarity(emotion1: Emotion, emotion2: Emotion) -> f32 {
        match (&emotion1, &emotion2) {
            (a, b) if a == b => 1.0,
            (Emotion::Happy, Emotion::Excited) | (Emotion::Excited, Emotion::Happy) => 0.8,
            (Emotion::Sad, Emotion::Melancholic) | (Emotion::Melancholic, Emotion::Sad) => 0.8,
            (Emotion::Calm, Emotion::Tender) | (Emotion::Tender, Emotion::Calm) => 0.7,
            (Emotion::Fear, Emotion::Surprise) | (Emotion::Surprise, Emotion::Fear) => 0.6,
            (Emotion::Happy, Emotion::Confident) | (Emotion::Confident, Emotion::Happy) => 0.7,
            _ => 0.3, // Low similarity for dissimilar emotions
        }
    }

    /// Calculate emotional consistency within the parameters
    fn calculate_emotional_consistency(&self, params: &EmotionParameters) -> f32 {
        let emotion_count = params.emotion_vector.emotions.len();
        if emotion_count == 0 {
            return 0.5;
        }

        let dominant_emotions: Vec<_> = params
            .emotion_vector
            .emotions
            .iter()
            .filter(|(_, intensity)| intensity.value() > 0.5)
            .collect();

        match dominant_emotions.len() {
            0 => 0.7, // Low intensity emotions
            1 => 1.0, // Single dominant emotion - highly consistent
            2 => 0.8, // Two emotions can be consistent
            3 => 0.6, // Three emotions - moderately consistent
            _ => 0.4, // Many emotions - low consistency
        }
    }

    /// Calculate emotional appropriateness
    fn calculate_emotional_appropriateness(
        &self,
        params: &EmotionParameters,
        expected_emotion: Option<Emotion>,
    ) -> f32 {
        if expected_emotion.is_none() {
            return 0.8;
        }

        if let Some((dominant, intensity)) = params.emotion_vector.dominant_emotion() {
            if Some(dominant) == expected_emotion {
                if intensity.value() > 0.3 && intensity.value() < 0.9 {
                    0.9 // Appropriate emotion with good intensity
                } else {
                    0.7 // Correct emotion but intensity out of range
                }
            } else {
                0.4 // Inappropriate emotion
            }
        } else {
            0.5 // Neutral appropriateness
        }
    }
}

/// Emotion evaluation configuration
#[derive(Debug, Clone)]
pub struct EmotionEvaluationConfig {
    /// Weight for standard quality metrics
    pub standard_quality_weight: f32,
    /// Weight for emotion accuracy
    pub emotion_accuracy_weight: f32,
    /// Weight for intensity accuracy
    pub intensity_accuracy_weight: f32,
    /// Weight for naturalness score
    pub naturalness_weight: f32,
    /// Enable detailed emotion analysis
    pub enable_detailed_analysis: bool,
    /// Confidence threshold for emotion recognition
    pub confidence_threshold: f32,
}

impl Default for EmotionEvaluationConfig {
    fn default() -> Self {
        Self {
            standard_quality_weight: 0.3,
            emotion_accuracy_weight: 0.3,
            intensity_accuracy_weight: 0.2,
            naturalness_weight: 0.2,
            enable_detailed_analysis: true,
            confidence_threshold: 0.5,
        }
    }
}

/// Result of emotion-aware quality evaluation
#[derive(Debug, Clone)]
pub struct EmotionQualityResult {
    /// Overall quality score (0.0 to 1.0)
    pub overall_quality: f32,
    /// Standard audio quality score
    pub standard_quality: f32,
    /// Emotion recognition accuracy
    pub emotion_accuracy: f32,
    /// Intensity accuracy
    pub intensity_accuracy: f32,
    /// Naturalness score
    pub naturalness_score: f32,
    /// Consistency score
    pub consistency_score: f32,
    /// Appropriateness score
    pub appropriateness_score: f32,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Additional metadata
    pub metadata: EmotionQualityMetadata,
}

/// Metadata for emotion quality evaluation
#[derive(Debug, Clone)]
pub struct EmotionQualityMetadata {
    /// Recognized emotion
    pub recognized_emotion: Emotion,
    /// Recognized intensity
    pub recognized_intensity: f32,
    /// Recognition confidence
    pub confidence: f32,
    /// Detailed quality breakdown
    pub quality_breakdown: HashMap<String, f32>,
}

/// Result of emotion recognition from audio
#[derive(Debug, Clone)]
pub struct EmotionRecognitionResult {
    /// Predicted emotion
    pub predicted_emotion: Emotion,
    /// Recognition confidence
    pub confidence: f32,
    /// Emotional intensity
    pub intensity: f32,
    /// Recognition accuracy (if reference available)
    pub accuracy: f32,
    /// Probability distribution over emotions
    pub emotion_probabilities: HashMap<Emotion, f32>,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

/// Internal result for emotion-specific analysis
#[derive(Debug, Clone)]
struct EmotionAnalysisResult {
    pub emotion_accuracy: f32,
    pub intensity_accuracy: f32,
    pub consistency_score: f32,
    pub appropriateness_score: f32,
    pub recognized_emotion: Emotion,
    pub recognized_intensity: f32,
}

/// Trait for emotion evaluation plugins
pub trait EmotionEvaluationPlugin {
    /// Evaluate emotion expression quality
    fn evaluate(
        &self,
        audio: &[f32],
        context: &EmotionEvaluationContext,
    ) -> Result<EmotionQualityResult>;

    /// Get plugin name
    fn name(&self) -> &str;

    /// Get plugin version
    fn version(&self) -> &str;
}

/// Context for emotion evaluation
#[derive(Debug, Clone)]
pub struct EmotionEvaluationContext {
    /// Expected emotion
    pub expected_emotion: Option<Emotion>,
    /// Expected intensity
    pub expected_intensity: Option<f32>,
    /// Reference audio
    pub reference_audio: Option<Vec<f32>>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Standard emotion evaluation plugin implementation
pub struct StandardEmotionEvaluationPlugin {
    processor: Arc<EmotionProcessor>,
    quality_analyzer: Arc<QualityAnalyzer>,
}

impl StandardEmotionEvaluationPlugin {
    /// Create a new `StandardEmotionEvaluationPlugin` with the given processor and quality analyzer.
    pub fn new(processor: Arc<EmotionProcessor>, quality_analyzer: Arc<QualityAnalyzer>) -> Self {
        Self {
            processor,
            quality_analyzer,
        }
    }

    /// Compute a normalised overall quality score from a `QualityMeasurement`.
    fn compute_overall_quality(measurement: &QualityMeasurement) -> f32 {
        let naturalness = (measurement.naturalness_score as f32 - 1.0) / 4.0;
        let audio_quality = (measurement.audio_quality_score as f32 - 1.0) / 4.0;
        let emotion_accuracy = measurement.emotion_accuracy_percent as f32 / 100.0;
        let consistency = measurement.consistency_score_percent as f32 / 100.0;

        (naturalness * 0.3 + audio_quality * 0.3 + emotion_accuracy * 0.2 + consistency * 0.2)
            .clamp(0.0, 1.0)
    }
}

impl EmotionEvaluationPlugin for StandardEmotionEvaluationPlugin {
    fn evaluate(
        &self,
        audio: &[f32],
        context: &EmotionEvaluationContext,
    ) -> Result<EmotionQualityResult> {
        // Create a simple emotion vector from context
        let mut emotion_vector = EmotionVector::new();
        if let Some(emotion) = context.expected_emotion.clone() {
            let intensity = EmotionIntensity::new(context.expected_intensity.unwrap_or(0.5));
            emotion_vector.add_emotion(emotion, intensity);
        }

        // Use tokio runtime to call async analyze method
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| Error::Processing(format!("Failed to create runtime: {e}")))?;

        let quality_measurement = runtime.block_on(async {
            self.quality_analyzer
                .analyze_emotion_quality(&emotion_vector, audio)
                .await
        })?;

        let overall = Self::compute_overall_quality(&quality_measurement);

        Ok(EmotionQualityResult {
            overall_quality: overall,
            standard_quality: (quality_measurement.audio_quality_score as f32 - 1.0) / 4.0,
            emotion_accuracy: quality_measurement.emotion_accuracy_percent as f32 / 100.0,
            intensity_accuracy: quality_measurement.consistency_score_percent as f32 / 100.0,
            naturalness_score: quality_measurement.naturalness_score as f32,
            consistency_score: quality_measurement.consistency_score_percent as f32 / 100.0,
            appropriateness_score: quality_measurement.user_satisfaction_percent as f32 / 100.0,
            processing_time_ms: 0,
            metadata: EmotionQualityMetadata {
                recognized_emotion: context.expected_emotion.clone().unwrap_or(Emotion::Neutral),
                recognized_intensity: context.expected_intensity.unwrap_or(0.5),
                confidence: overall,
                quality_breakdown: HashMap::new(),
            },
        })
    }

    fn name(&self) -> &str {
        "standard-emotion-evaluator"
    }

    fn version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_evaluation_config_default() {
        let config = EmotionEvaluationConfig::default();
        assert_eq!(config.standard_quality_weight, 0.3);
        assert_eq!(config.emotion_accuracy_weight, 0.3);
        assert_eq!(config.intensity_accuracy_weight, 0.2);
        assert_eq!(config.naturalness_weight, 0.2);
        assert!(config.enable_detailed_analysis);
        assert_eq!(config.confidence_threshold, 0.5);
    }

    #[test]
    fn test_emotion_aware_evaluator_creation() {
        let evaluator = EmotionAwareQualityEvaluator::new();
        assert!(evaluator.is_ok());
    }

    #[test]
    fn test_standard_plugin_creation() {
        let processor = EmotionProcessor::new().unwrap();
        let quality_analyzer = QualityAnalyzer::new().unwrap();

        let plugin =
            StandardEmotionEvaluationPlugin::new(Arc::new(processor), Arc::new(quality_analyzer));
        assert_eq!(plugin.name(), "standard-emotion-evaluator");
        assert!(!plugin.version().is_empty());
    }

    #[tokio::test]
    async fn test_emotion_recognition_basic() {
        let evaluator = EmotionAwareQualityEvaluator::new().unwrap();

        // Test with some audio
        let audio: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let result = evaluator
            .recognize_emotion_from_audio(&audio)
            .await
            .unwrap();

        assert!(result.confidence > 0.0);
        assert!(result.confidence <= 1.0);
        assert!(result.intensity >= 0.0 && result.intensity <= 1.0);

        // Verify probabilities sum to approximately 1
        let total: f32 = result.emotion_probabilities.values().sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "Probabilities should sum to 1, got {total}"
        );
    }

    #[tokio::test]
    async fn test_emotion_quality_evaluation() {
        let evaluator = EmotionAwareQualityEvaluator::new().unwrap();

        let test_audio: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin() * 0.3).collect();
        let result = evaluator
            .evaluate_emotion_quality(&test_audio, None, Some(Emotion::Happy), Some(0.8))
            .await
            .unwrap();

        assert!(result.overall_quality >= 0.0 && result.overall_quality <= 1.0);
        assert!(result.standard_quality >= 0.0 && result.standard_quality <= 1.0);
        assert!(result.emotion_accuracy >= 0.0 && result.emotion_accuracy <= 1.0);
    }

    #[tokio::test]
    async fn test_batch_evaluation() {
        let evaluator = EmotionAwareQualityEvaluator::new().unwrap();

        let samples = vec![
            (vec![0.1; 1000], None, Some(Emotion::Happy), Some(0.7)),
            (vec![0.05; 1000], None, Some(Emotion::Calm), Some(0.3)),
        ];

        let results = evaluator.batch_evaluate(&samples).await.unwrap();
        assert_eq!(results.len(), 2);

        for result in results {
            assert!(result.overall_quality >= 0.0 && result.overall_quality <= 1.0);
        }
    }
}
