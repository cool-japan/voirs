//! Perceptual Validation System for Human Evaluation of Emotional Expression
//!
//! This module provides tools for collecting and analyzing human evaluations
//! of emotional expression quality in synthesized speech.

use crate::types::{Emotion, EmotionIntensity, EmotionParameters};
use crate::Error;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// Configuration for perceptual validation studies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualValidationConfig {
    /// Maximum duration for evaluation sessions
    pub max_session_duration: Duration,
    /// Minimum number of evaluators required
    pub min_evaluators: usize,
    /// Target emotions to evaluate
    pub target_emotions: Vec<Emotion>,
    /// Audio sample duration in seconds
    pub sample_duration: f32,
    /// Random presentation order
    pub randomize_order: bool,
    /// Allow partial evaluations
    pub allow_partial: bool,
}

impl Default for PerceptualValidationConfig {
    fn default() -> Self {
        Self {
            max_session_duration: Duration::from_secs(1800), // 30 minutes
            min_evaluators: 5,
            target_emotions: vec![
                Emotion::Happy,
                Emotion::Sad,
                Emotion::Angry,
                Emotion::Fear,
                Emotion::Surprise,
                Emotion::Neutral,
            ],
            sample_duration: 5.0,
            randomize_order: true,
            allow_partial: true,
        }
    }
}

/// Evaluation criteria for perceptual validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationCriteria {
    /// Naturalness rating (1-10)
    pub naturalness: u8,
    /// Appropriateness of emotion (1-10)
    pub appropriateness: u8,
    /// Perceived emotion intensity (1-10)
    pub perceived_intensity: u8,
    /// Overall quality (1-10)
    pub overall_quality: u8,
    /// Emotion recognition accuracy (correctly identified)
    pub correct_emotion: bool,
    /// Additional comments
    pub comments: Option<String>,
}

impl EvaluationCriteria {
    /// Validate evaluation scores are within bounds
    pub fn validate(&self) -> crate::Result<()> {
        let scores = [
            self.naturalness,
            self.appropriateness,
            self.perceived_intensity,
            self.overall_quality,
        ];

        for &score in &scores {
            if score < 1 || score > 10 {
                return Err(Error::Validation(format!(
                    "Evaluation scores must be between 1-10, got {}",
                    score
                )));
            }
        }
        Ok(())
    }

    /// Calculate composite quality score
    pub fn composite_score(&self) -> f32 {
        let base_score = (self.naturalness
            + self.appropriateness
            + self.perceived_intensity
            + self.overall_quality) as f32
            / 4.0;

        // Bonus for correct emotion recognition
        if self.correct_emotion {
            base_score * 1.1
        } else {
            base_score * 0.8
        }
    }
}

/// Individual evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualEvaluation {
    /// Unique evaluation ID
    pub evaluation_id: Uuid,
    /// Evaluator ID (anonymous)
    pub evaluator_id: String,
    /// Target emotion being evaluated
    pub target_emotion: Emotion,
    /// Target intensity
    pub target_intensity: EmotionIntensity,
    /// Emotion parameters used
    pub emotion_parameters: EmotionParameters,
    /// Evaluation results
    pub criteria: EvaluationCriteria,
    /// Evaluation timestamp
    pub timestamp: SystemTime,
    /// Duration taken for evaluation
    pub evaluation_duration: Duration,
    /// Audio sample metadata
    pub sample_metadata: HashMap<String, String>,
}

impl PerceptualEvaluation {
    /// Create new evaluation
    pub fn new(
        evaluator_id: String,
        target_emotion: Emotion,
        target_intensity: EmotionIntensity,
        emotion_parameters: EmotionParameters,
        criteria: EvaluationCriteria,
        evaluation_duration: Duration,
    ) -> crate::Result<Self> {
        criteria.validate()?;

        Ok(Self {
            evaluation_id: Uuid::new_v4(),
            evaluator_id,
            target_emotion,
            target_intensity,
            emotion_parameters,
            criteria,
            timestamp: SystemTime::now(),
            evaluation_duration,
            sample_metadata: HashMap::new(),
        })
    }

    /// Add metadata to evaluation
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.sample_metadata.insert(key, value);
        self
    }

    /// Check if evaluation is valid
    pub fn is_valid(&self) -> bool {
        self.criteria.validate().is_ok() && self.evaluation_duration <= Duration::from_secs(300)
        // Max 5 minutes per evaluation
    }
}

/// Aggregated statistics from multiple evaluations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStatistics {
    /// Number of evaluations
    pub evaluation_count: usize,
    /// Number of unique evaluators
    pub evaluator_count: usize,
    /// Average naturalness score
    pub avg_naturalness: f32,
    /// Average appropriateness score
    pub avg_appropriateness: f32,
    /// Average perceived intensity
    pub avg_perceived_intensity: f32,
    /// Average overall quality
    pub avg_overall_quality: f32,
    /// Emotion recognition accuracy
    pub recognition_accuracy: f32,
    /// Average composite score
    pub avg_composite_score: f32,
    /// Standard deviation of scores
    pub score_std_dev: f32,
    /// Inter-evaluator agreement (Cronbach's alpha)
    pub inter_evaluator_agreement: f32,
}

/// Perceptual validation study manager
#[derive(Debug)]
pub struct PerceptualValidationStudy {
    /// Study configuration
    config: PerceptualValidationConfig,
    /// Collected evaluations
    evaluations: Vec<PerceptualEvaluation>,
    /// Study start time
    start_time: SystemTime,
    /// Study ID
    study_id: Uuid,
}

impl PerceptualValidationStudy {
    /// Create new validation study
    pub fn new(config: PerceptualValidationConfig) -> Self {
        Self {
            config,
            evaluations: Vec::new(),
            start_time: SystemTime::now(),
            study_id: Uuid::new_v4(),
        }
    }

    /// Add evaluation to study
    pub fn add_evaluation(&mut self, evaluation: PerceptualEvaluation) -> crate::Result<()> {
        // Check session duration limit
        if self.start_time.elapsed().unwrap_or(Duration::ZERO) > self.config.max_session_duration {
            return Err(Error::Validation(
                "Study session duration exceeded".to_string(),
            ));
        }

        if evaluation.is_valid() {
            self.evaluations.push(evaluation);
            Ok(())
        } else {
            Err(Error::Validation("Invalid evaluation data".to_string()))
        }
    }

    /// Get evaluations for specific emotion
    pub fn get_evaluations_for_emotion(&self, emotion: &Emotion) -> Vec<&PerceptualEvaluation> {
        self.evaluations
            .iter()
            .filter(|eval| eval.target_emotion == *emotion)
            .collect()
    }

    /// Calculate validation statistics
    pub fn calculate_statistics(&self) -> crate::Result<ValidationStatistics> {
        if self.evaluations.is_empty() {
            return Err(Error::Validation(
                "No evaluations available for statistics".to_string(),
            ));
        }

        let count = self.evaluations.len();
        let evaluator_count = self
            .evaluations
            .iter()
            .map(|e| e.evaluator_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .len();

        // Calculate averages
        let sum_naturalness: u32 = self
            .evaluations
            .iter()
            .map(|e| e.criteria.naturalness as u32)
            .sum();
        let sum_appropriateness: u32 = self
            .evaluations
            .iter()
            .map(|e| e.criteria.appropriateness as u32)
            .sum();
        let sum_perceived_intensity: u32 = self
            .evaluations
            .iter()
            .map(|e| e.criteria.perceived_intensity as u32)
            .sum();
        let sum_overall_quality: u32 = self
            .evaluations
            .iter()
            .map(|e| e.criteria.overall_quality as u32)
            .sum();

        let avg_naturalness = sum_naturalness as f32 / count as f32;
        let avg_appropriateness = sum_appropriateness as f32 / count as f32;
        let avg_perceived_intensity = sum_perceived_intensity as f32 / count as f32;
        let avg_overall_quality = sum_overall_quality as f32 / count as f32;

        // Calculate recognition accuracy
        let correct_recognitions = self
            .evaluations
            .iter()
            .filter(|e| e.criteria.correct_emotion)
            .count();
        let recognition_accuracy = correct_recognitions as f32 / count as f32;

        // Calculate composite scores
        let composite_scores: Vec<f32> = self
            .evaluations
            .iter()
            .map(|e| e.criteria.composite_score())
            .collect();
        let avg_composite_score = composite_scores.iter().sum::<f32>() / count as f32;

        // Calculate standard deviation
        let variance = composite_scores
            .iter()
            .map(|score| {
                let diff = score - avg_composite_score;
                diff * diff
            })
            .sum::<f32>()
            / count as f32;
        let score_std_dev = variance.sqrt();

        // Calculate inter-evaluator agreement (simplified Cronbach's alpha)
        let inter_evaluator_agreement = self.calculate_inter_evaluator_agreement();

        Ok(ValidationStatistics {
            evaluation_count: count,
            evaluator_count,
            avg_naturalness,
            avg_appropriateness,
            avg_perceived_intensity,
            avg_overall_quality,
            recognition_accuracy,
            avg_composite_score,
            score_std_dev,
            inter_evaluator_agreement,
        })
    }

    /// Calculate inter-evaluator agreement
    fn calculate_inter_evaluator_agreement(&self) -> f32 {
        // Simplified calculation - in production would use proper Cronbach's alpha
        if self.evaluations.len() < 2 {
            return 1.0;
        }

        let evaluator_scores: HashMap<String, Vec<f32>> =
            self.evaluations
                .iter()
                .fold(HashMap::new(), |mut acc, eval| {
                    acc.entry(eval.evaluator_id.clone())
                        .or_default()
                        .push(eval.criteria.composite_score());
                    acc
                });

        if evaluator_scores.len() < 2 {
            return 1.0;
        }

        // Calculate pairwise correlations (simplified)
        let evaluators: Vec<_> = evaluator_scores.keys().collect();
        let mut correlations = Vec::new();

        for i in 0..evaluators.len() {
            for j in (i + 1)..evaluators.len() {
                if let (Some(scores1), Some(scores2)) = (
                    evaluator_scores.get(evaluators[i]),
                    evaluator_scores.get(evaluators[j]),
                ) {
                    let correlation = self.calculate_correlation(scores1, scores2);
                    correlations.push(correlation);
                }
            }
        }

        correlations.iter().sum::<f32>() / correlations.len() as f32
    }

    /// Calculate simple correlation coefficient
    fn calculate_correlation(&self, scores1: &[f32], scores2: &[f32]) -> f32 {
        if scores1.is_empty() || scores2.is_empty() {
            return 0.0;
        }

        let min_len = scores1.len().min(scores2.len());
        if min_len < 2 {
            return 0.0;
        }

        let mean1 = scores1.iter().take(min_len).sum::<f32>() / min_len as f32;
        let mean2 = scores2.iter().take(min_len).sum::<f32>() / min_len as f32;

        let mut numerator = 0.0;
        let mut sum_sq1 = 0.0;
        let mut sum_sq2 = 0.0;

        for i in 0..min_len {
            let diff1 = scores1[i] - mean1;
            let diff2 = scores2[i] - mean2;
            numerator += diff1 * diff2;
            sum_sq1 += diff1 * diff1;
            sum_sq2 += diff2 * diff2;
        }

        let denominator = (sum_sq1 * sum_sq2).sqrt();
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Export study results to JSON
    pub fn export_results(&self) -> crate::Result<String> {
        let statistics = self.calculate_statistics()?;

        let export_data = serde_json::json!({
            "study_id": self.study_id,
            "config": self.config,
            "statistics": statistics,
            "evaluations": self.evaluations,
            "export_timestamp": SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs()
        });

        serde_json::to_string_pretty(&export_data).map_err(|e| Error::Serialization(e))
    }

    /// Check if study meets minimum requirements
    pub fn is_study_complete(&self) -> bool {
        let unique_evaluators = self
            .evaluations
            .iter()
            .map(|e| e.evaluator_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .len();

        unique_evaluators >= self.config.min_evaluators && !self.evaluations.is_empty()
    }

    /// Get study progress summary
    pub fn get_progress_summary(&self) -> HashMap<String, serde_json::Value> {
        let mut summary = HashMap::new();

        summary.insert(
            "total_evaluations".to_string(),
            serde_json::Value::from(self.evaluations.len()),
        );

        let unique_evaluators = self
            .evaluations
            .iter()
            .map(|e| e.evaluator_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .len();
        summary.insert(
            "unique_evaluators".to_string(),
            serde_json::Value::from(unique_evaluators),
        );

        let required_evaluators = self.config.min_evaluators;
        summary.insert(
            "required_evaluators".to_string(),
            serde_json::Value::from(required_evaluators),
        );

        let progress_percent = if required_evaluators > 0 {
            (unique_evaluators as f32 / required_evaluators as f32 * 100.0).min(100.0)
        } else {
            100.0
        };
        summary.insert(
            "progress_percent".to_string(),
            serde_json::Value::from(progress_percent),
        );

        summary.insert(
            "is_complete".to_string(),
            serde_json::Value::from(self.is_study_complete()),
        );

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EmotionDimensions, EmotionIntensity, EmotionParameters, EmotionVector};

    #[test]
    fn test_evaluation_criteria_validation() {
        let valid_criteria = EvaluationCriteria {
            naturalness: 8,
            appropriateness: 7,
            perceived_intensity: 6,
            overall_quality: 8,
            correct_emotion: true,
            comments: None,
        };

        assert!(valid_criteria.validate().is_ok());
        assert!((valid_criteria.composite_score() - 7.975).abs() < 0.01); // (8+7+6+8)/4 * 1.1 with floating point tolerance

        let invalid_criteria = EvaluationCriteria {
            naturalness: 11, // Invalid score
            appropriateness: 7,
            perceived_intensity: 6,
            overall_quality: 8,
            correct_emotion: true,
            comments: None,
        };

        assert!(invalid_criteria.validate().is_err());
    }

    #[test]
    fn test_perceptual_evaluation_creation() {
        let criteria = EvaluationCriteria {
            naturalness: 8,
            appropriateness: 7,
            perceived_intensity: 6,
            overall_quality: 8,
            correct_emotion: true,
            comments: Some("Good emotional expression".to_string()),
        };

        let mut emotion_vector = EmotionVector::new();
        emotion_vector.dimensions = EmotionDimensions::new(0.8, 0.6, 0.7);
        let params = EmotionParameters::new(emotion_vector);

        let evaluation = PerceptualEvaluation::new(
            "evaluator_001".to_string(),
            Emotion::Happy,
            EmotionIntensity::new(0.8),
            params,
            criteria,
            Duration::from_secs(45),
        );

        assert!(evaluation.is_ok());
        let eval = evaluation.unwrap();
        assert_eq!(eval.evaluator_id, "evaluator_001");
        assert_eq!(eval.target_emotion, Emotion::Happy);
        assert!(eval.is_valid());
    }

    #[test]
    fn test_validation_study() {
        let config = PerceptualValidationConfig {
            min_evaluators: 2,
            ..Default::default()
        };

        let mut study = PerceptualValidationStudy::new(config);

        // Add some evaluations
        let criteria1 = EvaluationCriteria {
            naturalness: 8,
            appropriateness: 7,
            perceived_intensity: 6,
            overall_quality: 8,
            correct_emotion: true,
            comments: None,
        };

        let criteria2 = EvaluationCriteria {
            naturalness: 7,
            appropriateness: 8,
            perceived_intensity: 7,
            overall_quality: 7,
            correct_emotion: false,
            comments: None,
        };

        let mut emotion_vector = EmotionVector::new();
        emotion_vector.dimensions = EmotionDimensions::new(0.8, 0.6, 0.7);
        let params = EmotionParameters::new(emotion_vector);

        let eval1 = PerceptualEvaluation::new(
            "evaluator_001".to_string(),
            Emotion::Happy,
            EmotionIntensity::new(0.8),
            params.clone(),
            criteria1,
            Duration::from_secs(45),
        )
        .unwrap();

        let eval2 = PerceptualEvaluation::new(
            "evaluator_002".to_string(),
            Emotion::Happy,
            EmotionIntensity::new(0.8),
            params,
            criteria2,
            Duration::from_secs(50),
        )
        .unwrap();

        assert!(study.add_evaluation(eval1).is_ok());
        assert!(study.add_evaluation(eval2).is_ok());
        assert!(study.is_study_complete());

        let stats = study.calculate_statistics().unwrap();
        assert_eq!(stats.evaluation_count, 2);
        assert_eq!(stats.evaluator_count, 2);
        assert_eq!(stats.recognition_accuracy, 0.5); // 1 out of 2 correct
    }

    #[test]
    fn test_emotion_filtering() {
        let config = PerceptualValidationConfig::default();
        let mut study = PerceptualValidationStudy::new(config);

        let mut emotion_vector = EmotionVector::new();
        emotion_vector.dimensions = EmotionDimensions::new(0.8, 0.6, 0.7);
        let params = EmotionParameters::new(emotion_vector);

        let criteria = EvaluationCriteria {
            naturalness: 8,
            appropriateness: 7,
            perceived_intensity: 6,
            overall_quality: 8,
            correct_emotion: true,
            comments: None,
        };

        // Add evaluations for different emotions
        let eval_happy = PerceptualEvaluation::new(
            "evaluator_001".to_string(),
            Emotion::Happy,
            EmotionIntensity::new(0.8),
            params.clone(),
            criteria.clone(),
            Duration::from_secs(45),
        )
        .unwrap();

        let eval_sad = PerceptualEvaluation::new(
            "evaluator_001".to_string(),
            Emotion::Sad,
            EmotionIntensity::new(0.6),
            params,
            criteria,
            Duration::from_secs(40),
        )
        .unwrap();

        study.add_evaluation(eval_happy).unwrap();
        study.add_evaluation(eval_sad).unwrap();

        let happy_evals = study.get_evaluations_for_emotion(&Emotion::Happy);
        let sad_evals = study.get_evaluations_for_emotion(&Emotion::Sad);

        assert_eq!(happy_evals.len(), 1);
        assert_eq!(sad_evals.len(), 1);
        assert_eq!(happy_evals[0].target_emotion, Emotion::Happy);
        assert_eq!(sad_evals[0].target_emotion, Emotion::Sad);
    }

    #[test]
    fn test_validation_config_default() {
        let config = PerceptualValidationConfig::default();

        assert_eq!(config.max_session_duration, Duration::from_secs(1800));
        assert_eq!(config.min_evaluators, 5);
        assert_eq!(config.target_emotions.len(), 6);
        assert_eq!(config.sample_duration, 5.0);
        assert!(config.randomize_order);
        assert!(config.allow_partial);

        // Check that default emotions include expected ones
        assert!(config.target_emotions.contains(&Emotion::Happy));
        assert!(config.target_emotions.contains(&Emotion::Sad));
        assert!(config.target_emotions.contains(&Emotion::Neutral));
    }

    #[test]
    fn test_evaluation_criteria_composite_score_variations() {
        // Test with correct emotion recognition
        let criteria_correct = EvaluationCriteria {
            naturalness: 8,
            appropriateness: 6,
            perceived_intensity: 7,
            overall_quality: 9,
            correct_emotion: true,
            comments: None,
        };

        let score_correct = criteria_correct.composite_score();
        let expected_correct = (8.0 + 6.0 + 7.0 + 9.0) / 4.0 * 1.1; // 7.5 * 1.1 = 8.25
        assert!((score_correct - expected_correct).abs() < 0.01);

        // Test with incorrect emotion recognition
        let criteria_incorrect = EvaluationCriteria {
            naturalness: 8,
            appropriateness: 6,
            perceived_intensity: 7,
            overall_quality: 9,
            correct_emotion: false,
            comments: None,
        };

        let score_incorrect = criteria_incorrect.composite_score();
        let expected_incorrect = (8.0 + 6.0 + 7.0 + 9.0) / 4.0 * 0.8; // 7.5 * 0.8 = 6.0
        assert!((score_incorrect - expected_incorrect).abs() < 0.01);

        // Verify correct recognition gives higher score
        assert!(score_correct > score_incorrect);
    }

    #[test]
    fn test_evaluation_criteria_boundary_validation() {
        // Test minimum valid scores (1)
        let min_criteria = EvaluationCriteria {
            naturalness: 1,
            appropriateness: 1,
            perceived_intensity: 1,
            overall_quality: 1,
            correct_emotion: false,
            comments: None,
        };
        assert!(min_criteria.validate().is_ok());

        // Test maximum valid scores (10)
        let max_criteria = EvaluationCriteria {
            naturalness: 10,
            appropriateness: 10,
            perceived_intensity: 10,
            overall_quality: 10,
            correct_emotion: true,
            comments: None,
        };
        assert!(max_criteria.validate().is_ok());

        // Test invalid low score (0)
        let invalid_low = EvaluationCriteria {
            naturalness: 0, // Invalid
            appropriateness: 5,
            perceived_intensity: 5,
            overall_quality: 5,
            correct_emotion: true,
            comments: None,
        };
        assert!(invalid_low.validate().is_err());

        // Test invalid high score (11)
        let invalid_high = EvaluationCriteria {
            naturalness: 5,
            appropriateness: 11, // Invalid
            perceived_intensity: 5,
            overall_quality: 5,
            correct_emotion: true,
            comments: None,
        };
        assert!(invalid_high.validate().is_err());
    }

    #[test]
    fn test_perceptual_evaluation_with_metadata() {
        let criteria = EvaluationCriteria {
            naturalness: 7,
            appropriateness: 8,
            perceived_intensity: 6,
            overall_quality: 7,
            correct_emotion: true,
            comments: Some("Clear emotional expression".to_string()),
        };

        let emotion_vector = EmotionVector::new();
        let params = EmotionParameters::new(emotion_vector);

        let evaluation = PerceptualEvaluation::new(
            "evaluator_meta".to_string(),
            Emotion::Confident,
            EmotionIntensity::new(0.9),
            params,
            criteria,
            Duration::from_secs(30),
        )
        .unwrap()
        .with_metadata("audio_format".to_string(), "wav".to_string())
        .with_metadata("sample_rate".to_string(), "44100".to_string());

        assert_eq!(evaluation.sample_metadata.len(), 2);
        assert_eq!(evaluation.sample_metadata["audio_format"], "wav");
        assert_eq!(evaluation.sample_metadata["sample_rate"], "44100");
        assert!(evaluation.is_valid());
    }

    #[test]
    fn test_perceptual_evaluation_validity_duration_check() {
        let criteria = EvaluationCriteria {
            naturalness: 8,
            appropriateness: 7,
            perceived_intensity: 6,
            overall_quality: 8,
            correct_emotion: true,
            comments: None,
        };

        let emotion_vector = EmotionVector::new();
        let params = EmotionParameters::new(emotion_vector);

        // Valid duration (under 5 minutes)
        let valid_eval = PerceptualEvaluation::new(
            "evaluator_time".to_string(),
            Emotion::Happy,
            EmotionIntensity::new(0.7),
            params.clone(),
            criteria.clone(),
            Duration::from_secs(299), // Just under 5 minutes
        )
        .unwrap();
        assert!(valid_eval.is_valid());

        // Invalid duration (over 5 minutes)
        let invalid_eval = PerceptualEvaluation::new(
            "evaluator_time".to_string(),
            Emotion::Happy,
            EmotionIntensity::new(0.7),
            params,
            criteria,
            Duration::from_secs(301), // Over 5 minutes
        )
        .unwrap();
        assert!(!invalid_eval.is_valid());
    }

    #[test]
    fn test_validation_study_session_duration_limit() {
        let config = PerceptualValidationConfig {
            max_session_duration: Duration::from_secs(1), // Very short for testing
            min_evaluators: 1,
            ..Default::default()
        };

        let mut study = PerceptualValidationStudy::new(config);

        // Wait to exceed session duration
        std::thread::sleep(Duration::from_millis(1100));

        let criteria = EvaluationCriteria {
            naturalness: 8,
            appropriateness: 7,
            perceived_intensity: 6,
            overall_quality: 8,
            correct_emotion: true,
            comments: None,
        };

        let emotion_vector = EmotionVector::new();
        let params = EmotionParameters::new(emotion_vector);

        let evaluation = PerceptualEvaluation::new(
            "late_evaluator".to_string(),
            Emotion::Happy,
            EmotionIntensity::new(0.8),
            params,
            criteria,
            Duration::from_secs(30),
        )
        .unwrap();

        // Should fail due to session duration exceeded
        assert!(study.add_evaluation(evaluation).is_err());
    }

    #[test]
    fn test_validation_study_invalid_evaluation_rejection() {
        let config = PerceptualValidationConfig::default();
        let mut study = PerceptualValidationStudy::new(config);

        let invalid_criteria = EvaluationCriteria {
            naturalness: 12, // Invalid score
            appropriateness: 7,
            perceived_intensity: 6,
            overall_quality: 8,
            correct_emotion: true,
            comments: None,
        };

        let emotion_vector = EmotionVector::new();
        let params = EmotionParameters::new(emotion_vector);

        let invalid_evaluation = PerceptualEvaluation::new(
            "invalid_evaluator".to_string(),
            Emotion::Happy,
            EmotionIntensity::new(0.8),
            params,
            invalid_criteria,
            Duration::from_secs(30),
        );

        // Should fail to create due to invalid criteria
        assert!(invalid_evaluation.is_err());
    }

    #[test]
    fn test_validation_statistics_with_multiple_evaluators() {
        let config = PerceptualValidationConfig {
            min_evaluators: 3,
            ..Default::default()
        };
        let mut study = PerceptualValidationStudy::new(config);

        let emotion_vector = EmotionVector::new();
        let params = EmotionParameters::new(emotion_vector);

        // Add evaluations from 3 different evaluators
        let evaluations = [
            ("eval_1", 8, 7, 6, 8, true),
            ("eval_2", 7, 8, 7, 7, true),
            ("eval_3", 6, 6, 8, 9, false),
            ("eval_1", 9, 8, 5, 8, true), // eval_1 does another
        ];

        for (evaluator_id, nat, app, perc, qual, correct) in evaluations.iter() {
            let criteria = EvaluationCriteria {
                naturalness: *nat,
                appropriateness: *app,
                perceived_intensity: *perc,
                overall_quality: *qual,
                correct_emotion: *correct,
                comments: None,
            };

            let evaluation = PerceptualEvaluation::new(
                evaluator_id.to_string(),
                Emotion::Happy,
                EmotionIntensity::new(0.8),
                params.clone(),
                criteria,
                Duration::from_secs(45),
            )
            .unwrap();

            study.add_evaluation(evaluation).unwrap();
        }

        let stats = study.calculate_statistics().unwrap();

        assert_eq!(stats.evaluation_count, 4);
        assert_eq!(stats.evaluator_count, 3); // Only 3 unique evaluators

        // Check averages
        let expected_avg_naturalness = (8.0 + 7.0 + 6.0 + 9.0) / 4.0; // 7.5
        assert!((stats.avg_naturalness - expected_avg_naturalness).abs() < 0.01);

        // Recognition accuracy: 3 out of 4 correct
        assert!((stats.recognition_accuracy - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_validation_statistics_empty_evaluations() {
        let config = PerceptualValidationConfig::default();
        let study = PerceptualValidationStudy::new(config);

        // Should fail with no evaluations
        assert!(study.calculate_statistics().is_err());
    }

    #[test]
    fn test_correlation_calculation() {
        let config = PerceptualValidationConfig::default();
        let study = PerceptualValidationStudy::new(config);

        // Test perfect positive correlation
        let scores1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let scores2 = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect linear relationship
        let correlation = study.calculate_correlation(&scores1, &scores2);
        assert!((correlation - 1.0).abs() < 0.01); // Should be close to 1.0

        // Test no correlation
        let scores3 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let scores4 = vec![5.0, 1.0, 4.0, 2.0, 3.0]; // Random order
        let correlation2 = study.calculate_correlation(&scores3, &scores4);
        assert!(correlation2.abs() < 0.5); // Should be low correlation

        // Test edge cases
        let empty: Vec<f32> = vec![];
        let single = vec![1.0];
        assert_eq!(study.calculate_correlation(&empty, &scores1), 0.0);
        assert_eq!(study.calculate_correlation(&single, &scores1), 0.0);
    }

    #[test]
    fn test_inter_evaluator_agreement_calculation() {
        let config = PerceptualValidationConfig::default();
        let mut study = PerceptualValidationStudy::new(config);

        let emotion_vector = EmotionVector::new();
        let params = EmotionParameters::new(emotion_vector);

        // Add evaluations from 2 evaluators with similar scoring patterns
        let similar_evals = [
            ("evaluator_A", 8, true),
            ("evaluator_B", 8, true),
            ("evaluator_A", 7, false),
            ("evaluator_B", 7, false),
        ];

        for (evaluator_id, score, correct) in similar_evals.iter() {
            let criteria = EvaluationCriteria {
                naturalness: *score,
                appropriateness: *score,
                perceived_intensity: *score,
                overall_quality: *score,
                correct_emotion: *correct,
                comments: None,
            };

            let evaluation = PerceptualEvaluation::new(
                evaluator_id.to_string(),
                Emotion::Happy,
                EmotionIntensity::new(0.8),
                params.clone(),
                criteria,
                Duration::from_secs(30),
            )
            .unwrap();

            study.add_evaluation(evaluation).unwrap();
        }

        let stats = study.calculate_statistics().unwrap();
        // Similar scoring should result in high agreement
        assert!(stats.inter_evaluator_agreement > 0.5);
    }

    #[test]
    fn test_study_completion_requirements() {
        let config = PerceptualValidationConfig {
            min_evaluators: 3,
            ..Default::default()
        };
        let mut study = PerceptualValidationStudy::new(config);

        // Initially incomplete
        assert!(!study.is_study_complete());

        let emotion_vector = EmotionVector::new();
        let params = EmotionParameters::new(emotion_vector);
        let criteria = EvaluationCriteria {
            naturalness: 8,
            appropriateness: 7,
            perceived_intensity: 6,
            overall_quality: 8,
            correct_emotion: true,
            comments: None,
        };

        // Add evaluations from different evaluators
        for i in 1..=3 {
            let evaluation = PerceptualEvaluation::new(
                format!("evaluator_{}", i),
                Emotion::Happy,
                EmotionIntensity::new(0.8),
                params.clone(),
                criteria.clone(),
                Duration::from_secs(30),
            )
            .unwrap();

            study.add_evaluation(evaluation).unwrap();
        }

        // Should now be complete
        assert!(study.is_study_complete());
    }

    #[test]
    fn test_progress_summary() {
        let config = PerceptualValidationConfig {
            min_evaluators: 5,
            ..Default::default()
        };
        let mut study = PerceptualValidationStudy::new(config);

        // Test initial progress
        let initial_progress = study.get_progress_summary();
        assert_eq!(
            initial_progress["total_evaluations"],
            serde_json::Value::from(0)
        );
        assert_eq!(
            initial_progress["unique_evaluators"],
            serde_json::Value::from(0)
        );
        assert_eq!(
            initial_progress["required_evaluators"],
            serde_json::Value::from(5)
        );
        assert_eq!(
            initial_progress["progress_percent"],
            serde_json::Value::from(0.0)
        );
        assert_eq!(
            initial_progress["is_complete"],
            serde_json::Value::from(false)
        );

        // Add evaluations from 3 evaluators
        let emotion_vector = EmotionVector::new();
        let params = EmotionParameters::new(emotion_vector);
        let criteria = EvaluationCriteria {
            naturalness: 8,
            appropriateness: 7,
            perceived_intensity: 6,
            overall_quality: 8,
            correct_emotion: true,
            comments: None,
        };

        for i in 1..=3 {
            let evaluation = PerceptualEvaluation::new(
                format!("prog_evaluator_{}", i),
                Emotion::Happy,
                EmotionIntensity::new(0.8),
                params.clone(),
                criteria.clone(),
                Duration::from_secs(30),
            )
            .unwrap();

            study.add_evaluation(evaluation).unwrap();
        }

        let progress = study.get_progress_summary();
        assert_eq!(progress["total_evaluations"], serde_json::Value::from(3));
        assert_eq!(progress["unique_evaluators"], serde_json::Value::from(3));
        // Check progress percentage (allow for floating point precision)
        if let serde_json::Value::Number(progress_val) = &progress["progress_percent"] {
            let actual = progress_val.as_f64().unwrap_or(0.0);
            assert!(
                (actual - 60.0).abs() < 0.01,
                "Expected ~60.0, got {}",
                actual
            );
        } else {
            panic!("progress_percent should be a number");
        }
        assert_eq!(progress["is_complete"], serde_json::Value::from(false));
    }

    #[test]
    fn test_export_results() {
        let config = PerceptualValidationConfig {
            min_evaluators: 1,
            ..Default::default()
        };
        let mut study = PerceptualValidationStudy::new(config);

        let emotion_vector = EmotionVector::new();
        let params = EmotionParameters::new(emotion_vector);
        let criteria = EvaluationCriteria {
            naturalness: 8,
            appropriateness: 7,
            perceived_intensity: 6,
            overall_quality: 8,
            correct_emotion: true,
            comments: Some("Export test".to_string()),
        };

        let evaluation = PerceptualEvaluation::new(
            "export_evaluator".to_string(),
            Emotion::Happy,
            EmotionIntensity::new(0.8),
            params,
            criteria,
            Duration::from_secs(45),
        )
        .unwrap();

        study.add_evaluation(evaluation).unwrap();

        let export_result = study.export_results();
        assert!(export_result.is_ok());

        let json_str = export_result.unwrap();
        assert!(json_str.contains("study_id"));
        assert!(json_str.contains("statistics"));
        assert!(json_str.contains("evaluations"));
        assert!(json_str.contains("export_evaluator"));
        assert!(json_str.contains("Export test"));

        // Verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert!(parsed["study_id"].is_string());
        assert!(parsed["statistics"].is_object());
        assert!(parsed["evaluations"].is_array());
    }

    #[test]
    fn test_different_emotion_targets() {
        let config = PerceptualValidationConfig::default();
        let mut study = PerceptualValidationStudy::new(config);

        let emotions_to_test = vec![
            Emotion::Happy,
            Emotion::Sad,
            Emotion::Angry,
            Emotion::Fear,
            Emotion::Surprise,
            Emotion::Neutral,
            Emotion::Custom("nostalgic".to_string()),
        ];

        let emotion_vector = EmotionVector::new();
        let params = EmotionParameters::new(emotion_vector);
        let criteria = EvaluationCriteria {
            naturalness: 7,
            appropriateness: 8,
            perceived_intensity: 6,
            overall_quality: 7,
            correct_emotion: true,
            comments: None,
        };

        for (i, emotion) in emotions_to_test.iter().enumerate() {
            let evaluation = PerceptualEvaluation::new(
                format!("evaluator_{}", i),
                emotion.clone(),
                EmotionIntensity::new(0.7),
                params.clone(),
                criteria.clone(),
                Duration::from_secs(40),
            )
            .unwrap();

            study.add_evaluation(evaluation).unwrap();
        }

        // Check that all emotions are represented
        for emotion in &emotions_to_test {
            let evals = study.get_evaluations_for_emotion(emotion);
            assert_eq!(evals.len(), 1);
            assert_eq!(&evals[0].target_emotion, emotion);
        }

        let stats = study.calculate_statistics().unwrap();
        assert_eq!(stats.evaluation_count, emotions_to_test.len());
    }

    #[test]
    fn test_validation_config_serialization() {
        let config = PerceptualValidationConfig {
            max_session_duration: Duration::from_secs(900),
            min_evaluators: 10,
            target_emotions: vec![
                Emotion::Happy,
                Emotion::Sad,
                Emotion::Custom("test".to_string()),
            ],
            sample_duration: 3.5,
            randomize_order: false,
            allow_partial: false,
        };

        // Test JSON serialization
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: PerceptualValidationConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(
            deserialized.max_session_duration,
            config.max_session_duration
        );
        assert_eq!(deserialized.min_evaluators, config.min_evaluators);
        assert_eq!(deserialized.target_emotions.len(), 3);
        assert_eq!(deserialized.sample_duration, config.sample_duration);
        assert_eq!(deserialized.randomize_order, config.randomize_order);
        assert_eq!(deserialized.allow_partial, config.allow_partial);
    }
}
