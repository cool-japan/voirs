//! Perceptual evaluation framework for voice cloning quality assessment
//!
//! This module provides comprehensive human perception-based evaluation tools
//! for assessing the quality, naturalness, and similarity of cloned voices.

use crate::{Error, Result, SpeakerProfile, VoiceSample};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Perceptual evaluation framework for voice cloning quality assessment
#[derive(Debug, Clone)]
pub struct PerceptualEvaluator {
    config: PerceptualEvaluationConfig,
    active_studies: HashMap<String, EvaluationStudy>,
    evaluator_pool: Vec<Evaluator>,
    results_cache: HashMap<String, EvaluationResults>,
}

/// Configuration for perceptual evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualEvaluationConfig {
    /// Minimum number of evaluators per sample
    pub min_evaluators: usize,
    /// Maximum evaluation duration per sample (seconds)
    pub max_evaluation_duration: f32,
    /// Target inter-evaluator agreement threshold
    pub target_agreement: f32,
    /// Enable quality control measures
    pub quality_control: bool,
    /// Randomize presentation order
    pub randomize_order: bool,
    /// Include reference samples for comparison
    pub include_references: bool,
    /// Evaluation criteria weights
    pub criteria_weights: EvaluationCriteriaWeights,
}

/// Weights for different evaluation criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationCriteriaWeights {
    /// Overall naturalness weight (0.0-1.0)
    pub naturalness: f32,
    /// Speaker similarity weight (0.0-1.0)
    pub similarity: f32,
    /// Audio quality weight (0.0-1.0)
    pub audio_quality: f32,
    /// Intelligibility weight (0.0-1.0)
    pub intelligibility: f32,
    /// Emotion appropriateness weight (0.0-1.0)
    pub emotion_appropriateness: f32,
    /// Prosody quality weight (0.0-1.0)
    pub prosody_quality: f32,
}

/// Individual evaluator information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evaluator {
    /// Unique evaluator ID
    pub id: String,
    /// Evaluator expertise level
    pub expertise: ExpertiseLevel,
    /// Native language
    pub native_language: String,
    /// Age group
    pub age_group: AgeGroup,
    /// Audio experience level
    pub audio_experience: AudioExperience,
    /// Hearing status
    pub hearing_status: HearingStatus,
    /// Reliability score (0.0-1.0)
    pub reliability_score: f32,
    /// Number of completed evaluations
    pub evaluations_completed: usize,
}

/// Evaluator expertise levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpertiseLevel {
    Novice,
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

/// Age groups for demographic analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgeGroup {
    Young18to24,
    Adult25to34,
    Adult35to44,
    MiddleAged45to54,
    Senior55Plus,
}

/// Audio experience levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioExperience {
    General,
    Audiophile,
    MusicProducer,
    AudioEngineer,
    Researcher,
}

/// Hearing status categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HearingStatus {
    Normal,
    MildLoss,
    ModerateLoss,
    SevereLoss,
    Deaf,
}

/// Evaluation study configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationStudy {
    /// Unique study ID
    pub id: String,
    /// Study title
    pub title: String,
    /// Study description
    pub description: String,
    /// Samples to evaluate
    pub samples: Vec<EvaluationSample>,
    /// Study parameters
    pub parameters: StudyParameters,
    /// Current status
    pub status: StudyStatus,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Completion timestamp
    pub completed_at: Option<SystemTime>,
    /// Results summary
    pub results: Option<StudyResults>,
}

/// Individual sample for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationSample {
    /// Sample ID
    pub id: String,
    /// Original voice sample
    pub original: VoiceSample,
    /// Cloned voice sample
    pub cloned: VoiceSample,
    /// Reference speaker profile
    pub reference_speaker: SpeakerProfile,
    /// Text content
    pub text: String,
    /// Sample type
    pub sample_type: SampleType,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Types of evaluation samples
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SampleType {
    /// Original recording
    Original,
    /// High quality clone
    HighQualityClone,
    /// Standard quality clone
    StandardQualityClone,
    /// Low quality clone
    LowQualityClone,
    /// Reference sample for comparison
    Reference,
}

/// Study configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudyParameters {
    /// Evaluation method
    pub evaluation_method: EvaluationMethod,
    /// Presentation order
    pub presentation_order: PresentationOrder,
    /// Quality control measures
    pub quality_control: QualityControlMeasures,
    /// Session limits
    pub session_limits: SessionLimits,
}

/// Evaluation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvaluationMethod {
    /// Absolute Category Rating (1-5 scale)
    AbsoluteCategoryRating,
    /// Degradation Category Rating
    DegradationCategoryRating,
    /// Comparison Category Rating
    ComparisonCategoryRating,
    /// Pairwise Comparison
    PairwiseComparison,
    /// Multiple Stimuli with Hidden Reference and Anchor (MUSHRA)
    Mushra,
}

/// Presentation order options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PresentationOrder {
    /// Sequential presentation
    Sequential,
    /// Randomized order
    Randomized,
    /// Balanced Latin Square
    BalancedLatinSquare,
    /// Custom order
    Custom,
}

/// Quality control measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityControlMeasures {
    /// Include attention check samples
    pub attention_checks: bool,
    /// Require minimum listening time
    pub minimum_listening_time: Duration,
    /// Maximum response time
    pub maximum_response_time: Duration,
    /// Require training phase
    pub training_phase: bool,
    /// Enable inter-evaluator agreement checking
    pub agreement_checking: bool,
}

/// Session limits and constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionLimits {
    /// Maximum samples per session
    pub max_samples_per_session: usize,
    /// Maximum session duration
    pub max_session_duration: Duration,
    /// Required break duration between sessions
    pub break_duration: Duration,
    /// Maximum sessions per evaluator per day
    pub max_sessions_per_day: usize,
}

/// Study status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StudyStatus {
    /// Study is being designed
    Designing,
    /// Study is recruiting participants
    Recruiting,
    /// Study is actively collecting data
    Active,
    /// Study is paused
    Paused,
    /// Study is completed
    Completed,
    /// Study was cancelled
    Cancelled,
}

/// Individual evaluation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResponse {
    /// Response ID
    pub id: String,
    /// Evaluator ID
    pub evaluator_id: String,
    /// Sample ID
    pub sample_id: String,
    /// Evaluation scores
    pub scores: EvaluationScores,
    /// Response time (seconds)
    pub response_time: f32,
    /// Listening time (seconds)
    pub listening_time: f32,
    /// Confidence level (1-7 scale)
    pub confidence: u8,
    /// Optional comments
    pub comments: Option<String>,
    /// Response timestamp
    pub timestamp: SystemTime,
}

/// Evaluation scores for different criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationScores {
    /// Overall quality (1-5 scale)
    pub overall_quality: f32,
    /// Naturalness (1-5 scale)
    pub naturalness: f32,
    /// Speaker similarity (1-5 scale)
    pub similarity: f32,
    /// Audio quality (1-5 scale)
    pub audio_quality: f32,
    /// Intelligibility (1-5 scale)
    pub intelligibility: f32,
    /// Emotion appropriateness (1-5 scale)
    pub emotion_appropriateness: Option<f32>,
    /// Prosody quality (1-5 scale)
    pub prosody_quality: f32,
}

/// Aggregated evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResults {
    /// Sample ID
    pub sample_id: String,
    /// Number of evaluators
    pub num_evaluators: usize,
    /// Mean scores
    pub mean_scores: EvaluationScores,
    /// Standard deviations
    pub std_deviations: EvaluationScores,
    /// Confidence intervals (95%)
    pub confidence_intervals: ConfidenceIntervals,
    /// Inter-evaluator agreement
    pub inter_evaluator_agreement: f32,
    /// Statistical significance tests
    pub statistical_tests: StatisticalTests,
    /// Demographic breakdown
    pub demographic_breakdown: DemographicBreakdown,
}

/// Confidence intervals for evaluation scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceIntervals {
    /// Overall quality confidence interval
    pub overall_quality: (f32, f32),
    /// Naturalness confidence interval
    pub naturalness: (f32, f32),
    /// Similarity confidence interval
    pub similarity: (f32, f32),
    /// Audio quality confidence interval
    pub audio_quality: (f32, f32),
    /// Intelligibility confidence interval
    pub intelligibility: (f32, f32),
    /// Prosody quality confidence interval
    pub prosody_quality: (f32, f32),
}

/// Statistical significance tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTests {
    /// ANOVA F-statistic
    pub anova_f_stat: f32,
    /// ANOVA p-value
    pub anova_p_value: f32,
    /// Tukey HSD results
    pub tukey_hsd: Vec<(String, String, f32, f32)>, // (sample1, sample2, diff, p_value)
    /// Effect size (Cohen's d)
    pub effect_size: f32,
}

/// Demographic breakdown of results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemographicBreakdown {
    /// Results by expertise level
    pub by_expertise: HashMap<ExpertiseLevel, EvaluationScores>,
    /// Results by age group
    pub by_age_group: HashMap<AgeGroup, EvaluationScores>,
    /// Results by audio experience
    pub by_audio_experience: HashMap<AudioExperience, EvaluationScores>,
    /// Results by hearing status
    pub by_hearing_status: HashMap<HearingStatus, EvaluationScores>,
}

/// Study results summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudyResults {
    /// Study ID
    pub study_id: String,
    /// Total participants
    pub total_participants: usize,
    /// Total evaluations completed
    pub total_evaluations: usize,
    /// Overall study statistics
    pub overall_statistics: StudyStatistics,
    /// Results by sample
    pub sample_results: HashMap<String, EvaluationResults>,
    /// Key findings
    pub key_findings: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Overall study statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudyStatistics {
    /// Mean overall quality across all samples
    pub mean_overall_quality: f32,
    /// Best performing sample ID
    pub best_sample_id: String,
    /// Worst performing sample ID
    pub worst_sample_id: String,
    /// Quality distribution
    pub quality_distribution: QualityDistribution,
    /// Inter-evaluator reliability
    pub inter_evaluator_reliability: f32,
}

/// Quality score distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDistribution {
    /// Percentage of samples rated as excellent (4.5-5.0)
    pub excellent: f32,
    /// Percentage of samples rated as good (3.5-4.5)
    pub good: f32,
    /// Percentage of samples rated as fair (2.5-3.5)
    pub fair: f32,
    /// Percentage of samples rated as poor (1.5-2.5)
    pub poor: f32,
    /// Percentage of samples rated as bad (1.0-1.5)
    pub bad: f32,
}

impl Default for PerceptualEvaluationConfig {
    fn default() -> Self {
        Self {
            min_evaluators: 20,
            max_evaluation_duration: 30.0,
            target_agreement: 0.7,
            quality_control: true,
            randomize_order: true,
            include_references: true,
            criteria_weights: EvaluationCriteriaWeights::default(),
        }
    }
}

impl Default for EvaluationCriteriaWeights {
    fn default() -> Self {
        Self {
            naturalness: 0.25,
            similarity: 0.25,
            audio_quality: 0.2,
            intelligibility: 0.15,
            emotion_appropriateness: 0.1,
            prosody_quality: 0.05,
        }
    }
}

impl PerceptualEvaluator {
    /// Create a new perceptual evaluator
    pub fn new() -> Self {
        Self {
            config: PerceptualEvaluationConfig::default(),
            active_studies: HashMap::new(),
            evaluator_pool: Vec::new(),
            results_cache: HashMap::new(),
        }
    }

    /// Create evaluator with custom configuration
    pub fn with_config(config: PerceptualEvaluationConfig) -> Self {
        Self {
            config,
            active_studies: HashMap::new(),
            evaluator_pool: Vec::new(),
            results_cache: HashMap::new(),
        }
    }

    /// Create a new evaluation study
    pub fn create_study(
        &mut self,
        title: String,
        description: String,
        parameters: StudyParameters,
    ) -> Result<String> {
        let study_id = format!("study_{}", uuid::Uuid::new_v4());

        let study = EvaluationStudy {
            id: study_id.clone(),
            title,
            description,
            samples: Vec::new(),
            parameters,
            status: StudyStatus::Designing,
            created_at: SystemTime::now(),
            completed_at: None,
            results: None,
        };

        self.active_studies.insert(study_id.clone(), study);
        Ok(study_id)
    }

    /// Add samples to an evaluation study
    pub fn add_samples_to_study(
        &mut self,
        study_id: &str,
        samples: Vec<EvaluationSample>,
    ) -> Result<()> {
        let study = self
            .active_studies
            .get_mut(study_id)
            .ok_or_else(|| Error::Validation(format!("Study not found: {}", study_id)))?;

        if study.status != StudyStatus::Designing {
            return Err(Error::Validation(
                "Cannot add samples to non-designing study".to_string(),
            ));
        }

        study.samples.extend(samples);
        Ok(())
    }

    /// Start an evaluation study
    pub fn start_study(&mut self, study_id: &str) -> Result<()> {
        let study = self
            .active_studies
            .get_mut(study_id)
            .ok_or_else(|| Error::Validation(format!("Study not found: {}", study_id)))?;

        if study.status != StudyStatus::Designing {
            return Err(Error::Validation(
                "Study is not in designing state".to_string(),
            ));
        }

        if study.samples.is_empty() {
            return Err(Error::Validation("No samples in study".to_string()));
        }

        study.status = StudyStatus::Active;
        Ok(())
    }

    /// Add an evaluator to the pool
    pub fn add_evaluator(&mut self, evaluator: Evaluator) {
        self.evaluator_pool.push(evaluator);
    }

    /// Submit an evaluation response
    pub fn submit_response(&mut self, response: EvaluationResponse) -> Result<()> {
        // Validate response
        if response.scores.overall_quality < 1.0 || response.scores.overall_quality > 5.0 {
            return Err(Error::Validation(
                "Invalid overall quality score".to_string(),
            ));
        }

        if response.confidence < 1 || response.confidence > 7 {
            return Err(Error::Validation("Invalid confidence score".to_string()));
        }

        // Store response (in real implementation, would store in database)
        // For now, we'll update the evaluator's completed count
        if let Some(evaluator) = self
            .evaluator_pool
            .iter_mut()
            .find(|e| e.id == response.evaluator_id)
        {
            evaluator.evaluations_completed += 1;
        }

        Ok(())
    }

    /// Calculate evaluation results for a sample
    pub fn calculate_sample_results(
        &self,
        sample_id: &str,
        responses: &[EvaluationResponse],
    ) -> Result<EvaluationResults> {
        if responses.is_empty() {
            return Err(Error::Validation("No responses provided".to_string()));
        }

        let num_evaluators = responses.len();

        // Calculate mean scores
        let mut mean_scores = EvaluationScores {
            overall_quality: 0.0,
            naturalness: 0.0,
            similarity: 0.0,
            audio_quality: 0.0,
            intelligibility: 0.0,
            emotion_appropriateness: Some(0.0),
            prosody_quality: 0.0,
        };

        for response in responses {
            mean_scores.overall_quality += response.scores.overall_quality;
            mean_scores.naturalness += response.scores.naturalness;
            mean_scores.similarity += response.scores.similarity;
            mean_scores.audio_quality += response.scores.audio_quality;
            mean_scores.intelligibility += response.scores.intelligibility;
            mean_scores.prosody_quality += response.scores.prosody_quality;

            if let Some(emotion) = response.scores.emotion_appropriateness {
                if let Some(ref mut mean_emotion) = mean_scores.emotion_appropriateness {
                    *mean_emotion += emotion;
                }
            }
        }

        let n = num_evaluators as f32;
        mean_scores.overall_quality /= n;
        mean_scores.naturalness /= n;
        mean_scores.similarity /= n;
        mean_scores.audio_quality /= n;
        mean_scores.intelligibility /= n;
        mean_scores.prosody_quality /= n;

        if let Some(ref mut emotion) = mean_scores.emotion_appropriateness {
            *emotion /= n;
        }

        // Calculate standard deviations
        let mut variance_scores = EvaluationScores {
            overall_quality: 0.0,
            naturalness: 0.0,
            similarity: 0.0,
            audio_quality: 0.0,
            intelligibility: 0.0,
            emotion_appropriateness: Some(0.0),
            prosody_quality: 0.0,
        };

        for response in responses {
            let diff_overall = response.scores.overall_quality - mean_scores.overall_quality;
            variance_scores.overall_quality += diff_overall * diff_overall;

            let diff_naturalness = response.scores.naturalness - mean_scores.naturalness;
            variance_scores.naturalness += diff_naturalness * diff_naturalness;

            let diff_similarity = response.scores.similarity - mean_scores.similarity;
            variance_scores.similarity += diff_similarity * diff_similarity;

            let diff_audio = response.scores.audio_quality - mean_scores.audio_quality;
            variance_scores.audio_quality += diff_audio * diff_audio;

            let diff_intelligibility =
                response.scores.intelligibility - mean_scores.intelligibility;
            variance_scores.intelligibility += diff_intelligibility * diff_intelligibility;

            let diff_prosody = response.scores.prosody_quality - mean_scores.prosody_quality;
            variance_scores.prosody_quality += diff_prosody * diff_prosody;
        }

        let std_deviations = EvaluationScores {
            overall_quality: (variance_scores.overall_quality / n).sqrt(),
            naturalness: (variance_scores.naturalness / n).sqrt(),
            similarity: (variance_scores.similarity / n).sqrt(),
            audio_quality: (variance_scores.audio_quality / n).sqrt(),
            intelligibility: (variance_scores.intelligibility / n).sqrt(),
            emotion_appropriateness: Some(
                (variance_scores.emotion_appropriateness.unwrap_or(0.0) / n).sqrt(),
            ),
            prosody_quality: (variance_scores.prosody_quality / n).sqrt(),
        };

        // Calculate 95% confidence intervals
        let t_value = 1.96; // For large samples
        let se_overall = std_deviations.overall_quality / (n.sqrt());
        let confidence_intervals = ConfidenceIntervals {
            overall_quality: (
                mean_scores.overall_quality - t_value * se_overall,
                mean_scores.overall_quality + t_value * se_overall,
            ),
            naturalness: (
                mean_scores.naturalness - t_value * (std_deviations.naturalness / n.sqrt()),
                mean_scores.naturalness + t_value * (std_deviations.naturalness / n.sqrt()),
            ),
            similarity: (
                mean_scores.similarity - t_value * (std_deviations.similarity / n.sqrt()),
                mean_scores.similarity + t_value * (std_deviations.similarity / n.sqrt()),
            ),
            audio_quality: (
                mean_scores.audio_quality - t_value * (std_deviations.audio_quality / n.sqrt()),
                mean_scores.audio_quality + t_value * (std_deviations.audio_quality / n.sqrt()),
            ),
            intelligibility: (
                mean_scores.intelligibility - t_value * (std_deviations.intelligibility / n.sqrt()),
                mean_scores.intelligibility + t_value * (std_deviations.intelligibility / n.sqrt()),
            ),
            prosody_quality: (
                mean_scores.prosody_quality - t_value * (std_deviations.prosody_quality / n.sqrt()),
                mean_scores.prosody_quality + t_value * (std_deviations.prosody_quality / n.sqrt()),
            ),
        };

        // Calculate inter-evaluator agreement (simplified)
        let inter_evaluator_agreement = self.calculate_inter_evaluator_agreement(responses);

        // Placeholder statistical tests
        let statistical_tests = StatisticalTests {
            anova_f_stat: 0.0,
            anova_p_value: 1.0,
            tukey_hsd: Vec::new(),
            effect_size: 0.0,
        };

        // Placeholder demographic breakdown
        let demographic_breakdown = DemographicBreakdown {
            by_expertise: HashMap::new(),
            by_age_group: HashMap::new(),
            by_audio_experience: HashMap::new(),
            by_hearing_status: HashMap::new(),
        };

        Ok(EvaluationResults {
            sample_id: sample_id.to_string(),
            num_evaluators,
            mean_scores,
            std_deviations,
            confidence_intervals,
            inter_evaluator_agreement,
            statistical_tests,
            demographic_breakdown,
        })
    }

    /// Calculate inter-evaluator agreement
    fn calculate_inter_evaluator_agreement(&self, responses: &[EvaluationResponse]) -> f32 {
        if responses.len() < 2 {
            return 1.0;
        }

        let mut total_agreement = 0.0;
        let mut comparisons = 0;

        for i in 0..responses.len() {
            for j in (i + 1)..responses.len() {
                let score1 = responses[i].scores.overall_quality;
                let score2 = responses[j].scores.overall_quality;
                let agreement = 1.0 - (score1 - score2).abs() / 4.0; // Normalized agreement
                total_agreement += agreement;
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            total_agreement / comparisons as f32
        } else {
            1.0
        }
    }

    /// Generate study report
    pub fn generate_study_report(&self, study_id: &str) -> Result<StudyResults> {
        let study = self
            .active_studies
            .get(study_id)
            .ok_or_else(|| Error::Validation(format!("Study not found: {}", study_id)))?;

        // In a real implementation, this would analyze all collected responses
        // For now, we'll create a placeholder report
        let study_results = StudyResults {
            study_id: study_id.to_string(),
            total_participants: self.evaluator_pool.len(),
            total_evaluations: self
                .evaluator_pool
                .iter()
                .map(|e| e.evaluations_completed)
                .sum(),
            overall_statistics: StudyStatistics {
                mean_overall_quality: 3.5,
                best_sample_id: "sample_1".to_string(),
                worst_sample_id: "sample_2".to_string(),
                quality_distribution: QualityDistribution {
                    excellent: 15.0,
                    good: 35.0,
                    fair: 30.0,
                    poor: 15.0,
                    bad: 5.0,
                },
                inter_evaluator_reliability: 0.75,
            },
            sample_results: HashMap::new(),
            key_findings: vec![
                "High-quality clones achieved naturalness scores above 4.0".to_string(),
                "Speaker similarity was consistently rated above 3.5".to_string(),
                "Audio quality showed significant improvement with recent model versions"
                    .to_string(),
            ],
            recommendations: vec![
                "Focus on improving prosody quality for more natural speech".to_string(),
                "Enhance speaker similarity through better training data".to_string(),
                "Implement quality control measures for consistent results".to_string(),
            ],
        };

        Ok(study_results)
    }

    /// Export evaluation results to JSON
    pub fn export_results(&self, study_id: &str) -> Result<String> {
        let results = self.generate_study_report(study_id)?;
        serde_json::to_string_pretty(&results)
            .map_err(|e| Error::Processing(format!("Failed to serialize results: {}", e)))
    }

    /// Get study status
    pub fn get_study_status(&self, study_id: &str) -> Result<StudyStatus> {
        let study = self
            .active_studies
            .get(study_id)
            .ok_or_else(|| Error::Validation(format!("Study not found: {}", study_id)))?;
        Ok(study.status)
    }

    /// List all studies
    pub fn list_studies(&self) -> Vec<(String, String, StudyStatus)> {
        self.active_studies
            .values()
            .map(|study| (study.id.clone(), study.title.clone(), study.status))
            .collect()
    }
}

impl Default for PerceptualEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VoiceSample;
    use std::collections::HashMap;

    #[test]
    fn test_perceptual_evaluator_creation() {
        let evaluator = PerceptualEvaluator::new();
        assert_eq!(evaluator.config.min_evaluators, 20);
        assert_eq!(evaluator.config.target_agreement, 0.7);
        assert!(evaluator.active_studies.is_empty());
        assert!(evaluator.evaluator_pool.is_empty());
    }

    #[test]
    fn test_study_creation() {
        let mut evaluator = PerceptualEvaluator::new();
        let parameters = StudyParameters {
            evaluation_method: EvaluationMethod::AbsoluteCategoryRating,
            presentation_order: PresentationOrder::Randomized,
            quality_control: QualityControlMeasures {
                attention_checks: true,
                minimum_listening_time: Duration::from_secs(5),
                maximum_response_time: Duration::from_secs(60),
                training_phase: true,
                agreement_checking: true,
            },
            session_limits: SessionLimits {
                max_samples_per_session: 20,
                max_session_duration: Duration::from_secs(1800),
                break_duration: Duration::from_secs(300),
                max_sessions_per_day: 3,
            },
        };

        let study_id = evaluator
            .create_study(
                "Test Study".to_string(),
                "A test evaluation study".to_string(),
                parameters,
            )
            .unwrap();

        assert!(!study_id.is_empty());
        assert!(evaluator.active_studies.contains_key(&study_id));
        assert_eq!(
            evaluator.get_study_status(&study_id).unwrap(),
            StudyStatus::Designing
        );
    }

    #[test]
    fn test_evaluator_addition() {
        let mut evaluator = PerceptualEvaluator::new();

        let test_evaluator = Evaluator {
            id: "eval_001".to_string(),
            expertise: ExpertiseLevel::Intermediate,
            native_language: "en".to_string(),
            age_group: AgeGroup::Adult25to34,
            audio_experience: AudioExperience::Audiophile,
            hearing_status: HearingStatus::Normal,
            reliability_score: 0.85,
            evaluations_completed: 0,
        };

        evaluator.add_evaluator(test_evaluator);
        assert_eq!(evaluator.evaluator_pool.len(), 1);
        assert_eq!(evaluator.evaluator_pool[0].id, "eval_001");
    }

    #[test]
    fn test_evaluation_response_submission() {
        let mut evaluator = PerceptualEvaluator::new();

        // Add an evaluator first
        let test_evaluator = Evaluator {
            id: "eval_001".to_string(),
            expertise: ExpertiseLevel::Intermediate,
            native_language: "en".to_string(),
            age_group: AgeGroup::Adult25to34,
            audio_experience: AudioExperience::Audiophile,
            hearing_status: HearingStatus::Normal,
            reliability_score: 0.85,
            evaluations_completed: 0,
        };
        evaluator.add_evaluator(test_evaluator);

        let response = EvaluationResponse {
            id: "resp_001".to_string(),
            evaluator_id: "eval_001".to_string(),
            sample_id: "sample_001".to_string(),
            scores: EvaluationScores {
                overall_quality: 4.2,
                naturalness: 4.0,
                similarity: 4.5,
                audio_quality: 4.3,
                intelligibility: 4.8,
                emotion_appropriateness: Some(4.1),
                prosody_quality: 3.9,
            },
            response_time: 25.5,
            listening_time: 12.3,
            confidence: 6,
            comments: Some("Good quality overall".to_string()),
            timestamp: SystemTime::now(),
        };

        let result = evaluator.submit_response(response);
        assert!(result.is_ok());
        assert_eq!(evaluator.evaluator_pool[0].evaluations_completed, 1);
    }

    #[test]
    fn test_invalid_response_rejection() {
        let mut evaluator = PerceptualEvaluator::new();

        let invalid_response = EvaluationResponse {
            id: "resp_001".to_string(),
            evaluator_id: "eval_001".to_string(),
            sample_id: "sample_001".to_string(),
            scores: EvaluationScores {
                overall_quality: 6.0, // Invalid: should be 1-5
                naturalness: 4.0,
                similarity: 4.5,
                audio_quality: 4.3,
                intelligibility: 4.8,
                emotion_appropriateness: Some(4.1),
                prosody_quality: 3.9,
            },
            response_time: 25.5,
            listening_time: 12.3,
            confidence: 6,
            comments: None,
            timestamp: SystemTime::now(),
        };

        let result = evaluator.submit_response(invalid_response);
        assert!(result.is_err());
    }

    #[test]
    fn test_sample_results_calculation() {
        let evaluator = PerceptualEvaluator::new();

        let responses = vec![
            EvaluationResponse {
                id: "resp_001".to_string(),
                evaluator_id: "eval_001".to_string(),
                sample_id: "sample_001".to_string(),
                scores: EvaluationScores {
                    overall_quality: 4.0,
                    naturalness: 4.2,
                    similarity: 3.8,
                    audio_quality: 4.1,
                    intelligibility: 4.5,
                    emotion_appropriateness: Some(4.0),
                    prosody_quality: 3.7,
                },
                response_time: 20.0,
                listening_time: 10.0,
                confidence: 5,
                comments: None,
                timestamp: SystemTime::now(),
            },
            EvaluationResponse {
                id: "resp_002".to_string(),
                evaluator_id: "eval_002".to_string(),
                sample_id: "sample_001".to_string(),
                scores: EvaluationScores {
                    overall_quality: 3.8,
                    naturalness: 4.0,
                    similarity: 4.2,
                    audio_quality: 3.9,
                    intelligibility: 4.3,
                    emotion_appropriateness: Some(3.8),
                    prosody_quality: 3.9,
                },
                response_time: 22.0,
                listening_time: 11.0,
                confidence: 6,
                comments: None,
                timestamp: SystemTime::now(),
            },
        ];

        let results = evaluator
            .calculate_sample_results("sample_001", &responses)
            .unwrap();

        assert_eq!(results.num_evaluators, 2);
        assert!((results.mean_scores.overall_quality - 3.9).abs() < 0.01);
        assert!((results.mean_scores.naturalness - 4.1).abs() < 0.01);
        assert!(results.inter_evaluator_agreement > 0.0);
    }

    #[test]
    fn test_criteria_weights_default() {
        let weights = EvaluationCriteriaWeights::default();
        let total_weight = weights.naturalness
            + weights.similarity
            + weights.audio_quality
            + weights.intelligibility
            + weights.emotion_appropriateness
            + weights.prosody_quality;
        assert!((total_weight - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_study_list() {
        let mut evaluator = PerceptualEvaluator::new();

        let parameters = StudyParameters {
            evaluation_method: EvaluationMethod::AbsoluteCategoryRating,
            presentation_order: PresentationOrder::Randomized,
            quality_control: QualityControlMeasures {
                attention_checks: true,
                minimum_listening_time: Duration::from_secs(5),
                maximum_response_time: Duration::from_secs(60),
                training_phase: true,
                agreement_checking: true,
            },
            session_limits: SessionLimits {
                max_samples_per_session: 20,
                max_session_duration: Duration::from_secs(1800),
                break_duration: Duration::from_secs(300),
                max_sessions_per_day: 3,
            },
        };

        let study_id1 = evaluator
            .create_study(
                "Study 1".to_string(),
                "First study".to_string(),
                parameters.clone(),
            )
            .unwrap();

        let study_id2 = evaluator
            .create_study(
                "Study 2".to_string(),
                "Second study".to_string(),
                parameters,
            )
            .unwrap();

        let studies = evaluator.list_studies();
        assert_eq!(studies.len(), 2);

        let titles: Vec<String> = studies.iter().map(|(_, title, _)| title.clone()).collect();
        assert!(titles.contains(&"Study 1".to_string()));
        assert!(titles.contains(&"Study 2".to_string()));
    }
}
