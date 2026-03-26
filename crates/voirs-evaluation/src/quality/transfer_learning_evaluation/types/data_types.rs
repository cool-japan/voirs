//! Transfer learning evaluation data types and struct definitions.

use crate::integration::RecommendationPriority;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use voirs_sdk::LanguageCode;

/// Convergence pattern type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConvergencePattern {
    /// Monotonic improvement
    Monotonic,
    /// Oscillating convergence
    Oscillating,
    /// Plateau reached
    Plateau,
    /// Divergent behavior
    Divergent,
    /// Irregular pattern
    Irregular,
}
/// Few-shot learning performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotPerformance {
    /// Performance by sample size
    pub performance_by_sample_size: HashMap<usize, f32>,
    /// Learning efficiency
    pub learning_efficiency: f32,
    /// Sample efficiency
    pub sample_efficiency: f32,
    /// Adaptation speed
    pub adaptation_speed: f32,
    /// Minimum samples needed
    pub min_samples_needed: usize,
    /// Performance saturation point
    pub saturation_point: Option<usize>,
    /// Few-shot learning curve
    pub learning_curve: Vec<LearningCurvePoint>,
}
/// Knowledge transfer assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeTransferAssessment {
    /// Phonetic knowledge transfer
    pub phonetic_knowledge_transfer: f32,
    /// Prosodic knowledge transfer
    pub prosodic_knowledge_transfer: f32,
    /// Acoustic knowledge transfer
    pub acoustic_knowledge_transfer: f32,
    /// Linguistic knowledge transfer
    pub linguistic_knowledge_transfer: f32,
    /// Cultural knowledge transfer
    pub cultural_knowledge_transfer: f32,
    /// Overall knowledge transfer score
    pub overall_knowledge_transfer: f32,
    /// Knowledge transfer efficiency
    pub transfer_efficiency: f32,
    /// Knowledge transfer consistency
    pub transfer_consistency: f32,
    /// Knowledge transfer coverage
    pub transfer_coverage: HashMap<LanguageCode, f32>,
}
/// Stability metrics for a language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityMetrics {
    /// Mean performance
    pub mean_performance: f32,
    /// Performance standard deviation
    pub performance_std: f32,
    /// Performance variance
    pub performance_variance: f32,
    /// Stability coefficient
    pub stability_coefficient: f32,
    /// Convergence epochs
    pub convergence_epochs: Option<usize>,
    /// Best achieved performance
    pub best_performance: f32,
    /// Final performance
    pub final_performance: f32,
}
/// Convergence analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceAnalysis {
    /// Convergence pattern
    pub convergence_pattern: ConvergencePattern,
    /// Convergence speed
    pub convergence_speed: f32,
    /// Convergence quality
    pub convergence_quality: f32,
    /// Early stopping recommendation
    pub early_stopping_epoch: Option<usize>,
    /// Convergence reliability
    pub convergence_reliability: f32,
    /// Plateau detection
    pub plateau_detected: bool,
    /// Plateau start epoch
    pub plateau_start_epoch: Option<usize>,
}
/// Transfer learning evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferLearningEvaluationResult {
    /// Source language
    pub source_language: LanguageCode,
    /// Target languages evaluated
    pub target_languages: Vec<LanguageCode>,
    /// Overall transfer learning score
    pub overall_transfer_score: f32,
    /// Cross-linguistic knowledge transfer assessment
    pub knowledge_transfer_assessment: KnowledgeTransferAssessment,
    /// Source-target transfer effectiveness
    pub transfer_effectiveness: HashMap<LanguageCode, f32>,
    /// Transfer learning stability analysis
    pub stability_analysis: TransferStabilityAnalysis,
    /// Few-shot learning performance
    pub few_shot_performance: HashMap<LanguageCode, FewShotPerformance>,
    /// Domain adaptation assessment
    pub domain_adaptation: HashMap<LanguageCode, DomainAdaptationResult>,
    /// Negative transfer detection results
    pub negative_transfer_detection: NegativeTransferDetectionResult,
    /// Transfer optimization recommendations
    pub transfer_optimization_recommendations: Vec<TransferOptimizationRecommendation>,
    /// Transfer learning metrics
    pub transfer_metrics: TransferLearningMetrics,
    /// Language transfer matrix
    pub language_transfer_matrix: HashMap<(LanguageCode, LanguageCode), f32>,
    /// Problematic transfer pairs
    pub problematic_transfer_pairs: Vec<ProblematicTransferPair>,
    /// Evaluation confidence
    pub evaluation_confidence: f32,
    /// Processing time
    pub processing_time: Duration,
}
/// Type of adaptation challenge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationChallengeType {
    /// Phonetic divergence
    PhoneticDivergence,
    /// Prosodic mismatch
    ProsodicMismatch,
    /// Acoustic incompatibility
    AcousticIncompatibility,
    /// Cultural differences
    CulturalDifferences,
    /// Limited training data
    LimitedTrainingData,
    /// Negative interference
    NegativeInterference,
}
/// Learning curve point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningCurvePoint {
    /// Number of samples
    pub samples: usize,
    /// Performance score
    pub performance: f32,
    /// Variance
    pub variance: f32,
    /// Confidence interval
    pub confidence_interval: (f32, f32),
}
/// Negative transfer detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegativeTransferDetectionResult {
    /// Negative transfer detected
    pub negative_transfer_detected: bool,
    /// Negative transfer severity
    pub negative_transfer_severity: f32,
    /// Affected language pairs
    pub affected_language_pairs: Vec<(LanguageCode, LanguageCode)>,
    /// Negative transfer sources
    pub negative_transfer_sources: Vec<NegativeTransferSource>,
    /// Mitigation strategies
    pub mitigation_strategies: Vec<String>,
    /// Performance degradation
    pub performance_degradation: HashMap<LanguageCode, f32>,
}
/// Type of transfer optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransferOptimizationRecommendationType {
    /// Improve source language selection
    SourceLanguageSelection,
    /// Enhance multi-task learning
    MultiTaskLearning,
    /// Optimize transfer timing
    TransferTiming,
    /// Improve domain adaptation
    DomainAdaptation,
    /// Enhance few-shot learning
    FewShotLearning,
    /// Reduce negative transfer
    NegativeTransferReduction,
    /// Increase model capacity
    ModelCapacityIncrease,
    /// Improve data quality
    DataQualityImprovement,
}
/// Domain adaptation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainAdaptationResult {
    /// Domain adaptation score
    pub adaptation_score: f32,
    /// Domain similarity
    pub domain_similarity: f32,
    /// Adaptation efficiency
    pub adaptation_efficiency: f32,
    /// Domain gap
    pub domain_gap: f32,
    /// Adaptation challenges
    pub adaptation_challenges: Vec<AdaptationChallenge>,
    /// Adaptation recommendations
    pub adaptation_recommendations: Vec<String>,
}
/// Problematic transfer pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblematicTransferPair {
    /// Source language
    pub source_language: LanguageCode,
    /// Target language
    pub target_language: LanguageCode,
    /// Problem severity
    pub problem_severity: f32,
    /// Problem types
    pub problem_types: Vec<TransferProblemType>,
    /// Problem description
    pub problem_description: String,
    /// Improvement strategies
    pub improvement_strategies: Vec<String>,
}
/// Transfer stability analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferStabilityAnalysis {
    /// Transfer convergence rate
    pub convergence_rate: f32,
    /// Transfer stability score
    pub stability_score: f32,
    /// Transfer consistency across languages
    pub cross_language_consistency: f32,
    /// Transfer robustness to noise
    pub noise_robustness: f32,
    /// Transfer performance variance
    pub performance_variance: f32,
    /// Stability metrics per language
    pub language_stability_metrics: HashMap<LanguageCode, StabilityMetrics>,
    /// Convergence analysis
    pub convergence_analysis: ConvergenceAnalysis,
}
/// Adaptation challenge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationChallenge {
    /// Challenge type
    pub challenge_type: AdaptationChallengeType,
    /// Challenge severity
    pub severity: f32,
    /// Challenge description
    pub description: String,
    /// Suggested solutions
    pub suggested_solutions: Vec<String>,
}
/// Source of negative transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegativeTransferSource {
    /// Source type
    pub source_type: NegativeTransferSourceType,
    /// Source language
    pub source_language: LanguageCode,
    /// Target language
    pub target_language: LanguageCode,
    /// Interference magnitude
    pub interference_magnitude: f32,
    /// Interference description
    pub description: String,
}
/// Implementation effort level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    /// Low effort
    Low,
    /// Medium effort
    Medium,
    /// High effort
    High,
    /// Very high effort
    VeryHigh,
}
/// Transfer learning evaluation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferLearningEvaluationConfig {
    /// Enable cross-linguistic knowledge transfer assessment
    pub enable_knowledge_transfer_assessment: bool,
    /// Enable source-target language transfer effectiveness evaluation
    pub enable_transfer_effectiveness: bool,
    /// Enable transfer learning stability analysis
    pub enable_stability_analysis: bool,
    /// Enable few-shot learning performance evaluation
    pub enable_few_shot_evaluation: bool,
    /// Enable domain adaptation assessment
    pub enable_domain_adaptation: bool,
    /// Enable negative transfer detection
    pub enable_negative_transfer_detection: bool,
    /// Enable transfer optimization recommendations
    pub enable_transfer_optimization: bool,
    /// Weight for knowledge transfer assessment
    pub knowledge_transfer_weight: f32,
    /// Weight for transfer effectiveness evaluation
    pub transfer_effectiveness_weight: f32,
    /// Weight for stability analysis
    pub stability_analysis_weight: f32,
    /// Weight for few-shot evaluation
    pub few_shot_evaluation_weight: f32,
    /// Weight for domain adaptation
    pub domain_adaptation_weight: f32,
    /// Minimum transfer effectiveness threshold
    pub min_transfer_effectiveness_threshold: f32,
    /// Maximum acceptable negative transfer
    pub max_negative_transfer_threshold: f32,
    /// Few-shot learning sample sizes to evaluate
    pub few_shot_sample_sizes: Vec<usize>,
    /// Transfer learning evaluation languages
    pub evaluation_languages: Vec<LanguageCode>,
}
/// Type of negative transfer source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NegativeTransferSourceType {
    /// Phonetic interference
    PhoneticInterference,
    /// Prosodic interference
    ProsodicInterference,
    /// Acoustic interference
    AcousticInterference,
    /// Linguistic interference
    LinguisticInterference,
    /// Cultural interference
    CulturalInterference,
    /// Model capacity limitations
    ModelCapacityLimitations,
}
/// Transfer optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferOptimizationRecommendation {
    /// Recommendation type
    pub recommendation_type: TransferOptimizationRecommendationType,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Target languages
    pub target_languages: Vec<LanguageCode>,
    /// Recommendation description
    pub description: String,
    /// Expected improvement
    pub expected_improvement: f32,
    /// Implementation effort
    pub implementation_effort: ImplementationEffort,
    /// Specific parameters
    pub parameters: HashMap<String, f32>,
}
/// Transfer learning metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferLearningMetrics {
    /// Transfer success rate
    pub transfer_success_rate: f32,
    /// Average transfer effectiveness
    pub average_transfer_effectiveness: f32,
    /// Transfer efficiency
    pub transfer_efficiency: f32,
    /// Cross-linguistic consistency
    pub cross_linguistic_consistency: f32,
    /// Knowledge preservation
    pub knowledge_preservation: f32,
    /// Adaptation speed
    pub adaptation_speed: f32,
    /// Negative transfer rate
    pub negative_transfer_rate: f32,
    /// Overall transfer quality
    pub overall_transfer_quality: f32,
}
/// Type of transfer problem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransferProblemType {
    /// Poor transfer effectiveness
    PoorTransferEffectiveness,
    /// Negative transfer
    NegativeTransfer,
    /// Unstable convergence
    UnstableConvergence,
    /// Insufficient adaptation
    InsufficientAdaptation,
    /// Knowledge interference
    KnowledgeInterference,
    /// Resource inefficiency
    ResourceInefficiency,
}
/// Transfer history entry
#[derive(Debug, Clone)]
pub struct TransferHistoryEntry {
    /// Epoch number
    pub epoch: usize,
    /// Performance score
    pub performance: f32,
    /// Loss value
    pub loss: f32,
    /// Validation score
    pub validation_score: Option<f32>,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
}
/// Transfer learning evaluator
pub struct TransferLearningEvaluator {
    /// Configuration
    pub(super) config: TransferLearningEvaluationConfig,
    /// Universal phoneme mapper
    pub(super) phoneme_mapper: crate::quality::universal_phoneme_mapping::UniversalPhonemeMapper,
    /// Cross-language intelligibility evaluator
    pub(super) intelligibility_evaluator:
        crate::quality::cross_language_intelligibility::CrossLanguageIntelligibilityEvaluator,
    /// Multilingual speaker model evaluator
    pub(super) speaker_model_evaluator:
        crate::quality::multilingual_speaker_models::MultilingualSpeakerModelEvaluator,
    /// Cross-cultural perceptual model
    pub(super) cultural_model: crate::perceptual::cross_cultural::CrossCulturalPerceptualModel,
    /// Transfer learning history cache
    pub(super) transfer_history_cache:
        HashMap<(LanguageCode, LanguageCode), Vec<TransferHistoryEntry>>,
    /// Language similarity matrix
    pub(crate) language_similarity_matrix: HashMap<(LanguageCode, LanguageCode), f32>,
}
