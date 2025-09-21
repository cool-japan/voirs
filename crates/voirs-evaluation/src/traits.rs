//! Core traits for the `VoiRS` evaluation system
//!
//! This module defines the fundamental interfaces for quality evaluation,
//! pronunciation assessment, and comparative analysis capabilities.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use voirs_recognizer::traits::PhonemeAlignment;
use voirs_sdk::{AudioBuffer, LanguageCode, VoirsError};

/// Result type for evaluation operations
pub type EvaluationResult<T> = Result<T, VoirsError>;

// ============================================================================
// Core Data Types
// ============================================================================

/// Overall quality score with component breakdown
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityScore {
    /// Overall quality score [0.0, 1.0]
    pub overall_score: f32,
    /// Individual component scores
    pub component_scores: HashMap<String, f32>,
    /// Quality recommendations
    pub recommendations: Vec<String>,
    /// Confidence in the evaluation [0.0, 1.0]
    pub confidence: f32,
    /// Processing time
    pub processing_time: Option<Duration>,
}

/// Pronunciation assessment result
#[derive(Debug, Clone, PartialEq)]
pub struct PronunciationScore {
    /// Overall pronunciation score [0.0, 1.0]
    pub overall_score: f32,
    /// Phoneme-level accuracy scores
    pub phoneme_scores: Vec<PhonemeAccuracyScore>,
    /// Word-level pronunciation scores
    pub word_scores: Vec<WordPronunciationScore>,
    /// Fluency score [0.0, 1.0]
    pub fluency_score: f32,
    /// Rhythm score [0.0, 1.0]
    pub rhythm_score: f32,
    /// Stress accuracy [0.0, 1.0]
    pub stress_accuracy: f32,
    /// Intonation accuracy [0.0, 1.0]
    pub intonation_accuracy: f32,
    /// Detailed feedback
    pub feedback: Vec<PronunciationFeedback>,
    /// Confidence in the assessment [0.0, 1.0]
    pub confidence: f32,
}

/// Individual phoneme accuracy score
#[derive(Debug, Clone, PartialEq)]
pub struct PhonemeAccuracyScore {
    /// Expected phoneme
    pub expected_phoneme: String,
    /// Actual phoneme (if detected)
    pub actual_phoneme: Option<String>,
    /// Accuracy score [0.0, 1.0]
    pub accuracy: f32,
    /// Duration accuracy [0.0, 1.0]
    pub duration_accuracy: f32,
    /// Position in the utterance
    pub position: usize,
    /// Timing information - start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
}

/// Word-level pronunciation score
#[derive(Debug, Clone, PartialEq)]
pub struct WordPronunciationScore {
    /// Word text
    pub word: String,
    /// Overall word accuracy [0.0, 1.0]
    pub accuracy: f32,
    /// Stress pattern accuracy [0.0, 1.0]
    pub stress_accuracy: f32,
    /// Syllable structure accuracy [0.0, 1.0]
    pub syllable_accuracy: f32,
    /// Individual phoneme scores for this word
    pub phoneme_scores: Vec<PhonemeAccuracyScore>,
    /// Word position in the utterance
    pub position: usize,
}

/// Pronunciation feedback item
#[derive(Debug, Clone, PartialEq)]
pub struct PronunciationFeedback {
    /// Position in the utterance
    pub position: usize,
    /// Type of feedback
    pub feedback_type: FeedbackType,
    /// Severity level [0.0, 1.0]
    pub severity: f32,
    /// Human-readable message
    pub message: String,
    /// Suggested improvement
    pub suggestion: Option<String>,
}

/// Types of pronunciation feedback
#[derive(Debug, Clone, PartialEq)]
pub enum FeedbackType {
    /// Phoneme substitution error
    PhonemeSubstitution,
    /// Phoneme deletion error
    PhonemeDeletion,
    /// Phoneme insertion error
    PhonemeInsertion,
    /// Stress placement error
    StressError,
    /// Duration error (too long/short)
    DurationError,
    /// Rhythm irregularity
    RhythmError,
    /// Intonation error
    IntonationError,
    /// General quality issue
    QualityIssue,
}

/// Comparison result between two systems/samples
#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonResult {
    /// System A identifier
    pub system_a: String,
    /// System B identifier
    pub system_b: String,
    /// Overall preference score [-1.0, 1.0] (negative favors A, positive favors B)
    pub preference_score: f32,
    /// Individual metric comparisons
    pub metric_comparisons: HashMap<String, MetricComparison>,
    /// Statistical significance of differences
    pub statistical_significance: HashMap<String, f32>,
    /// Detailed analysis
    pub analysis: String,
    /// Confidence in the comparison [0.0, 1.0]
    pub confidence: f32,
}

/// Comparison result for a specific metric
#[derive(Debug, Clone, PartialEq)]
pub struct MetricComparison {
    /// Metric name
    pub metric: String,
    /// Score for system A
    pub score_a: f32,
    /// Score for system B
    pub score_b: f32,
    /// Difference (B - A)
    pub difference: f32,
    /// Relative improvement (difference / `score_a`)
    pub relative_improvement: f32,
    /// Statistical significance
    pub p_value: f32,
}

/// Self-evaluation result
#[derive(Debug, Clone, PartialEq)]
pub struct SelfEvaluationResult {
    /// Quality assessment
    pub quality_score: QualityScore,
    /// Areas identified for improvement
    pub improvement_areas: Vec<ImprovementArea>,
    /// Suggested configuration changes
    pub suggested_changes: Vec<ConfigurationSuggestion>,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f32>,
    /// Overall system health [0.0, 1.0]
    pub system_health: f32,
}

/// Area identified for improvement
#[derive(Debug, Clone, PartialEq)]
pub struct ImprovementArea {
    /// Area name
    pub area: String,
    /// Current performance [0.0, 1.0]
    pub current_performance: f32,
    /// Target performance [0.0, 1.0]
    pub target_performance: f32,
    /// Priority level [0.0, 1.0]
    pub priority: f32,
    /// Specific suggestions
    pub suggestions: Vec<ImprovementSuggestion>,
}

/// Specific improvement suggestion
#[derive(Debug, Clone, PartialEq)]
pub struct ImprovementSuggestion {
    /// Suggestion type
    pub suggestion_type: SuggestionType,
    /// Description
    pub description: String,
    /// Expected impact [0.0, 1.0]
    pub expected_impact: f32,
    /// Implementation difficulty [0.0, 1.0]
    pub difficulty: f32,
    /// Priority [0.0, 1.0]
    pub priority: f32,
}

/// Configuration suggestion
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigurationSuggestion {
    /// Parameter name
    pub parameter: String,
    /// Current value
    pub current_value: String,
    /// Suggested value
    pub suggested_value: String,
    /// Rationale
    pub rationale: String,
    /// Expected improvement [0.0, 1.0]
    pub expected_improvement: f32,
}

/// Types of improvement suggestions
#[derive(Debug, Clone, PartialEq)]
pub enum SuggestionType {
    /// Model parameter adjustment
    ParameterTuning,
    /// Training data improvement
    DataImprovement,
    /// Architecture modification
    ArchitectureChange,
    /// Post-processing enhancement
    PostProcessing,
    /// Preprocessing optimization
    Preprocessing,
    /// Hardware optimization
    HardwareOptimization,
}

// ============================================================================
// Configuration Types
// ============================================================================

/// Quality evaluation configuration
#[derive(Debug, Clone, PartialEq)]
pub struct QualityEvaluationConfig {
    /// Enable objective quality metrics
    pub objective_metrics: bool,
    /// Enable subjective quality prediction
    pub subjective_metrics: bool,
    /// Enable perceptual quality metrics
    pub perceptual_metrics: bool,
    /// Specific metrics to compute
    pub metrics: Vec<QualityMetric>,
    /// Reference audio requirement
    pub require_reference: bool,
    /// Output detailed analysis
    pub detailed_analysis: bool,
    /// Confidence threshold for reliable scores
    pub confidence_threshold: f32,
    /// Enable deep learning-based MOS prediction
    pub use_deep_learning_mos: bool,
    /// Listener demographic adaptation settings
    pub demographic_adaptation: Option<DemographicProfile>,
    /// Cross-cultural perceptual modeling
    pub cultural_adaptation: Option<CulturalProfile>,
    /// Enable listening test simulation
    pub simulate_listening_tests: bool,
    /// Real-time processing mode
    pub real_time_mode: bool,
}

impl Default for QualityEvaluationConfig {
    fn default() -> Self {
        Self {
            objective_metrics: true,
            subjective_metrics: true,
            perceptual_metrics: false,
            metrics: vec![
                QualityMetric::MOS,
                QualityMetric::SpectralDistortion,
                QualityMetric::Naturalness,
                QualityMetric::Intelligibility,
                QualityMetric::ArtifactDetection,
            ],
            require_reference: false,
            detailed_analysis: true,
            confidence_threshold: 0.7,
            use_deep_learning_mos: false,
            demographic_adaptation: None,
            cultural_adaptation: None,
            simulate_listening_tests: false,
            real_time_mode: false,
        }
    }
}

/// Pronunciation evaluation configuration
#[derive(Debug, Clone)]
pub struct PronunciationEvaluationConfig {
    /// Target language
    pub language: LanguageCode,
    /// Enable phoneme-level scoring
    pub phoneme_level_scoring: bool,
    /// Enable word-level scoring
    pub word_level_scoring: bool,
    /// Enable prosody assessment
    pub prosody_assessment: bool,
    /// Specific metrics to compute
    pub metrics: Vec<PronunciationMetric>,
    /// Strictness level [0.0, 1.0] (0=lenient, 1=strict)
    pub strictness: f32,
    /// Native speaker reference
    pub native_reference: Option<AudioBuffer>,
    /// Custom pronunciation dictionary
    pub custom_dictionary: Option<HashMap<String, Vec<String>>>,
}

impl Default for PronunciationEvaluationConfig {
    fn default() -> Self {
        Self {
            language: LanguageCode::EnUs,
            phoneme_level_scoring: true,
            word_level_scoring: true,
            prosody_assessment: true,
            metrics: vec![
                PronunciationMetric::PhonemeAccuracy,
                PronunciationMetric::WordAccuracy,
                PronunciationMetric::Fluency,
                PronunciationMetric::Rhythm,
                PronunciationMetric::StressAccuracy,
            ],
            strictness: 0.7,
            native_reference: None,
            custom_dictionary: None,
        }
    }
}

/// Comparison evaluation configuration
#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonConfig {
    /// Metrics to compare
    pub metrics: Vec<ComparisonMetric>,
    /// Enable statistical analysis
    pub enable_statistical_analysis: bool,
    /// Significance threshold for statistical tests
    pub significance_threshold: f32,
    /// Number of bootstrap samples for confidence intervals
    pub bootstrap_samples: usize,
    /// Include human preference prediction
    pub predict_human_preference: bool,
    /// Detailed breakdown by categories
    pub detailed_breakdown: bool,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
            metrics: vec![
                ComparisonMetric::OverallQuality,
                ComparisonMetric::Naturalness,
                ComparisonMetric::Intelligibility,
                ComparisonMetric::PronunciationAccuracy,
            ],
            enable_statistical_analysis: true,
            significance_threshold: 0.05,
            bootstrap_samples: 1000,
            predict_human_preference: true,
            detailed_breakdown: true,
        }
    }
}

// ============================================================================
// Metric Enumerations
// ============================================================================

/// Quality evaluation metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QualityMetric {
    /// Mean Opinion Score prediction
    MOS,
    /// Perceptual Evaluation of Speech Quality
    PESQ,
    /// Short-Time Objective Intelligibility
    STOI,
    /// Mel Cepstral Distortion
    MCD,
    /// Spectral distortion measures
    SpectralDistortion,
    /// Naturalness assessment
    Naturalness,
    /// Intelligibility assessment
    Intelligibility,
    /// Speaker similarity
    SpeakerSimilarity,
    /// Prosody quality
    ProsodyQuality,
    /// Artifact detection
    ArtifactDetection,
}

/// Pronunciation evaluation metrics
#[derive(Debug, Clone, PartialEq)]
pub enum PronunciationMetric {
    /// Phoneme-level accuracy
    PhonemeAccuracy,
    /// Word-level accuracy
    WordAccuracy,
    /// Sentence-level accuracy
    SentenceAccuracy,
    /// Fluency assessment
    Fluency,
    /// Rhythm evaluation
    Rhythm,
    /// Stress pattern accuracy
    StressAccuracy,
    /// Intonation accuracy
    IntonationAccuracy,
    /// Speaking rate appropriateness
    SpeakingRate,
    /// Pause appropriateness
    PausePattern,
    /// Overall comprehensibility
    Comprehensibility,
}

/// Comparison metrics
#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonMetric {
    /// Overall quality comparison
    OverallQuality,
    /// Naturalness comparison
    Naturalness,
    /// Intelligibility comparison
    Intelligibility,
    /// Pronunciation accuracy comparison
    PronunciationAccuracy,
    /// Prosody comparison
    Prosody,
    /// Speaker consistency
    SpeakerConsistency,
    /// Artifact comparison
    Artifacts,
    /// Processing efficiency
    Efficiency,
}

// ============================================================================
// Core Traits
// ============================================================================

/// Trait for quality evaluation
#[async_trait]
pub trait QualityEvaluator: Send + Sync {
    /// Evaluate quality of generated audio
    async fn evaluate_quality(
        &self,
        generated: &AudioBuffer,
        reference: Option<&AudioBuffer>,
        config: Option<&QualityEvaluationConfig>,
    ) -> EvaluationResult<QualityScore>;

    /// Evaluate quality in batch mode
    async fn evaluate_quality_batch(
        &self,
        samples: &[(AudioBuffer, Option<AudioBuffer>)],
        config: Option<&QualityEvaluationConfig>,
    ) -> EvaluationResult<Vec<QualityScore>>;

    /// Get supported quality metrics
    fn supported_metrics(&self) -> Vec<QualityMetric>;

    /// Check if a reference is required for evaluation
    fn requires_reference(&self, metric: &QualityMetric) -> bool;

    /// Get evaluator metadata
    fn metadata(&self) -> QualityEvaluatorMetadata;
}

/// Trait for pronunciation evaluation
#[async_trait]
pub trait PronunciationEvaluator: Send + Sync {
    /// Evaluate pronunciation quality
    async fn evaluate_pronunciation(
        &self,
        audio: &AudioBuffer,
        text: &str,
        config: Option<&PronunciationEvaluationConfig>,
    ) -> EvaluationResult<PronunciationScore>;

    /// Evaluate pronunciation with phoneme alignment
    async fn evaluate_pronunciation_with_alignment(
        &self,
        audio: &AudioBuffer,
        alignment: &PhonemeAlignment,
        config: Option<&PronunciationEvaluationConfig>,
    ) -> EvaluationResult<PronunciationScore>;

    /// Evaluate pronunciation in batch mode
    async fn evaluate_pronunciation_batch(
        &self,
        samples: &[(AudioBuffer, String)],
        config: Option<&PronunciationEvaluationConfig>,
    ) -> EvaluationResult<Vec<PronunciationScore>>;

    /// Get supported pronunciation metrics
    fn supported_metrics(&self) -> Vec<PronunciationMetric>;

    /// Get supported languages
    fn supported_languages(&self) -> Vec<LanguageCode>;

    /// Get evaluator metadata
    fn metadata(&self) -> PronunciationEvaluatorMetadata;
}

/// Trait for comparative evaluation
#[async_trait]
pub trait ComparativeEvaluator: Send + Sync {
    /// Compare two audio samples
    async fn compare_samples(
        &self,
        sample_a: &AudioBuffer,
        sample_b: &AudioBuffer,
        config: Option<&ComparisonConfig>,
    ) -> EvaluationResult<ComparisonResult>;

    /// Compare two systems across multiple samples
    async fn compare_systems(
        &self,
        system_a_samples: &[AudioBuffer],
        system_b_samples: &[AudioBuffer],
        config: Option<&ComparisonConfig>,
    ) -> EvaluationResult<ComparisonResult>;

    /// Compare multiple systems (returns pairwise comparisons)
    async fn compare_multiple_systems(
        &self,
        systems: &HashMap<String, Vec<AudioBuffer>>,
        config: Option<&ComparisonConfig>,
    ) -> EvaluationResult<HashMap<(String, String), ComparisonResult>>;

    /// Get supported comparison metrics
    fn supported_metrics(&self) -> Vec<ComparisonMetric>;

    /// Get evaluator metadata
    fn metadata(&self) -> ComparativeEvaluatorMetadata;
}

/// Trait for self-evaluation capabilities
#[async_trait]
pub trait SelfEvaluator: Send + Sync {
    /// Perform self-evaluation of a TTS pipeline
    async fn self_evaluate(
        &self,
        pipeline: &voirs_sdk::pipeline::VoirsPipeline,
        test_texts: &[String],
        config: Option<&QualityEvaluationConfig>,
    ) -> EvaluationResult<SelfEvaluationResult>;

    /// Suggest improvements based on evaluation results
    async fn suggest_improvements(
        &self,
        evaluation: &SelfEvaluationResult,
    ) -> EvaluationResult<Vec<ImprovementSuggestion>>;

    /// Monitor system performance over time
    async fn monitor_performance(
        &self,
        pipeline: &voirs_sdk::pipeline::VoirsPipeline,
        monitoring_texts: &[String],
    ) -> EvaluationResult<PerformanceReport>;

    /// Get evaluator metadata
    fn metadata(&self) -> SelfEvaluatorMetadata;
}

// ============================================================================
// Metadata Types
// ============================================================================

/// Quality evaluator metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityEvaluatorMetadata {
    /// Evaluator name
    pub name: String,
    /// Version
    pub version: String,
    /// Description
    pub description: String,
    /// Supported metrics
    pub supported_metrics: Vec<QualityMetric>,
    /// Supported languages
    pub supported_languages: Vec<LanguageCode>,
    /// Requires reference audio
    pub requires_reference: bool,
    /// Processing speed factor
    pub processing_speed: f32,
}

/// Pronunciation evaluator metadata
#[derive(Debug, Clone, PartialEq)]
pub struct PronunciationEvaluatorMetadata {
    /// Evaluator name
    pub name: String,
    /// Version
    pub version: String,
    /// Description
    pub description: String,
    /// Supported metrics
    pub supported_metrics: Vec<PronunciationMetric>,
    /// Supported languages
    pub supported_languages: Vec<LanguageCode>,
    /// Accuracy benchmarks
    pub accuracy_benchmarks: HashMap<LanguageCode, f32>,
    /// Processing speed factor
    pub processing_speed: f32,
}

/// Comparative evaluator metadata
#[derive(Debug, Clone, PartialEq)]
pub struct ComparativeEvaluatorMetadata {
    /// Evaluator name
    pub name: String,
    /// Version
    pub version: String,
    /// Description
    pub description: String,
    /// Supported metrics
    pub supported_metrics: Vec<ComparisonMetric>,
    /// Statistical methods supported
    pub statistical_methods: Vec<String>,
    /// Processing speed factor
    pub processing_speed: f32,
}

/// Self-evaluator metadata
#[derive(Debug, Clone, PartialEq)]
pub struct SelfEvaluatorMetadata {
    /// Evaluator name
    pub name: String,
    /// Version
    pub version: String,
    /// Description
    pub description: String,
    /// Evaluation capabilities
    pub capabilities: Vec<String>,
    /// Supported improvement types
    pub improvement_types: Vec<SuggestionType>,
}

/// Performance monitoring report
#[derive(Debug, Clone, PartialEq)]
pub struct PerformanceReport {
    /// Report timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Overall system score [0.0, 1.0]
    pub overall_score: f32,
    /// Metric trends over time
    pub metric_trends: HashMap<String, Vec<(chrono::DateTime<chrono::Utc>, f32)>>,
    /// Performance alerts
    pub alerts: Vec<PerformanceAlert>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Performance alert
#[derive(Debug, Clone, PartialEq)]
pub struct PerformanceAlert {
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Metric affected
    pub metric: String,
    /// Current value
    pub current_value: f32,
    /// Expected value
    pub expected_value: f32,
    /// Suggested action
    pub suggested_action: Option<String>,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum AlertSeverity {
    /// Information only
    Info,
    /// Warning level
    Warning,
    /// Critical issue
    Critical,
}

/// Demographic profile for MOS prediction adaptation
#[derive(Debug, Clone, PartialEq)]
pub struct DemographicProfile {
    /// Age group
    pub age_group: AgeGroup,
    /// Gender
    pub gender: Gender,
    /// Education level
    pub education_level: EducationLevel,
    /// Audio expertise level
    pub audio_expertise: AudioExpertise,
    /// Language proficiency
    pub language_proficiency: LanguageProficiency,
    /// Hearing ability
    pub hearing_ability: HearingAbility,
}

/// Cultural profile for perceptual modeling
#[derive(Debug, Clone, PartialEq)]
pub struct CulturalProfile {
    /// Cultural background
    pub cultural_background: CulturalBackground,
    /// Regional preferences
    pub regional_preferences: RegionalPreferences,
    /// Language group
    pub language_group: LanguageGroup,
    /// Speech style preferences
    pub speech_style_preferences: SpeechStylePreferences,
    /// Perceptual biases
    pub perceptual_biases: Vec<PerceptualBias>,
}

/// Age group categories
#[derive(Debug, Clone, PartialEq)]
pub enum AgeGroup {
    /// 18-25 years
    Young,
    /// 26-40 years
    MiddleAge,
    /// 41-60 years
    Mature,
    /// 60+ years
    Senior,
}

/// Gender categories
#[derive(Debug, Clone, PartialEq)]
pub enum Gender {
    /// Male
    Male,
    /// Female
    Female,
    /// Non-binary
    NonBinary,
    /// Prefer not to say
    PreferNotToSay,
}

/// Education level categories
#[derive(Debug, Clone, PartialEq)]
pub enum EducationLevel {
    /// High school or less
    HighSchool,
    /// Some college
    SomeCollege,
    /// Bachelor's degree
    Bachelor,
    /// Graduate degree
    Graduate,
    /// Professional degree
    Professional,
}

/// Audio expertise level
#[derive(Debug, Clone, PartialEq)]
pub enum AudioExpertise {
    /// No specialized knowledge
    Novice,
    /// Some audio experience
    Intermediate,
    /// Audio professional
    Expert,
    /// Audio researcher
    Researcher,
}

/// Language proficiency level
#[derive(Debug, Clone, PartialEq)]
pub enum LanguageProficiency {
    /// Native speaker
    Native,
    /// Fluent
    Fluent,
    /// Intermediate
    Intermediate,
    /// Basic
    Basic,
    /// Learning
    Learning,
}

/// Hearing ability
#[derive(Debug, Clone, PartialEq)]
pub enum HearingAbility {
    /// Normal hearing
    Normal,
    /// Mild hearing loss
    Mild,
    /// Moderate hearing loss
    Moderate,
    /// Severe hearing loss
    Severe,
    /// Uses hearing aid
    HearingAid,
}

/// Cultural background
#[derive(Debug, Clone, PartialEq)]
pub enum CulturalBackground {
    /// Western culture
    Western,
    /// East Asian
    EastAsian,
    /// South Asian
    SouthAsian,
    /// Middle Eastern
    MiddleEastern,
    /// African
    African,
    /// Latin American
    LatinAmerican,
    /// Mixed/Other
    Mixed,
}

/// Regional preferences
#[derive(Debug, Clone, PartialEq)]
pub enum RegionalPreferences {
    /// American English
    AmericanEnglish,
    /// British English
    BritishEnglish,
    /// Australian English
    AustralianEnglish,
    /// Canadian English
    CanadianEnglish,
    /// Other regional variant
    Other(String),
}

/// Language group
#[derive(Debug, Clone, PartialEq)]
pub enum LanguageGroup {
    /// Germanic languages
    Germanic,
    /// Romance languages
    Romance,
    /// Slavic languages
    Slavic,
    /// Sino-Tibetan languages
    SinoTibetan,
    /// Japonic languages
    Japonic,
    /// Austronesian languages
    Austronesian,
    /// Other language family
    Other(String),
}

/// Speech style preferences
#[derive(Debug, Clone, PartialEq)]
pub enum SpeechStylePreferences {
    /// Formal speech
    Formal,
    /// Casual speech
    Casual,
    /// Expressive speech
    Expressive,
    /// Neutral speech
    Neutral,
    /// Regional accent
    RegionalAccent,
}

/// Perceptual bias
#[derive(Debug, Clone, PartialEq)]
pub struct PerceptualBias {
    /// Bias type
    pub bias_type: BiasType,
    /// Bias strength [-1.0, 1.0]
    pub strength: f32,
    /// Description
    pub description: String,
}

/// Types of perceptual biases
#[derive(Debug, Clone, PartialEq)]
pub enum BiasType {
    /// Preference for higher pitch
    PitchPreference,
    /// Preference for faster speech
    SpeedPreference,
    /// Preference for formal pronunciation
    FormalityPreference,
    /// Accent familiarity bias
    AccentFamiliarity,
    /// Gender voice preference
    GenderPreference,
    /// Age voice preference
    AgePreference,
    /// Cultural speech pattern preference
    CulturalPatternPreference,
}
