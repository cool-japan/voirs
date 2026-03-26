//! Data types for multilingual speaker model evaluation

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use voirs_sdk::LanguageCode;

/// Adaptation challenge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationChallenge {
    /// Challenge type
    pub challenge_type: AdaptationChallengeType,
    /// Challenge description
    pub description: String,
    /// Challenge severity
    pub severity: f32,
    /// Suggested solutions
    pub suggested_solutions: Vec<String>,
}

/// F0 adaptation for a specific language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F0Adaptation {
    /// Adaptation type
    pub adaptation_type: F0AdaptationType,
    /// Adaptation magnitude
    pub adaptation_magnitude: f32,
    /// Adaptation appropriateness
    pub adaptation_appropriateness: f32,
    /// Adaptation consistency
    pub adaptation_consistency: f32,
}
/// Type of formant adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FormantAdaptationType {
    /// No adaptation
    None,
    /// Frequency shifting
    FrequencyShifting,
    /// Bandwidth adjustment
    BandwidthAdjustment,
    /// Formant structure modification
    StructureModification,
    /// Complete remodeling
    CompleteRemodeling,
}
/// Prosodic characteristics analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodicCharacteristicsAnalysis {
    /// Intonation patterns per language
    pub intonation_patterns: HashMap<LanguageCode, IntonationPattern>,
    /// Stress patterns per language
    pub stress_patterns: HashMap<LanguageCode, StressPattern>,
    /// Prosodic consistency across languages
    pub prosodic_consistency: f32,
    /// Language-specific prosodic adaptations
    pub language_adaptations: HashMap<LanguageCode, ProsodicAdaptation>,
}
/// Type of F0 adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum F0AdaptationType {
    /// No adaptation
    None,
    /// Range scaling
    RangeScaling,
    /// Mean shifting
    MeanShifting,
    /// Contour modification
    ContourModification,
    /// Complete remodeling
    CompleteRemodeling,
}
/// Spectral statistics
#[derive(Debug, Clone)]
pub struct SpectralStatistics {
    /// Spectral centroid
    pub spectral_centroid: f32,
    /// Spectral spread
    pub spectral_spread: f32,
    /// Spectral tilt
    pub spectral_tilt: f32,
    /// Spectral roll-off
    pub spectral_rolloff: f32,
}
/// Temporal adaptation for a specific language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAdaptation {
    /// Adaptation type
    pub adaptation_type: TemporalAdaptationType,
    /// Adaptation magnitude
    pub adaptation_magnitude: f32,
    /// Adaptation appropriateness
    pub adaptation_appropriateness: f32,
    /// Adaptation consistency
    pub adaptation_consistency: f32,
}
/// Stress pattern characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressPattern {
    /// Stress type
    pub stress_type: StressType,
    /// Stress prominence
    pub stress_prominence: f32,
    /// Stress consistency
    pub stress_consistency: f32,
    /// Stress appropriateness
    pub stress_appropriateness: f32,
}
/// Specific adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecificAdaptation {
    /// Adaptation category
    pub category: AdaptationCategory,
    /// Adaptation description
    pub description: String,
    /// Adaptation effectiveness
    pub effectiveness: f32,
    /// Adaptation consistency
    pub consistency: f32,
}
/// Problematic language pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblematicLanguagePair {
    /// Source language
    pub source_language: LanguageCode,
    /// Target language
    pub target_language: LanguageCode,
    /// Problem severity
    pub problem_severity: f32,
    /// Problem types
    pub problem_types: Vec<LanguagePairProblemType>,
    /// Problem description
    pub problem_description: String,
    /// Improvement suggestions
    pub improvement_suggestions: Vec<String>,
}
/// Speaker model representation
#[derive(Debug, Clone)]
pub struct SpeakerModel {
    /// Speaker ID
    pub speaker_id: String,
    /// Reference language
    pub reference_language: LanguageCode,
    /// Voice characteristics
    pub voice_characteristics: VoiceCharacteristics,
    /// Supported languages
    pub supported_languages: Vec<LanguageCode>,
    /// Model quality per language
    pub language_quality: HashMap<LanguageCode, f32>,
}
/// Voice quality statistics
#[derive(Debug, Clone)]
pub struct VoiceQualityStatistics {
    /// Jitter
    pub jitter: f32,
    /// Shimmer
    pub shimmer: f32,
    /// Harmonic-to-noise ratio
    pub hnr: f32,
    /// Spectral noise
    pub spectral_noise: f32,
}
/// Type of temporal adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalAdaptationType {
    /// No adaptation
    None,
    /// Rate adjustment
    RateAdjustment,
    /// Pause modification
    PauseModification,
    /// Rhythm adjustment
    RhythmAdjustment,
    /// Complete remodeling
    CompleteRemodeling,
}
/// Intonation pattern characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntonationPattern {
    /// Intonation type
    pub intonation_type: IntonationType,
    /// Intonation range
    pub intonation_range: f32,
    /// Intonation variability
    pub intonation_variability: f32,
    /// Intonation appropriateness
    pub intonation_appropriateness: f32,
}
/// Language adaptation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageAdaptationResult {
    /// Language
    pub language: LanguageCode,
    /// Adaptation quality
    pub adaptation_quality: f32,
    /// Adaptation consistency
    pub adaptation_consistency: f32,
    /// Adaptation appropriateness
    pub adaptation_appropriateness: f32,
    /// Specific adaptations
    pub specific_adaptations: Vec<SpecificAdaptation>,
    /// Adaptation challenges
    pub adaptation_challenges: Vec<AdaptationChallenge>,
}
/// Spectral characteristics analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralCharacteristicsAnalysis {
    /// Spectral centroid per language
    pub spectral_centroid: HashMap<LanguageCode, f32>,
    /// Spectral spread per language
    pub spectral_spread: HashMap<LanguageCode, f32>,
    /// Spectral tilt per language
    pub spectral_tilt: HashMap<LanguageCode, f32>,
    /// Spectral consistency across languages
    pub spectral_consistency: f32,
    /// Language-specific spectral adaptations
    pub language_adaptations: HashMap<LanguageCode, SpectralAdaptation>,
}
/// Spectral adaptation for a specific language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralAdaptation {
    /// Adaptation type
    pub adaptation_type: SpectralAdaptationType,
    /// Adaptation magnitude
    pub adaptation_magnitude: f32,
    /// Adaptation appropriateness
    pub adaptation_appropriateness: f32,
    /// Adaptation consistency
    pub adaptation_consistency: f32,
}
/// F0 characteristics analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F0CharacteristicsAnalysis {
    /// Mean F0 per language
    pub mean_f0: HashMap<LanguageCode, f32>,
    /// F0 range per language
    pub f0_range: HashMap<LanguageCode, (f32, f32)>,
    /// F0 variability per language
    pub f0_variability: HashMap<LanguageCode, f32>,
    /// F0 consistency across languages
    pub f0_consistency: f32,
    /// Language-specific F0 adaptations
    pub language_adaptations: HashMap<LanguageCode, F0Adaptation>,
}
/// Voice quality characteristics analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceQualityCharacteristicsAnalysis {
    /// Jitter per language
    pub jitter: HashMap<LanguageCode, f32>,
    /// Shimmer per language
    pub shimmer: HashMap<LanguageCode, f32>,
    /// Harmonic-to-noise ratio per language
    pub harmonic_to_noise_ratio: HashMap<LanguageCode, f32>,
    /// Voice quality consistency across languages
    pub voice_quality_consistency: f32,
    /// Language-specific voice quality adaptations
    pub language_adaptations: HashMap<LanguageCode, VoiceQualityAdaptation>,
}
/// Type of intonation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntonationType {
    /// Rising
    Rising,
    /// Falling
    Falling,
    /// Level
    Level,
    /// Complex
    Complex,
}
/// Pause pattern characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PausePattern {
    /// Average pause duration
    pub average_pause_duration: f32,
    /// Pause frequency
    pub pause_frequency: f32,
    /// Pause variability
    pub pause_variability: f32,
    /// Pause appropriateness
    pub pause_appropriateness: f32,
}
/// Type of adaptation challenge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationChallengeType {
    /// Phonetic incompatibility
    PhoneticIncompatibility,
    /// Prosodic interference
    ProsodicInterference,
    /// Spectral mismatch
    SpectralMismatch,
    /// Temporal inconsistency
    TemporalInconsistency,
    /// Voice quality degradation
    VoiceQualityDegradation,
    /// Cultural inappropriateness
    CulturalInappropriateness,
}
/// F0 statistics
#[derive(Debug, Clone)]
pub struct F0Statistics {
    /// Mean F0
    pub mean_f0: f32,
    /// F0 standard deviation
    pub f0_std: f32,
    /// F0 range
    pub f0_range: (f32, f32),
    /// F0 variability
    pub f0_variability: f32,
}
/// Temporal statistics
#[derive(Debug, Clone)]
pub struct TemporalStatistics {
    /// Speaking rate
    pub speaking_rate: f32,
    /// Pause frequency
    pub pause_frequency: f32,
    /// Pause duration
    pub pause_duration: f32,
    /// Rhythm regularity
    pub rhythm_regularity: f32,
}
/// Temporal characteristics analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalCharacteristicsAnalysis {
    /// Speaking rate per language
    pub speaking_rate: HashMap<LanguageCode, f32>,
    /// Pause patterns per language
    pub pause_patterns: HashMap<LanguageCode, PausePattern>,
    /// Rhythm characteristics per language
    pub rhythm_characteristics: HashMap<LanguageCode, RhythmCharacteristics>,
    /// Temporal consistency across languages
    pub temporal_consistency: f32,
    /// Language-specific temporal adaptations
    pub language_adaptations: HashMap<LanguageCode, TemporalAdaptation>,
}
/// Multilingual speaker model evaluation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultilingualSpeakerModelConfig {
    /// Enable voice transfer quality assessment
    pub enable_voice_transfer_quality: bool,
    /// Enable speaker identity preservation analysis
    pub enable_speaker_identity_preservation: bool,
    /// Enable language-specific adaptation evaluation
    pub enable_language_adaptation: bool,
    /// Enable acoustic consistency analysis
    pub enable_acoustic_consistency: bool,
    /// Enable perceptual similarity assessment
    pub enable_perceptual_similarity: bool,
    /// Voice transfer quality weight
    pub voice_transfer_quality_weight: f32,
    /// Speaker identity preservation weight
    pub speaker_identity_preservation_weight: f32,
    /// Language adaptation weight
    pub language_adaptation_weight: f32,
    /// Acoustic consistency weight
    pub acoustic_consistency_weight: f32,
    /// Perceptual similarity weight
    pub perceptual_similarity_weight: f32,
    /// Minimum similarity threshold for speaker identification
    pub min_speaker_similarity_threshold: f32,
    /// Maximum acceptable voice transfer degradation
    pub max_voice_transfer_degradation: f32,
    /// Languages to evaluate
    pub target_languages: Vec<LanguageCode>,
}
/// Type of stress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StressType {
    /// Fixed stress
    Fixed,
    /// Variable stress
    Variable,
    /// Tonal
    Tonal,
}
/// Prosodic adaptation for a specific language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodicAdaptation {
    /// Adaptation type
    pub adaptation_type: ProsodicAdaptationType,
    /// Adaptation magnitude
    pub adaptation_magnitude: f32,
    /// Adaptation appropriateness
    pub adaptation_appropriateness: f32,
    /// Adaptation consistency
    pub adaptation_consistency: f32,
}
/// Type of spectral adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpectralAdaptationType {
    /// No adaptation
    None,
    /// Centroid shifting
    CentroidShifting,
    /// Spread modification
    SpreadModification,
    /// Tilt adjustment
    TiltAdjustment,
    /// Complete remodeling
    CompleteRemodeling,
}
/// Formant adaptation for a specific language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormantAdaptation {
    /// Adaptation type
    pub adaptation_type: FormantAdaptationType,
    /// Adaptation magnitude
    pub adaptation_magnitude: f32,
    /// Adaptation appropriateness
    pub adaptation_appropriateness: f32,
    /// Adaptation consistency
    pub adaptation_consistency: f32,
}
/// Multilingual speaker model evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultilingualSpeakerModelResult {
    /// Reference speaker language
    pub reference_language: LanguageCode,
    /// Target languages evaluated
    pub target_languages: Vec<LanguageCode>,
    /// Overall multilingual speaker model quality
    pub overall_quality: f32,
    /// Voice transfer quality scores
    pub voice_transfer_quality: HashMap<LanguageCode, f32>,
    /// Speaker identity preservation scores
    pub speaker_identity_preservation: HashMap<LanguageCode, f32>,
    /// Language adaptation scores
    pub language_adaptation: HashMap<LanguageCode, f32>,
    /// Acoustic consistency scores
    pub acoustic_consistency: HashMap<LanguageCode, f32>,
    /// Perceptual similarity scores
    pub perceptual_similarity: HashMap<LanguageCode, f32>,
    /// Cross-language similarity matrix
    pub cross_language_similarity: HashMap<(LanguageCode, LanguageCode), f32>,
    /// Voice characteristics analysis
    pub voice_characteristics: VoiceCharacteristicsAnalysis,
    /// Language-specific adaptations
    pub language_adaptations: HashMap<LanguageCode, LanguageAdaptationResult>,
    /// Problematic language pairs
    pub problematic_pairs: Vec<ProblematicLanguagePair>,
    /// Evaluation confidence
    pub evaluation_confidence: f32,
    /// Processing time
    pub processing_time: Duration,
}
/// Voice quality adaptation for a specific language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceQualityAdaptation {
    /// Adaptation type
    pub adaptation_type: VoiceQualityAdaptationType,
    /// Adaptation magnitude
    pub adaptation_magnitude: f32,
    /// Adaptation appropriateness
    pub adaptation_appropriateness: f32,
    /// Adaptation consistency
    pub adaptation_consistency: f32,
}
/// Formant characteristics analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormantCharacteristicsAnalysis {
    /// Formant frequencies per language
    pub formant_frequencies: HashMap<LanguageCode, Vec<f32>>,
    /// Formant bandwidths per language
    pub formant_bandwidths: HashMap<LanguageCode, Vec<f32>>,
    /// Formant consistency across languages
    pub formant_consistency: f32,
    /// Language-specific formant adaptations
    pub language_adaptations: HashMap<LanguageCode, FormantAdaptation>,
}
/// Type of voice quality adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoiceQualityAdaptationType {
    /// No adaptation
    None,
    /// Jitter adjustment
    JitterAdjustment,
    /// Shimmer adjustment
    ShimmerAdjustment,
    /// Noise reduction
    NoiseReduction,
    /// Complete remodeling
    CompleteRemodeling,
}
/// Adaptation category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationCategory {
    /// Phonetic adaptation
    Phonetic,
    /// Prosodic adaptation
    Prosodic,
    /// Spectral adaptation
    Spectral,
    /// Temporal adaptation
    Temporal,
    /// Voice quality adaptation
    VoiceQuality,
}
/// Type of language pair problem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LanguagePairProblemType {
    /// Voice transfer failure
    VoiceTransferFailure,
    /// Speaker identity loss
    SpeakerIdentityLoss,
    /// Adaptation inadequacy
    AdaptationInadequacy,
    /// Acoustic inconsistency
    AcousticInconsistency,
    /// Perceptual dissimilarity
    PerceptualDissimilarity,
}
/// Rhythm characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmCharacteristics {
    /// Rhythm regularity
    pub rhythm_regularity: f32,
    /// Stress pattern consistency
    pub stress_pattern_consistency: f32,
    /// Syllable timing variability
    pub syllable_timing_variability: f32,
    /// Rhythm appropriateness
    pub rhythm_appropriateness: f32,
}
/// Voice characteristics
#[derive(Debug, Clone)]
pub struct VoiceCharacteristics {
    /// Fundamental frequency statistics
    pub f0_stats: F0Statistics,
    /// Formant statistics
    pub formant_stats: FormantStatistics,
    /// Spectral statistics
    pub spectral_stats: SpectralStatistics,
    /// Temporal statistics
    pub temporal_stats: TemporalStatistics,
    /// Voice quality statistics
    pub voice_quality_stats: VoiceQualityStatistics,
    /// Prosodic statistics
    pub prosodic_stats: ProsodicStatistics,
}
/// Prosodic statistics
#[derive(Debug, Clone)]
pub struct ProsodicStatistics {
    /// Intonation range
    pub intonation_range: f32,
    /// Stress prominence
    pub stress_prominence: f32,
    /// Rhythm consistency
    pub rhythm_consistency: f32,
    /// Prosodic variability
    pub prosodic_variability: f32,
}
/// Formant statistics
#[derive(Debug, Clone)]
pub struct FormantStatistics {
    /// Mean formant frequencies
    pub mean_formants: Vec<f32>,
    /// Formant standard deviations
    pub formant_stds: Vec<f32>,
    /// Formant bandwidths
    pub formant_bandwidths: Vec<f32>,
}
/// Voice characteristics analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCharacteristicsAnalysis {
    /// Fundamental frequency characteristics
    pub f0_characteristics: F0CharacteristicsAnalysis,
    /// Formant characteristics
    pub formant_characteristics: FormantCharacteristicsAnalysis,
    /// Spectral characteristics
    pub spectral_characteristics: SpectralCharacteristicsAnalysis,
    /// Temporal characteristics
    pub temporal_characteristics: TemporalCharacteristicsAnalysis,
    /// Voice quality characteristics
    pub voice_quality_characteristics: VoiceQualityCharacteristicsAnalysis,
    /// Prosodic characteristics
    pub prosodic_characteristics: ProsodicCharacteristicsAnalysis,
}
/// Type of prosodic adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProsodicAdaptationType {
    /// No adaptation
    None,
    /// Intonation adjustment
    IntonationAdjustment,
    /// Stress modification
    StressModification,
    /// Rhythm adjustment
    RhythmAdjustment,
    /// Complete remodeling
    CompleteRemodeling,
}
