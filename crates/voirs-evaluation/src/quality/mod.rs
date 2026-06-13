//! Quality evaluation implementations
//!
//! This module provides comprehensive quality evaluation capabilities including:
//! - Objective quality metrics (PESQ, STOI, MCD)
//! - Subjective quality prediction (MOS)
//! - Perceptual quality assessment
//! - Multi-dimensional quality analysis

pub mod advanced_metrics;
pub mod children;
pub mod cross_language_intelligibility;
pub mod ecosystem;
pub mod elderly_pathological;
pub mod emotion;
pub mod evaluator;
pub mod f0_tracking;
pub mod human_ai_naturalness_correlation;
pub mod language_specific;
pub mod listening_simulation;
pub mod mcd;
pub mod multilingual_speaker_models;
pub mod neural;
pub mod p56_loudness;
pub mod pesq;
pub mod polqa;
pub mod psychoacoustic;
pub mod quality_regression;
pub mod realtime_monitor;
pub mod si_sdr;
pub mod singing;
pub mod spectral_analysis;
pub mod stoi;
pub mod transfer_learning_evaluation;
pub mod universal_phoneme_mapping;
pub mod voice_quality_dsp;
pub mod vuv;

pub use evaluator::QualityEvaluator;
pub use quality_regression::{
    QualityRegressionConfig, QualityRegressionDetector, QualityRegressionMeasurement,
    QualityRegressionReport,
};

pub use advanced_metrics::{
    AdvancedQualityConfig, AdvancedQualityEvaluator, AudioMetadata, IntelligibilityDomainScores,
    MultiDomainQualityScore, NaturalnessDomainScores, PerceptualDomainScores,
    QualityDriftDirection, QualityMeasurement, QualityTrendPrediction, SpeakerCharacteristics,
    TechnicalDomainScores,
};
pub use children::{
    AgeGroup, ChildIntelligibilityResult, ChildNaturalnessResult, ChildrenEvaluationConfig,
    ChildrenEvaluationResult, ChildrenSpeechEvaluator, DevelopmentalAssessment,
    DevelopmentalMilestone, EducationalProgressResult, ListenerFamiliarity, PhonemeAcquisition,
    SpeechCharacteristics,
};
pub use cross_language_intelligibility::{
    AccentExposure, AcousticAnalysisResult, ContextDependencyFactors,
    CrossLanguageIntelligibilityConfig, CrossLanguageIntelligibilityEvaluationTrait,
    CrossLanguageIntelligibilityEvaluator, CrossLanguageIntelligibilityResult,
    IntelligibilityProblemType, ListenerProficiencyProfile, ListeningExperience,
    PhonemeIntelligibilityScore, ProblematicRegion, ProficiencyLevel, ProsodicFeatureAnalysis,
    VoiceQualityMetrics, WordIntelligibilityScore,
};
pub use elderly_pathological::{
    AgeRelatedChanges, AssistiveTechnologyResult, ClinicalAssessmentResult, CommunicationContext,
    CommunicationEffectivenessResult, ElderlyAgeGroup, ElderlyPathologicalConfig,
    ElderlyPathologicalEvaluator, ElderlyPathologicalResult,
    ListenerFamiliarity as ElderlyListenerFamiliarity, PathologicalCondition, PathologicalFeatures,
    SeverityLevel,
};
pub use emotion::{
    CrossCulturalExpressionResult, CulturalRegion, EmotionFrameResult, EmotionRecognitionResult,
    EmotionType, EmotionalEvaluationConfig, EmotionalIntensity, EmotionalSpeechEvaluationResult,
    EmotionalSpeechEvaluationTrait, EmotionalSpeechEvaluator, ExpressionFeatureAnalysis,
    ExpressionStyle, ExpressivenessTransferResult, PersonalityPreservationResult, PersonalityTrait,
    ProsodicFeatures, StyleConsistencyResult,
};
pub use f0_tracking::{F0Algorithm, F0Contour, F0Frame, F0Statistics, F0Tracker, F0TrackingConfig};
pub use human_ai_naturalness_correlation::{
    AgeGroup as HumanAiAgeGroup, AiNaturalnessMetrics, BiasAnalysisResults, CalibrationCurve,
    CalibrationCurveType, ClusterAnalysis, ClusterQualityMetrics, CorrelationAnalysisResult,
    CulturalBackground, DataQualityIndicators, DetailedNaturalnessMetrics, DriftAnalysis,
    DriftDirection, EnergyNaturalnessFeatures, ExperienceLevel, F0NaturalnessFeatures, Factor,
    FactorAnalysis, FormantNaturalnessFeatures, Gender, HearingAbility,
    HumanAiNaturalnessCorrelationConfig, HumanAiNaturalnessCorrelationEvaluationTrait,
    HumanAiNaturalnessCorrelationEvaluator, HumanAiNaturalnessCorrelationResult,
    HumanNaturalnessRating, HumanRatingsSummary, ModelQualityIndicators, MultiDimensionalAnalysis,
    PcaAnalysis, PerceptualModelCalibration, PrincipalComponent, QualityAssessment,
    RaterDemographics, Recommendation, ReliabilityIndicators, RhythmNaturalnessFeatures,
    SpectralEnvelopeNaturalnessFeatures, StatisticalSignificanceResults, TemporalDynamicsAnalysis,
    TemporalPattern, TemporalPatternType, TimedCorrelation, VoiceQualityNaturalnessFeatures,
};
pub use language_specific::{
    AccentEvaluationResult, AccentFeature, AccentFeatureType, AccentModel, CodeSwitchingResult,
    CulturalModel, CulturalPreferenceResult, LanguageSegment, LanguageSpecificConfig,
    LanguageSpecificEvaluationTrait, LanguageSpecificEvaluator, LanguageSpecificResult,
    LanguageSwitch, PhonemicEvaluationResult, ProsodyEvaluationResult, ProsodyModel,
    RhythmCharacteristics, SoundChangeAnalysis, SwitchType, TimingType,
};
pub use listening_simulation::{
    BiasModel, ListeningTestResult, ListeningTestSimulator, QualityScaleTransformer,
    ReliabilityAssessment, ResponsePattern, VirtualListener,
};
pub use mcd::{MCDEvaluator, MCDStatistics};
pub use multilingual_speaker_models::{
    AdaptationCategory, AdaptationChallenge, AdaptationChallengeType, F0Adaptation,
    F0AdaptationType, F0CharacteristicsAnalysis, F0Statistics as MultilingualF0Statistics,
    FormantAdaptation, FormantAdaptationType, FormantCharacteristicsAnalysis, FormantStatistics,
    IntonationPattern, IntonationType, LanguageAdaptationResult, LanguagePairProblemType,
    MultilingualSpeakerModelConfig, MultilingualSpeakerModelEvaluationTrait,
    MultilingualSpeakerModelEvaluator, MultilingualSpeakerModelResult, PausePattern,
    ProblematicLanguagePair, ProsodicAdaptation, ProsodicAdaptationType,
    ProsodicCharacteristicsAnalysis, ProsodicStatistics,
    RhythmCharacteristics as MultilingualRhythmCharacteristics, SpeakerModel, SpecificAdaptation,
    SpectralAdaptation, SpectralAdaptationType, SpectralCharacteristicsAnalysis,
    SpectralStatistics, StressPattern, StressType, TemporalAdaptation, TemporalAdaptationType,
    TemporalCharacteristicsAnalysis, TemporalStatistics, VoiceCharacteristics,
    VoiceCharacteristicsAnalysis, VoiceQualityAdaptation, VoiceQualityAdaptationType,
    VoiceQualityCharacteristicsAnalysis, VoiceQualityStatistics,
};
pub use neural::{
    ActivationType, AdversarialResult, AttentionModule, FeatureExtractor, ModelArchitecture,
    NeuralConfig, NeuralEvaluator, NeuralQualityAssessment, QualityPredictor, SelfSupervisedResult,
};
pub use pesq::PESQEvaluator;
pub use polqa::{PolqaBandwidth, PolqaEvaluator};
pub use psychoacoustic::{
    CriticalBand, PsychoacousticAnalysis, PsychoacousticConfig, PsychoacousticEvaluator,
    TemporalMaskingAnalysis,
};
pub use realtime_monitor::{
    AlertSeverity, DetailedQualityMetrics, FrequencyDomainMetrics, PerceptualMetrics, QualityAlert,
    QualityTrend, RealTimeQualityConfig, RealTimeQualityMetrics, RealTimeQualityMonitor,
    TimeDomainMetrics,
};
pub use si_sdr::{BatchSISdrResults, LanguageSISdrConfig, SISdrEvaluator, SISdrResult};
pub use singing::{
    BreathControlAnalysis, HarmonicStructure, MusicalExpressiveness, MusicalKey, MusicalNote,
    PitchAccuracyResult, SingerIdentity, SingingEvaluationConfig, SingingEvaluationResult,
    SingingEvaluator, Tempo, TimbreProfile, TimeSignature, VibratoAnalysis, VoiceType,
};
pub use spectral_analysis::{
    AdvancedSpectralAnalysis, AuditorySceneAnalysis, CochlearImplantAnalysis,
    CochlearImplantStrategy, GammatoneChannelResponse, HearingAidAnalysis, HearingAidDistortion,
    HearingAidType, SpectralAnalysisConfig, SpectralAnalyzer, SpectralComplexityMetrics,
    TemporalEnvelopeAnalysis,
};
pub use stoi::{LanguageSpecificParams, STOIConfidenceInterval, STOIEvaluator};
pub use transfer_learning_evaluation::{
    AdaptationChallenge as TransferAdaptationChallenge,
    AdaptationChallengeType as TransferAdaptationChallengeType, ConvergenceAnalysis,
    ConvergencePattern, DomainAdaptationResult, FewShotPerformance, ImplementationEffort,
    KnowledgeTransferAssessment, LearningCurvePoint, NegativeTransferDetectionResult,
    NegativeTransferSource, NegativeTransferSourceType, ProblematicTransferPair, StabilityMetrics,
    TransferHistoryEntry, TransferLearningEvaluationConfig, TransferLearningEvaluationResult,
    TransferLearningEvaluationTrait, TransferLearningEvaluator, TransferLearningMetrics,
    TransferOptimizationRecommendation, TransferOptimizationRecommendationType,
    TransferProblemType, TransferStabilityAnalysis,
};
pub use universal_phoneme_mapping::{
    AcousticFeatures, AirstreamMechanism, ArticulatoryFeatures, ConsonantFeatures,
    CrossLanguageMapping, DurationCharacteristics, F0Characteristics, MannerOfArticulation,
    MappingType, PerceptualFeatures, PhonemeCandidate, PhonemeConverageAnalysis, PhonemeMapping,
    PhonemeSimilarity, PlaceOfArticulation, UniversalPhoneme, UniversalPhonemeMapper,
    UniversalPhonemeMappingConfig, Voicing, VowelBackness, VowelFeatures, VowelHeight, VowelLength,
    VowelRoundness, VowelTenseness,
};
pub use vuv::{
    VuvAccuracy, VuvAlgorithm, VuvAlignment, VuvAnalysis, VuvAnalyzer, VuvComparison, VuvConfig,
    VuvFrame, VuvStatistics,
};
