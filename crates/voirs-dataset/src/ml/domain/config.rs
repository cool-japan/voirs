//! Domain adaptation configuration types
//!
//! This module contains all configuration structures for domain adaptation,
//! including domain definitions, adaptation strategies, and preprocessing settings.

use crate::LanguageCode;
use serde::{Deserialize, Serialize};

/// Domain adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainAdaptationConfig {
    /// Source domain configuration
    pub source_domain: DomainConfig,
    /// Target domain configuration
    pub target_domain: DomainConfig,
    /// Adaptation strategy
    pub adaptation_strategy: AdaptationStrategy,
    /// Domain shift detection settings
    pub shift_detection: ShiftDetectionConfig,
    /// Transfer learning configuration
    pub transfer_learning: TransferLearningConfig,
    /// Data mixing settings
    pub data_mixing: DataMixingConfig,
}

/// Domain configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainConfig {
    /// Domain name
    pub name: String,
    /// Domain type
    pub domain_type: DomainType,
    /// Language(s) in this domain
    pub languages: Vec<LanguageCode>,
    /// Audio characteristics
    pub audio_characteristics: AudioCharacteristics,
    /// Text characteristics
    pub text_characteristics: TextCharacteristics,
    /// Speaker characteristics
    pub speaker_characteristics: SpeakerCharacteristics,
    /// Domain-specific preprocessing
    pub preprocessing: Vec<PreprocessingStep>,
}

/// Types of domains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DomainType {
    /// Studio recording
    Studio,
    /// Field recording
    Field,
    /// Telephone/VoIP
    Telephone,
    /// Broadcast
    Broadcast,
    /// Meeting/Conference
    Meeting,
    /// Interview
    Interview,
    /// Spontaneous speech
    Spontaneous,
    /// Read speech
    Read,
    /// Emotional speech
    Emotional,
    /// Synthetic speech
    Synthetic,
    /// Child speech
    Child,
    /// Elderly speech
    Elderly,
    /// Accented speech
    Accented,
    /// Noisy environment
    Noisy,
    /// Reverberant environment
    Reverberant,
    /// Multiple speakers
    MultiSpeaker,
}

/// Audio characteristics of a domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioCharacteristics {
    /// Sample rate range (min, max)
    pub sample_rate_range: (u32, u32),
    /// Bit depth range (min, max)
    pub bit_depth_range: (u32, u32),
    /// Channel configuration
    pub channel_config: ChannelConfig,
    /// Noise characteristics
    pub noise_characteristics: Vec<String>,
    /// Frequency characteristics
    pub frequency_characteristics: FrequencyCharacteristics,
    /// Formant characteristics
    pub formant_characteristics: FormantCharacteristics,
    /// Dynamic range (dB)
    pub dynamic_range: f32,
    /// Recording quality score (0.0-1.0)
    pub quality_score: f32,
}

/// Channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChannelConfig {
    /// Mono
    Mono,
    /// Stereo
    Stereo,
    /// Multi-channel
    MultiChannel(u32),
}

/// Frequency characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyCharacteristics {
    /// Fundamental frequency range (Hz)
    pub f0_range: (f32, f32),
    /// Spectral centroid range (Hz)
    pub spectral_centroid_range: (f32, f32),
    /// Bandwidth (Hz)
    pub bandwidth: f32,
    /// High-frequency emphasis
    pub high_freq_emphasis: f32,
}

/// Formant characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormantCharacteristics {
    /// F1 range (Hz)
    pub f1_range: (f32, f32),
    /// F2 range (Hz)
    pub f2_range: (f32, f32),
    /// F3 range (Hz)
    pub f3_range: (f32, f32),
    /// Formant bandwidth
    pub formant_bandwidth: f32,
    /// Vowel space dispersion
    pub vowel_space_dispersion: f32,
}

/// Text characteristics of a domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextCharacteristics {
    /// Text style
    pub text_style: TextStyle,
    /// Vocabulary size
    pub vocabulary_size: usize,
    /// Average sentence length
    pub avg_sentence_length: f32,
    /// Text complexity metrics
    pub complexity: TextComplexity,
    /// Out-of-vocabulary rate
    pub oov_rate: f32,
    /// Punctuation density
    pub punctuation_density: f32,
    /// Capitalization patterns
    pub capitalization_patterns: Vec<String>,
}

/// Text style
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TextStyle {
    /// Formal written text
    Formal,
    /// Informal/conversational
    Informal,
    /// Technical documentation
    Technical,
    /// News/journalistic
    News,
    /// Literary
    Literary,
    /// Social media
    SocialMedia,
    /// Transcribed speech
    Transcribed,
    /// Phonetic transcription
    Phonetic,
}

/// Text complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextComplexity {
    /// Average word length
    pub avg_word_length: f32,
    /// Lexical diversity (TTR)
    pub lexical_diversity: f32,
    /// Syntactic complexity score
    pub syntactic_complexity: f32,
    /// Reading level (grade level)
    pub reading_level: f32,
}

/// Speaker characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerCharacteristics {
    /// Number of speakers
    pub num_speakers: usize,
    /// Gender distribution
    pub gender_distribution: GenderDistribution,
    /// Age distribution
    pub age_distribution: AgeDistribution,
    /// Native language distribution
    pub native_languages: Vec<LanguageCode>,
    /// Speaking rate characteristics
    pub speaking_rate: SpeakingRateCharacteristics,
    /// Pause characteristics
    pub pause_characteristics: PauseCharacteristics,
    /// Voice quality characteristics
    pub voice_quality: VoiceQualityCharacteristics,
}

/// Age distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgeDistribution {
    /// Children (0-12)
    pub children: f32,
    /// Teenagers (13-19)
    pub teenagers: f32,
    /// Young adults (20-35)
    pub young_adults: f32,
    /// Middle-aged (36-55)
    pub middle_aged: f32,
    /// Older adults (56+)
    pub older_adults: f32,
}

/// Gender distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenderDistribution {
    /// Male speakers
    pub male: f32,
    /// Female speakers
    pub female: f32,
    /// Other/unspecified
    pub other: f32,
}

/// Speaking rate characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakingRateCharacteristics {
    /// Average words per minute
    pub avg_wpm: f32,
    /// Speaking rate variance
    pub rate_variance: f32,
    /// Articulation rate (syllables/second)
    pub articulation_rate: f32,
}

/// Pause characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PauseCharacteristics {
    /// Average pause duration (seconds)
    pub avg_pause_duration: f32,
    /// Pause frequency (pauses per minute)
    pub pause_frequency: f32,
    /// Filled pause ratio
    pub filled_pause_ratio: f32,
}

/// Voice quality characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceQualityCharacteristics {
    /// Breathiness score (0.0-1.0)
    pub breathiness: f32,
    /// Roughness score (0.0-1.0)
    pub roughness: f32,
    /// Hoarseness score (0.0-1.0)
    pub hoarseness: f32,
    /// Nasality score (0.0-1.0)
    pub nasality: f32,
    /// Vocal strain score (0.0-1.0)
    pub vocal_strain: f32,
}

/// Preprocessing steps for domain adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreprocessingStep {
    /// Audio normalization
    AudioNormalization(NormalizationType),
    /// Audio filtering
    AudioFiltering(FilterType),
    /// Resampling
    Resampling {
        target_rate: u32,
        quality: ResamplingQuality,
    },
    /// Noise reduction
    NoiseReduction { method: String, aggressiveness: f32 },
    /// Text normalization
    TextNormalization {
        lowercase: bool,
        remove_punctuation: bool,
        expand_abbreviations: bool,
    },
    /// Phonetic transcription
    PhoneticTranscription {
        phoneme_set: String,
        stress_marking: bool,
    },
    /// Feature extraction
    FeatureExtraction {
        features: Vec<String>,
        window_size: f32,
        hop_length: f32,
    },
    /// Data augmentation
    DataAugmentation {
        techniques: Vec<String>,
        probability: f32,
    },
}

/// Audio normalization types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationType {
    /// Peak normalization
    Peak,
    /// RMS normalization
    RMS,
    /// LUFS normalization
    LUFS,
}

/// Audio filter types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterType {
    /// High-pass filter
    HighPass(f32),
    /// Low-pass filter
    LowPass(f32),
    /// Band-pass filter
    BandPass(f32, f32),
}

/// Resampling quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResamplingQuality {
    /// Low quality (fast)
    Low,
    /// Medium quality
    Medium,
    /// High quality (slow)
    High,
}

/// Adaptation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationStrategy {
    /// No adaptation (baseline)
    None,
    /// Domain adversarial training
    DomainAdversarial {
        lambda: f32,
        gradient_reversal: bool,
    },
    /// Fine-tuning
    FineTuning {
        learning_rate: f32,
        freeze_layers: Vec<String>,
    },
    /// Multi-task learning
    MultiTask {
        task_weights: Vec<f32>,
        shared_layers: Vec<String>,
    },
    /// Progressive neural networks
    Progressive {
        lateral_connections: bool,
        adapter_size: usize,
    },
    /// Maximum mean discrepancy
    MMD { kernel: MMDKernel, lambda: f32 },
    /// Deep adaptation networks
    DeepAdaptation {
        layers: Vec<String>,
        adaptation_factor: f32,
    },
    /// Conditional domain adaptation
    ConditionalAdaptation {
        conditioning_features: Vec<String>,
        alignment_method: AlignmentMethod,
    },
    /// Curriculum learning
    CurriculumLearning {
        schedule: AdaptationSchedule,
        difficulty_metric: String,
    },
}

/// Feature alignment methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlignmentMethod {
    /// Linear alignment
    Linear,
    /// Non-linear alignment
    NonLinear,
    /// Adversarial alignment
    Adversarial,
}

/// MMD kernel types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MMDKernel {
    /// Linear kernel
    Linear,
    /// RBF kernel
    RBF(f32),
    /// Polynomial kernel
    Polynomial(u32),
}

/// Adaptation schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationSchedule {
    /// Linear progression
    Linear,
    /// Exponential progression
    Exponential(f32),
    /// Step-wise progression
    StepWise(Vec<f32>),
}

/// Domain shift detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShiftDetectionConfig {
    /// Detection method
    pub method: ShiftDetectionMethod,
    /// Monitoring interval (samples)
    pub monitoring_interval: usize,
    /// Alert configuration
    pub alert_config: AlertConfig,
    /// Features to monitor
    pub monitored_features: Vec<MonitoredFeature>,
    /// Statistical significance threshold
    pub significance_threshold: f64,
    /// Window size for drift detection
    pub window_size: usize,
}

/// Domain shift detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShiftDetectionMethod {
    /// Statistical tests
    Statistical { test: StatisticalTest, alpha: f64 },
    /// Distance-based detection
    Distance {
        metric: DistanceMetric,
        threshold: f64,
    },
    /// Density-based detection
    Density {
        estimator: DensityEstimator,
        threshold: f64,
    },
    /// Model-based detection
    ModelBased {
        model_type: String,
        retraining_threshold: f64,
    },
    /// Ensemble detection
    Ensemble {
        methods: Vec<ShiftDetectionMethod>,
        voting_strategy: String,
    },
    /// Online change point detection
    ChangePoint { algorithm: String, penalty: f64 },
    /// Adversarial validation
    AdversarialValidation {
        classifier_threshold: f64,
        validation_split: f32,
    },
}

/// Statistical tests for shift detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatisticalTest {
    /// Kolmogorov-Smirnov test
    KolmogorovSmirnov,
    /// Mann-Whitney U test
    MannWhitneyU,
    /// Chi-square test
    ChiSquare,
    /// Anderson-Darling test
    AndersonDarling,
}

/// Distance metrics for shift detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Wasserstein distance
    Wasserstein,
    /// Jensen-Shannon divergence
    JensenShannon,
    /// KL divergence
    KLDivergence,
    /// Maximum mean discrepancy
    MMD,
}

/// Density estimators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DensityEstimator {
    /// Kernel density estimation
    KDE,
    /// Gaussian mixture model
    GMM,
    /// Histogram
    Histogram,
}

/// Features to monitor for domain shift
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MonitoredFeature {
    /// Audio features
    Audio(String),
    /// Text features
    Text(String),
    /// Speaker features
    Speaker(String),
    /// Model predictions
    Predictions,
    /// Model confidence
    Confidence,
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Enable alerts
    pub enabled: bool,
    /// Alert threshold
    pub threshold: f64,
    /// Alert severity levels
    pub severity_levels: Vec<AlertSeverity>,
    /// Notification methods
    pub notification_methods: Vec<String>,
    /// Cooldown period (seconds)
    pub cooldown_period: u64,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Transfer learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferLearningConfig {
    /// Source model path
    pub source_model_path: String,
    /// Layers to transfer
    pub transfer_layers: TransferLayers,
    /// Fine-tuning strategy
    pub fine_tuning_strategy: FineTuningStrategy,
    /// Regularization settings
    pub regularization: RegularizationConfig,
    /// Learning rate schedule
    pub learning_rate_schedule: Vec<(usize, f32)>,
}

/// Layers to transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransferLayers {
    /// All layers
    All,
    /// First N layers
    First(usize),
    /// Last N layers
    Last(usize),
    /// Specific layers
    Specific(Vec<String>),
}

/// Fine-tuning strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FineTuningStrategy {
    /// Freeze all transferred layers
    Freeze,
    /// Gradual unfreezing
    GradualUnfreeze {
        unfreeze_schedule: Vec<(usize, Vec<String>)>,
    },
    /// Discriminative fine-tuning
    Discriminative { layer_learning_rates: Vec<f32> },
    /// Progressive resizing
    ProgressiveResize { size_schedule: Vec<(usize, f32)> },
    /// Layer-wise adaptive rates
    LayerWiseAdaptive { adaptation_factors: Vec<f32> },
    /// Triangular learning rates
    TriangularLR {
        base_lr: f32,
        max_lr: f32,
        step_size: usize,
    },
    /// Cosine annealing
    CosineAnnealing { t_max: usize, eta_min: f32 },
}

/// Regularization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationConfig {
    /// L1 regularization strength
    pub l1_strength: f32,
    /// L2 regularization strength
    pub l2_strength: f32,
    /// Dropout rate
    pub dropout_rate: f32,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Validation split
    pub validation_split: f32,
}

/// Data mixing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataMixingConfig {
    /// Mixing strategy
    pub strategy: MixingStrategy,
    /// Source domain weight
    pub source_weight: f32,
    /// Target domain weight
    pub target_weight: f32,
    /// Curriculum learning schedule
    pub curriculum_schedule: CurriculumSchedule,
    /// Mixing schedule
    pub mixing_schedule: MixingSchedule,
    /// Batch mixing ratio
    pub batch_mixing_ratio: f32,
}

/// Data mixing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MixingStrategy {
    /// Random mixing
    Random,
    /// Balanced mixing
    Balanced,
    /// Importance-weighted mixing
    ImportanceWeighted { importance_weights: Vec<f32> },
    /// Adversarial mixing
    Adversarial { discriminator_strength: f32 },
    /// Curriculum mixing
    Curriculum { difficulty_progression: Vec<f32> },
    /// Adaptive mixing
    Adaptive {
        adaptation_rate: f32,
        performance_threshold: f32,
    },
    /// MixUp augmentation
    MixUp { alpha: f32, beta: f32 },
}

/// Curriculum learning schedules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CurriculumSchedule {
    /// Linear progression
    Linear,
    /// Exponential progression
    Exponential(f32),
    /// Step-wise progression
    StepWise(Vec<f32>),
    /// Performance-based
    PerformanceBased {
        performance_threshold: f32,
        patience: usize,
    },
}

/// Mixing schedules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MixingSchedule {
    /// Static mixing ratios
    Static,
    /// Dynamic mixing based on performance
    Dynamic {
        adjustment_rate: f32,
        performance_window: usize,
    },
    /// Scheduled mixing ratio changes
    Scheduled { schedule: Vec<(usize, f32)> },
    /// Adaptive mixing based on domain shift
    Adaptive {
        shift_threshold: f64,
        adjustment_factor: f32,
    },
}

impl Default for DomainAdaptationConfig {
    fn default() -> Self {
        Self {
            source_domain: DomainConfig::new(
                "default_source".to_string(),
                DomainType::Studio,
                vec![LanguageCode::EnUs],
            ),
            target_domain: DomainConfig::new(
                "default_target".to_string(),
                DomainType::Studio,
                vec![LanguageCode::EnUs],
            ),
            adaptation_strategy: AdaptationStrategy::FineTuning {
                learning_rate: 1e-4,
                freeze_layers: vec![],
            },
            shift_detection: ShiftDetectionConfig::default(),
            transfer_learning: TransferLearningConfig::default(),
            data_mixing: DataMixingConfig::default(),
        }
    }
}

impl Default for SpeakerCharacteristics {
    fn default() -> Self {
        Self {
            num_speakers: 1,
            gender_distribution: GenderDistribution {
                male: 0.5,
                female: 0.5,
                other: 0.0,
            },
            age_distribution: AgeDistribution {
                children: 0.0,
                teenagers: 0.1,
                young_adults: 0.4,
                middle_aged: 0.4,
                older_adults: 0.1,
            },
            native_languages: vec![LanguageCode::EnUs],
            speaking_rate: SpeakingRateCharacteristics {
                avg_wpm: 150.0,
                rate_variance: 20.0,
                articulation_rate: 4.5,
            },
            pause_characteristics: PauseCharacteristics {
                avg_pause_duration: 0.5,
                pause_frequency: 10.0,
                filled_pause_ratio: 0.1,
            },
            voice_quality: VoiceQualityCharacteristics {
                breathiness: 0.1,
                roughness: 0.1,
                hoarseness: 0.1,
                nasality: 0.1,
                vocal_strain: 0.1,
            },
        }
    }
}

impl Default for ShiftDetectionConfig {
    fn default() -> Self {
        Self {
            method: ShiftDetectionMethod::Statistical {
                test: StatisticalTest::KolmogorovSmirnov,
                alpha: 0.05,
            },
            monitoring_interval: 1000,
            alert_config: AlertConfig {
                enabled: true,
                threshold: 0.95,
                severity_levels: vec![AlertSeverity::Medium, AlertSeverity::High],
                notification_methods: vec!["log".to_string()],
                cooldown_period: 300,
            },
            monitored_features: vec![MonitoredFeature::Audio("mfcc".to_string())],
            significance_threshold: 0.05,
            window_size: 100,
        }
    }
}

impl Default for TransferLearningConfig {
    fn default() -> Self {
        Self {
            source_model_path: "models/source_model.pt".to_string(),
            transfer_layers: TransferLayers::All,
            fine_tuning_strategy: FineTuningStrategy::GradualUnfreeze {
                unfreeze_schedule: vec![(10, vec!["layer1".to_string()])],
            },
            regularization: RegularizationConfig {
                l1_strength: 0.0,
                l2_strength: 1e-4,
                dropout_rate: 0.1,
                early_stopping_patience: 10,
                validation_split: 0.2,
            },
            learning_rate_schedule: vec![(0, 1e-4), (50, 1e-5)],
        }
    }
}

impl Default for DataMixingConfig {
    fn default() -> Self {
        Self {
            strategy: MixingStrategy::Balanced,
            source_weight: 0.7,
            target_weight: 0.3,
            curriculum_schedule: CurriculumSchedule::Linear,
            mixing_schedule: MixingSchedule::Static,
            batch_mixing_ratio: 0.5,
        }
    }
}

impl DomainConfig {
    /// Create a new domain configuration
    pub fn new(name: String, domain_type: DomainType, languages: Vec<LanguageCode>) -> Self {
        Self {
            name,
            domain_type,
            languages,
            audio_characteristics: AudioCharacteristics {
                sample_rate_range: (16000, 48000),
                bit_depth_range: (16, 24),
                channel_config: ChannelConfig::Mono,
                noise_characteristics: vec!["clean".to_string()],
                frequency_characteristics: FrequencyCharacteristics {
                    f0_range: (80.0, 400.0),
                    spectral_centroid_range: (1000.0, 3000.0),
                    bandwidth: 8000.0,
                    high_freq_emphasis: 0.0,
                },
                formant_characteristics: FormantCharacteristics {
                    f1_range: (200.0, 800.0),
                    f2_range: (800.0, 2500.0),
                    f3_range: (1800.0, 3500.0),
                    formant_bandwidth: 50.0,
                    vowel_space_dispersion: 1.0,
                },
                dynamic_range: 60.0,
                quality_score: 0.8,
            },
            text_characteristics: TextCharacteristics {
                text_style: TextStyle::Formal,
                vocabulary_size: 10000,
                avg_sentence_length: 20.0,
                complexity: TextComplexity {
                    avg_word_length: 5.0,
                    lexical_diversity: 0.7,
                    syntactic_complexity: 0.5,
                    reading_level: 12.0,
                },
                oov_rate: 0.05,
                punctuation_density: 0.1,
                capitalization_patterns: vec!["sentence_case".to_string()],
            },
            speaker_characteristics: SpeakerCharacteristics::default(),
            preprocessing: vec![],
        }
    }

    /// Check if this domain is compatible with another domain
    pub fn is_compatible_with(&self, other: &DomainConfig) -> bool {
        // Check language compatibility
        let language_overlap = self
            .languages
            .iter()
            .any(|lang| other.languages.contains(lang));

        // Check audio characteristics compatibility
        let sample_rate_overlap = self.audio_characteristics.sample_rate_range.0
            <= other.audio_characteristics.sample_rate_range.1
            && self.audio_characteristics.sample_rate_range.1
                >= other.audio_characteristics.sample_rate_range.0;

        // Check text style compatibility
        let text_compatible = matches!(
            (
                &self.text_characteristics.text_style,
                &other.text_characteristics.text_style
            ),
            (TextStyle::Formal, TextStyle::Formal)
                | (TextStyle::Informal, TextStyle::Informal)
                | (TextStyle::Technical, TextStyle::Technical)
                | (TextStyle::Transcribed, _)
                | (_, TextStyle::Transcribed)
        );

        language_overlap && sample_rate_overlap && text_compatible
    }

    /// Calculate similarity score with another domain
    pub fn similarity_score(&self, other: &DomainConfig) -> f32 {
        let mut score = 0.0;
        let mut factors = 0;

        // Language similarity
        let language_overlap = self
            .languages
            .iter()
            .filter(|lang| other.languages.contains(lang))
            .count() as f32;
        let language_union =
            (self.languages.len() + other.languages.len()) as f32 - language_overlap;
        if language_union > 0.0 {
            score += language_overlap / language_union;
            factors += 1;
        }

        // Audio characteristics similarity
        let sr_overlap = (self
            .audio_characteristics
            .sample_rate_range
            .1
            .min(other.audio_characteristics.sample_rate_range.1)
            - self
                .audio_characteristics
                .sample_rate_range
                .0
                .max(other.audio_characteristics.sample_rate_range.0))
            as f32;
        let sr_union = (self
            .audio_characteristics
            .sample_rate_range
            .1
            .max(other.audio_characteristics.sample_rate_range.1)
            - self
                .audio_characteristics
                .sample_rate_range
                .0
                .min(other.audio_characteristics.sample_rate_range.0))
            as f32;
        if sr_union > 0.0 {
            score += (sr_overlap / sr_union).max(0.0);
            factors += 1;
        }

        // Text complexity similarity
        let complexity_diff = (self.text_characteristics.complexity.lexical_diversity
            - other.text_characteristics.complexity.lexical_diversity)
            .abs();
        score += 1.0 - complexity_diff;
        factors += 1;

        // Speaker characteristics similarity
        let gender_similarity = 1.0
            - ((self.speaker_characteristics.gender_distribution.male
                - other.speaker_characteristics.gender_distribution.male)
                .abs()
                + (self.speaker_characteristics.gender_distribution.female
                    - other.speaker_characteristics.gender_distribution.female)
                    .abs())
                / 2.0;
        score += gender_similarity;
        factors += 1;

        if factors > 0 {
            score / factors as f32
        } else {
            0.0
        }
    }

    /// Get preprocessing steps for adaptation to target domain
    pub fn get_adaptation_preprocessing(&self, target: &DomainConfig) -> Vec<PreprocessingStep> {
        let mut steps = vec![];

        // Audio preprocessing
        if self.audio_characteristics.sample_rate_range
            != target.audio_characteristics.sample_rate_range
        {
            let target_rate = target.audio_characteristics.sample_rate_range.1;
            steps.push(PreprocessingStep::Resampling {
                target_rate,
                quality: ResamplingQuality::High,
            });
        }

        // Noise reduction if target is cleaner
        if target.audio_characteristics.quality_score > self.audio_characteristics.quality_score {
            steps.push(PreprocessingStep::NoiseReduction {
                method: "spectral_subtraction".to_string(),
                aggressiveness: 0.5,
            });
        }

        // Text normalization if styles differ
        if self.text_characteristics.text_style != target.text_characteristics.text_style {
            steps.push(PreprocessingStep::TextNormalization {
                lowercase: true,
                remove_punctuation: false,
                expand_abbreviations: true,
            });
        }

        steps
    }
}

impl Default for DomainConfig {
    fn default() -> Self {
        Self::new(
            "default".to_string(),
            DomainType::Studio,
            vec![LanguageCode::EnUs],
        )
    }
}
