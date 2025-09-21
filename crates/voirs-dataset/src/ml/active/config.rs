//! Active learning configuration types
//!
//! This module contains all configuration structures for active learning,
//! including sampling strategies, uncertainty metrics, and human-in-the-loop settings.

use serde::{Deserialize, Serialize};

/// Active learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveLearningConfig {
    /// Sampling strategy
    pub sampling_strategy: SamplingStrategy,
    /// Uncertainty metric to use
    pub uncertainty_metric: UncertaintyMetric,
    /// Diversity configuration
    pub diversity_config: DiversityConfig,
    /// Human-in-the-loop settings
    pub human_loop_config: HumanLoopConfig,
    /// Batch size for active learning
    pub batch_size: usize,
    /// Maximum iterations
    pub max_iterations: usize,
}

/// Sampling strategies for active learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Uncertainty-based sampling
    Uncertainty {
        /// Uncertainty threshold
        threshold: f32,
        /// Combine with diversity
        use_diversity: bool,
    },
    /// Diversity-based sampling
    Diversity {
        /// Diversity metric
        metric: DiversityMetric,
        /// Minimum diversity threshold
        min_diversity: f32,
    },
    /// Query by committee
    QueryByCommittee {
        /// Number of committee members
        committee_size: usize,
        /// Disagreement threshold
        disagreement_threshold: f32,
    },
    /// Expected model change
    ExpectedModelChange {
        /// Change threshold
        change_threshold: f32,
    },
    /// Information density
    InformationDensity {
        /// Beta parameter for density weighting
        beta: f32,
    },
    /// Batch mode sampling
    BatchMode {
        /// Strategy to use within batch
        inner_strategy: Box<SamplingStrategy>,
        /// Diversification factor
        diversification_factor: f32,
    },
    /// Hybrid approach
    Hybrid {
        /// Primary strategy
        primary: Box<SamplingStrategy>,
        /// Secondary strategy
        secondary: Box<SamplingStrategy>,
        /// Weighting between strategies
        primary_weight: f32,
    },
}

/// Uncertainty metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UncertaintyMetric {
    /// Entropy-based uncertainty
    Entropy,
    /// Margin sampling
    Margin,
    /// Least confidence
    LeastConfidence,
    /// Variance-based (for regression)
    Variance,
    /// Ensemble disagreement
    EnsembleDisagreement,
    /// BALD (Bayesian Active Learning by Disagreement)
    BALD,
    /// Mutual information
    MutualInformation,
}

/// Diversity metrics for sample selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiversityMetric {
    /// Cosine distance
    CosineDistance,
    /// Euclidean distance
    EuclideanDistance,
    /// K-means clustering diversity
    KMeansClustering,
    /// Maximum mean discrepancy
    MMD,
    /// Determinantal point processes
    DPP,
}

/// Diversity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityConfig {
    /// Weight for diversity vs uncertainty
    pub diversity_weight: f32,
    /// Feature space to use for diversity calculation
    pub feature_space: DiversityFeatureSpace,
    /// Number of clusters for clustering-based diversity
    pub num_clusters: usize,
    /// Minimum distance threshold
    pub min_distance_threshold: f32,
    /// Use dimensionality reduction
    pub use_dimensionality_reduction: bool,
    /// Target dimensionality
    pub target_dimensions: usize,
}

/// Feature spaces for diversity calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiversityFeatureSpace {
    /// Raw audio features
    AudioFeatures,
    /// Text embeddings
    TextEmbeddings,
    /// Spectral features
    SpectralFeatures,
    /// Learned representations
    LearnedRepresentations,
    /// Combined features
    Combined(Vec<DiversityFeatureSpace>),
    /// Custom feature extractor
    Custom(String),
}

/// Human-in-the-loop configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanLoopConfig {
    /// Annotation interface type
    pub interface_type: AnnotationInterfaceType,
    /// Quality assurance settings
    pub quality_assurance: QualityAssuranceConfig,
    /// Annotator configuration
    pub annotator_config: AnnotatorConfig,
    /// Feedback configuration
    pub feedback_config: FeedbackConfig,
    /// Enable real-time annotation
    pub enable_realtime: bool,
    /// Annotation timeout (seconds)
    pub annotation_timeout: Option<u64>,
}

/// Types of annotation interfaces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnnotationInterfaceType {
    /// Web-based interface
    Web {
        /// Server port
        port: u16,
        /// Enable audio playback
        enable_audio_playback: bool,
        /// Show spectrograms
        show_spectrograms: bool,
        /// Custom CSS file
        custom_css: Option<String>,
    },
    /// Command line interface
    CLI {
        /// Use colors
        use_colors: bool,
        /// Show progress bars
        show_progress: bool,
        /// Auto-play audio
        auto_play_audio: bool,
    },
    /// API-based interface
    API {
        /// API endpoint
        endpoint: String,
        /// Authentication token
        auth_token: Option<String>,
        /// Request timeout
        timeout: u64,
    },
    /// Custom interface
    Custom(String),
}

/// Quality assurance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssuranceConfig {
    /// Enable multi-annotator consensus
    pub enable_consensus: bool,
    /// Minimum agreement threshold
    pub min_agreement_threshold: f32,
    /// Number of annotators per sample
    pub annotators_per_sample: usize,
    /// Enable expert review
    pub enable_expert_review: bool,
    /// Review sample percentage
    pub review_sample_percentage: f32,
}

/// Annotator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatorConfig {
    /// Minimum expertise level required
    pub min_expertise_level: ExpertiseLevel,
    /// Enable annotator training
    pub enable_training: bool,
    /// Training sample count
    pub training_sample_count: usize,
    /// Track annotator performance
    pub track_performance: bool,
    /// Performance feedback frequency
    pub performance_feedback_frequency: u32,
}

/// Expertise levels for annotators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpertiseLevel {
    Novice,
    Intermediate,
    Expert,
    Specialist,
}

/// Feedback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackConfig {
    /// Enable immediate feedback
    pub enable_immediate_feedback: bool,
    /// Feedback update frequency
    pub update_frequency: UpdateFrequency,
    /// Use model predictions as suggestions
    pub use_model_suggestions: bool,
    /// Confidence threshold for suggestions
    pub suggestion_confidence_threshold: f32,
    /// Enable uncertainty visualization
    pub enable_uncertainty_visualization: bool,
    /// Show similar samples
    pub show_similar_samples: bool,
    /// Number of similar samples to show
    pub num_similar_samples: usize,
}

/// Update frequency for feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateFrequency {
    /// Update after each annotation
    Immediate,
    /// Update after a batch
    Batch,
    /// Update after a fixed number of samples
    FixedCount(usize),
    /// Update after a time interval
    TimeInterval(u64), // seconds
    /// Manual update
    Manual,
}

impl Default for ActiveLearningConfig {
    fn default() -> Self {
        Self {
            sampling_strategy: SamplingStrategy::Uncertainty {
                threshold: 0.5,
                use_diversity: false,
            },
            uncertainty_metric: UncertaintyMetric::Entropy,
            diversity_config: DiversityConfig::default(),
            human_loop_config: HumanLoopConfig::default(),
            batch_size: 10,
            max_iterations: 100,
        }
    }
}

impl Default for DiversityConfig {
    fn default() -> Self {
        Self {
            diversity_weight: 0.3,
            feature_space: DiversityFeatureSpace::AudioFeatures,
            num_clusters: 5,
            min_distance_threshold: 0.1,
            use_dimensionality_reduction: false,
            target_dimensions: 50,
        }
    }
}

impl Default for HumanLoopConfig {
    fn default() -> Self {
        Self {
            interface_type: AnnotationInterfaceType::CLI {
                use_colors: true,
                show_progress: true,
                auto_play_audio: false,
            },
            quality_assurance: QualityAssuranceConfig::default(),
            annotator_config: AnnotatorConfig::default(),
            feedback_config: FeedbackConfig::default(),
            enable_realtime: false,
            annotation_timeout: Some(300), // 5 minutes
        }
    }
}

impl Default for QualityAssuranceConfig {
    fn default() -> Self {
        Self {
            enable_consensus: false,
            min_agreement_threshold: 0.8,
            annotators_per_sample: 1,
            enable_expert_review: false,
            review_sample_percentage: 0.1,
        }
    }
}

impl Default for AnnotatorConfig {
    fn default() -> Self {
        Self {
            min_expertise_level: ExpertiseLevel::Novice,
            enable_training: true,
            training_sample_count: 10,
            track_performance: true,
            performance_feedback_frequency: 10,
        }
    }
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            enable_immediate_feedback: true,
            update_frequency: UpdateFrequency::Batch,
            use_model_suggestions: true,
            suggestion_confidence_threshold: 0.8,
            enable_uncertainty_visualization: true,
            show_similar_samples: true,
            num_similar_samples: 3,
        }
    }
}

impl SamplingStrategy {
    /// Check if this strategy uses uncertainty
    pub fn uses_uncertainty(&self) -> bool {
        matches!(
            self,
            SamplingStrategy::Uncertainty { .. }
                | SamplingStrategy::QueryByCommittee { .. }
                | SamplingStrategy::ExpectedModelChange { .. }
                | SamplingStrategy::Hybrid { .. }
        )
    }

    /// Check if this strategy uses diversity
    pub fn uses_diversity(&self) -> bool {
        matches!(
            self,
            SamplingStrategy::Diversity { .. }
                | SamplingStrategy::InformationDensity { .. }
                | SamplingStrategy::BatchMode { .. }
        ) || matches!(
            self,
            SamplingStrategy::Uncertainty {
                use_diversity: true,
                ..
            }
        )
    }

    /// Get the batch size multiplier for this strategy
    pub fn batch_size_multiplier(&self) -> f32 {
        match self {
            SamplingStrategy::QueryByCommittee { committee_size, .. } => *committee_size as f32,
            SamplingStrategy::BatchMode { .. } => 2.0,
            _ => 1.0,
        }
    }
}

impl UncertaintyMetric {
    /// Check if this metric requires ensemble models
    pub fn requires_ensemble(&self) -> bool {
        matches!(
            self,
            UncertaintyMetric::EnsembleDisagreement | UncertaintyMetric::BALD
        )
    }

    /// Check if this metric is suitable for classification
    pub fn is_classification_metric(&self) -> bool {
        matches!(
            self,
            UncertaintyMetric::Entropy
                | UncertaintyMetric::Margin
                | UncertaintyMetric::LeastConfidence
        )
    }

    /// Check if this metric is suitable for regression
    pub fn is_regression_metric(&self) -> bool {
        matches!(self, UncertaintyMetric::Variance)
    }
}

impl DiversityConfig {
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.diversity_weight < 0.0 || self.diversity_weight > 1.0 {
            return Err("Diversity weight must be between 0.0 and 1.0".to_string());
        }

        if self.num_clusters == 0 {
            return Err("Number of clusters must be greater than 0".to_string());
        }

        if self.min_distance_threshold < 0.0 {
            return Err("Minimum distance threshold must be non-negative".to_string());
        }

        if self.use_dimensionality_reduction && self.target_dimensions == 0 {
            return Err(
                "Target dimensions must be greater than 0 when using dimensionality reduction"
                    .to_string(),
            );
        }

        Ok(())
    }

    /// Get effective feature space
    pub fn effective_feature_space(&self) -> &DiversityFeatureSpace {
        &self.feature_space
    }
}

impl HumanLoopConfig {
    /// Check if real-time annotation is enabled
    pub fn is_realtime_enabled(&self) -> bool {
        self.enable_realtime
    }

    /// Get annotation timeout in milliseconds
    pub fn timeout_ms(&self) -> Option<u64> {
        self.annotation_timeout.map(|t| t * 1000)
    }

    /// Check if quality assurance is enabled
    pub fn has_quality_assurance(&self) -> bool {
        self.quality_assurance.enable_consensus || self.quality_assurance.enable_expert_review
    }
}

impl AnnotationInterfaceType {
    /// Get the interface name
    pub fn name(&self) -> &str {
        match self {
            AnnotationInterfaceType::Web { .. } => "web",
            AnnotationInterfaceType::CLI { .. } => "cli",
            AnnotationInterfaceType::API { .. } => "api",
            AnnotationInterfaceType::Custom(name) => name,
        }
    }

    /// Check if this interface supports audio playback
    pub fn supports_audio_playback(&self) -> bool {
        match self {
            AnnotationInterfaceType::Web {
                enable_audio_playback,
                ..
            } => *enable_audio_playback,
            AnnotationInterfaceType::CLI {
                auto_play_audio, ..
            } => *auto_play_audio,
            AnnotationInterfaceType::API { .. } => false,
            AnnotationInterfaceType::Custom(_) => false,
        }
    }

    /// Check if this interface supports visual feedback
    pub fn supports_visual_feedback(&self) -> bool {
        match self {
            AnnotationInterfaceType::Web {
                show_spectrograms, ..
            } => *show_spectrograms,
            AnnotationInterfaceType::CLI { use_colors, .. } => *use_colors,
            AnnotationInterfaceType::API { .. } => false,
            AnnotationInterfaceType::Custom(_) => false,
        }
    }
}

impl ExpertiseLevel {
    /// Get the numeric level
    pub fn level(&self) -> u8 {
        match self {
            ExpertiseLevel::Novice => 1,
            ExpertiseLevel::Intermediate => 2,
            ExpertiseLevel::Expert => 3,
            ExpertiseLevel::Specialist => 4,
        }
    }

    /// Check if this level meets the requirement
    pub fn meets_requirement(&self, required: &ExpertiseLevel) -> bool {
        self.level() >= required.level()
    }
}

impl FeedbackConfig {
    /// Check if feedback is enabled
    pub fn is_feedback_enabled(&self) -> bool {
        self.enable_immediate_feedback || !matches!(self.update_frequency, UpdateFrequency::Manual)
    }

    /// Check if suggestions are enabled
    pub fn suggestions_enabled(&self) -> bool {
        self.use_model_suggestions && self.suggestion_confidence_threshold > 0.0
    }

    /// Check if visualization features are enabled
    pub fn has_visualization(&self) -> bool {
        self.enable_uncertainty_visualization || self.show_similar_samples
    }
}
