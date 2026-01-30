//! Configuration types for zero-shot voice conversion

use serde::{Deserialize, Serialize};

/// Configuration for zero-shot voice conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroShotConfig {
    /// Enable zero-shot conversion
    pub enabled: bool,

    /// Quality threshold for reference selection
    pub quality_threshold: f32,

    /// Maximum number of reference voices to consider
    pub max_references: usize,

    /// Similarity threshold for voice matching
    pub similarity_threshold: f32,

    /// Conversion method selection
    pub conversion_method: ZeroShotMethod,

    /// Adaptation settings
    pub adaptation_settings: AdaptationSettings,

    /// Performance constraints
    pub performance_constraints: PerformanceConstraints,

    /// Quality preservation settings
    pub quality_preservation: QualityPreservationSettings,
}

/// Zero-shot conversion method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZeroShotMethod {
    /// Embedding interpolation method
    EmbeddingInterpolation,

    /// Style transfer method
    StyleTransfer,

    /// Neural adaptation method
    NeuralAdaptation,

    /// Hybrid approach
    Hybrid,

    /// Direct synthesis method
    DirectSynthesis,
}

/// Adaptation settings for zero-shot conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationSettings {
    /// Learning rate for adaptation
    pub learning_rate: f32,

    /// Number of adaptation steps
    pub adaptation_steps: usize,

    /// Regularization strength
    pub regularization: f32,

    /// Use adversarial training
    pub adversarial_training: bool,

    /// Feature alignment weight
    pub feature_alignment_weight: f32,

    /// Content preservation weight
    pub content_preservation_weight: f32,
}

/// Performance constraints for zero-shot conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConstraints {
    /// Maximum processing time per conversion (ms)
    pub max_processing_time: f32,

    /// Maximum memory usage (MB)
    pub max_memory_usage: f32,

    /// Target real-time factor
    pub target_rtf: f32,

    /// Enable GPU acceleration
    pub gpu_acceleration: bool,

    /// Batch processing size
    pub batch_size: usize,
}

/// Quality preservation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityPreservationSettings {
    /// Minimum output quality threshold
    pub min_quality_threshold: f32,

    /// Enable quality monitoring
    pub quality_monitoring: bool,

    /// Automatic quality adjustment
    pub auto_quality_adjustment: bool,

    /// Quality vs speed tradeoff (0.0 = speed, 1.0 = quality)
    pub quality_speed_tradeoff: f32,

    /// Enable content verification
    pub content_verification: bool,
}

impl Default for ZeroShotConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            quality_threshold: 0.7,
            max_references: 10,
            similarity_threshold: 0.6,
            conversion_method: ZeroShotMethod::Hybrid,
            adaptation_settings: AdaptationSettings {
                learning_rate: 0.001,
                adaptation_steps: 100,
                regularization: 0.01,
                adversarial_training: true,
                feature_alignment_weight: 1.0,
                content_preservation_weight: 1.0,
            },
            performance_constraints: PerformanceConstraints {
                max_processing_time: 1000.0,
                max_memory_usage: 500.0,
                target_rtf: 0.1,
                gpu_acceleration: true,
                batch_size: 4,
            },
            quality_preservation: QualityPreservationSettings {
                min_quality_threshold: 0.6,
                quality_monitoring: true,
                auto_quality_adjustment: true,
                quality_speed_tradeoff: 0.7,
                content_verification: true,
            },
        }
    }
}
