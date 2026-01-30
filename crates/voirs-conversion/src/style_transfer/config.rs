//! Configuration types for style transfer system

use serde::{Deserialize, Serialize};

/// Configuration for style transfer system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleTransferConfig {
    /// Enable style transfer
    pub enabled: bool,

    /// Content preservation weight (0.0 to 1.0)
    pub content_preservation_weight: f32,

    /// Style transfer strength (0.0 to 1.0)
    pub style_transfer_strength: f32,

    /// Quality threshold for transfer
    pub quality_threshold: f32,

    /// Transfer method selection
    pub transfer_method: StyleTransferMethod,

    /// Adaptation settings
    pub adaptation_settings: StyleAdaptationSettings,

    /// Feature extraction settings
    pub feature_extraction: FeatureExtractionSettings,

    /// Synthesis settings
    pub synthesis_settings: SynthesisSettings,

    /// Real-time processing settings
    pub realtime_settings: RealtimeProcessingSettings,
}

/// Style transfer method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StyleTransferMethod {
    /// Content-style decomposition
    ContentStyleDecomposition,

    /// Adversarial style transfer
    AdversarialTransfer,

    /// Cycle-consistent style transfer
    CycleConsistentTransfer,

    /// Neural style transfer
    NeuralStyleTransfer,

    /// Semantic style transfer
    SemanticStyleTransfer,

    /// Hierarchical style transfer
    HierarchicalTransfer,
}

/// Style adaptation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleAdaptationSettings {
    /// Adaptation learning rate
    pub learning_rate: f32,

    /// Number of adaptation iterations
    pub adaptation_iterations: usize,

    /// Regularization strength
    pub regularization_strength: f32,

    /// Content consistency weight
    pub content_consistency_weight: f32,

    /// Style consistency weight
    pub style_consistency_weight: f32,

    /// Perceptual loss weight
    pub perceptual_loss_weight: f32,

    /// Adversarial loss weight
    pub adversarial_loss_weight: f32,
}

/// Feature extraction settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractionSettings {
    /// Enable prosodic feature extraction
    pub enable_prosodic: bool,

    /// Enable spectral feature extraction
    pub enable_spectral: bool,

    /// Enable temporal feature extraction
    pub enable_temporal: bool,

    /// Enable semantic feature extraction
    pub enable_semantic: bool,

    /// Feature dimension
    pub feature_dimension: usize,

    /// Window size for analysis (ms)
    pub window_size: f32,

    /// Hop size for analysis (ms)
    pub hop_size: f32,

    /// Feature normalization method
    pub normalization_method: NormalizationMethod,
}

/// Normalization method for features
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// Z-score normalization
    ZScore,

    /// Min-max normalization
    MinMax,

    /// Unit normalization
    Unit,

    /// Quantile normalization
    Quantile,

    /// No normalization
    None,
}

/// Synthesis settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisSettings {
    /// Synthesis method
    pub synthesis_method: SynthesisMethod,

    /// Vocoder configuration
    pub vocoder_config: VocoderConfig,

    /// Post-processing settings
    pub post_processing: PostProcessingSettings,

    /// Quality enhancement settings
    pub quality_enhancement: QualityEnhancementSettings,
}

/// Synthesis method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SynthesisMethod {
    /// Neural vocoder synthesis
    NeuralVocoder,

    /// Parametric synthesis
    Parametric,

    /// Hybrid synthesis
    Hybrid,

    /// Direct waveform synthesis
    DirectWaveform,
}

/// Vocoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocoderConfig {
    /// Vocoder type
    pub vocoder_type: VocoderType,

    /// Hop length
    pub hop_length: usize,

    /// Filter length
    pub filter_length: usize,

    /// Window function
    pub window_function: String,

    /// Mel bins
    pub mel_bins: usize,

    /// Sample rate
    pub sample_rate: u32,
}

/// Vocoder type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VocoderType {
    /// HiFi-GAN vocoder
    HiFiGAN,

    /// WaveGlow vocoder
    WaveGlow,

    /// Parallel WaveGAN
    ParallelWaveGAN,

    /// MelGAN vocoder
    MelGAN,

    /// Universal vocoder
    Universal,
}

/// Post-processing settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostProcessingSettings {
    /// Enable noise reduction
    pub noise_reduction: bool,

    /// Enable dynamic range compression
    pub dynamic_range_compression: bool,

    /// Enable spectral enhancement
    pub spectral_enhancement: bool,

    /// Enable artifacts removal
    pub artifacts_removal: bool,

    /// Enhancement strength (0.0 to 1.0)
    pub enhancement_strength: f32,
}

/// Quality enhancement settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityEnhancementSettings {
    /// Enable super-resolution
    pub super_resolution: bool,

    /// Enable bandwidth extension
    pub bandwidth_extension: bool,

    /// Enable prosody enhancement
    pub prosody_enhancement: bool,

    /// Enhancement target quality
    pub target_quality: f32,

    /// Quality vs speed tradeoff
    pub quality_speed_tradeoff: f32,
}

/// Real-time processing settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeProcessingSettings {
    /// Enable real-time processing
    pub enabled: bool,

    /// Processing chunk size (samples)
    pub chunk_size: usize,

    /// Lookahead buffer size (samples)
    pub lookahead_size: usize,

    /// Maximum processing latency (ms)
    pub max_latency: f32,

    /// Enable GPU acceleration
    pub gpu_acceleration: bool,

    /// Thread pool size
    pub thread_pool_size: usize,
}

impl Default for StyleTransferConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            content_preservation_weight: 0.7,
            style_transfer_strength: 0.8,
            quality_threshold: 0.75,
            transfer_method: StyleTransferMethod::ContentStyleDecomposition,
            adaptation_settings: StyleAdaptationSettings {
                learning_rate: 0.001,
                adaptation_iterations: 50,
                regularization_strength: 0.01,
                content_consistency_weight: 1.0,
                style_consistency_weight: 1.0,
                perceptual_loss_weight: 0.5,
                adversarial_loss_weight: 0.1,
            },
            feature_extraction: FeatureExtractionSettings {
                enable_prosodic: true,
                enable_spectral: true,
                enable_temporal: true,
                enable_semantic: true,
                feature_dimension: 512,
                window_size: 25.0,
                hop_size: 10.0,
                normalization_method: NormalizationMethod::ZScore,
            },
            synthesis_settings: SynthesisSettings {
                synthesis_method: SynthesisMethod::NeuralVocoder,
                vocoder_config: VocoderConfig {
                    vocoder_type: VocoderType::HiFiGAN,
                    hop_length: 256,
                    filter_length: 1024,
                    window_function: "hann".to_string(),
                    mel_bins: 80,
                    sample_rate: 22050,
                },
                post_processing: PostProcessingSettings {
                    noise_reduction: true,
                    dynamic_range_compression: true,
                    spectral_enhancement: true,
                    artifacts_removal: true,
                    enhancement_strength: 0.5,
                },
                quality_enhancement: QualityEnhancementSettings {
                    super_resolution: true,
                    bandwidth_extension: true,
                    prosody_enhancement: true,
                    target_quality: 0.9,
                    quality_speed_tradeoff: 0.7,
                },
            },
            realtime_settings: RealtimeProcessingSettings {
                enabled: false,
                chunk_size: 1024,
                lookahead_size: 256,
                max_latency: 100.0,
                gpu_acceleration: true,
                thread_pool_size: 4,
            },
        }
    }
}
