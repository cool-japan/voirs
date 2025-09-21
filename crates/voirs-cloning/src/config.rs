//! Configuration for voice cloning

use crate::types::CloningMethod;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for voice cloning operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CloningConfig {
    /// Default cloning method
    pub default_method: CloningMethod,
    /// Output audio sample rate
    pub output_sample_rate: u32,
    /// Quality level (0.0 to 1.0)
    pub quality_level: f32,
    /// Enable GPU acceleration if available
    pub use_gpu: bool,
    /// Maximum number of concurrent cloning operations
    pub max_concurrent_operations: usize,
    /// Model configurations for different methods
    pub model_configs: HashMap<CloningMethod, ModelConfig>,
    /// Audio preprocessing settings
    pub preprocessing: PreprocessingConfig,
    /// Quality assessment settings
    pub quality_assessment: QualityAssessmentConfig,
    /// Performance optimization settings
    pub performance: PerformanceConfig,
    /// Enable cross-lingual voice cloning
    pub enable_cross_lingual: bool,
}

impl CloningConfig {
    /// Create a new cloning config with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for this config
    pub fn builder() -> CloningConfigBuilder {
        CloningConfigBuilder::new()
    }

    /// Validate the configuration
    pub fn validate(&self) -> crate::Result<()> {
        if self.output_sample_rate == 0 {
            return Err(crate::Error::Validation(
                "Output sample rate must be greater than 0".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&self.quality_level) {
            return Err(crate::Error::Validation(
                "Quality level must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.max_concurrent_operations == 0 {
            return Err(crate::Error::Validation(
                "Max concurrent operations must be greater than 0".to_string(),
            ));
        }

        // Validate model configs
        for (method, config) in &self.model_configs {
            config.validate().map_err(|e| {
                crate::Error::Validation(format!("Invalid config for {:?}: {}", method, e))
            })?;
        }

        self.preprocessing.validate()?;
        self.quality_assessment.validate()?;
        self.performance.validate()?;

        Ok(())
    }

    /// Get model config for a specific method
    pub fn get_model_config(&self, method: CloningMethod) -> ModelConfig {
        self.model_configs
            .get(&method)
            .cloned()
            .unwrap_or_else(|| ModelConfig::default_for_method(method))
    }
}

impl Default for CloningConfig {
    fn default() -> Self {
        let mut model_configs = HashMap::new();
        model_configs.insert(
            CloningMethod::ZeroShot,
            ModelConfig::default_for_method(CloningMethod::ZeroShot),
        );
        model_configs.insert(
            CloningMethod::OneShot,
            ModelConfig::default_for_method(CloningMethod::OneShot),
        );
        model_configs.insert(
            CloningMethod::FewShot,
            ModelConfig::default_for_method(CloningMethod::FewShot),
        );
        model_configs.insert(
            CloningMethod::Hybrid,
            ModelConfig::default_for_method(CloningMethod::Hybrid),
        );

        Self {
            default_method: CloningMethod::FewShot,
            output_sample_rate: 22050,
            quality_level: 0.8,
            use_gpu: true,
            max_concurrent_operations: 4,
            model_configs,
            preprocessing: PreprocessingConfig::default(),
            quality_assessment: QualityAssessmentConfig::default(),
            performance: PerformanceConfig::default(),
            enable_cross_lingual: false,
        }
    }
}

/// Model configuration for specific cloning methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model file path or identifier
    pub model_path: Option<String>,
    /// Model architecture type
    pub architecture: ModelArchitecture,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Model-specific parameters
    pub parameters: HashMap<String, f32>,
    /// Memory usage optimization
    pub memory_optimization: MemoryOptimization,
}

impl ModelConfig {
    /// Create default config for a specific cloning method
    pub fn default_for_method(method: CloningMethod) -> Self {
        let (architecture, embedding_dim) = match method {
            CloningMethod::ZeroShot => (ModelArchitecture::SpeakerEncoder, 256),
            CloningMethod::OneShot => (ModelArchitecture::MetaLearning, 512),
            CloningMethod::FewShot => (ModelArchitecture::AdaptationNetwork, 1024),
            CloningMethod::FineTuning => (ModelArchitecture::TransformerTTS, 1024),
            CloningMethod::VoiceConversion => (ModelArchitecture::VoiceConverter, 768),
            CloningMethod::Hybrid => (ModelArchitecture::AdaptationNetwork, 1024),
            CloningMethod::CrossLingual => (ModelArchitecture::TransformerTTS, 1024),
        };

        Self {
            model_path: None,
            architecture,
            embedding_dim,
            parameters: HashMap::new(),
            memory_optimization: MemoryOptimization::default(),
        }
    }

    /// Validate model configuration
    pub fn validate(&self) -> crate::Result<()> {
        if self.embedding_dim == 0 {
            return Err(crate::Error::Validation(
                "Embedding dimension must be greater than 0".to_string(),
            ));
        }

        if self.embedding_dim > 4096 {
            return Err(crate::Error::Validation(
                "Embedding dimension too large (max 4096)".to_string(),
            ));
        }

        self.memory_optimization.validate()?;

        Ok(())
    }
}

/// Model architecture types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelArchitecture {
    /// Speaker encoder for zero-shot cloning
    SpeakerEncoder,
    /// Meta-learning approach for one-shot cloning
    MetaLearning,
    /// Adaptation network for few-shot cloning
    AdaptationNetwork,
    /// Transformer-based TTS for fine-tuning
    TransformerTTS,
    /// Voice conversion model
    VoiceConverter,
    /// Custom architecture
    Custom(String),
}

impl ModelArchitecture {
    /// Get the string representation
    pub fn as_str(&self) -> &str {
        match self {
            ModelArchitecture::SpeakerEncoder => "speaker_encoder",
            ModelArchitecture::MetaLearning => "meta_learning",
            ModelArchitecture::AdaptationNetwork => "adaptation_network",
            ModelArchitecture::TransformerTTS => "transformer_tts",
            ModelArchitecture::VoiceConverter => "voice_converter",
            ModelArchitecture::Custom(name) => name,
        }
    }
}

/// Memory optimization settings
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryOptimization {
    /// Use gradient checkpointing
    pub gradient_checkpointing: bool,
    /// Model precision (16 or 32 bit)
    pub precision: ModelPrecision,
    /// Batch size for processing
    pub batch_size: usize,
    /// Maximum memory usage in MB
    pub max_memory_mb: Option<usize>,
}

impl MemoryOptimization {
    /// Validate memory optimization settings
    pub fn validate(&self) -> crate::Result<()> {
        if self.batch_size == 0 {
            return Err(crate::Error::Validation(
                "Batch size must be greater than 0".to_string(),
            ));
        }

        if let Some(max_memory) = self.max_memory_mb {
            if max_memory < 100 {
                return Err(crate::Error::Validation(
                    "Maximum memory must be at least 100MB".to_string(),
                ));
            }
        }

        Ok(())
    }
}

impl Default for MemoryOptimization {
    fn default() -> Self {
        Self {
            gradient_checkpointing: true,
            precision: ModelPrecision::Float16,
            batch_size: 1,
            max_memory_mb: Some(2048), // 2GB default
        }
    }
}

/// Model precision settings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelPrecision {
    /// 16-bit floating point
    Float16,
    /// 32-bit floating point
    Float32,
    /// Mixed precision
    Mixed,
}

/// Audio preprocessing configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Target sample rate for processing
    pub target_sample_rate: u32,
    /// Normalize audio amplitude
    pub normalize_audio: bool,
    /// Trim silence from start/end
    pub trim_silence: bool,
    /// Silence threshold for trimming
    pub silence_threshold: f32,
    /// Apply noise reduction
    pub noise_reduction: bool,
    /// Noise reduction strength (0.0 to 1.0)
    pub noise_reduction_strength: f32,
    /// Split long audio into segments
    pub segment_long_audio: bool,
    /// Maximum segment length in seconds
    pub max_segment_length: f32,
    /// Apply bandwidth extension
    pub bandwidth_extension: bool,
}

impl PreprocessingConfig {
    /// Validate preprocessing configuration
    pub fn validate(&self) -> crate::Result<()> {
        if self.target_sample_rate == 0 {
            return Err(crate::Error::Validation(
                "Target sample rate must be greater than 0".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&self.silence_threshold) {
            return Err(crate::Error::Validation(
                "Silence threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&self.noise_reduction_strength) {
            return Err(crate::Error::Validation(
                "Noise reduction strength must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.max_segment_length <= 0.0 {
            return Err(crate::Error::Validation(
                "Max segment length must be positive".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 22050,
            normalize_audio: true,
            trim_silence: true,
            silence_threshold: 0.01,
            noise_reduction: true,
            noise_reduction_strength: 0.3,
            segment_long_audio: true,
            max_segment_length: 10.0,
            bandwidth_extension: false,
        }
    }
}

/// Quality assessment configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityAssessmentConfig {
    /// Enable automatic quality assessment
    pub enabled: bool,
    /// Similarity threshold for acceptance
    pub similarity_threshold: f32,
    /// Quality metrics to compute
    pub metrics: Vec<QualityMetric>,
    /// Perceptual quality assessment
    pub perceptual_assessment: bool,
    /// Speaker verification threshold
    pub verification_threshold: f32,
}

impl QualityAssessmentConfig {
    /// Validate quality assessment configuration
    pub fn validate(&self) -> crate::Result<()> {
        if !(0.0..=1.0).contains(&self.similarity_threshold) {
            return Err(crate::Error::Validation(
                "Similarity threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&self.verification_threshold) {
            return Err(crate::Error::Validation(
                "Verification threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for QualityAssessmentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            similarity_threshold: 0.7,
            metrics: vec![
                QualityMetric::SpeakerSimilarity,
                QualityMetric::AudioQuality,
                QualityMetric::Naturalness,
            ],
            perceptual_assessment: true,
            verification_threshold: 0.8,
        }
    }
}

/// Quality metrics to compute
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QualityMetric {
    /// Speaker similarity score
    SpeakerSimilarity,
    /// Overall audio quality
    AudioQuality,
    /// Naturalness of speech
    Naturalness,
    /// Intelligibility score
    Intelligibility,
    /// Emotional expressiveness
    Expressiveness,
    /// Pronunciation accuracy
    Pronunciation,
}

impl QualityMetric {
    /// Get the string representation
    pub fn as_str(&self) -> &str {
        match self {
            QualityMetric::SpeakerSimilarity => "speaker_similarity",
            QualityMetric::AudioQuality => "audio_quality",
            QualityMetric::Naturalness => "naturalness",
            QualityMetric::Intelligibility => "intelligibility",
            QualityMetric::Expressiveness => "expressiveness",
            QualityMetric::Pronunciation => "pronunciation",
        }
    }
}

/// Performance optimization configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Number of worker threads
    pub num_threads: Option<usize>,
    /// Use CPU SIMD optimizations
    pub use_simd: bool,
    /// Cache size for embeddings
    pub embedding_cache_size: usize,
    /// Cache size for models
    pub model_cache_size: usize,
    /// Enable model quantization
    pub quantization: bool,
    /// Quantization precision
    pub quantization_bits: u8,
}

impl PerformanceConfig {
    /// Validate performance configuration
    pub fn validate(&self) -> crate::Result<()> {
        if let Some(threads) = self.num_threads {
            if threads == 0 {
                return Err(crate::Error::Validation(
                    "Number of threads must be greater than 0".to_string(),
                ));
            }
        }

        if self.embedding_cache_size == 0 {
            return Err(crate::Error::Validation(
                "Embedding cache size must be greater than 0".to_string(),
            ));
        }

        if self.model_cache_size == 0 {
            return Err(crate::Error::Validation(
                "Model cache size must be greater than 0".to_string(),
            ));
        }

        if self.quantization && (self.quantization_bits < 4 || self.quantization_bits > 32) {
            return Err(crate::Error::Validation(
                "Quantization bits must be between 4 and 32".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            num_threads: None, // Use system default
            use_simd: true,
            embedding_cache_size: 1000,
            model_cache_size: 10,
            quantization: false,
            quantization_bits: 8,
        }
    }
}

/// Builder for CloningConfig
#[derive(Debug, Clone)]
pub struct CloningConfigBuilder {
    config: CloningConfig,
}

impl CloningConfigBuilder {
    /// Create a new config builder
    pub fn new() -> Self {
        Self {
            config: CloningConfig::default(),
        }
    }

    /// Set default cloning method
    pub fn default_method(mut self, method: CloningMethod) -> Self {
        self.config.default_method = method;
        self
    }

    /// Set output sample rate
    pub fn output_sample_rate(mut self, sample_rate: u32) -> Self {
        self.config.output_sample_rate = sample_rate;
        self
    }

    /// Set quality level
    pub fn quality_level(mut self, level: f32) -> Self {
        self.config.quality_level = level.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable GPU usage
    pub fn use_gpu(mut self, use_gpu: bool) -> Self {
        self.config.use_gpu = use_gpu;
        self
    }

    /// Set maximum concurrent operations
    pub fn max_concurrent_operations(mut self, max: usize) -> Self {
        self.config.max_concurrent_operations = max;
        self
    }

    /// Set model config for a method
    pub fn model_config(mut self, method: CloningMethod, config: ModelConfig) -> Self {
        self.config.model_configs.insert(method, config);
        self
    }

    /// Set preprocessing config
    pub fn preprocessing(mut self, config: PreprocessingConfig) -> Self {
        self.config.preprocessing = config;
        self
    }

    /// Set quality assessment config
    pub fn quality_assessment(mut self, config: QualityAssessmentConfig) -> Self {
        self.config.quality_assessment = config;
        self
    }

    /// Set performance config
    pub fn performance(mut self, config: PerformanceConfig) -> Self {
        self.config.performance = config;
        self
    }

    /// Enable or disable cross-lingual cloning
    pub fn enable_cross_lingual(mut self, enable: bool) -> Self {
        self.config.enable_cross_lingual = enable;
        self
    }

    /// Build and validate the configuration
    pub fn build(self) -> crate::Result<CloningConfig> {
        self.config.validate()?;
        Ok(self.config)
    }

    /// Build without validation
    pub fn build_unchecked(self) -> CloningConfig {
        self.config
    }
}

impl Default for CloningConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloning_config_default() {
        let config = CloningConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.default_method, CloningMethod::FewShot);
        assert_eq!(config.output_sample_rate, 22050);
    }

    #[test]
    fn test_cloning_config_builder() {
        let config = CloningConfig::builder()
            .default_method(CloningMethod::OneShot)
            .output_sample_rate(16000)
            .quality_level(0.9)
            .use_gpu(false)
            .build()
            .unwrap();

        assert_eq!(config.default_method, CloningMethod::OneShot);
        assert_eq!(config.output_sample_rate, 16000);
        assert_eq!(config.quality_level, 0.9);
        assert!(!config.use_gpu);
    }

    #[test]
    fn test_config_validation() {
        let mut config = CloningConfig::default();

        // Test invalid sample rate
        config.output_sample_rate = 0;
        assert!(config.validate().is_err());

        config.output_sample_rate = 22050;

        // Test invalid quality level
        config.quality_level = 1.5;
        assert!(config.validate().is_err());

        config.quality_level = 0.8;

        // Test invalid max operations
        config.max_concurrent_operations = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_model_config_for_methods() {
        let zero_shot_config = ModelConfig::default_for_method(CloningMethod::ZeroShot);
        assert_eq!(
            zero_shot_config.architecture,
            ModelArchitecture::SpeakerEncoder
        );
        assert_eq!(zero_shot_config.embedding_dim, 256);

        let few_shot_config = ModelConfig::default_for_method(CloningMethod::FewShot);
        assert_eq!(
            few_shot_config.architecture,
            ModelArchitecture::AdaptationNetwork
        );
        assert_eq!(few_shot_config.embedding_dim, 1024);
    }

    #[test]
    fn test_preprocessing_config_validation() {
        let mut config = PreprocessingConfig::default();
        assert!(config.validate().is_ok());

        config.target_sample_rate = 0;
        assert!(config.validate().is_err());

        config.target_sample_rate = 22050;
        config.silence_threshold = 1.5;
        assert!(config.validate().is_err());

        config.silence_threshold = 0.01;
        config.max_segment_length = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_quality_assessment_config() {
        let config = QualityAssessmentConfig::default();
        assert!(config.validate().is_ok());
        assert!(config.enabled);
        assert!(config.metrics.contains(&QualityMetric::SpeakerSimilarity));
    }

    #[test]
    fn test_performance_config() {
        let config = PerformanceConfig::default();
        assert!(config.validate().is_ok());
        assert!(config.use_simd);
        assert_eq!(config.embedding_cache_size, 1000);
    }

    #[test]
    fn test_memory_optimization() {
        let mem_opt = MemoryOptimization::default();
        assert!(mem_opt.validate().is_ok());
        assert!(mem_opt.gradient_checkpointing);
        assert_eq!(mem_opt.precision, ModelPrecision::Float16);
    }

    #[test]
    fn test_quality_metric_string_conversion() {
        assert_eq!(
            QualityMetric::SpeakerSimilarity.as_str(),
            "speaker_similarity"
        );
        assert_eq!(QualityMetric::AudioQuality.as_str(), "audio_quality");
        assert_eq!(QualityMetric::Naturalness.as_str(), "naturalness");
    }

    #[test]
    fn test_model_architecture_string_conversion() {
        assert_eq!(
            ModelArchitecture::SpeakerEncoder.as_str(),
            "speaker_encoder"
        );
        assert_eq!(ModelArchitecture::MetaLearning.as_str(), "meta_learning");
        assert_eq!(
            ModelArchitecture::AdaptationNetwork.as_str(),
            "adaptation_network"
        );
    }
}
