//! Unified emotion configuration and builder patterns
//!
//! This module provides a comprehensive unified configuration system that consolidates
//! all emotion processing configurations into a single, manageable structure.
//! This replaces the previous scattered configuration approach and provides
//! a centralized way to configure all aspects of emotion processing.

use crate::types::{Emotion, EmotionIntensity, EmotionParameters};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Unified configuration for all emotion processing components
///
/// This comprehensive configuration structure consolidates all emotion processing
/// settings into a single, manageable configuration. It includes all previously
/// scattered configuration options and provides a unified interface for configuring
/// the entire emotion processing system.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionConfig {
    /// Enable emotion processing
    pub enabled: bool,
    /// Default emotion intensity
    pub default_intensity: EmotionIntensity,
    /// Maximum number of simultaneous emotions
    pub max_emotions: usize,
    /// Transition smoothing factor (0.0 to 1.0)
    pub transition_smoothing: f32,
    /// Prosody modification strength (0.0 to 1.0)
    pub prosody_strength: f32,
    /// Voice quality modification strength (0.0 to 1.0)
    pub voice_quality_strength: f32,
    /// Custom emotion mappings
    pub custom_emotions: HashMap<String, EmotionParameters>,
    /// Validation settings
    pub validation: ValidationConfig,
    /// Performance settings
    pub performance: PerformanceConfig,
    /// Learning system configuration (optional)
    pub learning: Option<LearningConfig>,
    /// History tracking configuration (optional)
    pub history: Option<HistoryConfig>,
    /// Natural variation configuration (optional)
    pub variation: Option<VariationConfig>,
    /// A/B testing configuration (optional)
    pub ab_testing: Option<ABTestingConfig>,
    /// Recognition configuration (optional)
    pub recognition: Option<RecognitionConfig>,
    /// Conversation context configuration (optional)
    pub conversation: Option<ConversationConfig>,
    /// Interpolation configuration (optional)
    pub interpolation: Option<InterpolationConfig>,
    /// Consistency management configuration (optional)
    pub consistency: Option<ConsistencyConfig>,
    /// Perceptual validation configuration (optional)
    pub perceptual_validation: Option<PerceptualValidationConfig>,
    /// Real-time processing configuration (optional)
    pub realtime: Option<RealtimeConfig>,
    /// Plugin system configuration (optional)
    pub plugins: Option<PluginSystemConfig>,
}

impl EmotionConfig {
    /// Create a new emotion config with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for this config
    pub fn builder() -> EmotionConfigBuilder {
        EmotionConfigBuilder::new()
    }

    /// Validate the configuration
    pub fn validate(&self) -> crate::Result<()> {
        if self.max_emotions == 0 {
            return Err(crate::Error::Validation(
                "max_emotions must be greater than 0".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&self.transition_smoothing) {
            return Err(crate::Error::Validation(
                "transition_smoothing must be between 0.0 and 1.0".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&self.prosody_strength) {
            return Err(crate::Error::Validation(
                "prosody_strength must be between 0.0 and 1.0".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&self.voice_quality_strength) {
            return Err(crate::Error::Validation(
                "voice_quality_strength must be between 0.0 and 1.0".to_string(),
            ));
        }

        self.validation.validate()?;
        self.performance.validate()?;

        Ok(())
    }

    /// Get custom emotion by name
    pub fn get_custom_emotion(&self, name: &str) -> Option<&EmotionParameters> {
        self.custom_emotions.get(name)
    }

    /// Add custom emotion
    pub fn add_custom_emotion(&mut self, name: String, params: EmotionParameters) {
        self.custom_emotions.insert(name, params);
    }
}

impl Default for EmotionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_intensity: EmotionIntensity::MEDIUM,
            max_emotions: 3,
            transition_smoothing: 0.7,
            prosody_strength: 0.8,
            voice_quality_strength: 0.5,
            custom_emotions: HashMap::new(),
            validation: ValidationConfig::default(),
            performance: PerformanceConfig::default(),
            learning: None,
            history: None,
            variation: None,
            ab_testing: None,
            recognition: None,
            conversation: None,
            interpolation: None,
            consistency: None,
            perceptual_validation: None,
            realtime: None,
            plugins: None,
        }
    }
}

/// Validation configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Validate emotion intensity ranges
    pub validate_intensity: bool,
    /// Validate prosody parameter ranges
    pub validate_prosody: bool,
    /// Maximum allowed pitch shift
    pub max_pitch_shift: f32,
    /// Maximum allowed tempo scale
    pub max_tempo_scale: f32,
    /// Maximum allowed energy scale
    pub max_energy_scale: f32,
}

impl ValidationConfig {
    /// Validate the validation config
    pub fn validate(&self) -> crate::Result<()> {
        if self.max_pitch_shift <= 0.0 {
            return Err(crate::Error::Validation(
                "max_pitch_shift must be positive".to_string(),
            ));
        }

        if self.max_tempo_scale <= 0.0 {
            return Err(crate::Error::Validation(
                "max_tempo_scale must be positive".to_string(),
            ));
        }

        if self.max_energy_scale <= 0.0 {
            return Err(crate::Error::Validation(
                "max_energy_scale must be positive".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            validate_intensity: true,
            validate_prosody: true,
            max_pitch_shift: 2.0,
            max_tempo_scale: 2.0,
            max_energy_scale: 3.0,
        }
    }
}

/// Performance configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Cache size for emotion states
    pub cache_size: usize,
    /// Use SIMD optimizations
    pub use_simd: bool,
    /// Use GPU acceleration (requires gpu feature)
    pub use_gpu: bool,
    /// Number of processing threads
    pub num_threads: Option<usize>,
    /// Buffer size for real-time processing
    pub buffer_size: usize,
}

impl PerformanceConfig {
    /// Validate the performance config
    pub fn validate(&self) -> crate::Result<()> {
        if self.cache_size == 0 {
            return Err(crate::Error::Validation(
                "cache_size must be greater than 0".to_string(),
            ));
        }

        if self.buffer_size == 0 {
            return Err(crate::Error::Validation(
                "buffer_size must be greater than 0".to_string(),
            ));
        }

        if let Some(threads) = self.num_threads {
            if threads == 0 {
                return Err(crate::Error::Validation(
                    "num_threads must be greater than 0 if specified".to_string(),
                ));
            }
        }

        Ok(())
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            cache_size: 1000,
            use_simd: true,
            use_gpu: cfg!(feature = "gpu"), // Enable GPU if feature is available
            num_threads: None,              // Use default
            buffer_size: 1024,
        }
    }
}

/// Builder for EmotionConfig
#[derive(Debug, Clone)]
pub struct EmotionConfigBuilder {
    config: EmotionConfig,
}

impl EmotionConfigBuilder {
    /// Create a new config builder
    pub fn new() -> Self {
        Self {
            config: EmotionConfig::default(),
        }
    }

    /// Enable or disable emotion processing
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Set default emotion intensity
    pub fn default_intensity(mut self, intensity: EmotionIntensity) -> Self {
        self.config.default_intensity = intensity;
        self
    }

    /// Set maximum number of simultaneous emotions
    pub fn max_emotions(mut self, max: usize) -> Self {
        self.config.max_emotions = max;
        self
    }

    /// Set transition smoothing factor
    pub fn transition_smoothing(mut self, smoothing: f32) -> Self {
        self.config.transition_smoothing = smoothing.clamp(0.0, 1.0);
        self
    }

    /// Set prosody modification strength
    pub fn prosody_strength(mut self, strength: f32) -> Self {
        self.config.prosody_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set voice quality modification strength
    pub fn voice_quality_strength(mut self, strength: f32) -> Self {
        self.config.voice_quality_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Add custom emotion
    pub fn custom_emotion(mut self, name: String, params: EmotionParameters) -> Self {
        self.config.custom_emotions.insert(name, params);
        self
    }

    /// Configure validation settings
    pub fn validation(mut self, validation: ValidationConfig) -> Self {
        self.config.validation = validation;
        self
    }

    /// Configure performance settings
    pub fn performance(mut self, performance: PerformanceConfig) -> Self {
        self.config.performance = performance;
        self
    }

    /// Enable or disable SIMD optimizations
    pub fn use_simd(mut self, use_simd: bool) -> Self {
        self.config.performance.use_simd = use_simd;
        self
    }

    /// Enable or disable GPU acceleration (requires gpu feature)
    pub fn use_gpu(mut self, use_gpu: bool) -> Self {
        self.config.performance.use_gpu = use_gpu;
        self
    }

    /// Set cache size for emotion states
    pub fn cache_size(mut self, size: usize) -> Self {
        self.config.performance.cache_size = size;
        self
    }

    /// Set buffer size for real-time processing
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.config.performance.buffer_size = size;
        self
    }

    /// Build and validate the configuration
    pub fn build(self) -> crate::Result<EmotionConfig> {
        self.config.validate()?;
        Ok(self.config)
    }

    /// Build without validation (use with caution)
    pub fn build_unchecked(self) -> EmotionConfig {
        self.config
    }

    // === Unified Configuration Methods ===

    /// Enable learning system with configuration
    pub fn learning(mut self, config: LearningConfig) -> Self {
        self.config.learning = Some(config);
        self
    }

    /// Enable history tracking with configuration
    pub fn history(mut self, config: HistoryConfig) -> Self {
        self.config.history = Some(config);
        self
    }

    /// Enable natural variation with configuration
    pub fn variation(mut self, config: VariationConfig) -> Self {
        self.config.variation = Some(config);
        self
    }

    /// Enable A/B testing with configuration
    pub fn ab_testing(mut self, config: ABTestingConfig) -> Self {
        self.config.ab_testing = Some(config);
        self
    }

    /// Enable emotion recognition with configuration
    pub fn recognition(mut self, config: RecognitionConfig) -> Self {
        self.config.recognition = Some(config);
        self
    }

    /// Enable conversation context with configuration
    pub fn conversation(mut self, config: ConversationConfig) -> Self {
        self.config.conversation = Some(config);
        self
    }

    /// Enable custom interpolation with configuration
    pub fn interpolation(mut self, config: InterpolationConfig) -> Self {
        self.config.interpolation = Some(config);
        self
    }

    /// Enable consistency management with configuration
    pub fn consistency(mut self, config: ConsistencyConfig) -> Self {
        self.config.consistency = Some(config);
        self
    }

    /// Enable perceptual validation with configuration
    pub fn perceptual_validation(mut self, config: PerceptualValidationConfig) -> Self {
        self.config.perceptual_validation = Some(config);
        self
    }

    /// Enable real-time processing with configuration
    pub fn realtime(mut self, config: RealtimeConfig) -> Self {
        self.config.realtime = Some(config);
        self
    }

    /// Enable plugin system with configuration
    pub fn plugins(mut self, config: PluginSystemConfig) -> Self {
        self.config.plugins = Some(config);
        self
    }

    // === Convenience Methods for Common Configurations ===

    /// Enable emotion learning with default settings
    pub fn enable_learning(mut self) -> Self {
        self.config.learning = Some(LearningConfig::default());
        self
    }

    /// Enable history tracking with default settings
    pub fn enable_history(mut self) -> Self {
        self.config.history = Some(HistoryConfig::default());
        self
    }

    /// Enable natural variation with default settings
    pub fn enable_variation(mut self) -> Self {
        self.config.variation = Some(VariationConfig::default());
        self
    }

    /// Enable emotion recognition with default settings
    pub fn enable_recognition(mut self) -> Self {
        self.config.recognition = Some(RecognitionConfig::default());
        self
    }

    /// Enable conversation context with default settings
    pub fn enable_conversation(mut self) -> Self {
        self.config.conversation = Some(ConversationConfig::default());
        self
    }

    /// Enable plugin system with default settings
    pub fn enable_plugins(mut self) -> Self {
        self.config.plugins = Some(PluginSystemConfig::default());
        self
    }

    /// Create a comprehensive configuration with all features enabled
    pub fn comprehensive() -> Self {
        Self::new()
            .enable_learning()
            .enable_history()
            .enable_variation()
            .enable_recognition()
            .enable_conversation()
            .enable_plugins()
            .interpolation(InterpolationConfig::default())
            .consistency(ConsistencyConfig::default())
            .realtime(RealtimeConfig::default())
    }

    /// Create a minimal configuration with only core features
    pub fn minimal() -> Self {
        Self::new()
            .enabled(true)
            .max_emotions(1)
            .prosody_strength(0.5)
            .voice_quality_strength(0.3)
    }

    /// Create a performance-optimized configuration
    pub fn performance_optimized() -> Self {
        Self::new()
            .use_simd(true)
            .use_gpu(true)
            .cache_size(2000)
            .buffer_size(2048)
            .enabled(true)
    }
}

impl Default for EmotionConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for plugin system
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PluginSystemConfig {
    /// Whether the plugin system is enabled
    pub enabled: bool,
    /// Maximum number of loaded plugins
    pub max_plugins: usize,
    /// Plugin execution timeout in milliseconds
    pub execution_timeout_ms: u64,
    /// Enable plugin hot-reloading (requires dynamic loading feature)
    pub hot_reload: bool,
    /// Plugin directories to scan for dynamic loading
    pub plugin_directories: Vec<String>,
    /// Plugin allowlist (empty = allow all)
    pub allowed_plugins: Vec<String>,
    /// Plugin blocklist
    pub blocked_plugins: Vec<String>,
    /// Enable plugin sandboxing (security feature)
    pub sandboxing: bool,
    /// Maximum memory usage per plugin in MB
    pub memory_limit_mb: usize,
    /// Enable plugin logging
    pub logging: bool,
    /// Plugin API version compatibility
    pub api_version: String,
}

impl Default for PluginSystemConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_plugins: 50,
            execution_timeout_ms: 1000,
            hot_reload: false,
            plugin_directories: vec!["./plugins".to_string()],
            allowed_plugins: Vec::new(),
            blocked_plugins: Vec::new(),
            sandboxing: true,
            memory_limit_mb: 100,
            logging: true,
            api_version: "1.0.0".to_string(),
        }
    }
}

/// Convenience type aliases for backward compatibility with existing code
pub type EmotionLearningConfig = LearningConfig;
pub type EmotionHistoryConfig = HistoryConfig;
pub type NaturalVariationConfig = VariationConfig;
pub type ABTestConfig = ABTestingConfig;
pub type EmotionRecognitionConfig = RecognitionConfig;
pub type EmotionConsistencyConfig = ConsistencyConfig;
pub type RealtimeEmotionConfig = RealtimeConfig;

// === Unified Sub-Configuration Structures ===

/// Configuration for emotion learning system
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Maximum number of feedback samples to store
    pub max_feedback_samples: usize,
    /// Learning rate for neural network training
    pub learning_rate: f32,
    /// Batch size for training
    pub batch_size: usize,
    /// Number of training epochs per update
    pub training_epochs: u32,
    /// Minimum feedback samples before starting learning
    pub min_samples_for_learning: usize,
    /// Weight decay for regularization
    pub weight_decay: f32,
    /// Enable GPU acceleration if available
    pub use_gpu: bool,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            max_feedback_samples: 10000,
            learning_rate: 0.001,
            batch_size: 32,
            training_epochs: 10,
            min_samples_for_learning: 50,
            weight_decay: 0.001,
            use_gpu: cfg!(feature = "gpu"),
        }
    }
}

/// Configuration for emotion history tracking
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoryConfig {
    /// Maximum number of entries to keep in memory
    pub max_entries: usize,
    /// Maximum age of entries before automatic cleanup
    pub max_age: Duration,
    /// Whether to automatically track duration of emotion states
    pub track_duration: bool,
    /// Minimum time between history entries (to avoid spam)
    pub min_interval: Duration,
    /// Whether to compress old entries
    pub enable_compression: bool,
    /// Sample rate for compressed entries (keep 1 in N)
    pub compression_rate: usize,
}

impl Default for HistoryConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            max_age: Duration::from_secs(24 * 60 * 60), // 24 hours
            track_duration: true,
            min_interval: Duration::from_millis(100),
            enable_compression: true,
            compression_rate: 10,
        }
    }
}

/// Configuration for natural variation generation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariationConfig {
    /// Base variation intensity (0.0 to 1.0)
    pub base_variation_intensity: f32,
    /// Temporal variation frequency (Hz)
    pub temporal_frequency: f32,
    /// Enable prosodic micro-variations
    pub enable_prosodic_variation: bool,
    /// Enable voice quality variations
    pub enable_voice_quality_variation: bool,
    /// Enable breathing pattern simulation
    pub enable_breathing_patterns: bool,
    /// Emotion-specific variation scaling
    pub emotion_scaling: HashMap<String, f32>,
    /// Speaker characteristics influence
    pub speaker_characteristics_influence: f32,
    /// Random seed for reproducible variations
    pub random_seed: Option<u64>,
    /// Variation smoothing factor
    pub smoothing_factor: f32,
}

impl Default for VariationConfig {
    fn default() -> Self {
        let mut emotion_scaling = HashMap::new();
        emotion_scaling.insert("Happy".to_string(), 1.2);
        emotion_scaling.insert("Sad".to_string(), 0.8);
        emotion_scaling.insert("Angry".to_string(), 1.5);
        emotion_scaling.insert("Calm".to_string(), 0.6);

        Self {
            base_variation_intensity: 0.3,
            temporal_frequency: 2.0,
            enable_prosodic_variation: true,
            enable_voice_quality_variation: true,
            enable_breathing_patterns: true,
            emotion_scaling,
            speaker_characteristics_influence: 0.5,
            random_seed: None,
            smoothing_factor: 0.7,
        }
    }
}

/// Configuration for A/B testing studies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ABTestingConfig {
    /// Test name/description
    pub test_name: String,
    /// Minimum number of comparisons required
    pub min_comparisons: usize,
    /// Target confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Statistical power (e.g., 0.8 for 80%)
    pub power: f64,
    /// Effect size to detect
    pub effect_size: f64,
    /// Test emotions to compare
    pub test_emotions: Vec<Emotion>,
    /// Randomization seed for reproducibility
    pub randomization_seed: Option<u64>,
    /// Maximum test duration
    pub max_duration: Duration,
    /// Balance allocation between variants
    pub balanced_allocation: bool,
}

impl Default for ABTestingConfig {
    fn default() -> Self {
        Self {
            test_name: "Emotion A/B Test".to_string(),
            min_comparisons: 30,
            confidence_level: 0.95,
            power: 0.8,
            effect_size: 0.5,
            test_emotions: vec![Emotion::Happy, Emotion::Sad],
            randomization_seed: None,
            max_duration: Duration::from_secs(3600), // 1 hour
            balanced_allocation: true,
        }
    }
}

/// Configuration for emotion recognition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecognitionConfig {
    /// Minimum confidence threshold for emotion detection
    pub confidence_threshold: f32,
    /// Whether to use context-aware analysis
    pub context_aware: bool,
    /// Maximum text length to analyze (for performance)
    pub max_text_length: usize,
    /// Weight for sentiment analysis in final decision
    pub sentiment_weight: f32,
    /// Weight for lexical analysis in final decision
    pub lexical_weight: f32,
    /// Weight for context analysis in final decision
    pub context_weight: f32,
}

impl Default for RecognitionConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.3,
            context_aware: true,
            max_text_length: 10000,
            sentiment_weight: 0.4,
            lexical_weight: 0.4,
            context_weight: 0.2,
        }
    }
}

/// Configuration for conversation context tracking
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConversationConfig {
    /// Maximum number of conversation turns to track
    pub max_history_size: usize,
    /// Maximum age of conversation context in seconds
    pub max_context_age_secs: u64,
    /// Weight for previous emotion influence (0.0 to 1.0)
    pub emotion_momentum_weight: f32,
    /// Weight for speaker relationship influence
    pub relationship_weight: f32,
    /// Weight for topic context influence
    pub topic_weight: f32,
    /// Enable automatic emotion adaptation
    pub auto_adaptation: bool,
    /// Minimum conversation turns before adaptation kicks in
    pub min_turns_for_adaptation: usize,
}

impl Default for ConversationConfig {
    fn default() -> Self {
        Self {
            max_history_size: 50,
            max_context_age_secs: 3600, // 1 hour
            emotion_momentum_weight: 0.3,
            relationship_weight: 0.3,
            topic_weight: 0.2,
            auto_adaptation: true,
            min_turns_for_adaptation: 3,
        }
    }
}

/// Configuration for emotion interpolation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterpolationConfig {
    /// Default interpolation method
    pub method: InterpolationMethod,
    /// Duration of transitions in milliseconds
    pub transition_duration_ms: u64,
    /// Minimum change threshold for starting a transition
    pub change_threshold: f32,
    /// Maximum number of simultaneous transitions
    pub max_concurrent_transitions: usize,
    /// Use dimension-based interpolation
    pub use_dimension_interpolation: bool,
}

/// Interpolation methods for emotion transitions
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum InterpolationMethod {
    /// Linear interpolation between emotions
    Linear,
    /// Cubic spline interpolation for smooth transitions
    CubicSpline,
    /// Exponential interpolation for natural feel
    Exponential,
    /// Sine wave interpolation for periodic emotions
    Sine,
}

impl Default for InterpolationConfig {
    fn default() -> Self {
        Self {
            method: InterpolationMethod::Linear,
            transition_duration_ms: 500,
            change_threshold: 0.1,
            max_concurrent_transitions: 3,
            use_dimension_interpolation: true,
        }
    }
}

/// Configuration for emotion consistency management
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsistencyConfig {
    /// Maximum allowed change per time unit
    pub max_change_rate: f32,
    /// Emotion momentum factor (0.0 to 1.0)
    pub momentum_factor: f32,
    /// Enable narrative context awareness
    pub enable_narrative_context: bool,
    /// Tags that influence consistency checking
    pub consistency_tags: Vec<String>,
    /// Smoothing window size for consistency checks
    pub smoothing_window_size: usize,
}

impl Default for ConsistencyConfig {
    fn default() -> Self {
        Self {
            max_change_rate: 0.5,
            momentum_factor: 0.3,
            enable_narrative_context: true,
            consistency_tags: vec!["dialogue".to_string(), "narrative".to_string()],
            smoothing_window_size: 5,
        }
    }
}

/// Configuration for perceptual validation studies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceptualValidationConfig {
    /// Number of evaluators per sample
    pub evaluators_per_sample: usize,
    /// Minimum inter-evaluator agreement threshold
    pub min_agreement_threshold: f64,
    /// Session duration limit in minutes
    pub session_duration_limit_mins: u32,
    /// Sample randomization for unbiased evaluation
    pub randomize_samples: bool,
    /// Include demographic information collection
    pub collect_demographics: bool,
    /// Maximum samples per evaluation session
    pub max_samples_per_session: usize,
    /// Enable detailed evaluation criteria
    pub detailed_criteria: bool,
}

impl Default for PerceptualValidationConfig {
    fn default() -> Self {
        Self {
            evaluators_per_sample: 5,
            min_agreement_threshold: 0.7,
            session_duration_limit_mins: 60,
            randomize_samples: true,
            collect_demographics: true,
            max_samples_per_session: 50,
            detailed_criteria: true,
        }
    }
}

/// Configuration for real-time emotion processing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RealtimeConfig {
    /// Buffer size for real-time processing
    pub buffer_size: usize,
    /// Maximum processing latency in milliseconds
    pub max_latency_ms: u64,
    /// Enable adaptive buffer sizing
    pub adaptive_buffering: bool,
    /// Thread pool size for parallel processing
    pub thread_pool_size: Option<usize>,
    /// Enable low-latency mode (reduced quality for speed)
    pub low_latency_mode: bool,
    /// Audio sample rate for processing
    pub sample_rate: u32,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1024,
            max_latency_ms: 10,
            adaptive_buffering: true,
            thread_pool_size: None,
            low_latency_mode: false,
            sample_rate: 22050,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_config_default() {
        let config = EmotionConfig::default();
        assert!(config.validate().is_ok());
        assert!(config.enabled);
        assert_eq!(config.max_emotions, 3);
    }

    #[test]
    fn test_emotion_config_builder() {
        let config = EmotionConfig::builder()
            .enabled(true)
            .max_emotions(5)
            .prosody_strength(0.9)
            .build()
            .unwrap();

        assert!(config.enabled);
        assert_eq!(config.max_emotions, 5);
        assert_eq!(config.prosody_strength, 0.9);
    }

    #[test]
    fn test_validation_errors() {
        let mut config = EmotionConfig::default();
        config.max_emotions = 0;
        assert!(config.validate().is_err());

        config.max_emotions = 1;
        config.transition_smoothing = 2.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_custom_emotions() {
        let mut config = EmotionConfig::default();
        let custom_emotion = EmotionParameters::neutral();
        config.add_custom_emotion("my_emotion".to_string(), custom_emotion.clone());

        assert_eq!(
            config.get_custom_emotion("my_emotion"),
            Some(&custom_emotion)
        );
        assert_eq!(config.get_custom_emotion("nonexistent"), None);
    }

    #[test]
    fn test_unified_configuration_builder() {
        let config = EmotionConfig::builder()
            .enabled(true)
            .enable_learning()
            .enable_history()
            .enable_variation()
            .enable_recognition()
            .enable_conversation()
            .build()
            .unwrap();

        assert!(config.enabled);
        assert!(config.learning.is_some());
        assert!(config.history.is_some());
        assert!(config.variation.is_some());
        assert!(config.recognition.is_some());
        assert!(config.conversation.is_some());
    }

    #[test]
    fn test_comprehensive_configuration() {
        let config = EmotionConfigBuilder::comprehensive().build().unwrap();

        assert!(config.learning.is_some());
        assert!(config.history.is_some());
        assert!(config.variation.is_some());
        assert!(config.recognition.is_some());
        assert!(config.conversation.is_some());
        assert!(config.interpolation.is_some());
        assert!(config.consistency.is_some());
        assert!(config.realtime.is_some());
        assert!(config.plugins.is_some());
    }

    #[test]
    fn test_minimal_configuration() {
        let config = EmotionConfigBuilder::minimal().build().unwrap();

        assert!(config.enabled);
        assert_eq!(config.max_emotions, 1);
        assert_eq!(config.prosody_strength, 0.5);
        assert_eq!(config.voice_quality_strength, 0.3);
        // Optional features should be None in minimal config
        assert!(config.learning.is_none());
        assert!(config.history.is_none());
    }

    #[test]
    fn test_performance_optimized_configuration() {
        let config = EmotionConfigBuilder::performance_optimized()
            .build()
            .unwrap();

        assert!(config.enabled);
        assert!(config.performance.use_simd);
        assert_eq!(config.performance.cache_size, 2000);
        assert_eq!(config.performance.buffer_size, 2048);
    }

    #[test]
    fn test_sub_configuration_defaults() {
        let learning_config = LearningConfig::default();
        assert_eq!(learning_config.max_feedback_samples, 10000);
        assert_eq!(learning_config.learning_rate, 0.001);

        let history_config = HistoryConfig::default();
        assert_eq!(history_config.max_entries, 1000);
        assert!(history_config.track_duration);

        let variation_config = VariationConfig::default();
        assert_eq!(variation_config.base_variation_intensity, 0.3);
        assert!(variation_config.enable_prosodic_variation);
    }

    #[test]
    fn test_serialization() {
        let config = EmotionConfigBuilder::comprehensive().build().unwrap();

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: EmotionConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.enabled, deserialized.enabled);
        assert_eq!(config.learning.is_some(), deserialized.learning.is_some());
    }
}
