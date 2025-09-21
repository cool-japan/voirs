//! VoiRS Ecosystem Integration
//!
//! This module provides seamless integration with the broader VoiRS ecosystem,
//! including shared configuration management, common error handling patterns,
//! and cross-crate data structure compatibility.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use voirs_recognizer::traits::{PhonemeAlignment, Transcript};
use voirs_sdk::{AudioBuffer, LanguageCode, VoirsError};

use crate::{EvaluationError, EvaluationResult};

/// Shared configuration manager for VoiRS ecosystem integration
#[derive(Debug, Clone)]
pub struct EcosystemConfig {
    /// Global configuration values
    pub global_config: Arc<RwLock<HashMap<String, ConfigValue>>>,
    /// Language-specific configurations
    pub language_configs: Arc<RwLock<HashMap<LanguageCode, LanguageConfig>>>,
    /// System-wide quality thresholds
    pub quality_thresholds: QualityThresholds,
    /// Integration settings
    pub integration_settings: IntegrationSettings,
}

/// Configuration value that can be shared across crates
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum ConfigValue {
    /// String configuration value
    String(String),
    /// Integer configuration value
    Integer(i64),
    /// Float configuration value
    Float(f64),
    /// Boolean configuration value
    Boolean(bool),
    /// Array of values
    Array(Vec<ConfigValue>),
    /// Object with key-value pairs
    Object(HashMap<String, ConfigValue>),
}

/// Language-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageConfig {
    /// Language code
    pub language: LanguageCode,
    /// Phoneme mapping for pronunciation evaluation
    pub phoneme_mapping: HashMap<String, String>,
    /// Language-specific quality weights
    pub quality_weights: HashMap<String, f32>,
    /// Pronunciation difficulty adjustments
    pub difficulty_adjustments: HashMap<String, f32>,
    /// Cultural adaptation settings
    pub cultural_adaptations: CulturalAdaptations,
}

/// Cultural adaptation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalAdaptations {
    /// Preferred feedback style
    pub feedback_style: FeedbackStyle,
    /// Tolerance for accent variations
    pub accent_tolerance: f32,
    /// Regional pronunciation variants
    pub regional_variants: HashMap<String, Vec<String>>,
    /// Cultural context preferences
    pub context_preferences: Vec<String>,
}

/// Feedback style preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackStyle {
    /// Direct and specific feedback
    Direct,
    /// Encouraging and supportive feedback
    Supportive,
    /// Detailed technical feedback
    Technical,
    /// Brief and concise feedback
    Concise,
    /// Culturally sensitive feedback
    CulturallySensitive,
}

/// System-wide quality thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum acceptable quality score
    pub minimum_quality: f32,
    /// Good quality threshold
    pub good_quality: f32,
    /// Excellent quality threshold
    pub excellent_quality: f32,
    /// Pronunciation accuracy thresholds
    pub pronunciation_thresholds: PronunciationThresholds,
    /// Confidence score thresholds
    pub confidence_thresholds: ConfidenceThresholds,
}

/// Pronunciation accuracy thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PronunciationThresholds {
    /// Minimum acceptable pronunciation score
    pub minimum_pronunciation: f32,
    /// Native-like pronunciation threshold
    pub native_like: f32,
    /// Phoneme accuracy threshold
    pub phoneme_accuracy: f32,
    /// Fluency threshold
    pub fluency: f32,
    /// Stress accuracy threshold
    pub stress_accuracy: f32,
}

/// Confidence score thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceThresholds {
    /// Low confidence threshold
    pub low_confidence: f32,
    /// Medium confidence threshold
    pub medium_confidence: f32,
    /// High confidence threshold
    pub high_confidence: f32,
    /// Minimum confidence for automatic decisions
    pub auto_decision_threshold: f32,
}

/// Integration settings for cross-crate compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationSettings {
    /// Enable automatic SDK integration
    pub auto_sdk_integration: bool,
    /// Enable shared caching across crates
    pub shared_caching: bool,
    /// Enable cross-crate error propagation
    pub cross_crate_errors: bool,
    /// Enable performance metrics sharing
    pub shared_metrics: bool,
    /// Data format version for compatibility
    pub data_format_version: String,
    /// Maximum processing timeout
    pub max_processing_timeout: std::time::Duration,
}

/// Ecosystem data bridge for cross-crate compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcosystemDataBridge {
    /// Audio metadata from various sources
    pub audio_metadata: HashMap<String, AudioMetadata>,
    /// Processing state shared across crates
    pub processing_state: ProcessingState,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Error context for debugging
    pub error_context: ErrorContext,
}

/// Audio metadata for ecosystem integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMetadata {
    /// Original source identifier
    pub source_id: String,
    /// Processing pipeline stage
    pub pipeline_stage: String,
    /// Quality metrics computed so far
    pub quality_metrics: HashMap<String, f32>,
    /// Processing timestamps
    pub timestamps: HashMap<String, std::time::SystemTime>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Processing state for pipeline coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingState {
    /// Current processing stage
    pub current_stage: String,
    /// Completed stages
    pub completed_stages: Vec<String>,
    /// Stage-specific results
    pub stage_results: HashMap<String, serde_json::Value>,
    /// Processing options
    pub options: HashMap<String, ConfigValue>,
}

/// Performance metrics for ecosystem optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Processing time per stage
    pub stage_times: HashMap<String, std::time::Duration>,
    /// Memory usage metrics
    pub memory_usage: HashMap<String, u64>,
    /// Throughput metrics
    pub throughput: HashMap<String, f64>,
    /// Error rates
    pub error_rates: HashMap<String, f32>,
    /// Cache hit rates
    pub cache_hit_rates: HashMap<String, f32>,
}

/// Error context for cross-crate debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    /// Original error source crate
    pub source_crate: String,
    /// Error propagation path
    pub propagation_path: Vec<String>,
    /// Context data at each stage
    pub context_data: HashMap<String, serde_json::Value>,
    /// Timestamp of original error
    pub timestamp: std::time::SystemTime,
    /// Recovery suggestions
    pub recovery_suggestions: Vec<String>,
}

/// Trait for ecosystem-aware evaluators
#[async_trait]
pub trait EcosystemEvaluator {
    /// Initialize with ecosystem configuration
    async fn initialize_with_ecosystem(&mut self, config: &EcosystemConfig)
        -> EvaluationResult<()>;

    /// Process with ecosystem data bridge
    async fn process_with_ecosystem(
        &self,
        audio: &AudioBuffer,
        bridge: &mut EcosystemDataBridge,
    ) -> EvaluationResult<serde_json::Value>;

    /// Get ecosystem-compatible results
    async fn get_ecosystem_results(&self) -> EvaluationResult<EcosystemResults>;

    /// Handle cross-crate errors
    fn handle_ecosystem_error(&self, error: VoirsError) -> EvaluationError;
}

/// Results format compatible with ecosystem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcosystemResults {
    /// Evaluation results in standard format
    pub evaluation_results: HashMap<String, serde_json::Value>,
    /// Quality scores compatible with other crates
    pub quality_scores: HashMap<String, f32>,
    /// Metadata for downstream processing
    pub metadata: HashMap<String, serde_json::Value>,
    /// Processing statistics
    pub processing_stats: PerformanceMetrics,
    /// Recommendations for other components
    pub recommendations: Vec<EcosystemRecommendation>,
}

/// Recommendation for ecosystem components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcosystemRecommendation {
    /// Target component or crate
    pub target_component: String,
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Recommendation description
    pub description: String,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Parameters for the recommendation
    pub parameters: HashMap<String, ConfigValue>,
}

/// Type of ecosystem recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Quality improvement suggestion
    QualityImprovement,
    /// Performance optimization
    PerformanceOptimization,
    /// Configuration adjustment
    ConfigurationAdjustment,
    /// Model parameter tuning
    ModelParameterTuning,
    /// Data preprocessing enhancement
    DataPreprocessingEnhancement,
    /// Error handling improvement
    ErrorHandlingImprovement,
}

/// Priority level for recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    /// Critical - must be addressed immediately
    Critical,
    /// High - should be addressed soon
    High,
    /// Medium - moderate priority
    Medium,
    /// Low - nice to have
    Low,
    /// Informational - for reference only
    Informational,
}

/// Implementation of ecosystem configuration
impl EcosystemConfig {
    /// Create default ecosystem configuration
    pub fn default() -> Self {
        Self {
            global_config: Arc::new(RwLock::new(HashMap::new())),
            language_configs: Arc::new(RwLock::new(HashMap::new())),
            quality_thresholds: QualityThresholds::default(),
            integration_settings: IntegrationSettings::default(),
        }
    }

    /// Create ecosystem configuration with custom settings
    pub fn new(
        quality_thresholds: QualityThresholds,
        integration_settings: IntegrationSettings,
    ) -> Self {
        Self {
            global_config: Arc::new(RwLock::new(HashMap::new())),
            language_configs: Arc::new(RwLock::new(HashMap::new())),
            quality_thresholds,
            integration_settings,
        }
    }

    /// Set global configuration value
    pub async fn set_global_config(&self, key: String, value: ConfigValue) {
        let mut config = self.global_config.write().await;
        config.insert(key, value);
    }

    /// Get global configuration value
    pub async fn get_global_config(&self, key: &str) -> Option<ConfigValue> {
        let config = self.global_config.read().await;
        config.get(key).cloned()
    }

    /// Set language-specific configuration
    pub async fn set_language_config(&self, language: LanguageCode, config: LanguageConfig) {
        let mut configs = self.language_configs.write().await;
        configs.insert(language, config);
    }

    /// Get language-specific configuration
    pub async fn get_language_config(&self, language: &LanguageCode) -> Option<LanguageConfig> {
        let configs = self.language_configs.read().await;
        configs.get(language).cloned()
    }

    /// Validate configuration compatibility
    pub fn validate_compatibility(&self) -> Result<(), EvaluationError> {
        // Check data format version compatibility
        let version = &self.integration_settings.data_format_version;
        if version.is_empty() {
            return Err(EvaluationError::ConfigurationError {
                message: "Data format version not specified".to_string(),
            });
        }

        // Validate quality thresholds
        let thresholds = &self.quality_thresholds;
        if thresholds.minimum_quality < 0.0 || thresholds.minimum_quality > 1.0 {
            return Err(EvaluationError::ConfigurationError {
                message: "Invalid minimum quality threshold".to_string(),
            });
        }

        if thresholds.good_quality <= thresholds.minimum_quality {
            return Err(EvaluationError::ConfigurationError {
                message: "Good quality threshold must be higher than minimum".to_string(),
            });
        }

        if thresholds.excellent_quality <= thresholds.good_quality {
            return Err(EvaluationError::ConfigurationError {
                message: "Excellent quality threshold must be higher than good".to_string(),
            });
        }

        Ok(())
    }
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            minimum_quality: 0.6,
            good_quality: 0.8,
            excellent_quality: 0.9,
            pronunciation_thresholds: PronunciationThresholds::default(),
            confidence_thresholds: ConfidenceThresholds::default(),
        }
    }
}

impl Default for PronunciationThresholds {
    fn default() -> Self {
        Self {
            minimum_pronunciation: 0.5,
            native_like: 0.95,
            phoneme_accuracy: 0.8,
            fluency: 0.7,
            stress_accuracy: 0.75,
        }
    }
}

impl Default for ConfidenceThresholds {
    fn default() -> Self {
        Self {
            low_confidence: 0.3,
            medium_confidence: 0.6,
            high_confidence: 0.8,
            auto_decision_threshold: 0.75,
        }
    }
}

impl Default for IntegrationSettings {
    fn default() -> Self {
        Self {
            auto_sdk_integration: true,
            shared_caching: true,
            cross_crate_errors: true,
            shared_metrics: true,
            data_format_version: "1.0.0".to_string(),
            max_processing_timeout: std::time::Duration::from_secs(300), // 5 minutes
        }
    }
}

impl Default for EcosystemDataBridge {
    fn default() -> Self {
        Self {
            audio_metadata: HashMap::new(),
            processing_state: ProcessingState::default(),
            performance_metrics: PerformanceMetrics::default(),
            error_context: ErrorContext::default(),
        }
    }
}

impl Default for ProcessingState {
    fn default() -> Self {
        Self {
            current_stage: "initialized".to_string(),
            completed_stages: Vec::new(),
            stage_results: HashMap::new(),
            options: HashMap::new(),
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            stage_times: HashMap::new(),
            memory_usage: HashMap::new(),
            throughput: HashMap::new(),
            error_rates: HashMap::new(),
            cache_hit_rates: HashMap::new(),
        }
    }
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self {
            source_crate: "voirs-evaluation".to_string(),
            propagation_path: Vec::new(),
            context_data: HashMap::new(),
            timestamp: std::time::SystemTime::now(),
            recovery_suggestions: Vec::new(),
        }
    }
}

/// Utility functions for ecosystem integration
pub mod utils {
    use super::*;

    /// Convert VoiRS error to ecosystem error with context
    pub fn convert_error_with_context(
        error: VoirsError,
        context: &ErrorContext,
    ) -> EvaluationError {
        match error {
            VoirsError::ModelError {
                model_type,
                message,
                source,
            } => EvaluationError::ModelError {
                message: format!(
                    "[{}] Model error ({}): {}",
                    context.source_crate, model_type, message
                ),
                source,
            },
            VoirsError::AudioError {
                message,
                buffer_info,
            } => EvaluationError::AudioProcessingError {
                message: format!("[{}] Audio error: {}", context.source_crate, message),
                source: buffer_info.map(|info| {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Buffer info: {:?}", info),
                    )) as Box<dyn std::error::Error + Send + Sync>
                }),
            },
            _ => EvaluationError::ModelError {
                message: format!("[{}] Ecosystem error: {}", context.source_crate, error),
                source: None,
            },
        }
    }

    /// Create recommendation for ecosystem component
    pub fn create_recommendation(
        target: &str,
        rec_type: RecommendationType,
        description: &str,
        priority: RecommendationPriority,
    ) -> EcosystemRecommendation {
        EcosystemRecommendation {
            target_component: target.to_string(),
            recommendation_type: rec_type,
            description: description.to_string(),
            priority,
            parameters: HashMap::new(),
        }
    }

    /// Merge performance metrics from multiple sources
    pub fn merge_performance_metrics(metrics: Vec<PerformanceMetrics>) -> PerformanceMetrics {
        let mut merged = PerformanceMetrics::default();

        for metric in &metrics {
            // Merge stage times by taking the sum
            for (stage, time) in &metric.stage_times {
                *merged
                    .stage_times
                    .entry(stage.clone())
                    .or_insert(std::time::Duration::ZERO) += *time;
            }

            // Merge memory usage by taking the maximum
            for (stage, memory) in &metric.memory_usage {
                let current = merged.memory_usage.entry(stage.clone()).or_insert(0);
                *current = (*current).max(*memory);
            }

            // Merge throughput by taking the average
            for (stage, throughput) in &metric.throughput {
                *merged.throughput.entry(stage.clone()).or_insert(0.0) += throughput;
            }

            // Merge error rates by taking the average
            for (stage, rate) in &metric.error_rates {
                *merged.error_rates.entry(stage.clone()).or_insert(0.0) += rate;
            }

            // Merge cache hit rates by taking the average
            for (stage, rate) in &metric.cache_hit_rates {
                *merged.cache_hit_rates.entry(stage.clone()).or_insert(0.0) += rate;
            }
        }

        // Average the throughput, error rates, and cache hit rates
        let count = metrics.len() as f64;
        if count > 1.0 {
            for throughput in merged.throughput.values_mut() {
                *throughput /= count;
            }
            for error_rate in merged.error_rates.values_mut() {
                *error_rate /= count as f32;
            }
            for hit_rate in merged.cache_hit_rates.values_mut() {
                *hit_rate /= count as f32;
            }
        }

        merged
    }

    /// Check if configuration is compatible with ecosystem version
    pub fn check_version_compatibility(
        current_version: &str,
        required_version: &str,
    ) -> Result<(), EvaluationError> {
        // Simple version compatibility check - in real implementation,
        // this would use proper semantic versioning
        if current_version.split('.').next() != required_version.split('.').next() {
            return Err(EvaluationError::ConfigurationError {
                message: format!(
                    "Version incompatibility: current {} vs required {}",
                    current_version, required_version
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ecosystem_config_creation() {
        let config = EcosystemConfig::default();
        assert!(config.validate_compatibility().is_ok());
    }

    #[tokio::test]
    async fn test_global_config_management() {
        let config = EcosystemConfig::default();

        // Set a configuration value
        config
            .set_global_config(
                "test_key".to_string(),
                ConfigValue::String("test_value".to_string()),
            )
            .await;

        // Retrieve the configuration value
        let value = config.get_global_config("test_key").await;
        assert!(value.is_some());

        if let Some(ConfigValue::String(s)) = value {
            assert_eq!(s, "test_value");
        } else {
            panic!("Expected string configuration value");
        }
    }

    #[tokio::test]
    async fn test_language_config_management() {
        let config = EcosystemConfig::default();
        let lang_config = LanguageConfig {
            language: LanguageCode::EnUs,
            phoneme_mapping: HashMap::new(),
            quality_weights: HashMap::new(),
            difficulty_adjustments: HashMap::new(),
            cultural_adaptations: CulturalAdaptations {
                feedback_style: FeedbackStyle::Direct,
                accent_tolerance: 0.8,
                regional_variants: HashMap::new(),
                context_preferences: vec!["formal".to_string()],
            },
        };

        config
            .set_language_config(LanguageCode::EnUs, lang_config.clone())
            .await;
        let retrieved = config.get_language_config(&LanguageCode::EnUs).await;

        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().language, LanguageCode::EnUs);
    }

    #[test]
    fn test_quality_thresholds_validation() {
        let mut thresholds = QualityThresholds::default();
        let settings = IntegrationSettings::default();
        let config = EcosystemConfig::new(thresholds.clone(), settings);
        assert!(config.validate_compatibility().is_ok());

        // Test invalid thresholds
        thresholds.minimum_quality = 1.5; // Invalid: > 1.0
        let config = EcosystemConfig::new(thresholds, IntegrationSettings::default());
        assert!(config.validate_compatibility().is_err());
    }

    #[test]
    fn test_performance_metrics_merging() {
        let mut metrics1 = PerformanceMetrics::default();
        metrics1
            .stage_times
            .insert("stage1".to_string(), std::time::Duration::from_millis(100));
        metrics1.throughput.insert("stage1".to_string(), 10.0);

        let mut metrics2 = PerformanceMetrics::default();
        metrics2
            .stage_times
            .insert("stage1".to_string(), std::time::Duration::from_millis(150));
        metrics2.throughput.insert("stage1".to_string(), 20.0);

        let merged = utils::merge_performance_metrics(vec![metrics1, metrics2]);

        assert_eq!(
            merged.stage_times.get("stage1"),
            Some(&std::time::Duration::from_millis(250))
        );
        assert_eq!(merged.throughput.get("stage1"), Some(&15.0)); // Average
    }

    #[test]
    fn test_version_compatibility() {
        assert!(utils::check_version_compatibility("1.0.0", "1.1.0").is_ok());
        assert!(utils::check_version_compatibility("1.0.0", "2.0.0").is_err());
        assert!(utils::check_version_compatibility("2.1.5", "2.3.1").is_ok());
    }

    #[test]
    fn test_ecosystem_recommendation_creation() {
        let rec = utils::create_recommendation(
            "voirs-acoustic",
            RecommendationType::QualityImprovement,
            "Increase sample rate for better quality",
            RecommendationPriority::High,
        );

        assert_eq!(rec.target_component, "voirs-acoustic");
        assert_eq!(rec.description, "Increase sample rate for better quality");
        assert!(matches!(rec.priority, RecommendationPriority::High));
    }

    #[test]
    fn test_error_context_conversion() {
        let context = ErrorContext {
            source_crate: "voirs-test".to_string(),
            ..Default::default()
        };

        let voirs_error = VoirsError::AudioError {
            message: "Test audio error".to_string(),
            buffer_info: None,
        };

        let eval_error = utils::convert_error_with_context(voirs_error, &context);
        assert!(matches!(
            eval_error,
            EvaluationError::AudioProcessingError { .. }
        ));
    }
}
