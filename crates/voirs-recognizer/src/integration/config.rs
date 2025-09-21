//! # Unified Configuration System
//!
//! This module provides a unified configuration system that integrates
//! recognition-specific configurations with the VoiRS SDK's hierarchical
//! configuration management system.

use crate::traits::{ASRConfig, AudioAnalysisConfig, PhonemeRecognitionConfig};
use serde::{Deserialize, Serialize};
use voirs_sdk::LanguageCode;

/// Unified configuration for the entire VoiRS ecosystem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedVoirsConfig {
    /// Recognition configuration
    pub recognition: RecognitionConfig,
    /// Synthesis configuration (from SDK)
    pub synthesis: Option<SynthesisConfig>,
    /// Global settings
    pub global: GlobalConfig,
    /// Performance settings
    pub performance: PerformanceConfig,
    /// Integration settings
    pub integration: IntegrationConfig,
}

/// Recognition-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionConfig {
    /// ASR configuration
    pub asr: ASRConfig,
    /// Phoneme recognition configuration
    pub phoneme: PhonemeRecognitionConfig,
    /// Audio analysis configuration
    pub analysis: AudioAnalysisConfig,
    /// Streaming configuration
    pub streaming: StreamingConfig,
}

/// Synthesis configuration placeholder (would be imported from SDK)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisConfig {
    /// Text-to-speech settings
    pub tts: TtsConfig,
    /// Voice settings
    pub voice: VoiceConfig,
}

/// TTS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsConfig {
    /// Model type
    pub model: String,
    /// Quality settings
    pub quality: QualityLevel,
    /// Speed settings
    pub speed: f32,
}

/// Voice configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConfig {
    /// Voice ID
    pub voice_id: String,
    /// Language
    pub language: LanguageCode,
    /// Emotion settings
    pub emotion: Option<EmotionConfig>,
}

/// Emotion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionConfig {
    /// Emotion type
    pub emotion_type: String,
    /// Intensity (0.0 to 1.0)
    pub intensity: f32,
}

/// Quality level enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityLevel {
    /// Low quality
    Low,
    /// Medium quality
    Medium,
    /// High quality
    High,
    /// Ultra high quality
    Ultra,
}

/// Global configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalConfig {
    /// Default language
    pub default_language: LanguageCode,
    /// Log level
    pub log_level: String,
    /// Debug mode
    pub debug: bool,
    /// Temporary directory
    pub temp_dir: Option<String>,
    /// Cache directory
    pub cache_dir: Option<String>,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Maximum memory usage in MB
    pub max_memory_mb: f32,
    /// Maximum CPU cores to use
    pub max_cpu_cores: u32,
    /// GPU settings
    pub gpu: GpuConfig,
    /// Batch processing settings
    pub batch: BatchConfig,
}

/// GPU configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Enable GPU acceleration
    pub enabled: bool,
    /// GPU device ID
    pub device_id: Option<u32>,
    /// Maximum GPU memory in MB
    pub max_memory_mb: Option<f32>,
    /// Mixed precision
    pub mixed_precision: bool,
}

/// Batch processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Default batch size
    pub default_batch_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Batch timeout in seconds
    pub timeout_seconds: u64,
}

/// Integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Component coordination
    pub coordination: CoordinationConfig,
    /// Pipeline settings
    pub pipeline: PipelineConfig,
    /// Monitoring settings
    pub monitoring: MonitoringConfig,
}

/// Component coordination configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationConfig {
    /// Enable component coordination
    pub enabled: bool,
    /// Coordination protocol
    pub protocol: String,
    /// Heartbeat interval in seconds
    pub heartbeat_interval: u64,
}

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Default pipeline mode
    pub default_mode: PipelineMode,
    /// Buffer size
    pub buffer_size: usize,
    /// Timeout settings
    pub timeout_seconds: u64,
}

/// Pipeline mode enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineMode {
    /// Sequential processing
    Sequential,
    /// Parallel processing
    Parallel,
    /// Streaming processing
    Streaming,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable monitoring
    pub enabled: bool,
    /// Metrics collection interval
    pub metrics_interval_seconds: u64,
    /// Health check interval
    pub health_check_interval_seconds: u64,
}

/// Streaming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Chunk size in samples
    pub chunk_size: usize,
    /// Overlap in samples
    pub overlap: usize,
    /// Latency mode
    pub latency_mode: LatencyMode,
    /// Buffer duration in seconds
    pub buffer_duration: f32,
}

/// Latency mode enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LatencyMode {
    /// Ultra low latency
    UltraLow,
    /// Low latency
    Low,
    /// Balanced
    Balanced,
    /// High accuracy
    HighAccuracy,
    /// Accurate mode (alias for HighAccuracy)
    Accurate,
}

/// Unified configuration builder
pub struct UnifiedConfigBuilder {
    config: UnifiedVoirsConfig,
}

impl UnifiedConfigBuilder {
    /// Create new unified config builder
    pub fn new() -> Self {
        Self {
            config: UnifiedVoirsConfig {
                recognition: RecognitionConfig::default(),
                synthesis: None,
                global: GlobalConfig::default(),
                performance: PerformanceConfig::default(),
                integration: IntegrationConfig::default(),
            },
        }
    }

    /// Set recognition configuration
    pub fn with_recognition(mut self, recognition: RecognitionConfig) -> Self {
        self.config.recognition = recognition;
        self
    }

    /// Set synthesis configuration
    pub fn with_synthesis(mut self, synthesis: SynthesisConfig) -> Self {
        self.config.synthesis = Some(synthesis);
        self
    }

    /// Set global configuration
    pub fn with_global(mut self, global: GlobalConfig) -> Self {
        self.config.global = global;
        self
    }

    /// Set performance configuration
    pub fn with_performance(mut self, performance: PerformanceConfig) -> Self {
        self.config.performance = performance;
        self
    }

    /// Set integration configuration
    pub fn with_integration(mut self, integration: IntegrationConfig) -> Self {
        self.config.integration = integration;
        self
    }

    /// Build the unified configuration
    pub fn build(self) -> UnifiedVoirsConfig {
        self.config
    }
}

impl Default for UnifiedConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration presets for common use cases
pub struct ConfigPresets;

impl ConfigPresets {
    /// Development preset
    pub fn development() -> UnifiedVoirsConfig {
        UnifiedConfigBuilder::new()
            .with_global(GlobalConfig {
                default_language: LanguageCode::EnUs,
                log_level: "debug".to_string(),
                debug: true,
                temp_dir: Some("/tmp/voirs".to_string()),
                cache_dir: Some("/tmp/voirs/cache".to_string()),
            })
            .with_performance(PerformanceConfig {
                max_memory_mb: 2048.0,
                max_cpu_cores: 4,
                gpu: GpuConfig {
                    enabled: false,
                    device_id: None,
                    max_memory_mb: None,
                    mixed_precision: false,
                },
                batch: BatchConfig {
                    default_batch_size: 1,
                    max_batch_size: 4,
                    timeout_seconds: 30,
                },
            })
            .build()
    }

    /// Production preset
    pub fn production() -> UnifiedVoirsConfig {
        UnifiedConfigBuilder::new()
            .with_global(GlobalConfig {
                default_language: LanguageCode::EnUs,
                log_level: "info".to_string(),
                debug: false,
                temp_dir: None,
                cache_dir: Some("/var/cache/voirs".to_string()),
            })
            .with_performance(PerformanceConfig {
                max_memory_mb: 8192.0,
                max_cpu_cores: 8,
                gpu: GpuConfig {
                    enabled: true,
                    device_id: Some(0),
                    max_memory_mb: Some(4096.0),
                    mixed_precision: true,
                },
                batch: BatchConfig {
                    default_batch_size: 8,
                    max_batch_size: 32,
                    timeout_seconds: 60,
                },
            })
            .build()
    }

    /// High performance preset
    pub fn high_performance() -> UnifiedVoirsConfig {
        UnifiedConfigBuilder::new()
            .with_global(GlobalConfig {
                default_language: LanguageCode::EnUs,
                log_level: "warn".to_string(),
                debug: false,
                temp_dir: None,
                cache_dir: Some("/var/cache/voirs".to_string()),
            })
            .with_performance(PerformanceConfig {
                max_memory_mb: 16384.0,
                max_cpu_cores: 16,
                gpu: GpuConfig {
                    enabled: true,
                    device_id: Some(0),
                    max_memory_mb: Some(8192.0),
                    mixed_precision: true,
                },
                batch: BatchConfig {
                    default_batch_size: 16,
                    max_batch_size: 64,
                    timeout_seconds: 120,
                },
            })
            .build()
    }

    /// Low resource preset
    pub fn low_resource() -> UnifiedVoirsConfig {
        UnifiedConfigBuilder::new()
            .with_global(GlobalConfig {
                default_language: LanguageCode::EnUs,
                log_level: "error".to_string(),
                debug: false,
                temp_dir: None,
                cache_dir: None,
            })
            .with_performance(PerformanceConfig {
                max_memory_mb: 512.0,
                max_cpu_cores: 2,
                gpu: GpuConfig {
                    enabled: false,
                    device_id: None,
                    max_memory_mb: None,
                    mixed_precision: false,
                },
                batch: BatchConfig {
                    default_batch_size: 1,
                    max_batch_size: 2,
                    timeout_seconds: 15,
                },
            })
            .build()
    }
}

/// Default implementations
impl Default for UnifiedVoirsConfig {
    fn default() -> Self {
        ConfigPresets::development()
    }
}

impl Default for RecognitionConfig {
    fn default() -> Self {
        Self {
            asr: ASRConfig::default(),
            phoneme: PhonemeRecognitionConfig::default(),
            analysis: AudioAnalysisConfig::default(),
            streaming: StreamingConfig::default(),
        }
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1024,
            overlap: 256,
            latency_mode: LatencyMode::Balanced,
            buffer_duration: 5.0,
        }
    }
}

impl Default for GlobalConfig {
    fn default() -> Self {
        Self {
            default_language: LanguageCode::EnUs,
            log_level: "info".to_string(),
            debug: false,
            temp_dir: None,
            cache_dir: None,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: 4096.0,
            max_cpu_cores: 8,
            gpu: GpuConfig::default(),
            batch: BatchConfig::default(),
        }
    }
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            device_id: None,
            max_memory_mb: None,
            mixed_precision: false,
        }
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            default_batch_size: 4,
            max_batch_size: 16,
            timeout_seconds: 60,
        }
    }
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            coordination: CoordinationConfig::default(),
            pipeline: PipelineConfig::default(),
            monitoring: MonitoringConfig::default(),
        }
    }
}

impl Default for CoordinationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            protocol: "http".to_string(),
            heartbeat_interval: 30,
        }
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            default_mode: PipelineMode::Sequential,
            buffer_size: 4096,
            timeout_seconds: 30,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            metrics_interval_seconds: 60,
            health_check_interval_seconds: 30,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_config_builder() {
        let config = UnifiedConfigBuilder::new()
            .with_global(GlobalConfig {
                default_language: LanguageCode::EnUs,
                log_level: "debug".to_string(),
                debug: true,
                temp_dir: Some("/tmp".to_string()),
                cache_dir: Some("/cache".to_string()),
            })
            .build();

        assert_eq!(config.global.default_language, LanguageCode::EnUs);
        assert_eq!(config.global.log_level, "debug");
        assert!(config.global.debug);
    }

    #[test]
    fn test_config_presets() {
        let dev_config = ConfigPresets::development();
        assert!(dev_config.global.debug);
        assert_eq!(dev_config.global.log_level, "debug");

        let prod_config = ConfigPresets::production();
        assert!(!prod_config.global.debug);
        assert_eq!(prod_config.global.log_level, "info");

        let hp_config = ConfigPresets::high_performance();
        assert!(hp_config.performance.gpu.enabled);
        assert_eq!(hp_config.performance.max_cpu_cores, 16);

        let lr_config = ConfigPresets::low_resource();
        assert!(!lr_config.performance.gpu.enabled);
        assert_eq!(lr_config.performance.max_memory_mb, 512.0);
    }

    #[test]
    fn test_default_config() {
        let config = UnifiedVoirsConfig::default();
        assert_eq!(config.global.default_language, LanguageCode::EnUs);
        assert_eq!(config.performance.max_memory_mb, 2048.0);
        assert_eq!(config.integration.coordination.enabled, true);
    }
}
