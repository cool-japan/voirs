//! Model configuration module.
//!
//! This module provides configuration options for model architecture,
//! backend selection, and optimization settings.

use serde::{Deserialize, Serialize};
use super::{DeviceType, PerformanceProfile, ValidationResult};
use std::path::PathBuf;

/// Vocoder model architecture types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelArchitecture {
    /// HiFi-GAN vocoder
    HiFiGAN,
    /// WaveGlow vocoder
    WaveGlow,
    /// DiffWave diffusion vocoder
    DiffWave,
    /// WaveNet vocoder
    WaveNet,
    /// Universal vocoder (auto-select)
    Universal,
}

impl Default for ModelArchitecture {
    fn default() -> Self {
        Self::HiFiGAN
    }
}

/// HiFi-GAN model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HiFiGANVariant {
    /// HiFi-GAN V1 (highest quality)
    V1,
    /// HiFi-GAN V2 (balanced)
    V2,
    /// HiFi-GAN V3 (fastest)
    V3,
    /// Universal HiFi-GAN
    Universal,
}

impl Default for HiFiGANVariant {
    fn default() -> Self {
        Self::V2
    }
}

/// Backend types for model inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendType {
    /// Candle (Rust-native)
    Candle,
    /// ONNX Runtime
    ONNX,
    /// PyTorch (via bindings)
    PyTorch,
    /// TensorFlow Lite
    TensorFlowLite,
    /// Auto-select backend
    Auto,
}

impl Default for BackendType {
    fn default() -> Self {
        Self::Candle
    }
}

/// Model optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable quantization
    pub enable_quantization: bool,
    /// Quantization precision (8, 16, or 32 bits)
    pub quantization_bits: u8,
    /// Enable graph optimization
    pub enable_graph_optimization: bool,
    /// Enable kernel fusion
    pub enable_kernel_fusion: bool,
    /// Enable constant folding
    pub enable_constant_folding: bool,
    /// Enable operator parallelization
    pub enable_operator_parallelization: bool,
    /// Maximum number of threads for model inference
    pub max_threads: usize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_quantization: false,
            quantization_bits: 16,
            enable_graph_optimization: true,
            enable_kernel_fusion: true,
            enable_constant_folding: true,
            enable_operator_parallelization: true,
            max_threads: 0, // 0 = auto-detect
        }
    }
}

/// Model caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable model caching
    pub enable_caching: bool,
    /// Cache directory path
    pub cache_dir: Option<PathBuf>,
    /// Maximum cache size in MB
    pub max_cache_size_mb: u64,
    /// Cache expiry time in hours
    pub cache_expiry_hours: u64,
    /// Enable preloading of frequently used models
    pub enable_preloading: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_dir: None, // Will use system default
            max_cache_size_mb: 1024, // 1GB
            cache_expiry_hours: 168, // 1 week
            enable_preloading: false,
        }
    }
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model architecture
    pub architecture: ModelArchitecture,
    /// HiFi-GAN variant (if applicable)
    pub hifigan_variant: HiFiGANVariant,
    /// Backend for model inference
    pub backend: BackendType,
    /// Device type for processing
    pub device: DeviceType,
    /// Performance profile
    pub profile: PerformanceProfile,
    /// Model file path
    pub model_path: Option<PathBuf>,
    /// Model URL for downloading
    pub model_url: Option<String>,
    /// Model checksum for verification
    pub model_checksum: Option<String>,
    /// Model metadata
    pub model_metadata: ModelMetadata,
    /// Optimization settings
    pub optimization: OptimizationConfig,
    /// Caching configuration
    pub cache: CacheConfig,
    /// Enable model validation on load
    pub enable_validation: bool,
    /// Enable model benchmarking
    pub enable_benchmarking: bool,
    /// Maximum model memory usage in MB
    pub max_memory_mb: Option<u64>,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model author/organization
    pub author: String,
    /// Model description
    pub description: String,
    /// Supported sample rates
    pub supported_sample_rates: Vec<u32>,
    /// Number of mel channels
    pub mel_channels: u32,
    /// Model file size in bytes
    pub file_size_bytes: Option<u64>,
    /// Model license
    pub license: String,
    /// Model creation date
    pub created_date: Option<String>,
    /// Model tags
    pub tags: Vec<String>,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            name: "Unknown".to_string(),
            version: "1.0.0".to_string(),
            author: "Unknown".to_string(),
            description: "Neural vocoder model".to_string(),
            supported_sample_rates: vec![22050, 44100],
            mel_channels: 80,
            file_size_bytes: None,
            license: "Unknown".to_string(),
            created_date: None,
            tags: vec![],
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architecture: ModelArchitecture::HiFiGAN,
            hifigan_variant: HiFiGANVariant::V2,
            backend: BackendType::Candle,
            device: DeviceType::Auto,
            profile: PerformanceProfile::Balanced,
            model_path: None,
            model_url: None,
            model_checksum: None,
            model_metadata: ModelMetadata::default(),
            optimization: OptimizationConfig::default(),
            cache: CacheConfig::default(),
            enable_validation: true,
            enable_benchmarking: false,
            max_memory_mb: None,
        }
    }
}

impl ModelConfig {
    /// Create configuration for high-quality processing
    pub fn high_quality() -> Self {
        Self {
            architecture: ModelArchitecture::HiFiGAN,
            hifigan_variant: HiFiGANVariant::V1,
            backend: BackendType::Candle,
            device: DeviceType::Auto,
            profile: PerformanceProfile::Quality,
            optimization: OptimizationConfig {
                enable_quantization: false,
                quantization_bits: 32,
                enable_graph_optimization: true,
                enable_kernel_fusion: true,
                enable_constant_folding: true,
                enable_operator_parallelization: true,
                max_threads: 0,
            },
            cache: CacheConfig {
                enable_caching: true,
                max_cache_size_mb: 2048,
                enable_preloading: true,
                ..Default::default()
            },
            enable_validation: true,
            enable_benchmarking: true,
            ..Default::default()
        }
    }

    /// Create configuration for real-time processing
    pub fn realtime() -> Self {
        Self {
            architecture: ModelArchitecture::HiFiGAN,
            hifigan_variant: HiFiGANVariant::V3,
            backend: BackendType::ONNX,
            device: DeviceType::Auto,
            profile: PerformanceProfile::Realtime,
            optimization: OptimizationConfig {
                enable_quantization: true,
                quantization_bits: 16,
                enable_graph_optimization: true,
                enable_kernel_fusion: true,
                enable_constant_folding: true,
                enable_operator_parallelization: true,
                max_threads: 2,
            },
            cache: CacheConfig {
                enable_caching: true,
                max_cache_size_mb: 256,
                enable_preloading: true,
                ..Default::default()
            },
            enable_validation: false,
            enable_benchmarking: false,
            max_memory_mb: Some(512),
            ..Default::default()
        }
    }

    /// Create configuration for low-resource environments
    pub fn low_resource() -> Self {
        Self {
            architecture: ModelArchitecture::HiFiGAN,
            hifigan_variant: HiFiGANVariant::V3,
            backend: BackendType::ONNX,
            device: DeviceType::Cpu,
            profile: PerformanceProfile::Speed,
            optimization: OptimizationConfig {
                enable_quantization: true,
                quantization_bits: 8,
                enable_graph_optimization: true,
                enable_kernel_fusion: false,
                enable_constant_folding: true,
                enable_operator_parallelization: false,
                max_threads: 1,
            },
            cache: CacheConfig {
                enable_caching: false,
                max_cache_size_mb: 128,
                enable_preloading: false,
                ..Default::default()
            },
            enable_validation: false,
            enable_benchmarking: false,
            max_memory_mb: Some(256),
            ..Default::default()
        }
    }

    /// Validate the model configuration
    pub fn validate(&self) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Validate model path
        if let Some(ref path) = self.model_path {
            if !path.exists() {
                errors.push(format!("Model file does not exist: {}", path.display()));
            }
        }

        // Validate quantization settings
        if self.optimization.enable_quantization {
            match self.optimization.quantization_bits {
                8 | 16 | 32 => {},
                _ => errors.push("Quantization bits must be 8, 16, or 32".to_string()),
            }
        }

        // Validate thread count
        if self.optimization.max_threads > 64 {
            warnings.push("High thread count may cause contention".to_string());
        }

        // Validate memory limits
        if let Some(max_memory) = self.max_memory_mb {
            if max_memory < 64 {
                warnings.push("Low memory limit may cause loading failures".to_string());
            }
        }

        // Validate cache settings
        if self.cache.enable_caching {
            if self.cache.max_cache_size_mb < 64 {
                warnings.push("Low cache size may reduce performance".to_string());
            }
            if self.cache.cache_expiry_hours == 0 {
                warnings.push("Zero cache expiry may cause excessive storage usage".to_string());
            }
        }

        // Architecture-specific validation
        match self.architecture {
            ModelArchitecture::HiFiGAN => {
                // HiFi-GAN specific validations
                if self.model_metadata.mel_channels < 32 {
                    warnings.push("Low mel channels may reduce quality for HiFi-GAN".to_string());
                }
            }
            ModelArchitecture::DiffWave => {
                // DiffWave specific validations
                if self.optimization.enable_quantization && self.optimization.quantization_bits < 16 {
                    warnings.push("Low quantization may affect DiffWave quality".to_string());
                }
            }
            _ => {}
        }

        // Backend-specific validation
        match self.backend {
            BackendType::ONNX => {
                if !self.optimization.enable_graph_optimization {
                    warnings.push("Graph optimization recommended for ONNX backend".to_string());
                }
            }
            BackendType::Candle => {
                if self.optimization.quantization_bits == 8 {
                    warnings.push("8-bit quantization may not be fully supported in Candle".to_string());
                }
            }
            _ => {}
        }

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
        }
    }

    /// Get estimated memory usage in MB
    pub fn estimated_memory_mb(&self) -> f32 {
        let mut memory_mb = 100.0; // Base model memory

        // Adjust for architecture
        memory_mb *= match self.architecture {
            ModelArchitecture::HiFiGAN => 1.0,
            ModelArchitecture::WaveGlow => 1.5,
            ModelArchitecture::DiffWave => 2.0,
            ModelArchitecture::WaveNet => 3.0,
            ModelArchitecture::Universal => 1.2,
        };

        // Adjust for quantization
        if self.optimization.enable_quantization {
            memory_mb *= match self.optimization.quantization_bits {
                8 => 0.25,
                16 => 0.5,
                32 => 1.0,
                _ => 1.0,
            };
        }

        // Add cache memory
        if self.cache.enable_caching {
            memory_mb += self.cache.max_cache_size_mb as f32 * 0.1; // Assume 10% cache usage
        }

        memory_mb
    }

    /// Check if GPU acceleration is available
    pub fn is_gpu_available(&self) -> bool {
        matches!(self.device, DeviceType::Cuda | DeviceType::Metal | DeviceType::Auto)
    }

    /// Get model identifier string
    pub fn model_identifier(&self) -> String {
        format!(
            "{:?}-{:?}-{:?}",
            self.architecture,
            self.hifigan_variant,
            self.backend
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_model_config() {
        let config = ModelConfig::default();
        assert_eq!(config.architecture, ModelArchitecture::HiFiGAN);
        assert_eq!(config.backend, BackendType::Candle);
        assert!(config.enable_validation);
    }

    #[test]
    fn test_high_quality_config() {
        let config = ModelConfig::high_quality();
        assert_eq!(config.hifigan_variant, HiFiGANVariant::V1);
        assert!(!config.optimization.enable_quantization);
        assert!(config.enable_benchmarking);
    }

    #[test]
    fn test_realtime_config() {
        let config = ModelConfig::realtime();
        assert_eq!(config.hifigan_variant, HiFiGANVariant::V3);
        assert!(config.optimization.enable_quantization);
        assert!(!config.enable_validation);
    }

    #[test]
    fn test_low_resource_config() {
        let config = ModelConfig::low_resource();
        assert_eq!(config.device, DeviceType::Cpu);
        assert_eq!(config.optimization.quantization_bits, 8);
        assert!(!config.cache.enable_caching);
    }

    #[test]
    fn test_config_validation() {
        let mut config = ModelConfig::default();
        
        // Valid configuration
        let result = config.validate();
        assert!(result.is_valid);

        // Invalid quantization bits
        config.optimization.enable_quantization = true;
        config.optimization.quantization_bits = 7;
        let result = config.validate();
        assert!(!result.is_valid);
    }

    #[test]
    fn test_memory_estimation() {
        let config = ModelConfig::default();
        let memory_mb = config.estimated_memory_mb();
        assert!(memory_mb > 0.0);
        assert!(memory_mb < 1000.0); // Reasonable upper bound
    }

    #[test]
    fn test_gpu_availability() {
        let mut config = ModelConfig::default();
        
        config.device = DeviceType::Cpu;
        assert!(!config.is_gpu_available());
        
        config.device = DeviceType::Cuda;
        assert!(config.is_gpu_available());
    }

    #[test]
    fn test_model_identifier() {
        let config = ModelConfig::default();
        let identifier = config.model_identifier();
        assert!(identifier.contains("HiFiGAN"));
        assert!(identifier.contains("V2"));
        assert!(identifier.contains("Candle"));
    }
}