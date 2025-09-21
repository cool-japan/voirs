//! Configuration management for acoustic models
//!
//! This module provides comprehensive configuration management for acoustic models,
//! including model-specific settings, synthesis parameters, and runtime options.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::{AcousticError, Result};

pub mod model;
pub mod runtime;
pub mod synthesis;

pub use model::*;
pub use runtime::*;
pub use synthesis::*;

/// Main configuration structure that combines all config types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticConfig {
    /// Model configuration
    pub model: ModelConfig,
    /// Synthesis configuration
    pub synthesis: SynthesisConfig,
    /// Runtime configuration
    pub runtime: RuntimeConfig,
}

impl AcousticConfig {
    /// Create new configuration with defaults
    pub fn new() -> Self {
        Self {
            model: ModelConfig::default(),
            synthesis: SynthesisConfig::default(),
            runtime: RuntimeConfig::default(),
        }
    }

    /// Load configuration from file
    pub fn load_from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| AcousticError::ConfigError(format!("Failed to read config file: {e}")))?;

        let config: AcousticConfig = serde_json::from_str(&content)
            .map_err(|e| AcousticError::ConfigError(format!("Failed to parse config: {e}")))?;

        config.validate()?;
        Ok(config)
    }

    /// Save configuration to file
    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| AcousticError::ConfigError(format!("Failed to serialize config: {e}")))?;

        std::fs::write(path, content)
            .map_err(|e| AcousticError::ConfigError(format!("Failed to write config file: {e}")))?;

        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        self.model.validate()?;
        self.synthesis.validate()?;
        self.runtime.validate()?;
        Ok(())
    }

    /// Merge with another configuration (other takes precedence)
    pub fn merge(&mut self, other: &AcousticConfig) {
        self.model.merge(&other.model);
        self.synthesis.merge(&other.synthesis);
        self.runtime.merge(&other.runtime);
    }
}

impl Default for AcousticConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Device configuration for model execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Device type (CPU, CUDA, Metal, etc.)
    pub device_type: DeviceType,
    /// Device index (for multi-GPU systems)
    pub device_index: Option<u32>,
    /// Use mixed precision (FP16/FP32)
    pub mixed_precision: bool,
    /// Maximum memory usage in MB
    pub max_memory_mb: Option<u32>,
}

impl DeviceConfig {
    /// Create CPU configuration
    pub fn cpu() -> Self {
        Self {
            device_type: DeviceType::Cpu,
            device_index: None,
            mixed_precision: false,
            max_memory_mb: None,
        }
    }

    /// Create CUDA configuration
    pub fn cuda(device_index: Option<u32>) -> Self {
        Self {
            device_type: DeviceType::Cuda,
            device_index,
            mixed_precision: true,
            max_memory_mb: None,
        }
    }

    /// Create Metal configuration
    pub fn metal() -> Self {
        Self {
            device_type: DeviceType::Metal,
            device_index: None,
            mixed_precision: true,
            max_memory_mb: None,
        }
    }
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self::cpu()
    }
}

/// Supported device types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceType {
    /// CPU execution
    Cpu,
    /// NVIDIA CUDA
    Cuda,
    /// Apple Metal
    Metal,
    /// OpenCL
    OpenCl,
}

impl DeviceType {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            DeviceType::Cpu => "cpu",
            DeviceType::Cuda => "cuda",
            DeviceType::Metal => "metal",
            DeviceType::OpenCl => "opencl",
        }
    }
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable model caching
    pub enabled: bool,
    /// Cache directory
    pub cache_dir: PathBuf,
    /// Maximum cache size in MB
    pub max_size_mb: u32,
    /// Cache TTL in seconds
    pub ttl_seconds: u32,
}

impl CacheConfig {
    /// Create new cache configuration
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            enabled: true,
            cache_dir,
            max_size_mb: 1024, // 1GB default
            ttl_seconds: 3600, // 1 hour default
        }
    }

    /// Disable caching
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            cache_dir: PathBuf::new(),
            max_size_mb: 0,
            ttl_seconds: 0,
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        let cache_dir = std::env::temp_dir().join("voirs-acoustic");
        Self::new(cache_dir)
    }
}

/// G2P (Grapheme-to-Phoneme) engine types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum G2pEngine {
    /// Rule-based G2P
    RuleBased,
    /// Neural sequence-to-sequence model
    Neural,
    /// Hybrid rule-based + neural
    Hybrid,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_config() {
        let cpu_config = DeviceConfig::cpu();
        assert_eq!(cpu_config.device_type, DeviceType::Cpu);
        assert!(!cpu_config.mixed_precision);

        let cuda_config = DeviceConfig::cuda(Some(0));
        assert_eq!(cuda_config.device_type, DeviceType::Cuda);
        assert!(cuda_config.mixed_precision);
        assert_eq!(cuda_config.device_index, Some(0));
    }

    #[test]
    fn test_device_type_string() {
        assert_eq!(DeviceType::Cpu.as_str(), "cpu");
        assert_eq!(DeviceType::Cuda.as_str(), "cuda");
        assert_eq!(DeviceType::Metal.as_str(), "metal");
        assert_eq!(DeviceType::OpenCl.as_str(), "opencl");
    }

    #[test]
    fn test_cache_config() {
        let cache = CacheConfig::disabled();
        assert!(!cache.enabled);
        assert_eq!(cache.max_size_mb, 0);

        let cache = CacheConfig::default();
        assert!(cache.enabled);
        assert_eq!(cache.max_size_mb, 1024);
    }

    #[test]
    fn test_acoustic_config() {
        let config = AcousticConfig::new();
        assert!(config.validate().is_ok());

        let mut config1 = AcousticConfig::default();
        let config2 = AcousticConfig::default();
        config1.merge(&config2);
        assert!(config1.validate().is_ok());
    }
}
