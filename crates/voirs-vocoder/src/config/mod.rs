//! Configuration system for voirs-vocoder.
//!
//! This module provides comprehensive configuration management including:
//! - Vocoding quality and performance settings
//! - Streaming and real-time processing configuration
//! - Model architecture and backend selection
//! - Audio processing and enhancement options

pub mod model;
pub mod streaming;
pub mod vocoding;

pub use model::*;
pub use streaming::*;
pub use vocoding::*;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Master configuration for the vocoder system
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VocoderConfig {
    /// Vocoding quality and processing settings
    pub vocoding: VocodingConfig,
    /// Streaming and real-time settings
    pub streaming: StreamingConfig,
    /// Model architecture and backend settings
    pub model: ModelConfig,
    /// Audio processing settings
    pub audio: AudioProcessingConfig,
}

/// Audio processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioProcessingConfig {
    /// Sample rate for audio processing
    pub sample_rate: u32,
    /// Number of channels (1=mono, 2=stereo)
    pub channels: u32,
    /// Bit depth for audio output
    pub bit_depth: u16,
    /// Enable audio enhancement
    pub enable_enhancement: bool,
    /// Enable noise reduction
    pub enable_noise_reduction: bool,
    /// Enable dynamic range compression
    pub enable_compression: bool,
    /// Gain adjustment in dB
    pub gain_db: f32,
}

impl Default for AudioProcessingConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            channels: 1,
            bit_depth: 16,
            enable_enhancement: true,
            enable_noise_reduction: true,
            enable_compression: false,
            gain_db: 0.0,
        }
    }
}

/// Device configuration for processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    /// CPU processing
    Cpu,
    /// CUDA GPU processing
    Cuda,
    /// Metal GPU processing (macOS)
    Metal,
    /// Automatic device selection
    Auto,
}

impl Default for DeviceType {
    fn default() -> Self {
        Self::Auto
    }
}

/// Performance profile for processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceProfile {
    /// Optimized for quality (slower)
    Quality,
    /// Balanced quality and speed
    Balanced,
    /// Optimized for speed (lower quality)
    Speed,
    /// Real-time processing (lowest latency)
    Realtime,
}

impl Default for PerformanceProfile {
    fn default() -> Self {
        Self::Balanced
    }
}

/// Configuration validation result
#[derive(Debug)]
pub struct ValidationResult {
    /// Whether the configuration is valid
    pub is_valid: bool,
    /// List of validation errors
    pub errors: Vec<String>,
    /// List of validation warnings
    pub warnings: Vec<String>,
}

impl VocoderConfig {
    /// Create configuration for high-quality processing
    pub fn high_quality() -> Self {
        Self {
            vocoding: VocodingConfig::high_quality(),
            streaming: StreamingConfig::high_quality(),
            model: ModelConfig::high_quality(),
            audio: AudioProcessingConfig {
                sample_rate: 44100,
                bit_depth: 24,
                enable_enhancement: true,
                enable_noise_reduction: true,
                enable_compression: false,
                ..Default::default()
            },
        }
    }

    /// Create configuration for real-time processing
    pub fn realtime() -> Self {
        Self {
            vocoding: VocodingConfig::realtime(),
            streaming: StreamingConfig::realtime(),
            model: ModelConfig::realtime(),
            audio: AudioProcessingConfig {
                sample_rate: 22050,
                bit_depth: 16,
                enable_enhancement: false,
                enable_noise_reduction: false,
                enable_compression: true,
                ..Default::default()
            },
        }
    }

    /// Create configuration for low-resource environments
    pub fn low_resource() -> Self {
        Self {
            vocoding: VocodingConfig::low_resource(),
            streaming: StreamingConfig::low_resource(),
            model: ModelConfig::low_resource(),
            audio: AudioProcessingConfig {
                sample_rate: 16000,
                bit_depth: 16,
                enable_enhancement: false,
                enable_noise_reduction: false,
                enable_compression: true,
                ..Default::default()
            },
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Validate sample rate
        if self.audio.sample_rate < 8000 {
            errors.push("Sample rate must be at least 8000 Hz".to_string());
        } else if self.audio.sample_rate > 96000 {
            warnings.push(
                "Sample rate above 96000 Hz may not be supported by all backends".to_string(),
            );
        }

        // Validate bit depth
        match self.audio.bit_depth {
            16 | 24 | 32 => {}
            _ => errors.push("Bit depth must be 16, 24, or 32".to_string()),
        }

        // Validate channels
        if self.audio.channels == 0 || self.audio.channels > 8 {
            errors.push("Number of channels must be between 1 and 8".to_string());
        }

        // Validate vocoding configuration
        let vocoding_result = self.vocoding.validate();
        errors.extend(vocoding_result.errors);
        warnings.extend(vocoding_result.warnings);

        // Validate streaming configuration
        let streaming_result = self.streaming.validate();
        errors.extend(streaming_result.errors);
        warnings.extend(streaming_result.warnings);

        // Validate model configuration
        let model_result = self.model.validate();
        errors.extend(model_result.errors);
        warnings.extend(model_result.warnings);

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
        }
    }

    /// Load configuration from TOML file
    pub fn load_from_file(path: &PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to TOML file
    pub fn save_to_file(&self, path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Load configuration from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Convert configuration to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = VocoderConfig::default();
        assert_eq!(config.audio.sample_rate, 22050);
        assert_eq!(config.audio.channels, 1);
        assert_eq!(config.audio.bit_depth, 16);
    }

    #[test]
    fn test_high_quality_config() {
        let config = VocoderConfig::high_quality();
        assert_eq!(config.audio.sample_rate, 44100);
        assert_eq!(config.audio.bit_depth, 24);
        assert!(config.audio.enable_enhancement);
    }

    #[test]
    fn test_realtime_config() {
        let config = VocoderConfig::realtime();
        assert_eq!(config.audio.sample_rate, 22050);
        assert_eq!(config.audio.bit_depth, 16);
        assert!(!config.audio.enable_enhancement);
    }

    #[test]
    fn test_low_resource_config() {
        let config = VocoderConfig::low_resource();
        assert_eq!(config.audio.sample_rate, 16000);
        assert_eq!(config.audio.bit_depth, 16);
        assert!(!config.audio.enable_enhancement);
    }

    #[test]
    fn test_config_validation() {
        let mut config = VocoderConfig::default();

        // Valid configuration should pass
        let result = config.validate();
        assert!(result.is_valid);
        assert!(result.errors.is_empty());

        // Invalid sample rate should fail
        config.audio.sample_rate = 1000;
        let result = config.validate();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_json_serialization() {
        let config = VocoderConfig::default();
        let json = config.to_json().unwrap();
        let deserialized = VocoderConfig::from_json(&json).unwrap();

        assert_eq!(config.audio.sample_rate, deserialized.audio.sample_rate);
        assert_eq!(config.audio.channels, deserialized.audio.channels);
    }
}
