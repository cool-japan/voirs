//! Vocoder model definitions and implementations.

pub mod diffwave;
pub mod hifigan;
pub mod singing;
pub mod spatial;
pub mod vits2;

use crate::{Result, VocoderError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Vocoder model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocoderModelConfig {
    /// Model name
    pub name: String,
    /// Model architecture type
    pub architecture: String,
    /// Model version
    pub version: String,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of mel channels
    pub mel_channels: u32,
    /// Model file path
    pub model_path: Option<String>,
    /// Model parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

impl VocoderModelConfig {
    /// Create new model configuration
    pub fn new(
        name: String,
        architecture: String,
        version: String,
        sample_rate: u32,
        mel_channels: u32,
    ) -> Self {
        Self {
            name,
            architecture,
            version,
            sample_rate,
            mel_channels,
            model_path: None,
            parameters: HashMap::new(),
        }
    }

    /// Set model path
    pub fn with_model_path(mut self, path: String) -> Self {
        self.model_path = Some(path);
        self
    }

    /// Add parameter
    pub fn with_parameter(mut self, key: String, value: serde_json::Value) -> Self {
        self.parameters.insert(key, value);
        self
    }
}

/// Model loader for different formats
pub struct ModelLoader {
    /// Supported model formats
    supported_formats: Vec<String>,
}

impl ModelLoader {
    /// Create new model loader
    pub fn new() -> Self {
        Self {
            supported_formats: vec![
                "safetensors".to_string(),
                "pytorch".to_string(),
                "onnx".to_string(),
            ],
        }
    }

    /// Load model from file
    pub fn load_from_file<P: AsRef<Path>>(&self, path: P) -> Result<VocoderModelConfig> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| VocoderError::ModelError("Invalid file extension".to_string()))?;

        if !self.supported_formats.iter().any(|f| f == extension) {
            return Err(VocoderError::ModelError(format!(
                "Unsupported model format: {extension}"
            )));
        }

        // Extract model name from filename
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Implement proper model loading based on format
        use crate::backends::loader::ModelLoader;

        let mut loader = ModelLoader::new();

        match tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(loader.load_from_file(path))
        {
            Ok(model_info) => {
                let sample_rate = model_info
                    .metadata
                    .supported_sample_rates
                    .first()
                    .copied()
                    .unwrap_or(22050);
                let config = VocoderModelConfig::new(
                    model_info.metadata.name.clone(),
                    format!("{:?}", model_info.format),
                    model_info.metadata.version.clone(),
                    sample_rate,
                    model_info.metadata.mel_channels,
                )
                .with_model_path(path.to_string_lossy().to_string());

                Ok(config)
            }
            Err(e) => {
                eprintln!("Warning: Could not load model info: {e}. Using basic config.");
                // Fallback to basic configuration
                Ok(VocoderModelConfig::new(
                    name,
                    "unknown".to_string(),
                    "1.0.0".to_string(),
                    22050,
                    80,
                )
                .with_model_path(path.to_string_lossy().to_string()))
            }
        }
    }

    /// Check if format is supported
    pub fn supports_format(&self, format: &str) -> bool {
        self.supported_formats.contains(&format.to_string())
    }

    /// Get supported formats
    pub fn supported_formats(&self) -> &[String] {
        &self.supported_formats
    }
}

impl Default for ModelLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config() {
        let config = VocoderModelConfig::new(
            "test_model".to_string(),
            "HiFi-GAN".to_string(),
            "1.0.0".to_string(),
            22050,
            80,
        )
        .with_parameter("batch_size".to_string(), serde_json::json!(1))
        .with_model_path("/path/to/model.safetensors".to_string());

        assert_eq!(config.name, "test_model");
        assert_eq!(config.architecture, "HiFi-GAN");
        assert_eq!(config.sample_rate, 22050);
        assert_eq!(config.mel_channels, 80);
        assert!(config.model_path.is_some());
        assert_eq!(config.parameters.len(), 1);
    }

    #[test]
    fn test_model_loader() {
        let loader = ModelLoader::new();

        assert!(loader.supports_format("safetensors"));
        assert!(loader.supports_format("pytorch"));
        assert!(loader.supports_format("onnx"));
        assert!(!loader.supports_format("unknown"));

        assert_eq!(loader.supported_formats().len(), 3);
    }
}
