//! Model management types for CLI.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    /// Acoustic models (text-to-mel conversion)
    Acoustic,
    /// Vocoder models (mel-to-audio conversion)
    Vocoder,
    /// Grapheme-to-phoneme models
    G2P,
}

/// Model information for CLI operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Unique model identifier
    pub id: String,
    /// Human-readable model name
    pub name: String,
    /// Type of model
    pub model_type: ModelType,
    /// Primary language supported
    pub language: String,
    /// Model description
    pub description: String,
    /// Model version
    pub version: String,
    /// Model size in megabytes
    pub size_mb: f64,
    /// Supported sample rate in Hz
    pub sample_rate: u32,
    /// Quality score (0.0-5.0)
    pub quality_score: f32,
    /// Supported inference backends
    pub supported_backends: Vec<String>,
    /// Whether model is installed locally
    pub is_installed: bool,
    /// Local installation path if installed
    pub installation_path: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ModelInfo {
    /// Create a new ModelInfo
    pub fn new(
        id: String,
        name: String,
        model_type: ModelType,
        language: String,
        description: String,
    ) -> Self {
        Self {
            id,
            name,
            model_type,
            language,
            description,
            version: "1.0.0".to_string(),
            size_mb: 0.0,
            sample_rate: 22050,
            quality_score: 3.5,
            supported_backends: vec!["pytorch".to_string()],
            is_installed: false,
            installation_path: None,
            metadata: HashMap::new(),
        }
    }

    /// Check if model supports a specific backend
    pub fn supports_backend(&self, backend: &str) -> bool {
        self.supported_backends.iter().any(|b| b == backend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_info_creation() {
        let model = ModelInfo::new(
            "test-model".to_string(),
            "Test Model".to_string(),
            ModelType::Acoustic,
            "en".to_string(),
            "A test model".to_string(),
        );

        assert_eq!(model.id, "test-model");
        assert_eq!(model.model_type, ModelType::Acoustic);
        assert!(!model.is_installed);
    }

    #[test]
    fn test_supports_backend() {
        let mut model = ModelInfo::new(
            "test".to_string(),
            "Test".to_string(),
            ModelType::Vocoder,
            "en".to_string(),
            "Test".to_string(),
        );

        model.supported_backends = vec!["pytorch".to_string(), "onnx".to_string()];

        assert!(model.supports_backend("pytorch"));
        assert!(model.supports_backend("onnx"));
        assert!(!model.supports_backend("tensorflow"));
    }
}
