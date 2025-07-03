//! Model loading and management module.
//!
//! This module provides functionality for loading, validating, and managing
//! neural vocoder models from various sources and formats.

use crate::{Result, VocoderError};
use crate::config::{ModelConfig, ModelMetadata};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;

/// Model format types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    /// SafeTensors format
    SafeTensors,
    /// ONNX format
    ONNX,
    /// PyTorch format
    PyTorch,
    /// TensorFlow format
    TensorFlow,
}

impl ModelFormat {
    /// Get file extensions for this format
    pub fn extensions(&self) -> &[&str] {
        match self {
            ModelFormat::SafeTensors => &["safetensors"],
            ModelFormat::ONNX => &["onnx"],
            ModelFormat::PyTorch => &["pt", "pth"],
            ModelFormat::TensorFlow => &["pb", "tflite"],
        }
    }

    /// Detect format from file extension
    pub fn from_path(path: &Path) -> Option<Self> {
        let extension = path.extension()?.to_str()?.to_lowercase();
        match extension.as_str() {
            "safetensors" => Some(ModelFormat::SafeTensors),
            "onnx" => Some(ModelFormat::ONNX),
            "pt" | "pth" => Some(ModelFormat::PyTorch),
            "pb" | "tflite" => Some(ModelFormat::TensorFlow),
            _ => None,
        }
    }
}

/// Model validation result
#[derive(Debug, Clone)]
pub struct ModelValidationResult {
    /// Whether the model is valid
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Model metadata
    pub metadata: Option<ModelMetadata>,
}

/// Model loader for different formats and sources
pub struct ModelLoader {
    cache_dir: Option<PathBuf>,
    models_cache: HashMap<String, ModelInfo>,
}

/// Information about a loaded model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model path
    pub path: PathBuf,
    /// Model format
    pub format: ModelFormat,
    /// Model metadata
    pub metadata: ModelMetadata,
    /// File size in bytes
    pub size_bytes: u64,
    /// Last modified time
    pub modified_time: std::time::SystemTime,
    /// Checksum (if available)
    pub checksum: Option<String>,
}

impl ModelLoader {
    /// Create new model loader
    pub fn new() -> Self {
        Self {
            cache_dir: None,
            models_cache: HashMap::new(),
        }
    }

    /// Create model loader with cache directory
    pub fn with_cache_dir<P: AsRef<Path>>(cache_dir: P) -> Self {
        Self {
            cache_dir: Some(cache_dir.as_ref().to_path_buf()),
            models_cache: HashMap::new(),
        }
    }

    /// Set cache directory
    pub fn set_cache_dir<P: AsRef<Path>>(&mut self, cache_dir: P) {
        self.cache_dir = Some(cache_dir.as_ref().to_path_buf());
    }

    /// Load model from local file
    pub async fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<ModelInfo> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Err(VocoderError::ModelError(
                format!("Model file not found: {}", path.display())
            ));
        }

        let format = ModelFormat::from_path(path)
            .ok_or_else(|| VocoderError::ModelError(
                format!("Unsupported model format: {}", path.display())
            ))?;

        let metadata = fs::metadata(path)
            .map_err(|e| VocoderError::ModelError(
                format!("Failed to read model metadata: {}", e)
            ))?;

        let size_bytes = metadata.len();
        let modified_time = metadata.modified()
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

        // Extract model metadata
        let model_metadata = self.extract_model_metadata(path, format).await?;

        // Calculate checksum if needed
        let checksum = self.calculate_checksum(path).await?;

        let model_info = ModelInfo {
            path: path.to_path_buf(),
            format,
            metadata: model_metadata,
            size_bytes,
            modified_time,
            checksum: Some(checksum),
        };

        // Cache the model info
        let cache_key = path.to_string_lossy().to_string();
        self.models_cache.insert(cache_key, model_info.clone());

        tracing::info!("Loaded model from {}", path.display());
        Ok(model_info)
    }

    /// Load model from URL
    pub async fn load_from_url(&mut self, url: &str, config: &ModelConfig) -> Result<ModelInfo> {
        let cache_dir = self.cache_dir.as_ref()
            .ok_or_else(|| VocoderError::ConfigError("Cache directory not set".to_string()))?;

        // Create cache directory if it doesn't exist
        fs::create_dir_all(cache_dir)
            .map_err(|e| VocoderError::ConfigError(
                format!("Failed to create cache directory: {}", e)
            ))?;

        // Generate cache file name from URL
        let cache_filename = self.url_to_filename(url);
        let cache_path = cache_dir.join(&cache_filename);

        // Check if model is already cached
        if cache_path.exists() {
            if let Some(checksum) = &config.model_checksum {
                let file_checksum = self.calculate_checksum(&cache_path).await?;
                if file_checksum == *checksum {
                    tracing::info!("Using cached model from {}", cache_path.display());
                    return self.load_from_file(&cache_path).await;
                }
            } else {
                tracing::info!("Using cached model from {}", cache_path.display());
                return self.load_from_file(&cache_path).await;
            }
        }

        // Download model
        tracing::info!("Downloading model from {}", url);
        self.download_model(url, &cache_path).await?;

        // Verify checksum if provided
        if let Some(expected_checksum) = &config.model_checksum {
            let actual_checksum = self.calculate_checksum(&cache_path).await?;
            if actual_checksum != *expected_checksum {
                fs::remove_file(&cache_path).ok(); // Clean up invalid file
                return Err(VocoderError::ModelError(
                    format!("Model checksum mismatch. Expected: {}, Got: {}", 
                           expected_checksum, actual_checksum)
                ));
            }
        }

        self.load_from_file(&cache_path).await
    }

    /// Validate model file
    pub async fn validate_model<P: AsRef<Path>>(&self, path: P) -> Result<ModelValidationResult> {
        let path = path.as_ref();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check if file exists
        if !path.exists() {
            errors.push(format!("Model file not found: {}", path.display()));
            return Ok(ModelValidationResult {
                is_valid: false,
                errors,
                warnings,
                metadata: None,
            });
        }

        // Check file format
        let format = match ModelFormat::from_path(path) {
            Some(f) => f,
            None => {
                errors.push(format!("Unsupported model format: {}", path.display()));
                return Ok(ModelValidationResult {
                    is_valid: false,
                    errors,
                    warnings,
                    metadata: None,
                });
            }
        };

        // Check file size
        let metadata = fs::metadata(path)
            .map_err(|e| VocoderError::ModelError(format!("Failed to read file metadata: {}", e)))?;
        
        let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
        if size_mb < 1.0 {
            warnings.push("Model file is very small and may be incomplete".to_string());
        } else if size_mb > 1000.0 {
            warnings.push("Model file is very large and may cause memory issues".to_string());
        }

        // Format-specific validation
        match format {
            ModelFormat::SafeTensors => {
                self.validate_safetensors(path, &mut errors, &mut warnings).await?;
            }
            ModelFormat::ONNX => {
                self.validate_onnx(path, &mut errors, &mut warnings).await?;
            }
            _ => {
                warnings.push(format!("Validation not implemented for {:?} format", format));
            }
        }

        // Extract metadata if possible
        let model_metadata = if errors.is_empty() {
            self.extract_model_metadata(path, format).await.ok()
        } else {
            None
        };

        Ok(ModelValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            metadata: model_metadata,
        })
    }

    /// Extract model metadata
    async fn extract_model_metadata(&self, path: &Path, format: ModelFormat) -> Result<ModelMetadata> {
        let mut metadata = ModelMetadata::default();
        
        metadata.name = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let file_metadata = fs::metadata(path)
            .map_err(|e| VocoderError::ModelError(format!("Failed to read file metadata: {}", e)))?;
        
        metadata.file_size_bytes = Some(file_metadata.len());

        if let Ok(created_time) = file_metadata.created() {
            if let Ok(duration) = created_time.duration_since(std::time::UNIX_EPOCH) {
                metadata.created_date = Some(
                    chrono::DateTime::from_timestamp(duration.as_secs() as i64, 0)
                        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                        .unwrap_or_default()
                );
            }
        }

        // Format-specific metadata extraction
        match format {
            ModelFormat::SafeTensors => {
                metadata.tags.push("safetensors".to_string());
            }
            ModelFormat::ONNX => {
                metadata.tags.push("onnx".to_string());
            }
            _ => {}
        }

        Ok(metadata)
    }

    /// Validate SafeTensors file
    async fn validate_safetensors(&self, path: &Path, errors: &mut Vec<String>, warnings: &mut Vec<String>) -> Result<()> {
        // Basic SafeTensors validation
        let file = fs::File::open(path)
            .map_err(|e| VocoderError::ModelError(format!("Failed to open SafeTensors file: {}", e)))?;
        
        let file_size = file.metadata()
            .map_err(|e| VocoderError::ModelError(format!("Failed to read file metadata: {}", e)))?
            .len();

        if file_size < 8 {
            errors.push("SafeTensors file is too small".to_string());
            return Ok(());
        }

        // Check SafeTensors header
        let mut buffer = [0u8; 8];
        use std::io::Read;
        let mut file = file;
        file.read_exact(&mut buffer)
            .map_err(|e| VocoderError::ModelError(format!("Failed to read SafeTensors header: {}", e)))?;

        // SafeTensors starts with the length of the header as a little-endian u64
        let header_length = u64::from_le_bytes(buffer);
        if header_length > file_size || header_length == 0 {
            errors.push("Invalid SafeTensors header".to_string());
        } else if header_length > 1024 * 1024 {
            warnings.push("SafeTensors header is very large".to_string());
        }

        Ok(())
    }

    /// Validate ONNX file
    async fn validate_onnx(&self, _path: &Path, _errors: &mut Vec<String>, warnings: &mut Vec<String>) -> Result<()> {
        // Basic ONNX validation would go here
        warnings.push("ONNX validation not fully implemented".to_string());
        Ok(())
    }

    /// Download model from URL
    async fn download_model(&self, url: &str, output_path: &Path) -> Result<()> {
        // This is a simplified implementation - in practice, you'd want to use
        // a proper HTTP client with progress tracking, resumable downloads, etc.
        
        tracing::info!("Downloading {} to {}", url, output_path.display());
        
        // For now, return an error as we don't have a proper HTTP client
        Err(VocoderError::ModelError(
            "Model downloading not implemented in this example".to_string()
        ))
    }

    /// Convert URL to cache filename
    fn url_to_filename(&self, url: &str) -> String {
        // Simple implementation - hash the URL to create a filename
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        url.hash(&mut hasher);
        let hash = hasher.finish();
        
        // Try to extract a meaningful filename from the URL
        let path_part = url.split('/').last().unwrap_or("model");
        format!("{}_{:x}", path_part, hash)
    }

    /// Calculate file checksum
    async fn calculate_checksum(&self, path: &Path) -> Result<String> {
        let content = fs::read(path)
            .map_err(|e| VocoderError::ModelError(format!("Failed to read file for checksum: {}", e)))?;
        
        // Simple checksum using hash
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        Ok(format!("{:x}", hasher.finish()))
    }

    /// List cached models
    pub fn list_cached_models(&self) -> Vec<&ModelInfo> {
        self.models_cache.values().collect()
    }

    /// Clear model cache
    pub fn clear_cache(&mut self) {
        self.models_cache.clear();
    }

    /// Get model info from cache
    pub fn get_cached_model(&self, path: &str) -> Option<&ModelInfo> {
        self.models_cache.get(path)
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
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_model_format_detection() {
        assert_eq!(ModelFormat::from_path(Path::new("model.safetensors")), Some(ModelFormat::SafeTensors));
        assert_eq!(ModelFormat::from_path(Path::new("model.onnx")), Some(ModelFormat::ONNX));
        assert_eq!(ModelFormat::from_path(Path::new("model.pt")), Some(ModelFormat::PyTorch));
        assert_eq!(ModelFormat::from_path(Path::new("model.unknown")), None);
    }

    #[test]
    fn test_model_format_extensions() {
        assert!(ModelFormat::SafeTensors.extensions().contains(&"safetensors"));
        assert!(ModelFormat::ONNX.extensions().contains(&"onnx"));
        assert!(ModelFormat::PyTorch.extensions().contains(&"pt"));
    }

    #[tokio::test]
    async fn test_model_loader_creation() {
        let loader = ModelLoader::new();
        assert!(loader.cache_dir.is_none());
        assert!(loader.models_cache.is_empty());
    }

    #[tokio::test]
    async fn test_model_loader_with_cache() {
        let temp_dir = tempdir().unwrap();
        let loader = ModelLoader::with_cache_dir(temp_dir.path());
        assert!(loader.cache_dir.is_some());
    }

    #[tokio::test]
    async fn test_load_nonexistent_model() {
        let mut loader = ModelLoader::new();
        let result = loader.load_from_file("nonexistent.safetensors").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_validate_nonexistent_model() {
        let loader = ModelLoader::new();
        let result = loader.validate_model("nonexistent.safetensors").await.unwrap();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[tokio::test]
    async fn test_create_dummy_model_file() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path().join("test.safetensors");
        
        // Create a dummy SafeTensors file with valid header
        let mut file = fs::File::create(&model_path).unwrap();
        
        // Write SafeTensors header (8 bytes for header length + minimal header)
        let header = b"{}"; // Minimal JSON header
        let header_len = header.len() as u64;
        file.write_all(&header_len.to_le_bytes()).unwrap();
        file.write_all(header).unwrap();
        
        let mut loader = ModelLoader::new();
        let result = loader.load_from_file(&model_path).await;
        assert!(result.is_ok());
        
        let model_info = result.unwrap();
        assert_eq!(model_info.format, ModelFormat::SafeTensors);
        assert!(model_info.size_bytes > 0);
    }

    #[tokio::test]
    async fn test_url_to_filename() {
        let loader = ModelLoader::new();
        let filename1 = loader.url_to_filename("https://example.com/model.safetensors");
        let filename2 = loader.url_to_filename("https://example.com/model.safetensors");
        let filename3 = loader.url_to_filename("https://example.com/other.safetensors");
        
        // Same URL should produce same filename
        assert_eq!(filename1, filename2);
        // Different URL should produce different filename
        assert_ne!(filename1, filename3);
    }
}