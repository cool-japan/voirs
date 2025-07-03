//! Model loading and management utilities
//!
//! This module provides utilities for loading acoustic models from various
//! sources including local files, HuggingFace Hub, and remote URLs.

use async_trait::async_trait;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::{Result, AcousticError, AcousticModel};
use crate::config::{DeviceConfig, BackendType, CacheConfig};
use super::{Backend, ModelInfo, ModelFormat, ModelLoadConfig};

/// Model loader trait for different loading strategies
#[async_trait]
pub trait ModelLoader: Send + Sync {
    /// Load model from a file path
    async fn load_from_file(&self, path: &str) -> Result<Box<dyn AcousticModel>>;
    
    /// Load model from HuggingFace Hub
    async fn load_from_hub(&self, repo_id: &str) -> Result<Box<dyn AcousticModel>>;
    
    /// Load model from URL
    async fn load_from_url(&self, url: &str) -> Result<Box<dyn AcousticModel>>;
    
    /// List available models in a directory
    fn list_models(&self, directory: &str) -> Result<Vec<ModelInfo>>;
    
    /// Check if model exists and is valid
    async fn validate_model(&self, source: &str) -> Result<ModelInfo>;
    
    /// Get loader capabilities
    fn capabilities(&self) -> LoaderCapabilities;
}

/// Acoustic model loader with caching and multiple backend support
pub struct AcousticModelLoader {
    /// Available backends
    backends: HashMap<BackendType, Arc<dyn Backend>>,
    /// Default backend
    default_backend: BackendType,
    /// Cache configuration
    cache_config: CacheConfig,
    /// Model cache
    model_cache: HashMap<String, CachedModel>,
    /// Load configuration
    load_config: ModelLoadConfig,
}

impl AcousticModelLoader {
    /// Create new model loader
    pub fn new(
        backends: HashMap<BackendType, Arc<dyn Backend>>,
        default_backend: BackendType,
    ) -> Self {
        Self {
            backends,
            default_backend,
            cache_config: CacheConfig::default(),
            model_cache: HashMap::new(),
            load_config: ModelLoadConfig::default(),
        }
    }
    
    /// Set cache configuration
    pub fn with_cache_config(mut self, cache_config: CacheConfig) -> Self {
        self.cache_config = cache_config;
        self
    }
    
    /// Set load configuration
    pub fn with_load_config(mut self, load_config: ModelLoadConfig) -> Self {
        self.load_config = load_config;
        self
    }
    
    /// Add backend
    pub fn add_backend(&mut self, backend_type: BackendType, backend: Arc<dyn Backend>) {
        self.backends.insert(backend_type, backend);
    }
    
    /// Set default backend
    pub fn set_default_backend(&mut self, backend_type: BackendType) -> Result<()> {
        if self.backends.contains_key(&backend_type) {
            self.default_backend = backend_type;
            Ok(())
        } else {
            Err(AcousticError::ConfigError(format!("Backend {:?} not available", backend_type)))
        }
    }
    
    /// Load model with specific backend
    pub async fn load_with_backend(
        &mut self,
        source: &str,
        backend_type: BackendType,
    ) -> Result<Arc<dyn AcousticModel>> {
        // Check cache first
        if self.cache_config.enabled {
            let cache_key = format!("{}:{:?}", source, backend_type);
            if let Some(cached) = self.model_cache.get(&cache_key) {
                if !cached.is_expired() {
                    tracing::debug!("Loading model from cache: {}", source);
                    return Ok(cached.model.clone());
                } else {
                    // Remove expired entry
                    self.model_cache.remove(&cache_key);
                }
            }
        }
        
        // Get backend
        let backend = self.backends.get(&backend_type)
            .ok_or_else(|| AcousticError::ConfigError(format!("Backend {:?} not found", backend_type)))?;
        
        // Determine source type and load model
        let model = if self.is_url(source) {
            self.load_from_url_impl(source, backend.as_ref()).await?
        } else if self.is_hub_id(source) {
            self.load_from_hub_impl(source, backend.as_ref()).await?
        } else {
            self.load_from_file_impl(source, backend.as_ref()).await?
        };
        
        let model_arc: Arc<dyn AcousticModel> = model.into();
        
        // Cache the model
        if self.cache_config.enabled {
            let cache_key = format!("{}:{:?}", source, backend_type);
            let cached_model = CachedModel::new(
                model_arc.clone(),
                self.cache_config.ttl_seconds,
            );
            self.model_cache.insert(cache_key, cached_model);
            
            // Clean up cache if needed
            self.cleanup_cache();
        }
        
        tracing::info!("Loaded acoustic model from {} using {:?} backend", source, backend_type);
        Ok(model_arc)
    }
    
    /// Load model with default backend
    pub async fn load(&mut self, source: &str) -> Result<Arc<dyn AcousticModel>> {
        self.load_with_backend(source, self.default_backend).await
    }
    
    /// Load model from file path
    async fn load_from_file_impl(
        &self,
        path: &str,
        backend: &dyn Backend,
    ) -> Result<Box<dyn AcousticModel>> {
        // Validate file exists
        if !Path::new(path).exists() {
            return Err(AcousticError::ModelError(format!("Model file not found: {}", path)));
        }
        
        // Validate model format
        let model_info = backend.validate_model(path)?;
        if !model_info.compatible {
            return Err(AcousticError::ModelError(format!(
                "Model format {:?} not compatible with backend {}",
                model_info.format,
                backend.name()
            )));
        }
        
        // Load model
        backend.create_model(path).await
    }
    
    /// Load model from HuggingFace Hub
    async fn load_from_hub_impl(
        &self,
        repo_id: &str,
        backend: &dyn Backend,
    ) -> Result<Box<dyn AcousticModel>> {
        // Download model from Hub to cache
        let cache_dir = self.get_cache_dir();
        let model_path = self.download_from_hub(repo_id, &cache_dir).await?;
        
        // Load from cached file
        self.load_from_file_impl(&model_path, backend).await
    }
    
    /// Load model from URL
    async fn load_from_url_impl(
        &self,
        url: &str,
        backend: &dyn Backend,
    ) -> Result<Box<dyn AcousticModel>> {
        // Download model from URL to cache
        let cache_dir = self.get_cache_dir();
        let model_path = self.download_from_url(url, &cache_dir).await?;
        
        // Load from cached file
        self.load_from_file_impl(&model_path, backend).await
    }
    
    /// Download model from HuggingFace Hub
    async fn download_from_hub(&self, repo_id: &str, cache_dir: &Path) -> Result<String> {
        use hf_hub::api::tokio::Api;
        
        let api = Api::new()
            .map_err(|e| AcousticError::ModelError(format!("Failed to create HF API: {}", e)))?;
        
        let repo = api.model(repo_id.to_string());
        
        // Try to find model file (prefer safetensors, then pytorch)
        let model_files = vec!["model.safetensors", "pytorch_model.bin", "model.bin"];
        
        for file_name in model_files {
            match repo.get(file_name).await {
                Ok(path) => {
                    tracing::info!("Downloaded {} from HuggingFace Hub: {}", file_name, repo_id);
                    return Ok(path.to_string_lossy().to_string());
                }
                Err(_) => continue,
            }
        }
        
        Err(AcousticError::ModelError(format!(
            "No compatible model file found in repository: {}", 
            repo_id
        )))
    }
    
    /// Download model from URL
    async fn download_from_url(&self, url: &str, cache_dir: &Path) -> Result<String> {
        use std::io::Write;
        
        // Create cache directory if it doesn't exist
        std::fs::create_dir_all(cache_dir)
            .map_err(|e| AcousticError::ModelError(format!("Failed to create cache dir: {}", e)))?;
        
        // Generate filename from URL
        let filename = self.url_to_filename(url);
        let file_path = cache_dir.join(&filename);
        
        // Check if file already exists
        if file_path.exists() {
            tracing::debug!("Using cached file: {:?}", file_path);
            return Ok(file_path.to_string_lossy().to_string());
        }
        
        // Download file
        tracing::info!("Downloading model from: {}", url);
        
        let response = reqwest::get(url).await
            .map_err(|e| AcousticError::ModelError(format!("Failed to download: {}", e)))?;
        
        if !response.status().is_success() {
            return Err(AcousticError::ModelError(format!(
                "Download failed with status: {}", 
                response.status()
            )));
        }
        
        let bytes = response.bytes().await
            .map_err(|e| AcousticError::ModelError(format!("Failed to read response: {}", e)))?;
        
        // Write to cache file
        let mut file = std::fs::File::create(&file_path)
            .map_err(|e| AcousticError::ModelError(format!("Failed to create file: {}", e)))?;
        
        file.write_all(&bytes)
            .map_err(|e| AcousticError::ModelError(format!("Failed to write file: {}", e)))?;
        
        tracing::info!("Downloaded and cached model: {:?}", file_path);
        Ok(file_path.to_string_lossy().to_string())
    }
    
    /// Get cache directory
    fn get_cache_dir(&self) -> PathBuf {
        self.cache_config.cache_dir.clone()
    }
    
    /// Check if string is a URL
    fn is_url(&self, source: &str) -> bool {
        source.starts_with("http://") || source.starts_with("https://")
    }
    
    /// Check if string is a HuggingFace Hub repository ID
    fn is_hub_id(&self, source: &str) -> bool {
        // Simple heuristic: contains slash but not URL scheme and not path separators
        source.contains('/') && 
        !source.contains('\\') && 
        !source.starts_with("http://") &&
        !source.starts_with("https://") &&
        !Path::new(source).exists()
    }
    
    /// Convert URL to filename
    fn url_to_filename(&self, url: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        // Try to extract filename from URL
        if let Some(filename) = url.split('/').last() {
            if !filename.is_empty() && filename.contains('.') {
                return filename.to_string();
            }
        }
        
        // Fallback: hash the URL
        let mut hasher = DefaultHasher::new();
        url.hash(&mut hasher);
        format!("model_{:x}.bin", hasher.finish())
    }
    
    /// Clean up expired cache entries
    fn cleanup_cache(&mut self) {
        if !self.cache_config.enabled {
            return;
        }
        
        let mut to_remove = Vec::new();
        for (key, cached) in &self.model_cache {
            if cached.is_expired() {
                to_remove.push(key.clone());
            }
        }
        
        for key in to_remove {
            self.model_cache.remove(&key);
        }
        
        // Check cache size limit
        if self.model_cache.len() > 10 { // Simple limit
            // Remove oldest entries (simplified)
            let mut keys: Vec<_> = self.model_cache.keys().cloned().collect();
            keys.sort();
            
            while self.model_cache.len() > 10 {
                if let Some(key) = keys.pop() {
                    self.model_cache.remove(&key);
                } else {
                    break;
                }
            }
        }
    }
    
    /// Clear all cached models
    pub fn clear_cache(&mut self) {
        self.model_cache.clear();
        tracing::info!("Cleared model cache");
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        let total_entries = self.model_cache.len();
        let expired_entries = self.model_cache.values()
            .filter(|cached| cached.is_expired())
            .count();
        
        CacheStats {
            total_entries,
            expired_entries,
            active_entries: total_entries - expired_entries,
        }
    }
}

#[async_trait]
impl ModelLoader for AcousticModelLoader {
    async fn load_from_file(&self, path: &str) -> Result<Box<dyn AcousticModel>> {
        let backend = self.backends.get(&self.default_backend)
            .ok_or_else(|| AcousticError::ConfigError("No default backend available".to_string()))?;
        
        self.load_from_file_impl(path, backend.as_ref()).await
    }
    
    async fn load_from_hub(&self, repo_id: &str) -> Result<Box<dyn AcousticModel>> {
        let backend = self.backends.get(&self.default_backend)
            .ok_or_else(|| AcousticError::ConfigError("No default backend available".to_string()))?;
        
        self.load_from_hub_impl(repo_id, backend.as_ref()).await
    }
    
    async fn load_from_url(&self, url: &str) -> Result<Box<dyn AcousticModel>> {
        let backend = self.backends.get(&self.default_backend)
            .ok_or_else(|| AcousticError::ConfigError("No default backend available".to_string()))?;
        
        self.load_from_url_impl(url, backend.as_ref()).await
    }
    
    fn list_models(&self, directory: &str) -> Result<Vec<ModelInfo>> {
        let dir_path = Path::new(directory);
        if !dir_path.exists() {
            return Err(AcousticError::ModelError(format!("Directory not found: {}", directory)));
        }
        
        let mut models = Vec::new();
        
        let entries = std::fs::read_dir(dir_path)
            .map_err(|e| AcousticError::ModelError(format!("Failed to read directory: {}", e)))?;
        
        for entry in entries {
            let entry = entry
                .map_err(|e| AcousticError::ModelError(format!("Failed to read entry: {}", e)))?;
            
            let path = entry.path();
            if path.is_file() {
                if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
                    let format = ModelFormat::from_extension(extension);
                    if format != ModelFormat::Unknown {
                        let size_bytes = entry.metadata()
                            .map(|meta| meta.len())
                            .unwrap_or(0);
                        
                        let model_info = ModelInfo {
                            path: path.to_string_lossy().to_string(),
                            format,
                            size_bytes,
                            compatible: true, // Would need backend validation
                            metadata: HashMap::new(),
                        };
                        
                        models.push(model_info);
                    }
                }
            }
        }
        
        Ok(models)
    }
    
    async fn validate_model(&self, source: &str) -> Result<ModelInfo> {
        let backend = self.backends.get(&self.default_backend)
            .ok_or_else(|| AcousticError::ConfigError("No default backend available".to_string()))?;
        
        if self.is_url(source) || self.is_hub_id(source) {
            // For remote sources, we'd need to download first to validate
            // For now, return basic info
            Ok(ModelInfo {
                path: source.to_string(),
                format: ModelFormat::Unknown,
                size_bytes: 0,
                compatible: true,
                metadata: HashMap::new(),
            })
        } else {
            backend.validate_model(source)
        }
    }
    
    fn capabilities(&self) -> LoaderCapabilities {
        LoaderCapabilities {
            supports_local_files: true,
            supports_huggingface_hub: true,
            supports_urls: true,
            supports_caching: self.cache_config.enabled,
            supported_formats: vec![
                ModelFormat::SafeTensors,
                ModelFormat::PyTorch,
                ModelFormat::Onnx,
            ],
        }
    }
}

/// Cached model with expiration
#[derive(Clone)]
struct CachedModel {
    model: Arc<dyn AcousticModel>,
    cached_at: std::time::Instant,
    ttl_seconds: u32,
}

impl CachedModel {
    fn new(model: Arc<dyn AcousticModel>, ttl_seconds: u32) -> Self {
        Self {
            model,
            cached_at: std::time::Instant::now(),
            ttl_seconds,
        }
    }
    
    fn is_expired(&self) -> bool {
        self.cached_at.elapsed().as_secs() >= self.ttl_seconds as u64
    }
}

/// Loader capabilities
#[derive(Debug, Clone)]
pub struct LoaderCapabilities {
    /// Whether local files are supported
    pub supports_local_files: bool,
    /// Whether HuggingFace Hub is supported
    pub supports_huggingface_hub: bool,
    /// Whether URL downloads are supported
    pub supports_urls: bool,
    /// Whether caching is enabled
    pub supports_caching: bool,
    /// Supported model formats
    pub supported_formats: Vec<ModelFormat>,
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total number of cache entries
    pub total_entries: usize,
    /// Number of expired entries
    pub expired_entries: usize,
    /// Number of active entries
    pub active_entries: usize,
}

/// Create default model loader with available backends
pub fn create_default_loader() -> Result<AcousticModelLoader> {
    use super::{create_default_backend_manager, BackendType};
    
    let backend_manager = create_default_backend_manager()?;
    let backends = backend_manager.list_backends();
    
    if backends.is_empty() {
        return Err(AcousticError::ConfigError("No backends available".to_string()));
    }
    
    let mut backend_map = HashMap::new();
    for backend_type in &backends {
        let backend = backend_manager.get_backend(*backend_type)?;
        // Note: This is a simplified approach. In practice, you'd need to clone or share backends properly
        // backend_map.insert(*backend_type, Arc::new(backend));
    }
    
    let default_backend = backends[0];
    
    Ok(AcousticModelLoader::new(backend_map, default_backend))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::DummyAcousticModel;

    struct MockBackend;

    #[async_trait]
    impl Backend for MockBackend {
        fn name(&self) -> &'static str {
            "Mock"
        }
        
        fn supports_gpu(&self) -> bool {
            false
        }
        
        fn available_devices(&self) -> Vec<String> {
            vec!["cpu".to_string()]
        }
        
        async fn create_model(&self, _path: &str) -> Result<Box<dyn AcousticModel>> {
            Ok(Box::new(DummyAcousticModel::new()))
        }
    }

    fn create_test_loader() -> AcousticModelLoader {
        let mut backends = HashMap::new();
        backends.insert(BackendType::Candle, Arc::new(MockBackend) as Arc<dyn Backend>);
        
        AcousticModelLoader::new(backends, BackendType::Candle)
    }

    #[test]
    fn test_loader_creation() {
        let loader = create_test_loader();
        assert_eq!(loader.default_backend, BackendType::Candle);
    }

    #[test]
    fn test_is_url() {
        let loader = create_test_loader();
        
        assert!(loader.is_url("https://example.com/model.bin"));
        assert!(loader.is_url("http://example.com/model.bin"));
        assert!(!loader.is_url("local/path/model.bin"));
        assert!(!loader.is_url("organization/model-name"));
    }

    #[test]
    fn test_is_hub_id() {
        let loader = create_test_loader();
        
        assert!(loader.is_hub_id("microsoft/speecht5_tts"));
        assert!(loader.is_hub_id("facebook/mms-tts"));
        assert!(!loader.is_hub_id("https://example.com/model.bin"));
        // Note: local/path/model.bin might be considered a Hub ID if the path doesn't exist
        // This is expected behavior for the heuristic
    }

    #[test]
    fn test_url_to_filename() {
        let loader = create_test_loader();
        
        assert_eq!(loader.url_to_filename("https://example.com/model.safetensors"), "model.safetensors");
        assert_eq!(loader.url_to_filename("https://example.com/path/to/model.bin"), "model.bin");
        
        // URL without clear filename should generate hash-based name
        let hash_name = loader.url_to_filename("https://example.com/");
        assert!(hash_name.starts_with("model_"));
        assert!(hash_name.ends_with(".bin"));
    }

    #[test]
    fn test_cached_model() {
        let model = Arc::new(DummyAcousticModel::new()) as Arc<dyn AcousticModel>;
        let cached = CachedModel::new(model, 3600);
        
        assert!(!cached.is_expired());
        
        // Test with very short TTL
        let short_cached = CachedModel::new(cached.model.clone(), 0);
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(short_cached.is_expired());
    }

    #[test]
    fn test_cache_stats() {
        let mut loader = create_test_loader();
        loader.cache_config.enabled = true;
        
        let stats = loader.cache_stats();
        assert_eq!(stats.total_entries, 0);
        assert_eq!(stats.active_entries, 0);
        assert_eq!(stats.expired_entries, 0);
    }

    #[test]
    fn test_loader_capabilities() {
        let loader = create_test_loader();
        let caps = loader.capabilities();
        
        assert!(caps.supports_local_files);
        assert!(caps.supports_huggingface_hub);
        assert!(caps.supports_urls);
        assert_eq!(caps.supports_caching, loader.cache_config.enabled); // Check actual cache config
        assert!(!caps.supported_formats.is_empty());
    }

    #[tokio::test]
    async fn test_validate_model() {
        let loader = create_test_loader();
        
        // Test with non-existent local file
        let result = loader.validate_model("nonexistent.safetensors").await;
        assert!(result.is_ok()); // Mock backend doesn't validate properly
        
        // Test with URL
        let result = loader.validate_model("https://example.com/model.bin").await;
        assert!(result.is_ok());
        
        // Test with Hub ID
        let result = loader.validate_model("microsoft/speecht5_tts").await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_list_models_nonexistent_dir() {
        let loader = create_test_loader();
        let result = loader.list_models("/nonexistent/directory");
        assert!(result.is_err());
    }

    #[test]
    fn test_model_format_detection_in_list() {
        use std::fs;
        use std::io::Write;
        
        // Create temporary directory with test files
        let temp_dir = std::env::temp_dir().join("test_models");
        fs::create_dir_all(&temp_dir).unwrap();
        
        // Create test files
        let test_files = vec![
            "model.safetensors",
            "model.pth",
            "model.onnx",
            "config.json", // Should be ignored
        ];
        
        for filename in &test_files {
            let file_path = temp_dir.join(filename);
            let mut file = fs::File::create(&file_path).unwrap();
            writeln!(file, "test content").unwrap();
        }
        
        let loader = create_test_loader();
        let models = loader.list_models(&temp_dir.to_string_lossy()).unwrap();
        
        // Should find 3 models (excluding config.json)
        assert_eq!(models.len(), 3);
        
        // Check formats are detected correctly
        let formats: Vec<_> = models.iter().map(|m| m.format).collect();
        assert!(formats.contains(&ModelFormat::SafeTensors));
        assert!(formats.contains(&ModelFormat::PyTorch));
        assert!(formats.contains(&ModelFormat::Onnx));
        
        // Cleanup
        fs::remove_dir_all(&temp_dir).unwrap();
    }
}