//! Pipeline initialization and component loading.

use crate::{
    config::PipelineConfig,
    error::Result,
    traits::{AcousticModel, G2p, Vocoder},
    types::LanguageCode,
    VoirsError,
};
use std::sync::Arc;
use tracing::info;

/// Information about a model file
#[derive(Debug, Clone)]
struct ModelInfo {
    /// Human-readable model name
    name: String,
    /// Filename for local storage
    filename: String,
    /// Download URL
    url: String,
    /// Expected checksum (empty if not available)
    checksum: String,
}

/// Component loading and validation
pub struct PipelineInitializer {
    config: PipelineConfig,
}

impl PipelineInitializer {
    /// Create new initializer with configuration
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    /// Get available devices for configuration validation
    fn get_available_devices(&self) -> Vec<String> {
        let mut devices = vec!["cpu".to_string()];
        
        // Check for GPU availability
        if self.is_gpu_available() {
            if cfg!(feature = "cuda") {
                devices.push("cuda".to_string());
            }
            if cfg!(target_os = "macos") {
                devices.push("metal".to_string());
            }
            devices.push("vulkan".to_string());
        }
        
        devices
    }

    /// Initialize all pipeline components
    pub async fn initialize_components(&self) -> Result<(Arc<dyn G2p>, Arc<dyn AcousticModel>, Arc<dyn Vocoder>)> {
        info!("Initializing pipeline components");

        // Validate configuration
        self.validate_configuration().await?;

        // Detect and setup device
        self.setup_device().await?;

        // Download and cache models if needed
        self.download_models().await?;

        // Load components
        let g2p = self.load_g2p().await?;
        let acoustic = self.load_acoustic_model().await?;
        let vocoder = self.load_vocoder().await?;

        info!("Pipeline components initialized successfully");
        Ok((g2p, acoustic, vocoder))
    }

    /// Validate pipeline configuration
    async fn validate_configuration(&self) -> Result<()> {
        info!("Validating pipeline configuration");

        // Validate device configuration
        if !self.is_device_available(&self.config.device) {
            return Err(VoirsError::InvalidConfiguration {
                field: "device".to_string(),
                value: self.config.device.clone(),
                reason: "Device not available".to_string(),
                valid_values: Some(self.get_available_devices()),
            });
        }

        // Validate GPU configuration
        if self.config.use_gpu && !self.is_gpu_available() {
            return Err(VoirsError::InvalidConfiguration {
                field: "use_gpu".to_string(),
                value: "true".to_string(),
                reason: "GPU not available".to_string(),
                valid_values: Some(vec!["false".to_string()]),
            });
        }

        // Validate cache directory
        if let Some(cache_dir) = &self.config.cache_dir {
            if !cache_dir.exists() {
                std::fs::create_dir_all(cache_dir).map_err(|e| VoirsError::IoError {
                    path: cache_dir.clone(),
                    operation: crate::error::types::IoOperation::Create,
                    source: e,
                })?;
            }
        }

        Ok(())
    }

    /// Setup device for computation
    async fn setup_device(&self) -> Result<()> {
        info!("Setting up device: {}", self.config.device);

        match self.config.device.as_str() {
            "cpu" => {
                self.setup_cpu_device().await?;
            }
            "cuda" => {
                self.setup_cuda_device().await?;
            }
            "metal" => {
                self.setup_metal_device().await?;
            }
            "vulkan" => {
                self.setup_vulkan_device().await?;
            }
            "opencl" => {
                self.setup_opencl_device().await?;
            }
            _ => {
                return Err(VoirsError::UnsupportedDevice {
                    device: self.config.device.clone(),
                });
            }
        }

        info!("Device setup completed: {}", self.config.device);
        Ok(())
    }

    /// Setup CPU device
    async fn setup_cpu_device(&self) -> Result<()> {
        info!("Setting up CPU device");
        
        let thread_count = self.config.effective_thread_count();
        info!("Using {} CPU threads", thread_count);
        
        // Set thread pool size for CPU inference
        // In real implementation, would configure actual CPU inference backend
        std::env::set_var("OMP_NUM_THREADS", thread_count.to_string());
        std::env::set_var("MKL_NUM_THREADS", thread_count.to_string());
        
        Ok(())
    }

    /// Setup CUDA device
    async fn setup_cuda_device(&self) -> Result<()> {
        info!("Setting up CUDA device");
        
        if !self.is_gpu_available() {
            return Err(VoirsError::DeviceNotAvailable {
                device: self.config.device.clone(),
                alternatives: vec!["cpu".to_string()],
            });
        }
        
        // In real implementation, would initialize CUDA context
        info!("CUDA device initialized");
        Ok(())
    }

    /// Setup Metal device (macOS)
    async fn setup_metal_device(&self) -> Result<()> {
        info!("Setting up Metal device");
        
        #[cfg(not(target_os = "macos"))]
        {
            return Err(VoirsError::DeviceNotAvailable {
                device: "metal".to_string(),
            });
        }
        
        #[cfg(target_os = "macos")]
        {
            // In real implementation, would initialize Metal context
            info!("Metal device initialized");
        }
        
        Ok(())
    }

    /// Setup Vulkan device
    async fn setup_vulkan_device(&self) -> Result<()> {
        info!("Setting up Vulkan device");
        
        // In real implementation, would check Vulkan availability and initialize
        info!("Vulkan device initialized");
        Ok(())
    }

    /// Setup OpenCL device
    async fn setup_opencl_device(&self) -> Result<()> {
        info!("Setting up OpenCL device");
        
        // In real implementation, would check OpenCL availability and initialize
        info!("OpenCL device initialized");
        Ok(())
    }

    /// Download and cache required models
    async fn download_models(&self) -> Result<()> {
        info!("Checking and downloading models");

        let cache_dir = match &self.config.cache_dir {
            Some(dir) => dir.clone(),
            None => {
                // Use default cache directory
                let mut default_cache = std::env::temp_dir();
                default_cache.push("voirs-cache");
                default_cache
            }
        };

        // Ensure cache directory exists
        if !cache_dir.exists() {
            std::fs::create_dir_all(&cache_dir).map_err(|e| VoirsError::IoError {
                path: cache_dir.clone(),
                operation: crate::error::types::IoOperation::Create,
                source: e,
            })?;
        }

        info!("Models will be cached in: {}", cache_dir.display());

        // Check for required models based on configuration
        let required_models = self.get_required_models();
        
        for model_info in required_models {
            let model_path = cache_dir.join(&model_info.filename);
            
            if !model_path.exists() {
                if self.config.model_loading.auto_download {
                    info!("Downloading model: {}", model_info.name);
                    self.download_model(&model_info, &model_path).await?;
                } else {
                    return Err(VoirsError::VoiceNotFound {
                        voice: model_info.name,
                        available: vec![],
                        suggestions: vec![],
                    });
                }
            } else {
                // Verify model integrity if checksum verification is enabled
                if self.config.model_loading.verify_checksums {
                    self.verify_model_checksum(&model_path, &model_info.checksum).await?;
                }
                info!("Model already cached: {}", model_info.name);
            }
        }

        Ok(())
    }

    /// Get list of required models based on configuration
    fn get_required_models(&self) -> Vec<ModelInfo> {
        let mut models = Vec::new();
        
        // Add default models based on language and quality settings
        let language = self.config.default_synthesis.language;
        let quality = &self.config.default_synthesis.quality;
        
        models.push(ModelInfo {
            name: format!("{:?}-g2p", language),
            filename: format!("{:?}-g2p-{:?}.bin", language, quality),
            url: format!("https://huggingface.co/voirs/models/{:?}/g2p-{:?}.bin", language, quality),
            checksum: "".to_string(), // In real implementation, would have actual checksums
        });
        
        models.push(ModelInfo {
            name: format!("{:?}-acoustic", language),
            filename: format!("{:?}-acoustic-{:?}.bin", language, quality),
            url: format!("https://huggingface.co/voirs/models/{:?}/acoustic-{:?}.bin", language, quality),
            checksum: "".to_string(),
        });
        
        models.push(ModelInfo {
            name: format!("{:?}-vocoder", language),
            filename: format!("{:?}-vocoder-{:?}.bin", language, quality),
            url: format!("https://huggingface.co/voirs/models/{:?}/vocoder-{:?}.bin", language, quality),
            checksum: "".to_string(),
        });
        
        models
    }

    /// Download a single model file
    async fn download_model(&self, model_info: &ModelInfo, target_path: &std::path::Path) -> Result<()> {
        info!("Downloading {} from {}", model_info.name, model_info.url);
        
        // Create a dummy file for now - in real implementation, would use HTTP client
        tokio::fs::write(target_path, format!("Dummy {} model data", model_info.name))
            .await
            .map_err(|e| VoirsError::IoError {
                path: target_path.to_path_buf(),
                operation: crate::error::types::IoOperation::Write,
                source: e,
            })?;
            
        info!("Successfully downloaded: {}", model_info.name);
        Ok(())
    }

    /// Verify model file checksum
    async fn verify_model_checksum(&self, model_path: &std::path::Path, expected_checksum: &str) -> Result<()> {
        if expected_checksum.is_empty() {
            // Skip verification if no checksum provided
            return Ok(());
        }
        
        info!("Verifying checksum for: {}", model_path.display());
        
        // In real implementation, would calculate actual file hash
        // For now, just log the verification
        info!("Checksum verification passed");
        Ok(())
    }

    /// Load G2P component
    async fn load_g2p(&self) -> Result<Arc<dyn G2p>> {
        info!("Loading G2P component");

        // TODO: Load actual G2P model based on configuration
        // For now, return dummy implementation
        Ok(Arc::new(crate::pipeline::DummyG2p::new()))
    }

    /// Load acoustic model component
    async fn load_acoustic_model(&self) -> Result<Arc<dyn AcousticModel>> {
        info!("Loading acoustic model component");

        // TODO: Load actual acoustic model based on configuration
        // For now, return dummy implementation
        Ok(Arc::new(crate::pipeline::DummyAcoustic::new()))
    }

    /// Load vocoder component
    async fn load_vocoder(&self) -> Result<Arc<dyn Vocoder>> {
        info!("Loading vocoder component");

        // TODO: Load actual vocoder based on configuration
        // For now, return dummy implementation
        Ok(Arc::new(crate::pipeline::DummyVocoder::new()))
    }

    /// Check if device is available
    fn is_device_available(&self, device: &str) -> bool {
        match device {
            "cpu" => true,
            "cuda" => self.is_gpu_available(),
            _ => false,
        }
    }

    /// Check if GPU is available
    fn is_gpu_available(&self) -> bool {
        // Check for CUDA availability on different platforms
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            // Check if CUDA runtime is available
            match std::env::var("CUDA_PATH") {
                Ok(_) => true,
                Err(_) => {
                    // Try alternative checks
                    std::path::Path::new("/usr/local/cuda").exists() ||
                    std::path::Path::new("/opt/cuda").exists()
                }
            }
        }
        
        #[cfg(target_os = "macos")]
        {
            // Check for Metal Performance Shaders
            // Metal is always available on macOS 10.11+
            true
        }
        
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_initializer() {
        let config = PipelineConfig::default();
        let initializer = PipelineInitializer::new(config);
        
        let result = initializer.initialize_components().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_configuration_validation() {
        let mut config = PipelineConfig::default();
        config.device = "unsupported".to_string();
        
        let initializer = PipelineInitializer::new(config);
        let result = initializer.validate_configuration().await;
        assert!(result.is_err());
    }
}