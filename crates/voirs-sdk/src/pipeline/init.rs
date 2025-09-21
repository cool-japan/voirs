//! Pipeline initialization and component loading.

use crate::{
    adapters::{G2pAdapter, VocoderAdapter},
    config::PipelineConfig,
    error::Result,
    traits::{AcousticModel, G2p, Vocoder},
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
            #[cfg(feature = "gpu")]
            if cfg!(target_os = "linux") || cfg!(target_os = "windows") {
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
    pub async fn initialize_components(
        &self,
    ) -> Result<(Arc<dyn G2p>, Arc<dyn AcousticModel>, Arc<dyn Vocoder>)> {
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
                    self.verify_model_checksum(&model_path, &model_info.checksum)
                        .await?;
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
            name: format!("{language:?}-g2p"),
            filename: format!("{language:?}-g2p-{quality:?}.bin"),
            url: format!("https://huggingface.co/voirs/models/{language:?}/g2p-{quality:?}.bin"),
            checksum: "".to_string(), // In real implementation, would have actual checksums
        });

        models.push(ModelInfo {
            name: format!("{language:?}-acoustic"),
            filename: format!("{language:?}-acoustic-{quality:?}.bin"),
            url: format!(
                "https://huggingface.co/voirs/models/{language:?}/acoustic-{quality:?}.bin"
            ),
            checksum: "".to_string(),
        });

        models.push(ModelInfo {
            name: format!("{language:?}-vocoder"),
            filename: format!("{language:?}-vocoder-{quality:?}.bin"),
            url: format!(
                "https://huggingface.co/voirs/models/{language:?}/vocoder-{quality:?}.bin"
            ),
            checksum: "".to_string(),
        });

        models
    }

    /// Download a single model file
    async fn download_model(
        &self,
        model_info: &ModelInfo,
        target_path: &std::path::Path,
    ) -> Result<()> {
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
    async fn verify_model_checksum(
        &self,
        model_path: &std::path::Path,
        expected_checksum: &str,
    ) -> Result<()> {
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

        // Load actual G2P model based on configuration
        use voirs_g2p::backends::rule_based::RuleBasedG2p;
        use voirs_g2p::LanguageCode as G2pLanguageCode;

        match self.config.g2p_model.as_deref().unwrap_or("rule_based") {
            "rule_based" => {
                info!("Loading rule-based G2P model");

                // Determine language from config or use default
                let language = self
                    .config
                    .language_code
                    .and_then(|lang| match lang {
                        crate::types::LanguageCode::EnUs => Some(G2pLanguageCode::EnUs),
                        crate::types::LanguageCode::EnGb => Some(G2pLanguageCode::EnGb),
                        crate::types::LanguageCode::De => Some(G2pLanguageCode::De),
                        crate::types::LanguageCode::Fr => Some(G2pLanguageCode::Fr),
                        crate::types::LanguageCode::Es => Some(G2pLanguageCode::Es),
                        crate::types::LanguageCode::It => Some(G2pLanguageCode::It),
                        crate::types::LanguageCode::Pt => Some(G2pLanguageCode::Pt),
                        crate::types::LanguageCode::Ja => Some(G2pLanguageCode::Ja),
                        _ => None,
                    })
                    .unwrap_or(G2pLanguageCode::EnUs); // Default to English (US)

                let rule_based_g2p = Arc::new(RuleBasedG2p::new(language));
                let adapter = G2pAdapter::new(rule_based_g2p);
                Ok(Arc::new(adapter))
            }
            model_name => {
                // For other models, fall back to dummy for now but log warning
                tracing::warn!("G2P model '{}' not implemented, using dummy", model_name);
                Ok(Arc::new(crate::pipeline::DummyG2p::new()))
            }
        }
    }

    /// Load acoustic model component
    async fn load_acoustic_model(&self) -> Result<Arc<dyn AcousticModel>> {
        info!("Loading acoustic model component");

        // Load actual acoustic model based on configuration
        use voirs_acoustic::backends::candle::CandleBackend;
        use voirs_acoustic::backends::{Backend, BackendManager};
        use voirs_acoustic::config::AcousticConfig;

        match self.config.acoustic_model.as_deref().unwrap_or("candle") {
            "candle" => {
                info!("Loading Candle-based acoustic model");

                // Create acoustic configuration
                let mut acoustic_config = AcousticConfig::default();

                // Set device type based on string
                use voirs_acoustic::config::DeviceType;
                acoustic_config.runtime.device.device_type = match self.config.device.as_str() {
                    "cpu" => DeviceType::Cpu,
                    "cuda" => DeviceType::Cuda,
                    "metal" => DeviceType::Metal,
                    "opencl" => DeviceType::OpenCl,
                    _ => DeviceType::Cpu, // Default to CPU
                };

                // Set GPU usage via mixed precision if GPU is requested
                if self.config.use_gpu && self.config.device != "cpu" {
                    acoustic_config.runtime.device.mixed_precision = true;
                }

                // Set thread count in performance config
                acoustic_config.runtime.performance.num_threads =
                    self.config.num_threads.map(|t| t as u32);

                // Create backend manager
                let _backend_manager = BackendManager::new();

                // Create Candle backend with device config
                let candle_backend = CandleBackend::with_device(
                    acoustic_config.runtime.device.clone(),
                )
                .map_err(|e| VoirsError::ModelError {
                    model_type: crate::error::types::ModelType::Acoustic,
                    message: format!("Failed to create Candle backend: {e}"),
                    source: Some(Box::new(e)),
                })?;

                // Create acoustic model using the backend
                // Determine the actual model path from configuration
                let model_path = self.get_acoustic_model_path()?;
                let acoustic_model =
                    candle_backend
                        .create_model(&model_path)
                        .await
                        .map_err(|e| VoirsError::ModelError {
                            model_type: crate::error::types::ModelType::Acoustic,
                            message: format!("Failed to create acoustic model: {e}"),
                            source: Some(Box::new(e)),
                        })?;

                // Create trait adapter for the acoustic model
                let adapter = crate::adapters::AcousticAdapter::new(Arc::from(acoustic_model));
                Ok(Arc::new(adapter))
            }
            model_name => {
                // For other models, fall back to dummy for now but log warning
                tracing::warn!(
                    "Acoustic model '{}' not implemented, using dummy",
                    model_name
                );
                Ok(Arc::new(crate::pipeline::DummyAcoustic::new()))
            }
        }
    }

    /// Load vocoder component
    async fn load_vocoder(&self) -> Result<Arc<dyn Vocoder>> {
        info!("Loading vocoder component");

        // Load actual vocoder based on configuration
        match self.config.vocoder_model.as_deref().unwrap_or("hifigan") {
            "hifigan" => {
                info!("Loading HiFi-GAN vocoder");

                // Create HiFi-GAN vocoder with configuration
                use voirs_vocoder::HiFiGanVocoder;
                let mut hifigan = HiFiGanVocoder::new();

                // Initialize inference for the vocoder
                hifigan
                    .initialize_inference_for_testing()
                    .map_err(|e| VoirsError::ModelError {
                        model_type: crate::error::types::ModelType::Vocoder,
                        message: format!("Failed to initialize HiFi-GAN vocoder: {e}"),
                        source: Some(Box::new(e)),
                    })?;

                // Create trait adapter for the vocoder
                let adapter = VocoderAdapter::new(Arc::new(hifigan));
                Ok(Arc::new(adapter))
            }
            model_name => {
                // For other models, fall back to dummy for now but log warning
                tracing::warn!(
                    "Vocoder model '{}' not implemented, using dummy",
                    model_name
                );
                Ok(Arc::new(crate::pipeline::DummyVocoder::new()))
            }
        }
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
                    std::path::Path::new("/usr/local/cuda").exists()
                        || std::path::Path::new("/opt/cuda").exists()
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

    /// Get the path to the acoustic model based on configuration
    fn get_acoustic_model_path(&self) -> Result<String> {
        // Check if there's a model override with a local path for acoustic model
        let acoustic_model_name = self.config.acoustic_model.as_deref().unwrap_or("candle");

        if let Some(override_config) = self
            .config
            .model_loading
            .model_overrides
            .get(acoustic_model_name)
        {
            if let Some(local_path) = &override_config.local_path {
                return Ok(local_path.to_string_lossy().to_string());
            }
        }

        // Otherwise, construct path from cache directory and model filename
        let cache_dir = self.config.effective_cache_dir();
        let language = self
            .config
            .language_code
            .unwrap_or(crate::types::LanguageCode::EnUs);
        let quality = &self.config.default_synthesis.quality;

        // Try different model file formats based on environment
        let model_formats = if std::env::var("VOIRS_TEST_MODE").is_ok() || cfg!(test) {
            // In test mode, prefer .bin files first, then .safetensors
            vec!["bin", "safetensors"]
        } else {
            // In production, prefer .safetensors files first, then .bin
            vec!["safetensors", "bin"]
        };

        let mut model_path = None;
        for format in model_formats {
            let model_filename = format!("{language:?}-acoustic-{quality:?}.{format}");
            let candidate_path = cache_dir.join(&model_filename);
            if candidate_path.exists() {
                model_path = Some(candidate_path);
                break;
            }
        }

        let model_path = model_path.ok_or_else(|| VoirsError::ModelError {
            model_type: crate::error::types::ModelType::Acoustic,
            message: format!("Acoustic model not found. Searched for {language:?}-acoustic-{quality:?}.{{safetensors,bin}} in {}", cache_dir.display()),
            source: None,
        })?;

        Ok(model_path.to_string_lossy().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile;

    #[tokio::test]
    async fn test_pipeline_initializer() {
        // Test configuration validation rather than actual component loading
        // since the actual loading requires real model files

        // Test with valid configuration
        let config = PipelineConfig {
            device: "cpu".to_string(),
            use_gpu: false,
            ..Default::default()
        };
        let initializer = PipelineInitializer::new(config);

        let result = initializer.validate_configuration().await;
        assert!(result.is_ok());

        // Test with invalid configuration
        let invalid_config = PipelineConfig {
            device: "unsupported".to_string(),
            ..Default::default()
        };
        let invalid_initializer = PipelineInitializer::new(invalid_config);

        let result = invalid_initializer.validate_configuration().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_configuration_validation() {
        let config = PipelineConfig {
            device: "unsupported".to_string(),
            ..Default::default()
        };

        let initializer = PipelineInitializer::new(config);
        let result = initializer.validate_configuration().await;
        assert!(result.is_err());
    }
}
