//! Pipeline initialization and component loading.

use crate::{
    adapters::{G2pAdapter, VocoderAdapter},
    config::PipelineConfig,
    error::Result,
    traits::{AcousticModel, G2p, Vocoder},
    VoirsError,
};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
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

/// Caller-supplied component overrides.
///
/// Any component provided here is used verbatim and its model files are never
/// downloaded or loaded. This is what backs
/// [`with_g2p`](crate::builder::VoirsPipelineBuilder::with_g2p),
/// [`with_acoustic_model`](crate::builder::VoirsPipelineBuilder::with_acoustic_model)
/// and [`with_vocoder`](crate::builder::VoirsPipelineBuilder::with_vocoder).
#[derive(Default, Clone)]
pub struct ComponentOverrides {
    /// Pre-built G2P component
    pub g2p: Option<Arc<dyn G2p>>,
    /// Pre-built acoustic model
    pub acoustic: Option<Arc<dyn AcousticModel>>,
    /// Pre-built vocoder
    pub vocoder: Option<Arc<dyn Vocoder>>,
}

impl ComponentOverrides {
    /// Whether every component is supplied by the caller
    fn is_complete(&self) -> bool {
        self.g2p.is_some() && self.acoustic.is_some() && self.vocoder.is_some()
    }
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
        self.initialize_components_with(ComponentOverrides::default(), false)
            .await
    }

    /// Initialize pipeline components, honoring caller-supplied overrides.
    ///
    /// When `test_mode` is `true`, components that are not supplied by the caller
    /// are filled in with the in-process stub implementations
    /// ([`DummyG2p`](crate::pipeline::DummyG2p) and friends). Test mode must be
    /// requested explicitly; it is never inferred.
    ///
    /// When `test_mode` is `false`, missing components are loaded for real and any
    /// failure to obtain real model weights is reported as an error — no stub
    /// component is ever substituted silently.
    pub async fn initialize_components_with(
        &self,
        overrides: ComponentOverrides,
        test_mode: bool,
    ) -> Result<(Arc<dyn G2p>, Arc<dyn AcousticModel>, Arc<dyn Vocoder>)> {
        info!("Initializing pipeline components (test_mode={test_mode})");

        if test_mode {
            let g2p = overrides
                .g2p
                .unwrap_or_else(|| Arc::new(crate::pipeline::DummyG2p::new()));
            let acoustic = overrides
                .acoustic
                .unwrap_or_else(|| Arc::new(crate::pipeline::DummyAcoustic::new()));
            let vocoder = overrides
                .vocoder
                .unwrap_or_else(|| Arc::new(crate::pipeline::DummyVocoder::new()));
            return Ok((g2p, acoustic, vocoder));
        }

        // Validate configuration
        self.validate_configuration().await?;

        // Detect and setup device
        self.setup_device().await?;

        // Download and cache models if needed. Skipped entirely when the caller
        // supplied every component, since no model file would be consumed.
        if !overrides.is_complete() {
            self.download_models(&overrides).await?;
        }

        // Load components
        let g2p = match overrides.g2p {
            Some(g2p) => g2p,
            None => self.load_g2p().await?,
        };
        let acoustic = match overrides.acoustic {
            Some(acoustic) => acoustic,
            None => self.load_acoustic_model().await?,
        };
        let vocoder = match overrides.vocoder {
            Some(vocoder) => vocoder,
            None => self.load_vocoder().await?,
        };

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
            Err(VoirsError::DeviceNotAvailable {
                device: "metal".to_string(),
                alternatives: vec!["cpu".to_string(), "cuda".to_string()],
            })
        }

        #[cfg(target_os = "macos")]
        {
            // In real implementation, would initialize Metal context
            info!("Metal device initialized");
            Ok(())
        }
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
    async fn download_models(&self, overrides: &ComponentOverrides) -> Result<()> {
        info!("Checking and downloading models");

        let cache_dir = self.config.effective_cache_dir();

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
        let required_models = self.get_required_models(overrides);

        for model_info in required_models {
            let model_path = cache_dir.join(&model_info.filename);

            if !model_path.exists() {
                if self.config.model_loading.auto_download {
                    info!("Downloading model: {}", model_info.name);
                    self.download_model(&model_info, &model_path).await?;
                    if self.config.model_loading.verify_checksums {
                        self.verify_model_checksum(&model_path, &model_info.checksum)
                            .await?;
                    }
                } else {
                    return Err(VoirsError::ModelNotFound {
                        model_name: model_info.name,
                        path: model_path,
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

    /// Base URL used to resolve model download URLs
    fn download_base_url(&self) -> String {
        if let Ok(url) = std::env::var("VOIRS_DOWNLOAD_BASE_URL") {
            return url.trim_end_matches('/').to_string();
        }

        if let Some(url) = &self.config.model_loading.download_base_url {
            return url.trim_end_matches('/').to_string();
        }

        "https://huggingface.co/voirs/models/resolve/main".to_string()
    }

    /// Get list of required models based on configuration.
    ///
    /// Only models that the load path actually opens are listed: the rule-based
    /// G2P backend and the HiFi-GAN vocoder are constructed in-process and do not
    /// consume a downloaded file, so requiring them would fail-close on downloads
    /// nothing ever reads.
    fn get_required_models(&self, overrides: &ComponentOverrides) -> Vec<ModelInfo> {
        let mut models = Vec::new();

        if overrides.acoustic.is_some() {
            return models;
        }

        // The acoustic model is the one component loaded from a weights file.
        let language = self
            .config
            .language_code
            .unwrap_or(self.config.default_synthesis.language);
        let quality = &self.config.default_synthesis.quality;
        let acoustic_name = self.config.acoustic_model.as_deref().unwrap_or("candle");

        // A configured local path takes precedence and needs no download.
        if let Some(override_config) = self
            .config
            .model_loading
            .model_overrides
            .get(acoustic_name)
            .filter(|entry| entry.local_path.is_some())
        {
            if override_config
                .local_path
                .as_ref()
                .is_some_and(|path| path.exists())
            {
                return models;
            }
        }

        let base_url = self.download_base_url();
        let filename = format!("{language:?}-acoustic-{quality:?}.safetensors");
        let checksum = self
            .config
            .model_loading
            .model_overrides
            .get(acoustic_name)
            .and_then(|entry| entry.checksum.clone())
            .unwrap_or_default();
        let url = self
            .config
            .model_loading
            .model_overrides
            .get(acoustic_name)
            .and_then(|entry| entry.url.clone())
            .unwrap_or_else(|| format!("{base_url}/acoustic/{filename}"));

        models.push(ModelInfo {
            name: format!("{language:?}-acoustic-{quality:?}"),
            filename,
            url,
            checksum,
        });

        models
    }

    /// Download a single model file over HTTPS.
    ///
    /// The response body is streamed to a temporary file next to the target and
    /// atomically renamed on success, so a failed or truncated download never
    /// leaves a file that later looks like a valid cached model.
    async fn download_model(
        &self,
        model_info: &ModelInfo,
        target_path: &std::path::Path,
    ) -> Result<()> {
        info!("Downloading {} from {}", model_info.name, model_info.url);

        // Install the pure-Rust rustls CryptoProvider before any TLS handshake
        // (reqwest is built with `rustls-no-provider`). Once-guarded.
        crate::ensure_crypto_provider();

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(
                self.config.model_loading.download_timeout_secs,
            ))
            .connect_timeout(std::time::Duration::from_secs(30))
            .user_agent(concat!("VoiRS-SDK/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| VoirsError::DownloadFailed {
                url: model_info.url.clone(),
                reason: format!("Failed to create HTTP client: {e}"),
                bytes_downloaded: 0,
                total_bytes: None,
            })?;

        let max_retries = self.config.model_loading.download_retries;
        let mut last_error: Option<VoirsError> = None;

        for attempt in 0..=max_retries {
            if attempt > 0 {
                let delay = std::time::Duration::from_secs(2u64.pow(attempt.min(5)));
                tracing::debug!(
                    "Retrying model download in {:?} (attempt {}/{})",
                    delay,
                    attempt + 1,
                    max_retries + 1
                );
                tokio::time::sleep(delay).await;
            }

            match self
                .download_model_attempt(&client, &model_info.url, target_path)
                .await
            {
                Ok(bytes) => {
                    info!(
                        "Successfully downloaded {} ({} bytes)",
                        model_info.name, bytes
                    );
                    return Ok(());
                }
                Err(e) => {
                    tracing::warn!("Model download attempt {} failed: {}", attempt + 1, e);
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| VoirsError::DownloadFailed {
            url: model_info.url.clone(),
            reason: "Unknown download failure".to_string(),
            bytes_downloaded: 0,
            total_bytes: None,
        }))
    }

    /// Perform one download attempt, streaming the body to disk
    async fn download_model_attempt(
        &self,
        client: &reqwest::Client,
        url: &str,
        target_path: &std::path::Path,
    ) -> Result<u64> {
        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| VoirsError::DownloadFailed {
                url: url.to_string(),
                reason: format!("HTTP request failed: {e}"),
                bytes_downloaded: 0,
                total_bytes: None,
            })?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(VoirsError::DownloadFailed {
                url: url.to_string(),
                reason: format!(
                    "HTTP {} {}",
                    status.as_u16(),
                    status.canonical_reason().unwrap_or("Unknown")
                ),
                bytes_downloaded: 0,
                total_bytes: None,
            });
        }

        let total_bytes = response.content_length();
        let temp_path = target_path.with_extension("part");

        let mut file =
            tokio::fs::File::create(&temp_path)
                .await
                .map_err(|e| VoirsError::IoError {
                    path: temp_path.clone(),
                    operation: crate::error::types::IoOperation::Create,
                    source: e,
                })?;

        let mut bytes_downloaded = 0u64;
        let mut stream = response.bytes_stream();

        while let Some(chunk) = futures::StreamExt::next(&mut stream).await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(e) => {
                    let _ = tokio::fs::remove_file(&temp_path).await;
                    return Err(VoirsError::DownloadFailed {
                        url: url.to_string(),
                        reason: format!("Failed to read response chunk: {e}"),
                        bytes_downloaded,
                        total_bytes,
                    });
                }
            };

            file.write_all(&chunk)
                .await
                .map_err(|e| VoirsError::IoError {
                    path: temp_path.clone(),
                    operation: crate::error::types::IoOperation::Write,
                    source: e,
                })?;
            bytes_downloaded += chunk.len() as u64;
        }

        file.flush().await.map_err(|e| VoirsError::IoError {
            path: temp_path.clone(),
            operation: crate::error::types::IoOperation::Write,
            source: e,
        })?;
        drop(file);

        if let Some(expected) = total_bytes {
            if bytes_downloaded != expected {
                let _ = tokio::fs::remove_file(&temp_path).await;
                return Err(VoirsError::DownloadFailed {
                    url: url.to_string(),
                    reason: format!(
                        "Downloaded size ({bytes_downloaded} bytes) does not match Content-Length ({expected} bytes)"
                    ),
                    bytes_downloaded,
                    total_bytes,
                });
            }
        }

        tokio::fs::rename(&temp_path, target_path)
            .await
            .map_err(|e| VoirsError::IoError {
                path: target_path.to_path_buf(),
                operation: crate::error::types::IoOperation::Write,
                source: e,
            })?;

        Ok(bytes_downloaded)
    }

    /// Verify a model file's SHA-256 checksum against the expected value.
    ///
    /// A mismatching file is rejected with [`VoirsError::ModelError`]. When no
    /// expected checksum is configured the file is hashed anyway and the digest
    /// is logged, so the value can be pinned in configuration afterwards.
    async fn verify_model_checksum(
        &self,
        model_path: &std::path::Path,
        expected_checksum: &str,
    ) -> Result<()> {
        let actual = Self::file_sha256(model_path).await?;

        if expected_checksum.is_empty() {
            info!(
                "No expected checksum configured for {}; computed SHA-256 {}",
                model_path.display(),
                actual
            );
            return Ok(());
        }

        if !actual.eq_ignore_ascii_case(expected_checksum.trim()) {
            return Err(VoirsError::ModelError {
                model_type: crate::error::types::ModelType::Acoustic,
                message: format!(
                    "Checksum mismatch for {}: expected {}, computed {}",
                    model_path.display(),
                    expected_checksum.trim(),
                    actual
                ),
                source: None,
            });
        }

        info!("Checksum verified for: {}", model_path.display());
        Ok(())
    }

    /// Compute the SHA-256 digest of a file, streaming it in chunks
    async fn file_sha256(path: &std::path::Path) -> Result<String> {
        use tokio::io::AsyncReadExt;

        let mut file = tokio::fs::File::open(path)
            .await
            .map_err(|e| VoirsError::IoError {
                path: path.to_path_buf(),
                operation: crate::error::types::IoOperation::Read,
                source: e,
            })?;

        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 64 * 1024];

        loop {
            let read = file
                .read(&mut buffer)
                .await
                .map_err(|e| VoirsError::IoError {
                    path: path.to_path_buf(),
                    operation: crate::error::types::IoOperation::Read,
                    source: e,
                })?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }

        Ok(hex::encode(hasher.finalize()))
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
            model_name => Err(VoirsError::ModelError {
                model_type: crate::error::types::ModelType::G2p,
                message: format!(
                    "G2P backend '{model_name}' is not implemented. Supported backends: rule_based"
                ),
                source: None,
            }),
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
            model_name => Err(VoirsError::ModelError {
                model_type: crate::error::types::ModelType::Acoustic,
                message: format!(
                    "Acoustic backend '{model_name}' is not implemented. Supported backends: candle"
                ),
                source: None,
            }),
        }
    }

    /// Load vocoder component
    async fn load_vocoder(&self) -> Result<Arc<dyn Vocoder>> {
        info!("Loading vocoder component");

        // Load actual vocoder based on configuration
        match self.config.vocoder_model.as_deref().unwrap_or("hifigan") {
            "hifigan" => {
                info!("Loading HiFi-GAN vocoder");

                use voirs_vocoder::HiFiGanVocoder;

                // The vocoder must be built from real weights. `voirs-vocoder`
                // currently only exposes `initialize_inference_for_testing`, which
                // populates the generator with a freshly initialized (untrained)
                // VarMap; using that outside test mode would emit fabricated audio,
                // so the SDK fails closed instead.
                let weights_path = self.get_vocoder_model_path()?;

                let hifigan = HiFiGanVocoder::load_from_file(&weights_path).map_err(|e| {
                    VoirsError::ModelError {
                        model_type: crate::error::types::ModelType::Vocoder,
                        message: format!(
                            "Failed to load HiFi-GAN vocoder from {weights_path}: {e}"
                        ),
                        source: Some(Box::new(e)),
                    }
                })?;

                if !hifigan.is_initialized() {
                    return Err(VoirsError::ModelError {
                        model_type: crate::error::types::ModelType::Vocoder,
                        message: format!(
                            "HiFi-GAN weights at {weights_path} could not be bound to the inference \
                             graph: voirs-vocoder does not yet expose a real weight-loading entry \
                             point. Refusing to synthesize with untrained weights."
                        ),
                        source: None,
                    });
                }

                // Create trait adapter for the vocoder
                let adapter = VocoderAdapter::new(Arc::new(hifigan));
                Ok(Arc::new(adapter))
            }
            model_name => Err(VoirsError::ModelError {
                model_type: crate::error::types::ModelType::Vocoder,
                message: format!(
                    "Vocoder backend '{model_name}' is not implemented. Supported backends: hifigan"
                ),
                source: None,
            }),
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
            .unwrap_or(self.config.default_synthesis.language);
        let quality = &self.config.default_synthesis.quality;

        // Prefer .safetensors, fall back to .bin
        let mut model_path = None;
        for format in ["safetensors", "bin"] {
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

    /// Get the path to the vocoder weights based on configuration
    fn get_vocoder_model_path(&self) -> Result<String> {
        let vocoder_model_name = self.config.vocoder_model.as_deref().unwrap_or("hifigan");

        if let Some(override_config) = self
            .config
            .model_loading
            .model_overrides
            .get(vocoder_model_name)
        {
            if let Some(local_path) = &override_config.local_path {
                if local_path.exists() {
                    return Ok(local_path.to_string_lossy().to_string());
                }
            }
        }

        let cache_dir = self.config.effective_cache_dir();
        let language = self
            .config
            .language_code
            .unwrap_or(self.config.default_synthesis.language);
        let quality = &self.config.default_synthesis.quality;

        for format in ["safetensors", "bin"] {
            let candidate = cache_dir.join(format!("{language:?}-vocoder-{quality:?}.{format}"));
            if candidate.exists() {
                return Ok(candidate.to_string_lossy().to_string());
            }
        }

        Err(VoirsError::ModelError {
            model_type: crate::error::types::ModelType::Vocoder,
            message: format!(
                "Vocoder model not found. Searched for {language:?}-vocoder-{quality:?}.{{safetensors,bin}} in {}",
                cache_dir.display()
            ),
            source: None,
        })
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

    fn cpu_config(cache_dir: &std::path::Path) -> PipelineConfig {
        PipelineConfig {
            device: "cpu".to_string(),
            use_gpu: false,
            cache_dir: Some(cache_dir.to_path_buf()),
            ..Default::default()
        }
    }

    /// The SHA-256 helper must compute the real digest of the file contents.
    #[tokio::test]
    async fn test_file_sha256_is_real() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("payload.bin");
        tokio::fs::write(&path, b"voirs").await.expect("write");

        let digest = PipelineInitializer::file_sha256(&path)
            .await
            .expect("digest");

        // SHA-256("voirs")
        let mut hasher = Sha256::new();
        hasher.update(b"voirs");
        assert_eq!(digest, hex::encode(hasher.finalize()));
        assert_eq!(digest.len(), 64);

        // A different file must produce a different digest.
        let other = dir.path().join("other.bin");
        tokio::fs::write(&other, b"voirs!").await.expect("write");
        let other_digest = PipelineInitializer::file_sha256(&other)
            .await
            .expect("digest");
        assert_ne!(digest, other_digest);
    }

    /// A checksum mismatch must be rejected, and a match accepted.
    #[tokio::test]
    async fn test_verify_model_checksum_compares_real_hash() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("model.safetensors");
        tokio::fs::write(&path, b"weights").await.expect("write");

        let initializer = PipelineInitializer::new(cpu_config(dir.path()));

        let mut hasher = Sha256::new();
        hasher.update(b"weights");
        let expected = hex::encode(hasher.finalize());

        initializer
            .verify_model_checksum(&path, &expected)
            .await
            .expect("matching checksum must verify");

        let wrong = "0".repeat(64);
        let err = initializer
            .verify_model_checksum(&path, &wrong)
            .await
            .expect_err("mismatching checksum must fail");
        assert!(
            format!("{err}").contains("Checksum mismatch"),
            "unexpected error: {err}"
        );
    }

    /// With auto-download disabled and no weights on disk, initialization must
    /// fail closed instead of fabricating a model file.
    #[tokio::test]
    async fn test_missing_models_fail_closed_and_write_nothing() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut config = cpu_config(dir.path());
        config.model_loading.auto_download = false;

        let initializer = PipelineInitializer::new(config);
        let result = initializer
            .initialize_components_with(ComponentOverrides::default(), false)
            .await;

        assert!(result.is_err(), "missing weights must not build a pipeline");

        // No placeholder model file may have been created.
        let entries: Vec<_> = std::fs::read_dir(dir.path())
            .expect("read dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert!(
            entries.is_empty(),
            "no files may be fabricated in the cache dir, found {entries:?}"
        );
    }

    /// Only the model the load path actually reads is required.
    #[test]
    fn test_required_models_lists_only_consumed_weights() {
        let dir = tempfile::tempdir().expect("temp dir");
        let initializer = PipelineInitializer::new(cpu_config(dir.path()));

        let models = initializer.get_required_models(&ComponentOverrides::default());
        assert_eq!(models.len(), 1);
        assert!(models[0].filename.contains("acoustic"));
        assert!(models[0].url.starts_with("https://"));

        // A caller-supplied acoustic model needs no download at all.
        let overrides = ComponentOverrides {
            acoustic: Some(Arc::new(crate::pipeline::DummyAcoustic::new())),
            ..Default::default()
        };
        assert!(initializer.get_required_models(&overrides).is_empty());
    }

    /// Test mode must be an explicit opt-in that yields the documented stubs.
    #[tokio::test]
    async fn test_mode_uses_stub_components() {
        let dir = tempfile::tempdir().expect("temp dir");
        let initializer = PipelineInitializer::new(cpu_config(dir.path()));

        let (g2p, acoustic, vocoder) = initializer
            .initialize_components_with(ComponentOverrides::default(), true)
            .await
            .expect("stub components");

        assert_eq!(g2p.metadata().name, "DummyG2p");
        assert_eq!(acoustic.metadata().name, "DummyAcoustic");
        assert_eq!(vocoder.metadata().name, "DummyVocoder");
    }
}
