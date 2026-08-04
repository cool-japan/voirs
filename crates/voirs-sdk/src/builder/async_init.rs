//! Async initialization logic for pipeline builder.

use crate::{
    error::Result,
    pipeline::VoirsPipeline,
    traits::{AcousticModel, G2p, Vocoder, VoiceManager},
    voice::DefaultVoiceManager,
    VoirsError,
};
use reqwest;
use std::sync::Arc;
use std::time::Instant;
use tokio::io::AsyncWriteExt;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use super::builder_impl::VoirsPipelineBuilder;

impl VoirsPipelineBuilder {
    /// Build the pipeline with async initialization.
    ///
    /// Components are resolved through the single real initialization path
    /// ([`crate::pipeline::init::PipelineInitializer`]); the advanced-feature
    /// controllers configured on the builder are constructed here and installed on
    /// the returned pipeline.
    pub async fn build(self) -> Result<VoirsPipeline> {
        let start_time = Instant::now();
        info!(
            "Building VoiRS pipeline with configuration: {:?}",
            self.config
        );

        // Validate configuration first
        self.validate().await?;

        // Initialize components
        let pipeline = self.initialize_pipeline().await?;

        let build_time = start_time.elapsed();
        info!(
            "VoiRS pipeline built successfully in {:.2}s",
            build_time.as_secs_f64()
        );

        Ok(pipeline)
    }

    /// Initialize the complete pipeline
    async fn initialize_pipeline(mut self) -> Result<VoirsPipeline> {
        // Initialize cache directory
        self.setup_cache_directory().await?;

        // Create or use provided voice manager and reuse it for the built pipeline
        // so that `set_voice` resolves against the same registry.
        let voice_manager = self.setup_voice_manager().await?;
        self.voice_manager = Some(voice_manager);

        // Download voice assets before component initialization if requested
        if let Some(voice_id) = self.voice_id.clone() {
            if self.auto_download && !self.test_mode {
                self.ensure_voice_available(&voice_id).await?;
            } else if self.test_mode {
                debug!("Skipping voice download in test mode");
            }
        }

        // Resolve components and build the core pipeline through the real path
        let inner = crate::pipeline::pipeline_impl::VoirsPipeline::from_builder_core(&self).await?;

        // Validate the components that were actually installed
        self.validate_g2p_component(&inner.g2p().await).await?;
        self.validate_acoustic_component(&inner.acoustic().await)
            .await?;
        self.validate_vocoder_component(&inner.vocoder().await)
            .await?;

        // Build the advanced-feature controllers from the configuration supplied
        // through the builder, so `.with_emotion_control(..)` and friends actually
        // reach the pipeline.
        #[cfg(feature = "emotion")]
        let emotion_controller = match self.emotion_config.take() {
            Some(builder) => Some(Arc::new(builder.build().await.map_err(|e| {
                VoirsError::model_error(format!("Failed to initialize emotion controller: {e}"))
            })?)),
            None => None,
        };

        #[cfg(feature = "cloning")]
        let voice_cloner = match self.cloning_config.take() {
            Some(builder) => Some(Arc::new(builder.build().await.map_err(|e| {
                VoirsError::model_error(format!("Failed to initialize voice cloner: {e}"))
            })?)),
            None => None,
        };

        #[cfg(feature = "conversion")]
        let voice_converter = match self.conversion_config.take() {
            Some(builder) => Some(Arc::new(builder.build().await.map_err(|e| {
                VoirsError::model_error(format!("Failed to initialize voice converter: {e}"))
            })?)),
            None => None,
        };

        #[cfg(feature = "singing")]
        let singing_controller = match self.singing_config.take() {
            Some(builder) => Some(Arc::new(builder.build().await.map_err(|e| {
                VoirsError::model_error(format!("Failed to initialize singing controller: {e}"))
            })?)),
            None => None,
        };

        #[cfg(feature = "spatial")]
        let spatial_controller = match self.spatial_config.take() {
            Some(builder) => Some(Arc::new(builder.build().await.map_err(|e| {
                VoirsError::model_error(format!(
                    "Failed to initialize spatial audio controller: {e}"
                ))
            })?)),
            None => None,
        };

        let pipeline = VoirsPipeline::from_parts(
            inner,
            #[cfg(feature = "emotion")]
            emotion_controller,
            #[cfg(feature = "cloning")]
            voice_cloner,
            #[cfg(feature = "conversion")]
            voice_converter,
            #[cfg(feature = "singing")]
            singing_controller,
            #[cfg(feature = "spatial")]
            spatial_controller,
        );

        // Perform post-initialization setup
        self.post_initialization_setup(&pipeline).await?;

        // Set pipeline state to ready
        pipeline.set_ready().await?;

        Ok(pipeline)
    }

    /// Setup cache directory with proper permissions
    async fn setup_cache_directory(&self) -> Result<()> {
        // Skip expensive file operations in test mode
        if self.test_mode {
            debug!("Skipping cache directory setup in test mode");
            return Ok(());
        }

        let cache_dir = self.config.effective_cache_dir();
        debug!("Setting up cache directory: {}", cache_dir.display());

        // Create cache directory if it doesn't exist
        tokio::fs::create_dir_all(&cache_dir)
            .await
            .map_err(|e| VoirsError::IoError {
                path: cache_dir.clone(),
                operation: crate::error::IoOperation::Create,
                source: e,
            })?;

        // Verify write permissions
        let test_file = cache_dir.join(".voirs_test");
        if let Err(e) = tokio::fs::write(&test_file, b"test").await {
            return Err(VoirsError::IoError {
                path: cache_dir,
                operation: crate::error::IoOperation::Write,
                source: e,
            });
        }

        // Clean up test file
        let _ = tokio::fs::remove_file(&test_file).await;

        debug!("Cache directory setup completed successfully");
        Ok(())
    }

    /// Setup voice manager
    async fn setup_voice_manager(&self) -> Result<Arc<RwLock<DefaultVoiceManager>>> {
        debug!("Setting up voice manager");

        let voice_manager: Arc<RwLock<DefaultVoiceManager>> =
            if let Some(ref manager) = self.voice_manager {
                manager.clone()
            } else {
                let cache_dir = self.config.effective_cache_dir();
                let mut manager = DefaultVoiceManager::new(&cache_dir);
                // Set test mode if enabled
                if self.test_mode {
                    manager.set_test_mode(true);
                }
                Arc::new(RwLock::new(manager))
            };

        // If test mode is enabled on existing manager, update it
        if self.test_mode {
            let mut manager = voice_manager.write().await;
            manager.set_test_mode(true);
        }

        debug!("Voice manager setup completed");
        Ok(voice_manager)
    }

    /// Ensure voice is available, downloading if necessary
    async fn ensure_voice_available(&self, voice_id: &str) -> Result<()> {
        debug!("Ensuring voice '{}' is available", voice_id);

        // Check if voice is available locally
        if !self.is_voice_locally_available(voice_id).await {
            info!(
                "Voice '{}' not found locally, attempting to download",
                voice_id
            );

            // Check if voice is available remotely
            if !self.is_voice_available_remotely(voice_id).await {
                return Err(VoirsError::VoiceNotFound {
                    voice: voice_id.to_string(),
                    available: self.list_available_voices().await,
                    suggestions: self.suggest_similar_voices(voice_id).await,
                });
            }

            // Download the voice
            self.download_voice(voice_id).await?;
        }

        Ok(())
    }

    /// Check if voice is available locally
    async fn is_voice_locally_available(&self, voice_id: &str) -> bool {
        let voice_dir = self
            .config
            .effective_cache_dir()
            .join("voices")
            .join(voice_id);

        // Check if voice directory exists and contains required files
        if voice_dir.exists() {
            let required_files = [
                "config.json",
                "acoustic_model.safetensors",
                "vocoder_model.safetensors",
            ];
            for file in &required_files {
                if !voice_dir.join(file).exists() {
                    debug!("Voice '{}' missing required file: {}", voice_id, file);
                    return false;
                }
            }

            debug!("Voice '{}' is available locally", voice_id);
            true
        } else {
            debug!("Voice '{}' not found locally", voice_id);
            false
        }
    }

    /// Download voice from remote repository
    async fn download_voice(&self, voice_id: &str) -> Result<()> {
        info!("Downloading voice '{}'", voice_id);

        let voice_dir = self
            .config
            .effective_cache_dir()
            .join("voices")
            .join(voice_id);

        // Create voice directory
        tokio::fs::create_dir_all(&voice_dir)
            .await
            .map_err(|e| VoirsError::IoError {
                path: voice_dir.clone(),
                operation: crate::error::IoOperation::Create,
                source: e,
            })?;

        // Download voice files from remote repository
        let download_tasks = vec![
            self.download_voice_file(voice_id, "config.json", &voice_dir),
            self.download_voice_file(voice_id, "acoustic_model.safetensors", &voice_dir),
            self.download_voice_file(voice_id, "vocoder_model.safetensors", &voice_dir),
            self.download_voice_file(voice_id, "g2p_model.json", &voice_dir), // Optional
        ];

        // Execute downloads in parallel
        let results = futures::future::join_all(download_tasks).await;

        // Check for download failures (ignore optional files)
        for (i, result) in results.into_iter().enumerate() {
            if let Err(e) = result {
                if i < 3 {
                    // Required files (config, acoustic, vocoder)
                    return Err(VoirsError::DownloadFailed {
                        url: format!("voice/{voice_id}"),
                        reason: e.to_string(),
                        bytes_downloaded: 0,
                        total_bytes: None,
                    });
                } else {
                    debug!("Optional file download failed: {}", e);
                }
            }
        }

        info!("Voice '{}' downloaded successfully", voice_id);
        Ok(())
    }

    /// Download a specific voice file
    async fn download_voice_file(
        &self,
        voice_id: &str,
        filename: &str,
        voice_dir: &std::path::Path,
    ) -> Result<()> {
        let start_time = Instant::now();
        debug!("Downloading {} for voice '{}'", filename, voice_id);

        // Get base URL from configuration or use default
        let base_url = self.get_download_base_url();
        let file_url = format!("{base_url}/voices/{voice_id}/{filename}");

        debug!("Download URL: {}", file_url);

        // Install the pure-Rust rustls CryptoProvider before any TLS handshake
        // (reqwest is built with `rustls-no-provider`). Once-guarded; safe to repeat.
        crate::ensure_crypto_provider();

        // Create HTTP client with timeout and retry configuration
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300)) // 5 minute timeout
            .connect_timeout(std::time::Duration::from_secs(30))
            .user_agent("VoiRS-SDK/1.0")
            .build()
            .map_err(|e| VoirsError::DownloadFailed {
                url: file_url.clone(),
                reason: format!("Failed to create HTTP client: {e}"),
                bytes_downloaded: 0,
                total_bytes: None,
            })?;

        // Attempt download with retry logic
        let mut retry_count = 0;
        let max_retries = 3;
        let mut last_error = None;

        while retry_count <= max_retries {
            if retry_count > 0 {
                let delay = std::time::Duration::from_secs(2u64.pow(retry_count));
                debug!(
                    "Retrying download in {:?} (attempt {}/{})",
                    delay,
                    retry_count + 1,
                    max_retries + 1
                );
                tokio::time::sleep(delay).await;
            }

            match self
                .download_file_attempt(&client, &file_url, voice_dir, filename)
                .await
            {
                Ok(bytes_downloaded) => {
                    let elapsed = start_time.elapsed();
                    info!(
                        "Successfully downloaded {} ({} bytes) in {:.2}s",
                        filename,
                        bytes_downloaded,
                        elapsed.as_secs_f64()
                    );
                    return Ok(());
                }
                Err(e) => {
                    warn!("Download attempt {} failed: {}", retry_count + 1, e);
                    last_error = Some(e);
                    retry_count += 1;
                }
            }
        }

        // All retries failed
        let error = last_error.unwrap_or_else(|| VoirsError::DownloadFailed {
            url: file_url.clone(),
            reason: "Unknown error".to_string(),
            bytes_downloaded: 0,
            total_bytes: None,
        });

        Err(VoirsError::DownloadFailed {
            url: file_url,
            reason: format!("Download failed after {} retries: {error}", max_retries + 1),
            bytes_downloaded: 0,
            total_bytes: None,
        })
    }

    /// Perform a single download attempt
    async fn download_file_attempt(
        &self,
        client: &reqwest::Client,
        url: &str,
        voice_dir: &std::path::Path,
        filename: &str,
    ) -> Result<u64> {
        // Send HTTP request
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

        // Check response status
        if !response.status().is_success() {
            return Err(VoirsError::DownloadFailed {
                url: url.to_string(),
                reason: {
                    let status = response.status();
                    format!(
                        "HTTP {} {}",
                        status.as_u16(),
                        status.canonical_reason().unwrap_or("Unknown")
                    )
                },
                bytes_downloaded: 0,
                total_bytes: None,
            });
        }

        // Get content length for progress tracking
        let total_bytes = response.content_length();
        debug!(
            "Starting download: {} bytes{}",
            total_bytes
                .map(|b| b.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            if total_bytes.is_some() {
                ""
            } else {
                " (unknown size)"
            }
        );

        // Create file for writing
        let file_path = voice_dir.join(filename);
        let mut file =
            tokio::fs::File::create(&file_path)
                .await
                .map_err(|e| VoirsError::IoError {
                    path: file_path.clone(),
                    operation: crate::error::IoOperation::Create,
                    source: e,
                })?;

        // Download with progress tracking
        let mut bytes_downloaded = 0u64;
        let mut stream = response.bytes_stream();

        while let Some(chunk) = futures::StreamExt::next(&mut stream).await {
            let chunk = chunk.map_err(|e| VoirsError::DownloadFailed {
                url: url.to_string(),
                reason: format!("Failed to read response chunk: {e}"),
                bytes_downloaded,
                total_bytes,
            })?;

            file.write_all(&chunk)
                .await
                .map_err(|e| VoirsError::IoError {
                    path: file_path.clone(),
                    operation: crate::error::IoOperation::Write,
                    source: e,
                })?;

            bytes_downloaded += chunk.len() as u64;

            // Log progress for large files
            if let Some(total) = total_bytes {
                if total > 10_000_000 && bytes_downloaded.is_multiple_of(5_000_000) {
                    // Log every 5MB for files larger than 10MB
                    debug!(
                        "Download progress: {} / {} bytes ({:.1}%)",
                        bytes_downloaded,
                        total,
                        (bytes_downloaded as f64 / total as f64) * 100.0
                    );
                }
            }
        }

        // Flush and close file
        file.flush().await.map_err(|e| VoirsError::IoError {
            path: file_path.clone(),
            operation: crate::error::IoOperation::Write,
            source: e,
        })?;

        // Validate file size if content-length was provided
        if let Some(expected_size) = total_bytes {
            if bytes_downloaded != expected_size {
                return Err(VoirsError::DownloadFailed {
                    url: url.to_string(),
                    reason: format!(
                        "Downloaded size ({bytes_downloaded} bytes) doesn't match expected size ({expected_size} bytes)"
                    ),
                    bytes_downloaded,
                    total_bytes,
                });
            }
        }

        debug!(
            "File downloaded successfully: {} ({} bytes)",
            filename, bytes_downloaded
        );
        Ok(bytes_downloaded)
    }

    /// Get base URL for downloads
    fn get_download_base_url(&self) -> String {
        // First check environment variable
        if let Ok(url) = std::env::var("VOIRS_DOWNLOAD_BASE_URL") {
            return url;
        }

        // Check configuration if available
        if let Some(url) = &self.config.model_loading.download_base_url {
            return url.clone();
        }

        // Default to public repository
        "https://huggingface.co/voirs/models/resolve/main".to_string()
    }

    /// List available voices (both local and remote)
    async fn list_available_voices(&self) -> Vec<String> {
        let mut voices = Vec::new();

        // Skip file system scanning in test mode for performance
        if !self.test_mode {
            // List local voices
            let voices_dir = self.config.effective_cache_dir().join("voices");
            if voices_dir.exists() {
                if let Ok(entries) = std::fs::read_dir(&voices_dir) {
                    for entry in entries.flatten() {
                        if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                            if let Some(voice_name) = entry.file_name().to_str() {
                                voices.push(voice_name.to_string());
                            }
                        }
                    }
                }
            }
        } else {
            debug!("Skipping voice directory scanning in test mode");
        }

        // Add well-known remote voices if not already present
        let remote_voices = vec![
            "en-us-female-01".to_string(),
            "en-us-male-01".to_string(),
            "en-gb-female-01".to_string(),
            "ja-jp-female-01".to_string(),
        ];

        for remote_voice in remote_voices {
            if !voices.contains(&remote_voice) {
                voices.push(remote_voice);
            }
        }

        voices
    }

    /// Suggest similar voices based on voice ID
    async fn suggest_similar_voices(&self, voice_id: &str) -> Vec<String> {
        let available_voices = self.list_available_voices().await;
        let mut suggestions = Vec::new();

        // Simple similarity matching based on language prefix
        let voice_prefix = voice_id.split('-').take(2).collect::<Vec<_>>().join("-");

        for voice in &available_voices {
            if voice.starts_with(&voice_prefix) && voice != voice_id {
                suggestions.push(voice.clone());
            }
        }

        // If no similar voices found, suggest popular ones
        if suggestions.is_empty() {
            suggestions.extend(
                available_voices
                    .into_iter()
                    .filter(|v| v.contains("en-us") || v.contains("en-gb"))
                    .take(3),
            );
        }

        suggestions
    }

    /// Check if voice is available remotely
    pub(crate) async fn is_voice_available_remotely(&self, voice_id: &str) -> bool {
        debug!("Checking remote availability for voice '{}'", voice_id);

        // Install the pure-Rust rustls CryptoProvider before any TLS handshake
        // (reqwest is built with `rustls-no-provider`). Once-guarded; safe to repeat.
        crate::ensure_crypto_provider();

        // Create HTTP client for quick availability check
        let client = match reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .connect_timeout(std::time::Duration::from_secs(10))
            .user_agent("VoiRS-SDK/1.0")
            .build()
        {
            Ok(client) => client,
            Err(e) => {
                debug!("Failed to create HTTP client for availability check: {}", e);
                return false;
            }
        };

        // Check for the main config file to determine if voice exists
        let base_url = self.get_download_base_url();
        let config_url = format!("{base_url}/voices/{voice_id}/config.json");

        debug!("Checking voice availability at: {}", config_url);

        match client.head(&config_url).send().await {
            Ok(response) => {
                let is_available = response.status().is_success();
                debug!(
                    "Voice '{}' remote availability: {} (HTTP {})",
                    voice_id,
                    is_available,
                    response.status().as_u16()
                );
                is_available
            }
            Err(e) => {
                debug!(
                    "Failed to check remote availability for voice '{}': {}",
                    voice_id, e
                );
                false
            }
        }
    }

    /// Check if all required voice files are available remotely
    #[allow(dead_code)]
    pub(crate) async fn check_voice_files_availability(
        &self,
        voice_id: &str,
    ) -> Result<Vec<String>> {
        debug!(
            "Checking availability of all files for voice '{}'",
            voice_id
        );

        // Install the pure-Rust rustls CryptoProvider before any TLS handshake
        // (reqwest is built with `rustls-no-provider`). Once-guarded; safe to repeat.
        crate::ensure_crypto_provider();

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .connect_timeout(std::time::Duration::from_secs(10))
            .user_agent("VoiRS-SDK/1.0")
            .build()
            .map_err(|e| VoirsError::DownloadFailed {
                url: "availability_check".to_string(),
                reason: format!("Failed to create HTTP client: {e}"),
                bytes_downloaded: 0,
                total_bytes: None,
            })?;

        let base_url = self.get_download_base_url();
        let required_files = vec![
            "config.json",
            "acoustic_model.safetensors",
            "vocoder_model.safetensors",
        ];
        let optional_files = vec!["g2p_model.json"];

        let mut available_files = Vec::new();
        let mut missing_required = Vec::new();

        // Check required files
        for filename in &required_files {
            let file_url = format!("{base_url}/voices/{voice_id}/{filename}");
            match client.head(&file_url).send().await {
                Ok(response) if response.status().is_success() => {
                    available_files.push(filename.to_string());
                    debug!("Required file '{}' is available", filename);
                }
                Ok(response) => {
                    missing_required.push(filename.to_string());
                    debug!(
                        "Required file '{}' not available (HTTP {})",
                        filename,
                        response.status().as_u16()
                    );
                }
                Err(e) => {
                    missing_required.push(filename.to_string());
                    debug!("Failed to check required file '{}': {}", filename, e);
                }
            }
        }

        // Check optional files
        for filename in &optional_files {
            let file_url = format!("{base_url}/voices/{voice_id}/{filename}");
            match client.head(&file_url).send().await {
                Ok(response) if response.status().is_success() => {
                    available_files.push(filename.to_string());
                    debug!("Optional file '{}' is available", filename);
                }
                Ok(_) | Err(_) => {
                    debug!("Optional file '{}' not available (this is okay)", filename);
                }
            }
        }

        if !missing_required.is_empty() {
            return Err(VoirsError::VoiceNotFound {
                voice: voice_id.to_string(),
                available: self.list_available_voices().await,
                suggestions: self.suggest_similar_voices(voice_id).await,
            });
        }

        debug!("Voice '{}' has all required files available", voice_id);
        Ok(available_files)
    }

    /// Validate G2P component after loading
    async fn validate_g2p_component(&self, g2p: &Arc<dyn G2p>) -> Result<()> {
        debug!("Validating G2P component");

        // Check if G2P supports the required language. Comparison goes through the
        // G2P backend's own language codes because the SDK has several aliases for
        // the same language (e.g. `JaJp` and `Ja`), and a bare `contains` would
        // report a spurious mismatch for an entirely correct component.
        let supported_languages = g2p.supported_languages();
        let required_language = self.config.default_synthesis.language;
        let required_backend_language =
            crate::pipeline::init::PipelineInitializer::g2p_language(required_language);

        let supported = supported_languages.iter().any(|lang| {
            *lang == required_language
                || (required_backend_language.is_some()
                    && crate::pipeline::init::PipelineInitializer::g2p_language(*lang)
                        == required_backend_language)
        });

        if !supported {
            warn!(
                "G2P component does not support required language: {:?}. Supported: {:?}",
                required_language, supported_languages
            );
        }

        // Test basic functionality
        let test_result = g2p.to_phonemes("test", Some(required_language)).await;
        if test_result.is_err() {
            return Err(VoirsError::ModelError {
                model_type: crate::error::ModelType::G2p,
                message: "G2P component failed basic functionality test".to_string(),
                source: None,
            });
        }

        debug!("G2P component validation completed");
        Ok(())
    }

    /// Validate acoustic model component after loading
    async fn validate_acoustic_component(&self, acoustic: &Arc<dyn AcousticModel>) -> Result<()> {
        debug!("Validating acoustic model component");

        // Check metadata
        let metadata = acoustic.metadata();
        debug!("Acoustic model metadata: {:?}", metadata);

        // Validate sample rate compatibility
        if metadata.sample_rate != self.config.default_synthesis.sample_rate {
            warn!(
                "Acoustic model sample rate ({}) differs from configured rate ({})",
                metadata.sample_rate, self.config.default_synthesis.sample_rate
            );
        }

        debug!("Acoustic model validation completed");
        Ok(())
    }

    /// Validate vocoder component after loading
    async fn validate_vocoder_component(&self, vocoder: &Arc<dyn Vocoder>) -> Result<()> {
        debug!("Validating vocoder component");

        // Check metadata
        let metadata = vocoder.metadata();
        debug!("Vocoder metadata: {:?}", metadata);

        // Validate sample rate compatibility
        if metadata.sample_rate != self.config.default_synthesis.sample_rate {
            warn!(
                "Vocoder sample rate ({}) differs from configured rate ({})",
                metadata.sample_rate, self.config.default_synthesis.sample_rate
            );
        }

        debug!("Vocoder validation completed");
        Ok(())
    }

    /// Perform post-initialization setup
    async fn post_initialization_setup(&self, pipeline: &VoirsPipeline) -> Result<()> {
        debug!("Performing post-initialization setup");

        // Synchronize all components
        pipeline.synchronize_components().await?;

        // Pre-warm components if configured (skip in test mode for performance)
        if self.config.model_loading.preload_models && !self.test_mode {
            self.prewarm_components(pipeline).await?;
        } else if self.test_mode {
            tracing::debug!("Skipping component pre-warming in test mode");
        }

        debug!("Post-initialization setup completed");
        Ok(())
    }

    /// Pre-warm components for faster first synthesis
    async fn prewarm_components(&self, pipeline: &VoirsPipeline) -> Result<()> {
        debug!("Pre-warming pipeline components");

        // Perform a small test synthesis to warm up all components
        let test_text = "Hello";
        if let Err(e) = pipeline.synthesize(test_text).await {
            warn!("Component pre-warming failed: {}", e);
            // Don't fail the build, just warn
        } else {
            debug!("Component pre-warming completed successfully");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{builder::builder_impl::VoirsPipelineBuilder, VoirsError};

    #[tokio::test]
    async fn test_pipeline_build() {
        // Test without specifying a voice to avoid voice download issues
        let builder = VoirsPipelineBuilder::new()
            .with_validation(false) // Disable validation for test
            .with_test_mode(true); // Enable test mode for fast testing

        let result = builder.build().await;
        if let Err(ref e) = result {
            eprintln!("Pipeline build failed: {e:?}");
        }
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_cache_directory_setup() {
        let builder = VoirsPipelineBuilder::new().with_test_mode(true);
        let result = builder.setup_cache_directory().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_voice_manager_setup() {
        let builder = VoirsPipelineBuilder::new().with_test_mode(true);
        let result = builder.setup_voice_manager().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_component_loading_in_test_mode() {
        use crate::pipeline::init::{ComponentOverrides, PipelineInitializer};

        let builder = VoirsPipelineBuilder::new().with_test_mode(true);
        let initializer = PipelineInitializer::new(builder.get_config());

        let (g2p, acoustic, vocoder) = initializer
            .initialize_components_with(ComponentOverrides::default(), true)
            .await
            .expect("test-mode components");

        assert!(!g2p.supported_languages().is_empty());
        assert!(!acoustic.metadata().name.is_empty());
        assert!(!vocoder.metadata().name.is_empty());
    }

    #[tokio::test]
    async fn test_component_validation() {
        use crate::pipeline::init::{ComponentOverrides, PipelineInitializer};

        let builder = VoirsPipelineBuilder::new().with_test_mode(true);
        let initializer = PipelineInitializer::new(builder.get_config());
        let (g2p, acoustic, vocoder) = initializer
            .initialize_components_with(ComponentOverrides::default(), true)
            .await
            .expect("test-mode components");

        assert!(builder.validate_g2p_component(&g2p).await.is_ok());
        assert!(builder.validate_acoustic_component(&acoustic).await.is_ok());
        assert!(builder.validate_vocoder_component(&vocoder).await.is_ok());
    }

    /// The custom-component injection path must be honored: injected components
    /// are used verbatim, and no model weights are required or downloaded.
    #[tokio::test]
    async fn test_custom_components_bypass_model_loading() {
        use crate::pipeline::init::{ComponentOverrides, PipelineInitializer};
        use crate::pipeline::{DummyAcoustic, DummyG2p, DummyVocoder};
        use std::sync::Arc;

        let cache_dir = tempfile::tempdir().expect("temp dir");

        let builder = VoirsPipelineBuilder::new()
            .with_cache_dir(cache_dir.path())
            .with_validation(false)
            .with_g2p(Arc::new(DummyG2p::new()))
            .with_acoustic_model(Arc::new(DummyAcoustic::new()))
            .with_vocoder(Arc::new(DummyVocoder::new()));

        // The builder must carry the injected components through.
        let overrides = ComponentOverrides {
            g2p: builder.get_custom_g2p(),
            acoustic: builder.get_custom_acoustic(),
            vocoder: builder.get_custom_vocoder(),
        };
        assert!(overrides.g2p.is_some());
        assert!(overrides.acoustic.is_some());
        assert!(overrides.vocoder.is_some());

        // Resolving components must succeed with NO model weights present and
        // NO test-mode opt-in, because every component was supplied.
        let initializer = PipelineInitializer::new(builder.get_config());
        let (g2p, acoustic, vocoder) = initializer
            .initialize_components_with(overrides, false)
            .await
            .expect("injected components must not require model weights");

        assert_eq!(g2p.metadata().name, "DummyG2p");
        assert_eq!(acoustic.metadata().name, "DummyAcoustic");
        assert_eq!(vocoder.metadata().name, "DummyVocoder");

        // Nothing may have been downloaded into the cache directory.
        let entries: Vec<_> = std::fs::read_dir(cache_dir.path())
            .expect("read dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert!(
            entries.is_empty(),
            "injected components must not trigger downloads, found {entries:?}"
        );
    }

    #[tokio::test]
    async fn test_voice_availability() {
        let builder = VoirsPipelineBuilder::new().with_test_mode(true);

        let is_available = builder.is_voice_locally_available("test-voice").await;
        eprintln!("Voice availability for 'test-voice': {is_available}");
        // The current implementation checks for actual files, so this will be false in tests
        // We should either create dummy files or modify the test to match the actual behavior
        assert!(!is_available);
    }

    #[tokio::test]
    async fn test_download_base_url() {
        let builder = VoirsPipelineBuilder::new().with_test_mode(true);

        // Test default URL
        let url = builder.get_download_base_url();
        assert!(!url.is_empty());
        assert!(url.starts_with("https://"));

        // Test environment variable override
        std::env::set_var("VOIRS_DOWNLOAD_BASE_URL", "https://example.com/test");
        let url = builder.get_download_base_url();
        assert_eq!(url, "https://example.com/test");

        // Clean up
        std::env::remove_var("VOIRS_DOWNLOAD_BASE_URL");
    }

    #[tokio::test]
    async fn test_remote_voice_availability() {
        let builder = VoirsPipelineBuilder::new().with_test_mode(true);

        // Test with a voice that doesn't exist
        let is_available = builder
            .is_voice_available_remotely("non-existent-voice")
            .await;
        // This should be false since the voice doesn't exist
        assert!(!is_available);

        // Note: We can't easily test a successful case without a real server
        // In a real environment, you would test with a known voice ID
    }

    #[tokio::test]
    async fn test_voice_files_availability() {
        let builder = VoirsPipelineBuilder::new().with_test_mode(true);

        // Test with a voice that doesn't exist
        let result = builder
            .check_voice_files_availability("non-existent-voice")
            .await;
        // This should return an error since the voice doesn't exist
        assert!(result.is_err());

        if let Err(e) = result {
            match e {
                VoirsError::VoiceNotFound { voice, .. } => {
                    assert_eq!(voice, "non-existent-voice");
                }
                _ => panic!("Expected VoiceNotFound error, got: {e:?}"),
            }
        }
    }
}
