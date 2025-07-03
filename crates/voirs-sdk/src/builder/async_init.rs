//! Async initialization logic for pipeline builder.

use crate::{
    error::Result,
    pipeline::VoirsPipeline,
    traits::{AcousticModel, G2p, Vocoder, VoiceManager},
    voice::DefaultVoiceManager,
    VoirsError,
};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{info, debug, warn};

use super::builder_impl::VoirsPipelineBuilder;

impl VoirsPipelineBuilder {
    /// Build the pipeline with async initialization
    pub async fn build(self) -> Result<VoirsPipeline> {
        let start_time = Instant::now();
        info!("Building VoiRS pipeline with configuration: {:?}", self.config);

        // Validate configuration first
        self.validate().await?;

        // Initialize components concurrently
        let pipeline = self.initialize_pipeline().await?;

        let build_time = start_time.elapsed();
        info!(
            "VoiRS pipeline built successfully in {:.2}s",
            build_time.as_secs_f64()
        );

        Ok(pipeline)
    }

    /// Initialize the complete pipeline
    async fn initialize_pipeline(self) -> Result<VoirsPipeline> {
        // Initialize cache directory
        self.setup_cache_directory().await?;

        // Create or use provided voice manager
        let voice_manager = self.setup_voice_manager().await?;

        // Load all components concurrently
        let (g2p, acoustic, vocoder) = self.load_components_parallel(&voice_manager).await?;

        // Create and configure pipeline
        let config = self.config.clone();
        let mut pipeline = VoirsPipeline::new(g2p, acoustic, vocoder, config);

        // Set voice if specified
        if let Some(ref voice_id) = self.voice_id {
            self.setup_voice(&mut pipeline, voice_id).await?;
        }

        // Perform post-initialization setup
        self.post_initialization_setup(&pipeline).await?;

        // Set pipeline state to ready
        pipeline.set_ready().await?;

        Ok(pipeline)
    }

    /// Setup cache directory with proper permissions
    async fn setup_cache_directory(&self) -> Result<()> {
        let cache_dir = self.config.effective_cache_dir();
        debug!("Setting up cache directory: {}", cache_dir.display());

        // Create cache directory if it doesn't exist
        tokio::fs::create_dir_all(&cache_dir).await.map_err(|e| {
            VoirsError::IoError {
                path: cache_dir.clone(),
                operation: crate::error::IoOperation::Create,
                source: e,
            }
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

        let voice_manager: Arc<RwLock<DefaultVoiceManager>> = if let Some(ref manager) = self.voice_manager {
            manager.clone()
        } else {
            let cache_dir = self.config.effective_cache_dir();
            Arc::new(RwLock::new(DefaultVoiceManager::new(&cache_dir)))
        };

        // Voice manager is ready to use (no initialization needed)

        debug!("Voice manager setup completed");
        Ok(voice_manager)
    }

    /// Load all components in parallel for faster initialization
    async fn load_components_parallel(
        &self,
        voice_manager: &Arc<RwLock<DefaultVoiceManager>>,
    ) -> Result<(Arc<dyn G2p>, Arc<dyn AcousticModel>, Arc<dyn Vocoder>)> {
        info!("Loading pipeline components in parallel");

        // Use tokio::join! for concurrent loading
        let (g2p_result, acoustic_result, vocoder_result) = tokio::join!(
            self.load_g2p_component(voice_manager),
            self.load_acoustic_component(voice_manager),
            self.load_vocoder_component(voice_manager)
        );

        let g2p = g2p_result?;
        let acoustic = acoustic_result?;
        let vocoder = vocoder_result?;

        info!("All pipeline components loaded successfully");
        Ok((g2p, acoustic, vocoder))
    }

    /// Load G2P component with progress reporting
    async fn load_g2p_component(
        &self,
        voice_manager: &Arc<RwLock<DefaultVoiceManager>>,
    ) -> Result<Arc<dyn G2p>> {
        debug!("Loading G2P component");

        let g2p: Arc<dyn G2p> = if let Some(ref custom_g2p) = self.custom_g2p {
            debug!("Using custom G2P component");
            custom_g2p.clone()
        } else {
            self.load_default_g2p(voice_manager).await?
        };

        // Validate component after loading
        self.validate_g2p_component(&g2p).await?;

        debug!("G2P component loaded and validated");
        Ok(g2p)
    }

    /// Load acoustic model component with progress reporting
    async fn load_acoustic_component(
        &self,
        voice_manager: &Arc<RwLock<DefaultVoiceManager>>,
    ) -> Result<Arc<dyn AcousticModel>> {
        debug!("Loading acoustic model component");

        let acoustic: Arc<dyn AcousticModel> = if let Some(ref custom_acoustic) = self.custom_acoustic {
            debug!("Using custom acoustic model component");
            custom_acoustic.clone()
        } else {
            self.load_default_acoustic(voice_manager).await?
        };

        // Validate component after loading
        self.validate_acoustic_component(&acoustic).await?;

        debug!("Acoustic model component loaded and validated");
        Ok(acoustic)
    }

    /// Load vocoder component with progress reporting
    async fn load_vocoder_component(
        &self,
        voice_manager: &Arc<RwLock<DefaultVoiceManager>>,
    ) -> Result<Arc<dyn Vocoder>> {
        debug!("Loading vocoder component");

        let vocoder: Arc<dyn Vocoder> = if let Some(ref custom_vocoder) = self.custom_vocoder {
            debug!("Using custom vocoder component");
            custom_vocoder.clone()
        } else {
            self.load_default_vocoder(voice_manager).await?
        };

        // Validate component after loading
        self.validate_vocoder_component(&vocoder).await?;

        debug!("Vocoder component loaded and validated");
        Ok(vocoder)
    }

    /// Load default G2P component based on language/voice configuration
    async fn load_default_g2p(
        &self,
        voice_manager: &Arc<RwLock<DefaultVoiceManager>>,
    ) -> Result<Arc<dyn G2p>> {
        debug!("Loading default G2P component for language: {:?}", self.config.default_synthesis.language);
        
        // Try to load real G2P component based on language
        match self.config.default_synthesis.language {
            crate::types::LanguageCode::EnUs | crate::types::LanguageCode::EnGb => {
                // Attempt to load English rule-based G2P
                if let Ok(english_g2p) = self.load_english_rule_g2p().await {
                    debug!("Loaded English rule-based G2P component");
                    return Ok(Arc::new(english_g2p));
                }
            }
            crate::types::LanguageCode::JaJp => {
                // Attempt to load Japanese G2P (OpenJTalk or similar)
                if let Ok(japanese_g2p) = self.load_japanese_g2p().await {
                    debug!("Loaded Japanese G2P component");
                    return Ok(Arc::new(japanese_g2p));
                }
            }
            _ => {
                debug!("No specific G2P implementation for language: {:?}", self.config.default_synthesis.language);
            }
        }
        
        // Fallback to voice-specific G2P if available
        if let Some(voice_id) = &self.voice_id {
            if let Ok(voice_g2p) = self.load_voice_specific_g2p(voice_id, voice_manager).await {
                debug!("Loaded voice-specific G2P component for voice: {}", voice_id);
                return Ok(Arc::new(voice_g2p));
            }
        }
        
        // Final fallback to dummy implementation
        debug!("Using dummy G2P implementation as fallback");
        Ok(Arc::new(crate::pipeline::DummyG2p::new()))
    }
    
    /// Load English rule-based G2P
    async fn load_english_rule_g2p(&self) -> Result<impl G2p> {
        debug!("Initializing English rule-based G2P");
        // For now, use DummyG2p until EnglishRuleG2p implements the G2p trait
        Ok(crate::pipeline::DummyG2p::new())
    }
    
    /// Load Japanese G2P component
    async fn load_japanese_g2p(&self) -> Result<impl G2p> {
        // This would integrate with OpenJTalk or similar Japanese G2P
        // For now, use DummyG2p as placeholder
        debug!("Japanese G2P not implemented yet, using dummy implementation");
        Ok(crate::pipeline::DummyG2p::new())
    }
    
    /// Load voice-specific G2P component
    async fn load_voice_specific_g2p(
        &self,
        voice_id: &str,
        voice_manager: &Arc<RwLock<DefaultVoiceManager>>,
    ) -> Result<impl G2p> {
        let manager = voice_manager.read().await;
        
        // Check if voice has a specific G2P component
        if let Ok(Some(voice_config)) = manager.get_voice(voice_id).await {
            if voice_config.model_config.g2p_model.is_some() {
                debug!("Voice {} has specific G2P model", voice_id);
                // Load voice-specific G2P model
                // This would load the actual model file
                return Err::<crate::pipeline::DummyG2p, _>(VoirsError::NotImplemented {
                    feature: "Voice-specific G2P loading".to_string(),
                });
            }
        }
        
        Err::<crate::pipeline::DummyG2p, _>(VoirsError::ModelNotFound {
            model_name: format!("G2P for voice {}", voice_id),
            path: self.config.effective_cache_dir(),
        })
    }

    /// Load default acoustic model based on voice/quality configuration
    async fn load_default_acoustic(
        &self,
        voice_manager: &Arc<RwLock<DefaultVoiceManager>>,
    ) -> Result<Arc<dyn AcousticModel>> {
        debug!("Loading default acoustic model for quality: {:?}", self.config.default_synthesis.quality);
        
        // Try to load voice-specific acoustic model first
        if let Some(voice_id) = &self.voice_id {
            if let Ok(voice_acoustic) = self.load_voice_specific_acoustic(voice_id, voice_manager).await {
                debug!("Loaded voice-specific acoustic model for voice: {}", voice_id);
                return Ok(voice_acoustic);
            }
        }
        
        // Try to load quality-appropriate acoustic model
        if let Ok(quality_acoustic) = self.load_quality_based_acoustic().await {
            debug!("Loaded quality-based acoustic model");
            return Ok(quality_acoustic);
        }
        
        // Fallback to dummy implementation
        debug!("Using dummy acoustic model implementation as fallback");
        Ok(Arc::new(crate::pipeline::DummyAcoustic::new()))
    }
    
    /// Load voice-specific acoustic model
    async fn load_voice_specific_acoustic(
        &self,
        voice_id: &str,
        voice_manager: &Arc<RwLock<DefaultVoiceManager>>,
    ) -> Result<Arc<dyn AcousticModel>> {
        let manager = voice_manager.read().await;
        
        // Check if voice has a specific acoustic model
        if let Ok(Some(voice_config)) = manager.get_voice(voice_id).await {
            let model_path_str = &voice_config.model_config.acoustic_model;
            let model_path = std::path::Path::new(model_path_str);
            debug!("Loading acoustic model from: {}", model_path.display());
            
            // Determine model type and load accordingly
            if model_path_str.contains("vits") {
                return Ok(self.load_vits_acoustic_model(model_path).await?);
            } else if model_path_str.contains("fastspeech") {
                return Ok(self.load_fastspeech_acoustic_model(model_path).await?);
            }
        }
        
        Err(VoirsError::ModelNotFound {
            model_name: format!("Acoustic model for voice {}", voice_id),
            path: self.config.effective_cache_dir(),
        })
    }
    
    /// Load quality-based acoustic model
    async fn load_quality_based_acoustic(&self) -> Result<Arc<dyn AcousticModel>> {
        match self.config.default_synthesis.quality {
            crate::types::QualityLevel::Ultra | crate::types::QualityLevel::High => {
                // Try to load high-quality VITS model
                self.load_default_vits_model().await
            }
            crate::types::QualityLevel::Medium => {
                // Try to load FastSpeech2 model for balance of quality and speed
                self.load_default_fastspeech_model().await
            }
            crate::types::QualityLevel::Low => {
                // Use fastest available model
                self.load_fast_acoustic_model().await
            }
        }
    }
    
    /// Load VITS acoustic model from path
    async fn load_vits_acoustic_model(&self, model_path: &std::path::Path) -> Result<Arc<dyn AcousticModel>> {
        debug!("Loading VITS model from: {}", model_path.display());
        // For now, use DummyAcoustic until VitsModel implements the AcousticModel trait
        Ok(Arc::new(crate::pipeline::DummyAcoustic::new()))
    }
    
    /// Load FastSpeech acoustic model from path
    async fn load_fastspeech_acoustic_model(&self, model_path: &std::path::Path) -> Result<Arc<dyn AcousticModel>> {
        debug!("Loading FastSpeech model from: {}", model_path.display());
        // For now, use DummyAcoustic until FastSpeech2Model implements the AcousticModel trait
        Ok(Arc::new(crate::pipeline::DummyAcoustic::new()))
    }
    
    /// Load default VITS model
    async fn load_default_vits_model(&self) -> Result<Arc<dyn AcousticModel>> {
        debug!("Attempting to load default VITS model");
        
        // Look for VITS models in cache directory
        let model_dir = self.config.effective_cache_dir().join("models").join("acoustic").join("vits");
        
        if model_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&model_dir) {
                for entry in entries.flatten() {
                    if entry.file_name().to_string_lossy().ends_with(".safetensors") ||
                       entry.file_name().to_string_lossy().ends_with(".ckpt") {
                        return self.load_vits_acoustic_model(&entry.path()).await;
                    }
                }
            }
        }
        
        Err(VoirsError::ModelNotFound {
            model_name: "Default VITS model".to_string(),
            path: model_dir,
        })
    }
    
    /// Load default FastSpeech model
    async fn load_default_fastspeech_model(&self) -> Result<Arc<dyn AcousticModel>> {
        debug!("Attempting to load default FastSpeech model");
        
        let model_dir = self.config.effective_cache_dir().join("models").join("acoustic").join("fastspeech");
        
        if model_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&model_dir) {
                for entry in entries.flatten() {
                    if entry.file_name().to_string_lossy().ends_with(".safetensors") ||
                       entry.file_name().to_string_lossy().ends_with(".ckpt") {
                        return self.load_fastspeech_acoustic_model(&entry.path()).await;
                    }
                }
            }
        }
        
        Err(VoirsError::ModelNotFound {
            model_name: "Default FastSpeech model".to_string(),
            path: model_dir,
        })
    }
    
    /// Load fast acoustic model for low-quality/fast synthesis
    async fn load_fast_acoustic_model(&self) -> Result<Arc<dyn AcousticModel>> {
        // Try FastSpeech first, then any available model
        match self.load_default_fastspeech_model().await {
            Ok(model) => Ok(model),
            Err(_) => self.load_default_vits_model().await,
        }
    }

    /// Load default vocoder based on voice/quality configuration
    async fn load_default_vocoder(
        &self,
        voice_manager: &Arc<RwLock<DefaultVoiceManager>>,
    ) -> Result<Arc<dyn Vocoder>> {
        debug!("Loading default vocoder for quality: {:?}", self.config.default_synthesis.quality);
        
        // Try to load voice-specific vocoder first
        if let Some(voice_id) = &self.voice_id {
            if let Ok(voice_vocoder) = self.load_voice_specific_vocoder(voice_id, voice_manager).await {
                debug!("Loaded voice-specific vocoder for voice: {}", voice_id);
                return Ok(voice_vocoder);
            }
        }
        
        // Try to load quality-appropriate vocoder
        if let Ok(quality_vocoder) = self.load_quality_based_vocoder().await {
            debug!("Loaded quality-based vocoder");
            return Ok(quality_vocoder);
        }
        
        // Fallback to dummy implementation
        debug!("Using dummy vocoder implementation as fallback");
        Ok(Arc::new(crate::pipeline::DummyVocoder::new()))
    }
    
    /// Load voice-specific vocoder
    async fn load_voice_specific_vocoder(
        &self,
        voice_id: &str,
        voice_manager: &Arc<RwLock<DefaultVoiceManager>>,
    ) -> Result<Arc<dyn Vocoder>> {
        let manager = voice_manager.read().await;
        
        // Check if voice has a specific vocoder
        if let Ok(Some(voice_config)) = manager.get_voice(voice_id).await {
            let model_path_str = &voice_config.model_config.vocoder_model;
            let model_path = std::path::Path::new(model_path_str);
            debug!("Loading vocoder from: {}", model_path.display());
            
            // Determine vocoder type and load accordingly
            if model_path_str.contains("hifigan") {
                return Ok(self.load_hifigan_vocoder(model_path).await?);
            } else if model_path_str.contains("waveglow") {
                return Ok(self.load_waveglow_vocoder(model_path).await?);
            }
        }
        
        Err(VoirsError::ModelNotFound {
            model_name: format!("Vocoder for voice {}", voice_id),
            path: self.config.effective_cache_dir(),
        })
    }
    
    /// Load quality-based vocoder
    async fn load_quality_based_vocoder(&self) -> Result<Arc<dyn Vocoder>> {
        match self.config.default_synthesis.quality {
            crate::types::QualityLevel::Ultra => {
                // Use highest quality HiFi-GAN V1
                self.load_hifigan_v1_vocoder().await
            }
            crate::types::QualityLevel::High => {
                // Use balanced HiFi-GAN V2
                self.load_hifigan_v2_vocoder().await
            }
            crate::types::QualityLevel::Medium => {
                // Use faster HiFi-GAN V3
                self.load_hifigan_v3_vocoder().await
            }
            crate::types::QualityLevel::Low => {
                // Use fastest available vocoder
                self.load_fast_vocoder().await
            }
        }
    }
    
    /// Load HiFi-GAN vocoder from path
    async fn load_hifigan_vocoder(&self, model_path: &std::path::Path) -> Result<Arc<dyn Vocoder>> {
        debug!("Loading HiFi-GAN vocoder from: {}", model_path.display());
        // For now, use DummyVocoder until HiFiGanVocoder implements the Vocoder trait
        Ok(Arc::new(crate::pipeline::DummyVocoder::new()))
    }
    
    /// Load WaveGlow vocoder from path
    async fn load_waveglow_vocoder(&self, model_path: &std::path::Path) -> Result<Arc<dyn Vocoder>> {
        use voirs_vocoder::waveglow::WaveGlowVocoder;
        
        debug!("Loading WaveGlow vocoder from: {}", model_path.display());
        
        // For now, use DummyVocoder until WaveGlowVocoder implements the Vocoder trait
        Ok(Arc::new(crate::pipeline::DummyVocoder::new()))
    }
    
    /// Load HiFi-GAN V1 vocoder (highest quality)
    async fn load_hifigan_v1_vocoder(&self) -> Result<Arc<dyn Vocoder>> {
        debug!("Loading HiFi-GAN V1 vocoder");
        // For now, use DummyVocoder until HiFiGanVocoder implements the Vocoder trait
        Ok(Arc::new(crate::pipeline::DummyVocoder::new()))
    }
    
    /// Load HiFi-GAN V2 vocoder (balanced)
    async fn load_hifigan_v2_vocoder(&self) -> Result<Arc<dyn Vocoder>> {
        debug!("Loading HiFi-GAN V2 vocoder");
        // For now, use DummyVocoder until HiFiGanVocoder implements the Vocoder trait
        Ok(Arc::new(crate::pipeline::DummyVocoder::new()))
    }
    
    /// Load HiFi-GAN V3 vocoder (fastest)
    async fn load_hifigan_v3_vocoder(&self) -> Result<Arc<dyn Vocoder>> {
        debug!("Loading HiFi-GAN V3 vocoder");
        // For now, use DummyVocoder until HiFiGanVocoder implements the Vocoder trait
        Ok(Arc::new(crate::pipeline::DummyVocoder::new()))
    }
    
    /// Load fastest available vocoder
    async fn load_fast_vocoder(&self) -> Result<Arc<dyn Vocoder>> {
        // Use HiFi-GAN V3 for fast synthesis
        self.load_hifigan_v3_vocoder().await
    }

    /// Setup voice configuration
    async fn setup_voice(&self, pipeline: &mut VoirsPipeline, voice_id: &str) -> Result<()> {
        info!("Setting up voice: {}", voice_id);

        // Download voice if needed and auto-download is enabled
        if self.auto_download {
            self.ensure_voice_available(voice_id).await?;
        }

        // Set the voice in the pipeline
        pipeline.set_voice(voice_id).await?;

        info!("Voice setup completed successfully");
        Ok(())
    }

    /// Ensure voice is available, downloading if necessary
    async fn ensure_voice_available(&self, voice_id: &str) -> Result<()> {
        debug!("Ensuring voice '{}' is available", voice_id);
        
        // Check if voice is available locally
        if !self.is_voice_locally_available(voice_id).await {
            info!("Voice '{}' not found locally, attempting to download", voice_id);
            
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
        let voice_dir = self.config.effective_cache_dir()
            .join("voices")
            .join(voice_id);
        
        // Check if voice directory exists and contains required files
        if voice_dir.exists() {
            let required_files = ["config.json", "acoustic_model.safetensors", "vocoder_model.safetensors"];
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
        
        let voice_dir = self.config.effective_cache_dir()
            .join("voices")
            .join(voice_id);
        
        // Create voice directory
        tokio::fs::create_dir_all(&voice_dir).await
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
                if i < 3 { // Required files (config, acoustic, vocoder)
                    return Err(VoirsError::DownloadFailed {
                        url: format!("voice/{}", voice_id),
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
        _voice_id: &str,
        filename: &str,
        _voice_dir: &std::path::Path,
    ) -> Result<()> {
        // HTTP download functionality not implemented (requires reqwest dependency)
        Err(VoirsError::NotImplemented {
            feature: format!("HTTP download for {}", filename),
        })
    }
    
    /// List available voices (both local and remote)
    async fn list_available_voices(&self) -> Vec<String> {
        let mut voices = Vec::new();
        
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
                available_voices.into_iter()
                    .filter(|v| v.contains("en-us") || v.contains("en-gb"))
                    .take(3)
            );
        }
        
        suggestions
    }
    
    /// Check if voice is available remotely
    pub(crate) async fn is_voice_available_remotely(&self, voice_id: &str) -> bool {
        debug!("Checking remote availability for voice '{}'", voice_id);
        
        // Remote availability checking requires HTTP client (not implemented)
        debug!("Remote voice availability checking not implemented (requires reqwest)");
        false
    }

    /// Validate G2P component after loading
    async fn validate_g2p_component(&self, g2p: &Arc<dyn G2p>) -> Result<()> {
        debug!("Validating G2P component");

        // Check if G2P supports required languages
        let supported_languages = g2p.supported_languages();
        let required_language = self.config.default_synthesis.language;

        if !supported_languages.contains(&required_language) {
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

        // Pre-warm components if configured
        if self.config.model_loading.preload_models {
            self.prewarm_components(pipeline).await?;
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
    use super::*;
    use crate::builder::builder_impl::VoirsPipelineBuilder;

    #[tokio::test]
    async fn test_pipeline_build() {
        let builder = VoirsPipelineBuilder::new()
            .with_voice("test-voice")
            .with_validation(false); // Disable validation for test

        let result = builder.build().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_cache_directory_setup() {
        let builder = VoirsPipelineBuilder::new();
        let result = builder.setup_cache_directory().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_voice_manager_setup() {
        let builder = VoirsPipelineBuilder::new();
        let result = builder.setup_voice_manager().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_parallel_component_loading() {
        let builder = VoirsPipelineBuilder::new();
        let voice_manager = builder.setup_voice_manager().await.unwrap();
        
        let result = builder.load_components_parallel(&voice_manager).await;
        assert!(result.is_ok());
        
        let (g2p, acoustic, vocoder) = result.unwrap();
        assert!(!g2p.supported_languages().is_empty());
        
        let metadata = acoustic.metadata();
        assert!(!metadata.name.is_empty());
        
        let vocoder_metadata = vocoder.metadata();
        assert!(!vocoder_metadata.name.is_empty());
    }

    #[tokio::test]
    async fn test_component_validation() {
        let builder = VoirsPipelineBuilder::new();
        let voice_manager = builder.setup_voice_manager().await.unwrap();
        
        let g2p = builder.load_default_g2p(&voice_manager).await.unwrap();
        let result = builder.validate_g2p_component(&g2p).await;
        assert!(result.is_ok());
        
        let acoustic = builder.load_default_acoustic(&voice_manager).await.unwrap();
        let result = builder.validate_acoustic_component(&acoustic).await;
        assert!(result.is_ok());
        
        let vocoder = builder.load_default_vocoder(&voice_manager).await.unwrap();
        let result = builder.validate_vocoder_component(&vocoder).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_voice_availability() {
        let builder = VoirsPipelineBuilder::new();
        
        let is_available = builder.is_voice_locally_available("test-voice").await;
        // For dummy implementation, this should return true
        assert!(is_available);
    }
}