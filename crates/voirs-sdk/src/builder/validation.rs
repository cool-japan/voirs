//! Configuration validation logic for pipeline builder.

use crate::{error::Result, traits::VoiceManager, voice::DefaultVoiceManager, VoirsError};

use super::builder_impl::VoirsPipelineBuilder;

impl VoirsPipelineBuilder {
    /// Validate configuration and dependencies
    pub(super) async fn validate(&self) -> Result<()> {
        if !self.validation_enabled {
            return Ok(());
        }

        tracing::debug!("Validating pipeline configuration");

        // Validate device availability
        self.validate_device()?;

        // Validate voice availability if specified
        if let Some(voice_id) = &self.voice_id {
            self.validate_voice(voice_id).await?;
        }

        // Validate synthesis configuration
        self.validate_synthesis_config()?;

        // Validate cache directory
        self.validate_cache_directory()?;

        // Validate component compatibility
        self.validate_component_compatibility()?;

        // Validate resource requirements
        self.validate_resource_requirements()?;

        // Validate advanced features
        self.validate_advanced_features()?;

        tracing::debug!("Pipeline configuration validation completed successfully");
        Ok(())
    }

    /// Validate device configuration
    fn validate_device(&self) -> Result<()> {
        tracing::debug!("Validating device configuration: {}", self.config.device);

        if self.config.use_gpu && self.config.device == "cpu" {
            return Err(VoirsError::InvalidConfiguration {
                field: "device".to_string(),
                value: self.config.device.clone(),
                reason: "GPU acceleration enabled but device is set to CPU".to_string(),
                valid_values: None,
            });
        }

        if self.config.device.starts_with("cuda") && !self.is_cuda_available() {
            tracing::warn!("CUDA device specified but CUDA is not available, falling back to CPU");
            // Don't fail validation, just warn - runtime will handle fallback
        }

        if self.config.device.starts_with("mps") && !self.is_mps_available() {
            tracing::warn!("MPS device specified but Metal Performance Shaders is not available");
        }

        // Validate device format
        if !self.is_valid_device_format(&self.config.device) {
            return Err(VoirsError::UnsupportedDevice {
                device: self.config.device.clone(),
            });
        }

        Ok(())
    }

    /// Validate voice configuration
    async fn validate_voice(&self, voice_id: &str) -> Result<()> {
        tracing::debug!("Validating voice configuration: {}", voice_id);

        // Create temporary voice manager for validation
        let cache_dir = self.config.effective_cache_dir();
        let voice_manager = DefaultVoiceManager::new(&cache_dir);

        // Check if voice is available locally
        if !voice_manager.is_voice_available(voice_id) {
            if self.auto_download {
                tracing::info!(
                    "Voice '{}' not available locally, will download during build",
                    voice_id
                );

                // Skip remote voice validation in test mode to avoid network calls
                if !cfg!(test) {
                    // Validate that the voice exists remotely
                    if !self.is_voice_available_remotely(voice_id).await {
                        let available_voices: Vec<String> = voice_manager
                            .list_voices()
                            .await?
                            .into_iter()
                            .map(|v| v.id)
                            .collect();
                        return Err(VoirsError::VoiceNotFound {
                            voice: voice_id.to_string(),
                            available: available_voices.clone(),
                            suggestions: available_voices.into_iter().take(3).collect(),
                        });
                    }
                } else {
                    tracing::debug!("Skipping remote voice availability check in test mode");
                }
            } else {
                let available_voices: Vec<String> = voice_manager
                    .list_voices()
                    .await?
                    .into_iter()
                    .map(|v| v.id)
                    .collect();
                return Err(VoirsError::VoiceNotFound {
                    voice: voice_id.to_string(),
                    available: available_voices.clone(),
                    suggestions: available_voices.into_iter().take(3).collect(),
                });
            }
        }

        // Validate voice compatibility with current configuration
        self.validate_voice_compatibility(voice_id)?;

        Ok(())
    }

    /// Validate synthesis configuration
    fn validate_synthesis_config(&self) -> Result<()> {
        tracing::debug!("Validating synthesis configuration");

        let config = &self.config.default_synthesis;

        // Validate speaking rate
        if !(0.5..=2.0).contains(&config.speaking_rate) {
            return Err(VoirsError::InvalidConfiguration {
                field: "speaking_rate".to_string(),
                value: config.speaking_rate.to_string(),
                reason: "Speaking rate must be between 0.5 and 2.0".to_string(),
                valid_values: None,
            });
        }

        // Validate pitch shift
        if !(-12.0..=12.0).contains(&config.pitch_shift) {
            return Err(VoirsError::InvalidConfiguration {
                field: "pitch_shift".to_string(),
                value: config.pitch_shift.to_string(),
                reason: "Pitch shift must be between -12.0 and 12.0 semitones".to_string(),
                valid_values: None,
            });
        }

        // Validate volume gain
        if !(-20.0..=20.0).contains(&config.volume_gain) {
            return Err(VoirsError::InvalidConfiguration {
                field: "volume_gain".to_string(),
                value: config.volume_gain.to_string(),
                reason: "Volume gain must be between -20.0 and 20.0 dB".to_string(),
                valid_values: None,
            });
        }

        // Validate sample rate
        if ![8000, 16000, 22050, 44100, 48000].contains(&config.sample_rate) {
            tracing::warn!("Unusual sample rate: {}Hz", config.sample_rate);
        }

        // Validate streaming chunk size
        if let Some(chunk_size) = config.streaming_chunk_size {
            if chunk_size == 0 || chunk_size > 1000 {
                return Err(VoirsError::InvalidConfiguration {
                    field: "streaming_chunk_size".to_string(),
                    value: chunk_size.to_string(),
                    reason: "Streaming chunk size must be between 1 and 1000 words".to_string(),
                    valid_values: None,
                });
            }
        }

        Ok(())
    }

    /// Validate cache directory
    fn validate_cache_directory(&self) -> Result<()> {
        tracing::debug!("Validating cache directory configuration");

        let cache_dir = self.config.effective_cache_dir();

        // Check if parent directory exists
        if let Some(parent) = cache_dir.parent() {
            if !parent.exists() {
                return Err(VoirsError::InvalidConfiguration {
                    field: "cache_dir".to_string(),
                    value: cache_dir.display().to_string(),
                    reason: format!(
                        "Cache directory parent does not exist: {}",
                        parent.display()
                    ),
                    valid_values: None,
                });
            }
        }

        // Validate cache size limits
        if self.config.max_cache_size_mb == 0 {
            return Err(VoirsError::InvalidConfiguration {
                field: "max_cache_size_mb".to_string(),
                value: self.config.max_cache_size_mb.to_string(),
                reason: "Cache size must be greater than 0".to_string(),
                valid_values: None,
            });
        }

        if self.config.max_cache_size_mb > 10_240 {
            // 10GB
            tracing::warn!(
                "Very large cache size configured: {}MB",
                self.config.max_cache_size_mb
            );
        }

        Ok(())
    }

    /// Validate component compatibility
    fn validate_component_compatibility(&self) -> Result<()> {
        tracing::debug!("Validating component compatibility");

        // Check G2P and acoustic model compatibility
        if self.custom_g2p.is_some() && self.custom_acoustic.is_some() {
            self.validate_g2p_acoustic_compatibility()?;
        }

        // Check acoustic model and vocoder compatibility
        if self.custom_acoustic.is_some() && self.custom_vocoder.is_some() {
            self.validate_acoustic_vocoder_compatibility()?;
        }

        // Validate GPU/CPU compatibility
        if self.config.use_gpu {
            self.validate_gpu_configuration()?;
        }

        // Validate quality level compatibility
        self.validate_quality_compatibility()?;

        Ok(())
    }

    /// Validate G2P and acoustic model compatibility
    fn validate_g2p_acoustic_compatibility(&self) -> Result<()> {
        tracing::debug!("Validating G2P and acoustic model compatibility");

        // Check language compatibility
        let target_language = self.config.default_synthesis.language;

        // G2P should support the target language
        if let Some(g2p) = &self.custom_g2p {
            let supported_languages = g2p.supported_languages();
            if !supported_languages.contains(&target_language) {
                return Err(VoirsError::ComponentSynchronizationFailed {
                    component: "G2P".to_string(),
                    reason: format!(
                        "G2P does not support target language {target_language:?}. Supported: {supported_languages:?}"
                    ),
                });
            }
        }

        // Check phoneme set compatibility
        // This would require extending the trait to expose phoneme information
        tracing::debug!("G2P and acoustic model appear compatible");
        Ok(())
    }

    /// Validate acoustic model and vocoder compatibility
    fn validate_acoustic_vocoder_compatibility(&self) -> Result<()> {
        tracing::debug!("Validating acoustic model and vocoder compatibility");

        if let (Some(acoustic), Some(vocoder)) = (&self.custom_acoustic, &self.custom_vocoder) {
            let acoustic_meta = acoustic.metadata();
            let vocoder_meta = vocoder.metadata();

            // Check sample rate compatibility
            if acoustic_meta.sample_rate != vocoder_meta.sample_rate {
                return Err(VoirsError::ComponentSynchronizationFailed {
                    component: "Acoustic-Vocoder".to_string(),
                    reason: format!(
                        "Sample rate mismatch: Acoustic={}, Vocoder={}",
                        acoustic_meta.sample_rate, vocoder_meta.sample_rate
                    ),
                });
            }

            // Check mel spectrogram dimensions compatibility
            if acoustic_meta.mel_channels != vocoder_meta.mel_channels {
                return Err(VoirsError::ComponentSynchronizationFailed {
                    component: "Acoustic-Vocoder".to_string(),
                    reason: format!(
                        "Mel channel mismatch: Acoustic={}, Vocoder={}",
                        acoustic_meta.mel_channels, vocoder_meta.mel_channels
                    ),
                });
            }

            tracing::debug!("Acoustic model and vocoder are compatible");
        }

        Ok(())
    }

    /// Validate GPU configuration
    fn validate_gpu_configuration(&self) -> Result<()> {
        tracing::debug!("Validating GPU configuration");

        // Warn about high thread count with GPU
        if let Some(threads) = self.config.num_threads {
            if threads > 16 {
                tracing::warn!(
                    "High thread count ({}) with GPU acceleration may not improve performance",
                    threads
                );
            }
        }

        // Check for GPU memory requirements
        let estimated_gpu_memory = self.estimate_gpu_memory_usage();
        if estimated_gpu_memory > 8192 {
            // 8GB
            tracing::warn!(
                "High GPU memory usage estimated: {}MB. Consider using lower quality settings.",
                estimated_gpu_memory
            );
        }

        // Validate device-specific settings
        if self.config.device.starts_with("cuda") && !self.is_cuda_available() {
            return Err(VoirsError::DeviceNotAvailable {
                device: self.config.device.clone(),
                alternatives: vec!["cpu".to_string(), "auto".to_string()],
            });
        }

        if self.config.device.starts_with("mps") && !self.is_mps_available() {
            return Err(VoirsError::DeviceNotAvailable {
                device: self.config.device.clone(),
                alternatives: vec!["cpu".to_string(), "auto".to_string()],
            });
        }

        Ok(())
    }

    /// Validate quality level compatibility
    fn validate_quality_compatibility(&self) -> Result<()> {
        tracing::debug!("Validating quality level compatibility");

        let quality = self.config.default_synthesis.quality;

        // Check if quality level is compatible with streaming settings
        if let Some(chunk_size) = self.config.default_synthesis.streaming_chunk_size {
            match quality {
                crate::types::QualityLevel::Ultra => {
                    if chunk_size < 100 {
                        tracing::warn!(
                            "Ultra quality with small chunk size ({}) may cause artifacts",
                            chunk_size
                        );
                    }
                }
                crate::types::QualityLevel::Low => {
                    if chunk_size > 500 {
                        tracing::warn!(
                            "Low quality with large chunk size ({}) may not improve latency",
                            chunk_size
                        );
                    }
                }
                _ => {}
            }
        }

        // Check if quality level is compatible with device
        if !self.config.use_gpu && quality == crate::types::QualityLevel::Ultra {
            tracing::warn!("Ultra quality without GPU acceleration may be very slow");
        }

        Ok(())
    }

    /// Estimate GPU memory usage
    fn estimate_gpu_memory_usage(&self) -> u32 {
        let mut memory_mb = 512; // Base GPU memory

        // Add memory based on quality level
        match self.config.default_synthesis.quality {
            crate::types::QualityLevel::Ultra => memory_mb += 3072, // 3GB for high-quality models
            crate::types::QualityLevel::High => memory_mb += 2048,  // 2GB
            crate::types::QualityLevel::Medium => memory_mb += 1024, // 1GB
            crate::types::QualityLevel::Low => memory_mb += 512,    // 512MB
        }

        // Add memory for custom components
        if self.custom_acoustic.is_some() {
            memory_mb += 1024; // Additional memory for custom acoustic model
        }

        if self.custom_vocoder.is_some() {
            memory_mb += 512; // Additional memory for custom vocoder
        }

        memory_mb
    }

    /// Validate resource requirements
    fn validate_resource_requirements(&self) -> Result<()> {
        tracing::debug!("Validating resource requirements");

        // Estimate memory requirements
        let estimated_memory_mb = self.estimate_memory_usage();
        if estimated_memory_mb > 8192 {
            // 8GB
            tracing::warn!(
                "High memory usage estimated: {}MB. Consider using lower quality settings.",
                estimated_memory_mb
            );
        }

        // Validate thread count
        if let Some(threads) = self.config.num_threads {
            let max_threads = std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1);

            if threads > max_threads * 2 {
                tracing::warn!(
                    "Thread count ({}) exceeds 2x available parallelism ({})",
                    threads,
                    max_threads
                );
            }
        }

        Ok(())
    }

    /// Check if CUDA is available
    fn is_cuda_available(&self) -> bool {
        // Skip expensive checks in test mode or when validation is disabled
        if !self.validation_enabled || cfg!(test) {
            tracing::debug!("Skipping CUDA availability check in test/no-validation mode");
            return false;
        }

        // Check for CUDA availability through multiple methods

        // Method 1: Check for nvidia-smi with timeout
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=count")
            .arg("--format=csv,noheader,nounits")
            .output()
        {
            if output.status.success() {
                if let Ok(count_str) = String::from_utf8(output.stdout) {
                    if let Ok(gpu_count) = count_str.trim().parse::<u32>() {
                        tracing::debug!("Found {} CUDA GPU(s)", gpu_count);
                        return gpu_count > 0;
                    }
                }
            }
        }

        // Method 2: Check for CUDA runtime library
        #[cfg(unix)]
        {
            let cuda_lib_paths = [
                "/usr/local/cuda/lib64/libcudart.so",
                "/usr/lib/x86_64-linux-gnu/libcudart.so",
                "/opt/cuda/lib64/libcudart.so",
            ];

            for path in &cuda_lib_paths {
                if std::path::Path::new(path).exists() {
                    tracing::debug!("Found CUDA runtime library at {}", path);
                    return true;
                }
            }
        }

        #[cfg(windows)]
        {
            // Check for CUDA on Windows
            if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
                let cudart_path = std::path::Path::new(&cuda_path)
                    .join("bin")
                    .join("cudart64_*.dll");

                if let Ok(entries) = glob::glob(&cudart_path.to_string_lossy()) {
                    if entries.count() > 0 {
                        tracing::debug!("Found CUDA runtime on Windows");
                        return true;
                    }
                }
            }
        }

        // Method 3: Environment variable check
        if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() {
            tracing::debug!("CUDA_VISIBLE_DEVICES environment variable found");
            return true;
        }

        tracing::debug!("CUDA not detected");
        false
    }

    /// Check if MPS (Metal Performance Shaders) is available
    fn is_mps_available(&self) -> bool {
        // Skip expensive checks in test mode or when validation is disabled
        if !self.validation_enabled || cfg!(test) {
            tracing::debug!("Skipping MPS availability check in test/no-validation mode");
            return false;
        }

        #[cfg(target_os = "macos")]
        {
            // Check if we're on Apple Silicon (ARM64)
            if cfg!(target_arch = "aarch64") {
                // MPS is available on Apple Silicon Macs (M1, M2, M3, etc.)
                tracing::debug!("MPS available on Apple Silicon");
                return true;
            }

            // Check for macOS version that supports MPS on Intel Macs
            if let Ok(output) = std::process::Command::new("sw_vers")
                .args(["-productVersion"])
                .output()
            {
                if let Ok(version_str) = String::from_utf8(output.stdout) {
                    let version = version_str.trim();
                    // Parse version (e.g., "12.3.1" -> [12, 3, 1])
                    let parts: Vec<u32> =
                        version.split('.').filter_map(|s| s.parse().ok()).collect();

                    // MPS requires macOS 12.3+ for Intel Macs with discrete GPUs
                    if parts.len() >= 2 {
                        let major = parts[0];
                        let minor = parts[1];
                        if major > 12 || (major == 12 && minor >= 3) {
                            tracing::debug!("MPS potentially available on macOS {}", version);
                            return true;
                        }
                    }
                }
            }
        }

        tracing::debug!("MPS not available");
        false
    }

    /// Check if device format is valid
    fn is_valid_device_format(&self, device: &str) -> bool {
        matches!(device, "cpu" | "cuda" | "mps" | "auto")
            || device.starts_with("cuda:")
            || device.starts_with("mps:")
    }

    /// Validate voice compatibility with current configuration
    fn validate_voice_compatibility(&self, voice_id: &str) -> Result<()> {
        // Validate voice ID format (should not be empty)
        if voice_id.is_empty() {
            return Err(VoirsError::invalid_configuration_legacy(
                "voice_id".to_string(),
                voice_id.to_string(),
                "Voice ID cannot be empty".to_string(),
            ));
        }

        // Check for valid voice ID format (alphanumeric, hyphens, underscores)
        if !voice_id
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
        {
            return Err(VoirsError::invalid_configuration_legacy(
                "voice_id".to_string(),
                voice_id.to_string(),
                "Must contain only alphanumeric characters, hyphens, and underscores".to_string(),
            ));
        }

        // Check voice ID length constraints
        if voice_id.len() > 64 {
            return Err(VoirsError::invalid_configuration_legacy(
                "voice_id".to_string(),
                voice_id.to_string(),
                "Voice ID is too long (max 64 characters)".to_string(),
            ));
        }

        // Basic language code validation if voice follows language_voice pattern
        if let Some((lang_part, _voice_part)) = voice_id.split_once('_') {
            if lang_part.len() < 2 || lang_part.len() > 3 {
                tracing::warn!(
                    "Voice ID '{}' may have invalid language code format",
                    voice_id
                );
            }
        }

        // Check compatibility with current device
        let device = &self.config.device;
        if device == "cuda" && !self.is_cuda_available() {
            return Err(VoirsError::device_not_available_legacy(format!(
                "CUDA required for voice '{voice_id}' but not available"
            )));
        }
        if device == "mps" && !self.is_mps_available() {
            return Err(VoirsError::device_not_available_legacy(format!(
                "MPS required for voice '{voice_id}' but not available"
            )));
        }

        tracing::debug!("Voice '{}' passed compatibility validation", voice_id);
        Ok(())
    }

    /// Estimate memory usage for current configuration
    fn estimate_memory_usage(&self) -> u32 {
        let mut memory_mb = 512; // Base memory for SDK

        // Add memory for quality level
        match self.config.default_synthesis.quality {
            crate::types::QualityLevel::Low => memory_mb += 256,
            crate::types::QualityLevel::Medium => memory_mb += 512,
            crate::types::QualityLevel::High => memory_mb += 1024,
            crate::types::QualityLevel::Ultra => memory_mb += 2048,
        }

        // Add cache memory
        memory_mb += self.config.max_cache_size_mb;

        // Add GPU memory overhead
        if self.config.use_gpu {
            memory_mb += 512;
        }

        memory_mb
    }

    /// Validate advanced features configuration
    fn validate_advanced_features(&self) -> Result<()> {
        tracing::debug!("Validating advanced features configuration");

        // Validate emotion control
        self.validate_emotion_control()?;

        // Validate voice cloning
        self.validate_voice_cloning()?;

        // Validate voice conversion
        self.validate_voice_conversion()?;

        // Validate singing synthesis
        self.validate_singing_synthesis()?;

        // Validate 3D spatial audio
        self.validate_spatial_audio()?;

        Ok(())
    }

    /// Validate emotion control configuration
    fn validate_emotion_control(&self) -> Result<()> {
        #[cfg(feature = "emotion")]
        {
            let config = &self.config.default_synthesis;

            if config.enable_emotion {
                tracing::debug!("Validating emotion control configuration");

                // Validate emotion intensity
                if !(0.0..=1.0).contains(&config.emotion_intensity) {
                    return Err(VoirsError::emotion_intensity_out_of_range(
                        config.emotion_intensity,
                    ));
                }

                // Validate emotion type if specified
                if let Some(emotion_type) = &config.emotion_type {
                    let supported_emotions = vec![
                        "happy",
                        "sad",
                        "angry",
                        "calm",
                        "excited",
                        "neutral",
                        "surprised",
                        "disgusted",
                        "fearful",
                        "contempt",
                        "energetic",
                        "relaxed",
                    ];

                    if !supported_emotions.contains(&emotion_type.as_str()) {
                        return Err(VoirsError::emotion_not_supported(
                            emotion_type.clone(),
                            supported_emotions
                                .into_iter()
                                .map(|s| s.to_string())
                                .collect(),
                        ));
                    }
                }

                // Validate emotion preset if specified
                if let Some(preset) = &config.emotion_preset {
                    let supported_presets =
                        vec!["happy", "sad", "angry", "calm", "excited", "neutral"];

                    if !supported_presets.contains(&preset.as_str()) {
                        return Err(VoirsError::EmotionConfigurationInvalid {
                            reason: format!(
                                "Invalid emotion preset '{}'. Supported presets: {:?}",
                                preset, supported_presets
                            ),
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// Validate voice cloning configuration
    fn validate_voice_cloning(&self) -> Result<()> {
        #[cfg(feature = "cloning")]
        {
            let config = &self.config.default_synthesis;

            if config.enable_cloning {
                tracing::debug!("Validating voice cloning configuration");

                // Validate cloning quality
                if !(0.0..=1.0).contains(&config.cloning_quality) {
                    return Err(VoirsError::voice_cloning_quality_too_low(
                        config.cloning_quality,
                        0.0,
                    ));
                }

                // Validate cloning method if specified
                if let Some(method) = &config.cloning_method {
                    let supported_methods = vec!["quick_clone", "deep_clone", "adaptive_clone"];
                    let method_str = format!("{:?}", method).to_lowercase();

                    if !supported_methods.iter().any(|&m| method_str.contains(m)) {
                        return Err(VoirsError::VoiceCloningMethodNotSupported {
                            method: method_str,
                            supported: supported_methods
                                .into_iter()
                                .map(|s| s.to_string())
                                .collect(),
                        });
                    }
                }

                // Validate device compatibility for cloning
                if !self.config.use_gpu && config.cloning_quality > 0.8 {
                    tracing::warn!(
                        "High-quality voice cloning without GPU acceleration may be very slow"
                    );
                }
            }
        }

        Ok(())
    }

    /// Validate voice conversion configuration
    fn validate_voice_conversion(&self) -> Result<()> {
        #[cfg(feature = "conversion")]
        {
            let config = &self.config.default_synthesis;

            if config.enable_conversion {
                tracing::debug!("Validating voice conversion configuration");

                // Validate conversion target if specified
                if let Some(target) = &config.conversion_target {
                    match target {
                        crate::builder::features::ConversionTarget::Gender(gender) => {
                            // Gender conversion is always supported
                            tracing::debug!("Gender conversion to {:?} is supported", gender);
                        }
                        crate::builder::features::ConversionTarget::Age(age) => {
                            // Age conversion is always supported
                            tracing::debug!("Age conversion to {:?} is supported", age);
                        }
                        crate::builder::features::ConversionTarget::Voice(voice_id) => {
                            // Validate target voice exists
                            if voice_id.is_empty() {
                                return Err(VoirsError::voice_conversion_target_invalid(
                                    "Target voice ID cannot be empty".to_string(),
                                    voice_id.clone(),
                                ));
                            }
                        }
                    }
                }

                // Validate real-time conversion constraints
                if config.realtime_conversion {
                    if config.quality == crate::types::QualityLevel::Ultra {
                        tracing::warn!(
                            "Ultra quality with real-time conversion may cause latency issues"
                        );
                    }

                    if !self.config.use_gpu {
                        tracing::warn!("Real-time voice conversion without GPU acceleration may not meet real-time constraints");
                    }
                }
            }
        }

        Ok(())
    }

    /// Validate singing synthesis configuration
    fn validate_singing_synthesis(&self) -> Result<()> {
        #[cfg(feature = "singing")]
        {
            let config = &self.config.default_synthesis;

            if config.enable_singing {
                tracing::debug!("Validating singing synthesis configuration");

                // Validate singing voice type if specified
                if let Some(voice_type) = &config.singing_voice_type {
                    let supported_types = vec![
                        "pop_vocalist",
                        "opera_singer",
                        "jazz_vocalist",
                        "rock_vocalist",
                        "choir",
                    ];
                    let voice_type_str = format!("{:?}", voice_type).to_lowercase();

                    if !supported_types.iter().any(|&t| voice_type_str.contains(t)) {
                        return Err(VoirsError::SingingTechniqueNotSupported {
                            technique: voice_type_str,
                            supported: supported_types.into_iter().map(|s| s.to_string()).collect(),
                        });
                    }
                }

                // Validate singing technique if specified
                if let Some(technique) = &config.singing_technique {
                    // Validate technique parameters
                    if !(0.0..=1.0).contains(&technique.breath_control) {
                        return Err(VoirsError::SingingError {
                            message: format!(
                                "Breath control {} is out of range (0.0-1.0)",
                                technique.breath_control
                            ),
                            voice_type: config
                                .singing_voice_type
                                .as_ref()
                                .map(|v| format!("{:?}", v)),
                        });
                    }

                    if !(0.0..=1.0).contains(&technique.vibrato_depth) {
                        return Err(VoirsError::SingingError {
                            message: format!(
                                "Vibrato depth {} is out of range (0.0-1.0)",
                                technique.vibrato_depth
                            ),
                            voice_type: config
                                .singing_voice_type
                                .as_ref()
                                .map(|v| format!("{:?}", v)),
                        });
                    }

                    if !(0.0..=1.0).contains(&technique.vocal_fry) {
                        return Err(VoirsError::SingingError {
                            message: format!(
                                "Vocal fry {} is out of range (0.0-1.0)",
                                technique.vocal_fry
                            ),
                            voice_type: config
                                .singing_voice_type
                                .as_ref()
                                .map(|v| format!("{:?}", v)),
                        });
                    }

                    if !(0.0..=1.0).contains(&technique.head_voice_ratio) {
                        return Err(VoirsError::SingingError {
                            message: format!(
                                "Head voice ratio {} is out of range (0.0-1.0)",
                                technique.head_voice_ratio
                            ),
                            voice_type: config
                                .singing_voice_type
                                .as_ref()
                                .map(|v| format!("{:?}", v)),
                        });
                    }
                }

                // Validate tempo if specified
                if let Some(tempo) = config.tempo {
                    if !(40.0..=300.0).contains(&tempo) {
                        return Err(VoirsError::tempo_out_of_range(tempo, 40.0, 300.0));
                    }
                }

                // Validate musical key if specified
                if let Some(_key) = &config.musical_key {
                    // Musical keys are validated by the enum, so no additional validation needed
                    tracing::debug!("Musical key validation passed");
                }
            }
        }

        Ok(())
    }

    /// Validate 3D spatial audio configuration
    fn validate_spatial_audio(&self) -> Result<()> {
        #[cfg(feature = "spatial")]
        {
            let config = &self.config.default_synthesis;

            if config.enable_spatial {
                tracing::debug!("Validating 3D spatial audio configuration");

                // Validate listener position if specified
                if let Some(position) = &config.listener_position {
                    // Check for extreme positions that might cause issues
                    if position.x.abs() > 1000.0
                        || position.y.abs() > 1000.0
                        || position.z.abs() > 1000.0
                    {
                        return Err(VoirsError::position_3d_invalid(
                            position.x,
                            position.y,
                            position.z,
                            "Position coordinates exceed reasonable limits (±1000.0)".to_string(),
                        ));
                    }

                    // Check for NaN or infinite values
                    if !position.x.is_finite() || !position.y.is_finite() || !position.z.is_finite()
                    {
                        return Err(VoirsError::position_3d_invalid(
                            position.x,
                            position.y,
                            position.z,
                            "Position coordinates must be finite numbers".to_string(),
                        ));
                    }
                }

                // Validate room size if specified
                if let Some(room_size) = &config.room_size {
                    // Room size validation is handled by the enum, but we can add warnings
                    match room_size {
                        crate::builder::features::RoomSize::Huge => {
                            if !self.config.use_gpu {
                                tracing::warn!("Large room acoustics without GPU acceleration may be computationally expensive");
                            }
                        }
                        _ => {}
                    }
                }

                // Validate reverb level
                if !(0.0..=1.0).contains(&config.reverb_level) {
                    return Err(VoirsError::room_acoustics_configuration_invalid(format!(
                        "Reverb level {} is out of range (0.0-1.0)",
                        config.reverb_level
                    )));
                }

                // Validate HRTF compatibility
                if config.hrtf_enabled {
                    if config.sample_rate < 44100 {
                        tracing::warn!("HRTF processing works best with sample rates >= 44.1kHz");
                    }
                }

                // Validate device compatibility for spatial audio
                if !self.config.use_gpu && config.hrtf_enabled {
                    tracing::warn!(
                        "HRTF processing without GPU acceleration may introduce latency"
                    );
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use crate::{builder::builder_impl::VoirsPipelineBuilder, types::QualityLevel};

    #[tokio::test]
    async fn test_device_validation() {
        // Valid CPU configuration
        let cpu_builder = VoirsPipelineBuilder::new()
            .with_gpu_acceleration(false)
            .with_device("cpu");
        assert!(cpu_builder.validate_device().is_ok());

        // Invalid GPU configuration (GPU enabled but CPU device)
        let invalid_builder = VoirsPipelineBuilder::new()
            .with_gpu_acceleration(true)
            .with_device("cpu");
        assert!(invalid_builder.validate_device().is_err());

        // Valid GPU configuration
        let gpu_builder = VoirsPipelineBuilder::new()
            .with_gpu_acceleration(true)
            .with_device("cuda");
        assert!(gpu_builder.validate_device().is_ok());

        // Invalid device format
        let invalid_device = VoirsPipelineBuilder::new().with_device("invalid-device");
        assert!(invalid_device.validate_device().is_err());
    }

    #[test]
    fn test_synthesis_config_validation() {
        // Valid configuration
        let valid_builder = VoirsPipelineBuilder::new()
            .with_speaking_rate(1.0)
            .with_pitch_shift(0.0)
            .with_volume_gain(0.0)
            .with_sample_rate(22050);
        assert!(valid_builder.validate_synthesis_config().is_ok());

        // Invalid speaking rate
        let invalid_rate = VoirsPipelineBuilder::new().with_speaking_rate(3.0); // Too high
        assert!(invalid_rate.validate_synthesis_config().is_err());

        // Invalid pitch shift
        let invalid_pitch = VoirsPipelineBuilder::new().with_pitch_shift(15.0); // Too high
        assert!(invalid_pitch.validate_synthesis_config().is_err());

        // Invalid volume gain
        let invalid_volume = VoirsPipelineBuilder::new().with_volume_gain(25.0); // Too high
        assert!(invalid_volume.validate_synthesis_config().is_err());
    }

    #[test]
    fn test_cache_directory_validation() {
        // Valid cache configuration
        let valid_builder = VoirsPipelineBuilder::new().with_cache_size(1024);
        assert!(valid_builder.validate_cache_directory().is_ok());

        // Invalid cache size (zero)
        let mut invalid_builder = VoirsPipelineBuilder::new();
        invalid_builder.config.max_cache_size_mb = 0;
        assert!(invalid_builder.validate_cache_directory().is_err());
    }

    #[test]
    fn test_memory_estimation() {
        let low_memory = VoirsPipelineBuilder::new()
            .with_quality(QualityLevel::Low)
            .with_cache_size(256)
            .with_gpu_acceleration(false);

        let high_memory = VoirsPipelineBuilder::new()
            .with_quality(QualityLevel::Ultra)
            .with_cache_size(2048)
            .with_gpu_acceleration(true);

        assert!(low_memory.estimate_memory_usage() < high_memory.estimate_memory_usage());
    }

    #[test]
    fn test_device_format_validation() {
        let builder = VoirsPipelineBuilder::new();

        assert!(builder.is_valid_device_format("cpu"));
        assert!(builder.is_valid_device_format("cuda"));
        assert!(builder.is_valid_device_format("mps"));
        assert!(builder.is_valid_device_format("cuda:0"));
        assert!(builder.is_valid_device_format("mps:0"));
        assert!(!builder.is_valid_device_format("invalid"));
        assert!(!builder.is_valid_device_format(""));
    }
}
