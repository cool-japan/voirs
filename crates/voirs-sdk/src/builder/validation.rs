//! Configuration validation logic for pipeline builder.

use crate::{
    error::Result,
    traits::VoiceManager,
    voice::DefaultVoiceManager,
    VoirsError,
};

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
                    reason: format!("Cache directory parent does not exist: {}", parent.display()),
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

        if self.config.max_cache_size_mb > 10_240 { // 10GB
            tracing::warn!("Very large cache size configured: {}MB", self.config.max_cache_size_mb);
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
                        "G2P does not support target language {:?}. Supported: {:?}",
                        target_language, supported_languages
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
        if estimated_gpu_memory > 8192 { // 8GB
            tracing::warn!(
                "High GPU memory usage estimated: {}MB. Consider using lower quality settings.",
                estimated_gpu_memory
            );
        }
        
        // Validate device-specific settings
        if self.config.device.starts_with("cuda") {
            if !self.is_cuda_available() {
                return Err(VoirsError::DeviceNotAvailable {
                    device: self.config.device.clone(),
                    alternatives: vec!["cpu".to_string(), "auto".to_string()],
                });
            }
        }
        
        if self.config.device.starts_with("mps") {
            if !self.is_mps_available() {
                return Err(VoirsError::DeviceNotAvailable {
                    device: self.config.device.clone(),
                    alternatives: vec!["cpu".to_string(), "auto".to_string()],
                });
            }
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
            tracing::warn!(
                "Ultra quality without GPU acceleration may be very slow"
            );
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
        if estimated_memory_mb > 8192 { // 8GB
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
        // Check for CUDA availability through multiple methods
        
        // Method 1: Check for nvidia-smi
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
        // TODO: Implement proper MPS detection for Apple Silicon
        cfg!(target_os = "macos")
    }

    /// Check if device format is valid
    fn is_valid_device_format(&self, device: &str) -> bool {
        matches!(
            device,
            "cpu" | "cuda" | "mps" | "auto"
        ) || device.starts_with("cuda:") || device.starts_with("mps:")
    }


    /// Validate voice compatibility with current configuration
    fn validate_voice_compatibility(&self, _voice_id: &str) -> Result<()> {
        // TODO: Implement voice compatibility checking
        // Check if voice supports the requested language, quality, etc.
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
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let invalid_device = VoirsPipelineBuilder::new()
            .with_device("invalid-device");
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
        let invalid_rate = VoirsPipelineBuilder::new()
            .with_speaking_rate(3.0); // Too high
        assert!(invalid_rate.validate_synthesis_config().is_err());

        // Invalid pitch shift
        let invalid_pitch = VoirsPipelineBuilder::new()
            .with_pitch_shift(15.0); // Too high
        assert!(invalid_pitch.validate_synthesis_config().is_err());

        // Invalid volume gain
        let invalid_volume = VoirsPipelineBuilder::new()
            .with_volume_gain(25.0); // Too high
        assert!(invalid_volume.validate_synthesis_config().is_err());
    }

    #[test]
    fn test_cache_directory_validation() {
        // Valid cache configuration
        let valid_builder = VoirsPipelineBuilder::new()
            .with_cache_size(1024);
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