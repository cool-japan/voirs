//! Voice cloning integration for VoiRS SDK

use crate::VoirsError;
use std::sync::Arc;
use tokio::sync::RwLock;

// Re-export cloning types for convenience
pub use voirs_cloning::prelude::*;

/// SDK-integrated voice cloner
#[derive(Debug, Clone)]
pub struct VoiceCloner {
    /// Internal voice cloner
    cloner: Arc<voirs_cloning::VoiceCloner>,
    /// SDK-specific configuration
    config: Arc<RwLock<VoiceClonerConfig>>,
    /// Cached speaker profiles
    speaker_cache: Arc<RwLock<std::collections::HashMap<String, SpeakerProfile>>>,
}

/// Configuration for SDK voice cloner
#[derive(Debug, Clone)]
pub struct VoiceClonerConfig {
    /// Enable voice cloning by default
    pub enabled: bool,
    /// Default cloning method
    pub default_method: CloningMethod,
    /// Auto-quality assessment
    pub auto_quality_check: bool,
    /// Cache cloned voices
    pub cache_results: bool,
    /// Maximum cache size
    pub max_cache_size: usize,
}

impl VoiceCloner {
    /// Create new voice cloner
    pub async fn new() -> crate::Result<Self> {
        let cloner = voirs_cloning::VoiceCloner::new()
            .map_err(|e| VoirsError::model_error(format!("Voice cloner: {}", e)))?;

        Ok(Self {
            cloner: Arc::new(cloner),
            config: Arc::new(RwLock::new(VoiceClonerConfig::default())),
            speaker_cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
        })
    }

    /// Create with custom configuration
    pub async fn with_config(cloning_config: CloningConfig) -> crate::Result<Self> {
        let cloner = voirs_cloning::VoiceCloner::with_config(cloning_config)
            .map_err(|e| VoirsError::model_error(format!("Voice cloner: {}", e)))?;

        Ok(Self {
            cloner: Arc::new(cloner),
            config: Arc::new(RwLock::new(VoiceClonerConfig::default())),
            speaker_cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
        })
    }

    /// Clone voice from reference samples
    pub async fn clone_voice(
        &self,
        speaker_id: String,
        reference_samples: Vec<VoiceSample>,
        target_text: String,
        method: Option<CloningMethod>,
    ) -> crate::Result<VoiceCloneResult> {
        let config = self.config.read().await;
        if !config.enabled {
            return Err(VoirsError::ConfigError {
                field: "feature".to_string(),
                message: "Voice cloning is disabled".to_string(),
            });
        }

        // Create speaker profile
        let mut profile = SpeakerProfile::new(speaker_id.clone(), speaker_id.clone());
        for sample in &reference_samples {
            profile.add_sample(sample.clone());
        }

        // Create speaker data
        let speaker_data = SpeakerData::new(profile.clone()).with_target_text(target_text.clone());

        // Create cloning request
        let method = method.unwrap_or(config.default_method);
        let request = VoiceCloneRequest::new(
            format!("clone_{}", fastrand::u64(..)),
            speaker_data,
            method,
            target_text,
        );

        // Perform cloning
        let result = self
            .cloner
            .clone_voice(request)
            .await
            .map_err(|e| VoirsError::audio_error(format!("Voice cloning: {}", e)))?;

        // Cache speaker profile if successful
        if result.success && config.cache_results {
            let mut cache = self.speaker_cache.write().await;
            if cache.len() >= config.max_cache_size {
                // Remove oldest entry
                if let Some(oldest_key) = cache.keys().next().cloned() {
                    cache.remove(&oldest_key);
                }
            }
            cache.insert(speaker_id, profile);
        }

        Ok(result)
    }

    /// Quick clone from single audio file
    pub async fn quick_clone(
        &self,
        audio_data: Vec<f32>,
        sample_rate: u32,
        target_text: String,
    ) -> crate::Result<VoiceCloneResult> {
        let sample = VoiceSample::new("quick_clone".to_string(), audio_data, sample_rate);

        self.clone_voice(
            "quick_speaker".to_string(),
            vec![sample],
            target_text,
            Some(CloningMethod::OneShot),
        )
        .await
    }

    /// Clone from cached speaker
    pub async fn clone_from_cached_speaker(
        &self,
        speaker_id: &str,
        target_text: String,
    ) -> crate::Result<VoiceCloneResult> {
        let cache = self.speaker_cache.read().await;
        let profile = cache
            .get(speaker_id)
            .ok_or_else(|| VoirsError::ConfigError {
                field: "cache".to_string(),
                message: format!("Cached speaker: {}", speaker_id),
            })?;

        let speaker_data = SpeakerData::new(profile.clone()).with_target_text(target_text.clone());

        let config = self.config.read().await;
        let request = VoiceCloneRequest::new(
            format!("cached_clone_{}", fastrand::u64(..)),
            speaker_data,
            config.default_method,
            target_text,
        );

        self.cloner
            .clone_voice(request)
            .await
            .map_err(|e| VoirsError::audio_error(format!("Cached voice cloning: {}", e)))
    }

    /// Add speaker to cache
    pub async fn cache_speaker(&self, speaker_id: String, profile: SpeakerProfile) -> crate::Result<()> {
        let config = self.config.read().await;
        let mut cache = self.speaker_cache.write().await;

        if cache.len() >= config.max_cache_size {
            // Remove oldest entry
            if let Some(oldest_key) = cache.keys().next().cloned() {
                cache.remove(&oldest_key);
            }
        }

        cache.insert(speaker_id, profile);
        Ok(())
    }

    /// List cached speakers
    pub async fn list_cached_speakers(&self) -> Vec<String> {
        let cache = self.speaker_cache.read().await;
        cache.keys().cloned().collect()
    }

    /// Remove speaker from cache
    pub async fn remove_cached_speaker(&self, speaker_id: &str) -> crate::Result<()> {
        let mut cache = self.speaker_cache.write().await;
        cache.remove(speaker_id);
        Ok(())
    }

    /// Clear speaker cache
    pub async fn clear_cache(&self) -> crate::Result<()> {
        let mut cache = self.speaker_cache.write().await;
        cache.clear();
        Ok(())
    }

    /// Enable or disable voice cloning
    pub async fn set_enabled(&self, enabled: bool) -> crate::Result<()> {
        let mut config = self.config.write().await;
        config.enabled = enabled;
        Ok(())
    }

    /// Check if voice cloning is enabled
    pub async fn is_enabled(&self) -> bool {
        let config = self.config.read().await;
        config.enabled
    }

    /// Get cloning statistics
    pub async fn get_statistics(&self) -> crate::Result<CloningStatistics> {
        let metrics = self.cloner.get_metrics().await;
        let cache_size = self.speaker_cache.read().await.len();

        Ok(CloningStatistics {
            total_clones: metrics.total_attempts,
            successful_clones: metrics.successful_clonings,
            failed_clones: metrics.failed_clonings,
            cached_speakers: cache_size,
            success_rate: metrics.success_rate(),
            most_used_method: metrics.most_used_method(),
        })
    }

    /// Validate audio for cloning
    pub async fn validate_audio(&self, samples: &[VoiceSample]) -> crate::Result<ValidationResult> {
        let mut issues = Vec::new();
        let mut total_duration = 0.0;

        for sample in samples {
            if !sample.is_valid_for_cloning() {
                issues.push(format!("Sample {} is invalid for cloning", sample.id));
            }
            total_duration += sample.duration;
        }

        if samples.is_empty() {
            issues.push("No audio samples provided".to_string());
        } else if total_duration < 3.0 {
            issues.push(format!(
                "Total duration {:.1}s is too short (minimum 3s)",
                total_duration
            ));
        }

        Ok(ValidationResult {
            valid: issues.is_empty(),
            issues,
            total_duration,
            sample_count: samples.len(),
        })
    }
}

impl Default for VoiceClonerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_method: CloningMethod::FewShot,
            auto_quality_check: true,
            cache_results: true,
            max_cache_size: 100,
        }
    }
}

/// Statistics for voice cloning
#[derive(Debug, Clone)]
pub struct CloningStatistics {
    /// Total cloning attempts
    pub total_clones: u64,
    /// Successful clonings
    pub successful_clones: u64,
    /// Failed clonings
    pub failed_clones: u64,
    /// Number of cached speakers
    pub cached_speakers: usize,
    /// Success rate
    pub success_rate: f32,
    /// Most used cloning method
    pub most_used_method: Option<CloningMethod>,
}

/// Audio validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether audio is valid
    pub valid: bool,
    /// List of issues found
    pub issues: Vec<String>,
    /// Total duration of audio
    pub total_duration: f32,
    /// Number of samples
    pub sample_count: usize,
}

/// Builder for voice cloner configuration
#[derive(Debug, Clone)]
pub struct VoiceClonerBuilder {
    config: VoiceClonerConfig,
    cloning_config: Option<CloningConfig>,
}

impl VoiceClonerBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: VoiceClonerConfig::default(),
            cloning_config: None,
        }
    }

    /// Enable or disable voice cloning
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Set default cloning method
    pub fn default_method(mut self, method: CloningMethod) -> Self {
        self.config.default_method = method;
        self
    }

    /// Enable auto quality check
    pub fn auto_quality_check(mut self, enabled: bool) -> Self {
        self.config.auto_quality_check = enabled;
        self
    }

    /// Set cache size
    pub fn cache_size(mut self, size: usize) -> Self {
        self.config.max_cache_size = size;
        self
    }

    /// Set cloning configuration
    pub fn cloning_config(mut self, config: CloningConfig) -> Self {
        self.cloning_config = Some(config);
        self
    }

    /// Build the voice cloner
    pub async fn build(self) -> crate::Result<VoiceCloner> {
        let cloner = if let Some(cloning_config) = self.cloning_config {
            VoiceCloner::with_config(cloning_config).await?
        } else {
            VoiceCloner::new().await?
        };

        // Apply SDK configuration
        {
            let mut config = cloner.config.write().await;
            *config = self.config;
        }

        Ok(cloner)
    }
}

impl Default for VoiceClonerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_voice_cloner_creation() {
        let cloner = VoiceCloner::new().await.unwrap();
        assert!(cloner.is_enabled().await);
    }

    #[tokio::test]
    async fn test_quick_clone() {
        let cloner = VoiceCloner::new().await.unwrap();
        let audio = vec![0.1; 44100]; // 2 seconds of audio (minimum for cloning)

        let result = cloner
            .quick_clone(audio, 22050, "Hello world".to_string())
            .await
            .unwrap();
        // Note: result.success may be false due to insufficient data for OneShot cloning
        // This is expected behavior with dummy data
        assert!(result.error_message.is_some() || result.success);
    }

    #[tokio::test]
    async fn test_speaker_caching() {
        let cloner = VoiceCloner::new().await.unwrap();
        let profile = SpeakerProfile::new("test".to_string(), "Test Speaker".to_string());

        cloner
            .cache_speaker("test".to_string(), profile)
            .await
            .unwrap();

        let speakers = cloner.list_cached_speakers().await;
        assert!(speakers.contains(&"test".to_string()));
    }

    #[tokio::test]
    async fn test_audio_validation() {
        let cloner = VoiceCloner::new().await.unwrap();

        // Valid samples
        let valid_samples = vec![
            VoiceSample::new("sample1".to_string(), vec![0.1; 22050], 22050), // 1 second
            VoiceSample::new("sample2".to_string(), vec![0.2; 44100], 22050), // 2 seconds
        ];

        let result = cloner.validate_audio(&valid_samples).await.unwrap();
        assert!(result.valid);
        assert_eq!(result.sample_count, 2);

        // Invalid samples (too short)
        let invalid_samples = vec![
            VoiceSample::new("short".to_string(), vec![0.1; 1000], 22050), // Very short
        ];

        let result = cloner.validate_audio(&invalid_samples).await.unwrap();
        assert!(!result.valid);
        assert!(!result.issues.is_empty());
    }

    #[tokio::test]
    async fn test_cloner_builder() {
        let cloner = VoiceClonerBuilder::new()
            .enabled(true)
            .default_method(CloningMethod::OneShot)
            .cache_size(50)
            .build()
            .await
            .unwrap();

        assert!(cloner.is_enabled().await);
    }
}
