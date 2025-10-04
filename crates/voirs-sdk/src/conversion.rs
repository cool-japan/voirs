//! Voice conversion integration for VoiRS SDK

use crate::VoirsError;
use std::sync::Arc;
use tokio::sync::RwLock;

// Re-export conversion types for convenience
pub use voirs_conversion::prelude::*;
pub use voirs_conversion::types::{AgeGroup, Gender};

/// SDK-integrated voice converter
#[derive(Debug, Clone)]
pub struct VoiceConverter {
    /// Internal voice converter
    converter: Arc<voirs_conversion::VoiceConverter>,
    /// SDK-specific configuration
    config: Arc<RwLock<VoiceConverterConfig>>,
    /// Cached conversion targets
    target_cache: Arc<RwLock<std::collections::HashMap<String, ConversionTarget>>>,
}

/// Configuration for SDK voice converter
#[derive(Debug, Clone)]
pub struct VoiceConverterConfig {
    /// Enable voice conversion by default
    pub enabled: bool,
    /// Default conversion type
    pub default_conversion_type: ConversionType,
    /// Enable real-time conversion
    pub realtime_enabled: bool,
    /// Cache conversion targets
    pub cache_targets: bool,
    /// Maximum cache size
    pub max_cache_size: usize,
    /// Quality level
    pub quality_level: f32,
}

impl VoiceConverter {
    /// Create new voice converter
    pub async fn new() -> crate::Result<Self> {
        let converter = voirs_conversion::VoiceConverter::new()
            .map_err(|e| VoirsError::model_error(format!("Voice converter: {}", e)))?;

        Ok(Self {
            converter: Arc::new(converter),
            config: Arc::new(RwLock::new(VoiceConverterConfig::default())),
            target_cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
        })
    }

    /// Create with custom configuration
    pub async fn with_config(conversion_config: ConversionConfig) -> crate::Result<Self> {
        let converter = voirs_conversion::VoiceConverter::with_config(conversion_config)
            .map_err(|e| VoirsError::model_error(format!("Voice converter: {}", e)))?;

        Ok(Self {
            converter: Arc::new(converter),
            config: Arc::new(RwLock::new(VoiceConverterConfig::default())),
            target_cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
        })
    }

    /// Convert voice with specified target
    pub async fn convert_voice(
        &self,
        source_audio: Vec<f32>,
        source_sample_rate: u32,
        target: ConversionTarget,
        conversion_type: Option<ConversionType>,
    ) -> crate::Result<ConversionResult> {
        let config = self.config.read().await;
        if !config.enabled {
            return Err(VoirsError::ConfigError {
                field: "feature".to_string(),
                message: "Voice conversion is disabled".to_string(),
            });
        }

        let conversion_type = conversion_type.unwrap_or(config.default_conversion_type.clone());

        // Create conversion request
        let request = ConversionRequest::new(
            format!("convert_{}", fastrand::u64(..)),
            source_audio,
            source_sample_rate,
            conversion_type,
            target,
        )
        .with_realtime(config.realtime_enabled)
        .with_quality_level(config.quality_level);

        // Perform conversion
        self.converter
            .convert(request)
            .await
            .map_err(|e| VoirsError::audio_error(format!("Voice conversion: {}", e)))
    }

    /// Convert to different age
    pub async fn convert_age(
        &self,
        source_audio: Vec<f32>,
        source_sample_rate: u32,
        target_age: AgeGroup,
    ) -> crate::Result<ConversionResult> {
        let characteristics = VoiceCharacteristics::for_age(target_age);
        let target = ConversionTarget::new(characteristics);

        self.convert_voice(
            source_audio,
            source_sample_rate,
            target,
            Some(ConversionType::AgeTransformation),
        )
        .await
    }

    /// Convert to different gender
    pub async fn convert_gender(
        &self,
        source_audio: Vec<f32>,
        source_sample_rate: u32,
        target_gender: Gender,
    ) -> crate::Result<ConversionResult> {
        let characteristics = VoiceCharacteristics::for_gender(target_gender);
        let target = ConversionTarget::new(characteristics);

        self.convert_voice(
            source_audio,
            source_sample_rate,
            target,
            Some(ConversionType::GenderTransformation),
        )
        .await
    }

    /// Apply pitch shift
    pub async fn pitch_shift(
        &self,
        source_audio: Vec<f32>,
        source_sample_rate: u32,
        pitch_factor: f32,
    ) -> crate::Result<ConversionResult> {
        let mut characteristics = VoiceCharacteristics::new();
        characteristics.pitch.mean_f0 *= pitch_factor;
        let target = ConversionTarget::new(characteristics);

        self.convert_voice(
            source_audio,
            source_sample_rate,
            target,
            Some(ConversionType::PitchShift),
        )
        .await
    }

    /// Apply speed transformation
    pub async fn change_speed(
        &self,
        source_audio: Vec<f32>,
        source_sample_rate: u32,
        speed_factor: f32,
    ) -> crate::Result<ConversionResult> {
        let mut characteristics = VoiceCharacteristics::new();
        characteristics.timing.speaking_rate = speed_factor;
        let target = ConversionTarget::new(characteristics);

        self.convert_voice(
            source_audio,
            source_sample_rate,
            target,
            Some(ConversionType::SpeedTransformation),
        )
        .await
    }

    /// Convert using cached target
    pub async fn convert_with_cached_target(
        &self,
        source_audio: Vec<f32>,
        source_sample_rate: u32,
        target_id: &str,
    ) -> crate::Result<ConversionResult> {
        let cache = self.target_cache.read().await;
        let target = cache
            .get(target_id)
            .ok_or_else(|| VoirsError::ConfigError {
                field: "cache".to_string(),
                message: format!("Cached target not found: {}", target_id),
            })?
            .clone();

        self.convert_voice(source_audio, source_sample_rate, target, None)
            .await
    }

    /// Cache conversion target
    pub async fn cache_target(&self, target_id: String, target: ConversionTarget) -> crate::Result<()> {
        let config = self.config.read().await;
        let mut cache = self.target_cache.write().await;

        if cache.len() >= config.max_cache_size {
            // Remove oldest entry
            if let Some(oldest_key) = cache.keys().next().cloned() {
                cache.remove(&oldest_key);
            }
        }

        cache.insert(target_id, target);
        Ok(())
    }

    /// List cached targets
    pub async fn list_cached_targets(&self) -> Vec<String> {
        let cache = self.target_cache.read().await;
        cache.keys().cloned().collect()
    }

    /// Remove target from cache
    pub async fn remove_cached_target(&self, target_id: &str) -> crate::Result<()> {
        let mut cache = self.target_cache.write().await;
        cache.remove(target_id);
        Ok(())
    }

    /// Clear target cache
    pub async fn clear_cache(&self) -> crate::Result<()> {
        let mut cache = self.target_cache.write().await;
        cache.clear();
        Ok(())
    }

    /// Enable or disable voice conversion
    pub async fn set_enabled(&self, enabled: bool) -> crate::Result<()> {
        let mut config = self.config.write().await;
        config.enabled = enabled;
        Ok(())
    }

    /// Check if voice conversion is enabled
    pub async fn is_enabled(&self) -> bool {
        let config = self.config.read().await;
        config.enabled
    }

    /// Set quality level
    pub async fn set_quality_level(&self, level: f32) -> crate::Result<()> {
        let mut config = self.config.write().await;
        config.quality_level = level.clamp(0.0, 1.0);
        Ok(())
    }

    /// Get current quality level
    pub async fn get_quality_level(&self) -> f32 {
        let config = self.config.read().await;
        config.quality_level
    }

    /// Enable or disable real-time conversion
    pub async fn set_realtime_enabled(&self, enabled: bool) -> crate::Result<()> {
        let mut config = self.config.write().await;
        config.realtime_enabled = enabled;
        Ok(())
    }

    /// Check if real-time conversion is enabled
    pub async fn is_realtime_enabled(&self) -> bool {
        let config = self.config.read().await;
        config.realtime_enabled
    }

    /// Get conversion statistics
    pub async fn get_statistics(&self) -> crate::Result<ConversionStatistics> {
        let cache_size = self.target_cache.read().await.len();

        Ok(ConversionStatistics {
            cached_targets: cache_size,
            realtime_enabled: self.is_realtime_enabled().await,
            quality_level: self.get_quality_level().await,
            processing_enabled: self.is_enabled().await,
        })
    }

    /// Validate audio for conversion
    pub async fn validate_audio(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> crate::Result<AudioValidationResult> {
        let mut issues = Vec::new();
        let duration = audio.len() as f32 / sample_rate as f32;

        if audio.is_empty() {
            issues.push("Audio is empty".to_string());
        } else if duration < 0.1 {
            issues.push("Audio too short (minimum 0.1 seconds)".to_string());
        } else if duration > 300.0 {
            issues.push("Audio too long (maximum 5 minutes)".to_string());
        }

        if sample_rate < 8000 {
            issues.push("Sample rate too low (minimum 8kHz)".to_string());
        } else if sample_rate > 96000 {
            issues.push("Sample rate too high (maximum 96kHz)".to_string());
        }

        // Check for silence
        let max_amplitude = audio.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        if max_amplitude < 0.001 {
            issues.push("Audio appears to be silent".to_string());
        }

        Ok(AudioValidationResult {
            valid: issues.is_empty(),
            issues,
            duration,
            sample_rate,
            max_amplitude,
        })
    }
}

impl Default for VoiceConverterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_conversion_type: ConversionType::SpeakerConversion,
            realtime_enabled: false,
            cache_targets: true,
            max_cache_size: 50,
            quality_level: 0.8,
        }
    }
}

/// Statistics for voice conversion
#[derive(Debug, Clone)]
pub struct ConversionStatistics {
    /// Number of cached targets
    pub cached_targets: usize,
    /// Whether real-time conversion is enabled
    pub realtime_enabled: bool,
    /// Current quality level
    pub quality_level: f32,
    /// Whether processing is enabled
    pub processing_enabled: bool,
}

/// Audio validation result
#[derive(Debug, Clone)]
pub struct AudioValidationResult {
    /// Whether audio is valid
    pub valid: bool,
    /// List of issues found
    pub issues: Vec<String>,
    /// Audio duration in seconds
    pub duration: f32,
    /// Sample rate
    pub sample_rate: u32,
    /// Maximum amplitude
    pub max_amplitude: f32,
}

/// Builder for voice converter configuration
#[derive(Debug, Clone)]
pub struct VoiceConverterBuilder {
    config: VoiceConverterConfig,
    conversion_config: Option<ConversionConfig>,
}

impl VoiceConverterBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: VoiceConverterConfig::default(),
            conversion_config: None,
        }
    }

    /// Enable or disable voice conversion
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Set default conversion type
    pub fn default_conversion_type(mut self, conversion_type: ConversionType) -> Self {
        self.config.default_conversion_type = conversion_type;
        self
    }

    /// Enable real-time conversion
    pub fn realtime_enabled(mut self, enabled: bool) -> Self {
        self.config.realtime_enabled = enabled;
        self
    }

    /// Set quality level
    pub fn quality_level(mut self, level: f32) -> Self {
        self.config.quality_level = level.clamp(0.0, 1.0);
        self
    }

    /// Set cache size
    pub fn cache_size(mut self, size: usize) -> Self {
        self.config.max_cache_size = size;
        self
    }

    /// Set conversion configuration
    pub fn conversion_config(mut self, config: ConversionConfig) -> Self {
        self.conversion_config = Some(config);
        self
    }

    /// Build the voice converter
    pub async fn build(self) -> crate::Result<VoiceConverter> {
        let converter = if let Some(conversion_config) = self.conversion_config {
            VoiceConverter::with_config(conversion_config).await?
        } else {
            VoiceConverter::new().await?
        };

        // Apply SDK configuration
        {
            let mut config = converter.config.write().await;
            *config = self.config;
        }

        Ok(converter)
    }
}

impl Default for VoiceConverterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_voice_converter_creation() {
        let converter = VoiceConverter::new().await.unwrap();
        assert!(converter.is_enabled().await);
    }

    #[tokio::test]
    async fn test_age_conversion() {
        let converter = VoiceConverter::new().await.unwrap();
        let audio = vec![0.1; 22050]; // 1 second of audio

        let result = converter
            .convert_age(audio, 22050, AgeGroup::Child)
            .await
            .unwrap();
        assert!(result.success);
    }

    #[tokio::test]
    async fn test_gender_conversion() {
        let converter = VoiceConverter::new().await.unwrap();
        let audio = vec![0.1; 22050]; // 1 second of audio

        let result = converter
            .convert_gender(audio, 22050, Gender::Female)
            .await
            .unwrap();
        assert!(result.success);
    }

    #[tokio::test]
    async fn test_pitch_shift() {
        let converter = VoiceConverter::new().await.unwrap();
        let audio = vec![0.1; 22050]; // 1 second of audio

        let result = converter.pitch_shift(audio, 22050, 1.2).await.unwrap(); // 20% higher
        assert!(result.success);
    }

    #[tokio::test]
    async fn test_target_caching() {
        let converter = VoiceConverter::new().await.unwrap();
        let characteristics = VoiceCharacteristics::for_age(AgeGroup::Teen);
        let target = ConversionTarget::new(characteristics);

        converter
            .cache_target("teen_voice".to_string(), target)
            .await
            .unwrap();

        let targets = converter.list_cached_targets().await;
        assert!(targets.contains(&"teen_voice".to_string()));
    }

    #[tokio::test]
    async fn test_audio_validation() {
        let converter = VoiceConverter::new().await.unwrap();

        // Valid audio
        let valid_audio = vec![0.1; 22050]; // 1 second
        let result = converter.validate_audio(&valid_audio, 22050).await.unwrap();
        assert!(result.valid);

        // Invalid audio (empty)
        let invalid_audio = vec![];
        let result = converter
            .validate_audio(&invalid_audio, 22050)
            .await
            .unwrap();
        assert!(!result.valid);
        assert!(!result.issues.is_empty());
    }

    #[tokio::test]
    async fn test_converter_builder() {
        let converter = VoiceConverterBuilder::new()
            .enabled(true)
            .default_conversion_type(ConversionType::PitchShift)
            .realtime_enabled(true)
            .quality_level(0.9)
            .build()
            .await
            .unwrap();

        assert!(converter.is_enabled().await);
        assert!(converter.is_realtime_enabled().await);
        assert_eq!(converter.get_quality_level().await, 0.9);
    }
}
