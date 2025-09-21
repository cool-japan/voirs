//! Emotion control integration for VoiRS SDK

use crate::{Result, VoirsError};
use std::sync::Arc;
use tokio::sync::RwLock;

// Re-export emotion types for convenience
pub use voirs_emotion::prelude::*;

/// SDK-integrated emotion controller
#[derive(Debug, Clone)]
pub struct EmotionController {
    /// Internal emotion processor
    processor: Arc<voirs_emotion::EmotionProcessor>,
    /// SDK-specific configuration
    config: Arc<RwLock<EmotionControllerConfig>>,
}

/// Configuration for SDK emotion controller
#[derive(Debug, Clone)]
pub struct EmotionControllerConfig {
    /// Enable emotion processing by default
    pub enabled: bool,
    /// Default emotion intensity
    pub default_intensity: f32,
    /// Auto-adjust based on text sentiment
    pub auto_sentiment_detection: bool,
    /// Cache emotion states
    pub cache_emotions: bool,
}

impl EmotionController {
    /// Create new emotion controller
    pub async fn new() -> Result<Self> {
        let processor = voirs_emotion::EmotionProcessor::new()
            .map_err(|e| VoirsError::model_error(format!("Emotion processor: {}", e)))?;

        Ok(Self {
            processor: Arc::new(processor),
            config: Arc::new(RwLock::new(EmotionControllerConfig::default())),
        })
    }

    /// Create with custom configuration
    pub async fn with_config(emotion_config: EmotionConfig) -> Result<Self> {
        let processor = voirs_emotion::EmotionProcessor::with_config(emotion_config)
            .map_err(|e| VoirsError::model_error(format!("Emotion processor: {}", e)))?;

        Ok(Self {
            processor: Arc::new(processor),
            config: Arc::new(RwLock::new(EmotionControllerConfig::default())),
        })
    }

    /// Set emotion for synthesis
    pub async fn set_emotion(&self, emotion: Emotion, intensity: Option<f32>) -> Result<()> {
        let config = self.config.read().await;
        if !config.enabled {
            return Ok(());
        }

        let intensity = intensity.unwrap_or(config.default_intensity);
        self.processor
            .set_emotion(emotion, Some(intensity))
            .await
            .map_err(|e| VoirsError::audio_error(format!("Set emotion: {}", e)))
    }

    /// Set multiple emotions with weights
    pub async fn set_emotion_mix(
        &self,
        emotions: std::collections::HashMap<Emotion, f32>,
    ) -> Result<()> {
        let config = self.config.read().await;
        if !config.enabled {
            return Ok(());
        }

        self.processor
            .set_emotion_mix(emotions)
            .await
            .map_err(|e| VoirsError::audio_error(format!("Set emotion mix: {}", e)))
    }

    /// Apply emotion from preset
    pub async fn apply_preset(&self, preset_name: &str, intensity: Option<f32>) -> Result<()> {
        let config = self.config.read().await;
        if !config.enabled {
            return Ok(());
        }

        // Get preset from library
        let library = EmotionPresetLibrary::with_defaults();
        let params = library
            .get_preset_parameters(preset_name, intensity)
            .ok_or_else(|| VoirsError::ConfigError {
                field: "preset".to_string(),
                message: format!("Emotion preset: {}", preset_name),
            })?;

        self.processor
            .apply_emotion_parameters(params)
            .await
            .map_err(|e| VoirsError::audio_error(format!("Apply preset: {}", e)))
    }

    /// Get current emotion parameters
    pub async fn get_current_parameters(&self) -> Result<EmotionParameters> {
        Ok(self.processor.get_current_parameters().await)
    }

    /// Reset to neutral emotion
    pub async fn reset_to_neutral(&self) -> Result<()> {
        self.processor
            .reset_to_neutral()
            .await
            .map_err(|e| VoirsError::audio_error(format!("Reset emotion: {}", e)))
    }

    /// Update emotion transition (call regularly for smooth transitions)
    pub async fn update(&self, delta_time_ms: f64) -> Result<()> {
        self.processor
            .update_transition(delta_time_ms)
            .await
            .map_err(|e| VoirsError::audio_error(format!("Update emotion: {}", e)))
    }

    /// Enable or disable emotion processing
    pub async fn set_enabled(&self, enabled: bool) -> Result<()> {
        let mut config = self.config.write().await;
        config.enabled = enabled;
        Ok(())
    }

    /// Check if emotion processing is enabled
    pub async fn is_enabled(&self) -> bool {
        let config = self.config.read().await;
        config.enabled
    }

    /// List available emotion presets
    pub fn list_presets(&self) -> Vec<String> {
        let library = EmotionPresetLibrary::with_defaults();
        library.list_presets()
    }

    /// Get presets by category
    pub fn get_presets_by_category(&self, category: &str) -> Vec<String> {
        let library = EmotionPresetLibrary::with_defaults();
        library
            .get_category(category)
            .iter()
            .map(|preset| preset.name.clone())
            .collect()
    }

    /// Auto-detect emotion from text sentiment (placeholder)
    pub async fn auto_detect_emotion(&self, text: &str) -> Result<Option<(Emotion, f32)>> {
        let config = self.config.read().await;
        if !config.auto_sentiment_detection {
            return Ok(None);
        }

        // Placeholder sentiment analysis - in reality would use NLP
        let emotion = if text.contains('!') {
            Some((Emotion::Excited, 0.7))
        } else if text.contains('?') {
            Some((Emotion::Neutral, 0.5))
        } else if text.to_lowercase().contains("sad") || text.to_lowercase().contains("sorry") {
            Some((Emotion::Sad, 0.6))
        } else if text.to_lowercase().contains("happy") || text.to_lowercase().contains("joy") {
            Some((Emotion::Happy, 0.7))
        } else {
            None
        };

        Ok(emotion)
    }

    /// Get emotion processing statistics
    pub async fn get_statistics(&self) -> Result<EmotionStatistics> {
        let history = self.processor.get_history().await;
        Ok(EmotionStatistics {
            total_emotions_applied: history.len(),
            current_emotion: self
                .processor
                .get_current_parameters()
                .await
                .emotion_vector
                .dominant_emotion(),
            processing_enabled: self.is_enabled().await,
        })
    }
}

impl Default for EmotionControllerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_intensity: 0.7,
            auto_sentiment_detection: false,
            cache_emotions: true,
        }
    }
}

/// Statistics for emotion processing
#[derive(Debug, Clone)]
pub struct EmotionStatistics {
    /// Total number of emotions applied
    pub total_emotions_applied: usize,
    /// Current dominant emotion
    pub current_emotion: Option<(Emotion, EmotionIntensity)>,
    /// Whether processing is enabled
    pub processing_enabled: bool,
}

/// Builder for emotion controller configuration
#[derive(Debug, Clone)]
pub struct EmotionControllerBuilder {
    config: EmotionControllerConfig,
    emotion_config: Option<EmotionConfig>,
}

impl EmotionControllerBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: EmotionControllerConfig::default(),
            emotion_config: None,
        }
    }

    /// Enable or disable emotion processing
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Set default intensity
    pub fn default_intensity(mut self, intensity: f32) -> Self {
        self.config.default_intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Enable auto sentiment detection
    pub fn auto_sentiment_detection(mut self, enabled: bool) -> Self {
        self.config.auto_sentiment_detection = enabled;
        self
    }

    /// Set emotion processing configuration
    pub fn emotion_config(mut self, config: EmotionConfig) -> Self {
        self.emotion_config = Some(config);
        self
    }

    /// Build the emotion controller
    pub async fn build(self) -> Result<EmotionController> {
        let controller = if let Some(emotion_config) = self.emotion_config {
            EmotionController::with_config(emotion_config).await?
        } else {
            EmotionController::new().await?
        };

        // Apply SDK configuration
        {
            let mut config = controller.config.write().await;
            *config = self.config;
        }

        Ok(controller)
    }
}

impl Default for EmotionControllerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_emotion_controller_creation() {
        let controller = EmotionController::new().await.unwrap();
        assert!(controller.is_enabled().await);
    }

    #[tokio::test]
    async fn test_emotion_setting() {
        // Use immediate transitions for testing
        let emotion_config = EmotionConfig::builder()
            .transition_smoothing(1.0)
            .build()
            .unwrap();
        let controller = EmotionController::with_config(emotion_config)
            .await
            .unwrap();
        controller
            .set_emotion(Emotion::Happy, Some(0.8))
            .await
            .unwrap();

        let params = controller.get_current_parameters().await.unwrap();
        let dominant = params.emotion_vector.dominant_emotion();
        assert!(dominant.is_some());
    }

    #[tokio::test]
    async fn test_preset_application() {
        // Use immediate transitions for testing
        let emotion_config = EmotionConfig::builder()
            .transition_smoothing(1.0)
            .build()
            .unwrap();
        let controller = EmotionController::with_config(emotion_config)
            .await
            .unwrap();
        controller.apply_preset("happy", Some(0.7)).await.unwrap();

        let params = controller.get_current_parameters().await.unwrap();
        assert!(!params.emotion_vector.emotions.is_empty());
    }

    #[tokio::test]
    async fn test_emotion_builder() {
        let controller = EmotionControllerBuilder::new()
            .enabled(true)
            .default_intensity(0.8)
            .auto_sentiment_detection(true)
            .build()
            .await
            .unwrap();

        assert!(controller.is_enabled().await);
    }

    #[tokio::test]
    async fn test_auto_emotion_detection() {
        let controller = EmotionControllerBuilder::new()
            .auto_sentiment_detection(true)
            .build()
            .await
            .unwrap();

        let emotion = controller
            .auto_detect_emotion("I'm so happy!")
            .await
            .unwrap();
        assert!(emotion.is_some());

        let emotion = controller
            .auto_detect_emotion("This is neutral text.")
            .await
            .unwrap();
        // May or may not detect emotion for neutral text
    }

    #[tokio::test]
    async fn test_preset_listing() {
        let controller = EmotionController::new().await.unwrap();
        let presets = controller.list_presets();
        assert!(!presets.is_empty());

        let positive_presets = controller.get_presets_by_category("positive");
        assert!(!positive_presets.is_empty());
    }
}
