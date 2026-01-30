//! Builder pattern implementation for EmotionProcessor

use crate::{
    config::EmotionConfig, cultural::CulturalEmotionAdapter, custom::CustomEmotionRegistry,
    history::EmotionHistoryConfig, Result,
};

use super::processor::EmotionProcessor;

/// Builder for EmotionProcessor
#[derive(Debug)]
pub struct EmotionProcessorBuilder {
    config: EmotionConfig,
    custom_registry: Option<CustomEmotionRegistry>,
    history_config: Option<EmotionHistoryConfig>,
    cultural_adapter: Option<CulturalEmotionAdapter>,
}

impl EmotionProcessorBuilder {
    /// Create a new processor builder
    pub fn new() -> Self {
        Self {
            config: EmotionConfig::default(),
            custom_registry: None,
            history_config: None,
            cultural_adapter: None,
        }
    }

    /// Set the emotion configuration
    pub fn config(mut self, config: EmotionConfig) -> Self {
        self.config = config;
        self
    }

    /// Enable or disable emotion processing
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Set maximum number of simultaneous emotions
    pub fn max_emotions(mut self, max: usize) -> Self {
        self.config.max_emotions = max;
        self
    }

    /// Set prosody modification strength
    pub fn prosody_strength(mut self, strength: f32) -> Self {
        self.config.prosody_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set custom emotion registry
    pub fn custom_registry(mut self, registry: CustomEmotionRegistry) -> Self {
        self.custom_registry = Some(registry);
        self
    }

    /// Set emotion history configuration
    pub fn history_config(mut self, config: EmotionHistoryConfig) -> Self {
        self.history_config = Some(config);
        self
    }

    /// Set cultural emotion adapter
    pub fn cultural_adapter(mut self, adapter: CulturalEmotionAdapter) -> Self {
        self.cultural_adapter = Some(adapter);
        self
    }

    /// Build the emotion processor
    pub fn build(self) -> Result<EmotionProcessor> {
        let registry = self.custom_registry.unwrap_or_default();
        let processor = EmotionProcessor::with_config_and_registry_and_history_and_cultural(
            self.config,
            registry,
            self.history_config.unwrap_or_default(),
            self.cultural_adapter,
        )?;

        Ok(processor)
    }
}

impl Default for EmotionProcessorBuilder {
    fn default() -> Self {
        Self::new()
    }
}
