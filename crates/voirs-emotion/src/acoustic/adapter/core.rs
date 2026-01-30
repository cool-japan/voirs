//! Core acoustic emotion adapter structures and constructors

use crate::{types::Emotion, Error, Result};
use std::collections::HashMap;

use super::super::config::AcousticIntegrationConfig;
use super::super::features::{
    BaselineCharacteristics, ProsodyPatterns, SpeakerEmotionFeatures, VoiceQualityProfile,
};
use super::super::integration::EmotionSpeakerMapping;
use super::super::params::AcousticEmotionMapping;

/// Enhanced acoustic model emotion adapter
#[derive(Debug)]
pub struct AcousticEmotionAdapter {
    /// Emotion to speaker characteristic mappings
    pub(super) speaker_mappings: HashMap<String, EmotionSpeakerMapping>,
    /// Base acoustic synthesis configuration
    pub(super) base_synthesis_config: Option<Box<dyn std::any::Any + Send + Sync>>,
    /// Emotion-to-acoustic parameter mappings
    pub(super) emotion_acoustic_mappings: HashMap<Emotion, AcousticEmotionMapping>,
    /// Integration configuration
    pub(super) integration_config: AcousticIntegrationConfig,
}

impl AcousticEmotionAdapter {
    /// Create new enhanced acoustic emotion adapter
    pub fn new() -> Self {
        Self {
            speaker_mappings: HashMap::new(),
            base_synthesis_config: None,
            emotion_acoustic_mappings: super::mappings::create_default_emotion_mappings(),
            integration_config: AcousticIntegrationConfig::default(),
        }
    }

    /// Create adapter with custom integration configuration
    pub fn with_config(config: AcousticIntegrationConfig) -> Self {
        Self {
            speaker_mappings: HashMap::new(),
            base_synthesis_config: None,
            emotion_acoustic_mappings: super::mappings::create_default_emotion_mappings(),
            integration_config: config,
        }
    }

    /// Set base acoustic synthesis configuration (placeholder for future integration)
    pub fn with_base_synthesis_config<T: std::any::Any + Send + Sync>(mut self, config: T) -> Self {
        self.base_synthesis_config = Some(Box::new(config));
        self
    }

    /// Get emotion mapping for a specific emotion
    pub fn get_emotion_mapping(&self, emotion: &Emotion) -> Option<&AcousticEmotionMapping> {
        self.emotion_acoustic_mappings.get(emotion)
    }

    /// Set custom emotion mapping
    pub fn set_emotion_mapping(&mut self, emotion: Emotion, mapping: AcousticEmotionMapping) {
        self.emotion_acoustic_mappings.insert(emotion, mapping);
    }

    /// Remove emotion mapping
    pub fn remove_emotion_mapping(&mut self, emotion: &Emotion) -> Option<AcousticEmotionMapping> {
        self.emotion_acoustic_mappings.remove(emotion)
    }

    /// Get integration configuration
    pub fn get_integration_config(&self) -> &AcousticIntegrationConfig {
        &self.integration_config
    }

    /// Set integration configuration
    pub fn set_integration_config(&mut self, config: AcousticIntegrationConfig) {
        self.integration_config = config;
    }

    /// Add emotion-speaker mapping
    pub fn add_speaker_mapping(&mut self, emotion_name: String, mapping: EmotionSpeakerMapping) {
        self.speaker_mappings.insert(emotion_name, mapping);
    }

    /// Get available speaker mappings
    pub fn get_speaker_mappings(&self) -> &HashMap<String, EmotionSpeakerMapping> {
        &self.speaker_mappings
    }

    /// Remove speaker mapping
    pub fn remove_speaker_mapping(&mut self, emotion_name: &str) -> Option<EmotionSpeakerMapping> {
        self.speaker_mappings.remove(emotion_name)
    }
}

impl Default for AcousticEmotionAdapter {
    fn default() -> Self {
        Self::new()
    }
}
