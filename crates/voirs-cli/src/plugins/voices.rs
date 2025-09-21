use super::{Plugin, PluginError, PluginResult, PluginType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use voirs_sdk::types::{AgeRange, Gender, QualityLevel, SpeakingStyle};
use voirs_sdk::voice::VoiceInfo;
use voirs_sdk::VoiceCharacteristics;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoicePluginConfig {
    pub voice_id: String,
    pub language: String,
    pub gender: String,
    pub style: String,
    pub speed_multiplier: f32,
    pub pitch_shift: f32,
    pub volume_gain: f32,
    pub emotion: Option<String>,
    pub custom_parameters: HashMap<String, serde_json::Value>,
}

impl Default for VoicePluginConfig {
    fn default() -> Self {
        Self {
            voice_id: "default".to_string(),
            language: "en-US".to_string(),
            gender: "neutral".to_string(),
            style: "standard".to_string(),
            speed_multiplier: 1.0,
            pitch_shift: 0.0,
            volume_gain: 0.0,
            emotion: None,
            custom_parameters: HashMap::new(),
        }
    }
}

pub trait VoicePlugin: Plugin {
    fn synthesize(&self, text: &str, config: &VoicePluginConfig) -> PluginResult<Vec<f32>>;
    fn get_voice_info(&self) -> VoiceInfo;
    fn get_supported_languages(&self) -> Vec<String>;
    fn get_voice_characteristics(&self) -> VoiceCharacteristics;
    fn supports_ssml(&self) -> bool;
    fn supports_emotions(&self) -> bool;
    fn get_supported_emotions(&self) -> Vec<String>;
    fn get_sample_rate(&self) -> u32;
    fn get_quality_levels(&self) -> Vec<String>;
    fn validate_text(&self, text: &str) -> PluginResult<()>;
    fn estimate_duration(&self, text: &str, config: &VoicePluginConfig) -> PluginResult<f32>;
}

pub struct VoicePluginManager {
    voices: HashMap<String, Arc<dyn VoicePlugin>>,
    voice_configs: HashMap<String, VoicePluginConfig>,
    active_voice: Option<String>,
}

impl VoicePluginManager {
    pub fn new() -> Self {
        Self {
            voices: HashMap::new(),
            voice_configs: HashMap::new(),
            active_voice: None,
        }
    }

    pub fn register_voice(&mut self, voice_id: String, voice: Arc<dyn VoicePlugin>) {
        self.voices.insert(voice_id.clone(), voice);
        self.voice_configs
            .insert(voice_id, VoicePluginConfig::default());
    }

    pub fn unregister_voice(&mut self, voice_id: &str) -> bool {
        let removed = self.voices.remove(voice_id).is_some();
        self.voice_configs.remove(voice_id);

        if self.active_voice.as_ref() == Some(&voice_id.to_string()) {
            self.active_voice = None;
        }

        removed
    }

    pub fn list_voices(&self) -> Vec<String> {
        self.voices.keys().cloned().collect()
    }

    pub fn get_voice(&self, voice_id: &str) -> Option<Arc<dyn VoicePlugin>> {
        self.voices.get(voice_id).cloned()
    }

    pub fn set_active_voice(&mut self, voice_id: &str) -> PluginResult<()> {
        if self.voices.contains_key(voice_id) {
            self.active_voice = Some(voice_id.to_string());
            Ok(())
        } else {
            Err(PluginError::NotFound(voice_id.to_string()))
        }
    }

    pub fn get_active_voice(&self) -> Option<&String> {
        self.active_voice.as_ref()
    }

    pub fn synthesize_with_voice(
        &self,
        voice_id: &str,
        text: &str,
        config: Option<&VoicePluginConfig>,
    ) -> PluginResult<Vec<f32>> {
        let voice = self
            .voices
            .get(voice_id)
            .ok_or_else(|| PluginError::NotFound(voice_id.to_string()))?;

        let default_config = VoicePluginConfig::default();
        let config = match config {
            Some(c) => c,
            None => self.voice_configs.get(voice_id).unwrap_or(&default_config),
        };

        voice.synthesize(text, config)
    }

    pub fn synthesize_with_active_voice(
        &self,
        text: &str,
        config: Option<&VoicePluginConfig>,
    ) -> PluginResult<Vec<f32>> {
        let voice_id = self
            .active_voice
            .as_ref()
            .ok_or_else(|| PluginError::ExecutionFailed("No active voice set".to_string()))?;

        self.synthesize_with_voice(voice_id, text, config)
    }

    pub fn update_voice_config(
        &mut self,
        voice_id: &str,
        config: VoicePluginConfig,
    ) -> PluginResult<()> {
        if self.voices.contains_key(voice_id) {
            self.voice_configs.insert(voice_id.to_string(), config);
            Ok(())
        } else {
            Err(PluginError::NotFound(voice_id.to_string()))
        }
    }

    pub fn get_voice_config(&self, voice_id: &str) -> Option<&VoicePluginConfig> {
        self.voice_configs.get(voice_id)
    }

    pub fn get_voice_info(&self, voice_id: &str) -> PluginResult<VoiceInfo> {
        let voice = self
            .voices
            .get(voice_id)
            .ok_or_else(|| PluginError::NotFound(voice_id.to_string()))?;

        Ok(voice.get_voice_info())
    }

    pub fn search_voices(
        &self,
        language: Option<&str>,
        gender: Option<&str>,
        style: Option<&str>,
    ) -> Vec<String> {
        self.voices
            .iter()
            .filter(|(voice_id, voice)| {
                let characteristics = voice.get_voice_characteristics();
                let voice_info = voice.get_voice_info();

                if let Some(lang) = language {
                    if !voice.get_supported_languages().contains(&lang.to_string()) {
                        return false;
                    }
                }

                if let Some(g) = gender {
                    if let Some(voice_gender) = &voice_info.config.characteristics.gender {
                        if voice_gender.to_string().to_lowercase() != g.to_lowercase() {
                            return false;
                        }
                    }
                }

                if let Some(s) = style {
                    if voice_info
                        .config
                        .characteristics
                        .style
                        .to_string()
                        .to_lowercase()
                        != s.to_lowercase()
                    {
                        return false;
                    }
                }

                true
            })
            .map(|(voice_id, _)| voice_id.clone())
            .collect()
    }

    pub fn validate_text_for_voice(&self, voice_id: &str, text: &str) -> PluginResult<()> {
        let voice = self
            .voices
            .get(voice_id)
            .ok_or_else(|| PluginError::NotFound(voice_id.to_string()))?;

        voice.validate_text(text)
    }

    pub fn estimate_synthesis_duration(
        &self,
        voice_id: &str,
        text: &str,
        config: Option<&VoicePluginConfig>,
    ) -> PluginResult<f32> {
        let voice = self
            .voices
            .get(voice_id)
            .ok_or_else(|| PluginError::NotFound(voice_id.to_string()))?;

        let default_config = VoicePluginConfig::default();
        let config = match config {
            Some(c) => c,
            None => self.voice_configs.get(voice_id).unwrap_or(&default_config),
        };

        voice.estimate_duration(text, config)
    }
}

impl Default for VoicePluginManager {
    fn default() -> Self {
        Self::new()
    }
}

// Example built-in voice plugin implementation
pub struct DefaultVoicePlugin {
    name: String,
    version: String,
    voice_id: String,
}

impl DefaultVoicePlugin {
    pub fn new(voice_id: &str) -> Self {
        Self {
            name: format!("default-voice-{}", voice_id),
            version: "1.0.0".to_string(),
            voice_id: voice_id.to_string(),
        }
    }
}

impl Plugin for DefaultVoicePlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn description(&self) -> &str {
        "Built-in default voice plugin"
    }

    fn plugin_type(&self) -> PluginType {
        PluginType::Voice
    }

    fn initialize(&mut self, _config: &serde_json::Value) -> PluginResult<()> {
        Ok(())
    }

    fn cleanup(&mut self) -> PluginResult<()> {
        Ok(())
    }

    fn get_capabilities(&self) -> Vec<String> {
        vec![
            "synthesize".to_string(),
            "get_voice_info".to_string(),
            "get_supported_languages".to_string(),
            "validate_text".to_string(),
            "estimate_duration".to_string(),
        ]
    }

    fn execute(&self, command: &str, args: &serde_json::Value) -> PluginResult<serde_json::Value> {
        match command {
            "get_voice_info" => {
                let info = self.get_voice_info();
                Ok(serde_json::to_value(info).map_err(PluginError::SerializationError)?)
            }
            "get_supported_languages" => Ok(serde_json::json!(self.get_supported_languages())),
            "validate_text" => {
                let text = args.get("text").and_then(|v| v.as_str()).ok_or_else(|| {
                    PluginError::ExecutionFailed("Missing text parameter".to_string())
                })?;

                self.validate_text(text)?;
                Ok(serde_json::json!({"valid": true}))
            }
            _ => Err(PluginError::ExecutionFailed(format!(
                "Unknown command: {}",
                command
            ))),
        }
    }
}

impl VoicePlugin for DefaultVoicePlugin {
    fn synthesize(&self, text: &str, config: &VoicePluginConfig) -> PluginResult<Vec<f32>> {
        // Placeholder implementation - generate simple sine wave based on text length
        let duration = text.len() as f32 * 0.1 * config.speed_multiplier;
        let sample_rate = self.get_sample_rate() as f32;
        let samples = (duration * sample_rate) as usize;

        let frequency = 440.0 * (2.0_f32).powf(config.pitch_shift / 12.0);
        let amplitude = 0.1 * (10.0_f32).powf(config.volume_gain / 20.0);

        let mut audio = Vec::with_capacity(samples);
        for i in 0..samples {
            let t = i as f32 / sample_rate;
            let sample = amplitude * (2.0 * std::f32::consts::PI * frequency * t).sin();
            audio.push(sample);
        }

        Ok(audio)
    }

    fn get_voice_info(&self) -> VoiceInfo {
        use voirs_sdk::types::{Gender, QualityLevel, SpeakingStyle};
        use voirs_sdk::VoiceConfig;

        let config = VoiceConfig {
            id: self.voice_id.clone(),
            name: format!("Default Voice {}", self.voice_id),
            characteristics: VoiceCharacteristics {
                gender: Some(Gender::NonBinary),
                age: Some(AgeRange::Adult),
                style: SpeakingStyle::Neutral,
                emotion_support: true,
                quality: QualityLevel::Medium,
            },
            language: voirs_sdk::types::LanguageCode::EnUs,
            model_config: Default::default(),
            metadata: HashMap::new(),
        };

        VoiceInfo::from_config(config)
    }

    fn get_supported_languages(&self) -> Vec<String> {
        vec!["en-US".to_string(), "en-GB".to_string()]
    }

    fn get_voice_characteristics(&self) -> VoiceCharacteristics {
        VoiceCharacteristics {
            gender: Some(Gender::NonBinary),
            age: Some(AgeRange::Adult),
            style: SpeakingStyle::Neutral,
            emotion_support: true,
            quality: QualityLevel::Medium,
        }
    }

    fn supports_ssml(&self) -> bool {
        false
    }

    fn supports_emotions(&self) -> bool {
        false
    }

    fn get_supported_emotions(&self) -> Vec<String> {
        vec![]
    }

    fn get_sample_rate(&self) -> u32 {
        22050
    }

    fn get_quality_levels(&self) -> Vec<String> {
        vec!["low".to_string(), "medium".to_string(), "high".to_string()]
    }

    fn validate_text(&self, text: &str) -> PluginResult<()> {
        if text.is_empty() {
            return Err(PluginError::ExecutionFailed(
                "Empty text not allowed".to_string(),
            ));
        }

        if text.len() > 10000 {
            return Err(PluginError::ExecutionFailed(
                "Text too long (max 10000 characters)".to_string(),
            ));
        }

        Ok(())
    }

    fn estimate_duration(&self, text: &str, config: &VoicePluginConfig) -> PluginResult<f32> {
        // Simple estimation: ~10 characters per second, adjusted by speed
        let base_duration = text.len() as f32 * 0.1;
        Ok(base_duration / config.speed_multiplier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_plugin_manager() {
        let mut manager = VoicePluginManager::new();
        let voice = Arc::new(DefaultVoicePlugin::new("test"));

        manager.register_voice("test".to_string(), voice);

        let voices = manager.list_voices();
        assert!(voices.contains(&"test".to_string()));

        let voice = manager.get_voice("test");
        assert!(voice.is_some());
    }

    #[test]
    fn test_default_voice_plugin() {
        let voice = DefaultVoicePlugin::new("test");
        assert_eq!(voice.name(), "default-voice-test");
        assert_eq!(voice.version(), "1.0.0");

        let config = VoicePluginConfig::default();
        let audio = voice.synthesize("Hello world", &config).unwrap();
        assert!(!audio.is_empty());
    }

    #[test]
    fn test_voice_validation() {
        let voice = DefaultVoicePlugin::new("test");

        // Valid text
        assert!(voice.validate_text("Hello world").is_ok());

        // Empty text should fail
        assert!(voice.validate_text("").is_err());

        // Very long text should fail
        let long_text = "a".repeat(10001);
        assert!(voice.validate_text(&long_text).is_err());
    }

    #[test]
    fn test_voice_search() {
        let mut manager = VoicePluginManager::new();
        let voice = Arc::new(DefaultVoicePlugin::new("test"));

        manager.register_voice("test".to_string(), voice);

        let results = manager.search_voices(Some("en-US"), None, None);
        assert!(results.contains(&"test".to_string()));

        let results = manager.search_voices(Some("fr-FR"), None, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_active_voice() {
        let mut manager = VoicePluginManager::new();
        let voice = Arc::new(DefaultVoicePlugin::new("test"));

        manager.register_voice("test".to_string(), voice);

        assert!(manager.set_active_voice("test").is_ok());
        assert_eq!(manager.get_active_voice(), Some(&"test".to_string()));

        let audio = manager.synthesize_with_active_voice("Hello", None).unwrap();
        assert!(!audio.is_empty());
    }
}
