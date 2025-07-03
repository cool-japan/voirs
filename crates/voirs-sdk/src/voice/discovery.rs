//! Voice discovery, search, and registry functionality.

use crate::types::{
    LanguageCode, VoiceConfig, VoiceCharacteristics, Gender, AgeRange, 
    SpeakingStyle, QualityLevel, ModelConfig, ModelFormat, DeviceRequirements
};
use std::collections::HashMap;

/// Voice registry for managing available voices
pub struct VoiceRegistry {
    /// Available voices
    pub(super) voices: HashMap<String, VoiceConfig>,
    
    /// Voice search index by language
    pub(super) language_index: HashMap<LanguageCode, Vec<String>>,
    
    /// Default voices per language
    pub(super) defaults: HashMap<LanguageCode, String>,
}

impl VoiceRegistry {
    /// Create new voice registry
    pub fn new() -> Self {
        let mut registry = Self {
            voices: HashMap::new(),
            language_index: HashMap::new(),
            defaults: HashMap::new(),
        };
        
        // Add some default voices
        registry.add_default_voices();
        registry
    }
    
    /// Register a new voice
    pub fn register_voice(&mut self, voice: VoiceConfig) {
        let voice_id = voice.id.clone();
        let language = voice.language;
        
        // Add to main registry
        self.voices.insert(voice_id.clone(), voice);
        
        // Update language index
        self.language_index
            .entry(language)
            .or_insert_with(Vec::new)
            .push(voice_id.clone());
        
        // Set as default if it's the first voice for this language
        if !self.defaults.contains_key(&language) {
            self.defaults.insert(language, voice_id);
        }
    }
    
    /// Get voice by ID
    pub fn get_voice(&self, voice_id: &str) -> Option<&VoiceConfig> {
        self.voices.get(voice_id)
    }
    
    /// List all voices
    pub fn list_voices(&self) -> Vec<&VoiceConfig> {
        self.voices.values().collect()
    }
    
    /// List voices for specific language
    pub fn voices_for_language(&self, language: LanguageCode) -> Vec<&VoiceConfig> {
        self.language_index
            .get(&language)
            .map(|voice_ids| {
                voice_ids
                    .iter()
                    .filter_map(|id| self.voices.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }
    
    /// Get default voice for language
    pub fn default_voice_for_language(&self, language: LanguageCode) -> Option<&VoiceConfig> {
        self.defaults
            .get(&language)
            .and_then(|id| self.voices.get(id))
    }
    
    /// Find voices matching criteria
    pub fn find_voices(&self, criteria: &VoiceSearchCriteria) -> Vec<&VoiceConfig> {
        self.voices
            .values()
            .filter(|voice| criteria.matches(voice))
            .collect()
    }
    
    /// Get voices grouped by language
    pub fn voices_by_language(&self) -> HashMap<LanguageCode, Vec<&VoiceConfig>> {
        let mut grouped = HashMap::new();
        
        for (language, voice_ids) in &self.language_index {
            let voices = voice_ids
                .iter()
                .filter_map(|id| self.voices.get(id))
                .collect();
            grouped.insert(*language, voices);
        }
        
        grouped
    }
    
    /// Get voice statistics
    pub fn get_statistics(&self) -> VoiceRegistryStats {
        let mut stats = VoiceRegistryStats::default();
        
        stats.total_voices = self.voices.len();
        stats.languages = self.language_index.keys().cloned().collect();
        
        // Count by characteristics
        for voice in self.voices.values() {
            // Count by gender
            match voice.characteristics.gender {
                Some(Gender::Male) => stats.male_voices += 1,
                Some(Gender::Female) => stats.female_voices += 1,
                Some(Gender::NonBinary) => stats.non_binary_voices += 1,
                None => stats.unspecified_gender += 1,
            }
            
            // Count by quality
            match voice.characteristics.quality {
                QualityLevel::Low => stats.low_quality += 1,
                QualityLevel::Medium => stats.medium_quality += 1,
                QualityLevel::High => stats.high_quality += 1,
                QualityLevel::Ultra => stats.ultra_quality += 1,
            }
            
            // Count emotion support
            if voice.characteristics.emotion_support {
                stats.emotion_support_voices += 1;
            }
        }
        
        stats
    }
    
    /// Clear all voices
    pub fn clear(&mut self) {
        self.voices.clear();
        self.language_index.clear();
        self.defaults.clear();
    }
    
    /// Remove voice by ID
    pub fn remove_voice(&mut self, voice_id: &str) -> Option<VoiceConfig> {
        if let Some(voice) = self.voices.remove(voice_id) {
            let language = voice.language;
            
            // Remove from language index
            if let Some(voice_ids) = self.language_index.get_mut(&language) {
                voice_ids.retain(|id| id != voice_id);
                if voice_ids.is_empty() {
                    self.language_index.remove(&language);
                }
            }
            
            // Update default if this was the default voice
            if self.defaults.get(&language) == Some(&voice.id) {
                self.defaults.remove(&language);
                // Set new default if there are other voices for this language
                if let Some(voice_ids) = self.language_index.get(&language) {
                    if let Some(new_default) = voice_ids.first() {
                        self.defaults.insert(language, new_default.clone());
                    }
                }
            }
            
            Some(voice)
        } else {
            None
        }
    }
    
    /// Add built-in default voices
    fn add_default_voices(&mut self) {
        // English (US) voices
        self.register_voice(VoiceConfig {
            id: "en-US-female-calm".to_string(),
            name: "English US Female Calm".to_string(),
            language: LanguageCode::EnUs,
            characteristics: VoiceCharacteristics {
                gender: Some(Gender::Female),
                age: Some(AgeRange::Adult),
                style: SpeakingStyle::Calm,
                emotion_support: false,
                quality: QualityLevel::High,
            },
            model_config: ModelConfig {
                g2p_model: Some("models/g2p/en-us.bin".to_string()),
                acoustic_model: "models/acoustic/en-us-female-calm.bin".to_string(),
                vocoder_model: "models/vocoder/hifigan-universal.bin".to_string(),
                format: ModelFormat::Candle,
                device_requirements: DeviceRequirements {
                    min_memory_mb: 512,
                    gpu_support: true,
                    compute_capabilities: vec!["cpu".to_string(), "cuda".to_string()],
                },
            },
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("description".to_string(), "High-quality female voice with calm speaking style".to_string());
                meta.insert("sample_rate".to_string(), "22050".to_string());
                meta.insert("version".to_string(), "1.0.0".to_string());
                meta
            },
        });
        
        self.register_voice(VoiceConfig {
            id: "en-US-male-news".to_string(),
            name: "English US Male News".to_string(),
            language: LanguageCode::EnUs,
            characteristics: VoiceCharacteristics {
                gender: Some(Gender::Male),
                age: Some(AgeRange::Adult),
                style: SpeakingStyle::News,
                emotion_support: false,
                quality: QualityLevel::High,
            },
            model_config: ModelConfig {
                g2p_model: Some("models/g2p/en-us.bin".to_string()),
                acoustic_model: "models/acoustic/en-us-male-news.bin".to_string(),
                vocoder_model: "models/vocoder/hifigan-universal.bin".to_string(),
                format: ModelFormat::Candle,
                device_requirements: DeviceRequirements {
                    min_memory_mb: 512,
                    gpu_support: true,
                    compute_capabilities: vec!["cpu".to_string(), "cuda".to_string()],
                },
            },
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("description".to_string(), "Professional male voice optimized for news reading".to_string());
                meta.insert("sample_rate".to_string(), "22050".to_string());
                meta.insert("version".to_string(), "1.0.0".to_string());
                meta
            },
        });
        
        // Japanese voice
        self.register_voice(VoiceConfig {
            id: "ja-JP-female-neutral".to_string(),
            name: "Japanese Female Neutral".to_string(),
            language: LanguageCode::JaJp,
            characteristics: VoiceCharacteristics {
                gender: Some(Gender::Female),
                age: Some(AgeRange::YoungAdult),
                style: SpeakingStyle::Neutral,
                emotion_support: true,
                quality: QualityLevel::High,
            },
            model_config: ModelConfig {
                g2p_model: Some("models/g2p/ja-jp.bin".to_string()),
                acoustic_model: "models/acoustic/ja-jp-female-neutral.bin".to_string(),
                vocoder_model: "models/vocoder/hifigan-japanese.bin".to_string(),
                format: ModelFormat::Candle,
                device_requirements: DeviceRequirements {
                    min_memory_mb: 768,
                    gpu_support: true,
                    compute_capabilities: vec!["cpu".to_string(), "cuda".to_string()],
                },
            },
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("description".to_string(), "Natural Japanese female voice with emotion support".to_string());
                meta.insert("sample_rate".to_string(), "22050".to_string());
                meta.insert("version".to_string(), "1.0.0".to_string());
                meta
            },
        });
    }
}

impl Default for VoiceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Voice search criteria
#[derive(Debug, Clone, Default)]
pub struct VoiceSearchCriteria {
    /// Language filter
    pub language: Option<LanguageCode>,
    
    /// Gender filter
    pub gender: Option<Gender>,
    
    /// Age range filter
    pub age_range: Option<AgeRange>,
    
    /// Speaking style filter
    pub style: Option<SpeakingStyle>,
    
    /// Minimum quality level
    pub min_quality: Option<QualityLevel>,
    
    /// Require emotion support
    pub emotion_support: Option<bool>,
    
    /// Text query for name/description search
    pub query: Option<String>,
    
    /// Maximum memory requirement (MB)
    pub max_memory_mb: Option<u32>,
    
    /// Required compute capabilities
    pub required_capabilities: Option<Vec<String>>,
}

impl VoiceSearchCriteria {
    /// Create new search criteria
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Filter by language
    pub fn language(mut self, language: LanguageCode) -> Self {
        self.language = Some(language);
        self
    }
    
    /// Filter by gender
    pub fn gender(mut self, gender: Gender) -> Self {
        self.gender = Some(gender);
        self
    }
    
    /// Filter by age range
    pub fn age_range(mut self, age: AgeRange) -> Self {
        self.age_range = Some(age);
        self
    }
    
    /// Filter by speaking style
    pub fn style(mut self, style: SpeakingStyle) -> Self {
        self.style = Some(style);
        self
    }
    
    /// Set minimum quality
    pub fn min_quality(mut self, quality: QualityLevel) -> Self {
        self.min_quality = Some(quality);
        self
    }
    
    /// Require emotion support
    pub fn with_emotion_support(mut self, required: bool) -> Self {
        self.emotion_support = Some(required);
        self
    }
    
    /// Search by text query
    pub fn query(mut self, query: impl Into<String>) -> Self {
        self.query = Some(query.into());
        self
    }
    
    /// Set maximum memory requirement
    pub fn max_memory_mb(mut self, max_memory: u32) -> Self {
        self.max_memory_mb = Some(max_memory);
        self
    }
    
    /// Set required compute capabilities
    pub fn required_capabilities(mut self, capabilities: Vec<String>) -> Self {
        self.required_capabilities = Some(capabilities);
        self
    }
    
    /// Check if voice matches criteria
    pub(crate) fn matches(&self, voice: &VoiceConfig) -> bool {
        // Language filter
        if let Some(lang) = self.language {
            if voice.language != lang {
                return false;
            }
        }
        
        // Gender filter
        if let Some(gender) = self.gender {
            if voice.characteristics.gender != Some(gender) {
                return false;
            }
        }
        
        // Age range filter
        if let Some(age) = self.age_range {
            if voice.characteristics.age != Some(age) {
                return false;
            }
        }
        
        // Style filter
        if let Some(style) = self.style {
            if voice.characteristics.style != style {
                return false;
            }
        }
        
        // Quality filter
        if let Some(min_quality) = self.min_quality {
            let quality_order = [QualityLevel::Low, QualityLevel::Medium, QualityLevel::High, QualityLevel::Ultra];
            let voice_quality_idx = quality_order.iter().position(|&q| q == voice.characteristics.quality).unwrap_or(0);
            let min_quality_idx = quality_order.iter().position(|&q| q == min_quality).unwrap_or(0);
            
            if voice_quality_idx < min_quality_idx {
                return false;
            }
        }
        
        // Emotion support filter
        if let Some(emotion_required) = self.emotion_support {
            if voice.characteristics.emotion_support != emotion_required {
                return false;
            }
        }
        
        // Memory requirement filter
        if let Some(max_memory) = self.max_memory_mb {
            if voice.model_config.device_requirements.min_memory_mb > max_memory {
                return false;
            }
        }
        
        // Compute capabilities filter
        if let Some(required_caps) = &self.required_capabilities {
            let voice_caps = &voice.model_config.device_requirements.compute_capabilities;
            if !required_caps.iter().any(|cap| voice_caps.contains(cap)) {
                return false;
            }
        }
        
        // Text query filter
        if let Some(query) = &self.query {
            let query_lower = query.to_lowercase();
            let name_match = voice.name.to_lowercase().contains(&query_lower);
            let id_match = voice.id.to_lowercase().contains(&query_lower);
            let desc_match = voice.metadata.get("description")
                .map(|desc| desc.to_lowercase().contains(&query_lower))
                .unwrap_or(false);
            
            if !name_match && !id_match && !desc_match {
                return false;
            }
        }
        
        true
    }
}

/// Voice registry statistics
#[derive(Debug, Clone, Default)]
pub struct VoiceRegistryStats {
    /// Total number of voices
    pub total_voices: usize,
    /// Available languages
    pub languages: Vec<LanguageCode>,
    /// Number of male voices
    pub male_voices: usize,
    /// Number of female voices
    pub female_voices: usize,
    /// Number of non-binary voices
    pub non_binary_voices: usize,
    /// Number of voices with unspecified gender
    pub unspecified_gender: usize,
    /// Number of low quality voices
    pub low_quality: usize,
    /// Number of medium quality voices
    pub medium_quality: usize,
    /// Number of high quality voices
    pub high_quality: usize,
    /// Number of ultra quality voices
    pub ultra_quality: usize,
    /// Number of voices with emotion support
    pub emotion_support_voices: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_registry_creation() {
        let registry = VoiceRegistry::new();
        
        // Should have default voices
        assert!(!registry.list_voices().is_empty());
        
        // Should have English voices
        let en_voices = registry.voices_for_language(LanguageCode::EnUs);
        assert!(!en_voices.is_empty());
        
        // Should have default voice for English
        let default = registry.default_voice_for_language(LanguageCode::EnUs);
        assert!(default.is_some());
    }

    #[test]
    fn test_voice_search_criteria() {
        let registry = VoiceRegistry::new();
        
        // Search by language
        let criteria = VoiceSearchCriteria::new().language(LanguageCode::EnUs);
        let results = registry.find_voices(&criteria);
        assert!(!results.is_empty());
        
        // Search by gender
        let criteria = VoiceSearchCriteria::new().gender(Gender::Female);
        let results = registry.find_voices(&criteria);
        assert!(!results.is_empty());
        
        // Combined search
        let criteria = VoiceSearchCriteria::new()
            .language(LanguageCode::EnUs)
            .gender(Gender::Male);
        let results = registry.find_voices(&criteria);
        
        // Should find male English voices
        for voice in results {
            assert_eq!(voice.language, LanguageCode::EnUs);
            assert_eq!(voice.characteristics.gender, Some(Gender::Male));
        }
    }

    #[test]
    fn test_voice_registry_statistics() {
        let registry = VoiceRegistry::new();
        let stats = registry.get_statistics();
        
        assert!(stats.total_voices > 0);
        assert!(!stats.languages.is_empty());
        assert!(stats.male_voices > 0);
        assert!(stats.female_voices > 0);
    }

    #[test]
    fn test_voice_removal() {
        let mut registry = VoiceRegistry::new();
        let initial_count = registry.voices.len();
        
        // Remove a voice
        let removed = registry.remove_voice("en-US-female-calm");
        assert!(removed.is_some());
        assert_eq!(registry.voices.len(), initial_count - 1);
        
        // Try to remove non-existent voice
        let removed = registry.remove_voice("non-existent");
        assert!(removed.is_none());
    }

    #[test]
    fn test_advanced_search_criteria() {
        let registry = VoiceRegistry::new();
        
        // Search with memory constraint
        let criteria = VoiceSearchCriteria::new().max_memory_mb(600);
        let results = registry.find_voices(&criteria);
        
        for voice in results {
            assert!(voice.model_config.device_requirements.min_memory_mb <= 600);
        }
        
        // Search with compute capability requirement
        let criteria = VoiceSearchCriteria::new()
            .required_capabilities(vec!["cpu".to_string()]);
        let results = registry.find_voices(&criteria);
        
        for voice in results {
            assert!(voice.model_config.device_requirements.compute_capabilities.contains(&"cpu".to_string()));
        }
    }

    #[test]
    fn test_voices_by_language() {
        let registry = VoiceRegistry::new();
        let grouped = registry.voices_by_language();
        
        assert!(!grouped.is_empty());
        assert!(grouped.contains_key(&LanguageCode::EnUs));
        assert!(grouped.contains_key(&LanguageCode::JaJp));
    }
}