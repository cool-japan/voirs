//! Voice management module with modular architecture.
//!
//! This module provides comprehensive voice management capabilities organized into
//! modular components:
//!
//! - [`discovery`] - Voice discovery, search, and registry functionality
//! - [`switching`] - Voice switching and management functionality  
//! - [`info`] - Voice information and metadata utilities
//!
//! # Example
//!
//! ```no_run
//! use voirs_sdk::voice::{DefaultVoiceManager, VoiceSearchCriteria};
//! use voirs_sdk::types::{LanguageCode, Gender};
//!
//! # async fn example() -> voirs_sdk::Result<()> {
//! // Create a voice manager
//! let mut manager = DefaultVoiceManager::new("/path/to/models");
//!
//! // Search for voices
//! let criteria = VoiceSearchCriteria::new()
//!     .language(LanguageCode::EnUs)
//!     .gender(Gender::Female);
//! let voice_id = {
//!     let voices = manager.search(&criteria);
//!     voices.first().map(|v| v.id.clone())
//! };
//!
//! // Switch to a voice
//! if let Some(id) = voice_id {
//!     manager.switch_to_voice(&id).await?;
//! }
//! # Ok(())
//! # }
//! ```

pub mod discovery;
pub mod info;
pub mod switching;

// Re-export the main types for convenience
pub use discovery::{VoiceRegistry, VoiceRegistryStats, VoiceSearchCriteria};
pub use info::{
    ModelInfo, SystemRequirements, VoiceComparator, VoiceComparison, VoiceCompatibility,
    VoiceComplexity, VoiceFeature, VoiceInfo, VoiceMetrics, VoiceSelectionCriteria, VoiceSummary,
    VoiceUsageStats,
};
pub use switching::{
    ConcurrentVoiceManager, DefaultVoiceManager, VoiceSwitch, VoiceValidationResult,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        AgeRange, Gender, LanguageCode, QualityLevel, SpeakingStyle, VoiceCharacteristics,
        VoiceConfig,
    };
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_integrated_voice_workflow() {
        let temp_dir = tempdir().unwrap();
        let mut manager = DefaultVoiceManager::new(temp_dir.path());
        // Disable test mode to test actual download and file system behavior
        manager.set_test_mode(false);

        // Test discovery and get first available voice ID
        let voice_id = {
            let criteria = VoiceSearchCriteria::new()
                .language(LanguageCode::EnUs)
                .gender(Gender::Female);
            let voices = manager.search(&criteria);
            assert!(!voices.is_empty());
            voices.first().unwrap().id.clone()
        };

        // Test switching (with download enabled)
        manager.set_download_enabled(true);
        let result = manager.switch_to_voice(&voice_id).await;
        assert!(result.is_ok());

        // Verify voice is now current
        assert_eq!(manager.current_voice(), Some(voice_id.as_str()));

        // Test voice validation
        let validation = manager.validate_voice_models(&voice_id).unwrap();
        assert!(validation.valid); // Should be valid after download
    }

    #[test]
    fn test_voice_registry_management() {
        let mut registry = VoiceRegistry::new();
        let initial_count = registry.list_voices().len();

        // Add a custom voice
        let custom_voice = VoiceConfig {
            id: "custom-test-voice".to_string(),
            name: "Custom Test Voice".to_string(),
            language: LanguageCode::EnUs,
            characteristics: VoiceCharacteristics {
                gender: Some(Gender::Male),
                age: Some(AgeRange::YoungAdult),
                style: SpeakingStyle::Casual,
                emotion_support: true,
                quality: QualityLevel::Ultra,
            },
            model_config: Default::default(),
            metadata: Default::default(),
        };

        registry.register_voice(custom_voice);
        assert_eq!(registry.list_voices().len(), initial_count + 1);

        // Test statistics
        let stats = registry.get_statistics();
        assert_eq!(stats.total_voices, initial_count + 1);
        assert!(stats.male_voices > 0);
        assert!(stats.emotion_support_voices > 0);

        // Test removal
        let removed = registry.remove_voice("custom-test-voice");
        assert!(removed.is_some());
        assert_eq!(registry.list_voices().len(), initial_count);
    }

    #[test]
    fn test_voice_comparison_and_selection() {
        // Create two test voices with different characteristics
        let voice1_config = VoiceConfig {
            id: "voice1".to_string(),
            name: "Voice 1".to_string(),
            language: LanguageCode::EnUs,
            characteristics: VoiceCharacteristics {
                gender: Some(Gender::Female),
                age: Some(AgeRange::Adult),
                style: SpeakingStyle::Neutral,
                emotion_support: true,
                quality: QualityLevel::Ultra,
            },
            model_config: Default::default(),
            metadata: Default::default(),
        };

        let mut voice2_config = voice1_config.clone();
        voice2_config.id = "voice2".to_string();
        voice2_config.name = "Voice 2".to_string();
        voice2_config.characteristics.quality = QualityLevel::Medium;
        voice2_config.characteristics.emotion_support = false;
        voice2_config.model_config.device_requirements.min_memory_mb = 256;

        let voice1_info = VoiceInfo::from_config(voice1_config);
        let voice2_info = VoiceInfo::from_config(voice2_config);

        // Test comparison
        let comparison = VoiceComparator::compare(&voice1_info, &voice2_info);
        assert!(comparison.quality_diff > 0.0); // voice1 has higher quality
        assert!(comparison.memory_diff > 0); // voice1 uses more memory

        // Test voice selection
        let voices = vec![voice1_info, voice2_info];

        // Select best voice with quality preference
        let criteria = VoiceSelectionCriteria {
            require_emotion_support: Some(true),
            prioritize_quality: true,
            ..Default::default()
        };

        let best = VoiceComparator::find_best_voice(&voices, &criteria);
        assert!(best.is_some());
        assert_eq!(best.unwrap().id(), "voice1"); // Should prefer voice1 due to emotion support and quality

        // Select best voice with memory constraint
        let criteria = VoiceSelectionCriteria {
            max_memory_mb: Some(300),
            prioritize_memory_efficiency: true,
            ..Default::default()
        };

        let best = VoiceComparator::find_best_voice(&voices, &criteria);
        assert!(best.is_some());
        assert_eq!(best.unwrap().id(), "voice2"); // Should prefer voice2 due to lower memory usage
    }

    #[tokio::test]
    async fn test_concurrent_voice_manager() {
        let temp_dir = tempdir().unwrap();
        let manager = ConcurrentVoiceManager::new(temp_dir.path());

        // Test concurrent read access
        let current1 = manager.current_voice().await;
        let current2 = manager.current_voice().await;
        assert_eq!(current1, current2);

        // Test configuration changes
        {
            let mut write_guard = manager.write().await;
            write_guard.set_download_enabled(false);
        }

        // Verify change persisted
        {
            let read_guard = manager.read().await;
            assert!(!read_guard.is_download_enabled());
        }
    }

    #[test]
    fn test_voice_info_serialization() {
        let voice_config = VoiceConfig {
            id: "test-voice".to_string(),
            name: "Test Voice".to_string(),
            language: LanguageCode::EnUs,
            characteristics: VoiceCharacteristics {
                gender: Some(Gender::Female),
                age: Some(AgeRange::Adult),
                style: SpeakingStyle::Neutral,
                emotion_support: true,
                quality: QualityLevel::High,
            },
            model_config: Default::default(),
            metadata: Default::default(),
        };

        let voice_info = VoiceInfo::from_config(voice_config);

        // Test JSON serialization
        let json = voice_info.to_json().unwrap();
        assert!(json.contains("test-voice"));
        assert!(json.contains("Test Voice"));

        // Test deserialization
        let restored = VoiceInfo::from_json(&json).unwrap();
        assert_eq!(restored.id(), voice_info.id());
        assert_eq!(restored.name(), voice_info.name());
        assert_eq!(restored.language(), voice_info.language());
    }

    #[test]
    fn test_advanced_voice_search() {
        let registry = VoiceRegistry::new();

        // Test search with memory constraints
        let criteria = VoiceSearchCriteria::new()
            .max_memory_mb(600)
            .min_quality(QualityLevel::Medium);
        let results = registry.find_voices(&criteria);

        for voice in results {
            assert!(voice.model_config.device_requirements.min_memory_mb <= 600);
            let quality_valid = matches!(
                voice.characteristics.quality,
                QualityLevel::Medium | QualityLevel::High | QualityLevel::Ultra
            );
            assert!(quality_valid);
        }

        // Test text search
        let criteria = VoiceSearchCriteria::new().query("female");
        let results = registry.find_voices(&criteria);

        for voice in results {
            let matches_name = voice.name.to_lowercase().contains("female");
            let matches_id = voice.id.to_lowercase().contains("female");
            let matches_desc = voice
                .metadata
                .get("description")
                .map(|desc| desc.to_lowercase().contains("female"))
                .unwrap_or(false);

            assert!(matches_name || matches_id || matches_desc);
        }
    }

    #[test]
    fn test_voice_features() {
        let voice_config = VoiceConfig {
            id: "feature-test-voice".to_string(),
            name: "Feature Test Voice".to_string(),
            language: LanguageCode::EnUs,
            characteristics: VoiceCharacteristics {
                gender: Some(Gender::Female),
                age: Some(AgeRange::Adult),
                style: SpeakingStyle::Neutral,
                emotion_support: true,
                quality: QualityLevel::Ultra,
            },
            model_config: Default::default(),
            metadata: Default::default(),
        };

        let voice_info = VoiceInfo::from_config(voice_config);

        // Test feature support
        assert!(voice_info.supports_feature(VoiceFeature::EmotionSupport));
        assert!(voice_info.supports_feature(VoiceFeature::HighQuality));
        assert!(voice_info.supports_feature(VoiceFeature::LowMemory)); // Default model config has 512MB
        assert!(!voice_info.supports_feature(VoiceFeature::GpuAcceleration)); // Default is false

        // Test summary generation
        let summary = voice_info.summary();
        assert_eq!(summary.id, "feature-test-voice");
        assert_eq!(summary.language, LanguageCode::EnUs);
        assert_eq!(summary.gender, Some(Gender::Female));
        assert!(summary.emotion_support);
        assert_eq!(summary.quality, QualityLevel::Ultra);
    }
}
