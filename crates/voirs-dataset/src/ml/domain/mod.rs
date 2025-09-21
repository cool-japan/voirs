//! Domain adaptation module
//!
//! This module provides comprehensive domain adaptation functionality for speech synthesis
//! datasets, including cross-domain data mixing, domain-specific preprocessing, transfer
//! learning support, and domain shift detection.

pub mod adapter;
pub mod config;
pub mod types;

// Re-export main types for convenience
pub use adapter::DomainAdapterImpl;
pub use config::*;
pub use types::*;

// Re-export for backward compatibility
pub use DomainAdapterImpl as DomainAdapter;

#[cfg(test)]
mod tests {
    use super::types::DomainAdapter;
    use super::*;
    use crate::{AudioData, DatasetSample, LanguageCode, QualityMetrics, SpeakerInfo};

    fn create_test_sample(duration: f32, text: &str) -> DatasetSample {
        let samples = vec![0.0; (duration * 16000.0) as usize];
        let audio = AudioData::new(samples, 16000, 1);
        DatasetSample {
            id: "test".to_string(),
            audio,
            text: text.to_string(),
            speaker: Some(SpeakerInfo {
                id: "speaker1".to_string(),
                name: Some("Test Speaker".to_string()),
                gender: None,
                age: None,
                accent: None,
                metadata: std::collections::HashMap::new(),
            }),
            language: LanguageCode::EnUs,
            quality: QualityMetrics::default(),
            phonemes: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_domain_adapter_creation() {
        let config = DomainAdaptationConfig::default();
        let adapter = DomainAdapterImpl::new(config);
        assert!(adapter.is_ok());
    }

    #[tokio::test]
    async fn test_domain_config_compatibility() {
        let source_config = DomainConfig::new(
            "source".to_string(),
            DomainType::Studio,
            vec![LanguageCode::EnUs],
        );

        let target_config = DomainConfig::new(
            "target".to_string(),
            DomainType::Studio,
            vec![LanguageCode::EnUs],
        );

        assert!(source_config.is_compatible_with(&target_config));

        let similarity = source_config.similarity_score(&target_config);
        assert!(similarity > 0.5);
    }

    #[tokio::test]
    async fn test_domain_shift_detection() {
        let config = DomainAdaptationConfig::default();
        let adapter = DomainAdapterImpl::new(config).unwrap();

        let source_samples = vec![
            create_test_sample(1.0, "Hello world"),
            create_test_sample(1.5, "Good morning"),
        ];

        let target_samples = vec![
            create_test_sample(2.0, "Bonjour monde"),
            create_test_sample(2.5, "Bonne journée"),
        ];

        let shift = adapter
            .detect_domain_shift(&source_samples, &target_samples)
            .await;
        assert!(shift.is_ok());

        let shift = shift.unwrap();
        assert!(shift.magnitude >= 0.0);
        assert!(shift.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_domain_adaptation() {
        let config = DomainAdaptationConfig::default();
        let adapter = DomainAdapterImpl::new(config).unwrap();

        let source_samples = vec![create_test_sample(1.0, "Source text")];
        let target_samples = vec![create_test_sample(1.0, "Target text")];

        let result = adapter.adapt_domain(&source_samples, &target_samples).await;
        assert!(result.is_ok());

        let adaptation_result = result.unwrap();
        assert!(!adaptation_result.adapted_samples.is_empty());
    }

    #[tokio::test]
    async fn test_domain_statistics() {
        let config = DomainAdaptationConfig::default();
        let adapter = DomainAdapterImpl::new(config).unwrap();

        let samples = vec![
            create_test_sample(1.0, "First sample"),
            create_test_sample(1.5, "Second sample"),
            create_test_sample(2.0, "Third sample"),
        ];

        let stats = adapter.get_domain_statistics(&samples).await;
        assert!(stats.is_ok());

        let domain_stats = stats.unwrap();
        assert_eq!(domain_stats.total_samples(), 3);
        assert!(domain_stats.average_audio_quality() >= 0.0);
    }

    #[tokio::test]
    async fn test_compatibility_validation() {
        let config = DomainAdaptationConfig::default();
        let adapter = DomainAdapterImpl::new(config).unwrap();

        let source_domain = DomainConfig::new(
            "source".to_string(),
            DomainType::Studio,
            vec![LanguageCode::EnUs],
        );

        let target_domain = DomainConfig::new(
            "target".to_string(),
            DomainType::Field,
            vec![LanguageCode::Fr],
        );

        let report = adapter
            .validate_compatibility(&source_domain, &target_domain)
            .await;
        assert!(report.is_ok());

        let compatibility_report = report.unwrap();
        assert!(compatibility_report.compatibility_score >= 0.0);
        assert!(compatibility_report.compatibility_score <= 1.0);
    }

    #[test]
    fn test_domain_shift_significance() {
        let shift = DomainShift::new(0.5, vec![0.1, 0.2, 0.3], vec!["audio".to_string()], 0.9);

        assert!(shift.is_significant(0.3));
        assert!(!shift.is_significant(0.7));
    }

    #[test]
    fn test_adaptation_result_success() {
        let samples = vec![create_test_sample(1.0, "test")];
        let result = AdaptationResult::success(0.5, samples);

        assert!(result.is_successful());
        assert_eq!(result.improvement, 0.5);
        assert_eq!(result.adapted_samples.len(), 1);
    }

    #[test]
    fn test_adaptation_result_failure() {
        let result = AdaptationResult::failure("Test error".to_string());

        assert!(!result.is_successful());
        assert_eq!(result.improvement, 0.0);
        assert!(result.adapted_samples.is_empty());
        assert_eq!(result.error_message(), Some(&"Test error".to_string()));
    }

    #[test]
    fn test_audio_statistics() {
        let mut stats = AudioStatistics::new();
        stats.sample_rate_distribution.insert(16000, 0.6);
        stats.sample_rate_distribution.insert(22050, 0.4);
        stats.channel_distribution.insert(1, 0.8);
        stats.channel_distribution.insert(2, 0.2);

        assert_eq!(stats.most_common_sample_rate(), Some(16000));
        assert_eq!(stats.most_common_channels(), Some(1));
        assert!(stats.is_predominantly_mono());
    }

    #[test]
    fn test_text_statistics() {
        let mut stats = TextStatistics::new();
        stats.language_distribution.insert(LanguageCode::EnUs, 0.7);
        stats.language_distribution.insert(LanguageCode::Fr, 0.3);

        assert!(stats.is_multilingual());
        assert!(stats.language_diversity() > 0.0);
        assert!(stats.language_diversity() <= 1.0);
    }

    #[test]
    fn test_speaker_statistics() {
        let mut stats = SpeakerStatistics::new();
        stats.gender_distribution.male = 0.45;
        stats.gender_distribution.female = 0.55;
        stats.age_distribution.young_adults = 0.4;
        stats.age_distribution.middle_aged = 0.6;

        assert!(stats.is_gender_balanced(0.2));
        assert_eq!(stats.dominant_gender(), "female");
        assert!(stats.age_diversity() > 0.0);
    }

    #[test]
    fn test_compatibility_report() {
        let mut report = CompatibilityReport::new(0.5, 0.8, 0.6, 0.9);

        assert!(!report.is_highly_compatible());
        assert!(report.adaptation_recommended());

        let recommendation = AdaptationRecommendation::new(
            RecommendationType::AudioPreprocessing,
            Priority::High,
            "Test recommendation".to_string(),
            0.3,
        );

        report.add_recommendation(recommendation);
        assert_eq!(report.high_priority_recommendations().len(), 1);
    }

    #[test]
    fn test_vocabulary_statistics() {
        let mut vocab_stats = VocabularyStatistics::new();
        vocab_stats.unique_words = 100;
        vocab_stats
            .word_frequency_distribution
            .insert("the".to_string(), 50.0);
        vocab_stats
            .word_frequency_distribution
            .insert("and".to_string(), 30.0);
        vocab_stats.frequent_words = vec![("the".to_string(), 50), ("and".to_string(), 30)];

        assert_eq!(
            vocab_stats.most_frequent_word(),
            Some(&("the".to_string(), 50))
        );
        assert!(vocab_stats.is_large_vocabulary(50));
        assert!(!vocab_stats.is_large_vocabulary(200));

        let richness = vocab_stats.vocabulary_richness();
        assert!(richness > 0.0);
    }
}
