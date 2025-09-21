//! Active learning module
//!
//! This module provides comprehensive active learning functionality for speech synthesis
//! datasets, including uncertainty-based sampling, diversity-based selection, human-in-the-loop
//! workflows, and annotation efficiency optimization.

pub mod config;
pub mod interfaces;
pub mod learner;
pub mod types;

// Re-export main types for convenience
pub use config::*;
pub use interfaces::{APIAnnotationInterface, CLIAnnotationInterface, WebAnnotationInterface};
pub use learner::ActiveLearnerImpl;
pub use types::*;

// Re-export for backward compatibility
pub use ActiveLearnerImpl as ActiveLearner;

#[cfg(test)]
mod tests {
    use super::types::ActiveLearner;
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
    async fn test_active_learner_creation() {
        let config = ActiveLearningConfig::default();
        let learner = ActiveLearnerImpl::new(config);
        assert!(learner.is_ok());
    }

    #[tokio::test]
    async fn test_sample_selection() {
        let config = ActiveLearningConfig::default();
        let learner = ActiveLearnerImpl::new(config).unwrap();

        let unlabeled_samples = vec![
            create_test_sample(1.0, "First sample"),
            create_test_sample(1.5, "Second sample"),
            create_test_sample(2.0, "Third sample"),
        ];

        let labeled_samples = vec![create_test_sample(1.2, "Labeled sample")];

        let result = learner
            .select_samples(&unlabeled_samples, &labeled_samples, 2)
            .await;
        assert!(result.is_ok());

        let selection_result = result.unwrap();
        assert!(selection_result.is_valid());
        assert_eq!(selection_result.num_selected(), 2);
        assert!(selection_result.average_uncertainty() >= 0.0);
        assert!(selection_result.average_diversity() >= 0.0);
    }

    #[tokio::test]
    async fn test_uncertainty_calculation() {
        let config = ActiveLearningConfig {
            uncertainty_metric: UncertaintyMetric::Entropy,
            ..Default::default()
        };
        let learner = ActiveLearnerImpl::new(config).unwrap();

        let samples = vec![
            create_test_sample(1.0, "Test sample"),
            create_test_sample(0.5, "Short sample"),
        ];

        let uncertainties = learner.calculate_uncertainty(&samples).await;
        assert!(uncertainties.is_ok());

        let uncertainty_scores = uncertainties.unwrap();
        assert_eq!(uncertainty_scores.len(), 2);
        assert!(uncertainty_scores
            .iter()
            .all(|&score| (0.0..=1.0).contains(&score)));
    }

    #[tokio::test]
    async fn test_diversity_calculation() {
        let config = ActiveLearningConfig::default();
        let learner = ActiveLearnerImpl::new(config).unwrap();

        let samples = vec![
            create_test_sample(1.0, "First sample"),
            create_test_sample(1.5, "Second sample"),
        ];

        let reference_samples = vec![create_test_sample(1.2, "Reference sample")];

        let diversities = learner
            .calculate_diversity(&samples, &reference_samples)
            .await;
        assert!(diversities.is_ok());

        let diversity_scores = diversities.unwrap();
        assert_eq!(diversity_scores.len(), 2);
        assert!(diversity_scores
            .iter()
            .all(|&score| (0.0..=1.0).contains(&score)));
    }

    #[tokio::test]
    async fn test_annotation_update() {
        let config = ActiveLearningConfig::default();
        let mut learner = ActiveLearnerImpl::new(config).unwrap();

        let annotations = vec![AnnotationResult::new(
            "sample_1".to_string(),
            "annotator_1".to_string(),
            QualityMetrics {
                overall_quality: Some(0.8),
                snr: Some(25.0),
                ..Default::default()
            },
        )];

        let result = learner.update_model(&annotations).await;
        assert!(result.is_ok());
        assert_eq!(learner.annotation_count(), 1);
    }

    #[tokio::test]
    async fn test_annotation_statistics() {
        let config = ActiveLearningConfig::default();
        let mut learner = ActiveLearnerImpl::new(config).unwrap();

        // Add some annotations
        let annotations = vec![
            AnnotationResult::new(
                "sample_1".to_string(),
                "annotator_1".to_string(),
                QualityMetrics {
                    overall_quality: Some(0.8),
                    ..Default::default()
                },
            ),
            AnnotationResult::new(
                "sample_2".to_string(),
                "annotator_2".to_string(),
                QualityMetrics {
                    overall_quality: Some(0.6),
                    ..Default::default()
                },
            ),
        ];

        learner.update_model(&annotations).await.unwrap();

        let stats = learner.get_annotation_statistics().await;
        assert!(stats.is_ok());

        let annotation_stats = stats.unwrap();
        assert_eq!(annotation_stats.total_annotations, 2);
        assert_eq!(annotation_stats.num_annotators(), 2);
    }

    #[test]
    fn test_config_validation() {
        let config = ActiveLearningConfig {
            batch_size: 0,
            ..Default::default()
        };

        let learner = ActiveLearnerImpl::new(config);
        assert!(learner.is_err());
    }

    #[test]
    fn test_sampling_strategy_properties() {
        let uncertainty_strategy = SamplingStrategy::Uncertainty {
            threshold: 0.5,
            use_diversity: true,
        };

        assert!(uncertainty_strategy.uses_uncertainty());
        assert!(uncertainty_strategy.uses_diversity());

        let diversity_strategy = SamplingStrategy::Diversity {
            metric: DiversityMetric::CosineDistance,
            min_diversity: 0.3,
        };

        assert!(!diversity_strategy.uses_uncertainty());
        assert!(diversity_strategy.uses_diversity());
    }

    #[test]
    fn test_uncertainty_metric_properties() {
        assert!(UncertaintyMetric::Entropy.is_classification_metric());
        assert!(!UncertaintyMetric::Entropy.is_regression_metric());
        assert!(!UncertaintyMetric::Entropy.requires_ensemble());

        assert!(!UncertaintyMetric::Variance.is_classification_metric());
        assert!(UncertaintyMetric::Variance.is_regression_metric());

        assert!(UncertaintyMetric::EnsembleDisagreement.requires_ensemble());
        assert!(UncertaintyMetric::BALD.requires_ensemble());
    }

    #[test]
    fn test_diversity_config_validation() {
        let mut config = DiversityConfig::default();
        assert!(config.validate().is_ok());

        config.diversity_weight = 1.5;
        assert!(config.validate().is_err());

        config.diversity_weight = 0.5;
        config.num_clusters = 0;
        assert!(config.validate().is_err());

        config.num_clusters = 5;
        config.min_distance_threshold = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_annotation_interface_properties() {
        let web_interface = AnnotationInterfaceType::Web {
            port: 8080,
            enable_audio_playback: true,
            show_spectrograms: true,
            custom_css: None,
        };

        assert_eq!(web_interface.name(), "web");
        assert!(web_interface.supports_audio_playback());
        assert!(web_interface.supports_visual_feedback());

        let cli_interface = AnnotationInterfaceType::CLI {
            use_colors: true,
            show_progress: true,
            auto_play_audio: false,
        };

        assert_eq!(cli_interface.name(), "cli");
        assert!(!cli_interface.supports_audio_playback());
        assert!(cli_interface.supports_visual_feedback());

        let api_interface = AnnotationInterfaceType::API {
            endpoint: "https://api.example.com".to_string(),
            auth_token: Some("token".to_string()),
            timeout: 30,
        };

        assert_eq!(api_interface.name(), "api");
        assert!(!api_interface.supports_audio_playback());
        assert!(!api_interface.supports_visual_feedback());
    }

    #[test]
    fn test_expertise_level() {
        let novice = ExpertiseLevel::Novice;
        let expert = ExpertiseLevel::Expert;

        assert_eq!(novice.level(), 1);
        assert_eq!(expert.level(), 3);

        assert!(!novice.meets_requirement(&expert));
        assert!(expert.meets_requirement(&novice));
        assert!(expert.meets_requirement(&expert));
    }

    #[test]
    fn test_feedback_config_properties() {
        let config = FeedbackConfig {
            enable_immediate_feedback: true,
            use_model_suggestions: true,
            suggestion_confidence_threshold: 0.8,
            enable_uncertainty_visualization: true,
            show_similar_samples: true,
            ..Default::default()
        };

        assert!(config.is_feedback_enabled());
        assert!(config.suggestions_enabled());
        assert!(config.has_visualization());

        let disabled_config = FeedbackConfig {
            enable_immediate_feedback: false,
            update_frequency: UpdateFrequency::Manual,
            use_model_suggestions: false,
            enable_uncertainty_visualization: false,
            show_similar_samples: false,
            ..Default::default()
        };

        assert!(!disabled_config.is_feedback_enabled());
        assert!(!disabled_config.suggestions_enabled());
        assert!(!disabled_config.has_visualization());
    }

    #[test]
    fn test_human_loop_config_properties() {
        let config = HumanLoopConfig {
            enable_realtime: true,
            annotation_timeout: Some(300),
            quality_assurance: QualityAssuranceConfig {
                enable_consensus: true,
                enable_expert_review: true,
                ..Default::default()
            },
            ..Default::default()
        };

        assert!(config.is_realtime_enabled());
        assert_eq!(config.timeout_ms(), Some(300000));
        assert!(config.has_quality_assurance());

        let simple_config = HumanLoopConfig {
            enable_realtime: false,
            annotation_timeout: None,
            quality_assurance: QualityAssuranceConfig {
                enable_consensus: false,
                enable_expert_review: false,
                ..Default::default()
            },
            ..Default::default()
        };

        assert!(!simple_config.is_realtime_enabled());
        assert_eq!(simple_config.timeout_ms(), None);
        assert!(!simple_config.has_quality_assurance());
    }
}
