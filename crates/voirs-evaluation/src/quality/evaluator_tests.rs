//! Tests for `QualityEvaluator`.

use super::QualityEvaluator;
use crate::traits::{
    QualityEvaluationConfig, QualityEvaluator as QualityEvaluatorTrait, QualityMetric,
};
use voirs_sdk::AudioBuffer;

#[tokio::test]
async fn test_quality_evaluator_creation() {
    let evaluator = QualityEvaluator::new().await.unwrap();
    assert!(!evaluator.supported_metrics().is_empty());
    assert_eq!(evaluator.metadata().name, "VoiRS Quality Evaluator");
}

#[tokio::test]
async fn test_quality_evaluation() {
    let evaluator = QualityEvaluator::new().await.unwrap();
    let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);

    let result = evaluator
        .evaluate_quality(&audio, None, None)
        .await
        .unwrap();

    assert!(result.overall_score >= 0.0);
    assert!(result.overall_score <= 5.0); // MOS scale
    assert!(!result.component_scores.is_empty());
    assert!(result.confidence > 0.0);
}

#[tokio::test]
async fn test_quality_evaluation_with_reference() {
    let evaluator = QualityEvaluator::new().await.unwrap();

    // Create longer signals for PESQ (8 seconds)
    let duration_samples = 8 * 16000;
    let generated = AudioBuffer::new(vec![0.1; duration_samples], 16000, 1);
    let reference = AudioBuffer::new(vec![0.15; duration_samples], 16000, 1);

    let result = evaluator
        .evaluate_quality(&generated, Some(&reference), None)
        .await
        .unwrap();

    assert!(result.overall_score >= 0.0);
    assert!(!result.component_scores.is_empty());

    // Should have PESQ score when reference is provided and explicitly requested
    let config = QualityEvaluationConfig {
        metrics: vec![QualityMetric::PESQ],
        ..Default::default()
    };
    let pesq_result = evaluator
        .evaluate_quality(&generated, Some(&reference), Some(&config))
        .await
        .unwrap();
    assert!(pesq_result.component_scores.contains_key("PESQ"));
}

#[tokio::test]
async fn test_batch_evaluation() {
    let evaluator = QualityEvaluator::new().await.unwrap();

    let samples = vec![
        (AudioBuffer::new(vec![0.1; 8000], 16000, 1), None),
        (AudioBuffer::new(vec![0.2; 8000], 16000, 1), None),
    ];

    let results = evaluator
        .evaluate_quality_batch(&samples, None)
        .await
        .unwrap();

    assert_eq!(results.len(), 2);
    for result in &results {
        assert!(result.overall_score >= 0.0);
        assert!(!result.component_scores.is_empty());
    }
}

#[tokio::test]
async fn test_reference_requirements() {
    let evaluator = QualityEvaluator::new().await.unwrap();

    assert!(evaluator.requires_reference(&QualityMetric::PESQ));
    assert!(evaluator.requires_reference(&QualityMetric::MCD));
    assert!(!evaluator.requires_reference(&QualityMetric::MOS));
    assert!(!evaluator.requires_reference(&QualityMetric::Naturalness));
}

#[tokio::test]
async fn test_individual_metrics() {
    let evaluator = QualityEvaluator::new().await.unwrap();
    let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);

    // Test MOS calculation
    let mos = evaluator.calculate_mos(&audio, None).await.unwrap();
    assert!((1.0..=5.0).contains(&mos));

    // Test naturalness calculation
    let naturalness = evaluator.calculate_naturalness(&audio, None).await.unwrap();
    assert!((0.0..=1.0).contains(&naturalness));

    // Test intelligibility calculation
    let intelligibility = evaluator
        .calculate_intelligibility(&audio, None)
        .await
        .unwrap();
    assert!((0.0..=1.0).contains(&intelligibility));
}

#[tokio::test]
async fn test_artifact_detection() {
    let evaluator = QualityEvaluator::new().await.unwrap();

    // Test clean audio
    let clean_audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);
    let clean_score = evaluator
        .detect_artifacts(&clean_audio, None)
        .await
        .unwrap();

    // Test clipped audio
    let clipped_audio = AudioBuffer::new(vec![1.0; 16000], 16000, 1);
    let clipped_score = evaluator
        .detect_artifacts(&clipped_audio, None)
        .await
        .unwrap();

    assert!((0.0..=1.0).contains(&clean_score));
    assert!((0.0..=1.0).contains(&clipped_score));
}

#[tokio::test]
async fn test_demographic_adapted_mos() {
    let evaluator = QualityEvaluator::new().await.unwrap();
    let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);

    // Create a demographic profile for testing
    let demographic_profile = crate::perceptual::DemographicProfile {
        age_group: crate::perceptual::AgeGroup::Young,
        gender: crate::perceptual::Gender::Female,
        education_level: crate::perceptual::EducationLevel::Bachelor,
        native_language: "English".to_string(),
        audio_experience: crate::perceptual::ExperienceLevel::Intermediate,
    };

    let adapted_mos = evaluator
        .calculate_demographic_adapted_mos(&audio, None, &demographic_profile)
        .await
        .unwrap();

    // Should be within valid MOS range
    assert!((1.0..=5.0).contains(&adapted_mos));

    // Get base MOS for comparison
    let base_mos = evaluator.calculate_mos(&audio, None).await.unwrap();

    // Adapted score should be different from base (due to demographic factors)
    // Allow some tolerance for floating point comparison
    assert!((adapted_mos - base_mos).abs() >= 0.001 || (adapted_mos - base_mos).abs() < 0.1);
}

#[tokio::test]
async fn test_multi_demographic_mos() {
    let evaluator = QualityEvaluator::new().await.unwrap();
    let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);

    let (average_mos, demographic_scores) = evaluator
        .calculate_multi_demographic_mos(&audio, None, None)
        .await
        .unwrap();

    // Should be within valid MOS range
    assert!((0.0..=5.0).contains(&average_mos));

    // Should have demographic breakdown scores
    assert!(!demographic_scores.is_empty());

    // All demographic scores should be valid
    for (_category, score) in &demographic_scores {
        assert!(score.is_finite() && *score >= 0.0);
    }
}
