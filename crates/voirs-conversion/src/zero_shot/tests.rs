//! Tests for zero-shot voice conversion

use super::*;
use crate::types::{
    AgeGroup, Gender, PitchCharacteristics, QualityCharacteristics, SpectralCharacteristics,
    TimingCharacteristics, VoiceCharacteristics,
};
use std::time::Instant;

#[test]
fn test_zero_shot_config_creation() {
    let config = ZeroShotConfig::default();
    assert!(config.enabled);
    assert_eq!(config.quality_threshold, 0.7);
    assert_eq!(config.max_references, 10);
}

#[test]
fn test_zero_shot_converter_creation() {
    let config = ZeroShotConfig::default();
    let converter = ZeroShotConverter::new(config);
    assert_eq!(converter.metrics().successful_conversions, 0);
}

#[test]
fn test_reference_voice_database() {
    let mut db = ReferenceVoiceDatabase::new();
    assert_eq!(db.metadata().total_voices, 0);

    let voice = ReferenceVoice {
        speaker_id: "test_speaker".to_string(),
        name: "Test Speaker".to_string(),
        audio_samples: Vec::new(),
        embedding: SpeakerEmbedding {
            data: vec![0.0; 256],
            confidence: 0.8,
        },
        characteristics: VoiceCharacteristics {
            pitch: PitchCharacteristics {
                mean_f0: 200.0,
                range: 12.0,
                jitter: 0.1,
                stability: 0.8,
            },
            timing: TimingCharacteristics {
                speaking_rate: 1.0,
                pause_duration: 1.0,
                rhythm_regularity: 0.7,
            },
            spectral: SpectralCharacteristics {
                formant_shift: 0.1,
                brightness: 0.0,
                spectral_tilt: 0.0,
                harmonicity: 0.8,
            },
            quality: QualityCharacteristics {
                breathiness: 0.1,
                roughness: 0.1,
                stability: 0.8,
                resonance: 0.7,
            },
            age_group: Some(AgeGroup::YoungAdult),
            gender: Some(Gender::Female),
            accent: Some("american".to_string()),
            custom_params: std::collections::HashMap::new(),
        },
        quality_scores: QualityScores {
            overall: 0.8,
            clarity: 0.85,
            naturalness: 0.8,
            consistency: 0.75,
            recording_quality: 0.9,
            prosody_quality: 0.8,
        },
        metadata: VoiceMetadata {
            language: "en".to_string(),
            accent: Some("american".to_string()),
            gender: Some("female".to_string()),
            age_group: Some("adult".to_string()),
            recording_environment: Some("studio".to_string()),
            tags: vec!["clear".to_string(), "professional".to_string()],
            created: Instant::now(),
            modified: None,
        },
        last_used: None,
    };

    db.add_voice(voice).unwrap();
    assert_eq!(db.metadata().total_voices, 1);
}

#[test]
fn test_quality_assessment() {
    let assessor = QualityAssessor::new();
    let original = vec![0.5, -0.3, 0.8, -0.2, 0.1];
    let converted = vec![0.4, -0.25, 0.75, -0.15, 0.05];

    let quality = assessor
        .assess_overall_quality(&original, &converted, 16000)
        .unwrap();
    assert!(quality >= 0.0 && quality <= 1.0);
}

#[test]
fn test_style_features_creation() {
    let features = StyleFeatures {
        prosodic: ProsodicStyleFeatures {
            intonation_patterns: vec![0.0; 5],
            rhythm_characteristics: vec![0.0; 5],
            stress_patterns: vec![0.0; 5],
            pausing_behavior: vec![0.0; 5],
        },
        spectral: SpectralStyleFeatures {
            formant_characteristics: vec![0.0; 5],
            spectral_envelope: vec![0.0; 5],
            harmonic_content: vec![0.0; 5],
            noise_characteristics: vec![0.0; 5],
        },
        temporal: TemporalStyleFeatures {
            speaking_rate_variations: vec![0.0; 5],
            articulation_patterns: vec![0.0; 5],
            transition_characteristics: vec![0.0; 5],
            timing_precision: vec![0.0; 5],
        },
        voice_quality: VoiceQualityFeatures {
            breathiness: 0.3,
            roughness: 0.2,
            creakiness: 0.1,
            tenseness: 0.4,
            overall_quality: 0.8,
        },
    };

    assert_eq!(features.prosodic.intonation_patterns.len(), 5);
    assert_eq!(features.voice_quality.overall_quality, 0.8);
}

#[test]
fn test_metrics_initialization() {
    let metrics = ZeroShotMetrics::default();
    assert_eq!(metrics.successful_conversions, 0);
    assert_eq!(metrics.failed_conversions, 0);
    assert_eq!(metrics.avg_processing_time, 0.0);
}

#[test]
fn test_zero_shot_method_enum() {
    let method = ZeroShotMethod::Hybrid;
    assert_eq!(method, ZeroShotMethod::Hybrid);
    assert_ne!(method, ZeroShotMethod::EmbeddingInterpolation);
}

#[test]
fn test_quality_classification() {
    let classification = QualityClassification::Good;
    assert_eq!(classification, QualityClassification::Good);
    assert_ne!(classification, QualityClassification::Poor);
}
