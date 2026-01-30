//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{
    types::{SpeakerCharacteristics, SpeakerProfile, VoiceSample},
    Error, Result,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, trace, warn};

use super::types::*;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::VoiceSample;
    #[tokio::test]
    async fn test_few_shot_learner_creation() {
        let config = FewShotConfig::default();
        let learner = FewShotLearner::new(config);
        assert!(learner.is_ok());
    }
    #[test]
    fn test_sample_quality_computation() {
        let audio = vec![0.1, -0.1, 0.2, -0.2, 0.05];
        let sample = VoiceSample::new("test".to_string(), audio, 16000);
        let quality = SampleQuality::from_sample(&sample);
        assert!(quality.overall_quality >= 0.0 && quality.overall_quality <= 1.0);
        assert!(quality.snr >= 0.0);
        assert!(quality.spectral_clarity >= 0.0 && quality.spectral_clarity <= 1.0);
    }
    #[tokio::test]
    async fn test_feature_extraction() {
        let config = FewShotConfig::default();
        let device = Device::Cpu;
        let extractor = FeatureExtractor::new(config.clone(), device).unwrap();
        let audio = vec![0.1; 1000];
        let sample = VoiceSample::new("test".to_string(), audio, 16000);
        let features = extractor.extract_features(&sample).await.unwrap();
        assert_eq!(features.len(), config.embedding_dim);
    }
    #[tokio::test]
    async fn test_prototypical_adaptation() {
        let config = FewShotConfig {
            num_shots: 3,
            meta_algorithm: MetaLearningAlgorithm::ProtoNet,
            quality_threshold: 0.1,
            ..Default::default()
        };
        let mut learner = FewShotLearner::new(config).unwrap();
        let mut samples = Vec::new();
        for i in 0..5 {
            let audio = vec![0.1 * (i + 1) as f32; 1000];
            samples.push(VoiceSample::new(format!("sample_{}", i), audio, 16000));
        }
        let result = learner
            .adapt_speaker("test_speaker", &samples)
            .await
            .unwrap();
        assert!(!result.speaker_embedding.is_empty());
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert_eq!(result.algorithm, MetaLearningAlgorithm::ProtoNet);
        assert!(result.samples_used >= 3);
    }
    #[test]
    fn test_distance_metrics() {
        let config = FewShotConfig::default();
        let learner = FewShotLearner::new(config).unwrap();
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![1.0, 0.0, 0.0];
        let sim_ab = learner.compute_similarity(&a, &b).unwrap();
        let sim_ac = learner.compute_similarity(&a, &c).unwrap();
        assert!(sim_ac > sim_ab);
        assert!(sim_ac > 0.9);
    }
    #[tokio::test]
    async fn test_cross_lingual_adaptation() {
        let config = FewShotConfig {
            num_shots: 3,
            enable_cross_lingual: true,
            meta_algorithm: MetaLearningAlgorithm::ProtoNet,
            quality_threshold: 0.1,
            ..Default::default()
        };
        let mut learner = FewShotLearner::new(config).unwrap();
        let mut samples = Vec::new();
        for i in 0..5 {
            let audio = vec![0.1 * (i + 1) as f32; 1000];
            samples.push(VoiceSample::new(format!("sample_{}", i), audio, 16000));
        }
        let result = learner
            .adapt_speaker_cross_lingual("test_speaker", &samples, "en", "es")
            .await
            .unwrap();
        assert!(!result.speaker_embedding.is_empty());
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(result.samples_used >= 3);
        assert!(result.cross_lingual_info.is_some());
        let cross_lingual_info = result.cross_lingual_info.unwrap();
        assert_eq!(cross_lingual_info.source_language, "en");
        assert_eq!(cross_lingual_info.target_language, "es");
        assert!(cross_lingual_info.phonetic_similarity > 0.0);
        assert!(cross_lingual_info.language_adaptation_applied);
    }
    #[test]
    fn test_phonetic_similarity_calculation() {
        let config = FewShotConfig::default();
        let learner = FewShotLearner::new(config).unwrap();
        assert_eq!(learner.calculate_phonetic_similarity("en", "en"), 1.0);
        let en_es_sim = learner.calculate_phonetic_similarity("en", "es");
        assert!(en_es_sim > 0.6);
        let fr_es_sim = learner.calculate_phonetic_similarity("fr", "es");
        assert!(fr_es_sim > 0.8);
        let en_zh_sim = learner.calculate_phonetic_similarity("en", "zh");
        assert!(en_zh_sim < 0.5);
        assert_eq!(
            learner.calculate_phonetic_similarity("en", "fr"),
            learner.calculate_phonetic_similarity("fr", "en")
        );
    }
    #[test]
    fn test_language_adaptation_matrix() {
        let config = FewShotConfig::default();
        let learner = FewShotLearner::new(config).unwrap();
        let matrix_en_zh = learner.get_language_adaptation_matrix("en", "zh").unwrap();
        let matrix_en_es = learner.get_language_adaptation_matrix("en", "es").unwrap();
        assert_eq!(matrix_en_zh.len(), learner.config.embedding_dim);
        assert_eq!(matrix_en_es.len(), learner.config.embedding_dim);
        assert_ne!(matrix_en_zh, matrix_en_es);
    }
    #[test]
    fn test_phonetic_shifts() {
        let config = FewShotConfig::default();
        let learner = FewShotLearner::new(config).unwrap();
        let shifts_en_es = learner.get_phonetic_shifts("en", "es");
        let shifts_en_zh = learner.get_phonetic_shifts("en", "zh");
        let shifts_unknown = learner.get_phonetic_shifts("unknown1", "unknown2");
        assert!(!shifts_en_es.is_empty());
        assert!(!shifts_en_zh.is_empty());
        assert!(shifts_unknown.iter().all(|&x| x == 0.0));
        let max_zh_shift = shifts_en_zh.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let max_es_shift = shifts_en_es.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(max_zh_shift > max_es_shift);
    }
    #[tokio::test]
    async fn test_cross_lingual_disabled_error() {
        let config = FewShotConfig {
            enable_cross_lingual: false,
            ..Default::default()
        };
        let mut learner = FewShotLearner::new(config).unwrap();
        let samples = vec![VoiceSample::new("test".to_string(), vec![0.1; 1000], 16000)];
        let result = learner
            .adapt_speaker_cross_lingual("test_speaker", &samples, "en", "es")
            .await;
        assert!(result.is_err());
        if let Err(error) = result {
            assert!(error
                .to_string()
                .contains("Cross-lingual learning is not enabled"));
        }
    }
    #[tokio::test]
    async fn test_cross_lingual_insufficient_data() {
        let config = FewShotConfig {
            enable_cross_lingual: true,
            num_shots: 3,
            ..Default::default()
        };
        let mut learner = FewShotLearner::new(config).unwrap();
        let samples = vec![
            VoiceSample::new("sample1".to_string(), vec![0.1; 1000], 16000),
            VoiceSample::new("sample2".to_string(), vec![0.2; 1000], 16000),
        ];
        let result = learner
            .adapt_speaker_cross_lingual("test_speaker", &samples, "en", "es")
            .await;
        assert!(result.is_err());
        if let Err(error) = result {
            assert!(error.to_string().contains("Need at least 3 samples"));
        }
    }
}
