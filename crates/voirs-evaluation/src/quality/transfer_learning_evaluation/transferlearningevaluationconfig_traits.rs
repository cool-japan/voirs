//! # TransferLearningEvaluationConfig - Trait Implementations
//!
//! This module contains trait implementations for `TransferLearningEvaluationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use voirs_sdk::{AudioBuffer, LanguageCode};

use super::types::TransferLearningEvaluationConfig;

impl Default for TransferLearningEvaluationConfig {
    fn default() -> Self {
        Self {
            enable_knowledge_transfer_assessment: true,
            enable_transfer_effectiveness: true,
            enable_stability_analysis: true,
            enable_few_shot_evaluation: true,
            enable_domain_adaptation: true,
            enable_negative_transfer_detection: true,
            enable_transfer_optimization: true,
            knowledge_transfer_weight: 0.25,
            transfer_effectiveness_weight: 0.25,
            stability_analysis_weight: 0.2,
            few_shot_evaluation_weight: 0.15,
            domain_adaptation_weight: 0.15,
            min_transfer_effectiveness_threshold: 0.6,
            max_negative_transfer_threshold: 0.2,
            few_shot_sample_sizes: vec![1, 5, 10, 20, 50],
            evaluation_languages: vec![
                LanguageCode::EnUs,
                LanguageCode::EsEs,
                LanguageCode::FrFr,
                LanguageCode::DeDe,
                LanguageCode::JaJp,
                LanguageCode::ZhCn,
                LanguageCode::Ar,
                LanguageCode::Hi,
                LanguageCode::RuRu,
                LanguageCode::PtBr,
            ],
        }
    }
}
