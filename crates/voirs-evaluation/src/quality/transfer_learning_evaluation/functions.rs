//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::traits::EvaluationResult;
use async_trait::async_trait;
use std::collections::HashMap;
use voirs_recognizer::traits::PhonemeAlignment;
use voirs_sdk::{AudioBuffer, LanguageCode};

use super::types::{
    ConvergencePattern, NegativeTransferSourceType, TransferHistoryEntry,
    TransferLearningEvaluationConfig, TransferLearningEvaluationResult, TransferLearningEvaluator,
};

/// Transfer learning evaluation trait
#[async_trait]
pub trait TransferLearningEvaluationTrait {
    /// Evaluate transfer learning performance
    async fn evaluate_transfer_learning(
        &mut self,
        source_language: LanguageCode,
        target_audios: &HashMap<LanguageCode, AudioBuffer>,
        reference_audios: Option<&HashMap<LanguageCode, AudioBuffer>>,
        phoneme_alignments: Option<&HashMap<LanguageCode, PhonemeAlignment>>,
        transfer_history: Option<&HashMap<(LanguageCode, LanguageCode), Vec<TransferHistoryEntry>>>,
    ) -> EvaluationResult<TransferLearningEvaluationResult>;
    /// Get supported languages
    fn get_supported_languages(&self) -> Vec<LanguageCode>;
    /// Get transfer history
    fn get_transfer_history(
        &self,
        source_language: LanguageCode,
        target_language: LanguageCode,
    ) -> Option<&Vec<TransferHistoryEntry>>;
    /// Add transfer history entry
    fn add_transfer_history_entry(
        &mut self,
        source_language: LanguageCode,
        target_language: LanguageCode,
        entry: TransferHistoryEntry,
    );
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_transfer_learning_evaluator_creation() {
        let config = TransferLearningEvaluationConfig::default();
        let evaluator = TransferLearningEvaluator::new(config);
        assert!(!evaluator.get_supported_languages().is_empty());
        assert!(!evaluator.language_similarity_matrix.is_empty());
    }
    #[tokio::test]
    async fn test_language_similarity_calculation() {
        let config = TransferLearningEvaluationConfig::default();
        let evaluator = TransferLearningEvaluator::new(config);
        let similarity =
            evaluator.calculate_language_similarity(LanguageCode::EnUs, LanguageCode::EsEs);
        assert!(similarity >= 0.0 && similarity <= 1.0);
        let same_language_similarity =
            evaluator.calculate_language_similarity(LanguageCode::EnUs, LanguageCode::EnUs);
        assert_eq!(same_language_similarity, 1.0);
    }
    #[tokio::test]
    async fn test_transfer_learning_evaluation() {
        let config = TransferLearningEvaluationConfig::default();
        let mut evaluator = TransferLearningEvaluator::new(config);
        let mut target_audios = HashMap::new();
        target_audios.insert(
            LanguageCode::EsEs,
            AudioBuffer::new(vec![0.1; 16000], 16000, 1),
        );
        target_audios.insert(
            LanguageCode::FrFr,
            AudioBuffer::new(vec![0.12; 16000], 16000, 1),
        );
        let result = evaluator
            .evaluate_transfer_learning(LanguageCode::EnUs, &target_audios, None, None, None)
            .await
            .unwrap();
        assert_eq!(result.source_language, LanguageCode::EnUs);
        assert_eq!(result.target_languages.len(), 2);
        assert!(result.overall_transfer_score >= 0.0 && result.overall_transfer_score <= 1.0);
        assert!(result.evaluation_confidence >= 0.0 && result.evaluation_confidence <= 1.0);
    }
    #[test]
    fn test_few_shot_performance_simulation() {
        let config = TransferLearningEvaluationConfig::default();
        let evaluator = TransferLearningEvaluator::new(config);
        let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);
        let performance = evaluator
            .simulate_few_shot_performance(LanguageCode::EnUs, LanguageCode::EsEs, &audio, None, 10)
            .unwrap();
        assert!(performance >= 0.0 && performance <= 1.0);
    }
    #[test]
    fn test_negative_transfer_detection() {
        let config = TransferLearningEvaluationConfig::default();
        let evaluator = TransferLearningEvaluator::new(config);
        let sources = evaluator.identify_negative_transfer_sources(
            LanguageCode::EnUs,
            LanguageCode::ZhCn,
            0.3,
        );
        assert!(!sources.is_empty());
        assert!(sources.iter().any(|s| matches!(
            s.source_type,
            NegativeTransferSourceType::PhoneticInterference
        )));
    }
    #[test]
    fn test_transfer_history_management() {
        let config = TransferLearningEvaluationConfig::default();
        let mut evaluator = TransferLearningEvaluator::new(config);
        let entry = TransferHistoryEntry {
            epoch: 1,
            performance: 0.8,
            loss: 0.2,
            validation_score: Some(0.75),
            timestamp: std::time::SystemTime::now(),
        };
        evaluator.add_transfer_history_entry(LanguageCode::EnUs, LanguageCode::EsEs, entry);
        let history = evaluator.get_transfer_history(LanguageCode::EnUs, LanguageCode::EsEs);
        assert!(history.is_some());
        assert_eq!(history.unwrap().len(), 1);
    }
    #[test]
    fn test_convergence_pattern_analysis() {
        let config = TransferLearningEvaluationConfig::default();
        let evaluator = TransferLearningEvaluator::new(config);
        let pattern = evaluator
            .analyze_individual_convergence_pattern(LanguageCode::EnUs, LanguageCode::EsEs)
            .unwrap();
        assert!(matches!(
            pattern,
            ConvergencePattern::Monotonic
                | ConvergencePattern::Oscillating
                | ConvergencePattern::Plateau
                | ConvergencePattern::Divergent
                | ConvergencePattern::Irregular
        ));
    }
    #[test]
    fn test_domain_adaptation_assessment() {
        let config = TransferLearningEvaluationConfig::default();
        let evaluator = TransferLearningEvaluator::new(config);
        let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);
        let adaptation = evaluator
            .calculate_domain_adaptation(LanguageCode::EnUs, LanguageCode::EsEs, &audio, None)
            .unwrap();
        assert!(adaptation.adaptation_score >= 0.0 && adaptation.adaptation_score <= 1.0);
        assert!(adaptation.domain_similarity >= 0.0 && adaptation.domain_similarity <= 1.0);
        assert!(adaptation.domain_gap >= 0.0 && adaptation.domain_gap <= 1.0);
    }
}
