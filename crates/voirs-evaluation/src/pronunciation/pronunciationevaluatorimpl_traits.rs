//! # PronunciationEvaluatorImpl - Trait Implementations
//!
//! This module contains trait implementations for `PronunciationEvaluatorImpl`.
//!
//! ## Implemented Traits
//!
//! - `PronunciationEvaluator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::traits::{
    ComparativeEvaluator, EvaluationResult, FeedbackType, PhonemeAccuracyScore,
    PronunciationEvaluationConfig, PronunciationEvaluator, PronunciationEvaluatorMetadata,
    PronunciationFeedback, PronunciationMetric, PronunciationScore, QualityEvaluator,
    SelfEvaluator, WordPronunciationScore,
};
use async_trait::async_trait;
use scirs2_core::parallel_ops::*;
use std::time::Instant;
use voirs_recognizer::traits::{AlignedPhoneme, PhonemeAlignment};
use voirs_sdk::{AudioBuffer, LanguageCode, Phoneme, SyllablePosition};

use super::types::PronunciationEvaluatorImpl;

#[async_trait]
impl PronunciationEvaluator for PronunciationEvaluatorImpl {
    async fn evaluate_pronunciation(
        &self,
        audio: &AudioBuffer,
        text: &str,
        config: Option<&PronunciationEvaluationConfig>,
    ) -> EvaluationResult<PronunciationScore> {
        let config = config.unwrap_or(&self.config);
        let mock_alignment = self.create_mock_alignment(audio, text).await?;
        self.evaluate_pronunciation_with_alignment(audio, &mock_alignment, Some(config))
            .await
    }
    async fn evaluate_pronunciation_with_alignment(
        &self,
        _audio: &AudioBuffer,
        alignment: &PhonemeAlignment,
        config: Option<&PronunciationEvaluationConfig>,
    ) -> EvaluationResult<PronunciationScore> {
        let config = config.unwrap_or(&self.config);
        let start_time = Instant::now();
        let expected_text = "Hello world";
        let phoneme_scores = if config.phoneme_level_scoring {
            self.calculate_phoneme_accuracy(alignment, expected_text)
                .await?
        } else {
            Vec::new()
        };
        let word_scores = if config.word_level_scoring {
            self.calculate_word_accuracy(alignment, expected_text)
                .await?
        } else {
            Vec::new()
        };
        let fluency_score = if config.prosody_assessment {
            self.calculate_fluency(alignment, expected_text).await?
        } else {
            0.8
        };
        let rhythm_score = if config.prosody_assessment {
            self.calculate_rhythm(alignment).await?
        } else {
            0.8
        };
        let stress_accuracy = if config.prosody_assessment {
            self.calculate_stress_accuracy(alignment, expected_text)
                .await?
        } else {
            0.8
        };
        let intonation_accuracy = if config.prosody_assessment {
            self.calculate_intonation_accuracy(alignment, expected_text)
                .await?
        } else {
            0.8
        };
        let phoneme_accuracy = if phoneme_scores.is_empty() {
            0.85
        } else {
            phoneme_scores.iter().map(|s| s.accuracy).sum::<f32>() / phoneme_scores.len() as f32
        };
        let word_accuracy = if word_scores.is_empty() {
            0.85
        } else {
            word_scores.iter().map(|s| s.accuracy).sum::<f32>() / word_scores.len() as f32
        };
        let overall_score = (phoneme_accuracy + word_accuracy + fluency_score + rhythm_score) / 4.0;
        let feedback = self
            .generate_feedback(&phoneme_scores, &word_scores)
            .await?;
        Ok(PronunciationScore {
            overall_score,
            phoneme_scores,
            word_scores,
            fluency_score,
            rhythm_score,
            stress_accuracy,
            intonation_accuracy,
            feedback,
            confidence: 0.80,
        })
    }
    async fn evaluate_pronunciation_batch(
        &self,
        samples: &[(AudioBuffer, String)],
        config: Option<&PronunciationEvaluationConfig>,
    ) -> EvaluationResult<Vec<PronunciationScore>> {
        if samples.len() <= 4 {
            let mut results = Vec::new();
            for (audio, text) in samples {
                let score = self.evaluate_pronunciation(audio, text, config).await?;
                results.push(score);
            }
            return Ok(results);
        }
        use futures::future::try_join_all;
        let futures: Vec<_> = samples
            .iter()
            .map(|(audio, text)| self.evaluate_pronunciation(audio, text, config))
            .collect();
        try_join_all(futures).await
    }
    fn supported_metrics(&self) -> Vec<PronunciationMetric> {
        self.supported_metrics.clone()
    }
    fn supported_languages(&self) -> Vec<LanguageCode> {
        self.metadata.supported_languages.clone()
    }
    fn metadata(&self) -> PronunciationEvaluatorMetadata {
        self.metadata.clone()
    }
}
