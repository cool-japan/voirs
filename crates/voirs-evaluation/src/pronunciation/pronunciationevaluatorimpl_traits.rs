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
        let alignment = self.align_phonemes_to_audio(audio, text).await?;
        self.evaluate_pronunciation_with_alignment(audio, &alignment, text, Some(config))
            .await
    }
    async fn evaluate_pronunciation_with_alignment(
        &self,
        _audio: &AudioBuffer,
        alignment: &PhonemeAlignment,
        reference_text: &str,
        config: Option<&PronunciationEvaluationConfig>,
    ) -> EvaluationResult<PronunciationScore> {
        let config = config.unwrap_or(&self.config);
        // The phoneme/word accuracy aggregates and the prosody scores are all real,
        // alignment-derived computations (see `align_phonemes_to_audio`,
        // `calculate_phoneme_accuracy`, `calculate_word_accuracy`); they are always
        // computed against the caller's actual `reference_text` rather than a
        // hardcoded placeholder. `phoneme_level_scoring`/`word_level_scoring` only
        // control whether the detailed per-item breakdown is included in the result;
        // the aggregates they feed into `overall_score` are never skipped.
        let phoneme_scores_full = self
            .calculate_phoneme_accuracy(alignment, reference_text)
            .await?;
        let word_scores_full = self
            .calculate_word_accuracy(alignment, reference_text)
            .await?;
        let fluency_score = self.calculate_fluency(alignment, reference_text).await?;
        let rhythm_score = self.calculate_rhythm(alignment).await?;
        let stress_accuracy = self
            .calculate_stress_accuracy(alignment, reference_text)
            .await?;
        let intonation_accuracy = self
            .calculate_intonation_accuracy(alignment, reference_text)
            .await?;
        let phoneme_accuracy = if phoneme_scores_full.is_empty() {
            0.0
        } else {
            phoneme_scores_full.iter().map(|s| s.accuracy).sum::<f32>()
                / phoneme_scores_full.len() as f32
        };
        let word_accuracy = if word_scores_full.is_empty() {
            0.0
        } else {
            word_scores_full.iter().map(|s| s.accuracy).sum::<f32>()
                / word_scores_full.len() as f32
        };
        let overall_score = (phoneme_accuracy + word_accuracy + fluency_score + rhythm_score) / 4.0;
        let phoneme_scores = if config.phoneme_level_scoring {
            phoneme_scores_full
        } else {
            Vec::new()
        };
        let word_scores = if config.word_level_scoring {
            word_scores_full
        } else {
            Vec::new()
        };
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
            // The aligner's own confidence: the mean acoustic Goodness-of-Pronunciation
            // score across all reference phonemes (see `align_phonemes_to_audio`).
            // Genuinely reflects how much real acoustic evidence backed the alignment
            // rather than a fixed placeholder.
            confidence: alignment.alignment_confidence.clamp(0.0, 1.0),
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
