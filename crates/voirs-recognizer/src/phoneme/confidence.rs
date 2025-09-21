//! Confidence estimation for phoneme recognition
//!
//! This module provides confidence scoring and uncertainty quantification
//! for phoneme recognition and alignment results.

use super::analysis::{AlignedPhoneme, WordAlignment};
use crate::RecognitionError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Confidence estimation for phoneme recognition
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConfidenceEstimator {
    /// Acoustic model confidence weights
    pub acoustic_weights: HashMap<String, f32>,
    /// Language model confidence weights  
    pub language_weights: HashMap<String, f32>,
    /// Alignment quality thresholds
    pub alignment_thresholds: AlignmentThresholds,
}

/// Thresholds for alignment quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentThresholds {
    /// Minimum confidence for high-quality alignment
    pub high_quality: f32,
    /// Minimum confidence for acceptable alignment
    pub acceptable: f32,
    /// Maximum boundary deviation for good alignment
    pub max_boundary_deviation_ms: f32,
    /// Maximum duration ratio deviation
    pub max_duration_ratio_deviation: f32,
}

impl Default for AlignmentThresholds {
    fn default() -> Self {
        Self {
            high_quality: 0.8,
            acceptable: 0.6,
            max_boundary_deviation_ms: 50.0,
            max_duration_ratio_deviation: 0.3,
        }
    }
}

impl ConfidenceEstimator {
    /// Create a new confidence estimator
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Estimate confidence for phoneme alignment
    pub fn estimate_phoneme_confidence(
        &self,
        phoneme: &AlignedPhoneme,
        context: &AlignmentContext,
    ) -> Result<f32, RecognitionError> {
        let mut confidence = phoneme.confidence;

        // Adjust based on acoustic model confidence
        if let Some(acoustic_score) = context.acoustic_scores.get(&phoneme.phoneme.symbol) {
            confidence *= acoustic_score;
        }

        // Adjust based on duration consistency
        let duration_consistency = self.calculate_duration_consistency(phoneme, context)?;
        confidence *= duration_consistency;

        // Adjust based on boundary precision
        let boundary_precision = self.calculate_boundary_precision(phoneme, context)?;
        confidence *= boundary_precision;

        Ok(confidence.min(1.0).max(0.0))
    }

    /// Estimate confidence for word alignment
    pub fn estimate_word_confidence(
        &self,
        word: &WordAlignment,
        context: &AlignmentContext,
    ) -> Result<f32, RecognitionError> {
        if word.phonemes.is_empty() {
            return Ok(0.0);
        }

        // Calculate average phoneme confidence
        let mut total_confidence = 0.0;
        for phoneme in &word.phonemes {
            total_confidence += self.estimate_phoneme_confidence(phoneme, context)?;
        }

        let avg_phoneme_confidence = total_confidence / word.phonemes.len() as f32;

        // Adjust based on word-level factors
        let word_duration = word.end_time - word.start_time;
        let expected_duration = self.estimate_expected_word_duration(&word.word);
        let duration_ratio = if expected_duration > 0.0 {
            (word_duration / expected_duration).min(expected_duration / word_duration)
        } else {
            1.0
        };

        let word_confidence = avg_phoneme_confidence * duration_ratio;

        Ok(word_confidence.min(1.0).max(0.0))
    }

    /// Calculate duration consistency score
    fn calculate_duration_consistency(
        &self,
        phoneme: &AlignedPhoneme,
        _context: &AlignmentContext,
    ) -> Result<f32, RecognitionError> {
        let actual_duration = phoneme.end_time - phoneme.start_time;

        // Estimate expected duration based on phoneme type
        let expected_duration = self.estimate_expected_phoneme_duration(&phoneme.phoneme.symbol);

        if expected_duration > 0.0 {
            let ratio =
                (actual_duration / expected_duration).min(expected_duration / actual_duration);
            let max_deviation = self.alignment_thresholds.max_duration_ratio_deviation;
            let score = 1.0 - ((1.0 - ratio) / max_deviation).min(1.0);
            Ok(score.max(0.0))
        } else {
            Ok(1.0)
        }
    }

    /// Calculate boundary precision score
    fn calculate_boundary_precision(
        &self,
        phoneme: &AlignedPhoneme,
        context: &AlignmentContext,
    ) -> Result<f32, RecognitionError> {
        let mut boundary_score = 1.0;

        // Check for audio artifacts that might affect boundary precision
        if context.signal_quality.snr_db < 10.0 {
            boundary_score *= 0.7; // Low SNR reduces boundary precision
        }

        if context.signal_quality.thd_percent > 10.0 {
            boundary_score *= 0.8; // High distortion affects boundaries
        }

        // Check phoneme duration reasonableness
        let duration = phoneme.end_time - phoneme.start_time;
        if !(0.01..=1.5).contains(&duration) {
            boundary_score *= 0.6; // Extreme durations suggest poor boundaries
        }

        // Factor in recording quality
        boundary_score *= context.signal_quality.recording_quality;

        Ok(boundary_score.min(1.0).max(0.0))
    }

    /// Estimate acoustic likelihood for phoneme
    pub fn estimate_acoustic_likelihood(
        &self,
        phoneme: &AlignedPhoneme,
        context: &AlignmentContext,
    ) -> Result<f32, RecognitionError> {
        let mut likelihood = phoneme.confidence;

        // Get acoustic model score if available
        if let Some(acoustic_score) = context.acoustic_scores.get(&phoneme.phoneme.symbol) {
            likelihood = (likelihood + acoustic_score) / 2.0;
        }

        // Adjust based on signal quality
        let quality_factor = self.calculate_quality_factor(&context.signal_quality)?;
        likelihood *= quality_factor;

        // Adjust based on speaker characteristics
        let speaker_factor = self.calculate_speaker_factor(&context.speaker_info)?;
        likelihood *= speaker_factor;

        Ok(likelihood.min(1.0).max(0.0))
    }

    /// Calculate confidence based on cross-model agreement
    pub fn calculate_cross_model_agreement(
        &self,
        phoneme: &AlignedPhoneme,
        model_results: &[ModelResult],
    ) -> Result<f32, RecognitionError> {
        if model_results.is_empty() {
            return Ok(phoneme.confidence);
        }

        let target_symbol = &phoneme.phoneme.symbol;
        let mut agreement_count = 0;
        let mut total_confidence = 0.0;

        for result in model_results {
            if &result.predicted_phoneme == target_symbol {
                agreement_count += 1;
            }
            total_confidence += result.confidence;
        }

        let agreement_ratio = agreement_count as f32 / model_results.len() as f32;
        let avg_confidence = total_confidence / model_results.len() as f32;

        // Combine agreement and average confidence
        let cross_model_score = (agreement_ratio * 0.7) + (avg_confidence * 0.3);

        Ok(cross_model_score.min(1.0).max(0.0))
    }

    /// Analyze temporal consistency across phoneme sequence
    pub fn analyze_temporal_consistency(
        &self,
        phonemes: &[AlignedPhoneme],
    ) -> Result<f32, RecognitionError> {
        if phonemes.len() < 2 {
            return Ok(1.0);
        }

        let mut consistency_score = 1.0;
        let mut gap_penalties = 0.0;
        let mut overlap_penalties = 0.0;
        let mut duration_consistency = 0.0;

        for window in phonemes.windows(2) {
            let current = &window[0];
            let next = &window[1];

            // Check for temporal gaps
            let gap = next.start_time - current.end_time;
            if gap > 0.1 {
                gap_penalties += gap * 0.5; // Penalty for large gaps
            }

            // Check for overlaps
            if current.end_time > next.start_time {
                overlap_penalties += (current.end_time - next.start_time) * 2.0;
            }

            // Check duration consistency with expected values
            let current_expected = self.estimate_expected_phoneme_duration(&current.phoneme.symbol);
            let next_expected = self.estimate_expected_phoneme_duration(&next.phoneme.symbol);

            let current_ratio = if current_expected > 0.0 {
                (current.duration() / current_expected).min(current_expected / current.duration())
            } else {
                1.0
            };

            let next_ratio = if next_expected > 0.0 {
                (next.duration() / next_expected).min(next_expected / next.duration())
            } else {
                1.0
            };

            duration_consistency += (current_ratio + next_ratio) / 2.0;
        }

        // Calculate final consistency score
        consistency_score -= gap_penalties.min(0.3);
        consistency_score -= overlap_penalties.min(0.4);

        let avg_duration_consistency = duration_consistency / (phonemes.len() - 1) as f32;
        consistency_score = (consistency_score * 0.6) + (avg_duration_consistency * 0.4);

        Ok(consistency_score.min(1.0).max(0.0))
    }

    /// Calculate quality factor based on signal metrics
    fn calculate_quality_factor(&self, quality: &SignalQuality) -> Result<f32, RecognitionError> {
        let mut factor = 1.0;

        // SNR factor
        if quality.snr_db >= 20.0 {
            factor *= 1.0;
        } else if quality.snr_db >= 10.0 {
            factor *= 0.8 + (quality.snr_db - 10.0) * 0.02; // Linear interpolation
        } else {
            factor *= 0.5 + quality.snr_db * 0.03;
        }

        // THD factor
        if quality.thd_percent <= 5.0 {
            factor *= 1.0;
        } else if quality.thd_percent <= 15.0 {
            factor *= 1.0 - (quality.thd_percent - 5.0) * 0.03;
        } else {
            factor *= 0.7;
        }

        // Recording quality factor
        factor *= quality.recording_quality;

        Ok(factor.min(1.0).max(0.1))
    }

    /// Calculate speaker factor based on characteristics
    fn calculate_speaker_factor(&self, speaker: &SpeakerInfo) -> Result<f32, RecognitionError> {
        let mut factor = 1.0;

        // Speaking rate factor
        if let Some(rate) = speaker.speaking_rate {
            if (8.0..=16.0).contains(&rate) {
                factor *= 1.0; // Normal rate
            } else if !(8.0..=16.0).contains(&rate) {
                factor *= 0.9; // Slower or faster than normal
            }
            if !(4.0..=24.0).contains(&rate) {
                factor *= 0.7; // Very slow or very fast
            }
        }

        // Voice characteristics factor
        match speaker.voice_characteristics.quality {
            VoiceQuality::Clear => factor *= 1.0,
            VoiceQuality::Unclear => factor *= 0.8,
            VoiceQuality::Hoarse => factor *= 0.7,
            VoiceQuality::Breathy => factor *= 0.9,
            VoiceQuality::Unknown => factor *= 0.95,
        }

        // Clarity factor
        factor *= speaker.voice_characteristics.clarity;

        Ok(factor.min(1.0).max(0.3))
    }

    /// Estimate expected phoneme duration
    fn estimate_expected_phoneme_duration(&self, phoneme_symbol: &str) -> f32 {
        // Rough estimates for common phonemes (in seconds)
        match phoneme_symbol {
            // Vowels - typically longer
            "a" | "e" | "i" | "o" | "u" | "ə" | "ɛ" | "ɪ" | "ɔ" | "ʊ" | "æ" => 0.12,
            // Plosives - typically shorter
            "p" | "b" | "t" | "d" | "k" | "g" => 0.08,
            // Fricatives - medium duration
            "f" | "v" | "s" | "z" | "θ" | "ð" | "ʃ" | "ʒ" | "h" => 0.10,
            // Nasals - medium duration
            "m" | "n" | "ŋ" => 0.09,
            // Liquids - medium duration
            "l" | "r" | "ɹ" => 0.09,
            // Glides - shorter
            "w" | "j" => 0.07,
            // Default
            _ => 0.10,
        }
    }

    /// Estimate expected word duration
    fn estimate_expected_word_duration(&self, word: &str) -> f32 {
        // Simple estimate: 0.1 seconds per character
        // In a real implementation, this would use phoneme-based estimation
        word.len() as f32 * 0.1
    }

    /// Classify alignment quality based on confidence
    #[must_use]
    pub fn classify_alignment_quality(&self, confidence: f32) -> AlignmentQuality {
        if confidence >= self.alignment_thresholds.high_quality {
            AlignmentQuality::High
        } else if confidence >= self.alignment_thresholds.acceptable {
            AlignmentQuality::Acceptable
        } else {
            AlignmentQuality::Low
        }
    }
}

/// Alignment quality classification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlignmentQuality {
    /// High quality alignment (>= 0.8 confidence)
    High,
    /// Acceptable alignment (>= 0.6 confidence)
    Acceptable,
    /// Low quality alignment (< 0.6 confidence)
    Low,
}

/// Context information for confidence estimation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AlignmentContext {
    /// Acoustic model scores for each phoneme
    pub acoustic_scores: HashMap<String, f32>,
    /// Language model scores
    pub language_scores: HashMap<String, f32>,
    /// Signal quality metrics
    pub signal_quality: SignalQuality,
    /// Speaker characteristics
    pub speaker_info: SpeakerInfo,
}

/// Signal quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalQuality {
    /// Signal-to-noise ratio
    pub snr_db: f32,
    /// Total harmonic distortion
    pub thd_percent: f32,
    /// Recording quality score
    pub recording_quality: f32,
}

/// Speaker information for context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerInfo {
    /// Native language (if known)
    pub native_language: Option<String>,
    /// Speaking rate (phonemes per second)
    pub speaking_rate: Option<f32>,
    /// Voice characteristics
    pub voice_characteristics: VoiceCharacteristics,
}

/// Voice characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCharacteristics {
    /// Fundamental frequency range
    pub f0_range: (f32, f32),
    /// Voice quality descriptor
    pub quality: VoiceQuality,
    /// Speech clarity level
    pub clarity: f32,
}

/// Model result for cross-model agreement analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResult {
    /// Model name/identifier
    pub model_name: String,
    /// Predicted phoneme symbol
    pub predicted_phoneme: String,
    /// Model confidence score
    pub confidence: f32,
    /// Additional model-specific data
    pub metadata: HashMap<String, f32>,
}

/// Voice quality types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VoiceQuality {
    /// Clear, well-articulated speech
    Clear,
    /// Slightly unclear or muffled
    Unclear,
    /// Hoarse or rough voice quality
    Hoarse,
    /// Breathy voice quality
    Breathy,
    /// Unknown quality
    Unknown,
}

impl Default for VoiceCharacteristics {
    fn default() -> Self {
        Self {
            f0_range: (80.0, 300.0),
            quality: VoiceQuality::Unknown,
            clarity: 0.8,
        }
    }
}

impl Default for SpeakerInfo {
    fn default() -> Self {
        Self {
            native_language: None,
            speaking_rate: Some(12.0), // phonemes per second
            voice_characteristics: VoiceCharacteristics::default(),
        }
    }
}

impl Default for SignalQuality {
    fn default() -> Self {
        Self {
            snr_db: 20.0,
            thd_percent: 5.0,
            recording_quality: 0.8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voirs_sdk::{types::SyllablePosition, Phoneme};

    fn create_test_phoneme() -> AlignedPhoneme {
        AlignedPhoneme {
            phoneme: Phoneme {
                symbol: "a".to_string(),
                ipa_symbol: "a".to_string(),
                stress: 0,
                syllable_position: SyllablePosition::Nucleus,
                duration_ms: Some(120.0),
                confidence: 0.9,
            },
            start_time: 0.0,
            end_time: 0.12,
            confidence: 0.9,
        }
    }

    #[test]
    fn test_confidence_estimator_creation() {
        let estimator = ConfidenceEstimator::new();
        assert!(estimator.alignment_thresholds.high_quality > 0.0);
        assert!(estimator.alignment_thresholds.acceptable > 0.0);
    }

    #[test]
    fn test_phoneme_confidence_estimation() {
        let estimator = ConfidenceEstimator::new();
        let phoneme = create_test_phoneme();
        let context = AlignmentContext::default();

        let confidence = estimator.estimate_phoneme_confidence(&phoneme, &context);
        assert!(confidence.is_ok());
        let confidence_value = confidence.unwrap();
        assert!((0.0..=1.0).contains(&confidence_value));
    }

    #[test]
    fn test_alignment_quality_classification() {
        let estimator = ConfidenceEstimator::new();

        assert_eq!(
            estimator.classify_alignment_quality(0.9),
            AlignmentQuality::High
        );
        assert_eq!(
            estimator.classify_alignment_quality(0.7),
            AlignmentQuality::Acceptable
        );
        assert_eq!(
            estimator.classify_alignment_quality(0.5),
            AlignmentQuality::Low
        );
    }

    #[test]
    fn test_duration_consistency() {
        let estimator = ConfidenceEstimator::new();
        let phoneme = create_test_phoneme();
        let context = AlignmentContext::default();

        let consistency = estimator.calculate_duration_consistency(&phoneme, &context);
        assert!(consistency.is_ok());
        let consistency_value = consistency.unwrap();
        assert!((0.0..=1.0).contains(&consistency_value));
    }

    #[test]
    fn test_expected_phoneme_duration() {
        let estimator = ConfidenceEstimator::new();

        // Test vowel duration
        let vowel_duration = estimator.estimate_expected_phoneme_duration("a");
        assert!(vowel_duration > 0.0);

        // Test consonant duration
        let consonant_duration = estimator.estimate_expected_phoneme_duration("p");
        assert!(consonant_duration > 0.0);

        // Vowels should generally be longer than plosives
        assert!(vowel_duration > consonant_duration);
    }

    #[test]
    fn test_acoustic_likelihood_estimation() {
        let estimator = ConfidenceEstimator::new();
        let phoneme = create_test_phoneme();
        let mut context = AlignmentContext::default();

        // Add acoustic score
        context.acoustic_scores.insert("a".to_string(), 0.8);

        let likelihood = estimator.estimate_acoustic_likelihood(&phoneme, &context);
        assert!(likelihood.is_ok());
        let likelihood_value = likelihood.unwrap();
        assert!((0.0..=1.0).contains(&likelihood_value));
    }

    #[test]
    fn test_cross_model_agreement() {
        let estimator = ConfidenceEstimator::new();
        let phoneme = create_test_phoneme();

        let model_results = vec![
            ModelResult {
                model_name: "model1".to_string(),
                predicted_phoneme: "a".to_string(),
                confidence: 0.9,
                metadata: HashMap::new(),
            },
            ModelResult {
                model_name: "model2".to_string(),
                predicted_phoneme: "a".to_string(),
                confidence: 0.8,
                metadata: HashMap::new(),
            },
            ModelResult {
                model_name: "model3".to_string(),
                predicted_phoneme: "e".to_string(), // Disagreement
                confidence: 0.7,
                metadata: HashMap::new(),
            },
        ];

        let agreement = estimator.calculate_cross_model_agreement(&phoneme, &model_results);
        assert!(agreement.is_ok());
        let agreement_value = agreement.unwrap();

        // Should be reasonable agreement since 2/3 models agree
        assert!((0.5..=1.0).contains(&agreement_value));
    }

    #[test]
    fn test_temporal_consistency_analysis() {
        let estimator = ConfidenceEstimator::new();

        let phonemes = vec![
            create_test_phoneme_with_timing("h", 0.0, 0.05),
            create_test_phoneme_with_timing("ɛ", 0.05, 0.15),
            create_test_phoneme_with_timing("l", 0.15, 0.20),
            create_test_phoneme_with_timing("oʊ", 0.20, 0.35),
        ];

        let consistency = estimator.analyze_temporal_consistency(&phonemes);
        assert!(consistency.is_ok());
        let consistency_value = consistency.unwrap();
        assert!((0.0..=1.0).contains(&consistency_value));

        // Should be high consistency for well-aligned phonemes
        assert!(consistency_value > 0.7);
    }

    #[test]
    fn test_temporal_consistency_with_overlaps() {
        let estimator = ConfidenceEstimator::new();

        let phonemes = vec![
            create_test_phoneme_with_timing("a", 0.0, 0.15),
            create_test_phoneme_with_timing("b", 0.10, 0.20), // Overlap!
        ];

        let consistency = estimator.analyze_temporal_consistency(&phonemes);
        assert!(consistency.is_ok());
        let consistency_value = consistency.unwrap();

        // Should have lower consistency due to overlap (adjust expectation)
        assert!(consistency_value < 1.0);

        // Test without overlap for comparison
        let good_phonemes = vec![
            create_test_phoneme_with_timing("a", 0.0, 0.10),
            create_test_phoneme_with_timing("b", 0.10, 0.20), // No overlap
        ];

        let good_consistency = estimator
            .analyze_temporal_consistency(&good_phonemes)
            .unwrap();

        // Overlapping should have lower consistency than non-overlapping
        assert!(consistency_value <= good_consistency);
    }

    #[test]
    fn test_quality_factor_calculation() {
        let estimator = ConfidenceEstimator::new();

        // High quality signal
        let high_quality = SignalQuality {
            snr_db: 25.0,
            thd_percent: 2.0,
            recording_quality: 0.95,
        };

        let factor = estimator.calculate_quality_factor(&high_quality);
        assert!(factor.is_ok());
        let factor_value = factor.unwrap();
        assert!(factor_value > 0.9);

        // Low quality signal
        let low_quality = SignalQuality {
            snr_db: 5.0,
            thd_percent: 20.0,
            recording_quality: 0.4,
        };

        let factor = estimator.calculate_quality_factor(&low_quality);
        assert!(factor.is_ok());
        let factor_value = factor.unwrap();
        assert!(factor_value < 0.5);
    }

    #[test]
    fn test_speaker_factor_calculation() {
        let estimator = ConfidenceEstimator::new();

        // Clear speaker
        let clear_speaker = SpeakerInfo {
            native_language: Some("en".to_string()),
            speaking_rate: Some(12.0), // Normal rate
            voice_characteristics: VoiceCharacteristics {
                f0_range: (100.0, 200.0),
                quality: VoiceQuality::Clear,
                clarity: 0.95,
            },
        };

        let factor = estimator.calculate_speaker_factor(&clear_speaker);
        assert!(factor.is_ok());
        let factor_value = factor.unwrap();
        assert!(factor_value > 0.9);

        // Unclear speaker
        let unclear_speaker = SpeakerInfo {
            native_language: Some("unknown".to_string()),
            speaking_rate: Some(25.0), // Very fast
            voice_characteristics: VoiceCharacteristics {
                f0_range: (80.0, 300.0),
                quality: VoiceQuality::Hoarse,
                clarity: 0.5,
            },
        };

        let factor = estimator.calculate_speaker_factor(&unclear_speaker);
        assert!(factor.is_ok());
        let factor_value = factor.unwrap();
        assert!(factor_value < 0.7);
    }

    fn create_test_phoneme_with_timing(symbol: &str, start: f32, end: f32) -> AlignedPhoneme {
        AlignedPhoneme {
            phoneme: Phoneme {
                symbol: symbol.to_string(),
                ipa_symbol: symbol.to_string(),
                stress: 0,
                syllable_position: SyllablePosition::Nucleus,
                duration_ms: Some((end - start) * 1000.0),
                confidence: 0.9,
            },
            start_time: start,
            end_time: end,
            confidence: 0.9,
        }
    }
}
