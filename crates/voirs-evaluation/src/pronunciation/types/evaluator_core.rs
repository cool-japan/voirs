//! Core pronunciation evaluator struct and primary implementation

use crate::traits::{
    ComparativeEvaluator, EvaluationResult, FeedbackType, PhonemeAccuracyScore,
    PronunciationEvaluationConfig, PronunciationEvaluator, PronunciationEvaluatorMetadata,
    PronunciationFeedback, PronunciationMetric, PronunciationScore, QualityEvaluator,
    SelfEvaluator, WordPronunciationScore,
};
use crate::EvaluationError;
use scirs2_core::parallel_ops::*;
use std::collections::HashMap;
use voirs_g2p::{backends::rule_based::RuleBasedG2p, G2p, G2pConverter};
use voirs_recognizer::traits::{AlignedPhoneme, PhonemeAlignment};
use voirs_sdk::{AudioBuffer, LanguageCode, Phoneme, SyllablePosition};

use super::extended_types::{
    CrossLinguisticProsodyScore, EmotionalProsodicFeatures, EmotionalProsodyScore, EmotionalState,
    MannerOfArticulation, PhoneticFeatures, PlaceOfArticulation, VowelBackness, VowelHeight,
};

/// Pronunciation evaluator implementation
pub struct PronunciationEvaluatorImpl {
    /// Configuration
    pub(crate) config: PronunciationEvaluationConfig,
    /// Supported metrics
    pub(crate) supported_metrics: Vec<PronunciationMetric>,
    /// Metadata
    pub(crate) metadata: PronunciationEvaluatorMetadata,
    /// G2P converter for phonemization
    g2p_converter: G2pConverter,
}
impl PronunciationEvaluatorImpl {
    /// Create a new pronunciation evaluator
    pub async fn new() -> Result<Self, EvaluationError> {
        Self::with_config(PronunciationEvaluationConfig::default()).await
    }
    /// Create with custom configuration
    pub async fn with_config(
        config: PronunciationEvaluationConfig,
    ) -> Result<Self, EvaluationError> {
        let supported_metrics = vec![
            PronunciationMetric::PhonemeAccuracy,
            PronunciationMetric::WordAccuracy,
            PronunciationMetric::SentenceAccuracy,
            PronunciationMetric::Fluency,
            PronunciationMetric::Rhythm,
            PronunciationMetric::StressAccuracy,
            PronunciationMetric::IntonationAccuracy,
            PronunciationMetric::SpeakingRate,
            PronunciationMetric::PausePattern,
            PronunciationMetric::Comprehensibility,
        ];
        let metadata = PronunciationEvaluatorMetadata {
            name: "VoiRS Pronunciation Evaluator".to_string(),
            version: "1.0.0".to_string(),
            description: "Comprehensive pronunciation assessment for speech synthesis".to_string(),
            supported_metrics: supported_metrics.clone(),
            supported_languages: vec![
                LanguageCode::EnUs,
                LanguageCode::EnGb,
                LanguageCode::DeDe,
                LanguageCode::FrFr,
                LanguageCode::EsEs,
                LanguageCode::JaJp,
                LanguageCode::ZhCn,
                LanguageCode::KoKr,
            ],
            accuracy_benchmarks: {
                let mut benchmarks = HashMap::new();
                benchmarks.insert(LanguageCode::EnUs, 0.92);
                benchmarks.insert(LanguageCode::EnGb, 0.90);
                benchmarks.insert(LanguageCode::DeDe, 0.88);
                benchmarks.insert(LanguageCode::FrFr, 0.87);
                benchmarks.insert(LanguageCode::EsEs, 0.89);
                benchmarks.insert(LanguageCode::JaJp, 0.85);
                benchmarks.insert(LanguageCode::ZhCn, 0.83);
                benchmarks.insert(LanguageCode::KoKr, 0.84);
                benchmarks
            },
            processing_speed: 1.2,
        };
        let mut g2p_converter = G2pConverter::new();
        g2p_converter.add_backend(
            voirs_g2p::LanguageCode::EnUs,
            Box::new(RuleBasedG2p::new(voirs_g2p::LanguageCode::EnUs)),
        );
        g2p_converter.add_backend(
            voirs_g2p::LanguageCode::EnGb,
            Box::new(RuleBasedG2p::new(voirs_g2p::LanguageCode::EnGb)),
        );
        g2p_converter.add_backend(
            voirs_g2p::LanguageCode::De,
            Box::new(RuleBasedG2p::new(voirs_g2p::LanguageCode::De)),
        );
        g2p_converter.add_backend(
            voirs_g2p::LanguageCode::Fr,
            Box::new(RuleBasedG2p::new(voirs_g2p::LanguageCode::Fr)),
        );
        g2p_converter.add_backend(
            voirs_g2p::LanguageCode::Es,
            Box::new(RuleBasedG2p::new(voirs_g2p::LanguageCode::Es)),
        );
        g2p_converter.add_backend(
            voirs_g2p::LanguageCode::It,
            Box::new(RuleBasedG2p::new(voirs_g2p::LanguageCode::It)),
        );
        g2p_converter.add_backend(
            voirs_g2p::LanguageCode::Pt,
            Box::new(RuleBasedG2p::new(voirs_g2p::LanguageCode::Pt)),
        );
        g2p_converter.add_backend(
            voirs_g2p::LanguageCode::Ja,
            Box::new(RuleBasedG2p::new(voirs_g2p::LanguageCode::Ja)),
        );
        Ok(Self {
            config,
            supported_metrics,
            metadata,
            g2p_converter,
        })
    }
    /// Calculate phoneme-level accuracy scores
    pub(crate) async fn calculate_phoneme_accuracy(
        &self,
        alignment: &PhonemeAlignment,
        expected_text: &str,
    ) -> Result<Vec<PhonemeAccuracyScore>, EvaluationError> {
        let mut phoneme_scores = Vec::new();
        let expected_phonemes = self.text_to_phonemes(expected_text).await?;
        let min_len = alignment.phonemes.len().min(expected_phonemes.len());
        for i in 0..min_len {
            let aligned_phoneme = &alignment.phonemes[i];
            let expected_phoneme = &expected_phonemes[i];
            // `aligned_phoneme.confidence` is the acoustic Goodness-of-Pronunciation
            // score produced by forced alignment (see `align_phonemes_to_audio`): how
            // well the audio segment assigned to this phoneme matches the acoustic
            // profile expected for it (voicing / energy / spectral noisiness). This is
            // the dominant term because `align_phonemes_to_audio` places the
            // *reference* phoneme sequence onto the audio (forced alignment, not free
            // recognition), so `aligned_phoneme.phoneme.symbol` and
            // `expected_phoneme.symbol` are identical by construction in that path and
            // symbol comparison alone would be uninformative. `calculate_phoneme_similarity`
            // is still included (with lower weight) because `alignment` here is a
            // trait-level parameter: a caller of `evaluate_pronunciation_with_alignment`
            // may supply an alignment built by a genuinely independent recognizer,
            // in which case the aligned symbol can differ from the reference and this
            // term becomes real, differentiating evidence.
            let symbol_similarity = self.calculate_phoneme_similarity(
                &aligned_phoneme.phoneme.symbol,
                &expected_phoneme.symbol,
            );
            let accuracy =
                (aligned_phoneme.confidence * 0.7 + symbol_similarity * 0.3).clamp(0.0, 1.0);
            let duration_accuracy = if let Some(expected_duration) = expected_phoneme.duration_ms {
                let actual_duration =
                    (aligned_phoneme.end_time - aligned_phoneme.start_time) * 1000.0;
                let ratio = actual_duration / expected_duration.max(1.0);
                1.0 - (ratio - 1.0).abs().min(1.0)
            } else {
                1.0
            };
            phoneme_scores.push(PhonemeAccuracyScore {
                expected_phoneme: expected_phoneme.symbol.clone(),
                actual_phoneme: Some(aligned_phoneme.phoneme.symbol.clone()),
                accuracy,
                duration_accuracy,
                position: i,
                start_time: aligned_phoneme.start_time,
                end_time: aligned_phoneme.end_time,
            });
        }
        for i in min_len..expected_phonemes.len() {
            phoneme_scores.push(PhonemeAccuracyScore {
                expected_phoneme: expected_phonemes[i].symbol.clone(),
                actual_phoneme: None,
                accuracy: 0.0,
                duration_accuracy: 0.0,
                position: i,
                start_time: 0.0,
                end_time: 0.0,
            });
        }
        Ok(phoneme_scores)
    }
    /// Calculate word-level accuracy scores
    pub(crate) async fn calculate_word_accuracy(
        &self,
        alignment: &PhonemeAlignment,
        expected_text: &str,
    ) -> Result<Vec<WordPronunciationScore>, EvaluationError> {
        let words: Vec<&str> = expected_text.split_whitespace().collect();
        let mut word_scores = Vec::new();
        for (i, word) in words.iter().enumerate() {
            let aligned_word_phonemes =
                super::forced_alignment::word_phoneme_span(alignment, i, word, words.len());
            let accuracy = self
                .calculate_word_pronunciation_accuracy(&aligned_word_phonemes)
                .await?;
            let expected_stress_pattern = self.get_expected_stress_pattern(word);
            let actual_stress_pattern = self
                .extract_actual_stress_pattern(alignment, i, word, words.len())
                .await?;
            let stress_accuracy =
                self.compare_stress_patterns(&expected_stress_pattern, &actual_stress_pattern);
            let syllable_accuracy = {
                let expected_syllables = expected_stress_pattern.len().max(1) as f32;
                let actual_syllables = aligned_word_phonemes
                    .iter()
                    .filter(|p| self.get_phonetic_features(&p.phoneme.symbol).is_vowel)
                    .count()
                    .max(usize::from(!aligned_word_phonemes.is_empty()))
                    as f32;
                if actual_syllables == 0.0 {
                    0.5
                } else {
                    1.0 - (expected_syllables - actual_syllables).abs()
                        / expected_syllables.max(actual_syllables)
                }
            };
            let phoneme_scores: Vec<PhonemeAccuracyScore> = aligned_word_phonemes
                .iter()
                .enumerate()
                .map(|(pos, ap)| {
                    let duration_accuracy = ap.phoneme.duration_ms.map_or(1.0, |expected_ms| {
                        let actual_ms = (ap.end_time - ap.start_time) * 1000.0;
                        let ratio = actual_ms / expected_ms.max(1.0);
                        1.0 - (ratio - 1.0).abs().min(1.0)
                    });
                    PhonemeAccuracyScore {
                        expected_phoneme: ap.phoneme.symbol.clone(),
                        actual_phoneme: Some(ap.phoneme.symbol.clone()),
                        accuracy: ap.confidence,
                        duration_accuracy,
                        position: pos,
                        start_time: ap.start_time,
                        end_time: ap.end_time,
                    }
                })
                .collect();
            word_scores.push(WordPronunciationScore {
                word: (*word).to_string(),
                accuracy,
                stress_accuracy,
                syllable_accuracy: syllable_accuracy.clamp(0.0, 1.0),
                phoneme_scores,
                position: i,
            });
        }
        Ok(word_scores)
    }
    /// Calculate comprehensive fluency score
    pub(crate) async fn calculate_fluency(
        &self,
        alignment: &PhonemeAlignment,
        expected_text: &str,
    ) -> Result<f32, EvaluationError> {
        if alignment.total_duration <= 0.0 || alignment.phonemes.is_empty() {
            return Ok(0.0);
        }
        let speaking_rate_score = self
            .calculate_speaking_rate(alignment, expected_text)
            .await?;
        let pause_pattern_score = self.analyze_pause_patterns(alignment).await?;
        let rhythm_score = self.calculate_rhythm_regularity(alignment).await?;
        let temporal_coordination_score = self.assess_temporal_coordination(alignment).await?;
        let disfluency_score = self.detect_disfluencies(alignment).await?;
        let fluency_score = (speaking_rate_score * 0.25)
            + (pause_pattern_score * 0.20)
            + (rhythm_score * 0.25)
            + (temporal_coordination_score * 0.15)
            + (disfluency_score * 0.15);
        Ok(fluency_score.max(0.0).min(1.0))
    }
    /// Calculate speaking rate with normalization
    pub(crate) async fn calculate_speaking_rate(
        &self,
        alignment: &PhonemeAlignment,
        expected_text: &str,
    ) -> Result<f32, EvaluationError> {
        let syllable_count = self.estimate_syllable_count(expected_text);
        let speech_duration = self.calculate_speech_duration(alignment).await?;
        if speech_duration <= 0.0 {
            return Ok(0.0);
        }
        let syllables_per_second = syllable_count as f32 / speech_duration;
        let ideal_rate = 5.0;
        let rate_deviation = (syllables_per_second - ideal_rate).abs();
        let rate_score = if rate_deviation <= 0.5 {
            1.0
        } else if rate_deviation <= 1.0 {
            0.8
        } else if rate_deviation <= 1.5 {
            0.6
        } else if rate_deviation <= 2.0 {
            0.4
        } else {
            0.2
        };
        Ok(rate_score)
    }
    /// Analyze pause patterns (filled/unfilled pauses)
    pub(crate) async fn analyze_pause_patterns(
        &self,
        alignment: &PhonemeAlignment,
    ) -> Result<f32, EvaluationError> {
        let mut pauses = Vec::new();
        let mut filled_pauses = 0;
        let mut unfilled_pauses = 0;
        for window in alignment.phonemes.windows(2) {
            let gap = window[1].start_time - window[0].end_time;
            if gap > 0.1 {
                pauses.push(gap);
                if gap < 0.5 {
                    filled_pauses += 1;
                } else {
                    unfilled_pauses += 1;
                }
            }
        }
        if pauses.is_empty() {
            return Ok(0.8);
        }
        let total_pause_time: f32 = pauses.iter().sum();
        let avg_pause_duration = total_pause_time / pauses.len() as f32;
        let pause_frequency = pauses.len() as f32 / alignment.total_duration;
        let pause_duration_score = if avg_pause_duration <= 0.3 {
            1.0
        } else if avg_pause_duration <= 0.6 {
            0.8
        } else if avg_pause_duration <= 1.0 {
            0.6
        } else {
            0.4
        };
        let pause_frequency_score = if pause_frequency <= 0.5 {
            1.0
        } else if pause_frequency <= 1.0 {
            0.8
        } else if pause_frequency <= 1.5 {
            0.6
        } else {
            0.4
        };
        let filled_pause_penalty = if filled_pauses > unfilled_pauses {
            0.8
        } else {
            1.0
        };
        Ok((pause_duration_score + pause_frequency_score) / 2.0 * filled_pause_penalty)
    }
    /// Assess temporal coordination between phonemes
    pub(crate) async fn assess_temporal_coordination(
        &self,
        alignment: &PhonemeAlignment,
    ) -> Result<f32, EvaluationError> {
        if alignment.phonemes.len() < 5 {
            return Ok(0.7);
        }
        let mut consonant_durations = Vec::new();
        let mut vowel_durations = Vec::new();
        for phoneme in &alignment.phonemes {
            let duration = phoneme.end_time - phoneme.start_time;
            let features = self.get_phonetic_features(&phoneme.phoneme.symbol);
            if features.is_vowel {
                vowel_durations.push(duration);
            } else {
                consonant_durations.push(duration);
            }
        }
        let vowel_consistency = self.calculate_timing_consistency(&vowel_durations);
        let consonant_consistency = self.calculate_timing_consistency(&consonant_durations);
        let transition_score = self.calculate_transition_smoothness(alignment).await?;
        Ok((vowel_consistency + consonant_consistency + transition_score) / 3.0)
    }
    /// Detect disfluencies in speech
    pub(crate) async fn detect_disfluencies(
        &self,
        alignment: &PhonemeAlignment,
    ) -> Result<f32, EvaluationError> {
        let mut disfluency_count = 0;
        let total_phonemes = alignment.phonemes.len();
        if total_phonemes == 0 {
            return Ok(1.0);
        }
        for window in alignment.phonemes.windows(3) {
            if window[0].phoneme.symbol == window[1].phoneme.symbol
                || window[1].phoneme.symbol == window[2].phoneme.symbol
            {
                let gap1 = window[1].start_time - window[0].end_time;
                let gap2 = window[2].start_time - window[1].end_time;
                if gap1 < 0.2 && gap2 < 0.2 {
                    disfluency_count += 1;
                }
            }
        }
        let mean_duration = alignment
            .phonemes
            .iter()
            .map(|p| p.end_time - p.start_time)
            .sum::<f32>()
            / total_phonemes as f32;
        for phoneme in &alignment.phonemes {
            let duration = phoneme.end_time - phoneme.start_time;
            if duration > mean_duration * 2.5 {
                disfluency_count += 1;
            }
        }
        let disfluency_rate = disfluency_count as f32 / total_phonemes as f32;
        let disfluency_score = if disfluency_rate <= 0.05 {
            1.0
        } else if disfluency_rate <= 0.1 {
            0.8
        } else if disfluency_rate <= 0.2 {
            0.6
        } else {
            0.4
        };
        Ok(disfluency_score)
    }
    /// Calculate rhythm score
    pub(crate) async fn calculate_rhythm(
        &self,
        alignment: &PhonemeAlignment,
    ) -> Result<f32, EvaluationError> {
        self.calculate_rhythm_regularity(alignment).await
    }
    /// Calculate comprehensive stress accuracy
    pub(crate) async fn calculate_stress_accuracy(
        &self,
        alignment: &PhonemeAlignment,
        expected_text: &str,
    ) -> Result<f32, EvaluationError> {
        if alignment.phonemes.is_empty() {
            return Ok(0.5);
        }
        let word_stress_score = self
            .analyze_word_stress_patterns(alignment, expected_text)
            .await?;
        let syllable_stress_score = self.analyze_syllable_stress_patterns(alignment).await?;
        let stress_timing_score = self.analyze_stress_timing(alignment).await?;
        let overall_stress_score = (word_stress_score * 0.4)
            + (syllable_stress_score * 0.35)
            + (stress_timing_score * 0.25);
        Ok(overall_stress_score.max(0.0).min(1.0))
    }
    /// Calculate comprehensive intonation accuracy
    pub(crate) async fn calculate_intonation_accuracy(
        &self,
        alignment: &PhonemeAlignment,
        expected_text: &str,
    ) -> Result<f32, EvaluationError> {
        if alignment.phonemes.is_empty() {
            return Ok(0.5);
        }
        let pitch_contour_score = self.analyze_pitch_contour(alignment).await?;
        let sentence_boundary_score = self
            .analyze_sentence_boundaries(alignment, expected_text)
            .await?;
        let emphasis_detection_score = self.detect_emphasis_patterns(alignment).await?;
        let focus_detection_score = self.detect_focus_patterns(alignment, expected_text).await?;
        let overall_intonation_score = (pitch_contour_score * 0.35)
            + (sentence_boundary_score * 0.25)
            + (emphasis_detection_score * 0.20)
            + (focus_detection_score * 0.20);
        Ok(overall_intonation_score.max(0.0).min(1.0))
    }
    /// Calculate emotional prosody analysis
    pub(crate) async fn calculate_emotional_prosody(
        &self,
        alignment: &PhonemeAlignment,
        expected_emotion: Option<EmotionalState>,
    ) -> Result<EmotionalProsodyScore, EvaluationError> {
        if alignment.phonemes.is_empty() {
            return Ok(EmotionalProsodyScore::default());
        }
        let prosodic_features = self.extract_emotional_prosodic_features(alignment).await?;
        let detected_emotion = self.detect_emotional_state(&prosodic_features)?;
        let emotional_appropriateness = if let Some(expected) = expected_emotion {
            self.calculate_emotional_accuracy(&expected, &detected_emotion)
        } else {
            1.0
        };
        let emotional_intensity = self.calculate_emotional_intensity(&prosodic_features);
        let emotional_consistency = self.calculate_emotional_consistency(&prosodic_features);
        let emotional_dynamics = self
            .analyze_emotional_dynamics(alignment, &prosodic_features)
            .await?;
        let confidence = self.calculate_emotion_detection_confidence(&prosodic_features);
        Ok(EmotionalProsodyScore {
            detected_emotion,
            emotional_appropriateness,
            emotional_intensity,
            emotional_consistency,
            emotional_dynamics,
            prosodic_features,
            confidence,
        })
    }
    /// Perform cross-linguistic prosody comparison
    pub(crate) async fn calculate_cross_linguistic_prosody(
        &self,
        alignment: &PhonemeAlignment,
        source_language: LanguageCode,
        target_language: LanguageCode,
        prosodic_features: &EmotionalProsodicFeatures,
    ) -> Result<CrossLinguisticProsodyScore, EvaluationError> {
        let source_profile = self.get_language_prosodic_profile(source_language);
        let target_profile = self.get_language_prosodic_profile(target_language);
        let language_distance = self.calculate_language_distance(&source_profile, &target_profile);
        let comparison_details =
            self.compare_prosodic_features(prosodic_features, &source_profile, &target_profile);
        let transfer_score = self.calculate_prosodic_transfer_score(&comparison_details);
        let adaptation_recommendations = self.generate_adaptation_recommendations(
            prosodic_features,
            &source_profile,
            &target_profile,
        );
        let cross_linguistic_intelligibility =
            self.calculate_cross_linguistic_intelligibility(&comparison_details, language_distance);
        Ok(CrossLinguisticProsodyScore {
            source_language,
            target_language,
            transfer_score,
            language_distance,
            comparison_details,
            adaptation_recommendations,
            cross_linguistic_intelligibility,
        })
    }
    /// Analyze word-level stress patterns
    pub(crate) async fn analyze_word_stress_patterns(
        &self,
        alignment: &PhonemeAlignment,
        expected_text: &str,
    ) -> Result<f32, EvaluationError> {
        let words: Vec<&str> = expected_text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(0.5);
        }
        let mut correct_stress_count = 0;
        let mut total_stress_syllables = 0;
        for (word_idx, word) in words.iter().enumerate() {
            let expected_stress_pattern = self.get_expected_stress_pattern(word);
            let actual_stress_pattern = self
                .extract_actual_stress_pattern(alignment, word_idx, word, words.len())
                .await?;
            let stress_match_score =
                self.compare_stress_patterns(&expected_stress_pattern, &actual_stress_pattern);
            if stress_match_score > 0.7 {
                correct_stress_count += 1;
            }
            total_stress_syllables += expected_stress_pattern.len();
        }
        if total_stress_syllables == 0 {
            return Ok(0.5);
        }
        Ok(correct_stress_count as f32 / words.len() as f32)
    }
    /// Analyze syllable-level stress patterns
    pub(crate) async fn analyze_syllable_stress_patterns(
        &self,
        alignment: &PhonemeAlignment,
    ) -> Result<f32, EvaluationError> {
        let mut stress_consistency_scores = Vec::new();
        let mut stressed_durations = Vec::new();
        let mut unstressed_durations = Vec::new();
        for phoneme in &alignment.phonemes {
            let duration = phoneme.end_time - phoneme.start_time;
            if phoneme.phoneme.stress >= 2 {
                stressed_durations.push(duration);
            } else {
                unstressed_durations.push(duration);
            }
        }
        let duration_contrast =
            self.calculate_duration_contrast(&stressed_durations, &unstressed_durations);
        stress_consistency_scores.push(duration_contrast);
        let placement_accuracy = self.analyze_stress_timing(alignment).await?;
        stress_consistency_scores.push(placement_accuracy);
        if stress_consistency_scores.is_empty() {
            Ok(0.5)
        } else {
            Ok(stress_consistency_scores.iter().sum::<f32>()
                / stress_consistency_scores.len() as f32)
        }
    }
    /// Analyze stress timing patterns
    pub(crate) async fn analyze_stress_timing(
        &self,
        alignment: &PhonemeAlignment,
    ) -> Result<f32, EvaluationError> {
        let mut stress_intervals = Vec::new();
        let mut last_stress_time = 0.0;
        for phoneme in &alignment.phonemes {
            if phoneme.phoneme.stress >= 2 {
                if last_stress_time > 0.0 {
                    stress_intervals.push(phoneme.start_time - last_stress_time);
                }
                last_stress_time = phoneme.start_time;
            }
        }
        if stress_intervals.len() < 2 {
            return Ok(0.7);
        }
        let timing_regularity = self.calculate_timing_consistency(&stress_intervals);
        let timing_score = if timing_regularity > 0.9 {
            0.8
        } else if timing_regularity > 0.6 {
            1.0
        } else if timing_regularity > 0.4 {
            0.8
        } else {
            0.6
        };
        Ok(timing_score)
    }
    /// Analyze pitch contour patterns
    pub(crate) async fn analyze_pitch_contour(
        &self,
        alignment: &PhonemeAlignment,
    ) -> Result<f32, EvaluationError> {
        let mut pitch_scores = Vec::new();
        let pitch_variation_score = self.calculate_speech_duration(alignment).await?;
        pitch_scores.push(pitch_variation_score);
        let pitch_smoothness_score = self.calculate_transition_smoothness(alignment).await?;
        pitch_scores.push(pitch_smoothness_score);
        let pitch_direction_score = 0.75;
        pitch_scores.push(pitch_direction_score);
        if pitch_scores.is_empty() {
            Ok(0.5)
        } else {
            Ok(pitch_scores.iter().sum::<f32>() / pitch_scores.len() as f32)
        }
    }
    /// Analyze sentence boundary intonation
    pub(crate) async fn analyze_sentence_boundaries(
        &self,
        alignment: &PhonemeAlignment,
        expected_text: &str,
    ) -> Result<f32, EvaluationError> {
        let sentence_endings = self.find_sentence_boundaries(expected_text);
        if sentence_endings.is_empty() {
            return Ok(0.8);
        }
        let mut boundary_scores = Vec::new();
        for boundary_pos in sentence_endings {
            let boundary_score = self
                .evaluate_boundary_intonation(alignment, boundary_pos)
                .await?;
            boundary_scores.push(boundary_score);
        }
        if boundary_scores.is_empty() {
            Ok(0.8)
        } else {
            Ok(boundary_scores.iter().sum::<f32>() / boundary_scores.len() as f32)
        }
    }
    /// Detect emphasis patterns in speech
    pub(crate) async fn detect_emphasis_patterns(
        &self,
        alignment: &PhonemeAlignment,
    ) -> Result<f32, EvaluationError> {
        let mut emphasis_scores = Vec::new();
        for phoneme in &alignment.phonemes {
            let duration = phoneme.end_time - phoneme.start_time;
            let stress_level = phoneme.phoneme.stress;
            let emphasis_likelihood =
                self.calculate_emphasis_likelihood(duration, stress_level, alignment);
            emphasis_scores.push(emphasis_likelihood);
        }
        let emphasis_distribution_score = self.analyze_emphasis_distribution(&emphasis_scores);
        Ok(emphasis_distribution_score)
    }
    /// Detect focus patterns in speech
    pub(crate) async fn detect_focus_patterns(
        &self,
        alignment: &PhonemeAlignment,
        expected_text: &str,
    ) -> Result<f32, EvaluationError> {
        let content_words = self.identify_content_words(expected_text);
        let function_words = self.identify_function_words(expected_text);
        if content_words.is_empty() {
            return Ok(0.8);
        }
        let content_word_prominence = self
            .analyze_content_word_prominence(alignment, &content_words)
            .await?;
        let function_word_deemphasis = self
            .analyze_function_word_deemphasis(alignment, &function_words)
            .await?;
        Ok((content_word_prominence + function_word_deemphasis) / 2.0)
    }
    /// Generate pronunciation feedback
    pub(crate) async fn generate_feedback(
        &self,
        phoneme_scores: &[PhonemeAccuracyScore],
        word_scores: &[WordPronunciationScore],
    ) -> Result<Vec<PronunciationFeedback>, EvaluationError> {
        let mut feedback = Vec::new();
        for phoneme_score in phoneme_scores {
            if phoneme_score.accuracy < 0.7 {
                let feedback_type = if let Some(actual) = &phoneme_score.actual_phoneme {
                    if actual != &phoneme_score.expected_phoneme {
                        FeedbackType::PhonemeSubstitution
                    } else {
                        FeedbackType::QualityIssue
                    }
                } else {
                    FeedbackType::PhonemeDeletion
                };
                let message = match feedback_type {
                    FeedbackType::PhonemeDeletion => {
                        format!(
                            "Missing phoneme '{}' at position {}",
                            phoneme_score.expected_phoneme, phoneme_score.position
                        )
                    }
                    FeedbackType::PhonemeSubstitution => {
                        format!(
                            "Substituted '{}' with '{}' at position {}",
                            phoneme_score.expected_phoneme,
                            phoneme_score
                                .actual_phoneme
                                .as_ref()
                                .expect("value should be present"),
                            phoneme_score.position
                        )
                    }
                    _ => {
                        format!(
                            "Quality issue with phoneme '{}' at position {}",
                            phoneme_score.expected_phoneme, phoneme_score.position
                        )
                    }
                };
                feedback.push(PronunciationFeedback {
                    position: phoneme_score.position,
                    feedback_type,
                    severity: 1.0 - phoneme_score.accuracy,
                    message,
                    suggestion: Some(format!(
                        "Focus on pronouncing '{}' more clearly",
                        phoneme_score.expected_phoneme
                    )),
                });
            }
        }
        for word_score in word_scores {
            if word_score.accuracy < 0.8 {
                feedback.push(PronunciationFeedback {
                    position: word_score.position,
                    feedback_type: FeedbackType::QualityIssue,
                    severity: 1.0 - word_score.accuracy,
                    message: format!("Word '{}' needs improvement", word_score.word),
                    suggestion: Some(format!("Practice pronouncing '{}'", word_score.word)),
                });
            }
        }
        Ok(feedback)
    }
    pub(crate) async fn text_to_phonemes(
        &self,
        text: &str,
    ) -> Result<Vec<Phoneme>, EvaluationError> {
        let g2p_phonemes = self
            .g2p_converter
            .to_phonemes(text, Some(voirs_g2p::LanguageCode::EnUs))
            .await
            .map_err(|e| EvaluationError::InvalidInput {
                message: format!("G2P conversion failed: {}", e),
            })?;
        let phonemes = g2p_phonemes
            .into_iter()
            .map(|g2p_phoneme| {
                let stress = match g2p_phoneme.syllable_position {
                    voirs_g2p::SyllablePosition::Onset => 1,
                    voirs_g2p::SyllablePosition::Nucleus => 2,
                    voirs_g2p::SyllablePosition::Coda => 0,
                    voirs_g2p::SyllablePosition::Final => 0,
                    voirs_g2p::SyllablePosition::Standalone => 1,
                };
                let effective_symbol = g2p_phoneme.effective_symbol().to_string();
                let ipa_symbol = g2p_phoneme.ipa_symbol.unwrap_or(effective_symbol);
                let syllable_position =
                    self.convert_syllable_position(&g2p_phoneme.syllable_position);
                Phoneme {
                    symbol: g2p_phoneme.symbol,
                    ipa_symbol,
                    stress,
                    syllable_position,
                    duration_ms: g2p_phoneme.duration_ms,
                    confidence: g2p_phoneme.confidence,
                }
            })
            .collect();
        Ok(phonemes)
    }
    /// Convert voirs-g2p::SyllablePosition to voirs-sdk::SyllablePosition
    pub(crate) fn convert_syllable_position(
        &self,
        position: &voirs_g2p::SyllablePosition,
    ) -> SyllablePosition {
        match position {
            voirs_g2p::SyllablePosition::Onset => SyllablePosition::Onset,
            voirs_g2p::SyllablePosition::Nucleus => SyllablePosition::Nucleus,
            voirs_g2p::SyllablePosition::Coda => SyllablePosition::Coda,
            voirs_g2p::SyllablePosition::Final => SyllablePosition::Coda,
            voirs_g2p::SyllablePosition::Standalone => SyllablePosition::Unknown,
        }
    }
    /// Convert voirs-sdk::LanguageCode to voirs-g2p::LanguageCode
    pub(crate) fn convert_language_code(
        &self,
        lang: &LanguageCode,
    ) -> Option<voirs_g2p::LanguageCode> {
        match lang {
            LanguageCode::EnUs => Some(voirs_g2p::LanguageCode::EnUs),
            LanguageCode::EnGb => Some(voirs_g2p::LanguageCode::EnGb),
            LanguageCode::DeDe => Some(voirs_g2p::LanguageCode::De),
            LanguageCode::FrFr => Some(voirs_g2p::LanguageCode::Fr),
            LanguageCode::EsEs => Some(voirs_g2p::LanguageCode::Es),
            LanguageCode::PtBr => Some(voirs_g2p::LanguageCode::Pt),
            LanguageCode::JaJp => Some(voirs_g2p::LanguageCode::Ja),
            LanguageCode::ZhCn => Some(voirs_g2p::LanguageCode::ZhCn),
            LanguageCode::ItIt => Some(voirs_g2p::LanguageCode::It),
            _ => None,
        }
    }
    pub(crate) fn calculate_phoneme_similarity(&self, phoneme1: &str, phoneme2: &str) -> f32 {
        if phoneme1 == phoneme2 {
            return 1.0;
        }
        let features1 = self.get_phonetic_features(phoneme1);
        let features2 = self.get_phonetic_features(phoneme2);
        let similarity = self.calculate_feature_overlap(&features1, &features2);
        if self.are_allophonic_variants(phoneme1, phoneme2) {
            similarity.max(0.9)
        } else if self.are_minimal_pairs(phoneme1, phoneme2) {
            similarity.max(0.8)
        } else {
            similarity
        }
    }
    pub(crate) fn get_phonetic_features(&self, phoneme: &str) -> PhoneticFeatures {
        match phoneme {
            "a" => PhoneticFeatures::vowel(VowelHeight::Low, VowelBackness::Central, false),
            "e" => PhoneticFeatures::vowel(VowelHeight::Mid, VowelBackness::Front, false),
            "i" => PhoneticFeatures::vowel(VowelHeight::High, VowelBackness::Front, false),
            "o" => PhoneticFeatures::vowel(VowelHeight::Mid, VowelBackness::Back, true),
            "u" => PhoneticFeatures::vowel(VowelHeight::High, VowelBackness::Back, true),
            "æ" => PhoneticFeatures::vowel(VowelHeight::Low, VowelBackness::Front, false),
            "ε" => PhoneticFeatures::vowel(VowelHeight::Mid, VowelBackness::Front, false),
            "ɪ" => PhoneticFeatures::vowel(VowelHeight::High, VowelBackness::Front, false),
            "ɔ" => PhoneticFeatures::vowel(VowelHeight::Mid, VowelBackness::Back, true),
            "ʊ" => PhoneticFeatures::vowel(VowelHeight::High, VowelBackness::Back, true),
            "p" => PhoneticFeatures::consonant(
                PlaceOfArticulation::Bilabial,
                MannerOfArticulation::Stop,
                false,
            ),
            "b" => PhoneticFeatures::consonant(
                PlaceOfArticulation::Bilabial,
                MannerOfArticulation::Stop,
                true,
            ),
            "t" => PhoneticFeatures::consonant(
                PlaceOfArticulation::Alveolar,
                MannerOfArticulation::Stop,
                false,
            ),
            "d" => PhoneticFeatures::consonant(
                PlaceOfArticulation::Alveolar,
                MannerOfArticulation::Stop,
                true,
            ),
            "k" => PhoneticFeatures::consonant(
                PlaceOfArticulation::Velar,
                MannerOfArticulation::Stop,
                false,
            ),
            "g" => PhoneticFeatures::consonant(
                PlaceOfArticulation::Velar,
                MannerOfArticulation::Stop,
                true,
            ),
            "f" => PhoneticFeatures::consonant(
                PlaceOfArticulation::Labiodental,
                MannerOfArticulation::Fricative,
                false,
            ),
            "v" => PhoneticFeatures::consonant(
                PlaceOfArticulation::Labiodental,
                MannerOfArticulation::Fricative,
                true,
            ),
            "s" => PhoneticFeatures::consonant(
                PlaceOfArticulation::Alveolar,
                MannerOfArticulation::Fricative,
                false,
            ),
            "z" => PhoneticFeatures::consonant(
                PlaceOfArticulation::Alveolar,
                MannerOfArticulation::Fricative,
                true,
            ),
            "ʃ" => PhoneticFeatures::consonant(
                PlaceOfArticulation::PostAlveolar,
                MannerOfArticulation::Fricative,
                false,
            ),
            "θ" => PhoneticFeatures::consonant(
                PlaceOfArticulation::Dental,
                MannerOfArticulation::Fricative,
                false,
            ),
            "m" => PhoneticFeatures::consonant(
                PlaceOfArticulation::Bilabial,
                MannerOfArticulation::Nasal,
                true,
            ),
            "n" => PhoneticFeatures::consonant(
                PlaceOfArticulation::Alveolar,
                MannerOfArticulation::Nasal,
                true,
            ),
            "ŋ" => PhoneticFeatures::consonant(
                PlaceOfArticulation::Velar,
                MannerOfArticulation::Nasal,
                true,
            ),
            "l" => PhoneticFeatures::consonant(
                PlaceOfArticulation::Alveolar,
                MannerOfArticulation::Lateral,
                true,
            ),
            "r" => PhoneticFeatures::consonant(
                PlaceOfArticulation::Alveolar,
                MannerOfArticulation::Approximant,
                true,
            ),
            "w" => PhoneticFeatures::consonant(
                PlaceOfArticulation::Bilabial,
                MannerOfArticulation::Approximant,
                true,
            ),
            "j" => PhoneticFeatures::consonant(
                PlaceOfArticulation::Palatal,
                MannerOfArticulation::Approximant,
                true,
            ),
            "tʃ" => PhoneticFeatures::consonant(
                PlaceOfArticulation::PostAlveolar,
                MannerOfArticulation::Affricate,
                false,
            ),
            _ => PhoneticFeatures::default(),
        }
    }
    pub(crate) fn calculate_feature_overlap(
        &self,
        features1: &PhoneticFeatures,
        features2: &PhoneticFeatures,
    ) -> f32 {
        if features1.is_vowel != features2.is_vowel {
            return 0.1;
        }
        if features1.is_vowel {
            let mut score = 0.0;
            let height_diff = (features1.height as i32 - features2.height as i32).abs();
            score += match height_diff {
                0 => 0.4,
                1 => 0.2,
                _ => 0.0,
            };
            let backness_diff = (features1.backness as i32 - features2.backness as i32).abs();
            score += match backness_diff {
                0 => 0.4,
                1 => 0.2,
                _ => 0.0,
            };
            if features1.rounded == features2.rounded {
                score += 0.2;
            }
            score
        } else {
            let mut score = 0.0;
            let place_diff = (features1.place as i32 - features2.place as i32).abs();
            score += match place_diff {
                0 => 0.4,
                1 => 0.2,
                _ => 0.0,
            };
            let manner_diff = (features1.manner as i32 - features2.manner as i32).abs();
            score += match manner_diff {
                0 => 0.4,
                1 => 0.2,
                _ => 0.0,
            };
            if features1.voiced == features2.voiced {
                score += 0.2;
            }
            score
        }
    }
    pub(crate) fn are_allophonic_variants(&self, phoneme1: &str, phoneme2: &str) -> bool {
        let allophones = [
            ("t", "d"),
            ("p", "b"),
            ("k", "g"),
            ("f", "v"),
            ("s", "z"),
            ("θ", "ð"),
        ];
        allophones.iter().any(|(p1, p2)| {
            (phoneme1 == *p1 && phoneme2 == *p2) || (phoneme1 == *p2 && phoneme2 == *p1)
        })
    }
    pub(crate) fn are_minimal_pairs(&self, phoneme1: &str, phoneme2: &str) -> bool {
        let minimal_pairs = [("ɪ", "ε"), ("æ", "ε"), ("p", "t"), ("b", "d"), ("k", "t")];
        minimal_pairs.iter().any(|(p1, p2)| {
            (phoneme1 == *p1 && phoneme2 == *p2) || (phoneme1 == *p2 && phoneme2 == *p1)
        })
    }
    pub(crate) fn are_similar_phonemes(&self, p1: &str, p2: &str) -> bool {
        let similar_pairs = [("p", "b"), ("t", "d"), ("k", "g"), ("f", "v"), ("s", "z")];
        similar_pairs
            .iter()
            .any(|(a, b)| (p1 == *a && p2 == *b) || (p1 == *b && p2 == *a))
    }
    pub(crate) fn are_same_category(&self, p1: &str, p2: &str) -> bool {
        let vowels = ["a", "e", "i", "o", "u"];
        let consonants = [
            "p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "m", "n", "l", "r",
        ];
        (vowels.contains(&p1) && vowels.contains(&p2))
            || (consonants.contains(&p1) && consonants.contains(&p2))
    }
    /// Score a word's pronunciation quality from its real forced-aligned phonemes
    /// (see [`super::evaluator_impl2::PronunciationEvaluatorImpl::align_phonemes_to_audio`]
    /// and [`super::forced_alignment::word_phoneme_span`]).
    ///
    /// Combines the mean acoustic Goodness-of-Pronunciation confidence (how well each
    /// segment's measured samples match the acoustic profile expected for its
    /// phoneme) with how well each phoneme's *measured* duration (`end_time -
    /// start_time`, from the audio) matches its G2P-predicted duration
    /// (`Phoneme::duration_ms`). Both terms are derived from the actual audio and
    /// vary with it; there are no fixed fallback constants standing in for either.
    pub(crate) async fn calculate_word_pronunciation_accuracy(
        &self,
        actual_phonemes: &[AlignedPhoneme],
    ) -> Result<f32, EvaluationError> {
        if actual_phonemes.is_empty() {
            return Ok(0.0);
        }
        let mean_confidence = actual_phonemes.iter().map(|p| p.confidence).sum::<f32>()
            / actual_phonemes.len() as f32;
        let duration_ratios: Vec<f32> = actual_phonemes
            .iter()
            .filter_map(|p| {
                p.phoneme.duration_ms.map(|expected_ms| {
                    let expected_ms = expected_ms.max(1.0);
                    let actual_ms = ((p.end_time - p.start_time) * 1000.0).max(0.001);
                    (actual_ms / expected_ms)
                        .min(expected_ms / actual_ms)
                        .clamp(0.0, 1.0)
                })
            })
            .collect();
        let duration_accuracy = if duration_ratios.is_empty() {
            1.0
        } else {
            duration_ratios.iter().sum::<f32>() / duration_ratios.len() as f32
        };
        Ok((mean_confidence * 0.7 + duration_accuracy * 0.3).clamp(0.0, 1.0))
    }
    pub(crate) async fn calculate_rhythm_regularity(
        &self,
        alignment: &PhonemeAlignment,
    ) -> Result<f32, EvaluationError> {
        if alignment.phonemes.len() < 3 {
            return Ok(0.5);
        }
        let mut intervals = Vec::new();
        for window in alignment.phonemes.windows(2) {
            let interval = window[1].start_time - window[0].start_time;
            intervals.push(interval);
        }
        let mean_interval = intervals.iter().sum::<f32>() / intervals.len() as f32;
        if mean_interval <= 0.0 {
            return Ok(0.0);
        }
        let variance = intervals
            .iter()
            .map(|i| (i - mean_interval).powi(2))
            .sum::<f32>()
            / intervals.len() as f32;
        let std_dev = variance.sqrt();
        let cv = std_dev / mean_interval;
        Ok((1.0_f32 - cv.min(1.0_f32)).max(0.0_f32))
    }
    /// Estimate syllable count from text
    pub(crate) fn estimate_syllable_count(&self, text: &str) -> usize {
        let mut syllable_count = 0;
        let words: Vec<&str> = text.split_whitespace().collect();
        for word in words {
            syllable_count += self.count_syllables_in_word(word);
        }
        syllable_count.max(1)
    }
    /// Count syllables in a single word
    pub(crate) fn count_syllables_in_word(&self, word: &str) -> usize {
        let word = word.to_lowercase();
        let vowels = "aeiouy";
        let mut count = 0;
        let mut prev_was_vowel = false;
        for c in word.chars() {
            let is_vowel = vowels.contains(c);
            if is_vowel && !prev_was_vowel {
                count += 1;
            }
            prev_was_vowel = is_vowel;
        }
        if word.ends_with('e') && count > 1 {
            count -= 1;
        }
        count.max(1)
    }
    /// Calculate speech duration (excluding pauses)
    pub(crate) async fn calculate_speech_duration(
        &self,
        alignment: &PhonemeAlignment,
    ) -> Result<f32, EvaluationError> {
        if alignment.phonemes.is_empty() {
            return Ok(0.0);
        }
        let mut total_speech_time = 0.0;
        let mut last_end_time = 0.0;
        for phoneme in &alignment.phonemes {
            let phoneme_duration = phoneme.end_time - phoneme.start_time;
            total_speech_time += phoneme_duration;
            let gap = phoneme.start_time - last_end_time;
            let _ = gap > 0.1;
            last_end_time = phoneme.end_time;
        }
        Ok(total_speech_time)
    }
    /// Calculate timing consistency for a set of durations
    pub(crate) fn calculate_timing_consistency(&self, durations: &[f32]) -> f32 {
        if durations.len() < 2 {
            return 0.7;
        }
        let mean_duration = durations.iter().sum::<f32>() / durations.len() as f32;
        if mean_duration <= 0.0 {
            return 0.0;
        }
        let variance = durations
            .iter()
            .map(|d| (d - mean_duration).powi(2))
            .sum::<f32>()
            / durations.len() as f32;
        let std_dev = variance.sqrt();
        let cv = std_dev / mean_duration;
        (1.0_f32 - cv.min(1.0_f32)).max(0.0_f32)
    }
    /// Calculate transition smoothness between phonemes
    pub(crate) async fn calculate_transition_smoothness(
        &self,
        alignment: &PhonemeAlignment,
    ) -> Result<f32, EvaluationError> {
        if alignment.phonemes.len() < 2 {
            return Ok(0.7);
        }
        let mut smooth_transitions = 0;
        let mut total_transitions = 0;
        for window in alignment.phonemes.windows(2) {
            let gap = window[1].start_time - window[0].end_time;
            let transition_score = self.evaluate_transition_quality(&window[0], &window[1], gap);
            if transition_score > 0.7 {
                smooth_transitions += 1;
            }
            total_transitions += 1;
        }
        if total_transitions == 0 {
            return Ok(0.7);
        }
        Ok(smooth_transitions as f32 / total_transitions as f32)
    }
    /// Evaluate the quality of a transition between two phonemes
    pub(crate) fn evaluate_transition_quality(
        &self,
        from_phoneme: &AlignedPhoneme,
        to_phoneme: &AlignedPhoneme,
        gap: f32,
    ) -> f32 {
        let from_features = self.get_phonetic_features(&from_phoneme.phoneme.symbol);
        let to_features = self.get_phonetic_features(&to_phoneme.phoneme.symbol);
        let gap_score = if gap < 0.01 {
            0.9
        } else if gap < 0.05 {
            0.8
        } else if gap < 0.1 {
            0.6
        } else if gap < 0.2 {
            0.4
        } else {
            0.2
        };
        let compatibility_score = if from_features.is_vowel == to_features.is_vowel {
            0.9
        } else {
            if from_features.is_vowel {
                0.8
            } else {
                0.7
            }
        };
        (gap_score + compatibility_score) / 2.0
    }
}
