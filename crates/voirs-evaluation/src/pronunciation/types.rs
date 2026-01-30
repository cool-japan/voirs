//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum PlaceOfArticulation {
    Bilabial = 0,
    Labiodental = 1,
    Dental = 2,
    Alveolar = 3,
    PostAlveolar = 4,
    Palatal = 5,
    Velar = 6,
    Glottal = 7,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum MannerOfArticulation {
    Stop = 0,
    Fricative = 1,
    Affricate = 2,
    Nasal = 3,
    Lateral = 4,
    Approximant = 5,
}
/// Pause placement preferences
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PausePlacement {
    /// Between phrases
    Phrasal,
    /// Between sentences
    Sentential,
    /// Within phrases (rare)
    Intraphrasal,
    /// Breath groups
    Respiratory,
}
/// Language-specific rhythm norms
#[derive(Debug, Clone)]
pub struct RhythmNorms {
    /// Variability coefficient norms for syllables
    pub syllable_variability: (f32, f32),
    /// Variability coefficient norms for vowels
    pub vowel_variability: (f32, f32),
    /// Stress pattern regularity
    pub stress_regularity: f32,
    /// Typical rhythm class index
    pub rhythm_class_index: f32,
}
/// Prosodic features for adaptation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProsodicFeature {
    /// Fundamental frequency patterns
    F0Range,
    /// Speaking rate
    SpeakingRate,
    /// Stress patterns
    StressPattern,
    /// Rhythm timing
    RhythmTiming,
    /// Pause patterns
    PausePattern,
    /// Intonation contours
    IntonationContour,
}
/// Prosodic timing types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimingType {
    /// Stress-timed languages (English, German)
    StressTimed,
    /// Syllable-timed languages (Spanish, French, Italian)
    SyllableTimed,
    /// Mora-timed languages (Japanese)
    MoraTimed,
    /// Mixed timing
    Mixed,
}
/// Pronunciation evaluator implementation
pub struct PronunciationEvaluatorImpl {
    /// Configuration
    pub(super) config: PronunciationEvaluationConfig,
    /// Supported metrics
    pub(super) supported_metrics: Vec<PronunciationMetric>,
    /// Metadata
    pub(super) metadata: PronunciationEvaluatorMetadata,
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
    pub(super) async fn calculate_phoneme_accuracy(
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
            let accuracy = self.calculate_phoneme_similarity(
                &aligned_phoneme.phoneme.symbol,
                &expected_phoneme.symbol,
            );
            let duration_accuracy = if let Some(expected_duration) = expected_phoneme.duration_ms {
                let actual_duration =
                    (aligned_phoneme.end_time - aligned_phoneme.start_time) * 1000.0;
                let ratio = actual_duration / expected_duration;
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
    pub(super) async fn calculate_word_accuracy(
        &self,
        alignment: &PhonemeAlignment,
        expected_text: &str,
    ) -> Result<Vec<WordPronunciationScore>, EvaluationError> {
        let words: Vec<&str> = expected_text.split_whitespace().collect();
        let mut word_scores = Vec::new();
        for (i, word) in words.iter().enumerate() {
            let word_phonemes = self
                .get_word_phonemes_from_alignment(alignment, i, word)
                .await?;
            let accuracy = self
                .calculate_word_pronunciation_accuracy(word, &word_phonemes)
                .await?;
            word_scores.push(WordPronunciationScore {
                word: (*word).to_string(),
                accuracy,
                stress_accuracy: 0.85,
                syllable_accuracy: 0.90,
                phoneme_scores: vec![],
                position: i,
            });
        }
        Ok(word_scores)
    }
    /// Calculate comprehensive fluency score
    pub(super) async fn calculate_fluency(
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
    async fn calculate_speaking_rate(
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
    async fn analyze_pause_patterns(
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
    async fn assess_temporal_coordination(
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
    async fn detect_disfluencies(
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
    pub(super) async fn calculate_rhythm(
        &self,
        alignment: &PhonemeAlignment,
    ) -> Result<f32, EvaluationError> {
        self.calculate_rhythm_regularity(alignment).await
    }
    /// Calculate comprehensive stress accuracy
    pub(super) async fn calculate_stress_accuracy(
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
    pub(super) async fn calculate_intonation_accuracy(
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
    pub(super) async fn calculate_emotional_prosody(
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
    async fn calculate_cross_linguistic_prosody(
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
    async fn analyze_word_stress_patterns(
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
                .extract_actual_stress_pattern(alignment, word_idx, word)
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
    async fn analyze_syllable_stress_patterns(
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
    async fn analyze_stress_timing(
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
    async fn analyze_pitch_contour(
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
    async fn analyze_sentence_boundaries(
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
    async fn detect_emphasis_patterns(
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
    async fn detect_focus_patterns(
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
    pub(super) async fn generate_feedback(
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
                            phoneme_score.actual_phoneme.as_ref().unwrap(),
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
    async fn text_to_phonemes(&self, text: &str) -> Result<Vec<Phoneme>, EvaluationError> {
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
    fn convert_syllable_position(
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
    fn convert_language_code(&self, lang: &LanguageCode) -> Option<voirs_g2p::LanguageCode> {
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
    fn mock_phonemize(&self, word: &str) -> Vec<Phoneme> {
        let phoneme_map = self.create_phoneme_mapping();
        let mut phonemes = Vec::new();
        let chars: Vec<char> = word.to_lowercase().chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let mut found = false;
            if i < chars.len() - 1 {
                let bigram = format!("{}{}", chars[i], chars[i + 1]);
                if let Some(phoneme_sym) = phoneme_map.get(&bigram) {
                    phonemes.push(Phoneme {
                        symbol: phoneme_sym.clone(),
                        ipa_symbol: phoneme_sym.clone(),
                        stress: self.estimate_stress_level(word, i),
                        syllable_position: self.determine_syllable_position(word, i),
                        duration_ms: Some(self.estimate_phoneme_duration(phoneme_sym)),
                        confidence: 0.9,
                    });
                    i += 2;
                    found = true;
                }
            }
            if !found {
                let single_char = chars[i].to_string();
                let phoneme_sym = phoneme_map
                    .get(&single_char)
                    .unwrap_or(&single_char)
                    .clone();
                phonemes.push(Phoneme {
                    symbol: phoneme_sym.clone(),
                    ipa_symbol: phoneme_sym,
                    stress: self.estimate_stress_level(word, i),
                    syllable_position: self.determine_syllable_position(word, i),
                    duration_ms: Some(self.estimate_phoneme_duration(&chars[i].to_string())),
                    confidence: 0.8,
                });
                i += 1;
            }
        }
        phonemes
    }
    fn create_phoneme_mapping(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("a".to_string(), "æ".to_string());
        map.insert("e".to_string(), "ε".to_string());
        map.insert("i".to_string(), "ɪ".to_string());
        map.insert("o".to_string(), "ɔ".to_string());
        map.insert("u".to_string(), "ʊ".to_string());
        map.insert("ai".to_string(), "aɪ".to_string());
        map.insert("au".to_string(), "aʊ".to_string());
        map.insert("oi".to_string(), "ɔɪ".to_string());
        map.insert("ou".to_string(), "oʊ".to_string());
        map.insert("th".to_string(), "θ".to_string());
        map.insert("sh".to_string(), "ʃ".to_string());
        map.insert("ch".to_string(), "tʃ".to_string());
        map.insert("ng".to_string(), "ŋ".to_string());
        map.insert("ph".to_string(), "f".to_string());
        map.insert("p".to_string(), "p".to_string());
        map.insert("b".to_string(), "b".to_string());
        map.insert("t".to_string(), "t".to_string());
        map.insert("d".to_string(), "d".to_string());
        map.insert("k".to_string(), "k".to_string());
        map.insert("g".to_string(), "g".to_string());
        map.insert("f".to_string(), "f".to_string());
        map.insert("v".to_string(), "v".to_string());
        map.insert("s".to_string(), "s".to_string());
        map.insert("z".to_string(), "z".to_string());
        map.insert("m".to_string(), "m".to_string());
        map.insert("n".to_string(), "n".to_string());
        map.insert("l".to_string(), "l".to_string());
        map.insert("r".to_string(), "r".to_string());
        map.insert("w".to_string(), "w".to_string());
        map.insert("y".to_string(), "j".to_string());
        map.insert("h".to_string(), "h".to_string());
        map
    }
    fn estimate_stress_level(&self, word: &str, position: usize) -> u8 {
        let word_len = word.len();
        if word_len <= 3 {
            return 1;
        }
        if position < word_len / 3 {
            2
        } else {
            u8::from(position < 2 * word_len / 3)
        }
    }
    fn determine_syllable_position(
        &self,
        word: &str,
        position: usize,
    ) -> voirs_sdk::types::SyllablePosition {
        let word_len = word.len();
        let relative_pos = position as f32 / word_len as f32;
        if relative_pos < 0.33 {
            voirs_sdk::types::SyllablePosition::Onset
        } else if relative_pos < 0.67 {
            voirs_sdk::types::SyllablePosition::Nucleus
        } else {
            voirs_sdk::types::SyllablePosition::Coda
        }
    }
    fn estimate_phoneme_duration(&self, phoneme: &str) -> f32 {
        match phoneme {
            "æ" | "ε" | "ɪ" | "ɔ" | "ʊ" | "aɪ" | "aʊ" | "ɔɪ" | "oʊ" => 150.0,
            "f" | "v" | "s" | "z" | "ʃ" | "θ" => 120.0,
            "p" | "b" | "t" | "d" | "k" | "g" => 80.0,
            "m" | "n" | "ŋ" => 100.0,
            "l" | "r" => 90.0,
            "w" | "j" => 70.0,
            _ => 100.0,
        }
    }
    pub(super) fn calculate_phoneme_similarity(&self, phoneme1: &str, phoneme2: &str) -> f32 {
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
    fn get_phonetic_features(&self, phoneme: &str) -> PhoneticFeatures {
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
    fn calculate_feature_overlap(
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
    fn are_allophonic_variants(&self, phoneme1: &str, phoneme2: &str) -> bool {
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
    fn are_minimal_pairs(&self, phoneme1: &str, phoneme2: &str) -> bool {
        let minimal_pairs = [("ɪ", "ε"), ("æ", "ε"), ("p", "t"), ("b", "d"), ("k", "t")];
        minimal_pairs.iter().any(|(p1, p2)| {
            (phoneme1 == *p1 && phoneme2 == *p2) || (phoneme1 == *p2 && phoneme2 == *p1)
        })
    }
    fn are_similar_phonemes(&self, p1: &str, p2: &str) -> bool {
        let similar_pairs = [("p", "b"), ("t", "d"), ("k", "g"), ("f", "v"), ("s", "z")];
        similar_pairs
            .iter()
            .any(|(a, b)| (p1 == *a && p2 == *b) || (p1 == *b && p2 == *a))
    }
    fn are_same_category(&self, p1: &str, p2: &str) -> bool {
        let vowels = ["a", "e", "i", "o", "u"];
        let consonants = [
            "p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "m", "n", "l", "r",
        ];
        (vowels.contains(&p1) && vowels.contains(&p2))
            || (consonants.contains(&p1) && consonants.contains(&p2))
    }
    async fn get_word_phonemes_from_alignment(
        &self,
        _alignment: &PhonemeAlignment,
        _word_index: usize,
        _word: &str,
    ) -> Result<Vec<Phoneme>, EvaluationError> {
        Ok(vec![])
    }
    async fn calculate_word_pronunciation_accuracy(
        &self,
        word: &str,
        actual_phonemes: &[Phoneme],
    ) -> Result<f32, EvaluationError> {
        let expected_phonemes = self.mock_phonemize(word);
        if expected_phonemes.is_empty() || actual_phonemes.is_empty() {
            return Ok(0.0);
        }
        let alignment_matrix = self.compute_dtw_alignment(&expected_phonemes, actual_phonemes)?;
        let aligned_pairs =
            self.extract_optimal_path(&alignment_matrix, &expected_phonemes, actual_phonemes)?;
        let mut total_score = 0.0;
        let mut phoneme_count = 0;
        for (expected_opt, actual_opt) in aligned_pairs {
            phoneme_count += 1;
            match (expected_opt, actual_opt) {
                (Some(expected), Some(actual)) => {
                    let phoneme_similarity =
                        self.calculate_phoneme_similarity(&expected.symbol, &actual.symbol);
                    let duration_accuracy = if let Some(expected_dur) = expected.duration_ms {
                        let actual_dur = actual.duration_ms.unwrap_or(100.0);
                        let duration_ratio =
                            (actual_dur / expected_dur).min(expected_dur / actual_dur);
                        duration_ratio.max(0.3)
                    } else {
                        1.0
                    };
                    let stress_accuracy = if expected.stress == actual.stress {
                        1.0
                    } else if (i32::from(expected.stress) - i32::from(actual.stress)).abs() == 1 {
                        0.8
                    } else {
                        0.6
                    };
                    let combined_score =
                        phoneme_similarity * 0.6 + duration_accuracy * 0.2 + stress_accuracy * 0.2;
                    total_score += combined_score;
                }
                (Some(_), None) => {
                    total_score += 0.0;
                }
                (None, Some(_)) => {
                    total_score += 0.3;
                }
                (None, None) => {
                    continue;
                }
            }
        }
        let base_accuracy = if phoneme_count > 0 {
            total_score / phoneme_count as f32
        } else {
            0.0
        };
        let length_penalty = self.calculate_length_penalty(&expected_phonemes, actual_phonemes);
        let syllable_structure_bonus = self
            .calculate_syllable_structure_accuracy(word, actual_phonemes)
            .await?;
        let final_accuracy = (base_accuracy + syllable_structure_bonus) * length_penalty;
        Ok(final_accuracy.max(0.0).min(1.0))
    }
    fn compute_dtw_alignment(
        &self,
        expected: &[Phoneme],
        actual: &[Phoneme],
    ) -> Result<Vec<Vec<f32>>, EvaluationError> {
        let rows = expected.len() + 1;
        let cols = actual.len() + 1;
        let mut dtw_matrix = vec![vec![f32::INFINITY; cols]; rows];
        dtw_matrix[0][0] = 0.0;
        for i in 1..rows {
            dtw_matrix[i][0] = dtw_matrix[i - 1][0] + 1.0;
        }
        for j in 1..cols {
            dtw_matrix[0][j] = dtw_matrix[0][j - 1] + 1.0;
        }
        for i in 1..rows {
            for j in 1..cols {
                let similarity = self
                    .calculate_phoneme_similarity(&expected[i - 1].symbol, &actual[j - 1].symbol);
                let substitution_cost = 1.0 - similarity;
                dtw_matrix[i][j] = substitution_cost
                    + [
                        dtw_matrix[i - 1][j] + 1.0,
                        dtw_matrix[i][j - 1] + 1.0,
                        dtw_matrix[i - 1][j - 1],
                    ]
                    .iter()
                    .fold(f32::INFINITY, |a, &b| a.min(b));
            }
        }
        Ok(dtw_matrix)
    }
    fn extract_optimal_path<'a>(
        &self,
        matrix: &[Vec<f32>],
        expected: &'a [Phoneme],
        actual: &'a [Phoneme],
    ) -> Result<Vec<(Option<&'a Phoneme>, Option<&'a Phoneme>)>, EvaluationError> {
        let mut path = Vec::new();
        let mut i = expected.len();
        let mut j = actual.len();
        while i > 0 || j > 0 {
            if i == 0 {
                path.push((None, Some(&actual[j - 1])));
                j -= 1;
            } else if j == 0 {
                path.push((Some(&expected[i - 1]), None));
                i -= 1;
            } else {
                let diag = matrix[i - 1][j - 1];
                let up = matrix[i - 1][j];
                let left = matrix[i][j - 1];
                if diag <= up && diag <= left {
                    path.push((Some(&expected[i - 1]), Some(&actual[j - 1])));
                    i -= 1;
                    j -= 1;
                } else if up <= left {
                    path.push((Some(&expected[i - 1]), None));
                    i -= 1;
                } else {
                    path.push((None, Some(&actual[j - 1])));
                    j -= 1;
                }
            }
        }
        path.reverse();
        Ok(path)
    }
    fn calculate_length_penalty(&self, expected: &[Phoneme], actual: &[Phoneme]) -> f32 {
        if expected.is_empty() {
            return if actual.is_empty() { 1.0 } else { 0.5 };
        }
        let length_ratio = actual.len() as f32 / expected.len() as f32;
        if !(0.5..=2.0).contains(&length_ratio) {
            0.7
        } else if !(0.8..=1.25).contains(&length_ratio) {
            0.9
        } else {
            1.0
        }
    }
    async fn calculate_syllable_structure_accuracy(
        &self,
        word: &str,
        _actual_phonemes: &[Phoneme],
    ) -> Result<f32, EvaluationError> {
        let vowel_count = word.chars().filter(|c| "aeiou".contains(*c)).count();
        let expected_syllables = vowel_count.max(1);
        if expected_syllables <= 3 {
            Ok(0.1)
        } else {
            Ok(0.05)
        }
    }
    async fn calculate_rhythm_regularity(
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
    fn estimate_syllable_count(&self, text: &str) -> usize {
        let mut syllable_count = 0;
        let words: Vec<&str> = text.split_whitespace().collect();
        for word in words {
            syllable_count += self.count_syllables_in_word(word);
        }
        syllable_count.max(1)
    }
    /// Count syllables in a single word
    fn count_syllables_in_word(&self, word: &str) -> usize {
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
    async fn calculate_speech_duration(
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
    fn calculate_timing_consistency(&self, durations: &[f32]) -> f32 {
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
    async fn calculate_transition_smoothness(
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
    fn evaluate_transition_quality(
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
impl PronunciationEvaluatorImpl {
    /// Create a mock alignment for testing purposes
    pub(super) async fn create_mock_alignment(
        &self,
        audio: &AudioBuffer,
        text: &str,
    ) -> Result<PhonemeAlignment, EvaluationError> {
        let phonemes = self.text_to_phonemes(text).await?;
        let total_duration = audio.samples().len() as f32 / audio.sample_rate() as f32;
        let mut aligned_phonemes = Vec::new();
        let phoneme_duration = total_duration / phonemes.len() as f32;
        for (i, phoneme) in phonemes.into_iter().enumerate() {
            let start_time = i as f32 * phoneme_duration;
            let end_time = start_time + phoneme_duration;
            aligned_phonemes.push(AlignedPhoneme {
                phoneme,
                start_time,
                end_time,
                confidence: 0.9,
            });
        }
        Ok(PhonemeAlignment {
            phonemes: aligned_phonemes,
            total_duration,
            alignment_confidence: 0.9,
            word_alignments: Vec::new(),
        })
    }
    fn get_expected_stress_pattern(&self, word: &str) -> Vec<u8> {
        let syllable_count = word
            .chars()
            .filter(|c| "aeiouAEIOU".contains(*c))
            .count()
            .max(1);
        let mut pattern = vec![0; syllable_count];
        if !pattern.is_empty() {
            pattern[0] = 2;
        }
        pattern
    }
    async fn extract_actual_stress_pattern(
        &self,
        alignment: &PhonemeAlignment,
        _word_idx: usize,
        word: &str,
    ) -> Result<Vec<u8>, EvaluationError> {
        let syllable_count = word
            .chars()
            .filter(|c| "aeiouAEIOU".contains(*c))
            .count()
            .max(1);
        let mut pattern = vec![0; syllable_count];
        if !alignment.phonemes.is_empty() {
            let total_duration: f32 = alignment
                .phonemes
                .iter()
                .map(|p| p.end_time - p.start_time)
                .sum();
            let avg_duration = total_duration / alignment.phonemes.len() as f32;
            for (i, phoneme) in alignment.phonemes.iter().enumerate() {
                let duration = phoneme.end_time - phoneme.start_time;
                let syllable_idx = (i * syllable_count) / alignment.phonemes.len();
                if syllable_idx < pattern.len() && duration > avg_duration * 1.2 {
                    pattern[syllable_idx] = 1;
                }
            }
        }
        Ok(pattern)
    }
    fn compare_stress_patterns(&self, expected: &[u8], actual: &[u8]) -> f32 {
        if expected.is_empty() && actual.is_empty() {
            return 1.0;
        }
        if expected.is_empty() || actual.is_empty() {
            return 0.0;
        }
        let min_len = expected.len().min(actual.len());
        let mut matches = 0;
        for i in 0..min_len {
            if expected[i] == actual[i] {
                matches += 1;
            } else if (i32::from(expected[i]) - i32::from(actual[i])).abs() == 1 {
                matches += 1;
            }
        }
        matches as f32 / expected.len().max(actual.len()) as f32
    }
    fn find_sentence_boundaries(&self, text: &str) -> Vec<usize> {
        let mut boundaries = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        for (i, &ch) in chars.iter().enumerate() {
            if ch == '.' || ch == '!' || ch == '?' {
                boundaries.push(i);
            }
        }
        if boundaries.is_empty() || *boundaries.last().unwrap() != chars.len() - 1 {
            boundaries.push(chars.len() - 1);
        }
        boundaries
    }
    async fn evaluate_boundary_intonation(
        &self,
        alignment: &PhonemeAlignment,
        boundary_pos: usize,
    ) -> Result<f32, EvaluationError> {
        if alignment.phonemes.is_empty() || boundary_pos >= alignment.phonemes.len() {
            return Ok(0.5);
        }
        let boundary_phoneme = &alignment.phonemes[boundary_pos.min(alignment.phonemes.len() - 1)];
        let duration = boundary_phoneme.end_time - boundary_phoneme.start_time;
        let score = if duration > 0.15 { 0.8 } else { 0.5 };
        Ok(score)
    }
    fn calculate_emphasis_likelihood(
        &self,
        duration: f32,
        stress_level: u8,
        _alignment: &PhonemeAlignment,
    ) -> f32 {
        let duration_factor = (duration / 0.1).min(2.0) * 0.3;
        let stress_factor = f32::from(stress_level) / 2.0 * 0.7;
        (duration_factor + stress_factor).min(1.0)
    }
    fn analyze_emphasis_distribution(&self, emphasis_scores: &[f32]) -> f32 {
        if emphasis_scores.is_empty() {
            return 0.5;
        }
        let avg_emphasis = emphasis_scores.iter().sum::<f32>() / emphasis_scores.len() as f32;
        let variance = emphasis_scores
            .iter()
            .map(|&score| (score - avg_emphasis).powi(2))
            .sum::<f32>()
            / emphasis_scores.len() as f32;
        let optimal_variance = 0.1;
        let variance_score = 1.0 - (variance - optimal_variance).abs() / optimal_variance;
        variance_score.max(0.0).min(1.0)
    }
    fn identify_content_words(&self, text: &str) -> Vec<String> {
        let function_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        ];
        text.split_whitespace()
            .filter(|word| !function_words.contains(&word.to_lowercase().as_str()))
            .map(std::string::ToString::to_string)
            .collect()
    }
    fn identify_function_words(&self, text: &str) -> Vec<String> {
        let function_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        ];
        text.split_whitespace()
            .filter(|word| function_words.contains(&word.to_lowercase().as_str()))
            .map(std::string::ToString::to_string)
            .collect()
    }
    async fn analyze_content_word_prominence(
        &self,
        alignment: &PhonemeAlignment,
        content_words: &[String],
    ) -> Result<f32, EvaluationError> {
        if content_words.is_empty() || alignment.phonemes.is_empty() {
            return Ok(0.5);
        }
        let total_duration: f32 = alignment
            .phonemes
            .iter()
            .map(|p| p.end_time - p.start_time)
            .sum();
        let avg_duration = total_duration / alignment.phonemes.len() as f32;
        let prominent_phonemes = alignment
            .phonemes
            .iter()
            .filter(|p| p.end_time - p.start_time > avg_duration * 1.1)
            .count();
        let prominence_ratio = prominent_phonemes as f32 / content_words.len() as f32;
        Ok(prominence_ratio.min(1.0))
    }
    async fn analyze_function_word_deemphasis(
        &self,
        alignment: &PhonemeAlignment,
        function_words: &[String],
    ) -> Result<f32, EvaluationError> {
        if function_words.is_empty() || alignment.phonemes.is_empty() {
            return Ok(1.0);
        }
        let total_duration: f32 = alignment
            .phonemes
            .iter()
            .map(|p| p.end_time - p.start_time)
            .sum();
        let avg_duration = total_duration / alignment.phonemes.len() as f32;
        let deemphasized_phonemes = alignment
            .phonemes
            .iter()
            .filter(|p| p.end_time - p.start_time < avg_duration * 0.9)
            .count();
        let deemphasis_ratio = deemphasized_phonemes as f32 / function_words.len() as f32;
        Ok(deemphasis_ratio.min(1.0))
    }
    /// Extract prosodic features for emotional analysis
    pub(super) async fn extract_emotional_prosodic_features(
        &self,
        alignment: &PhonemeAlignment,
    ) -> Result<EmotionalProsodicFeatures, EvaluationError> {
        if alignment.phonemes.is_empty() {
            return Ok(EmotionalProsodicFeatures::default());
        }
        let mean_f0 = self.estimate_mean_f0(alignment);
        let f0_std = self.estimate_f0_std(alignment);
        let f0_range = self.estimate_f0_range(alignment);
        let speaking_rate = self.calculate_speaking_rate_for_emotion(alignment).await?;
        let (mean_energy, energy_std) = self.calculate_energy_features(alignment);
        let (pause_frequency, pause_duration_mean) = self.calculate_pause_features(alignment);
        let jitter = self.estimate_jitter(alignment);
        let shimmer = self.estimate_shimmer(alignment);
        let rhythm_regularity = self.calculate_rhythm_regularity(alignment).await?;
        let stress_pattern_strength = self.calculate_stress_pattern_strength(alignment);
        Ok(EmotionalProsodicFeatures {
            mean_f0,
            f0_std,
            f0_range,
            speaking_rate,
            mean_energy,
            energy_std,
            pause_frequency,
            pause_duration_mean,
            jitter,
            shimmer,
            rhythm_regularity,
            stress_pattern_strength,
        })
    }
    /// Detect emotional state from prosodic features
    pub(super) fn detect_emotional_state(
        &self,
        features: &EmotionalProsodicFeatures,
    ) -> Result<EmotionalState, EvaluationError> {
        let high_f0 = features.mean_f0 > 180.0;
        let high_f0_variation = features.f0_std > 30.0;
        let fast_speech = features.speaking_rate > 6.0;
        let slow_speech = features.speaking_rate < 4.0;
        let high_energy = features.mean_energy > 0.7;
        let low_energy = features.mean_energy < 0.3;
        let frequent_pauses = features.pause_frequency > 1.0;
        let long_pauses = features.pause_duration_mean > 0.5;
        let irregular_rhythm = features.rhythm_regularity < 0.5;
        let emotion = match (
            high_f0,
            high_f0_variation,
            fast_speech,
            slow_speech,
            high_energy,
            low_energy,
        ) {
            (true, _, true, _, true, _) if irregular_rhythm => EmotionalState::Angry,
            (true, true, true, _, true, _) => EmotionalState::Excited,
            (true, _, _, _, true, _) => EmotionalState::Happy,
            (false, _, _, true, _, true) if long_pauses => EmotionalState::Sad,
            (_, true, _, _, _, _) if frequent_pauses && irregular_rhythm => EmotionalState::Anxious,
            (_, false, _, false, _, false) if features.rhythm_regularity > 0.7 => {
                EmotionalState::Confident
            }
            (_, true, _, _, _, _) if frequent_pauses => EmotionalState::Uncertain,
            _ => EmotionalState::Neutral,
        };
        Ok(emotion)
    }
    /// Calculate emotional accuracy between expected and detected emotions
    pub(super) fn calculate_emotional_accuracy(
        &self,
        expected: &EmotionalState,
        detected: &EmotionalState,
    ) -> f32 {
        if expected == detected {
            return 1.0;
        }
        match (expected, detected) {
            (EmotionalState::Happy, EmotionalState::Excited)
            | (EmotionalState::Excited, EmotionalState::Happy) => 0.8,
            (EmotionalState::Sad, EmotionalState::Disappointed)
            | (EmotionalState::Disappointed, EmotionalState::Sad) => 0.8,
            (EmotionalState::Angry, EmotionalState::Frustrated)
            | (EmotionalState::Frustrated, EmotionalState::Angry) => 0.8,
            (EmotionalState::Anxious, EmotionalState::Uncertain)
            | (EmotionalState::Uncertain, EmotionalState::Anxious) => 0.7,
            (EmotionalState::Happy, EmotionalState::Confident)
            | (EmotionalState::Confident, EmotionalState::Happy) => 0.6,
            (EmotionalState::Sad, EmotionalState::Anxious)
            | (EmotionalState::Anxious, EmotionalState::Sad) => 0.5,
            (EmotionalState::Happy, EmotionalState::Sad)
            | (EmotionalState::Sad, EmotionalState::Happy)
            | (EmotionalState::Excited, EmotionalState::Disappointed)
            | (EmotionalState::Disappointed, EmotionalState::Excited) => 0.1,
            (EmotionalState::Neutral, _) | (_, EmotionalState::Neutral) => 0.5,
            _ => 0.3,
        }
    }
    /// Calculate emotional intensity
    fn calculate_emotional_intensity(&self, features: &EmotionalProsodicFeatures) -> f32 {
        let f0_intensity = (features.f0_std / 50.0).min(1.0);
        let energy_intensity = features.mean_energy;
        let rate_intensity = ((features.speaking_rate - 5.0).abs() / 3.0).min(1.0);
        let rhythm_intensity = 1.0 - features.rhythm_regularity;
        (f0_intensity + energy_intensity + rate_intensity + rhythm_intensity) / 4.0
    }
    /// Calculate emotional consistency
    fn calculate_emotional_consistency(&self, features: &EmotionalProsodicFeatures) -> f32 {
        let f0_consistency = 1.0 - (features.f0_std / 100.0).min(1.0);
        let energy_consistency = 1.0 - features.energy_std;
        let rhythm_consistency = features.rhythm_regularity;
        (f0_consistency + energy_consistency + rhythm_consistency) / 3.0
    }
    /// Analyze emotional dynamics over time
    pub(super) async fn analyze_emotional_dynamics(
        &self,
        alignment: &PhonemeAlignment,
        features: &EmotionalProsodicFeatures,
    ) -> Result<EmotionalDynamics, EvaluationError> {
        let mut emotion_trajectory = Vec::new();
        let mut emotion_transitions = Vec::new();
        let window_size = 1.0;
        let num_windows = (alignment.total_duration / window_size).ceil() as usize;
        let mut last_emotion = EmotionalState::Neutral;
        for i in 0..num_windows {
            let start_time = i as f32 * window_size;
            let end_time = ((i + 1) as f32 * window_size).min(alignment.total_duration);
            let window_features = self.extract_window_features(alignment, start_time, end_time)?;
            let window_emotion = self.detect_emotional_state(&window_features)?;
            let confidence = self.calculate_emotion_detection_confidence(&window_features);
            emotion_trajectory.push((start_time, window_emotion, confidence));
            if i > 0 && window_emotion != last_emotion {
                let transition_start = (i - 1) as f32 * window_size;
                let smoothness = self.calculate_transition_smoothness_emotion(
                    alignment,
                    transition_start,
                    start_time,
                );
                emotion_transitions.push(EmotionalTransition {
                    start_time: transition_start,
                    end_time: start_time,
                    from_emotion: last_emotion,
                    to_emotion: window_emotion,
                    smoothness,
                });
            }
            last_emotion = window_emotion;
        }
        let emotional_stability = self.calculate_emotional_stability(&emotion_trajectory);
        let peak_intensity = features.mean_energy.max((features.f0_std / 50.0).min(1.0));
        Ok(EmotionalDynamics {
            emotion_trajectory,
            emotional_stability,
            peak_intensity,
            emotion_transitions,
        })
    }
    /// Calculate confidence in emotion detection
    fn calculate_emotion_detection_confidence(&self, features: &EmotionalProsodicFeatures) -> f32 {
        let f0_confidence = (features.f0_std / 30.0).min(1.0);
        let energy_confidence = features.mean_energy;
        let rhythm_confidence = 1.0 - features.rhythm_regularity;
        ((f0_confidence + energy_confidence + rhythm_confidence) / 3.0).max(0.3)
    }
    fn calculate_duration_contrast(
        &self,
        stressed_durations: &[f32],
        unstressed_durations: &[f32],
    ) -> f32 {
        if stressed_durations.is_empty() || unstressed_durations.is_empty() {
            return 0.5;
        }
        let stressed_avg = stressed_durations.iter().sum::<f32>() / stressed_durations.len() as f32;
        let unstressed_avg =
            unstressed_durations.iter().sum::<f32>() / unstressed_durations.len() as f32;
        if unstressed_avg == 0.0 {
            return 0.5;
        }
        let ratio = stressed_avg / unstressed_avg;
        if (1.2..=2.0).contains(&ratio) {
            1.0
        } else if (1.1..=2.5).contains(&ratio) {
            0.8
        } else if (1.0..=3.0).contains(&ratio) {
            0.6
        } else {
            0.3
        }
    }
    /// Estimate mean F0 from alignment duration patterns
    fn estimate_mean_f0(&self, alignment: &PhonemeAlignment) -> f32 {
        let mut f0_estimates = Vec::new();
        for phoneme in &alignment.phonemes {
            let duration = phoneme.end_time - phoneme.start_time;
            let features = self.get_phonetic_features(&phoneme.phoneme.symbol);
            let base_f0 = if features.is_vowel {
                150.0 + f32::from(phoneme.phoneme.stress) * 20.0
            } else {
                140.0 + duration * 100.0
            };
            f0_estimates.push(base_f0);
        }
        if f0_estimates.is_empty() {
            150.0
        } else {
            f0_estimates.iter().sum::<f32>() / f0_estimates.len() as f32
        }
    }
    /// Estimate F0 standard deviation
    fn estimate_f0_std(&self, alignment: &PhonemeAlignment) -> f32 {
        let mean_f0 = self.estimate_mean_f0(alignment);
        let mut f0_estimates = Vec::new();
        for phoneme in &alignment.phonemes {
            let duration = phoneme.end_time - phoneme.start_time;
            let features = self.get_phonetic_features(&phoneme.phoneme.symbol);
            let base_f0 = if features.is_vowel {
                150.0 + f32::from(phoneme.phoneme.stress) * 20.0
            } else {
                140.0 + duration * 100.0
            };
            f0_estimates.push(base_f0);
        }
        if f0_estimates.len() < 2 {
            return 20.0;
        }
        let variance = f0_estimates
            .iter()
            .map(|f0| (f0 - mean_f0).powi(2))
            .sum::<f32>()
            / f0_estimates.len() as f32;
        variance.sqrt()
    }
    /// Estimate F0 range (max - min)
    fn estimate_f0_range(&self, alignment: &PhonemeAlignment) -> f32 {
        let mut f0_estimates = Vec::new();
        for phoneme in &alignment.phonemes {
            let duration = phoneme.end_time - phoneme.start_time;
            let features = self.get_phonetic_features(&phoneme.phoneme.symbol);
            let base_f0 = if features.is_vowel {
                150.0 + f32::from(phoneme.phoneme.stress) * 20.0
            } else {
                140.0 + duration * 100.0
            };
            f0_estimates.push(base_f0);
        }
        if f0_estimates.is_empty() {
            return 60.0;
        }
        let max_f0 = f0_estimates
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_f0 = f0_estimates.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        max_f0 - min_f0
    }
    /// Calculate speaking rate for emotion analysis
    async fn calculate_speaking_rate_for_emotion(
        &self,
        alignment: &PhonemeAlignment,
    ) -> Result<f32, EvaluationError> {
        if alignment.phonemes.is_empty() || alignment.total_duration <= 0.0 {
            return Ok(5.0);
        }
        let syllable_count = alignment
            .phonemes
            .iter()
            .filter(|p| {
                let features = self.get_phonetic_features(&p.phoneme.symbol);
                features.is_vowel
            })
            .count();
        let speech_duration = self.calculate_speech_duration(alignment).await?;
        if speech_duration <= 0.0 {
            Ok(5.0)
        } else {
            Ok(syllable_count as f32 / speech_duration)
        }
    }
    /// Calculate energy features
    fn calculate_energy_features(&self, alignment: &PhonemeAlignment) -> (f32, f32) {
        let mut energy_values = Vec::new();
        for phoneme in &alignment.phonemes {
            let duration = phoneme.end_time - phoneme.start_time;
            let stress_factor = f32::from(phoneme.phoneme.stress) / 2.0;
            let energy = (duration * 2.0 + stress_factor * 0.3).min(1.0);
            energy_values.push(energy);
        }
        if energy_values.is_empty() {
            return (0.5, 0.1);
        }
        let mean_energy = energy_values.iter().sum::<f32>() / energy_values.len() as f32;
        let variance = energy_values
            .iter()
            .map(|e| (e - mean_energy).powi(2))
            .sum::<f32>()
            / energy_values.len() as f32;
        let energy_std = variance.sqrt();
        (mean_energy, energy_std)
    }
    /// Calculate pause features
    fn calculate_pause_features(&self, alignment: &PhonemeAlignment) -> (f32, f32) {
        let mut pauses = Vec::new();
        for window in alignment.phonemes.windows(2) {
            let gap = window[1].start_time - window[0].end_time;
            if gap > 0.05 {
                pauses.push(gap);
            }
        }
        if pauses.is_empty() {
            return (0.0, 0.0);
        }
        let pause_frequency = pauses.len() as f32 / alignment.total_duration;
        let pause_duration_mean = pauses.iter().sum::<f32>() / pauses.len() as f32;
        (pause_frequency, pause_duration_mean)
    }
    /// Estimate jitter (pitch period variation)
    fn estimate_jitter(&self, alignment: &PhonemeAlignment) -> f32 {
        let mut timing_variations = Vec::new();
        for window in alignment.phonemes.windows(2) {
            let dur1 = window[0].end_time - window[0].start_time;
            let dur2 = window[1].end_time - window[1].start_time;
            if dur1 > 0.0 && dur2 > 0.0 {
                let variation = (dur2 - dur1).abs() / dur1;
                timing_variations.push(variation);
            }
        }
        if timing_variations.is_empty() {
            0.01
        } else {
            (timing_variations.iter().sum::<f32>() / timing_variations.len() as f32).min(0.1)
        }
    }
    /// Estimate shimmer (amplitude variation)
    fn estimate_shimmer(&self, alignment: &PhonemeAlignment) -> f32 {
        let mut amplitude_variations = Vec::new();
        for window in alignment.phonemes.windows(2) {
            let stress1 = f32::from(window[0].phoneme.stress);
            let stress2 = f32::from(window[1].phoneme.stress);
            let variation = (stress2 - stress1).abs() / (stress1 + 1.0);
            amplitude_variations.push(variation);
        }
        if amplitude_variations.is_empty() {
            0.05
        } else {
            (amplitude_variations.iter().sum::<f32>() / amplitude_variations.len() as f32).min(0.2)
        }
    }
    /// Calculate stress pattern strength
    fn calculate_stress_pattern_strength(&self, alignment: &PhonemeAlignment) -> f32 {
        if alignment.phonemes.is_empty() {
            return 0.5;
        }
        let mut stress_contrast = 0.0;
        let mut stress_pairs = 0;
        for window in alignment.phonemes.windows(2) {
            let stress_diff =
                (i32::from(window[1].phoneme.stress) - i32::from(window[0].phoneme.stress)).abs();
            stress_contrast += stress_diff as f32;
            stress_pairs += 1;
        }
        if stress_pairs == 0 {
            0.5
        } else {
            (stress_contrast / stress_pairs as f32 / 2.0).min(1.0)
        }
    }
    /// Extract features for a time window
    fn extract_window_features(
        &self,
        alignment: &PhonemeAlignment,
        start_time: f32,
        end_time: f32,
    ) -> Result<EmotionalProsodicFeatures, EvaluationError> {
        let window_phonemes: Vec<_> = alignment
            .phonemes
            .iter()
            .filter(|p| p.start_time >= start_time && p.end_time <= end_time)
            .cloned()
            .collect();
        if window_phonemes.is_empty() {
            return Ok(EmotionalProsodicFeatures::default());
        }
        let window_alignment = PhonemeAlignment {
            phonemes: window_phonemes,
            total_duration: end_time - start_time,
            alignment_confidence: alignment.alignment_confidence,
            word_alignments: vec![],
        };
        let mean_f0 = self.estimate_mean_f0(&window_alignment);
        let f0_std = self.estimate_f0_std(&window_alignment);
        let f0_range = self.estimate_f0_range(&window_alignment);
        let (mean_energy, energy_std) = self.calculate_energy_features(&window_alignment);
        let (pause_frequency, pause_duration_mean) =
            self.calculate_pause_features(&window_alignment);
        let jitter = self.estimate_jitter(&window_alignment);
        let shimmer = self.estimate_shimmer(&window_alignment);
        let stress_pattern_strength = self.calculate_stress_pattern_strength(&window_alignment);
        let rhythm_regularity = self.calculate_timing_consistency(
            &window_alignment
                .phonemes
                .iter()
                .map(|p| p.end_time - p.start_time)
                .collect::<Vec<_>>(),
        );
        let syllable_count = window_alignment
            .phonemes
            .iter()
            .filter(|p| {
                let features = self.get_phonetic_features(&p.phoneme.symbol);
                features.is_vowel
            })
            .count();
        let speaking_rate = if window_alignment.total_duration > 0.0 {
            syllable_count as f32 / window_alignment.total_duration
        } else {
            5.0
        };
        Ok(EmotionalProsodicFeatures {
            mean_f0,
            f0_std,
            f0_range,
            speaking_rate,
            mean_energy,
            energy_std,
            pause_frequency,
            pause_duration_mean,
            jitter,
            shimmer,
            rhythm_regularity,
            stress_pattern_strength,
        })
    }
    /// Calculate transition smoothness for emotions
    fn calculate_transition_smoothness_emotion(
        &self,
        alignment: &PhonemeAlignment,
        start_time: f32,
        end_time: f32,
    ) -> f32 {
        let transition_phonemes: Vec<_> = alignment
            .phonemes
            .iter()
            .filter(|p| p.start_time >= start_time && p.end_time <= end_time)
            .collect();
        if transition_phonemes.len() < 2 {
            return 0.5;
        }
        let mut smoothness_scores = Vec::new();
        for window in transition_phonemes.windows(2) {
            let dur1 = window[0].end_time - window[0].start_time;
            let dur2 = window[1].end_time - window[1].start_time;
            let stress1 = f32::from(window[0].phoneme.stress);
            let stress2 = f32::from(window[1].phoneme.stress);
            let duration_smoothness = 1.0 - ((dur2 - dur1).abs() / dur1.max(0.01)).min(1.0);
            let stress_smoothness = 1.0 - ((stress2 - stress1).abs() / 2.0).min(1.0);
            smoothness_scores.push((duration_smoothness + stress_smoothness) / 2.0);
        }
        if smoothness_scores.is_empty() {
            0.5
        } else {
            smoothness_scores.iter().sum::<f32>() / smoothness_scores.len() as f32
        }
    }
    /// Calculate emotional stability from trajectory
    fn calculate_emotional_stability(&self, trajectory: &[(f32, EmotionalState, f32)]) -> f32 {
        if trajectory.len() < 2 {
            return 1.0;
        }
        let mut changes = 0;
        let mut last_emotion = trajectory[0].1;
        for &(_, emotion, _) in trajectory.iter().skip(1) {
            if emotion != last_emotion {
                changes += 1;
            }
            last_emotion = emotion;
        }
        let change_rate = changes as f32 / trajectory.len() as f32;
        (1.0 - change_rate).max(0.0)
    }
    /// Get language-specific prosodic profile
    fn get_language_prosodic_profile(&self, language: LanguageCode) -> LanguageProsodicProfile {
        match language {
            LanguageCode::EnUs => LanguageProsodicProfile {
                language,
                f0_range: (80.0, 300.0),
                speaking_rate_range: (3.5, 5.5),
                timing_preference: TimingType::StressTimed,
                intonation_patterns: vec![IntonationPattern {
                    name: "falling".to_string(),
                    contour: vec![(0.0, 1.0), (1.0, 0.7)],
                    frequency: 0.6,
                    function: IntonationFunction::Statement,
                }],
                pause_characteristics: PauseCharacteristics {
                    typical_frequency: 0.3,
                    duration_distribution: vec![(0.2, 0.4), (0.5, 0.4), (1.0, 0.2)],
                    placement_preferences: vec![
                        PausePlacement::Phrasal,
                        PausePlacement::Sentential,
                    ],
                },
                rhythm_norms: RhythmNorms {
                    syllable_variability: (0.5, 0.2),
                    vowel_variability: (0.4, 0.15),
                    stress_regularity: 0.7,
                    rhythm_class_index: 0.6,
                },
            },
            LanguageCode::JaJp => LanguageProsodicProfile {
                language,
                f0_range: (100.0, 250.0),
                speaking_rate_range: (3.0, 4.5),
                timing_preference: TimingType::MoraTimed,
                intonation_patterns: vec![IntonationPattern {
                    name: "high-low".to_string(),
                    contour: vec![(0.0, 1.0), (1.0, 0.5)],
                    frequency: 0.7,
                    function: IntonationFunction::Statement,
                }],
                pause_characteristics: PauseCharacteristics {
                    typical_frequency: 0.4,
                    duration_distribution: vec![(0.3, 0.5), (0.6, 0.3), (1.2, 0.2)],
                    placement_preferences: vec![
                        PausePlacement::Phrasal,
                        PausePlacement::Respiratory,
                    ],
                },
                rhythm_norms: RhythmNorms {
                    syllable_variability: (0.3, 0.1),
                    vowel_variability: (0.25, 0.1),
                    stress_regularity: 0.9,
                    rhythm_class_index: 0.3,
                },
            },
            LanguageCode::DeDe => LanguageProsodicProfile {
                language,
                f0_range: (85.0, 280.0),
                speaking_rate_range: (3.8, 4.8),
                timing_preference: TimingType::StressTimed,
                intonation_patterns: vec![IntonationPattern {
                    name: "falling".to_string(),
                    contour: vec![(0.0, 1.0), (1.0, 0.6)],
                    frequency: 0.5,
                    function: IntonationFunction::Statement,
                }],
                pause_characteristics: PauseCharacteristics {
                    typical_frequency: 0.25,
                    duration_distribution: vec![(0.2, 0.3), (0.4, 0.5), (0.8, 0.2)],
                    placement_preferences: vec![
                        PausePlacement::Phrasal,
                        PausePlacement::Sentential,
                    ],
                },
                rhythm_norms: RhythmNorms {
                    syllable_variability: (0.6, 0.25),
                    vowel_variability: (0.45, 0.2),
                    stress_regularity: 0.65,
                    rhythm_class_index: 0.65,
                },
            },
            _ => LanguageProsodicProfile {
                language,
                f0_range: (90.0, 280.0),
                speaking_rate_range: (3.5, 4.5),
                timing_preference: TimingType::Mixed,
                intonation_patterns: vec![IntonationPattern {
                    name: "neutral".to_string(),
                    contour: vec![(0.0, 1.0), (1.0, 0.8)],
                    frequency: 0.8,
                    function: IntonationFunction::Statement,
                }],
                pause_characteristics: PauseCharacteristics {
                    typical_frequency: 0.3,
                    duration_distribution: vec![(0.3, 0.6), (0.6, 0.3), (1.0, 0.1)],
                    placement_preferences: vec![PausePlacement::Phrasal],
                },
                rhythm_norms: RhythmNorms {
                    syllable_variability: (0.5, 0.2),
                    vowel_variability: (0.4, 0.15),
                    stress_regularity: 0.5,
                    rhythm_class_index: 0.5,
                },
            },
        }
    }
    /// Calculate distance between two language prosodic profiles
    fn calculate_language_distance(
        &self,
        source: &LanguageProsodicProfile,
        target: &LanguageProsodicProfile,
    ) -> f32 {
        let f0_distance = {
            let source_mid = (source.f0_range.0 + source.f0_range.1) / 2.0;
            let target_mid = (target.f0_range.0 + target.f0_range.1) / 2.0;
            (source_mid - target_mid).abs() / 200.0
        };
        let rate_distance = {
            let source_mid = (source.speaking_rate_range.0 + source.speaking_rate_range.1) / 2.0;
            let target_mid = (target.speaking_rate_range.0 + target.speaking_rate_range.1) / 2.0;
            (source_mid - target_mid).abs() / 5.0
        };
        let timing_distance = if source.timing_preference == target.timing_preference {
            0.0
        } else {
            0.5
        };
        let pause_distance = (source.pause_characteristics.typical_frequency
            - target.pause_characteristics.typical_frequency)
            .abs();
        (f0_distance * 0.3 + rate_distance * 0.3 + timing_distance * 0.2 + pause_distance * 0.2)
            .min(1.0)
    }
    /// Compare prosodic features between languages
    fn compare_prosodic_features(
        &self,
        features: &EmotionalProsodicFeatures,
        source_profile: &LanguageProsodicProfile,
        target_profile: &LanguageProsodicProfile,
    ) -> LanguageComparisonDetails {
        let f0_similarity = {
            let in_source_range = features.mean_f0 >= source_profile.f0_range.0
                && features.mean_f0 <= source_profile.f0_range.1;
            let in_target_range = features.mean_f0 >= target_profile.f0_range.0
                && features.mean_f0 <= target_profile.f0_range.1;
            match (in_source_range, in_target_range) {
                (true, true) => 1.0,
                (true, false) => 0.7,
                (false, true) => 0.3,
                (false, false) => 0.1,
            }
        };
        let rhythm_similarity = {
            let source_rate_mid =
                (source_profile.speaking_rate_range.0 + source_profile.speaking_rate_range.1) / 2.0;
            let target_rate_mid =
                (target_profile.speaking_rate_range.0 + target_profile.speaking_rate_range.1) / 2.0;
            let source_diff = (features.speaking_rate - source_rate_mid).abs();
            let target_diff = (features.speaking_rate - target_rate_mid).abs();
            if source_diff < target_diff {
                0.8 - (source_diff / 5.0).min(0.8)
            } else {
                0.2 + (0.8 - (target_diff / 5.0).min(0.8))
            }
        };
        let stress_similarity = features.stress_pattern_strength;
        let timing_similarity = features.rhythm_regularity;
        let intonation_similarity = 1.0 - (features.f0_std / 50.0).min(1.0);
        let pause_similarity = 1.0
            - (features.pause_frequency - source_profile.pause_characteristics.typical_frequency)
                .abs()
                .min(1.0);
        LanguageComparisonDetails {
            f0_similarity,
            rhythm_similarity,
            stress_similarity,
            timing_similarity,
            intonation_similarity,
            pause_similarity,
        }
    }
    /// Calculate prosodic transfer score
    fn calculate_prosodic_transfer_score(&self, comparison: &LanguageComparisonDetails) -> f32 {
        let weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10];
        let scores = [
            comparison.f0_similarity,
            comparison.rhythm_similarity,
            comparison.stress_similarity,
            comparison.timing_similarity,
            comparison.intonation_similarity,
            comparison.pause_similarity,
        ];
        scores
            .iter()
            .zip(weights.iter())
            .map(|(score, weight)| score * weight)
            .sum()
    }
    /// Generate adaptation recommendations
    fn generate_adaptation_recommendations(
        &self,
        features: &EmotionalProsodicFeatures,
        _source_profile: &LanguageProsodicProfile,
        target_profile: &LanguageProsodicProfile,
    ) -> Vec<ProsodyAdaptation> {
        let mut recommendations = Vec::new();
        let target_f0_mid = (target_profile.f0_range.0 + target_profile.f0_range.1) / 2.0;
        if (features.mean_f0 - target_f0_mid).abs() > 30.0 {
            recommendations.push(ProsodyAdaptation {
                feature: ProsodicFeature::F0Range,
                current_value: features.mean_f0,
                target_value: target_f0_mid,
                importance: 0.8,
                recommendation: format!(
                    "Adjust pitch range: current {} Hz, target {} Hz",
                    features.mean_f0 as i32, target_f0_mid as i32
                ),
            });
        }
        let target_rate_mid =
            (target_profile.speaking_rate_range.0 + target_profile.speaking_rate_range.1) / 2.0;
        if (features.speaking_rate - target_rate_mid).abs() > 1.0 {
            recommendations.push(ProsodyAdaptation {
                feature: ProsodicFeature::SpeakingRate,
                current_value: features.speaking_rate,
                target_value: target_rate_mid,
                importance: 0.7,
                recommendation: format!(
                    "Adjust speaking rate: current {:.1} syll/s, target {:.1} syll/s",
                    features.speaking_rate, target_rate_mid
                ),
            });
        }
        if (features.pause_frequency - target_profile.pause_characteristics.typical_frequency).abs()
            > 0.2
        {
            recommendations.push(ProsodyAdaptation {
                feature: ProsodicFeature::PausePattern,
                current_value: features.pause_frequency,
                target_value: target_profile.pause_characteristics.typical_frequency,
                importance: 0.6,
                recommendation: format!(
                    "Adjust pause frequency: current {:.2}, target {:.2}",
                    features.pause_frequency,
                    target_profile.pause_characteristics.typical_frequency
                ),
            });
        }
        recommendations
    }
    /// Calculate cross-linguistic intelligibility
    fn calculate_cross_linguistic_intelligibility(
        &self,
        comparison: &LanguageComparisonDetails,
        language_distance: f32,
    ) -> f32 {
        let prosodic_intelligibility = comparison.f0_similarity * 0.3
            + comparison.rhythm_similarity * 0.25
            + comparison.stress_similarity * 0.2
            + comparison.timing_similarity * 0.15
            + comparison.intonation_similarity * 0.1;
        let distance_penalty = language_distance * 0.3;
        (prosodic_intelligibility - distance_penalty)
            .max(0.0)
            .min(1.0)
    }
}
/// Functions of intonation patterns
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntonationFunction {
    /// Declarative statements
    Statement,
    /// Yes/no questions
    YesNoQuestion,
    /// Wh-questions
    WhQuestion,
    /// Commands/imperatives
    Command,
    /// Emotional expression
    Emotional,
    /// Focus/emphasis
    Focus,
    /// Continuation/listing
    Continuation,
}
/// Complete emotional prosody analysis result
#[derive(Debug, Clone)]
pub struct EmotionalProsodyScore {
    /// Detected primary emotional state
    pub detected_emotion: EmotionalState,
    /// How well the detected emotion matches expected emotion
    pub emotional_appropriateness: f32,
    /// Intensity of emotional expression (0.0 = flat, 1.0 = very expressive)
    pub emotional_intensity: f32,
    /// Consistency of emotional expression throughout
    pub emotional_consistency: f32,
    /// Emotional dynamics analysis
    pub emotional_dynamics: EmotionalDynamics,
    /// Underlying prosodic features
    pub prosodic_features: EmotionalProsodicFeatures,
    /// Confidence in emotion detection
    pub confidence: f32,
}
/// Language-specific pause characteristics
#[derive(Debug, Clone)]
pub struct PauseCharacteristics {
    /// Typical pause frequency (pauses per second)
    pub typical_frequency: f32,
    /// Typical pause duration distribution
    pub duration_distribution: Vec<(f32, f32)>,
    /// Pause placement preferences
    pub placement_preferences: Vec<PausePlacement>,
}
/// Language-specific prosodic parameters for cross-linguistic comparison
#[derive(Debug, Clone)]
pub struct LanguageProsodicProfile {
    /// Language code
    pub language: LanguageCode,
    /// Typical F0 range for this language
    pub f0_range: (f32, f32),
    /// Typical speaking rate for this language
    pub speaking_rate_range: (f32, f32),
    /// Stress timing vs syllable timing preference
    pub timing_preference: TimingType,
    /// Intonation pattern characteristics
    pub intonation_patterns: Vec<IntonationPattern>,
    /// Typical pause patterns
    pub pause_characteristics: PauseCharacteristics,
    /// Rhythm metrics norms
    pub rhythm_norms: RhythmNorms,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum VowelBackness {
    Front = 0,
    Central = 1,
    Back = 2,
}
/// Cross-linguistic prosody comparison result
#[derive(Debug, Clone)]
pub struct CrossLinguisticProsodyScore {
    /// Source language
    pub source_language: LanguageCode,
    /// Target language for comparison
    pub target_language: LanguageCode,
    /// Prosodic transfer score (how well source patterns fit target)
    pub transfer_score: f32,
    /// Language distance metric
    pub language_distance: f32,
    /// Specific comparison results
    pub comparison_details: LanguageComparisonDetails,
    /// Adaptation recommendations
    pub adaptation_recommendations: Vec<ProsodyAdaptation>,
    /// Overall cross-linguistic intelligibility
    pub cross_linguistic_intelligibility: f32,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum VowelHeight {
    High = 0,
    Mid = 1,
    Low = 2,
}
/// Emotional transition between states
#[derive(Debug, Clone)]
pub struct EmotionalTransition {
    /// Start time of transition
    pub start_time: f32,
    /// End time of transition
    pub end_time: f32,
    /// From emotional state
    pub from_emotion: EmotionalState,
    /// To emotional state
    pub to_emotion: EmotionalState,
    /// Transition smoothness (0.0 = abrupt, 1.0 = smooth)
    pub smoothness: f32,
}
/// Enhanced word-level assessment with stress and syllables
#[derive(Debug, Clone)]
pub struct EnhancedWordScore {
    /// Base word pronunciation score
    pub base_score: WordPronunciationScore,
    /// Lexical stress accuracy
    pub lexical_stress_accuracy: f32,
    /// Syllable boundary accuracy
    pub syllable_boundary_accuracy: f32,
    /// Syllable structure correctness
    pub syllable_structure_score: f32,
    /// Individual syllable scores
    pub syllable_scores: Vec<SyllableAccuracyScore>,
}
/// Prosody adaptation recommendation
#[derive(Debug, Clone)]
pub struct ProsodyAdaptation {
    /// Prosodic feature to adapt
    pub feature: ProsodicFeature,
    /// Current value/pattern
    pub current_value: f32,
    /// Target value/pattern for better cross-linguistic performance
    pub target_value: f32,
    /// Importance of this adaptation
    pub importance: f32,
    /// Specific recommendation text
    pub recommendation: String,
}
/// Emotional dynamics over time
#[derive(Debug, Clone)]
pub struct EmotionalDynamics {
    /// Emotional trajectory over time
    pub emotion_trajectory: Vec<(f32, EmotionalState, f32)>,
    /// Emotional stability (low values indicate rapid changes)
    pub emotional_stability: f32,
    /// Peak emotional intensity
    pub peak_intensity: f32,
    /// Emotional transitions
    pub emotion_transitions: Vec<EmotionalTransition>,
}
/// Syllable-level accuracy score
#[derive(Debug, Clone)]
pub struct SyllableAccuracyScore {
    /// Expected syllable
    pub expected_syllable: SyllableInfo,
    /// Actual syllable (if detected)
    pub actual_syllable: Option<SyllableInfo>,
    /// Onset accuracy
    pub onset_accuracy: f32,
    /// Nucleus accuracy
    pub nucleus_accuracy: f32,
    /// Coda accuracy
    pub coda_accuracy: f32,
    /// Stress accuracy
    pub stress_accuracy: f32,
    /// Overall syllable score
    pub overall_accuracy: f32,
}
/// Phonetic features for similarity analysis
#[derive(Debug, Clone, PartialEq)]
pub(super) struct PhoneticFeatures {
    is_vowel: bool,
    height: VowelHeight,
    backness: VowelBackness,
    rounded: bool,
    place: PlaceOfArticulation,
    manner: MannerOfArticulation,
    voiced: bool,
}
impl PhoneticFeatures {
    fn vowel(height: VowelHeight, backness: VowelBackness, rounded: bool) -> Self {
        Self {
            is_vowel: true,
            height,
            backness,
            rounded,
            place: PlaceOfArticulation::Glottal,
            manner: MannerOfArticulation::Approximant,
            voiced: true,
        }
    }
    pub(super) fn consonant(
        place: PlaceOfArticulation,
        manner: MannerOfArticulation,
        voiced: bool,
    ) -> Self {
        Self {
            is_vowel: false,
            height: VowelHeight::Mid,
            backness: VowelBackness::Central,
            rounded: false,
            place,
            manner,
            voiced,
        }
    }
}
/// Detailed comparison between language prosodic patterns
#[derive(Debug, Clone)]
pub struct LanguageComparisonDetails {
    /// F0 pattern similarity
    pub f0_similarity: f32,
    /// Rhythm pattern similarity
    pub rhythm_similarity: f32,
    /// Stress pattern similarity
    pub stress_similarity: f32,
    /// Timing pattern similarity
    pub timing_similarity: f32,
    /// Intonation pattern similarity
    pub intonation_similarity: f32,
    /// Pause pattern similarity
    pub pause_similarity: f32,
}
/// Single pronunciation of a word
#[derive(Debug, Clone)]
pub struct WordPronunciation {
    /// Phoneme sequence
    pub phonemes: Vec<String>,
    /// Syllable breakdown
    pub syllables: Vec<SyllableInfo>,
    /// Frequency/likelihood of this pronunciation
    pub frequency: f32,
    /// Language-specific notes
    pub notes: Option<String>,
}
/// Emotional state classification for prosody analysis
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmotionalState {
    /// Neutral emotional state
    Neutral,
    /// Positive emotions
    Happy,
    /// High-energy positive emotion
    Excited,
    /// Unexpected positive reaction
    Surprised,
    /// Negative emotions
    Sad,
    /// High-energy negative emotion
    Angry,
    /// Persistent negative emotion
    Frustrated,
    /// Low-energy negative emotion
    Disappointed,
    /// Complex emotions
    Anxious,
    /// Self-assured emotional state
    Confident,
    /// Hesitant or doubtful emotional state
    Uncertain,
    /// Deliberate emphasis or stress
    Emphatic,
}
/// Enhanced syllable information
#[derive(Debug, Clone)]
pub struct SyllableInfo {
    /// Syllable text
    pub text: String,
    /// Phonemes in this syllable
    pub phonemes: Vec<String>,
    /// Stress level (0=unstressed, 1=secondary, 2=primary)
    pub stress_level: u8,
    /// Onset phonemes
    pub onset: Vec<String>,
    /// Nucleus (vowel core)
    pub nucleus: Vec<String>,
    /// Coda phonemes
    pub coda: Vec<String>,
    /// Start time in alignment
    pub start_time: f32,
    /// End time in alignment
    pub end_time: f32,
    /// Confidence score
    pub confidence: f32,
}
/// Pronunciation dictionary entry for multiple pronunciations
#[derive(Debug, Clone)]
pub struct PronunciationEntry {
    /// Word text
    pub word: String,
    /// Multiple possible pronunciations
    pub pronunciations: Vec<WordPronunciation>,
}
/// Language-specific intonation patterns
#[derive(Debug, Clone)]
pub struct IntonationPattern {
    /// Pattern name
    pub name: String,
    /// Typical F0 contour points (time, `relative_f0`)
    pub contour: Vec<(f32, f32)>,
    /// Usage frequency in the language
    pub frequency: f32,
    /// Semantic/pragmatic function
    pub function: IntonationFunction,
}
/// Prosodic features for emotional analysis
#[derive(Debug, Clone)]
pub struct EmotionalProsodicFeatures {
    /// Average pitch (F0) in Hz
    pub mean_f0: f32,
    /// Pitch standard deviation
    pub f0_std: f32,
    /// Pitch range (max - min)
    pub f0_range: f32,
    /// Speaking rate (syllables per second)
    pub speaking_rate: f32,
    /// Energy/intensity measures
    pub mean_energy: f32,
    /// Standard deviation of energy values
    pub energy_std: f32,
    /// Timing features
    pub pause_frequency: f32,
    /// Average duration of pauses
    pub pause_duration_mean: f32,
    /// Voice quality indicators
    pub jitter: f32,
    /// Amplitude variation between periods
    pub shimmer: f32,
    /// Rhythmic features
    pub rhythm_regularity: f32,
    /// Strength of stress pattern contrasts
    pub stress_pattern_strength: f32,
}
