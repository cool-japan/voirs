//! Extended pronunciation evaluator implementation (forced alignment, emotional prosody, cross-linguistic)

use crate::traits::{
    PronunciationEvaluationConfig, PronunciationEvaluatorMetadata, PronunciationMetric,
};
use crate::EvaluationError;
use scirs2_core::parallel_ops::*;
use voirs_recognizer::traits::{AlignedPhoneme, PhonemeAlignment, WordAlignment};
use voirs_sdk::{AudioBuffer, LanguageCode, Phoneme};

use super::evaluator_core::PronunciationEvaluatorImpl;
use super::extended_types::{
    EmotionalDynamics, EmotionalProsodicFeatures, EmotionalState, EmotionalTransition,
    IntonationFunction, IntonationPattern, LanguageComparisonDetails, LanguageProsodicProfile,
    MannerOfArticulation, PauseCharacteristics, PausePlacement, ProsodicFeature, ProsodyAdaptation,
    RhythmNorms, TimingType,
};
use super::forced_alignment;

impl PronunciationEvaluatorImpl {
    /// Force-align the reference `text`'s phoneme sequence onto `audio`.
    ///
    /// This is a genuine forced alignment: as in standard forced-alignment systems
    /// (e.g. Montreal Forced Aligner / HTK), the *identity* of each phoneme is fixed
    /// by the reference text (via [`PronunciationEvaluatorImpl::text_to_phonemes`]) —
    /// only the *timing* is determined from the audio. Timing is found with a
    /// dynamic-programming search over a real acoustic boundary-evidence function
    /// (frame energy + spectral flux; see [`forced_alignment::segment_via_dp`])
    /// instead of dividing the duration evenly. Each resulting segment is additionally
    /// scored against the acoustic profile expected for its phoneme (voicing, energy
    /// level, spectral noisiness) to produce a real, audio-dependent confidence — a
    /// Goodness-of-Pronunciation (GOP) style proxy consumed by
    /// [`PronunciationEvaluatorImpl::calculate_phoneme_accuracy`]. Because the
    /// aligner places the *reference* phonemes onto the audio, per-phoneme
    /// pronunciation-correctness evidence necessarily comes from this acoustic
    /// confidence, not from comparing phoneme symbols (which are identical to the
    /// reference by construction).
    ///
    /// When `audio` has no samples (or an invalid sample rate) every phoneme is
    /// reported with `confidence: 0.0` and zero duration rather than a fabricated
    /// uniform placement, since there is no acoustic evidence to place it from.
    pub(crate) async fn align_phonemes_to_audio(
        &self,
        audio: &AudioBuffer,
        text: &str,
    ) -> Result<PhonemeAlignment, EvaluationError> {
        let phonemes = self.text_to_phonemes(text).await?;
        let samples = audio.samples();
        let sample_rate = audio.sample_rate();
        let total_duration = if sample_rate > 0 {
            samples.len() as f32 / sample_rate as f32
        } else {
            0.0
        };

        if phonemes.is_empty() {
            return Ok(PhonemeAlignment {
                phonemes: Vec::new(),
                total_duration,
                alignment_confidence: 0.0,
                word_alignments: Vec::new(),
            });
        }

        if samples.is_empty() || sample_rate == 0 {
            let aligned_phonemes: Vec<AlignedPhoneme> = phonemes
                .into_iter()
                .map(|phoneme| AlignedPhoneme {
                    phoneme,
                    start_time: 0.0,
                    end_time: 0.0,
                    confidence: 0.0,
                })
                .collect();
            let word_alignments = self.build_word_alignments(text, &aligned_phonemes).await?;
            return Ok(PhonemeAlignment {
                phonemes: aligned_phonemes,
                total_duration: 0.0,
                alignment_confidence: 0.0,
                word_alignments,
            });
        }

        let grid = forced_alignment::extract_frame_grid(samples, sample_rate);
        // Require at least 2 frames per segment where possible, so each segment's
        // acoustic features (RMS/ZCR/voicing) are estimated from more than a single
        // analysis window; relax to 1 when the audio is too short for that.
        let min_len = if grid.frames.len() >= phonemes.len() * 2 {
            2
        } else {
            1
        };
        let dp_boundaries = if grid.frames.len() >= phonemes.len() && grid.frames.len() >= 2 {
            let novelty = forced_alignment::compute_novelty(&grid.frames);
            let hi = novelty.iter().cloned().fold(f64::MIN, f64::max);
            let lo = novelty.iter().cloned().fold(f64::MAX, f64::min);
            // A flat novelty function (e.g. silence or a constant signal) carries no
            // boundary evidence; fall back to uniform division rather than let the DP
            // arbitrarily collapse boundaries at the first tied index.
            if hi - lo > 1e-9 {
                forced_alignment::segment_via_dp(&novelty, phonemes.len(), min_len)
            } else {
                None
            }
        } else {
            None
        };

        let sample_boundaries: Vec<usize> = match dp_boundaries {
            Some(frame_boundaries) => frame_boundaries
                .iter()
                .map(|&b| {
                    if b >= grid.frames.len() {
                        samples.len()
                    } else {
                        (b * grid.hop_len).min(samples.len())
                    }
                })
                .collect(),
            None => forced_alignment::uniform_sample_boundaries(samples.len(), phonemes.len()),
        };

        let global_peak_rms =
            forced_alignment::peak_rms(samples, forced_alignment::frame_len_for(sample_rate));

        let mut aligned_phonemes = Vec::with_capacity(phonemes.len());
        for (i, phoneme) in phonemes.into_iter().enumerate() {
            let start = sample_boundaries[i].min(samples.len());
            let end = sample_boundaries[i + 1].clamp(start, samples.len());
            let segment = &samples[start..end];
            let confidence =
                self.acoustic_phoneme_confidence(segment, sample_rate, &phoneme, global_peak_rms);
            aligned_phonemes.push(AlignedPhoneme {
                phoneme,
                start_time: start as f32 / sample_rate as f32,
                end_time: end as f32 / sample_rate as f32,
                confidence,
            });
        }

        let alignment_confidence = if aligned_phonemes.is_empty() {
            0.0
        } else {
            aligned_phonemes.iter().map(|p| p.confidence).sum::<f32>()
                / aligned_phonemes.len() as f32
        };

        let word_alignments = self.build_word_alignments(text, &aligned_phonemes).await?;

        Ok(PhonemeAlignment {
            phonemes: aligned_phonemes,
            total_duration,
            alignment_confidence,
            word_alignments,
        })
    }

    /// Score how well an audio segment matches the acoustic profile expected for
    /// `phoneme`: a real, audio-dependent Goodness-of-Pronunciation (GOP) style proxy
    /// combining voicing (normalized-autocorrelation strength vs. the phoneme's
    /// expected voicing), energy level (segment RMS relative to the utterance peak vs.
    /// the phoneme's expected sonority tier) and spectral noisiness (zero-crossing
    /// rate vs. the phoneme's expected tier). Every input is measured directly from
    /// `segment`'s samples, so the score genuinely varies with the audio content; the
    /// phoneme-class -> expected-tier mapping is a documented acoustic-phonetic
    /// heuristic (obstruents are lower-energy and spectrally noisier than sonorants),
    /// not a trained classifier.
    fn acoustic_phoneme_confidence(
        &self,
        segment: &[f32],
        sample_rate: u32,
        phoneme: &Phoneme,
        global_peak_rms: f32,
    ) -> f32 {
        if segment.len() < 2 {
            return 0.0;
        }
        let features = self.get_phonetic_features(&phoneme.symbol);

        let measured_voicing = forced_alignment::voicing_strength(segment, sample_rate);
        let measured_energy = forced_alignment::rms(segment) / global_peak_rms.max(1e-6);
        let measured_zcr = forced_alignment::zero_crossing_rate(segment);

        let expected_voicing = if features.voiced { 1.0 } else { 0.0 };
        let expected_energy = if features.is_vowel {
            0.8
        } else {
            match features.manner {
                MannerOfArticulation::Approximant => 0.65,
                MannerOfArticulation::Nasal | MannerOfArticulation::Lateral => 0.6,
                MannerOfArticulation::Fricative | MannerOfArticulation::Affricate => 0.45,
                MannerOfArticulation::Stop => 0.25,
            }
        };
        let expected_zcr = if features.is_vowel {
            0.15
        } else {
            match (features.voiced, features.manner) {
                (
                    true,
                    MannerOfArticulation::Nasal
                    | MannerOfArticulation::Lateral
                    | MannerOfArticulation::Approximant,
                ) => 0.15,
                (true, MannerOfArticulation::Fricative | MannerOfArticulation::Affricate) => 0.45,
                (true, MannerOfArticulation::Stop) => 0.3,
                (false, _) => 0.75,
            }
        };

        let voicing_match = 1.0 - (measured_voicing - expected_voicing).abs();
        let energy_match = 1.0 - (measured_energy.min(1.5) - expected_energy).abs().min(1.0);
        let zcr_match = 1.0 - (measured_zcr - expected_zcr).abs();

        (voicing_match * 0.4 + energy_match * 0.35 + zcr_match * 0.25).clamp(0.0, 1.0)
    }

    /// Build word-level alignment spans by phonemizing each whitespace-delimited word
    /// individually and consuming that many phonemes sequentially from
    /// `aligned_phonemes`. This mirrors how the rule-based G2P backends used by
    /// [`PronunciationEvaluatorImpl::text_to_phonemes`] phonemize the full text (a
    /// concatenation of per-word phonemizations), so the counts line up in the common
    /// case. If the aligned sequence runs out early the remaining words are simply
    /// omitted rather than padded with fabricated entries.
    pub(crate) async fn build_word_alignments(
        &self,
        text: &str,
        aligned_phonemes: &[AlignedPhoneme],
    ) -> Result<Vec<WordAlignment>, EvaluationError> {
        let mut word_alignments = Vec::new();
        let mut cursor = 0usize;
        for word in text.split_whitespace() {
            if cursor >= aligned_phonemes.len() {
                break;
            }
            let word_len = self.text_to_phonemes(word).await?.len().max(1);
            let end = (cursor + word_len).min(aligned_phonemes.len());
            let span = &aligned_phonemes[cursor..end];
            if span.is_empty() {
                cursor = end;
                continue;
            }
            let start_time = span.first().map_or(0.0, |p| p.start_time);
            let end_time = span.last().map_or(start_time, |p| p.end_time);
            let confidence = span.iter().map(|p| p.confidence).sum::<f32>() / span.len() as f32;
            word_alignments.push(WordAlignment {
                word: word.to_string(),
                start_time,
                end_time,
                phonemes: span.to_vec(),
                confidence,
            });
            cursor = end;
        }
        Ok(word_alignments)
    }
    pub(crate) fn get_expected_stress_pattern(&self, word: &str) -> Vec<u8> {
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
    pub(crate) async fn extract_actual_stress_pattern(
        &self,
        alignment: &PhonemeAlignment,
        word_idx: usize,
        word: &str,
        total_words: usize,
    ) -> Result<Vec<u8>, EvaluationError> {
        let syllable_count = word
            .chars()
            .filter(|c| "aeiouAEIOU".contains(*c))
            .count()
            .max(1);
        let mut pattern = vec![0; syllable_count];
        // Scope the analysis to just this word's own aligned phonemes (real timing
        // from `align_phonemes_to_audio`), not the whole utterance, so that each word
        // in a multi-word sentence gets its own stress evidence instead of all words
        // sharing one whole-sentence duration profile.
        let word_phonemes =
            forced_alignment::word_phoneme_span(alignment, word_idx, word, total_words);
        if !word_phonemes.is_empty() {
            let total_duration: f32 = word_phonemes
                .iter()
                .map(|p| p.end_time - p.start_time)
                .sum();
            let avg_duration = total_duration / word_phonemes.len() as f32;
            for (i, phoneme) in word_phonemes.iter().enumerate() {
                let duration = phoneme.end_time - phoneme.start_time;
                let syllable_idx = (i * syllable_count) / word_phonemes.len();
                if syllable_idx < pattern.len() && duration > avg_duration * 1.2 {
                    pattern[syllable_idx] = 1;
                }
            }
        }
        Ok(pattern)
    }
    pub(crate) fn compare_stress_patterns(&self, expected: &[u8], actual: &[u8]) -> f32 {
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
    pub(crate) fn find_sentence_boundaries(&self, text: &str) -> Vec<usize> {
        let mut boundaries = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        for (i, &ch) in chars.iter().enumerate() {
            if ch == '.' || ch == '!' || ch == '?' {
                boundaries.push(i);
            }
        }
        if boundaries.is_empty()
            || *boundaries.last().expect("collection should not be empty") != chars.len() - 1
        {
            boundaries.push(chars.len() - 1);
        }
        boundaries
    }
    pub(crate) async fn evaluate_boundary_intonation(
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
    pub(crate) fn calculate_emphasis_likelihood(
        &self,
        duration: f32,
        stress_level: u8,
        _alignment: &PhonemeAlignment,
    ) -> f32 {
        let duration_factor = (duration / 0.1).min(2.0) * 0.3;
        let stress_factor = f32::from(stress_level) / 2.0 * 0.7;
        (duration_factor + stress_factor).min(1.0)
    }
    pub(crate) fn analyze_emphasis_distribution(&self, emphasis_scores: &[f32]) -> f32 {
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
    pub(crate) fn identify_content_words(&self, text: &str) -> Vec<String> {
        let function_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        ];
        text.split_whitespace()
            .filter(|word| !function_words.contains(&word.to_lowercase().as_str()))
            .map(std::string::ToString::to_string)
            .collect()
    }
    pub(crate) fn identify_function_words(&self, text: &str) -> Vec<String> {
        let function_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        ];
        text.split_whitespace()
            .filter(|word| function_words.contains(&word.to_lowercase().as_str()))
            .map(std::string::ToString::to_string)
            .collect()
    }
    pub(crate) async fn analyze_content_word_prominence(
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
    pub(crate) async fn analyze_function_word_deemphasis(
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
    pub(crate) async fn extract_emotional_prosodic_features(
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
    pub(crate) fn detect_emotional_state(
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
    pub(crate) fn calculate_emotional_accuracy(
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
    pub(crate) fn calculate_emotional_intensity(
        &self,
        features: &EmotionalProsodicFeatures,
    ) -> f32 {
        let f0_intensity = (features.f0_std / 50.0).min(1.0);
        let energy_intensity = features.mean_energy;
        let rate_intensity = ((features.speaking_rate - 5.0).abs() / 3.0).min(1.0);
        let rhythm_intensity = 1.0 - features.rhythm_regularity;
        (f0_intensity + energy_intensity + rate_intensity + rhythm_intensity) / 4.0
    }
    /// Calculate emotional consistency
    pub(crate) fn calculate_emotional_consistency(
        &self,
        features: &EmotionalProsodicFeatures,
    ) -> f32 {
        let f0_consistency = 1.0 - (features.f0_std / 100.0).min(1.0);
        let energy_consistency = 1.0 - features.energy_std;
        let rhythm_consistency = features.rhythm_regularity;
        (f0_consistency + energy_consistency + rhythm_consistency) / 3.0
    }
    /// Analyze emotional dynamics over time
    pub(crate) async fn analyze_emotional_dynamics(
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
    pub(crate) fn calculate_emotion_detection_confidence(
        &self,
        features: &EmotionalProsodicFeatures,
    ) -> f32 {
        let f0_confidence = (features.f0_std / 30.0).min(1.0);
        let energy_confidence = features.mean_energy;
        let rhythm_confidence = 1.0 - features.rhythm_regularity;
        ((f0_confidence + energy_confidence + rhythm_confidence) / 3.0).max(0.3)
    }
    pub(crate) fn calculate_duration_contrast(
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
    pub(crate) fn estimate_mean_f0(&self, alignment: &PhonemeAlignment) -> f32 {
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
    pub(crate) fn estimate_f0_std(&self, alignment: &PhonemeAlignment) -> f32 {
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
    pub(crate) fn estimate_f0_range(&self, alignment: &PhonemeAlignment) -> f32 {
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
    pub(crate) async fn calculate_speaking_rate_for_emotion(
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
    pub(crate) fn calculate_energy_features(&self, alignment: &PhonemeAlignment) -> (f32, f32) {
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
    pub(crate) fn calculate_pause_features(&self, alignment: &PhonemeAlignment) -> (f32, f32) {
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
    pub(crate) fn estimate_jitter(&self, alignment: &PhonemeAlignment) -> f32 {
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
    pub(crate) fn estimate_shimmer(&self, alignment: &PhonemeAlignment) -> f32 {
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
    pub(crate) fn calculate_stress_pattern_strength(&self, alignment: &PhonemeAlignment) -> f32 {
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
    pub(crate) fn extract_window_features(
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
    pub(crate) fn calculate_transition_smoothness_emotion(
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
    pub(crate) fn calculate_emotional_stability(
        &self,
        trajectory: &[(f32, EmotionalState, f32)],
    ) -> f32 {
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
    pub(crate) fn get_language_prosodic_profile(
        &self,
        language: LanguageCode,
    ) -> LanguageProsodicProfile {
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
    pub(crate) fn calculate_language_distance(
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
    pub(crate) fn compare_prosodic_features(
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
    pub(crate) fn calculate_prosodic_transfer_score(
        &self,
        comparison: &LanguageComparisonDetails,
    ) -> f32 {
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
    pub(crate) fn generate_adaptation_recommendations(
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
    pub(crate) fn calculate_cross_linguistic_intelligibility(
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
