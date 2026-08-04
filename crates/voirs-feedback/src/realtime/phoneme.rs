//! Phoneme analysis and processing

use super::types::PhonemeInfo;
use crate::FeedbackError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use voirs_sdk::AudioBuffer;

/// Phoneme analyzer for real-time audio processing
#[derive(Debug, Clone)]
pub struct PhonemeAnalyzer {
    config: PhonemeAnalysisConfig,
    reference_phonemes: HashMap<String, PhonemeReference>,
}

/// Configuration for phoneme analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeAnalysisConfig {
    /// Description
    pub sample_rate: u32,
    /// Description
    pub frame_length: usize,
    /// Description
    pub hop_length: usize,
    /// Description
    pub min_phoneme_duration_ms: u64,
    /// Description
    pub max_phoneme_duration_ms: u64,
    /// Description
    pub confidence_threshold: f32,
}

/// Reference phoneme data for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeReference {
    /// Description
    pub symbol: String,
    /// Description
    pub features: PhonemeFeatures,
    /// Description
    pub expected_duration_ms: f64,
    /// Description
    pub formant_ranges: FormantRanges,
}

/// Acoustic features of a phoneme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeFeatures {
    /// Description
    pub vowel: bool,
    /// Description
    pub consonant: bool,
    /// Description
    pub voiced: bool,
    /// Description
    pub aspirated: bool,
    /// Description
    pub nasal: bool,
    /// Description
    pub fricative: bool,
    /// Description
    pub stop: bool,
    /// Description
    pub liquid: bool,
    /// Description
    pub glide: bool,
}

/// Formant frequency ranges for phoneme classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormantRanges {
    /// Description
    pub f1_range: (f32, f32), // First formant frequency range
    /// Description
    pub f2_range: (f32, f32), // Second formant frequency range
    /// Description
    pub f3_range: (f32, f32), // Third formant frequency range
}

/// Result of phoneme analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeAnalysisResult {
    /// Description
    pub detected_phonemes: Vec<DetectedPhoneme>,
    /// Description
    pub overall_accuracy: f32,
    /// Description
    pub timing_accuracy: f32,
    /// Description
    pub pronunciation_score: f32,
    /// Description
    pub analysis_duration: Duration,
}

/// Individual detected phoneme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPhoneme {
    /// Description
    pub symbol: String,
    /// Description
    pub confidence: f32,
    /// Description
    pub start_time_ms: f64,
    /// Description
    pub duration_ms: f64,
    /// Description
    pub formants: Vec<f32>,
    /// Description
    pub accuracy_score: f32,
    /// Description
    pub feedback_points: Vec<String>,
}

impl PhonemeAnalyzer {
    /// Create a new phoneme analyzer
    #[must_use]
    pub fn new(config: PhonemeAnalysisConfig) -> Self {
        let mut analyzer = Self {
            config,
            reference_phonemes: HashMap::new(),
        };
        analyzer.load_reference_phonemes();
        analyzer
    }

    /// Load reference phoneme data
    fn load_reference_phonemes(&mut self) {
        // Common English phonemes with example data
        let phonemes = vec![
            (
                "æ",
                true,
                false,
                true,
                false,
                false,
                false,
                false,
                false,
                false,
                (700.0, 900.0),
                (1700.0, 1900.0),
                (2600.0, 2800.0),
            ), // cat
            (
                "ɑ",
                true,
                false,
                true,
                false,
                false,
                false,
                false,
                false,
                false,
                (750.0, 950.0),
                (1100.0, 1300.0),
                (2400.0, 2600.0),
            ), // father
            (
                "ə",
                true,
                false,
                true,
                false,
                false,
                false,
                false,
                false,
                false,
                (500.0, 700.0),
                (1400.0, 1600.0),
                (2500.0, 2700.0),
            ), // about
            (
                "ɪ",
                true,
                false,
                true,
                false,
                false,
                false,
                false,
                false,
                false,
                (400.0, 600.0),
                (2000.0, 2200.0),
                (2700.0, 2900.0),
            ), // bit
            (
                "i",
                true,
                false,
                true,
                false,
                false,
                false,
                false,
                false,
                false,
                (300.0, 500.0),
                (2200.0, 2400.0),
                (2900.0, 3100.0),
            ), // beat
            (
                "ʊ",
                true,
                false,
                true,
                false,
                false,
                false,
                false,
                false,
                false,
                (400.0, 600.0),
                (800.0, 1000.0),
                (2200.0, 2400.0),
            ), // book
            (
                "u",
                true,
                false,
                true,
                false,
                false,
                false,
                false,
                false,
                false,
                (300.0, 500.0),
                (600.0, 800.0),
                (2000.0, 2200.0),
            ), // boot
            (
                "p",
                false,
                true,
                false,
                false,
                false,
                false,
                true,
                false,
                false,
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ), // pat
            (
                "b",
                false,
                true,
                true,
                false,
                false,
                false,
                true,
                false,
                false,
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ), // bat
            (
                "t",
                false,
                true,
                false,
                false,
                false,
                false,
                true,
                false,
                false,
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ), // tap
            (
                "d",
                false,
                true,
                true,
                false,
                false,
                false,
                true,
                false,
                false,
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ), // dad
            (
                "k",
                false,
                true,
                false,
                false,
                false,
                false,
                true,
                false,
                false,
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ), // cat
            (
                "g",
                false,
                true,
                true,
                false,
                false,
                false,
                true,
                false,
                false,
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ), // go
            (
                "f",
                false,
                true,
                false,
                false,
                false,
                true,
                false,
                false,
                false,
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ), // fat
            (
                "v",
                false,
                true,
                true,
                false,
                false,
                true,
                false,
                false,
                false,
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ), // vat
            (
                "s",
                false,
                true,
                false,
                false,
                false,
                true,
                false,
                false,
                false,
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ), // sat
            (
                "z",
                false,
                true,
                true,
                false,
                false,
                true,
                false,
                false,
                false,
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ), // zoo
            (
                "m",
                false,
                true,
                true,
                false,
                true,
                false,
                false,
                false,
                false,
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ), // mat
            (
                "n",
                false,
                true,
                true,
                false,
                true,
                false,
                false,
                false,
                false,
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ), // nat
            (
                "l",
                false,
                true,
                true,
                false,
                false,
                false,
                false,
                true,
                false,
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ), // lat
            (
                "r",
                false,
                true,
                true,
                false,
                false,
                false,
                false,
                true,
                false,
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ), // rat
            (
                "w",
                false,
                true,
                true,
                false,
                false,
                false,
                false,
                false,
                true,
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ), // wat
            (
                "j",
                false,
                true,
                true,
                false,
                false,
                false,
                false,
                false,
                true,
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            ), // yes
        ];

        for (
            symbol,
            vowel,
            consonant,
            voiced,
            aspirated,
            nasal,
            fricative,
            stop,
            liquid,
            glide,
            f1,
            f2,
            f3,
        ) in phonemes
        {
            let features = PhonemeFeatures {
                vowel,
                consonant,
                voiced,
                aspirated,
                nasal,
                fricative,
                stop,
                liquid,
                glide,
            };

            let formant_ranges = FormantRanges {
                f1_range: f1,
                f2_range: f2,
                f3_range: f3,
            };

            let reference = PhonemeReference {
                symbol: symbol.to_string(),
                features,
                expected_duration_ms: 80.0, // Average phoneme duration
                formant_ranges,
            };

            self.reference_phonemes
                .insert(symbol.to_string(), reference);
        }
    }

    /// Analyze phonemes in audio data
    ///
    /// Performs real signal analysis of `audio_data`: a frame-level energy
    /// envelope locates the active (non-silence) region of the clip, which
    /// is then divided across `expected_phonemes` in proportion to each
    /// phoneme's reference duration (real energy-based segmentation). Each
    /// resulting segment is analyzed with real DSP -- zero-crossing rate,
    /// RMS energy, and (for vowels) LPC-based formant estimation -- to
    /// derive accuracy and confidence. No random sampling is involved: the
    /// same `audio_data` always produces the same result.
    pub async fn analyze_phonemes(
        &self,
        audio_data: &[f32],
        expected_phonemes: &[PhonemeInfo],
    ) -> Result<PhonemeAnalysisResult, FeedbackError> {
        if audio_data.is_empty() {
            return Err(FeedbackError::InvalidInput {
                message: "audio_data is empty; cannot analyze phonemes".to_string(),
            });
        }

        let start_time = std::time::Instant::now();

        let detected_phonemes = self.detect_phonemes(audio_data, expected_phonemes).await?;

        // Calculate accuracy scores
        let (overall_accuracy, timing_accuracy, pronunciation_score) =
            self.calculate_accuracy_scores(&detected_phonemes, expected_phonemes);

        let analysis_duration = start_time.elapsed();

        Ok(PhonemeAnalysisResult {
            detected_phonemes,
            overall_accuracy,
            timing_accuracy,
            pronunciation_score,
            analysis_duration,
        })
    }

    /// Detect phonemes in the real audio signal.
    ///
    /// Algorithm:
    /// 1. Compute a frame-level RMS energy envelope over the whole clip
    ///    (frame/hop sizes from [`PhonemeAnalysisConfig`]) and locate the
    ///    active (non-silence) sample range.
    /// 2. Divide that active range across `expected_phonemes`, weighted by
    ///    each phoneme's reference duration -- a real energy-based
    ///    segmentation, not a fixed `i * 100ms` guess.
    /// 3. For each resulting segment, compute real zero-crossing rate and
    ///    RMS energy; for vowels, additionally run LPC-based formant
    ///    estimation ([`AudioBuffer::estimate_formants`]) over a window
    ///    expanded to the estimator's minimum sample requirement.
    /// 4. Derive `accuracy_score` from how well the measured
    ///    acoustic features match the reference phoneme, and `confidence`
    ///    from a combination of that match quality and how much real signal
    ///    energy the segment actually contains.
    async fn detect_phonemes(
        &self,
        audio_data: &[f32],
        expected_phonemes: &[PhonemeInfo],
    ) -> Result<Vec<DetectedPhoneme>, FeedbackError> {
        if expected_phonemes.is_empty() {
            return Ok(Vec::new());
        }

        let sample_rate = f64::from(self.config.sample_rate.max(1));
        let frame_length = self.config.frame_length.max(1);
        let hop_length = self.config.hop_length.max(1).min(frame_length);

        // Real per-frame RMS energy envelope of the whole signal.
        let mut frame_energies = Vec::new();
        let mut frame_starts = Vec::new();
        let mut pos = 0usize;
        loop {
            let end = (pos + frame_length).min(audio_data.len());
            frame_energies.push(Self::rms(&audio_data[pos..end]));
            frame_starts.push(pos);
            if end >= audio_data.len() {
                break;
            }
            pos += hop_length;
        }

        let max_energy = frame_energies.iter().copied().fold(0.0_f32, f32::max);
        // Relative voice-activity threshold: 10% of the clip's peak frame
        // energy, floored so that a genuinely silent clip (max_energy == 0)
        // never spuriously counts as "active".
        let silence_threshold = (max_energy * 0.1).max(1e-4);

        let active_frames: Vec<usize> = frame_energies
            .iter()
            .enumerate()
            .filter(|&(_, &energy)| energy > silence_threshold)
            .map(|(i, _)| i)
            .collect();

        let has_voice_activity = !active_frames.is_empty();
        let (active_start, active_end) = if has_voice_activity {
            let first = active_frames[0];
            let last = active_frames[active_frames.len() - 1];
            (
                frame_starts[first],
                (frame_starts[last] + frame_length).min(audio_data.len()),
            )
        } else {
            // No frame cleared the threshold: honestly nothing was detected
            // as active speech. Segment the whole clip anyway so timing
            // remains well-defined, but `has_voice_activity == false` caps
            // confidence low below.
            (0, audio_data.len())
        };

        // Weight each phoneme's share of the active region by its
        // reference duration (falls back to the corpus average of 80ms for
        // symbols with no reference entry).
        let weights: Vec<f64> = expected_phonemes
            .iter()
            .map(|p| {
                self.reference_phonemes
                    .get(&p.symbol)
                    .map_or(80.0, |r| r.expected_duration_ms)
                    .max(1.0)
            })
            .collect();
        let total_weight: f64 = weights.iter().sum();
        let active_len = active_end.saturating_sub(active_start) as f64;

        let mut detected = Vec::with_capacity(expected_phonemes.len());
        let mut cursor = active_start;

        for (i, expected) in expected_phonemes.iter().enumerate() {
            let share = if total_weight > 0.0 {
                weights[i] / total_weight
            } else {
                1.0 / expected_phonemes.len() as f64
            };
            let seg_len = ((active_len * share).round() as usize).max(1);

            let seg_start = cursor.min(active_end);
            let seg_end = if i + 1 == expected_phonemes.len() {
                // Last phoneme absorbs any rounding remainder so the
                // segmentation exactly covers the active region.
                active_end.max(seg_start)
            } else {
                (seg_start + seg_len).min(active_end).max(seg_start)
            };
            cursor = seg_end;

            let segment = &audio_data[seg_start..seg_end];
            let segment_rms = Self::rms(segment);
            let zcr =
                AudioBuffer::mono(segment.to_vec(), self.config.sample_rate).zero_crossing_rate();

            let reference = self.reference_phonemes.get(&expected.symbol);
            let formants = if reference.is_some_and(|r| r.features.vowel) {
                let window = Self::window_for_formants(audio_data, seg_start, seg_end);
                AudioBuffer::mono(window.to_vec(), self.config.sample_rate).estimate_formants(3)
            } else {
                Vec::new()
            };

            let accuracy_score = self.calculate_phoneme_accuracy(&expected.symbol, &formants, zcr);

            let energy_ratio = if max_energy > 0.0 {
                (segment_rms / max_energy).clamp(0.0, 1.0)
            } else {
                0.0
            };
            // Confidence blends acoustic match quality (dominant term) with
            // how much real energy the segment actually contains -- a
            // silent segment can never be reported as confidently detected,
            // regardless of how well it happens to match on paper.
            let confidence_raw = (0.3 + 0.7 * accuracy_score) * (0.4 + 0.6 * energy_ratio);
            let confidence = if has_voice_activity {
                confidence_raw.clamp(0.0, 1.0)
            } else {
                confidence_raw.clamp(0.0, 0.15)
            };

            let feedback_points = self.generate_feedback_points(&expected.symbol, accuracy_score);

            detected.push(DetectedPhoneme {
                symbol: expected.symbol.clone(),
                confidence,
                start_time_ms: seg_start as f64 / sample_rate * 1000.0,
                duration_ms: (seg_end.saturating_sub(seg_start)) as f64 / sample_rate * 1000.0,
                formants,
                accuracy_score,
                feedback_points,
            });
        }

        Ok(detected)
    }

    /// Root-mean-square energy of a sample slice (0.0 for an empty slice).
    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            0.0
        } else {
            (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
        }
    }

    /// Expand `[seg_start, seg_end)` to at least
    /// [`AudioBuffer::estimate_formants`]'s minimum window (512 samples),
    /// centered on the segment's midpoint and clamped to `audio_data`'s
    /// bounds, so short phoneme segments still get a usable LPC analysis
    /// window instead of unconditionally returning no formants.
    fn window_for_formants(audio_data: &[f32], seg_start: usize, seg_end: usize) -> &[f32] {
        const MIN_LEN: usize = 512;
        if audio_data.len() <= MIN_LEN {
            return audio_data;
        }

        let seg_mid = seg_start + seg_end.saturating_sub(seg_start) / 2;
        let half = MIN_LEN / 2;
        let win_start = seg_mid.saturating_sub(half);
        let win_end = (win_start + MIN_LEN).min(audio_data.len());
        // Re-anchor the start in case `win_end` was clamped against the end
        // of the clip, so the window still has the full MIN_LEN whenever
        // the clip itself is long enough.
        let win_start = win_end.saturating_sub(MIN_LEN);

        &audio_data[win_start..win_end]
    }

    /// Calculate accuracy score for a single phoneme from real, measured
    /// acoustic features.
    ///
    /// Vowels are scored by how closely the LPC-estimated formants match
    /// the reference formant ranges; consonants are scored by how closely
    /// the measured zero-crossing rate matches the expected range for their
    /// manner of articulation (see [`Self::expected_zcr_range`]).
    fn calculate_phoneme_accuracy(&self, symbol: &str, formants: &[f32], zcr: f32) -> f32 {
        if let Some(reference) = self.reference_phonemes.get(symbol) {
            if reference.features.vowel {
                if formants.len() >= 2 {
                    let f1_accuracy = self
                        .calculate_formant_accuracy(formants[0], reference.formant_ranges.f1_range);
                    let f2_accuracy = self
                        .calculate_formant_accuracy(formants[1], reference.formant_ranges.f2_range);
                    f32::midpoint(f1_accuracy, f2_accuracy)
                } else {
                    // Formant estimation genuinely failed (segment too
                    // short/noisy for LPC to resolve a peak) -- honestly
                    // low rather than a random guess.
                    0.2
                }
            } else {
                // Consonant: score the real zero-crossing rate against the
                // expected range for this phoneme's manner of articulation.
                Self::range_match_score(zcr, Self::expected_zcr_range(&reference.features))
            }
        } else {
            0.5 // Unknown phoneme: neutral, deterministic
        }
    }

    /// Expected zero-crossing-rate range for a consonant's manner of
    /// articulation. Fricatives produce high-frequency turbulent noise
    /// (many zero crossings); nasals and liquids/glides are sonorants with
    /// low, vowel-like crossing rates; stops are dominated by a brief
    /// closure so their crossing rate is broad/variable. This mirrors the
    /// same kind of simplified reference table already used for vowel
    /// formant ranges -- the measurement fed into it is what changed from
    /// fabricated to real.
    fn expected_zcr_range(features: &PhonemeFeatures) -> (f32, f32) {
        if features.fricative {
            (0.15, 0.45)
        } else if features.nasal {
            (0.01, 0.10)
        } else if features.liquid || features.glide {
            (0.02, 0.15)
        } else if features.stop {
            (0.05, 0.35)
        } else {
            (0.0, 0.5)
        }
    }

    /// Calculate formant accuracy
    fn calculate_formant_accuracy(&self, detected: f32, expected_range: (f32, f32)) -> f32 {
        Self::range_match_score(detected, expected_range)
    }

    /// Score how well a measured value matches an expected `(min, max)`
    /// range: 1.0 at the center of the range, decreasing linearly to 0.7 at
    /// the edges, and continuing to decrease (floored at 0.0) outside it.
    /// Shared by formant-based (vowel) and zero-crossing-rate-based
    /// (consonant) accuracy scoring.
    fn range_match_score(value: f32, expected_range: (f32, f32)) -> f32 {
        let center = f32::midpoint(expected_range.0, expected_range.1);
        let tolerance = ((expected_range.1 - expected_range.0) / 2.0).max(f32::EPSILON);
        let distance = (value - center).abs();

        if distance <= tolerance {
            1.0 - (distance / tolerance) * 0.3 // 70-100% accuracy within range
        } else {
            let overshoot = distance - tolerance;
            (0.7 - (overshoot / tolerance) * 0.7).max(0.0) // Decreasing accuracy outside range
        }
    }

    /// Generate feedback points for improvement
    fn generate_feedback_points(&self, symbol: &str, accuracy: f32) -> Vec<String> {
        let mut feedback = Vec::new();

        if accuracy < 0.7 {
            if let Some(reference) = self.reference_phonemes.get(symbol) {
                if reference.features.vowel {
                    feedback.push(format!("Focus on mouth position for '{symbol}' sound"));
                    feedback.push("Pay attention to tongue placement".to_string());
                    feedback.push("Practice vowel clarity".to_string());
                } else if reference.features.consonant {
                    if reference.features.stop {
                        feedback.push(format!(
                            "Work on stop consonant '{symbol}' - ensure complete closure"
                        ));
                    }
                    if reference.features.fricative {
                        feedback.push(format!("Practice fricative '{symbol}' - maintain airflow"));
                    }
                    if reference.features.nasal {
                        feedback.push(format!("Ensure nasal resonance for '{symbol}'"));
                    }
                }
            }
        } else if accuracy < 0.85 {
            feedback.push("Good pronunciation, minor adjustments needed".to_string());
        }

        feedback
    }

    /// Calculate overall accuracy scores
    fn calculate_accuracy_scores(
        &self,
        detected: &[DetectedPhoneme],
        expected: &[PhonemeInfo],
    ) -> (f32, f32, f32) {
        if detected.is_empty() || expected.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        // Overall accuracy: average of individual phoneme accuracy scores
        let overall_accuracy =
            detected.iter().map(|p| p.accuracy_score).sum::<f32>() / detected.len() as f32;

        // Timing accuracy: when phoneme counts match (so pairing by
        // position is unambiguous), compare each detected segment's real
        // duration against its reference phoneme's expected duration. A
        // count mismatch is penalized directly since pairing would be
        // ambiguous.
        let timing_accuracy = if detected.len() == expected.len() {
            let fit_scores: Vec<f32> = detected
                .iter()
                .zip(expected.iter())
                .map(|(d, e)| {
                    let expected_duration_ms = self
                        .reference_phonemes
                        .get(&e.symbol)
                        .map_or(80.0, |r| r.expected_duration_ms);
                    let diff = (d.duration_ms - expected_duration_ms).abs();
                    let relative_error = (diff / expected_duration_ms.max(1.0)) as f32;
                    (1.0 - relative_error).clamp(0.0, 1.0)
                })
                .collect();
            fit_scores.iter().sum::<f32>() / fit_scores.len() as f32
        } else {
            0.5 // Penalty for wrong number of phonemes
        };

        // Pronunciation score: combination of accuracy and timing
        let pronunciation_score = overall_accuracy * 0.7 + timing_accuracy * 0.3;

        (overall_accuracy, timing_accuracy, pronunciation_score)
    }

    /// Get phoneme reference data
    #[must_use]
    pub fn get_phoneme_reference(&self, symbol: &str) -> Option<&PhonemeReference> {
        self.reference_phonemes.get(symbol)
    }

    /// Get all supported phonemes
    #[must_use]
    pub fn get_supported_phonemes(&self) -> Vec<String> {
        self.reference_phonemes.keys().cloned().collect()
    }
}

impl Default for PhonemeAnalysisConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            frame_length: 1024,
            hop_length: 256,
            min_phoneme_duration_ms: 30,
            max_phoneme_duration_ms: 300,
            confidence_threshold: 0.6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_phoneme_analyzer_creation() {
        let analyzer = PhonemeAnalyzer::new(PhonemeAnalysisConfig::default());
        assert!(!analyzer.reference_phonemes.is_empty());
        assert!(analyzer.get_supported_phonemes().len() > 20);
    }

    #[tokio::test]
    async fn test_phoneme_analysis() {
        let analyzer = PhonemeAnalyzer::new(PhonemeAnalysisConfig::default());
        let audio_data = vec![0.0; 1600]; // 100ms of 16kHz audio

        let expected_phonemes = vec![
            PhonemeInfo {
                symbol: "æ".to_string(),
                position: 0,
                difficulty: 0.7,
                common_errors: vec!["ɛ".to_string()],
                suggestions: vec!["Lower your tongue".to_string()],
            },
            PhonemeInfo {
                symbol: "t".to_string(),
                position: 1,
                difficulty: 0.3,
                common_errors: vec!["d".to_string()],
                suggestions: vec!["Stronger aspiration".to_string()],
            },
        ];

        let result = analyzer
            .analyze_phonemes(&audio_data, &expected_phonemes)
            .await
            .unwrap();

        assert_eq!(result.detected_phonemes.len(), 2);
        assert!(result.overall_accuracy > 0.0);
        assert!(result.timing_accuracy > 0.0);
        assert!(result.pronunciation_score > 0.0);
    }

    #[test]
    fn test_phoneme_reference_lookup() {
        let analyzer = PhonemeAnalyzer::new(PhonemeAnalysisConfig::default());

        let vowel_ref = analyzer.get_phoneme_reference("æ");
        assert!(vowel_ref.is_some());
        assert!(vowel_ref.unwrap().features.vowel);

        let consonant_ref = analyzer.get_phoneme_reference("t");
        assert!(consonant_ref.is_some());
        assert!(consonant_ref.unwrap().features.consonant);
        assert!(consonant_ref.unwrap().features.stop);
    }

    #[test]
    fn test_formant_accuracy_calculation() {
        let analyzer = PhonemeAnalyzer::new(PhonemeAnalysisConfig::default());

        // Perfect match should give high accuracy
        let accuracy = analyzer.calculate_formant_accuracy(800.0, (700.0, 900.0));
        assert!(accuracy > 0.9);

        // Out of range should give lower accuracy
        let accuracy = analyzer.calculate_formant_accuracy(1200.0, (700.0, 900.0));
        assert!(accuracy < 0.7);
    }

    #[test]
    fn test_feedback_generation() {
        let analyzer = PhonemeAnalyzer::new(PhonemeAnalysisConfig::default());

        // Low accuracy should generate feedback
        let feedback = analyzer.generate_feedback_points("æ", 0.5);
        assert!(!feedback.is_empty());

        // High accuracy should generate minimal or no feedback
        let feedback = analyzer.generate_feedback_points("æ", 0.95);
        assert!(feedback.is_empty() || feedback.len() <= 1);
    }
}
