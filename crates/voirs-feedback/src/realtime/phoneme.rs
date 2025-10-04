//! Phoneme analysis and processing

use super::types::*;
use crate::FeedbackError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

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
    pub async fn analyze_phonemes(
        &self,
        audio_data: &[f32],
        expected_phonemes: &[PhonemeInfo],
    ) -> Result<PhonemeAnalysisResult, FeedbackError> {
        let start_time = std::time::Instant::now();

        // Simulate phoneme detection (in a real implementation, this would use signal processing)
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

    /// Detect phonemes in audio signal
    async fn detect_phonemes(
        &self,
        audio_data: &[f32],
        expected_phonemes: &[PhonemeInfo],
    ) -> Result<Vec<DetectedPhoneme>, FeedbackError> {
        let mut detected = Vec::new();

        // Simulate detection based on expected phonemes with some variation
        for (i, expected) in expected_phonemes.iter().enumerate() {
            let start_time = i as f64 * 100.0; // 100ms per phoneme simulation
            let duration = 100.0 + (scirs2_core::random::random::<f64>() - 0.5) * 20.0; // ±10ms variation from base 100ms

            // Simulate formant detection
            let formants = if let Some(reference) = self.reference_phonemes.get(&expected.symbol) {
                if reference.features.vowel {
                    vec![
                        reference.formant_ranges.f1_range.0
                            + (scirs2_core::random::random::<f64>() as f32
                                * (reference.formant_ranges.f1_range.1
                                    - reference.formant_ranges.f1_range.0)),
                        reference.formant_ranges.f2_range.0
                            + (scirs2_core::random::random::<f64>() as f32
                                * (reference.formant_ranges.f2_range.1
                                    - reference.formant_ranges.f2_range.0)),
                        reference.formant_ranges.f3_range.0
                            + (scirs2_core::random::random::<f64>() as f32
                                * (reference.formant_ranges.f3_range.1
                                    - reference.formant_ranges.f3_range.0)),
                    ]
                } else {
                    vec![] // Consonants might not have clear formants
                }
            } else {
                vec![]
            };

            // Calculate confidence and accuracy
            let confidence = 0.7 + scirs2_core::random::random::<f64>() as f32 * 0.3; // 70-100% confidence
            let accuracy_score = self.calculate_phoneme_accuracy(&expected.symbol, &formants);

            // Generate feedback points
            let feedback_points = self.generate_feedback_points(&expected.symbol, accuracy_score);

            detected.push(DetectedPhoneme {
                symbol: expected.symbol.clone(),
                confidence,
                start_time_ms: start_time,
                duration_ms: duration,
                formants,
                accuracy_score,
                feedback_points,
            });
        }

        Ok(detected)
    }

    /// Calculate accuracy score for a single phoneme
    fn calculate_phoneme_accuracy(&self, symbol: &str, formants: &[f32]) -> f32 {
        if let Some(reference) = self.reference_phonemes.get(symbol) {
            if reference.features.vowel && formants.len() >= 2 {
                // For vowels, check formant accuracy
                let f1_accuracy =
                    self.calculate_formant_accuracy(formants[0], reference.formant_ranges.f1_range);
                let f2_accuracy =
                    self.calculate_formant_accuracy(formants[1], reference.formant_ranges.f2_range);
                (f1_accuracy + f2_accuracy) / 2.0
            } else {
                // For consonants, use a simulated accuracy
                0.8 + (scirs2_core::random::random::<f64>() * 0.2) as f32
            }
        } else {
            0.5 // Unknown phoneme
        }
    }

    /// Calculate formant accuracy
    fn calculate_formant_accuracy(&self, detected: f32, expected_range: (f32, f32)) -> f32 {
        let center = (expected_range.0 + expected_range.1) / 2.0;
        let tolerance = (expected_range.1 - expected_range.0) / 2.0;
        let distance = (detected - center).abs();

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
                    feedback.push(format!("Focus on mouth position for '{}' sound", symbol));
                    feedback.push("Pay attention to tongue placement".to_string());
                    feedback.push("Practice vowel clarity".to_string());
                } else if reference.features.consonant {
                    if reference.features.stop {
                        feedback.push(format!(
                            "Work on stop consonant '{}' - ensure complete closure",
                            symbol
                        ));
                    }
                    if reference.features.fricative {
                        feedback.push(format!(
                            "Practice fricative '{}' - maintain airflow",
                            symbol
                        ));
                    }
                    if reference.features.nasal {
                        feedback.push(format!("Ensure nasal resonance for '{}'", symbol));
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

        // Timing accuracy: simplified calculation since expected timing is not available
        let timing_accuracy = if detected.len() == expected.len() {
            // Give a neutral score if phoneme count matches
            0.8
        } else {
            0.5 // Penalty for wrong number of phonemes
        };

        // Pronunciation score: combination of accuracy and timing
        let pronunciation_score = overall_accuracy * 0.7 + timing_accuracy * 0.3;

        (overall_accuracy, timing_accuracy, pronunciation_score)
    }

    /// Get phoneme reference data
    pub fn get_phoneme_reference(&self, symbol: &str) -> Option<&PhonemeReference> {
        self.reference_phonemes.get(symbol)
    }

    /// Get all supported phonemes
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

// Simple random number generation for simulation
mod rand {
    use std::cell::Cell;

    thread_local! {
        static SEED: Cell<u64> = Cell::new(1);
    }

    pub fn random<T>() -> T
    where
        T: From<f64>,
    {
        SEED.with(|seed| {
            let s = seed.get();
            seed.set(s.wrapping_mul(1103515245).wrapping_add(12345));
            T::from((s as f64) / (u64::MAX as f64))
        })
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
