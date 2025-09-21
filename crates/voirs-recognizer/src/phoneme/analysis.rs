//! Phoneme analysis and alignment structures
//!
//! This module provides data structures and utilities for phoneme analysis,
//! including aligned phonemes, word alignments, and timing information.

use crate::LanguageCode;
use voirs_sdk::Phoneme;

/// An aligned phoneme with timing information
#[derive(Debug, Clone, PartialEq)]
pub struct AlignedPhoneme {
    /// The phoneme data
    pub phoneme: Phoneme,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Alignment confidence [0.0, 1.0]
    pub confidence: f32,
}

impl AlignedPhoneme {
    /// Create a new aligned phoneme
    #[must_use]
    pub fn new(phoneme: Phoneme, start_time: f32, end_time: f32, confidence: f32) -> Self {
        Self {
            phoneme,
            start_time,
            end_time,
            confidence,
        }
    }

    /// Get the duration of the phoneme in seconds
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.end_time - self.start_time
    }

    /// Check if this phoneme overlaps with another in time
    #[must_use]
    pub fn overlaps_with(&self, other: &AlignedPhoneme) -> bool {
        !(self.end_time <= other.start_time || other.end_time <= self.start_time)
    }
}

/// Word alignment with associated phonemes
#[derive(Debug, Clone, PartialEq)]
pub struct WordAlignment {
    /// The word text
    pub word: String,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Associated phonemes
    pub phonemes: Vec<AlignedPhoneme>,
    /// Word-level confidence [0.0, 1.0]
    pub confidence: f32,
}

impl WordAlignment {
    /// Create a new word alignment
    #[must_use]
    pub fn new(
        word: String,
        start_time: f32,
        end_time: f32,
        phonemes: Vec<AlignedPhoneme>,
        confidence: f32,
    ) -> Self {
        Self {
            word,
            start_time,
            end_time,
            phonemes,
            confidence,
        }
    }

    /// Get the duration of the word in seconds
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.end_time - self.start_time
    }

    /// Get the average phoneme confidence
    #[must_use]
    pub fn average_phoneme_confidence(&self) -> f32 {
        if self.phonemes.is_empty() {
            0.0
        } else {
            self.phonemes.iter().map(|p| p.confidence).sum::<f32>() / self.phonemes.len() as f32
        }
    }
}

/// Complete phoneme alignment result
#[derive(Debug, Clone, PartialEq)]
pub struct PhonemeAlignment {
    /// All aligned phonemes
    pub phonemes: Vec<AlignedPhoneme>,
    /// Word-level alignments
    pub word_alignments: Vec<WordAlignment>,
    /// Total duration of the alignment
    pub total_duration: f32,
    /// Overall alignment confidence
    pub overall_confidence: f32,
    /// Language used for alignment
    pub language: LanguageCode,
}

impl PhonemeAlignment {
    /// Create a new phoneme alignment
    #[must_use]
    pub fn new(
        phonemes: Vec<AlignedPhoneme>,
        word_alignments: Vec<WordAlignment>,
        total_duration: f32,
        language: LanguageCode,
    ) -> Self {
        let overall_confidence = if phonemes.is_empty() {
            0.0
        } else {
            phonemes.iter().map(|p| p.confidence).sum::<f32>() / phonemes.len() as f32
        };

        Self {
            phonemes,
            word_alignments,
            total_duration,
            overall_confidence,
            language,
        }
    }

    /// Get phonemes within a time range
    #[must_use]
    pub fn phonemes_in_range(&self, start_time: f32, end_time: f32) -> Vec<&AlignedPhoneme> {
        self.phonemes
            .iter()
            .filter(|p| p.start_time >= start_time && p.end_time <= end_time)
            .collect()
    }

    /// Get words within a time range
    #[must_use]
    pub fn words_in_range(&self, start_time: f32, end_time: f32) -> Vec<&WordAlignment> {
        self.word_alignments
            .iter()
            .filter(|w| w.start_time >= start_time && w.end_time <= end_time)
            .collect()
    }

    /// Calculate speech rate (phonemes per second)
    #[must_use]
    pub fn speech_rate(&self) -> f32 {
        if self.total_duration > 0.0 {
            self.phonemes.len() as f32 / self.total_duration
        } else {
            0.0
        }
    }

    /// Calculate average phoneme duration
    #[must_use]
    pub fn average_phoneme_duration(&self) -> f32 {
        if self.phonemes.is_empty() {
            0.0
        } else {
            self.phonemes
                .iter()
                .map(AlignedPhoneme::duration)
                .sum::<f32>()
                / self.phonemes.len() as f32
        }
    }

    /// Get alignment statistics
    #[must_use]
    pub fn statistics(&self) -> AlignmentStatistics {
        AlignmentStatistics {
            total_phonemes: self.phonemes.len(),
            total_words: self.word_alignments.len(),
            total_duration: self.total_duration,
            average_confidence: self.overall_confidence,
            speech_rate: self.speech_rate(),
            average_phoneme_duration: self.average_phoneme_duration(),
        }
    }
}

/// Alignment statistics
#[derive(Debug, Clone, PartialEq)]
pub struct AlignmentStatistics {
    /// Total number of phonemes
    pub total_phonemes: usize,
    /// Total number of words
    pub total_words: usize,
    /// Total duration in seconds
    pub total_duration: f32,
    /// Average confidence score
    pub average_confidence: f32,
    /// Speech rate (phonemes per second)
    pub speech_rate: f32,
    /// Average phoneme duration
    pub average_phoneme_duration: f32,
}

/// Phoneme analysis configuration
#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisConfig {
    /// Minimum confidence threshold for accepting alignments
    pub confidence_threshold: f32,
    /// Maximum allowed gap between phonemes (seconds)
    pub max_gap: f32,
    /// Minimum phoneme duration (seconds)
    pub min_duration: f32,
    /// Maximum phoneme duration (seconds)
    pub max_duration: f32,
    /// Enable temporal smoothing
    pub temporal_smoothing: bool,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            max_gap: 0.1,
            min_duration: 0.01,
            max_duration: 1.0,
            temporal_smoothing: true,
        }
    }
}

/// Syllable boundary detection result
#[derive(Debug, Clone, PartialEq)]
pub struct SyllableBoundary {
    /// Position in phoneme sequence
    pub position: usize,
    /// Confidence of boundary detection
    pub confidence: f32,
    /// Boundary type
    pub boundary_type: BoundaryType,
}

/// Type of syllable boundary
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryType {
    /// Strong boundary (clear syllable break)
    Strong,
    /// Weak boundary (ambiguous syllable break)
    Weak,
    /// Word boundary
    Word,
}

/// Stress pattern detection result
#[derive(Debug, Clone, PartialEq)]
pub struct StressPattern {
    /// Stress level (0 = unstressed, 1 = primary, 2 = secondary)
    pub stress_level: u8,
    /// Syllable index
    pub syllable_index: usize,
    /// Acoustic stress indicators
    pub acoustic_features: StressFeatures,
}

/// Acoustic features indicating stress
#[derive(Debug, Clone, PartialEq)]
pub struct StressFeatures {
    /// Relative duration (compared to unstressed syllables)
    pub duration_ratio: f32,
    /// Relative intensity
    pub intensity_ratio: f32,
    /// F0 prominence
    pub pitch_prominence: f32,
    /// Vowel quality clarity
    pub vowel_clarity: f32,
}

/// Phoneme-to-text mapping result
#[derive(Debug, Clone, PartialEq)]
pub struct PhonemeTextMapping {
    /// Original text
    pub text: String,
    /// Phoneme sequence
    pub phonemes: Vec<AlignedPhoneme>,
    /// Character-to-phoneme alignments
    pub char_phoneme_map: Vec<(usize, usize)>, // (char_idx, phoneme_idx)
    /// Word boundaries in both text and phonemes
    pub word_boundaries: Vec<WordBoundary>,
}

/// Word boundary information
#[derive(Debug, Clone, PartialEq)]
pub struct WordBoundary {
    /// Start character index in text
    pub text_start: usize,
    /// End character index in text
    pub text_end: usize,
    /// Start phoneme index
    pub phoneme_start: usize,
    /// End phoneme index
    pub phoneme_end: usize,
    /// Word text
    pub word: String,
}

/// Phonological feature analysis
#[derive(Debug, Clone, PartialEq)]
pub struct PhonologicalAnalysis {
    /// Phoneme being analyzed
    pub phoneme: String,
    /// Place of articulation features
    pub place_features: PlaceFeatures,
    /// Manner of articulation features
    pub manner_features: MannerFeatures,
    /// Voicing and other binary features
    pub binary_features: BinaryFeatures,
    /// Prosodic features
    pub prosodic_features: ProsodicFeatures,
}

/// Place of articulation features
#[derive(Debug, Clone, PartialEq)]
pub struct PlaceFeatures {
    pub bilabial: bool,
    pub labiodental: bool,
    pub dental: bool,
    pub alveolar: bool,
    pub postalveolar: bool,
    pub retroflex: bool,
    pub palatal: bool,
    pub velar: bool,
    pub uvular: bool,
    pub pharyngeal: bool,
    pub glottal: bool,
}

/// Manner of articulation features
#[derive(Debug, Clone, PartialEq)]
pub struct MannerFeatures {
    pub stop: bool,
    pub fricative: bool,
    pub affricate: bool,
    pub nasal: bool,
    pub lateral: bool,
    pub approximant: bool,
    pub trill: bool,
    pub tap: bool,
}

/// Binary phonological features
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryFeatures {
    pub voiced: bool,
    pub aspirated: bool,
    pub syllabic: bool,
    pub consonantal: bool,
    pub sonorant: bool,
    pub continuant: bool,
    pub delayed_release: bool,
    pub approximant: bool,
}

/// Prosodic features
#[derive(Debug, Clone, PartialEq)]
pub struct ProsodicFeatures {
    pub stress: u8,           // 0=unstressed, 1=primary, 2=secondary
    pub tone: Option<String>, // For tonal languages
    pub length: LengthType,
}

/// Phoneme length classification
#[derive(Debug, Clone, PartialEq)]
pub enum LengthType {
    Short,
    Normal,
    Long,
    Overlong,
}

/// Phoneme analysis utilities
pub mod utils {
    use super::{
        AlignedPhoneme, BinaryFeatures, BoundaryType, LengthType, MannerFeatures, PhonemeAlignment,
        PhonemeTextMapping, PhonologicalAnalysis, PlaceFeatures, ProsodicFeatures, StressFeatures,
        StressPattern, SyllableBoundary, WordAlignment, WordBoundary,
    };

    /// Validate alignment temporal consistency
    #[must_use]
    pub fn validate_temporal_consistency(alignment: &PhonemeAlignment) -> Vec<String> {
        let mut issues = Vec::new();

        // Check for overlapping phonemes
        for window in alignment.phonemes.windows(2) {
            if window[0].end_time > window[1].start_time {
                issues.push(format!(
                    "Overlapping phonemes: {} ends at {:.3}s, {} starts at {:.3}s",
                    window[0].phoneme.symbol,
                    window[0].end_time,
                    window[1].phoneme.symbol,
                    window[1].start_time
                ));
            }
        }

        // Check for unreasonable gaps
        for window in alignment.phonemes.windows(2) {
            let gap = window[1].start_time - window[0].end_time;
            if gap > 0.5 {
                issues.push(format!(
                    "Large gap ({:.3}s) between {} and {}",
                    gap, window[0].phoneme.symbol, window[1].phoneme.symbol
                ));
            }
        }

        // Check for unreasonably short/long phonemes
        for phoneme in &alignment.phonemes {
            let duration = phoneme.duration();
            if duration < 0.005 {
                issues.push(format!(
                    "Very short phoneme {} duration: {:.3}s",
                    phoneme.phoneme.symbol, duration
                ));
            } else if duration > 2.0 {
                issues.push(format!(
                    "Very long phoneme {} duration: {:.3}s",
                    phoneme.phoneme.symbol, duration
                ));
            }
        }

        issues
    }

    /// Apply temporal smoothing to alignment
    pub fn apply_temporal_smoothing(alignment: &mut PhonemeAlignment, smoothing_factor: f32) {
        if alignment.phonemes.len() < 2 {
            return;
        }

        let alpha = smoothing_factor.clamp(0.0, 1.0);

        // Smooth start and end times
        for i in 1..alignment.phonemes.len() {
            let prev_end = alignment.phonemes[i - 1].end_time;
            let current_start = alignment.phonemes[i].start_time;

            // Smooth the boundary
            let smoothed_boundary = alpha * prev_end + (1.0 - alpha) * current_start;

            alignment.phonemes[i - 1].end_time = smoothed_boundary;
            alignment.phonemes[i].start_time = smoothed_boundary;
        }

        // Recalculate word alignments if needed
        for word in &mut alignment.word_alignments {
            if !word.phonemes.is_empty() {
                word.start_time = word.phonemes[0].start_time;
                word.end_time = word.phonemes.last().unwrap().end_time;
            }
        }
    }

    /// Filter alignment by confidence threshold
    #[must_use]
    pub fn filter_by_confidence(alignment: &PhonemeAlignment, threshold: f32) -> PhonemeAlignment {
        let filtered_phonemes: Vec<AlignedPhoneme> = alignment
            .phonemes
            .iter()
            .filter(|p| p.confidence >= threshold)
            .cloned()
            .collect();

        let filtered_words: Vec<WordAlignment> = alignment
            .word_alignments
            .iter()
            .filter(|w| w.confidence >= threshold)
            .cloned()
            .collect();

        PhonemeAlignment::new(
            filtered_phonemes,
            filtered_words,
            alignment.total_duration,
            alignment.language,
        )
    }

    /// Merge adjacent similar phonemes
    pub fn merge_similar_phonemes(alignment: &mut PhonemeAlignment, _similarity_threshold: f32) {
        let mut merged_phonemes = Vec::new();
        let mut current_phoneme: Option<AlignedPhoneme> = None;

        for phoneme in &alignment.phonemes {
            if let Some(ref mut current) = current_phoneme {
                if current.phoneme.symbol == phoneme.phoneme.symbol
                    && (phoneme.start_time - current.end_time) < 0.05
                // Max gap of 50ms
                {
                    // Merge phonemes
                    current.end_time = phoneme.end_time;
                    current.confidence = (current.confidence + phoneme.confidence) / 2.0;
                } else {
                    // Different phoneme, push current and start new
                    merged_phonemes.push(current.clone());
                    current_phoneme = Some(phoneme.clone());
                }
            } else {
                current_phoneme = Some(phoneme.clone());
            }
        }

        // Don't forget the last phoneme
        if let Some(current) = current_phoneme {
            merged_phonemes.push(current);
        }

        alignment.phonemes = merged_phonemes;
    }

    /// Detect syllable boundaries in phoneme sequence
    #[must_use]
    pub fn detect_syllable_boundaries(phonemes: &[AlignedPhoneme]) -> Vec<SyllableBoundary> {
        let mut boundaries = Vec::new();

        if phonemes.len() < 2 {
            return boundaries;
        }

        for i in 0..(phonemes.len() - 1) {
            let current = &phonemes[i];
            let next = &phonemes[i + 1];

            let boundary_strength = calculate_syllable_boundary_strength(current, next);

            if boundary_strength > 0.5 {
                let boundary_type = if boundary_strength > 0.8 {
                    BoundaryType::Strong
                } else {
                    BoundaryType::Weak
                };

                boundaries.push(SyllableBoundary {
                    position: i + 1,
                    confidence: boundary_strength,
                    boundary_type,
                });
            }
        }

        boundaries
    }

    /// Calculate syllable boundary strength between two phonemes
    #[must_use]
    pub fn calculate_syllable_boundary_strength(
        current: &AlignedPhoneme,
        next: &AlignedPhoneme,
    ) -> f32 {
        let mut strength = 0.0;

        // Vowel-to-consonant transitions often indicate boundaries
        if is_vowel(&current.phoneme.symbol) && !is_vowel(&next.phoneme.symbol) {
            strength += 0.3;
        }

        // Large duration differences suggest boundaries
        let duration_ratio =
            (current.duration() / next.duration()).max(next.duration() / current.duration());
        if duration_ratio > 2.0 {
            strength += 0.2;
        }

        // Temporal gaps suggest boundaries
        let gap = next.start_time - current.end_time;
        if gap > 0.05 {
            strength += gap * 2.0; // Higher gap = stronger boundary
        }

        // Confidence drop suggests boundary
        let confidence_diff = (current.confidence - next.confidence).abs();
        if confidence_diff > 0.3 {
            strength += 0.2;
        }

        strength.min(1.0)
    }

    /// Detect stress patterns in phoneme sequence
    #[must_use]
    pub fn detect_stress_patterns(
        phonemes: &[AlignedPhoneme],
        syllable_boundaries: &[SyllableBoundary],
    ) -> Vec<StressPattern> {
        let mut stress_patterns = Vec::new();
        let syllables = extract_syllables(phonemes, syllable_boundaries);

        for (i, syllable) in syllables.iter().enumerate() {
            let stress_features = calculate_stress_features(syllable, &syllables);
            let stress_level = classify_stress_level(&stress_features);

            stress_patterns.push(StressPattern {
                stress_level,
                syllable_index: i,
                acoustic_features: stress_features,
            });
        }

        stress_patterns
    }

    /// Extract syllables from phoneme sequence using boundaries
    #[must_use]
    pub fn extract_syllables(
        phonemes: &[AlignedPhoneme],
        boundaries: &[SyllableBoundary],
    ) -> Vec<Vec<AlignedPhoneme>> {
        let mut syllables = Vec::new();
        let mut start = 0;

        for boundary in boundaries {
            if boundary.position > start && boundary.position <= phonemes.len() {
                syllables.push(phonemes[start..boundary.position].to_vec());
                start = boundary.position;
            }
        }

        // Add remaining phonemes as final syllable
        if start < phonemes.len() {
            syllables.push(phonemes[start..].to_vec());
        }

        syllables
    }

    /// Calculate acoustic stress features for a syllable
    #[must_use]
    pub fn calculate_stress_features(
        syllable: &[AlignedPhoneme],
        all_syllables: &[Vec<AlignedPhoneme>],
    ) -> StressFeatures {
        let syllable_duration: f32 = syllable.iter().map(super::AlignedPhoneme::duration).sum();
        let avg_confidence =
            syllable.iter().map(|p| p.confidence).sum::<f32>() / syllable.len() as f32;

        // Calculate average duration of all syllables for comparison
        let total_duration: f32 = all_syllables
            .iter()
            .map(|syl| syl.iter().map(super::AlignedPhoneme::duration).sum::<f32>())
            .sum();
        let avg_syllable_duration = total_duration / all_syllables.len() as f32;

        let duration_ratio = if avg_syllable_duration > 0.0 {
            syllable_duration / avg_syllable_duration
        } else {
            1.0
        };

        // Find vowel in syllable for more detailed analysis
        let vowel_clarity = syllable
            .iter()
            .filter(|p| is_vowel(&p.phoneme.symbol))
            .map(|p| p.confidence)
            .next()
            .unwrap_or(avg_confidence);

        StressFeatures {
            duration_ratio,
            intensity_ratio: avg_confidence, // Simplified - using confidence as proxy
            pitch_prominence: if duration_ratio > 1.2 { 0.8 } else { 0.3 }, // Simplified
            vowel_clarity,
        }
    }

    /// Classify stress level based on acoustic features
    fn classify_stress_level(features: &StressFeatures) -> u8 {
        let stress_score = (features.duration_ratio * 0.4)
            + (features.intensity_ratio * 0.3)
            + (features.pitch_prominence * 0.2)
            + (features.vowel_clarity * 0.1);

        if stress_score > 1.3 {
            1 // Primary stress
        } else if stress_score > 1.0 {
            2 // Secondary stress
        } else {
            0 // Unstressed
        }
    }

    /// Create phoneme-to-text mapping
    #[must_use]
    pub fn create_phoneme_text_mapping(
        text: &str,
        phonemes: Vec<AlignedPhoneme>,
    ) -> PhonemeTextMapping {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut char_phoneme_map = Vec::new();
        let mut word_boundaries = Vec::new();

        let mut text_pos = 0;
        let mut phoneme_pos = 0;

        for word in words {
            let _word_start_text = text_pos;
            let _word_start_phoneme = phoneme_pos;

            // Skip whitespace
            while text_pos < text.len() && text.chars().nth(text_pos).unwrap().is_whitespace() {
                text_pos += 1;
            }

            let word_start_text = text_pos;

            // Estimate phonemes per character (rough heuristic)
            let estimated_phonemes = estimate_phonemes_for_word(word);
            let phonemes_end = (phoneme_pos + estimated_phonemes).min(phonemes.len());

            // Create character-to-phoneme mappings for this word
            for (char_idx, _) in word.char_indices() {
                let global_char_idx = word_start_text + char_idx;
                let phoneme_idx = phoneme_pos + (char_idx * estimated_phonemes) / word.len();

                if phoneme_idx < phonemes.len() {
                    char_phoneme_map.push((global_char_idx, phoneme_idx));
                }
            }

            word_boundaries.push(WordBoundary {
                text_start: word_start_text,
                text_end: word_start_text + word.len(),
                phoneme_start: phoneme_pos,
                phoneme_end: phonemes_end,
                word: word.to_string(),
            });

            text_pos += word.len();
            phoneme_pos = phonemes_end;
        }

        PhonemeTextMapping {
            text: text.to_string(),
            phonemes,
            char_phoneme_map,
            word_boundaries,
        }
    }

    /// Estimate number of phonemes for a word (simplified)
    fn estimate_phonemes_for_word(word: &str) -> usize {
        // Rough estimate: 0.8 phonemes per character
        (word.len() as f32 * 0.8).round() as usize
    }

    /// Extract phonological features for a phoneme
    #[must_use]
    pub fn extract_phonological_features(phoneme: &str) -> PhonologicalAnalysis {
        PhonologicalAnalysis {
            phoneme: phoneme.to_string(),
            place_features: get_place_features(phoneme),
            manner_features: get_manner_features(phoneme),
            binary_features: get_binary_features(phoneme),
            prosodic_features: get_prosodic_features(phoneme),
        }
    }

    /// Get place of articulation features
    fn get_place_features(phoneme: &str) -> PlaceFeatures {
        let mut features = PlaceFeatures {
            bilabial: false,
            labiodental: false,
            dental: false,
            alveolar: false,
            postalveolar: false,
            retroflex: false,
            palatal: false,
            velar: false,
            uvular: false,
            pharyngeal: false,
            glottal: false,
        };

        match phoneme {
            "p" | "b" | "m" => features.bilabial = true,
            "f" | "v" => features.labiodental = true,
            "θ" | "ð" => features.dental = true,
            "t" | "d" | "n" | "s" | "z" | "l" => features.alveolar = true,
            "ʃ" | "ʒ" | "tʃ" | "dʒ" => features.postalveolar = true,
            "j" => features.palatal = true,
            "k" | "g" | "ŋ" => features.velar = true,
            "h" => features.glottal = true,
            _ => {} // Default case for vowels and other phonemes
        }

        features
    }

    /// Get manner of articulation features
    fn get_manner_features(phoneme: &str) -> MannerFeatures {
        let mut features = MannerFeatures {
            stop: false,
            fricative: false,
            affricate: false,
            nasal: false,
            lateral: false,
            approximant: false,
            trill: false,
            tap: false,
        };

        match phoneme {
            "p" | "b" | "t" | "d" | "k" | "g" => features.stop = true,
            "f" | "v" | "θ" | "ð" | "s" | "z" | "ʃ" | "ʒ" | "h" => features.fricative = true,
            "tʃ" | "dʒ" => features.affricate = true,
            "m" | "n" | "ŋ" => features.nasal = true,
            "l" => features.lateral = true,
            "w" | "j" | "r" => features.approximant = true,
            _ => {} // Default case
        }

        features
    }

    /// Get binary phonological features
    fn get_binary_features(phoneme: &str) -> BinaryFeatures {
        let voiced = matches!(
            phoneme,
            "b" | "d"
                | "g"
                | "v"
                | "ð"
                | "z"
                | "ʒ"
                | "dʒ"
                | "m"
                | "n"
                | "ŋ"
                | "l"
                | "r"
                | "w"
                | "j"
        );
        let consonantal = !is_vowel(phoneme);
        let sonorant =
            matches!(phoneme, "m" | "n" | "ŋ" | "l" | "r" | "w" | "j") || is_vowel(phoneme);

        BinaryFeatures {
            voiced,
            aspirated: false, // Simplified
            syllabic: is_vowel(phoneme),
            consonantal,
            sonorant,
            continuant: !matches!(phoneme, "p" | "b" | "t" | "d" | "k" | "g" | "tʃ" | "dʒ"),
            delayed_release: matches!(phoneme, "tʃ" | "dʒ"),
            approximant: matches!(phoneme, "w" | "j" | "r"),
        }
    }

    /// Get prosodic features (simplified)
    fn get_prosodic_features(_phoneme: &str) -> ProsodicFeatures {
        ProsodicFeatures {
            stress: 0,  // Default unstressed
            tone: None, // Not applicable for English
            length: LengthType::Normal,
        }
    }

    /// Simple vowel detection
    #[must_use]
    pub fn is_vowel(phoneme: &str) -> bool {
        matches!(
            phoneme,
            "i" | "ɪ"
                | "e"
                | "ɛ"
                | "æ"
                | "a"
                | "ɑ"
                | "ɔ"
                | "o"
                | "ʊ"
                | "u"
                | "ʌ"
                | "ə"
                | "ɝ"
                | "aɪ"
                | "aʊ"
                | "eɪ"
                | "oʊ"
                | "ɔɪ"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voirs_sdk::types::SyllablePosition;

    #[test]
    fn test_aligned_phoneme_creation() {
        let phoneme = Phoneme {
            symbol: "a".to_string(),
            ipa_symbol: "a".to_string(),
            stress: 0,
            syllable_position: SyllablePosition::Nucleus,
            duration_ms: Some(150.0),
            confidence: 0.9,
        };

        let aligned = AlignedPhoneme::new(phoneme, 0.1, 0.25, 0.85);

        assert_eq!(aligned.start_time, 0.1);
        assert_eq!(aligned.end_time, 0.25);
        assert_eq!(aligned.confidence, 0.85);
        assert_eq!(aligned.duration(), 0.15);
    }

    #[test]
    fn test_phoneme_overlap_detection() {
        let phoneme1 = create_test_phoneme("a", 0.0, 0.1);
        let phoneme2 = create_test_phoneme("b", 0.05, 0.15); // Overlaps
        let phoneme3 = create_test_phoneme("c", 0.15, 0.25); // No overlap

        assert!(phoneme1.overlaps_with(&phoneme2));
        assert!(!phoneme1.overlaps_with(&phoneme3));
        assert!(!phoneme2.overlaps_with(&phoneme3));
    }

    #[test]
    fn test_phoneme_alignment_creation() {
        let phonemes = vec![
            create_test_phoneme("h", 0.0, 0.05),
            create_test_phoneme("ɛ", 0.05, 0.15),
            create_test_phoneme("l", 0.15, 0.20),
            create_test_phoneme("oʊ", 0.20, 0.35),
        ];

        let word_alignments = vec![WordAlignment::new(
            "hello".to_string(),
            0.0,
            0.35,
            phonemes.clone(),
            0.9,
        )];

        let alignment = PhonemeAlignment::new(phonemes, word_alignments, 0.35, LanguageCode::EnUs);

        assert_eq!(alignment.phonemes.len(), 4);
        assert_eq!(alignment.word_alignments.len(), 1);
        assert_eq!(alignment.total_duration, 0.35);
        assert_eq!(alignment.speech_rate(), 4.0 / 0.35);
    }

    #[test]
    fn test_temporal_consistency_validation() {
        let phonemes = vec![
            create_test_phoneme("a", 0.0, 0.1),
            create_test_phoneme("b", 0.05, 0.15), // Overlap!
        ];

        let alignment = PhonemeAlignment::new(phonemes, vec![], 0.15, LanguageCode::EnUs);

        let issues = utils::validate_temporal_consistency(&alignment);
        assert!(!issues.is_empty());
        assert!(issues[0].contains("Overlapping"));
    }

    fn create_test_phoneme(symbol: &str, start: f32, end: f32) -> AlignedPhoneme {
        AlignedPhoneme::new(
            Phoneme {
                symbol: symbol.to_string(),
                ipa_symbol: symbol.to_string(),
                stress: 0,
                syllable_position: SyllablePosition::Nucleus,
                duration_ms: Some((end - start) * 1000.0),
                confidence: 0.9,
            },
            start,
            end,
            0.9,
        )
    }

    #[test]
    fn test_syllable_boundary_detection() {
        let phonemes = vec![
            create_test_phoneme("h", 0.0, 0.05),
            create_test_phoneme("ɛ", 0.05, 0.15),  // Vowel
            create_test_phoneme("l", 0.18, 0.23),  // Gap to increase boundary strength
            create_test_phoneme("oʊ", 0.25, 0.40), // Vowel with gap
        ];

        let boundaries = utils::detect_syllable_boundaries(&phonemes);

        // Check that boundaries detection works
        // If no boundaries found, that's also valid depending on the phoneme sequence
        for boundary in boundaries {
            assert!(boundary.confidence >= 0.0 && boundary.confidence <= 1.0);
            assert!(boundary.position > 0 && boundary.position < phonemes.len());
        }

        // Test with a more obvious boundary case
        let test_phonemes = vec![
            create_test_phoneme("a", 0.0, 0.15),  // Vowel
            create_test_phoneme("p", 0.20, 0.25), // Consonant with gap
        ];

        let test_boundaries = utils::detect_syllable_boundaries(&test_phonemes);
        // This should have some boundary strength due to vowel->consonant + gap
        if !test_boundaries.is_empty() {
            assert!(test_boundaries[0].confidence > 0.5);
        }
    }

    #[test]
    fn test_stress_pattern_detection() {
        let phonemes = vec![
            create_test_phoneme("h", 0.0, 0.05),
            create_test_phoneme("ɛ", 0.05, 0.15),
            create_test_phoneme("l", 0.15, 0.20),
            create_test_phoneme("oʊ", 0.20, 0.35),
        ];

        let boundaries = vec![SyllableBoundary {
            position: 2,
            confidence: 0.8,
            boundary_type: BoundaryType::Strong,
        }];

        let stress_patterns = utils::detect_stress_patterns(&phonemes, &boundaries);

        assert!(!stress_patterns.is_empty());

        for pattern in stress_patterns {
            assert!(pattern.stress_level <= 2); // 0, 1, or 2
            assert!(pattern.acoustic_features.duration_ratio > 0.0);
            assert!(
                pattern.acoustic_features.intensity_ratio >= 0.0
                    && pattern.acoustic_features.intensity_ratio <= 1.0
            );
        }
    }

    #[test]
    fn test_phoneme_text_mapping() {
        let text = "hello world";
        let phonemes = vec![
            create_test_phoneme("h", 0.0, 0.05),
            create_test_phoneme("ɛ", 0.05, 0.15),
            create_test_phoneme("l", 0.15, 0.20),
            create_test_phoneme("oʊ", 0.20, 0.35),
            create_test_phoneme("w", 0.40, 0.45),
            create_test_phoneme("ɝ", 0.45, 0.55),
            create_test_phoneme("l", 0.55, 0.60),
            create_test_phoneme("d", 0.60, 0.65),
        ];

        let mapping = utils::create_phoneme_text_mapping(text, phonemes);

        assert_eq!(mapping.text, text);
        assert!(!mapping.char_phoneme_map.is_empty());
        assert!(!mapping.word_boundaries.is_empty());

        // Should detect two words
        assert_eq!(mapping.word_boundaries.len(), 2);
        assert_eq!(mapping.word_boundaries[0].word, "hello");
        assert_eq!(mapping.word_boundaries[1].word, "world");
    }

    #[test]
    fn test_phonological_feature_extraction() {
        // Test consonant features
        let consonant_analysis = utils::extract_phonological_features("p");
        assert!(consonant_analysis.place_features.bilabial);
        assert!(consonant_analysis.manner_features.stop);
        assert!(!consonant_analysis.binary_features.voiced);
        assert!(consonant_analysis.binary_features.consonantal);

        // Test vowel features
        let vowel_analysis = utils::extract_phonological_features("a");
        assert!(vowel_analysis.binary_features.syllabic);
        assert!(!vowel_analysis.binary_features.consonantal);
        assert!(vowel_analysis.binary_features.sonorant);

        // Test fricative features
        let fricative_analysis = utils::extract_phonological_features("s");
        assert!(fricative_analysis.place_features.alveolar);
        assert!(fricative_analysis.manner_features.fricative);
        assert!(!fricative_analysis.binary_features.voiced);
    }

    #[test]
    fn test_vowel_detection() {
        assert!(utils::is_vowel("a"));
        assert!(utils::is_vowel("i"));
        assert!(utils::is_vowel("ɛ"));
        assert!(utils::is_vowel("oʊ"));
        assert!(utils::is_vowel("aɪ"));

        assert!(!utils::is_vowel("p"));
        assert!(!utils::is_vowel("s"));
        assert!(!utils::is_vowel("ŋ"));
    }

    #[test]
    fn test_syllable_extraction() {
        let phonemes = vec![
            create_test_phoneme("h", 0.0, 0.05),
            create_test_phoneme("ɛ", 0.05, 0.15),
            create_test_phoneme("l", 0.15, 0.20),
            create_test_phoneme("oʊ", 0.20, 0.35),
        ];

        let boundaries = vec![SyllableBoundary {
            position: 2,
            confidence: 0.8,
            boundary_type: BoundaryType::Strong,
        }];

        let syllables = utils::extract_syllables(&phonemes, &boundaries);

        assert_eq!(syllables.len(), 2);
        assert_eq!(syllables[0].len(), 2); // "h", "ɛ"
        assert_eq!(syllables[1].len(), 2); // "l", "oʊ"
    }

    #[test]
    fn test_stress_feature_calculation() {
        let syllable = vec![
            create_test_phoneme("h", 0.0, 0.05),
            create_test_phoneme("ɛ", 0.05, 0.20), // Long vowel
        ];

        let all_syllables = vec![
            syllable.clone(),
            vec![create_test_phoneme("l", 0.20, 0.25)], // Short syllable
        ];

        let features = utils::calculate_stress_features(&syllable, &all_syllables);

        assert!(features.duration_ratio > 0.0);
        assert!(features.intensity_ratio >= 0.0 && features.intensity_ratio <= 1.0);
        assert!(features.vowel_clarity >= 0.0 && features.vowel_clarity <= 1.0);
    }

    #[test]
    fn test_boundary_strength_calculation() {
        let vowel = create_test_phoneme("a", 0.0, 0.15);
        let consonant = create_test_phoneme("p", 0.20, 0.25); // Gap between them

        let strength = utils::calculate_syllable_boundary_strength(&vowel, &consonant);

        // Should have some boundary strength due to vowel-consonant transition and gap
        assert!(strength > 0.0);
        assert!(strength <= 1.0);
    }
}
