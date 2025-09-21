//! Quality analysis and scoring utilities for G2P conversion.

use super::phoneme_analysis::validate_phonemes;
use super::phoneme_analysis::{analyze_phoneme_sequence, is_vowel, PhonemeAnalysis};
use crate::{LanguageCode, Phoneme, Result};
use std::collections::HashSet;

/// Advanced phoneme quality scoring (optimized for performance)
pub fn score_phoneme_quality(phonemes: &[Phoneme], language: LanguageCode) -> PhonemeQualityScore {
    let mut score = PhonemeQualityScore::default();

    if phonemes.is_empty() {
        return score;
    }

    // Single-pass optimization: calculate confidence and distinctiveness together
    let mut confidence_sum = 0.0f32;
    let mut unique_phonemes = HashSet::with_capacity(phonemes.len());
    let mut low_confidence_count = 0;

    for phoneme in phonemes {
        confidence_sum += phoneme.confidence;
        unique_phonemes.insert(phoneme.effective_symbol());

        if phoneme.confidence < 0.6 {
            low_confidence_count += 1;
        }
    }

    score.average_confidence = confidence_sum / phonemes.len() as f32;
    score.distinctiveness_score = (unique_phonemes.len() as f32 / phonemes.len() as f32).min(1.0);

    // Calculate naturalness score based on phoneme transitions
    score.naturalness_score = calculate_naturalness_score(phonemes, language);

    // Calculate overall quality score
    score.overall_score = (score.average_confidence * 0.4)
        + (score.naturalness_score * 0.35)
        + (score.distinctiveness_score * 0.25);

    // Add quality factors (optimized to use pre-calculated low_confidence_count)
    score.quality_factors =
        identify_quality_factors_optimized(phonemes, language, low_confidence_count);

    score
}

/// Advanced quality metrics for comprehensive phoneme analysis
#[derive(Debug, Clone, Default)]
pub struct AdvancedQualityMetrics {
    /// Phonological validity score (0.0 to 1.0)
    pub phonological_validity: f32,
    /// Articulatory consistency score (0.0 to 1.0)
    pub articulatory_consistency: f32,
    /// Perceptual quality score (0.0 to 1.0)
    pub perceptual_quality: f32,
    /// Rhythmic pattern score (0.0 to 1.0)
    pub rhythmic_pattern: f32,
    /// Cross-linguistic consistency (0.0 to 1.0)
    pub cross_linguistic_consistency: f32,
    /// Machine learning confidence (0.0 to 1.0)
    pub ml_confidence: f32,
    /// Detailed quality factors
    pub detailed_factors: Vec<DetailedQualityFactor>,
}

/// Detailed quality factor with specific recommendations
#[derive(Debug, Clone)]
pub struct DetailedQualityFactor {
    pub factor_type: QualityFactorType,
    pub severity: QualitySeverity,
    pub description: String,
    pub recommendation: String,
    pub confidence: f32,
    pub affected_phonemes: Vec<usize>, // indices of affected phonemes
}

/// Types of quality factors
#[derive(Debug, Clone, PartialEq)]
pub enum QualityFactorType {
    LowConfidence,
    UncommonTransition,
    PhonologicalViolation,
    ArticulatoryInconsistency,
    RhythmicIrregularity,
    PerceptualIssue,
    CrossLinguisticConflict,
}

/// Severity levels for quality issues
#[derive(Debug, Clone, PartialEq)]
pub enum QualitySeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Advanced phoneme quality analyzer
pub struct AdvancedQualityAnalyzer {
    language_models: std::collections::HashMap<LanguageCode, LanguagePhonologyModel>,
    transition_weights: std::collections::HashMap<(String, String), f32>,
    perceptual_weights: std::collections::HashMap<String, f32>,
}

/// Language-specific phonology model
#[derive(Debug, Clone)]
pub struct LanguagePhonologyModel {
    pub valid_phonemes: HashSet<String>,
    pub common_transitions: std::collections::HashMap<(String, String), f32>,
    pub stress_patterns: Vec<Vec<u8>>,
    pub syllable_structures: Vec<String>,
    pub coarticulation_rules: Vec<CoarticulationRule>,
}

/// Coarticulation rule for articulatory consistency
#[derive(Debug, Clone)]
pub struct CoarticulationRule {
    pub context: String,
    pub target: String,
    pub modification: ArticulatoryModification,
    pub strength: f32,
}

/// Articulatory modification types
#[derive(Debug, Clone)]
pub enum ArticulatoryModification {
    VoicingAssimilation,
    PlaceAssimilation,
    MannerAssimilation,
    Palatalization,
    Labialization,
    Nasalization,
}

impl AdvancedQualityAnalyzer {
    /// Create a new advanced quality analyzer
    pub fn new() -> Self {
        let mut analyzer = Self {
            language_models: std::collections::HashMap::new(),
            transition_weights: std::collections::HashMap::new(),
            perceptual_weights: std::collections::HashMap::new(),
        };

        analyzer.initialize_language_models();
        analyzer.initialize_transition_weights();
        analyzer.initialize_perceptual_weights();

        analyzer
    }

    /// Analyze phoneme quality with advanced metrics
    pub fn analyze_quality(
        &self,
        phonemes: &[Phoneme],
        language: LanguageCode,
        context: Option<&str>,
    ) -> AdvancedQualityMetrics {
        let mut metrics = AdvancedQualityMetrics::default();

        if phonemes.is_empty() {
            return metrics;
        }

        // Calculate phonological validity
        metrics.phonological_validity = self.calculate_phonological_validity(phonemes, language);

        // Calculate articulatory consistency
        metrics.articulatory_consistency =
            self.calculate_articulatory_consistency(phonemes, language);

        // Calculate perceptual quality
        metrics.perceptual_quality = self.calculate_perceptual_quality(phonemes, language);

        // Calculate rhythmic pattern score
        metrics.rhythmic_pattern = self.calculate_rhythmic_pattern(phonemes);

        // Calculate cross-linguistic consistency
        metrics.cross_linguistic_consistency =
            self.calculate_cross_linguistic_consistency(phonemes);

        // Calculate ML confidence
        metrics.ml_confidence = self.calculate_ml_confidence(phonemes, context);

        // Generate detailed quality factors
        metrics.detailed_factors = self.generate_detailed_factors(phonemes, language, &metrics);

        metrics
    }

    fn initialize_language_models(&mut self) {
        // Initialize English phonology model
        let mut en_valid_phonemes = HashSet::new();
        for phoneme in &[
            "AH", "AA", "AE", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH",
            "UW", "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R",
            "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH",
        ] {
            en_valid_phonemes.insert(phoneme.to_string());
        }

        let en_model = LanguagePhonologyModel {
            valid_phonemes: en_valid_phonemes,
            common_transitions: std::collections::HashMap::new(),
            stress_patterns: vec![vec![1, 0], vec![0, 1], vec![1, 0, 0]],
            syllable_structures: vec!["CV".to_string(), "CVC".to_string(), "CCVC".to_string()],
            coarticulation_rules: vec![],
        };

        self.language_models.insert(LanguageCode::EnUs, en_model);
    }

    fn initialize_transition_weights(&mut self) {
        // Common phoneme transitions with natural weights
        let transitions = vec![
            (("T", "AH"), 0.9),
            (("D", "AH"), 0.9),
            (("S", "AH"), 0.8),
            (("N", "AH"), 0.8),
            (("AH", "N"), 0.9),
            (("AH", "T"), 0.8),
            (("IY", "N"), 0.9),
            (("AE", "T"), 0.8),
        ];

        for ((first, second), weight) in transitions {
            self.transition_weights
                .insert((first.to_string(), second.to_string()), weight);
        }
    }

    fn initialize_perceptual_weights(&mut self) {
        // Perceptual importance weights for different phonemes
        let weights = vec![
            ("AH", 0.7),
            ("AE", 0.8),
            ("IY", 0.9),
            ("UW", 0.9),
            ("T", 0.8),
            ("D", 0.8),
            ("S", 0.9),
            ("Z", 0.9),
            ("M", 0.8),
            ("N", 0.8),
            ("L", 0.7),
            ("R", 0.9),
        ];

        for (phoneme, weight) in weights {
            self.perceptual_weights.insert(phoneme.to_string(), weight);
        }
    }

    fn calculate_phonological_validity(&self, phonemes: &[Phoneme], language: LanguageCode) -> f32 {
        if let Some(model) = self.language_models.get(&language) {
            let valid_count = phonemes
                .iter()
                .filter(|p| model.valid_phonemes.contains(p.effective_symbol()))
                .count();
            valid_count as f32 / phonemes.len() as f32
        } else {
            0.5 // Default for unknown languages
        }
    }

    fn calculate_articulatory_consistency(
        &self,
        phonemes: &[Phoneme],
        _language: LanguageCode,
    ) -> f32 {
        if phonemes.len() < 2 {
            return 1.0;
        }

        let mut consistency_sum = 0.0;
        let mut transition_count = 0;

        for window in phonemes.windows(2) {
            let first = window[0].effective_symbol();
            let second = window[1].effective_symbol();

            let consistency = self
                .transition_weights
                .get(&(first.to_string(), second.to_string()))
                .copied()
                .unwrap_or(0.5); // Default consistency for unknown transitions

            consistency_sum += consistency;
            transition_count += 1;
        }

        if transition_count > 0 {
            consistency_sum / transition_count as f32
        } else {
            1.0
        }
    }

    fn calculate_perceptual_quality(&self, phonemes: &[Phoneme], _language: LanguageCode) -> f32 {
        let mut quality_sum = 0.0;
        let mut phoneme_count = 0;

        for phoneme in phonemes {
            let base_weight = self
                .perceptual_weights
                .get(phoneme.effective_symbol())
                .copied()
                .unwrap_or(0.6);

            let confidence_adjusted = base_weight * phoneme.confidence;
            quality_sum += confidence_adjusted;
            phoneme_count += 1;
        }

        if phoneme_count > 0 {
            quality_sum / phoneme_count as f32
        } else {
            0.0
        }
    }

    fn calculate_rhythmic_pattern(&self, phonemes: &[Phoneme]) -> f32 {
        if phonemes.len() < 3 {
            return 1.0;
        }

        // Analyze stress patterns and syllable boundaries
        let stress_pattern: Vec<u8> = phonemes.iter().map(|p| p.stress).collect();
        let mut pattern_score = 0.0;
        let mut pattern_count = 0;

        // Check for regular stress patterns
        for window in stress_pattern.windows(3) {
            let is_regular = (window[0] != window[1]) || (window[1] != window[2]);
            if is_regular {
                pattern_score += 1.0;
            }
            pattern_count += 1;
        }

        if pattern_count > 0 {
            pattern_score / pattern_count as f32
        } else {
            1.0
        }
    }

    fn calculate_cross_linguistic_consistency(&self, phonemes: &[Phoneme]) -> f32 {
        // Check for phonemes that are consistent across languages
        let universal_phonemes = ["AH", "IY", "UW", "M", "N", "T", "K", "S"];
        let universal_count = phonemes
            .iter()
            .filter(|p| {
                universal_phonemes
                    .iter()
                    .any(|&up| up == p.effective_symbol())
            })
            .count();

        if phonemes.is_empty() {
            1.0
        } else {
            (universal_count as f32 / phonemes.len() as f32) * 0.5 + 0.5
        }
    }

    fn calculate_ml_confidence(&self, phonemes: &[Phoneme], _context: Option<&str>) -> f32 {
        if phonemes.is_empty() {
            return 0.0;
        }

        let confidence_sum: f32 = phonemes.iter().map(|p| p.confidence).sum();

        // Apply contextual adjustments if available
        confidence_sum / phonemes.len() as f32
    }

    fn generate_detailed_factors(
        &self,
        phonemes: &[Phoneme],
        language: LanguageCode,
        metrics: &AdvancedQualityMetrics,
    ) -> Vec<DetailedQualityFactor> {
        let mut factors = Vec::new();

        // Check for low confidence phonemes
        for (i, phoneme) in phonemes.iter().enumerate() {
            if phoneme.confidence < 0.6 {
                factors.push(DetailedQualityFactor {
                    factor_type: QualityFactorType::LowConfidence,
                    severity: if phoneme.confidence < 0.3 {
                        QualitySeverity::Critical
                    } else {
                        QualitySeverity::Medium
                    },
                    description: format!(
                        "Low confidence phoneme: {} ({})",
                        phoneme.effective_symbol(),
                        phoneme.confidence
                    ),
                    recommendation: "Consider re-processing this segment with alternative models"
                        .to_string(),
                    confidence: 1.0 - phoneme.confidence,
                    affected_phonemes: vec![i],
                });
            }
        }

        // Check for phonological violations
        if metrics.phonological_validity < 0.8 {
            factors.push(DetailedQualityFactor {
                factor_type: QualityFactorType::PhonologicalViolation,
                severity: QualitySeverity::High,
                description: format!(
                    "Phonological validity: {:.2}",
                    metrics.phonological_validity
                ),
                recommendation: "Review phoneme inventory for target language".to_string(),
                confidence: 1.0 - metrics.phonological_validity,
                affected_phonemes: (0..phonemes.len()).collect(),
            });
        }

        factors
    }
}

impl Default for AdvancedQualityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Phoneme quality score (legacy interface)
#[derive(Debug, Clone, Default)]
pub struct PhonemeQualityScore {
    pub average_confidence: f32,
    pub naturalness_score: f32,
    pub distinctiveness_score: f32,
    pub overall_score: f32,
    pub quality_factors: Vec<String>,
}

/// Validation report for phoneme sequences
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<String>,
}

/// Validation error types
#[derive(Debug, Clone)]
pub enum ValidationError {
    InvalidPhoneme {
        position: usize,
        symbol: String,
        message: String,
    },
    PhonotacticViolation {
        phonemes: Vec<String>,
        message: String,
    },
}

/// Text segment for streaming processing
#[derive(Debug, Clone)]
pub struct TextSegment {
    pub text: String,
    pub start_offset: usize,
    pub end_offset: usize,
    pub sentence_count: usize,
}

/// Comprehensive debug summary for G2P conversion analysis
#[derive(Debug, Clone)]
pub struct ConversionDebugSummary {
    pub input_text: String,
    pub language: LanguageCode,
    pub phoneme_count: usize,
    pub character_count: usize,
    pub phonemes_per_character: f32,
    pub unique_phoneme_count: usize,
    pub phoneme_symbols: Vec<String>,
    pub conversion_time_ms: f64,
    pub phoneme_analysis: PhonemeAnalysis,
    pub validation_report: ValidationReport,
    pub quality_score: PhonemeQualityScore,
}

impl ConversionDebugSummary {
    /// Generate a formatted report
    pub fn format_report(&self) -> String {
        format!(
            "G2P Conversion Debug Report\n\
            ===========================\n\
            Input: '{}'\n\
            Language: {:?}\n\
            Character Count: {}\n\
            Phoneme Count: {}\n\
            Unique Phonemes: {}\n\
            Phonemes/Character: {:.2}\n\
            Conversion Time: {:.2}ms\n\
            \n\
            Quality Metrics:\n\
            - Average Confidence: {:.2}\n\
            - Naturalness Score: {:.2}\n\
            - Distinctiveness: {:.2}\n\
            - Overall Score: {:.2}\n\
            \n\
            Analysis:\n\
            - Vowels: {}\n\
            - Consonants: {}\n\
            - Syllables: {}\n\
            - Words: {}\n\
            - Total Duration: {:.2}ms\n\
            \n\
            Phoneme Sequence: {}\n\
            \n\
            Quality Factors: {:?}\n\
            Validation: Valid: {}, Errors: {}, Warnings: {}\n",
            self.input_text,
            self.language,
            self.character_count,
            self.phoneme_count,
            self.unique_phoneme_count,
            self.phonemes_per_character,
            self.conversion_time_ms,
            self.quality_score.average_confidence,
            self.quality_score.naturalness_score,
            self.quality_score.distinctiveness_score,
            self.quality_score.overall_score,
            self.phoneme_analysis.vowel_count,
            self.phoneme_analysis.consonant_count,
            self.phoneme_analysis.syllable_count,
            self.phoneme_analysis.word_count,
            self.phoneme_analysis.total_duration_ms,
            self.phoneme_symbols.join(" "),
            self.quality_score.quality_factors,
            self.validation_report.is_valid,
            self.validation_report.errors.len(),
            self.validation_report.warnings.len()
        )
    }
}

/// Generate a comprehensive debug summary for G2P conversion results
///
/// This function provides detailed analysis and statistics for debugging
/// and performance monitoring of G2P conversions.
pub fn generate_conversion_debug_summary(
    input_text: &str,
    phonemes: &[Phoneme],
    language: LanguageCode,
    conversion_time_ms: f64,
) -> ConversionDebugSummary {
    let phoneme_analysis = analyze_phoneme_sequence(phonemes);
    let validation_report =
        validate_phoneme_sequence_advanced(phonemes, language).unwrap_or_default();
    let quality_score = score_phoneme_quality(phonemes, language);

    // Calculate additional statistics
    let phoneme_count = phonemes.len();
    let character_count = input_text.len();
    let phonemes_per_character = if character_count > 0 {
        phoneme_count as f32 / character_count as f32
    } else {
        0.0
    };

    // Extract phoneme symbols for analysis
    let phoneme_symbols: Vec<String> = phonemes
        .iter()
        .map(|p| p.effective_symbol().to_string())
        .collect();

    // Count unique phonemes
    let mut unique_phonemes = HashSet::new();
    for symbol in &phoneme_symbols {
        unique_phonemes.insert(symbol);
    }

    ConversionDebugSummary {
        input_text: input_text.to_string(),
        language,
        phoneme_count,
        character_count,
        phonemes_per_character,
        unique_phoneme_count: unique_phonemes.len(),
        phoneme_symbols,
        conversion_time_ms,
        phoneme_analysis,
        validation_report,
        quality_score,
    }
}

/// Advanced phoneme sequence validation
pub fn validate_phoneme_sequence_advanced(
    phonemes: &[Phoneme],
    language: LanguageCode,
) -> Result<ValidationReport> {
    let mut report = ValidationReport {
        is_valid: true,
        errors: Vec::new(),
        warnings: Vec::new(),
    };

    // Basic validation using existing function
    if !validate_phonemes(phonemes, language) {
        report.is_valid = false;
        report.errors.push(ValidationError::PhonotacticViolation {
            phonemes: phonemes
                .iter()
                .map(|p| p.effective_symbol().to_string())
                .collect(),
            message: "Phonotactic constraints violated".to_string(),
        });
    }

    // Additional quality checks
    for (i, phoneme) in phonemes.iter().enumerate() {
        if phoneme.confidence < 0.3 {
            report.warnings.push(format!(
                "Very low confidence ({:.2}) for phoneme '{}' at position {}",
                phoneme.confidence,
                phoneme.effective_symbol(),
                i
            ));
        }
    }

    Ok(report)
}

/// Text segmentation for streaming G2P processing
pub fn segment_text_for_streaming(
    text: &str,
    max_chars_per_segment: usize,
) -> Result<Vec<TextSegment>> {
    let mut segments = Vec::new();
    let mut current_start = 0;

    // Split text into sentences first
    let sentences = segment_into_sentences(text)?;
    let mut current_text = String::new();
    let mut sentence_count = 0;

    for sentence in sentences {
        // Check if adding this sentence would exceed the limit
        if current_text.len() + sentence.len() > max_chars_per_segment && !current_text.is_empty() {
            // Create segment with current content
            let segment = TextSegment {
                text: current_text.clone(),
                start_offset: current_start,
                end_offset: current_start + current_text.len(),
                sentence_count,
            };
            segments.push(segment);

            // Reset for next segment
            current_start += current_text.len();
            current_text.clear();
            sentence_count = 0;
        }

        // Add sentence to current segment
        if !current_text.is_empty() {
            current_text.push(' ');
        }
        current_text.push_str(&sentence);
        sentence_count += 1;
    }

    // Add final segment if any text remains
    if !current_text.is_empty() {
        let segment = TextSegment {
            text: current_text.clone(),
            start_offset: current_start,
            end_offset: current_start + current_text.len(),
            sentence_count,
        };
        segments.push(segment);
    }

    Ok(segments)
}

/// Split text into sentences
pub fn segment_into_sentences(text: &str) -> Result<Vec<String>> {
    let mut sentences = Vec::new();
    let mut current_sentence = String::new();

    // Define sentence ending characters
    let sentence_endings = ['.', '!', '?', '。', '！', '？'];

    for ch in text.chars() {
        current_sentence.push(ch);

        if sentence_endings.contains(&ch) {
            sentences.push(current_sentence.trim().to_string());
            current_sentence.clear();
        }
    }

    // Add remaining text as final sentence
    if !current_sentence.trim().is_empty() {
        sentences.push(current_sentence.trim().to_string());
    }

    Ok(sentences)
}

// Private helper functions

/// Calculate naturalness based on phoneme transitions
fn calculate_naturalness_score(phonemes: &[Phoneme], language: LanguageCode) -> f32 {
    if phonemes.len() < 2 {
        return 1.0;
    }

    let mut total_score = 0.0;
    let mut transition_count = 0;

    for window in phonemes.windows(2) {
        let (p1, p2) = (window[0].effective_symbol(), window[1].effective_symbol());
        let transition_score = get_transition_naturalness(p1, p2, language);
        total_score += transition_score;
        transition_count += 1;
    }

    if transition_count > 0 {
        total_score / transition_count as f32
    } else {
        1.0
    }
}

/// Get naturalness score for phoneme transitions
fn get_transition_naturalness(p1: &str, p2: &str, _language: LanguageCode) -> f32 {
    // Simplified naturalness scoring based on common patterns
    match (is_vowel(p1), is_vowel(p2)) {
        (true, true) => 0.7,   // Vowel-vowel transitions are less natural
        (false, false) => 0.8, // Consonant clusters vary in naturalness
        (true, false) => 0.95, // Vowel-consonant is very natural
        (false, true) => 0.9,  // Consonant-vowel is natural
    }
}

/// Optimized version of identify_quality_factors that reuses pre-calculated values
fn identify_quality_factors_optimized(
    phonemes: &[Phoneme],
    language: LanguageCode,
    low_confidence_count: usize,
) -> Vec<String> {
    let mut factors = Vec::new();

    // Use pre-calculated low confidence count
    if low_confidence_count > 0 {
        factors.push(format!("{low_confidence_count} low-confidence phonemes"));
    }

    // Check vowel-consonant balance (single pass)
    let mut vowel_count = 0;
    for phoneme in phonemes {
        if is_vowel(phoneme.effective_symbol()) {
            vowel_count += 1;
        }
    }

    let consonant_count = phonemes.len() - vowel_count;
    let ratio = if consonant_count > 0 {
        vowel_count as f32 / consonant_count as f32
    } else {
        vowel_count as f32
    };

    if ratio > 2.0 {
        factors.push("Too many vowels relative to consonants".to_string());
    } else if ratio < 0.2 {
        factors.push("Too few vowels relative to consonants".to_string());
    }

    // Language-specific factors
    if language == LanguageCode::Ja {
        // Check for proper mora structure
        if phonemes.len() % 2 != 0 {
            factors.push("Irregular mora structure for Japanese".to_string());
        }
    }

    factors
}

/// Count sentences in text
#[allow(dead_code)]
fn count_sentences(text: &str) -> usize {
    text.chars()
        .filter(|&c| matches!(c, '.' | '!' | '?' | '。' | '！' | '？'))
        .count()
}
