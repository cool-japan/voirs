//! Diagnostic and profiling utilities for G2P operations.

use super::phoneme_analysis::validate_phonemes;
use super::phoneme_analysis::{analyze_phoneme_sequence, is_consonant, is_vowel};
use super::text_processing::postprocess_phonemes;
use crate::{
    G2pDiagnosticContext, G2pError, LanguageCode, Phoneme, PhoneticFeatures, ProcessingStage,
    Result,
};
use std::time::Instant;

/// Enhanced error diagnostics for G2P conversion failures
pub fn diagnose_conversion_error(
    text: &str,
    phonemes: &[Phoneme],
    language: LanguageCode,
) -> DiagnosticReport {
    let mut report = DiagnosticReport::default();

    // Check text complexity
    if text.len() > 1000 {
        report
            .warnings
            .push("Text is very long, consider chunking".to_string());
    }

    if !text.is_ascii() {
        report
            .warnings
            .push("Non-ASCII characters detected, ensure proper language detection".to_string());
    }

    // Check phoneme quality
    let analysis = analyze_phoneme_sequence(phonemes);

    if analysis.average_confidence < 0.5 {
        report
            .errors
            .push("Low average confidence in phoneme conversion".to_string());
    }

    if analysis.vowel_count == 0 && !phonemes.is_empty() {
        report
            .errors
            .push("No vowels detected in non-empty phoneme sequence".to_string());
    }

    if analysis.vowel_consonant_ratio > 3.0 {
        report
            .warnings
            .push("Unusual vowel-to-consonant ratio detected".to_string());
    }

    // Language-specific checks
    match language {
        LanguageCode::Ja => {
            if text.chars().any(|c| c.is_ascii_alphabetic()) && !text.is_ascii() {
                report
                    .warnings
                    .push("Mixed script detected in Japanese text".to_string());
            }
        }
        LanguageCode::EnUs | LanguageCode::EnGb => {
            if !text.is_ascii() {
                report
                    .warnings
                    .push("Non-ASCII characters in English text".to_string());
            }
        }
        _ => {}
    }

    report.language = language;
    report.text_length = text.len();
    report.phoneme_count = phonemes.len();

    report
}

/// Diagnostic report for G2P conversion analysis
#[derive(Debug, Clone, Default)]
pub struct DiagnosticReport {
    pub language: LanguageCode,
    pub text_length: usize,
    pub phoneme_count: usize,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// Performance profiling utilities for G2P operations
pub struct G2pProfiler {
    start_time: Instant,
    checkpoints: Vec<(String, Instant)>,
}

impl G2pProfiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            checkpoints: Vec::new(),
        }
    }

    /// Add a checkpoint with a label
    pub fn checkpoint(&mut self, label: &str) {
        self.checkpoints.push((label.to_string(), Instant::now()));
    }

    /// Generate a performance report
    pub fn report(&self) -> PerformanceReport {
        let total_duration = self.start_time.elapsed();
        let mut stages = Vec::new();

        let mut last_time = self.start_time;
        for (label, time) in &self.checkpoints {
            stages.push(PerformanceStage {
                label: label.clone(),
                duration_ms: (time.duration_since(last_time)).as_secs_f32() * 1000.0,
            });
            last_time = *time;
        }

        PerformanceReport {
            total_duration_ms: total_duration.as_secs_f32() * 1000.0,
            stages,
        }
    }
}

impl Default for G2pProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance report for G2P operations
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub total_duration_ms: f32,
    pub stages: Vec<PerformanceStage>,
}

/// Individual performance stage
#[derive(Debug, Clone)]
pub struct PerformanceStage {
    pub label: String,
    pub duration_ms: f32,
}

/// Batch phoneme processing with enhanced error handling
pub async fn batch_process_phonemes(
    phoneme_batches: Vec<Vec<Phoneme>>,
    language: LanguageCode,
) -> Result<Vec<Vec<Phoneme>>> {
    let mut results = Vec::new();

    for batch in phoneme_batches {
        match validate_phonemes(&batch, language) {
            true => {
                let processed = postprocess_phonemes(batch, language);
                results.push(processed);
            }
            false => {
                return Err(G2pError::InvalidInput(
                    "Invalid phoneme sequence in batch".to_string(),
                ));
            }
        }
    }

    Ok(results)
}

/// Enhanced phoneme feature extraction
pub fn extract_phonetic_features(phoneme: &Phoneme) -> PhoneticFeatures {
    let symbol = phoneme.effective_symbol();

    if is_vowel(symbol) {
        extract_vowel_features(symbol)
    } else if is_consonant(symbol) {
        extract_consonant_features(symbol)
    } else {
        PhoneticFeatures::new()
    }
}

/// Extract vowel-specific phonetic features
fn extract_vowel_features(symbol: &str) -> PhoneticFeatures {
    match symbol {
        "i" | "iː" => PhoneticFeatures::vowel("high", "front", false),
        "u" | "uː" => PhoneticFeatures::vowel("high", "back", true),
        "e" | "eɪ" => PhoneticFeatures::vowel("mid", "front", false),
        "o" | "oʊ" => PhoneticFeatures::vowel("mid", "back", true),
        "a" | "æ" => PhoneticFeatures::vowel("low", "front", false),
        "ɑ" => PhoneticFeatures::vowel("low", "back", false),
        "ə" => PhoneticFeatures::vowel("mid", "central", false),
        _ => PhoneticFeatures::vowel("unknown", "unknown", false),
    }
}

/// Extract consonant-specific phonetic features
fn extract_consonant_features(symbol: &str) -> PhoneticFeatures {
    match symbol {
        "p" | "b" => PhoneticFeatures::consonant("stop", "bilabial", symbol == "b"),
        "t" | "d" => PhoneticFeatures::consonant("stop", "alveolar", symbol == "d"),
        "k" | "g" => PhoneticFeatures::consonant("stop", "velar", symbol == "g"),
        "f" | "v" => PhoneticFeatures::consonant("fricative", "labiodental", symbol == "v"),
        "s" | "z" => PhoneticFeatures::consonant("fricative", "alveolar", symbol == "z"),
        "ʃ" | "ʒ" => PhoneticFeatures::consonant("fricative", "postalveolar", symbol == "ʒ"),
        "m" => PhoneticFeatures::consonant("nasal", "bilabial", true),
        "n" => PhoneticFeatures::consonant("nasal", "alveolar", true),
        "ŋ" => PhoneticFeatures::consonant("nasal", "velar", true),
        "l" => PhoneticFeatures::consonant("liquid", "alveolar", true),
        "ɹ" | "r" => PhoneticFeatures::consonant("liquid", "alveolar", true),
        "j" => PhoneticFeatures::consonant("glide", "palatal", true),
        "w" => PhoneticFeatures::consonant("glide", "labial", true),
        "h" => PhoneticFeatures::consonant("fricative", "glottal", false),
        _ => PhoneticFeatures::consonant("unknown", "unknown", false),
    }
}

/// Create diagnostic context for G2P error reporting
pub fn create_diagnostic_context(
    text: &str,
    stage: ProcessingStage,
    language: LanguageCode,
) -> G2pDiagnosticContext {
    G2pDiagnosticContext {
        input_text: text.to_string(),
        language,
        backend: "default".to_string(),
        stage,
        context: std::collections::HashMap::new(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    }
}

/// Create diagnostic context with additional details
pub fn create_diagnostic_context_with_details(
    text: &str,
    stage: ProcessingStage,
    language: LanguageCode,
    details: std::collections::HashMap<String, String>,
) -> G2pDiagnosticContext {
    G2pDiagnosticContext {
        input_text: text.to_string(),
        language,
        backend: "default".to_string(),
        stage,
        context: details,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    }
}

/// Create formatted error report
pub fn create_g2p_error_report(error: &G2pError, context: &G2pDiagnosticContext) -> String {
    format!(
        "G2P Error Report\n\
        ================\n\
        Error: {:?}\n\
        Input: '{}'\n\
        Language: {:?}\n\
        Processing Stage: {:?}\n\
        Additional Info: {:?}\n",
        error, context.input_text, context.language, context.stage, context.context
    )
}
