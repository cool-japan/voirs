//! # VoiRS G2P (Grapheme-to-Phoneme) Conversion
//!
//! Converts text to phonemes using various backends including rule-based,
//! neural, and hybrid approaches for multiple languages.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Result type for G2P operations
pub type Result<T> = std::result::Result<T, G2pError>;

/// G2P-specific error types
#[derive(Error, Debug)]
pub enum G2pError {
    #[error("G2P conversion failed: {0}")]
    ConversionError(String),

    #[error("Unsupported language: {0:?}")]
    UnsupportedLanguage(LanguageCode),

    #[error("Model loading failed: {0}")]
    ModelError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Phoneme validation failed: {0}")]
    PhonemeValidationError(String),

    #[error("Backend error: {backend} - {message}")]
    BackendError { backend: String, message: String },

    #[error("Preprocessing error: {0}")]
    PreprocessingError(String),

    #[error("Performance optimization failed: {0}")]
    OptimizationError(String),
}

/// Diagnostic context for G2P conversion issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G2pDiagnosticContext {
    /// Original input text
    pub input_text: String,
    /// Detected or specified language
    pub language: LanguageCode,
    /// Backend used for conversion
    pub backend: String,
    /// Processing stage where error occurred
    pub stage: ProcessingStage,
    /// Additional context information
    pub context: HashMap<String, String>,
    /// Timestamp of the error
    pub timestamp: u64,
}

/// Processing stages for diagnostic context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingStage {
    /// Text preprocessing stage
    Preprocessing,
    /// Language detection stage
    LanguageDetection,
    /// Backend selection stage
    BackendSelection,
    /// Phoneme conversion stage
    PhonemeConversion,
    /// Phoneme validation stage
    PhonemeValidation,
    /// Post-processing stage
    PostProcessing,
}

impl G2pDiagnosticContext {
    /// Create a new diagnostic context
    pub fn new(
        input_text: String,
        language: LanguageCode,
        backend: String,
        stage: ProcessingStage,
    ) -> Self {
        Self {
            input_text,
            language,
            backend,
            stage,
            context: HashMap::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Add context information
    pub fn add_context(mut self, key: String, value: String) -> Self {
        self.context.insert(key, value);
        self
    }

    /// Get a formatted diagnostic report
    pub fn format_diagnostic_report(&self) -> String {
        format!(
            "G2P Diagnostic Report\n\
            ====================\n\
            Input Text: {}\n\
            Language: {:?}\n\
            Backend: {}\n\
            Processing Stage: {:?}\n\
            Timestamp: {}\n\
            Context: {:?}",
            self.input_text, self.language, self.backend, self.stage, self.timestamp, self.context
        )
    }
}

/// Language codes supported by VoiRS
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub enum LanguageCode {
    /// English (US)
    #[default]
    EnUs,
    /// English (UK)
    EnGb,
    /// Japanese
    Ja,
    /// Mandarin Chinese
    ZhCn,
    /// Korean
    Ko,
    /// German
    De,
    /// French
    Fr,
    /// Spanish
    Es,
    /// Italian
    It,
    /// Portuguese
    Pt,
}

impl LanguageCode {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            LanguageCode::EnUs => "en-US",
            LanguageCode::EnGb => "en-GB",
            LanguageCode::Ja => "ja",
            LanguageCode::ZhCn => "zh-CN",
            LanguageCode::Ko => "ko",
            LanguageCode::De => "de",
            LanguageCode::Fr => "fr",
            LanguageCode::Es => "es",
            LanguageCode::It => "it",
            LanguageCode::Pt => "pt",
        }
    }
}

/// Syllable position for phonemes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SyllablePosition {
    /// Beginning of syllable
    Onset,
    /// Vowel part of syllable
    Nucleus,
    /// End of syllable
    Coda,
    /// End of word/syllable
    Final,
    /// Standalone syllable
    Standalone,
}

/// Phonetic features for IPA classification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhoneticFeatures {
    /// Vowel/consonant classification
    pub manner: Option<String>, // vowel, plosive, fricative, nasal, etc.
    /// Place of articulation
    pub place: Option<String>, // bilabial, alveolar, velar, etc.
    /// Voice/voiceless
    pub voice: Option<bool>,
    /// Front/central/back (for vowels)
    pub frontness: Option<String>,
    /// High/mid/low (for vowels)
    pub height: Option<String>,
    /// Rounded/unrounded (for vowels)
    pub rounded: Option<bool>,
    /// Additional features
    pub other: HashMap<String, String>,
}

impl PhoneticFeatures {
    /// Create new empty phonetic features
    pub fn new() -> Self {
        Self {
            manner: None,
            place: None,
            voice: None,
            frontness: None,
            height: None,
            rounded: None,
            other: HashMap::new(),
        }
    }

    /// Create vowel features
    pub fn vowel(height: &str, frontness: &str, rounded: bool) -> Self {
        Self {
            manner: Some("vowel".to_string()),
            place: None,
            voice: Some(true), // vowels are voiced
            frontness: Some(frontness.to_string()),
            height: Some(height.to_string()),
            rounded: Some(rounded),
            other: HashMap::new(),
        }
    }

    /// Create consonant features
    pub fn consonant(manner: &str, place: &str, voiced: bool) -> Self {
        Self {
            manner: Some(manner.to_string()),
            place: Some(place.to_string()),
            voice: Some(voiced),
            frontness: None,
            height: None,
            rounded: None,
            other: HashMap::new(),
        }
    }
}

impl Default for PhoneticFeatures {
    fn default() -> Self {
        Self::new()
    }
}

/// A phoneme with its symbol and detailed features
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Phoneme {
    /// Phoneme symbol (IPA or language-specific)
    pub symbol: String,
    /// IPA symbol if different from main symbol
    pub ipa_symbol: Option<String>,
    /// Language-specific notation (ARPAbet, SAMPA, etc.)
    pub language_notation: Option<String>,
    /// Stress level: 0=none, 1=primary, 2=secondary, 3=tertiary
    pub stress: u8,
    /// Position within syllable
    pub syllable_position: SyllablePosition,
    /// Duration in milliseconds (if available)
    pub duration_ms: Option<f32>,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Structured phonetic features
    pub phonetic_features: Option<PhoneticFeatures>,
    /// Optional custom features
    pub custom_features: Option<HashMap<String, String>>,
    /// Word boundary marker
    pub is_word_boundary: bool,
    /// Syllable boundary marker
    pub is_syllable_boundary: bool,
}

impl Phoneme {
    /// Create new phoneme with default values
    pub fn new<S: Into<String>>(symbol: S) -> Self {
        Self {
            symbol: symbol.into(),
            ipa_symbol: None,
            language_notation: None,
            stress: 0,
            syllable_position: SyllablePosition::Standalone,
            duration_ms: None,
            confidence: 1.0,
            phonetic_features: None,
            custom_features: None,
            is_word_boundary: false,
            is_syllable_boundary: false,
        }
    }

    /// Fast constructor for single-character phonemes (DummyG2p optimization)
    #[inline]
    pub fn from_char(c: char) -> Self {
        Self {
            symbol: c.to_string(),
            ipa_symbol: None,
            language_notation: None,
            stress: 0,
            syllable_position: SyllablePosition::Standalone,
            duration_ms: None,
            confidence: 1.0,
            phonetic_features: None,
            custom_features: None,
            is_word_boundary: false,
            is_syllable_boundary: false,
        }
    }

    /// Create phoneme with stress and syllable position
    pub fn with_stress<S: Into<String>>(
        symbol: S,
        stress: u8,
        syllable_position: SyllablePosition,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            ipa_symbol: None,
            language_notation: None,
            stress,
            syllable_position,
            duration_ms: None,
            confidence: 1.0,
            phonetic_features: None,
            custom_features: None,
            is_word_boundary: false,
            is_syllable_boundary: false,
        }
    }

    /// Create phoneme with confidence score
    pub fn with_confidence<S: Into<String>>(symbol: S, confidence: f32) -> Self {
        Self {
            symbol: symbol.into(),
            ipa_symbol: None,
            language_notation: None,
            stress: 0,
            syllable_position: SyllablePosition::Standalone,
            duration_ms: None,
            confidence,
            phonetic_features: None,
            custom_features: None,
            is_word_boundary: false,
            is_syllable_boundary: false,
        }
    }

    /// Create phoneme with duration in milliseconds
    pub fn with_duration<S: Into<String>>(symbol: S, duration_ms: f32) -> Self {
        Self {
            symbol: symbol.into(),
            ipa_symbol: None,
            language_notation: None,
            stress: 0,
            syllable_position: SyllablePosition::Standalone,
            duration_ms: Some(duration_ms),
            confidence: 1.0,
            phonetic_features: None,
            custom_features: None,
            is_word_boundary: false,
            is_syllable_boundary: false,
        }
    }

    /// Create phoneme with custom features
    pub fn with_custom_features<S: Into<String>>(
        symbol: S,
        features: HashMap<String, String>,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            ipa_symbol: None,
            language_notation: None,
            stress: 0,
            syllable_position: SyllablePosition::Standalone,
            duration_ms: None,
            confidence: 1.0,
            phonetic_features: None,
            custom_features: Some(features),
            is_word_boundary: false,
            is_syllable_boundary: false,
        }
    }

    /// Create a fully specified phoneme
    #[allow(clippy::too_many_arguments)]
    pub fn full<S: Into<String>>(
        symbol: S,
        ipa_symbol: Option<String>,
        language_notation: Option<String>,
        stress: u8,
        syllable_position: SyllablePosition,
        duration_ms: Option<f32>,
        confidence: f32,
        phonetic_features: Option<PhoneticFeatures>,
        custom_features: Option<HashMap<String, String>>,
        is_word_boundary: bool,
        is_syllable_boundary: bool,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            ipa_symbol,
            language_notation,
            stress,
            syllable_position,
            duration_ms,
            confidence,
            phonetic_features,
            custom_features,
            is_word_boundary,
            is_syllable_boundary,
        }
    }

    /// Create phoneme with IPA symbol
    pub fn with_ipa<S: Into<String>, I: Into<String>>(symbol: S, ipa_symbol: I) -> Self {
        Self {
            symbol: symbol.into(),
            ipa_symbol: Some(ipa_symbol.into()),
            language_notation: None,
            stress: 0,
            syllable_position: SyllablePosition::Standalone,
            duration_ms: None,
            confidence: 1.0,
            phonetic_features: None,
            custom_features: None,
            is_word_boundary: false,
            is_syllable_boundary: false,
        }
    }

    /// Create phoneme with phonetic features
    pub fn with_phonetic_features<S: Into<String>>(symbol: S, features: PhoneticFeatures) -> Self {
        Self {
            symbol: symbol.into(),
            ipa_symbol: None,
            language_notation: None,
            stress: 0,
            syllable_position: SyllablePosition::Standalone,
            duration_ms: None,
            confidence: 1.0,
            phonetic_features: Some(features),
            custom_features: None,
            is_word_boundary: false,
            is_syllable_boundary: false,
        }
    }

    /// Create word boundary marker
    pub fn word_boundary() -> Self {
        Self {
            symbol: " ".to_string(),
            ipa_symbol: None,
            language_notation: None,
            stress: 0,
            syllable_position: SyllablePosition::Standalone,
            duration_ms: None,
            confidence: 1.0,
            phonetic_features: None,
            custom_features: None,
            is_word_boundary: true,
            is_syllable_boundary: false,
        }
    }

    /// Create syllable boundary marker
    pub fn syllable_boundary() -> Self {
        Self {
            symbol: ".".to_string(),
            ipa_symbol: None,
            language_notation: None,
            stress: 0,
            syllable_position: SyllablePosition::Standalone,
            duration_ms: None,
            confidence: 1.0,
            phonetic_features: None,
            custom_features: None,
            is_word_boundary: false,
            is_syllable_boundary: true,
        }
    }

    /// Check if phoneme is a vowel based on phonetic features
    pub fn is_vowel(&self) -> bool {
        self.phonetic_features
            .as_ref()
            .and_then(|f| f.manner.as_ref())
            .map(|m| m == "vowel")
            .unwrap_or(false)
    }

    /// Check if phoneme is a consonant based on phonetic features
    pub fn is_consonant(&self) -> bool {
        self.phonetic_features
            .as_ref()
            .and_then(|f| f.manner.as_ref())
            .map(|m| m != "vowel")
            .unwrap_or(false)
    }

    /// Get effective symbol (IPA if available, otherwise main symbol)
    pub fn effective_symbol(&self) -> &str {
        self.ipa_symbol.as_ref().unwrap_or(&self.symbol)
    }

    /// Check if phoneme has primary stress
    pub fn has_primary_stress(&self) -> bool {
        self.stress == 1
    }

    /// Check if phoneme has secondary stress
    pub fn has_secondary_stress(&self) -> bool {
        self.stress == 2
    }

    /// Check if phoneme has any stress
    pub fn has_stress(&self) -> bool {
        self.stress > 0
    }

    /// Get duration or estimate based on phoneme type
    pub fn duration_or_estimate(&self) -> f32 {
        if let Some(duration) = self.duration_ms {
            return duration;
        }

        // Provide rough duration estimates based on phoneme characteristics
        if self.is_word_boundary || self.is_syllable_boundary {
            return 0.0;
        }

        if self.is_vowel() {
            120.0 // vowels are typically longer
        } else if self.is_consonant() {
            80.0 // consonants are typically shorter
        } else {
            100.0 // default estimate
        }
    }

    /// Check if phoneme is at the beginning of a syllable
    pub fn is_syllable_initial(&self) -> bool {
        matches!(self.syllable_position, SyllablePosition::Onset)
    }
}

/// G2P metadata information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G2pMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model description
    pub description: String,
    /// Supported languages
    pub supported_languages: Vec<LanguageCode>,
    /// Accuracy scores per language (if available)
    pub accuracy_scores: HashMap<LanguageCode, f32>,
}

/// Trait for grapheme-to-phoneme conversion
#[async_trait]
pub trait G2p: Send + Sync {
    /// Convert text to phonemes
    async fn to_phonemes(&self, text: &str, lang: Option<LanguageCode>) -> Result<Vec<Phoneme>>;

    /// Get supported languages
    fn supported_languages(&self) -> Vec<LanguageCode>;

    /// Get model metadata
    fn metadata(&self) -> G2pMetadata;
}

pub mod accuracy;
pub mod advanced;
pub mod backends;
pub mod config;
pub mod detection;
pub mod english;
pub mod models;
pub mod optimization;
pub mod performance;
pub mod preprocessing;
pub mod rules;
pub mod ssml;
pub mod ssml_legacy;
pub mod training;
pub mod utils;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::backends::{ChinesePinyinG2p, JapaneseDictG2p};
    pub use crate::{
        DummyG2p, G2p, G2pConverter, G2pError, G2pMetadata, LanguageCode, Phoneme,
        PhoneticFeatures, Result, SyllablePosition,
    };
    pub use async_trait::async_trait;
}

// Types are already public in the root module

/// G2P converter with multiple backend support
pub struct G2pConverter {
    backends: HashMap<LanguageCode, Box<dyn G2p>>,
    default_backend: Option<Box<dyn G2p>>,
}

impl G2pConverter {
    /// Create new G2P converter
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
            default_backend: None,
        }
    }

    /// Add backend for specific language
    pub fn add_backend(&mut self, language: LanguageCode, backend: Box<dyn G2p>) {
        self.backends.insert(language, backend);
    }

    /// Set default backend for unknown languages
    pub fn set_default_backend(&mut self, backend: Box<dyn G2p>) {
        self.default_backend = Some(backend);
    }

    /// Get backend for language
    fn get_backend(&self, language: Option<LanguageCode>) -> Result<&dyn G2p> {
        if let Some(lang) = language {
            if let Some(backend) = self.backends.get(&lang) {
                return Ok(backend.as_ref());
            }
        }

        if let Some(default) = &self.default_backend {
            Ok(default.as_ref())
        } else {
            Err(G2pError::ConfigError(
                "No G2P backend available".to_string(),
            ))
        }
    }
}

impl Default for G2pConverter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl G2p for G2pConverter {
    async fn to_phonemes(&self, text: &str, lang: Option<LanguageCode>) -> Result<Vec<Phoneme>> {
        let backend = self.get_backend(lang)?;
        backend.to_phonemes(text, lang).await
    }

    fn supported_languages(&self) -> Vec<LanguageCode> {
        let mut languages: Vec<LanguageCode> = self.backends.keys().copied().collect();

        // Add languages from default backend if available
        if let Some(default) = &self.default_backend {
            languages.extend(default.supported_languages());
        }

        languages.sort();
        languages.dedup();
        languages
    }

    fn metadata(&self) -> G2pMetadata {
        let mut accuracy_scores = HashMap::new();

        // Collect accuracy scores from all backends
        for (lang, backend) in &self.backends {
            let backend_metadata = backend.metadata();
            if let Some(score) = backend_metadata.accuracy_scores.get(lang) {
                accuracy_scores.insert(*lang, *score);
            }
        }

        // Add default accuracy scores for backends if not provided
        for lang in self.supported_languages() {
            accuracy_scores.entry(lang).or_insert_with(|| {
                match lang {
                    LanguageCode::EnUs | LanguageCode::EnGb => 0.85, // English typically higher
                    LanguageCode::De
                    | LanguageCode::Fr
                    | LanguageCode::Es
                    | LanguageCode::It
                    | LanguageCode::Pt => 0.80, // European languages
                    LanguageCode::Ja => 0.75,                        // Japanese G2P is complex
                    LanguageCode::ZhCn | LanguageCode::Ko => 0.70, // CJK languages are challenging
                }
            });
        }

        G2pMetadata {
            name: "VoiRS G2P Converter".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: "Multi-backend grapheme-to-phoneme converter".to_string(),
            supported_languages: self.supported_languages(),
            accuracy_scores,
        }
    }
}

/// Dummy G2P backend for testing and fallback
pub struct DummyG2p {
    supported_langs: Vec<LanguageCode>,
}

impl DummyG2p {
    /// Create new dummy G2P backend
    pub fn new() -> Self {
        Self {
            supported_langs: vec![LanguageCode::EnUs],
        }
    }

    /// Create with custom supported languages
    pub fn with_languages(languages: Vec<LanguageCode>) -> Self {
        Self {
            supported_langs: languages,
        }
    }
}

impl Default for DummyG2p {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl G2p for DummyG2p {
    async fn to_phonemes(&self, text: &str, _lang: Option<LanguageCode>) -> Result<Vec<Phoneme>> {
        // Simple character-to-phoneme mapping for testing
        let phonemes: Vec<Phoneme> = text
            .chars()
            .filter(|c| c.is_alphabetic())
            .map(Phoneme::from_char)
            .collect();

        // Debug logging only when trace level is enabled to avoid performance impact
        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!(
                "DummyG2p: Generated {} phonemes for '{}'",
                phonemes.len(),
                text
            );
        }
        Ok(phonemes)
    }

    fn supported_languages(&self) -> Vec<LanguageCode> {
        // Return cloned vector - optimized for common single-language case
        self.supported_langs.clone()
    }

    fn metadata(&self) -> G2pMetadata {
        G2pMetadata {
            name: "Dummy G2P".to_string(),
            version: "0.1.0".to_string(),
            description: "Dummy G2P backend for testing".to_string(),
            supported_languages: self.supported_languages(),
            accuracy_scores: HashMap::new(),
        }
    }
}

// Implement G2p for Box<dyn G2p> to enable trait object usage
#[async_trait]
impl G2p for Box<dyn G2p> {
    async fn to_phonemes(&self, text: &str, lang: Option<LanguageCode>) -> Result<Vec<Phoneme>> {
        self.as_ref().to_phonemes(text, lang).await
    }

    fn supported_languages(&self) -> Vec<LanguageCode> {
        self.as_ref().supported_languages()
    }

    fn metadata(&self) -> G2pMetadata {
        self.as_ref().metadata()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_g2p_converter() {
        let mut converter = G2pConverter::new();

        // Add dummy backend for English
        converter.add_backend(LanguageCode::EnUs, Box::new(DummyG2p::new()));

        // Test conversion
        let phonemes = converter
            .to_phonemes("hello", Some(LanguageCode::EnUs))
            .await
            .unwrap();
        assert_eq!(phonemes.len(), 5); // h-e-l-l-o

        // Test supported languages
        let languages = converter.supported_languages();
        assert!(languages.contains(&LanguageCode::EnUs));
    }

    #[tokio::test]
    async fn test_dummy_g2p() {
        let g2p = DummyG2p::new();

        let phonemes = g2p.to_phonemes("test", None).await.unwrap();
        assert_eq!(phonemes.len(), 4);
        assert_eq!(phonemes[0].symbol, "t");
        assert_eq!(phonemes[1].symbol, "e");

        let languages = g2p.supported_languages();
        assert_eq!(languages, vec![LanguageCode::EnUs]);
    }

    #[tokio::test]
    async fn test_english_rule_g2p() {
        use crate::rules::EnglishRuleG2p;

        let g2p = EnglishRuleG2p::new().unwrap();

        // Test basic dictionary words - "the" should split into ð and ə
        let phonemes = g2p
            .to_phonemes("the", Some(LanguageCode::EnUs))
            .await
            .unwrap();
        assert_eq!(phonemes.len(), 2);
        assert_eq!(phonemes[0].symbol, "ð");
        assert_eq!(phonemes[1].symbol, "ə");

        // Test rule-based conversion
        let phonemes = g2p
            .to_phonemes("cat", Some(LanguageCode::EnUs))
            .await
            .unwrap();
        assert_eq!(phonemes.len(), 3);
        assert_eq!(phonemes[0].symbol, "k"); // c -> k
        assert_eq!(phonemes[1].symbol, "æ"); // a -> æ
        assert_eq!(phonemes[2].symbol, "t"); // t -> t

        // Test multiple words
        let phonemes = g2p
            .to_phonemes("hello world", Some(LanguageCode::EnUs))
            .await
            .unwrap();
        assert!(phonemes.len() > 5); // Multiple phonemes with word boundary

        // Test supported languages
        let languages = g2p.supported_languages();
        assert!(languages.contains(&LanguageCode::EnUs));
        assert!(languages.contains(&LanguageCode::EnGb));
    }

    #[tokio::test]
    async fn test_english_rule_g2p_vowel_patterns() {
        use crate::rules::EnglishRuleG2p;

        let g2p = EnglishRuleG2p::new().unwrap();

        // Test magic-e pattern
        let phonemes = g2p
            .to_phonemes("cake", Some(LanguageCode::EnUs))
            .await
            .unwrap();
        assert_eq!(phonemes.len(), 1);
        assert_eq!(phonemes[0].symbol, "eɪk"); // ake -> eɪk

        // Test consonant digraphs
        let phonemes = g2p
            .to_phonemes("ship", Some(LanguageCode::EnUs))
            .await
            .unwrap();
        assert_eq!(phonemes.len(), 3);
        assert_eq!(phonemes[0].symbol, "ʃ"); // sh -> ʃ
        assert_eq!(phonemes[1].symbol, "ɪ"); // i -> ɪ
        assert_eq!(phonemes[2].symbol, "p"); // p -> p

        // Test vowel combinations
        let phonemes = g2p
            .to_phonemes("tree", Some(LanguageCode::EnUs))
            .await
            .unwrap();
        assert_eq!(phonemes.len(), 3);
        assert_eq!(phonemes[0].symbol, "t"); // t -> t
        assert_eq!(phonemes[1].symbol, "r"); // r -> r
        assert_eq!(phonemes[2].symbol, "iː"); // ee -> iː
    }
}
