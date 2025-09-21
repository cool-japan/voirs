use crate::error::VoirsError;
use crate::types::VoirsResult;
use regex::Regex;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct TextValidationConfig {
    pub min_length: usize,
    pub max_length: usize,
    pub allowed_character_sets: Vec<CharacterSet>,
    pub forbidden_patterns: Vec<String>,
    pub allowed_languages: Option<Vec<String>>,
    pub enable_profanity_filter: bool,
    pub normalize_whitespace: bool,
    pub allow_empty: bool,
    pub encoding_detection: bool,
}

impl Default for TextValidationConfig {
    fn default() -> Self {
        Self {
            min_length: 1,
            max_length: 10000,
            allowed_character_sets: vec![
                CharacterSet::Latin,
                CharacterSet::Numbers,
                CharacterSet::BasicPunctuation,
                CharacterSet::Whitespace,
            ],
            forbidden_patterns: vec![],
            allowed_languages: None,
            enable_profanity_filter: false,
            normalize_whitespace: true,
            allow_empty: false,
            encoding_detection: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CharacterSet {
    Latin,
    Cyrillic,
    Greek,
    Arabic,
    Hebrew,
    Chinese,
    Japanese,
    Korean,
    Thai,
    Hindi,
    Numbers,
    BasicPunctuation,
    ExtendedPunctuation,
    MathematicalSymbols,
    Whitespace,
    Control,
    Custom(String),
}

impl CharacterSet {
    fn get_regex(&self) -> &'static str {
        match self {
            Self::Latin => r"[a-zA-ZÀ-ÿĀ-žǍ-ǽ]",
            Self::Cyrillic => r"[\u0400-\u04FF]",
            Self::Greek => r"[\u0370-\u03FF]",
            Self::Arabic => r"[\u0600-\u06FF]",
            Self::Hebrew => r"[\u0590-\u05FF]",
            Self::Chinese => r"[\u4E00-\u9FFF]",
            Self::Japanese => r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]",
            Self::Korean => r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]",
            Self::Thai => r"[\u0E00-\u0E7F]",
            Self::Hindi => r"[\u0900-\u097F]",
            Self::Numbers => r"[0-9]",
            Self::BasicPunctuation => r"[.,!?;:()\x22\x27-]",
            Self::ExtendedPunctuation => r"[.,!?;:()\[\]{}\x22\x27\x60~@#$%^&*+=|\\/<>_-]",
            Self::MathematicalSymbols => r"[+\-*/=<>≤≥≠∞∑∏∫∂∇]",
            Self::Whitespace => r"[\s]",
            Self::Control => r"[\x00-\x1F\x7F-\x9F]",
            Self::Custom(_) => "", // Custom patterns need special handling
        }
    }

    fn check_character(&self, ch: char) -> bool {
        if let Self::Custom(pattern) = self {
            if let Ok(regex) = Regex::new(pattern) {
                return regex.is_match(&ch.to_string());
            }
            return false;
        }

        let regex_str = self.get_regex();
        if let Ok(regex) = Regex::new(regex_str) {
            regex.is_match(&ch.to_string())
        } else {
            false
        }
    }
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub normalized_text: String,
    pub detected_encoding: Option<String>,
    pub detected_language: Option<String>,
    pub character_set_analysis: CharacterSetAnalysis,
    pub validation_errors: Vec<ValidationError>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CharacterSetAnalysis {
    pub total_characters: usize,
    pub character_set_distribution: std::collections::HashMap<CharacterSet, usize>,
    pub unsupported_characters: Vec<char>,
    pub dominant_character_set: Option<CharacterSet>,
}

#[derive(Debug, Clone)]
pub enum ValidationError {
    TooShort {
        actual: usize,
        minimum: usize,
    },
    TooLong {
        actual: usize,
        maximum: usize,
    },
    ForbiddenCharacter {
        character: char,
        position: usize,
    },
    ForbiddenPattern {
        pattern: String,
        positions: Vec<usize>,
    },
    UnsupportedLanguage {
        detected: String,
        allowed: Vec<String>,
    },
    InvalidEncoding {
        detected: String,
        expected: Vec<String>,
    },
    ProfanityDetected {
        matches: Vec<String>,
    },
    EmptyInput,
    InvalidFormat {
        reason: String,
    },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort { actual, minimum } => {
                write!(
                    f,
                    "Text too short: {actual} characters (minimum: {minimum})"
                )
            }
            Self::TooLong { actual, maximum } => {
                write!(f, "Text too long: {actual} characters (maximum: {maximum})")
            }
            Self::ForbiddenCharacter {
                character,
                position,
            } => {
                write!(
                    f,
                    "Forbidden character '{character}' at position {position}"
                )
            }
            Self::ForbiddenPattern { pattern, positions } => {
                write!(
                    f,
                    "Forbidden pattern '{pattern}' found at positions: {positions:?}"
                )
            }
            Self::UnsupportedLanguage { detected, allowed } => {
                write!(
                    f,
                    "Unsupported language '{detected}' (allowed: {allowed:?})"
                )
            }
            Self::InvalidEncoding { detected, expected } => {
                write!(f, "Invalid encoding '{detected}' (expected: {expected:?})")
            }
            Self::ProfanityDetected { matches } => {
                write!(f, "Profanity detected: {matches:?}")
            }
            Self::EmptyInput => write!(f, "Empty input not allowed"),
            Self::InvalidFormat { reason } => write!(f, "Invalid format: {reason}"),
        }
    }
}

pub struct TextValidator {
    config: TextValidationConfig,
    profanity_list: HashSet<String>,
    forbidden_patterns: Vec<Regex>,
}

impl TextValidator {
    pub fn new(config: TextValidationConfig) -> VoirsResult<Self> {
        let forbidden_patterns = config
            .forbidden_patterns
            .iter()
            .map(|pattern| Regex::new(pattern))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                VoirsError::invalid_config(
                    "forbidden_patterns",
                    format!("{:?}", config.forbidden_patterns),
                    format!("Invalid regex: {e}"),
                )
            })?;

        Ok(Self {
            config,
            profanity_list: Self::load_default_profanity_list(),
            forbidden_patterns,
        })
    }

    pub fn validate(&self, text: &str) -> VoirsResult<ValidationResult> {
        let mut result = ValidationResult {
            is_valid: true,
            normalized_text: text.to_string(),
            detected_encoding: None,
            detected_language: None,
            character_set_analysis: CharacterSetAnalysis {
                total_characters: text.len(),
                character_set_distribution: std::collections::HashMap::new(),
                unsupported_characters: Vec::new(),
                dominant_character_set: None,
            },
            validation_errors: Vec::new(),
            warnings: Vec::new(),
        };

        // Check for empty input
        if text.is_empty() {
            if !self.config.allow_empty {
                result.validation_errors.push(ValidationError::EmptyInput);
                result.is_valid = false;
            }
            return Ok(result);
        }

        // Encoding detection
        if self.config.encoding_detection {
            result.detected_encoding = self.detect_encoding(text);
        }

        // Normalize whitespace
        if self.config.normalize_whitespace {
            result.normalized_text = self.normalize_whitespace(&result.normalized_text);
        }

        // Extract text to avoid borrow checker issues
        let normalized_text = result.normalized_text.clone();

        // Length validation
        self.validate_length(&normalized_text, &mut result);

        // Character set validation
        self.validate_character_sets(&normalized_text, &mut result);

        // Pattern validation
        self.validate_patterns(&normalized_text, &mut result);

        // Language validation
        if let Some(ref allowed_languages) = self.config.allowed_languages {
            self.validate_language(&normalized_text, allowed_languages, &mut result);
        }

        // Profanity filtering
        if self.config.enable_profanity_filter {
            self.check_profanity(&normalized_text, &mut result);
        }

        Ok(result)
    }

    fn validate_length(&self, text: &str, result: &mut ValidationResult) {
        let length = text.chars().count();

        if length < self.config.min_length {
            result.validation_errors.push(ValidationError::TooShort {
                actual: length,
                minimum: self.config.min_length,
            });
            result.is_valid = false;
        }

        if length > self.config.max_length {
            result.validation_errors.push(ValidationError::TooLong {
                actual: length,
                maximum: self.config.max_length,
            });
            result.is_valid = false;
        }
    }

    fn validate_character_sets(&self, text: &str, result: &mut ValidationResult) {
        let mut character_distribution = std::collections::HashMap::new();
        let mut unsupported_chars = Vec::new();

        for (pos, ch) in text.char_indices() {
            let mut supported = false;

            for charset in &self.config.allowed_character_sets {
                if charset.check_character(ch) {
                    *character_distribution.entry(charset.clone()).or_insert(0) += 1;
                    supported = true;
                    break;
                }
            }

            if !supported {
                unsupported_chars.push(ch);
                result
                    .validation_errors
                    .push(ValidationError::ForbiddenCharacter {
                        character: ch,
                        position: pos,
                    });
                result.is_valid = false;
            }
        }

        // Find dominant character set
        let dominant = character_distribution
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(charset, _)| charset.clone());

        result.character_set_analysis = CharacterSetAnalysis {
            total_characters: text.chars().count(),
            character_set_distribution: character_distribution,
            unsupported_characters: unsupported_chars,
            dominant_character_set: dominant,
        };
    }

    fn validate_patterns(&self, text: &str, result: &mut ValidationResult) {
        for (i, regex) in self.forbidden_patterns.iter().enumerate() {
            let matches: Vec<_> = regex.find_iter(text).collect();
            if !matches.is_empty() {
                let positions = matches.iter().map(|m| m.start()).collect();
                result
                    .validation_errors
                    .push(ValidationError::ForbiddenPattern {
                        pattern: self.config.forbidden_patterns[i].clone(),
                        positions,
                    });
                result.is_valid = false;
            }
        }
    }

    fn validate_language(
        &self,
        text: &str,
        allowed_languages: &[String],
        result: &mut ValidationResult,
    ) {
        if let Some(detected_lang) = self.detect_language(text) {
            result.detected_language = Some(detected_lang.clone());

            if !allowed_languages.contains(&detected_lang) {
                result
                    .validation_errors
                    .push(ValidationError::UnsupportedLanguage {
                        detected: detected_lang,
                        allowed: allowed_languages.to_vec(),
                    });
                result.is_valid = false;
            }
        }
    }

    fn check_profanity(&self, text: &str, result: &mut ValidationResult) {
        let text_lower = text.to_lowercase();
        let mut matches = Vec::new();

        for profanity in &self.profanity_list {
            if text_lower.contains(profanity) {
                matches.push(profanity.clone());
            }
        }

        if !matches.is_empty() {
            result
                .validation_errors
                .push(ValidationError::ProfanityDetected { matches });
            result.is_valid = false;
        }
    }

    fn normalize_whitespace(&self, text: &str) -> String {
        // Normalize various whitespace characters to standard spaces
        text.chars()
            .map(|c| if c.is_whitespace() { ' ' } else { c })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn detect_encoding(&self, text: &str) -> Option<String> {
        // Simple encoding detection based on character analysis
        let mut non_ascii_count = 0;
        let mut total_count = 0;

        for ch in text.chars() {
            total_count += 1;
            if !ch.is_ascii() {
                non_ascii_count += 1;
            }
        }

        if total_count == 0 {
            return Some("ASCII".to_string());
        }

        let non_ascii_ratio = non_ascii_count as f64 / total_count as f64;

        if non_ascii_ratio == 0.0 {
            Some("ASCII".to_string())
        } else {
            Some("UTF-8".to_string()) // Default to UTF-8 for any Unicode content
        }
    }

    fn detect_language(&self, text: &str) -> Option<String> {
        // Simple language detection based on character set analysis
        let mut char_counts = std::collections::HashMap::new();

        for ch in text.chars() {
            if CharacterSet::Latin.check_character(ch) {
                *char_counts.entry("latin").or_insert(0) += 1;
            } else if CharacterSet::Cyrillic.check_character(ch) {
                *char_counts.entry("cyrillic").or_insert(0) += 1;
            } else if CharacterSet::Chinese.check_character(ch) {
                *char_counts.entry("chinese").or_insert(0) += 1;
            } else if CharacterSet::Japanese.check_character(ch) {
                *char_counts.entry("japanese").or_insert(0) += 1;
            } else if CharacterSet::Korean.check_character(ch) {
                *char_counts.entry("korean").or_insert(0) += 1;
            } else if CharacterSet::Arabic.check_character(ch) {
                *char_counts.entry("arabic").or_insert(0) += 1;
            } else if CharacterSet::Hebrew.check_character(ch) {
                *char_counts.entry("hebrew").or_insert(0) += 1;
            }
        }

        char_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(lang, _)| lang.to_string())
    }

    fn load_default_profanity_list() -> HashSet<String> {
        // Basic profanity list - in production this would be loaded from a file
        vec!["damn", "hell", "shit", "fuck", "bitch", "ass", "crap"]
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    pub fn add_forbidden_pattern(&mut self, pattern: &str) -> VoirsResult<()> {
        let regex = Regex::new(pattern).map_err(|e| {
            VoirsError::invalid_config("forbidden_pattern", pattern, format!("Invalid regex: {e}"))
        })?;

        self.forbidden_patterns.push(regex);
        self.config.forbidden_patterns.push(pattern.to_string());
        Ok(())
    }

    pub fn add_profanity_word(&mut self, word: &str) {
        self.profanity_list.insert(word.to_lowercase());
    }

    pub fn remove_profanity_word(&mut self, word: &str) {
        self.profanity_list.remove(&word.to_lowercase());
    }

    pub fn update_config(&mut self, config: TextValidationConfig) -> VoirsResult<()> {
        // Re-compile forbidden patterns
        let forbidden_patterns = config
            .forbidden_patterns
            .iter()
            .map(|pattern| Regex::new(pattern))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                VoirsError::invalid_config(
                    "forbidden_patterns",
                    format!("{:?}", config.forbidden_patterns),
                    format!("Invalid regex: {e}"),
                )
            })?;

        self.forbidden_patterns = forbidden_patterns;
        self.config = config;
        Ok(())
    }
}

pub fn validate_text_basic(text: &str) -> VoirsResult<bool> {
    let validator = TextValidator::new(TextValidationConfig::default())?;
    let result = validator.validate(text)?;
    Ok(result.is_valid)
}

pub fn validate_text_with_config(
    text: &str,
    config: TextValidationConfig,
) -> VoirsResult<ValidationResult> {
    let validator = TextValidator::new(config)?;
    validator.validate(text)
}

pub fn normalize_text(text: &str) -> String {
    let config = TextValidationConfig {
        normalize_whitespace: true,
        ..TextValidationConfig::default()
    };

    if let Ok(validator) = TextValidator::new(config) {
        if let Ok(result) = validator.validate(text) {
            return result.normalized_text;
        }
    }

    text.to_string()
}
