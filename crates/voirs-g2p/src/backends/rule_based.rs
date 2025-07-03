//! Rule-based G2P implementation with comprehensive phoneme mappings.

use async_trait::async_trait;
use std::collections::HashMap;
use crate::{G2p, G2pMetadata, LanguageCode, Phoneme, Result};
use crate::preprocessing::TextPreprocessor;

/// A phonological rule for grapheme-to-phoneme conversion
#[derive(Debug, Clone)]
pub struct PhonologicalRule {
    /// Input pattern (grapheme)
    pub pattern: String,
    /// Output phoneme(s)
    pub phoneme: String,
    /// Left context (what must come before)
    pub left_context: Option<String>,
    /// Right context (what must come after)
    pub right_context: Option<String>,
    /// Priority (higher values take precedence)
    pub priority: u32,
}

impl PhonologicalRule {
    pub fn new(pattern: &str, phoneme: &str) -> Self {
        Self {
            pattern: pattern.to_string(),
            phoneme: phoneme.to_string(),
            left_context: None,
            right_context: None,
            priority: 0,
        }
    }
    
    pub fn with_context(pattern: &str, phoneme: &str, left: Option<&str>, right: Option<&str>) -> Self {
        Self {
            pattern: pattern.to_string(),
            phoneme: phoneme.to_string(),
            left_context: left.map(String::from),
            right_context: right.map(String::from),
            priority: 10, // Context-aware rules have higher priority
        }
    }
    
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }
}

/// Rule-based G2P converter with advanced phonological rules
pub struct RuleBasedG2p {
    language: LanguageCode,
    rules: Vec<PhonologicalRule>,
    syllable_patterns: HashMap<String, Vec<String>>,
    stress_rules: Vec<StressRule>,
}

/// Stress assignment rule
#[derive(Debug, Clone)]
pub struct StressRule {
    /// Pattern to match (e.g., syllable ending)
    pub pattern: String,
    /// Stress level (0=none, 1=primary, 2=secondary)
    pub stress: u8,
    /// Position from end of word (-1 = last syllable, -2 = second to last, etc.)
    pub position: i32,
}

impl RuleBasedG2p {
    /// Create new rule-based G2P for language
    pub fn new(language: LanguageCode) -> Self {
        let mut g2p = Self {
            language,
            rules: Vec::new(),
            syllable_patterns: HashMap::new(),
            stress_rules: Vec::new(),
        };
        g2p.load_default_rules();
        g2p.load_stress_rules();
        g2p
    }
    
    /// Load comprehensive phonological rules for the language
    fn load_default_rules(&mut self) {
        match self.language {
            LanguageCode::EnUs | LanguageCode::EnGb => self.load_english_rules(),
            LanguageCode::De => self.load_german_rules(),
            LanguageCode::Fr => self.load_french_rules(),
            LanguageCode::Es => self.load_spanish_rules(),
            LanguageCode::Ja => self.load_japanese_rules(),
            _ => self.load_fallback_rules(),
        }
    }
    
    /// Load English phonological rules
    fn load_english_rules(&mut self) {
        // Vowels with context
        self.rules.push(PhonologicalRule::with_context("a", "eɪ", None, Some("e")).with_priority(50)); // "ate"
        self.rules.push(PhonologicalRule::with_context("a", "ɑː", Some("f"), None).with_priority(40)); // "father"
        self.rules.push(PhonologicalRule::with_context("a", "ɔː", Some("w"), None).with_priority(40)); // "water"
        self.rules.push(PhonologicalRule::new("a", "æ").with_priority(10)); // default "cat"
        
        self.rules.push(PhonologicalRule::with_context("e", "iː", None, Some("e")).with_priority(50)); // "see"
        self.rules.push(PhonologicalRule::with_context("e", "ɪ", None, Some("r")).with_priority(40)); // "her"
        self.rules.push(PhonologicalRule::new("e", "ɛ").with_priority(10)); // default "bed"
        
        self.rules.push(PhonologicalRule::with_context("i", "aɪ", None, Some("e")).with_priority(50)); // "bite"
        self.rules.push(PhonologicalRule::with_context("i", "aɪ", None, Some("gh")).with_priority(45)); // "light"
        self.rules.push(PhonologicalRule::new("i", "ɪ").with_priority(10)); // default "bit"
        
        self.rules.push(PhonologicalRule::with_context("o", "oʊ", None, Some("e")).with_priority(50)); // "note"
        self.rules.push(PhonologicalRule::with_context("o", "uː", None, Some("o")).with_priority(45)); // "moon"
        self.rules.push(PhonologicalRule::new("o", "ɑ").with_priority(10)); // default "hot"
        
        self.rules.push(PhonologicalRule::with_context("u", "juː", Some("h"), None).with_priority(50)); // "huge"
        self.rules.push(PhonologicalRule::with_context("u", "uː", None, Some("e")).with_priority(45)); // "tune"
        self.rules.push(PhonologicalRule::new("u", "ʌ").with_priority(10)); // default "but"
        
        // Consonants
        self.rules.push(PhonologicalRule::with_context("th", "θ", None, Some("ing")).with_priority(50)); // "thing"
        self.rules.push(PhonologicalRule::with_context("th", "ð", Some("e"), None).with_priority(40)); // "the"
        self.rules.push(PhonologicalRule::new("th", "θ").with_priority(30)); // default "think"
        
        self.rules.push(PhonologicalRule::with_context("ch", "tʃ", None, None).with_priority(40));
        self.rules.push(PhonologicalRule::with_context("sh", "ʃ", None, None).with_priority(40));
        self.rules.push(PhonologicalRule::with_context("ph", "f", None, None).with_priority(40));
        self.rules.push(PhonologicalRule::with_context("gh", "", None, Some("t")).with_priority(40)); // "light"
        self.rules.push(PhonologicalRule::with_context("gh", "f", None, None).with_priority(30)); // "laugh"
        
        self.rules.push(PhonologicalRule::new("b", "b"));
        self.rules.push(PhonologicalRule::new("c", "k"));
        self.rules.push(PhonologicalRule::new("d", "d"));
        self.rules.push(PhonologicalRule::new("f", "f"));
        self.rules.push(PhonologicalRule::new("g", "g"));
        self.rules.push(PhonologicalRule::new("h", "h"));
        self.rules.push(PhonologicalRule::new("j", "dʒ"));
        self.rules.push(PhonologicalRule::new("k", "k"));
        self.rules.push(PhonologicalRule::new("l", "l"));
        self.rules.push(PhonologicalRule::new("m", "m"));
        self.rules.push(PhonologicalRule::new("n", "n"));
        self.rules.push(PhonologicalRule::new("p", "p"));
        self.rules.push(PhonologicalRule::new("q", "kw"));
        self.rules.push(PhonologicalRule::new("r", "r"));
        self.rules.push(PhonologicalRule::new("s", "s"));
        self.rules.push(PhonologicalRule::new("t", "t"));
        self.rules.push(PhonologicalRule::new("v", "v"));
        self.rules.push(PhonologicalRule::new("w", "w"));
        self.rules.push(PhonologicalRule::new("x", "ks"));
        self.rules.push(PhonologicalRule::new("y", "j"));
        self.rules.push(PhonologicalRule::new("z", "z"));
        
        // Sort rules by priority (highest first)
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }
    
    /// Load German phonological rules
    fn load_german_rules(&mut self) {
        // German vowels
        self.rules.push(PhonologicalRule::new("ä", "ɛ"));
        self.rules.push(PhonologicalRule::new("ö", "ø"));
        self.rules.push(PhonologicalRule::new("ü", "y"));
        self.rules.push(PhonologicalRule::new("ß", "s"));
        
        // German consonants
        self.rules.push(PhonologicalRule::with_context("ch", "x", Some("a"), None).with_priority(40)); // "ach"
        self.rules.push(PhonologicalRule::with_context("ch", "ç", Some("i"), None).with_priority(40)); // "ich"
        self.rules.push(PhonologicalRule::new("ch", "ç").with_priority(30));
        
        self.rules.push(PhonologicalRule::new("sch", "ʃ"));
        self.rules.push(PhonologicalRule::new("sp", "ʃp"));
        self.rules.push(PhonologicalRule::new("st", "ʃt"));
        
        // Basic consonants
        self.rules.push(PhonologicalRule::new("b", "b"));
        self.rules.push(PhonologicalRule::new("d", "d"));
        self.rules.push(PhonologicalRule::new("f", "f"));
        self.rules.push(PhonologicalRule::new("g", "g"));
        self.rules.push(PhonologicalRule::new("h", "h"));
        self.rules.push(PhonologicalRule::new("j", "j"));
        self.rules.push(PhonologicalRule::new("k", "k"));
        self.rules.push(PhonologicalRule::new("l", "l"));
        self.rules.push(PhonologicalRule::new("m", "m"));
        self.rules.push(PhonologicalRule::new("n", "n"));
        self.rules.push(PhonologicalRule::new("p", "p"));
        self.rules.push(PhonologicalRule::new("r", "ʁ"));
        self.rules.push(PhonologicalRule::new("s", "s"));
        self.rules.push(PhonologicalRule::new("t", "t"));
        self.rules.push(PhonologicalRule::new("v", "f"));
        self.rules.push(PhonologicalRule::new("w", "v"));
        self.rules.push(PhonologicalRule::new("z", "ts"));
        
        // Basic vowels
        self.rules.push(PhonologicalRule::new("a", "a"));
        self.rules.push(PhonologicalRule::new("e", "e"));
        self.rules.push(PhonologicalRule::new("i", "i"));
        self.rules.push(PhonologicalRule::new("o", "o"));
        self.rules.push(PhonologicalRule::new("u", "u"));
        
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }
    
    /// Load French phonological rules
    fn load_french_rules(&mut self) {
        // French vowels
        self.rules.push(PhonologicalRule::new("é", "e"));
        self.rules.push(PhonologicalRule::new("è", "ɛ"));
        self.rules.push(PhonologicalRule::new("ê", "ɛ"));
        self.rules.push(PhonologicalRule::new("à", "a"));
        self.rules.push(PhonologicalRule::new("ù", "y"));
        self.rules.push(PhonologicalRule::new("ç", "s"));
        
        // French consonants
        self.rules.push(PhonologicalRule::new("j", "ʒ"));
        self.rules.push(PhonologicalRule::new("r", "ʁ"));
        
        // Basic letters
        self.rules.push(PhonologicalRule::new("a", "a"));
        self.rules.push(PhonologicalRule::new("e", "ə"));
        self.rules.push(PhonologicalRule::new("i", "i"));
        self.rules.push(PhonologicalRule::new("o", "o"));
        self.rules.push(PhonologicalRule::new("u", "u"));
        
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }
    
    /// Load Spanish phonological rules
    fn load_spanish_rules(&mut self) {
        // Spanish has very regular pronunciation
        self.rules.push(PhonologicalRule::new("ñ", "ɲ"));
        self.rules.push(PhonologicalRule::new("ll", "ʎ"));
        self.rules.push(PhonologicalRule::new("rr", "r"));
        
        // Basic vowels (Spanish has 5 pure vowels)
        self.rules.push(PhonologicalRule::new("a", "a"));
        self.rules.push(PhonologicalRule::new("e", "e"));
        self.rules.push(PhonologicalRule::new("i", "i"));
        self.rules.push(PhonologicalRule::new("o", "o"));
        self.rules.push(PhonologicalRule::new("u", "u"));
        
        // Consonants
        self.rules.push(PhonologicalRule::new("b", "b"));
        self.rules.push(PhonologicalRule::new("c", "k"));
        self.rules.push(PhonologicalRule::new("d", "d"));
        self.rules.push(PhonologicalRule::new("f", "f"));
        self.rules.push(PhonologicalRule::new("g", "g"));
        self.rules.push(PhonologicalRule::new("h", "")); // Silent in Spanish
        self.rules.push(PhonologicalRule::new("j", "x"));
        self.rules.push(PhonologicalRule::new("k", "k"));
        self.rules.push(PhonologicalRule::new("l", "l"));
        self.rules.push(PhonologicalRule::new("m", "m"));
        self.rules.push(PhonologicalRule::new("n", "n"));
        self.rules.push(PhonologicalRule::new("p", "p"));
        self.rules.push(PhonologicalRule::new("q", "k"));
        self.rules.push(PhonologicalRule::new("r", "ɾ"));
        self.rules.push(PhonologicalRule::new("s", "s"));
        self.rules.push(PhonologicalRule::new("t", "t"));
        self.rules.push(PhonologicalRule::new("v", "b"));
        self.rules.push(PhonologicalRule::new("w", "w"));
        self.rules.push(PhonologicalRule::new("x", "ks"));
        self.rules.push(PhonologicalRule::new("y", "j"));
        self.rules.push(PhonologicalRule::new("z", "θ")); // European Spanish
        
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }
    
    /// Load basic Japanese phonological rules
    fn load_japanese_rules(&mut self) {
        // This is a simplified version - real Japanese G2P would need proper kana support
        // Hiragana vowels
        self.rules.push(PhonologicalRule::new("あ", "a"));
        self.rules.push(PhonologicalRule::new("い", "i"));
        self.rules.push(PhonologicalRule::new("う", "u"));
        self.rules.push(PhonologicalRule::new("え", "e"));
        self.rules.push(PhonologicalRule::new("お", "o"));
        
        // Basic consonants + vowels
        self.rules.push(PhonologicalRule::new("か", "ka"));
        self.rules.push(PhonologicalRule::new("き", "ki"));
        self.rules.push(PhonologicalRule::new("く", "ku"));
        self.rules.push(PhonologicalRule::new("け", "ke"));
        self.rules.push(PhonologicalRule::new("こ", "ko"));
        
        // This would need to be much more comprehensive for real Japanese support
    }
    
    /// Load fallback rules for unsupported languages
    fn load_fallback_rules(&mut self) {
        // Basic Latin alphabet mapping
        for c in "abcdefghijklmnopqrstuvwxyz".chars() {
            self.rules.push(PhonologicalRule::new(&c.to_string(), &c.to_string()));
        }
    }
    
    /// Load stress assignment rules
    fn load_stress_rules(&mut self) {
        match self.language {
            LanguageCode::EnUs | LanguageCode::EnGb => {
                // English stress rules (simplified)
                self.stress_rules.push(StressRule {
                    pattern: "tion".to_string(),
                    stress: 1,
                    position: -2, // Second to last syllable
                });
                self.stress_rules.push(StressRule {
                    pattern: "ity".to_string(),
                    stress: 1,
                    position: -3, // Third to last syllable
                });
            }
            LanguageCode::Es => {
                // Spanish stress rules
                self.stress_rules.push(StressRule {
                    pattern: "vocal".to_string(), // Words ending in vowel, n, or s
                    stress: 1,
                    position: -2, // Penultimate syllable
                });
            }
            _ => {
                // Default: stress first syllable
                self.stress_rules.push(StressRule {
                    pattern: "".to_string(),
                    stress: 1,
                    position: 1,
                });
            }
        }
    }
    
    /// Apply phonological rules to convert text to phonemes
    fn apply_rules(&self, text: &str) -> Vec<Phoneme> {
        let mut phonemes = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        
        while i < chars.len() {
            let mut matched = false;
            
            // Try to match rules in order of priority
            for rule in &self.rules {
                if let Some(phoneme) = self.try_match_rule(rule, &chars, i) {
                    if !phoneme.is_empty() {
                        phonemes.push(Phoneme::new(phoneme));
                    }
                    i += rule.pattern.chars().count();
                    matched = true;
                    break;
                }
            }
            
            if !matched {
                // No rule matched, skip this character
                i += 1;
            }
        }
        
        phonemes
    }
    
    /// Try to match a rule at the given position
    fn try_match_rule(&self, rule: &PhonologicalRule, chars: &[char], pos: usize) -> Option<String> {
        let pattern_chars: Vec<char> = rule.pattern.chars().collect();
        
        // Check if pattern matches at current position
        if pos + pattern_chars.len() > chars.len() {
            return None;
        }
        
        for (i, &pattern_char) in pattern_chars.iter().enumerate() {
            if chars[pos + i] != pattern_char {
                return None;
            }
        }
        
        // Check left context if specified
        if let Some(ref left_context) = rule.left_context {
            let left_chars: Vec<char> = left_context.chars().collect();
            if pos < left_chars.len() {
                return None;
            }
            for (i, &context_char) in left_chars.iter().enumerate() {
                if chars[pos - left_chars.len() + i] != context_char {
                    return None;
                }
            }
        }
        
        // Check right context if specified
        if let Some(ref right_context) = rule.right_context {
            let right_chars: Vec<char> = right_context.chars().collect();
            let right_start = pos + pattern_chars.len();
            if right_start + right_chars.len() > chars.len() {
                return None;
            }
            for (i, &context_char) in right_chars.iter().enumerate() {
                if chars[right_start + i] != context_char {
                    return None;
                }
            }
        }
        
        Some(rule.phoneme.clone())
    }
}

#[async_trait]
impl G2p for RuleBasedG2p {
    async fn to_phonemes(&self, text: &str, _lang: Option<LanguageCode>) -> Result<Vec<Phoneme>> {
        // Preprocess the text
        let preprocessor = TextPreprocessor::new(self.language);
        let processed_text = preprocessor.preprocess(text)?;
        
        // Convert to lowercase for rule matching
        let clean_text = processed_text.to_lowercase();
        
        // Apply phonological rules
        let phonemes = self.apply_rules(&clean_text);
        
        tracing::debug!("RuleBasedG2p: Generated {} phonemes for '{}' -> '{}'", 
                       phonemes.len(), text, clean_text);
        
        Ok(phonemes)
    }
    
    fn supported_languages(&self) -> Vec<LanguageCode> {
        vec![self.language]
    }
    
    fn metadata(&self) -> G2pMetadata {
        let mut accuracy_scores = HashMap::new();
        
        // Add estimated accuracy scores per language
        match self.language {
            LanguageCode::EnUs | LanguageCode::EnGb => {
                accuracy_scores.insert(self.language, 0.75); // Conservative estimate
            }
            LanguageCode::Es => {
                accuracy_scores.insert(self.language, 0.85); // Spanish is more regular
            }
            LanguageCode::De => {
                accuracy_scores.insert(self.language, 0.80);
            }
            LanguageCode::Fr => {
                accuracy_scores.insert(self.language, 0.70);
            }
            LanguageCode::Ja => {
                accuracy_scores.insert(self.language, 0.60); // Very simplified implementation
            }
            _ => {
                accuracy_scores.insert(self.language, 0.50); // Fallback rules
            }
        }
        
        G2pMetadata {
            name: "Rule-based G2P".to_string(),
            version: "0.2.0".to_string(),
            description: format!("Advanced rule-based G2P for {}", self.language.as_str()),
            supported_languages: vec![self.language],
            accuracy_scores,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_english_rule_based_g2p() {
        let g2p = RuleBasedG2p::new(LanguageCode::EnUs);
        
        // Test basic words
        let phonemes = g2p.to_phonemes("cat", None).await.unwrap();
        assert!(!phonemes.is_empty());
        
        // Test with context-aware rules
        let phonemes = g2p.to_phonemes("the", None).await.unwrap();
        assert!(!phonemes.is_empty());
        
        // Test digraphs
        let phonemes = g2p.to_phonemes("think", None).await.unwrap();
        assert!(!phonemes.is_empty());
    }

    #[tokio::test]
    async fn test_german_rule_based_g2p() {
        let g2p = RuleBasedG2p::new(LanguageCode::De);
        
        let phonemes = g2p.to_phonemes("Hallo", None).await.unwrap();
        assert!(!phonemes.is_empty());
        
        // Test German umlauts
        let phonemes = g2p.to_phonemes("für", None).await.unwrap();
        assert!(!phonemes.is_empty());
    }

    #[tokio::test]
    async fn test_spanish_rule_based_g2p() {
        let g2p = RuleBasedG2p::new(LanguageCode::Es);
        
        let phonemes = g2p.to_phonemes("hola", None).await.unwrap();
        assert!(!phonemes.is_empty());
        
        // Test Spanish ñ
        let phonemes = g2p.to_phonemes("niño", None).await.unwrap();
        assert!(!phonemes.is_empty());
    }

    #[test]
    fn test_rule_matching() {
        let g2p = RuleBasedG2p::new(LanguageCode::EnUs);
        
        // Test basic rule matching
        let text = "cat";
        let phonemes = g2p.apply_rules(text);
        assert_eq!(phonemes.len(), 3);
    }

    #[test]
    fn test_phonological_rule_creation() {
        let rule = PhonologicalRule::new("th", "θ");
        assert_eq!(rule.pattern, "th");
        assert_eq!(rule.phoneme, "θ");
        assert_eq!(rule.priority, 0);
        
        let context_rule = PhonologicalRule::with_context("th", "ð", Some("e"), None);
        assert_eq!(context_rule.left_context, Some("e".to_string()));
        assert_eq!(context_rule.priority, 10);
    }

    #[test]
    fn test_supported_languages() {
        let g2p = RuleBasedG2p::new(LanguageCode::EnUs);
        let languages = g2p.supported_languages();
        assert_eq!(languages, vec![LanguageCode::EnUs]);
    }

    #[test]
    fn test_metadata() {
        let g2p = RuleBasedG2p::new(LanguageCode::EnUs);
        let metadata = g2p.metadata();
        
        assert_eq!(metadata.name, "Rule-based G2P");
        assert_eq!(metadata.version, "0.2.0");
        assert!(metadata.accuracy_scores.contains_key(&LanguageCode::EnUs));
    }
}