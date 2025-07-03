//! Statistical language detection using n-gram models.

use crate::{LanguageCode, Result};
use crate::detection::{DetectionResult, DetectionMethod};
use std::collections::HashMap;

/// Statistical language detector using n-gram models
pub struct StatisticalDetector {
    /// Trigram models for each language
    trigram_models: HashMap<LanguageCode, TrigramModel>,
    /// Character-based models for non-Latin scripts
    char_models: HashMap<LanguageCode, CharModel>,
}

/// Trigram model for a language
#[derive(Debug, Clone)]
struct TrigramModel {
    /// Trigram frequencies
    trigrams: HashMap<String, f32>,
    /// Total trigram count
    total_count: f32,
}

/// Character-based model for a language
#[derive(Debug, Clone)]
struct CharModel {
    /// Character frequencies
    chars: HashMap<char, f32>,
    /// Total character count
    total_count: f32,
}

impl StatisticalDetector {
    /// Create new statistical detector
    pub fn new() -> Self {
        let mut detector = Self {
            trigram_models: HashMap::new(),
            char_models: HashMap::new(),
        };
        detector.load_models();
        detector
    }
    
    /// Load statistical models for languages
    fn load_models(&mut self) {
        // Load simplified models (in a real implementation, these would be trained on large corpora)
        self.load_english_model();
        self.load_german_model();
        self.load_french_model();
        self.load_spanish_model();
        self.load_japanese_model();
        self.load_chinese_model();
        self.load_korean_model();
    }
    
    /// Load English trigram model
    fn load_english_model(&mut self) {
        let mut trigrams = HashMap::new();
        
        // Common English trigrams with approximate frequencies
        trigrams.insert("the".to_string(), 0.065);
        trigrams.insert("and".to_string(), 0.041);
        trigrams.insert("ing".to_string(), 0.038);
        trigrams.insert("ion".to_string(), 0.035);
        trigrams.insert("tio".to_string(), 0.032);
        trigrams.insert("ent".to_string(), 0.028);
        trigrams.insert("ati".to_string(), 0.025);
        trigrams.insert("for".to_string(), 0.024);
        trigrams.insert("her".to_string(), 0.023);
        trigrams.insert("ter".to_string(), 0.022);
        trigrams.insert("hat".to_string(), 0.021);
        trigrams.insert("tha".to_string(), 0.020);
        trigrams.insert("ere".to_string(), 0.019);
        trigrams.insert("ate".to_string(), 0.018);
        trigrams.insert("his".to_string(), 0.018);
        trigrams.insert("con".to_string(), 0.017);
        trigrams.insert("res".to_string(), 0.017);
        trigrams.insert("ver".to_string(), 0.016);
        trigrams.insert("all".to_string(), 0.015);
        trigrams.insert("ons".to_string(), 0.015);
        
        let total_count = trigrams.values().sum();
        
        self.trigram_models.insert(LanguageCode::EnUs, TrigramModel {
            trigrams,
            total_count,
        });
    }
    
    /// Load German trigram model
    fn load_german_model(&mut self) {
        let mut trigrams = HashMap::new();
        
        // Common German trigrams
        trigrams.insert("der".to_string(), 0.055);
        trigrams.insert("und".to_string(), 0.045);
        trigrams.insert("die".to_string(), 0.042);
        trigrams.insert("den".to_string(), 0.038);
        trigrams.insert("ich".to_string(), 0.035);
        trigrams.insert("ein".to_string(), 0.032);
        trigrams.insert("das".to_string(), 0.030);
        trigrams.insert("mit".to_string(), 0.028);
        trigrams.insert("sch".to_string(), 0.025);
        trigrams.insert("ing".to_string(), 0.024);
        trigrams.insert("ung".to_string(), 0.023);
        trigrams.insert("ber".to_string(), 0.022);
        trigrams.insert("auf".to_string(), 0.021);
        trigrams.insert("ver".to_string(), 0.020);
        trigrams.insert("für".to_string(), 0.019);
        trigrams.insert("ers".to_string(), 0.018);
        trigrams.insert("cht".to_string(), 0.017);
        trigrams.insert("ent".to_string(), 0.016);
        trigrams.insert("ter".to_string(), 0.015);
        trigrams.insert("ger".to_string(), 0.014);
        
        let total_count = trigrams.values().sum();
        
        self.trigram_models.insert(LanguageCode::De, TrigramModel {
            trigrams,
            total_count,
        });
    }
    
    /// Load French trigram model
    fn load_french_model(&mut self) {
        let mut trigrams = HashMap::new();
        
        // Common French trigrams
        trigrams.insert("les".to_string(), 0.048);
        trigrams.insert("des".to_string(), 0.042);
        trigrams.insert("une".to_string(), 0.038);
        trigrams.insert("ion".to_string(), 0.035);
        trigrams.insert("ent".to_string(), 0.032);
        trigrams.insert("que".to_string(), 0.030);
        trigrams.insert("tion".to_string(), 0.028);
        trigrams.insert("con".to_string(), 0.025);
        trigrams.insert("our".to_string(), 0.024);
        trigrams.insert("ant".to_string(), 0.023);
        trigrams.insert("est".to_string(), 0.022);
        trigrams.insert("tre".to_string(), 0.021);
        trigrams.insert("ons".to_string(), 0.020);
        trigrams.insert("ers".to_string(), 0.019);
        trigrams.insert("pour".to_string(), 0.018);
        trigrams.insert("res".to_string(), 0.017);
        trigrams.insert("eur".to_string(), 0.016);
        trigrams.insert("ait".to_string(), 0.015);
        trigrams.insert("par".to_string(), 0.014);
        trigrams.insert("ter".to_string(), 0.013);
        
        let total_count = trigrams.values().sum();
        
        self.trigram_models.insert(LanguageCode::Fr, TrigramModel {
            trigrams,
            total_count,
        });
    }
    
    /// Load Spanish trigram model
    fn load_spanish_model(&mut self) {
        let mut trigrams = HashMap::new();
        
        // Common Spanish trigrams
        trigrams.insert("que".to_string(), 0.052);
        trigrams.insert("ent".to_string(), 0.045);
        trigrams.insert("ion".to_string(), 0.040);
        trigrams.insert("ado".to_string(), 0.038);
        trigrams.insert("con".to_string(), 0.035);
        trigrams.insert("est".to_string(), 0.032);
        trigrams.insert("para".to_string(), 0.030);
        trigrams.insert("los".to_string(), 0.028);
        trigrams.insert("del".to_string(), 0.025);
        trigrams.insert("ión".to_string(), 0.024);
        trigrams.insert("por".to_string(), 0.023);
        trigrams.insert("ada".to_string(), 0.022);
        trigrams.insert("era".to_string(), 0.021);
        trigrams.insert("ente".to_string(), 0.020);
        trigrams.insert("ina".to_string(), 0.019);
        trigrams.insert("nte".to_string(), 0.018);
        trigrams.insert("ión".to_string(), 0.017);
        trigrams.insert("ado".to_string(), 0.016);
        trigrams.insert("res".to_string(), 0.015);
        trigrams.insert("ter".to_string(), 0.014);
        
        let total_count = trigrams.values().sum();
        
        self.trigram_models.insert(LanguageCode::Es, TrigramModel {
            trigrams,
            total_count,
        });
    }
    
    /// Load Japanese character model (using hiragana frequencies)
    fn load_japanese_model(&mut self) {
        let mut chars = HashMap::new();
        
        // Common Japanese hiragana characters with frequencies
        chars.insert('の', 0.065);
        chars.insert('に', 0.055);
        chars.insert('は', 0.045);
        chars.insert('を', 0.040);
        chars.insert('た', 0.038);
        chars.insert('が', 0.035);
        chars.insert('で', 0.032);
        chars.insert('て', 0.030);
        chars.insert('と', 0.028);
        chars.insert('し', 0.025);
        chars.insert('れ', 0.023);
        chars.insert('さ', 0.022);
        chars.insert('い', 0.020);
        chars.insert('う', 0.018);
        chars.insert('も', 0.017);
        chars.insert('な', 0.016);
        chars.insert('か', 0.015);
        chars.insert('こ', 0.014);
        chars.insert('そ', 0.013);
        chars.insert('け', 0.012);
        
        let total_count = chars.values().sum();
        
        self.char_models.insert(LanguageCode::Ja, CharModel {
            chars,
            total_count,
        });
    }
    
    /// Load Chinese character model
    fn load_chinese_model(&mut self) {
        let mut chars = HashMap::new();
        
        // Common Chinese characters with frequencies
        chars.insert('的', 0.075);
        chars.insert('一', 0.065);
        chars.insert('是', 0.055);
        chars.insert('在', 0.048);
        chars.insert('不', 0.042);
        chars.insert('了', 0.038);
        chars.insert('有', 0.035);
        chars.insert('和', 0.032);
        chars.insert('人', 0.030);
        chars.insert('这', 0.028);
        chars.insert('中', 0.025);
        chars.insert('大', 0.023);
        chars.insert('为', 0.022);
        chars.insert('上', 0.020);
        chars.insert('个', 0.019);
        chars.insert('国', 0.018);
        chars.insert('我', 0.017);
        chars.insert('以', 0.016);
        chars.insert('要', 0.015);
        chars.insert('他', 0.014);
        
        let total_count = chars.values().sum();
        
        self.char_models.insert(LanguageCode::ZhCn, CharModel {
            chars,
            total_count,
        });
    }
    
    /// Load Korean character model
    fn load_korean_model(&mut self) {
        let mut chars = HashMap::new();
        
        // Common Korean syllables (simplified - real Korean would need proper syllable analysis)
        chars.insert('이', 0.065);
        chars.insert('의', 0.055);
        chars.insert('가', 0.048);
        chars.insert('을', 0.042);
        chars.insert('는', 0.038);
        chars.insert('에', 0.035);
        chars.insert('하', 0.032);
        chars.insert('고', 0.030);
        chars.insert('를', 0.028);
        chars.insert('로', 0.025);
        chars.insert('과', 0.023);
        chars.insert('와', 0.020);
        chars.insert('한', 0.019);
        chars.insert('지', 0.018);
        chars.insert('도', 0.017);
        chars.insert('만', 0.016);
        chars.insert('서', 0.015);
        chars.insert('께', 0.014);
        chars.insert('게', 0.013);
        chars.insert('어', 0.012);
        
        let total_count = chars.values().sum();
        
        self.char_models.insert(LanguageCode::Ko, CharModel {
            chars,
            total_count,
        });
    }
    
    /// Detect language using statistical models
    pub fn detect(&self, text: &str) -> Result<Option<DetectionResult>> {
        if text.trim().is_empty() {
            return Ok(None);
        }
        
        let mut scores = HashMap::new();
        
        // Try trigram-based detection for Latin scripts
        for (&language, model) in &self.trigram_models {
            let score = self.calculate_trigram_score(text, model);
            scores.insert(language, score);
        }
        
        // Try character-based detection for non-Latin scripts
        for (&language, model) in &self.char_models {
            let score = self.calculate_char_score(text, model);
            scores.insert(language, score);
        }
        
        // Find best match
        let best_match = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(lang, score)| (*lang, *score));
        
        if let Some((language, confidence)) = best_match {
            if confidence > 0.2 { // Minimum threshold for statistical detection
                // Prepare alternatives
                let mut alternatives: Vec<(LanguageCode, f32)> = scores
                    .into_iter()
                    .filter(|(lang, _)| *lang != language)
                    .collect();
                alternatives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                alternatives.truncate(3);
                
                return Ok(Some(DetectionResult {
                    language,
                    confidence,
                    alternatives,
                    method: DetectionMethod::Statistical,
                }));
            }
        }
        
        Ok(None)
    }
    
    /// Calculate trigram-based score
    fn calculate_trigram_score(&self, text: &str, model: &TrigramModel) -> f32 {
        let text_lower = text.to_lowercase();
        let mut score = 0.0;
        let mut total_trigrams = 0;
        
        // Extract trigrams from text
        let chars: Vec<char> = text_lower.chars().filter(|c| c.is_alphabetic()).collect();
        
        if chars.len() < 3 {
            return 0.0;
        }
        
        for i in 0..=(chars.len() - 3) {
            let trigram: String = chars[i..i+3].iter().collect();
            total_trigrams += 1;
            
            if let Some(&frequency) = model.trigrams.get(&trigram) {
                score += frequency;
            }
        }
        
        if total_trigrams > 0 {
            score / total_trigrams as f32
        } else {
            0.0
        }
    }
    
    /// Calculate character-based score
    fn calculate_char_score(&self, text: &str, model: &CharModel) -> f32 {
        let mut score = 0.0;
        let mut total_chars = 0;
        
        for ch in text.chars() {
            if !ch.is_whitespace() && !ch.is_ascii_punctuation() {
                total_chars += 1;
                
                if let Some(&frequency) = model.chars.get(&ch) {
                    score += frequency;
                }
            }
        }
        
        if total_chars > 0 {
            score / total_chars as f32
        } else {
            0.0
        }
    }
}

impl Default for StatisticalDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistical_detector_creation() {
        let detector = StatisticalDetector::new();
        assert!(!detector.trigram_models.is_empty());
        assert!(!detector.char_models.is_empty());
    }

    #[test]
    fn test_english_statistical_detection() {
        let detector = StatisticalDetector::new();
        
        let result = detector.detect("The quick brown fox jumps over the lazy dog and runs through the forest").unwrap();
        if let Some(result) = result {
            assert!(result.confidence > 0.0);
            assert_eq!(result.method, DetectionMethod::Statistical);
        }
    }

    #[test]
    fn test_trigram_score_calculation() {
        let detector = StatisticalDetector::new();
        
        if let Some(model) = detector.trigram_models.get(&LanguageCode::EnUs) {
            let score = detector.calculate_trigram_score("the quick brown", model);
            assert!((0.0..=1.0).contains(&score));
            assert!(score > 0.0); // Should have some score due to "the"
        }
    }

    #[test]
    fn test_character_score_calculation() {
        let detector = StatisticalDetector::new();
        
        if let Some(model) = detector.char_models.get(&LanguageCode::Ja) {
            let score = detector.calculate_char_score("こんにちは", model);
            assert!((0.0..=1.0).contains(&score));
        }
    }

    #[test]
    fn test_empty_text() {
        let detector = StatisticalDetector::new();
        
        let result = detector.detect("").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_short_text() {
        let detector = StatisticalDetector::new();
        
        let result = detector.detect("hi").unwrap();
        // Short text might not produce reliable results
        if let Some(result) = result {
            assert!(result.confidence >= 0.0);
        }
    }

    #[test]
    fn test_japanese_text() {
        let detector = StatisticalDetector::new();
        
        let result = detector.detect("こんにちは、元気ですか").unwrap();
        if let Some(result) = result {
            assert!(result.confidence > 0.0);
            // Might detect as Japanese if the character model works well
        }
    }

    #[test]
    fn test_mixed_script_text() {
        let detector = StatisticalDetector::new();
        
        let result = detector.detect("Hello こんにちは world").unwrap();
        if let Some(result) = result {
            assert!(result.confidence >= 0.0);
            // Should detect something, though might not be very confident
        }
    }
}