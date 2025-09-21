//! Dictionary-based Japanese G2P implementation.
//!
//! This module provides a comprehensive Japanese G2P implementation using
//! a built-in phonetic dictionary for common Japanese words and fallback
//! romanization rules.

use crate::{G2p, G2pMetadata, LanguageCode, Phoneme, PhoneticFeatures, Result, SyllablePosition};
use async_trait::async_trait;
use std::collections::HashMap;
use tracing::{debug, info};

/// Dictionary-based Japanese G2P implementation
pub struct JapaneseDictG2p {
    /// Dictionary mapping Japanese words to phonemes
    word_dict: HashMap<String, Vec<String>>,
    /// Hiragana to phoneme mapping
    hiragana_map: HashMap<char, String>,
    /// Katakana to phoneme mapping  
    katakana_map: HashMap<char, String>,
    /// Phoneme cache for performance
    cache: HashMap<String, Vec<Phoneme>>,
    /// Maximum cache size
    max_cache_size: usize,
}

impl JapaneseDictG2p {
    /// Create new Japanese dictionary-based G2P
    pub fn new() -> Self {
        let mut g2p = Self {
            word_dict: HashMap::new(),
            hiragana_map: HashMap::new(),
            katakana_map: HashMap::new(),
            cache: HashMap::new(),
            max_cache_size: 10000,
        };

        g2p.load_hiragana_mapping();
        g2p.load_katakana_mapping();
        g2p.load_common_japanese_words();

        info!(
            "Initialized Japanese dictionary G2P with {} words",
            g2p.word_dict.len()
        );
        g2p
    }

    /// Load hiragana to phoneme mapping
    fn load_hiragana_mapping(&mut self) {
        let mappings = [
            // Basic vowels
            ('あ', "a"),
            ('い', "i"),
            ('う', "ɯ"),
            ('え', "e"),
            ('お', "o"),
            // K series
            ('か', "ka"),
            ('き', "ki"),
            ('く', "kɯ"),
            ('け', "ke"),
            ('こ', "ko"),
            ('が', "ɡa"),
            ('ぎ', "ɡi"),
            ('ぐ', "ɡɯ"),
            ('げ', "ɡe"),
            ('ご', "ɡo"),
            // S series
            ('さ', "sa"),
            ('し', "ʃi"),
            ('す', "sɯ"),
            ('せ', "se"),
            ('そ', "so"),
            ('ざ', "za"),
            ('じ', "dʒi"),
            ('ず', "zɯ"),
            ('ぜ', "ze"),
            ('ぞ', "zo"),
            // T series
            ('た', "ta"),
            ('ち', "tʃi"),
            ('つ', "tsɯ"),
            ('て', "te"),
            ('と', "to"),
            ('だ', "da"),
            ('ぢ', "dʒi"),
            ('づ', "dzɯ"),
            ('で', "de"),
            ('ど', "do"),
            // N series
            ('な', "na"),
            ('に', "ni"),
            ('ぬ', "nɯ"),
            ('ね', "ne"),
            ('の', "no"),
            // H series
            ('は', "ha"),
            ('ひ', "hi"),
            ('ふ', "ɸɯ"),
            ('へ', "he"),
            ('ほ', "ho"),
            ('ば', "ba"),
            ('び', "bi"),
            ('ぶ', "bɯ"),
            ('べ', "be"),
            ('ぼ', "bo"),
            ('ぱ', "pa"),
            ('ぴ', "pi"),
            ('ぷ', "pɯ"),
            ('ぺ', "pe"),
            ('ぽ', "po"),
            // M series
            ('ま', "ma"),
            ('み', "mi"),
            ('む', "mɯ"),
            ('め', "me"),
            ('も', "mo"),
            // Y series
            ('や', "ja"),
            ('ゆ', "jɯ"),
            ('よ', "jo"),
            // R series
            ('ら', "ɾa"),
            ('り', "ɾi"),
            ('る', "ɾɯ"),
            ('れ', "ɾe"),
            ('ろ', "ɾo"),
            // W series
            ('わ', "wa"),
            ('ゐ', "wi"),
            ('ゑ', "we"),
            ('を', "wo"),
            // Special
            ('ん', "ɴ"), // Syllabic nasal
            ('っ', "ʔ"), // Sokuon (glottal stop)
            // Long vowel mark
            ('ー', "ː"),
        ];

        for (hiragana, phoneme) in mappings {
            self.hiragana_map.insert(hiragana, phoneme.to_string());
        }

        debug!("Loaded {} hiragana mappings", self.hiragana_map.len());
    }

    /// Load katakana to phoneme mapping
    fn load_katakana_mapping(&mut self) {
        let mappings = [
            // Basic vowels
            ('ア', "a"),
            ('イ', "i"),
            ('ウ', "ɯ"),
            ('エ', "e"),
            ('オ', "o"),
            // K series
            ('カ', "ka"),
            ('キ', "ki"),
            ('ク', "kɯ"),
            ('ケ', "ke"),
            ('コ', "ko"),
            ('ガ', "ɡa"),
            ('ギ', "ɡi"),
            ('グ', "ɡɯ"),
            ('ゲ', "ɡe"),
            ('ゴ', "ɡo"),
            // S series
            ('サ', "sa"),
            ('シ', "ʃi"),
            ('ス', "sɯ"),
            ('セ', "se"),
            ('ソ', "so"),
            ('ザ', "za"),
            ('ジ', "dʒi"),
            ('ズ', "zɯ"),
            ('ゼ', "ze"),
            ('ゾ', "zo"),
            // T series
            ('タ', "ta"),
            ('チ', "tʃi"),
            ('ツ', "tsɯ"),
            ('テ', "te"),
            ('ト', "to"),
            ('ダ', "da"),
            ('ヂ', "dʒi"),
            ('ヅ', "dzɯ"),
            ('デ', "de"),
            ('ド', "do"),
            // N series
            ('ナ', "na"),
            ('ニ', "ni"),
            ('ヌ', "nɯ"),
            ('ネ', "ne"),
            ('ノ', "no"),
            // H series
            ('ハ', "ha"),
            ('ヒ', "hi"),
            ('フ', "ɸɯ"),
            ('ヘ', "he"),
            ('ホ', "ho"),
            ('バ', "ba"),
            ('ビ', "bi"),
            ('ブ', "bɯ"),
            ('ベ', "be"),
            ('ボ', "bo"),
            ('パ', "pa"),
            ('ピ', "pi"),
            ('プ', "pɯ"),
            ('ペ', "pe"),
            ('ポ', "po"),
            // M series
            ('マ', "ma"),
            ('ミ', "mi"),
            ('ム', "mɯ"),
            ('メ', "me"),
            ('モ', "mo"),
            // Y series
            ('ヤ', "ja"),
            ('ユ', "jɯ"),
            ('ヨ', "jo"),
            // R series
            ('ラ', "ɾa"),
            ('リ', "ɾi"),
            ('ル', "ɾɯ"),
            ('レ', "ɾe"),
            ('ロ', "ɾo"),
            // W series
            ('ワ', "wa"),
            ('ヰ', "wi"),
            ('ヱ', "we"),
            ('ヲ', "wo"),
            // Special
            ('ン', "ɴ"), // Syllabic nasal
            ('ッ', "ʔ"), // Sokuon (glottal stop)
            // Long vowel mark
            ('ー', "ː"),
            // Extended katakana for foreign words
            ('ヴ', "vɯ"), // V sound
        ];

        for (katakana, phoneme) in mappings {
            self.katakana_map.insert(katakana, phoneme.to_string());
        }

        debug!("Loaded {} katakana mappings", self.katakana_map.len());
    }

    /// Load common Japanese words with known pronunciations
    fn load_common_japanese_words(&mut self) {
        let words = [
            // Common greetings and expressions
            ("こんにちは", vec!["ko", "ɴ", "ni", "tʃi", "wa"]),
            ("こんばんは", vec!["ko", "ɴ", "ba", "ɴ", "wa"]),
            ("おはよう", vec!["o", "ha", "jo", "ː"]),
            ("ありがとう", vec!["a", "ɾi", "ɡa", "to", "ː"]),
            ("さようなら", vec!["sa", "jo", "ː", "na", "ɾa"]),
            ("はじめまして", vec!["ha", "dʒi", "me", "ma", "ʃi", "te"]),
            ("よろしく", vec!["jo", "ɾo", "ʃi", "kɯ"]),
            ("すみません", vec!["sɯ", "mi", "ma", "se", "ɴ"]),
            // Numbers
            ("いち", vec!["i", "tʃi"]),
            ("に", vec!["ni"]),
            ("さん", vec!["sa", "ɴ"]),
            ("よん", vec!["jo", "ɴ"]),
            ("ご", vec!["ɡo"]),
            ("ろく", vec!["ɾo", "kɯ"]),
            ("なな", vec!["na", "na"]),
            ("はち", vec!["ha", "tʃi"]),
            ("きゅう", vec!["kj", "ɯː"]),
            ("じゅう", vec!["dʒ", "ɯː"]),
            // Common adjectives
            ("おおきい", vec!["oː", "ki", "ː"]),
            ("ちいさい", vec!["tʃi", "ː", "sa", "i"]),
            ("あたらしい", vec!["a", "ta", "ɾa", "ʃi", "ː"]),
            ("ふるい", vec!["ɸɯ", "ɾɯ", "i"]),
            ("いい", vec!["i", "ː"]),
            ("わるい", vec!["wa", "ɾɯ", "i"]),
            ("たかい", vec!["ta", "ka", "i"]),
            ("やすい", vec!["ja", "sɯ", "i"]),
            // Common verbs
            ("いく", vec!["i", "kɯ"]),
            ("くる", vec!["kɯ", "ɾɯ"]),
            ("する", vec!["sɯ", "ɾɯ"]),
            ("みる", vec!["mi", "ɾɯ"]),
            ("きく", vec!["ki", "kɯ"]),
            ("はなす", vec!["ha", "na", "sɯ"]),
            ("たべる", vec!["ta", "be", "ɾɯ"]),
            ("のむ", vec!["no", "mɯ"]),
            ("よむ", vec!["jo", "mɯ"]),
            ("かく", vec!["ka", "kɯ"]),
            // Common nouns
            ("ひと", vec!["hi", "to"]),
            ("いえ", vec!["i", "e"]),
            ("みず", vec!["mi", "zɯ"]),
            ("たべもの", vec!["ta", "be", "mo", "no"]),
            ("でんしゃ", vec!["de", "ɴ", "ʃa"]),
            ("がっこう", vec!["ɡa", "ʔ", "ko", "ː"]),
            ("しごと", vec!["ʃi", "ɡo", "to"]),
            ("ともだち", vec!["to", "mo", "da", "tʃi"]),
            ("かぞく", vec!["ka", "zo", "kɯ"]),
            ("せんせい", vec!["se", "ɴ", "se", "ː"]),
            // Time expressions
            ("きょう", vec!["kj", "oː"]),
            ("あした", vec!["a", "ʃi", "ta"]),
            ("きのう", vec!["ki", "no", "ː"]),
            ("いま", vec!["i", "ma"]),
            ("あさ", vec!["a", "sa"]),
            ("ひる", vec!["hi", "ɾɯ"]),
            ("よる", vec!["jo", "ɾɯ"]),
            // Places
            ("とうきょう", vec!["to", "ː", "kj", "oː"]),
            ("おおさか", vec!["oː", "sa", "ka"]),
            ("にほん", vec!["ni", "ho", "ɴ"]),
            ("アメリカ", vec!["a", "me", "ɾi", "ka"]),
            // Common particles and function words
            ("です", vec!["de", "sɯ"]),
            ("だった", vec!["da", "ʔ", "ta"]),
            ("である", vec!["de", "a", "ɾɯ"]),
            ("ではない", vec!["de", "wa", "na", "i"]),
        ];

        for (word, phonemes) in words {
            self.word_dict.insert(
                word.to_string(),
                phonemes.iter().map(|s| s.to_string()).collect(),
            );
        }

        debug!(
            "Loaded {} Japanese word pronunciations",
            self.word_dict.len()
        );
    }

    /// Convert Japanese text to phonemes
    pub async fn japanese_to_phonemes(&mut self, text: &str) -> Result<Vec<Phoneme>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // Check cache first
        if let Some(cached_phonemes) = self.cache.get(text) {
            debug!("Using cached phonemes for: {}", text);
            return Ok(cached_phonemes.clone());
        }

        debug!("Converting Japanese text to phonemes: {}", text);

        let mut phonemes = Vec::new();

        // First try word dictionary lookup
        if let Some(word_phonemes) = self.word_dict.get(text) {
            for phoneme_str in word_phonemes {
                phonemes.push(self.create_japanese_phoneme(phoneme_str));
            }
        } else {
            // Character-by-character conversion
            let chars: Vec<char> = text.chars().collect();
            let mut i = 0;

            while i < chars.len() {
                let ch = chars[i];

                // Handle multi-character combinations first
                if i + 1 < chars.len() {
                    let two_char = format!("{}{}", ch, chars[i + 1]);
                    if let Some(phoneme) = self.handle_multi_character(&two_char) {
                        phonemes.push(self.create_japanese_phoneme(&phoneme));
                        i += 2;
                        continue;
                    }
                }

                // Single character conversion
                if let Some(phoneme_str) = self.convert_single_character(ch) {
                    phonemes.push(self.create_japanese_phoneme(&phoneme_str));
                }

                i += 1;
            }
        }

        // Cache the result if reasonable size
        if phonemes.len() <= 100 && self.cache.len() < self.max_cache_size {
            self.cache.insert(text.to_string(), phonemes.clone());
        }

        debug!("Generated {} phonemes for Japanese text", phonemes.len());
        Ok(phonemes)
    }

    /// Handle multi-character combinations
    fn handle_multi_character(&self, two_char: &str) -> Option<String> {
        match two_char {
            // Long vowels
            "あー" => Some("aː".to_string()),
            "いー" => Some("iː".to_string()),
            "うー" => Some("ɯː".to_string()),
            "えー" => Some("eː".to_string()),
            "おー" => Some("oː".to_string()),

            // Palatalized consonants
            "きゃ" => Some("kja".to_string()),
            "きゅ" => Some("kjɯ".to_string()),
            "きょ" => Some("kjo".to_string()),
            "しゃ" => Some("ʃa".to_string()),
            "しゅ" => Some("ʃɯ".to_string()),
            "しょ" => Some("ʃo".to_string()),
            "ちゃ" => Some("tʃa".to_string()),
            "ちゅ" => Some("tʃɯ".to_string()),
            "ちょ" => Some("tʃo".to_string()),
            "にゃ" => Some("nja".to_string()),
            "にゅ" => Some("njɯ".to_string()),
            "にょ" => Some("njo".to_string()),
            "ひゃ" => Some("hja".to_string()),
            "ひゅ" => Some("hjɯ".to_string()),
            "ひょ" => Some("hjo".to_string()),
            "みゃ" => Some("mja".to_string()),
            "みゅ" => Some("mjɯ".to_string()),
            "みょ" => Some("mjo".to_string()),
            "りゃ" => Some("ɾja".to_string()),
            "りゅ" => Some("ɾjɯ".to_string()),
            "りょ" => Some("ɾjo".to_string()),

            // Extended katakana for foreign words
            "ファ" => Some("ɸa".to_string()),
            "フィ" => Some("ɸi".to_string()),
            "フェ" => Some("ɸe".to_string()),
            "フォ" => Some("ɸo".to_string()),
            "ティ" => Some("ti".to_string()),
            "ディ" => Some("di".to_string()),
            "トゥ" => Some("tɯ".to_string()),
            "ドゥ" => Some("dɯ".to_string()),
            "シェ" => Some("ʃe".to_string()),
            "ジェ" => Some("dʒe".to_string()),
            "チェ" => Some("tʃe".to_string()),

            _ => None,
        }
    }

    /// Convert single character to phoneme
    fn convert_single_character(&self, ch: char) -> Option<String> {
        // Try hiragana first
        if let Some(phoneme) = self.hiragana_map.get(&ch) {
            return Some(phoneme.clone());
        }

        // Try katakana
        if let Some(phoneme) = self.katakana_map.get(&ch) {
            return Some(phoneme.clone());
        }

        // Handle space and punctuation
        match ch {
            ' ' => Some(" ".to_string()),
            '。' => Some("".to_string()), // Period (silence)
            '、' => Some("".to_string()), // Comma (short pause)
            '？' => Some("".to_string()), // Question mark
            '！' => Some("".to_string()), // Exclamation mark
            _ => {
                debug!("Unknown character: {}", ch);
                None
            }
        }
    }

    /// Create Japanese phoneme with appropriate features
    fn create_japanese_phoneme(&self, symbol: &str) -> Phoneme {
        let mut phoneme = Phoneme::new(symbol);

        // Set IPA symbol
        phoneme.ipa_symbol = Some(symbol.to_string());

        // Set phonetic features based on phoneme type
        if let Some(features) = self.get_japanese_phonetic_features(symbol) {
            phoneme.phonetic_features = Some(features);
        }

        // Set syllable position
        phoneme.syllable_position = self.determine_syllable_position(symbol);

        // Set duration estimate
        phoneme.duration_ms = Some(self.estimate_japanese_duration(symbol));

        // Set confidence
        phoneme.confidence = 0.85; // Dictionary-based confidence

        phoneme
    }

    /// Get phonetic features for Japanese phonemes
    fn get_japanese_phonetic_features(&self, symbol: &str) -> Option<PhoneticFeatures> {
        match symbol {
            // Vowels
            "a" => Some(PhoneticFeatures::vowel("low", "central", false)),
            "i" => Some(PhoneticFeatures::vowel("high", "front", false)),
            "ɯ" => Some(PhoneticFeatures::vowel("high", "back", false)), // Unrounded
            "e" => Some(PhoneticFeatures::vowel("mid", "front", false)),
            "o" => Some(PhoneticFeatures::vowel("mid", "back", true)),

            // Consonants
            "k" => Some(PhoneticFeatures::consonant("plosive", "velar", false)),
            "ɡ" => Some(PhoneticFeatures::consonant("plosive", "velar", true)),
            "s" => Some(PhoneticFeatures::consonant("fricative", "alveolar", false)),
            "z" => Some(PhoneticFeatures::consonant("fricative", "alveolar", true)),
            "t" => Some(PhoneticFeatures::consonant("plosive", "alveolar", false)),
            "d" => Some(PhoneticFeatures::consonant("plosive", "alveolar", true)),
            "n" => Some(PhoneticFeatures::consonant("nasal", "alveolar", true)),
            "h" => Some(PhoneticFeatures::consonant("fricative", "glottal", false)),
            "b" => Some(PhoneticFeatures::consonant("plosive", "bilabial", true)),
            "p" => Some(PhoneticFeatures::consonant("plosive", "bilabial", false)),
            "m" => Some(PhoneticFeatures::consonant("nasal", "bilabial", true)),
            "j" => Some(PhoneticFeatures::consonant("approximant", "palatal", true)),
            "ɾ" => Some(PhoneticFeatures::consonant("tap", "alveolar", true)),
            "w" => Some(PhoneticFeatures::consonant(
                "approximant",
                "labio-velar",
                true,
            )),
            "ɴ" => Some(PhoneticFeatures::consonant("nasal", "uvular", true)),
            "ʔ" => Some(PhoneticFeatures::consonant("plosive", "glottal", false)),
            "ʃ" => Some(PhoneticFeatures::consonant(
                "fricative",
                "post-alveolar",
                false,
            )),
            "tʃ" => Some(PhoneticFeatures::consonant(
                "affricate",
                "post-alveolar",
                false,
            )),
            "dʒ" => Some(PhoneticFeatures::consonant(
                "affricate",
                "post-alveolar",
                true,
            )),
            "ts" => Some(PhoneticFeatures::consonant("affricate", "alveolar", false)),
            "ɸ" => Some(PhoneticFeatures::consonant("fricative", "bilabial", false)),

            _ => None,
        }
    }

    /// Determine syllable position for Japanese phonemes
    fn determine_syllable_position(&self, symbol: &str) -> SyllablePosition {
        match symbol {
            // Vowels are typically nucleus
            "a" | "i" | "ɯ" | "e" | "o" => SyllablePosition::Nucleus,
            // Special markers
            "ʔ" | "ɴ" => SyllablePosition::Coda,
            " " => SyllablePosition::Standalone,
            // Consonants are typically onset
            _ => SyllablePosition::Onset,
        }
    }

    /// Estimate duration for Japanese phonemes (mora-based timing)
    fn estimate_japanese_duration(&self, symbol: &str) -> f32 {
        match symbol {
            // Long vowels
            s if s.contains('ː') => 250.0,
            // Special sounds
            "ʔ" => 80.0,  // Sokuon (short)
            "ɴ" => 120.0, // Syllabic nasal
            " " => 150.0, // Word boundary pause
            // Regular mora
            _ => 130.0,
        }
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache.len(), self.max_cache_size)
    }
}

impl Default for JapaneseDictG2p {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl G2p for JapaneseDictG2p {
    async fn to_phonemes(&self, text: &str, _lang: Option<LanguageCode>) -> Result<Vec<Phoneme>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // Note: We need to clone self to get mutable access
        // This is a design limitation we'll accept for now
        let mut g2p_clone = self.clone();
        g2p_clone.japanese_to_phonemes(text).await
    }

    fn supported_languages(&self) -> Vec<LanguageCode> {
        vec![LanguageCode::Ja]
    }

    fn metadata(&self) -> G2pMetadata {
        let mut accuracy_scores = HashMap::new();
        accuracy_scores.insert(LanguageCode::Ja, 0.82); // Dictionary-based accuracy

        G2pMetadata {
            name: "Japanese Dictionary G2P".to_string(),
            version: "1.0.0".to_string(),
            description: "Dictionary-based Japanese G2P with hiragana/katakana support".to_string(),
            supported_languages: vec![LanguageCode::Ja],
            accuracy_scores,
        }
    }
}

// Implement Clone for the struct
impl Clone for JapaneseDictG2p {
    fn clone(&self) -> Self {
        Self {
            word_dict: self.word_dict.clone(),
            hiragana_map: self.hiragana_map.clone(),
            katakana_map: self.katakana_map.clone(),
            cache: HashMap::new(), // Don't clone cache
            max_cache_size: self.max_cache_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_japanese_dict_creation() {
        let g2p = JapaneseDictG2p::new();
        assert!(!g2p.word_dict.is_empty());
        assert!(!g2p.hiragana_map.is_empty());
        assert!(!g2p.katakana_map.is_empty());
    }

    #[tokio::test]
    async fn test_hiragana_conversion() {
        let g2p = JapaneseDictG2p::new();

        // Test simple hiragana
        let phonemes = g2p.to_phonemes("あ", None).await.unwrap();
        assert_eq!(phonemes.len(), 1);
        assert_eq!(phonemes[0].symbol, "a");

        // Test word
        let phonemes = g2p.to_phonemes("こんにちは", None).await.unwrap();
        assert!(!phonemes.is_empty());
        println!(
            "こんにちは phonemes: {:?}",
            phonemes.iter().map(|p| &p.symbol).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn test_katakana_conversion() {
        let g2p = JapaneseDictG2p::new();

        // Test katakana
        let phonemes = g2p.to_phonemes("アメリカ", None).await.unwrap();
        assert!(!phonemes.is_empty());
        println!(
            "アメリカ phonemes: {:?}",
            phonemes.iter().map(|p| &p.symbol).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn test_mixed_script() {
        let g2p = JapaneseDictG2p::new();

        // Test mixed hiragana/katakana
        let phonemes = g2p.to_phonemes("こんにちはアメリカ", None).await.unwrap();
        assert!(!phonemes.is_empty());
    }

    #[test]
    fn test_phonetic_features() {
        let g2p = JapaneseDictG2p::new();

        let features = g2p.get_japanese_phonetic_features("a").unwrap();
        assert_eq!(features.manner, Some("vowel".to_string()));
        assert_eq!(features.height, Some("low".to_string()));

        let features = g2p.get_japanese_phonetic_features("k").unwrap();
        assert_eq!(features.manner, Some("plosive".to_string()));
        assert_eq!(features.place, Some("velar".to_string()));
    }

    #[test]
    fn test_supported_languages() {
        let g2p = JapaneseDictG2p::new();
        let languages = g2p.supported_languages();
        assert_eq!(languages, vec![LanguageCode::Ja]);
    }

    #[test]
    fn test_metadata() {
        let g2p = JapaneseDictG2p::new();
        let metadata = g2p.metadata();

        assert_eq!(metadata.name, "Japanese Dictionary G2P");
        assert_eq!(metadata.supported_languages, vec![LanguageCode::Ja]);
        assert!(metadata.accuracy_scores.contains_key(&LanguageCode::Ja));
    }
}
