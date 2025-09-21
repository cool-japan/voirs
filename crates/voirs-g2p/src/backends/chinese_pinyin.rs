//! Chinese (Mandarin) G2P implementation with pinyin support.
//!
//! This module provides Chinese G2P conversion using pinyin romanization
//! and tone information for Mandarin Chinese.

use crate::{G2p, G2pMetadata, LanguageCode, Phoneme, PhoneticFeatures, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use tracing::{debug, info};

/// Chinese (Mandarin) G2P with pinyin support
pub struct ChinesePinyinG2p {
    /// Dictionary mapping Chinese characters to pinyin
    char_dict: HashMap<char, Vec<String>>,
    /// Pinyin to IPA mapping
    pinyin_map: HashMap<String, String>,
    /// Tone markers mapping
    tone_map: HashMap<char, u8>,
    /// Phoneme cache for performance
    cache: HashMap<String, Vec<Phoneme>>,
    /// Maximum cache size
    max_cache_size: usize,
    /// Include tone information in phonemes
    include_tones: bool,
}

impl ChinesePinyinG2p {
    /// Create new Chinese pinyin G2P
    pub fn new() -> Self {
        let mut g2p = Self {
            char_dict: HashMap::new(),
            pinyin_map: HashMap::new(),
            tone_map: HashMap::new(),
            cache: HashMap::new(),
            max_cache_size: 10000,
            include_tones: true,
        };

        g2p.load_tone_mapping();
        g2p.load_pinyin_ipa_mapping();
        g2p.load_common_chinese_characters();

        info!(
            "Initialized Chinese pinyin G2P with {} characters",
            g2p.char_dict.len()
        );
        g2p
    }

    /// Load tone number to tone level mapping
    fn load_tone_mapping(&mut self) {
        let mappings = [
            ('1', 1), // First tone (high level)
            ('2', 2), // Second tone (rising)
            ('3', 3), // Third tone (falling-rising)
            ('4', 4), // Fourth tone (falling)
            ('5', 0), // Neutral tone
        ];

        for (tone_char, tone_level) in mappings {
            self.tone_map.insert(tone_char, tone_level);
        }

        debug!("Loaded {} tone mappings", self.tone_map.len());
    }

    /// Load pinyin to IPA mapping
    fn load_pinyin_ipa_mapping(&mut self) {
        let mappings = [
            // Basic vowels
            ("a", "a"),
            ("o", "o"),
            ("e", "ɤ"),
            ("i", "i"),
            ("u", "u"),
            ("ü", "y"),
            // Consonants
            ("b", "p"),
            ("p", "pʰ"),
            ("m", "m"),
            ("f", "f"),
            ("d", "t"),
            ("t", "tʰ"),
            ("n", "n"),
            ("l", "l"),
            ("g", "k"),
            ("k", "kʰ"),
            ("h", "x"),
            ("j", "tɕ"),
            ("q", "tɕʰ"),
            ("x", "ɕ"),
            ("z", "ts"),
            ("c", "tsʰ"),
            ("s", "s"),
            ("zh", "tʂ"),
            ("ch", "tʂʰ"),
            ("sh", "ʂ"),
            ("r", "ɻ"),
            ("y", "j"),
            ("w", "w"),
            // Compound vowels
            ("ai", "ai"),
            ("ei", "ei"),
            ("ao", "au"),
            ("ou", "ou"),
            ("an", "an"),
            ("en", "ən"),
            ("ang", "aŋ"),
            ("eng", "əŋ"),
            ("ong", "oŋ"),
            ("er", "ɚ"),
            // i-related compounds
            ("ia", "ia"),
            ("ie", "iɛ"),
            ("iao", "iau"),
            ("iu", "iou"),
            ("ian", "iɛn"),
            ("in", "in"),
            ("iang", "iaŋ"),
            ("ing", "iŋ"),
            ("iong", "ioŋ"),
            // u-related compounds
            ("ua", "ua"),
            ("uo", "uo"),
            ("uai", "uai"),
            ("ui", "uei"),
            ("uan", "uan"),
            ("un", "uən"),
            ("uang", "uaŋ"),
            ("ueng", "uəŋ"),
            // ü-related compounds
            ("üe", "yɛ"),
            ("üan", "yɛn"),
            ("ün", "yn"),
            // Special cases
            ("zhi", "tʂɻ"),
            ("chi", "tʂʰɻ"),
            ("shi", "ʂɻ"),
            ("ri", "ɻɻ"),
            ("zi", "tsɻ"),
            ("ci", "tsʰɻ"),
            ("si", "sɻ"),
        ];

        for (pinyin, ipa) in mappings {
            self.pinyin_map.insert(pinyin.to_string(), ipa.to_string());
        }

        debug!("Loaded {} pinyin-IPA mappings", self.pinyin_map.len());
    }

    /// Load common Chinese characters with pinyin
    fn load_common_chinese_characters(&mut self) {
        let characters = [
            // Common characters and their pinyin pronunciations
            ('我', vec!["wo3".to_string()]),
            ('你', vec!["ni3".to_string()]),
            ('他', vec!["ta1".to_string()]),
            ('她', vec!["ta1".to_string()]),
            ('的', vec!["de5".to_string()]),
            ('一', vec!["yi1".to_string()]),
            ('二', vec!["er4".to_string()]),
            ('三', vec!["san1".to_string()]),
            ('四', vec!["si4".to_string()]),
            ('五', vec!["wu3".to_string()]),
            ('六', vec!["liu4".to_string()]),
            ('七', vec!["qi1".to_string()]),
            ('八', vec!["ba1".to_string()]),
            ('九', vec!["jiu3".to_string()]),
            ('十', vec!["shi2".to_string()]),
            ('百', vec!["bai3".to_string()]),
            ('千', vec!["qian1".to_string()]),
            ('万', vec!["wan4".to_string()]),
            // Common verbs
            ('是', vec!["shi4".to_string()]),
            ('有', vec!["you3".to_string()]),
            ('在', vec!["zai4".to_string()]),
            ('去', vec!["qu4".to_string()]),
            ('来', vec!["lai2".to_string()]),
            ('看', vec!["kan4".to_string()]),
            ('听', vec!["ting1".to_string()]),
            ('说', vec!["shuo1".to_string()]),
            ('做', vec!["zuo4".to_string()]),
            ('吃', vec!["chi1".to_string()]),
            ('喝', vec!["he1".to_string()]),
            ('买', vec!["mai3".to_string()]),
            ('卖', vec!["mai4".to_string()]),
            // Common adjectives
            ('大', vec!["da4".to_string()]),
            ('小', vec!["xiao3".to_string()]),
            ('好', vec!["hao3".to_string()]),
            ('坏', vec!["huai4".to_string()]),
            ('新', vec!["xin1".to_string()]),
            ('旧', vec!["jiu4".to_string()]),
            ('高', vec!["gao1".to_string()]),
            ('低', vec!["di1".to_string()]),
            ('快', vec!["kuai4".to_string()]),
            ('慢', vec!["man4".to_string()]),
            // Common nouns
            ('人', vec!["ren2".to_string()]),
            ('家', vec!["jia1".to_string()]),
            ('水', vec!["shui3".to_string()]),
            ('火', vec!["huo3".to_string()]),
            ('天', vec!["tian1".to_string()]),
            ('地', vec!["di4".to_string()]),
            ('山', vec!["shan1".to_string()]),
            ('海', vec!["hai3".to_string()]),
            ('车', vec!["che1".to_string()]),
            ('飞', vec!["fei1".to_string()]),
            ('机', vec!["ji1".to_string()]),
            ('书', vec!["shu1".to_string()]),
            ('学', vec!["xue2".to_string()]),
            ('校', vec!["xiao4".to_string()]),
            ('中', vec!["zhong1".to_string()]),
            ('国', vec!["guo2".to_string()]),
            // Time words
            ('今', vec!["jin1".to_string()]),
            ('天', vec!["tian1".to_string()]),
            ('昨', vec!["zuo2".to_string()]),
            ('明', vec!["ming2".to_string()]),
            ('年', vec!["nian2".to_string()]),
            ('月', vec!["yue4".to_string()]),
            ('日', vec!["ri4".to_string()]),
            ('时', vec!["shi2".to_string()]),
            ('分', vec!["fen1".to_string()]),
            ('秒', vec!["miao3".to_string()]),
            // Colors
            ('红', vec!["hong2".to_string()]),
            ('绿', vec!["lü4".to_string()]),
            ('蓝', vec!["lan2".to_string()]),
            ('黄', vec!["huang2".to_string()]),
            ('白', vec!["bai2".to_string()]),
            ('黑', vec!["hei1".to_string()]),
            // Greetings and polite expressions
            ('您', vec!["nin2".to_string()]),
            ('请', vec!["qing3".to_string()]),
            ('谢', vec!["xie4".to_string()]),
            ('对', vec!["dui4".to_string()]),
            ('起', vec!["qi3".to_string()]),
            ('不', vec!["bu4".to_string()]),
            ('没', vec!["mei2".to_string()]),
            ('很', vec!["hen3".to_string()]),
            ('太', vec!["tai4".to_string()]),
            ('非', vec!["fei1".to_string()]),
            ('常', vec!["chang2".to_string()]),
        ];

        for (character, pinyin_list) in characters {
            self.char_dict.insert(character, pinyin_list);
        }

        debug!(
            "Loaded {} Chinese character pronunciations",
            self.char_dict.len()
        );
    }

    /// Convert Chinese text to phonemes
    pub async fn chinese_to_phonemes(&mut self, text: &str) -> Result<Vec<Phoneme>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // Check cache first
        if let Some(cached_phonemes) = self.cache.get(text) {
            debug!("Using cached phonemes for: {}", text);
            return Ok(cached_phonemes.clone());
        }

        debug!("Converting Chinese text to phonemes: {}", text);

        let mut phonemes = Vec::new();

        // Process each character
        for ch in text.chars() {
            if let Some(pinyin_list) = self.char_dict.get(&ch) {
                // Use the first pinyin pronunciation (most common)
                if let Some(pinyin) = pinyin_list.first() {
                    let phoneme = self.convert_pinyin_to_phoneme(pinyin)?;
                    phonemes.push(phoneme);
                }
            } else {
                // Handle unknown characters
                debug!("Unknown Chinese character: {}", ch);
                match ch {
                    ' ' => phonemes.push(Phoneme::word_boundary()),
                    '，' | '。' | '！' | '？' => {
                        // Chinese punctuation - add short pause
                        let mut phoneme = Phoneme::new("");
                        phoneme.duration_ms = Some(200.0);
                        phonemes.push(phoneme);
                    }
                    _ => {
                        // Skip unknown characters
                        continue;
                    }
                }
            }
        }

        // Cache the result if reasonable size
        if phonemes.len() <= 100 && self.cache.len() < self.max_cache_size {
            self.cache.insert(text.to_string(), phonemes.clone());
        }

        debug!("Generated {} phonemes for Chinese text", phonemes.len());
        Ok(phonemes)
    }

    /// Convert pinyin string to phoneme
    fn convert_pinyin_to_phoneme(&self, pinyin: &str) -> Result<Phoneme> {
        // Extract tone from pinyin (last character should be tone number)
        let (syllable, tone) = if let Some(last_char) = pinyin.chars().last() {
            if last_char.is_ascii_digit() {
                let tone_level = self.tone_map.get(&last_char).copied().unwrap_or(0);
                let syllable = &pinyin[..pinyin.len() - 1];
                (syllable, tone_level)
            } else {
                (pinyin, 0) // No tone specified
            }
        } else {
            (pinyin, 0)
        };

        // Convert pinyin to IPA
        let ipa_symbol = self
            .pinyin_map
            .get(syllable)
            .or_else(|| {
                // Try to find partial matches for compound sounds
                self.find_best_pinyin_match(syllable)
            })
            .unwrap_or(&syllable.to_string())
            .clone();

        // Create phoneme
        let mut phoneme = Phoneme::new(ipa_symbol.clone());
        phoneme.ipa_symbol = Some(ipa_symbol.clone());
        phoneme.language_notation = Some(pinyin.to_string());
        phoneme.stress = tone;
        phoneme.confidence = 0.85; // Dictionary-based confidence

        // Set phonetic features
        if let Some(features) = self.get_chinese_phonetic_features(&ipa_symbol) {
            phoneme.phonetic_features = Some(features);
        }

        // Set duration with tone consideration
        phoneme.duration_ms = Some(self.estimate_chinese_duration(tone));

        // Add tone information if enabled
        if self.include_tones && tone > 0 {
            let mut custom_features = HashMap::new();
            custom_features.insert("tone".to_string(), tone.to_string());
            custom_features.insert("tone_name".to_string(), self.get_tone_name(tone));
            phoneme.custom_features = Some(custom_features);
        }

        Ok(phoneme)
    }

    /// Find best pinyin match for compound sounds
    fn find_best_pinyin_match(&self, syllable: &str) -> Option<&String> {
        // Try to find the longest matching prefix
        for len in (1..=syllable.len()).rev() {
            let prefix = &syllable[..len];
            if let Some(ipa) = self.pinyin_map.get(prefix) {
                return Some(ipa);
            }
        }
        None
    }

    /// Get phonetic features for Chinese phonemes
    fn get_chinese_phonetic_features(&self, ipa: &str) -> Option<PhoneticFeatures> {
        // This is a simplified mapping - in practice, you'd want more detailed features
        match ipa {
            // Vowels
            "a" => Some(PhoneticFeatures::vowel("low", "central", false)),
            "o" => Some(PhoneticFeatures::vowel("mid", "back", true)),
            "ɤ" => Some(PhoneticFeatures::vowel("mid", "back", false)),
            "i" => Some(PhoneticFeatures::vowel("high", "front", false)),
            "u" => Some(PhoneticFeatures::vowel("high", "back", true)),
            "y" => Some(PhoneticFeatures::vowel("high", "front", true)),

            // Consonants
            "p" => Some(PhoneticFeatures::consonant("plosive", "bilabial", false)),
            "pʰ" => Some(PhoneticFeatures::consonant("plosive", "bilabial", false)),
            "m" => Some(PhoneticFeatures::consonant("nasal", "bilabial", true)),
            "f" => Some(PhoneticFeatures::consonant(
                "fricative",
                "labiodental",
                false,
            )),
            "t" => Some(PhoneticFeatures::consonant("plosive", "alveolar", false)),
            "tʰ" => Some(PhoneticFeatures::consonant("plosive", "alveolar", false)),
            "n" => Some(PhoneticFeatures::consonant("nasal", "alveolar", true)),
            "l" => Some(PhoneticFeatures::consonant("lateral", "alveolar", true)),
            "k" => Some(PhoneticFeatures::consonant("plosive", "velar", false)),
            "kʰ" => Some(PhoneticFeatures::consonant("plosive", "velar", false)),
            "x" => Some(PhoneticFeatures::consonant("fricative", "velar", false)),
            "tɕ" => Some(PhoneticFeatures::consonant(
                "affricate",
                "alveolo-palatal",
                false,
            )),
            "tɕʰ" => Some(PhoneticFeatures::consonant(
                "affricate",
                "alveolo-palatal",
                false,
            )),
            "ɕ" => Some(PhoneticFeatures::consonant(
                "fricative",
                "alveolo-palatal",
                false,
            )),
            "ts" => Some(PhoneticFeatures::consonant("affricate", "alveolar", false)),
            "tsʰ" => Some(PhoneticFeatures::consonant("affricate", "alveolar", false)),
            "s" => Some(PhoneticFeatures::consonant("fricative", "alveolar", false)),
            "tʂ" => Some(PhoneticFeatures::consonant("affricate", "retroflex", false)),
            "tʂʰ" => Some(PhoneticFeatures::consonant("affricate", "retroflex", false)),
            "ʂ" => Some(PhoneticFeatures::consonant("fricative", "retroflex", false)),
            "ɻ" => Some(PhoneticFeatures::consonant(
                "approximant",
                "retroflex",
                true,
            )),
            "j" => Some(PhoneticFeatures::consonant("approximant", "palatal", true)),
            "w" => Some(PhoneticFeatures::consonant(
                "approximant",
                "labio-velar",
                true,
            )),

            _ => None,
        }
    }

    /// Get tone name from tone number
    fn get_tone_name(&self, tone: u8) -> String {
        match tone {
            1 => "high_level".to_string(),
            2 => "rising".to_string(),
            3 => "falling_rising".to_string(),
            4 => "falling".to_string(),
            _ => "neutral".to_string(),
        }
    }

    /// Estimate duration for Chinese phonemes considering tone
    fn estimate_chinese_duration(&self, tone: u8) -> f32 {
        // Base duration
        let base_duration = 180.0; // Chinese syllables are typically longer

        // Tone affects duration
        match tone {
            1 => base_duration,       // High level (neutral)
            2 => base_duration * 1.1, // Rising (slightly longer)
            3 => base_duration * 1.3, // Falling-rising (longest)
            4 => base_duration * 0.9, // Falling (shorter)
            _ => base_duration * 0.8, // Neutral (shortest)
        }
    }

    /// Enable or disable tone information
    pub fn set_include_tones(&mut self, include_tones: bool) {
        self.include_tones = include_tones;
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

impl Default for ChinesePinyinG2p {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl G2p for ChinesePinyinG2p {
    async fn to_phonemes(&self, text: &str, _lang: Option<LanguageCode>) -> Result<Vec<Phoneme>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // Clone self to get mutable access
        let mut g2p_clone = self.clone();
        g2p_clone.chinese_to_phonemes(text).await
    }

    fn supported_languages(&self) -> Vec<LanguageCode> {
        vec![LanguageCode::ZhCn]
    }

    fn metadata(&self) -> G2pMetadata {
        let mut accuracy_scores = HashMap::new();
        accuracy_scores.insert(LanguageCode::ZhCn, 0.80); // Dictionary-based accuracy

        G2pMetadata {
            name: "Chinese Pinyin G2P".to_string(),
            version: "1.0.0".to_string(),
            description: "Chinese (Mandarin) G2P with pinyin and tone support".to_string(),
            supported_languages: vec![LanguageCode::ZhCn],
            accuracy_scores,
        }
    }
}

// Implement Clone for the struct
impl Clone for ChinesePinyinG2p {
    fn clone(&self) -> Self {
        Self {
            char_dict: self.char_dict.clone(),
            pinyin_map: self.pinyin_map.clone(),
            tone_map: self.tone_map.clone(),
            cache: HashMap::new(), // Don't clone cache
            max_cache_size: self.max_cache_size,
            include_tones: self.include_tones,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_chinese_g2p_creation() {
        let g2p = ChinesePinyinG2p::new();
        assert!(!g2p.char_dict.is_empty());
        assert!(!g2p.pinyin_map.is_empty());
        assert!(!g2p.tone_map.is_empty());
    }

    #[tokio::test]
    async fn test_chinese_character_conversion() {
        let g2p = ChinesePinyinG2p::new();

        // Test simple character
        let phonemes = g2p.to_phonemes("我", None).await.unwrap();
        assert_eq!(phonemes.len(), 1);
        assert!(phonemes[0].language_notation.is_some());
        println!(
            "我 phonemes: {:?}",
            phonemes.iter().map(|p| &p.symbol).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn test_chinese_phrase_conversion() {
        let g2p = ChinesePinyinG2p::new();

        // Test phrase
        let phonemes = g2p.to_phonemes("你好", None).await.unwrap();
        assert_eq!(phonemes.len(), 2);
        println!(
            "你好 phonemes: {:?}",
            phonemes.iter().map(|p| &p.symbol).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn test_tone_information() {
        let g2p = ChinesePinyinG2p::new();

        let phonemes = g2p.to_phonemes("我", None).await.unwrap();
        assert_eq!(phonemes.len(), 1);

        // Check tone information
        assert!(phonemes[0].stress > 0); // Should have tone
        if let Some(ref features) = phonemes[0].custom_features {
            assert!(features.contains_key("tone"));
            assert!(features.contains_key("tone_name"));
        }
    }

    #[test]
    fn test_pinyin_conversion() {
        let g2p = ChinesePinyinG2p::new();

        let phoneme = g2p.convert_pinyin_to_phoneme("wo3").unwrap();
        assert_eq!(phoneme.stress, 3); // Third tone
        assert_eq!(phoneme.language_notation, Some("wo3".to_string()));
    }

    #[test]
    fn test_supported_languages() {
        let g2p = ChinesePinyinG2p::new();
        let languages = g2p.supported_languages();
        assert_eq!(languages, vec![LanguageCode::ZhCn]);
    }

    #[test]
    fn test_metadata() {
        let g2p = ChinesePinyinG2p::new();
        let metadata = g2p.metadata();

        assert_eq!(metadata.name, "Chinese Pinyin G2P");
        assert_eq!(metadata.supported_languages, vec![LanguageCode::ZhCn]);
        assert!(metadata.accuracy_scores.contains_key(&LanguageCode::ZhCn));
    }

    #[test]
    fn test_tone_duration() {
        let g2p = ChinesePinyinG2p::new();

        // Test that different tones have different durations
        let dur1 = g2p.estimate_chinese_duration(1);
        let _dur2 = g2p.estimate_chinese_duration(2);
        let dur3 = g2p.estimate_chinese_duration(3);
        let dur4 = g2p.estimate_chinese_duration(4);

        assert!(dur3 > dur1); // Third tone should be longest
        assert!(dur4 < dur1); // Fourth tone should be shorter
    }
}
