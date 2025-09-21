//! Rule-based G2P implementation for English and other languages.

use crate::preprocessing::TextPreprocessor;
use crate::{G2p, G2pError, G2pMetadata, LanguageCode, Phoneme, PhoneticFeatures, Result};
use async_trait::async_trait;
use regex::Regex;
use std::collections::HashMap;

/// A G2P transformation rule
#[derive(Debug, Clone)]
pub struct G2pRule {
    /// Pattern to match (regex)
    pub pattern: Regex,
    /// Replacement phonemes
    pub replacement: String,
    /// Rule priority (higher = applied first)
    pub priority: i32,
    /// Context requirements (optional)
    pub context: Option<String>,
}

impl G2pRule {
    /// Create new G2P rule
    pub fn new(pattern: &str, replacement: &str, priority: i32) -> Result<Self> {
        let regex = Regex::new(pattern).map_err(|e| {
            G2pError::ConfigError(format!("Invalid regex pattern '{pattern}': {e}"))
        })?;

        Ok(Self {
            pattern: regex,
            replacement: replacement.to_string(),
            priority,
            context: None,
        })
    }

    /// Create rule with context
    pub fn with_context(
        pattern: &str,
        replacement: &str,
        priority: i32,
        context: &str,
    ) -> Result<Self> {
        let mut rule = Self::new(pattern, replacement, priority)?;
        rule.context = Some(context.to_string());
        Ok(rule)
    }
}

/// Rule-based G2P converter for English
pub struct EnglishRuleG2p {
    rules: Vec<G2pRule>,
    word_dict: HashMap<String, Vec<String>>,
    preprocessor: TextPreprocessor,
}

impl EnglishRuleG2p {
    /// Create new English rule-based G2P
    pub fn new() -> Result<Self> {
        let mut converter = Self {
            rules: Vec::new(),
            word_dict: HashMap::new(),
            preprocessor: TextPreprocessor::new(LanguageCode::EnUs),
        };

        converter.load_english_rules()?;
        converter.load_common_words();

        Ok(converter)
    }

    /// Create phoneme with IPA symbol and phonetic features
    #[allow(dead_code)]
    fn create_phoneme_with_features(
        symbol: &str,
        ipa: &str,
        features: PhoneticFeatures,
    ) -> Phoneme {
        Phoneme::full(
            symbol.to_string(),
            Some(ipa.to_string()),
            None, // language_notation
            0,    // stress
            crate::SyllablePosition::Standalone,
            None, // duration_ms
            1.0,  // confidence
            Some(features),
            None,  // custom_features
            false, // is_word_boundary
            false, // is_syllable_boundary
        )
    }

    /// Create vowel phoneme with IPA and features
    #[allow(dead_code)]
    fn create_vowel(
        symbol: &str,
        ipa: &str,
        height: &str,
        frontness: &str,
        rounded: bool,
    ) -> Phoneme {
        let features = PhoneticFeatures::vowel(height, frontness, rounded);
        Self::create_phoneme_with_features(symbol, ipa, features)
    }

    /// Create consonant phoneme with IPA and features
    #[allow(dead_code)]
    fn create_consonant(
        symbol: &str,
        ipa: &str,
        manner: &str,
        place: &str,
        voiced: bool,
    ) -> Phoneme {
        let features = PhoneticFeatures::consonant(manner, place, voiced);
        Self::create_phoneme_with_features(symbol, ipa, features)
    }

    /// Load English pronunciation rules
    fn load_english_rules(&mut self) -> Result<()> {
        // Magic-e patterns (highest priority) - use specific word patterns
        self.add_rule("ake$", "eɪk", 100)?; // cake, make, take
        self.add_rule("ame$", "eɪm", 100)?; // came, name, game
        self.add_rule("ane$", "eɪn", 100)?; // cane, lane, plane
        self.add_rule("ape$", "eɪp", 100)?; // tape, cape, grape
        self.add_rule("ate$", "eɪt", 100)?; // late, gate, state
        self.add_rule("ave$", "eɪv", 100)?; // cave, wave, save

        self.add_rule("ike$", "aɪk", 100)?; // bike, like, mike
        self.add_rule("ime$", "aɪm", 100)?; // time, lime, dime
        self.add_rule("ine$", "aɪn", 100)?; // line, mine, fine
        self.add_rule("ipe$", "aɪp", 100)?; // pipe, ripe, wipe
        self.add_rule("ite$", "aɪt", 100)?; // bite, kite, white
        self.add_rule("ive$", "aɪv", 100)?; // five, hive, drive

        self.add_rule("oke$", "oʊk", 100)?; // joke, poke, smoke
        self.add_rule("ome$", "oʊm", 100)?; // home, come, dome
        self.add_rule("one$", "oʊn", 100)?; // bone, cone, phone
        self.add_rule("ope$", "oʊp", 100)?; // hope, rope, scope
        self.add_rule("ote$", "oʊt", 100)?; // note, vote, quote
        self.add_rule("ove$", "oʊv", 100)?; // dove, cove, grove

        self.add_rule("uke$", "juːk", 100)?; // duke, nuke, fluke
        self.add_rule("ume$", "juːm", 100)?; // fume, plume
        self.add_rule("une$", "juːn", 100)?; // tune, dune, prune
        self.add_rule("upe$", "juːp", 100)?; // (rare pattern)
        self.add_rule("ute$", "juːt", 100)?; // cute, mute, flute

        // Consonant digraphs (high priority)
        self.add_rule("ch", "tʃ", 90)?; // ch -> /tʃ/ (chair, much)
        self.add_rule("sh", "ʃ", 90)?; // sh -> /ʃ/ (ship, wash)
        self.add_rule("th", "θ", 90)?; // th -> /θ/ (think, math)
        self.add_rule("ph", "f", 90)?; // ph -> /f/ (phone, graph)
        self.add_rule("gh", "f", 85)?; // gh -> /f/ (laugh, rough) - sometimes silent
        self.add_rule("ck", "k", 90)?; // ck -> /k/ (back, lock)
        self.add_rule("ng", "ŋ", 90)?; // ng -> /ŋ/ (sing, long)

        // Vowel combinations (medium-high priority)
        self.add_rule("ea", "iː", 80)?; // ea -> /iː/ (eat, read)
        self.add_rule("ee", "iː", 85)?; // ee -> /iː/ (see, tree)
        self.add_rule("oo", "uː", 80)?; // oo -> /uː/ (book, moon)
        self.add_rule("ai", "eɪ", 80)?; // ai -> /eɪ/ (rain, pain)
        self.add_rule("ay", "eɪ", 80)?; // ay -> /eɪ/ (day, play)
        self.add_rule("oi", "ɔɪ", 80)?; // oi -> /ɔɪ/ (oil, coin)
        self.add_rule("oy", "ɔɪ", 80)?; // oy -> /ɔɪ/ (boy, toy)
        self.add_rule("ou", "aʊ", 80)?; // ou -> /aʊ/ (out, house)
        self.add_rule("ow", "aʊ", 75)?; // ow -> /aʊ/ or /oʊ/ (cow, how)
        self.add_rule("au", "ɔː", 80)?; // au -> /ɔː/ (caught, taught)
        self.add_rule("aw", "ɔː", 80)?; // aw -> /ɔː/ (saw, law)

        // R-controlled vowels (medium priority)
        self.add_rule("ar", "ɑːr", 75)?; // ar -> /ɑːr/ (car, far)
        self.add_rule("er", "ɜːr", 75)?; // er -> /ɜːr/ (her, term)
        self.add_rule("ir", "ɜːr", 75)?; // ir -> /ɜːr/ (bird, girl)
        self.add_rule("or", "ɔːr", 75)?; // or -> /ɔːr/ (for, corn)
        self.add_rule("ur", "ɜːr", 75)?; // ur -> /ɜːr/ (hurt, turn)

        // Single vowels (lower priority)
        self.add_rule("a", "æ", 50)?; // a -> /æ/ (cat, bad)
        self.add_rule("e", "ɛ", 50)?; // e -> /ɛ/ (bed, red)
        self.add_rule("i", "ɪ", 50)?; // i -> /ɪ/ (sit, bit)
        self.add_rule("o", "ɒ", 50)?; // o -> /ɒ/ (hot, dog)
        self.add_rule("u", "ʌ", 50)?; // u -> /ʌ/ (cut, but)
        self.add_rule("y", "ɪ", 45)?; // y -> /ɪ/ (gym, myth)

        // Consonants (medium priority)
        self.add_rule("b", "b", 60)?;

        // C rules - specific patterns for soft c
        self.add_rule("ce", "s", 65)?; // ce -> /s/ (cell, nice)
        self.add_rule("ci", "s", 65)?; // ci -> /s/ (city, nice)
        self.add_rule("cy", "s", 65)?; // cy -> /s/ (cycle, icy)
        self.add_rule("c", "k", 60)?; // c -> /k/ (cat, cow)

        self.add_rule("d", "d", 60)?;
        self.add_rule("f", "f", 60)?;

        // G rules - specific patterns for soft g
        self.add_rule("ge", "dʒ", 65)?; // ge -> /dʒ/ (age, page)
        self.add_rule("gi", "dʒ", 65)?; // gi -> /dʒ/ (giant, magic)
        self.add_rule("gy", "dʒ", 65)?; // gy -> /dʒ/ (gym)
        self.add_rule("g", "ɡ", 60)?; // g -> /ɡ/ (go, big)
        self.add_rule("h", "h", 60)?;
        self.add_rule("j", "dʒ", 60)?;
        self.add_rule("k", "k", 60)?;
        self.add_rule("l", "l", 60)?;
        self.add_rule("m", "m", 60)?;
        self.add_rule("n", "n", 60)?;
        self.add_rule("p", "p", 60)?;
        self.add_rule("q", "k", 60)?; // q -> /k/
        self.add_rule("r", "r", 60)?;
        self.add_rule("s", "s", 60)?;
        self.add_rule("t", "t", 60)?;
        self.add_rule("v", "v", 60)?;
        self.add_rule("w", "w", 60)?;
        self.add_rule("x", "ks", 60)?; // x -> /ks/
        self.add_rule("z", "z", 60)?;

        // Sort rules by priority (highest first)
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));

        Ok(())
    }

    /// Add a rule to the rule set
    fn add_rule(&mut self, pattern: &str, replacement: &str, priority: i32) -> Result<()> {
        let rule = G2pRule::new(pattern, replacement, priority)?;
        self.rules.push(rule);
        Ok(())
    }

    /// Load common English words with known pronunciations
    fn load_common_words(&mut self) {
        // High-frequency irregular words with IPA symbols
        let words = [
            ("the", vec!["ðə".to_string()]), // voiced dental fricative + schwa
            ("to", vec!["tuː".to_string()]), // voiceless alveolar plosive + close back rounded
            ("of", vec!["ʌv".to_string()]), // open-mid back unrounded + voiced labiodental fricative
            ("and", vec!["ænd".to_string()]),
            ("a", vec!["eɪ".to_string(), "ə".to_string()]),
            ("in", vec!["ɪn".to_string()]),
            ("is", vec!["ɪz".to_string()]),
            ("it", vec!["ɪt".to_string()]),
            ("you", vec!["juː".to_string()]),
            ("that", vec!["ðæt".to_string()]),
            ("he", vec!["hiː".to_string()]),
            ("was", vec!["wʌz".to_string()]),
            ("for", vec!["fɔːr".to_string()]),
            ("on", vec!["ɒn".to_string()]),
            ("are", vec!["ɑːr".to_string()]),
            ("as", vec!["æz".to_string()]),
            ("with", vec!["wɪð".to_string()]),
            ("his", vec!["hɪz".to_string()]),
            ("they", vec!["ðeɪ".to_string()]),
            ("i", vec!["aɪ".to_string()]),
            ("at", vec!["æt".to_string()]),
            ("be", vec!["biː".to_string()]),
            ("this", vec!["ðɪs".to_string()]),
            ("have", vec!["hæv".to_string()]),
            ("from", vec!["frʌm".to_string()]),
            ("or", vec!["ɔːr".to_string()]),
            ("one", vec!["wʌn".to_string()]),
            ("had", vec!["hæd".to_string()]),
            ("by", vec!["baɪ".to_string()]),
            ("word", vec!["wɜːrd".to_string()]),
            ("but", vec!["bʌt".to_string()]),
            ("not", vec!["nɒt".to_string()]),
            ("what", vec!["wʌt".to_string()]),
            ("all", vec!["ɔːl".to_string()]),
            ("were", vec!["wɜːr".to_string()]),
            ("we", vec!["wiː".to_string()]),
            ("when", vec!["wɛn".to_string()]),
            ("your", vec!["jʊər".to_string()]),
            ("can", vec!["kæn".to_string()]),
            ("said", vec!["sɛd".to_string()]),
            ("there", vec!["ðɛər".to_string()]),
            ("each", vec!["iːtʃ".to_string()]),
            ("which", vec!["wɪtʃ".to_string()]),
            ("do", vec!["duː".to_string()]),
            ("how", vec!["haʊ".to_string()]),
            ("their", vec!["ðɛər".to_string()]),
            ("if", vec!["ɪf".to_string()]),
            ("will", vec!["wɪl".to_string()]),
            ("up", vec!["ʌp".to_string()]),
            ("other", vec!["ʌðər".to_string()]),
            ("about", vec!["əbaʊt".to_string()]),
            ("out", vec!["aʊt".to_string()]),
            ("many", vec!["mɛni".to_string()]),
            ("then", vec!["ðɛn".to_string()]),
            ("them", vec!["ðɛm".to_string()]),
            ("these", vec!["ðiːz".to_string()]),
            ("so", vec!["soʊ".to_string()]),
            ("some", vec!["sʌm".to_string()]),
            ("her", vec!["hɜːr".to_string()]),
            ("would", vec!["wʊd".to_string()]),
            ("make", vec!["meɪk".to_string()]),
            ("like", vec!["laɪk".to_string()]),
            ("into", vec!["ɪntuː".to_string()]),
            ("him", vec!["hɪm".to_string()]),
            ("time", vec!["taɪm".to_string()]),
            ("has", vec!["hæz".to_string()]),
            ("two", vec!["tuː".to_string()]),
            ("more", vec!["mɔːr".to_string()]),
            ("go", vec!["ɡoʊ".to_string()]),
            ("no", vec!["noʊ".to_string()]),
            ("way", vec!["weɪ".to_string()]),
            ("could", vec!["kʊd".to_string()]),
            ("my", vec!["maɪ".to_string()]),
            ("than", vec!["ðæn".to_string()]),
            ("first", vec!["fɜːrst".to_string()]),
            ("water", vec!["wɔːtər".to_string()]),
            ("been", vec!["bɪn".to_string()]),
            ("call", vec!["kɔːl".to_string()]),
            ("who", vec!["huː".to_string()]),
            ("its", vec!["ɪts".to_string()]),
            ("now", vec!["naʊ".to_string()]),
            ("find", vec!["faɪnd".to_string()]),
            ("long", vec!["lɔːŋ".to_string()]),
            ("down", vec!["daʊn".to_string()]),
            ("day", vec!["deɪ".to_string()]),
            ("did", vec!["dɪd".to_string()]),
            ("get", vec!["ɡɛt".to_string()]),
            ("come", vec!["kʌm".to_string()]),
            ("made", vec!["meɪd".to_string()]),
            ("may", vec!["meɪ".to_string()]),
            ("part", vec!["pɑːrt".to_string()]),
        ];

        for (word, phonemes) in words.iter() {
            self.word_dict.insert(word.to_string(), phonemes.clone());
        }
    }

    /// Convert text to phonemes using rules and dictionary
    fn text_to_phonemes(&self, text: &str) -> Vec<Phoneme> {
        let text = text.to_lowercase();
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut all_phonemes = Vec::new();

        for (i, word) in words.iter().enumerate() {
            // Clean word (remove punctuation)
            let clean_word: String = word.chars().filter(|c| c.is_alphabetic()).collect();

            if clean_word.is_empty() {
                continue;
            }

            // Check dictionary first
            if let Some(dict_phonemes) = self.word_dict.get(&clean_word) {
                for phoneme_str in dict_phonemes {
                    // Split concatenated IPA string into individual phonemes
                    let individual_phonemes = self.split_ipa_string(phoneme_str);
                    for ipa_symbol in individual_phonemes {
                        all_phonemes.push(Phoneme::with_ipa(&ipa_symbol, &ipa_symbol));
                    }
                }
            } else {
                // Apply rules
                let word_phonemes = self.apply_rules(&clean_word);
                all_phonemes.extend(word_phonemes);
            }

            // Add word boundary (except after last word)
            if i < words.len() - 1 {
                all_phonemes.push(Phoneme::word_boundary());
            }
        }

        all_phonemes
    }

    /// Split concatenated IPA string into individual phoneme symbols
    fn split_ipa_string(&self, ipa_string: &str) -> Vec<String> {
        // Common multi-character IPA symbols that should be kept together
        let multi_char_symbols = [
            "dʒ", "tʃ", "eɪ", "aɪ", "ɔɪ", "aʊ", "oʊ", "ɪə", "eə",
            "ʊə", // diphthongs and affricates
            "ɜː", "ɑː", "ɔː", "uː", "iː", "ɒː", // long vowels
            "ʌr", "ɜr", "ɑr", "ɔr", "ər", "ɪr", "ʊr", "eər", "ɑər", "ɔər", // r-colored vowels
            "θ", "ð", "ʃ", "ʒ", "ŋ", "ʔ", // single special symbols
        ];

        let mut result = Vec::new();
        let mut remaining = ipa_string;

        while !remaining.is_empty() {
            let mut matched = false;

            // Try to match multi-character symbols first (longest first)
            for &symbol in &multi_char_symbols {
                if remaining.starts_with(symbol) {
                    result.push(symbol.to_string());
                    remaining = &remaining[symbol.len()..];
                    matched = true;
                    break;
                }
            }

            // If no multi-character symbol matched, take single character
            if !matched {
                let mut chars = remaining.chars();
                if let Some(c) = chars.next() {
                    result.push(c.to_string());
                    remaining = chars.as_str();
                }
            }
        }

        result
    }

    /// Apply G2P rules to a word
    fn apply_rules(&self, word: &str) -> Vec<Phoneme> {
        // First, try to match the whole word against ending patterns (magic-e patterns)
        for rule in &self.rules {
            if rule.pattern.is_match(word) && rule.pattern.as_str().ends_with('$') {
                // This is a word-ending pattern that should consume the whole word
                return vec![Phoneme::with_ipa(&rule.replacement, &rule.replacement)];
            }
        }

        // Process character by character for other patterns
        let mut remaining = word.to_string();
        let mut phonemes = Vec::new();

        while !remaining.is_empty() {
            let mut matched = false;

            // Try each rule in priority order
            for rule in &self.rules {
                if let Some(mat) = rule.pattern.find(&remaining) {
                    if mat.start() == 0 {
                        // Rule matches at the beginning
                        phonemes.push(Phoneme::with_ipa(&rule.replacement, &rule.replacement));
                        remaining = remaining[mat.end()..].to_string();
                        matched = true;
                        break;
                    }
                }
            }

            // If no rule matched, take the first character as-is
            if !matched {
                if let Some(first_char) = remaining.chars().next() {
                    let char_str = first_char.to_string();
                    phonemes.push(Phoneme::with_ipa(&char_str, &char_str));
                    remaining = remaining[first_char.len_utf8()..].to_string();
                }
            }
        }

        phonemes
    }
}

impl Default for EnglishRuleG2p {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            rules: Vec::new(),
            word_dict: HashMap::new(),
            preprocessor: TextPreprocessor::new(LanguageCode::EnUs),
        })
    }
}

#[async_trait]
impl G2p for EnglishRuleG2p {
    async fn to_phonemes(&self, text: &str, _lang: Option<LanguageCode>) -> Result<Vec<Phoneme>> {
        tracing::debug!("EnglishRuleG2p: Converting '{text}' to phonemes");

        // Preprocess the text
        let preprocessed_text = self.preprocessor.preprocess(text)?;
        tracing::debug!("EnglishRuleG2p: Preprocessed text: '{preprocessed_text}'");

        let phonemes = self.text_to_phonemes(&preprocessed_text);

        tracing::debug!(
            "EnglishRuleG2p: Generated {len} phonemes",
            len = phonemes.len()
        );
        Ok(phonemes)
    }

    fn supported_languages(&self) -> Vec<LanguageCode> {
        vec![LanguageCode::EnUs, LanguageCode::EnGb]
    }

    fn metadata(&self) -> G2pMetadata {
        let mut accuracy_scores = HashMap::new();
        accuracy_scores.insert(LanguageCode::EnUs, 0.75); // Estimated accuracy
        accuracy_scores.insert(LanguageCode::EnGb, 0.70);

        G2pMetadata {
            name: "English Rule-based G2P".to_string(),
            version: "0.1.0".to_string(),
            description: "Rule-based grapheme-to-phoneme converter for English".to_string(),
            supported_languages: self.supported_languages(),
            accuracy_scores,
        }
    }
}
