//! Unknown word handling strategies for TTS pipeline
//!
//! This module contains methods for handling unknown words including:
//! - Fallback rules application
//! - Letter-by-letter phoneme generation
//! - Similar word pronunciation finding
//! - Edit distance and phonetic similarity calculation
//! - Phonetic pattern matching
//! - Pronunciation transformation

use super::*;
use crate::{AcousticError, LanguageCode, Phoneme, Result};
use log::debug;
use std::collections::HashMap;

impl TtsPipeline {
    pub(crate) fn apply_unknown_word_strategy(
        &self,
        word: &str,
        g2p_config: &G2pConfig,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        match g2p_config.unknown_word_strategy {
            UnknownWordStrategy::FallbackRules => self.apply_fallback_rules(word, phoneme_set),
            UnknownWordStrategy::LetterByLetter => {
                self.letter_by_letter_phonemes(word, phoneme_set)
            }
            UnknownWordStrategy::Skip => Ok(vec![]),
            UnknownWordStrategy::SimilarWord => {
                self.find_similar_word_pronunciation(word, phoneme_set)
            }
            UnknownWordStrategy::Error => Err(AcousticError::InputError {
                message: format!("Unknown word: '{word}'. No pronunciation available."),
            }),
        }
    }
    /// Enhanced G2P rules that mimic neural network behavior
    #[allow(dead_code)]
    fn enhanced_g2p_rules(
        &self,
        word: &str,
        language: LanguageCode,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        match language {
            LanguageCode::EnUs | LanguageCode::EnGb => {
                self.enhanced_english_rules(word, phoneme_set)
            }
            _ => self.apply_fallback_rules(word, phoneme_set),
        }
    }
    /// Enhanced English G2P rules
    #[allow(dead_code)]
    fn enhanced_english_rules(
        &self,
        word: &str,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        let word_lower = word.to_lowercase();
        let mut phonemes = Vec::new();
        if word_lower.ends_with("ing") {
            let base = &word_lower[..word_lower.len() - 3];
            phonemes.extend(self.process_english_base(base, phoneme_set)?);
            phonemes.extend(self.create_phonemes(&["IH", "NG"], phoneme_set)?);
        } else if word_lower.ends_with("ed") {
            let base = &word_lower[..word_lower.len() - 2];
            phonemes.extend(self.process_english_base(base, phoneme_set)?);
            phonemes.extend(self.create_phonemes(&["D"], phoneme_set)?);
        } else if word_lower.ends_with("s") && word_lower.len() > 1 {
            let base = &word_lower[..word_lower.len() - 1];
            phonemes.extend(self.process_english_base(base, phoneme_set)?);
            phonemes.extend(self.create_phonemes(&["S"], phoneme_set)?);
        } else {
            phonemes.extend(self.process_english_base(&word_lower, phoneme_set)?);
        }
        Ok(phonemes)
    }
    /// Process English base word
    #[allow(dead_code)]
    fn process_english_base(
        &self,
        word: &str,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        let mut result = Vec::new();
        let chars: Vec<char> = word.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let phoneme_symbols = match chars.get(i..i + 2) {
                Some(['c', 'h']) => {
                    i += 2;
                    vec!["CH"]
                }
                Some(['s', 'h']) => {
                    i += 2;
                    vec!["SH"]
                }
                Some(['t', 'h']) => {
                    i += 2;
                    vec!["TH"]
                }
                Some(['p', 'h']) => {
                    i += 2;
                    vec!["F"]
                }
                Some(['g', 'h']) => {
                    i += 2;
                    vec!["G"]
                }
                _ => {
                    let phoneme = self.english_char_to_phoneme(chars[i]);
                    i += 1;
                    vec![phoneme]
                }
            };
            for symbol in phoneme_symbols {
                if phoneme_set.symbols.contains(&symbol.to_string()) {
                    result.push(crate::Phoneme::new(symbol));
                }
            }
        }
        Ok(result)
    }
    /// Convert English character to phoneme
    #[allow(dead_code)]
    fn english_char_to_phoneme(&self, ch: char) -> &'static str {
        match ch {
            'a' => "AE",
            'e' => "EH",
            'i' => "IH",
            'o' => "AO",
            'u' => "UH",
            'b' => "B",
            'c' => "K",
            'd' => "D",
            'f' => "F",
            'g' => "G",
            'h' => "HH",
            'j' => "JH",
            'k' => "K",
            'l' => "L",
            'm' => "M",
            'n' => "N",
            'p' => "P",
            'q' => "K",
            'r' => "R",
            's' => "S",
            't' => "T",
            'v' => "V",
            'w' => "W",
            'x' => "K",
            'y' => "Y",
            'z' => "Z",
            _ => "AH",
        }
    }
    /// Apply English G2P rules
    #[allow(dead_code)]
    pub(crate) fn apply_english_g2p_rules(
        &self,
        word: &str,
        _g2p_config: &G2pConfig,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        self.enhanced_english_rules(word, phoneme_set)
    }
    /// Apply German G2P rules (simplified)
    #[allow(dead_code)]
    pub(crate) fn apply_german_g2p_rules(
        &self,
        word: &str,
        _g2p_config: &G2pConfig,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        self.apply_fallback_rules(word, phoneme_set)
    }
    /// Apply French G2P rules (simplified)
    #[allow(dead_code)]
    pub(crate) fn apply_french_g2p_rules(
        &self,
        word: &str,
        _g2p_config: &G2pConfig,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        self.apply_fallback_rules(word, phoneme_set)
    }
    /// Apply Spanish G2P rules (simplified)
    #[allow(dead_code)]
    pub(crate) fn apply_spanish_g2p_rules(
        &self,
        word: &str,
        _g2p_config: &G2pConfig,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        self.apply_fallback_rules(word, phoneme_set)
    }
    /// Apply Japanese G2P rules with hiragana, katakana, and basic kanji support
    #[allow(dead_code)]
    pub(crate) fn apply_japanese_g2p_rules(
        &self,
        word: &str,
        _g2p_config: &G2pConfig,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        debug!("Applying Japanese G2P rules for: '{}'", word);
        let mut phonemes = Vec::new();
        for ch in word.chars() {
            let (phoneme_symbols, duration) = match ch {
                'あ' => (vec!["a"], 0.12),
                'い' => (vec!["i"], 0.10),
                'う' => (vec!["u"], 0.11),
                'え' => (vec!["e"], 0.11),
                'お' => (vec!["o"], 0.12),
                'か' => (vec!["k", "a"], 0.14),
                'き' => (vec!["k", "i"], 0.13),
                'く' => (vec!["k", "u"], 0.13),
                'け' => (vec!["k", "e"], 0.13),
                'こ' => (vec!["k", "o"], 0.14),
                'が' => (vec!["g", "a"], 0.14),
                'ぎ' => (vec!["g", "i"], 0.13),
                'ぐ' => (vec!["g", "u"], 0.13),
                'げ' => (vec!["g", "e"], 0.13),
                'ご' => (vec!["g", "o"], 0.14),
                'さ' => (vec!["s", "a"], 0.14),
                'し' => (vec!["sh", "i"], 0.13),
                'す' => (vec!["s", "u"], 0.13),
                'せ' => (vec!["s", "e"], 0.13),
                'そ' => (vec!["s", "o"], 0.14),
                'ざ' => (vec!["z", "a"], 0.14),
                'じ' => (vec!["zh", "i"], 0.13),
                'ず' => (vec!["z", "u"], 0.13),
                'ぜ' => (vec!["z", "e"], 0.13),
                'ぞ' => (vec!["z", "o"], 0.14),
                'た' => (vec!["t", "a"], 0.14),
                'ち' => (vec!["ch", "i"], 0.13),
                'つ' => (vec!["ts", "u"], 0.13),
                'て' => (vec!["t", "e"], 0.13),
                'と' => (vec!["t", "o"], 0.14),
                'だ' => (vec!["d", "a"], 0.14),
                'ぢ' => (vec!["d", "i"], 0.13),
                'づ' => (vec!["d", "u"], 0.13),
                'で' => (vec!["d", "e"], 0.13),
                'ど' => (vec!["d", "o"], 0.14),
                'な' => (vec!["n", "a"], 0.14),
                'に' => (vec!["n", "i"], 0.13),
                'ぬ' => (vec!["n", "u"], 0.13),
                'ね' => (vec!["n", "e"], 0.13),
                'の' => (vec!["n", "o"], 0.14),
                'は' => (vec!["h", "a"], 0.14),
                'ひ' => (vec!["h", "i"], 0.13),
                'ふ' => (vec!["f", "u"], 0.13),
                'へ' => (vec!["h", "e"], 0.13),
                'ほ' => (vec!["h", "o"], 0.14),
                'ば' => (vec!["b", "a"], 0.14),
                'び' => (vec!["b", "i"], 0.13),
                'ぶ' => (vec!["b", "u"], 0.13),
                'べ' => (vec!["b", "e"], 0.13),
                'ぼ' => (vec!["b", "o"], 0.14),
                'ぱ' => (vec!["p", "a"], 0.14),
                'ぴ' => (vec!["p", "i"], 0.13),
                'ぷ' => (vec!["p", "u"], 0.13),
                'ぺ' => (vec!["p", "e"], 0.13),
                'ぽ' => (vec!["p", "o"], 0.14),
                'ま' => (vec!["m", "a"], 0.14),
                'み' => (vec!["m", "i"], 0.13),
                'む' => (vec!["m", "u"], 0.13),
                'め' => (vec!["m", "e"], 0.13),
                'も' => (vec!["m", "o"], 0.14),
                'や' => (vec!["y", "a"], 0.13),
                'ゆ' => (vec!["y", "u"], 0.13),
                'よ' => (vec!["y", "o"], 0.13),
                'ら' => (vec!["r", "a"], 0.14),
                'り' => (vec!["r", "i"], 0.13),
                'る' => (vec!["r", "u"], 0.13),
                'れ' => (vec!["r", "e"], 0.13),
                'ろ' => (vec!["r", "o"], 0.14),
                'わ' => (vec!["w", "a"], 0.13),
                'ゐ' => (vec!["w", "i"], 0.13),
                'ゑ' => (vec!["w", "e"], 0.13),
                'を' => (vec!["w", "o"], 0.13),
                'ん' => (vec!["N"], 0.10),
                'ア' => (vec!["a"], 0.12),
                'イ' => (vec!["i"], 0.10),
                'ウ' => (vec!["u"], 0.11),
                'エ' => (vec!["e"], 0.11),
                'オ' => (vec!["o"], 0.12),
                'カ' => (vec!["k", "a"], 0.14),
                'キ' => (vec!["k", "i"], 0.13),
                'ク' => (vec!["k", "u"], 0.13),
                'ケ' => (vec!["k", "e"], 0.13),
                'コ' => (vec!["k", "o"], 0.14),
                'サ' => (vec!["s", "a"], 0.14),
                'シ' => (vec!["sh", "i"], 0.13),
                'ス' => (vec!["s", "u"], 0.13),
                'セ' => (vec!["s", "e"], 0.13),
                'ソ' => (vec!["s", "o"], 0.14),
                'タ' => (vec!["t", "a"], 0.14),
                'チ' => (vec!["ch", "i"], 0.13),
                'ツ' => (vec!["ts", "u"], 0.13),
                'テ' => (vec!["t", "e"], 0.13),
                'ト' => (vec!["t", "o"], 0.14),
                'ナ' => (vec!["n", "a"], 0.14),
                'ニ' => (vec!["n", "i"], 0.13),
                'ヌ' => (vec!["n", "u"], 0.13),
                'ネ' => (vec!["n", "e"], 0.13),
                'ノ' => (vec!["n", "o"], 0.14),
                'ハ' => (vec!["h", "a"], 0.14),
                'ヒ' => (vec!["h", "i"], 0.13),
                'フ' => (vec!["f", "u"], 0.13),
                'ヘ' => (vec!["h", "e"], 0.13),
                'ホ' => (vec!["h", "o"], 0.14),
                'マ' => (vec!["m", "a"], 0.14),
                'ミ' => (vec!["m", "i"], 0.13),
                'ム' => (vec!["m", "u"], 0.13),
                'メ' => (vec!["m", "e"], 0.13),
                'モ' => (vec!["m", "o"], 0.14),
                'ヤ' => (vec!["y", "a"], 0.13),
                'ユ' => (vec!["y", "u"], 0.13),
                'ヨ' => (vec!["y", "o"], 0.13),
                'ラ' => (vec!["r", "a"], 0.14),
                'リ' => (vec!["r", "i"], 0.13),
                'ル' => (vec!["r", "u"], 0.13),
                'レ' => (vec!["r", "e"], 0.13),
                'ロ' => (vec!["r", "o"], 0.14),
                'ワ' => (vec!["w", "a"], 0.13),
                'ヲ' => (vec!["w", "o"], 0.13),
                'ン' => (vec!["N"], 0.10),
                'っ' | 'ッ' => (vec!["Q"], 0.08),
                'ー' => (vec!["LONG"], 0.06),
                '。' | '.' => (vec!["sil"], 0.30),
                '、' | ',' => (vec!["sp"], 0.15),
                ' ' => (vec!["sp"], 0.10),
                c if c.is_ascii_alphabetic() => {
                    let symbol = match c.to_ascii_lowercase() {
                        'a' => "a",
                        'e' => "e",
                        'i' => "i",
                        'o' => "o",
                        'u' => "u",
                        'k' => "k",
                        'g' => "g",
                        's' => "s",
                        'z' => "z",
                        't' => "t",
                        'd' => "d",
                        'n' => "n",
                        'h' => "h",
                        'b' => "b",
                        'p' => "p",
                        'm' => "m",
                        'y' => "y",
                        'r' => "r",
                        'w' => "w",
                        'f' => "f",
                        'j' => "j",
                        'l' => "r",
                        'v' => "b",
                        _ => "a",
                    };
                    (vec![symbol], 0.10)
                }
                _ => (vec!["sp"], 0.05),
            };
            for symbol in phoneme_symbols {
                if phoneme_set.symbols.contains(&symbol.to_string()) {
                    let mut phoneme = crate::Phoneme::new(symbol);
                    phoneme.duration = Some(duration);
                    phonemes.push(phoneme);
                }
            }
        }
        if !phonemes.is_empty() {
            let mut final_pause = crate::Phoneme::new("sil");
            final_pause.duration = Some(0.20);
            phonemes.push(final_pause);
        }
        debug!("Generated {} phonemes for Japanese text", phonemes.len());
        Ok(phonemes)
    }
    /// Apply generic G2P rules for unsupported languages
    #[allow(dead_code)]
    pub(crate) fn apply_generic_g2p_rules(
        &self,
        word: &str,
        _g2p_config: &G2pConfig,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        self.apply_fallback_rules(word, phoneme_set)
    }
    /// Helper to create phonemes from symbol array
    #[allow(dead_code)]
    fn create_phonemes(
        &self,
        symbols: &[&str],
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        let mut phonemes = Vec::new();
        for &symbol in symbols {
            if phoneme_set.symbols.contains(&symbol.to_string()) {
                phonemes.push(crate::Phoneme::new(symbol));
            }
        }
        Ok(phonemes)
    }
    /// Apply advanced language-specific pronunciation rules
    #[allow(dead_code)]
    fn apply_fallback_rules(
        &self,
        word: &str,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        debug!("Applying enhanced fallback rules for: '{}'", word);
        let normalized_word = word.to_lowercase();
        let chars: Vec<char> = normalized_word.chars().collect();
        let mut phonemes = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let current_char = chars[i];
            let next_char = chars.get(i + 1).copied();
            let prev_char = if i > 0 {
                chars.get(i - 1).copied()
            } else {
                None
            };
            let next2_char = chars.get(i + 2).copied();
            let phoneme_symbols = match current_char {
                'a' => match next_char {
                    Some('r') => vec!["AA", "R"],
                    Some('i') | Some('y') => vec!["EY"],
                    Some('u') => vec!["AO"],
                    Some('w') => vec!["AO"],
                    Some('l') if next2_char.is_none_or(|c| !c.is_alphabetic()) => {
                        vec!["AO", "L"]
                    }
                    _ => {
                        if chars.len() > i + 2
                            && chars[chars.len() - 1] == 'e'
                            && chars[chars.len() - 2] != 'e'
                        {
                            vec!["EY"]
                        } else {
                            vec!["AE"]
                        }
                    }
                },
                'e' => match next_char {
                    Some('a') => vec!["IY"],
                    Some('e') => vec!["IY"],
                    Some('i') => vec!["EY"],
                    Some('r') => vec!["ER"],
                    Some('w') => vec!["UW"],
                    _ => {
                        if i == chars.len() - 1 {
                            vec![]
                        } else {
                            vec!["EH"]
                        }
                    }
                },
                'i' => match next_char {
                    Some('r') => vec!["ER"],
                    Some('e') => vec!["AY"],
                    Some('g') if next2_char == Some('h') => vec!["AY"],
                    _ => {
                        if chars.len() > i + 2 && chars[chars.len() - 1] == 'e' {
                            vec!["AY"]
                        } else {
                            vec!["IH"]
                        }
                    }
                },
                'o' => match next_char {
                    Some('a') => vec!["OW"],
                    Some('o') => vec!["UW"],
                    Some('u') => vec!["AW"],
                    Some('w') => vec!["OW"],
                    Some('r') => vec!["AO", "R"],
                    Some('y') => vec!["OY"],
                    _ => {
                        if chars.len() > i + 2 && chars[chars.len() - 1] == 'e' {
                            vec!["OW"]
                        } else {
                            vec!["AA"]
                        }
                    }
                },
                'u' => match next_char {
                    Some('r') => vec!["ER"],
                    Some('e') => vec!["UW"],
                    Some('i') => vec!["UW"],
                    _ => {
                        if chars.len() > i + 2 && chars[chars.len() - 1] == 'e' {
                            vec!["UW"]
                        } else {
                            vec!["UH"]
                        }
                    }
                },
                'y' => {
                    if i == 0 {
                        vec!["Y"]
                    } else if i == chars.len() - 1 {
                        vec!["IY"]
                    } else {
                        vec!["IH"]
                    }
                }
                'c' => match next_char {
                    Some('h') => {
                        i += 1;
                        vec!["CH"]
                    }
                    Some('k') => {
                        i += 1;
                        vec!["K"]
                    }
                    Some('e') | Some('i') | Some('y') => vec!["S"],
                    _ => vec!["K"],
                },
                'g' => match next_char {
                    Some('h') => {
                        i += 1;
                        if prev_char == Some('u') {
                            vec![]
                        } else {
                            vec!["F"]
                        }
                    }
                    Some('n') => {
                        if i == chars.len() - 2 {
                            vec!["NG"]
                        } else {
                            vec!["G", "N"]
                        }
                    }
                    Some('e') | Some('i') | Some('y') => vec!["JH"],
                    _ => vec!["G"],
                },
                'p' => match next_char {
                    Some('h') => {
                        i += 1;
                        vec!["F"]
                    }
                    _ => vec!["P"],
                },
                's' => match next_char {
                    Some('h') => {
                        i += 1;
                        vec!["SH"]
                    }
                    Some('s') => vec!["S"],
                    _ => {
                        if (ModelManager::is_vowel(prev_char) && ModelManager::is_vowel(next_char))
                            || (i == chars.len() - 1 && ModelManager::is_vowel(prev_char))
                        {
                            vec!["Z"]
                        } else {
                            vec!["S"]
                        }
                    }
                },
                't' => match next_char {
                    Some('h') => {
                        i += 1;
                        vec!["TH"]
                    }
                    Some('i') if next2_char == Some('o') => vec!["SH"],
                    _ => vec!["T"],
                },
                'w' => match next_char {
                    Some('h') => {
                        i += 1;
                        vec!["W"]
                    }
                    Some('r') => {
                        i += 1;
                        vec!["R"]
                    }
                    _ => vec!["W"],
                },
                'x' => vec!["K", "S"],
                'q' => {
                    if next_char == Some('u') {
                        i += 1;
                        vec!["K", "W"]
                    } else {
                        vec!["K"]
                    }
                }
                'b' => vec!["B"],
                'd' => vec!["D"],
                'f' => vec!["F"],
                'h' => {
                    if i == 0 || !ModelManager::is_vowel(prev_char) {
                        vec!["HH"]
                    } else {
                        vec![]
                    }
                }
                'j' => vec!["JH"],
                'k' => vec!["K"],
                'l' => vec!["L"],
                'm' => vec!["M"],
                'n' => vec!["N"],
                'r' => vec!["R"],
                'v' => vec!["V"],
                'z' => vec!["Z"],
                _ => {
                    if current_char.is_alphabetic() {
                        vec!["AH"]
                    } else {
                        vec![]
                    }
                }
            };
            for symbol in phoneme_symbols {
                if phoneme_set.symbols.contains(&symbol.to_string()) {
                    phonemes.push(crate::Phoneme::new(symbol));
                } else {
                    phonemes.push(crate::Phoneme::new("AH"));
                }
            }
            i += 1;
        }
        ModelManager::apply_stress_rules_static(&mut phonemes);
        debug!(
            "Enhanced G2P produced {} phonemes for '{}'",
            phonemes.len(),
            word
        );
        Ok(phonemes)
    }
    /// Generate enhanced letter-by-letter phonemes with better mapping
    #[allow(dead_code)]
    fn letter_by_letter_phonemes(
        &self,
        word: &str,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        debug!(
            "Generating enhanced letter-by-letter phonemes for: '{}'",
            word
        );
        let mut phonemes = Vec::new();
        for (i, ch) in word.chars().enumerate() {
            if ch.is_alphabetic() {
                let lowercase_ch = ch
                    .to_lowercase()
                    .next()
                    .expect("to_lowercase() should always produce at least one character");
                let phoneme_symbol = match lowercase_ch {
                    'a' => "EY",
                    'b' => "B",
                    'c' => "S",
                    'd' => "D",
                    'e' => "IY",
                    'f' => "F",
                    'g' => "JH",
                    'h' => "EY",
                    'i' => "AY",
                    'j' => "JH",
                    'k' => "K",
                    'l' => "L",
                    'm' => "M",
                    'n' => "N",
                    'o' => "OW",
                    'p' => "P",
                    'q' => "K",
                    'r' => "R",
                    's' => "S",
                    't' => "T",
                    'u' => "UW",
                    'v' => "V",
                    'w' => "W",
                    'x' => "K",
                    'y' => "W",
                    'z' => "Z",
                    _ => "AH",
                };
                let symbol = if phoneme_set.symbols.contains(&phoneme_symbol.to_string()) {
                    phoneme_symbol
                } else {
                    "AH"
                };
                let mut phoneme = crate::Phoneme::new(symbol);
                let base_duration = 0.12;
                let variation = (i as f32 * 0.005) % 0.02;
                phoneme.duration = Some(base_duration + variation);
                phonemes.push(phoneme);
                if i < word.len() - 1 {
                    let mut pause = crate::Phoneme::new("SIL");
                    pause.duration = Some(0.05);
                    phonemes.push(pause);
                }
            }
        }
        debug!(
            "Letter-by-letter G2P produced {} phonemes for '{}'",
            phonemes.len(),
            word
        );
        Ok(phonemes)
    }
    /// Find similar word pronunciation
    #[allow(dead_code)]
    fn find_similar_word_pronunciation(
        &self,
        word: &str,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        debug!("Finding similar word pronunciation for: '{}'", word);
        let similar_word = self.find_most_similar_word(word);
        if let Some((similar, similarity_score)) = similar_word {
            if similarity_score > 0.7 {
                debug!(
                    "Found similar word '{}' with similarity {:.2} for '{}'",
                    similar, similarity_score, word
                );
                return self.apply_phonetic_transformation(word, &similar, phoneme_set);
            }
        }
        self.apply_pattern_based_pronunciation(word, phoneme_set)
    }
    /// Find the most similar known word using multiple similarity metrics
    #[allow(dead_code)]
    fn find_most_similar_word(&self, target: &str) -> Option<(String, f32)> {
        let known_words = self.get_known_words();
        let mut best_match: Option<(String, f32)> = None;
        for word in known_words {
            let edit_distance_sim = self.calculate_edit_distance_similarity(target, &word);
            let phonetic_sim = self.calculate_phonetic_similarity(target, &word);
            let pattern_sim = self.calculate_pattern_similarity(target, &word);
            let combined_similarity =
                edit_distance_sim * 0.4 + phonetic_sim * 0.4 + pattern_sim * 0.2;
            if let Some((_, current_best)) = &best_match {
                if combined_similarity > *current_best {
                    best_match = Some((word, combined_similarity));
                }
            } else {
                best_match = Some((word, combined_similarity));
            }
        }
        best_match
    }
    /// Calculate edit distance similarity (normalized Levenshtein distance)
    #[allow(dead_code)]
    fn calculate_edit_distance_similarity(&self, s1: &str, s2: &str) -> f32 {
        let distance = self.levenshtein_distance(s1, s2);
        let max_len = s1.len().max(s2.len()) as f32;
        if max_len == 0.0 {
            return 1.0;
        }
        1.0 - (distance as f32 / max_len)
    }
    /// Calculate Levenshtein distance between two strings
    #[allow(dead_code)]
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        let len1 = chars1.len();
        let len2 = chars2.len();
        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];
        for (i, row) in matrix.iter_mut().enumerate().take(len1 + 1) {
            row[0] = i;
        }
        for (j, cell) in matrix[0].iter_mut().enumerate().take(len2 + 1) {
            *cell = j;
        }
        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }
        matrix[len1][len2]
    }
    /// Calculate phonetic similarity using sound patterns
    #[allow(dead_code)]
    fn calculate_phonetic_similarity(&self, s1: &str, s2: &str) -> f32 {
        let phonetic1 = self.word_to_phonetic_pattern(s1);
        let phonetic2 = self.word_to_phonetic_pattern(s2);
        self.pattern_similarity(&phonetic1, &phonetic2)
    }
    /// Convert word to phonetic pattern representation
    #[allow(dead_code)]
    fn word_to_phonetic_pattern(&self, word: &str) -> Vec<char> {
        word.chars()
            .map(|c| self.char_to_phonetic_class(c))
            .collect()
    }
    /// Map character to phonetic class
    #[allow(dead_code)]
    fn char_to_phonetic_class(&self, c: char) -> char {
        match c.to_ascii_lowercase() {
            'a' | 'e' | 'i' | 'o' | 'u' => 'V',
            'p' | 'b' | 't' | 'd' | 'k' | 'g' => 'P',
            'f' | 'v' | 's' | 'z' | 'h' => 'F',
            'm' | 'n' => 'N',
            'l' | 'r' => 'L',
            'w' | 'y' => 'G',
            _ => 'C',
        }
    }
    /// Calculate pattern similarity between two phonetic patterns
    #[allow(dead_code)]
    fn pattern_similarity(&self, pattern1: &[char], pattern2: &[char]) -> f32 {
        if pattern1.is_empty() && pattern2.is_empty() {
            return 1.0;
        }
        let matching_positions = pattern1
            .iter()
            .zip(pattern2.iter())
            .filter(|(a, b)| a == b)
            .count();
        let max_len = pattern1.len().max(pattern2.len());
        if max_len == 0 {
            return 1.0;
        }
        matching_positions as f32 / max_len as f32
    }
    /// Calculate structural pattern similarity (prefixes, suffixes, etc.)
    #[allow(dead_code)]
    fn calculate_pattern_similarity(&self, s1: &str, s2: &str) -> f32 {
        let mut similarity = 0.0;
        let prefix_len = self.common_prefix_length(s1, s2);
        similarity += (prefix_len as f32 / s1.len().max(s2.len()) as f32) * 0.5;
        let suffix_len = self.common_suffix_length(s1, s2);
        similarity += (suffix_len as f32 / s1.len().max(s2.len()) as f32) * 0.5;
        similarity.min(1.0)
    }
    /// Calculate common prefix length
    #[allow(dead_code)]
    fn common_prefix_length(&self, s1: &str, s2: &str) -> usize {
        s1.chars()
            .zip(s2.chars())
            .take_while(|(a, b)| a == b)
            .count()
    }
    /// Calculate common suffix length
    #[allow(dead_code)]
    fn common_suffix_length(&self, s1: &str, s2: &str) -> usize {
        s1.chars()
            .rev()
            .zip(s2.chars().rev())
            .take_while(|(a, b)| a == b)
            .count()
    }
    /// Get list of known words for similarity matching
    #[allow(dead_code)]
    fn get_known_words(&self) -> Vec<String> {
        vec![
            "hello".to_string(),
            "world".to_string(),
            "voice".to_string(),
            "speech".to_string(),
            "text".to_string(),
            "synthesis".to_string(),
            "example".to_string(),
            "test".to_string(),
            "word".to_string(),
            "sound".to_string(),
            "language".to_string(),
            "pronunciation".to_string(),
            "phoneme".to_string(),
            "acoustic".to_string(),
            "model".to_string(),
        ]
    }
    /// Apply phonetic transformation based on similar word
    #[allow(dead_code)]
    fn apply_phonetic_transformation(
        &self,
        target: &str,
        similar: &str,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        debug!(
            "Applying phonetic transformation from '{}' to '{}'",
            similar, target
        );
        let similar_phonemes = self.enhanced_english_rules(similar, phoneme_set)?;
        let transformed = self.transform_pronunciation(target, similar, similar_phonemes)?;
        Ok(transformed)
    }
    /// Transform pronunciation based on character mapping
    #[allow(dead_code)]
    fn transform_pronunciation(
        &self,
        target: &str,
        source: &str,
        source_phonemes: Vec<crate::Phoneme>,
    ) -> Result<Vec<crate::Phoneme>> {
        let target_len = target.len();
        let _source_len = source.len();
        if target_len == 0 {
            return Ok(vec![]);
        }
        let mut result = Vec::new();
        if source_phonemes.is_empty() {
            return Ok(result);
        }
        for (i, _) in target.chars().enumerate() {
            let source_idx = (i * source_phonemes.len()) / target_len;
            if source_idx < source_phonemes.len() {
                result.push(source_phonemes[source_idx].clone());
            }
        }
        Ok(result)
    }
    /// Apply pattern-based pronunciation when no similar word is found
    #[allow(dead_code)]
    fn apply_pattern_based_pronunciation(
        &self,
        word: &str,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        debug!("Applying pattern-based pronunciation for: '{}'", word);
        self.enhanced_english_rules(word, phoneme_set)
    }
}
