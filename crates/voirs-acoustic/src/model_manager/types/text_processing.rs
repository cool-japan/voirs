//! Text preprocessing and normalization methods for TTS pipeline
//!
//! This module contains all text processing functionality including:
//! - Unicode normalization
//! - Number and ordinal expansion
//! - Abbreviation expansion
//! - Special character handling
//! - Whitespace normalization
//! - Text tokenization
//! - Contractions handling
//! - Compound word handling

use super::*;
use crate::{LanguageCode, Result};
use log::{debug, info};

impl TtsPipeline {
    /// Preprocess text for G2P conversion
    #[allow(dead_code)]
    fn preprocess_text(&self, text: &str, language: LanguageCode) -> Result<String> {
        let mut normalized = text.to_string();
        normalized = self.normalize_unicode(&normalized)?;
        normalized = self.expand_numbers(&normalized, language)?;
        normalized = self.expand_abbreviations(&normalized, language)?;
        normalized = self.handle_special_characters(&normalized)?;
        normalized = self.normalize_whitespace(&normalized)?;
        info!("Preprocessed text: '{}' -> '{}'", text, normalized);
        Ok(normalized)
    }
    /// Normalize Unicode characters
    fn normalize_unicode(&self, text: &str) -> Result<String> {
        let normalized = text.to_lowercase();
        let normalized = normalized
            .replace(['\u{2018}', '\u{2019}'], "'")
            .replace(['"', '"'], "\"")
            .replace(['–', '—'], "-")
            .replace('…', "...");
        Ok(normalized)
    }
    /// Expand numbers to their spoken form
    fn expand_numbers(&self, text: &str, language: LanguageCode) -> Result<String> {
        use regex::Regex;
        let number_regex =
            Regex::new(r"\b\d+\b").expect("Hardcoded number regex pattern should be valid");
        let mut result = text.to_string();
        result = number_regex
            .replace_all(&result, |caps: &regex::Captures| {
                let number_str = &caps[0];
                if let Ok(num) = number_str.parse::<i32>() {
                    self.number_to_words(num, language)
                } else {
                    number_str.to_string()
                }
            })
            .to_string();
        let ordinal_regex = Regex::new(r"\b(\d+)(st|nd|rd|th)\b")
            .expect("Hardcoded ordinal regex pattern should be valid");
        result = ordinal_regex
            .replace_all(&result, |caps: &regex::Captures| {
                let number_str = &caps[1];
                if let Ok(num) = number_str.parse::<i32>() {
                    self.ordinal_to_words(num, language)
                } else {
                    caps[0].to_string()
                }
            })
            .to_string();
        Ok(result)
    }
    /// Convert number to words
    #[allow(clippy::only_used_in_recursion)]
    fn number_to_words(&self, num: i32, _language: LanguageCode) -> String {
        match num {
            0 => "zero".to_string(),
            1 => "one".to_string(),
            2 => "two".to_string(),
            3 => "three".to_string(),
            4 => "four".to_string(),
            5 => "five".to_string(),
            6 => "six".to_string(),
            7 => "seven".to_string(),
            8 => "eight".to_string(),
            9 => "nine".to_string(),
            10 => "ten".to_string(),
            11 => "eleven".to_string(),
            12 => "twelve".to_string(),
            13 => "thirteen".to_string(),
            14 => "fourteen".to_string(),
            15 => "fifteen".to_string(),
            16 => "sixteen".to_string(),
            17 => "seventeen".to_string(),
            18 => "eighteen".to_string(),
            19 => "nineteen".to_string(),
            20 => "twenty".to_string(),
            21..=99 => {
                let tens = num / 10;
                let ones = num % 10;
                let tens_word = match tens {
                    2 => "twenty",
                    3 => "thirty",
                    4 => "forty",
                    5 => "fifty",
                    6 => "sixty",
                    7 => "seventy",
                    8 => "eighty",
                    9 => "ninety",
                    _ => "",
                };
                if ones == 0 {
                    tens_word.to_string()
                } else {
                    format!("{} {}", tens_word, self.number_to_words(ones, _language))
                }
            }
            100..=999 => {
                let hundreds = num / 100;
                let remainder = num % 100;
                let hundreds_word =
                    format!("{} hundred", self.number_to_words(hundreds, _language));
                if remainder == 0 {
                    hundreds_word
                } else {
                    format!(
                        "{} {}",
                        hundreds_word,
                        self.number_to_words(remainder, _language)
                    )
                }
            }
            1000..=999999 => {
                let thousands = num / 1000;
                let remainder = num % 1000;
                let thousands_word =
                    format!("{} thousand", self.number_to_words(thousands, _language));
                if remainder == 0 {
                    thousands_word
                } else {
                    format!(
                        "{} {}",
                        thousands_word,
                        self.number_to_words(remainder, _language)
                    )
                }
            }
            _ => num.to_string(),
        }
    }
    /// Convert ordinal number to words
    fn ordinal_to_words(&self, num: i32, language: LanguageCode) -> String {
        match num {
            1 => "first".to_string(),
            2 => "second".to_string(),
            3 => "third".to_string(),
            4 => "fourth".to_string(),
            5 => "fifth".to_string(),
            8 => "eighth".to_string(),
            9 => "ninth".to_string(),
            12 => "twelfth".to_string(),
            _ => {
                let base = self.number_to_words(num, language);
                if base.ends_with("y") {
                    format!("{}ieth", &base[..base.len() - 1])
                } else {
                    format!("{base}th")
                }
            }
        }
    }
    /// Expand abbreviations to their full form
    fn expand_abbreviations(&self, text: &str, _language: LanguageCode) -> Result<String> {
        let mut result = text.to_string();
        let abbreviations = [
            ("dr.", "doctor"),
            ("mr.", "mister"),
            ("mrs.", "misses"),
            ("ms.", "miss"),
            ("prof.", "professor"),
            ("etc.", "et cetera"),
            ("vs.", "versus"),
            ("e.g.", "for example"),
            ("i.e.", "that is"),
            ("a.m.", "a m"),
            ("p.m.", "p m"),
            ("u.s.", "united states"),
            ("u.k.", "united kingdom"),
            ("st.", "saint"),
            ("ave.", "avenue"),
            ("blvd.", "boulevard"),
            ("rd.", "road"),
            ("jr.", "junior"),
            ("sr.", "senior"),
            ("inc.", "incorporated"),
            ("ltd.", "limited"),
            ("co.", "company"),
            ("corp.", "corporation"),
        ];
        for (abbr, expansion) in &abbreviations {
            result = result.replace(abbr, expansion);
        }
        Ok(result)
    }
    /// Handle special characters and punctuation
    fn handle_special_characters(&self, text: &str) -> Result<String> {
        let mut result = text.to_string();
        result = result.replace("&", " and ");
        result = result.replace("@", " at ");
        result = result.replace("#", " number ");
        result = result.replace("%", " percent ");
        result = result.replace("$", " dollar ");
        result = result.replace("£", " pound ");
        result = result.replace("€", " euro ");
        result = result.replace("¥", " yen ");
        result = result.replace("+", " plus ");
        result = result.replace("=", " equals ");
        result = result.replace("°", " degree ");
        result = result
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '\'' || *c == '-')
            .collect();
        Ok(result)
    }
    /// Normalize whitespace
    fn normalize_whitespace(&self, text: &str) -> Result<String> {
        let normalized = text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_string();
        Ok(normalized)
    }
    /// Tokenize text into words
    #[allow(dead_code)]
    fn tokenize_text(&self, text: &str, language: LanguageCode) -> Result<Vec<String>> {
        let mut words: Vec<String> = text
            .split_whitespace()
            .map(|w| w.to_string())
            .filter(|w| !w.is_empty())
            .collect();
        words = self.handle_contractions(words, language)?;
        words = self.handle_compound_words(words, language)?;
        words = self.handle_hyphenated_words(words, language)?;
        debug!("Tokenized into {} words: {:?}", words.len(), words);
        Ok(words)
    }
    /// Handle contractions based on language
    fn handle_contractions(
        &self,
        words: Vec<String>,
        language: LanguageCode,
    ) -> Result<Vec<String>> {
        let mut result = Vec::new();
        for word in words {
            match language {
                LanguageCode::EnUs | LanguageCode::EnGb => match word.as_str() {
                    "can't" => {
                        result.push("can".to_string());
                        result.push("not".to_string());
                    }
                    "won't" => {
                        result.push("will".to_string());
                        result.push("not".to_string());
                    }
                    "don't" => {
                        result.push("do".to_string());
                        result.push("not".to_string());
                    }
                    "doesn't" => {
                        result.push("does".to_string());
                        result.push("not".to_string());
                    }
                    "didn't" => {
                        result.push("did".to_string());
                        result.push("not".to_string());
                    }
                    "isn't" => {
                        result.push("is".to_string());
                        result.push("not".to_string());
                    }
                    "aren't" => {
                        result.push("are".to_string());
                        result.push("not".to_string());
                    }
                    "wasn't" => {
                        result.push("was".to_string());
                        result.push("not".to_string());
                    }
                    "weren't" => {
                        result.push("were".to_string());
                        result.push("not".to_string());
                    }
                    "haven't" => {
                        result.push("have".to_string());
                        result.push("not".to_string());
                    }
                    "hasn't" => {
                        result.push("has".to_string());
                        result.push("not".to_string());
                    }
                    "hadn't" => {
                        result.push("had".to_string());
                        result.push("not".to_string());
                    }
                    "shouldn't" => {
                        result.push("should".to_string());
                        result.push("not".to_string());
                    }
                    "wouldn't" => {
                        result.push("would".to_string());
                        result.push("not".to_string());
                    }
                    "couldn't" => {
                        result.push("could".to_string());
                        result.push("not".to_string());
                    }
                    "mustn't" => {
                        result.push("must".to_string());
                        result.push("not".to_string());
                    }
                    "i'm" => {
                        result.push("i".to_string());
                        result.push("am".to_string());
                    }
                    "you're" => {
                        result.push("you".to_string());
                        result.push("are".to_string());
                    }
                    "he's" => {
                        result.push("he".to_string());
                        result.push("is".to_string());
                    }
                    "she's" => {
                        result.push("she".to_string());
                        result.push("is".to_string());
                    }
                    "it's" => {
                        result.push("it".to_string());
                        result.push("is".to_string());
                    }
                    "we're" => {
                        result.push("we".to_string());
                        result.push("are".to_string());
                    }
                    "they're" => {
                        result.push("they".to_string());
                        result.push("are".to_string());
                    }
                    "i've" => {
                        result.push("i".to_string());
                        result.push("have".to_string());
                    }
                    "you've" => {
                        result.push("you".to_string());
                        result.push("have".to_string());
                    }
                    "we've" => {
                        result.push("we".to_string());
                        result.push("have".to_string());
                    }
                    "they've" => {
                        result.push("they".to_string());
                        result.push("have".to_string());
                    }
                    "i'll" => {
                        result.push("i".to_string());
                        result.push("will".to_string());
                    }
                    "you'll" => {
                        result.push("you".to_string());
                        result.push("will".to_string());
                    }
                    "he'll" => {
                        result.push("he".to_string());
                        result.push("will".to_string());
                    }
                    "she'll" => {
                        result.push("she".to_string());
                        result.push("will".to_string());
                    }
                    "it'll" => {
                        result.push("it".to_string());
                        result.push("will".to_string());
                    }
                    "we'll" => {
                        result.push("we".to_string());
                        result.push("will".to_string());
                    }
                    "they'll" => {
                        result.push("they".to_string());
                        result.push("will".to_string());
                    }
                    "i'd" => {
                        result.push("i".to_string());
                        result.push("would".to_string());
                    }
                    "you'd" => {
                        result.push("you".to_string());
                        result.push("would".to_string());
                    }
                    "he'd" => {
                        result.push("he".to_string());
                        result.push("would".to_string());
                    }
                    "she'd" => {
                        result.push("she".to_string());
                        result.push("would".to_string());
                    }
                    "it'd" => {
                        result.push("it".to_string());
                        result.push("would".to_string());
                    }
                    "we'd" => {
                        result.push("we".to_string());
                        result.push("would".to_string());
                    }
                    "they'd" => {
                        result.push("they".to_string());
                        result.push("would".to_string());
                    }
                    _ => {
                        if word.ends_with("'s") && word.len() > 2 {
                            let base = &word[..word.len() - 2];
                            result.push(base.to_string());
                            result.push("is".to_string());
                        } else {
                            result.push(word);
                        }
                    }
                },
                LanguageCode::FrFr => match word.as_str() {
                    "c'est" => {
                        result.push("ce".to_string());
                        result.push("est".to_string());
                    }
                    "n'est" => {
                        result.push("ne".to_string());
                        result.push("est".to_string());
                    }
                    "l'est" => {
                        result.push("le".to_string());
                        result.push("est".to_string());
                    }
                    "d'une" => {
                        result.push("de".to_string());
                        result.push("une".to_string());
                    }
                    "qu'il" => {
                        result.push("que".to_string());
                        result.push("il".to_string());
                    }
                    "qu'elle" => {
                        result.push("que".to_string());
                        result.push("elle".to_string());
                    }
                    _ => {
                        if word.contains('\'') {
                            let parts: Vec<&str> = word.split('\'').collect();
                            if parts.len() == 2 {
                                match parts[0] {
                                    "l" => {
                                        result.push("le".to_string());
                                        result.push(parts[1].to_string());
                                    }
                                    "d" => {
                                        result.push("de".to_string());
                                        result.push(parts[1].to_string());
                                    }
                                    "n" => {
                                        result.push("ne".to_string());
                                        result.push(parts[1].to_string());
                                    }
                                    "c" => {
                                        result.push("ce".to_string());
                                        result.push(parts[1].to_string());
                                    }
                                    "qu" => {
                                        result.push("que".to_string());
                                        result.push(parts[1].to_string());
                                    }
                                    _ => result.push(word),
                                }
                            } else {
                                result.push(word);
                            }
                        } else {
                            result.push(word);
                        }
                    }
                },
                _ => {
                    result.push(word);
                }
            }
        }
        Ok(result)
    }
    /// Handle compound words based on language
    fn handle_compound_words(
        &self,
        words: Vec<String>,
        language: LanguageCode,
    ) -> Result<Vec<String>> {
        let mut result = Vec::new();
        for word in words {
            match language {
                LanguageCode::DeDe if word.len() > 10 => {
                    let parts = self.split_german_compound(&word);
                    result.extend(parts);
                }
                LanguageCode::DeDe => {
                    result.push(word);
                }
                _ => {
                    result.push(word);
                }
            }
        }
        Ok(result)
    }
    /// Simple German compound word splitting
    fn split_german_compound(&self, word: &str) -> Vec<String> {
        let joining_morphemes = ["s", "es", "n", "en", "er"];
        for i in 3..word.len() - 3 {
            let left = &word[..i];
            let right = &word[i..];
            for morpheme in &joining_morphemes {
                if left.ends_with(morpheme) && right.len() > 3 {
                    return vec![
                        left[..left.len() - morpheme.len()].to_string(),
                        right.to_string(),
                    ];
                }
            }
        }
        vec![word.to_string()]
    }
    /// Handle hyphenated words
    fn handle_hyphenated_words(
        &self,
        words: Vec<String>,
        _language: LanguageCode,
    ) -> Result<Vec<String>> {
        let mut result = Vec::new();
        for word in words {
            if word.contains('-') {
                let parts: Vec<String> = word
                    .split('-')
                    .map(|s| s.to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                if parts.len() > 1 {
                    result.extend(parts);
                } else {
                    result.push(word);
                }
            } else {
                result.push(word);
            }
        }
        Ok(result)
    }
}
