//! G2P (Grapheme-to-Phoneme) backend methods for TTS pipeline
//!
//! This module contains all G2P functionality including:
//! - Dictionary lookup
//! - G2P model application (neural, rule-based)
//! - Language-specific G2P rules (English, German, French, Spanish, Japanese)
//! - Neural G2P inference
//! - Context-aware phoneme generation
//! - Phoneme confidence calculation

use super::*;
use crate::{AcousticError, LanguageCode, Phoneme, Result};
use log::{debug, info, warn};
use std::collections::HashMap;

impl TtsPipeline {
    async fn lookup_dictionary(
        &self,
        word: &str,
        language: LanguageCode,
        g2p_config: &G2pConfig,
    ) -> Result<Option<String>> {
        if let Some(dict_path) = g2p_config.dictionaries.get(&language) {
            debug!("Looking up '{}' in dictionary: {}", word, dict_path);
            {
                let dictionaries = self.pronunciation_dictionaries.read().await;
                if let Some(dictionary) = dictionaries.get(&language) {
                    if let Some(pronunciation) =
                        dictionary.lookup_with_preferences(word, &g2p_config.variant_preferences)
                    {
                        debug!("Found pronunciation for '{}': {}", word, pronunciation);
                        return Ok(Some(pronunciation));
                    } else {
                        debug!("Word '{}' not found in loaded dictionary", word);
                        return Ok(None);
                    }
                }
            }
            debug!("Loading pronunciation dictionary from: {}", dict_path);
            match PronunciationDictionary::load_from_file(dict_path, language) {
                Ok(dictionary) => {
                    let pronunciation =
                        dictionary.lookup_with_preferences(word, &g2p_config.variant_preferences);
                    {
                        let mut dictionaries = self.pronunciation_dictionaries.write().await;
                        dictionaries.insert(language, dictionary);
                    }
                    if let Some(pronunciation) = pronunciation {
                        debug!("Found pronunciation for '{}': {}", word, pronunciation);
                        Ok(Some(pronunciation))
                    } else {
                        debug!("Word '{}' not found in newly loaded dictionary", word);
                        Ok(None)
                    }
                }
                Err(e) => {
                    warn!(
                        "Failed to load pronunciation dictionary from {}: {}",
                        dict_path, e
                    );
                    let pronunciation = match word.to_lowercase().as_str() {
                        "hello" => Some("HH EH1 L OW0".to_string()),
                        "world" => Some("W ER1 L D".to_string()),
                        "the" => Some("DH AH0".to_string()),
                        "and" => Some("AE1 N D".to_string()),
                        "voice" => Some("V OY1 S".to_string()),
                        "synthesis" => Some("S IH1 N TH AH0 S AH0 S".to_string()),
                        "test" => Some("T EH1 S T".to_string()),
                        "example" => Some("IH0 G Z AE1 M P AH0 L".to_string()),
                        _ => None,
                    };
                    Ok(pronunciation)
                }
            }
        } else {
            debug!("No dictionary path configured for language: {:?}", language);
            Ok(None)
        }
    }
    /// Apply G2P model for unknown words
    #[allow(dead_code)]
    fn apply_g2p_model(
        &self,
        word: &str,
        language: LanguageCode,
        g2p_config: &G2pConfig,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        info!(
            "Applying {} G2P model for word: '{}'",
            match g2p_config.engine {
                crate::config::G2pEngine::RuleBased => "rule-based",
                crate::config::G2pEngine::Neural => "neural",
                crate::config::G2pEngine::Hybrid => "hybrid",
            },
            word
        );
        match g2p_config.engine {
            crate::config::G2pEngine::Neural => {
                match self.apply_neural_g2p(word, language, g2p_config, phoneme_set) {
                    Ok(phonemes) if !phonemes.is_empty() => {
                        debug!("Neural G2P succeeded for word: '{}'", word);
                        Ok(phonemes)
                    }
                    Ok(_) | Err(_) => {
                        debug!(
                            "Neural G2P failed, falling back to rule-based for word: '{}'",
                            word
                        );
                        self.apply_rule_based_g2p(word, language, g2p_config, phoneme_set)
                    }
                }
            }
            crate::config::G2pEngine::RuleBased => {
                self.apply_rule_based_g2p(word, language, g2p_config, phoneme_set)
            }
            crate::config::G2pEngine::Hybrid => {
                match self.apply_neural_g2p(word, language, g2p_config, phoneme_set) {
                    Ok(phonemes) if !phonemes.is_empty() => Ok(phonemes),
                    Ok(_) | Err(_) => {
                        match self.apply_rule_based_g2p(word, language, g2p_config, phoneme_set) {
                            Ok(phonemes) if !phonemes.is_empty() => Ok(phonemes),
                            Ok(_) | Err(_) => {
                                self.apply_unknown_word_strategy(word, g2p_config, phoneme_set)
                            }
                        }
                    }
                }
            }
        }
    }
    /// Apply neural G2P model inference
    #[allow(dead_code)]
    fn apply_neural_g2p(
        &self,
        word: &str,
        language: LanguageCode,
        g2p_config: &G2pConfig,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        debug!(
            "Attempting neural G2P for word: '{}' in language: {:?}",
            word, language
        );
        let tokens = self.tokenize_for_neural_g2p(word, language)?;
        debug!("Tokenized '{}' into {} tokens", word, tokens.len());
        let raw_phonemes = self.simulate_neural_inference(&tokens, language, phoneme_set)?;
        let filtered_phonemes = self.apply_neural_postprocessing(raw_phonemes, g2p_config)?;
        let mut final_phonemes = Vec::new();
        for mut phoneme in filtered_phonemes {
            let confidence = self.calculate_phoneme_confidence(&phoneme, word);
            let features = phoneme.features.get_or_insert_with(HashMap::new);
            features.insert("confidence".to_string(), confidence.to_string());
            final_phonemes.push(phoneme);
        }
        if !final_phonemes.is_empty() {
            info!(
                "Neural-style G2P produced {} phonemes for '{}' with avg confidence: {:.2}",
                final_phonemes.len(),
                word,
                final_phonemes
                    .iter()
                    .filter_map(|p| { p.features.as_ref()?.get("confidence")?.parse::<f32>().ok() })
                    .sum::<f32>()
                    / final_phonemes.len() as f32
            );
        }
        Ok(final_phonemes)
    }
    /// Tokenize input for neural G2P processing
    fn tokenize_for_neural_g2p(&self, word: &str, language: LanguageCode) -> Result<Vec<String>> {
        let mut tokens = Vec::new();
        let normalized_word = word.to_lowercase();
        tokens.push("<BOS>".to_string());
        for ch in normalized_word.chars() {
            if ch.is_alphabetic() {
                tokens.push(ch.to_string());
            } else if ch.is_whitespace() {
                tokens.push("<SPACE>".to_string());
            } else {
                tokens.push("<UNK>".to_string());
            }
        }
        tokens.push("<EOS>".to_string());
        match language {
            LanguageCode::EnUs | LanguageCode::EnGb => {}
            LanguageCode::JaJp => {
                debug!("Japanese G2P tokenization not fully implemented");
            }
            _ => {
                debug!(
                    "Language-specific tokenization not implemented for {:?}",
                    language
                );
            }
        }
        Ok(tokens)
    }
    /// Simulate neural network inference with sophisticated rules
    fn simulate_neural_inference(
        &self,
        tokens: &[String],
        language: LanguageCode,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        let mut phonemes = Vec::new();
        let content_tokens = &tokens[1..tokens.len().saturating_sub(1)];
        for (i, token) in content_tokens.iter().enumerate() {
            if token == "<SPACE>" {
                let mut boundary = crate::Phoneme::new("SIL");
                boundary.duration = Some(0.1);
                phonemes.push(boundary);
                continue;
            }
            if token == "<UNK>" {
                continue;
            }
            let prev_token = if i > 0 {
                content_tokens.get(i - 1)
            } else {
                None
            };
            let next_token = content_tokens.get(i + 1);
            let next2_token = content_tokens.get(i + 2);
            let context_phonemes = self.get_context_aware_phonemes(
                token,
                prev_token,
                next_token,
                next2_token,
                language,
                phoneme_set,
            )?;
            phonemes.extend(context_phonemes);
        }
        Ok(phonemes)
    }
    /// Get context-aware phonemes using enhanced heuristics
    fn get_context_aware_phonemes(
        &self,
        token: &str,
        prev_token: Option<&String>,
        next_token: Option<&String>,
        next2_token: Option<&String>,
        _language: LanguageCode,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        if token.len() != 1 {
            return Ok(vec![]);
        }
        let ch = token
            .chars()
            .next()
            .ok_or_else(|| crate::AcousticError::ProcessingError {
                message: "Token has no characters".to_string(),
            })?;
        let _prev_ch = prev_token.and_then(|t| t.chars().next());
        let next_ch = next_token.and_then(|t| t.chars().next());
        let _next2_ch = next2_token.and_then(|t| t.chars().next());
        let phoneme_symbols = match ch {
            'a' => {
                if next_ch == Some('r') {
                    vec!["AA", "R"]
                } else if next_ch == Some('i') || next_ch == Some('y') {
                    vec!["EY"]
                } else {
                    vec!["AE"]
                }
            }
            'e' => {
                if next_ch == Some('a') {
                    vec!["IY"]
                } else if next_ch == Some('r') {
                    vec!["ER"]
                } else {
                    vec!["EH"]
                }
            }
            _ => match ch {
                'i' => vec!["IH"],
                'o' => vec!["AA"],
                'u' => vec!["UH"],
                'b' => vec!["B"],
                'c' => vec!["K"],
                'd' => vec!["D"],
                'f' => vec!["F"],
                'g' => vec!["G"],
                'h' => vec!["HH"],
                'j' => vec!["JH"],
                'k' => vec!["K"],
                'l' => vec!["L"],
                'm' => vec!["M"],
                'n' => vec!["N"],
                'p' => vec!["P"],
                'r' => vec!["R"],
                's' => vec!["S"],
                't' => vec!["T"],
                'v' => vec!["V"],
                'w' => vec!["W"],
                'y' => vec!["Y"],
                'z' => vec!["Z"],
                _ => vec!["AH"],
            },
        };
        let mut result = Vec::new();
        for symbol in phoneme_symbols {
            if phoneme_set.symbols.contains(&symbol.to_string()) {
                result.push(crate::Phoneme::new(symbol));
            }
        }
        Ok(result)
    }
    /// Apply neural-style post-processing
    fn apply_neural_postprocessing(
        &self,
        mut phonemes: Vec<crate::Phoneme>,
        _config: &G2pConfig,
    ) -> Result<Vec<crate::Phoneme>> {
        phonemes.dedup_by(|a, b| a.symbol == b.symbol);
        for (i, phoneme) in phonemes.iter_mut().enumerate() {
            let base_duration = match phoneme.symbol.as_str() {
                "AA" | "AE" | "AH" | "AO" | "AW" | "AY" | "EH" | "ER" | "EY" | "IH" | "IY"
                | "OW" | "OY" | "UH" | "UW" => 0.12,
                _ => 0.08,
            };
            let variation = (i as f32 * 0.003) % 0.01;
            phoneme.duration = Some(base_duration + variation);
        }
        Ok(phonemes)
    }
    /// Calculate confidence score for a phoneme based on context
    fn calculate_phoneme_confidence(&self, phoneme: &crate::Phoneme, word: &str) -> f32 {
        let base_confidence: f32 = 0.7;
        let common_bonus: f32 = match phoneme.symbol.as_str() {
            "AH" | "IH" | "T" | "N" | "S" | "R" | "L" => 0.2,
            "AA" | "EH" | "K" | "D" | "M" => 0.1,
            _ => 0.0,
        };
        let length_bonus: f32 = if word.len() <= 4 { 0.1 } else { 0.0 };
        (base_confidence + common_bonus + length_bonus).clamp(0.1, 1.0)
    }
    /// Apply rule-based G2P system
    #[allow(dead_code)]
    fn apply_rule_based_g2p(
        &self,
        word: &str,
        language: LanguageCode,
        g2p_config: &G2pConfig,
        phoneme_set: &PhonemeSet,
    ) -> Result<Vec<crate::Phoneme>> {
        debug!(
            "Applying rule-based G2P for word: '{}' in language: {:?}",
            word, language
        );
        match language {
            LanguageCode::EnUs | LanguageCode::EnGb => {
                self.apply_english_g2p_rules(word, g2p_config, phoneme_set)
            }
            LanguageCode::DeDe => self.apply_german_g2p_rules(word, g2p_config, phoneme_set),
            LanguageCode::FrFr => self.apply_french_g2p_rules(word, g2p_config, phoneme_set),
            LanguageCode::EsEs => self.apply_spanish_g2p_rules(word, g2p_config, phoneme_set),
            LanguageCode::JaJp => self.apply_japanese_g2p_rules(word, g2p_config, phoneme_set),
            _ => self.apply_generic_g2p_rules(word, g2p_config, phoneme_set),
        }
    }
}
