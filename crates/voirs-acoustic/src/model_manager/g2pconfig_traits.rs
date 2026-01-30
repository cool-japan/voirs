//! # G2pConfig - Trait Implementations
//!
//! This module contains trait implementations for `G2pConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::structs::{
    G2pConfig, PhonemeSet, StressConfig, UnknownWordStrategy, VariantPreferences,
};
use crate::LanguageCode;
use std::collections::HashMap;

impl Default for G2pConfig {
    fn default() -> Self {
        let mut phoneme_sets = HashMap::new();
        phoneme_sets.insert(
            LanguageCode::EnUs,
            PhonemeSet {
                symbols: vec![
                    "AA".to_string(),
                    "AE".to_string(),
                    "AH".to_string(),
                    "AO".to_string(),
                    "AW".to_string(),
                    "AY".to_string(),
                    "EH".to_string(),
                    "ER".to_string(),
                    "EY".to_string(),
                    "IH".to_string(),
                    "IY".to_string(),
                    "OW".to_string(),
                    "OY".to_string(),
                    "UH".to_string(),
                    "UW".to_string(),
                    "B".to_string(),
                    "CH".to_string(),
                    "D".to_string(),
                    "DH".to_string(),
                    "F".to_string(),
                    "G".to_string(),
                    "HH".to_string(),
                    "JH".to_string(),
                    "K".to_string(),
                    "L".to_string(),
                    "M".to_string(),
                    "N".to_string(),
                    "NG".to_string(),
                    "P".to_string(),
                    "R".to_string(),
                    "S".to_string(),
                    "SH".to_string(),
                    "T".to_string(),
                    "TH".to_string(),
                    "V".to_string(),
                    "W".to_string(),
                    "Y".to_string(),
                    "Z".to_string(),
                    "ZH".to_string(),
                ],
                vowels: vec![
                    "AA".to_string(),
                    "AE".to_string(),
                    "AH".to_string(),
                    "AO".to_string(),
                    "AW".to_string(),
                    "AY".to_string(),
                    "EH".to_string(),
                    "ER".to_string(),
                    "EY".to_string(),
                    "IH".to_string(),
                    "IY".to_string(),
                    "OW".to_string(),
                    "OY".to_string(),
                    "UH".to_string(),
                    "UW".to_string(),
                ],
                consonants: vec![
                    "B".to_string(),
                    "CH".to_string(),
                    "D".to_string(),
                    "DH".to_string(),
                    "F".to_string(),
                    "G".to_string(),
                    "HH".to_string(),
                    "JH".to_string(),
                    "K".to_string(),
                    "L".to_string(),
                    "M".to_string(),
                    "N".to_string(),
                    "NG".to_string(),
                    "P".to_string(),
                    "R".to_string(),
                    "S".to_string(),
                    "SH".to_string(),
                    "T".to_string(),
                    "TH".to_string(),
                    "V".to_string(),
                    "W".to_string(),
                    "Y".to_string(),
                    "Z".to_string(),
                    "ZH".to_string(),
                ],
                special_symbols: vec![
                    "_".to_string(),
                    " ".to_string(),
                    "<pad>".to_string(),
                    "<unk>".to_string(),
                    "<bos>".to_string(),
                    "<eos>".to_string(),
                ],
                stress_markers: vec!["0".to_string(), "1".to_string(), "2".to_string()],
            },
        );
        let mut dictionaries = HashMap::new();
        dictionaries.insert(LanguageCode::EnUs, "cmudict.dict".to_string());
        Self {
            engine: crate::config::G2pEngine::Hybrid,
            phoneme_sets,
            dictionaries,
            stress_config: StressConfig::default(),
            unknown_word_strategy: UnknownWordStrategy::FallbackRules,
            variant_preferences: VariantPreferences::default(),
        }
    }
}
