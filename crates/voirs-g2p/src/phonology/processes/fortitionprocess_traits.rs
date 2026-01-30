//! # FortitionProcess - Trait Implementations
//!
//! This module contains trait implementations for `FortitionProcess`.
//!
//! ## Implemented Traits
//!
//! - `PhonologicalProcess`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::{PhonologicalProcess, ProcessConfig, ProcessType};
use crate::{LanguageCode, Phoneme, Result};

use super::types::*;

impl PhonologicalProcess for FortitionProcess {
    fn apply(&self, phonemes: &[Phoneme], config: &ProcessConfig) -> Result<Vec<Phoneme>> {
        if !self.applies_for_language() {
            return Ok(phonemes.to_vec());
        }
        let mut result = Vec::with_capacity(phonemes.len());
        for i in 0..phonemes.len() {
            let phoneme = &phonemes[i];
            let prev_is_stressed_vowel = i > 0 && {
                let prev = &phonemes[i - 1];
                self.is_vowel(prev) && self.is_stressed(prev)
            };
            let next_is_vowel = i + 1 < phonemes.len() && self.is_vowel(&phonemes[i + 1]);
            if prev_is_stressed_vowel && next_is_vowel && config.aggressiveness > 0.6 {
                if let Some(fortified) = self.fortify(phoneme) {
                    result.push(fortified);
                    continue;
                }
            }
            result.push(phoneme.clone());
        }
        Ok(result)
    }
    fn process_type(&self) -> ProcessType {
        ProcessType::Fortition
    }
    fn name(&self) -> &str {
        "Fortition"
    }
}
