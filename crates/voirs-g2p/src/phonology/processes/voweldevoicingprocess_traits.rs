//! # VowelDevoicingProcess - Trait Implementations
//!
//! This module contains trait implementations for `VowelDevoicingProcess`.
//!
//! ## Implemented Traits
//!
//! - `PhonologicalProcess`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::{PhonologicalProcess, ProcessConfig, ProcessType};
use crate::{LanguageCode, Phoneme, Result};

use super::types::*;

impl PhonologicalProcess for VowelDevoicingProcess {
    fn apply(&self, phonemes: &[Phoneme], config: &ProcessConfig) -> Result<Vec<Phoneme>> {
        if !self.applies_for_language() {
            return Ok(phonemes.to_vec());
        }
        let mut result = Vec::with_capacity(phonemes.len());
        for i in 0..phonemes.len() {
            let phoneme = &phonemes[i];
            if self.is_high_vowel(phoneme) {
                let prev_is_voiceless = i > 0 && self.is_voiceless(&phonemes[i - 1]);
                let next_is_voiceless =
                    i + 1 < phonemes.len() && self.is_voiceless(&phonemes[i + 1]);
                let is_final = i == phonemes.len() - 1;
                if (prev_is_voiceless && (next_is_voiceless || is_final))
                    && config.aggressiveness > 0.3
                {
                    result.push(self.devoice_vowel(phoneme));
                    continue;
                }
            }
            result.push(phoneme.clone());
        }
        Ok(result)
    }
    fn process_type(&self) -> ProcessType {
        ProcessType::VowelDevoicing
    }
    fn name(&self) -> &str {
        "Vowel Devoicing"
    }
}
