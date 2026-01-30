//! # TFlappingProcess - Trait Implementations
//!
//! This module contains trait implementations for `TFlappingProcess`.
//!
//! ## Implemented Traits
//!
//! - `PhonologicalProcess`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::{PhonologicalProcess, ProcessConfig, ProcessType};
use crate::{LanguageCode, Phoneme, Result};

use super::types::*;

impl PhonologicalProcess for TFlappingProcess {
    fn apply(&self, phonemes: &[Phoneme], config: &ProcessConfig) -> Result<Vec<Phoneme>> {
        if !self.applies_for_language() {
            return Ok(phonemes.to_vec());
        }
        let mut result = Vec::with_capacity(phonemes.len());
        for i in 0..phonemes.len() {
            let phoneme = &phonemes[i];
            let prev_is_vowel = i > 0 && self.is_vowel(&phonemes[i - 1]);
            let next_is_vowel = i + 1 < phonemes.len() && self.is_vowel(&phonemes[i + 1]);
            if self.is_flappable(phoneme)
                && prev_is_vowel
                && next_is_vowel
                && config.aggressiveness > 0.5
            {
                result.push(self.to_flap(phoneme));
                continue;
            }
            result.push(phoneme.clone());
        }
        Ok(result)
    }
    fn process_type(&self) -> ProcessType {
        ProcessType::TFlapping
    }
    fn name(&self) -> &str {
        "T-Flapping"
    }
}
