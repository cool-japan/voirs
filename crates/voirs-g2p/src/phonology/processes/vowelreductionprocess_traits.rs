//! # VowelReductionProcess - Trait Implementations
//!
//! This module contains trait implementations for `VowelReductionProcess`.
//!
//! ## Implemented Traits
//!
//! - `PhonologicalProcess`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::{PhonologicalProcess, ProcessConfig, ProcessType};
use crate::{LanguageCode, Phoneme, Result};

use super::types::*;

impl PhonologicalProcess for VowelReductionProcess {
    fn apply(&self, phonemes: &[Phoneme], config: &ProcessConfig) -> Result<Vec<Phoneme>> {
        let mut result = Vec::new();
        for phoneme in phonemes {
            if self.should_reduce(phoneme) && config.aggressiveness > 0.7 {
                let reduced = self.reduce_vowel(&phoneme.symbol);
                let mut new_phoneme = phoneme.clone();
                new_phoneme.symbol = reduced;
                result.push(new_phoneme);
            } else {
                result.push(phoneme.clone());
            }
        }
        Ok(result)
    }
    fn process_type(&self) -> ProcessType {
        ProcessType::VowelReduction
    }
    fn name(&self) -> &str {
        "Vowel Reduction"
    }
}
