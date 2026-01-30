//! # NasalizationProcess - Trait Implementations
//!
//! This module contains trait implementations for `NasalizationProcess`.
//!
//! ## Implemented Traits
//!
//! - `PhonologicalProcess`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::{PhonologicalProcess, ProcessConfig, ProcessType};
use crate::{LanguageCode, Phoneme, Result};

use super::types::*;

impl PhonologicalProcess for NasalizationProcess {
    fn apply(&self, phonemes: &[Phoneme], config: &ProcessConfig) -> Result<Vec<Phoneme>> {
        if !self.applies_for_language() {
            return Ok(phonemes.to_vec());
        }
        let mut result = Vec::with_capacity(phonemes.len());
        for i in 0..phonemes.len() {
            let phoneme = &phonemes[i];
            if self.is_vowel(phoneme) && i + 1 < phonemes.len() {
                let next = &phonemes[i + 1];
                if self.is_nasal(next) && config.aggressiveness > 0.3 {
                    result.push(self.nasalize_vowel(phoneme));
                    continue;
                }
            }
            result.push(phoneme.clone());
        }
        Ok(result)
    }
    fn process_type(&self) -> ProcessType {
        ProcessType::Nasalization
    }
    fn name(&self) -> &str {
        "Nasalization"
    }
}
