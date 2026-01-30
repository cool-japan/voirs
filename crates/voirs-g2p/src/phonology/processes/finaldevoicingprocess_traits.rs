//! # FinalDevoicingProcess - Trait Implementations
//!
//! This module contains trait implementations for `FinalDevoicingProcess`.
//!
//! ## Implemented Traits
//!
//! - `PhonologicalProcess`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::{PhonologicalProcess, ProcessConfig, ProcessType};
use crate::{LanguageCode, Phoneme, Result};

use super::types::*;

impl PhonologicalProcess for FinalDevoicingProcess {
    fn apply(&self, phonemes: &[Phoneme], config: &ProcessConfig) -> Result<Vec<Phoneme>> {
        if !self.applies_for_language() {
            return Ok(phonemes.to_vec());
        }
        let mut result = Vec::with_capacity(phonemes.len());
        for (i, phoneme) in phonemes.iter().enumerate() {
            let is_final = i == phonemes.len() - 1;
            if is_final && self.is_voiced_obstruent(phoneme) && config.aggressiveness > 0.4 {
                if let Some(devoiced) = self.devoice(phoneme) {
                    result.push(devoiced);
                    continue;
                }
            }
            result.push(phoneme.clone());
        }
        Ok(result)
    }
    fn process_type(&self) -> ProcessType {
        ProcessType::FinalDevoicing
    }
    fn name(&self) -> &str {
        "Final Devoicing"
    }
}
