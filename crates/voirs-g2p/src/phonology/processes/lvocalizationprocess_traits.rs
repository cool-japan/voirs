//! # LVocalizationProcess - Trait Implementations
//!
//! This module contains trait implementations for `LVocalizationProcess`.
//!
//! ## Implemented Traits
//!
//! - `PhonologicalProcess`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::{PhonologicalProcess, ProcessConfig, ProcessType};
use crate::{Phoneme, Result};

use super::types::LVocalizationProcess;

impl PhonologicalProcess for LVocalizationProcess {
    fn apply(&self, phonemes: &[Phoneme], config: &ProcessConfig) -> Result<Vec<Phoneme>> {
        if !self.applies_for_language() {
            return Ok(phonemes.to_vec());
        }

        if config.aggressiveness < 0.5 {
            return Ok(phonemes.to_vec());
        }

        let mut result = Vec::with_capacity(phonemes.len());

        for (i, phoneme) in phonemes.iter().enumerate() {
            // Vocalize /l/ if in coda position
            if self.is_l_phoneme(phoneme) && self.is_coda_position(i, phonemes) {
                result.push(self.vocalize_l(phoneme));
            } else {
                result.push(phoneme.clone());
            }
        }

        Ok(result)
    }

    fn process_type(&self) -> ProcessType {
        ProcessType::LVocalization
    }

    fn name(&self) -> &str {
        "L-Vocalization"
    }
}
