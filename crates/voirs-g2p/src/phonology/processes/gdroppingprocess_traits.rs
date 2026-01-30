//! # GDroppingProcess - Trait Implementations
//!
//! This module contains trait implementations for `GDroppingProcess`.
//!
//! ## Implemented Traits
//!
//! - `PhonologicalProcess`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::{PhonologicalProcess, ProcessConfig, ProcessType};
use crate::{Phoneme, Result};

use super::types::GDroppingProcess;

impl PhonologicalProcess for GDroppingProcess {
    fn apply(&self, phonemes: &[Phoneme], config: &ProcessConfig) -> Result<Vec<Phoneme>> {
        if !self.applies_for_language() {
            return Ok(phonemes.to_vec());
        }

        if config.aggressiveness < 0.6 {
            return Ok(phonemes.to_vec());
        }

        let mut result = Vec::with_capacity(phonemes.len());

        for phoneme in phonemes.iter() {
            // Drop G if it's velar nasal in unstressed syllable
            if self.is_velar_nasal(phoneme) && self.is_unstressed(phoneme) {
                result.push(self.drop_g(phoneme));
            } else {
                result.push(phoneme.clone());
            }
        }

        Ok(result)
    }

    fn process_type(&self) -> ProcessType {
        ProcessType::GDropping
    }

    fn name(&self) -> &str {
        "G-Dropping"
    }
}
