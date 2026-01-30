//! # HDroppingProcess - Trait Implementations
//!
//! This module contains trait implementations for `HDroppingProcess`.
//!
//! ## Implemented Traits
//!
//! - `PhonologicalProcess`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::{PhonologicalProcess, ProcessConfig, ProcessType};
use crate::{Phoneme, Result};

use super::types::HDroppingProcess;

impl PhonologicalProcess for HDroppingProcess {
    fn apply(&self, phonemes: &[Phoneme], config: &ProcessConfig) -> Result<Vec<Phoneme>> {
        if !self.applies_for_language() {
            return Ok(phonemes.to_vec());
        }

        if config.aggressiveness < 0.4 {
            return Ok(phonemes.to_vec());
        }

        let mut result = Vec::with_capacity(phonemes.len());

        for (i, phoneme) in phonemes.iter().enumerate() {
            // Drop H if conditions are met
            if self.should_drop_h(phoneme, i, phonemes) {
                continue; // Skip this phoneme (drop it)
            }
            result.push(phoneme.clone());
        }

        Ok(result)
    }

    fn process_type(&self) -> ProcessType {
        ProcessType::HDropping
    }

    fn name(&self) -> &str {
        "H-Dropping"
    }
}
