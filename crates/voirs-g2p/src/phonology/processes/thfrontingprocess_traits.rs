//! # ThFrontingProcess - Trait Implementations
//!
//! This module contains trait implementations for `ThFrontingProcess`.
//!
//! ## Implemented Traits
//!
//! - `PhonologicalProcess`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::{PhonologicalProcess, ProcessConfig, ProcessType};
use crate::{Phoneme, Result};

use super::types::ThFrontingProcess;

impl PhonologicalProcess for ThFrontingProcess {
    fn apply(&self, phonemes: &[Phoneme], config: &ProcessConfig) -> Result<Vec<Phoneme>> {
        if !self.applies_for_language() {
            return Ok(phonemes.to_vec());
        }

        if config.aggressiveness < 0.5 {
            return Ok(phonemes.to_vec());
        }

        let mut result = Vec::with_capacity(phonemes.len());

        for phoneme in phonemes.iter() {
            // Apply TH-fronting if this is a dental fricative
            if self.is_dental_fricative(phoneme) {
                result.push(self.front_dental(phoneme));
            } else {
                result.push(phoneme.clone());
            }
        }

        Ok(result)
    }

    fn process_type(&self) -> ProcessType {
        ProcessType::ThFronting
    }

    fn name(&self) -> &str {
        "TH-Fronting"
    }
}
