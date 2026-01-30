//! # YodCoalescenceProcess - Trait Implementations
//!
//! This module contains trait implementations for `YodCoalescenceProcess`.
//!
//! ## Implemented Traits
//!
//! - `PhonologicalProcess`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::{PhonologicalProcess, ProcessConfig, ProcessType};
use crate::{Phoneme, Result};

use super::types::YodCoalescenceProcess;

impl PhonologicalProcess for YodCoalescenceProcess {
    fn apply(&self, phonemes: &[Phoneme], config: &ProcessConfig) -> Result<Vec<Phoneme>> {
        if !self.applies_for_language() {
            return Ok(phonemes.to_vec());
        }

        if config.aggressiveness < 0.4 {
            return Ok(phonemes.to_vec());
        }

        let mut result = Vec::with_capacity(phonemes.len());
        let mut i = 0;

        while i < phonemes.len() {
            // Check for /tj/ or /dj/ sequence
            if i + 1 < phonemes.len()
                && self.is_alveolar_stop(&phonemes[i])
                && self.is_yod(&phonemes[i + 1])
            {
                // Coalesce the two phonemes into an affricate
                result.push(self.coalesce(&phonemes[i], &phonemes[i + 1]));
                i += 2; // Skip both phonemes
            } else {
                result.push(phonemes[i].clone());
                i += 1;
            }
        }

        Ok(result)
    }

    fn process_type(&self) -> ProcessType {
        ProcessType::YodCoalescence
    }

    fn name(&self) -> &str {
        "Yod-Coalescence"
    }
}
