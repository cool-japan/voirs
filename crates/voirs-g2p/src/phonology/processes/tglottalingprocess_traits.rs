//! # TGlottalingProcess - Trait Implementations
//!
//! This module contains trait implementations for `TGlottalingProcess`.
//!
//! ## Implemented Traits
//!
//! - `PhonologicalProcess`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::{PhonologicalProcess, ProcessConfig, ProcessType};
use crate::{Phoneme, Result};

use super::types::TGlottalingProcess;

impl PhonologicalProcess for TGlottalingProcess {
    fn apply(&self, phonemes: &[Phoneme], config: &ProcessConfig) -> Result<Vec<Phoneme>> {
        if !self.applies_for_language() {
            return Ok(phonemes.to_vec());
        }

        if config.aggressiveness < 0.5 {
            return Ok(phonemes.to_vec());
        }

        let mut result = Vec::with_capacity(phonemes.len());

        for (i, phoneme) in phonemes.iter().enumerate() {
            // Glottalize /t/ if in appropriate position
            if self.is_t_phoneme(phoneme) && self.is_coda_or_intervocalic_unstressed(i, phonemes) {
                result.push(self.glottalize_t(phoneme));
            } else {
                result.push(phoneme.clone());
            }
        }

        Ok(result)
    }

    fn process_type(&self) -> ProcessType {
        ProcessType::TGlottaling
    }

    fn name(&self) -> &str {
        "T-Glottaling"
    }
}
