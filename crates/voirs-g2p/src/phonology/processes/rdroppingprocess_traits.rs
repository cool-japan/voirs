//! # RDroppingProcess - Trait Implementations
//!
//! This module contains trait implementations for `RDroppingProcess`.
//!
//! ## Implemented Traits
//!
//! - `PhonologicalProcess`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::{PhonologicalProcess, ProcessConfig, ProcessType};
use crate::{LanguageCode, Phoneme, Result};

use super::types::*;

impl PhonologicalProcess for RDroppingProcess {
    fn apply(&self, phonemes: &[Phoneme], config: &ProcessConfig) -> Result<Vec<Phoneme>> {
        if !self.applies_for_language() {
            return Ok(phonemes.to_vec());
        }
        let mut result = Vec::with_capacity(phonemes.len());
        for (i, phoneme) in phonemes.iter().enumerate() {
            if self.is_r_phoneme(phoneme)
                && config.aggressiveness > 0.3
                && self.should_drop_r(phonemes, i)
            {
                continue;
            }
            result.push(phoneme.clone());
        }
        Ok(result)
    }
    fn process_type(&self) -> ProcessType {
        ProcessType::RDropping
    }
    fn name(&self) -> &str {
        "R-Dropping"
    }
}
