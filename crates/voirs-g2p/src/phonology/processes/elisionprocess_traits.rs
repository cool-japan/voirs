//! # ElisionProcess - Trait Implementations
//!
//! This module contains trait implementations for `ElisionProcess`.
//!
//! ## Implemented Traits
//!
//! - `PhonologicalProcess`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::{PhonologicalProcess, ProcessConfig, ProcessType};
use crate::{LanguageCode, Phoneme, Result};

use super::types::*;

impl PhonologicalProcess for ElisionProcess {
    fn apply(&self, phonemes: &[Phoneme], config: &ProcessConfig) -> Result<Vec<Phoneme>> {
        if config.aggressiveness < 0.8 {
            return Ok(phonemes.to_vec());
        }
        let mut result = Vec::new();
        for (i, phoneme) in phonemes.iter().enumerate() {
            if !self.can_elide(phoneme, i, phonemes) {
                result.push(phoneme.clone());
            }
        }
        Ok(result)
    }
    fn process_type(&self) -> ProcessType {
        ProcessType::Elision
    }
    fn name(&self) -> &str {
        "Elision"
    }
}
