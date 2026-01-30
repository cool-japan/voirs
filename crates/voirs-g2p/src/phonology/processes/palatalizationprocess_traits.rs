//! # PalatalizationProcess - Trait Implementations
//!
//! This module contains trait implementations for `PalatalizationProcess`.
//!
//! ## Implemented Traits
//!
//! - `PhonologicalProcess`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::{PhonologicalProcess, ProcessConfig, ProcessType};
use crate::{LanguageCode, Phoneme, Result};

use super::types::*;

impl PhonologicalProcess for PalatalizationProcess {
    fn apply(&self, phonemes: &[Phoneme], config: &ProcessConfig) -> Result<Vec<Phoneme>> {
        if !self.applies_for_language() {
            return Ok(phonemes.to_vec());
        }
        let mut result = Vec::with_capacity(phonemes.len());
        for i in 0..phonemes.len() {
            let phoneme = &phonemes[i];
            if i + 1 < phonemes.len() {
                let next = &phonemes[i + 1];
                if self.is_palatalizing_context(next) && config.aggressiveness > 0.5 {
                    if let Some(palatalized) = self.palatalize(phoneme) {
                        result.push(palatalized);
                        continue;
                    }
                }
            }
            result.push(phoneme.clone());
        }
        Ok(result)
    }
    fn process_type(&self) -> ProcessType {
        ProcessType::Palatalization
    }
    fn name(&self) -> &str {
        "Palatalization"
    }
}
