//! # AssimilationProcess - Trait Implementations
//!
//! This module contains trait implementations for `AssimilationProcess`.
//!
//! ## Implemented Traits
//!
//! - `PhonologicalProcess`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::AssimilationProcess;
use super::{PhonologicalProcess, ProcessConfig, ProcessType};
use crate::{LanguageCode, Phoneme, Result};

impl PhonologicalProcess for AssimilationProcess {
    fn apply(&self, phonemes: &[Phoneme], _config: &ProcessConfig) -> Result<Vec<Phoneme>> {
        let mut result = Vec::new();
        for i in 0..phonemes.len() {
            let current = phonemes[i].symbol.as_str();
            if i + 1 < phonemes.len() {
                let next = phonemes[i + 1].symbol.as_str();
                if self.should_assimilate(current, next) {
                    if let Some(assimilated) = self.get_assimilated_phoneme(current, next) {
                        result.push(Phoneme::new(assimilated));
                        continue;
                    }
                }
            }
            result.push(phonemes[i].clone());
        }
        Ok(result)
    }
    fn process_type(&self) -> ProcessType {
        ProcessType::PlaceAssimilation
    }
    fn name(&self) -> &str {
        "Place Assimilation"
    }
}
