//! # VoicingAssimilationProcess - Trait Implementations
//!
//! This module contains trait implementations for `VoicingAssimilationProcess`.
//!
//! ## Implemented Traits
//!
//! - `PhonologicalProcess`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::{PhonologicalProcess, ProcessConfig, ProcessType};
use crate::{LanguageCode, Phoneme, Result};

use super::types::*;

impl PhonologicalProcess for VoicingAssimilationProcess {
    fn apply(&self, phonemes: &[Phoneme], config: &ProcessConfig) -> Result<Vec<Phoneme>> {
        match self.language {
            LanguageCode::De => {
                let mut result = Vec::new();
                for i in 0..phonemes.len() {
                    let current = phonemes[i].symbol.as_str();
                    if i + 1 < phonemes.len() && config.aggressiveness > 0.5 {
                        let next = phonemes[i + 1].symbol.as_str();
                        if self.should_voice_assimilate(current, next) {
                            if let Some(voiced) = self.get_voiced_counterpart(current) {
                                result.push(Phoneme::new(voiced));
                                continue;
                            }
                        }
                    }
                    result.push(phonemes[i].clone());
                }
                Ok(result)
            }
            _ => Ok(phonemes.to_vec()),
        }
    }
    fn process_type(&self) -> ProcessType {
        ProcessType::VoicingAssimilation
    }
    fn name(&self) -> &str {
        "Voicing Assimilation"
    }
}
