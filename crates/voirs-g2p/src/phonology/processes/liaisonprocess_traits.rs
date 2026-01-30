//! # LiaisonProcess - Trait Implementations
//!
//! This module contains trait implementations for `LiaisonProcess`.
//!
//! ## Implemented Traits
//!
//! - `PhonologicalProcess`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::{PhonologicalProcess, ProcessConfig, ProcessType};
use crate::{LanguageCode, Phoneme, Result};

use super::types::*;

impl PhonologicalProcess for LiaisonProcess {
    fn apply(&self, phonemes: &[Phoneme], _config: &ProcessConfig) -> Result<Vec<Phoneme>> {
        Ok(phonemes.to_vec())
    }
    fn process_type(&self) -> ProcessType {
        ProcessType::Liaison
    }
    fn name(&self) -> &str {
        "Liaison"
    }
}
