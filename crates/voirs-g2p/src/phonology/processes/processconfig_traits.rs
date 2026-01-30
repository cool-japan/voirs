//! # ProcessConfig - Trait Implementations
//!
//! This module contains trait implementations for `ProcessConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::{PhonologicalProcess, ProcessConfig, ProcessType};
use crate::{LanguageCode, Phoneme, Result};

use super::types::*;

impl Default for ProcessConfig {
    fn default() -> Self {
        Self {
            enable_place_assimilation: true,
            enable_voicing_assimilation: false,
            enable_vowel_reduction: false,
            enable_elision: false,
            enable_liaison: false,
            enable_nasalization: false,
            enable_palatalization: false,
            enable_lenition: false,
            enable_fortition: false,
            enable_r_dropping: false,
            enable_t_flapping: false,
            enable_final_devoicing: false,
            enable_vowel_devoicing: false,
            enable_h_dropping: false,
            enable_th_fronting: false,
            enable_l_vocalization: false,
            enable_g_dropping: false,
            enable_t_glottaling: false,
            enable_yod_coalescence: false,
            language: LanguageCode::EnUs,
            aggressiveness: 0.5,
        }
    }
}
