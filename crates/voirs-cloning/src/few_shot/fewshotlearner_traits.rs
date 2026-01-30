//! # FewShotLearner - Trait Implementations
//!
//! This module contains trait implementations for `FewShotLearner`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl std::fmt::Debug for FewShotLearner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FewShotLearner")
            .field("config", &self.config)
            .field("device", &format!("{:?}", self.device))
            .field("meta_model", &self.meta_model.is_some())
            .field("training_history_len", &self.training_history.len())
            .field("metrics", &self.metrics)
            .finish()
    }
}
