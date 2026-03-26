//! # NegativeTransferDetectionResult - Trait Implementations
//!
//! This module contains trait implementations for `NegativeTransferDetectionResult`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::NegativeTransferDetectionResult;

impl Default for NegativeTransferDetectionResult {
    fn default() -> Self {
        Self {
            negative_transfer_detected: false,
            negative_transfer_severity: 0.0,
            affected_language_pairs: Vec::new(),
            negative_transfer_sources: Vec::new(),
            mitigation_strategies: Vec::new(),
            performance_degradation: HashMap::new(),
        }
    }
}
