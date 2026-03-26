//! # KnowledgeTransferAssessment - Trait Implementations
//!
//! This module contains trait implementations for `KnowledgeTransferAssessment`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::KnowledgeTransferAssessment;

impl Default for KnowledgeTransferAssessment {
    fn default() -> Self {
        Self {
            phonetic_knowledge_transfer: 0.5,
            prosodic_knowledge_transfer: 0.5,
            acoustic_knowledge_transfer: 0.5,
            linguistic_knowledge_transfer: 0.5,
            cultural_knowledge_transfer: 0.5,
            overall_knowledge_transfer: 0.5,
            transfer_efficiency: 0.5,
            transfer_consistency: 0.5,
            transfer_coverage: HashMap::new(),
        }
    }
}
