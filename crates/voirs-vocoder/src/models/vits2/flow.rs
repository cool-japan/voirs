//! Flow-based models for VITS2

use serde::{Deserialize, Serialize};

/// Flow configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowConfig {
    /// Number of flow blocks
    pub n_blocks: u32,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self { n_blocks: 4 }
    }
}
