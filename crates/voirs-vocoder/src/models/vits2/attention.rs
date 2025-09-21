//! Attention mechanisms for VITS2

use serde::{Deserialize, Serialize};

/// Attention configuration placeholder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    /// Number of attention heads
    pub n_heads: u32,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self { n_heads: 2 }
    }
}
