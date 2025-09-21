//! Posterior encoder for VITS2

use serde::{Deserialize, Serialize};

/// Posterior encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PosteriorEncoderConfig {
    /// Hidden channels
    pub hidden_channels: u32,
}

impl Default for PosteriorEncoderConfig {
    fn default() -> Self {
        Self {
            hidden_channels: 512,
        }
    }
}
