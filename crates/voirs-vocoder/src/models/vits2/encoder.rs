//! Encoder modules for VITS2

use serde::{Deserialize, Serialize};

/// Encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    /// Hidden channels
    pub hidden_channels: u32,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            hidden_channels: 192,
        }
    }
}
