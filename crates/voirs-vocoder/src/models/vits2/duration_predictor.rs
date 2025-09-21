//! Duration predictor for VITS2

use serde::{Deserialize, Serialize};

/// Duration predictor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurationPredictorConfig {
    /// Hidden channels
    pub hidden_channels: u32,
}

impl Default for DurationPredictorConfig {
    fn default() -> Self {
        Self {
            hidden_channels: 256,
        }
    }
}
