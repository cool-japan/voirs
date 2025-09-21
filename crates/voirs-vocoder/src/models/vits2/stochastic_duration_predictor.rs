//! Stochastic duration predictor for VITS2

use serde::{Deserialize, Serialize};

/// Stochastic duration predictor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StochasticDurationPredictorConfig {
    /// Hidden channels
    pub hidden_channels: u32,
    /// Number of flows
    pub n_flows: u32,
}

impl Default for StochasticDurationPredictorConfig {
    fn default() -> Self {
        Self {
            hidden_channels: 192,
            n_flows: 4,
        }
    }
}
