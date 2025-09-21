//! HiFi-GAN vocoder implementation.

pub mod generator;
pub mod inference;
pub mod mrf;
pub mod variants;

// Re-export commonly used types
pub use variants::HiFiGanVariants;

use serde::{Deserialize, Serialize};

/// HiFi-GAN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiFiGanConfig {
    /// Generator architecture variant
    pub variant: HiFiGanVariant,
    /// Number of mel channels
    pub mel_channels: u32,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of residual blocks
    pub num_residual_blocks: u32,
    /// Upsampling rates
    pub upsample_rates: Vec<u32>,
    /// Upsampling kernel sizes
    pub upsample_kernel_sizes: Vec<u32>,
    /// MRF kernel sizes
    pub mrf_kernel_sizes: Vec<u32>,
    /// MRF dilation sizes
    pub mrf_dilation_sizes: Vec<Vec<u32>>,
    /// Initial channels
    pub initial_channels: u32,
    /// Leaky ReLU slope
    pub leaky_relu_slope: f32,
}

/// HiFi-GAN variants
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum HiFiGanVariant {
    /// V1 - Highest quality, more parameters
    V1,
    /// V2 - Balanced quality and speed
    V2,
    /// V3 - Fastest, fewer parameters
    V3,
}

impl HiFiGanVariant {
    /// Get variant name
    pub fn name(&self) -> &'static str {
        match self {
            HiFiGanVariant::V1 => "HiFi-GAN V1",
            HiFiGanVariant::V2 => "HiFi-GAN V2",
            HiFiGanVariant::V3 => "HiFi-GAN V3",
        }
    }

    /// Get default configuration for variant
    pub fn default_config(&self) -> HiFiGanConfig {
        match self {
            HiFiGanVariant::V1 => HiFiGanConfig {
                variant: *self,
                mel_channels: 80,
                sample_rate: 22050,
                num_residual_blocks: 3,
                upsample_rates: vec![8, 8, 2, 2],
                upsample_kernel_sizes: vec![16, 16, 4, 4],
                mrf_kernel_sizes: vec![3, 7, 11],
                mrf_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
                initial_channels: 512,
                leaky_relu_slope: 0.1,
            },
            HiFiGanVariant::V2 => HiFiGanConfig {
                variant: *self,
                mel_channels: 80,
                sample_rate: 22050,
                num_residual_blocks: 3,
                upsample_rates: vec![8, 8, 4, 2],
                upsample_kernel_sizes: vec![16, 16, 8, 4],
                mrf_kernel_sizes: vec![3, 7, 11],
                mrf_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
                initial_channels: 256,
                leaky_relu_slope: 0.1,
            },
            HiFiGanVariant::V3 => HiFiGanConfig {
                variant: *self,
                mel_channels: 80,
                sample_rate: 22050,
                num_residual_blocks: 2,
                upsample_rates: vec![8, 8, 8, 2],
                upsample_kernel_sizes: vec![16, 16, 16, 4],
                mrf_kernel_sizes: vec![3, 5, 7],
                mrf_dilation_sizes: vec![vec![1, 2, 4], vec![1, 2, 4], vec![1, 2, 4]],
                initial_channels: 128,
                leaky_relu_slope: 0.1,
            },
        }
    }
}

impl Default for HiFiGanConfig {
    fn default() -> Self {
        HiFiGanVariant::V1.default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hifigan_variants() {
        let v1 = HiFiGanVariant::V1;
        let v2 = HiFiGanVariant::V2;
        let v3 = HiFiGanVariant::V3;

        assert_eq!(v1.name(), "HiFi-GAN V1");
        assert_eq!(v2.name(), "HiFi-GAN V2");
        assert_eq!(v3.name(), "HiFi-GAN V3");

        let config_v1 = v1.default_config();
        let config_v2 = v2.default_config();
        let config_v3 = v3.default_config();

        assert_eq!(config_v1.initial_channels, 512);
        assert_eq!(config_v2.initial_channels, 256);
        assert_eq!(config_v3.initial_channels, 128);

        assert_eq!(config_v1.num_residual_blocks, 3);
        assert_eq!(config_v2.num_residual_blocks, 3);
        assert_eq!(config_v3.num_residual_blocks, 2);
    }

    #[test]
    fn test_hifigan_config() {
        let config = HiFiGanConfig::default();

        assert_eq!(config.mel_channels, 80);
        assert_eq!(config.sample_rate, 22050);
        assert!(!config.upsample_rates.is_empty());
        assert!(!config.mrf_kernel_sizes.is_empty());
        assert_eq!(config.leaky_relu_slope, 0.1);
    }
}
