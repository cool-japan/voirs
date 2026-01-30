//! UnivNet configuration and model variants.

use crate::Result;
use serde::{Deserialize, Serialize};

/// UnivNet model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnivNetConfig {
    /// Number of mel-spectrogram channels
    pub num_mels: usize,

    /// Sample rate (Hz)
    pub sample_rate: u32,

    /// Number of upsampling layers
    pub num_upsamples: usize,

    /// Upsampling rates for each layer
    pub upsample_rates: Vec<usize>,

    /// Kernel sizes for upsampling layers
    pub upsample_kernel_sizes: Vec<usize>,

    /// Initial number of channels after first convolution
    pub upsample_initial_channel: usize,

    /// Number of LVC (Location-Variable Convolution) blocks per upsampling layer
    pub num_lvc_blocks: usize,

    /// LVC block kernel sizes
    pub lvc_kernel_sizes: Vec<usize>,

    /// LVC block dilations
    pub lvc_dilations: Vec<Vec<usize>>,

    /// Location kernel size for LVC
    pub location_kernel_size: usize,

    /// Conditional normalization configuration
    pub use_cond_norm: bool,

    /// Discriminator configuration for training (not used in inference)
    pub discriminator_config: Option<DiscriminatorConfig>,
}

/// Configuration for discriminators (used in training)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscriminatorConfig {
    /// Multi-Period Discriminator periods
    pub mpd_periods: Vec<usize>,

    /// Multi-Resolution Spectrogram Discriminator FFT sizes
    pub mrsd_fft_sizes: Vec<usize>,

    /// MRSD hop lengths
    pub mrsd_hop_lengths: Vec<usize>,

    /// MRSD window lengths
    pub mrsd_window_lengths: Vec<usize>,
}

impl Default for DiscriminatorConfig {
    fn default() -> Self {
        Self {
            mpd_periods: vec![2, 3, 5, 7, 11],
            mrsd_fft_sizes: vec![1024, 2048, 512],
            mrsd_hop_lengths: vec![120, 240, 50],
            mrsd_window_lengths: vec![600, 1200, 240],
        }
    }
}

impl UnivNetConfig {
    /// Fast 24kHz configuration (4.5M parameters, MOS 4.21)
    ///
    /// Optimized for low-latency synthesis with good quality.
    pub fn fast_24khz() -> Self {
        Self {
            num_mels: 80,
            sample_rate: 24000,
            num_upsamples: 4,
            upsample_rates: vec![5, 4, 3, 2],
            upsample_kernel_sizes: vec![11, 9, 7, 5],
            upsample_initial_channel: 384,
            num_lvc_blocks: 2,
            lvc_kernel_sizes: vec![3, 7],
            lvc_dilations: vec![vec![1, 3, 5], vec![1, 3, 5]],
            location_kernel_size: 5,
            use_cond_norm: false, // Disabled for simplicity in fast config
            discriminator_config: Some(DiscriminatorConfig::default()),
        }
    }

    /// Base 24kHz configuration (8.9M parameters, MOS 4.42)
    ///
    /// Balanced quality and speed for production use.
    pub fn base_24khz() -> Self {
        Self {
            num_mels: 80,
            sample_rate: 24000,
            num_upsamples: 4,
            upsample_rates: vec![8, 5, 4, 2],
            upsample_kernel_sizes: vec![16, 11, 9, 5],
            upsample_initial_channel: 512,
            num_lvc_blocks: 3,
            lvc_kernel_sizes: vec![3, 7, 11],
            lvc_dilations: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            location_kernel_size: 7,
            use_cond_norm: true,
            discriminator_config: Some(DiscriminatorConfig::default()),
        }
    }

    /// Large 24kHz configuration (15.2M parameters, MOS 4.51)
    ///
    /// Highest quality for research and production applications.
    pub fn large_24khz() -> Self {
        Self {
            num_mels: 80,
            sample_rate: 24000,
            num_upsamples: 4,
            upsample_rates: vec![8, 8, 4, 2],
            upsample_kernel_sizes: vec![16, 16, 9, 5],
            upsample_initial_channel: 640,
            num_lvc_blocks: 4,
            lvc_kernel_sizes: vec![3, 7, 11, 15],
            lvc_dilations: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            location_kernel_size: 9,
            use_cond_norm: true,
            discriminator_config: Some(DiscriminatorConfig::default()),
        }
    }

    /// 16kHz configuration for lower sample rate applications
    pub fn base_16khz() -> Self {
        Self {
            num_mels: 80,
            sample_rate: 16000,
            num_upsamples: 3,
            upsample_rates: vec![8, 5, 2],
            upsample_kernel_sizes: vec![16, 11, 5],
            upsample_initial_channel: 512,
            num_lvc_blocks: 3,
            lvc_kernel_sizes: vec![3, 7, 11],
            lvc_dilations: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            location_kernel_size: 7,
            use_cond_norm: true,
            discriminator_config: Some(DiscriminatorConfig::default()),
        }
    }

    /// 48kHz high-resolution configuration
    pub fn base_48khz() -> Self {
        Self {
            num_mels: 80,
            sample_rate: 48000,
            num_upsamples: 5,
            upsample_rates: vec![6, 5, 4, 3, 2],
            upsample_kernel_sizes: vec![13, 11, 9, 7, 5],
            upsample_initial_channel: 512,
            num_lvc_blocks: 3,
            lvc_kernel_sizes: vec![3, 7, 11],
            lvc_dilations: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            location_kernel_size: 7,
            use_cond_norm: true,
            discriminator_config: Some(DiscriminatorConfig::default()),
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.num_mels == 0 {
            return Err(crate::VocoderError::ConfigError(
                "num_mels must be > 0".to_string(),
            ));
        }

        if self.sample_rate == 0 {
            return Err(crate::VocoderError::ConfigError(
                "sample_rate must be > 0".to_string(),
            ));
        }

        if self.num_upsamples == 0 {
            return Err(crate::VocoderError::ConfigError(
                "num_upsamples must be > 0".to_string(),
            ));
        }

        if self.upsample_rates.len() != self.num_upsamples {
            return Err(crate::VocoderError::ConfigError(format!(
                "upsample_rates length {} doesn't match num_upsamples {}",
                self.upsample_rates.len(),
                self.num_upsamples
            )));
        }

        if self.upsample_kernel_sizes.len() != self.num_upsamples {
            return Err(crate::VocoderError::ConfigError(format!(
                "upsample_kernel_sizes length {} doesn't match num_upsamples {}",
                self.upsample_kernel_sizes.len(),
                self.num_upsamples
            )));
        }

        if self.num_lvc_blocks == 0 {
            return Err(crate::VocoderError::ConfigError(
                "num_lvc_blocks must be > 0".to_string(),
            ));
        }

        if self.lvc_kernel_sizes.len() != self.num_lvc_blocks {
            return Err(crate::VocoderError::ConfigError(format!(
                "lvc_kernel_sizes length {} doesn't match num_lvc_blocks {}",
                self.lvc_kernel_sizes.len(),
                self.num_lvc_blocks
            )));
        }

        if self.lvc_dilations.len() != self.num_lvc_blocks {
            return Err(crate::VocoderError::ConfigError(format!(
                "lvc_dilations length {} doesn't match num_lvc_blocks {}",
                self.lvc_dilations.len(),
                self.num_lvc_blocks
            )));
        }

        Ok(())
    }

    /// Calculate total upsampling factor
    pub fn total_upsample_factor(&self) -> usize {
        self.upsample_rates.iter().product()
    }

    /// Estimate number of parameters
    pub fn estimate_parameters(&self) -> usize {
        let mut params = 0;

        // Input convolution
        params += self.upsample_initial_channel * self.num_mels * 7;

        // Upsampling layers
        let mut current_channels = self.upsample_initial_channel;
        for i in 0..self.num_upsamples {
            let next_channels = current_channels / 2;
            params += current_channels * next_channels * self.upsample_kernel_sizes[i];

            // LVC blocks
            for j in 0..self.num_lvc_blocks {
                let kernel_size = self.lvc_kernel_sizes[j];
                let num_dilations = self.lvc_dilations[j].len();

                // LVC convolution parameters
                params += next_channels * next_channels * kernel_size * num_dilations;

                // Location kernel parameters
                params += next_channels * self.location_kernel_size;
            }

            current_channels = next_channels;
        }

        // Output convolution
        params += current_channels * 7;

        params
    }
}

/// Predefined UnivNet model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnivNetVariant {
    /// Fast 24kHz model (4.5M params, MOS 4.21)
    Fast,
    /// Base 24kHz model (8.9M params, MOS 4.42)
    Base,
    /// Large 24kHz model (15.2M params, MOS 4.51)
    Large,
    /// Base 16kHz model
    Base16k,
    /// Base 48kHz high-resolution model
    Base48k,
}

impl UnivNetVariant {
    /// Get configuration for this variant
    pub fn config(&self) -> UnivNetConfig {
        match self {
            Self::Fast => UnivNetConfig::fast_24khz(),
            Self::Base => UnivNetConfig::base_24khz(),
            Self::Large => UnivNetConfig::large_24khz(),
            Self::Base16k => UnivNetConfig::base_16khz(),
            Self::Base48k => UnivNetConfig::base_48khz(),
        }
    }

    /// Get model name for downloading pretrained weights
    pub fn model_name(&self) -> &'static str {
        match self {
            Self::Fast => "univnet-fast-24khz",
            Self::Base => "univnet-base-24khz",
            Self::Large => "univnet-large-24khz",
            Self::Base16k => "univnet-base-16khz",
            Self::Base48k => "univnet-base-48khz",
        }
    }

    /// Get expected sample rate
    pub fn sample_rate(&self) -> u32 {
        match self {
            Self::Fast | Self::Base | Self::Large => 24000,
            Self::Base16k => 16000,
            Self::Base48k => 48000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let config = UnivNetConfig::base_24khz();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_invalid_mels() {
        let mut config = UnivNetConfig::base_24khz();
        config.num_mels = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_invalid_sample_rate() {
        let mut config = UnivNetConfig::base_24khz();
        config.sample_rate = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_mismatched_upsample_rates() {
        let mut config = UnivNetConfig::base_24khz();
        config.upsample_rates.push(2);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_total_upsample_factor() {
        let config = UnivNetConfig::base_24khz();
        let factor = config.total_upsample_factor();
        assert_eq!(factor, 8 * 5 * 4 * 2);
    }

    #[test]
    fn test_parameter_estimation() {
        let configs = [
            UnivNetConfig::fast_24khz(),
            UnivNetConfig::base_24khz(),
            UnivNetConfig::large_24khz(),
        ];

        for config in configs {
            let params = config.estimate_parameters();
            assert!(params > 1_000_000); // At least 1M parameters
            assert!(params < 20_000_000); // Less than 20M parameters
        }
    }

    #[test]
    fn test_all_variants() {
        for variant in [
            UnivNetVariant::Fast,
            UnivNetVariant::Base,
            UnivNetVariant::Large,
            UnivNetVariant::Base16k,
            UnivNetVariant::Base48k,
        ] {
            let config = variant.config();
            assert!(config.validate().is_ok());
            assert_eq!(config.sample_rate, variant.sample_rate());
        }
    }

    #[test]
    fn test_discriminator_config_default() {
        let disc_config = DiscriminatorConfig::default();
        assert_eq!(disc_config.mpd_periods.len(), 5);
        assert_eq!(disc_config.mrsd_fft_sizes.len(), 3);
        assert_eq!(disc_config.mrsd_hop_lengths.len(), 3);
        assert_eq!(disc_config.mrsd_window_lengths.len(), 3);
    }
}
