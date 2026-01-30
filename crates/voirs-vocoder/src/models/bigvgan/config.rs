//! BigVGAN configuration structures.
//!
//! Defines the architecture and hyperparameters for BigVGAN vocoder.

use super::activation::ActivationConfig;
use serde::{Deserialize, Serialize};

/// BigVGAN model variants with different trade-offs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BigVGANVariant {
    /// Base variant (24kHz, good quality/speed trade-off)
    Base,
    /// Large variant (24kHz, highest quality)
    Large,
    /// Fast variant (24kHz, optimized for speed)
    Fast,
    /// Ultra variant (48kHz, ultra-high quality)
    Ultra,
}

impl BigVGANVariant {
    /// Get the model name for loading from HuggingFace
    pub fn model_name(&self) -> &'static str {
        match self {
            BigVGANVariant::Base => "bigvgan_base_24khz",
            BigVGANVariant::Large => "bigvgan_large_24khz",
            BigVGANVariant::Fast => "bigvgan_fast_24khz",
            BigVGANVariant::Ultra => "bigvgan_ultra_48khz",
        }
    }

    /// Get the default configuration for this variant
    pub fn config(&self) -> BigVGANConfig {
        match self {
            BigVGANVariant::Base => BigVGANConfig::base_24khz(),
            BigVGANVariant::Large => BigVGANConfig::large_24khz(),
            BigVGANVariant::Fast => BigVGANConfig::fast_24khz(),
            BigVGANVariant::Ultra => BigVGANConfig::ultra_48khz(),
        }
    }
}

/// BigVGAN generator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BigVGANConfig {
    /// Number of mel bands in input spectrogram
    pub num_mels: usize,

    /// Number of upsampling layers
    pub num_upsamples: usize,

    /// Upsampling rates for each layer
    pub upsample_rates: Vec<usize>,

    /// Upsampling kernel sizes for each layer
    pub upsample_kernel_sizes: Vec<usize>,

    /// Initial number of channels
    pub upsample_initial_channel: usize,

    /// Residual block kernel sizes
    pub resblock_kernel_sizes: Vec<usize>,

    /// Residual block dilation sizes
    pub resblock_dilation_sizes: Vec<Vec<usize>>,

    /// Target sample rate
    pub sample_rate: u32,

    /// Hop length for mel spectrogram
    pub hop_length: usize,

    /// Window size for mel spectrogram
    pub win_length: usize,

    /// Activation function configuration
    pub activation_config: ActivationConfig,

    /// Whether to use anti-aliased multi-periodicity composition (AMPBlock)
    pub use_amp_block: bool,

    /// Number of periods for multi-periodicity composition
    pub amp_block_periods: Vec<usize>,

    /// Whether to use causal (streaming) mode
    pub causal: bool,
}

impl BigVGANConfig {
    /// Base 24kHz configuration (balanced quality and speed)
    pub fn base_24khz() -> Self {
        Self {
            num_mels: 80,
            num_upsamples: 4,
            upsample_rates: vec![8, 8, 2, 2],
            upsample_kernel_sizes: vec![16, 16, 4, 4],
            upsample_initial_channel: 512,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            sample_rate: 24000,
            hop_length: 256,
            win_length: 1024,
            activation_config: ActivationConfig {
                alpha: 1.0,
                filter_order: 12,
                cutoff_frequency: 0.47,
                learnable_alpha: true,
            },
            use_amp_block: true,
            amp_block_periods: vec![2, 3, 5, 7, 11],
            causal: false,
        }
    }

    /// Large 24kHz configuration (highest quality)
    pub fn large_24khz() -> Self {
        Self {
            num_mels: 100,
            num_upsamples: 4,
            upsample_rates: vec![8, 8, 2, 2],
            upsample_kernel_sizes: vec![16, 16, 4, 4],
            upsample_initial_channel: 1024,
            resblock_kernel_sizes: vec![3, 5, 7, 11, 13],
            resblock_dilation_sizes: vec![
                vec![1, 3, 5, 7],
                vec![1, 3, 5, 7],
                vec![1, 3, 5, 7],
                vec![1, 3, 5, 7],
                vec![1, 3, 5, 7],
            ],
            sample_rate: 24000,
            hop_length: 256,
            win_length: 1024,
            activation_config: ActivationConfig {
                alpha: 1.0,
                filter_order: 16,
                cutoff_frequency: 0.48,
                learnable_alpha: true,
            },
            use_amp_block: true,
            amp_block_periods: vec![2, 3, 5, 7, 11, 13],
            causal: false,
        }
    }

    /// Fast 24kHz configuration (optimized for speed)
    pub fn fast_24khz() -> Self {
        Self {
            num_mels: 80,
            num_upsamples: 4,
            upsample_rates: vec![8, 8, 2, 2],
            upsample_kernel_sizes: vec![16, 16, 4, 4],
            upsample_initial_channel: 256,
            resblock_kernel_sizes: vec![3, 7],
            resblock_dilation_sizes: vec![vec![1, 3], vec![1, 3]],
            sample_rate: 24000,
            hop_length: 256,
            win_length: 1024,
            activation_config: ActivationConfig {
                alpha: 1.0,
                filter_order: 8,
                cutoff_frequency: 0.45,
                learnable_alpha: false, // Faster without learnable alpha
            },
            use_amp_block: false, // Disable AMP for speed
            amp_block_periods: vec![],
            causal: false,
        }
    }

    /// Ultra 48kHz configuration (ultra-high quality)
    pub fn ultra_48khz() -> Self {
        Self {
            num_mels: 100,
            num_upsamples: 5,
            upsample_rates: vec![8, 8, 4, 2, 2],
            upsample_kernel_sizes: vec![16, 16, 8, 4, 4],
            upsample_initial_channel: 1024,
            resblock_kernel_sizes: vec![3, 5, 7, 11, 13, 17],
            resblock_dilation_sizes: vec![
                vec![1, 3, 5, 7],
                vec![1, 3, 5, 7],
                vec![1, 3, 5, 7],
                vec![1, 3, 5, 7],
                vec![1, 3, 5, 7],
                vec![1, 3, 5, 7],
            ],
            sample_rate: 48000,
            hop_length: 512,
            win_length: 2048,
            activation_config: ActivationConfig {
                alpha: 1.0,
                filter_order: 20,
                cutoff_frequency: 0.48,
                learnable_alpha: true,
            },
            use_amp_block: true,
            amp_block_periods: vec![2, 3, 5, 7, 11, 13, 17],
            causal: false,
        }
    }

    /// Streaming/causal configuration for real-time applications
    pub fn streaming_24khz() -> Self {
        let mut config = Self::base_24khz();
        config.causal = true;
        config.use_amp_block = false; // Disable AMP for lower latency
        config.upsample_initial_channel = 384; // Reduce capacity for speed
        config
    }

    /// Calculate total upsampling factor
    pub fn total_upsample_factor(&self) -> usize {
        self.upsample_rates.iter().product()
    }

    /// Calculate receptive field size
    pub fn receptive_field(&self) -> usize {
        let mut rf = 1;

        // Contribution from upsampling layers
        for &kernel_size in &self.upsample_kernel_sizes {
            rf += kernel_size - 1;
        }

        // Contribution from residual blocks
        for (kernel_size, dilations) in self
            .resblock_kernel_sizes
            .iter()
            .zip(self.resblock_dilation_sizes.iter())
        {
            for &dilation in dilations {
                rf += (kernel_size - 1) * dilation;
            }
        }

        rf
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.num_upsamples != self.upsample_rates.len() {
            return Err(format!(
                "num_upsamples ({}) must match upsample_rates length ({})",
                self.num_upsamples,
                self.upsample_rates.len()
            ));
        }

        if self.num_upsamples != self.upsample_kernel_sizes.len() {
            return Err(format!(
                "num_upsamples ({}) must match upsample_kernel_sizes length ({})",
                self.num_upsamples,
                self.upsample_kernel_sizes.len()
            ));
        }

        if self.resblock_kernel_sizes.len() != self.resblock_dilation_sizes.len() {
            return Err(format!(
                "resblock_kernel_sizes length ({}) must match resblock_dilation_sizes length ({})",
                self.resblock_kernel_sizes.len(),
                self.resblock_dilation_sizes.len()
            ));
        }

        if self.num_mels == 0 {
            return Err("num_mels must be greater than 0".to_string());
        }

        if self.sample_rate == 0 {
            return Err("sample_rate must be greater than 0".to_string());
        }

        if self.hop_length == 0 {
            return Err("hop_length must be greater than 0".to_string());
        }

        if self.use_amp_block && self.amp_block_periods.is_empty() {
            return Err(
                "amp_block_periods must not be empty when use_amp_block is true".to_string(),
            );
        }

        Ok(())
    }
}

impl Default for BigVGANConfig {
    fn default() -> Self {
        Self::base_24khz()
    }
}

/// Model metadata for BigVGAN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BigVGANMetadata {
    /// Model variant
    pub variant: BigVGANVariant,
    /// Training dataset information
    pub dataset: String,
    /// Model version
    pub version: String,
    /// Number of training steps
    pub training_steps: Option<usize>,
    /// Target MOS (Mean Opinion Score)
    pub target_mos: Option<f32>,
    /// Model size in parameters
    pub num_parameters: Option<usize>,
}

impl BigVGANMetadata {
    /// Create metadata for a variant
    pub fn for_variant(variant: BigVGANVariant) -> Self {
        match variant {
            BigVGANVariant::Base => Self {
                variant,
                dataset: "LibriTTS + VCTK + CommonVoice".to_string(),
                version: "1.0.0".to_string(),
                training_steps: Some(1_000_000),
                target_mos: Some(4.5),
                num_parameters: Some(14_000_000),
            },
            BigVGANVariant::Large => Self {
                variant,
                dataset: "LibriTTS + VCTK + CommonVoice + Multilingual".to_string(),
                version: "1.0.0".to_string(),
                training_steps: Some(2_000_000),
                target_mos: Some(4.7),
                num_parameters: Some(112_000_000),
            },
            BigVGANVariant::Fast => Self {
                variant,
                dataset: "LibriTTS".to_string(),
                version: "1.0.0".to_string(),
                training_steps: Some(500_000),
                target_mos: Some(4.3),
                num_parameters: Some(4_000_000),
            },
            BigVGANVariant::Ultra => Self {
                variant,
                dataset: "High-resolution audio corpus (48kHz)".to_string(),
                version: "1.0.0".to_string(),
                training_steps: Some(3_000_000),
                target_mos: Some(4.8),
                num_parameters: Some(150_000_000),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bigvgan_variant_model_names() {
        assert_eq!(BigVGANVariant::Base.model_name(), "bigvgan_base_24khz");
        assert_eq!(BigVGANVariant::Large.model_name(), "bigvgan_large_24khz");
        assert_eq!(BigVGANVariant::Fast.model_name(), "bigvgan_fast_24khz");
        assert_eq!(BigVGANVariant::Ultra.model_name(), "bigvgan_ultra_48khz");
    }

    #[test]
    fn test_bigvgan_config_base() {
        let config = BigVGANConfig::base_24khz();
        assert_eq!(config.num_mels, 80);
        assert_eq!(config.sample_rate, 24000);
        assert!(config.use_amp_block);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_bigvgan_config_large() {
        let config = BigVGANConfig::large_24khz();
        assert_eq!(config.num_mels, 100);
        assert_eq!(config.sample_rate, 24000);
        assert_eq!(config.upsample_initial_channel, 1024);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_bigvgan_config_fast() {
        let config = BigVGANConfig::fast_24khz();
        assert_eq!(config.upsample_initial_channel, 256);
        assert!(!config.use_amp_block);
        assert!(!config.activation_config.learnable_alpha);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_bigvgan_config_ultra() {
        let config = BigVGANConfig::ultra_48khz();
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.num_upsamples, 5);
        assert_eq!(config.hop_length, 512);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_bigvgan_config_streaming() {
        let config = BigVGANConfig::streaming_24khz();
        assert!(config.causal);
        assert!(!config.use_amp_block);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_total_upsample_factor() {
        let config = BigVGANConfig::base_24khz();
        // 8 * 8 * 2 * 2 = 256
        assert_eq!(config.total_upsample_factor(), 256);

        let config_ultra = BigVGANConfig::ultra_48khz();
        // 8 * 8 * 4 * 2 * 2 = 1024
        assert_eq!(config_ultra.total_upsample_factor(), 1024);
    }

    #[test]
    fn test_receptive_field_calculation() {
        let config = BigVGANConfig::base_24khz();
        let rf = config.receptive_field();
        assert!(rf > 0);
        // Receptive field should be reasonable (not too small or too large)
        assert!(rf < 10000);
    }

    #[test]
    fn test_config_validation_invalid_upsamples() {
        let mut config = BigVGANConfig::base_24khz();
        config.num_upsamples = 5; // Mismatch with upsample_rates length
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_resblock() {
        let mut config = BigVGANConfig::base_24khz();
        config.resblock_kernel_sizes.push(15); // Length mismatch
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_zero_mels() {
        let mut config = BigVGANConfig::base_24khz();
        config.num_mels = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_amp_block_no_periods() {
        let mut config = BigVGANConfig::base_24khz();
        config.use_amp_block = true;
        config.amp_block_periods.clear();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_metadata_for_variants() {
        for variant in [
            BigVGANVariant::Base,
            BigVGANVariant::Large,
            BigVGANVariant::Fast,
            BigVGANVariant::Ultra,
        ] {
            let metadata = BigVGANMetadata::for_variant(variant);
            assert_eq!(metadata.variant, variant);
            assert!(!metadata.dataset.is_empty());
            assert!(metadata.training_steps.is_some());
            assert!(metadata.target_mos.is_some());
        }
    }

    #[test]
    fn test_default_config() {
        let config = BigVGANConfig::default();
        assert_eq!(config.sample_rate, 24000); // Should be base_24khz
        assert!(config.validate().is_ok());
    }
}
