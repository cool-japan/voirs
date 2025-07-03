//! HiFi-GAN model variants with different architectures.

use crate::{Result, VocoderError};
use super::{HiFiGanConfig, HiFiGanVariant};
use serde::{Deserialize, Serialize};

/// HiFi-GAN model variants with predefined configurations
#[derive(Debug, Clone)]
pub struct HiFiGanVariants;

impl HiFiGanVariants {
    /// Get configuration for V1 variant (highest quality)
    pub fn v1() -> HiFiGanConfig {
        HiFiGanConfig {
            variant: HiFiGanVariant::V1,
            mel_channels: 80,
            sample_rate: 22050,
            num_residual_blocks: 3,
            upsample_rates: vec![8, 8, 2, 2],
            upsample_kernel_sizes: vec![16, 16, 4, 4],
            mrf_kernel_sizes: vec![3, 7, 11],
            mrf_dilation_sizes: vec![
                vec![1, 3, 5],
                vec![1, 3, 5],
                vec![1, 3, 5],
            ],
            initial_channels: 512,
            leaky_relu_slope: 0.1,
        }
    }

    /// Get configuration for V2 variant (balanced)
    pub fn v2() -> HiFiGanConfig {
        HiFiGanConfig {
            variant: HiFiGanVariant::V2,
            mel_channels: 80,
            sample_rate: 22050,
            num_residual_blocks: 3,
            upsample_rates: vec![8, 8, 4, 2],
            upsample_kernel_sizes: vec![16, 16, 8, 4],
            mrf_kernel_sizes: vec![3, 7, 11],
            mrf_dilation_sizes: vec![
                vec![1, 3, 5],
                vec![1, 3, 5],
                vec![1, 3, 5],
            ],
            initial_channels: 256,
            leaky_relu_slope: 0.1,
        }
    }

    /// Get configuration for V3 variant (fastest)
    pub fn v3() -> HiFiGanConfig {
        HiFiGanConfig {
            variant: HiFiGanVariant::V3,
            mel_channels: 80,
            sample_rate: 22050,
            num_residual_blocks: 2,
            upsample_rates: vec![8, 8, 8, 2],
            upsample_kernel_sizes: vec![16, 16, 16, 4],
            mrf_kernel_sizes: vec![3, 5, 7],
            mrf_dilation_sizes: vec![
                vec![1, 2, 4],
                vec![1, 2, 4],
                vec![1, 2, 4],
            ],
            initial_channels: 128,
            leaky_relu_slope: 0.1,
        }
    }

    /// Get configuration for specific variant
    pub fn get_variant(variant: HiFiGanVariant) -> HiFiGanConfig {
        match variant {
            HiFiGanVariant::V1 => Self::v1(),
            HiFiGanVariant::V2 => Self::v2(),
            HiFiGanVariant::V3 => Self::v3(),
        }
    }

    /// Get all available variants
    pub fn all_variants() -> Vec<HiFiGanVariant> {
        vec![
            HiFiGanVariant::V1,
            HiFiGanVariant::V2,
            HiFiGanVariant::V3,
        ]
    }

    /// Get variant by name
    pub fn from_name(name: &str) -> Result<HiFiGanVariant> {
        match name.to_lowercase().as_str() {
            "v1" | "hifigan-v1" | "hifigan_v1" => Ok(HiFiGanVariant::V1),
            "v2" | "hifigan-v2" | "hifigan_v2" => Ok(HiFiGanVariant::V2),
            "v3" | "hifigan-v3" | "hifigan_v3" => Ok(HiFiGanVariant::V3),
            _ => Err(VocoderError::ConfigError(format!(
                "Unknown HiFi-GAN variant: {}",
                name
            ))),
        }
    }

    /// Create custom variant with modified parameters
    pub fn custom(
        base_variant: HiFiGanVariant,
        modifications: VariantModifications,
    ) -> HiFiGanConfig {
        let mut config = Self::get_variant(base_variant);
        
        if let Some(sample_rate) = modifications.sample_rate {
            config.sample_rate = sample_rate;
        }
        
        if let Some(mel_channels) = modifications.mel_channels {
            config.mel_channels = mel_channels;
        }
        
        if let Some(initial_channels) = modifications.initial_channels {
            config.initial_channels = initial_channels;
        }
        
        if let Some(num_residual_blocks) = modifications.num_residual_blocks {
            config.num_residual_blocks = num_residual_blocks;
        }
        
        if let Some(upsample_rates) = modifications.upsample_rates {
            config.upsample_rates = upsample_rates;
        }
        
        if let Some(upsample_kernel_sizes) = modifications.upsample_kernel_sizes {
            config.upsample_kernel_sizes = upsample_kernel_sizes;
        }
        
        if let Some(mrf_kernel_sizes) = modifications.mrf_kernel_sizes {
            config.mrf_kernel_sizes = mrf_kernel_sizes;
        }
        
        if let Some(mrf_dilation_sizes) = modifications.mrf_dilation_sizes {
            config.mrf_dilation_sizes = mrf_dilation_sizes;
        }
        
        if let Some(leaky_relu_slope) = modifications.leaky_relu_slope {
            config.leaky_relu_slope = leaky_relu_slope;
        }
        
        config
    }
}

/// Modifications for creating custom variants
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VariantModifications {
    /// Sample rate override
    pub sample_rate: Option<u32>,
    /// Mel channels override
    pub mel_channels: Option<u32>,
    /// Initial channels override
    pub initial_channels: Option<u32>,
    /// Number of residual blocks override
    pub num_residual_blocks: Option<u32>,
    /// Upsample rates override
    pub upsample_rates: Option<Vec<u32>>,
    /// Upsample kernel sizes override
    pub upsample_kernel_sizes: Option<Vec<u32>>,
    /// MRF kernel sizes override
    pub mrf_kernel_sizes: Option<Vec<u32>>,
    /// MRF dilation sizes override
    pub mrf_dilation_sizes: Option<Vec<Vec<u32>>>,
    /// Leaky ReLU slope override
    pub leaky_relu_slope: Option<f32>,
}

impl VariantModifications {
    /// Create new modifications builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set sample rate
    pub fn sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = Some(sample_rate);
        self
    }

    /// Set mel channels
    pub fn mel_channels(mut self, mel_channels: u32) -> Self {
        self.mel_channels = Some(mel_channels);
        self
    }

    /// Set initial channels
    pub fn initial_channels(mut self, initial_channels: u32) -> Self {
        self.initial_channels = Some(initial_channels);
        self
    }

    /// Set number of residual blocks
    pub fn num_residual_blocks(mut self, num_residual_blocks: u32) -> Self {
        self.num_residual_blocks = Some(num_residual_blocks);
        self
    }

    /// Set upsample rates
    pub fn upsample_rates(mut self, upsample_rates: Vec<u32>) -> Self {
        self.upsample_rates = Some(upsample_rates);
        self
    }

    /// Set upsample kernel sizes
    pub fn upsample_kernel_sizes(mut self, upsample_kernel_sizes: Vec<u32>) -> Self {
        self.upsample_kernel_sizes = Some(upsample_kernel_sizes);
        self
    }

    /// Set MRF kernel sizes
    pub fn mrf_kernel_sizes(mut self, mrf_kernel_sizes: Vec<u32>) -> Self {
        self.mrf_kernel_sizes = Some(mrf_kernel_sizes);
        self
    }

    /// Set MRF dilation sizes
    pub fn mrf_dilation_sizes(mut self, mrf_dilation_sizes: Vec<Vec<u32>>) -> Self {
        self.mrf_dilation_sizes = Some(mrf_dilation_sizes);
        self
    }

    /// Set leaky ReLU slope
    pub fn leaky_relu_slope(mut self, leaky_relu_slope: f32) -> Self {
        self.leaky_relu_slope = Some(leaky_relu_slope);
        self
    }
}

/// Variant comparison utilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantComparison {
    /// Variant name
    pub name: String,
    /// Estimated parameters
    pub parameters: u64,
    /// Estimated model size (MB)
    pub model_size_mb: f32,
    /// Upsampling factor
    pub upsampling_factor: u32,
    /// Receptive field size
    pub receptive_field_size: u32,
    /// Quality score (estimated)
    pub quality_score: f32,
    /// Speed score (estimated)
    pub speed_score: f32,
}

impl HiFiGanVariants {
    /// Compare all variants
    pub fn compare_variants() -> Vec<VariantComparison> {
        vec![
            Self::analyze_variant(HiFiGanVariant::V1),
            Self::analyze_variant(HiFiGanVariant::V2),
            Self::analyze_variant(HiFiGanVariant::V3),
        ]
    }

    /// Analyze a specific variant
    pub fn analyze_variant(variant: HiFiGanVariant) -> VariantComparison {
        let config = Self::get_variant(variant);
        
        let parameters = Self::estimate_parameters(&config);
        let model_size_mb = (parameters * 4) as f32 / 1024.0 / 1024.0; // 4 bytes per float32
        let upsampling_factor = config.upsample_rates.iter().product();
        let receptive_field_size = Self::estimate_receptive_field(&config);
        
        // Estimated scores based on variant characteristics
        let (quality_score, speed_score) = match variant {
            HiFiGanVariant::V1 => (4.5, 2.0), // High quality, slower
            HiFiGanVariant::V2 => (4.0, 3.5), // Balanced
            HiFiGanVariant::V3 => (3.5, 4.5), // Lower quality, faster
        };

        VariantComparison {
            name: variant.name().to_string(),
            parameters,
            model_size_mb,
            upsampling_factor,
            receptive_field_size,
            quality_score,
            speed_score,
        }
    }

    /// Estimate parameters for a configuration
    fn estimate_parameters(config: &HiFiGanConfig) -> u64 {
        let mut total = 0u64;
        
        // Input convolution
        total += (config.mel_channels * config.initial_channels * 7) as u64;
        
        // Upsampling blocks
        let mut current_channels = config.initial_channels;
        for (&upsample_rate, &kernel_size) in config.upsample_rates.iter()
            .zip(config.upsample_kernel_sizes.iter())
        {
            let output_channels = current_channels / 2;
            
            // Transposed convolution
            total += (current_channels * output_channels * kernel_size) as u64;
            
            // MRF blocks
            for &mrf_kernel in &config.mrf_kernel_sizes {
                total += (output_channels * output_channels * mrf_kernel * 3) as u64;
            }
            
            current_channels = output_channels;
        }
        
        // Output convolution
        total += (current_channels * 1 * 7) as u64;
        
        total
    }

    /// Estimate receptive field size
    fn estimate_receptive_field(config: &HiFiGanConfig) -> u32 {
        let kernel_contribution = config.mrf_kernel_sizes.iter()
            .zip(config.mrf_dilation_sizes.iter())
            .map(|(&kernel, dilations)| {
                let max_dilation = *dilations.iter().max().unwrap_or(&1);
                (kernel - 1) * max_dilation + 1
            })
            .sum::<u32>();
        
        let upsample_contribution = config.upsample_kernel_sizes.iter().sum::<u32>();
        
        kernel_contribution + upsample_contribution
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variant_configs() {
        let v1 = HiFiGanVariants::v1();
        let v2 = HiFiGanVariants::v2();
        let v3 = HiFiGanVariants::v3();

        // Test V1 (highest quality)
        assert_eq!(v1.initial_channels, 512);
        assert_eq!(v1.num_residual_blocks, 3);
        assert_eq!(v1.upsample_rates, vec![8, 8, 2, 2]);

        // Test V2 (balanced)
        assert_eq!(v2.initial_channels, 256);
        assert_eq!(v2.num_residual_blocks, 3);
        assert_eq!(v2.upsample_rates, vec![8, 8, 4, 2]);

        // Test V3 (fastest)
        assert_eq!(v3.initial_channels, 128);
        assert_eq!(v3.num_residual_blocks, 2);
        assert_eq!(v3.upsample_rates, vec![8, 8, 8, 2]);
    }

    #[test]
    fn test_variant_from_name() {
        assert_eq!(HiFiGanVariants::from_name("v1").unwrap(), HiFiGanVariant::V1);
        assert_eq!(HiFiGanVariants::from_name("V2").unwrap(), HiFiGanVariant::V2);
        assert_eq!(HiFiGanVariants::from_name("hifigan-v3").unwrap(), HiFiGanVariant::V3);
        
        assert!(HiFiGanVariants::from_name("invalid").is_err());
    }

    #[test]
    fn test_all_variants() {
        let variants = HiFiGanVariants::all_variants();
        assert_eq!(variants.len(), 3);
        assert!(variants.contains(&HiFiGanVariant::V1));
        assert!(variants.contains(&HiFiGanVariant::V2));
        assert!(variants.contains(&HiFiGanVariant::V3));
    }

    #[test]
    fn test_custom_variant() {
        let modifications = VariantModifications::new()
            .sample_rate(44100)
            .mel_channels(128)
            .initial_channels(1024);

        let custom = HiFiGanVariants::custom(HiFiGanVariant::V1, modifications);
        
        assert_eq!(custom.sample_rate, 44100);
        assert_eq!(custom.mel_channels, 128);
        assert_eq!(custom.initial_channels, 1024);
        // Other parameters should remain from V1
        assert_eq!(custom.num_residual_blocks, 3);
        assert_eq!(custom.upsample_rates, vec![8, 8, 2, 2]);
    }

    #[test]
    fn test_variant_analysis() {
        let v1_analysis = HiFiGanVariants::analyze_variant(HiFiGanVariant::V1);
        let v2_analysis = HiFiGanVariants::analyze_variant(HiFiGanVariant::V2);
        let v3_analysis = HiFiGanVariants::analyze_variant(HiFiGanVariant::V3);

        // V1 should have more parameters than V2, V2 more than V3
        assert!(v1_analysis.parameters > v2_analysis.parameters);
        assert!(v2_analysis.parameters > v3_analysis.parameters);

        // V1 should have higher quality score but lower speed score
        assert!(v1_analysis.quality_score > v3_analysis.quality_score);
        assert!(v1_analysis.speed_score < v3_analysis.speed_score);

        // All should have correct upsampling factor for their respective configs
        assert_eq!(v1_analysis.upsampling_factor, 256); // 8*8*2*2
        assert_eq!(v2_analysis.upsampling_factor, 512); // 8*8*4*2
        assert_eq!(v3_analysis.upsampling_factor, 1024); // 8*8*8*2
    }

    #[test]
    fn test_variant_comparison() {
        let comparisons = HiFiGanVariants::compare_variants();
        
        assert_eq!(comparisons.len(), 3);
        assert!(comparisons.iter().any(|c| c.name == "HiFi-GAN V1"));
        assert!(comparisons.iter().any(|c| c.name == "HiFi-GAN V2"));
        assert!(comparisons.iter().any(|c| c.name == "HiFi-GAN V3"));
        
        // Check that all have reasonable parameter counts
        for comparison in &comparisons {
            assert!(comparison.parameters > 0);
            assert!(comparison.model_size_mb > 0.0);
            assert!(comparison.upsampling_factor > 0);
            assert!(comparison.receptive_field_size > 0);
            assert!(comparison.quality_score > 0.0);
            assert!(comparison.speed_score > 0.0);
        }
    }

    #[test]
    fn test_variant_modifications_builder() {
        let modifications = VariantModifications::new()
            .sample_rate(44100)
            .mel_channels(128)
            .initial_channels(1024)
            .num_residual_blocks(4)
            .leaky_relu_slope(0.2);

        assert_eq!(modifications.sample_rate, Some(44100));
        assert_eq!(modifications.mel_channels, Some(128));
        assert_eq!(modifications.initial_channels, Some(1024));
        assert_eq!(modifications.num_residual_blocks, Some(4));
        assert_eq!(modifications.leaky_relu_slope, Some(0.2));
    }
}