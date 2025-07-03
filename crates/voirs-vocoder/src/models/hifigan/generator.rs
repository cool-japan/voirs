//! HiFi-GAN generator implementation.

use crate::{Result, VocoderError};
use super::{HiFiGanConfig, mrf::MultiReceptiveField};
use serde::{Deserialize, Serialize};

#[cfg(feature = "candle")]
use candle_core::{Device, Tensor};
#[cfg(feature = "candle")]
use candle_nn::{Conv1d, ConvTranspose1d, Module, VarBuilder};

/// HiFi-GAN Generator
#[derive(Debug)]
pub struct HiFiGanGenerator {
    /// Configuration
    config: HiFiGanConfig,
    /// Input convolution layer
    #[cfg(feature = "candle")]
    input_conv: Conv1d,
    /// Upsampling blocks
    #[cfg(feature = "candle")]
    upsample_blocks: Vec<UpsampleBlock>,
    /// Output convolution layer
    #[cfg(feature = "candle")]
    output_conv: Conv1d,
    /// Device
    #[cfg(feature = "candle")]
    device: Device,
}

/// Upsampling block with transposed convolution and MRF
#[derive(Debug)]
struct UpsampleBlock {
    /// Transposed convolution
    #[cfg(feature = "candle")]
    transpose_conv: ConvTranspose1d,
    /// Multi-receptive field block
    #[cfg(feature = "candle")]
    mrf: MultiReceptiveField,
}

impl HiFiGanGenerator {
    /// Create new HiFi-GAN generator
    #[cfg(feature = "candle")]
    pub fn new(config: HiFiGanConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        
        // Input convolution: mel_channels -> initial_channels
        let input_conv = candle_nn::conv1d(
            config.mel_channels as usize,
            config.initial_channels as usize,
            7, // kernel_size
            candle_nn::Conv1dConfig {
                padding: 3, // (7-1)/2 = 3 to maintain input size
                ..Default::default()
            },
            vb.pp("input_conv"),
        )?;

        // Build upsampling blocks
        let mut upsample_blocks = Vec::new();
        let mut current_channels = config.initial_channels;
        
        for (i, (&upsample_rate, &kernel_size)) in config.upsample_rates.iter()
            .zip(config.upsample_kernel_sizes.iter())
            .enumerate()
        {
            let output_channels = current_channels / 2;
            
            // Transposed convolution for upsampling
            let transpose_conv = candle_nn::conv_transpose1d(
                current_channels as usize,
                output_channels as usize,
                kernel_size as usize,
                candle_nn::ConvTranspose1dConfig {
                    stride: upsample_rate as usize,
                    padding: (kernel_size - upsample_rate) as usize / 2,
                    dilation: 1,
                    groups: 1,
                    output_padding: 0,
                },
                vb.pp(format!("upsample_blocks.{}.transpose_conv", i)),
            )?;
            
            // Multi-receptive field block
            let mrf = MultiReceptiveField::new(
                output_channels,
                &config.mrf_kernel_sizes,
                &config.mrf_dilation_sizes,
                config.leaky_relu_slope,
                vb.pp(format!("upsample_blocks.{}.mrf", i)),
            )?;
            
            upsample_blocks.push(UpsampleBlock {
                transpose_conv,
                mrf,
            });
            
            current_channels = output_channels;
        }
        
        // Output convolution: final_channels -> 1 (mono audio)
        let output_conv = candle_nn::conv1d(
            current_channels as usize,
            1,
            7, // kernel_size
            candle_nn::Conv1dConfig {
                padding: 3, // (7-1)/2 = 3 to maintain input size
                ..Default::default()
            },
            vb.pp("output_conv"),
        )?;

        Ok(Self {
            config,
            input_conv,
            upsample_blocks,
            output_conv,
            device,
        })
    }

    /// Create generator without Candle (for testing)
    #[cfg(not(feature = "candle"))]
    pub fn new(config: HiFiGanConfig) -> Result<Self> {
        Ok(Self {
            config,
        })
    }

    /// Forward pass through generator
    #[cfg(feature = "candle")]
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // Input convolution
        let mut x = self.input_conv.forward(mel)?;
        
        // Apply upsampling blocks
        for block in &self.upsample_blocks {
            // Transposed convolution
            x = block.transpose_conv.forward(&x)?;
            
            // Leaky ReLU activation: max(x, slope * x)
            let negative_part = x.affine(self.config.leaky_relu_slope as f64, 0.0)?;
            x = x.maximum(&negative_part)?;
            
            // Multi-receptive field
            x = block.mrf.forward(&x)?;
        }
        
        // Output convolution
        x = self.output_conv.forward(&x)?;
        
        // Tanh activation for audio output
        x = x.tanh()?;
        
        Ok(x)
    }

    /// Forward pass placeholder for non-Candle builds
    #[cfg(not(feature = "candle"))]
    pub fn forward(&self, _mel: &[Vec<f32>]) -> Result<Vec<f32>> {
        Err(VocoderError::ModelError(
            "Candle feature not enabled for HiFi-GAN generator".to_string(),
        ))
    }

    /// Get configuration
    pub fn config(&self) -> &HiFiGanConfig {
        &self.config
    }

    /// Get device
    #[cfg(feature = "candle")]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Calculate total upsampling factor
    pub fn total_upsampling_factor(&self) -> u32 {
        self.config.upsample_rates.iter().product()
    }

    /// Calculate receptive field size
    pub fn receptive_field_size(&self) -> u32 {
        // Simplified calculation - actual receptive field depends on kernel sizes and dilations
        let kernel_contribution = self.config.mrf_kernel_sizes.iter()
            .zip(self.config.mrf_dilation_sizes.iter())
            .map(|(&kernel, dilations)| {
                let max_dilation = *dilations.iter().max().unwrap_or(&1);
                (kernel - 1) * max_dilation + 1
            })
            .sum::<u32>();
        
        // Add upsampling contributions
        let upsample_contribution = self.config.upsample_kernel_sizes.iter().sum::<u32>();
        
        kernel_contribution + upsample_contribution
    }
}

/// Generator statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorStats {
    /// Number of parameters
    pub num_parameters: u64,
    /// Model size in bytes
    pub model_size_bytes: u64,
    /// Total upsampling factor
    pub upsampling_factor: u32,
    /// Receptive field size
    pub receptive_field_size: u32,
}

impl HiFiGanGenerator {
    /// Get generator statistics
    pub fn stats(&self) -> GeneratorStats {
        GeneratorStats {
            num_parameters: self.estimate_parameters(),
            model_size_bytes: self.estimate_model_size(),
            upsampling_factor: self.total_upsampling_factor(),
            receptive_field_size: self.receptive_field_size(),
        }
    }

    /// Estimate number of parameters
    fn estimate_parameters(&self) -> u64 {
        let mut total = 0u64;
        
        // Input convolution
        total += (self.config.mel_channels * self.config.initial_channels * 7) as u64;
        
        // Upsampling blocks
        let mut current_channels = self.config.initial_channels;
        for (&upsample_rate, &kernel_size) in self.config.upsample_rates.iter()
            .zip(self.config.upsample_kernel_sizes.iter())
        {
            let output_channels = current_channels / 2;
            
            // Transposed convolution parameters
            total += (current_channels * output_channels * kernel_size) as u64;
            
            // MRF parameters (simplified estimate)
            for &mrf_kernel in &self.config.mrf_kernel_sizes {
                total += (output_channels * output_channels * mrf_kernel * 3) as u64; // 3 residual blocks
            }
            
            current_channels = output_channels;
        }
        
        // Output convolution
        total += (current_channels * 1 * 7) as u64;
        
        total
    }

    /// Estimate model size in bytes
    fn estimate_model_size(&self) -> u64 {
        // Assume 32-bit floats
        self.estimate_parameters() * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_config() {
        let config = HiFiGanConfig::default();
        
        #[cfg(not(feature = "candle"))]
        {
            let generator = HiFiGanGenerator::new(config.clone());
            assert!(generator.is_ok());
            let gen = generator.unwrap();
            assert_eq!(gen.config().mel_channels, config.mel_channels);
            assert_eq!(gen.total_upsampling_factor(), 256); // 8*8*2*2
        }
    }

    #[test]
    fn test_generator_stats() {
        let config = HiFiGanConfig::default();
        
        #[cfg(not(feature = "candle"))]
        {
            let generator = HiFiGanGenerator::new(config).unwrap();
            
            let stats = generator.stats();
            assert!(stats.num_parameters > 0);
            assert!(stats.model_size_bytes > 0);
            assert_eq!(stats.upsampling_factor, 256);
            assert!(stats.receptive_field_size > 0);
        }
    }

    #[test]
    fn test_upsampling_factor() {
        let config = HiFiGanConfig::default();
        
        #[cfg(not(feature = "candle"))]
        {
            let generator = HiFiGanGenerator::new(config).unwrap();
            
            // Default V1 config: [8, 8, 2, 2]
            assert_eq!(generator.total_upsampling_factor(), 256);
        }
    }

    #[test]
    fn test_receptive_field() {
        let config = HiFiGanConfig::default();
        
        #[cfg(not(feature = "candle"))]
        {
            let generator = HiFiGanGenerator::new(config).unwrap();
            
            let rf_size = generator.receptive_field_size();
            assert!(rf_size > 0);
            assert!(rf_size < 10000); // Reasonable upper bound
        }
    }
}