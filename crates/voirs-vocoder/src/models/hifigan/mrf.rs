//! Multi-Receptive Field (MRF) implementation for HiFi-GAN.

use crate::{Result, VocoderError};
use serde::{Deserialize, Serialize};

#[cfg(feature = "candle")]
use candle_core::{Device, Tensor};
#[cfg(feature = "candle")]
use candle_nn::{Conv1d, Module, VarBuilder};

/// Multi-Receptive Field block
#[derive(Debug)]
pub struct MultiReceptiveField {
    /// Residual blocks with different kernel sizes
    #[cfg(feature = "candle")]
    residual_blocks: Vec<ResidualBlock>,
    /// Number of channels
    channels: u32,
    /// Kernel sizes
    kernel_sizes: Vec<u32>,
    /// Dilation sizes for each kernel
    dilation_sizes: Vec<Vec<u32>>,
    /// Leaky ReLU slope
    leaky_relu_slope: f32,
}

/// Residual block with dilated convolutions
#[derive(Debug)]
struct ResidualBlock {
    /// Dilated convolution layers
    #[cfg(feature = "candle")]
    conv_layers: Vec<Conv1d>,
    /// Kernel size
    kernel_size: u32,
    /// Dilation sizes
    dilations: Vec<u32>,
    /// Leaky ReLU slope
    leaky_relu_slope: f32,
}

impl MultiReceptiveField {
    /// Create new Multi-Receptive Field block
    #[cfg(feature = "candle")]
    pub fn new(
        channels: u32,
        kernel_sizes: &[u32],
        dilation_sizes: &[Vec<u32>],
        leaky_relu_slope: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        if kernel_sizes.len() != dilation_sizes.len() {
            return Err(VocoderError::ConfigError(
                "Kernel sizes and dilation sizes must have the same length".to_string(),
            ));
        }

        let mut residual_blocks = Vec::new();
        
        for (i, (&kernel_size, dilations)) in kernel_sizes.iter().zip(dilation_sizes.iter()).enumerate() {
            let block = ResidualBlock::new(
                channels,
                kernel_size,
                dilations,
                leaky_relu_slope,
                vb.pp(format!("residual_blocks.{}", i)),
            )?;
            residual_blocks.push(block);
        }

        Ok(Self {
            residual_blocks,
            channels,
            kernel_sizes: kernel_sizes.to_vec(),
            dilation_sizes: dilation_sizes.to_vec(),
            leaky_relu_slope,
        })
    }

    /// Create MRF without Candle (for testing)
    #[cfg(not(feature = "candle"))]
    pub fn new(
        channels: u32,
        kernel_sizes: &[u32],
        dilation_sizes: &[Vec<u32>],
        leaky_relu_slope: f32,
    ) -> Result<Self> {
        if kernel_sizes.len() != dilation_sizes.len() {
            return Err(VocoderError::ConfigError(
                "Kernel sizes and dilation sizes must have the same length".to_string(),
            ));
        }

        Ok(Self {
            channels,
            kernel_sizes: kernel_sizes.to_vec(),
            dilation_sizes: dilation_sizes.to_vec(),
            leaky_relu_slope,
        })
    }

    /// Forward pass through MRF
    #[cfg(feature = "candle")]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut outputs = Vec::new();
        
        // Process input through each residual block
        for block in &self.residual_blocks {
            let output = block.forward(x)?;
            outputs.push(output);
        }
        
        // Sum all outputs (parallel processing)
        let mut result = outputs[0].clone();
        for output in outputs.iter().skip(1) {
            result = (&result + output)?;
        }
        
        // Average the outputs
        let scale = 1.0 / outputs.len() as f32;
        result = result.affine(scale as f64, 0.0)?;
        
        Ok(result)
    }

    /// Forward pass placeholder for non-Candle builds
    #[cfg(not(feature = "candle"))]
    pub fn forward(&self, _x: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        Err(VocoderError::ModelError(
            "Candle feature not enabled for MRF".to_string(),
        ))
    }

    /// Get configuration
    pub fn config(&self) -> MRFConfig {
        MRFConfig {
            channels: self.channels,
            kernel_sizes: self.kernel_sizes.clone(),
            dilation_sizes: self.dilation_sizes.clone(),
            leaky_relu_slope: self.leaky_relu_slope,
        }
    }

    /// Calculate receptive field size
    pub fn receptive_field_size(&self) -> u32 {
        self.kernel_sizes.iter()
            .zip(self.dilation_sizes.iter())
            .map(|(&kernel, dilations)| {
                let max_dilation = *dilations.iter().max().unwrap_or(&1);
                (kernel - 1) * max_dilation + 1
            })
            .max()
            .unwrap_or(1)
    }
}

impl ResidualBlock {
    /// Create new residual block
    #[cfg(feature = "candle")]
    fn new(
        channels: u32,
        kernel_size: u32,
        dilations: &[u32],
        leaky_relu_slope: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut conv_layers = Vec::new();
        
        for (i, &dilation) in dilations.iter().enumerate() {
            // Calculate padding to maintain input size: (kernel_size - 1) * dilation / 2
            let padding = (kernel_size - 1) * dilation / 2;
            
            let conv = candle_nn::conv1d(
                channels as usize,
                channels as usize,
                kernel_size as usize,
                candle_nn::Conv1dConfig {
                    padding: padding as usize,
                    dilation: dilation as usize,
                    ..Default::default()
                },
                vb.pp(format!("conv_layers.{}", i)),
            )?;
            conv_layers.push(conv);
        }

        Ok(Self {
            conv_layers,
            kernel_size,
            dilations: dilations.to_vec(),
            leaky_relu_slope,
        })
    }

    /// Forward pass through residual block
    #[cfg(feature = "candle")]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut residual = x.clone();
        
        for (i, conv) in self.conv_layers.iter().enumerate() {
            // Apply convolution
            residual = conv.forward(&residual)?;
            
            // Apply Leaky ReLU activation (except for last layer)
            if i < self.conv_layers.len() - 1 {
                let negative_part = residual.affine(self.leaky_relu_slope as f64, 0.0)?;
                residual = residual.maximum(&negative_part)?;
            }
        }
        
        // Add residual connection
        let output = (x + &residual)?;
        
        // Apply final Leaky ReLU
        let negative_part = output.affine(self.leaky_relu_slope as f64, 0.0)?;
        let output = output.maximum(&negative_part)?;
        
        Ok(output)
    }
}

/// MRF configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MRFConfig {
    /// Number of channels
    pub channels: u32,
    /// Kernel sizes for each residual block
    pub kernel_sizes: Vec<u32>,
    /// Dilation sizes for each kernel
    pub dilation_sizes: Vec<Vec<u32>>,
    /// Leaky ReLU slope
    pub leaky_relu_slope: f32,
}

impl Default for MRFConfig {
    fn default() -> Self {
        Self {
            channels: 512,
            kernel_sizes: vec![3, 7, 11],
            dilation_sizes: vec![
                vec![1, 3, 5],
                vec![1, 3, 5],
                vec![1, 3, 5],
            ],
            leaky_relu_slope: 0.1,
        }
    }
}

/// MRF statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MRFStats {
    /// Number of residual blocks
    pub num_blocks: usize,
    /// Total parameters
    pub num_parameters: u64,
    /// Receptive field size
    pub receptive_field_size: u32,
    /// Average kernel size
    pub avg_kernel_size: f32,
    /// Average dilation
    pub avg_dilation: f32,
}

impl MultiReceptiveField {
    /// Get MRF statistics
    pub fn stats(&self) -> MRFStats {
        let num_blocks = self.residual_blocks.len();
        let num_parameters = self.estimate_parameters();
        let receptive_field_size = self.receptive_field_size();
        
        let avg_kernel_size = self.kernel_sizes.iter().sum::<u32>() as f32 / num_blocks as f32;
        
        let total_dilations: u32 = self.dilation_sizes.iter()
            .flat_map(|dilations| dilations.iter())
            .sum();
        let num_dilations = self.dilation_sizes.iter()
            .map(|dilations| dilations.len())
            .sum::<usize>();
        let avg_dilation = total_dilations as f32 / num_dilations as f32;

        MRFStats {
            num_blocks,
            num_parameters,
            receptive_field_size,
            avg_kernel_size,
            avg_dilation,
        }
    }

    /// Estimate number of parameters
    fn estimate_parameters(&self) -> u64 {
        let mut total = 0u64;
        
        for (&kernel_size, dilations) in self.kernel_sizes.iter().zip(self.dilation_sizes.iter()) {
            // Each residual block has multiple conv layers
            let layers_per_block = dilations.len();
            let params_per_layer = (self.channels * self.channels * kernel_size) as u64;
            total += params_per_layer * layers_per_block as u64;
        }
        
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mrf_config() {
        let config = MRFConfig::default();
        
        assert_eq!(config.channels, 512);
        assert_eq!(config.kernel_sizes.len(), 3);
        assert_eq!(config.dilation_sizes.len(), 3);
        assert_eq!(config.leaky_relu_slope, 0.1);
    }

    #[test]
    fn test_mrf_creation() {
        let kernel_sizes = vec![3, 7, 11];
        let dilation_sizes = vec![
            vec![1, 3, 5],
            vec![1, 3, 5],
            vec![1, 3, 5],
        ];
        
        #[cfg(not(feature = "candle"))]
        {
            let mrf = MultiReceptiveField::new(
                512,
                &kernel_sizes,
                &dilation_sizes,
                0.1,
            );
            
            assert!(mrf.is_ok());
            let mrf = mrf.unwrap();
            
            let config = mrf.config();
            assert_eq!(config.channels, 512);
            assert_eq!(config.kernel_sizes, kernel_sizes);
            assert_eq!(config.dilation_sizes, dilation_sizes);
        }
    }

    #[test]
    fn test_mrf_mismatched_sizes() {
        let kernel_sizes = vec![3, 7];
        let dilation_sizes = vec![
            vec![1, 3, 5],
            vec![1, 3, 5],
            vec![1, 3, 5],
        ];
        
        #[cfg(not(feature = "candle"))]
        {
            let mrf = MultiReceptiveField::new(
                512,
                &kernel_sizes,
                &dilation_sizes,
                0.1,
            );
            
            assert!(mrf.is_err());
        }
    }

    #[test]
    fn test_receptive_field_calculation() {
        let kernel_sizes = vec![3, 7, 11];
        let dilation_sizes = vec![
            vec![1, 3, 5],
            vec![1, 3, 5],
            vec![1, 3, 5],
        ];
        
        #[cfg(not(feature = "candle"))]
        {
            let mrf = MultiReceptiveField::new(
                512,
                &kernel_sizes,
                &dilation_sizes,
                0.1,
            ).unwrap();
            
            let rf_size = mrf.receptive_field_size();
            
            // Should be max of (kernel-1)*max_dilation + 1 for each kernel
            // For kernel=11, max_dilation=5: (11-1)*5 + 1 = 51
            assert_eq!(rf_size, 51);
        }
    }

    #[test]
    fn test_mrf_stats() {
        let kernel_sizes = vec![3, 7, 11];
        let dilation_sizes = vec![
            vec![1, 3, 5],
            vec![1, 3, 5],
            vec![1, 3, 5],
        ];
        
        #[cfg(not(feature = "candle"))]
        {
            let mrf = MultiReceptiveField::new(
                512,
                &kernel_sizes,
                &dilation_sizes,
                0.1,
            ).unwrap();
            
            let stats = mrf.stats();
            
            assert_eq!(stats.num_blocks, 3);
            assert!(stats.num_parameters > 0);
            assert_eq!(stats.receptive_field_size, 51);
            assert_eq!(stats.avg_kernel_size, 7.0); // (3+7+11)/3
            assert_eq!(stats.avg_dilation, 3.0); // (1+3+5)*3/9
        }
    }
}