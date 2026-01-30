//! Location-Variable Convolution (LVC) module.
//!
//! LVC is a key component of UnivNet that adapts convolutional kernels
//! based on input location, allowing the model to capture location-dependent
//! patterns in the signal.

use crate::Result;
use serde::{Deserialize, Serialize};

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "candle")]
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder};

/// Configuration for Location-Variable Convolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LVCConfig {
    /// Kernel size for the main convolution
    pub kernel_size: usize,
    /// Dilation rate
    pub dilation: usize,
    /// Location kernel size (for computing location-dependent weights)
    pub location_kernel_size: usize,
    /// Whether to use causal convolution
    pub causal: bool,
}

impl Default for LVCConfig {
    fn default() -> Self {
        Self {
            kernel_size: 3,
            dilation: 1,
            location_kernel_size: 7,
            causal: false,
        }
    }
}

/// Location-Variable Convolution Block
///
/// Implements adaptive convolution where kernel weights vary with input location.
/// This allows the model to capture location-dependent patterns in audio signals.
#[cfg(feature = "candle")]
pub struct LVCBlock {
    /// Main convolution layer
    main_conv: Conv1d,

    /// Location embedding network (produces location-dependent features)
    location_conv: Conv1d,

    /// Configuration
    config: LVCConfig,

    /// Number of channels
    channels: usize,
}

#[cfg(feature = "candle")]
impl LVCBlock {
    /// Create a new LVC block
    pub fn new(vb: VarBuilder, channels: usize, config: LVCConfig) -> Result<Self> {
        // Main convolution with dilation
        let padding = (config.kernel_size - 1) * config.dilation / 2;
        let main_config = Conv1dConfig {
            padding,
            dilation: config.dilation,
            ..Default::default()
        };

        let main_weight = vb
            .pp("main")
            .get((channels, channels, config.kernel_size), "weight")
            .map_err(|e| {
                crate::VocoderError::ModelError(format!("Failed to create main conv weight: {}", e))
            })?;

        let main_bias = vb.pp("main").get(channels, "bias").map_err(|e| {
            crate::VocoderError::ModelError(format!("Failed to create main conv bias: {}", e))
        })?;

        let main_conv = Conv1d::new(main_weight, Some(main_bias), main_config);

        // Location convolution (predicts location-dependent modulation)
        let location_padding = config.location_kernel_size / 2;
        let location_config = Conv1dConfig {
            padding: location_padding,
            ..Default::default()
        };

        let location_weight = vb
            .pp("location")
            .get((channels, channels, config.location_kernel_size), "weight")
            .map_err(|e| {
                crate::VocoderError::ModelError(format!(
                    "Failed to create location conv weight: {}",
                    e
                ))
            })?;

        let location_bias = vb.pp("location").get(channels, "bias").map_err(|e| {
            crate::VocoderError::ModelError(format!("Failed to create location conv bias: {}", e))
        })?;

        let location_conv = Conv1d::new(location_weight, Some(location_bias), location_config);

        Ok(Self {
            main_conv,
            location_conv,
            config,
            channels,
        })
    }

    /// Forward pass through LVC block
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, channels, time]
    /// * `condition` - Optional conditioning input (e.g., mel spectrogram)
    ///
    /// # Returns
    /// Output tensor [batch, channels, time]
    pub fn forward(&self, x: &Tensor, condition: Option<&Tensor>) -> Result<Tensor> {
        // Compute location-dependent features
        let location_features = self.location_conv.forward(x).map_err(|e| {
            crate::VocoderError::ProcessingError(format!("Location conv error: {}", e))
        })?;

        // Apply sigmoid to get modulation weights in [0, 1]
        // sigmoid(x) = 1 / (1 + exp(-x))
        let neg_features = location_features
            .neg()
            .map_err(|e| crate::VocoderError::ProcessingError(format!("Negation error: {}", e)))?;
        let exp_neg = neg_features
            .exp()
            .map_err(|e| crate::VocoderError::ProcessingError(format!("Exp error: {}", e)))?;
        let one_plus_exp = (&exp_neg + 1.0)
            .map_err(|e| crate::VocoderError::ProcessingError(format!("Addition error: {}", e)))?;
        let location_weights = one_plus_exp
            .recip()
            .map_err(|e| crate::VocoderError::ProcessingError(format!("Recip error: {}", e)))?;

        // Apply main convolution
        let main_output = self
            .main_conv
            .forward(x)
            .map_err(|e| crate::VocoderError::ProcessingError(format!("Main conv error: {}", e)))?;

        // Modulate output with location-dependent weights
        let modulated = main_output.broadcast_mul(&location_weights).map_err(|e| {
            crate::VocoderError::ProcessingError(format!("Modulation error: {}", e))
        })?;

        // Add skip connection
        let output = (x + &modulated).map_err(|e| {
            crate::VocoderError::ProcessingError(format!("Skip connection error: {}", e))
        })?;

        Ok(output)
    }

    /// Get configuration
    pub fn config(&self) -> &LVCConfig {
        &self.config
    }

    /// Get number of channels
    pub fn channels(&self) -> usize {
        self.channels
    }
}

/// Residual LVC block with multiple layers
#[cfg(feature = "candle")]
pub struct ResLVCBlock {
    /// LVC layers
    lvc_layers: Vec<LVCBlock>,
    /// Number of layers
    num_layers: usize,
}

#[cfg(feature = "candle")]
impl ResLVCBlock {
    /// Create a new residual LVC block
    pub fn new(
        vb: VarBuilder,
        channels: usize,
        kernel_sizes: &[usize],
        dilations: &[usize],
        location_kernel_size: usize,
    ) -> Result<Self> {
        if kernel_sizes.len() != dilations.len() {
            return Err(crate::VocoderError::ConfigError(
                "kernel_sizes and dilations must have the same length".to_string(),
            ));
        }

        let num_layers = kernel_sizes.len();
        let mut lvc_layers = Vec::new();

        for (i, (&kernel_size, &dilation)) in kernel_sizes.iter().zip(dilations.iter()).enumerate()
        {
            let config = LVCConfig {
                kernel_size,
                dilation,
                location_kernel_size,
                causal: false,
            };

            let lvc = LVCBlock::new(vb.pp(format!("lvc_{}", i)), channels, config)?;

            lvc_layers.push(lvc);
        }

        Ok(Self {
            lvc_layers,
            num_layers,
        })
    }

    /// Forward pass through all LVC layers
    pub fn forward(&self, x: &Tensor, condition: Option<&Tensor>) -> Result<Tensor> {
        let mut output = x.clone();

        for lvc_layer in &self.lvc_layers {
            output = lvc_layer.forward(&output, condition)?;
        }

        Ok(output)
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lvc_config_default() {
        let config = LVCConfig::default();
        assert_eq!(config.kernel_size, 3);
        assert_eq!(config.dilation, 1);
        assert_eq!(config.location_kernel_size, 7);
        assert!(!config.causal);
    }

    #[test]
    fn test_lvc_config_custom() {
        let config = LVCConfig {
            kernel_size: 5,
            dilation: 2,
            location_kernel_size: 9,
            causal: true,
        };

        assert_eq!(config.kernel_size, 5);
        assert_eq!(config.dilation, 2);
        assert_eq!(config.location_kernel_size, 9);
        assert!(config.causal);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_lvc_block_creation() {
        use candle_core::Device;
        use candle_nn::VarBuilder;

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = LVCConfig::default();
        let lvc = LVCBlock::new(vb, 256, config);

        assert!(lvc.is_ok());
        let lvc = lvc.unwrap();
        assert_eq!(lvc.channels(), 256);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_res_lvc_block_creation() {
        use candle_core::Device;
        use candle_nn::VarBuilder;

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let kernel_sizes = vec![3, 7, 11];
        let dilations = vec![1, 3, 5];

        let res_lvc = ResLVCBlock::new(vb, 256, &kernel_sizes, &dilations, 7);

        assert!(res_lvc.is_ok());
        let res_lvc = res_lvc.unwrap();
        assert_eq!(res_lvc.num_layers(), 3);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_res_lvc_block_mismatched_lengths() {
        use candle_core::Device;
        use candle_nn::VarBuilder;

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let kernel_sizes = vec![3, 7];
        let dilations = vec![1, 3, 5];

        let res_lvc = ResLVCBlock::new(vb, 256, &kernel_sizes, &dilations, 7);

        assert!(res_lvc.is_err());
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_lvc_block_forward() {
        use candle_core::{DType, Device};
        use candle_nn::VarBuilder;

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = LVCConfig::default();
        let lvc = LVCBlock::new(vb, 64, config).unwrap();

        // Create test input [batch=1, channels=64, time=100]
        let x = Tensor::zeros((1, 64, 100), DType::F32, &device).unwrap();

        let output = lvc.forward(&x, None);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.dims(), &[1, 64, 100]);
    }
}
