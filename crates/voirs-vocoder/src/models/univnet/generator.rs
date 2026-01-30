//! UnivNet generator network architecture.

use super::config::UnivNetConfig;
use super::lvc::ResLVCBlock;
use crate::Result;

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "candle")]
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder};

/// Conditional normalization layer
///
/// Applies affine transformation conditioned on mel spectrogram features.
#[cfg(feature = "candle")]
pub struct ConditionalNorm {
    /// Scale convolution (learns gamma from condition)
    scale_conv: Conv1d,
    /// Bias convolution (learns beta from condition)
    bias_conv: Conv1d,
}

#[cfg(feature = "candle")]
impl ConditionalNorm {
    /// Create a new conditional normalization layer
    pub fn new(vb: VarBuilder, channels: usize, condition_channels: usize) -> Result<Self> {
        // 1x1 convolution to produce scale (gamma)
        let scale_weight = vb
            .pp("scale")
            .get((channels, condition_channels, 1), "weight")
            .map_err(|e| {
                crate::VocoderError::ModelError(format!("Failed to create scale weight: {}", e))
            })?;

        let scale_bias = vb.pp("scale").get(channels, "bias").map_err(|e| {
            crate::VocoderError::ModelError(format!("Failed to create scale bias: {}", e))
        })?;

        let scale_conv = Conv1d::new(scale_weight, Some(scale_bias), Conv1dConfig::default());

        // 1x1 convolution to produce bias (beta)
        let bias_weight = vb
            .pp("bias")
            .get((channels, condition_channels, 1), "weight")
            .map_err(|e| {
                crate::VocoderError::ModelError(format!("Failed to create bias weight: {}", e))
            })?;

        let bias_bias = vb.pp("bias").get(channels, "bias").map_err(|e| {
            crate::VocoderError::ModelError(format!("Failed to create bias bias: {}", e))
        })?;

        let bias_conv = Conv1d::new(bias_weight, Some(bias_bias), Conv1dConfig::default());

        Ok(Self {
            scale_conv,
            bias_conv,
        })
    }

    /// Forward pass: x_normalized = (x - mean) / std * scale + bias
    pub fn forward(&self, x: &Tensor, condition: &Tensor) -> Result<Tensor> {
        // Compute instance normalization statistics
        let mean = x.mean_keepdim(2).map_err(|e| {
            crate::VocoderError::ProcessingError(format!("Mean computation error: {}", e))
        })?;

        let centered = x
            .broadcast_sub(&mean)
            .map_err(|e| crate::VocoderError::ProcessingError(format!("Centering error: {}", e)))?;

        let variance = centered
            .sqr()
            .map_err(|e| {
                crate::VocoderError::ProcessingError(format!("Variance computation error: {}", e))
            })?
            .mean_keepdim(2)
            .map_err(|e| {
                crate::VocoderError::ProcessingError(format!("Variance mean error: {}", e))
            })?;

        let std = (&variance + 1e-5)
            .map_err(|e| {
                crate::VocoderError::ProcessingError(format!("Variance offset error: {}", e))
            })?
            .sqrt()
            .map_err(|e| crate::VocoderError::ProcessingError(format!("Sqrt error: {}", e)))?;

        let normalized = centered.broadcast_div(&std).map_err(|e| {
            crate::VocoderError::ProcessingError(format!("Normalization error: {}", e))
        })?;

        // Compute conditional scale and bias
        let scale = self.scale_conv.forward(condition).map_err(|e| {
            crate::VocoderError::ProcessingError(format!("Scale conv error: {}", e))
        })?;

        let bias = self
            .bias_conv
            .forward(condition)
            .map_err(|e| crate::VocoderError::ProcessingError(format!("Bias conv error: {}", e)))?;

        // Apply affine transformation
        let scaled = normalized
            .broadcast_mul(&scale)
            .map_err(|e| crate::VocoderError::ProcessingError(format!("Scaling error: {}", e)))?;

        let output = scaled.broadcast_add(&bias).map_err(|e| {
            crate::VocoderError::ProcessingError(format!("Bias addition error: {}", e))
        })?;

        Ok(output)
    }
}

/// UnivNet Generator network
#[cfg(feature = "candle")]
pub struct UnivNetGenerator {
    /// Initial convolution: mel -> initial channels
    input_conv: Conv1d,

    /// Upsampling blocks with transpose convolutions
    upsampling_blocks: Vec<ConvTranspose1d>,

    /// LVC blocks after each upsampling
    lvc_blocks: Vec<ResLVCBlock>,

    /// Conditional normalization layers (optional)
    cond_norms: Vec<Option<ConditionalNorm>>,

    /// Final convolution to waveform
    output_conv: Conv1d,

    /// Configuration
    config: UnivNetConfig,
}

#[cfg(feature = "candle")]
impl UnivNetGenerator {
    /// Create a new UnivNet generator
    pub fn new(vb: VarBuilder, config: UnivNetConfig) -> Result<Self> {
        // Validate configuration
        config
            .validate()
            .map_err(|e| crate::VocoderError::ConfigError(format!("Invalid config: {}", e)))?;

        // Input convolution: mel -> initial channels
        let input_weight = vb
            .pp("input_conv")
            .get(
                (config.upsample_initial_channel, config.num_mels, 7),
                "weight",
            )
            .map_err(|e| {
                crate::VocoderError::ModelError(format!("Failed to create input conv: {}", e))
            })?;

        let input_bias = vb
            .pp("input_conv")
            .get(config.upsample_initial_channel, "bias")
            .map_err(|e| {
                crate::VocoderError::ModelError(format!("Failed to create input bias: {}", e))
            })?;

        let input_conv = Conv1d::new(
            input_weight,
            Some(input_bias),
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
        );

        // Create upsampling blocks and LVC blocks
        let mut upsampling_blocks = Vec::new();
        let mut lvc_blocks = Vec::new();
        let mut cond_norms = Vec::new();

        let mut current_channels = config.upsample_initial_channel;

        for i in 0..config.num_upsamples {
            let next_channels = current_channels / 2;
            let upsample_rate = config.upsample_rates[i];
            let kernel_size = config.upsample_kernel_sizes[i];

            // Upsampling transpose convolution
            let stride = upsample_rate;
            let padding = (kernel_size - upsample_rate) / 2;

            let upsample_config = ConvTranspose1dConfig {
                stride,
                padding,
                ..Default::default()
            };

            let upsample_weight = vb
                .pp(format!("upsample_{}", i))
                .get((current_channels, next_channels, kernel_size), "weight")
                .map_err(|e| {
                    crate::VocoderError::ModelError(format!(
                        "Failed to create upsample {}: {}",
                        i, e
                    ))
                })?;

            let upsample_bias = vb
                .pp(format!("upsample_{}", i))
                .get(next_channels, "bias")
                .map_err(|e| {
                    crate::VocoderError::ModelError(format!(
                        "Failed to create upsample bias {}: {}",
                        i, e
                    ))
                })?;

            let upsample =
                ConvTranspose1d::new(upsample_weight, Some(upsample_bias), upsample_config);
            upsampling_blocks.push(upsample);

            // ResLVC block
            let res_lvc = ResLVCBlock::new(
                vb.pp(format!("res_lvc_{}", i)),
                next_channels,
                &config.lvc_kernel_sizes,
                &config
                    .lvc_dilations
                    .iter()
                    .map(|d| d[0])
                    .collect::<Vec<_>>(),
                config.location_kernel_size,
            )?;
            lvc_blocks.push(res_lvc);

            // Conditional normalization (if enabled)
            if config.use_cond_norm {
                let cond_norm = ConditionalNorm::new(
                    vb.pp(format!("cond_norm_{}", i)),
                    next_channels,
                    config.num_mels,
                )?;
                cond_norms.push(Some(cond_norm));
            } else {
                cond_norms.push(None);
            }

            current_channels = next_channels;
        }

        // Final convolution to waveform (mono)
        let output_weight = vb
            .pp("output_conv")
            .get((1, current_channels, 7), "weight")
            .map_err(|e| {
                crate::VocoderError::ModelError(format!("Failed to create output conv: {}", e))
            })?;

        let output_bias = vb.pp("output_conv").get(1, "bias").map_err(|e| {
            crate::VocoderError::ModelError(format!("Failed to create output bias: {}", e))
        })?;

        let output_conv = Conv1d::new(
            output_weight,
            Some(output_bias),
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
        );

        Ok(Self {
            input_conv,
            upsampling_blocks,
            lvc_blocks,
            cond_norms,
            output_conv,
            config,
        })
    }

    /// Forward pass: mel spectrogram -> waveform
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // Initial convolution
        let mut x = self.input_conv.forward(mel).map_err(|e| {
            crate::VocoderError::ProcessingError(format!("Input conv error: {}", e))
        })?;

        // Apply LeakyReLU
        x = x
            .relu()
            .map_err(|e| crate::VocoderError::ProcessingError(format!("ReLU error: {}", e)))?;

        // Process through upsampling blocks
        for i in 0..self.config.num_upsamples {
            // Upsampling
            x = self.upsampling_blocks[i].forward(&x).map_err(|e| {
                crate::VocoderError::ProcessingError(format!("Upsample {} error: {}", i, e))
            })?;

            // Conditional normalization (if enabled)
            if let Some(ref cond_norm) = self.cond_norms[i] {
                x = cond_norm.forward(&x, mel)?;
            }

            // LeakyReLU activation
            x = x.relu().map_err(|e| {
                crate::VocoderError::ProcessingError(format!("ReLU {} error: {}", i, e))
            })?;

            // LVC block
            x = self.lvc_blocks[i].forward(&x, Some(mel))?;
        }

        // Final convolution
        x = self.output_conv.forward(&x).map_err(|e| {
            crate::VocoderError::ProcessingError(format!("Output conv error: {}", e))
        })?;

        // Apply tanh to bound output to [-1, 1]
        let waveform = x
            .tanh()
            .map_err(|e| crate::VocoderError::ProcessingError(format!("Tanh error: {}", e)))?;

        Ok(waveform)
    }

    /// Get model configuration
    pub fn config(&self) -> &UnivNetConfig {
        &self.config
    }

    /// Calculate number of parameters
    pub fn num_parameters(&self) -> usize {
        self.config.estimate_parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarBuilder;

    #[cfg(feature = "candle")]
    #[test]
    fn test_conditional_norm_creation() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let cond_norm = ConditionalNorm::new(vb, 256, 80);
        assert!(cond_norm.is_ok());
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_univnet_generator_creation() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = UnivNetConfig::fast_24khz();
        let generator = UnivNetGenerator::new(vb, config);
        assert!(generator.is_ok());
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_univnet_all_configs() {
        let device = Device::Cpu;

        let configs = [
            UnivNetConfig::fast_24khz(),
            UnivNetConfig::base_24khz(),
            UnivNetConfig::large_24khz(),
        ];

        for config in configs {
            let vb = VarBuilder::zeros(DType::F32, &device);
            let generator = UnivNetGenerator::new(vb, config);
            assert!(generator.is_ok());
        }
    }

    #[test]
    fn test_parameter_count_reasonable() {
        let configs = [
            UnivNetConfig::fast_24khz(),
            UnivNetConfig::base_24khz(),
            UnivNetConfig::large_24khz(),
        ];

        for config in configs {
            let params = config.estimate_parameters();
            assert!(params > 1_000_000, "Too few parameters: {}", params);
            assert!(params < 30_000_000, "Too many parameters: {}", params);
        }
    }
}
