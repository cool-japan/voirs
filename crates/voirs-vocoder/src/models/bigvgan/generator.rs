//! BigVGAN generator network architecture.
//!
//! Implements the generator network with anti-aliased periodic activations
//! and multi-periodicity composition for high-quality audio generation.

// Every type below (`AMPBlock`, `ResBlock`, `MRF`, `BigVGANGenerator`, ...) is
// only defined when the `candle` feature is enabled, so these imports are
// only used in that configuration.
#[cfg(feature = "candle")]
use super::activation::{ActivationConfig, AntiAliasedSnakeActivation};
#[cfg(feature = "candle")]
use super::config::BigVGANConfig;
#[cfg(feature = "candle")]
use crate::Result;

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "candle")]
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder};

/// Anti-aliased Multi-Periodicity Composition Block (AMPBlock)
///
/// Applies multiple periodic activations with different frequencies
/// to capture rich harmonic structure.
#[cfg(feature = "candle")]
pub struct AMPBlock {
    /// Periodic activation layers with different periods
    activations: Vec<AntiAliasedSnakeActivation>,
    /// Convolution for combining multiple periods
    combine_conv: Conv1d,
    /// Configuration
    periods: Vec<usize>,
}

#[cfg(feature = "candle")]
impl AMPBlock {
    /// Create a new AMP block
    pub fn new(
        vb: VarBuilder,
        channels: usize,
        periods: Vec<usize>,
        activation_config: &ActivationConfig,
    ) -> Result<Self> {
        let mut activations = Vec::new();

        // Create activation for each period
        for (i, &period) in periods.iter().enumerate() {
            let mut period_config = activation_config.clone();
            // Scale alpha by period for different frequencies
            period_config.alpha = activation_config.alpha / (period as f32);

            let activation =
                AntiAliasedSnakeActivation::new(vb.pp(format!("activation_{}", i)), period_config)?;
            activations.push(activation);
        }

        // Convolution to combine different periodic components
        let combine_conv = Conv1d::new(
            vb.pp("combine")
                .get((periods.len() * channels, channels, 1), "weight")
                .map_err(|e| {
                    crate::VocoderError::ModelError(format!("Failed to create combine conv: {}", e))
                })?,
            Some(vb.pp("combine").get(channels, "bias").map_err(|e| {
                crate::VocoderError::ModelError(format!("Failed to create bias: {}", e))
            })?),
            Conv1dConfig::default(),
        );

        Ok(Self {
            activations,
            combine_conv,
            periods: periods.clone(),
        })
    }

    /// Forward pass through AMP block
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut period_outputs = Vec::new();

        // Apply each periodic activation
        for activation in &self.activations {
            let activated = activation.forward(x)?;
            period_outputs.push(activated);
        }

        // Concatenate along channel dimension
        let concatenated = Tensor::cat(&period_outputs, 1).map_err(|e| {
            crate::VocoderError::ProcessingError(format!(
                "Failed to concatenate period outputs: {}",
                e
            ))
        })?;

        // Combine with 1x1 convolution
        let combined = self.combine_conv.forward(&concatenated).map_err(|e| {
            crate::VocoderError::ProcessingError(format!("Failed to combine periods: {}", e))
        })?;

        Ok(combined)
    }
}

/// Residual block with anti-aliased activation
#[cfg(feature = "candle")]
pub struct ResBlock {
    /// Convolutions in residual path
    convs: Vec<Conv1d>,
    /// Anti-aliased activations
    activations: Vec<AntiAliasedSnakeActivation>,
    /// Number of layers
    num_layers: usize,
}

#[cfg(feature = "candle")]
impl ResBlock {
    /// Create a new residual block
    pub fn new(
        vb: VarBuilder,
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
        activation_config: &ActivationConfig,
    ) -> Result<Self> {
        let num_layers = dilations.len();
        let mut convs = Vec::new();
        let mut activations = Vec::new();

        for (i, &dilation) in dilations.iter().enumerate() {
            // Create convolution with dilation
            let padding = (kernel_size - 1) * dilation / 2;
            let conv_config = Conv1dConfig {
                padding,
                dilation,
                ..Default::default()
            };

            let weight = vb
                .pp(format!("conv_{}", i))
                .get((channels, channels, kernel_size), "weight")
                .map_err(|e| {
                    crate::VocoderError::ModelError(format!("Failed to create conv weight: {}", e))
                })?;

            let bias = vb
                .pp(format!("conv_{}", i))
                .get(channels, "bias")
                .map_err(|e| {
                    crate::VocoderError::ModelError(format!("Failed to create conv bias: {}", e))
                })?;

            let conv = Conv1d::new(weight, Some(bias), conv_config);
            convs.push(conv);

            // Create anti-aliased activation
            let activation = AntiAliasedSnakeActivation::new(
                vb.pp(format!("activation_{}", i)),
                activation_config.clone(),
            )?;
            activations.push(activation);
        }

        Ok(Self {
            convs,
            activations,
            num_layers,
        })
    }

    /// Forward pass through residual block
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut residual = x.clone();

        for i in 0..self.num_layers {
            // Apply convolution
            let conv_out = self.convs[i].forward(&residual).map_err(|e| {
                crate::VocoderError::ProcessingError(format!("Conv {} error: {}", i, e))
            })?;

            // Apply anti-aliased activation
            let activated = self.activations[i].forward(&conv_out)?;

            residual = activated;
        }

        // Add skip connection
        let output = (x + &residual).map_err(|e| {
            crate::VocoderError::ProcessingError(format!("Failed to add residual: {}", e))
        })?;

        Ok(output)
    }
}

/// Multi-Receptive Field Fusion (MRF) module
///
/// Combines multiple residual blocks with different kernel sizes
/// for multi-scale feature extraction.
#[cfg(feature = "candle")]
pub struct MRF {
    /// Residual blocks with different receptive fields
    res_blocks: Vec<ResBlock>,
}

#[cfg(feature = "candle")]
impl MRF {
    /// Create a new MRF module
    pub fn new(
        vb: VarBuilder,
        channels: usize,
        kernel_sizes: &[usize],
        dilations: &[Vec<usize>],
        activation_config: &ActivationConfig,
    ) -> Result<Self> {
        let mut res_blocks = Vec::new();

        for (i, (&kernel_size, dilation_set)) in
            kernel_sizes.iter().zip(dilations.iter()).enumerate()
        {
            let res_block = ResBlock::new(
                vb.pp(format!("resblock_{}", i)),
                channels,
                kernel_size,
                dilation_set,
                activation_config,
            )?;
            res_blocks.push(res_block);
        }

        Ok(Self { res_blocks })
    }

    /// Forward pass through MRF
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut outputs = Vec::new();

        // Process through each residual block
        for res_block in &self.res_blocks {
            let output = res_block.forward(x)?;
            outputs.push(output);
        }

        // Average the outputs
        let mut sum = outputs[0].clone();

        for output in &outputs[1..] {
            sum = (&sum + output).map_err(|e| {
                crate::VocoderError::ProcessingError(format!("Failed to sum outputs: {}", e))
            })?;
        }

        let avg = (sum / (outputs.len() as f64)).map_err(|e| {
            crate::VocoderError::ProcessingError(format!("Failed to average outputs: {}", e))
        })?;

        Ok(avg)
    }
}

/// BigVGAN Generator network
#[cfg(feature = "candle")]
pub struct BigVGANGenerator {
    /// Initial convolution
    input_conv: Conv1d,

    /// Upsampling blocks
    upsampling_blocks: Vec<(ConvTranspose1d, AntiAliasedSnakeActivation)>,

    /// MRF modules after each upsampling
    mrf_modules: Vec<MRF>,

    /// AMP blocks (optional)
    amp_blocks: Vec<Option<AMPBlock>>,

    /// Final convolution to waveform
    output_conv: Conv1d,

    /// Output activation
    output_activation: AntiAliasedSnakeActivation,

    /// Configuration
    config: BigVGANConfig,
}

#[cfg(feature = "candle")]
impl BigVGANGenerator {
    /// Create a new BigVGAN generator
    pub fn new(vb: VarBuilder, config: BigVGANConfig) -> Result<Self> {
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

        // Create upsampling blocks, MRF modules, and AMP blocks
        let mut upsampling_blocks = Vec::new();
        let mut mrf_modules = Vec::new();
        let mut amp_blocks = Vec::new();

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

            // Activation after upsampling
            let activation = AntiAliasedSnakeActivation::new(
                vb.pp(format!("upsample_activation_{}", i)),
                config.activation_config.clone(),
            )?;

            upsampling_blocks.push((upsample, activation));

            // MRF module
            let mrf = MRF::new(
                vb.pp(format!("mrf_{}", i)),
                next_channels,
                &config.resblock_kernel_sizes,
                &config.resblock_dilation_sizes,
                &config.activation_config,
            )?;
            mrf_modules.push(mrf);

            // AMP block (if enabled)
            if config.use_amp_block {
                let amp = AMPBlock::new(
                    vb.pp(format!("amp_{}", i)),
                    next_channels,
                    config.amp_block_periods.clone(),
                    &config.activation_config,
                )?;
                amp_blocks.push(Some(amp));
            } else {
                amp_blocks.push(None);
            }

            current_channels = next_channels;
        }

        // Final convolution to waveform (mono)
        // Conv1d expects (out_channels, in_channels, kernel_size)
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

        // Final activation
        let output_activation = AntiAliasedSnakeActivation::new(
            vb.pp("output_activation"),
            config.activation_config.clone(),
        )?;

        Ok(Self {
            input_conv,
            upsampling_blocks,
            mrf_modules,
            amp_blocks,
            output_conv,
            output_activation,
            config,
        })
    }

    /// Forward pass: mel spectrogram -> waveform
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // Initial convolution
        let mut x = self.input_conv.forward(mel).map_err(|e| {
            crate::VocoderError::ProcessingError(format!("Input conv error: {}", e))
        })?;

        // Process through upsampling blocks
        for i in 0..self.config.num_upsamples {
            // Upsampling
            let (upsample, activation) = &self.upsampling_blocks[i];
            x = upsample.forward(&x).map_err(|e| {
                crate::VocoderError::ProcessingError(format!("Upsample {} error: {}", i, e))
            })?;

            x = activation.forward(&x)?;

            // MRF module
            x = self.mrf_modules[i].forward(&x)?;

            // AMP block (if enabled)
            if let Some(ref amp) = self.amp_blocks[i] {
                x = amp.forward(&x)?;
            }
        }

        // Final convolution and activation
        x = self.output_conv.forward(&x).map_err(|e| {
            crate::VocoderError::ProcessingError(format!("Output conv error: {}", e))
        })?;

        x = self.output_activation.forward(&x)?;

        // Apply tanh to bound output to [-1, 1]
        let waveform = x
            .tanh()
            .map_err(|e| crate::VocoderError::ProcessingError(format!("Tanh error: {}", e)))?;

        Ok(waveform)
    }

    /// Get model configuration
    pub fn config(&self) -> &BigVGANConfig {
        &self.config
    }

    /// Calculate number of parameters
    ///
    /// Computes the total number of trainable parameters in the generator network.
    /// This includes all convolution weights, biases, and activation parameters.
    pub fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Input convolution parameters
        // Conv1d: (out_channels, in_channels, kernel_size) + bias
        total += self.config.upsample_initial_channel * self.config.num_mels * 7; // Weights
        total += self.config.upsample_initial_channel; // Bias

        let mut current_channels = self.config.upsample_initial_channel;

        // Upsampling blocks + MRF modules + AMP blocks
        for i in 0..self.config.num_upsamples {
            let next_channels = current_channels / 2;
            let kernel_size = self.config.upsample_kernel_sizes[i];

            // Upsampling transpose convolution
            total += current_channels * next_channels * kernel_size; // Weights
            total += next_channels; // Bias

            // Anti-aliased activation after upsampling
            // Each activation has learnable alpha parameters per channel
            total += next_channels;

            // MRF (Multi-Receptive Field) module parameters
            // Each MRF contains multiple ResBlocks
            for &kernel_size in &self.config.resblock_kernel_sizes {
                for dilations in &self.config.resblock_dilation_sizes {
                    // Each ResBlock has convolutions with specified dilations
                    for _ in dilations {
                        // Conv1d in residual path: (channels, channels, kernel_size)
                        total += next_channels * next_channels * kernel_size; // Weights
                        total += next_channels; // Bias

                        // Anti-aliased activation parameters
                        total += next_channels;
                    }
                }
            }

            // AMP (Anti-aliased Multi-Periodicity) block (if enabled)
            if self.config.use_amp_block {
                let num_periods = self.config.amp_block_periods.len();

                // Each period has an anti-aliased activation
                total += num_periods * next_channels;

                // Combine convolution: (num_periods * channels, channels, 1)
                total += num_periods * next_channels * next_channels; // Weights
                total += next_channels; // Bias
            }

            current_channels = next_channels;
        }

        // Output convolution parameters
        // Conv1d: (1, current_channels, 7) + bias
        total += current_channels * 7; // Weights (1 output channel)
        total += 1; // Bias

        // Output activation
        total += 1; // Alpha parameters for output activation

        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarBuilder;

    #[cfg(feature = "candle")]
    #[test]
    fn test_resblock_creation() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = ActivationConfig::default();
        let dilations = vec![1, 3, 5];

        let resblock = ResBlock::new(vb, 256, 3, &dilations, &config);
        assert!(resblock.is_ok());
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_mrf_creation() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = ActivationConfig::default();
        let kernel_sizes = vec![3, 7, 11];
        let dilations = vec![vec![1, 3, 5]; 3];

        let mrf = MRF::new(vb, 256, &kernel_sizes, &dilations, &config);
        assert!(mrf.is_ok());
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_bigvgan_generator_creation() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let config = BigVGANConfig::fast_24khz(); // Use fast config for testing
        let generator = BigVGANGenerator::new(vb, config);
        assert!(generator.is_ok());
    }

    #[test]
    fn test_bigvgan_config_validation() {
        let config = BigVGANConfig::base_24khz();
        assert!(config.validate().is_ok());
    }
}
