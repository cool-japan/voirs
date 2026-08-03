//! Building blocks of the VITS2 generator.
//!
//! These are the real convolutional primitives the decoder is made of:
//!
//! * [`GeneratorActivation`] — the activation applied between convolutions,
//! * [`ResidualBlock`] — HiFi-GAN `ResBlock1`, a stack of dilated convolution
//!   pairs with residual connections,
//! * [`UpsampleBlock`] — a transposed convolution followed by multi-receptive
//!   field fusion over several [`ResidualBlock`]s.
//!
//! All of them own real `candle_nn` weights registered in the caller's
//! [`candle_nn::VarMap`]; nothing here is a scalar stand-in for a convolution.

use crate::{Result, VocoderError};
use candle_core::{Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder};
use serde::{Deserialize, Serialize};

/// Activation applied between the generator's convolutions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratorActivation {
    /// `max(x, slope * x)` — the HiFi-GAN / VITS default
    LeakyRelu,
    /// `max(x, 0)`
    Relu,
    /// Gaussian error linear unit
    Gelu,
    /// `x * sigmoid(x)`
    Silu,
}

impl GeneratorActivation {
    /// Parse an activation name (case-insensitive).
    ///
    /// # Errors
    /// Returns [`VocoderError::ModelError`] for an unsupported name.
    pub fn parse(name: &str) -> Result<Self> {
        match name
            .trim()
            .to_ascii_lowercase()
            .replace(['_', '-', ' '], "")
            .as_str()
        {
            "leakyrelu" => Ok(Self::LeakyRelu),
            "relu" => Ok(Self::Relu),
            "gelu" => Ok(Self::Gelu),
            "silu" | "swish" => Ok(Self::Silu),
            other => Err(VocoderError::ModelError(format!(
                "Unsupported generator activation '{other}' \
                 (expected LeakyReLU, ReLU, GELU or SiLU)"
            ))),
        }
    }

    /// Apply the activation to a tensor.
    ///
    /// # Errors
    /// Returns [`VocoderError::CandleError`] when the tensor operation fails.
    pub fn apply(self, x: &Tensor, leaky_relu_slope: f64) -> Result<Tensor> {
        Ok(match self {
            Self::LeakyRelu => candle_nn::ops::leaky_relu(x, leaky_relu_slope)?,
            Self::Relu => x.relu()?,
            Self::Gelu => x.gelu()?,
            Self::Silu => x.silu()?,
        })
    }
}

/// Multi-Receptive Field Fusion (MRF) block configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MRFConfig {
    /// Number of residual blocks
    pub n_blocks: u32,
    /// Kernel sizes for different receptive fields
    pub kernel_sizes: Vec<u32>,
    /// Dilation rates for each kernel size
    pub dilation_rates: Vec<Vec<u32>>,
    /// Use causal convolutions
    pub causal: bool,
}

impl Default for MRFConfig {
    fn default() -> Self {
        Self {
            n_blocks: 3,
            kernel_sizes: vec![3, 7, 11],
            dilation_rates: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            causal: false,
        }
    }
}

/// Configuration of a single dilated residual block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualBlockConfig {
    /// Number of channels
    pub channels: u32,
    /// Kernel size (must be odd so the block preserves the time axis)
    pub kernel_size: u32,
    /// Dilation rates, one convolution pair per entry
    pub dilation_rates: Vec<u32>,
    /// Use weight normalization
    pub use_weight_norm: bool,
    /// Leaky ReLU slope
    pub leaky_relu_slope: f32,
}

/// Residual block with dilated convolutions (HiFi-GAN `ResBlock1`).
///
/// For every dilation `d` it applies
/// `x = x + convs2[d]( act( convs1[d]( act(x) ) ) )`
/// where `convs1[d]` is dilated by `d` and `convs2[d]` is not dilated. Padding is
/// chosen so the time axis is preserved exactly.
#[derive(Debug, Clone)]
pub struct ResidualBlock {
    /// Block configuration
    pub config: ResidualBlockConfig,
    /// Dilated convolutions (one per dilation rate)
    convs1: Vec<Conv1d>,
    /// Non-dilated convolutions (one per dilation rate)
    convs2: Vec<Conv1d>,
    /// Activation applied before each convolution
    activation: GeneratorActivation,
}

impl ResidualBlock {
    /// Create a new residual block whose weights are registered under `vb`.
    ///
    /// `activation` is applied before each convolution;
    /// [`ResidualBlockConfig::leaky_relu_slope`] is used when it is
    /// [`GeneratorActivation::LeakyRelu`].
    ///
    /// # Errors
    /// Returns [`VocoderError::ModelError`] when the configuration cannot yield a
    /// shape-preserving convolution (even kernel size, zero channels, zero
    /// dilation), and [`VocoderError::CandleError`] if the tensors cannot be
    /// created.
    pub fn new(
        config: ResidualBlockConfig,
        activation: GeneratorActivation,
        vb: VarBuilder,
    ) -> Result<Self> {
        if config.channels == 0 {
            return Err(VocoderError::ModelError(
                "Residual block channels must be greater than 0".to_string(),
            ));
        }
        if config.kernel_size == 0 || config.kernel_size.is_multiple_of(2) {
            return Err(VocoderError::ModelError(format!(
                "Residual block kernel size must be odd (got {}) so the block preserves length",
                config.kernel_size
            )));
        }
        if config.dilation_rates.is_empty() {
            return Err(VocoderError::ModelError(
                "Residual block requires at least one dilation rate".to_string(),
            ));
        }
        if config.dilation_rates.contains(&0) {
            return Err(VocoderError::ModelError(
                "Residual block dilation rates must be greater than 0".to_string(),
            ));
        }

        let channels = config.channels as usize;
        let kernel_size = config.kernel_size as usize;
        let mut convs1 = Vec::with_capacity(config.dilation_rates.len());
        let mut convs2 = Vec::with_capacity(config.dilation_rates.len());

        for (i, &dilation) in config.dilation_rates.iter().enumerate() {
            let dilation = dilation as usize;
            convs1.push(candle_nn::conv1d(
                channels,
                channels,
                kernel_size,
                Conv1dConfig {
                    padding: (kernel_size - 1) * dilation / 2,
                    dilation,
                    ..Default::default()
                },
                vb.pp("convs1").pp(i),
            )?);
            convs2.push(candle_nn::conv1d(
                channels,
                channels,
                kernel_size,
                Conv1dConfig {
                    padding: (kernel_size - 1) / 2,
                    ..Default::default()
                },
                vb.pp("convs2").pp(i),
            )?);
        }

        Ok(Self {
            config,
            convs1,
            convs2,
            activation,
        })
    }

    /// Forward pass over a `[batch, channels, time]` tensor.
    ///
    /// # Errors
    /// Returns [`VocoderError::VocodingError`] on a channel-count mismatch and
    /// [`VocoderError::CandleError`] on tensor failures.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, channels, _) = x.dims3()?;
        if channels != self.config.channels as usize {
            return Err(VocoderError::VocodingError(format!(
                "Residual block expects {} channels, got {channels}",
                self.config.channels
            )));
        }

        let slope = self.config.leaky_relu_slope as f64;
        let mut out = x.clone();
        for (conv1, conv2) in self.convs1.iter().zip(self.convs2.iter()) {
            let h = self.activation.apply(&out, slope)?;
            let h = conv1.forward(&h)?;
            let h = self.activation.apply(&h, slope)?;
            let h = conv2.forward(&h)?;
            out = (out + h)?;
        }
        Ok(out)
    }

    /// Forward pass over a flat `[channels * time]` buffer in row-major
    /// (channel-major) order.
    ///
    /// # Errors
    /// Returns [`VocoderError::VocodingError`] when the buffer length is not a
    /// multiple of the block's channel count.
    pub fn forward_slice(&self, input: &[f32], device: &Device) -> Result<Vec<f32>> {
        let channels = self.config.channels as usize;
        if input.is_empty() || !input.len().is_multiple_of(channels) {
            return Err(VocoderError::VocodingError(format!(
                "Residual block input length {} is not a positive multiple of {channels} channels",
                input.len()
            )));
        }
        let time = input.len() / channels;
        let x = Tensor::from_slice(input, (1, channels, time), device)?;
        let out = self.forward(&x)?;
        Ok(out.flatten_all()?.to_vec1::<f32>()?)
    }

    /// Number of scalars held by this block.
    pub fn num_parameters(&self) -> u64 {
        self.convs1
            .iter()
            .chain(self.convs2.iter())
            .map(conv1d_parameters)
            .sum()
    }

    /// Receptive field of the block in samples.
    pub fn receptive_field(&self) -> u32 {
        let k = self.config.kernel_size;
        let mut field = 1u32;
        for &d in &self.config.dilation_rates {
            field += (k - 1) * d + (k - 1);
        }
        field
    }
}

/// Count the parameters of a `Conv1d` layer.
pub(crate) fn conv1d_parameters(conv: &Conv1d) -> u64 {
    conv.weight().elem_count() as u64 + conv.bias().map_or(0, |b| b.elem_count() as u64)
}

/// Count the parameters of a `ConvTranspose1d` layer.
pub(crate) fn conv_transpose1d_parameters(conv: &ConvTranspose1d) -> u64 {
    conv.weight().elem_count() as u64 + conv.bias().map_or(0, |b| b.elem_count() as u64)
}

/// Configuration of a single upsampling stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsampleBlockConfig {
    /// Input channels
    pub input_channels: u32,
    /// Output channels
    pub output_channels: u32,
    /// Upsampling rate (transposed-convolution stride)
    pub upsample_rate: u32,
    /// Kernel size for the transposed convolution
    pub kernel_size: u32,
    /// MRF configuration
    pub mrf_config: MRFConfig,
}

/// Upsampling block: one transposed convolution followed by multi-receptive
/// field fusion over several dilated residual blocks.
#[derive(Debug, Clone)]
pub struct UpsampleBlock {
    /// Block configuration
    pub config: UpsampleBlockConfig,
    /// Transposed convolution performing the actual upsampling
    up: ConvTranspose1d,
    /// MRF blocks for multi-receptive field processing
    mrf_blocks: Vec<ResidualBlock>,
    /// Activation applied before the transposed convolution
    activation: GeneratorActivation,
    /// Negative slope used when the activation is leaky ReLU
    leaky_relu_slope: f32,
}

impl UpsampleBlock {
    /// Create a new upsampling block.
    ///
    /// `vb_up` receives the transposed-convolution weights. The MRF residual
    /// blocks are registered under `vb_resblocks` at indices
    /// `resblock_base .. resblock_base + kernel_sizes.len()`, which lets the
    /// caller lay the parameters out with the flat HiFi-GAN naming
    /// (`ups.{i}` / `resblocks.{i * K + j}`).
    ///
    /// # Errors
    /// Returns [`VocoderError::ModelError`] for inconsistent configurations and
    /// [`VocoderError::CandleError`] on tensor failures.
    pub fn new(
        config: UpsampleBlockConfig,
        activation: GeneratorActivation,
        leaky_relu_slope: f32,
        vb_up: VarBuilder,
        vb_resblocks: &VarBuilder,
        resblock_base: usize,
    ) -> Result<Self> {
        if config.input_channels == 0 || config.output_channels == 0 {
            return Err(VocoderError::ModelError(
                "Upsample block channels must be greater than 0".to_string(),
            ));
        }
        if config.upsample_rate == 0 {
            return Err(VocoderError::ModelError(
                "Upsample rate must be greater than 0".to_string(),
            ));
        }
        if config.kernel_size < config.upsample_rate
            || !(config.kernel_size - config.upsample_rate).is_multiple_of(2)
        {
            return Err(VocoderError::ModelError(format!(
                "Upsample kernel size ({}) must be >= the rate ({}) and differ from it by an \
                 even amount so the stage upsamples by exactly the rate",
                config.kernel_size, config.upsample_rate
            )));
        }
        if config.mrf_config.kernel_sizes.is_empty() {
            return Err(VocoderError::ModelError(
                "MRF configuration requires at least one kernel size".to_string(),
            ));
        }
        if config.mrf_config.kernel_sizes.len() != config.mrf_config.dilation_rates.len() {
            return Err(VocoderError::ModelError(
                "MRF kernel sizes and dilation rate groups must have the same length".to_string(),
            ));
        }

        let up = candle_nn::conv_transpose1d(
            config.input_channels as usize,
            config.output_channels as usize,
            config.kernel_size as usize,
            ConvTranspose1dConfig {
                stride: config.upsample_rate as usize,
                padding: (config.kernel_size - config.upsample_rate) as usize / 2,
                dilation: 1,
                groups: 1,
                output_padding: 0,
            },
            vb_up,
        )?;

        let mut mrf_blocks = Vec::with_capacity(config.mrf_config.kernel_sizes.len());
        for (j, (&kernel_size, dilation_rates)) in config
            .mrf_config
            .kernel_sizes
            .iter()
            .zip(config.mrf_config.dilation_rates.iter())
            .enumerate()
        {
            let block_config = ResidualBlockConfig {
                channels: config.output_channels,
                kernel_size,
                dilation_rates: dilation_rates.clone(),
                use_weight_norm: true,
                leaky_relu_slope,
            };
            mrf_blocks.push(ResidualBlock::new(
                block_config,
                activation,
                vb_resblocks.pp(resblock_base + j),
            )?);
        }

        Ok(Self {
            config,
            up,
            mrf_blocks,
            activation,
            leaky_relu_slope,
        })
    }

    /// Forward pass over a `[batch, channels, time]` tensor.
    ///
    /// # Errors
    /// Returns [`VocoderError::VocodingError`] on a channel mismatch and
    /// [`VocoderError::CandleError`] on tensor failures.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, channels, _) = x.dims3()?;
        if channels != self.config.input_channels as usize {
            return Err(VocoderError::VocodingError(format!(
                "Upsample block expects {} channels, got {channels}",
                self.config.input_channels
            )));
        }

        let x = self.activation.apply(x, self.leaky_relu_slope as f64)?;
        let x = self.up.forward(&x)?;

        // Multi-receptive field fusion: average the outputs of all MRF blocks.
        let mut fused: Option<Tensor> = None;
        for block in &self.mrf_blocks {
            let out = block.forward(&x)?;
            fused = Some(match fused {
                Some(acc) => (acc + out)?,
                None => out,
            });
        }
        let fused = fused.ok_or_else(|| {
            VocoderError::VocodingError("Upsample block has no MRF residual blocks".to_string())
        })?;
        Ok(fused.affine(1.0 / self.mrf_blocks.len() as f64, 0.0)?)
    }

    /// Forward pass over a flat `[input_channels * time]` buffer.
    ///
    /// # Errors
    /// Returns [`VocoderError::VocodingError`] when the buffer length is not a
    /// multiple of the block's input channel count.
    pub fn forward_slice(&self, input: &[f32], device: &Device) -> Result<Vec<f32>> {
        let channels = self.config.input_channels as usize;
        if input.is_empty() || !input.len().is_multiple_of(channels) {
            return Err(VocoderError::VocodingError(format!(
                "Upsample block input length {} is not a positive multiple of {channels} channels",
                input.len()
            )));
        }
        let time = input.len() / channels;
        let x = Tensor::from_slice(input, (1, channels, time), device)?;
        let out = self.forward(&x)?;
        Ok(out.flatten_all()?.to_vec1::<f32>()?)
    }

    /// Number of scalars held by this block.
    pub fn num_parameters(&self) -> u64 {
        conv_transpose1d_parameters(&self.up)
            + self
                .mrf_blocks
                .iter()
                .map(|b| b.num_parameters())
                .sum::<u64>()
    }

    /// MRF residual blocks of this stage.
    pub fn mrf_blocks(&self) -> &[ResidualBlock] {
        &self.mrf_blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::vits2::params::seed_varmap;
    use candle_core::DType;
    use candle_nn::VarMap;

    fn ramp(len: usize, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.37 + offset).sin()) * 0.5)
            .collect()
    }

    #[test]
    fn test_residual_block_preserves_shape_and_uses_neighbours() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let config = ResidualBlockConfig {
            channels: 4,
            kernel_size: 3,
            dilation_rates: vec![1, 3],
            use_weight_norm: true,
            leaky_relu_slope: 0.1,
        };
        let block =
            ResidualBlock::new(config, GeneratorActivation::LeakyRelu, vb.pp("rb")).expect("block");
        seed_varmap(&varmap, 4242, &device).expect("seed");

        // 4 channels x 16 frames
        let input = ramp(4 * 16, 0.0);
        let output = block.forward_slice(&input, &device).expect("forward");
        assert_eq!(output.len(), input.len());

        // A dilated convolution has a receptive field: changing a single sample
        // must change *neighbouring* outputs too. Elementwise scaling could not.
        let mut perturbed = input.clone();
        perturbed[8] += 1.0;
        let perturbed_out = block.forward_slice(&perturbed, &device).expect("forward");
        let changed: Vec<usize> = output
            .iter()
            .zip(perturbed_out.iter())
            .enumerate()
            .filter(|(_, (a, b))| (*a - *b).abs() > 1e-6)
            .map(|(i, _)| i)
            .collect();
        assert!(
            changed.len() > 1,
            "dilated convolution must propagate a perturbation to neighbours, changed = {changed:?}"
        );
        // The perturbation must reach other channels as well (real conv mixes channels).
        assert!(changed.iter().any(|&i| i / 16 != 0));

        assert!(block.num_parameters() > 0);
        assert_eq!(
            block.num_parameters(),
            // 2 dilations x 2 convs x (4*4*3 weights + 4 bias)
            2 * 2 * (4 * 4 * 3 + 4)
        );
        assert!(block.receptive_field() > 1);
    }

    #[test]
    fn test_residual_block_rejects_even_kernel() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let config = ResidualBlockConfig {
            channels: 4,
            kernel_size: 4,
            dilation_rates: vec![1],
            use_weight_norm: true,
            leaky_relu_slope: 0.1,
        };
        assert!(ResidualBlock::new(config, GeneratorActivation::LeakyRelu, vb.pp("rb")).is_err());
    }

    #[test]
    fn test_residual_block_rejects_bad_input_length() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let config = ResidualBlockConfig {
            channels: 4,
            kernel_size: 3,
            dilation_rates: vec![1],
            use_weight_norm: true,
            leaky_relu_slope: 0.1,
        };
        let block =
            ResidualBlock::new(config, GeneratorActivation::LeakyRelu, vb.pp("rb")).expect("block");
        assert!(block.forward_slice(&[0.1, 0.2, 0.3], &device).is_err());
    }

    #[test]
    fn test_upsample_block_scales_time_axis() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let config = UpsampleBlockConfig {
            input_channels: 8,
            output_channels: 4,
            upsample_rate: 2,
            kernel_size: 4,
            mrf_config: MRFConfig {
                n_blocks: 2,
                kernel_sizes: vec![3, 5],
                dilation_rates: vec![vec![1, 3], vec![1, 3]],
                causal: false,
            },
        };
        let resblocks = vb.pp("resblocks");
        let block = UpsampleBlock::new(
            config,
            GeneratorActivation::LeakyRelu,
            0.1,
            vb.pp("ups").pp(0),
            &resblocks,
            0,
        )
        .expect("block");
        seed_varmap(&varmap, 77, &device).expect("seed");

        let frames = 12;
        let input = ramp(8 * frames, 0.5);
        let output = block.forward_slice(&input, &device).expect("forward");
        // 4 output channels x (12 * 2) frames
        assert_eq!(output.len(), 4 * frames * 2);
        assert!(block.num_parameters() > 0);
        assert_eq!(block.mrf_blocks().len(), 2);

        // Output must depend on the input.
        let other = ramp(8 * frames, 2.5);
        let other_out = block.forward_slice(&other, &device).expect("forward");
        assert_ne!(output, other_out);
    }
}
