//! VITS2 Neural Vocoder Generator
//!
//! This module implements the VITS2 generator (decoder) network that converts a
//! latent / mel representation into a waveform. It is a real HiFi-GAN-style
//! generator built on `candle_nn`:
//!
//! ```text
//! conv_pre (Conv1d, k=7)  ->  [+ cond (Conv1d, k=1) when conditioning is given]
//!   for each upsampling stage i:
//!       LeakyReLU  ->  ups.i (ConvTranspose1d, stride = upsample_rate)
//!       ->  mean over j of resblocks.{i*K+j}(x)      (multi-receptive-field fusion)
//!   LeakyReLU  ->  conv_post (Conv1d, k=7)  ->  tanh
//! ```
//!
//! Every layer owns real weights registered in a [`VarMap`], so
//! [`Vits2Generator::num_parameters`] reports the number of scalars the model
//! actually holds, the output genuinely depends on the input latent and on the
//! weights, and [`Vits2Generator::load_weights`] can populate the network from a
//! SafeTensors checkpoint. Parameter names follow the reference HiFi-GAN / VITS
//! `state_dict` layout (`conv_pre`, `ups.{i}`, `resblocks.{k}.convs1.{d}`,
//! `conv_post`, `cond`) so that converted checkpoints map across directly.

use super::blocks::conv1d_parameters;
pub use super::blocks::{
    GeneratorActivation, MRFConfig, ResidualBlock, ResidualBlockConfig, UpsampleBlock,
    UpsampleBlockConfig,
};
use super::params::{
    count_parameters, load_safetensors_into_varmap_with_mode, seed_varmap, split_indexed_prefix,
    strip_checkpoint_prefixes, LoadMode, WeightLoadReport,
};
use crate::{Result, VocoderError};
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder, VarMap,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Default seed used when a generator is created without an explicit seed.
pub const DEFAULT_GENERATOR_SEED: u64 = 0x5649_5453_3200_0001;

/// VITS2 Generator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorConfig {
    /// Input dimension (latent / mel channel count)
    pub input_dim: u32,
    /// Reference hidden width (used for memory accounting)
    pub hidden_channels: u32,
    /// Channel count directly after the input convolution
    pub initial_channel: u32,
    /// Upsampling rates for each upsampling block
    pub upsample_rates: Vec<u32>,
    /// Kernel sizes for upsampling layers
    pub upsample_kernel_sizes: Vec<u32>,
    /// Kernel sizes for residual blocks
    pub resblock_kernel_sizes: Vec<u32>,
    /// Dilation sizes for residual blocks
    pub resblock_dilation_sizes: Vec<Vec<u32>>,
    /// Global conditioning channels (for speaker/emotion embedding)
    pub gin_channels: u32,
    /// Whether checkpoints for this model store PyTorch `weight_norm` parameter
    /// pairs (`weight_g` / `weight_v`).
    ///
    /// This describes the *checkpoint format*, not a runtime option: inference
    /// always uses plain weights, and [`Vits2Generator::load_weights`] fuses any
    /// `weight_g`/`weight_v` pair it encounters into the effective weight.
    pub use_weight_norm: bool,
    /// Use spectral normalization.
    ///
    /// Spectral normalization is a discriminator-side training technique and is
    /// not implemented for the generator; setting this to `true` is rejected by
    /// [`GeneratorConfig::validate`] rather than silently ignored.
    pub use_spectral_norm: bool,
    /// Activation applied between convolutions.
    ///
    /// One of `LeakyReLU`, `ReLU`, `GELU` or `SiLU` (case-insensitive); see
    /// [`GeneratorActivation`]. Unknown values are rejected by
    /// [`GeneratorConfig::validate`].
    pub activation: String,
    /// Negative slope used when [`Self::activation`] is `LeakyReLU`
    pub leaky_relu_slope: f32,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            input_dim: 192,
            hidden_channels: 512,
            initial_channel: 512,
            upsample_rates: vec![8, 8, 2, 2],
            upsample_kernel_sizes: vec![16, 16, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            gin_channels: 256,
            use_weight_norm: true,
            use_spectral_norm: false,
            activation: "LeakyReLU".to_string(),
            leaky_relu_slope: 0.1,
        }
    }
}

/// Main VITS2 Generator
///
/// Not `Clone`: the parameters live in a shared [`VarMap`], so a clone would
/// alias the same weights while carrying its own `weights_loaded` flag — which
/// would let `is_pretrained()` report a state that does not match the tensors.
pub struct Vits2Generator {
    /// Generator configuration
    pub config: GeneratorConfig,
    /// Input convolution (`input_dim` -> `initial_channel`)
    conv_pre: Conv1d,
    /// Global conditioning projection (`gin_channels` -> `initial_channel`)
    cond: Option<Conv1d>,
    /// Upsampling blocks
    upsample_blocks: Vec<UpsampleBlock>,
    /// Output convolution (last channel count -> 1)
    conv_post: Conv1d,
    /// Activation applied between convolutions
    activation: GeneratorActivation,
    /// All learnable parameters
    varmap: VarMap,
    /// Compute device
    device: Device,
    /// Whether a checkpoint has been loaded into [`Self::varmap`]
    weights_loaded: bool,
}

impl std::fmt::Debug for Vits2Generator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vits2Generator")
            .field("config", &self.config)
            .field("upsample_blocks", &self.upsample_blocks.len())
            .field("device", &self.device)
            .field("weights_loaded", &self.weights_loaded)
            .finish()
    }
}

impl Vits2Generator {
    /// Create a generator on the CPU with deterministic initialization.
    ///
    /// The weights are pseudo-random but reproducible (see
    /// [`DEFAULT_GENERATOR_SEED`]); the model is **not** trained. Call
    /// [`Vits2Generator::load_weights`] to load a real checkpoint, or use
    /// [`Vits2Generator::generate_pretrained`] to fail closed when no checkpoint
    /// has been loaded.
    ///
    /// # Errors
    /// Returns [`VocoderError::ModelError`] for an invalid configuration.
    pub fn new(config: GeneratorConfig) -> Result<Self> {
        Self::new_seeded(config, Device::Cpu, DEFAULT_GENERATOR_SEED)
    }

    /// Create a generator on a specific device with deterministic initialization.
    ///
    /// # Errors
    /// Returns [`VocoderError::ModelError`] for an invalid configuration.
    pub fn new_with_device(config: GeneratorConfig, device: Device) -> Result<Self> {
        Self::new_seeded(config, device, DEFAULT_GENERATOR_SEED)
    }

    /// Create a generator with an explicit initialization seed.
    ///
    /// # Errors
    /// Returns [`VocoderError::ModelError`] for an invalid configuration and
    /// [`VocoderError::CandleError`] when parameter tensors cannot be created.
    pub fn new_seeded(config: GeneratorConfig, device: Device, seed: u64) -> Result<Self> {
        config.validate()?;
        let activation = GeneratorActivation::parse(&config.activation)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let conv_pre = candle_nn::conv1d(
            config.input_dim as usize,
            config.initial_channel as usize,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("conv_pre"),
        )?;

        let cond = if config.gin_channels > 0 {
            Some(candle_nn::conv1d(
                config.gin_channels as usize,
                config.initial_channel as usize,
                1,
                Conv1dConfig::default(),
                vb.pp("cond"),
            )?)
        } else {
            None
        };

        let num_kernels = config.resblock_kernel_sizes.len();
        let vb_ups = vb.pp("ups");
        let vb_resblocks = vb.pp("resblocks");

        let mut upsample_blocks = Vec::with_capacity(config.upsample_rates.len());
        let mut current_channels = config.initial_channel;

        for (i, (&upsample_rate, &kernel_size)) in config
            .upsample_rates
            .iter()
            .zip(config.upsample_kernel_sizes.iter())
            .enumerate()
        {
            // HiFi-GAN halves the channel count at every stage.
            let output_channels = (config.initial_channel >> (i + 1)).max(1);

            let block_config = UpsampleBlockConfig {
                input_channels: current_channels,
                output_channels,
                upsample_rate,
                kernel_size,
                mrf_config: MRFConfig {
                    n_blocks: num_kernels as u32,
                    kernel_sizes: config.resblock_kernel_sizes.clone(),
                    dilation_rates: config.resblock_dilation_sizes.clone(),
                    causal: false,
                },
            };

            // Flat `resblocks.{i * K + j}` numbering, matching HiFi-GAN.
            upsample_blocks.push(UpsampleBlock::new(
                block_config,
                activation,
                config.leaky_relu_slope,
                vb_ups.pp(i),
                &vb_resblocks,
                i * num_kernels,
            )?);

            current_channels = output_channels;
        }

        let conv_post = candle_nn::conv1d(
            current_channels as usize,
            1,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("conv_post"),
        )?;

        seed_varmap(&varmap, seed, &device)?;

        Ok(Self {
            config,
            conv_pre,
            cond,
            upsample_blocks,
            conv_post,
            activation,
            varmap,
            device,
            weights_loaded: false,
        })
    }

    /// Generate a waveform from a latent representation.
    ///
    /// `latent` is a flat `[input_dim * frames]` buffer in channel-major order
    /// (all frames of channel 0, then channel 1, ...). `conditioning`, when
    /// given, must contain exactly `gin_channels` values.
    ///
    /// The returned buffer has `frames * product(upsample_rates)` samples.
    ///
    /// # Errors
    /// Returns [`VocoderError::VocodingError`] for empty or mis-shaped input and
    /// [`VocoderError::CandleError`] on tensor failures.
    pub fn generate(&self, latent: &[f32], conditioning: Option<&[f32]>) -> Result<Vec<f32>> {
        let input_dim = self.config.input_dim as usize;
        if latent.is_empty() {
            return Err(VocoderError::VocodingError(
                "Empty latent input".to_string(),
            ));
        }
        if !latent.len().is_multiple_of(input_dim) {
            return Err(VocoderError::VocodingError(format!(
                "Latent length {} is not a multiple of the generator input dimension {input_dim}",
                latent.len()
            )));
        }
        let frames = latent.len() / input_dim;
        let latent_tensor = Tensor::from_slice(latent, (1, input_dim, frames), &self.device)?;

        let cond_tensor = match conditioning {
            Some(cond) => Some(self.conditioning_tensor(cond)?),
            None => None,
        };

        let audio = self.generate_tensor(&latent_tensor, cond_tensor.as_ref())?;
        Ok(audio.flatten_all()?.to_vec1::<f32>()?)
    }

    /// Generate a waveform, failing closed when no checkpoint has been loaded.
    ///
    /// Use this in production paths where fabricated audio from an untrained
    /// network would be worse than an explicit error.
    ///
    /// # Errors
    /// Returns [`VocoderError::ModelError`] when [`Vits2Generator::load_weights`]
    /// has not been called successfully.
    pub fn generate_pretrained(
        &self,
        latent: &[f32],
        conditioning: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        if !self.weights_loaded {
            return Err(VocoderError::ModelError(
                "VITS2 generator has no pretrained weights loaded — \
                 call load_weights(<path to .safetensors>) first"
                    .to_string(),
            ));
        }
        self.generate(latent, conditioning)
    }

    /// Build the conditioning tensor `[1, gin_channels, 1]` from a slice.
    fn conditioning_tensor(&self, conditioning: &[f32]) -> Result<Tensor> {
        let gin = self.config.gin_channels as usize;
        if gin == 0 {
            return Err(VocoderError::VocodingError(
                "Generator was configured with gin_channels = 0 but conditioning was supplied"
                    .to_string(),
            ));
        }
        if conditioning.len() != gin {
            return Err(VocoderError::VocodingError(format!(
                "Conditioning vector has {} values but gin_channels is {gin}",
                conditioning.len()
            )));
        }
        Ok(Tensor::from_slice(conditioning, (1, gin, 1), &self.device)?)
    }

    /// Forward pass over tensors.
    ///
    /// `latent` is `[batch, input_dim, frames]` and `conditioning` (optional) is
    /// `[batch, gin_channels, 1]`. The result is `[batch, 1, samples]`.
    ///
    /// # Errors
    /// Returns [`VocoderError::VocodingError`] on shape mismatches and
    /// [`VocoderError::CandleError`] on tensor failures.
    pub fn generate_tensor(
        &self,
        latent: &Tensor,
        conditioning: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_, channels, _) = latent.dims3()?;
        if channels != self.config.input_dim as usize {
            return Err(VocoderError::VocodingError(format!(
                "Generator expects {} latent channels, got {channels}",
                self.config.input_dim
            )));
        }

        let mut x = self.conv_pre.forward(latent)?;

        if let Some(cond) = conditioning {
            let conv = self.cond.as_ref().ok_or_else(|| {
                VocoderError::VocodingError(
                    "Generator was configured with gin_channels = 0 but conditioning was supplied"
                        .to_string(),
                )
            })?;
            let (_, cond_channels, _) = cond.dims3()?;
            if cond_channels != self.config.gin_channels as usize {
                return Err(VocoderError::VocodingError(format!(
                    "Conditioning has {cond_channels} channels but gin_channels is {}",
                    self.config.gin_channels
                )));
            }
            x = x.broadcast_add(&conv.forward(cond)?)?;
        }

        for block in &self.upsample_blocks {
            x = block.forward(&x)?;
        }

        let x = self
            .activation
            .apply(&x, self.config.leaky_relu_slope as f64)?;
        let x = self.conv_post.forward(&x)?;
        Ok(x.tanh()?)
    }

    /// Number of latent frames of context needed for artifact-free streaming.
    ///
    /// This is the real receptive field of the network, derived from the
    /// configured kernel sizes, dilations and upsampling rates (plus one frame of
    /// margin), expressed in input frames. Feeding at least this many frames of
    /// left *and* right context makes a chunked pass bit-comparable to a
    /// one-shot pass.
    pub fn recommended_context_frames(&self) -> usize {
        // `jump` is the size of one step at the current resolution, measured in
        // input frames; `rf` is the receptive field in input frames.
        let mut rf = 1.0f64;
        let mut jump = 1.0f64;

        // conv_pre: kernel 7, dilation 1.
        rf += 6.0 * jump;

        // Widest MRF residual block, in units of the post-upsampling resolution.
        let widest_block = self
            .config
            .resblock_kernel_sizes
            .iter()
            .zip(self.config.resblock_dilation_sizes.iter())
            .map(|(&k, dilations)| {
                dilations
                    .iter()
                    .map(|&d| ((k - 1) * d + (k - 1)) as f64)
                    .sum::<f64>()
            })
            .fold(0.0f64, f64::max);

        for (&rate, &kernel) in self
            .config
            .upsample_rates
            .iter()
            .zip(self.config.upsample_kernel_sizes.iter())
        {
            // A transposed convolution reads `(k - 1) / stride + 1` taps at the
            // pre-upsampling resolution.
            let taps = ((kernel.saturating_sub(1)) / rate.max(1) + 1) as f64;
            rf += (taps - 1.0) * jump;
            jump /= rate.max(1) as f64;
            rf += widest_block * jump;
        }

        // conv_post: kernel 7, dilation 1.
        rf += 6.0 * jump;

        rf.ceil() as usize + 1
    }

    /// Create a streaming state sized for this generator's receptive field.
    pub fn new_state(&self) -> GeneratorState {
        GeneratorState::with_context_frames(self.recommended_context_frames())
    }

    /// Generate audio for one chunk of a stream.
    ///
    /// This is real streaming, not a per-chunk re-invocation of
    /// [`Vits2Generator::generate`]: the state keeps the trailing
    /// [`GeneratorState::context_frames`] already-emitted frames as **left**
    /// context and holds back the same number of newly received frames as
    /// **right** context. Only frames that have both are emitted; the rest stay
    /// buffered until the next call or until [`Vits2Generator::flush_streaming`].
    ///
    /// With `context_frames >= recommended_context_frames()` the concatenation of
    /// all streamed chunks plus the flush equals the one-shot result.
    ///
    /// Returns an empty buffer while the lookahead is still filling up.
    ///
    /// # Errors
    /// Returns [`VocoderError::VocodingError`] for mis-shaped input and
    /// [`VocoderError::CandleError`] on tensor failures.
    pub fn generate_streaming(
        &self,
        latent_chunk: &[f32],
        conditioning: Option<&[f32]>,
        state: &mut GeneratorState,
    ) -> Result<Vec<f32>> {
        let input_dim = self.config.input_dim as usize;
        if latent_chunk.is_empty() {
            return Err(VocoderError::VocodingError(
                "Empty latent chunk".to_string(),
            ));
        }
        if !latent_chunk.len().is_multiple_of(input_dim) {
            return Err(VocoderError::VocodingError(format!(
                "Latent chunk length {} is not a multiple of the generator input dimension \
                 {input_dim}",
                latent_chunk.len()
            )));
        }

        let chunk_frames = latent_chunk.len() / input_dim;
        state.frame_count += chunk_frames as u64;

        let history_frames = state.latent_context.len() / input_dim;
        let pending_frames = state.pending_latent.len() / input_dim;
        let available = pending_frames + chunk_frames;
        let emit_frames = available.saturating_sub(state.context_frames);

        if emit_frames == 0 {
            // Not enough right context yet: buffer and emit nothing.
            state.pending_latent = concat_frames(
                &[
                    (state.pending_latent.as_slice(), pending_frames),
                    (latent_chunk, chunk_frames),
                ],
                input_dim,
            );
            return Ok(Vec::new());
        }

        let combined = concat_frames(
            &[
                (state.latent_context.as_slice(), history_frames),
                (state.pending_latent.as_slice(), pending_frames),
                (latent_chunk, chunk_frames),
            ],
            input_dim,
        );
        let total_frames = history_frames + available;

        let audio = self.generate(&combined, conditioning)?;
        let upsample = self.total_upsampling_factor() as usize;
        let start = (history_frames * upsample).min(audio.len());
        let end = ((history_frames + emit_frames) * upsample).min(audio.len());
        let output = audio[start..end].to_vec();

        // Everything up to `history_frames + emit_frames` has been emitted; keep
        // its tail as left context and the remainder as the lookahead buffer.
        let emitted_end = history_frames + emit_frames;
        let keep_history = state.context_frames.min(emitted_end);
        state.latent_context = slice_frames(
            &combined,
            total_frames,
            input_dim,
            emitted_end - keep_history,
            keep_history,
        );
        state.pending_latent = slice_frames(
            &combined,
            total_frames,
            input_dim,
            emitted_end,
            total_frames - emitted_end,
        );

        state.update_overlap(&output, state.overlap_size);
        Ok(output)
    }

    /// Emit the audio for the frames still held back as lookahead.
    ///
    /// Call once the latent stream has ended. Returns an empty buffer when
    /// nothing is buffered.
    ///
    /// # Errors
    /// Returns [`VocoderError::VocodingError`] for inconsistent state and
    /// [`VocoderError::CandleError`] on tensor failures.
    pub fn flush_streaming(
        &self,
        conditioning: Option<&[f32]>,
        state: &mut GeneratorState,
    ) -> Result<Vec<f32>> {
        let input_dim = self.config.input_dim as usize;
        let pending_frames = state.pending_latent.len() / input_dim;
        if pending_frames == 0 {
            return Ok(Vec::new());
        }
        let history_frames = state.latent_context.len() / input_dim;

        let combined = concat_frames(
            &[
                (state.latent_context.as_slice(), history_frames),
                (state.pending_latent.as_slice(), pending_frames),
            ],
            input_dim,
        );
        let total_frames = history_frames + pending_frames;

        let audio = self.generate(&combined, conditioning)?;
        let upsample = self.total_upsampling_factor() as usize;
        let start = (history_frames * upsample).min(audio.len());
        let output = audio[start..].to_vec();

        let keep_history = state.context_frames.min(total_frames);
        state.latent_context = slice_frames(
            &combined,
            total_frames,
            input_dim,
            total_frames - keep_history,
            keep_history,
        );
        state.pending_latent.clear();
        state.update_overlap(&output, state.overlap_size);
        Ok(output)
    }

    /// Load generator weights from a SafeTensors checkpoint.
    ///
    /// Accepted names follow the reference HiFi-GAN / VITS `state_dict` layout,
    /// optionally wrapped in container prefixes (`module.`, `net_g.`, `dec.`,
    /// ...). PyTorch `weight_norm` parameter pairs (`weight_g` / `weight_v`) are
    /// fused into plain weights.
    ///
    /// Fail-closed: the checkpoint must supply **every** generator parameter.
    /// A partial checkpoint is rejected rather than leaving some layers at their
    /// pseudo-random initialization while `is_pretrained()` reports success; use
    /// [`Vits2Generator::load_weights_partial`] when a partial load is intended.
    ///
    /// # Errors
    /// Returns [`VocoderError::ModelError`] when the file cannot be read/parsed,
    /// when no tensor matched a model parameter, or when the checkpoint left any
    /// parameter unset.
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<WeightLoadReport> {
        let report = load_safetensors_into_varmap_with_mode(
            &mut self.varmap,
            path.as_ref(),
            &self.device,
            map_generator_weight_name,
            LoadMode::Strict,
        )?;
        self.weights_loaded = true;
        Ok(report)
    }

    /// Load whatever the checkpoint supplies, tolerating uncovered parameters.
    ///
    /// Parameters the checkpoint does not cover keep their pseudo-random
    /// initialization, so the generator is **not** marked pretrained and
    /// [`Vits2Generator::generate_pretrained`] keeps failing closed. Inspect
    /// [`WeightLoadReport::missing_parameters`] to see what is still missing.
    ///
    /// # Errors
    /// Returns [`VocoderError::ModelError`] when the file cannot be read/parsed
    /// or when no tensor matched a model parameter.
    pub fn load_weights_partial<P: AsRef<Path>>(&mut self, path: P) -> Result<WeightLoadReport> {
        let report = load_safetensors_into_varmap_with_mode(
            &mut self.varmap,
            path.as_ref(),
            &self.device,
            map_generator_weight_name,
            LoadMode::Partial,
        )?;
        if report.is_complete() {
            self.weights_loaded = true;
        }
        Ok(report)
    }

    /// Save the current weights to a SafeTensors file.
    ///
    /// # Errors
    /// Returns [`VocoderError::CandleError`] when serialization fails.
    pub fn save_weights<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.varmap.save(path.as_ref())?;
        Ok(())
    }

    /// Whether a checkpoint has been loaded into this generator.
    pub fn is_pretrained(&self) -> bool {
        self.weights_loaded
    }

    /// Upsampling blocks of this generator.
    pub fn upsample_blocks(&self) -> &[UpsampleBlock] {
        &self.upsample_blocks
    }

    /// Compute device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Total number of scalars actually held by this generator.
    ///
    /// # Errors
    /// Returns [`VocoderError::ModelError`] if the parameter store is poisoned.
    pub fn parameter_count(&self) -> Result<u64> {
        count_parameters(&self.varmap)
    }

    /// Total number of parameters, or `0` if the parameter store is unavailable.
    ///
    /// Prefer [`Vits2Generator::parameter_count`] which surfaces errors.
    pub fn num_parameters(&self) -> u64 {
        self.parameter_count().unwrap_or(0)
    }

    /// Total upsampling factor (samples produced per input frame).
    pub fn total_upsampling_factor(&self) -> u32 {
        self.config.upsample_rates.iter().product()
    }

    /// Memory required by the parameters, in MB.
    pub fn memory_requirements_mb(&self) -> f32 {
        self.num_parameters() as f32 * 4.0 / (1024.0 * 1024.0)
    }
}

/// Map a checkpoint tensor name onto an internal generator parameter name.
///
/// Recognized layouts (after stripping container prefixes such as `module.`,
/// `net_g.` or `dec.`):
///
/// * `conv_pre.*`, `conv_post.*`, `cond.*`
/// * `ups.{i}.*`
/// * `resblocks.{k}.convs1.{d}.*`, `resblocks.{k}.convs2.{d}.*`
/// * the aliases `input_conv.*` / `output_conv.*` used by some exporters
///
/// Anything else returns `None` and is reported as unmapped rather than being
/// silently forced into an unrelated parameter.
fn map_generator_weight_name(name: &str) -> Option<String> {
    let stripped = strip_checkpoint_prefixes(name);

    let renamed = if let Some(rest) = stripped.strip_prefix("input_conv.") {
        format!("conv_pre.{rest}")
    } else if let Some(rest) = stripped.strip_prefix("output_conv.") {
        format!("conv_post.{rest}")
    } else {
        stripped.to_string()
    };

    const KNOWN_ROOTS: [&str; 3] = ["conv_pre.", "conv_post.", "cond."];
    if KNOWN_ROOTS.iter().any(|root| renamed.starts_with(root))
        || split_indexed_prefix(&renamed, "ups").is_some()
        || split_indexed_prefix(&renamed, "resblocks").is_some()
    {
        Some(renamed)
    } else {
        None
    }
}

/// Concatenate several channel-major latent buffers along the time axis.
///
/// Each part is `(buffer, frames)` where `buffer` holds `input_dim * frames`
/// values laid out channel by channel.
fn concat_frames(parts: &[(&[f32], usize)], input_dim: usize) -> Vec<f32> {
    let total: usize = parts.iter().map(|(_, frames)| *frames).sum();
    if total == 0 || input_dim == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0f32; input_dim * total];
    for channel in 0..input_dim {
        let dst = &mut out[channel * total..(channel + 1) * total];
        let mut offset = 0usize;
        for (buffer, frames) in parts {
            if *frames == 0 {
                continue;
            }
            let src = &buffer[channel * frames..(channel + 1) * frames];
            dst[offset..offset + frames].copy_from_slice(src);
            offset += frames;
        }
    }
    out
}

/// Extract `count` frames starting at `start` from a channel-major buffer.
fn slice_frames(
    buffer: &[f32],
    total_frames: usize,
    input_dim: usize,
    start: usize,
    count: usize,
) -> Vec<f32> {
    if count == 0 || total_frames == 0 || input_dim == 0 || start + count > total_frames {
        return Vec::new();
    }
    let mut out = vec![0.0f32; input_dim * count];
    for channel in 0..input_dim {
        let src = &buffer[channel * total_frames..(channel + 1) * total_frames];
        out[channel * count..(channel + 1) * count].copy_from_slice(&src[start..start + count]);
    }
    out
}

/// Generator state for streaming synthesis
#[derive(Debug, Clone)]
pub struct GeneratorState {
    /// Previous audio samples, kept for overlap-add by downstream code
    pub overlap_buffer: Vec<f32>,
    /// Number of samples retained in [`Self::overlap_buffer`]
    pub overlap_size: usize,
    /// Trailing already-emitted latent frames, used as left context
    pub latent_context: Vec<f32>,
    /// Received-but-not-yet-emitted latent frames, kept as right context
    pub pending_latent: Vec<f32>,
    /// Number of latent frames used as left and right context
    pub context_frames: usize,
    /// Auxiliary hidden states keyed by name
    pub hidden_states: HashMap<String, Vec<f32>>,
    /// Number of latent frames received so far
    pub frame_count: u64,
}

impl Default for GeneratorState {
    fn default() -> Self {
        Self {
            overlap_buffer: Vec::new(),
            overlap_size: 0,
            latent_context: Vec::new(),
            pending_latent: Vec::new(),
            // Covers the receptive field of the default HiFi-GAN configuration.
            // Prefer `Vits2Generator::new_state()`, which computes the exact
            // value from the actual configuration.
            context_frames: 32,
            hidden_states: HashMap::new(),
            frame_count: 0,
        }
    }
}

impl GeneratorState {
    /// Create a state with an explicit context size, in latent frames.
    pub fn with_context_frames(context_frames: usize) -> Self {
        Self {
            context_frames,
            ..Default::default()
        }
    }

    /// Reset state for a new sequence
    pub fn reset(&mut self) {
        self.overlap_buffer.clear();
        self.latent_context.clear();
        self.pending_latent.clear();
        self.hidden_states.clear();
        self.frame_count = 0;
    }

    /// Update the overlap buffer with the tail of `new_samples`
    pub fn update_overlap(&mut self, new_samples: &[f32], overlap_size: usize) {
        self.overlap_size = overlap_size;
        let start = new_samples.len().saturating_sub(overlap_size);
        self.overlap_buffer = new_samples[start..].to_vec();
    }
}

impl GeneratorConfig {
    /// Validate generator configuration
    ///
    /// # Errors
    /// Returns [`VocoderError::ModelError`] describing the first violated
    /// constraint.
    pub fn validate(&self) -> Result<()> {
        if self.input_dim == 0 {
            return Err(VocoderError::ModelError(
                "Input dimension must be greater than 0".to_string(),
            ));
        }

        if self.hidden_channels == 0 {
            return Err(VocoderError::ModelError(
                "Hidden channels must be greater than 0".to_string(),
            ));
        }

        if self.initial_channel == 0 {
            return Err(VocoderError::ModelError(
                "Initial channel count must be greater than 0".to_string(),
            ));
        }

        if self.upsample_rates.is_empty() {
            return Err(VocoderError::ModelError(
                "Upsample rates cannot be empty".to_string(),
            ));
        }

        if self.upsample_rates.len() != self.upsample_kernel_sizes.len() {
            return Err(VocoderError::ModelError(
                "Upsample rates and kernel sizes must have the same length".to_string(),
            ));
        }

        for (i, (&rate, &kernel)) in self
            .upsample_rates
            .iter()
            .zip(self.upsample_kernel_sizes.iter())
            .enumerate()
        {
            if rate == 0 {
                return Err(VocoderError::ModelError(format!(
                    "Upsample rate {i} must be greater than 0"
                )));
            }
            if kernel < rate || !(kernel - rate).is_multiple_of(2) {
                return Err(VocoderError::ModelError(format!(
                    "Upsample kernel size {kernel} at index {i} must be >= the rate {rate} and \
                     differ from it by an even amount"
                )));
            }
        }

        if self.resblock_kernel_sizes.is_empty() {
            return Err(VocoderError::ModelError(
                "Residual block kernel sizes cannot be empty".to_string(),
            ));
        }

        if self.resblock_kernel_sizes.len() != self.resblock_dilation_sizes.len() {
            return Err(VocoderError::ModelError(
                "Residual block kernel sizes and dilation sizes must have the same length"
                    .to_string(),
            ));
        }

        for (i, &kernel) in self.resblock_kernel_sizes.iter().enumerate() {
            if kernel == 0 || kernel.is_multiple_of(2) {
                return Err(VocoderError::ModelError(format!(
                    "Residual block kernel size {i} must be odd (got {kernel})"
                )));
            }
        }

        for (i, dilation_group) in self.resblock_dilation_sizes.iter().enumerate() {
            if dilation_group.is_empty() {
                return Err(VocoderError::ModelError(format!(
                    "Dilation group {i} cannot be empty"
                )));
            }
            if dilation_group.contains(&0) {
                return Err(VocoderError::ModelError(format!(
                    "Dilation group {i} must contain only positive dilations"
                )));
            }
        }

        // Reject rather than silently ignore options the generator cannot honor.
        GeneratorActivation::parse(&self.activation)?;
        if self.use_spectral_norm {
            return Err(VocoderError::ModelError(
                "Spectral normalization is a discriminator-side training technique and is not \
                 implemented for the VITS2 generator; set use_spectral_norm = false"
                    .to_string(),
            ));
        }

        Ok(())
    }

    /// Exact number of parameters the corresponding [`Vits2Generator`] allocates.
    ///
    /// Mirrors the layer shapes built by [`Vits2Generator::new_seeded`], so it
    /// equals `Vits2Generator::parameter_count()` for the same configuration.
    pub fn parameter_count(&self) -> u64 {
        let initial = self.initial_channel as u64;
        let mut total = 0u64;

        // conv_pre: Conv1d(input_dim -> initial, kernel 7)
        total += initial * self.input_dim as u64 * 7 + initial;

        // cond: Conv1d(gin_channels -> initial, kernel 1)
        if self.gin_channels > 0 {
            total += initial * self.gin_channels as u64 + initial;
        }

        let mut current = initial;
        for (i, (&_rate, &kernel)) in self
            .upsample_rates
            .iter()
            .zip(self.upsample_kernel_sizes.iter())
            .enumerate()
        {
            let out_channels = (self.initial_channel >> (i + 1)).max(1) as u64;

            // ConvTranspose1d weight is [in, out, kernel] plus an out-sized bias.
            total += current * out_channels * kernel as u64 + out_channels;

            for (&res_kernel, dilations) in self
                .resblock_kernel_sizes
                .iter()
                .zip(self.resblock_dilation_sizes.iter())
            {
                let per_conv = out_channels * out_channels * res_kernel as u64 + out_channels;
                // convs1 and convs2, one pair per dilation.
                total += 2 * dilations.len() as u64 * per_conv;
            }

            current = out_channels;
        }

        // conv_post: Conv1d(current -> 1, kernel 7)
        total += current * 7 + 1;

        total
    }

    /// Create high-quality configuration
    pub fn high_quality() -> Self {
        Self {
            input_dim: 256,
            hidden_channels: 768,
            initial_channel: 768,
            upsample_rates: vec![8, 8, 2, 2],
            upsample_kernel_sizes: vec![16, 16, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            gin_channels: 512,
            use_weight_norm: true,
            use_spectral_norm: false,
            activation: "LeakyReLU".to_string(),
            leaky_relu_slope: 0.1,
        }
    }

    /// Create fast configuration
    pub fn fast() -> Self {
        Self {
            input_dim: 128,
            hidden_channels: 256,
            initial_channel: 256,
            upsample_rates: vec![8, 8, 4],
            upsample_kernel_sizes: vec![16, 16, 8],
            resblock_kernel_sizes: vec![3, 7],
            resblock_dilation_sizes: vec![vec![1, 3], vec![1, 3]],
            gin_channels: 128,
            use_weight_norm: true,
            use_spectral_norm: false,
            activation: "LeakyReLU".to_string(),
            leaky_relu_slope: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small but structurally complete configuration used to keep tests fast.
    fn tiny_config() -> GeneratorConfig {
        GeneratorConfig {
            input_dim: 8,
            hidden_channels: 16,
            initial_channel: 16,
            upsample_rates: vec![4, 2],
            upsample_kernel_sizes: vec![8, 4],
            resblock_kernel_sizes: vec![3, 5],
            resblock_dilation_sizes: vec![vec![1, 3], vec![1, 3]],
            gin_channels: 4,
            use_weight_norm: true,
            use_spectral_norm: false,
            activation: "LeakyReLU".to_string(),
            leaky_relu_slope: 0.1,
        }
    }

    fn ramp(len: usize, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.37 + offset).sin()) * 0.5)
            .collect()
    }

    #[test]
    fn test_generator_config_validation() {
        let config = GeneratorConfig::default();
        assert!(config.validate().is_ok());
        assert!(GeneratorConfig::high_quality().validate().is_ok());
        assert!(GeneratorConfig::fast().validate().is_ok());

        let mut invalid_config = config.clone();
        invalid_config.input_dim = 0;
        assert!(invalid_config.validate().is_err());

        let mut invalid_config = config.clone();
        invalid_config.upsample_kernel_sizes.pop();
        assert!(invalid_config.validate().is_err());

        // Even residual kernel sizes cannot preserve the time axis.
        let mut invalid_config = config.clone();
        invalid_config.resblock_kernel_sizes = vec![4, 7, 11];
        assert!(invalid_config.validate().is_err());

        // A kernel smaller than the stride cannot upsample exactly.
        let mut invalid_config = config.clone();
        invalid_config.upsample_kernel_sizes = vec![4, 16, 4, 4];
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_generator_output_length_and_input_sensitivity() {
        let generator = Vits2Generator::new(tiny_config()).expect("generator");
        let frames = 6;
        let latent = ramp(8 * frames, 0.0);

        let audio = generator.generate(&latent, None).expect("generate");
        assert_eq!(audio.len(), frames * 4 * 2);
        assert!(audio.iter().all(|s| s.is_finite() && s.abs() <= 1.0));

        let other_latent = ramp(8 * frames, 1.7);
        let other_audio = generator.generate(&other_latent, None).expect("generate");
        assert_ne!(
            audio, other_audio,
            "generator output must depend on the latent input"
        );

        // Non-trivial: a scalar pipeline would give a constant output for a
        // constant input; a real conv stack has per-position bias/padding effects.
        let constant = vec![0.25f32; 8 * frames];
        let constant_audio = generator.generate(&constant, None).expect("generate");
        assert_eq!(constant_audio.len(), audio.len());
        assert!(constant_audio.iter().any(|&s| s != constant_audio[0]));
    }

    #[test]
    fn test_generator_is_deterministic_for_a_given_seed() {
        let a = Vits2Generator::new_seeded(tiny_config(), Device::Cpu, 11).expect("a");
        let b = Vits2Generator::new_seeded(tiny_config(), Device::Cpu, 11).expect("b");
        let c = Vits2Generator::new_seeded(tiny_config(), Device::Cpu, 12).expect("c");

        let latent = ramp(8 * 5, 0.3);
        let out_a = a.generate(&latent, None).expect("a");
        let out_b = b.generate(&latent, None).expect("b");
        let out_c = c.generate(&latent, None).expect("c");

        assert_eq!(out_a, out_b, "same seed must give identical models");
        assert_ne!(out_a, out_c, "different seeds must give different models");
    }

    #[test]
    fn test_generator_conditioning_changes_output() {
        let generator = Vits2Generator::new(tiny_config()).expect("generator");
        let latent = ramp(8 * 5, 0.9);

        let plain = generator.generate(&latent, None).expect("plain");
        let cond_a = generator
            .generate(&latent, Some(&[0.5, -0.25, 0.75, 0.1]))
            .expect("cond a");
        let cond_b = generator
            .generate(&latent, Some(&[-0.5, 0.25, -0.75, -0.1]))
            .expect("cond b");

        assert_ne!(plain, cond_a);
        assert_ne!(cond_a, cond_b);

        // Wrong conditioning width must be rejected, never silently ignored.
        assert!(generator.generate(&latent, Some(&[0.5])).is_err());
    }

    #[test]
    fn test_generator_rejects_bad_latent_shape() {
        let generator = Vits2Generator::new(tiny_config()).expect("generator");
        assert!(generator.generate(&[], None).is_err());
        // 8 channels expected; 12 is not a multiple of 8.
        assert!(generator.generate(&vec![0.1; 12], None).is_err());
    }

    #[test]
    fn test_config_rejects_unsupported_options() {
        let mut config = tiny_config();
        config.activation = "mystery".to_string();
        assert!(config.validate().is_err());
        assert!(Vits2Generator::new(config).is_err());

        let mut config = tiny_config();
        config.use_spectral_norm = true;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_activation_is_honored() {
        let mut leaky = tiny_config();
        leaky.activation = "LeakyReLU".to_string();
        let mut gelu = tiny_config();
        gelu.activation = "GELU".to_string();

        let a = Vits2Generator::new_seeded(leaky, Device::Cpu, 2024).expect("leaky");
        let b = Vits2Generator::new_seeded(gelu, Device::Cpu, 2024).expect("gelu");

        let latent = ramp(8 * 5, 0.55);
        assert_ne!(
            a.generate(&latent, None).expect("a"),
            b.generate(&latent, None).expect("b"),
            "the configured activation must affect the output"
        );
    }

    #[test]
    fn test_analytic_parameter_count_matches_allocation() {
        for config in [tiny_config(), GeneratorConfig::fast()] {
            let analytic = config.parameter_count();
            let generator = Vits2Generator::new(config).expect("generator");
            assert_eq!(analytic, generator.parameter_count().expect("count"));
        }
    }

    #[test]
    fn test_num_parameters_matches_layer_sum() {
        let generator = Vits2Generator::new(tiny_config()).expect("generator");
        let reported = generator.parameter_count().expect("count");
        let from_layers: u64 = conv1d_parameters(&generator.conv_pre)
            + generator.cond.as_ref().map_or(0, conv1d_parameters)
            + generator
                .upsample_blocks()
                .iter()
                .map(|b| b.num_parameters())
                .sum::<u64>()
            + conv1d_parameters(&generator.conv_post);
        assert_eq!(reported, from_layers);
        assert!(reported > 0);
        assert!(generator.memory_requirements_mb() > 0.0);
        assert_eq!(generator.total_upsampling_factor(), 8);
    }

    #[test]
    fn test_weight_round_trip_changes_output() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("vits2_generator.safetensors");

        let trained =
            Vits2Generator::new_seeded(tiny_config(), Device::Cpu, 5150).expect("trained");
        trained.save_weights(&path).expect("save");

        let mut target = Vits2Generator::new_seeded(tiny_config(), Device::Cpu, 1).expect("target");
        assert!(!target.is_pretrained());

        let latent = ramp(8 * 5, 0.42);
        let before = target.generate(&latent, None).expect("before");
        assert!(target.generate_pretrained(&latent, None).is_err());

        let report = target.load_weights(&path).expect("load");
        assert!(report.loaded > 0);
        assert_eq!(report.shape_mismatches, 0);
        assert!(target.is_pretrained());

        let after = target.generate(&latent, None).expect("after");
        assert_ne!(before, after, "loading weights must change the output");
        let expected = trained.generate(&latent, None).expect("expected");
        for (a, b) in after.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "loaded model must reproduce source");
        }
        assert!(target.generate_pretrained(&latent, None).is_ok());
    }

    #[test]
    fn test_load_weights_rejects_partial_checkpoint() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("partial.safetensors");

        // A checkpoint holding only conv_pre: every other layer would keep its
        // pseudo-random initialization.
        let device = Device::Cpu;
        let source = Vits2Generator::new_seeded(tiny_config(), device.clone(), 31).expect("src");
        let full = dir.path().join("full.safetensors");
        source.save_weights(&full).expect("save");

        let data = std::fs::read(&full).expect("read");
        let st = safetensors::SafeTensors::deserialize(&data).expect("parse");
        let subset: Vec<(String, Vec<usize>, Vec<u8>)> = st
            .tensors()
            .into_iter()
            .filter(|(name, _)| name.starts_with("conv_pre."))
            .map(|(name, view)| (name, view.shape().to_vec(), view.data().to_vec()))
            .collect();
        assert!(!subset.is_empty());

        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        for (name, shape, bytes) in &subset {
            let values: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let tensor = Tensor::from_vec(values, shape.clone(), &device).expect("tensor");
            let _ = vb
                .get_with_hints(shape.clone(), name, candle_nn::Init::Const(0.0))
                .expect("register");
            varmap.set_one(name, &tensor).expect("set");
        }
        varmap.save(&path).expect("save subset");

        let mut target = Vits2Generator::new_seeded(tiny_config(), device, 1).expect("target");
        let err = target
            .load_weights(&path)
            .expect_err("a partial checkpoint must be rejected");
        assert!(err.to_string().contains("Incomplete VITS2 checkpoint"));
        assert!(!target.is_pretrained());

        // Partial mode reports the gap and still refuses pretrained inference.
        let report = target.load_weights_partial(&path).expect("partial load");
        assert!(report.loaded > 0);
        assert!(report.missing_parameters > 0);
        assert!(!report.is_complete());
        assert!(!target.is_pretrained());
        let latent = ramp(8 * 4, 0.1);
        assert!(target.generate_pretrained(&latent, None).is_err());
    }

    #[test]
    fn test_load_weights_fails_closed_for_unrelated_checkpoint() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("unrelated.safetensors");

        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _ = candle_nn::linear(3, 3, vb.pp("totally_unrelated")).expect("linear");
        varmap.save(&path).expect("save");

        let mut generator = Vits2Generator::new(tiny_config()).expect("generator");
        let err = generator.load_weights(&path).expect_err("must fail closed");
        assert!(err.to_string().contains("No VITS2 weights"));
        assert!(!generator.is_pretrained());
    }

    #[test]
    fn test_map_generator_weight_name() {
        assert_eq!(
            map_generator_weight_name("module.net_g.dec.ups.0.weight").as_deref(),
            Some("ups.0.weight")
        );
        assert_eq!(
            map_generator_weight_name("resblocks.4.convs1.2.bias").as_deref(),
            Some("resblocks.4.convs1.2.bias")
        );
        assert_eq!(
            map_generator_weight_name("input_conv.weight").as_deref(),
            Some("conv_pre.weight")
        );
        assert_eq!(map_generator_weight_name("enc_p.emb.weight"), None);
    }

    #[test]
    fn test_streaming_matches_one_shot() {
        let generator = Vits2Generator::new(tiny_config()).expect("generator");
        let input_dim = 8usize;
        let total_frames = 96usize;
        let latent = ramp(input_dim * total_frames, 0.15);

        let one_shot = generator.generate(&latent, None).expect("one shot");

        let mut state = generator.new_state();
        assert!(state.context_frames >= generator.recommended_context_frames());

        let chunk_frames = 16usize;
        let mut streamed = Vec::new();
        for start in (0..total_frames).step_by(chunk_frames) {
            let end = (start + chunk_frames).min(total_frames);
            let mut chunk = Vec::with_capacity(input_dim * (end - start));
            for channel in 0..input_dim {
                let row = &latent[channel * total_frames..(channel + 1) * total_frames];
                chunk.extend_from_slice(&row[start..end]);
            }
            streamed.extend(
                generator
                    .generate_streaming(&chunk, None, &mut state)
                    .expect("chunk"),
            );
        }
        streamed.extend(generator.flush_streaming(None, &mut state).expect("flush"));

        assert_eq!(streamed.len(), one_shot.len());
        assert_eq!(state.frame_count, total_frames as u64);
        assert!(state.pending_latent.is_empty());

        // With full left and right context the chunked pass must reproduce the
        // one-shot pass; a naive per-chunk call cannot.
        let max_err = streamed
            .iter()
            .zip(one_shot.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-4,
            "streaming with context must match one-shot synthesis (max err {max_err})"
        );

        // Sanity: the streamed audio is not trivially constant.
        assert!(streamed.iter().any(|&s| (s - streamed[0]).abs() > 1e-6));
    }

    #[test]
    fn test_streaming_without_context_is_worse_than_with_context() {
        let generator = Vits2Generator::new(tiny_config()).expect("generator");
        let input_dim = 8usize;
        let total_frames = 64usize;
        let chunk_frames = 16usize;
        let latent = ramp(input_dim * total_frames, 0.6);
        let one_shot = generator.generate(&latent, None).expect("one shot");

        let run = |context: usize| -> Vec<f32> {
            let mut state = GeneratorState::with_context_frames(context);
            let mut out = Vec::new();
            for start in (0..total_frames).step_by(chunk_frames) {
                let end = (start + chunk_frames).min(total_frames);
                let mut chunk = Vec::with_capacity(input_dim * (end - start));
                for channel in 0..input_dim {
                    let row = &latent[channel * total_frames..(channel + 1) * total_frames];
                    chunk.extend_from_slice(&row[start..end]);
                }
                out.extend(
                    generator
                        .generate_streaming(&chunk, None, &mut state)
                        .expect("chunk"),
                );
            }
            out.extend(generator.flush_streaming(None, &mut state).expect("flush"));
            out
        };

        let err = |audio: &[f32]| -> f32 {
            audio
                .iter()
                .zip(one_shot.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        };

        let no_context = run(0);
        let with_context = run(generator.recommended_context_frames());
        assert_eq!(no_context.len(), one_shot.len());
        assert_eq!(with_context.len(), one_shot.len());
        assert!(
            err(&no_context) > err(&with_context),
            "context must reduce the streaming boundary error"
        );
    }

    #[test]
    fn test_generator_state() {
        let mut state = GeneratorState::default();
        state.update_overlap(&[0.1, 0.2, 0.3, 0.4], 2);
        assert_eq!(state.overlap_buffer, vec![0.3, 0.4]);

        state.latent_context = vec![1.0, 2.0];
        state.frame_count = 7;
        state.reset();
        assert!(state.overlap_buffer.is_empty());
        assert!(state.latent_context.is_empty());
        assert_eq!(state.frame_count, 0);
    }

    #[test]
    fn test_default_config_generator_builds_and_runs() {
        let generator = Vits2Generator::new(GeneratorConfig::default()).expect("generator");
        assert_eq!(generator.total_upsampling_factor(), 256);
        assert!(generator.parameter_count().expect("count") > 1_000_000);

        let frames = 2;
        let latent = ramp(192 * frames, 0.0);
        let audio = generator.generate(&latent, None).expect("generate");
        assert_eq!(audio.len(), frames * 256);
        assert!(audio.iter().all(|s| s.is_finite()));
    }
}
