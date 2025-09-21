//! VITS Duration Predictor implementation
//!
//! Predicts phoneme durations for alignment in the VITS model using
//! convolutional neural networks and differentiable duration modeling.

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{layer_norm, Conv1d, Conv1dConfig, Module, VarBuilder};
use serde::{Deserialize, Serialize};

use crate::{AcousticError, Phoneme, Result};

/// Configuration for duration predictor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurationConfig {
    /// Input dimension (from text encoder)
    pub input_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of CNN layers
    pub n_layers: usize,
    /// Kernel size for convolutions
    pub kernel_size: usize,
    /// Dropout probability
    pub dropout: f64,
    /// Filter channels for convolutional layers
    pub filter_channels: usize,
}

impl Default for DurationConfig {
    fn default() -> Self {
        Self {
            input_dim: 192,
            hidden_dim: 256,
            n_layers: 2,
            kernel_size: 3,
            dropout: 0.5,
            filter_channels: 256,
        }
    }
}

/// Convolutional block with residual connections
pub struct ConvBlock {
    conv: Conv1d,
    norm: candle_nn::LayerNorm,
    dropout: f64,
}

impl ConvBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dropout: f64,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let conv_config = Conv1dConfig {
            padding: kernel_size / 2,
            stride: 1,
            ..Default::default()
        };

        let conv = candle_nn::conv1d(
            in_channels,
            out_channels,
            kernel_size,
            conv_config,
            vb.pp("conv"),
        )?;

        let norm = layer_norm(out_channels, 1e-5, vb.pp("norm"))?;

        Ok(Self {
            conv,
            norm,
            dropout,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let mut h = self.conv.forward(x)?;
        h = self.norm.forward(&h)?;
        h = h.relu()?;

        // Apply dropout
        if self.dropout > 0.0 {
            h = candle_nn::ops::dropout(&h, self.dropout as f32)?;
        }

        Ok(h)
    }
}

/// Residual convolutional block
pub struct ResidualConvBlock {
    conv_block: ConvBlock,
    projection: Option<Conv1d>,
}

impl ResidualConvBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dropout: f64,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let conv_block = ConvBlock::new(
            in_channels,
            out_channels,
            kernel_size,
            dropout,
            vb.pp("conv_block"),
        )?;

        // Projection layer if dimensions don't match
        let projection = if in_channels != out_channels {
            Some(candle_nn::conv1d(
                in_channels,
                out_channels,
                1,
                Default::default(),
                vb.pp("projection"),
            )?)
        } else {
            None
        };

        Ok(Self {
            conv_block,
            projection,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let residual = if let Some(ref proj) = self.projection {
            proj.forward(x)?
        } else {
            x.clone()
        };

        let h = self.conv_block.forward(x)?;
        let output = (&residual + &h)?;

        Ok(output)
    }
}

/// VITS Duration Predictor
pub struct DurationPredictor {
    config: DurationConfig,
    device: Device,

    // Network layers
    input_conv: Conv1d,
    conv_blocks: Vec<ResidualConvBlock>,
    output_conv: Conv1d,
}

impl DurationPredictor {
    pub fn new(config: DurationConfig, device: Device) -> Result<Self> {
        let vs = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&vs, DType::F32, &device);

        Self::load_with_varbuilder(config, device, vb)
    }

    pub fn load_with_varbuilder(
        config: DurationConfig,
        device: Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Input convolution to project from text encoder dimension to hidden dimension
        let input_conv = candle_nn::conv1d(
            config.input_dim,
            config.filter_channels,
            config.kernel_size,
            Conv1dConfig {
                padding: config.kernel_size / 2,
                stride: 1,
                ..Default::default()
            },
            vb.pp("input_conv"),
        )
        .map_err(|e| AcousticError::ModelError(format!("Failed to create input_conv: {e}")))?;

        // Convolutional blocks
        let mut conv_blocks = Vec::new();
        for i in 0..config.n_layers {
            let block = ResidualConvBlock::new(
                config.filter_channels,
                config.filter_channels,
                config.kernel_size,
                config.dropout,
                vb.pp(format!("conv_block_{i}")),
            )
            .map_err(|e| {
                AcousticError::ModelError(format!("Failed to create conv block {i}: {e}"))
            })?;

            conv_blocks.push(block);
        }

        // Output convolution to predict durations (1 channel for duration)
        let output_conv = candle_nn::conv1d(
            config.filter_channels,
            1, // Single output channel for duration
            1, // 1x1 convolution
            Default::default(),
            vb.pp("output_conv"),
        )
        .map_err(|e| AcousticError::ModelError(format!("Failed to create output_conv: {e}")))?;

        Ok(Self {
            config,
            device,
            input_conv,
            conv_blocks,
            output_conv,
        })
    }

    /// Predict log-durations from text encoder outputs
    ///
    /// # Arguments
    /// * `text_encoding` - Text encoder outputs with shape [batch_size, input_dim, seq_len]
    ///
    /// # Returns
    /// * Log-duration predictions with shape [batch_size, 1, seq_len]
    pub fn forward(&self, text_encoding: &Tensor) -> Result<Tensor> {
        // Validate input shape
        let input_shape = text_encoding.dims();
        if input_shape.len() != 3 {
            return Err(AcousticError::InputError(format!(
                "Expected 3D tensor [batch, input_dim, seq_len], got {input_shape:?}"
            )));
        }

        let (batch_size, input_dim, seq_len) = text_encoding.dims3().map_err(|e| {
            AcousticError::ModelError(format!("Failed to get tensor dimensions: {e}"))
        })?;

        if input_dim != self.config.input_dim {
            return Err(AcousticError::InputError(format!(
                "Expected {} input dimensions, got {}",
                self.config.input_dim, input_dim
            )));
        }

        tracing::debug!(
            "DurationPredictor forward: input shape [{}, {}, {}]",
            batch_size,
            input_dim,
            seq_len
        );

        // Input convolution
        let mut h = self
            .input_conv
            .forward(text_encoding)
            .map_err(|e| AcousticError::ModelError(format!("Input convolution failed: {e}")))?;

        tracing::debug!("After input_conv: {:?}", h.dims());

        // Apply convolutional blocks
        for (i, block) in self.conv_blocks.iter().enumerate() {
            h = block
                .forward(&h)
                .map_err(|e| AcousticError::ModelError(format!("Conv block {i} failed: {e}")))?;
        }

        tracing::debug!("After conv blocks: {:?}", h.dims());

        // Output convolution to predict log-durations
        let log_durations = self
            .output_conv
            .forward(&h)
            .map_err(|e| AcousticError::ModelError(format!("Output convolution failed: {e}")))?;

        tracing::debug!("Log durations shape: {:?}", log_durations.dims());

        Ok(log_durations)
    }

    /// Predict phoneme durations from text encoding
    ///
    /// # Arguments
    /// * `text_encoding` - Text encoder outputs
    /// * `inference` - Whether this is inference (applies noise reduction)
    ///
    /// # Returns
    /// * Duration predictions in frames
    pub fn predict_durations(&self, text_encoding: &Tensor, inference: bool) -> Result<Tensor> {
        let log_durations = self.forward(text_encoding)?;

        // Convert log-durations to durations
        let mut durations = log_durations
            .exp()
            .map_err(|e| AcousticError::ModelError(format!("Exponential failed: {e}")))?;

        if inference {
            // During inference, apply noise reduction and rounding
            let noise_scale = 0.667; // Empirical noise scale for inference
            durations = (durations * noise_scale)
                .map_err(|e| AcousticError::ModelError(format!("Noise scaling failed: {e}")))?;
        }

        // Ensure minimum duration (at least 1 frame)
        let ones = Tensor::ones(durations.dims(), DType::F32, &self.device)
            .map_err(|e| AcousticError::ModelError(format!("Creating ones tensor failed: {e}")))?;

        durations = durations
            .maximum(&ones)
            .map_err(|e| AcousticError::ModelError(format!("Maximum operation failed: {e}")))?;

        Ok(durations)
    }

    /// Predict phoneme durations from phoneme sequence
    ///
    /// # Arguments
    /// * `phonemes` - Input phoneme sequence
    ///
    /// # Returns
    /// * Vector of duration predictions (in frames)
    pub fn predict_phoneme_durations(&self, phonemes: &[Phoneme]) -> Result<Vec<f32>> {
        self.predict_phoneme_durations_with_seed(phonemes, None)
    }

    pub fn predict_phoneme_durations_with_seed(
        &self,
        phonemes: &[Phoneme],
        seed: Option<u64>,
    ) -> Result<Vec<f32>> {
        if phonemes.is_empty() {
            return Ok(Vec::new());
        }

        tracing::debug!("Predicting durations for {} phonemes", phonemes.len());

        // For now, use simple heuristic-based duration prediction
        // In a full implementation, this would use the neural network
        let mut durations = Vec::new();

        // Initialize deterministic random number generator if seed is provided
        let mut rng_state = seed.unwrap_or(42); // Default seed for deterministic behavior when no seed provided

        for phoneme in phonemes.iter() {
            let base_duration = match phoneme.symbol.as_str() {
                // Vowels tend to be longer
                "AA" | "AE" | "AH" | "AO" | "AW" | "AY" | "EH" | "ER" | "EY" | "IH" | "IY"
                | "OW" | "OY" | "UH" | "UW" => 8.0,
                // Consonants
                "B" | "CH" | "D" | "DH" | "F" | "G" | "HH" | "JH" | "K" | "L" | "M" | "N"
                | "NG" | "P" | "R" | "S" | "SH" | "T" | "TH" | "V" | "W" | "Y" | "Z" | "ZH" => 4.0,
                // Special tokens
                "<pad>" | "<unk>" => 1.0,
                "<bos>" | "<eos>" => 2.0,
                // Default
                _ => 6.0,
            };

            // Add variation using deterministic random generation
            // Always use deterministic generation for reproducibility
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let random_val = rng_state as f32 / u64::MAX as f32;
            let variation = (random_val - 0.5) * 0.4 + 1.0; // 0.8 to 1.2

            let duration = base_duration * variation;

            durations.push(duration.max(1.0)); // Minimum 1 frame
        }

        tracing::debug!("Predicted durations: {:?}", durations);

        Ok(durations)
    }

    /// Align phoneme sequence to mel spectrogram using predicted durations
    ///
    /// # Arguments
    /// * `text_encoding` - Text encoder outputs [batch, input_dim, seq_len]
    /// * `durations` - Duration predictions [batch, 1, seq_len]
    ///
    /// # Returns
    /// * Aligned text encoding [batch, input_dim, total_frames]
    pub fn align_text_to_mel(&self, text_encoding: &Tensor, durations: &Tensor) -> Result<Tensor> {
        let (batch_size, input_dim, seq_len) = text_encoding.dims3().map_err(|e| {
            AcousticError::ModelError(format!("Failed to get text encoding dimensions: {e}"))
        })?;

        let (dur_batch, dur_channels, dur_seq) = durations.dims3().map_err(|e| {
            AcousticError::ModelError(format!("Failed to get duration dimensions: {e}"))
        })?;

        // Validate dimensions
        if batch_size != dur_batch || seq_len != dur_seq || dur_channels != 1 {
            return Err(AcousticError::InputError(format!(
                "Dimension mismatch: text [{batch_size}, {input_dim}, {seq_len}], durations [{dur_batch}, {dur_channels}, {dur_seq}]"
            )));
        }

        // Squeeze duration channel dimension
        let durations = durations
            .squeeze(1)
            .map_err(|e| AcousticError::ModelError(format!("Failed to squeeze durations: {e}")))?;

        // For simplicity, use a basic upsampling approach
        // In a full implementation, this would use more sophisticated alignment algorithms

        // Calculate total frames needed
        let total_frames = durations
            .sum_all()
            .map_err(|e| AcousticError::ModelError(format!("Failed to sum durations: {e}")))?
            .to_scalar::<f32>()
            .map_err(|e| AcousticError::ModelError(format!("Failed to convert to scalar: {e}")))?
            as usize;

        tracing::debug!("Aligning {} phonemes to {} frames", seq_len, total_frames);

        // For now, use simple repetition-based alignment
        // This is a placeholder - a full implementation would use differentiable upsampling
        let repeat_factor = total_frames / seq_len.max(1);
        let aligned = text_encoding.repeat(&[1, 1, repeat_factor]).map_err(|e| {
            AcousticError::ModelError(format!("Failed to align text encoding: {e}"))
        })?;

        Ok(aligned)
    }
}

/// Differentiable upsampling using durations
pub fn duration_based_upsampling(
    text_encoding: &Tensor,
    durations: &Tensor,
    _device: &Device,
) -> Result<Tensor> {
    let (batch_size, _channels, seq_len) = text_encoding
        .dims3()
        .map_err(|e| AcousticError::ModelError(format!("Invalid text encoding shape: {e}")))?;

    // Calculate total output length
    let _total_frames = durations
        .sum_all()
        .map_err(|e| AcousticError::ModelError(format!("Failed to sum durations: {e}")))?
        .to_scalar::<f32>()
        .map_err(|e| AcousticError::ModelError(format!("Failed to convert to scalar: {e}")))?
        as usize;

    // Simple implementation: repeat each phoneme encoding for its duration
    // In practice, this would use more sophisticated interpolation
    let mut output_data = Vec::new();

    for batch_idx in 0..batch_size {
        let mut batch_output = Vec::new();

        for seq_idx in 0..seq_len {
            let duration = durations
                .get(batch_idx)?
                .get(seq_idx)?
                .to_scalar::<f32>()
                .map_err(|e| {
                    AcousticError::ModelError(format!("Failed to get duration scalar: {e}"))
                })? as usize;

            let frame_encoding = text_encoding.get(batch_idx)?.narrow(1, seq_idx, 1)?; // [channels, 1]

            // Repeat this encoding for 'duration' frames
            for _ in 0..duration {
                batch_output.push(frame_encoding.clone());
            }
        }

        // Concatenate all frames for this batch
        if !batch_output.is_empty() {
            let batch_tensor =
                Tensor::cat(&batch_output.iter().collect::<Vec<_>>(), 1).map_err(|e| {
                    AcousticError::ModelError(format!("Failed to concatenate frames: {e}"))
                })?;
            output_data.push(batch_tensor);
        }
    }

    if output_data.is_empty() {
        return Err(AcousticError::ModelError(
            "No output data generated".to_string(),
        ));
    }

    // Stack all batches
    let result = Tensor::stack(&output_data, 0)
        .map_err(|e| AcousticError::ModelError(format!("Failed to stack batches: {e}")))?;

    Ok(result)
}
