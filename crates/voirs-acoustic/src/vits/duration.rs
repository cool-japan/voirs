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
        .map_err(|e| AcousticError::ModelError {
            message: format!("Failed to create input_conv: {e}"),
        })?;

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
            .map_err(|e| AcousticError::ModelError {
                message: format!("Failed to create conv block {i}: {e}"),
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
        .map_err(|e| AcousticError::ModelError {
            message: format!("Failed to create output_conv: {e}"),
        })?;

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
            return Err(AcousticError::InputError {
                message: format!(
                    "Expected 3D tensor [batch, input_dim, seq_len], got {input_shape:?}"
                ),
            });
        }

        let (batch_size, input_dim, seq_len) =
            text_encoding
                .dims3()
                .map_err(|e| AcousticError::ModelError {
                    message: format!("Failed to get tensor dimensions: {e}"),
                })?;

        if input_dim != self.config.input_dim {
            return Err(AcousticError::InputError {
                message: format!(
                    "Expected {} input dimensions, got {}",
                    self.config.input_dim, input_dim
                ),
            });
        }

        tracing::debug!(
            "DurationPredictor forward: input shape [{}, {}, {}]",
            batch_size,
            input_dim,
            seq_len
        );

        // Input convolution
        let mut h =
            self.input_conv
                .forward(text_encoding)
                .map_err(|e| AcousticError::ModelError {
                    message: format!("Input convolution failed: {e}"),
                })?;

        tracing::debug!("After input_conv: {:?}", h.dims());

        // Apply convolutional blocks
        for (i, block) in self.conv_blocks.iter().enumerate() {
            h = block.forward(&h).map_err(|e| AcousticError::ModelError {
                message: format!("Conv block {i} failed: {e}"),
            })?;
        }

        tracing::debug!("After conv blocks: {:?}", h.dims());

        // Output convolution to predict log-durations
        let log_durations =
            self.output_conv
                .forward(&h)
                .map_err(|e| AcousticError::ModelError {
                    message: format!("Output convolution failed: {e}"),
                })?;

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
        let mut durations = log_durations.exp().map_err(|e| AcousticError::ModelError {
            message: format!("Exponential failed: {e}"),
        })?;

        if inference {
            // During inference, apply noise reduction and rounding
            let noise_scale = 0.667; // Empirical noise scale for inference
            durations = (durations * noise_scale).map_err(|e| AcousticError::ModelError {
                message: format!("Noise scaling failed: {e}"),
            })?;
        }

        // Ensure minimum duration (at least 1 frame)
        let ones = Tensor::ones(durations.dims(), DType::F32, &self.device).map_err(|e| {
            AcousticError::ModelError {
                message: format!("Creating ones tensor failed: {e}"),
            }
        })?;

        durations = durations
            .maximum(&ones)
            .map_err(|e| AcousticError::ModelError {
                message: format!("Maximum operation failed: {e}"),
            })?;

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

    /// Align phoneme sequence to mel spectrogram using predicted durations.
    ///
    /// Each phoneme encoding is repeated exactly `round(duration[b][p]).max(1)` times
    /// along the time axis so that the output faithfully reflects per-phoneme duration.
    ///
    /// # Arguments
    /// * `text_encoding` - Text encoder outputs `[batch, input_dim, seq_len]`
    /// * `durations`     - Duration predictions `[batch, 1, seq_len]` (float frames)
    ///
    /// # Returns
    /// * Aligned text encoding `[batch, input_dim, total_frames]`
    pub fn align_text_to_mel(&self, text_encoding: &Tensor, durations: &Tensor) -> Result<Tensor> {
        let (batch_size, input_dim, seq_len) =
            text_encoding
                .dims3()
                .map_err(|e| AcousticError::ModelError {
                    message: format!("Failed to get text encoding dimensions: {e}"),
                })?;

        let (dur_batch, dur_channels, dur_seq) =
            durations.dims3().map_err(|e| AcousticError::ModelError {
                message: format!("Failed to get duration dimensions: {e}"),
            })?;

        if batch_size != dur_batch || seq_len != dur_seq || dur_channels != 1 {
            return Err(AcousticError::InputError {
                message: format!(
                    "Dimension mismatch: text [{batch_size}, {input_dim}, {seq_len}], \
                     durations [{dur_batch}, {dur_channels}, {dur_seq}]"
                ),
            });
        }

        // Squeeze channel dimension → [batch, seq_len]
        let durations_sq = durations
            .squeeze(1)
            .map_err(|e| AcousticError::ModelError {
                message: format!("Failed to squeeze durations: {e}"),
            })?;

        // Extract duration values as Vec<Vec<f32>>: [batch][seq_len]
        let dur_data: Vec<Vec<f32>> =
            durations_sq
                .to_vec2::<f32>()
                .map_err(|e| AcousticError::ModelError {
                    message: format!("Failed to extract duration data: {e}"),
                })?;

        // Permute text_encoding to [batch, seq_len, input_dim] for easy per-phoneme access
        let text_permuted =
            text_encoding
                .permute((0, 2, 1))
                .map_err(|e| AcousticError::ModelError {
                    message: format!("Failed to permute text encoding: {e}"),
                })?;

        // Extract as [batch][seq_len][input_dim]
        let text_data: Vec<Vec<Vec<f32>>> =
            text_permuted
                .to_vec3::<f32>()
                .map_err(|e| AcousticError::ModelError {
                    message: format!("Failed to extract text encoding data: {e}"),
                })?;

        // Build per-batch expanded vectors and track the maximum total-frame count
        // so that batches with different sums can be padded to a uniform length.
        let mut all_batches: Vec<Vec<f32>> = Vec::with_capacity(batch_size);
        let mut max_total_frames: usize = 0;

        for b in 0..batch_size {
            let mut expanded: Vec<f32> = Vec::new();
            for (p, &dur_val) in dur_data[b].iter().enumerate() {
                // Round to nearest integer; clamp to minimum 1 frame
                let repeat = (dur_val.round() as usize).max(1);
                for _ in 0..repeat {
                    expanded.extend_from_slice(&text_data[b][p]);
                }
            }
            let total_frames = expanded.len() / input_dim;
            if total_frames > max_total_frames {
                max_total_frames = total_frames;
            }
            all_batches.push(expanded);
        }

        tracing::debug!(
            "align_text_to_mel: {} phonemes → {} frames (max across batch)",
            seq_len,
            max_total_frames
        );

        // Pad shorter batches with zeros so all share the same time dimension
        for batch_data in &mut all_batches {
            let target_len = max_total_frames * input_dim;
            batch_data.resize(target_len, 0.0_f32);
        }

        // Flatten to a single Vec and create tensor with shape [batch, max_total_frames, input_dim]
        let flat: Vec<f32> = all_batches.into_iter().flatten().collect();
        let result = Tensor::from_vec(
            flat,
            (batch_size, max_total_frames, input_dim),
            text_encoding.device(),
        )
        .and_then(|t| t.permute((0, 2, 1))) // → [batch, input_dim, max_total_frames]
        .map_err(|e| AcousticError::ModelError {
            message: format!("Failed to create aligned tensor: {e}"),
        })?;

        Ok(result)
    }
}

/// Duration-based upsampling: expand each phoneme encoding along the time axis.
///
/// Each phoneme at position `p` in batch `b` is repeated
/// `round(durations[b][p]).max(1)` times.  All batches are padded to the same
/// maximum total-frame count with zeros.
///
/// # Arguments
/// * `text_encoding` - Shape `[batch, channels, seq_len]`
/// * `durations`     - Shape `[batch, seq_len]` (float frame counts, **not** `[batch,1,seq_len]`)
/// * `_device`       - Unused; the output device is inherited from `text_encoding`
///
/// # Returns
/// * Expanded tensor with shape `[batch, channels, max_total_frames]`
pub fn duration_based_upsampling(
    text_encoding: &Tensor,
    durations: &Tensor,
    _device: &Device,
) -> Result<Tensor> {
    let (batch_size, channels, seq_len) =
        text_encoding
            .dims3()
            .map_err(|e| AcousticError::ModelError {
                message: format!("Invalid text encoding shape: {e}"),
            })?;

    // Durations must be [batch, seq_len] — squeeze a leading channel dim if present
    let durations_2d = match durations.dims() {
        [b, 1, s] if *b == batch_size && *s == seq_len => {
            durations
                .squeeze(1)
                .map_err(|e| AcousticError::ModelError {
                    message: format!("Failed to squeeze duration channel dim: {e}"),
                })?
        }
        [b, s] if *b == batch_size && *s == seq_len => durations.clone(),
        other => {
            return Err(AcousticError::InputError {
                message: format!(
                    "duration_based_upsampling: expected durations shape [{batch_size}, {seq_len}] \
                     or [{batch_size}, 1, {seq_len}], got {other:?}"
                ),
            });
        }
    };

    // Extract duration values: [batch][seq_len]
    let dur_data: Vec<Vec<f32>> =
        durations_2d
            .to_vec2::<f32>()
            .map_err(|e| AcousticError::ModelError {
                message: format!("Failed to extract duration data: {e}"),
            })?;

    // Permute text_encoding to [batch, seq_len, channels] for slice access
    let text_permuted =
        text_encoding
            .permute((0, 2, 1))
            .map_err(|e| AcousticError::ModelError {
                message: format!("Failed to permute text encoding: {e}"),
            })?;

    let text_data: Vec<Vec<Vec<f32>>> =
        text_permuted
            .to_vec3::<f32>()
            .map_err(|e| AcousticError::ModelError {
                message: format!("Failed to extract text encoding data: {e}"),
            })?;

    // Build expanded data per batch; track maximum total-frame count for padding
    let mut all_batches: Vec<Vec<f32>> = Vec::with_capacity(batch_size);
    let mut max_total_frames: usize = 0;

    for b in 0..batch_size {
        let mut expanded: Vec<f32> = Vec::new();
        for (p, &dur_val) in dur_data[b].iter().enumerate() {
            let repeat = (dur_val.round() as usize).max(1);
            for _ in 0..repeat {
                expanded.extend_from_slice(&text_data[b][p]);
            }
        }
        let total_frames = expanded.len() / channels;
        if total_frames > max_total_frames {
            max_total_frames = total_frames;
        }
        all_batches.push(expanded);
    }

    if max_total_frames == 0 {
        return Err(AcousticError::ModelError {
            message: "duration_based_upsampling produced zero output frames".to_string(),
        });
    }

    // Zero-pad shorter batches
    for batch_data in &mut all_batches {
        batch_data.resize(max_total_frames * channels, 0.0_f32);
    }

    let flat: Vec<f32> = all_batches.into_iter().flatten().collect();
    // Create [batch, max_total_frames, channels] then permute → [batch, channels, max_total_frames]
    let result = Tensor::from_vec(
        flat,
        (batch_size, max_total_frames, channels),
        text_encoding.device(),
    )
    .and_then(|t| t.permute((0, 2, 1)))
    .map_err(|e| AcousticError::ModelError {
        message: format!("Failed to create upsampled tensor: {e}"),
    })?;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn cpu() -> Device {
        Device::Cpu
    }

    /// Build a `DurationPredictor` using the default config with CPU device.
    fn make_predictor() -> DurationPredictor {
        let config = DurationConfig::default();
        DurationPredictor::new(config, cpu()).expect("DurationPredictor::new failed")
    }

    // -------------------------------------------------------------------------
    // Helper: create a [batch, input_dim, seq_len] text-encoding tensor filled
    // with sequential values so that phoneme slices are distinguishable.
    //
    // Phoneme p in batch b gets value: (b * seq_len + p) as f32
    // -------------------------------------------------------------------------
    fn make_text_encoding(batch: usize, dim: usize, seq: usize) -> Tensor {
        let mut data = vec![0.0_f32; batch * dim * seq];
        for b in 0..batch {
            for p in 0..seq {
                let val = (b * seq + p) as f32;
                for d in 0..dim {
                    // layout is [batch, dim, seq] = row-major → index: b*(dim*seq) + d*seq + p
                    data[b * (dim * seq) + d * seq + p] = val;
                }
            }
        }
        Tensor::from_vec(data, (batch, dim, seq), &cpu()).expect("make_text_encoding failed")
    }

    // -------------------------------------------------------------------------
    // Test 1: basic expansion with mixed durations [1, 1, 2]
    // text_encoding=[1,4,3], durations=[1,1,2] (shape [1,1,3])
    // Expected output shape: [1, 4, 4]  (1+1+2 = 4 frames)
    // -------------------------------------------------------------------------
    #[test]
    fn test_align_text_to_mel_basic() {
        let pred = make_predictor();
        let text = make_text_encoding(1, 4, 3);
        // durations [1,1,3]: shape [batch=1, channels=1, seq=3]
        let dur_data: Vec<f32> = vec![1.0, 1.0, 2.0];
        let durations =
            Tensor::from_vec(dur_data, (1usize, 1usize, 3usize), &cpu()).expect("dur tensor");

        let aligned = pred
            .align_text_to_mel(&text, &durations)
            .expect("align_text_to_mel failed");

        let dims = aligned.dims();
        assert_eq!(dims.len(), 3, "output must be 3-D");
        assert_eq!(dims[0], 1, "batch dim must be 1");
        assert_eq!(dims[1], 4, "channel dim must equal input_dim=4");
        assert_eq!(dims[2], 4, "time dim must equal sum(durations)=4");
    }

    // -------------------------------------------------------------------------
    // Test 2: all durations = 1 → output time == seq_len (no expansion/collapse)
    // -------------------------------------------------------------------------
    #[test]
    fn test_align_duration_one_each() {
        let pred = make_predictor();
        let seq_len = 5_usize;
        let input_dim = 4_usize;
        let text = make_text_encoding(1, input_dim, seq_len);
        let dur_data: Vec<f32> = vec![1.0; seq_len];
        let durations =
            Tensor::from_vec(dur_data, (1usize, 1usize, seq_len), &cpu()).expect("dur tensor");

        let aligned = pred
            .align_text_to_mel(&text, &durations)
            .expect("align_text_to_mel failed");

        assert_eq!(
            aligned.dims(),
            &[1, input_dim, seq_len],
            "when every duration is 1, output time must equal seq_len"
        );
    }

    // -------------------------------------------------------------------------
    // Test 3: zero duration is clamped to 1 → output must not be shorter than seq_len
    // -------------------------------------------------------------------------
    #[test]
    fn test_align_zero_duration_clamped() {
        let pred = make_predictor();
        let seq_len = 3_usize;
        let input_dim = 4_usize;
        let text = make_text_encoding(1, input_dim, seq_len);
        // First phoneme has duration 0 — should be treated as 1
        let dur_data: Vec<f32> = vec![0.0, 1.0, 1.0];
        let durations =
            Tensor::from_vec(dur_data, (1usize, 1usize, seq_len), &cpu()).expect("dur tensor");

        let aligned = pred
            .align_text_to_mel(&text, &durations)
            .expect("align_text_to_mel failed");

        let time_dim = aligned.dims()[2];
        assert!(
            time_dim >= seq_len,
            "clamping zero durations means output time ({time_dim}) >= seq_len ({seq_len})"
        );
    }

    // -------------------------------------------------------------------------
    // Test 4: first frame of output equals first phoneme encoding (when dur[0]=1)
    // -------------------------------------------------------------------------
    #[test]
    fn test_align_preserves_content() {
        let pred = make_predictor();
        let input_dim = 4_usize;
        let seq_len = 3_usize;
        let text = make_text_encoding(1, input_dim, seq_len);

        // dur[0] = 1 so the first output frame must be phoneme 0's encoding
        let dur_data: Vec<f32> = vec![1.0, 2.0, 1.0];
        let durations =
            Tensor::from_vec(dur_data, (1usize, 1usize, seq_len), &cpu()).expect("dur tensor");

        let aligned = pred
            .align_text_to_mel(&text, &durations)
            .expect("align_text_to_mel failed");

        // Extract the first time-step across all channels: aligned[:, :, 0]
        let first_frame = aligned
            .narrow(2, 0, 1) // [1, input_dim, 1]
            .expect("narrow failed")
            .squeeze(2) // [1, input_dim]
            .expect("squeeze failed")
            .to_vec2::<f32>()
            .expect("to_vec2 failed");

        // The expected value for phoneme 0, batch 0 is 0.0 (see make_text_encoding)
        for &val in &first_frame[0] {
            assert_eq!(
                val, 0.0_f32,
                "first output frame must equal first phoneme encoding (phoneme 0 value = 0.0)"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Test 5: uniform durations of 2 → output shape [1, 4, 6]
    // text_encoding [1, 4, 3] + all-2 durations → 3 phonemes × 2 frames = 6 frames
    // -------------------------------------------------------------------------
    #[test]
    fn test_align_uniform_durations() {
        let pred = make_predictor();
        let batch = 1_usize;
        let channels = 4_usize;
        let seq_len = 3_usize;
        let text = make_text_encoding(batch, channels, seq_len);

        // All durations = 2 → 3 phonemes each repeated twice → 6 mel frames
        let dur_data: Vec<f32> = vec![2.0, 2.0, 2.0];
        let durations =
            Tensor::from_vec(dur_data, (batch, 1usize, seq_len), &cpu()).expect("dur tensor");

        let aligned = pred
            .align_text_to_mel(&text, &durations)
            .expect("align_text_to_mel failed");

        assert_eq!(
            aligned.dims(),
            &[1, 4, 6],
            "uniform durations=2 over 3 phonemes must yield [1, 4, 6] output"
        );
    }

    // -------------------------------------------------------------------------
    // Test 6: variable durations [1, 2, 3] → sum = 6 frames output
    // -------------------------------------------------------------------------
    #[test]
    fn test_align_variable_durations() {
        let pred = make_predictor();
        let batch = 1_usize;
        let channels = 4_usize;
        let seq_len = 3_usize;
        let text = make_text_encoding(batch, channels, seq_len);

        // Durations [1, 2, 3] → total = 6 frames
        let dur_data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let expected_total: usize = 6; // 1 + 2 + 3
        let durations =
            Tensor::from_vec(dur_data, (batch, 1usize, seq_len), &cpu()).expect("dur tensor");

        let aligned = pred
            .align_text_to_mel(&text, &durations)
            .expect("align_text_to_mel failed");

        assert_eq!(
            aligned.dims(),
            &[1, channels, expected_total],
            "variable durations [1,2,3] must yield time-dim = 6"
        );
    }

    // -------------------------------------------------------------------------
    // Test 7: output time-dim equals sum of (rounded) durations
    // -------------------------------------------------------------------------
    #[test]
    fn test_align_shape() {
        let pred = make_predictor();
        let batch = 2_usize;
        let channels = 8_usize;
        let seq_len = 5_usize;
        let text = make_text_encoding(batch, channels, seq_len);

        // Durations: each batch item has the same durations [1, 3, 2, 4, 2] → sum = 12
        let dur_data: Vec<f32> = vec![
            1.0, 3.0, 2.0, 4.0, 2.0, // batch 0
            1.0, 3.0, 2.0, 4.0, 2.0, // batch 1
        ];
        let expected_total: usize = 12; // 1+3+2+4+2
        let durations =
            Tensor::from_vec(dur_data, (batch, 1usize, seq_len), &cpu()).expect("dur tensor");

        let aligned = pred
            .align_text_to_mel(&text, &durations)
            .expect("align_text_to_mel failed");

        assert_eq!(
            aligned.dims()[0],
            batch,
            "batch dimension must be preserved"
        );
        assert_eq!(
            aligned.dims()[1],
            channels,
            "channel dimension must be preserved"
        );
        assert_eq!(
            aligned.dims()[2],
            expected_total,
            "time-dim must equal sum of durations"
        );
    }

    // -------------------------------------------------------------------------
    // Test 8: content verification — specific frames map to the correct phoneme
    // durations [2, 1, 3] for a [1, 4, 3] encoding:
    //   frames 0-1  → phoneme 0 (value 0.0)
    //   frame  2    → phoneme 1 (value 1.0)
    //   frames 3-5  → phoneme 2 (value 2.0)
    // -------------------------------------------------------------------------
    #[test]
    fn test_align_content() {
        let pred = make_predictor();
        let channels = 4_usize;
        let seq_len = 3_usize;
        // make_text_encoding: phoneme p in batch 0 gets constant value p as f32
        let text = make_text_encoding(1, channels, seq_len);

        let dur_data: Vec<f32> = vec![2.0, 1.0, 3.0]; // frames: 0-1 → ph0, 2 → ph1, 3-5 → ph2
        let durations =
            Tensor::from_vec(dur_data, (1usize, 1usize, seq_len), &cpu()).expect("dur tensor");

        let aligned = pred
            .align_text_to_mel(&text, &durations)
            .expect("align_text_to_mel failed");

        // Helper: extract all channel values at a specific time-frame
        let frame_values = |frame_idx: usize| -> Vec<f32> {
            aligned
                .narrow(2, frame_idx, 1)
                .expect("narrow time failed")
                .squeeze(2)
                .expect("squeeze failed")
                .to_vec2::<f32>()
                .expect("to_vec2 failed")
                .remove(0)
        };

        // Phoneme 0: frames 0 and 1 must all equal 0.0
        for frame_idx in 0..2 {
            let vals = frame_values(frame_idx);
            for (ch, &v) in vals.iter().enumerate() {
                assert_eq!(
                    v, 0.0_f32,
                    "frame {frame_idx} channel {ch}: expected phoneme-0 value 0.0, got {v}"
                );
            }
        }

        // Phoneme 1: frame 2 must all equal 1.0
        {
            let vals = frame_values(2);
            for (ch, &v) in vals.iter().enumerate() {
                assert_eq!(
                    v, 1.0_f32,
                    "frame 2 channel {ch}: expected phoneme-1 value 1.0, got {v}"
                );
            }
        }

        // Phoneme 2: frames 3, 4, 5 must all equal 2.0
        for frame_idx in 3..6 {
            let vals = frame_values(frame_idx);
            for (ch, &v) in vals.iter().enumerate() {
                assert_eq!(
                    v, 2.0_f32,
                    "frame {frame_idx} channel {ch}: expected phoneme-2 value 2.0, got {v}"
                );
            }
        }
    }
}
