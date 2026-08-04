//! Neural model architectures for spatial audio processing

use super::types::*;
use crate::{Error, Result};
use candle_core::{Device, Module, Tensor, Var, D};
use candle_nn::{Linear, VarBuilder, VarMap};
use std::collections::HashMap;

/// Trait for different neural model implementations
pub trait NeuralModel {
    /// Forward pass through the model
    fn forward(&self, input: &NeuralInputFeatures) -> Result<NeuralSpatialOutput>;

    /// Get model configuration
    fn config(&self) -> &NeuralSpatialConfig;

    /// Update model parameters
    fn update_parameters(&mut self, params: &HashMap<String, Tensor>) -> Result<()>;

    /// Get model performance metrics
    fn metrics(&self) -> NeuralPerformanceMetrics;

    /// Save model to file
    fn save(&self, path: &str) -> Result<()>;

    /// Load model from file
    fn load(&mut self, path: &str) -> Result<()>;

    /// Get memory usage in bytes
    fn memory_usage(&self) -> usize;

    /// Set quality level (0.0-1.0)
    fn set_quality(&mut self, quality: f32) -> Result<()>;

    /// Build the model's input tensor for one input-feature sample.
    ///
    /// The resulting tensor's shape is model-specific (documented on each
    /// implementation) and is the value that should be passed to
    /// [`NeuralModel::forward_tensor`].
    fn input_tensor(&self, input: &NeuralInputFeatures) -> Result<Tensor>;

    /// Differentiable forward pass used by [`super::training::NeuralTrainer`].
    ///
    /// Unlike [`NeuralModel::forward`], this returns the raw output `Tensor`
    /// with its autograd graph intact (no data is copied out to `Vec<f32>`),
    /// so a caller can build a loss `Tensor` from the result and call
    /// `.backward()` to obtain real gradients with respect to every `Var` in
    /// [`NeuralModel::trainable_vars`].
    fn forward_tensor(&self, input: &Tensor) -> Result<Tensor>;

    /// Expose the candle [`Var`]s backing this model's trainable weights.
    ///
    /// Used to build a real optimizer (e.g. `candle_nn::AdamW`) over the
    /// model. Returns an empty vector for models with no trainable
    /// parameters.
    fn trainable_vars(&self) -> Vec<Var>;
}

/// Feedforward neural network implementation
pub struct FeedforwardModel {
    config: NeuralSpatialConfig,
    layers: Vec<Linear>,
    device: Device,
    metrics: NeuralPerformanceMetrics,
    /// Backing store for every weight/bias `Var` in `layers`. Kept around
    /// (rather than dropped after construction) so parameters can be updated
    /// in place, persisted with real tensor data via [`VarMap::save`] /
    /// [`VarMap::load`] (safetensors format), and exposed to the trainer.
    varmap: VarMap,
}

/// Convolutional neural network implementation
pub struct ConvolutionalModel {
    config: NeuralSpatialConfig,
    conv_layers: Vec<candle_nn::Conv1d>,
    linear_layers: Vec<Linear>,
    device: Device,
    metrics: NeuralPerformanceMetrics,
    /// Backing store for every weight/bias `Var`, see [`FeedforwardModel::varmap`].
    varmap: VarMap,
}

/// Transformer model implementation
pub struct TransformerModel {
    config: NeuralSpatialConfig,
    encoder: TransformerEncoder,
    decoder: TransformerDecoder,
    /// Learned projection from the raw feature vector (`config.input_dim`) to
    /// the transformer's model dimension. Built once at construction time and
    /// trained like any other layer - never regenerated per forward call.
    input_projection: Linear,
    /// Learned projection from the model dimension back to the flat binaural
    /// output (`output_channels * buffer_size`).
    output_projection: Linear,
    device: Device,
    metrics: NeuralPerformanceMetrics,
    /// Backing store for every weight/bias `Var`, see [`FeedforwardModel::varmap`].
    varmap: VarMap,
}

/// Transformer encoder layer
pub struct TransformerEncoder {
    attention: MultiHeadAttention,
    feedforward: FeedForwardLayer,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

/// Transformer decoder layer
pub struct TransformerDecoder {
    self_attention: MultiHeadAttention,
    cross_attention: MultiHeadAttention,
    feedforward: FeedForwardLayer,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
}

/// Multi-head attention mechanism
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
}

/// Feed-forward layer
pub struct FeedForwardLayer {
    linear1: Linear,
    linear2: Linear,
    /// Dropout probability. Currently unused: this crate only implements the
    /// inference path, and dropout is defined to be the identity function at
    /// inference time, so no masking is applied here.
    dropout: f32,
}

/// Layer normalization
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl MultiHeadAttention {
    /// Build a new multi-head attention block with freshly-initialized,
    /// trainable Q/K/V/output projections registered under `vb`.
    fn new(num_heads: usize, head_dim: usize, model_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            num_heads,
            head_dim,
            query: candle_nn::linear(model_dim, model_dim, vb.pp("query"))?,
            key: candle_nn::linear(model_dim, model_dim, vb.pp("key"))?,
            value: candle_nn::linear(model_dim, model_dim, vb.pp("value"))?,
            output: candle_nn::linear(model_dim, model_dim, vb.pp("output"))?,
        })
    }

    /// Real scaled dot-product multi-head attention.
    ///
    /// `query_input` provides Q; `kv_input` provides K and V (pass the same
    /// tensor for self-attention, or the encoder output for cross-attention).
    /// Both inputs are expected to have shape `(batch, seq_len, model_dim)`.
    fn forward(&self, query_input: &Tensor, kv_input: &Tensor) -> Result<Tensor> {
        let q = self.query.forward(query_input)?;
        let k = self.key.forward(kv_input)?;
        let v = self.value.forward(kv_input)?;

        let q = self.split_heads(&q)?;
        let k = self.split_heads(&k)?;
        let v = self.split_heads(&v)?;

        // Scaled dot-product attention: softmax(Q K^T / sqrt(d_k)) V
        let scale = (self.head_dim as f64).sqrt();
        let scores = q
            .matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)?
            .affine(1.0 / scale, 0.0)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let context = attn.matmul(&v)?;

        let merged = self.merge_heads(&context)?;
        Ok(self.output.forward(&merged)?)
    }

    /// Reshape `(batch, seq_len, model_dim)` into `(batch, num_heads, seq_len, head_dim)`.
    fn split_heads(&self, x: &Tensor) -> Result<Tensor> {
        let batch = x.dim(0)?;
        let seq_len = x.dim(1)?;
        Ok(x.reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?)
    }

    /// Inverse of [`Self::split_heads`]: `(batch, num_heads, seq_len, head_dim)`
    /// back to `(batch, seq_len, model_dim)`.
    fn merge_heads(&self, x: &Tensor) -> Result<Tensor> {
        let batch = x.dim(0)?;
        let seq_len = x.dim(2)?;
        Ok(x.transpose(1, 2)?.contiguous()?.reshape((
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ))?)
    }
}

impl FeedForwardLayer {
    /// Build a new position-wise feed-forward block with freshly-initialized,
    /// trainable weights registered under `vb`.
    fn new(model_dim: usize, ff_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear1: candle_nn::linear(model_dim, ff_dim, vb.pp("linear1"))?,
            linear2: candle_nn::linear(ff_dim, model_dim, vb.pp("linear2"))?,
            dropout: 0.1,
        })
    }

    /// `linear2(relu(linear1(x)))`, the standard transformer feed-forward block.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.linear1.forward(x)?.relu()?;
        Ok(self.linear2.forward(&hidden)?)
    }
}

impl LayerNorm {
    /// Build a new layer-normalization block with trainable scale/shift
    /// parameters (initialized to 1/0 respectively, the standard LayerNorm
    /// starting point) registered under `vb`.
    fn new(model_dim: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(model_dim, "weight", candle_nn::Init::Const(1.0))?;
        let bias = vb.get_with_hints(model_dim, "bias", candle_nn::Init::Const(0.0))?;
        Ok(Self {
            weight,
            bias,
            eps: 1e-5,
        })
    }

    /// Normalize over the last dimension, then apply the learned scale/shift:
    /// `weight * (x - mean) / sqrt(var + eps) + bias`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let centered = x.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let denom = variance.affine(1.0, self.eps)?.sqrt()?;
        let normalized = centered.broadcast_div(&denom)?;
        Ok(normalized
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?)
    }
}

/// Apply every `(name, tensor)` pair in `params` to the corresponding `Var`
/// in `varmap`, using [`VarMap::set_one`] so shapes are validated by candle
/// itself. Returns a descriptive error (including the offending parameter
/// name) on the first failure - either the name doesn't exist in this
/// model's varmap, or the provided tensor's shape doesn't match the existing
/// parameter's shape.
fn apply_parameter_updates(varmap: &mut VarMap, params: &HashMap<String, Tensor>) -> Result<()> {
    if params.is_empty() {
        return Err(Error::LegacyProcessing(
            "update_parameters called with no parameters to apply".to_string(),
        ));
    }
    for (name, tensor) in params {
        varmap.set_one(name, tensor).map_err(|e| {
            Error::LegacyProcessing(format!("Failed to update parameter '{name}': {e}"))
        })?;
    }
    Ok(())
}

/// Compute the exact trainable parameter count for `varmap` (the sum of
/// `elem_count()` over every registered `Var`), used to derive a real memory
/// estimate instead of a hand-derived architecture formula that can drift out
/// of sync with the actual layers.
fn varmap_param_count(varmap: &VarMap) -> usize {
    varmap.all_vars().iter().map(|var| var.elem_count()).sum()
}

/// [`ConvolutionalModel`]'s downsampling stages stop shrinking the sequence
/// once its length is at or below this many frames (matching the
/// `dims()[2] > 2` guard in [`ConvolutionalModel::forward_tensor`], which
/// skips the stride-2 `index_select` once the sequence gets this short).
const CONV_DOWNSAMPLE_MIN_LEN: usize = 2;

/// Predict the exact sequence length [`ConvolutionalModel::forward_tensor`]
/// produces after `num_stages` "same-padding conv + conditional stride-2
/// downsample" stages, starting from `input_len`.
///
/// Each stage halves the length (rounding up, matching `step_by(2)`) unless
/// the length is already at or below [`CONV_DOWNSAMPLE_MIN_LEN`], in which
/// case that stage leaves it unchanged - mirroring `forward_tensor`'s guard
/// exactly. [`ConvolutionalModel::new`] uses this to size the first `Linear`
/// layer correctly; previously that size was a hand-derived `input_dim / 4`
/// formula that assumed a flat 4x downsample regardless of `num_stages`,
/// which drifted out of sync with the real (up to 8x, for three stages)
/// downsampling actually performed and made every forward pass fail with a
/// matmul shape-mismatch error for most `input_dim` values.
fn conv_output_sequence_length(input_len: usize, num_stages: usize) -> usize {
    let mut len = input_len;
    for _ in 0..num_stages {
        if len > CONV_DOWNSAMPLE_MIN_LEN {
            len = len.div_ceil(2);
        }
    }
    len
}

impl FeedforwardModel {
    /// Create a new feedforward neural network model
    pub fn new(config: NeuralSpatialConfig, device: Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let mut layers = Vec::new();
        let mut input_dim = config.input_dim;

        for &hidden_dim in &config.hidden_dims {
            layers.push(candle_nn::linear(
                input_dim,
                hidden_dim,
                vb.pp(format!("layer_{}", layers.len())),
            )?);
            input_dim = hidden_dim;
        }

        // Output layer for binaural audio
        let output_dim = config.output_channels * config.buffer_size;
        layers.push(candle_nn::linear(input_dim, output_dim, vb.pp("output"))?);

        Ok(Self {
            config,
            layers,
            device,
            metrics: NeuralPerformanceMetrics::default(),
            varmap,
        })
    }
}

impl NeuralModel for FeedforwardModel {
    fn forward(&self, input: &NeuralInputFeatures) -> Result<NeuralSpatialOutput> {
        let input_tensor = self.input_tensor(input)?;
        let x = self.forward_tensor(&input_tensor)?;

        // Convert output tensor to binaural audio
        let output_data = x
            .to_vec2::<f32>()
            .map_err(|e| Error::LegacyProcessing(format!("Failed to extract output data: {e}")))?;

        let binaural_audio = self.tensor_to_binaural_audio(&output_data[0]);
        let confidence = self.estimate_confidence(&output_data[0]);

        Ok(NeuralSpatialOutput {
            binaural_audio,
            confidence,
            latency_ms: 0.0, // Will be set by processor
            quality_score: self.config.quality,
            metadata: HashMap::new(),
        })
    }

    fn config(&self) -> &NeuralSpatialConfig {
        &self.config
    }

    fn input_tensor(&self, input: &NeuralInputFeatures) -> Result<Tensor> {
        let input_vec = self.features_to_vector(input);
        Tensor::from_vec(input_vec, (1, self.config.input_dim), &self.device)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to create input tensor: {e}")))
    }

    fn forward_tensor(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();

        // Forward pass through hidden layers
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x).map_err(|e| {
                Error::LegacyProcessing(format!("Forward pass failed at layer {i}: {e}"))
            })?;

            // Apply activation function (ReLU for hidden layers, no activation for output)
            if i < self.layers.len() - 1 {
                x = x
                    .relu()
                    .map_err(|e| Error::LegacyProcessing(format!("ReLU activation failed: {e}")))?;
            }
        }

        // Bound the output to valid audio sample range; this is the model's
        // real output activation, applied once here so both the inference
        // path (`forward`) and the training path (`NeuralTrainer`) see
        // identical numerics.
        x.tanh()
            .map_err(|e| Error::LegacyProcessing(format!("Output activation failed: {e}")))
    }

    fn trainable_vars(&self) -> Vec<Var> {
        self.varmap.all_vars()
    }

    fn update_parameters(&mut self, params: &HashMap<String, Tensor>) -> Result<()> {
        apply_parameter_updates(&mut self.varmap, params)?;

        self.metrics.last_updated = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(())
    }

    fn metrics(&self) -> NeuralPerformanceMetrics {
        self.metrics.clone()
    }

    fn save(&self, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        // Persist the real weight tensors in safetensors format alongside the
        // metadata file.
        let weights_path = format!("{path}.safetensors");
        self.varmap.save(&weights_path).map_err(|e| {
            Error::LegacyProcessing(format!(
                "Failed to save model weights to {weights_path}: {e}"
            ))
        })?;

        // Create the model save data structure
        let save_data = serde_json::json!({
            "model_type": "feedforward",
            "config": self.config,
            "layer_count": self.layers.len(),
            "metrics": self.metrics,
            "weights_path": weights_path,
            "saved_at": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            "version": "1.0"
        });

        // Write model configuration and metadata
        let mut file = File::create(path)
            .map_err(|e| Error::LegacyConfig(format!("Failed to create model file {path}: {e}")))?;

        file.write_all(save_data.to_string().as_bytes())
            .map_err(|e| Error::LegacyConfig(format!("Failed to write model data: {e}")))?;

        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<()> {
        use std::fs;

        // Read the saved model file
        let model_data = fs::read_to_string(path)
            .map_err(|e| Error::LegacyConfig(format!("Failed to read model file {path}: {e}")))?;

        // Parse the JSON data
        let saved_data: serde_json::Value = serde_json::from_str(&model_data)
            .map_err(|e| Error::LegacyConfig(format!("Failed to parse model file: {e}")))?;

        // Validate model type
        let model_type = saved_data["model_type"]
            .as_str()
            .ok_or_else(|| Error::LegacyConfig("Missing model_type in saved file".to_string()))?;

        if model_type != "feedforward" {
            return Err(Error::LegacyConfig(format!(
                "Model type mismatch: expected 'feedforward', found '{model_type}'"
            )));
        }

        // Load configuration
        let loaded_config: NeuralSpatialConfig =
            serde_json::from_value(saved_data["config"].clone())
                .map_err(|e| Error::LegacyConfig(format!("Failed to parse saved config: {e}")))?;

        // Update current configuration
        self.config = loaded_config;

        // Load metrics if available
        if let Ok(loaded_metrics) =
            serde_json::from_value::<NeuralPerformanceMetrics>(saved_data["metrics"].clone())
        {
            self.metrics = loaded_metrics;
        }

        // Load the real weight tensors. `VarMap::load` looks up every `Var`
        // already registered in `self.varmap` by name in the safetensors file
        // and copies its data in, failing with a descriptive error if a name
        // is missing or a shape doesn't match - never silently keeping the
        // untrained construction-time weights.
        let weights_path = saved_data["weights_path"]
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| format!("{path}.safetensors"));

        self.varmap.load(&weights_path).map_err(|e| {
            Error::LegacyConfig(format!(
                "Failed to load model weights from {weights_path}: {e}"
            ))
        })?;

        Ok(())
    }

    fn memory_usage(&self) -> usize {
        varmap_param_count(&self.varmap) * 4 // 4 bytes per f32 parameter
    }

    fn set_quality(&mut self, quality: f32) -> Result<()> {
        self.config.quality = quality.clamp(0.0, 1.0);
        Ok(())
    }
}

impl FeedforwardModel {
    fn features_to_vector(&self, input: &NeuralInputFeatures) -> Vec<f32> {
        let mut vec = Vec::with_capacity(self.config.input_dim);

        // Position features (3D coordinates)
        vec.push(input.position.x);
        vec.push(input.position.y);
        vec.push(input.position.z);

        // Listener orientation (quaternion)
        vec.extend_from_slice(&input.listener_orientation);

        // Audio features
        vec.extend_from_slice(&input.audio_features);

        // Room features
        vec.extend_from_slice(&input.room_features);

        // HRTF features (if available)
        if let Some(ref hrtf_features) = input.hrtf_features {
            vec.extend_from_slice(hrtf_features);
        }

        // Temporal context
        vec.extend_from_slice(&input.temporal_context);

        // User features (if available)
        if let Some(ref user_features) = input.user_features {
            vec.extend_from_slice(user_features);
        }

        // Pad or truncate to match input_dim
        vec.resize(self.config.input_dim, 0.0);

        vec
    }

    fn tensor_to_binaural_audio(&self, output_data: &[f32]) -> Vec<Vec<f32>> {
        let samples_per_channel = self.config.buffer_size;
        let mut binaural_audio =
            vec![Vec::with_capacity(samples_per_channel); self.config.output_channels];

        for (i, &sample) in output_data.iter().enumerate() {
            let channel = i % self.config.output_channels;
            if binaural_audio[channel].len() < samples_per_channel {
                // Already bounded to [-1, 1] by the tanh applied in `forward_tensor`.
                binaural_audio[channel].push(sample);
            }
        }

        binaural_audio
    }

    fn estimate_confidence(&self, output_data: &[f32]) -> f32 {
        // Confidence estimation based on output signal characteristics
        if output_data.is_empty() {
            return 0.0;
        }

        // Calculate signal properties
        let mean = output_data.iter().sum::<f32>() / output_data.len() as f32;
        let variance =
            output_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / output_data.len() as f32;
        let std_dev = variance.sqrt();

        // Calculate signal-to-noise ratio estimate
        let signal_power =
            output_data.iter().map(|x| x.powi(2)).sum::<f32>() / output_data.len() as f32;
        let noise_estimate = std_dev.min(0.1); // Cap noise estimate
        let snr = if noise_estimate > 0.0 {
            (signal_power / noise_estimate.powi(2)).log10() * 10.0
        } else {
            30.0 // High SNR if no noise
        };

        // Calculate dynamic range
        let max_val = output_data
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
        let dynamic_range = if max_val > 0.0 { max_val } else { 0.1 };

        // Combine metrics for confidence score
        let snr_score = (snr / 30.0).clamp(0.0, 1.0); // Normalize SNR (30dB = 1.0)
        let dynamic_score = dynamic_range.clamp(0.0, 1.0);
        let stability_score = (1.0 - (std_dev / (max_val + 1e-6))).clamp(0.0, 1.0);

        // Weighted combination
        (0.4 * snr_score + 0.3 * dynamic_score + 0.3 * stability_score).clamp(0.0, 1.0)
    }
}

impl ConvolutionalModel {
    /// Create a new convolutional neural network model
    pub fn new(config: NeuralSpatialConfig, device: Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        // Create convolutional layers for temporal-spatial processing
        let mut conv_layers = Vec::new();
        let mut in_channels = 1; // Start with 1 input channel
        let conv_channels = vec![16, 32, 64]; // Increasing channel complexity

        for (i, &out_channels) in conv_channels.iter().enumerate() {
            let kernel_size = if i == 0 { 7 } else { 3 }; // Larger kernel for first layer
            let conv = candle_nn::conv1d(
                in_channels,
                out_channels,
                kernel_size,
                candle_nn::Conv1dConfig {
                    stride: 1,
                    padding: kernel_size / 2,
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
                vb.pp(format!("conv_{i}")),
            )?;
            conv_layers.push(conv);
            in_channels = out_channels;
        }

        // Create linear layers after convolutional feature extraction. The
        // flattened size is `final_channels * final_sequence_length`: the
        // channel count is exactly `in_channels` (the last conv layer's
        // output channels, tracked above), and the sequence length is the
        // *real*, exactly-predicted post-downsampling length (see
        // `conv_output_sequence_length`) - not a hand-derived formula that
        // can silently drift out of sync with `forward_tensor`'s actual
        // downsampling and break every forward pass with a shape mismatch.
        let mut linear_layers = Vec::new();
        let final_seq_len = conv_output_sequence_length(config.input_dim, conv_layers.len());
        let conv_output_size = in_channels * final_seq_len;
        let mut input_dim = conv_output_size;

        for &hidden_dim in &config.hidden_dims {
            linear_layers.push(candle_nn::linear(
                input_dim,
                hidden_dim,
                vb.pp(format!("linear_{}", linear_layers.len())),
            )?);
            input_dim = hidden_dim;
        }

        // Output layer for binaural audio
        let output_dim = config.output_channels * config.buffer_size;
        linear_layers.push(candle_nn::linear(input_dim, output_dim, vb.pp("output"))?);

        Ok(Self {
            config,
            conv_layers,
            linear_layers,
            device,
            metrics: NeuralPerformanceMetrics::default(),
            varmap,
        })
    }
}

impl NeuralModel for ConvolutionalModel {
    fn forward(&self, input: &NeuralInputFeatures) -> Result<NeuralSpatialOutput> {
        let input_tensor = self.input_tensor(input)?;
        let x = self.forward_tensor(&input_tensor)?;

        // Convert output tensor to binaural audio
        let output_data = x
            .to_vec2::<f32>()
            .map_err(|e| Error::LegacyProcessing(format!("Failed to extract output data: {e}")))?;

        let binaural_audio = self.tensor_to_binaural_audio(&output_data[0]);
        let confidence = self.estimate_confidence(&output_data[0]);

        Ok(NeuralSpatialOutput {
            binaural_audio,
            confidence,
            latency_ms: 0.0, // Will be set by processor
            quality_score: self.config.quality,
            metadata: HashMap::new(),
        })
    }

    fn config(&self) -> &NeuralSpatialConfig {
        &self.config
    }

    fn input_tensor(&self, input: &NeuralInputFeatures) -> Result<Tensor> {
        let input_vec = self.features_to_vector(input);
        let seq_len = input_vec.len();

        // Reshape input for 1D convolution: (batch_size, channels, sequence_length)
        Tensor::from_vec(input_vec, (1, 1, seq_len), &self.device)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to create input tensor: {e}")))
    }

    fn forward_tensor(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();

        // Apply convolutional layers with pooling
        for (i, conv_layer) in self.conv_layers.iter().enumerate() {
            x = conv_layer.forward(&x).map_err(|e| {
                Error::LegacyProcessing(format!("Conv layer {i} forward pass failed: {e}"))
            })?;

            // Apply ReLU activation
            x = x
                .relu()
                .map_err(|e| Error::LegacyProcessing(format!("ReLU activation failed: {e}")))?;

            // Apply simple stride-based downsampling instead of max pooling.
            // Note: Candle doesn't have max_pool1d, so we use strided
            // convolution approach. The `> CONV_DOWNSAMPLE_MIN_LEN` guard
            // must stay in lockstep with `conv_output_sequence_length`
            // (used by `ConvolutionalModel::new` to size the first `Linear`
            // layer) - see that function's doc comment.
            let current_shape = x.shape();
            if current_shape.dims().len() >= 3 && current_shape.dims()[2] > CONV_DOWNSAMPLE_MIN_LEN
            {
                // Simple downsampling by taking every 2nd element
                let indices: Vec<usize> = (0..current_shape.dims()[2]).step_by(2).collect();
                let indices_tensor = Tensor::from_vec(
                    indices.iter().map(|&i| i as u32).collect::<Vec<u32>>(),
                    (indices.len(),),
                    &self.device,
                )
                .map_err(|e| {
                    Error::LegacyProcessing(format!("Failed to create indices tensor: {e}"))
                })?;
                x = x
                    .index_select(&indices_tensor, 2)
                    .map_err(|e| Error::LegacyProcessing(format!("Downsampling failed: {e}")))?;
            }
        }

        // Flatten for linear layers
        let batch_size = x
            .dim(0)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to get batch dimension: {e}")))?;
        let flattened_size = x.elem_count() / batch_size;
        x = x
            .reshape((batch_size, flattened_size))
            .map_err(|e| Error::LegacyProcessing(format!("Failed to flatten tensor: {e}")))?;

        // Apply linear layers
        for (i, linear_layer) in self.linear_layers.iter().enumerate() {
            x = linear_layer.forward(&x).map_err(|e| {
                Error::LegacyProcessing(format!("Linear layer {i} forward pass failed: {e}"))
            })?;

            // Apply ReLU for hidden layers, no activation for output layer
            if i < self.linear_layers.len() - 1 {
                x = x
                    .relu()
                    .map_err(|e| Error::LegacyProcessing(format!("ReLU activation failed: {e}")))?;
            }
        }

        // Bound the output to valid audio sample range (see
        // `FeedforwardModel::forward_tensor` for why this lives here).
        x.tanh()
            .map_err(|e| Error::LegacyProcessing(format!("Output activation failed: {e}")))
    }

    fn trainable_vars(&self) -> Vec<Var> {
        self.varmap.all_vars()
    }

    fn update_parameters(&mut self, params: &HashMap<String, Tensor>) -> Result<()> {
        apply_parameter_updates(&mut self.varmap, params)?;

        self.metrics.last_updated = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(())
    }

    fn metrics(&self) -> NeuralPerformanceMetrics {
        self.metrics.clone()
    }

    fn save(&self, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let weights_path = format!("{path}.safetensors");
        self.varmap.save(&weights_path).map_err(|e| {
            Error::LegacyProcessing(format!(
                "Failed to save model weights to {weights_path}: {e}"
            ))
        })?;

        // Create comprehensive model save data structure
        let save_data = serde_json::json!({
            "model_type": "convolutional",
            "config": self.config,
            "conv_layers": {
                "count": self.conv_layers.len(),
                "filters": self.conv_layers.iter().enumerate().map(|(i, _)| {
                    format!("conv_layer_{i}")
                }).collect::<Vec<_>>()
            },
            "linear_layers": {
                "count": self.linear_layers.len(),
                "layers": self.linear_layers.iter().enumerate().map(|(i, _)| {
                    if i < self.linear_layers.len() - 1 {
                        format!("linear_{i}")
                    } else {
                        "output".to_string()
                    }
                }).collect::<Vec<_>>()
            },
            "metrics": self.metrics,
            "weights_path": weights_path,
            "saved_at": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            "version": "1.0"
        });

        // Write comprehensive model data
        let mut file = File::create(path)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to create model file: {e}")))?;

        file.write_all(save_data.to_string().as_bytes())
            .map_err(|e| Error::LegacyProcessing(format!("Failed to write model data: {e}")))?;

        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<()> {
        use std::fs;

        // Read the saved model file
        let model_data = fs::read_to_string(path).map_err(|e| {
            Error::LegacyProcessing(format!("Failed to read model file {path}: {e}"))
        })?;

        // Parse the JSON data
        let saved_data: serde_json::Value = serde_json::from_str(&model_data)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to parse model file: {e}")))?;

        // Validate model type
        let model_type = saved_data["model_type"].as_str().ok_or_else(|| {
            Error::LegacyProcessing("Missing model_type in saved file".to_string())
        })?;

        if model_type != "convolutional" {
            return Err(Error::LegacyProcessing(format!(
                "Model type mismatch: expected 'convolutional', found '{model_type}'"
            )));
        }

        // Load configuration
        let loaded_config: NeuralSpatialConfig =
            serde_json::from_value(saved_data["config"].clone()).map_err(|e| {
                Error::LegacyProcessing(format!("Failed to parse saved config: {e}"))
            })?;

        // Update current configuration
        self.config = loaded_config;

        // Load metrics if available
        if let Ok(loaded_metrics) =
            serde_json::from_value::<NeuralPerformanceMetrics>(saved_data["metrics"].clone())
        {
            self.metrics = loaded_metrics;
        }

        // Load the real weight tensors (see `FeedforwardModel::load`).
        let weights_path = saved_data["weights_path"]
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| format!("{path}.safetensors"));

        self.varmap.load(&weights_path).map_err(|e| {
            Error::LegacyProcessing(format!(
                "Failed to load model weights from {weights_path}: {e}"
            ))
        })?;

        Ok(())
    }

    fn memory_usage(&self) -> usize {
        varmap_param_count(&self.varmap) * 4 // 4 bytes per f32 parameter
    }

    fn set_quality(&mut self, quality: f32) -> Result<()> {
        self.config.quality = quality.clamp(0.0, 1.0);
        Ok(())
    }
}

impl ConvolutionalModel {
    fn features_to_vector(&self, input: &NeuralInputFeatures) -> Vec<f32> {
        let mut vec = Vec::with_capacity(self.config.input_dim);

        // Position features (3D coordinates)
        vec.push(input.position.x);
        vec.push(input.position.y);
        vec.push(input.position.z);

        // Listener orientation (quaternion)
        vec.extend_from_slice(&input.listener_orientation);

        // Audio features
        vec.extend_from_slice(&input.audio_features);

        // Room features
        vec.extend_from_slice(&input.room_features);

        // HRTF features (if available)
        if let Some(ref hrtf_features) = input.hrtf_features {
            vec.extend_from_slice(hrtf_features);
        }

        // Temporal context
        vec.extend_from_slice(&input.temporal_context);

        // User features (if available)
        if let Some(ref user_features) = input.user_features {
            vec.extend_from_slice(user_features);
        }

        // Pad or truncate to match input_dim
        vec.resize(self.config.input_dim, 0.0);

        vec
    }

    fn tensor_to_binaural_audio(&self, output_data: &[f32]) -> Vec<Vec<f32>> {
        let samples_per_channel = self.config.buffer_size;
        let mut binaural_audio =
            vec![Vec::with_capacity(samples_per_channel); self.config.output_channels];

        for (i, &sample) in output_data.iter().enumerate() {
            let channel = i % self.config.output_channels;
            if binaural_audio[channel].len() < samples_per_channel {
                // Already bounded to [-1, 1] by the tanh applied in `forward_tensor`.
                binaural_audio[channel].push(sample);
            }
        }

        binaural_audio
    }

    fn estimate_confidence(&self, output_data: &[f32]) -> f32 {
        // Confidence estimation based on output signal characteristics
        if output_data.is_empty() {
            return 0.0;
        }

        // Calculate signal properties
        let mean = output_data.iter().sum::<f32>() / output_data.len() as f32;
        let variance =
            output_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / output_data.len() as f32;
        let std_dev = variance.sqrt();

        // Calculate signal-to-noise ratio estimate
        let signal_power =
            output_data.iter().map(|x| x.powi(2)).sum::<f32>() / output_data.len() as f32;
        let noise_estimate = std_dev.min(0.1); // Cap noise estimate
        let snr = if noise_estimate > 0.0 {
            (signal_power / noise_estimate.powi(2)).log10() * 10.0
        } else {
            30.0 // High SNR if no noise
        };

        // Calculate dynamic range
        let max_val = output_data
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
        let dynamic_range = if max_val > 0.0 { max_val } else { 0.1 };

        // Combine metrics for confidence score
        let snr_score = (snr / 30.0).clamp(0.0, 1.0); // Normalize SNR (30dB = 1.0)
        let dynamic_score = dynamic_range.clamp(0.0, 1.0);
        let stability_score = (1.0 - (std_dev / (max_val + 1e-6))).clamp(0.0, 1.0);

        // Weighted combination
        (0.4 * snr_score + 0.3 * dynamic_score + 0.3 * stability_score).clamp(0.0, 1.0)
    }
}

impl TransformerModel {
    /// Create a new transformer neural network model
    pub fn new(config: NeuralSpatialConfig, device: Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        // Calculate attention dimensions
        let model_dim = *config.hidden_dims.first().unwrap_or(&512);
        let num_heads = 8;
        if !model_dim.is_multiple_of(num_heads) {
            return Err(Error::LegacyConfig(format!(
                "Transformer model dimension ({model_dim}) must be divisible by the number of \
                 attention heads ({num_heads}); adjust `hidden_dims`"
            )));
        }
        let head_dim = model_dim / num_heads;
        let ff_dim = model_dim * 4;

        // Create encoder with real, trainable attention/feed-forward/layer-norm weights.
        let encoder = TransformerEncoder {
            attention: MultiHeadAttention::new(
                num_heads,
                head_dim,
                model_dim,
                vb.pp("encoder.attention"),
            )?,
            feedforward: FeedForwardLayer::new(model_dim, ff_dim, vb.pp("encoder.ff"))?,
            norm1: LayerNorm::new(model_dim, vb.pp("encoder.norm1"))?,
            norm2: LayerNorm::new(model_dim, vb.pp("encoder.norm2"))?,
        };

        // Create decoder with its own independent set of weights.
        let decoder = TransformerDecoder {
            self_attention: MultiHeadAttention::new(
                num_heads,
                head_dim,
                model_dim,
                vb.pp("decoder.self_attention"),
            )?,
            cross_attention: MultiHeadAttention::new(
                num_heads,
                head_dim,
                model_dim,
                vb.pp("decoder.cross_attention"),
            )?,
            feedforward: FeedForwardLayer::new(model_dim, ff_dim, vb.pp("decoder.ff"))?,
            norm1: LayerNorm::new(model_dim, vb.pp("decoder.norm1"))?,
            norm2: LayerNorm::new(model_dim, vb.pp("decoder.norm2"))?,
            norm3: LayerNorm::new(model_dim, vb.pp("decoder.norm3"))?,
        };

        // Learned input/output projections, built once here (never regenerated
        // per forward call - see the `transformer-model-random-weights-per-call`
        // fix this replaces).
        let input_projection =
            candle_nn::linear(config.input_dim, model_dim, vb.pp("input_projection"))?;
        let output_dim = config.output_channels * config.buffer_size;
        let output_projection =
            candle_nn::linear(model_dim, output_dim, vb.pp("output_projection"))?;

        Ok(Self {
            config,
            encoder,
            decoder,
            input_projection,
            output_projection,
            device,
            metrics: NeuralPerformanceMetrics::default(),
            varmap,
        })
    }
}

impl NeuralModel for TransformerModel {
    fn forward(&self, input: &NeuralInputFeatures) -> Result<NeuralSpatialOutput> {
        let input_tensor = self.input_tensor(input)?;
        let output_tensor = self.forward_tensor(&input_tensor)?;

        // Convert to output format; shape is (1, seq_len, output_dim) with seq_len == 1.
        let output_data = output_tensor
            .to_vec3::<f32>()
            .map_err(|e| Error::LegacyProcessing(format!("Failed to extract output: {e}")))?;

        let flat_output = output_data[0][0].clone();
        let binaural_audio = self.tensor_to_binaural_audio(&flat_output);
        let confidence = self.estimate_confidence(&flat_output);

        Ok(NeuralSpatialOutput {
            binaural_audio,
            confidence,
            latency_ms: 0.0, // Will be set by processor
            quality_score: self.config.quality,
            metadata: HashMap::new(),
        })
    }

    fn config(&self) -> &NeuralSpatialConfig {
        &self.config
    }

    fn input_tensor(&self, input: &NeuralInputFeatures) -> Result<Tensor> {
        let input_vec = self.features_to_vector(input);
        let seq_len = 1; // Each call processes a single spatial-audio frame.
        let input_dim = input_vec.len();

        Tensor::from_vec(input_vec, (1, seq_len, input_dim), &self.device)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to create input tensor: {e}")))
    }

    fn forward_tensor(&self, input: &Tensor) -> Result<Tensor> {
        // Project the raw feature vector into the transformer's model dimension
        // using the learned (constructed-once, trained) projection layer.
        let projected = self
            .input_projection
            .forward(input)
            .map_err(|e| Error::LegacyProcessing(format!("Input projection failed: {e}")))?;

        // Real encoder/decoder forward passes (multi-head attention + feed-forward
        // + residual/layer-norm), see `encoder_forward`/`decoder_forward`.
        let encoded = self.encoder_forward(&projected)?;
        let decoded = self.decoder_forward(&encoded, &encoded)?;

        let output = self
            .output_projection
            .forward(&decoded)
            .map_err(|e| Error::LegacyProcessing(format!("Output projection failed: {e}")))?;

        // Bound the output to valid audio sample range (see
        // `FeedforwardModel::forward_tensor` for why this lives here).
        output
            .tanh()
            .map_err(|e| Error::LegacyProcessing(format!("Output activation failed: {e}")))
    }

    fn trainable_vars(&self) -> Vec<Var> {
        self.varmap.all_vars()
    }

    fn update_parameters(&mut self, params: &HashMap<String, Tensor>) -> Result<()> {
        apply_parameter_updates(&mut self.varmap, params)?;

        self.metrics.last_updated = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(())
    }

    fn metrics(&self) -> NeuralPerformanceMetrics {
        self.metrics.clone()
    }

    fn save(&self, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let model_dim = *self.config.hidden_dims.first().unwrap_or(&512);
        let num_heads = 8; // Fixed number of attention heads
        let ff_dim = model_dim * 4; // Standard transformer feedforward dimension

        let weights_path = format!("{path}.safetensors");
        self.varmap.save(&weights_path).map_err(|e| {
            Error::LegacyProcessing(format!(
                "Failed to save model weights to {weights_path}: {e}"
            ))
        })?;

        // Create comprehensive transformer model save data
        let save_data = serde_json::json!({
            "model_type": "transformer",
            "config": self.config,
            "architecture": {
                "model_dim": model_dim,
                "num_heads": num_heads,
                "ff_dim": ff_dim,
                "encoder_layers": 1,
                "decoder_layers": 1
            },
            "components": {
                "encoder": {
                    "attention": ["query", "key", "value", "output"],
                    "feedforward": ["linear1", "linear2"],
                    "layer_norms": ["norm1", "norm2"]
                },
                "decoder": {
                    "self_attention": ["query", "key", "value", "output"],
                    "cross_attention": ["query", "key", "value", "output"],
                    "feedforward": ["linear1", "linear2"],
                    "layer_norms": ["norm1", "norm2", "norm3"]
                }
            },
            "metrics": self.metrics,
            "parameter_count": varmap_param_count(&self.varmap),
            "weights_path": weights_path,
            "saved_at": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            "version": "1.0"
        });

        // Write comprehensive transformer model data
        let mut file = File::create(path)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to create model file: {e}")))?;

        file.write_all(save_data.to_string().as_bytes())
            .map_err(|e| Error::LegacyProcessing(format!("Failed to write model data: {e}")))?;

        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<()> {
        use std::fs;

        // Read the saved model file
        let model_data = fs::read_to_string(path).map_err(|e| {
            Error::LegacyProcessing(format!("Failed to read model file {path}: {e}"))
        })?;

        // Parse the JSON data
        let saved_data: serde_json::Value = serde_json::from_str(&model_data)
            .map_err(|e| Error::LegacyProcessing(format!("Failed to parse model file: {e}")))?;

        // Validate model type
        let model_type = saved_data["model_type"].as_str().ok_or_else(|| {
            Error::LegacyProcessing("Missing model_type in saved file".to_string())
        })?;

        if model_type != "transformer" {
            return Err(Error::LegacyProcessing(format!(
                "Model type mismatch: expected 'transformer', found '{model_type}'"
            )));
        }

        // Load configuration
        let loaded_config: NeuralSpatialConfig =
            serde_json::from_value(saved_data["config"].clone()).map_err(|e| {
                Error::LegacyProcessing(format!("Failed to parse saved config: {e}"))
            })?;

        // Update current configuration
        self.config = loaded_config;

        // Load metrics if available
        if let Ok(loaded_metrics) =
            serde_json::from_value::<NeuralPerformanceMetrics>(saved_data["metrics"].clone())
        {
            self.metrics = loaded_metrics;
        }

        // Load the real weight tensors (see `FeedforwardModel::load`). This is
        // also where an architecture mismatch (e.g. a different `hidden_dims`
        // producing a different `model_dim`) is caught: the shapes registered
        // in `self.varmap` at construction time won't match the saved
        // tensors, and `VarMap::load` fails with a descriptive error rather
        // than silently loading nothing.
        let weights_path = saved_data["weights_path"]
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| format!("{path}.safetensors"));

        self.varmap.load(&weights_path).map_err(|e| {
            Error::LegacyProcessing(format!(
                "Failed to load model weights from {weights_path}: {e}"
            ))
        })?;

        Ok(())
    }

    fn memory_usage(&self) -> usize {
        varmap_param_count(&self.varmap) * 4 // 4 bytes per f32 parameter
    }

    fn set_quality(&mut self, quality: f32) -> Result<()> {
        self.config.quality = quality.clamp(0.0, 1.0);
        Ok(())
    }
}

impl TransformerModel {
    fn features_to_vector(&self, input: &NeuralInputFeatures) -> Vec<f32> {
        let mut vec = Vec::with_capacity(self.config.input_dim);

        // Position features (3D coordinates)
        vec.push(input.position.x);
        vec.push(input.position.y);
        vec.push(input.position.z);

        // Listener orientation (quaternion)
        vec.extend_from_slice(&input.listener_orientation);

        // Audio features
        vec.extend_from_slice(&input.audio_features);

        // Room features
        vec.extend_from_slice(&input.room_features);

        // HRTF features (if available)
        if let Some(ref hrtf_features) = input.hrtf_features {
            vec.extend_from_slice(hrtf_features);
        }

        // Temporal context
        vec.extend_from_slice(&input.temporal_context);

        // User features (if available)
        if let Some(ref user_features) = input.user_features {
            vec.extend_from_slice(user_features);
        }

        // Pad or truncate to match input_dim
        vec.resize(self.config.input_dim, 0.0);

        vec
    }

    fn tensor_to_binaural_audio(&self, output_data: &[f32]) -> Vec<Vec<f32>> {
        let samples_per_channel = self.config.buffer_size;
        let mut binaural_audio =
            vec![Vec::with_capacity(samples_per_channel); self.config.output_channels];

        for (i, &sample) in output_data.iter().enumerate() {
            let channel = i % self.config.output_channels;
            if binaural_audio[channel].len() < samples_per_channel {
                // Already bounded to [-1, 1] by the tanh applied in `forward_tensor`.
                binaural_audio[channel].push(sample);
            }
        }

        binaural_audio
    }

    fn estimate_confidence(&self, output_data: &[f32]) -> f32 {
        // Confidence estimation based on output signal characteristics
        if output_data.is_empty() {
            return 0.0;
        }

        // Calculate signal properties
        let mean = output_data.iter().sum::<f32>() / output_data.len() as f32;
        let variance =
            output_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / output_data.len() as f32;
        let std_dev = variance.sqrt();

        // Calculate signal-to-noise ratio estimate
        let signal_power =
            output_data.iter().map(|x| x.powi(2)).sum::<f32>() / output_data.len() as f32;
        let noise_estimate = std_dev.min(0.1); // Cap noise estimate
        let snr = if noise_estimate > 0.0 {
            (signal_power / noise_estimate.powi(2)).log10() * 10.0
        } else {
            30.0 // High SNR if no noise
        };

        // Calculate dynamic range
        let max_val = output_data
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
        let dynamic_range = if max_val > 0.0 { max_val } else { 0.1 };

        // Combine metrics for confidence score
        let snr_score = (snr / 30.0).clamp(0.0, 1.0); // Normalize SNR (30dB = 1.0)
        let dynamic_score = dynamic_range.clamp(0.0, 1.0);
        let stability_score = (1.0 - (std_dev / (max_val + 1e-6))).clamp(0.0, 1.0);

        // Weighted combination
        (0.4 * snr_score + 0.3 * dynamic_score + 0.3 * stability_score).clamp(0.0, 1.0)
    }

    /// Real transformer encoder forward pass: multi-head self-attention with a
    /// residual connection and layer norm, followed by a position-wise
    /// feed-forward block with its own residual connection and layer norm.
    fn encoder_forward(&self, input: &Tensor) -> Result<Tensor> {
        let attn_out = self.encoder.attention.forward(input, input)?;
        let residual1 = (input + &attn_out)
            .map_err(|e| Error::LegacyProcessing(format!("Encoder residual add failed: {e}")))?;
        let normed1 = self.encoder.norm1.forward(&residual1)?;

        let ff_out = self.encoder.feedforward.forward(&normed1)?;
        let residual2 = (&normed1 + &ff_out)
            .map_err(|e| Error::LegacyProcessing(format!("Encoder residual add failed: {e}")))?;
        let normed2 = self.encoder.norm2.forward(&residual2)?;

        Ok(normed2)
    }

    /// Real transformer decoder forward pass: masked-free self-attention
    /// (there is no autoregressive target in this single-frame architecture),
    /// cross-attention against the encoder output, and a feed-forward block -
    /// each with its own residual connection and layer norm.
    fn decoder_forward(&self, encoder_output: &Tensor, decoder_input: &Tensor) -> Result<Tensor> {
        let self_attn = self
            .decoder
            .self_attention
            .forward(decoder_input, decoder_input)?;
        let residual1 = (decoder_input + &self_attn)
            .map_err(|e| Error::LegacyProcessing(format!("Decoder residual add failed: {e}")))?;
        let normed1 = self.decoder.norm1.forward(&residual1)?;

        let cross_attn = self
            .decoder
            .cross_attention
            .forward(&normed1, encoder_output)?;
        let residual2 = (&normed1 + &cross_attn)
            .map_err(|e| Error::LegacyProcessing(format!("Decoder residual add failed: {e}")))?;
        let normed2 = self.decoder.norm2.forward(&residual2)?;

        let ff_out = self.decoder.feedforward.forward(&normed2)?;
        let residual3 = (&normed2 + &ff_out)
            .map_err(|e| Error::LegacyProcessing(format!("Decoder residual add failed: {e}")))?;
        let normed3 = self.decoder.norm3.forward(&residual3)?;

        Ok(normed3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Position3D;

    fn tiny_feedforward_config() -> NeuralSpatialConfig {
        NeuralSpatialConfig {
            model_type: NeuralModelType::Feedforward,
            hidden_dims: vec![8],
            input_dim: 8,
            output_channels: 1,
            sample_rate: 48000,
            buffer_size: 4,
            use_gpu: false,
            quality: 0.8,
            realtime_constraints: RealtimeConstraints::default(),
            training_config: None,
        }
    }

    fn tiny_transformer_config() -> NeuralSpatialConfig {
        NeuralSpatialConfig {
            model_type: NeuralModelType::Transformer,
            hidden_dims: vec![8], // model_dim=8, divisible by the fixed 8 attention heads
            input_dim: 6,
            output_channels: 1,
            sample_rate: 48000,
            buffer_size: 4,
            use_gpu: false,
            quality: 0.8,
            realtime_constraints: RealtimeConstraints::default(),
            training_config: None,
        }
    }

    fn tiny_convolutional_config(input_dim: usize) -> NeuralSpatialConfig {
        NeuralSpatialConfig {
            model_type: NeuralModelType::Convolutional,
            hidden_dims: vec![8],
            input_dim,
            output_channels: 1,
            sample_rate: 48000,
            buffer_size: 4,
            use_gpu: false,
            quality: 0.8,
            realtime_constraints: RealtimeConstraints::default(),
            training_config: None,
        }
    }

    fn sample_input() -> NeuralInputFeatures {
        NeuralInputFeatures {
            position: Position3D::new(0.5, -0.3, 0.2),
            listener_orientation: [1.0, 0.0, 0.0, 0.0],
            audio_features: Vec::new(),
            room_features: Vec::new(),
            hrtf_features: None,
            temporal_context: Vec::new(),
            user_features: None,
        }
    }

    /// Build a unique temporary file path (never a hardcoded absolute path)
    /// for save/load round-trip tests.
    fn temp_model_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "voirs_spatial_test_{tag}_{}_{}.json",
            std::process::id(),
            fastrand::u64(..)
        ))
    }

    // ---- TransformerModel: no more `Tensor::randn` per forward call ----

    #[test]
    fn test_transformer_forward_is_deterministic() {
        let model = TransformerModel::new(tiny_transformer_config(), Device::Cpu)
            .expect("model construction");
        let input = sample_input();

        let first = model.forward(&input).expect("first forward pass");
        let second = model.forward(&input).expect("second forward pass");

        assert_eq!(
            first.binaural_audio, second.binaural_audio,
            "TransformerModel::forward must be deterministic for fixed weights and input \
             (regression test for the old per-call Tensor::randn projection bug)"
        );
    }

    #[test]
    fn test_transformer_output_depends_on_its_parameters() {
        // If the encoder/decoder still bypassed real attention with a bare
        // ReLU/add passthrough, updating the attention weights would have no
        // effect on the output. With real attention wired in, changing a
        // weight must change the result.
        let mut model = TransformerModel::new(tiny_transformer_config(), Device::Cpu)
            .expect("model construction");
        let input = sample_input();

        let before = model.forward(&input).expect("forward before update");

        // Perturb every real, registered parameter by a large constant offset
        // and apply it through the public `update_parameters` API.
        let real_names = model_var_names(&model);
        assert!(
            !real_names.is_empty(),
            "transformer must have real parameters"
        );
        let mut real_params = HashMap::new();
        for name in &real_names {
            let current = model
                .varmap
                .data()
                .lock()
                .expect("varmap lock")
                .get(name)
                .expect("named var exists")
                .as_tensor()
                .clone();
            let perturbed = (current + 10.0).expect("perturb tensor");
            real_params.insert(name.clone(), perturbed);
        }
        model
            .update_parameters(&real_params)
            .expect("update_parameters should apply real tensors");

        let after = model.forward(&input).expect("forward after update");
        assert_ne!(
            before.binaural_audio, after.binaural_audio,
            "output must change when the model's real parameters change"
        );
    }

    /// Helper exposing every parameter name currently registered in a
    /// model's varmap (test-only introspection).
    fn model_var_names(model: &TransformerModel) -> Vec<String> {
        model
            .varmap
            .data()
            .lock()
            .expect("varmap lock")
            .keys()
            .cloned()
            .collect()
    }

    // ---- update_parameters: real assignment, shape-validated ----

    #[test]
    fn test_update_parameters_rejects_unknown_name() {
        let mut model = FeedforwardModel::new(tiny_feedforward_config(), Device::Cpu)
            .expect("model construction");
        let mut params = HashMap::new();
        params.insert(
            "does_not_exist.weight".to_string(),
            Tensor::zeros((8, 8), candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        let result = model.update_parameters(&params);
        assert!(
            result.is_err(),
            "updating an unknown parameter name must fail, not silently succeed"
        );
    }

    #[test]
    fn test_update_parameters_rejects_shape_mismatch() {
        let mut model = FeedforwardModel::new(tiny_feedforward_config(), Device::Cpu)
            .expect("model construction");
        let mut params = HashMap::new();
        // "layer_0.weight" is real (registered at construction) but the wrong shape.
        params.insert(
            "layer_0.weight".to_string(),
            Tensor::zeros((2, 2), candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        let result = model.update_parameters(&params);
        assert!(
            result.is_err(),
            "a shape-mismatched tensor must be rejected, not silently accepted"
        );
    }

    #[test]
    fn test_update_parameters_actually_changes_weights() {
        let mut model = FeedforwardModel::new(tiny_feedforward_config(), Device::Cpu)
            .expect("model construction");
        let sum_before: f32 = model
            .trainable_vars()
            .iter()
            .map(|v| v.sum_all().unwrap().to_scalar::<f32>().unwrap())
            .sum();

        let mut params = HashMap::new();
        params.insert(
            "output.bias".to_string(),
            Tensor::ones(4, candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        model
            .update_parameters(&params)
            .expect("update_parameters with a real, correctly-shaped tensor must succeed");

        let sum_after: f32 = model
            .trainable_vars()
            .iter()
            .map(|v| v.sum_all().unwrap().to_scalar::<f32>().unwrap())
            .sum();

        assert_ne!(
            sum_before, sum_after,
            "update_parameters must really write into the model's weights, not just print progress"
        );
    }

    // ---- save/load: real safetensors round-trip ----

    #[test]
    fn test_feedforward_save_load_round_trip_restores_real_weights() {
        let mut model = FeedforwardModel::new(tiny_feedforward_config(), Device::Cpu)
            .expect("model construction");

        // Mutate the model away from its construction-time initialization so
        // the round trip can't accidentally "succeed" by comparing two
        // freshly-initialized (but different) models.
        let mut params = HashMap::new();
        params.insert(
            "output.bias".to_string(),
            Tensor::ones(4, candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        model.update_parameters(&params).expect("perturb weights");

        let input = sample_input();
        let expected = model.forward(&input).expect("forward on trained model");

        let path = temp_model_path("feedforward_roundtrip");
        model.save(path.to_str().unwrap()).expect("save");

        let mut reloaded = FeedforwardModel::new(tiny_feedforward_config(), Device::Cpu)
            .expect("fresh model construction");
        reloaded
            .load(path.to_str().unwrap())
            .expect("load should restore the real saved weights");

        let actual = reloaded.forward(&input).expect("forward on reloaded model");
        assert_eq!(
            expected.binaural_audio, actual.binaural_audio,
            "loading a saved model must restore bit-identical weights, not just metadata"
        );

        // Clean up both the metadata file and the safetensors weights file.
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{}.safetensors", path.to_str().unwrap()));
    }

    #[test]
    fn test_load_missing_file_fails_closed() {
        let mut model = FeedforwardModel::new(tiny_feedforward_config(), Device::Cpu)
            .expect("model construction");
        let missing_path = temp_model_path("does_not_exist");
        let result = model.load(missing_path.to_str().unwrap());
        assert!(
            result.is_err(),
            "loading a nonexistent model file must return an error, never a fake success"
        );
    }

    #[test]
    fn test_transformer_save_load_round_trip_restores_real_weights() {
        let mut model = TransformerModel::new(tiny_transformer_config(), Device::Cpu)
            .expect("model construction");

        let real_names = model_var_names(&model);
        let mut real_params = HashMap::new();
        for name in &real_names {
            let current = model
                .varmap
                .data()
                .lock()
                .expect("varmap lock")
                .get(name)
                .expect("named var exists")
                .as_tensor()
                .clone();
            let perturbed = (current + 3.0).expect("perturb tensor");
            real_params.insert(name.clone(), perturbed);
        }
        model
            .update_parameters(&real_params)
            .expect("perturb transformer weights");

        let input = sample_input();
        let expected = model.forward(&input).expect("forward on perturbed model");

        let path = temp_model_path("transformer_roundtrip");
        model.save(path.to_str().unwrap()).expect("save");

        let mut reloaded = TransformerModel::new(tiny_transformer_config(), Device::Cpu)
            .expect("fresh model construction");
        reloaded
            .load(path.to_str().unwrap())
            .expect("load should restore the real saved weights");

        let actual = reloaded.forward(&input).expect("forward on reloaded model");
        assert_eq!(
            expected.binaural_audio, actual.binaural_audio,
            "loading a saved transformer must restore bit-identical weights"
        );

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{}.safetensors", path.to_str().unwrap()));
    }

    // ---- memory_usage: exact, not a hand-derived estimate ----

    #[test]
    fn test_memory_usage_matches_exact_parameter_count() {
        let model = FeedforwardModel::new(tiny_feedforward_config(), Device::Cpu)
            .expect("model construction");
        let exact: usize = model
            .trainable_vars()
            .iter()
            .map(|v| v.elem_count())
            .sum::<usize>()
            * 4;
        assert_eq!(model.memory_usage(), exact);
        assert!(exact > 0);
    }

    // ---- ConvolutionalModel: real flattened-size prediction, not a
    // hand-derived `/4` formula that silently drifts out of sync with the
    // actual conv+downsample stack and breaks every forward pass ----

    #[test]
    fn test_conv_output_sequence_length_matches_actual_downsampling() {
        // Three downsampling stages (the fixed `conv_channels = [16, 32, 64]`
        // in `ConvolutionalModel::new`), covering both the general
        // ceil-halving case and the "already at the floor" edge case.
        assert_eq!(conv_output_sequence_length(32, 3), 4); // 32->16->8->4
        assert_eq!(conv_output_sequence_length(16, 3), 2); // 16->8->4->2
        assert_eq!(conv_output_sequence_length(8, 3), 2); // 8->4->2->2 (floor)
        assert_eq!(conv_output_sequence_length(1, 3), 1); // already <= floor
        assert_eq!(conv_output_sequence_length(100, 3), 13); // 100->50->25->13
    }

    #[test]
    fn test_convolutional_forward_succeeds_across_input_dims() {
        // Regression test: `conv_output_size` used to be a hand-derived
        // `64 * (input_dim / 4)` formula that assumed a flat 4x downsample,
        // but the real three-stage stack downsamples by up to 8x - so the
        // first `Linear` layer was built with the wrong input dimension and
        // every forward pass failed with a matmul shape-mismatch error for
        // most `input_dim` values (verified: input_dim=16/32/64 all failed
        // before this fix; only the coincidental input_dim=8 case happened
        // to work). This must now succeed across a range of realistic sizes.
        for input_dim in [8usize, 13, 16, 31, 32, 64, 100] {
            let model = ConvolutionalModel::new(tiny_convolutional_config(input_dim), Device::Cpu)
                .unwrap_or_else(|e| {
                    panic!("model construction failed for input_dim={input_dim}: {e}")
                });
            let input = sample_input();
            let result = model.forward(&input);
            assert!(
                result.is_ok(),
                "forward() failed for input_dim={input_dim}: {result:?}"
            );
        }
    }

    #[test]
    fn test_convolutional_forward_is_deterministic() {
        let model = ConvolutionalModel::new(tiny_convolutional_config(32), Device::Cpu)
            .expect("model construction");
        let input = sample_input();

        let first = model.forward(&input).expect("first forward pass");
        let second = model.forward(&input).expect("second forward pass");

        assert_eq!(
            first.binaural_audio, second.binaural_audio,
            "ConvolutionalModel::forward must be deterministic for fixed weights and input"
        );
    }

    #[test]
    fn test_convolutional_update_parameters_actually_changes_weights() {
        let mut model = ConvolutionalModel::new(tiny_convolutional_config(32), Device::Cpu)
            .expect("model construction");
        let sum_before: f32 = model
            .trainable_vars()
            .iter()
            .map(|v| v.sum_all().unwrap().to_scalar::<f32>().unwrap())
            .sum();

        let mut params = HashMap::new();
        params.insert(
            "output.bias".to_string(),
            Tensor::ones(4, candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        model
            .update_parameters(&params)
            .expect("update_parameters with a real, correctly-shaped tensor must succeed");

        let sum_after: f32 = model
            .trainable_vars()
            .iter()
            .map(|v| v.sum_all().unwrap().to_scalar::<f32>().unwrap())
            .sum();

        assert_ne!(
            sum_before, sum_after,
            "update_parameters must really write into the model's weights"
        );
    }

    #[test]
    fn test_convolutional_save_load_round_trip_restores_real_weights() {
        let mut model = ConvolutionalModel::new(tiny_convolutional_config(32), Device::Cpu)
            .expect("model construction");

        let mut params = HashMap::new();
        params.insert(
            "output.bias".to_string(),
            Tensor::ones(4, candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        model.update_parameters(&params).expect("perturb weights");

        let input = sample_input();
        let expected = model.forward(&input).expect("forward on trained model");

        let path = temp_model_path("convolutional_roundtrip");
        model.save(path.to_str().unwrap()).expect("save");

        let mut reloaded = ConvolutionalModel::new(tiny_convolutional_config(32), Device::Cpu)
            .expect("fresh model construction");
        reloaded
            .load(path.to_str().unwrap())
            .expect("load should restore the real saved weights");

        let actual = reloaded.forward(&input).expect("forward on reloaded model");
        assert_eq!(
            expected.binaural_audio, actual.binaural_audio,
            "loading a saved convolutional model must restore bit-identical weights"
        );

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{}.safetensors", path.to_str().unwrap()));
    }
}
