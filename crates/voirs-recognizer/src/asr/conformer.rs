//! Conformer: Convolution-augmented Transformer for Speech Recognition
//!
//! This module implements the Conformer architecture which combines the strengths
//! of convolutional neural networks and transformers for speech recognition.
//!
//! Reference: "Conformer: Convolution-augmented Transformer for Speech Recognition"
//! by Anmol Gulati et al. (<https://arxiv.org/abs/2005.08100>)

use crate::integration::PipelineResult;
use crate::traits::{
    ASRConfig, ASRFeature, ASRMetadata, ASRModel, AudioStream, Transcript, TranscriptChunk,
    TranscriptStream,
};
use crate::{RecognitionError, VoirsError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use voirs_sdk::{AudioBuffer, LanguageCode};

/// Load one feed-forward network's parameters from a checkpoint.
fn load_feed_forward(
    network: &mut FeedForwardNetwork,
    header: &crate::asr::weights::SafetensorsHeader,
    prefix: &str,
    slot: &str,
    model_dim: usize,
    hidden_dim: usize,
) -> Result<(), RecognitionError> {
    network.linear1_weights = header.read_matrix(
        &format!("{prefix}.{slot}.linear1.weight"),
        hidden_dim,
        model_dim,
    )?;
    network.linear1_bias =
        header.read_vector(&format!("{prefix}.{slot}.linear1.bias"), hidden_dim)?;
    network.linear2_weights = header.read_matrix(
        &format!("{prefix}.{slot}.linear2.weight"),
        model_dim,
        hidden_dim,
    )?;
    network.linear2_bias =
        header.read_vector(&format!("{prefix}.{slot}.linear2.bias"), model_dim)?;
    Ok(())
}

/// Convert a frequency in Hz to the mel scale (HTK formula).
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert a mel-scale value back to Hz (HTK formula).
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Build a triangular mel filterbank spanning 0 Hz to Nyquist.
///
/// Returns `n_mels` filters, each holding one weight per FFT bin.
fn mel_filterbank(n_mels: usize, n_bins: usize, sample_rate: f32) -> Vec<Vec<f32>> {
    let mut filters = vec![vec![0.0_f32; n_bins]; n_mels];
    if n_mels == 0 || n_bins < 2 {
        return filters;
    }

    let nyquist = sample_rate / 2.0;
    let mel_max = hz_to_mel(nyquist);

    // n_mels + 2 equally spaced mel points give n_mels overlapping triangles.
    #[allow(clippy::cast_precision_loss)]
    let points: Vec<f32> = (0..n_mels + 2)
        .map(|i| {
            let mel = mel_max * i as f32 / (n_mels + 1) as f32;
            mel_to_hz(mel)
        })
        .collect();

    #[allow(clippy::cast_precision_loss)]
    let bin_width = nyquist / (n_bins - 1) as f32;

    for (m, filter) in filters.iter_mut().enumerate() {
        let (left, center, right) = (points[m], points[m + 1], points[m + 2]);
        for (bin, weight) in filter.iter_mut().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let freq = bin as f32 * bin_width;
            if freq > left && freq < center && center > left {
                *weight = (freq - left) / (center - left);
            } else if freq >= center && freq < right && right > center {
                *weight = (right - freq) / (right - center);
            }
        }
    }

    filters
}

/// Numerically stable in-place softmax over a slice of scores.
fn softmax_in_place(scores: &mut [f32]) {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        let uniform = if scores.is_empty() {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            {
                1.0 / scores.len() as f32
            }
        };
        scores.fill(uniform);
        return;
    }
    let mut sum = 0.0_f32;
    for score in scores.iter_mut() {
        *score = (*score - max).exp();
        sum += *score;
    }
    if sum > 0.0 {
        for score in scores.iter_mut() {
            *score /= sum;
        }
    }
}

/// Logistic sigmoid.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Multiply every row of `input` by `weights` transposed.
///
/// `weights` is stored as `weights[output_unit][input_unit]`, so the result of a row
/// `x` is the vector `y` with `y[o] = Σ_i weights[o][i] * x[i]`.
///
/// # Errors
/// Returns [`RecognitionError::ModelError`] when a weight row is shorter than the input
/// frames, which would silently truncate the dot product.
fn matmul_rows(
    input: &[Vec<f32>],
    weights: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, RecognitionError> {
    let Some(first) = input.first() else {
        return Ok(Vec::new());
    };
    let input_dim = first.len();

    if let Some(row) = weights.iter().find(|row| row.len() != input_dim) {
        return Err(RecognitionError::ModelError {
            message: format!(
                "Weight row has {} columns but the input frames have {input_dim}",
                row.len()
            ),
            source: None,
        });
    }

    let mut output = Vec::with_capacity(input.len());
    for frame in input {
        if frame.len() != input_dim {
            return Err(RecognitionError::ModelError {
                message: format!(
                    "Ragged input sequence: frame with {} values among frames of {input_dim}",
                    frame.len()
                ),
                source: None,
            });
        }
        let mut row_out = Vec::with_capacity(weights.len());
        for weight_row in weights {
            let mut sum = 0.0_f32;
            for (w, x) in weight_row.iter().zip(frame.iter()) {
                sum += w * x;
            }
            row_out.push(sum);
        }
        output.push(row_out);
    }

    Ok(output)
}

/// Conformer model configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Conformer Config
pub struct ConformerConfig {
    /// Number of encoder blocks
    pub num_blocks: usize,
    /// Encoder dimension
    pub encoder_dim: usize,
    /// Number of attention heads
    pub attention_heads: usize,
    /// Feed-forward dimension
    pub feed_forward_dim: usize,
    /// Convolution kernel size
    pub conv_kernel_size: usize,
    /// Dropout rate
    pub dropout_rate: f32,
    /// Input feature dimensions (mel-spectrogram)
    pub input_dim: usize,
    /// Vocabulary size for output projection
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_seq_length: usize,
    /// Whether to use relative positional encoding
    pub use_relative_positional_encoding: bool,
    /// Macaron-style feed-forward factor
    pub macaron_style: bool,
    /// Convolution module activation
    pub conv_activation: ActivationType,
}

impl Default for ConformerConfig {
    fn default() -> Self {
        Self {
            num_blocks: 16,
            encoder_dim: 512,
            attention_heads: 8,
            feed_forward_dim: 2048,
            conv_kernel_size: 31,
            dropout_rate: 0.1,
            input_dim: 80,
            vocab_size: 5000,
            max_seq_length: 5000,
            use_relative_positional_encoding: true,
            macaron_style: true,
            conv_activation: ActivationType::Swish,
        }
    }
}

/// Activation function types for Conformer
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Activation Type
pub enum ActivationType {
    /// Re l u
    ReLU,
    /// G e l u
    GELU,
    /// Swish
    Swish,
    /// G l u
    GLU,
}

/// Multi-head attention configuration
#[derive(Debug, Clone)]
/// Multi Head Attention Config
pub struct MultiHeadAttentionConfig {
    /// num heads
    pub num_heads: usize,
    /// head dim
    pub head_dim: usize,
    /// dropout rate
    pub dropout_rate: f32,
    /// use relative positional encoding
    pub use_relative_positional_encoding: bool,
}

/// Convolution module configuration
#[derive(Debug, Clone)]
/// Convolution Config
pub struct ConvolutionConfig {
    /// kernel size
    pub kernel_size: usize,
    /// activation
    pub activation: ActivationType,
    /// dropout rate
    pub dropout_rate: f32,
}

/// Feed-forward network configuration
#[derive(Debug, Clone)]
/// Feed Forward Config
pub struct FeedForwardConfig {
    /// hidden dim
    pub hidden_dim: usize,
    /// dropout rate
    pub dropout_rate: f32,
    /// activation
    pub activation: ActivationType,
}

/// Where a [`ConformerModel`]'s parameters came from.
///
/// This is what separates a runnable model from an architecture skeleton. Transcription
/// is only permitted for [`ConformerWeightSource::Checkpoint`]: a randomly initialised
/// network has no learned representation, so any text decoded from it would be noise
/// dressed up as a transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConformerWeightSource {
    /// Parameters were drawn from a Xavier/Glorot distribution, never trained.
    RandomInit,
    /// Parameters were loaded from a real checkpoint at this path.
    Checkpoint {
        /// Path the parameters were read from.
        path: std::path::PathBuf,
        /// Real number of scalar parameters loaded.
        parameter_count: usize,
    },
}

impl ConformerWeightSource {
    /// Whether these parameters are trained and therefore usable for transcription.
    #[must_use]
    pub fn is_trained(&self) -> bool {
        matches!(self, Self::Checkpoint { .. })
    }
}

/// Conformer encoder block
#[derive(Debug, Clone)]
/// Conformer Block
pub struct ConformerBlock {
    /// Multi-head self-attention
    attention: MultiHeadAttention,
    /// Convolution module
    convolution: ConvolutionModule,
    /// Feed-forward networks (macaron-style has two)
    feed_forward_1: FeedForwardNetwork,
    feed_forward_2: Option<FeedForwardNetwork>,
    /// Layer normalization layers
    layer_norm_1: LayerNormalization,
    layer_norm_2: LayerNormalization,
    layer_norm_3: LayerNormalization,
    layer_norm_4: Option<LayerNormalization>,
    /// Dropout
    dropout_rate: f32,
}

/// Multi-head self-attention module
#[derive(Debug, Clone)]
/// Multi Head Attention
pub struct MultiHeadAttention {
    config: MultiHeadAttentionConfig,
    /// Query, Key, Value weight matrices
    query_weights: Vec<Vec<f32>>,
    key_weights: Vec<Vec<f32>>,
    value_weights: Vec<Vec<f32>>,
    /// Output projection weights
    output_weights: Vec<Vec<f32>>,
    /// Relative positional encoding parameters
    relative_position_bias: Option<Vec<Vec<f32>>>,
}

/// Convolution module for local feature extraction
#[derive(Debug, Clone)]
/// Convolution Module
pub struct ConvolutionModule {
    config: ConvolutionConfig,
    /// Pointwise convolution 1
    pointwise_conv1_weights: Vec<Vec<f32>>,
    /// Depthwise convolution
    depthwise_conv_weights: Vec<Vec<f32>>,
    /// Pointwise convolution 2
    pointwise_conv2_weights: Vec<Vec<f32>>,
    /// Batch normalization parameters
    batch_norm_gamma: Vec<f32>,
    batch_norm_beta: Vec<f32>,
    /// GLU (Gated Linear Unit) weights
    glu_weights: Option<Vec<Vec<f32>>>,
}

/// Feed-forward network
#[derive(Debug, Clone)]
/// Feed Forward Network
pub struct FeedForwardNetwork {
    config: FeedForwardConfig,
    /// Linear transformation weights
    linear1_weights: Vec<Vec<f32>>,
    linear2_weights: Vec<Vec<f32>>,
    /// Bias terms
    linear1_bias: Vec<f32>,
    linear2_bias: Vec<f32>,
}

/// Layer normalization
#[derive(Debug, Clone)]
/// Layer Normalization
pub struct LayerNormalization {
    /// Learnable scale parameter
    gamma: Vec<f32>,
    /// Learnable shift parameter
    beta: Vec<f32>,
    /// Small constant for numerical stability
    eps: f32,
}

/// Positional encoding for Conformer
#[derive(Debug, Clone)]
/// Positional Encoding
pub struct PositionalEncoding {
    /// Maximum sequence length
    max_length: usize,
    /// Model dimension
    d_model: usize,
    /// Pre-computed positional encodings
    encodings: Vec<Vec<f32>>,
}

/// Conformer model implementation
#[derive(Clone)]
pub struct ConformerModel {
    /// Model configuration
    config: ConformerConfig,
    /// Input feature projection
    input_projection: Vec<Vec<f32>>,
    /// Positional encoding
    positional_encoding: PositionalEncoding,
    /// Conformer blocks
    blocks: Vec<ConformerBlock>,
    /// Output projection to vocabulary
    output_projection: Vec<Vec<f32>>,
    /// Model statistics
    stats: Arc<RwLock<ConformerStats>>,
    /// Supported languages
    supported_languages: Vec<LanguageCode>,
    /// Provenance of the parameters currently held by this model
    weight_source: ConformerWeightSource,
}

/// Statistics and metrics for Conformer model
#[derive(Debug, Default, Clone)]
/// Conformer Stats
pub struct ConformerStats {
    /// Total inference count
    pub inference_count: u64,
    /// Total processing time
    pub total_processing_time_ms: u64,
    /// Average processing time per inference
    pub avg_processing_time_ms: f64,
    /// Number of successful inferences
    pub successful_inferences: u64,
    /// Number of failed inferences
    pub failed_inferences: u64,
    /// Model accuracy metrics
    pub accuracy_metrics: AccuracyMetrics,
}

/// Accuracy metrics for the model
#[derive(Debug, Default, Clone)]
/// Accuracy Metrics
pub struct AccuracyMetrics {
    /// Word Error Rate (WER)
    pub word_error_rate: f32,
    /// Character Error Rate (CER)
    pub character_error_rate: f32,
    /// Confidence scores
    pub average_confidence: f32,
    /// Language detection accuracy
    pub language_detection_accuracy: f32,
}

impl ConformerModel {
    /// Create a new Conformer model with default configuration
    pub async fn new() -> Result<Self, RecognitionError> {
        Self::with_config(ConformerConfig::default()).await
    }

    /// Create a new Conformer model with custom configuration
    pub async fn with_config(config: ConformerConfig) -> Result<Self, RecognitionError> {
        tracing::info!(
            "Initializing Conformer model with {} blocks",
            config.num_blocks
        );

        let input_projection = Self::initialize_input_projection(&config);
        let positional_encoding =
            PositionalEncoding::new(config.max_seq_length, config.encoder_dim);
        let blocks = Self::initialize_conformer_blocks(&config)?;
        let output_projection = Self::initialize_output_projection(&config);

        let supported_languages = vec![
            LanguageCode::EnUs,
            LanguageCode::EnGb,
            LanguageCode::DeDe,
            LanguageCode::FrFr,
            LanguageCode::EsEs,
            LanguageCode::JaJp,
            LanguageCode::ZhCn,
            LanguageCode::KoKr,
        ];

        Ok(Self {
            config,
            input_projection,
            positional_encoding,
            blocks,
            output_projection,
            stats: Arc::new(RwLock::new(ConformerStats::default())),
            supported_languages,
            // `with_config` only allocates the architecture; nothing has been trained.
            weight_source: ConformerWeightSource::RandomInit,
        })
    }

    /// Load real trained parameters from a `safetensors` checkpoint.
    ///
    /// The checkpoint must use the VoiRS Conformer tensor layout, which mirrors the
    /// module structure of this implementation:
    ///
    /// ```text
    /// input_projection.weight               [encoder_dim, input_dim]
    /// output_projection.weight              [vocab_size, encoder_dim]
    /// blocks.{i}.attention.query.weight     [heads*head_dim, encoder_dim]
    /// blocks.{i}.attention.key.weight       [heads*head_dim, encoder_dim]
    /// blocks.{i}.attention.value.weight     [heads*head_dim, encoder_dim]
    /// blocks.{i}.attention.output.weight    [encoder_dim, heads*head_dim]
    /// blocks.{i}.conv.pointwise1.weight     [2*encoder_dim, encoder_dim]
    /// blocks.{i}.conv.depthwise.weight      [encoder_dim, conv_kernel_size]
    /// blocks.{i}.conv.pointwise2.weight     [encoder_dim, encoder_dim]
    /// blocks.{i}.conv.norm.weight/.bias     [encoder_dim]
    /// blocks.{i}.ff1.linear1.weight/.bias   [feed_forward_dim, encoder_dim] / [feed_forward_dim]
    /// blocks.{i}.ff1.linear2.weight/.bias   [encoder_dim, feed_forward_dim] / [encoder_dim]
    /// blocks.{i}.ff2.*                      same as ff1, only when macaron_style
    /// blocks.{i}.layer_norm{1,2,3}.weight/.bias  [encoder_dim]
    /// blocks.{i}.layer_norm4.weight/.bias   [encoder_dim], only when macaron_style
    /// ```
    ///
    /// Every tensor is shape-checked against `config` before it is accepted, so a
    /// mismatched checkpoint is rejected instead of being silently truncated.
    ///
    /// # Errors
    /// Returns [`RecognitionError::ModelLoadError`] when the file is missing, is not a
    /// valid `safetensors` container, lacks a required tensor, or declares a shape that
    /// does not match `config`.
    pub async fn from_checkpoint(
        path: impl AsRef<std::path::Path>,
        config: ConformerConfig,
    ) -> Result<Self, RecognitionError> {
        let header = super::weights::SafetensorsHeader::read(path.as_ref())?;
        let mut model = Self::with_config(config).await?;
        model.load_parameters(&header)?;
        model.weight_source = ConformerWeightSource::Checkpoint {
            path: header.path.clone(),
            parameter_count: model.parameter_count(),
        };
        tracing::info!(
            "Loaded Conformer checkpoint {} ({} parameters)",
            header.path.display(),
            model.parameter_count()
        );
        Ok(model)
    }

    /// Overwrite every parameter with the corresponding tensor from `header`.
    fn load_parameters(
        &mut self,
        header: &super::weights::SafetensorsHeader,
    ) -> Result<(), RecognitionError> {
        let dim = self.config.encoder_dim;
        let inner = self.config.attention_heads * (dim / self.config.attention_heads);
        let hidden = self.config.feed_forward_dim;

        self.input_projection =
            header.read_matrix("input_projection.weight", dim, self.config.input_dim)?;
        self.output_projection =
            header.read_matrix("output_projection.weight", self.config.vocab_size, dim)?;

        for (index, block) in self.blocks.iter_mut().enumerate() {
            let prefix = format!("blocks.{index}");

            block.attention.query_weights =
                header.read_matrix(&format!("{prefix}.attention.query.weight"), inner, dim)?;
            block.attention.key_weights =
                header.read_matrix(&format!("{prefix}.attention.key.weight"), inner, dim)?;
            block.attention.value_weights =
                header.read_matrix(&format!("{prefix}.attention.value.weight"), inner, dim)?;
            block.attention.output_weights =
                header.read_matrix(&format!("{prefix}.attention.output.weight"), dim, inner)?;

            block.convolution.pointwise_conv1_weights =
                header.read_matrix(&format!("{prefix}.conv.pointwise1.weight"), dim * 2, dim)?;
            block.convolution.depthwise_conv_weights = header.read_matrix(
                &format!("{prefix}.conv.depthwise.weight"),
                dim,
                self.config.conv_kernel_size,
            )?;
            block.convolution.pointwise_conv2_weights =
                header.read_matrix(&format!("{prefix}.conv.pointwise2.weight"), dim, dim)?;
            block.convolution.batch_norm_gamma =
                header.read_vector(&format!("{prefix}.conv.norm.weight"), dim)?;
            block.convolution.batch_norm_beta =
                header.read_vector(&format!("{prefix}.conv.norm.bias"), dim)?;

            load_feed_forward(&mut block.feed_forward_1, header, &prefix, "ff1", dim, hidden)?;
            if let Some(ff2) = block.feed_forward_2.as_mut() {
                load_feed_forward(ff2, header, &prefix, "ff2", dim, hidden)?;
            }

            for (slot, name) in [
                (&mut block.layer_norm_1, "layer_norm1"),
                (&mut block.layer_norm_2, "layer_norm2"),
                (&mut block.layer_norm_3, "layer_norm3"),
            ] {
                slot.gamma = header.read_vector(&format!("{prefix}.{name}.weight"), dim)?;
                slot.beta = header.read_vector(&format!("{prefix}.{name}.bias"), dim)?;
            }
            if let Some(ln4) = block.layer_norm_4.as_mut() {
                ln4.gamma = header.read_vector(&format!("{prefix}.layer_norm4.weight"), dim)?;
                ln4.beta = header.read_vector(&format!("{prefix}.layer_norm4.bias"), dim)?;
            }
        }

        Ok(())
    }

    /// Total number of scalar parameters really held by this model.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        let matrix = |m: &Vec<Vec<f32>>| m.iter().map(Vec::len).sum::<usize>();
        let mut total = matrix(&self.input_projection) + matrix(&self.output_projection);

        for block in &self.blocks {
            total += matrix(&block.attention.query_weights)
                + matrix(&block.attention.key_weights)
                + matrix(&block.attention.value_weights)
                + matrix(&block.attention.output_weights);
            total += matrix(&block.convolution.pointwise_conv1_weights)
                + matrix(&block.convolution.depthwise_conv_weights)
                + matrix(&block.convolution.pointwise_conv2_weights)
                + block.convolution.batch_norm_gamma.len()
                + block.convolution.batch_norm_beta.len();
            for ff in [Some(&block.feed_forward_1), block.feed_forward_2.as_ref()]
                .into_iter()
                .flatten()
            {
                total += matrix(&ff.linear1_weights)
                    + matrix(&ff.linear2_weights)
                    + ff.linear1_bias.len()
                    + ff.linear2_bias.len();
            }
            for ln in [
                Some(&block.layer_norm_1),
                Some(&block.layer_norm_2),
                Some(&block.layer_norm_3),
                block.layer_norm_4.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                total += ln.gamma.len() + ln.beta.len();
            }
        }

        total
    }

    /// Provenance of this model's parameters.
    #[must_use]
    pub fn weight_source(&self) -> &ConformerWeightSource {
        &self.weight_source
    }

    /// Features this instance can really deliver.
    ///
    /// An untrained model advertises nothing, because every inference entry point
    /// refuses to run. Word timestamps and language detection are not advertised even
    /// when trained: this implementation exposes neither CTC frame alignment nor a
    /// language-identification head.
    fn advertised_features(&self) -> Vec<ASRFeature> {
        if self.weight_source.is_trained() {
            vec![ASRFeature::StreamingInference]
        } else {
            Vec::new()
        }
    }

    /// The typed error returned when transcription is attempted without trained weights.
    fn untrained_error() -> RecognitionError {
        RecognitionError::ModelLoadError {
            message: "Conformer model has randomly initialised parameters and has not been \
                      trained, so it cannot transcribe. Load real parameters with \
                      ConformerModel::from_checkpoint(path, config), or use OnnxConformer \
                      (`onnx` feature) with an exported graph."
                .to_string(),
            source: None,
        }
    }

    /// Initialize input feature projection layer
    fn initialize_input_projection(config: &ConformerConfig) -> Vec<Vec<f32>> {
        let mut weights = Vec::new();
        for _ in 0..config.encoder_dim {
            let mut row = Vec::new();
            for _ in 0..config.input_dim {
                // Xavier/Glorot initialization
                let limit = (6.0 / (config.input_dim + config.encoder_dim) as f32).sqrt();
                row.push(scirs2_core::random::random::<f32>() * 2.0 * limit - limit);
            }
            weights.push(row);
        }
        weights
    }

    /// Initialize Conformer encoder blocks
    fn initialize_conformer_blocks(
        config: &ConformerConfig,
    ) -> Result<Vec<ConformerBlock>, RecognitionError> {
        let mut blocks = Vec::new();

        for i in 0..config.num_blocks {
            tracing::debug!(
                "Initializing Conformer block {}/{}",
                i + 1,
                config.num_blocks
            );

            let attention_config = MultiHeadAttentionConfig {
                num_heads: config.attention_heads,
                head_dim: config.encoder_dim / config.attention_heads,
                dropout_rate: config.dropout_rate,
                use_relative_positional_encoding: config.use_relative_positional_encoding,
            };

            let conv_config = ConvolutionConfig {
                kernel_size: config.conv_kernel_size,
                activation: config.conv_activation.clone(),
                dropout_rate: config.dropout_rate,
            };

            let ff_config = FeedForwardConfig {
                hidden_dim: config.feed_forward_dim,
                dropout_rate: config.dropout_rate,
                activation: ActivationType::Swish,
            };

            let attention = MultiHeadAttention::new(attention_config, config.encoder_dim)?;
            let convolution = ConvolutionModule::new(conv_config, config.encoder_dim)?;
            let feed_forward_1 = FeedForwardNetwork::new(ff_config.clone(), config.encoder_dim)?;
            let feed_forward_2 = if config.macaron_style {
                Some(FeedForwardNetwork::new(ff_config, config.encoder_dim)?)
            } else {
                None
            };

            let layer_norm_1 = LayerNormalization::new(config.encoder_dim);
            let layer_norm_2 = LayerNormalization::new(config.encoder_dim);
            let layer_norm_3 = LayerNormalization::new(config.encoder_dim);
            let layer_norm_4 = if config.macaron_style {
                Some(LayerNormalization::new(config.encoder_dim))
            } else {
                None
            };

            blocks.push(ConformerBlock {
                attention,
                convolution,
                feed_forward_1,
                feed_forward_2,
                layer_norm_1,
                layer_norm_2,
                layer_norm_3,
                layer_norm_4,
                dropout_rate: config.dropout_rate,
            });
        }

        Ok(blocks)
    }

    /// Initialize output projection layer
    fn initialize_output_projection(config: &ConformerConfig) -> Vec<Vec<f32>> {
        let mut weights = Vec::new();
        for _ in 0..config.vocab_size {
            let mut row = Vec::new();
            for _ in 0..config.encoder_dim {
                // Xavier/Glorot initialization
                let limit = (6.0 / (config.encoder_dim + config.vocab_size) as f32).sqrt();
                row.push(scirs2_core::random::random::<f32>() * 2.0 * limit - limit);
            }
            weights.push(row);
        }
        weights
    }

    /// Extract mel-spectrogram features from audio
    async fn extract_features(
        &self,
        audio: &AudioBuffer,
    ) -> Result<Vec<Vec<f32>>, RecognitionError> {
        let sample_rate = audio.sample_rate();
        let samples = audio.samples();

        // Ensure audio is sampled at 16kHz
        if sample_rate != 16000 {
            return Err(VoirsError::AudioError {
                message: format!("Expected 16kHz sample rate, got {}Hz", sample_rate),
                buffer_info: None,
            }
            .into());
        }

        // Extract mel-spectrogram features
        let features = self.compute_mel_spectrogram(samples).await?;

        tracing::debug!(
            "Extracted features with shape: {}x{}",
            features.len(),
            features.first().map_or(0, |f| f.len())
        );

        Ok(features)
    }

    /// Compute a real log-mel spectrogram from audio samples.
    ///
    /// 25 ms Hann-windowed frames with a 10 ms hop are transformed with
    /// [`scirs2_fft::rfft`], the power spectrum is passed through a triangular mel
    /// filterbank spanning 0 Hz to Nyquist, and the filter energies are returned as
    /// natural logs with a floor.
    ///
    /// # Errors
    /// Returns [`RecognitionError::AudioProcessingError`] if the FFT fails.
    async fn compute_mel_spectrogram(
        &self,
        samples: &[f32],
    ) -> Result<Vec<Vec<f32>>, RecognitionError> {
        const SAMPLE_RATE: f32 = 16_000.0;
        let window_size = 400; // 25 ms at 16 kHz
        let hop_size = 160; // 10 ms at 16 kHz
        let n_mels = self.config.input_dim;

        if samples.len() < window_size {
            return Ok(Vec::new());
        }

        // Periodic Hann window.
        #[allow(clippy::cast_precision_loss)]
        let window: Vec<f64> = (0..window_size)
            .map(|i| {
                let phase =
                    2.0 * std::f64::consts::PI * i as f64 / window_size as f64;
                0.5 * (1.0 - phase.cos())
            })
            .collect();

        let n_bins = window_size / 2 + 1;
        let filterbank = mel_filterbank(n_mels, n_bins, SAMPLE_RATE);

        let num_frames = (samples.len() - window_size) / hop_size + 1;
        let mut features = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_size;
            let frame = &samples[start..start + window_size];

            let windowed: Vec<f64> = frame
                .iter()
                .zip(window.iter())
                .map(|(&sample, &w)| f64::from(sample) * w)
                .collect();

            let spectrum = scirs2_fft::rfft(&windowed, None).map_err(|e| {
                RecognitionError::AudioProcessingError {
                    message: format!("FFT failed for frame {frame_idx}: {e}"),
                    source: None,
                }
            })?;

            let power: Vec<f32> = spectrum
                .iter()
                .take(n_bins)
                .map(|c| {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        (c.re * c.re + c.im * c.im) as f32
                    }
                })
                .collect();

            let mut frame_features = Vec::with_capacity(n_mels);
            for filter in &filterbank {
                let energy: f32 = filter
                    .iter()
                    .zip(power.iter())
                    .map(|(weight, value)| weight * value)
                    .sum();
                frame_features.push(energy.max(1e-10).ln().max(-80.0));
            }

            features.push(frame_features);
        }

        Ok(features)
    }

    /// Forward pass through the Conformer model
    async fn forward(&self, features: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>, RecognitionError> {
        if features.is_empty() {
            return Err(VoirsError::AudioError {
                message: "Empty feature sequence".to_string(),
                buffer_info: None,
            }
            .into());
        }

        tracing::debug!(
            "Processing sequence of length {} through Conformer",
            features.len()
        );

        // Input projection
        let mut hidden_states = self.apply_input_projection(&features)?;

        // Add positional encoding
        hidden_states = self.add_positional_encoding(hidden_states)?;

        // Process through Conformer blocks
        for (block_idx, block) in self.blocks.iter().enumerate() {
            tracing::debug!("Processing block {}/{}", block_idx + 1, self.blocks.len());
            hidden_states = self.process_conformer_block(hidden_states, block).await?;
        }

        // Output projection
        let logits = self.apply_output_projection(&hidden_states)?;

        Ok(logits)
    }

    /// Apply input projection to features
    fn apply_input_projection(
        &self,
        features: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, RecognitionError> {
        let mut projected = Vec::new();

        for frame in features {
            if frame.len() != self.config.input_dim {
                return Err(VoirsError::AudioError {
                    message: format!(
                        "Expected input dimension {}, got {}",
                        self.config.input_dim,
                        frame.len()
                    ),
                    buffer_info: None,
                }
                .into());
            }

            let mut output = vec![0.0; self.config.encoder_dim];
            for (i, row) in self.input_projection.iter().enumerate() {
                for (j, &weight) in row.iter().enumerate() {
                    output[i] += weight * frame[j];
                }
            }
            projected.push(output);
        }

        Ok(projected)
    }

    /// Add positional encoding to hidden states
    fn add_positional_encoding(
        &self,
        mut hidden_states: Vec<Vec<f32>>,
    ) -> Result<Vec<Vec<f32>>, RecognitionError> {
        for (i, state) in hidden_states.iter_mut().enumerate() {
            if i >= self.positional_encoding.encodings.len() {
                break;
            }

            for (j, pos_enc) in self.positional_encoding.encodings[i].iter().enumerate() {
                if j < state.len() {
                    state[j] += pos_enc;
                }
            }
        }

        Ok(hidden_states)
    }

    /// Process through a single Conformer block
    async fn process_conformer_block(
        &self,
        mut input: Vec<Vec<f32>>,
        block: &ConformerBlock,
    ) -> Result<Vec<Vec<f32>>, RecognitionError> {
        // Macaron-style half-step feed-forward: x = x + 1/2 * FFN(LN(x)).
        // The residual connection was previously missing, so the half-step replaced the
        // input instead of being added to it.
        if let Some(ff2) = &block.feed_forward_2 {
            let macaron_residual = input.clone();
            if let Some(ln4) = &block.layer_norm_4 {
                input = self.apply_layer_norm(&input, ln4)?;
            }
            input = self.apply_feed_forward(&input, ff2, 0.5).await?;
            input = self.add_residual_connection(input, macaron_residual)?;
        }

        // Multi-head self-attention
        let attention_residual = input.clone();
        input = self.apply_layer_norm(&input, &block.layer_norm_1)?;
        input = self
            .apply_multi_head_attention(&input, &block.attention)
            .await?;
        input = self.add_residual_connection(input, attention_residual)?;

        // Convolution module
        let conv_residual = input.clone();
        input = self.apply_layer_norm(&input, &block.layer_norm_2)?;
        input = self
            .apply_convolution_module(&input, &block.convolution)
            .await?;
        input = self.add_residual_connection(input, conv_residual)?;

        // Feed-forward network
        let ff_residual = input.clone();
        input = self.apply_layer_norm(&input, &block.layer_norm_3)?;
        input = self
            .apply_feed_forward(&input, &block.feed_forward_1, 1.0)
            .await?;
        input = self.add_residual_connection(input, ff_residual)?;

        Ok(input)
    }

    /// Apply scaled dot-product multi-head self-attention.
    ///
    /// This is the real computation: each frame is projected through the module's own
    /// learned `query`/`key`/`value` matrices, attention scores are formed per head as
    /// `Q Kᵀ / sqrt(head_dim)`, normalised with a numerically stable softmax, applied to
    /// `V`, and the concatenated heads are projected back through `output_weights`.
    ///
    /// # Errors
    /// Returns [`RecognitionError::ModelError`] if the stored weight matrices do not
    /// match the sequence's model dimension.
    async fn apply_multi_head_attention(
        &self,
        input: &[Vec<f32>],
        attention: &MultiHeadAttention,
    ) -> Result<Vec<Vec<f32>>, RecognitionError> {
        let seq_len = input.len();
        let model_dim = input[0].len();
        let head_dim = attention.config.head_dim;
        let num_heads = attention.config.num_heads;
        let inner_dim = num_heads * head_dim;

        tracing::debug!("Applying multi-head attention with {num_heads} heads");

        if attention.query_weights.len() != inner_dim
            || attention.output_weights.len() != model_dim
        {
            return Err(RecognitionError::ModelError {
                message: format!(
                    "Attention weight shapes do not match the model: expected {inner_dim} \
                     projection rows and {model_dim} output rows, found {} and {}",
                    attention.query_weights.len(),
                    attention.output_weights.len()
                ),
                source: None,
            });
        }

        // Project the whole sequence into Q, K and V.
        let queries = matmul_rows(input, &attention.query_weights)?;
        let keys = matmul_rows(input, &attention.key_weights)?;
        let values = matmul_rows(input, &attention.value_weights)?;

        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut context = vec![vec![0.0_f32; inner_dim]; seq_len];

        for head in 0..num_heads {
            let offset = head * head_dim;
            for (i, context_row) in context.iter_mut().enumerate() {
                // Scores of query i against every key.
                let mut scores = Vec::with_capacity(seq_len);
                for key in keys.iter() {
                    let mut dot = 0.0_f32;
                    for d in 0..head_dim {
                        dot += queries[i][offset + d] * key[offset + d];
                    }
                    scores.push(dot * scale);
                }

                // Learned relative-position bias, when the checkpoint provides one.
                if let Some(bias) = &attention.relative_position_bias {
                    for (j, score) in scores.iter_mut().enumerate() {
                        let distance = j as isize - i as isize;
                        let index = (distance + bias[head].len() as isize / 2)
                            .clamp(0, bias[head].len() as isize - 1)
                            as usize;
                        *score += bias[head][index];
                    }
                }

                softmax_in_place(&mut scores);

                for (j, weight) in scores.iter().enumerate() {
                    for d in 0..head_dim {
                        context_row[offset + d] += weight * values[j][offset + d];
                    }
                }
            }
        }

        matmul_rows(&context, &attention.output_weights)
    }

    /// Apply the Conformer convolution module.
    ///
    /// Real computation, in the order given by the Conformer paper (section 2.2):
    /// pointwise convolution to `2 * d` channels, gated linear unit, depthwise
    /// convolution over time using the module's own per-channel kernel, normalisation
    /// with the learned `gamma`/`beta`, the configured activation, and a second
    /// pointwise convolution back to `d` channels.
    ///
    /// Because inference-time batch-norm running statistics are not part of the stored
    /// parameters, normalisation uses the per-channel statistics of the current
    /// utterance before applying the learned affine transform.
    ///
    /// # Errors
    /// Returns [`RecognitionError::ModelError`] when the stored kernels do not match the
    /// sequence's channel count.
    async fn apply_convolution_module(
        &self,
        input: &[Vec<f32>],
        conv_module: &ConvolutionModule,
    ) -> Result<Vec<Vec<f32>>, RecognitionError> {
        let seq_len = input.len();
        let model_dim = input[0].len();
        let kernel_size = conv_module.config.kernel_size;

        tracing::debug!("Applying convolution module with kernel size {kernel_size}");

        if conv_module.pointwise_conv1_weights.len() != model_dim * 2
            || conv_module.depthwise_conv_weights.len() != model_dim
            || conv_module.pointwise_conv2_weights.len() != model_dim
        {
            return Err(RecognitionError::ModelError {
                message: format!(
                    "Convolution module shapes do not match the model dimension {model_dim}: \
                     pointwise1 rows {}, depthwise rows {}, pointwise2 rows {}",
                    conv_module.pointwise_conv1_weights.len(),
                    conv_module.depthwise_conv_weights.len(),
                    conv_module.pointwise_conv2_weights.len()
                ),
                source: None,
            });
        }

        // Pointwise convolution 1: d -> 2d, then GLU gating back down to d.
        let expanded = matmul_rows(input, &conv_module.pointwise_conv1_weights)?;
        let mut gated = vec![vec![0.0_f32; model_dim]; seq_len];
        for (t, frame) in expanded.iter().enumerate() {
            for c in 0..model_dim {
                let value = frame[c];
                let gate = frame[model_dim + c];
                gated[t][c] = value * sigmoid(gate);
            }
        }

        // Depthwise convolution over time with the module's own learned kernel.
        let padding = kernel_size / 2;
        let mut convolved = vec![vec![0.0_f32; model_dim]; seq_len];
        for c in 0..model_dim {
            let kernel = &conv_module.depthwise_conv_weights[c];
            let taps = kernel.len().min(kernel_size);
            for t in 0..seq_len {
                let mut sum = 0.0_f32;
                for (k, &tap) in kernel.iter().take(taps).enumerate() {
                    let index = t as isize + k as isize - padding as isize;
                    if index >= 0 && (index as usize) < seq_len {
                        sum += tap * gated[index as usize][c];
                    }
                }
                convolved[t][c] = sum;
            }
        }

        // Normalisation with the learned affine parameters, then the activation.
        #[allow(clippy::cast_precision_loss)]
        let frames = seq_len as f32;
        for c in 0..model_dim {
            let mean = convolved.iter().map(|frame| frame[c]).sum::<f32>() / frames;
            let variance = convolved
                .iter()
                .map(|frame| (frame[c] - mean).powi(2))
                .sum::<f32>()
                / frames;
            let inv_std = 1.0 / (variance + 1e-5).sqrt();
            let gamma = conv_module.batch_norm_gamma.get(c).copied().unwrap_or(1.0);
            let beta = conv_module.batch_norm_beta.get(c).copied().unwrap_or(0.0);
            for frame in &mut convolved {
                let normalized = (frame[c] - mean) * inv_std * gamma + beta;
                frame[c] = self.apply_activation(normalized, &conv_module.config.activation);
            }
        }

        // Pointwise convolution 2: d -> d.
        matmul_rows(&convolved, &conv_module.pointwise_conv2_weights)
    }

    /// Apply the position-wise feed-forward network.
    ///
    /// Real computation: `scale * W2 · activation(W1 · x + b1) + b2`, using the
    /// network's own stored matrices and biases. `scale` is `0.5` for the macaron-style
    /// half-step feed-forward and `1.0` for the full-step one.
    ///
    /// # Errors
    /// Returns [`RecognitionError::ModelError`] when the stored matrices do not match
    /// the sequence's model dimension.
    async fn apply_feed_forward(
        &self,
        input: &[Vec<f32>],
        ff_network: &FeedForwardNetwork,
        scale: f32,
    ) -> Result<Vec<Vec<f32>>, RecognitionError> {
        let model_dim = input[0].len();
        let hidden_dim = ff_network.config.hidden_dim;

        tracing::debug!("Applying feed-forward network with scale {scale}");

        if ff_network.linear1_weights.len() != hidden_dim
            || ff_network.linear2_weights.len() != model_dim
        {
            return Err(RecognitionError::ModelError {
                message: format!(
                    "Feed-forward shapes do not match: expected {hidden_dim} hidden rows and \
                     {model_dim} output rows, found {} and {}",
                    ff_network.linear1_weights.len(),
                    ff_network.linear2_weights.len()
                ),
                source: None,
            });
        }

        let mut hidden = matmul_rows(input, &ff_network.linear1_weights)?;
        for frame in &mut hidden {
            for (unit, value) in frame.iter_mut().enumerate() {
                *value += ff_network.linear1_bias.get(unit).copied().unwrap_or(0.0);
                *value = self.apply_activation(*value, &ff_network.config.activation);
            }
        }

        let mut output = matmul_rows(&hidden, &ff_network.linear2_weights)?;
        for frame in &mut output {
            for (unit, value) in frame.iter_mut().enumerate() {
                *value += ff_network.linear2_bias.get(unit).copied().unwrap_or(0.0);
                *value *= scale;
            }
        }

        Ok(output)
    }

    /// Apply layer normalization
    fn apply_layer_norm(
        &self,
        input: &[Vec<f32>],
        layer_norm: &LayerNormalization,
    ) -> Result<Vec<Vec<f32>>, RecognitionError> {
        let mut output = Vec::new();

        for sequence in input {
            let mut normalized = Vec::new();

            // Calculate mean and variance
            let mean = sequence.iter().sum::<f32>() / sequence.len() as f32;
            let variance =
                sequence.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / sequence.len() as f32;

            let std_dev = (variance + layer_norm.eps).sqrt();

            // Normalize and apply learnable parameters
            for (i, &value) in sequence.iter().enumerate() {
                let normalized_value = (value - mean) / std_dev;
                let scaled_value = normalized_value * layer_norm.gamma[i] + layer_norm.beta[i];
                normalized.push(scaled_value);
            }

            output.push(normalized);
        }

        Ok(output)
    }

    /// Add residual connection
    fn add_residual_connection(
        &self,
        mut input: Vec<Vec<f32>>,
        residual: Vec<Vec<f32>>,
    ) -> Result<Vec<Vec<f32>>, RecognitionError> {
        for (i, sequence) in input.iter_mut().enumerate() {
            for (j, value) in sequence.iter_mut().enumerate() {
                *value += residual[i][j];
            }
        }
        Ok(input)
    }

    /// Apply activation function
    fn apply_activation(&self, x: f32, activation: &ActivationType) -> f32 {
        match activation {
            ActivationType::ReLU => x.max(0.0),
            ActivationType::GELU => {
                // Approximation of GELU
                0.5 * x * (1.0 + (0.797_884_560_8 * (x + 0.044_715 * x.powi(3))).tanh())
            }
            ActivationType::Swish => x / (1.0 + (-x).exp()),
            ActivationType::GLU => {
                // Simplified GLU (needs proper gating)
                x * (x / (1.0 + (-x).exp()))
            }
        }
    }

    /// Apply output projection
    fn apply_output_projection(
        &self,
        hidden_states: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, RecognitionError> {
        let mut logits = Vec::new();

        for state in hidden_states {
            let mut output = vec![0.0; self.config.vocab_size];
            for (i, row) in self.output_projection.iter().enumerate() {
                for (j, &weight) in row.iter().enumerate() {
                    if j < state.len() {
                        output[i] += weight * state[j];
                    }
                }
            }
            logits.push(output);
        }

        Ok(logits)
    }

    /// Greedy CTC decoding.
    ///
    /// Returns the decoded text together with a real confidence: the mean softmax
    /// posterior of the arg-max token across frames, which varies with the actual
    /// logits rather than being a fixed constant.
    ///
    /// # Errors
    /// Returns [`RecognitionError::ModelError`] if a frame has no logits.
    async fn decode_logits(
        &self,
        logits: Vec<Vec<f32>>,
    ) -> Result<(String, f32), RecognitionError> {
        let mut tokens = Vec::with_capacity(logits.len());
        let mut posterior_sum = 0.0_f32;

        for frame_logits in &logits {
            if frame_logits.is_empty() {
                return Err(RecognitionError::ModelError {
                    message: "Decoder received a frame with no logits".to_string(),
                    source: None,
                });
            }

            let mut probabilities = frame_logits.clone();
            softmax_in_place(&mut probabilities);

            let mut max_idx = 0;
            let mut max_val = probabilities[0];
            for (i, &probability) in probabilities.iter().enumerate().skip(1) {
                if probability > max_val {
                    max_val = probability;
                    max_idx = i;
                }
            }

            posterior_sum += max_val;
            tokens.push(max_idx);
        }

        let confidence = if logits.is_empty() {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            {
                posterior_sum / logits.len() as f32
            }
        };

        let text = self.tokens_to_text(&tokens).await?;

        Ok((text, confidence))
    }

    /// Convert token IDs to text
    async fn tokens_to_text(&self, tokens: &[usize]) -> Result<String, RecognitionError> {
        // Simplified token-to-text conversion
        // In practice, this would use a proper tokenizer/vocabulary

        let mut words = Vec::new();
        let mut current_word = String::new();

        for &token_id in tokens {
            if token_id == 0 {
                // Blank token (CTC)
                continue;
            }
            if token_id == 1 {
                // Space token
                if !current_word.is_empty() {
                    words.push(current_word.clone());
                    current_word.clear();
                }
            } else {
                // Character token (simplified mapping)
                // token_id >= 2 maps to alphabet characters a-z (indices 0-25).
                // Use wrapping arithmetic and modulo to stay within valid range,
                // guarding against any token_id value that may appear with random
                // model weights (vocab_size can be 5000+).
                let char_index = (token_id - 2) % 26;
                let ch = (b'a' + char_index as u8) as char;
                current_word.push(ch);
            }
        }

        if !current_word.is_empty() {
            words.push(current_word);
        }

        Ok(words.join(" "))
    }

    /// Update model statistics
    async fn update_stats(&self, processing_time_ms: u64, success: bool) {
        let mut stats = self.stats.write().await;
        stats.inference_count += 1;
        stats.total_processing_time_ms += processing_time_ms;
        stats.avg_processing_time_ms =
            stats.total_processing_time_ms as f64 / stats.inference_count as f64;

        if success {
            stats.successful_inferences += 1;
        } else {
            stats.failed_inferences += 1;
        }
    }

    /// Get model statistics
    pub async fn get_stats(&self) -> ConformerStats {
        (*self.stats.read().await).clone()
    }
}

#[async_trait::async_trait]
impl ASRModel for ConformerModel {
    async fn transcribe(
        &self,
        audio: &AudioBuffer,
        config: Option<&ASRConfig>,
    ) -> crate::traits::RecognitionResult<Transcript> {
        let start_time = std::time::Instant::now();

        tracing::info!(
            "Starting Conformer transcription for {:.2}s audio",
            audio.duration()
        );

        let result = async {
            // Refuse to decode from untrained parameters: the architecture is real but
            // random weights carry no learned representation, so any text produced would
            // be noise presented as a transcript.
            if !self.weight_source.is_trained() {
                return Err(Self::untrained_error().into());
            }

            // Extract features
            let features = self.extract_features(audio).await?;

            // Forward pass through the model
            let logits = self.forward(features).await?;

            // Decode to text, keeping the decoder's own mean per-frame posterior as the
            // confidence rather than a fixed constant.
            let (text, confidence) = self.decode_logits(logits).await?;

            let language = config
                .and_then(|c| c.language)
                .unwrap_or(LanguageCode::EnUs);

            let result = Transcript {
                text: text.clone(),
                language,
                confidence,
                word_timestamps: vec![], // CTC frame alignment is not exposed yet
                sentence_boundaries: vec![],
                processing_duration: Some(start_time.elapsed()),
            };

            Ok(result)
        }
        .await;

        let processing_time = start_time.elapsed().as_millis() as u64;
        self.update_stats(processing_time, result.is_ok()).await;

        match &result {
            Ok(r) => {
                tracing::info!(
                    "Conformer transcription completed: \"{}\" (confidence: {:.2})",
                    r.text,
                    r.confidence
                );
            }
            Err(e) => {
                tracing::error!("Conformer transcription failed: {}", e);
            }
        }

        result
    }

    /// Model metadata.
    ///
    /// Every number here is derived from this instance rather than asserted:
    /// `model_size_mb` is the real parameter count times four bytes, `wer_benchmarks` is
    /// empty because VoiRS has measured no WER for this implementation, and
    /// `inference_speed` is `0.0`, meaning "not measured".
    fn metadata(&self) -> ASRMetadata {
        #[allow(clippy::cast_precision_loss)]
        let model_size_mb =
            (self.parameter_count() * std::mem::size_of::<f32>()) as f32 / (1024.0 * 1024.0);

        let description = match &self.weight_source {
            ConformerWeightSource::RandomInit => "Convolution-augmented Transformer for Speech \
                 Recognition. Parameters are randomly initialised and untrained: \
                 transcription is refused until a checkpoint is loaded."
                .to_string(),
            ConformerWeightSource::Checkpoint { path, .. } => format!(
                "Convolution-augmented Transformer for Speech Recognition, parameters loaded \
                 from {}",
                path.display()
            ),
        };

        ASRMetadata {
            name: "Conformer".to_string(),
            version: "1.0.0".to_string(),
            description,
            supported_languages: self.supported_languages(),
            architecture: "Conformer".to_string(),
            model_size_mb,
            // 0.0 == not measured. Call the benchmarking suite on real hardware to get a
            // real figure instead.
            inference_speed: 0.0,
            // Empty: no WER has been measured for this implementation.
            wer_benchmarks: HashMap::new(),
            supported_features: self.advertised_features(),
        }
    }

    fn supports_feature(&self, feature: ASRFeature) -> bool {
        self.advertised_features().contains(&feature)
    }

    fn supported_languages(&self) -> Vec<LanguageCode> {
        self.supported_languages.clone()
    }

    async fn transcribe_streaming(
        &self,
        audio_stream: AudioStream,
        config: Option<&ASRConfig>,
    ) -> crate::traits::RecognitionResult<TranscriptStream> {
        use futures::StreamExt;

        // Same invariant as `transcribe`: no decoding from untrained parameters.
        if !self.weight_source.is_trained() {
            return Err(Self::untrained_error().into());
        }

        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        let model = self.clone();
        let config_owned = config.cloned();

        tokio::spawn(async move {
            // `audio_stream` is a Pin<Box<dyn Stream<Item = AudioBuffer> + Send>>;
            // we need it to be Unpin so that `.next()` works in async move.
            let mut audio_stream = audio_stream;
            let mut chunk_index: usize = 0;
            let mut accumulated_text = String::new();
            // Lookahead buffer: accumulate raw samples across chunks so the
            // Conformer encoder sees sufficient context (≥ one mel-spectrogram
            // window) before producing a partial transcript.
            let mut sample_buffer: Vec<f32> = Vec::new();
            let mut confidence_sum = 0.0_f32;
            let mut confidence_count = 0_usize;
            // Number of raw samples per chunk emit — 1 second at 16 kHz.
            const CHUNK_SAMPLES: usize = 16_000;

            while let Some(audio_chunk) = audio_stream.next().await {
                // Buffer incoming samples.
                let new_samples = audio_chunk.samples();
                sample_buffer.extend_from_slice(new_samples);

                // Only run inference once the buffer holds enough context.
                while sample_buffer.len() >= CHUNK_SAMPLES {
                    let window: Vec<f32> = sample_buffer[..CHUNK_SAMPLES].to_vec();
                    sample_buffer.drain(..CHUNK_SAMPLES);

                    // Build a temporary AudioBuffer for the window.
                    let window_buf = AudioBuffer::new(window, 16_000, 1);

                    // Extract features and run a forward pass.
                    let partial_result = async {
                        let features = model.extract_features(&window_buf).await?;
                        let logits = model.forward(features).await?;
                        model.decode_logits(logits).await
                    }
                    .await;

                    let start_time = chunk_index as f32;
                    let end_time = start_time + 1.0;

                    match partial_result {
                        Ok((text, confidence)) => {
                            accumulated_text.push_str(&text);
                            accumulated_text.push(' ');
                            confidence_sum += confidence;
                            confidence_count += 1;

                            let chunk = TranscriptChunk {
                                text,
                                is_final: false,
                                start_time,
                                end_time,
                                confidence,
                            };
                            if sender.send(Ok(chunk)).is_err() {
                                return;
                            }
                        }
                        Err(e) => {
                            if sender.send(Err(e.into())).is_err() {
                                return;
                            }
                        }
                    }
                    chunk_index += 1;
                }
            }

            // Flush any remaining samples in the lookahead buffer.
            if !sample_buffer.is_empty() {
                let remainder_buf = AudioBuffer::new(sample_buffer.clone(), 16_000, 1);
                let flush_result = async {
                    let features = model.extract_features(&remainder_buf).await?;
                    let logits = model.forward(features).await?;
                    model.decode_logits(logits).await
                }
                .await;

                if let Ok((text, confidence)) = flush_result {
                    accumulated_text.push_str(&text);
                    accumulated_text.push(' ');
                    confidence_sum += confidence;
                    confidence_count += 1;
                }
            }

            // Emit the final consolidated transcript chunk.
            let final_text = accumulated_text.trim().to_string();
            let total_duration = chunk_index as f32;
            // Real aggregate confidence: the mean of the per-chunk posteriors that were
            // actually produced, not a constant.
            #[allow(clippy::cast_precision_loss)]
            let final_confidence = if confidence_count == 0 {
                0.0
            } else {
                confidence_sum / confidence_count as f32
            };
            let final_chunk = TranscriptChunk {
                text: final_text,
                is_final: true,
                start_time: 0.0,
                end_time: total_duration,
                confidence: final_confidence,
            };
            let _ = sender.send(Ok(final_chunk));
        });

        Ok(Box::pin(
            tokio_stream::wrappers::UnboundedReceiverStream::new(receiver),
        ))
    }
}

// Implementation for helper components

impl MultiHeadAttention {
    fn new(config: MultiHeadAttentionConfig, model_dim: usize) -> Result<Self, RecognitionError> {
        let head_dim = config.head_dim;
        let num_heads = config.num_heads;

        Ok(Self {
            config,
            query_weights: Self::initialize_attention_weights(model_dim, num_heads * head_dim),
            key_weights: Self::initialize_attention_weights(model_dim, num_heads * head_dim),
            value_weights: Self::initialize_attention_weights(model_dim, num_heads * head_dim),
            output_weights: Self::initialize_attention_weights(num_heads * head_dim, model_dim),
            relative_position_bias: None, // Simplified
        })
    }

    fn initialize_attention_weights(input_dim: usize, output_dim: usize) -> Vec<Vec<f32>> {
        let mut weights = Vec::new();
        for _ in 0..output_dim {
            let mut row = Vec::new();
            for _ in 0..input_dim {
                let limit = (6.0 / (input_dim + output_dim) as f32).sqrt();
                row.push(scirs2_core::random::random::<f32>() * 2.0 * limit - limit);
            }
            weights.push(row);
        }
        weights
    }
}

impl ConvolutionModule {
    fn new(config: ConvolutionConfig, model_dim: usize) -> Result<Self, RecognitionError> {
        Ok(Self {
            pointwise_conv1_weights: Self::initialize_conv_weights(model_dim, model_dim * 2),
            depthwise_conv_weights: Self::initialize_depthwise_weights(
                model_dim,
                config.kernel_size,
            ),
            pointwise_conv2_weights: Self::initialize_conv_weights(model_dim, model_dim),
            batch_norm_gamma: vec![1.0; model_dim],
            batch_norm_beta: vec![0.0; model_dim],
            glu_weights: None,
            config,
        })
    }

    fn initialize_conv_weights(input_channels: usize, output_channels: usize) -> Vec<Vec<f32>> {
        let mut weights = Vec::new();
        for _ in 0..output_channels {
            let mut row = Vec::new();
            for _ in 0..input_channels {
                let limit = (6.0 / (input_channels + output_channels) as f32).sqrt();
                row.push(scirs2_core::random::random::<f32>() * 2.0 * limit - limit);
            }
            weights.push(row);
        }
        weights
    }

    /// One kernel per channel, sized from the module's configured kernel size.
    ///
    /// Previously this always allocated 31 taps regardless of `kernel_size`, so a
    /// configured kernel other than 31 silently used the wrong number of taps.
    fn initialize_depthwise_weights(channels: usize, kernel_size: usize) -> Vec<Vec<f32>> {
        let mut weights = Vec::new();
        for _ in 0..channels {
            let mut row = Vec::new();
            for _ in 0..kernel_size {
                row.push(scirs2_core::random::random::<f32>() * 0.1 - 0.05);
            }
            weights.push(row);
        }
        weights
    }
}

impl FeedForwardNetwork {
    fn new(config: FeedForwardConfig, model_dim: usize) -> Result<Self, RecognitionError> {
        let hidden_dim = config.hidden_dim;
        Ok(Self {
            linear1_weights: Self::initialize_linear_weights(model_dim, hidden_dim),
            linear2_weights: Self::initialize_linear_weights(hidden_dim, model_dim),
            linear1_bias: vec![0.0; hidden_dim],
            linear2_bias: vec![0.0; model_dim],
            config,
        })
    }

    fn initialize_linear_weights(input_dim: usize, output_dim: usize) -> Vec<Vec<f32>> {
        let mut weights = Vec::new();
        for _ in 0..output_dim {
            let mut row = Vec::new();
            for _ in 0..input_dim {
                let limit = (6.0 / (input_dim + output_dim) as f32).sqrt();
                row.push(scirs2_core::random::random::<f32>() * 2.0 * limit - limit);
            }
            weights.push(row);
        }
        weights
    }
}

impl LayerNormalization {
    fn new(model_dim: usize) -> Self {
        Self {
            gamma: vec![1.0; model_dim],
            beta: vec![0.0; model_dim],
            eps: 1e-6,
        }
    }
}

impl PositionalEncoding {
    fn new(max_length: usize, d_model: usize) -> Self {
        let mut encodings = Vec::new();

        for pos in 0..max_length {
            let mut encoding = Vec::new();
            for i in 0..d_model {
                let angle = pos as f32 / 10000_f32.powf((2 * (i / 2)) as f32 / d_model as f32);
                let value = if i % 2 == 0 { angle.sin() } else { angle.cos() };
                encoding.push(value);
            }
            encodings.push(encoding);
        }

        Self {
            max_length,
            d_model,
            encodings,
        }
    }
}

/// Factory function to create a Conformer ASR model
pub async fn create_conformer_asr() -> Result<Arc<dyn ASRModel>, RecognitionError> {
    let model = ConformerModel::new().await?;
    Ok(Arc::new(model))
}

/// Factory function to create a Conformer ASR model with custom configuration
pub async fn create_conformer_asr_with_config(
    config: ConformerConfig,
) -> Result<Arc<dyn ASRModel>, RecognitionError> {
    let model = ConformerModel::with_config(config).await?;
    Ok(Arc::new(model))
}

#[cfg(test)]
#[path = "conformer_tests.rs"]
mod tests;
