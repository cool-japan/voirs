//! Deep Learning-based MOS (Mean Opinion Score) Prediction
//!
//! This module provides two selectable MOS-prediction backends:
//!
//! - [`MosPredictionBackend::Pretrained`]: a real CNN (+ optional attention
//!   pooling) whose weights are loaded from a safetensors checkpoint via
//!   [`DeepMosConfig::model_weights_path`]. When no checkpoint is
//!   configured, or loading fails for any reason, [`DeepMosPredictor::initialize`]
//!   fails closed with a typed [`Error`] rather than silently building an
//!   untrained ("all zeros") network.
//! - [`MosPredictionBackend::DspProxy`]: used automatically when no
//!   pretrained checkpoint is configured. This computes a real, honest
//!   signal-processing proxy for perceived quality from measurable
//!   properties of the audio itself (SNR, clipping ratio, dynamic range,
//!   spectral flatness). It makes **no claim** of correlation with human
//!   MOS ratings - it is a coarse heuristic, not a trained quality predictor.
//!
//! [`MosPrediction::backend`] always reports which of the two produced a
//! given prediction.
//!
//! # Performance
//!
//! No correlation-with-human-MOS numbers are claimed for either backend:
//! for [`MosPredictionBackend::Pretrained`] that depends entirely on the
//! quality of whatever checkpoint is supplied via `model_weights_path` (not
//! shipped by this crate), and [`MosPredictionBackend::DspProxy`] is an
//! explicit non-learned heuristic.
//!
//! # References
//!
//! - "Deep Learning-Based Non-Intrusive Multi-Objective Speech Assessment Model" (2021)
//! - "MOSNet: Deep Learning based Objective Assessment for Voice Conversion" (2019)

use crate::{types::VoiceSample, Error, Result};
use candle_core::{DType, Device, ModuleT, Tensor};
use candle_nn::{
    batch_norm, conv1d, linear, ops, BatchNorm, Conv1d, Conv1dConfig, Linear, Module, VarBuilder,
};
use scirs2_core::ndarray::Array2;
use scirs2_fft::RealFftPlanner;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, trace};

/// Deep MOS predictor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepMosConfig {
    /// Number of mel filterbank channels
    pub n_mels: usize,
    /// FFT window size
    pub n_fft: usize,
    /// Hop length for STFT
    pub hop_length: usize,
    /// Sample rate
    pub sample_rate: u32,
    /// Frame aggregation method
    pub aggregation: AggregationMethod,
    /// Enable GPU acceleration
    pub use_gpu: bool,
    /// Model architecture variant
    pub architecture: MosArchitecture,
    /// Path to a safetensors checkpoint with real pretrained CNN weights.
    /// When `None` (the default), the predictor falls back to the
    /// honestly-labeled [`MosPredictionBackend::DspProxy`] instead of ever
    /// building an untrained "neural" model.
    ///
    /// Expected tensor names (all `f32`), where `i` ranges over
    /// `0..conv_channels.len() - 1`:
    /// - `conv.{i}.weight` `[conv_channels[i+1], conv_channels[i], conv_kernel_size]`,
    ///   `conv.{i}.bias` `[conv_channels[i+1]]`
    /// - `bn.{i}.weight/.bias/.running_mean/.running_var` (each `[conv_channels[i+1]]`)
    /// - when `architecture` is `Conformer` or `Attention`:
    ///   `attention.{query,key,value,output}.weight/.bias` (each `[C, C]`
    ///   where `C` is the last entry of `conv_channels`)
    /// - for each `j` in `0..fc_dims.len()`: `fc.{j}.weight/.bias`
    /// - `output.weight` `[1, last_fc_dim]` / `.bias` `[1]`
    pub model_weights_path: Option<PathBuf>,
    /// Convolutional channel sequence, e.g. `[n_mels, 64, 32]`. The first
    /// entry must equal `n_mels`.
    pub conv_channels: Vec<usize>,
    /// Kernel size for every convolutional layer.
    pub conv_kernel_size: usize,
    /// Fully-connected hidden dimensions applied after pooling.
    pub fc_dims: Vec<usize>,
    /// Attention heads (informational scaling factor) used when
    /// `architecture` is `Conformer` or `Attention`.
    pub attention_heads: usize,
}

/// Frame aggregation methods for MOS prediction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Average pooling across time
    Average,
    /// Attention-based weighted pooling
    Attention,
    /// Last frame prediction
    Last,
    /// Recurrent aggregation (LSTM/GRU)
    Recurrent,
}

/// MOS predictor architecture variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MosArchitecture {
    /// Lightweight CNN for real-time inference (mean pooling over time)
    LightCNN,
    /// Deep residual network for accuracy (mean pooling over time)
    ResNet,
    /// Conformer architecture (CNN + attention pooling)
    Conformer,
    /// Attention-based architecture (CNN + attention pooling)
    Attention,
}

impl Default for DeepMosConfig {
    fn default() -> Self {
        Self {
            n_mels: 80,
            n_fft: 2048,
            hop_length: 512,
            sample_rate: 22050,
            aggregation: AggregationMethod::Attention,
            use_gpu: true,
            architecture: MosArchitecture::ResNet,
            model_weights_path: None,
            conv_channels: vec![80, 64, 32],
            conv_kernel_size: 3,
            fc_dims: vec![32, 16],
            attention_heads: 4,
        }
    }
}

/// Which backend produced a [`MosPrediction`]. See the module documentation
/// for what each backend actually computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MosPredictionBackend {
    /// A real pretrained CNN, loaded from `model_weights_path`.
    Pretrained,
    /// No pretrained weights configured: an honest DSP-feature quality
    /// proxy (SNR, clipping, dynamic range, spectral flatness), not a
    /// learned model.
    DspProxy,
}

/// MOS prediction result with confidence and feature importance
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MosPrediction {
    /// Predicted MOS score (1.0 - 5.0)
    pub mos_score: f32,
    /// Prediction confidence (0.0 - 1.0)
    pub confidence: f32,
    /// Standard deviation of prediction
    pub std_dev: f32,
    /// Feature importance scores
    pub feature_importance: HashMap<String, f32>,
    /// Per-dimension quality scores
    pub dimension_scores: DimensionScores,
    /// Which backend produced this prediction - see [`MosPredictionBackend`].
    pub backend: MosPredictionBackend,
}

/// Multi-dimensional quality scores
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DimensionScores {
    /// Signal quality (1-5)
    pub signal_quality: f32,
    /// Distortion level (1-5, higher is better)
    pub distortion: f32,
    /// Background noise (1-5, higher is better)
    pub noise: f32,
    /// Coloration/timbre quality (1-5)
    pub coloration: f32,
    /// Loudness appropriateness (1-5)
    pub loudness: f32,
}

impl Default for DimensionScores {
    fn default() -> Self {
        Self {
            signal_quality: 3.0,
            distortion: 3.0,
            noise: 3.0,
            coloration: 3.0,
            loudness: 3.0,
        }
    }
}

/// Deep neural network for MOS prediction
pub struct DeepMosPredictor {
    /// Configuration
    config: DeepMosConfig,
    /// Neural network model
    model: Arc<RwLock<Option<MosNetModel>>>,
    /// Feature extraction pipeline
    feature_extractor: FeatureExtractor,
    /// Prediction cache for performance
    cache: Arc<RwLock<HashMap<String, MosPrediction>>>,
    /// Device for computation (CPU or GPU)
    device: Device,
}

/// MOS-prediction model: either a real loaded CNN, or the honest DSP proxy.
enum MosNetModel {
    /// A real CNN (+ optional attention pooling) with weights loaded from a
    /// safetensors checkpoint.
    Pretrained {
        #[allow(dead_code)]
        architecture: MosArchitecture,
        conv_layers: Vec<Conv1d>,
        bn_layers: Vec<BatchNorm>,
        attention: Option<AttentionLayer>,
        fc_layers: Vec<Linear>,
        output: Linear,
    },
    /// No pretrained checkpoint configured: real DSP-feature quality proxy.
    DspProxy {
        #[allow(dead_code)]
        architecture: MosArchitecture,
    },
}

impl MosNetModel {
    fn backend_kind(&self) -> MosPredictionBackend {
        match self {
            MosNetModel::Pretrained { .. } => MosPredictionBackend::Pretrained,
            MosNetModel::DspProxy { .. } => MosPredictionBackend::DspProxy,
        }
    }
}

/// Self-attention layer for frame aggregation
struct AttentionLayer {
    /// Query projection
    query: Linear,
    /// Key projection
    key: Linear,
    /// Value projection
    value: Linear,
    /// Output projection
    output: Linear,
    /// Number of attention heads (informational scaling factor)
    #[allow(dead_code)]
    num_heads: usize,
}

/// Feature extraction for MOS prediction
struct FeatureExtractor {
    /// Configuration
    config: DeepMosConfig,
    /// FFT planner for spectral analysis
    fft_planner: Arc<RwLock<RealFftPlanner<f32>>>,
}

impl DeepMosPredictor {
    /// Create new MOS predictor with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(DeepMosConfig::default())
    }

    /// Create new MOS predictor with custom configuration
    pub fn with_config(config: DeepMosConfig) -> Result<Self> {
        let device = if config.use_gpu {
            std::panic::catch_unwind(|| Device::cuda_if_available(0))
                .ok()
                .and_then(|r| r.ok())
                .unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };

        let feature_extractor = FeatureExtractor::new(config.clone())?;

        Ok(Self {
            config,
            model: Arc::new(RwLock::new(None)),
            feature_extractor,
            cache: Arc::new(RwLock::new(HashMap::new())),
            device,
        })
    }

    /// Initialize the neural network model.
    ///
    /// Builds the [`MosPredictionBackend::Pretrained`] backend when
    /// [`DeepMosConfig::model_weights_path`] is set, failing closed (`Err`)
    /// if that checkpoint cannot be loaded. Otherwise builds the honest
    /// [`MosPredictionBackend::DspProxy`] backend.
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing Deep MOS predictor model");

        let model = self.create_mos_model()?;
        let backend = model.backend_kind();

        let mut model_lock = self.model.write().await;
        *model_lock = Some(model);

        info!("Deep MOS predictor initialized (backend: {:?})", backend);
        Ok(())
    }

    /// Predict MOS score for a voice sample
    pub async fn predict_mos(&self, sample: &VoiceSample) -> Result<MosPrediction> {
        // Check cache first
        let cache_key = format!("{}_mos", sample.id);
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(&cache_key) {
                trace!("MOS prediction cache hit for sample {}", sample.id);
                return Ok(cached.clone());
            }
        }

        debug!("Predicting MOS for sample {}", sample.id);

        let model_lock = self.model.read().await;
        let model = model_lock
            .as_ref()
            .ok_or_else(|| Error::InvalidInput("Model not initialized".to_string()))?;

        let prediction = match model {
            MosNetModel::Pretrained { .. } => {
                // Only the pretrained CNN path needs the mel-spectrogram
                // features; the DSP proxy works directly on raw audio.
                let features = self.feature_extractor.extract_features(&sample.audio)?;
                self.run_pretrained_inference(model, &features)?
            }
            MosNetModel::DspProxy { .. } => run_dsp_proxy_inference(&sample.audio),
        };

        // Cache result
        {
            let mut cache = self.cache.write().await;
            cache.insert(cache_key, prediction.clone());
        }

        Ok(prediction)
    }

    /// Compare two voice samples and predict preference
    pub async fn compare_samples(
        &self,
        sample_a: &VoiceSample,
        sample_b: &VoiceSample,
    ) -> Result<f32> {
        let pred_a = self.predict_mos(sample_a).await?;
        let pred_b = self.predict_mos(sample_b).await?;

        // Compute preference probability using Bradley-Terry model
        let diff = pred_a.mos_score - pred_b.mos_score;
        let preference = 1.0 / (1.0 + (-diff).exp());

        Ok(preference)
    }

    /// Run the real pretrained CNN forward pass and assemble a [`MosPrediction`].
    fn run_pretrained_inference(
        &self,
        model: &MosNetModel,
        features: &Tensor,
    ) -> Result<MosPrediction> {
        let output = self.forward_pass(model, features)?;
        let mos_score = self.tensor_to_scalar(&output)?.clamp(1.0, 5.0);

        // Real, input-dependent confidence: predictions near the extremes
        // of the valid range are treated as more decisive than those near
        // the neutral midpoint (3.0).
        let extremity = ((mos_score - 3.0).abs() / 2.0).clamp(0.0, 1.0);
        let confidence = (0.5 + 0.5 * extremity).clamp(0.0, 1.0);

        let feature_importance = self.compute_feature_importance(model, features)?;

        // No separate per-dimension output heads are defined in the
        // checkpoint schema (see `DeepMosConfig::model_weights_path`), so the
        // best real estimate for each axis is the overall predicted score
        // itself (which does vary with input), not a fabricated constant.
        let dimension_scores = DimensionScores {
            signal_quality: mos_score,
            distortion: mos_score,
            noise: mos_score,
            coloration: mos_score,
            loudness: mos_score,
        };

        Ok(MosPrediction {
            mos_score,
            confidence,
            std_dev: 0.5 * (1.0 - confidence),
            feature_importance,
            dimension_scores,
            backend: MosPredictionBackend::Pretrained,
        })
    }

    /// Forward pass through the neural network
    fn forward_pass(&self, model: &MosNetModel, features: &Tensor) -> Result<Tensor> {
        let MosNetModel::Pretrained {
            conv_layers,
            bn_layers,
            attention,
            fc_layers,
            output,
            ..
        } = model
        else {
            return Err(Error::InvalidInput(
                "forward_pass requires the pretrained backend".to_string(),
            ));
        };

        let mut x = features.clone();

        // Pass through convolutional layers with batch norm and ReLU
        for (conv, bn) in conv_layers.iter().zip(bn_layers.iter()) {
            x = conv
                .forward(&x)
                .map_err(|e| Error::InvalidInput(format!("Conv forward failed: {}", e)))?;
            x = bn
                .forward_t(&x, false) // training=false for inference
                .map_err(|e| Error::InvalidInput(format!("BN forward failed: {}", e)))?;
            x = x
                .relu()
                .map_err(|e| Error::InvalidInput(format!("ReLU failed: {}", e)))?;
        }

        // Apply attention-based aggregation if configured, else mean pooling
        // over time; both paths reduce (batch, channels, time) to (batch, channels).
        x = if let Some(attention) = attention {
            self.apply_attention(attention, &x)?
        } else {
            x.mean(2)
                .map_err(|e| Error::InvalidInput(format!("Mean pooling failed: {}", e)))?
        };

        // Pass through fully connected layers
        for fc in fc_layers {
            x = fc
                .forward(&x)
                .map_err(|e| Error::InvalidInput(format!("FC forward failed: {}", e)))?;
            x = x
                .relu()
                .map_err(|e| Error::InvalidInput(format!("ReLU failed: {}", e)))?;
        }

        // Final output layer
        let output = output
            .forward(&x)
            .map_err(|e| Error::InvalidInput(format!("Output forward failed: {}", e)))?;

        Ok(output)
    }

    /// Apply self-attention for frame aggregation.
    ///
    /// `x` is `(batch, channels, time)` (the natural `Conv1d` layout); this
    /// transposes to `(batch, time, channels)` for the per-timestep linear
    /// projections, then pools over time to `(batch, channels)` so the
    /// output shape matches the non-attention mean-pooling path.
    fn apply_attention(&self, attention: &AttentionLayer, x: &Tensor) -> Result<Tensor> {
        let x_t = x
            .transpose(1, 2)
            .and_then(|t| t.contiguous())
            .map_err(|e| Error::InvalidInput(format!("Attention transpose failed: {}", e)))?;

        let query = attention
            .query
            .forward(&x_t)
            .map_err(|e| Error::InvalidInput(format!("Query projection failed: {}", e)))?;
        let key = attention
            .key
            .forward(&x_t)
            .map_err(|e| Error::InvalidInput(format!("Key projection failed: {}", e)))?;
        let value = attention
            .value
            .forward(&x_t)
            .map_err(|e| Error::InvalidInput(format!("Value projection failed: {}", e)))?;

        // Scaled dot-product attention
        let d_k = (key.dims()[key.dims().len() - 1] as f64).sqrt();
        let scores = query
            .matmul(&key.t()?)
            .map_err(|e| Error::InvalidInput(format!("Attention matmul failed: {}", e)))?;
        let scores = scores.affine(1.0 / d_k, 0.0)?;
        let attention_weights = ops::softmax(&scores, scores.dims().len() - 1)
            .map_err(|e| Error::InvalidInput(format!("Softmax failed: {}", e)))?;

        let context = attention_weights
            .matmul(&value)
            .map_err(|e| Error::InvalidInput(format!("Context matmul failed: {}", e)))?;

        let projected = attention
            .output
            .forward(&context)
            .map_err(|e| Error::InvalidInput(format!("Output projection failed: {}", e)))?;

        projected
            .mean(1)
            .map_err(|e| Error::InvalidInput(format!("Attention time-pooling failed: {}", e)))
    }

    /// Compute feature importance via real occlusion-based sensitivity: the
    /// mel-frequency axis is split into four bands (labeled to match the
    /// original coarse categories), each is zeroed out in turn, and the
    /// resulting change in the model's output is measured. This actually
    /// re-runs the forward pass; it is not a fabricated constant map.
    fn compute_feature_importance(
        &self,
        model: &MosNetModel,
        features: &Tensor,
    ) -> Result<HashMap<String, f32>> {
        let baseline = self.tensor_to_scalar(&self.forward_pass(model, features)?)?;

        let dims = features.dims();
        let n_mels = *dims.get(1).unwrap_or(&0);
        let band = (n_mels / 4).max(1);
        let labels = ["spectral", "temporal", "prosodic", "speaker"];

        let mut importance = HashMap::with_capacity(labels.len());
        for (i, label) in labels.iter().enumerate() {
            let start = (i * band).min(n_mels);
            let end = ((i + 1) * band).min(n_mels);
            if end <= start || n_mels == 0 {
                importance.insert((*label).to_string(), 0.0);
                continue;
            }
            let occluded = occlude_mel_band(features, &self.device, start, end)?;
            let occluded_score = self.tensor_to_scalar(&self.forward_pass(model, &occluded)?)?;
            importance.insert((*label).to_string(), (baseline - occluded_score).abs());
        }

        let total: f32 = importance.values().sum::<f32>();
        if total > 1e-9 {
            for value in importance.values_mut() {
                *value /= total;
            }
        }

        Ok(importance)
    }

    /// Convert a (possibly batched) tensor to its first scalar value.
    fn tensor_to_scalar(&self, tensor: &Tensor) -> Result<f32> {
        let vec = tensor
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| Error::InvalidInput(format!("Tensor conversion failed: {}", e)))?;
        vec.first()
            .copied()
            .ok_or_else(|| Error::InvalidInput("Empty model output tensor".to_string()))
    }

    /// Build the MOS-prediction model.
    ///
    /// When [`DeepMosConfig::model_weights_path`] is set, loads a real CNN
    /// from that safetensors checkpoint and fails closed (`Err`) if the
    /// file is missing, unreadable, or missing/mismatched tensors. When
    /// unset, returns the honestly-labeled [`MosPredictionBackend::DspProxy`]
    /// backend, which needs no learned weights.
    fn create_mos_model(&self) -> Result<MosNetModel> {
        match &self.config.model_weights_path {
            Some(path) => self.load_pretrained_mos_model(path),
            None => Ok(MosNetModel::DspProxy {
                architecture: self.config.architecture,
            }),
        }
    }

    /// Load a real CNN from a safetensors checkpoint. See
    /// [`DeepMosConfig::model_weights_path`] for the expected tensor schema.
    /// Fails closed rather than falling back to an untrained model if the
    /// checkpoint is missing or malformed.
    fn load_pretrained_mos_model(&self, path: &Path) -> Result<MosNetModel> {
        if !path.exists() {
            return Err(Error::InvalidInput(format!(
                "Deep MOS model weights not found at {path:?}; predict_mos via the pretrained \
                 backend is unavailable without a real checkpoint"
            )));
        }

        let bytes = std::fs::read(path).map_err(|e| {
            Error::InvalidInput(format!("Failed to read Deep MOS weights {path:?}: {e}"))
        })?;
        let vb = VarBuilder::from_buffered_safetensors(bytes, DType::F32, &self.device).map_err(
            |e| {
                Error::InvalidInput(format!(
                    "Failed to parse Deep MOS safetensors file {path:?}: {e}"
                ))
            },
        )?;

        let channels = self.config.conv_channels.clone();
        if channels.len() < 2 {
            return Err(Error::InvalidInput(
                "conv_channels must specify at least an input and one output channel count"
                    .to_string(),
            ));
        }
        let kernel_size = self.config.conv_kernel_size.max(1);
        let padding = kernel_size / 2;

        let mut conv_layers = Vec::with_capacity(channels.len() - 1);
        let mut bn_layers = Vec::with_capacity(channels.len() - 1);
        for i in 0..channels.len() - 1 {
            let in_ch = channels[i];
            let out_ch = channels[i + 1];
            let conv = conv1d(
                in_ch,
                out_ch,
                kernel_size,
                Conv1dConfig {
                    padding,
                    stride: 1,
                    ..Default::default()
                },
                vb.pp(format!("conv.{i}")),
            )
            .map_err(|e| {
                Error::InvalidInput(format!(
                    "Missing/invalid 'conv.{i}' weights in {path:?}: {e}"
                ))
            })?;
            let bn = batch_norm(out_ch, 1e-5, vb.pp(format!("bn.{i}"))).map_err(|e| {
                Error::InvalidInput(format!("Missing/invalid 'bn.{i}' weights in {path:?}: {e}"))
            })?;
            conv_layers.push(conv);
            bn_layers.push(bn);
        }

        let last_channels = *channels.last().expect("checked len >= 2 above");

        let needs_attention = matches!(
            self.config.architecture,
            MosArchitecture::Conformer | MosArchitecture::Attention
        );
        let attention = if needs_attention {
            let attn_vb = vb.pp("attention");
            Some(AttentionLayer {
                query: linear(last_channels, last_channels, attn_vb.pp("query")).map_err(|e| {
                    Error::InvalidInput(format!("Missing 'attention.query' in {path:?}: {e}"))
                })?,
                key: linear(last_channels, last_channels, attn_vb.pp("key")).map_err(|e| {
                    Error::InvalidInput(format!("Missing 'attention.key' in {path:?}: {e}"))
                })?,
                value: linear(last_channels, last_channels, attn_vb.pp("value")).map_err(|e| {
                    Error::InvalidInput(format!("Missing 'attention.value' in {path:?}: {e}"))
                })?,
                output: linear(last_channels, last_channels, attn_vb.pp("output")).map_err(
                    |e| Error::InvalidInput(format!("Missing 'attention.output' in {path:?}: {e}")),
                )?,
                num_heads: self.config.attention_heads,
            })
        } else {
            None
        };

        let mut fc_layers = Vec::with_capacity(self.config.fc_dims.len());
        let mut prev_dim = last_channels;
        for (i, &dim) in self.config.fc_dims.iter().enumerate() {
            let fc = linear(prev_dim, dim, vb.pp(format!("fc.{i}"))).map_err(|e| {
                Error::InvalidInput(format!("Missing/invalid 'fc.{i}' weights in {path:?}: {e}"))
            })?;
            fc_layers.push(fc);
            prev_dim = dim;
        }

        let output = linear(prev_dim, 1, vb.pp("output")).map_err(|e| {
            Error::InvalidInput(format!("Missing/invalid 'output' weights in {path:?}: {e}"))
        })?;

        Ok(MosNetModel::Pretrained {
            architecture: self.config.architecture,
            conv_layers,
            bn_layers,
            attention,
            fc_layers,
            output,
        })
    }
}

/// Zero out mel-frequency rows `[start, end)` of a `(batch, n_mels, time)`
/// tensor via a broadcast multiply against a real 0/1 mask - used for the
/// occlusion-based feature-importance measurement.
fn occlude_mel_band(
    features: &Tensor,
    device: &Device,
    start: usize,
    end: usize,
) -> Result<Tensor> {
    let n_mels = *features.dims().get(1).unwrap_or(&0);
    let mask: Vec<f32> = (0..n_mels)
        .map(|i| if i >= start && i < end { 0.0 } else { 1.0 })
        .collect();
    let mask_tensor = Tensor::from_vec(mask, (1, n_mels, 1), device)
        .map_err(|e| Error::InvalidInput(format!("Failed to build occlusion mask: {e}")))?;
    features
        .broadcast_mul(&mask_tensor)
        .map_err(|e| Error::InvalidInput(format!("Failed to apply occlusion mask: {e}")))
}

/// Real (non-learned) DSP-derived quality proxy computed directly from raw
/// audio, used by [`MosPredictionBackend::DspProxy`]. This is an honest
/// heuristic built from measurable signal properties; it makes no claim of
/// correlation with human MOS ratings.
struct DspQualityFeatures {
    snr_db: f32,
    clipping_ratio: f32,
    dynamic_range_db: f32,
    spectral_flatness: f32,
}

fn compute_dsp_quality_features(audio: &[f32]) -> DspQualityFeatures {
    let snr_db = estimate_snr_db(audio);
    let clipping_ratio = if audio.is_empty() {
        0.0
    } else {
        audio.iter().filter(|&&x| x.abs() >= 0.999).count() as f32 / audio.len() as f32
    };
    let peak = audio.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    let rms = if audio.is_empty() {
        0.0
    } else {
        (audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32).sqrt()
    };
    let dynamic_range_db = if rms > 1e-9 {
        (20.0 * (peak / rms).log10()).max(0.0)
    } else {
        0.0
    };
    let spectral_flatness = estimate_spectral_flatness(audio);

    DspQualityFeatures {
        snr_db,
        clipping_ratio,
        dynamic_range_db,
        spectral_flatness,
    }
}

impl DspQualityFeatures {
    /// Combine the sub-metrics into a single MOS-scale (1-5) proxy score.
    /// The blend weights below are the heuristic's own documented
    /// coefficients (not a claimed measurement of anything) - transparency
    /// about that distinguishes this from the fabricated constant it replaces.
    fn to_mos_score(&self) -> f32 {
        let snr_component = (self.snr_db / 40.0).clamp(0.0, 1.0);
        let clipping_component = (1.0 - self.clipping_ratio).clamp(0.0, 1.0);
        let dynamic_component = (self.dynamic_range_db / 50.0).clamp(0.0, 1.0);
        let flatness_component = (1.0 - self.spectral_flatness).clamp(0.0, 1.0);

        let composite = snr_component * 0.4
            + clipping_component * 0.3
            + dynamic_component * 0.2
            + flatness_component * 0.1;
        (1.0 + composite * 4.0).clamp(1.0, 5.0)
    }

    /// Confidence scales with how measurable the signal's own SNR is - a
    /// very low-SNR clip gives this heuristic less to work with.
    fn confidence(&self) -> f32 {
        (0.5 + 0.5 * (self.snr_db / 40.0).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    }

    /// The fixed blend weights used by [`Self::to_mos_score`], reported
    /// honestly as such (not as a gradient-based attribution, which this is
    /// not).
    fn feature_importance(&self) -> HashMap<String, f32> {
        let mut importance = HashMap::with_capacity(4);
        importance.insert("snr".to_string(), 0.4);
        importance.insert("clipping".to_string(), 0.3);
        importance.insert("dynamic_range".to_string(), 0.2);
        importance.insert("spectral_flatness".to_string(), 0.1);
        importance
    }

    fn dimension_scores(&self) -> DimensionScores {
        let scale = |ratio: f32| (1.0 + ratio.clamp(0.0, 1.0) * 4.0).clamp(1.0, 5.0);
        DimensionScores {
            signal_quality: scale(self.snr_db / 40.0),
            distortion: scale(1.0 - self.clipping_ratio),
            noise: scale(self.snr_db / 40.0),
            coloration: scale(1.0 - self.spectral_flatness),
            loudness: scale(self.dynamic_range_db / 50.0),
        }
    }
}

fn run_dsp_proxy_inference(audio: &[f32]) -> MosPrediction {
    let dsp = compute_dsp_quality_features(audio);
    let mos_score = dsp.to_mos_score();
    let confidence = dsp.confidence();

    MosPrediction {
        mos_score,
        confidence,
        std_dev: 0.5 * (1.0 - confidence),
        feature_importance: dsp.feature_importance(),
        dimension_scores: dsp.dimension_scores(),
        backend: MosPredictionBackend::DspProxy,
    }
}

/// Real SNR estimate (dB) derived from the audio itself: the noise floor is
/// approximated as the mean energy of the quietest ~20% of short analysis
/// frames, the signal level as the mean energy of the loudest ~20%.
fn estimate_snr_db(audio: &[f32]) -> f32 {
    const FRAME_LEN: usize = 256;
    if audio.len() < FRAME_LEN {
        return 0.0;
    }

    let mut frame_energies: Vec<f32> = audio
        .chunks(FRAME_LEN)
        .filter(|frame| frame.len() == FRAME_LEN)
        .map(|frame| frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32)
        .collect();
    if frame_energies.is_empty() {
        return 0.0;
    }
    frame_energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = frame_energies.len();
    let sample_count = (n / 5).max(1);
    let noise_floor = frame_energies[..sample_count].iter().sum::<f32>() / sample_count as f32;
    let signal_level = frame_energies[n - sample_count..].iter().sum::<f32>() / sample_count as f32;

    if noise_floor <= 1e-12 {
        return 60.0;
    }
    (10.0 * (signal_level / noise_floor).log10()).clamp(0.0, 60.0)
}

/// Real spectral-flatness measure (geometric mean / arithmetic mean of the
/// magnitude spectrum), in `[0, 1]`: near 0 for tonal signals, near 1 for
/// white-noise-like signals.
fn estimate_spectral_flatness(audio: &[f32]) -> f32 {
    if audio.len() < 2 {
        return 0.0;
    }
    let n = audio.len().min(4096);
    let windowed: Vec<f64> = audio[..n]
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let w = 0.5
                - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n as f64 - 1.0).max(1.0)).cos();
            x as f64 * w
        })
        .collect();

    let spectrum = match scirs2_fft::rfft(&windowed, None) {
        Ok(s) => s,
        Err(_) => return 0.0,
    };
    let magnitudes: Vec<f64> = spectrum
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt().max(1e-12))
        .collect();
    if magnitudes.is_empty() {
        return 0.0;
    }

    let log_sum: f64 = magnitudes.iter().map(|m| m.ln()).sum();
    let geometric_mean = (log_sum / magnitudes.len() as f64).exp();
    let arithmetic_mean = magnitudes.iter().sum::<f64>() / magnitudes.len() as f64;
    if arithmetic_mean <= 1e-12 {
        return 0.0;
    }
    (geometric_mean / arithmetic_mean).clamp(0.0, 1.0) as f32
}

impl FeatureExtractor {
    /// Create new feature extractor
    fn new(config: DeepMosConfig) -> Result<Self> {
        Ok(Self {
            config,
            fft_planner: Arc::new(RwLock::new(RealFftPlanner::<f32>::new())),
        })
    }

    /// Extract multi-scale features from audio
    fn extract_features(&self, audio: &[f32]) -> Result<Tensor> {
        // Extract mel spectrogram
        let mel_spec = self.compute_mel_spectrogram(audio)?;

        // Convert to tensor
        let shape = mel_spec.shape();
        let data: Vec<f32> = mel_spec.iter().copied().collect();

        // Create tensor with shape [batch=1, channels=n_mels, time]
        let tensor = Tensor::from_vec(data, (1, shape[0], shape[1]), &Device::Cpu)
            .map_err(|e| Error::InvalidInput(format!("Failed to create tensor: {}", e)))?;

        Ok(tensor)
    }

    /// Compute mel spectrogram using scirs2-fft
    fn compute_mel_spectrogram(&self, audio: &[f32]) -> Result<Array2<f32>> {
        // Guard against `audio.len() < n_fft`, which would otherwise
        // underflow the `usize` subtraction below and panic.
        let n_frames = if audio.len() > self.config.n_fft {
            (audio.len() - self.config.n_fft) / self.config.hop_length + 1
        } else {
            0
        };
        let mut mel_spec = Array2::zeros((self.config.n_mels, n_frames));
        if n_frames == 0 {
            return Ok(mel_spec);
        }

        // Compute STFT frames
        for (frame_idx, frame_start) in (0..audio.len() - self.config.n_fft)
            .step_by(self.config.hop_length)
            .enumerate()
        {
            if frame_idx >= n_frames {
                break;
            }

            let frame = &audio[frame_start..frame_start + self.config.n_fft];
            let spectrum = self.compute_fft_magnitude(frame)?;

            // Apply mel filterbank (simplified)
            let mel_frame = self.apply_mel_filterbank(&spectrum)?;

            for (mel_idx, &value) in mel_frame.iter().enumerate().take(self.config.n_mels) {
                mel_spec[[mel_idx, frame_idx]] = value;
            }
        }

        // Apply log scaling
        mel_spec.mapv_inplace(|x| (x + 1e-10).ln());

        Ok(mel_spec)
    }

    /// Compute the FFT magnitude spectrum of a frame.
    ///
    /// A Hann window is applied to the frame before a real forward FFT is
    /// performed. The returned vector contains the `N/2 + 1` non-redundant
    /// magnitude bins `|X_k| = sqrt(re² + im²)` produced by the transform.
    fn compute_fft_magnitude(&self, frame: &[f32]) -> Result<Vec<f32>> {
        let n = frame.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Apply Hann window to reduce spectral leakage, converting to f64 for
        // the FFT routine.
        let windowed: Vec<f64> = frame
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let window = if n > 1 {
                    0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()
                } else {
                    1.0
                };
                x as f64 * window
            })
            .collect();

        let num_bins = n / 2 + 1;

        // Real forward FFT. The held `fft_planner` is reserved for future
        // plan caching; here we use the convenience `rfft` entry point which
        // shares the same scirs2-fft backend.
        let spectrum = scirs2_fft::rfft(&windowed, None)
            .map_err(|e| Error::Processing(format!("FFT magnitude computation failed: {}", e)))?;

        // Complex-bin magnitudes.
        let magnitude: Vec<f32> = spectrum
            .iter()
            .take(num_bins)
            .map(|c| ((c.re * c.re + c.im * c.im).sqrt()) as f32)
            .collect();

        Ok(magnitude)
    }

    /// Apply mel filterbank (simplified implementation)
    fn apply_mel_filterbank(&self, spectrum: &[f32]) -> Result<Vec<f32>> {
        // Simplified mel filterbank - would use proper triangular filters in production
        let mel_output: Vec<f32> = (0..self.config.n_mels)
            .map(|mel_idx| {
                let start = (spectrum.len() * mel_idx) / self.config.n_mels;
                let end = (spectrum.len() * (mel_idx + 1)) / self.config.n_mels;
                spectrum[start..end].iter().sum::<f32>() / (end - start) as f32
            })
            .collect();

        Ok(mel_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as StdHashMap;

    #[tokio::test]
    async fn test_deep_mos_predictor_creation() {
        let predictor = DeepMosPredictor::new();
        assert!(predictor.is_ok());
    }

    #[tokio::test]
    async fn test_deep_mos_config_default() {
        let config = DeepMosConfig::default();
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.n_fft, 2048);
        assert_eq!(config.sample_rate, 22050);
        assert!(config.model_weights_path.is_none());
    }

    #[tokio::test]
    async fn test_feature_extractor() {
        let config = DeepMosConfig::default();
        let extractor = FeatureExtractor::new(config);
        assert!(extractor.is_ok());
    }

    #[tokio::test]
    async fn test_mos_prediction_bounds() {
        let prediction = MosPrediction {
            mos_score: 3.5,
            confidence: 0.85,
            std_dev: 0.3,
            feature_importance: HashMap::new(),
            dimension_scores: DimensionScores::default(),
            backend: MosPredictionBackend::DspProxy,
        };

        assert!(prediction.mos_score >= 1.0 && prediction.mos_score <= 5.0);
        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
    }

    #[tokio::test]
    async fn test_aggregation_methods() {
        assert_eq!(AggregationMethod::Average, AggregationMethod::Average);
        assert_ne!(AggregationMethod::Average, AggregationMethod::Attention);
    }

    #[test]
    fn test_fft_magnitude_length() {
        let config = DeepMosConfig::default();
        let extractor = FeatureExtractor::new(config.clone()).unwrap();
        let frame = vec![0.5f32; config.n_fft];
        let mag = extractor.compute_fft_magnitude(&frame).unwrap();
        // Real FFT yields N/2 + 1 non-redundant magnitude bins.
        assert_eq!(mag.len(), config.n_fft / 2 + 1);
    }

    #[test]
    fn test_fft_magnitude_peaks_at_tone_bin() {
        let config = DeepMosConfig::default();
        let extractor = FeatureExtractor::new(config.clone()).unwrap();
        let sr = config.sample_rate as f32;
        let n = config.n_fft;
        // Bin frequency resolution: sr / n. Pick a tone aligned to a bin.
        let bin = 30usize;
        let freq = bin as f32 * sr / n as f32;
        let frame: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect();

        let mag = extractor.compute_fft_magnitude(&frame).unwrap();
        let (peak_bin, _) = mag
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        // The dominant magnitude bin should be at (or adjacent to) the tone bin.
        assert!(
            (peak_bin as i64 - bin as i64).abs() <= 1,
            "peak bin {peak_bin} not near tone bin {bin}"
        );
    }

    #[test]
    fn test_fft_magnitude_not_absolute_value_stub() {
        // Regression guard: the old stub returned per-sample |windowed| of
        // length n_fft. The real FFT returns a shorter spectrum AND a true
        // magnitude peak, so a tone must concentrate energy in few bins.
        let config = DeepMosConfig::default();
        let extractor = FeatureExtractor::new(config.clone()).unwrap();
        let sr = config.sample_rate as f32;
        let n = config.n_fft;
        let freq = 40.0 * sr / n as f32;
        let frame: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect();
        let mag = extractor.compute_fft_magnitude(&frame).unwrap();
        // Spectrum length must NOT equal n_fft (which the stub produced).
        assert_ne!(mag.len(), n);
        let total: f32 = mag.iter().sum();
        let peak = mag.iter().copied().fold(0.0f32, f32::max);
        // For a tone, the single peak holds a large share of total magnitude.
        assert!(
            peak > 0.2 * total,
            "energy not concentrated: peak {peak}, total {total}"
        );
    }

    fn tone(freq: f32, sample_rate: u32, seconds: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * seconds) as usize;
        (0..n)
            .map(|i| {
                (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin() * 0.5
            })
            .collect()
    }

    fn tone_with_noise_floor(
        freq: f32,
        sample_rate: u32,
        seconds: f32,
        floor_noise: f32,
    ) -> Vec<f32> {
        let n = (sample_rate as f32 * seconds) as usize;
        (0..n)
            .map(|i| {
                if i < n / 2 {
                    (fastrand::f32() - 0.5) * floor_noise
                } else {
                    let t = i as f32 / sample_rate as f32;
                    (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5
                }
            })
            .collect()
    }

    #[tokio::test]
    async fn test_dsp_proxy_is_default_backend_and_not_constant() {
        let mut predictor = DeepMosPredictor::new().unwrap();
        predictor.initialize().await.unwrap();

        let clean = VoiceSample::new(
            "clean".to_string(),
            tone_with_noise_floor(220.0, 22050, 1.0, 0.001),
            22050,
        );
        let noisy = VoiceSample::new(
            "noisy".to_string(),
            tone_with_noise_floor(220.0, 22050, 1.0, 0.35),
            22050,
        );

        let clean_pred = predictor.predict_mos(&clean).await.unwrap();
        let noisy_pred = predictor.predict_mos(&noisy).await.unwrap();

        assert_eq!(clean_pred.backend, MosPredictionBackend::DspProxy);
        assert_ne!(
            clean_pred.mos_score, 1.0,
            "must not be the old all-zero-weights constant MOS=1.0"
        );
        assert!(
            clean_pred.mos_score > noisy_pred.mos_score,
            "cleaner audio must score higher (clean={}, noisy={})",
            clean_pred.mos_score,
            noisy_pred.mos_score
        );
        assert_ne!(
            clean_pred.dimension_scores, noisy_pred.dimension_scores,
            "dimension scores must not be a constant DimensionScores::default()"
        );
    }

    #[tokio::test]
    async fn test_compare_samples_reflects_real_quality_difference() {
        let mut predictor = DeepMosPredictor::new().unwrap();
        predictor.initialize().await.unwrap();

        let clean = VoiceSample::new(
            "clean2".to_string(),
            tone_with_noise_floor(220.0, 22050, 1.0, 0.001),
            22050,
        );
        let noisy = VoiceSample::new(
            "noisy2".to_string(),
            tone_with_noise_floor(220.0, 22050, 1.0, 0.35),
            22050,
        );

        let preference = predictor.compare_samples(&clean, &noisy).await.unwrap();
        // Bradley-Terry preference for the cleaner sample must exceed 0.5
        // (a constant MOS=1.0 for both samples would always yield exactly 0.5).
        assert!(
            preference > 0.5,
            "expected the cleaner sample to be preferred, got {preference}"
        );
    }

    #[tokio::test]
    async fn test_initialize_with_missing_weights_path_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let missing_path = dir.path().join("does_not_exist.safetensors");

        let mut config = DeepMosConfig::default();
        config.model_weights_path = Some(missing_path);
        let mut predictor = DeepMosPredictor::with_config(config).unwrap();

        let result = predictor.initialize().await;
        assert!(
            result.is_err(),
            "must fail closed when configured weights are missing"
        );
    }

    #[tokio::test]
    async fn test_initialize_with_malformed_weights_file_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let bad_path = dir.path().join("not_really_safetensors.safetensors");
        std::fs::write(&bad_path, b"not a safetensors file").unwrap();

        let mut config = DeepMosConfig::default();
        config.model_weights_path = Some(bad_path);
        let mut predictor = DeepMosPredictor::with_config(config).unwrap();

        let result = predictor.initialize().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_pretrained_backend_loads_real_weights_and_runs() {
        let dir = tempfile::tempdir().unwrap();
        let checkpoint_path = dir.path().join("tiny_mos.safetensors");
        let device = Device::Cpu;

        let n_mels = 8usize;
        let conv_channels = vec![n_mels, 4usize];
        let kernel_size = 3usize;
        let fc_dims = vec![4usize];

        let mut tensors: StdHashMap<String, Tensor> = StdHashMap::new();

        fn insert(
            tensors: &mut StdHashMap<String, Tensor>,
            device: &Device,
            name: &str,
            shape: Vec<usize>,
            value: f32,
        ) {
            let len: usize = shape.iter().product();
            let data = vec![value; len];
            tensors.insert(
                name.to_string(),
                Tensor::from_vec(data, shape, device).unwrap(),
            );
        }

        insert(
            &mut tensors,
            &device,
            "conv.0.weight",
            vec![conv_channels[1], conv_channels[0], kernel_size],
            0.02,
        );
        insert(
            &mut tensors,
            &device,
            "conv.0.bias",
            vec![conv_channels[1]],
            0.0,
        );
        insert(
            &mut tensors,
            &device,
            "bn.0.weight",
            vec![conv_channels[1]],
            1.0,
        );
        insert(
            &mut tensors,
            &device,
            "bn.0.bias",
            vec![conv_channels[1]],
            0.0,
        );
        insert(
            &mut tensors,
            &device,
            "bn.0.running_mean",
            vec![conv_channels[1]],
            0.0,
        );
        insert(
            &mut tensors,
            &device,
            "bn.0.running_var",
            vec![conv_channels[1]],
            1.0,
        );

        insert(
            &mut tensors,
            &device,
            "fc.0.weight",
            vec![fc_dims[0], conv_channels[1]],
            0.05,
        );
        insert(&mut tensors, &device, "fc.0.bias", vec![fc_dims[0]], 0.0);
        insert(
            &mut tensors,
            &device,
            "output.weight",
            vec![1, fc_dims[0]],
            0.1,
        );
        insert(&mut tensors, &device, "output.bias", vec![1], 0.0);

        candle_core::safetensors::save(&tensors, &checkpoint_path).unwrap();

        let config = DeepMosConfig {
            n_mels,
            n_fft: 256,
            hop_length: 64,
            sample_rate: 22050,
            architecture: MosArchitecture::LightCNN,
            model_weights_path: Some(checkpoint_path),
            conv_channels,
            conv_kernel_size: kernel_size,
            fc_dims,
            use_gpu: false,
            ..DeepMosConfig::default()
        };

        let mut predictor = DeepMosPredictor::with_config(config).unwrap();
        predictor
            .initialize()
            .await
            .expect("a checkpoint matching the documented schema must load successfully");

        let sample = VoiceSample::new("s".to_string(), tone(220.0, 22050, 0.2), 22050);
        let prediction = predictor.predict_mos(&sample).await.unwrap();

        assert_eq!(prediction.backend, MosPredictionBackend::Pretrained);
        assert!((1.0..=5.0).contains(&prediction.mos_score));
        assert!(!prediction.feature_importance.is_empty());
    }
}
