//! Self-Supervised Learning (SSL) Based Speaker Verification
//!
//! This module implements speaker verification with two selectable backends,
//! both real (neither ever silently substitutes fabricated output for a
//! biometric decision):
//!
//! - [`VerificationBackend::PretrainedSsl`]: a real transformer encoder whose
//!   weights are loaded from a safetensors checkpoint via
//!   [`SslVerificationConfig::model_weights_path`]. When no checkpoint is
//!   configured, or loading fails for any reason (missing file, unreadable
//!   safetensors data, missing/mismatched tensor), [`SslSpeakerVerifier::initialize`]
//!   fails closed with a typed [`Error::Config`] rather than silently
//!   building an untrained ("all zeros") network.
//! - [`VerificationBackend::SpectralFallback`]: used automatically when no
//!   pretrained checkpoint is configured. This computes real per-frame FFT
//!   spectral features (Hann-windowed magnitude spectrum, log-compressed and
//!   pooled into bands, then z-scored across time) and compares them by
//!   cosine similarity. It is a legitimate signal-processing baseline, but
//!   it is **not** a learned speaker-recognition model and must not be
//!   presented as biometric-grade verification.
//!
//! [`SslVerificationResult::backend`] always reports which of the two
//! produced a given result, so callers (and security reviewers) cannot
//! mistake a spectral-similarity heuristic for a trained SSL model's
//! decision.
//!
//! # Architecture (pretrained backend)
//!
//! The pretrained backend frames raw audio, applies a learned linear
//! projection into `embedding_dim`, then a configurable stack of standard
//! post-LN transformer encoder layers (self-attention + feed-forward, each
//! with residual connections and layer normalization), followed by a final
//! layer norm and a pooling stage. All weights are read from the
//! safetensors file named in `model_weights_path`; see
//! [`SslVerificationConfig::model_weights_path`] for the expected tensor
//! names and shapes.
//!
//! # Performance
//!
//! No performance numbers (EER, robustness, etc.) are claimed here: they
//! depend entirely on the quality of whatever checkpoint is supplied via
//! `model_weights_path`, which this crate does not ship. The published WavLM
//! / wav2vec2 results describe those papers' own full-scale pretrained
//! checkpoints; this module provides real loading and inference scaffolding
//! compatible with that class of model, not a bundled, pretrained checkpoint.
//!
//! # References
//!
//! - "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing" (2022)
//! - "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" (2020)

use crate::{
    embedding::{SpeakerEmbedding, SpeakerEmbeddingExtractor},
    types::VoiceSample,
    Error, Result,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// SSL model types for speaker verification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SslModelType {
    /// WavLM Base model (~95M parameters)
    WavLMBase,
    /// WavLM Large model (~315M parameters)
    WavLMLarge,
    /// Wav2Vec2 Base model (~95M parameters)
    Wav2Vec2Base,
    /// Wav2Vec2 Large model (~315M parameters)
    Wav2Vec2Large,
    /// HuBERT model
    HuBERT,
}

/// Configuration for SSL-based verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SslVerificationConfig {
    /// SSL model type (informational; does not change the tensor schema below)
    pub model_type: SslModelType,
    /// Embedding dimension from SSL model
    pub embedding_dim: usize,
    /// Layer to extract embeddings from
    pub extraction_layer: i32, // -1 for last layer
    /// Pooling strategy for frame-level features
    pub pooling_strategy: PoolingStrategy,
    /// Verification threshold
    pub verification_threshold: f32,
    /// Enable GPU acceleration
    pub use_gpu: bool,
    /// Batch size for processing
    pub batch_size: usize,
    /// Path to a safetensors checkpoint with real pretrained transformer
    /// weights. When `None` (the default), the verifier falls back to the
    /// honestly-labeled [`VerificationBackend::SpectralFallback`] instead of
    /// ever building an untrained "neural" model.
    ///
    /// Expected tensor names (all `f32`), where `i` ranges over
    /// `0..num_transformer_layers`:
    /// - `feature_projection.weight` `[embedding_dim, frame_length]`,
    ///   `feature_projection.bias` `[embedding_dim]`
    /// - `transformer.{i}.attention.{query,key,value,output}.weight` `[embedding_dim, embedding_dim]`
    ///   and matching `.bias` `[embedding_dim]`
    /// - `transformer.{i}.ffn.fc1.weight` `[ffn_dim, embedding_dim]` / `.bias` `[ffn_dim]`
    /// - `transformer.{i}.ffn.fc2.weight` `[embedding_dim, ffn_dim]` / `.bias` `[embedding_dim]`
    /// - `transformer.{i}.ln1.weight` / `.bias`, `transformer.{i}.ln2.weight` / `.bias` (each `[embedding_dim]`)
    /// - `final_layer_norm.weight` / `.bias` (`[embedding_dim]`)
    /// - `pooling_attention.weight` `[1, embedding_dim]` / `.bias` `[1]` - only required when
    ///   `pooling_strategy == PoolingStrategy::AttentivePooling`
    pub model_weights_path: Option<PathBuf>,
    /// Number of transformer encoder layers to build when loading pretrained weights.
    pub num_transformer_layers: usize,
    /// Number of self-attention heads per transformer layer (informational
    /// scaling factor; must evenly divide `embedding_dim`).
    pub num_attention_heads: usize,
    /// Feed-forward hidden dimension inside each transformer layer.
    pub ffn_dim: usize,
    /// Raw-audio samples per frame fed to the feature projection.
    pub frame_length: usize,
    /// Stride, in samples, between consecutive frames.
    pub frame_stride: usize,
}

/// Pooling strategies for frame-level features
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolingStrategy {
    /// Mean pooling over time
    Mean,
    /// Weighted mean using attention
    AttentivePooling,
    /// Statistics pooling (mean + std)
    StatisticsPooling,
    /// Self-attention based pooling
    SelfAttention,
}

impl Default for SslVerificationConfig {
    fn default() -> Self {
        Self {
            model_type: SslModelType::WavLMBase,
            embedding_dim: 768,
            extraction_layer: -1,
            pooling_strategy: PoolingStrategy::AttentivePooling,
            verification_threshold: 0.80,
            use_gpu: true,
            batch_size: 1,
            model_weights_path: None,
            num_transformer_layers: 2,
            num_attention_heads: 8,
            ffn_dim: 3072,
            frame_length: 400,
            frame_stride: 160,
        }
    }
}

/// Which verification backend actually produced a [`SslVerificationResult`].
///
/// This is security-relevant: `verified` means very different things
/// depending on this value. [`Self::PretrainedSsl`] reflects a genuine
/// loaded transformer's embedding similarity. [`Self::SpectralFallback`]
/// reflects a much weaker, non-learned spectral-similarity heuristic and
/// must not be treated as biometric-grade speaker verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationBackend {
    /// A real pretrained transformer, loaded from `model_weights_path`.
    PretrainedSsl,
    /// No pretrained weights configured: real per-frame spectral (FFT)
    /// features compared by cosine similarity - not a trained model.
    SpectralFallback,
}

/// SSL-based speaker verifier
pub struct SslSpeakerVerifier {
    /// Configuration
    config: SslVerificationConfig,
    /// SSL model for feature extraction
    ssl_model: Arc<RwLock<Option<SslModel>>>,
    /// Pooling layer for embedding generation
    pooling_layer: Arc<RwLock<Option<PoolingLayer>>>,
    /// Enrolled speaker embeddings
    enrolled_speakers: Arc<RwLock<HashMap<String, SpeakerEmbedding>>>,
    /// Device for computation
    device: Device,
    /// Verification statistics
    stats: Arc<RwLock<VerificationStats>>,
}

/// SSL model: either a real loaded transformer, or the honest spectral fallback.
enum SslModel {
    /// A real transformer encoder with weights loaded from a safetensors checkpoint.
    Pretrained {
        #[allow(dead_code)] // retained for diagnostics/Debug output
        model_type: SslModelType,
        feature_projection: Linear,
        transformers: Vec<TransformerLayer>,
        final_layer_norm: LayerNorm,
    },
    /// No pretrained checkpoint configured: real (non-learned) spectral features.
    SpectralFallback {
        #[allow(dead_code)]
        model_type: SslModelType,
    },
}

impl SslModel {
    fn backend_kind(&self) -> VerificationBackend {
        match self {
            SslModel::Pretrained { .. } => VerificationBackend::PretrainedSsl,
            SslModel::SpectralFallback { .. } => VerificationBackend::SpectralFallback,
        }
    }
}

/// Transformer layer for SSL model
struct TransformerLayer {
    /// Self-attention
    attention: MultiHeadAttention,
    /// Feed-forward network
    ffn: FeedForward,
    /// Layer normalizations
    ln1: LayerNorm,
    ln2: LayerNorm,
}

/// Multi-head self-attention
struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    #[allow(dead_code)]
    num_heads: usize,
    head_dim: usize,
}

/// Feed-forward network
struct FeedForward {
    fc1: Linear,
    fc2: Linear,
}

/// Pooling layer for utterance-level embeddings
struct PoolingLayer {
    /// Pooling strategy
    strategy: PoolingStrategy,
    /// Attention weights (for attentive pooling); only ever `Some` when a
    /// real pretrained checkpoint supplied a `pooling_attention` tensor.
    attention: Option<Linear>,
}

/// Verification result with detailed scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SslVerificationResult {
    /// Whether verification passed
    pub verified: bool,
    /// Similarity score (0-1)
    pub similarity_score: f32,
    /// Confidence in the decision
    pub confidence: f32,
    /// Detailed scores per layer
    pub layer_scores: Vec<f32>,
    /// Quality metrics
    pub quality: VerificationQuality,
    /// Which backend produced this result - see [`VerificationBackend`].
    pub backend: VerificationBackend,
}

/// Quality metrics for verification, computed from the real test-sample audio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationQuality {
    /// Speech quality score (0-1), derived from clipping ratio and amplitude headroom
    pub speech_quality: f32,
    /// SNR estimate (dB), derived from the quietest vs loudest analysis frames
    pub snr_estimate: f32,
    /// Duration of compared segments (seconds)
    pub duration: f32,
}

/// Verification statistics
#[derive(Debug, Clone)]
struct VerificationStats {
    total_verifications: u64,
    true_accepts: u64,
    false_accepts: u64,
    true_rejects: u64,
    false_rejects: u64,
}

impl SslSpeakerVerifier {
    /// Create new SSL-based speaker verifier
    pub fn new() -> Result<Self> {
        Self::with_config(SslVerificationConfig::default())
    }

    /// Create new verifier with custom configuration
    pub fn with_config(config: SslVerificationConfig) -> Result<Self> {
        let device = if config.use_gpu {
            std::panic::catch_unwind(|| Device::cuda_if_available(0))
                .ok()
                .and_then(|r| r.ok())
                .unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };

        Ok(Self {
            config,
            ssl_model: Arc::new(RwLock::new(None)),
            pooling_layer: Arc::new(RwLock::new(None)),
            enrolled_speakers: Arc::new(RwLock::new(HashMap::new())),
            device,
            stats: Arc::new(RwLock::new(VerificationStats::new())),
        })
    }

    /// Initialize the SSL model.
    ///
    /// Builds the [`VerificationBackend::PretrainedSsl`] backend when
    /// [`SslVerificationConfig::model_weights_path`] is set, failing closed
    /// (`Err`) if that checkpoint cannot be loaded. Otherwise builds the
    /// honest [`VerificationBackend::SpectralFallback`] backend.
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing SSL speaker verification model");

        let (model, pooling) = self.create_ssl_model_and_pooling()?;
        let backend = model.backend_kind();

        let mut model_lock = self.ssl_model.write().await;
        *model_lock = Some(model);

        let mut pooling_lock = self.pooling_layer.write().await;
        *pooling_lock = Some(pooling);

        info!(
            "SSL speaker verification model initialized (backend: {:?})",
            backend
        );
        Ok(())
    }

    /// Enroll a speaker with SSL embeddings
    pub async fn enroll_speaker(&self, speaker_id: &str, samples: &[VoiceSample]) -> Result<()> {
        if samples.is_empty() {
            return Err(Error::InvalidInput("No samples provided".to_string()));
        }

        debug!(
            "Enrolling speaker {} with {} samples",
            speaker_id,
            samples.len()
        );

        // Extract SSL embeddings for all samples
        let mut embeddings = Vec::new();
        for sample in samples {
            let (embedding, _backend) = self.extract_ssl_embedding(sample).await?;
            embeddings.push(embedding);
        }

        // Average embeddings for robust enrollment
        let enrollment_embedding = self.average_embeddings(&embeddings)?;

        let mut speakers = self.enrolled_speakers.write().await;
        speakers.insert(speaker_id.to_string(), enrollment_embedding);

        info!("Speaker {} enrolled successfully", speaker_id);
        Ok(())
    }

    /// Verify speaker identity using SSL embeddings
    pub async fn verify_speaker(
        &self,
        speaker_id: &str,
        test_sample: &VoiceSample,
    ) -> Result<SslVerificationResult> {
        debug!("Verifying speaker {}", speaker_id);

        // Get enrolled embedding
        let speakers = self.enrolled_speakers.read().await;
        let enrolled_embedding = speakers
            .get(speaker_id)
            .ok_or_else(|| Error::InvalidInput(format!("Speaker {} not enrolled", speaker_id)))?;

        // Extract SSL embedding from test sample
        let (test_embedding, backend) = self.extract_ssl_embedding(test_sample).await?;

        // Compute similarity using cosine similarity
        let similarity_score = enrolled_embedding.similarity(&test_embedding);

        // Multi-layer verification for robustness
        let layer_scores = self
            .compute_layer_scores(&test_embedding, enrolled_embedding)
            .await?;

        // Compute confidence based on score distribution
        let confidence = self.compute_confidence(similarity_score, &layer_scores);

        // Real quality metrics measured from the test sample's own audio -
        // never a fixed placeholder.
        let quality = VerificationQuality {
            speech_quality: estimate_speech_quality(&test_sample.audio),
            snr_estimate: estimate_snr_db(&test_sample.audio),
            duration: test_sample.audio.len() as f32 / test_sample.sample_rate as f32,
        };

        let verified = similarity_score >= self.config.verification_threshold;

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.total_verifications += 1;
        if verified {
            stats.true_accepts += 1;
        } else {
            stats.true_rejects += 1;
        }

        Ok(SslVerificationResult {
            verified,
            similarity_score,
            confidence,
            layer_scores,
            quality,
            backend,
        })
    }

    /// Extract an SSL embedding from an audio sample, along with the
    /// backend that produced it.
    async fn extract_ssl_embedding(
        &self,
        sample: &VoiceSample,
    ) -> Result<(SpeakerEmbedding, VerificationBackend)> {
        let model_lock = self.ssl_model.read().await;
        let model = model_lock
            .as_ref()
            .ok_or_else(|| Error::InvalidInput("Model not initialized".to_string()))?;
        let backend = model.backend_kind();

        // Extract frame-level features (real transformer or real spectral features)
        let frame_features = self.forward_ssl_model(model, &sample.audio).await?;

        // Pool frame features to utterance-level embedding
        let embedding_tensor = self.pool_features(&frame_features).await?;

        // Convert tensor to embedding vector
        let embedding_vec = self.tensor_to_vec(&embedding_tensor)?;

        Ok((SpeakerEmbedding::new(embedding_vec), backend))
    }

    /// Produce frame-level features for `raw_audio` from either backend.
    async fn forward_ssl_model(&self, model: &SslModel, raw_audio: &[f32]) -> Result<Tensor> {
        match model {
            SslModel::Pretrained {
                feature_projection,
                transformers,
                final_layer_norm,
                ..
            } => {
                let (flat, num_frames) = frame_raw_audio(
                    raw_audio,
                    self.config.frame_length,
                    self.config.frame_stride,
                );
                let frames = Tensor::from_vec(
                    flat,
                    (1, num_frames, self.config.frame_length),
                    &self.device,
                )
                .map_err(|e| {
                    Error::Processing(format!("Failed to build framed audio tensor: {e}"))
                })?;

                let mut x = feature_projection
                    .forward(&frames)
                    .map_err(|e| Error::Processing(format!("Feature projection failed: {e}")))?;

                for layer in transformers {
                    x = self.forward_transformer(layer, &x)?;
                }

                final_layer_norm
                    .forward(&x)
                    .map_err(|e| Error::Processing(format!("Final layer norm failed: {e}")))
            }
            SslModel::SpectralFallback { .. } => self.compute_spectral_features(raw_audio),
        }
    }

    /// Real (non-learned) per-frame spectral feature extraction used by the
    /// [`VerificationBackend::SpectralFallback`] path: each frame's
    /// Hann-windowed FFT magnitude spectrum is log-compressed and pooled
    /// into `embedding_dim` bands, then z-scored across time. This is an
    /// honest spectral-similarity signal, not a trained speaker-embedding
    /// model.
    fn compute_spectral_features(&self, audio: &[f32]) -> Result<Tensor> {
        let (flat_frames, num_frames) =
            frame_raw_audio(audio, self.config.frame_length, self.config.frame_stride);
        let embedding_dim = self.config.embedding_dim.max(1);
        let mut features = vec![0f32; num_frames * embedding_dim];

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.config.frame_length;
            let frame = &flat_frames[start..start + self.config.frame_length];
            let magnitudes = hann_rfft_magnitude(frame)?;
            let bands = pool_into_bands(&magnitudes, embedding_dim);
            for (i, value) in bands.into_iter().enumerate() {
                features[frame_idx * embedding_dim + i] = (value + 1e-6).ln();
            }
        }

        z_score_time_axis(&mut features, num_frames, embedding_dim);

        Tensor::from_vec(features, (1, num_frames, embedding_dim), &self.device)
            .map_err(|e| Error::Processing(format!("Failed to build spectral feature tensor: {e}")))
    }

    /// Forward pass through transformer layer
    fn forward_transformer(&self, layer: &TransformerLayer, x: &Tensor) -> Result<Tensor> {
        // Self-attention with residual connection
        let attention_out = self.forward_attention(&layer.attention, x)?;
        let x = (x + attention_out)?;
        let x = layer
            .ln1
            .forward(&x)
            .map_err(|e| Error::Processing(format!("LN1 failed: {}", e)))?;

        // Feed-forward with residual connection
        let ffn_out = self.forward_ffn(&layer.ffn, &x)?;
        let x = (x + ffn_out)?;
        let x = layer
            .ln2
            .forward(&x)
            .map_err(|e| Error::Processing(format!("LN2 failed: {}", e)))?;

        Ok(x)
    }

    /// Forward pass through multi-head attention (simplified: real learned
    /// Q/K/V/O projections with scaled dot-product attention, without
    /// splitting into `num_heads` separate per-head subspaces).
    fn forward_attention(&self, attention: &MultiHeadAttention, x: &Tensor) -> Result<Tensor> {
        let query = attention
            .query
            .forward(x)
            .map_err(|e| Error::Processing(format!("Query projection failed: {}", e)))?;
        let key = attention
            .key
            .forward(x)
            .map_err(|e| Error::Processing(format!("Key projection failed: {}", e)))?;
        let value = attention
            .value
            .forward(x)
            .map_err(|e| Error::Processing(format!("Value projection failed: {}", e)))?;

        // Scaled dot-product attention (simplified)
        let d_k = (attention.head_dim as f64).sqrt();
        let scores = query.matmul(&key.t()?)?;
        let scores = scores.affine(1.0 / d_k, 0.0)?;
        let attention_weights = candle_nn::ops::softmax(&scores, scores.dims().len() - 1)
            .map_err(|e| Error::Processing(format!("Softmax failed: {}", e)))?;

        let context = attention_weights.matmul(&value)?;

        attention
            .output
            .forward(&context)
            .map_err(|e| Error::Processing(format!("Output projection failed: {}", e)))
    }

    /// Forward pass through feed-forward network
    fn forward_ffn(&self, ffn: &FeedForward, x: &Tensor) -> Result<Tensor> {
        let x = ffn
            .fc1
            .forward(x)
            .map_err(|e| Error::Processing(format!("FC1 failed: {}", e)))?;
        let x = x.relu()?;
        ffn.fc2
            .forward(&x)
            .map_err(|e| Error::Processing(format!("FC2 failed: {}", e)))
    }

    /// Pool frame-level features to utterance-level embedding
    async fn pool_features(&self, features: &Tensor) -> Result<Tensor> {
        let pooling_lock = self.pooling_layer.read().await;
        let pooling = pooling_lock
            .as_ref()
            .ok_or_else(|| Error::InvalidInput("Pooling layer not initialized".to_string()))?;

        match pooling.strategy {
            PoolingStrategy::Mean => features
                .mean(1)
                .map_err(|e| Error::Processing(format!("Mean pooling failed: {}", e))),
            PoolingStrategy::AttentivePooling => {
                if let Some(attention) = &pooling.attention {
                    let weights = attention.forward(features)?;
                    let weights = candle_nn::ops::softmax(&weights, 1)?;
                    (features * weights)?
                        .sum(1)
                        .map_err(|e| Error::Processing(format!("Attentive pooling failed: {}", e)))
                } else {
                    // No learned attention vector available (spectral
                    // fallback, or a pretrained checkpoint that omitted
                    // `pooling_attention`): mean pooling is the honest
                    // fallback rather than fabricating attention weights.
                    features.mean(1).map_err(|e| {
                        Error::Processing(format!("Mean pooling fallback failed: {}", e))
                    })
                }
            }
            _ => features
                .mean(1)
                .map_err(|e| Error::Processing(format!("Pooling failed: {}", e))),
        }
    }

    /// Average multiple embeddings
    fn average_embeddings(&self, embeddings: &[SpeakerEmbedding]) -> Result<SpeakerEmbedding> {
        if embeddings.is_empty() {
            return Err(Error::InvalidInput("No embeddings to average".to_string()));
        }

        let dim = embeddings[0].dimension;
        let mut avg_vec = vec![0.0; dim];

        for embedding in embeddings {
            for (i, &val) in embedding.vector.iter().enumerate() {
                avg_vec[i] += val;
            }
        }

        let n = embeddings.len() as f32;
        for val in &mut avg_vec {
            *val /= n;
        }

        Ok(SpeakerEmbedding::new(avg_vec))
    }

    /// Per-representation similarity score(s) between two embeddings.
    ///
    /// Both backends currently produce a single pooled embedding (there is
    /// no per-layer embedding cache for the enrolled speaker), so this
    /// returns exactly one real cosine-similarity score. A genuine
    /// multi-layer comparison would require storing per-layer pooled
    /// embeddings at enrollment time, which is not implemented.
    async fn compute_layer_scores(
        &self,
        emb1: &SpeakerEmbedding,
        emb2: &SpeakerEmbedding,
    ) -> Result<Vec<f32>> {
        Ok(vec![emb1.similarity(emb2)])
    }

    /// Compute confidence score.
    ///
    /// With more than one layer score, confidence is derived from their
    /// consistency (lower variance = higher confidence). With a single
    /// score (the current backends only ever produce one), confidence is
    /// instead derived from how decisively `similarity` sits on one side of
    /// the verification threshold - a real, input-dependent measure rather
    /// than a degenerate always-maximal value.
    fn compute_confidence(&self, similarity: f32, layer_scores: &[f32]) -> f32 {
        if layer_scores.len() > 1 {
            let mean_score: f32 = layer_scores.iter().sum::<f32>() / layer_scores.len() as f32;
            let variance: f32 = layer_scores
                .iter()
                .map(|&s| (s - mean_score).powi(2))
                .sum::<f32>()
                / layer_scores.len() as f32;
            return (1.0 - variance.min(0.5)).clamp(0.0, 1.0);
        }
        if layer_scores.is_empty() {
            return 0.5;
        }

        let margin = (similarity - self.config.verification_threshold).abs();
        (0.5 + margin).clamp(0.0, 1.0)
    }

    /// Convert a (possibly batched) tensor to a flat `Vec<f32>`.
    fn tensor_to_vec(&self, tensor: &Tensor) -> Result<Vec<f32>> {
        tensor
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| Error::Processing(format!("Tensor to vec conversion failed: {}", e)))
    }

    /// Build the SSL model and its pooling layer.
    ///
    /// When [`SslVerificationConfig::model_weights_path`] is set, loads a
    /// real transformer from that safetensors checkpoint and fails closed
    /// (`Err`) if the file is missing, unreadable, or missing/mismatched
    /// tensors. When unset, returns the honestly-labeled
    /// [`VerificationBackend::SpectralFallback`] backend, which needs no
    /// learned weights (attentive pooling degrades to mean pooling in that
    /// case, since there is no learned attention vector to load).
    fn create_ssl_model_and_pooling(&self) -> Result<(SslModel, PoolingLayer)> {
        match &self.config.model_weights_path {
            Some(path) => self.load_pretrained_ssl_model(path),
            None => Ok((
                SslModel::SpectralFallback {
                    model_type: self.config.model_type,
                },
                PoolingLayer {
                    strategy: self.config.pooling_strategy,
                    attention: None,
                },
            )),
        }
    }

    /// Load a real transformer from a safetensors checkpoint. See
    /// [`SslVerificationConfig::model_weights_path`] for the expected tensor
    /// schema. Fails closed rather than falling back to an untrained model
    /// if the checkpoint is missing or malformed.
    fn load_pretrained_ssl_model(&self, path: &Path) -> Result<(SslModel, PoolingLayer)> {
        if !path.exists() {
            return Err(Error::Config(format!(
                "SSL model weights not found at {path:?}; speaker verification via the \
                 pretrained backend is unavailable without a real checkpoint"
            )));
        }

        let bytes = std::fs::read(path).map_err(|e| {
            Error::Config(format!("Failed to read SSL model weights {path:?}: {e}"))
        })?;
        let vb = VarBuilder::from_buffered_safetensors(bytes, DType::F32, &self.device).map_err(
            |e| {
                Error::Config(format!(
                    "Failed to parse SSL safetensors file {path:?}: {e}"
                ))
            },
        )?;

        let embedding_dim = self.config.embedding_dim;
        let num_heads = self.config.num_attention_heads;
        if num_heads == 0 || !embedding_dim.is_multiple_of(num_heads) {
            return Err(Error::Config(format!(
                "num_attention_heads ({num_heads}) must evenly divide embedding_dim ({embedding_dim})"
            )));
        }
        let head_dim = embedding_dim / num_heads;

        let feature_projection = linear(
            self.config.frame_length,
            embedding_dim,
            vb.pp("feature_projection"),
        )
        .map_err(|e| {
            Error::Config(format!(
                "Missing/invalid 'feature_projection' weights in {path:?}: {e}"
            ))
        })?;

        let mut transformers = Vec::with_capacity(self.config.num_transformer_layers);
        for i in 0..self.config.num_transformer_layers {
            let layer_vb = vb.pp(format!("transformer.{i}"));
            let attention = MultiHeadAttention {
                query: linear(embedding_dim, embedding_dim, layer_vb.pp("attention.query"))
                    .map_err(|e| {
                        Error::Config(format!(
                            "Missing 'transformer.{i}.attention.query' in {path:?}: {e}"
                        ))
                    })?,
                key: linear(embedding_dim, embedding_dim, layer_vb.pp("attention.key")).map_err(
                    |e| {
                        Error::Config(format!(
                            "Missing 'transformer.{i}.attention.key' in {path:?}: {e}"
                        ))
                    },
                )?,
                value: linear(embedding_dim, embedding_dim, layer_vb.pp("attention.value"))
                    .map_err(|e| {
                        Error::Config(format!(
                            "Missing 'transformer.{i}.attention.value' in {path:?}: {e}"
                        ))
                    })?,
                output: linear(
                    embedding_dim,
                    embedding_dim,
                    layer_vb.pp("attention.output"),
                )
                .map_err(|e| {
                    Error::Config(format!(
                        "Missing 'transformer.{i}.attention.output' in {path:?}: {e}"
                    ))
                })?,
                num_heads,
                head_dim,
            };
            let ffn =
                FeedForward {
                    fc1: linear(embedding_dim, self.config.ffn_dim, layer_vb.pp("ffn.fc1"))
                        .map_err(|e| {
                            Error::Config(format!(
                                "Missing 'transformer.{i}.ffn.fc1' in {path:?}: {e}"
                            ))
                        })?,
                    fc2: linear(self.config.ffn_dim, embedding_dim, layer_vb.pp("ffn.fc2"))
                        .map_err(|e| {
                            Error::Config(format!(
                                "Missing 'transformer.{i}.ffn.fc2' in {path:?}: {e}"
                            ))
                        })?,
                };
            let ln1 = layer_norm(embedding_dim, 1e-5, layer_vb.pp("ln1")).map_err(|e| {
                Error::Config(format!("Missing 'transformer.{i}.ln1' in {path:?}: {e}"))
            })?;
            let ln2 = layer_norm(embedding_dim, 1e-5, layer_vb.pp("ln2")).map_err(|e| {
                Error::Config(format!("Missing 'transformer.{i}.ln2' in {path:?}: {e}"))
            })?;

            transformers.push(TransformerLayer {
                attention,
                ffn,
                ln1,
                ln2,
            });
        }

        let final_layer_norm = layer_norm(embedding_dim, 1e-5, vb.pp("final_layer_norm"))
            .map_err(|e| Error::Config(format!("Missing 'final_layer_norm' in {path:?}: {e}")))?;

        let pooling_attention = if self.config.pooling_strategy == PoolingStrategy::AttentivePooling
        {
            Some(
                linear(embedding_dim, 1, vb.pp("pooling_attention")).map_err(|e| {
                    Error::Config(format!("Missing 'pooling_attention' in {path:?}: {e}"))
                })?,
            )
        } else {
            None
        };

        let model = SslModel::Pretrained {
            model_type: self.config.model_type,
            feature_projection,
            transformers,
            final_layer_norm,
        };
        let pooling = PoolingLayer {
            strategy: self.config.pooling_strategy,
            attention: pooling_attention,
        };

        Ok((model, pooling))
    }
}

/// Split `audio` into fixed-length, possibly-overlapping frames, flattened
/// row-major into a single `Vec<f32>` (length `num_frames * frame_length`).
/// The final frame is zero-padded if `audio` runs out early; very short
/// clips always yield at least one (zero-padded) frame.
fn frame_raw_audio(audio: &[f32], frame_length: usize, frame_stride: usize) -> (Vec<f32>, usize) {
    let frame_length = frame_length.max(1);
    let frame_stride = frame_stride.max(1);
    let num_frames = if audio.len() <= frame_length {
        1
    } else {
        1 + (audio.len() - frame_length).div_ceil(frame_stride)
    };

    let mut flat = Vec::with_capacity(num_frames * frame_length);
    for i in 0..num_frames {
        let start = i * frame_stride;
        for j in 0..frame_length {
            flat.push(audio.get(start + j).copied().unwrap_or(0.0));
        }
    }
    (flat, num_frames)
}

/// Hann-windowed real-FFT magnitude spectrum of a single frame.
fn hann_rfft_magnitude(frame: &[f32]) -> Result<Vec<f32>> {
    let n = frame.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let windowed: Vec<f64> = frame
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let w = if n > 1 {
                0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()
            } else {
                1.0
            };
            x as f64 * w
        })
        .collect();

    let spectrum = scirs2_fft::rfft(&windowed, None)
        .map_err(|e| Error::Processing(format!("FFT failed in spectral fallback: {e}")))?;

    Ok(spectrum
        .iter()
        .map(|c| ((c.re * c.re + c.im * c.im).sqrt()) as f32)
        .collect())
}

/// Pool a magnitude spectrum into `num_bands` contiguous frequency bands by
/// averaging, mirroring a crude (non-mel) filterbank.
fn pool_into_bands(magnitudes: &[f32], num_bands: usize) -> Vec<f32> {
    if magnitudes.is_empty() || num_bands == 0 {
        return vec![0.0; num_bands];
    }
    (0..num_bands)
        .map(|band| {
            let start = magnitudes.len() * band / num_bands;
            let end = (magnitudes.len() * (band + 1) / num_bands)
                .max(start + 1)
                .min(magnitudes.len());
            let slice = &magnitudes[start..end];
            slice.iter().sum::<f32>() / slice.len() as f32
        })
        .collect()
}

/// Z-score `features` (row-major `[num_frames, dim]`) independently per
/// feature dimension across the time (frame) axis, so absolute loudness
/// does not dominate the spectral-fallback embedding.
fn z_score_time_axis(features: &mut [f32], num_frames: usize, dim: usize) {
    if num_frames == 0 || dim == 0 {
        return;
    }
    for d in 0..dim {
        let mut mean = 0.0f32;
        for t in 0..num_frames {
            mean += features[t * dim + d];
        }
        mean /= num_frames as f32;

        let mut variance = 0.0f32;
        for t in 0..num_frames {
            let diff = features[t * dim + d] - mean;
            variance += diff * diff;
        }
        variance /= num_frames as f32;
        let std_dev = variance.sqrt().max(1e-6);

        for t in 0..num_frames {
            let idx = t * dim + d;
            features[idx] = (features[idx] - mean) / std_dev;
        }
    }
}

/// Real, non-fabricated speech-quality proxy in `[0, 1]`, derived from the
/// actual audio: penalizes hard clipping and rewards amplitude headroom
/// (mean-to-peak ratio). This is a coarse heuristic, not a MOS predictor.
fn estimate_speech_quality(audio: &[f32]) -> f32 {
    if audio.is_empty() {
        return 0.0;
    }
    let mean_abs = audio.iter().map(|x| x.abs()).sum::<f32>() / audio.len() as f32;
    let peak = audio.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    let clipping_ratio =
        audio.iter().filter(|&&x| x.abs() >= 0.999).count() as f32 / audio.len() as f32;
    let headroom = if peak > 1e-6 {
        (mean_abs / peak).clamp(0.0, 1.0)
    } else {
        0.0
    };

    ((1.0 - clipping_ratio) * (0.4 + 0.6 * headroom)).clamp(0.0, 1.0)
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

impl VerificationStats {
    fn new() -> Self {
        Self {
            total_verifications: 0,
            true_accepts: 0,
            false_accepts: 0,
            true_rejects: 0,
            false_rejects: 0,
        }
    }

    /// Compute Equal Error Rate (EER)
    pub fn compute_eer(&self) -> f32 {
        if self.total_verifications == 0 {
            return 0.0;
        }

        let far = self.false_accepts as f32 / (self.false_accepts + self.true_rejects) as f32;
        let frr = self.false_rejects as f32 / (self.false_rejects + self.true_accepts) as f32;

        (far + frr) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as StdHashMap;

    #[tokio::test]
    async fn test_ssl_verifier_creation() {
        let verifier = SslSpeakerVerifier::new();
        assert!(verifier.is_ok());
    }

    #[tokio::test]
    async fn test_ssl_config_default() {
        let config = SslVerificationConfig::default();
        assert_eq!(config.embedding_dim, 768);
        assert_eq!(config.verification_threshold, 0.80);
        assert!(config.model_weights_path.is_none());
    }

    #[test]
    fn test_pooling_strategies() {
        assert_eq!(PoolingStrategy::Mean, PoolingStrategy::Mean);
        assert_ne!(PoolingStrategy::Mean, PoolingStrategy::AttentivePooling);
    }

    #[test]
    fn test_ssl_model_types() {
        let model = SslModelType::WavLMBase;
        assert_eq!(model, SslModelType::WavLMBase);
    }

    #[test]
    fn test_verification_stats() {
        let stats = VerificationStats::new();
        assert_eq!(stats.total_verifications, 0);
        assert_eq!(stats.compute_eer(), 0.0);
    }

    fn tone(freq: f32, sample_rate: u32, seconds: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * seconds) as usize;
        (0..n)
            .map(|i| {
                (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin() * 0.5
            })
            .collect()
    }

    /// Half silence (plus a tiny noise floor), half a full-amplitude tone -
    /// gives a genuine, measurable SNR contrast to test against.
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
    async fn test_initialize_without_weights_uses_spectral_fallback() {
        let mut verifier = SslSpeakerVerifier::new().unwrap();
        verifier.initialize().await.unwrap();

        let sample = VoiceSample::new("s".to_string(), tone(440.0, 16000, 0.5), 16000);
        verifier
            .enroll_speaker("alice", std::slice::from_ref(&sample))
            .await
            .unwrap();
        let result = verifier.verify_speaker("alice", &sample).await.unwrap();
        assert_eq!(result.backend, VerificationBackend::SpectralFallback);
    }

    #[tokio::test]
    async fn test_initialize_with_missing_weights_path_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let missing_path = dir.path().join("does_not_exist.safetensors");

        let mut config = SslVerificationConfig::default();
        config.model_weights_path = Some(missing_path);
        let mut verifier = SslSpeakerVerifier::with_config(config).unwrap();

        let result = verifier.initialize().await;
        assert!(
            result.is_err(),
            "must fail closed when configured weights are missing, not silently build an \
             untrained model"
        );
    }

    #[tokio::test]
    async fn test_initialize_with_malformed_weights_file_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let bad_path = dir.path().join("not_really_safetensors.safetensors");
        std::fs::write(&bad_path, b"not a safetensors file").unwrap();

        let mut config = SslVerificationConfig::default();
        config.model_weights_path = Some(bad_path);
        let mut verifier = SslSpeakerVerifier::with_config(config).unwrap();

        let result = verifier.initialize().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_pretrained_backend_loads_real_weights_and_runs() {
        // Build a tiny-but-real checkpoint matching the documented schema,
        // then verify the pretrained backend actually loads and drives a
        // full forward pass end-to-end.
        let dir = tempfile::tempdir().unwrap();
        let checkpoint_path = dir.path().join("tiny_ssl.safetensors");

        let embedding_dim = 8usize;
        let ffn_dim = 16usize;
        let frame_length = 32usize;
        let device = Device::Cpu;

        let mut tensors: StdHashMap<String, Tensor> = StdHashMap::new();

        // NOTE: weight matrices intentionally vary per-element (not a
        // constant fill). A constant-fill weight matrix makes every output
        // *feature* identical across the embedding dimension, which
        // LayerNorm's mean-subtraction then collapses to exactly zero for
        // every element - a degenerate test fixture, not a real checkpoint
        // (real trained weights are never literally constant). `insert`
        // varies each element by its flat index so the resulting embedding
        // is non-degenerate, matching what an actual checkpoint looks like.
        fn insert(
            tensors: &mut StdHashMap<String, Tensor>,
            device: &Device,
            name: &str,
            shape: Vec<usize>,
            scale: f32,
        ) {
            let len: usize = shape.iter().product();
            let data: Vec<f32> = (0..len).map(|i| scale * (1.0 + (i as f32) * 0.1)).collect();
            tensors.insert(
                name.to_string(),
                Tensor::from_vec(data, shape, device).unwrap(),
            );
        }

        insert(
            &mut tensors,
            &device,
            "feature_projection.weight",
            vec![embedding_dim, frame_length],
            0.01,
        );
        insert(
            &mut tensors,
            &device,
            "feature_projection.bias",
            vec![embedding_dim],
            0.0,
        );
        for i in 0..1 {
            let prefix = format!("transformer.{i}");
            for sub in ["query", "key", "value", "output"] {
                insert(
                    &mut tensors,
                    &device,
                    &format!("{prefix}.attention.{sub}.weight"),
                    vec![embedding_dim, embedding_dim],
                    0.02,
                );
                insert(
                    &mut tensors,
                    &device,
                    &format!("{prefix}.attention.{sub}.bias"),
                    vec![embedding_dim],
                    0.0,
                );
            }
            insert(
                &mut tensors,
                &device,
                &format!("{prefix}.ffn.fc1.weight"),
                vec![ffn_dim, embedding_dim],
                0.03,
            );
            insert(
                &mut tensors,
                &device,
                &format!("{prefix}.ffn.fc1.bias"),
                vec![ffn_dim],
                0.0,
            );
            insert(
                &mut tensors,
                &device,
                &format!("{prefix}.ffn.fc2.weight"),
                vec![embedding_dim, ffn_dim],
                0.03,
            );
            insert(
                &mut tensors,
                &device,
                &format!("{prefix}.ffn.fc2.bias"),
                vec![embedding_dim],
                0.0,
            );
            insert(
                &mut tensors,
                &device,
                &format!("{prefix}.ln1.weight"),
                vec![embedding_dim],
                1.0,
            );
            insert(
                &mut tensors,
                &device,
                &format!("{prefix}.ln1.bias"),
                vec![embedding_dim],
                0.0,
            );
            insert(
                &mut tensors,
                &device,
                &format!("{prefix}.ln2.weight"),
                vec![embedding_dim],
                1.0,
            );
            insert(
                &mut tensors,
                &device,
                &format!("{prefix}.ln2.bias"),
                vec![embedding_dim],
                0.0,
            );
        }
        insert(
            &mut tensors,
            &device,
            "final_layer_norm.weight",
            vec![embedding_dim],
            1.0,
        );
        insert(
            &mut tensors,
            &device,
            "final_layer_norm.bias",
            vec![embedding_dim],
            0.0,
        );

        candle_core::safetensors::save(&tensors, &checkpoint_path).unwrap();

        let config = SslVerificationConfig {
            embedding_dim,
            num_transformer_layers: 1,
            num_attention_heads: 2,
            ffn_dim,
            frame_length,
            frame_stride: 16,
            pooling_strategy: PoolingStrategy::Mean,
            model_weights_path: Some(checkpoint_path),
            use_gpu: false,
            ..SslVerificationConfig::default()
        };

        let mut verifier = SslSpeakerVerifier::with_config(config).unwrap();
        verifier
            .initialize()
            .await
            .expect("a checkpoint matching the documented schema must load successfully");

        let sample = VoiceSample::new("s".to_string(), tone(220.0, 16000, 0.2), 16000);
        verifier
            .enroll_speaker("dave", std::slice::from_ref(&sample))
            .await
            .unwrap();
        let result = verifier.verify_speaker("dave", &sample).await.unwrap();

        assert_eq!(result.backend, VerificationBackend::PretrainedSsl);
        assert!(result.similarity_score.is_finite());
        // An identical enrollment/test sample through a real (even tiny,
        // untrained) deterministic transformer should be highly self-similar.
        assert!(
            result.similarity_score > 0.9,
            "identical input should be near-perfectly self-similar, got {}",
            result.similarity_score
        );
    }

    #[tokio::test]
    async fn test_spectral_fallback_similarity_reflects_audio_content() {
        let mut verifier = SslSpeakerVerifier::new().unwrap();
        verifier.initialize().await.unwrap();

        let enrolled_audio = tone(220.0, 16000, 0.5);
        let same_audio = tone(220.0, 16000, 0.5);
        // A high-frequency square-ish wave: very different spectral content.
        let different_audio: Vec<f32> = (0..(16000usize / 2))
            .map(|i| if i % 3 == 0 { 0.9 } else { -0.9 })
            .collect();

        let enrolled_sample = VoiceSample::new("enroll".to_string(), enrolled_audio, 16000);
        verifier
            .enroll_speaker("bob", std::slice::from_ref(&enrolled_sample))
            .await
            .unwrap();

        let same_sample = VoiceSample::new("same".to_string(), same_audio, 16000);
        let same_result = verifier.verify_speaker("bob", &same_sample).await.unwrap();

        let diff_sample = VoiceSample::new("diff".to_string(), different_audio, 16000);
        let diff_result = verifier.verify_speaker("bob", &diff_sample).await.unwrap();

        assert!(
            same_result.similarity_score > diff_result.similarity_score,
            "similarity must reflect real audio content, not a constant identity transform \
             (same={}, diff={})",
            same_result.similarity_score,
            diff_result.similarity_score
        );
    }

    #[tokio::test]
    async fn test_quality_metrics_vary_with_audio_not_hardcoded() {
        let mut verifier = SslSpeakerVerifier::new().unwrap();
        verifier.initialize().await.unwrap();

        let clean_audio = tone_with_noise_floor(220.0, 16000, 1.0, 0.001);
        let noisy_audio = tone_with_noise_floor(220.0, 16000, 1.0, 0.35);

        let clean_sample = VoiceSample::new("clean".to_string(), clean_audio, 16000);
        verifier
            .enroll_speaker("carol", std::slice::from_ref(&clean_sample))
            .await
            .unwrap();
        let clean_result = verifier
            .verify_speaker("carol", &clean_sample)
            .await
            .unwrap();

        let noisy_sample = VoiceSample::new("noisy".to_string(), noisy_audio, 16000);
        let noisy_result = verifier
            .verify_speaker("carol", &noisy_sample)
            .await
            .unwrap();

        assert!(
            clean_result.quality.snr_estimate > noisy_result.quality.snr_estimate,
            "snr_estimate must be computed from the actual audio, not a fixed placeholder \
             (clean={}, noisy={})",
            clean_result.quality.snr_estimate,
            noisy_result.quality.snr_estimate
        );
        assert_ne!(
            clean_result.quality.speech_quality, 0.85,
            "must not be the old hardcoded placeholder value"
        );
        assert_ne!(
            clean_result.quality.snr_estimate, 20.0,
            "must not be the old hardcoded placeholder value"
        );
    }
}
