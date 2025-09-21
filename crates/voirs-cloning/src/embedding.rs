//! Speaker embedding extraction and management

use crate::{types::VoiceSample, Error, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{conv2d, linear, Conv2d, Linear, Module, VarBuilder};
use ndarray::Array2;
use realfft::RealFftPlanner;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, trace};

/// Speaker embedding vector with metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerEmbedding {
    /// Embedding vector
    pub vector: Vec<f32>,
    /// Dimension of the embedding
    pub dimension: usize,
    /// Confidence score of the embedding (0.0-1.0)
    pub confidence: f32,
    /// Speaker characteristics metadata
    pub metadata: EmbeddingMetadata,
}

/// Metadata associated with speaker embedding
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingMetadata {
    /// Gender classification (if available)
    pub gender: Option<String>,
    /// Age estimation (if available)
    pub age_estimate: Option<f32>,
    /// Language/accent information
    pub language: Option<String>,
    /// Emotional characteristics
    pub emotion: Option<String>,
    /// Voice quality indicators
    pub voice_quality: VoiceQuality,
    /// Extraction timestamp
    pub extraction_time: Option<f64>,
}

/// Voice quality characteristics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceQuality {
    /// Fundamental frequency statistics
    pub f0_mean: f32,
    pub f0_std: f32,
    /// Spectral characteristics
    pub spectral_centroid: f32,
    pub spectral_bandwidth: f32,
    /// Voice quality metrics
    pub jitter: f32,
    pub shimmer: f32,
    /// Energy characteristics
    pub energy_mean: f32,
    pub energy_std: f32,
}

/// Advanced speaker embedding extractor with neural networks (thread-safe)
pub struct SpeakerEmbeddingExtractor {
    /// Model configuration (immutable, can be shared)
    config: Arc<EmbeddingConfig>,
    /// Neural network device
    device: Device,
    /// Embedding network (thread-safe)
    embedding_network: Arc<RwLock<Option<EmbeddingNetwork>>>,
    /// Feature extractor (thread-safe)
    feature_extractor: Arc<RwLock<FeatureExtractor>>,
    /// Normalization statistics (thread-safe)
    normalization_stats: Arc<RwLock<Option<NormalizationStats>>>,
}

impl std::fmt::Debug for SpeakerEmbeddingExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeakerEmbeddingExtractor")
            .field("config", &self.config)
            .field("device", &"<Device>")
            .field(
                "embedding_network",
                &"<Arc<RwLock<Option<EmbeddingNetwork>>>>",
            )
            .field("feature_extractor", &"<Arc<RwLock<FeatureExtractor>>>")
            .field(
                "normalization_stats",
                &"<Arc<RwLock<Option<NormalizationStats>>>>",
            )
            .finish()
    }
}

/// Configuration for embedding extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding dimension
    pub dimension: usize,
    /// Window size for processing (samples)
    pub window_size: usize,
    /// Hop size for overlapping windows
    pub hop_size: usize,
    /// FFT size for spectral analysis
    pub fft_size: usize,
    /// Number of mel filters
    pub num_mel_filters: usize,
    /// Neural network architecture
    pub network_architecture: NetworkArchitecture,
    /// Feature extraction method
    pub feature_method: FeatureExtractionMethod,
    /// Preprocessing options
    pub preprocessing: PreprocessingConfig,
    /// Batch processing size
    pub batch_size: usize,
}

/// Neural network architecture options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkArchitecture {
    /// ResNet-based architecture
    ResNet { layers: Vec<usize> },
    /// Transformer-based architecture
    Transformer {
        num_layers: usize,
        num_heads: usize,
        hidden_dim: usize,
    },
    /// CNN-based architecture
    CNN {
        conv_layers: Vec<(usize, usize)>, // (channels, kernel_size)
        fc_layers: Vec<usize>,
    },
    /// TDNN (Time Delay Neural Network)
    TDNN { layers: Vec<TDNNLayer> },
}

/// TDNN layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TDNNLayer {
    pub input_dim: usize,
    pub output_dim: usize,
    pub context: Vec<i32>, // Time context offsets
}

/// Feature extraction methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureExtractionMethod {
    /// Mel-frequency cepstral coefficients
    MFCC,
    /// Mel-scale spectrogram
    MelSpectrogram,
    /// Raw spectrogram
    Spectrogram,
    /// Log mel-scale spectrogram
    LogMel,
    /// Perceptual Linear Prediction coefficients
    PLP,
    /// Filter bank features
    FilterBank,
}

/// Preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Apply voice activity detection
    pub vad_enabled: bool,
    /// Noise reduction enabled
    pub noise_reduction: bool,
    /// Normalization method
    pub normalization: NormalizationMethod,
    /// Augmentation options
    pub augmentation: Option<AugmentationConfig>,
}

/// Normalization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// Z-score normalization
    ZScore,
    /// Min-max normalization
    MinMax,
    /// Cepstral mean normalization
    CMN,
    /// Cepstral mean and variance normalization
    CMVN,
}

/// Data augmentation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationConfig {
    /// Add noise (SNR range)
    pub noise_snr_range: Option<(f32, f32)>,
    /// Speed perturbation range
    pub speed_range: Option<(f32, f32)>,
    /// Volume perturbation range
    pub volume_range: Option<(f32, f32)>,
    /// Spectral augmentation
    pub spec_augment: bool,
}

/// Normalization statistics
#[derive(Debug, Clone)]
pub struct NormalizationStats {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub min: Vec<f32>,
    pub max: Vec<f32>,
}

/// Online learning configuration for adaptive embedding updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineLearningConfig {
    /// Initial learning rate for adaptation
    pub initial_learning_rate: f32,
    /// Learning rate decay factor per step
    pub decay_factor: f32,
    /// Minimum learning rate
    pub min_learning_rate: f32,
    /// Convergence threshold for similarity
    pub convergence_threshold: f32,
    /// Maximum number of adaptation steps
    pub max_steps: usize,
}

/// Configuration for streaming embedding extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Duration of each processing window (seconds)
    pub window_duration: f32,
    /// Hop duration between windows (seconds)
    pub hop_duration: f32,
    /// Enable temporal smoothing across windows
    pub temporal_smoothing: bool,
    /// Smoothing factor for temporal averaging
    pub smoothing_factor: f32,
    /// Method for aggregating window embeddings
    pub aggregation_method: AggregationMethod,
    /// Minimum confidence threshold for window inclusion
    pub min_confidence_threshold: f32,
}

/// Configuration for adaptive refinement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementConfig {
    /// Base refinement weight
    pub base_refinement_weight: f32,
    /// Quality amplification factor
    pub quality_amplification: f32,
    /// Maximum refinement weight
    pub max_refinement_weight: f32,
    /// Convergence threshold for refinement
    pub convergence_threshold: f32,
    /// Window size for convergence checking
    pub convergence_window: usize,
}

/// Aggregation methods for streaming embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Simple averaging
    Average,
    /// Confidence-weighted averaging
    Weighted,
    /// Quality-weighted averaging
    QualityWeighted,
    /// Temporal-weighted averaging (recent samples have higher weight)
    TemporalWeighted,
}

/// Metrics for adaptation steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationMetrics {
    pub step: usize,
    pub learning_rate: f32,
    pub similarity_to_base: f32,
    pub similarity_to_previous: f32,
    pub confidence_change: f32,
    pub adaptation_time: std::time::Duration,
    pub quality_score: f32,
}

/// Result of streaming embedding extraction
#[derive(Debug, Clone)]
pub struct StreamingEmbeddingResult {
    /// Aggregated embedding from all windows
    pub aggregated_embedding: SpeakerEmbedding,
    /// Individual window embeddings
    pub window_embeddings: Vec<SpeakerEmbedding>,
    /// Streaming statistics
    pub streaming_stats: StreamingStats,
}

/// Statistics for streaming processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingStats {
    pub num_windows: usize,
    pub avg_confidence: f32,
    pub confidence_std: f32,
    pub total_duration: f32,
    pub processing_time: std::time::Duration,
}

/// Result of adaptive refinement
#[derive(Debug, Clone)]
pub struct RefinementResult {
    /// Final refined embedding
    pub refined_embedding: SpeakerEmbedding,
    /// History of refinement iterations
    pub refinement_history: Vec<RefinementIteration>,
    /// Total similarity improvement
    pub total_similarity_improvement: f32,
    /// Total confidence change
    pub total_confidence_change: f32,
    /// Whether convergence was achieved
    pub convergence_achieved: bool,
}

/// Metrics for individual refinement iterations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementIteration {
    pub iteration: usize,
    pub sample_index: usize,
    pub quality_feedback: f32,
    pub adaptive_weight: f32,
    pub similarity_before: f32,
    pub similarity_after: f32,
    pub confidence_change: f32,
    pub iteration_time: std::time::Duration,
}

/// Embedding stability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingStability {
    /// Consistency of embeddings over time
    pub temporal_consistency: f32,
    /// Stability of confidence scores
    pub confidence_stability: f32,
    /// Rate of embedding drift
    pub drift_rate: f32,
    /// Overall stability score
    pub stability_score: f32,
}

/// Speaker identification match result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerMatch {
    /// Speaker identifier
    pub speaker_id: String,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f32,
    /// Confidence of the match
    pub confidence: f32,
    /// Distance metric (lower is better)
    pub distance: f32,
}

/// Feature extractor for audio preprocessing
pub struct FeatureExtractor {
    config: EmbeddingConfig,
    fft_planner: RealFftPlanner<f32>,
    mel_filterbank: Option<Array2<f32>>,
    dct_matrix: Option<Array2<f32>>,
}

/// Neural network for embedding extraction
#[derive(Debug)]
struct EmbeddingNetwork {
    layers: Vec<Linear>,
    conv_layers: Vec<Conv2d>,
    device: Device,
    architecture: NetworkArchitecture,
}

impl SpeakerEmbedding {
    /// Create new embedding
    pub fn new(vector: Vec<f32>) -> Self {
        let dimension = vector.len();
        Self {
            vector,
            dimension,
            confidence: 1.0,
            metadata: EmbeddingMetadata::default(),
        }
    }

    /// Create embedding with metadata
    pub fn with_metadata(vector: Vec<f32>, metadata: EmbeddingMetadata) -> Self {
        let dimension = vector.len();
        Self {
            vector,
            dimension,
            confidence: 1.0,
            metadata,
        }
    }

    /// Calculate cosine similarity to another embedding
    pub fn similarity(&self, other: &Self) -> f32 {
        if self.dimension != other.dimension {
            return 0.0;
        }

        let dot_product: f32 = self
            .vector
            .iter()
            .zip(&other.vector)
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Calculate Euclidean distance to another embedding
    pub fn distance(&self, other: &Self) -> f32 {
        if self.dimension != other.dimension {
            return f32::INFINITY;
        }

        let squared_diff: f32 = self
            .vector
            .iter()
            .zip(&other.vector)
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        squared_diff.sqrt()
    }

    /// L2 normalize the embedding
    pub fn normalize(&mut self) {
        let norm: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut self.vector {
                *value /= norm;
            }
        }
    }

    /// Check if embedding is valid
    pub fn is_valid(&self) -> bool {
        !self.vector.is_empty()
            && self.vector.iter().all(|x| x.is_finite())
            && self.confidence >= 0.0
            && self.confidence <= 1.0
    }

    /// Get embedding quality score based on confidence and voice quality
    pub fn quality_score(&self) -> f32 {
        let voice_quality_score = self.metadata.voice_quality.overall_quality();
        (self.confidence + voice_quality_score) / 2.0
    }
}

impl SpeakerEmbeddingExtractor {
    /// Create new extractor (thread-safe)
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        let device = Device::Cpu; // Could be GPU
        let feature_extractor = FeatureExtractor::new(config.clone())?;

        Ok(Self {
            config: Arc::new(config),
            device,
            embedding_network: Arc::new(RwLock::new(None)),
            feature_extractor: Arc::new(RwLock::new(feature_extractor)),
            normalization_stats: Arc::new(RwLock::new(None)),
        })
    }

    /// Create new extractor with GPU support (thread-safe)
    pub fn with_device(config: EmbeddingConfig, device: Device) -> Result<Self> {
        let feature_extractor = FeatureExtractor::new(config.clone())?;

        Ok(Self {
            config: Arc::new(config),
            device,
            embedding_network: Arc::new(RwLock::new(None)),
            feature_extractor: Arc::new(RwLock::new(feature_extractor)),
            normalization_stats: Arc::new(RwLock::new(None)),
        })
    }

    /// Initialize neural network (thread-safe)
    pub async fn initialize_network(&self) -> Result<()> {
        let varmap = candle_nn::VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &self.device);

        let network = self.create_embedding_network(vs)?;

        let mut network_lock = self.embedding_network.write().await;
        *network_lock = Some(network);

        Ok(())
    }

    /// Extract embedding from voice sample (thread-safe)
    pub async fn extract(&self, sample: &VoiceSample) -> Result<SpeakerEmbedding> {
        // Preprocess audio
        let processed_audio = self.preprocess_audio(sample).await?;

        // Extract features (thread-safe)
        let features = {
            let mut feature_extractor = self.feature_extractor.write().await;
            feature_extractor.extract_features(&processed_audio, sample.sample_rate)?
        };

        // Apply normalization if available (thread-safe)
        let normalized_features = {
            let stats_lock = self.normalization_stats.read().await;
            if let Some(stats) = stats_lock.as_ref() {
                self.normalize_features(&features, stats)?
            } else {
                features
            }
        };

        // Extract embedding using neural network (thread-safe)
        let embedding_vector = {
            let network_lock = self.embedding_network.read().await;
            if let Some(network) = network_lock.as_ref() {
                self.extract_with_network(&normalized_features, network)
                    .await?
            } else {
                // Fallback to classical methods
                self.extract_classical(&normalized_features)?
            }
        };

        // Compute voice quality metrics
        let voice_quality = self.compute_voice_quality(&processed_audio, sample.sample_rate)?;

        // Compute confidence based on signal quality and consistency
        let confidence = self.compute_confidence(&processed_audio, &embedding_vector)?;

        // Create metadata
        let metadata = EmbeddingMetadata {
            gender: None,       // Would require additional model
            age_estimate: None, // Would require additional model
            language: None,
            emotion: None,
            voice_quality,
            extraction_time: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64(),
            ),
        };

        let mut embedding = SpeakerEmbedding {
            vector: embedding_vector,
            dimension: self.config.dimension,
            confidence,
            metadata,
        };

        // Normalize embedding
        embedding.normalize();

        Ok(embedding)
    }

    /// Extract embeddings from multiple samples (batch processing, thread-safe)
    pub async fn extract_batch(&self, samples: &[VoiceSample]) -> Result<Vec<SpeakerEmbedding>> {
        let mut embeddings = Vec::new();
        let batch_size = self.config.batch_size;

        for chunk in samples.chunks(batch_size) {
            for sample in chunk {
                let embedding = self.extract(sample).await?;
                embeddings.push(embedding);
            }
            // Yield control point for better concurrency
            tokio::task::yield_now().await;
        }

        Ok(embeddings)
    }

    /// Average multiple embeddings (thread-safe)
    pub fn average_embeddings(&self, embeddings: &[SpeakerEmbedding]) -> Result<SpeakerEmbedding> {
        if embeddings.is_empty() {
            return Err(Error::Processing("No embeddings to average".to_string()));
        }

        let dimension = embeddings[0].dimension;
        let mut averaged_vector = vec![0.0; dimension];
        let mut total_confidence = 0.0;

        // Weighted average by confidence
        for embedding in embeddings {
            if embedding.dimension != dimension {
                return Err(Error::Processing(
                    "Inconsistent embedding dimensions".to_string(),
                ));
            }

            for (i, &value) in embedding.vector.iter().enumerate() {
                averaged_vector[i] += value * embedding.confidence;
            }
            total_confidence += embedding.confidence;
        }

        if total_confidence > 0.0 {
            for value in &mut averaged_vector {
                *value /= total_confidence;
            }
        }

        let averaged_confidence = total_confidence / embeddings.len() as f32;

        // Average voice quality metrics
        let averaged_voice_quality = self.average_voice_quality(embeddings)?;

        let metadata = EmbeddingMetadata {
            gender: None,
            age_estimate: None,
            language: None,
            emotion: None,
            voice_quality: averaged_voice_quality,
            extraction_time: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64(),
            ),
        };

        Ok(SpeakerEmbedding {
            vector: averaged_vector,
            dimension,
            confidence: averaged_confidence,
            metadata,
        })
    }

    /// Real-time embedding adaptation with incremental updates (thread-safe)
    pub async fn adapt_embedding_realtime(
        &self,
        base_embedding: &SpeakerEmbedding,
        adaptation_sample: &VoiceSample,
        adaptation_rate: f32,
    ) -> Result<SpeakerEmbedding> {
        trace!("Starting real-time embedding adaptation");

        // Extract embedding from adaptation sample
        let sample_embedding = self.extract(adaptation_sample).await?;

        // Compute adaptive learning rate based on sample quality
        let quality_score = sample_embedding.metadata.voice_quality.overall_quality();
        let adaptive_rate = adaptation_rate * quality_score;

        // Exponential moving average update
        let mut adapted_vector = base_embedding.vector.clone();
        for (i, (&base_val, &sample_val)) in base_embedding
            .vector
            .iter()
            .zip(&sample_embedding.vector)
            .enumerate()
        {
            adapted_vector[i] = base_val * (1.0 - adaptive_rate) + sample_val * adaptive_rate;
        }

        // Update confidence based on consistency
        let consistency = base_embedding.similarity(&sample_embedding);
        let updated_confidence =
            (base_embedding.confidence + sample_embedding.confidence * consistency) / 2.0;

        let mut adapted_embedding = SpeakerEmbedding {
            vector: adapted_vector,
            dimension: base_embedding.dimension,
            confidence: updated_confidence,
            metadata: sample_embedding.metadata.clone(),
        };

        // Normalize the adapted embedding
        adapted_embedding.normalize();

        debug!(
            "Real-time adaptation completed: consistency {:.3}, confidence {:.3}",
            consistency, updated_confidence
        );

        Ok(adapted_embedding)
    }

    /// Online learning with multiple samples for continuous adaptation (thread-safe)
    pub async fn online_learning_adaptation(
        &self,
        base_embedding: &mut SpeakerEmbedding,
        samples: &[VoiceSample],
        learning_config: &OnlineLearningConfig,
    ) -> Result<Vec<AdaptationMetrics>> {
        info!(
            "Starting online learning adaptation with {} samples",
            samples.len()
        );

        let mut adaptation_metrics = Vec::new();
        let mut current_embedding = base_embedding.clone();

        for (step, sample) in samples.iter().enumerate() {
            let step_start = std::time::Instant::now();

            // Adaptive learning rate with decay
            let step_rate = learning_config.initial_learning_rate
                * (learning_config.decay_factor.powf(step as f32));

            // Apply adaptation
            let adapted = self
                .adapt_embedding_realtime(&current_embedding, sample, step_rate)
                .await?;

            // Compute metrics for this step
            let similarity_to_base = base_embedding.similarity(&adapted);
            let similarity_to_previous = current_embedding.similarity(&adapted);

            let metrics = AdaptationMetrics {
                step,
                learning_rate: step_rate,
                similarity_to_base,
                similarity_to_previous,
                confidence_change: adapted.confidence - current_embedding.confidence,
                adaptation_time: step_start.elapsed(),
                quality_score: adapted.metadata.voice_quality.overall_quality(),
            };

            adaptation_metrics.push(metrics);
            current_embedding = adapted;

            // Early stopping if convergence is reached
            if similarity_to_previous > learning_config.convergence_threshold {
                info!(
                    "Convergence reached at step {} with similarity {:.3}",
                    step, similarity_to_previous
                );
                break;
            }
        }

        // Update the base embedding
        *base_embedding = current_embedding;

        info!(
            "Online learning completed after {} steps",
            adaptation_metrics.len()
        );
        Ok(adaptation_metrics)
    }

    /// Streaming embedding extraction with buffering for real-time applications (thread-safe)
    pub async fn extract_streaming(
        &self,
        audio_stream: &[f32],
        sample_rate: u32,
        streaming_config: &StreamingConfig,
    ) -> Result<StreamingEmbeddingResult> {
        trace!("Starting streaming embedding extraction");

        let window_size = (sample_rate as f32 * streaming_config.window_duration) as usize;
        let hop_size = (sample_rate as f32 * streaming_config.hop_duration) as usize;

        if audio_stream.len() < window_size {
            return Err(Error::Processing(
                "Insufficient audio data for streaming".to_string(),
            ));
        }

        let mut embeddings = Vec::new();
        let mut confidences = Vec::new();

        // Process overlapping windows
        for (window_idx, start) in (0..audio_stream.len()).step_by(hop_size).enumerate() {
            let end = (start + window_size).min(audio_stream.len());
            if end - start < window_size / 2 {
                break; // Skip incomplete windows
            }

            let window_audio = &audio_stream[start..end];
            let sample_id = format!("stream_window_{}", window_idx);
            let window_sample = VoiceSample::new(sample_id, window_audio.to_vec(), sample_rate);

            // Extract embedding for this window
            let window_embedding = self.extract(&window_sample).await?;
            confidences.push(window_embedding.confidence);
            embeddings.push(window_embedding);
        }

        if embeddings.is_empty() {
            return Err(Error::Processing("No valid windows extracted".to_string()));
        }

        // Compute streaming statistics
        let avg_confidence = confidences.iter().sum::<f32>() / confidences.len() as f32;
        let confidence_std = {
            let variance = confidences
                .iter()
                .map(|c| (c - avg_confidence).powi(2))
                .sum::<f32>()
                / confidences.len() as f32;
            variance.sqrt()
        };

        // Apply temporal smoothing if enabled
        let final_embeddings = if streaming_config.temporal_smoothing {
            self.apply_temporal_smoothing(&embeddings, streaming_config.smoothing_factor)?
        } else {
            embeddings.clone()
        };

        // Compute aggregated embedding
        let aggregated_embedding =
            if streaming_config.aggregation_method == AggregationMethod::Weighted {
                self.weighted_aggregation(&final_embeddings)?
            } else {
                self.average_embeddings(&final_embeddings)?
            };

        Ok(StreamingEmbeddingResult {
            aggregated_embedding,
            window_embeddings: final_embeddings,
            streaming_stats: StreamingStats {
                num_windows: embeddings.len(),
                avg_confidence,
                confidence_std,
                total_duration: audio_stream.len() as f32 / sample_rate as f32,
                processing_time: std::time::Duration::from_secs(0), // Would be measured in practice
            },
        })
    }

    /// Incremental embedding update using exponential moving average
    pub fn update_embedding_incremental(
        &self,
        current_embedding: &mut SpeakerEmbedding,
        new_embedding: &SpeakerEmbedding,
        update_weight: f32,
    ) -> Result<f32> {
        if current_embedding.dimension != new_embedding.dimension {
            return Err(Error::Processing(
                "Embedding dimension mismatch".to_string(),
            ));
        }

        let similarity_before = current_embedding.similarity(new_embedding);

        // Update vector using exponential moving average
        for (current_val, &new_val) in current_embedding
            .vector
            .iter_mut()
            .zip(&new_embedding.vector)
        {
            *current_val = *current_val * (1.0 - update_weight) + new_val * update_weight;
        }

        // Update confidence
        current_embedding.confidence = current_embedding.confidence * (1.0 - update_weight)
            + new_embedding.confidence * update_weight;

        // Update metadata with latest extraction time
        current_embedding.metadata.extraction_time = new_embedding.metadata.extraction_time;

        // Re-normalize
        current_embedding.normalize();

        Ok(similarity_before)
    }

    /// Adaptive embedding refinement based on quality feedback (thread-safe)
    pub async fn refine_embedding_adaptive(
        &self,
        base_embedding: &SpeakerEmbedding,
        refinement_samples: &[(VoiceSample, f32)], // (sample, quality_feedback)
        refinement_config: &RefinementConfig,
    ) -> Result<RefinementResult> {
        info!(
            "Starting adaptive embedding refinement with {} samples",
            refinement_samples.len()
        );

        let mut refined_embedding = base_embedding.clone();
        let mut refinement_history = Vec::new();

        // Sort samples by quality feedback (highest first)
        let mut sorted_samples: Vec<_> = refinement_samples.iter().enumerate().collect();
        sorted_samples.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());

        for (iteration, (idx, (sample, quality_feedback))) in sorted_samples.iter().enumerate() {
            let iteration_start = std::time::Instant::now();

            // Extract embedding for refinement sample
            let sample_embedding = self.extract(sample).await?;

            // Compute adaptive weight based on quality feedback
            let base_weight = refinement_config.base_refinement_weight;
            let quality_weight = *quality_feedback * refinement_config.quality_amplification;
            let adaptive_weight =
                (base_weight * quality_weight).clamp(0.0, refinement_config.max_refinement_weight);

            // Apply refinement
            let similarity = self.update_embedding_incremental(
                &mut refined_embedding,
                &sample_embedding,
                adaptive_weight,
            )?;

            let iteration_metrics = RefinementIteration {
                iteration,
                sample_index: *idx,
                quality_feedback: *quality_feedback,
                adaptive_weight,
                similarity_before: similarity,
                similarity_after: refined_embedding.similarity(&sample_embedding),
                confidence_change: refined_embedding.confidence - base_embedding.confidence,
                iteration_time: iteration_start.elapsed(),
            };

            refinement_history.push(iteration_metrics);

            // Early stopping based on convergence
            if iteration > 0 {
                let recent_improvements: Vec<f32> = refinement_history
                    .iter()
                    .rev()
                    .take(refinement_config.convergence_window)
                    .map(|r| r.similarity_after - r.similarity_before)
                    .collect();

                if recent_improvements.len() == refinement_config.convergence_window {
                    let avg_improvement =
                        recent_improvements.iter().sum::<f32>() / recent_improvements.len() as f32;
                    if avg_improvement.abs() < refinement_config.convergence_threshold {
                        info!(
                            "Refinement converged at iteration {} with improvement {:.6}",
                            iteration, avg_improvement
                        );
                        break;
                    }
                }
            }
        }

        let total_similarity_improvement = refined_embedding.similarity(base_embedding);
        let total_confidence_change = refined_embedding.confidence - base_embedding.confidence;
        let refinement_history_len = refinement_history.len();

        Ok(RefinementResult {
            refined_embedding,
            refinement_history,
            total_similarity_improvement,
            total_confidence_change,
            convergence_achieved: refinement_history_len < refinement_samples.len(),
        })
    }

    /// Preprocess audio sample
    async fn preprocess_audio(&self, sample: &VoiceSample) -> Result<Vec<f32>> {
        let mut audio = sample.get_normalized_audio();

        // Apply voice activity detection if enabled
        if self.config.preprocessing.vad_enabled {
            audio = self.apply_vad(&audio, sample.sample_rate)?;
        }

        // Apply noise reduction if enabled
        if self.config.preprocessing.noise_reduction {
            audio = self.apply_noise_reduction(&audio)?;
        }

        // Apply augmentation if configured
        if let Some(aug_config) = &self.config.preprocessing.augmentation {
            audio = self.apply_augmentation(&audio, aug_config, sample.sample_rate)?;
        }

        Ok(audio)
    }

    /// Apply voice activity detection
    fn apply_vad(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        // Simplified VAD based on energy thresholding
        let frame_size = (sample_rate as f32 * 0.025) as usize; // 25ms frames
        let hop_size = (sample_rate as f32 * 0.010) as usize; // 10ms hop

        let mut active_frames = Vec::new();
        let energy_threshold = self.compute_energy_threshold(audio);

        for chunk in audio.chunks(frame_size) {
            let energy = chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32;
            if energy > energy_threshold {
                active_frames.extend_from_slice(chunk);
            }
        }

        if active_frames.is_empty() {
            Ok(audio.to_vec()) // Return original if no speech detected
        } else {
            Ok(active_frames)
        }
    }

    /// Compute energy threshold for VAD
    fn compute_energy_threshold(&self, audio: &[f32]) -> f32 {
        let frame_size = 512;
        let mut energies = Vec::new();

        for chunk in audio.chunks(frame_size) {
            let energy = chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32;
            energies.push(energy);
        }

        if energies.is_empty() {
            return 0.0;
        }

        energies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let percentile_50 = energies[energies.len() / 2];
        percentile_50 * 2.0 // Simple threshold
    }

    /// Apply noise reduction
    fn apply_noise_reduction(&self, audio: &[f32]) -> Result<Vec<f32>> {
        // Simplified spectral subtraction
        Ok(audio.to_vec()) // Placeholder
    }

    /// Apply data augmentation
    fn apply_augmentation(
        &self,
        audio: &[f32],
        config: &AugmentationConfig,
        sample_rate: u32,
    ) -> Result<Vec<f32>> {
        let mut augmented = audio.to_vec();

        // Speed perturbation
        if let Some((min_speed, max_speed)) = config.speed_range {
            let speed_factor = min_speed + rand::random::<f32>() * (max_speed - min_speed);
            augmented = self.change_speed(&augmented, speed_factor)?;
        }

        // Volume perturbation
        if let Some((min_vol, max_vol)) = config.volume_range {
            let volume_factor = min_vol + rand::random::<f32>() * (max_vol - min_vol);
            for sample in &mut augmented {
                *sample *= volume_factor;
            }
        }

        Ok(augmented)
    }

    /// Change audio speed
    fn change_speed(&self, audio: &[f32], factor: f32) -> Result<Vec<f32>> {
        if factor <= 0.0 {
            return Err(Error::Processing(
                "Speed factor must be positive".to_string(),
            ));
        }

        let new_length = (audio.len() as f32 / factor) as usize;
        let mut resampled = Vec::with_capacity(new_length);

        for i in 0..new_length {
            let src_index = (i as f32 * factor) as usize;
            if src_index < audio.len() {
                resampled.push(audio[src_index]);
            } else {
                resampled.push(0.0);
            }
        }

        Ok(resampled)
    }

    /// Extract embedding using neural network
    async fn extract_with_network(
        &self,
        features: &Array2<f32>,
        network: &EmbeddingNetwork,
    ) -> Result<Vec<f32>> {
        // Convert features to tensor
        let tensor = self.features_to_tensor(features)?;

        // Forward pass through network
        let output = network.forward(&tensor)?;

        // Convert output tensor to vector
        let embedding = output
            .to_vec1::<f32>()
            .map_err(|e| Error::Processing(format!("Failed to convert tensor to vector: {}", e)))?;

        Ok(embedding)
    }

    /// Extract embedding using classical methods
    fn extract_classical(&self, features: &Array2<f32>) -> Result<Vec<f32>> {
        // Simple statistical aggregation of features
        let (time_frames, feature_dim) = features.dim();
        let mut embedding = vec![0.0; self.config.dimension.min(feature_dim * 4)];

        // Mean pooling
        for i in 0..feature_dim {
            let column_mean = features.column(i).mean().unwrap_or(0.0);
            if i < embedding.len() {
                embedding[i] = column_mean;
            }
        }

        // Standard deviation
        for i in 0..feature_dim {
            let column_std = features.column(i).std(0.0);
            let idx = feature_dim + i;
            if idx < embedding.len() {
                embedding[idx] = column_std;
            }
        }

        // Min and max values
        for i in 0..feature_dim {
            let column = features.column(i);
            if let (Some(&min_val), Some(&max_val)) = (
                column.iter().min_by(|a, b| a.partial_cmp(b).unwrap()),
                column.iter().max_by(|a, b| a.partial_cmp(b).unwrap()),
            ) {
                let min_idx = feature_dim * 2 + i;
                let max_idx = feature_dim * 3 + i;
                if min_idx < embedding.len() {
                    embedding[min_idx] = min_val;
                }
                if max_idx < embedding.len() {
                    embedding[max_idx] = max_val;
                }
            }
        }

        // Resize to target dimension
        embedding.resize(self.config.dimension, 0.0);

        Ok(embedding)
    }

    /// Compute voice quality metrics
    fn compute_voice_quality(&self, audio: &[f32], sample_rate: u32) -> Result<VoiceQuality> {
        if audio.is_empty() {
            return Ok(VoiceQuality::default());
        }

        // Fundamental frequency statistics
        let f0_values = self.extract_f0_contour(audio, sample_rate)?;
        let f0_mean = f0_values.iter().sum::<f32>() / f0_values.len() as f32;
        let f0_variance =
            f0_values.iter().map(|f| (f - f0_mean).powi(2)).sum::<f32>() / f0_values.len() as f32;
        let f0_std = f0_variance.sqrt();

        // Energy statistics
        let energy_values: Vec<f32> = audio
            .chunks(512)
            .map(|chunk| chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32)
            .collect();
        let energy_mean = energy_values.iter().sum::<f32>() / energy_values.len() as f32;
        let energy_variance = energy_values
            .iter()
            .map(|e| (e - energy_mean).powi(2))
            .sum::<f32>()
            / energy_values.len() as f32;
        let energy_std = energy_variance.sqrt();

        // Spectral characteristics (simplified)
        let spectral_centroid = self.compute_spectral_centroid(audio, sample_rate)?;
        let spectral_bandwidth = self.compute_spectral_bandwidth(audio, sample_rate)?;

        // Voice quality metrics (simplified)
        let jitter = self.compute_jitter(&f0_values);
        let shimmer = self.compute_shimmer(&energy_values);

        Ok(VoiceQuality {
            f0_mean,
            f0_std,
            spectral_centroid,
            spectral_bandwidth,
            jitter,
            shimmer,
            energy_mean,
            energy_std,
        })
    }

    /// Extract F0 contour from audio
    fn extract_f0_contour(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let frame_size = (sample_rate as f32 * 0.025) as usize; // 25ms
        let hop_size = (sample_rate as f32 * 0.010) as usize; // 10ms
        let mut f0_values = Vec::new();

        for i in (0..audio.len()).step_by(hop_size) {
            let end = (i + frame_size).min(audio.len());
            let frame = &audio[i..end];

            if frame.len() >= frame_size / 2 {
                let f0 = self.estimate_f0_autocorr(frame, sample_rate);
                f0_values.push(f0);
            }
        }

        if f0_values.is_empty() {
            f0_values.push(0.0);
        }

        Ok(f0_values)
    }

    /// Estimate F0 using autocorrelation
    fn estimate_f0_autocorr(&self, frame: &[f32], sample_rate: u32) -> f32 {
        let min_period = sample_rate / 500; // 500 Hz max
        let max_period = sample_rate / 50; // 50 Hz min

        let mut max_corr = 0.0;
        let mut best_period = min_period;

        for period in min_period..max_period.min(frame.len() as u32 / 2) {
            let mut correlation = 0.0;
            let period_samples = period as usize;

            for i in 0..(frame.len() - period_samples) {
                correlation += frame[i] * frame[i + period_samples];
            }

            if correlation > max_corr {
                max_corr = correlation;
                best_period = period;
            }
        }

        if max_corr > 0.0 {
            sample_rate as f32 / best_period as f32
        } else {
            0.0
        }
    }

    /// Compute spectral centroid
    fn compute_spectral_centroid(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        // Simplified spectral centroid calculation
        Ok(sample_rate as f32 / 4.0) // Placeholder
    }

    /// Compute spectral bandwidth
    fn compute_spectral_bandwidth(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        // Simplified spectral bandwidth calculation
        Ok(sample_rate as f32 / 8.0) // Placeholder
    }

    /// Compute jitter (F0 perturbation)
    fn compute_jitter(&self, f0_values: &[f32]) -> f32 {
        if f0_values.len() < 2 {
            return 0.0;
        }

        let mut period_diffs = Vec::new();
        for i in 1..f0_values.len() {
            if f0_values[i] > 0.0 && f0_values[i - 1] > 0.0 {
                let period1 = 1.0 / f0_values[i - 1];
                let period2 = 1.0 / f0_values[i];
                period_diffs.push((period2 - period1).abs());
            }
        }

        if period_diffs.is_empty() {
            0.0
        } else {
            period_diffs.iter().sum::<f32>() / period_diffs.len() as f32
        }
    }

    /// Compute shimmer (amplitude perturbation)
    fn compute_shimmer(&self, energy_values: &[f32]) -> f32 {
        if energy_values.len() < 2 {
            return 0.0;
        }

        let mut amplitude_diffs = Vec::new();
        for i in 1..energy_values.len() {
            let amp1 = energy_values[i - 1].sqrt();
            let amp2 = energy_values[i].sqrt();
            if amp1 > 0.0 {
                amplitude_diffs.push(((amp2 - amp1) / amp1).abs());
            }
        }

        if amplitude_diffs.is_empty() {
            0.0
        } else {
            amplitude_diffs.iter().sum::<f32>() / amplitude_diffs.len() as f32
        }
    }

    /// Compute confidence score
    fn compute_confidence(&self, audio: &[f32], embedding: &[f32]) -> Result<f32> {
        // Base confidence on signal quality
        let snr = self.estimate_snr(audio);
        let snr_confidence = (snr / 20.0).clamp(0.0, 1.0); // Normalize SNR

        // Base confidence on embedding magnitude consistency
        let embedding_norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_confidence = if embedding_norm > 0.1 && embedding_norm < 10.0 {
            1.0
        } else {
            0.5
        };

        // Combine confidence measures
        Ok((snr_confidence + norm_confidence) / 2.0)
    }

    /// Estimate signal-to-noise ratio
    fn estimate_snr(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        let signal_power = audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32;
        let noise_power = signal_power * 0.01; // Simplified noise estimation

        if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            30.0 // High SNR
        }
    }

    /// Create neural embedding network
    fn create_embedding_network(&self, vs: VarBuilder) -> Result<EmbeddingNetwork> {
        match &self.config.network_architecture {
            NetworkArchitecture::CNN {
                conv_layers,
                fc_layers,
            } => {
                let mut conv_layers_nn = Vec::new();

                // Create convolutional layers
                let mut in_channels = 1;
                for (i, &(out_channels, kernel_size)) in conv_layers.iter().enumerate() {
                    let conv = conv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        candle_nn::Conv2dConfig::default(),
                        vs.pp(&format!("conv_{}", i)),
                    )?;
                    conv_layers_nn.push(conv);
                    in_channels = out_channels;
                }

                // Create fully connected layers
                let mut fc_layers_nn = Vec::new();
                let mut current_dim = fc_layers.first().copied().unwrap_or(256);

                for (i, &output_dim) in fc_layers.iter().enumerate() {
                    let fc = linear(current_dim, output_dim, vs.pp(&format!("fc_{}", i)))?;
                    fc_layers_nn.push(fc);
                    current_dim = output_dim;
                }

                // Add final embedding layer
                let final_layer = linear(current_dim, self.config.dimension, vs.pp("embedding"))?;
                fc_layers_nn.push(final_layer);

                Ok(EmbeddingNetwork {
                    layers: fc_layers_nn,
                    conv_layers: conv_layers_nn,
                    device: self.device.clone(),
                    architecture: self.config.network_architecture.clone(),
                })
            }
            _ => {
                // Default simple fully connected network
                let layer1 = linear(self.config.num_mel_filters, 256, vs.pp("fc1"))?;
                let layer2 = linear(256, 128, vs.pp("fc2"))?;
                let layer3 = linear(128, self.config.dimension, vs.pp("embedding"))?;

                Ok(EmbeddingNetwork {
                    layers: vec![layer1, layer2, layer3],
                    conv_layers: Vec::new(),
                    device: self.device.clone(),
                    architecture: self.config.network_architecture.clone(),
                })
            }
        }
    }

    /// Convert features to tensor
    fn features_to_tensor(&self, features: &Array2<f32>) -> Result<Tensor> {
        let (time_frames, feature_dim) = features.dim();
        let data: Vec<f32> = features.iter().cloned().collect();

        Tensor::from_vec(data, (time_frames, feature_dim), &self.device)
            .map_err(|e| Error::Processing(format!("Failed to create tensor: {}", e)))
    }

    /// Normalize features using statistics
    fn normalize_features(
        &self,
        features: &Array2<f32>,
        stats: &NormalizationStats,
    ) -> Result<Array2<f32>> {
        let mut normalized = features.clone();
        let (_, feature_dim) = features.dim();

        match self.config.preprocessing.normalization {
            NormalizationMethod::ZScore => {
                for i in 0..feature_dim {
                    if i < stats.mean.len() && i < stats.std.len() && stats.std[i] > 0.0 {
                        for mut row in normalized.column_mut(i) {
                            *row = (*row - stats.mean[i]) / stats.std[i];
                        }
                    }
                }
            }
            NormalizationMethod::MinMax => {
                for i in 0..feature_dim {
                    if i < stats.min.len() && i < stats.max.len() {
                        let range = stats.max[i] - stats.min[i];
                        if range > 0.0 {
                            for mut row in normalized.column_mut(i) {
                                *row = (*row - stats.min[i]) / range;
                            }
                        }
                    }
                }
            }
            _ => {
                // Other normalization methods
            }
        }

        Ok(normalized)
    }

    /// Average voice quality metrics
    fn average_voice_quality(&self, embeddings: &[SpeakerEmbedding]) -> Result<VoiceQuality> {
        if embeddings.is_empty() {
            return Ok(VoiceQuality::default());
        }

        let mut avg_quality = VoiceQuality::default();

        for embedding in embeddings {
            let quality = &embedding.metadata.voice_quality;
            avg_quality.f0_mean += quality.f0_mean;
            avg_quality.f0_std += quality.f0_std;
            avg_quality.spectral_centroid += quality.spectral_centroid;
            avg_quality.spectral_bandwidth += quality.spectral_bandwidth;
            avg_quality.jitter += quality.jitter;
            avg_quality.shimmer += quality.shimmer;
            avg_quality.energy_mean += quality.energy_mean;
            avg_quality.energy_std += quality.energy_std;
        }

        let n = embeddings.len() as f32;
        avg_quality.f0_mean /= n;
        avg_quality.f0_std /= n;
        avg_quality.spectral_centroid /= n;
        avg_quality.spectral_bandwidth /= n;
        avg_quality.jitter /= n;
        avg_quality.shimmer /= n;
        avg_quality.energy_mean /= n;
        avg_quality.energy_std /= n;

        Ok(avg_quality)
    }

    /// Apply temporal smoothing to embedding sequence
    fn apply_temporal_smoothing(
        &self,
        embeddings: &[SpeakerEmbedding],
        smoothing_factor: f32,
    ) -> Result<Vec<SpeakerEmbedding>> {
        if embeddings.is_empty() {
            return Ok(vec![]);
        }

        if embeddings.len() == 1 {
            return Ok(embeddings.to_vec());
        }

        let mut smoothed = Vec::with_capacity(embeddings.len());
        smoothed.push(embeddings[0].clone()); // First embedding unchanged

        for i in 1..embeddings.len() {
            let mut smoothed_vector = embeddings[i].vector.clone();

            // Apply exponential smoothing
            for (j, (&current, &previous)) in embeddings[i]
                .vector
                .iter()
                .zip(&smoothed[i - 1].vector)
                .enumerate()
            {
                smoothed_vector[j] =
                    previous * smoothing_factor + current * (1.0 - smoothing_factor);
            }

            let smoothed_confidence = smoothed[i - 1].confidence * smoothing_factor
                + embeddings[i].confidence * (1.0 - smoothing_factor);

            let mut smoothed_embedding = SpeakerEmbedding {
                vector: smoothed_vector,
                dimension: embeddings[i].dimension,
                confidence: smoothed_confidence,
                metadata: embeddings[i].metadata.clone(),
            };

            smoothed_embedding.normalize();
            smoothed.push(smoothed_embedding);
        }

        Ok(smoothed)
    }

    /// Weighted aggregation of embeddings
    fn weighted_aggregation(&self, embeddings: &[SpeakerEmbedding]) -> Result<SpeakerEmbedding> {
        if embeddings.is_empty() {
            return Err(Error::Processing("No embeddings to aggregate".to_string()));
        }

        let dimension = embeddings[0].dimension;
        let mut weighted_vector = vec![0.0; dimension];
        let mut total_weight = 0.0;

        // Use quality score as weight
        for embedding in embeddings {
            let weight = embedding.metadata.voice_quality.overall_quality() * embedding.confidence;
            total_weight += weight;

            for (i, &value) in embedding.vector.iter().enumerate() {
                weighted_vector[i] += value * weight;
            }
        }

        if total_weight > 0.0 {
            for value in &mut weighted_vector {
                *value /= total_weight;
            }
        }

        let avg_confidence =
            embeddings.iter().map(|e| e.confidence).sum::<f32>() / embeddings.len() as f32;
        let avg_quality = self.average_voice_quality(embeddings)?;

        let metadata = EmbeddingMetadata {
            gender: None,
            age_estimate: None,
            language: None,
            emotion: None,
            voice_quality: avg_quality,
            extraction_time: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64(),
            ),
        };

        let mut aggregated = SpeakerEmbedding {
            vector: weighted_vector,
            dimension,
            confidence: avg_confidence,
            metadata,
        };

        aggregated.normalize();
        Ok(aggregated)
    }

    /// Compute embedding stability over time
    pub fn compute_embedding_stability(
        &self,
        embeddings: &[SpeakerEmbedding],
    ) -> Result<EmbeddingStability> {
        if embeddings.len() < 2 {
            return Ok(EmbeddingStability {
                temporal_consistency: 1.0,
                confidence_stability: 1.0,
                drift_rate: 0.0,
                stability_score: 1.0,
            });
        }

        let mut similarities = Vec::new();
        let mut confidence_changes = Vec::new();

        // Compute pairwise similarities and confidence changes
        for i in 1..embeddings.len() {
            let similarity = embeddings[i - 1].similarity(&embeddings[i]);
            similarities.push(similarity);

            let conf_change = (embeddings[i].confidence - embeddings[i - 1].confidence).abs();
            confidence_changes.push(conf_change);
        }

        // Temporal consistency (average similarity between consecutive embeddings)
        let temporal_consistency = similarities.iter().sum::<f32>() / similarities.len() as f32;

        // Confidence stability (1 - average confidence change)
        let avg_conf_change =
            confidence_changes.iter().sum::<f32>() / confidence_changes.len() as f32;
        let confidence_stability = (1.0 - avg_conf_change).max(0.0);

        // Drift rate (change in similarity over time)
        let drift_rate = if similarities.len() > 1 {
            let early_avg = similarities[..similarities.len() / 2].iter().sum::<f32>()
                / (similarities.len() / 2) as f32;
            let late_avg = similarities[similarities.len() / 2..].iter().sum::<f32>()
                / (similarities.len() - similarities.len() / 2) as f32;
            (early_avg - late_avg).abs()
        } else {
            0.0
        };

        // Overall stability score
        let stability_score =
            (temporal_consistency + confidence_stability) / 2.0 * (1.0 - drift_rate);

        Ok(EmbeddingStability {
            temporal_consistency,
            confidence_stability,
            drift_rate,
            stability_score,
        })
    }

    /// Fast similarity-based speaker identification
    pub fn identify_speaker_fast(
        &self,
        query_embedding: &SpeakerEmbedding,
        speaker_database: &[(String, SpeakerEmbedding)],
        threshold: f32,
    ) -> Result<Vec<SpeakerMatch>> {
        let mut matches = Vec::new();

        for (speaker_id, ref_embedding) in speaker_database {
            let similarity = query_embedding.similarity(ref_embedding);

            if similarity >= threshold {
                matches.push(SpeakerMatch {
                    speaker_id: speaker_id.clone(),
                    similarity,
                    confidence: ref_embedding.confidence,
                    distance: 1.0 - similarity,
                });
            }
        }

        // Sort by similarity (highest first)
        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

        Ok(matches)
    }
}

impl std::fmt::Debug for FeatureExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeatureExtractor")
            .field("config", &self.config)
            .field("fft_planner", &"<RealFftPlanner>")
            .field(
                "mel_filterbank",
                &self.mel_filterbank.as_ref().map(|_| "<mel_filterbank>"),
            )
            .field(
                "dct_matrix",
                &self.dct_matrix.as_ref().map(|_| "<dct_matrix>"),
            )
            .finish()
    }
}

impl FeatureExtractor {
    /// Create new feature extractor
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        let mut fft_planner = RealFftPlanner::new();

        let mel_filterbank = if matches!(
            config.feature_method,
            FeatureExtractionMethod::MelSpectrogram
                | FeatureExtractionMethod::LogMel
                | FeatureExtractionMethod::MFCC
        ) {
            Some(Self::create_mel_filterbank(
                config.fft_size / 2 + 1,
                config.num_mel_filters,
                8000.0,
            )?)
        } else {
            None
        };

        let dct_matrix = if matches!(config.feature_method, FeatureExtractionMethod::MFCC) {
            Some(Self::create_dct_matrix(
                config.num_mel_filters,
                config.num_mel_filters,
            )?)
        } else {
            None
        };

        Ok(Self {
            config,
            fft_planner,
            mel_filterbank,
            dct_matrix,
        })
    }

    /// Extract features from preprocessed audio
    pub fn extract_features(&mut self, audio: &[f32], sample_rate: u32) -> Result<Array2<f32>> {
        match self.config.feature_method {
            FeatureExtractionMethod::MFCC => self.extract_mfcc(audio, sample_rate),
            FeatureExtractionMethod::MelSpectrogram => {
                self.extract_mel_spectrogram(audio, sample_rate)
            }
            FeatureExtractionMethod::LogMel => self.extract_log_mel(audio, sample_rate),
            FeatureExtractionMethod::Spectrogram => self.extract_spectrogram(audio, sample_rate),
            FeatureExtractionMethod::PLP => self.extract_plp(audio, sample_rate),
            FeatureExtractionMethod::FilterBank => self.extract_filterbank(audio, sample_rate),
        }
    }

    /// Extract MFCC features
    fn extract_mfcc(&mut self, audio: &[f32], sample_rate: u32) -> Result<Array2<f32>> {
        // Extract mel spectrogram first
        let mel_spec = self.extract_mel_spectrogram(audio, sample_rate)?;

        // Apply DCT to get MFCC
        if let Some(dct_matrix) = &self.dct_matrix {
            let (time_frames, _) = mel_spec.dim();
            let mut mfcc = Array2::zeros((time_frames, self.config.num_mel_filters));

            for t in 0..time_frames {
                let mel_frame = mel_spec.row(t);
                let mut mfcc_frame = mfcc.row_mut(t);

                for i in 0..self.config.num_mel_filters {
                    let mut sum = 0.0;
                    for j in 0..self.config.num_mel_filters {
                        sum += dct_matrix[[i, j]] * mel_frame[j];
                    }
                    mfcc_frame[i] = sum;
                }
            }

            Ok(mfcc)
        } else {
            Ok(mel_spec)
        }
    }

    /// Extract mel spectrogram
    fn extract_mel_spectrogram(&mut self, audio: &[f32], sample_rate: u32) -> Result<Array2<f32>> {
        let spectrogram = self.extract_spectrogram(audio, sample_rate)?;

        if let Some(mel_filterbank) = &self.mel_filterbank {
            let (time_frames, _) = spectrogram.dim();
            let mut mel_spec = Array2::zeros((time_frames, self.config.num_mel_filters));

            for t in 0..time_frames {
                let spec_frame = spectrogram.row(t);
                let mut mel_frame = mel_spec.row_mut(t);

                for i in 0..self.config.num_mel_filters {
                    let mut sum = 0.0;
                    for j in 0..spec_frame.len() {
                        sum += mel_filterbank[[i, j]] * spec_frame[j];
                    }
                    mel_frame[i] = sum;
                }
            }

            Ok(mel_spec)
        } else {
            Ok(spectrogram)
        }
    }

    /// Extract log mel spectrogram
    fn extract_log_mel(&mut self, audio: &[f32], sample_rate: u32) -> Result<Array2<f32>> {
        let mut mel_spec = self.extract_mel_spectrogram(audio, sample_rate)?;

        // Apply log transformation
        mel_spec.mapv_inplace(|x| (x + 1e-8).ln());

        Ok(mel_spec)
    }

    /// Extract spectrogram
    fn extract_spectrogram(&mut self, audio: &[f32], sample_rate: u32) -> Result<Array2<f32>> {
        let window_size = self.config.window_size;
        let hop_size = self.config.hop_size;
        let fft_size = self.config.fft_size;

        let num_frames = (audio.len().saturating_sub(window_size)) / hop_size + 1;
        let num_bins = fft_size / 2 + 1;
        let mut spectrogram = Array2::zeros((num_frames, num_bins));

        let window = self.create_hann_window(window_size);

        for (frame_idx, i) in (0..audio.len()).step_by(hop_size).enumerate() {
            if frame_idx >= num_frames {
                break;
            }

            let end = (i + window_size).min(audio.len());
            let mut frame = vec![0.0; fft_size];

            // Apply window and zero-pad
            for (j, &sample) in audio[i..end].iter().enumerate() {
                if j < window_size {
                    frame[j] = sample * window[j];
                }
            }

            // Compute FFT using realfft
            let fft = self.fft_planner.plan_fft_forward(fft_size);
            let mut spectrum = fft.make_output_vec();
            fft.process(&mut frame, &mut spectrum).unwrap_or_default();

            // Compute magnitude spectrum
            for (j, complex_val) in spectrum.iter().take(num_bins).enumerate() {
                spectrogram[[frame_idx, j]] = complex_val.norm();
            }
        }

        Ok(spectrogram)
    }

    /// Extract PLP features (simplified)
    fn extract_plp(&mut self, audio: &[f32], sample_rate: u32) -> Result<Array2<f32>> {
        // Simplified PLP extraction - in practice would use bark scale and equal loudness
        self.extract_mel_spectrogram(audio, sample_rate)
    }

    /// Extract filter bank features
    fn extract_filterbank(&mut self, audio: &[f32], sample_rate: u32) -> Result<Array2<f32>> {
        self.extract_mel_spectrogram(audio, sample_rate)
    }

    /// Create Hann window
    fn create_hann_window(&self, size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos())
            })
            .collect()
    }

    /// Create mel filterbank
    fn create_mel_filterbank(
        num_fft_bins: usize,
        num_mel_filters: usize,
        sample_rate: f32,
    ) -> Result<Array2<f32>> {
        let mut filterbank = Array2::zeros((num_mel_filters, num_fft_bins));

        // Mel scale conversion
        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

        let low_freq_mel = hz_to_mel(0.0);
        let high_freq_mel = hz_to_mel(sample_rate / 2.0);

        // Create filter bank
        let mel_points: Vec<f32> = (0..=num_mel_filters + 1)
            .map(|i| {
                mel_to_hz(
                    low_freq_mel
                        + (high_freq_mel - low_freq_mel) * i as f32 / (num_mel_filters + 1) as f32,
                )
            })
            .collect();

        let bin_points: Vec<usize> = mel_points
            .iter()
            .map(|&freq| ((num_fft_bins - 1) as f32 * freq / (sample_rate / 2.0)) as usize)
            .collect();

        for m in 0..num_mel_filters {
            for k in bin_points[m]..bin_points[m + 1] {
                if k < num_fft_bins {
                    filterbank[[m, k]] =
                        (k - bin_points[m]) as f32 / (bin_points[m + 1] - bin_points[m]) as f32;
                }
            }
            for k in bin_points[m + 1]..bin_points[m + 2] {
                if k < num_fft_bins {
                    filterbank[[m, k]] = (bin_points[m + 2] - k) as f32
                        / (bin_points[m + 2] - bin_points[m + 1]) as f32;
                }
            }
        }

        Ok(filterbank)
    }

    /// Create DCT matrix for MFCC
    fn create_dct_matrix(input_size: usize, output_size: usize) -> Result<Array2<f32>> {
        let mut dct_matrix = Array2::zeros((output_size, input_size));

        for i in 0..output_size {
            for j in 0..input_size {
                dct_matrix[[i, j]] =
                    (std::f32::consts::PI * i as f32 * (j as f32 + 0.5) / input_size as f32).cos();
                if i == 0 {
                    dct_matrix[[i, j]] *= (1.0 / input_size as f32).sqrt();
                } else {
                    dct_matrix[[i, j]] *= (2.0 / input_size as f32).sqrt();
                }
            }
        }

        Ok(dct_matrix)
    }
}

impl EmbeddingNetwork {
    /// Forward pass through the network
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();

        // Apply convolutional layers if present
        for conv_layer in &self.conv_layers {
            x = conv_layer.forward(&x)?;
            x = x.relu()?;
        }

        // Flatten for fully connected layers
        if !self.conv_layers.is_empty() {
            let shape = x.shape();
            let batch_size = shape.dims()[0];
            let flattened_size = shape.elem_count() / batch_size;
            x = x.reshape(&[batch_size, flattened_size])?;
        }

        // Apply fully connected layers
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;

            // Apply ReLU activation except for the last layer
            if i < self.layers.len() - 1 {
                x = x.relu()?;
            }
        }

        Ok(x)
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            window_size: 1024,
            hop_size: 512,
            fft_size: 1024,
            num_mel_filters: 80,
            network_architecture: NetworkArchitecture::CNN {
                conv_layers: vec![(32, 3), (64, 3), (128, 3)],
                fc_layers: vec![256, 128],
            },
            feature_method: FeatureExtractionMethod::MelSpectrogram,
            preprocessing: PreprocessingConfig::default(),
            batch_size: 8,
        }
    }
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            vad_enabled: true,
            noise_reduction: false,
            normalization: NormalizationMethod::ZScore,
            augmentation: None,
        }
    }
}

impl Default for EmbeddingMetadata {
    fn default() -> Self {
        Self {
            gender: None,
            age_estimate: None,
            language: None,
            emotion: None,
            voice_quality: VoiceQuality::default(),
            extraction_time: None,
        }
    }
}

impl Default for VoiceQuality {
    fn default() -> Self {
        Self {
            f0_mean: 0.0,
            f0_std: 0.0,
            spectral_centroid: 0.0,
            spectral_bandwidth: 0.0,
            jitter: 0.0,
            shimmer: 0.0,
            energy_mean: 0.0,
            energy_std: 0.0,
        }
    }
}

impl VoiceQuality {
    /// Compute overall quality score
    pub fn overall_quality(&self) -> f32 {
        // Simplified quality score based on voice quality metrics
        let f0_stability = 1.0 / (1.0 + self.jitter * 1000.0); // Lower jitter = higher quality
        let amplitude_stability = 1.0 / (1.0 + self.shimmer * 100.0); // Lower shimmer = higher quality
        let energy_consistency = if self.energy_std > 0.0 {
            self.energy_mean / (self.energy_mean + self.energy_std)
        } else {
            1.0
        };

        (f0_stability + amplitude_stability + energy_consistency) / 3.0
    }
}

impl Default for SpeakerEmbeddingExtractor {
    fn default() -> Self {
        Self::new(EmbeddingConfig::default())
            .expect("Failed to create default SpeakerEmbeddingExtractor")
    }
}

impl Default for OnlineLearningConfig {
    fn default() -> Self {
        Self {
            initial_learning_rate: 0.01,
            decay_factor: 0.99,
            min_learning_rate: 0.001,
            convergence_threshold: 0.95,
            max_steps: 100,
        }
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            window_duration: 1.0, // 1 second windows
            hop_duration: 0.5,    // 0.5 second hops (50% overlap)
            temporal_smoothing: true,
            smoothing_factor: 0.3,
            aggregation_method: AggregationMethod::Weighted,
            min_confidence_threshold: 0.5,
        }
    }
}

impl Default for RefinementConfig {
    fn default() -> Self {
        Self {
            base_refinement_weight: 0.1,
            quality_amplification: 2.0,
            max_refinement_weight: 0.5,
            convergence_threshold: 0.001,
            convergence_window: 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::VoiceSample;

    #[tokio::test]
    async fn test_embedding_extraction() {
        let mut extractor = SpeakerEmbeddingExtractor::new(EmbeddingConfig::default()).unwrap();

        let audio = vec![0.1; 16000]; // 1 second of audio
        let sample = VoiceSample::new("test".to_string(), audio, 16000);

        let embedding = extractor.extract(&sample).await.unwrap();
        assert_eq!(embedding.dimension, 512);
        assert!(embedding.is_valid());
    }

    #[test]
    fn test_embedding_similarity() {
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0];
        let vec3 = vec![1.0, 0.0, 0.0];

        let emb1 = SpeakerEmbedding::new(vec1);
        let emb2 = SpeakerEmbedding::new(vec2);
        let emb3 = SpeakerEmbedding::new(vec3);

        assert!((emb1.similarity(&emb2) - 0.0).abs() < 0.001); // Orthogonal vectors
        assert!((emb1.similarity(&emb3) - 1.0).abs() < 0.001); // Identical vectors
    }

    #[tokio::test]
    async fn test_batch_extraction() {
        let mut extractor = SpeakerEmbeddingExtractor::new(EmbeddingConfig::default()).unwrap();

        let samples: Vec<VoiceSample> = (0..3)
            .map(|i| {
                let audio = vec![0.1 * i as f32; 8000];
                VoiceSample::new(format!("sample_{}", i), audio, 16000)
            })
            .collect();

        let embeddings = extractor.extract_batch(&samples).await.unwrap();
        assert_eq!(embeddings.len(), 3);

        for embedding in &embeddings {
            assert!(embedding.is_valid());
        }
    }

    #[tokio::test]
    async fn test_embedding_averaging() {
        let extractor = SpeakerEmbeddingExtractor::new(EmbeddingConfig::default()).unwrap();

        let embeddings = vec![
            SpeakerEmbedding::new(vec![1.0, 0.0, 0.0]),
            SpeakerEmbedding::new(vec![0.0, 1.0, 0.0]),
            SpeakerEmbedding::new(vec![0.0, 0.0, 1.0]),
        ];

        let averaged = extractor.average_embeddings(&embeddings).unwrap();
        assert_eq!(averaged.dimension, 3);

        // Should be [1/3, 1/3, 1/3] (not normalized)
        let expected_value = 1.0 / 3.0;
        for &value in &averaged.vector {
            assert!(
                (value - expected_value).abs() < 0.01,
                "Expected {}, got {}",
                expected_value,
                value
            );
        }
    }

    #[test]
    fn test_voice_quality_metrics() {
        let quality = VoiceQuality {
            f0_mean: 150.0,
            f0_std: 10.0,
            spectral_centroid: 2000.0,
            spectral_bandwidth: 1000.0,
            jitter: 0.01,
            shimmer: 0.05,
            energy_mean: 0.1,
            energy_std: 0.02,
        };

        let overall_quality = quality.overall_quality();
        assert!(overall_quality > 0.0 && overall_quality <= 1.0);
    }

    #[test]
    fn test_feature_extractor() {
        let config = EmbeddingConfig::default();
        let mut extractor = FeatureExtractor::new(config).unwrap();

        let audio = vec![0.1; 16000];
        let features = extractor.extract_features(&audio, 16000).unwrap();

        let (time_frames, feature_dim) = features.dim();
        assert!(time_frames > 0);
        assert!(feature_dim > 0);
    }

    #[tokio::test]
    async fn test_real_time_adaptation() {
        let mut extractor = SpeakerEmbeddingExtractor::new(EmbeddingConfig::default()).unwrap();

        let base_audio = (0..16000).map(|i| 0.1 * (i as f32 * 0.001).sin()).collect();
        let adaptation_audio = (0..16000)
            .map(|i| 0.12 * (i as f32 * 0.002).sin())
            .collect();

        let base_sample = VoiceSample::new("base".to_string(), base_audio, 16000);
        let adaptation_sample = VoiceSample::new("adaptation".to_string(), adaptation_audio, 16000);

        let base_embedding = extractor.extract(&base_sample).await.unwrap();
        let adapted_embedding = extractor
            .adapt_embedding_realtime(
                &base_embedding,
                &adaptation_sample,
                0.3, // Medium adaptation rate
            )
            .await
            .unwrap();

        assert_eq!(adapted_embedding.dimension, base_embedding.dimension);
        assert!(adapted_embedding.is_valid());

        // Adapted embedding should be different but similar to base
        let similarity = base_embedding.similarity(&adapted_embedding);
        assert!(similarity > 0.8); // Should be quite similar
        assert!(similarity < 1.0); // But not identical
    }

    #[tokio::test]
    async fn test_online_learning() {
        let mut extractor = SpeakerEmbeddingExtractor::new(EmbeddingConfig::default()).unwrap();

        let base_audio = vec![0.1; 16000];
        let base_sample = VoiceSample::new("base".to_string(), base_audio, 16000);
        let mut base_embedding = extractor.extract(&base_sample).await.unwrap();

        // Create adaptation samples
        let adaptation_samples: Vec<VoiceSample> = (0..3)
            .map(|i| {
                let audio = vec![0.1 + 0.01 * i as f32; 8000];
                VoiceSample::new(format!("adapt_{}", i), audio, 16000)
            })
            .collect();

        let mut learning_config = OnlineLearningConfig::default();
        learning_config.convergence_threshold = 0.99; // Higher threshold to process all samples
        let metrics = extractor
            .online_learning_adaptation(&mut base_embedding, &adaptation_samples, &learning_config)
            .await
            .unwrap();

        assert!(metrics.len() >= 1); // Should process at least one sample
        assert!(metrics.len() <= adaptation_samples.len()); // Should not process more than available
        assert!(base_embedding.is_valid());

        // Check that learning rates decay
        for i in 1..metrics.len() {
            assert!(metrics[i].learning_rate <= metrics[i - 1].learning_rate);
        }
    }

    #[tokio::test]
    async fn test_streaming_extraction() {
        let mut extractor = SpeakerEmbeddingExtractor::new(EmbeddingConfig::default()).unwrap();

        // Create a longer audio stream
        let audio_stream: Vec<f32> = (0..32000).map(|i| 0.1 * (i as f32 * 0.001).sin()).collect();

        let streaming_config = StreamingConfig::default();
        let result = extractor
            .extract_streaming(&audio_stream, 16000, &streaming_config)
            .await
            .unwrap();

        assert!(result.streaming_stats.num_windows > 1);
        assert!(result.aggregated_embedding.is_valid());
        assert!(!result.window_embeddings.is_empty());
        assert_eq!(
            result.window_embeddings.len(),
            result.streaming_stats.num_windows
        );
    }

    #[tokio::test]
    async fn test_adaptive_refinement() {
        let mut extractor = SpeakerEmbeddingExtractor::new(EmbeddingConfig::default()).unwrap();

        let base_audio = vec![0.1; 16000];
        let base_sample = VoiceSample::new("base".to_string(), base_audio, 16000);
        let base_embedding = extractor.extract(&base_sample).await.unwrap();

        // Create refinement samples with quality feedback
        let refinement_samples: Vec<(VoiceSample, f32)> = (0..3)
            .map(|i| {
                let audio = vec![0.1 + 0.005 * i as f32; 8000];
                let sample = VoiceSample::new(format!("refine_{}", i), audio, 16000);
                let quality = 0.8 + 0.05 * i as f32; // Increasing quality
                (sample, quality)
            })
            .collect();

        let refinement_config = RefinementConfig::default();
        let result = extractor
            .refine_embedding_adaptive(&base_embedding, &refinement_samples, &refinement_config)
            .await
            .unwrap();

        assert!(result.refined_embedding.is_valid());
        assert!(!result.refinement_history.is_empty());
        assert!(result.refinement_history.len() <= refinement_samples.len());

        // Higher quality samples should have higher adaptive weights
        for i in 1..result.refinement_history.len() {
            if result.refinement_history[i].quality_feedback
                > result.refinement_history[i - 1].quality_feedback
            {
                assert!(
                    result.refinement_history[i].adaptive_weight
                        >= result.refinement_history[i - 1].adaptive_weight
                );
            }
        }
    }

    #[test]
    fn test_incremental_update() {
        let extractor = SpeakerEmbeddingExtractor::new(EmbeddingConfig::default()).unwrap();

        let mut current_embedding = SpeakerEmbedding::new(vec![1.0, 0.0, 0.0]);
        let new_embedding = SpeakerEmbedding::new(vec![0.0, 1.0, 0.0]);

        let similarity_before = extractor
            .update_embedding_incremental(
                &mut current_embedding,
                &new_embedding,
                0.5, // 50% update weight
            )
            .unwrap();

        // Verify the embedding was updated
        assert!(current_embedding.vector[0] < 1.0); // Should have moved away from original
        assert!(current_embedding.vector[1] > 0.0); // Should have moved toward new
        assert!(similarity_before >= 0.0);
        assert!(current_embedding.is_valid());
    }

    #[test]
    fn test_temporal_smoothing() {
        let extractor = SpeakerEmbeddingExtractor::new(EmbeddingConfig::default()).unwrap();

        let embeddings = vec![
            SpeakerEmbedding::new(vec![1.0, 0.0, 0.0]),
            SpeakerEmbedding::new(vec![0.5, 0.5, 0.0]),
            SpeakerEmbedding::new(vec![0.0, 1.0, 0.0]),
        ];

        let smoothed = extractor
            .apply_temporal_smoothing(&embeddings, 0.3)
            .unwrap();

        assert_eq!(smoothed.len(), embeddings.len());

        // First embedding should be unchanged
        assert_eq!(smoothed[0].vector, embeddings[0].vector);

        // Subsequent embeddings should be smoothed
        for i in 1..smoothed.len() {
            assert!(smoothed[i].is_valid());
        }
    }

    #[test]
    fn test_weighted_aggregation() {
        let extractor = SpeakerEmbeddingExtractor::new(EmbeddingConfig::default()).unwrap();

        let mut embeddings = vec![
            SpeakerEmbedding::new(vec![1.0, 0.0, 0.0]),
            SpeakerEmbedding::new(vec![0.0, 1.0, 0.0]),
        ];

        // Set different confidence levels
        embeddings[0].confidence = 0.9;
        embeddings[1].confidence = 0.3;

        let aggregated = extractor.weighted_aggregation(&embeddings).unwrap();

        assert!(aggregated.is_valid());
        assert_eq!(aggregated.dimension, 3);

        // The result should be closer to the high-confidence embedding
        assert!(aggregated.vector[0] > aggregated.vector[1]);
    }

    #[test]
    fn test_embedding_stability() {
        let extractor = SpeakerEmbeddingExtractor::new(EmbeddingConfig::default()).unwrap();

        // Create a sequence of similar embeddings (stable case)
        let stable_embeddings = vec![
            SpeakerEmbedding::new(vec![1.0, 0.0, 0.0]),
            SpeakerEmbedding::new(vec![0.95, 0.05, 0.0]),
            SpeakerEmbedding::new(vec![0.9, 0.1, 0.0]),
        ];

        let stability = extractor
            .compute_embedding_stability(&stable_embeddings)
            .unwrap();

        assert!(stability.stability_score > 0.0);
        assert!(stability.temporal_consistency > 0.8); // Should be high for similar embeddings
        assert!(stability.drift_rate >= 0.0);
    }

    #[test]
    fn test_speaker_identification() {
        let extractor = SpeakerEmbeddingExtractor::new(EmbeddingConfig::default()).unwrap();

        let query = SpeakerEmbedding::new(vec![1.0, 0.0, 0.0]);

        let speaker_database = vec![
            (
                "speaker1".to_string(),
                SpeakerEmbedding::new(vec![0.95, 0.05, 0.0]),
            ),
            (
                "speaker2".to_string(),
                SpeakerEmbedding::new(vec![0.0, 1.0, 0.0]),
            ),
            (
                "speaker3".to_string(),
                SpeakerEmbedding::new(vec![0.9, 0.1, 0.0]),
            ),
        ];

        let matches = extractor
            .identify_speaker_fast(&query, &speaker_database, 0.8)
            .unwrap();

        assert!(!matches.is_empty());

        // Results should be sorted by similarity (highest first)
        for i in 1..matches.len() {
            assert!(matches[i - 1].similarity >= matches[i].similarity);
        }

        // All matches should meet the threshold
        for match_result in &matches {
            assert!(match_result.similarity >= 0.8);
            assert_eq!(match_result.distance, 1.0 - match_result.similarity);
        }
    }

    #[test]
    fn test_configuration_defaults() {
        let online_config = OnlineLearningConfig::default();
        assert!(online_config.initial_learning_rate > 0.0);
        assert!(online_config.decay_factor > 0.0 && online_config.decay_factor < 1.0);

        let streaming_config = StreamingConfig::default();
        assert!(streaming_config.window_duration > 0.0);
        assert!(streaming_config.hop_duration > 0.0);
        assert!(streaming_config.hop_duration < streaming_config.window_duration);

        let refinement_config = RefinementConfig::default();
        assert!(refinement_config.base_refinement_weight > 0.0);
        assert!(
            refinement_config.max_refinement_weight >= refinement_config.base_refinement_weight
        );
    }
}
