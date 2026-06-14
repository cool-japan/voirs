//! Implementation of SpeakerEmbeddingExtractor and FeatureExtractor

use super::types::*;
use crate::{types::VoiceSample, Error, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{conv2d, linear, Conv2d, Linear, Module, VarBuilder};
use scirs2_core::ndarray::Array2;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, trace};

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
        sorted_samples.sort_by(|a, b| {
            b.1 .1
                .partial_cmp(&a.1 .1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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

        energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
            let speed_factor =
                min_speed + scirs2_core::random::random::<f32>() * (max_speed - min_speed);
            augmented = self.change_speed(&augmented, speed_factor)?;
        }

        // Volume perturbation
        if let Some((min_vol, max_vol)) = config.volume_range {
            let volume_factor =
                min_vol + scirs2_core::random::random::<f32>() * (max_vol - min_vol);
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
                column
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)),
                column
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)),
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

    /// Compute the magnitude spectrum of an audio frame via a real FFT.
    ///
    /// A Hann window is applied before the transform to reduce spectral
    /// leakage. The returned vector contains the `N/2 + 1` non-redundant
    /// magnitude bins produced by [`scirs2_fft::rfft`].
    fn rfft_magnitudes(audio: &[f32]) -> Vec<f32> {
        let n = audio.len();
        if n == 0 {
            return Vec::new();
        }
        // Hann window to limit spectral leakage.
        let windowed: Vec<f64> = audio
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

        let num_bins = n / 2 + 1;
        let spectrum = scirs2_fft::rfft(&windowed, None)
            .unwrap_or_else(|_| vec![scirs2_core::Complex::new(0.0, 0.0); num_bins]);

        spectrum
            .iter()
            .take(num_bins)
            .map(|c| ((c.re * c.re + c.im * c.im).sqrt()) as f32)
            .collect()
    }

    /// Compute the spectral centroid (Hz): the magnitude-weighted mean
    /// frequency of the signal, `Σ(f_k·|X_k|) / Σ|X_k|` with `f_k = k·sr/N`.
    fn compute_spectral_centroid(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        if audio.is_empty() {
            return Ok(0.0);
        }
        let n = audio.len();
        let mags = Self::rfft_magnitudes(audio);

        let mut weighted = 0.0f64;
        let mut total = 0.0f64;
        for (k, &m) in mags.iter().enumerate() {
            let freq = k as f64 * sample_rate as f64 / n as f64;
            weighted += freq * m as f64;
            total += m as f64;
        }

        if total > 0.0 {
            Ok((weighted / total) as f32)
        } else {
            Ok(0.0)
        }
    }

    /// Compute the spectral bandwidth (Hz): the magnitude-weighted standard
    /// deviation of frequency about the spectral centroid,
    /// `sqrt(Σ((f_k − centroid)²·|X_k|) / Σ|X_k|)`.
    fn compute_spectral_bandwidth(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        if audio.is_empty() {
            return Ok(0.0);
        }
        let n = audio.len();
        let centroid = self.compute_spectral_centroid(audio, sample_rate)? as f64;
        let mags = Self::rfft_magnitudes(audio);

        let mut weighted_var = 0.0f64;
        let mut total = 0.0f64;
        for (k, &m) in mags.iter().enumerate() {
            let freq = k as f64 * sample_rate as f64 / n as f64;
            let diff = freq - centroid;
            weighted_var += diff * diff * m as f64;
            total += m as f64;
        }

        if total > 0.0 {
            Ok((weighted_var / total).sqrt() as f32)
        } else {
            Ok(0.0)
        }
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
                        vs.pp(format!("conv_{}", i)),
                    )?;
                    conv_layers_nn.push(conv);
                    in_channels = out_channels;
                }

                // Create fully connected layers
                let mut fc_layers_nn = Vec::new();
                let mut current_dim = fc_layers.first().copied().unwrap_or(256);

                for (i, &output_dim) in fc_layers.iter().enumerate() {
                    let fc = linear(current_dim, output_dim, vs.pp(format!("fc_{}", i)))?;
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
        matches.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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

            // Compute FFT using rfft
            let frame_f64: Vec<f64> = frame.iter().map(|&x| x as f64).collect();
            let spectrum = scirs2_fft::rfft(&frame_f64, None)
                .unwrap_or_else(|_| vec![scirs2_core::Complex::new(0.0, 0.0); num_bins]);

            // Compute magnitude spectrum
            for (j, complex_val) in spectrum.iter().take(num_bins).enumerate() {
                spectrogram[[frame_idx, j]] = (complex_val.re * complex_val.re
                    + complex_val.im * complex_val.im)
                    .sqrt() as f32;
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

#[cfg(test)]
mod spectral_tests {
    use super::*;

    /// Generate a deterministic pure sine tone (no RNG).
    fn sine_tone(freq: f32, sample_rate: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn test_spectral_centroid_matches_tone() {
        let config = EmbeddingConfig::default();
        let extractor = SpeakerEmbeddingExtractor::new(config).unwrap();
        let sr = 16000u32;
        let tone_freq = 2000.0f32;
        let audio = sine_tone(tone_freq, sr, 4096);

        let centroid = extractor.compute_spectral_centroid(&audio, sr).unwrap();
        // The centroid of a single tone should land near the tone frequency.
        // Allow a window for windowing/leakage effects.
        assert!(
            (centroid - tone_freq).abs() < 200.0,
            "centroid {centroid} not near tone {tone_freq}"
        );
    }

    #[test]
    fn test_spectral_centroid_tracks_frequency() {
        let config = EmbeddingConfig::default();
        let extractor = SpeakerEmbeddingExtractor::new(config).unwrap();
        let sr = 16000u32;
        let low = sine_tone(1000.0, sr, 4096);
        let high = sine_tone(4000.0, sr, 4096);

        let c_low = extractor.compute_spectral_centroid(&low, sr).unwrap();
        let c_high = extractor.compute_spectral_centroid(&high, sr).unwrap();
        assert!(
            c_high > c_low,
            "higher tone should have higher centroid: {c_high} vs {c_low}"
        );
    }

    #[test]
    fn test_spectral_bandwidth_pure_tone_is_small() {
        let config = EmbeddingConfig::default();
        let extractor = SpeakerEmbeddingExtractor::new(config).unwrap();
        let sr = 16000u32;
        let tone = sine_tone(2000.0, sr, 4096);

        let bw = extractor.compute_spectral_bandwidth(&tone, sr).unwrap();
        // A pure tone concentrates energy at one frequency => small bandwidth
        // relative to the Nyquist range.
        assert!(bw >= 0.0, "bandwidth must be non-negative");
        assert!(
            bw < (sr as f32) / 8.0,
            "pure-tone bandwidth {bw} unexpectedly large"
        );
    }

    #[test]
    fn test_spectral_centroid_ignores_old_placeholder() {
        // Regression guard: the old stub returned sample_rate/4 regardless of
        // input. A flat (silent) signal must not produce that value.
        let config = EmbeddingConfig::default();
        let extractor = SpeakerEmbeddingExtractor::new(config).unwrap();
        let sr = 16000u32;
        let silence = vec![0.0f32; 2048];
        let centroid = extractor.compute_spectral_centroid(&silence, sr).unwrap();
        assert_eq!(centroid, 0.0, "silence should yield zero centroid");
    }
}
