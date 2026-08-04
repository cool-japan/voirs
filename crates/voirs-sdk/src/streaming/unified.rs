//! Unified streaming interface for all VoiRS advanced features.
//!
//! This module provides a unified streaming interface that can handle multiple
//! advanced features (emotion, cloning, conversion, singing, spatial audio) in
//! a coordinated streaming pipeline.
//!
//! [`UnifiedStreamingPipeline`] does not synthesize audio itself: it is a
//! post-processing coordinator that drives a real [`StreamingPipeline`]
//! (G2P → acoustic model → vocoder) and then runs each resulting audio chunk
//! through the [`FeatureStreamingProcessor`]s configured for the request.
//! Without an attached [`StreamingPipeline`] (see
//! [`UnifiedStreamingPipeline::with_synthesis_pipeline`]),
//! [`UnifiedStreamingSynthesis::start_unified_streaming`] fails closed with
//! [`VoirsError::FeatureUnavailable`] rather than returning a stream that
//! silently yields no audio.

use super::pipeline::StreamingPipeline;
use crate::types::{AdvancedFeature, LanguageCode, SynthesisConfig};
use crate::{AudioBuffer, VoirsError, VoirsResult};
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Unified streaming synthesis request that can handle multiple features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedStreamingRequest {
    /// The text to synthesize
    pub text: String,
    /// Language for synthesis
    pub language: LanguageCode,
    /// Base synthesis configuration
    pub synthesis_config: SynthesisConfig,
    /// Enabled advanced features and their configurations
    pub feature_configs: HashMap<AdvancedFeature, FeatureStreamingConfig>,
    /// Streaming parameters
    pub streaming_params: StreamingParameters,
}

/// Streaming parameters for unified synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingParameters {
    /// Chunk size in milliseconds
    pub chunk_size_ms: u32,
    /// Overlap between chunks in milliseconds
    pub overlap_ms: u32,
    /// Maximum latency tolerance in milliseconds
    pub max_latency_ms: u32,
    /// Quality vs speed preference (0.0 = speed, 1.0 = quality)
    pub quality_preference: f32,
    /// Enable predictive processing
    pub predictive_processing: bool,
}

/// Feature-specific streaming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureStreamingConfig {
    /// Emotion control streaming configuration
    Emotion {
        /// Emotion type (e.g., "happy", "sad")
        emotion_type: String,
        /// Emotion intensity (0.0-1.0)
        intensity: f32,
        /// Enable emotion transitions
        enable_transitions: bool,
        /// Transition duration in milliseconds
        transition_duration_ms: u32,
    },
    /// Voice cloning streaming configuration
    Cloning {
        /// Speaker profile ID
        speaker_id: String,
        /// Adaptation strength (0.0-1.0)
        adaptation_strength: f32,
        /// Enable real-time fine-tuning
        realtime_adaptation: bool,
    },
    /// Voice conversion streaming configuration
    Conversion {
        /// Target voice characteristics
        target_age: Option<u8>,
        target_gender: Option<String>,
        /// Conversion strength (0.0-1.0)
        conversion_strength: f32,
        /// Preserve original prosody
        preserve_prosody: bool,
    },
    /// Singing synthesis streaming configuration
    Singing {
        /// Musical score (simplified format)
        score: Vec<MusicalNote>,
        /// Singing technique
        technique: String,
        /// Vibrato settings
        vibrato_rate: f32,
        vibrato_depth: f32,
    },
    /// Spatial audio streaming configuration
    SpatialAudio {
        /// 3D position (x, y, z)
        position: [f32; 3],
        /// Listener position (x, y, z)
        listener_position: [f32; 3],
        /// Room acoustics model
        room_model: String,
        /// Enable head tracking
        head_tracking: bool,
    },
}

/// Musical note for singing synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicalNote {
    /// Note pitch (e.g., "C4", "A#3")
    pub pitch: String,
    /// Duration in beats
    pub duration: f32,
    /// Start time in beats
    pub start_time: f32,
    /// Lyrics for this note
    pub lyrics: String,
}

/// Streaming synthesis result with feature processing
#[derive(Debug, Clone)]
pub struct UnifiedStreamingResult {
    /// Synthesized audio chunk
    pub audio: AudioBuffer,
    /// Processing metadata
    pub metadata: StreamingMetadata,
    /// Feature-specific results
    pub feature_results: HashMap<AdvancedFeature, FeatureStreamingResult>,
}

/// Metadata for streaming synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingMetadata {
    /// Chunk sequence number
    pub chunk_id: u32,
    /// Timestamp in milliseconds
    pub timestamp_ms: u64,
    /// Processing latency for this chunk
    pub latency_ms: u32,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
}

/// Quality metrics for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Signal-to-noise ratio
    pub snr_db: f32,
    /// Total harmonic distortion
    pub thd_percent: f32,
    /// Spectral centroid
    pub spectral_centroid: f32,
    /// Confidence score (0.0-1.0)
    pub confidence_score: f32,
}

/// Performance metrics for streaming
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Processing time for this chunk in milliseconds
    pub processing_time_ms: u32,
    /// Memory usage in MB
    pub memory_usage_mb: u32,
    /// CPU utilization (0.0-1.0)
    pub cpu_utilization: f32,
    /// GPU utilization (0.0-1.0), if applicable
    pub gpu_utilization: Option<f32>,
}

/// Feature-specific streaming result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureStreamingResult {
    /// Emotion processing result
    Emotion {
        /// Applied emotion intensity
        applied_intensity: f32,
        /// Detected emotion transitions
        transitions: Vec<EmotionTransition>,
    },
    /// Voice cloning result
    Cloning {
        /// Similarity score to target speaker
        similarity_score: f32,
        /// Adaptation quality
        adaptation_quality: f32,
    },
    /// Voice conversion result
    Conversion {
        /// Conversion success rate
        success_rate: f32,
        /// Detected voice characteristics
        detected_characteristics: HashMap<String, f32>,
    },
    /// Singing synthesis result
    Singing {
        /// Pitch accuracy
        pitch_accuracy: f32,
        /// Rhythm accuracy
        rhythm_accuracy: f32,
        /// Musical expression score
        expression_score: f32,
    },
    /// Spatial audio result
    SpatialAudio {
        /// 3D audio positioning accuracy
        positioning_accuracy: f32,
        /// Room acoustics quality
        acoustics_quality: f32,
        /// Binaural rendering quality
        binaural_quality: f32,
    },
}

/// Emotion transition event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionTransition {
    /// Transition start time in milliseconds
    pub start_time_ms: u32,
    /// Transition duration in milliseconds
    pub duration_ms: u32,
    /// Source emotion
    pub from_emotion: String,
    /// Target emotion
    pub to_emotion: String,
    /// Transition curve type
    pub curve_type: String,
}

/// Unified streaming interface trait
#[async_trait]
pub trait UnifiedStreamingSynthesis: Send + Sync {
    /// Start streaming synthesis with unified feature support
    async fn start_unified_streaming(
        &self,
        request: UnifiedStreamingRequest,
    ) -> VoirsResult<Pin<Box<dyn Stream<Item = VoirsResult<UnifiedStreamingResult>> + Send>>>;

    /// Update streaming parameters during synthesis
    async fn update_streaming_parameters(&self, params: StreamingParameters) -> VoirsResult<()>;

    /// Update feature configuration during streaming
    async fn update_feature_config(
        &self,
        feature: AdvancedFeature,
        config: FeatureStreamingConfig,
    ) -> VoirsResult<()>;

    /// Stop streaming synthesis
    async fn stop_streaming(&self) -> VoirsResult<()>;

    /// Get current streaming status
    async fn get_streaming_status(&self) -> VoirsResult<StreamingStatus>;
}

/// Current streaming status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingStatus {
    /// Whether streaming is active
    pub is_active: bool,
    /// Current chunk being processed
    pub current_chunk: u32,
    /// Total chunks processed
    pub total_chunks_processed: u32,
    /// Average processing latency
    pub avg_latency_ms: f32,
    /// Active features
    pub active_features: Vec<AdvancedFeature>,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU usage percentage (0-100)
    pub cpu_percent: f32,
    /// Memory usage in MB
    pub memory_mb: u32,
    /// GPU usage percentage (0-100), if applicable
    pub gpu_percent: Option<f32>,
    /// Network bandwidth usage in Mbps
    pub network_mbps: f32,
}

/// Unified streaming pipeline implementation
pub struct UnifiedStreamingPipeline {
    /// Feature processors, shared (`Arc`) so a request's active subset can be
    /// cloned into the `'static` generator stream returned by
    /// `start_unified_streaming`.
    feature_processors: HashMap<AdvancedFeature, Arc<dyn FeatureStreamingProcessor>>,
    /// Current streaming state, behind a lock so it can be updated from
    /// inside the returned stream (which outlives the `&self` borrow of
    /// `start_unified_streaming`) and observed by `get_streaming_status`.
    streaming_state: Arc<RwLock<Option<StreamingState>>>,
    /// Performance monitor
    performance_monitor: Arc<dyn StreamingPerformanceMonitor>,
    /// Real synthesis backend (G2P + acoustic model + vocoder) used to
    /// actually generate audio for `start_unified_streaming` to post-process.
    /// `None` until [`Self::with_synthesis_pipeline`] is called.
    synthesis_pipeline: Option<Arc<StreamingPipeline>>,
}

/// Internal streaming state
#[derive(Debug)]
struct StreamingState {
    /// Current request configuration
    #[allow(dead_code)]
    // Retained for introspection/debugging even though no field reads it yet.
    request: UnifiedStreamingRequest,
    /// Chunk counter
    chunk_counter: u32,
    /// Start time
    start_time: std::time::Instant,
    /// Performance metrics history
    metrics_history: Vec<PerformanceMetrics>,
    /// Set once the underlying synthesis stream has been fully drained (or
    /// failed). `total_chunks_processed`/`metrics_history` remain queryable
    /// after completion; only `is_active` flips to `false`.
    finished: bool,
}

/// Feature-specific streaming processor trait
#[async_trait]
pub trait FeatureStreamingProcessor: Send + Sync {
    /// Process audio chunk with feature-specific logic
    async fn process_chunk(
        &self,
        audio: &AudioBuffer,
        config: &FeatureStreamingConfig,
        metadata: &StreamingMetadata,
    ) -> VoirsResult<(AudioBuffer, FeatureStreamingResult)>;

    /// Update configuration during streaming
    async fn update_config(&self, config: &FeatureStreamingConfig) -> VoirsResult<()>;

    /// Get processing capabilities
    fn get_capabilities(&self) -> FeatureCapabilities;
}

/// Feature processing capabilities
#[derive(Debug, Clone)]
pub struct FeatureCapabilities {
    /// Supports real-time processing
    pub realtime_capable: bool,
    /// Minimum chunk size in milliseconds
    pub min_chunk_size_ms: u32,
    /// Maximum chunk size in milliseconds
    pub max_chunk_size_ms: u32,
    /// Required overlap in milliseconds
    pub required_overlap_ms: u32,
    /// Latency contribution in milliseconds
    pub latency_contribution_ms: u32,
}

/// Streaming performance monitor trait
#[async_trait]
pub trait StreamingPerformanceMonitor: Send + Sync {
    /// Record performance metrics for a chunk
    async fn record_chunk_metrics(&self, metrics: &PerformanceMetrics);

    /// Get average performance over time window
    async fn get_average_performance(&self, window_seconds: u32)
        -> VoirsResult<PerformanceMetrics>;

    /// Check if performance is within acceptable limits
    async fn check_performance_limits(&self, limits: &PerformanceLimits) -> VoirsResult<bool>;

    /// Get performance recommendations
    async fn get_performance_recommendations(&self) -> Vec<PerformanceRecommendation>;
}

/// Performance limits for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceLimits {
    /// Maximum acceptable latency in milliseconds
    pub max_latency_ms: u32,
    /// Maximum CPU utilization (0.0-1.0)
    pub max_cpu_utilization: f32,
    /// Maximum memory usage in MB
    pub max_memory_mb: u32,
    /// Minimum quality threshold (0.0-1.0)
    pub min_quality_threshold: f32,
}

/// Performance optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Description of the recommendation
    pub description: String,
    /// Expected performance impact
    pub expected_impact: String,
    /// Priority level
    pub priority: RecommendationPriority,
}

/// Types of performance recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Reduce chunk size
    ReduceChunkSize,
    /// Disable non-essential features
    DisableFeatures,
    /// Switch to GPU processing
    UseGpuAcceleration,
    /// Reduce quality settings
    ReduceQuality,
    /// Increase buffer size
    IncreaseBufferSize,
    /// Optimize thread allocation
    OptimizeThreads,
}

/// Priority level for recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecommendationPriority {
    /// Low priority - minor optimization
    Low,
    /// Medium priority - noticeable improvement
    Medium,
    /// High priority - significant performance issue
    High,
    /// Critical priority - system may fail
    Critical,
}

impl Default for StreamingParameters {
    fn default() -> Self {
        Self {
            chunk_size_ms: 100,
            overlap_ms: 10,
            max_latency_ms: 500,
            quality_preference: 0.7,
            predictive_processing: true,
        }
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            snr_db: 0.0,
            thd_percent: 0.0,
            spectral_centroid: 0.0,
            confidence_score: 1.0,
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            processing_time_ms: 0,
            memory_usage_mb: 0,
            cpu_utilization: 0.0,
            gpu_utilization: None,
        }
    }
}

impl Default for UnifiedStreamingPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl UnifiedStreamingPipeline {
    /// Create a new unified streaming pipeline.
    ///
    /// No synthesis backend is attached yet — call
    /// [`Self::with_synthesis_pipeline`] before
    /// [`UnifiedStreamingSynthesis::start_unified_streaming`], or that call
    /// will fail closed with [`VoirsError::FeatureUnavailable`].
    pub fn new() -> Self {
        Self {
            feature_processors: HashMap::new(),
            streaming_state: Arc::new(RwLock::new(None)),
            performance_monitor: Arc::new(DefaultStreamingPerformanceMonitor::new()),
            synthesis_pipeline: None,
        }
    }

    /// Attach the real G2P/acoustic-model/vocoder-backed [`StreamingPipeline`]
    /// that actually produces the audio chunks this pipeline post-processes.
    #[must_use]
    pub fn with_synthesis_pipeline(mut self, pipeline: Arc<StreamingPipeline>) -> Self {
        self.synthesis_pipeline = Some(pipeline);
        self
    }

    /// Add a feature processor to the pipeline
    pub fn add_feature_processor(
        &mut self,
        feature: AdvancedFeature,
        processor: Arc<dyn FeatureStreamingProcessor>,
    ) {
        self.feature_processors.insert(feature, processor);
    }

    /// Remove a feature processor from the pipeline
    pub fn remove_feature_processor(&mut self, feature: &AdvancedFeature) {
        self.feature_processors.remove(feature);
    }

    /// Get supported features
    pub fn supported_features(&self) -> Vec<AdvancedFeature> {
        self.feature_processors.keys().copied().collect()
    }
}

#[async_trait]
impl UnifiedStreamingSynthesis for UnifiedStreamingPipeline {
    async fn start_unified_streaming(
        &self,
        request: UnifiedStreamingRequest,
    ) -> VoirsResult<Pin<Box<dyn Stream<Item = VoirsResult<UnifiedStreamingResult>> + Send>>> {
        // Validate request
        self.validate_request(&request)?;

        let Some(synthesis_pipeline) = self.synthesis_pipeline.clone() else {
            return Err(VoirsError::FeatureUnavailable {
                feature: "unified_streaming".to_string(),
                reason: "no synthesis backend configured; call \
                    UnifiedStreamingPipeline::with_synthesis_pipeline(..) before streaming"
                    .to_string(),
            });
        };

        // Resolve the concrete processors needed for this request up front so the
        // generator below only needs to hold owned, 'static handles (it outlives
        // this `&self` call).
        let active_processors: Vec<(
            AdvancedFeature,
            FeatureStreamingConfig,
            Arc<dyn FeatureStreamingProcessor>,
        )> = request
            .feature_configs
            .iter()
            .filter_map(|(feature, config)| {
                self.feature_processors
                    .get(feature)
                    .map(|processor| (*feature, config.clone(), Arc::clone(processor)))
            })
            .collect();

        let streaming_state = Arc::clone(&self.streaming_state);
        let performance_monitor = Arc::clone(&self.performance_monitor);
        let synthesis_config = request.synthesis_config.clone();
        let text = request.text.clone();

        *streaming_state.write().await = Some(StreamingState {
            request,
            chunk_counter: 0,
            start_time: std::time::Instant::now(),
            metrics_history: Vec::new(),
            finished: false,
        });

        // Real synthesis: drives G2P -> acoustic model -> vocoder over `text`.
        let base_stream = synthesis_pipeline
            .synthesize_stream_with_config(&text, &synthesis_config)
            .await?;

        let stream = async_stream::stream! {
            futures::pin_mut!(base_stream);
            let stream_start = std::time::Instant::now();

            while let Some(chunk_result) = base_stream.next().await {
                let chunk = match chunk_result {
                    Ok(chunk) => chunk,
                    Err(e) => {
                        yield Err(e);
                        continue;
                    }
                };

                let chunk_wall_start = std::time::Instant::now();
                let mut audio = chunk.audio.clone();
                let mut feature_results = HashMap::new();

                // Real per-chunk memory footprint of the audio buffer itself
                // (varies with actual chunk size - never a fabricated constant).
                let audio_bytes_mb = std::mem::size_of_val(audio.samples()) as u32 / (1024 * 1024);

                let mut metadata = StreamingMetadata {
                    chunk_id: chunk.chunk_id as u32,
                    timestamp_ms: stream_start.elapsed().as_millis() as u64,
                    latency_ms: chunk.processing_time.as_millis() as u32,
                    quality_metrics: QualityMetrics {
                        snr_db: 0.0,
                        thd_percent: 0.0,
                        spectral_centroid: 0.0,
                        // Real, per-chunk confidence derived from actual synthesis
                        // timing/phoneme-density heuristics (AudioChunk::new).
                        confidence_score: chunk.metadata.confidence_score,
                    },
                    performance_metrics: PerformanceMetrics {
                        processing_time_ms: chunk.processing_time.as_millis() as u32,
                        memory_usage_mb: audio_bytes_mb,
                        cpu_utilization: 0.0,
                        gpu_utilization: None,
                    },
                };

                let mut processing_failed = false;
                for (feature, config, processor) in &active_processors {
                    match processor.process_chunk(&audio, config, &metadata).await {
                        Ok((processed_audio, result)) => {
                            audio = processed_audio;
                            feature_results.insert(*feature, result);
                        }
                        Err(e) => {
                            yield Err(e);
                            processing_failed = true;
                            break;
                        }
                    }
                }
                if processing_failed {
                    continue;
                }

                // Fold in the real wall-clock time spent running feature processors
                // on top of the base synthesis latency already recorded above.
                metadata.latency_ms = metadata
                    .latency_ms
                    .saturating_add(chunk_wall_start.elapsed().as_millis() as u32);

                performance_monitor
                    .record_chunk_metrics(&metadata.performance_metrics)
                    .await;

                {
                    let mut state_guard = streaming_state.write().await;
                    if let Some(state) = state_guard.as_mut() {
                        state.chunk_counter += 1;
                        state.metrics_history.push(metadata.performance_metrics.clone());
                    }
                }

                yield Ok(UnifiedStreamingResult {
                    audio,
                    metadata,
                    feature_results,
                });
            }

            // The base stream is exhausted (or errored out permanently): mark the
            // session finished so `get_streaming_status` honestly reports
            // `is_active: false` while still exposing the real final counters.
            if let Some(state) = streaming_state.write().await.as_mut() {
                state.finished = true;
            }
        };

        Ok(Box::pin(stream))
    }

    async fn update_streaming_parameters(&self, params: StreamingParameters) -> VoirsResult<()> {
        let mut state_guard = self.streaming_state.write().await;
        match state_guard.as_mut() {
            Some(state) => {
                state.request.streaming_params = params;
                Ok(())
            }
            None => Err(VoirsError::InvalidStateTransition {
                from: "idle".to_string(),
                to: "streaming".to_string(),
                reason: "no active streaming session to update parameters for".to_string(),
            }),
        }
    }

    async fn update_feature_config(
        &self,
        feature: AdvancedFeature,
        config: FeatureStreamingConfig,
    ) -> VoirsResult<()> {
        if let Some(processor) = self.feature_processors.get(&feature) {
            processor.update_config(&config).await?;
        }
        Ok(())
    }

    async fn stop_streaming(&self) -> VoirsResult<()> {
        *self.streaming_state.write().await = None;
        Ok(())
    }

    async fn get_streaming_status(&self) -> VoirsResult<StreamingStatus> {
        let state_guard = self.streaming_state.read().await;
        let (is_active, current_chunk, total_chunks_processed, avg_latency_ms) =
            match state_guard.as_ref() {
                Some(state) => {
                    let avg_latency_ms = if state.metrics_history.is_empty() {
                        0.0
                    } else {
                        state
                            .metrics_history
                            .iter()
                            .map(|m| m.processing_time_ms as f32)
                            .sum::<f32>()
                            / state.metrics_history.len() as f32
                    };
                    (
                        !state.finished,
                        state.chunk_counter,
                        state.chunk_counter,
                        avg_latency_ms,
                    )
                }
                None => (false, 0, 0, 0.0),
            };
        drop(state_guard);

        let resource_utilization = self
            .performance_monitor
            .get_average_performance(60)
            .await
            .map(|perf| ResourceUtilization {
                cpu_percent: perf.cpu_utilization * 100.0,
                memory_mb: perf.memory_usage_mb,
                gpu_percent: perf.gpu_utilization.map(|g| g * 100.0),
                network_mbps: 0.0,
            })
            .unwrap_or(ResourceUtilization {
                cpu_percent: 0.0,
                memory_mb: 0,
                gpu_percent: None,
                network_mbps: 0.0,
            });

        Ok(StreamingStatus {
            is_active,
            current_chunk,
            total_chunks_processed,
            avg_latency_ms,
            active_features: self.supported_features(),
            resource_utilization,
        })
    }
}

impl UnifiedStreamingPipeline {
    /// Validate streaming request
    fn validate_request(&self, request: &UnifiedStreamingRequest) -> VoirsResult<()> {
        // Check if requested features are supported
        for feature in request.feature_configs.keys() {
            if !self.feature_processors.contains_key(feature) {
                return Err(VoirsError::FeatureUnavailable {
                    feature: format!("{:?}", feature),
                    reason: "Feature processor not available".to_string(),
                });
            }
        }

        // Validate streaming parameters
        if request.streaming_params.chunk_size_ms == 0 {
            return Err(VoirsError::InvalidConfiguration {
                field: "chunk_size_ms".to_string(),
                value: "0".to_string(),
                reason: "Chunk size must be greater than zero".to_string(),
                valid_values: Some(vec!["1".to_string(), "100".to_string(), "1000".to_string()]),
            });
        }

        Ok(())
    }
}

/// Default performance monitor implementation
///
/// Every method here computes real results from `metrics_history` (populated
/// by genuine [`Self::record_chunk_metrics`] calls during streaming) — never a
/// fabricated constant.
struct DefaultStreamingPerformanceMonitor {
    metrics_history:
        std::sync::Arc<std::sync::Mutex<Vec<(std::time::Instant, PerformanceMetrics)>>>,
}

impl DefaultStreamingPerformanceMonitor {
    fn new() -> Self {
        Self {
            metrics_history: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl StreamingPerformanceMonitor for DefaultStreamingPerformanceMonitor {
    async fn record_chunk_metrics(&self, metrics: &PerformanceMetrics) {
        if let Ok(mut history) = self.metrics_history.lock() {
            history.push((std::time::Instant::now(), metrics.clone()));
            // Keep only last 1000 metrics
            if history.len() > 1000 {
                history.remove(0);
            }
        }
    }

    async fn get_average_performance(
        &self,
        window_seconds: u32,
    ) -> VoirsResult<PerformanceMetrics> {
        let history = self
            .metrics_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let cutoff = std::time::Duration::from_secs(window_seconds as u64);
        let now = std::time::Instant::now();
        let windowed: Vec<&PerformanceMetrics> = history
            .iter()
            .filter(|(recorded_at, _)| now.duration_since(*recorded_at) <= cutoff)
            .map(|(_, metrics)| metrics)
            .collect();

        if windowed.is_empty() {
            return Ok(PerformanceMetrics::default());
        }

        let count = windowed.len() as f32;
        let processing_time_ms = (windowed
            .iter()
            .map(|m| m.processing_time_ms as f32)
            .sum::<f32>()
            / count) as u32;
        let memory_usage_mb = (windowed
            .iter()
            .map(|m| m.memory_usage_mb as f32)
            .sum::<f32>()
            / count) as u32;
        let cpu_utilization = windowed.iter().map(|m| m.cpu_utilization).sum::<f32>() / count;
        let gpu_samples: Vec<f32> = windowed.iter().filter_map(|m| m.gpu_utilization).collect();
        let gpu_utilization = if gpu_samples.is_empty() {
            None
        } else {
            Some(gpu_samples.iter().sum::<f32>() / gpu_samples.len() as f32)
        };

        Ok(PerformanceMetrics {
            processing_time_ms,
            memory_usage_mb,
            cpu_utilization,
            gpu_utilization,
        })
    }

    async fn check_performance_limits(&self, limits: &PerformanceLimits) -> VoirsResult<bool> {
        let history = self
            .metrics_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let within_limits = history.iter().all(|(_, m)| {
            m.processing_time_ms <= limits.max_latency_ms
                && m.cpu_utilization <= limits.max_cpu_utilization
                && m.memory_usage_mb <= limits.max_memory_mb
        });
        Ok(within_limits)
    }

    async fn get_performance_recommendations(&self) -> Vec<PerformanceRecommendation> {
        let history = self
            .metrics_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        if history.is_empty() {
            return Vec::new();
        }

        let count = history.len() as f32;
        let avg_latency_ms = history
            .iter()
            .map(|(_, m)| m.processing_time_ms as f32)
            .sum::<f32>()
            / count;
        let avg_cpu = history.iter().map(|(_, m)| m.cpu_utilization).sum::<f32>() / count;

        let mut recommendations = Vec::new();
        if avg_latency_ms > 200.0 {
            recommendations.push(PerformanceRecommendation {
                recommendation_type: RecommendationType::ReduceChunkSize,
                description: format!(
                    "Average chunk processing latency is {avg_latency_ms:.1}ms across {} \
                     recorded chunks; smaller chunks may reduce end-to-end latency",
                    history.len()
                ),
                expected_impact: "Lower per-chunk latency".to_string(),
                priority: RecommendationPriority::Medium,
            });
        }
        if avg_cpu > 0.9 {
            recommendations.push(PerformanceRecommendation {
                recommendation_type: RecommendationType::UseGpuAcceleration,
                description: format!(
                    "Average CPU utilization is {:.0}%; consider GPU acceleration or disabling \
                     non-essential features",
                    avg_cpu * 100.0
                ),
                expected_impact: "Reduced CPU load".to_string(),
                priority: RecommendationPriority::High,
            });
        }
        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{DummyAcoustic, DummyG2p, DummyVocoder};
    use crate::streaming::StreamingConfig;

    fn test_synthesis_pipeline() -> Arc<StreamingPipeline> {
        Arc::new(StreamingPipeline::new(
            Arc::new(DummyG2p::new()),
            Arc::new(DummyAcoustic::new()),
            Arc::new(DummyVocoder::new()),
            StreamingConfig::default(),
        ))
    }

    fn test_request(text: &str) -> UnifiedStreamingRequest {
        UnifiedStreamingRequest {
            text: text.to_string(),
            language: LanguageCode::EnUs,
            synthesis_config: SynthesisConfig::default(),
            feature_configs: HashMap::new(),
            streaming_params: StreamingParameters::default(),
        }
    }

    #[test]
    fn test_unified_streaming_request_creation() {
        let request = test_request("Hello, world!");

        assert_eq!(request.text, "Hello, world!");
        assert_eq!(request.language, LanguageCode::EnUs);
    }

    #[test]
    fn test_streaming_parameters_default() {
        let params = StreamingParameters::default();
        assert_eq!(params.chunk_size_ms, 100);
        assert_eq!(params.overlap_ms, 10);
        assert_eq!(params.max_latency_ms, 500);
        assert_eq!(params.quality_preference, 0.7);
        assert!(params.predictive_processing);
    }

    #[test]
    fn test_feature_streaming_config_emotion() {
        let config = FeatureStreamingConfig::Emotion {
            emotion_type: "happy".to_string(),
            intensity: 0.8,
            enable_transitions: true,
            transition_duration_ms: 200,
        };

        match config {
            FeatureStreamingConfig::Emotion {
                emotion_type,
                intensity,
                ..
            } => {
                assert_eq!(emotion_type, "happy");
                assert_eq!(intensity, 0.8);
            }
            _ => panic!("Expected emotion config"),
        }
    }

    #[test]
    fn test_unified_streaming_pipeline_creation() {
        let pipeline = UnifiedStreamingPipeline::new();
        assert!(pipeline.supported_features().is_empty());
    }

    #[tokio::test]
    async fn test_start_unified_streaming_without_backend_fails_closed() {
        // Direct regression test for the fabrication bug: without a synthesis
        // backend attached, the old implementation returned `Ok` with a stream
        // that silently yielded zero chunks. It must now fail closed instead.
        let pipeline = UnifiedStreamingPipeline::new();
        let result = pipeline
            .start_unified_streaming(test_request("Hello"))
            .await;
        match result {
            Err(VoirsError::FeatureUnavailable { .. }) => {}
            Err(other) => panic!("expected FeatureUnavailable, got {other:?}"),
            Ok(_) => panic!("expected an error without a configured synthesis backend"),
        }
    }

    #[tokio::test]
    async fn test_start_unified_streaming_produces_real_nonempty_audio() {
        let pipeline =
            UnifiedStreamingPipeline::new().with_synthesis_pipeline(test_synthesis_pipeline());

        let mut stream = pipeline
            .start_unified_streaming(test_request("Hello there, this is real synthesis."))
            .await
            .unwrap();

        let mut chunk_count = 0;
        let mut total_samples = 0usize;
        while let Some(result) = stream.next().await {
            let result = result.unwrap();
            total_samples += result.audio.samples().len();
            chunk_count += 1;
        }

        assert!(chunk_count > 0, "expected at least one real audio chunk");
        assert!(total_samples > 0, "expected non-empty synthesized audio");
    }

    #[tokio::test]
    async fn test_streaming_status_reflects_real_activity_then_completion() {
        let pipeline =
            UnifiedStreamingPipeline::new().with_synthesis_pipeline(test_synthesis_pipeline());

        // Before streaming starts, honestly idle.
        let idle_status = pipeline.get_streaming_status().await.unwrap();
        assert!(!idle_status.is_active);
        assert_eq!(idle_status.total_chunks_processed, 0);

        let mut stream = pipeline
            .start_unified_streaming(test_request(
                "This sentence is long enough to span multiple streaming chunks for a real test.",
            ))
            .await
            .unwrap();

        // Mid-stream: consume exactly one chunk, then check status without
        // draining the rest.
        let first = stream.next().await;
        assert!(first.is_some());
        let mid_status = pipeline.get_streaming_status().await.unwrap();
        assert!(mid_status.is_active, "session must be active mid-stream");
        assert!(mid_status.total_chunks_processed >= 1);

        // Drain the rest.
        let mut total_chunks = 1u32;
        while let Some(result) = stream.next().await {
            result.unwrap();
            total_chunks += 1;
        }

        let final_status = pipeline.get_streaming_status().await.unwrap();
        assert!(
            !final_status.is_active,
            "session must be inactive once the stream is exhausted"
        );
        assert_eq!(final_status.total_chunks_processed, total_chunks);
        assert!(
            final_status.avg_latency_ms >= 0.0,
            "average latency must be a real (non-negative) measurement"
        );
    }

    #[tokio::test]
    async fn test_get_average_performance_reflects_recorded_history_not_default() {
        // Direct regression test: the old implementation ignored history entirely
        // and always returned `PerformanceMetrics::default()`.
        let monitor = DefaultStreamingPerformanceMonitor::new();

        monitor
            .record_chunk_metrics(&PerformanceMetrics {
                processing_time_ms: 100,
                memory_usage_mb: 10,
                cpu_utilization: 0.5,
                gpu_utilization: None,
            })
            .await;
        monitor
            .record_chunk_metrics(&PerformanceMetrics {
                processing_time_ms: 300,
                memory_usage_mb: 30,
                cpu_utilization: 0.9,
                gpu_utilization: None,
            })
            .await;

        let avg = monitor.get_average_performance(3600).await.unwrap();
        assert_eq!(avg.processing_time_ms, 200); // real mean of 100 and 300
        assert_eq!(avg.memory_usage_mb, 20);
        assert!((avg.cpu_utilization - 0.7).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_get_average_performance_window_excludes_old_samples() {
        let monitor = DefaultStreamingPerformanceMonitor::new();
        monitor
            .record_chunk_metrics(&PerformanceMetrics {
                processing_time_ms: 999,
                memory_usage_mb: 999,
                cpu_utilization: 1.0,
                gpu_utilization: None,
            })
            .await;

        // A zero-second window excludes every sample recorded strictly before "now".
        let avg = monitor.get_average_performance(0).await.unwrap();
        assert_eq!(avg, PerformanceMetrics::default());
    }

    #[tokio::test]
    async fn test_check_performance_limits_detects_real_violation() {
        // Direct regression test: the old implementation always returned `Ok(true)`
        // regardless of the limits argument.
        let monitor = DefaultStreamingPerformanceMonitor::new();
        monitor
            .record_chunk_metrics(&PerformanceMetrics {
                processing_time_ms: 500,
                memory_usage_mb: 50,
                cpu_utilization: 0.95,
                gpu_utilization: None,
            })
            .await;

        let generous_limits = PerformanceLimits {
            max_latency_ms: 1000,
            max_cpu_utilization: 1.0,
            max_memory_mb: 100,
            min_quality_threshold: 0.0,
        };
        assert!(monitor
            .check_performance_limits(&generous_limits)
            .await
            .unwrap());

        let strict_limits = PerformanceLimits {
            max_latency_ms: 10,
            max_cpu_utilization: 0.1,
            max_memory_mb: 1,
            min_quality_threshold: 0.0,
        };
        assert!(!monitor
            .check_performance_limits(&strict_limits)
            .await
            .unwrap());
    }
}
