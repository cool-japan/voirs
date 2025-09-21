//! Streaming audio evaluation for real-time quality assessment
//!
//! This module provides capabilities for evaluating audio quality in real-time
//! streaming scenarios, enabling low-latency assessment of ongoing audio streams.

use crate::traits::{EvaluationResult, QualityScore};
use crate::EvaluationError;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::task;
use voirs_sdk::AudioBuffer;

/// Configuration for streaming audio evaluation
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Size of each audio chunk in samples
    pub chunk_size: usize,
    /// Overlap between chunks (in samples)
    pub overlap_size: usize,
    /// Maximum buffer size (in chunks)
    pub max_buffer_chunks: usize,
    /// Target latency for evaluation (in milliseconds)
    pub target_latency_ms: u64,
    /// Enable real-time quality monitoring
    pub enable_quality_monitoring: bool,
    /// Enable automatic adaptation based on performance
    pub enable_adaptive_processing: bool,
    /// Enable advanced quality prediction
    pub enable_quality_prediction: bool,
    /// Enable network-aware adaptation
    pub enable_network_adaptation: bool,
    /// Enable multi-threaded processing
    pub enable_parallel_processing: bool,
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
    /// Quality prediction horizon (in chunks)
    pub prediction_horizon: usize,
    /// Network condition monitoring interval (in milliseconds)
    pub network_monitor_interval_ms: u64,
    /// Anomaly detection sensitivity (0.0 to 1.0)
    pub anomaly_sensitivity: f32,
    /// Enable predictive buffer management
    pub enable_predictive_buffering: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1024,       // ~64ms at 16kHz
            overlap_size: 256,      // 25% overlap
            max_buffer_chunks: 10,  // ~640ms buffer
            target_latency_ms: 100, // 100ms target latency
            enable_quality_monitoring: true,
            enable_adaptive_processing: true,
            enable_quality_prediction: true,
            enable_network_adaptation: false, // Disabled by default
            enable_parallel_processing: true,
            enable_anomaly_detection: true,
            prediction_horizon: 5,             // Predict 5 chunks ahead
            network_monitor_interval_ms: 1000, // Monitor every second
            anomaly_sensitivity: 0.7,          // Moderate sensitivity
            enable_predictive_buffering: true,
        }
    }
}

/// Audio chunk for streaming processing
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Audio samples
    pub samples: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Timestamp when chunk was created
    pub timestamp: Instant,
    /// Sequence number
    pub sequence: u64,
}

impl AudioChunk {
    /// Create a new audio chunk
    pub fn new(samples: Vec<f32>, sample_rate: u32, sequence: u64) -> Self {
        Self {
            samples,
            sample_rate,
            timestamp: Instant::now(),
            sequence,
        }
    }

    /// Get duration of this chunk in seconds
    pub fn duration(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }

    /// Convert to AudioBuffer
    pub fn to_audio_buffer(&self) -> AudioBuffer {
        AudioBuffer::mono(self.samples.clone(), self.sample_rate)
    }
}

/// Real-time quality metrics for streaming audio
#[derive(Debug, Clone)]
pub struct StreamingQualityMetrics {
    /// Current signal-to-noise ratio estimate
    pub snr_estimate: f32,
    /// Dynamic range estimate
    pub dynamic_range: f32,
    /// Spectral flatness
    pub spectral_flatness: f32,
    /// Energy level
    pub energy_level: f32,
    /// Clipping detection
    pub clipping_detected: bool,
    /// Silence detection
    pub silence_detected: bool,
    /// Processing latency in milliseconds
    pub processing_latency_ms: f64,
    /// Timestamp of measurement
    pub timestamp: Instant,
}

impl Default for StreamingQualityMetrics {
    fn default() -> Self {
        Self {
            snr_estimate: 0.0,
            dynamic_range: 0.0,
            spectral_flatness: 0.0,
            energy_level: 0.0,
            clipping_detected: false,
            silence_detected: false,
            processing_latency_ms: 0.0,
            timestamp: Instant::now(),
        }
    }
}

/// Advanced quality prediction results
#[derive(Debug, Clone)]
pub struct QualityPrediction {
    /// Predicted quality score for next chunk
    pub predicted_score: f32,
    /// Confidence in prediction (0.0 to 1.0)
    pub confidence: f32,
    /// Predicted trend (positive = improving, negative = degrading)
    pub trend: f32,
    /// Risk assessment for quality drops
    pub risk_level: RiskLevel,
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Risk level for quality prediction
#[derive(Debug, Clone, PartialEq)]
pub enum RiskLevel {
    /// Low risk - quality is stable
    Low,
    /// Medium risk - some concerns but manageable
    Medium,
    /// High risk - significant quality issues likely
    High,
    /// Critical risk - immediate action required
    Critical,
}

/// Network condition information
#[derive(Debug, Clone)]
pub struct NetworkCondition {
    /// Estimated bandwidth (bits per second)
    pub bandwidth_estimate: u64,
    /// Round-trip time in milliseconds
    pub rtt_ms: f64,
    /// Packet loss rate (0.0 to 1.0)
    pub packet_loss_rate: f32,
    /// Jitter in milliseconds
    pub jitter_ms: f64,
    /// Network quality score (0.0 to 1.0)
    pub quality_score: f32,
    /// Timestamp of measurement
    pub timestamp: Instant,
}

impl Default for NetworkCondition {
    fn default() -> Self {
        Self {
            bandwidth_estimate: 1_000_000, // 1 Mbps default
            rtt_ms: 50.0,
            packet_loss_rate: 0.0,
            jitter_ms: 5.0,
            quality_score: 1.0,
            timestamp: Instant::now(),
        }
    }
}

/// Anomaly detection result
#[derive(Debug, Clone)]
pub struct AnomalyDetection {
    /// Whether an anomaly was detected
    pub anomaly_detected: bool,
    /// Anomaly score (higher = more anomalous)
    pub anomaly_score: f32,
    /// Type of anomaly detected
    pub anomaly_type: AnomalyType,
    /// Description of the anomaly
    pub description: String,
    /// Severity level
    pub severity: AnomalySeverity,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
}

/// Types of audio quality anomalies
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyType {
    /// No anomaly detected
    None,
    /// Sudden drop in audio quality
    SuddenQualityDrop,
    /// Unexpected silence in audio stream
    UnexpectedSilence,
    /// Audio clipping or saturation detected
    ExcessiveClipping,
    /// Frequency spectrum imbalance
    FrequencyImbalance,
    /// Processing latency exceeds thresholds
    ProcessingDelay,
    /// Network-related quality issues
    NetworkIssue,
    /// Unknown type of anomaly
    Unknown,
}

/// Severity levels for anomalies
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalySeverity {
    /// Informational - minor deviation from normal
    Info,
    /// Warning - notable deviation requiring attention
    Warning,
    /// Error - significant issue that impacts quality
    Error,
    /// Critical - severe issue requiring immediate action
    Critical,
}

/// Advanced processing statistics
#[derive(Debug, Clone)]
pub struct AdvancedProcessingStats {
    /// Basic processing statistics
    pub basic_stats: ProcessingStats,
    /// Quality prediction accuracy
    pub prediction_accuracy: f32,
    /// Network adaptation efficiency
    pub network_adaptation_efficiency: f32,
    /// Anomaly detection rate
    pub anomaly_detection_rate: f32,
    /// Parallel processing efficiency
    pub parallel_efficiency: f32,
    /// Buffer utilization statistics
    pub buffer_utilization: BufferUtilization,
}

/// Buffer utilization statistics
#[derive(Debug, Clone)]
pub struct BufferUtilization {
    /// Average buffer fill percentage
    pub average_fill_percentage: f32,
    /// Peak buffer usage
    pub peak_usage: f32,
    /// Buffer underruns count
    pub underruns: u64,
    /// Buffer overruns count
    pub overruns: u64,
    /// Adaptive adjustments made
    pub adaptive_adjustments: u64,
}

/// Streaming audio evaluator for real-time quality assessment
pub struct StreamingEvaluator {
    /// Configuration
    config: StreamingConfig,
    /// Circular buffer for audio chunks
    buffer: Arc<Mutex<VecDeque<AudioChunk>>>,
    /// Current sequence number
    sequence: Arc<Mutex<u64>>,
    /// Quality metrics history
    metrics_history: Arc<Mutex<VecDeque<StreamingQualityMetrics>>>,
    /// Channel for sending quality updates
    quality_sender: Option<mpsc::UnboundedSender<StreamingQualityMetrics>>,
    /// Processing statistics
    processing_stats: Arc<Mutex<ProcessingStats>>,
    /// Quality prediction engine
    quality_predictor: Arc<Mutex<QualityPredictor>>,
    /// Network condition monitor
    network_monitor: Arc<Mutex<NetworkMonitor>>,
    /// Anomaly detection engine
    anomaly_detector: Arc<Mutex<AnomalyDetector>>,
    /// Advanced processing statistics
    advanced_stats: Arc<Mutex<AdvancedProcessingStats>>,
    /// Parallel processing pool
    parallel_pool: Option<Arc<tokio::task::JoinSet<()>>>,
}

/// Processing statistics for performance monitoring
#[derive(Debug, Default, Clone)]
pub struct ProcessingStats {
    /// Total number of audio chunks processed
    pub total_chunks_processed: u64,
    /// Total time spent processing chunks in milliseconds
    pub total_processing_time_ms: f64,
    /// Average processing latency in milliseconds
    pub average_latency_ms: f64,
    /// Peak processing latency in milliseconds
    pub peak_latency_ms: f64,
    /// Number of chunks dropped due to buffer overflow
    pub dropped_chunks: u64,
}

/// Quality prediction engine
#[derive(Debug)]
struct QualityPredictor {
    /// Historical quality scores
    quality_history: VecDeque<f32>,
    /// Prediction models (simple moving averages and trends)
    prediction_models: HashMap<String, PredictionModel>,
    /// Prediction accuracy tracking
    prediction_accuracy: f32,
}

/// Simple prediction model
#[derive(Debug)]
struct PredictionModel {
    /// Model type identifier
    model_type: String,
    /// Model parameters
    parameters: Vec<f32>,
    /// Recent predictions vs actual values
    accuracy_history: VecDeque<f32>,
}

/// Network condition monitor
#[derive(Debug)]
struct NetworkMonitor {
    /// Current network condition
    current_condition: NetworkCondition,
    /// Network history
    condition_history: VecDeque<NetworkCondition>,
    /// Last monitoring time
    last_check: Instant,
}

/// Anomaly detection engine
#[derive(Debug)]
struct AnomalyDetector {
    /// Baseline quality metrics for comparison
    baseline_metrics: StreamingQualityMetrics,
    /// Recent anomaly detections
    anomaly_history: VecDeque<AnomalyDetection>,
    /// Detection thresholds
    thresholds: AnomalyThresholds,
}

/// Thresholds for anomaly detection
#[derive(Debug)]
struct AnomalyThresholds {
    quality_drop_threshold: f32,
    silence_duration_threshold: Duration,
    clipping_rate_threshold: f32,
    latency_threshold: f64,
    energy_deviation_threshold: f32,
}

impl StreamingEvaluator {
    /// Create a new streaming evaluator
    pub fn new(config: StreamingConfig) -> Self {
        let quality_predictor = QualityPredictor {
            quality_history: VecDeque::new(),
            prediction_models: {
                let mut models = HashMap::new();
                models.insert(
                    "moving_average".to_string(),
                    PredictionModel {
                        model_type: "moving_average".to_string(),
                        parameters: vec![0.2, 0.8], // Short and long-term weights
                        accuracy_history: VecDeque::new(),
                    },
                );
                models.insert(
                    "trend_analysis".to_string(),
                    PredictionModel {
                        model_type: "trend_analysis".to_string(),
                        parameters: vec![0.1, 0.3, 0.6], // Trend weights
                        accuracy_history: VecDeque::new(),
                    },
                );
                models
            },
            prediction_accuracy: 0.5,
        };

        let network_monitor = NetworkMonitor {
            current_condition: NetworkCondition::default(),
            condition_history: VecDeque::new(),
            last_check: Instant::now(),
        };

        let anomaly_detector = AnomalyDetector {
            baseline_metrics: StreamingQualityMetrics::default(),
            anomaly_history: VecDeque::new(),
            thresholds: AnomalyThresholds {
                quality_drop_threshold: 0.3,
                silence_duration_threshold: Duration::from_secs(2),
                clipping_rate_threshold: 0.1,
                latency_threshold: 200.0, // 200ms
                energy_deviation_threshold: 0.5,
            },
        };

        let advanced_stats = AdvancedProcessingStats {
            basic_stats: ProcessingStats::default(),
            prediction_accuracy: 0.5,
            network_adaptation_efficiency: 1.0,
            anomaly_detection_rate: 0.0,
            parallel_efficiency: 1.0,
            buffer_utilization: BufferUtilization {
                average_fill_percentage: 0.0,
                peak_usage: 0.0,
                underruns: 0,
                overruns: 0,
                adaptive_adjustments: 0,
            },
        };

        Self {
            config,
            buffer: Arc::new(Mutex::new(VecDeque::new())),
            sequence: Arc::new(Mutex::new(0)),
            metrics_history: Arc::new(Mutex::new(VecDeque::new())),
            quality_sender: None,
            processing_stats: Arc::new(Mutex::new(ProcessingStats::default())),
            quality_predictor: Arc::new(Mutex::new(quality_predictor)),
            network_monitor: Arc::new(Mutex::new(network_monitor)),
            anomaly_detector: Arc::new(Mutex::new(anomaly_detector)),
            advanced_stats: Arc::new(Mutex::new(advanced_stats)),
            parallel_pool: None,
        }
    }

    /// Set up quality monitoring channel
    pub fn setup_quality_monitoring(&mut self) -> mpsc::UnboundedReceiver<StreamingQualityMetrics> {
        let (sender, receiver) = mpsc::unbounded_channel();
        self.quality_sender = Some(sender);
        receiver
    }

    /// Process an incoming audio chunk
    pub async fn process_chunk(&mut self, chunk: AudioChunk) -> EvaluationResult<()> {
        let start_time = Instant::now();

        // Add chunk to buffer
        {
            let mut buffer = self
                .buffer
                .lock()
                .map_err(|_| EvaluationError::ProcessingError {
                    message: "Failed to lock buffer".to_string(),
                    source: None,
                })?;

            buffer.push_back(chunk.clone());

            // Remove old chunks if buffer is too large
            while buffer.len() > self.config.max_buffer_chunks {
                buffer.pop_front();
                let mut stats =
                    self.processing_stats
                        .lock()
                        .map_err(|_| EvaluationError::ProcessingError {
                            message: "Failed to lock processing stats".to_string(),
                            source: None,
                        })?;
                stats.dropped_chunks += 1;
            }
        }

        // Calculate quality metrics for this chunk
        if self.config.enable_quality_monitoring {
            let metrics = self.calculate_chunk_quality_metrics(&chunk).await?;
            let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;

            // Update processing statistics
            {
                let mut stats =
                    self.processing_stats
                        .lock()
                        .map_err(|_| EvaluationError::ProcessingError {
                            message: "Failed to lock processing stats".to_string(),
                            source: None,
                        })?;

                stats.total_chunks_processed += 1;
                stats.total_processing_time_ms += processing_time;
                stats.average_latency_ms =
                    stats.total_processing_time_ms / stats.total_chunks_processed as f64;

                if processing_time > stats.peak_latency_ms {
                    stats.peak_latency_ms = processing_time;
                }
            }

            // Store metrics
            {
                let mut history =
                    self.metrics_history
                        .lock()
                        .map_err(|_| EvaluationError::ProcessingError {
                            message: "Failed to lock metrics history".to_string(),
                            source: None,
                        })?;

                history.push_back(metrics.clone());

                // Keep only recent metrics
                while history.len() > 100 {
                    history.pop_front();
                }
            }

            // Send quality update if monitoring is enabled
            if let Some(ref sender) = self.quality_sender {
                let mut updated_metrics = metrics;
                updated_metrics.processing_latency_ms = processing_time;
                let _ = sender.send(updated_metrics);
            }
        }

        // Adaptive processing adjustment
        if self.config.enable_adaptive_processing {
            self.adapt_processing_parameters().await?;
        }

        Ok(())
    }

    /// Calculate quality metrics for a single chunk
    async fn calculate_chunk_quality_metrics(
        &self,
        chunk: &AudioChunk,
    ) -> EvaluationResult<StreamingQualityMetrics> {
        let samples = &chunk.samples;

        if samples.is_empty() {
            return Ok(StreamingQualityMetrics::default());
        }

        // Energy level calculation
        let energy_level = samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32;
        let energy_level = energy_level.sqrt();

        // Dynamic range calculation
        let max_val = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let rms = energy_level;
        let dynamic_range = if rms > 0.0 {
            20.0 * (max_val / rms).log10()
        } else {
            0.0
        };

        // Clipping detection
        let clipping_threshold = 0.95;
        let clipping_detected = samples.iter().any(|&x| x.abs() > clipping_threshold);

        // Silence detection
        let silence_threshold = 0.001;
        let silence_detected = energy_level < silence_threshold;

        // Simple SNR estimate (assumes noise floor)
        let noise_floor = 0.01;
        let snr_estimate = if energy_level > noise_floor {
            20.0 * (energy_level / noise_floor).log10()
        } else {
            0.0
        };

        // Spectral flatness (simplified calculation)
        let spectral_flatness = self.calculate_spectral_flatness(samples);

        Ok(StreamingQualityMetrics {
            snr_estimate,
            dynamic_range,
            spectral_flatness,
            energy_level,
            clipping_detected,
            silence_detected,
            processing_latency_ms: 0.0, // Will be set by caller
            timestamp: Instant::now(),
        })
    }

    /// Calculate spectral flatness for a chunk
    fn calculate_spectral_flatness(&self, samples: &[f32]) -> f32 {
        if samples.len() < 32 {
            return 0.5; // Default value for short segments
        }

        // Simple spectral flatness using windowed energy
        let window_size = 32;
        let num_windows = samples.len() / window_size;

        if num_windows < 2 {
            return 0.5;
        }

        let mut window_energies = Vec::new();
        for i in 0..num_windows {
            let start = i * window_size;
            let end = (start + window_size).min(samples.len());
            let energy: f32 = samples[start..end].iter().map(|&x| x * x).sum();
            window_energies.push(energy.max(1e-10)); // Avoid log(0)
        }

        // Geometric mean
        let geometric_mean =
            window_energies.iter().map(|&x| x.ln()).sum::<f32>() / window_energies.len() as f32;
        let geometric_mean = geometric_mean.exp();

        // Arithmetic mean
        let arithmetic_mean = window_energies.iter().sum::<f32>() / window_energies.len() as f32;

        // Spectral flatness
        if arithmetic_mean > 0.0 {
            (geometric_mean / arithmetic_mean).min(1.0)
        } else {
            0.0
        }
    }

    /// Adapt processing parameters based on performance
    async fn adapt_processing_parameters(&mut self) -> EvaluationResult<()> {
        let stats = {
            let stats_guard =
                self.processing_stats
                    .lock()
                    .map_err(|_| EvaluationError::ProcessingError {
                        message: "Failed to lock processing stats".to_string(),
                        source: None,
                    })?;
            ProcessingStats {
                total_chunks_processed: stats_guard.total_chunks_processed,
                total_processing_time_ms: stats_guard.total_processing_time_ms,
                average_latency_ms: stats_guard.average_latency_ms,
                peak_latency_ms: stats_guard.peak_latency_ms,
                dropped_chunks: stats_guard.dropped_chunks,
            }
        };

        // Adapt if latency is too high
        if stats.average_latency_ms > self.config.target_latency_ms as f64 * 1.5 {
            // Increase chunk size to reduce processing overhead
            if self.config.chunk_size < 4096 {
                self.config.chunk_size = (self.config.chunk_size as f32 * 1.2) as usize;
                self.config.overlap_size = self.config.chunk_size / 4; // Maintain 25% overlap
            }
        } else if stats.average_latency_ms < self.config.target_latency_ms as f64 * 0.5 {
            // Decrease chunk size for better responsiveness
            if self.config.chunk_size > 256 {
                self.config.chunk_size = (self.config.chunk_size as f32 * 0.8) as usize;
                self.config.overlap_size = self.config.chunk_size / 4;
            }
        }

        Ok(())
    }

    /// Get current quality metrics
    pub fn get_current_metrics(&self) -> EvaluationResult<Option<StreamingQualityMetrics>> {
        let history =
            self.metrics_history
                .lock()
                .map_err(|_| EvaluationError::ProcessingError {
                    message: "Failed to lock metrics history".to_string(),
                    source: None,
                })?;

        Ok(history.back().cloned())
    }

    /// Get processing statistics
    pub fn get_processing_stats(&self) -> EvaluationResult<ProcessingStats> {
        let stats = self
            .processing_stats
            .lock()
            .map_err(|_| EvaluationError::ProcessingError {
                message: "Failed to lock processing stats".to_string(),
                source: None,
            })?;

        Ok(ProcessingStats {
            total_chunks_processed: stats.total_chunks_processed,
            total_processing_time_ms: stats.total_processing_time_ms,
            average_latency_ms: stats.average_latency_ms,
            peak_latency_ms: stats.peak_latency_ms,
            dropped_chunks: stats.dropped_chunks,
        })
    }

    /// Get buffered audio for comprehensive analysis
    pub fn get_buffered_audio(&self) -> EvaluationResult<Option<AudioBuffer>> {
        let buffer = self
            .buffer
            .lock()
            .map_err(|_| EvaluationError::ProcessingError {
                message: "Failed to lock buffer".to_string(),
                source: None,
            })?;

        if buffer.is_empty() {
            return Ok(None);
        }

        // Combine all chunks in buffer
        let first_chunk = &buffer[0];
        let sample_rate = first_chunk.sample_rate;

        let mut combined_samples = Vec::new();
        for chunk in buffer.iter() {
            combined_samples.extend_from_slice(&chunk.samples);
        }

        Ok(Some(AudioBuffer::mono(combined_samples, sample_rate)))
    }

    /// Reset the streaming evaluator
    pub fn reset(&mut self) -> EvaluationResult<()> {
        // Clear buffer
        {
            let mut buffer = self
                .buffer
                .lock()
                .map_err(|_| EvaluationError::ProcessingError {
                    message: "Failed to lock buffer".to_string(),
                    source: None,
                })?;
            buffer.clear();
        }

        // Reset sequence
        {
            let mut sequence =
                self.sequence
                    .lock()
                    .map_err(|_| EvaluationError::ProcessingError {
                        message: "Failed to lock sequence".to_string(),
                        source: None,
                    })?;
            *sequence = 0;
        }

        // Clear metrics history
        {
            let mut history =
                self.metrics_history
                    .lock()
                    .map_err(|_| EvaluationError::ProcessingError {
                        message: "Failed to lock metrics history".to_string(),
                        source: None,
                    })?;
            history.clear();
        }

        // Reset processing stats
        {
            let mut stats =
                self.processing_stats
                    .lock()
                    .map_err(|_| EvaluationError::ProcessingError {
                        message: "Failed to lock processing stats".to_string(),
                        source: None,
                    })?;
            *stats = ProcessingStats::default();
        }

        Ok(())
    }

    /// Generate quality prediction for upcoming chunks
    pub async fn predict_quality(&mut self) -> EvaluationResult<QualityPrediction> {
        if !self.config.enable_quality_prediction {
            return Ok(QualityPrediction {
                predicted_score: 0.5,
                confidence: 0.0,
                trend: 0.0,
                risk_level: RiskLevel::Low,
                recommendations: vec!["Quality prediction disabled".to_string()],
            });
        }

        let mut predictor =
            self.quality_predictor
                .lock()
                .map_err(|_| EvaluationError::ProcessingError {
                    message: "Failed to lock quality predictor".to_string(),
                    source: None,
                })?;

        // Get recent quality history
        let metrics_history =
            self.metrics_history
                .lock()
                .map_err(|_| EvaluationError::ProcessingError {
                    message: "Failed to lock metrics history".to_string(),
                    source: None,
                })?;

        if metrics_history.len() < 3 {
            return Ok(QualityPrediction {
                predicted_score: 0.5,
                confidence: 0.3,
                trend: 0.0,
                risk_level: RiskLevel::Medium,
                recommendations: vec!["Insufficient data for prediction".to_string()],
            });
        }

        // Extract quality scores from recent metrics
        let recent_scores: Vec<f32> = metrics_history
            .iter()
            .rev()
            .take(self.config.prediction_horizon)
            .map(|m| (m.snr_estimate + m.dynamic_range + m.energy_level * 10.0) / 3.0)
            .collect();

        // Update predictor with recent scores
        for &score in &recent_scores {
            predictor.quality_history.push_back(score);
            if predictor.quality_history.len() > 20 {
                predictor.quality_history.pop_front();
            }
        }

        // Calculate predictions using different models
        let mut predictions = Vec::new();
        let mut confidences = Vec::new();

        // Moving average prediction
        if let Some(ma_model) = predictor.prediction_models.get("moving_average") {
            let short_ma = recent_scores.iter().take(3).sum::<f32>() / 3.0;
            let long_ma = recent_scores.iter().sum::<f32>() / recent_scores.len() as f32;
            let ma_prediction =
                ma_model.parameters[0] * short_ma + ma_model.parameters[1] * long_ma;
            predictions.push(ma_prediction);
            confidences.push(0.7);
        }

        // Trend analysis prediction
        if let Some(trend_model) = predictor.prediction_models.get("trend_analysis") {
            let trend = if recent_scores.len() >= 2 {
                (recent_scores[0] - recent_scores[recent_scores.len() - 1])
                    / recent_scores.len() as f32
            } else {
                0.0
            };
            let trend_prediction = recent_scores[0] + trend * trend_model.parameters[1];
            predictions.push(trend_prediction);
            confidences.push(0.6);
        }

        // Ensemble prediction
        let predicted_score = if !predictions.is_empty() {
            predictions.iter().sum::<f32>() / predictions.len() as f32
        } else {
            0.5
        };

        let confidence = if !confidences.is_empty() {
            confidences.iter().sum::<f32>() / confidences.len() as f32
        } else {
            0.3
        };

        // Calculate trend
        let trend = if recent_scores.len() >= 2 {
            recent_scores[0] - recent_scores[recent_scores.len() - 1]
        } else {
            0.0
        };

        // Determine risk level
        let risk_level = match predicted_score {
            s if s < 0.2 => RiskLevel::Critical,
            s if s < 0.4 => RiskLevel::High,
            s if s < 0.6 => RiskLevel::Medium,
            _ => RiskLevel::Low,
        };

        // Generate recommendations
        let mut recommendations = Vec::new();
        if predicted_score < 0.5 {
            recommendations
                .push("Consider reducing chunk size for better responsiveness".to_string());
        }
        if trend < -0.1 {
            recommendations.push("Quality trending downward - monitor closely".to_string());
        }
        if confidence < 0.5 {
            recommendations.push("Low prediction confidence - collect more data".to_string());
        }

        Ok(QualityPrediction {
            predicted_score,
            confidence,
            trend,
            risk_level,
            recommendations,
        })
    }

    /// Monitor network conditions and adapt accordingly
    pub async fn monitor_network_conditions(&mut self) -> EvaluationResult<NetworkCondition> {
        if !self.config.enable_network_adaptation {
            return Ok(NetworkCondition::default());
        }

        let mut monitor =
            self.network_monitor
                .lock()
                .map_err(|_| EvaluationError::ProcessingError {
                    message: "Failed to lock network monitor".to_string(),
                    source: None,
                })?;

        let now = Instant::now();
        if now.duration_since(monitor.last_check).as_millis()
            < self.config.network_monitor_interval_ms as u128
        {
            return Ok(monitor.current_condition.clone());
        }

        // Simulate network condition monitoring
        // In a real implementation, this would measure actual network metrics
        let mut condition = NetworkCondition::default();

        // Simulate varying network conditions
        let random_factor = (now.elapsed().as_secs() % 100) as f64 / 100.0;
        condition.bandwidth_estimate = (1_000_000.0 * (0.5 + random_factor)) as u64;
        condition.rtt_ms = 20.0 + random_factor * 100.0;
        condition.packet_loss_rate = (random_factor * 0.05) as f32;
        condition.jitter_ms = random_factor * 20.0;

        // Calculate quality score based on metrics
        let bandwidth_score = (condition.bandwidth_estimate as f32 / 2_000_000.0).min(1.0);
        let latency_score = 1.0 - (condition.rtt_ms / 200.0).min(1.0) as f32;
        let loss_score = 1.0 - condition.packet_loss_rate * 20.0;
        let jitter_score = 1.0 - (condition.jitter_ms / 50.0).min(1.0) as f32;

        condition.quality_score =
            (bandwidth_score + latency_score + loss_score + jitter_score) / 4.0;
        condition.timestamp = now;

        // Store in history
        monitor.condition_history.push_back(condition.clone());
        if monitor.condition_history.len() > 50 {
            monitor.condition_history.pop_front();
        }

        monitor.current_condition = condition.clone();
        monitor.last_check = now;

        // Drop the mutex guard before calling adapt function
        drop(monitor);

        // Adapt configuration based on network conditions
        self.adapt_to_network_conditions(&condition).await?;

        Ok(condition)
    }

    /// Detect anomalies in audio quality
    pub async fn detect_anomalies(
        &mut self,
        current_metrics: &StreamingQualityMetrics,
    ) -> EvaluationResult<AnomalyDetection> {
        if !self.config.enable_anomaly_detection {
            return Ok(AnomalyDetection {
                anomaly_detected: false,
                anomaly_score: 0.0,
                anomaly_type: AnomalyType::None,
                description: "Anomaly detection disabled".to_string(),
                severity: AnomalySeverity::Info,
                recommended_actions: Vec::new(),
            });
        }

        let mut detector =
            self.anomaly_detector
                .lock()
                .map_err(|_| EvaluationError::ProcessingError {
                    message: "Failed to lock anomaly detector".to_string(),
                    source: None,
                })?;

        // Update baseline if this is first measurement
        if detector.baseline_metrics.energy_level == 0.0 {
            detector.baseline_metrics = current_metrics.clone();
            return Ok(AnomalyDetection {
                anomaly_detected: false,
                anomaly_score: 0.0,
                anomaly_type: AnomalyType::None,
                description: "Establishing baseline".to_string(),
                severity: AnomalySeverity::Info,
                recommended_actions: Vec::new(),
            });
        }

        let mut anomaly_score = 0.0;
        let mut detected_anomalies = Vec::new();

        // Check for quality drops
        let quality_diff = detector.baseline_metrics.snr_estimate - current_metrics.snr_estimate;
        if quality_diff > detector.thresholds.quality_drop_threshold {
            anomaly_score += quality_diff * 2.0;
            detected_anomalies.push(AnomalyType::SuddenQualityDrop);
        }

        // Check for excessive clipping
        if current_metrics.clipping_detected {
            anomaly_score += 0.5;
            detected_anomalies.push(AnomalyType::ExcessiveClipping);
        }

        // Check for unexpected silence
        if current_metrics.silence_detected && detector.baseline_metrics.energy_level > 0.01 {
            anomaly_score += 0.3;
            detected_anomalies.push(AnomalyType::UnexpectedSilence);
        }

        // Check for processing delays
        if current_metrics.processing_latency_ms > detector.thresholds.latency_threshold {
            anomaly_score += 0.4;
            detected_anomalies.push(AnomalyType::ProcessingDelay);
        }

        // Check for energy deviations
        let energy_diff =
            (current_metrics.energy_level - detector.baseline_metrics.energy_level).abs();
        if energy_diff > detector.thresholds.energy_deviation_threshold {
            anomaly_score += energy_diff;
            detected_anomalies.push(AnomalyType::FrequencyImbalance);
        }

        // Apply sensitivity scaling
        anomaly_score *= self.config.anomaly_sensitivity;

        // Determine primary anomaly type and severity
        let (anomaly_type, severity) = if detected_anomalies.is_empty() {
            (AnomalyType::None, AnomalySeverity::Info)
        } else {
            let primary_anomaly = detected_anomalies[0].clone();
            let severity = match anomaly_score {
                s if s > 1.0 => AnomalySeverity::Critical,
                s if s > 0.7 => AnomalySeverity::Error,
                s if s > 0.4 => AnomalySeverity::Warning,
                _ => AnomalySeverity::Info,
            };
            (primary_anomaly, severity)
        };

        let anomaly_detected = anomaly_score > 0.3;

        // Generate description and recommendations
        let description = match &anomaly_type {
            AnomalyType::None => "No anomalies detected".to_string(),
            AnomalyType::SuddenQualityDrop => format!("Quality dropped by {:.2}", quality_diff),
            AnomalyType::ExcessiveClipping => "Audio clipping detected".to_string(),
            AnomalyType::UnexpectedSilence => "Unexpected silence period".to_string(),
            AnomalyType::ProcessingDelay => format!(
                "Processing latency: {:.1}ms",
                current_metrics.processing_latency_ms
            ),
            AnomalyType::FrequencyImbalance => "Energy level deviation detected".to_string(),
            _ => "Unknown anomaly detected".to_string(),
        };

        let mut recommended_actions = Vec::new();
        if anomaly_detected {
            match anomaly_type {
                AnomalyType::SuddenQualityDrop => {
                    recommended_actions.push("Check audio source quality".to_string());
                    recommended_actions.push("Verify network conditions".to_string());
                }
                AnomalyType::ExcessiveClipping => {
                    recommended_actions.push("Reduce input gain".to_string());
                    recommended_actions.push("Check for audio saturation".to_string());
                }
                AnomalyType::ProcessingDelay => {
                    recommended_actions.push("Increase chunk size".to_string());
                    recommended_actions.push("Check system resources".to_string());
                }
                _ => {
                    recommended_actions.push("Monitor closely".to_string());
                }
            }
        }

        let detection = AnomalyDetection {
            anomaly_detected,
            anomaly_score,
            anomaly_type,
            description,
            severity,
            recommended_actions,
        };

        // Store in history
        detector.anomaly_history.push_back(detection.clone());
        if detector.anomaly_history.len() > 100 {
            detector.anomaly_history.pop_front();
        }

        Ok(detection)
    }

    /// Adapt configuration based on network conditions
    async fn adapt_to_network_conditions(
        &mut self,
        condition: &NetworkCondition,
    ) -> EvaluationResult<()> {
        // Adjust chunk size based on bandwidth
        if condition.bandwidth_estimate < 500_000 {
            // Less than 500 kbps
            if self.config.chunk_size > 512 {
                self.config.chunk_size = 512;
                self.config.overlap_size = 128;
            }
        } else if condition.bandwidth_estimate > 2_000_000 {
            // More than 2 Mbps
            if self.config.chunk_size < 2048 {
                self.config.chunk_size = 2048;
                self.config.overlap_size = 512;
            }
        }

        // Adjust target latency based on RTT
        if condition.rtt_ms > 100.0 {
            self.config.target_latency_ms = (condition.rtt_ms * 2.0) as u64;
        }

        // Adjust buffer size based on jitter
        if condition.jitter_ms > 20.0 {
            self.config.max_buffer_chunks = (15.0 * condition.jitter_ms / 20.0) as usize;
        }

        Ok(())
    }

    /// Get advanced processing statistics
    pub fn get_advanced_stats(&self) -> EvaluationResult<AdvancedProcessingStats> {
        let stats = self
            .advanced_stats
            .lock()
            .map_err(|_| EvaluationError::ProcessingError {
                message: "Failed to lock advanced stats".to_string(),
                source: None,
            })?;

        Ok(stats.clone())
    }

    /// Get recent anomaly detections
    pub fn get_recent_anomalies(&self, limit: usize) -> EvaluationResult<Vec<AnomalyDetection>> {
        let detector =
            self.anomaly_detector
                .lock()
                .map_err(|_| EvaluationError::ProcessingError {
                    message: "Failed to lock anomaly detector".to_string(),
                    source: None,
                })?;

        Ok(detector
            .anomaly_history
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect())
    }

    /// Get network condition history
    pub fn get_network_history(&self, limit: usize) -> EvaluationResult<Vec<NetworkCondition>> {
        let monitor =
            self.network_monitor
                .lock()
                .map_err(|_| EvaluationError::ProcessingError {
                    message: "Failed to lock network monitor".to_string(),
                    source: None,
                })?;

        Ok(monitor
            .condition_history
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect())
    }
}

/// Utility function to split audio buffer into chunks
pub fn chunk_audio_buffer(
    audio: &AudioBuffer,
    chunk_size: usize,
    overlap_size: usize,
) -> Vec<AudioChunk> {
    let samples = audio.samples();
    let sample_rate = audio.sample_rate();
    let mut chunks = Vec::new();
    let step_size = chunk_size - overlap_size;

    let mut sequence = 0;
    let mut start = 0;

    while start < samples.len() {
        let end = (start + chunk_size).min(samples.len());
        let chunk_samples = samples[start..end].to_vec();

        // Only create chunk if it has meaningful content
        if chunk_samples.len() >= chunk_size / 2 {
            chunks.push(AudioChunk::new(chunk_samples, sample_rate, sequence));
            sequence += 1;
        }

        start += step_size;

        // Stop if next chunk would be too small
        if start >= samples.len() {
            break;
        }
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use voirs_sdk::AudioBuffer;

    #[tokio::test]
    async fn test_streaming_evaluator_creation() {
        let config = StreamingConfig::default();
        let evaluator = StreamingEvaluator::new(config);

        // Test that we can get initial stats
        let stats = evaluator.get_processing_stats().unwrap();
        assert_eq!(stats.total_chunks_processed, 0);
    }

    #[tokio::test]
    async fn test_chunk_processing() {
        let config = StreamingConfig::default();
        let mut evaluator = StreamingEvaluator::new(config);

        // Create a test chunk
        let samples = vec![0.1, 0.2, -0.1, -0.2, 0.3, -0.3]; // Simple test signal
        let chunk = AudioChunk::new(samples, 16000, 0);

        // Process the chunk
        let result = evaluator.process_chunk(chunk).await;
        assert!(result.is_ok());

        // Check that metrics were calculated
        let metrics = evaluator.get_current_metrics().unwrap();
        assert!(metrics.is_some());

        let metrics = metrics.unwrap();
        assert!(metrics.energy_level > 0.0);
        assert!(!metrics.silence_detected);
    }

    #[tokio::test]
    async fn test_audio_chunking() {
        let samples = vec![0.1; 1000]; // 1000 samples
        let audio = AudioBuffer::mono(samples, 16000);

        let chunks = chunk_audio_buffer(&audio, 256, 64);

        // Should create multiple chunks with overlap
        assert!(chunks.len() > 1);

        // Check first chunk
        assert_eq!(chunks[0].samples.len(), 256);
        assert_eq!(chunks[0].sample_rate, 16000);
        assert_eq!(chunks[0].sequence, 0);
    }

    #[tokio::test]
    async fn test_quality_monitoring() {
        let config = StreamingConfig {
            enable_quality_monitoring: true,
            ..Default::default()
        };
        let mut evaluator = StreamingEvaluator::new(config);
        let _receiver = evaluator.setup_quality_monitoring();

        // Process a chunk with known characteristics
        let samples = vec![0.5; 512]; // High energy signal
        let chunk = AudioChunk::new(samples, 16000, 0);

        let result = evaluator.process_chunk(chunk).await;
        assert!(result.is_ok());

        // Check metrics
        let metrics = evaluator.get_current_metrics().unwrap().unwrap();
        assert!(metrics.energy_level > 0.4);
        assert!(!metrics.silence_detected);
        assert!(!metrics.clipping_detected);
    }

    #[tokio::test]
    async fn test_silence_detection() {
        let config = StreamingConfig::default();
        let mut evaluator = StreamingEvaluator::new(config);

        // Process a silent chunk
        let samples = vec![0.0001; 512]; // Very low energy
        let chunk = AudioChunk::new(samples, 16000, 0);

        let result = evaluator.process_chunk(chunk).await;
        assert!(result.is_ok());

        let metrics = evaluator.get_current_metrics().unwrap().unwrap();
        assert!(metrics.silence_detected);
        assert!(metrics.energy_level < 0.001);
    }

    #[tokio::test]
    async fn test_clipping_detection() {
        let config = StreamingConfig::default();
        let mut evaluator = StreamingEvaluator::new(config);

        // Process a clipped chunk
        let samples = vec![0.98, -0.97, 0.99, -0.98]; // Near clipping threshold
        let chunk = AudioChunk::new(samples, 16000, 0);

        let result = evaluator.process_chunk(chunk).await;
        assert!(result.is_ok());

        let metrics = evaluator.get_current_metrics().unwrap().unwrap();
        assert!(metrics.clipping_detected);
    }

    #[test]
    fn test_streaming_config_defaults() {
        let config = StreamingConfig::default();
        assert_eq!(config.chunk_size, 1024);
        assert_eq!(config.overlap_size, 256);
        assert_eq!(config.max_buffer_chunks, 10);
        assert_eq!(config.target_latency_ms, 100);
        assert!(config.enable_quality_monitoring);
        assert!(config.enable_adaptive_processing);
    }
}
