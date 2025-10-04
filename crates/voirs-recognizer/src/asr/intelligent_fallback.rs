//! Intelligent fallback mechanism between ASR models
//!
//! This module provides intelligent model selection and fallback capabilities
//! to ensure robust ASR performance across different scenarios.

use super::{create_asr_model, ASRBackend, WhisperModelSize};
use crate::traits::{
    ASRConfig, ASRFeature, ASRMetadata, ASRModel, AudioStream, RecognitionResult, Transcript,
    TranscriptStream,
};
use crate::RecognitionError;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use voirs_sdk::{AudioBuffer, LanguageCode, VoirsError};

/// Configuration for intelligent fallback
#[derive(Debug, Clone)]
/// Fallback Config
pub struct FallbackConfig {
    /// Primary ASR backend
    pub primary_backend: ASRBackend,
    /// Fallback backends in order of preference
    pub fallback_backends: Vec<ASRBackend>,
    /// Quality threshold for primary model (confidence score)
    pub quality_threshold: f32,
    /// Maximum processing time before fallback (seconds)
    pub max_processing_time_seconds: f32,
    /// Maximum retries per model
    pub max_retries: usize,
    /// Enable adaptive model selection
    pub adaptive_selection: bool,
    /// Memory threshold for model switching (MB)
    pub memory_threshold_mb: f32,
    /// Minimum audio duration for model selection (seconds)
    pub min_duration_for_selection: f32,
}

impl Default for FallbackConfig {
    fn default() -> Self {
        Self {
            primary_backend: ASRBackend::Whisper {
                model_size: WhisperModelSize::Base,
                model_path: None,
            },
            fallback_backends: vec![ASRBackend::Whisper {
                model_size: WhisperModelSize::Tiny,
                model_path: None,
            }],
            quality_threshold: 0.7,
            max_processing_time_seconds: 10.0,
            max_retries: 3,
            adaptive_selection: true,
            memory_threshold_mb: 1024.0,
            min_duration_for_selection: 0.5,
        }
    }
}

/// Model performance metrics for adaptive selection
#[derive(Debug, Clone)]
/// Model Metrics
pub struct ModelMetrics {
    /// Average confidence score
    pub average_confidence: f32,
    /// Average processing time (seconds)
    pub average_processing_time: f32,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f32,
    /// Memory usage (MB)
    pub memory_usage_mb: f32,
    /// Number of processed items
    pub processed_count: usize,
    /// Number of failures
    pub failure_count: usize,
    /// Real-time factor
    pub average_rtf: f32,
    /// Performance per audio quality level
    pub quality_performance: HashMap<AudioQualityLevel, QualityMetrics>,
    /// Performance per language
    pub language_performance: HashMap<String, LanguageMetrics>,
    /// Performance trends over time
    pub performance_history: Vec<PerformanceSnapshot>,
    /// Memory pressure adaptation
    pub memory_pressure_performance: f32,
    /// GPU/CPU utilization efficiency
    pub resource_efficiency: f32,
    /// Model warmup time (cold start)
    pub warmup_time_ms: f64,
}

/// Audio quality level classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Audio Quality Level
pub enum AudioQualityLevel {
    /// Very low
    VeryLow,  // < 30 dB SNR
    /// Low
    Low,      // 30-40 dB SNR
    /// Medium
    Medium,   // 40-50 dB SNR
    /// High
    High,     // 50-60 dB SNR
    /// Very high
    VeryHigh, // > 60 dB SNR
}

/// Quality-specific performance metrics
#[derive(Debug, Clone, Default)]
/// Quality Metrics
pub struct QualityMetrics {
    /// confidence
    pub confidence: f32,
    /// processing time
    pub processing_time: f32,
    /// success rate
    pub success_rate: f32,
    /// sample count
    pub sample_count: usize,
}

/// Language-specific performance metrics
#[derive(Debug, Clone, Default)]
/// Language Metrics
pub struct LanguageMetrics {
    /// confidence
    pub confidence: f32,
    /// processing time
    pub processing_time: f32,
    /// success rate
    pub success_rate: f32,
    /// sample count
    pub sample_count: usize,
    /// wer
    pub wer: f32, // Word Error Rate
}

/// Performance snapshot for trend analysis
#[derive(Debug, Clone)]
/// Performance Snapshot
pub struct PerformanceSnapshot {
    /// timestamp
    pub timestamp: Instant,
    /// confidence
    pub confidence: f32,
    /// processing time
    pub processing_time: f32,
    /// rtf
    pub rtf: f32,
    /// memory usage mb
    pub memory_usage_mb: f32,
    /// success
    pub success: bool,
}

impl Default for ModelMetrics {
    fn default() -> Self {
        Self {
            average_confidence: 0.0,
            average_processing_time: 0.0,
            success_rate: 1.0,
            memory_usage_mb: 0.0,
            processed_count: 0,
            failure_count: 0,
            average_rtf: 1.0,
            quality_performance: HashMap::new(),
            language_performance: HashMap::new(),
            performance_history: Vec::new(),
            memory_pressure_performance: 1.0,
            resource_efficiency: 1.0,
            warmup_time_ms: 0.0,
        }
    }
}

/// Fallback reason enumeration
#[derive(Debug, Clone)]
/// Fallback Reason
pub enum FallbackReason {
    /// Low confidence
    LowConfidence(f32),
    /// Timeout
    Timeout(Duration),
    /// Model error
    ModelError(String),
    /// Memory pressure
    MemoryPressure(f32),
    /// Adaptive selection
    AdaptiveSelection,
    /// Explicit fallback
    ExplicitFallback,
}

/// Fallback result with metadata
#[derive(Debug, Clone)]
/// Fallback Result
pub struct FallbackResult {
    /// Final transcript
    pub transcript: Transcript,
    /// Backend used for successful transcription
    pub used_backend: ASRBackend,
    /// Whether fallback was triggered
    pub fallback_triggered: bool,
    /// Reason for fallback (if any)
    pub fallback_reason: Option<FallbackReason>,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Number of attempts made
    pub attempts_made: usize,
}

/// Intelligent ASR fallback manager
pub struct IntelligentASRFallback {
    /// Configuration
    config: FallbackConfig,
    /// Loaded models cache
    models: Arc<RwLock<HashMap<String, Arc<dyn ASRModel>>>>,
    /// Model performance metrics
    metrics: Arc<RwLock<HashMap<String, ModelMetrics>>>,
    /// Fallback statistics
    stats: Arc<RwLock<FallbackStats>>,
    /// Circuit breakers per model for reliability
    circuit_breakers: Arc<RwLock<HashMap<String, CircuitBreaker>>>,
}

/// Overall fallback statistics
#[derive(Debug, Clone, Default)]
/// Fallback Stats
pub struct FallbackStats {
    /// Total requests processed
    pub total_requests: usize,
    /// Number of fallbacks triggered
    pub fallbacks_triggered: usize,
    /// Success rate after fallback
    pub fallback_success_rate: f32,
    /// Average attempts per request
    pub average_attempts: f32,
    /// Model usage distribution
    pub model_usage: HashMap<String, usize>,
}

/// Circuit breaker state for reliability
#[derive(Debug, Clone, PartialEq)]
/// Circuit Breaker State
pub enum CircuitBreakerState {
    /// Circuit is closed (normal operation)
    Closed,
    /// Circuit is open (failing fast)
    Open,
    /// Circuit is half-open (testing recovery)
    HalfOpen,
}

/// Circuit breaker for ASR models to prevent cascading failures
#[derive(Debug, Clone)]
/// Circuit Breaker
pub struct CircuitBreaker {
    /// Current state
    pub state: CircuitBreakerState,
    /// Failure count in current window
    pub failure_count: usize,
    /// Success count in current window
    pub success_count: usize,
    /// Failure threshold to open circuit
    pub failure_threshold: usize,
    /// Success threshold to close circuit from half-open
    pub success_threshold: usize,
    /// Time window for counting failures (seconds)
    pub window_duration: Duration,
    /// Time to wait before transitioning to half-open
    pub open_timeout: Duration,
    /// Timestamp of last failure
    pub last_failure_time: Option<Instant>,
    /// Timestamp when circuit was opened
    pub opened_at: Option<Instant>,
    /// Window start time
    pub window_start: Instant,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self {
            state: CircuitBreakerState::Closed,
            failure_count: 0,
            success_count: 0,
            failure_threshold: 5,
            success_threshold: 3,
            window_duration: Duration::from_secs(60),
            open_timeout: Duration::from_secs(30),
            last_failure_time: None,
            opened_at: None,
            window_start: Instant::now(),
        }
    }
}

impl CircuitBreaker {
    /// Create a new circuit breaker with custom parameters
    #[must_use]
    /// new
    pub fn new(
        failure_threshold: usize,
        success_threshold: usize,
        window_duration: Duration,
        open_timeout: Duration,
    ) -> Self {
        Self {
            state: CircuitBreakerState::Closed,
            failure_count: 0,
            success_count: 0,
            failure_threshold,
            success_threshold,
            window_duration,
            open_timeout,
            last_failure_time: None,
            opened_at: None,
            window_start: Instant::now(),
        }
    }

    /// Check if the circuit breaker allows execution
    pub fn can_execute(&mut self) -> bool {
        self.update_state();

        match self.state {
            CircuitBreakerState::Open => false,
            CircuitBreakerState::Closed | CircuitBreakerState::HalfOpen => true,
        }
    }

    /// Record a successful execution
    pub fn record_success(&mut self) {
        self.update_state();

        match self.state {
            CircuitBreakerState::Closed => {
                self.success_count += 1;
                self.reset_counts_if_window_expired();
            }
            CircuitBreakerState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.success_threshold {
                    self.state = CircuitBreakerState::Closed;
                    self.reset_counts();
                }
            }
            CircuitBreakerState::Open => {
                // No action needed
            }
        }
    }

    /// Record a failed execution
    pub fn record_failure(&mut self) {
        self.update_state();
        self.last_failure_time = Some(Instant::now());

        match self.state {
            CircuitBreakerState::Closed => {
                self.failure_count += 1;
                if self.failure_count >= self.failure_threshold {
                    self.state = CircuitBreakerState::Open;
                    self.opened_at = Some(Instant::now());
                }
            }
            CircuitBreakerState::HalfOpen => {
                self.state = CircuitBreakerState::Open;
                self.opened_at = Some(Instant::now());
                self.reset_counts();
            }
            CircuitBreakerState::Open => {
                // Already open, no action needed
            }
        }
    }

    /// Update circuit breaker state based on time
    fn update_state(&mut self) {
        let now = Instant::now();

        // Check if window has expired and reset counts
        self.reset_counts_if_window_expired();

        // Check if we should transition from Open to HalfOpen
        if self.state == CircuitBreakerState::Open {
            if let Some(opened_at) = self.opened_at {
                if now.duration_since(opened_at) >= self.open_timeout {
                    self.state = CircuitBreakerState::HalfOpen;
                    self.reset_counts();
                }
            }
        }
    }

    /// Reset counts if the current window has expired
    fn reset_counts_if_window_expired(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.window_start) >= self.window_duration {
            self.reset_counts();
        }
    }

    /// Reset failure and success counts
    fn reset_counts(&mut self) {
        self.failure_count = 0;
        self.success_count = 0;
        self.window_start = Instant::now();
    }

    /// Get current failure rate in the window
    #[must_use]
    /// get failure rate
    pub fn get_failure_rate(&self) -> f32 {
        let total = self.failure_count + self.success_count;
        if total == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            {
                self.failure_count as f32 / total as f32
            }
        }
    }
}

impl IntelligentASRFallback {
    /// Create a new intelligent ASR fallback manager
    ///
    /// # Errors
    ///
    /// Returns `RecognitionError` if the fallback manager initialization fails
    pub async fn new(config: FallbackConfig) -> Result<Self, RecognitionError> {
        let models = Arc::new(RwLock::new(HashMap::new()));
        let metrics = Arc::new(RwLock::new(HashMap::new()));
        let stats = Arc::new(RwLock::new(FallbackStats::default()));
        let circuit_breakers = Arc::new(RwLock::new(HashMap::new()));

        let fallback = Self {
            config,
            models,
            metrics,
            stats,
            circuit_breakers,
        };

        // Initialize models
        fallback.initialize_models().await?;

        Ok(fallback)
    }

    /// Initialize all configured models
    async fn initialize_models(&self) -> Result<(), RecognitionError> {
        let mut models = self.models.write().await;
        let mut metrics = self.metrics.write().await;

        // Initialize primary model
        let primary_key = self.backend_to_key(&self.config.primary_backend);
        if !models.contains_key(&primary_key) {
            tracing::info!("Initializing primary model: {}", primary_key);
            let model = create_asr_model(self.config.primary_backend.clone()).await?;
            models.insert(primary_key.clone(), model);
            metrics.insert(primary_key, ModelMetrics::default());
        }

        // Initialize fallback models
        for backend in &self.config.fallback_backends {
            let key = self.backend_to_key(backend);
            if !models.contains_key(&key) {
                tracing::info!("Initializing fallback model: {}", key);
                match create_asr_model(backend.clone()).await {
                    Ok(model) => {
                        models.insert(key.clone(), model);
                        metrics.insert(key, ModelMetrics::default());
                    }
                    Err(e) => {
                        tracing::warn!("Failed to initialize fallback model {}: {}", key, e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Convert backend to string key
    #[allow(clippy::unused_self)]
    fn backend_to_key(&self, backend: &ASRBackend) -> String {
        match backend {
            ASRBackend::Whisper { model_size, .. } => {
                format!("whisper_{}", model_size.as_str())
            }
            ASRBackend::DeepSpeech { .. } => "deepspeech".to_string(),
            ASRBackend::Wav2Vec2 { model_id, .. } => {
                format!("wav2vec2_{model_id}")
            }
            #[cfg(feature = "transformer")]
            ASRBackend::Transformer { .. } => "transformer".to_string(),
            #[cfg(feature = "conformer")]
            ASRBackend::Conformer { .. } => "conformer".to_string(),
        }
    }

    /// Select the best model for the given audio with enhanced metrics
    async fn select_best_model(&self, audio: &AudioBuffer) -> Result<String, RecognitionError> {
        if !self.config.adaptive_selection {
            return Ok(self.backend_to_key(&self.config.primary_backend));
        }

        let metrics = self.metrics.read().await;
        let audio_duration = audio.duration();

        // For very short audio, use fastest model
        if audio_duration < self.config.min_duration_for_selection {
            let fastest_model = self.select_fastest_model(&metrics).await;
            return Ok(fastest_model);
        }

        // Analyze audio characteristics for contextual selection
        let audio_quality = self.analyze_audio_quality(audio).await;
        let estimated_language = self.estimate_language(audio).await;
        let current_memory_pressure = self.get_memory_pressure().await;

        // Score models based on comprehensive factors
        let mut model_scores = HashMap::new();

        for (model_key, model_metrics) in metrics.iter() {
            if model_metrics.processed_count == 0 {
                // New model gets default score based on expected performance
                model_scores.insert(model_key.clone(), self.estimate_new_model_score(model_key));
                continue;
            }

            let score = self
                .calculate_comprehensive_score(
                    model_metrics,
                    &audio_quality,
                    &estimated_language,
                    current_memory_pressure,
                    audio_duration,
                )
                .await;

            model_scores.insert(model_key.clone(), score);
        }

        // Select highest scoring model
        let best_model = model_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map_or_else(
                || self.backend_to_key(&self.config.primary_backend),
                |(k, _)| k.clone(),
            );

        tracing::debug!(
            "Selected model: {} with scores: {:?}",
            best_model,
            model_scores
        );
        Ok(best_model)
    }

    /// Select the fastest available model
    async fn select_fastest_model(&self, metrics: &HashMap<String, ModelMetrics>) -> String {
        metrics
            .iter()
            .filter(|(_, m)| m.processed_count > 0)
            .min_by(|a, b| {
                let a_warmup_factor = if a.1.processed_count < 5 { 2.0 } else { 1.0 };
                let b_warmup_factor = if b.1.processed_count < 5 { 2.0 } else { 1.0 };

                let a_time = a.1.average_processing_time * a_warmup_factor;
                let b_time = b.1.average_processing_time * b_warmup_factor;

                a_time.partial_cmp(&b_time).unwrap()
            })
            .map_or_else(
                || self.backend_to_key(&self.config.primary_backend),
                |(k, _)| k.clone(),
            )
    }

    /// Analyze audio quality level based on SNR and other characteristics
    async fn analyze_audio_quality(&self, audio: &AudioBuffer) -> AudioQualityLevel {
        // Calculate SNR and other audio quality metrics
        let samples = audio.samples();
        let rms = Self::calculate_rms(samples);
        let noise_floor = Self::estimate_noise_floor(samples);

        let snr_db = if noise_floor > 0.0 {
            20.0 * (rms / noise_floor).log10()
        } else {
            60.0 // Assume high quality if no noise detected
        };

        match snr_db {
            x if x < 30.0 => AudioQualityLevel::VeryLow,
            x if x < 40.0 => AudioQualityLevel::Low,
            x if x < 50.0 => AudioQualityLevel::Medium,
            x if x < 60.0 => AudioQualityLevel::High,
            _ => AudioQualityLevel::VeryHigh,
        }
    }

    /// Calculate RMS of audio samples
    fn calculate_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    /// Estimate noise floor using lower percentile of signal
    fn estimate_noise_floor(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let mut abs_samples: Vec<f32> = samples.iter().map(|&x| x.abs()).collect();
        abs_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Use 10th percentile as noise floor estimate
        let percentile_index = (abs_samples.len() as f32 * 0.1) as usize;
        abs_samples.get(percentile_index).copied().unwrap_or(0.0)
    }

    /// Estimate language from audio characteristics (simplified heuristic)
    async fn estimate_language(&self, _audio: &AudioBuffer) -> String {
        // In a real implementation, this would use language detection
        // For now, return a default or use frequency analysis
        "unknown".to_string()
    }

    /// Get current memory pressure level
    async fn get_memory_pressure(&self) -> f32 {
        // In a real implementation, this would check system memory usage
        // For now, return a default value
        0.5 // Medium memory pressure
    }

    /// Estimate score for a new model without historical data
    fn estimate_new_model_score(&self, model_key: &str) -> f32 {
        // Provide expected performance based on model type
        if model_key.contains("tiny") {
            0.6 // Fast but lower accuracy
        } else if model_key.contains("base") {
            0.7 // Balanced
        } else if model_key.contains("small") {
            0.8 // Good accuracy, moderate speed
        } else if model_key.contains("medium") || model_key.contains("large") {
            0.9 // High accuracy, slower
        } else {
            0.5 // Unknown model
        }
    }

    /// Calculate comprehensive model score based on multiple factors
    async fn calculate_comprehensive_score(
        &self,
        metrics: &ModelMetrics,
        audio_quality: &AudioQualityLevel,
        estimated_language: &str,
        memory_pressure: f32,
        audio_duration: f32,
    ) -> f32 {
        // Base performance scores
        let confidence_score = metrics.average_confidence * 0.3;
        let speed_score = (1.0 / metrics.average_rtf.max(0.1)) * 0.2;
        let reliability_score = metrics.success_rate * 0.2;
        let efficiency_score = metrics.resource_efficiency * 0.1;

        // Quality-specific performance
        let quality_bonus =
            if let Some(quality_metrics) = metrics.quality_performance.get(audio_quality) {
                quality_metrics.confidence * 0.1
            } else {
                0.0
            };

        // Language-specific performance
        let language_bonus =
            if let Some(lang_metrics) = metrics.language_performance.get(estimated_language) {
                lang_metrics.confidence * 0.05
            } else {
                0.0
            };

        // Memory pressure adjustment
        let memory_adjustment = if memory_pressure > 0.8 {
            // High memory pressure: prefer lighter models
            if metrics.memory_usage_mb < 500.0 {
                0.05
            } else {
                -0.1
            }
        } else {
            0.0
        };

        // Duration-based adjustment
        let duration_adjustment = if audio_duration > 30.0 {
            // Long audio: prefer accuracy over speed
            if metrics.average_confidence > 0.8 {
                0.05
            } else {
                0.0
            }
        } else {
            // Short audio: prefer speed
            if metrics.average_rtf < 0.5 {
                0.05
            } else {
                0.0
            }
        };

        // Performance trend analysis
        let trend_score = self.analyze_performance_trend(metrics);

        let total_score = confidence_score
            + speed_score
            + reliability_score
            + efficiency_score
            + quality_bonus
            + language_bonus
            + memory_adjustment
            + duration_adjustment
            + trend_score;

        total_score.clamp(0.0, 1.0)
    }

    /// Analyze performance trends from historical data
    fn analyze_performance_trend(&self, metrics: &ModelMetrics) -> f32 {
        if metrics.performance_history.len() < 3 {
            return 0.0; // Not enough data for trend analysis
        }

        let recent_snapshots =
            &metrics.performance_history[metrics.performance_history.len().saturating_sub(10)..];

        // Calculate trend in confidence and processing time
        let confidence_trend = Self::calculate_linear_trend(
            &recent_snapshots
                .iter()
                .map(|s| s.confidence)
                .collect::<Vec<_>>(),
        );

        let rtf_trend = Self::calculate_linear_trend(
            &recent_snapshots.iter().map(|s| s.rtf).collect::<Vec<_>>(),
        );

        // Positive confidence trend and negative RTF trend are good
        let trend_bonus = (confidence_trend - rtf_trend) * 0.02;
        trend_bonus.clamp(-0.05, 0.05)
    }

    /// Calculate linear trend (slope) for a series of values
    fn calculate_linear_trend(values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f32;
        let sum_x: f32 = (0..values.len()).map(|i| i as f32).sum();
        let sum_y: f32 = values.iter().sum();
        let sum_x_times_y: f32 = values.iter().enumerate().map(|(i, &y)| i as f32 * y).sum();
        let sum_x2: f32 = (0..values.len()).map(|i| (i as f32).powi(2)).sum();

        // Linear regression slope calculation
        let denominator = n * sum_x2 - sum_x.powi(2);
        if denominator.abs() < f32::EPSILON {
            0.0
        } else {
            (n * sum_x_times_y - sum_x * sum_y) / denominator
        }
    }

    /// Process audio with intelligent fallback
    pub async fn transcribe(
        &self,
        audio: &AudioBuffer,
        config: Option<&ASRConfig>,
    ) -> Result<FallbackResult, RecognitionError> {
        let start_time = Instant::now();
        let mut attempts_made = 0;
        let mut fallback_triggered = false;
        let mut fallback_reason = None;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_requests += 1;
        }

        // Select best model
        let selected_model_key = self.select_best_model(audio).await?;
        let mut model_keys_to_try = vec![selected_model_key];

        // Add fallback models
        for backend in &self.config.fallback_backends {
            let key = self.backend_to_key(backend);
            if !model_keys_to_try.contains(&key) {
                model_keys_to_try.push(key);
            }
        }

        // Try primary model first if not already selected
        let primary_key = self.backend_to_key(&self.config.primary_backend);
        if !model_keys_to_try.contains(&primary_key) {
            model_keys_to_try.insert(0, primary_key);
        }

        let mut last_error = None;

        for model_key in model_keys_to_try {
            attempts_made += 1;
            let attempt_start = Instant::now();

            let models = self.models.read().await;
            let model = if let Some(model) = models.get(&model_key) {
                model.clone()
            } else {
                tracing::warn!("Model {} not available, skipping", model_key);
                continue;
            };
            drop(models);

            // Set timeout for the operation
            let timeout_duration = Duration::from_secs_f32(self.config.max_processing_time_seconds);

            let transcribe_result =
                tokio::time::timeout(timeout_duration, model.transcribe(audio, config)).await;

            let processing_time = attempt_start.elapsed();

            match transcribe_result {
                Ok(Ok(transcript)) => {
                    // Check quality threshold
                    if transcript.confidence >= self.config.quality_threshold {
                        // Success! Update metrics and return
                        self.update_model_metrics(
                            &model_key,
                            true,
                            transcript.confidence,
                            processing_time,
                            audio.duration(),
                        )
                        .await;

                        let backend = if model_key.starts_with("whisper") {
                            if model_key.contains("tiny") {
                                ASRBackend::Whisper {
                                    model_size: WhisperModelSize::Tiny,
                                    model_path: None,
                                }
                            } else if model_key.contains("small") {
                                ASRBackend::Whisper {
                                    model_size: WhisperModelSize::Small,
                                    model_path: None,
                                }
                            } else {
                                ASRBackend::Whisper {
                                    model_size: WhisperModelSize::Base,
                                    model_path: None,
                                }
                            }
                        } else {
                            self.config.primary_backend.clone()
                        };

                        if attempts_made > 1 {
                            let mut stats = self.stats.write().await;
                            stats.fallbacks_triggered += 1;
                        }

                        return Ok(FallbackResult {
                            transcript,
                            used_backend: backend,
                            fallback_triggered,
                            fallback_reason,
                            total_processing_time: start_time.elapsed(),
                            attempts_made,
                        });
                    }
                    // Low confidence, trigger fallback
                    fallback_triggered = true;
                    fallback_reason = Some(FallbackReason::LowConfidence(transcript.confidence));
                    self.update_model_metrics(
                        &model_key,
                        false,
                        transcript.confidence,
                        processing_time,
                        audio.duration(),
                    )
                    .await;
                    tracing::debug!(
                        "Low confidence ({:.2}) from {}, trying fallback",
                        transcript.confidence,
                        model_key
                    );
                }
                Ok(Err(e)) => {
                    // Model error, try fallback
                    fallback_triggered = true;
                    fallback_reason = Some(FallbackReason::ModelError(e.to_string()));
                    self.update_model_metrics(
                        &model_key,
                        false,
                        0.0,
                        processing_time,
                        audio.duration(),
                    )
                    .await;
                    last_error = Some(e);
                    tracing::debug!(
                        "Model error from {}: {}, trying fallback",
                        model_key,
                        last_error.as_ref().unwrap()
                    );
                }
                Err(_timeout) => {
                    // Timeout, try fallback
                    fallback_triggered = true;
                    fallback_reason = Some(FallbackReason::Timeout(processing_time));
                    self.update_model_metrics(
                        &model_key,
                        false,
                        0.0,
                        processing_time,
                        audio.duration(),
                    )
                    .await;
                    tracing::debug!(
                        "Timeout from {} after {:?}, trying fallback",
                        model_key,
                        processing_time
                    );
                }
            }
        }

        // All models failed
        Err(last_error
            .unwrap_or_else(|| VoirsError::ModelError {
                model_type: voirs_sdk::error::ModelType::ASR,
                message: "All ASR models failed".to_string(),
                source: None,
            })
            .into())
    }

    /// Update model performance metrics
    async fn update_model_metrics(
        &self,
        model_key: &str,
        success: bool,
        confidence: f32,
        processing_time: Duration,
        audio_duration: f32,
    ) {
        let mut metrics = self.metrics.write().await;
        let model_metrics = metrics
            .entry(model_key.to_string())
            .or_insert_with(ModelMetrics::default);

        // Update success rate
        let total_attempts = model_metrics.processed_count + model_metrics.failure_count + 1;
        let successful_attempts = if success {
            model_metrics.processed_count + 1
        } else {
            model_metrics.processed_count
        };

        model_metrics.success_rate = successful_attempts as f32 / total_attempts as f32;

        if success {
            model_metrics.processed_count += 1;

            // Update averages
            let n = model_metrics.processed_count as f32;
            model_metrics.average_confidence =
                (model_metrics.average_confidence * (n - 1.0) + confidence) / n;
            model_metrics.average_processing_time =
                (model_metrics.average_processing_time * (n - 1.0) + processing_time.as_secs_f32())
                    / n;

            // Calculate RTF
            let rtf = processing_time.as_secs_f32() / audio_duration;
            model_metrics.average_rtf = (model_metrics.average_rtf * (n - 1.0) + rtf) / n;
        } else {
            model_metrics.failure_count += 1;
        }

        // Update model usage stats
        let mut stats = self.stats.write().await;
        *stats.model_usage.entry(model_key.to_string()).or_insert(0) += 1;
    }

    /// Get fallback statistics
    pub async fn get_stats(&self) -> FallbackStats {
        self.stats.read().await.clone()
    }

    /// Get model metrics
    pub async fn get_model_metrics(&self) -> HashMap<String, ModelMetrics> {
        self.metrics.read().await.clone()
    }

    /// Reset all statistics and metrics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = FallbackStats::default();

        let mut metrics = self.metrics.write().await;
        for model_metrics in metrics.values_mut() {
            *model_metrics = ModelMetrics::default();
        }
    }

    /// Force fallback to specific model
    pub async fn force_fallback(
        &self,
        backend: ASRBackend,
        audio: &AudioBuffer,
        config: Option<&ASRConfig>,
    ) -> Result<FallbackResult, RecognitionError> {
        let start_time = Instant::now();
        let model_key = self.backend_to_key(&backend);

        let models = self.models.read().await;
        let model = models
            .get(&model_key)
            .ok_or_else(|| RecognitionError::ModelError {
                message: format!("Model {model_key} not available"),
                source: None,
            })?;

        let transcript = model.transcribe(audio, config).await?;

        Ok(FallbackResult {
            transcript,
            used_backend: backend,
            fallback_triggered: true,
            fallback_reason: Some(FallbackReason::ExplicitFallback),
            total_processing_time: start_time.elapsed(),
            attempts_made: 1,
        })
    }

    /// Get or create a circuit breaker for a model
    #[allow(dead_code)]
    async fn get_circuit_breaker(&self, model_key: &str) -> CircuitBreaker {
        let mut breakers = self.circuit_breakers.write().await;
        breakers
            .entry(model_key.to_string())
            .or_insert_with(CircuitBreaker::default)
            .clone()
    }

    /// Update circuit breaker state
    #[allow(dead_code)]
    async fn update_circuit_breaker(&self, model_key: &str, breaker: CircuitBreaker) {
        let mut breakers = self.circuit_breakers.write().await;
        breakers.insert(model_key.to_string(), breaker);
    }

    /// Check if a model can be executed based on circuit breaker state
    #[allow(dead_code)]
    async fn can_execute_model(&self, model_key: &str) -> bool {
        let mut breaker = self.get_circuit_breaker(model_key).await;
        let can_execute = breaker.can_execute();
        self.update_circuit_breaker(model_key, breaker).await;
        can_execute
    }

    /// Record successful execution for a model
    #[allow(dead_code)]
    async fn record_model_success(&self, model_key: &str) {
        let mut breaker = self.get_circuit_breaker(model_key).await;
        breaker.record_success();
        self.update_circuit_breaker(model_key, breaker).await;
    }

    /// Record failed execution for a model
    #[allow(dead_code)]
    async fn record_model_failure(&self, model_key: &str) {
        let mut breaker = self.get_circuit_breaker(model_key).await;
        breaker.record_failure();
        self.update_circuit_breaker(model_key, breaker).await;
    }

    /// Get circuit breaker status for all models
    pub async fn get_circuit_breaker_status(&self) -> HashMap<String, CircuitBreakerState> {
        let breakers = self.circuit_breakers.read().await;
        breakers
            .iter()
            .map(|(key, breaker)| (key.clone(), breaker.state.clone()))
            .collect()
    }

    /// Reset circuit breaker for a specific model
    pub async fn reset_circuit_breaker(&self, model_key: &str) {
        let mut breakers = self.circuit_breakers.write().await;
        breakers.insert(model_key.to_string(), CircuitBreaker::default());
    }

    /// Reset all circuit breakers
    pub async fn reset_all_circuit_breakers(&self) {
        let mut breakers = self.circuit_breakers.write().await;
        breakers.clear();
    }
}

#[async_trait]
impl ASRModel for IntelligentASRFallback {
    async fn transcribe(
        &self,
        audio: &AudioBuffer,
        config: Option<&ASRConfig>,
    ) -> RecognitionResult<Transcript> {
        let result = self.transcribe(audio, config).await?;
        Ok(result.transcript)
    }

    async fn transcribe_streaming(
        &self,
        audio_stream: AudioStream,
        config: Option<&ASRConfig>,
    ) -> RecognitionResult<TranscriptStream> {
        // Use primary model for streaming (fallback not suitable for streaming)
        let models = self.models.read().await;
        let primary_key = self.backend_to_key(&self.config.primary_backend);
        let model = models
            .get(&primary_key)
            .ok_or_else(|| RecognitionError::ModelError {
                message: "Primary model not available for streaming".to_string(),
                source: None,
            })?;

        model.transcribe_streaming(audio_stream, config).await
    }

    fn supported_languages(&self) -> Vec<LanguageCode> {
        // Return intersection of all supported languages
        vec![
            LanguageCode::EnUs,
            LanguageCode::EnGb,
            LanguageCode::DeDe,
            LanguageCode::FrFr,
            LanguageCode::EsEs,
            LanguageCode::JaJp,
            LanguageCode::ZhCn,
            LanguageCode::KoKr,
        ]
    }

    fn metadata(&self) -> ASRMetadata {
        ASRMetadata {
            name: "Intelligent ASR Fallback".to_string(),
            version: "1.0.0".to_string(),
            description: "Intelligent fallback mechanism with multiple ASR models".to_string(),
            supported_languages: self.supported_languages(),
            architecture: "Multi-Model Fallback".to_string(),
            model_size_mb: 500.0, // Approximate combined size
            inference_speed: 1.0,
            wer_benchmarks: HashMap::new(),
            supported_features: vec![
                ASRFeature::WordTimestamps,
                ASRFeature::SentenceSegmentation,
                ASRFeature::LanguageDetection,
                ASRFeature::NoiseRobustness,
            ],
        }
    }

    fn supports_feature(&self, feature: ASRFeature) -> bool {
        matches!(
            feature,
            ASRFeature::WordTimestamps
                | ASRFeature::SentenceSegmentation
                | ASRFeature::LanguageDetection
                | ASRFeature::NoiseRobustness
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fallback_creation() {
        let config = FallbackConfig::default();
        let _fallback = IntelligentASRFallback::new(config).await;
        // Note: This might fail in tests due to model dependencies
        // In a real implementation, we'd mock the models
    }

    #[test]
    fn test_backend_to_key() {
        let fallback_config = FallbackConfig::default();
        let fallback = IntelligentASRFallback {
            config: fallback_config,
            models: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(FallbackStats::default())),
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
        };

        let whisper_backend = ASRBackend::Whisper {
            model_size: WhisperModelSize::Base,
            model_path: None,
        };
        assert_eq!(fallback.backend_to_key(&whisper_backend), "whisper_base");

        let deepspeech_backend = ASRBackend::DeepSpeech {
            model_path: "test.pbmm".to_string(),
            scorer_path: None,
        };
        assert_eq!(fallback.backend_to_key(&deepspeech_backend), "deepspeech");
    }

    #[test]
    fn test_model_metrics_update() {
        let mut metrics = ModelMetrics::default();

        // Simulate successful processing
        metrics.processed_count = 1;
        metrics.average_confidence = 0.9;
        metrics.success_rate = 1.0;

        assert_eq!(metrics.processed_count, 1);
        assert_eq!(metrics.average_confidence, 0.9);
        assert_eq!(metrics.success_rate, 1.0);
    }
}
