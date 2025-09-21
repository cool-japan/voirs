//! Intelligent Model Manager for Enhanced ASR Model Switching
//!
//! This module extends the intelligent fallback system with enhanced capabilities including:
//! - Context-aware model selection based on audio content analysis
//! - Resource-aware switching with dynamic resource monitoring
//! - Adaptive quality thresholds based on usage patterns
//! - Smart pre-loading and caching strategies
//! - Cold start optimization

use crate::preprocessing::{AdaptiveProcessor, AudioContentType, AudioFeatures};
use crate::traits::ASRModel;
use crate::RecognitionError;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
// Note: sysinfo dependency removed for now, using mock system monitoring
use tokio::sync::{Mutex, RwLock};
use voirs_sdk::AudioBuffer;

/// Enhanced model switching configuration
#[derive(Debug, Clone)]
pub struct IntelligentModelConfig {
    /// Enable context-aware switching
    pub context_aware_switching: bool,
    /// Enable resource-aware switching
    pub resource_aware_switching: bool,
    /// Enable adaptive quality thresholds
    pub adaptive_quality_thresholds: bool,
    /// Enable model pre-loading
    pub model_preloading: bool,
    /// Maximum models to keep loaded
    pub max_loaded_models: usize,
    /// Model eviction timeout (seconds)
    pub model_eviction_timeout: f32,
    /// Resource monitoring interval (seconds)
    pub resource_monitoring_interval: f32,
    /// Cold start optimization
    pub cold_start_optimization: bool,
    /// Minimum confidence for switching decisions
    pub min_confidence_for_switching: f32,
    /// Learning rate for adaptive thresholds
    pub learning_rate: f32,
}

impl Default for IntelligentModelConfig {
    fn default() -> Self {
        Self {
            context_aware_switching: true,
            resource_aware_switching: true,
            adaptive_quality_thresholds: true,
            model_preloading: true,
            max_loaded_models: 3,
            model_eviction_timeout: 300.0, // 5 minutes
            resource_monitoring_interval: 1.0,
            cold_start_optimization: true,
            min_confidence_for_switching: 0.6,
            learning_rate: 0.01,
        }
    }
}

/// System resource status
#[derive(Debug, Clone)]
pub struct ResourceStatus {
    /// CPU usage percentage (0-100)
    pub cpu_usage: f32,
    /// Memory usage percentage (0-100)
    pub memory_usage: f32,
    /// Available memory in MB
    pub available_memory_mb: f32,
    /// GPU memory usage (if available)
    pub gpu_memory_usage: Option<f32>,
    /// System load average
    pub load_average: f32,
    /// Temperature (if available)
    pub temperature: Option<f32>,
}

/// Model context information for intelligent switching
#[derive(Debug, Clone)]
pub struct ModelContext {
    /// Content type classification
    pub content_type: AudioContentType,
    /// Audio quality level
    pub quality_level: AudioQualityLevel,
    /// Expected processing time
    pub expected_processing_time: f32,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
    /// Priority level
    pub priority: ProcessingPriority,
}

/// Audio quality classification for model selection
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AudioQualityLevel {
    Poor,      // Heavy noise, low SNR
    Fair,      // Some noise, medium SNR
    Good,      // Light noise, good SNR
    Excellent, // Clean audio, high SNR
}

/// Model resource requirements
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    /// Memory requirement in MB
    pub memory_mb: f32,
    /// CPU intensity (0-1)
    pub cpu_intensity: f32,
    /// GPU requirement (0-1, None if GPU not required)
    pub gpu_requirement: Option<f32>,
    /// Minimum warmup time in ms
    pub warmup_time_ms: f64,
}

/// Processing priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProcessingPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Model preloading strategy
#[derive(Debug, Clone)]
pub enum PreloadingStrategy {
    /// Keep most frequently used models loaded
    FrequencyBased,
    /// Keep models for predicted content types loaded
    PredictiveBased,
    /// Keep lightweight models always loaded
    PerformanceBased,
    /// Adaptive strategy based on usage patterns
    AdaptiveBased,
}

/// Adaptive quality threshold manager
#[derive(Debug, Clone)]
pub struct AdaptiveThresholds {
    /// Current quality threshold per content type
    pub content_type_thresholds: HashMap<AudioContentType, f32>,
    /// Current quality threshold per audio quality level
    pub quality_level_thresholds: HashMap<AudioQualityLevel, f32>,
    /// Threshold adjustment history
    pub adjustment_history: VecDeque<ThresholdAdjustment>,
    /// Learning parameters
    pub learning_rate: f32,
    pub decay_factor: f32,
}

/// Threshold adjustment record
#[derive(Debug, Clone)]
pub struct ThresholdAdjustment {
    pub timestamp: Instant,
    pub content_type: AudioContentType,
    pub old_threshold: f32,
    pub new_threshold: f32,
    pub success_rate: f32,
}

/// Model performance predictor
#[derive(Debug)]
pub struct ModelPerformancePredictor {
    /// Historical performance data
    performance_data: HashMap<String, VecDeque<PerformanceRecord>>,
    /// Context-performance mappings
    context_mappings: HashMap<ModelContext, Vec<PerformanceRecord>>,
    /// Prediction accuracy tracker
    prediction_accuracy: f32,
}

/// Performance record for prediction training
#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    pub model_key: String,
    pub context: ModelContext,
    pub actual_confidence: f32,
    pub actual_processing_time: f32,
    pub actual_resource_usage: ResourceStatus,
    pub timestamp: Instant,
}

/// Model cache with intelligent eviction
pub struct IntelligentModelCache {
    /// Loaded models
    loaded_models: HashMap<String, CachedModel>,
    /// Model access patterns
    access_patterns: HashMap<String, AccessPattern>,
    /// Preloading strategy
    preloading_strategy: PreloadingStrategy,
    /// Maximum cache size
    max_cache_size: usize,
}

/// Cached model with metadata
pub struct CachedModel {
    pub model: Arc<dyn ASRModel>,
    pub load_time: Instant,
    pub last_access: Instant,
    pub access_count: usize,
    pub resource_usage: ResourceRequirements,
    pub warmup_complete: bool,
}

/// Model access pattern tracker
#[derive(Debug, Clone)]
pub struct AccessPattern {
    pub frequency: f32,
    pub recency: f32,
    pub context_correlation: HashMap<AudioContentType, f32>,
    pub time_patterns: Vec<AccessTime>,
}

/// Access time record
#[derive(Debug, Clone)]
pub struct AccessTime {
    pub timestamp: Instant,
    pub context: AudioContentType,
    pub success: bool,
}

/// Enhanced intelligent model manager
pub struct IntelligentModelManager {
    config: IntelligentModelConfig,
    resource_status: Arc<RwLock<ResourceStatus>>,
    adaptive_thresholds: Arc<RwLock<AdaptiveThresholds>>,
    performance_predictor: Arc<Mutex<ModelPerformancePredictor>>,
    model_cache: Arc<Mutex<IntelligentModelCache>>,
    adaptive_processor: Arc<Mutex<AdaptiveProcessor>>,
    usage_statistics: Arc<RwLock<UsageStatistics>>,
}

/// Usage statistics for optimization
#[derive(Debug, Clone, Default)]
pub struct UsageStatistics {
    pub total_requests: usize,
    pub context_distribution: HashMap<AudioContentType, usize>,
    pub quality_distribution: HashMap<AudioQualityLevel, usize>,
    pub model_success_rates: HashMap<String, f32>,
    pub average_switch_time: f32,
    pub resource_efficiency_score: f32,
}

impl IntelligentModelManager {
    /// Create a new intelligent model manager
    pub async fn new(config: IntelligentModelConfig) -> Result<Self, RecognitionError> {
        let resource_status = Arc::new(RwLock::new(ResourceStatus {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            available_memory_mb: 8192.0,
            gpu_memory_usage: None,
            load_average: 0.0,
            temperature: None,
        }));

        let adaptive_thresholds = Arc::new(RwLock::new(AdaptiveThresholds {
            content_type_thresholds: [
                (AudioContentType::Speech, 0.8),
                (AudioContentType::Music, 0.7),
                (AudioContentType::Noise, 0.6),
                (AudioContentType::Mixed, 0.75),
                (AudioContentType::Silence, 0.9),
            ]
            .iter()
            .cloned()
            .collect(),
            quality_level_thresholds: [
                (AudioQualityLevel::Poor, 0.6),
                (AudioQualityLevel::Fair, 0.7),
                (AudioQualityLevel::Good, 0.8),
                (AudioQualityLevel::Excellent, 0.85),
            ]
            .iter()
            .cloned()
            .collect(),
            adjustment_history: VecDeque::new(),
            learning_rate: config.learning_rate,
            decay_factor: 0.99,
        }));

        let performance_predictor = Arc::new(Mutex::new(ModelPerformancePredictor {
            performance_data: HashMap::new(),
            context_mappings: HashMap::new(),
            prediction_accuracy: 0.5,
        }));

        let model_cache = Arc::new(Mutex::new(IntelligentModelCache {
            loaded_models: HashMap::new(),
            access_patterns: HashMap::new(),
            preloading_strategy: PreloadingStrategy::AdaptiveBased,
            max_cache_size: config.max_loaded_models,
        }));

        // Initialize adaptive processor for content analysis
        let adaptive_config = crate::preprocessing::AdaptiveConfig::default();
        let adaptive_processor = Arc::new(Mutex::new(
            AdaptiveProcessor::new(adaptive_config).map_err(|e| {
                RecognitionError::AudioProcessingError {
                    message: format!("Failed to initialize adaptive processor: {}", e),
                    source: Some(Box::new(e)),
                }
            })?,
        ));

        let usage_statistics = Arc::new(RwLock::new(UsageStatistics::default()));

        let manager = Self {
            config,
            resource_status,
            adaptive_thresholds,
            performance_predictor,
            model_cache,
            adaptive_processor,
            usage_statistics,
        };

        // Start resource monitoring if enabled
        if manager.config.resource_aware_switching {
            manager.start_resource_monitoring().await;
        }

        Ok(manager)
    }

    /// Start system resource monitoring (simplified mock implementation)
    async fn start_resource_monitoring(&self) {
        let resource_status = self.resource_status.clone();
        let interval = self.config.resource_monitoring_interval;

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(Duration::from_secs_f32(interval));
            let mut cpu_counter: f32 = 0.0;

            loop {
                interval_timer.tick().await;

                // Mock system resource monitoring
                // In a real implementation, this would use proper system monitoring
                {
                    let mut status = resource_status.write().await;

                    // Simulate varying CPU usage
                    cpu_counter += 0.1f32;
                    status.cpu_usage = (30.0f32 + 20.0f32 * cpu_counter.sin()).abs();

                    // Simulate memory usage fluctuation
                    status.memory_usage = 60.0f32 + 10.0f32 * (cpu_counter * 0.5f32).cos();
                    status.available_memory_mb =
                        8192.0f32 - (status.memory_usage / 100.0f32 * 8192.0f32);
                    status.load_average = status.cpu_usage / 100.0f32 * 4.0f32;
                }
            }
        });
    }

    /// Analyze audio context for intelligent model selection
    pub async fn analyze_context(
        &self,
        audio: &AudioBuffer,
    ) -> Result<ModelContext, RecognitionError> {
        let start_time = Instant::now();

        // Use adaptive processor to analyze audio content
        let adaptive_result = {
            let mut processor = self.adaptive_processor.lock().await;
            processor.analyze_and_adapt(audio)?
        };

        // Determine audio quality level
        let quality_level = self
            .determine_quality_level(&adaptive_result.features)
            .await;

        // Estimate resource requirements based on audio characteristics
        let resource_requirements = self
            .estimate_resource_requirements(
                &adaptive_result.parameters.content_type,
                &quality_level,
                audio.duration(),
            )
            .await;

        // Determine processing priority
        let priority = self
            .determine_priority(
                &adaptive_result.parameters.content_type,
                &quality_level,
                audio.duration(),
            )
            .await;

        // Estimate processing time
        let expected_processing_time = self
            .estimate_processing_time(
                &adaptive_result.parameters.content_type,
                &quality_level,
                audio.duration(),
            )
            .await;

        let analysis_time = start_time.elapsed().as_secs_f32();
        tracing::debug!("Context analysis completed in {:.3}s", analysis_time);

        Ok(ModelContext {
            content_type: adaptive_result.parameters.content_type,
            quality_level,
            expected_processing_time,
            resource_requirements,
            priority,
        })
    }

    /// Select optimal model based on context and resources
    pub async fn select_optimal_model(
        &self,
        context: &ModelContext,
        available_models: &[String],
    ) -> Result<String, RecognitionError> {
        let resource_status = self.resource_status.read().await;
        let adaptive_thresholds = self.adaptive_thresholds.read().await;

        // Get current quality threshold for this context
        let quality_threshold = adaptive_thresholds
            .content_type_thresholds
            .get(&context.content_type)
            .copied()
            .unwrap_or(0.75);

        // Score each available model
        let mut model_scores = HashMap::new();

        for model_key in available_models {
            let score = self
                .calculate_contextual_score(model_key, context, &resource_status, quality_threshold)
                .await;

            model_scores.insert(model_key.clone(), score);
        }

        // Select highest scoring model
        let best_model = model_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, _)| k.clone())
            .ok_or_else(|| RecognitionError::ModelError {
                message: "No suitable model found".to_string(),
                source: None,
            })?;

        // Update usage statistics
        {
            let mut stats = self.usage_statistics.write().await;
            stats.total_requests += 1;
            *stats
                .context_distribution
                .entry(context.content_type)
                .or_insert(0) += 1;
            *stats
                .quality_distribution
                .entry(context.quality_level.clone())
                .or_insert(0) += 1;
        }

        tracing::debug!(
            "Selected model {} for {:?} with scores: {:?}",
            best_model,
            context.content_type,
            model_scores
        );

        Ok(best_model)
    }

    /// Calculate contextual score for a model
    async fn calculate_contextual_score(
        &self,
        model_key: &str,
        context: &ModelContext,
        resource_status: &ResourceStatus,
        quality_threshold: f32,
    ) -> f32 {
        let mut score = 0.0;

        // Base model performance score
        let base_score = self.get_base_model_score(model_key, &context.content_type);
        score += base_score * 0.3;

        // Resource efficiency score
        let resource_score = self.calculate_resource_efficiency_score(
            model_key,
            &context.resource_requirements,
            resource_status,
        );
        score += resource_score * 0.25;

        // Quality expectation score
        let quality_score = self.calculate_quality_expectation_score(
            model_key,
            &context.quality_level,
            quality_threshold,
        );
        score += quality_score * 0.25;

        // Speed requirement score
        let speed_score = self.calculate_speed_score(
            model_key,
            context.expected_processing_time,
            &context.priority,
        );
        score += speed_score * 0.2;

        score.clamp(0.0, 1.0)
    }

    /// Get base performance score for a model with specific content type
    fn get_base_model_score(&self, model_key: &str, content_type: &AudioContentType) -> f32 {
        // Model-content type performance mapping
        match (model_key, content_type) {
            (key, AudioContentType::Speech) if key.contains("whisper") => {
                if key.contains("large") {
                    0.95
                } else if key.contains("medium") {
                    0.9
                } else if key.contains("small") {
                    0.85
                } else if key.contains("base") {
                    0.8
                } else {
                    0.75
                }
            }
            (key, AudioContentType::Music) if key.contains("whisper") => {
                if key.contains("large") {
                    0.8
                } else if key.contains("medium") {
                    0.75
                } else {
                    0.7
                }
            }
            (key, AudioContentType::Speech) if key.contains("deepspeech") => 0.85,
            (key, AudioContentType::Speech) if key.contains("wav2vec2") => 0.88,
            (_, AudioContentType::Noise) => 0.4,
            (_, AudioContentType::Silence) => 0.9,
            _ => 0.6, // Default score
        }
    }

    /// Calculate resource efficiency score
    fn calculate_resource_efficiency_score(
        &self,
        _model_key: &str,
        requirements: &ResourceRequirements,
        status: &ResourceStatus,
    ) -> f32 {
        // Memory efficiency
        let memory_efficiency = if status.available_memory_mb > requirements.memory_mb * 2.0 {
            1.0
        } else if status.available_memory_mb > requirements.memory_mb {
            0.8
        } else {
            0.3
        };

        // CPU efficiency
        let cpu_efficiency = if status.cpu_usage < 50.0 {
            1.0 - (requirements.cpu_intensity * 0.2)
        } else if status.cpu_usage < 80.0 {
            0.8 - (requirements.cpu_intensity * 0.3)
        } else {
            0.5 - (requirements.cpu_intensity * 0.4)
        };

        // GPU efficiency (if applicable)
        let gpu_efficiency = if let (Some(gpu_req), Some(gpu_usage)) =
            (requirements.gpu_requirement, status.gpu_memory_usage)
        {
            if gpu_usage < 50.0 {
                1.0 - (gpu_req * 0.1)
            } else {
                0.7 - (gpu_req * 0.2)
            }
        } else {
            1.0 // No GPU requirement
        };

        (memory_efficiency + cpu_efficiency + gpu_efficiency) / 3.0
    }

    /// Calculate quality expectation score
    fn calculate_quality_expectation_score(
        &self,
        model_key: &str,
        quality_level: &AudioQualityLevel,
        threshold: f32,
    ) -> f32 {
        let expected_performance = match (model_key, quality_level) {
            (key, AudioQualityLevel::Excellent) if key.contains("large") => 0.95,
            (key, AudioQualityLevel::Excellent) if key.contains("medium") => 0.9,
            (key, AudioQualityLevel::Good) if key.contains("base") => 0.85,
            (key, AudioQualityLevel::Fair) if key.contains("small") => 0.8,
            (key, AudioQualityLevel::Poor) if key.contains("tiny") => 0.7,
            _ => 0.75,
        };

        // Score based on whether expected performance meets threshold
        if expected_performance >= threshold {
            expected_performance
        } else {
            expected_performance * 0.7 // Penalty for not meeting threshold
        }
    }

    /// Calculate speed requirement score
    fn calculate_speed_score(
        &self,
        model_key: &str,
        _expected_time: f32,
        priority: &ProcessingPriority,
    ) -> f32 {
        let model_speed_factor: f32 = if model_key.contains("tiny") {
            1.0
        } else if model_key.contains("small") {
            0.8
        } else if model_key.contains("base") {
            0.6
        } else if model_key.contains("medium") {
            0.4
        } else if model_key.contains("large") {
            0.2
        } else {
            0.5
        };

        let priority_multiplier = match priority {
            ProcessingPriority::Critical => 2.0,
            ProcessingPriority::High => 1.5,
            ProcessingPriority::Normal => 1.0,
            ProcessingPriority::Low => 0.8,
        };

        (model_speed_factor * priority_multiplier).min(1.0f32)
    }

    /// Determine audio quality level from features
    async fn determine_quality_level(&self, features: &AudioFeatures) -> AudioQualityLevel {
        match features.snr_db {
            x if x < 10.0 => AudioQualityLevel::Poor,
            x if x < 20.0 => AudioQualityLevel::Fair,
            x if x < 30.0 => AudioQualityLevel::Good,
            _ => AudioQualityLevel::Excellent,
        }
    }

    /// Estimate resource requirements for given context
    async fn estimate_resource_requirements(
        &self,
        content_type: &AudioContentType,
        quality_level: &AudioQualityLevel,
        duration: f32,
    ) -> ResourceRequirements {
        let base_memory = match content_type {
            AudioContentType::Speech => 512.0,
            AudioContentType::Music => 768.0,
            AudioContentType::Mixed => 640.0,
            AudioContentType::Noise => 256.0,
            AudioContentType::Silence => 128.0,
        };

        let quality_multiplier = match quality_level {
            AudioQualityLevel::Poor => 1.2, // More processing needed
            AudioQualityLevel::Fair => 1.0,
            AudioQualityLevel::Good => 0.9,
            AudioQualityLevel::Excellent => 0.8,
        };

        let duration_factor = (duration / 10.0).min(2.0).max(0.5);

        ResourceRequirements {
            memory_mb: base_memory * quality_multiplier * duration_factor,
            cpu_intensity: 0.7 * quality_multiplier,
            gpu_requirement: Some(0.5 * quality_multiplier),
            warmup_time_ms: 500.0 * quality_multiplier as f64,
        }
    }

    /// Determine processing priority
    async fn determine_priority(
        &self,
        content_type: &AudioContentType,
        quality_level: &AudioQualityLevel,
        duration: f32,
    ) -> ProcessingPriority {
        match (content_type, quality_level, duration) {
            (AudioContentType::Speech, AudioQualityLevel::Poor, d) if d < 5.0 => {
                ProcessingPriority::High
            }
            (AudioContentType::Speech, _, d) if d < 2.0 => ProcessingPriority::High,
            (AudioContentType::Music, AudioQualityLevel::Excellent, _) => {
                ProcessingPriority::Normal
            }
            (AudioContentType::Noise, _, _) => ProcessingPriority::Low,
            (AudioContentType::Silence, _, _) => ProcessingPriority::Low,
            _ => ProcessingPriority::Normal,
        }
    }

    /// Estimate processing time
    async fn estimate_processing_time(
        &self,
        content_type: &AudioContentType,
        quality_level: &AudioQualityLevel,
        duration: f32,
    ) -> f32 {
        let base_rtf = match content_type {
            AudioContentType::Speech => 0.3,
            AudioContentType::Music => 0.4,
            AudioContentType::Mixed => 0.35,
            AudioContentType::Noise => 0.2,
            AudioContentType::Silence => 0.1,
        };

        let quality_factor = match quality_level {
            AudioQualityLevel::Poor => 1.3,
            AudioQualityLevel::Fair => 1.1,
            AudioQualityLevel::Good => 1.0,
            AudioQualityLevel::Excellent => 0.9,
        };

        duration * base_rtf * quality_factor
    }

    /// Update adaptive thresholds based on results
    pub async fn update_adaptive_thresholds(
        &self,
        context: &ModelContext,
        _actual_confidence: f32,
        success: bool,
    ) -> Result<(), RecognitionError> {
        if !self.config.adaptive_quality_thresholds {
            return Ok(());
        }

        let mut thresholds = self.adaptive_thresholds.write().await;

        let current_threshold = thresholds
            .content_type_thresholds
            .get(&context.content_type)
            .copied()
            .unwrap_or(0.75);

        // Adjust threshold based on results
        let adjustment = if success {
            // Success: lower threshold slightly to allow more aggressive optimization
            -thresholds.learning_rate * 0.5
        } else {
            // Failure: raise threshold to be more conservative
            thresholds.learning_rate * 1.5
        };

        let new_threshold = (current_threshold + adjustment).clamp(0.3, 0.95);

        // Record adjustment
        thresholds
            .adjustment_history
            .push_back(ThresholdAdjustment {
                timestamp: Instant::now(),
                content_type: context.content_type,
                old_threshold: current_threshold,
                new_threshold,
                success_rate: if success { 1.0 } else { 0.0 },
            });

        // Keep history bounded
        if thresholds.adjustment_history.len() > 1000 {
            thresholds.adjustment_history.pop_front();
        }

        // Update threshold
        thresholds
            .content_type_thresholds
            .insert(context.content_type, new_threshold);

        tracing::debug!(
            "Updated threshold for {:?}: {:.3} -> {:.3} (success: {})",
            context.content_type,
            current_threshold,
            new_threshold,
            success
        );

        Ok(())
    }

    /// Get current usage statistics
    pub async fn get_usage_statistics(&self) -> UsageStatistics {
        self.usage_statistics.read().await.clone()
    }

    /// Get current resource status
    pub async fn get_resource_status(&self) -> ResourceStatus {
        self.resource_status.read().await.clone()
    }

    /// Get adaptive thresholds
    pub async fn get_adaptive_thresholds(&self) -> HashMap<AudioContentType, f32> {
        self.adaptive_thresholds
            .read()
            .await
            .content_type_thresholds
            .clone()
    }

    /// Reset statistics and thresholds
    pub async fn reset(&self) -> Result<(), RecognitionError> {
        let mut stats = self.usage_statistics.write().await;
        *stats = UsageStatistics::default();

        let mut thresholds = self.adaptive_thresholds.write().await;
        thresholds.adjustment_history.clear();

        // Reset to default thresholds
        thresholds.content_type_thresholds = [
            (AudioContentType::Speech, 0.8),
            (AudioContentType::Music, 0.7),
            (AudioContentType::Noise, 0.6),
            (AudioContentType::Mixed, 0.75),
            (AudioContentType::Silence, 0.9),
        ]
        .iter()
        .cloned()
        .collect();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessing::AdaptiveConfig;

    #[tokio::test]
    async fn test_intelligent_model_manager_creation() {
        let config = IntelligentModelConfig::default();
        let manager = IntelligentModelManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_context_analysis() {
        let config = IntelligentModelConfig::default();
        let manager = IntelligentModelManager::new(config).await.unwrap();

        let samples = vec![0.1; 16000]; // 1 second of test audio
        let audio = AudioBuffer::mono(samples, 16000);

        let context = manager.analyze_context(&audio).await;
        assert!(context.is_ok());

        let context = context.unwrap();
        assert!(matches!(
            context.priority,
            ProcessingPriority::Normal | ProcessingPriority::High | ProcessingPriority::Low
        ));
    }

    #[tokio::test]
    async fn test_model_selection() {
        let config = IntelligentModelConfig::default();
        let manager = IntelligentModelManager::new(config).await.unwrap();

        let context = ModelContext {
            content_type: AudioContentType::Speech,
            quality_level: AudioQualityLevel::Good,
            expected_processing_time: 2.0,
            resource_requirements: ResourceRequirements {
                memory_mb: 512.0,
                cpu_intensity: 0.7,
                gpu_requirement: Some(0.5),
                warmup_time_ms: 500.0,
            },
            priority: ProcessingPriority::Normal,
        };

        let available_models = vec![
            "whisper_tiny".to_string(),
            "whisper_base".to_string(),
            "whisper_small".to_string(),
        ];

        let selected = manager
            .select_optimal_model(&context, &available_models)
            .await;
        assert!(selected.is_ok());
        assert!(available_models.contains(&selected.unwrap()));
    }

    #[test]
    fn test_resource_efficiency_calculation() {
        let config = IntelligentModelConfig::default();
        let manager = IntelligentModelManager {
            config,
            resource_status: Arc::new(RwLock::new(ResourceStatus {
                cpu_usage: 30.0,
                memory_usage: 60.0,
                available_memory_mb: 4096.0,
                gpu_memory_usage: Some(40.0),
                load_average: 1.5,
                temperature: None,
            })),
            adaptive_thresholds: Arc::new(RwLock::new(AdaptiveThresholds {
                content_type_thresholds: HashMap::new(),
                quality_level_thresholds: HashMap::new(),
                adjustment_history: VecDeque::new(),
                learning_rate: 0.01,
                decay_factor: 0.99,
            })),
            performance_predictor: Arc::new(Mutex::new(ModelPerformancePredictor {
                performance_data: HashMap::new(),
                context_mappings: HashMap::new(),
                prediction_accuracy: 0.5,
            })),
            model_cache: Arc::new(Mutex::new(IntelligentModelCache {
                loaded_models: HashMap::new(),
                access_patterns: HashMap::new(),
                preloading_strategy: PreloadingStrategy::AdaptiveBased,
                max_cache_size: 3,
            })),
            adaptive_processor: Arc::new(Mutex::new(
                AdaptiveProcessor::new(AdaptiveConfig::default()).unwrap(),
            )),
            usage_statistics: Arc::new(RwLock::new(UsageStatistics::default())),
        };

        let requirements = ResourceRequirements {
            memory_mb: 1024.0,
            cpu_intensity: 0.5,
            gpu_requirement: Some(0.3),
            warmup_time_ms: 500.0,
        };

        let status = ResourceStatus {
            cpu_usage: 30.0,
            memory_usage: 60.0,
            available_memory_mb: 4096.0,
            gpu_memory_usage: Some(40.0),
            load_average: 1.5,
            temperature: None,
        };

        let score =
            manager.calculate_resource_efficiency_score("whisper_base", &requirements, &status);
        assert!(score >= 0.0 && score <= 1.0);
    }
}
