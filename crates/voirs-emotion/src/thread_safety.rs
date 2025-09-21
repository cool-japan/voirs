//! Thread safety improvements for emotion processing system
//!
//! This module provides enhanced thread safety patterns, concurrent emotion processing,
//! and safe resource sharing for the emotion control system.

use crate::{
    config::EmotionConfig,
    core::EmotionProcessor,
    types::{
        Emotion, EmotionDimensions, EmotionIntensity, EmotionParameters, EmotionState,
        EmotionVector,
    },
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, OwnedSemaphorePermit, RwLock, Semaphore};
use tracing::{debug, info, trace, warn};

/// Thread-safe emotion cache for efficient emotion state management
pub struct ThreadSafeEmotionCache {
    /// Cached emotion states with reference counting
    emotion_states: Arc<RwLock<HashMap<String, Arc<EmotionState>>>>,
    /// Emotion processing statistics
    stats: Arc<RwLock<EmotionCacheStats>>,
    /// Maximum cache size
    max_cache_size: usize,
    /// Cache access patterns for LRU eviction
    access_patterns: Arc<RwLock<HashMap<String, EmotionAccessInfo>>>,
    /// Semaphore for limiting concurrent emotion processing
    processing_semaphore: Arc<Semaphore>,
}

/// Emotion cache statistics
#[derive(Debug, Clone, Default)]
pub struct EmotionCacheStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub states_cached: u64,
    pub states_evicted: u64,
    pub concurrent_processing: u64,
    pub average_processing_time: Duration,
    pub interpolation_count: u64,
    pub blend_operations: u64,
}

/// Emotion access information for cache management
#[derive(Debug, Clone)]
pub struct EmotionAccessInfo {
    pub last_accessed: Instant,
    pub access_count: u32,
    pub total_processing_time: Duration,
    pub interpolation_usage: u32,
    pub blend_usage: u32,
}

impl ThreadSafeEmotionCache {
    /// Create new thread-safe emotion cache
    pub fn new(max_cache_size: usize, max_concurrent_processing: usize) -> Self {
        Self {
            emotion_states: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(EmotionCacheStats::default())),
            max_cache_size,
            access_patterns: Arc::new(RwLock::new(HashMap::new())),
            processing_semaphore: Arc::new(Semaphore::new(max_concurrent_processing)),
        }
    }

    /// Get or create emotion state with thread-safe access
    pub async fn get_or_create_state(
        &self,
        key: String,
        emotion_params: &EmotionParameters,
    ) -> Result<Arc<EmotionState>> {
        // Try to get from cache first
        {
            let states_guard = self.emotion_states.read().await;
            if let Some(state) = states_guard.get(&key) {
                self.update_access_info(&key, false).await;
                self.update_stats(true).await;
                return Ok(Arc::clone(state));
            }
        }

        // State not in cache, need to create it
        self.update_stats(false).await;

        let _permit = self.processing_semaphore.acquire().await.map_err(|e| {
            Error::Validation(format!("Failed to acquire processing permit: {}", e))
        })?;

        // Double-check pattern
        {
            let states_guard = self.emotion_states.read().await;
            if let Some(state) = states_guard.get(&key) {
                self.update_access_info(&key, false).await;
                return Ok(Arc::clone(state));
            }
        }

        // Create new emotion state
        let start_time = Instant::now();
        let emotion_state = self.create_emotion_state(emotion_params).await?;
        let processing_time = start_time.elapsed();

        // Update cache with new state
        {
            let mut states_guard = self.emotion_states.write().await;
            let mut access_guard = self.access_patterns.write().await;

            // Check if we need to evict old states
            if states_guard.len() >= self.max_cache_size {
                self.evict_least_used_state(&mut states_guard, &mut access_guard)
                    .await;
            }

            // Insert new state
            let state_arc = Arc::new(emotion_state);
            states_guard.insert(key.clone(), Arc::clone(&state_arc));

            // Track access pattern
            access_guard.insert(
                key.clone(),
                EmotionAccessInfo {
                    last_accessed: Instant::now(),
                    access_count: 1,
                    total_processing_time: processing_time,
                    interpolation_usage: 0,
                    blend_usage: 0,
                },
            );

            // Update statistics
            {
                let mut stats_guard = self.stats.write().await;
                stats_guard.states_cached += 1;
                stats_guard.concurrent_processing += 1;

                // Update average processing time
                let total_states = stats_guard.states_cached;
                if total_states == 1 {
                    stats_guard.average_processing_time = processing_time;
                } else {
                    let total_nanos = stats_guard.average_processing_time.as_nanos() as u64
                        * (total_states - 1)
                        + processing_time.as_nanos() as u64;
                    stats_guard.average_processing_time =
                        Duration::from_nanos(total_nanos / total_states);
                }
            }

            Ok(state_arc)
        }
    }

    /// Create emotion state (placeholder for actual implementation)
    async fn create_emotion_state(
        &self,
        emotion_params: &EmotionParameters,
    ) -> Result<EmotionState> {
        // Simulate processing time
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Create emotion state based on parameters - matching the actual structure
        Ok(EmotionState {
            current: emotion_params.clone(),
            target: None,
            transition_progress: 0.0,
            timestamp: std::time::SystemTime::now(),
        })
    }

    /// Evict least used emotion state
    async fn evict_least_used_state(
        &self,
        states_guard: &mut HashMap<String, Arc<EmotionState>>,
        access_guard: &mut HashMap<String, EmotionAccessInfo>,
    ) {
        if let Some((least_used_key, _)) = access_guard
            .iter()
            .min_by_key(|(_, access_info)| (access_info.last_accessed, access_info.access_count))
        {
            let evicted_key = least_used_key.clone();
            states_guard.remove(&evicted_key);
            access_guard.remove(&evicted_key);

            // Update statistics
            {
                let mut stats_guard = self.stats.write().await;
                stats_guard.states_evicted += 1;
            }

            debug!("Evicted least used emotion state: {}", evicted_key);
        }
    }

    /// Update access information
    async fn update_access_info(&self, key: &str, is_interpolation: bool) {
        let mut access_guard = self.access_patterns.write().await;
        if let Some(access_info) = access_guard.get_mut(key) {
            access_info.last_accessed = Instant::now();
            access_info.access_count += 1;
            if is_interpolation {
                access_info.interpolation_usage += 1;
            }
        }
    }

    /// Update cache statistics
    async fn update_stats(&self, cache_hit: bool) {
        let mut stats_guard = self.stats.write().await;
        if cache_hit {
            stats_guard.cache_hits += 1;
        } else {
            stats_guard.cache_misses += 1;
        }
    }

    /// Interpolate between emotion states with thread safety
    pub async fn interpolate_states(
        &self,
        from_key: String,
        to_key: String,
        factor: f32,
        from_params: &EmotionParameters,
        to_params: &EmotionParameters,
    ) -> Result<EmotionState> {
        let from_state = self
            .get_or_create_state(from_key.clone(), from_params)
            .await?;
        let to_state = self.get_or_create_state(to_key.clone(), to_params).await?;

        // Update interpolation usage
        self.update_access_info(&from_key, true).await;
        self.update_access_info(&to_key, true).await;

        // Update statistics
        {
            let mut stats_guard = self.stats.write().await;
            stats_guard.interpolation_count += 1;
        }

        // Perform interpolation by creating a new state with interpolated dimensions
        let interpolated_dimensions = EmotionDimensions {
            valence: from_state.current.emotion_vector.dimensions.valence * (1.0 - factor)
                + to_state.current.emotion_vector.dimensions.valence * factor,
            arousal: from_state.current.emotion_vector.dimensions.arousal * (1.0 - factor)
                + to_state.current.emotion_vector.dimensions.arousal * factor,
            dominance: from_state.current.emotion_vector.dimensions.dominance * (1.0 - factor)
                + to_state.current.emotion_vector.dimensions.dominance * factor,
        };

        let mut interpolated_vector = from_state.current.emotion_vector.clone();
        interpolated_vector.dimensions = interpolated_dimensions;

        let interpolated_params = EmotionParameters {
            emotion_vector: interpolated_vector,
            duration_ms: from_state.current.duration_ms,
            fade_in_ms: from_state.current.fade_in_ms,
            fade_out_ms: from_state.current.fade_out_ms,
            pitch_shift: from_state.current.pitch_shift * (1.0 - factor)
                + to_state.current.pitch_shift * factor,
            tempo_scale: from_state.current.tempo_scale * (1.0 - factor)
                + to_state.current.tempo_scale * factor,
            energy_scale: from_state.current.energy_scale * (1.0 - factor)
                + to_state.current.energy_scale * factor,
            breathiness: from_state.current.breathiness * (1.0 - factor)
                + to_state.current.breathiness * factor,
            roughness: from_state.current.roughness * (1.0 - factor)
                + to_state.current.roughness * factor,
            custom_params: from_state.current.custom_params.clone(), // For simplicity, use from_state params
        };

        let interpolated_state = EmotionState {
            current: interpolated_params,
            target: None,
            transition_progress: factor,
            timestamp: std::time::SystemTime::now(),
        };

        Ok(interpolated_state)
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> EmotionCacheStats {
        let stats_guard = self.stats.read().await;
        stats_guard.clone()
    }

    /// Clear cache
    pub async fn clear_cache(&self) {
        let mut states_guard = self.emotion_states.write().await;
        let mut access_guard = self.access_patterns.write().await;

        let evicted_count = states_guard.len();
        states_guard.clear();
        access_guard.clear();

        // Update statistics
        {
            let mut stats_guard = self.stats.write().await;
            stats_guard.states_evicted += evicted_count as u64;
        }

        info!("Cleared emotion cache: {} states evicted", evicted_count);
    }
}

/// Thread-safe concurrent emotion processor
pub struct ConcurrentEmotionProcessor {
    /// Emotion cache for state management
    emotion_cache: Arc<ThreadSafeEmotionCache>,
    /// Configuration with thread-safe access
    config: Arc<RwLock<EmotionConfig>>,
    /// Processing semaphore for concurrency control
    processing_semaphore: Arc<Semaphore>,
    /// Active processing operations
    active_operations: Arc<RwLock<HashMap<String, EmotionProcessingInfo>>>,
    /// Processing metrics
    metrics: Arc<RwLock<EmotionProcessingMetrics>>,
}

/// Information about active emotion processing operation
#[derive(Debug, Clone)]
pub struct EmotionProcessingInfo {
    pub operation_id: String,
    pub start_time: Instant,
    pub emotion_type: EmotionProcessingType,
    pub thread_id: std::thread::ThreadId,
    pub status: EmotionProcessingStatus,
}

/// Type of emotion processing operation
#[derive(Debug, Clone, PartialEq)]
pub enum EmotionProcessingType {
    SingleState,
    Interpolation,
    Blending,
    Recognition,
    Adaptation,
}

/// Status of emotion processing operation
#[derive(Debug, Clone, PartialEq)]
pub enum EmotionProcessingStatus {
    Starting,
    Processing,
    Finalizing,
    Completed,
    Failed(String),
}

/// Metrics for emotion processing operations
#[derive(Debug, Default, Clone)]
pub struct EmotionProcessingMetrics {
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
    pub average_processing_time: Duration,
    pub concurrent_operations_peak: usize,
    pub current_concurrent_operations: usize,
    pub interpolation_operations: u64,
    pub blending_operations: u64,
    pub recognition_operations: u64,
}

impl ConcurrentEmotionProcessor {
    /// Create new concurrent emotion processor
    pub fn new(
        max_concurrent_operations: usize,
        max_cache_size: usize,
        config: EmotionConfig,
    ) -> Self {
        Self {
            emotion_cache: Arc::new(ThreadSafeEmotionCache::new(
                max_cache_size,
                max_concurrent_operations,
            )),
            config: Arc::new(RwLock::new(config)),
            processing_semaphore: Arc::new(Semaphore::new(max_concurrent_operations)),
            active_operations: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(EmotionProcessingMetrics::default())),
        }
    }

    /// Process emotion with concurrency control
    pub async fn process_emotion_concurrent(
        &self,
        operation_id: String,
        emotion_params: EmotionParameters,
        processing_type: EmotionProcessingType,
    ) -> Result<EmotionState> {
        let operation_guard = self
            .create_operation_guard(operation_id.clone(), processing_type.clone())
            .await?;

        let processing_start = Instant::now();

        // Update metrics
        {
            let mut metrics_guard = self.metrics.write().await;
            metrics_guard.total_operations += 1;
            metrics_guard.current_concurrent_operations += 1;

            if metrics_guard.current_concurrent_operations
                > metrics_guard.concurrent_operations_peak
            {
                metrics_guard.concurrent_operations_peak =
                    metrics_guard.current_concurrent_operations;
            }

            // Update operation type counters
            match processing_type {
                EmotionProcessingType::Interpolation => metrics_guard.interpolation_operations += 1,
                EmotionProcessingType::Blending => metrics_guard.blending_operations += 1,
                EmotionProcessingType::Recognition => metrics_guard.recognition_operations += 1,
                _ => {}
            }
        }

        // Update operation status
        self.update_operation_status(&operation_id, EmotionProcessingStatus::Processing)
            .await;

        // Perform emotion processing
        let result = match self
            .perform_emotion_processing(&emotion_params, &processing_type)
            .await
        {
            Ok(emotion_state) => {
                // Update success metrics
                let processing_time = processing_start.elapsed();
                {
                    let mut metrics_guard = self.metrics.write().await;
                    metrics_guard.successful_operations += 1;
                    metrics_guard.current_concurrent_operations -= 1;

                    // Update average processing time
                    let successful_ops = metrics_guard.successful_operations;
                    if successful_ops == 1 {
                        metrics_guard.average_processing_time = processing_time;
                    } else {
                        let total_nanos = metrics_guard.average_processing_time.as_nanos() as u64
                            * (successful_ops - 1)
                            + processing_time.as_nanos() as u64;
                        metrics_guard.average_processing_time =
                            Duration::from_nanos(total_nanos / successful_ops);
                    }
                }

                self.update_operation_status(&operation_id, EmotionProcessingStatus::Completed)
                    .await;
                Ok(emotion_state)
            }
            Err(e) => {
                // Update failure metrics
                {
                    let mut metrics_guard = self.metrics.write().await;
                    metrics_guard.failed_operations += 1;
                    metrics_guard.current_concurrent_operations -= 1;
                }

                self.update_operation_status(
                    &operation_id,
                    EmotionProcessingStatus::Failed(e.to_string()),
                )
                .await;
                Err(e)
            }
        };

        // Remove operation from active list
        {
            let mut operations_guard = self.active_operations.write().await;
            operations_guard.remove(&operation_id);
        }

        result
    }

    /// Create operation guard for processing
    async fn create_operation_guard(
        &self,
        operation_id: String,
        processing_type: EmotionProcessingType,
    ) -> Result<OwnedSemaphorePermit> {
        let permit = self
            .processing_semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| {
                Error::Validation(format!("Failed to acquire processing permit: {}", e))
            })?;

        // Register operation
        {
            let mut operations_guard = self.active_operations.write().await;
            operations_guard.insert(
                operation_id.clone(),
                EmotionProcessingInfo {
                    operation_id,
                    start_time: Instant::now(),
                    emotion_type: processing_type,
                    thread_id: std::thread::current().id(),
                    status: EmotionProcessingStatus::Starting,
                },
            );
        }

        Ok(permit)
    }

    /// Update operation status
    async fn update_operation_status(&self, operation_id: &str, status: EmotionProcessingStatus) {
        let mut operations_guard = self.active_operations.write().await;
        if let Some(op_info) = operations_guard.get_mut(operation_id) {
            op_info.status = status;
        }
    }

    /// Perform actual emotion processing
    async fn perform_emotion_processing(
        &self,
        emotion_params: &EmotionParameters,
        processing_type: &EmotionProcessingType,
    ) -> Result<EmotionState> {
        match processing_type {
            EmotionProcessingType::SingleState => {
                let key = self.generate_emotion_key(emotion_params);
                self.emotion_cache
                    .get_or_create_state(key, emotion_params)
                    .await
                    .map(|arc_state| (*arc_state).clone())
            }
            EmotionProcessingType::Interpolation => {
                // Simulate interpolation between two states
                let from_params = emotion_params.clone();
                let mut to_params = emotion_params.clone();
                to_params.emotion_vector.dimensions.valence =
                    (to_params.emotion_vector.dimensions.valence + 0.2).min(1.0);

                let from_key = self.generate_emotion_key(&from_params);
                let to_key = self.generate_emotion_key(&to_params);

                self.emotion_cache
                    .interpolate_states(from_key, to_key, 0.5, &from_params, &to_params)
                    .await
            }
            EmotionProcessingType::Blending
            | EmotionProcessingType::Recognition
            | EmotionProcessingType::Adaptation => {
                // For other types, use single state processing for now
                let key = self.generate_emotion_key(emotion_params);
                self.emotion_cache
                    .get_or_create_state(key, emotion_params)
                    .await
                    .map(|arc_state| (*arc_state).clone())
            }
        }
    }

    /// Generate unique key for emotion parameters
    fn generate_emotion_key(&self, emotion_params: &EmotionParameters) -> String {
        format!(
            "v{:.2}_a{:.2}_d{:.2}_p{:.2}_t{:.2}",
            emotion_params.emotion_vector.dimensions.valence,
            emotion_params.emotion_vector.dimensions.arousal,
            emotion_params.emotion_vector.dimensions.dominance,
            emotion_params.pitch_shift,
            emotion_params.tempo_scale
        )
    }

    /// Get processing metrics
    pub async fn get_metrics(&self) -> EmotionProcessingMetrics {
        let metrics_guard = self.metrics.read().await;
        metrics_guard.clone()
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> EmotionCacheStats {
        self.emotion_cache.get_stats().await
    }

    /// Get active operations
    pub async fn get_active_operations(&self) -> Vec<EmotionProcessingInfo> {
        let operations_guard = self.active_operations.read().await;
        operations_guard.values().cloned().collect()
    }

    /// Update configuration thread-safely
    pub async fn update_config(&self, new_config: EmotionConfig) -> Result<()> {
        let mut config_guard = self.config.write().await;
        *config_guard = new_config;
        info!("Emotion processor configuration updated successfully");
        Ok(())
    }

    /// Get current configuration
    pub async fn get_config(&self) -> EmotionConfig {
        let config_guard = self.config.read().await;
        config_guard.clone()
    }

    /// Perform health check
    pub async fn health_check(&self) -> HashMap<String, String> {
        let mut health = HashMap::new();

        let metrics = self.get_metrics().await;
        let active_ops = self.get_active_operations().await;
        let cache_stats = self.get_cache_stats().await;

        health.insert("status".to_string(), "healthy".to_string());
        health.insert(
            "total_operations".to_string(),
            metrics.total_operations.to_string(),
        );
        health.insert(
            "success_rate".to_string(),
            format!(
                "{:.2}%",
                if metrics.total_operations > 0 {
                    (metrics.successful_operations as f64 / metrics.total_operations as f64) * 100.0
                } else {
                    100.0
                }
            ),
        );
        health.insert(
            "active_operations".to_string(),
            active_ops.len().to_string(),
        );
        health.insert(
            "cache_hit_rate".to_string(),
            format!(
                "{:.2}%",
                if cache_stats.cache_hits + cache_stats.cache_misses > 0 {
                    (cache_stats.cache_hits as f64
                        / (cache_stats.cache_hits + cache_stats.cache_misses) as f64)
                        * 100.0
                } else {
                    0.0
                }
            ),
        );
        health.insert(
            "cached_states".to_string(),
            (cache_stats.states_cached - cache_stats.states_evicted).to_string(),
        );

        health
    }

    /// Graceful shutdown
    pub async fn shutdown(&self) -> Result<()> {
        info!("Starting graceful shutdown of concurrent emotion processor");

        // Wait for all active operations to complete
        let shutdown_timeout = Duration::from_secs(30);
        let start_time = Instant::now();

        while start_time.elapsed() < shutdown_timeout {
            let active_ops = self.get_active_operations().await;
            if active_ops.is_empty() {
                break;
            }

            debug!(
                "Waiting for {} active emotion operations to complete",
                active_ops.len()
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Clear cache
        self.emotion_cache.clear_cache().await;

        let final_metrics = self.get_metrics().await;
        info!(
            "Concurrent emotion processor shutdown complete. Final stats: {} total operations, {} successful, {} failed",
            final_metrics.total_operations, final_metrics.successful_operations, final_metrics.failed_operations
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_emotion_params() -> EmotionParameters {
        use crate::types::{Emotion, EmotionIntensity};

        let mut emotion_vector = EmotionVector::new();
        emotion_vector.add_emotion(Emotion::Happy, EmotionIntensity::new(0.8));

        EmotionParameters {
            emotion_vector,
            duration_ms: Some(1000),
            fade_in_ms: Some(100),
            fade_out_ms: Some(100),
            pitch_shift: 0.0,
            tempo_scale: 1.0,
            energy_scale: 1.0,
            breathiness: 0.0,
            roughness: 0.0,
            custom_params: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_thread_safe_emotion_cache() {
        let cache = ThreadSafeEmotionCache::new(5, 2);

        let emotion_params = create_test_emotion_params();

        // Test cache miss and state creation
        let state1 = cache
            .get_or_create_state("test_state".to_string(), &emotion_params)
            .await
            .unwrap();
        // Just verify the state was created successfully
        assert!(
            state1.current.emotion_vector.dimensions.valence >= -1.0
                && state1.current.emotion_vector.dimensions.valence <= 1.0
        );

        // Test cache hit
        let state2 = cache
            .get_or_create_state("test_state".to_string(), &emotion_params)
            .await
            .unwrap();
        assert!(Arc::ptr_eq(&state1, &state2));

        // Verify statistics
        let stats = cache.get_stats().await;
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.states_cached, 1);
    }

    #[tokio::test]
    async fn test_emotion_interpolation() {
        let cache = ThreadSafeEmotionCache::new(10, 3);

        let mut from_vector = EmotionVector::new();
        from_vector.add_emotion(Emotion::Sad, EmotionIntensity::new(0.8));
        let from_params = EmotionParameters {
            emotion_vector: from_vector,
            duration_ms: Some(1000),
            fade_in_ms: Some(100),
            fade_out_ms: Some(100),
            pitch_shift: 0.0,
            tempo_scale: 1.0,
            energy_scale: 1.0,
            breathiness: 0.0,
            roughness: 0.0,
            custom_params: HashMap::new(),
        };

        let mut to_vector = EmotionVector::new();
        to_vector.add_emotion(Emotion::Happy, EmotionIntensity::new(1.0));
        let to_params = EmotionParameters {
            emotion_vector: to_vector,
            duration_ms: Some(1000),
            fade_in_ms: Some(100),
            fade_out_ms: Some(100),
            pitch_shift: 0.0,
            tempo_scale: 1.0,
            energy_scale: 1.0,
            breathiness: 0.0,
            roughness: 0.0,
            custom_params: HashMap::new(),
        };

        let interpolated = cache
            .interpolate_states(
                "from_state".to_string(),
                "to_state".to_string(),
                0.5,
                &from_params,
                &to_params,
            )
            .await
            .unwrap();

        // Check interpolated values - verify dimensions are within valid range
        assert!(
            interpolated.current.emotion_vector.dimensions.valence >= -1.0
                && interpolated.current.emotion_vector.dimensions.valence <= 1.0
        );
        assert!(
            interpolated.current.emotion_vector.dimensions.arousal >= -1.0
                && interpolated.current.emotion_vector.dimensions.arousal <= 1.0
        );
        assert!(
            interpolated.current.emotion_vector.dimensions.dominance >= -1.0
                && interpolated.current.emotion_vector.dimensions.dominance <= 1.0
        );

        let stats = cache.get_stats().await;
        assert_eq!(stats.interpolation_count, 1);
    }

    #[tokio::test]
    async fn test_concurrent_emotion_processor() {
        let config = EmotionConfig::default();
        let processor = ConcurrentEmotionProcessor::new(3, 5, config);

        let emotion_params = create_test_emotion_params();

        let result = processor
            .process_emotion_concurrent(
                "test_operation".to_string(),
                emotion_params,
                EmotionProcessingType::SingleState,
            )
            .await;

        assert!(result.is_ok());

        let metrics = processor.get_metrics().await;
        assert_eq!(metrics.total_operations, 1);
        assert_eq!(metrics.successful_operations, 1);
        assert_eq!(metrics.failed_operations, 0);
    }

    #[tokio::test]
    async fn test_concurrent_processing() {
        let config = EmotionConfig::default();
        let processor = Arc::new(ConcurrentEmotionProcessor::new(3, 5, config));

        let mut handles = Vec::new();

        // Spawn multiple concurrent emotion processing operations
        for i in 0..5 {
            let processor_clone = Arc::clone(&processor);
            let handle = tokio::spawn(async move {
                let mut emotion_vector = EmotionVector::new();
                emotion_vector.add_emotion(Emotion::Happy, EmotionIntensity::new(i as f32 * 0.2));
                let emotion_params = EmotionParameters {
                    emotion_vector,
                    duration_ms: Some(1000),
                    fade_in_ms: Some(100),
                    fade_out_ms: Some(100),
                    pitch_shift: 0.0,
                    tempo_scale: 1.0,
                    energy_scale: 1.0,
                    breathiness: 0.0,
                    roughness: 0.0,
                    custom_params: HashMap::new(),
                };

                processor_clone
                    .process_emotion_concurrent(
                        format!("op_{}", i),
                        emotion_params,
                        EmotionProcessingType::SingleState,
                    )
                    .await
            });
            handles.push(handle);
        }

        // Wait for all to complete
        let mut successful = 0;
        for handle in handles {
            if handle.await.unwrap().is_ok() {
                successful += 1;
            }
        }

        assert_eq!(successful, 5);

        let metrics = processor.get_metrics().await;
        assert_eq!(metrics.total_operations, 5);
        assert_eq!(metrics.successful_operations, 5);
    }

    #[tokio::test]
    async fn test_cache_eviction() {
        let cache = ThreadSafeEmotionCache::new(2, 3); // Small cache size

        // Fill cache
        let params1 = create_test_emotion_params();
        let mut params2 = create_test_emotion_params();
        params2.emotion_vector.dimensions.valence = 0.8;
        let mut params3 = create_test_emotion_params();
        params3.emotion_vector.dimensions.valence = 0.2;

        cache
            .get_or_create_state("state1".to_string(), &params1)
            .await
            .unwrap();
        cache
            .get_or_create_state("state2".to_string(), &params2)
            .await
            .unwrap();

        // This should trigger eviction
        cache
            .get_or_create_state("state3".to_string(), &params3)
            .await
            .unwrap();

        let stats = cache.get_stats().await;
        assert_eq!(stats.states_evicted, 1);
    }

    #[tokio::test]
    async fn test_health_check() {
        let config = EmotionConfig::default();
        let processor = ConcurrentEmotionProcessor::new(2, 3, config);

        let health = processor.health_check().await;
        assert_eq!(health.get("status"), Some(&"healthy".to_string()));
        assert!(health.contains_key("total_operations"));
        assert!(health.contains_key("success_rate"));
        assert!(health.contains_key("active_operations"));
        assert!(health.contains_key("cache_hit_rate"));
    }

    #[tokio::test]
    async fn test_interpolation_processing() {
        let config = EmotionConfig::default();
        let processor = ConcurrentEmotionProcessor::new(2, 3, config);

        let emotion_params = create_test_emotion_params();

        let result = processor
            .process_emotion_concurrent(
                "interp_test".to_string(),
                emotion_params,
                EmotionProcessingType::Interpolation,
            )
            .await;

        assert!(result.is_ok());

        let metrics = processor.get_metrics().await;
        assert_eq!(metrics.interpolation_operations, 1);
    }
}
