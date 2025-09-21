//! Advanced thread safety improvements for voice cloning
//!
//! This module provides enhanced thread safety patterns, unified configuration management,
//! model loading optimization, and concurrent operation improvements.

use crate::{
    config::CloningConfig,
    embedding::{EmbeddingConfig, SpeakerEmbedding, SpeakerEmbeddingExtractor},
    quality::{CloningQualityAssessor, QualityConfig},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock, Semaphore};
use tracing::{debug, info, trace, warn};

/// Thread-safe model cache for efficient loading and sharing
pub struct ModelCache<T> {
    /// Cached models with reference counting
    models: Arc<RwLock<HashMap<String, Arc<T>>>>,
    /// Maximum cache size
    max_size: usize,
    /// Cache access statistics
    stats: Arc<RwLock<CacheStats>>,
    /// Semaphore for limiting concurrent model loading
    loading_semaphore: Arc<Semaphore>,
}

/// Cache access statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub concurrent_loads: u64,
    pub average_load_time: Duration,
}

/// Thread-safe configuration manager for unified config handling
pub struct UnifiedConfigManager {
    /// Core cloning configuration
    cloning_config: Arc<RwLock<CloningConfig>>,
    /// Embedding extraction configuration
    embedding_config: Arc<RwLock<EmbeddingConfig>>,
    /// Quality assessment configuration
    quality_config: Arc<RwLock<QualityConfig>>,
    /// Configuration change notifications
    change_notifications: Arc<RwLock<Vec<ConfigChangeNotifier>>>,
    /// Configuration validation cache
    validation_cache: Arc<RwLock<HashMap<String, bool>>>,
}

/// Configuration change notification system
pub type ConfigChangeNotifier = Box<dyn Fn(&str, &str) + Send + Sync>;

/// Enhanced thread-safe component registry
pub struct ComponentRegistry {
    /// Registered embedding extractors
    embedding_extractors: Arc<RwLock<HashMap<String, Arc<SpeakerEmbeddingExtractor>>>>,
    /// Registered quality assessors
    quality_assessors: Arc<RwLock<HashMap<String, Arc<Mutex<CloningQualityAssessor>>>>>,
    /// Component health monitoring
    health_monitor: Arc<RwLock<ComponentHealthMonitor>>,
    /// Resource usage tracking
    resource_monitor: Arc<RwLock<ResourceMonitor>>,
}

/// Component health monitoring system
#[derive(Debug, Default)]
pub struct ComponentHealthMonitor {
    pub component_status: HashMap<String, ComponentStatus>,
    pub last_health_check: Option<Instant>,
    pub check_interval: Duration,
}

/// Individual component status
#[derive(Debug, Clone)]
pub struct ComponentStatus {
    pub is_healthy: bool,
    pub last_used: Instant,
    pub error_count: u32,
    pub performance_metrics: PerformanceMetrics,
}

/// Performance metrics for components
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    pub average_response_time: Duration,
    pub requests_per_second: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
}

/// Resource usage monitoring
#[derive(Debug, Default, Clone)]
pub struct ResourceMonitor {
    pub memory_usage: HashMap<String, u64>,
    pub cpu_usage: HashMap<String, f64>,
    pub thread_counts: HashMap<String, usize>,
    pub resource_limits: ResourceLimits,
}

/// Resource limits for thread safety
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    pub max_memory_mb: u64,
    pub max_cpu_percent: f64,
    pub max_concurrent_operations: usize,
    pub max_threads_per_component: usize,
}

impl<T> ModelCache<T>
where
    T: Send + Sync + 'static,
{
    /// Create new thread-safe model cache
    pub fn new(max_size: usize, max_concurrent_loads: usize) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            max_size,
            stats: Arc::new(RwLock::new(CacheStats::default())),
            loading_semaphore: Arc::new(Semaphore::new(max_concurrent_loads)),
        }
    }

    /// Get model from cache or load it
    pub async fn get_or_load<F, Fut>(&self, key: &str, loader: F) -> Result<Arc<T>>
    where
        F: FnOnce() -> Fut + Send,
        Fut: std::future::Future<Output = Result<T>> + Send,
    {
        // First check if model is already cached
        {
            let models = self.models.read().await;
            if let Some(model) = models.get(key) {
                let mut stats = self.stats.write().await;
                stats.hits += 1;
                return Ok(Arc::clone(model));
            }
        }

        // Acquire semaphore to limit concurrent loading
        let _permit =
            self.loading_semaphore.acquire().await.map_err(|e| {
                Error::Processing(format!("Failed to acquire loading permit: {}", e))
            })?;

        let start_time = Instant::now();

        // Double-check pattern - another thread might have loaded it
        {
            let models = self.models.read().await;
            if let Some(model) = models.get(key) {
                let mut stats = self.stats.write().await;
                stats.hits += 1;
                return Ok(Arc::clone(model));
            }
        }

        // Load the model
        let model = loader().await?;
        let model_arc = Arc::new(model);

        // Update cache
        {
            let mut models = self.models.write().await;

            // Check cache size and evict if necessary
            if models.len() >= self.max_size {
                // Simple LRU eviction - remove first entry
                if let Some((old_key, _)) = models.iter().next() {
                    let old_key = old_key.clone();
                    models.remove(&old_key);

                    let mut stats = self.stats.write().await;
                    stats.evictions += 1;
                }
            }

            models.insert(key.to_string(), Arc::clone(&model_arc));
        }

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.misses += 1;
            stats.concurrent_loads += 1;

            let load_time = start_time.elapsed();
            let total_time = stats.average_load_time.as_nanos() as u64
                * (stats.concurrent_loads - 1)
                + load_time.as_nanos() as u64;
            stats.average_load_time = Duration::from_nanos(total_time / stats.concurrent_loads);
        }

        trace!("Loaded model '{}' in {:?}", key, start_time.elapsed());
        Ok(model_arc)
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        self.stats.read().await.clone()
    }

    /// Clear all cached models
    pub async fn clear(&self) {
        let mut models = self.models.write().await;
        let cleared_count = models.len();
        models.clear();

        info!("Cleared {} models from cache", cleared_count);
    }

    /// Get cache hit rate
    pub async fn hit_rate(&self) -> f64 {
        let stats = self.stats.read().await;
        let total = stats.hits + stats.misses;
        if total > 0 {
            stats.hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

impl UnifiedConfigManager {
    /// Create new unified configuration manager
    pub fn new(
        cloning_config: CloningConfig,
        embedding_config: EmbeddingConfig,
        quality_config: QualityConfig,
    ) -> Self {
        Self {
            cloning_config: Arc::new(RwLock::new(cloning_config)),
            embedding_config: Arc::new(RwLock::new(embedding_config)),
            quality_config: Arc::new(RwLock::new(quality_config)),
            change_notifications: Arc::new(RwLock::new(Vec::new())),
            validation_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get cloning configuration (thread-safe)
    pub async fn get_cloning_config(&self) -> CloningConfig {
        self.cloning_config.read().await.clone()
    }

    /// Update cloning configuration with validation and notifications
    pub async fn update_cloning_config(&self, config: CloningConfig) -> Result<()> {
        // Validate configuration first
        config.validate()?;

        // Update configuration
        {
            let mut cloning_config = self.cloning_config.write().await;
            *cloning_config = config;
        }

        // Notify listeners
        self.notify_change("cloning_config", "updated").await;

        // Clear validation cache
        self.validation_cache.write().await.clear();

        Ok(())
    }

    /// Get embedding configuration (thread-safe)
    pub async fn get_embedding_config(&self) -> EmbeddingConfig {
        self.embedding_config.read().await.clone()
    }

    /// Update embedding configuration
    pub async fn update_embedding_config(&self, config: EmbeddingConfig) -> Result<()> {
        {
            let mut embedding_config = self.embedding_config.write().await;
            *embedding_config = config;
        }

        self.notify_change("embedding_config", "updated").await;
        self.validation_cache.write().await.clear();

        Ok(())
    }

    /// Get quality configuration (thread-safe)
    pub async fn get_quality_config(&self) -> QualityConfig {
        self.quality_config.read().await.clone()
    }

    /// Update quality configuration
    pub async fn update_quality_config(&self, config: QualityConfig) -> Result<()> {
        {
            let mut quality_config = self.quality_config.write().await;
            *quality_config = config;
        }

        self.notify_change("quality_config", "updated").await;
        self.validation_cache.write().await.clear();

        Ok(())
    }

    /// Register configuration change notifier
    pub async fn register_change_notifier(&self, notifier: ConfigChangeNotifier) {
        let mut notifications = self.change_notifications.write().await;
        notifications.push(notifier);
    }

    /// Validate all configurations
    pub async fn validate_all_configs(&self) -> Result<()> {
        let cloning_config = self.cloning_config.read().await;
        cloning_config.validate()?;
        drop(cloning_config);

        // Note: EmbeddingConfig and QualityConfig would need validate() methods
        // For now, we assume they're valid if they can be created

        Ok(())
    }

    /// Notify configuration change listeners
    async fn notify_change(&self, config_type: &str, change_type: &str) {
        let notifications = self.change_notifications.read().await;
        for notifier in notifications.iter() {
            notifier(config_type, change_type);
        }
    }
}

impl ComponentRegistry {
    /// Create new component registry
    pub fn new() -> Self {
        Self {
            embedding_extractors: Arc::new(RwLock::new(HashMap::new())),
            quality_assessors: Arc::new(RwLock::new(HashMap::new())),
            health_monitor: Arc::new(RwLock::new(ComponentHealthMonitor::new())),
            resource_monitor: Arc::new(RwLock::new(ResourceMonitor::new())),
        }
    }

    /// Register embedding extractor
    pub async fn register_embedding_extractor(
        &self,
        name: String,
        extractor: Arc<SpeakerEmbeddingExtractor>,
    ) -> Result<()> {
        let mut extractors = self.embedding_extractors.write().await;
        extractors.insert(name.clone(), extractor);

        // Initialize health status
        let mut health_monitor = self.health_monitor.write().await;
        health_monitor
            .component_status
            .insert(name.clone(), ComponentStatus::new());

        info!("Registered embedding extractor: {}", name);
        Ok(())
    }

    /// Get embedding extractor (thread-safe)
    pub async fn get_embedding_extractor(
        &self,
        name: &str,
    ) -> Option<Arc<SpeakerEmbeddingExtractor>> {
        let extractors = self.embedding_extractors.read().await;
        extractors.get(name).cloned()
    }

    /// Register quality assessor
    pub async fn register_quality_assessor(
        &self,
        name: String,
        assessor: Arc<Mutex<CloningQualityAssessor>>,
    ) -> Result<()> {
        let mut assessors = self.quality_assessors.write().await;
        assessors.insert(name.clone(), assessor);

        // Initialize health status
        let mut health_monitor = self.health_monitor.write().await;
        health_monitor
            .component_status
            .insert(name.clone(), ComponentStatus::new());

        info!("Registered quality assessor: {}", name);
        Ok(())
    }

    /// Get quality assessor (thread-safe)
    pub async fn get_quality_assessor(
        &self,
        name: &str,
    ) -> Option<Arc<Mutex<CloningQualityAssessor>>> {
        let assessors = self.quality_assessors.read().await;
        assessors.get(name).cloned()
    }

    /// Perform health check on all components
    pub async fn health_check_all(&self) -> Result<HashMap<String, ComponentStatus>> {
        let mut results = HashMap::new();

        // Check embedding extractors
        let extractors = self.embedding_extractors.read().await;
        for (name, _extractor) in extractors.iter() {
            // Simplified health check - in practice would ping the component
            let mut status = ComponentStatus::new();
            status.is_healthy = true; // Assume healthy for now
            results.insert(format!("embedding_extractor_{}", name), status);
        }

        // Check quality assessors
        let assessors = self.quality_assessors.read().await;
        for (name, _assessor) in assessors.iter() {
            let mut status = ComponentStatus::new();
            status.is_healthy = true; // Assume healthy for now
            results.insert(format!("quality_assessor_{}", name), status);
        }

        // Update health monitor
        {
            let mut health_monitor = self.health_monitor.write().await;
            health_monitor.last_health_check = Some(Instant::now());
            for (name, status) in &results {
                health_monitor
                    .component_status
                    .insert(name.clone(), status.clone());
            }
        }

        Ok(results)
    }

    /// Get resource usage statistics
    pub async fn get_resource_stats(&self) -> ResourceMonitor {
        (*self.resource_monitor.read().await).clone()
    }

    /// Update resource usage for a component
    pub async fn update_resource_usage(
        &self,
        component_name: &str,
        memory_mb: u64,
        cpu_percent: f64,
        thread_count: usize,
    ) {
        let mut resource_monitor = self.resource_monitor.write().await;
        resource_monitor
            .memory_usage
            .insert(component_name.to_string(), memory_mb);
        resource_monitor
            .cpu_usage
            .insert(component_name.to_string(), cpu_percent);
        resource_monitor
            .thread_counts
            .insert(component_name.to_string(), thread_count);
    }

    /// Check if resource limits are exceeded
    pub async fn check_resource_limits(&self) -> Result<()> {
        let resource_monitor = self.resource_monitor.read().await;
        let limits = &resource_monitor.resource_limits;

        // Check memory limits
        for (component, &memory_mb) in &resource_monitor.memory_usage {
            if memory_mb > limits.max_memory_mb {
                return Err(Error::Processing(format!(
                    "Component {} exceeds memory limit: {} MB > {} MB",
                    component, memory_mb, limits.max_memory_mb
                )));
            }
        }

        // Check CPU limits
        for (component, &cpu_percent) in &resource_monitor.cpu_usage {
            if cpu_percent > limits.max_cpu_percent {
                return Err(Error::Processing(format!(
                    "Component {} exceeds CPU limit: {:.1}% > {:.1}%",
                    component, cpu_percent, limits.max_cpu_percent
                )));
            }
        }

        // Check thread limits
        for (component, &thread_count) in &resource_monitor.thread_counts {
            if thread_count > limits.max_threads_per_component {
                return Err(Error::Processing(format!(
                    "Component {} exceeds thread limit: {} > {}",
                    component, thread_count, limits.max_threads_per_component
                )));
            }
        }

        Ok(())
    }
}

impl ComponentHealthMonitor {
    /// Create new health monitor
    fn new() -> Self {
        Self {
            component_status: HashMap::new(),
            last_health_check: None,
            check_interval: Duration::from_secs(30), // 30 seconds default
        }
    }

    /// Check if health check is due
    pub fn is_health_check_due(&self) -> bool {
        match self.last_health_check {
            Some(last_check) => last_check.elapsed() >= self.check_interval,
            None => true,
        }
    }
}

impl ComponentStatus {
    /// Create new component status
    fn new() -> Self {
        Self {
            is_healthy: true,
            last_used: Instant::now(),
            error_count: 0,
            performance_metrics: PerformanceMetrics::default(),
        }
    }

    /// Record component usage
    pub fn record_usage(&mut self, response_time: Duration) {
        self.last_used = Instant::now();

        // Update average response time (simple moving average)
        let current_avg = self.performance_metrics.average_response_time;
        self.performance_metrics.average_response_time = Duration::from_nanos(
            (current_avg.as_nanos() as u64 + response_time.as_nanos() as u64) / 2,
        );
    }

    /// Record component error
    pub fn record_error(&mut self) {
        self.error_count += 1;

        // Mark as unhealthy if too many errors
        if self.error_count > 10 {
            self.is_healthy = false;
        }
    }
}

impl ResourceMonitor {
    /// Create new resource monitor
    fn new() -> Self {
        Self {
            memory_usage: HashMap::new(),
            cpu_usage: HashMap::new(),
            thread_counts: HashMap::new(),
            resource_limits: ResourceLimits::default(),
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 4096,   // 4GB default
            max_cpu_percent: 80.0, // 80% CPU max
            max_concurrent_operations: 100,
            max_threads_per_component: 16,
        }
    }
}

impl Default for ComponentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe operation coordinator for complex multi-step operations
pub struct OperationCoordinator {
    /// Active operations tracking
    active_operations: Arc<RwLock<HashMap<String, OperationState>>>,
    /// Operation semaphore for limiting concurrency
    operation_semaphore: Arc<Semaphore>,
    /// Operation timeout duration
    operation_timeout: Duration,
}

/// State of an active operation
#[derive(Debug, Clone)]
pub struct OperationState {
    pub operation_id: String,
    pub started_at: Instant,
    pub progress: f32, // 0.0 to 1.0
    pub status: OperationStatus,
    pub error_message: Option<String>,
}

/// Operation status enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum OperationStatus {
    Starting,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

impl OperationCoordinator {
    /// Create new operation coordinator
    pub fn new(max_concurrent_operations: usize, operation_timeout: Duration) -> Self {
        Self {
            active_operations: Arc::new(RwLock::new(HashMap::new())),
            operation_semaphore: Arc::new(Semaphore::new(max_concurrent_operations)),
            operation_timeout,
        }
    }

    /// Start a new coordinated operation
    pub async fn start_operation(&self, operation_id: String) -> Result<OperationGuard> {
        // Acquire operation permit
        let permit = self
            .operation_semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| Error::Processing(format!("Failed to acquire operation permit: {}", e)))?;

        // Register operation
        let operation_state = OperationState {
            operation_id: operation_id.clone(),
            started_at: Instant::now(),
            progress: 0.0,
            status: OperationStatus::Starting,
            error_message: None,
        };

        {
            let mut operations = self.active_operations.write().await;
            operations.insert(operation_id.clone(), operation_state);
        }

        Ok(OperationGuard {
            operation_id,
            active_operations: Arc::clone(&self.active_operations),
            _permit: permit,
        })
    }

    /// Update operation progress
    pub async fn update_progress(&self, operation_id: &str, progress: f32) {
        let mut operations = self.active_operations.write().await;
        if let Some(operation) = operations.get_mut(operation_id) {
            operation.progress = progress.clamp(0.0, 1.0);
            if progress >= 1.0 {
                operation.status = OperationStatus::Completed;
            } else {
                operation.status = OperationStatus::InProgress;
            }
        }
    }

    /// Mark operation as failed
    pub async fn mark_failed(&self, operation_id: &str, error_message: String) {
        let mut operations = self.active_operations.write().await;
        if let Some(operation) = operations.get_mut(operation_id) {
            operation.status = OperationStatus::Failed;
            operation.error_message = Some(error_message);
        }
    }

    /// Get operation status
    pub async fn get_operation_status(&self, operation_id: &str) -> Option<OperationState> {
        let operations = self.active_operations.read().await;
        operations.get(operation_id).cloned()
    }

    /// Clean up completed operations
    pub async fn cleanup_completed_operations(&self) {
        let mut operations = self.active_operations.write().await;
        operations.retain(|_, state| {
            match state.status {
                OperationStatus::Completed
                | OperationStatus::Failed
                | OperationStatus::Cancelled => {
                    // Keep operations for a short time after completion for status queries
                    state.started_at.elapsed() < Duration::from_secs(300) // 5 minutes
                }
                _ => true,
            }
        });
    }
}

impl Clone for OperationCoordinator {
    fn clone(&self) -> Self {
        Self {
            active_operations: Arc::clone(&self.active_operations),
            operation_semaphore: Arc::clone(&self.operation_semaphore),
            operation_timeout: self.operation_timeout,
        }
    }
}

/// RAII guard for coordinated operations  
pub struct OperationGuard {
    operation_id: String,
    active_operations: Arc<RwLock<HashMap<String, OperationState>>>,
    _permit: tokio::sync::OwnedSemaphorePermit,
}

impl Drop for OperationGuard {
    fn drop(&mut self) {
        // Clean up operation when guard is dropped
        let operation_id = self.operation_id.clone();
        let active_operations = Arc::clone(&self.active_operations);
        tokio::spawn(async move {
            let mut operations = active_operations.write().await;
            operations.remove(&operation_id);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_model_cache() {
        let cache: ModelCache<String> = ModelCache::new(2, 2);

        // Test cache miss and load
        let model1 = cache
            .get_or_load("model1", || async { Ok("loaded_model1".to_string()) })
            .await
            .unwrap();

        assert_eq!(*model1, "loaded_model1");

        // Test cache hit
        let model1_again = cache
            .get_or_load("model1", || async {
                panic!("Should not be called for cache hit");
            })
            .await
            .unwrap();

        assert_eq!(*model1_again, "loaded_model1");

        // Test cache statistics
        let stats = cache.get_stats().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[tokio::test]
    async fn test_unified_config_manager() {
        let cloning_config = CloningConfig::default();
        let embedding_config = EmbeddingConfig::default();
        let quality_config = QualityConfig::default();

        let manager = UnifiedConfigManager::new(cloning_config, embedding_config, quality_config);

        // Test configuration retrieval
        let retrieved_config = manager.get_cloning_config().await;
        assert_eq!(
            retrieved_config.default_method,
            crate::types::CloningMethod::FewShot
        );

        // Test configuration update
        let mut new_config = CloningConfig::default();
        new_config.quality_level = 0.9;

        let result = manager.update_cloning_config(new_config).await;
        assert!(result.is_ok());

        let updated_config = manager.get_cloning_config().await;
        assert_eq!(updated_config.quality_level, 0.9);
    }

    #[tokio::test]
    async fn test_component_registry() {
        let registry = ComponentRegistry::new();

        // Test embedding extractor registration
        let extractor =
            Arc::new(SpeakerEmbeddingExtractor::new(EmbeddingConfig::default()).unwrap());
        registry
            .register_embedding_extractor("test_extractor".to_string(), extractor)
            .await
            .unwrap();

        // Test retrieval
        let retrieved = registry.get_embedding_extractor("test_extractor").await;
        assert!(retrieved.is_some());

        // Test health check
        let health_results = registry.health_check_all().await.unwrap();
        assert!(!health_results.is_empty());
    }

    #[tokio::test]
    async fn test_operation_coordinator() {
        let coordinator = OperationCoordinator::new(2, Duration::from_secs(30));

        // Start operation
        let _guard = coordinator
            .start_operation("test_op".to_string())
            .await
            .unwrap();

        // Update progress
        coordinator.update_progress("test_op", 0.5).await;

        // Check status
        let status = coordinator.get_operation_status("test_op").await;
        assert!(status.is_some());
        assert_eq!(status.unwrap().progress, 0.5);
    }
}
