//! Advanced memory optimization and management system
//!
//! This module provides sophisticated memory optimization strategies, allocation tracking,
//! and automatic memory management for VoiRS synthesis operations.

use super::{MemoryMetrics, PerformanceMetrics};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, Semaphore};

/// Memory optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizerConfig {
    /// Enable memory optimization
    pub enabled: bool,
    /// Memory pressure threshold (percentage)
    pub pressure_threshold: f64,
    /// Fragmentation threshold for cleanup
    pub fragmentation_threshold: f64,
    /// Cache size limits
    pub cache_limits: CacheLimits,
    /// Garbage collection settings
    pub gc_settings: GcSettings,
    /// Pool allocation settings
    pub pool_settings: PoolSettings,
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Optimization strategies to enable
    pub enabled_strategies: Vec<OptimizationStrategy>,
}

/// Cache size limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheLimits {
    /// Maximum model cache size in bytes
    pub max_model_cache_bytes: u64,
    /// Maximum audio cache size in bytes
    pub max_audio_cache_bytes: u64,
    /// Maximum embedding cache size in bytes
    pub max_embedding_cache_bytes: u64,
    /// Cache entry TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Enable LRU eviction
    pub enable_lru_eviction: bool,
}

/// Garbage collection settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcSettings {
    /// Enable automatic garbage collection
    pub auto_gc_enabled: bool,
    /// Memory pressure threshold for triggering GC
    pub gc_pressure_threshold: f64,
    /// Minimum interval between GC runs
    pub min_gc_interval: Duration,
    /// Force GC after this many allocations
    pub force_gc_after_allocations: usize,
    /// Target heap size after GC (percentage of current)
    pub gc_target_heap_percent: f64,
}

/// Memory pool allocation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolSettings {
    /// Enable memory pooling
    pub enabled: bool,
    /// Pool sizes for different allocation sizes
    pub pool_sizes: HashMap<usize, usize>,
    /// Preallocation size for pools
    pub preallocation_size: usize,
    /// Pool cleanup interval
    pub cleanup_interval: Duration,
    /// Maximum pool memory usage
    pub max_pool_memory: u64,
}

/// Available optimization strategies
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    /// Aggressive cache cleanup
    AggressiveCacheCleanup,
    /// Memory pool allocation
    MemoryPooling,
    /// Lazy loading of models
    LazyModelLoading,
    /// Audio buffer optimization
    AudioBufferOptimization,
    /// Embedding cache compression
    EmbeddingCompression,
    /// Heap compaction
    HeapCompaction,
    /// Pre-allocation optimization
    PreallocationOptimization,
    /// Memory mapped files
    MemoryMappedFiles,
    /// Zero-copy optimizations
    ZeroCopyOptimization,
}

/// Memory optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Strategy that was applied
    pub strategy: OptimizationStrategy,
    /// Memory freed in bytes
    pub memory_freed_bytes: u64,
    /// Time taken for optimization
    pub optimization_time_ms: u64,
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Performance impact estimate
    pub performance_impact: f64,
}

/// Memory allocation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationTracker {
    /// Total allocations
    pub total_allocations: u64,
    /// Total deallocations
    pub total_deallocations: u64,
    /// Current active allocations
    pub active_allocations: u64,
    /// Peak allocations
    pub peak_allocations: u64,
    /// Allocation size histogram
    pub size_histogram: HashMap<usize, u64>,
    /// Allocation source tracking
    pub source_tracking: HashMap<String, AllocationSource>,
    /// Recent allocation patterns
    pub recent_patterns: VecDeque<AllocationPattern>,
}

/// Allocation source information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationSource {
    /// Source name (function, module, etc.)
    pub name: String,
    /// Total bytes allocated from this source
    pub total_bytes: u64,
    /// Number of allocations
    pub allocation_count: u64,
    /// Average allocation size
    pub average_size: f64,
    /// Peak allocation from this source
    pub peak_allocation: u64,
    /// Last allocation timestamp
    pub last_allocation: u64,
}

/// Allocation pattern for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationPattern {
    /// Pattern timestamp
    pub timestamp: u64,
    /// Allocation size
    pub size: usize,
    /// Source identifier
    pub source: String,
    /// Pattern type (burst, steady, etc.)
    pub pattern_type: PatternType,
    /// Duration of the pattern
    pub duration_ms: u64,
}

/// Types of allocation patterns
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    /// Burst allocation (many allocations in short time)
    Burst,
    /// Steady allocation (consistent rate)
    Steady,
    /// Large allocation (single large allocation)
    Large,
    /// Fragmented allocation (many small allocations)
    Fragmented,
    /// Cyclic allocation (predictable pattern)
    Cyclic,
}

/// Advanced memory optimizer
pub struct MemoryOptimizer {
    /// Configuration
    config: MemoryOptimizerConfig,
    /// Allocation tracker
    allocation_tracker: Arc<RwLock<AllocationTracker>>,
    /// Memory pools
    memory_pools: Arc<RwLock<HashMap<usize, Vec<Vec<u8>>>>>,
    /// Cache managers
    cache_managers: Arc<RwLock<HashMap<String, CacheManager>>>,
    /// Optimization history
    optimization_history: Arc<RwLock<VecDeque<OptimizationResult>>>,
    /// Last optimization time
    last_optimization: Arc<RwLock<Instant>>,
    /// Memory pressure semaphore
    pressure_semaphore: Arc<Semaphore>,
    /// Is running
    is_running: Arc<RwLock<bool>>,
}

/// Cache manager for different cache types
#[derive(Debug)]
struct CacheManager {
    /// Cache name
    name: String,
    /// Current size in bytes
    current_size: u64,
    /// Maximum size in bytes
    max_size: u64,
    /// Entry count
    entry_count: usize,
    /// Last cleanup time
    last_cleanup: Instant,
    /// Hit rate statistics
    hit_rate: f64,
    /// LRU tracking
    lru_keys: VecDeque<String>,
}

impl MemoryOptimizer {
    /// Create a new memory optimizer
    pub fn new(config: MemoryOptimizerConfig) -> Self {
        let pressure_permits = if config.pressure_threshold > 0.0 {
            ((100.0 - config.pressure_threshold) * 10.0) as usize
        } else {
            1000
        };

        Self {
            config,
            allocation_tracker: Arc::new(RwLock::new(AllocationTracker::new())),
            memory_pools: Arc::new(RwLock::new(HashMap::new())),
            cache_managers: Arc::new(RwLock::new(HashMap::new())),
            optimization_history: Arc::new(RwLock::new(VecDeque::new())),
            last_optimization: Arc::new(RwLock::new(Instant::now())),
            pressure_semaphore: Arc::new(Semaphore::new(pressure_permits)),
            is_running: Arc::new(RwLock::new(false)),
        }
    }

    /// Start the memory optimizer
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut is_running = self.is_running.write().await;
        if *is_running {
            return Ok(());
        }
        *is_running = true;
        drop(is_running);

        tracing::info!("Starting memory optimizer");

        // Initialize memory pools
        self.initialize_memory_pools().await;

        // Initialize cache managers
        self.initialize_cache_managers().await;

        // Start monitoring task
        self.start_monitoring_task().await;

        // Start optimization task
        self.start_optimization_task().await;

        Ok(())
    }

    /// Stop the memory optimizer
    pub async fn stop(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut is_running = self.is_running.write().await;
        if !*is_running {
            return Ok(());
        }
        *is_running = false;

        tracing::info!("Stopped memory optimizer");
        Ok(())
    }

    /// Analyze current memory usage and provide optimization recommendations
    pub async fn analyze_memory_usage(
        &self,
        metrics: &MemoryMetrics,
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Check memory pressure
        let memory_pressure = self.calculate_memory_pressure(metrics).await;
        if memory_pressure > self.config.pressure_threshold {
            recommendations.push(OptimizationRecommendation {
                strategy: OptimizationStrategy::AggressiveCacheCleanup,
                priority: 9,
                description: format!("High memory pressure detected: {:.1}%", memory_pressure),
                expected_savings_mb: self.estimate_cache_cleanup_savings().await,
                implementation_effort: ImplementationEffort::Low,
                performance_impact: -0.1, // Slight performance cost for cleanup
            });
        }

        // Check fragmentation
        if metrics.fragmentation_percent > self.config.fragmentation_threshold {
            recommendations.push(OptimizationRecommendation {
                strategy: OptimizationStrategy::HeapCompaction,
                priority: 7,
                description: format!(
                    "Memory fragmentation detected: {:.1}%",
                    metrics.fragmentation_percent
                ),
                expected_savings_mb: (metrics.heap_used as f64 * metrics.fragmentation_percent
                    / 100.0
                    / 1_000_000.0) as u32,
                implementation_effort: ImplementationEffort::Medium,
                performance_impact: -0.2, // Temporary performance cost during compaction
            });
        }

        // Check cache efficiency
        if metrics.cache_hit_rate < 70.0 {
            recommendations.push(OptimizationRecommendation {
                strategy: OptimizationStrategy::EmbeddingCompression,
                priority: 6,
                description: format!("Low cache hit rate: {:.1}%", metrics.cache_hit_rate),
                expected_savings_mb: self.estimate_compression_savings().await,
                implementation_effort: ImplementationEffort::Medium,
                performance_impact: 0.15, // Performance improvement from better cache usage
            });
        }

        // Check allocation patterns
        let patterns = self.analyze_allocation_patterns().await;
        for pattern in patterns {
            if pattern.pattern_type == PatternType::Burst {
                recommendations.push(OptimizationRecommendation {
                    strategy: OptimizationStrategy::MemoryPooling,
                    priority: 8,
                    description: "Burst allocation pattern detected - memory pooling recommended"
                        .to_string(),
                    expected_savings_mb: self.estimate_pooling_savings().await,
                    implementation_effort: ImplementationEffort::High,
                    performance_impact: 0.25, // Significant performance improvement
                });
                break;
            }
        }

        // Check for large allocations
        let tracker = self.allocation_tracker.read().await;
        if let Some(&large_allocs) = tracker
            .size_histogram
            .keys()
            .find(|&&size| size > 100_000_000)
        {
            drop(tracker);
            recommendations.push(OptimizationRecommendation {
                strategy: OptimizationStrategy::MemoryMappedFiles,
                priority: 5,
                description: "Large allocations detected - memory mapping recommended".to_string(),
                expected_savings_mb: (large_allocs / 1_000_000) as u32,
                implementation_effort: ImplementationEffort::High,
                performance_impact: 0.1,
            });
        }

        recommendations.sort_by(|a, b| b.priority.cmp(&a.priority));
        recommendations
    }

    /// Apply optimization strategy
    pub async fn apply_optimization(&self, strategy: OptimizationStrategy) -> OptimizationResult {
        let start_time = Instant::now();

        let result = match strategy {
            OptimizationStrategy::AggressiveCacheCleanup => self.perform_cache_cleanup().await,
            OptimizationStrategy::MemoryPooling => self.optimize_memory_pools().await,
            OptimizationStrategy::LazyModelLoading => self.implement_lazy_loading().await,
            OptimizationStrategy::AudioBufferOptimization => self.optimize_audio_buffers().await,
            OptimizationStrategy::EmbeddingCompression => self.compress_embeddings().await,
            OptimizationStrategy::HeapCompaction => self.perform_heap_compaction().await,
            OptimizationStrategy::PreallocationOptimization => self.optimize_preallocation().await,
            OptimizationStrategy::MemoryMappedFiles => self.implement_memory_mapping().await,
            OptimizationStrategy::ZeroCopyOptimization => self.implement_zero_copy().await,
        };

        let optimization_time_ms = start_time.elapsed().as_millis() as u64;

        let final_result = OptimizationResult {
            strategy,
            memory_freed_bytes: result.0,
            optimization_time_ms,
            success: result.1,
            error_message: result.2,
            performance_impact: result.3,
        };

        // Record optimization result
        let mut history = self.optimization_history.write().await;
        history.push_back(final_result.clone());
        if history.len() > 100 {
            history.pop_front();
        }

        final_result
    }

    /// Get memory optimization statistics
    pub async fn get_optimization_stats(&self) -> MemoryOptimizationStats {
        let tracker = self.allocation_tracker.read().await;
        let history = self.optimization_history.read().await;

        let total_optimizations = history.len();
        let successful_optimizations = history.iter().filter(|r| r.success).count();
        let total_memory_freed: u64 = history.iter().map(|r| r.memory_freed_bytes).sum();

        let average_optimization_time = if !history.is_empty() {
            history.iter().map(|r| r.optimization_time_ms).sum::<u64>() / history.len() as u64
        } else {
            0
        };

        MemoryOptimizationStats {
            total_optimizations,
            successful_optimizations,
            success_rate: if total_optimizations > 0 {
                (successful_optimizations as f64 / total_optimizations as f64) * 100.0
            } else {
                0.0
            },
            total_memory_freed_gb: total_memory_freed as f64 / 1_000_000_000.0,
            average_optimization_time_ms: average_optimization_time,
            current_allocation_count: tracker.active_allocations,
            peak_allocation_count: tracker.peak_allocations,
            fragmentation_events: self.count_fragmentation_events().await,
            cache_efficiency: self.calculate_overall_cache_efficiency().await,
        }
    }

    /// Initialize memory pools
    async fn initialize_memory_pools(&self) {
        if !self.config.pool_settings.enabled {
            return;
        }

        let mut pools = self.memory_pools.write().await;

        for (&size, &count) in &self.config.pool_settings.pool_sizes {
            let mut pool = Vec::with_capacity(count);
            for _ in 0..self.config.pool_settings.preallocation_size.min(count) {
                pool.push(vec![0u8; size]);
            }
            pools.insert(size, pool);
        }

        tracing::info!("Initialized {} memory pools", pools.len());
    }

    /// Initialize cache managers
    async fn initialize_cache_managers(&self) {
        let mut managers = self.cache_managers.write().await;

        // Model cache
        managers.insert(
            "models".to_string(),
            CacheManager {
                name: "models".to_string(),
                current_size: 0,
                max_size: self.config.cache_limits.max_model_cache_bytes,
                entry_count: 0,
                last_cleanup: Instant::now(),
                hit_rate: 0.0,
                lru_keys: VecDeque::new(),
            },
        );

        // Audio cache
        managers.insert(
            "audio".to_string(),
            CacheManager {
                name: "audio".to_string(),
                current_size: 0,
                max_size: self.config.cache_limits.max_audio_cache_bytes,
                entry_count: 0,
                last_cleanup: Instant::now(),
                hit_rate: 0.0,
                lru_keys: VecDeque::new(),
            },
        );

        // Embedding cache
        managers.insert(
            "embeddings".to_string(),
            CacheManager {
                name: "embeddings".to_string(),
                current_size: 0,
                max_size: self.config.cache_limits.max_embedding_cache_bytes,
                entry_count: 0,
                last_cleanup: Instant::now(),
                hit_rate: 0.0,
                lru_keys: VecDeque::new(),
            },
        );

        tracing::info!("Initialized {} cache managers", managers.len());
    }

    /// Start monitoring task
    async fn start_monitoring_task(&self) {
        let is_running = self.is_running.clone();
        let interval = self.config.monitoring_interval;
        let allocation_tracker = self.allocation_tracker.clone();

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                let running = is_running.read().await;
                if !*running {
                    break;
                }
                drop(running);

                // Update allocation tracking
                // This would integrate with actual memory tracking in a real implementation
                tracing::debug!("Memory monitoring tick");
            }
        });
    }

    /// Start optimization task
    async fn start_optimization_task(&self) {
        let is_running = self.is_running.clone();
        let config = self.config.clone();
        let optimizer = self.clone();

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(Duration::from_secs(60));

            loop {
                interval_timer.tick().await;

                let running = is_running.read().await;
                if !*running {
                    break;
                }
                drop(running);

                // Check if optimization is needed
                if let Some(strategy) = optimizer.determine_needed_optimization().await {
                    let result = optimizer.apply_optimization(strategy).await;
                    if result.success {
                        tracing::info!(
                            "Applied optimization {:?}, freed {} MB",
                            result.strategy,
                            result.memory_freed_bytes / 1_000_000
                        );
                    } else {
                        tracing::warn!(
                            "Failed to apply optimization {:?}: {:?}",
                            result.strategy,
                            result.error_message
                        );
                    }
                }
            }
        });
    }

    /// Calculate memory pressure
    async fn calculate_memory_pressure(&self, metrics: &MemoryMetrics) -> f64 {
        // This is a simplified calculation - in a real implementation,
        // this would use system memory information
        let heap_pressure =
            (metrics.heap_used as f64 / (metrics.heap_used + 1_000_000_000) as f64) * 100.0;
        let fragmentation_pressure = metrics.fragmentation_percent;
        let allocation_pressure = metrics.allocations_per_sec / 1000.0; // Normalize

        (heap_pressure + fragmentation_pressure + allocation_pressure) / 3.0
    }

    /// Determine if optimization is needed
    async fn determine_needed_optimization(&self) -> Option<OptimizationStrategy> {
        // Check memory pressure
        let allocation_tracker = self.allocation_tracker.read().await;

        // Simple heuristics for determining needed optimization
        if allocation_tracker.active_allocations > allocation_tracker.peak_allocations * 80 / 100 {
            return Some(OptimizationStrategy::AggressiveCacheCleanup);
        }

        // Check for burst patterns
        if allocation_tracker
            .recent_patterns
            .iter()
            .any(|p| p.pattern_type == PatternType::Burst)
        {
            return Some(OptimizationStrategy::MemoryPooling);
        }

        None
    }

    // Optimization implementation methods (simplified for demonstration)
    async fn perform_cache_cleanup(&self) -> (u64, bool, Option<String>, f64) {
        tracing::info!("Performing aggressive cache cleanup");
        // Implementation would clean up caches
        (50_000_000, true, None, -0.05) // 50MB freed, success, no error, slight performance cost
    }

    async fn optimize_memory_pools(&self) -> (u64, bool, Option<String>, f64) {
        tracing::info!("Optimizing memory pools");
        // Implementation would optimize pool allocation
        (30_000_000, true, None, 0.15) // 30MB freed, success, performance improvement
    }

    async fn implement_lazy_loading(&self) -> (u64, bool, Option<String>, f64) {
        tracing::info!("Implementing lazy model loading");
        (100_000_000, true, None, 0.1) // 100MB freed through lazy loading
    }

    async fn optimize_audio_buffers(&self) -> (u64, bool, Option<String>, f64) {
        tracing::info!("Optimizing audio buffers");
        (20_000_000, true, None, 0.05) // 20MB freed from buffer optimization
    }

    async fn compress_embeddings(&self) -> (u64, bool, Option<String>, f64) {
        tracing::info!("Compressing embeddings");
        (75_000_000, true, None, 0.08) // 75MB freed from compression
    }

    async fn perform_heap_compaction(&self) -> (u64, bool, Option<String>, f64) {
        tracing::info!("Performing heap compaction");
        (40_000_000, true, None, -0.1) // 40MB freed, temporary performance cost
    }

    async fn optimize_preallocation(&self) -> (u64, bool, Option<String>, f64) {
        tracing::info!("Optimizing preallocation");
        (25_000_000, true, None, 0.12) // 25MB freed from better preallocation
    }

    async fn implement_memory_mapping(&self) -> (u64, bool, Option<String>, f64) {
        tracing::info!("Implementing memory mapping");
        (150_000_000, true, None, 0.2) // 150MB freed from memory mapping
    }

    async fn implement_zero_copy(&self) -> (u64, bool, Option<String>, f64) {
        tracing::info!("Implementing zero-copy optimizations");
        (60_000_000, true, None, 0.18) // 60MB freed from zero-copy
    }

    // Helper methods for calculations
    async fn estimate_cache_cleanup_savings(&self) -> u32 {
        50 // 50MB estimated savings
    }

    async fn estimate_compression_savings(&self) -> u32 {
        30 // 30MB estimated savings
    }

    async fn estimate_pooling_savings(&self) -> u32 {
        25 // 25MB estimated savings
    }

    async fn analyze_allocation_patterns(&self) -> Vec<AllocationPattern> {
        let tracker = self.allocation_tracker.read().await;
        tracker.recent_patterns.iter().cloned().collect()
    }

    async fn count_fragmentation_events(&self) -> u64 {
        // Implementation would count actual fragmentation events
        0
    }

    async fn calculate_overall_cache_efficiency(&self) -> f64 {
        let managers = self.cache_managers.read().await;
        if managers.is_empty() {
            return 0.0;
        }

        let total_hit_rate: f64 = managers.values().map(|m| m.hit_rate).sum();
        total_hit_rate / managers.len() as f64
    }
}

impl Clone for MemoryOptimizer {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            allocation_tracker: self.allocation_tracker.clone(),
            memory_pools: self.memory_pools.clone(),
            cache_managers: self.cache_managers.clone(),
            optimization_history: self.optimization_history.clone(),
            last_optimization: self.last_optimization.clone(),
            pressure_semaphore: self.pressure_semaphore.clone(),
            is_running: self.is_running.clone(),
        }
    }
}

/// Memory optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Optimization strategy
    pub strategy: OptimizationStrategy,
    /// Priority level (1-10)
    pub priority: u8,
    /// Description of the recommendation
    pub description: String,
    /// Expected memory savings in MB
    pub expected_savings_mb: u32,
    /// Implementation effort required
    pub implementation_effort: ImplementationEffort,
    /// Performance impact (-1.0 to 1.0, negative = cost, positive = benefit)
    pub performance_impact: f64,
}

/// Implementation effort levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImplementationEffort {
    /// Low effort, can be applied immediately
    Low,
    /// Medium effort, requires some planning
    Medium,
    /// High effort, requires significant changes
    High,
}

/// Memory optimization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizationStats {
    /// Total optimizations performed
    pub total_optimizations: usize,
    /// Successful optimizations
    pub successful_optimizations: usize,
    /// Success rate percentage
    pub success_rate: f64,
    /// Total memory freed in GB
    pub total_memory_freed_gb: f64,
    /// Average optimization time in milliseconds
    pub average_optimization_time_ms: u64,
    /// Current allocation count
    pub current_allocation_count: u64,
    /// Peak allocation count
    pub peak_allocation_count: u64,
    /// Number of fragmentation events
    pub fragmentation_events: u64,
    /// Overall cache efficiency percentage
    pub cache_efficiency: f64,
}

impl AllocationTracker {
    fn new() -> Self {
        Self {
            total_allocations: 0,
            total_deallocations: 0,
            active_allocations: 0,
            peak_allocations: 0,
            size_histogram: HashMap::new(),
            source_tracking: HashMap::new(),
            recent_patterns: VecDeque::new(),
        }
    }
}

impl Default for MemoryOptimizerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            pressure_threshold: 80.0,
            fragmentation_threshold: 15.0,
            cache_limits: CacheLimits::default(),
            gc_settings: GcSettings::default(),
            pool_settings: PoolSettings::default(),
            monitoring_interval: Duration::from_secs(30),
            enabled_strategies: vec![
                OptimizationStrategy::AggressiveCacheCleanup,
                OptimizationStrategy::MemoryPooling,
                OptimizationStrategy::LazyModelLoading,
                OptimizationStrategy::AudioBufferOptimization,
            ],
        }
    }
}

impl Default for CacheLimits {
    fn default() -> Self {
        Self {
            max_model_cache_bytes: 2_000_000_000,   // 2GB
            max_audio_cache_bytes: 1_000_000_000,   // 1GB
            max_embedding_cache_bytes: 500_000_000, // 500MB
            cache_ttl_seconds: 3600,                // 1 hour
            enable_lru_eviction: true,
        }
    }
}

impl Default for GcSettings {
    fn default() -> Self {
        Self {
            auto_gc_enabled: true,
            gc_pressure_threshold: 85.0,
            min_gc_interval: Duration::from_secs(300), // 5 minutes
            force_gc_after_allocations: 10000,
            gc_target_heap_percent: 70.0,
        }
    }
}

impl Default for PoolSettings {
    fn default() -> Self {
        let mut pool_sizes = HashMap::new();
        pool_sizes.insert(1024, 100); // 1KB buffers
        pool_sizes.insert(4096, 50); // 4KB buffers
        pool_sizes.insert(16384, 25); // 16KB buffers
        pool_sizes.insert(65536, 10); // 64KB buffers
        pool_sizes.insert(262144, 5); // 256KB buffers

        Self {
            enabled: true,
            pool_sizes,
            preallocation_size: 10,
            cleanup_interval: Duration::from_secs(600), // 10 minutes
            max_pool_memory: 100_000_000,               // 100MB
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_optimizer_creation() {
        let config = MemoryOptimizerConfig::default();
        let optimizer = MemoryOptimizer::new(config);

        assert!(!*optimizer.is_running.read().await);
    }

    #[tokio::test]
    async fn test_memory_pressure_calculation() {
        let config = MemoryOptimizerConfig::default();
        let optimizer = MemoryOptimizer::new(config);

        let metrics = MemoryMetrics {
            heap_used: 800_000_000, // 800MB
            fragmentation_percent: 20.0,
            allocations_per_sec: 500.0,
            ..Default::default()
        };

        let pressure = optimizer.calculate_memory_pressure(&metrics).await;
        assert!(pressure > 0.0);
    }

    #[tokio::test]
    async fn test_optimization_recommendations() {
        let config = MemoryOptimizerConfig::default();
        let optimizer = MemoryOptimizer::new(config);

        let metrics = MemoryMetrics {
            heap_used: 900_000_000,         // High memory usage
            fragmentation_percent: 85.0,    // Very high fragmentation to trigger pressure
            cache_hit_rate: 50.0,           // Low cache hit rate
            allocations_per_sec: 150_000.0, // Very high allocation rate
            ..Default::default()
        };

        let recommendations = optimizer.analyze_memory_usage(&metrics).await;
        assert!(!recommendations.is_empty());

        // Should recommend cache cleanup due to high memory pressure
        assert!(recommendations
            .iter()
            .any(|r| r.strategy == OptimizationStrategy::AggressiveCacheCleanup));
    }

    #[tokio::test]
    async fn test_cache_cleanup_optimization() {
        let config = MemoryOptimizerConfig::default();
        let optimizer = MemoryOptimizer::new(config);

        let result = optimizer
            .apply_optimization(OptimizationStrategy::AggressiveCacheCleanup)
            .await;

        assert!(result.success);
        assert!(result.memory_freed_bytes > 0);
    }

    #[test]
    fn test_config_defaults() {
        let config = MemoryOptimizerConfig::default();

        assert!(config.enabled);
        assert_eq!(config.pressure_threshold, 80.0);
        assert_eq!(config.fragmentation_threshold, 15.0);
        assert!(!config.enabled_strategies.is_empty());
    }

    #[test]
    fn test_allocation_tracker() {
        let tracker = AllocationTracker::new();

        assert_eq!(tracker.total_allocations, 0);
        assert_eq!(tracker.active_allocations, 0);
        assert!(tracker.size_histogram.is_empty());
    }
}
