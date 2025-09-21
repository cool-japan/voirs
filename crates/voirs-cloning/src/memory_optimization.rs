//! Memory optimization for mobile and edge deployment
//!
//! This module provides comprehensive memory management features including memory pools,
//! efficient data structures, automatic garbage collection, and mobile-specific optimizations
//! to minimize memory footprint while maintaining performance.

use crate::{
    embedding::SpeakerEmbedding,
    types::{SpeakerProfile, VoiceSample},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::{
    backtrace::Backtrace,
    collections::{HashMap, HashSet, VecDeque},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Weak,
    },
    thread,
    time::{Duration, Instant, SystemTime},
};
use tokio::sync::RwLock;
use tracing::{debug, error, info, trace, warn};

/// Configuration for memory optimization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryOptimizationConfig {
    /// Enable memory optimization
    pub enable_optimization: bool,
    /// Maximum memory usage in bytes (0 = unlimited)
    pub max_memory_usage: usize,
    /// Enable automatic garbage collection
    pub enable_garbage_collection: bool,
    /// Garbage collection interval
    pub gc_interval: Duration,
    /// Memory pressure threshold (0.0-1.0)
    pub memory_pressure_threshold: f32,
    /// Enable memory pool for frequent allocations
    pub enable_memory_pool: bool,
    /// Pool initial size for different data types
    pub pool_initial_sizes: MemoryPoolSizes,
    /// Enable data compression for stored embeddings
    pub enable_compression: bool,
    /// Compression quality (0.0-1.0, higher = better quality, more memory)
    pub compression_quality: f32,
    /// Enable lazy loading for non-critical data
    pub enable_lazy_loading: bool,
    /// Cache size limits
    pub cache_limits: CacheLimits,
    /// Enable mobile-specific optimizations
    pub enable_mobile_optimizations: bool,
    /// Enable memory mapping for large data
    pub enable_memory_mapping: bool,
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_optimization: true,
            max_memory_usage: 512 * 1024 * 1024, // 512MB
            enable_garbage_collection: true,
            gc_interval: Duration::from_secs(30),
            memory_pressure_threshold: 0.8,
            enable_memory_pool: true,
            pool_initial_sizes: MemoryPoolSizes::default(),
            enable_compression: true,
            compression_quality: 0.7,
            enable_lazy_loading: true,
            cache_limits: CacheLimits::default(),
            enable_mobile_optimizations: true,
            enable_memory_mapping: false, // Disabled by default for compatibility
        }
    }
}

impl MemoryOptimizationConfig {
    /// Create configuration optimized for mobile devices
    pub fn mobile_optimized() -> Self {
        Self {
            enable_optimization: true,
            max_memory_usage: 128 * 1024 * 1024, // 128MB for mobile
            enable_garbage_collection: true,
            gc_interval: Duration::from_secs(15), // More frequent GC on mobile
            memory_pressure_threshold: 0.7,       // Lower threshold for mobile
            enable_memory_pool: true,
            pool_initial_sizes: MemoryPoolSizes::mobile_optimized(),
            enable_compression: true,
            compression_quality: 0.5, // Lower quality for memory savings
            enable_lazy_loading: true,
            cache_limits: CacheLimits::mobile_optimized(),
            enable_mobile_optimizations: true,
            enable_memory_mapping: false,
        }
    }

    /// Create configuration optimized for edge devices
    pub fn edge_optimized() -> Self {
        Self {
            enable_optimization: true,
            max_memory_usage: 64 * 1024 * 1024, // 64MB for edge devices
            enable_garbage_collection: true,
            gc_interval: Duration::from_secs(10), // Very frequent GC
            memory_pressure_threshold: 0.6,       // Even lower threshold
            enable_memory_pool: true,
            pool_initial_sizes: MemoryPoolSizes::edge_optimized(),
            enable_compression: true,
            compression_quality: 0.3, // Aggressive compression
            enable_lazy_loading: true,
            cache_limits: CacheLimits::edge_optimized(),
            enable_mobile_optimizations: true,
            enable_memory_mapping: true, // Use memory mapping for edge
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.memory_pressure_threshold) {
            return Err(Error::Config(
                "Memory pressure threshold must be between 0.0 and 1.0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.compression_quality) {
            return Err(Error::Config(
                "Compression quality must be between 0.0 and 1.0".to_string(),
            ));
        }
        if self.gc_interval.as_millis() == 0 {
            return Err(Error::Config(
                "Garbage collection interval must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Memory pool sizes for different data types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryPoolSizes {
    /// Pool size for speaker embeddings
    pub embeddings_pool_size: usize,
    /// Pool size for voice samples
    pub samples_pool_size: usize,
    /// Pool size for audio buffers
    pub audio_buffers_pool_size: usize,
    /// Pool size for temporary vectors
    pub temp_vectors_pool_size: usize,
}

impl Default for MemoryPoolSizes {
    fn default() -> Self {
        Self {
            embeddings_pool_size: 100,
            samples_pool_size: 50,
            audio_buffers_pool_size: 20,
            temp_vectors_pool_size: 200,
        }
    }
}

impl MemoryPoolSizes {
    pub fn mobile_optimized() -> Self {
        Self {
            embeddings_pool_size: 50,
            samples_pool_size: 25,
            audio_buffers_pool_size: 10,
            temp_vectors_pool_size: 100,
        }
    }

    pub fn edge_optimized() -> Self {
        Self {
            embeddings_pool_size: 20,
            samples_pool_size: 10,
            audio_buffers_pool_size: 5,
            temp_vectors_pool_size: 50,
        }
    }
}

/// Cache size limits for different components
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheLimits {
    /// Maximum speaker profiles to cache
    pub max_speaker_profiles: usize,
    /// Maximum embeddings to cache
    pub max_embeddings: usize,
    /// Maximum voice samples to cache
    pub max_voice_samples: usize,
    /// Maximum quality assessments to cache
    pub max_quality_assessments: usize,
}

impl Default for CacheLimits {
    fn default() -> Self {
        Self {
            max_speaker_profiles: 100,
            max_embeddings: 200,
            max_voice_samples: 50,
            max_quality_assessments: 100,
        }
    }
}

impl CacheLimits {
    pub fn mobile_optimized() -> Self {
        Self {
            max_speaker_profiles: 50,
            max_embeddings: 100,
            max_voice_samples: 25,
            max_quality_assessments: 50,
        }
    }

    pub fn edge_optimized() -> Self {
        Self {
            max_speaker_profiles: 20,
            max_embeddings: 50,
            max_voice_samples: 10,
            max_quality_assessments: 25,
        }
    }
}

/// Memory pool for efficient allocation and reuse
#[derive(Debug)]
pub struct MemoryPool<T> {
    pool: Arc<RwLock<VecDeque<T>>>,
    max_size: usize,
    current_size: Arc<RwLock<usize>>,
    allocation_count: Arc<RwLock<usize>>,
    deallocation_count: Arc<RwLock<usize>>,
}

impl<T> MemoryPool<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    /// Create a new memory pool
    pub fn new(initial_size: usize, max_size: usize) -> Self {
        let mut pool = VecDeque::with_capacity(max_size);

        // Pre-allocate initial objects
        for _ in 0..initial_size {
            pool.push_back(T::default());
        }

        Self {
            pool: Arc::new(RwLock::new(pool)),
            max_size,
            current_size: Arc::new(RwLock::new(initial_size)),
            allocation_count: Arc::new(RwLock::new(0)),
            deallocation_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Get an object from the pool
    pub async fn get(&self) -> PooledObject<T> {
        let mut pool = self.pool.write().await;
        let object = if let Some(obj) = pool.pop_front() {
            obj
        } else {
            // Pool is empty, create new object
            T::default()
        };

        let mut allocation_count = self.allocation_count.write().await;
        *allocation_count += 1;

        PooledObject::new(object, self.pool.clone(), self.deallocation_count.clone())
    }

    /// Return an object to the pool
    pub async fn return_object(&self, object: T) {
        let mut pool = self.pool.write().await;
        let current_size = {
            let size_guard = self.current_size.read().await;
            *size_guard
        };

        if current_size < self.max_size {
            pool.push_back(object);
            let mut size_guard = self.current_size.write().await;
            *size_guard += 1;
        }
        // If pool is full, object is dropped (deallocated)
    }

    /// Get pool statistics
    pub async fn get_stats(&self) -> MemoryPoolStats {
        let pool = self.pool.read().await;
        let allocation_count = *self.allocation_count.read().await;
        let deallocation_count = *self.deallocation_count.read().await;

        MemoryPoolStats {
            current_size: pool.len(),
            max_size: self.max_size,
            allocation_count,
            deallocation_count,
            hit_rate: if allocation_count > 0 {
                (allocation_count - deallocation_count) as f32 / allocation_count as f32
            } else {
                0.0
            },
        }
    }
}

/// Pooled object with automatic return to pool
#[derive(Debug)]
pub struct PooledObject<T: Send + Sync + 'static> {
    object: Option<T>,
    pool: Arc<RwLock<VecDeque<T>>>,
    deallocation_count: Arc<RwLock<usize>>,
}

impl<T: Send + Sync + 'static> PooledObject<T> {
    fn new(
        object: T,
        pool: Arc<RwLock<VecDeque<T>>>,
        deallocation_count: Arc<RwLock<usize>>,
    ) -> Self {
        Self {
            object: Some(object),
            pool,
            deallocation_count,
        }
    }

    /// Get reference to the pooled object
    pub fn get(&self) -> Option<&T> {
        self.object.as_ref()
    }

    /// Get mutable reference to the pooled object
    pub fn get_mut(&mut self) -> Option<&mut T> {
        self.object.as_mut()
    }
}

impl<T: Send + Sync + 'static> Drop for PooledObject<T> {
    fn drop(&mut self) {
        if let Some(object) = self.object.take() {
            // Return object to pool asynchronously
            let pool = self.pool.clone();
            let deallocation_count = self.deallocation_count.clone();

            tokio::spawn(async move {
                let mut pool_guard = pool.write().await;
                pool_guard.push_back(object);

                let mut dealloc_count = deallocation_count.write().await;
                *dealloc_count += 1;
            });
        }
    }
}

/// Statistics for memory pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolStats {
    /// Current number of objects in pool
    pub current_size: usize,
    /// Maximum pool size
    pub max_size: usize,
    /// Total allocations
    pub allocation_count: usize,
    /// Total deallocations
    pub deallocation_count: usize,
    /// Pool hit rate (objects reused vs new allocations)
    pub hit_rate: f32,
}

/// Compressed embedding for memory optimization
#[derive(Debug, Clone)]
pub struct CompressedEmbedding {
    /// Compressed data
    pub compressed_data: Vec<u8>,
    /// Original dimension
    pub original_dimension: usize,
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Quality score after compression
    pub quality_score: f32,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl CompressedEmbedding {
    /// Compress a speaker embedding
    pub fn compress(embedding: &SpeakerEmbedding, quality: f32) -> Result<Self> {
        // Simplified compression - in production, use proper compression algorithms
        let original_size = embedding.vector.len() * 4; // Assuming f32

        // Quantize to lower precision based on quality
        let quantized: Vec<u8> = embedding
            .vector
            .iter()
            .map(|&x| {
                let quantized = if quality > 0.8 {
                    // High quality: use 16-bit quantization
                    ((x * 32767.0).clamp(-32768.0, 32767.0) as i16)
                        .to_le_bytes()
                        .to_vec()
                } else if quality > 0.5 {
                    // Medium quality: use 8-bit quantization
                    vec![((x * 127.0).clamp(-128.0, 127.0) as i8) as u8; 2]
                } else {
                    // Low quality: use 4-bit quantization (packed)
                    vec![((x * 7.0).clamp(-8.0, 7.0) as i8 + 8) as u8; 2]
                };
                quantized
            })
            .flatten()
            .collect();

        let compressed_size = quantized.len();
        let compression_ratio = original_size as f32 / compressed_size as f32;

        // Estimate quality loss
        let quality_loss = (1.0 - quality) * 0.1;
        let final_quality = embedding.confidence * (1.0 - quality_loss);

        let mut metadata = HashMap::new();
        metadata.insert("compression_method".to_string(), "quantization".to_string());
        metadata.insert(
            "original_confidence".to_string(),
            embedding.confidence.to_string(),
        );

        Ok(Self {
            compressed_data: quantized,
            original_dimension: embedding.vector.len(),
            compression_ratio,
            quality_score: final_quality,
            metadata,
        })
    }

    /// Decompress back to speaker embedding
    pub fn decompress(&self) -> Result<SpeakerEmbedding> {
        let quality = self
            .metadata
            .get("compression_method")
            .map(|s| if s == "quantization" { 0.7 } else { 0.5 })
            .unwrap_or(0.5);

        // Reconstruct vector based on compression method
        let mut vector = Vec::with_capacity(self.original_dimension);
        let bytes_per_element = self.compressed_data.len() / self.original_dimension;

        for i in 0..self.original_dimension {
            let start_idx = i * bytes_per_element;
            let element = if bytes_per_element >= 2 {
                // 16-bit or 8-bit quantization
                let bytes = &self.compressed_data[start_idx..start_idx + 2];
                if quality > 0.8 {
                    // 16-bit
                    i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32767.0
                } else {
                    // 8-bit
                    (bytes[0] as i8) as f32 / 127.0
                }
            } else {
                // 4-bit quantization
                let byte = self.compressed_data[start_idx];
                (byte as i8 - 8) as f32 / 7.0
            };

            vector.push(element);
        }

        Ok(SpeakerEmbedding::new(vector))
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.compressed_data.len()
            + self
                .metadata
                .iter()
                .map(|(k, v)| k.len() + v.len())
                .sum::<usize>()
            + 32 // Overhead for fields
    }
}

/// Memory leak detection and audit system
#[derive(Debug)]
pub struct MemoryLeakDetector {
    /// Active allocations tracking
    active_allocations: Arc<RwLock<HashMap<String, AllocationInfo>>>,
    /// Memory allocation patterns
    allocation_patterns: Arc<RwLock<HashMap<String, AllocationPattern>>>,
    /// Weak reference registry for leak detection
    weak_registry: Arc<RwLock<HashMap<String, WeakRefInfo>>>,
    /// Memory pressure alerts
    pressure_alerts: Arc<RwLock<Vec<MemoryPressureAlert>>>,
    /// Leak detection statistics
    leak_stats: Arc<RwLock<LeakDetectionStats>>,
    /// Configuration
    config: LeakDetectionConfig,
    /// Global allocation counters
    total_allocations: AtomicU64,
    total_deallocations: AtomicU64,
    total_bytes_allocated: AtomicU64,
    total_bytes_deallocated: AtomicU64,
}

/// Configuration for memory leak detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakDetectionConfig {
    /// Enable leak detection
    pub enabled: bool,
    /// Track allocation stack traces
    pub track_stack_traces: bool,
    /// Maximum tracked allocations
    pub max_tracked_allocations: usize,
    /// Leak detection interval
    pub detection_interval: Duration,
    /// Threshold for considering allocation as potential leak (in seconds)
    pub leak_threshold: Duration,
    /// Enable automatic cleanup of detected leaks
    pub auto_cleanup: bool,
    /// Memory pressure alert threshold (percentage)
    pub pressure_alert_threshold: f64,
    /// Enable detailed allocation profiling
    pub enable_profiling: bool,
}

/// Information about a memory allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationInfo {
    /// Unique allocation ID
    pub id: String,
    /// Allocation size in bytes
    pub size: usize,
    /// Allocation timestamp
    pub timestamp: SystemTime,
    /// Allocation type
    pub allocation_type: AllocationType,
    /// Stack trace at allocation (if enabled)
    pub stack_trace: Option<String>,
    /// Thread ID that performed allocation
    pub thread_id: String,
    /// Reference count at time of allocation
    pub ref_count: usize,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Types of memory allocations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AllocationType {
    /// Speaker embedding allocation
    SpeakerEmbedding,
    /// Voice sample buffer allocation
    VoiceSample,
    /// Model weights allocation
    ModelWeights,
    /// Audio processing buffer allocation
    AudioBuffer,
    /// Cache allocation
    Cache,
    /// Temporary allocation
    Temporary,
    /// Unknown allocation type
    Unknown,
}

/// Allocation pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationPattern {
    /// Pattern name
    pub name: String,
    /// Allocation type
    pub allocation_type: AllocationType,
    /// Frequency of allocation
    pub frequency: u64,
    /// Average allocation size
    pub avg_size: f64,
    /// Peak allocation size
    pub peak_size: usize,
    /// Average lifetime
    pub avg_lifetime: Duration,
    /// Leak probability score (0.0-1.0)
    pub leak_probability: f32,
    /// First seen timestamp
    pub first_seen: SystemTime,
    /// Last seen timestamp
    pub last_seen: SystemTime,
}

/// Weak reference tracking information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeakRefInfo {
    /// Reference ID
    pub id: String,
    /// Target allocation ID
    pub target_allocation_id: String,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last access timestamp
    pub last_accessed: SystemTime,
    /// Access count
    pub access_count: u64,
    /// Is reference still alive
    pub is_alive: bool,
}

/// Memory pressure alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPressureAlert {
    /// Alert ID
    pub id: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert message
    pub message: String,
    /// Current memory usage
    pub current_usage: usize,
    /// Memory limit
    pub memory_limit: usize,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Alert metadata
    pub metadata: HashMap<String, String>,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Critical alert
    Critical,
    /// Emergency alert requiring immediate action
    Emergency,
}

/// Types of memory alerts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlertType {
    /// Memory usage exceeding threshold
    MemoryPressure,
    /// Potential memory leak detected
    PotentialLeak,
    /// Memory fragmentation detected
    Fragmentation,
    /// Allocation rate too high
    HighAllocationRate,
    /// Garbage collection failure
    GcFailure,
    /// Pool exhaustion
    PoolExhaustion,
}

/// Leak detection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakDetectionStats {
    /// Total number of tracked allocations
    pub tracked_allocations: usize,
    /// Number of potential leaks detected
    pub potential_leaks: usize,
    /// Number of confirmed leaks
    pub confirmed_leaks: usize,
    /// Number of leaks cleaned up automatically
    pub auto_cleaned_leaks: usize,
    /// Total memory potentially leaked (bytes)
    pub total_leaked_bytes: usize,
    /// Average allocation lifetime
    pub avg_allocation_lifetime: Duration,
    /// Memory pressure events
    pub pressure_events: usize,
    /// Last scan timestamp
    pub last_scan: SystemTime,
    /// Scan duration statistics
    pub scan_duration_stats: DurationStats,
}

/// Duration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurationStats {
    /// Minimum duration
    pub min: Duration,
    /// Maximum duration
    pub max: Duration,
    /// Average duration
    pub avg: Duration,
    /// Total duration
    pub total: Duration,
    /// Sample count
    pub count: u64,
}

/// Memory audit report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAuditReport {
    /// Report generation timestamp
    pub timestamp: SystemTime,
    /// Overall memory health score (0.0-1.0)
    pub health_score: f32,
    /// Current memory usage breakdown
    pub memory_usage: MemoryUsageBreakdown,
    /// Detected issues
    pub detected_issues: Vec<MemoryIssue>,
    /// Allocation patterns analysis
    pub allocation_patterns: Vec<AllocationPattern>,
    /// Leak detection summary
    pub leak_summary: LeakSummary,
    /// Performance impact analysis
    pub performance_impact: PerformanceImpactAnalysis,
    /// Recommendations
    pub recommendations: Vec<MemoryRecommendation>,
    /// Detailed statistics
    pub detailed_stats: DetailedMemoryStats,
}

/// Memory usage breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageBreakdown {
    /// Total allocated memory
    pub total_allocated: usize,
    /// Memory by allocation type
    pub by_type: HashMap<AllocationType, usize>,
    /// Memory by component
    pub by_component: HashMap<String, usize>,
    /// Pool memory usage
    pub pool_usage: HashMap<String, MemoryPoolStats>,
    /// Cache memory usage
    pub cache_usage: usize,
    /// Fragmentation percentage
    pub fragmentation_percentage: f32,
}

/// Memory issue description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryIssue {
    /// Issue ID
    pub id: String,
    /// Issue type
    pub issue_type: MemoryIssueType,
    /// Severity level
    pub severity: AlertSeverity,
    /// Issue description
    pub description: String,
    /// Affected allocations
    pub affected_allocations: Vec<String>,
    /// Estimated impact (bytes)
    pub estimated_impact: usize,
    /// Detection timestamp
    pub detected_at: SystemTime,
    /// Suggested resolution
    pub suggested_resolution: String,
}

/// Types of memory issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryIssueType {
    /// Memory leak
    Leak,
    /// Memory fragmentation
    Fragmentation,
    /// Excessive allocation rate
    ExcessiveAllocation,
    /// Pool inefficiency
    PoolInefficiency,
    /// Cache thrashing
    CacheThrashing,
    /// Reference cycle
    ReferenceCycle,
    /// Memory pressure
    MemoryPressure,
}

/// Leak detection summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakSummary {
    /// Total potential leaks
    pub total_potential_leaks: usize,
    /// Confirmed leaks
    pub confirmed_leaks: usize,
    /// False positives
    pub false_positives: usize,
    /// Total leaked bytes
    pub total_leaked_bytes: usize,
    /// Leak by type
    pub leaks_by_type: HashMap<AllocationType, usize>,
    /// Average leak size
    pub avg_leak_size: usize,
    /// Oldest leak age
    pub oldest_leak_age: Duration,
}

/// Performance impact analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImpactAnalysis {
    /// Memory allocation overhead
    pub allocation_overhead_ms: f64,
    /// Garbage collection impact
    pub gc_impact_ms: f64,
    /// Memory pressure impact on performance
    pub pressure_impact_percentage: f32,
    /// Cache miss rate due to memory issues
    pub cache_miss_rate: f32,
    /// Overall performance degradation
    pub performance_degradation_percentage: f32,
}

/// Memory optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecommendation {
    /// Recommendation ID
    pub id: String,
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Description
    pub description: String,
    /// Expected benefit
    pub expected_benefit: String,
    /// Implementation effort
    pub implementation_effort: ImplementationEffort,
    /// Estimated memory savings (bytes)
    pub estimated_savings: usize,
}

/// Types of optimization recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Increase pool size
    IncreasePoolSize,
    /// Enable compression
    EnableCompression,
    /// Reduce cache size
    ReduceCacheSize,
    /// Implement lazy loading
    ImplementLazyLoading,
    /// Fix memory leak
    FixMemoryLeak,
    /// Optimize allocation patterns
    OptimizeAllocationPatterns,
    /// Enable garbage collection
    EnableGarbageCollection,
    /// Reduce fragmentation
    ReduceFragmentation,
}

/// Recommendation priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RecommendationPriority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Implementation effort estimates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImplementationEffort {
    /// Minimal effort (configuration change)
    Minimal,
    /// Low effort (simple code change)
    Low,
    /// Medium effort (moderate refactoring)
    Medium,
    /// High effort (significant changes)
    High,
    /// Very high effort (major redesign)
    VeryHigh,
}

/// Detailed memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedMemoryStats {
    /// Allocation rate (allocations per second)
    pub allocation_rate: f64,
    /// Deallocation rate (deallocations per second)
    pub deallocation_rate: f64,
    /// Average allocation size
    pub avg_allocation_size: f64,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// Memory efficiency percentage
    pub memory_efficiency: f32,
    /// Fragmentation index
    pub fragmentation_index: f32,
    /// Pool hit rates
    pub pool_hit_rates: HashMap<String, f32>,
    /// Cache performance metrics
    pub cache_metrics: CachePerformanceMetrics,
}

/// Cache performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePerformanceMetrics {
    /// Cache hit rate
    pub hit_rate: f32,
    /// Cache miss rate
    pub miss_rate: f32,
    /// Average lookup time
    pub avg_lookup_time: Duration,
    /// Cache turnover rate
    pub turnover_rate: f32,
    /// Memory efficiency
    pub memory_efficiency: f32,
}

/// Memory manager for optimization
#[derive(Debug)]
pub struct MemoryManager {
    /// Configuration
    config: MemoryOptimizationConfig,
    /// Memory pools
    embedding_pool: MemoryPool<Vec<f32>>,
    sample_pool: MemoryPool<Vec<f32>>,
    /// Compressed embeddings cache
    compressed_cache: Arc<RwLock<HashMap<String, CompressedEmbedding>>>,
    /// Weak reference cache for automatic cleanup
    weak_refs: Arc<RwLock<HashMap<String, Weak<SpeakerEmbedding>>>>,
    /// Memory usage tracking
    current_memory_usage: Arc<RwLock<usize>>,
    /// Last garbage collection time
    last_gc: Arc<RwLock<Instant>>,
    /// Statistics
    stats: Arc<RwLock<MemoryStats>>,
}

impl MemoryManager {
    /// Create a new memory manager
    pub fn new(config: MemoryOptimizationConfig) -> Result<Self> {
        config.validate()?;

        let embedding_pool = MemoryPool::new(
            config.pool_initial_sizes.embeddings_pool_size,
            config.pool_initial_sizes.embeddings_pool_size * 2,
        );

        let sample_pool = MemoryPool::new(
            config.pool_initial_sizes.samples_pool_size,
            config.pool_initial_sizes.samples_pool_size * 2,
        );

        Ok(Self {
            config,
            embedding_pool,
            sample_pool,
            compressed_cache: Arc::new(RwLock::new(HashMap::new())),
            weak_refs: Arc::new(RwLock::new(HashMap::new())),
            current_memory_usage: Arc::new(RwLock::new(0)),
            last_gc: Arc::new(RwLock::new(Instant::now())),
            stats: Arc::new(RwLock::new(MemoryStats::new())),
        })
    }

    /// Check if garbage collection should run
    pub async fn should_run_gc(&self) -> bool {
        if !self.config.enable_garbage_collection {
            return false;
        }

        let last_gc = *self.last_gc.read().await;
        let memory_usage = *self.current_memory_usage.read().await;

        // Check time-based trigger
        let time_trigger = last_gc.elapsed() >= self.config.gc_interval;

        // Check memory pressure trigger
        let memory_trigger = if self.config.max_memory_usage > 0 {
            memory_usage as f32 / self.config.max_memory_usage as f32
                > self.config.memory_pressure_threshold
        } else {
            false
        };

        time_trigger || memory_trigger
    }

    /// Run garbage collection
    pub async fn run_garbage_collection(&self) -> Result<GarbageCollectionResult> {
        let start_time = Instant::now();
        let initial_memory = *self.current_memory_usage.read().await;

        debug!(
            "Starting garbage collection (memory usage: {} bytes)",
            initial_memory
        );

        // Clean up weak references
        let mut weak_refs = self.weak_refs.write().await;
        let initial_weak_count = weak_refs.len();
        weak_refs.retain(|_, weak_ref| weak_ref.strong_count() > 0);
        let cleaned_weak_refs = initial_weak_count - weak_refs.len();

        // Clean up compressed cache if over limit
        let mut compressed_cache = self.compressed_cache.write().await;
        let initial_cache_count = compressed_cache.len();
        let max_cache_size = self.config.cache_limits.max_embeddings;

        if compressed_cache.len() > max_cache_size {
            // Remove oldest entries (simplified LRU)
            let to_remove = compressed_cache.len() - max_cache_size;
            let keys_to_remove: Vec<String> =
                compressed_cache.keys().take(to_remove).cloned().collect();
            for key in keys_to_remove {
                compressed_cache.remove(&key);
            }
        }
        let cleaned_cache_entries = initial_cache_count - compressed_cache.len();

        // Update last GC time
        let mut last_gc = self.last_gc.write().await;
        *last_gc = Instant::now();

        // Calculate memory savings (estimated)
        let memory_saved = cleaned_weak_refs * 512 + cleaned_cache_entries * 256; // Rough estimate
        let mut current_memory = self.current_memory_usage.write().await;
        *current_memory = current_memory.saturating_sub(memory_saved);
        let final_memory = *current_memory;

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.gc_runs += 1;
        stats.total_memory_freed += memory_saved;
        stats.last_gc_duration = start_time.elapsed();

        let result = GarbageCollectionResult {
            duration: start_time.elapsed(),
            memory_before: initial_memory,
            memory_after: final_memory,
            memory_freed: memory_saved,
            objects_collected: cleaned_weak_refs + cleaned_cache_entries,
            cache_entries_cleaned: cleaned_cache_entries,
        };

        info!(
            "Garbage collection completed: freed {} bytes in {:?}",
            memory_saved, result.duration
        );

        Ok(result)
    }

    /// Compress and cache an embedding
    pub async fn compress_embedding(&self, id: String, embedding: &SpeakerEmbedding) -> Result<()> {
        if !self.config.enable_compression {
            return Ok(());
        }

        let compressed = CompressedEmbedding::compress(embedding, self.config.compression_quality)?;

        let mut cache = self.compressed_cache.write().await;
        let memory_usage = compressed.memory_usage();

        cache.insert(id, compressed);

        // Update memory usage
        let mut current_memory = self.current_memory_usage.write().await;
        *current_memory += memory_usage;

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.compressed_embeddings += 1;
        stats.total_compressed_memory += memory_usage;

        Ok(())
    }

    /// Get compressed embedding from cache
    pub async fn get_compressed_embedding(&self, id: &str) -> Option<CompressedEmbedding> {
        let cache = self.compressed_cache.read().await;
        cache.get(id).cloned()
    }

    /// Get memory pool for embeddings
    pub fn get_embedding_pool(&self) -> &MemoryPool<Vec<f32>> {
        &self.embedding_pool
    }

    /// Get memory pool for samples
    pub fn get_sample_pool(&self) -> &MemoryPool<Vec<f32>> {
        &self.sample_pool
    }

    /// Get current memory usage
    pub async fn get_memory_usage(&self) -> usize {
        *self.current_memory_usage.read().await
    }

    /// Get memory statistics
    pub async fn get_stats(&self) -> MemoryStats {
        self.stats.read().await.clone()
    }

    /// Check if memory pressure is high
    pub async fn is_memory_pressure_high(&self) -> bool {
        let current_usage = *self.current_memory_usage.read().await;
        if self.config.max_memory_usage == 0 {
            return false;
        }

        current_usage as f32 / self.config.max_memory_usage as f32
            > self.config.memory_pressure_threshold
    }

    /// Get memory optimization recommendations
    pub async fn get_optimization_recommendations(&self) -> Vec<MemoryOptimizationRecommendation> {
        let mut recommendations = Vec::new();
        let stats = self.stats.read().await;
        let current_usage = *self.current_memory_usage.read().await;

        // Check memory pressure
        if self.is_memory_pressure_high().await {
            recommendations.push(MemoryOptimizationRecommendation {
                category: OptimizationCategory::MemoryPressure,
                description: "High memory pressure detected. Consider reducing cache sizes or enabling more aggressive compression.".to_string(),
                impact: OptimizationImpact::High,
                estimated_savings: current_usage / 4, // Estimated 25% savings
            });
        }

        // Check GC frequency
        if stats.gc_runs > 0 && stats.last_gc_duration > Duration::from_millis(100) {
            recommendations.push(MemoryOptimizationRecommendation {
                category: OptimizationCategory::GarbageCollection,
                description: "Garbage collection is taking too long. Consider reducing GC frequency or optimizing data structures.".to_string(),
                impact: OptimizationImpact::Medium,
                estimated_savings: 0,
            });
        }

        // Check pool efficiency
        let embedding_pool_stats = self.embedding_pool.get_stats().await;
        if embedding_pool_stats.hit_rate < 0.5 {
            recommendations.push(MemoryOptimizationRecommendation {
                category: OptimizationCategory::PoolEfficiency,
                description: "Memory pool hit rate is low. Consider increasing pool sizes for better efficiency.".to_string(),
                impact: OptimizationImpact::Medium,
                estimated_savings: 0,
            });
        }

        recommendations
    }
}

/// Result of garbage collection operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GarbageCollectionResult {
    /// Duration of garbage collection
    pub duration: Duration,
    /// Memory usage before GC
    pub memory_before: usize,
    /// Memory usage after GC
    pub memory_after: usize,
    /// Memory freed
    pub memory_freed: usize,
    /// Number of objects collected
    pub objects_collected: usize,
    /// Cache entries cleaned
    pub cache_entries_cleaned: usize,
}

/// Memory optimization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Number of garbage collection runs
    pub gc_runs: usize,
    /// Total memory freed by GC
    pub total_memory_freed: usize,
    /// Last garbage collection duration
    pub last_gc_duration: Duration,
    /// Number of compressed embeddings
    pub compressed_embeddings: usize,
    /// Total memory saved by compression
    pub total_compressed_memory: usize,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// Average memory usage
    pub average_memory_usage: f32,
}

impl MemoryStats {
    fn new() -> Self {
        Self {
            gc_runs: 0,
            total_memory_freed: 0,
            last_gc_duration: Duration::ZERO,
            compressed_embeddings: 0,
            total_compressed_memory: 0,
            peak_memory_usage: 0,
            average_memory_usage: 0.0,
        }
    }
}

/// Memory optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizationRecommendation {
    /// Category of optimization
    pub category: OptimizationCategory,
    /// Human-readable description
    pub description: String,
    /// Expected impact of the optimization
    pub impact: OptimizationImpact,
    /// Estimated memory savings in bytes
    pub estimated_savings: usize,
}

/// Categories of memory optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationCategory {
    MemoryPressure,
    GarbageCollection,
    PoolEfficiency,
    CacheOptimization,
    CompressionOpportunity,
}

/// Impact levels for optimizations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationImpact {
    Low,
    Medium,
    High,
    Critical,
}

impl Default for LeakDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            track_stack_traces: false, // Disabled by default for performance
            max_tracked_allocations: 10000,
            detection_interval: Duration::from_secs(60),
            leak_threshold: Duration::from_secs(300), // 5 minutes
            auto_cleanup: false,                      // Conservative default
            pressure_alert_threshold: 0.85,
            enable_profiling: true,
        }
    }
}

impl Default for DurationStats {
    fn default() -> Self {
        Self {
            min: Duration::from_secs(0),
            max: Duration::from_secs(0),
            avg: Duration::from_secs(0),
            total: Duration::from_secs(0),
            count: 0,
        }
    }
}

impl Default for LeakDetectionStats {
    fn default() -> Self {
        Self {
            tracked_allocations: 0,
            potential_leaks: 0,
            confirmed_leaks: 0,
            auto_cleaned_leaks: 0,
            total_leaked_bytes: 0,
            avg_allocation_lifetime: Duration::from_secs(0),
            pressure_events: 0,
            last_scan: SystemTime::now(),
            scan_duration_stats: DurationStats::default(),
        }
    }
}

impl MemoryLeakDetector {
    /// Create new memory leak detector
    pub fn new(config: LeakDetectionConfig) -> Self {
        Self {
            active_allocations: Arc::new(RwLock::new(HashMap::new())),
            allocation_patterns: Arc::new(RwLock::new(HashMap::new())),
            weak_registry: Arc::new(RwLock::new(HashMap::new())),
            pressure_alerts: Arc::new(RwLock::new(Vec::new())),
            leak_stats: Arc::new(RwLock::new(LeakDetectionStats::default())),
            config,
            total_allocations: AtomicU64::new(0),
            total_deallocations: AtomicU64::new(0),
            total_bytes_allocated: AtomicU64::new(0),
            total_bytes_deallocated: AtomicU64::new(0),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(LeakDetectionConfig::default())
    }

    /// Track a new allocation
    pub async fn track_allocation(&self, allocation_info: AllocationInfo) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Update atomic counters
        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.total_bytes_allocated
            .fetch_add(allocation_info.size as u64, Ordering::Relaxed);

        // Check if we've reached the tracking limit
        let mut allocations = self.active_allocations.write().await;
        if allocations.len() >= self.config.max_tracked_allocations {
            // Remove oldest allocation if at limit
            if let Some((oldest_id, _)) = allocations
                .iter()
                .min_by_key(|(_, info)| info.timestamp)
                .map(|(id, info)| (id.clone(), info.clone()))
            {
                allocations.remove(&oldest_id);
            }
        }

        // Add new allocation
        allocations.insert(allocation_info.id.clone(), allocation_info.clone());
        drop(allocations);

        // Update allocation patterns
        self.update_allocation_patterns(&allocation_info).await;

        // Check for memory pressure
        self.check_memory_pressure().await?;

        Ok(())
    }

    /// Track deallocation
    pub async fn track_deallocation(&self, allocation_id: &str, size: usize) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Update atomic counters
        self.total_deallocations.fetch_add(1, Ordering::Relaxed);
        self.total_bytes_deallocated
            .fetch_add(size as u64, Ordering::Relaxed);

        // Remove from active allocations
        let mut allocations = self.active_allocations.write().await;
        if let Some(allocation_info) = allocations.remove(allocation_id) {
            // Update pattern statistics with lifetime information
            let lifetime = SystemTime::now()
                .duration_since(allocation_info.timestamp)
                .unwrap_or_default();
            self.update_allocation_lifetime(&allocation_info.allocation_type, lifetime)
                .await;
        }

        Ok(())
    }

    /// Run leak detection scan
    pub async fn run_leak_detection_scan(&self) -> Result<Vec<String>> {
        if !self.config.enabled {
            return Ok(vec![]);
        }

        let scan_start = Instant::now();
        let mut detected_leaks = Vec::new();

        let allocations = self.active_allocations.read().await;
        let current_time = SystemTime::now();

        for (id, allocation_info) in allocations.iter() {
            let age = current_time
                .duration_since(allocation_info.timestamp)
                .unwrap_or_default();

            if age > self.config.leak_threshold {
                // Potential leak detected
                detected_leaks.push(id.clone());

                warn!(
                    "Potential memory leak detected: allocation {} aged {} seconds",
                    id,
                    age.as_secs()
                );

                // Create memory issue
                let issue = MemoryIssue {
                    id: format!("leak_{}", uuid::Uuid::new_v4()),
                    issue_type: MemoryIssueType::Leak,
                    severity: if age > self.config.leak_threshold * 2 {
                        AlertSeverity::Critical
                    } else {
                        AlertSeverity::Warning
                    },
                    description: format!(
                        "Allocation {} has been active for {} seconds",
                        id,
                        age.as_secs()
                    ),
                    affected_allocations: vec![id.clone()],
                    estimated_impact: allocation_info.size,
                    detected_at: current_time,
                    suggested_resolution: "Check for missing deallocation or reference cycles"
                        .to_string(),
                };

                // Auto-cleanup if enabled and it's a confirmed leak
                if self.config.auto_cleanup && age > self.config.leak_threshold * 3 {
                    self.attempt_leak_cleanup(id, allocation_info).await;
                }
            }
        }

        drop(allocations);

        // Update statistics
        let scan_duration = scan_start.elapsed();
        let mut stats = self.leak_stats.write().await;
        stats.potential_leaks = detected_leaks.len();
        stats.last_scan = current_time;
        stats.scan_duration_stats.count += 1;
        stats.scan_duration_stats.total += scan_duration;
        stats.scan_duration_stats.avg = Duration::from_nanos(
            (stats.scan_duration_stats.total.as_nanos() / stats.scan_duration_stats.count as u128)
                as u64,
        );

        if scan_duration < stats.scan_duration_stats.min
            || stats.scan_duration_stats.min == Duration::from_secs(0)
        {
            stats.scan_duration_stats.min = scan_duration;
        }
        if scan_duration > stats.scan_duration_stats.max {
            stats.scan_duration_stats.max = scan_duration;
        }

        debug!(
            "Leak detection scan completed in {:?}, found {} potential leaks",
            scan_duration,
            detected_leaks.len()
        );

        Ok(detected_leaks)
    }

    /// Generate comprehensive memory audit report
    pub async fn generate_audit_report(&self) -> Result<MemoryAuditReport> {
        let current_time = SystemTime::now();

        // Calculate memory usage breakdown
        let memory_usage = self.calculate_memory_usage_breakdown().await;

        // Detect issues
        let detected_issues = self.detect_memory_issues().await;

        // Analyze allocation patterns
        let allocation_patterns = self.analyze_allocation_patterns().await;

        // Generate leak summary
        let leak_summary = self.generate_leak_summary().await;

        // Analyze performance impact
        let performance_impact = self.analyze_performance_impact().await;

        // Generate recommendations
        let recommendations = self
            .generate_recommendations(&detected_issues, &memory_usage)
            .await;

        // Calculate detailed statistics
        let detailed_stats = self.calculate_detailed_stats().await;

        // Calculate overall health score
        let health_score =
            self.calculate_health_score(&detected_issues, &memory_usage, &performance_impact);

        Ok(MemoryAuditReport {
            timestamp: current_time,
            health_score,
            memory_usage,
            detected_issues,
            allocation_patterns,
            leak_summary,
            performance_impact,
            recommendations,
            detailed_stats,
        })
    }

    /// Update allocation patterns
    async fn update_allocation_patterns(&self, allocation_info: &AllocationInfo) {
        let pattern_key = format!(
            "{:?}_{}",
            allocation_info.allocation_type,
            allocation_info.size / 1024
        ); // Group by type and KB size

        let mut patterns = self.allocation_patterns.write().await;
        let pattern = patterns
            .entry(pattern_key.clone())
            .or_insert_with(|| AllocationPattern {
                name: pattern_key,
                allocation_type: allocation_info.allocation_type,
                frequency: 0,
                avg_size: 0.0,
                peak_size: 0,
                avg_lifetime: Duration::from_secs(0),
                leak_probability: 0.0,
                first_seen: allocation_info.timestamp,
                last_seen: allocation_info.timestamp,
            });

        pattern.frequency += 1;
        pattern.avg_size = (pattern.avg_size * (pattern.frequency - 1) as f64
            + allocation_info.size as f64)
            / pattern.frequency as f64;
        pattern.peak_size = pattern.peak_size.max(allocation_info.size);
        pattern.last_seen = allocation_info.timestamp;
    }

    /// Update allocation lifetime statistics
    async fn update_allocation_lifetime(
        &self,
        allocation_type: &AllocationType,
        lifetime: Duration,
    ) {
        let pattern_key = format!("{:?}_lifetime", allocation_type);

        let mut patterns = self.allocation_patterns.write().await;
        if let Some(pattern) = patterns.get_mut(&pattern_key) {
            let current_avg = pattern.avg_lifetime.as_secs_f64();
            let new_avg = (current_avg * (pattern.frequency - 1) as f64 + lifetime.as_secs_f64())
                / pattern.frequency as f64;
            pattern.avg_lifetime = Duration::from_secs_f64(new_avg);
        }
    }

    /// Check for memory pressure
    async fn check_memory_pressure(&self) -> Result<()> {
        let total_allocated = self.total_bytes_allocated.load(Ordering::Relaxed);
        let total_deallocated = self.total_bytes_deallocated.load(Ordering::Relaxed);
        let current_usage = total_allocated.saturating_sub(total_deallocated);

        // This is a simplified check - in real implementation would use system memory info
        let estimated_limit = 512 * 1024 * 1024; // 512MB estimate
        let usage_percentage = current_usage as f64 / estimated_limit as f64;

        if usage_percentage > self.config.pressure_alert_threshold {
            let alert = MemoryPressureAlert {
                id: format!("pressure_{}", uuid::Uuid::new_v4()),
                severity: if usage_percentage > 0.95 {
                    AlertSeverity::Emergency
                } else if usage_percentage > 0.9 {
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::Warning
                },
                alert_type: AlertType::MemoryPressure,
                message: format!(
                    "Memory usage at {:.1}% of estimated limit",
                    usage_percentage * 100.0
                ),
                current_usage: current_usage as usize,
                memory_limit: estimated_limit,
                suggested_actions: vec![
                    "Run garbage collection".to_string(),
                    "Clear unnecessary caches".to_string(),
                    "Check for memory leaks".to_string(),
                ],
                timestamp: SystemTime::now(),
                metadata: HashMap::new(),
            };

            let mut alerts = self.pressure_alerts.write().await;
            alerts.push(alert);

            let mut stats = self.leak_stats.write().await;
            stats.pressure_events += 1;
        }

        Ok(())
    }

    /// Attempt to cleanup a detected leak
    async fn attempt_leak_cleanup(&self, allocation_id: &str, allocation_info: &AllocationInfo) {
        warn!("Attempting automatic cleanup of leak: {}", allocation_id);

        // In a real implementation, this would try to safely cleanup the leaked resource
        // For now, we'll just log and update statistics

        let mut stats = self.leak_stats.write().await;
        stats.auto_cleaned_leaks += 1;
        stats.total_leaked_bytes += allocation_info.size;

        info!(
            "Attempted cleanup of allocation {} ({} bytes)",
            allocation_id, allocation_info.size
        );
    }

    /// Calculate memory usage breakdown
    async fn calculate_memory_usage_breakdown(&self) -> MemoryUsageBreakdown {
        let allocations = self.active_allocations.read().await;
        let mut by_type = HashMap::new();
        let mut total_allocated = 0;

        for allocation_info in allocations.values() {
            total_allocated += allocation_info.size;
            *by_type.entry(allocation_info.allocation_type).or_insert(0) += allocation_info.size;
        }

        MemoryUsageBreakdown {
            total_allocated,
            by_type,
            by_component: HashMap::new(), // Would be populated with actual component tracking
            pool_usage: HashMap::new(),   // Would be populated with pool statistics
            cache_usage: 0,               // Would be calculated from cache statistics
            fragmentation_percentage: 0.0, // Would be calculated from heap analysis
        }
    }

    /// Detect various memory issues
    async fn detect_memory_issues(&self) -> Vec<MemoryIssue> {
        let mut issues = Vec::new();
        let current_time = SystemTime::now();

        // Check for potential leaks
        let allocations = self.active_allocations.read().await;
        for (id, allocation_info) in allocations.iter() {
            let age = current_time
                .duration_since(allocation_info.timestamp)
                .unwrap_or_default();

            if age > self.config.leak_threshold {
                issues.push(MemoryIssue {
                    id: format!("leak_{}", uuid::Uuid::new_v4()),
                    issue_type: MemoryIssueType::Leak,
                    severity: if age > self.config.leak_threshold * 2 {
                        AlertSeverity::Critical
                    } else {
                        AlertSeverity::Warning
                    },
                    description: format!(
                        "Potential memory leak: allocation active for {} seconds",
                        age.as_secs()
                    ),
                    affected_allocations: vec![id.clone()],
                    estimated_impact: allocation_info.size,
                    detected_at: current_time,
                    suggested_resolution: "Review allocation lifecycle and ensure proper cleanup"
                        .to_string(),
                });
            }
        }

        // Check allocation rate
        let allocation_rate = self.calculate_allocation_rate();
        if allocation_rate > 1000.0 {
            // More than 1000 allocations per second
            issues.push(MemoryIssue {
                id: format!("high_alloc_{}", uuid::Uuid::new_v4()),
                issue_type: MemoryIssueType::ExcessiveAllocation,
                severity: AlertSeverity::Warning,
                description: format!(
                    "High allocation rate: {:.0} allocations/second",
                    allocation_rate
                ),
                affected_allocations: vec![],
                estimated_impact: 0,
                detected_at: current_time,
                suggested_resolution: "Consider object pooling or allocation batching".to_string(),
            });
        }

        issues
    }

    /// Analyze allocation patterns
    async fn analyze_allocation_patterns(&self) -> Vec<AllocationPattern> {
        let patterns = self.allocation_patterns.read().await;
        let mut result = Vec::new();

        for pattern in patterns.values() {
            let mut analyzed_pattern = pattern.clone();

            // Calculate leak probability based on lifetime and frequency
            let avg_lifetime_hours = pattern.avg_lifetime.as_secs_f64() / 3600.0;
            let leak_probability = if avg_lifetime_hours > 1.0 {
                (avg_lifetime_hours / 24.0).min(1.0) as f32 // Higher probability for longer lifetimes
            } else {
                0.0
            };

            analyzed_pattern.leak_probability = leak_probability;
            result.push(analyzed_pattern);
        }

        result
    }

    /// Generate leak summary
    async fn generate_leak_summary(&self) -> LeakSummary {
        let stats = self.leak_stats.read().await;
        let allocations = self.active_allocations.read().await;

        let mut leaks_by_type = HashMap::new();
        let mut total_leaked_bytes = 0;
        let mut oldest_leak_age = Duration::from_secs(0);
        let current_time = SystemTime::now();

        for allocation_info in allocations.values() {
            let age = current_time
                .duration_since(allocation_info.timestamp)
                .unwrap_or_default();
            if age > self.config.leak_threshold {
                *leaks_by_type
                    .entry(allocation_info.allocation_type)
                    .or_insert(0) += 1;
                total_leaked_bytes += allocation_info.size;
                oldest_leak_age = oldest_leak_age.max(age);
            }
        }

        LeakSummary {
            total_potential_leaks: stats.potential_leaks,
            confirmed_leaks: stats.confirmed_leaks,
            false_positives: stats.potential_leaks.saturating_sub(stats.confirmed_leaks),
            total_leaked_bytes,
            leaks_by_type,
            avg_leak_size: if stats.potential_leaks > 0 {
                total_leaked_bytes / stats.potential_leaks
            } else {
                0
            },
            oldest_leak_age,
        }
    }

    /// Analyze performance impact
    async fn analyze_performance_impact(&self) -> PerformanceImpactAnalysis {
        let stats = self.leak_stats.read().await;

        PerformanceImpactAnalysis {
            allocation_overhead_ms: stats.scan_duration_stats.avg.as_secs_f64() * 1000.0,
            gc_impact_ms: 0.0, // Would be calculated from GC statistics
            pressure_impact_percentage: (stats.pressure_events as f32
                / stats.tracked_allocations.max(1) as f32)
                * 100.0,
            cache_miss_rate: 0.0, // Would be calculated from cache statistics
            performance_degradation_percentage: 0.0, // Would be calculated from performance metrics
        }
    }

    /// Generate optimization recommendations
    async fn generate_recommendations(
        &self,
        issues: &[MemoryIssue],
        memory_usage: &MemoryUsageBreakdown,
    ) -> Vec<MemoryRecommendation> {
        let mut recommendations = Vec::new();

        // Check for leaks
        let leak_count = issues
            .iter()
            .filter(|i| i.issue_type == MemoryIssueType::Leak)
            .count();
        if leak_count > 0 {
            recommendations.push(MemoryRecommendation {
                id: format!("fix_leaks_{}", uuid::Uuid::new_v4()),
                recommendation_type: RecommendationType::FixMemoryLeak,
                priority: if leak_count > 10 {
                    RecommendationPriority::Critical
                } else {
                    RecommendationPriority::High
                },
                description: format!("Fix {} detected memory leaks", leak_count),
                expected_benefit: "Reduce memory usage and improve stability".to_string(),
                implementation_effort: ImplementationEffort::Medium,
                estimated_savings: issues
                    .iter()
                    .filter(|i| i.issue_type == MemoryIssueType::Leak)
                    .map(|i| i.estimated_impact)
                    .sum(),
            });
        }

        // Check for high allocation rate
        if issues
            .iter()
            .any(|i| i.issue_type == MemoryIssueType::ExcessiveAllocation)
        {
            recommendations.push(MemoryRecommendation {
                id: format!("optimize_alloc_{}", uuid::Uuid::new_v4()),
                recommendation_type: RecommendationType::OptimizeAllocationPatterns,
                priority: RecommendationPriority::Medium,
                description: "Implement object pooling to reduce allocation overhead".to_string(),
                expected_benefit: "Reduce allocation rate and improve performance".to_string(),
                implementation_effort: ImplementationEffort::Medium,
                estimated_savings: memory_usage.total_allocated / 10, // Estimate 10% savings
            });
        }

        // Check for large memory usage
        if memory_usage.total_allocated > 256 * 1024 * 1024 {
            // 256MB
            recommendations.push(MemoryRecommendation {
                id: format!("enable_compression_{}", uuid::Uuid::new_v4()),
                recommendation_type: RecommendationType::EnableCompression,
                priority: RecommendationPriority::Medium,
                description: "Enable data compression to reduce memory footprint".to_string(),
                expected_benefit: "Reduce memory usage by 30-50%".to_string(),
                implementation_effort: ImplementationEffort::Low,
                estimated_savings: memory_usage.total_allocated / 3, // Estimate 33% savings
            });
        }

        recommendations
    }

    /// Calculate detailed statistics
    async fn calculate_detailed_stats(&self) -> DetailedMemoryStats {
        let allocation_rate = self.calculate_allocation_rate();
        let deallocation_rate = self.calculate_deallocation_rate();

        DetailedMemoryStats {
            allocation_rate,
            deallocation_rate,
            avg_allocation_size: self.calculate_avg_allocation_size(),
            peak_memory_usage: self.total_bytes_allocated.load(Ordering::Relaxed) as usize,
            memory_efficiency: self.calculate_memory_efficiency(),
            fragmentation_index: 0.0, // Would be calculated from heap analysis
            pool_hit_rates: HashMap::new(), // Would be populated with pool statistics
            cache_metrics: CachePerformanceMetrics {
                hit_rate: 0.0,
                miss_rate: 0.0,
                avg_lookup_time: Duration::from_nanos(0),
                turnover_rate: 0.0,
                memory_efficiency: 0.0,
            },
        }
    }

    /// Calculate overall health score
    fn calculate_health_score(
        &self,
        issues: &[MemoryIssue],
        memory_usage: &MemoryUsageBreakdown,
        performance_impact: &PerformanceImpactAnalysis,
    ) -> f32 {
        let mut score = 1.0;

        // Penalize for issues
        let critical_issues = issues
            .iter()
            .filter(|i| i.severity == AlertSeverity::Critical)
            .count();
        let warning_issues = issues
            .iter()
            .filter(|i| i.severity == AlertSeverity::Warning)
            .count();

        score -= (critical_issues as f32 * 0.2) + (warning_issues as f32 * 0.1);

        // Penalize for high memory usage
        let memory_usage_mb = memory_usage.total_allocated as f32 / (1024.0 * 1024.0);
        if memory_usage_mb > 512.0 {
            score -= 0.2;
        } else if memory_usage_mb > 256.0 {
            score -= 0.1;
        }

        // Penalize for performance impact
        score -= performance_impact.performance_degradation_percentage / 100.0;

        score.clamp(0.0, 1.0)
    }

    /// Calculate allocation rate
    fn calculate_allocation_rate(&self) -> f64 {
        // Simplified calculation - in real implementation would track over time windows
        let total_allocations = self.total_allocations.load(Ordering::Relaxed);
        total_allocations as f64 / 60.0 // Allocations per minute, converted to per second
    }

    /// Calculate deallocation rate
    fn calculate_deallocation_rate(&self) -> f64 {
        let total_deallocations = self.total_deallocations.load(Ordering::Relaxed);
        total_deallocations as f64 / 60.0 // Deallocations per minute, converted to per second
    }

    /// Calculate average allocation size
    fn calculate_avg_allocation_size(&self) -> f64 {
        let total_allocations = self.total_allocations.load(Ordering::Relaxed);
        let total_bytes = self.total_bytes_allocated.load(Ordering::Relaxed);

        if total_allocations > 0 {
            total_bytes as f64 / total_allocations as f64
        } else {
            0.0
        }
    }

    /// Calculate memory efficiency
    fn calculate_memory_efficiency(&self) -> f32 {
        let allocated = self.total_bytes_allocated.load(Ordering::Relaxed);
        let deallocated = self.total_bytes_deallocated.load(Ordering::Relaxed);

        if allocated > 0 {
            deallocated as f32 / allocated as f32
        } else {
            1.0
        }
    }

    /// Get current statistics
    pub async fn get_statistics(&self) -> LeakDetectionStats {
        self.leak_stats.read().await.clone()
    }

    /// Clear old alerts
    pub async fn clear_old_alerts(&self, max_age: Duration) {
        let mut alerts = self.pressure_alerts.write().await;
        let cutoff_time = SystemTime::now() - max_age;
        alerts.retain(|alert| alert.timestamp > cutoff_time);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_memory_optimization_config_default() {
        let config = MemoryOptimizationConfig::default();
        assert!(config.enable_optimization);
        assert_eq!(config.max_memory_usage, 512 * 1024 * 1024);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_memory_optimization_config_mobile() {
        let config = MemoryOptimizationConfig::mobile_optimized();
        assert!(config.enable_optimization);
        assert_eq!(config.max_memory_usage, 128 * 1024 * 1024);
        assert_eq!(config.memory_pressure_threshold, 0.7);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_memory_optimization_config_edge() {
        let config = MemoryOptimizationConfig::edge_optimized();
        assert!(config.enable_optimization);
        assert_eq!(config.max_memory_usage, 64 * 1024 * 1024);
        assert_eq!(config.memory_pressure_threshold, 0.6);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_memory_pool_sizes() {
        let default_sizes = MemoryPoolSizes::default();
        let mobile_sizes = MemoryPoolSizes::mobile_optimized();
        let edge_sizes = MemoryPoolSizes::edge_optimized();

        assert!(mobile_sizes.embeddings_pool_size < default_sizes.embeddings_pool_size);
        assert!(edge_sizes.embeddings_pool_size < mobile_sizes.embeddings_pool_size);
    }

    #[test]
    fn test_cache_limits() {
        let default_limits = CacheLimits::default();
        let mobile_limits = CacheLimits::mobile_optimized();
        let edge_limits = CacheLimits::edge_optimized();

        assert!(mobile_limits.max_speaker_profiles < default_limits.max_speaker_profiles);
        assert!(edge_limits.max_speaker_profiles < mobile_limits.max_speaker_profiles);
    }

    #[tokio::test]
    async fn test_memory_pool_basic_operations() {
        let pool: MemoryPool<Vec<f32>> = MemoryPool::new(5, 10);

        // Get object from pool
        let pooled_obj = pool.get().await;
        assert!(pooled_obj.get().is_some());

        // Check stats
        let stats = pool.get_stats().await;
        assert_eq!(stats.allocation_count, 1);
        assert!(stats.current_size <= 10);
    }

    #[test]
    fn test_compressed_embedding() {
        let embedding = SpeakerEmbedding::new(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        let compressed = CompressedEmbedding::compress(&embedding, 0.7).unwrap();
        assert!(compressed.compression_ratio > 1.0);
        assert!(compressed.quality_score > 0.0);

        let decompressed = compressed.decompress().unwrap();
        assert_eq!(decompressed.vector.len(), embedding.vector.len());
    }

    #[tokio::test]
    async fn test_memory_manager_creation() {
        let config = MemoryOptimizationConfig::default();
        let manager = MemoryManager::new(config);
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_memory_manager_gc_check() {
        let config = MemoryOptimizationConfig::mobile_optimized();
        let manager = MemoryManager::new(config).unwrap();

        // Initially should not need GC
        assert!(!manager.should_run_gc().await);
    }

    #[test]
    fn test_memory_stats() {
        let stats = MemoryStats::new();
        assert_eq!(stats.gc_runs, 0);
        assert_eq!(stats.total_memory_freed, 0);
        assert_eq!(stats.compressed_embeddings, 0);
    }

    #[test]
    fn test_garbage_collection_result() {
        let result = GarbageCollectionResult {
            duration: Duration::from_millis(50),
            memory_before: 1000,
            memory_after: 800,
            memory_freed: 200,
            objects_collected: 5,
            cache_entries_cleaned: 3,
        };

        assert_eq!(result.memory_freed, 200);
        assert_eq!(result.objects_collected, 5);
    }
}
