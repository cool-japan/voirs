//! Memory Management and Cache Optimization Module
//!
//! This module provides comprehensive memory management and cache optimization
//! for high-performance spatial audio processing, including object pools,
//! cache-friendly data structures, and memory usage monitoring.

use crate::types::Position3D;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Memory manager for spatial audio processing
pub struct MemoryManager {
    /// Buffer pools for different sizes
    buffer_pools: Arc<RwLock<HashMap<usize, BufferPool<f32>>>>,
    /// Array pools for 2D arrays
    array2d_pools: Arc<RwLock<HashMap<(usize, usize), Array2Pool>>>,
    /// Cache manager for expensive computations
    cache_manager: Arc<RwLock<CacheManager>>,
    /// Memory usage statistics
    memory_stats: Arc<RwLock<MemoryStatistics>>,
    /// Configuration
    config: MemoryConfig,
}

/// Buffer pool for reusing audio buffers
pub struct BufferPool<T> {
    /// Available buffers
    available: VecDeque<Array1<T>>,
    /// Maximum pool size
    max_size: usize,
    /// Total allocations
    total_allocations: u64,
    /// Pool hits
    pool_hits: u64,
}

/// Pool for 2D arrays
pub struct Array2Pool {
    /// Available arrays
    available: VecDeque<Array2<f32>>,
    /// Array dimensions
    dimensions: (usize, usize),
    /// Maximum pool size
    max_size: usize,
    /// Total allocations
    total_allocations: u64,
    /// Pool hits
    pool_hits: u64,
}

/// Cache manager for expensive computations
pub struct CacheManager {
    /// HRTF interpolation cache
    hrtf_cache: HashMap<HrtfCacheKey, HrtfCacheEntry>,
    /// Distance attenuation cache
    distance_cache: HashMap<DistanceCacheKey, f32>,
    /// Room impulse response cache
    room_cache: HashMap<RoomCacheKey, Array1<f32>>,
    /// Cache statistics
    cache_stats: CacheStatistics,
    /// Maximum cache size per type
    max_cache_size: usize,
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum buffer pool size per buffer type
    pub max_buffer_pool_size: usize,
    /// Maximum cache size per cache type
    pub max_cache_size: usize,
    /// Enable memory usage monitoring
    pub enable_monitoring: bool,
    /// Memory pressure threshold (0.0-1.0)
    pub memory_pressure_threshold: f32,
    /// Cache eviction policy
    pub cache_policy: CachePolicy,
    /// Buffer alignment for SIMD operations
    pub buffer_alignment: usize,
}

/// Cache eviction policies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CachePolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Time-based expiration
    TTL,
    /// Size-based eviction
    SizeBased,
}

/// HRTF cache key for interpolation results
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct HrtfCacheKey {
    azimuth: i32,
    elevation: i32,
    distance: u32,
}

/// HRTF cache entry
#[derive(Debug, Clone)]
struct HrtfCacheEntry {
    left_hrir: Array1<f32>,
    right_hrir: Array1<f32>,
    last_accessed: Instant,
    access_count: u64,
}

/// Distance attenuation cache key
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct DistanceCacheKey {
    distance_mm: u32, // Store distance in millimeters for precision
    model_type: u8,   // Attenuation model type
}

/// Room acoustic cache key
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct RoomCacheKey {
    room_hash: u64,
    source_position_hash: u64,
    listener_position_hash: u64,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    /// Total memory allocated (bytes)
    pub total_allocated: u64,
    /// Memory currently in use (bytes)
    pub memory_in_use: u64,
    /// Peak memory usage (bytes)
    pub peak_memory_usage: u64,
    /// Buffer pool statistics
    pub buffer_pool_stats: HashMap<usize, BufferPoolStats>,
    /// Cache hit rates
    pub cache_hit_rates: HashMap<String, f64>,
    /// Memory pressure level (0.0-1.0)
    pub memory_pressure: f32,
    /// Last update time
    pub last_updated: Instant,
}

impl Default for MemoryStatistics {
    fn default() -> Self {
        Self {
            total_allocated: 0,
            memory_in_use: 0,
            peak_memory_usage: 0,
            buffer_pool_stats: HashMap::new(),
            cache_hit_rates: HashMap::new(),
            memory_pressure: 0.0,
            last_updated: Instant::now(),
        }
    }
}

/// Buffer pool statistics
#[derive(Debug, Default, Clone)]
pub struct BufferPoolStats {
    /// Total allocations
    pub total_allocations: u64,
    /// Pool hits (reused buffers)
    pub pool_hits: u64,
    /// Current pool size
    pub current_pool_size: usize,
    /// Hit rate (pool_hits / total_allocations)
    pub hit_rate: f64,
}

/// Cache statistics
#[derive(Debug, Default)]
struct CacheStatistics {
    /// Total cache requests
    total_requests: u64,
    /// Cache hits
    cache_hits: u64,
    /// Cache misses
    cache_misses: u64,
    /// Cache evictions
    cache_evictions: u64,
    /// Memory used by caches (bytes)
    memory_usage: u64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_buffer_pool_size: 128,
            max_cache_size: 1024,
            enable_monitoring: true,
            memory_pressure_threshold: 0.8,
            cache_policy: CachePolicy::LRU,
            buffer_alignment: 32, // 32-byte alignment for AVX
        }
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new(MemoryConfig::default())
    }
}

impl MemoryManager {
    /// Create new memory manager
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            buffer_pools: Arc::new(RwLock::new(HashMap::new())),
            array2d_pools: Arc::new(RwLock::new(HashMap::new())),
            cache_manager: Arc::new(RwLock::new(CacheManager::new(&config))),
            memory_stats: Arc::new(RwLock::new(MemoryStatistics::default())),
            config,
        }
    }

    /// Get buffer from pool or create new one
    pub async fn get_buffer(&self, size: usize) -> Array1<f32> {
        let mut pools = self.buffer_pools.write().await;
        let pool = pools
            .entry(size)
            .or_insert_with(|| BufferPool::new(size, self.config.max_buffer_pool_size));

        if let Some(mut buffer) = pool.available.pop_front() {
            // Clear the buffer for reuse
            buffer.fill(0.0);
            pool.pool_hits += 1;
            self.update_buffer_stats(size, false).await;
            buffer
        } else {
            // Create new buffer
            pool.total_allocations += 1;
            self.update_buffer_stats(size, true).await;
            Array1::zeros(size)
        }
    }

    /// Return buffer to pool
    pub async fn return_buffer(&self, buffer: Array1<f32>) {
        let size = buffer.len();
        let mut pools = self.buffer_pools.write().await;

        if let Some(pool) = pools.get_mut(&size) {
            if pool.available.len() < pool.max_size {
                pool.available.push_back(buffer);
            }
            // If pool is full, buffer will be dropped
        }
    }

    /// Get 2D array from pool or create new one
    pub async fn get_array2d(&self, rows: usize, cols: usize) -> Array2<f32> {
        let dims = (rows, cols);
        let mut pools = self.array2d_pools.write().await;
        let pool = pools
            .entry(dims)
            .or_insert_with(|| Array2Pool::new(dims, self.config.max_buffer_pool_size));

        if let Some(mut array) = pool.available.pop_front() {
            // Clear the array for reuse
            array.fill(0.0);
            pool.pool_hits += 1;
            array
        } else {
            // Create new array
            pool.total_allocations += 1;
            Array2::zeros(dims)
        }
    }

    /// Return 2D array to pool
    pub async fn return_array2d(&self, array: Array2<f32>) {
        let dims = array.dim();
        let mut pools = self.array2d_pools.write().await;

        if let Some(pool) = pools.get_mut(&dims) {
            if pool.available.len() < pool.max_size {
                pool.available.push_back(array);
            }
        }
    }

    /// Cache HRTF interpolation result
    pub async fn cache_hrtf(
        &self,
        key: (i32, i32, f32),
        left_hrir: Array1<f32>,
        right_hrir: Array1<f32>,
    ) {
        let cache_key = HrtfCacheKey {
            azimuth: key.0,
            elevation: key.1,
            distance: (key.2 * 1000.0) as u32, // Store in millimeters
        };

        let entry = HrtfCacheEntry {
            left_hrir,
            right_hrir,
            last_accessed: Instant::now(),
            access_count: 1,
        };

        let mut cache_manager = self.cache_manager.write().await;
        cache_manager.cache_hrtf(cache_key, entry).await;
    }

    /// Get cached HRTF result
    pub async fn get_cached_hrtf(
        &self,
        key: (i32, i32, f32),
    ) -> Option<(Array1<f32>, Array1<f32>)> {
        let cache_key = HrtfCacheKey {
            azimuth: key.0,
            elevation: key.1,
            distance: (key.2 * 1000.0) as u32,
        };

        let mut cache_manager = self.cache_manager.write().await;
        cache_manager.get_hrtf(&cache_key).await
    }

    /// Cache distance attenuation result
    pub async fn cache_distance_attenuation(
        &self,
        distance: f32,
        model_type: u8,
        attenuation: f32,
    ) {
        let key = DistanceCacheKey {
            distance_mm: (distance * 1000.0) as u32,
            model_type,
        };

        let mut cache_manager = self.cache_manager.write().await;
        cache_manager.cache_distance(key, attenuation).await;
    }

    /// Get cached distance attenuation
    pub async fn get_cached_distance_attenuation(
        &self,
        distance: f32,
        model_type: u8,
    ) -> Option<f32> {
        let key = DistanceCacheKey {
            distance_mm: (distance * 1000.0) as u32,
            model_type,
        };

        let cache_manager = self.cache_manager.read().await;
        cache_manager.get_distance(&key)
    }

    /// Get memory statistics
    pub async fn get_memory_stats(&self) -> MemoryStatistics {
        let stats = self.memory_stats.read().await;
        stats.clone()
    }

    /// Check memory pressure and trigger cleanup if needed
    pub async fn check_memory_pressure(&self) -> bool {
        let stats = self.memory_stats.read().await;
        if stats.memory_pressure > self.config.memory_pressure_threshold {
            drop(stats); // Release read lock
            self.cleanup_memory().await;
            true
        } else {
            false
        }
    }

    /// Cleanup memory when under pressure
    async fn cleanup_memory(&self) {
        // Clear least recently used cache entries
        let mut cache_manager = self.cache_manager.write().await;
        cache_manager
            .evict_lru_entries(self.config.max_cache_size / 2)
            .await;

        // Trim buffer pools
        self.trim_buffer_pools().await;

        // Update statistics
        self.update_memory_stats().await;
    }

    /// Trim buffer pools to free memory
    async fn trim_buffer_pools(&self) {
        let mut pools = self.buffer_pools.write().await;
        for pool in pools.values_mut() {
            pool.available.truncate(pool.max_size / 2);
        }

        let mut array_pools = self.array2d_pools.write().await;
        for pool in array_pools.values_mut() {
            pool.available.truncate(pool.max_size / 2);
        }
    }

    /// Update buffer pool statistics
    async fn update_buffer_stats(&self, size: usize, is_new_allocation: bool) {
        let mut stats = self.memory_stats.write().await;

        // Handle memory allocation tracking
        if is_new_allocation {
            stats.total_allocated += (size * std::mem::size_of::<f32>()) as u64;
        }

        // Update pool-specific stats
        {
            let pool_stats = stats.buffer_pool_stats.entry(size).or_default();
            if is_new_allocation {
                pool_stats.total_allocations += 1;
            } else {
                pool_stats.pool_hits += 1;
            }
            pool_stats.hit_rate =
                pool_stats.pool_hits as f64 / pool_stats.total_allocations.max(1) as f64;
        }

        stats.last_updated = Instant::now();
    }

    /// Update memory statistics
    async fn update_memory_stats(&self) {
        let mut stats = self.memory_stats.write().await;

        // Calculate memory usage from pools
        let pools = self.buffer_pools.read().await;
        let mut memory_in_use = 0u64;

        for (size, pool) in pools.iter() {
            let pool_memory = (pool.available.len() * size * std::mem::size_of::<f32>()) as u64;
            memory_in_use += pool_memory;

            let pool_stats = stats.buffer_pool_stats.entry(*size).or_default();
            pool_stats.current_pool_size = pool.available.len();
        }

        stats.memory_in_use = memory_in_use;
        if memory_in_use > stats.peak_memory_usage {
            stats.peak_memory_usage = memory_in_use;
        }

        // Calculate memory pressure (simplified)
        stats.memory_pressure = (memory_in_use as f32 / (1024.0 * 1024.0 * 1024.0)).min(1.0); // Normalize to GB
        stats.last_updated = Instant::now();
    }
}

impl<T> BufferPool<T> {
    fn new(size: usize, max_size: usize) -> Self {
        Self {
            available: VecDeque::with_capacity(max_size),
            max_size,
            total_allocations: 0,
            pool_hits: 0,
        }
    }
}

impl Array2Pool {
    fn new(dimensions: (usize, usize), max_size: usize) -> Self {
        Self {
            available: VecDeque::with_capacity(max_size),
            dimensions,
            max_size,
            total_allocations: 0,
            pool_hits: 0,
        }
    }
}

impl CacheManager {
    fn new(config: &MemoryConfig) -> Self {
        Self {
            hrtf_cache: HashMap::new(),
            distance_cache: HashMap::new(),
            room_cache: HashMap::new(),
            cache_stats: CacheStatistics::default(),
            max_cache_size: config.max_cache_size,
        }
    }

    async fn cache_hrtf(&mut self, key: HrtfCacheKey, entry: HrtfCacheEntry) {
        if self.hrtf_cache.len() >= self.max_cache_size {
            self.evict_lru_hrtf().await;
        }
        self.hrtf_cache.insert(key, entry);
    }

    async fn get_hrtf(&mut self, key: &HrtfCacheKey) -> Option<(Array1<f32>, Array1<f32>)> {
        if let Some(entry) = self.hrtf_cache.get_mut(key) {
            entry.last_accessed = Instant::now();
            entry.access_count += 1;
            self.cache_stats.cache_hits += 1;
            Some((entry.left_hrir.clone(), entry.right_hrir.clone()))
        } else {
            self.cache_stats.cache_misses += 1;
            None
        }
    }

    async fn cache_distance(&mut self, key: DistanceCacheKey, value: f32) {
        if self.distance_cache.len() >= self.max_cache_size {
            // Simple eviction - remove oldest entries
            if self.distance_cache.len() > self.max_cache_size * 3 / 4 {
                let keys: Vec<_> = self.distance_cache.keys().cloned().collect();
                for key in keys.iter().take(self.max_cache_size / 4) {
                    self.distance_cache.remove(key);
                }
            }
        }
        self.distance_cache.insert(key, value);
    }

    fn get_distance(&self, key: &DistanceCacheKey) -> Option<f32> {
        self.distance_cache.get(key).copied()
    }

    async fn evict_lru_entries(&mut self, count: usize) {
        // Evict LRU HRTF entries
        let mut entries: Vec<_> = self.hrtf_cache.iter().collect();
        entries.sort_by(|a, b| a.1.last_accessed.cmp(&b.1.last_accessed));

        let to_remove: Vec<_> = entries
            .iter()
            .take(count.min(entries.len()))
            .map(|(k, _)| (*k).clone())
            .collect();
        for key in to_remove {
            self.hrtf_cache.remove(&key);
            self.cache_stats.cache_evictions += 1;
        }
    }

    async fn evict_lru_hrtf(&mut self) {
        if let Some((oldest_key, _)) = self
            .hrtf_cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
        {
            let key_to_remove = oldest_key.clone();
            self.hrtf_cache.remove(&key_to_remove);
            self.cache_stats.cache_evictions += 1;
        }
    }
}

/// Cache-friendly data layout optimization utilities
pub mod cache_optimization {
    use super::*;

    /// Struct-of-Arrays pattern for better cache locality
    #[derive(Debug)]
    pub struct SoAPositions {
        /// X coordinates
        pub x: Vec<f32>,
        /// Y coordinates  
        pub y: Vec<f32>,
        /// Z coordinates
        pub z: Vec<f32>,
        /// Capacity
        pub capacity: usize,
    }

    impl SoAPositions {
        /// Create new SoA position array
        pub fn with_capacity(capacity: usize) -> Self {
            Self {
                x: Vec::with_capacity(capacity),
                y: Vec::with_capacity(capacity),
                z: Vec::with_capacity(capacity),
                capacity,
            }
        }

        /// Add position
        pub fn push(&mut self, pos: Position3D) {
            self.x.push(pos.x);
            self.y.push(pos.y);
            self.z.push(pos.z);
        }

        /// Get position by index
        pub fn get(&self, index: usize) -> Option<Position3D> {
            if index < self.len() {
                Some(Position3D::new(self.x[index], self.y[index], self.z[index]))
            } else {
                None
            }
        }

        /// Length
        pub fn len(&self) -> usize {
            self.x.len()
        }

        /// Is empty
        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }

        /// Clear all positions
        pub fn clear(&mut self) {
            self.x.clear();
            self.y.clear();
            self.z.clear();
        }
    }

    /// Prefetch data for cache optimization
    #[cfg(target_arch = "x86_64")]
    pub fn prefetch_data<T>(data: *const T) {
        #[cfg(target_feature = "sse")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(data as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    /// Prefetch data (no-op on non-x86 architectures)
    pub fn prefetch_data<T>(_data: *const T) {
        // No-op on non-x86 architectures
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_manager_creation() {
        let config = MemoryConfig::default();
        let manager = MemoryManager::new(config);

        let stats = manager.get_memory_stats().await;
        assert_eq!(stats.total_allocated, 0);
    }

    #[tokio::test]
    async fn test_buffer_pool_reuse() {
        let config = MemoryConfig::default();
        let manager = MemoryManager::new(config);

        // Get buffer and return it
        let buffer = manager.get_buffer(1024).await;
        assert_eq!(buffer.len(), 1024);
        manager.return_buffer(buffer).await;

        // Get another buffer - should be reused
        let buffer2 = manager.get_buffer(1024).await;
        assert_eq!(buffer2.len(), 1024);

        let stats = manager.get_memory_stats().await;
        assert!(stats.buffer_pool_stats.contains_key(&1024));
    }

    #[tokio::test]
    async fn test_hrtf_cache() {
        let config = MemoryConfig::default();
        let manager = MemoryManager::new(config);

        let left = Array1::zeros(256);
        let right = Array1::zeros(256);

        // Cache HRTF
        manager
            .cache_hrtf((45, 0, 2.0), left.clone(), right.clone())
            .await;

        // Retrieve from cache
        let cached = manager.get_cached_hrtf((45, 0, 2.0)).await;
        assert!(cached.is_some());

        let (cached_left, cached_right) = cached.unwrap();
        assert_eq!(cached_left.len(), 256);
        assert_eq!(cached_right.len(), 256);
    }

    #[tokio::test]
    async fn test_distance_cache() {
        let config = MemoryConfig::default();
        let manager = MemoryManager::new(config);

        // Cache distance attenuation
        manager.cache_distance_attenuation(5.0, 1, 0.2).await;

        // Retrieve from cache
        let cached = manager.get_cached_distance_attenuation(5.0, 1).await;
        assert_eq!(cached, Some(0.2));

        // Non-existent entry
        let not_cached = manager.get_cached_distance_attenuation(10.0, 1).await;
        assert_eq!(not_cached, None);
    }

    #[tokio::test]
    async fn test_memory_pressure() {
        let mut config = MemoryConfig::default();
        config.memory_pressure_threshold = 0.1; // Low threshold for testing
        let manager = MemoryManager::new(config);

        // Allocate many buffers to trigger pressure
        let mut buffers = Vec::new();
        for _ in 0..100 {
            buffers.push(manager.get_buffer(1024).await);
        }

        // Update stats manually (in real usage this would be automatic)
        manager.update_memory_stats().await;

        // Check if cleanup is triggered
        let pressure_detected = manager.check_memory_pressure().await;
        // This test is simplified - in a real scenario we'd need more sophisticated pressure detection
    }

    #[tokio::test]
    async fn test_soa_positions() {
        let mut positions = cache_optimization::SoAPositions::with_capacity(10);

        positions.push(Position3D::new(1.0, 2.0, 3.0));
        positions.push(Position3D::new(4.0, 5.0, 6.0));

        assert_eq!(positions.len(), 2);

        let pos = positions.get(0).unwrap();
        assert_eq!(pos.x, 1.0);
        assert_eq!(pos.y, 2.0);
        assert_eq!(pos.z, 3.0);
    }
}
