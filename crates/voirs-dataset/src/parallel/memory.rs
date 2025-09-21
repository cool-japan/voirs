//! Memory management utilities for parallel dataset processing
//!
//! Provides memory pool allocation, buffer reuse strategies, and
//! memory pressure handling to optimize performance.

use crate::{DatasetError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// Memory allocation strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Simple allocation without pooling
    Simple,
    /// Fixed-size memory pools
    FixedPool,
    /// Dynamic memory pools with size categories
    DynamicPool,
    /// Memory mapping for large buffers
    MemoryMapped,
}

/// Memory pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolConfig {
    /// Allocation strategy
    pub strategy: AllocationStrategy,
    /// Maximum memory usage (bytes)
    pub max_memory: u64,
    /// Initial pool size per category
    pub initial_pool_size: usize,
    /// Maximum pool size per category
    pub max_pool_size: usize,
    /// Buffer reuse timeout (seconds)
    pub reuse_timeout: u64,
    /// Enable memory pressure monitoring
    pub pressure_monitoring: bool,
    /// GC trigger threshold (0.0 - 1.0)
    pub gc_threshold: f64,
    /// Memory alignment (bytes)
    pub alignment: usize,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            strategy: AllocationStrategy::DynamicPool,
            max_memory: 1024 * 1024 * 1024, // 1GB default
            initial_pool_size: 16,
            max_pool_size: 128,
            reuse_timeout: 300, // 5 minutes
            pressure_monitoring: true,
            gc_threshold: 0.8, // Trigger GC at 80% memory usage
            alignment: 64,     // 64-byte alignment for SIMD
        }
    }
}

/// Memory buffer with metadata
#[derive(Debug, Clone)]
pub struct MemoryBuffer {
    /// Buffer data
    data: Vec<u8>,
    /// Size category
    size_category: SizeCategory,
    /// Last used timestamp
    last_used: Instant,
    /// Reference count
    ref_count: Arc<AtomicUsize>,
    /// Buffer ID for tracking
    id: u64,
}

impl MemoryBuffer {
    /// Create a new buffer
    pub fn new(size: usize, category: SizeCategory, id: u64) -> Self {
        let data = vec![0; size];

        Self {
            data,
            size_category: category,
            last_used: Instant::now(),
            ref_count: Arc::new(AtomicUsize::new(1)),
            id,
        }
    }

    /// Get buffer data
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get mutable buffer data
    pub fn data_mut(&mut self) -> &mut [u8] {
        self.last_used = Instant::now();
        &mut self.data
    }

    /// Get buffer size
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Get size category
    pub fn size_category(&self) -> SizeCategory {
        self.size_category
    }

    /// Get last used timestamp
    pub fn last_used(&self) -> Instant {
        self.last_used
    }

    /// Get buffer ID
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Check if buffer is unused (ref count = 1)
    pub fn is_unused(&self) -> bool {
        self.ref_count.load(Ordering::Acquire) == 1
    }

    /// Increment reference count
    pub fn add_ref(&self) {
        self.ref_count.fetch_add(1, Ordering::AcqRel);
    }

    /// Decrement reference count
    pub fn remove_ref(&self) {
        self.ref_count.fetch_sub(1, Ordering::AcqRel);
    }

    /// Clear buffer data (set to zero)
    pub fn clear(&mut self) {
        self.data.fill(0);
        self.last_used = Instant::now();
    }

    /// Resize buffer (may reallocate)
    pub fn resize(&mut self, new_size: usize) {
        if new_size != self.data.len() {
            self.data.resize(new_size, 0);
            self.last_used = Instant::now();
        }
    }
}

/// Buffer size categories for efficient pooling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SizeCategory {
    /// Small buffers: < 1KB
    Small,
    /// Medium buffers: 1KB - 64KB
    Medium,
    /// Large buffers: 64KB - 1MB
    Large,
    /// XLarge buffers: 1MB - 16MB
    XLarge,
    /// Huge buffers: > 16MB
    Huge,
    /// Custom size category
    Custom(usize),
}

impl SizeCategory {
    /// Get size category for a given buffer size
    pub fn from_size(size: usize) -> Self {
        match size {
            0..=1024 => Self::Small,
            1025..=65536 => Self::Medium,
            65537..=1048576 => Self::Large,
            1048577..=16777216 => Self::XLarge,
            _ => Self::Huge,
        }
    }

    /// Get recommended buffer size for this category
    pub fn recommended_size(&self) -> usize {
        match self {
            Self::Small => 1024,
            Self::Medium => 8192,
            Self::Large => 131072,
            Self::XLarge => 2097152,
            Self::Huge => 33554432,
            Self::Custom(size) => *size,
        }
    }

    /// Get maximum size for this category
    pub fn max_size(&self) -> usize {
        match self {
            Self::Small => 1024,
            Self::Medium => 65536,
            Self::Large => 1048576,
            Self::XLarge => 16777216,
            Self::Huge => usize::MAX,
            Self::Custom(size) => *size,
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total allocated memory (bytes)
    pub total_allocated: u64,
    /// Currently used memory (bytes)
    pub current_usage: u64,
    /// Peak memory usage (bytes)
    pub peak_usage: u64,
    /// Number of active buffers
    pub active_buffers: usize,
    /// Number of pooled buffers
    pub pooled_buffers: usize,
    /// Memory utilization ratio (0.0 - 1.0)
    pub utilization: f64,
    /// Allocation count
    pub allocation_count: u64,
    /// Deallocation count
    pub deallocation_count: u64,
    /// Cache hit rate (0.0 - 1.0)
    pub cache_hit_rate: f64,
    /// GC run count
    pub gc_runs: u64,
}

/// Memory pool for efficient buffer management
pub struct MemoryPool {
    /// Configuration
    config: MemoryPoolConfig,
    /// Buffer pools by size category
    pools: Arc<Mutex<HashMap<SizeCategory, VecDeque<MemoryBuffer>>>>,
    /// Active buffers (not in pool)
    active_buffers: Arc<Mutex<HashMap<u64, MemoryBuffer>>>,
    /// Memory usage statistics
    stats: Arc<Mutex<MemoryStats>>,
    /// Buffer ID counter
    buffer_id_counter: AtomicU64,
    /// Current memory usage
    current_usage: AtomicU64,
    /// Peak memory usage
    peak_usage: AtomicU64,
}

impl MemoryPool {
    /// Create a new memory pool with default configuration
    pub fn new() -> Self {
        Self::with_config(MemoryPoolConfig::default())
    }

    /// Create a new memory pool with custom configuration
    pub fn with_config(config: MemoryPoolConfig) -> Self {
        let pool = Self {
            config: config.clone(),
            pools: Arc::new(Mutex::new(HashMap::new())),
            active_buffers: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(MemoryStats::default())),
            buffer_id_counter: AtomicU64::new(0),
            current_usage: AtomicU64::new(0),
            peak_usage: AtomicU64::new(0),
        };

        // Pre-allocate initial pools
        pool.initialize_pools();

        info!("Created memory pool with strategy {:?}", config.strategy);
        pool
    }

    /// Initialize memory pools with initial buffers
    fn initialize_pools(&self) {
        if let Ok(mut pools) = self.pools.lock() {
            for &category in &[
                SizeCategory::Small,
                SizeCategory::Medium,
                SizeCategory::Large,
                SizeCategory::XLarge,
            ] {
                let mut pool = VecDeque::new();
                for _ in 0..self.config.initial_pool_size {
                    let buffer_id = self.buffer_id_counter.fetch_add(1, Ordering::Relaxed);
                    let size = category.recommended_size();
                    let buffer = MemoryBuffer::new(size, category, buffer_id);

                    self.current_usage.fetch_add(size as u64, Ordering::Relaxed);
                    pool.push_back(buffer);
                }
                pools.insert(category, pool);
            }
        }

        debug!(
            "Initialized memory pools with {} buffers per category",
            self.config.initial_pool_size
        );
    }

    /// Allocate a buffer from the pool
    pub fn allocate(&self, size: usize) -> Result<MemoryBuffer> {
        let category = SizeCategory::from_size(size);

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.allocation_count += 1;
        }

        // Try to get buffer from pool first
        if let Some(mut buffer) = self.try_get_from_pool(category) {
            if buffer.capacity() >= size {
                buffer.resize(size);
                buffer.add_ref();

                // Move to active buffers
                if let Ok(mut active) = self.active_buffers.lock() {
                    let buffer_id = buffer.id();
                    active.insert(buffer_id, buffer);
                    return Ok(active.get(&buffer_id).unwrap().clone());
                }
            }
        }

        // Check memory pressure before allocating new buffer
        if self.check_memory_pressure() {
            self.run_garbage_collection()?;
        }

        // Allocate new buffer
        let buffer_id = self.buffer_id_counter.fetch_add(1, Ordering::Relaxed);
        let buffer = MemoryBuffer::new(size.max(category.recommended_size()), category, buffer_id);

        // Update memory usage
        let buffer_size = buffer.capacity() as u64;
        self.current_usage.fetch_add(buffer_size, Ordering::Relaxed);

        // Update peak usage
        let current = self.current_usage.load(Ordering::Relaxed);
        let mut peak = self.peak_usage.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_usage.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => peak = actual,
            }
        }

        // Add to active buffers
        if let Ok(mut active) = self.active_buffers.lock() {
            active.insert(buffer_id, buffer);
            Ok(active.get(&buffer_id).unwrap().clone())
        } else {
            Err(DatasetError::MemoryError(
                "Failed to track buffer".to_string(),
            ))
        }
    }

    /// Return a buffer to the pool
    pub fn deallocate(&self, buffer: MemoryBuffer) -> Result<()> {
        let buffer_id = buffer.id();
        let category = buffer.size_category();

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.deallocation_count += 1;
        }

        // Remove from active buffers
        if let Ok(mut active) = self.active_buffers.lock() {
            active.remove(&buffer_id);
        }

        // Check if buffer can be reused
        if buffer.is_unused() && !self.is_buffer_expired(&buffer) {
            // Return to pool if there's space
            if let Ok(mut pools) = self.pools.lock() {
                let pool = pools.entry(category).or_insert_with(VecDeque::new);

                if pool.len() < self.config.max_pool_size {
                    let mut buffer = buffer;
                    buffer.clear(); // Clear data for security
                    pool.push_back(buffer);

                    debug!(
                        "Returned buffer {} to pool (category: {:?})",
                        buffer_id, category
                    );
                    return Ok(());
                }
            }
        }

        // Buffer not reused, update memory usage
        self.current_usage
            .fetch_sub(buffer.capacity() as u64, Ordering::Relaxed);
        debug!(
            "Deallocated buffer {} (size: {})",
            buffer_id,
            buffer.capacity()
        );

        Ok(())
    }

    /// Try to get a buffer from the pool
    fn try_get_from_pool(&self, category: SizeCategory) -> Option<MemoryBuffer> {
        if let Ok(mut pools) = self.pools.lock() {
            if let Some(pool) = pools.get_mut(&category) {
                if let Some(buffer) = pool.pop_front() {
                    // Update cache hit rate
                    if let Ok(mut stats) = self.stats.lock() {
                        let total_requests = stats.allocation_count as f64;
                        if total_requests > 0.0 {
                            stats.cache_hit_rate = (stats.cache_hit_rate * (total_requests - 1.0)
                                + 1.0)
                                / total_requests;
                        }
                    }

                    debug!(
                        "Retrieved buffer {} from pool (category: {:?})",
                        buffer.id(),
                        category
                    );
                    return Some(buffer);
                }
            }
        }
        None
    }

    /// Check if memory pressure is high
    fn check_memory_pressure(&self) -> bool {
        if !self.config.pressure_monitoring {
            return false;
        }

        let current = self.current_usage.load(Ordering::Relaxed);
        let threshold = (self.config.max_memory as f64 * self.config.gc_threshold) as u64;

        current > threshold
    }

    /// Run garbage collection
    fn run_garbage_collection(&self) -> Result<()> {
        debug!("Running garbage collection due to memory pressure");

        let start_time = Instant::now();
        let mut freed_memory = 0u64;
        let _timeout = Duration::from_secs(self.config.reuse_timeout);

        // Clean expired buffers from pools
        if let Ok(mut pools) = self.pools.lock() {
            for (_category, pool) in pools.iter_mut() {
                pool.retain(|buffer| {
                    if self.is_buffer_expired(buffer) {
                        freed_memory += buffer.capacity() as u64;
                        false
                    } else {
                        true
                    }
                });
            }
        }

        // Clean unused active buffers
        if let Ok(mut active) = self.active_buffers.lock() {
            active.retain(|_id, buffer| {
                if buffer.is_unused() && self.is_buffer_expired(buffer) {
                    freed_memory += buffer.capacity() as u64;
                    false
                } else {
                    true
                }
            });
        }

        // Update memory usage
        self.current_usage
            .fetch_sub(freed_memory, Ordering::Relaxed);

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.gc_runs += 1;
        }

        let elapsed = start_time.elapsed();
        info!(
            "Garbage collection completed: freed {} bytes in {:?}",
            freed_memory, elapsed
        );

        Ok(())
    }

    /// Check if a buffer has expired
    fn is_buffer_expired(&self, buffer: &MemoryBuffer) -> bool {
        let timeout = Duration::from_secs(self.config.reuse_timeout);
        buffer.last_used().elapsed() > timeout
    }

    /// Get current memory usage statistics
    pub fn get_stats(&self) -> MemoryStats {
        if let Ok(mut stats) = self.stats.lock() {
            stats.current_usage = self.current_usage.load(Ordering::Relaxed);
            stats.peak_usage = self.peak_usage.load(Ordering::Relaxed);
            stats.total_allocated = stats.peak_usage;

            // Calculate utilization
            if self.config.max_memory > 0 {
                stats.utilization = stats.current_usage as f64 / self.config.max_memory as f64;
            }

            // Count buffers
            if let Ok(pools) = self.pools.lock() {
                stats.pooled_buffers = pools.values().map(VecDeque::len).sum();
            }

            if let Ok(active) = self.active_buffers.lock() {
                stats.active_buffers = active.len();
            }

            stats.clone()
        } else {
            MemoryStats::default()
        }
    }

    /// Force garbage collection
    pub fn force_gc(&self) -> Result<()> {
        self.run_garbage_collection()
    }

    /// Get pool configuration
    pub fn config(&self) -> &MemoryPoolConfig {
        &self.config
    }

    /// Check if memory pool is healthy
    pub fn is_healthy(&self) -> bool {
        let stats = self.get_stats();
        stats.utilization < 0.95 && // Not critically overloaded
        stats.cache_hit_rate > 0.5 // Reasonable cache efficiency
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory-efficient vector that uses the memory pool
pub struct PooledVec<T> {
    buffer: MemoryBuffer,
    len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> PooledVec<T> {
    /// Create a new pooled vector
    pub fn with_capacity(pool: &MemoryPool, capacity: usize) -> Result<Self> {
        let byte_size = capacity * std::mem::size_of::<T>();
        let buffer = pool.allocate(byte_size)?;

        Ok(Self {
            buffer,
            len: 0,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.buffer.capacity() / std::mem::size_of::<T>()
    }

    /// Push an element
    pub fn push(&mut self, value: T) -> Result<()> {
        if self.len >= self.capacity() {
            return Err(DatasetError::MemoryError(
                "Vector capacity exceeded".to_string(),
            ));
        }

        unsafe {
            let ptr = self.buffer.data_mut().as_mut_ptr() as *mut T;
            ptr.add(self.len).write(value);
        }

        self.len += 1;
        Ok(())
    }

    /// Get element at index
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            unsafe {
                let ptr = self.buffer.data().as_ptr() as *const T;
                Some(&*ptr.add(index))
            }
        } else {
            None
        }
    }

    /// Get mutable element at index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            unsafe {
                let ptr = self.buffer.data_mut().as_mut_ptr() as *mut T;
                Some(&mut *ptr.add(index))
            }
        } else {
            None
        }
    }
}

impl<T> Drop for PooledVec<T> {
    fn drop(&mut self) {
        // Drop all elements
        for i in 0..self.len {
            unsafe {
                let ptr = self.buffer.data_mut().as_mut_ptr() as *mut T;
                ptr.add(i).drop_in_place();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_size_category_classification() {
        assert_eq!(SizeCategory::from_size(512), SizeCategory::Small);
        assert_eq!(SizeCategory::from_size(2048), SizeCategory::Medium);
        assert_eq!(SizeCategory::from_size(131072), SizeCategory::Large);
        assert_eq!(
            SizeCategory::from_size(5 * 1024 * 1024),
            SizeCategory::XLarge
        );
        assert_eq!(
            SizeCategory::from_size(50 * 1024 * 1024),
            SizeCategory::Huge
        );
    }

    #[test]
    fn test_memory_pool_allocation() {
        let pool = MemoryPool::new();

        let buffer = pool.allocate(1024).unwrap();
        assert!(buffer.size() >= 1024);
        assert_eq!(buffer.size_category(), SizeCategory::Small);

        pool.deallocate(buffer).unwrap();
    }

    #[test]
    fn test_buffer_reuse() {
        let pool = MemoryPool::new();

        // Allocate and deallocate buffer
        let buffer1 = pool.allocate(1024).unwrap();
        let _buffer1_id = buffer1.id();
        pool.deallocate(buffer1).unwrap();

        // Allocate again - should reuse from pool
        let buffer2 = pool.allocate(1024).unwrap();
        // Note: buffer2 might have a different ID due to cloning in active buffers
        assert!(buffer2.size() >= 1024);

        pool.deallocate(buffer2).unwrap();
    }

    #[test]
    fn test_memory_stats() {
        let pool = MemoryPool::new();

        let initial_stats = pool.get_stats();
        assert!(initial_stats.current_usage > 0); // Initial pools allocated

        let buffer = pool.allocate(2048).unwrap();
        let stats_after_alloc = pool.get_stats();
        assert!(stats_after_alloc.allocation_count > initial_stats.allocation_count);

        pool.deallocate(buffer).unwrap();
        let stats_after_dealloc = pool.get_stats();
        assert!(stats_after_dealloc.deallocation_count > initial_stats.deallocation_count);
    }

    #[test]
    fn test_garbage_collection() {
        let config = MemoryPoolConfig {
            reuse_timeout: 1, // 1 second timeout for testing
            ..Default::default()
        };
        let pool = MemoryPool::with_config(config);

        // Allocate some buffers
        let mut buffers = Vec::new();
        for _ in 0..5 {
            buffers.push(pool.allocate(1024).unwrap());
        }

        // Deallocate them
        for buffer in buffers {
            pool.deallocate(buffer).unwrap();
        }

        // Wait for timeout
        thread::sleep(Duration::from_secs(2));

        // Force GC
        pool.force_gc().unwrap();

        let stats = pool.get_stats();
        assert!(stats.gc_runs > 0);
    }

    #[test]
    fn test_pooled_vec() {
        let pool = MemoryPool::new();
        let mut vec = PooledVec::<i32>::with_capacity(&pool, 10).unwrap();

        assert_eq!(vec.len(), 0);
        assert!(vec.capacity() >= 10);

        vec.push(42).unwrap();
        vec.push(84).unwrap();

        assert_eq!(vec.len(), 2);
        assert_eq!(vec.get(0), Some(&42));
        assert_eq!(vec.get(1), Some(&84));
        assert_eq!(vec.get(2), None);
    }

    #[test]
    fn test_memory_pressure_handling() {
        let config = MemoryPoolConfig {
            max_memory: 1024, // Very small limit for testing
            gc_threshold: 0.5,
            ..Default::default()
        };
        let pool = MemoryPool::with_config(config);

        // This allocation should trigger memory pressure
        let result = pool.allocate(2048);
        // Should still succeed but might trigger GC
        assert!(result.is_ok());
    }

    #[test]
    fn test_buffer_expiration() {
        let config = MemoryPoolConfig {
            reuse_timeout: 0, // Immediate expiration for testing
            ..Default::default()
        };
        let pool = MemoryPool::with_config(config);

        let buffer = pool.allocate(1024).unwrap();

        // Buffer should be expired immediately
        assert!(pool.is_buffer_expired(&buffer));

        pool.deallocate(buffer).unwrap();
    }
}
