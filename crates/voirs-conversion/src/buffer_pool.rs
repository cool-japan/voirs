//! Buffer pooling for efficient memory reuse in audio processing
//!
//! This module provides thread-safe buffer pools to reduce allocations
//! in hot paths like FFT operations and audio transformations.

use parking_lot::Mutex;
use scirs2_core::Complex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Pool statistics for monitoring performance
#[derive(Debug, Clone, Default)]
pub struct PoolStatistics {
    /// Number of buffer acquisitions from pool
    pub hits: u64,
    /// Number of new buffer allocations
    pub misses: u64,
    /// Total bytes allocated
    pub bytes_allocated: u64,
    /// Total bytes reused from pool
    pub bytes_reused: u64,
}

impl PoolStatistics {
    /// Calculate hit rate (0.0-1.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Calculate total memory efficiency
    pub fn memory_efficiency(&self) -> f64 {
        let total = self.bytes_allocated + self.bytes_reused;
        if total == 0 {
            0.0
        } else {
            self.bytes_reused as f64 / total as f64
        }
    }
}

/// Thread-local buffer pool for f32 audio samples
pub struct AudioBufferPool {
    /// Pool of reusable buffers
    buffers: Arc<Mutex<Vec<Vec<f32>>>>,
    /// Maximum number of buffers to pool
    max_pooled: usize,
    /// Default capacity for new buffers
    default_capacity: usize,
    /// Pool statistics (atomic counters)
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
    bytes_allocated: Arc<AtomicU64>,
    bytes_reused: Arc<AtomicU64>,
}

impl AudioBufferPool {
    /// Create a new buffer pool
    ///
    /// # Arguments
    /// * `max_pooled` - Maximum number of buffers to keep in pool
    /// * `default_capacity` - Default capacity for newly allocated buffers
    pub fn new(max_pooled: usize, default_capacity: usize) -> Self {
        Self {
            buffers: Arc::new(Mutex::new(Vec::with_capacity(max_pooled))),
            max_pooled,
            default_capacity,
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
            bytes_allocated: Arc::new(AtomicU64::new(0)),
            bytes_reused: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get a buffer from the pool or allocate a new one
    ///
    /// # Arguments
    /// * `min_capacity` - Minimum required capacity
    ///
    /// # Returns
    /// A buffer with at least the requested capacity
    pub fn acquire(&self, min_capacity: usize) -> PooledBuffer {
        let mut buffers = self.buffers.lock();

        // Try to find a buffer with sufficient capacity
        if let Some(pos) = buffers
            .iter()
            .position(|buf| buf.capacity() >= min_capacity)
        {
            let mut buffer = buffers.swap_remove(pos);
            let capacity = buffer.capacity();
            buffer.clear();

            // Update statistics
            self.hits.fetch_add(1, Ordering::Relaxed);
            self.bytes_reused.fetch_add(
                (capacity * std::mem::size_of::<f32>()) as u64,
                Ordering::Relaxed,
            );

            return PooledBuffer {
                buffer: Some(buffer),
                pool: Arc::clone(&self.buffers),
                max_pooled: self.max_pooled,
            };
        }

        // Allocate new buffer if none available
        drop(buffers);
        let capacity = min_capacity.max(self.default_capacity);

        // Update statistics
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.bytes_allocated.fetch_add(
            (capacity * std::mem::size_of::<f32>()) as u64,
            Ordering::Relaxed,
        );

        PooledBuffer {
            buffer: Some(Vec::with_capacity(capacity)),
            pool: Arc::clone(&self.buffers),
            max_pooled: self.max_pooled,
        }
    }

    /// Get the number of buffers currently in the pool
    pub fn pool_size(&self) -> usize {
        self.buffers.lock().len()
    }

    /// Clear all buffers from the pool
    pub fn clear(&self) {
        self.buffers.lock().clear();
    }

    /// Get pool statistics
    pub fn statistics(&self) -> PoolStatistics {
        PoolStatistics {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            bytes_allocated: self.bytes_allocated.load(Ordering::Relaxed),
            bytes_reused: self.bytes_reused.load(Ordering::Relaxed),
        }
    }

    /// Reset pool statistics
    pub fn reset_statistics(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.bytes_allocated.store(0, Ordering::Relaxed);
        self.bytes_reused.store(0, Ordering::Relaxed);
    }
}

impl Default for AudioBufferPool {
    fn default() -> Self {
        Self::new(16, 4096)
    }
}

/// A buffer borrowed from the pool that will be returned on drop
pub struct PooledBuffer {
    buffer: Option<Vec<f32>>,
    pool: Arc<Mutex<Vec<Vec<f32>>>>,
    max_pooled: usize,
}

impl PooledBuffer {
    /// Get a mutable reference to the buffer (internal use)
    fn get_mut(&mut self) -> &mut Vec<f32> {
        self.buffer.as_mut().expect("Buffer already consumed")
    }

    /// Get a reference to the buffer (internal use)
    fn get_ref(&self) -> &Vec<f32> {
        self.buffer.as_ref().expect("Buffer already consumed")
    }

    /// Consume the pooled buffer and take ownership of the inner Vec
    pub fn into_inner(mut self) -> Vec<f32> {
        self.buffer.take().expect("Buffer already consumed")
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(mut buffer) = self.buffer.take() {
            let mut pool = self.pool.lock();
            if pool.len() < self.max_pooled {
                buffer.clear();
                pool.push(buffer);
            }
        }
    }
}

impl std::ops::Deref for PooledBuffer {
    type Target = Vec<f32>;

    fn deref(&self) -> &Self::Target {
        self.get_ref()
    }
}

impl std::ops::DerefMut for PooledBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

/// Thread-local buffer pool for Complex numbers (FFT operations)
pub struct ComplexBufferPool {
    /// Pool of reusable Complex buffers
    buffers: Arc<Mutex<Vec<Vec<Complex<f32>>>>>,
    /// Maximum number of buffers to pool
    max_pooled: usize,
    /// Default capacity for new buffers
    default_capacity: usize,
    /// Pool statistics
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
    bytes_allocated: Arc<AtomicU64>,
    bytes_reused: Arc<AtomicU64>,
}

impl ComplexBufferPool {
    /// Create a new Complex buffer pool
    pub fn new(max_pooled: usize, default_capacity: usize) -> Self {
        Self {
            buffers: Arc::new(Mutex::new(Vec::with_capacity(max_pooled))),
            max_pooled,
            default_capacity,
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
            bytes_allocated: Arc::new(AtomicU64::new(0)),
            bytes_reused: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Acquire a Complex buffer from the pool
    pub fn acquire(&self, min_capacity: usize) -> PooledComplexBuffer {
        let mut buffers = self.buffers.lock();

        // Try to find a buffer with sufficient capacity
        if let Some(pos) = buffers
            .iter()
            .position(|buf| buf.capacity() >= min_capacity)
        {
            let mut buffer = buffers.swap_remove(pos);
            let capacity = buffer.capacity();
            buffer.clear();

            // Update statistics
            self.hits.fetch_add(1, Ordering::Relaxed);
            self.bytes_reused.fetch_add(
                (capacity * std::mem::size_of::<Complex<f32>>()) as u64,
                Ordering::Relaxed,
            );

            return PooledComplexBuffer {
                buffer: Some(buffer),
                pool: Arc::clone(&self.buffers),
                max_pooled: self.max_pooled,
            };
        }

        // Allocate new buffer if none available
        drop(buffers);
        let capacity = min_capacity.max(self.default_capacity);

        // Update statistics
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.bytes_allocated.fetch_add(
            (capacity * std::mem::size_of::<Complex<f32>>()) as u64,
            Ordering::Relaxed,
        );

        PooledComplexBuffer {
            buffer: Some(Vec::with_capacity(capacity)),
            pool: Arc::clone(&self.buffers),
            max_pooled: self.max_pooled,
        }
    }

    /// Get pool size
    pub fn pool_size(&self) -> usize {
        self.buffers.lock().len()
    }

    /// Clear pool
    pub fn clear(&self) {
        self.buffers.lock().clear();
    }

    /// Get statistics
    pub fn statistics(&self) -> PoolStatistics {
        PoolStatistics {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            bytes_allocated: self.bytes_allocated.load(Ordering::Relaxed),
            bytes_reused: self.bytes_reused.load(Ordering::Relaxed),
        }
    }

    /// Reset statistics
    pub fn reset_statistics(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.bytes_allocated.store(0, Ordering::Relaxed);
        self.bytes_reused.store(0, Ordering::Relaxed);
    }
}

impl Default for ComplexBufferPool {
    fn default() -> Self {
        Self::new(16, 4096)
    }
}

/// A Complex buffer borrowed from the pool
pub struct PooledComplexBuffer {
    buffer: Option<Vec<Complex<f32>>>,
    pool: Arc<Mutex<Vec<Vec<Complex<f32>>>>>,
    max_pooled: usize,
}

impl PooledComplexBuffer {
    fn get_mut(&mut self) -> &mut Vec<Complex<f32>> {
        self.buffer.as_mut().expect("Buffer already consumed")
    }

    fn get_ref(&self) -> &Vec<Complex<f32>> {
        self.buffer.as_ref().expect("Buffer already consumed")
    }

    /// Consume and take ownership of the buffer
    pub fn into_inner(mut self) -> Vec<Complex<f32>> {
        self.buffer.take().expect("Buffer already consumed")
    }
}

impl Drop for PooledComplexBuffer {
    fn drop(&mut self) {
        if let Some(mut buffer) = self.buffer.take() {
            let mut pool = self.pool.lock();
            if pool.len() < self.max_pooled {
                buffer.clear();
                pool.push(buffer);
            }
        }
    }
}

impl std::ops::Deref for PooledComplexBuffer {
    type Target = Vec<Complex<f32>>;

    fn deref(&self) -> &Self::Target {
        self.get_ref()
    }
}

impl std::ops::DerefMut for PooledComplexBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

// Thread-local buffer pool instances
thread_local! {
    static BUFFER_POOL: AudioBufferPool = AudioBufferPool::default();
    static COMPLEX_BUFFER_POOL: ComplexBufferPool = ComplexBufferPool::default();
}

/// Get a buffer from the thread-local pool
///
/// # Arguments
/// * `min_capacity` - Minimum required capacity
///
/// # Returns
/// A pooled buffer that will be returned to the pool on drop
///
/// # Example
/// ```
/// use voirs_conversion::buffer_pool::get_pooled_buffer;
///
/// let mut buffer = get_pooled_buffer(1024);
/// buffer.extend_from_slice(&[0.0f32; 1024]);
/// // Buffer is automatically returned to pool when dropped
/// ```
pub fn get_pooled_buffer(min_capacity: usize) -> PooledBuffer {
    BUFFER_POOL.with(|pool| pool.acquire(min_capacity))
}

/// Get a Complex buffer from the thread-local pool
///
/// # Arguments
/// * `min_capacity` - Minimum required capacity
///
/// # Returns
/// A pooled Complex buffer for FFT operations
///
/// # Example
/// ```
/// use voirs_conversion::buffer_pool::get_pooled_complex_buffer;
/// use scirs2_core::Complex;
///
/// let mut buffer = get_pooled_complex_buffer(1024);
/// buffer.push(Complex::new(1.0, 0.0));
/// // Buffer is automatically returned to pool when dropped
/// ```
pub fn get_pooled_complex_buffer(min_capacity: usize) -> PooledComplexBuffer {
    COMPLEX_BUFFER_POOL.with(|pool| pool.acquire(min_capacity))
}

/// Get the size of the thread-local buffer pool
pub fn pool_size() -> usize {
    BUFFER_POOL.with(|pool| pool.pool_size())
}

/// Get the size of the thread-local Complex buffer pool
pub fn complex_pool_size() -> usize {
    COMPLEX_BUFFER_POOL.with(|pool| pool.pool_size())
}

/// Clear the thread-local buffer pool
pub fn clear_pool() {
    BUFFER_POOL.with(|pool| pool.clear());
}

/// Clear the thread-local Complex buffer pool
pub fn clear_complex_pool() {
    COMPLEX_BUFFER_POOL.with(|pool| pool.clear());
}

/// Get statistics for the thread-local buffer pool
pub fn pool_statistics() -> PoolStatistics {
    BUFFER_POOL.with(|pool| pool.statistics())
}

/// Get statistics for the thread-local Complex buffer pool
pub fn complex_pool_statistics() -> PoolStatistics {
    COMPLEX_BUFFER_POOL.with(|pool| pool.statistics())
}

/// Reset statistics for the thread-local buffer pool
pub fn reset_pool_statistics() {
    BUFFER_POOL.with(|pool| pool.reset_statistics());
}

/// Reset statistics for the thread-local Complex buffer pool
pub fn reset_complex_pool_statistics() {
    COMPLEX_BUFFER_POOL.with(|pool| pool.reset_statistics());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_pool_acquire_release() {
        let pool = AudioBufferPool::new(4, 1024);

        // Acquire buffer
        let buffer = pool.acquire(512);
        assert!(buffer.capacity() >= 512);
        assert_eq!(pool.pool_size(), 0);

        // Release buffer by dropping
        drop(buffer);
        assert_eq!(pool.pool_size(), 1);

        // Reuse buffer
        let buffer = pool.acquire(512);
        assert_eq!(pool.pool_size(), 0);
    }

    #[test]
    fn test_buffer_pool_capacity_growth() {
        let pool = AudioBufferPool::new(4, 1024);

        // Request larger buffer than default
        let buffer = pool.acquire(2048);
        assert!(buffer.capacity() >= 2048);
    }

    #[test]
    fn test_buffer_pool_max_pooled() {
        let pool = AudioBufferPool::new(2, 1024);

        // Fill pool to maximum
        let b1 = pool.acquire(512);
        let b2 = pool.acquire(512);
        let b3 = pool.acquire(512);

        drop(b1);
        drop(b2);
        assert_eq!(pool.pool_size(), 2);

        // Third buffer should not be pooled (exceeds max)
        drop(b3);
        assert_eq!(pool.pool_size(), 2);
    }

    #[test]
    fn test_pooled_buffer_deref() {
        let pool = AudioBufferPool::new(4, 1024);
        let mut buffer = pool.acquire(512);

        // Test mutable deref
        buffer.extend_from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer[0], 1.0);
    }

    #[test]
    fn test_pooled_buffer_into_inner() {
        let pool = AudioBufferPool::new(4, 1024);
        let mut buffer = pool.acquire(512);
        buffer.extend_from_slice(&[1.0, 2.0, 3.0]);

        let vec = buffer.into_inner();
        assert_eq!(vec.len(), 3);
        assert_eq!(vec[0], 1.0);

        // Buffer was consumed, not returned to pool
        assert_eq!(pool.pool_size(), 0);
    }

    #[test]
    fn test_thread_local_pool() {
        // Clear any existing buffers
        clear_pool();
        assert_eq!(pool_size(), 0);

        // Acquire and release
        let buffer = get_pooled_buffer(1024);
        drop(buffer);
        assert_eq!(pool_size(), 1);

        // Reuse
        let buffer = get_pooled_buffer(512);
        assert_eq!(pool_size(), 0);
        drop(buffer);
        assert_eq!(pool_size(), 1);
    }

    #[test]
    fn test_clear_pool() {
        clear_pool();
        let b1 = get_pooled_buffer(512);
        let b2 = get_pooled_buffer(512);
        drop(b1);
        drop(b2);
        assert_eq!(pool_size(), 2);

        clear_pool();
        assert_eq!(pool_size(), 0);
    }

    #[test]
    fn test_pool_statistics() {
        let pool = AudioBufferPool::new(4, 1024);
        pool.reset_statistics();

        // First acquisition - should be a miss
        let b1 = pool.acquire(512);
        drop(b1);

        let stats = pool.statistics();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);
        assert!(stats.bytes_allocated > 0);

        // Second acquisition - should be a hit
        let b2 = pool.acquire(512);
        drop(b2);

        let stats = pool.statistics();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);
        assert!(stats.bytes_reused > 0);

        // Check hit rate
        assert!((stats.hit_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_pool_statistics_memory_efficiency() {
        let pool = AudioBufferPool::new(4, 1024);
        pool.reset_statistics();

        // Allocate and reuse multiple times
        for _ in 0..10 {
            let b = pool.acquire(512);
            drop(b);
        }

        let stats = pool.statistics();
        // First allocation is a miss, rest are hits
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 9);

        // Memory efficiency should be high (90%)
        assert!(stats.memory_efficiency() > 0.85);
    }

    #[test]
    fn test_complex_buffer_pool() {
        let pool = ComplexBufferPool::new(4, 1024);

        // Acquire complex buffer
        let mut buffer = pool.acquire(512);
        assert!(buffer.capacity() >= 512);

        // Use the buffer
        buffer.push(Complex::new(1.0, 2.0));
        buffer.push(Complex::new(3.0, 4.0));
        assert_eq!(buffer.len(), 2);

        // Release and verify pooling
        drop(buffer);
        assert_eq!(pool.pool_size(), 1);

        // Reuse buffer
        let buffer = pool.acquire(512);
        assert_eq!(buffer.len(), 0); // Should be cleared
        assert_eq!(pool.pool_size(), 0);
    }

    #[test]
    fn test_complex_buffer_statistics() {
        let pool = ComplexBufferPool::new(4, 1024);
        pool.reset_statistics();

        // First acquisition
        let b1 = pool.acquire(512);
        drop(b1);

        let stats = pool.statistics();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);

        // Verify bytes allocated for Complex<f32>
        let expected_bytes = 1024 * std::mem::size_of::<Complex<f32>>();
        assert_eq!(stats.bytes_allocated as usize, expected_bytes);

        // Second acquisition - hit
        let b2 = pool.acquire(512);
        drop(b2);

        let stats = pool.statistics();
        assert_eq!(stats.hits, 1);
        assert!(stats.bytes_reused > 0);
    }

    #[test]
    fn test_thread_local_complex_pool() {
        clear_complex_pool();
        reset_complex_pool_statistics();

        // Acquire and release
        let mut buffer = get_pooled_complex_buffer(1024);
        buffer.push(Complex::new(1.0, 0.0));
        drop(buffer);

        assert_eq!(complex_pool_size(), 1);

        // Check statistics
        let stats = complex_pool_statistics();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);

        // Reuse
        let buffer = get_pooled_complex_buffer(512);
        drop(buffer);

        let stats = complex_pool_statistics();
        assert_eq!(stats.hits, 1);
        assert!(stats.hit_rate() > 0.4);
    }

    #[test]
    fn test_pooled_complex_buffer_deref() {
        let pool = ComplexBufferPool::new(4, 1024);
        let mut buffer = pool.acquire(512);

        // Test mutable deref
        buffer.extend_from_slice(&[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);

        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer[0], Complex::new(1.0, 2.0));
        assert_eq!(buffer[1], Complex::new(3.0, 4.0));
    }

    #[test]
    fn test_pooled_complex_buffer_into_inner() {
        let pool = ComplexBufferPool::new(4, 1024);
        let mut buffer = pool.acquire(512);

        buffer.push(Complex::new(1.0, 2.0));
        let vec = buffer.into_inner();

        assert_eq!(vec.len(), 1);
        assert_eq!(vec[0], Complex::new(1.0, 2.0));

        // Buffer was consumed, not returned to pool
        assert_eq!(pool.pool_size(), 0);
    }

    #[test]
    fn test_statistics_reset() {
        let pool = AudioBufferPool::new(4, 1024);

        // Generate some activity
        for _ in 0..5 {
            let b = pool.acquire(512);
            drop(b);
        }

        let stats = pool.statistics();
        assert!(stats.hits > 0 || stats.misses > 0);

        // Reset statistics
        pool.reset_statistics();
        let stats = pool.statistics();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.bytes_allocated, 0);
        assert_eq!(stats.bytes_reused, 0);
    }

    #[test]
    fn test_thread_local_statistics() {
        reset_pool_statistics();
        reset_complex_pool_statistics();

        // Regular buffer stats
        let b1 = get_pooled_buffer(1024);
        drop(b1);

        let stats = pool_statistics();
        assert_eq!(stats.misses, 1);

        // Complex buffer stats
        let b2 = get_pooled_complex_buffer(1024);
        drop(b2);

        let complex_stats = complex_pool_statistics();
        assert_eq!(complex_stats.misses, 1);

        // Verify they're independent
        assert_eq!(stats.misses, 1);
        assert_eq!(complex_stats.misses, 1);
    }
}
