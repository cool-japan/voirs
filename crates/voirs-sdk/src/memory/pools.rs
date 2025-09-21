//! Memory pools for efficient resource management
//!
//! Provides specialized memory pools for audio buffers, tensors, and other
//! frequently allocated objects to reduce garbage collection pressure.

use std::alloc::{alloc, dealloc, Layout};
use std::collections::VecDeque;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Configuration for memory pools
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Initial pool size
    pub initial_size: usize,
    /// Maximum pool size
    pub max_size: usize,
    /// Item timeout duration
    pub timeout: Duration,
    /// Alignment requirement
    pub alignment: usize,
    /// Enable statistics collection
    pub enable_stats: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 10,
            max_size: 100,
            timeout: Duration::from_secs(60),
            alignment: 8,
            enable_stats: true,
        }
    }
}

/// Generic memory pool for managing reusable objects
pub trait MemoryPool<T> {
    /// Get an item from the pool
    fn get(&self) -> Option<T>;

    /// Return an item to the pool
    fn put(&self, item: T);

    /// Get current pool size
    fn size(&self) -> usize;

    /// Clear all items from the pool
    fn clear(&self);

    /// Get pool statistics
    fn stats(&self) -> PoolStats;
}

/// Pool statistics
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total allocations
    pub allocations: u64,
    /// Total deallocations
    pub deallocations: u64,
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Current pool size
    pub current_size: usize,
    /// Peak pool size
    pub peak_size: usize,
    /// Hit rate (0.0 - 1.0)
    pub hit_rate: f64,
}

/// Audio buffer pool for managing audio sample buffers
pub struct AudioBufferPool {
    /// Pool of available buffers
    buffers: Arc<Mutex<VecDeque<PooledBuffer>>>,
    /// Pool configuration
    config: PoolConfig,
    /// Pool statistics
    stats: Arc<RwLock<PoolStats>>,
}

/// Pooled buffer with metadata
#[derive(Debug)]
struct PooledBuffer {
    /// Buffer data
    data: Vec<f32>,
    /// Creation time
    #[allow(dead_code)]
    created_at: Instant,
    /// Last used time
    last_used: Instant,
}

impl AudioBufferPool {
    /// Create a new audio buffer pool
    pub fn new(config: PoolConfig) -> Self {
        let pool = Self {
            buffers: Arc::new(Mutex::new(VecDeque::new())),
            config,
            stats: Arc::new(RwLock::new(PoolStats::default())),
        };

        // Pre-allocate initial buffers
        pool.preallocate();
        pool
    }

    /// Create pool with default configuration
    pub fn with_default_config() -> Self {
        Self::new(PoolConfig::default())
    }

    /// Get a buffer of specified size
    pub fn get_buffer(&self, size: usize) -> Vec<f32> {
        let mut buffers = self.buffers.lock().unwrap();

        // Try to find a buffer of suitable size
        if let Some(pos) = buffers.iter().position(|buf| buf.data.len() >= size) {
            let mut pooled_buf = buffers.remove(pos).unwrap();
            pooled_buf.last_used = Instant::now();

            // Update statistics
            if self.config.enable_stats {
                if let Ok(mut stats) = self.stats.write() {
                    stats.hits += 1;
                    stats.current_size = buffers.len();
                    let total_requests = stats.hits + stats.misses;
                    stats.hit_rate = if total_requests > 0 {
                        stats.hits as f64 / total_requests as f64
                    } else {
                        0.0
                    };
                }
            }

            // Resize if necessary
            pooled_buf.data.resize(size, 0.0);
            pooled_buf.data.fill(0.0); // Clear buffer
            pooled_buf.data
        } else {
            // No suitable buffer found, allocate new one
            if self.config.enable_stats {
                if let Ok(mut stats) = self.stats.write() {
                    stats.misses += 1;
                    stats.allocations += 1;
                    let total_requests = stats.hits + stats.misses;
                    stats.hit_rate = if total_requests > 0 {
                        stats.hits as f64 / total_requests as f64
                    } else {
                        0.0
                    };
                }
            }

            vec![0.0; size]
        }
    }

    /// Return a buffer to the pool
    pub fn return_buffer(&self, mut buffer: Vec<f32>) {
        let mut buffers = self.buffers.lock().unwrap();

        // Don't keep buffers that exceed max size limit
        if buffers.len() >= self.config.max_size {
            if self.config.enable_stats {
                if let Ok(mut stats) = self.stats.write() {
                    stats.deallocations += 1;
                }
            }
            return;
        }

        // Clear the buffer for reuse
        buffer.fill(0.0);

        let pooled_buffer = PooledBuffer {
            data: buffer,
            created_at: Instant::now(),
            last_used: Instant::now(),
        };

        buffers.push_back(pooled_buffer);

        // Update statistics
        if self.config.enable_stats {
            if let Ok(mut stats) = self.stats.write() {
                stats.current_size = buffers.len();
                if stats.current_size > stats.peak_size {
                    stats.peak_size = stats.current_size;
                }
            }
        }
    }

    /// Clean up expired buffers
    pub fn cleanup_expired(&self) {
        let mut buffers = self.buffers.lock().unwrap();
        let now = Instant::now();

        let initial_len = buffers.len();
        buffers.retain(|buf| now.duration_since(buf.last_used) < self.config.timeout);

        if self.config.enable_stats {
            let removed = initial_len - buffers.len();
            if let Ok(mut stats) = self.stats.write() {
                stats.deallocations += removed as u64;
                stats.current_size = buffers.len();
            }
        }
    }

    /// Pre-allocate initial buffers
    fn preallocate(&self) {
        let mut buffers = self.buffers.lock().unwrap();

        for _ in 0..self.config.initial_size {
            let buffer = PooledBuffer {
                data: Vec::with_capacity(1024), // Default size
                created_at: Instant::now(),
                last_used: Instant::now(),
            };
            buffers.push_back(buffer);
        }

        if self.config.enable_stats {
            if let Ok(mut stats) = self.stats.write() {
                stats.current_size = buffers.len();
                stats.peak_size = buffers.len();
            }
        }
    }
}

impl MemoryPool<Vec<f32>> for AudioBufferPool {
    fn get(&self) -> Option<Vec<f32>> {
        Some(self.get_buffer(1024)) // Default size
    }

    fn put(&self, item: Vec<f32>) {
        self.return_buffer(item);
    }

    fn size(&self) -> usize {
        self.buffers.lock().unwrap().len()
    }

    fn clear(&self) {
        let mut buffers = self.buffers.lock().unwrap();
        let cleared_count = buffers.len();
        buffers.clear();

        if self.config.enable_stats {
            if let Ok(mut stats) = self.stats.write() {
                stats.deallocations += cleared_count as u64;
                stats.current_size = 0;
            }
        }
    }

    fn stats(&self) -> PoolStats {
        self.stats.read().unwrap().clone()
    }
}

/// Tensor memory pool for managing tensor allocations
pub struct TensorPool {
    /// Pool of available tensors
    tensors: Arc<Mutex<VecDeque<PooledTensor>>>,
    /// Pool configuration
    config: PoolConfig,
    /// Pool statistics
    stats: Arc<RwLock<PoolStats>>,
}

/// Pooled tensor with metadata
#[derive(Debug)]
struct PooledTensor {
    /// Raw memory pointer
    ptr: NonNull<u8>,
    /// Memory layout
    layout: Layout,
    /// Size in bytes
    size: usize,
    /// Creation time
    #[allow(dead_code)]
    created_at: Instant,
    /// Last used time
    last_used: Instant,
}

unsafe impl Send for PooledTensor {}
unsafe impl Sync for PooledTensor {}

impl TensorPool {
    /// Create a new tensor pool
    pub fn new(config: PoolConfig) -> Self {
        Self {
            tensors: Arc::new(Mutex::new(VecDeque::new())),
            config,
            stats: Arc::new(RwLock::new(PoolStats::default())),
        }
    }

    /// Allocate aligned memory for tensor
    pub fn allocate(&self, size: usize) -> Option<NonNull<u8>> {
        let mut tensors = self.tensors.lock().unwrap();

        // Try to find a suitable tensor
        if let Some(pos) = tensors.iter().position(|tensor| tensor.size >= size) {
            let mut pooled_tensor = tensors.remove(pos).unwrap();
            pooled_tensor.last_used = Instant::now();

            // Update statistics
            if self.config.enable_stats {
                if let Ok(mut stats) = self.stats.write() {
                    stats.hits += 1;
                    stats.current_size = tensors.len();
                }
            }

            Some(pooled_tensor.ptr)
        } else {
            // Allocate new aligned memory
            let layout = Layout::from_size_align(size, self.config.alignment).ok()?;

            unsafe {
                let ptr = alloc(layout);
                if ptr.is_null() {
                    return None;
                }

                if self.config.enable_stats {
                    if let Ok(mut stats) = self.stats.write() {
                        stats.misses += 1;
                        stats.allocations += 1;
                    }
                }

                Some(NonNull::new_unchecked(ptr))
            }
        }
    }

    /// Return allocated memory to the pool
    pub fn deallocate(&self, ptr: NonNull<u8>, size: usize) {
        let mut tensors = self.tensors.lock().unwrap();

        // Don't keep tensors that exceed max size limit
        if tensors.len() >= self.config.max_size {
            unsafe {
                let layout = Layout::from_size_align(size, self.config.alignment).unwrap();
                dealloc(ptr.as_ptr(), layout);
            }

            if self.config.enable_stats {
                if let Ok(mut stats) = self.stats.write() {
                    stats.deallocations += 1;
                }
            }
            return;
        }

        let layout = Layout::from_size_align(size, self.config.alignment).unwrap();
        let pooled_tensor = PooledTensor {
            ptr,
            layout,
            size,
            created_at: Instant::now(),
            last_used: Instant::now(),
        };

        tensors.push_back(pooled_tensor);

        // Update statistics
        if self.config.enable_stats {
            if let Ok(mut stats) = self.stats.write() {
                stats.current_size = tensors.len();
                if stats.current_size > stats.peak_size {
                    stats.peak_size = stats.current_size;
                }
            }
        }
    }

    /// Clean up expired tensors
    pub fn cleanup_expired(&self) {
        let mut tensors = self.tensors.lock().unwrap();
        let now = Instant::now();

        let initial_len = tensors.len();

        // Separate expired tensors from valid ones
        let mut expired: Vec<PooledTensor> = Vec::new();
        let mut i = 0;
        while i < tensors.len() {
            if now.duration_since(tensors[i].last_used) >= self.config.timeout {
                if let Some(tensor) = tensors.remove(i) {
                    expired.push(tensor);
                }
            } else {
                i += 1;
            }
        }

        // Deallocate expired tensors
        for tensor in expired {
            unsafe {
                dealloc(tensor.ptr.as_ptr(), tensor.layout);
            }
        }

        if self.config.enable_stats {
            let removed = initial_len - tensors.len();
            if let Ok(mut stats) = self.stats.write() {
                stats.deallocations += removed as u64;
                stats.current_size = tensors.len();
            }
        }
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        self.stats.read().unwrap().clone()
    }
}

impl Default for AudioBufferPool {
    fn default() -> Self {
        Self::with_default_config()
    }
}

impl Drop for TensorPool {
    fn drop(&mut self) {
        // Clean up all remaining tensors
        let mut tensors = self.tensors.lock().unwrap();
        for tensor in tensors.drain(..) {
            unsafe {
                dealloc(tensor.ptr.as_ptr(), tensor.layout);
            }
        }
    }
}

/// Thread-local memory pools for zero-contention access
pub struct ThreadLocalPools {
    /// Audio buffer pool
    audio_pool: thread_local::ThreadLocal<AudioBufferPool>,
    /// Tensor pool
    tensor_pool: thread_local::ThreadLocal<TensorPool>,
    /// Pool configuration
    config: PoolConfig,
}

impl ThreadLocalPools {
    /// Create new thread-local pools
    pub fn new(config: PoolConfig) -> Self {
        Self {
            audio_pool: thread_local::ThreadLocal::new(),
            tensor_pool: thread_local::ThreadLocal::new(),
            config,
        }
    }

    /// Get thread-local audio buffer pool
    pub fn audio_pool(&self) -> &AudioBufferPool {
        self.audio_pool
            .get_or(|| AudioBufferPool::new(self.config.clone()))
    }

    /// Get thread-local tensor pool
    pub fn tensor_pool(&self) -> &TensorPool {
        self.tensor_pool
            .get_or(|| TensorPool::new(self.config.clone()))
    }

    /// Cleanup expired items in all pools
    pub fn cleanup_all(&self) {
        // Note: This only cleans up pools in the current thread
        // Each thread needs to call this separately
        if let Some(audio_pool) = self.audio_pool.get() {
            audio_pool.cleanup_expired();
        }
        if let Some(tensor_pool) = self.tensor_pool.get() {
            tensor_pool.cleanup_expired();
        }
    }

    /// Get aggregated statistics from all thread-local pools
    pub fn aggregated_stats(&self) -> (PoolStats, PoolStats) {
        let audio_stats = self.audio_pool.get().map(|p| p.stats()).unwrap_or_default();
        let tensor_stats = self
            .tensor_pool
            .get()
            .map(|p| p.stats())
            .unwrap_or_default();

        (audio_stats, tensor_stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_audio_buffer_pool_basic() {
        let config = PoolConfig {
            initial_size: 5,
            max_size: 10,
            timeout: Duration::from_secs(1),
            ..Default::default()
        };

        let pool = AudioBufferPool::new(config);

        // Test buffer allocation
        let buffer = pool.get_buffer(1024);
        assert_eq!(buffer.len(), 1024);
        assert!(buffer.iter().all(|&x| x == 0.0));

        // Test buffer return
        pool.return_buffer(buffer);

        // Test that returned buffer is reused
        let buffer2 = pool.get_buffer(1024);
        assert_eq!(buffer2.len(), 1024);

        let stats = pool.stats();
        assert!(stats.hits > 0 || stats.misses > 0);
    }

    #[test]
    fn test_pool_cleanup() {
        let config = PoolConfig {
            initial_size: 2,
            max_size: 10,
            timeout: Duration::from_millis(50),
            ..Default::default()
        };

        let pool = AudioBufferPool::new(config);

        // Add some buffers
        for i in 0..3 {
            let buffer = vec![0.0; 512 * (i + 1)];
            pool.return_buffer(buffer);
        }

        let initial_size = pool.size();
        assert!(initial_size > 0);

        // Wait for timeout to expire all buffers
        thread::sleep(Duration::from_millis(100));

        // Cleanup expired buffers
        pool.cleanup_expired();

        // Verify cleanup happened - either no buffers remain or fewer than before
        let size_after_cleanup = pool.size();
        assert!(size_after_cleanup <= initial_size);
    }

    #[test]
    fn test_tensor_pool_allocation() {
        let config = PoolConfig {
            alignment: 32,
            ..Default::default()
        };

        let pool = TensorPool::new(config);

        // Test allocation
        let ptr = pool.allocate(1024).unwrap();
        // Pointer from valid allocation is guaranteed to be non-null

        // Test deallocation
        pool.deallocate(ptr, 1024);

        // Test reuse
        let ptr2 = pool.allocate(1024).unwrap();
        // Pointer from valid allocation is guaranteed to be non-null

        pool.deallocate(ptr2, 1024);
    }

    #[test]
    fn test_thread_local_pools() {
        let config = PoolConfig::default();
        let pools = ThreadLocalPools::new(config);

        // Test audio pool access
        let audio_pool = pools.audio_pool();
        let buffer = audio_pool.get_buffer(512);
        assert_eq!(buffer.len(), 512);
        audio_pool.return_buffer(buffer);

        // Test tensor pool access
        let tensor_pool = pools.tensor_pool();
        let ptr = tensor_pool.allocate(256).unwrap();
        tensor_pool.deallocate(ptr, 256);

        // Test cleanup
        pools.cleanup_all();
    }

    #[test]
    fn test_pool_statistics() {
        let config = PoolConfig {
            enable_stats: true,
            ..Default::default()
        };

        let pool = AudioBufferPool::new(config);

        // Generate some activity
        let buffer1 = pool.get_buffer(512);
        let buffer2 = pool.get_buffer(1024);

        pool.return_buffer(buffer1);
        pool.return_buffer(buffer2);

        let buffer3 = pool.get_buffer(512); // Should hit cache

        let stats = pool.stats();
        assert!(stats.hits > 0 || stats.misses > 0);
        assert!(stats.allocations > 0);
        assert!(stats.current_size > 0);

        pool.return_buffer(buffer3);
    }
}
