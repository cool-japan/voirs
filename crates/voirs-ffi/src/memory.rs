//! Advanced memory management for FFI operations.

pub mod allocators;
pub mod debug;
pub mod refcount;
pub mod zero_copy;

pub use allocators::*;
pub use debug::*;
pub use refcount::*;
pub use zero_copy::*;

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::{
    collections::HashMap,
    sync::Arc,
    // Note: unused imports removed: c_uchar, c_uint, c_void, ptr, AtomicU64, Ordering
};

/// Global memory statistics
static MEMORY_STATS: Lazy<Mutex<MemoryStats>> = Lazy::new(|| Mutex::new(MemoryStats::new()));

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub current_allocations: u64,
    pub peak_allocations: u64,
    pub total_bytes_allocated: u64,
    pub current_bytes_allocated: u64,
    pub peak_bytes_allocated: u64,
}

impl MemoryStats {
    fn new() -> Self {
        Self {
            total_allocations: 0,
            total_deallocations: 0,
            current_allocations: 0,
            peak_allocations: 0,
            total_bytes_allocated: 0,
            current_bytes_allocated: 0,
            peak_bytes_allocated: 0,
        }
    }

    fn record_allocation(&mut self, size: u64) {
        self.total_allocations += 1;
        self.current_allocations += 1;
        self.total_bytes_allocated += size;
        self.current_bytes_allocated += size;

        if self.current_allocations > self.peak_allocations {
            self.peak_allocations = self.current_allocations;
        }

        if self.current_bytes_allocated > self.peak_bytes_allocated {
            self.peak_bytes_allocated = self.current_bytes_allocated;
        }
    }

    fn record_deallocation(&mut self, size: u64) {
        self.total_deallocations += 1;
        self.current_allocations = self.current_allocations.saturating_sub(1);
        self.current_bytes_allocated = self.current_bytes_allocated.saturating_sub(size);
    }
}

/// Reference-counted buffer for shared audio data
pub struct RefCountedBuffer {
    inner: Arc<BufferInner>,
}

struct BufferInner {
    data: Vec<f32>,
    sample_rate: u32,
    channels: u32,
    size_bytes: u64,
}

impl RefCountedBuffer {
    pub fn new(data: Vec<f32>, sample_rate: u32, channels: u32) -> Self {
        let size_bytes = data.len() * std::mem::size_of::<f32>();

        {
            let mut stats = MEMORY_STATS.lock();
            stats.record_allocation(size_bytes as u64);
        }

        Self {
            inner: Arc::new(BufferInner {
                data,
                sample_rate,
                channels,
                size_bytes: size_bytes as u64,
            }),
        }
    }

    pub fn data(&self) -> &[f32] {
        &self.inner.data
    }

    pub fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }

    pub fn channels(&self) -> u32 {
        self.inner.channels
    }

    pub fn size_bytes(&self) -> u64 {
        self.inner.size_bytes
    }

    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }
}

impl Clone for RefCountedBuffer {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Drop for BufferInner {
    fn drop(&mut self) {
        let mut stats = MEMORY_STATS.lock();
        stats.record_deallocation(self.size_bytes);
    }
}

/// Memory pool for efficient allocation of commonly-used buffer sizes
pub struct MemoryPool {
    pools: HashMap<usize, Vec<Vec<f32>>>,
    max_pool_size: usize,
}

impl MemoryPool {
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            pools: HashMap::new(),
            max_pool_size,
        }
    }

    pub fn allocate(&mut self, size: usize) -> Vec<f32> {
        if let Some(pool) = self.pools.get_mut(&size) {
            if let Some(mut buffer) = pool.pop() {
                buffer.clear();
                buffer.resize(size, 0.0);
                return buffer;
            }
        }

        // Create new buffer if pool is empty
        vec![0.0; size]
    }

    pub fn deallocate(&mut self, mut buffer: Vec<f32>) {
        let size = buffer.capacity();

        let pool = self.pools.entry(size).or_default();
        if pool.len() < self.max_pool_size {
            buffer.clear();
            pool.push(buffer);
        }
        // Otherwise let the buffer drop naturally
    }

    pub fn clear(&mut self) {
        self.pools.clear();
    }

    pub fn pool_stats(&self) -> HashMap<usize, usize> {
        self.pools
            .iter()
            .map(|(&size, pool)| (size, pool.len()))
            .collect()
    }
}

/// Global memory pool instance
static MEMORY_POOL: Lazy<Mutex<MemoryPool>> = Lazy::new(|| {
    Mutex::new(MemoryPool::new(10)) // Keep max 10 buffers per size
});

/// Allocate buffer from memory pool
pub fn pool_allocate(size: usize) -> Vec<f32> {
    let mut pool = MEMORY_POOL.lock();
    pool.allocate(size)
}

/// Return buffer to memory pool
pub fn pool_deallocate(buffer: Vec<f32>) {
    let mut pool = MEMORY_POOL.lock();
    pool.deallocate(buffer);
}

/// Get current memory statistics
pub fn get_memory_stats() -> MemoryStats {
    MEMORY_STATS.lock().clone()
}

/// Reset memory statistics (for testing)
pub fn reset_memory_stats() {
    let mut stats = MEMORY_STATS.lock();
    *stats = MemoryStats::new();
}

/// Check for memory leaks
pub fn check_memory_leaks() -> bool {
    let stats = MEMORY_STATS.lock();
    stats.current_allocations == 0 && stats.current_bytes_allocated == 0
}

/// Get memory statistics as JSON string
#[no_mangle]
pub extern "C" fn voirs_memory_get_stats() -> *mut std::os::raw::c_char {
    let stats = get_memory_stats();
    let json = format!(
        "{{\"total_allocations\":{},\"total_deallocations\":{},\"current_allocations\":{},\"peak_allocations\":{},\"total_bytes\":{},\"current_bytes\":{},\"peak_bytes\":{}}}",
        stats.total_allocations,
        stats.total_deallocations,
        stats.current_allocations,
        stats.peak_allocations,
        stats.total_bytes_allocated,
        stats.current_bytes_allocated,
        stats.peak_bytes_allocated
    );

    crate::string_to_c_str(&json)
}

/// Check if there are memory leaks
#[no_mangle]
pub extern "C" fn voirs_memory_check_leaks() -> std::os::raw::c_int {
    if check_memory_leaks() {
        1
    } else {
        0
    }
}

/// Reset memory statistics
#[no_mangle]
pub extern "C" fn voirs_memory_reset_stats() {
    reset_memory_stats();
}

/// Clear memory pools
#[no_mangle]
pub extern "C" fn voirs_memory_clear_pools() {
    let mut pool = MEMORY_POOL.lock();
    pool.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats() {
        reset_memory_stats();

        let initial_stats = get_memory_stats();
        assert_eq!(initial_stats.current_allocations, 0);

        let buffer = RefCountedBuffer::new(vec![1.0, 2.0, 3.0, 4.0], 44100, 1);
        let stats_after_alloc = get_memory_stats();
        assert_eq!(stats_after_alloc.current_allocations, 1);
        assert!(stats_after_alloc.current_bytes_allocated > 0);

        drop(buffer);
        let stats_after_drop = get_memory_stats();
        assert_eq!(stats_after_drop.current_allocations, 0);
        assert_eq!(stats_after_drop.current_bytes_allocated, 0);
    }

    #[test]
    fn test_ref_counted_buffer() {
        reset_memory_stats();

        let buffer1 = RefCountedBuffer::new(vec![1.0, 2.0, 3.0], 44100, 1);
        assert_eq!(buffer1.ref_count(), 1);

        let buffer2 = buffer1.clone();
        assert_eq!(buffer1.ref_count(), 2);
        assert_eq!(buffer2.ref_count(), 2);

        assert_eq!(buffer1.data(), &[1.0, 2.0, 3.0]);
        assert_eq!(buffer2.data(), &[1.0, 2.0, 3.0]);

        drop(buffer2);
        assert_eq!(buffer1.ref_count(), 1);
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new(5);

        // Allocate and deallocate
        let buffer1 = pool.allocate(100);
        assert_eq!(buffer1.len(), 100);

        pool.deallocate(buffer1);

        // Should reuse the buffer
        let buffer2 = pool.allocate(100);
        assert_eq!(buffer2.len(), 100);

        let stats = pool.pool_stats();
        assert_eq!(stats.get(&100), Some(&0)); // Pool should be empty since we allocated from it
    }

    #[test]
    fn test_pool_functions() {
        reset_memory_stats();

        let buffer = pool_allocate(50);
        assert_eq!(buffer.len(), 50);

        pool_deallocate(buffer);
        // Should not crash and should reuse buffer on next allocation
    }

    #[test]
    fn test_memory_leak_detection() {
        reset_memory_stats();
        assert!(check_memory_leaks());

        let _buffer = RefCountedBuffer::new(vec![1.0, 2.0], 44100, 1);
        assert!(!check_memory_leaks());
    }
}
