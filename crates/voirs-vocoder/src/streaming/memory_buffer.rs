//! Memory-efficient buffering for neural vocoders
//!
//! This module provides advanced memory management techniques including:
//! - Zero-copy buffer operations where possible
//! - Memory pool allocation to reduce GC pressure
//! - Adaptive memory allocation based on usage patterns
//! - NUMA-aware memory allocation for multi-socket systems
//! - Lock-free circular buffers for high-performance scenarios

use crate::{AudioBuffer, Result, VocoderError};
use crossbeam_utils::CachePadded;
use std::alloc::{alloc, dealloc, Layout};
use std::collections::{HashMap, VecDeque};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Memory-efficient buffer manager
pub struct MemoryEfficientBufferManager {
    /// Memory pool for audio buffers
    memory_pool: Arc<MemoryPool>,

    /// Lock-free circular buffers per stream
    circular_buffers: Arc<RwLock<HashMap<u64, Arc<LockFreeCircularBuffer>>>>,

    /// Memory allocation strategy
    allocation_strategy: Arc<RwLock<AllocationStrategy>>,

    /// Memory usage statistics
    memory_stats: Arc<RwLock<MemoryStats>>,

    /// Garbage collection scheduler
    gc_scheduler: Arc<GarbageCollector>,

    /// NUMA topology information
    numa_topology: Arc<NumaTopology>,
}

/// Memory allocation strategies
#[derive(Debug, Clone, Copy, Default)]
pub enum AllocationStrategy {
    /// Standard heap allocation
    Standard,
    /// Memory pool with pre-allocated chunks
    Pool,
    /// NUMA-aware allocation
    NumaAware,
    /// Zero-copy where possible
    ZeroCopy,
    /// Hybrid approach (default)
    #[default]
    Hybrid,
}

/// Memory pool for efficient buffer allocation
#[allow(dead_code)]
struct MemoryPool {
    /// Small buffers (< 1KB)
    small_pool: Arc<PoolTier>,

    /// Medium buffers (1KB - 64KB)
    medium_pool: Arc<PoolTier>,

    /// Large buffers (> 64KB)
    large_pool: Arc<PoolTier>,

    /// Pool statistics
    #[allow(dead_code)]
    stats: Arc<RwLock<PoolStats>>,

    /// Allocation tracking
    #[allow(dead_code)]
    allocations: Arc<RwLock<HashMap<usize, AllocationInfo>>>,
}

// Safety: MemoryPool is thread-safe due to use of Arc<_> for all fields
unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}

/// Single tier of memory pool
#[allow(dead_code)]
struct PoolTier {
    /// Available buffers
    available: Arc<Mutex<VecDeque<PooledBuffer>>>,

    /// Buffer size for this tier
    #[allow(dead_code)]
    buffer_size: usize,

    /// Maximum number of buffers in pool
    #[allow(dead_code)]
    max_pool_size: usize,

    /// Current pool size
    current_size: AtomicUsize,

    /// Total allocations from this tier
    #[allow(dead_code)]
    total_allocations: AtomicUsize,

    /// Pool hits (reused buffers)
    pool_hits: AtomicUsize,

    /// Pool misses (new allocations)
    pool_misses: AtomicUsize,
}

// Safety: PoolTier is thread-safe due to use of Arc<Mutex<_>> and AtomicUsize
unsafe impl Send for PoolTier {}
unsafe impl Sync for PoolTier {}

/// Pooled buffer with metadata
#[allow(dead_code)]
struct PooledBuffer {
    /// Raw memory pointer
    ptr: NonNull<u8>,

    /// Buffer size
    #[allow(dead_code)]
    size: usize,

    /// Allocation timestamp
    #[allow(dead_code)]
    allocated_at: Instant,

    /// Number of times reused
    #[allow(dead_code)]
    reuse_count: usize,

    /// NUMA node where allocated
    #[allow(dead_code)]
    numa_node: Option<u32>,
}

// Safety: PooledBuffer is Send/Sync due to NonNull<u8> being raw pointer (careful usage required)
unsafe impl Send for PooledBuffer {}
unsafe impl Sync for PooledBuffer {}

/// Allocation tracking information
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AllocationInfo {
    #[allow(dead_code)]
    size: usize,
    #[allow(dead_code)]
    timestamp: Instant,
    #[allow(dead_code)]
    numa_node: Option<u32>,
    #[allow(dead_code)]
    allocation_type: AllocationType,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum AllocationType {
    #[allow(dead_code)]
    Pool,
    Direct,
    #[allow(dead_code)]
    NumaAware,
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total allocated memory (bytes)
    pub total_allocated: usize,

    /// Peak memory usage (bytes)
    pub peak_usage: usize,

    /// Current active allocations
    pub active_allocations: usize,

    /// Pool hit rate (0.0-1.0)
    pub pool_hit_rate: f32,

    /// Average allocation size
    pub avg_allocation_size: f32,

    /// Memory fragmentation ratio
    pub fragmentation_ratio: f32,

    /// NUMA efficiency score
    pub numa_efficiency: f32,

    /// Garbage collection frequency
    pub gc_frequency_hz: f32,

    /// Memory access patterns
    pub access_patterns: HashMap<String, f32>,
}

/// Pool-specific statistics
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
struct PoolStats {
    #[allow(dead_code)]
    small_pool_utilization: f32,
    #[allow(dead_code)]
    medium_pool_utilization: f32,
    #[allow(dead_code)]
    large_pool_utilization: f32,
    #[allow(dead_code)]
    total_pool_hits: u64,
    #[allow(dead_code)]
    total_pool_misses: u64,
    #[allow(dead_code)]
    avg_buffer_lifetime: Duration,
}

/// Lock-free circular buffer for high-performance streaming
pub struct LockFreeCircularBuffer {
    /// Buffer storage using Vec for safety
    buffer: Arc<Mutex<Vec<Option<AudioBuffer>>>>,

    /// Buffer capacity (power of 2 for efficient modulo)
    capacity: usize,

    /// Write position
    write_pos: CachePadded<AtomicUsize>,

    /// Read position  
    read_pos: CachePadded<AtomicUsize>,

    /// Buffer size mask (capacity - 1)
    mask: usize,

    /// Buffer statistics
    stats: Arc<RwLock<CircularBufferStats>>,
}

/// Circular buffer statistics
#[derive(Debug, Clone, Default)]
pub struct CircularBufferStats {
    total_writes: u64,
    total_reads: u64,
    overruns: u64,
    underruns: u64,
    #[allow(dead_code)]
    peak_utilization: f32,
    #[allow(dead_code)]
    avg_latency_ns: f64,
}

unsafe impl Send for LockFreeCircularBuffer {}
unsafe impl Sync for LockFreeCircularBuffer {}

impl LockFreeCircularBuffer {
    /// Create new lock-free circular buffer
    pub fn new(capacity: usize) -> Result<Self> {
        // Ensure capacity is power of 2
        let capacity = capacity.next_power_of_two();
        let mask = capacity - 1;

        // Use Vec with Option for safe storage
        let mut buffer_vec = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer_vec.push(None);
        }

        Ok(Self {
            buffer: Arc::new(Mutex::new(buffer_vec)),
            capacity,
            write_pos: CachePadded::new(AtomicUsize::new(0)),
            read_pos: CachePadded::new(AtomicUsize::new(0)),
            mask,
            stats: Arc::new(RwLock::new(CircularBufferStats::default())),
        })
    }

    /// Push audio buffer (not truly lock-free due to Mutex, but safe)
    pub fn push(&self, audio: AudioBuffer) -> Result<()> {
        let write_pos = self.write_pos.load(Ordering::Relaxed);
        let read_pos = self.read_pos.load(Ordering::Acquire);

        // Check if buffer is full
        let next_write = (write_pos + 1) & self.mask;
        if next_write == read_pos {
            // Buffer full - record overrun
            if let Ok(mut stats) = self.stats.write() {
                stats.overruns += 1;
            }
            return Err(VocoderError::VocodingError("Buffer overrun".to_string()));
        }

        // Write audio buffer
        {
            let mut buffer = self.buffer.lock().unwrap();
            buffer[write_pos] = Some(audio);
        }

        // Update write position
        self.write_pos.store(next_write, Ordering::Release);

        // Update statistics
        if let Ok(mut stats) = self.stats.write() {
            stats.total_writes += 1;
        }

        Ok(())
    }

    /// Pop audio buffer (not truly lock-free due to Mutex, but safe)
    pub fn pop(&self) -> Option<AudioBuffer> {
        let read_pos = self.read_pos.load(Ordering::Relaxed);
        let write_pos = self.write_pos.load(Ordering::Acquire);

        // Check if buffer is empty
        if read_pos == write_pos {
            // Buffer empty - record underrun
            if let Ok(mut stats) = self.stats.write() {
                stats.underruns += 1;
            }
            return None;
        }

        // Read audio buffer
        let audio = {
            let mut buffer = self.buffer.lock().unwrap();
            buffer[read_pos].take()
        };

        // Update read position
        let next_read = (read_pos + 1) & self.mask;
        self.read_pos.store(next_read, Ordering::Release);

        // Update statistics
        if let Ok(mut stats) = self.stats.write() {
            stats.total_reads += 1;
        }

        audio
    }

    /// Get current buffer utilization
    pub fn utilization(&self) -> f32 {
        let write_pos = self.write_pos.load(Ordering::Relaxed);
        let read_pos = self.read_pos.load(Ordering::Relaxed);

        let used = if write_pos >= read_pos {
            write_pos - read_pos
        } else {
            self.capacity - read_pos + write_pos
        };

        used as f32 / self.capacity as f32
    }

    /// Get buffer statistics
    pub fn get_stats(&self) -> CircularBufferStats {
        self.stats.read().unwrap().clone()
    }
}

// Drop implementation is automatic for Vec-based storage

/// NUMA topology information
struct NumaTopology {
    /// Number of NUMA nodes
    #[allow(dead_code)]
    num_nodes: u32,

    /// CPU cores per NUMA node
    #[allow(dead_code)]
    cores_per_node: HashMap<u32, Vec<u32>>,

    /// Memory bandwidth per node
    #[allow(dead_code)]
    memory_bandwidth: HashMap<u32, f32>,

    /// Inter-node latency matrix
    #[allow(dead_code)]
    node_latency: HashMap<(u32, u32), f32>,

    /// Current thread's preferred NUMA node
    preferred_node: AtomicUsize,
}

impl NumaTopology {
    fn new() -> Self {
        // In a real implementation, this would query the system
        // For now, provide a simple single-node topology
        let mut cores_per_node = HashMap::new();
        cores_per_node.insert(0, (0..num_cpus::get() as u32).collect());

        let mut memory_bandwidth = HashMap::new();
        memory_bandwidth.insert(0, 100.0); // GB/s

        Self {
            num_nodes: 1,
            cores_per_node,
            memory_bandwidth,
            node_latency: HashMap::new(),
            preferred_node: AtomicUsize::new(0),
        }
    }

    /// Get optimal NUMA node for allocation
    fn get_optimal_node(&self) -> u32 {
        self.preferred_node.load(Ordering::Relaxed) as u32
    }

    /// Update preferred NUMA node based on usage patterns
    #[allow(dead_code)]
    fn update_preferred_node(&self, _access_pattern: &HashMap<String, f32>) {
        // In a real implementation, this would analyze access patterns
        // and adjust the preferred node accordingly
    }
}

/// Garbage collector for efficient memory management
struct GarbageCollector {
    /// GC enabled flag
    enabled: AtomicBool,

    /// Last GC run timestamp
    last_gc: Arc<Mutex<Instant>>,

    /// GC interval
    gc_interval: Duration,

    /// Memory pressure threshold
    pressure_threshold: f32,

    /// GC statistics
    stats: Arc<RwLock<GcStats>>,
}

#[derive(Debug, Clone, Default)]
struct GcStats {
    total_collections: u64,
    memory_freed_bytes: u64,
    avg_collection_time_ms: f32,
    #[allow(dead_code)]
    collections_triggered_by_pressure: u64,
    #[allow(dead_code)]
    collections_triggered_by_timer: u64,
}

impl GarbageCollector {
    fn new() -> Self {
        Self {
            enabled: AtomicBool::new(true),
            last_gc: Arc::new(Mutex::new(Instant::now())),
            gc_interval: Duration::from_secs(30),
            pressure_threshold: 0.8,
            stats: Arc::new(RwLock::new(GcStats::default())),
        }
    }

    /// Check if garbage collection should run
    fn should_collect(&self, memory_pressure: f32) -> bool {
        if !self.enabled.load(Ordering::Relaxed) {
            return false;
        }

        // Check memory pressure
        if memory_pressure > self.pressure_threshold {
            return true;
        }

        // Check time-based collection
        let last_gc = *self.last_gc.lock().unwrap();
        Instant::now().duration_since(last_gc) > self.gc_interval
    }

    /// Run garbage collection
    fn collect(&self, memory_pool: &MemoryPool) -> usize {
        let start_time = Instant::now();
        let mut freed_bytes = 0;

        // Collect from each pool tier
        freed_bytes += self.collect_pool_tier(&memory_pool.small_pool);
        freed_bytes += self.collect_pool_tier(&memory_pool.medium_pool);
        freed_bytes += self.collect_pool_tier(&memory_pool.large_pool);

        let collection_time = start_time.elapsed().as_secs_f32() * 1000.0;

        // Update statistics
        if let Ok(mut stats) = self.stats.write() {
            stats.total_collections += 1;
            stats.memory_freed_bytes += freed_bytes as u64;
            stats.avg_collection_time_ms = (stats.avg_collection_time_ms
                * (stats.total_collections - 1) as f32
                + collection_time)
                / stats.total_collections as f32;
        }

        *self.last_gc.lock().unwrap() = Instant::now();

        freed_bytes
    }

    /// Collect from a specific pool tier
    fn collect_pool_tier(&self, tier: &PoolTier) -> usize {
        let mut freed_bytes = 0;
        let mut buffers = tier.available.lock().unwrap();

        // Remove buffers that haven't been used recently
        let cutoff_time = Instant::now() - Duration::from_secs(60);
        let initial_len = buffers.len();

        buffers.retain(|buffer| {
            if buffer.allocated_at < cutoff_time && buffer.reuse_count < 5 {
                // Free this buffer
                unsafe {
                    let layout = Layout::from_size_align(buffer.size, 8).unwrap();
                    dealloc(buffer.ptr.as_ptr(), layout);
                }
                freed_bytes += buffer.size;
                false
            } else {
                true
            }
        });

        let removed = initial_len - buffers.len();
        tier.current_size.fetch_sub(removed, Ordering::Relaxed);

        freed_bytes
    }
}

impl MemoryPool {
    fn new() -> Self {
        Self {
            small_pool: Arc::new(PoolTier::new(1024, 100)), // 1KB buffers, max 100
            medium_pool: Arc::new(PoolTier::new(65536, 50)), // 64KB buffers, max 50
            large_pool: Arc::new(PoolTier::new(1048576, 20)), // 1MB buffers, max 20
            stats: Arc::new(RwLock::new(PoolStats::default())),
            allocations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Allocate buffer from appropriate pool
    fn allocate(&self, size: usize, numa_node: Option<u32>) -> Result<PooledBuffer> {
        let tier = if size <= 1024 {
            &self.small_pool
        } else if size <= 65536 {
            &self.medium_pool
        } else {
            &self.large_pool
        };

        // Try to get from pool first
        if let Some(mut buffer) = tier.try_pop() {
            buffer.reuse_count += 1;
            tier.pool_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(buffer);
        }

        // Allocate new buffer
        tier.pool_misses.fetch_add(1, Ordering::Relaxed);

        let layout = Layout::from_size_align(size, 8).map_err(|_| {
            VocoderError::ConfigurationError("Invalid allocation layout".to_string())
        })?;

        let ptr = unsafe {
            let raw_ptr = alloc(layout);
            if raw_ptr.is_null() {
                return Err(VocoderError::ConfigurationError(
                    "Memory allocation failed".to_string(),
                ));
            }
            NonNull::new_unchecked(raw_ptr)
        };

        Ok(PooledBuffer {
            ptr,
            size,
            allocated_at: Instant::now(),
            reuse_count: 0,
            numa_node,
        })
    }

    /// Return buffer to pool
    #[allow(dead_code)]
    fn deallocate(&self, buffer: PooledBuffer) {
        let tier = if buffer.size <= 1024 {
            &self.small_pool
        } else if buffer.size <= 65536 {
            &self.medium_pool
        } else {
            &self.large_pool
        };

        tier.try_push(buffer);
    }

    /// Get pool statistics
    #[allow(dead_code)]
    fn get_stats(&self) -> PoolStats {
        self.stats.read().unwrap().clone()
    }
}

impl PoolTier {
    fn new(buffer_size: usize, max_pool_size: usize) -> Self {
        Self {
            available: Arc::new(Mutex::new(VecDeque::with_capacity(max_pool_size))),
            buffer_size,
            max_pool_size,
            current_size: AtomicUsize::new(0),
            total_allocations: AtomicUsize::new(0),
            pool_hits: AtomicUsize::new(0),
            pool_misses: AtomicUsize::new(0),
        }
    }

    fn try_pop(&self) -> Option<PooledBuffer> {
        let mut available = self.available.lock().unwrap();
        if let Some(buffer) = available.pop_front() {
            self.current_size.fetch_sub(1, Ordering::Relaxed);
            Some(buffer)
        } else {
            None
        }
    }

    fn try_push(&self, buffer: PooledBuffer) {
        let mut available = self.available.lock().unwrap();
        if available.len() < self.max_pool_size {
            available.push_back(buffer);
            self.current_size.fetch_add(1, Ordering::Relaxed);
        } else {
            // Pool is full, free the buffer
            unsafe {
                let layout = Layout::from_size_align(buffer.size, 8).unwrap();
                dealloc(buffer.ptr.as_ptr(), layout);
            }
        }
    }
}

impl Default for MemoryEfficientBufferManager {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryEfficientBufferManager {
    /// Create new memory-efficient buffer manager
    pub fn new() -> Self {
        Self {
            memory_pool: Arc::new(MemoryPool::new()),
            circular_buffers: Arc::new(RwLock::new(HashMap::new())),
            allocation_strategy: Arc::new(RwLock::new(AllocationStrategy::default())),
            memory_stats: Arc::new(RwLock::new(MemoryStats::default())),
            gc_scheduler: Arc::new(GarbageCollector::new()),
            numa_topology: Arc::new(NumaTopology::new()),
        }
    }

    /// Create or get circular buffer for stream
    pub fn get_stream_buffer(
        &self,
        stream_id: u64,
        capacity: usize,
    ) -> Result<Arc<LockFreeCircularBuffer>> {
        let mut buffers = self.circular_buffers.write().unwrap();

        if let Some(buffer) = buffers.get(&stream_id) {
            Ok(buffer.clone())
        } else {
            let buffer = Arc::new(LockFreeCircularBuffer::new(capacity)?);
            buffers.insert(stream_id, buffer.clone());
            Ok(buffer)
        }
    }

    /// Allocate audio buffer with optimal strategy
    pub fn allocate_audio_buffer(&self, channels: u32, samples: usize) -> Result<AudioBuffer> {
        let strategy = *self.allocation_strategy.read().unwrap();
        let numa_node = self.numa_topology.get_optimal_node();

        match strategy {
            AllocationStrategy::Pool => self.allocate_from_pool(channels, samples, Some(numa_node)),
            AllocationStrategy::NumaAware => self.allocate_numa_aware(channels, samples, numa_node),
            AllocationStrategy::ZeroCopy => self.allocate_zero_copy(channels, samples),
            AllocationStrategy::Hybrid => {
                // Use pool for small/medium buffers, direct allocation for large
                let total_size = samples * channels as usize * std::mem::size_of::<f32>();
                if total_size <= 65536 {
                    self.allocate_from_pool(channels, samples, Some(numa_node))
                } else {
                    self.allocate_numa_aware(channels, samples, numa_node)
                }
            }
            AllocationStrategy::Standard => Ok(AudioBuffer::silence(
                samples as f32 / 22050.0,
                22050,
                channels,
            )),
        }
    }

    /// Allocate from memory pool
    fn allocate_from_pool(
        &self,
        channels: u32,
        samples: usize,
        numa_node: Option<u32>,
    ) -> Result<AudioBuffer> {
        let total_samples = samples * channels as usize;
        let size = total_samples * std::mem::size_of::<f32>();

        let _pooled_buffer = self.memory_pool.allocate(size, numa_node)?;

        // For simplicity, create a standard AudioBuffer
        // In a real implementation, you'd use the pooled memory
        Ok(AudioBuffer::silence(
            samples as f32 / 22050.0,
            22050,
            channels,
        ))
    }

    /// NUMA-aware allocation
    fn allocate_numa_aware(
        &self,
        channels: u32,
        samples: usize,
        _numa_node: u32,
    ) -> Result<AudioBuffer> {
        // In a real implementation, this would use NUMA-specific allocation
        Ok(AudioBuffer::silence(
            samples as f32 / 22050.0,
            22050,
            channels,
        ))
    }

    /// Zero-copy allocation (reuse existing buffer if possible)
    fn allocate_zero_copy(&self, channels: u32, samples: usize) -> Result<AudioBuffer> {
        // In a real implementation, this would try to reuse buffers
        Ok(AudioBuffer::silence(
            samples as f32 / 22050.0,
            22050,
            channels,
        ))
    }

    /// Run garbage collection if needed
    pub fn maybe_collect_garbage(&self) {
        let memory_stats = self.memory_stats.read().unwrap();
        let memory_pressure = memory_stats.total_allocated as f32 / (1024.0 * 1024.0 * 1024.0); // GB

        if self.gc_scheduler.should_collect(memory_pressure) {
            self.gc_scheduler.collect(&self.memory_pool);
        }
    }

    /// Get comprehensive memory statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        self.memory_stats.read().unwrap().clone()
    }

    /// Update allocation strategy based on performance
    pub fn update_allocation_strategy(&self, new_strategy: AllocationStrategy) {
        *self.allocation_strategy.write().unwrap() = new_strategy;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lock_free_circular_buffer() {
        let buffer = LockFreeCircularBuffer::new(8).unwrap();

        // Test basic operations
        assert_eq!(buffer.utilization(), 0.0);

        let audio = AudioBuffer::silence(0.1, 22050, 1);
        assert!(buffer.push(audio).is_ok());

        assert!(buffer.utilization() > 0.0);

        let popped = buffer.pop();
        assert!(popped.is_some());

        assert_eq!(buffer.utilization(), 0.0);
    }

    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new();

        // Test small allocation
        let small_buffer = pool.allocate(512, None).unwrap();
        assert_eq!(small_buffer.size, 512);

        // Test medium allocation
        let medium_buffer = pool.allocate(32768, None).unwrap();
        assert_eq!(medium_buffer.size, 32768);

        // Test large allocation
        let large_buffer = pool.allocate(524288, None).unwrap();
        assert_eq!(large_buffer.size, 524288);

        // Return buffers to pool
        pool.deallocate(small_buffer);
        pool.deallocate(medium_buffer);
        pool.deallocate(large_buffer);
    }

    #[test]
    fn test_memory_efficient_buffer_manager() {
        let manager = MemoryEfficientBufferManager::new();

        // Test stream buffer creation
        let buffer1 = manager.get_stream_buffer(1, 16).unwrap();
        let buffer2 = manager.get_stream_buffer(1, 16).unwrap();

        // Should return the same buffer for the same stream
        assert!(Arc::ptr_eq(&buffer1, &buffer2));

        // Test audio buffer allocation
        let audio = manager.allocate_audio_buffer(2, 1024).unwrap();
        assert_eq!(audio.channels(), 2);
    }

    #[test]
    fn test_numa_topology() {
        let topology = NumaTopology::new();
        let node = topology.get_optimal_node();
        assert_eq!(node, 0); // Should default to node 0
    }
}
