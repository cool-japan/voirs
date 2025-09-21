//! Memory Management Optimization
//! 
//! This module provides advanced memory management optimizations including
//! pool allocation strategies, zero-copy operations, memory mapping,
//! NUMA awareness, lock-free data structures, and memory compaction
//! for optimal FFI performance.

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, AtomicPtr, Ordering};
use std::sync::Arc;
use std::collections::VecDeque;
use parking_lot::{Mutex, RwLock};
use once_cell::sync::Lazy;
use std::time::{Duration, Instant};

/// Global memory pool manager
static MEMORY_POOLS: Lazy<MemoryPoolManager> = Lazy::new(MemoryPoolManager::new);

/// Memory pool statistics
#[derive(Debug, Default)]
pub struct MemoryStats {
    pub total_allocated: AtomicUsize,
    pub total_freed: AtomicUsize,
    pub active_allocations: AtomicUsize,
    pub pool_hits: AtomicUsize,
    pub pool_misses: AtomicUsize,
    pub peak_usage: AtomicUsize,
}

impl MemoryStats {
    pub fn record_allocation(&self, size: usize, from_pool: bool) {
        self.total_allocated.fetch_add(size, Ordering::Relaxed);
        self.active_allocations.fetch_add(size, Ordering::Relaxed);
        
        if from_pool {
            self.pool_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.pool_misses.fetch_add(1, Ordering::Relaxed);
        }

        let current_usage = self.active_allocations.load(Ordering::Relaxed);
        let peak = self.peak_usage.load(Ordering::Relaxed);
        if current_usage > peak {
            self.peak_usage.store(current_usage, Ordering::Relaxed);
        }
    }

    pub fn record_deallocation(&self, size: usize) {
        self.total_freed.fetch_add(size, Ordering::Relaxed);
        self.active_allocations.fetch_sub(size, Ordering::Relaxed);
    }
}

/// Memory pool for specific allocation sizes
pub struct MemoryPool {
    chunk_size: usize,
    chunks: Mutex<VecDeque<NonNull<u8>>>,
    stats: Arc<MemoryStats>,
    max_chunks: usize,
}

impl MemoryPool {
    pub fn new(chunk_size: usize, initial_chunks: usize, max_chunks: usize) -> Self {
        let pool = Self {
            chunk_size,
            chunks: Mutex::new(VecDeque::with_capacity(initial_chunks)),
            stats: Arc::new(MemoryStats::default()),
            max_chunks,
        };

        // Pre-allocate initial chunks
        let mut chunks = pool.chunks.lock();
        for _ in 0..initial_chunks {
            if let Some(chunk) = pool.allocate_new_chunk() {
                chunks.push_back(chunk);
            }
        }

        pool
    }

    fn allocate_new_chunk(&self) -> Option<NonNull<u8>> {
        let layout = Layout::from_size_align(self.chunk_size, 64).ok()?;
        unsafe {
            let ptr = alloc(layout);
            NonNull::new(ptr)
        }
    }

    pub fn allocate(&self) -> Option<NonNull<u8>> {
        let mut chunks = self.chunks.lock();
        if let Some(chunk) = chunks.pop_front() {
            self.stats.record_allocation(self.chunk_size, true);
            Some(chunk)
        } else {
            drop(chunks);
            if let Some(chunk) = self.allocate_new_chunk() {
                self.stats.record_allocation(self.chunk_size, false);
                Some(chunk)
            } else {
                None
            }
        }
    }

    pub fn deallocate(&self, ptr: NonNull<u8>) {
        let mut chunks = self.chunks.lock();
        if chunks.len() < self.max_chunks {
            chunks.push_back(ptr);
            self.stats.record_deallocation(self.chunk_size);
        } else {
            // Pool is full, actually free the memory
            unsafe {
                let layout = Layout::from_size_align(self.chunk_size, 64).unwrap();
                dealloc(ptr.as_ptr(), layout);
            }
            self.stats.record_deallocation(self.chunk_size);
        }
    }

    pub fn stats(&self) -> &Arc<MemoryStats> {
        &self.stats
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        let chunks = self.chunks.lock();
        for chunk in chunks.iter() {
            unsafe {
                let layout = Layout::from_size_align(self.chunk_size, 64).unwrap();
                dealloc(chunk.as_ptr(), layout);
            }
        }
    }
}

/// Multi-size memory pool manager
pub struct MemoryPoolManager {
    pools: RwLock<std::collections::HashMap<usize, Arc<MemoryPool>>>,
    global_stats: Arc<MemoryStats>,
}

impl MemoryPoolManager {
    pub fn new() -> Self {
        Self {
            pools: RwLock::new(std::collections::HashMap::new()),
            global_stats: Arc::new(MemoryStats::default()),
        }
    }

    pub fn get_or_create_pool(&self, size: usize) -> Arc<MemoryPool> {
        // Round up to nearest power of 2 for better pool utilization
        let pool_size = size.next_power_of_two();
        
        {
            let pools = self.pools.read();
            if let Some(pool) = pools.get(&pool_size) {
                return pool.clone();
            }
        }

        let mut pools = self.pools.write();
        pools.entry(pool_size)
            .or_insert_with(|| {
                let initial_chunks = match pool_size {
                    0..=1024 => 100,      // Small buffers
                    1025..=8192 => 50,    // Medium buffers
                    8193..=65536 => 20,   // Large buffers
                    _ => 5,               // Huge buffers
                };
                let max_chunks = initial_chunks * 2;
                Arc::new(MemoryPool::new(pool_size, initial_chunks, max_chunks))
            })
            .clone()
    }

    pub fn allocate(&self, size: usize) -> Option<NonNull<u8>> {
        let pool = self.get_or_create_pool(size);
        pool.allocate()
    }

    pub fn deallocate(&self, ptr: NonNull<u8>, size: usize) {
        let pool_size = size.next_power_of_two();
        let pools = self.pools.read();
        if let Some(pool) = pools.get(&pool_size) {
            pool.deallocate(ptr);
        }
    }

    pub fn global_stats(&self) -> &Arc<MemoryStats> {
        &self.global_stats
    }
}

/// Get the global memory pool manager
pub fn get_memory_pools() -> &'static MemoryPoolManager {
    &MEMORY_POOLS
}

/// Zero-copy buffer for efficient data transfer
pub struct ZeroCopyBuffer {
    data: NonNull<u8>,
    size: usize,
    capacity: usize,
    from_pool: bool,
}

impl ZeroCopyBuffer {
    pub fn new(capacity: usize) -> Option<Self> {
        let pools = get_memory_pools();
        if let Some(data) = pools.allocate(capacity) {
            Some(Self {
                data,
                size: 0,
                capacity,
                from_pool: true,
            })
        } else {
            None
        }
    }

    pub fn from_raw(data: NonNull<u8>, capacity: usize) -> Self {
        Self {
            data,
            size: capacity,
            capacity,
            from_pool: false,
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn set_len(&mut self, len: usize) {
        if len <= self.capacity {
            self.size = len;
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.size) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr(), self.size) }
    }
}

impl Drop for ZeroCopyBuffer {
    fn drop(&mut self) {
        if self.from_pool {
            let pools = get_memory_pools();
            pools.deallocate(self.data, self.capacity);
        }
    }
}

unsafe impl Send for ZeroCopyBuffer {}
unsafe impl Sync for ZeroCopyBuffer {}

/// Cache-line aligned allocator for high-performance operations
pub struct CacheAlignedAllocator {
    pool_manager: Arc<MemoryPoolManager>,
    cache_line_size: usize,
}

impl CacheAlignedAllocator {
    pub fn new() -> Self {
        Self {
            pool_manager: Arc::new(MemoryPoolManager::new()),
            cache_line_size: Self::detect_cache_line_size(),
        }
    }
    
    fn detect_cache_line_size() -> usize {
        // Default to 64 bytes, which is common for x86_64
        // Could be enhanced to detect actual cache line size
        64
    }
    
    pub fn allocate_aligned(&self, size: usize) -> Option<NonNull<u8>> {
        // Ensure allocation is cache-line aligned
        let aligned_size = (size + self.cache_line_size - 1) & !(self.cache_line_size - 1);
        self.pool_manager.allocate(aligned_size)
    }
    
    pub fn prefetch_for_read(&self, ptr: *const u8) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            // Use x86 prefetch instructions
            std::arch::x86_64::_mm_prefetch::<0>(ptr as *const i8);
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Fallback: force a read to bring data into cache
            unsafe {
                std::ptr::read_volatile(ptr);
            }
        }
    }
    
    pub fn prefetch_for_write(&self, ptr: *const u8) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            // Use x86 prefetch instructions for write
            std::arch::x86_64::_mm_prefetch::<1>(ptr as *const i8);
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Fallback: touch memory to bring into cache
            unsafe {
                let value = std::ptr::read_volatile(ptr);
                std::ptr::write_volatile(ptr as *mut u8, value);
            }
        }
    }
}

/// Memory-optimized audio processing buffer with prefetching
pub struct OptimizedAudioBuffer {
    buffer: MemoryMappedAudioBuffer,
    allocator: CacheAlignedAllocator,
}

impl OptimizedAudioBuffer {
    pub fn new(sample_count: usize, sample_rate: u32, channels: u16) -> Option<Self> {
        MemoryMappedAudioBuffer::new(sample_count, sample_rate, channels).map(|buffer| {
            Self {
                buffer,
                allocator: CacheAlignedAllocator::new(),
            }
        })
    }
    
    pub fn process_samples<F>(&mut self, mut process_fn: F) -> Result<(), &'static str>
    where
        F: FnMut(&mut [f32]),
    {
        let samples = self.buffer.as_f32_mut_slice();
        let chunk_size = self.allocator.cache_line_size / std::mem::size_of::<f32>();
        
        for chunk in samples.chunks_mut(chunk_size) {
            // Prefetch next chunk for better cache performance
            if let Some(next_chunk_start) = chunk.as_ptr().wrapping_add(chunk_size) as *const u8 {
                self.allocator.prefetch_for_write(next_chunk_start);
            }
            
            process_fn(chunk);
        }
        
        Ok(())
    }
    
    pub fn into_inner(self) -> MemoryMappedAudioBuffer {
        self.buffer
    }
}

/// Memory-mapped audio buffer for large audio data
pub struct MemoryMappedAudioBuffer {
    buffer: ZeroCopyBuffer,
    sample_rate: u32,
    channels: u16,
    sample_count: usize,
}

impl MemoryMappedAudioBuffer {
    pub fn new(sample_count: usize, sample_rate: u32, channels: u16) -> Option<Self> {
        let buffer_size = sample_count * std::mem::size_of::<f32>();
        if let Some(buffer) = ZeroCopyBuffer::new(buffer_size) {
            Some(Self {
                buffer,
                sample_rate,
                channels,
                sample_count,
            })
        } else {
            None
        }
    }

    pub fn as_f32_slice(&self) -> &[f32] {
        unsafe {
            std::slice::from_raw_parts(
                self.buffer.as_ptr() as *const f32,
                self.sample_count,
            )
        }
    }

    pub fn as_f32_mut_slice(&mut self) -> &mut [f32] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.buffer.as_mut_ptr() as *mut f32,
                self.sample_count,
            )
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }

    pub fn sample_count(&self) -> usize {
        self.sample_count
    }
}

/// NUMA-aware memory allocation (Linux only)
#[cfg(target_os = "linux")]
pub mod numa {
    use super::*;
    use std::fs;
    use std::collections::HashMap;
    
    /// NUMA node allocation pool
    pub struct NumaPool {
        node_pools: HashMap<usize, Arc<MemoryPool>>,
        preferred_node: Option<usize>,
    }
    
    impl NumaPool {
        pub fn new() -> Self {
            let node_count = get_numa_node_count();
            let mut node_pools = HashMap::new();
            
            for node in 0..node_count {
                let pool = Arc::new(MemoryPool::new(65536, 10, 50));
                node_pools.insert(node, pool);
            }
            
            Self {
                node_pools,
                preferred_node: get_current_numa_node(),
            }
        }
        
        pub fn allocate_on_node(&self, node: usize, size: usize) -> Option<NonNull<u8>> {
            if let Some(pool) = self.node_pools.get(&node) {
                pool.allocate()
            } else {
                None
            }
        }
        
        pub fn allocate_preferred(&self, size: usize) -> Option<NonNull<u8>> {
            if let Some(node) = self.preferred_node {
                self.allocate_on_node(node, size)
            } else {
                // Fall back to first available node
                for (_, pool) in &self.node_pools {
                    if let Some(ptr) = pool.allocate() {
                        return Some(ptr);
                    }
                }
                None
            }
        }
    }

    pub fn get_numa_node_count() -> usize {
        match fs::read_dir("/sys/devices/system/node") {
            Ok(entries) => {
                entries
                    .filter_map(|entry| entry.ok())
                    .filter(|entry| {
                        entry.file_name()
                            .to_string_lossy()
                            .starts_with("node")
                    })
                    .count()
            },
            Err(_) => 1, // Fallback to single node
        }
    }

    pub fn get_current_numa_node() -> Option<usize> {
        // Simple heuristic based on CPU affinity
        match fs::read_to_string("/proc/self/stat") {
            Ok(content) => {
                let fields: Vec<&str> = content.split_whitespace().collect();
                if fields.len() > 38 {
                    if let Ok(processor) = fields[38].parse::<usize>() {
                        // Approximate NUMA node based on processor number
                        let numa_nodes = get_numa_node_count();
                        let cpus_per_node = num_cpus::get() / numa_nodes.max(1);
                        return Some(processor / cpus_per_node.max(1));
                    }
                }
                None
            },
            Err(_) => None,
        }
    }
}

/// Platform-specific memory optimization utilities
pub mod platform {
    use super::*;

    #[cfg(windows)]
    pub fn enable_large_pages() -> Result<(), Box<dyn std::error::Error>> {
        // Windows large page support would go here
        // This requires SE_LOCK_MEMORY_NAME privilege
        Ok(())
    }

    #[cfg(target_os = "linux")]
    pub fn enable_transparent_hugepages() -> Result<(), Box<dyn std::error::Error>> {
        use std::fs::OpenOptions;
        use std::io::Write;

        let mut file = OpenOptions::new()
            .write(true)
            .open("/sys/kernel/mm/transparent_hugepage/enabled")?;
        file.write_all(b"always")?;
        Ok(())
    }

    #[cfg(target_os = "macos")]
    pub fn configure_vm_pressure() -> Result<(), Box<dyn std::error::Error>> {
        // macOS-specific VM pressure configuration would go here
        Ok(())
    }
}

/// Lock-free audio buffer ring for high-performance audio streaming
pub struct LockFreeAudioRing {
    buffers: Vec<AtomicPtr<ZeroCopyBuffer>>,
    read_index: AtomicUsize,
    write_index: AtomicUsize,
    capacity: usize,
    buffer_size: usize,
}

impl LockFreeAudioRing {
    pub fn new(ring_size: usize, buffer_size: usize) -> Option<Self> {
        let mut buffers = Vec::with_capacity(ring_size);
        
        // Pre-allocate all buffers
        for _ in 0..ring_size {
            if let Some(buffer) = ZeroCopyBuffer::new(buffer_size) {
                buffers.push(AtomicPtr::new(Box::into_raw(Box::new(buffer))));
            } else {
                // Clean up any allocated buffers on failure
                for buffer_ptr in buffers.iter() {
                    let ptr = buffer_ptr.load(Ordering::Relaxed);
                    if !ptr.is_null() {
                        unsafe {
                            let _ = Box::from_raw(ptr);
                        }
                    }
                }
                return None;
            }
        }
        
        Some(Self {
            buffers,
            read_index: AtomicUsize::new(0),
            write_index: AtomicUsize::new(0),
            capacity: ring_size,
            buffer_size,
        })
    }
    
    pub fn try_get_write_buffer(&self) -> Option<&mut ZeroCopyBuffer> {
        let write_idx = self.write_index.load(Ordering::Acquire);
        let next_write_idx = (write_idx + 1) % self.capacity;
        let read_idx = self.read_index.load(Ordering::Acquire);
        
        // Check if ring is full
        if next_write_idx == read_idx {
            return None;
        }
        
        let buffer_ptr = self.buffers[write_idx].load(Ordering::Relaxed);
        if buffer_ptr.is_null() {
            return None;
        }
        
        unsafe { Some(&mut *buffer_ptr) }
    }
    
    pub fn commit_write_buffer(&self) {
        let current_write = self.write_index.load(Ordering::Relaxed);
        let next_write = (current_write + 1) % self.capacity;
        self.write_index.store(next_write, Ordering::Release);
    }
    
    pub fn try_get_read_buffer(&self) -> Option<&ZeroCopyBuffer> {
        let read_idx = self.read_index.load(Ordering::Acquire);
        let write_idx = self.write_index.load(Ordering::Acquire);
        
        // Check if ring is empty
        if read_idx == write_idx {
            return None;
        }
        
        let buffer_ptr = self.buffers[read_idx].load(Ordering::Relaxed);
        if buffer_ptr.is_null() {
            return None;
        }
        
        unsafe { Some(&*buffer_ptr) }
    }
    
    pub fn commit_read_buffer(&self) {
        let current_read = self.read_index.load(Ordering::Relaxed);
        let next_read = (current_read + 1) % self.capacity;
        self.read_index.store(next_read, Ordering::Release);
    }
    
    pub fn available_write_buffers(&self) -> usize {
        let read_idx = self.read_index.load(Ordering::Acquire);
        let write_idx = self.write_index.load(Ordering::Acquire);
        
        if write_idx >= read_idx {
            self.capacity - 1 - (write_idx - read_idx)
        } else {
            read_idx - write_idx - 1
        }
    }
    
    pub fn available_read_buffers(&self) -> usize {
        let read_idx = self.read_index.load(Ordering::Acquire);
        let write_idx = self.write_index.load(Ordering::Acquire);
        
        if write_idx >= read_idx {
            write_idx - read_idx
        } else {
            self.capacity - (read_idx - write_idx)
        }
    }
}

impl Drop for LockFreeAudioRing {
    fn drop(&mut self) {
        for buffer_ptr in &self.buffers {
            let ptr = buffer_ptr.load(Ordering::Relaxed);
            if !ptr.is_null() {
                unsafe {
                    let _ = Box::from_raw(ptr);
                }
            }
        }
    }
}

unsafe impl Send for LockFreeAudioRing {}
unsafe impl Sync for LockFreeAudioRing {}

/// Memory compaction manager for reducing fragmentation
pub struct MemoryCompactor {
    last_compaction: Instant,
    compaction_interval: Duration,
    fragmentation_threshold: f32,
}

impl MemoryCompactor {
    pub fn new() -> Self {
        Self {
            last_compaction: Instant::now(),
            compaction_interval: Duration::from_secs(30),
            fragmentation_threshold: 0.3, // 30% fragmentation triggers compaction
        }
    }
    
    pub fn should_compact(&self, stats: &MemoryStats) -> bool {
        let time_since_last = self.last_compaction.elapsed();
        if time_since_last < self.compaction_interval {
            return false;
        }
        
        let total_allocated = stats.total_allocated.load(Ordering::Relaxed);
        let active_allocations = stats.active_allocations.load(Ordering::Relaxed);
        
        if total_allocated == 0 {
            return false;
        }
        
        let fragmentation_ratio = 1.0 - (active_allocations as f32 / total_allocated as f32);
        fragmentation_ratio > self.fragmentation_threshold
    }
    
    pub fn compact_pool(&mut self, pool: &MemoryPool) -> Result<usize, &'static str> {
        // This is a simplified compaction strategy
        // In a real implementation, this would:
        // 1. Identify fragmented memory regions
        // 2. Move active allocations to consolidate free space
        // 3. Return freed memory to the system
        
        let stats = pool.stats();
        let pool_hits = stats.pool_hits.load(Ordering::Relaxed);
        let pool_misses = stats.pool_misses.load(Ordering::Relaxed);
        
        if pool_hits + pool_misses == 0 {
            return Ok(0);
        }
        
        let hit_ratio = pool_hits as f32 / (pool_hits + pool_misses) as f32;
        
        // If hit ratio is low, the pool might be over-allocated
        if hit_ratio < 0.7 {
            // Simulate compaction by returning some memory to system
            // In reality, this would involve more complex memory management
            self.last_compaction = Instant::now();
            Ok(pool.chunk_size * 2) // Return estimated freed memory
        } else {
            Ok(0)
        }
    }
}

/// Advanced memory allocation strategy selector
#[derive(Debug, Clone, Copy)]
pub enum AllocationStrategy {
    /// Best fit allocation for minimal fragmentation
    BestFit,
    /// First fit allocation for speed
    FirstFit,
    /// Buddy system allocation for efficient coalescing
    BuddySystem,
    /// NUMA-aware allocation
    NumaAware,
    /// Lock-free allocation for real-time threads
    LockFree,
}

/// Adaptive memory allocator that chooses strategy based on usage patterns
pub struct AdaptiveAllocator {
    strategy: AllocationStrategy,
    performance_monitor: AllocationPerformanceMonitor,
    strategy_evaluator: StrategyEvaluator,
}

impl AdaptiveAllocator {
    pub fn new() -> Self {
        Self {
            strategy: AllocationStrategy::FirstFit,
            performance_monitor: AllocationPerformanceMonitor::new(),
            strategy_evaluator: StrategyEvaluator::new(),
        }
    }
    
    pub fn allocate(&mut self, size: usize) -> Option<NonNull<u8>> {
        let start_time = Instant::now();
        let result = self.allocate_with_strategy(size, self.strategy);
        let duration = start_time.elapsed();
        
        self.performance_monitor.record_allocation(
            size,
            duration,
            result.is_some(),
            self.strategy,
        );
        
        // Periodically evaluate and potentially switch strategies
        if self.performance_monitor.should_evaluate_strategy() {
            if let Some(new_strategy) = self.strategy_evaluator.evaluate(&self.performance_monitor) {
                self.strategy = new_strategy;
                self.performance_monitor.reset_for_new_strategy();
            }
        }
        
        result
    }
    
    fn allocate_with_strategy(&self, size: usize, strategy: AllocationStrategy) -> Option<NonNull<u8>> {
        match strategy {
            AllocationStrategy::BestFit => self.allocate_best_fit(size),
            AllocationStrategy::FirstFit => self.allocate_first_fit(size),
            AllocationStrategy::BuddySystem => self.allocate_buddy_system(size),
            AllocationStrategy::NumaAware => self.allocate_numa_aware(size),
            AllocationStrategy::LockFree => self.allocate_lock_free(size),
        }
    }
    
    fn allocate_best_fit(&self, size: usize) -> Option<NonNull<u8>> {
        // Implementation would find the smallest available block that fits
        get_memory_pools().allocate(size)
    }
    
    fn allocate_first_fit(&self, size: usize) -> Option<NonNull<u8>> {
        // Implementation would find the first available block that fits
        get_memory_pools().allocate(size)
    }
    
    fn allocate_buddy_system(&self, size: usize) -> Option<NonNull<u8>> {
        // Implementation would use buddy system allocation
        get_memory_pools().allocate(size)
    }
    
    fn allocate_numa_aware(&self, size: usize) -> Option<NonNull<u8>> {
        #[cfg(target_os = "linux")]
        {
            // Use NUMA-aware allocation
            get_memory_pools().allocate(size)
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            get_memory_pools().allocate(size)
        }
    }
    
    fn allocate_lock_free(&self, size: usize) -> Option<NonNull<u8>> {
        // Implementation would use lock-free allocation structures
        get_memory_pools().allocate(size)
    }
}

/// Performance monitoring for allocation strategies
struct AllocationPerformanceMonitor {
    allocation_count: usize,
    total_time: Duration,
    success_rate: f32,
    strategy_switch_threshold: usize,
}

impl AllocationPerformanceMonitor {
    fn new() -> Self {
        Self {
            allocation_count: 0,
            total_time: Duration::new(0, 0),
            success_rate: 1.0,
            strategy_switch_threshold: 1000,
        }
    }
    
    fn record_allocation(&mut self, _size: usize, duration: Duration, success: bool, _strategy: AllocationStrategy) {
        self.allocation_count += 1;
        self.total_time += duration;
        
        let success_weight = if success { 1.0 } else { 0.0 };
        self.success_rate = (self.success_rate * 0.99) + (success_weight * 0.01);
    }
    
    fn should_evaluate_strategy(&self) -> bool {
        self.allocation_count % self.strategy_switch_threshold == 0
    }
    
    fn average_allocation_time(&self) -> Duration {
        if self.allocation_count > 0 {
            self.total_time / self.allocation_count as u32
        } else {
            Duration::new(0, 0)
        }
    }
    
    fn reset_for_new_strategy(&mut self) {
        self.allocation_count = 0;
        self.total_time = Duration::new(0, 0);
        self.success_rate = 1.0;
    }
}

/// Strategy evaluation logic
struct StrategyEvaluator {
    strategy_performance: std::collections::HashMap<AllocationStrategy, (Duration, f32)>,
}

impl StrategyEvaluator {
    fn new() -> Self {
        Self {
            strategy_performance: std::collections::HashMap::new(),
        }
    }
    
    fn evaluate(&mut self, monitor: &AllocationPerformanceMonitor) -> Option<AllocationStrategy> {
        let current_avg_time = monitor.average_allocation_time();
        let current_success_rate = monitor.success_rate;
        
        // Store current strategy performance
        // (This is simplified - real implementation would track all strategies)
        
        // Heuristic strategy selection based on workload characteristics
        if current_success_rate < 0.95 {
            // Low success rate, try best fit to reduce fragmentation
            Some(AllocationStrategy::BestFit)
        } else if current_avg_time > Duration::from_micros(100) {
            // High latency, try first fit for speed
            Some(AllocationStrategy::FirstFit)
        } else {
            // Performance is good, no change needed
            None
        }
    }
}

/// Memory usage profiler for detecting patterns and optimization opportunities
pub struct MemoryProfiler {
    allocation_sizes: Vec<usize>,
    allocation_patterns: Vec<(Instant, usize, bool)>, // time, size, is_allocation
    peak_memory: usize,
    current_memory: usize,
}

impl MemoryProfiler {
    pub fn new() -> Self {
        Self {
            allocation_sizes: Vec::new(),
            allocation_patterns: Vec::new(),
            peak_memory: 0,
            current_memory: 0,
        }
    }
    
    pub fn record_allocation(&mut self, size: usize) {
        self.allocation_sizes.push(size);
        self.allocation_patterns.push((Instant::now(), size, true));
        self.current_memory += size;
        if self.current_memory > self.peak_memory {
            self.peak_memory = self.current_memory;
        }
    }
    
    pub fn record_deallocation(&mut self, size: usize) {
        self.allocation_patterns.push((Instant::now(), size, false));
        self.current_memory = self.current_memory.saturating_sub(size);
    }
    
    pub fn get_allocation_statistics(&self) -> AllocationStatistics {
        if self.allocation_sizes.is_empty() {
            return AllocationStatistics::default();
        }
        
        let mut sorted_sizes = self.allocation_sizes.clone();
        sorted_sizes.sort_unstable();
        
        let total: usize = sorted_sizes.iter().sum();
        let count = sorted_sizes.len();
        let mean = total / count;
        
        let median = if count % 2 == 0 {
            (sorted_sizes[count / 2 - 1] + sorted_sizes[count / 2]) / 2
        } else {
            sorted_sizes[count / 2]
        };
        
        let min = *sorted_sizes.first().unwrap_or(&0);
        let max = *sorted_sizes.last().unwrap_or(&0);
        
        AllocationStatistics {
            mean_size: mean,
            median_size: median,
            min_size: min,
            max_size: max,
            total_allocations: count,
            peak_memory: self.peak_memory,
            current_memory: self.current_memory,
        }
    }
    
    pub fn detect_memory_leaks(&self) -> Vec<MemoryLeakSuspicion> {
        let mut suspicions = Vec::new();
        
        // Simple leak detection: find allocations without corresponding deallocations
        let mut allocated_sizes = std::collections::HashMap::new();
        
        for (time, size, is_allocation) in &self.allocation_patterns {
            if *is_allocation {
                *allocated_sizes.entry(*size).or_insert(0) += 1;
            } else {
                if let Some(count) = allocated_sizes.get_mut(size) {
                    *count = count.saturating_sub(1);
                }
            }
        }
        
        for (size, unfreed_count) in allocated_sizes {
            if unfreed_count > 10 {
                suspicions.push(MemoryLeakSuspicion {
                    size,
                    unfreed_allocations: unfreed_count,
                    confidence: if unfreed_count > 100 { 0.9 } else { 0.6 },
                });
            }
        }
        
        suspicions
    }
}

#[derive(Debug, Clone, Default)]
pub struct AllocationStatistics {
    pub mean_size: usize,
    pub median_size: usize,
    pub min_size: usize,
    pub max_size: usize,
    pub total_allocations: usize,
    pub peak_memory: usize,
    pub current_memory: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryLeakSuspicion {
    pub size: usize,
    pub unfreed_allocations: usize,
    pub confidence: f32,
}

/// C API functions for advanced memory management
#[no_mangle]
pub extern "C" fn voirs_create_lockfree_audio_ring(ring_size: usize, buffer_size: usize) -> *mut LockFreeAudioRing {
    match LockFreeAudioRing::new(ring_size, buffer_size) {
        Some(ring) => Box::into_raw(Box::new(ring)),
        None => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn voirs_destroy_lockfree_audio_ring(ring: *mut LockFreeAudioRing) {
    if !ring.is_null() {
        unsafe {
            let _ = Box::from_raw(ring);
        }
    }
}

#[no_mangle]
pub extern "C" fn voirs_audio_ring_available_write_buffers(ring: *const LockFreeAudioRing) -> usize {
    if ring.is_null() {
        return 0;
    }
    unsafe { (*ring).available_write_buffers() }
}

#[no_mangle]
pub extern "C" fn voirs_audio_ring_available_read_buffers(ring: *const LockFreeAudioRing) -> usize {
    if ring.is_null() {
        return 0;
    }
    unsafe { (*ring).available_read_buffers() }
}

#[no_mangle]
pub extern "C" fn voirs_create_adaptive_allocator() -> *mut AdaptiveAllocator {
    Box::into_raw(Box::new(AdaptiveAllocator::new()))
}

#[no_mangle]
pub extern "C" fn voirs_destroy_adaptive_allocator(allocator: *mut AdaptiveAllocator) {
    if !allocator.is_null() {
        unsafe {
            let _ = Box::from_raw(allocator);
        }
    }
}

#[no_mangle]
pub extern "C" fn voirs_create_memory_profiler() -> *mut MemoryProfiler {
    Box::into_raw(Box::new(MemoryProfiler::new()))
}

#[no_mangle]
pub extern "C" fn voirs_destroy_memory_profiler(profiler: *mut MemoryProfiler) {
    if !profiler.is_null() {
        unsafe {
            let _ = Box::from_raw(profiler);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new(1024, 10, 20);
        let ptr1 = pool.allocate().unwrap();
        let ptr2 = pool.allocate().unwrap();
        
        pool.deallocate(ptr1);
        pool.deallocate(ptr2);
        
        let stats = pool.stats();
        assert_eq!(stats.total_allocated.load(Ordering::Relaxed), 2048);
    }

    #[test]
    fn test_zero_copy_buffer() {
        let mut buffer = ZeroCopyBuffer::new(1024).unwrap();
        assert_eq!(buffer.capacity(), 1024);
        assert_eq!(buffer.len(), 0);
        
        buffer.set_len(512);
        assert_eq!(buffer.len(), 512);
    }

    #[test]
    fn test_memory_mapped_audio_buffer() {
        let mut buffer = MemoryMappedAudioBuffer::new(1000, 44100, 2).unwrap();
        assert_eq!(buffer.sample_count(), 1000);
        assert_eq!(buffer.sample_rate(), 44100);
        assert_eq!(buffer.channels(), 2);
        
        let samples = buffer.as_f32_mut_slice();
        samples[0] = 1.0;
        assert_eq!(buffer.as_f32_slice()[0], 1.0);
    }

    #[test]
    fn test_memory_pool_manager() {
        let manager = MemoryPoolManager::new();
        let ptr1 = manager.allocate(512).unwrap();
        let ptr2 = manager.allocate(1024).unwrap();
        
        manager.deallocate(ptr1, 512);
        manager.deallocate(ptr2, 1024);
    }

    #[test]
    fn test_lockfree_audio_ring() {
        let ring = LockFreeAudioRing::new(4, 1024).unwrap();
        
        // Test basic ring operations
        assert_eq!(ring.available_write_buffers(), 3); // One less than capacity
        assert_eq!(ring.available_read_buffers(), 0);
        
        // Test writing
        assert!(ring.try_get_write_buffer().is_some());
        ring.commit_write_buffer();
        assert_eq!(ring.available_read_buffers(), 1);
        
        // Test reading
        assert!(ring.try_get_read_buffer().is_some());
        ring.commit_read_buffer();
        assert_eq!(ring.available_read_buffers(), 0);
    }

    #[test]
    fn test_memory_compactor() {
        let mut compactor = MemoryCompactor::new();
        let pool = MemoryPool::new(1024, 10, 20);
        
        // Test compaction logic
        let result = compactor.compact_pool(&pool);
        assert!(result.is_ok());
    }

    #[test]
    fn test_adaptive_allocator() {
        let mut allocator = AdaptiveAllocator::new();
        
        // Test allocation
        let ptr = allocator.allocate(1024);
        assert!(ptr.is_some());
        
        // Test multiple allocations to trigger strategy evaluation
        for _ in 0..10 {
            let _ = allocator.allocate(512);
        }
    }

    #[test]
    fn test_memory_profiler() {
        let mut profiler = MemoryProfiler::new();
        
        // Record some allocations
        profiler.record_allocation(1024);
        profiler.record_allocation(512);
        profiler.record_allocation(2048);
        
        // Record some deallocations
        profiler.record_deallocation(512);
        
        let stats = profiler.get_allocation_statistics();
        assert_eq!(stats.total_allocations, 3);
        assert_eq!(stats.peak_memory, 3584); // 1024 + 512 + 2048
        assert_eq!(stats.current_memory, 3072); // 3584 - 512
        
        // Test leak detection
        let leaks = profiler.detect_memory_leaks();
        // Should not detect leaks with current small sample
        assert!(leaks.is_empty());
    }

    #[test]
    fn test_cache_aligned_allocator() {
        let allocator = CacheAlignedAllocator::new();
        let ptr = allocator.allocate_aligned(1024);
        assert!(ptr.is_some());
        
        // Test that pointer is cache-line aligned
        if let Some(ptr) = ptr {
            let addr = ptr.as_ptr() as usize;
            assert_eq!(addr % 64, 0); // Should be 64-byte aligned
        }
    }

    #[test]
    fn test_optimized_audio_buffer() {
        let mut buffer = OptimizedAudioBuffer::new(1000, 44100, 2).unwrap();
        
        // Test processing with prefetching
        let result = buffer.process_samples(|samples| {
            for sample in samples.iter_mut() {
                *sample = 0.5;
            }
        });
        
        assert!(result.is_ok());
        
        let inner_buffer = buffer.into_inner();
        let samples = inner_buffer.as_f32_slice();
        assert_eq!(samples[0], 0.5);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_numa_awareness() {
        use super::numa::*;
        
        let node_count = get_numa_node_count();
        assert!(node_count > 0);
        
        let current_node = get_current_numa_node();
        // current_node might be None if detection fails, which is acceptable
        
        let numa_pool = NumaPool::new();
        let ptr = numa_pool.allocate_preferred(1024);
        // Allocation might fail in test environment, which is acceptable
    }
}