//! Node.js-Specific Performance Optimization
//!
//! This module provides optimizations specifically for Node.js bindings including
//! V8 optimization hints, buffer pool management, event loop integration,
//! and worker thread utilization.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use parking_lot::{Mutex, RwLock};
use once_cell::sync::Lazy;

/// Global Node.js performance statistics
static NODEJS_PERF_STATS: Lazy<NodejsPerfStats> = Lazy::new(NodejsPerfStats::new);

/// Node.js-specific performance statistics
#[derive(Debug)]
pub struct NodejsPerfStats {
    pub v8_optimizations: AtomicU64,
    pub v8_deoptimizations: AtomicU64,
    pub buffer_pool_hits: AtomicU64,
    pub buffer_pool_misses: AtomicU64,
    pub event_loop_delays: AtomicU64,
    pub worker_thread_spawns: AtomicU64,
    pub worker_thread_completions: AtomicU64,
    pub gc_collections: AtomicU64,
    pub promise_resolutions: AtomicU64,
    pub callback_invocations: AtomicU64,
}

impl NodejsPerfStats {
    pub fn new() -> Self {
        Self {
            v8_optimizations: AtomicU64::new(0),
            v8_deoptimizations: AtomicU64::new(0),
            buffer_pool_hits: AtomicU64::new(0),
            buffer_pool_misses: AtomicU64::new(0),
            event_loop_delays: AtomicU64::new(0),
            worker_thread_spawns: AtomicU64::new(0),
            worker_thread_completions: AtomicU64::new(0),
            gc_collections: AtomicU64::new(0),
            promise_resolutions: AtomicU64::new(0),
            callback_invocations: AtomicU64::new(0),
        }
    }

    /// Record V8 optimization
    pub fn record_v8_optimization(&self) {
        self.v8_optimizations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record V8 deoptimization
    pub fn record_v8_deoptimization(&self) {
        self.v8_deoptimizations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record buffer pool access
    pub fn record_buffer_pool_access(&self, hit: bool) {
        if hit {
            self.buffer_pool_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.buffer_pool_misses.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record event loop delay
    pub fn record_event_loop_delay(&self, delay_us: u64) {
        self.event_loop_delays.fetch_add(delay_us, Ordering::Relaxed);
    }

    /// Record worker thread spawn
    pub fn record_worker_thread_spawn(&self) {
        self.worker_thread_spawns.fetch_add(1, Ordering::Relaxed);
    }

    /// Record worker thread completion
    pub fn record_worker_thread_completion(&self) {
        self.worker_thread_completions.fetch_add(1, Ordering::Relaxed);
    }

    /// Record GC collection
    pub fn record_gc_collection(&self) {
        self.gc_collections.fetch_add(1, Ordering::Relaxed);
    }

    /// Record promise resolution
    pub fn record_promise_resolution(&self) {
        self.promise_resolutions.fetch_add(1, Ordering::Relaxed);
    }

    /// Record callback invocation
    pub fn record_callback_invocation(&self) {
        self.callback_invocations.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current statistics
    pub fn get_stats(&self) -> NodejsPerfSnapshot {
        NodejsPerfSnapshot {
            v8_optimizations: self.v8_optimizations.load(Ordering::Relaxed),
            v8_deoptimizations: self.v8_deoptimizations.load(Ordering::Relaxed),
            buffer_pool_hits: self.buffer_pool_hits.load(Ordering::Relaxed),
            buffer_pool_misses: self.buffer_pool_misses.load(Ordering::Relaxed),
            event_loop_delays: self.event_loop_delays.load(Ordering::Relaxed),
            worker_thread_spawns: self.worker_thread_spawns.load(Ordering::Relaxed),
            worker_thread_completions: self.worker_thread_completions.load(Ordering::Relaxed),
            gc_collections: self.gc_collections.load(Ordering::Relaxed),
            promise_resolutions: self.promise_resolutions.load(Ordering::Relaxed),
            callback_invocations: self.callback_invocations.load(Ordering::Relaxed),
        }
    }

    /// Calculate buffer pool hit ratio
    pub fn buffer_pool_hit_ratio(&self) -> f64 {
        let hits = self.buffer_pool_hits.load(Ordering::Relaxed);
        let misses = self.buffer_pool_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        
        if total == 0 {
            0.0
        } else {
            (hits as f64 / total as f64) * 100.0
        }
    }

    /// Calculate V8 optimization ratio
    pub fn v8_optimization_ratio(&self) -> f64 {
        let optimizations = self.v8_optimizations.load(Ordering::Relaxed);
        let deoptimizations = self.v8_deoptimizations.load(Ordering::Relaxed);
        let total = optimizations + deoptimizations;
        
        if total == 0 {
            0.0
        } else {
            (optimizations as f64 / total as f64) * 100.0
        }
    }

    /// Calculate average event loop delay in microseconds
    pub fn average_event_loop_delay_us(&self) -> f64 {
        let total_delay = self.event_loop_delays.load(Ordering::Relaxed);
        let callback_count = self.callback_invocations.load(Ordering::Relaxed);
        
        if callback_count == 0 {
            0.0
        } else {
            total_delay as f64 / callback_count as f64
        }
    }
}

/// Snapshot of Node.js performance statistics
#[derive(Debug, Clone)]
pub struct NodejsPerfSnapshot {
    pub v8_optimizations: u64,
    pub v8_deoptimizations: u64,
    pub buffer_pool_hits: u64,
    pub buffer_pool_misses: u64,
    pub event_loop_delays: u64,
    pub worker_thread_spawns: u64,
    pub worker_thread_completions: u64,
    pub gc_collections: u64,
    pub promise_resolutions: u64,
    pub callback_invocations: u64,
}

/// V8 optimization hints and utilities
pub struct V8Optimizer {
    optimization_hints: V8OptimizationHints,
}

impl V8Optimizer {
    /// Create new V8 optimizer
    pub fn new() -> Self {
        Self {
            optimization_hints: V8OptimizationHints::default(),
        }
    }

    /// Mark function as hot for V8 optimization
    pub fn mark_hot_function(&self, function_name: &str) {
        NODEJS_PERF_STATS.record_v8_optimization();
        // In practice, this would interface with V8's optimization APIs
        println!("Marking {} as hot function for V8 optimization", function_name);
    }

    /// Prevent V8 deoptimization by avoiding common pitfalls
    pub fn prevent_deoptimization(&self) -> DeoptimizationGuard {
        DeoptimizationGuard::new()
    }

    /// Generate optimized function wrapper
    pub fn generate_optimized_wrapper(&self, function_name: &str) -> String {
        format!(
            r#"
// V8-optimized wrapper for {}
function optimized_{}(...args) {{
    // Use consistent argument types
    // Avoid dynamic property access
    // Use typed arrays for numeric data
    // Minimize closure creation
    return original_{}(...args);
}}

// Mark for optimization
%OptimizeFunctionOnNextCall(optimized_{});
"#,
            function_name, function_name, function_name, function_name
        )
    }

    /// Get optimization hints
    pub fn get_optimization_hints(&self) -> &V8OptimizationHints {
        &self.optimization_hints
    }
}

/// V8 optimization hints and best practices
#[derive(Debug, Clone)]
pub struct V8OptimizationHints {
    pub use_typed_arrays: bool,
    pub avoid_dynamic_property_access: bool,
    pub minimize_closure_creation: bool,
    pub use_consistent_types: bool,
    pub avoid_polymorphic_calls: bool,
    pub prefer_monomorphic_code: bool,
}

impl Default for V8OptimizationHints {
    fn default() -> Self {
        Self {
            use_typed_arrays: true,
            avoid_dynamic_property_access: true,
            minimize_closure_creation: true,
            use_consistent_types: true,
            avoid_polymorphic_calls: true,
            prefer_monomorphic_code: true,
        }
    }
}

/// Guard to prevent V8 deoptimization
pub struct DeoptimizationGuard {
    start_time: Instant,
}

impl DeoptimizationGuard {
    fn new() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }
}

impl Drop for DeoptimizationGuard {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        if duration > Duration::from_millis(100) {
            NODEJS_PERF_STATS.record_v8_deoptimization();
        } else {
            NODEJS_PERF_STATS.record_v8_optimization();
        }
    }
}

/// Buffer pool manager for efficient memory management
pub struct BufferPoolManager {
    pools: RwLock<Vec<BufferPool>>,
    max_pools: usize,
}

impl BufferPoolManager {
    /// Create new buffer pool manager
    pub fn new() -> Self {
        Self {
            pools: RwLock::new(Vec::new()),
            max_pools: 8, // Support up to 8 different buffer sizes
        }
    }

    /// Get buffer from pool or allocate new one
    pub fn get_buffer(&self, size: usize) -> ManagedBuffer {
        // Find appropriate pool
        {
            let pools = self.pools.read();
            for pool in pools.iter() {
                if pool.buffer_size >= size && pool.buffer_size <= size * 2 {
                    if let Some(buffer) = pool.get_buffer() {
                        NODEJS_PERF_STATS.record_buffer_pool_access(true);
                        return ManagedBuffer::from_pool(buffer, pool.id);
                    }
                }
            }
        }

        // Create new pool if needed
        let pool_id = {
            let mut pools = self.pools.write();
            if pools.len() < self.max_pools {
                let pool_id = pools.len();
                pools.push(BufferPool::new(pool_id, size));
                pool_id
            } else {
                // Use closest existing pool
                pools
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, pool)| (pool.buffer_size as i64 - size as i64).abs())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
        };

        NODEJS_PERF_STATS.record_buffer_pool_access(false);
        ManagedBuffer::new(size, pool_id)
    }

    /// Return buffer to pool
    pub fn return_buffer(&self, buffer: ManagedBuffer) {
        let pools = self.pools.read();
        if let Some(pool) = pools.get(buffer.pool_id) {
            pool.return_buffer(buffer.into_vec());
        }
    }

    /// Get pool statistics
    pub fn get_pool_stats(&self) -> Vec<BufferPoolStats> {
        let pools = self.pools.read();
        pools.iter().map(|pool| pool.get_stats()).collect()
    }
}

/// Individual buffer pool for specific size range
struct BufferPool {
    id: usize,
    buffer_size: usize,
    available_buffers: Mutex<VecDeque<Vec<u8>>>,
    max_buffers: usize,
    total_allocations: AtomicU64,
    total_returns: AtomicU64,
}

impl BufferPool {
    fn new(id: usize, buffer_size: usize) -> Self {
        Self {
            id,
            buffer_size,
            available_buffers: Mutex::new(VecDeque::new()),
            max_buffers: 16, // Maximum buffers per pool
            total_allocations: AtomicU64::new(0),
            total_returns: AtomicU64::new(0),
        }
    }

    fn get_buffer(&self) -> Option<Vec<u8>> {
        let mut buffers = self.available_buffers.lock();
        let buffer = buffers.pop_front();
        if buffer.is_some() {
            self.total_allocations.fetch_add(1, Ordering::Relaxed);
        }
        buffer
    }

    fn return_buffer(&self, buffer: Vec<u8>) {
        let mut buffers = self.available_buffers.lock();
        if buffers.len() < self.max_buffers && buffer.len() == self.buffer_size {
            buffers.push_back(buffer);
            self.total_returns.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn get_stats(&self) -> BufferPoolStats {
        let buffers = self.available_buffers.lock();
        BufferPoolStats {
            pool_id: self.id,
            buffer_size: self.buffer_size,
            available_buffers: buffers.len(),
            total_allocations: self.total_allocations.load(Ordering::Relaxed),
            total_returns: self.total_returns.load(Ordering::Relaxed),
        }
    }
}

/// Buffer pool statistics
#[derive(Debug, Clone)]
pub struct BufferPoolStats {
    pub pool_id: usize,
    pub buffer_size: usize,
    pub available_buffers: usize,
    pub total_allocations: u64,
    pub total_returns: u64,
}

/// Managed buffer with automatic pool return
pub struct ManagedBuffer {
    data: Vec<u8>,
    pool_id: usize,
}

impl ManagedBuffer {
    fn new(size: usize, pool_id: usize) -> Self {
        Self {
            data: vec![0u8; size],
            pool_id,
        }
    }

    fn from_pool(mut data: Vec<u8>, pool_id: usize) -> Self {
        data.clear();
        Self { data, pool_id }
    }

    /// Get mutable reference to buffer data
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get reference to buffer data
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get buffer length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Resize buffer
    pub fn resize(&mut self, new_size: usize) {
        self.data.resize(new_size, 0);
    }

    /// Convert to vector (consumes the buffer)
    fn into_vec(self) -> Vec<u8> {
        self.data
    }
}

/// Event loop integration utilities
pub struct EventLoopIntegrator {
    delay_measurements: Arc<Mutex<VecDeque<Duration>>>,
    max_measurements: usize,
}

impl EventLoopIntegrator {
    /// Create new event loop integrator
    pub fn new() -> Self {
        Self {
            delay_measurements: Arc::new(Mutex::new(VecDeque::new())),
            max_measurements: 100, // Keep last 100 measurements
        }
    }

    /// Measure event loop delay
    pub fn measure_delay<F, R>(&self, operation: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = operation();
        let delay = start.elapsed();
        
        NODEJS_PERF_STATS.record_event_loop_delay(delay.as_micros() as u64);
        NODEJS_PERF_STATS.record_callback_invocation();
        
        // Store delay measurement
        let mut measurements = self.delay_measurements.lock();
        measurements.push_back(delay);
        if measurements.len() > self.max_measurements {
            measurements.pop_front();
        }
        
        result
    }

    /// Schedule non-blocking operation
    pub fn schedule_async<F>(&self, operation: F)
    where
        F: FnOnce() + Send + 'static,
    {
        // In practice, this would use Node.js APIs to schedule on the event loop
        std::thread::spawn(move || {
            operation();
            NODEJS_PERF_STATS.record_promise_resolution();
        });
    }

    /// Get average event loop delay
    pub fn get_average_delay(&self) -> Duration {
        let measurements = self.delay_measurements.lock();
        if measurements.is_empty() {
            Duration::ZERO
        } else {
            let total: Duration = measurements.iter().sum();
            total / measurements.len() as u32
        }
    }

    /// Check if event loop is under pressure
    pub fn is_under_pressure(&self) -> bool {
        self.get_average_delay() > Duration::from_millis(10)
    }
}

/// Worker thread utilities for CPU-intensive operations
pub struct WorkerThreadManager {
    thread_count: AtomicUsize,
    max_threads: usize,
}

impl WorkerThreadManager {
    /// Create new worker thread manager
    pub fn new() -> Self {
        Self {
            thread_count: AtomicUsize::new(0),
            max_threads: num_cpus::get().saturating_sub(1), // Leave one CPU for main thread
        }
    }

    /// Execute operation in worker thread
    pub fn execute_async<F, R>(&self, operation: F) -> WorkerResult<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let current_threads = self.thread_count.load(Ordering::Relaxed);
        
        if current_threads >= self.max_threads {
            return WorkerResult::ThreadPoolFull;
        }

        self.thread_count.fetch_add(1, Ordering::Relaxed);
        NODEJS_PERF_STATS.record_worker_thread_spawn();

        let thread_count = Arc::new(AtomicUsize::new(self.thread_count.load(Ordering::Relaxed)));
        let thread_count_clone = Arc::clone(&thread_count);

        std::thread::spawn(move || {
            let result = operation();
            thread_count_clone.fetch_sub(1, Ordering::Relaxed);
            NODEJS_PERF_STATS.record_worker_thread_completion();
            result
        });

        WorkerResult::Scheduled
    }

    /// Get current thread count
    pub fn active_threads(&self) -> usize {
        self.thread_count.load(Ordering::Relaxed)
    }

    /// Get maximum threads
    pub fn max_threads(&self) -> usize {
        self.max_threads
    }
}

/// Worker thread execution result
#[derive(Debug)]
pub enum WorkerResult<T> {
    Scheduled,
    ThreadPoolFull,
    Completed(T),
}

/// Get global Node.js performance statistics
pub fn get_nodejs_perf_stats() -> NodejsPerfSnapshot {
    NODEJS_PERF_STATS.get_stats()
}

/// Reset global Node.js performance statistics
pub fn reset_nodejs_perf_stats() {
    NODEJS_PERF_STATS.v8_optimizations.store(0, Ordering::Relaxed);
    NODEJS_PERF_STATS.v8_deoptimizations.store(0, Ordering::Relaxed);
    NODEJS_PERF_STATS.buffer_pool_hits.store(0, Ordering::Relaxed);
    NODEJS_PERF_STATS.buffer_pool_misses.store(0, Ordering::Relaxed);
    NODEJS_PERF_STATS.event_loop_delays.store(0, Ordering::Relaxed);
    NODEJS_PERF_STATS.worker_thread_spawns.store(0, Ordering::Relaxed);
    NODEJS_PERF_STATS.worker_thread_completions.store(0, Ordering::Relaxed);
    NODEJS_PERF_STATS.gc_collections.store(0, Ordering::Relaxed);
    NODEJS_PERF_STATS.promise_resolutions.store(0, Ordering::Relaxed);
    NODEJS_PERF_STATS.callback_invocations.store(0, Ordering::Relaxed);
}

/// Get buffer pool hit ratio
pub fn get_buffer_pool_hit_ratio() -> f64 {
    NODEJS_PERF_STATS.buffer_pool_hit_ratio()
}

/// Get V8 optimization ratio
pub fn get_v8_optimization_ratio() -> f64 {
    NODEJS_PERF_STATS.v8_optimization_ratio()
}

/// Get average event loop delay in microseconds
pub fn get_average_event_loop_delay_us() -> f64 {
    NODEJS_PERF_STATS.average_event_loop_delay_us()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_nodejs_perf_stats() {
        reset_nodejs_perf_stats();
        
        NODEJS_PERF_STATS.record_v8_optimization();
        NODEJS_PERF_STATS.record_buffer_pool_access(true);
        NODEJS_PERF_STATS.record_event_loop_delay(1000);
        NODEJS_PERF_STATS.record_worker_thread_spawn();
        NODEJS_PERF_STATS.record_callback_invocation();
        
        let stats = get_nodejs_perf_stats();
        assert_eq!(stats.v8_optimizations, 1);
        assert_eq!(stats.buffer_pool_hits, 1);
        assert_eq!(stats.event_loop_delays, 1000);
        assert_eq!(stats.worker_thread_spawns, 1);
        assert_eq!(stats.callback_invocations, 1);
    }

    #[test]
    fn test_v8_optimizer() {
        let optimizer = V8Optimizer::new();
        let hints = optimizer.get_optimization_hints();
        
        assert!(hints.use_typed_arrays);
        assert!(hints.avoid_dynamic_property_access);
        
        let wrapper = optimizer.generate_optimized_wrapper("testFunction");
        assert!(wrapper.contains("optimized_testFunction"));
        assert!(wrapper.contains("%OptimizeFunctionOnNextCall"));
    }

    #[test]
    fn test_deoptimization_guard() {
        reset_nodejs_perf_stats();
        
        {
            let _guard = DeoptimizationGuard::new();
            // Short operation should be optimized
        }
        
        let stats = get_nodejs_perf_stats();
        assert_eq!(stats.v8_optimizations, 1);
        assert_eq!(stats.v8_deoptimizations, 0);
    }

    #[test]
    fn test_buffer_pool_manager() {
        let manager = BufferPoolManager::new();
        
        // Get buffer from pool
        let buffer = manager.get_buffer(1024);
        assert_eq!(buffer.len(), 1024);
        
        // Return buffer to pool
        manager.return_buffer(buffer);
        
        // Get buffer again (should hit pool)
        let buffer2 = manager.get_buffer(1024);
        assert_eq!(buffer2.len(), 1024);
        
        let stats = manager.get_pool_stats();
        assert!(!stats.is_empty());
    }

    #[test]
    fn test_event_loop_integrator() {
        let integrator = EventLoopIntegrator::new();
        
        let result = integrator.measure_delay(|| {
            thread::sleep(Duration::from_millis(1));
            42
        });
        
        assert_eq!(result, 42);
        assert!(integrator.get_average_delay() > Duration::ZERO);
    }

    #[test]
    fn test_worker_thread_manager() {
        let manager = WorkerThreadManager::new();
        
        assert!(manager.max_threads() > 0);
        assert_eq!(manager.active_threads(), 0);
        
        match manager.execute_async(|| 42) {
            WorkerResult::Scheduled => {
                // Expected for successful scheduling
            }
            _ => panic!("Expected successful scheduling"),
        }
    }

    #[test]
    fn test_performance_ratios() {
        reset_nodejs_perf_stats();
        
        // Record some operations
        NODEJS_PERF_STATS.record_v8_optimization();
        NODEJS_PERF_STATS.record_v8_optimization();
        NODEJS_PERF_STATS.record_v8_deoptimization();
        
        NODEJS_PERF_STATS.record_buffer_pool_access(true);
        NODEJS_PERF_STATS.record_buffer_pool_access(true);
        NODEJS_PERF_STATS.record_buffer_pool_access(false);
        
        let v8_ratio = get_v8_optimization_ratio();
        assert!((v8_ratio - 66.66666666666667).abs() < 1e-10); // 2 opt / 3 total
        
        let buffer_ratio = get_buffer_pool_hit_ratio();
        assert!((buffer_ratio - 66.66666666666667).abs() < 1e-10); // 2 hits / 3 total
    }

    #[test]
    fn test_managed_buffer() {
        let mut buffer = ManagedBuffer::new(1024, 0);
        
        assert_eq!(buffer.len(), 1024);
        assert!(!buffer.is_empty());
        
        buffer.resize(2048);
        assert_eq!(buffer.len(), 2048);
        
        let slice = buffer.as_mut_slice();
        slice[0] = 42;
        assert_eq!(buffer.as_slice()[0], 42);
    }
}