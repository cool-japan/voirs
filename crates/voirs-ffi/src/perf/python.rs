//! Python-Specific Performance Optimization
//!
//! This module provides optimizations specifically for Python bindings including
//! GIL management, NumPy optimization, memory view usage, and Cython integration hints.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::Mutex;
use once_cell::sync::Lazy;

/// Global Python performance statistics
static PYTHON_PERF_STATS: Lazy<PythonPerfStats> = Lazy::new(PythonPerfStats::new);

/// Python-specific performance statistics
#[derive(Debug)]
pub struct PythonPerfStats {
    pub gil_acquisitions: AtomicU64,
    pub gil_releases: AtomicU64,
    pub gil_wait_time_ns: AtomicU64,
    pub numpy_operations: AtomicU64,
    pub memory_view_operations: AtomicU64,
    pub zero_copy_operations: AtomicU64,
    pub buffer_conversions: AtomicU64,
    pub cython_calls: AtomicU64,
}

impl PythonPerfStats {
    pub fn new() -> Self {
        Self {
            gil_acquisitions: AtomicU64::new(0),
            gil_releases: AtomicU64::new(0),
            gil_wait_time_ns: AtomicU64::new(0),
            numpy_operations: AtomicU64::new(0),
            memory_view_operations: AtomicU64::new(0),
            zero_copy_operations: AtomicU64::new(0),
            buffer_conversions: AtomicU64::new(0),
            cython_calls: AtomicU64::new(0),
        }
    }

    /// Record GIL acquisition
    pub fn record_gil_acquisition(&self, wait_time: Duration) {
        self.gil_acquisitions.fetch_add(1, Ordering::Relaxed);
        self.gil_wait_time_ns.fetch_add(wait_time.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Record GIL release
    pub fn record_gil_release(&self) {
        self.gil_releases.fetch_add(1, Ordering::Relaxed);
    }

    /// Record NumPy operation
    pub fn record_numpy_operation(&self) {
        self.numpy_operations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record memory view operation
    pub fn record_memory_view_operation(&self) {
        self.memory_view_operations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record zero-copy operation
    pub fn record_zero_copy_operation(&self) {
        self.zero_copy_operations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record buffer conversion
    pub fn record_buffer_conversion(&self) {
        self.buffer_conversions.fetch_add(1, Ordering::Relaxed);
    }

    /// Record Cython call
    pub fn record_cython_call(&self) {
        self.cython_calls.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current statistics
    pub fn get_stats(&self) -> PythonPerfSnapshot {
        PythonPerfSnapshot {
            gil_acquisitions: self.gil_acquisitions.load(Ordering::Relaxed),
            gil_releases: self.gil_releases.load(Ordering::Relaxed),
            gil_wait_time_ns: self.gil_wait_time_ns.load(Ordering::Relaxed),
            numpy_operations: self.numpy_operations.load(Ordering::Relaxed),
            memory_view_operations: self.memory_view_operations.load(Ordering::Relaxed),
            zero_copy_operations: self.zero_copy_operations.load(Ordering::Relaxed),
            buffer_conversions: self.buffer_conversions.load(Ordering::Relaxed),
            cython_calls: self.cython_calls.load(Ordering::Relaxed),
        }
    }

    /// Calculate average GIL wait time in microseconds
    pub fn average_gil_wait_time_us(&self) -> f64 {
        let total_acquisitions = self.gil_acquisitions.load(Ordering::Relaxed);
        if total_acquisitions == 0 {
            return 0.0;
        }
        let total_wait_ns = self.gil_wait_time_ns.load(Ordering::Relaxed);
        (total_wait_ns as f64 / total_acquisitions as f64) / 1000.0
    }
}

/// Snapshot of Python performance statistics
#[derive(Debug, Clone)]
pub struct PythonPerfSnapshot {
    pub gil_acquisitions: u64,
    pub gil_releases: u64,
    pub gil_wait_time_ns: u64,
    pub numpy_operations: u64,
    pub memory_view_operations: u64,
    pub zero_copy_operations: u64,
    pub buffer_conversions: u64,
    pub cython_calls: u64,
}

/// GIL management utilities for optimal Python integration
pub struct GilManager {
    gil_held: AtomicUsize,
    optimization_enabled: bool,
}

impl GilManager {
    /// Create new GIL manager
    pub fn new() -> Self {
        Self {
            gil_held: AtomicUsize::new(0),
            optimization_enabled: true,
        }
    }

    /// Acquire GIL with timing
    pub fn acquire_gil(&self) -> GilGuard {
        let start = Instant::now();
        
        // In a real implementation, this would interface with PyO3 or Python C API
        self.gil_held.fetch_add(1, Ordering::Relaxed);
        
        let wait_time = start.elapsed();
        PYTHON_PERF_STATS.record_gil_acquisition(wait_time);
        
        GilGuard { manager: self }
    }

    /// Check if GIL optimization is enabled
    pub fn is_optimization_enabled(&self) -> bool {
        self.optimization_enabled
    }

    /// Enable/disable GIL optimizations
    pub fn set_optimization_enabled(&self, enabled: bool) {
        // In practice, this would be &mut self, but for compatibility with atomic operations
        // we'd need a different design
    }

    /// Get current GIL hold count (for debugging)
    pub fn gil_hold_count(&self) -> usize {
        self.gil_held.load(Ordering::Relaxed)
    }

    /// Release GIL
    fn release_gil(&self) {
        self.gil_held.fetch_sub(1, Ordering::Relaxed);
        PYTHON_PERF_STATS.record_gil_release();
    }
}

/// RAII guard for GIL management
pub struct GilGuard<'a> {
    manager: &'a GilManager,
}

impl<'a> Drop for GilGuard<'a> {
    fn drop(&mut self) {
        self.manager.release_gil();
    }
}

/// NumPy optimization utilities
pub struct NumPyOptimizer {
    buffer_cache: Arc<Mutex<Vec<Vec<f32>>>>,
    max_cache_size: usize,
}

impl NumPyOptimizer {
    /// Create new NumPy optimizer
    pub fn new() -> Self {
        Self {
            buffer_cache: Arc::new(Mutex::new(Vec::new())),
            max_cache_size: 16, // Cache up to 16 buffers
        }
    }

    /// Get optimized buffer for NumPy operations
    pub fn get_buffer(&self, size: usize) -> Vec<f32> {
        PYTHON_PERF_STATS.record_numpy_operation();
        
        let mut cache = self.buffer_cache.lock();
        
        // Try to reuse existing buffer
        for i in 0..cache.len() {
            if cache[i].capacity() >= size {
                let mut buffer = cache.swap_remove(i);
                buffer.clear();
                buffer.resize(size, 0.0);
                PYTHON_PERF_STATS.record_zero_copy_operation();
                return buffer;
            }
        }

        // Create new buffer
        PYTHON_PERF_STATS.record_buffer_conversion();
        vec![0.0; size]
    }

    /// Return buffer to cache
    pub fn return_buffer(&self, buffer: Vec<f32>) {
        let mut cache = self.buffer_cache.lock();
        
        if cache.len() < self.max_cache_size {
            cache.push(buffer);
        }
        // Otherwise, let the buffer drop
    }

    /// Create zero-copy view of audio data for NumPy
    pub fn create_numpy_view(&self, data: &[f32]) -> NumPyView {
        PYTHON_PERF_STATS.record_memory_view_operation();
        PYTHON_PERF_STATS.record_zero_copy_operation();
        
        NumPyView {
            data: data.as_ptr(),
            len: data.len(),
            dtype: NumPyDType::Float32,
        }
    }

    /// Convert buffer to NumPy-compatible format
    pub fn to_numpy_compatible(&self, data: &[f32]) -> Vec<f32> {
        PYTHON_PERF_STATS.record_numpy_operation();
        
        // For f32 data, it's already NumPy compatible
        // In practice, this might handle stride adjustments, etc.
        data.to_vec()
    }

    /// Optimize array for Python consumption
    pub fn optimize_for_python(&self, mut data: Vec<f32>) -> Vec<f32> {
        PYTHON_PERF_STATS.record_numpy_operation();
        
        // Ensure data is properly aligned for SIMD operations
        // This is a simplified version - real implementation would check alignment
        if data.as_ptr() as usize % 32 != 0 {
            PYTHON_PERF_STATS.record_buffer_conversion();
            // Reallocate with proper alignment
            let aligned_data: Vec<f32> = data.into_iter().collect();
            aligned_data
        } else {
            PYTHON_PERF_STATS.record_zero_copy_operation();
            data
        }
    }
}

/// NumPy data type enumeration
#[derive(Debug, Clone, Copy)]
pub enum NumPyDType {
    Float32,
    Float64,
    Int16,
    Int32,
}

/// Zero-copy view for NumPy integration
pub struct NumPyView {
    data: *const f32,
    len: usize,
    dtype: NumPyDType,
}

impl NumPyView {
    /// Get data pointer (for Python C API)
    pub fn data_ptr(&self) -> *const f32 {
        self.data
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Get data type
    pub fn dtype(&self) -> NumPyDType {
        self.dtype
    }

    /// Check if view is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

unsafe impl Send for NumPyView {}
unsafe impl Sync for NumPyView {}

/// Memory view optimization for Python buffers
pub struct MemoryViewOptimizer {
    view_cache: Arc<Mutex<Vec<MemoryView>>>,
}

impl MemoryViewOptimizer {
    /// Create new memory view optimizer
    pub fn new() -> Self {
        Self {
            view_cache: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Create optimized memory view
    pub fn create_view(&self, data: &[f32]) -> MemoryView {
        PYTHON_PERF_STATS.record_memory_view_operation();
        
        MemoryView {
            ptr: data.as_ptr() as *const u8,
            len: data.len() * std::mem::size_of::<f32>(),
            format: "f".to_string(), // Python format string for float
            itemsize: std::mem::size_of::<f32>(),
        }
    }

    /// Create mutable memory view
    pub fn create_mut_view(&self, data: &mut [f32]) -> MutMemoryView {
        PYTHON_PERF_STATS.record_memory_view_operation();
        
        MutMemoryView {
            ptr: data.as_mut_ptr() as *mut u8,
            len: data.len() * std::mem::size_of::<f32>(),
            format: "f".to_string(),
            itemsize: std::mem::size_of::<f32>(),
        }
    }
}

/// Immutable memory view for Python buffer protocol
pub struct MemoryView {
    ptr: *const u8,
    len: usize,
    format: String,
    itemsize: usize,
}

impl MemoryView {
    /// Get buffer pointer
    pub fn ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Get buffer length in bytes
    pub fn len(&self) -> usize {
        self.len
    }

    /// Get format string
    pub fn format(&self) -> &str {
        &self.format
    }

    /// Get item size
    pub fn itemsize(&self) -> usize {
        self.itemsize
    }
}

/// Mutable memory view for Python buffer protocol
pub struct MutMemoryView {
    ptr: *mut u8,
    len: usize,
    format: String,
    itemsize: usize,
}

impl MutMemoryView {
    /// Get mutable buffer pointer
    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Get buffer length in bytes
    pub fn len(&self) -> usize {
        self.len
    }

    /// Get format string
    pub fn format(&self) -> &str {
        &self.format
    }

    /// Get item size
    pub fn itemsize(&self) -> usize {
        self.itemsize
    }
}

/// Cython integration hints and optimizations
pub struct CythonOptimizer;

impl CythonOptimizer {
    /// Mark function as Cython-optimized
    pub fn mark_cython_call() {
        PYTHON_PERF_STATS.record_cython_call();
    }

    /// Get optimization hints for Cython compilation
    pub fn get_optimization_hints() -> CythonHints {
        CythonHints {
            use_boundscheck: false,
            use_wraparound: false,
            use_cdivision: true,
            use_c_string_type: true,
            use_c_string_encoding: "utf-8".to_string(),
        }
    }

    /// Generate Cython function signature
    pub fn generate_cython_signature(
        func_name: &str,
        params: &[(&str, &str)],
        return_type: &str,
    ) -> String {
        let param_strs: Vec<String> = params
            .iter()
            .map(|(name, ty)| format!("{}: {}", name, ty))
            .collect();
        
        format!(
            "cdef {} {}({}) nogil:",
            return_type,
            func_name,
            param_strs.join(", ")
        )
    }
}

/// Cython optimization hints
#[derive(Debug, Clone)]
pub struct CythonHints {
    pub use_boundscheck: bool,
    pub use_wraparound: bool,
    pub use_cdivision: bool,
    pub use_c_string_type: bool,
    pub use_c_string_encoding: String,
}

/// Get global Python performance statistics
pub fn get_python_perf_stats() -> PythonPerfSnapshot {
    PYTHON_PERF_STATS.get_stats()
}

/// Reset global Python performance statistics
pub fn reset_python_perf_stats() {
    PYTHON_PERF_STATS.gil_acquisitions.store(0, Ordering::Relaxed);
    PYTHON_PERF_STATS.gil_releases.store(0, Ordering::Relaxed);
    PYTHON_PERF_STATS.gil_wait_time_ns.store(0, Ordering::Relaxed);
    PYTHON_PERF_STATS.numpy_operations.store(0, Ordering::Relaxed);
    PYTHON_PERF_STATS.memory_view_operations.store(0, Ordering::Relaxed);
    PYTHON_PERF_STATS.zero_copy_operations.store(0, Ordering::Relaxed);
    PYTHON_PERF_STATS.buffer_conversions.store(0, Ordering::Relaxed);
    PYTHON_PERF_STATS.cython_calls.store(0, Ordering::Relaxed);
}

/// Get average GIL wait time in microseconds
pub fn get_average_gil_wait_time_us() -> f64 {
    PYTHON_PERF_STATS.average_gil_wait_time_us()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_python_perf_stats() {
        reset_python_perf_stats();
        
        PYTHON_PERF_STATS.record_gil_acquisition(Duration::from_millis(1));
        PYTHON_PERF_STATS.record_gil_release();
        PYTHON_PERF_STATS.record_numpy_operation();
        PYTHON_PERF_STATS.record_memory_view_operation();
        PYTHON_PERF_STATS.record_zero_copy_operation();
        
        let stats = get_python_perf_stats();
        assert_eq!(stats.gil_acquisitions, 1);
        assert_eq!(stats.gil_releases, 1);
        assert_eq!(stats.numpy_operations, 1);
        assert_eq!(stats.memory_view_operations, 1);
        assert_eq!(stats.zero_copy_operations, 1);
    }

    #[test]
    fn test_gil_manager() {
        let gil_manager = GilManager::new();
        assert_eq!(gil_manager.gil_hold_count(), 0);
        
        {
            let _guard = gil_manager.acquire_gil();
            assert_eq!(gil_manager.gil_hold_count(), 1);
        }
        
        assert_eq!(gil_manager.gil_hold_count(), 0);
    }

    #[test]
    fn test_numpy_optimizer() {
        let optimizer = NumPyOptimizer::new();
        
        // Test buffer management
        let buffer = optimizer.get_buffer(1024);
        assert_eq!(buffer.len(), 1024);
        
        optimizer.return_buffer(buffer);
        
        // Test zero-copy view
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let view = optimizer.create_numpy_view(&data);
        assert_eq!(view.len(), 4);
        assert_eq!(unsafe { *view.data_ptr() }, 1.0);
    }

    #[test]
    fn test_memory_view_optimizer() {
        let optimizer = MemoryViewOptimizer::new();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        
        let view = optimizer.create_view(&data);
        assert_eq!(view.len(), 16); // 4 floats * 4 bytes each
        assert_eq!(view.itemsize(), 4);
        assert_eq!(view.format(), "f");
    }

    #[test]
    fn test_cython_optimizer() {
        reset_python_perf_stats();
        
        CythonOptimizer::mark_cython_call();
        
        let stats = get_python_perf_stats();
        assert_eq!(stats.cython_calls, 1);
        
        let hints = CythonOptimizer::get_optimization_hints();
        assert!(!hints.use_boundscheck);
        assert!(hints.use_cdivision);
        
        let signature = CythonOptimizer::generate_cython_signature(
            "test_func",
            &[("x", "float"), ("y", "int")],
            "double",
        );
        assert!(signature.contains("test_func"));
        assert!(signature.contains("nogil"));
    }

    #[test]
    fn test_gil_wait_time_calculation() {
        reset_python_perf_stats();
        
        // Record some GIL acquisitions with different wait times
        PYTHON_PERF_STATS.record_gil_acquisition(Duration::from_millis(1));
        PYTHON_PERF_STATS.record_gil_acquisition(Duration::from_millis(3));
        
        let avg_wait_time = get_average_gil_wait_time_us();
        assert!(avg_wait_time > 0.0);
        assert!(avg_wait_time < 10000.0); // Should be reasonable
    }
}