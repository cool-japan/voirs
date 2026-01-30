//! Batch Operation Utilities for Efficient FFI Processing
//!
//! This module provides optimized batch operations for common FFI scenarios,
//! reducing overhead and improving throughput for multi-operation workflows.

use crate::{VoirsAudioBuffer, VoirsErrorCode};
use std::os::raw::{c_char, c_uint};
use std::ptr;

/// Batch text synthesis configuration
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VoirsBatchConfig {
    /// Voice ID to use for all synthesis operations
    pub voice_id: *const c_char,
    /// Sample rate for audio output
    pub sample_rate: c_uint,
    /// Number of parallel workers
    pub num_workers: c_uint,
    /// Enable caching for repeated texts
    pub enable_cache: bool,
    /// Maximum batch size before auto-flush
    pub max_batch_size: c_uint,
}

impl Default for VoirsBatchConfig {
    fn default() -> Self {
        Self {
            voice_id: ptr::null(),
            sample_rate: 22050,
            num_workers: 4,
            enable_cache: true,
            max_batch_size: 100,
        }
    }
}

/// Batch operation statistics
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VoirsBatchStats {
    /// Total number of items processed
    pub items_processed: u64,
    /// Number of successful operations
    pub items_succeeded: u64,
    /// Number of failed operations
    pub items_failed: u64,
    /// Total processing time in milliseconds
    pub total_time_ms: u64,
    /// Average time per item in milliseconds
    pub avg_time_per_item_ms: u64,
    /// Cache hit count (if caching enabled)
    pub cache_hits: u64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
}

/// Batch operation for efficient string operations
pub struct BatchStringProcessor {
    buffer: Vec<String>,
    max_size: usize,
}

impl BatchStringProcessor {
    /// Create a new batch string processor
    pub fn new(max_size: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(max_size),
            max_size,
        }
    }

    /// Add a string to the batch
    pub fn add(&mut self, s: String) -> bool {
        if self.buffer.len() >= self.max_size {
            return false;
        }
        self.buffer.push(s);
        true
    }

    /// Check if batch is full
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.max_size
    }

    /// Get current batch size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Process all batched items with a provided function
    pub fn process<F, R>(&mut self, mut func: F) -> Vec<R>
    where
        F: FnMut(&str) -> R,
    {
        let results: Vec<R> = self.buffer.iter().map(|s| func(s.as_str())).collect();
        self.buffer.clear();
        results
    }

    /// Clear the batch without processing
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

/// Efficient batch memory allocator for FFI buffers
pub struct BatchMemoryAllocator {
    pool: Vec<Vec<u8>>,
    chunk_size: usize,
}

impl BatchMemoryAllocator {
    /// Create a new batch memory allocator
    pub fn new(chunk_size: usize, initial_pool_size: usize) -> Self {
        let pool = (0..initial_pool_size)
            .map(|_| Vec::with_capacity(chunk_size))
            .collect();

        Self { pool, chunk_size }
    }

    /// Acquire a buffer from the pool
    pub fn acquire(&mut self) -> Vec<u8> {
        self.pool
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(self.chunk_size))
    }

    /// Return a buffer to the pool
    pub fn release(&mut self, mut buffer: Vec<u8>) {
        buffer.clear();
        if buffer.capacity() == self.chunk_size {
            self.pool.push(buffer);
        }
    }

    /// Get current pool size
    pub fn pool_size(&self) -> usize {
        self.pool.len()
    }

    /// Preallocate additional buffers
    pub fn preallocate(&mut self, count: usize) {
        for _ in 0..count {
            self.pool.push(Vec::with_capacity(self.chunk_size));
        }
    }
}

/// C API: Create a batch configuration with default values
#[no_mangle]
pub extern "C" fn voirs_batch_config_create() -> *mut VoirsBatchConfig {
    Box::into_raw(Box::new(VoirsBatchConfig::default()))
}

/// C API: Free a batch configuration
///
/// # Safety
/// The config pointer must be valid and not used after this call
#[no_mangle]
pub unsafe extern "C" fn voirs_batch_config_free(config: *mut VoirsBatchConfig) {
    if !config.is_null() {
        drop(Box::from_raw(config));
    }
}

/// C API: Set voice ID for batch operations
///
/// # Safety
/// Both config and voice_id pointers must be valid
#[no_mangle]
pub unsafe extern "C" fn voirs_batch_config_set_voice(
    config: *mut VoirsBatchConfig,
    voice_id: *const c_char,
) -> VoirsErrorCode {
    if config.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    (*config).voice_id = voice_id;
    VoirsErrorCode::Success
}

/// C API: Set sample rate for batch operations
///
/// # Safety
/// The config pointer must be valid
#[no_mangle]
pub unsafe extern "C" fn voirs_batch_config_set_sample_rate(
    config: *mut VoirsBatchConfig,
    sample_rate: c_uint,
) -> VoirsErrorCode {
    if config.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    (*config).sample_rate = sample_rate;
    VoirsErrorCode::Success
}

/// C API: Set number of parallel workers
///
/// # Safety
/// The config pointer must be valid
#[no_mangle]
pub unsafe extern "C" fn voirs_batch_config_set_workers(
    config: *mut VoirsBatchConfig,
    num_workers: c_uint,
) -> VoirsErrorCode {
    if config.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    (*config).num_workers = num_workers;
    VoirsErrorCode::Success
}

/// C API: Enable or disable caching
///
/// # Safety
/// The config pointer must be valid
#[no_mangle]
pub unsafe extern "C" fn voirs_batch_config_set_cache(
    config: *mut VoirsBatchConfig,
    enable: bool,
) -> VoirsErrorCode {
    if config.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    (*config).enable_cache = enable;
    VoirsErrorCode::Success
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_string_processor() {
        let mut processor = BatchStringProcessor::new(5);

        assert!(processor.is_empty());
        assert!(!processor.is_full());

        for i in 0..5 {
            assert!(processor.add(format!("text{}", i)));
        }

        assert!(processor.is_full());
        assert!(!processor.add("overflow".to_string()));

        let results = processor.process(|s| s.len());
        assert_eq!(results.len(), 5);
        assert!(processor.is_empty());
    }

    #[test]
    fn test_batch_memory_allocator() {
        let mut allocator = BatchMemoryAllocator::new(1024, 10);
        assert_eq!(allocator.pool_size(), 10);

        let buffer1 = allocator.acquire();
        assert_eq!(allocator.pool_size(), 9);

        allocator.release(buffer1);
        assert_eq!(allocator.pool_size(), 10);

        allocator.preallocate(5);
        assert_eq!(allocator.pool_size(), 15);
    }

    #[test]
    fn test_batch_config_creation() {
        let config = VoirsBatchConfig::default();
        assert_eq!(config.sample_rate, 22050);
        assert_eq!(config.num_workers, 4);
        assert!(config.enable_cache);
        assert_eq!(config.max_batch_size, 100);
    }

    #[test]
    fn test_batch_config_c_api() {
        unsafe {
            let config = voirs_batch_config_create();
            assert!(!config.is_null());

            let result = voirs_batch_config_set_sample_rate(config, 44100);
            assert_eq!(result, VoirsErrorCode::Success);
            assert_eq!((*config).sample_rate, 44100);

            let result = voirs_batch_config_set_workers(config, 8);
            assert_eq!(result, VoirsErrorCode::Success);
            assert_eq!((*config).num_workers, 8);

            let result = voirs_batch_config_set_cache(config, false);
            assert_eq!(result, VoirsErrorCode::Success);
            assert!(!(*config).enable_cache);

            voirs_batch_config_free(config);
        }
    }

    #[test]
    fn test_batch_stats_default() {
        let stats = VoirsBatchStats::default();
        assert_eq!(stats.items_processed, 0);
        assert_eq!(stats.items_succeeded, 0);
        assert_eq!(stats.items_failed, 0);
        assert_eq!(stats.total_time_ms, 0);
        assert_eq!(stats.avg_time_per_item_ms, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.peak_memory_bytes, 0);
    }
}
