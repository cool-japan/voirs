//! Cache optimization module for improved data locality and memory access patterns
//!
//! Provides optimizations for:
//! - Data locality optimization
//! - Cache-friendly algorithms
//! - Prefetching strategies
//! - Memory access patterns
//! - Cache-aware data structures

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

pub mod algorithms;
pub mod features;
pub mod locality;
pub mod patterns;
pub mod prefetch;

/// Cache line size on most modern processors
pub const CACHE_LINE_SIZE: usize = 64;

/// Simple error type for cache operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheError;

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cache allocation error")
    }
}

impl std::error::Error for CacheError {}

/// Cache-aligned buffer for optimal memory access
#[derive(Debug)]
pub struct CacheAlignedBuffer<T> {
    ptr: NonNull<T>,
    layout: Layout,
    len: usize,
}

impl<T> CacheAlignedBuffer<T> {
    /// Create a new cache-aligned buffer
    pub fn new(len: usize) -> Result<Self, CacheError> {
        if len == 0 {
            return Err(CacheError);
        }

        let layout = Layout::from_size_align(len * std::mem::size_of::<T>(), CACHE_LINE_SIZE)
            .map_err(|_| CacheError)?;

        let ptr = unsafe {
            let raw_ptr = alloc(layout);
            NonNull::new(raw_ptr).ok_or(CacheError)?.cast::<T>()
        };

        Ok(Self { ptr, layout, len })
    }

    /// Get a pointer to the buffer
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Get a mutable pointer to the buffer
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Get buffer length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get as slice
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> Drop for CacheAlignedBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

unsafe impl<T: Send> Send for CacheAlignedBuffer<T> {}
unsafe impl<T: Sync> Sync for CacheAlignedBuffer<T> {}

/// Cache-optimized matrix structure for efficient data access
#[derive(Debug)]
pub struct CacheMatrix<T> {
    data: CacheAlignedBuffer<T>,
    rows: usize,
    cols: usize,
    row_stride: usize, // Padded for cache alignment
}

impl<T: Default + Copy> CacheMatrix<T> {
    /// Create a new cache-optimized matrix
    pub fn new(rows: usize, cols: usize) -> Result<Self, CacheError> {
        // Pad row stride to cache line boundary
        let element_size = std::mem::size_of::<T>();
        let elements_per_cache_line = CACHE_LINE_SIZE / element_size;
        let row_stride = ((cols + elements_per_cache_line - 1) / elements_per_cache_line)
            * elements_per_cache_line;

        let total_elements = rows * row_stride;
        let mut data = CacheAlignedBuffer::new(total_elements)?;

        // Initialize with default values using efficient bulk operations
        let slice = data.as_mut_slice();
        // Use vectorized fill operation which is more efficient than manual iteration
        slice.fill(T::default());

        Ok(Self {
            data,
            rows,
            cols,
            row_stride,
        })
    }

    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.rows && col < self.cols {
            let index = row * self.row_stride + col;
            self.data.as_slice().get(index)
        } else {
            None
        }
    }

    /// Get mutable element at (row, col)
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.rows && col < self.cols {
            let index = row * self.row_stride + col;
            self.data.as_mut_slice().get_mut(index)
        } else {
            None
        }
    }

    /// Get row slice
    pub fn row(&self, row: usize) -> Option<&[T]> {
        if row < self.rows {
            let start = row * self.row_stride;
            let end = start + self.cols;
            Some(&self.data.as_slice()[start..end])
        } else {
            None
        }
    }

    /// Get mutable row slice
    pub fn row_mut(&mut self, row: usize) -> Option<&mut [T]> {
        if row < self.rows {
            let start = row * self.row_stride;
            let end = start + self.cols;
            Some(&mut self.data.as_mut_slice()[start..end])
        } else {
            None
        }
    }

    /// Get dimensions
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get row stride (including padding)
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }
}

/// Cache configuration and optimization settings
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// L1 cache size (bytes)
    pub l1_cache_size: usize,

    /// L2 cache size (bytes)
    pub l2_cache_size: usize,

    /// L3 cache size (bytes)
    pub l3_cache_size: usize,

    /// Cache line size (bytes)
    pub cache_line_size: usize,

    /// Prefetch distance (cache lines ahead)
    pub prefetch_distance: usize,

    /// Enable aggressive prefetching
    pub aggressive_prefetch: bool,

    /// Enable data blocking optimizations
    pub enable_blocking: bool,

    /// Block size for tiled algorithms
    pub block_size: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_cache_size: 32 * 1024,       // 32KB L1 cache
            l2_cache_size: 256 * 1024,      // 256KB L2 cache
            l3_cache_size: 8 * 1024 * 1024, // 8MB L3 cache
            cache_line_size: CACHE_LINE_SIZE,
            prefetch_distance: 4,
            aggressive_prefetch: false,
            enable_blocking: true,
            block_size: 64,
        }
    }
}

/// Cache optimizer for audio processing operations
pub struct CacheOptimizer {
    config: CacheConfig,
}

impl CacheOptimizer {
    /// Create new cache optimizer
    pub fn new(config: CacheConfig) -> Self {
        Self { config }
    }

    /// Calculate optimal block size for given data size
    pub fn optimal_block_size(&self, _data_size: usize) -> usize {
        // Try to fit blocks in L1 cache
        let max_block_elements = self.config.l1_cache_size / 4; // Assume f32
        let suggested_block = (max_block_elements as f32).sqrt() as usize;

        // Ensure it's a reasonable size and cache-aligned
        let min_block = 16;
        let max_block = 512;

        suggested_block
            .clamp(min_block, max_block)
            .next_power_of_two()
    }

    /// Get cache configuration
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Prefetch memory location (simplified for compatibility)
    pub fn prefetch(&self, _ptr: *const u8, _locality: u8) {
        // Simplified no-op implementation for stable Rust
        // In a real implementation, this would use target-specific intrinsics
    }

    /// Memory prefetch for sequential access pattern
    pub fn prefetch_sequential<T>(&self, data: &[T], start_idx: usize) {
        if self.config.aggressive_prefetch && start_idx + self.config.prefetch_distance < data.len()
        {
            let prefetch_idx = start_idx + self.config.prefetch_distance;
            let ptr = &data[prefetch_idx] as *const T as *const u8;
            self.prefetch(ptr, 0); // NTA (non-temporal access)
        }
    }

    /// Memory prefetch for strided access pattern
    pub fn prefetch_strided<T>(&self, data: &[T], start_idx: usize, stride: usize) {
        if self.config.aggressive_prefetch {
            for i in 1..=self.config.prefetch_distance {
                let prefetch_idx = start_idx + i * stride;
                if prefetch_idx < data.len() {
                    let ptr = &data[prefetch_idx] as *const T as *const u8;
                    self.prefetch(ptr, 1); // T2 locality
                }
            }
        }
    }
}

impl Default for CacheOptimizer {
    fn default() -> Self {
        Self::new(CacheConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_aligned_buffer() {
        let mut buffer = CacheAlignedBuffer::<f32>::new(1024).unwrap();
        assert_eq!(buffer.len(), 1024);

        // Test alignment
        let ptr = buffer.as_ptr() as usize;
        assert_eq!(ptr % CACHE_LINE_SIZE, 0);

        // Test basic operations
        let slice = buffer.as_mut_slice();
        slice[0] = 1.0;
        slice[1023] = 2.0;

        assert_eq!(slice[0], 1.0);
        assert_eq!(slice[1023], 2.0);
    }

    #[test]
    fn test_cache_matrix() {
        let mut matrix = CacheMatrix::<f32>::new(10, 8).unwrap();
        assert_eq!(matrix.shape(), (10, 8));

        // Test element access
        *matrix.get_mut(5, 3).unwrap() = 42.0;
        assert_eq!(*matrix.get(5, 3).unwrap(), 42.0);

        // Test row access
        let row = matrix.row_mut(5).unwrap();
        row[3] = 100.0;
        assert_eq!(*matrix.get(5, 3).unwrap(), 100.0);
    }

    #[test]
    fn test_cache_optimizer() {
        let optimizer = CacheOptimizer::default();

        // Test block size calculation
        let block_size = optimizer.optimal_block_size(1000000);
        assert!(block_size >= 16);
        assert!(block_size <= 512);
        assert!(block_size.is_power_of_two());
    }

    #[test]
    fn test_cache_config() {
        let config = CacheConfig::default();
        assert_eq!(config.cache_line_size, CACHE_LINE_SIZE);
        assert!(config.l1_cache_size > 0);
        assert!(config.l2_cache_size > config.l1_cache_size);
        assert!(config.l3_cache_size > config.l2_cache_size);
    }
}
