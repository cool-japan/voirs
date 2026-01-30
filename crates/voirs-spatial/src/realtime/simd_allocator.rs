//! SIMD-Aligned Memory Allocator
//!
//! Provides memory allocation with SIMD alignment guarantees for optimal
//! vectorized processing performance. Ensures audio buffers are properly
//! aligned for AVX2, AVX-512, and NEON SIMD instructions.

#![allow(clippy::undocumented_unsafe_blocks)] // Safety comments provided inline
#![allow(unsafe_code)] // Required for custom memory allocation

use std::alloc::{alloc, dealloc, Layout};
use std::mem;
use std::ptr::NonNull;
use std::slice;

/// SIMD alignment requirement (64 bytes for AVX-512)
pub const SIMD_ALIGNMENT: usize = 64;

/// SIMD-aligned audio buffer
///
/// This buffer is guaranteed to be aligned to 64-byte boundaries, making it
/// optimal for SIMD operations. The alignment prevents unaligned access
/// penalties and enables use of aligned SIMD load/store instructions.
///
/// # Example
///
/// ```rust
/// use voirs_spatial::realtime::{SimdAlignedBuffer, SIMD_ALIGNMENT};
///
/// let buffer = SimdAlignedBuffer::<f32>::new(1024);
/// assert_eq!(buffer.as_ptr() as usize % SIMD_ALIGNMENT, 0);
/// ```
pub struct SimdAlignedBuffer<T> {
    /// Pointer to aligned memory
    ptr: NonNull<T>,

    /// Number of elements
    len: usize,

    /// Allocation layout for deallocation
    layout: Layout,
}

impl<T> SimdAlignedBuffer<T> {
    /// Create new SIMD-aligned buffer
    ///
    /// # Arguments
    ///
    /// * `len` - Number of elements
    ///
    /// # Panics
    ///
    /// Panics if allocation fails
    pub fn new(len: usize) -> Self {
        assert!(len > 0, "Buffer length must be > 0");

        let size = len * mem::size_of::<T>();
        let layout = Layout::from_size_align(size, SIMD_ALIGNMENT).expect("Invalid layout");

        // Safety: We allocate memory with proper size and alignment
        // NonNull::new_unchecked is safe because we check for null and abort
        unsafe {
            let ptr = alloc(layout) as *mut T;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }

            Self {
                ptr: NonNull::new_unchecked(ptr),
                len,
                layout,
            }
        }
    }

    /// Create new zero-initialized buffer
    pub fn zeroed(len: usize) -> Self
    where
        T: Copy,
    {
        let mut buffer = Self::new(len);
        // Safety: ptr is valid for `len` elements and properly aligned
        // write_bytes with 0 is safe for Copy types
        unsafe {
            std::ptr::write_bytes(buffer.ptr.as_ptr(), 0, len);
        }
        buffer
    }

    /// Get buffer length
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get raw pointer
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Get mutable raw pointer
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Get as slice
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // Safety: ptr is valid for `len` elements
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get as mutable slice
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // Safety: ptr is valid for `len` elements and we have exclusive access
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Verify alignment
    #[inline]
    pub fn is_aligned(&self) -> bool {
        (self.ptr.as_ptr() as usize).is_multiple_of(SIMD_ALIGNMENT)
    }
}

impl<T> Drop for SimdAlignedBuffer<T> {
    fn drop(&mut self) {
        // Safety: ptr was allocated with this layout, so we can deallocate with it
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

// Safety: Safe to send between threads if T is Send
// SimdAlignedBuffer owns its data via NonNull and manages its own allocation/deallocation
#[allow(clippy::non_send_fields_in_send_ty)]
#[allow(unsafe_code)]
unsafe impl<T: Send> Send for SimdAlignedBuffer<T> {}

// Safety: Safe to share between threads if T is Sync
// Access to data is mediated through standard Rust references
#[allow(unsafe_code)]
unsafe impl<T: Sync> Sync for SimdAlignedBuffer<T> {}

impl<T> std::ops::Index<usize> for SimdAlignedBuffer<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<T> std::ops::IndexMut<usize> for SimdAlignedBuffer<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}

/// SIMD-aligned memory pool allocator
///
/// Manages a pool of SIMD-aligned buffers for reuse, reducing allocation
/// overhead in real-time audio threads.
pub struct SimdAllocator<T> {
    /// Pool of available buffers
    pool: Vec<SimdAlignedBuffer<T>>,

    /// Maximum pool size
    max_pool_size: usize,

    /// Statistics
    total_allocations: u64,
    pool_hits: u64,
}

impl<T> SimdAllocator<T> {
    /// Create new SIMD allocator
    ///
    /// # Arguments
    ///
    /// * `max_pool_size` - Maximum number of buffers to keep in pool
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            pool: Vec::with_capacity(max_pool_size),
            max_pool_size,
            total_allocations: 0,
            pool_hits: 0,
        }
    }

    /// Allocate or reuse buffer
    ///
    /// Returns a buffer from the pool if available, otherwise allocates new
    pub fn allocate(&mut self, len: usize) -> SimdAlignedBuffer<T> {
        self.total_allocations += 1;

        // Try to reuse from pool
        if let Some(buffer) = self.pool.pop() {
            if buffer.len() == len {
                self.pool_hits += 1;
                return buffer;
            }
            // Wrong size, drop and allocate new
        }

        SimdAlignedBuffer::new(len)
    }

    /// Allocate zero-initialized buffer
    pub fn allocate_zeroed(&mut self, len: usize) -> SimdAlignedBuffer<T>
    where
        T: Copy,
    {
        let mut buffer = self.allocate(len);
        // Safety: buffer.as_mut_ptr() is valid for `len` elements and properly aligned
        // write_bytes with 0 is valid for any type
        unsafe {
            std::ptr::write_bytes(buffer.as_mut_ptr(), 0, len);
        }
        buffer
    }

    /// Return buffer to pool for reuse
    pub fn deallocate(&mut self, buffer: SimdAlignedBuffer<T>) {
        if self.pool.len() < self.max_pool_size {
            self.pool.push(buffer);
        }
        // Otherwise drop buffer
    }

    /// Get pool hit rate
    pub fn hit_rate(&self) -> f64 {
        if self.total_allocations == 0 {
            0.0
        } else {
            self.pool_hits as f64 / self.total_allocations as f64
        }
    }

    /// Clear the pool
    pub fn clear(&mut self) {
        self.pool.clear();
    }
}

impl<T> Default for SimdAllocator<T> {
    fn default() -> Self {
        Self::new(16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment() {
        let buffer = SimdAlignedBuffer::<f32>::new(1024);
        assert!(buffer.is_aligned());
        assert_eq!(buffer.as_ptr() as usize % SIMD_ALIGNMENT, 0);
    }

    #[test]
    fn test_zeroed() {
        let buffer = SimdAlignedBuffer::<f32>::zeroed(256);
        for &val in buffer.as_slice() {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_allocator_pool() {
        let mut allocator = SimdAllocator::<f32>::new(4);

        // Allocate and deallocate
        let buf1 = allocator.allocate(512);
        allocator.deallocate(buf1);

        // Should reuse from pool
        let _buf2 = allocator.allocate(512);
        assert_eq!(allocator.pool_hits, 1);
        assert!(allocator.hit_rate() > 0.0);
    }

    #[test]
    fn test_allocator_different_sizes() {
        let mut allocator = SimdAllocator::<f32>::new(4);

        let buf1 = allocator.allocate(512);
        allocator.deallocate(buf1);

        // Different size shouldn't reuse
        let _buf2 = allocator.allocate(1024);
        assert_eq!(allocator.pool_hits, 0);
    }

    #[test]
    fn test_slice_access() {
        let mut buffer = SimdAlignedBuffer::<f32>::new(16);

        // Write via mutable slice
        for (i, val) in buffer.as_mut_slice().iter_mut().enumerate() {
            *val = i as f32;
        }

        // Read via slice
        for (i, &val) in buffer.as_slice().iter().enumerate() {
            assert_eq!(val, i as f32);
        }
    }

    #[test]
    fn test_index_access() {
        let mut buffer = SimdAlignedBuffer::<f32>::new(16);

        buffer[0] = 1.0;
        buffer[15] = 15.0;

        assert_eq!(buffer[0], 1.0);
        assert_eq!(buffer[15], 15.0);
    }
}
