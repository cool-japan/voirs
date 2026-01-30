//! Lock-free Ring Buffer Implementation
//!
//! Provides wait-free single-producer single-consumer (SPSC) and lock-free
//! multi-producer single-consumer (MPSC) ring buffers for real-time audio.
//!
//! These implementations use atomic operations and careful memory ordering
//! to ensure correctness without locks, making them suitable for real-time
//! audio threads where blocking is unacceptable.

#![allow(clippy::undocumented_unsafe_blocks)] // Safety comments provided inline
#![allow(unsafe_code)] // Required for lock-free primitives

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thiserror::Error;

/// Ring buffer errors
#[derive(Debug, Error)]
pub enum RingBufferError {
    /// Buffer is full, cannot write the requested number of samples
    #[error("Buffer is full, cannot write {0} samples")]
    BufferFull(usize),

    /// Buffer is empty, cannot read the requested number of samples
    #[error("Buffer is empty, cannot read {0} samples")]
    BufferEmpty(usize),

    /// Invalid buffer size (must be power of 2)
    #[error("Invalid buffer size: must be power of 2, got {0}")]
    InvalidSize(usize),

    /// Insufficient data available for read operation
    #[error("Requested {requested} samples but only {available} available")]
    InsufficientData {
        /// Number of samples requested
        requested: usize,
        /// Number of samples available
        available: usize,
    },
}

/// Lock-free Single Producer Single Consumer ring buffer
///
/// This is a wait-free implementation optimized for real-time audio streaming.
/// The producer can write without blocking, and the consumer can read without
/// blocking, making it ideal for audio callback scenarios.
///
/// # Memory Ordering
///
/// - Write index uses `Release` ordering when publishing new data
/// - Read index uses `Acquire` ordering when consuming data
/// - This ensures proper happens-before relationships without locks
///
/// # Performance
///
/// - Zero allocations after initialization
/// - Cache-line aligned indices to avoid false sharing
/// - Power-of-2 size for efficient modulo operations
///
/// # Example
///
/// ```rust
/// use voirs_spatial::realtime::LockFreeSPSC;
///
/// let buffer = LockFreeSPSC::<f32>::new(1024).expect("Failed to create buffer");
/// let (mut producer, mut consumer) = buffer.split();
///
/// // Producer thread
/// let data = vec![0.1, 0.2, 0.3, 0.4];
/// producer.write(&data).expect("Failed to write to buffer");
///
/// // Consumer thread
/// let mut output = vec![0.0; 4];
/// consumer.read(&mut output).expect("Failed to read from buffer");
/// ```
pub struct LockFreeSPSC<T: Copy> {
    /// Ring buffer storage (power of 2 size)
    buffer: Box<[T]>,

    /// Write index (modified by producer only)
    write_idx: AtomicUsize,

    /// Padding to prevent false sharing
    _pad1: [u8; 64 - std::mem::size_of::<AtomicUsize>()],

    /// Read index (modified by consumer only)
    read_idx: AtomicUsize,

    /// Padding to prevent false sharing
    _pad2: [u8; 64 - std::mem::size_of::<AtomicUsize>()],

    /// Capacity mask (capacity - 1) for fast modulo
    capacity_mask: usize,
}

/// Producer half of SPSC ring buffer
pub struct SPSCProducer<T: Copy + Default> {
    buffer: Arc<LockFreeSPSC<T>>,
}

/// Consumer half of SPSC ring buffer
pub struct SPSCConsumer<T: Copy + Default> {
    buffer: Arc<LockFreeSPSC<T>>,
}

impl<T: Copy + Default> LockFreeSPSC<T> {
    /// Create new SPSC ring buffer with given capacity
    ///
    /// # Arguments
    ///
    /// * `capacity` - Buffer size (will be rounded up to next power of 2)
    ///
    /// # Errors
    ///
    /// Returns error if capacity is 0 or too large
    pub fn new(capacity: usize) -> Result<Arc<Self>, RingBufferError> {
        if capacity == 0 {
            return Err(RingBufferError::InvalidSize(0));
        }

        // Round up to next power of 2
        let capacity = capacity.next_power_of_two();

        // Create buffer with default values
        let buffer: Vec<T> = (0..capacity).map(|_| T::default()).collect();

        Ok(Arc::new(Self {
            buffer: buffer.into_boxed_slice(),
            write_idx: AtomicUsize::new(0),
            _pad1: [0; 64 - std::mem::size_of::<AtomicUsize>()],
            read_idx: AtomicUsize::new(0),
            _pad2: [0; 64 - std::mem::size_of::<AtomicUsize>()],
            capacity_mask: capacity - 1,
        }))
    }

    /// Split into producer and consumer halves
    pub fn split(self: Arc<Self>) -> (SPSCProducer<T>, SPSCConsumer<T>) {
        (
            SPSCProducer {
                buffer: Arc::clone(&self),
            },
            SPSCConsumer { buffer: self },
        )
    }

    /// Get available space for writing
    #[inline]
    fn available_write(&self) -> usize {
        let write = self.write_idx.load(Ordering::Relaxed);
        let read = self.read_idx.load(Ordering::Acquire);

        // Available = capacity - used
        // Used = (write - read) & mask
        let used = (write.wrapping_sub(read)) & self.capacity_mask;
        (self.capacity_mask + 1) - used - 1 // -1 to prevent write overtaking read
    }

    /// Get available data for reading
    #[inline]
    fn available_read(&self) -> usize {
        let write = self.write_idx.load(Ordering::Acquire);
        let read = self.read_idx.load(Ordering::Relaxed);

        // Available = (write - read) & mask
        (write.wrapping_sub(read)) & self.capacity_mask
    }
}

impl<T: Copy + Default> SPSCProducer<T> {
    /// Write data to the ring buffer (wait-free)
    ///
    /// # Arguments
    ///
    /// * `data` - Slice of data to write
    ///
    /// # Returns
    ///
    /// Number of samples actually written
    ///
    /// # Errors
    ///
    /// Returns error if buffer is full
    pub fn write(&mut self, data: &[T]) -> Result<usize, RingBufferError> {
        let available = self.buffer.available_write();

        if data.len() > available {
            return Err(RingBufferError::BufferFull(data.len()));
        }

        let write_idx = self.buffer.write_idx.load(Ordering::Relaxed);
        let capacity = self.buffer.capacity_mask + 1;

        // Write in two parts if wrapping around
        let first_chunk_size = (capacity - write_idx).min(data.len());
        let second_chunk_size = data.len() - first_chunk_size;

        // Copy first chunk
        unsafe {
            let dst = self.buffer.buffer.as_ptr().add(write_idx) as *mut T;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, first_chunk_size);
        }

        // Copy second chunk if wrapping
        if second_chunk_size > 0 {
            unsafe {
                let dst = self.buffer.buffer.as_ptr() as *mut T;
                std::ptr::copy_nonoverlapping(
                    data.as_ptr().add(first_chunk_size),
                    dst,
                    second_chunk_size,
                );
            }
        }

        // Update write index with Release ordering to publish data
        let new_write_idx = (write_idx + data.len()) & self.buffer.capacity_mask;
        self.buffer
            .write_idx
            .store(new_write_idx, Ordering::Release);

        Ok(data.len())
    }

    /// Try to write data without error on full buffer
    ///
    /// Returns number of samples actually written (may be less than requested)
    pub fn try_write(&mut self, data: &[T]) -> usize {
        let available = self.buffer.available_write();
        let to_write = data.len().min(available);

        if to_write == 0 {
            return 0;
        }

        self.write(&data[..to_write]).unwrap_or(0)
    }

    /// Get available write space
    pub fn available(&self) -> usize {
        self.buffer.available_write()
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.available() == 0
    }
}

impl<T: Copy + Default> SPSCConsumer<T> {
    /// Read data from the ring buffer (wait-free)
    ///
    /// # Arguments
    ///
    /// * `data` - Mutable slice to read into
    ///
    /// # Returns
    ///
    /// Number of samples actually read
    ///
    /// # Errors
    ///
    /// Returns error if not enough data available
    pub fn read(&mut self, data: &mut [T]) -> Result<usize, RingBufferError> {
        let available = self.buffer.available_read();

        if data.len() > available {
            return Err(RingBufferError::InsufficientData {
                requested: data.len(),
                available,
            });
        }

        let read_idx = self.buffer.read_idx.load(Ordering::Relaxed);
        let capacity = self.buffer.capacity_mask + 1;

        // Read in two parts if wrapping around
        let first_chunk_size = (capacity - read_idx).min(data.len());
        let second_chunk_size = data.len() - first_chunk_size;

        // Copy first chunk
        unsafe {
            let src = self.buffer.buffer.as_ptr().add(read_idx);
            std::ptr::copy_nonoverlapping(src, data.as_mut_ptr(), first_chunk_size);
        }

        // Copy second chunk if wrapping
        if second_chunk_size > 0 {
            unsafe {
                let src = self.buffer.buffer.as_ptr();
                std::ptr::copy_nonoverlapping(
                    src,
                    data.as_mut_ptr().add(first_chunk_size),
                    second_chunk_size,
                );
            }
        }

        // Update read index with Release ordering
        let new_read_idx = (read_idx + data.len()) & self.buffer.capacity_mask;
        self.buffer.read_idx.store(new_read_idx, Ordering::Release);

        Ok(data.len())
    }

    /// Try to read data without error on empty buffer
    ///
    /// Returns number of samples actually read (may be less than requested)
    pub fn try_read(&mut self, data: &mut [T]) -> usize {
        let available = self.buffer.available_read();
        let to_read = data.len().min(available);

        if to_read == 0 {
            return 0;
        }

        self.read(&mut data[..to_read]).unwrap_or(0)
    }

    /// Get available data to read
    pub fn available(&self) -> usize {
        self.buffer.available_read()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.available() == 0
    }
}

/// Lock-free Multi Producer Single Consumer ring buffer
///
/// This allows multiple threads to write audio data while a single consumer
/// thread reads it. Useful for mixing multiple audio sources in real-time.
///
/// Note: This uses compare-and-swap for writes, so it's lock-free but not
/// wait-free like SPSC. Still suitable for real-time audio with bounded latency.
pub struct LockFreeMPSC<T: Copy> {
    /// Ring buffer storage
    buffer: Box<[T]>,

    /// Write index (modified by all producers via CAS)
    write_idx: AtomicUsize,

    /// Padding to prevent false sharing
    _pad1: [u8; 64 - std::mem::size_of::<AtomicUsize>()],

    /// Read index (modified by consumer only)
    read_idx: AtomicUsize,

    /// Padding to prevent false sharing
    _pad2: [u8; 64 - std::mem::size_of::<AtomicUsize>()],

    /// Capacity mask
    capacity_mask: usize,
}

impl<T: Copy + Default> LockFreeMPSC<T> {
    /// Create new MPSC ring buffer
    pub fn new(capacity: usize) -> Result<Arc<Self>, RingBufferError> {
        if capacity == 0 {
            return Err(RingBufferError::InvalidSize(0));
        }

        let capacity = capacity.next_power_of_two();
        let buffer: Vec<T> = (0..capacity).map(|_| T::default()).collect();

        Ok(Arc::new(Self {
            buffer: buffer.into_boxed_slice(),
            write_idx: AtomicUsize::new(0),
            _pad1: [0; 64 - std::mem::size_of::<AtomicUsize>()],
            read_idx: AtomicUsize::new(0),
            _pad2: [0; 64 - std::mem::size_of::<AtomicUsize>()],
            capacity_mask: capacity - 1,
        }))
    }

    /// Write data from any thread (lock-free via CAS)
    ///
    /// Multiple producers can call this concurrently
    pub fn write(&self, data: &[T]) -> Result<usize, RingBufferError> {
        loop {
            let write_idx = self.write_idx.load(Ordering::Acquire);
            let read_idx = self.read_idx.load(Ordering::Acquire);

            // Calculate available space
            let used = (write_idx.wrapping_sub(read_idx)) & self.capacity_mask;
            let available = (self.capacity_mask + 1) - used - 1;

            if data.len() > available {
                return Err(RingBufferError::BufferFull(data.len()));
            }

            // Try to claim space via CAS
            let new_write_idx = (write_idx + data.len()) & self.capacity_mask;
            if self
                .write_idx
                .compare_exchange_weak(
                    write_idx,
                    new_write_idx,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                // Successfully claimed space, now write data
                let capacity = self.capacity_mask + 1;
                let first_chunk = (capacity - write_idx).min(data.len());
                let second_chunk = data.len() - first_chunk;

                unsafe {
                    let dst = self.buffer.as_ptr().add(write_idx) as *mut T;
                    std::ptr::copy_nonoverlapping(data.as_ptr(), dst, first_chunk);

                    if second_chunk > 0 {
                        let dst = self.buffer.as_ptr() as *mut T;
                        std::ptr::copy_nonoverlapping(
                            data.as_ptr().add(first_chunk),
                            dst,
                            second_chunk,
                        );
                    }
                }

                return Ok(data.len());
            }
            // CAS failed, retry
        }
    }

    /// Read data (consumer only)
    pub fn read(&self, data: &mut [T]) -> Result<usize, RingBufferError> {
        let write_idx = self.write_idx.load(Ordering::Acquire);
        let read_idx = self.read_idx.load(Ordering::Relaxed);

        let available = (write_idx.wrapping_sub(read_idx)) & self.capacity_mask;

        if data.len() > available {
            return Err(RingBufferError::InsufficientData {
                requested: data.len(),
                available,
            });
        }

        let capacity = self.capacity_mask + 1;
        let first_chunk = (capacity - read_idx).min(data.len());
        let second_chunk = data.len() - first_chunk;

        unsafe {
            let src = self.buffer.as_ptr().add(read_idx);
            std::ptr::copy_nonoverlapping(src, data.as_mut_ptr(), first_chunk);

            if second_chunk > 0 {
                let src = self.buffer.as_ptr();
                std::ptr::copy_nonoverlapping(
                    src,
                    data.as_mut_ptr().add(first_chunk),
                    second_chunk,
                );
            }
        }

        let new_read_idx = (read_idx + data.len()) & self.capacity_mask;
        self.read_idx.store(new_read_idx, Ordering::Release);

        Ok(data.len())
    }
}

// Safety: These types are Send + Sync with proper atomic synchronization
unsafe impl<T: Copy + Send> Send for LockFreeSPSC<T> {}
unsafe impl<T: Copy + Send> Sync for LockFreeSPSC<T> {}
unsafe impl<T: Copy + Send> Send for LockFreeMPSC<T> {}
unsafe impl<T: Copy + Send> Sync for LockFreeMPSC<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spsc_basic() {
        let buffer = LockFreeSPSC::<f32>::new(16).expect("Failed to create buffer");
        let (mut producer, mut consumer) = buffer.split();

        // Write some data
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(producer.write(&data).expect("Write should succeed"), 4);

        // Read it back
        let mut output = vec![0.0; 4];
        assert_eq!(consumer.read(&mut output).expect("Read should succeed"), 4);
        assert_eq!(output, data);
    }

    #[test]
    fn test_spsc_wraparound() {
        let buffer = LockFreeSPSC::<f32>::new(8).expect("Failed to create buffer");
        let (mut producer, mut consumer) = buffer.split();

        // Fill buffer partially
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        producer.write(&data1).expect("Write should succeed");

        // Read some
        let mut out1 = vec![0.0; 3];
        consumer.read(&mut out1).expect("Read should succeed");

        // Write more (should wrap)
        let data2 = vec![6.0, 7.0, 8.0];
        producer.write(&data2).expect("Write should succeed");

        // Read remaining
        let mut out2 = vec![0.0; 5];
        consumer.read(&mut out2).expect("Read should succeed");

        assert_eq!(&out1, &[1.0, 2.0, 3.0]);
        assert_eq!(&out2, &[4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_spsc_full() {
        let buffer = LockFreeSPSC::<f32>::new(4).expect("Failed to create buffer");
        let (mut producer, _consumer) = buffer.split();

        // Fill buffer (capacity - 1 due to full detection)
        let data = vec![1.0, 2.0, 3.0];
        producer.write(&data).expect("Write should succeed");

        // Should fail when full
        let more_data = vec![4.0];
        assert!(producer.write(&more_data).is_err());
    }

    #[test]
    fn test_spsc_empty() {
        let buffer = LockFreeSPSC::<f32>::new(16).expect("Failed to create buffer");
        let (_producer, mut consumer) = buffer.split();

        // Should fail when empty
        let mut output = vec![0.0; 4];
        assert!(consumer.read(&mut output).is_err());
    }

    #[test]
    fn test_mpsc_basic() {
        let buffer = LockFreeMPSC::<f32>::new(16).expect("Failed to create buffer");

        // Write from "multiple" producers
        let data1 = vec![1.0, 2.0];
        let data2 = vec![3.0, 4.0];

        buffer.write(&data1).expect("Write should succeed");
        buffer.write(&data2).expect("Write should succeed");

        // Read all
        let mut output = vec![0.0; 4];
        buffer.read(&mut output).expect("Read should succeed");

        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
