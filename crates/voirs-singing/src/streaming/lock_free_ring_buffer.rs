//! Lock-free ring buffer for ultra-low latency audio streaming
//!
//! Implements a wait-free SPSC (Single Producer Single Consumer) ring buffer
//! optimized for real-time audio applications with zero-copy operations.
//!
//! # Safety
//!
//! This module uses `unsafe` blocks for performance-critical zero-copy memory operations.
//! All unsafe code has been carefully reviewed for correctness:
//! - Pointer arithmetic is bounds-checked
//! - Memory aliasing is prevented through proper synchronization
//! - All accesses respect the ring buffer invariants

// Allow unsafe code in this module for performance-critical zero-copy operations
#![allow(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Lock-free ring buffer for audio chunks
///
/// This implementation uses atomic operations for thread-safe, wait-free
/// operation suitable for real-time audio synthesis.
#[derive(Clone)]
pub struct LockFreeRingBuffer {
    /// Audio samples buffer
    buffer: Arc<Vec<f32>>,

    /// Write position (producer)
    write_pos: Arc<AtomicUsize>,

    /// Read position (consumer)
    read_pos: Arc<AtomicUsize>,

    /// Buffer capacity in samples
    capacity: usize,
}

impl LockFreeRingBuffer {
    /// Create a new lock-free ring buffer
    ///
    /// # Arguments
    /// * `capacity` - Buffer capacity in samples (must be power of 2 for optimal performance)
    pub fn new(capacity: usize) -> Self {
        // Ensure capacity is power of 2 for efficient modulo operations
        let capacity = capacity.next_power_of_two();

        Self {
            buffer: Arc::new(vec![0.0; capacity]),
            write_pos: Arc::new(AtomicUsize::new(0)),
            read_pos: Arc::new(AtomicUsize::new(0)),
            capacity,
        }
    }

    /// Write samples to the buffer (producer)
    ///
    /// Returns the number of samples actually written.
    /// Uses relaxed ordering for maximum performance in SPSC scenario.
    pub fn write(&self, samples: &[f32]) -> usize {
        let write_idx = self.write_pos.load(Ordering::Relaxed);
        let read_idx = self.read_pos.load(Ordering::Acquire);

        let available = self.available_write_space(write_idx, read_idx);
        let to_write = samples.len().min(available);

        if to_write == 0 {
            return 0;
        }

        // Write samples in two chunks if wrapping around
        let end_idx = write_idx + to_write;
        if end_idx <= self.capacity {
            // No wrap-around
            // SAFETY: We've verified that write_idx + to_write <= capacity, so all accesses are in bounds.
            // The buffer is exclusively owned by the writer at these indices due to SPSC semantics.
            unsafe {
                let buffer_ptr = self.buffer.as_ptr() as *mut f32;
                std::ptr::copy_nonoverlapping(
                    samples.as_ptr(),
                    buffer_ptr.add(write_idx),
                    to_write,
                );
            }
        } else {
            // Wrap-around case
            let first_chunk_size = self.capacity - write_idx;
            let second_chunk_size = to_write - first_chunk_size;

            // SAFETY: first_chunk_size + second_chunk_size == to_write, all accesses are in bounds.
            // The buffer is exclusively owned by the writer at these indices due to SPSC semantics.
            unsafe {
                let buffer_ptr = self.buffer.as_ptr() as *mut f32;

                // Write first chunk to end of buffer
                std::ptr::copy_nonoverlapping(
                    samples.as_ptr(),
                    buffer_ptr.add(write_idx),
                    first_chunk_size,
                );

                // Write second chunk to beginning of buffer
                std::ptr::copy_nonoverlapping(
                    samples.as_ptr().add(first_chunk_size),
                    buffer_ptr,
                    second_chunk_size,
                );
            }
        }

        // Update write position with release ordering
        let new_write_pos = (write_idx + to_write) % self.capacity;
        self.write_pos.store(new_write_pos, Ordering::Release);

        to_write
    }

    /// Read samples from the buffer (consumer)
    ///
    /// Returns the number of samples actually read.
    /// Uses relaxed ordering for maximum performance in SPSC scenario.
    pub fn read(&self, output: &mut [f32]) -> usize {
        let read_idx = self.read_pos.load(Ordering::Relaxed);
        let write_idx = self.write_pos.load(Ordering::Acquire);

        let available = self.available_read_samples(read_idx, write_idx);
        let to_read = output.len().min(available);

        if to_read == 0 {
            return 0;
        }

        // Read samples in two chunks if wrapping around
        let end_idx = read_idx + to_read;
        if end_idx <= self.capacity {
            // No wrap-around
            // SAFETY: We've verified that read_idx + to_read <= capacity, so all accesses are in bounds.
            // The reader has exclusive access to these indices due to SPSC semantics and proper ordering.
            unsafe {
                let buffer_ptr = self.buffer.as_ptr();
                std::ptr::copy_nonoverlapping(
                    buffer_ptr.add(read_idx),
                    output.as_mut_ptr(),
                    to_read,
                );
            }
        } else {
            // Wrap-around case
            let first_chunk_size = self.capacity - read_idx;
            let second_chunk_size = to_read - first_chunk_size;

            // SAFETY: first_chunk_size + second_chunk_size == to_read, all accesses are in bounds.
            // The reader has exclusive access to these indices due to SPSC semantics and proper ordering.
            unsafe {
                let buffer_ptr = self.buffer.as_ptr();

                // Read first chunk from end of buffer
                std::ptr::copy_nonoverlapping(
                    buffer_ptr.add(read_idx),
                    output.as_mut_ptr(),
                    first_chunk_size,
                );

                // Read second chunk from beginning of buffer
                std::ptr::copy_nonoverlapping(
                    buffer_ptr,
                    output.as_mut_ptr().add(first_chunk_size),
                    second_chunk_size,
                );
            }
        }

        // Update read position with release ordering
        let new_read_pos = (read_idx + to_read) % self.capacity;
        self.read_pos.store(new_read_pos, Ordering::Release);

        to_read
    }

    /// Get number of samples available for reading
    pub fn available_samples(&self) -> usize {
        let read_idx = self.read_pos.load(Ordering::Relaxed);
        let write_idx = self.write_pos.load(Ordering::Acquire);
        self.available_read_samples(read_idx, write_idx)
    }

    /// Get number of samples that can be written
    pub fn available_space(&self) -> usize {
        let write_idx = self.write_pos.load(Ordering::Relaxed);
        let read_idx = self.read_pos.load(Ordering::Acquire);
        self.available_write_space(write_idx, read_idx)
    }

    /// Calculate available read samples
    #[inline]
    fn available_read_samples(&self, read_idx: usize, write_idx: usize) -> usize {
        if write_idx >= read_idx {
            write_idx - read_idx
        } else {
            self.capacity - read_idx + write_idx
        }
    }

    /// Calculate available write space
    #[inline]
    fn available_write_space(&self, write_idx: usize, read_idx: usize) -> usize {
        // Reserve one sample to distinguish full from empty
        self.capacity - self.available_read_samples(read_idx, write_idx) - 1
    }

    /// Clear the buffer (reset read/write positions)
    pub fn clear(&self) {
        self.write_pos.store(0, Ordering::Release);
        self.read_pos.store(0, Ordering::Release);
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        let read_idx = self.read_pos.load(Ordering::Relaxed);
        let write_idx = self.write_pos.load(Ordering::Acquire);
        read_idx == write_idx
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.available_space() == 0
    }

    /// Get fill percentage (0.0 to 1.0)
    pub fn fill_percentage(&self) -> f32 {
        self.available_samples() as f32 / self.capacity as f32
    }
}

/// Statistics for lock-free ring buffer
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RingBufferStats {
    /// Total samples written
    pub samples_written: u64,

    /// Total samples read
    pub samples_read: u64,

    /// Number of write overruns (attempted writes when full)
    pub write_overruns: u64,

    /// Number of read underruns (attempted reads when empty)
    pub read_underruns: u64,

    /// Peak fill level observed
    pub peak_fill_level: f32,

    /// Average fill level
    pub avg_fill_level: f32,
}

impl RingBufferStats {
    /// Update statistics after write operation
    pub fn record_write(&mut self, samples_written: usize, requested: usize) {
        self.samples_written += samples_written as u64;
        if samples_written < requested {
            self.write_overruns += 1;
        }
    }

    /// Update statistics after read operation
    pub fn record_read(&mut self, samples_read: usize, requested: usize) {
        self.samples_read += samples_read as u64;
        if samples_read < requested {
            self.read_underruns += 1;
        }
    }

    /// Update fill level statistics
    pub fn update_fill_level(&mut self, current_fill: f32) {
        if current_fill > self.peak_fill_level {
            self.peak_fill_level = current_fill;
        }

        // Update moving average
        let total_samples = self.samples_written + self.samples_read;
        if total_samples > 0 {
            self.avg_fill_level = (self.avg_fill_level * (total_samples - 1) as f32 + current_fill)
                / total_samples as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_creation() {
        let buffer = LockFreeRingBuffer::new(1024);
        assert_eq!(buffer.capacity(), 1024);
        assert!(buffer.is_empty());
        assert!(!buffer.is_full());
    }

    #[test]
    fn test_ring_buffer_write_read() {
        let buffer = LockFreeRingBuffer::new(1024);

        // Write some samples
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let written = buffer.write(&input);
        assert_eq!(written, 5);
        assert_eq!(buffer.available_samples(), 5);

        // Read samples back
        let mut output = vec![0.0; 5];
        let read = buffer.read(&mut output);
        assert_eq!(read, 5);
        assert_eq!(output, input);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_ring_buffer_wrap_around() {
        let buffer = LockFreeRingBuffer::new(8);

        // Fill buffer partially
        let input1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        buffer.write(&input1);

        // Read some samples
        let mut output1 = vec![0.0; 3];
        buffer.read(&mut output1);

        // Write more samples causing wrap-around
        let input2 = vec![6.0, 7.0, 8.0, 9.0];
        let written = buffer.write(&input2);
        assert_eq!(written, 4);

        // Read all remaining samples
        let mut output2 = vec![0.0; 6];
        let read = buffer.read(&mut output2);
        assert_eq!(read, 6);

        // Check correct values
        assert_eq!(&output2[..], &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_ring_buffer_full_condition() {
        let buffer = LockFreeRingBuffer::new(8);

        // Try to fill buffer completely
        let input = vec![1.0; 10];
        let written = buffer.write(&input);

        // Should write capacity - 1 samples (one reserved for full/empty distinction)
        assert_eq!(written, 7);
        assert!(buffer.is_full());
        assert_eq!(buffer.available_space(), 0);
    }

    #[test]
    fn test_ring_buffer_empty_read() {
        let buffer = LockFreeRingBuffer::new(1024);

        let mut output = vec![0.0; 100];
        let read = buffer.read(&mut output);
        assert_eq!(read, 0);
    }

    #[test]
    fn test_ring_buffer_fill_percentage() {
        let buffer = LockFreeRingBuffer::new(1024);

        buffer.write(&vec![1.0; 256]);
        let fill = buffer.fill_percentage();
        assert!((fill - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_ring_buffer_clear() {
        let buffer = LockFreeRingBuffer::new(1024);

        buffer.write(&vec![1.0; 100]);
        assert!(!buffer.is_empty());

        buffer.clear();
        assert!(buffer.is_empty());
        assert_eq!(buffer.available_samples(), 0);
    }

    #[test]
    fn test_ring_buffer_stats() {
        let mut stats = RingBufferStats::default();

        stats.record_write(100, 100);
        assert_eq!(stats.samples_written, 100);
        assert_eq!(stats.write_overruns, 0);

        stats.record_write(50, 100);
        assert_eq!(stats.samples_written, 150);
        assert_eq!(stats.write_overruns, 1);

        stats.record_read(100, 100);
        assert_eq!(stats.samples_read, 100);
        assert_eq!(stats.read_underruns, 0);

        stats.update_fill_level(0.75);
        assert_eq!(stats.peak_fill_level, 0.75);
    }

    #[test]
    fn test_concurrent_write_read() {
        use std::thread;

        let buffer = LockFreeRingBuffer::new(8192);
        let buffer_clone = buffer.clone();

        // Producer thread
        let producer = thread::spawn(move || {
            for i in 0..100 {
                let samples = vec![i as f32; 64];
                while buffer_clone.write(&samples) < samples.len() {
                    thread::yield_now();
                }
            }
        });

        // Consumer thread
        let consumer = thread::spawn(move || {
            let mut total_read = 0;
            while total_read < 100 * 64 {
                let mut output = vec![0.0; 64];
                let read = buffer.read(&mut output);
                total_read += read;
                if read == 0 {
                    thread::yield_now();
                }
            }
            total_read
        });

        producer.join().unwrap();
        let total_read = consumer.join().unwrap();
        assert_eq!(total_read, 100 * 64);
    }
}
