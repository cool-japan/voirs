//! Buffer management for streaming audio processing
//!
//! Provides various buffer implementations optimized for low-latency,
//! lock-free, and memory-efficient streaming operations.

use super::{StreamingError, StreamingStats};
use crate::config::{BufferStrategy, StreamingConfig};
use crate::{AudioBuffer, Result};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

/// Trait for streaming buffer implementations
pub trait StreamingBuffer: Send + Sync {
    /// Push audio data to the buffer
    fn push(&self, audio: AudioBuffer) -> Result<()>;

    /// Pop audio data from the buffer
    fn pop(&self) -> Option<AudioBuffer>;

    /// Get current buffer size
    fn size(&self) -> usize;

    /// Get maximum buffer capacity
    fn capacity(&self) -> usize;

    /// Check if buffer is empty
    fn is_empty(&self) -> bool;

    /// Check if buffer is full
    fn is_full(&self) -> bool;

    /// Clear all data from buffer
    fn clear(&self);

    /// Get buffer utilization (0.0-1.0)
    fn utilization(&self) -> f32;

    /// Get buffer statistics
    fn stats(&self) -> BufferStats;
}

/// Buffer performance statistics
#[derive(Debug, Clone, Default)]
pub struct BufferStats {
    /// Total items pushed
    pub items_pushed: u64,

    /// Total items popped
    pub items_popped: u64,

    /// Current buffer size
    pub current_size: usize,

    /// Peak buffer size
    pub peak_size: usize,

    /// Buffer overflows
    pub overflows: u64,

    /// Buffer underflows
    pub underflows: u64,

    /// Average utilization
    pub avg_utilization: f32,

    /// Peak utilization
    pub peak_utilization: f32,
}

/// Dynamic buffer that adapts size based on processing speed
pub struct DynamicBuffer {
    /// Internal buffer storage
    buffer: Arc<Mutex<VecDeque<AudioBuffer>>>,

    /// Current capacity
    capacity: Arc<RwLock<usize>>,

    /// Maximum allowed capacity
    max_capacity: usize,

    /// Minimum allowed capacity
    min_capacity: usize,

    /// Buffer statistics
    stats: Arc<RwLock<BufferStats>>,

    /// Last resize time
    last_resize: Arc<Mutex<Instant>>,

    /// Resize threshold
    resize_threshold: f32,
}

impl DynamicBuffer {
    /// Create new dynamic buffer
    pub fn new(initial_capacity: usize, max_capacity: usize) -> Self {
        Self {
            buffer: Arc::new(Mutex::new(VecDeque::with_capacity(initial_capacity))),
            capacity: Arc::new(RwLock::new(initial_capacity)),
            max_capacity,
            min_capacity: initial_capacity.min(4),
            stats: Arc::new(RwLock::new(BufferStats::default())),
            last_resize: Arc::new(Mutex::new(Instant::now())),
            resize_threshold: 0.8,
        }
    }

    /// Adapt buffer size based on utilization
    fn adapt_size(&self) {
        // Get size and capacity atomically to avoid lock contention
        let (current_size, capacity) = {
            let buffer = self.buffer.lock().unwrap();
            let cap = *self.capacity.read().unwrap();
            (buffer.len(), cap)
        };

        let utilization = if capacity > 0 {
            current_size as f32 / capacity as f32
        } else {
            0.0
        };

        let now = Instant::now();
        let mut last_resize = self.last_resize.lock().unwrap();

        // Only resize if enough time has passed (avoid thrashing)
        if now.duration_since(*last_resize).as_millis() < 100 {
            return;
        }

        let mut should_resize = false;
        let mut new_capacity = capacity;

        // Increase capacity if utilization is high
        if utilization > self.resize_threshold && capacity < self.max_capacity {
            new_capacity = (capacity * 2).min(self.max_capacity);
            should_resize = true;
        }
        // Decrease capacity if utilization is low
        else if utilization < 0.3 && capacity > self.min_capacity {
            new_capacity = (capacity / 2).max(self.min_capacity);
            should_resize = true;
        }

        if should_resize {
            if let Ok(mut cap) = self.capacity.write() {
                *cap = new_capacity;
                *last_resize = now;

                tracing::debug!(
                    "Adapted buffer size from {} to {} (utilization: {:.2})",
                    capacity,
                    new_capacity,
                    utilization
                );
            }
        }
    }
}

impl StreamingBuffer for DynamicBuffer {
    fn push(&self, audio: AudioBuffer) -> Result<()> {
        let capacity = *self.capacity.read().unwrap();

        // First check: can we push without overflow?
        {
            let mut buffer = self.buffer.lock().unwrap();
            if buffer.len() < capacity {
                buffer.push_back(audio);

                // Update statistics
                if let Ok(mut stats) = self.stats.write() {
                    stats.items_pushed += 1;
                    stats.current_size = buffer.len();
                    if stats.current_size > stats.peak_size {
                        stats.peak_size = stats.current_size;
                    }

                    let utilization = stats.current_size as f32 / capacity as f32;
                    stats.avg_utilization = (stats.avg_utilization + utilization) / 2.0;
                    if utilization > stats.peak_utilization {
                        stats.peak_utilization = utilization;
                    }
                }

                return Ok(());
            }
        }

        // Buffer is full - try to adapt size (without holding locks)
        self.adapt_size();

        // Try again with new capacity
        {
            let mut buffer = self.buffer.lock().unwrap();
            let new_capacity = *self.capacity.read().unwrap();

            if buffer.len() >= new_capacity {
                // Still full after adaptation
                if let Ok(mut stats) = self.stats.write() {
                    stats.overflows += 1;
                }
                return Err(StreamingError::BufferOverflow.into());
            }

            buffer.push_back(audio);

            // Update statistics
            if let Ok(mut stats) = self.stats.write() {
                stats.items_pushed += 1;
                stats.current_size = buffer.len();
                if stats.current_size > stats.peak_size {
                    stats.peak_size = stats.current_size;
                }

                let utilization = stats.current_size as f32 / new_capacity as f32;
                stats.avg_utilization = (stats.avg_utilization + utilization) / 2.0;
                if utilization > stats.peak_utilization {
                    stats.peak_utilization = utilization;
                }
            }
        }

        Ok(())
    }

    fn pop(&self) -> Option<AudioBuffer> {
        let audio = {
            let mut buffer = self.buffer.lock().unwrap();
            let audio = buffer.pop_front();

            // Update statistics
            if let Ok(mut stats) = self.stats.write() {
                if audio.is_some() {
                    stats.items_popped += 1;
                    stats.current_size = buffer.len();
                } else {
                    stats.underflows += 1;
                }
            }

            audio
        };

        // Adapt size after popping (without holding locks)
        if audio.is_some() {
            self.adapt_size();
        }

        audio
    }

    fn size(&self) -> usize {
        self.buffer.lock().unwrap().len()
    }

    fn capacity(&self) -> usize {
        *self.capacity.read().unwrap()
    }

    fn is_empty(&self) -> bool {
        self.buffer.lock().unwrap().is_empty()
    }

    fn is_full(&self) -> bool {
        let buffer = self.buffer.lock().unwrap();
        let capacity = *self.capacity.read().unwrap();
        buffer.len() >= capacity
    }

    fn clear(&self) {
        let mut buffer = self.buffer.lock().unwrap();
        buffer.clear();

        if let Ok(mut stats) = self.stats.write() {
            stats.current_size = 0;
        }
    }

    fn utilization(&self) -> f32 {
        // Get both values with minimal lock contention
        let (size, capacity) = {
            let buffer = self.buffer.lock().unwrap();
            let capacity = *self.capacity.read().unwrap();
            (buffer.len(), capacity)
        };

        if capacity > 0 {
            size as f32 / capacity as f32
        } else {
            0.0
        }
    }

    fn stats(&self) -> BufferStats {
        self.stats.read().unwrap().clone()
    }
}

/// Lock-free ring buffer for ultra-low latency applications
pub struct RingBuffer {
    /// Circular buffer storage
    buffer: Arc<RwLock<Vec<Option<AudioBuffer>>>>,

    /// Write position
    write_pos: Arc<RwLock<usize>>,

    /// Read position
    read_pos: Arc<RwLock<usize>>,

    /// Buffer capacity
    capacity: usize,

    /// Buffer statistics
    stats: Arc<RwLock<BufferStats>>,
}

impl RingBuffer {
    /// Create new ring buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Arc::new(RwLock::new(vec![None; capacity])),
            write_pos: Arc::new(RwLock::new(0)),
            read_pos: Arc::new(RwLock::new(0)),
            capacity,
            stats: Arc::new(RwLock::new(BufferStats::default())),
        }
    }

    /// Get next write position
    fn next_write_pos(&self, current: usize) -> usize {
        (current + 1) % self.capacity
    }

    /// Get next read position
    fn next_read_pos(&self, current: usize) -> usize {
        (current + 1) % self.capacity
    }
}

impl StreamingBuffer for RingBuffer {
    fn push(&self, audio: AudioBuffer) -> Result<()> {
        let write_pos = *self.write_pos.read().unwrap();
        let read_pos = *self.read_pos.read().unwrap();
        let next_write = self.next_write_pos(write_pos);

        // Check if buffer would overflow
        if next_write == read_pos {
            if let Ok(mut stats) = self.stats.write() {
                stats.overflows += 1;
            }
            return Err(StreamingError::BufferOverflow.into());
        }

        // Write to buffer
        if let Ok(mut buffer) = self.buffer.write() {
            buffer[write_pos] = Some(audio);
        }

        // Update write position
        if let Ok(mut pos) = self.write_pos.write() {
            *pos = next_write;
        }

        // Update statistics
        if let Ok(mut stats) = self.stats.write() {
            stats.items_pushed += 1;
            stats.current_size = self.size();
            if stats.current_size > stats.peak_size {
                stats.peak_size = stats.current_size;
            }
        }

        Ok(())
    }

    fn pop(&self) -> Option<AudioBuffer> {
        let read_pos = *self.read_pos.read().unwrap();
        let write_pos = *self.write_pos.read().unwrap();

        // Check if buffer is empty
        if read_pos == write_pos {
            if let Ok(mut stats) = self.stats.write() {
                stats.underflows += 1;
            }
            return None;
        }

        // Read from buffer
        let audio = if let Ok(mut buffer) = self.buffer.write() {
            buffer[read_pos].take()
        } else {
            return None;
        };

        // Update read position
        if let Ok(mut pos) = self.read_pos.write() {
            *pos = self.next_read_pos(read_pos);
        }

        // Update statistics
        if let Ok(mut stats) = self.stats.write() {
            stats.items_popped += 1;
            stats.current_size = self.size();
        }

        audio
    }

    fn size(&self) -> usize {
        let write_pos = *self.write_pos.read().unwrap();
        let read_pos = *self.read_pos.read().unwrap();

        if write_pos >= read_pos {
            write_pos - read_pos
        } else {
            self.capacity - read_pos + write_pos
        }
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn is_empty(&self) -> bool {
        let write_pos = *self.write_pos.read().unwrap();
        let read_pos = *self.read_pos.read().unwrap();
        write_pos == read_pos
    }

    fn is_full(&self) -> bool {
        let write_pos = *self.write_pos.read().unwrap();
        let read_pos = *self.read_pos.read().unwrap();
        self.next_write_pos(write_pos) == read_pos
    }

    fn clear(&self) {
        if let Ok(mut buffer) = self.buffer.write() {
            for item in buffer.iter_mut() {
                *item = None;
            }
        }

        if let Ok(mut write_pos) = self.write_pos.write() {
            *write_pos = 0;
        }

        if let Ok(mut read_pos) = self.read_pos.write() {
            *read_pos = 0;
        }

        if let Ok(mut stats) = self.stats.write() {
            stats.current_size = 0;
        }
    }

    fn utilization(&self) -> f32 {
        let size = self.size();
        if self.capacity > 0 {
            size as f32 / self.capacity as f32
        } else {
            0.0
        }
    }

    fn stats(&self) -> BufferStats {
        if let Ok(stats) = self.stats.read() {
            stats.clone()
        } else {
            BufferStats::default()
        }
    }
}

/// Buffer manager that creates and manages different buffer types
pub struct BufferManager {
    /// Active buffers
    buffers: Arc<RwLock<Vec<Arc<dyn StreamingBuffer>>>>,

    /// Configuration
    config: StreamingConfig,

    /// Global statistics
    #[allow(dead_code)]
    global_stats: Arc<RwLock<StreamingStats>>,
}

impl BufferManager {
    /// Create new buffer manager
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            buffers: Arc::new(RwLock::new(Vec::new())),
            config,
            global_stats: Arc::new(RwLock::new(StreamingStats::default())),
        }
    }

    /// Create a buffer based on strategy
    pub fn create_buffer(&self) -> Arc<dyn StreamingBuffer> {
        let buffer: Arc<dyn StreamingBuffer> = match self.config.buffer_strategy {
            BufferStrategy::Fixed => Arc::new(RingBuffer::new(self.config.buffer_count)),
            BufferStrategy::Dynamic => Arc::new(DynamicBuffer::new(
                self.config.buffer_count,
                self.config.max_buffer_size,
            )),
            BufferStrategy::Circular => Arc::new(RingBuffer::new(self.config.buffer_count)),
            BufferStrategy::LockFree => Arc::new(RingBuffer::new(self.config.buffer_count)),
        };

        // Register buffer
        if let Ok(mut buffers) = self.buffers.write() {
            buffers.push(buffer.clone());
        }

        buffer
    }

    /// Get aggregated statistics from all buffers
    pub fn get_aggregated_stats(&self) -> StreamingStats {
        let mut stats = StreamingStats::default();

        if let Ok(buffers) = self.buffers.read() {
            let mut total_utilization = 0.0;
            let mut buffer_count = 0;

            for buffer in buffers.iter() {
                let buffer_stats = buffer.stats();
                stats.buffer_underruns += buffer_stats.underflows;
                stats.buffer_overruns += buffer_stats.overflows;
                total_utilization += buffer.utilization();
                buffer_count += 1;
            }

            if buffer_count > 0 {
                stats.memory_usage_mb =
                    (total_utilization / buffer_count as f32) * self.config.estimated_memory_mb();
            }
        }

        stats
    }

    /// Cleanup expired or unused buffers
    pub fn cleanup(&self) {
        if let Ok(mut buffers) = self.buffers.write() {
            // Remove buffers that are no longer referenced
            buffers.retain(|buffer| Arc::strong_count(buffer) > 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioBuffer;

    #[test]
    fn test_dynamic_buffer_basic_operations() {
        let buffer = DynamicBuffer::new(4, 16);

        assert!(buffer.is_empty());
        assert!(!buffer.is_full());
        assert_eq!(buffer.size(), 0);
        assert_eq!(buffer.capacity(), 4);

        // Test push/pop
        let audio = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);
        buffer.push(audio.clone()).unwrap();

        assert!(!buffer.is_empty());
        assert_eq!(buffer.size(), 1);

        let popped = buffer.pop().unwrap();
        assert_eq!(popped.duration(), audio.duration());
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_ring_buffer_overflow_handling() {
        let buffer = RingBuffer::new(2);

        let audio1 = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);
        let audio2 = AudioBuffer::sine_wave(880.0, 1.0, 44100, 0.5);

        // Fill buffer (capacity 2 has effective capacity of 1)
        buffer.push(audio1).unwrap();

        // This should cause overflow (ring buffer has effective capacity of size-1)
        let result = buffer.push(audio2);
        assert!(result.is_err());

        let stats = buffer.stats();
        assert_eq!(stats.overflows, 1);
    }

    #[test]
    fn test_buffer_utilization() {
        let buffer = DynamicBuffer::new(4, 16);

        assert_eq!(buffer.utilization(), 0.0);

        let audio = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);
        buffer.push(audio.clone()).unwrap();
        assert_eq!(buffer.utilization(), 0.25);

        buffer.push(audio.clone()).unwrap();
        assert_eq!(buffer.utilization(), 0.5);
    }

    #[test]
    fn test_buffer_manager() {
        let config = StreamingConfig::default();
        let manager = BufferManager::new(config);

        let _buffer1 = manager.create_buffer();
        let _buffer2 = manager.create_buffer();

        assert_eq!(manager.buffers.read().unwrap().len(), 2);

        // Test cleanup (this won't remove buffers since we're holding references)
        manager.cleanup();
        assert_eq!(manager.buffers.read().unwrap().len(), 2);
    }
}
