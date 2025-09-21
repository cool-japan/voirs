//! Buffer management for streaming synthesis
//!
//! This module provides thread-safe circular buffers for efficient memory
//! management in real-time streaming scenarios with flow control.

use crate::{AcousticError, Result};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// Thread-safe circular buffer with flow control
pub struct CircularBuffer<T> {
    buffer: Arc<Mutex<VecDeque<T>>>,
    capacity: usize,
    not_full: Arc<Condvar>,
    not_empty: Arc<Condvar>,
}

impl<T: Clone> CircularBuffer<T> {
    /// Create new circular buffer with given capacity
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(AcousticError::InvalidConfiguration(
                "Buffer capacity must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            buffer: Arc::new(Mutex::new(VecDeque::with_capacity(capacity))),
            capacity,
            not_full: Arc::new(Condvar::new()),
            not_empty: Arc::new(Condvar::new()),
        })
    }

    /// Push item to buffer (blocks if full)
    pub fn push(&self, item: T) -> Result<()> {
        let mut buffer = self
            .buffer
            .lock()
            .map_err(|_| AcousticError::Processing("Buffer lock poisoned".to_string()))?;

        // Wait for space if buffer is full
        while buffer.len() >= self.capacity {
            buffer = self.not_full.wait(buffer).map_err(|_| {
                AcousticError::Processing("Buffer condition variable failed".to_string())
            })?;
        }

        buffer.push_back(item);
        self.not_empty.notify_one();

        Ok(())
    }

    /// Try to push item without blocking
    pub fn try_push(&self, item: T) -> Result<bool> {
        let mut buffer = self
            .buffer
            .lock()
            .map_err(|_| AcousticError::Processing("Buffer lock poisoned".to_string()))?;

        if buffer.len() >= self.capacity {
            return Ok(false);
        }

        buffer.push_back(item);
        self.not_empty.notify_one();
        Ok(true)
    }

    /// Push item with timeout
    pub fn push_timeout(&self, item: T, timeout: Duration) -> Result<bool> {
        let mut buffer = self
            .buffer
            .lock()
            .map_err(|_| AcousticError::Processing("Buffer lock poisoned".to_string()))?;

        let deadline = std::time::Instant::now() + timeout;

        while buffer.len() >= self.capacity {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return Ok(false);
            }

            let result = self.not_full.wait_timeout(buffer, remaining).map_err(|_| {
                AcousticError::Processing("Buffer condition variable failed".to_string())
            })?;

            buffer = result.0;
            if result.1.timed_out() {
                return Ok(false);
            }
        }

        buffer.push_back(item);
        self.not_empty.notify_one();
        Ok(true)
    }

    /// Pop item from buffer (blocks if empty)
    pub fn pop(&self) -> Option<T> {
        let mut buffer = match self.buffer.lock() {
            Ok(b) => b,
            Err(_) => return None,
        };

        while buffer.is_empty() {
            buffer = match self.not_empty.wait(buffer) {
                Ok(b) => b,
                Err(_) => return None,
            };
        }

        let item = buffer.pop_front();
        self.not_full.notify_one();
        item
    }

    /// Try to pop item without blocking
    pub fn try_pop(&self) -> Option<T> {
        let mut buffer = match self.buffer.lock() {
            Ok(b) => b,
            Err(_) => return None,
        };

        let item = buffer.pop_front();
        if item.is_some() {
            self.not_full.notify_one();
        }
        item
    }

    /// Pop item with timeout
    pub fn pop_timeout(&self, timeout: Duration) -> Option<T> {
        let mut buffer = match self.buffer.lock() {
            Ok(b) => b,
            Err(_) => return None,
        };

        let deadline = std::time::Instant::now() + timeout;

        while buffer.is_empty() {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return None;
            }

            let result = match self.not_empty.wait_timeout(buffer, remaining) {
                Ok(r) => r,
                Err(_) => return None,
            };

            buffer = result.0;
            if result.1.timed_out() {
                return None;
            }
        }

        let item = buffer.pop_front();
        if item.is_some() {
            self.not_full.notify_one();
        }
        item
    }

    /// Drain multiple items at once
    pub fn drain(&mut self, count: usize) -> Vec<T> {
        let mut buffer = match self.buffer.lock() {
            Ok(b) => b,
            Err(_) => return Vec::new(),
        };

        let actual_count = count.min(buffer.len());
        let mut items = Vec::with_capacity(actual_count);

        for _ in 0..actual_count {
            if let Some(item) = buffer.pop_front() {
                items.push(item);
            }
        }

        if !items.is_empty() {
            self.not_full.notify_all();
        }

        items
    }

    /// Get current buffer length
    pub fn len(&self) -> usize {
        match self.buffer.lock() {
            Ok(buffer) => buffer.len(),
            Err(_) => 0,
        }
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get buffer utilization (0.0 to 1.0)
    pub fn utilization(&self) -> f32 {
        self.len() as f32 / self.capacity as f32
    }

    /// Clear all items from buffer
    pub fn clear(&mut self) {
        if let Ok(mut buffer) = self.buffer.lock() {
            buffer.clear();
            self.not_full.notify_all();
        }
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// Get available space
    pub fn available_space(&self) -> usize {
        self.capacity.saturating_sub(self.len())
    }
}

impl<T: Clone> Clone for CircularBuffer<T> {
    fn clone(&self) -> Self {
        Self {
            buffer: Arc::clone(&self.buffer),
            capacity: self.capacity,
            not_full: Arc::clone(&self.not_full),
            not_empty: Arc::clone(&self.not_empty),
        }
    }
}

/// Memory pool for reusing allocated objects
pub struct MemoryPool<T> {
    pool: Arc<Mutex<Vec<T>>>,
    factory: Box<dyn Fn() -> T + Send + Sync>,
    max_size: usize,
}

impl<T: Clone + Send + 'static> MemoryPool<T> {
    /// Create new memory pool with factory function
    pub fn new<F>(factory: F, initial_size: usize, max_size: usize) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        let mut pool = Vec::with_capacity(initial_size);
        for _ in 0..initial_size {
            pool.push(factory());
        }

        Self {
            pool: Arc::new(Mutex::new(pool)),
            factory: Box::new(factory),
            max_size,
        }
    }

    /// Get object from pool or create new one
    pub fn acquire(&self) -> T {
        if let Ok(mut pool) = self.pool.lock() {
            if let Some(item) = pool.pop() {
                return item;
            }
        }

        // Pool is empty, create new object
        (self.factory)()
    }

    /// Return object to pool
    pub fn release(&self, item: T) {
        if let Ok(mut pool) = self.pool.lock() {
            if pool.len() < self.max_size {
                pool.push(item);
            }
            // If pool is full, just drop the item
        }
    }

    /// Get current pool size
    pub fn size(&self) -> usize {
        match self.pool.lock() {
            Ok(pool) => pool.len(),
            Err(_) => 0,
        }
    }

    /// Clear all items from pool
    pub fn clear(&self) {
        if let Ok(mut pool) = self.pool.lock() {
            pool.clear();
        }
    }
}

/// Buffer statistics for monitoring
#[derive(Debug, Clone)]
pub struct BufferStats {
    /// Current length
    pub length: usize,
    /// Maximum capacity
    pub capacity: usize,
    /// Utilization percentage (0.0 to 1.0)
    pub utilization: f32,
    /// Available space
    pub available_space: usize,
    /// Number of push operations
    pub push_count: u64,
    /// Number of pop operations  
    pub pop_count: u64,
    /// Number of timeouts
    pub timeout_count: u64,
}

/// Buffer monitor for collecting statistics
pub struct BufferMonitor<T> {
    buffer: CircularBuffer<T>,
    stats: Arc<Mutex<BufferStats>>,
}

impl<T: Clone> BufferMonitor<T> {
    /// Create new monitored buffer
    pub fn new(capacity: usize) -> Result<Self> {
        let buffer = CircularBuffer::new(capacity)?;
        let stats = Arc::new(Mutex::new(BufferStats {
            length: 0,
            capacity,
            utilization: 0.0,
            available_space: capacity,
            push_count: 0,
            pop_count: 0,
            timeout_count: 0,
        }));

        Ok(Self { buffer, stats })
    }

    /// Push with statistics tracking
    pub fn push(&self, item: T) -> Result<()> {
        let result = self.buffer.push(item);
        self.update_stats();
        if let Ok(mut stats) = self.stats.lock() {
            stats.push_count += 1;
        }
        result
    }

    /// Pop with statistics tracking
    pub fn pop(&self) -> Option<T> {
        let result = self.buffer.pop();
        self.update_stats();
        if result.is_some() {
            if let Ok(mut stats) = self.stats.lock() {
                stats.pop_count += 1;
            }
        }
        result
    }

    /// Get current statistics
    pub fn stats(&self) -> BufferStats {
        self.update_stats();
        match self.stats.lock() {
            Ok(stats) => stats.clone(),
            Err(_) => BufferStats {
                length: 0,
                capacity: self.buffer.capacity(),
                utilization: 0.0,
                available_space: self.buffer.capacity(),
                push_count: 0,
                pop_count: 0,
                timeout_count: 0,
            },
        }
    }

    fn update_stats(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.length = self.buffer.len();
            stats.utilization = self.buffer.utilization();
            stats.available_space = self.buffer.available_space();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    #[test]
    fn test_circular_buffer_basic() {
        let buffer = CircularBuffer::new(3).unwrap();

        assert!(buffer.try_push(1).unwrap());
        assert!(buffer.try_push(2).unwrap());
        assert!(buffer.try_push(3).unwrap());
        assert!(!buffer.try_push(4).unwrap()); // Buffer full

        assert_eq!(buffer.try_pop(), Some(1));
        assert_eq!(buffer.try_pop(), Some(2));
        assert_eq!(buffer.try_pop(), Some(3));
        assert_eq!(buffer.try_pop(), None); // Buffer empty
    }

    #[test]
    fn test_circular_buffer_capacity() {
        let buffer: CircularBuffer<i32> = CircularBuffer::new(5).unwrap();
        assert_eq!(buffer.capacity(), 5);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        assert!(!buffer.is_full());
        assert_eq!(buffer.available_space(), 5);
    }

    #[test]
    fn test_circular_buffer_drain() {
        let mut buffer = CircularBuffer::new(5).unwrap();

        for i in 1..=5 {
            buffer.try_push(i).unwrap();
        }

        let drained = buffer.drain(3);
        assert_eq!(drained, vec![1, 2, 3]);
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new(|| vec![0u8; 1024], 2, 5);

        let item1 = pool.acquire();
        let item2 = pool.acquire();
        let _item3 = pool.acquire(); // Should create new since pool is empty

        assert_eq!(pool.size(), 0); // All items acquired

        pool.release(item1);
        pool.release(item2);

        assert_eq!(pool.size(), 2); // Items returned to pool

        let _reused = pool.acquire();
        assert_eq!(pool.size(), 1); // One item reused
    }

    #[test]
    fn test_buffer_monitor() {
        let monitor = BufferMonitor::new(3).unwrap();

        monitor.push(1).unwrap();
        monitor.push(2).unwrap();

        let stats = monitor.stats();
        assert_eq!(stats.length, 2);
        assert_eq!(stats.capacity, 3);
        assert!((stats.utilization - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(stats.push_count, 2);
        assert_eq!(stats.pop_count, 0);

        let item = monitor.pop();
        assert_eq!(item, Some(1));

        let stats = monitor.stats();
        assert_eq!(stats.pop_count, 1);
    }

    #[test]
    fn test_concurrent_access() {
        let buffer = Arc::new(CircularBuffer::new(100).unwrap());
        let counter = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let buffer = Arc::clone(&buffer);
                let counter = Arc::clone(&counter);

                thread::spawn(move || {
                    for i in 0..25 {
                        buffer.push(i).unwrap();
                        counter.fetch_add(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(counter.load(Ordering::SeqCst), 100);
        assert_eq!(buffer.len(), 100);
    }
}
