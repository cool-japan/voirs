use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::cell::UnsafeCell;
use std::ptr;
use crate::VocoderError;
use super::{MemoryConfig, Result};

const CACHE_LINE_SIZE: usize = 64;

#[repr(align(64))]
struct CacheAligned<T>(T);

pub struct LockFreeRingBuffer<T> {
    buffer: Box<[UnsafeCell<T>]>,
    capacity: usize,
    mask: usize,
    head: CacheAligned<AtomicUsize>,
    tail: CacheAligned<AtomicUsize>,
}

impl<T> LockFreeRingBuffer<T> {
    pub fn new(capacity: usize) -> Result<Self> {
        if !capacity.is_power_of_two() {
            return Err(VocoderError::Other("Capacity must be power of two".to_string()));
        }

        let buffer = (0..capacity)
            .map(|_| UnsafeCell::new(unsafe { std::mem::zeroed() }))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Ok(Self {
            buffer,
            capacity,
            mask: capacity - 1,
            head: CacheAligned(AtomicUsize::new(0)),
            tail: CacheAligned(AtomicUsize::new(0)),
        })
    }

    pub fn try_push(&self, item: T) -> Result<()> {
        let current_tail = self.tail.0.load(Ordering::Acquire);
        let next_tail = (current_tail + 1) & self.mask;
        
        if next_tail == self.head.0.load(Ordering::Acquire) {
            return Err(VocoderError::Other("Buffer full".to_string()));
        }

        unsafe {
            ptr::write(self.buffer[current_tail].get(), item);
        }

        self.tail.0.store(next_tail, Ordering::Release);
        Ok(())
    }

    pub fn try_pop(&self) -> Option<T> {
        let current_head = self.head.0.load(Ordering::Acquire);
        
        if current_head == self.tail.0.load(Ordering::Acquire) {
            return None;
        }

        let item = unsafe { ptr::read(self.buffer[current_head].get()) };
        let next_head = (current_head + 1) & self.mask;
        
        self.head.0.store(next_head, Ordering::Release);
        Some(item)
    }

    pub fn is_empty(&self) -> bool {
        self.head.0.load(Ordering::Acquire) == self.tail.0.load(Ordering::Acquire)
    }

    pub fn is_full(&self) -> bool {
        let current_tail = self.tail.0.load(Ordering::Acquire);
        let next_tail = (current_tail + 1) & self.mask;
        next_tail == self.head.0.load(Ordering::Acquire)
    }

    pub fn len(&self) -> usize {
        let head = self.head.0.load(Ordering::Acquire);
        let tail = self.tail.0.load(Ordering::Acquire);
        (tail.wrapping_sub(head)) & self.mask
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

unsafe impl<T: Send> Send for LockFreeRingBuffer<T> {}
unsafe impl<T: Send> Sync for LockFreeRingBuffer<T> {}

pub struct SPSCQueue<T> {
    buffer: LockFreeRingBuffer<T>,
    stats: Arc<QueueStats>,
}

impl<T> SPSCQueue<T> {
    pub fn new(capacity: usize) -> Result<Self> {
        Ok(Self {
            buffer: LockFreeRingBuffer::new(capacity)?,
            stats: Arc::new(QueueStats::default()),
        })
    }

    pub fn push(&self, item: T) -> Result<()> {
        self.stats.push_attempts.fetch_add(1, Ordering::SeqCst);
        
        match self.buffer.try_push(item) {
            Ok(()) => {
                self.stats.successful_pushes.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            Err(e) => {
                self.stats.failed_pushes.fetch_add(1, Ordering::SeqCst);
                Err(e)
            }
        }
    }

    pub fn pop(&self) -> Option<T> {
        self.stats.pop_attempts.fetch_add(1, Ordering::SeqCst);
        
        match self.buffer.try_pop() {
            Some(item) => {
                self.stats.successful_pops.fetch_add(1, Ordering::SeqCst);
                Some(item)
            }
            None => {
                self.stats.failed_pops.fetch_add(1, Ordering::SeqCst);
                None
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.buffer.is_full()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn capacity(&self) -> usize {
        self.buffer.capacity()
    }

    pub fn stats(&self) -> Arc<QueueStats> {
        Arc::clone(&self.stats)
    }
}

pub struct MPMCQueue<T> {
    buffer: LockFreeRingBuffer<T>,
    stats: Arc<QueueStats>,
    producer_lock: AtomicUsize,
    consumer_lock: AtomicUsize,
}

impl<T> MPMCQueue<T> {
    pub fn new(capacity: usize) -> Result<Self> {
        Ok(Self {
            buffer: LockFreeRingBuffer::new(capacity)?,
            stats: Arc::new(QueueStats::default()),
            producer_lock: AtomicUsize::new(0),
            consumer_lock: AtomicUsize::new(0),
        })
    }

    pub fn push(&self, item: T) -> Result<()> {
        self.stats.push_attempts.fetch_add(1, Ordering::SeqCst);
        
        // Simple spinlock for producers
        while self.producer_lock.compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed).is_err() {
            std::hint::spin_loop();
        }

        let result = self.buffer.try_push(item);
        
        self.producer_lock.store(0, Ordering::Release);
        
        match result {
            Ok(()) => {
                self.stats.successful_pushes.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            Err(e) => {
                self.stats.failed_pushes.fetch_add(1, Ordering::SeqCst);
                Err(e)
            }
        }
    }

    pub fn pop(&self) -> Option<T> {
        self.stats.pop_attempts.fetch_add(1, Ordering::SeqCst);
        
        // Simple spinlock for consumers
        while self.consumer_lock.compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed).is_err() {
            std::hint::spin_loop();
        }

        let result = self.buffer.try_pop();
        
        self.consumer_lock.store(0, Ordering::Release);
        
        match result {
            Some(item) => {
                self.stats.successful_pops.fetch_add(1, Ordering::SeqCst);
                Some(item)
            }
            None => {
                self.stats.failed_pops.fetch_add(1, Ordering::SeqCst);
                None
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.buffer.is_full()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn capacity(&self) -> usize {
        self.buffer.capacity()
    }

    pub fn stats(&self) -> Arc<QueueStats> {
        Arc::clone(&self.stats)
    }
}

#[derive(Debug, Clone, Default)]
pub struct QueueStats {
    pub push_attempts: AtomicUsize,
    pub successful_pushes: AtomicUsize,
    pub failed_pushes: AtomicUsize,
    pub pop_attempts: AtomicUsize,
    pub successful_pops: AtomicUsize,
    pub failed_pops: AtomicUsize,
}

impl QueueStats {
    pub fn push_success_rate(&self) -> f32 {
        let total = self.push_attempts.load(Ordering::SeqCst);
        if total == 0 {
            return 0.0;
        }
        
        let successful = self.successful_pushes.load(Ordering::SeqCst);
        successful as f32 / total as f32
    }

    pub fn pop_success_rate(&self) -> f32 {
        let total = self.pop_attempts.load(Ordering::SeqCst);
        if total == 0 {
            return 0.0;
        }
        
        let successful = self.successful_pops.load(Ordering::SeqCst);
        successful as f32 / total as f32
    }
}

pub struct StreamingBufferManager {
    config: MemoryConfig,
    audio_queues: Vec<Arc<SPSCQueue<Vec<f32>>>>,
    mel_queues: Vec<Arc<SPSCQueue<Vec<Vec<f32>>>>>,
    stats: Arc<BufferManagerStats>,
}

impl StreamingBufferManager {
    pub fn new(config: MemoryConfig) -> Result<Self> {
        let num_audio_queues = 4;
        let num_mel_queues = 4;
        
        let mut audio_queues = Vec::new();
        let mut mel_queues = Vec::new();
        
        for _ in 0..num_audio_queues {
            audio_queues.push(Arc::new(SPSCQueue::new(config.max_buffers)?));
        }
        
        for _ in 0..num_mel_queues {
            mel_queues.push(Arc::new(SPSCQueue::new(config.max_buffers)?));
        }
        
        Ok(Self {
            config,
            audio_queues,
            mel_queues,
            stats: Arc::new(BufferManagerStats::default()),
        })
    }

    pub fn get_audio_queue(&self, index: usize) -> Option<Arc<SPSCQueue<Vec<f32>>>> {
        self.audio_queues.get(index).cloned()
    }

    pub fn get_mel_queue(&self, index: usize) -> Option<Arc<SPSCQueue<Vec<Vec<f32>>>>> {
        self.mel_queues.get(index).cloned()
    }

    pub fn num_audio_queues(&self) -> usize {
        self.audio_queues.len()
    }

    pub fn num_mel_queues(&self) -> usize {
        self.mel_queues.len()
    }

    pub fn stats(&self) -> Arc<BufferManagerStats> {
        Arc::clone(&self.stats)
    }

    pub fn total_memory_usage(&self) -> usize {
        let audio_memory: usize = self.audio_queues.iter()
            .map(|q| q.capacity() * std::mem::size_of::<Vec<f32>>())
            .sum();
        
        let mel_memory: usize = self.mel_queues.iter()
            .map(|q| q.capacity() * std::mem::size_of::<Vec<Vec<f32>>>())
            .sum();
        
        audio_memory + mel_memory
    }
}

#[derive(Debug, Clone, Default)]
pub struct BufferManagerStats {
    pub queues_created: AtomicUsize,
    pub total_memory_allocated: AtomicUsize,
    pub peak_memory_usage: AtomicUsize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lock_free_ring_buffer_basic() {
        let buffer: LockFreeRingBuffer<i32> = LockFreeRingBuffer::new(8).unwrap();
        
        assert!(buffer.is_empty());
        assert!(!buffer.is_full());
        assert_eq!(buffer.len(), 0);
        
        buffer.try_push(1).unwrap();
        buffer.try_push(2).unwrap();
        
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.try_pop(), Some(1));
        assert_eq!(buffer.try_pop(), Some(2));
        assert_eq!(buffer.try_pop(), None);
    }

    #[test]
    fn test_lock_free_ring_buffer_wrap_around() {
        let buffer: LockFreeRingBuffer<i32> = LockFreeRingBuffer::new(4).unwrap();
        
        // Fill the buffer
        for i in 0..3 {
            buffer.try_push(i).unwrap();
        }
        
        assert!(buffer.is_full());
        assert!(buffer.try_push(99).is_err());
        
        // Pop one and push one
        assert_eq!(buffer.try_pop(), Some(0));
        buffer.try_push(100).unwrap();
        
        assert_eq!(buffer.try_pop(), Some(1));
        assert_eq!(buffer.try_pop(), Some(2));
        assert_eq!(buffer.try_pop(), Some(100));
    }

    #[test]
    fn test_spsc_queue() {
        let queue = SPSCQueue::new(8).unwrap();
        
        queue.push(1).unwrap();
        queue.push(2).unwrap();
        
        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.pop(), None);
        
        let stats = queue.stats();
        assert_eq!(stats.push_attempts.load(Ordering::SeqCst), 2);
        assert_eq!(stats.successful_pushes.load(Ordering::SeqCst), 2);
        assert_eq!(stats.pop_attempts.load(Ordering::SeqCst), 3);
        assert_eq!(stats.successful_pops.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_mpmc_queue() {
        let queue = MPMCQueue::new(8).unwrap();
        
        queue.push(1).unwrap();
        queue.push(2).unwrap();
        
        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.pop(), None);
        
        let stats = queue.stats();
        assert_eq!(stats.push_attempts.load(Ordering::SeqCst), 2);
        assert_eq!(stats.successful_pushes.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_streaming_buffer_manager() {
        let config = MemoryConfig::default();
        let manager = StreamingBufferManager::new(config).unwrap();
        
        assert_eq!(manager.num_audio_queues(), 4);
        assert_eq!(manager.num_mel_queues(), 4);
        
        let audio_queue = manager.get_audio_queue(0).unwrap();
        let audio_data = vec![1.0, 2.0, 3.0];
        
        audio_queue.push(audio_data.clone()).unwrap();
        assert_eq!(audio_queue.pop(), Some(audio_data));
    }

    #[test]
    fn test_queue_stats() {
        let queue = SPSCQueue::new(4).unwrap();
        
        // Fill queue
        for i in 0..3 {
            queue.push(i).unwrap();
        }
        
        // Try to push when full
        assert!(queue.push(99).is_err());
        
        let stats = queue.stats();
        assert_eq!(stats.push_success_rate(), 0.75); // 3 out of 4 pushes succeeded
        
        // Pop some items
        queue.pop();
        queue.pop();
        queue.pop();
        queue.pop(); // This should fail
        
        assert_eq!(stats.pop_success_rate(), 0.75); // 3 out of 4 pops succeeded
    }

    #[test]
    fn test_memory_usage_calculation() {
        let config = MemoryConfig::default();
        let manager = StreamingBufferManager::new(config).unwrap();
        
        let memory_usage = manager.total_memory_usage();
        assert!(memory_usage > 0);
    }

    #[test]
    fn test_power_of_two_capacity() {
        // Valid power of two
        assert!(LockFreeRingBuffer::<i32>::new(8).is_ok());
        assert!(LockFreeRingBuffer::<i32>::new(16).is_ok());
        
        // Invalid non-power of two
        assert!(LockFreeRingBuffer::<i32>::new(10).is_err());
        assert!(LockFreeRingBuffer::<i32>::new(15).is_err());
    }
}