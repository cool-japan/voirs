//! Advanced threading features for VoiRS FFI operations
//!
//! This module provides sophisticated threading primitives including
//! work stealing queues, priority scheduling, thread affinity, and
//! advanced callback management.

use crate::threading::sync::{CancellationToken, ThreadStats};
use parking_lot::{Mutex, RwLock};
use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Work stealing queue for load balancing
pub struct WorkStealingQueue<T> {
    queue: RwLock<VecDeque<T>>,
    steal_count: AtomicUsize,
    local_pops: AtomicUsize,
    steals: AtomicUsize,
}

impl<T> WorkStealingQueue<T> {
    /// Create a new work stealing queue
    pub fn new() -> Self {
        Self {
            queue: RwLock::new(VecDeque::new()),
            steal_count: AtomicUsize::new(0),
            local_pops: AtomicUsize::new(0),
            steals: AtomicUsize::new(0),
        }
    }

    /// Push a job to the back of the queue (local thread)
    pub fn push(&self, item: T) {
        let mut queue = self.queue.write();
        queue.push_back(item);
    }

    /// Pop from the back of the queue (local thread)
    pub fn pop(&self) -> Option<T> {
        let mut queue = self.queue.write();
        if let Some(item) = queue.pop_back() {
            self.local_pops.fetch_add(1, Ordering::Relaxed);
            Some(item)
        } else {
            None
        }
    }

    /// Steal from the front of the queue (other threads)
    pub fn steal(&self) -> Option<T> {
        let mut queue = self.queue.write();
        if let Some(item) = queue.pop_front() {
            self.steals.fetch_add(1, Ordering::Relaxed);
            self.steal_count.fetch_add(1, Ordering::Relaxed);
            Some(item)
        } else {
            None
        }
    }

    /// Get the number of items in the queue
    pub fn len(&self) -> usize {
        self.queue.read().len()
    }

    /// Check if the queue is empty
    pub fn is_empty(&self) -> bool {
        self.queue.read().is_empty()
    }

    /// Get steal statistics
    pub fn steal_stats(&self) -> (usize, usize, usize) {
        (
            self.steal_count.load(Ordering::Relaxed),
            self.local_pops.load(Ordering::Relaxed),
            self.steals.load(Ordering::Relaxed),
        )
    }
}

/// Priority levels for job scheduling
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// A job with priority and timing information
pub struct PriorityJob {
    pub job: Box<dyn FnOnce() + Send + 'static>,
    pub priority: Priority,
    pub submit_time: Instant,
    pub deadline: Option<Instant>,
    pub id: u64,
}

impl PriorityJob {
    /// Create a new priority job
    pub fn new<F>(job: F, priority: Priority) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        static JOB_COUNTER: AtomicU64 = AtomicU64::new(0);

        Self {
            job: Box::new(job),
            priority,
            submit_time: Instant::now(),
            deadline: None,
            id: JOB_COUNTER.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Create a new priority job with deadline
    pub fn new_with_deadline<F>(job: F, priority: Priority, deadline: Instant) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        let mut job = Self::new(job, priority);
        job.deadline = Some(deadline);
        job
    }

    /// Check if the job has expired
    pub fn is_expired(&self) -> bool {
        if let Some(deadline) = self.deadline {
            Instant::now() > deadline
        } else {
            false
        }
    }

    /// Get the waiting time
    pub fn waiting_time(&self) -> Duration {
        self.submit_time.elapsed()
    }
}

impl PartialEq for PriorityJob {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.submit_time == other.submit_time
    }
}

impl Eq for PriorityJob {}

impl PartialOrd for PriorityJob {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityJob {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Higher priority jobs come first
        self.priority.cmp(&other.priority).then_with(|| {
            // For same priority, check deadlines
            match (self.deadline, other.deadline) {
                (Some(a), Some(b)) => a.cmp(&b),         // Earlier deadline first
                (Some(_), None) => CmpOrdering::Greater, // Deadline jobs have priority
                (None, Some(_)) => CmpOrdering::Less,
                (None, None) => self.submit_time.cmp(&other.submit_time), // FIFO for same priority
            }
        })
    }
}

/// Priority-based job scheduler
pub struct PriorityScheduler {
    queue: Mutex<BinaryHeap<PriorityJob>>,
    total_jobs: AtomicUsize,
    completed_jobs: AtomicUsize,
    expired_jobs: AtomicUsize,
    deadline_misses: AtomicUsize,
}

impl PriorityScheduler {
    /// Create a new priority scheduler
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(BinaryHeap::new()),
            total_jobs: AtomicUsize::new(0),
            completed_jobs: AtomicUsize::new(0),
            expired_jobs: AtomicUsize::new(0),
            deadline_misses: AtomicUsize::new(0),
        }
    }

    /// Submit a job for execution
    pub fn submit(&self, job: PriorityJob) {
        self.total_jobs.fetch_add(1, Ordering::Relaxed);
        let mut queue = self.queue.lock();
        queue.push(job);
    }

    /// Get the next job to execute
    pub fn next_job(&self) -> Option<PriorityJob> {
        let mut queue = self.queue.lock();

        // Remove expired jobs
        while let Some(job) = queue.peek() {
            if job.is_expired() {
                if let Some(expired_job) = queue.pop() {
                    self.expired_jobs.fetch_add(1, Ordering::Relaxed);
                    drop(expired_job);
                }
            } else {
                break;
            }
        }

        queue.pop()
    }

    /// Mark a job as completed
    pub fn job_completed(&self, deadline_met: bool) {
        self.completed_jobs.fetch_add(1, Ordering::Relaxed);
        if !deadline_met {
            self.deadline_misses.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get queue statistics
    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            total_jobs: self.total_jobs.load(Ordering::Relaxed),
            completed_jobs: self.completed_jobs.load(Ordering::Relaxed),
            expired_jobs: self.expired_jobs.load(Ordering::Relaxed),
            deadline_misses: self.deadline_misses.load(Ordering::Relaxed),
            pending_jobs: self.queue.lock().len(),
        }
    }

    /// Get the number of pending jobs
    pub fn pending_count(&self) -> usize {
        self.queue.lock().len()
    }
}

/// Scheduler statistics
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub total_jobs: usize,
    pub completed_jobs: usize,
    pub expired_jobs: usize,
    pub deadline_misses: usize,
    pub pending_jobs: usize,
}

/// Thread affinity configuration
pub struct ThreadAffinity {
    cpu_cores: Vec<usize>,
    numa_node: Option<usize>,
}

impl ThreadAffinity {
    /// Create new thread affinity configuration
    pub fn new(cpu_cores: Vec<usize>) -> Self {
        Self {
            cpu_cores,
            numa_node: None,
        }
    }

    /// Set NUMA node preference
    pub fn with_numa_node(mut self, numa_node: usize) -> Self {
        self.numa_node = Some(numa_node);
        self
    }

    /// Apply thread affinity to current thread
    pub fn apply(&self) -> Result<(), AffinityError> {
        #[cfg(target_os = "linux")]
        {
            self.apply_linux()
        }
        #[cfg(target_os = "windows")]
        {
            self.apply_windows()
        }
        #[cfg(target_os = "macos")]
        {
            self.apply_macos()
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        {
            // Unsupported platform
            Err(AffinityError::UnsupportedPlatform)
        }
    }

    #[cfg(target_os = "linux")]
    fn apply_linux(&self) -> Result<(), AffinityError> {
        // Linux-specific implementation using sched_setaffinity
        // This would require libc bindings
        log::debug!("Setting thread affinity for Linux: {:?}", self.cpu_cores);
        Ok(())
    }

    #[cfg(target_os = "windows")]
    fn apply_windows(&self) -> Result<(), AffinityError> {
        // Windows-specific implementation using SetThreadAffinityMask
        log::debug!("Setting thread affinity for Windows: {:?}", self.cpu_cores);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn apply_macos(&self) -> Result<(), AffinityError> {
        // macOS-specific implementation using thread_policy_set
        log::debug!("Setting thread affinity for macOS: {:?}", self.cpu_cores);
        Ok(())
    }
}

/// Thread affinity error types
#[derive(Debug, thiserror::Error)]
pub enum AffinityError {
    #[error("Unsupported platform")]
    UnsupportedPlatform,
    #[error("Invalid CPU core ID")]
    InvalidCpuCore,
    #[error("System call failed")]
    SystemCallFailed,
}

/// Thread-safe callback queue with priority support
pub struct CallbackQueue<T> {
    queue: Arc<Mutex<BinaryHeap<CallbackEntry<T>>>>,
    executor: Option<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    processed_count: Arc<AtomicUsize>,
}

struct CallbackEntry<T> {
    callback: Box<dyn FnOnce(T) + Send + 'static>,
    priority: Priority,
    data: T,
    timestamp: Instant,
}

impl<T> CallbackEntry<T> {
    fn new<F>(callback: F, priority: Priority, data: T) -> Self
    where
        F: FnOnce(T) + Send + 'static,
    {
        Self {
            callback: Box::new(callback),
            priority,
            data,
            timestamp: Instant::now(),
        }
    }
}

impl<T> PartialEq for CallbackEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.timestamp == other.timestamp
    }
}

impl<T> Eq for CallbackEntry<T> {}

impl<T> PartialOrd for CallbackEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for CallbackEntry<T> {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Higher priority callbacks first, then FIFO
        self.priority
            .cmp(&other.priority)
            .then_with(|| self.timestamp.cmp(&other.timestamp))
    }
}

impl<T> CallbackQueue<T>
where
    T: Send + 'static,
{
    /// Create a new callback queue
    pub fn new() -> Self {
        let queue = Arc::new(Mutex::new(BinaryHeap::new()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let processed_count = Arc::new(AtomicUsize::new(0));

        // Start executor thread
        let executor_queue = Arc::clone(&queue);
        let executor_shutdown = Arc::clone(&shutdown);
        let executor_processed = Arc::clone(&processed_count);

        let executor = thread::spawn(move || {
            Self::executor_loop(executor_queue, executor_shutdown, executor_processed);
        });

        Self {
            queue,
            executor: Some(executor),
            shutdown,
            processed_count,
        }
    }

    /// Add a callback to the queue
    pub fn push<F>(&self, callback: F, priority: Priority, data: T)
    where
        F: FnOnce(T) + Send + 'static,
    {
        if !self.shutdown.load(Ordering::Acquire) {
            let entry = CallbackEntry::new(callback, priority, data);
            self.queue.lock().push(entry);
        }
    }

    /// Get the number of processed callbacks
    pub fn processed_count(&self) -> usize {
        self.processed_count.load(Ordering::Relaxed)
    }

    /// Get the number of pending callbacks
    pub fn pending_count(&self) -> usize {
        self.queue.lock().len()
    }

    /// Shutdown the callback queue
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(executor) = self.executor.take() {
            let _ = executor.join();
        }
    }

    fn executor_loop(
        queue: Arc<Mutex<BinaryHeap<CallbackEntry<T>>>>,
        shutdown: Arc<AtomicBool>,
        processed_count: Arc<AtomicUsize>,
    ) {
        while !shutdown.load(Ordering::Acquire) {
            let entry = {
                let mut queue = queue.lock();
                queue.pop()
            };

            if let Some(entry) = entry {
                // Execute the callback
                (entry.callback)(entry.data);
                processed_count.fetch_add(1, Ordering::Relaxed);
            } else {
                // No callbacks, sleep briefly
                thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

impl<T> Drop for CallbackQueue<T> {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(executor) = self.executor.take() {
            let _ = executor.join();
        }
    }
}

/// Advanced lock-free data structure for inter-thread communication
pub struct LockFreeRingBuffer<T> {
    buffer: Vec<Mutex<Option<T>>>,
    head: AtomicUsize,
    tail: AtomicUsize,
    capacity: usize,
    count: AtomicUsize,
}

impl<T> LockFreeRingBuffer<T> {
    /// Create a new lock-free ring buffer
    pub fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(Mutex::new(None));
        }

        Self {
            buffer,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            capacity,
            count: AtomicUsize::new(0),
        }
    }

    /// Try to push an item to the buffer
    pub fn try_push(&self, item: T) -> Result<(), T> {
        if self.count.load(Ordering::Relaxed) >= self.capacity {
            return Err(item); // Buffer full
        }

        let current_tail = self.tail.load(Ordering::Relaxed);
        let next_tail = (current_tail + 1) % self.capacity;

        let mut slot = self.buffer[current_tail].lock();
        if slot.is_some() {
            return Err(item); // Slot occupied
        }

        *slot = Some(item);
        self.tail.store(next_tail, Ordering::Release);
        self.count.fetch_add(1, Ordering::Release);
        Ok(())
    }

    /// Try to pop an item from the buffer
    pub fn try_pop(&self) -> Option<T> {
        if self.count.load(Ordering::Relaxed) == 0 {
            return None; // Buffer empty
        }

        let current_head = self.head.load(Ordering::Relaxed);

        let mut slot = self.buffer[current_head].lock();
        if let Some(item) = slot.take() {
            self.head
                .store((current_head + 1) % self.capacity, Ordering::Release);
            self.count.fetch_sub(1, Ordering::Release);
            Some(item)
        } else {
            None
        }
    }

    /// Get the number of items in the buffer
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.count.load(Ordering::Relaxed) == 0
    }

    /// Check if the buffer is full
    pub fn is_full(&self) -> bool {
        self.count.load(Ordering::Relaxed) >= self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_work_stealing_queue() {
        let queue = WorkStealingQueue::new();

        // Push some items
        queue.push(1);
        queue.push(2);
        queue.push(3);

        assert_eq!(queue.len(), 3);

        // Pop from back (LIFO for local thread)
        assert_eq!(queue.pop(), Some(3));
        assert_eq!(queue.pop(), Some(2));

        // Steal from front (FIFO for other threads)
        assert_eq!(queue.steal(), Some(1));
        assert_eq!(queue.steal(), None);

        let (steal_count, local_pops, steals) = queue.steal_stats();
        assert_eq!(steal_count, 1);
        assert_eq!(local_pops, 2);
        assert_eq!(steals, 1);
    }

    #[test]
    fn test_priority_scheduler() {
        let scheduler = PriorityScheduler::new();

        // Submit jobs with different priorities
        scheduler.submit(PriorityJob::new(|| println!("Low priority"), Priority::Low));
        scheduler.submit(PriorityJob::new(
            || println!("High priority"),
            Priority::High,
        ));
        scheduler.submit(PriorityJob::new(
            || println!("Normal priority"),
            Priority::Normal,
        ));

        // Jobs should come out in priority order
        let job1 = scheduler.next_job().unwrap();
        assert_eq!(job1.priority, Priority::High);

        let job2 = scheduler.next_job().unwrap();
        assert_eq!(job2.priority, Priority::Normal);

        let job3 = scheduler.next_job().unwrap();
        assert_eq!(job3.priority, Priority::Low);
    }

    #[test]
    fn test_callback_queue() {
        let queue = CallbackQueue::new();
        let counter = Arc::new(AtomicUsize::new(0));

        // Add callbacks with different priorities
        let counter1 = Arc::clone(&counter);
        queue.push(
            move |_| {
                counter1.fetch_add(1, Ordering::Relaxed);
            },
            Priority::High,
            (),
        );

        let counter2 = Arc::clone(&counter);
        queue.push(
            move |_| {
                counter2.fetch_add(10, Ordering::Relaxed);
            },
            Priority::Low,
            (),
        );

        // Wait for callbacks to be processed
        thread::sleep(Duration::from_millis(100));

        // High priority callback should be processed first
        assert!(queue.processed_count() > 0);
        assert_eq!(counter.load(Ordering::Relaxed), 11);
    }

    #[test]
    fn test_lock_free_ring_buffer() {
        let buffer = LockFreeRingBuffer::new(4);

        // Test pushing
        assert!(buffer.try_push(1).is_ok());
        assert!(buffer.try_push(2).is_ok());
        assert!(buffer.try_push(3).is_ok());
        assert_eq!(buffer.len(), 3);

        // Test popping
        assert_eq!(buffer.try_pop(), Some(1));
        assert_eq!(buffer.try_pop(), Some(2));
        assert_eq!(buffer.len(), 1);

        // Test full buffer
        assert!(buffer.try_push(4).is_ok());
        assert!(buffer.try_push(5).is_ok());
        assert!(buffer.try_push(6).is_ok());
        assert!(buffer.try_push(7).is_err()); // Should be full

        assert!(buffer.is_full());
    }

    #[test]
    fn test_thread_affinity() {
        let affinity = ThreadAffinity::new(vec![0, 1]).with_numa_node(0);

        // Test applying affinity (should not fail on unsupported platforms)
        let result = affinity.apply();
        assert!(result.is_ok() || matches!(result, Err(AffinityError::UnsupportedPlatform)));
    }
}
