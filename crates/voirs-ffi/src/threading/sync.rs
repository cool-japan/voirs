//! Synchronization primitives for thread-safe VoiRS operations
//!
//! This module provides advanced synchronization utilities including
//! reader-writer locks, atomic operations, condition variables, and barriers.

use parking_lot::{Condvar, Mutex, RwLock};
use std::sync::{
    atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

/// High-performance reader-writer lock for shared audio data
pub struct VoirsRwLock<T> {
    inner: RwLock<T>,
    read_count: AtomicUsize,
    write_count: AtomicUsize,
}

impl<T> VoirsRwLock<T> {
    /// Create a new reader-writer lock
    pub fn new(data: T) -> Self {
        Self {
            inner: RwLock::new(data),
            read_count: AtomicUsize::new(0),
            write_count: AtomicUsize::new(0),
        }
    }

    /// Acquire a read lock
    pub fn read(&self) -> parking_lot::RwLockReadGuard<T> {
        self.read_count.fetch_add(1, Ordering::Relaxed);
        self.inner.read()
    }

    /// Acquire a write lock
    pub fn write(&self) -> parking_lot::RwLockWriteGuard<T> {
        self.write_count.fetch_add(1, Ordering::Relaxed);
        self.inner.write()
    }

    /// Try to acquire a read lock without blocking
    pub fn try_read(&self) -> Option<parking_lot::RwLockReadGuard<T>> {
        self.inner.try_read().map(|guard| {
            self.read_count.fetch_add(1, Ordering::Relaxed);
            guard
        })
    }

    /// Try to acquire a write lock without blocking
    pub fn try_write(&self) -> Option<parking_lot::RwLockWriteGuard<T>> {
        self.inner.try_write().map(|guard| {
            self.write_count.fetch_add(1, Ordering::Relaxed);
            guard
        })
    }

    /// Get read operation count
    pub fn read_count(&self) -> usize {
        self.read_count.load(Ordering::Relaxed)
    }

    /// Get write operation count
    pub fn write_count(&self) -> usize {
        self.write_count.load(Ordering::Relaxed)
    }
}

/// Atomic counter for tracking synthesis operations
pub struct AtomicCounter {
    value: AtomicU64,
    max_value: AtomicU64,
}

impl AtomicCounter {
    /// Create a new atomic counter
    pub fn new(initial: u64, max: u64) -> Self {
        Self {
            value: AtomicU64::new(initial),
            max_value: AtomicU64::new(max),
        }
    }

    /// Increment the counter, returning the new value
    pub fn increment(&self) -> Option<u64> {
        let current = self.value.load(Ordering::Acquire);
        let max = self.max_value.load(Ordering::Acquire);

        if current >= max {
            return None;
        }

        let new_value = self.value.fetch_add(1, Ordering::AcqRel) + 1;
        if new_value <= max {
            Some(new_value)
        } else {
            // Roll back if we exceeded max
            self.value.fetch_sub(1, Ordering::AcqRel);
            None
        }
    }

    /// Decrement the counter, returning the new value
    pub fn decrement(&self) -> u64 {
        self.value.fetch_sub(1, Ordering::AcqRel).saturating_sub(1)
    }

    /// Get the current value
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Acquire)
    }

    /// Reset the counter to zero
    pub fn reset(&self) {
        self.value.store(0, Ordering::Release);
    }

    /// Set the maximum value
    pub fn set_max(&self, max: u64) {
        self.max_value.store(max, Ordering::Release);
    }
}

/// Condition variable for coordinating synthesis operations
pub struct VoirsCondvar {
    condvar: Condvar,
    mutex: Mutex<CondvarState>,
}

struct CondvarState {
    notified: bool,
    wait_count: u32,
}

impl VoirsCondvar {
    /// Create a new condition variable
    pub fn new() -> Self {
        Self {
            condvar: Condvar::new(),
            mutex: Mutex::new(CondvarState {
                notified: false,
                wait_count: 0,
            }),
        }
    }

    /// Wait for notification
    pub fn wait(&self) {
        let mut state = self.mutex.lock();
        state.wait_count += 1;

        while !state.notified {
            self.condvar.wait(&mut state);
        }

        state.wait_count -= 1;
        if state.wait_count == 0 {
            state.notified = false;
        }
    }

    /// Wait for notification with timeout
    pub fn wait_timeout(&self, timeout: Duration) -> bool {
        let mut state = self.mutex.lock();
        state.wait_count += 1;

        let start = Instant::now();
        while !state.notified && start.elapsed() < timeout {
            if self
                .condvar
                .wait_for(&mut state, timeout - start.elapsed())
                .timed_out()
            {
                state.wait_count -= 1;
                return false;
            }
        }

        state.wait_count -= 1;
        if state.wait_count == 0 {
            state.notified = false;
        }

        true
    }

    /// Notify one waiting thread
    pub fn notify_one(&self) {
        let mut state = self.mutex.lock();
        state.notified = true;
        self.condvar.notify_one();
    }

    /// Notify all waiting threads
    pub fn notify_all(&self) {
        let mut state = self.mutex.lock();
        state.notified = true;
        self.condvar.notify_all();
    }

    /// Get the number of waiting threads
    pub fn wait_count(&self) -> u32 {
        self.mutex.lock().wait_count
    }
}

/// Synchronization barrier for coordinating multiple threads
pub struct VoirsBarrier {
    inner: Arc<BarrierInner>,
}

struct BarrierInner {
    mutex: Mutex<BarrierState>,
    condvar: Condvar,
    count: u32,
}

struct BarrierState {
    waiting: u32,
    generation: u32,
}

impl VoirsBarrier {
    /// Create a new barrier for the specified number of threads
    pub fn new(count: u32) -> Self {
        Self {
            inner: Arc::new(BarrierInner {
                mutex: Mutex::new(BarrierState {
                    waiting: 0,
                    generation: 0,
                }),
                condvar: Condvar::new(),
                count,
            }),
        }
    }

    /// Wait for all threads to reach the barrier
    pub fn wait(&self) -> bool {
        let mut state = self.inner.mutex.lock();
        let generation = state.generation;
        state.waiting += 1;

        if state.waiting == self.inner.count {
            // Last thread to arrive
            state.waiting = 0;
            state.generation = state.generation.wrapping_add(1);
            self.inner.condvar.notify_all();
            true // This thread is the leader
        } else {
            // Wait for other threads
            while state.generation == generation {
                self.inner.condvar.wait(&mut state);
            }
            false // This thread is a follower
        }
    }

    /// Get the barrier count
    pub fn count(&self) -> u32 {
        self.inner.count
    }

    /// Get the number of threads currently waiting
    pub fn waiting(&self) -> u32 {
        self.inner.mutex.lock().waiting
    }
}

impl Clone for VoirsBarrier {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Atomic flag for cancellation support
pub struct CancellationToken {
    cancelled: AtomicBool,
    reason: Mutex<Option<String>>,
}

impl CancellationToken {
    /// Create a new cancellation token
    pub fn new() -> Self {
        Self {
            cancelled: AtomicBool::new(false),
            reason: Mutex::new(None),
        }
    }

    /// Cancel the operation with an optional reason
    pub fn cancel(&self, reason: Option<String>) {
        self.cancelled.store(true, Ordering::Release);
        if let Some(reason) = reason {
            *self.reason.lock() = Some(reason);
        }
    }

    /// Check if the operation is cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Get the cancellation reason if available
    pub fn reason(&self) -> Option<String> {
        self.reason.lock().clone()
    }

    /// Reset the cancellation state
    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::Release);
        *self.reason.lock() = None;
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

/// Atomic statistics for monitoring thread performance
#[derive(Default)]
pub struct ThreadStats {
    pub operations_started: AtomicU64,
    pub operations_completed: AtomicU64,
    pub operations_failed: AtomicU64,
    pub total_processing_time_ns: AtomicU64,
    pub peak_memory_usage: AtomicUsize,
    pub current_memory_usage: AtomicUsize,
}

impl ThreadStats {
    /// Create new thread statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record operation start
    pub fn start_operation(&self) {
        self.operations_started.fetch_add(1, Ordering::Relaxed);
    }

    /// Record operation completion
    pub fn complete_operation(&self, processing_time_ns: u64) {
        self.operations_completed.fetch_add(1, Ordering::Relaxed);
        self.total_processing_time_ns
            .fetch_add(processing_time_ns, Ordering::Relaxed);
    }

    /// Record operation failure
    pub fn fail_operation(&self) {
        self.operations_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Update memory usage
    pub fn update_memory_usage(&self, current: usize) {
        self.current_memory_usage.store(current, Ordering::Relaxed);

        // Update peak if necessary
        let mut peak = self.peak_memory_usage.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_memory_usage.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => peak = actual,
            }
        }
    }

    /// Get average processing time in nanoseconds
    pub fn average_processing_time_ns(&self) -> f64 {
        let completed = self.operations_completed.load(Ordering::Relaxed);
        if completed == 0 {
            0.0
        } else {
            self.total_processing_time_ns.load(Ordering::Relaxed) as f64 / completed as f64
        }
    }

    /// Get success rate as a percentage
    pub fn success_rate(&self) -> f64 {
        let completed = self.operations_completed.load(Ordering::Relaxed);
        let failed = self.operations_failed.load(Ordering::Relaxed);
        let total = completed + failed;

        if total == 0 {
            100.0
        } else {
            (completed as f64 / total as f64) * 100.0
        }
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.operations_started.store(0, Ordering::Relaxed);
        self.operations_completed.store(0, Ordering::Relaxed);
        self.operations_failed.store(0, Ordering::Relaxed);
        self.total_processing_time_ns.store(0, Ordering::Relaxed);
        self.peak_memory_usage.store(0, Ordering::Relaxed);
        self.current_memory_usage.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_voirs_rwlock() {
        let lock = VoirsRwLock::new(42);

        // Test read access
        {
            let guard = lock.read();
            assert_eq!(*guard, 42);
            assert_eq!(lock.read_count(), 1);
        }

        // Test write access
        {
            let mut guard = lock.write();
            *guard = 100;
            assert_eq!(lock.write_count(), 1);
        }

        // Verify write worked
        {
            let guard = lock.read();
            assert_eq!(*guard, 100);
        }
    }

    #[test]
    fn test_atomic_counter() {
        let counter = AtomicCounter::new(0, 5);

        // Test increment
        assert_eq!(counter.increment(), Some(1));
        assert_eq!(counter.increment(), Some(2));
        assert_eq!(counter.get(), 2);

        // Test decrement
        assert_eq!(counter.decrement(), 1);
        assert_eq!(counter.get(), 1);

        // Test max limit
        for _ in 0..5 {
            counter.increment();
        }
        assert_eq!(counter.increment(), None); // Should fail at max
    }

    #[test]
    fn test_condvar() {
        use std::sync::Arc;

        let condvar = Arc::new(VoirsCondvar::new());
        let condvar_clone = Arc::clone(&condvar);

        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            condvar_clone.notify_one();
        });

        let start = Instant::now();
        condvar.wait();
        let elapsed = start.elapsed();

        handle.join().unwrap();
        assert!(elapsed >= Duration::from_millis(40));
    }

    #[test]
    fn test_barrier() {
        let barrier = VoirsBarrier::new(3);
        let mut handles = vec![];

        for i in 0..3 {
            let barrier_clone = barrier.clone();
            handles.push(thread::spawn(move || {
                thread::sleep(Duration::from_millis(i * 10));
                barrier_clone.wait()
            }));
        }

        let results: Vec<bool> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // Exactly one thread should be the leader
        assert_eq!(results.iter().filter(|&&x| x).count(), 1);
    }

    #[test]
    fn test_cancellation_token() {
        let token = CancellationToken::new();

        assert!(!token.is_cancelled());

        token.cancel(Some("Test cancellation".to_string()));
        assert!(token.is_cancelled());
        assert_eq!(token.reason(), Some("Test cancellation".to_string()));

        token.reset();
        assert!(!token.is_cancelled());
        assert_eq!(token.reason(), None);
    }

    #[test]
    fn test_thread_stats() {
        let stats = ThreadStats::new();

        stats.start_operation();
        stats.complete_operation(1000);
        stats.update_memory_usage(1024);

        assert_eq!(stats.operations_started.load(Ordering::Relaxed), 1);
        assert_eq!(stats.operations_completed.load(Ordering::Relaxed), 1);
        assert_eq!(stats.average_processing_time_ns(), 1000.0);
        assert_eq!(stats.success_rate(), 100.0);
    }
}
