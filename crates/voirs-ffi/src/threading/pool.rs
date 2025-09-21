//! Thread pool management for VoiRS FFI operations
//!
//! This module provides a high-performance thread pool implementation
//! optimized for speech synthesis workloads with dynamic load balancing.

use crate::threading::sync::{CancellationToken, ThreadStats};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// High-performance thread pool for VoiRS operations
pub struct VoirsThreadPool {
    workers: Vec<Worker>,
    sender: flume::Sender<Job>,
    receiver: flume::Receiver<Job>,
    shutdown: Arc<AtomicBool>,
    stats: Arc<ThreadStats>,
    config: PoolConfig,
}

/// Thread pool configuration
#[derive(Clone)]
pub struct PoolConfig {
    /// Minimum number of threads
    pub min_threads: usize,
    /// Maximum number of threads
    pub max_threads: usize,
    /// Thread idle timeout before shutdown
    pub idle_timeout: Duration,
    /// Queue size limit
    pub queue_size_limit: usize,
    /// Enable work stealing
    pub work_stealing: bool,
    /// Thread priority (0-99, higher is more important)
    pub thread_priority: u8,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            min_threads: 2,
            max_threads: num_cpus::get().max(4),
            idle_timeout: Duration::from_secs(30),
            queue_size_limit: 1000,
            work_stealing: true,
            thread_priority: 50,
        }
    }
}

type Job = Box<dyn FnOnce() + Send + 'static>;

struct Worker {
    id: usize,
    thread: Option<JoinHandle<()>>,
    local_queue: Arc<Mutex<VecDeque<Job>>>,
    active: Arc<AtomicBool>,
}

impl VoirsThreadPool {
    /// Create a new thread pool with default configuration
    pub fn new() -> Self {
        Self::with_config(PoolConfig::default())
    }

    /// Create a new thread pool with custom configuration
    pub fn with_config(config: PoolConfig) -> Self {
        let (sender, receiver) = flume::bounded(config.queue_size_limit);
        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(ThreadStats::new());

        let mut workers = Vec::with_capacity(config.min_threads);

        // Create initial worker threads
        for id in 0..config.min_threads {
            let worker = Worker::new(
                id,
                Arc::clone(&shutdown),
                receiver.clone(),
                Arc::clone(&stats),
                config.clone(),
            );
            workers.push(worker);
        }

        Self {
            workers,
            sender,
            receiver,
            shutdown,
            stats,
            config,
        }
    }

    /// Execute a closure on the thread pool
    pub fn execute<F>(&self, job: F) -> Result<(), ThreadPoolError>
    where
        F: FnOnce() + Send + 'static,
    {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(ThreadPoolError::ShutdownInProgress);
        }

        let job = Box::new(job);

        // Try work stealing first if enabled
        if self.config.work_stealing {
            if let Some(worker) = self.find_idle_worker() {
                let mut local_queue = worker.local_queue.lock();
                if local_queue.len() < 10 {
                    // Avoid overloading local queues
                    local_queue.push_back(job);
                    return Ok(());
                }
            }
        }

        // Fall back to global queue
        self.sender.try_send(job).map_err(|_| {
            if self.sender.is_full() {
                ThreadPoolError::QueueFull
            } else {
                ThreadPoolError::SendError
            }
        })?;

        // Check if we need to spawn more workers
        self.maybe_spawn_worker();

        Ok(())
    }

    /// Execute a closure with cancellation support
    pub fn execute_with_cancellation<F>(
        &self,
        job: F,
        cancellation_token: Arc<CancellationToken>,
    ) -> Result<(), ThreadPoolError>
    where
        F: FnOnce() + Send + 'static,
    {
        self.execute(move || {
            if !cancellation_token.is_cancelled() {
                job();
            }
        })
    }

    /// Get the number of active workers
    pub fn active_workers(&self) -> usize {
        self.workers
            .iter()
            .filter(|w| w.active.load(Ordering::Relaxed))
            .count()
    }

    /// Get the number of queued jobs
    pub fn queued_jobs(&self) -> usize {
        self.receiver.len()
    }

    /// Get thread pool statistics
    pub fn stats(&self) -> &ThreadStats {
        &self.stats
    }

    /// Shutdown the thread pool gracefully
    pub fn shutdown(&mut self) -> Result<(), ThreadPoolError> {
        self.shutdown.store(true, Ordering::Release);

        // Close the sender to prevent new jobs
        drop(self.sender.clone());

        // Wait for all workers to finish
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                thread.join().map_err(|_| ThreadPoolError::JoinError)?;
            }
        }

        Ok(())
    }

    /// Force shutdown the thread pool immediately
    pub fn force_shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        // Don't wait for workers to finish
        self.workers.clear();
    }

    /// Resize the thread pool
    pub fn resize(&mut self, new_size: usize) -> Result<(), ThreadPoolError> {
        if new_size == 0 {
            return Err(ThreadPoolError::InvalidPoolSize);
        }

        let current_size = self.workers.len();

        if new_size > current_size {
            // Add more workers
            for id in current_size..new_size {
                let worker = Worker::new(
                    id,
                    Arc::clone(&self.shutdown),
                    self.receiver.clone(),
                    Arc::clone(&self.stats),
                    self.config.clone(),
                );
                self.workers.push(worker);
            }
        } else if new_size < current_size {
            // Remove workers (they will shut down naturally when idle)
            for worker in self.workers.drain(new_size..) {
                worker.active.store(false, Ordering::Release);
            }
        }

        self.config.max_threads = new_size;
        Ok(())
    }

    fn find_idle_worker(&self) -> Option<&Worker> {
        self.workers
            .iter()
            .find(|w| w.active.load(Ordering::Relaxed) && w.local_queue.lock().len() < 5)
    }

    fn maybe_spawn_worker(&self) {
        let current_workers = self.workers.len();
        let queue_size = self.receiver.len();

        // Spawn new worker if queue is getting full and we haven't reached max
        if queue_size > current_workers * 2 && current_workers < self.config.max_threads {
            let worker = Worker::new(
                current_workers,
                Arc::clone(&self.shutdown),
                self.receiver.clone(),
                Arc::clone(&self.stats),
                self.config.clone(),
            );

            // This would require making workers mutable, which complicates the API
            // For now, we'll use a different approach in a full implementation
        }
    }
}

impl Worker {
    fn new(
        id: usize,
        shutdown: Arc<AtomicBool>,
        receiver: flume::Receiver<Job>,
        stats: Arc<ThreadStats>,
        config: PoolConfig,
    ) -> Self {
        let local_queue = Arc::new(Mutex::new(VecDeque::new()));
        let active = Arc::new(AtomicBool::new(true));

        let local_queue_clone = Arc::clone(&local_queue);
        let active_clone = Arc::clone(&active);

        let thread = thread::spawn(move || {
            Worker::run(
                id,
                shutdown,
                receiver,
                local_queue_clone,
                active_clone,
                stats,
                config,
            );
        });

        Self {
            id,
            thread: Some(thread),
            local_queue,
            active,
        }
    }

    fn run(
        id: usize,
        shutdown: Arc<AtomicBool>,
        receiver: flume::Receiver<Job>,
        local_queue: Arc<Mutex<VecDeque<Job>>>,
        active: Arc<AtomicBool>,
        stats: Arc<ThreadStats>,
        config: PoolConfig,
    ) {
        log::debug!("Worker {} starting", id);

        let mut last_activity = Instant::now();

        while !shutdown.load(Ordering::Acquire) {
            let job = {
                // Try local queue first
                let mut local = local_queue.lock();
                if let Some(job) = local.pop_front() {
                    Some(job)
                } else {
                    drop(local);
                    // Try global queue
                    match receiver.recv_timeout(Duration::from_millis(100)) {
                        Ok(job) => Some(job),
                        Err(flume::RecvTimeoutError::Timeout) => None,
                        Err(flume::RecvTimeoutError::Disconnected) => break,
                    }
                }
            };

            if let Some(job) = job {
                last_activity = Instant::now();
                active.store(true, Ordering::Release);

                stats.start_operation();
                let start = Instant::now();

                // Execute the job
                job();

                let elapsed = start.elapsed();
                stats.complete_operation(elapsed.as_nanos() as u64);
            } else {
                // Check for idle timeout
                if last_activity.elapsed() > config.idle_timeout {
                    log::debug!("Worker {} idle timeout", id);
                    active.store(false, Ordering::Release);
                    break;
                }
            }
        }

        log::debug!("Worker {} shutting down", id);
    }
}

/// Thread pool error types
#[derive(Debug, thiserror::Error)]
pub enum ThreadPoolError {
    #[error("Thread pool is shutting down")]
    ShutdownInProgress,
    #[error("Job queue is full")]
    QueueFull,
    #[error("Failed to send job to worker")]
    SendError,
    #[error("Failed to join worker thread")]
    JoinError,
    #[error("Invalid pool size")]
    InvalidPoolSize,
}

impl Default for VoirsThreadPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for VoirsThreadPool {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

/// Global thread pool instance
static GLOBAL_POOL: std::sync::OnceLock<Mutex<VoirsThreadPool>> = std::sync::OnceLock::new();

/// Get or initialize the global thread pool
pub fn global_pool() -> &'static Mutex<VoirsThreadPool> {
    GLOBAL_POOL.get_or_init(|| Mutex::new(VoirsThreadPool::new()))
}

/// Execute a job on the global thread pool
pub fn execute_global<F>(job: F) -> Result<(), ThreadPoolError>
where
    F: FnOnce() + Send + 'static,
{
    global_pool().lock().execute(job)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn test_thread_pool_creation() {
        let pool = VoirsThreadPool::new();
        assert!(pool.active_workers() > 0);
        assert_eq!(pool.queued_jobs(), 0);
    }

    #[test]
    fn test_job_execution() {
        let pool = VoirsThreadPool::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        pool.execute(move || {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();

        // Wait a bit for job to complete (increased timeout for reliability)
        std::thread::sleep(Duration::from_millis(500));
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_multiple_jobs() {
        let pool = VoirsThreadPool::new();
        let counter = Arc::new(AtomicUsize::new(0));

        for _ in 0..10 {
            let counter_clone = Arc::clone(&counter);
            pool.execute(move || {
                counter_clone.fetch_add(1, Ordering::Relaxed);
            })
            .unwrap();
        }

        // Wait for jobs to complete
        std::thread::sleep(Duration::from_millis(500));
        assert_eq!(counter.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn test_cancellation() {
        let pool = VoirsThreadPool::new();
        let token = Arc::new(CancellationToken::new());
        let counter = Arc::new(AtomicUsize::new(0));

        // Cancel before execution
        token.cancel(None);

        let counter_clone = Arc::clone(&counter);
        pool.execute_with_cancellation(
            move || {
                counter_clone.fetch_add(1, Ordering::Relaxed);
            },
            token,
        )
        .unwrap();

        // Wait and verify job was cancelled
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_pool_resize() {
        let mut pool = VoirsThreadPool::new();
        let initial_size = pool.workers.len();

        pool.resize(initial_size + 2).unwrap();
        assert_eq!(pool.workers.len(), initial_size + 2);

        pool.resize(initial_size).unwrap();
        assert_eq!(pool.workers.len(), initial_size);
    }

    #[test]
    fn test_global_pool() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        execute_global(move || {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();

        std::thread::sleep(Duration::from_millis(500));
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }
}
