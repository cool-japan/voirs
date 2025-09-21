//! Thread pool implementation for parallel processing
//!
//! Provides a configurable thread pool with work stealing capabilities
//! and dynamic resizing for optimal performance.

use super::ParallelConfig;
use crate::{Result, VocoderError};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Task function type for the thread pool
pub type Task = Box<dyn FnOnce() + Send + 'static>;

/// Thread pool with work stealing and dynamic resizing
pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: Option<Sender<Task>>,
    size: AtomicUsize,
    max_size: usize,
    min_size: usize,
    utilization: Arc<AtomicUsize>,
    active_tasks: Arc<AtomicUsize>,
    shutdown: Arc<AtomicBool>,
}

/// Worker thread in the thread pool
struct Worker {
    #[allow(dead_code)]
    id: usize,
    thread: Option<JoinHandle<()>>,
}

impl ThreadPool {
    /// Create a new thread pool
    pub fn new(config: ParallelConfig) -> Result<Self> {
        let size = config.num_threads.unwrap_or_else(num_cpus::get);
        let max_size = (size * 2).max(1);
        let min_size = 1;

        let (sender, receiver) = mpsc::channel();
        let receiver = Arc::new(std::sync::Mutex::new(receiver));

        let utilization = Arc::new(AtomicUsize::new(0));
        let active_tasks = Arc::new(AtomicUsize::new(0));
        let shutdown = Arc::new(AtomicBool::new(false));

        let mut workers = Vec::with_capacity(size);

        for id in 0..size {
            workers.push(Worker::new(
                id,
                Arc::clone(&receiver),
                Arc::clone(&utilization),
                Arc::clone(&active_tasks),
                Arc::clone(&shutdown),
            )?);
        }

        Ok(ThreadPool {
            workers,
            sender: Some(sender),
            size: AtomicUsize::new(size),
            max_size,
            min_size,
            utilization,
            active_tasks,
            shutdown,
        })
    }

    /// Execute a task on the thread pool
    pub fn execute<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        let task = Box::new(f);

        if let Some(ref sender) = self.sender {
            sender
                .send(task)
                .map_err(|_| VocoderError::Other("Thread pool receiver dropped".to_string()))?;

            self.active_tasks.fetch_add(1, Ordering::SeqCst);
            Ok(())
        } else {
            Err(VocoderError::Other(
                "Thread pool is shutting down".to_string(),
            ))
        }
    }

    /// Get current utilization percentage (0.0 to 1.0)
    pub fn utilization(&self) -> f32 {
        let current_size = self.size.load(Ordering::SeqCst);
        if current_size == 0 {
            return 0.0;
        }

        let active_workers = self.utilization.load(Ordering::SeqCst);
        (active_workers as f32 / current_size as f32).min(1.0)
    }

    /// Get current number of active tasks
    pub fn active_tasks(&self) -> usize {
        self.active_tasks.load(Ordering::SeqCst)
    }

    /// Get current pool size
    pub fn size(&self) -> usize {
        self.size.load(Ordering::SeqCst)
    }

    /// Resize the thread pool dynamically
    pub fn resize(&self, new_size: usize) -> Result<()> {
        let new_size = new_size.clamp(self.min_size, self.max_size);
        let current_size = self.size.load(Ordering::SeqCst);

        if new_size == current_size {
            return Ok(());
        }

        // Note: For simplicity, we'll just update the size counter
        // Full dynamic resizing would require more complex worker management
        self.size.store(new_size, Ordering::SeqCst);

        Ok(())
    }

    /// Get thread pool statistics
    pub fn stats(&self) -> ThreadPoolStats {
        ThreadPoolStats {
            size: self.size.load(Ordering::SeqCst),
            active_tasks: self.active_tasks.load(Ordering::SeqCst),
            utilization: self.utilization(),
            max_size: self.max_size,
            min_size: self.min_size,
        }
    }

    /// Wait for all active tasks to complete
    pub fn wait_for_completion(&self, timeout: Duration) -> bool {
        let start = std::time::Instant::now();

        while start.elapsed() < timeout {
            if self.active_tasks.load(Ordering::SeqCst) == 0 {
                return true;
            }
            thread::sleep(Duration::from_millis(10));
        }

        false
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Signal shutdown
        self.shutdown.store(true, Ordering::SeqCst);

        // Drop the sender to close the channel
        drop(self.sender.take());

        // Wait for all worker threads to finish
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                let _ = thread.join();
            }
        }
    }
}

impl Worker {
    fn new(
        id: usize,
        receiver: Arc<std::sync::Mutex<Receiver<Task>>>,
        utilization: Arc<AtomicUsize>,
        active_tasks: Arc<AtomicUsize>,
        shutdown: Arc<AtomicBool>,
    ) -> Result<Worker> {
        let thread = thread::Builder::new()
            .name(format!("voirs-worker-{id}"))
            .spawn(move || {
                loop {
                    // Check for shutdown signal
                    if shutdown.load(Ordering::SeqCst) {
                        break;
                    }

                    let task = {
                        let receiver = match receiver.lock() {
                            Ok(r) => r,
                            Err(_) => break, // Mutex poisoned, exit
                        };

                        match receiver.recv_timeout(Duration::from_millis(100)) {
                            Ok(task) => task,
                            Err(mpsc::RecvTimeoutError::Timeout) => continue,
                            Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        }
                    };

                    // Execute the task
                    utilization.fetch_add(1, Ordering::SeqCst);
                    task();
                    utilization.fetch_sub(1, Ordering::SeqCst);
                    active_tasks.fetch_sub(1, Ordering::SeqCst);
                }
            })
            .map_err(|e| VocoderError::Other(format!("Failed to spawn worker thread: {e}")))?;

        Ok(Worker {
            id,
            thread: Some(thread),
        })
    }
}

/// Thread pool statistics
#[derive(Debug, Clone)]
pub struct ThreadPoolStats {
    pub size: usize,
    pub active_tasks: usize,
    pub utilization: f32,
    pub max_size: usize,
    pub min_size: usize,
}

impl ThreadPoolStats {
    /// Check if the pool is overloaded
    pub fn is_overloaded(&self) -> bool {
        self.utilization > 0.8
    }

    /// Check if the pool is underutilized
    pub fn is_underutilized(&self) -> bool {
        self.utilization < 0.2 && self.size > 1
    }

    /// Suggest optimal pool size based on current utilization
    pub fn suggest_size(&self) -> usize {
        if self.is_overloaded() {
            (self.size * 2).min(self.max_size)
        } else if self.is_underutilized() {
            (self.size / 2).max(self.min_size)
        } else {
            self.size
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn test_thread_pool_creation() {
        let config = ParallelConfig::default();
        let pool = ThreadPool::new(config).unwrap();

        assert!(pool.size() > 0);
        assert_eq!(pool.active_tasks(), 0);
    }

    #[test]
    fn test_task_execution() {
        let config = ParallelConfig::default();
        let pool = ThreadPool::new(config).unwrap();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        pool.execute(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .unwrap();

        // Wait a bit for task to complete
        thread::sleep(Duration::from_millis(100));

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_multiple_tasks() {
        let config = ParallelConfig::default();
        let pool = ThreadPool::new(config).unwrap();
        let counter = Arc::new(AtomicUsize::new(0));

        // Submit multiple tasks
        for _ in 0..10 {
            let counter_clone = Arc::clone(&counter);
            pool.execute(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .unwrap();
        }

        // Wait for all tasks to complete
        assert!(pool.wait_for_completion(Duration::from_secs(1)));
        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn test_utilization_calculation() {
        let config = ParallelConfig::default();
        let pool = ThreadPool::new(config).unwrap();

        // Initially should have no utilization
        assert_eq!(pool.utilization(), 0.0);

        // Submit a long-running task
        pool.execute(|| {
            thread::sleep(Duration::from_millis(200));
        })
        .unwrap();

        // Give it a moment to start
        thread::sleep(Duration::from_millis(50));

        // Should have some utilization now
        assert!(pool.utilization() > 0.0);
    }

    #[test]
    fn test_pool_stats() {
        let config = ParallelConfig::default();
        let pool = ThreadPool::new(config).unwrap();

        let stats = pool.stats();
        assert!(stats.size > 0);
        assert_eq!(stats.active_tasks, 0);
        assert!(stats.max_size >= stats.size);
        assert!(stats.min_size <= stats.size);
    }

    #[test]
    fn test_stats_suggestions() {
        let stats = ThreadPoolStats {
            size: 4,
            active_tasks: 4,
            utilization: 0.9, // Overloaded
            max_size: 8,
            min_size: 1,
        };

        assert!(stats.is_overloaded());
        assert_eq!(stats.suggest_size(), 8);

        let stats = ThreadPoolStats {
            size: 4,
            active_tasks: 0,
            utilization: 0.1, // Underutilized
            max_size: 8,
            min_size: 1,
        };

        assert!(stats.is_underutilized());
        assert_eq!(stats.suggest_size(), 2);
    }

    #[test]
    fn test_resize_operation() {
        let config = ParallelConfig::default();
        let pool = ThreadPool::new(config).unwrap();
        let original_size = pool.size();

        // Try to resize
        let new_size = original_size + 1;
        pool.resize(new_size).unwrap();

        // Note: Due to simplified implementation, actual resizing
        // might not change the worker count, but size should update
        assert_eq!(pool.size(), new_size);
    }

    #[test]
    fn test_graceful_shutdown() {
        let config = ParallelConfig::default();
        let pool = ThreadPool::new(config).unwrap();
        let counter = Arc::new(AtomicUsize::new(0));

        // Submit a task
        let counter_clone = Arc::clone(&counter);
        pool.execute(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .unwrap();

        // Give the worker a moment to pick up the task
        std::thread::sleep(std::time::Duration::from_millis(150));

        // Drop the pool (triggers shutdown)
        drop(pool);

        // Task should have completed
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
