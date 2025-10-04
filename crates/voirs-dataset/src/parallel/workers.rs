//! Worker thread management for parallel dataset processing
//!
//! Provides configurable thread pools with work stealing queues,
//! load balancing, and resource monitoring.

use crate::{DatasetError, Result};
use scirs2_core::parallel_ops::{ThreadPool, ThreadPoolBuilder};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Worker pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    /// Number of worker threads (None = auto-detect)
    pub num_threads: Option<usize>,
    /// Stack size for worker threads (in bytes)
    pub stack_size: Option<usize>,
    /// Thread name prefix
    pub thread_name_prefix: String,
    /// Enable work stealing between threads
    pub work_stealing: bool,
    /// Thread priority (0 = normal, higher = more priority)
    pub thread_priority: i32,
    /// Panic handler strategy
    pub panic_handler: PanicHandlerStrategy,
}

/// Panic handling strategies for worker threads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PanicHandlerStrategy {
    /// Abort the entire process on panic
    Abort,
    /// Log and continue with remaining workers
    LogAndContinue,
    /// Restart the panicked worker
    RestartWorker,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            num_threads: None,                 // Auto-detect based on CPU cores
            stack_size: Some(8 * 1024 * 1024), // 8MB default stack
            thread_name_prefix: "dataset-worker".to_string(),
            work_stealing: true,
            thread_priority: 0,
            panic_handler: PanicHandlerStrategy::LogAndContinue,
        }
    }
}

/// Work item for the worker pool
pub trait WorkItem: Send + Sync {
    type Output: Send;
    type Error: Send + std::error::Error;

    /// Execute the work item
    fn execute(self) -> std::result::Result<Self::Output, Self::Error>;

    /// Get work item priority (higher = more urgent)
    fn priority(&self) -> u8 {
        0
    }

    /// Estimate work complexity (for load balancing)
    fn complexity(&self) -> u64 {
        1
    }
}

/// Resource usage statistics for worker monitoring
#[derive(Debug, Clone, Default)]
pub struct ResourceStats {
    /// Current memory usage (bytes)
    pub memory_usage: u64,
    /// Current CPU utilization (0.0 - 1.0)
    pub cpu_utilization: f64,
    /// Number of active workers
    pub active_workers: usize,
    /// Number of queued tasks
    pub queued_tasks: usize,
    /// Tasks completed per second
    pub throughput: f64,
    /// Average task execution time (seconds)
    pub avg_execution_time: f64,
}

/// Load balancing strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin task assignment
    RoundRobin,
    /// Least loaded worker first
    LeastLoaded,
    /// Work complexity-based assignment
    ComplexityAware,
    /// Priority-based assignment
    PriorityBased,
}

/// Worker pool for parallel dataset processing
pub struct WorkerPool {
    /// Rayon thread pool
    thread_pool: ThreadPool,
    /// Configuration
    config: WorkerConfig,
    /// Resource usage statistics
    stats: Arc<RwLock<ResourceStats>>,
    /// Task counter for throughput calculation
    task_counter: Arc<AtomicUsize>,
    /// Start time for throughput calculation
    start_time: Instant,
    /// Load balancing strategy
    load_balancing: LoadBalancingStrategy,
}

impl WorkerPool {
    /// Create a new worker pool with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(WorkerConfig::default())
    }

    /// Create a new worker pool with custom configuration
    pub fn with_config(config: WorkerConfig) -> Result<Self> {
        let num_threads = config.num_threads.unwrap_or_else(|| {
            let cores = num_cpus::get();
            debug!("Auto-detected {} CPU cores for worker pool", cores);
            cores
        });

        let thread_name_prefix = config.thread_name_prefix.clone();
        let mut builder = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(move |i| format!("{thread_name_prefix}-{i}"));

        if let Some(stack_size) = config.stack_size {
            builder = builder.stack_size(stack_size);
        }

        // Set up panic handler
        match config.panic_handler {
            PanicHandlerStrategy::Abort => {
                builder = builder.panic_handler(|_| std::process::abort());
            }
            PanicHandlerStrategy::LogAndContinue => {
                builder = builder.panic_handler(|err| {
                    warn!("Worker thread panicked: {:?}", err);
                });
            }
            PanicHandlerStrategy::RestartWorker => {
                builder = builder.panic_handler(|err| {
                    warn!("Worker thread panicked, will restart: {:?}", err);
                    // Note: Rayon doesn't support automatic worker restart
                    // This would need custom implementation
                });
            }
        }

        let thread_pool = builder.build().map_err(|e| {
            DatasetError::ProcessingError(format!("Failed to create thread pool: {e}"))
        })?;

        info!("Created worker pool with {} threads", num_threads);

        Ok(Self {
            thread_pool,
            config,
            stats: Arc::new(RwLock::new(ResourceStats::default())),
            task_counter: Arc::new(AtomicUsize::new(0)),
            start_time: Instant::now(),
            load_balancing: LoadBalancingStrategy::LeastLoaded,
        })
    }

    /// Execute a single work item
    pub async fn execute<W>(&self, work: W) -> Result<W::Output>
    where
        W: WorkItem + 'static,
        W::Output: 'static,
        W::Error: 'static + std::fmt::Debug,
    {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let task_counter = Arc::clone(&self.task_counter);
        let stats = Arc::clone(&self.stats);

        self.thread_pool.spawn(move || {
            let start_time = Instant::now();
            let result = work.execute();
            let execution_time = start_time.elapsed();

            // Update statistics
            task_counter.fetch_add(1, Ordering::Relaxed);

            // Update statistics synchronously
            if let Ok(mut stats_guard) = stats.try_write() {
                stats_guard.avg_execution_time =
                    (stats_guard.avg_execution_time + execution_time.as_secs_f64()) / 2.0;
            }

            let _ = tx.send(result);
        });

        rx.await
            .map_err(|_| DatasetError::ProcessingError("Worker task was cancelled".to_string()))?
            .map_err(|e| DatasetError::ProcessingError(format!("Work item failed: {e:?}")))
    }

    /// Execute multiple work items in parallel
    pub async fn execute_batch<W>(&self, work_items: Vec<W>) -> Result<Vec<W::Output>>
    where
        W: WorkItem + 'static,
        W::Output: 'static,
        W::Error: 'static + std::fmt::Debug,
    {
        let (tx, mut rx) = tokio::sync::mpsc::channel(work_items.len());
        let task_counter = Arc::clone(&self.task_counter);
        let stats = Arc::clone(&self.stats);

        // Sort work items by priority and complexity for load balancing
        let mut work_items = work_items;
        match self.load_balancing {
            LoadBalancingStrategy::PriorityBased => {
                work_items.sort_by_key(|b| std::cmp::Reverse(b.priority()));
            }
            LoadBalancingStrategy::ComplexityAware => {
                work_items.sort_by_key(|b| std::cmp::Reverse(b.complexity()));
            }
            _ => {} // No sorting needed
        }

        let total_items = work_items.len();

        for (index, work) in work_items.into_iter().enumerate() {
            let tx = tx.clone();
            let task_counter = Arc::clone(&task_counter);
            let stats = Arc::clone(&stats);

            self.thread_pool.spawn(move || {
                let start_time = Instant::now();
                let result = work.execute();
                let execution_time = start_time.elapsed();

                // Update statistics
                task_counter.fetch_add(1, Ordering::Relaxed);

                // Update statistics synchronously
                if let Ok(mut stats_guard) = stats.try_write() {
                    stats_guard.avg_execution_time =
                        (stats_guard.avg_execution_time + execution_time.as_secs_f64()) / 2.0;
                }

                let _ = tx.try_send((index, result));
            });
        }

        drop(tx); // Close the sender

        // Collect results in order with timeout
        let mut results: Vec<Option<std::result::Result<W::Output, W::Error>>> =
            (0..total_items).map(|_| None).collect();

        let mut collected = 0;
        while collected < total_items {
            match tokio::time::timeout(std::time::Duration::from_secs(10), rx.recv()).await {
                Ok(Some((index, result))) => {
                    results[index] = Some(result);
                    collected += 1;
                }
                Ok(None) => break, // Channel closed
                Err(_) => {
                    return Err(DatasetError::ProcessingError(
                        "Timeout waiting for worker results".to_string(),
                    ));
                }
            }
        }

        // Convert results and handle errors
        let mut outputs = Vec::with_capacity(total_items);
        for (i, result_opt) in results.into_iter().enumerate() {
            match result_opt {
                Some(Ok(output)) => outputs.push(output),
                Some(Err(e)) => {
                    return Err(DatasetError::ProcessingError(format!(
                        "Work item {i} failed: {e:?}"
                    )))
                }
                None => {
                    return Err(DatasetError::ProcessingError(format!(
                        "Work item {i} was not completed"
                    )))
                }
            }
        }

        Ok(outputs)
    }

    /// Get current resource usage statistics
    pub async fn get_stats(&self) -> ResourceStats {
        let mut stats = self.stats.read().await.clone();

        // Update throughput
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let total_tasks = self.task_counter.load(Ordering::Relaxed);
        stats.throughput = if elapsed > 0.0 {
            total_tasks as f64 / elapsed
        } else {
            0.0
        };

        // Update active workers (approximation)
        stats.active_workers = self.thread_pool.current_num_threads();

        stats
    }

    /// Set load balancing strategy
    pub fn set_load_balancing(&mut self, strategy: LoadBalancingStrategy) {
        self.load_balancing = strategy;
        debug!("Set load balancing strategy to {:?}", strategy);
    }

    /// Get worker pool configuration
    pub fn config(&self) -> &WorkerConfig {
        &self.config
    }

    /// Get number of worker threads
    pub fn num_threads(&self) -> usize {
        self.thread_pool.current_num_threads()
    }

    /// Check if the worker pool is healthy
    pub async fn is_healthy(&self) -> bool {
        let stats = self.get_stats().await;

        // Basic health checks
        stats.active_workers > 0 &&
        stats.cpu_utilization < 0.95 &&  // Not completely overloaded
        stats.memory_usage < 8 * 1024 * 1024 * 1024 // Less than 8GB memory usage
    }

    /// Gracefully shutdown the worker pool
    pub fn shutdown(self) {
        info!(
            "Shutting down worker pool with {} threads",
            self.num_threads()
        );
        // Rayon ThreadPool will be dropped and automatically shut down
    }
}

impl Default for WorkerPool {
    fn default() -> Self {
        Self::new().expect("Failed to create default worker pool")
    }
}

/// Convenience trait for parallel processing
pub trait ParallelProcessor {
    /// Process items in parallel using the worker pool
    fn process_parallel<T, F, R>(
        &self,
        items: Vec<T>,
        processor: F,
    ) -> impl std::future::Future<Output = Result<Vec<R>>> + Send
    where
        T: Send + Sync + 'static,
        F: Fn(T) -> R + Send + Sync + Clone + 'static,
        R: Send + Sync + 'static;
}

impl ParallelProcessor for WorkerPool {
    async fn process_parallel<T, F, R>(&self, items: Vec<T>, processor: F) -> Result<Vec<R>>
    where
        T: Send + Sync + 'static,
        F: Fn(T) -> R + Send + Sync + Clone + 'static,
        R: Send + Sync + 'static,
    {
        struct ProcessorWorkItem<T, F, R> {
            item: T,
            processor: F,
            _phantom: std::marker::PhantomData<R>,
        }

        impl<T, F, R> WorkItem for ProcessorWorkItem<T, F, R>
        where
            T: Send + Sync,
            F: Fn(T) -> R + Send + Sync,
            R: Send + Sync,
        {
            type Output = R;
            type Error = DatasetError;

            fn execute(self) -> std::result::Result<Self::Output, Self::Error> {
                Ok((self.processor)(self.item))
            }
        }

        let work_items: Vec<_> = items
            .into_iter()
            .map(|item| ProcessorWorkItem {
                item,
                processor: processor.clone(),
                _phantom: std::marker::PhantomData,
            })
            .collect();

        self.execute_batch(work_items).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    struct TestWorkItem {
        value: i32,
        delay_ms: u64,
    }

    impl WorkItem for TestWorkItem {
        type Output = i32;
        type Error = DatasetError;

        fn execute(self) -> std::result::Result<Self::Output, Self::Error> {
            std::thread::sleep(Duration::from_millis(self.delay_ms));
            Ok(self.value * 2)
        }

        fn complexity(&self) -> u64 {
            self.delay_ms
        }
    }

    #[tokio::test]
    async fn test_worker_pool_creation() {
        let pool = WorkerPool::new().unwrap();
        assert!(pool.num_threads() > 0);
        assert!(pool.is_healthy().await);
    }

    #[tokio::test]
    async fn test_single_task_execution() {
        let pool = WorkerPool::new().unwrap();
        let work = TestWorkItem {
            value: 5,
            delay_ms: 10,
        };

        let result = pool.execute(work).await.unwrap();
        assert_eq!(result, 10);
    }

    #[tokio::test]
    async fn test_batch_execution() {
        let pool = WorkerPool::new().unwrap();
        let work_items: Vec<TestWorkItem> = (1..=5)
            .map(|i| TestWorkItem {
                value: i,
                delay_ms: 10,
            })
            .collect();

        let results = pool.execute_batch(work_items).await.unwrap();

        let expected: Vec<i32> = (1..=5).map(|i| i * 2).collect();

        assert_eq!(results, expected);
    }

    #[tokio::test]
    async fn test_parallel_processor_trait() {
        let pool = WorkerPool::new().unwrap();
        let items: Vec<i32> = (1..=5).collect();

        let results = pool.process_parallel(items, |x| x * 3).await.unwrap();
        let expected: Vec<i32> = (1..=5).map(|i| i * 3).collect();

        assert_eq!(results, expected);
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let pool = WorkerPool::new().unwrap();

        // Execute some tasks
        for i in 1..=5 {
            let work = TestWorkItem {
                value: i,
                delay_ms: 5,
            };
            let _ = pool.execute(work).await.unwrap();
        }

        let stats = pool.get_stats().await;
        assert!(stats.throughput > 0.0);
        assert!(stats.avg_execution_time > 0.0);
        assert_eq!(stats.active_workers, pool.num_threads());
    }

    #[tokio::test]
    async fn test_load_balancing_strategies() {
        let mut pool = WorkerPool::new().unwrap();

        // Test different strategies
        for strategy in [
            LoadBalancingStrategy::RoundRobin,
            LoadBalancingStrategy::LeastLoaded,
            LoadBalancingStrategy::ComplexityAware,
            LoadBalancingStrategy::PriorityBased,
        ] {
            pool.set_load_balancing(strategy);

            let work_items: Vec<TestWorkItem> = (1..=3)
                .map(|i| TestWorkItem {
                    value: i,
                    delay_ms: 5,
                })
                .collect();

            let results = pool.execute_batch(work_items).await.unwrap();
            assert_eq!(results.len(), 3);

            // All results should be valid (non-zero values)
            for result in results {
                assert!(result > 0);
            }
        }
    }

    #[tokio::test]
    async fn test_custom_config() {
        let config = WorkerConfig {
            num_threads: Some(2),
            thread_name_prefix: "test-worker".to_string(),
            work_stealing: false,
            ..Default::default()
        };

        let pool = WorkerPool::with_config(config).unwrap();
        assert_eq!(pool.num_threads(), 2);
        assert_eq!(pool.config().thread_name_prefix, "test-worker");
        assert!(!pool.config().work_stealing);
    }
}
