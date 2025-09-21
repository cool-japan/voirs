//! Work-stealing queue implementation for parallel processing
//!
//! Provides high-performance work-stealing queues for distributing
//! work across multiple threads with minimal contention.

use crate::{Result, VocoderError};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

/// Work-stealing queue for parallel task distribution
pub struct WorkStealingQueue<T> {
    local_queue: Mutex<VecDeque<T>>,
    global_queue: Arc<Mutex<VecDeque<T>>>,
    worker_id: usize,
    stats: Arc<WorkStealingStats>,
}

/// Statistics for work-stealing operations
#[derive(Debug)]
pub struct WorkStealingStats {
    pub tasks_pushed: AtomicUsize,
    pub tasks_popped: AtomicUsize,
    pub tasks_stolen: AtomicUsize,
    pub steal_attempts: AtomicUsize,
    pub successful_steals: AtomicUsize,
    pub failed_steals: AtomicUsize,
}

impl Default for WorkStealingStats {
    fn default() -> Self {
        Self {
            tasks_pushed: AtomicUsize::new(0),
            tasks_popped: AtomicUsize::new(0),
            tasks_stolen: AtomicUsize::new(0),
            steal_attempts: AtomicUsize::new(0),
            successful_steals: AtomicUsize::new(0),
            failed_steals: AtomicUsize::new(0),
        }
    }
}

/// Snapshot of work-stealing statistics
#[derive(Debug, Clone, Default)]
pub struct WorkStealingStatsSnapshot {
    pub tasks_pushed: usize,
    pub tasks_popped: usize,
    pub tasks_stolen: usize,
    pub steal_attempts: usize,
    pub successful_steals: usize,
    pub failed_steals: usize,
}

impl WorkStealingStatsSnapshot {
    /// Calculate steal success rate
    pub fn steal_success_rate(&self) -> f64 {
        if self.steal_attempts == 0 {
            0.0
        } else {
            self.successful_steals as f64 / self.steal_attempts as f64
        }
    }

    /// Calculate work stealing efficiency
    pub fn work_stealing_efficiency(&self) -> f64 {
        let total_work = self.tasks_pushed + self.tasks_stolen;
        if total_work == 0 {
            0.0
        } else {
            self.tasks_stolen as f64 / total_work as f64
        }
    }
}

impl<T> WorkStealingQueue<T> {
    /// Create a new work-stealing queue
    pub fn new(worker_id: usize, global_queue: Arc<Mutex<VecDeque<T>>>) -> Self {
        Self {
            local_queue: Mutex::new(VecDeque::new()),
            global_queue,
            worker_id,
            stats: Arc::new(WorkStealingStats::default()),
        }
    }

    /// Push a task to the local queue
    pub fn push(&self, task: T) -> Result<()> {
        let mut local = self
            .local_queue
            .lock()
            .map_err(|_| VocoderError::RuntimeError("Failed to lock local queue".to_string()))?;

        local.push_back(task);
        self.stats.tasks_pushed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    /// Pop a task from the local queue
    pub fn pop(&self) -> Option<T> {
        if let Ok(mut local) = self.local_queue.lock() {
            if let Some(task) = local.pop_front() {
                self.stats.tasks_popped.fetch_add(1, Ordering::SeqCst);
                return Some(task);
            }
        }
        None
    }

    /// Try to steal work from another worker's queue
    pub fn steal(&self, other: &WorkStealingQueue<T>) -> Option<T> {
        self.stats.steal_attempts.fetch_add(1, Ordering::SeqCst);

        // Try to steal from the other worker's local queue
        if let Ok(mut other_local) = other.local_queue.try_lock() {
            if let Some(task) = other_local.pop_back() {
                self.stats.successful_steals.fetch_add(1, Ordering::SeqCst);
                self.stats.tasks_stolen.fetch_add(1, Ordering::SeqCst);
                other.stats.tasks_stolen.fetch_add(1, Ordering::SeqCst);
                return Some(task);
            }
        }

        // If that fails, try the global queue
        if let Ok(mut global) = self.global_queue.try_lock() {
            if let Some(task) = global.pop_front() {
                self.stats.successful_steals.fetch_add(1, Ordering::SeqCst);
                return Some(task);
            }
        }

        self.stats.failed_steals.fetch_add(1, Ordering::SeqCst);
        None
    }

    /// Push overflow work to the global queue
    pub fn push_to_global(&self, task: T) -> Result<()> {
        let mut global = self
            .global_queue
            .lock()
            .map_err(|_| VocoderError::RuntimeError("Failed to lock global queue".to_string()))?;

        global.push_back(task);
        Ok(())
    }

    /// Get the current local queue size
    pub fn local_size(&self) -> usize {
        self.local_queue.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Get the current global queue size
    pub fn global_size(&self) -> usize {
        self.global_queue.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Get work-stealing statistics snapshot
    pub fn stats(&self) -> WorkStealingStatsSnapshot {
        let stats = self.stats.as_ref();
        WorkStealingStatsSnapshot {
            tasks_pushed: stats.tasks_pushed.load(Ordering::SeqCst),
            tasks_popped: stats.tasks_popped.load(Ordering::SeqCst),
            tasks_stolen: stats.tasks_stolen.load(Ordering::SeqCst),
            steal_attempts: stats.steal_attempts.load(Ordering::SeqCst),
            successful_steals: stats.successful_steals.load(Ordering::SeqCst),
            failed_steals: stats.failed_steals.load(Ordering::SeqCst),
        }
    }

    /// Get worker ID
    pub fn worker_id(&self) -> usize {
        self.worker_id
    }
}

/// Work-stealing scheduler for managing multiple workers
pub struct WorkStealingScheduler<T> {
    workers: Vec<Arc<WorkStealingQueue<T>>>,
    global_queue: Arc<Mutex<VecDeque<T>>>,
    round_robin_counter: AtomicUsize,
}

impl<T: Clone> WorkStealingScheduler<T> {
    /// Create a new work-stealing scheduler
    pub fn new(num_workers: usize) -> Self {
        let global_queue = Arc::new(Mutex::new(VecDeque::<T>::new()));
        let mut workers = Vec::with_capacity(num_workers);

        for worker_id in 0..num_workers {
            workers.push(Arc::new(WorkStealingQueue::new(
                worker_id,
                Arc::clone(&global_queue),
            )));
        }

        Self {
            workers,
            global_queue,
            round_robin_counter: AtomicUsize::new(0),
        }
    }

    /// Submit a task to the scheduler
    pub fn submit(&self, task: T) -> Result<()> {
        // Round-robin assignment to workers
        let worker_idx =
            self.round_robin_counter.fetch_add(1, Ordering::SeqCst) % self.workers.len();

        if let Some(worker) = self.workers.get(worker_idx) {
            // Try to push to the selected worker's local queue
            if worker.push(task.clone()).is_ok() {
                return Ok(());
            }
        }

        // If that fails, push to global queue
        let mut global = self
            .global_queue
            .lock()
            .map_err(|_| VocoderError::RuntimeError("Failed to lock global queue".to_string()))?;
        global.push_back(task);
        Ok(())
    }

    /// Get work for a specific worker
    pub fn get_work(&self, worker_id: usize) -> Option<T> {
        if let Some(worker) = self.workers.get(worker_id) {
            // First try local queue
            if let Some(task) = worker.pop() {
                return Some(task);
            }

            // Then try work stealing from other workers
            for other_worker in &self.workers {
                if other_worker.worker_id() != worker_id {
                    if let Some(task) = worker.steal(other_worker) {
                        return Some(task);
                    }
                }
            }
        }

        None
    }

    /// Get the number of workers
    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    /// Get total pending tasks across all queues
    pub fn total_pending(&self) -> usize {
        let local_total: usize = self.workers.iter().map(|w| w.local_size()).sum();
        let global_total = self.global_queue.lock().map(|q| q.len()).unwrap_or(0);
        local_total + global_total
    }

    /// Get aggregated statistics across all workers
    pub fn aggregate_stats(&self) -> WorkStealingStats {
        let total_stats = WorkStealingStats::default();

        for worker in &self.workers {
            let worker_stats = worker.stats();
            total_stats
                .tasks_pushed
                .fetch_add(worker_stats.tasks_pushed, Ordering::SeqCst);
            total_stats
                .tasks_popped
                .fetch_add(worker_stats.tasks_popped, Ordering::SeqCst);
            total_stats
                .tasks_stolen
                .fetch_add(worker_stats.tasks_stolen, Ordering::SeqCst);
            total_stats
                .steal_attempts
                .fetch_add(worker_stats.steal_attempts, Ordering::SeqCst);
            total_stats
                .successful_steals
                .fetch_add(worker_stats.successful_steals, Ordering::SeqCst);
            total_stats
                .failed_steals
                .fetch_add(worker_stats.failed_steals, Ordering::SeqCst);
        }

        total_stats
    }
}

impl WorkStealingStats {
    /// Calculate steal success rate
    pub fn steal_success_rate(&self) -> f32 {
        let total_attempts = self.steal_attempts.load(Ordering::SeqCst);
        if total_attempts == 0 {
            return 0.0;
        }

        let successful = self.successful_steals.load(Ordering::SeqCst);
        successful as f32 / total_attempts as f32
    }

    /// Calculate work stealing efficiency
    pub fn work_stealing_efficiency(&self) -> f32 {
        let total_tasks =
            self.tasks_popped.load(Ordering::SeqCst) + self.tasks_stolen.load(Ordering::SeqCst);
        if total_tasks == 0 {
            return 0.0;
        }

        let stolen = self.tasks_stolen.load(Ordering::SeqCst);
        stolen as f32 / total_tasks as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::Arc;

    #[test]
    fn test_work_stealing_queue_creation() {
        let global_queue = Arc::new(Mutex::new(VecDeque::<i32>::new()));
        let queue = WorkStealingQueue::new(0, global_queue);

        assert_eq!(queue.worker_id(), 0);
        assert_eq!(queue.local_size(), 0);
    }

    #[test]
    fn test_push_pop_operations() {
        let global_queue = Arc::new(Mutex::new(VecDeque::<i32>::new()));
        let queue = WorkStealingQueue::new(0, global_queue);

        queue.push(42).unwrap();
        assert_eq!(queue.local_size(), 1);

        let task = queue.pop();
        assert_eq!(task, Some(42));
        assert_eq!(queue.local_size(), 0);
    }

    #[test]
    fn test_work_stealing() {
        let global_queue = Arc::new(Mutex::new(VecDeque::<i32>::new()));
        let queue1 = WorkStealingQueue::new(0, Arc::clone(&global_queue));
        let queue2 = WorkStealingQueue::new(1, global_queue);

        // Push tasks to queue1
        queue1.push(1).unwrap();
        queue1.push(2).unwrap();
        queue1.push(3).unwrap();

        // Queue2 should be able to steal from queue1
        let stolen_task = queue2.steal(&queue1);
        assert!(stolen_task.is_some());

        // Check statistics
        let stats = queue2.stats();
        assert_eq!(stats.steal_attempts, 1);
        assert_eq!(stats.successful_steals, 1);
    }

    #[test]
    fn test_work_stealing_scheduler() {
        let scheduler = WorkStealingScheduler::new(4);

        assert_eq!(scheduler.num_workers(), 4);
        assert_eq!(scheduler.total_pending(), 0);

        // Submit some tasks
        scheduler.submit(1).unwrap();
        scheduler.submit(2).unwrap();
        scheduler.submit(3).unwrap();

        assert!(scheduler.total_pending() > 0);

        // Workers should be able to get work
        let task = scheduler.get_work(0);
        assert!(task.is_some());
    }

    #[test]
    fn test_scheduler_work_distribution() {
        let scheduler = WorkStealingScheduler::new(2);

        // Submit many tasks
        for i in 0..10 {
            scheduler.submit(i).unwrap();
        }

        let mut retrieved_tasks = Vec::new();

        // Both workers should be able to get work
        while let Some(task) = scheduler.get_work(0) {
            retrieved_tasks.push(task);
        }

        while let Some(task) = scheduler.get_work(1) {
            retrieved_tasks.push(task);
        }

        assert_eq!(retrieved_tasks.len(), 10);
    }

    #[test]
    fn test_stats_calculation() {
        let global_queue = Arc::new(Mutex::new(VecDeque::<i32>::new()));
        let queue1 = WorkStealingQueue::new(0, Arc::clone(&global_queue));
        let queue2 = WorkStealingQueue::new(1, global_queue);

        // Generate some activity
        queue1.push(1).unwrap();
        queue1.push(2).unwrap();
        queue1.pop();
        queue2.steal(&queue1);

        let stats = queue2.stats();
        assert!(stats.steal_success_rate() > 0.0);
        assert!(stats.work_stealing_efficiency() >= 0.0);
    }

    #[test]
    fn test_scheduler_aggregate_stats() {
        let scheduler = WorkStealingScheduler::new(3);

        // Submit and process some tasks
        for i in 0..5 {
            scheduler.submit(i).unwrap();
        }

        // Get work to generate statistics
        for worker_id in 0..3 {
            scheduler.get_work(worker_id);
        }

        let stats = scheduler.aggregate_stats();
        assert!(stats.tasks_pushed.load(Ordering::SeqCst) > 0);
    }

    #[test]
    fn test_global_queue_fallback() {
        let global_queue = Arc::new(Mutex::new(VecDeque::<i32>::new()));
        let queue = WorkStealingQueue::new(0, Arc::clone(&global_queue));

        // Push to global queue directly
        queue.push_to_global(42).unwrap();
        assert_eq!(queue.global_size(), 1);

        // Should be able to steal from global queue
        let stolen = queue.steal(&queue);
        assert_eq!(stolen, Some(42));
    }
}
