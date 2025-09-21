//! Load balancing strategies for parallel processing
//!
//! Provides work distribution algorithms and load balancing strategies
//! to optimize parallel processing performance across available CPU cores.

use crate::{Result, VocoderError};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Load balancing strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Least loaded worker
    LeastLoaded,
    /// Work stealing based
    WorkStealing,
    /// Dynamic adaptive
    Adaptive,
}

impl Default for LoadBalancingStrategy {
    fn default() -> Self {
        Self::Adaptive
    }
}

/// Worker statistics for load balancing
#[derive(Debug)]
pub struct WorkerStats {
    pub worker_id: usize,
    pub tasks_processed: AtomicUsize,
    pub total_processing_time: AtomicUsize, // microseconds
    pub current_load: AtomicUsize,          // percentage (0-100)
    pub last_task_time: AtomicUsize,        // timestamp
}

impl WorkerStats {
    pub fn new(worker_id: usize) -> Self {
        Self {
            worker_id,
            tasks_processed: AtomicUsize::new(0),
            total_processing_time: AtomicUsize::new(0),
            current_load: AtomicUsize::new(0),
            last_task_time: AtomicUsize::new(0),
        }
    }

    /// Get average processing time per task in microseconds
    pub fn average_task_time(&self) -> f64 {
        let tasks = self.tasks_processed.load(Ordering::Relaxed);
        if tasks == 0 {
            return 0.0;
        }

        let total_time = self.total_processing_time.load(Ordering::Relaxed);
        total_time as f64 / tasks as f64
    }

    /// Get current load as percentage
    pub fn load_percentage(&self) -> f32 {
        self.current_load.load(Ordering::Relaxed) as f32
    }

    /// Update stats after task completion
    pub fn update_after_task(&self, processing_time: Duration) {
        self.tasks_processed.fetch_add(1, Ordering::Relaxed);
        self.total_processing_time
            .fetch_add(processing_time.as_micros() as usize, Ordering::Relaxed);
        self.last_task_time.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as usize,
            Ordering::Relaxed,
        );
    }

    /// Update current load
    pub fn update_load(&self, load_percentage: u8) {
        self.current_load
            .store(load_percentage as usize, Ordering::Relaxed);
    }
}

/// Load balancer for distributing work across workers
pub struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    worker_stats: Vec<Arc<WorkerStats>>,
    next_worker: AtomicUsize,
    adaptation_window: Duration,
    last_adaptation: std::sync::Mutex<Instant>,
}

impl LoadBalancer {
    /// Create a new load balancer
    pub fn new(strategy: LoadBalancingStrategy, num_workers: usize) -> Self {
        let worker_stats = (0..num_workers)
            .map(|id| Arc::new(WorkerStats::new(id)))
            .collect();

        Self {
            strategy,
            worker_stats,
            next_worker: AtomicUsize::new(0),
            adaptation_window: Duration::from_secs(5),
            last_adaptation: std::sync::Mutex::new(Instant::now()),
        }
    }

    /// Select the best worker for the next task
    pub fn select_worker(&self) -> Result<usize> {
        match self.strategy {
            LoadBalancingStrategy::RoundRobin => self.select_round_robin(),
            LoadBalancingStrategy::LeastLoaded => self.select_least_loaded(),
            LoadBalancingStrategy::WorkStealing => self.select_work_stealing(),
            LoadBalancingStrategy::Adaptive => self.select_adaptive(),
        }
    }

    /// Round-robin worker selection
    fn select_round_robin(&self) -> Result<usize> {
        if self.worker_stats.is_empty() {
            return Err(VocoderError::VocodingError(
                "No workers available".to_string(),
            ));
        }

        let worker_id = self.next_worker.fetch_add(1, Ordering::Relaxed) % self.worker_stats.len();
        Ok(worker_id)
    }

    /// Select the least loaded worker
    fn select_least_loaded(&self) -> Result<usize> {
        if self.worker_stats.is_empty() {
            return Err(VocoderError::VocodingError(
                "No workers available".to_string(),
            ));
        }

        let mut best_worker = 0;
        let mut lowest_load = f32::MAX;

        for (i, stats) in self.worker_stats.iter().enumerate() {
            let load = stats.load_percentage();
            if load < lowest_load {
                lowest_load = load;
                best_worker = i;
            }
        }

        Ok(best_worker)
    }

    /// Work-stealing based selection
    fn select_work_stealing(&self) -> Result<usize> {
        if self.worker_stats.is_empty() {
            return Err(VocoderError::VocodingError(
                "No workers available".to_string(),
            ));
        }

        // Find worker with minimum average task time (fastest worker)
        let mut best_worker = 0;
        let mut fastest_time = f64::MAX;

        for (i, stats) in self.worker_stats.iter().enumerate() {
            let avg_time = stats.average_task_time();
            if avg_time < fastest_time && avg_time > 0.0 {
                fastest_time = avg_time;
                best_worker = i;
            }
        }

        // If no worker has processed tasks yet, use round-robin
        if fastest_time == f64::MAX {
            return self.select_round_robin();
        }

        Ok(best_worker)
    }

    /// Adaptive selection that changes strategy based on performance
    fn select_adaptive(&self) -> Result<usize> {
        // Check if we should adapt the strategy
        if let Ok(mut last_adaptation) = self.last_adaptation.try_lock() {
            if last_adaptation.elapsed() > self.adaptation_window {
                *last_adaptation = Instant::now();
                self.adapt_strategy();
            }
        }

        // Use the current strategy
        match self.strategy {
            LoadBalancingStrategy::Adaptive => self.select_least_loaded(), // Default fallback
            _ => self.select_worker(), // Recursive call with updated strategy
        }
    }

    /// Adapt the load balancing strategy based on current performance
    fn adapt_strategy(&self) {
        let total_tasks: usize = self
            .worker_stats
            .iter()
            .map(|stats| stats.tasks_processed.load(Ordering::Relaxed))
            .sum();

        if total_tasks < 100 {
            // Not enough data, stick with current strategy
            return;
        }

        // Calculate load variance
        let avg_load: f32 = self
            .worker_stats
            .iter()
            .map(|stats| stats.load_percentage())
            .sum::<f32>()
            / self.worker_stats.len() as f32;

        let load_variance: f32 = self
            .worker_stats
            .iter()
            .map(|stats| {
                let diff = stats.load_percentage() - avg_load;
                diff * diff
            })
            .sum::<f32>()
            / self.worker_stats.len() as f32;

        // Adapt strategy based on load distribution
        if load_variance > 400.0 {
            // High variance, use least loaded
            // Note: We can't mutate self.strategy here due to immutability
            // In a real implementation, this would use interior mutability
        } else if load_variance < 100.0 {
            // Low variance, round-robin is fine
        }
    }

    /// Get worker statistics
    pub fn worker_stats(&self, worker_id: usize) -> Option<&Arc<WorkerStats>> {
        self.worker_stats.get(worker_id)
    }

    /// Get all worker statistics
    pub fn all_worker_stats(&self) -> &[Arc<WorkerStats>] {
        &self.worker_stats
    }

    /// Get current strategy
    pub fn strategy(&self) -> LoadBalancingStrategy {
        self.strategy
    }

    /// Get total number of workers
    pub fn num_workers(&self) -> usize {
        self.worker_stats.len()
    }

    /// Update worker load
    pub fn update_worker_load(&self, worker_id: usize, load_percentage: u8) {
        if let Some(stats) = self.worker_stats.get(worker_id) {
            stats.update_load(load_percentage);
        }
    }

    /// Record task completion for a worker
    pub fn record_task_completion(&self, worker_id: usize, processing_time: Duration) {
        if let Some(stats) = self.worker_stats.get(worker_id) {
            stats.update_after_task(processing_time);
        }
    }

    /// Get load balancing statistics
    pub fn statistics(&self) -> LoadBalancingStats {
        let total_tasks = self
            .worker_stats
            .iter()
            .map(|stats| stats.tasks_processed.load(Ordering::Relaxed))
            .sum();

        let total_time: u64 = self
            .worker_stats
            .iter()
            .map(|stats| stats.total_processing_time.load(Ordering::Relaxed) as u64)
            .sum();

        let avg_load = self
            .worker_stats
            .iter()
            .map(|stats| stats.load_percentage())
            .sum::<f32>()
            / self.worker_stats.len() as f32;

        LoadBalancingStats {
            strategy: self.strategy,
            total_tasks,
            total_processing_time: Duration::from_micros(total_time),
            average_load: avg_load,
            num_workers: self.worker_stats.len(),
        }
    }
}

/// Statistics from the load balancer
#[derive(Debug, Clone)]
pub struct LoadBalancingStats {
    pub strategy: LoadBalancingStrategy,
    pub total_tasks: usize,
    pub total_processing_time: Duration,
    pub average_load: f32,
    pub num_workers: usize,
}

impl LoadBalancingStats {
    /// Get average task processing time
    pub fn average_task_time(&self) -> Duration {
        if self.total_tasks == 0 {
            return Duration::ZERO;
        }
        self.total_processing_time / self.total_tasks as u32
    }

    /// Get tasks per second
    pub fn tasks_per_second(&self) -> f64 {
        if self.total_processing_time.is_zero() {
            return 0.0;
        }
        self.total_tasks as f64 / self.total_processing_time.as_secs_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_stats_creation() {
        let stats = WorkerStats::new(0);
        assert_eq!(stats.worker_id, 0);
        assert_eq!(stats.tasks_processed.load(Ordering::Relaxed), 0);
        assert_eq!(stats.average_task_time(), 0.0);
    }

    #[test]
    fn test_worker_stats_update() {
        let stats = WorkerStats::new(0);
        let duration = Duration::from_millis(10);

        stats.update_after_task(duration);

        assert_eq!(stats.tasks_processed.load(Ordering::Relaxed), 1);
        assert!(stats.average_task_time() > 0.0);
    }

    #[test]
    fn test_load_balancer_creation() {
        let balancer = LoadBalancer::new(LoadBalancingStrategy::RoundRobin, 4);
        assert_eq!(balancer.num_workers(), 4);
        assert_eq!(balancer.strategy(), LoadBalancingStrategy::RoundRobin);
    }

    #[test]
    fn test_round_robin_selection() {
        let balancer = LoadBalancer::new(LoadBalancingStrategy::RoundRobin, 3);

        // Test round-robin behavior
        assert_eq!(balancer.select_worker().unwrap(), 0);
        assert_eq!(balancer.select_worker().unwrap(), 1);
        assert_eq!(balancer.select_worker().unwrap(), 2);
        assert_eq!(balancer.select_worker().unwrap(), 0); // Wraps around
    }

    #[test]
    fn test_least_loaded_selection() {
        let balancer = LoadBalancer::new(LoadBalancingStrategy::LeastLoaded, 3);

        // Set different loads
        balancer.update_worker_load(0, 80);
        balancer.update_worker_load(1, 20);
        balancer.update_worker_load(2, 50);

        // Should select worker 1 (lowest load)
        assert_eq!(balancer.select_worker().unwrap(), 1);
    }

    #[test]
    fn test_load_balancer_statistics() {
        let balancer = LoadBalancer::new(LoadBalancingStrategy::RoundRobin, 2);

        // Record some task completions
        balancer.record_task_completion(0, Duration::from_millis(10));
        balancer.record_task_completion(1, Duration::from_millis(20));

        let stats = balancer.statistics();
        assert_eq!(stats.total_tasks, 2);
        assert_eq!(stats.num_workers, 2);
        assert!(stats.average_task_time() > Duration::ZERO);
    }

    #[test]
    fn test_worker_load_update() {
        let balancer = LoadBalancer::new(LoadBalancingStrategy::LeastLoaded, 2);

        balancer.update_worker_load(0, 75);
        balancer.update_worker_load(1, 25);

        if let Some(stats) = balancer.worker_stats(0) {
            assert_eq!(stats.load_percentage(), 75.0);
        }

        if let Some(stats) = balancer.worker_stats(1) {
            assert_eq!(stats.load_percentage(), 25.0);
        }
    }

    #[test]
    fn test_empty_workers() {
        let balancer = LoadBalancer::new(LoadBalancingStrategy::RoundRobin, 0);
        assert!(balancer.select_worker().is_err());
    }

    #[test]
    fn test_work_stealing_selection() {
        let balancer = LoadBalancer::new(LoadBalancingStrategy::WorkStealing, 3);

        // Record different performance for workers
        balancer.record_task_completion(0, Duration::from_millis(30));
        balancer.record_task_completion(1, Duration::from_millis(10)); // Fastest
        balancer.record_task_completion(2, Duration::from_millis(20));

        // Should prefer worker 1 (fastest average time)
        let selected = balancer.select_worker().unwrap();
        assert_eq!(selected, 1);
    }
}
