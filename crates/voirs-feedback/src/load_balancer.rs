//! Load balancing system for distributing feedback requests across multiple workers

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tokio::time::sleep;
use uuid::Uuid;

use crate::traits::{
    AdaptiveState, FeedbackResponse, SessionState, SessionStatistics, SessionStats,
    UserPreferences, UserProgress,
};

/// Worker node in the load balancer
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerNode {
    /// Unique worker ID
    pub id: String,
    /// Worker endpoint URL
    pub endpoint: String,
    /// Current load (0.0 to 1.0)
    pub current_load: f64,
    /// Maximum concurrent requests
    pub max_concurrent: usize,
    /// Current active requests
    pub active_requests: usize,
    /// Health status
    pub health_status: WorkerHealth,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// Total processed requests
    pub total_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Last health check timestamp
    pub last_health_check: DateTime<Utc>,
    /// Weight for load distribution (1.0 = normal, 2.0 = twice as powerful)
    pub weight: f64,
}

/// Health status of a worker node
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WorkerHealth {
    /// Worker is healthy and available
    Healthy,
    /// Worker is degraded but still functional
    Degraded,
    /// Worker is unhealthy and should not receive requests
    Unhealthy,
    /// Worker status is unknown
    Unknown,
}

/// Load balancing algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LoadBalancingAlgorithm {
    /// Round robin distribution
    RoundRobin,
    /// Weighted round robin
    WeightedRoundRobin,
    /// Least connections
    LeastConnections,
    /// Least response time
    LeastResponseTime,
    /// Weighted least connections
    WeightedLeastConnections,
    /// Resource-based (CPU, memory)
    ResourceBased,
}

/// Request to be processed by a worker
#[derive(Debug, Clone)]
pub struct WorkerRequest {
    /// Unique request ID
    pub id: String,
    /// Session state
    pub session: SessionState,
    /// Request timestamp
    pub timestamp: DateTime<Utc>,
    /// Priority level (1-10, 10 being highest)
    pub priority: u8,
    /// Estimated processing time in milliseconds
    pub estimated_time_ms: u32,
}

/// Response from a worker
#[derive(Debug, Clone)]
pub struct WorkerResponse {
    /// Original request ID
    pub request_id: String,
    /// Worker ID that processed the request
    pub worker_id: String,
    /// Processing result
    pub result: Result<FeedbackResponse, String>,
    /// Processing time in milliseconds
    pub processing_time_ms: u32,
    /// Response timestamp
    pub timestamp: DateTime<Utc>,
}

/// Load balancer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerConfig {
    /// Load balancing algorithm
    pub algorithm: LoadBalancingAlgorithm,
    /// Health check interval in seconds
    pub health_check_interval_seconds: u32,
    /// Maximum queue size per worker
    pub max_queue_size: usize,
    /// Request timeout in seconds
    pub request_timeout_seconds: u32,
    /// Maximum retries for failed requests
    pub max_retries: u32,
    /// Enable automatic scaling
    pub auto_scaling_enabled: bool,
    /// Target CPU utilization for scaling (0.0 to 1.0)
    pub target_cpu_utilization: f64,
    /// Target response time in milliseconds
    pub target_response_time_ms: u32,
}

impl Default for LoadBalancerConfig {
    fn default() -> Self {
        Self {
            algorithm: LoadBalancingAlgorithm::WeightedLeastConnections,
            health_check_interval_seconds: 30,
            max_queue_size: 1000,
            request_timeout_seconds: 30,
            max_retries: 3,
            auto_scaling_enabled: true,
            target_cpu_utilization: 0.7,
            target_response_time_ms: 500,
        }
    }
}

/// Load balancer statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerStats {
    /// Total requests processed
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Average response time across all workers
    pub avg_response_time_ms: f64,
    /// Current queue size
    pub queue_size: usize,
    /// Active workers
    pub active_workers: usize,
    /// Total workers
    pub total_workers: usize,
    /// Requests per second
    pub requests_per_second: f64,
    /// Worker utilization (0.0 to 1.0)
    pub worker_utilization: f64,
}

/// Main load balancer
pub struct LoadBalancer {
    /// Configuration
    config: LoadBalancerConfig,
    /// Worker nodes
    workers: Arc<RwLock<HashMap<String, WorkerNode>>>,
    /// Request queue
    request_queue: Arc<RwLock<VecDeque<WorkerRequest>>>,
    /// Current round robin index
    round_robin_index: Arc<RwLock<usize>>,
    /// Statistics
    stats: Arc<RwLock<LoadBalancerStats>>,
    /// Concurrent request limiter
    semaphore: Arc<Semaphore>,
    /// Response history for metrics
    response_history: Arc<RwLock<Vec<WorkerResponse>>>,
    /// Last stats update time
    last_stats_update: Arc<RwLock<Instant>>,
}

impl LoadBalancer {
    /// Create a new load balancer
    #[must_use]
    pub fn new(config: LoadBalancerConfig) -> Self {
        let max_concurrent = config.max_queue_size * 2; // Allow some overhead
        Self {
            config,
            workers: Arc::new(RwLock::new(HashMap::new())),
            request_queue: Arc::new(RwLock::new(VecDeque::new())),
            round_robin_index: Arc::new(RwLock::new(0)),
            stats: Arc::new(RwLock::new(LoadBalancerStats {
                total_requests: 0,
                successful_requests: 0,
                failed_requests: 0,
                avg_response_time_ms: 0.0,
                queue_size: 0,
                active_workers: 0,
                total_workers: 0,
                requests_per_second: 0.0,
                worker_utilization: 0.0,
            })),
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            response_history: Arc::new(RwLock::new(Vec::new())),
            last_stats_update: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Add a worker node to the load balancer
    pub async fn add_worker(&self, worker: WorkerNode) -> Result<(), String> {
        {
            let mut workers = self.workers.write().await;

            if workers.contains_key(&worker.id) {
                return Err(format!("Worker with ID {} already exists", worker.id));
            }

            workers.insert(worker.id.clone(), worker);
        } // Release the lock before calling update_stats

        self.update_stats().await;

        Ok(())
    }

    /// Remove a worker node from the load balancer
    pub async fn remove_worker(&self, worker_id: &str) -> Result<(), String> {
        let removed = {
            let mut workers = self.workers.write().await;
            workers.remove(worker_id)
        }; // Release the lock before calling update_stats

        if removed.is_none() {
            return Err(format!("Worker with ID {worker_id} not found"));
        }

        self.update_stats().await;
        Ok(())
    }

    /// Get the best worker for a request based on the configured algorithm
    pub async fn select_worker(&self, request: &WorkerRequest) -> Result<String, String> {
        let workers = self.workers.read().await;

        if workers.is_empty() {
            return Err("No workers available".to_string());
        }

        let healthy_workers: Vec<&WorkerNode> = workers
            .values()
            .filter(|w| w.health_status == WorkerHealth::Healthy)
            .collect();

        if healthy_workers.is_empty() {
            return Err("No healthy workers available".to_string());
        }

        match self.config.algorithm {
            LoadBalancingAlgorithm::RoundRobin => self.select_round_robin(&healthy_workers).await,
            LoadBalancingAlgorithm::WeightedRoundRobin => {
                self.select_weighted_round_robin(&healthy_workers).await
            }
            LoadBalancingAlgorithm::LeastConnections => {
                self.select_least_connections(&healthy_workers).await
            }
            LoadBalancingAlgorithm::LeastResponseTime => {
                self.select_least_response_time(&healthy_workers).await
            }
            LoadBalancingAlgorithm::WeightedLeastConnections => {
                self.select_weighted_least_connections(&healthy_workers)
                    .await
            }
            LoadBalancingAlgorithm::ResourceBased => {
                self.select_resource_based(&healthy_workers).await
            }
        }
    }

    /// Select worker using round robin algorithm
    async fn select_round_robin(&self, workers: &[&WorkerNode]) -> Result<String, String> {
        let mut index = self.round_robin_index.write().await;
        *index = (*index + 1) % workers.len();
        Ok(workers[*index].id.clone())
    }

    /// Select worker using weighted round robin algorithm
    async fn select_weighted_round_robin(&self, workers: &[&WorkerNode]) -> Result<String, String> {
        let total_weight: f64 = workers.iter().map(|w| w.weight).sum();
        let mut cumulative_weight = 0.0;
        let target_weight = scirs2_core::random::random::<f64>() * total_weight;

        for worker in workers {
            cumulative_weight += worker.weight;
            if cumulative_weight >= target_weight {
                return Ok(worker.id.clone());
            }
        }

        // Fallback to first worker
        Ok(workers[0].id.clone())
    }

    /// Select worker using least connections algorithm
    async fn select_least_connections(&self, workers: &[&WorkerNode]) -> Result<String, String> {
        let best_worker = workers
            .iter()
            .min_by_key(|w| w.active_requests)
            .ok_or("No workers available")?;

        Ok(best_worker.id.clone())
    }

    /// Select worker using least response time algorithm
    async fn select_least_response_time(&self, workers: &[&WorkerNode]) -> Result<String, String> {
        let best_worker = workers
            .iter()
            .min_by(|a, b| {
                a.avg_response_time_ms
                    .partial_cmp(&b.avg_response_time_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or("No workers available")?;

        Ok(best_worker.id.clone())
    }

    /// Select worker using weighted least connections algorithm
    async fn select_weighted_least_connections(
        &self,
        workers: &[&WorkerNode],
    ) -> Result<String, String> {
        let best_worker = workers
            .iter()
            .min_by(|a, b| {
                let a_score = a.active_requests as f64 / a.weight;
                let b_score = b.active_requests as f64 / b.weight;
                a_score
                    .partial_cmp(&b_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or("No workers available")?;

        Ok(best_worker.id.clone())
    }

    /// Select worker using resource-based algorithm
    async fn select_resource_based(&self, workers: &[&WorkerNode]) -> Result<String, String> {
        let best_worker = workers
            .iter()
            .min_by(|a, b| {
                let a_score = a.current_load;
                let b_score = b.current_load;
                a_score
                    .partial_cmp(&b_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or("No workers available")?;

        Ok(best_worker.id.clone())
    }

    /// Submit a request to be processed
    pub async fn submit_request(&self, request: WorkerRequest) -> Result<String, String> {
        // Check if we can accept more requests
        let _permit = self
            .semaphore
            .try_acquire()
            .map_err(|_| "Load balancer is at capacity")?;

        // Select the best worker
        let worker_id = self.select_worker(&request).await?;

        // Add to queue. Scoped so the write guard is dropped before
        // `update_stats()` below, which itself takes a read lock on
        // `request_queue` -- tokio's `RwLock` is not reentrant, so holding
        // this guard across that call would deadlock the current task
        // against itself.
        {
            let mut queue = self.request_queue.write().await;
            if queue.len() >= self.config.max_queue_size {
                return Err("Request queue is full".to_string());
            }

            queue.push_back(request.clone());
        }

        // Update worker active requests. Scoped for the same reason:
        // `update_stats()` also takes a read lock on `workers`.
        {
            let mut workers = self.workers.write().await;
            if let Some(worker) = workers.get_mut(&worker_id) {
                worker.active_requests += 1;
            }
        }

        // Update statistics
        self.update_stats().await;

        Ok(request.id)
    }

    /// Process a response from a worker
    pub async fn process_response(&self, response: WorkerResponse) -> Result<(), String> {
        // Update worker statistics. Scoped so the write guard is dropped
        // before `update_stats()` below, which itself takes a read lock on
        // `workers` -- tokio's `RwLock` is not reentrant, so holding this
        // guard across that call would deadlock the current task against
        // itself (the read would wait forever for a write guard that only
        // this same task holds and can never release while awaiting).
        {
            let mut workers = self.workers.write().await;
            if let Some(worker) = workers.get_mut(&response.worker_id) {
                worker.active_requests = worker.active_requests.saturating_sub(1);
                worker.total_requests += 1;

                if response.result.is_err() {
                    worker.failed_requests += 1;
                }

                // Update average response time (exponential moving average)
                let alpha = 0.1; // Smoothing factor
                worker.avg_response_time_ms = alpha * f64::from(response.processing_time_ms)
                    + (1.0 - alpha) * worker.avg_response_time_ms;

                // Update current load
                worker.current_load = worker.active_requests as f64 / worker.max_concurrent as f64;
            }
        }

        // Store response for metrics. Scoped for the same reason:
        // `update_stats()` also takes a read lock on `response_history`.
        {
            let mut history = self.response_history.write().await;
            history.push(response);

            // Keep only recent responses (last 1000)
            if history.len() > 1000 {
                history.drain(0..500);
            }
        }

        // Update statistics
        self.update_stats().await;

        Ok(())
    }

    /// Update load balancer statistics
    async fn update_stats(&self) {
        let workers = self.workers.read().await;
        let queue = self.request_queue.read().await;
        let history = self.response_history.read().await;

        let mut stats = self.stats.write().await;
        stats.total_workers = workers.len();
        stats.active_workers = workers
            .values()
            .filter(|w| w.health_status == WorkerHealth::Healthy)
            .count();
        stats.queue_size = queue.len();

        // Calculate statistics from response history
        if !history.is_empty() {
            stats.total_requests = history.len() as u64;
            stats.successful_requests = history.iter().filter(|r| r.result.is_ok()).count() as u64;
            stats.failed_requests = history.iter().filter(|r| r.result.is_err()).count() as u64;

            let total_time: u32 = history.iter().map(|r| r.processing_time_ms).sum();
            stats.avg_response_time_ms = f64::from(total_time) / history.len() as f64;
        }

        // Calculate worker utilization
        if !workers.is_empty() {
            let total_utilization: f64 = workers.values().map(|w| w.current_load).sum();
            stats.worker_utilization = total_utilization / workers.len() as f64;
        }

        // Calculate requests per second from the real response timestamps
        // recorded in `response_history` over a trailing 60-second window.
        // `history` is already read-locked above, so this needs no
        // additional lock acquisition.
        let window = chrono::Duration::seconds(60);
        let window_start = Utc::now() - window;
        let recent_responses = history
            .iter()
            .filter(|response| response.timestamp > window_start)
            .count();
        stats.requests_per_second = recent_responses as f64 / window.num_seconds() as f64;
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> LoadBalancerStats {
        self.stats.read().await.clone()
    }

    /// Get all worker nodes
    pub async fn get_workers(&self) -> Vec<WorkerNode> {
        self.workers.read().await.values().cloned().collect()
    }

    /// Update worker health status
    pub async fn update_worker_health(
        &self,
        worker_id: &str,
        health: WorkerHealth,
    ) -> Result<(), String> {
        let mut workers = self.workers.write().await;
        if let Some(worker) = workers.get_mut(worker_id) {
            worker.health_status = health;
            worker.last_health_check = Utc::now();
            Ok(())
        } else {
            Err(format!("Worker with ID {worker_id} not found"))
        }
    }

    /// Start health check monitoring
    pub async fn start_health_monitoring(&self) {
        let workers = self.workers.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            loop {
                sleep(Duration::from_secs(u64::from(
                    config.health_check_interval_seconds,
                )))
                .await;

                let mut workers_guard = workers.write().await;
                let now = Utc::now();

                for worker in workers_guard.values_mut() {
                    let time_since_check = now.signed_duration_since(worker.last_health_check);

                    // Mark as unknown if no health check for too long
                    if time_since_check.num_seconds()
                        > i64::from(config.health_check_interval_seconds) * 2
                    {
                        worker.health_status = WorkerHealth::Unknown;
                    }
                }
            }
        });
    }

    /// Clear request queue
    pub async fn clear_queue(&self) {
        let mut queue = self.request_queue.write().await;
        queue.clear();
    }

    /// Get queue size
    pub async fn get_queue_size(&self) -> usize {
        self.request_queue.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_session_state() -> SessionState {
        SessionState {
            session_id: Uuid::new_v4(),
            user_id: "test_user".to_string(),
            start_time: Utc::now(),
            last_activity: Utc::now(),
            current_task: None,
            stats: SessionStats::default(),
            preferences: UserPreferences::default(),
            adaptive_state: AdaptiveState::default(),
            current_exercise: None,
            session_stats: SessionStatistics::default(),
        }
    }

    #[tokio::test]
    async fn test_load_balancer_creation() {
        let config = LoadBalancerConfig::default();
        let lb = LoadBalancer::new(config);

        let stats = lb.get_stats().await;
        assert_eq!(stats.total_workers, 0);
        assert_eq!(stats.active_workers, 0);
    }

    #[tokio::test]
    async fn test_add_remove_worker() {
        let config = LoadBalancerConfig::default();
        let lb = LoadBalancer::new(config);

        let worker = WorkerNode {
            id: "worker1".to_string(),
            endpoint: "http://localhost:8080".to_string(),
            current_load: 0.0,
            max_concurrent: 10,
            active_requests: 0,
            health_status: WorkerHealth::Healthy,
            avg_response_time_ms: 100.0,
            total_requests: 0,
            failed_requests: 0,
            last_health_check: Utc::now(),
            weight: 1.0,
        };

        // Add worker
        lb.add_worker(worker.clone()).await.unwrap();
        let workers = lb.get_workers().await;
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].id, "worker1");

        // Remove worker
        lb.remove_worker(&worker.id).await.unwrap();
        let workers = lb.get_workers().await;
        assert_eq!(workers.len(), 0);
    }

    #[tokio::test]
    async fn test_worker_selection() {
        let config = LoadBalancerConfig::default();
        let lb = LoadBalancer::new(config);

        // Test with no workers
        let request = WorkerRequest {
            id: "test_request".to_string(),
            session: create_test_session_state(),
            timestamp: Utc::now(),
            priority: 5,
            estimated_time_ms: 100,
        };

        let result = lb.select_worker(&request).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "No workers available");

        // Add a worker and test selection
        let worker = WorkerNode {
            id: "worker1".to_string(),
            endpoint: "http://localhost:8080".to_string(),
            current_load: 0.0,
            max_concurrent: 10,
            active_requests: 0,
            health_status: WorkerHealth::Healthy,
            avg_response_time_ms: 100.0,
            total_requests: 0,
            failed_requests: 0,
            last_health_check: Utc::now(),
            weight: 1.0,
        };

        lb.add_worker(worker.clone()).await.unwrap();

        let selected = lb.select_worker(&request).await;
        assert!(selected.is_ok());
        assert_eq!(selected.unwrap(), "worker1");

        // Test stats after adding worker
        let stats = lb.get_stats().await;
        assert_eq!(stats.total_workers, 1);
        assert_eq!(stats.active_workers, 1);
    }

    fn create_test_worker(id: &str) -> WorkerNode {
        WorkerNode {
            id: id.to_string(),
            endpoint: "http://localhost:8080".to_string(),
            current_load: 0.0,
            max_concurrent: 10,
            active_requests: 0,
            health_status: WorkerHealth::Healthy,
            avg_response_time_ms: 100.0,
            total_requests: 0,
            failed_requests: 0,
            last_health_check: Utc::now(),
            weight: 1.0,
        }
    }

    /// Regression test: `process_response` used to hold a write lock on
    /// `workers` (and `response_history`) all the way through its call to
    /// `update_stats()`, which itself takes a read lock on the same `RwLock`s.
    /// Tokio's `RwLock` is not reentrant, so that self-deadlocked the calling
    /// task forever. This test bounds the call with a timeout so a
    /// regression shows up as a clear failure instead of an indefinite hang.
    #[tokio::test]
    async fn test_process_response_does_not_deadlock() {
        let lb = LoadBalancer::new(LoadBalancerConfig::default());
        lb.add_worker(create_test_worker("worker1")).await.unwrap();

        let response = WorkerResponse {
            request_id: "req1".to_string(),
            worker_id: "worker1".to_string(),
            result: Ok(FeedbackResponse::default()),
            processing_time_ms: 42,
            timestamp: Utc::now(),
        };

        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            lb.process_response(response),
        )
        .await;

        assert!(
            outcome.is_ok(),
            "process_response did not return within 5s -- likely deadlocked"
        );
        assert!(outcome.unwrap().is_ok());

        let stats = lb.get_stats().await;
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.successful_requests, 1);

        // The worker's own counters should also have been updated for real.
        let workers = lb.get_workers().await;
        assert_eq!(workers[0].total_requests, 1);
    }

    /// Regression test for the analogous self-deadlock hazard in
    /// `submit_request` (write locks on `request_queue` and `workers` held
    /// across the `update_stats()` call).
    #[tokio::test]
    async fn test_submit_request_does_not_deadlock() {
        let lb = LoadBalancer::new(LoadBalancerConfig::default());
        lb.add_worker(create_test_worker("worker1")).await.unwrap();

        let request = WorkerRequest {
            id: "test_request".to_string(),
            session: create_test_session_state(),
            timestamp: Utc::now(),
            priority: 5,
            estimated_time_ms: 100,
        };

        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            lb.submit_request(request),
        )
        .await;

        assert!(
            outcome.is_ok(),
            "submit_request did not return within 5s -- likely deadlocked"
        );
        assert!(outcome.unwrap().is_ok());
    }

    /// `requests_per_second` must be computed from the real timestamps on
    /// recorded responses, not hardcoded to zero: a response inside the
    /// trailing window should raise it above zero, while a response from an
    /// hour ago should not contribute to the rate even though it still
    /// counts toward `total_requests`.
    #[tokio::test]
    async fn test_requests_per_second_reflects_real_response_timestamps() {
        let lb = LoadBalancer::new(LoadBalancerConfig::default());
        lb.add_worker(create_test_worker("worker1")).await.unwrap();

        let recent = WorkerResponse {
            request_id: "1".to_string(),
            worker_id: "worker1".to_string(),
            result: Ok(FeedbackResponse::default()),
            processing_time_ms: 10,
            timestamp: Utc::now(),
        };
        lb.process_response(recent).await.unwrap();

        let stats_with_one_recent = lb.get_stats().await;
        assert!(
            stats_with_one_recent.requests_per_second > 0.0,
            "a response inside the trailing window must yield a positive rate"
        );

        let stale = WorkerResponse {
            request_id: "2".to_string(),
            worker_id: "worker1".to_string(),
            result: Ok(FeedbackResponse::default()),
            processing_time_ms: 10,
            timestamp: Utc::now() - chrono::Duration::hours(1),
        };
        lb.process_response(stale).await.unwrap();

        let stats_with_stale_added = lb.get_stats().await;
        assert_eq!(
            stats_with_stale_added.total_requests, 2,
            "total_requests counts all history regardless of age"
        );
        assert_eq!(
            stats_with_stale_added.requests_per_second, stats_with_one_recent.requests_per_second,
            "a response from an hour ago must not inflate the trailing-window rate"
        );
    }
}
