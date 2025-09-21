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
            return Err(format!("Worker with ID {} not found", worker_id));
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
        let target_weight = rand::random::<f64>() * total_weight;

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
                    .unwrap()
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
                a_score.partial_cmp(&b_score).unwrap()
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
                a_score.partial_cmp(&b_score).unwrap()
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

        // Add to queue
        let mut queue = self.request_queue.write().await;
        if queue.len() >= self.config.max_queue_size {
            return Err("Request queue is full".to_string());
        }

        queue.push_back(request.clone());

        // Update worker active requests
        let mut workers = self.workers.write().await;
        if let Some(worker) = workers.get_mut(&worker_id) {
            worker.active_requests += 1;
        }

        // Update statistics
        self.update_stats().await;

        Ok(request.id)
    }

    /// Process a response from a worker
    pub async fn process_response(&self, response: WorkerResponse) -> Result<(), String> {
        // Update worker statistics
        let mut workers = self.workers.write().await;
        if let Some(worker) = workers.get_mut(&response.worker_id) {
            worker.active_requests = worker.active_requests.saturating_sub(1);
            worker.total_requests += 1;

            if response.result.is_err() {
                worker.failed_requests += 1;
            }

            // Update average response time (exponential moving average)
            let alpha = 0.1; // Smoothing factor
            worker.avg_response_time_ms = alpha * response.processing_time_ms as f64
                + (1.0 - alpha) * worker.avg_response_time_ms;

            // Update current load
            worker.current_load = worker.active_requests as f64 / worker.max_concurrent as f64;
        }

        // Store response for metrics
        let mut history = self.response_history.write().await;
        history.push(response);

        // Keep only recent responses (last 1000)
        if history.len() > 1000 {
            history.drain(0..500);
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
            stats.avg_response_time_ms = total_time as f64 / history.len() as f64;
        }

        // Calculate worker utilization
        if !workers.is_empty() {
            let total_utilization: f64 = workers.values().map(|w| w.current_load).sum();
            stats.worker_utilization = total_utilization / workers.len() as f64;
        }

        // Calculate requests per second - simplified to avoid potential deadlock
        stats.requests_per_second = 0.0; // Simplified for now
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
            Err(format!("Worker with ID {} not found", worker_id))
        }
    }

    /// Start health check monitoring
    pub async fn start_health_monitoring(&self) {
        let workers = self.workers.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            loop {
                sleep(Duration::from_secs(
                    config.health_check_interval_seconds as u64,
                ))
                .await;

                let mut workers_guard = workers.write().await;
                let now = Utc::now();

                for worker in workers_guard.values_mut() {
                    let time_since_check = now.signed_duration_since(worker.last_health_check);

                    // Mark as unknown if no health check for too long
                    if time_since_check.num_seconds()
                        > config.health_check_interval_seconds as i64 * 2
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
}
