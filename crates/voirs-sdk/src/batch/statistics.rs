//! Batch processing statistics and metrics.

use super::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Statistics for batch processing operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStatistics {
    /// Total number of requests processed
    pub total_requests: usize,

    /// Number of successful requests
    pub successful_requests: usize,

    /// Number of failed requests
    pub failed_requests: usize,

    /// Total processing time in milliseconds
    #[serde(with = "duration_millis")]
    pub total_time: Duration,

    /// Average processing time per request in milliseconds
    #[serde(with = "duration_millis")]
    pub avg_time_per_request: Duration,

    /// Peak memory usage in megabytes
    pub peak_memory_mb: usize,

    /// Number of retries performed
    pub total_retries: usize,

    /// Start time of batch processing (not serialized)
    #[serde(skip)]
    pub start_time: Option<Instant>,

    /// End time of batch processing (not serialized)
    #[serde(skip)]
    pub end_time: Option<Instant>,

    /// Detailed metrics per worker
    pub worker_metrics: HashMap<usize, WorkerMetrics>,
}

// Helper module for duration serialization
mod duration_millis {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_millis().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis))
    }
}

impl BatchStatistics {
    /// Create new batch statistics.
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            total_time: Duration::from_secs(0),
            avg_time_per_request: Duration::from_secs(0),
            peak_memory_mb: 0,
            total_retries: 0,
            start_time: None,
            end_time: None,
            worker_metrics: HashMap::new(),
        }
    }

    /// Calculate success rate as percentage.
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            (self.successful_requests as f64 / self.total_requests as f64) * 100.0
        }
    }

    /// Calculate failure rate as percentage.
    pub fn failure_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            (self.failed_requests as f64 / self.total_requests as f64) * 100.0
        }
    }

    /// Calculate throughput in requests per second.
    pub fn throughput(&self) -> f64 {
        if self.total_time.as_secs() == 0 {
            0.0
        } else {
            self.total_requests as f64 / self.total_time.as_secs_f64()
        }
    }

    /// Update average processing time.
    pub fn update_average_time(&mut self) {
        if self.total_requests > 0 {
            self.avg_time_per_request =
                self.total_time / self.total_requests.try_into().unwrap_or(1);
        }
    }

    /// Add worker metrics.
    pub fn add_worker_metrics(&mut self, worker_id: usize, metrics: WorkerMetrics) {
        self.worker_metrics.insert(worker_id, metrics);
    }

    /// Get worker metrics for a specific worker.
    pub fn get_worker_metrics(&self, worker_id: usize) -> Option<&WorkerMetrics> {
        self.worker_metrics.get(&worker_id)
    }

    /// Get total number of workers used.
    pub fn worker_count(&self) -> usize {
        self.worker_metrics.len()
    }

    /// Generate a summary report.
    pub fn summary(&self) -> String {
        format!(
            "Batch Processing Summary:\n\
             - Total Requests: {}\n\
             - Successful: {} ({:.2}%)\n\
             - Failed: {} ({:.2}%)\n\
             - Total Time: {:?}\n\
             - Avg Time/Request: {:?}\n\
             - Throughput: {:.2} req/s\n\
             - Peak Memory: {} MB\n\
             - Total Retries: {}\n\
             - Workers Used: {}",
            self.total_requests,
            self.successful_requests,
            self.success_rate(),
            self.failed_requests,
            self.failure_rate(),
            self.total_time,
            self.avg_time_per_request,
            self.throughput(),
            self.peak_memory_mb,
            self.total_retries,
            self.worker_count()
        )
    }
}

impl Default for BatchStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics for individual worker performance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerMetrics {
    /// Worker ID
    pub worker_id: usize,

    /// Number of requests processed
    pub requests_processed: usize,

    /// Number of successful requests
    pub successful_requests: usize,

    /// Number of failed requests
    pub failed_requests: usize,

    /// Total processing time in milliseconds
    #[serde(with = "duration_millis")]
    pub total_time: Duration,

    /// Average processing time in milliseconds
    #[serde(with = "duration_millis")]
    pub avg_time: Duration,

    /// Peak memory usage in megabytes
    pub peak_memory_mb: usize,

    /// Idle time in milliseconds
    #[serde(with = "duration_millis")]
    pub idle_time: Duration,
}

impl WorkerMetrics {
    /// Create new worker metrics.
    pub fn new(worker_id: usize) -> Self {
        Self {
            worker_id,
            requests_processed: 0,
            successful_requests: 0,
            failed_requests: 0,
            total_time: Duration::from_secs(0),
            avg_time: Duration::from_secs(0),
            peak_memory_mb: 0,
            idle_time: Duration::from_secs(0),
        }
    }

    /// Calculate worker utilization as percentage.
    pub fn utilization(&self) -> f64 {
        let total = self.total_time + self.idle_time;
        if total.as_secs() == 0 {
            0.0
        } else {
            (self.total_time.as_secs_f64() / total.as_secs_f64()) * 100.0
        }
    }

    /// Calculate success rate.
    pub fn success_rate(&self) -> f64 {
        if self.requests_processed == 0 {
            0.0
        } else {
            (self.successful_requests as f64 / self.requests_processed as f64) * 100.0
        }
    }

    /// Update average time.
    pub fn update_average_time(&mut self) {
        if self.requests_processed > 0 {
            self.avg_time = self.total_time / self.requests_processed.try_into().unwrap_or(1);
        }
    }
}

/// Processing metrics for monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetrics {
    /// Current queue size
    pub queue_size: usize,

    /// Active workers count
    pub active_workers: usize,

    /// Idle workers count
    pub idle_workers: usize,

    /// Current memory usage in megabytes
    pub current_memory_mb: usize,

    /// CPU usage percentage
    pub cpu_usage: f64,

    /// Estimated time remaining
    pub estimated_time_remaining: Option<Duration>,

    /// Current throughput (requests per second)
    pub current_throughput: f64,
}

impl ProcessingMetrics {
    /// Create new processing metrics.
    pub fn new() -> Self {
        Self {
            queue_size: 0,
            active_workers: 0,
            idle_workers: 0,
            current_memory_mb: 0,
            cpu_usage: 0.0,
            estimated_time_remaining: None,
            current_throughput: 0.0,
        }
    }

    /// Calculate total workers.
    pub fn total_workers(&self) -> usize {
        self.active_workers + self.idle_workers
    }

    /// Calculate worker utilization percentage.
    pub fn worker_utilization(&self) -> f64 {
        let total = self.total_workers();
        if total == 0 {
            0.0
        } else {
            (self.active_workers as f64 / total as f64) * 100.0
        }
    }
}

impl Default for ProcessingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_statistics_new() {
        let stats = BatchStatistics::new();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.success_rate(), 0.0);
        assert_eq!(stats.throughput(), 0.0);
    }

    #[test]
    fn test_batch_statistics_success_rate() {
        let mut stats = BatchStatistics::new();
        stats.total_requests = 100;
        stats.successful_requests = 90;
        stats.failed_requests = 10;

        assert_eq!(stats.success_rate(), 90.0);
        assert_eq!(stats.failure_rate(), 10.0);
    }

    #[test]
    fn test_batch_statistics_throughput() {
        let mut stats = BatchStatistics::new();
        stats.total_requests = 100;
        stats.total_time = Duration::from_secs(10);

        assert_eq!(stats.throughput(), 10.0);
    }

    #[test]
    fn test_worker_metrics() {
        let mut metrics = WorkerMetrics::new(0);
        metrics.requests_processed = 10;
        metrics.successful_requests = 9;
        metrics.failed_requests = 1;
        metrics.total_time = Duration::from_secs(5);
        metrics.idle_time = Duration::from_secs(5);

        assert_eq!(metrics.success_rate(), 90.0);
        assert_eq!(metrics.utilization(), 50.0);

        metrics.update_average_time();
        assert!(metrics.avg_time > Duration::from_secs(0));
    }

    #[test]
    fn test_processing_metrics() {
        let mut metrics = ProcessingMetrics::new();
        metrics.active_workers = 8;
        metrics.idle_workers = 2;

        assert_eq!(metrics.total_workers(), 10);
        assert_eq!(metrics.worker_utilization(), 80.0);
    }

    #[test]
    fn test_batch_statistics_summary() {
        let mut stats = BatchStatistics::new();
        stats.total_requests = 100;
        stats.successful_requests = 95;
        stats.failed_requests = 5;
        stats.total_time = Duration::from_secs(60);
        stats.update_average_time();

        let summary = stats.summary();
        assert!(summary.contains("Total Requests: 100"));
        assert!(summary.contains("Successful: 95"));
    }

    #[test]
    fn test_worker_metrics_addition() {
        let mut stats = BatchStatistics::new();
        let metrics = WorkerMetrics::new(0);

        stats.add_worker_metrics(0, metrics.clone());
        assert_eq!(stats.worker_count(), 1);

        let retrieved = stats.get_worker_metrics(0);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().worker_id, 0);
    }
}
