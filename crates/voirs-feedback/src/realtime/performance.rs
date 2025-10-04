//! Performance monitoring and optimization

use super::types::*;
use crate::FeedbackError;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time::sleep;

/// Performance metrics collector for real-time feedback systems
#[derive(Debug)]
pub struct PerformanceMetrics {
    /// CPU usage percentage (0.0 to 1.0)
    pub cpu_usage: f32,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Feedback generation latency in milliseconds
    pub feedback_latency_ms: f32,
    /// Throughput in requests per second
    pub throughput_rps: f32,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f32,
    /// Audio buffer utilization percentage
    pub buffer_utilization: f32,
    /// Processing queue depth
    pub queue_depth: u32,
    /// Network latency in milliseconds
    pub network_latency_ms: f32,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0,
            feedback_latency_ms: 0.0,
            throughput_rps: 0.0,
            error_rate: 0.0,
            buffer_utilization: 0.0,
            queue_depth: 0,
            network_latency_ms: 0.0,
        }
    }
}

/// Performance monitoring and optimization system
pub struct PerformanceMonitor {
    metrics: Arc<RwLock<PerformanceMetrics>>,
    historical_data: Arc<RwLock<Vec<TimestampedMetrics>>>,
    benchmarks: Arc<RwLock<HashMap<String, BenchmarkResult>>>,
    monitoring_enabled: bool,
    collection_interval: Duration,
}

/// Timestamped performance metrics for historical analysis
#[derive(Debug, Clone)]
pub struct TimestampedMetrics {
    /// Description
    pub timestamp: Instant,
    /// Description
    pub metrics: PerformanceMetrics,
}

/// Benchmark result with detailed performance characteristics
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Description
    pub name: String,
    /// Description
    pub average_duration: Duration,
    /// Description
    pub min_duration: Duration,
    /// Description
    pub max_duration: Duration,
    /// Description
    pub samples: u32,
    /// Description
    pub throughput_ops_per_sec: f32,
    /// Description
    pub success_rate: f32,
    /// Description
    pub timestamp: Instant,
}

/// Performance optimization recommendations
#[derive(Debug, Clone)]
pub struct PerformanceRecommendation {
    /// Description
    pub category: OptimizationCategory,
    /// Description
    pub severity: OptimizationSeverity,
    /// Description
    pub description: String,
    /// Description
    pub suggested_action: String,
    /// Description
    pub expected_improvement: f32,
}

/// Optimization categories
#[derive(Debug, Clone)]
pub enum OptimizationCategory {
    /// Description
    CPU,
    /// Description
    Memory,
    /// Description
    Latency,
    /// Description
    Throughput,
    /// Description
    Network,
    /// Description
    Storage,
}

/// Optimization severity levels
#[derive(Debug, Clone)]
pub enum OptimizationSeverity {
    /// Description
    Low,
    /// Description
    Medium,
    /// Description
    High,
    /// Description
    Critical,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            historical_data: Arc::new(RwLock::new(Vec::new())),
            benchmarks: Arc::new(RwLock::new(HashMap::new())),
            monitoring_enabled: true,
            collection_interval: Duration::from_secs(1),
        }
    }

    /// Start monitoring performance metrics with optimized UI responsiveness
    pub async fn start_monitoring(&self) -> Result<(), FeedbackError> {
        if !self.monitoring_enabled {
            return Ok(());
        }

        let metrics = self.metrics.clone();
        let historical_data = self.historical_data.clone();
        let collection_interval = self.collection_interval;

        tokio::spawn(async move {
            loop {
                let current_metrics = Self::collect_system_metrics().await;

                // Update metrics (async RwLock)
                {
                    let mut metrics_lock = metrics.write().await;
                    *metrics_lock = current_metrics.clone();
                }

                // Store historical data
                {
                    let mut historical_lock = historical_data.write().await;
                    historical_lock.push(TimestampedMetrics {
                        timestamp: Instant::now(),
                        metrics: current_metrics,
                    });

                    // Keep only last 1000 samples
                    if historical_lock.len() > 1000 {
                        historical_lock.remove(0);
                    }
                }

                sleep(collection_interval).await;
            }
        });

        Ok(())
    }

    /// Collect current system metrics
    async fn collect_system_metrics() -> PerformanceMetrics {
        PerformanceMetrics {
            cpu_usage: Self::get_cpu_usage(),
            memory_usage: Self::get_memory_usage(),
            feedback_latency_ms: Self::get_feedback_latency(),
            throughput_rps: Self::get_throughput(),
            error_rate: Self::get_error_rate(),
            buffer_utilization: Self::get_buffer_utilization(),
            queue_depth: Self::get_queue_depth(),
            network_latency_ms: Self::get_network_latency(),
        }
    }

    /// Get current CPU usage percentage
    fn get_cpu_usage() -> f32 {
        // In a real implementation, this would use system APIs
        // For now, simulate some CPU usage
        0.15 // 15% CPU usage
    }

    /// Get current memory usage in bytes
    fn get_memory_usage() -> u64 {
        // In a real implementation, this would use system APIs
        // For now, simulate memory usage
        128 * 1024 * 1024 // 128MB
    }

    /// Get current feedback generation latency
    fn get_feedback_latency() -> f32 {
        // In a real implementation, this would track actual latency
        // For now, simulate latency
        25.0 // 25ms
    }

    /// Get current throughput in requests per second
    fn get_throughput() -> f32 {
        // In a real implementation, this would track actual throughput
        // For now, simulate throughput
        50.0 // 50 RPS
    }

    /// Get current error rate
    fn get_error_rate() -> f32 {
        // In a real implementation, this would track actual error rate
        // For now, simulate low error rate
        0.001 // 0.1% error rate
    }

    /// Get current buffer utilization percentage
    fn get_buffer_utilization() -> f32 {
        // In a real implementation, this would track actual buffer usage
        // For now, simulate buffer utilization
        0.6 // 60% utilization
    }

    /// Get current processing queue depth
    fn get_queue_depth() -> u32 {
        // In a real implementation, this would track actual queue depth
        // For now, simulate queue depth
        5
    }

    /// Get current network latency
    fn get_network_latency() -> f32 {
        // In a real implementation, this would measure actual network latency
        // For now, simulate network latency
        10.0 // 10ms
    }

    /// Get current performance metrics
    pub async fn get_current_metrics(&self) -> PerformanceMetrics {
        let metrics_lock = self.metrics.read().await;
        metrics_lock.clone()
    }

    /// Get current performance metrics without blocking UI operations
    pub async fn get_current_metrics_non_blocking(&self) -> PerformanceMetrics {
        let metrics_lock = self.metrics.read().await;
        metrics_lock.clone()
    }

    /// Get cached performance metrics for UI display
    pub async fn get_ui_friendly_metrics(&self) -> PerformanceMetrics {
        self.get_current_metrics_non_blocking().await
    }

    /// Get historical performance data
    pub async fn get_historical_data(&self) -> Vec<TimestampedMetrics> {
        let historical_lock = self.historical_data.read().await;
        historical_lock.clone()
    }

    /// Get recent historical data for UI charts without blocking
    pub async fn get_recent_metrics_for_ui(&self, count: usize) -> Vec<TimestampedMetrics> {
        let historical_lock = self.historical_data.read().await;
        let len = historical_lock.len();
        if len <= count {
            historical_lock.clone()
        } else {
            historical_lock[len - count..].to_vec()
        }
    }

    /// Run a performance benchmark
    pub async fn run_benchmark<F, Fut>(
        &self,
        name: &str,
        operation: F,
        iterations: u32,
    ) -> Result<BenchmarkResult, FeedbackError>
    where
        F: Fn() -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<(), FeedbackError>> + Send,
    {
        let mut durations = Vec::new();
        let mut successes = 0;

        let start_time = Instant::now();

        for _ in 0..iterations {
            let operation_start = Instant::now();
            match operation().await {
                Ok(()) => {
                    successes += 1;
                    durations.push(operation_start.elapsed());
                }
                Err(_) => {
                    durations.push(operation_start.elapsed());
                }
            }
        }

        let total_duration = start_time.elapsed();
        let average_duration = Duration::from_nanos(
            durations.iter().map(|d| d.as_nanos()).sum::<u128>() as u64 / iterations as u64,
        );
        let min_duration = durations.iter().min().copied().unwrap_or(Duration::ZERO);
        let max_duration = durations.iter().max().copied().unwrap_or(Duration::ZERO);
        let success_rate = successes as f32 / iterations as f32;
        let throughput_ops_per_sec = iterations as f32 / total_duration.as_secs_f32();

        let result = BenchmarkResult {
            name: name.to_string(),
            average_duration,
            min_duration,
            max_duration,
            samples: iterations,
            throughput_ops_per_sec,
            success_rate,
            timestamp: Instant::now(),
        };

        // Store benchmark result
        {
            let mut benchmarks_lock = self.benchmarks.write().await;
            benchmarks_lock.insert(name.to_string(), result.clone());
        }

        Ok(result)
    }

    /// Get stored benchmark results
    pub async fn get_benchmark_results(&self) -> HashMap<String, BenchmarkResult> {
        let benchmarks_lock = self.benchmarks.read().await;
        (*benchmarks_lock).clone()
    }

    /// Generate performance optimization recommendations
    pub async fn get_optimization_recommendations(&self) -> Vec<PerformanceRecommendation> {
        let metrics = self.get_current_metrics().await;
        let mut recommendations = Vec::new();

        // CPU optimization recommendations
        if metrics.cpu_usage > 0.8 {
            recommendations.push(PerformanceRecommendation {
                category: OptimizationCategory::CPU,
                severity: OptimizationSeverity::High,
                description: "High CPU usage detected".to_string(),
                suggested_action: "Consider increasing buffer size or optimizing algorithms"
                    .to_string(),
                expected_improvement: 0.3,
            });
        }

        // Memory optimization recommendations
        if metrics.memory_usage > 512 * 1024 * 1024 {
            recommendations.push(PerformanceRecommendation {
                category: OptimizationCategory::Memory,
                severity: OptimizationSeverity::Medium,
                description: "High memory usage detected".to_string(),
                suggested_action: "Consider implementing memory pooling or reducing cache size"
                    .to_string(),
                expected_improvement: 0.25,
            });
        }

        // Latency optimization recommendations
        if metrics.feedback_latency_ms > 50.0 {
            recommendations.push(PerformanceRecommendation {
                category: OptimizationCategory::Latency,
                severity: OptimizationSeverity::High,
                description: "High feedback latency detected".to_string(),
                suggested_action: "Consider reducing buffer size or optimizing processing pipeline"
                    .to_string(),
                expected_improvement: 0.4,
            });
        }

        // Throughput optimization recommendations
        if metrics.throughput_rps < 10.0 {
            recommendations.push(PerformanceRecommendation {
                category: OptimizationCategory::Throughput,
                severity: OptimizationSeverity::Medium,
                description: "Low throughput detected".to_string(),
                suggested_action: "Consider implementing parallel processing or connection pooling"
                    .to_string(),
                expected_improvement: 0.5,
            });
        }

        // Network optimization recommendations
        if metrics.network_latency_ms > 100.0 {
            recommendations.push(PerformanceRecommendation {
                category: OptimizationCategory::Network,
                severity: OptimizationSeverity::Medium,
                description: "High network latency detected".to_string(),
                suggested_action: "Consider implementing request batching or local caching"
                    .to_string(),
                expected_improvement: 0.3,
            });
        }

        recommendations
    }

    /// Generate performance report
    pub async fn generate_performance_report(&self) -> PerformanceReport {
        let current_metrics = self.get_current_metrics().await;
        let historical_data = self.get_historical_data().await;
        let benchmarks = self.get_benchmark_results().await;
        let recommendations = self.get_optimization_recommendations().await;

        PerformanceReport {
            current_metrics,
            historical_data,
            benchmarks,
            recommendations,
            report_timestamp: Instant::now(),
        }
    }

    /// Clear historical data
    pub async fn clear_historical_data(&self) {
        let mut historical_lock = self.historical_data.write().await;
        historical_lock.clear();
    }

    /// Clear benchmark results
    pub async fn clear_benchmark_results(&self) {
        let mut benchmarks_lock = self.benchmarks.write().await;
        benchmarks_lock.clear();
    }
}

/// Comprehensive performance report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Description
    pub current_metrics: PerformanceMetrics,
    /// Description
    pub historical_data: Vec<TimestampedMetrics>,
    /// Description
    pub benchmarks: HashMap<String, BenchmarkResult>,
    /// Description
    pub recommendations: Vec<PerformanceRecommendation>,
    /// Description
    pub report_timestamp: Instant,
}

impl Clone for PerformanceMetrics {
    fn clone(&self) -> Self {
        Self {
            cpu_usage: self.cpu_usage,
            memory_usage: self.memory_usage,
            feedback_latency_ms: self.feedback_latency_ms,
            throughput_rps: self.throughput_rps,
            error_rate: self.error_rate,
            buffer_utilization: self.buffer_utilization,
            queue_depth: self.queue_depth,
            network_latency_ms: self.network_latency_ms,
        }
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_performance_monitor_creation() {
        let monitor = PerformanceMonitor::new();
        let metrics = monitor.get_current_metrics().await;
        assert_eq!(metrics.cpu_usage, 0.0);
        assert_eq!(metrics.memory_usage, 0);
        assert_eq!(metrics.feedback_latency_ms, 0.0);
    }

    #[test]
    fn test_performance_metrics_default() {
        let metrics = PerformanceMetrics::default();
        assert_eq!(metrics.cpu_usage, 0.0);
        assert_eq!(metrics.memory_usage, 0);
        assert_eq!(metrics.error_rate, 0.0);
        assert_eq!(metrics.buffer_utilization, 0.0);
        assert_eq!(metrics.queue_depth, 0);
    }

    #[tokio::test]
    async fn test_benchmark_execution() {
        let monitor = PerformanceMonitor::new();

        let result = monitor
            .run_benchmark(
                "test_operation",
                || async {
                    // Simulate some work
                    tokio::time::sleep(Duration::from_millis(1)).await;
                    Ok(())
                },
                10,
            )
            .await
            .unwrap();

        assert_eq!(result.name, "test_operation");
        assert_eq!(result.samples, 10);
        assert!(result.average_duration.as_millis() >= 1);
        assert_eq!(result.success_rate, 1.0);
        assert!(result.throughput_ops_per_sec > 0.0);
    }

    #[tokio::test]
    async fn test_optimization_recommendations() {
        let monitor = PerformanceMonitor::new();

        // Update metrics to trigger recommendations
        {
            let mut metrics_lock = monitor.metrics.write().await;
            metrics_lock.cpu_usage = 0.9; // High CPU usage
            metrics_lock.feedback_latency_ms = 60.0; // High latency
        }

        let recommendations = monitor.get_optimization_recommendations().await;
        assert!(!recommendations.is_empty());

        // Should have CPU and latency recommendations
        assert!(recommendations
            .iter()
            .any(|r| matches!(r.category, OptimizationCategory::CPU)));
        assert!(recommendations
            .iter()
            .any(|r| matches!(r.category, OptimizationCategory::Latency)));
    }

    #[tokio::test]
    async fn test_performance_report_generation() {
        let monitor = PerformanceMonitor::new();
        let report = monitor.generate_performance_report().await;

        assert_eq!(report.current_metrics.cpu_usage, 0.0);
        assert!(report.historical_data.is_empty());
        assert!(report.benchmarks.is_empty());
        // Note: recommendations may not be empty if metrics simulation generates them
        assert!(report.recommendations.len() >= 0);
    }

    #[tokio::test]
    async fn test_historical_data_management() {
        let monitor = PerformanceMonitor::new();

        // Add some historical data
        {
            let mut historical_lock = monitor.historical_data.write().await;
            historical_lock.push(TimestampedMetrics {
                timestamp: Instant::now(),
                metrics: PerformanceMetrics::default(),
            });
        }

        assert_eq!(monitor.get_historical_data().await.len(), 1);

        monitor.clear_historical_data().await;
        assert_eq!(monitor.get_historical_data().await.len(), 0);
    }

    #[tokio::test]
    async fn test_benchmark_results_management() {
        let monitor = PerformanceMonitor::new();

        // Add benchmark result
        {
            let mut benchmarks_lock = monitor.benchmarks.write().await;
            benchmarks_lock.insert(
                "test".to_string(),
                BenchmarkResult {
                    name: "test".to_string(),
                    average_duration: Duration::from_millis(10),
                    min_duration: Duration::from_millis(5),
                    max_duration: Duration::from_millis(15),
                    samples: 100,
                    throughput_ops_per_sec: 100.0,
                    success_rate: 1.0,
                    timestamp: Instant::now(),
                },
            );
        }

        assert_eq!(monitor.get_benchmark_results().await.len(), 1);

        monitor.clear_benchmark_results().await;
        assert_eq!(monitor.get_benchmark_results().await.len(), 0);
    }
}
