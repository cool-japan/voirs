//! Performance monitoring and optimization

use super::types::*;
use crate::FeedbackError;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time::sleep;

/// Maximum number of recent latency samples retained for averaging.
const MAX_LATENCY_SAMPLES: usize = 200;

/// Performance metrics collector for real-time feedback systems.
///
/// `cpu_usage` and `memory_usage` are real OS-level measurements taken at
/// collection time (see [`PerformanceMonitor::get_cpu_usage`] /
/// [`PerformanceMonitor::get_memory_usage`]).
///
/// The remaining application-level fields are derived from real events
/// recorded via [`PerformanceMonitor::record_feedback_latency`],
/// [`PerformanceMonitor::record_request_result`],
/// [`PerformanceMonitor::record_network_latency`], and
/// [`PerformanceMonitor::set_buffer_utilization`]. They are `Option<f32>`
/// specifically so that "no events recorded yet" (`None`) is distinguishable
/// from "events were recorded and the measured value happens to be zero"
/// (`Some(0.0)`) -- never a fabricated placeholder.
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// CPU usage fraction (0.0 to 1.0), from a real OS query.
    pub cpu_usage: f32,
    /// Resident memory usage of this process in bytes, from a real OS query.
    pub memory_usage: u64,
    /// Average feedback-generation latency in milliseconds over the recent
    /// sample window, or `None` if no latency samples have been recorded.
    pub feedback_latency_ms: Option<f32>,
    /// Average throughput in requests/sec since monitoring began, or `None`
    /// if no time has elapsed yet to measure a rate over.
    pub throughput_rps: Option<f32>,
    /// Fraction of recorded requests that failed (0.0 to 1.0), or `None` if
    /// no requests have been recorded yet.
    pub error_rate: Option<f32>,
    /// Audio buffer utilization (0.0 to 1.0) as last reported by the audio
    /// pipeline, or `None` if never reported.
    pub buffer_utilization: Option<f32>,
    /// Current processing queue depth, as last reported by the pipeline.
    /// Always a real count (0 is a valid, meaningful "empty queue" state).
    pub queue_depth: u32,
    /// Average network latency in milliseconds over the recent sample
    /// window, or `None` if no network latency samples have been recorded.
    pub network_latency_ms: Option<f32>,
}

/// Real, event-driven counters backing the application-level fields of
/// [`PerformanceMetrics`]. These are updated by [`PerformanceMonitor`]'s
/// `record_*`/`set_*` methods as real events occur elsewhere in the
/// system, and read (never fabricated) when a snapshot is collected.
#[derive(Clone)]
struct RealtimeCounters {
    latency_samples_ms: Arc<RwLock<VecDeque<f32>>>,
    network_latency_samples_ms: Arc<RwLock<VecDeque<f32>>>,
    total_requests: Arc<AtomicU64>,
    total_errors: Arc<AtomicU64>,
    buffer_utilization: Arc<RwLock<Option<f32>>>,
    queue_depth: Arc<AtomicU32>,
}

impl RealtimeCounters {
    fn new() -> Self {
        Self {
            latency_samples_ms: Arc::new(RwLock::new(VecDeque::with_capacity(MAX_LATENCY_SAMPLES))),
            network_latency_samples_ms: Arc::new(RwLock::new(VecDeque::with_capacity(
                MAX_LATENCY_SAMPLES,
            ))),
            total_requests: Arc::new(AtomicU64::new(0)),
            total_errors: Arc::new(AtomicU64::new(0)),
            buffer_utilization: Arc::new(RwLock::new(None)),
            queue_depth: Arc::new(AtomicU32::new(0)),
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
    counters: RealtimeCounters,
    monitor_start: Instant,
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
    #[must_use]
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            historical_data: Arc::new(RwLock::new(Vec::new())),
            benchmarks: Arc::new(RwLock::new(HashMap::new())),
            monitoring_enabled: true,
            collection_interval: Duration::from_secs(1),
            counters: RealtimeCounters::new(),
            monitor_start: Instant::now(),
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
        let counters = self.counters.clone();
        let monitor_start = self.monitor_start;

        tokio::spawn(async move {
            loop {
                let current_metrics = Self::collect_system_metrics(&counters, monitor_start).await;

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

    /// Collect a real metrics snapshot immediately, without waiting for the
    /// background collection loop's next tick, and update the cached
    /// current metrics / historical data with it. Useful for callers that
    /// have not (or not yet) called [`Self::start_monitoring`].
    pub async fn collect_now(&self) -> PerformanceMetrics {
        let current_metrics =
            Self::collect_system_metrics(&self.counters, self.monitor_start).await;

        {
            let mut metrics_lock = self.metrics.write().await;
            *metrics_lock = current_metrics.clone();
        }
        {
            let mut historical_lock = self.historical_data.write().await;
            historical_lock.push(TimestampedMetrics {
                timestamp: Instant::now(),
                metrics: current_metrics.clone(),
            });
            if historical_lock.len() > 1000 {
                historical_lock.remove(0);
            }
        }

        current_metrics
    }

    /// Record a real feedback-generation latency sample in milliseconds.
    /// Feeds `feedback_latency_ms` in subsequently collected metrics.
    pub async fn record_feedback_latency(&self, latency_ms: f32) {
        let mut samples = self.counters.latency_samples_ms.write().await;
        if samples.len() >= MAX_LATENCY_SAMPLES {
            samples.pop_front();
        }
        samples.push_back(latency_ms.max(0.0));
    }

    /// Record a real network round-trip latency sample in milliseconds.
    /// Feeds `network_latency_ms` in subsequently collected metrics.
    pub async fn record_network_latency(&self, latency_ms: f32) {
        let mut samples = self.counters.network_latency_samples_ms.write().await;
        if samples.len() >= MAX_LATENCY_SAMPLES {
            samples.pop_front();
        }
        samples.push_back(latency_ms.max(0.0));
    }

    /// Record the real outcome of a processed request. Feeds both
    /// `throughput_rps` (via the running request count) and `error_rate`.
    pub fn record_request_result(&self, success: bool) {
        self.counters.total_requests.fetch_add(1, Ordering::Relaxed);
        if !success {
            self.counters.total_errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Report the real current audio buffer utilization (0.0 to 1.0).
    pub async fn set_buffer_utilization(&self, utilization: f32) {
        let mut value = self.counters.buffer_utilization.write().await;
        *value = Some(utilization.clamp(0.0, 1.0));
    }

    /// Set the real current processing queue depth.
    pub fn set_queue_depth(&self, depth: u32) {
        self.counters.queue_depth.store(depth, Ordering::Relaxed);
    }

    /// Collect current system metrics: real OS queries for CPU/memory, real
    /// event-derived values for the application-level fields.
    async fn collect_system_metrics(
        counters: &RealtimeCounters,
        monitor_start: Instant,
    ) -> PerformanceMetrics {
        PerformanceMetrics {
            cpu_usage: Self::get_cpu_usage().await,
            memory_usage: Self::get_memory_usage().await,
            feedback_latency_ms: Self::average_of(&counters.latency_samples_ms).await,
            throughput_rps: Self::compute_throughput(counters, monitor_start),
            error_rate: Self::compute_error_rate(counters),
            buffer_utilization: *counters.buffer_utilization.read().await,
            queue_depth: counters.queue_depth.load(Ordering::Relaxed),
            network_latency_ms: Self::average_of(&counters.network_latency_samples_ms).await,
        }
    }

    /// Mean of a sample ring buffer, or `None` when empty.
    async fn average_of(samples: &Arc<RwLock<VecDeque<f32>>>) -> Option<f32> {
        let samples = samples.read().await;
        if samples.is_empty() {
            None
        } else {
            Some(samples.iter().sum::<f32>() / samples.len() as f32)
        }
    }

    /// Average requests/sec since monitoring began. `None` only in the
    /// degenerate startup case where essentially no time has elapsed yet;
    /// once any time has passed, zero recorded requests is honestly
    /// reported as `Some(0.0)`, not `None`.
    fn compute_throughput(counters: &RealtimeCounters, monitor_start: Instant) -> Option<f32> {
        let elapsed = monitor_start.elapsed().as_secs_f32();
        if elapsed < f32::EPSILON {
            return None;
        }
        let total = counters.total_requests.load(Ordering::Relaxed) as f32;
        Some(total / elapsed)
    }

    /// Fraction of recorded requests that failed. `None` when zero requests
    /// have ever been recorded (the rate is undefined, not zero).
    fn compute_error_rate(counters: &RealtimeCounters) -> Option<f32> {
        let total = counters.total_requests.load(Ordering::Relaxed);
        if total == 0 {
            return None;
        }
        let errors = counters.total_errors.load(Ordering::Relaxed);
        Some(errors as f32 / total as f32)
    }

    /// Get current CPU usage as a fraction (0.0 to 1.0).
    ///
    /// Linux: computed from two `/proc/stat` samples 50ms apart (mirrors
    /// the pattern used by
    /// [`crate::performance_monitoring::PerformanceMonitor`]). macOS:
    /// computed from `ps -A -o %cpu=`, summed across processes and
    /// normalized by core count (mirrors the pattern used by `voirs-ffi`'s
    /// `MacOSPerformanceMonitor`). Other platforms: `0.0` -- no portable
    /// query is available, so this honestly reports zero rather than a
    /// fabricated reading.
    async fn get_cpu_usage() -> f32 {
        #[cfg(target_os = "linux")]
        {
            fn read_idle_and_total() -> Option<(u64, u64)> {
                let content = std::fs::read_to_string("/proc/stat").ok()?;
                let cpu_line = content.lines().find(|l| l.starts_with("cpu "))?;
                let fields: Vec<u64> = cpu_line
                    .split_whitespace()
                    .skip(1)
                    .filter_map(|t| t.parse::<u64>().ok())
                    .collect();
                if fields.len() < 4 {
                    return None;
                }
                // fields: user nice system idle [iowait irq softirq ...]
                let idle = fields[3];
                let total: u64 = fields.iter().sum();
                Some((idle, total))
            }

            let before = read_idle_and_total();
            sleep(Duration::from_millis(50)).await;
            let after = read_idle_and_total();

            match (before, after) {
                (Some((idle0, total0)), Some((idle1, total1))) => {
                    let total_delta = total1.saturating_sub(total0) as f32;
                    let idle_delta = idle1.saturating_sub(idle0) as f32;
                    if total_delta <= 0.0 {
                        0.0
                    } else {
                        (1.0 - idle_delta / total_delta).clamp(0.0, 1.0)
                    }
                }
                _ => 0.0,
            }
        }

        #[cfg(target_os = "macos")]
        {
            match tokio::process::Command::new("ps")
                .args(["-A", "-o", "%cpu="])
                .output()
                .await
            {
                Ok(output) if output.status.success() => {
                    let text = String::from_utf8_lossy(&output.stdout);
                    let total_percent: f32 = text
                        .lines()
                        .filter_map(|line| line.trim().parse::<f32>().ok())
                        .sum();
                    let core_count = num_cpus::get().max(1) as f32;
                    (total_percent / core_count / 100.0).clamp(0.0, 1.0)
                }
                _ => 0.0,
            }
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            0.0
        }
    }

    /// Get current process resident memory usage in bytes.
    ///
    /// Linux: parsed from `/proc/self/status` (`VmRSS`, reported in KiB).
    /// macOS: parsed from `ps -o rss= -p <pid>` (RSS reported in KiB).
    /// Other platforms: `0` -- no portable query available.
    async fn get_memory_usage() -> u64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = tokio::fs::read_to_string("/proc/self/status").await {
                if let Some(kb) = content
                    .lines()
                    .find(|line| line.starts_with("VmRSS:"))
                    .and_then(|line| line.split_whitespace().nth(1))
                    .and_then(|token| token.parse::<u64>().ok())
                {
                    return kb * 1024;
                }
            }
            0
        }

        #[cfg(target_os = "macos")]
        {
            let pid = std::process::id().to_string();
            match tokio::process::Command::new("ps")
                .args(["-o", "rss=", "-p", &pid])
                .output()
                .await
            {
                Ok(output) if output.status.success() => String::from_utf8_lossy(&output.stdout)
                    .trim()
                    .parse::<u64>()
                    .map(|kb| kb * 1024)
                    .unwrap_or(0),
                _ => 0,
            }
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            0
        }
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
            durations
                .iter()
                .map(std::time::Duration::as_nanos)
                .sum::<u128>() as u64
                / u64::from(iterations),
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

        // Latency optimization recommendations -- only evaluated once real
        // latency samples have actually been recorded.
        if let Some(latency) = metrics.feedback_latency_ms {
            if latency > 50.0 {
                recommendations.push(PerformanceRecommendation {
                    category: OptimizationCategory::Latency,
                    severity: OptimizationSeverity::High,
                    description: "High feedback latency detected".to_string(),
                    suggested_action:
                        "Consider reducing buffer size or optimizing processing pipeline"
                            .to_string(),
                    expected_improvement: 0.4,
                });
            }
        }

        // Throughput optimization recommendations -- a system that has not
        // yet served any request must not be flagged as "low throughput".
        if let Some(throughput) = metrics.throughput_rps {
            if throughput < 10.0 {
                recommendations.push(PerformanceRecommendation {
                    category: OptimizationCategory::Throughput,
                    severity: OptimizationSeverity::Medium,
                    description: "Low throughput detected".to_string(),
                    suggested_action:
                        "Consider implementing parallel processing or connection pooling"
                            .to_string(),
                    expected_improvement: 0.5,
                });
            }
        }

        // Network optimization recommendations -- only evaluated once real
        // network latency samples have actually been recorded.
        if let Some(network_latency) = metrics.network_latency_ms {
            if network_latency > 100.0 {
                recommendations.push(PerformanceRecommendation {
                    category: OptimizationCategory::Network,
                    severity: OptimizationSeverity::Medium,
                    description: "High network latency detected".to_string(),
                    suggested_action: "Consider implementing request batching or local caching"
                        .to_string(),
                    expected_improvement: 0.3,
                });
            }
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
        assert_eq!(metrics.feedback_latency_ms, None);
    }

    #[test]
    fn test_performance_metrics_default() {
        let metrics = PerformanceMetrics::default();
        assert_eq!(metrics.cpu_usage, 0.0);
        assert_eq!(metrics.memory_usage, 0);
        assert_eq!(metrics.error_rate, None);
        assert_eq!(metrics.buffer_utilization, None);
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
            metrics_lock.feedback_latency_ms = Some(60.0); // High latency
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

    /// A system that has not yet served any request must not be flagged as
    /// "low throughput" -- `None` (no data) must be treated differently
    /// from a genuinely low measured rate.
    #[tokio::test]
    async fn test_no_throughput_recommendation_without_data() {
        let monitor = PerformanceMonitor::new();
        let recommendations = monitor.get_optimization_recommendations().await;
        assert!(!recommendations
            .iter()
            .any(|r| matches!(r.category, OptimizationCategory::Throughput)));
    }

    #[tokio::test]
    async fn test_performance_report_generation() {
        let monitor = PerformanceMonitor::new();
        let report = monitor.generate_performance_report().await;

        assert_eq!(report.current_metrics.cpu_usage, 0.0);
        assert!(report.historical_data.is_empty());
        assert!(report.benchmarks.is_empty());
        // A freshly-created monitor with no recorded events has nothing to
        // flag: no optimization recommendations should be generated.
        assert!(report.recommendations.is_empty());
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

    /// Recording real latency samples must change the reported average --
    /// proving `feedback_latency_ms` is a real function of recorded events,
    /// not a constant.
    #[tokio::test]
    async fn test_record_feedback_latency_changes_reported_average() {
        let monitor = PerformanceMonitor::new();

        let before = monitor.collect_now().await;
        assert_eq!(before.feedback_latency_ms, None);

        monitor.record_feedback_latency(20.0).await;
        monitor.record_feedback_latency(40.0).await;

        let after = monitor.collect_now().await;
        assert_eq!(after.feedback_latency_ms, Some(30.0));
    }

    /// Recording request outcomes must change both throughput and error
    /// rate, and a zero-error run must report an honest `Some(0.0)` (not
    /// `None`) once at least one request has been recorded.
    #[tokio::test]
    async fn test_record_request_result_changes_throughput_and_error_rate() {
        let monitor = PerformanceMonitor::new();

        let before = monitor.collect_now().await;
        assert_eq!(before.error_rate, None);

        monitor.record_request_result(true);
        monitor.record_request_result(true);
        monitor.record_request_result(false);

        let after = monitor.collect_now().await;
        assert_eq!(after.error_rate, Some(1.0 / 3.0));
        assert!(after.throughput_rps.is_some());
        assert!(after.throughput_rps.unwrap() > 0.0);
    }

    /// `set_buffer_utilization` / `set_queue_depth` must be reflected in
    /// the next collected snapshot.
    #[tokio::test]
    async fn test_buffer_utilization_and_queue_depth_are_real() {
        let monitor = PerformanceMonitor::new();

        let before = monitor.collect_now().await;
        assert_eq!(before.buffer_utilization, None);
        assert_eq!(before.queue_depth, 0);

        monitor.set_buffer_utilization(0.42).await;
        monitor.set_queue_depth(7);

        let after = monitor.collect_now().await;
        assert_eq!(after.buffer_utilization, Some(0.42));
        assert_eq!(after.queue_depth, 7);
    }

    /// `set_buffer_utilization` must clamp to the documented 0.0-1.0 range
    /// rather than storing an out-of-range value verbatim.
    #[tokio::test]
    async fn test_buffer_utilization_is_clamped() {
        let monitor = PerformanceMonitor::new();
        monitor.set_buffer_utilization(1.5).await;
        let after = monitor.collect_now().await;
        assert_eq!(after.buffer_utilization, Some(1.0));
    }

    /// CPU usage must be a real OS query result: on supported platforms it
    /// succeeds and returns an in-range fraction; on unsupported platforms
    /// it honestly reports 0.0. Either way it must never panic.
    #[tokio::test]
    async fn test_cpu_usage_is_real_or_honestly_zero() {
        let usage = PerformanceMonitor::get_cpu_usage().await;
        assert!((0.0..=1.0).contains(&usage), "usage = {usage}");
    }

    /// Memory usage must be a real OS query result for this very process --
    /// on Linux/macOS it must be nonzero (this test process is definitely
    /// resident in memory); elsewhere it honestly reports 0.
    #[tokio::test]
    async fn test_memory_usage_is_real() {
        let usage = PerformanceMonitor::get_memory_usage().await;
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        assert!(usage > 0, "expected a real nonzero RSS reading");
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        let _ = usage;
    }
}
