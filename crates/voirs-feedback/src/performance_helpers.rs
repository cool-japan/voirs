//! Performance monitoring helper functions
//!
//! This module provides convenient functions for performance monitoring,
//! profiling, and optimization analysis.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Simple performance timer
///
/// # Examples
///
/// ```
/// use voirs_feedback::performance_helpers::Timer;
///
/// let timer = Timer::start();
/// // ... do some work ...
/// let elapsed = timer.elapsed();
/// println!("Operation took: {:?}", elapsed);
/// ```
#[derive(Debug, Clone)]
pub struct Timer {
    start: Instant,
}

impl Timer {
    /// Start a new timer
    #[must_use]
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Get elapsed time since start
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Get elapsed time in milliseconds
    #[must_use]
    pub fn elapsed_ms(&self) -> u128 {
        self.elapsed().as_millis()
    }

    /// Get elapsed time in microseconds
    #[must_use]
    pub fn elapsed_micros(&self) -> u128 {
        self.elapsed().as_micros()
    }

    /// Restart the timer
    pub fn restart(&mut self) {
        self.start = Instant::now();
    }
}

/// Performance snapshot for monitoring
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    /// Timestamp of snapshot
    pub timestamp: Instant,
    /// CPU usage percentage (if available)
    pub cpu_usage: Option<f32>,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Active operations count
    pub active_operations: usize,
    /// Throughput (operations per second)
    pub throughput: f32,
    /// Average latency in milliseconds
    pub avg_latency_ms: f32,
}

impl PerformanceSnapshot {
    /// Create a new performance snapshot
    #[must_use]
    pub fn new(
        memory_usage: usize,
        active_operations: usize,
        throughput: f32,
        avg_latency_ms: f32,
    ) -> Self {
        Self {
            timestamp: Instant::now(),
            cpu_usage: None,
            memory_usage,
            active_operations,
            throughput,
            avg_latency_ms,
        }
    }

    /// Check if performance is within acceptable bounds
    #[must_use]
    pub fn is_healthy(&self, max_latency_ms: f32, max_memory_mb: f32) -> bool {
        let memory_mb = self.memory_usage as f32 / (1024.0 * 1024.0);
        self.avg_latency_ms <= max_latency_ms && memory_mb <= max_memory_mb
    }
}

/// Simple operation profiler
#[derive(Debug)]
pub struct OperationProfiler {
    operations: HashMap<String, Vec<Duration>>,
}

impl Default for OperationProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl OperationProfiler {
    /// Create a new operation profiler
    #[must_use]
    pub fn new() -> Self {
        Self {
            operations: HashMap::new(),
        }
    }

    /// Record an operation duration
    pub fn record(&mut self, operation: &str, duration: Duration) {
        self.operations
            .entry(operation.to_string())
            .or_default()
            .push(duration);
    }

    /// Get statistics for an operation
    #[must_use]
    pub fn stats(&self, operation: &str) -> Option<OperationStats> {
        self.operations.get(operation).map(|durations| {
            let count = durations.len();
            let total: Duration = durations.iter().sum();
            let avg = total / count as u32;

            let mut sorted = durations.clone();
            sorted.sort();

            let p50 = sorted[count / 2];
            let p95 = sorted[count * 95 / 100];
            let p99 = sorted[count * 99 / 100];

            let min = *sorted.first().unwrap();
            let max = *sorted.last().unwrap();

            OperationStats {
                operation: operation.to_string(),
                count,
                total,
                avg,
                min,
                max,
                p50,
                p95,
                p99,
            }
        })
    }

    /// Get all operation names
    #[must_use]
    pub fn operations(&self) -> Vec<String> {
        self.operations.keys().cloned().collect()
    }

    /// Clear all recorded data
    pub fn clear(&mut self) {
        self.operations.clear();
    }
}

/// Statistics for a profiled operation
#[derive(Debug, Clone)]
pub struct OperationStats {
    /// Operation name
    pub operation: String,
    /// Number of samples
    pub count: usize,
    /// Total duration
    pub total: Duration,
    /// Average duration
    pub avg: Duration,
    /// Minimum duration
    pub min: Duration,
    /// Maximum duration
    pub max: Duration,
    /// 50th percentile (median)
    pub p50: Duration,
    /// 95th percentile
    pub p95: Duration,
    /// 99th percentile
    pub p99: Duration,
}

impl OperationStats {
    /// Format stats as a readable string
    #[must_use]
    pub fn format(&self) -> String {
        format!(
            "{}: count={}, avg={:?}, p50={:?}, p95={:?}, p99={:?}, min={:?}, max={:?}",
            self.operation, self.count, self.avg, self.p50, self.p95, self.p99, self.min, self.max
        )
    }
}

/// Calculate throughput (operations per second)
#[must_use]
pub fn calculate_throughput(operation_count: usize, duration: Duration) -> f32 {
    if duration.as_secs_f32() == 0.0 {
        return 0.0;
    }
    operation_count as f32 / duration.as_secs_f32()
}

/// Calculate average latency from total time and count
#[must_use]
pub fn calculate_avg_latency(total_duration: Duration, count: usize) -> Duration {
    if count == 0 {
        return Duration::from_secs(0);
    }
    total_duration / count as u32
}

/// Check if latency is within target
#[must_use]
pub fn is_latency_acceptable(latency: Duration, target_ms: u64) -> bool {
    latency.as_millis() <= u128::from(target_ms)
}

/// Calculate Real-Time Factor (RTF)
///
/// RTF = `processing_time` / `audio_duration`
/// RTF < 1.0 means faster than real-time
#[must_use]
pub fn calculate_rtf(processing_time: Duration, audio_duration: Duration) -> f32 {
    if audio_duration.as_secs_f32() == 0.0 {
        return f32::INFINITY;
    }
    processing_time.as_secs_f32() / audio_duration.as_secs_f32()
}

/// Estimate memory usage per operation
#[must_use]
pub fn estimate_memory_per_operation(total_memory: usize, operation_count: usize) -> usize {
    if operation_count == 0 {
        return 0;
    }
    total_memory / operation_count
}

/// Format bytes as human-readable string
#[must_use]
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Format duration as human-readable string with appropriate unit
#[must_use]
pub fn format_duration_auto(duration: Duration) -> String {
    let nanos = duration.as_nanos();

    if nanos < 1_000 {
        format!("{nanos}ns")
    } else if nanos < 1_000_000 {
        format!("{:.2}µs", nanos as f64 / 1_000.0)
    } else if nanos < 1_000_000_000 {
        format!("{:.2}ms", nanos as f64 / 1_000_000.0)
    } else {
        format!("{:.2}s", duration.as_secs_f64())
    }
}

/// Simple rate limiter
#[derive(Debug)]
pub struct RateLimiter {
    max_operations: usize,
    time_window: Duration,
    operations: Vec<Instant>,
}

impl RateLimiter {
    /// Create a new rate limiter
    #[must_use]
    pub fn new(max_operations: usize, time_window: Duration) -> Self {
        Self {
            max_operations,
            time_window,
            operations: Vec::with_capacity(max_operations),
        }
    }

    /// Check if operation is allowed
    pub fn allow(&mut self) -> bool {
        let now = Instant::now();

        // Remove old operations outside time window
        self.operations
            .retain(|&op_time| now.duration_since(op_time) < self.time_window);

        if self.operations.len() < self.max_operations {
            self.operations.push(now);
            true
        } else {
            false
        }
    }

    /// Get current operation count in window
    #[must_use]
    pub fn current_count(&self) -> usize {
        self.operations.len()
    }

    /// Reset the rate limiter
    pub fn reset(&mut self) {
        self.operations.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_timer() {
        let timer = Timer::start();
        thread::sleep(Duration::from_millis(10));
        assert!(timer.elapsed_ms() >= 10);
    }

    #[test]
    fn test_operation_profiler() {
        let mut profiler = OperationProfiler::new();

        profiler.record("test_op", Duration::from_millis(10));
        profiler.record("test_op", Duration::from_millis(20));
        profiler.record("test_op", Duration::from_millis(30));

        let stats = profiler.stats("test_op").unwrap();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.min, Duration::from_millis(10));
        assert_eq!(stats.max, Duration::from_millis(30));
    }

    #[test]
    fn test_calculate_throughput() {
        let throughput = calculate_throughput(1000, Duration::from_secs(1));
        assert_eq!(throughput, 1000.0);
    }

    #[test]
    fn test_calculate_rtf() {
        let rtf = calculate_rtf(Duration::from_millis(50), Duration::from_millis(100));
        assert_eq!(rtf, 0.5); // Faster than real-time
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_format_duration_auto() {
        assert_eq!(format_duration_auto(Duration::from_nanos(500)), "500ns");
        assert_eq!(format_duration_auto(Duration::from_micros(500)), "500.00µs");
        assert_eq!(format_duration_auto(Duration::from_millis(500)), "500.00ms");
        assert_eq!(format_duration_auto(Duration::from_secs(2)), "2.00s");
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(3, Duration::from_secs(1));

        assert!(limiter.allow());
        assert!(limiter.allow());
        assert!(limiter.allow());
        assert!(!limiter.allow()); // Fourth should be denied

        assert_eq!(limiter.current_count(), 3);

        limiter.reset();
        assert_eq!(limiter.current_count(), 0);
        assert!(limiter.allow());
    }

    #[test]
    fn test_performance_snapshot() {
        let snapshot = PerformanceSnapshot::new(100_000_000, 10, 1000.0, 50.0);

        assert!(snapshot.is_healthy(100.0, 200.0)); // 100MB, 100ms
        assert!(!snapshot.is_healthy(10.0, 200.0)); // Too high latency
    }
}
