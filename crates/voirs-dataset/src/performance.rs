//! Performance monitoring and profiling utilities
//!
//! Provides comprehensive performance tracking for dataset operations,
//! including timing, memory usage, and bottleneck identification.

use crate::{DatasetError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Performance metrics for a specific operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Operation name
    pub operation: String,
    /// Total execution time
    pub duration: Duration,
    /// Memory usage before operation (bytes)
    pub memory_before: u64,
    /// Memory usage after operation (bytes)
    pub memory_after: u64,
    /// Peak memory usage during operation (bytes)
    pub peak_memory: u64,
    /// Number of samples processed
    pub samples_processed: usize,
    /// Throughput (samples per second)
    pub throughput: f64,
    /// CPU utilization (0.0 - 1.0)
    pub cpu_utilization: f64,
    /// Timestamp of measurement
    pub timestamp: std::time::SystemTime,
}

impl PerformanceMetrics {
    /// Calculate memory delta
    pub fn memory_delta(&self) -> i64 {
        self.memory_after as i64 - self.memory_before as i64
    }

    /// Calculate throughput in samples per second
    pub fn calculate_throughput(&self) -> f64 {
        if self.duration.as_secs_f64() > 0.0 {
            self.samples_processed as f64 / self.duration.as_secs_f64()
        } else {
            0.0
        }
    }
}

/// Performance profiler for tracking operation metrics
#[derive(Debug)]
pub struct PerformanceProfiler {
    /// Collected metrics
    metrics: Arc<Mutex<Vec<PerformanceMetrics>>>,
    /// Current active operations
    active_operations: Arc<Mutex<HashMap<String, ProfilerSession>>>,
    /// Configuration
    config: ProfilerConfig,
}

/// Configuration for performance profiler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Enable memory tracking
    pub track_memory: bool,
    /// Enable CPU utilization tracking
    pub track_cpu: bool,
    /// Maximum number of metrics to store
    pub max_metrics: usize,
    /// Minimum duration to record (filter out very fast operations)
    pub min_duration_ms: u64,
    /// Enable detailed logging
    pub detailed_logging: bool,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            track_memory: true,
            track_cpu: true,
            max_metrics: 10000,
            min_duration_ms: 1,
            detailed_logging: false,
        }
    }
}

/// Active profiling session
#[derive(Debug)]
pub struct ProfilerSession {
    /// Operation name
    operation: String,
    /// Start time
    start_time: Instant,
    /// Start memory
    start_memory: u64,
    /// Peak memory observed
    peak_memory: u64,
    /// Samples being processed
    samples_count: usize,
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new() -> Self {
        Self::with_config(ProfilerConfig::default())
    }

    /// Create a new performance profiler with custom configuration
    pub fn with_config(config: ProfilerConfig) -> Self {
        Self {
            metrics: Arc::new(Mutex::new(Vec::new())),
            active_operations: Arc::new(Mutex::new(HashMap::new())),
            config,
        }
    }

    /// Start profiling an operation
    pub fn start_operation(&self, operation: &str, samples_count: usize) -> Result<()> {
        let mut active = self.active_operations.lock().map_err(|_| {
            DatasetError::ProcessingError("Failed to acquire profiler lock".to_string())
        })?;

        let start_memory = if self.config.track_memory {
            self.get_memory_usage()
        } else {
            0
        };

        let session = ProfilerSession {
            operation: operation.to_string(),
            start_time: Instant::now(),
            start_memory,
            peak_memory: start_memory,
            samples_count,
        };

        active.insert(operation.to_string(), session);

        if self.config.detailed_logging {
            debug!("Started profiling operation: {}", operation);
        }

        Ok(())
    }

    /// End profiling an operation and record metrics
    pub fn end_operation(&self, operation: &str) -> Result<PerformanceMetrics> {
        let mut active = self.active_operations.lock().map_err(|_| {
            DatasetError::ProcessingError("Failed to acquire profiler lock".to_string())
        })?;

        let session = active.remove(operation).ok_or_else(|| {
            DatasetError::ProcessingError(format!("No active session for operation: {operation}"))
        })?;

        let duration = session.start_time.elapsed();
        let memory_after = if self.config.track_memory {
            self.get_memory_usage()
        } else {
            0
        };

        let cpu_utilization = if self.config.track_cpu {
            self.get_cpu_utilization()
        } else {
            0.0
        };

        let metrics = PerformanceMetrics {
            operation: operation.to_string(),
            duration,
            memory_before: session.start_memory,
            memory_after,
            peak_memory: session.peak_memory,
            samples_processed: session.samples_count,
            throughput: if duration.as_secs_f64() > 0.0 {
                session.samples_count as f64 / duration.as_secs_f64()
            } else {
                0.0
            },
            cpu_utilization,
            timestamp: std::time::SystemTime::now(),
        };

        // Only record if duration meets minimum threshold
        if duration.as_millis() >= self.config.min_duration_ms as u128 {
            let mut stored_metrics = self.metrics.lock().map_err(|_| {
                DatasetError::ProcessingError("Failed to acquire metrics lock".to_string())
            })?;

            stored_metrics.push(metrics.clone());

            // Limit stored metrics to prevent unbounded growth
            if stored_metrics.len() > self.config.max_metrics {
                stored_metrics.remove(0);
            }

            if self.config.detailed_logging {
                info!(
                    "Operation '{}' completed in {:.2}ms, throughput: {:.1} samples/sec",
                    operation,
                    duration.as_secs_f64() * 1000.0,
                    metrics.throughput
                );
            }
        }

        Ok(metrics)
    }

    /// Update peak memory for an active operation
    pub fn update_peak_memory(&self, operation: &str) -> Result<()> {
        if !self.config.track_memory {
            return Ok(());
        }

        let mut active = self.active_operations.lock().map_err(|_| {
            DatasetError::ProcessingError("Failed to acquire profiler lock".to_string())
        })?;

        if let Some(session) = active.get_mut(operation) {
            let current_memory = self.get_memory_usage();
            if current_memory > session.peak_memory {
                session.peak_memory = current_memory;
            }
        }

        Ok(())
    }

    /// Get all collected metrics
    pub fn get_metrics(&self) -> Result<Vec<PerformanceMetrics>> {
        let metrics = self.metrics.lock().map_err(|_| {
            DatasetError::ProcessingError("Failed to acquire metrics lock".to_string())
        })?;
        Ok(metrics.clone())
    }

    /// Get performance summary
    pub fn get_summary(&self) -> Result<PerformanceSummary> {
        let metrics = self.get_metrics()?;

        if metrics.is_empty() {
            return Ok(PerformanceSummary::default());
        }

        let mut operation_stats: HashMap<String, OperationStats> = HashMap::new();
        let mut total_duration = Duration::default();
        let mut total_samples = 0;
        let mut total_memory_delta = 0i64;

        for metric in &metrics {
            total_duration += metric.duration;
            total_samples += metric.samples_processed;
            total_memory_delta += metric.memory_delta();

            let stats = operation_stats
                .entry(metric.operation.clone())
                .or_insert_with(|| OperationStats {
                    operation: metric.operation.clone(),
                    call_count: 0,
                    total_duration: Duration::default(),
                    avg_duration: Duration::default(),
                    min_duration: Duration::from_secs(u64::MAX),
                    max_duration: Duration::default(),
                    total_samples: 0,
                    avg_throughput: 0.0,
                    peak_memory: 0,
                });

            stats.call_count += 1;
            stats.total_duration += metric.duration;
            stats.total_samples += metric.samples_processed;
            stats.min_duration = stats.min_duration.min(metric.duration);
            stats.max_duration = stats.max_duration.max(metric.duration);
            stats.peak_memory = stats.peak_memory.max(metric.peak_memory);
        }

        // Calculate averages
        for stats in operation_stats.values_mut() {
            stats.avg_duration = stats.total_duration / stats.call_count as u32;
            if stats.total_duration.as_secs_f64() > 0.0 {
                stats.avg_throughput =
                    stats.total_samples as f64 / stats.total_duration.as_secs_f64();
            }
        }

        let avg_throughput = if total_duration.as_secs_f64() > 0.0 {
            total_samples as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };

        Ok(PerformanceSummary {
            total_operations: metrics.len(),
            total_duration,
            total_samples,
            avg_throughput,
            total_memory_delta,
            operation_stats: operation_stats.into_values().collect(),
        })
    }

    /// Clear all collected metrics
    pub fn clear_metrics(&self) -> Result<()> {
        let mut metrics = self.metrics.lock().map_err(|_| {
            DatasetError::ProcessingError("Failed to acquire metrics lock".to_string())
        })?;
        metrics.clear();
        Ok(())
    }

    /// Get current memory usage (simplified implementation)
    fn get_memory_usage(&self) -> u64 {
        // In a real implementation, this would use platform-specific APIs
        // For now, return a placeholder value
        0
    }

    /// Get current CPU utilization (simplified implementation)
    fn get_cpu_utilization(&self) -> f64 {
        // In a real implementation, this would measure actual CPU usage
        // For now, return a placeholder value
        0.0
    }
}

/// Statistics for a specific operation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationStats {
    pub operation: String,
    pub call_count: usize,
    pub total_duration: Duration,
    pub avg_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub total_samples: usize,
    pub avg_throughput: f64,
    pub peak_memory: u64,
}

/// Performance summary across all operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_operations: usize,
    pub total_duration: Duration,
    pub total_samples: usize,
    pub avg_throughput: f64,
    pub total_memory_delta: i64,
    pub operation_stats: Vec<OperationStats>,
}

impl Default for PerformanceSummary {
    fn default() -> Self {
        Self {
            total_operations: 0,
            total_duration: Duration::default(),
            total_samples: 0,
            avg_throughput: 0.0,
            total_memory_delta: 0,
            operation_stats: Vec::new(),
        }
    }
}

/// RAII guard for automatic operation profiling
pub struct ProfilerGuard<'a> {
    profiler: &'a PerformanceProfiler,
    operation: String,
}

impl<'a> ProfilerGuard<'a> {
    /// Create a new profiler guard
    pub fn new(
        profiler: &'a PerformanceProfiler,
        operation: &str,
        samples_count: usize,
    ) -> Result<Self> {
        profiler.start_operation(operation, samples_count)?;
        Ok(Self {
            profiler,
            operation: operation.to_string(),
        })
    }

    /// Update peak memory during the operation
    pub fn update_peak_memory(&self) -> Result<()> {
        self.profiler.update_peak_memory(&self.operation)
    }
}

impl<'a> Drop for ProfilerGuard<'a> {
    fn drop(&mut self) {
        if let Err(e) = self.profiler.end_operation(&self.operation) {
            warn!(
                "Failed to end profiling operation '{}': {}",
                self.operation, e
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_profiler_basic_operation() {
        let profiler = PerformanceProfiler::new();

        profiler.start_operation("test_op", 100).unwrap();
        thread::sleep(Duration::from_millis(10));
        let metrics = profiler.end_operation("test_op").unwrap();

        assert_eq!(metrics.operation, "test_op");
        assert_eq!(metrics.samples_processed, 100);
        assert!(metrics.duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_profiler_guard() {
        let profiler = PerformanceProfiler::new();

        {
            let _guard = ProfilerGuard::new(&profiler, "guard_test", 50).unwrap();
            thread::sleep(Duration::from_millis(5));
        }

        let metrics = profiler.get_metrics().unwrap();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].operation, "guard_test");
        assert_eq!(metrics[0].samples_processed, 50);
    }

    #[test]
    fn test_performance_summary() {
        let profiler = PerformanceProfiler::new();

        // Add some test metrics
        profiler.start_operation("op1", 100).unwrap();
        thread::sleep(Duration::from_millis(5));
        profiler.end_operation("op1").unwrap();

        profiler.start_operation("op2", 200).unwrap();
        thread::sleep(Duration::from_millis(10));
        profiler.end_operation("op2").unwrap();

        let summary = profiler.get_summary().unwrap();
        assert_eq!(summary.total_operations, 2);
        assert_eq!(summary.total_samples, 300);
        assert_eq!(summary.operation_stats.len(), 2);
    }

    #[test]
    fn test_metrics_limit() {
        let config = ProfilerConfig {
            max_metrics: 2,
            min_duration_ms: 0,
            ..Default::default()
        };
        let profiler = PerformanceProfiler::with_config(config);

        // Add 3 operations
        for i in 0..3 {
            profiler.start_operation(&format!("op{i}"), 10).unwrap();
            thread::sleep(Duration::from_millis(1));
            profiler.end_operation(&format!("op{i}")).unwrap();
        }

        let metrics = profiler.get_metrics().unwrap();
        assert_eq!(metrics.len(), 2); // Should be limited to max_metrics
    }
}
