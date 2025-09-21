//! Advanced Performance Monitoring for VoiRS Evaluation
//!
//! This module provides comprehensive performance monitoring and optimization
//! suggestions for evaluation workflows, helping identify bottlenecks and
//! optimize system performance.

use crate::EvaluationError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitorConfig {
    /// Maximum number of measurements to keep in history
    pub max_history_size: usize,
    /// Sampling interval for continuous monitoring
    pub sampling_interval_ms: u64,
    /// Threshold for slow operation warnings (milliseconds)
    pub slow_operation_threshold_ms: u64,
    /// Enable memory usage monitoring
    pub monitor_memory: bool,
    /// Enable CPU usage monitoring
    pub monitor_cpu: bool,
    /// Enable detailed per-metric timing
    pub detailed_metric_timing: bool,
}

impl Default for PerformanceMonitorConfig {
    fn default() -> Self {
        Self {
            max_history_size: 1000,
            sampling_interval_ms: 100,
            slow_operation_threshold_ms: 1000,
            monitor_memory: true,
            monitor_cpu: true,
            detailed_metric_timing: true,
        }
    }
}

/// Performance measurement for a single operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMeasurement {
    /// Operation name/identifier
    pub operation: String,
    /// Start time
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Memory usage before operation (bytes)
    pub memory_before_bytes: Option<u64>,
    /// Memory usage after operation (bytes)
    pub memory_after_bytes: Option<u64>,
    /// CPU usage during operation (percentage)
    pub cpu_usage_percent: Option<f32>,
    /// Audio buffer size processed
    pub audio_buffer_size: Option<usize>,
    /// Sample rate of processed audio
    pub sample_rate: Option<u32>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Aggregated performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Operation name
    pub operation: String,
    /// Number of measurements
    pub measurement_count: usize,
    /// Total duration across all measurements
    pub total_duration_ms: u64,
    /// Average duration
    pub avg_duration_ms: f64,
    /// Minimum duration
    pub min_duration_ms: u64,
    /// Maximum duration
    pub max_duration_ms: u64,
    /// Standard deviation of duration
    pub std_dev_duration_ms: f64,
    /// 95th percentile duration
    pub p95_duration_ms: u64,
    /// 99th percentile duration
    pub p99_duration_ms: u64,
    /// Average memory usage
    pub avg_memory_usage_mb: Option<f64>,
    /// Average CPU usage
    pub avg_cpu_usage_percent: Option<f32>,
    /// Operations per second
    pub ops_per_second: f64,
}

/// Performance optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Recommendation category
    pub category: String,
    /// Severity level (1-10, 10 being most critical)
    pub severity: u8,
    /// Description of the issue
    pub description: String,
    /// Specific recommendation
    pub recommendation: String,
    /// Expected improvement
    pub expected_improvement: String,
    /// Implementation complexity (Low, Medium, High)
    pub complexity: String,
}

/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    /// Alert type
    pub alert_type: PerformanceAlertType,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Operation that triggered the alert
    pub operation: String,
    /// Alert message
    pub message: String,
    /// Current value that triggered the alert
    pub current_value: f64,
    /// Threshold that was exceeded
    pub threshold: f64,
}

/// Types of performance alerts
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PerformanceAlertType {
    /// Operation taking too long
    SlowOperation,
    /// High memory usage
    HighMemoryUsage,
    /// High CPU usage
    HighCpuUsage,
    /// Performance degradation
    PerformanceDegradation,
    /// Memory leak detected
    MemoryLeak,
}

/// Performance monitoring system
pub struct PerformanceMonitor {
    config: PerformanceMonitorConfig,
    measurements: Arc<Mutex<VecDeque<PerformanceMeasurement>>>,
    stats_cache: Arc<RwLock<HashMap<String, PerformanceStats>>>,
    alerts: Arc<Mutex<VecDeque<PerformanceAlert>>>,
    baseline_stats: Arc<RwLock<HashMap<String, PerformanceStats>>>,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(config: PerformanceMonitorConfig) -> Self {
        Self {
            config,
            measurements: Arc::new(Mutex::new(VecDeque::new())),
            stats_cache: Arc::new(RwLock::new(HashMap::new())),
            alerts: Arc::new(Mutex::new(VecDeque::new())),
            baseline_stats: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start monitoring an operation
    pub fn start_operation(&self, operation: &str) -> OperationTimer {
        OperationTimer::new(operation.to_string(), self.config.clone())
    }

    /// Record a completed measurement
    pub async fn record_measurement(&self, measurement: PerformanceMeasurement) {
        // Add to measurements history
        {
            let mut measurements = self.measurements.lock().unwrap();
            measurements.push_back(measurement.clone());

            // Maintain history size limit
            while measurements.len() > self.config.max_history_size {
                measurements.pop_front();
            }
        }

        // Update stats cache
        self.update_stats_cache(&measurement).await;

        // Check for alerts
        self.check_performance_alerts(&measurement).await;
    }

    /// Get performance statistics for an operation
    pub async fn get_stats(&self, operation: &str) -> Option<PerformanceStats> {
        let stats_cache = self.stats_cache.read().await;
        stats_cache.get(operation).cloned()
    }

    /// Get all performance statistics
    pub async fn get_all_stats(&self) -> HashMap<String, PerformanceStats> {
        let stats_cache = self.stats_cache.read().await;
        stats_cache.clone()
    }

    /// Get recent alerts
    pub fn get_recent_alerts(&self, limit: usize) -> Vec<PerformanceAlert> {
        let alerts = self.alerts.lock().unwrap();
        alerts.iter().rev().take(limit).cloned().collect()
    }

    /// Generate optimization recommendations
    pub async fn generate_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();
        let stats = self.get_all_stats().await;

        for (operation, stat) in &stats {
            // Check for slow operations
            if stat.avg_duration_ms > self.config.slow_operation_threshold_ms as f64 {
                recommendations.push(OptimizationRecommendation {
                    category: String::from("Performance"),
                    severity: 7,
                    description: format!(
                        "Operation '{}' has high average duration ({:.2}ms)",
                        operation, stat.avg_duration_ms
                    ),
                    recommendation: String::from(
                        "Consider optimizing the algorithm or using parallel processing",
                    ),
                    expected_improvement: String::from("30-50% reduction in processing time"),
                    complexity: String::from("Medium"),
                });
            }

            // Check for high variation in timing
            if stat.std_dev_duration_ms > stat.avg_duration_ms * 0.5 {
                recommendations.push(OptimizationRecommendation {
                    category: String::from("Consistency"),
                    severity: 5,
                    description: format!(
                        "Operation '{}' has inconsistent timing (std dev: {:.2}ms)",
                        operation, stat.std_dev_duration_ms
                    ),
                    recommendation: "Investigate variable load factors and consider caching"
                        .to_string(),
                    expected_improvement: String::from("More predictable performance"),
                    complexity: String::from("Low"),
                });
            }

            // Check for low throughput
            if stat.ops_per_second < 1.0 {
                recommendations.push(OptimizationRecommendation {
                    category: String::from("Throughput"),
                    severity: 6,
                    description: format!(
                        "Operation '{}' has low throughput ({:.2} ops/sec)",
                        operation, stat.ops_per_second
                    ),
                    recommendation: "Consider batch processing or algorithm optimization"
                        .to_string(),
                    expected_improvement: String::from("2-10x increase in throughput"),
                    complexity: String::from("High"),
                });
            }

            // Check for high memory usage
            if let Some(avg_memory) = stat.avg_memory_usage_mb {
                if avg_memory > 100.0 {
                    recommendations.push(OptimizationRecommendation {
                        category: String::from("Memory"),
                        severity: 4,
                        description: format!(
                            "Operation '{}' uses high memory ({:.2} MB)",
                            operation, avg_memory
                        ),
                        recommendation: "Consider streaming processing or memory pooling"
                            .to_string(),
                        expected_improvement: String::from("50-70% reduction in memory usage"),
                        complexity: String::from("Medium"),
                    });
                }
            }
        }

        recommendations
    }

    /// Create a performance report
    pub async fn create_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# VoiRS Evaluation Performance Report\n\n");

        let stats = self.get_all_stats().await;
        let recommendations = self.generate_recommendations().await;
        let recent_alerts = self.get_recent_alerts(10);

        // Performance Overview
        report.push_str("## Performance Overview\n\n");
        if stats.is_empty() {
            report.push_str("No performance data available.\n\n");
        } else {
            report
                .push_str("| Operation | Avg Duration | Ops/Sec | P95 Duration | Memory Usage |\n");
            report.push_str(
                "|-----------|--------------|---------|--------------|---------------|\n",
            );

            for (operation, stat) in &stats {
                let memory_str = stat
                    .avg_memory_usage_mb
                    .map(|m| format!("{:.1} MB", m))
                    .unwrap_or_else(|| String::from("N/A"));

                report.push_str(&format!(
                    "| {} | {:.2}ms | {:.2} | {}ms | {} |\n",
                    operation,
                    stat.avg_duration_ms,
                    stat.ops_per_second,
                    stat.p95_duration_ms,
                    memory_str
                ));
            }
            report.push_str("\n");
        }

        // Recent Alerts
        if !recent_alerts.is_empty() {
            report.push_str("## Recent Alerts\n\n");
            for alert in &recent_alerts {
                let icon = match alert.alert_type {
                    PerformanceAlertType::SlowOperation => "🐌",
                    PerformanceAlertType::HighMemoryUsage => "🧠",
                    PerformanceAlertType::HighCpuUsage => "⚡",
                    PerformanceAlertType::PerformanceDegradation => "📉",
                    PerformanceAlertType::MemoryLeak => "🔴",
                };

                report.push_str(&format!(
                    "- {} **{}**: {} ({})\n",
                    icon,
                    alert.operation,
                    alert.message,
                    alert.timestamp.format("%Y-%m-%d %H:%M:%S")
                ));
            }
            report.push_str("\n");
        }

        // Optimization Recommendations
        if !recommendations.is_empty() {
            report.push_str("## Optimization Recommendations\n\n");

            // Sort by severity (highest first)
            let mut sorted_recommendations = recommendations;
            sorted_recommendations.sort_by(|a, b| b.severity.cmp(&a.severity));

            for (i, rec) in sorted_recommendations.iter().enumerate() {
                let priority = match rec.severity {
                    8..=10 => "🔴 High",
                    5..=7 => "🟡 Medium",
                    1..=4 => "🟢 Low",
                    _ => "⚪ Unknown",
                };

                report.push_str(&format!(
                    "### {}. {} - {}\n\n",
                    i + 1,
                    rec.category,
                    priority
                ));
                report.push_str(&format!("**Issue:** {}\n\n", rec.description));
                report.push_str(&format!("**Recommendation:** {}\n\n", rec.recommendation));
                report.push_str(&format!(
                    "**Expected Improvement:** {}\n\n",
                    rec.expected_improvement
                ));
                report.push_str(&format!(
                    "**Implementation Complexity:** {}\n\n",
                    rec.complexity
                ));
            }
        }

        report
    }

    /// Set baseline performance statistics
    pub async fn set_baseline(&self, operation: &str, stats: PerformanceStats) {
        let mut baseline_stats = self.baseline_stats.write().await;
        baseline_stats.insert(operation.to_string(), stats);
    }

    /// Clear all performance data
    pub async fn clear_data(&self) {
        {
            let mut measurements = self.measurements.lock().unwrap();
            measurements.clear();
        }

        {
            let mut stats_cache = self.stats_cache.write().await;
            stats_cache.clear();
        }

        {
            let mut alerts = self.alerts.lock().unwrap();
            alerts.clear();
        }
    }

    /// Update statistics cache
    async fn update_stats_cache(&self, measurement: &PerformanceMeasurement) {
        let mut stats_cache = self.stats_cache.write().await;

        // Get recent measurements for this operation
        let measurements = self.measurements.lock().unwrap();
        let operation_measurements: Vec<_> = measurements
            .iter()
            .filter(|m| m.operation == measurement.operation)
            .collect();

        if operation_measurements.is_empty() {
            return;
        }

        // Calculate statistics
        let durations: Vec<u64> = operation_measurements
            .iter()
            .map(|m| m.duration_ms)
            .collect();
        let total_duration: u64 = durations.iter().sum();
        let count = durations.len();
        let avg_duration = total_duration as f64 / count as f64;

        let min_duration = *durations.iter().min().unwrap();
        let max_duration = *durations.iter().max().unwrap();

        // Calculate standard deviation
        let variance = durations
            .iter()
            .map(|&d| (d as f64 - avg_duration).powi(2))
            .sum::<f64>()
            / count as f64;
        let std_dev = variance.sqrt();

        // Calculate percentiles
        let mut sorted_durations = durations.clone();
        sorted_durations.sort_unstable();
        let p95_idx = ((count as f64 * 0.95) as usize).min(count - 1);
        let p99_idx = ((count as f64 * 0.99) as usize).min(count - 1);
        let p95_duration = sorted_durations[p95_idx];
        let p99_duration = sorted_durations[p99_idx];

        // Calculate memory usage
        let avg_memory_usage_mb = if operation_measurements
            .iter()
            .any(|m| m.memory_after_bytes.is_some())
        {
            let memory_values: Vec<u64> = operation_measurements
                .iter()
                .filter_map(|m| m.memory_after_bytes)
                .collect();

            if !memory_values.is_empty() {
                Some(
                    memory_values.iter().sum::<u64>() as f64
                        / memory_values.len() as f64
                        / 1_048_576.0,
                )
            } else {
                None
            }
        } else {
            None
        };

        // Calculate CPU usage
        let avg_cpu_usage = if operation_measurements
            .iter()
            .any(|m| m.cpu_usage_percent.is_some())
        {
            let cpu_values: Vec<f32> = operation_measurements
                .iter()
                .filter_map(|m| m.cpu_usage_percent)
                .collect();

            if !cpu_values.is_empty() {
                Some(cpu_values.iter().sum::<f32>() / cpu_values.len() as f32)
            } else {
                None
            }
        } else {
            None
        };

        // Calculate operations per second
        let ops_per_second = if avg_duration > 0.0 {
            1000.0 / avg_duration
        } else {
            0.0
        };

        let stats = PerformanceStats {
            operation: measurement.operation.clone(),
            measurement_count: count,
            total_duration_ms: total_duration,
            avg_duration_ms: avg_duration,
            min_duration_ms: min_duration,
            max_duration_ms: max_duration,
            std_dev_duration_ms: std_dev,
            p95_duration_ms: p95_duration,
            p99_duration_ms: p99_duration,
            avg_memory_usage_mb,
            avg_cpu_usage_percent: avg_cpu_usage,
            ops_per_second,
        };

        stats_cache.insert(measurement.operation.clone(), stats);
    }

    /// Check for performance alerts
    async fn check_performance_alerts(&self, measurement: &PerformanceMeasurement) {
        let mut alerts_to_add = Vec::new();

        // Check for slow operations
        if measurement.duration_ms > self.config.slow_operation_threshold_ms {
            alerts_to_add.push(PerformanceAlert {
                alert_type: PerformanceAlertType::SlowOperation,
                timestamp: chrono::Utc::now(),
                operation: measurement.operation.clone(),
                message: format!(
                    "Operation took {}ms (threshold: {}ms)",
                    measurement.duration_ms, self.config.slow_operation_threshold_ms
                ),
                current_value: measurement.duration_ms as f64,
                threshold: self.config.slow_operation_threshold_ms as f64,
            });
        }

        // Check for high memory usage
        if let (Some(before), Some(after)) = (
            measurement.memory_before_bytes,
            measurement.memory_after_bytes,
        ) {
            let memory_increase_mb = (after as i64 - before as i64) as f64 / 1_048_576.0;
            if memory_increase_mb > 50.0 {
                // 50MB threshold
                alerts_to_add.push(PerformanceAlert {
                    alert_type: PerformanceAlertType::HighMemoryUsage,
                    timestamp: chrono::Utc::now(),
                    operation: measurement.operation.clone(),
                    message: format!(
                        "Operation increased memory usage by {:.1}MB",
                        memory_increase_mb
                    ),
                    current_value: memory_increase_mb,
                    threshold: 50.0,
                });
            }
        }

        // Check for high CPU usage
        if let Some(cpu_usage) = measurement.cpu_usage_percent {
            if cpu_usage > 80.0 {
                alerts_to_add.push(PerformanceAlert {
                    alert_type: PerformanceAlertType::HighCpuUsage,
                    timestamp: chrono::Utc::now(),
                    operation: measurement.operation.clone(),
                    message: format!("Operation used {:.1}% CPU", cpu_usage),
                    current_value: cpu_usage as f64,
                    threshold: 80.0,
                });
            }
        }

        // Add alerts
        if !alerts_to_add.is_empty() {
            let mut alerts = self.alerts.lock().unwrap();
            for alert in alerts_to_add {
                alerts.push_back(alert);
            }

            // Maintain alert history size
            while alerts.len() > 100 {
                alerts.pop_front();
            }
        }
    }
}

/// Timer for measuring operation performance
pub struct OperationTimer {
    operation: String,
    start_time: Instant,
    start_timestamp: chrono::DateTime<chrono::Utc>,
    memory_before: Option<u64>,
    config: PerformanceMonitorConfig,
    metadata: HashMap<String, String>,
}

impl OperationTimer {
    fn new(operation: String, config: PerformanceMonitorConfig) -> Self {
        let memory_before = if config.monitor_memory {
            Self::get_memory_usage()
        } else {
            None
        };

        Self {
            operation,
            start_time: Instant::now(),
            start_timestamp: chrono::Utc::now(),
            memory_before,
            config,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the measurement
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Finish timing and create measurement
    pub async fn finish(self, monitor: &PerformanceMonitor) -> Result<(), EvaluationError> {
        let duration = self.start_time.elapsed();
        let duration_ms = duration.as_millis() as u64;

        let memory_after = if self.config.monitor_memory {
            Self::get_memory_usage()
        } else {
            None
        };

        let cpu_usage = if self.config.monitor_cpu {
            Self::get_cpu_usage()
        } else {
            None
        };

        let measurement = PerformanceMeasurement {
            operation: self.operation,
            start_time: self.start_timestamp,
            duration_ms,
            memory_before_bytes: self.memory_before,
            memory_after_bytes: memory_after,
            cpu_usage_percent: cpu_usage,
            audio_buffer_size: None, // Can be set via metadata
            sample_rate: None,       // Can be set via metadata
            metadata: self.metadata,
        };

        monitor.record_measurement(measurement).await;
        Ok(())
    }

    /// Get current memory usage (simplified implementation)
    fn get_memory_usage() -> Option<u64> {
        // This is a simplified implementation
        // In practice, you would use a proper memory monitoring library
        None
    }

    /// Get current CPU usage (simplified implementation)
    fn get_cpu_usage() -> Option<f32> {
        // This is a simplified implementation
        // In practice, you would use a proper CPU monitoring library
        None
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new(PerformanceMonitorConfig::default())
    }
}

/// Advanced profiling system for detailed performance analysis
#[derive(Clone)]
pub struct AdvancedProfiler {
    monitor: Arc<PerformanceMonitor>,
    call_stack: Arc<Mutex<Vec<CallFrame>>>,
    hotspots: Arc<Mutex<HashMap<String, HotspotData>>>,
    sampling_enabled: Arc<std::sync::atomic::AtomicBool>,
}

/// Call frame for stack-based profiling
#[derive(Debug, Clone)]
pub struct CallFrame {
    /// Function or operation name
    pub name: String,
    /// Entry timestamp
    pub entry_time: Instant,
    /// Memory usage at entry
    pub memory_at_entry: Option<usize>,
    /// Parent frame index
    pub parent_index: Option<usize>,
}

/// Hotspot data for performance analysis
#[derive(Debug, Clone)]
pub struct HotspotData {
    /// Total time spent in this function/operation
    pub total_time: Duration,
    /// Number of times called
    pub call_count: usize,
    /// Average time per call
    pub avg_time: Duration,
    /// Maximum time spent in single call
    pub max_time: Duration,
    /// Total memory allocated
    pub total_memory: usize,
    /// Percentage of total runtime
    pub runtime_percentage: f64,
}

/// Performance regression detector
pub struct RegressionDetector {
    baseline_stats: HashMap<String, PerformanceStats>,
    sensitivity_threshold: f64,
    window_size: usize,
}

impl AdvancedProfiler {
    /// Create a new advanced profiler
    pub fn new(monitor: Arc<PerformanceMonitor>) -> Self {
        Self {
            monitor,
            call_stack: Arc::new(Mutex::new(Vec::new())),
            hotspots: Arc::new(Mutex::new(HashMap::new())),
            sampling_enabled: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        }
    }

    /// Enter a function/operation for profiling
    pub fn enter_function(&self, name: &str) -> ProfilerScope {
        if !self
            .sampling_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return ProfilerScope::disabled();
        }

        let frame = CallFrame {
            name: name.to_string(),
            entry_time: Instant::now(),
            memory_at_entry: Self::get_current_memory(),
            parent_index: {
                let stack = self.call_stack.lock().unwrap();
                if stack.is_empty() {
                    None
                } else {
                    Some(stack.len() - 1)
                }
            },
        };

        let index = {
            let mut stack = self.call_stack.lock().unwrap();
            stack.push(frame);
            stack.len() - 1
        };

        ProfilerScope::new(self.clone(), index)
    }

    /// Exit a function/operation
    pub fn exit_function(&self, frame_index: usize) {
        let frame = {
            let mut stack = self.call_stack.lock().unwrap();
            if frame_index >= stack.len() {
                return;
            }
            stack.remove(frame_index)
        };

        let duration = frame.entry_time.elapsed();
        let memory_delta = Self::get_current_memory()
            .and_then(|current| {
                frame
                    .memory_at_entry
                    .map(|entry| current.saturating_sub(entry))
            })
            .unwrap_or(0);

        // Update hotspot data
        {
            let mut hotspots = self.hotspots.lock().unwrap();
            let hotspot = hotspots.entry(frame.name).or_insert_with(|| HotspotData {
                total_time: Duration::ZERO,
                call_count: 0,
                avg_time: Duration::ZERO,
                max_time: Duration::ZERO,
                total_memory: 0,
                runtime_percentage: 0.0,
            });

            hotspot.total_time += duration;
            hotspot.call_count += 1;
            hotspot.avg_time = hotspot.total_time / hotspot.call_count as u32;
            hotspot.max_time = hotspot.max_time.max(duration);
            hotspot.total_memory += memory_delta;
        }
    }

    /// Get hotspot analysis
    pub fn get_hotspots(&self) -> Vec<(String, HotspotData)> {
        let hotspots = self.hotspots.lock().unwrap();
        let total_time: Duration = hotspots.values().map(|h| h.total_time).sum();

        let mut results: Vec<_> = hotspots
            .iter()
            .map(|(name, data)| {
                let mut data = data.clone();
                data.runtime_percentage = if total_time.as_nanos() > 0 {
                    (data.total_time.as_nanos() as f64 / total_time.as_nanos() as f64) * 100.0
                } else {
                    0.0
                };
                (name.clone(), data)
            })
            .collect();

        // Sort by total time descending
        results.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));
        results
    }

    /// Generate flame graph data
    pub fn generate_flame_graph(&self) -> String {
        let hotspots = self.get_hotspots();
        let mut flame_graph = String::new();

        flame_graph.push_str("Function,Time(ms),Calls,Avg(ms),Memory(KB)\n");

        for (name, data) in hotspots {
            flame_graph.push_str(&format!(
                "{},{:.2},{},{:.2},{}\n",
                name,
                data.total_time.as_secs_f64() * 1000.0,
                data.call_count,
                data.avg_time.as_secs_f64() * 1000.0,
                data.total_memory / 1024
            ));
        }

        flame_graph
    }

    /// Reset profiling data
    pub fn reset(&self) {
        self.call_stack.lock().unwrap().clear();
        self.hotspots.lock().unwrap().clear();
    }

    /// Enable or disable sampling
    pub fn set_sampling(&self, enabled: bool) {
        self.sampling_enabled
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    fn get_current_memory() -> Option<usize> {
        // Platform-specific memory tracking would go here
        // For now, return None as a placeholder
        None
    }
}

/// RAII scope for automatic function profiling
pub struct ProfilerScope {
    profiler: Option<AdvancedProfiler>,
    frame_index: usize,
}

impl ProfilerScope {
    fn new(profiler: AdvancedProfiler, frame_index: usize) -> Self {
        Self {
            profiler: Some(profiler),
            frame_index,
        }
    }

    fn disabled() -> Self {
        Self {
            profiler: None,
            frame_index: 0,
        }
    }
}

impl Drop for ProfilerScope {
    fn drop(&mut self) {
        if let Some(profiler) = &self.profiler {
            profiler.exit_function(self.frame_index);
        }
    }
}

impl RegressionDetector {
    /// Create a new regression detector
    pub fn new(sensitivity_threshold: f64, window_size: usize) -> Self {
        Self {
            baseline_stats: HashMap::new(),
            sensitivity_threshold,
            window_size,
        }
    }

    /// Set baseline performance stats
    pub fn set_baseline(&mut self, stats: HashMap<String, PerformanceStats>) {
        self.baseline_stats = stats;
    }

    /// Detect performance regressions
    pub fn detect_regressions(
        &self,
        current_stats: &HashMap<String, PerformanceStats>,
    ) -> Vec<String> {
        let mut regressions = Vec::new();

        for (operation, current) in current_stats {
            if let Some(baseline) = self.baseline_stats.get(operation) {
                // Check duration regression
                let duration_change =
                    (current.avg_duration_ms - baseline.avg_duration_ms) / baseline.avg_duration_ms;
                if duration_change > self.sensitivity_threshold {
                    regressions.push(format!(
                        "Duration regression in '{}': {:.1}% slower ({:.2}ms → {:.2}ms)",
                        operation,
                        duration_change * 100.0,
                        baseline.avg_duration_ms,
                        current.avg_duration_ms
                    ));
                }

                // Check memory regression
                if let (Some(baseline_mem), Some(current_mem)) =
                    (baseline.avg_memory_usage_mb, current.avg_memory_usage_mb)
                {
                    let memory_change = (current_mem - baseline_mem) / baseline_mem;
                    if memory_change > self.sensitivity_threshold {
                        regressions.push(format!(
                            "Memory regression in '{}': {:.1}% increase ({:.2}MB → {:.2}MB)",
                            operation,
                            memory_change * 100.0,
                            baseline_mem,
                            current_mem
                        ));
                    }
                }

                // Check throughput regression
                let throughput_change =
                    (baseline.ops_per_second - current.ops_per_second) / baseline.ops_per_second;
                if throughput_change > self.sensitivity_threshold {
                    regressions.push(format!(
                        "Throughput regression in '{}': {:.1}% decrease ({:.2} → {:.2} ops/sec)",
                        operation,
                        throughput_change * 100.0,
                        baseline.ops_per_second,
                        current.ops_per_second
                    ));
                }
            }
        }

        regressions
    }
}

/// Macro for easy function profiling
#[macro_export]
macro_rules! profile_function {
    ($profiler:expr, $name:expr, $body:block) => {{
        let _scope = $profiler.enter_function($name);
        $body
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_performance_monitor_creation() {
        let config = PerformanceMonitorConfig::default();
        let monitor = PerformanceMonitor::new(config);

        let stats = monitor.get_all_stats().await;
        assert!(stats.is_empty());
    }

    #[tokio::test]
    async fn test_operation_timing() {
        let monitor = PerformanceMonitor::default();

        {
            let timer = monitor.start_operation("test_operation");
            sleep(Duration::from_millis(10)).await;
            timer.finish(&monitor).await.unwrap();
        }

        // Allow time for async processing
        sleep(Duration::from_millis(50)).await;

        let stats = monitor.get_stats("test_operation").await;
        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert_eq!(stats.operation, "test_operation");
        assert_eq!(stats.measurement_count, 1);
        assert!(stats.avg_duration_ms >= 10.0);
    }

    #[tokio::test]
    async fn test_multiple_measurements() {
        let monitor = PerformanceMonitor::default();

        // Record multiple measurements
        for i in 0..5 {
            let mut timer = monitor.start_operation("multi_test");
            timer.add_metadata("iteration", &i.to_string());
            sleep(Duration::from_millis(5 + i * 2)).await;
            timer.finish(&monitor).await.unwrap();
        }

        sleep(Duration::from_millis(50)).await;

        let stats = monitor.get_stats("multi_test").await;
        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert_eq!(stats.measurement_count, 5);
        assert!(stats.avg_duration_ms > 0.0);
        assert!(stats.std_dev_duration_ms >= 0.0);
    }

    #[tokio::test]
    async fn test_performance_alerts() {
        let config = PerformanceMonitorConfig {
            slow_operation_threshold_ms: 10,
            ..Default::default()
        };
        let monitor = PerformanceMonitor::new(config);

        // Create a slow operation
        {
            let timer = monitor.start_operation("slow_operation");
            sleep(Duration::from_millis(20)).await;
            timer.finish(&monitor).await.unwrap();
        }

        sleep(Duration::from_millis(50)).await;

        let alerts = monitor.get_recent_alerts(10);
        assert!(!alerts.is_empty());
        assert_eq!(alerts[0].alert_type, PerformanceAlertType::SlowOperation);
    }

    #[tokio::test]
    async fn test_optimization_recommendations() {
        let config = PerformanceMonitorConfig {
            slow_operation_threshold_ms: 5,
            ..Default::default()
        };
        let monitor = PerformanceMonitor::new(config);

        // Create operations that will trigger recommendations
        {
            let timer = monitor.start_operation("slow_op");
            sleep(Duration::from_millis(10)).await;
            timer.finish(&monitor).await.unwrap();
        }

        sleep(Duration::from_millis(50)).await;

        let recommendations = monitor.generate_recommendations().await;
        assert!(!recommendations.is_empty());
        assert!(recommendations.iter().any(|r| r.category == "Performance"));
    }

    #[tokio::test]
    async fn test_performance_report() {
        let monitor = PerformanceMonitor::default();

        {
            let timer = monitor.start_operation("report_test");
            sleep(Duration::from_millis(5)).await;
            timer.finish(&monitor).await.unwrap();
        }

        sleep(Duration::from_millis(50)).await;

        let report = monitor.create_report().await;
        assert!(report.contains("Performance Report"));
        assert!(report.contains("report_test"));
    }

    #[test]
    fn test_performance_measurement() {
        let measurement = PerformanceMeasurement {
            operation: String::from("test"),
            start_time: chrono::Utc::now(),
            duration_ms: 100,
            memory_before_bytes: Some(1000),
            memory_after_bytes: Some(1200),
            cpu_usage_percent: Some(50.0),
            audio_buffer_size: Some(16000),
            sample_rate: Some(44100),
            metadata: HashMap::new(),
        };

        assert_eq!(measurement.operation, "test");
        assert_eq!(measurement.duration_ms, 100);
    }

    #[test]
    fn test_performance_stats() {
        let stats = PerformanceStats {
            operation: String::from("test_op"),
            measurement_count: 10,
            total_duration_ms: 1000,
            avg_duration_ms: 100.0,
            min_duration_ms: 50,
            max_duration_ms: 200,
            std_dev_duration_ms: 25.0,
            p95_duration_ms: 180,
            p99_duration_ms: 195,
            avg_memory_usage_mb: Some(50.0),
            avg_cpu_usage_percent: Some(25.0),
            ops_per_second: 10.0,
        };

        assert_eq!(stats.operation, "test_op");
        assert_eq!(stats.measurement_count, 10);
        assert_eq!(stats.ops_per_second, 10.0);
    }
}
