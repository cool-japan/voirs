//! Advanced Performance Profiling and Tracing System
//!
//! This module provides comprehensive performance profiling capabilities including:
//! - Hierarchical span tracing with parent-child relationships
//! - Detailed timing statistics (min/max/avg/p50/p95/p99)
//! - Memory allocation tracking and profiling
//! - Operation-level performance analysis
//! - Export capabilities for external analysis tools
//! - Real-time performance visualization data generation

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::{AcousticError, Result};

/// Performance profiling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    /// Enable detailed span tracing
    pub enable_tracing: bool,
    /// Enable memory profiling
    pub enable_memory_tracking: bool,
    /// Maximum number of spans to retain in history
    pub max_span_history: usize,
    /// Sample rate for memory snapshots (1.0 = all operations)
    pub memory_sample_rate: f32,
    /// Minimum duration to track (spans shorter than this are ignored)
    pub min_duration_ms: f32,
    /// Enable automatic report generation
    pub auto_report: bool,
    /// Report generation interval (seconds)
    pub report_interval_secs: u64,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            enable_tracing: true,
            enable_memory_tracking: true,
            max_span_history: 10000,
            memory_sample_rate: 0.1, // 10% sampling to reduce overhead
            min_duration_ms: 0.1,    // Track operations >= 0.1ms
            auto_report: false,
            report_interval_secs: 60,
        }
    }
}

impl ProfilingConfig {
    /// Development configuration with detailed tracing
    pub fn development() -> Self {
        Self {
            enable_tracing: true,
            enable_memory_tracking: true,
            max_span_history: 50000,
            memory_sample_rate: 1.0, // Track all operations
            min_duration_ms: 0.01,
            auto_report: true,
            report_interval_secs: 30,
        }
    }

    /// Production configuration with optimized overhead
    pub fn production() -> Self {
        Self {
            enable_tracing: true,
            enable_memory_tracking: true,
            max_span_history: 5000,
            memory_sample_rate: 0.05, // 5% sampling
            min_duration_ms: 1.0,     // Only track operations >= 1ms
            auto_report: false,
            report_interval_secs: 300,
        }
    }

    /// Minimal configuration for benchmarking
    pub fn minimal() -> Self {
        Self {
            enable_tracing: false,
            enable_memory_tracking: false,
            max_span_history: 0,
            memory_sample_rate: 0.0,
            min_duration_ms: 1000.0,
            auto_report: false,
            report_interval_secs: 0,
        }
    }
}

/// Span identifier for hierarchical tracing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpanId(u64);

impl SpanId {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        SpanId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// Performance span representing a measured operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSpan {
    /// Unique span identifier
    pub id: SpanId,
    /// Parent span identifier (if nested)
    pub parent_id: Option<SpanId>,
    /// Operation name
    pub name: String,
    /// Start timestamp
    pub start_time: SystemTime,
    /// Duration of the operation
    pub duration: Duration,
    /// Memory allocated during this span (bytes)
    pub memory_allocated: u64,
    /// Memory freed during this span (bytes)
    pub memory_freed: u64,
    /// Custom metadata tags
    pub tags: HashMap<String, String>,
    /// Child span count
    pub child_count: usize,
}

impl PerformanceSpan {
    /// Calculate net memory change
    pub fn net_memory_change(&self) -> i64 {
        self.memory_allocated as i64 - self.memory_freed as i64
    }

    /// Duration in milliseconds
    pub fn duration_ms(&self) -> f64 {
        self.duration.as_secs_f64() * 1000.0
    }

    /// Duration in microseconds
    pub fn duration_us(&self) -> f64 {
        self.duration.as_secs_f64() * 1_000_000.0
    }
}

/// Timing statistics for an operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStatistics {
    /// Total number of samples
    pub count: usize,
    /// Minimum duration
    pub min_duration: Duration,
    /// Maximum duration
    pub max_duration: Duration,
    /// Average duration
    pub avg_duration: Duration,
    /// Median duration (p50)
    pub median_duration: Duration,
    /// 95th percentile duration
    pub p95_duration: Duration,
    /// 99th percentile duration
    pub p99_duration: Duration,
    /// Total cumulative duration
    pub total_duration: Duration,
}

/// Memory statistics for an operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatistics {
    /// Total allocations
    pub total_allocations: u64,
    /// Total frees
    pub total_frees: u64,
    /// Peak memory usage
    pub peak_memory_bytes: u64,
    /// Average memory usage
    pub avg_memory_bytes: u64,
    /// Current memory usage
    pub current_memory_bytes: i64,
}

/// Operation profile containing aggregated statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationProfile {
    /// Operation name
    pub operation: String,
    /// Timing statistics
    pub timing: TimingStatistics,
    /// Memory statistics
    pub memory: MemoryStatistics,
    /// Total invocations
    pub invocation_count: usize,
    /// Failure count
    pub failure_count: usize,
}

/// Active span guard that automatically completes the span on drop
pub struct SpanGuard {
    id: SpanId,
    name: String,
    start: Instant,
    start_memory: u64,
    profiler: Arc<Mutex<PerformanceProfilerInner>>,
    tags: HashMap<String, String>,
}

impl SpanGuard {
    /// Add a tag to this span
    pub fn add_tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.tags.insert(key.into(), value.into());
    }

    /// Add multiple tags to this span
    pub fn add_tags<I, K, V>(&mut self, tags: I)
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        for (k, v) in tags {
            self.tags.insert(k.into(), v.into());
        }
    }

    /// Get the span ID
    pub fn id(&self) -> SpanId {
        self.id
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        let end_memory = get_current_memory();

        let memory_allocated = end_memory.saturating_sub(self.start_memory);
        let memory_freed = self.start_memory.saturating_sub(end_memory);

        if let Ok(mut profiler) = self.profiler.lock() {
            profiler.complete_span(
                self.id,
                duration,
                memory_allocated,
                memory_freed,
                std::mem::take(&mut self.tags),
            );
        }
    }
}

/// Inner state of the performance profiler
struct PerformanceProfilerInner {
    config: ProfilingConfig,
    spans: Vec<PerformanceSpan>,
    active_spans: HashMap<SpanId, (String, Option<SpanId>, Instant, u64)>,
    operation_stats: HashMap<String, Vec<Duration>>,
    operation_memory: HashMap<String, Vec<u64>>,
    start_time: Instant,
}

impl PerformanceProfilerInner {
    fn new(config: ProfilingConfig) -> Self {
        Self {
            config,
            spans: Vec::new(),
            active_spans: HashMap::new(),
            operation_stats: HashMap::new(),
            operation_memory: HashMap::new(),
            start_time: Instant::now(),
        }
    }

    fn begin_span(&mut self, name: String, parent_id: Option<SpanId>) -> (SpanId, u64) {
        let id = SpanId::new();
        let start_memory = get_current_memory();
        self.active_spans
            .insert(id, (name, parent_id, Instant::now(), start_memory));
        (id, start_memory)
    }

    fn complete_span(
        &mut self,
        id: SpanId,
        duration: Duration,
        memory_allocated: u64,
        memory_freed: u64,
        tags: HashMap<String, String>,
    ) {
        if !self.config.enable_tracing {
            return;
        }

        if duration.as_secs_f32() * 1000.0 < self.config.min_duration_ms {
            self.active_spans.remove(&id);
            return;
        }

        if let Some((name, parent_id, _start, _mem)) = self.active_spans.remove(&id) {
            // Count children
            let child_count = self
                .active_spans
                .values()
                .filter(|(_, parent, _, _)| *parent == Some(id))
                .count();

            let span = PerformanceSpan {
                id,
                parent_id,
                name: name.clone(),
                start_time: SystemTime::now(),
                duration,
                memory_allocated,
                memory_freed,
                tags,
                child_count,
            };

            // Record statistics
            self.operation_stats
                .entry(name.clone())
                .or_default()
                .push(duration);

            if self.config.enable_memory_tracking
                && fastrand::f32() < self.config.memory_sample_rate
            {
                self.operation_memory
                    .entry(name)
                    .or_default()
                    .push(memory_allocated);
            }

            // Store span
            self.spans.push(span);

            // Trim if exceeds max history
            if self.spans.len() > self.config.max_span_history {
                let excess = self.spans.len() - self.config.max_span_history;
                self.spans.drain(0..excess);
            }
        }
    }

    fn get_operation_profile(&self, operation: &str) -> Option<OperationProfile> {
        let durations = self.operation_stats.get(operation)?;
        if durations.is_empty() {
            return None;
        }

        let timing = Self::calculate_timing_stats(durations);
        let memory = self
            .operation_memory
            .get(operation)
            .map(|m| Self::calculate_memory_stats(m))
            .unwrap_or_else(|| MemoryStatistics {
                total_allocations: 0,
                total_frees: 0,
                peak_memory_bytes: 0,
                avg_memory_bytes: 0,
                current_memory_bytes: 0,
            });

        let failure_count = self
            .spans
            .iter()
            .filter(|s| s.name == operation && s.tags.contains_key("error"))
            .count();

        Some(OperationProfile {
            operation: operation.to_string(),
            timing,
            memory,
            invocation_count: durations.len(),
            failure_count,
        })
    }

    fn calculate_timing_stats(durations: &[Duration]) -> TimingStatistics {
        let mut sorted = durations.to_vec();
        sorted.sort();

        let count = sorted.len();
        let min_duration = *sorted.first().expect("durations slice is non-empty");
        let max_duration = *sorted.last().expect("durations slice is non-empty");
        let total_duration: Duration = sorted.iter().sum();
        let avg_duration = total_duration / count as u32;

        let median_duration = sorted[count / 2];
        let p95_duration = sorted[(count as f64 * 0.95) as usize];
        let p99_duration = sorted[(count as f64 * 0.99) as usize];

        TimingStatistics {
            count,
            min_duration,
            max_duration,
            avg_duration,
            median_duration,
            p95_duration,
            p99_duration,
            total_duration,
        }
    }

    fn calculate_memory_stats(allocations: &[u64]) -> MemoryStatistics {
        if allocations.is_empty() {
            return MemoryStatistics {
                total_allocations: 0,
                total_frees: 0,
                peak_memory_bytes: 0,
                avg_memory_bytes: 0,
                current_memory_bytes: 0,
            };
        }

        let total_allocations: u64 = allocations.iter().sum();
        let peak_memory_bytes = *allocations.iter().max().expect("checked non-empty above");
        let avg_memory_bytes = total_allocations / allocations.len() as u64;

        MemoryStatistics {
            total_allocations,
            total_frees: 0, // We don't track individual frees in this simple implementation
            peak_memory_bytes,
            avg_memory_bytes,
            current_memory_bytes: 0,
        }
    }
}

/// Performance profiler for detailed operation tracking
#[derive(Clone)]
pub struct PerformanceProfiler {
    inner: Arc<Mutex<PerformanceProfilerInner>>,
}

impl PerformanceProfiler {
    /// Create a new performance profiler with the given configuration
    pub fn new(config: ProfilingConfig) -> Self {
        info!("Initializing performance profiler");
        Self {
            inner: Arc::new(Mutex::new(PerformanceProfilerInner::new(config))),
        }
    }

    /// Create profiler with default configuration
    pub fn default_config() -> Self {
        Self::new(ProfilingConfig::default())
    }

    /// Begin a new performance span
    pub fn begin_span(&self, name: impl Into<String>) -> Result<SpanGuard> {
        self.begin_span_with_parent(name, None)
    }

    /// Begin a new performance span with a parent
    pub fn begin_span_with_parent(
        &self,
        name: impl Into<String>,
        parent_id: Option<SpanId>,
    ) -> Result<SpanGuard> {
        let name_str = name.into();
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Lock error: {}", e),
            })?;

        let (id, start_memory) = inner.begin_span(name_str.clone(), parent_id);

        Ok(SpanGuard {
            id,
            name: name_str,
            start: Instant::now(),
            start_memory,
            profiler: Arc::clone(&self.inner),
            tags: HashMap::new(),
        })
    }

    /// Get profile statistics for a specific operation
    pub fn get_operation_profile(&self, operation: &str) -> Result<Option<OperationProfile>> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Lock error: {}", e),
            })?;

        Ok(inner.get_operation_profile(operation))
    }

    /// Get all recorded spans
    pub fn get_spans(&self) -> Result<Vec<PerformanceSpan>> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Lock error: {}", e),
            })?;

        Ok(inner.spans.clone())
    }

    /// Get all operation profiles
    pub fn get_all_profiles(&self) -> Result<Vec<OperationProfile>> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Lock error: {}", e),
            })?;

        Ok(inner
            .operation_stats
            .keys()
            .filter_map(|op| inner.get_operation_profile(op))
            .collect())
    }

    /// Generate a comprehensive profiling report
    pub fn generate_report(&self) -> Result<ProfilingReport> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Lock error: {}", e),
            })?;

        let uptime = inner.start_time.elapsed();
        let total_spans = inner.spans.len();
        let active_spans = inner.active_spans.len();

        let operation_profiles = inner
            .operation_stats
            .keys()
            .filter_map(|op| inner.get_operation_profile(op))
            .collect();

        Ok(ProfilingReport {
            timestamp: SystemTime::now(),
            uptime,
            total_spans_recorded: total_spans,
            active_spans,
            operation_profiles,
        })
    }

    /// Clear all profiling data
    pub fn clear(&self) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Lock error: {}", e),
            })?;

        inner.spans.clear();
        inner.operation_stats.clear();
        inner.operation_memory.clear();

        info!("Profiling data cleared");
        Ok(())
    }

    /// Get current configuration
    pub fn config(&self) -> Result<ProfilingConfig> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| AcousticError::ProcessingError {
                message: format!("Lock error: {}", e),
            })?;

        Ok(inner.config.clone())
    }
}

/// Comprehensive profiling report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingReport {
    /// Report timestamp
    pub timestamp: SystemTime,
    /// Profiler uptime
    pub uptime: Duration,
    /// Total spans recorded
    pub total_spans_recorded: usize,
    /// Currently active spans
    pub active_spans: usize,
    /// Per-operation profiles
    pub operation_profiles: Vec<OperationProfile>,
}

impl ProfilingReport {
    /// Export report as JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| AcousticError::ProcessingError {
            message: format!("JSON export failed: {}", e),
        })
    }

    /// Export report as human-readable text
    pub fn to_text(&self) -> String {
        let mut output = String::new();

        output.push_str("=== Performance Profiling Report ===\n");
        output.push_str(&format!("Timestamp: {:?}\n", self.timestamp));
        output.push_str(&format!("Uptime: {:?}\n", self.uptime));
        output.push_str(&format!("Total Spans: {}\n", self.total_spans_recorded));
        output.push_str(&format!("Active Spans: {}\n\n", self.active_spans));

        output.push_str("=== Operation Profiles ===\n");
        for profile in &self.operation_profiles {
            output.push_str(&format!("\nOperation: {}\n", profile.operation));
            output.push_str(&format!("  Invocations: {}\n", profile.invocation_count));
            output.push_str(&format!("  Failures: {}\n", profile.failure_count));
            output.push_str(&format!("  Min: {:?}\n", profile.timing.min_duration));
            output.push_str(&format!("  Max: {:?}\n", profile.timing.max_duration));
            output.push_str(&format!("  Avg: {:?}\n", profile.timing.avg_duration));
            output.push_str(&format!("  P50: {:?}\n", profile.timing.median_duration));
            output.push_str(&format!("  P95: {:?}\n", profile.timing.p95_duration));
            output.push_str(&format!("  P99: {:?}\n", profile.timing.p99_duration));
            output.push_str(&format!(
                "  Peak Memory: {} KB\n",
                profile.memory.peak_memory_bytes / 1024
            ));
        }

        output
    }

    /// Get the slowest operations
    pub fn slowest_operations(&self, n: usize) -> Vec<&OperationProfile> {
        let mut sorted = self.operation_profiles.iter().collect::<Vec<_>>();
        sorted.sort_by(|a, b| b.timing.avg_duration.cmp(&a.timing.avg_duration));
        sorted.into_iter().take(n).collect()
    }

    /// Get the most memory-intensive operations
    pub fn most_memory_intensive(&self, n: usize) -> Vec<&OperationProfile> {
        let mut sorted = self.operation_profiles.iter().collect::<Vec<_>>();
        sorted.sort_by(|a, b| b.memory.peak_memory_bytes.cmp(&a.memory.peak_memory_bytes));
        sorted.into_iter().take(n).collect()
    }
}

/// Get current memory usage (approximate)
fn get_current_memory() -> u64 {
    // This is a simplified implementation
    // In production, you might use platform-specific APIs
    // or memory profiling tools for more accurate tracking
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb) = line.split_whitespace().nth(1) {
                        if let Ok(kb_val) = kb.parse::<u64>() {
                            return kb_val * 1024;
                        }
                    }
                }
            }
        }
    }

    // Fallback: return 0 (no tracking)
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_profiler_basic() {
        let profiler = PerformanceProfiler::new(ProfilingConfig::development());

        {
            let _span = profiler.begin_span("test_operation").unwrap();
            thread::sleep(Duration::from_millis(10));
        }

        let profile = profiler
            .get_operation_profile("test_operation")
            .unwrap()
            .expect("Profile should exist");

        assert_eq!(profile.invocation_count, 1);
        assert!(profile.timing.avg_duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_profiler_multiple_spans() {
        let profiler = PerformanceProfiler::new(ProfilingConfig::development());

        for i in 0..5 {
            let _span = profiler.begin_span("repeated_operation").unwrap();
            thread::sleep(Duration::from_millis(5));
        }

        let profile = profiler
            .get_operation_profile("repeated_operation")
            .unwrap()
            .expect("Profile should exist");

        assert_eq!(profile.invocation_count, 5);
    }

    #[test]
    fn test_profiler_nested_spans() {
        let profiler = PerformanceProfiler::new(ProfilingConfig::development());

        {
            let parent = profiler.begin_span("parent_operation").unwrap();
            thread::sleep(Duration::from_millis(5));

            {
                let _child = profiler
                    .begin_span_with_parent("child_operation", Some(parent.id()))
                    .unwrap();
                thread::sleep(Duration::from_millis(3));
            }
        }

        let parent_profile = profiler
            .get_operation_profile("parent_operation")
            .unwrap()
            .expect("Parent profile should exist");
        let child_profile = profiler
            .get_operation_profile("child_operation")
            .unwrap()
            .expect("Child profile should exist");

        assert_eq!(parent_profile.invocation_count, 1);
        assert_eq!(child_profile.invocation_count, 1);
    }

    #[test]
    fn test_profiler_tags() {
        let profiler = PerformanceProfiler::new(ProfilingConfig::development());

        {
            let mut span = profiler.begin_span("tagged_operation").unwrap();
            span.add_tag("user_id", "12345");
            span.add_tag("request_type", "synthesis");
            thread::sleep(Duration::from_millis(5));
        }

        let spans = profiler.get_spans().unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].tags.get("user_id").unwrap(), "12345");
    }

    #[test]
    fn test_report_generation() {
        let profiler = PerformanceProfiler::new(ProfilingConfig::development());

        for _ in 0..3 {
            let _span = profiler.begin_span("report_test").unwrap();
            thread::sleep(Duration::from_millis(5));
        }

        let report = profiler.generate_report().unwrap();
        assert_eq!(report.total_spans_recorded, 3);
        assert_eq!(report.operation_profiles.len(), 1);

        let json = report.to_json().unwrap();
        assert!(json.contains("report_test"));

        let text = report.to_text();
        assert!(text.contains("report_test"));
    }

    #[test]
    fn test_profiler_clear() {
        let profiler = PerformanceProfiler::new(ProfilingConfig::development());

        {
            let _span = profiler.begin_span("clear_test").unwrap();
            thread::sleep(Duration::from_millis(1)); // Ensure span is long enough to be recorded
        }

        assert_eq!(profiler.get_spans().unwrap().len(), 1);

        profiler.clear().unwrap();

        assert_eq!(profiler.get_spans().unwrap().len(), 0);
    }

    #[test]
    fn test_timing_statistics() {
        let durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
            Duration::from_millis(40),
            Duration::from_millis(50),
        ];

        let stats = PerformanceProfilerInner::calculate_timing_stats(&durations);

        assert_eq!(stats.count, 5);
        assert_eq!(stats.min_duration, Duration::from_millis(10));
        assert_eq!(stats.max_duration, Duration::from_millis(50));
        assert_eq!(stats.median_duration, Duration::from_millis(30));
    }

    #[test]
    fn test_config_presets() {
        let dev = ProfilingConfig::development();
        assert!(dev.enable_tracing);
        assert_eq!(dev.memory_sample_rate, 1.0);

        let prod = ProfilingConfig::production();
        assert_eq!(prod.memory_sample_rate, 0.05);

        let minimal = ProfilingConfig::minimal();
        assert!(!minimal.enable_tracing);
    }
}
