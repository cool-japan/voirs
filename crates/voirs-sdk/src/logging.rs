//! Logging configuration and utilities for VoiRS.

use crate::{config::LoggingConfig, error::Result, VoirsError};
// Note: fs and Path imports removed as they're not used in this logging module
use tracing::Level;
use tracing_subscriber::{
    fmt::{self},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Layer,
};

/// Initialize logging based on configuration
pub fn init_logging(config: &LoggingConfig) -> Result<()> {
    let level = parse_level(&config.level)?;

    // Create base filter
    let mut env_filter = EnvFilter::builder()
        .with_default_directive(level.into())
        .from_env()
        .map_err(|e| VoirsError::config_error(format!("Invalid log filter: {e}")))?;

    // Add module-specific log levels
    for (module, module_level) in &config.module_levels {
        let module_level = parse_level(module_level)?;
        env_filter = env_filter.add_directive(
            format!("{module}={module_level}")
                .parse()
                .map_err(|e| VoirsError::config_error(format!("Invalid module filter: {e}")))?,
        );
    }

    // Create layers
    let console_layer = if config.structured {
        fmt::layer().json().boxed()
    } else {
        fmt::layer().pretty().boxed()
    };

    // Build base subscriber
    let mut layers = Vec::new();
    layers.push(console_layer);

    // Add file layer if configured
    if let Some(file_path) = &config.file_path {
        // Ensure parent directory exists
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Create or open log file
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(file_path)?;

        let file_layer = if config.structured {
            fmt::layer()
                .with_writer(file)
                .with_target(true)
                .with_thread_ids(true)
                .with_thread_names(true)
                .json()
                .boxed()
        } else {
            fmt::layer()
                .with_writer(file)
                .with_target(true)
                .with_thread_ids(true)
                .with_thread_names(true)
                .boxed()
        };
        layers.push(file_layer);
    }

    // Add metrics layer if enabled
    if config.metrics {
        use tracing_subscriber::filter::FilterFn;

        let metrics_filter = FilterFn::new(|metadata| {
            metadata.target().starts_with("voirs::metrics")
                || metadata
                    .fields()
                    .iter()
                    .any(|f| f.name().starts_with("metric_"))
        });

        let metrics_layer = fmt::layer()
            .with_writer(std::io::stderr)
            .with_target(true)
            .with_thread_ids(false)
            .json() // Use JSON format for structured metrics
            .with_filter(metrics_filter)
            .boxed();

        layers.push(metrics_layer);
    }

    // Initialize subscriber
    tracing_subscriber::registry()
        .with(env_filter)
        .with(layers)
        .init();

    tracing::info!("VoiRS logging initialized");
    tracing::debug!("Log level: {}", config.level);
    tracing::debug!("Structured logging: {}", config.structured);
    tracing::debug!("Metrics enabled: {}", config.metrics);

    if let Some(file_path) = &config.file_path {
        tracing::debug!("File logging: {}", file_path.display());
    }

    Ok(())
}

/// Parse log level from string
fn parse_level(level_str: &str) -> Result<Level> {
    match level_str.to_lowercase().as_str() {
        "trace" => Ok(Level::TRACE),
        "debug" => Ok(Level::DEBUG),
        "info" => Ok(Level::INFO),
        "warn" | "warning" => Ok(Level::WARN),
        "error" => Ok(Level::ERROR),
        _ => Err(VoirsError::config_error(format!(
            "Invalid log level: {level_str}. Valid levels: trace, debug, info, warn, error"
        ))),
    }
}

/// Performance timing utilities
pub struct PerfTimer {
    start: std::time::Instant,
    name: String,
}

impl PerfTimer {
    /// Start a new performance timer
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        tracing::debug!("Starting timer: {}", name);
        Self {
            start: std::time::Instant::now(),
            name,
        }
    }

    /// Record elapsed time and log it
    pub fn elapsed(&self) -> std::time::Duration {
        let elapsed = self.start.elapsed();
        tracing::debug!(
            "Timer '{}': {:.2}ms",
            self.name,
            elapsed.as_secs_f64() * 1000.0
        );
        elapsed
    }

    /// Record elapsed time with custom message
    pub fn elapsed_with_message(&self, message: &str) -> std::time::Duration {
        let elapsed = self.start.elapsed();
        tracing::info!("{}: {:.2}ms", message, elapsed.as_secs_f64() * 1000.0);
        elapsed
    }
}

impl Drop for PerfTimer {
    fn drop(&mut self) {
        self.elapsed();
    }
}

/// Memory usage tracking utilities
pub struct MemoryTracker {
    initial_usage: u64,
    name: String,
}

impl MemoryTracker {
    /// Start memory tracking
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let initial_usage = get_memory_usage();
        tracing::debug!(
            "Starting memory tracking: {} (initial: {} KB)",
            name,
            initial_usage / 1024
        );
        Self {
            initial_usage,
            name,
        }
    }

    /// Get current memory delta
    pub fn delta(&self) -> i64 {
        let current = get_memory_usage();
        current as i64 - self.initial_usage as i64
    }

    /// Log current memory delta
    pub fn log_delta(&self) {
        let delta = self.delta();
        if delta > 0 {
            tracing::debug!("Memory '{}': +{} KB", self.name, delta / 1024);
        } else {
            tracing::debug!("Memory '{}': {} KB", self.name, delta / 1024);
        }
    }
}

impl Drop for MemoryTracker {
    fn drop(&mut self) {
        self.log_delta();
    }
}

/// Get current memory usage (approximate)
fn get_memory_usage() -> u64 {
    // This is a simplified implementation
    // In production, you might want to use more sophisticated memory tracking
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|content| {
                content
                    .lines()
                    .find(|line| line.starts_with("VmRSS:"))
                    .and_then(|line| {
                        line.split_whitespace()
                            .nth(1)
                            .and_then(|s| s.parse::<u64>().ok())
                            .map(|kb| kb * 1024) // Convert to bytes
                    })
            })
            .unwrap_or(0)
    }

    #[cfg(not(target_os = "linux"))]
    {
        0 // Placeholder for other platforms
    }
}

/// Metrics collection and emission utilities
pub mod metrics {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// Global metrics registry for application-wide statistics
    pub struct MetricsRegistry {
        counters: HashMap<String, Arc<AtomicU64>>,
        histograms: HashMap<String, Vec<f64>>,
        gauges: HashMap<String, Arc<AtomicU64>>,
    }

    impl MetricsRegistry {
        /// Create a new metrics registry
        pub fn new() -> Self {
            Self {
                counters: HashMap::new(),
                histograms: HashMap::new(),
                gauges: HashMap::new(),
            }
        }

        /// Increment a counter metric
        pub fn increment_counter(&mut self, name: &str, value: u64) {
            let counter = self
                .counters
                .entry(name.to_string())
                .or_insert_with(|| Arc::new(AtomicU64::new(0)));
            counter.fetch_add(value, Ordering::Relaxed);
        }

        /// Record a histogram value
        pub fn record_histogram(&mut self, name: &str, value: f64) {
            self.histograms
                .entry(name.to_string())
                .or_default()
                .push(value);
        }

        /// Set a gauge value
        pub fn set_gauge(&mut self, name: &str, value: u64) {
            let gauge = self
                .gauges
                .entry(name.to_string())
                .or_insert_with(|| Arc::new(AtomicU64::new(0)));
            gauge.store(value, Ordering::Relaxed);
        }

        /// Get current counter value
        pub fn get_counter(&self, name: &str) -> u64 {
            self.counters
                .get(name)
                .map(|c| c.load(Ordering::Relaxed))
                .unwrap_or(0)
        }

        /// Get histogram statistics
        pub fn get_histogram_stats(&self, name: &str) -> Option<HistogramStats> {
            self.histograms.get(name).map(|values| {
                if values.is_empty() {
                    return HistogramStats::default();
                }

                let mut sorted = values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let count = sorted.len();
                let sum: f64 = sorted.iter().sum();
                let mean = sum / count as f64;
                let min = sorted[0];
                let max = sorted[count - 1];
                let p50 = sorted[count / 2];
                let p95 = sorted[(count as f64 * 0.95) as usize];
                let p99 = sorted[(count as f64 * 0.99) as usize];

                HistogramStats {
                    count,
                    sum,
                    mean,
                    min,
                    max,
                    p50,
                    p95,
                    p99,
                }
            })
        }

        /// Get current gauge value
        pub fn get_gauge(&self, name: &str) -> u64 {
            self.gauges
                .get(name)
                .map(|g| g.load(Ordering::Relaxed))
                .unwrap_or(0)
        }

        /// Export all metrics as structured data
        pub fn export_metrics(&self) -> MetricsSnapshot {
            let counters: HashMap<String, u64> = self
                .counters
                .iter()
                .map(|(k, v)| (k.clone(), v.load(Ordering::Relaxed)))
                .collect();

            let histograms: HashMap<String, HistogramStats> = self
                .histograms
                .iter()
                .filter_map(|(k, _)| self.get_histogram_stats(k).map(|stats| (k.clone(), stats)))
                .collect();

            let gauges: HashMap<String, u64> = self
                .gauges
                .iter()
                .map(|(k, v)| (k.clone(), v.load(Ordering::Relaxed)))
                .collect();

            MetricsSnapshot {
                timestamp: std::time::SystemTime::now(),
                counters,
                histograms,
                gauges,
            }
        }
    }

    impl Default for MetricsRegistry {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Histogram statistics
    #[derive(Debug, Clone)]
    pub struct HistogramStats {
        pub count: usize,
        pub sum: f64,
        pub mean: f64,
        pub min: f64,
        pub max: f64,
        pub p50: f64,
        pub p95: f64,
        pub p99: f64,
    }

    impl Default for HistogramStats {
        fn default() -> Self {
            Self {
                count: 0,
                sum: 0.0,
                mean: 0.0,
                min: 0.0,
                max: 0.0,
                p50: 0.0,
                p95: 0.0,
                p99: 0.0,
            }
        }
    }

    /// Snapshot of all metrics at a point in time
    #[derive(Debug)]
    pub struct MetricsSnapshot {
        pub timestamp: std::time::SystemTime,
        pub counters: HashMap<String, u64>,
        pub histograms: HashMap<String, HistogramStats>,
        pub gauges: HashMap<String, u64>,
    }

    /// Emit a counter metric via structured logging
    pub fn emit_counter(name: &str, value: u64, tags: Option<&[(&str, &str)]>) {
        let tags_str = format_tags(tags);
        tracing::info!(
            target: "voirs::metrics",
            metric_type = "counter",
            metric_name = %name,
            metric_value = %value,
            metric_tags = %tags_str,
            "Counter metric"
        );
    }

    /// Emit a gauge metric via structured logging
    pub fn emit_gauge(name: &str, value: f64, tags: Option<&[(&str, &str)]>) {
        let tags_str = format_tags(tags);
        tracing::info!(
            target: "voirs::metrics",
            metric_type = "gauge",
            metric_name = %name,
            metric_value = %value,
            metric_tags = %tags_str,
            "Gauge metric"
        );
    }

    /// Emit a histogram/timing metric via structured logging
    pub fn emit_histogram(name: &str, value: f64, unit: &str, tags: Option<&[(&str, &str)]>) {
        let tags_str = format_tags(tags);
        tracing::info!(
            target: "voirs::metrics",
            metric_type = "histogram",
            metric_name = %name,
            metric_value = %value,
            metric_unit = %unit,
            metric_tags = %tags_str,
            "Histogram metric"
        );
    }

    /// Emit synthesis performance metrics
    pub fn emit_synthesis_metrics(
        duration: Duration,
        audio_duration: Duration,
        text_length: usize,
        model_type: &str,
        voice_id: &str,
    ) {
        let rtf = audio_duration.as_secs_f64() / duration.as_secs_f64();
        let tags = &[("model_type", model_type), ("voice_id", voice_id)];

        emit_histogram(
            "synthesis.duration",
            duration.as_secs_f64(),
            "seconds",
            Some(tags),
        );
        emit_histogram(
            "synthesis.audio_duration",
            audio_duration.as_secs_f64(),
            "seconds",
            Some(tags),
        );
        emit_histogram("synthesis.real_time_factor", rtf, "ratio", Some(tags));
        emit_gauge("synthesis.text_length", text_length as f64, Some(tags));
        emit_counter("synthesis.requests", 1, Some(tags));
    }

    /// Emit model loading metrics
    pub fn emit_model_metrics(
        model_type: &str,
        load_duration: Duration,
        model_size_mb: f64,
        device: &str,
    ) {
        let tags = &[("model_type", model_type), ("device", device)];

        emit_histogram(
            "model.load_duration",
            load_duration.as_secs_f64(),
            "seconds",
            Some(tags),
        );
        emit_gauge("model.size_mb", model_size_mb, Some(tags));
        emit_counter("model.loads", 1, Some(tags));
    }

    /// Emit error metrics
    pub fn emit_error_metrics(error_type: &str, component: &str, severity: &str) {
        let tags = &[
            ("error_type", error_type),
            ("component", component),
            ("severity", severity),
        ];

        emit_counter("errors.total", 1, Some(tags));
    }

    /// Emit memory usage metrics
    pub fn emit_memory_metrics(component: &str, memory_mb: f64) {
        let tags = &[("component", component)];
        emit_gauge("memory.usage_mb", memory_mb, Some(tags));
    }

    /// Emit audio quality metrics
    pub fn emit_quality_metrics(
        metric_name: &str,
        score: f64,
        model_type: &str,
        has_reference: bool,
    ) {
        let tags = &[
            ("metric", metric_name),
            ("model_type", model_type),
            (
                "has_reference",
                if has_reference { "true" } else { "false" },
            ),
        ];

        emit_histogram("quality.score", score, "score", Some(tags));
        emit_counter("quality.evaluations", 1, Some(tags));
    }

    /// Format tags for logging
    fn format_tags(tags: Option<&[(&str, &str)]>) -> String {
        match tags {
            Some(tags) => tags
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(","),
            None => String::new(),
        }
    }

    /// Metrics timer that automatically emits timing data when dropped
    pub struct MetricsTimer {
        name: String,
        start: Instant,
        tags: Vec<(String, String)>,
    }

    impl MetricsTimer {
        /// Start a new metrics timer
        pub fn new(name: impl Into<String>) -> Self {
            Self {
                name: name.into(),
                start: Instant::now(),
                tags: Vec::new(),
            }
        }

        /// Add a tag to the timer
        pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
            self.tags.push((key.into(), value.into()));
            self
        }

        /// Record the elapsed time
        pub fn finish(self) {
            let duration = self.start.elapsed();
            let tags: Vec<(&str, &str)> = self
                .tags
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();

            emit_histogram(
                &self.name,
                duration.as_secs_f64(),
                "seconds",
                if tags.is_empty() { None } else { Some(&tags) },
            );
        }
    }

    impl Drop for MetricsTimer {
        fn drop(&mut self) {
            let duration = self.start.elapsed();
            let tags: Vec<(&str, &str)> = self
                .tags
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();

            emit_histogram(
                &self.name,
                duration.as_secs_f64(),
                "seconds",
                if tags.is_empty() { None } else { Some(&tags) },
            );
        }
    }
}

/// Macros for convenient logging with context
#[macro_export]
macro_rules! log_synthesis {
    ($level:ident, $text:expr, $($arg:tt)*) => {
        tracing::$level!(
            target: "voirs::synthesis",
            text = %$text,
            $($arg)*
        );
    };
}

#[macro_export]
macro_rules! log_audio {
    ($level:ident, $duration:expr, $sample_rate:expr, $($arg:tt)*) => {
        tracing::$level!(
            target: "voirs::audio",
            duration_sec = %$duration,
            sample_rate = %$sample_rate,
            $($arg)*
        );
    };
}

#[macro_export]
macro_rules! log_model {
    ($level:ident, $model_type:expr, $($arg:tt)*) => {
        tracing::$level!(
            target: "voirs::model",
            model_type = %$model_type,
            $($arg)*
        );
    };
}

/// Specialized loggers for different components
pub mod synthesis {

    pub fn log_start(text: &str) {
        tracing::info!(
            target: "voirs::synthesis",
            text = %text,
            "Starting synthesis"
        );
    }

    pub fn log_complete(text: &str, duration_ms: f64, audio_duration_sec: f32) {
        tracing::info!(
            target: "voirs::synthesis",
            text = %text,
            processing_time_ms = %duration_ms,
            audio_duration_sec = %audio_duration_sec,
            real_time_factor = %(audio_duration_sec * 1000.0 / duration_ms as f32),
            "Synthesis complete"
        );
    }

    pub fn log_error(text: &str, error: &dyn std::error::Error) {
        tracing::error!(
            target: "voirs::synthesis",
            text = %text,
            error = %error,
            "Synthesis failed"
        );
    }
}

pub mod model {
    use std::path::Path;

    pub fn log_load(model_type: &str, path: &Path) {
        tracing::info!(
            target: "voirs::model",
            model_type = %model_type,
            path = %path.display(),
            "Loading model"
        );
    }

    pub fn log_load_complete(model_type: &str, duration_ms: f64) {
        tracing::info!(
            target: "voirs::model",
            model_type = %model_type,
            load_time_ms = %duration_ms,
            "Model loaded successfully"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_parse_level() {
        assert_eq!(parse_level("info").unwrap(), Level::INFO);
        assert_eq!(parse_level("DEBUG").unwrap(), Level::DEBUG);
        assert_eq!(parse_level("warn").unwrap(), Level::WARN);
        assert!(parse_level("invalid").is_err());
    }

    #[test]
    fn test_perf_timer() {
        let timer = PerfTimer::new("test");
        std::thread::sleep(std::time::Duration::from_millis(1));
        let elapsed = timer.elapsed();
        assert!(elapsed.as_millis() >= 1);
    }

    #[test]
    fn test_memory_tracker() {
        let tracker = MemoryTracker::new("test");
        // Allocate some memory
        let _data: Vec<u8> = vec![0; 1024 * 1024]; // 1MB
        tracker.log_delta();
        // Memory delta might be 0 on some systems, but at least it shouldn't crash
    }

    #[test]
    fn test_logging_init() {
        let config = LoggingConfig {
            level: "debug".to_string(),
            structured: false,
            file_path: None,
            max_file_size_mb: 10,
            max_files: 5,
            metrics: false,
            module_levels: std::collections::HashMap::new(),
        };

        // This might fail if logging is already initialized in other tests
        let _ = init_logging(&config);
    }

    #[test]
    fn test_file_logging() {
        let temp_dir = tempdir().unwrap();
        let log_file = temp_dir.path().join("test.log");

        // Test file creation manually
        std::fs::create_dir_all(log_file.parent().unwrap()).unwrap();
        let _file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_file)
            .unwrap();

        assert!(log_file.exists());
    }

    #[test]
    fn test_metrics_registry() {
        let mut registry = metrics::MetricsRegistry::new();

        // Test counter
        registry.increment_counter("test_counter", 5);
        assert_eq!(registry.get_counter("test_counter"), 5);
        registry.increment_counter("test_counter", 3);
        assert_eq!(registry.get_counter("test_counter"), 8);

        // Test gauge
        registry.set_gauge("test_gauge", 42);
        assert_eq!(registry.get_gauge("test_gauge"), 42);

        // Test histogram
        registry.record_histogram("test_histogram", 1.5);
        registry.record_histogram("test_histogram", 2.5);
        registry.record_histogram("test_histogram", 3.5);

        let stats = registry.get_histogram_stats("test_histogram").unwrap();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.min, 1.5);
        assert_eq!(stats.max, 3.5);
        assert_eq!(stats.mean, 2.5);
    }

    #[test]
    fn test_metrics_timer() {
        let timer = metrics::MetricsTimer::new("test_timer")
            .with_tag("component", "test")
            .with_tag("operation", "benchmark");

        std::thread::sleep(std::time::Duration::from_millis(1));
        timer.finish();
        // Timer should emit metrics when finished
    }

    #[test]
    fn test_metrics_emission() {
        // Test metric emission functions don't panic
        metrics::emit_counter("test.counter", 1, Some(&[("tag", "value")]));
        metrics::emit_gauge("test.gauge", 42.0, None);
        metrics::emit_histogram("test.histogram", 1.5, "seconds", Some(&[("unit", "time")]));

        // Test specialized metrics
        metrics::emit_synthesis_metrics(
            std::time::Duration::from_millis(500),
            std::time::Duration::from_secs(2),
            50,
            "vits",
            "en-us-female",
        );

        metrics::emit_model_metrics("acoustic", std::time::Duration::from_secs(3), 100.5, "cuda");

        metrics::emit_error_metrics("timeout", "synthesis", "warning");
        metrics::emit_memory_metrics("model_cache", 512.0);
        metrics::emit_quality_metrics("pesq", 4.2, "vits", true);
    }
}
