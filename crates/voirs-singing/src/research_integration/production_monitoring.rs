// Production Monitoring System - Phase 5 Feature
//
// Comprehensive production monitoring with Prometheus metrics, OpenTelemetry tracing,
// and advanced observability for singing synthesis systems.

use crate::Error;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Prometheus-compatible metrics collector for singing synthesis
#[derive(Debug, Clone)]
pub struct PrometheusMetrics {
    /// Counter metrics (monotonically increasing)
    counters: Arc<RwLock<HashMap<String, f64>>>,
    /// Gauge metrics (can go up or down)
    gauges: Arc<RwLock<HashMap<String, f64>>>,
    /// Histogram metrics (distribution of values)
    histograms: Arc<RwLock<HashMap<String, Vec<f64>>>>,
    /// Metric metadata
    metadata: Arc<RwLock<HashMap<String, MetricMetadata>>>,
}

#[derive(Debug, Clone)]
pub struct MetricMetadata {
    pub name: String,
    pub help: String,
    pub metric_type: MetricType,
    pub labels: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

impl PrometheusMetrics {
    pub fn new() -> Self {
        Self {
            counters: Arc::new(RwLock::new(HashMap::new())),
            gauges: Arc::new(RwLock::new(HashMap::new())),
            histograms: Arc::new(RwLock::new(HashMap::new())),
            metadata: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Increment a counter metric
    pub fn increment_counter(&self, name: &str, value: f64) -> Result<(), Error> {
        let mut counters = self
            .counters
            .write()
            .map_err(|_| Error::Processing("Failed to acquire counter lock".into()))?;
        *counters.entry(name.to_string()).or_insert(0.0) += value;
        Ok(())
    }

    /// Set a gauge metric
    pub fn set_gauge(&self, name: &str, value: f64) -> Result<(), Error> {
        let mut gauges = self
            .gauges
            .write()
            .map_err(|_| Error::Processing("Failed to acquire gauge lock".into()))?;
        gauges.insert(name.to_string(), value);
        Ok(())
    }

    /// Record a histogram observation
    pub fn observe_histogram(&self, name: &str, value: f64) -> Result<(), Error> {
        let mut histograms = self
            .histograms
            .write()
            .map_err(|_| Error::Processing("Failed to acquire histogram lock".into()))?;
        histograms
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(value);
        Ok(())
    }

    /// Register metric metadata
    pub fn register_metric(
        &self,
        name: &str,
        help: &str,
        metric_type: MetricType,
        labels: HashMap<String, String>,
    ) -> Result<(), Error> {
        let mut metadata = self
            .metadata
            .write()
            .map_err(|_| Error::Processing("Failed to acquire metadata lock".into()))?;
        metadata.insert(
            name.to_string(),
            MetricMetadata {
                name: name.to_string(),
                help: help.to_string(),
                metric_type,
                labels,
            },
        );
        Ok(())
    }

    /// Export metrics in Prometheus text format
    pub fn export_prometheus_format(&self) -> Result<String, Error> {
        let mut output = String::new();

        let metadata = self
            .metadata
            .read()
            .map_err(|_| Error::Processing("Failed to read metadata".into()))?;
        let counters = self
            .counters
            .read()
            .map_err(|_| Error::Processing("Failed to read counters".into()))?;
        let gauges = self
            .gauges
            .read()
            .map_err(|_| Error::Processing("Failed to read gauges".into()))?;
        let histograms = self
            .histograms
            .read()
            .map_err(|_| Error::Processing("Failed to read histograms".into()))?;

        // Export counters
        for (name, value) in counters.iter() {
            if let Some(meta) = metadata.get(name) {
                output.push_str(&format!("# HELP {} {}\n", name, meta.help));
                output.push_str(&format!("# TYPE {} counter\n", name));
            }
            output.push_str(&format!("{} {}\n", name, value));
        }

        // Export gauges
        for (name, value) in gauges.iter() {
            if let Some(meta) = metadata.get(name) {
                output.push_str(&format!("# HELP {} {}\n", name, meta.help));
                output.push_str(&format!("# TYPE {} gauge\n", name));
            }
            output.push_str(&format!("{} {}\n", name, value));
        }

        // Export histograms
        for (name, values) in histograms.iter() {
            if let Some(meta) = metadata.get(name) {
                output.push_str(&format!("# HELP {} {}\n", name, meta.help));
                output.push_str(&format!("# TYPE {} histogram\n", name));
            }

            let count = values.len();
            let sum: f64 = values.iter().sum();

            output.push_str(&format!("{}_count {}\n", name, count));
            output.push_str(&format!("{}_sum {}\n", name, sum));

            // Compute histogram buckets
            let buckets = vec![0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0];
            for bucket in buckets {
                let count_below = values.iter().filter(|&&v| v <= bucket).count();
                output.push_str(&format!(
                    "{}_bucket{{le=\"{}\"}} {}\n",
                    name, bucket, count_below
                ));
            }
            output.push_str(&format!("{}_bucket{{le=\"+Inf\"}} {}\n", name, count));
        }

        Ok(output)
    }

    /// Get counter value
    pub fn get_counter(&self, name: &str) -> Result<f64, Error> {
        let counters = self
            .counters
            .read()
            .map_err(|_| Error::Processing("Failed to read counters".into()))?;
        Ok(counters.get(name).copied().unwrap_or(0.0))
    }

    /// Get gauge value
    pub fn get_gauge(&self, name: &str) -> Result<f64, Error> {
        let gauges = self
            .gauges
            .read()
            .map_err(|_| Error::Processing("Failed to read gauges".into()))?;
        Ok(gauges.get(name).copied().unwrap_or(0.0))
    }

    /// Get histogram statistics
    pub fn get_histogram_stats(&self, name: &str) -> Result<HistogramStats, Error> {
        let histograms = self
            .histograms
            .read()
            .map_err(|_| Error::Processing("Failed to read histograms".into()))?;

        let values = histograms
            .get(name)
            .ok_or_else(|| Error::Processing(format!("Histogram {} not found", name)))?;

        if values.is_empty() {
            return Ok(HistogramStats::default());
        }

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let count = sorted.len();
        let sum: f64 = sorted.iter().sum();
        let mean = sum / count as f64;

        let p50 = sorted[count / 2];
        let p95 = sorted[count * 95 / 100];
        let p99 = sorted[count * 99 / 100];

        Ok(HistogramStats {
            count,
            sum,
            mean,
            p50,
            p95,
            p99,
            min: sorted[0],
            max: sorted[count - 1],
        })
    }

    /// Reset all metrics
    pub fn reset(&self) -> Result<(), Error> {
        self.counters
            .write()
            .map_err(|_| Error::Processing("Failed to acquire counter lock".into()))?
            .clear();
        self.gauges
            .write()
            .map_err(|_| Error::Processing("Failed to acquire gauge lock".into()))?
            .clear();
        self.histograms
            .write()
            .map_err(|_| Error::Processing("Failed to acquire histogram lock".into()))?
            .clear();
        Ok(())
    }
}

impl Default for PrometheusMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct HistogramStats {
    pub count: usize,
    pub sum: f64,
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub min: f64,
    pub max: f64,
}

impl Default for HistogramStats {
    fn default() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            mean: 0.0,
            p50: 0.0,
            p95: 0.0,
            p99: 0.0,
            min: 0.0,
            max: 0.0,
        }
    }
}

/// OpenTelemetry-compatible tracing system
#[derive(Debug, Clone)]
pub struct OpenTelemetryTracer {
    spans: Arc<RwLock<Vec<Span>>>,
    trace_id_counter: Arc<RwLock<u64>>,
}

#[derive(Debug, Clone)]
pub struct Span {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub name: String,
    pub start_time: SystemTime,
    pub end_time: Option<SystemTime>,
    pub attributes: HashMap<String, String>,
    pub events: Vec<SpanEvent>,
    pub status: SpanStatus,
}

#[derive(Debug, Clone)]
pub struct SpanEvent {
    pub timestamp: SystemTime,
    pub name: String,
    pub attributes: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SpanStatus {
    Unset,
    Ok,
    Error(String),
}

impl OpenTelemetryTracer {
    pub fn new() -> Self {
        Self {
            spans: Arc::new(RwLock::new(Vec::new())),
            trace_id_counter: Arc::new(RwLock::new(0)),
        }
    }

    /// Start a new span
    pub fn start_span(&self, name: &str, parent_span_id: Option<String>) -> Result<String, Error> {
        let mut counter = self
            .trace_id_counter
            .write()
            .map_err(|_| Error::Processing("Failed to acquire trace counter lock".into()))?;
        *counter += 1;
        let span_id = format!("span_{}", *counter);
        let trace_id = format!("trace_{}", *counter);

        let span = Span {
            trace_id,
            span_id: span_id.clone(),
            parent_span_id,
            name: name.to_string(),
            start_time: SystemTime::now(),
            end_time: None,
            attributes: HashMap::new(),
            events: Vec::new(),
            status: SpanStatus::Unset,
        };

        let mut spans = self
            .spans
            .write()
            .map_err(|_| Error::Processing("Failed to acquire spans lock".into()))?;
        spans.push(span);

        Ok(span_id)
    }

    /// End a span
    pub fn end_span(&self, span_id: &str, status: SpanStatus) -> Result<(), Error> {
        let mut spans = self
            .spans
            .write()
            .map_err(|_| Error::Processing("Failed to acquire spans lock".into()))?;

        if let Some(span) = spans.iter_mut().find(|s| s.span_id == span_id) {
            span.end_time = Some(SystemTime::now());
            span.status = status;
        }

        Ok(())
    }

    /// Add attributes to a span
    pub fn add_span_attribute(&self, span_id: &str, key: &str, value: &str) -> Result<(), Error> {
        let mut spans = self
            .spans
            .write()
            .map_err(|_| Error::Processing("Failed to acquire spans lock".into()))?;

        if let Some(span) = spans.iter_mut().find(|s| s.span_id == span_id) {
            span.attributes.insert(key.to_string(), value.to_string());
        }

        Ok(())
    }

    /// Add an event to a span
    pub fn add_span_event(&self, span_id: &str, event_name: &str) -> Result<(), Error> {
        let mut spans = self
            .spans
            .write()
            .map_err(|_| Error::Processing("Failed to acquire spans lock".into()))?;

        if let Some(span) = spans.iter_mut().find(|s| s.span_id == span_id) {
            span.events.push(SpanEvent {
                timestamp: SystemTime::now(),
                name: event_name.to_string(),
                attributes: HashMap::new(),
            });
        }

        Ok(())
    }

    /// Get all spans
    pub fn get_spans(&self) -> Result<Vec<Span>, Error> {
        let spans = self
            .spans
            .read()
            .map_err(|_| Error::Processing("Failed to read spans".into()))?;
        Ok(spans.clone())
    }

    /// Export spans in OTLP JSON format
    pub fn export_otlp_json(&self) -> Result<String, Error> {
        let spans = self.get_spans()?;

        let mut json = String::from("{\"resourceSpans\":[{\"scopeSpans\":[{\"spans\":[");

        for (i, span) in spans.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }

            let start_nanos = span
                .start_time
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos();

            let end_nanos = span
                .end_time
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .unwrap_or(Duration::ZERO)
                .as_nanos();

            json.push_str(&format!(
                "{{\"traceId\":\"{}\",\"spanId\":\"{}\",\"name\":\"{}\",\"startTimeUnixNano\":{},\"endTimeUnixNano\":{},\"status\":{{\"code\":{}}}}}",
                span.trace_id,
                span.span_id,
                span.name,
                start_nanos,
                end_nanos,
                match span.status {
                    SpanStatus::Unset => 0,
                    SpanStatus::Ok => 1,
                    SpanStatus::Error(_) => 2,
                }
            ));
        }

        json.push_str("]}]}]}");
        Ok(json)
    }

    /// Clear all spans
    pub fn clear(&self) -> Result<(), Error> {
        self.spans
            .write()
            .map_err(|_| Error::Processing("Failed to acquire spans lock".into()))?
            .clear();
        Ok(())
    }
}

impl Default for OpenTelemetryTracer {
    fn default() -> Self {
        Self::new()
    }
}

/// Production monitoring system combining metrics and tracing
#[derive(Debug, Clone)]
pub struct ProductionMonitor {
    metrics: PrometheusMetrics,
    tracer: OpenTelemetryTracer,
    start_time: Instant,
}

impl ProductionMonitor {
    pub fn new() -> Self {
        let monitor = Self {
            metrics: PrometheusMetrics::new(),
            tracer: OpenTelemetryTracer::new(),
            start_time: Instant::now(),
        };

        // Register standard metrics
        let _ = monitor.register_standard_metrics();

        monitor
    }

    fn register_standard_metrics(&self) -> Result<(), Error> {
        // Synthesis metrics
        self.metrics.register_metric(
            "singing_synthesis_requests_total",
            "Total number of synthesis requests",
            MetricType::Counter,
            HashMap::new(),
        )?;

        self.metrics.register_metric(
            "singing_synthesis_duration_seconds",
            "Duration of synthesis operations in seconds",
            MetricType::Histogram,
            HashMap::new(),
        )?;

        self.metrics.register_metric(
            "singing_synthesis_errors_total",
            "Total number of synthesis errors",
            MetricType::Counter,
            HashMap::new(),
        )?;

        // Performance metrics
        self.metrics.register_metric(
            "singing_cpu_usage_percent",
            "CPU usage percentage",
            MetricType::Gauge,
            HashMap::new(),
        )?;

        self.metrics.register_metric(
            "singing_memory_usage_bytes",
            "Memory usage in bytes",
            MetricType::Gauge,
            HashMap::new(),
        )?;

        self.metrics.register_metric(
            "singing_gpu_utilization_percent",
            "GPU utilization percentage",
            MetricType::Gauge,
            HashMap::new(),
        )?;

        // Quality metrics
        self.metrics.register_metric(
            "singing_quality_score",
            "Synthesis quality score (0-1)",
            MetricType::Gauge,
            HashMap::new(),
        )?;

        Ok(())
    }

    /// Record a synthesis request
    pub fn record_synthesis_request(&self) -> Result<(), Error> {
        self.metrics
            .increment_counter("singing_synthesis_requests_total", 1.0)
    }

    /// Record synthesis duration
    pub fn record_synthesis_duration(&self, duration_secs: f64) -> Result<(), Error> {
        self.metrics
            .observe_histogram("singing_synthesis_duration_seconds", duration_secs)
    }

    /// Record synthesis error
    pub fn record_synthesis_error(&self) -> Result<(), Error> {
        self.metrics
            .increment_counter("singing_synthesis_errors_total", 1.0)
    }

    /// Update CPU usage
    pub fn update_cpu_usage(&self, percent: f64) -> Result<(), Error> {
        self.metrics.set_gauge("singing_cpu_usage_percent", percent)
    }

    /// Update memory usage
    pub fn update_memory_usage(&self, bytes: f64) -> Result<(), Error> {
        self.metrics.set_gauge("singing_memory_usage_bytes", bytes)
    }

    /// Update GPU utilization
    pub fn update_gpu_utilization(&self, percent: f64) -> Result<(), Error> {
        self.metrics
            .set_gauge("singing_gpu_utilization_percent", percent)
    }

    /// Update quality score
    pub fn update_quality_score(&self, score: f64) -> Result<(), Error> {
        self.metrics.set_gauge("singing_quality_score", score)
    }

    /// Start a traced operation
    pub fn start_trace(&self, operation: &str) -> Result<String, Error> {
        self.tracer.start_span(operation, None)
    }

    /// End a traced operation
    pub fn end_trace(&self, span_id: &str, success: bool) -> Result<(), Error> {
        let status = if success {
            SpanStatus::Ok
        } else {
            SpanStatus::Error("Operation failed".to_string())
        };
        self.tracer.end_span(span_id, status)
    }

    /// Get metrics in Prometheus format
    pub fn export_metrics(&self) -> Result<String, Error> {
        self.metrics.export_prometheus_format()
    }

    /// Get traces in OTLP format
    pub fn export_traces(&self) -> Result<String, Error> {
        self.tracer.export_otlp_json()
    }

    /// Get uptime in seconds
    pub fn uptime_seconds(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Get comprehensive monitoring stats
    pub fn get_stats(&self) -> Result<MonitoringStats, Error> {
        Ok(MonitoringStats {
            uptime_seconds: self.uptime_seconds(),
            total_requests: self
                .metrics
                .get_counter("singing_synthesis_requests_total")?,
            total_errors: self.metrics.get_counter("singing_synthesis_errors_total")?,
            cpu_usage_percent: self.metrics.get_gauge("singing_cpu_usage_percent")?,
            memory_usage_bytes: self.metrics.get_gauge("singing_memory_usage_bytes")?,
            gpu_utilization_percent: self.metrics.get_gauge("singing_gpu_utilization_percent")?,
            quality_score: self.metrics.get_gauge("singing_quality_score")?,
            latency_stats: self
                .metrics
                .get_histogram_stats("singing_synthesis_duration_seconds")?,
        })
    }

    /// Access metrics collector
    pub fn metrics(&self) -> &PrometheusMetrics {
        &self.metrics
    }

    /// Access tracer
    pub fn tracer(&self) -> &OpenTelemetryTracer {
        &self.tracer
    }
}

impl Default for ProductionMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct MonitoringStats {
    pub uptime_seconds: f64,
    pub total_requests: f64,
    pub total_errors: f64,
    pub cpu_usage_percent: f64,
    pub memory_usage_bytes: f64,
    pub gpu_utilization_percent: f64,
    pub quality_score: f64,
    pub latency_stats: HistogramStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prometheus_metrics_counter() {
        let metrics = PrometheusMetrics::new();

        metrics.increment_counter("test_counter", 1.0).unwrap();
        metrics.increment_counter("test_counter", 2.0).unwrap();

        assert_eq!(metrics.get_counter("test_counter").unwrap(), 3.0);
    }

    #[test]
    fn test_prometheus_metrics_gauge() {
        let metrics = PrometheusMetrics::new();

        metrics.set_gauge("test_gauge", 42.0).unwrap();
        assert_eq!(metrics.get_gauge("test_gauge").unwrap(), 42.0);

        metrics.set_gauge("test_gauge", 100.0).unwrap();
        assert_eq!(metrics.get_gauge("test_gauge").unwrap(), 100.0);
    }

    #[test]
    fn test_prometheus_metrics_histogram() {
        let metrics = PrometheusMetrics::new();

        metrics.observe_histogram("test_histogram", 0.1).unwrap();
        metrics.observe_histogram("test_histogram", 0.5).unwrap();
        metrics.observe_histogram("test_histogram", 1.0).unwrap();

        let stats = metrics.get_histogram_stats("test_histogram").unwrap();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.min, 0.1);
        assert_eq!(stats.max, 1.0);
    }

    #[test]
    fn test_prometheus_export_format() {
        let metrics = PrometheusMetrics::new();

        metrics
            .register_metric(
                "test_counter",
                "Test counter metric",
                MetricType::Counter,
                HashMap::new(),
            )
            .unwrap();

        metrics.increment_counter("test_counter", 5.0).unwrap();

        let export = metrics.export_prometheus_format().unwrap();
        assert!(export.contains("# HELP test_counter Test counter metric"));
        assert!(export.contains("# TYPE test_counter counter"));
        assert!(export.contains("test_counter 5"));
    }

    #[test]
    fn test_opentelemetry_span_lifecycle() {
        let tracer = OpenTelemetryTracer::new();

        let span_id = tracer.start_span("test_operation", None).unwrap();
        tracer.add_span_attribute(&span_id, "key", "value").unwrap();
        tracer
            .add_span_event(&span_id, "processing_started")
            .unwrap();
        tracer.end_span(&span_id, SpanStatus::Ok).unwrap();

        let spans = tracer.get_spans().unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].name, "test_operation");
        assert_eq!(spans[0].status, SpanStatus::Ok);
        assert!(spans[0].end_time.is_some());
    }

    #[test]
    fn test_opentelemetry_otlp_export() {
        let tracer = OpenTelemetryTracer::new();

        let span_id = tracer.start_span("test_export", None).unwrap();
        tracer.end_span(&span_id, SpanStatus::Ok).unwrap();

        let json = tracer.export_otlp_json().unwrap();
        assert!(json.contains("test_export"));
        assert!(json.contains("traceId"));
        assert!(json.contains("spanId"));
    }

    #[test]
    fn test_production_monitor_synthesis_tracking() {
        let monitor = ProductionMonitor::new();

        monitor.record_synthesis_request().unwrap();
        monitor.record_synthesis_request().unwrap();
        monitor.record_synthesis_duration(0.5).unwrap();

        let stats = monitor.get_stats().unwrap();
        assert_eq!(stats.total_requests, 2.0);
        assert_eq!(stats.latency_stats.count, 1);
    }

    #[test]
    fn test_production_monitor_resource_tracking() {
        let monitor = ProductionMonitor::new();

        monitor.update_cpu_usage(75.0).unwrap();
        monitor
            .update_memory_usage(1024.0 * 1024.0 * 512.0)
            .unwrap();
        monitor.update_gpu_utilization(80.0).unwrap();

        // Record at least one synthesis duration to create the histogram
        monitor.record_synthesis_duration(0.1).unwrap();

        let stats = monitor.get_stats().unwrap();
        assert_eq!(stats.cpu_usage_percent, 75.0);
        assert_eq!(stats.memory_usage_bytes, 1024.0 * 1024.0 * 512.0);
        assert_eq!(stats.gpu_utilization_percent, 80.0);
    }

    #[test]
    fn test_production_monitor_tracing() {
        let monitor = ProductionMonitor::new();

        let span_id = monitor.start_trace("synthesis_operation").unwrap();
        monitor.end_trace(&span_id, true).unwrap();

        let traces = monitor.export_traces().unwrap();
        assert!(traces.contains("synthesis_operation"));
    }

    #[test]
    fn test_production_monitor_uptime() {
        let monitor = ProductionMonitor::new();
        std::thread::sleep(Duration::from_millis(100));

        let uptime = monitor.uptime_seconds();
        assert!(uptime >= 0.1);
    }

    #[test]
    fn test_histogram_percentiles() {
        let metrics = PrometheusMetrics::new();

        for i in 1..=100 {
            metrics
                .observe_histogram("percentile_test", i as f64 / 100.0)
                .unwrap();
        }

        let stats = metrics.get_histogram_stats("percentile_test").unwrap();
        assert_eq!(stats.count, 100);
        assert!(stats.p50 > 0.4 && stats.p50 < 0.6);
        assert!(stats.p95 > 0.9);
        assert!(stats.p99 > 0.95);
    }

    #[test]
    fn test_metrics_reset() {
        let metrics = PrometheusMetrics::new();

        metrics.increment_counter("test_counter", 10.0).unwrap();
        metrics.set_gauge("test_gauge", 20.0).unwrap();

        metrics.reset().unwrap();

        assert_eq!(metrics.get_counter("test_counter").unwrap(), 0.0);
        assert_eq!(metrics.get_gauge("test_gauge").unwrap(), 0.0);
    }
}
