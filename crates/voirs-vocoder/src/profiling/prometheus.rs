//! Prometheus Metrics Exporter
//!
//! Provides integration with Prometheus for production monitoring and observability.
//! Exports vocoder performance metrics in Prometheus exposition format.

use super::{AdvancedProfiler, ProfilerMetrics};
use std::fmt::Write;

/// Prometheus metrics exporter for vocoder performance
///
/// This struct provides methods to export profiling metrics in Prometheus format,
/// enabling integration with standard monitoring infrastructure.
///
/// # Examples
///
/// ```rust
/// use voirs_vocoder::profiling::{AdvancedProfiler, PrometheusExporter};
///
/// let profiler = AdvancedProfiler::new();
/// let exporter = PrometheusExporter::new("voirs_vocoder", &profiler);
///
/// // After some operations...
/// let metrics = exporter.export_metrics();
/// println!("{}", metrics);
/// ```
pub struct PrometheusExporter<'a> {
    /// Namespace prefix for all metrics
    namespace: &'a str,
    /// Reference to the profiler
    profiler: &'a AdvancedProfiler,
    /// Optional subsystem name
    subsystem: Option<&'a str>,
}

impl<'a> PrometheusExporter<'a> {
    /// Create a new Prometheus exporter
    ///
    /// # Arguments
    ///
    /// * `namespace` - Metric namespace prefix (e.g., "voirs_vocoder")
    /// * `profiler` - Reference to the profiler to export metrics from
    pub fn new(namespace: &'a str, profiler: &'a AdvancedProfiler) -> Self {
        Self {
            namespace,
            profiler,
            subsystem: None,
        }
    }

    /// Create a new Prometheus exporter with subsystem
    ///
    /// # Arguments
    ///
    /// * `namespace` - Metric namespace prefix (e.g., "voirs")
    /// * `subsystem` - Subsystem name (e.g., "vocoder")
    /// * `profiler` - Reference to the profiler to export metrics from
    pub fn with_subsystem(
        namespace: &'a str,
        subsystem: &'a str,
        profiler: &'a AdvancedProfiler,
    ) -> Self {
        Self {
            namespace,
            profiler,
            subsystem: Some(subsystem),
        }
    }

    /// Get the full metric name with namespace and optional subsystem
    fn metric_name(&self, name: &str) -> String {
        if let Some(subsystem) = self.subsystem {
            format!("{}_{}_{}", self.namespace, subsystem, name)
        } else {
            format!("{}_{}", self.namespace, name)
        }
    }

    /// Export all metrics in Prometheus exposition format
    ///
    /// Returns a string containing all metrics in the format expected by Prometheus.
    /// This can be served via an HTTP endpoint for Prometheus to scrape.
    pub fn export_metrics(&self) -> String {
        let metrics = self.profiler.get_metrics();
        let mut output = String::with_capacity(4096);

        // Total operations counter
        self.write_counter(
            &mut output,
            "operations_total",
            "Total number of vocoding operations",
            metrics.total_operations as f64,
        );

        // Processing time summary
        self.write_gauge(
            &mut output,
            "processing_time_seconds_total",
            "Total processing time in seconds",
            metrics.total_time.as_secs_f64(),
        );

        // Latency metrics (in milliseconds for better readability)
        if let Some(min_latency) = metrics.min_latency {
            self.write_gauge(
                &mut output,
                "latency_min_ms",
                "Minimum observed latency in milliseconds",
                min_latency.as_secs_f64() * 1000.0,
            );
        }

        if let Some(max_latency) = metrics.max_latency {
            self.write_gauge(
                &mut output,
                "latency_max_ms",
                "Maximum observed latency in milliseconds",
                max_latency.as_secs_f64() * 1000.0,
            );
        }

        self.write_gauge(
            &mut output,
            "latency_avg_ms",
            "Average latency in milliseconds",
            metrics.avg_latency.as_secs_f64() * 1000.0,
        );

        // Latency percentiles (Summary metric type)
        self.write_summary_header(&mut output, "latency_ms", "Operation latency distribution");
        self.write_summary_quantile(
            &mut output,
            "latency_ms",
            0.5,
            metrics.p50_latency.as_secs_f64() * 1000.0,
        );
        self.write_summary_quantile(
            &mut output,
            "latency_ms",
            0.95,
            metrics.p95_latency.as_secs_f64() * 1000.0,
        );
        self.write_summary_quantile(
            &mut output,
            "latency_ms",
            0.99,
            metrics.p99_latency.as_secs_f64() * 1000.0,
        );
        self.write_summary_count(&mut output, "latency_ms", metrics.total_operations);
        self.write_summary_sum(
            &mut output,
            "latency_ms",
            metrics.total_time.as_secs_f64() * 1000.0,
        );

        // Real-time factor (RTF) metrics
        self.write_gauge(
            &mut output,
            "rtf_min",
            "Minimum real-time factor",
            metrics.rtf_stats.min_rtf as f64,
        );

        self.write_gauge(
            &mut output,
            "rtf_max",
            "Maximum real-time factor",
            metrics.rtf_stats.max_rtf as f64,
        );

        self.write_gauge(
            &mut output,
            "rtf_avg",
            "Average real-time factor",
            metrics.rtf_stats.avg_rtf as f64,
        );

        self.write_gauge(
            &mut output,
            "realtime_percentage",
            "Percentage of operations meeting real-time constraints",
            metrics.rtf_stats.realtime_percentage as f64,
        );

        // Throughput metrics
        self.write_gauge(
            &mut output,
            "audio_seconds_total",
            "Total audio seconds processed",
            metrics.throughput_stats.total_audio_seconds as f64,
        );

        self.write_gauge(
            &mut output,
            "frames_total",
            "Total frames processed",
            metrics.throughput_stats.total_frames as f64,
        );

        self.write_gauge(
            &mut output,
            "frames_per_second_avg",
            "Average frames per second throughput",
            metrics.throughput_stats.avg_frames_per_second as f64,
        );

        // Latency breakdown by stage
        self.write_gauge(
            &mut output,
            "stage_preprocessing_ms",
            "Average preprocessing stage latency in milliseconds",
            metrics.latency_breakdown.preprocessing.as_secs_f64() * 1000.0,
        );

        self.write_gauge(
            &mut output,
            "stage_inference_ms",
            "Average inference stage latency in milliseconds",
            metrics.latency_breakdown.inference.as_secs_f64() * 1000.0,
        );

        self.write_gauge(
            &mut output,
            "stage_postprocessing_ms",
            "Average postprocessing stage latency in milliseconds",
            metrics.latency_breakdown.postprocessing.as_secs_f64() * 1000.0,
        );

        // Memory statistics (if enabled)
        if let Some(ref memory_stats) = metrics.memory_stats {
            self.write_gauge(
                &mut output,
                "memory_peak_bytes",
                "Peak memory usage in bytes",
                memory_stats.peak_memory_bytes as f64,
            );

            self.write_gauge(
                &mut output,
                "memory_avg_bytes",
                "Average memory usage in bytes",
                memory_stats.avg_memory_bytes as f64,
            );

            self.write_gauge(
                &mut output,
                "memory_current_bytes",
                "Current memory usage in bytes",
                memory_stats.current_memory_bytes as f64,
            );
        }

        output
    }

    /// Write a counter metric
    fn write_counter(&self, output: &mut String, name: &str, help: &str, value: f64) {
        let metric_name = self.metric_name(name);
        let _ = writeln!(output, "# HELP {} {}", metric_name, help);
        let _ = writeln!(output, "# TYPE {} counter", metric_name);
        let _ = writeln!(output, "{} {}", metric_name, value);
    }

    /// Write a gauge metric
    fn write_gauge(&self, output: &mut String, name: &str, help: &str, value: f64) {
        let metric_name = self.metric_name(name);
        let _ = writeln!(output, "# HELP {} {}", metric_name, help);
        let _ = writeln!(output, "# TYPE {} gauge", metric_name);
        let _ = writeln!(output, "{} {}", metric_name, value);
    }

    /// Write summary metric header
    fn write_summary_header(&self, output: &mut String, name: &str, help: &str) {
        let metric_name = self.metric_name(name);
        let _ = writeln!(output, "# HELP {} {}", metric_name, help);
        let _ = writeln!(output, "# TYPE {} summary", metric_name);
    }

    /// Write summary quantile
    fn write_summary_quantile(&self, output: &mut String, name: &str, quantile: f64, value: f64) {
        let metric_name = self.metric_name(name);
        let _ = writeln!(
            output,
            "{}{{quantile=\"{}\"}} {}",
            metric_name, quantile, value
        );
    }

    /// Write summary count
    fn write_summary_count(&self, output: &mut String, name: &str, count: u64) {
        let metric_name = self.metric_name(name);
        let _ = writeln!(output, "{}_count {}", metric_name, count);
    }

    /// Write summary sum
    fn write_summary_sum(&self, output: &mut String, name: &str, sum: f64) {
        let metric_name = self.metric_name(name);
        let _ = writeln!(output, "{}_sum {}", metric_name, sum);
    }

    /// Export metrics with custom labels
    ///
    /// # Arguments
    ///
    /// * `labels` - Additional labels to add to all metrics (e.g., `vec![("model", "hifigan")]`)
    pub fn export_metrics_with_labels(&self, labels: &[(&str, &str)]) -> String {
        let metrics = self.profiler.get_metrics();
        let mut output = String::with_capacity(4096);

        let label_str = if labels.is_empty() {
            String::new()
        } else {
            let pairs: Vec<String> = labels
                .iter()
                .map(|(k, v)| format!("{}=\"{}\"", k, v))
                .collect();
            format!("{{{}}}", pairs.join(","))
        };

        // Total operations counter with labels
        self.write_labeled_metric(
            &mut output,
            "operations_total",
            "Total number of vocoding operations",
            "counter",
            &label_str,
            metrics.total_operations as f64,
        );

        // Add other metrics with labels...
        self.write_labeled_metric(
            &mut output,
            "latency_avg_ms",
            "Average latency in milliseconds",
            "gauge",
            &label_str,
            metrics.avg_latency.as_secs_f64() * 1000.0,
        );

        self.write_labeled_metric(
            &mut output,
            "rtf_avg",
            "Average real-time factor",
            "gauge",
            &label_str,
            metrics.rtf_stats.avg_rtf as f64,
        );

        self.write_labeled_metric(
            &mut output,
            "frames_per_second_avg",
            "Average frames per second throughput",
            "gauge",
            &label_str,
            metrics.throughput_stats.avg_frames_per_second as f64,
        );

        output
    }

    /// Write a metric with custom labels
    fn write_labeled_metric(
        &self,
        output: &mut String,
        name: &str,
        help: &str,
        metric_type: &str,
        labels: &str,
        value: f64,
    ) {
        let metric_name = self.metric_name(name);
        let _ = writeln!(output, "# HELP {} {}", metric_name, help);
        let _ = writeln!(output, "# TYPE {} {}", metric_name, metric_type);
        let _ = writeln!(output, "{}{} {}", metric_name, labels, value);
    }
}

/// Helper function to start a Prometheus metrics server
///
/// This is a convenience function for quickly setting up an HTTP endpoint
/// that serves Prometheus metrics. In production, you would typically integrate
/// this with your existing HTTP server.
///
/// # Arguments
///
/// * `profiler` - Reference to the profiler
/// * `port` - Port to listen on (default: 9090)
///
/// # Examples
///
/// ```no_run
/// use voirs_vocoder::profiling::AdvancedProfiler;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let profiler = AdvancedProfiler::new();
///
/// // Start metrics server in background
/// // serve_prometheus_metrics(&profiler, 9090).await?;
/// # Ok(())
/// # }
/// ```
///
/// Then configure Prometheus to scrape `http://localhost:9090/metrics`
#[allow(dead_code)]
pub fn prometheus_metrics_text(profiler: &AdvancedProfiler) -> String {
    let exporter = PrometheusExporter::new("voirs_vocoder", profiler);
    exporter.export_metrics()
}

/// Helper function to export metrics with a specific namespace and subsystem
#[allow(dead_code)]
pub fn prometheus_metrics_with_subsystem(
    namespace: &str,
    subsystem: &str,
    profiler: &AdvancedProfiler,
) -> String {
    let exporter = PrometheusExporter::with_subsystem(namespace, subsystem, profiler);
    exporter.export_metrics()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiling::AdvancedProfiler;

    #[test]
    fn test_prometheus_exporter_creation() {
        let profiler = AdvancedProfiler::new();
        let exporter = PrometheusExporter::new("test", &profiler);

        assert_eq!(exporter.namespace, "test");
        assert!(exporter.subsystem.is_none());
    }

    #[test]
    fn test_prometheus_exporter_with_subsystem() {
        let profiler = AdvancedProfiler::new();
        let exporter = PrometheusExporter::with_subsystem("voirs", "vocoder", &profiler);

        assert_eq!(exporter.namespace, "voirs");
        assert_eq!(exporter.subsystem, Some("vocoder"));
    }

    #[test]
    fn test_metric_name_without_subsystem() {
        let profiler = AdvancedProfiler::new();
        let exporter = PrometheusExporter::new("test", &profiler);

        assert_eq!(exporter.metric_name("operations"), "test_operations");
    }

    #[test]
    fn test_metric_name_with_subsystem() {
        let profiler = AdvancedProfiler::new();
        let exporter = PrometheusExporter::with_subsystem("voirs", "vocoder", &profiler);

        assert_eq!(
            exporter.metric_name("operations"),
            "voirs_vocoder_operations"
        );
    }

    #[test]
    fn test_export_metrics_format() {
        let profiler = AdvancedProfiler::new();
        let exporter = PrometheusExporter::new("test", &profiler);

        let metrics = exporter.export_metrics();

        // Verify Prometheus format
        assert!(metrics.contains("# HELP"));
        assert!(metrics.contains("# TYPE"));
        assert!(metrics.contains("test_operations_total"));
        assert!(metrics.contains("test_latency_avg_ms"));
        assert!(metrics.contains("test_rtf_avg"));
    }

    #[test]
    fn test_export_metrics_with_labels() {
        let profiler = AdvancedProfiler::new();
        let exporter = PrometheusExporter::new("test", &profiler);

        let labels = vec![("model", "hifigan"), ("backend", "candle")];
        let metrics = exporter.export_metrics_with_labels(&labels);

        // Verify labels are present
        assert!(metrics.contains("model=\"hifigan\""));
        assert!(metrics.contains("backend=\"candle\""));
    }

    #[test]
    fn test_prometheus_metrics_text() {
        let profiler = AdvancedProfiler::new();
        let metrics = prometheus_metrics_text(&profiler);

        assert!(metrics.contains("voirs_vocoder_"));
        assert!(!metrics.is_empty());
    }

    #[test]
    fn test_prometheus_metrics_with_subsystem_helper() {
        let profiler = AdvancedProfiler::new();
        let metrics = prometheus_metrics_with_subsystem("voirs", "vocoder", &profiler);

        assert!(metrics.contains("voirs_vocoder_"));
        assert!(!metrics.is_empty());
    }
}
