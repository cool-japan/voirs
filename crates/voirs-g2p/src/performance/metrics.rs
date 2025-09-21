//! Real-time metrics export for production monitoring

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::performance::batch::{BatchPhonemeProcessor, BatchProcessingStats};
use crate::performance::cache::{CacheStats, G2pCache};
use crate::performance::compression::{AdaptiveCompressionManager, CompressionStats};
use crate::performance::monitoring::{PerformanceMetric, PerformanceMonitor};
use crate::{Phoneme, Result};

/// Real-time metrics export formats
#[derive(Debug, Clone, PartialEq)]
pub enum MetricsFormat {
    /// JSON format for REST APIs and dashboards
    Json,
    /// Prometheus-style metrics for monitoring systems
    Prometheus,
    /// CSV format for data analysis
    Csv,
    /// Human-readable format for debugging
    HumanReadable,
}

/// Serializable performance metric for export
#[derive(Debug, Clone, Serialize)]
pub struct SerializablePerformanceMetric {
    pub name: String,
    pub total_time_ms: u64,
    pub call_count: u64,
    pub min_time_ms: u64,
    pub max_time_ms: u64,
    pub avg_time_ms: u64,
}

impl From<&PerformanceMetric> for SerializablePerformanceMetric {
    fn from(metric: &PerformanceMetric) -> Self {
        Self {
            name: metric.name.clone(),
            total_time_ms: metric.total_time.as_millis() as u64,
            call_count: metric.call_count,
            min_time_ms: metric.min_time.as_millis() as u64,
            max_time_ms: metric.max_time.as_millis() as u64,
            avg_time_ms: metric.avg_time.as_millis() as u64,
        }
    }
}

/// Comprehensive metrics snapshot for export
#[derive(Debug, Clone, Serialize)]
pub struct MetricsSnapshot {
    pub timestamp: u64,
    pub cache_metrics: CacheStats,
    pub performance_metrics: Vec<SerializablePerformanceMetric>,
    pub batch_processing_stats: Option<BatchProcessingStats>,
    pub compression_stats: Option<CompressionStats>,
}

/// Real-time metrics exporter for production monitoring
pub struct MetricsExporter {
    performance_monitor: Arc<PerformanceMonitor>,
    cache: Option<Arc<G2pCache<String, Vec<Phoneme>>>>,
    batch_processor: Option<Arc<BatchPhonemeProcessor>>,
    compression_manager: Option<Arc<AdaptiveCompressionManager>>,
}

impl MetricsExporter {
    /// Create new metrics exporter with performance monitor
    pub fn new(performance_monitor: Arc<PerformanceMonitor>) -> Self {
        Self {
            performance_monitor,
            cache: None,
            batch_processor: None,
            compression_manager: None,
        }
    }

    /// Add cache monitoring
    pub fn with_cache(mut self, cache: Arc<G2pCache<String, Vec<Phoneme>>>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Add batch processor monitoring
    pub fn with_batch_processor(mut self, processor: Arc<BatchPhonemeProcessor>) -> Self {
        self.batch_processor = Some(processor);
        self
    }

    /// Add compression manager monitoring
    pub fn with_compression_manager(mut self, manager: Arc<AdaptiveCompressionManager>) -> Self {
        self.compression_manager = Some(manager);
        self
    }

    /// Capture current metrics snapshot
    pub fn capture_snapshot(&self) -> MetricsSnapshot {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let cache_metrics = self.cache.as_ref().map(|c| c.stats()).unwrap_or_default();

        let performance_metrics = self
            .performance_monitor
            .get_metrics()
            .iter()
            .map(|m| m.into())
            .collect();

        let batch_processing_stats = self.batch_processor.as_ref().map(|p| p.get_stats());

        let compression_stats = self.compression_manager.as_ref().map(|m| m.stats());

        MetricsSnapshot {
            timestamp,
            cache_metrics,
            performance_metrics,
            batch_processing_stats,
            compression_stats,
        }
    }

    /// Export metrics in specified format
    pub fn export_metrics(&self, format: MetricsFormat) -> Result<String> {
        let snapshot = self.capture_snapshot();

        match format {
            MetricsFormat::Json => self.export_json(&snapshot),
            MetricsFormat::Prometheus => self.export_prometheus(&snapshot),
            MetricsFormat::Csv => self.export_csv(&snapshot),
            MetricsFormat::HumanReadable => self.export_human_readable(&snapshot),
        }
    }

    /// Export as JSON
    fn export_json(&self, snapshot: &MetricsSnapshot) -> Result<String> {
        serde_json::to_string_pretty(snapshot)
            .map_err(|e| crate::G2pError::InvalidInput(format!("JSON export failed: {e}")))
    }

    /// Export as Prometheus metrics
    fn export_prometheus(&self, snapshot: &MetricsSnapshot) -> Result<String> {
        let mut output = String::new();

        // Cache metrics
        output.push_str(&format!(
            "# HELP g2p_cache_hits_total Total cache hits\n# TYPE g2p_cache_hits_total counter\ng2p_cache_hits_total {}\n",
            snapshot.cache_metrics.hits
        ));
        output.push_str(&format!(
            "# HELP g2p_cache_misses_total Total cache misses\n# TYPE g2p_cache_misses_total counter\ng2p_cache_misses_total {}\n",
            snapshot.cache_metrics.misses
        ));
        output.push_str(&format!(
            "# HELP g2p_cache_hit_rate Cache hit rate ratio\n# TYPE g2p_cache_hit_rate gauge\ng2p_cache_hit_rate {:.4}\n",
            snapshot.cache_metrics.hit_rate()
        ));

        // Performance metrics
        for metric in &snapshot.performance_metrics {
            output.push_str(&format!(
                "# HELP g2p_operation_duration_seconds Operation duration in seconds\n# TYPE g2p_operation_duration_seconds histogram\ng2p_operation_duration_seconds{{operation=\"{}\"}} {:.6}\n",
                metric.name,
                metric.avg_time_ms as f64 / 1000.0
            ));
            output.push_str(&format!(
                "# HELP g2p_operation_calls_total Total operation calls\n# TYPE g2p_operation_calls_total counter\ng2p_operation_calls_total{{operation=\"{}\"}} {}\n",
                metric.name,
                metric.call_count
            ));
        }

        // Batch processing stats
        if let Some(batch_stats) = &snapshot.batch_processing_stats {
            output.push_str(&format!(
                "# HELP g2p_batches_processed_total Total batches processed\n# TYPE g2p_batches_processed_total counter\ng2p_batches_processed_total {}\n",
                batch_stats.batches_processed
            ));
            output.push_str(&format!(
                "# HELP g2p_phonemes_processed_total Total phonemes processed\n# TYPE g2p_phonemes_processed_total counter\ng2p_phonemes_processed_total {}\n",
                batch_stats.phonemes_processed
            ));
        }

        Ok(output)
    }

    /// Export as CSV
    fn export_csv(&self, snapshot: &MetricsSnapshot) -> Result<String> {
        let mut output = String::new();
        output.push_str("timestamp,metric_type,metric_name,value\n");

        // Cache metrics
        output.push_str(&format!(
            "{},cache,hits,{}\n",
            snapshot.timestamp, snapshot.cache_metrics.hits
        ));
        output.push_str(&format!(
            "{},cache,misses,{}\n",
            snapshot.timestamp, snapshot.cache_metrics.misses
        ));
        output.push_str(&format!(
            "{},cache,hit_rate,{:.4}\n",
            snapshot.timestamp,
            snapshot.cache_metrics.hit_rate()
        ));

        // Performance metrics
        for metric in &snapshot.performance_metrics {
            output.push_str(&format!(
                "{},performance,{}_avg_time_ms,{}\n",
                snapshot.timestamp, metric.name, metric.avg_time_ms
            ));
            output.push_str(&format!(
                "{},performance,{}_call_count,{}\n",
                snapshot.timestamp, metric.name, metric.call_count
            ));
        }

        Ok(output)
    }

    /// Export in human-readable format
    fn export_human_readable(&self, snapshot: &MetricsSnapshot) -> Result<String> {
        let mut output = String::new();
        output.push_str(&format!(
            "=== G2P Performance Metrics (timestamp: {}) ===\n\n",
            snapshot.timestamp
        ));

        // Cache metrics
        output.push_str("Cache Performance:\n");
        output.push_str(&format!("  Hits: {}\n", snapshot.cache_metrics.hits));
        output.push_str(&format!("  Misses: {}\n", snapshot.cache_metrics.misses));
        output.push_str(&format!(
            "  Hit Rate: {:.2}%\n",
            snapshot.cache_metrics.hit_rate() * 100.0
        ));
        output.push_str(&format!(
            "  Total Size: {}\n\n",
            snapshot.cache_metrics.total_size
        ));

        // Performance metrics
        if !snapshot.performance_metrics.is_empty() {
            output.push_str("Operation Performance:\n");
            for metric in &snapshot.performance_metrics {
                output.push_str(&format!("  {}:\n", metric.name));
                output.push_str(&format!("    Calls: {}\n", metric.call_count));
                output.push_str(&format!("    Avg Time: {}ms\n", metric.avg_time_ms));
                output.push_str(&format!("    Min Time: {}ms\n", metric.min_time_ms));
                output.push_str(&format!("    Max Time: {}ms\n", metric.max_time_ms));
            }
            output.push('\n');
        }

        // Batch processing stats
        if let Some(batch_stats) = &snapshot.batch_processing_stats {
            output.push_str("Batch Processing:\n");
            output.push_str(&format!(
                "  Batches Processed: {}\n",
                batch_stats.batches_processed
            ));
            output.push_str(&format!(
                "  Phonemes Processed: {}\n",
                batch_stats.phonemes_processed
            ));
            output.push_str(&format!(
                "  Avg Phonemes/Batch: {:.1}\n",
                batch_stats.phonemes_processed as f64 / batch_stats.batches_processed.max(1) as f64
            ));
            output.push_str(&format!(
                "  Avg Batch Time: {:.1}ms\n",
                batch_stats.avg_batch_time_ms
            ));
            output.push_str(&format!(
                "  Memory Efficiency: {:.2}\n",
                batch_stats.memory_efficiency
            ));
        }

        Ok(output)
    }

    /// Start periodic metrics export to a callback function
    pub async fn start_periodic_export<F>(
        &self,
        interval: Duration,
        format: MetricsFormat,
        mut callback: F,
    ) where
        F: FnMut(String) + Send + 'static,
    {
        let mut interval_timer = tokio::time::interval(interval);
        loop {
            interval_timer.tick().await;
            if let Ok(metrics) = self.export_metrics(format.clone()) {
                callback(metrics);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance::cache::G2pCache;
    use crate::Phoneme;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn test_metrics_exporter_creation() {
        let monitor = Arc::new(PerformanceMonitor::new(true));
        let exporter = MetricsExporter::new(monitor);

        let snapshot = exporter.capture_snapshot();
        assert!(snapshot.timestamp > 0);
        assert_eq!(snapshot.performance_metrics.len(), 0);
    }

    #[test]
    fn test_json_export() {
        let monitor = Arc::new(PerformanceMonitor::new(true));
        monitor.record_timing("test_op", Duration::from_millis(10));

        let exporter = MetricsExporter::new(monitor);
        let json_result = exporter.export_metrics(MetricsFormat::Json);

        assert!(json_result.is_ok());
        let json = json_result.unwrap();
        assert!(json.contains("timestamp"));
        assert!(json.contains("test_op"));
    }

    #[test]
    fn test_prometheus_export() {
        let monitor = Arc::new(PerformanceMonitor::new(true));
        monitor.record_timing("test_operation", Duration::from_millis(50));

        let cache: G2pCache<String, Vec<Phoneme>> = G2pCache::new(10);
        cache.insert("test".to_string(), vec![Phoneme::new("test")]);
        let _value = cache.get(&"test".to_string());

        let exporter = MetricsExporter::new(monitor).with_cache(Arc::new(cache));

        let prometheus_result = exporter.export_metrics(MetricsFormat::Prometheus);
        assert!(prometheus_result.is_ok());

        let prometheus = prometheus_result.unwrap();
        assert!(prometheus.contains("g2p_cache_hits_total"));
        assert!(prometheus.contains("g2p_operation_duration_seconds"));
        assert!(prometheus.contains("test_operation"));
    }

    #[test]
    fn test_csv_export() {
        let monitor = Arc::new(PerformanceMonitor::new(true));
        monitor.record_timing("csv_test", Duration::from_millis(25));

        let exporter = MetricsExporter::new(monitor);
        let csv_result = exporter.export_metrics(MetricsFormat::Csv);

        assert!(csv_result.is_ok());
        let csv = csv_result.unwrap();
        assert!(csv.contains("timestamp,metric_type,metric_name,value"));
        assert!(csv.contains("performance,csv_test"));
    }

    #[test]
    fn test_human_readable_export() {
        let monitor = Arc::new(PerformanceMonitor::new(true));
        monitor.record_timing("readable_test", Duration::from_millis(15));

        let exporter = MetricsExporter::new(monitor);
        let readable_result = exporter.export_metrics(MetricsFormat::HumanReadable);

        assert!(readable_result.is_ok());
        let readable = readable_result.unwrap();
        assert!(readable.contains("G2P Performance Metrics"));
        assert!(readable.contains("Cache Performance:"));
        assert!(readable.contains("Operation Performance:"));
        assert!(readable.contains("readable_test"));
    }

    #[test]
    fn test_metrics_format_clone() {
        let format = MetricsFormat::Json;
        let cloned = format.clone();
        assert_eq!(format, cloned);
    }

    #[test]
    fn test_serializable_performance_metric() {
        let monitor = Arc::new(PerformanceMonitor::new(true));
        monitor.record_timing("test_metric", Duration::from_millis(100));

        let metric = monitor.get_metric("test_metric").unwrap();
        let serializable: SerializablePerformanceMetric = (&metric).into();

        assert_eq!(serializable.name, "test_metric");
        assert_eq!(serializable.call_count, 1);
        assert_eq!(serializable.avg_time_ms, 100);
    }
}
