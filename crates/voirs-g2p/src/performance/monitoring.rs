//! Performance monitoring and profiling integration

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Performance monitoring and profiling integration
pub struct PerformanceMonitor {
    metrics: Arc<Mutex<HashMap<String, PerformanceMetric>>>,
    profiling_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetric {
    pub name: String,
    pub total_time: Duration,
    pub call_count: u64,
    pub min_time: Duration,
    pub max_time: Duration,
    pub avg_time: Duration,
    pub last_updated: Instant,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new(profiling_enabled: bool) -> Self {
        Self {
            metrics: Arc::new(Mutex::new(HashMap::new())),
            profiling_enabled,
        }
    }

    /// Record a timing measurement
    pub fn record_timing(&self, name: &str, duration: Duration) {
        if !self.profiling_enabled {
            return;
        }

        let mut metrics = self.metrics.lock().unwrap();
        let metric = metrics
            .entry(name.to_string())
            .or_insert_with(|| PerformanceMetric {
                name: name.to_string(),
                total_time: Duration::ZERO,
                call_count: 0,
                min_time: Duration::MAX,
                max_time: Duration::ZERO,
                avg_time: Duration::ZERO,
                last_updated: Instant::now(),
            });

        metric.total_time += duration;
        metric.call_count += 1;
        metric.min_time = metric.min_time.min(duration);
        metric.max_time = metric.max_time.max(duration);
        metric.avg_time = metric.total_time / metric.call_count as u32;
        metric.last_updated = Instant::now();
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> Vec<PerformanceMetric> {
        self.metrics.lock().unwrap().values().cloned().collect()
    }

    /// Clear all metrics
    pub fn clear_metrics(&self) {
        self.metrics.lock().unwrap().clear();
    }

    /// Get metric by name
    pub fn get_metric(&self, name: &str) -> Option<PerformanceMetric> {
        self.metrics.lock().unwrap().get(name).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_performance_monitor() {
        let monitor = PerformanceMonitor::new(true);

        // Record some timings
        monitor.record_timing("test_op", Duration::from_millis(10));
        monitor.record_timing("test_op", Duration::from_millis(20));

        let metrics = monitor.get_metrics();
        assert_eq!(metrics.len(), 1);

        let metric = &metrics[0];
        assert_eq!(metric.name, "test_op");
        assert_eq!(metric.call_count, 2);
        assert_eq!(metric.min_time, Duration::from_millis(10));
        assert_eq!(metric.max_time, Duration::from_millis(20));
        assert_eq!(metric.avg_time, Duration::from_millis(15));
    }

    #[test]
    fn test_disabled_monitoring() {
        let monitor = PerformanceMonitor::new(false);

        monitor.record_timing("test_op", Duration::from_millis(10));

        let metrics = monitor.get_metrics();
        assert_eq!(metrics.len(), 0);
    }

    #[test]
    fn test_clear_metrics() {
        let monitor = PerformanceMonitor::new(true);

        monitor.record_timing("test_op", Duration::from_millis(10));
        assert_eq!(monitor.get_metrics().len(), 1);

        monitor.clear_metrics();
        assert_eq!(monitor.get_metrics().len(), 0);
    }

    #[test]
    fn test_get_metric_by_name() {
        let monitor = PerformanceMonitor::new(true);

        monitor.record_timing("test_op", Duration::from_millis(10));
        monitor.record_timing("other_op", Duration::from_millis(20));

        let metric = monitor.get_metric("test_op").unwrap();
        assert_eq!(metric.name, "test_op");
        assert_eq!(metric.call_count, 1);

        assert!(monitor.get_metric("nonexistent").is_none());
    }
}
