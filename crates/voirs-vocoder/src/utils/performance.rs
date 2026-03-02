//! Advanced Performance Monitoring Utilities
//!
//! This module provides comprehensive performance monitoring and profiling tools
//! for vocoder operations, including:
//! - Real-Time Factor (RTF) measurement
//! - Latency tracking with percentiles
//! - Throughput analysis
//! - Resource utilization monitoring
//! - Performance regression detection

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Performance metrics for a single operation
#[derive(Debug, Clone, Copy)]
pub struct PerformanceMetrics {
    /// Real-Time Factor (processing time / audio duration)
    pub rtf: f32,
    /// Latency in milliseconds
    pub latency_ms: f32,
    /// Audio duration in seconds
    pub audio_duration_s: f32,
    /// Number of samples processed
    pub sample_count: usize,
    /// Timestamp when measurement was taken
    pub timestamp: Instant,
}

impl PerformanceMetrics {
    /// Create new performance metrics
    pub fn new(processing_time: Duration, audio_duration_s: f32, sample_count: usize) -> Self {
        let latency_ms = processing_time.as_micros() as f32 / 1000.0;
        let rtf = processing_time.as_secs_f32() / audio_duration_s;

        Self {
            rtf,
            latency_ms,
            audio_duration_s,
            sample_count,
            timestamp: Instant::now(),
        }
    }

    /// Check if performance is real-time capable (RTF < 1.0)
    #[inline]
    pub fn is_realtime_capable(&self) -> bool {
        self.rtf < 1.0
    }

    /// Get throughput in samples per second
    #[inline]
    pub fn throughput_sps(&self) -> f32 {
        self.sample_count as f32 / (self.latency_ms / 1000.0)
    }

    /// Performance tier classification
    pub fn performance_tier(&self) -> PerformanceTier {
        if self.rtf < 0.1 {
            PerformanceTier::Excellent
        } else if self.rtf < 0.5 {
            PerformanceTier::Good
        } else if self.rtf < 1.0 {
            PerformanceTier::Fair
        } else {
            PerformanceTier::Poor
        }
    }
}

/// Performance tier classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceTier {
    /// RTF < 0.1 (10× faster than real-time)
    Excellent,
    /// RTF < 0.5 (2× faster than real-time)
    Good,
    /// RTF < 1.0 (faster than real-time)
    Fair,
    /// RTF ≥ 1.0 (slower than real-time)
    Poor,
}

impl std::fmt::Display for PerformanceTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Excellent => write!(f, "🚀 Excellent"),
            Self::Good => write!(f, "✨ Good"),
            Self::Fair => write!(f, "⚡ Fair"),
            Self::Poor => write!(f, "⚠️  Poor"),
        }
    }
}

/// Aggregated performance statistics
#[derive(Debug, Clone)]
pub struct PerformanceStatistics {
    /// Number of measurements
    pub count: usize,
    /// Mean RTF
    pub mean_rtf: f32,
    /// Median RTF
    pub median_rtf: f32,
    /// Minimum RTF
    pub min_rtf: f32,
    /// Maximum RTF
    pub max_rtf: f32,
    /// 95th percentile RTF
    pub p95_rtf: f32,
    /// 99th percentile RTF
    pub p99_rtf: f32,
    /// Mean latency in milliseconds
    pub mean_latency_ms: f32,
    /// 95th percentile latency
    pub p95_latency_ms: f32,
    /// 99th percentile latency
    pub p99_latency_ms: f32,
    /// Total samples processed
    pub total_samples: usize,
    /// Mean throughput (samples/second)
    pub mean_throughput_sps: f32,
}

impl PerformanceStatistics {
    /// Calculate statistics from a collection of metrics
    pub fn from_metrics(metrics: &[PerformanceMetrics]) -> Self {
        if metrics.is_empty() {
            return Self::empty();
        }

        let count = metrics.len();

        // Collect RTF and latency values
        let mut rtfs: Vec<f32> = metrics.iter().map(|m| m.rtf).collect();
        let mut latencies: Vec<f32> = metrics.iter().map(|m| m.latency_ms).collect();

        // Sort for percentile calculation
        rtfs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate statistics
        let mean_rtf = rtfs.iter().sum::<f32>() / count as f32;
        let median_rtf = rtfs[count / 2];
        let min_rtf = rtfs[0];
        let max_rtf = rtfs[count - 1];
        let p95_rtf = rtfs[(count as f32 * 0.95) as usize];
        let p99_rtf = rtfs[(count as f32 * 0.99) as usize];

        let mean_latency_ms = latencies.iter().sum::<f32>() / count as f32;
        let p95_latency_ms = latencies[(count as f32 * 0.95) as usize];
        let p99_latency_ms = latencies[(count as f32 * 0.99) as usize];

        let total_samples: usize = metrics.iter().map(|m| m.sample_count).sum();
        let total_time_s: f32 = metrics.iter().map(|m| m.latency_ms).sum::<f32>() / 1000.0;
        let mean_throughput_sps = if total_time_s > 0.0 {
            total_samples as f32 / total_time_s
        } else {
            0.0
        };

        Self {
            count,
            mean_rtf,
            median_rtf,
            min_rtf,
            max_rtf,
            p95_rtf,
            p99_rtf,
            mean_latency_ms,
            p95_latency_ms,
            p99_latency_ms,
            total_samples,
            mean_throughput_sps,
        }
    }

    /// Create empty statistics
    fn empty() -> Self {
        Self {
            count: 0,
            mean_rtf: 0.0,
            median_rtf: 0.0,
            min_rtf: 0.0,
            max_rtf: 0.0,
            p95_rtf: 0.0,
            p99_rtf: 0.0,
            mean_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            total_samples: 0,
            mean_throughput_sps: 0.0,
        }
    }

    /// Print formatted statistics
    pub fn print_report(&self, title: &str) {
        println!("\n📊 {}", title);
        println!("═══════════════════════════════════════════════");
        println!("  Measurements: {}", self.count);
        println!("  Total samples: {}", self.total_samples);
        println!();
        println!("  RTF Statistics:");
        println!("  ├─ Mean:   {:.4}×", self.mean_rtf);
        println!("  ├─ Median: {:.4}×", self.median_rtf);
        println!("  ├─ Min:    {:.4}×", self.min_rtf);
        println!("  ├─ Max:    {:.4}×", self.max_rtf);
        println!("  ├─ P95:    {:.4}×", self.p95_rtf);
        println!("  └─ P99:    {:.4}×", self.p99_rtf);
        println!();
        println!("  Latency Statistics:");
        println!("  ├─ Mean: {:.2} ms", self.mean_latency_ms);
        println!("  ├─ P95:  {:.2} ms", self.p95_latency_ms);
        println!("  └─ P99:  {:.2} ms", self.p99_latency_ms);
        println!();
        println!("  Throughput: {:.0} samples/sec", self.mean_throughput_sps);
        println!("═══════════════════════════════════════════════");
    }
}

/// Rolling window performance monitor
pub struct PerformanceMonitor {
    /// Window of recent metrics
    metrics: Arc<Mutex<VecDeque<PerformanceMetrics>>>,
    /// Maximum window size
    window_size: usize,
    /// Name/label for this monitor
    name: String,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(name: impl Into<String>, window_size: usize) -> Self {
        Self {
            metrics: Arc::new(Mutex::new(VecDeque::with_capacity(window_size))),
            window_size,
            name: name.into(),
        }
    }

    /// Record a new measurement
    pub fn record(&self, processing_time: Duration, audio_duration_s: f32, sample_count: usize) {
        let metric = PerformanceMetrics::new(processing_time, audio_duration_s, sample_count);
        let mut metrics = self.metrics.lock();

        metrics.push_back(metric);

        // Keep only the most recent measurements
        while metrics.len() > self.window_size {
            metrics.pop_front();
        }
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> PerformanceStatistics {
        let metrics = self.metrics.lock();
        let metrics_vec: Vec<PerformanceMetrics> = metrics.iter().copied().collect();
        PerformanceStatistics::from_metrics(&metrics_vec)
    }

    /// Get the most recent metric
    pub fn latest(&self) -> Option<PerformanceMetrics> {
        self.metrics.lock().back().copied()
    }

    /// Check if performance has regressed (compared to baseline)
    pub fn detect_regression(&self, baseline_rtf: f32, threshold: f32) -> bool {
        if let Some(latest) = self.latest() {
            latest.rtf > baseline_rtf * (1.0 + threshold)
        } else {
            false
        }
    }

    /// Clear all measurements
    pub fn clear(&self) {
        self.metrics.lock().clear();
    }

    /// Get monitor name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get window size
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Get current number of measurements
    pub fn count(&self) -> usize {
        self.metrics.lock().len()
    }

    /// Print current statistics
    pub fn print_report(&self) {
        let stats = self.get_statistics();
        stats.print_report(&format!("{} Performance Report", self.name));
    }
}

/// Timer for measuring operation duration
pub struct PerformanceTimer {
    start: Instant,
    label: String,
}

impl PerformanceTimer {
    /// Create and start a new timer
    pub fn start(label: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
            label: label.into(),
        }
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Stop the timer and print result
    pub fn stop(self) {
        let elapsed = self.elapsed();
        println!(
            "⏱️  {}: {:.2}ms",
            self.label,
            elapsed.as_micros() as f32 / 1000.0
        );
    }

    /// Stop and return elapsed time without printing
    pub fn stop_silent(self) -> Duration {
        self.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics_creation() {
        let processing_time = Duration::from_millis(50);
        let audio_duration = 1.0; // 1 second
        let sample_count = 22050;

        let metrics = PerformanceMetrics::new(processing_time, audio_duration, sample_count);

        assert_eq!(metrics.latency_ms, 50.0);
        assert_eq!(metrics.rtf, 0.05);
        assert_eq!(metrics.audio_duration_s, 1.0);
        assert_eq!(metrics.sample_count, 22050);
        assert!(metrics.is_realtime_capable());
    }

    #[test]
    fn test_performance_tier() {
        let excellent = PerformanceMetrics::new(Duration::from_millis(10), 1.0, 22050);
        assert_eq!(excellent.performance_tier(), PerformanceTier::Excellent);

        let good = PerformanceMetrics::new(Duration::from_millis(300), 1.0, 22050);
        assert_eq!(good.performance_tier(), PerformanceTier::Good);

        let fair = PerformanceMetrics::new(Duration::from_millis(800), 1.0, 22050);
        assert_eq!(fair.performance_tier(), PerformanceTier::Fair);

        let poor = PerformanceMetrics::new(Duration::from_millis(1200), 1.0, 22050);
        assert_eq!(poor.performance_tier(), PerformanceTier::Poor);
    }

    #[test]
    fn test_performance_monitor() {
        let monitor = PerformanceMonitor::new("Test", 10);

        // Record some measurements
        for i in 1..=5 {
            let processing_time = Duration::from_millis(i * 10);
            monitor.record(processing_time, 1.0, 22050);
        }

        assert_eq!(monitor.count(), 5);

        let stats = monitor.get_statistics();
        assert_eq!(stats.count, 5);
        assert!(stats.mean_rtf > 0.0);

        // Test latest
        let latest = monitor.latest().unwrap();
        assert_eq!(latest.latency_ms, 50.0);
    }

    #[test]
    fn test_performance_statistics() {
        let metrics = vec![
            PerformanceMetrics::new(Duration::from_millis(10), 1.0, 22050),
            PerformanceMetrics::new(Duration::from_millis(20), 1.0, 22050),
            PerformanceMetrics::new(Duration::from_millis(30), 1.0, 22050),
            PerformanceMetrics::new(Duration::from_millis(40), 1.0, 22050),
            PerformanceMetrics::new(Duration::from_millis(50), 1.0, 22050),
        ];

        let stats = PerformanceStatistics::from_metrics(&metrics);

        assert_eq!(stats.count, 5);
        assert_eq!(stats.mean_latency_ms, 30.0);
        assert_eq!(stats.total_samples, 22050 * 5);
    }

    #[test]
    fn test_regression_detection() {
        let monitor = PerformanceMonitor::new("Regression Test", 10);

        // Baseline: 50ms for 1 second audio (RTF = 0.05)
        monitor.record(Duration::from_millis(50), 1.0, 22050);

        // No regression
        assert!(!monitor.detect_regression(0.05, 0.1)); // Within 10% threshold

        // Add a regressed measurement (150ms, RTF = 0.15, which is 3x baseline)
        monitor.record(Duration::from_millis(150), 1.0, 22050);

        // Should detect regression with 10% threshold
        assert!(monitor.detect_regression(0.05, 0.1));
    }

    #[test]
    fn test_throughput_calculation() {
        let metrics = PerformanceMetrics::new(Duration::from_millis(100), 1.0, 22050);

        let throughput = metrics.throughput_sps();
        assert!((throughput - 220_500.0).abs() < 1.0); // 22050 samples / 0.1 seconds
    }
}
