//! Performance monitoring and target validation for voice cloning operations
//!
//! This module provides comprehensive performance monitoring capabilities to track
//! and validate voice cloning performance against defined targets.

use crate::{Error, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Performance targets for voice cloning operations
#[derive(Debug, Clone)]
pub struct PerformanceTargets {
    /// Target adaptation time: <2 minutes for few-shot adaptation
    pub adaptation_time_target: Duration,
    /// Target synthesis speed: 0.1× RTF with cloned voices
    pub synthesis_rtf_target: f64,
    /// Target memory usage: <1GB for adaptation process
    pub memory_usage_target: u64,
    /// Target quality score: 85%+ similarity to original
    pub quality_score_target: f64,
    /// Target concurrent adaptations: 10+ simultaneous adaptations
    pub concurrent_adaptations_target: usize,
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            adaptation_time_target: Duration::from_secs(120), // 2 minutes
            synthesis_rtf_target: 0.1,
            memory_usage_target: 1024 * 1024 * 1024, // 1GB
            quality_score_target: 0.85,
            concurrent_adaptations_target: 10,
        }
    }
}

/// Performance metrics collected during voice cloning operations
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Actual adaptation time measured
    pub adaptation_time: Duration,
    /// Actual synthesis real-time factor
    pub synthesis_rtf: f64,
    /// Actual memory usage during adaptation
    pub memory_usage: u64,
    /// Actual quality score achieved
    pub quality_score: f64,
    /// Number of concurrent adaptations supported
    pub concurrent_adaptations: usize,
    /// Timestamp when metrics were collected
    pub timestamp: SystemTime,
}

/// Performance measurement result comparing actual vs target
#[derive(Debug, Clone)]
pub struct PerformanceMeasurement {
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Performance targets
    pub targets: PerformanceTargets,
    /// Whether each target was met
    pub target_results: TargetResults,
    /// Overall performance score (0.0-1.0)
    pub overall_score: f64,
}

/// Results of target validation
#[derive(Debug, Clone)]
pub struct TargetResults {
    /// Whether adaptation time target was met
    pub adaptation_time_met: bool,
    /// Whether synthesis RTF target was met
    pub synthesis_rtf_met: bool,
    /// Whether memory usage target was met
    pub memory_usage_met: bool,
    /// Whether quality score target was met
    pub quality_score_met: bool,
    /// Whether concurrent adaptations target was met
    pub concurrent_adaptations_met: bool,
}

/// Statistics for performance monitoring over time
#[derive(Debug, Clone)]
pub struct PerformanceStatistics {
    /// Total number of measurements
    pub total_measurements: usize,
    /// Average adaptation time
    pub avg_adaptation_time: Duration,
    /// Average synthesis RTF
    pub avg_synthesis_rtf: f64,
    /// Average memory usage
    pub avg_memory_usage: u64,
    /// Average quality score
    pub avg_quality_score: f64,
    /// Peak concurrent adaptations achieved
    pub peak_concurrent_adaptations: usize,
    /// Percentage of measurements meeting all targets
    pub target_compliance_rate: f64,
}

/// Performance monitor for tracking voice cloning performance
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Performance targets
    targets: PerformanceTargets,
    /// Historical measurements
    measurements: Arc<RwLock<Vec<PerformanceMeasurement>>>,
    /// Current active adaptations counter
    active_adaptations: Arc<RwLock<usize>>,
    /// Maximum measurements to retain
    max_measurements: usize,
}

impl PerformanceMonitor {
    /// Create a new performance monitor with default targets
    pub fn new() -> Self {
        Self::with_targets(PerformanceTargets::default())
    }

    /// Create a new performance monitor with custom targets
    pub fn with_targets(targets: PerformanceTargets) -> Self {
        Self {
            targets,
            measurements: Arc::new(RwLock::new(Vec::new())),
            active_adaptations: Arc::new(RwLock::new(0)),
            max_measurements: 1000, // Keep last 1000 measurements
        }
    }

    /// Start monitoring an adaptation operation
    pub async fn start_adaptation_monitoring(&self) -> AdaptationMonitor<'_> {
        let mut active = self.active_adaptations.write().await;
        *active += 1;
        let concurrent_count = *active;

        AdaptationMonitor {
            start_time: Instant::now(),
            start_memory: Self::get_memory_usage(),
            concurrent_adaptations: concurrent_count,
            monitor: self,
        }
    }

    /// Record a performance measurement
    pub async fn record_measurement(&self, measurement: PerformanceMeasurement) {
        let mut measurements = self.measurements.write().await;
        measurements.push(measurement.clone());

        // Keep only the most recent measurements
        if measurements.len() > self.max_measurements {
            measurements.remove(0);
        }

        // Log if targets are not being met
        self.log_performance_status(&measurement).await;
    }

    /// Get current performance statistics
    pub async fn get_statistics(&self) -> Result<PerformanceStatistics> {
        let measurements = self.measurements.read().await;

        if measurements.is_empty() {
            return Ok(PerformanceStatistics {
                total_measurements: 0,
                avg_adaptation_time: Duration::from_secs(0),
                avg_synthesis_rtf: 0.0,
                avg_memory_usage: 0,
                avg_quality_score: 0.0,
                peak_concurrent_adaptations: 0,
                target_compliance_rate: 0.0,
            });
        }

        let total = measurements.len();
        let mut total_adaptation_time = Duration::from_secs(0);
        let mut total_synthesis_rtf = 0.0;
        let mut total_memory_usage = 0u64;
        let mut total_quality_score = 0.0;
        let mut peak_concurrent = 0;
        let mut compliant_measurements = 0;

        for measurement in measurements.iter() {
            total_adaptation_time += measurement.metrics.adaptation_time;
            total_synthesis_rtf += measurement.metrics.synthesis_rtf;
            total_memory_usage += measurement.metrics.memory_usage;
            total_quality_score += measurement.metrics.quality_score;
            peak_concurrent = peak_concurrent.max(measurement.metrics.concurrent_adaptations);

            // Check if all targets were met
            if measurement.target_results.adaptation_time_met
                && measurement.target_results.synthesis_rtf_met
                && measurement.target_results.memory_usage_met
                && measurement.target_results.quality_score_met
                && measurement.target_results.concurrent_adaptations_met
            {
                compliant_measurements += 1;
            }
        }

        Ok(PerformanceStatistics {
            total_measurements: total,
            avg_adaptation_time: total_adaptation_time / total as u32,
            avg_synthesis_rtf: total_synthesis_rtf / total as f64,
            avg_memory_usage: total_memory_usage / total as u64,
            avg_quality_score: total_quality_score / total as f64,
            peak_concurrent_adaptations: peak_concurrent,
            target_compliance_rate: compliant_measurements as f64 / total as f64,
        })
    }

    /// Get the current performance targets
    pub fn get_targets(&self) -> &PerformanceTargets {
        &self.targets
    }

    /// Update performance targets
    pub fn update_targets(&mut self, targets: PerformanceTargets) {
        self.targets = targets;
        info!("Performance targets updated");
    }

    /// Generate a performance report
    pub async fn generate_report(&self) -> Result<String> {
        let stats = self.get_statistics().await?;
        let measurements = self.measurements.read().await;

        let mut report = String::new();
        report.push_str("=== Voice Cloning Performance Report ===\n\n");

        report.push_str(&format!(
            "Total Measurements: {}\n",
            stats.total_measurements
        ));
        report.push_str(&format!(
            "Target Compliance Rate: {:.1}%\n",
            stats.target_compliance_rate * 100.0
        ));
        report.push_str("\n--- Performance Averages ---\n");
        report.push_str(&format!(
            "Adaptation Time: {:.2}s (target: {:.2}s)\n",
            stats.avg_adaptation_time.as_secs_f64(),
            self.targets.adaptation_time_target.as_secs_f64()
        ));
        report.push_str(&format!(
            "Synthesis RTF: {:.3} (target: {:.3})\n",
            stats.avg_synthesis_rtf, self.targets.synthesis_rtf_target
        ));
        report.push_str(&format!(
            "Memory Usage: {:.1} MB (target: {:.1} MB)\n",
            stats.avg_memory_usage as f64 / (1024.0 * 1024.0),
            self.targets.memory_usage_target as f64 / (1024.0 * 1024.0)
        ));
        report.push_str(&format!(
            "Quality Score: {:.3} (target: {:.3})\n",
            stats.avg_quality_score, self.targets.quality_score_target
        ));
        report.push_str(&format!(
            "Peak Concurrent Adaptations: {} (target: {})\n",
            stats.peak_concurrent_adaptations, self.targets.concurrent_adaptations_target
        ));

        // Add recent performance trend
        if measurements.len() >= 10 {
            let recent = &measurements[measurements.len() - 10..];
            let recent_compliance = recent.iter().filter(|m| m.overall_score >= 0.8).count() as f64
                / recent.len() as f64;
            report.push_str(&format!(
                "\nRecent Performance Trend (last 10): {:.1}% compliance\n",
                recent_compliance * 100.0
            ));
        }

        Ok(report)
    }

    /// Get memory usage (simplified implementation)
    fn get_memory_usage() -> u64 {
        // In a real implementation, this would use system APIs to get actual memory usage
        // For now, return a placeholder value
        256 * 1024 * 1024 // 256 MB
    }

    /// Log performance status
    async fn log_performance_status(&self, measurement: &PerformanceMeasurement) {
        let results = &measurement.target_results;

        if !results.adaptation_time_met {
            warn!(
                "Adaptation time target not met: {:.2}s > {:.2}s",
                measurement.metrics.adaptation_time.as_secs_f64(),
                self.targets.adaptation_time_target.as_secs_f64()
            );
        }

        if !results.synthesis_rtf_met {
            warn!(
                "Synthesis RTF target not met: {:.3} > {:.3}",
                measurement.metrics.synthesis_rtf, self.targets.synthesis_rtf_target
            );
        }

        if !results.memory_usage_met {
            warn!(
                "Memory usage target not met: {:.1} MB > {:.1} MB",
                measurement.metrics.memory_usage as f64 / (1024.0 * 1024.0),
                self.targets.memory_usage_target as f64 / (1024.0 * 1024.0)
            );
        }

        if !results.quality_score_met {
            warn!(
                "Quality score target not met: {:.3} < {:.3}",
                measurement.metrics.quality_score, self.targets.quality_score_target
            );
        }

        if measurement.overall_score >= 0.8 {
            info!(
                "Performance targets mostly met (score: {:.2})",
                measurement.overall_score
            );
        } else {
            warn!(
                "Performance targets not met (score: {:.2})",
                measurement.overall_score
            );
        }
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Monitor for tracking individual adaptation operations
pub struct AdaptationMonitor<'a> {
    start_time: Instant,
    start_memory: u64,
    concurrent_adaptations: usize,
    monitor: &'a PerformanceMonitor,
}

impl<'a> AdaptationMonitor<'a> {
    /// Complete the adaptation monitoring and record results
    pub async fn complete(
        self,
        synthesis_rtf: f64,
        quality_score: f64,
    ) -> Result<PerformanceMeasurement> {
        let adaptation_time = self.start_time.elapsed();
        let end_memory = PerformanceMonitor::get_memory_usage();
        let memory_used = end_memory.saturating_sub(self.start_memory);

        let metrics = PerformanceMetrics {
            adaptation_time,
            synthesis_rtf,
            memory_usage: memory_used,
            quality_score,
            concurrent_adaptations: self.concurrent_adaptations,
            timestamp: SystemTime::now(),
        };

        let targets = &self.monitor.targets;
        let target_results = TargetResults {
            adaptation_time_met: adaptation_time <= targets.adaptation_time_target,
            synthesis_rtf_met: synthesis_rtf <= targets.synthesis_rtf_target,
            memory_usage_met: memory_used <= targets.memory_usage_target,
            quality_score_met: quality_score >= targets.quality_score_target,
            concurrent_adaptations_met: self.concurrent_adaptations
                >= targets.concurrent_adaptations_target,
        };

        // Calculate overall score (percentage of targets met)
        let targets_met_count = [
            target_results.adaptation_time_met,
            target_results.synthesis_rtf_met,
            target_results.memory_usage_met,
            target_results.quality_score_met,
            target_results.concurrent_adaptations_met,
        ]
        .iter()
        .filter(|&&met| met)
        .count();

        let overall_score = targets_met_count as f64 / 5.0;

        let measurement = PerformanceMeasurement {
            metrics,
            targets: targets.clone(),
            target_results,
            overall_score,
        };

        // Record the measurement
        self.monitor.record_measurement(measurement.clone()).await;

        // Decrement active adaptations counter
        {
            let mut active = self.monitor.active_adaptations.write().await;
            *active = active.saturating_sub(1);
        }

        debug!(
            "Adaptation completed with performance score: {:.2}",
            overall_score
        );

        Ok(measurement)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_performance_monitor_creation() {
        let monitor = PerformanceMonitor::new();
        let targets = monitor.get_targets();

        assert_eq!(targets.adaptation_time_target, Duration::from_secs(120));
        assert_eq!(targets.synthesis_rtf_target, 0.1);
        assert_eq!(targets.memory_usage_target, 1024 * 1024 * 1024);
        assert_eq!(targets.quality_score_target, 0.85);
        assert_eq!(targets.concurrent_adaptations_target, 10);
    }

    #[tokio::test]
    async fn test_adaptation_monitoring() {
        let monitor = PerformanceMonitor::new();
        let adaptation_monitor = monitor.start_adaptation_monitoring().await;

        // Simulate some processing time
        tokio::time::sleep(Duration::from_millis(10)).await;

        let measurement = adaptation_monitor.complete(0.05, 0.90).await.unwrap();

        assert!(measurement.metrics.adaptation_time > Duration::from_millis(5));
        assert_eq!(measurement.metrics.synthesis_rtf, 0.05);
        assert_eq!(measurement.metrics.quality_score, 0.90);
        assert!(measurement.target_results.synthesis_rtf_met);
        assert!(measurement.target_results.quality_score_met);
    }

    #[tokio::test]
    async fn test_performance_statistics() {
        let monitor = PerformanceMonitor::new();

        // Record some test measurements
        for i in 0..5 {
            let adaptation_monitor = monitor.start_adaptation_monitoring().await;
            tokio::time::sleep(Duration::from_millis(10)).await;
            let _measurement = adaptation_monitor
                .complete(0.05 + i as f64 * 0.01, 0.85 + i as f64 * 0.01)
                .await
                .unwrap();
        }

        let stats = monitor.get_statistics().await.unwrap();
        assert_eq!(stats.total_measurements, 5);
        assert!(stats.avg_synthesis_rtf > 0.05);
        assert!(stats.avg_quality_score >= 0.85);
    }

    #[tokio::test]
    async fn test_performance_report_generation() {
        let monitor = PerformanceMonitor::new();

        let adaptation_monitor = monitor.start_adaptation_monitoring().await;
        let _measurement = adaptation_monitor.complete(0.08, 0.90).await.unwrap();

        let report = monitor.generate_report().await.unwrap();
        assert!(report.contains("Voice Cloning Performance Report"));
        assert!(report.contains("Total Measurements: 1"));
        assert!(report.contains("Synthesis RTF:"));
        assert!(report.contains("Quality Score:"));
    }

    #[tokio::test]
    async fn test_custom_targets() {
        let custom_targets = PerformanceTargets {
            adaptation_time_target: Duration::from_secs(60),
            synthesis_rtf_target: 0.2,
            memory_usage_target: 512 * 1024 * 1024,
            quality_score_target: 0.90,
            concurrent_adaptations_target: 5,
        };

        let monitor = PerformanceMonitor::with_targets(custom_targets.clone());
        assert_eq!(
            monitor.get_targets().adaptation_time_target,
            Duration::from_secs(60)
        );
        assert_eq!(monitor.get_targets().synthesis_rtf_target, 0.2);
        assert_eq!(monitor.get_targets().quality_score_target, 0.90);
    }
}
