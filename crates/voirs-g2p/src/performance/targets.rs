//! Performance target monitoring for G2P systems
//!
//! This module implements comprehensive performance target monitoring with the specific
//! goals of <1ms latency for typical sentences and <100MB memory footprint per language model.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::LanguageCode;

/// Performance targets based on TODO.md requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    /// Maximum latency per sentence (target: <1ms for 20-50 characters)
    pub max_latency_ms: f64,
    /// Maximum memory footprint per language model (target: <100MB)
    pub max_memory_mb: f64,
    /// Minimum batch processing throughput (target: >1000 sentences/second)
    pub min_throughput_sentences_per_sec: f64,
    /// Maximum CPU usage percentage for real-time processing
    pub max_cpu_usage_percent: f64,
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            max_latency_ms: 1.0,                      // <1ms target
            max_memory_mb: 100.0,                     // <100MB target
            min_throughput_sentences_per_sec: 1000.0, // >1000 sentences/sec target
            max_cpu_usage_percent: 80.0,              // <80% CPU usage
        }
    }
}

/// Latency measurement for different sentence lengths
#[derive(Debug, Clone, Serialize)]
pub struct LatencyMeasurement {
    pub sentence_length: usize,
    pub latency_ms: f64,
    pub phoneme_count: usize,
    pub language: LanguageCode,
    pub timestamp: std::time::SystemTime,
}

/// Memory usage snapshot
#[derive(Debug, Clone, Serialize)]
pub struct MemorySnapshot {
    pub total_memory_mb: f64,
    pub model_memory_mb: f64,
    pub cache_memory_mb: f64,
    pub working_memory_mb: f64,
    pub language: LanguageCode,
    pub timestamp: std::time::SystemTime,
}

/// Throughput measurement
#[derive(Debug, Clone, Serialize)]
pub struct ThroughputMeasurement {
    pub sentences_processed: usize,
    pub duration_ms: f64,
    pub sentences_per_second: f64,
    pub avg_sentence_length: f64,
    pub language: LanguageCode,
    pub timestamp: std::time::SystemTime,
}

/// Performance target violation
#[derive(Debug, Clone, Serialize)]
pub struct TargetViolation {
    pub target_type: String,
    pub target_value: f64,
    pub actual_value: f64,
    pub severity: ViolationSeverity,
    pub language: Option<LanguageCode>,
    pub timestamp: std::time::SystemTime,
    pub context: String,
}

/// Severity of performance target violations
#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum ViolationSeverity {
    /// Minor violation (within 10% of target)
    Minor,
    /// Moderate violation (10-50% over target)
    Moderate,
    /// Critical violation (>50% over target)
    Critical,
}

/// Performance target monitor
pub struct PerformanceTargetMonitor {
    targets: PerformanceTargets,
    latency_measurements: Arc<Mutex<Vec<LatencyMeasurement>>>,
    memory_snapshots: Arc<Mutex<Vec<MemorySnapshot>>>,
    throughput_measurements: Arc<Mutex<Vec<ThroughputMeasurement>>>,
    violations: Arc<Mutex<Vec<TargetViolation>>>,
    max_measurements: usize,
}

impl PerformanceTargetMonitor {
    /// Create a new performance target monitor
    pub fn new(targets: PerformanceTargets) -> Self {
        Self {
            targets,
            latency_measurements: Arc::new(Mutex::new(Vec::new())),
            memory_snapshots: Arc::new(Mutex::new(Vec::new())),
            throughput_measurements: Arc::new(Mutex::new(Vec::new())),
            violations: Arc::new(Mutex::new(Vec::new())),
            max_measurements: 1000, // Keep last 1000 measurements
        }
    }

    /// Create with default targets
    pub fn new_with_defaults() -> Self {
        Self::new(PerformanceTargets::default())
    }

    /// Record a latency measurement
    pub fn record_latency(
        &self,
        sentence: &str,
        latency: Duration,
        phoneme_count: usize,
        language: LanguageCode,
    ) -> Result<()> {
        let latency_ms = latency.as_millis() as f64;

        let measurement = LatencyMeasurement {
            sentence_length: sentence.len(),
            latency_ms,
            phoneme_count,
            language,
            timestamp: std::time::SystemTime::now(),
        };

        // Store measurement
        {
            let mut measurements = self.latency_measurements.lock().unwrap();
            measurements.push(measurement.clone());

            // Keep only recent measurements
            if measurements.len() > self.max_measurements {
                measurements.remove(0);
            }
        }

        // Check for violations
        self.check_latency_target(&measurement)?;

        Ok(())
    }

    /// Record a memory usage snapshot
    pub fn record_memory_usage(
        &self,
        total_memory_mb: f64,
        model_memory_mb: f64,
        cache_memory_mb: f64,
        working_memory_mb: f64,
        language: LanguageCode,
    ) -> Result<()> {
        let snapshot = MemorySnapshot {
            total_memory_mb,
            model_memory_mb,
            cache_memory_mb,
            working_memory_mb,
            language,
            timestamp: std::time::SystemTime::now(),
        };

        // Store snapshot
        {
            let mut snapshots = self.memory_snapshots.lock().unwrap();
            snapshots.push(snapshot.clone());

            // Keep only recent snapshots
            if snapshots.len() > self.max_measurements {
                snapshots.remove(0);
            }
        }

        // Check for violations
        self.check_memory_target(&snapshot)?;

        Ok(())
    }

    /// Record throughput measurement
    pub fn record_throughput(
        &self,
        sentences_processed: usize,
        duration: Duration,
        avg_sentence_length: f64,
        language: LanguageCode,
    ) -> Result<()> {
        let duration_ms = duration.as_millis() as f64;
        let sentences_per_second = if duration_ms > 0.0 {
            (sentences_processed as f64) / (duration_ms / 1000.0)
        } else {
            0.0
        };

        let measurement = ThroughputMeasurement {
            sentences_processed,
            duration_ms,
            sentences_per_second,
            avg_sentence_length,
            language,
            timestamp: std::time::SystemTime::now(),
        };

        // Store measurement
        {
            let mut measurements = self.throughput_measurements.lock().unwrap();
            measurements.push(measurement.clone());

            // Keep only recent measurements
            if measurements.len() > self.max_measurements {
                measurements.remove(0);
            }
        }

        // Check for violations
        self.check_throughput_target(&measurement)?;

        Ok(())
    }

    /// Check latency target compliance
    fn check_latency_target(&self, measurement: &LatencyMeasurement) -> Result<()> {
        if measurement.latency_ms > self.targets.max_latency_ms {
            let severity = if measurement.latency_ms > self.targets.max_latency_ms * 1.5 {
                ViolationSeverity::Critical
            } else if measurement.latency_ms > self.targets.max_latency_ms * 1.1 {
                ViolationSeverity::Moderate
            } else {
                ViolationSeverity::Minor
            };

            let violation = TargetViolation {
                target_type: "latency".to_string(),
                target_value: self.targets.max_latency_ms,
                actual_value: measurement.latency_ms,
                severity,
                language: Some(measurement.language),
                timestamp: measurement.timestamp,
                context: format!(
                    "Sentence length: {} chars, {} phonemes",
                    measurement.sentence_length, measurement.phoneme_count
                ),
            };

            self.violations.lock().unwrap().push(violation);
        }

        Ok(())
    }

    /// Check memory target compliance
    fn check_memory_target(&self, snapshot: &MemorySnapshot) -> Result<()> {
        if snapshot.model_memory_mb > self.targets.max_memory_mb {
            let severity = if snapshot.model_memory_mb > self.targets.max_memory_mb * 1.5 {
                ViolationSeverity::Critical
            } else if snapshot.model_memory_mb > self.targets.max_memory_mb * 1.1 {
                ViolationSeverity::Moderate
            } else {
                ViolationSeverity::Minor
            };

            let violation = TargetViolation {
                target_type: "memory".to_string(),
                target_value: self.targets.max_memory_mb,
                actual_value: snapshot.model_memory_mb,
                severity,
                language: Some(snapshot.language),
                timestamp: snapshot.timestamp,
                context: format!(
                    "Total: {:.1}MB, Cache: {:.1}MB, Working: {:.1}MB",
                    snapshot.total_memory_mb, snapshot.cache_memory_mb, snapshot.working_memory_mb
                ),
            };

            self.violations.lock().unwrap().push(violation);
        }

        Ok(())
    }

    /// Check throughput target compliance
    fn check_throughput_target(&self, measurement: &ThroughputMeasurement) -> Result<()> {
        if measurement.sentences_per_second < self.targets.min_throughput_sentences_per_sec {
            let severity = if measurement.sentences_per_second
                < self.targets.min_throughput_sentences_per_sec * 0.5
            {
                ViolationSeverity::Critical
            } else if measurement.sentences_per_second
                < self.targets.min_throughput_sentences_per_sec * 0.9
            {
                ViolationSeverity::Moderate
            } else {
                ViolationSeverity::Minor
            };

            let violation = TargetViolation {
                target_type: "throughput".to_string(),
                target_value: self.targets.min_throughput_sentences_per_sec,
                actual_value: measurement.sentences_per_second,
                severity,
                language: Some(measurement.language),
                timestamp: measurement.timestamp,
                context: format!(
                    "Processed {} sentences in {:.1}ms, avg length: {:.1} chars",
                    measurement.sentences_processed,
                    measurement.duration_ms,
                    measurement.avg_sentence_length
                ),
            };

            self.violations.lock().unwrap().push(violation);
        }

        Ok(())
    }

    /// Get performance summary
    pub fn get_performance_summary(&self) -> PerformanceSummary {
        let latency_measurements = self.latency_measurements.lock().unwrap();
        let memory_snapshots = self.memory_snapshots.lock().unwrap();
        let throughput_measurements = self.throughput_measurements.lock().unwrap();
        let violations = self.violations.lock().unwrap();

        // Calculate latency statistics
        let latency_stats = if latency_measurements.is_empty() {
            None
        } else {
            let latencies: Vec<f64> = latency_measurements.iter().map(|m| m.latency_ms).collect();
            Some(LatencyStats {
                avg_latency_ms: latencies.iter().sum::<f64>() / latencies.len() as f64,
                min_latency_ms: latencies.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                max_latency_ms: latencies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                p95_latency_ms: percentile(&latencies, 95.0),
                p99_latency_ms: percentile(&latencies, 99.0),
                measurements_count: latencies.len(),
            })
        };

        // Calculate memory statistics
        let memory_stats = if memory_snapshots.is_empty() {
            None
        } else {
            let model_memory: Vec<f64> =
                memory_snapshots.iter().map(|s| s.model_memory_mb).collect();
            Some(MemoryStats {
                avg_model_memory_mb: model_memory.iter().sum::<f64>() / model_memory.len() as f64,
                max_model_memory_mb: model_memory
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                avg_total_memory_mb: memory_snapshots
                    .iter()
                    .map(|s| s.total_memory_mb)
                    .sum::<f64>()
                    / memory_snapshots.len() as f64,
                snapshots_count: memory_snapshots.len(),
            })
        };

        // Calculate throughput statistics
        let throughput_stats = if throughput_measurements.is_empty() {
            None
        } else {
            let throughputs: Vec<f64> = throughput_measurements
                .iter()
                .map(|m| m.sentences_per_second)
                .collect();
            Some(ThroughputStats {
                avg_throughput_sentences_per_sec: throughputs.iter().sum::<f64>()
                    / throughputs.len() as f64,
                max_throughput_sentences_per_sec: throughputs
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                measurements_count: throughputs.len(),
            })
        };

        // Violation summary
        let violation_summary = ViolationSummary {
            total_violations: violations.len(),
            critical_violations: violations
                .iter()
                .filter(|v| v.severity == ViolationSeverity::Critical)
                .count(),
            moderate_violations: violations
                .iter()
                .filter(|v| v.severity == ViolationSeverity::Moderate)
                .count(),
            minor_violations: violations
                .iter()
                .filter(|v| v.severity == ViolationSeverity::Minor)
                .count(),
        };

        PerformanceSummary {
            targets: self.targets.clone(),
            latency_stats,
            memory_stats,
            throughput_stats,
            violation_summary,
        }
    }

    /// Check if all targets are currently being met
    pub fn are_targets_met(&self) -> bool {
        let violations = self.violations.lock().unwrap();

        // No critical violations in recent measurements
        let recent_critical_violations = violations
            .iter()
            .rev()
            .take(10) // Check last 10 violations
            .any(|v| v.severity == ViolationSeverity::Critical);

        !recent_critical_violations
    }

    /// Generate performance target report
    pub fn generate_report(&self) -> String {
        let summary = self.get_performance_summary();

        let mut report = String::new();
        report.push_str("🎯 G2P Performance Target Report\n");
        report.push_str("===============================\n\n");

        // Targets overview
        report.push_str(&format!(
            "📋 Performance Targets:\n\
             ├── Max Latency: {:.1}ms per sentence\n\
             ├── Max Memory: {:.1}MB per language model\n\
             ├── Min Throughput: {:.0} sentences/sec\n\
             └── Max CPU Usage: {:.1}%\n\n",
            summary.targets.max_latency_ms,
            summary.targets.max_memory_mb,
            summary.targets.min_throughput_sentences_per_sec,
            summary.targets.max_cpu_usage_percent
        ));

        // Latency results
        if let Some(latency) = summary.latency_stats {
            let latency_status = if latency.p95_latency_ms <= summary.targets.max_latency_ms {
                "✅ PASSED"
            } else {
                "❌ FAILED"
            };

            report.push_str(&format!(
                "⏱️  Latency Performance: {}\n\
                 ├── Average: {:.2}ms\n\
                 ├── 95th percentile: {:.2}ms\n\
                 ├── 99th percentile: {:.2}ms\n\
                 ├── Min: {:.2}ms, Max: {:.2}ms\n\
                 └── Measurements: {} samples\n\n",
                latency_status,
                latency.avg_latency_ms,
                latency.p95_latency_ms,
                latency.p99_latency_ms,
                latency.min_latency_ms,
                latency.max_latency_ms,
                latency.measurements_count
            ));
        }

        // Memory results
        if let Some(memory) = summary.memory_stats {
            let memory_status = if memory.max_model_memory_mb <= summary.targets.max_memory_mb {
                "✅ PASSED"
            } else {
                "❌ FAILED"
            };

            report.push_str(&format!(
                "💾 Memory Performance: {}\n\
                 ├── Average Model Memory: {:.1}MB\n\
                 ├── Peak Model Memory: {:.1}MB\n\
                 ├── Average Total Memory: {:.1}MB\n\
                 └── Snapshots: {} samples\n\n",
                memory_status,
                memory.avg_model_memory_mb,
                memory.max_model_memory_mb,
                memory.avg_total_memory_mb,
                memory.snapshots_count
            ));
        }

        // Throughput results
        if let Some(throughput) = summary.throughput_stats {
            let throughput_status = if throughput.avg_throughput_sentences_per_sec
                >= summary.targets.min_throughput_sentences_per_sec
            {
                "✅ PASSED"
            } else {
                "❌ FAILED"
            };

            report.push_str(&format!(
                "🚀 Throughput Performance: {}\n\
                 ├── Average: {:.0} sentences/sec\n\
                 ├── Peak: {:.0} sentences/sec\n\
                 └── Measurements: {} samples\n\n",
                throughput_status,
                throughput.avg_throughput_sentences_per_sec,
                throughput.max_throughput_sentences_per_sec,
                throughput.measurements_count
            ));
        }

        // Violations summary
        let violations = &summary.violation_summary;
        report.push_str(&format!(
            "⚠️  Target Violations:\n\
             ├── Total: {}\n\
             ├── Critical: {}\n\
             ├── Moderate: {}\n\
             └── Minor: {}\n\n",
            violations.total_violations,
            violations.critical_violations,
            violations.moderate_violations,
            violations.minor_violations
        ));

        let overall_status = if self.are_targets_met() {
            "✅ ALL TARGETS MET"
        } else {
            "❌ TARGETS NOT MET"
        };

        report.push_str(&format!("🏆 Overall Status: {overall_status}\n"));

        report
    }
}

/// Performance summary statistics
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceSummary {
    pub targets: PerformanceTargets,
    pub latency_stats: Option<LatencyStats>,
    pub memory_stats: Option<MemoryStats>,
    pub throughput_stats: Option<ThroughputStats>,
    pub violation_summary: ViolationSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct LatencyStats {
    pub avg_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub measurements_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryStats {
    pub avg_model_memory_mb: f64,
    pub max_model_memory_mb: f64,
    pub avg_total_memory_mb: f64,
    pub snapshots_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ThroughputStats {
    pub avg_throughput_sentences_per_sec: f64,
    pub max_throughput_sentences_per_sec: f64,
    pub measurements_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ViolationSummary {
    pub total_violations: usize,
    pub critical_violations: usize,
    pub moderate_violations: usize,
    pub minor_violations: usize,
}

/// Calculate percentile from a vector of values
fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let index = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_target_monitor_creation() {
        let monitor = PerformanceTargetMonitor::new_with_defaults();
        assert_eq!(monitor.targets.max_latency_ms, 1.0);
        assert_eq!(monitor.targets.max_memory_mb, 100.0);
        assert_eq!(monitor.targets.min_throughput_sentences_per_sec, 1000.0);
    }

    #[test]
    fn test_latency_recording() {
        let monitor = PerformanceTargetMonitor::new_with_defaults();

        let result = monitor.record_latency(
            "hello world",
            Duration::from_millis(1),
            5,
            LanguageCode::EnUs,
        );

        assert!(result.is_ok());

        let summary = monitor.get_performance_summary();
        assert!(summary.latency_stats.is_some());
        assert_eq!(summary.latency_stats.unwrap().measurements_count, 1);
    }

    #[test]
    fn test_memory_recording() {
        let monitor = PerformanceTargetMonitor::new_with_defaults();

        let result = monitor.record_memory_usage(
            150.0, // total
            80.0,  // model
            20.0,  // cache
            50.0,  // working
            LanguageCode::EnUs,
        );

        assert!(result.is_ok());

        let summary = monitor.get_performance_summary();
        assert!(summary.memory_stats.is_some());
        assert_eq!(summary.memory_stats.unwrap().snapshots_count, 1);
    }

    #[test]
    fn test_violation_detection() {
        let monitor = PerformanceTargetMonitor::new_with_defaults();

        // Record a latency that exceeds the target
        let _ = monitor.record_latency(
            "very long sentence that should take longer to process",
            Duration::from_millis(5), // > 1ms target
            20,
            LanguageCode::EnUs,
        );

        let summary = monitor.get_performance_summary();
        assert!(summary.violation_summary.total_violations > 0);
        assert!(!monitor.are_targets_met());
    }

    #[test]
    fn test_percentile_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(&values, 50.0), 6.0); // index = (0.5 * 9).round() = 5, so values[5] = 6.0
        assert_eq!(percentile(&values, 90.0), 9.0); // index = (0.9 * 9).round() = 8, so values[8] = 9.0
        assert_eq!(percentile(&values, 95.0), 10.0);
    }
}
