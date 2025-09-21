//! Automated regression detection for performance benchmarks.
//!
//! This module provides functionality to detect performance regressions
//! by comparing benchmark results across different runs and versions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Represents a single benchmark measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMeasurement {
    /// Name of the benchmark measurement
    pub name: String,
    /// Measured value
    pub value: f64,
    /// Unit of measurement (e.g., "ms", "MB/s")
    pub unit: String,
    /// Unix timestamp when measurement was taken
    pub timestamp: u64,
    /// Git commit hash when measurement was taken
    pub git_commit: Option<String>,
    /// Version of the software when measurement was taken
    pub version: String,
}

/// Performance regression detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult {
    /// Name of the measurement that was analyzed
    pub measurement_name: String,
    /// Current measured value
    pub current_value: f64,
    /// Baseline value for comparison
    pub baseline_value: f64,
    /// Percentage change from baseline (positive indicates regression)
    pub change_percentage: f64,
    /// Whether this constitutes a regression
    pub is_regression: bool,
    /// Severity level of the regression
    pub severity: RegressionSeverity,
    /// Whether regression threshold was exceeded
    pub threshold_exceeded: bool,
}

/// Severity level for performance regressions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegressionSeverity {
    /// Minor performance regression (5-10% performance loss)
    Minor,
    /// Major performance regression (10-25% performance loss)
    Major,
    /// Critical performance regression (>25% performance loss)
    Critical,
}

/// Configuration for regression detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionConfig {
    /// Threshold for minor regressions (5% by default)
    pub minor_threshold: f64,
    /// Threshold for major regressions (10% by default)
    pub major_threshold: f64,
    /// Threshold for critical regressions (25% by default)
    pub critical_threshold: f64,
    /// Number of previous measurements to consider for baseline
    pub baseline_window: usize,
    /// Minimum samples required for regression detection
    pub min_samples: usize,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            minor_threshold: 0.05,
            major_threshold: 0.10,
            critical_threshold: 0.25,
            baseline_window: 10,
            min_samples: 3,
        }
    }
}

/// Performance regression detector
pub struct RegressionDetector {
    config: RegressionConfig,
    measurements: Vec<BenchmarkMeasurement>,
    baseline_cache: HashMap<String, f64>,
}

impl RegressionDetector {
    /// Create a new regression detector with default configuration
    pub fn new() -> Self {
        Self {
            config: RegressionConfig::default(),
            measurements: Vec::new(),
            baseline_cache: HashMap::new(),
        }
    }

    /// Create a new regression detector with custom configuration
    pub fn with_config(config: RegressionConfig) -> Self {
        Self {
            config,
            measurements: Vec::new(),
            baseline_cache: HashMap::new(),
        }
    }

    /// Add a new benchmark measurement
    pub fn add_measurement(&mut self, measurement: BenchmarkMeasurement) {
        // Invalidate baseline cache for this measurement name
        self.baseline_cache.remove(&measurement.name);
        self.measurements.push(measurement);
    }

    /// Load measurements from a JSON file
    pub fn load_from_file<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let measurements: Vec<BenchmarkMeasurement> = serde_json::from_str(&content)?;
        self.measurements.extend(measurements);
        self.baseline_cache.clear();
        Ok(())
    }

    /// Save measurements to a JSON file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_json::to_string_pretty(&self.measurements)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Detect regressions for a specific measurement name
    pub fn detect_regression(&mut self, measurement_name: &str) -> Option<RegressionResult> {
        // First, calculate baseline to ensure it's cached
        let baseline = self.calculate_baseline(measurement_name)?;

        // Then filter measurements
        let measurements: Vec<_> = self
            .measurements
            .iter()
            .filter(|m| m.name == measurement_name)
            .collect();

        if measurements.len() < self.config.min_samples {
            return None;
        }

        // Get the most recent measurement
        let current = measurements.last()?;
        let current_value = current.value;

        // Calculate change percentage (positive means performance degradation)
        let change_percentage = (current_value - baseline) / baseline;

        // Determine if this is a regression and its severity
        let is_regression = change_percentage > self.config.minor_threshold;
        let severity = if change_percentage > self.config.critical_threshold {
            RegressionSeverity::Critical
        } else if change_percentage > self.config.major_threshold {
            RegressionSeverity::Major
        } else {
            RegressionSeverity::Minor
        };

        let threshold_exceeded = change_percentage > self.config.minor_threshold;

        Some(RegressionResult {
            measurement_name: measurement_name.to_string(),
            current_value,
            baseline_value: baseline,
            change_percentage,
            is_regression,
            severity,
            threshold_exceeded,
        })
    }

    /// Detect regressions for all measurements
    pub fn detect_all_regressions(&mut self) -> Vec<RegressionResult> {
        let mut measurement_names: Vec<String> =
            self.measurements.iter().map(|m| m.name.clone()).collect();
        measurement_names.sort();
        measurement_names.dedup();

        measurement_names
            .into_iter()
            .filter_map(|name| self.detect_regression(&name))
            .collect()
    }

    /// Get a reference to all measurements
    pub fn measurements(&self) -> &[BenchmarkMeasurement] {
        &self.measurements
    }

    /// Calculate baseline value for a measurement (average of recent measurements)
    fn calculate_baseline(&mut self, measurement_name: &str) -> Option<f64> {
        if let Some(cached_baseline) = self.baseline_cache.get(measurement_name) {
            return Some(*cached_baseline);
        }

        let mut measurements: Vec<_> = self
            .measurements
            .iter()
            .filter(|m| m.name == measurement_name)
            .collect();

        if measurements.len() < self.config.min_samples {
            return None;
        }

        // Sort by timestamp (most recent first)
        measurements.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        // Take the baseline window (excluding the most recent measurement)
        let baseline_measurements = &measurements[1..];
        let window_size = self.config.baseline_window.min(baseline_measurements.len());

        if window_size == 0 {
            return None;
        }

        let baseline_values: Vec<f64> = baseline_measurements
            .iter()
            .take(window_size)
            .map(|m| m.value)
            .collect();

        // Calculate average baseline
        let sum: f64 = baseline_values.iter().sum();
        let average = sum / baseline_values.len() as f64;

        // Cache the result
        self.baseline_cache
            .insert(measurement_name.to_string(), average);

        Some(average)
    }

    /// Create a benchmark measurement from current timestamp
    pub fn create_measurement(
        name: String,
        value: f64,
        unit: String,
        git_commit: Option<String>,
        version: String,
    ) -> BenchmarkMeasurement {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        BenchmarkMeasurement {
            name,
            value,
            unit,
            timestamp,
            git_commit,
            version,
        }
    }

    /// Generate a regression report
    pub fn generate_report(&mut self) -> String {
        let regressions = self.detect_all_regressions();

        if regressions.is_empty() {
            return "No performance regressions detected.\n".to_string();
        }

        let mut report = String::new();
        report.push_str("Performance Regression Report\n");
        report.push_str("==============================\n\n");

        // Group by severity
        let critical: Vec<_> = regressions
            .iter()
            .filter(|r| r.severity == RegressionSeverity::Critical)
            .collect();
        let major: Vec<_> = regressions
            .iter()
            .filter(|r| r.severity == RegressionSeverity::Major)
            .collect();
        let minor: Vec<_> = regressions
            .iter()
            .filter(|r| r.severity == RegressionSeverity::Minor)
            .collect();

        if !critical.is_empty() {
            report.push_str("🔴 CRITICAL REGRESSIONS:\n");
            for regression in critical {
                report.push_str(&format!(
                    "  - {}: {:.2}% slower (was {:.2}ms, now {:.2}ms)\n",
                    regression.measurement_name,
                    regression.change_percentage * 100.0,
                    regression.baseline_value,
                    regression.current_value
                ));
            }
            report.push('\n');
        }

        if !major.is_empty() {
            report.push_str("🟡 MAJOR REGRESSIONS:\n");
            for regression in major {
                report.push_str(&format!(
                    "  - {}: {:.2}% slower (was {:.2}ms, now {:.2}ms)\n",
                    regression.measurement_name,
                    regression.change_percentage * 100.0,
                    regression.baseline_value,
                    regression.current_value
                ));
            }
            report.push('\n');
        }

        if !minor.is_empty() {
            report.push_str("🟢 MINOR REGRESSIONS:\n");
            for regression in minor {
                report.push_str(&format!(
                    "  - {}: {:.2}% slower (was {:.2}ms, now {:.2}ms)\n",
                    regression.measurement_name,
                    regression.change_percentage * 100.0,
                    regression.baseline_value,
                    regression.current_value
                ));
            }
            report.push('\n');
        }

        report
    }
}

impl Default for RegressionDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_regression_detector_creation() {
        let detector = RegressionDetector::new();
        assert_eq!(detector.measurements.len(), 0);
        assert_eq!(detector.baseline_cache.len(), 0);
    }

    #[test]
    fn test_add_measurement() {
        let mut detector = RegressionDetector::new();
        let measurement = RegressionDetector::create_measurement(
            "test_metric".to_string(),
            100.0,
            "ms".to_string(),
            Some("abc123".to_string()),
            "1.0.0".to_string(),
        );
        detector.add_measurement(measurement);
        assert_eq!(detector.measurements.len(), 1);
    }

    #[test]
    fn test_regression_detection() {
        let mut detector = RegressionDetector::new();

        // Add baseline measurements
        for i in 0..5 {
            let measurement = RegressionDetector::create_measurement(
                "test_metric".to_string(),
                100.0 + (i as f64),
                "ms".to_string(),
                Some(format!("commit{}", i)),
                "1.0.0".to_string(),
            );
            detector.add_measurement(measurement);
            // Small delay to ensure different timestamps
            thread::sleep(Duration::from_millis(1));
        }

        // Add a measurement that represents a regression
        let regression_measurement = RegressionDetector::create_measurement(
            "test_metric".to_string(),
            120.0, // 20% slower
            "ms".to_string(),
            Some("regression_commit".to_string()),
            "1.0.1".to_string(),
        );
        detector.add_measurement(regression_measurement);

        let result = detector.detect_regression("test_metric");
        assert!(result.is_some());

        let regression = result.unwrap();
        assert!(regression.is_regression);
        assert_eq!(regression.severity, RegressionSeverity::Major);
        assert!(regression.change_percentage > 0.10);
    }

    #[test]
    fn test_no_regression_with_insufficient_data() {
        let mut detector = RegressionDetector::new();
        let measurement = RegressionDetector::create_measurement(
            "test_metric".to_string(),
            100.0,
            "ms".to_string(),
            Some("abc123".to_string()),
            "1.0.0".to_string(),
        );
        detector.add_measurement(measurement);

        let result = detector.detect_regression("test_metric");
        assert!(result.is_none());
    }

    #[test]
    fn test_regression_severity_levels() {
        let config = RegressionConfig {
            minor_threshold: 0.05,
            major_threshold: 0.10,
            critical_threshold: 0.25,
            baseline_window: 5,
            min_samples: 3,
        };

        let mut detector = RegressionDetector::with_config(config);

        // Add baseline measurements
        for i in 0..4 {
            let measurement = RegressionDetector::create_measurement(
                "test_metric".to_string(),
                100.0,
                "ms".to_string(),
                Some(format!("commit{}", i)),
                "1.0.0".to_string(),
            );
            detector.add_measurement(measurement);
            thread::sleep(Duration::from_millis(1));
        }

        // Test critical regression (40% slower than baseline)
        let critical_measurement = RegressionDetector::create_measurement(
            "test_metric".to_string(),
            150.0, // This should result in > 25% change even with baseline averaging
            "ms".to_string(),
            Some("critical_commit".to_string()),
            "1.0.1".to_string(),
        );
        detector.add_measurement(critical_measurement);

        let result = detector.detect_regression("test_metric");
        assert!(result.is_some());
        let regression = result.unwrap();
        // The change should be significant enough to be classified as Critical
        assert!(regression.change_percentage > 0.25); // Should be > 25%
        assert_eq!(regression.severity, RegressionSeverity::Critical);
    }

    #[test]
    fn test_report_generation() {
        let mut detector = RegressionDetector::new();

        // Add measurements that will trigger regressions
        for i in 0..4 {
            let measurement = RegressionDetector::create_measurement(
                "test_metric".to_string(),
                100.0,
                "ms".to_string(),
                Some(format!("commit{}", i)),
                "1.0.0".to_string(),
            );
            detector.add_measurement(measurement);
            thread::sleep(Duration::from_millis(1));
        }

        // Add regression
        let regression_measurement = RegressionDetector::create_measurement(
            "test_metric".to_string(),
            115.0,
            "ms".to_string(),
            Some("regression_commit".to_string()),
            "1.0.1".to_string(),
        );
        detector.add_measurement(regression_measurement);

        let report = detector.generate_report();
        assert!(report.contains("MAJOR REGRESSIONS"));
        assert!(report.contains("test_metric"));
    }
}
