//! Automated Benchmark Update System
//!
//! This module provides automated benchmark baseline management, historical tracking,
//! and performance regression detection for the VoiRS evaluation system.

use crate::EvaluationError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

/// Errors specific to automated benchmark updates
#[derive(Error, Debug)]
pub enum BenchmarkError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Benchmark not found: {0}")]
    BenchmarkNotFound(String),
    #[error("Invalid baseline data: {0}")]
    InvalidBaseline(String),
    #[error("Performance regression detected: {0}")]
    PerformanceRegression(String),
}

/// Benchmark measurement data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMeasurement {
    /// Name of the benchmark
    pub name: String,
    /// Value of the measurement (e.g., execution time, accuracy score)
    pub value: f64,
    /// Unit of measurement (e.g., "ms", "accuracy", "MB/s")
    pub unit: String,
    /// Timestamp when measurement was taken
    pub timestamp: u64,
    /// Git commit hash (if available)
    pub commit_hash: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Historical benchmark data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkHistory {
    /// Benchmark name
    pub name: String,
    /// All measurements for this benchmark
    pub measurements: Vec<BenchmarkMeasurement>,
    /// Current baseline value
    pub baseline: f64,
    /// Threshold for detecting regressions (percentage)
    pub regression_threshold: f64,
    /// Threshold for detecting improvements (percentage)
    pub improvement_threshold: f64,
}

/// Configuration for automated benchmark updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Directory to store benchmark data
    pub data_directory: PathBuf,
    /// Minimum number of measurements before updating baseline
    pub min_measurements_for_update: usize,
    /// Maximum age of measurements to consider (in days)
    pub max_measurement_age_days: u64,
    /// Default regression threshold (percentage)
    pub default_regression_threshold: f64,
    /// Default improvement threshold (percentage)
    pub default_improvement_threshold: f64,
    /// Whether to automatically update baselines
    pub auto_update_baselines: bool,
    /// Whether to generate reports
    pub generate_reports: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            data_directory: PathBuf::from("benchmark_data"),
            min_measurements_for_update: 5,
            max_measurement_age_days: 30,
            default_regression_threshold: 10.0, // 10% regression
            default_improvement_threshold: 5.0, // 5% improvement
            auto_update_baselines: true,
            generate_reports: true,
        }
    }
}

/// Automated benchmark update manager
pub struct AutomatedBenchmarkManager {
    config: BenchmarkConfig,
    histories: HashMap<String, BenchmarkHistory>,
}

impl AutomatedBenchmarkManager {
    /// Create a new benchmark manager
    pub fn new(config: BenchmarkConfig) -> Result<Self, BenchmarkError> {
        std::fs::create_dir_all(&config.data_directory)?;

        let mut manager = Self {
            config,
            histories: HashMap::new(),
        };

        manager.load_histories()?;
        Ok(manager)
    }

    /// Load benchmark histories from disk
    fn load_histories(&mut self) -> Result<(), BenchmarkError> {
        let data_dir = &self.config.data_directory;

        if !data_dir.exists() {
            return Ok(());
        }

        for entry in std::fs::read_dir(data_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    let contents = std::fs::read_to_string(&path)?;
                    let history: BenchmarkHistory = serde_json::from_str(&contents)?;
                    self.histories.insert(stem.to_string(), history);
                }
            }
        }

        Ok(())
    }

    /// Save benchmark histories to disk
    fn save_histories(&self) -> Result<(), BenchmarkError> {
        for (name, history) in &self.histories {
            let file_path = self.config.data_directory.join(format!("{}.json", name));
            let contents = serde_json::to_string_pretty(history)?;
            std::fs::write(file_path, contents)?;
        }

        Ok(())
    }

    /// Add a new benchmark measurement
    pub fn add_measurement(
        &mut self,
        measurement: BenchmarkMeasurement,
    ) -> Result<(), BenchmarkError> {
        let name = measurement.name.clone();
        let auto_update_baselines = self.config.auto_update_baselines;
        let max_measurement_age_days = self.config.max_measurement_age_days;
        let min_measurements_for_update = self.config.min_measurements_for_update;

        // Check if we need to create a new history entry
        let is_new_history = !self.histories.contains_key(&name);

        if is_new_history {
            let new_history = BenchmarkHistory {
                name: name.clone(),
                measurements: Vec::new(),
                baseline: measurement.value,
                regression_threshold: self.config.default_regression_threshold,
                improvement_threshold: self.config.default_improvement_threshold,
            };
            self.histories.insert(name.clone(), new_history);
        }

        // Get the history and check for regression
        if let Some(history) = self.histories.get(&name) {
            self.check_regression(&measurement, history)?;
        }

        // Now safely get mutable reference and add measurement
        if let Some(history) = self.histories.get_mut(&name) {
            history.measurements.push(measurement);

            // Clean old measurements
            Self::clean_old_measurements_static(history, max_measurement_age_days);

            // Update baseline if needed
            if auto_update_baselines {
                Self::update_baseline_static(history, min_measurements_for_update)?;
            }
        }

        // Save to disk
        self.save_histories()?;

        Ok(())
    }

    /// Check for performance regression
    fn check_regression(
        &self,
        measurement: &BenchmarkMeasurement,
        history: &BenchmarkHistory,
    ) -> Result<(), BenchmarkError> {
        if history.measurements.is_empty() {
            return Ok(());
        }

        let baseline = history.baseline;
        let current = measurement.value;

        // Calculate percentage change (assuming lower is better for most metrics)
        let percentage_change = ((current - baseline) / baseline) * 100.0;

        if percentage_change > history.regression_threshold {
            return Err(BenchmarkError::PerformanceRegression(format!(
                "Benchmark '{}' shows {:.2}% regression (current: {:.4}, baseline: {:.4})",
                measurement.name, percentage_change, current, baseline
            )));
        }

        Ok(())
    }

    /// Update baseline if improvement is detected
    fn update_baseline(&mut self, history: &mut BenchmarkHistory) -> Result<(), BenchmarkError> {
        if history.measurements.len() < self.config.min_measurements_for_update {
            return Ok(());
        }

        // Calculate statistics from recent measurements
        let recent_measurements: Vec<f64> = history
            .measurements
            .iter()
            .rev()
            .take(self.config.min_measurements_for_update)
            .map(|m| m.value)
            .collect();

        let average = recent_measurements.iter().sum::<f64>() / recent_measurements.len() as f64;
        let improvement = ((history.baseline - average) / history.baseline) * 100.0;

        if improvement > history.improvement_threshold {
            println!(
                "🚀 Updating baseline for '{}': {:.4} → {:.4} ({:.2}% improvement)",
                history.name, history.baseline, average, improvement
            );
            history.baseline = average;
        }

        Ok(())
    }

    /// Clean old measurements based on configuration
    fn clean_old_measurements(&mut self, history: &mut BenchmarkHistory) {
        Self::clean_old_measurements_static(history, self.config.max_measurement_age_days);
    }

    /// Static helper to clean old measurements
    fn clean_old_measurements_static(
        history: &mut BenchmarkHistory,
        max_measurement_age_days: u64,
    ) {
        let cutoff_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - (max_measurement_age_days * 24 * 60 * 60);

        history.measurements.retain(|m| m.timestamp > cutoff_time);
    }

    /// Static helper to update baseline
    fn update_baseline_static(
        history: &mut BenchmarkHistory,
        min_measurements_for_update: usize,
    ) -> Result<(), BenchmarkError> {
        if history.measurements.len() < min_measurements_for_update {
            return Ok(());
        }

        // Calculate statistics from recent measurements
        let recent_measurements: Vec<f64> = history
            .measurements
            .iter()
            .rev()
            .take(min_measurements_for_update)
            .map(|m| m.value)
            .collect();

        if recent_measurements.is_empty() {
            return Ok(());
        }

        // Calculate mean of recent measurements
        let sum: f64 = recent_measurements.iter().sum();
        let mean = sum / recent_measurements.len() as f64;

        // Calculate variance to check for stability
        let variance: f64 = recent_measurements
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / recent_measurements.len() as f64;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if mean != 0.0 {
            std_dev / mean.abs()
        } else {
            0.0
        };

        // Only update baseline if measurements are stable (low coefficient of variation)
        if coefficient_of_variation < 0.1 {
            // 10% threshold for stability
            history.baseline = mean;
        }

        Ok(())
    }

    /// Get benchmark history
    pub fn get_history(&self, name: &str) -> Option<&BenchmarkHistory> {
        self.histories.get(name)
    }

    /// Generate a performance report
    pub fn generate_report(&self) -> Result<String, BenchmarkError> {
        let mut report = String::new();
        report.push_str("# Automated Benchmark Report\n\n");

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        report.push_str(&format!(
            "Generated: {}\n\n",
            chrono::DateTime::from_timestamp(now as i64, 0).unwrap()
        ));

        for (name, history) in &self.histories {
            report.push_str(&format!("## Benchmark: {}\n\n", name));
            report.push_str(&format!(
                "- **Current Baseline**: {:.4}\n",
                history.baseline
            ));
            report.push_str(&format!(
                "- **Total Measurements**: {}\n",
                history.measurements.len()
            ));

            if let Some(latest) = history.measurements.last() {
                let change = ((latest.value - history.baseline) / history.baseline) * 100.0;
                report.push_str(&format!(
                    "- **Latest Measurement**: {:.4} ({:+.2}%)\n",
                    latest.value, change
                ));
            }

            // Calculate trend
            if history.measurements.len() >= 5 {
                let recent_avg: f64 = history
                    .measurements
                    .iter()
                    .rev()
                    .take(5)
                    .map(|m| m.value)
                    .sum::<f64>()
                    / 5.0;

                let older_avg: f64 = if history.measurements.len() >= 10 {
                    history
                        .measurements
                        .iter()
                        .rev()
                        .skip(5)
                        .take(5)
                        .map(|m| m.value)
                        .sum::<f64>()
                        / 5.0
                } else {
                    history.baseline
                };

                let trend = ((recent_avg - older_avg) / older_avg) * 100.0;
                let trend_indicator = if trend > 2.0 {
                    "📈 Improving"
                } else if trend < -2.0 {
                    "📉 Declining"
                } else {
                    "➡️  Stable"
                };

                report.push_str(&format!(
                    "- **Trend**: {} ({:+.2}%)\n",
                    trend_indicator, trend
                ));
            }

            report.push_str("\n");
        }

        Ok(report)
    }

    /// Export benchmark data for external analysis
    pub fn export_data(&self, format: &str) -> Result<String, BenchmarkError> {
        match format.to_lowercase().as_str() {
            "json" => Ok(serde_json::to_string_pretty(&self.histories)?),
            "csv" => {
                let mut csv = String::new();
                csv.push_str("benchmark,timestamp,value,unit,commit_hash\n");

                for history in self.histories.values() {
                    for measurement in &history.measurements {
                        csv.push_str(&format!(
                            "{},{},{},{},{}\n",
                            measurement.name,
                            measurement.timestamp,
                            measurement.value,
                            measurement.unit,
                            measurement.commit_hash.as_deref().unwrap_or("unknown")
                        ));
                    }
                }

                Ok(csv)
            }
            _ => Err(BenchmarkError::InvalidBaseline(format!(
                "Unsupported export format: {}",
                format
            ))),
        }
    }
}

/// Helper function to get current timestamp
pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Helper function to get git commit hash
pub fn get_git_commit_hash() -> Option<String> {
    std::process::Command::new("git")
        .args(&["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_config() -> (BenchmarkConfig, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let config = BenchmarkConfig {
            data_directory: temp_dir.path().to_path_buf(),
            min_measurements_for_update: 3,
            max_measurement_age_days: 1,
            default_regression_threshold: 15.0,
            default_improvement_threshold: 10.0,
            auto_update_baselines: true,
            generate_reports: true,
        };
        (config, temp_dir)
    }

    #[test]
    fn test_benchmark_manager_creation() {
        let (config, _temp_dir) = create_test_config();
        let manager = AutomatedBenchmarkManager::new(config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_add_measurement() {
        let (config, _temp_dir) = create_test_config();
        let mut manager = AutomatedBenchmarkManager::new(config).unwrap();

        let measurement = BenchmarkMeasurement {
            name: "test_benchmark".to_string(),
            value: 100.0,
            unit: "ms".to_string(),
            timestamp: current_timestamp(),
            commit_hash: Some("abc123".to_string()),
            metadata: HashMap::new(),
        };

        let result = manager.add_measurement(measurement);
        assert!(result.is_ok());
        assert!(manager.get_history("test_benchmark").is_some());
    }

    #[test]
    fn test_regression_detection() {
        let (config, _temp_dir) = create_test_config();
        let mut manager = AutomatedBenchmarkManager::new(config).unwrap();

        // Add baseline measurement
        let baseline = BenchmarkMeasurement {
            name: "test_benchmark".to_string(),
            value: 100.0,
            unit: "ms".to_string(),
            timestamp: current_timestamp(),
            commit_hash: None,
            metadata: HashMap::new(),
        };
        manager.add_measurement(baseline).unwrap();

        // Add regression measurement (20% worse)
        let regression = BenchmarkMeasurement {
            name: "test_benchmark".to_string(),
            value: 120.0,
            unit: "ms".to_string(),
            timestamp: current_timestamp(),
            commit_hash: None,
            metadata: HashMap::new(),
        };

        let result = manager.add_measurement(regression);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BenchmarkError::PerformanceRegression(_)
        ));
    }

    #[test]
    fn test_baseline_update() {
        let (config, _temp_dir) = create_test_config();
        let mut manager = AutomatedBenchmarkManager::new(config).unwrap();

        // Add several improving measurements
        for i in 0..5 {
            let measurement = BenchmarkMeasurement {
                name: "test_benchmark".to_string(),
                value: 100.0 - (i as f64 * 2.0), // Improving performance
                unit: "ms".to_string(),
                timestamp: current_timestamp(),
                commit_hash: None,
                metadata: HashMap::new(),
            };
            manager.add_measurement(measurement).unwrap();
        }

        let history = manager.get_history("test_benchmark").unwrap();
        assert!(history.baseline < 100.0); // Baseline should be updated
    }

    #[test]
    fn test_report_generation() {
        let (config, _temp_dir) = create_test_config();
        let mut manager = AutomatedBenchmarkManager::new(config).unwrap();

        let measurement = BenchmarkMeasurement {
            name: "test_benchmark".to_string(),
            value: 100.0,
            unit: "ms".to_string(),
            timestamp: current_timestamp(),
            commit_hash: None,
            metadata: HashMap::new(),
        };
        manager.add_measurement(measurement).unwrap();

        let report = manager.generate_report().unwrap();
        assert!(report.contains("Automated Benchmark Report"));
        assert!(report.contains("test_benchmark"));
    }

    #[test]
    fn test_data_export() {
        let (config, _temp_dir) = create_test_config();
        let mut manager = AutomatedBenchmarkManager::new(config).unwrap();

        let measurement = BenchmarkMeasurement {
            name: "test_benchmark".to_string(),
            value: 100.0,
            unit: "ms".to_string(),
            timestamp: current_timestamp(),
            commit_hash: Some("abc123".to_string()),
            metadata: HashMap::new(),
        };
        manager.add_measurement(measurement).unwrap();

        // Test JSON export
        let json_export = manager.export_data("json").unwrap();
        assert!(json_export.contains("test_benchmark"));

        // Test CSV export
        let csv_export = manager.export_data("csv").unwrap();
        assert!(csv_export.contains("benchmark,timestamp,value,unit,commit_hash"));
        assert!(csv_export.contains("test_benchmark"));
    }
}
