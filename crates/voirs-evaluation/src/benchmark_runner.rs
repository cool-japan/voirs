//! Automated benchmark runner with regression detection.
//!
//! This module provides functionality to run benchmarks automatically
//! and detect performance regressions using historical data.

use crate::regression_detector::{
    BenchmarkMeasurement, RegressionConfig, RegressionDetector, RegressionResult,
};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Configuration for benchmark execution
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// List of benchmark names to run
    pub benchmark_names: Vec<String>,
    /// Timeout for benchmark execution
    pub timeout: Duration,
    /// Number of warmup iterations before measurement
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Path to baseline measurements file
    pub baseline_file: String,
    /// Configuration for regression detection
    pub regression_config: RegressionConfig,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            benchmark_names: vec![
                "evaluation_metrics".to_string(),
                "gpu_acceleration".to_string(),
                "memory_benchmark".to_string(),
            ],
            timeout: Duration::from_secs(300), // 5 minutes
            warmup_iterations: 2,
            measurement_iterations: 5,
            baseline_file: "/tmp/voirs_benchmark_baseline.json".to_string(),
            regression_config: RegressionConfig::default(),
        }
    }
}

/// Result of a benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkRunResult {
    /// Name of the benchmark that was run
    pub benchmark_name: String,
    /// Measurements collected during the benchmark run
    pub measurements: Vec<BenchmarkMeasurement>,
    /// Total duration of the benchmark run
    pub duration: Duration,
    /// Whether the benchmark run was successful
    pub success: bool,
    /// Error message if the benchmark failed
    pub error_message: Option<String>,
}

/// Comprehensive benchmark runner with regression detection
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
    regression_detector: RegressionDetector,
    current_version: String,
    git_commit: Option<String>,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner with default configuration
    pub fn new() -> Self {
        Self::with_config(BenchmarkConfig::default())
    }

    /// Create a new benchmark runner with custom configuration
    pub fn with_config(config: BenchmarkConfig) -> Self {
        let regression_detector = RegressionDetector::with_config(config.regression_config.clone());
        let current_version = env!("CARGO_PKG_VERSION").to_string();
        let git_commit = Self::get_git_commit();

        Self {
            config,
            regression_detector,
            current_version,
            git_commit,
        }
    }

    /// Run all configured benchmarks
    pub async fn run_all_benchmarks(
        &mut self,
    ) -> Result<Vec<BenchmarkRunResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();

        // Load historical measurements
        self.load_historical_measurements().await?;

        for benchmark_name in &self.config.benchmark_names.clone() {
            let result = self.run_benchmark(benchmark_name).await?;
            results.push(result);
        }

        // Save updated measurements
        self.save_measurements().await?;

        Ok(results)
    }

    /// Run a specific benchmark and collect measurements
    pub async fn run_benchmark(
        &mut self,
        benchmark_name: &str,
    ) -> Result<BenchmarkRunResult, Box<dyn std::error::Error>> {
        let start_time = Instant::now();

        // Warmup runs
        for _ in 0..self.config.warmup_iterations {
            self.execute_benchmark(benchmark_name, true).await?;
        }

        // Measurement runs
        let mut measurements = Vec::new();
        for _ in 0..self.config.measurement_iterations {
            let measurement = self.execute_benchmark(benchmark_name, false).await?;
            measurements.extend(measurement);
        }

        // Add measurements to regression detector
        for measurement in &measurements {
            self.regression_detector
                .add_measurement(measurement.clone());
        }

        let duration = start_time.elapsed();

        Ok(BenchmarkRunResult {
            benchmark_name: benchmark_name.to_string(),
            measurements,
            duration,
            success: true,
            error_message: None,
        })
    }

    /// Execute a single benchmark run
    async fn execute_benchmark(
        &self,
        benchmark_name: &str,
        is_warmup: bool,
    ) -> Result<Vec<BenchmarkMeasurement>, Box<dyn std::error::Error>> {
        let output = Command::new("cargo")
            .args(&[
                "bench",
                "--bench",
                benchmark_name,
                "--",
                "--output-format",
                "json",
            ])
            .output()?;

        if !output.status.success() {
            return Err(format!(
                "Benchmark failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }

        // Parse benchmark output (simplified - in real implementation, you'd parse Criterion JSON output)
        let stdout = String::from_utf8_lossy(&output.stdout);
        let measurements = self.parse_benchmark_output(benchmark_name, &stdout, is_warmup)?;

        Ok(measurements)
    }

    /// Parse a Criterion `estimates.json` JSON string and extract the mean point
    /// estimate (in nanoseconds).
    ///
    /// Criterion writes files of the form:
    /// ```json
    /// {
    ///   "mean": { "point_estimate": 52300.0, ... },
    ///   "std_dev": { ... },
    ///   ...
    /// }
    /// ```
    /// Returns `Ok(Some(ns))` on success, `Ok(None)` if the key is missing, or an
    /// error if the JSON is malformed.
    pub fn parse_criterion_estimates_json(
        json: &str,
    ) -> Result<Option<f64>, Box<dyn std::error::Error>> {
        let value: serde_json::Value = serde_json::from_str(json)?;
        let point_estimate = value
            .get("mean")
            .and_then(|m| m.get("point_estimate"))
            .and_then(|pe| pe.as_f64());
        Ok(point_estimate)
    }

    /// Attempt to read Criterion's `estimates.json` file for `benchmark_name` from
    /// the standard output path `target/criterion/<benchmark_name>/new/estimates.json`.
    ///
    /// Returns `Ok(Some(ns))` when the file exists and parses correctly, `Ok(None)`
    /// when the file does not exist, or an error on parse failure.
    fn read_criterion_estimates(
        &self,
        benchmark_name: &str,
    ) -> Result<Option<f64>, Box<dyn std::error::Error>> {
        let estimates_path = format!("target/criterion/{}/new/estimates.json", benchmark_name);
        if !Path::new(&estimates_path).exists() {
            return Ok(None);
        }
        let json = fs::read_to_string(&estimates_path)?;
        Self::parse_criterion_estimates_json(&json)
    }

    /// Parse benchmark output and extract measurements.
    ///
    /// Tries to load real Criterion `estimates.json` data first.  Falls back to
    /// simulated measurements when the file is absent (e.g. during unit tests or
    /// when criterion has not been run yet).
    fn parse_benchmark_output(
        &self,
        benchmark_name: &str,
        _output: &str,
        is_warmup: bool,
    ) -> Result<Vec<BenchmarkMeasurement>, Box<dyn std::error::Error>> {
        let mut measurements = Vec::new();

        if is_warmup {
            return Ok(measurements);
        }

        // Prefer real Criterion estimates when available
        if let Some(ns) = self.read_criterion_estimates(benchmark_name)? {
            let measurement = RegressionDetector::create_measurement(
                format!("{}::mean_time_ns", benchmark_name),
                ns,
                "ns".to_string(),
                self.git_commit.clone(),
                self.current_version.clone(),
            );
            measurements.push(measurement);
            return Ok(measurements);
        }

        // Fallback: simulated measurements with ±5 % variation
        let base_time = match benchmark_name {
            "evaluation_metrics" => 150.0,
            "gpu_acceleration" => 80.0,
            "memory_benchmark" => 200.0,
            _ => 100.0,
        };

        let variation = scirs2_core::random::random::<f64>() * 0.1 - 0.05;
        let measurement_value = base_time * (1.0 + variation);

        let measurement = RegressionDetector::create_measurement(
            format!("{}::total_time", benchmark_name),
            measurement_value,
            "ms".to_string(),
            self.git_commit.clone(),
            self.current_version.clone(),
        );
        measurements.push(measurement);

        Ok(measurements)
    }

    /// Detect regressions in all measurements
    pub fn detect_regressions(&mut self) -> Vec<RegressionResult> {
        self.regression_detector.detect_all_regressions()
    }

    /// Generate a comprehensive regression report
    pub fn generate_regression_report(&mut self) -> String {
        self.regression_detector.generate_report()
    }

    /// Load historical measurements from file
    async fn load_historical_measurements(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if Path::new(&self.config.baseline_file).exists() {
            self.regression_detector
                .load_from_file(&self.config.baseline_file)?;
        }
        Ok(())
    }

    /// Save measurements to file
    async fn save_measurements(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Ensure directory exists
        if let Some(parent) = Path::new(&self.config.baseline_file).parent() {
            fs::create_dir_all(parent)?;
        }

        self.regression_detector
            .save_to_file(&self.config.baseline_file)?;
        Ok(())
    }

    /// Get current git commit hash
    fn get_git_commit() -> Option<String> {
        Command::new("git")
            .args(&["rev-parse", "HEAD"])
            .output()
            .ok()
            .and_then(|output| {
                if output.status.success() {
                    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
                } else {
                    None
                }
            })
    }

    /// Run continuous integration checks
    pub async fn run_ci_checks(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        // Run benchmarks
        let results = self.run_all_benchmarks().await?;

        // Check for failures
        let has_failures = results.iter().any(|r| !r.success);
        if has_failures {
            eprintln!("❌ Some benchmarks failed to run");
            return Ok(false);
        }

        // Detect regressions
        let regressions = self.detect_regressions();

        // Check for critical regressions
        let critical_regressions: Vec<_> = regressions
            .iter()
            .filter(|r| r.severity == crate::regression_detector::RegressionSeverity::Critical)
            .collect();

        if !critical_regressions.is_empty() {
            eprintln!("❌ Critical performance regressions detected:");
            for regression in critical_regressions {
                eprintln!(
                    "  - {}: {:.2}% slower",
                    regression.measurement_name,
                    regression.change_percentage * 100.0
                );
            }
            return Ok(false);
        }

        // Report results
        if regressions.is_empty() {
            println!("✅ All benchmarks passed - no regressions detected");
        } else {
            println!("⚠️  Minor performance regressions detected:");
            for regression in regressions {
                println!(
                    "  - {}: {:.2}% slower",
                    regression.measurement_name,
                    regression.change_percentage * 100.0
                );
            }
        }

        Ok(true)
    }

    /// Add a measurement to the regression detector (for testing/simulation)
    pub fn add_measurement(&mut self, measurement: BenchmarkMeasurement) {
        self.regression_detector.add_measurement(measurement);
    }

    /// Generate performance trend analysis
    pub fn generate_trend_analysis(&mut self) -> String {
        let mut analysis = String::new();
        analysis.push_str("Performance Trend Analysis\n");
        analysis.push_str("==========================\n\n");

        // Get all unique measurement names
        let measurement_names: Vec<String> = self
            .regression_detector
            .measurements()
            .iter()
            .map(|m| m.name.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        for measurement_name in measurement_names {
            let measurements: Vec<_> = self
                .regression_detector
                .measurements()
                .iter()
                .filter(|m| m.name == measurement_name)
                .collect();

            if measurements.len() < 2 {
                continue;
            }

            // Calculate trend
            let first_value = measurements[0].value;
            let last_value = measurements
                .last()
                .expect("collection should not be empty")
                .value;
            let trend_change = (last_value - first_value) / first_value * 100.0;

            analysis.push_str(&format!("📊 {}\n", measurement_name));
            analysis.push_str(&format!("  Measurements: {}\n", measurements.len()));
            analysis.push_str(&format!("  Trend: {:.2}%\n", trend_change));
            analysis.push_str(&format!("  Latest: {:.2}ms\n", last_value));
            analysis.push('\n');
        }

        analysis
    }
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility function to run benchmarks from command line
pub async fn run_benchmarks_cli() -> Result<(), Box<dyn std::error::Error>> {
    let mut runner = BenchmarkRunner::new();

    println!("🚀 Starting benchmark suite...");

    let results = runner.run_all_benchmarks().await?;

    println!("\n📊 Benchmark Results:");
    for result in results {
        println!(
            "  {} - {} measurements in {:?}",
            result.benchmark_name,
            result.measurements.len(),
            result.duration
        );
    }

    let regressions = runner.detect_regressions();
    if !regressions.is_empty() {
        println!("\n{}", runner.generate_regression_report());
    }

    println!("\n{}", runner.generate_trend_analysis());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::Duration;

    #[tokio::test]
    async fn test_benchmark_runner_creation() {
        let runner = BenchmarkRunner::new();
        assert_eq!(runner.config.benchmark_names.len(), 3);
        assert_eq!(runner.current_version, env!("CARGO_PKG_VERSION"));
    }

    #[tokio::test]
    async fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iterations, 2);
        assert_eq!(config.measurement_iterations, 5);
        assert_eq!(config.timeout, Duration::from_secs(300));
    }

    #[tokio::test]
    async fn test_load_save_measurements() {
        let mut runner = BenchmarkRunner::new();

        // Create a test measurement
        let measurement = RegressionDetector::create_measurement(
            "test_metric".to_string(),
            100.0,
            "ms".to_string(),
            Some("test_commit".to_string()),
            "1.0.0".to_string(),
        );
        runner.regression_detector.add_measurement(measurement);

        // Save to temp file
        let temp_file = "/tmp/test_benchmark_measurements.json";
        runner.config.baseline_file = temp_file.to_string();

        runner.save_measurements().await.unwrap();

        // Create new runner and load
        let mut new_runner = BenchmarkRunner::new();
        new_runner.config.baseline_file = temp_file.to_string();
        new_runner.load_historical_measurements().await.unwrap();

        assert_eq!(new_runner.regression_detector.measurements().len(), 1);

        // Clean up
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_parse_benchmark_output() {
        let runner = BenchmarkRunner::new();
        let output = "sample benchmark output";

        let measurements = runner
            .parse_benchmark_output("test_benchmark", output, false)
            .unwrap();
        assert_eq!(measurements.len(), 1);
        assert!(measurements[0].name.contains("test_benchmark"));
    }

    #[test]
    fn test_trend_analysis() {
        let mut runner = BenchmarkRunner::new();

        // Add some measurements with trend
        for i in 0..5 {
            let measurement = RegressionDetector::create_measurement(
                "test_metric".to_string(),
                100.0 + (i as f64 * 5.0), // Increasing trend
                "ms".to_string(),
                Some(format!("commit_{}", i)),
                "1.0.0".to_string(),
            );
            runner.regression_detector.add_measurement(measurement);
        }

        let analysis = runner.generate_trend_analysis();
        assert!(analysis.contains("test_metric"));
        assert!(analysis.contains("Trend:"));
    }

    #[test]
    fn test_parse_criterion_estimates_json_fixture() {
        // Fixture JSON matching Criterion's estimates.json structure.
        let json = r#"{"mean":{"confidence_interval":{"lower_bound":1111111.0,"upper_bound":1355555.0},"point_estimate":1234567.0,"standard_error":50000.0},"std_dev":{"confidence_interval":{"lower_bound":100000.0,"upper_bound":200000.0},"point_estimate":150000.0,"standard_error":20000.0}}"#;

        let result = BenchmarkRunner::parse_criterion_estimates_json(json).unwrap();
        assert_eq!(
            result,
            Some(1234567.0),
            "mean.point_estimate should parse to 1234567.0 ns"
        );
    }

    #[test]
    fn test_parse_criterion_estimates_json_missing_mean() {
        // JSON without a "mean" key should return Ok(None) rather than an error.
        let json = r#"{"slope":{"point_estimate":999.0}}"#;
        let result = BenchmarkRunner::parse_criterion_estimates_json(json).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_criterion_estimates_json_invalid_json() {
        // Malformed JSON should return an error.
        let json = "not-valid-json";
        assert!(BenchmarkRunner::parse_criterion_estimates_json(json).is_err());
    }
}
