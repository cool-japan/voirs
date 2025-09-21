//! Performance regression testing infrastructure
//!
//! This module provides comprehensive performance regression testing capabilities
//! including baseline tracking, automated regression detection, and CI integration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use voirs_recognizer::prelude::*;
use voirs_recognizer::{PerformanceRequirements, PerformanceValidator};

/// Performance benchmark result for regression tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark name/identifier
    pub name: String,
    /// Timestamp when benchmark was run
    pub timestamp: u64,
    /// Git commit hash (if available)
    pub commit_hash: Option<String>,
    /// Real-time factor
    pub rtf: f32,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Startup time in milliseconds
    pub startup_time_ms: u64,
    /// Streaming latency in milliseconds
    pub streaming_latency_ms: u64,
    /// Processing throughput (samples per second)
    pub throughput_samples_per_sec: f64,
    /// CPU utilization percentage
    pub cpu_utilization: f32,
    /// Audio duration in seconds
    pub audio_duration: f32,
    /// Test configuration
    pub config: TestConfiguration,
}

/// Test configuration for different benchmark scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfiguration {
    /// Model size/type being tested
    pub model_type: String,
    /// Audio sample rate
    pub sample_rate: u32,
    /// Audio duration in seconds
    pub duration: f32,
    /// Number of channels
    pub channels: u32,
    /// Additional features enabled
    pub features: Vec<String>,
}

/// Performance regression analysis result
#[derive(Debug, Clone)]
pub struct RegressionAnalysis {
    /// Whether any regressions were detected
    pub has_regressions: bool,
    /// Specific regressions found
    pub regressions: Vec<PerformanceRegression>,
    /// Performance improvements found
    pub improvements: Vec<PerformanceImprovement>,
    /// Overall performance delta
    pub overall_delta: f32,
}

/// Individual performance regression
#[derive(Debug, Clone)]
pub struct PerformanceRegression {
    /// Metric that regressed
    pub metric: String,
    /// Current value
    pub current_value: f64,
    /// Baseline value
    pub baseline_value: f64,
    /// Percentage change (negative = regression)
    pub percentage_change: f32,
    /// Severity level
    pub severity: RegressionSeverity,
}

/// Individual performance improvement
#[derive(Debug, Clone)]
pub struct PerformanceImprovement {
    /// Metric that improved
    pub metric: String,
    /// Current value
    pub current_value: f64,
    /// Baseline value
    pub baseline_value: f64,
    /// Percentage change (positive = improvement)
    pub percentage_change: f32,
}

/// Severity levels for performance regressions
#[derive(Debug, Clone, PartialEq)]
pub enum RegressionSeverity {
    /// Minor regression (5-15% degradation)
    Minor,
    /// Major regression (15-30% degradation)
    Major,
    /// Critical regression (>30% degradation)
    Critical,
}

/// Performance regression testing infrastructure
pub struct RegressionTester {
    /// Performance validator
    validator: PerformanceValidator,
    /// Baseline results storage path
    baseline_path: String,
    /// Historical results storage path
    history_path: String,
    /// Regression thresholds
    thresholds: RegressionThresholds,
}

/// Thresholds for detecting performance regressions
#[derive(Debug, Clone)]
pub struct RegressionThresholds {
    /// RTF regression threshold (percentage)
    pub rtf_threshold: f32,
    /// Memory usage regression threshold (percentage)
    pub memory_threshold: f32,
    /// Startup time regression threshold (percentage)
    pub startup_threshold: f32,
    /// Latency regression threshold (percentage)
    pub latency_threshold: f32,
    /// Throughput regression threshold (percentage)
    pub throughput_threshold: f32,
}

impl Default for RegressionThresholds {
    fn default() -> Self {
        Self {
            rtf_threshold: 15.0,        // 15% RTF increase is concerning
            memory_threshold: 20.0,     // 20% memory increase is concerning
            startup_threshold: 25.0,    // 25% startup time increase is concerning
            latency_threshold: 10.0,    // 10% latency increase is concerning
            throughput_threshold: 10.0, // 10% throughput decrease is concerning
        }
    }
}

impl RegressionTester {
    /// Create new regression tester
    pub fn new() -> Self {
        Self {
            validator: PerformanceValidator::new().with_verbose(false),
            baseline_path: "tests/benchmarks/baseline.json".to_string(),
            history_path: "tests/benchmarks/history.json".to_string(),
            thresholds: RegressionThresholds::default(),
        }
    }

    /// Create new regression tester with custom paths and thresholds
    pub fn with_config(
        baseline_path: String,
        history_path: String,
        thresholds: RegressionThresholds,
    ) -> Self {
        Self {
            validator: PerformanceValidator::new().with_verbose(false),
            baseline_path,
            history_path,
            thresholds,
        }
    }

    /// Run comprehensive performance benchmark
    pub async fn run_benchmark(&self, config: TestConfiguration) -> BenchmarkResult {
        // Create test audio based on configuration
        let sample_count = (config.duration * config.sample_rate as f32) as usize;
        let audio = if config.channels == 1 {
            AudioBuffer::mono(vec![0.0f32; sample_count], config.sample_rate)
        } else {
            // For stereo, interleave the samples (left, right, left, right, ...)
            let mut interleaved = Vec::with_capacity(sample_count * 2);
            for _ in 0..sample_count {
                interleaved.push(0.0f32); // Left channel
                interleaved.push(0.0f32); // Right channel
            }
            AudioBuffer::stereo(interleaved, config.sample_rate)
        };

        // Measure processing time (simulate model inference)
        let start_time = Instant::now();

        // Simulate different model processing times based on model type
        let processing_delay = match config.model_type.as_str() {
            "small" => Duration::from_millis((config.duration * 50.0) as u64), // 0.05 RTF
            "base" => Duration::from_millis((config.duration * 100.0) as u64), // 0.1 RTF
            "large" => Duration::from_millis((config.duration * 200.0) as u64), // 0.2 RTF
            _ => Duration::from_millis((config.duration * 100.0) as u64),      // Default to base
        };

        tokio::time::sleep(processing_delay).await;
        let processing_time = start_time.elapsed();

        // Mock startup function with realistic timing
        let startup_fn = || async {
            let startup_delay = match config.model_type.as_str() {
                "small" => Duration::from_millis(500),
                "base" => Duration::from_millis(1500),
                "large" => Duration::from_millis(3000),
                _ => Duration::from_millis(1500),
            };
            tokio::time::sleep(startup_delay).await;
            Ok(())
        };

        // Run comprehensive validation
        let streaming_latency = Some(Duration::from_millis(100));
        let validation = self
            .validator
            .validate_comprehensive(&audio, startup_fn, processing_time, streaming_latency)
            .await
            .expect("Performance validation should succeed");

        // Get current timestamp and commit hash
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let commit_hash = get_git_commit_hash();

        BenchmarkResult {
            name: format!(
                "{}_{}_{}s",
                config.model_type, config.sample_rate, config.duration
            ),
            timestamp,
            commit_hash,
            rtf: validation.metrics.rtf,
            memory_usage: validation.metrics.memory_usage,
            startup_time_ms: validation.metrics.startup_time_ms,
            streaming_latency_ms: validation.metrics.streaming_latency_ms,
            throughput_samples_per_sec: validation.metrics.throughput_samples_per_sec,
            cpu_utilization: validation.metrics.cpu_utilization,
            audio_duration: config.duration,
            config,
        }
    }

    /// Load baseline benchmark results
    pub fn load_baseline(&self) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
        if !Path::new(&self.baseline_path).exists() {
            return Ok(Vec::new());
        }

        let content = fs::read_to_string(&self.baseline_path)?;
        let baselines: Vec<BenchmarkResult> = serde_json::from_str(&content)?;
        Ok(baselines)
    }

    /// Save baseline benchmark results
    pub fn save_baseline(
        &self,
        results: &[BenchmarkResult],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Ensure directory exists
        if let Some(parent) = Path::new(&self.baseline_path).parent() {
            fs::create_dir_all(parent)?;
        }

        let content = serde_json::to_string_pretty(results)?;
        fs::write(&self.baseline_path, content)?;
        Ok(())
    }

    /// Load historical benchmark results
    pub fn load_history(&self) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
        if !Path::new(&self.history_path).exists() {
            return Ok(Vec::new());
        }

        let content = fs::read_to_string(&self.history_path)?;
        let history: Vec<BenchmarkResult> = serde_json::from_str(&content)?;
        Ok(history)
    }

    /// Append to historical benchmark results
    pub fn append_to_history(
        &self,
        result: &BenchmarkResult,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut history = self.load_history().unwrap_or_default();
        history.push(result.clone());

        // Keep only last 100 results to prevent file from growing too large
        if history.len() > 100 {
            history.drain(0..history.len() - 100);
        }

        // Ensure directory exists
        if let Some(parent) = Path::new(&self.history_path).parent() {
            fs::create_dir_all(parent)?;
        }

        let content = serde_json::to_string_pretty(&history)?;
        fs::write(&self.history_path, content)?;
        Ok(())
    }

    /// Analyze performance regression against baseline
    pub fn analyze_regression(
        &self,
        current_result: &BenchmarkResult,
        baseline_results: &[BenchmarkResult],
    ) -> RegressionAnalysis {
        // Find matching baseline
        let baseline = baseline_results
            .iter()
            .find(|b| b.name == current_result.name)
            .cloned();

        if baseline.is_none() {
            return RegressionAnalysis {
                has_regressions: false,
                regressions: Vec::new(),
                improvements: Vec::new(),
                overall_delta: 0.0,
            };
        }

        let baseline = baseline.unwrap();
        let mut regressions = Vec::new();
        let mut improvements = Vec::new();

        // Analyze RTF (lower is better)
        let rtf_change =
            calculate_percentage_change(baseline.rtf as f64, current_result.rtf as f64);
        if rtf_change > self.thresholds.rtf_threshold {
            regressions.push(PerformanceRegression {
                metric: "RTF".to_string(),
                current_value: current_result.rtf as f64,
                baseline_value: baseline.rtf as f64,
                percentage_change: rtf_change,
                severity: classify_severity(rtf_change),
            });
        } else if rtf_change < -5.0 {
            improvements.push(PerformanceImprovement {
                metric: "RTF".to_string(),
                current_value: current_result.rtf as f64,
                baseline_value: baseline.rtf as f64,
                percentage_change: rtf_change.abs(),
            });
        }

        // Analyze memory usage (lower is better)
        let memory_change = calculate_percentage_change(
            baseline.memory_usage as f64,
            current_result.memory_usage as f64,
        );
        if memory_change >= self.thresholds.memory_threshold {
            regressions.push(PerformanceRegression {
                metric: "Memory Usage".to_string(),
                current_value: current_result.memory_usage as f64,
                baseline_value: baseline.memory_usage as f64,
                percentage_change: memory_change,
                severity: classify_severity(memory_change),
            });
        } else if memory_change < -10.0 {
            improvements.push(PerformanceImprovement {
                metric: "Memory Usage".to_string(),
                current_value: current_result.memory_usage as f64,
                baseline_value: baseline.memory_usage as f64,
                percentage_change: memory_change.abs(),
            });
        }

        // Analyze startup time (lower is better)
        let startup_change = calculate_percentage_change(
            baseline.startup_time_ms as f64,
            current_result.startup_time_ms as f64,
        );
        if startup_change > self.thresholds.startup_threshold {
            regressions.push(PerformanceRegression {
                metric: "Startup Time".to_string(),
                current_value: current_result.startup_time_ms as f64,
                baseline_value: baseline.startup_time_ms as f64,
                percentage_change: startup_change,
                severity: classify_severity(startup_change),
            });
        } else if startup_change < -10.0 {
            improvements.push(PerformanceImprovement {
                metric: "Startup Time".to_string(),
                current_value: current_result.startup_time_ms as f64,
                baseline_value: baseline.startup_time_ms as f64,
                percentage_change: startup_change.abs(),
            });
        }

        // Analyze throughput (higher is better)
        let throughput_change = calculate_percentage_change(
            baseline.throughput_samples_per_sec,
            current_result.throughput_samples_per_sec,
        );
        if throughput_change < -self.thresholds.throughput_threshold {
            regressions.push(PerformanceRegression {
                metric: "Throughput".to_string(),
                current_value: current_result.throughput_samples_per_sec,
                baseline_value: baseline.throughput_samples_per_sec,
                percentage_change: throughput_change.abs(),
                severity: classify_severity(throughput_change.abs()),
            });
        } else if throughput_change > 10.0 {
            improvements.push(PerformanceImprovement {
                metric: "Throughput".to_string(),
                current_value: current_result.throughput_samples_per_sec,
                baseline_value: baseline.throughput_samples_per_sec,
                percentage_change: throughput_change,
            });
        }

        // Calculate overall performance delta (weighted average)
        let overall_delta = (rtf_change * 0.3 + memory_change * 0.2 + startup_change * 0.2
            - throughput_change * 0.3);

        RegressionAnalysis {
            has_regressions: !regressions.is_empty(),
            regressions,
            improvements,
            overall_delta,
        }
    }

    /// Generate CI-friendly report
    pub fn generate_ci_report(&self, analysis: &RegressionAnalysis) -> String {
        let mut report = String::new();

        if analysis.has_regressions {
            report.push_str("❌ PERFORMANCE REGRESSION DETECTED\n\n");

            for regression in &analysis.regressions {
                let severity_emoji = match regression.severity {
                    RegressionSeverity::Minor => "⚠️",
                    RegressionSeverity::Major => "🚨",
                    RegressionSeverity::Critical => "🔥",
                };

                report.push_str(&format!(
                    "{} {}: {:.2}% regression (current: {:.3}, baseline: {:.3})\n",
                    severity_emoji,
                    regression.metric,
                    regression.percentage_change,
                    regression.current_value,
                    regression.baseline_value
                ));
            }
        } else {
            report.push_str("✅ NO PERFORMANCE REGRESSIONS DETECTED\n\n");
        }

        if !analysis.improvements.is_empty() {
            report.push_str("🚀 Performance Improvements:\n");
            for improvement in &analysis.improvements {
                report.push_str(&format!(
                    "  • {}: {:.2}% improvement\n",
                    improvement.metric, improvement.percentage_change
                ));
            }
        }

        report.push_str(&format!(
            "\nOverall Performance Delta: {:.2}%\n",
            analysis.overall_delta
        ));

        report
    }
}

impl Default for RegressionTester {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate percentage change between two values
fn calculate_percentage_change(baseline: f64, current: f64) -> f32 {
    if baseline == 0.0 {
        return 0.0;
    }
    ((current - baseline) / baseline * 100.0) as f32
}

/// Classify severity based on percentage change
fn classify_severity(percentage_change: f32) -> RegressionSeverity {
    if percentage_change >= 30.0 {
        RegressionSeverity::Critical
    } else if percentage_change >= 15.0 {
        RegressionSeverity::Major
    } else {
        RegressionSeverity::Minor
    }
}

/// Get current git commit hash
fn get_git_commit_hash() -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
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

    #[tokio::test]
    async fn test_benchmark_execution() {
        let tester = RegressionTester::new();

        let config = TestConfiguration {
            model_type: "base".to_string(),
            sample_rate: 16000,
            duration: 1.0,
            channels: 1,
            features: vec!["whisper".to_string()],
        };

        let result = tester.run_benchmark(config).await;

        assert!(result.rtf > 0.0);
        assert!(result.memory_usage > 0);
        assert!(result.startup_time_ms > 0);
        assert!(result.throughput_samples_per_sec > 0.0);
    }

    #[test]
    fn test_percentage_change_calculation() {
        assert_eq!(calculate_percentage_change(100.0, 110.0), 10.0);
        assert_eq!(calculate_percentage_change(100.0, 90.0), -10.0);
        assert_eq!(calculate_percentage_change(0.0, 10.0), 0.0);
        assert_eq!(calculate_percentage_change(200.0, 200.0), 0.0);
    }

    #[test]
    fn test_severity_classification() {
        assert_eq!(classify_severity(5.0), RegressionSeverity::Minor);
        assert_eq!(classify_severity(20.0), RegressionSeverity::Major);
        assert_eq!(classify_severity(35.0), RegressionSeverity::Critical);
    }

    #[tokio::test]
    async fn test_regression_analysis() {
        let tester = RegressionTester::new();

        // Create baseline result
        let baseline = BenchmarkResult {
            name: "test_benchmark".to_string(),
            timestamp: 1000000,
            commit_hash: Some("abc123".to_string()),
            rtf: 0.2,
            memory_usage: 500_000_000,
            startup_time_ms: 1500,
            streaming_latency_ms: 100,
            throughput_samples_per_sec: 16000.0,
            cpu_utilization: 20.0,
            audio_duration: 1.0,
            config: TestConfiguration {
                model_type: "base".to_string(),
                sample_rate: 16000,
                duration: 1.0,
                channels: 1,
                features: vec!["whisper".to_string()],
            },
        };

        // Create current result with regression
        let current = BenchmarkResult {
            name: "test_benchmark".to_string(),
            timestamp: 2000000,
            commit_hash: Some("def456".to_string()),
            rtf: 0.25,                 // 25% increase (regression)
            memory_usage: 600_000_000, // 20% increase (regression)
            startup_time_ms: 1500,
            streaming_latency_ms: 100,
            throughput_samples_per_sec: 14000.0, // Decrease (regression)
            cpu_utilization: 25.0,
            audio_duration: 1.0,
            config: baseline.config.clone(),
        };

        let analysis = tester.analyze_regression(&current, &[baseline]);

        assert!(analysis.has_regressions);
        assert!(!analysis.regressions.is_empty());

        // Should detect RTF and throughput regressions
        let regression_metrics: Vec<_> = analysis.regressions.iter().map(|r| &r.metric).collect();
        assert!(regression_metrics.contains(&&"RTF".to_string()));
        assert!(regression_metrics.contains(&&"Memory Usage".to_string()));
        assert!(regression_metrics.contains(&&"Throughput".to_string()));
    }
}
