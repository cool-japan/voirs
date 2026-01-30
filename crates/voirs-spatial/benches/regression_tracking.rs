//! Performance regression tracking and detection system
//!
//! This module provides automated performance regression detection for benchmarks.
//! It stores baseline benchmark results and detects when performance degrades beyond
//! acceptable thresholds.
//!
//! # Usage
//!
//! 1. Run benchmarks and save baseline:
//!    ```bash
//!    cargo bench --bench regression_tracking -- --save-baseline
//!    ```
//!
//! 2. Run benchmarks and check for regressions:
//!    ```bash
//!    cargo bench --bench regression_tracking
//!    ```
//!
//! 3. Generate regression report:
//!    ```bash
//!    cargo bench --bench regression_tracking -- --regression-report
//!    ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use voirs_spatial::{Position3D, SIMDSpatialOps};

/// Benchmark baseline data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkBaseline {
    /// Benchmark name
    pub name: String,
    /// Mean execution time in nanoseconds
    pub mean_ns: f64,
    /// Standard deviation in nanoseconds
    pub std_dev_ns: f64,
    /// Throughput (elements per second)
    pub throughput: Option<f64>,
    /// Timestamp when baseline was recorded
    pub timestamp: String,
    /// Git commit hash (if available)
    pub commit_hash: Option<String>,
}

/// Performance regression threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionThresholds {
    /// Maximum acceptable performance degradation (e.g., 0.10 = 10% slower)
    pub max_degradation: f64,
    /// Minimum acceptable performance improvement to consider significant (e.g., 0.05 = 5% faster)
    pub min_improvement: f64,
    /// Number of standard deviations to consider as noise
    pub noise_threshold: f64,
}

impl Default for RegressionThresholds {
    fn default() -> Self {
        Self {
            max_degradation: 0.10, // 10% slower is a regression
            min_improvement: 0.05, // 5% faster is an improvement
            noise_threshold: 2.0,  // 2 standard deviations
        }
    }
}

/// Regression detection result
#[derive(Debug, Clone, PartialEq)]
pub enum RegressionStatus {
    /// No significant change detected
    NoChange,
    /// Performance improved significantly
    Improved { percent: f64 },
    /// Performance regressed (degraded)
    Regressed { percent: f64 },
    /// No baseline available for comparison
    NoBaseline,
}

/// Performance regression tracker
pub struct RegressionTracker {
    baselines: HashMap<String, BenchmarkBaseline>,
    thresholds: RegressionThresholds,
    baseline_path: PathBuf,
}

impl RegressionTracker {
    /// Create a new regression tracker
    pub fn new(baseline_path: impl AsRef<Path>) -> Self {
        Self {
            baselines: HashMap::new(),
            thresholds: RegressionThresholds::default(),
            baseline_path: baseline_path.as_ref().to_path_buf(),
        }
    }

    /// Create tracker with custom thresholds
    pub fn with_thresholds(
        baseline_path: impl AsRef<Path>,
        thresholds: RegressionThresholds,
    ) -> Self {
        Self {
            baselines: HashMap::new(),
            thresholds,
            baseline_path: baseline_path.as_ref().to_path_buf(),
        }
    }

    /// Load baselines from disk
    pub fn load_baselines(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if !self.baseline_path.exists() {
            return Ok(()); // No baselines yet
        }

        let content = fs::read_to_string(&self.baseline_path)?;
        self.baselines = serde_json::from_str(&content)?;
        Ok(())
    }

    /// Save baselines to disk
    pub fn save_baselines(&self) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_json::to_string_pretty(&self.baselines)?;
        fs::write(&self.baseline_path, content)?;
        Ok(())
    }

    /// Add or update a baseline
    pub fn update_baseline(&mut self, baseline: BenchmarkBaseline) {
        self.baselines.insert(baseline.name.clone(), baseline);
    }

    /// Check for regression given current benchmark result
    pub fn check_regression(
        &self,
        name: &str,
        current_mean_ns: f64,
        current_std_dev_ns: f64,
    ) -> RegressionStatus {
        let Some(baseline) = self.baselines.get(name) else {
            return RegressionStatus::NoBaseline;
        };

        // Calculate percentage change
        let percent_change = (current_mean_ns - baseline.mean_ns) / baseline.mean_ns;

        // Check if change is within noise threshold
        let noise_margin =
            self.thresholds.noise_threshold * (baseline.std_dev_ns + current_std_dev_ns) / 2.0;
        let absolute_change = (current_mean_ns - baseline.mean_ns).abs();

        if absolute_change < noise_margin {
            return RegressionStatus::NoChange;
        }

        // Determine if this is a regression or improvement
        if percent_change > self.thresholds.max_degradation {
            RegressionStatus::Regressed {
                percent: percent_change * 100.0,
            }
        } else if percent_change < -self.thresholds.min_improvement {
            RegressionStatus::Improved {
                percent: -percent_change * 100.0,
            }
        } else {
            RegressionStatus::NoChange
        }
    }

    /// Generate regression report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("# Performance Regression Report\n\n");

        if self.baselines.is_empty() {
            report.push_str("No baseline data available.\n");
            return report;
        }

        report.push_str(&format!(
            "Baseline file: {}\n",
            self.baseline_path.display()
        ));
        report.push_str(&format!(
            "Total benchmarks tracked: {}\n\n",
            self.baselines.len()
        ));

        report.push_str("## Baselines\n\n");
        report.push_str("| Benchmark | Mean (ns) | Std Dev (ns) | Timestamp |\n");
        report.push_str("|-----------|-----------|--------------|----------|\n");

        for (name, baseline) in &self.baselines {
            report.push_str(&format!(
                "| {} | {:.2} | {:.2} | {} |\n",
                name, baseline.mean_ns, baseline.std_dev_ns, baseline.timestamp
            ));
        }

        report
    }

    /// Generate detailed comparison report
    pub fn generate_comparison_report(&self, current_results: &[(String, f64, f64)]) -> String {
        let mut report = String::new();
        report.push_str("# Performance Comparison Report\n\n");

        let mut regressions = Vec::new();
        let mut improvements = Vec::new();
        let mut no_changes = Vec::new();
        let mut no_baselines = Vec::new();

        for (name, mean_ns, std_dev_ns) in current_results {
            let status = self.check_regression(name, *mean_ns, *std_dev_ns);
            match status {
                RegressionStatus::Regressed { percent } => {
                    regressions.push((name, percent, *mean_ns));
                }
                RegressionStatus::Improved { percent } => {
                    improvements.push((name, percent, *mean_ns));
                }
                RegressionStatus::NoChange => {
                    no_changes.push((name, *mean_ns));
                }
                RegressionStatus::NoBaseline => {
                    no_baselines.push((name, *mean_ns));
                }
            }
        }

        // Report regressions
        if !regressions.is_empty() {
            report.push_str("## ⚠️ Performance Regressions Detected\n\n");
            report.push_str("| Benchmark | Degradation | Current Mean (ns) |\n");
            report.push_str("|-----------|-------------|-------------------|\n");
            for (name, percent, mean_ns) in &regressions {
                report.push_str(&format!(
                    "| {} | {:.2}% | {:.2} |\n",
                    name, percent, mean_ns
                ));
            }
            report.push('\n');
        }

        // Report improvements
        if !improvements.is_empty() {
            report.push_str("## ✅ Performance Improvements\n\n");
            report.push_str("| Benchmark | Improvement | Current Mean (ns) |\n");
            report.push_str("|-----------|-------------|-------------------|\n");
            for (name, percent, mean_ns) in &improvements {
                report.push_str(&format!(
                    "| {} | {:.2}% | {:.2} |\n",
                    name, percent, mean_ns
                ));
            }
            report.push('\n');
        }

        // Report no changes
        if !no_changes.is_empty() {
            report.push_str("## ➡️ No Significant Changes\n\n");
            report.push_str(&format!(
                "{} benchmarks showed no significant change.\n\n",
                no_changes.len()
            ));
        }

        // Report missing baselines
        if !no_baselines.is_empty() {
            report.push_str("## ℹ️ Missing Baselines\n\n");
            report.push_str("| Benchmark | Current Mean (ns) |\n");
            report.push_str("|-----------|-------------------|\n");
            for (name, mean_ns) in &no_baselines {
                report.push_str(&format!("| {} | {:.2} |\n", name, mean_ns));
            }
            report.push('\n');
        }

        // Summary
        report.push_str("## Summary\n\n");
        report.push_str(&format!("- Total benchmarks: {}\n", current_results.len()));
        report.push_str(&format!("- Regressions: {}\n", regressions.len()));
        report.push_str(&format!("- Improvements: {}\n", improvements.len()));
        report.push_str(&format!("- No change: {}\n", no_changes.len()));
        report.push_str(&format!("- Missing baselines: {}\n", no_baselines.len()));

        if !regressions.is_empty() {
            report.push_str("\n⚠️ **WARNING**: Performance regressions detected!\n");
        } else if !no_baselines.is_empty() {
            report.push_str("\nℹ️ Some benchmarks are missing baseline data.\n");
        } else {
            report.push_str("\n✅ All benchmarks passed regression checks.\n");
        }

        report
    }
}

// ============================================================================
// Benchmark Functions
// ============================================================================

/// Benchmark distance calculations (regression tracking)
fn bench_distance_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression/distance");

    for count in [1000, 10000] {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &n| {
            let positions: Vec<Position3D> = (0..n)
                .map(|i| {
                    let angle = (i as f32) * 0.01;
                    Position3D::new(angle.cos() * 5.0, angle.sin() * 5.0, 1.0)
                })
                .collect();
            let listener = Position3D::new(0.0, 0.0, 0.0);

            b.iter(|| {
                for pos in &positions {
                    black_box(pos.distance_to(black_box(&listener)));
                }
            });
        });
    }

    group.finish();
}

/// Benchmark SIMD operations (regression tracking)
fn bench_simd_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression/simd");

    for count in [1000, 10000] {
        group.bench_with_input(BenchmarkId::new("distances", count), &count, |b, &n| {
            let positions: Vec<Position3D> = (0..n)
                .map(|i| {
                    Position3D::new(
                        (i as f32).sin(),
                        (i as f32).cos(),
                        (i as f32).tan().abs().min(10.0),
                    )
                })
                .collect();

            b.iter(|| {
                black_box(SIMDSpatialOps::distances(
                    black_box(Position3D::new(0.0, 0.0, 0.0)),
                    black_box(&positions),
                ));
            });
        });

        group.bench_with_input(BenchmarkId::new("normalize", count), &count, |b, &n| {
            let positions: Vec<Position3D> = (0..n)
                .map(|i| Position3D::new((i as f32) * 0.1, (i as f32) * 0.2, (i as f32) * 0.3))
                .collect();

            b.iter(|| {
                let mut positions_copy = positions.clone();
                SIMDSpatialOps::normalize_batch(black_box(&mut positions_copy));
                black_box(positions_copy);
            });
        });
    }

    group.finish();
}

/// Benchmark vector operations (regression tracking)
fn bench_vector_ops_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression/vector_ops");

    for count in [1000, 10000] {
        group.bench_with_input(BenchmarkId::new("add", count), &count, |b, &n| {
            let positions_a: Vec<Position3D> = (0..n)
                .map(|i| Position3D::new((i as f32).sin(), (i as f32).cos(), 1.0))
                .collect();
            let positions_b: Vec<Position3D> = (0..n)
                .map(|i| Position3D::new((i as f32).cos(), (i as f32).sin(), 0.5))
                .collect();

            b.iter(|| {
                for (a, b) in positions_a.iter().zip(positions_b.iter()) {
                    black_box(a.add(black_box(b)));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("dot", count), &count, |b, &n| {
            let positions_a: Vec<Position3D> = (0..n)
                .map(|i| Position3D::new((i as f32).sin(), (i as f32).cos(), 1.0))
                .collect();
            let positions_b: Vec<Position3D> = (0..n)
                .map(|i| Position3D::new((i as f32).cos(), (i as f32).sin(), 0.5))
                .collect();

            b.iter(|| {
                for (a, b) in positions_a.iter().zip(positions_b.iter()) {
                    black_box(a.dot(black_box(b)));
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_distance_regression,
    bench_simd_regression,
    bench_vector_ops_regression
);
criterion_main!(benches);

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_regression_tracker_creation() {
        let tracker = RegressionTracker::new("/tmp/test_baselines.json");
        assert!(tracker.baselines.is_empty());
    }

    #[test]
    fn test_baseline_update() {
        let mut tracker = RegressionTracker::new("/tmp/test_baselines.json");
        let baseline = BenchmarkBaseline {
            name: "test_bench".to_string(),
            mean_ns: 1000.0,
            std_dev_ns: 50.0,
            throughput: Some(1000000.0),
            timestamp: "2025-12-05T00:00:00Z".to_string(),
            commit_hash: Some("abc123".to_string()),
        };

        tracker.update_baseline(baseline.clone());
        assert_eq!(tracker.baselines.len(), 1);
        assert_eq!(tracker.baselines.get("test_bench").unwrap().mean_ns, 1000.0);
    }

    #[test]
    fn test_regression_detection_no_baseline() {
        let tracker = RegressionTracker::new("/tmp/test_baselines.json");
        let status = tracker.check_regression("unknown_bench", 1000.0, 50.0);
        assert_eq!(status, RegressionStatus::NoBaseline);
    }

    #[test]
    fn test_regression_detection_no_change() {
        let mut tracker = RegressionTracker::new("/tmp/test_baselines.json");
        tracker.update_baseline(BenchmarkBaseline {
            name: "test_bench".to_string(),
            mean_ns: 1000.0,
            std_dev_ns: 50.0,
            throughput: None,
            timestamp: "2025-12-05T00:00:00Z".to_string(),
            commit_hash: None,
        });

        // Small change within noise threshold
        let status = tracker.check_regression("test_bench", 1005.0, 52.0);
        assert_eq!(status, RegressionStatus::NoChange);
    }

    #[test]
    fn test_regression_detection_regressed() {
        let mut tracker = RegressionTracker::new("/tmp/test_baselines.json");
        tracker.update_baseline(BenchmarkBaseline {
            name: "test_bench".to_string(),
            mean_ns: 1000.0,
            std_dev_ns: 50.0,
            throughput: None,
            timestamp: "2025-12-05T00:00:00Z".to_string(),
            commit_hash: None,
        });

        // 15% slower - should be detected as regression (threshold is 10%)
        let status = tracker.check_regression("test_bench", 1150.0, 50.0);
        match status {
            RegressionStatus::Regressed { percent } => {
                assert!(percent > 10.0);
                assert!(percent < 20.0);
            }
            _ => panic!("Expected regression status"),
        }
    }

    #[test]
    fn test_regression_detection_improved() {
        let mut tracker = RegressionTracker::new("/tmp/test_baselines.json");
        tracker.update_baseline(BenchmarkBaseline {
            name: "test_bench".to_string(),
            mean_ns: 1000.0,
            std_dev_ns: 50.0,
            throughput: None,
            timestamp: "2025-12-05T00:00:00Z".to_string(),
            commit_hash: None,
        });

        // 10% faster - should be detected as improvement (threshold is 5%)
        let status = tracker.check_regression("test_bench", 900.0, 50.0);
        match status {
            RegressionStatus::Improved { percent } => {
                assert!(percent > 5.0);
                assert!(percent < 15.0);
            }
            _ => panic!("Expected improvement status"),
        }
    }

    #[test]
    fn test_baseline_save_and_load() {
        use std::fs;

        let temp_path = env::temp_dir().join("test_baseline_save_load.json");
        let mut tracker = RegressionTracker::new(&temp_path);

        // Add some baselines
        tracker.update_baseline(BenchmarkBaseline {
            name: "bench1".to_string(),
            mean_ns: 1000.0,
            std_dev_ns: 50.0,
            throughput: None,
            timestamp: "2025-12-05T00:00:00Z".to_string(),
            commit_hash: None,
        });
        tracker.update_baseline(BenchmarkBaseline {
            name: "bench2".to_string(),
            mean_ns: 2000.0,
            std_dev_ns: 100.0,
            throughput: Some(500000.0),
            timestamp: "2025-12-05T00:00:00Z".to_string(),
            commit_hash: Some("xyz789".to_string()),
        });

        // Save
        tracker.save_baselines().unwrap();

        // Load into new tracker
        let mut new_tracker = RegressionTracker::new(&temp_path);
        new_tracker.load_baselines().unwrap();

        assert_eq!(new_tracker.baselines.len(), 2);
        assert_eq!(new_tracker.baselines.get("bench1").unwrap().mean_ns, 1000.0);
        assert_eq!(new_tracker.baselines.get("bench2").unwrap().mean_ns, 2000.0);

        // Cleanup
        let _ = fs::remove_file(&temp_path);
    }

    #[test]
    fn test_report_generation() {
        let mut tracker = RegressionTracker::new("/tmp/test_report.json");
        tracker.update_baseline(BenchmarkBaseline {
            name: "test_bench".to_string(),
            mean_ns: 1000.0,
            std_dev_ns: 50.0,
            throughput: Some(1000000.0),
            timestamp: "2025-12-05T00:00:00Z".to_string(),
            commit_hash: Some("abc123".to_string()),
        });

        let report = tracker.generate_report();
        assert!(report.contains("Performance Regression Report"));
        assert!(report.contains("test_bench"));
        assert!(report.contains("1000.00"));
    }

    #[test]
    fn test_comparison_report() {
        let mut tracker = RegressionTracker::new("/tmp/test_comparison.json");
        tracker.update_baseline(BenchmarkBaseline {
            name: "bench_ok".to_string(),
            mean_ns: 1000.0,
            std_dev_ns: 50.0,
            throughput: None,
            timestamp: "2025-12-05T00:00:00Z".to_string(),
            commit_hash: None,
        });
        tracker.update_baseline(BenchmarkBaseline {
            name: "bench_regressed".to_string(),
            mean_ns: 1000.0,
            std_dev_ns: 50.0,
            throughput: None,
            timestamp: "2025-12-05T00:00:00Z".to_string(),
            commit_hash: None,
        });
        tracker.update_baseline(BenchmarkBaseline {
            name: "bench_improved".to_string(),
            mean_ns: 1000.0,
            std_dev_ns: 50.0,
            throughput: None,
            timestamp: "2025-12-05T00:00:00Z".to_string(),
            commit_hash: None,
        });

        let current_results = vec![
            ("bench_ok".to_string(), 1005.0, 52.0),        // No change
            ("bench_regressed".to_string(), 1200.0, 50.0), // 20% slower
            ("bench_improved".to_string(), 850.0, 45.0),   // 15% faster
            ("bench_new".to_string(), 500.0, 25.0),        // No baseline
        ];

        let report = tracker.generate_comparison_report(&current_results);
        assert!(report.contains("Performance Comparison Report"));
        assert!(report.contains("Regressions Detected"));
        assert!(report.contains("bench_regressed"));
        assert!(report.contains("Improvements"));
        assert!(report.contains("bench_improved"));
        assert!(report.contains("Missing Baselines"));
        assert!(report.contains("bench_new"));
    }

    #[test]
    fn test_custom_thresholds() {
        let thresholds = RegressionThresholds {
            max_degradation: 0.20, // 20% threshold
            min_improvement: 0.10, // 10% threshold
            noise_threshold: 1.0,  // 1 std dev
        };

        let mut tracker =
            RegressionTracker::with_thresholds("/tmp/test_thresholds.json", thresholds);
        tracker.update_baseline(BenchmarkBaseline {
            name: "test_bench".to_string(),
            mean_ns: 1000.0,
            std_dev_ns: 50.0,
            throughput: None,
            timestamp: "2025-12-05T00:00:00Z".to_string(),
            commit_hash: None,
        });

        // 15% slower - should NOT be regression with 20% threshold
        let status = tracker.check_regression("test_bench", 1150.0, 50.0);
        assert_eq!(status, RegressionStatus::NoChange);

        // 25% slower - should be regression
        let status = tracker.check_regression("test_bench", 1250.0, 50.0);
        match status {
            RegressionStatus::Regressed { .. } => {}
            _ => panic!("Expected regression status"),
        }
    }
}
