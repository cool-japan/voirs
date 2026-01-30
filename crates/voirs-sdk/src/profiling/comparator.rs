//! Performance comparison and regression detection.

use super::ProfileSession;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of comparing two performance sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Whether regression was detected
    pub has_regression: bool,

    /// Overall performance change (percentage)
    pub overall_change_percent: f64,

    /// Stage-by-stage comparison
    pub stage_comparisons: HashMap<String, StageComparison>,

    /// Memory usage comparison
    pub memory_comparison: Option<MemoryComparison>,

    /// Summary of changes
    pub summary: String,
}

/// Comparison of a specific stage between sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageComparison {
    /// Stage name
    pub stage_name: String,

    /// Baseline duration (ms)
    pub baseline_ms: f64,

    /// Current duration (ms)
    pub current_ms: f64,

    /// Change percentage
    pub change_percent: f64,

    /// Whether this represents a regression
    pub is_regression: bool,
}

/// Comparison of memory usage between sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryComparison {
    /// Baseline peak memory (MB)
    pub baseline_peak_mb: f64,

    /// Current peak memory (MB)
    pub current_peak_mb: f64,

    /// Change percentage
    pub change_percent: f64,

    /// Whether this represents increased memory usage
    pub is_increase: bool,
}

/// Performance comparator for detecting regressions.
pub struct PerformanceComparator {
    regression_threshold: f64,
}

impl PerformanceComparator {
    /// Create a new performance comparator.
    pub fn new() -> Self {
        Self {
            regression_threshold: 10.0, // 10% slower is considered regression
        }
    }

    /// Create with custom regression threshold.
    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            regression_threshold: threshold,
        }
    }

    /// Compare two profiling sessions.
    pub async fn compare(
        &self,
        current: &ProfileSession,
        baseline: &ProfileSession,
    ) -> ComparisonResult {
        let mut has_regression = false;

        // Compare stage metrics
        let stage_comparisons = self.compare_stages(current, baseline, &mut has_regression);

        // Compare memory usage
        let memory_comparison = self.compare_memory(current, baseline, &mut has_regression);

        // Calculate overall change
        let overall_change = self.calculate_overall_change(current, baseline);

        let summary = self.generate_comparison_summary(&stage_comparisons, &memory_comparison);

        ComparisonResult {
            has_regression,
            overall_change_percent: overall_change,
            stage_comparisons,
            memory_comparison,
            summary,
        }
    }

    fn compare_stages(
        &self,
        current: &ProfileSession,
        baseline: &ProfileSession,
        has_regression: &mut bool,
    ) -> HashMap<String, StageComparison> {
        let mut comparisons = HashMap::new();

        for (stage_name, current_metrics) in &current.stage_metrics {
            if let Some(baseline_metrics) = baseline.stage_metrics.get(stage_name) {
                let baseline_ms = baseline_metrics.avg_duration.as_secs_f64() * 1000.0;
                let current_ms = current_metrics.avg_duration.as_secs_f64() * 1000.0;

                let change_percent = if baseline_ms > 0.0 {
                    ((current_ms - baseline_ms) / baseline_ms) * 100.0
                } else {
                    0.0
                };

                let is_regression = change_percent > self.regression_threshold;
                if is_regression {
                    *has_regression = true;
                }

                comparisons.insert(
                    stage_name.clone(),
                    StageComparison {
                        stage_name: stage_name.clone(),
                        baseline_ms,
                        current_ms,
                        change_percent,
                        is_regression,
                    },
                );
            }
        }

        comparisons
    }

    fn compare_memory(
        &self,
        current: &ProfileSession,
        baseline: &ProfileSession,
        has_regression: &mut bool,
    ) -> Option<MemoryComparison> {
        let current_peak = current
            .memory_snapshots
            .iter()
            .map(|s| s.allocated_mb())
            .fold(0.0f64, f64::max);

        let baseline_peak = baseline
            .memory_snapshots
            .iter()
            .map(|s| s.allocated_mb())
            .fold(0.0f64, f64::max);

        if current_peak == 0.0 || baseline_peak == 0.0 {
            return None;
        }

        let change_percent = ((current_peak - baseline_peak) / baseline_peak) * 100.0;
        let is_increase = change_percent > self.regression_threshold;

        if is_increase {
            *has_regression = true;
        }

        Some(MemoryComparison {
            baseline_peak_mb: baseline_peak,
            current_peak_mb: current_peak,
            change_percent,
            is_increase,
        })
    }

    fn calculate_overall_change(&self, current: &ProfileSession, baseline: &ProfileSession) -> f64 {
        let current_duration = current.duration().map(|d| d.as_secs_f64()).unwrap_or(0.0);
        let baseline_duration = baseline.duration().map(|d| d.as_secs_f64()).unwrap_or(0.0);

        if baseline_duration > 0.0 {
            ((current_duration - baseline_duration) / baseline_duration) * 100.0
        } else {
            0.0
        }
    }

    fn generate_comparison_summary(
        &self,
        stage_comparisons: &HashMap<String, StageComparison>,
        memory_comparison: &Option<MemoryComparison>,
    ) -> String {
        let regression_count = stage_comparisons
            .values()
            .filter(|c| c.is_regression)
            .count();

        let mut summary = String::new();

        if regression_count > 0 {
            summary.push_str(&format!(
                "⚠️  {} stage(s) show performance regression. ",
                regression_count
            ));
        } else {
            summary.push_str("✓ No stage regressions detected. ");
        }

        if let Some(mem) = memory_comparison {
            if mem.is_increase {
                summary.push_str(&format!(
                    "⚠️  Memory usage increased by {:.1}%.",
                    mem.change_percent
                ));
            } else {
                summary.push_str("✓ Memory usage stable.");
            }
        }

        summary
    }
}

impl Default for PerformanceComparator {
    fn default() -> Self {
        Self::new()
    }
}

/// Regression detector for identifying performance degradation.
pub struct RegressionDetector {
    threshold: f64,
    history_size: usize,
}

impl RegressionDetector {
    /// Create a new regression detector.
    pub fn new(threshold: f64, history_size: usize) -> Self {
        Self {
            threshold,
            history_size,
        }
    }

    /// Detect regressions in a series of sessions.
    pub fn detect_regressions(&self, sessions: &[ProfileSession]) -> Vec<RegressionAlert> {
        let mut alerts = Vec::new();

        if sessions.len() < 2 {
            return alerts;
        }

        // Compare consecutive sessions
        for i in 1..sessions.len() {
            let current = &sessions[i];
            let previous = &sessions[i - 1];

            let comparator = PerformanceComparator::with_threshold(self.threshold);
            // We can't use async in sync context, so we'll create a simplified comparison
            let comparison = self.simple_compare(current, previous);

            if comparison.has_regression {
                alerts.push(RegressionAlert {
                    session_id: current.id.clone(),
                    session_name: current.name.clone(),
                    regression_type: RegressionType::Performance,
                    severity: if comparison.overall_change_percent > 50.0 {
                        "Critical"
                    } else if comparison.overall_change_percent > 25.0 {
                        "High"
                    } else {
                        "Medium"
                    }
                    .to_string(),
                    description: comparison.summary,
                });
            }
        }

        alerts
    }

    fn simple_compare(
        &self,
        current: &ProfileSession,
        baseline: &ProfileSession,
    ) -> ComparisonResult {
        let current_duration = current.duration().map(|d| d.as_secs_f64()).unwrap_or(0.0);
        let baseline_duration = baseline.duration().map(|d| d.as_secs_f64()).unwrap_or(0.0);

        let change_percent = if baseline_duration > 0.0 {
            ((current_duration - baseline_duration) / baseline_duration) * 100.0
        } else {
            0.0
        };

        let has_regression = change_percent > self.threshold;

        ComparisonResult {
            has_regression,
            overall_change_percent: change_percent,
            stage_comparisons: HashMap::new(),
            memory_comparison: None,
            summary: if has_regression {
                format!("Performance degraded by {:.1}%", change_percent)
            } else {
                "No regression detected".to_string()
            },
        }
    }
}

/// Alert for detected regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAlert {
    pub session_id: String,
    pub session_name: String,
    pub regression_type: RegressionType,
    pub severity: String,
    pub description: String,
}

/// Type of regression detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegressionType {
    /// Performance regression
    Performance,
    /// Memory regression
    Memory,
    /// Quality regression
    Quality,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_comparison() {
        let comparison = StageComparison {
            stage_name: "Test".to_string(),
            baseline_ms: 100.0,
            current_ms: 150.0,
            change_percent: 50.0,
            is_regression: true,
        };

        assert_eq!(comparison.change_percent, 50.0);
        assert!(comparison.is_regression);
    }

    #[test]
    fn test_memory_comparison() {
        let comparison = MemoryComparison {
            baseline_peak_mb: 100.0,
            current_peak_mb: 120.0,
            change_percent: 20.0,
            is_increase: true,
        };

        assert_eq!(comparison.change_percent, 20.0);
        assert!(comparison.is_increase);
    }

    #[test]
    fn test_comparator_creation() {
        let comparator = PerformanceComparator::new();
        assert_eq!(comparator.regression_threshold, 10.0);

        let comparator = PerformanceComparator::with_threshold(25.0);
        assert_eq!(comparator.regression_threshold, 25.0);
    }

    #[test]
    fn test_regression_detector() {
        let detector = RegressionDetector::new(10.0, 50);
        assert_eq!(detector.threshold, 10.0);
        assert_eq!(detector.history_size, 50);
    }

    #[test]
    fn test_regression_type() {
        assert_eq!(RegressionType::Performance, RegressionType::Performance);
        assert_ne!(RegressionType::Performance, RegressionType::Memory);
    }
}
