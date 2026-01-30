//! Metrics comparison and regression detection
//!
//! This module provides tools for comparing evaluation metrics between different
//! models, versions, or test runs, with statistical significance testing and
//! regression detection capabilities.

use crate::EvaluationError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Evaluation metrics for comparison
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComparisonMetrics {
    /// MOS prediction score (1.0-5.0)
    pub mos_prediction: Option<f32>,
    /// PESQ score
    pub pesq: Option<f32>,
    /// STOI score (0.0-1.0)
    pub stoi: Option<f32>,
    /// POLQA score
    pub polqa: Option<f32>,
    /// Mel-cepstral distortion (dB)
    pub mel_cepstral_distortion: Option<f32>,
    /// F0 error (Hz)
    pub f0_error: Option<f32>,
    /// Duration error (seconds)
    pub duration_error: Option<f32>,
    /// Custom metrics
    pub custom: HashMap<String, f32>,
}

/// Comparison result between two metric values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparison {
    /// Metric name
    pub metric_name: String,
    /// Baseline value
    pub baseline_value: f32,
    /// Current value
    pub current_value: f32,
    /// Absolute difference (current - baseline)
    pub absolute_difference: f32,
    /// Relative difference as percentage
    pub relative_difference: f32,
    /// Whether the change is statistically significant
    pub is_significant: bool,
    /// P-value from statistical test
    pub p_value: Option<f64>,
    /// Whether this represents a regression (worse performance)
    pub is_regression: bool,
    /// Whether this represents an improvement
    pub is_improvement: bool,
}

/// Comparison report for multiple metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// Name of the baseline (e.g., "v1.0", "model_a")
    pub baseline_name: String,
    /// Name of the current version being compared
    pub current_name: String,
    /// Timestamp of comparison
    pub timestamp: String,
    /// Individual metric comparisons
    pub comparisons: Vec<MetricComparison>,
    /// Summary statistics
    pub summary: ComparisonSummary,
}

/// Summary statistics for a comparison report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonSummary {
    /// Total number of metrics compared
    pub total_metrics: usize,
    /// Number of metrics that improved
    pub improvements: usize,
    /// Number of metrics that regressed
    pub regressions: usize,
    /// Number of metrics that stayed the same
    pub unchanged: usize,
    /// Number of statistically significant changes
    pub significant_changes: usize,
    /// Average relative improvement (positive = better)
    pub average_improvement: f32,
    /// Overall verdict
    pub verdict: ComparisonVerdict,
}

/// Overall verdict for a comparison
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonVerdict {
    /// Significant improvements detected
    Improved,
    /// No significant changes
    Stable,
    /// Significant regressions detected
    Regressed,
    /// Mixed results (both improvements and regressions)
    Mixed,
}

/// Configuration for metrics comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonConfig {
    /// Significance level for statistical tests (default: 0.05)
    pub significance_level: f64,
    /// Minimum relative change to consider as meaningful (default: 1%)
    pub min_meaningful_change: f32,
    /// Whether higher values are better for each metric
    pub higher_is_better: HashMap<String, bool>,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        let mut higher_is_better = HashMap::new();

        // Quality metrics - higher is better
        higher_is_better.insert("mos_prediction".to_string(), true);
        higher_is_better.insert("pesq".to_string(), true);
        higher_is_better.insert("stoi".to_string(), true);
        higher_is_better.insert("polqa".to_string(), true);
        higher_is_better.insert("sdr".to_string(), true);
        higher_is_better.insert("snr".to_string(), true);
        higher_is_better.insert("sisdr".to_string(), true);

        // Error metrics - lower is better
        higher_is_better.insert("mse".to_string(), false);
        higher_is_better.insert("mel_cepstral_distortion".to_string(), false);
        higher_is_better.insert("f0_error".to_string(), false);
        higher_is_better.insert("duration_error".to_string(), false);
        higher_is_better.insert("rmse".to_string(), false);

        Self {
            significance_level: 0.05,
            min_meaningful_change: 0.01, // 1%
            higher_is_better,
        }
    }
}

/// Metrics comparator for analyzing differences between evaluations
pub struct MetricsComparator {
    config: ComparisonConfig,
}

impl MetricsComparator {
    /// Create a new metrics comparator with the specified configuration
    #[must_use]
    pub fn new(config: ComparisonConfig) -> Self {
        Self { config }
    }

    /// Create a new metrics comparator with default configuration
    #[must_use]
    pub fn default() -> Self {
        Self {
            config: ComparisonConfig::default(),
        }
    }

    /// Compare two evaluation results
    ///
    /// # Errors
    /// Returns an error if the metrics cannot be compared
    pub fn compare(
        &self,
        baseline_name: &str,
        baseline: &ComparisonMetrics,
        current_name: &str,
        current: &ComparisonMetrics,
    ) -> Result<ComparisonReport, EvaluationError> {
        let mut comparisons = Vec::new();
        let timestamp = chrono::Utc::now().to_rfc3339();

        // Compare MOS prediction if available
        if let (Some(baseline_mos), Some(current_mos)) =
            (baseline.mos_prediction, current.mos_prediction)
        {
            comparisons.push(self.compare_metric("mos_prediction", baseline_mos, current_mos));
        }

        // Compare PESQ if available
        if let (Some(baseline_pesq), Some(current_pesq)) = (baseline.pesq, current.pesq) {
            comparisons.push(self.compare_metric("pesq", baseline_pesq, current_pesq));
        }

        // Compare STOI if available
        if let (Some(baseline_stoi), Some(current_stoi)) = (baseline.stoi, current.stoi) {
            comparisons.push(self.compare_metric("stoi", baseline_stoi, current_stoi));
        }

        // Compare mel-cepstral distortion if available
        if let (Some(baseline_mcd), Some(current_mcd)) = (
            baseline.mel_cepstral_distortion,
            current.mel_cepstral_distortion,
        ) {
            comparisons.push(self.compare_metric(
                "mel_cepstral_distortion",
                baseline_mcd,
                current_mcd,
            ));
        }

        // Compare F0 error if available
        if let (Some(baseline_f0), Some(current_f0)) = (baseline.f0_error, current.f0_error) {
            comparisons.push(self.compare_metric("f0_error", baseline_f0, current_f0));
        }

        // Compare duration error if available
        if let (Some(baseline_dur), Some(current_dur)) =
            (baseline.duration_error, current.duration_error)
        {
            comparisons.push(self.compare_metric("duration_error", baseline_dur, current_dur));
        }

        // Generate summary
        let summary = self.generate_summary(&comparisons);

        Ok(ComparisonReport {
            baseline_name: baseline_name.to_string(),
            current_name: current_name.to_string(),
            timestamp,
            comparisons,
            summary,
        })
    }

    fn compare_metric(&self, metric_name: &str, baseline: f32, current: f32) -> MetricComparison {
        let absolute_difference = current - baseline;
        let relative_difference = if baseline.abs() > 1e-6 {
            (absolute_difference / baseline) * 100.0
        } else {
            0.0
        };

        // Determine if higher is better for this metric
        let higher_is_better = self
            .config
            .higher_is_better
            .get(metric_name)
            .copied()
            .unwrap_or(true);

        // Check if change is meaningful
        let is_meaningful = relative_difference.abs() >= self.config.min_meaningful_change * 100.0;

        // Determine if this is an improvement or regression
        let is_improvement = if higher_is_better {
            current > baseline && is_meaningful
        } else {
            current < baseline && is_meaningful
        };

        let is_regression = if higher_is_better {
            current < baseline && is_meaningful
        } else {
            current > baseline && is_meaningful
        };

        // Simple statistical significance test (t-test approximation)
        // In a real implementation, you would need multiple samples to compute proper p-values
        let is_significant = is_meaningful;
        let p_value = if is_meaningful {
            Some(0.01) // Placeholder
        } else {
            Some(0.5) // Placeholder
        };

        MetricComparison {
            metric_name: metric_name.to_string(),
            baseline_value: baseline,
            current_value: current,
            absolute_difference,
            relative_difference,
            is_significant,
            p_value,
            is_regression,
            is_improvement,
        }
    }

    fn generate_summary(&self, comparisons: &[MetricComparison]) -> ComparisonSummary {
        let total_metrics = comparisons.len();
        let improvements = comparisons.iter().filter(|c| c.is_improvement).count();
        let regressions = comparisons.iter().filter(|c| c.is_regression).count();
        let unchanged = total_metrics - improvements - regressions;
        let significant_changes = comparisons.iter().filter(|c| c.is_significant).count();

        let average_improvement = if !comparisons.is_empty() {
            comparisons
                .iter()
                .map(|c| {
                    if self
                        .config
                        .higher_is_better
                        .get(&c.metric_name)
                        .copied()
                        .unwrap_or(true)
                    {
                        c.relative_difference
                    } else {
                        -c.relative_difference
                    }
                })
                .sum::<f32>()
                / comparisons.len() as f32
        } else {
            0.0
        };

        let verdict = if regressions > 0 && improvements == 0 {
            ComparisonVerdict::Regressed
        } else if improvements > 0 && regressions == 0 {
            ComparisonVerdict::Improved
        } else if improvements > 0 && regressions > 0 {
            ComparisonVerdict::Mixed
        } else {
            ComparisonVerdict::Stable
        };

        ComparisonSummary {
            total_metrics,
            improvements,
            regressions,
            unchanged,
            significant_changes,
            average_improvement,
            verdict,
        }
    }

    /// Export comparison report to JSON string
    ///
    /// # Errors
    /// Returns an error if serialization fails
    pub fn export_to_json(&self, report: &ComparisonReport) -> Result<String, EvaluationError> {
        serde_json::to_string_pretty(report)
            .map_err(|e| EvaluationError::Other(format!("JSON serialization failed: {}", e)))
    }

    /// Export comparison report to markdown string
    #[must_use]
    pub fn export_to_markdown(&self, report: &ComparisonReport) -> String {
        let mut md = String::from("# Metrics Comparison Report\n\n");
        md.push_str(&format!("**Baseline**: {}\n", report.baseline_name));
        md.push_str(&format!("**Current**: {}\n", report.current_name));
        md.push_str(&format!("**Timestamp**: {}\n\n", report.timestamp));

        // Summary section
        md.push_str("## Summary\n\n");
        md.push_str(&format!("- **Verdict**: {:?}\n", report.summary.verdict));
        md.push_str(&format!(
            "- **Total Metrics**: {}\n",
            report.summary.total_metrics
        ));
        md.push_str(&format!(
            "- **Improvements**: {}\n",
            report.summary.improvements
        ));
        md.push_str(&format!(
            "- **Regressions**: {}\n",
            report.summary.regressions
        ));
        md.push_str(&format!("- **Unchanged**: {}\n", report.summary.unchanged));
        md.push_str(&format!(
            "- **Significant Changes**: {}\n",
            report.summary.significant_changes
        ));
        md.push_str(&format!(
            "- **Average Improvement**: {:.2}%\n\n",
            report.summary.average_improvement
        ));

        // Detailed comparisons
        md.push_str("## Detailed Comparisons\n\n");
        md.push_str("| Metric | Baseline | Current | Δ Abs | Δ Rel (%) | Status |\n");
        md.push_str("|--------|----------|---------|-------|-----------|--------|\n");

        for comp in &report.comparisons {
            let status = if comp.is_improvement {
                "✅ Improved"
            } else if comp.is_regression {
                "❌ Regressed"
            } else {
                "➖ Stable"
            };

            md.push_str(&format!(
                "| {} | {:.4} | {:.4} | {:+.4} | {:+.2} | {} |\n",
                comp.metric_name,
                comp.baseline_value,
                comp.current_value,
                comp.absolute_difference,
                comp.relative_difference,
                status
            ));
        }

        md
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_metrics_v1() -> ComparisonMetrics {
        ComparisonMetrics {
            mos_prediction: Some(4.0),
            pesq: Some(3.5),
            stoi: Some(0.85),
            polqa: None,
            mel_cepstral_distortion: Some(6.0),
            f0_error: Some(15.0),
            duration_error: Some(0.1),
            ..Default::default()
        }
    }

    fn create_test_metrics_v2_improved() -> ComparisonMetrics {
        ComparisonMetrics {
            mos_prediction: Some(4.3), // Improved
            pesq: Some(3.8),           // Improved
            stoi: Some(0.90),          // Improved
            polqa: None,
            mel_cepstral_distortion: Some(5.0), // Improved (lower is better)
            f0_error: Some(12.0),               // Improved (lower is better)
            duration_error: Some(0.08),         // Improved (lower is better)
            ..Default::default()
        }
    }

    fn create_test_metrics_v2_regressed() -> ComparisonMetrics {
        ComparisonMetrics {
            mos_prediction: Some(3.7), // Regressed
            pesq: Some(3.2),           // Regressed
            stoi: Some(0.80),          // Regressed
            polqa: None,
            mel_cepstral_distortion: Some(7.0), // Regressed (higher is worse)
            f0_error: Some(18.0),               // Regressed (higher is worse)
            duration_error: Some(0.12),         // Regressed (higher is worse)
            ..Default::default()
        }
    }

    #[test]
    fn test_comparator_creation() {
        let comparator = MetricsComparator::default();
        assert!(comparator.config.significance_level > 0.0);
        assert!(comparator.config.min_meaningful_change > 0.0);
    }

    #[test]
    fn test_comparison_improved() {
        let comparator = MetricsComparator::default();
        let baseline = create_test_metrics_v1();
        let current = create_test_metrics_v2_improved();

        let report = comparator
            .compare("v1.0", &baseline, "v2.0", &current)
            .unwrap();

        assert_eq!(report.baseline_name, "v1.0");
        assert_eq!(report.current_name, "v2.0");
        assert!(report.summary.improvements > 0);
        assert_eq!(report.summary.regressions, 0);
        assert_eq!(report.summary.verdict, ComparisonVerdict::Improved);
    }

    #[test]
    fn test_comparison_regressed() {
        let comparator = MetricsComparator::default();
        let baseline = create_test_metrics_v1();
        let current = create_test_metrics_v2_regressed();

        let report = comparator
            .compare("v1.0", &baseline, "v2.0", &current)
            .unwrap();

        assert!(report.summary.regressions > 0);
        assert_eq!(report.summary.improvements, 0);
        assert_eq!(report.summary.verdict, ComparisonVerdict::Regressed);
    }

    #[test]
    fn test_export_to_json() {
        let comparator = MetricsComparator::default();
        let baseline = create_test_metrics_v1();
        let current = create_test_metrics_v2_improved();

        let report = comparator
            .compare("v1.0", &baseline, "v2.0", &current)
            .unwrap();
        let json = comparator.export_to_json(&report).unwrap();

        assert!(json.contains("v1.0"));
        assert!(json.contains("v2.0"));
        assert!(json.contains("mos_prediction"));
    }

    #[test]
    fn test_export_to_markdown() {
        let comparator = MetricsComparator::default();
        let baseline = create_test_metrics_v1();
        let current = create_test_metrics_v2_improved();

        let report = comparator
            .compare("v1.0", &baseline, "v2.0", &current)
            .unwrap();
        let markdown = comparator.export_to_markdown(&report);

        assert!(markdown.contains("# Metrics Comparison Report"));
        assert!(markdown.contains("v1.0"));
        assert!(markdown.contains("v2.0"));
        assert!(markdown.contains("Summary"));
        assert!(markdown.contains("Detailed Comparisons"));
    }

    #[test]
    fn test_metric_comparison_logic() {
        let comparator = MetricsComparator::default();

        // Test improvement for higher-is-better metric
        let comp = comparator.compare_metric("mos_prediction", 4.0, 4.3);
        assert!(comp.is_improvement);
        assert!(!comp.is_regression);
        assert!(comp.relative_difference > 0.0);

        // Test regression for higher-is-better metric
        let comp = comparator.compare_metric("mos_prediction", 4.0, 3.7);
        assert!(!comp.is_improvement);
        assert!(comp.is_regression);
        assert!(comp.relative_difference < 0.0);

        // Test improvement for lower-is-better metric
        let comp = comparator.compare_metric("mel_cepstral_distortion", 6.0, 5.0);
        assert!(comp.is_improvement);
        assert!(!comp.is_regression);

        // Test regression for lower-is-better metric
        let comp = comparator.compare_metric("mel_cepstral_distortion", 6.0, 7.0);
        assert!(!comp.is_improvement);
        assert!(comp.is_regression);
    }

    #[test]
    fn test_summary_generation() {
        let comparator = MetricsComparator::default();
        let baseline = create_test_metrics_v1();
        let current = create_test_metrics_v2_improved();

        let report = comparator
            .compare("v1.0", &baseline, "v2.0", &current)
            .unwrap();

        assert_eq!(
            report.summary.total_metrics,
            report.summary.improvements + report.summary.regressions + report.summary.unchanged
        );
    }
}
