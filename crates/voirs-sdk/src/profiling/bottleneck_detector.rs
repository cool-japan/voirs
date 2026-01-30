//! Bottleneck detection and performance analysis.

use super::{ProfileSession, StageMetrics};
use serde::{Deserialize, Serialize};

/// Severity level of a detected bottleneck.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    /// Minor performance issue
    Low,
    /// Moderate performance impact
    Medium,
    /// Significant performance degradation
    High,
    /// Critical performance bottleneck
    Critical,
}

impl BottleneckSeverity {
    /// Get severity as a percentage threshold.
    pub fn threshold(&self) -> f64 {
        match self {
            Self::Low => 10.0,       // 10% slower than expected
            Self::Medium => 25.0,    // 25% slower
            Self::High => 50.0,      // 50% slower
            Self::Critical => 100.0, // 100%+ slower
        }
    }

    /// Get color code for display.
    pub fn color(&self) -> &'static str {
        match self {
            Self::Low => "yellow",
            Self::Medium => "orange",
            Self::High => "red",
            Self::Critical => "darkred",
        }
    }
}

/// A detected performance bottleneck.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    /// Component or stage with the bottleneck
    pub component: String,

    /// Severity of the bottleneck
    pub severity: BottleneckSeverity,

    /// Percentage slower than expected
    pub slowdown_percent: f64,

    /// Expected duration
    pub expected_duration_ms: f64,

    /// Actual duration
    pub actual_duration_ms: f64,

    /// Recommended optimization
    pub recommendation: String,

    /// Impact on overall performance
    pub impact_description: String,
}

impl Bottleneck {
    /// Create a new bottleneck detection.
    pub fn new(
        component: impl Into<String>,
        severity: BottleneckSeverity,
        slowdown_percent: f64,
        expected_ms: f64,
        actual_ms: f64,
    ) -> Self {
        Self {
            component: component.into(),
            severity,
            slowdown_percent,
            expected_duration_ms: expected_ms,
            actual_duration_ms: actual_ms,
            recommendation: String::new(),
            impact_description: String::new(),
        }
    }

    /// Add recommendation.
    pub fn with_recommendation(mut self, recommendation: impl Into<String>) -> Self {
        self.recommendation = recommendation.into();
        self
    }

    /// Add impact description.
    pub fn with_impact(mut self, impact: impl Into<String>) -> Self {
        self.impact_description = impact.into();
        self
    }

    /// Get human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "[{:?}] {} is {:.1}% slower than expected ({:.1}ms vs {:.1}ms expected)",
            self.severity,
            self.component,
            self.slowdown_percent,
            self.actual_duration_ms,
            self.expected_duration_ms
        )
    }
}

/// Bottleneck detector for analyzing performance issues.
pub struct BottleneckDetector {
    regression_threshold: f64,
}

impl BottleneckDetector {
    /// Create a new bottleneck detector.
    pub fn new(regression_threshold: f64) -> Self {
        Self {
            regression_threshold,
        }
    }

    /// Detect bottlenecks in a profiling session.
    pub async fn detect(&self, session: &ProfileSession) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        // Analyze stage metrics
        for (stage_name, metrics) in &session.stage_metrics {
            if let Some(bottleneck) = self.analyze_stage_metrics(stage_name, metrics).await {
                bottlenecks.push(bottleneck);
            }
        }

        // Analyze memory usage
        if !session.memory_snapshots.is_empty() {
            if let Some(bottleneck) = self.analyze_memory_usage(session).await {
                bottlenecks.push(bottleneck);
            }
        }

        // Sort by severity
        bottlenecks.sort_by(|a, b| {
            b.slowdown_percent
                .partial_cmp(&a.slowdown_percent)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        bottlenecks
    }

    /// Analyze stage metrics for bottlenecks.
    async fn analyze_stage_metrics(
        &self,
        stage_name: &str,
        metrics: &StageMetrics,
    ) -> Option<Bottleneck> {
        // Calculate expected duration based on typical distribution
        let expected_percent = metrics.stage.expected_percentage();
        let total_duration_ms = metrics.total_duration.as_secs_f64() * 1000.0;

        // Calculate what this stage's duration should be
        let session_total_ms = total_duration_ms / (metrics.percentage_of_total / 100.0);
        let expected_ms = session_total_ms * (expected_percent / 100.0);
        let actual_ms = total_duration_ms / metrics.execution_count as f64;

        // Calculate slowdown percentage
        let slowdown_percent = if expected_ms > 0.0 {
            ((actual_ms - expected_ms) / expected_ms) * 100.0
        } else {
            0.0
        };

        // Determine severity
        if slowdown_percent > self.regression_threshold {
            let severity = self.determine_severity(slowdown_percent);

            let recommendation = self.get_stage_recommendation(stage_name, slowdown_percent);
            let impact = format!(
                "This stage is taking {:.1}% of total pipeline time (expected: {:.1}%)",
                metrics.percentage_of_total, expected_percent
            );

            Some(
                Bottleneck::new(
                    stage_name,
                    severity,
                    slowdown_percent,
                    expected_ms,
                    actual_ms,
                )
                .with_recommendation(recommendation)
                .with_impact(impact),
            )
        } else {
            None
        }
    }

    /// Analyze memory usage for potential issues.
    async fn analyze_memory_usage(&self, session: &ProfileSession) -> Option<Bottleneck> {
        let snapshots = &session.memory_snapshots;
        if snapshots.is_empty() {
            return None;
        }

        // Calculate memory growth
        let first_memory = snapshots.first()?.allocated_bytes;
        let peak_memory = snapshots.iter().map(|s| s.allocated_bytes).max()?;

        let memory_growth = peak_memory.saturating_sub(first_memory);
        let growth_percent = if first_memory > 0 {
            (memory_growth as f64 / first_memory as f64) * 100.0
        } else {
            0.0
        };

        // Check for excessive memory growth
        if growth_percent > 200.0 {
            // More than 3x growth
            let severity = if growth_percent > 500.0 {
                BottleneckSeverity::Critical
            } else if growth_percent > 300.0 {
                BottleneckSeverity::High
            } else {
                BottleneckSeverity::Medium
            };

            Some(
                Bottleneck::new(
                    "Memory Usage",
                    severity,
                    growth_percent,
                    first_memory as f64 / 1_048_576.0, // MB
                    peak_memory as f64 / 1_048_576.0,  // MB
                )
                .with_recommendation(
                    "Consider implementing streaming or batching to reduce memory footprint",
                )
                .with_impact("Excessive memory usage may lead to OOM errors or swapping"),
            )
        } else {
            None
        }
    }

    /// Determine severity based on slowdown percentage.
    fn determine_severity(&self, slowdown_percent: f64) -> BottleneckSeverity {
        if slowdown_percent >= 100.0 {
            BottleneckSeverity::Critical
        } else if slowdown_percent >= 50.0 {
            BottleneckSeverity::High
        } else if slowdown_percent >= 25.0 {
            BottleneckSeverity::Medium
        } else {
            BottleneckSeverity::Low
        }
    }

    /// Get optimization recommendation for a stage.
    fn get_stage_recommendation(&self, stage_name: &str, _slowdown: f64) -> String {
        match stage_name {
            name if name.contains("G2P") || name.contains("g2p") => {
                "Consider caching G2P results or using a faster G2P backend".to_string()
            }
            name if name.contains("Acoustic") || name.contains("acoustic") => {
                "Enable GPU acceleration or use a smaller acoustic model".to_string()
            }
            name if name.contains("Vocoder") || name.contains("vocoder") => {
                "Enable GPU acceleration or reduce vocoder quality settings".to_string()
            }
            name if name.contains("Memory") => {
                "Implement batching or streaming to reduce peak memory usage".to_string()
            }
            _ => "Profile this component in more detail to identify specific optimizations"
                .to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiling::PipelineStage;

    #[test]
    fn test_bottleneck_severity_threshold() {
        assert_eq!(BottleneckSeverity::Low.threshold(), 10.0);
        assert_eq!(BottleneckSeverity::Medium.threshold(), 25.0);
        assert_eq!(BottleneckSeverity::High.threshold(), 50.0);
        assert_eq!(BottleneckSeverity::Critical.threshold(), 100.0);
    }

    #[test]
    fn test_bottleneck_creation() {
        let bottleneck = Bottleneck::new(
            "Test Component",
            BottleneckSeverity::High,
            75.0,
            100.0,
            175.0,
        );

        assert_eq!(bottleneck.component, "Test Component");
        assert_eq!(bottleneck.severity, BottleneckSeverity::High);
        assert_eq!(bottleneck.slowdown_percent, 75.0);
    }

    #[test]
    fn test_bottleneck_with_recommendation() {
        let bottleneck = Bottleneck::new("Test", BottleneckSeverity::Low, 15.0, 100.0, 115.0)
            .with_recommendation("Enable GPU")
            .with_impact("Minor performance impact");

        assert_eq!(bottleneck.recommendation, "Enable GPU");
        assert_eq!(bottleneck.impact_description, "Minor performance impact");
    }

    #[test]
    fn test_bottleneck_summary() {
        let bottleneck = Bottleneck::new("Vocoder", BottleneckSeverity::High, 50.0, 100.0, 150.0);
        let summary = bottleneck.summary();

        assert!(summary.contains("Vocoder"));
        assert!(summary.contains("50.0%"));
    }

    #[tokio::test]
    async fn test_bottleneck_detector_creation() {
        let detector = BottleneckDetector::new(10.0);
        assert_eq!(detector.regression_threshold, 10.0);
    }

    #[tokio::test]
    async fn test_severity_determination() {
        let detector = BottleneckDetector::new(10.0);

        assert_eq!(detector.determine_severity(5.0), BottleneckSeverity::Low);
        assert_eq!(
            detector.determine_severity(30.0),
            BottleneckSeverity::Medium
        );
        assert_eq!(detector.determine_severity(60.0), BottleneckSeverity::High);
        assert_eq!(
            detector.determine_severity(150.0),
            BottleneckSeverity::Critical
        );
    }
}
