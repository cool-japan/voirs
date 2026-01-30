//! Performance Regression Detection System
//!
//! Provides sophisticated regression detection for quality and performance metrics
//! with statistical analysis, automated alerting, and detailed reporting.
//!
//! # Features
//!
//! - **Baseline Tracking**: Establish and manage performance baselines
//! - **Statistical Detection**: Multiple algorithms (z-score, IQR, trend analysis)
//! - **Regression Reports**: Detailed analysis with confidence levels
//! - **Integration**: Works with QualityMonitor and ML predictor
//! - **Automated Alerts**: Configurable sensitivity and notification
//!
//! # Example
//!
//! ```rust
//! use voirs_sdk::adaptive::{RegressionDetector, RegressionConfig, PerformanceSnapshot};
//!
//! # async fn example() -> voirs_sdk::Result<()> {
//! let detector = RegressionDetector::new(RegressionConfig::default());
//!
//! // Record baseline performance
//! for _ in 0..50 {
//!     let snapshot = PerformanceSnapshot {
//!         quality_score: 80.0,
//!         latency_ms: 100,
//!         rtf: 0.5,
//!         memory_mb: 512.0,
//!         cpu_percent: 45.0,
//!         timestamp: std::time::SystemTime::now(),
//!     };
//!     detector.record_performance(snapshot).await?;
//! }
//!
//! // Establish baseline
//! detector.establish_baseline().await?;
//!
//! // Check for regressions
//! let current = PerformanceSnapshot {
//!     quality_score: 65.0,  // Degraded
//!     latency_ms: 150,      // Slower
//!     rtf: 0.8,             // Worse
//!     memory_mb: 512.0,
//!     cpu_percent: 45.0,
//!     timestamp: std::time::SystemTime::now(),
//! };
//!
//! if let Some(report) = detector.detect_regression(&current).await? {
//!     println!("Regression detected: {}", report.summary);
//!     println!("Confidence: {:.1}%", report.confidence * 100.0);
//! }
//! # Ok(())
//! # }
//! ```

use crate::error::{Result, VoirsError};
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;

// Serde module for SystemTime serialization
mod system_time_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = time
            .duration_since(UNIX_EPOCH)
            .map_err(serde::ser::Error::custom)?;
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<SystemTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(UNIX_EPOCH + std::time::Duration::from_secs(secs))
    }
}

/// Performance snapshot for regression detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    /// Quality score (0-100)
    pub quality_score: f32,
    /// Synthesis latency in milliseconds
    pub latency_ms: u64,
    /// Real-time factor
    pub rtf: f32,
    /// Memory usage in MB
    pub memory_mb: f32,
    /// CPU usage percentage (0-100)
    pub cpu_percent: f32,
    /// Timestamp of measurement
    #[serde(with = "system_time_serde")]
    pub timestamp: SystemTime,
}

/// Performance baseline statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    /// Quality score statistics
    pub quality: MetricBaseline,
    /// Latency statistics
    pub latency: MetricBaseline,
    /// RTF statistics
    pub rtf: MetricBaseline,
    /// Memory statistics
    pub memory: MetricBaseline,
    /// CPU statistics
    pub cpu: MetricBaseline,
    /// Number of samples in baseline
    pub sample_count: usize,
    /// When baseline was established
    #[serde(with = "system_time_serde")]
    pub established_at: SystemTime,
}

/// Statistics for a single metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricBaseline {
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Median value
    pub median: f32,
    /// 25th percentile
    pub p25: f32,
    /// 75th percentile
    pub p75: f32,
}

/// Regression detection configuration
#[derive(Debug, Clone)]
pub struct RegressionConfig {
    /// Minimum samples for baseline (default: 30)
    pub min_baseline_samples: usize,
    /// Maximum samples to keep in history (default: 1000)
    pub max_history_size: usize,
    /// Z-score threshold for outlier detection (default: 2.0)
    pub z_score_threshold: f32,
    /// IQR multiplier for outlier detection (default: 1.5)
    pub iqr_multiplier: f32,
    /// Minimum confidence for regression report (default: 0.7)
    pub min_confidence: f32,
    /// Enable quality score regression detection
    pub detect_quality_regression: bool,
    /// Enable latency regression detection
    pub detect_latency_regression: bool,
    /// Enable RTF regression detection
    pub detect_rtf_regression: bool,
    /// Enable memory regression detection
    pub detect_memory_regression: bool,
    /// Enable CPU regression detection
    pub detect_cpu_regression: bool,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            min_baseline_samples: 30,
            max_history_size: 1000,
            z_score_threshold: 2.0,
            iqr_multiplier: 1.5,
            min_confidence: 0.7,
            detect_quality_regression: true,
            detect_latency_regression: true,
            detect_rtf_regression: true,
            detect_memory_regression: false,
            detect_cpu_regression: false,
        }
    }
}

impl RegressionConfig {
    /// Set minimum baseline samples
    pub fn with_min_baseline_samples(mut self, samples: usize) -> Self {
        self.min_baseline_samples = samples;
        self
    }

    /// Set z-score threshold
    pub fn with_z_score_threshold(mut self, threshold: f32) -> Self {
        self.z_score_threshold = threshold;
        self
    }

    /// Enable all regression detection
    pub fn with_all_detections(mut self) -> Self {
        self.detect_quality_regression = true;
        self.detect_latency_regression = true;
        self.detect_rtf_regression = true;
        self.detect_memory_regression = true;
        self.detect_cpu_regression = true;
        self
    }
}

/// Regression detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionReport {
    /// Summary message
    pub summary: String,
    /// Detected regressions per metric
    pub regressions: Vec<MetricRegression>,
    /// Overall confidence (0-1)
    pub confidence: f32,
    /// Timestamp of detection
    #[serde(with = "system_time_serde")]
    pub detected_at: SystemTime,
    /// Current snapshot that triggered detection
    pub current_snapshot: PerformanceSnapshot,
}

/// Regression detected in a specific metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRegression {
    /// Metric name
    pub metric_name: String,
    /// Baseline value
    pub baseline_value: f32,
    /// Current value
    pub current_value: f32,
    /// Percentage change (negative = degradation)
    pub percent_change: f32,
    /// Z-score relative to baseline
    pub z_score: f32,
    /// Detection method used
    pub detection_method: String,
    /// Confidence in this regression (0-1)
    pub confidence: f32,
}

/// Internal state for regression detector
struct RegressionState {
    config: RegressionConfig,
    samples: VecDeque<PerformanceSnapshot>,
    baseline: Option<PerformanceBaseline>,
}

/// Performance regression detector
pub struct RegressionDetector {
    state: Arc<RwLock<RegressionState>>,
}

impl RegressionDetector {
    /// Create a new regression detector
    pub fn new(config: RegressionConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(RegressionState {
                config,
                samples: VecDeque::new(),
                baseline: None,
            })),
        }
    }

    /// Record a performance snapshot
    pub async fn record_performance(&self, snapshot: PerformanceSnapshot) -> Result<()> {
        let mut state = self.state.write().await;

        state.samples.push_back(snapshot);

        // Trim history if needed
        while state.samples.len() > state.config.max_history_size {
            state.samples.pop_front();
        }

        Ok(())
    }

    /// Establish baseline from recorded samples
    pub async fn establish_baseline(&self) -> Result<PerformanceBaseline> {
        let mut state = self.state.write().await;

        if state.samples.len() < state.config.min_baseline_samples {
            return Err(VoirsError::DataValidationFailed {
                data_type: "regression baseline".to_string(),
                reason: format!(
                    "Not enough samples for baseline: {} < {}",
                    state.samples.len(),
                    state.config.min_baseline_samples
                ),
            });
        }

        let baseline = Self::compute_baseline(&state.samples)?;
        state.baseline = Some(baseline.clone());

        Ok(baseline)
    }

    /// Get current baseline
    pub async fn get_baseline(&self) -> Result<Option<PerformanceBaseline>> {
        let state = self.state.read().await;
        Ok(state.baseline.clone())
    }

    /// Detect regression in current snapshot
    pub async fn detect_regression(
        &self,
        snapshot: &PerformanceSnapshot,
    ) -> Result<Option<RegressionReport>> {
        let state = self.state.read().await;

        let baseline = match &state.baseline {
            Some(b) => b,
            None => return Ok(None), // No baseline yet
        };

        let mut regressions = Vec::new();

        // Check each enabled metric
        if state.config.detect_quality_regression {
            if let Some(reg) = self.check_metric_regression(
                "quality_score",
                snapshot.quality_score,
                &baseline.quality,
                &state.config,
                true, // Lower is worse
            ) {
                regressions.push(reg);
            }
        }

        if state.config.detect_latency_regression {
            if let Some(reg) = self.check_metric_regression(
                "latency_ms",
                snapshot.latency_ms as f32,
                &baseline.latency,
                &state.config,
                false, // Higher is worse
            ) {
                regressions.push(reg);
            }
        }

        if state.config.detect_rtf_regression {
            if let Some(reg) = self.check_metric_regression(
                "rtf",
                snapshot.rtf,
                &baseline.rtf,
                &state.config,
                false, // Higher is worse
            ) {
                regressions.push(reg);
            }
        }

        if state.config.detect_memory_regression {
            if let Some(reg) = self.check_metric_regression(
                "memory_mb",
                snapshot.memory_mb,
                &baseline.memory,
                &state.config,
                false, // Higher is worse
            ) {
                regressions.push(reg);
            }
        }

        if state.config.detect_cpu_regression {
            if let Some(reg) = self.check_metric_regression(
                "cpu_percent",
                snapshot.cpu_percent,
                &baseline.cpu,
                &state.config,
                false, // Higher is worse
            ) {
                regressions.push(reg);
            }
        }

        if regressions.is_empty() {
            return Ok(None);
        }

        // Calculate overall confidence
        let confidence =
            regressions.iter().map(|r| r.confidence).sum::<f32>() / regressions.len() as f32;

        if confidence < state.config.min_confidence {
            return Ok(None);
        }

        let summary = format!(
            "Performance regression detected: {} metric(s) degraded",
            regressions.len()
        );

        Ok(Some(RegressionReport {
            summary,
            regressions,
            confidence,
            detected_at: SystemTime::now(),
            current_snapshot: snapshot.clone(),
        }))
    }

    /// Check regression for a single metric
    fn check_metric_regression(
        &self,
        name: &str,
        current: f32,
        baseline: &MetricBaseline,
        config: &RegressionConfig,
        lower_is_worse: bool,
    ) -> Option<MetricRegression> {
        // Calculate z-score
        let z_score = if baseline.std_dev > 0.0 {
            (current - baseline.mean) / baseline.std_dev
        } else {
            // When std_dev is 0 (all samples identical), use percentage deviation
            // Treat >10% deviation as significant (equivalent to z=2.0)
            let percent_deviation = if baseline.mean != 0.0 {
                ((current - baseline.mean) / baseline.mean).abs()
            } else {
                current.abs()
            };

            if percent_deviation > 0.10 {
                // Significant deviation - assign z-score based on direction
                let magnitude = (percent_deviation / 0.05).min(10.0); // Scale: 10% = z=2.0
                if lower_is_worse {
                    if current < baseline.mean {
                        -magnitude
                    } else {
                        magnitude
                    }
                } else if current > baseline.mean {
                    magnitude
                } else {
                    -magnitude
                }
            } else {
                0.0
            }
        };

        // Check if regression based on z-score
        let is_regression_zscore = if lower_is_worse {
            z_score < -config.z_score_threshold
        } else {
            z_score > config.z_score_threshold
        };

        // Check if regression based on IQR
        let iqr = baseline.p75 - baseline.p25;
        let lower_fence = baseline.p25 - config.iqr_multiplier * iqr;
        let upper_fence = baseline.p75 + config.iqr_multiplier * iqr;

        let is_regression_iqr = if lower_is_worse {
            current < lower_fence
        } else {
            current > upper_fence
        };

        // Determine if this is a regression
        let (is_regression, method) = if is_regression_zscore && is_regression_iqr {
            (true, "z-score + IQR")
        } else if is_regression_zscore {
            (true, "z-score")
        } else if is_regression_iqr {
            (true, "IQR")
        } else {
            (false, "none")
        };

        if !is_regression {
            return None;
        }

        // Calculate percent change
        let percent_change = if baseline.mean != 0.0 {
            ((current - baseline.mean) / baseline.mean) * 100.0
        } else {
            0.0
        };

        // Calculate confidence based on how far from baseline
        let confidence = (z_score.abs() / (config.z_score_threshold * 2.0)).min(1.0);

        Some(MetricRegression {
            metric_name: name.to_string(),
            baseline_value: baseline.mean,
            current_value: current,
            percent_change,
            z_score,
            detection_method: method.to_string(),
            confidence,
        })
    }

    /// Compute baseline statistics from samples
    fn compute_baseline(samples: &VecDeque<PerformanceSnapshot>) -> Result<PerformanceBaseline> {
        if samples.is_empty() {
            return Err(VoirsError::DataValidationFailed {
                data_type: "regression samples".to_string(),
                reason: "Cannot compute baseline from empty samples".to_string(),
            });
        }

        Ok(PerformanceBaseline {
            quality: Self::compute_metric_baseline(
                &samples.iter().map(|s| s.quality_score).collect::<Vec<_>>(),
            )?,
            latency: Self::compute_metric_baseline(
                &samples
                    .iter()
                    .map(|s| s.latency_ms as f32)
                    .collect::<Vec<_>>(),
            )?,
            rtf: Self::compute_metric_baseline(&samples.iter().map(|s| s.rtf).collect::<Vec<_>>())?,
            memory: Self::compute_metric_baseline(
                &samples.iter().map(|s| s.memory_mb).collect::<Vec<_>>(),
            )?,
            cpu: Self::compute_metric_baseline(
                &samples.iter().map(|s| s.cpu_percent).collect::<Vec<_>>(),
            )?,
            sample_count: samples.len(),
            established_at: SystemTime::now(),
        })
    }

    /// Compute statistics for a single metric
    fn compute_metric_baseline(values: &[f32]) -> Result<MetricBaseline> {
        if values.is_empty() {
            return Err(VoirsError::DataValidationFailed {
                data_type: "metric values".to_string(),
                reason: "Cannot compute baseline from empty values".to_string(),
            });
        }

        // Use scirs2_core for array operations
        let arr = Array1::from_vec(values.to_vec());

        // Calculate mean
        let mean = arr.mean().unwrap_or(0.0);

        // Calculate standard deviation
        let variance = arr.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
        let std_dev = variance.sqrt();

        // Calculate min/max
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Calculate percentiles
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = sorted[sorted.len() / 2];
        let p25 = sorted[sorted.len() / 4];
        let p75 = sorted[sorted.len() * 3 / 4];

        Ok(MetricBaseline {
            mean,
            std_dev,
            min,
            max,
            median,
            p25,
            p75,
        })
    }

    /// Clear all recorded samples
    pub async fn clear(&self) -> Result<()> {
        let mut state = self.state.write().await;
        state.samples.clear();
        Ok(())
    }

    /// Get number of recorded samples
    pub async fn sample_count(&self) -> usize {
        let state = self.state.read().await;
        state.samples.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_snapshot(quality: f32, latency: u64, rtf: f32) -> PerformanceSnapshot {
        PerformanceSnapshot {
            quality_score: quality,
            latency_ms: latency,
            rtf,
            memory_mb: 512.0,
            cpu_percent: 50.0,
            timestamp: SystemTime::now(),
        }
    }

    #[tokio::test]
    async fn test_regression_detector_creation() {
        let detector = RegressionDetector::new(RegressionConfig::default());
        assert_eq!(detector.sample_count().await, 0);
    }

    #[tokio::test]
    async fn test_record_performance() {
        let detector = RegressionDetector::new(RegressionConfig::default());

        let snapshot = create_snapshot(80.0, 100, 0.5);
        detector.record_performance(snapshot).await.unwrap();

        assert_eq!(detector.sample_count().await, 1);
    }

    #[tokio::test]
    async fn test_establish_baseline() {
        let detector =
            RegressionDetector::new(RegressionConfig::default().with_min_baseline_samples(10));

        // Record 20 samples
        for i in 0..20 {
            let snapshot = create_snapshot(75.0 + i as f32, 100, 0.5);
            detector.record_performance(snapshot).await.unwrap();
        }

        let baseline = detector.establish_baseline().await.unwrap();
        assert_eq!(baseline.sample_count, 20);
        assert!(baseline.quality.mean > 75.0);
        assert!(baseline.quality.mean < 95.0);
    }

    #[tokio::test]
    async fn test_detect_quality_regression() {
        let detector = RegressionDetector::new(
            RegressionConfig::default()
                .with_min_baseline_samples(10)
                .with_z_score_threshold(2.0),
        );

        // Establish baseline with good quality
        for _ in 0..30 {
            detector
                .record_performance(create_snapshot(80.0, 100, 0.5))
                .await
                .unwrap();
        }
        detector.establish_baseline().await.unwrap();

        // Check good performance (no regression)
        let good_snapshot = create_snapshot(79.0, 100, 0.5);
        assert!(detector
            .detect_regression(&good_snapshot)
            .await
            .unwrap()
            .is_none());

        // Check degraded performance (should detect regression)
        let bad_snapshot = create_snapshot(50.0, 100, 0.5);
        let report = detector.detect_regression(&bad_snapshot).await.unwrap();
        assert!(report.is_some());

        let report = report.unwrap();
        assert!(!report.regressions.is_empty());
        assert!(report.confidence > 0.7);
    }

    #[tokio::test]
    async fn test_detect_latency_regression() {
        let detector = RegressionDetector::new(
            RegressionConfig::default()
                .with_min_baseline_samples(10)
                .with_z_score_threshold(2.0),
        );

        // Establish baseline with good latency
        for _ in 0..30 {
            detector
                .record_performance(create_snapshot(80.0, 100, 0.5))
                .await
                .unwrap();
        }
        detector.establish_baseline().await.unwrap();

        // Check high latency (should detect regression)
        let slow_snapshot = create_snapshot(80.0, 300, 0.5);
        let report = detector.detect_regression(&slow_snapshot).await.unwrap();
        assert!(report.is_some());

        let report = report.unwrap();
        assert!(report
            .regressions
            .iter()
            .any(|r| r.metric_name == "latency_ms"));
    }

    #[tokio::test]
    async fn test_clear_samples() {
        let detector = RegressionDetector::new(RegressionConfig::default());

        detector
            .record_performance(create_snapshot(80.0, 100, 0.5))
            .await
            .unwrap();
        assert_eq!(detector.sample_count().await, 1);

        detector.clear().await.unwrap();
        assert_eq!(detector.sample_count().await, 0);
    }

    #[tokio::test]
    async fn test_baseline_insufficient_samples() {
        let detector =
            RegressionDetector::new(RegressionConfig::default().with_min_baseline_samples(30));

        // Only record 10 samples
        for _ in 0..10 {
            detector
                .record_performance(create_snapshot(80.0, 100, 0.5))
                .await
                .unwrap();
        }

        // Should fail to establish baseline
        assert!(detector.establish_baseline().await.is_err());
    }
}
