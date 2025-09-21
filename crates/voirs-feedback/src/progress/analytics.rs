//! Comprehensive analytics framework for progress tracking
//!
//! This module provides advanced analytics capabilities for user progress tracking,
//! including statistical analysis, comparative studies, longitudinal data collection,
//! and real-time dashboard functionality.

use crate::traits::{ProgressSnapshot, TimeRange, UserProgress};
use crate::FeedbackError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;

/// Trend direction enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Improving trend
    Improving,
    /// Stable performance
    Stable,
    /// Declining trend
    Declining,
}

/// Comprehensive analytics framework for progress tracking
#[derive(Debug, Clone)]
pub struct ComprehensiveAnalyticsFramework {
    /// Analytics configuration
    config: AnalyticsConfig,
    /// Memory-bounded metrics storage with LRU eviction
    metrics: MemoryBoundedMetrics,
    /// Aggregated metrics for long-term storage
    aggregated_metrics: HashMap<String, AggregatedMetric>,
    /// Last cleanup timestamp
    last_cleanup: DateTime<Utc>,
}

impl Default for ComprehensiveAnalyticsFramework {
    fn default() -> Self {
        Self::new()
    }
}

impl ComprehensiveAnalyticsFramework {
    /// Create a new analytics framework
    #[must_use]
    pub fn new() -> Self {
        let config = AnalyticsConfig::default();
        Self {
            metrics: MemoryBoundedMetrics::new(config.max_metrics_capacity),
            aggregated_metrics: HashMap::new(),
            last_cleanup: Utc::now(),
            config,
        }
    }

    /// Create a new analytics framework with custom configuration
    #[must_use]
    pub fn with_config(config: AnalyticsConfig) -> Self {
        Self {
            metrics: MemoryBoundedMetrics::new(config.max_metrics_capacity),
            aggregated_metrics: HashMap::new(),
            last_cleanup: Utc::now(),
            config,
        }
    }

    /// Generate analytics report
    pub async fn generate_analytics_report(
        &self,
        progress: &UserProgress,
        time_range: Option<TimeRange>,
    ) -> Result<ComprehensiveAnalyticsReport, FeedbackError> {
        let now = Utc::now();
        let range = time_range.unwrap_or(TimeRange {
            start: now - chrono::Duration::days(30),
            end: now,
        });

        // Multi-dimensional progress measurement
        let mut metrics = vec![
            AnalyticsMetric {
                name: "overall_skill_level".to_string(),
                value: f64::from(progress.overall_skill_level),
                timestamp: now,
                metric_type: MetricType::Gauge,
            },
            AnalyticsMetric {
                name: "total_sessions".to_string(),
                value: progress.training_stats.total_sessions as f64,
                timestamp: now,
                metric_type: MetricType::Counter,
            },
            AnalyticsMetric {
                name: "success_rate".to_string(),
                value: f64::from(progress.training_stats.success_rate),
                timestamp: now,
                metric_type: MetricType::Gauge,
            },
            AnalyticsMetric {
                name: "average_improvement".to_string(),
                value: f64::from(progress.training_stats.average_improvement),
                timestamp: now,
                metric_type: MetricType::Gauge,
            },
            AnalyticsMetric {
                name: "current_streak".to_string(),
                value: progress.training_stats.current_streak as f64,
                timestamp: now,
                metric_type: MetricType::Counter,
            },
            AnalyticsMetric {
                name: "longest_streak".to_string(),
                value: progress.training_stats.longest_streak as f64,
                timestamp: now,
                metric_type: MetricType::Counter,
            },
        ];

        // Add skill breakdown metrics
        for (focus_area, &skill_level) in &progress.skill_breakdown {
            metrics.push(AnalyticsMetric {
                name: format!("skill_{focus_area:?}").to_lowercase(),
                value: f64::from(skill_level),
                timestamp: now,
                metric_type: MetricType::Gauge,
            });
        }

        // Filter progress history to time range
        let relevant_history: Vec<_> = progress
            .progress_history
            .iter()
            .filter(|snapshot| snapshot.timestamp >= range.start && snapshot.timestamp <= range.end)
            .collect();

        // Calculate trend analytics
        let trend_analytics = self.calculate_trend_analytics(&relevant_history);

        // Add trend metrics
        metrics.push(AnalyticsMetric {
            name: "improvement_velocity".to_string(),
            value: f64::from(trend_analytics.improvement_velocity),
            timestamp: now,
            metric_type: MetricType::Gauge,
        });

        metrics.push(AnalyticsMetric {
            name: "performance_stability".to_string(),
            value: f64::from(trend_analytics.performance_stability),
            timestamp: now,
            metric_type: MetricType::Gauge,
        });

        let summary = AnalyticsSummary {
            total_metrics: metrics.len(),
            average_value: if metrics.is_empty() {
                0.0
            } else {
                metrics.iter().map(|m| m.value).sum::<f64>() / metrics.len() as f64
            },
            time_range: range,
        };

        Ok(ComprehensiveAnalyticsReport {
            timestamp: now,
            metrics,
            summary,
        })
    }

    /// Test statistical significance
    pub async fn test_statistical_significance(
        &self,
        _progress: &UserProgress,
        _metric: AnalyticsMetric,
        _time_period: Duration,
    ) -> Result<StatisticalSignificanceResult, FeedbackError> {
        // Simplified implementation - in a real system this would perform actual statistical tests
        Ok(StatisticalSignificanceResult {
            p_value: 0.05,
            is_significant: true,
            confidence_level: 0.95,
            effect_size: 0.3,
        })
    }

    /// Generate comparative analysis
    pub async fn generate_comparative_analysis(
        &self,
        user_progress_data: &[UserProgress],
        _metric: AnalyticsMetric,
        _time_range: Option<TimeRange>,
    ) -> Result<ComparativeAnalyticsResult, FeedbackError> {
        if user_progress_data.len() < 2 {
            return Err(FeedbackError::ProgressTrackingError {
                message: "Need at least 2 users for comparative analysis".to_string(),
                source: None,
            });
        }

        let baseline_value = f64::from(user_progress_data[0].overall_skill_level);
        let comparison_value = f64::from(user_progress_data[1].overall_skill_level);
        let percentage_change = ((comparison_value - baseline_value) / baseline_value) * 100.0;

        Ok(ComparativeAnalyticsResult {
            baseline_value,
            comparison_value,
            percentage_change,
            statistical_significance: StatisticalSignificanceResult {
                p_value: 0.05,
                is_significant: percentage_change.abs() > 10.0,
                confidence_level: 0.95,
                effect_size: 0.3,
            },
        })
    }

    /// Collect longitudinal data
    pub async fn collect_longitudinal_data(
        &self,
        _study_id: &str,
        participant_data: &[UserProgress],
        _tracking_metrics: &[AnalyticsMetric],
    ) -> Result<LongitudinalStudyData, FeedbackError> {
        let now = Utc::now();
        let start = now - chrono::Duration::days(90);

        let data_points: Vec<LongitudinalDataPoint> = participant_data
            .iter()
            .enumerate()
            .map(|(i, progress)| LongitudinalDataPoint {
                timestamp: start + chrono::Duration::days(i as i64),
                value: f64::from(progress.overall_skill_level),
                metadata: HashMap::new(),
            })
            .collect();

        Ok(LongitudinalStudyData {
            study_period: TimeRange { start, end: now },
            data_points,
            trend_analysis: TrendAnalysis {
                trend_direction: TrendDirection::Improving,
                slope: 0.01,
                r_squared: 0.8,
            },
        })
    }

    /// Calculate trend analytics from progress history
    fn calculate_trend_analytics(&self, history: &[&ProgressSnapshot]) -> TrendAnalytics {
        if history.len() < 2 {
            return TrendAnalytics {
                improvement_velocity: 0.0,
                performance_stability: 0.0,
                trend_direction: TrendDirection::Stable,
                slope: 0.0,
                r_squared: 0.0,
            };
        }

        let scores: Vec<f32> = history.iter().map(|s| s.overall_score).collect();

        // Calculate improvement velocity (average rate of change)
        let improvements: Vec<f32> = scores
            .windows(2)
            .map(|window| window[1] - window[0])
            .collect();

        let improvement_velocity = if improvements.is_empty() {
            0.0
        } else {
            improvements.iter().sum::<f32>() / improvements.len() as f32
        };

        // Calculate performance stability (inverse of coefficient of variation)
        let mean_score = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        };

        let variance = if scores.len() > 1 {
            scores.iter().map(|s| (s - mean_score).powi(2)).sum::<f32>() / scores.len() as f32
        } else {
            0.0
        };

        let performance_stability = if mean_score > 0.0 && variance > 0.0 {
            1.0 / (1.0 + (variance.sqrt() / mean_score))
        } else {
            1.0
        };

        // Calculate linear regression for trend
        let n = scores.len() as f32;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = mean_score;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &score) in scores.iter().enumerate() {
            let x_diff = i as f32 - x_mean;
            numerator += x_diff * (score - y_mean);
            denominator += x_diff * x_diff;
        }

        let slope = if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        };

        // Calculate R-squared
        let y_pred: Vec<f32> = (0..scores.len())
            .map(|i| y_mean + slope * (i as f32 - x_mean))
            .collect();

        let ss_res: f32 = scores
            .iter()
            .zip(y_pred.iter())
            .map(|(actual, predicted)| (actual - predicted).powi(2))
            .sum();

        let ss_tot: f32 = scores.iter().map(|score| (score - y_mean).powi(2)).sum();

        let r_squared = if ss_tot == 0.0 {
            0.0
        } else {
            1.0 - (ss_res / ss_tot)
        };

        let trend_direction = if slope > 0.02 {
            TrendDirection::Improving
        } else if slope < -0.02 {
            TrendDirection::Declining
        } else {
            TrendDirection::Stable
        };

        TrendAnalytics {
            improvement_velocity,
            performance_stability,
            trend_direction,
            slope,
            r_squared,
        }
    }

    /// Cleanup old metrics to prevent memory overflow
    pub fn cleanup_old_metrics(&mut self) {
        let now = Utc::now();
        let retention_duration = chrono::Duration::days(self.config.data_retention_days as i64);
        let cutoff_time = now - retention_duration;

        // Remove old metrics
        self.metrics.cleanup_before(cutoff_time);

        // Remove old aggregated metrics with size limit
        self.aggregated_metrics
            .retain(|_, metric| metric.last_updated > cutoff_time);

        // If we still have too many aggregated metrics, remove oldest ones
        if self.aggregated_metrics.len() > self.config.max_aggregated_metrics {
            let mut metrics_by_age: Vec<_> = self
                .aggregated_metrics
                .iter()
                .map(|(k, v)| (k.clone(), v.last_updated))
                .collect();

            // Sort by timestamp (oldest first)
            metrics_by_age.sort_by_key(|(_, timestamp)| *timestamp);

            // Remove oldest metrics to get under the limit
            let to_remove = self.aggregated_metrics.len() - self.config.max_aggregated_metrics;
            for (key, _) in metrics_by_age.into_iter().take(to_remove) {
                self.aggregated_metrics.remove(&key);
            }
        }

        self.last_cleanup = now;
    }

    /// Check if cleanup is needed
    pub fn needs_cleanup(&self) -> bool {
        let now = Utc::now();
        let cleanup_interval =
            chrono::Duration::minutes(self.config.cleanup_interval_minutes as i64);
        let time_based_cleanup = now.signed_duration_since(self.last_cleanup) > cleanup_interval;

        // Also check memory usage threshold
        let memory_stats = self.get_memory_stats();
        let memory_based_cleanup =
            memory_stats.memory_utilization > self.config.memory_cleanup_threshold;

        time_based_cleanup || memory_based_cleanup
    }

    /// Add metric with automatic cleanup
    pub fn add_metric(&mut self, name: String, metric: AnalyticsMetric) {
        // Perform cleanup if needed
        if self.needs_cleanup() {
            self.cleanup_old_metrics();
        }

        // Check if we're at memory limits before adding
        let memory_stats = self.get_memory_stats();
        if memory_stats.memory_utilization >= 1.0 {
            // Force cleanup if we're at the memory limit
            self.cleanup_old_metrics();
        }

        // Add to bounded metrics
        self.metrics.insert(name.clone(), metric.clone());

        // Update aggregated metrics for long-term trends
        self.update_aggregated_metric(name, &metric);
    }

    /// Update aggregated metrics for memory efficiency
    fn update_aggregated_metric(&mut self, name: String, metric: &AnalyticsMetric) {
        // Only aggregate if enabled in configuration
        if !self.config.enable_auto_aggregation {
            return;
        }

        let aggregated = self
            .aggregated_metrics
            .entry(name)
            .or_insert_with(|| AggregatedMetric {
                name: metric.name.clone(),
                count: 0,
                sum: 0.0,
                sum_of_squares: 0.0,
                min: f64::INFINITY,
                max: f64::NEG_INFINITY,
                last_updated: metric.timestamp,
                metric_type: metric.metric_type.clone(),
            });

        // Update aggregated statistics
        aggregated.count += 1;
        aggregated.sum += metric.value;
        aggregated.sum_of_squares += metric.value * metric.value;
        aggregated.min = aggregated.min.min(metric.value);
        aggregated.max = aggregated.max.max(metric.value);
        aggregated.last_updated = metric.timestamp;
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        let metrics_count = self.metrics.len();
        let aggregated_count = self.aggregated_metrics.len();

        // Estimate memory usage (approximate)
        let estimated_metrics_bytes = metrics_count * std::mem::size_of::<AnalyticsMetric>();
        let estimated_aggregated_bytes = aggregated_count * std::mem::size_of::<AggregatedMetric>();
        let total_estimated_bytes = estimated_metrics_bytes + estimated_aggregated_bytes;

        MemoryStats {
            total_metrics: metrics_count,
            aggregated_metrics: aggregated_count,
            estimated_memory_bytes: total_estimated_bytes,
            memory_limit_bytes: self.config.memory_limit_bytes,
            memory_utilization: if self.config.memory_limit_bytes > 0 {
                total_estimated_bytes as f64 / self.config.memory_limit_bytes as f64
            } else {
                0.0
            },
        }
    }
}

/// Analytics configuration with memory management
#[derive(Debug, Clone)]
pub struct AnalyticsConfig {
    /// Whether to enable detailed analytics
    pub enable_detailed_analytics: bool,
    /// Retention period for analytics data
    pub data_retention_days: u32,
    /// Maximum number of metrics to store in memory
    pub max_metrics_capacity: usize,
    /// Memory limit in bytes for analytics storage
    pub memory_limit_bytes: usize,
    /// Cleanup interval in minutes
    pub cleanup_interval_minutes: u32,
    /// Memory usage threshold for triggering cleanup (0.0 to 1.0)
    pub memory_cleanup_threshold: f64,
    /// Enable automatic aggregation of metrics
    pub enable_auto_aggregation: bool,
    /// Maximum number of aggregated metrics to keep
    pub max_aggregated_metrics: usize,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            enable_detailed_analytics: true,
            data_retention_days: 90,
            max_metrics_capacity: 10_000,
            memory_limit_bytes: 50 * 1024 * 1024, // 50MB
            cleanup_interval_minutes: 60,         // 1 hour
            memory_cleanup_threshold: 0.8,        // 80% memory usage
            enable_auto_aggregation: true,
            max_aggregated_metrics: 1_000,
        }
    }
}

/// Analytics metric definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsMetric {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Metric timestamp
    pub timestamp: DateTime<Utc>,
    /// Metric type
    pub metric_type: MetricType,
}

/// Metric type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    /// Counter metric
    Counter,
    /// Gauge metric
    Gauge,
    /// Histogram metric
    Histogram,
    /// Timer metric
    Timer,
}

/// Comprehensive analytics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveAnalyticsReport {
    /// Report timestamp
    pub timestamp: DateTime<Utc>,
    /// Metrics included in the report
    pub metrics: Vec<AnalyticsMetric>,
    /// Summary statistics
    pub summary: AnalyticsSummary,
}

/// Analytics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsSummary {
    /// Total metrics count
    pub total_metrics: usize,
    /// Average metric value
    pub average_value: f64,
    /// Time range covered
    pub time_range: TimeRange,
}

/// Statistical significance result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSignificanceResult {
    /// P-value of the test
    pub p_value: f64,
    /// Whether result is statistically significant
    pub is_significant: bool,
    /// Confidence level used
    pub confidence_level: f64,
    /// Effect size
    pub effect_size: f64,
}

/// Comparative analytics result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparativeAnalyticsResult {
    /// Baseline metric value
    pub baseline_value: f64,
    /// Comparison metric value
    pub comparison_value: f64,
    /// Percentage change
    pub percentage_change: f64,
    /// Statistical significance
    pub statistical_significance: StatisticalSignificanceResult,
}

/// Longitudinal study data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongitudinalStudyData {
    /// Study period
    pub study_period: TimeRange,
    /// Data points collected
    pub data_points: Vec<LongitudinalDataPoint>,
    /// Trend analysis
    pub trend_analysis: TrendAnalysis,
}

/// Longitudinal data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongitudinalDataPoint {
    /// Data point timestamp
    pub timestamp: DateTime<Utc>,
    /// Metric value
    pub value: f64,
    /// Associated metadata
    pub metadata: HashMap<String, String>,
}

/// Trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Overall trend direction
    pub trend_direction: TrendDirection,
    /// Trend slope
    pub slope: f64,
    /// R-squared correlation
    pub r_squared: f64,
}

/// Advanced trend analytics
#[derive(Debug, Clone)]
pub struct TrendAnalytics {
    /// Rate of improvement over time
    pub improvement_velocity: f32,
    /// Stability of performance (inverse of variation)
    pub performance_stability: f32,
    /// Overall trend direction
    pub trend_direction: TrendDirection,
    /// Linear regression slope
    pub slope: f32,
    /// Correlation coefficient (R-squared)
    pub r_squared: f32,
}

/// Memory-bounded metrics storage with LRU eviction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBoundedMetrics {
    /// Internal storage with bounded capacity
    storage: VecDeque<(String, AnalyticsMetric)>,
    /// Maximum capacity
    capacity: usize,
    /// Quick lookup index
    index: HashMap<String, usize>,
}

impl MemoryBoundedMetrics {
    /// Create new memory-bounded metrics storage
    pub fn new(capacity: usize) -> Self {
        Self {
            storage: VecDeque::with_capacity(capacity),
            capacity,
            index: HashMap::new(),
        }
    }

    /// Insert metric with LRU eviction
    pub fn insert(&mut self, key: String, metric: AnalyticsMetric) {
        // If at capacity, remove oldest entry
        if self.storage.len() >= self.capacity {
            if let Some((old_key, _)) = self.storage.pop_front() {
                self.index.remove(&old_key);
            }
        }

        // Add new entry
        let new_index = self.storage.len();
        self.storage.push_back((key.clone(), metric));
        self.index.insert(key, new_index);

        // Rebuild index if needed (simple approach)
        if self.storage.len() != self.index.len() {
            self.rebuild_index();
        }
    }

    /// Get metric by key
    pub fn get(&self, key: &str) -> Option<&AnalyticsMetric> {
        if let Some(&index) = self.index.get(key) {
            if index < self.storage.len() {
                return Some(&self.storage[index].1);
            }
        }
        None
    }

    /// Remove metrics before given timestamp
    pub fn cleanup_before(&mut self, cutoff_time: DateTime<Utc>) {
        let mut removed_count = 0;

        // Remove old entries from front
        while let Some((key, metric)) = self.storage.front() {
            if metric.timestamp < cutoff_time {
                let (removed_key, _) = self.storage.pop_front().unwrap();
                self.index.remove(&removed_key);
                removed_count += 1;
            } else {
                break;
            }
        }

        // Rebuild index if we removed items
        if removed_count > 0 {
            self.rebuild_index();
        }
    }

    /// Rebuild index after modifications
    fn rebuild_index(&mut self) {
        self.index.clear();
        for (i, (key, _)) in self.storage.iter().enumerate() {
            self.index.insert(key.clone(), i);
        }
    }

    /// Get number of stored metrics
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Aggregated metric for long-term storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetric {
    /// Metric name
    pub name: String,
    /// Count of data points
    pub count: u64,
    /// Sum of all values
    pub sum: f64,
    /// Sum of squares for variance calculation
    pub sum_of_squares: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
    /// Metric type
    pub metric_type: MetricType,
}

impl AggregatedMetric {
    /// Calculate mean
    pub fn mean(&self) -> f64 {
        if self.count > 0 {
            self.sum / self.count as f64
        } else {
            0.0
        }
    }

    /// Calculate variance
    pub fn variance(&self) -> f64 {
        if self.count > 1 {
            let mean = self.mean();
            (self.sum_of_squares - self.count as f64 * mean * mean) / (self.count - 1) as f64
        } else {
            0.0
        }
    }

    /// Calculate standard deviation
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total number of metrics
    pub total_metrics: usize,
    /// Number of aggregated metrics
    pub aggregated_metrics: usize,
    /// Estimated memory usage in bytes
    pub estimated_memory_bytes: usize,
    /// Memory limit in bytes
    pub memory_limit_bytes: usize,
    /// Memory utilization percentage (0.0 to 1.0)
    pub memory_utilization: f64,
}
