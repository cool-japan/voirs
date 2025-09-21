//! Enhanced Performance Monitoring System
//!
//! This module provides advanced performance monitoring capabilities specifically designed
//! to track and ensure compliance with the critical success factors for technical performance:
//! - Sub-100ms real-time feedback latency
//! - >99.9% system uptime
//! - <2% error rate across all features
//! - >95% cross-platform compatibility

use crate::metrics_dashboard::{
    CompatibilityRecord, ErrorRecord, ErrorSeverity, LatencyRecord, ServiceStatus, UptimeRecord,
};
use crate::traits::*;
use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio::time::{interval, sleep};

/// Enhanced performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedPerformanceConfig {
    /// Target maximum latency in milliseconds for real-time operations
    pub target_max_latency_ms: u32,
    /// Target minimum uptime percentage (99.9% = 0.999)
    pub target_min_uptime: f32,
    /// Target maximum error rate (2% = 0.02)
    pub target_max_error_rate: f32,
    /// Target minimum compatibility score (95% = 0.95)
    pub target_min_compatibility: f32,
    /// Performance monitoring interval in seconds
    pub monitoring_interval_seconds: u64,
    /// Number of historical performance samples to keep
    pub max_performance_samples: usize,
    /// Enable automatic performance optimization
    pub enable_auto_optimization: bool,
    /// Alert threshold multiplier for performance degradation detection
    pub alert_threshold_multiplier: f32,
    /// Enable predictive performance analytics
    pub enable_predictive_analytics: bool,
}

impl Default for EnhancedPerformanceConfig {
    fn default() -> Self {
        Self {
            target_max_latency_ms: 100,
            target_min_uptime: 0.999,
            target_max_error_rate: 0.02,
            target_min_compatibility: 0.95,
            monitoring_interval_seconds: 10,
            max_performance_samples: 10000,
            enable_auto_optimization: true,
            alert_threshold_multiplier: 1.5,
            enable_predictive_analytics: true,
        }
    }
}

/// Real-time performance tracker with advanced monitoring capabilities
#[derive(Debug, Clone)]
pub struct EnhancedPerformanceMonitor {
    /// Performance statistics
    stats: Arc<RwLock<PerformanceStatistics>>,
    /// Latency measurements
    latency_tracker: Arc<RwLock<LatencyTracker>>,
    /// Uptime monitoring
    uptime_monitor: Arc<RwLock<UptimeMonitor>>,
    /// Error rate tracking
    error_tracker: Arc<RwLock<ErrorTracker>>,
    /// Compatibility monitoring
    compatibility_monitor: Arc<RwLock<CompatibilityMonitor>>,
    /// Performance optimization engine
    optimizer: Arc<RwLock<PerformanceOptimizer>>,
    /// Configuration
    config: EnhancedPerformanceConfig,
}

/// Comprehensive performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStatistics {
    /// Current average latency in milliseconds
    pub current_avg_latency_ms: f32,
    /// Current system uptime percentage
    pub current_uptime_percentage: f32,
    /// Current error rate
    pub current_error_rate: f32,
    /// Current compatibility score
    pub current_compatibility_score: f32,
    /// Performance trend indicators
    pub performance_trends: PerformanceTrends,
    /// Alert status
    pub alert_status: AlertStatus,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
    /// Performance health score (0.0 to 1.0)
    pub health_score: f32,
}

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    /// Latency trend (improving, stable, degrading)
    pub latency_trend: TrendDirection,
    /// Uptime trend
    pub uptime_trend: TrendDirection,
    /// Error rate trend
    pub error_rate_trend: TrendDirection,
    /// Compatibility trend
    pub compatibility_trend: TrendDirection,
    /// Overall performance trend
    pub overall_trend: TrendDirection,
    /// Confidence level of trend analysis (0.0 to 1.0)
    pub trend_confidence: f32,
}

/// Trend direction enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Unknown,
}

/// Performance alert status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertStatus {
    /// Active alerts
    pub active_alerts: Vec<PerformanceAlert>,
    /// Total alert count in current period
    pub alert_count: u32,
    /// Last alert timestamp
    pub last_alert: Option<DateTime<Utc>>,
    /// Overall alert severity
    pub overall_severity: AlertSeverity,
}

/// Performance alert structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    /// Alert ID
    pub id: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Timestamp when alert was raised
    pub timestamp: DateTime<Utc>,
    /// Metric that triggered the alert
    pub trigger_metric: String,
    /// Current value of the metric
    pub current_value: f32,
    /// Threshold value that was exceeded
    pub threshold_value: f32,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
}

/// Alert type enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlertType {
    LatencyThresholdExceeded,
    UptimeThresholdViolated,
    ErrorRateThresholdExceeded,
    CompatibilityScoreDecreased,
    PerformanceDegradation,
    SystemOverload,
    PredictiveAlert,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Advanced latency tracking with statistical analysis
#[derive(Debug, Clone)]
pub struct LatencyTracker {
    /// Recent latency measurements
    measurements: VecDeque<LatencyMeasurement>,
    /// Statistical summary
    stats: LatencyStatistics,
    /// Per-operation latency tracking
    operation_stats: HashMap<String, LatencyStatistics>,
    /// Latency percentiles
    percentiles: LatencyPercentiles,
}

/// Individual latency measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMeasurement {
    pub timestamp: DateTime<Utc>,
    pub operation: String,
    pub latency_ms: u32,
    pub platform: String,
    pub user_id: String,
    pub context: LatencyContext,
}

/// Latency measurement context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyContext {
    /// CPU usage at time of measurement
    pub cpu_usage: f32,
    /// Memory usage at time of measurement
    pub memory_usage: f32,
    /// Network latency if applicable
    pub network_latency_ms: Option<u32>,
    /// Concurrent operation count
    pub concurrent_operations: u32,
}

/// Statistical analysis of latency data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStatistics {
    /// Average latency
    pub avg_latency_ms: f32,
    /// Median latency
    pub median_latency_ms: f32,
    /// Minimum latency
    pub min_latency_ms: u32,
    /// Maximum latency
    pub max_latency_ms: u32,
    /// Standard deviation
    pub std_deviation: f32,
    /// Total measurements
    pub measurement_count: u32,
    /// Measurements exceeding target threshold
    pub threshold_violations: u32,
}

/// Latency percentile analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    pub p50: u32, // Median
    pub p75: u32,
    pub p90: u32,
    pub p95: u32,
    pub p99: u32,
    pub p99_9: u32,
}

/// Enhanced uptime monitoring with incident tracking
#[derive(Debug, Clone)]
pub struct UptimeMonitor {
    /// Service uptime records
    service_records: HashMap<String, ServiceUptimeRecord>,
    /// Incident history
    incidents: VecDeque<UptimeIncident>,
    /// Overall uptime statistics
    overall_stats: UptimeStatistics,
}

/// Service-specific uptime record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceUptimeRecord {
    /// Service identifier
    pub service_name: String,
    /// Current status
    pub current_status: ServiceStatus,
    /// Uptime percentage for current period
    pub uptime_percentage: f32,
    /// Total uptime in milliseconds
    pub total_uptime_ms: u64,
    /// Total downtime in milliseconds
    pub total_downtime_ms: u64,
    /// Last status change
    pub last_status_change: DateTime<Utc>,
    /// Status change history
    pub status_history: VecDeque<StatusChange>,
}

/// Status change event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusChange {
    pub timestamp: DateTime<Utc>,
    pub from_status: ServiceStatus,
    pub to_status: ServiceStatus,
    pub reason: String,
    pub impact_severity: AlertSeverity,
}

/// Uptime incident record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UptimeIncident {
    /// Incident ID
    pub id: String,
    /// Service affected
    pub service_name: String,
    /// Incident start time
    pub start_time: DateTime<Utc>,
    /// Incident end time (None if ongoing)
    pub end_time: Option<DateTime<Utc>>,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Incident severity
    pub severity: AlertSeverity,
    /// Root cause description
    pub root_cause: String,
    /// Resolution actions taken
    pub resolution_actions: Vec<String>,
    /// Impact on users
    pub user_impact: UserImpact,
}

/// User impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserImpact {
    /// Number of users affected
    pub affected_users: u32,
    /// Percentage of total user base affected
    pub affected_percentage: f32,
    /// Service degradation level
    pub degradation_level: DegradationLevel,
    /// User-reported issues
    pub user_reports: u32,
}

/// Service degradation levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DegradationLevel {
    None,
    Minor,
    Moderate,
    Severe,
    Complete,
}

/// Overall uptime statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UptimeStatistics {
    /// Overall system uptime percentage
    pub overall_uptime_percentage: f32,
    /// Total number of incidents
    pub total_incidents: u32,
    /// Average incident duration in minutes
    pub avg_incident_duration_minutes: f32,
    /// Mean time to recovery (MTTR) in minutes
    pub mttr_minutes: f32,
    /// Mean time between failures (MTBF) in hours
    pub mtbf_hours: f32,
}

/// Advanced error tracking and categorization
#[derive(Debug, Clone)]
pub struct ErrorTracker {
    /// Error records
    error_records: VecDeque<ErrorRecord>,
    /// Error statistics by category
    error_stats: HashMap<String, ErrorCategoryStats>,
    /// Overall error statistics
    overall_stats: ErrorStatistics,
    /// Error pattern analysis
    pattern_analyzer: ErrorPatternAnalyzer,
}

/// Error statistics for a specific category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorCategoryStats {
    /// Error category name
    pub category: String,
    /// Total error count
    pub error_count: u32,
    /// Error rate (errors per unit time)
    pub error_rate: f32,
    /// Most common error messages
    pub common_errors: Vec<String>,
    /// Trend direction
    pub trend: TrendDirection,
    /// Last error timestamp
    pub last_error: Option<DateTime<Utc>>,
}

/// Overall error statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStatistics {
    /// Total error count
    pub total_errors: u32,
    /// Current error rate
    pub current_error_rate: f32,
    /// Error rate by severity
    pub error_rate_by_severity: HashMap<ErrorSeverity, f32>,
    /// Top error categories
    pub top_error_categories: Vec<String>,
    /// Error resolution statistics
    pub resolution_stats: ErrorResolutionStats,
}

/// Error resolution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResolutionStats {
    /// Average time to resolve errors in minutes
    pub avg_resolution_time_minutes: f32,
    /// Percentage of auto-resolved errors
    pub auto_resolution_rate: f32,
    /// Percentage of unresolved errors
    pub unresolved_rate: f32,
    /// Recurrence rate of resolved errors
    pub recurrence_rate: f32,
}

/// Error pattern analysis engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPatternAnalyzer {
    /// Detected error patterns
    pub detected_patterns: Vec<ErrorPattern>,
    /// Pattern confidence scores
    pub pattern_confidence: HashMap<String, f32>,
    /// Predicted error hotspots
    pub predicted_hotspots: Vec<ErrorHotspot>,
}

/// Error pattern structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    /// Pattern ID
    pub id: String,
    /// Pattern description
    pub description: String,
    /// Pattern frequency
    pub frequency: u32,
    /// Pattern conditions
    pub conditions: Vec<String>,
    /// Suggested mitigations
    pub mitigations: Vec<String>,
}

/// Error hotspot prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorHotspot {
    /// Component or area name
    pub component: String,
    /// Predicted error probability
    pub error_probability: f32,
    /// Predicted impact severity
    pub predicted_severity: AlertSeverity,
    /// Recommended preventive actions
    pub preventive_actions: Vec<String>,
}

/// Cross-platform compatibility monitoring
#[derive(Debug, Clone)]
pub struct CompatibilityMonitor {
    /// Platform compatibility records
    platform_records: HashMap<String, PlatformCompatibilityRecord>,
    /// Compatibility test results
    test_results: HashMap<String, CompatibilityTestSuite>,
    /// Overall compatibility statistics
    overall_stats: CompatibilityStatistics,
}

/// Platform-specific compatibility record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCompatibilityRecord {
    /// Platform identifier
    pub platform: String,
    /// Platform version
    pub version: String,
    /// Overall compatibility score
    pub compatibility_score: f32,
    /// Feature compatibility breakdown
    pub feature_compatibility: HashMap<String, f32>,
    /// Known issues
    pub known_issues: Vec<CompatibilityIssue>,
    /// Last tested timestamp
    pub last_tested: DateTime<Utc>,
}

/// Compatibility issue record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityIssue {
    /// Issue ID
    pub id: String,
    /// Issue description
    pub description: String,
    /// Affected features
    pub affected_features: Vec<String>,
    /// Severity level
    pub severity: AlertSeverity,
    /// Workaround available
    pub has_workaround: bool,
    /// Workaround description
    pub workaround: Option<String>,
    /// Fix status
    pub fix_status: FixStatus,
}

/// Fix status enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FixStatus {
    Open,
    InProgress,
    Fixed,
    WontFix,
    Duplicate,
}

/// Compatibility test suite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityTestSuite {
    /// Test suite name
    pub name: String,
    /// Platform being tested
    pub platform: String,
    /// Test results
    pub test_results: HashMap<String, TestResult>,
    /// Overall test score
    pub overall_score: f32,
    /// Test execution timestamp
    pub executed_at: DateTime<Utc>,
}

/// Individual test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test name
    pub test_name: String,
    /// Test passed status
    pub passed: bool,
    /// Test score (0.0 to 1.0)
    pub score: f32,
    /// Execution time in milliseconds
    pub execution_time_ms: u32,
    /// Error message if failed
    pub error_message: Option<String>,
}

/// Overall compatibility statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityStatistics {
    /// Average compatibility score across all platforms
    pub avg_compatibility_score: f32,
    /// Platform coverage percentage
    pub platform_coverage: f32,
    /// Number of supported platforms
    pub supported_platforms: u32,
    /// Number of known compatibility issues
    pub known_issues_count: u32,
    /// Percentage of issues with workarounds
    pub workaround_availability: f32,
}

/// Performance optimization engine
#[derive(Debug, Clone)]
pub struct PerformanceOptimizer {
    /// Optimization strategies
    strategies: Vec<OptimizationStrategy>,
    /// Applied optimizations
    applied_optimizations: HashMap<String, AppliedOptimization>,
    /// Optimization effectiveness tracking
    effectiveness_tracker: OptimizationEffectivenessTracker,
}

/// Optimization strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStrategy {
    /// Strategy ID
    pub id: String,
    /// Strategy name
    pub name: String,
    /// Strategy description
    pub description: String,
    /// Target performance aspect
    pub target_aspect: PerformanceAspect,
    /// Expected improvement percentage
    pub expected_improvement: f32,
    /// Implementation complexity
    pub complexity: OptimizationComplexity,
    /// Prerequisites
    pub prerequisites: Vec<String>,
    /// Implementation steps
    pub implementation_steps: Vec<String>,
}

/// Performance aspect enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PerformanceAspect {
    Latency,
    Throughput,
    MemoryUsage,
    CpuUsage,
    ErrorRate,
    Compatibility,
    Overall,
}

/// Optimization complexity levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationComplexity {
    Low,
    Medium,
    High,
    Critical,
}

/// Applied optimization record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedOptimization {
    /// Optimization strategy ID
    pub strategy_id: String,
    /// Application timestamp
    pub applied_at: DateTime<Utc>,
    /// Expected improvement
    pub expected_improvement: f32,
    /// Measured improvement
    pub measured_improvement: Option<f32>,
    /// Status of the optimization
    pub status: OptimizationStatus,
    /// Rollback information
    pub rollback_info: Option<RollbackInfo>,
}

/// Optimization status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationStatus {
    Applied,
    Monitoring,
    Effective,
    Ineffective,
    RolledBack,
}

/// Rollback information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackInfo {
    /// Rollback timestamp
    pub rolled_back_at: DateTime<Utc>,
    /// Rollback reason
    pub reason: String,
    /// Performance before rollback
    pub performance_before: f32,
    /// Performance after rollback
    pub performance_after: f32,
}

/// Optimization effectiveness tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationEffectivenessTracker {
    /// Effectiveness scores by strategy
    pub strategy_effectiveness: HashMap<String, f32>,
    /// Overall optimization success rate
    pub overall_success_rate: f32,
    /// Most effective strategies
    pub most_effective_strategies: Vec<String>,
    /// Least effective strategies
    pub least_effective_strategies: Vec<String>,
}

impl Default for PerformanceStatistics {
    fn default() -> Self {
        Self {
            current_avg_latency_ms: 0.0,
            current_uptime_percentage: 100.0,
            current_error_rate: 0.0,
            current_compatibility_score: 100.0,
            performance_trends: PerformanceTrends::default(),
            alert_status: AlertStatus::default(),
            last_updated: Utc::now(),
            health_score: 1.0,
        }
    }
}

impl Default for PerformanceTrends {
    fn default() -> Self {
        Self {
            latency_trend: TrendDirection::Stable,
            uptime_trend: TrendDirection::Stable,
            error_rate_trend: TrendDirection::Stable,
            compatibility_trend: TrendDirection::Stable,
            overall_trend: TrendDirection::Stable,
            trend_confidence: 0.0,
        }
    }
}

impl Default for AlertStatus {
    fn default() -> Self {
        Self {
            active_alerts: Vec::new(),
            alert_count: 0,
            last_alert: None,
            overall_severity: AlertSeverity::Info,
        }
    }
}

impl Default for LatencyStatistics {
    fn default() -> Self {
        Self {
            avg_latency_ms: 0.0,
            median_latency_ms: 0.0,
            min_latency_ms: 0,
            max_latency_ms: 0,
            std_deviation: 0.0,
            measurement_count: 0,
            threshold_violations: 0,
        }
    }
}

impl EnhancedPerformanceMonitor {
    /// Create a new enhanced performance monitor
    pub async fn new(
        config: EnhancedPerformanceConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Self {
            stats: Arc::new(RwLock::new(PerformanceStatistics::default())),
            latency_tracker: Arc::new(RwLock::new(LatencyTracker {
                measurements: VecDeque::new(),
                stats: LatencyStatistics::default(),
                operation_stats: HashMap::new(),
                percentiles: LatencyPercentiles {
                    p50: 0,
                    p75: 0,
                    p90: 0,
                    p95: 0,
                    p99: 0,
                    p99_9: 0,
                },
            })),
            uptime_monitor: Arc::new(RwLock::new(UptimeMonitor {
                service_records: HashMap::new(),
                incidents: VecDeque::new(),
                overall_stats: UptimeStatistics {
                    overall_uptime_percentage: 100.0,
                    total_incidents: 0,
                    avg_incident_duration_minutes: 0.0,
                    mttr_minutes: 0.0,
                    mtbf_hours: 0.0,
                },
            })),
            error_tracker: Arc::new(RwLock::new(ErrorTracker {
                error_records: VecDeque::new(),
                error_stats: HashMap::new(),
                overall_stats: ErrorStatistics {
                    total_errors: 0,
                    current_error_rate: 0.0,
                    error_rate_by_severity: HashMap::new(),
                    top_error_categories: Vec::new(),
                    resolution_stats: ErrorResolutionStats {
                        avg_resolution_time_minutes: 0.0,
                        auto_resolution_rate: 0.0,
                        unresolved_rate: 0.0,
                        recurrence_rate: 0.0,
                    },
                },
                pattern_analyzer: ErrorPatternAnalyzer {
                    detected_patterns: Vec::new(),
                    pattern_confidence: HashMap::new(),
                    predicted_hotspots: Vec::new(),
                },
            })),
            compatibility_monitor: Arc::new(RwLock::new(CompatibilityMonitor {
                platform_records: HashMap::new(),
                test_results: HashMap::new(),
                overall_stats: CompatibilityStatistics {
                    avg_compatibility_score: 100.0,
                    platform_coverage: 0.0,
                    supported_platforms: 0,
                    known_issues_count: 0,
                    workaround_availability: 0.0,
                },
            })),
            optimizer: Arc::new(RwLock::new(PerformanceOptimizer {
                strategies: Vec::new(),
                applied_optimizations: HashMap::new(),
                effectiveness_tracker: OptimizationEffectivenessTracker {
                    strategy_effectiveness: HashMap::new(),
                    overall_success_rate: 0.0,
                    most_effective_strategies: Vec::new(),
                    least_effective_strategies: Vec::new(),
                },
            })),
            config,
        })
    }

    /// Record a latency measurement
    pub async fn record_latency(
        &self,
        measurement: LatencyMeasurement,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut tracker = self.latency_tracker.write().await;

        // Add to measurements
        tracker.measurements.push_back(measurement.clone());

        // Maintain max samples limit
        if tracker.measurements.len() > self.config.max_performance_samples {
            tracker.measurements.pop_front();
        }

        // Update statistics
        self.update_latency_statistics(&mut tracker).await;

        // Check for alerts
        if measurement.latency_ms > self.config.target_max_latency_ms {
            self.raise_latency_alert(measurement).await?;
        }

        Ok(())
    }

    /// Update latency statistics
    async fn update_latency_statistics(&self, tracker: &mut LatencyTracker) {
        if tracker.measurements.is_empty() {
            return;
        }

        let latencies: Vec<u32> = tracker.measurements.iter().map(|m| m.latency_ms).collect();

        let sum: u32 = latencies.iter().sum();
        let count = latencies.len() as u32;

        tracker.stats.avg_latency_ms = sum as f32 / count as f32;
        tracker.stats.min_latency_ms = *latencies.iter().min().unwrap_or(&0);
        tracker.stats.max_latency_ms = *latencies.iter().max().unwrap_or(&0);
        tracker.stats.measurement_count = count;

        // Calculate median
        let mut sorted_latencies = latencies.clone();
        sorted_latencies.sort_unstable();
        tracker.stats.median_latency_ms = if sorted_latencies.len() % 2 == 0 {
            let mid = sorted_latencies.len() / 2;
            (sorted_latencies[mid - 1] + sorted_latencies[mid]) as f32 / 2.0
        } else {
            sorted_latencies[sorted_latencies.len() / 2] as f32
        };

        // Calculate percentiles
        if !sorted_latencies.is_empty() {
            tracker.percentiles.p50 = percentile(&sorted_latencies, 50.0);
            tracker.percentiles.p75 = percentile(&sorted_latencies, 75.0);
            tracker.percentiles.p90 = percentile(&sorted_latencies, 90.0);
            tracker.percentiles.p95 = percentile(&sorted_latencies, 95.0);
            tracker.percentiles.p99 = percentile(&sorted_latencies, 99.0);
            tracker.percentiles.p99_9 = percentile(&sorted_latencies, 99.9);
        }

        // Count threshold violations
        tracker.stats.threshold_violations = latencies
            .iter()
            .filter(|&&l| l > self.config.target_max_latency_ms)
            .count() as u32;
    }

    /// Raise a latency alert
    async fn raise_latency_alert(
        &self,
        measurement: LatencyMeasurement,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let alert = PerformanceAlert {
            id: format!("latency-{}", Utc::now().timestamp()),
            alert_type: AlertType::LatencyThresholdExceeded,
            severity: if measurement.latency_ms > (self.config.target_max_latency_ms * 2) {
                AlertSeverity::Critical
            } else {
                AlertSeverity::Warning
            },
            message: format!(
                "Latency threshold exceeded: {}ms > {}ms for operation '{}'",
                measurement.latency_ms, self.config.target_max_latency_ms, measurement.operation
            ),
            timestamp: Utc::now(),
            trigger_metric: "latency".to_string(),
            current_value: measurement.latency_ms as f32,
            threshold_value: self.config.target_max_latency_ms as f32,
            suggested_actions: vec![
                "Check system resource usage".to_string(),
                "Review concurrent operation load".to_string(),
                "Consider enabling performance optimizations".to_string(),
            ],
        };

        let mut stats = self.stats.write().await;
        stats.alert_status.active_alerts.push(alert);
        stats.alert_status.alert_count += 1;
        stats.alert_status.last_alert = Some(Utc::now());

        Ok(())
    }

    /// Get current performance statistics
    pub async fn get_performance_stats(&self) -> PerformanceStatistics {
        let stats = self.stats.read().await;
        stats.clone()
    }

    /// Get detailed latency analysis
    pub async fn get_latency_analysis(&self) -> LatencyAnalysisReport {
        let tracker = self.latency_tracker.read().await;

        LatencyAnalysisReport {
            overall_stats: tracker.stats.clone(),
            percentiles: tracker.percentiles.clone(),
            operation_breakdown: tracker.operation_stats.clone(),
            recent_measurements: tracker
                .measurements
                .iter()
                .rev()
                .take(100)
                .cloned()
                .collect(),
            threshold_compliance: ThresholdCompliance {
                target_threshold_ms: self.config.target_max_latency_ms,
                compliance_rate: if tracker.stats.measurement_count > 0 {
                    ((tracker.stats.measurement_count - tracker.stats.threshold_violations) as f32
                        / tracker.stats.measurement_count as f32)
                        * 100.0
                } else {
                    100.0
                },
                violation_count: tracker.stats.threshold_violations,
                total_measurements: tracker.stats.measurement_count,
            },
        }
    }

    /// Start continuous performance monitoring
    pub async fn start_monitoring(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let config = self.config.clone();
        let stats = Arc::clone(&self.stats);
        let latency_tracker = Arc::clone(&self.latency_tracker);

        tokio::spawn(async move {
            let mut interval = interval(tokio::time::Duration::from_secs(
                config.monitoring_interval_seconds,
            ));

            loop {
                interval.tick().await;

                // Update overall performance statistics
                let mut stats_guard = stats.write().await;
                let latency_guard = latency_tracker.read().await;

                stats_guard.current_avg_latency_ms = latency_guard.stats.avg_latency_ms;
                stats_guard.last_updated = Utc::now();

                // Calculate health score
                let latency_score =
                    if stats_guard.current_avg_latency_ms <= config.target_max_latency_ms as f32 {
                        1.0
                    } else {
                        (config.target_max_latency_ms as f32 / stats_guard.current_avg_latency_ms)
                            .min(1.0)
                    };

                stats_guard.health_score = (latency_score
                    + stats_guard.current_uptime_percentage / 100.0
                    + (1.0 - stats_guard.current_error_rate)
                    + stats_guard.current_compatibility_score / 100.0)
                    / 4.0;

                drop(stats_guard);
                drop(latency_guard);
            }
        });

        Ok(())
    }
}

/// Helper function to calculate percentiles
fn percentile(sorted_values: &[u32], percentile: f32) -> u32 {
    if sorted_values.is_empty() {
        return 0;
    }

    let index = (percentile / 100.0 * (sorted_values.len() - 1) as f32).round() as usize;
    sorted_values[index.min(sorted_values.len() - 1)]
}

/// Latency analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyAnalysisReport {
    pub overall_stats: LatencyStatistics,
    pub percentiles: LatencyPercentiles,
    pub operation_breakdown: HashMap<String, LatencyStatistics>,
    pub recent_measurements: Vec<LatencyMeasurement>,
    pub threshold_compliance: ThresholdCompliance,
}

/// Threshold compliance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdCompliance {
    pub target_threshold_ms: u32,
    pub compliance_rate: f32,
    pub violation_count: u32,
    pub total_measurements: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_enhanced_performance_monitor_creation() {
        let config = EnhancedPerformanceConfig::default();
        let monitor = EnhancedPerformanceMonitor::new(config).await.unwrap();

        let stats = monitor.get_performance_stats().await;
        assert_eq!(stats.current_avg_latency_ms, 0.0);
        assert_eq!(stats.current_uptime_percentage, 100.0);
    }

    #[tokio::test]
    async fn test_latency_recording() {
        let config = EnhancedPerformanceConfig::default();
        let monitor = EnhancedPerformanceMonitor::new(config).await.unwrap();

        let measurement = LatencyMeasurement {
            timestamp: Utc::now(),
            operation: "test_operation".to_string(),
            latency_ms: 75,
            platform: "test".to_string(),
            user_id: "user1".to_string(),
            context: LatencyContext {
                cpu_usage: 50.0,
                memory_usage: 60.0,
                network_latency_ms: Some(10),
                concurrent_operations: 5,
            },
        };

        monitor.record_latency(measurement).await.unwrap();

        let analysis = monitor.get_latency_analysis().await;
        assert_eq!(analysis.overall_stats.measurement_count, 1);
        assert_eq!(analysis.overall_stats.avg_latency_ms, 75.0);
    }

    #[tokio::test]
    async fn test_latency_threshold_alert() {
        let mut config = EnhancedPerformanceConfig::default();
        config.target_max_latency_ms = 100;
        let monitor = EnhancedPerformanceMonitor::new(config).await.unwrap();

        let measurement = LatencyMeasurement {
            timestamp: Utc::now(),
            operation: "slow_operation".to_string(),
            latency_ms: 150, // Exceeds threshold
            platform: "test".to_string(),
            user_id: "user1".to_string(),
            context: LatencyContext {
                cpu_usage: 80.0,
                memory_usage: 90.0,
                network_latency_ms: Some(20),
                concurrent_operations: 10,
            },
        };

        monitor.record_latency(measurement).await.unwrap();

        let stats = monitor.get_performance_stats().await;
        assert_eq!(stats.alert_status.active_alerts.len(), 1);
        assert_eq!(
            stats.alert_status.active_alerts[0].alert_type,
            AlertType::LatencyThresholdExceeded
        );
    }

    #[tokio::test]
    async fn test_percentile_calculation() {
        let values = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

        // For 10 values [10,20,30,40,50,60,70,80,90,100] (indices 0-9):
        // 50th percentile: index = (0.5 * 9).round() = 4.5.round() = 5 -> values[5] = 60
        assert_eq!(percentile(&values, 50.0), 60);
        // 90th percentile: index = (0.9 * 9).round() = 8.1.round() = 8 -> values[8] = 90
        assert_eq!(percentile(&values, 90.0), 90);
        // 95th percentile: index = (0.95 * 9).round() = 8.55.round() = 9 -> values[9] = 100
        assert_eq!(percentile(&values, 95.0), 100);
    }

    #[tokio::test]
    async fn test_performance_health_score() {
        let config = EnhancedPerformanceConfig::default();
        let monitor = EnhancedPerformanceMonitor::new(config).await.unwrap();

        // Record some good performance measurements
        let measurement = LatencyMeasurement {
            timestamp: Utc::now(),
            operation: "fast_operation".to_string(),
            latency_ms: 50,
            platform: "test".to_string(),
            user_id: "user1".to_string(),
            context: LatencyContext {
                cpu_usage: 30.0,
                memory_usage: 40.0,
                network_latency_ms: Some(5),
                concurrent_operations: 2,
            },
        };

        monitor.record_latency(measurement).await.unwrap();

        // Manually trigger stats update
        {
            let mut stats = monitor.stats.write().await;
            let latency_tracker = monitor.latency_tracker.read().await;
            stats.current_avg_latency_ms = latency_tracker.stats.avg_latency_ms;
        }

        let stats = monitor.get_performance_stats().await;
        assert!(stats.health_score > 0.0);
    }

    #[tokio::test]
    async fn test_config_defaults() {
        let config = EnhancedPerformanceConfig::default();

        assert_eq!(config.target_max_latency_ms, 100);
        assert_eq!(config.target_min_uptime, 0.999);
        assert_eq!(config.target_max_error_rate, 0.02);
        assert_eq!(config.target_min_compatibility, 0.95);
        assert!(config.enable_auto_optimization);
        assert!(config.enable_predictive_analytics);
    }
}
