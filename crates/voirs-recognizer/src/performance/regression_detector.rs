//! Advanced Performance Regression Detection System
//!
//! This module provides enhanced regression detection capabilities with statistical analysis,
//! trend monitoring, and automated alerting for production environments.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Advanced performance regression detector with statistical analysis
pub struct AdvancedRegressionDetector {
    /// Configuration for detection thresholds and behavior
    config: DetectorConfig,
    /// Historical performance data storage
    data_store: PerformanceDataStore,
    /// Statistical analyzer for trend detection
    analyzer: StatisticalAnalyzer,
    /// Alert manager for notifications
    alert_manager: AlertManager,
}

/// Configuration for the regression detector
#[derive(Debug, Clone, Serialize, Deserialize)]
/// Detector Config
pub struct DetectorConfig {
    /// Statistical confidence level (e.g., 0.95 for 95% confidence)
    pub confidence_level: f64,
    /// Minimum number of data points for statistical analysis
    pub min_samples: usize,
    /// Rolling window size for trend analysis
    pub window_size: usize,
    /// Sensitivity for change detection (lower = more sensitive)
    pub sensitivity: f64,
    /// Enable automated alerts
    pub enable_alerts: bool,
    /// Alert channels configuration
    pub alert_channels: Vec<AlertChannel>,
    /// Baseline update strategy
    pub baseline_strategy: BaselineStrategy,
}

/// Strategy for updating performance baselines
#[derive(Debug, Clone, Serialize, Deserialize)]
/// Baseline Strategy
pub enum BaselineStrategy {
    /// Never update baseline automatically
    Manual,
    /// Update when improvement is detected
    OnImprovement,
    /// Update on rolling window basis
    RollingWindow {
        /// Number of days
        days: u32,
    },
    /// Update when statistical significance threshold is met
    Statistical {
        /// P-value threshold
        p_value: f64,
    },
}

/// Alert channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
/// Alert Channel
pub struct AlertChannel {
    /// Channel type (email, slack, webhook, etc.)
    pub channel_type: AlertChannelType,
    /// Channel-specific configuration
    pub config: HashMap<String, String>,
    /// Severity levels to trigger this channel
    pub severity_levels: Vec<AlertSeverity>,
}

/// Types of alert channels
#[derive(Debug, Clone, Serialize, Deserialize)]
/// Alert Channel Type
pub enum AlertChannelType {
    /// Email
    Email,
    /// Slack
    Slack,
    /// Webhook
    Webhook,
    /// File
    File,
    /// Console
    Console,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
/// Alert Severity
pub enum AlertSeverity {
    /// Info
    Info,
    /// Warning
    Warning,
    /// Critical
    Critical,
    /// Emergency
    Emergency,
}

/// Performance data storage and retrieval
pub struct PerformanceDataStore {
    /// Path to data storage file
    data_path: String,
    /// In-memory cache of recent data
    cache: VecDeque<PerformanceDataPoint>,
    /// Maximum cache size
    max_cache_size: usize,
}

/// Individual performance measurement data point
#[derive(Debug, Clone, Serialize, Deserialize)]
/// Performance Data Point
pub struct PerformanceDataPoint {
    /// Timestamp when measurement was taken
    pub timestamp: u64,
    /// Git commit hash (if available)
    pub commit_hash: Option<String>,
    /// Build/CI job identifier
    pub build_id: Option<String>,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Test configuration that produced these metrics
    pub test_config: TestConfig,
    /// Environment information
    pub environment: EnvironmentInfo,
}

/// Core performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
/// Performance Metrics
pub struct PerformanceMetrics {
    /// Real-time factor
    pub rtf: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Startup time in milliseconds
    pub startup_time_ms: u64,
    /// Processing latency in milliseconds
    pub latency_ms: u64,
    /// Throughput in samples per second
    pub throughput_sps: f64,
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// Model accuracy (if available)
    pub accuracy: Option<f64>,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Test configuration for reproducible measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
/// Test Config
pub struct TestConfig {
    /// Model type/size
    pub model: String,
    /// Audio sample rate
    pub sample_rate: u32,
    /// Test duration in seconds
    pub duration: f64,
    /// Number of audio channels
    pub channels: u32,
    /// Enabled features
    pub features: Vec<String>,
    /// Additional configuration parameters
    pub parameters: HashMap<String, String>,
}

/// Environment information for context
#[derive(Debug, Clone, Serialize, Deserialize)]
/// Environment Info
pub struct EnvironmentInfo {
    /// Operating system
    pub os: String,
    /// CPU model
    pub cpu: String,
    /// Available RAM in GB
    pub ram_gb: u32,
    /// Rust version
    pub rust_version: String,
    /// Compiler optimization level
    pub optimization: String,
    /// CI/CD environment flag
    pub is_ci: bool,
}

/// Statistical analyzer for performance trends
pub struct StatisticalAnalyzer {
    /// Configuration
    config: DetectorConfig,
}

/// Alert manager for notifications
pub struct AlertManager {
    /// Configured alert channels
    channels: Vec<AlertChannel>,
    /// Rate limiting to prevent spam
    rate_limiter: HashMap<String, SystemTime>,
}

/// Result of regression analysis
#[derive(Debug, Clone)]
/// Regression Analysis Result
pub struct RegressionAnalysisResult {
    /// Whether regression was detected
    pub has_regression: bool,
    /// Statistical confidence of the detection
    pub confidence: f64,
    /// Detected regressions by metric
    pub regressions: Vec<MetricRegression>,
    /// Detected improvements by metric  
    pub improvements: Vec<MetricImprovement>,
    /// Trend analysis results
    pub trends: TrendAnalysis,
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Individual metric regression
#[derive(Debug, Clone)]
/// Metric Regression
pub struct MetricRegression {
    /// Metric name
    pub metric: String,
    /// Current value
    pub current_value: f64,
    /// Baseline value (statistical mean)
    pub baseline_value: f64,
    /// Standard deviation from baseline
    pub std_deviations: f64,
    /// Percentage change
    pub percentage_change: f64,
    /// Statistical significance (p-value)
    pub p_value: f64,
    /// Severity assessment
    pub severity: AlertSeverity,
}

/// Individual metric improvement
#[derive(Debug, Clone)]
/// Metric Improvement
pub struct MetricImprovement {
    /// Metric name
    pub metric: String,
    /// Current value
    pub current_value: f64,
    /// Baseline value
    pub baseline_value: f64,
    /// Percentage improvement
    pub percentage_improvement: f64,
    /// Statistical significance
    pub p_value: f64,
}

/// Trend analysis for performance metrics over time
#[derive(Debug, Clone)]
/// Trend Analysis
pub struct TrendAnalysis {
    /// Overall trend direction
    pub trend_direction: TrendDirection,
    /// Strength of the trend (0.0 to 1.0)
    pub trend_strength: f64,
    /// Projected future performance
    pub projection: Option<PerformanceProjection>,
    /// Seasonal patterns detected
    pub patterns: Vec<SeasonalPattern>,
}

/// Direction of performance trend
#[derive(Debug, Clone, PartialEq)]
/// Trend Direction
pub enum TrendDirection {
    /// Improving
    Improving,
    /// Degrading
    Degrading,
    /// Stable
    Stable,
    /// Volatile
    Volatile,
}

/// Performance projection based on trend analysis
#[derive(Debug, Clone)]
/// Performance Projection
pub struct PerformanceProjection {
    /// Projected values for next measurement
    pub next_values: PerformanceMetrics,
    /// Confidence interval for projections
    pub confidence_interval: f64,
    /// Time horizon for projection
    pub time_horizon_days: u32,
}

/// Detected seasonal pattern in performance
#[derive(Debug, Clone)]
/// Seasonal Pattern
pub struct SeasonalPattern {
    /// Pattern type (daily, weekly, etc.)
    pub pattern_type: String,
    /// Strength of the pattern
    pub strength: f64,
    /// Description of the pattern
    pub description: String,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            min_samples: 10,
            window_size: 50,
            sensitivity: 0.05,
            enable_alerts: true,
            alert_channels: vec![AlertChannel {
                channel_type: AlertChannelType::Console,
                config: HashMap::new(),
                severity_levels: vec![
                    AlertSeverity::Warning,
                    AlertSeverity::Critical,
                    AlertSeverity::Emergency,
                ],
            }],
            baseline_strategy: BaselineStrategy::RollingWindow { days: 7 },
        }
    }
}

impl AdvancedRegressionDetector {
    /// Create new advanced regression detector
    pub fn new(config: DetectorConfig, data_path: String) -> Self {
        Self {
            analyzer: StatisticalAnalyzer::new(&config),
            alert_manager: AlertManager::new(config.alert_channels.clone()),
            data_store: PerformanceDataStore::new(data_path, config.window_size * 2),
            config,
        }
    }

    /// Analyze current performance against historical data
    pub async fn analyze_performance(
        &mut self,
        current_metrics: PerformanceMetrics,
        test_config: TestConfig,
        environment: EnvironmentInfo,
    ) -> Result<RegressionAnalysisResult, Box<dyn std::error::Error>> {
        // Create new data point
        let data_point = PerformanceDataPoint {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            commit_hash: get_git_commit_hash(),
            build_id: std::env::var("BUILD_ID").ok(),
            metrics: current_metrics,
            test_config,
            environment,
        };

        // Store the data point
        self.data_store.store_data_point(&data_point).await?;

        // Get historical data for analysis
        let historical_data = self
            .data_store
            .get_historical_data(&data_point.test_config)?;

        // Perform statistical analysis
        let analysis = self
            .analyzer
            .analyze_regression(&data_point, &historical_data)?;

        // Send alerts if regressions detected
        if analysis.has_regression && self.config.enable_alerts {
            self.alert_manager.send_regression_alert(&analysis).await?;
        }

        // Update baseline if needed
        self.update_baseline_if_needed(&data_point, &analysis)
            .await?;

        Ok(analysis)
    }

    /// Update baseline based on configured strategy
    async fn update_baseline_if_needed(
        &mut self,
        data_point: &PerformanceDataPoint,
        analysis: &RegressionAnalysisResult,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match &self.config.baseline_strategy {
            BaselineStrategy::Manual => {
                // No automatic updates
            }
            BaselineStrategy::OnImprovement => {
                if !analysis.improvements.is_empty() {
                    self.data_store.update_baseline(data_point).await?;
                }
            }
            BaselineStrategy::RollingWindow { days } => {
                let cutoff = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
                    - (*days as u64 * 24 * 3600);

                self.data_store.update_rolling_baseline(cutoff).await?;
            }
            BaselineStrategy::Statistical { p_value } => {
                // Update if improvements are statistically significant
                let significant_improvements = analysis
                    .improvements
                    .iter()
                    .any(|imp| imp.p_value < *p_value);

                if significant_improvements {
                    self.data_store.update_baseline(data_point).await?;
                }
            }
        }
        Ok(())
    }

    /// Generate comprehensive regression report
    pub fn generate_report(&self, analysis: &RegressionAnalysisResult) -> String {
        let mut report = String::new();

        // Header
        report.push_str("# Performance Regression Analysis Report\n\n");
        report.push_str(&format!(
            "**Timestamp:** {}\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));
        report.push_str(&format!(
            "**Confidence Level:** {:.1}%\n\n",
            self.config.confidence_level * 100.0
        ));

        // Overall status
        if analysis.has_regression {
            report.push_str("## 🚨 REGRESSION DETECTED\n\n");

            for regression in &analysis.regressions {
                let severity_emoji = match regression.severity {
                    AlertSeverity::Info => "ℹ️",
                    AlertSeverity::Warning => "⚠️",
                    AlertSeverity::Critical => "🚨",
                    AlertSeverity::Emergency => "🔥",
                };

                report.push_str(&format!(
                    "{} **{}**: {:.2}% regression ({:.3} σ from baseline)\n",
                    severity_emoji,
                    regression.metric,
                    regression.percentage_change,
                    regression.std_deviations
                ));

                report.push_str(&format!(
                    "  - Current: {:.3}, Baseline: {:.3} (p-value: {:.4})\n\n",
                    regression.current_value, regression.baseline_value, regression.p_value
                ));
            }
        } else {
            report.push_str("## ✅ NO REGRESSIONS DETECTED\n\n");
        }

        // Improvements
        if !analysis.improvements.is_empty() {
            report.push_str("## 🚀 Performance Improvements\n\n");
            for improvement in &analysis.improvements {
                report.push_str(&format!(
                    "- **{}**: {:.2}% improvement (p-value: {:.4})\n",
                    improvement.metric, improvement.percentage_improvement, improvement.p_value
                ));
            }
            report.push_str("\n");
        }

        // Trend analysis
        report.push_str("## 📈 Trend Analysis\n\n");
        match analysis.trends.trend_direction {
            TrendDirection::Improving => report.push_str("📈 **Overall trend: IMPROVING**\n"),
            TrendDirection::Degrading => report.push_str("📉 **Overall trend: DEGRADING**\n"),
            TrendDirection::Stable => report.push_str("➡️ **Overall trend: STABLE**\n"),
            TrendDirection::Volatile => report.push_str("📊 **Overall trend: VOLATILE**\n"),
        }

        report.push_str(&format!(
            "**Trend strength:** {:.2}\n\n",
            analysis.trends.trend_strength
        ));

        // Recommendations
        if !analysis.recommendations.is_empty() {
            report.push_str("## 💡 Recommendations\n\n");
            for (i, rec) in analysis.recommendations.iter().enumerate() {
                report.push_str(&format!("{}. {}\n", i + 1, rec));
            }
            report.push_str("\n");
        }

        // Statistical details
        report.push_str("## 📊 Statistical Analysis\n\n");
        report.push_str(&format!(
            "- **Confidence:** {:.2}%\n",
            analysis.confidence * 100.0
        ));
        report.push_str(&format!(
            "- **Sample size:** {} measurements\n",
            self.data_store.cache.len()
        ));
        report.push_str(&format!(
            "- **Detection sensitivity:** {:.3}\n",
            self.config.sensitivity
        ));

        report
    }
}

impl StatisticalAnalyzer {
    fn new(config: &DetectorConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    fn analyze_regression(
        &self,
        current: &PerformanceDataPoint,
        historical: &[PerformanceDataPoint],
    ) -> Result<RegressionAnalysisResult, Box<dyn std::error::Error>> {
        if historical.len() < self.config.min_samples {
            return Ok(RegressionAnalysisResult {
                has_regression: false,
                confidence: 0.0,
                regressions: Vec::new(),
                improvements: Vec::new(),
                trends: TrendAnalysis {
                    trend_direction: TrendDirection::Stable,
                    trend_strength: 0.0,
                    projection: None,
                    patterns: Vec::new(),
                },
                recommendations: vec![
                    "Insufficient historical data for statistical analysis".to_string(),
                    format!(
                        "Need at least {} samples, have {}",
                        self.config.min_samples,
                        historical.len()
                    ),
                ],
            });
        }

        let mut regressions = Vec::new();
        let mut improvements = Vec::new();

        // Analyze each metric
        self.analyze_metric_regression(
            "RTF",
            current.metrics.rtf,
            &historical.iter().map(|d| d.metrics.rtf).collect::<Vec<_>>(),
            &mut regressions,
            &mut improvements,
        );

        self.analyze_metric_regression(
            "Memory Usage",
            current.metrics.memory_usage as f64,
            &historical
                .iter()
                .map(|d| d.metrics.memory_usage as f64)
                .collect::<Vec<_>>(),
            &mut regressions,
            &mut improvements,
        );

        self.analyze_metric_regression(
            "Startup Time",
            current.metrics.startup_time_ms as f64,
            &historical
                .iter()
                .map(|d| d.metrics.startup_time_ms as f64)
                .collect::<Vec<_>>(),
            &mut regressions,
            &mut improvements,
        );

        self.analyze_metric_regression(
            "Throughput",
            current.metrics.throughput_sps,
            &historical
                .iter()
                .map(|d| d.metrics.throughput_sps)
                .collect::<Vec<_>>(),
            &mut regressions,
            &mut improvements,
        );

        // Analyze trends
        let trends = self.analyze_trends(historical);

        // Generate recommendations
        let recommendations = self.generate_recommendations(&regressions, &trends);

        Ok(RegressionAnalysisResult {
            has_regression: !regressions.is_empty(),
            confidence: self.config.confidence_level,
            regressions,
            improvements,
            trends,
            recommendations,
        })
    }

    fn analyze_metric_regression(
        &self,
        metric_name: &str,
        current_value: f64,
        historical_values: &[f64],
        regressions: &mut Vec<MetricRegression>,
        improvements: &mut Vec<MetricImprovement>,
    ) {
        let mean = historical_values.iter().sum::<f64>() / historical_values.len() as f64;
        let variance = historical_values
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / (historical_values.len() - 1) as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return; // No variance in historical data
        }

        let z_score = (current_value - mean) / std_dev;
        let p_value = self.calculate_p_value(z_score);
        let percentage_change = ((current_value - mean) / mean * 100.0).abs();

        // Check for regression (performance degradation)
        let is_regression = match metric_name {
            "RTF" | "Memory Usage" | "Startup Time" | "Latency" => {
                // Lower is better for these metrics
                z_score > 2.0 && p_value < self.config.sensitivity
            }
            "Throughput" | "Accuracy" => {
                // Higher is better for these metrics
                z_score < -2.0 && p_value < self.config.sensitivity
            }
            _ => false,
        };

        // Check for improvement
        let is_improvement = match metric_name {
            "RTF" | "Memory Usage" | "Startup Time" | "Latency" => {
                // Lower is better for these metrics
                z_score < -2.0 && p_value < self.config.sensitivity
            }
            "Throughput" | "Accuracy" => {
                // Higher is better for these metrics
                z_score > 2.0 && p_value < self.config.sensitivity
            }
            _ => false,
        };

        if is_regression {
            let severity = if z_score.abs() > 4.0 {
                AlertSeverity::Emergency
            } else if z_score.abs() > 3.0 {
                AlertSeverity::Critical
            } else {
                AlertSeverity::Warning
            };

            regressions.push(MetricRegression {
                metric: metric_name.to_string(),
                current_value,
                baseline_value: mean,
                std_deviations: z_score.abs(),
                percentage_change,
                p_value,
                severity,
            });
        } else if is_improvement {
            improvements.push(MetricImprovement {
                metric: metric_name.to_string(),
                current_value,
                baseline_value: mean,
                percentage_improvement: percentage_change,
                p_value,
            });
        }
    }

    fn calculate_p_value(&self, z_score: f64) -> f64 {
        // Simplified p-value calculation for two-tailed test
        // In production, use a proper statistical library
        let abs_z = z_score.abs();
        if abs_z > 3.0 {
            0.001
        } else if abs_z > 2.5 {
            0.01
        } else if abs_z > 2.0 {
            0.05
        } else if abs_z > 1.5 {
            0.1
        } else {
            0.2
        }
    }

    fn analyze_trends(&self, historical: &[PerformanceDataPoint]) -> TrendAnalysis {
        // Simplified trend analysis
        // In production, implement proper time series analysis

        if historical.len() < 5 {
            return TrendAnalysis {
                trend_direction: TrendDirection::Stable,
                trend_strength: 0.0,
                projection: None,
                patterns: Vec::new(),
            };
        }

        // Calculate simple linear trend for RTF
        let rtf_values: Vec<f64> = historical.iter().map(|d| d.metrics.rtf).collect();
        let trend_slope = self.calculate_linear_trend(&rtf_values);

        let trend_direction = if trend_slope > 0.001 {
            TrendDirection::Degrading
        } else if trend_slope < -0.001 {
            TrendDirection::Improving
        } else {
            TrendDirection::Stable
        };

        TrendAnalysis {
            trend_direction,
            trend_strength: trend_slope.abs(),
            projection: None,
            patterns: Vec::new(),
        }
    }

    fn calculate_linear_trend(&self, values: &[f64]) -> f64 {
        let n = values.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = values.iter().sum::<f64>() / n;

        let numerator: f64 = values
            .iter()
            .enumerate()
            .map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean))
            .sum();

        let denominator: f64 = (0..values.len()).map(|i| (i as f64 - x_mean).powi(2)).sum();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    fn generate_recommendations(
        &self,
        regressions: &[MetricRegression],
        trends: &TrendAnalysis,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if regressions.is_empty() {
            recommendations.push("Performance is within acceptable bounds".to_string());
        } else {
            recommendations
                .push("Performance regressions detected - investigate recent changes".to_string());

            for regression in regressions {
                match regression.metric.as_str() {
                    "RTF" => recommendations.push(
                        "Consider optimizing model inference or reducing computation complexity"
                            .to_string(),
                    ),
                    "Memory Usage" => recommendations
                        .push("Check for memory leaks or optimize data structures".to_string()),
                    "Startup Time" => recommendations.push(
                        "Review initialization code and model loading efficiency".to_string(),
                    ),
                    "Throughput" => recommendations
                        .push("Analyze processing pipeline for bottlenecks".to_string()),
                    _ => {}
                }
            }
        }

        match trends.trend_direction {
            TrendDirection::Degrading => {
                recommendations.push(
                    "Long-term performance trend is degrading - consider architectural review"
                        .to_string(),
                );
            }
            TrendDirection::Volatile => {
                recommendations.push(
                    "Performance is volatile - investigate environmental factors".to_string(),
                );
            }
            _ => {}
        }

        recommendations
    }
}

impl PerformanceDataStore {
    fn new(data_path: String, max_cache_size: usize) -> Self {
        Self {
            data_path,
            cache: VecDeque::new(),
            max_cache_size,
        }
    }

    async fn store_data_point(
        &mut self,
        data_point: &PerformanceDataPoint,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Add to cache
        self.cache.push_back(data_point.clone());
        if self.cache.len() > self.max_cache_size {
            self.cache.pop_front();
        }

        // Append to file
        self.append_to_file(data_point).await?;
        Ok(())
    }

    async fn append_to_file(
        &self,
        data_point: &PerformanceDataPoint,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Ensure directory exists
        if let Some(parent) = Path::new(&self.data_path).parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Load existing data
        let mut all_data = self.load_all_data().await.unwrap_or_default();
        all_data.push(data_point.clone());

        // Keep only last 1000 entries to prevent unbounded growth
        if all_data.len() > 1000 {
            all_data.drain(0..all_data.len() - 1000);
        }

        // Write back to file
        let content = serde_json::to_string_pretty(&all_data)?;
        tokio::fs::write(&self.data_path, content).await?;
        Ok(())
    }

    async fn load_all_data(&self) -> Result<Vec<PerformanceDataPoint>, Box<dyn std::error::Error>> {
        if !Path::new(&self.data_path).exists() {
            return Ok(Vec::new());
        }

        let content = tokio::fs::read_to_string(&self.data_path).await?;
        let data: Vec<PerformanceDataPoint> = serde_json::from_str(&content)?;
        Ok(data)
    }

    fn get_historical_data(
        &self,
        test_config: &TestConfig,
    ) -> Result<Vec<PerformanceDataPoint>, Box<dyn std::error::Error>> {
        // Filter cache for matching test configuration
        let matching_data: Vec<PerformanceDataPoint> = self
            .cache
            .iter()
            .filter(|d| self.configs_match(&d.test_config, test_config))
            .cloned()
            .collect();

        Ok(matching_data)
    }

    fn configs_match(&self, a: &TestConfig, b: &TestConfig) -> bool {
        a.model == b.model
            && a.sample_rate == b.sample_rate
            && a.channels == b.channels
            && (a.duration - b.duration).abs() < 0.1
    }

    async fn update_baseline(
        &mut self,
        _data_point: &PerformanceDataPoint,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Implementation for baseline updates
        // This would typically involve updating a separate baseline file
        Ok(())
    }

    async fn update_rolling_baseline(
        &mut self,
        _cutoff_timestamp: u64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Implementation for rolling baseline updates
        Ok(())
    }
}

impl AlertManager {
    fn new(channels: Vec<AlertChannel>) -> Self {
        Self {
            channels,
            rate_limiter: HashMap::new(),
        }
    }

    async fn send_regression_alert(
        &mut self,
        analysis: &RegressionAnalysisResult,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let max_severity = analysis
            .regressions
            .iter()
            .map(|r| &r.severity)
            .max()
            .unwrap_or(&AlertSeverity::Info);

        // Clone channels to avoid borrow conflicts
        let channels_to_alert: Vec<AlertChannel> = self
            .channels
            .iter()
            .filter(|channel| channel.severity_levels.contains(max_severity))
            .cloned()
            .collect();

        for channel in channels_to_alert {
            self.send_alert_to_channel(&channel, analysis).await?;
        }

        Ok(())
    }

    async fn send_alert_to_channel(
        &mut self,
        channel: &AlertChannel,
        analysis: &RegressionAnalysisResult,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Rate limiting check
        let now = SystemTime::now();
        let channel_key = format!("{:?}", channel.channel_type);

        if let Some(&last_alert) = self.rate_limiter.get(&channel_key) {
            if now.duration_since(last_alert)? < Duration::from_secs(300) {
                return Ok(()); // Skip if less than 5 minutes since last alert
            }
        }

        self.rate_limiter.insert(channel_key, now);

        match channel.channel_type {
            AlertChannelType::Console => {
                println!("🚨 PERFORMANCE REGRESSION ALERT 🚨");
                for regression in &analysis.regressions {
                    println!(
                        "• {}: {:.2}% regression",
                        regression.metric, regression.percentage_change
                    );
                }
            }
            AlertChannelType::File => {
                if let Some(file_path) = channel.config.get("path") {
                    let alert_message = format!(
                        "[{}] Performance regression detected: {} metrics affected\n",
                        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
                        analysis.regressions.len()
                    );
                    tokio::fs::write(file_path, alert_message).await?;
                }
            }
            AlertChannelType::Webhook => {
                // Implementation for webhook alerts
                if let Some(url) = channel.config.get("url") {
                    println!("Would send webhook alert to: {}", url);
                }
            }
            _ => {
                // Other channel types would be implemented here
            }
        }

        Ok(())
    }
}

/// Helper function to get git commit hash
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
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_regression_detection() {
        let temp_dir = tempdir().unwrap();
        let data_path = temp_dir
            .path()
            .join("test_data.json")
            .to_string_lossy()
            .to_string();

        let config = DetectorConfig::default();
        let mut detector = AdvancedRegressionDetector::new(config, data_path);

        let test_config = TestConfig {
            model: "test".to_string(),
            sample_rate: 16000,
            duration: 1.0,
            channels: 1,
            features: vec!["test".to_string()],
            parameters: HashMap::new(),
        };

        let environment = EnvironmentInfo {
            os: "test".to_string(),
            cpu: "test".to_string(),
            ram_gb: 8,
            rust_version: "1.78".to_string(),
            optimization: "release".to_string(),
            is_ci: true,
        };

        // Add some baseline data
        for i in 0..15 {
            let metrics = PerformanceMetrics {
                rtf: 0.2 + (i as f64 * 0.001),
                memory_usage: 500_000_000,
                startup_time_ms: 1500,
                latency_ms: 100,
                throughput_sps: 16000.0,
                cpu_utilization: 20.0,
                accuracy: None,
                custom_metrics: HashMap::new(),
            };

            detector
                .analyze_performance(metrics, test_config.clone(), environment.clone())
                .await
                .unwrap();
        }

        // Test with regressed performance
        let regressed_metrics = PerformanceMetrics {
            rtf: 0.35, // Significant regression
            memory_usage: 500_000_000,
            startup_time_ms: 1500,
            latency_ms: 100,
            throughput_sps: 16000.0,
            cpu_utilization: 20.0,
            accuracy: None,
            custom_metrics: HashMap::new(),
        };

        let result = detector
            .analyze_performance(regressed_metrics, test_config, environment)
            .await
            .unwrap();

        assert!(result.has_regression);
        assert!(!result.regressions.is_empty());

        let rtf_regression = result
            .regressions
            .iter()
            .find(|r| r.metric == "RTF")
            .expect("Should detect RTF regression");

        assert!(rtf_regression.percentage_change > 50.0);
    }

    #[test]
    fn test_statistical_analysis() {
        let config = DetectorConfig::default();
        let analyzer = StatisticalAnalyzer::new(&config);

        let historical_values = vec![0.2, 0.21, 0.19, 0.2, 0.22, 0.18, 0.2, 0.21, 0.19, 0.2];
        let mut regressions = Vec::new();
        let mut improvements = Vec::new();

        // Test normal value (should not trigger)
        analyzer.analyze_metric_regression(
            "RTF",
            0.21,
            &historical_values,
            &mut regressions,
            &mut improvements,
        );

        assert!(regressions.is_empty());

        // Test regressed value (should trigger)
        analyzer.analyze_metric_regression(
            "RTF",
            0.35,
            &historical_values,
            &mut regressions,
            &mut improvements,
        );

        assert!(!regressions.is_empty());
        assert_eq!(regressions[0].metric, "RTF");
    }
}
