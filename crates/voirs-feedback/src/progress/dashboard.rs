//! Dashboard functionality for real-time progress monitoring
//!
//! This module provides comprehensive dashboard capabilities including:
//! - Real-time data aggregation and display
//! - Dashboard configuration and customization
//! - Metrics collection and visualization support
//! - Performance monitoring and analytics integration

use crate::traits::UserProgress;
use crate::FeedbackError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Dashboard configuration for customizing display and behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Dashboard title
    pub title: String,
    /// Metrics to display on the dashboard
    pub metrics: Vec<String>,
    /// Refresh interval in seconds
    pub refresh_interval: u32,
    /// Whether to enable real-time updates
    pub enable_realtime: bool,
    /// Maximum number of data points to display
    pub max_data_points: usize,
    /// Theme configuration
    pub theme: DashboardTheme,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            title: "Progress Dashboard".to_string(),
            metrics: vec![
                "average_skill_level".to_string(),
                "total_users".to_string(),
                "success_rate".to_string(),
                "improvement_velocity".to_string(),
            ],
            refresh_interval: 30,
            enable_realtime: true,
            max_data_points: 100,
            theme: DashboardTheme::default(),
        }
    }
}

/// Dashboard theme configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardTheme {
    /// Primary color scheme
    pub primary_color: String,
    /// Secondary color scheme
    pub secondary_color: String,
    /// Background color
    pub background_color: String,
    /// Text color
    pub text_color: String,
}

impl Default for DashboardTheme {
    fn default() -> Self {
        Self {
            primary_color: "#007bff".to_string(),
            secondary_color: "#6c757d".to_string(),
            background_color: "#ffffff".to_string(),
            text_color: "#212529".to_string(),
        }
    }
}

/// Real-time dashboard data containing current metrics and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeDashboardData {
    /// Dashboard configuration
    pub config: DashboardConfig,
    /// Current metric values mapped by metric name
    pub current_metrics: HashMap<String, f64>,
    /// Historical data points for trending
    pub historical_data: HashMap<String, Vec<MetricDataPoint>>,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
    /// Dashboard status information
    pub status: DashboardStatus,
}

/// Individual metric data point for historical trending
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDataPoint {
    /// Timestamp of the data point
    pub timestamp: DateTime<Utc>,
    /// Metric value at this point in time
    pub value: f64,
    /// Optional metadata for the data point
    pub metadata: Option<HashMap<String, String>>,
}

/// Dashboard status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardStatus {
    /// Whether the dashboard is currently active
    pub is_active: bool,
    /// Number of connected clients/viewers
    pub connected_clients: u32,
    /// Last error message if any
    pub last_error: Option<String>,
    /// Uptime in seconds
    pub uptime_seconds: u64,
}

impl Default for DashboardStatus {
    fn default() -> Self {
        Self {
            is_active: true,
            connected_clients: 0,
            last_error: None,
            uptime_seconds: 0,
        }
    }
}

/// Dashboard metrics generator for creating real-time data
pub struct DashboardMetricsGenerator {
    /// Configuration for the generator
    config: DashboardConfig,
    /// Cached historical data
    historical_cache: HashMap<String, Vec<MetricDataPoint>>,
}

impl DashboardMetricsGenerator {
    /// Create a new dashboard metrics generator
    pub fn new(config: DashboardConfig) -> Self {
        Self {
            config,
            historical_cache: HashMap::new(),
        }
    }

    /// Generate dashboard data from user progress information
    pub async fn generate_dashboard_data(
        &mut self,
        all_progress: &[UserProgress],
    ) -> Result<RealTimeDashboardData, FeedbackError> {
        let mut current_metrics = HashMap::new();
        let now = Utc::now();

        // Create a copy of metrics to avoid borrowing issues
        let metrics_to_calculate = self.config.metrics.clone();

        // Calculate metrics based on configuration
        for metric_name in &metrics_to_calculate {
            let value = self.calculate_metric(metric_name, all_progress)?;
            current_metrics.insert(metric_name.clone(), value);

            // Update historical data
            self.update_historical_data(metric_name.clone(), value, now);
        }

        // Trim historical data to max_data_points
        self.trim_historical_data();

        Ok(RealTimeDashboardData {
            config: self.config.clone(),
            current_metrics,
            historical_data: self.historical_cache.clone(),
            last_updated: now,
            status: DashboardStatus::default(),
        })
    }

    /// Calculate a specific metric from user progress data
    fn calculate_metric(
        &self,
        metric_name: &str,
        all_progress: &[UserProgress],
    ) -> Result<f64, FeedbackError> {
        match metric_name {
            "average_skill_level" => {
                if all_progress.is_empty() {
                    Ok(0.0)
                } else {
                    let total: f64 = all_progress
                        .iter()
                        .map(|p| f64::from(p.overall_skill_level))
                        .sum();
                    Ok(total / all_progress.len() as f64)
                }
            }
            "total_users" => Ok(all_progress.len() as f64),
            "success_rate" => {
                if all_progress.is_empty() {
                    Ok(0.0)
                } else {
                    let total_success_rate: f64 = all_progress
                        .iter()
                        .map(|p| f64::from(p.training_stats.success_rate))
                        .sum();
                    Ok(total_success_rate / all_progress.len() as f64)
                }
            }
            "improvement_velocity" => {
                if all_progress.is_empty() {
                    Ok(0.0)
                } else {
                    let total_improvement: f64 = all_progress
                        .iter()
                        .map(|p| f64::from(p.training_stats.average_improvement))
                        .sum();
                    Ok(total_improvement / all_progress.len() as f64)
                }
            }
            "total_sessions" => {
                let total_sessions: u64 = all_progress
                    .iter()
                    .map(|p| p.training_stats.total_sessions as u64)
                    .sum();
                Ok(total_sessions as f64)
            }
            "average_session_duration" => {
                if all_progress.is_empty() {
                    Ok(0.0)
                } else {
                    let total_duration: f64 = all_progress
                        .iter()
                        .map(|p| p.total_practice_time.as_secs() as f64)
                        .sum();
                    Ok(total_duration / all_progress.len() as f64)
                }
            }
            "current_streak_average" => {
                if all_progress.is_empty() {
                    Ok(0.0)
                } else {
                    let total_streak: u32 = all_progress
                        .iter()
                        .map(|p| p.training_stats.current_streak as u32)
                        .sum();
                    Ok(total_streak as f64 / all_progress.len() as f64)
                }
            }
            _ => Err(FeedbackError::ProgressTrackingError {
                message: format!("Unknown metric: {}", metric_name),
                source: None,
            }),
        }
    }

    /// Update historical data for a metric
    fn update_historical_data(
        &mut self,
        metric_name: String,
        value: f64,
        timestamp: DateTime<Utc>,
    ) {
        let data_points = self
            .historical_cache
            .entry(metric_name)
            .or_insert_with(Vec::new);

        data_points.push(MetricDataPoint {
            timestamp,
            value,
            metadata: None,
        });
    }

    /// Trim historical data to configured maximum
    fn trim_historical_data(&mut self) {
        for data_points in self.historical_cache.values_mut() {
            if data_points.len() > self.config.max_data_points {
                let excess = data_points.len() - self.config.max_data_points;
                data_points.drain(0..excess);
            }
        }
    }

    /// Get available metric names
    pub fn get_available_metrics() -> Vec<String> {
        vec![
            "average_skill_level".to_string(),
            "total_users".to_string(),
            "success_rate".to_string(),
            "improvement_velocity".to_string(),
            "total_sessions".to_string(),
            "average_session_duration".to_string(),
            "current_streak_average".to_string(),
        ]
    }

    /// Update dashboard configuration
    pub fn update_config(&mut self, new_config: DashboardConfig) {
        self.config = new_config;
    }

    /// Get current configuration
    pub fn get_config(&self) -> &DashboardConfig {
        &self.config
    }

    /// Clear historical data
    pub fn clear_historical_data(&mut self) {
        self.historical_cache.clear();
    }

    /// Get historical data for a specific metric
    pub fn get_historical_data(&self, metric_name: &str) -> Option<&Vec<MetricDataPoint>> {
        self.historical_cache.get(metric_name)
    }
}

/// Builder for creating dashboard configurations
pub struct DashboardConfigBuilder {
    config: DashboardConfig,
}

impl DashboardConfigBuilder {
    /// Create a new dashboard config builder
    pub fn new() -> Self {
        Self {
            config: DashboardConfig::default(),
        }
    }

    /// Set dashboard title
    pub fn title<S: Into<String>>(mut self, title: S) -> Self {
        self.config.title = title.into();
        self
    }

    /// Add metrics to display
    pub fn metrics(mut self, metrics: Vec<String>) -> Self {
        self.config.metrics = metrics;
        self
    }

    /// Add a single metric
    pub fn add_metric<S: Into<String>>(mut self, metric: S) -> Self {
        self.config.metrics.push(metric.into());
        self
    }

    /// Set refresh interval
    pub fn refresh_interval(mut self, seconds: u32) -> Self {
        self.config.refresh_interval = seconds;
        self
    }

    /// Enable or disable real-time updates
    pub fn realtime(mut self, enabled: bool) -> Self {
        self.config.enable_realtime = enabled;
        self
    }

    /// Set maximum data points
    pub fn max_data_points(mut self, max: usize) -> Self {
        self.config.max_data_points = max;
        self
    }

    /// Set theme
    pub fn theme(mut self, theme: DashboardTheme) -> Self {
        self.config.theme = theme;
        self
    }

    /// Build the final configuration
    pub fn build(self) -> DashboardConfig {
        self.config
    }
}

impl Default for DashboardConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{SessionScores, TrainingStatistics, UserProgress};
    use std::time::Duration;

    fn create_test_user_progress(skill_level: f32, success_rate: f32) -> UserProgress {
        UserProgress {
            user_id: "test_user".to_string(),
            overall_skill_level: skill_level,
            skill_breakdown: HashMap::new(),
            progress_history: Vec::new(),
            achievements: Vec::new(),
            training_stats: TrainingStatistics {
                total_sessions: 10,
                successful_sessions: 8,
                total_training_time: Duration::from_secs(300),
                exercises_completed: 20,
                success_rate,
                average_improvement: 0.1,
                current_streak: 5,
                longest_streak: 10,
            },
            goals: Vec::new(),
            last_updated: Utc::now(),
            average_scores: SessionScores::default(),
            skill_levels: HashMap::new(),
            recent_sessions: Vec::new(),
            personal_bests: HashMap::new(),
            session_count: 10,
            total_practice_time: Duration::from_secs(300),
        }
    }

    #[tokio::test]
    async fn test_dashboard_metrics_generator_creation() {
        let config = DashboardConfig::default();
        let generator = DashboardMetricsGenerator::new(config);

        assert_eq!(generator.config.title, "Progress Dashboard");
        assert!(!generator.config.metrics.is_empty());
    }

    #[tokio::test]
    async fn test_generate_dashboard_data() {
        let config = DashboardConfig::default();
        let mut generator = DashboardMetricsGenerator::new(config);

        let progress_data = vec![
            create_test_user_progress(0.8, 0.9),
            create_test_user_progress(0.7, 0.8),
        ];

        let dashboard_data = generator
            .generate_dashboard_data(&progress_data)
            .await
            .unwrap();

        assert_eq!(dashboard_data.current_metrics.len(), 4); // Default metrics
        assert!(dashboard_data
            .current_metrics
            .contains_key("average_skill_level"));
        assert!(dashboard_data.current_metrics.contains_key("total_users"));
        assert_eq!(dashboard_data.current_metrics["total_users"], 2.0);
    }

    #[tokio::test]
    async fn test_calculate_average_skill_level() {
        let config = DashboardConfig::default();
        let generator = DashboardMetricsGenerator::new(config);

        let progress_data = vec![
            create_test_user_progress(0.8, 0.9),
            create_test_user_progress(0.6, 0.7),
        ];

        let avg_skill = generator
            .calculate_metric("average_skill_level", &progress_data)
            .unwrap();
        assert!((avg_skill - 0.7).abs() < 0.001); // (0.8 + 0.6) / 2 = 0.7
    }

    #[tokio::test]
    async fn test_dashboard_config_builder() {
        let config = DashboardConfigBuilder::new()
            .title("Test Dashboard")
            .add_metric("test_metric")
            .refresh_interval(60)
            .realtime(false)
            .max_data_points(50)
            .build();

        assert_eq!(config.title, "Test Dashboard");
        assert!(config.metrics.contains(&"test_metric".to_string()));
        assert_eq!(config.refresh_interval, 60);
        assert!(!config.enable_realtime);
        assert_eq!(config.max_data_points, 50);
    }

    #[tokio::test]
    async fn test_historical_data_trimming() {
        let config = DashboardConfigBuilder::new().max_data_points(2).build();
        let mut generator = DashboardMetricsGenerator::new(config);

        let progress_data = vec![create_test_user_progress(0.8, 0.9)];

        // Generate data multiple times to exceed max_data_points
        for _ in 0..5 {
            generator
                .generate_dashboard_data(&progress_data)
                .await
                .unwrap();
        }

        // Check that historical data is trimmed
        for data_points in generator.historical_cache.values() {
            assert!(data_points.len() <= 2);
        }
    }

    #[test]
    fn test_get_available_metrics() {
        let metrics = DashboardMetricsGenerator::get_available_metrics();
        assert!(metrics.contains(&"average_skill_level".to_string()));
        assert!(metrics.contains(&"total_users".to_string()));
        assert!(metrics.len() >= 4);
    }
}
