// src/bi_dashboard.rs
//! Business Intelligence Dashboard for `VoiRS` Feedback System
//!
//! This module provides a unified business intelligence dashboard that consolidates
//! analytics, metrics, monitoring, and health data into actionable insights. It enables
//! data-driven decision making and comprehensive system oversight.
//!
//! # Features
//! - Unified dashboard with KPIs (Key Performance Indicators)
//! - Real-time and historical metrics
//! - Customizable widgets and layouts
//! - Alert integration and health monitoring
//! - Exportable reports (JSON, CSV, PDF)
//! - Trend analysis and forecasting
//!
//! # Examples
//! ```
//! use voirs_feedback::bi_dashboard::{BiDashboard, DashboardConfig, WidgetType};
//! use std::time::Duration;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create dashboard
//! let mut dashboard = BiDashboard::new(DashboardConfig::default());
//!
//! // Add widgets
//! dashboard.add_widget("api_usage", WidgetType::LineChart {
//!     title: "API Usage".to_string(),
//!     data_source: "usage_analytics".to_string(),
//! });
//!
//! // Get dashboard snapshot
//! let snapshot = dashboard.get_snapshot().await?;
//! println!("Total users: {}", snapshot.kpis.total_users);
//! println!("System health: {:?}", snapshot.health_status);
//! # Ok(())
//! # }
//! ```

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

/// Dashboard errors
#[derive(Debug, Error)]
#[allow(missing_docs)]
pub enum DashboardError {
    /// Widget not found
    #[error("Widget not found: {0}")]
    WidgetNotFound(String),

    /// Invalid configuration
    #[error("Invalid dashboard configuration: {0}")]
    InvalidConfig(String),

    /// Data source error
    #[error("Data source error: {0}")]
    DataSourceError(String),

    /// Export error
    #[error("Export error: {0}")]
    ExportError(String),
}

/// Result type for dashboard operations
pub type DashboardResult<T> = Result<T, DashboardError>;

/// Widget types for dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum WidgetType {
    /// Key Performance Indicator (single number)
    KPI {
        title: String,
        metric: String,
        format: String,
    },
    /// Line chart for time series
    LineChart { title: String, data_source: String },
    /// Bar chart for comparisons
    BarChart { title: String, data_source: String },
    /// Pie chart for distributions
    PieChart { title: String, data_source: String },
    /// Table for detailed data
    Table {
        title: String,
        columns: Vec<String>,
        data_source: String,
    },
    /// Gauge for progress/status
    Gauge {
        title: String,
        metric: String,
        min: f64,
        max: f64,
    },
    /// Heatmap for correlation
    Heatmap { title: String, data_source: String },
    /// Alert list
    AlertList { title: String, max_alerts: usize },
}

/// Widget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Widget {
    /// Widget ID
    pub id: String,
    /// Widget type and configuration
    pub widget_type: WidgetType,
    /// Position (row, column)
    pub position: (u32, u32),
    /// Size (width, height)
    pub size: (u32, u32),
    /// Refresh interval
    pub refresh_interval: std::time::Duration,
    /// Last updated
    pub last_updated: DateTime<Utc>,
}

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Dashboard name
    pub name: String,
    /// Auto-refresh enabled
    pub auto_refresh: bool,
    /// Default refresh interval
    pub default_refresh_interval: std::time::Duration,
    /// Grid columns
    pub grid_columns: u32,
    /// Grid rows
    pub grid_rows: u32,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            name: "Main Dashboard".to_string(),
            auto_refresh: true,
            default_refresh_interval: std::time::Duration::from_secs(30),
            grid_columns: 12,
            grid_rows: 8,
        }
    }
}

/// Key Performance Indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KPIs {
    /// Total active users
    pub total_users: u64,
    /// Active sessions
    pub active_sessions: u64,
    /// API calls per minute
    pub api_calls_per_min: f64,
    /// Average response time (ms)
    pub avg_response_time_ms: f64,
    /// Error rate (percentage)
    pub error_rate: f64,
    /// System uptime (percentage)
    pub uptime: f64,
    /// User satisfaction score
    pub satisfaction_score: f64,
}

impl Default for KPIs {
    fn default() -> Self {
        Self {
            total_users: 0,
            active_sessions: 0,
            api_calls_per_min: 0.0,
            avg_response_time_ms: 0.0,
            error_rate: 0.0,
            uptime: 100.0,
            satisfaction_score: 0.0,
        }
    }
}

/// Health status for dashboard
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum HealthStatus {
    /// All systems operational
    Healthy,
    /// Minor issues detected
    Warning,
    /// Significant issues
    Degraded,
    /// Critical issues
    Critical,
}

/// Dashboard snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSnapshot {
    /// Timestamp of snapshot
    pub timestamp: DateTime<Utc>,
    /// Key performance indicators
    pub kpis: KPIs,
    /// Overall health status
    pub health_status: HealthStatus,
    /// Active alerts count
    pub active_alerts: u32,
    /// System metrics
    pub system_metrics: HashMap<String, f64>,
    /// Recent events (last 100)
    pub recent_events: Vec<DashboardEvent>,
}

/// Dashboard event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event type
    pub event_type: String,
    /// Event severity
    pub severity: String,
    /// Event message
    pub message: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Trend analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Metric name
    pub metric: String,
    /// Trend direction (positive = increasing, negative = decreasing)
    pub direction: f64,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,
    /// Predicted value for next period
    pub forecast: f64,
    /// Analysis period
    pub period: String,
}

/// Dashboard report format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum ReportFormat {
    /// JSON format
    JSON,
    /// CSV format
    CSV,
    /// Markdown format
    Markdown,
}

/// Business Intelligence Dashboard
pub struct BiDashboard {
    config: DashboardConfig,
    widgets: Arc<RwLock<HashMap<String, Widget>>>,
    kpis: Arc<RwLock<KPIs>>,
    events: Arc<RwLock<Vec<DashboardEvent>>>,
    metrics_history: Arc<RwLock<HashMap<String, Vec<(DateTime<Utc>, f64)>>>>,
}

impl BiDashboard {
    /// Create a new BI dashboard
    #[must_use]
    pub fn new(config: DashboardConfig) -> Self {
        Self {
            config,
            widgets: Arc::new(RwLock::new(HashMap::new())),
            kpis: Arc::new(RwLock::new(KPIs::default())),
            events: Arc::new(RwLock::new(Vec::new())),
            metrics_history: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add a widget to the dashboard
    pub async fn add_widget(&mut self, id: &str, widget_type: WidgetType) -> DashboardResult<()> {
        let widget = Widget {
            id: id.to_string(),
            widget_type,
            position: (0, 0), // Auto-layout
            size: (4, 2),     // Default size
            refresh_interval: self.config.default_refresh_interval,
            last_updated: Utc::now(),
        };

        let mut widgets = self.widgets.write().await;
        widgets.insert(id.to_string(), widget);

        Ok(())
    }

    /// Remove a widget from the dashboard
    pub async fn remove_widget(&mut self, id: &str) -> DashboardResult<()> {
        let mut widgets = self.widgets.write().await;
        widgets
            .remove(id)
            .ok_or_else(|| DashboardError::WidgetNotFound(id.to_string()))?;

        Ok(())
    }

    /// Update KPIs
    pub async fn update_kpis(&self, kpis: KPIs) {
        let mut current_kpis = self.kpis.write().await;
        *current_kpis = kpis;
    }

    /// Record a dashboard event
    pub async fn record_event(
        &self,
        event_type: String,
        severity: String,
        message: String,
        metadata: HashMap<String, String>,
    ) {
        let event = DashboardEvent {
            timestamp: Utc::now(),
            event_type,
            severity,
            message,
            metadata,
        };

        let mut events = self.events.write().await;
        events.push(event);

        // Keep only last 1000 events
        if events.len() > 1000 {
            let remove_count = events.len() - 1000;
            events.drain(0..remove_count);
        }
    }

    /// Record metric value
    pub async fn record_metric(&self, metric: &str, value: f64) {
        let mut history = self.metrics_history.write().await;
        let metric_data = history.entry(metric.to_string()).or_insert_with(Vec::new);

        metric_data.push((Utc::now(), value));

        // Keep only last 10000 data points per metric
        if metric_data.len() > 10000 {
            metric_data.drain(0..metric_data.len() - 10000);
        }
    }

    /// Get current dashboard snapshot
    pub async fn get_snapshot(&self) -> DashboardResult<DashboardSnapshot> {
        let kpis = self.kpis.read().await.clone();
        let events = self.events.read().await;

        // Calculate health status based on KPIs
        let health_status = self.calculate_health_status(&kpis);

        // Count active alerts (events with severity >= warning in last hour)
        let one_hour_ago = Utc::now() - ChronoDuration::hours(1);
        let active_alerts = events
            .iter()
            .filter(|e| {
                e.timestamp > one_hour_ago
                    && (e.severity == "warning"
                        || e.severity == "critical"
                        || e.severity == "error")
            })
            .count() as u32;

        // Get system metrics
        let metrics_history = self.metrics_history.read().await;
        let mut system_metrics = HashMap::new();
        for (metric, values) in metrics_history.iter() {
            if let Some((_, latest_value)) = values.last() {
                system_metrics.insert(metric.clone(), *latest_value);
            }
        }

        // Get recent events (last 100)
        let recent_events: Vec<_> = events.iter().rev().take(100).cloned().collect();

        Ok(DashboardSnapshot {
            timestamp: Utc::now(),
            kpis,
            health_status,
            active_alerts,
            system_metrics,
            recent_events,
        })
    }

    /// Analyze trends for a metric
    pub async fn analyze_trend(
        &self,
        metric: &str,
        period: std::time::Duration,
    ) -> DashboardResult<TrendAnalysis> {
        let history = self.metrics_history.read().await;

        let metric_data = history.get(metric).ok_or_else(|| {
            DashboardError::DataSourceError(format!("Metric not found: {metric}"))
        })?;

        if metric_data.len() < 2 {
            return Err(DashboardError::DataSourceError(
                "Insufficient data for trend analysis".to_string(),
            ));
        }

        // Filter data to period
        let period_start =
            Utc::now() - ChronoDuration::from_std(period).unwrap_or(ChronoDuration::hours(24));
        let period_data: Vec<_> = metric_data
            .iter()
            .filter(|(ts, _)| *ts >= period_start)
            .collect();

        if period_data.is_empty() {
            return Err(DashboardError::DataSourceError(
                "No data in specified period".to_string(),
            ));
        }

        // Simple linear regression
        let n = period_data.len() as f64;
        let x_values: Vec<f64> = (0..period_data.len()).map(|i| i as f64).collect();
        let y_values: Vec<f64> = period_data.iter().map(|(_, v)| *v).collect();

        let sum_x: f64 = x_values.iter().sum();
        let sum_y: f64 = y_values.iter().sum();
        let sum_xy: f64 = x_values.iter().zip(&y_values).map(|(x, y)| x * y).sum();
        let sum_x2: f64 = x_values.iter().map(|x| x * x).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // Calculate R-squared for confidence
        let mean_y = sum_y / n;
        let ss_tot: f64 = y_values.iter().map(|y| (y - mean_y).powi(2)).sum();
        let ss_res: f64 = y_values
            .iter()
            .enumerate()
            .map(|(i, y)| {
                let predicted = slope * (i as f64) + intercept;
                (y - predicted).powi(2)
            })
            .sum();

        let r_squared = if ss_tot > 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        // Forecast next value
        let forecast = slope * n + intercept;

        Ok(TrendAnalysis {
            metric: metric.to_string(),
            direction: slope,
            confidence: r_squared,
            forecast,
            period: format!("{} hours", period.as_secs() / 3600),
        })
    }

    /// Export dashboard data
    pub async fn export_report(&self, format: ReportFormat) -> DashboardResult<String> {
        let snapshot = self.get_snapshot().await?;

        match format {
            ReportFormat::JSON => serde_json::to_string_pretty(&snapshot)
                .map_err(|e| DashboardError::ExportError(e.to_string())),
            ReportFormat::CSV => {
                let mut csv = String::from("Metric,Value\n");
                csv.push_str(&format!("Total Users,{}\n", snapshot.kpis.total_users));
                csv.push_str(&format!(
                    "Active Sessions,{}\n",
                    snapshot.kpis.active_sessions
                ));
                csv.push_str(&format!(
                    "API Calls/Min,{:.2}\n",
                    snapshot.kpis.api_calls_per_min
                ));
                csv.push_str(&format!(
                    "Avg Response Time,{:.2}ms\n",
                    snapshot.kpis.avg_response_time_ms
                ));
                csv.push_str(&format!("Error Rate,{:.2}%\n", snapshot.kpis.error_rate));
                csv.push_str(&format!("Uptime,{:.2}%\n", snapshot.kpis.uptime));
                csv.push_str(&format!(
                    "Satisfaction,{:.2}\n",
                    snapshot.kpis.satisfaction_score
                ));
                Ok(csv)
            }
            ReportFormat::Markdown => {
                let mut md = "# Dashboard Report\n\n".to_string();
                md.push_str(&format!(
                    "**Generated**: {}\n\n",
                    snapshot.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
                ));
                md.push_str(&format!(
                    "**Health Status**: {:?}\n\n",
                    snapshot.health_status
                ));
                md.push_str("## Key Performance Indicators\n\n");
                md.push_str("| Metric | Value |\n");
                md.push_str("|--------|-------|\n");
                md.push_str(&format!(
                    "| Total Users | {} |\n",
                    snapshot.kpis.total_users
                ));
                md.push_str(&format!(
                    "| Active Sessions | {} |\n",
                    snapshot.kpis.active_sessions
                ));
                md.push_str(&format!(
                    "| API Calls/Min | {:.2} |\n",
                    snapshot.kpis.api_calls_per_min
                ));
                md.push_str(&format!(
                    "| Avg Response Time | {:.2}ms |\n",
                    snapshot.kpis.avg_response_time_ms
                ));
                md.push_str(&format!(
                    "| Error Rate | {:.2}% |\n",
                    snapshot.kpis.error_rate
                ));
                md.push_str(&format!("| Uptime | {:.2}% |\n", snapshot.kpis.uptime));
                md.push_str(&format!(
                    "| Satisfaction | {:.2} |\n",
                    snapshot.kpis.satisfaction_score
                ));
                md.push_str(&format!(
                    "\n**Active Alerts**: {}\n",
                    snapshot.active_alerts
                ));
                Ok(md)
            }
        }
    }

    /// Get widgets list
    pub async fn list_widgets(&self) -> Vec<String> {
        let widgets = self.widgets.read().await;
        widgets.keys().cloned().collect()
    }

    // Private helper methods

    fn calculate_health_status(&self, kpis: &KPIs) -> HealthStatus {
        // Check critical thresholds
        if kpis.uptime < 95.0 || kpis.error_rate > 10.0 || kpis.avg_response_time_ms > 1000.0 {
            return HealthStatus::Critical;
        }

        // Check degraded thresholds
        if kpis.uptime < 98.0 || kpis.error_rate > 5.0 || kpis.avg_response_time_ms > 500.0 {
            return HealthStatus::Degraded;
        }

        // Check warning thresholds
        if kpis.uptime < 99.5 || kpis.error_rate > 2.0 || kpis.avg_response_time_ms > 200.0 {
            return HealthStatus::Warning;
        }

        HealthStatus::Healthy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dashboard_creation() {
        let config = DashboardConfig::default();
        let dashboard = BiDashboard::new(config);

        let widgets = dashboard.list_widgets().await;
        assert_eq!(widgets.len(), 0);
    }

    #[tokio::test]
    async fn test_add_remove_widget() {
        let mut dashboard = BiDashboard::new(DashboardConfig::default());

        dashboard
            .add_widget(
                "test_widget",
                WidgetType::KPI {
                    title: "Test KPI".to_string(),
                    metric: "test_metric".to_string(),
                    format: "number".to_string(),
                },
            )
            .await
            .unwrap();

        let widgets = dashboard.list_widgets().await;
        assert_eq!(widgets.len(), 1);

        dashboard.remove_widget("test_widget").await.unwrap();

        let widgets = dashboard.list_widgets().await;
        assert_eq!(widgets.len(), 0);
    }

    #[tokio::test]
    async fn test_kpi_update() {
        let dashboard = BiDashboard::new(DashboardConfig::default());

        let kpis = KPIs {
            total_users: 1000,
            active_sessions: 50,
            api_calls_per_min: 100.0,
            avg_response_time_ms: 50.0,
            error_rate: 0.5,
            uptime: 99.9,
            satisfaction_score: 4.5,
        };

        dashboard.update_kpis(kpis.clone()).await;

        let snapshot = dashboard.get_snapshot().await.unwrap();
        assert_eq!(snapshot.kpis.total_users, 1000);
        assert_eq!(snapshot.kpis.active_sessions, 50);
    }

    #[tokio::test]
    async fn test_event_recording() {
        let dashboard = BiDashboard::new(DashboardConfig::default());

        dashboard
            .record_event(
                "test_event".to_string(),
                "info".to_string(),
                "Test message".to_string(),
                HashMap::new(),
            )
            .await;

        let snapshot = dashboard.get_snapshot().await.unwrap();
        assert_eq!(snapshot.recent_events.len(), 1);
    }

    #[tokio::test]
    async fn test_metric_recording() {
        let dashboard = BiDashboard::new(DashboardConfig::default());

        dashboard.record_metric("cpu_usage", 45.5).await;
        dashboard.record_metric("cpu_usage", 50.0).await;

        let snapshot = dashboard.get_snapshot().await.unwrap();
        assert_eq!(snapshot.system_metrics.get("cpu_usage"), Some(&50.0));
    }

    #[tokio::test]
    async fn test_health_status_calculation() {
        let dashboard = BiDashboard::new(DashboardConfig::default());

        // Healthy status
        let healthy_kpis = KPIs {
            uptime: 99.9,
            error_rate: 0.1,
            avg_response_time_ms: 100.0,
            ..Default::default()
        };
        dashboard.update_kpis(healthy_kpis).await;
        let snapshot = dashboard.get_snapshot().await.unwrap();
        assert_eq!(snapshot.health_status, HealthStatus::Healthy);

        // Critical status
        let critical_kpis = KPIs {
            uptime: 90.0,
            error_rate: 15.0,
            avg_response_time_ms: 2000.0,
            ..Default::default()
        };
        dashboard.update_kpis(critical_kpis).await;
        let snapshot = dashboard.get_snapshot().await.unwrap();
        assert_eq!(snapshot.health_status, HealthStatus::Critical);
    }

    #[tokio::test]
    async fn test_export_json() {
        let dashboard = BiDashboard::new(DashboardConfig::default());

        let export = dashboard.export_report(ReportFormat::JSON).await.unwrap();
        assert!(export.contains("\"total_users\""));
    }

    #[tokio::test]
    async fn test_export_csv() {
        let dashboard = BiDashboard::new(DashboardConfig::default());

        let export = dashboard.export_report(ReportFormat::CSV).await.unwrap();
        assert!(export.contains("Metric,Value"));
        assert!(export.contains("Total Users,"));
    }

    #[tokio::test]
    async fn test_export_markdown() {
        let dashboard = BiDashboard::new(DashboardConfig::default());

        let export = dashboard
            .export_report(ReportFormat::Markdown)
            .await
            .unwrap();
        assert!(export.contains("# Dashboard Report"));
        assert!(export.contains("Total Users"));
    }

    #[tokio::test]
    async fn test_trend_analysis() {
        let dashboard = BiDashboard::new(DashboardConfig::default());

        // Record some trending data
        for i in 0..10 {
            dashboard
                .record_metric("test_metric", i as f64 * 10.0)
                .await;
        }

        let trend = dashboard
            .analyze_trend("test_metric", std::time::Duration::from_secs(3600))
            .await
            .unwrap();

        assert!(trend.direction > 0.0); // Should be increasing
        assert!(trend.confidence > 0.9); // Should have high confidence for linear data
    }
}
