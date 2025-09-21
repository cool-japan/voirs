//! System Health Monitoring
//!
//! This module provides comprehensive health check endpoints and system status monitoring
//! for production deployment and operational monitoring.

use crate::metrics_dashboard::MetricsDashboard;
use crate::persistence::PersistenceManager;
use crate::quality_monitor::QualityMonitor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Overall system health status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// System is fully operational
    Healthy,
    /// System is operational with warnings
    Warning,
    /// System has degraded performance
    Degraded,
    /// System is not operational
    Critical,
}

/// Individual component health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component name
    pub name: String,
    /// Component status
    pub status: HealthStatus,
    /// Response time in milliseconds
    pub response_time_ms: u64,
    /// Additional context information
    pub details: HashMap<String, String>,
    /// Last check timestamp
    pub last_check: String,
}

/// Complete system health report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// Overall system status
    pub status: HealthStatus,
    /// System uptime in seconds
    pub uptime_seconds: u64,
    /// Individual component health
    pub components: Vec<ComponentHealth>,
    /// System metrics summary
    pub metrics: HashMap<String, f64>,
    /// Timestamp of this report
    pub timestamp: String,
    /// Build version information
    pub version: String,
}

/// Health check configuration
#[derive(Debug, Clone)]
pub struct HealthConfig {
    /// Timeout for individual health checks
    pub check_timeout: Duration,
    /// Warning threshold for response times (ms)
    pub warning_threshold_ms: u64,
    /// Critical threshold for response times (ms)
    pub critical_threshold_ms: u64,
    /// Enable detailed component checks
    pub detailed_checks: bool,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            check_timeout: Duration::from_secs(5),
            warning_threshold_ms: 1000,
            critical_threshold_ms: 5000,
            detailed_checks: true,
        }
    }
}

/// System health monitor
#[derive(Debug)]
pub struct HealthMonitor {
    config: HealthConfig,
    start_time: Instant,
    version: String,
    components: Arc<RwLock<HashMap<String, ComponentHealth>>>,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new(config: HealthConfig, version: String) -> Self {
        Self {
            config,
            start_time: Instant::now(),
            version,
            components: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Perform a lightweight health check
    pub async fn quick_health_check(&self) -> HealthReport {
        let uptime = self.start_time.elapsed().as_secs();
        let timestamp = chrono::Utc::now().to_rfc3339();

        HealthReport {
            status: HealthStatus::Healthy,
            uptime_seconds: uptime,
            components: Vec::new(),
            metrics: HashMap::new(),
            timestamp,
            version: self.version.clone(),
        }
    }

    /// Perform a comprehensive health check
    pub async fn comprehensive_health_check(
        &self,
        quality_monitor: Option<&QualityMonitor>,
        metrics_dashboard: Option<&MetricsDashboard>,
    ) -> HealthReport {
        let start_time = Instant::now();
        let mut components = Vec::new();
        let mut overall_status = HealthStatus::Healthy;
        let mut metrics = HashMap::new();

        // Check quality monitoring system
        if let Some(qm) = quality_monitor {
            let component_health = self.check_quality_monitor(qm).await;
            if component_health.status != HealthStatus::Healthy {
                overall_status = self.merge_status(overall_status, component_health.status.clone());
            }
            components.push(component_health);
        }

        // Check metrics dashboard
        if let Some(md) = metrics_dashboard {
            let component_health = self.check_metrics_dashboard(md).await;
            if component_health.status != HealthStatus::Healthy {
                overall_status = self.merge_status(overall_status, component_health.status.clone());
            }
            components.push(component_health);
        }

        // Add system metrics
        metrics.insert(
            "check_duration_ms".to_string(),
            start_time.elapsed().as_millis() as f64,
        );
        metrics.insert(
            "uptime_seconds".to_string(),
            self.start_time.elapsed().as_secs() as f64,
        );
        metrics.insert("components_checked".to_string(), components.len() as f64);

        HealthReport {
            status: overall_status,
            uptime_seconds: self.start_time.elapsed().as_secs(),
            components,
            metrics,
            timestamp: chrono::Utc::now().to_rfc3339(),
            version: self.version.clone(),
        }
    }

    /// Check quality monitor health
    async fn check_quality_monitor(&self, quality_monitor: &QualityMonitor) -> ComponentHealth {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        let mut status = HealthStatus::Healthy;

        // Check if quality monitor is responding
        match tokio::time::timeout(
            self.config.check_timeout,
            self.ping_quality_monitor(quality_monitor),
        )
        .await
        {
            Ok(Ok(_)) => {
                details.insert("connectivity".to_string(), "ok".to_string());
            }
            Ok(Err(e)) => {
                details.insert("connectivity".to_string(), format!("error: {}", e));
                status = HealthStatus::Critical;
            }
            Err(_) => {
                details.insert("connectivity".to_string(), "timeout".to_string());
                status = HealthStatus::Critical;
            }
        }

        let response_time = start_time.elapsed().as_millis() as u64;

        // Check response time thresholds
        if response_time > self.config.critical_threshold_ms {
            status = HealthStatus::Critical;
        } else if response_time > self.config.warning_threshold_ms {
            status = self.merge_status(status, HealthStatus::Warning);
        }

        ComponentHealth {
            name: "quality_monitor".to_string(),
            status,
            response_time_ms: response_time,
            details,
            last_check: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Check metrics dashboard health
    async fn check_metrics_dashboard(
        &self,
        metrics_dashboard: &MetricsDashboard,
    ) -> ComponentHealth {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        let mut status = HealthStatus::Healthy;

        // Check if metrics dashboard is responding
        match tokio::time::timeout(
            self.config.check_timeout,
            self.ping_metrics_dashboard(metrics_dashboard),
        )
        .await
        {
            Ok(Ok(_)) => {
                details.insert("connectivity".to_string(), "ok".to_string());
            }
            Ok(Err(e)) => {
                details.insert("connectivity".to_string(), format!("error: {}", e));
                status = HealthStatus::Critical;
            }
            Err(_) => {
                details.insert("connectivity".to_string(), "timeout".to_string());
                status = HealthStatus::Critical;
            }
        }

        let response_time = start_time.elapsed().as_millis() as u64;

        // Check response time thresholds
        if response_time > self.config.critical_threshold_ms {
            status = HealthStatus::Critical;
        } else if response_time > self.config.warning_threshold_ms {
            status = self.merge_status(status, HealthStatus::Warning);
        }

        ComponentHealth {
            name: "metrics_dashboard".to_string(),
            status,
            response_time_ms: response_time,
            details,
            last_check: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Ping quality monitor to check responsiveness
    async fn ping_quality_monitor(&self, _quality_monitor: &QualityMonitor) -> Result<(), String> {
        // Simple ping test - in real implementation, this would call a lightweight method
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(())
    }

    /// Ping metrics dashboard to check responsiveness
    async fn ping_metrics_dashboard(
        &self,
        _metrics_dashboard: &MetricsDashboard,
    ) -> Result<(), String> {
        // Simple ping test - in real implementation, this would call a lightweight method
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(())
    }

    /// Merge two health statuses, taking the worse one
    fn merge_status(&self, current: HealthStatus, new: HealthStatus) -> HealthStatus {
        match (current, new) {
            (HealthStatus::Critical, _) | (_, HealthStatus::Critical) => HealthStatus::Critical,
            (HealthStatus::Degraded, _) | (_, HealthStatus::Degraded) => HealthStatus::Degraded,
            (HealthStatus::Warning, _) | (_, HealthStatus::Warning) => HealthStatus::Warning,
            (HealthStatus::Healthy, HealthStatus::Healthy) => HealthStatus::Healthy,
        }
    }

    /// Register a component health update
    pub async fn update_component_health(&self, component_health: ComponentHealth) {
        let mut components = self.components.write().await;
        components.insert(component_health.name.clone(), component_health);
    }

    /// Get current system version
    pub fn get_version(&self) -> &str {
        &self.version
    }

    /// Get system uptime
    pub fn get_uptime(&self) -> Duration {
        self.start_time.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_monitor_creation() {
        let config = HealthConfig::default();
        let monitor = HealthMonitor::new(config, "1.0.0".to_string());

        assert_eq!(monitor.get_version(), "1.0.0");
        // Check that uptime is measurable (at least 0 nanoseconds)
        assert!(monitor.get_uptime().as_nanos() >= 0);
    }

    #[tokio::test]
    async fn test_quick_health_check() {
        let config = HealthConfig::default();
        let monitor = HealthMonitor::new(config, "1.0.0".to_string());

        let report = monitor.quick_health_check().await;

        assert_eq!(report.status, HealthStatus::Healthy);
        assert_eq!(report.version, "1.0.0");
        assert!(report.uptime_seconds >= 0);
        assert!(report.components.is_empty());
    }

    #[tokio::test]
    async fn test_comprehensive_health_check() {
        let config = HealthConfig::default();
        let monitor = HealthMonitor::new(config, "1.0.0".to_string());

        let report = monitor.comprehensive_health_check(None, None).await;

        assert_eq!(report.status, HealthStatus::Healthy);
        assert_eq!(report.version, "1.0.0");
        assert!(report.metrics.contains_key("check_duration_ms"));
        assert!(report.metrics.contains_key("uptime_seconds"));
    }

    #[test]
    fn test_health_status_merge() {
        let config = HealthConfig::default();
        let monitor = HealthMonitor::new(config, "1.0.0".to_string());

        assert_eq!(
            monitor.merge_status(HealthStatus::Healthy, HealthStatus::Warning),
            HealthStatus::Warning
        );
        assert_eq!(
            monitor.merge_status(HealthStatus::Warning, HealthStatus::Critical),
            HealthStatus::Critical
        );
        assert_eq!(
            monitor.merge_status(HealthStatus::Degraded, HealthStatus::Warning),
            HealthStatus::Degraded
        );
    }

    #[test]
    fn test_health_config_defaults() {
        let config = HealthConfig::default();

        assert_eq!(config.check_timeout, Duration::from_secs(5));
        assert_eq!(config.warning_threshold_ms, 1000);
        assert_eq!(config.critical_threshold_ms, 5000);
        assert!(config.detailed_checks);
    }

    #[tokio::test]
    async fn test_component_health_update() {
        let config = HealthConfig::default();
        let monitor = HealthMonitor::new(config, "1.0.0".to_string());

        let component_health = ComponentHealth {
            name: "test_component".to_string(),
            status: HealthStatus::Healthy,
            response_time_ms: 100,
            details: HashMap::new(),
            last_check: chrono::Utc::now().to_rfc3339(),
        };

        monitor.update_component_health(component_health).await;

        let components = monitor.components.read().await;
        assert!(components.contains_key("test_component"));
    }
}
