//! # Integrated Performance Monitoring
//!
//! This module provides performance monitoring capabilities that integrate
//! with the VoiRS ecosystem for comprehensive performance tracking and
//! optimization.

use crate::RecognitionError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Integrated performance monitor for VoiRS ecosystem
#[derive(Debug)]
pub struct IntegratedPerformanceMonitor {
    /// Performance metrics
    metrics: Arc<RwLock<PerformanceMetrics>>,
    /// Component tracking
    components: Arc<RwLock<HashMap<String, ComponentPerformance>>>,
    /// Start time
    start_time: Instant,
}

/// Performance metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total operations performed
    pub total_operations: u64,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Average processing time
    pub average_processing_time: Duration,
    /// Peak memory usage in MB
    pub peak_memory_mb: f32,
    /// Current memory usage in MB
    pub current_memory_mb: f32,
    /// CPU utilization percentage
    pub cpu_utilization: f32,
    /// GPU utilization percentage
    pub gpu_utilization: Option<f32>,
    /// Throughput (operations per second)
    pub throughput: f32,
    /// Error rate
    pub error_rate: f32,
    /// Uptime
    pub uptime: Duration,
}

/// Component performance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentPerformance {
    /// Component name
    pub name: String,
    /// Operations count
    pub operations: u64,
    /// Total processing time
    pub processing_time: Duration,
    /// Memory usage in MB
    pub memory_mb: f32,
    /// Error count
    pub errors: u64,
    /// Last update time
    pub last_update: std::time::SystemTime,
}

/// Performance measurement
#[derive(Debug)]
pub struct PerformanceMeasurement {
    /// Component name
    pub component: String,
    /// Operation type
    pub operation: String,
    /// Start time
    start_time: Instant,
    /// Memory before operation
    memory_before: f32,
}

impl IntegratedPerformanceMonitor {
    /// Create new integrated performance monitor
    pub fn new() -> Result<Self, RecognitionError> {
        Ok(Self {
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            components: Arc::new(RwLock::new(HashMap::new())),
            start_time: Instant::now(),
        })
    }

    /// Start measuring performance for an operation
    pub async fn start_measurement(
        &self,
        component: &str,
        operation: &str,
    ) -> PerformanceMeasurement {
        let memory_before = self.get_current_memory_usage().await;

        PerformanceMeasurement {
            component: component.to_string(),
            operation: operation.to_string(),
            start_time: Instant::now(),
            memory_before,
        }
    }

    /// End measurement and record performance
    pub async fn end_measurement(
        &self,
        measurement: PerformanceMeasurement,
        success: bool,
    ) -> Result<(), RecognitionError> {
        let duration = measurement.start_time.elapsed();
        let memory_after = self.get_current_memory_usage().await;
        let _memory_delta = memory_after - measurement.memory_before;

        // Update overall metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_operations += 1;
            metrics.total_processing_time += duration;
            metrics.average_processing_time = Duration::from_nanos(
                metrics.total_processing_time.as_nanos() as u64 / metrics.total_operations,
            );

            if memory_after > metrics.peak_memory_mb {
                metrics.peak_memory_mb = memory_after;
            }
            metrics.current_memory_mb = memory_after;
            metrics.uptime = self.start_time.elapsed();

            // Update throughput (operations per second)
            metrics.throughput = metrics.total_operations as f32 / metrics.uptime.as_secs_f32();

            if !success {
                let error_count = metrics.total_operations as f32 * metrics.error_rate + 1.0;
                metrics.error_rate = error_count / metrics.total_operations as f32;
            }
        }

        // Update component metrics
        {
            let mut components = self.components.write().await;
            let component_perf = components
                .entry(measurement.component.clone())
                .or_insert_with(|| ComponentPerformance {
                    name: measurement.component.clone(),
                    operations: 0,
                    processing_time: Duration::default(),
                    memory_mb: 0.0,
                    errors: 0,
                    last_update: std::time::SystemTime::now(),
                });

            component_perf.operations += 1;
            component_perf.processing_time += duration;
            component_perf.memory_mb = memory_after;
            component_perf.last_update = std::time::SystemTime::now();

            if !success {
                component_perf.errors += 1;
            }
        }

        Ok(())
    }

    /// Get current performance metrics
    pub async fn get_metrics(&self) -> PerformanceMetrics {
        let mut metrics = self.metrics.read().await.clone();
        metrics.uptime = self.start_time.elapsed();
        metrics.cpu_utilization = self.get_cpu_utilization().await;
        metrics.gpu_utilization = self.get_gpu_utilization().await;
        metrics
    }

    /// Get component performance
    pub async fn get_component_performance(&self, component: &str) -> Option<ComponentPerformance> {
        self.components.read().await.get(component).cloned()
    }

    /// Get all component performances
    pub async fn get_all_component_performances(&self) -> HashMap<String, ComponentPerformance> {
        self.components.read().await.clone()
    }

    /// Reset performance metrics
    pub async fn reset_metrics(&self) {
        let mut metrics = self.metrics.write().await;
        *metrics = PerformanceMetrics::default();

        let mut components = self.components.write().await;
        components.clear();
    }

    /// Get performance summary
    pub async fn get_performance_summary(&self) -> PerformanceSummary {
        let metrics = self.get_metrics().await;
        let components = self.get_all_component_performances().await;

        PerformanceSummary {
            overall_metrics: metrics,
            component_count: components.len(),
            top_memory_consumers: self.get_top_memory_consumers(&components, 5),
            top_time_consumers: self.get_top_time_consumers(&components, 5),
            health_status: self.assess_health_status(&components).await,
        }
    }

    /// Get current memory usage in MB
    async fn get_current_memory_usage(&self) -> f32 {
        // Platform-specific memory usage detection
        #[cfg(target_os = "linux")]
        {
            if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
                for line in contents.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<f32>() {
                                return kb / 1024.0; // Convert KB to MB
                            }
                        }
                    }
                }
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            // Fallback for other platforms using ps command
            if let Ok(output) = std::process::Command::new("ps")
                .args(&["-o", "rss=", "-p"])
                .arg(std::process::id().to_string())
                .output()
            {
                if let Ok(output_str) = String::from_utf8(output.stdout) {
                    if let Ok(kb) = output_str.trim().parse::<f32>() {
                        return kb / 1024.0; // Convert KB to MB
                    }
                }
            }
        }

        // Fallback to zero if unable to determine
        0.0
    }

    /// Get CPU utilization
    async fn get_cpu_utilization(&self) -> f32 {
        // Simplified CPU utilization (would need more sophisticated implementation)
        // For now, return a placeholder value
        25.0
    }

    /// Get GPU utilization
    async fn get_gpu_utilization(&self) -> Option<f32> {
        // GPU utilization detection would be implemented here
        // This would require platform-specific code (NVIDIA-ML, etc.)
        None
    }

    /// Get top memory consumers
    fn get_top_memory_consumers(
        &self,
        components: &HashMap<String, ComponentPerformance>,
        count: usize,
    ) -> Vec<ComponentPerformance> {
        let mut sorted: Vec<_> = components.values().cloned().collect();
        sorted.sort_by(|a, b| {
            b.memory_mb
                .partial_cmp(&a.memory_mb)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(count).collect()
    }

    /// Get top time consumers
    fn get_top_time_consumers(
        &self,
        components: &HashMap<String, ComponentPerformance>,
        count: usize,
    ) -> Vec<ComponentPerformance> {
        let mut sorted: Vec<_> = components.values().cloned().collect();
        sorted.sort_by(|a, b| b.processing_time.cmp(&a.processing_time));
        sorted.into_iter().take(count).collect()
    }

    /// Assess health status
    async fn assess_health_status(
        &self,
        _components: &HashMap<String, ComponentPerformance>,
    ) -> HealthStatus {
        let metrics = self.metrics.read().await;

        // Simple health assessment based on error rate and resource usage
        if metrics.error_rate > 0.1 {
            HealthStatus::Critical
        } else if metrics.error_rate > 0.05 || metrics.current_memory_mb > 2048.0 {
            HealthStatus::Warning
        } else {
            HealthStatus::Healthy
        }
    }
}

/// Performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    /// Overall metrics
    pub overall_metrics: PerformanceMetrics,
    /// Number of components
    pub component_count: usize,
    /// Top memory consumers
    pub top_memory_consumers: Vec<ComponentPerformance>,
    /// Top time consumers
    pub top_time_consumers: Vec<ComponentPerformance>,
    /// Health status
    pub health_status: HealthStatus,
}

/// Health status enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// System is healthy
    Healthy,
    /// System has warnings
    Warning,
    /// System is in critical state
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_performance_monitor_creation() {
        let monitor = IntegratedPerformanceMonitor::new();
        assert!(monitor.is_ok());
    }

    #[tokio::test]
    async fn test_performance_measurement() {
        let monitor = IntegratedPerformanceMonitor::new().unwrap();

        let measurement = monitor
            .start_measurement("test_component", "test_operation")
            .await;
        sleep(Duration::from_millis(10)).await;

        let result = monitor.end_measurement(measurement, true).await;
        assert!(result.is_ok());

        let metrics = monitor.get_metrics().await;
        assert_eq!(metrics.total_operations, 1);
        assert!(metrics.average_processing_time.as_millis() >= 10);
    }

    #[tokio::test]
    async fn test_component_performance_tracking() {
        let monitor = IntegratedPerformanceMonitor::new().unwrap();

        let measurement = monitor
            .start_measurement("asr_component", "transcribe")
            .await;
        sleep(Duration::from_millis(5)).await;
        monitor.end_measurement(measurement, true).await.unwrap();

        let component_perf = monitor.get_component_performance("asr_component").await;
        assert!(component_perf.is_some());

        let perf = component_perf.unwrap();
        assert_eq!(perf.name, "asr_component");
        assert_eq!(perf.operations, 1);
        assert_eq!(perf.errors, 0);
    }

    #[tokio::test]
    async fn test_performance_summary() {
        let monitor = IntegratedPerformanceMonitor::new().unwrap();

        // Simulate some operations
        for i in 0..3 {
            let measurement = monitor
                .start_measurement(&format!("component_{}", i), "operation")
                .await;
            sleep(Duration::from_millis(5)).await;
            monitor.end_measurement(measurement, true).await.unwrap();
        }

        let summary = monitor.get_performance_summary().await;
        assert_eq!(summary.component_count, 3);
        assert_eq!(summary.overall_metrics.total_operations, 3);
        assert_eq!(summary.health_status, HealthStatus::Healthy);
    }

    #[tokio::test]
    async fn test_error_tracking() {
        let monitor = IntegratedPerformanceMonitor::new().unwrap();

        // Simulate successful operation
        let measurement1 = monitor.start_measurement("test", "op1").await;
        monitor.end_measurement(measurement1, true).await.unwrap();

        // Simulate failed operation
        let measurement2 = monitor.start_measurement("test", "op2").await;
        monitor.end_measurement(measurement2, false).await.unwrap();

        let metrics = monitor.get_metrics().await;
        assert_eq!(metrics.total_operations, 2);
        assert_eq!(metrics.error_rate, 0.5);

        let component_perf = monitor.get_component_performance("test").await.unwrap();
        assert_eq!(component_perf.operations, 2);
        assert_eq!(component_perf.errors, 1);
    }
}
