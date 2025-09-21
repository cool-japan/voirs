//! # VoiRS Ecosystem Integration
//!
//! This module provides seamless integration between the VoiRS recognizer
//! and the broader VoiRS ecosystem, including unified configuration management,
//! pipeline integration, and performance monitoring.

pub mod config;
pub mod performance;
pub mod pipeline;
pub mod traits;

// Re-export specific items to avoid naming conflicts
pub use config::{
    ConfigPresets, CoordinationConfig, GlobalConfig, IntegrationConfig, MonitoringConfig,
    PerformanceConfig, RecognitionConfig, StreamingConfig, UnifiedConfigBuilder,
    UnifiedVoirsConfig,
};
pub use performance::{
    ComponentPerformance, IntegratedPerformanceMonitor,
    PerformanceMetrics as IntegratedPerformanceMetrics,
};
pub use pipeline::{
    PipelineBuilder, PipelineMetadata, PipelineMetrics, PipelineProcessingConfig, PipelineResult,
    PipelineStage, ProcessingMode, ResourceUsage, UnifiedVoirsPipeline,
};
pub use traits::{
    AlertSeverity, ComponentConfig, ComponentHealth, EcosystemComponent, LogLevel, MessagePriority,
    PerformanceMetrics as TraitsPerformanceMetrics, PerformanceThresholds, ResourceMetrics,
};

use crate::RecognitionError;
use std::collections::HashMap;
use voirs_sdk::config::PipelineConfig;

/// Integration manager for VoiRS ecosystem components
#[derive(Debug)]
pub struct VoirsIntegrationManager {
    /// Configuration hierarchy
    config_hierarchy: PipelineConfig,
    /// Performance monitoring
    performance_monitor: IntegratedPerformanceMonitor,
    /// Component registry
    component_registry: HashMap<String, ComponentInfo>,
}

/// Component information for integration
#[derive(Debug, Clone)]
pub struct ComponentInfo {
    /// Component name
    pub name: String,
    /// Component version
    pub version: String,
    /// Component capabilities
    pub capabilities: Vec<String>,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
}

/// Resource requirements for components
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    /// Memory requirement in MB
    pub memory_mb: f32,
    /// CPU cores required
    pub cpu_cores: u32,
    /// GPU memory requirement in MB
    pub gpu_memory_mb: Option<f32>,
    /// Network bandwidth in Mbps
    pub network_bandwidth_mbps: Option<f32>,
}

impl VoirsIntegrationManager {
    /// Create new integration manager
    pub fn new() -> Result<Self, RecognitionError> {
        Ok(Self {
            config_hierarchy: PipelineConfig::default(),
            performance_monitor: IntegratedPerformanceMonitor::new()?,
            component_registry: HashMap::new(),
        })
    }

    /// Register a component in the ecosystem
    pub fn register_component(&mut self, info: ComponentInfo) -> Result<(), RecognitionError> {
        self.component_registry.insert(info.name.clone(), info);
        Ok(())
    }

    /// Get component information
    pub fn get_component_info(&self, name: &str) -> Option<&ComponentInfo> {
        self.component_registry.get(name)
    }

    /// Get all registered components
    pub fn get_all_components(&self) -> Vec<&ComponentInfo> {
        self.component_registry.values().collect()
    }

    /// Get configuration hierarchy
    pub fn get_config_hierarchy(&self) -> &PipelineConfig {
        &self.config_hierarchy
    }

    /// Get performance monitor
    pub fn get_performance_monitor(&self) -> &IntegratedPerformanceMonitor {
        &self.performance_monitor
    }

    /// Check ecosystem health
    pub fn health_check(&self) -> Result<EcosystemHealthStatus, RecognitionError> {
        let mut status = EcosystemHealthStatus::default();

        // Check each component
        for (name, info) in &self.component_registry {
            let component_status = self.check_component_health(name, info)?;
            status
                .component_statuses
                .insert(name.clone(), component_status);
        }

        // Overall health assessment
        let healthy_count = status
            .component_statuses
            .values()
            .filter(|s| s.is_healthy)
            .count();

        status.overall_health = if healthy_count == status.component_statuses.len() {
            HealthLevel::Healthy
        } else if healthy_count > status.component_statuses.len() / 2 {
            HealthLevel::Warning
        } else {
            HealthLevel::Critical
        };

        Ok(status)
    }

    /// Check individual component health
    fn check_component_health(
        &self,
        name: &str,
        info: &ComponentInfo,
    ) -> Result<ComponentHealthStatus, RecognitionError> {
        // This would normally check actual component status
        // For now, we'll simulate health checking
        Ok(ComponentHealthStatus {
            name: name.to_string(),
            is_healthy: true,
            uptime: std::time::Duration::from_secs(3600), // 1 hour uptime
            memory_usage_mb: info.resource_requirements.memory_mb * 0.8, // 80% of requirement
            cpu_usage_percent: 25.0,
            last_error: None,
        })
    }
}

impl Default for VoirsIntegrationManager {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

/// Overall ecosystem health status
#[derive(Debug, Default)]
pub struct EcosystemHealthStatus {
    /// Overall health level
    pub overall_health: HealthLevel,
    /// Individual component statuses
    pub component_statuses: HashMap<String, ComponentHealthStatus>,
}

/// Health level enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum HealthLevel {
    /// All systems healthy
    Healthy,
    /// Some issues detected
    Warning,
    /// Critical issues
    Critical,
}

impl Default for HealthLevel {
    fn default() -> Self {
        Self::Healthy
    }
}

/// Component health status
#[derive(Debug, Clone)]
pub struct ComponentHealthStatus {
    /// Component name
    pub name: String,
    /// Is component healthy
    pub is_healthy: bool,
    /// Component uptime
    pub uptime: std::time::Duration,
    /// Memory usage in MB
    pub memory_usage_mb: f32,
    /// CPU usage percentage
    pub cpu_usage_percent: f32,
    /// Last error if any
    pub last_error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_manager_creation() {
        let manager = VoirsIntegrationManager::new();
        assert!(manager.is_ok());
    }

    #[test]
    fn test_component_registration() {
        let mut manager = VoirsIntegrationManager::new().unwrap();

        let component_info = ComponentInfo {
            name: "test-component".to_string(),
            version: "1.0.0".to_string(),
            capabilities: vec!["transcription".to_string()],
            resource_requirements: ResourceRequirements {
                memory_mb: 512.0,
                cpu_cores: 2,
                gpu_memory_mb: Some(1024.0),
                network_bandwidth_mbps: None,
            },
        };

        assert!(manager.register_component(component_info).is_ok());
        assert!(manager.get_component_info("test-component").is_some());
    }

    #[test]
    fn test_health_check() {
        let mut manager = VoirsIntegrationManager::new().unwrap();

        let component_info = ComponentInfo {
            name: "test-component".to_string(),
            version: "1.0.0".to_string(),
            capabilities: vec!["transcription".to_string()],
            resource_requirements: ResourceRequirements {
                memory_mb: 512.0,
                cpu_cores: 2,
                gpu_memory_mb: Some(1024.0),
                network_bandwidth_mbps: None,
            },
        };

        manager.register_component(component_info).unwrap();

        let health = manager.health_check().unwrap();
        assert_eq!(health.overall_health, HealthLevel::Healthy);
        assert_eq!(health.component_statuses.len(), 1);
    }
}
