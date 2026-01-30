// Copyright (c) 2024 VoiRS Contributors
// Licensed under MIT OR Apache-2.0

//! High availability architecture for production deployments
//!
//! This module provides infrastructure for building highly available `VoiRS` Recognizer
//! deployments with features like load balancing, failover, health monitoring, and
//! distributed state management.

pub mod circuit_breaker;
pub mod failover;
pub mod health_check;
pub mod load_balancer;
pub mod state_management;

use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

/// High availability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighAvailabilityConfig {
    /// Enable high availability features
    pub enabled: bool,
    /// Minimum number of healthy instances
    pub min_healthy_instances: usize,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Failover timeout
    pub failover_timeout: Duration,
    /// Enable circuit breaker
    pub enable_circuit_breaker: bool,
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerConfig,
    /// Enable load balancing
    pub enable_load_balancing: bool,
    /// Load balancing strategy
    pub load_balancing_strategy: LoadBalancingStrategy,
}

impl Default for HighAvailabilityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_healthy_instances: 2,
            health_check_interval: Duration::from_secs(10),
            failover_timeout: Duration::from_secs(30),
            enable_circuit_breaker: true,
            circuit_breaker: CircuitBreakerConfig::default(),
            enable_load_balancing: true,
            load_balancing_strategy: LoadBalancingStrategy::RoundRobin,
        }
    }
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Failure threshold before opening circuit
    pub failure_threshold: usize,
    /// Success threshold before closing circuit
    pub success_threshold: usize,
    /// Timeout duration in open state
    pub timeout: Duration,
    /// Half-open retry count
    pub half_open_max_retries: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
            half_open_max_retries: 3,
        }
    }
}

/// Load balancing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Least connections
    LeastConnections,
    /// Least response time
    LeastResponseTime,
    /// Weighted round-robin
    WeightedRoundRobin,
    /// Random selection
    Random,
    /// IP hash-based
    IpHash,
}

/// Service instance status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InstanceStatus {
    /// Instance is healthy
    Healthy,
    /// Instance is degraded
    Degraded,
    /// Instance is unhealthy
    Unhealthy,
    /// Instance is offline
    Offline,
}

/// Service instance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInstance {
    /// Instance ID
    pub id: String,
    /// Instance address
    pub address: String,
    /// Instance port
    pub port: u16,
    /// Instance status
    pub status: InstanceStatus,
    /// Instance weight (for weighted load balancing)
    pub weight: u32,
    /// Active connections
    pub active_connections: usize,
    /// Average response time
    pub avg_response_time: Duration,
}

/// High availability error
#[derive(Debug, Error)]
pub enum HighAvailabilityError {
    /// No healthy instances available
    #[error("No healthy instances available")]
    NoHealthyInstances,
    /// Failover failed
    #[error("Failover failed: {0}")]
    FailoverFailed(String),
    /// Circuit breaker open
    #[error("Circuit breaker is open")]
    CircuitBreakerOpen,
    /// Health check failed
    #[error("Health check failed: {0}")]
    HealthCheckFailed(String),
    /// Load balancer error
    #[error("Load balancer error: {0}")]
    LoadBalancerError(String),
    /// State synchronization error
    #[error("State synchronization failed: {0}")]
    StateSyncError(String),
}

/// High availability result type
pub type Result<T> = std::result::Result<T, HighAvailabilityError>;

/// High availability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighAvailabilityMetrics {
    /// Total number of instances
    pub total_instances: usize,
    /// Number of healthy instances
    pub healthy_instances: usize,
    /// Number of degraded instances
    pub degraded_instances: usize,
    /// Number of unhealthy instances
    pub unhealthy_instances: usize,
    /// Total failovers
    pub total_failovers: u64,
    /// Successful failovers
    pub successful_failovers: u64,
    /// Failed failovers
    pub failed_failovers: u64,
    /// Average failover time
    pub avg_failover_time: Duration,
    /// Circuit breaker trips
    pub circuit_breaker_trips: u64,
    /// Uptime percentage
    pub uptime_percentage: f32,
}

impl Default for HighAvailabilityMetrics {
    fn default() -> Self {
        Self {
            total_instances: 0,
            healthy_instances: 0,
            degraded_instances: 0,
            unhealthy_instances: 0,
            total_failovers: 0,
            successful_failovers: 0,
            failed_failovers: 0,
            avg_failover_time: Duration::from_secs(0),
            circuit_breaker_trips: 0,
            uptime_percentage: 100.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ha_config_default() {
        let config = HighAvailabilityConfig::default();
        assert!(config.enabled);
        assert_eq!(config.min_healthy_instances, 2);
        assert!(config.enable_circuit_breaker);
        assert!(config.enable_load_balancing);
    }

    #[test]
    fn test_load_balancing_strategies() {
        assert_eq!(
            LoadBalancingStrategy::RoundRobin,
            LoadBalancingStrategy::RoundRobin
        );
        assert_ne!(
            LoadBalancingStrategy::RoundRobin,
            LoadBalancingStrategy::Random
        );
    }

    #[test]
    fn test_instance_status() {
        assert_eq!(InstanceStatus::Healthy, InstanceStatus::Healthy);
        assert_ne!(InstanceStatus::Healthy, InstanceStatus::Degraded);
    }

    #[test]
    fn test_ha_metrics_default() {
        let metrics = HighAvailabilityMetrics::default();
        assert_eq!(metrics.total_instances, 0);
        assert_eq!(metrics.total_failovers, 0);
        assert!((metrics.uptime_percentage - 100.0).abs() < f32::EPSILON);
    }
}
