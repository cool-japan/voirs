// Copyright (c) 2024 VoiRS Contributors
// Licensed under MIT OR Apache-2.0

//! Health checking for service instances

use super::{InstanceStatus, Result, ServiceInstance};
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    /// Instance ID
    pub instance_id: String,
    /// Status
    pub status: InstanceStatus,
    /// Response time
    pub response_time: Duration,
    /// Error message if unhealthy
    pub error: Option<String>,
    /// Timestamp
    pub timestamp: Instant,
}

/// Health checker for monitoring service instances
pub struct HealthChecker {
    check_interval: Duration,
    timeout: Duration,
    last_check: Arc<RwLock<Option<Instant>>>,
}

impl HealthChecker {
    /// Create a new health checker
    #[must_use]
    pub fn new(check_interval: Duration, timeout: Duration) -> Self {
        Self {
            check_interval,
            timeout,
            last_check: Arc::new(RwLock::new(None)),
        }
    }

    /// Perform health check on an instance
    pub async fn check_instance(&self, instance: &ServiceInstance) -> HealthCheckResult {
        let start = Instant::now();

        debug!("Checking health of instance {}", instance.id);

        // Simulate health check (in real impl, would make HTTP/gRPC call)
        tokio::time::sleep(Duration::from_millis(10)).await;

        let response_time = start.elapsed();
        let status = if response_time < self.timeout {
            InstanceStatus::Healthy
        } else {
            InstanceStatus::Degraded
        };

        *self.last_check.write() = Some(Instant::now());

        HealthCheckResult {
            instance_id: instance.id.clone(),
            status,
            response_time,
            error: None,
            timestamp: Instant::now(),
        }
    }

    /// Perform health checks on multiple instances
    pub async fn check_instances(&self, instances: Vec<ServiceInstance>) -> Vec<HealthCheckResult> {
        let mut results = Vec::new();

        for instance in instances {
            let result = self.check_instance(&instance).await;
            results.push(result);
        }

        results
    }

    /// Get time since last check
    #[must_use]
    pub fn time_since_last_check(&self) -> Option<Duration> {
        self.last_check.read().map(|t| t.elapsed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_checker_basic() {
        let checker = HealthChecker::new(Duration::from_secs(10), Duration::from_millis(100));

        let instance = ServiceInstance {
            id: "instance-1".to_string(),
            address: "localhost".to_string(),
            port: 8080,
            status: InstanceStatus::Healthy,
            weight: 1,
            active_connections: 0,
            avg_response_time: Duration::from_millis(50),
        };

        let result = checker.check_instance(&instance).await;
        assert_eq!(result.instance_id, "instance-1");
        assert!(result.error.is_none());
    }

    #[tokio::test]
    async fn test_health_checker_multiple_instances() {
        let checker = HealthChecker::new(Duration::from_secs(10), Duration::from_millis(100));

        let instances = vec![
            ServiceInstance {
                id: "instance-1".to_string(),
                address: "localhost".to_string(),
                port: 8080,
                status: InstanceStatus::Healthy,
                weight: 1,
                active_connections: 0,
                avg_response_time: Duration::from_millis(50),
            },
            ServiceInstance {
                id: "instance-2".to_string(),
                address: "localhost".to_string(),
                port: 8081,
                status: InstanceStatus::Healthy,
                weight: 1,
                active_connections: 0,
                avg_response_time: Duration::from_millis(50),
            },
        ];

        let results = checker.check_instances(instances).await;
        assert_eq!(results.len(), 2);
    }
}
