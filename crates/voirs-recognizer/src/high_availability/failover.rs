// Copyright (c) 2024 VoiRS Contributors
// Licensed under MIT OR Apache-2.0

//! Automatic failover mechanism for high availability

use super::{HighAvailabilityError, InstanceStatus, Result, ServiceInstance};
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// Failover event
#[derive(Debug, Clone)]
pub struct FailoverEvent {
    /// Event timestamp
    pub timestamp: Instant,
    /// Source instance ID
    pub from_instance: String,
    /// Target instance ID
    pub to_instance: String,
    /// Failover reason
    pub reason: String,
    /// Failover duration
    pub duration: Duration,
    /// Success status
    pub success: bool,
}

/// Failover manager
pub struct FailoverManager {
    timeout: Duration,
    max_retries: usize,
    failover_history: Arc<RwLock<Vec<FailoverEvent>>>,
}

impl FailoverManager {
    /// Create a new failover manager
    #[must_use]
    pub fn new(timeout: Duration, max_retries: usize) -> Self {
        Self {
            timeout,
            max_retries,
            failover_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Perform failover from unhealthy instance to healthy backup
    pub async fn failover(
        &self,
        from_instance: &ServiceInstance,
        to_instance: &ServiceInstance,
        reason: String,
    ) -> Result<FailoverEvent> {
        let start = Instant::now();

        info!(
            "Initiating failover from {} to {} (reason: {})",
            from_instance.id, to_instance.id, reason
        );

        // Check if target instance is healthy
        if to_instance.status != InstanceStatus::Healthy {
            warn!("Target instance {} is not healthy", to_instance.id);
            return Err(HighAvailabilityError::FailoverFailed(
                "Target instance not healthy".to_string(),
            ));
        }

        // Simulate failover process
        tokio::time::sleep(Duration::from_millis(100)).await;

        let duration = start.elapsed();

        if duration > self.timeout {
            let event = FailoverEvent {
                timestamp: start,
                from_instance: from_instance.id.clone(),
                to_instance: to_instance.id.clone(),
                reason: reason.clone(),
                duration,
                success: false,
            };

            self.failover_history.write().push(event.clone());

            return Err(HighAvailabilityError::FailoverFailed(
                "Failover timeout exceeded".to_string(),
            ));
        }

        let event = FailoverEvent {
            timestamp: start,
            from_instance: from_instance.id.clone(),
            to_instance: to_instance.id.clone(),
            reason,
            duration,
            success: true,
        };

        self.failover_history.write().push(event.clone());

        info!("Failover completed successfully in {:?}", duration);

        Ok(event)
    }

    /// Get failover history
    #[must_use]
    pub fn get_history(&self) -> Vec<FailoverEvent> {
        self.failover_history.read().clone()
    }

    /// Get successful failover count
    #[must_use]
    pub fn successful_failover_count(&self) -> usize {
        self.failover_history
            .read()
            .iter()
            .filter(|e| e.success)
            .count()
    }

    /// Get failed failover count
    #[must_use]
    pub fn failed_failover_count(&self) -> usize {
        self.failover_history
            .read()
            .iter()
            .filter(|e| !e.success)
            .count()
    }

    /// Calculate average failover time
    #[must_use]
    pub fn average_failover_time(&self) -> Duration {
        let history = self.failover_history.read();

        if history.is_empty() {
            return Duration::from_secs(0);
        }

        let total: Duration = history.iter().map(|e| e.duration).sum();
        total / history.len() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_healthy_instance(id: &str) -> ServiceInstance {
        ServiceInstance {
            id: id.to_string(),
            address: "localhost".to_string(),
            port: 8080,
            status: InstanceStatus::Healthy,
            weight: 1,
            active_connections: 0,
            avg_response_time: Duration::from_millis(50),
        }
    }

    fn create_unhealthy_instance(id: &str) -> ServiceInstance {
        ServiceInstance {
            id: id.to_string(),
            address: "localhost".to_string(),
            port: 8080,
            status: InstanceStatus::Unhealthy,
            weight: 1,
            active_connections: 0,
            avg_response_time: Duration::from_millis(50),
        }
    }

    #[tokio::test]
    async fn test_failover_success() {
        let manager = FailoverManager::new(Duration::from_secs(5), 3);

        let from = create_unhealthy_instance("instance-1");
        let to = create_healthy_instance("instance-2");

        let result = manager
            .failover(&from, &to, "Health check failed".to_string())
            .await;

        assert!(result.is_ok());
        let event = result.unwrap();
        assert!(event.success);
        assert_eq!(event.from_instance, "instance-1");
        assert_eq!(event.to_instance, "instance-2");
    }

    #[tokio::test]
    async fn test_failover_to_unhealthy_instance() {
        let manager = FailoverManager::new(Duration::from_secs(5), 3);

        let from = create_unhealthy_instance("instance-1");
        let to = create_unhealthy_instance("instance-2");

        let result = manager
            .failover(&from, &to, "Health check failed".to_string())
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_failover_history() {
        let manager = FailoverManager::new(Duration::from_secs(5), 3);

        let from = create_unhealthy_instance("instance-1");
        let to = create_healthy_instance("instance-2");

        let _ = manager
            .failover(&from, &to, "Test failover".to_string())
            .await;

        let history = manager.get_history();
        assert_eq!(history.len(), 1);
        assert_eq!(manager.successful_failover_count(), 1);
        assert_eq!(manager.failed_failover_count(), 0);
    }

    #[tokio::test]
    async fn test_average_failover_time() {
        let manager = FailoverManager::new(Duration::from_secs(5), 3);

        let from = create_unhealthy_instance("instance-1");
        let to = create_healthy_instance("instance-2");

        // Perform multiple failovers
        for i in 0..3 {
            let _ = manager
                .failover(&from, &to, format!("Failover {}", i))
                .await;
        }

        let avg_time = manager.average_failover_time();
        assert!(avg_time.as_millis() > 0);
    }
}
