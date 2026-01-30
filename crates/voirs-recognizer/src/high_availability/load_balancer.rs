// Copyright (c) 2024 VoiRS Contributors
// Licensed under MIT OR Apache-2.0

//! Load balancing strategies for distributing requests across service instances

use super::{
    HighAvailabilityError, InstanceStatus, LoadBalancingStrategy, Result, ServiceInstance,
};
use parking_lot::RwLock;
use std::sync::Arc;
use tracing::debug;

/// Load balancer for distributing requests
pub struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    instances: Arc<RwLock<Vec<ServiceInstance>>>,
    current_index: Arc<RwLock<usize>>,
}

impl LoadBalancer {
    /// Create a new load balancer
    #[must_use]
    pub fn new(strategy: LoadBalancingStrategy, instances: Vec<ServiceInstance>) -> Self {
        Self {
            strategy,
            instances: Arc::new(RwLock::new(instances)),
            current_index: Arc::new(RwLock::new(0)),
        }
    }

    /// Select next instance based on strategy
    pub fn select_instance(&self) -> Result<ServiceInstance> {
        let instances = self.instances.read();

        // Filter healthy instances
        let healthy: Vec<&ServiceInstance> = instances
            .iter()
            .filter(|i| i.status == InstanceStatus::Healthy)
            .collect();

        if healthy.is_empty() {
            return Err(HighAvailabilityError::NoHealthyInstances);
        }

        match self.strategy {
            LoadBalancingStrategy::RoundRobin => self.round_robin(&healthy),
            LoadBalancingStrategy::LeastConnections => self.least_connections(&healthy),
            LoadBalancingStrategy::LeastResponseTime => self.least_response_time(&healthy),
            LoadBalancingStrategy::WeightedRoundRobin => self.weighted_round_robin(&healthy),
            LoadBalancingStrategy::Random => self.random_selection(&healthy),
            LoadBalancingStrategy::IpHash => self.round_robin(&healthy), // Simplified
        }
    }

    /// Round-robin selection
    fn round_robin(&self, instances: &[&ServiceInstance]) -> Result<ServiceInstance> {
        let mut index = self.current_index.write();
        let selected = instances[*index % instances.len()].clone();
        *index = (*index + 1) % instances.len();
        Ok(selected.clone())
    }

    /// Least connections selection
    fn least_connections(&self, instances: &[&ServiceInstance]) -> Result<ServiceInstance> {
        let selected = instances
            .iter()
            .min_by_key(|i| i.active_connections)
            .ok_or(HighAvailabilityError::NoHealthyInstances)?;

        Ok((*selected).clone())
    }

    /// Least response time selection
    fn least_response_time(&self, instances: &[&ServiceInstance]) -> Result<ServiceInstance> {
        let selected = instances
            .iter()
            .min_by_key(|i| i.avg_response_time)
            .ok_or(HighAvailabilityError::NoHealthyInstances)?;

        Ok((*selected).clone())
    }

    /// Weighted round-robin selection
    fn weighted_round_robin(&self, instances: &[&ServiceInstance]) -> Result<ServiceInstance> {
        // Simplified: select based on weight
        let total_weight: u32 = instances.iter().map(|i| i.weight).sum();
        if total_weight == 0 {
            return self.round_robin(instances);
        }

        let mut index = self.current_index.write();
        let selected = instances[*index % instances.len()];
        *index += 1;

        Ok(selected.clone())
    }

    /// Random selection
    fn random_selection(&self, instances: &[&ServiceInstance]) -> Result<ServiceInstance> {
        use scirs2_core::random::{thread_rng, Rng};
        let mut rng = thread_rng();
        let idx = rng.gen_range(0..instances.len());
        Ok(instances[idx].clone())
    }

    /// Update instance list
    pub fn update_instances(&self, instances: Vec<ServiceInstance>) {
        *self.instances.write() = instances;
    }

    /// Get current instance count
    #[must_use]
    pub fn instance_count(&self) -> usize {
        self.instances.read().len()
    }

    /// Get healthy instance count
    #[must_use]
    pub fn healthy_instance_count(&self) -> usize {
        self.instances
            .read()
            .iter()
            .filter(|i| i.status == InstanceStatus::Healthy)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_instances() -> Vec<ServiceInstance> {
        vec![
            ServiceInstance {
                id: "instance-1".to_string(),
                address: "localhost".to_string(),
                port: 8080,
                status: InstanceStatus::Healthy,
                weight: 2,
                active_connections: 5,
                avg_response_time: Duration::from_millis(50),
            },
            ServiceInstance {
                id: "instance-2".to_string(),
                address: "localhost".to_string(),
                port: 8081,
                status: InstanceStatus::Healthy,
                weight: 1,
                active_connections: 3,
                avg_response_time: Duration::from_millis(30),
            },
            ServiceInstance {
                id: "instance-3".to_string(),
                address: "localhost".to_string(),
                port: 8082,
                status: InstanceStatus::Unhealthy,
                weight: 1,
                active_connections: 0,
                avg_response_time: Duration::from_millis(100),
            },
        ]
    }

    #[test]
    fn test_load_balancer_round_robin() {
        let instances = create_test_instances();
        let lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin, instances);

        assert_eq!(lb.healthy_instance_count(), 2);

        let instance1 = lb.select_instance().unwrap();
        let instance2 = lb.select_instance().unwrap();

        assert_ne!(instance1.id, instance2.id);
    }

    #[test]
    fn test_load_balancer_least_connections() {
        let instances = create_test_instances();
        let lb = LoadBalancer::new(LoadBalancingStrategy::LeastConnections, instances);

        let instance = lb.select_instance().unwrap();
        assert_eq!(instance.id, "instance-2"); // Has least connections (3)
    }

    #[test]
    fn test_load_balancer_least_response_time() {
        let instances = create_test_instances();
        let lb = LoadBalancer::new(LoadBalancingStrategy::LeastResponseTime, instances);

        let instance = lb.select_instance().unwrap();
        assert_eq!(instance.id, "instance-2"); // Has fastest response time
    }

    #[test]
    fn test_load_balancer_no_healthy_instances() {
        let instances = vec![ServiceInstance {
            id: "instance-1".to_string(),
            address: "localhost".to_string(),
            port: 8080,
            status: InstanceStatus::Unhealthy,
            weight: 1,
            active_connections: 0,
            avg_response_time: Duration::from_millis(50),
        }];

        let lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin, instances);
        let result = lb.select_instance();

        assert!(result.is_err());
    }

    #[test]
    fn test_load_balancer_update_instances() {
        let instances = create_test_instances();
        let lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin, instances);

        assert_eq!(lb.instance_count(), 3);

        let new_instances = vec![ServiceInstance {
            id: "new-instance".to_string(),
            address: "localhost".to_string(),
            port: 9000,
            status: InstanceStatus::Healthy,
            weight: 1,
            active_connections: 0,
            avg_response_time: Duration::from_millis(50),
        }];

        lb.update_instances(new_instances);
        assert_eq!(lb.instance_count(), 1);
    }
}
