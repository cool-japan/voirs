//! Horizontal scaling management for distributed VoiRS feedback system
//!
//! This module provides automatic scaling capabilities, load monitoring,
//! and dynamic resource allocation across multiple instances.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::{Mutex, RwLock};
use tokio::time::interval;

use super::sharding::{ClusterHealth, ShardingManager};
use super::{PersistenceError, PersistenceResult};

/// Scaling errors
#[derive(Error, Debug)]
pub enum ScalingError {
    /// Resource limit exceeded
    #[error("Resource limit exceeded: {resource}")]
    ResourceLimitExceeded {
        /// Resource name
        resource: String,
    },

    /// Scaling operation failed
    #[error("Scaling operation failed: {message}")]
    ScalingFailed {
        /// Error message
        message: String,
    },

    /// Invalid scaling configuration
    #[error("Invalid scaling configuration: {message}")]
    InvalidConfig {
        /// Error message
        message: String,
    },
}

/// Horizontal scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConfig {
    /// Minimum number of instances
    pub min_instances: usize,
    /// Maximum number of instances
    pub max_instances: usize,
    /// CPU utilization threshold for scaling up (0.0 to 1.0)
    pub cpu_scale_up_threshold: f32,
    /// CPU utilization threshold for scaling down (0.0 to 1.0)
    pub cpu_scale_down_threshold: f32,
    /// Memory utilization threshold for scaling up (0.0 to 1.0)
    pub memory_scale_up_threshold: f32,
    /// Memory utilization threshold for scaling down (0.0 to 1.0)
    pub memory_scale_down_threshold: f32,
    /// Request rate threshold (requests per second)
    pub request_rate_threshold: f32,
    /// Scaling cooldown period in seconds
    pub scaling_cooldown_seconds: u64,
    /// Auto-scaling enabled
    pub auto_scaling_enabled: bool,
    /// Target response time in milliseconds
    pub target_response_time_ms: u64,
    /// Load balancing strategy
    pub load_balancing_strategy: LoadBalancingStrategy,
    /// Health check configuration
    pub health_check: HealthCheckConfig,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            min_instances: 2,
            max_instances: 10,
            cpu_scale_up_threshold: 0.7,
            cpu_scale_down_threshold: 0.3,
            memory_scale_up_threshold: 0.8,
            memory_scale_down_threshold: 0.4,
            request_rate_threshold: 1000.0,
            scaling_cooldown_seconds: 300, // 5 minutes
            auto_scaling_enabled: true,
            target_response_time_ms: 100,
            load_balancing_strategy: LoadBalancingStrategy::RoundRobin,
            health_check: HealthCheckConfig::default(),
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Least connections
    LeastConnections,
    /// Least response time
    LeastResponseTime,
    /// Weighted round-robin
    WeightedRoundRobin,
    /// Resource-based distribution
    ResourceBased,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Health check interval in seconds
    pub interval_seconds: u64,
    /// Health check timeout in seconds
    pub timeout_seconds: u64,
    /// Number of consecutive failed checks before marking unhealthy
    pub failure_threshold: u32,
    /// Number of consecutive successful checks before marking healthy
    pub success_threshold: u32,
    /// Health check endpoint path
    pub endpoint_path: String,
    /// Expected HTTP status code
    pub expected_status_code: u16,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            interval_seconds: 30,
            timeout_seconds: 5,
            failure_threshold: 3,
            success_threshold: 2,
            endpoint_path: "/health".to_string(),
            expected_status_code: 200,
        }
    }
}

/// Instance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceInfo {
    /// Instance identifier
    pub instance_id: String,
    /// Instance endpoint URL
    pub endpoint: String,
    /// Instance status
    pub status: InstanceStatus,
    /// Resource utilization
    pub resources: ResourceUtilization,
    /// Performance metrics
    pub metrics: InstanceMetrics,
    /// Instance weight for load balancing
    pub weight: f32,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last health check timestamp
    pub last_health_check: Option<DateTime<Utc>>,
    /// Region/zone information
    pub zone: Option<String>,
}

/// Instance status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InstanceStatus {
    /// Instance is healthy and serving traffic
    Healthy,
    /// Instance has degraded performance
    Degraded,
    /// Instance is unhealthy
    Unhealthy,
    /// Instance is starting up
    Starting,
    /// Instance is shutting down
    Stopping,
    /// Instance is terminated
    Terminated,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f32,
    /// Memory utilization (0.0 to 1.0)
    pub memory_utilization: f32,
    /// Disk utilization (0.0 to 1.0)
    pub disk_utilization: f32,
    /// Network utilization (0.0 to 1.0)
    pub network_utilization: f32,
    /// Active connection count
    pub active_connections: u32,
    /// Queue size
    pub queue_size: u32,
}

/// Instance performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceMetrics {
    /// Average response time in milliseconds
    pub avg_response_time_ms: f32,
    /// Request rate (requests per second)
    pub request_rate: f32,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f32,
    /// Throughput (requests per second)
    pub throughput: f32,
    /// 95th percentile response time
    pub p95_response_time_ms: f32,
    /// Total requests handled
    pub total_requests: u64,
    /// Total errors
    pub total_errors: u64,
    /// Uptime in seconds
    pub uptime_seconds: u64,
}

/// Scaling decision
#[derive(Debug, Clone)]
pub struct ScalingDecision {
    /// Action to take
    pub action: ScalingAction,
    /// Reason for the decision
    pub reason: String,
    /// Target instance count
    pub target_instances: usize,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f32,
    /// Decision timestamp
    pub timestamp: DateTime<Utc>,
}

/// Scaling actions
#[derive(Debug, Clone, PartialEq)]
pub enum ScalingAction {
    /// Scale up by adding instances
    ScaleUp(usize),
    /// Scale down by removing instances
    ScaleDown(usize),
    /// No scaling needed
    NoChange,
    /// Manual intervention required
    ManualIntervention,
}

/// Scaling event history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Scaling action taken
    pub action: String,
    /// Previous instance count
    pub previous_instances: usize,
    /// New instance count
    pub new_instances: usize,
    /// Trigger reason
    pub reason: String,
    /// Event duration
    pub duration: Duration,
    /// Success status
    pub success: bool,
}

/// Horizontal scaling manager
pub struct HorizontalScalingManager {
    /// Scaling configuration
    config: ScalingConfig,
    /// Current instances
    instances: Arc<RwLock<HashMap<String, InstanceInfo>>>,
    /// Sharding manager reference
    sharding_manager: Arc<ShardingManager>,
    /// Last scaling action timestamp
    last_scaling_action: Arc<Mutex<Option<DateTime<Utc>>>>,
    /// Scaling event history
    scaling_history: Arc<Mutex<Vec<ScalingEvent>>>,
    /// Load balancer state
    load_balancer_state: Arc<RwLock<LoadBalancerState>>,
}

/// Load balancer state
#[derive(Debug, Clone)]
pub struct LoadBalancerState {
    /// Current round-robin index
    pub round_robin_index: usize,
    /// Connection counts per instance
    pub connection_counts: HashMap<String, u32>,
    /// Response time history per instance
    pub response_times: HashMap<String, Vec<f32>>,
}

impl HorizontalScalingManager {
    /// Create a new horizontal scaling manager
    pub async fn new(
        config: ScalingConfig,
        sharding_manager: Arc<ShardingManager>,
    ) -> PersistenceResult<Self> {
        let manager = Self {
            config,
            instances: Arc::new(RwLock::new(HashMap::new())),
            sharding_manager,
            last_scaling_action: Arc::new(Mutex::new(None)),
            scaling_history: Arc::new(Mutex::new(Vec::new())),
            load_balancer_state: Arc::new(RwLock::new(LoadBalancerState {
                round_robin_index: 0,
                connection_counts: HashMap::new(),
                response_times: HashMap::new(),
            })),
        };

        // Start monitoring if auto-scaling is enabled
        if manager.config.auto_scaling_enabled {
            manager.start_auto_scaling().await;
        }

        Ok(manager)
    }

    /// Start auto-scaling monitoring
    async fn start_auto_scaling(&self) {
        let instances = self.instances.clone();
        let config = self.config.clone();
        let last_scaling_action = self.last_scaling_action.clone();
        let scaling_history = self.scaling_history.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30)); // Check every 30 seconds

            loop {
                interval.tick().await;

                if let Ok(decision) =
                    Self::make_scaling_decision(&instances, &config, &last_scaling_action).await
                {
                    if decision.action != ScalingAction::NoChange {
                        log::info!("Scaling decision: {:?}", decision);

                        // Record scaling event
                        let event = ScalingEvent {
                            timestamp: decision.timestamp,
                            action: format!("{:?}", decision.action),
                            previous_instances: instances.read().await.len(),
                            new_instances: decision.target_instances,
                            reason: decision.reason.clone(),
                            duration: Duration::from_secs(0), // Will be updated after completion
                            success: false,                   // Will be updated after completion
                        };

                        scaling_history.lock().await.push(event);
                    }
                }
            }
        });
    }

    /// Make a scaling decision based on current metrics
    async fn make_scaling_decision(
        instances: &Arc<RwLock<HashMap<String, InstanceInfo>>>,
        config: &ScalingConfig,
        last_scaling_action: &Arc<Mutex<Option<DateTime<Utc>>>>,
    ) -> PersistenceResult<ScalingDecision> {
        let instances_guard = instances.read().await;
        let current_count = instances_guard.len();

        // Check cooldown period
        if let Some(last_action) = *last_scaling_action.lock().await {
            let cooldown = Duration::from_secs(config.scaling_cooldown_seconds);
            if Utc::now()
                .signed_duration_since(last_action)
                .to_std()
                .unwrap_or(Duration::ZERO)
                < cooldown
            {
                return Ok(ScalingDecision {
                    action: ScalingAction::NoChange,
                    reason: "Scaling cooldown period active".to_string(),
                    target_instances: current_count,
                    confidence: 1.0,
                    timestamp: Utc::now(),
                });
            }
        }

        // Calculate average metrics
        let mut total_cpu = 0.0;
        let mut total_memory = 0.0;
        let mut total_response_time = 0.0;
        let mut total_request_rate = 0.0;
        let healthy_instances = instances_guard
            .values()
            .filter(|i| i.status == InstanceStatus::Healthy)
            .count();

        for instance in instances_guard.values() {
            if instance.status == InstanceStatus::Healthy {
                total_cpu += instance.resources.cpu_utilization;
                total_memory += instance.resources.memory_utilization;
                total_response_time += instance.metrics.avg_response_time_ms;
                total_request_rate += instance.metrics.request_rate;
            }
        }

        if healthy_instances == 0 {
            return Ok(ScalingDecision {
                action: ScalingAction::ManualIntervention,
                reason: "No healthy instances available".to_string(),
                target_instances: config.min_instances,
                confidence: 1.0,
                timestamp: Utc::now(),
            });
        }

        let avg_cpu = total_cpu / healthy_instances as f32;
        let avg_memory = total_memory / healthy_instances as f32;
        let avg_response_time = total_response_time / healthy_instances as f32;
        let avg_request_rate = total_request_rate / healthy_instances as f32;

        // Determine scaling action
        let mut scale_up_reasons = Vec::new();
        let mut scale_down_reasons = Vec::new();

        if avg_cpu > config.cpu_scale_up_threshold {
            scale_up_reasons.push(format!("High CPU utilization: {:.2}", avg_cpu));
        } else if avg_cpu < config.cpu_scale_down_threshold {
            scale_down_reasons.push(format!("Low CPU utilization: {:.2}", avg_cpu));
        }

        if avg_memory > config.memory_scale_up_threshold {
            scale_up_reasons.push(format!("High memory utilization: {:.2}", avg_memory));
        } else if avg_memory < config.memory_scale_down_threshold {
            scale_down_reasons.push(format!("Low memory utilization: {:.2}", avg_memory));
        }

        if avg_response_time > config.target_response_time_ms as f32 {
            scale_up_reasons.push(format!("High response time: {:.2}ms", avg_response_time));
        }

        if avg_request_rate > config.request_rate_threshold {
            scale_up_reasons.push(format!("High request rate: {:.2} req/s", avg_request_rate));
        }

        // Make decision
        if !scale_up_reasons.is_empty() && current_count < config.max_instances {
            let scale_up_count = 1; // Scale up by 1 instance at a time
            Ok(ScalingDecision {
                action: ScalingAction::ScaleUp(scale_up_count),
                reason: scale_up_reasons.join(", "),
                target_instances: current_count + scale_up_count,
                confidence: 0.8,
                timestamp: Utc::now(),
            })
        } else if !scale_down_reasons.is_empty() && current_count > config.min_instances {
            let scale_down_count = 1; // Scale down by 1 instance at a time
            Ok(ScalingDecision {
                action: ScalingAction::ScaleDown(scale_down_count),
                reason: scale_down_reasons.join(", "),
                target_instances: current_count - scale_down_count,
                confidence: 0.7,
                timestamp: Utc::now(),
            })
        } else {
            Ok(ScalingDecision {
                action: ScalingAction::NoChange,
                reason: "All metrics within thresholds".to_string(),
                target_instances: current_count,
                confidence: 0.9,
                timestamp: Utc::now(),
            })
        }
    }

    /// Add a new instance to the cluster
    pub async fn add_instance(&self, instance: InstanceInfo) -> PersistenceResult<()> {
        let instance_id = instance.instance_id.clone();
        let mut instances = self.instances.write().await;
        instances.insert(instance_id.clone(), instance);

        // Update load balancer state
        let mut lb_state = self.load_balancer_state.write().await;
        lb_state.connection_counts.insert(instance_id.clone(), 0);
        lb_state.response_times.insert(instance_id, Vec::new());

        Ok(())
    }

    /// Remove an instance from the cluster
    pub async fn remove_instance(&self, instance_id: &str) -> PersistenceResult<()> {
        let mut instances = self.instances.write().await;
        instances.remove(instance_id);

        // Update load balancer state
        let mut lb_state = self.load_balancer_state.write().await;
        lb_state.connection_counts.remove(instance_id);
        lb_state.response_times.remove(instance_id);

        Ok(())
    }

    /// Update instance metrics
    pub async fn update_instance_metrics(
        &self,
        instance_id: &str,
        resources: ResourceUtilization,
        metrics: InstanceMetrics,
    ) -> PersistenceResult<()> {
        let mut instances = self.instances.write().await;
        if let Some(instance) = instances.get_mut(instance_id) {
            instance.resources = resources;
            instance.metrics = metrics;
            instance.last_health_check = Some(Utc::now());
        }
        Ok(())
    }

    /// Get the best instance for load balancing
    pub async fn get_best_instance(&self) -> PersistenceResult<Option<String>> {
        let instances = self.instances.read().await;
        let healthy_instances: Vec<_> = instances
            .values()
            .filter(|i| i.status == InstanceStatus::Healthy)
            .collect();

        if healthy_instances.is_empty() {
            return Ok(None);
        }

        let mut lb_state = self.load_balancer_state.write().await;

        match self.config.load_balancing_strategy {
            LoadBalancingStrategy::RoundRobin => {
                let instance =
                    &healthy_instances[lb_state.round_robin_index % healthy_instances.len()];
                lb_state.round_robin_index += 1;
                Ok(Some(instance.instance_id.clone()))
            }
            LoadBalancingStrategy::LeastConnections => {
                let best_instance = healthy_instances
                    .iter()
                    .min_by_key(|i| lb_state.connection_counts.get(&i.instance_id).unwrap_or(&0))
                    .unwrap();
                Ok(Some(best_instance.instance_id.clone()))
            }
            LoadBalancingStrategy::LeastResponseTime => {
                let best_instance = healthy_instances
                    .iter()
                    .min_by(|a, b| {
                        a.metrics
                            .avg_response_time_ms
                            .partial_cmp(&b.metrics.avg_response_time_ms)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap();
                Ok(Some(best_instance.instance_id.clone()))
            }
            LoadBalancingStrategy::ResourceBased => {
                let best_instance = healthy_instances
                    .iter()
                    .min_by(|a, b| {
                        let score_a = a.resources.cpu_utilization + a.resources.memory_utilization;
                        let score_b = b.resources.cpu_utilization + b.resources.memory_utilization;
                        score_a
                            .partial_cmp(&score_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap();
                Ok(Some(best_instance.instance_id.clone()))
            }
            LoadBalancingStrategy::WeightedRoundRobin => {
                // Simplified weighted round-robin
                let best_instance = healthy_instances
                    .iter()
                    .max_by(|a, b| {
                        a.weight
                            .partial_cmp(&b.weight)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap();
                Ok(Some(best_instance.instance_id.clone()))
            }
        }
    }

    /// Get cluster scaling status
    pub async fn get_scaling_status(&self) -> ScalingStatus {
        let instances = self.instances.read().await;
        let history = self.scaling_history.lock().await;

        let healthy_count = instances
            .values()
            .filter(|i| i.status == InstanceStatus::Healthy)
            .count();

        let total_cpu = instances
            .values()
            .filter(|i| i.status == InstanceStatus::Healthy)
            .map(|i| i.resources.cpu_utilization)
            .sum::<f32>();

        let total_memory = instances
            .values()
            .filter(|i| i.status == InstanceStatus::Healthy)
            .map(|i| i.resources.memory_utilization)
            .sum::<f32>();

        let avg_cpu = if healthy_count > 0 {
            total_cpu / healthy_count as f32
        } else {
            0.0
        };

        let avg_memory = if healthy_count > 0 {
            total_memory / healthy_count as f32
        } else {
            0.0
        };

        ScalingStatus {
            total_instances: instances.len(),
            healthy_instances: healthy_count,
            target_instances: self.config.min_instances.max(healthy_count),
            avg_cpu_utilization: avg_cpu,
            avg_memory_utilization: avg_memory,
            scaling_enabled: self.config.auto_scaling_enabled,
            last_scaling_event: history.last().cloned(),
            cluster_health: self.sharding_manager.get_cluster_health(),
        }
    }
}

/// Scaling status information
#[derive(Debug, Clone)]
pub struct ScalingStatus {
    /// Total number of instances
    pub total_instances: usize,
    /// Number of healthy instances
    pub healthy_instances: usize,
    /// Target number of instances
    pub target_instances: usize,
    /// Average CPU utilization
    pub avg_cpu_utilization: f32,
    /// Average memory utilization
    pub avg_memory_utilization: f32,
    /// Whether auto-scaling is enabled
    pub scaling_enabled: bool,
    /// Last scaling event
    pub last_scaling_event: Option<ScalingEvent>,
    /// Cluster health from sharding manager
    pub cluster_health: ClusterHealth,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn create_test_instance(id: &str, cpu: f32, memory: f32) -> InstanceInfo {
        InstanceInfo {
            instance_id: id.to_string(),
            endpoint: format!("http://instance-{}", id),
            status: InstanceStatus::Healthy,
            resources: ResourceUtilization {
                cpu_utilization: cpu,
                memory_utilization: memory,
                disk_utilization: 0.5,
                network_utilization: 0.3,
                active_connections: 10,
                queue_size: 5,
            },
            metrics: InstanceMetrics {
                avg_response_time_ms: 50.0,
                request_rate: 100.0,
                error_rate: 0.01,
                throughput: 99.0,
                p95_response_time_ms: 75.0,
                total_requests: 1000,
                total_errors: 10,
                uptime_seconds: 3600,
            },
            weight: 1.0,
            created_at: Utc::now(),
            last_health_check: Some(Utc::now()),
            zone: Some("us-east-1a".to_string()),
        }
    }

    #[tokio::test]
    async fn test_scaling_decision_scale_up() {
        let instances = Arc::new(RwLock::new(HashMap::new()));
        let config = ScalingConfig {
            cpu_scale_up_threshold: 0.7,
            max_instances: 5,
            ..Default::default()
        };
        let last_scaling_action = Arc::new(Mutex::new(None));

        // Add high CPU utilization instance
        instances.write().await.insert(
            "instance1".to_string(),
            create_test_instance("instance1", 0.8, 0.5),
        );

        let decision = HorizontalScalingManager::make_scaling_decision(
            &instances,
            &config,
            &last_scaling_action,
        )
        .await
        .unwrap();

        match decision.action {
            ScalingAction::ScaleUp(count) => {
                assert_eq!(count, 1);
            }
            other => {
                assert!(false, "Expected ScaleUp action, got {:?}", other);
            }
        }
    }

    #[tokio::test]
    async fn test_scaling_decision_scale_down() {
        let instances = Arc::new(RwLock::new(HashMap::new()));
        let config = ScalingConfig {
            cpu_scale_down_threshold: 0.3,
            min_instances: 1,
            ..Default::default()
        };
        let last_scaling_action = Arc::new(Mutex::new(None));

        // Add low CPU utilization instances
        instances.write().await.insert(
            "instance1".to_string(),
            create_test_instance("instance1", 0.2, 0.3),
        );
        instances.write().await.insert(
            "instance2".to_string(),
            create_test_instance("instance2", 0.1, 0.2),
        );

        let decision = HorizontalScalingManager::make_scaling_decision(
            &instances,
            &config,
            &last_scaling_action,
        )
        .await
        .unwrap();

        match decision.action {
            ScalingAction::ScaleDown(count) => {
                assert_eq!(count, 1);
            }
            other => {
                assert!(false, "Expected ScaleDown action, got {:?}", other);
            }
        }
    }

    #[tokio::test]
    async fn test_load_balancing_strategies() {
        let config = ScalingConfig {
            load_balancing_strategy: LoadBalancingStrategy::RoundRobin,
            ..Default::default()
        };

        let sharding_manager = Arc::new(
            crate::persistence::sharding::ShardingManager::new(
                crate::persistence::sharding::ShardingConfig {
                    strategy: crate::persistence::sharding::ShardingStrategy::HashBased,
                    virtual_nodes: 3,
                    replication_factor: 2,
                    auto_rebalancing: true,
                    shards: vec![],
                    consistency_level: crate::persistence::sharding::ConsistencyLevel::Eventual,
                },
                HashMap::new(),
            )
            .await
            .unwrap(),
        );

        let manager = HorizontalScalingManager::new(config, sharding_manager)
            .await
            .unwrap();

        // Add test instances
        manager
            .add_instance(create_test_instance("instance1", 0.5, 0.4))
            .await
            .unwrap();
        manager
            .add_instance(create_test_instance("instance2", 0.3, 0.6))
            .await
            .unwrap();

        // Test round-robin
        let instance1 = manager.get_best_instance().await.unwrap();
        let instance2 = manager.get_best_instance().await.unwrap();

        assert!(instance1.is_some());
        assert!(instance2.is_some());
        assert_ne!(instance1, instance2); // Should alternate
    }
}
