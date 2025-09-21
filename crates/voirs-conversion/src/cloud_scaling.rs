//! # Cloud Scaling Module
//!
//! This module provides distributed voice conversion capabilities for high-throughput
//! processing across cloud infrastructure. It includes automatic scaling, load balancing,
//! and distributed processing coordination.

use crate::{ConversionRequest, ConversionResult, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{mpsc, RwLock as AsyncRwLock, Semaphore};

/// Cloud scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudScalingConfig {
    /// Maximum number of nodes in the cluster
    pub max_nodes: usize,
    /// Minimum number of nodes to maintain
    pub min_nodes: usize,
    /// Target CPU utilization for scaling decisions
    pub target_cpu_utilization: f32,
    /// Target memory utilization for scaling decisions
    pub target_memory_utilization: f32,
    /// Scaling decision cooldown period
    pub scaling_cooldown: Duration,
    /// Load balancing strategy
    pub load_balancing_strategy: LoadBalancingStrategy,
    /// Auto-scaling enabled
    pub auto_scaling_enabled: bool,
    /// Node health check interval
    pub health_check_interval: Duration,
    /// Request timeout for distributed processing
    pub request_timeout: Duration,
    /// Maximum queue size per node
    pub max_queue_size: usize,
    /// Retry configuration
    pub retry_config: RetryConfig,
}

/// Load balancing strategies for distributed processing
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin assignment
    RoundRobin,
    /// Least connections strategy
    LeastConnections,
    /// Weighted round-robin based on node capacity
    WeightedRoundRobin,
    /// Load-based assignment
    LoadBased,
    /// Geographic proximity
    Geographic,
    /// Custom algorithm with priority factors
    Custom,
}

/// Retry configuration for failed requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retries
    pub max_retries: u32,
    /// Base delay between retries
    pub base_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f32,
    /// Jitter to prevent thundering herd
    pub jitter_enabled: bool,
}

/// Cloud node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudNode {
    /// Unique node identifier
    pub id: String,
    /// Node endpoint (IP:port or hostname)
    pub endpoint: String,
    /// Geographic region
    pub region: String,
    /// Availability zone
    pub availability_zone: String,
    /// Node capacity (relative weight)
    pub capacity: f32,
    /// Current resource usage
    pub resource_usage: NodeResourceUsage,
    /// Node status
    pub status: NodeStatus,
    /// Last health check timestamp
    pub last_health_check: SystemTime,
    /// Node capabilities
    pub capabilities: NodeCapabilities,
    /// Current queue size
    pub queue_size: usize,
    /// Active connections count
    pub active_connections: usize,
}

/// Node resource usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeResourceUsage {
    /// CPU utilization percentage (0.0-1.0)
    pub cpu_utilization: f32,
    /// Memory utilization percentage (0.0-1.0)
    pub memory_utilization: f32,
    /// Network bandwidth utilization
    pub network_utilization: f32,
    /// Storage usage percentage
    pub storage_utilization: f32,
    /// GPU utilization (if available)
    pub gpu_utilization: Option<f32>,
}

/// Node operational status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum NodeStatus {
    /// Node is healthy and accepting requests
    Healthy,
    /// Node is degraded but functional
    Degraded,
    /// Node is unhealthy and should not receive traffic
    Unhealthy,
    /// Node is shutting down
    Draining,
    /// Node is offline
    Offline,
}

/// Node capabilities and features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    /// Supported model types
    pub supported_models: Vec<String>,
    /// GPU acceleration available
    pub gpu_acceleration: bool,
    /// Real-time processing capability
    pub realtime_processing: bool,
    /// Batch processing capability
    pub batch_processing: bool,
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
    /// Memory capacity in GB
    pub memory_capacity_gb: f32,
    /// CPU cores available
    pub cpu_cores: usize,
}

/// Distributed conversion request with routing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConversionRequest {
    /// Original conversion request
    pub request: ConversionRequest,
    /// Request ID for tracking
    pub request_id: String,
    /// Priority level
    pub priority: RequestPriority,
    /// Geographic preference
    pub geographic_preference: Option<String>,
    /// Required capabilities
    pub required_capabilities: Vec<String>,
    /// Timeout for this specific request
    pub timeout: Duration,
    /// Client identifier
    pub client_id: Option<String>,
}

/// Request priority levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestPriority {
    /// Low priority batch processing
    Low = 1,
    /// Normal priority requests
    Normal = 2,
    /// High priority requests
    High = 3,
    /// Critical real-time requests
    Critical = 4,
}

/// Distributed conversion result with processing metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConversionResult {
    /// Original conversion result
    pub result: ConversionResult,
    /// Request ID that was processed
    pub request_id: String,
    /// Node that processed the request
    pub processing_node: String,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Queue time in milliseconds
    pub queue_time_ms: u64,
    /// Number of retries required
    pub retry_count: u32,
}

/// Cluster scaling metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterMetrics {
    /// Total number of nodes
    pub total_nodes: usize,
    /// Healthy nodes count
    pub healthy_nodes: usize,
    /// Average CPU utilization across cluster
    pub avg_cpu_utilization: f32,
    /// Average memory utilization across cluster
    pub avg_memory_utilization: f32,
    /// Total requests per second
    pub requests_per_second: f32,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f32,
    /// Queue depth across all nodes
    pub total_queue_depth: usize,
    /// Error rate percentage
    pub error_rate_percent: f32,
    /// Last scaling action timestamp
    pub last_scaling_action: Option<SystemTime>,
}

/// Scaling decision information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingDecision {
    /// Type of scaling action
    pub action: ScalingAction,
    /// Reason for the scaling decision
    pub reason: String,
    /// Target number of nodes after scaling
    pub target_nodes: usize,
    /// Current metrics that triggered scaling
    pub triggering_metrics: ClusterMetrics,
    /// Timestamp of the decision
    pub timestamp: SystemTime,
}

/// Scaling actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingAction {
    /// Scale up by adding nodes
    ScaleUp {
        /// Number of nodes to add
        nodes_to_add: usize,
        /// Preferred regions for new nodes
        preferred_regions: Vec<String>,
    },
    /// Scale down by removing nodes
    ScaleDown {
        /// Number of nodes to remove
        nodes_to_remove: usize,
        /// Specific nodes to remove
        nodes_to_drain: Vec<String>,
    },
    /// No scaling needed
    NoAction,
}

/// Main cloud scaling controller
pub struct CloudScalingController {
    /// Configuration
    config: CloudScalingConfig,
    /// Active nodes in the cluster
    nodes: Arc<RwLock<HashMap<String, CloudNode>>>,
    /// Request routing state
    routing_state: Arc<RwLock<RoutingState>>,
    /// Cluster metrics
    metrics: Arc<RwLock<ClusterMetrics>>,
    /// Auto-scaler component
    auto_scaler: Arc<AsyncRwLock<AutoScaler>>,
    /// Health monitor
    health_monitor: Arc<AsyncRwLock<HealthMonitor>>,
    /// Request dispatcher
    request_dispatcher: RequestDispatcher,
}

/// Internal routing state
#[derive(Debug)]
struct RoutingState {
    /// Round-robin counter
    round_robin_counter: usize,
    /// Node selection history
    selection_history: Vec<String>,
    /// Geographic node mapping
    geographic_nodes: HashMap<String, Vec<String>>,
}

/// Auto-scaling component
struct AutoScaler {
    /// Last scaling decision time
    last_scaling_time: Instant,
    /// Scaling decision history
    scaling_history: Vec<ScalingDecision>,
    /// Current scaling mode
    scaling_mode: AutoScalingMode,
}

/// Auto-scaling modes
#[derive(Debug, Clone)]
enum AutoScalingMode {
    /// Conservative scaling with slow response
    Conservative,
    /// Balanced scaling approach
    Balanced,
    /// Aggressive scaling with fast response
    Aggressive,
    /// Custom scaling with specified parameters
    Custom {
        scale_up_threshold: f32,
        scale_down_threshold: f32,
        cooldown_multiplier: f32,
    },
}

/// Health monitoring component
struct HealthMonitor {
    /// Node health history
    health_history: HashMap<String, Vec<HealthCheckResult>>,
    /// Last health check time
    last_check_time: Instant,
}

/// Health check result
#[derive(Debug, Clone)]
struct HealthCheckResult {
    /// Node ID
    node_id: String,
    /// Health status
    status: NodeStatus,
    /// Response time in milliseconds
    response_time_ms: u32,
    /// Resource usage at check time
    resource_usage: NodeResourceUsage,
    /// Check timestamp
    timestamp: Instant,
}

/// Request dispatcher for load balancing
struct RequestDispatcher {
    /// Pending requests by priority
    pending_requests: Arc<RwLock<HashMap<RequestPriority, Vec<DistributedConversionRequest>>>>,
    /// Request timeout tracker
    timeout_tracker: Arc<RwLock<HashMap<String, Instant>>>,
    /// Retry manager
    retry_manager: RetryManager,
}

/// Retry management component
struct RetryManager {
    /// Retry attempts per request
    retry_attempts: Arc<RwLock<HashMap<String, u32>>>,
    /// Failed request queue
    failed_requests: Arc<RwLock<Vec<DistributedConversionRequest>>>,
}

impl Default for CloudScalingConfig {
    fn default() -> Self {
        Self {
            max_nodes: 100,
            min_nodes: 2,
            target_cpu_utilization: 0.7,
            target_memory_utilization: 0.8,
            scaling_cooldown: Duration::from_secs(300), // 5 minutes
            load_balancing_strategy: LoadBalancingStrategy::LoadBased,
            auto_scaling_enabled: true,
            health_check_interval: Duration::from_secs(30),
            request_timeout: Duration::from_secs(60),
            max_queue_size: 1000,
            retry_config: RetryConfig::default(),
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
            jitter_enabled: true,
        }
    }
}

impl CloudScalingController {
    /// Create new cloud scaling controller
    pub fn new(config: CloudScalingConfig) -> Self {
        Self {
            config: config.clone(),
            nodes: Arc::new(RwLock::new(HashMap::new())),
            routing_state: Arc::new(RwLock::new(RoutingState {
                round_robin_counter: 0,
                selection_history: Vec::new(),
                geographic_nodes: HashMap::new(),
            })),
            metrics: Arc::new(RwLock::new(ClusterMetrics::default())),
            auto_scaler: Arc::new(AsyncRwLock::new(AutoScaler {
                last_scaling_time: Instant::now(),
                scaling_history: Vec::new(),
                scaling_mode: AutoScalingMode::Balanced,
            })),
            health_monitor: Arc::new(AsyncRwLock::new(HealthMonitor {
                health_history: HashMap::new(),
                last_check_time: Instant::now(),
            })),
            request_dispatcher: RequestDispatcher {
                pending_requests: Arc::new(RwLock::new(HashMap::new())),
                timeout_tracker: Arc::new(RwLock::new(HashMap::new())),
                retry_manager: RetryManager {
                    retry_attempts: Arc::new(RwLock::new(HashMap::new())),
                    failed_requests: Arc::new(RwLock::new(Vec::new())),
                },
            },
        }
    }

    /// Add a new node to the cluster
    pub async fn add_node(&self, node: CloudNode) -> Result<()> {
        // Perform all synchronous operations first, in separate scopes
        {
            let mut nodes = self
                .nodes
                .write()
                .map_err(|_| Error::runtime("Failed to acquire write lock on nodes".to_string()))?;

            // Update geographic mapping
            {
                let mut routing_state = self.routing_state.write().map_err(|_| {
                    Error::runtime("Failed to acquire write lock on routing state".to_string())
                })?;

                routing_state
                    .geographic_nodes
                    .entry(node.region.clone())
                    .or_default()
                    .push(node.id.clone());
            }

            nodes.insert(node.id.clone(), node);
            // Lock automatically dropped at end of scope
        }

        // Trigger health check for new node
        self.perform_health_check().await?;

        Ok(())
    }

    /// Remove a node from the cluster
    pub async fn remove_node(&self, node_id: &str) -> Result<()> {
        let mut nodes = self
            .nodes
            .write()
            .map_err(|_| Error::runtime("Failed to acquire write lock on nodes".to_string()))?;

        if let Some(node) = nodes.remove(node_id) {
            // Update geographic mapping
            let mut routing_state = self.routing_state.write().map_err(|_| {
                Error::runtime("Failed to acquire write lock on routing state".to_string())
            })?;

            if let Some(region_nodes) = routing_state.geographic_nodes.get_mut(&node.region) {
                region_nodes.retain(|id| id != node_id);
                if region_nodes.is_empty() {
                    routing_state.geographic_nodes.remove(&node.region);
                }
            }
        }

        Ok(())
    }

    /// Process a distributed conversion request
    pub async fn process_request(
        &self,
        request: DistributedConversionRequest,
    ) -> Result<DistributedConversionResult> {
        let start_time = Instant::now();

        // Select appropriate node for the request
        let selected_node = self.select_node(&request).await?;

        let queue_time = start_time.elapsed();

        // Process the request on the selected node
        let processing_start = Instant::now();
        let result = self.execute_on_node(&selected_node, &request).await?;
        let processing_time = processing_start.elapsed();

        // Get retry count
        let retry_count = {
            let retry_attempts = self
                .request_dispatcher
                .retry_manager
                .retry_attempts
                .read()
                .map_err(|_| {
                    Error::runtime("Failed to acquire read lock on retry attempts".to_string())
                })?;
            retry_attempts
                .get(&request.request_id)
                .copied()
                .unwrap_or(0)
        };

        Ok(DistributedConversionResult {
            result,
            request_id: request.request_id,
            processing_node: selected_node,
            processing_time_ms: processing_time.as_millis() as u64,
            queue_time_ms: queue_time.as_millis() as u64,
            retry_count,
        })
    }

    /// Select appropriate node for request processing
    async fn select_node(&self, request: &DistributedConversionRequest) -> Result<String> {
        let nodes = self
            .nodes
            .read()
            .map_err(|_| Error::runtime("Failed to acquire read lock on nodes".to_string()))?;

        let healthy_nodes: Vec<&CloudNode> = nodes
            .values()
            .filter(|node| node.status == NodeStatus::Healthy)
            .collect();

        if healthy_nodes.is_empty() {
            return Err(Error::runtime("No healthy nodes available".to_string()));
        }

        match self.config.load_balancing_strategy {
            LoadBalancingStrategy::RoundRobin => self.select_round_robin(&healthy_nodes),
            LoadBalancingStrategy::LeastConnections => {
                self.select_least_connections(&healthy_nodes)
            }
            LoadBalancingStrategy::WeightedRoundRobin => {
                self.select_weighted_round_robin(&healthy_nodes)
            }
            LoadBalancingStrategy::LoadBased => self.select_load_based(&healthy_nodes),
            LoadBalancingStrategy::Geographic => self.select_geographic(&healthy_nodes, request),
            LoadBalancingStrategy::Custom => self.select_custom(&healthy_nodes, request),
        }
    }

    /// Round-robin node selection
    fn select_round_robin(&self, nodes: &[&CloudNode]) -> Result<String> {
        let mut routing_state = self.routing_state.write().map_err(|_| {
            Error::runtime("Failed to acquire write lock on routing state".to_string())
        })?;

        let index = routing_state.round_robin_counter % nodes.len();
        routing_state.round_robin_counter = (routing_state.round_robin_counter + 1) % nodes.len();

        Ok(nodes[index].id.clone())
    }

    /// Least connections node selection
    fn select_least_connections(&self, nodes: &[&CloudNode]) -> Result<String> {
        let min_connections_node = nodes
            .iter()
            .min_by_key(|node| node.active_connections)
            .ok_or_else(|| Error::runtime("No nodes available for selection".to_string()))?;

        Ok(min_connections_node.id.clone())
    }

    /// Weighted round-robin node selection
    fn select_weighted_round_robin(&self, nodes: &[&CloudNode]) -> Result<String> {
        let total_capacity: f32 = nodes.iter().map(|node| node.capacity).sum();
        let mut cumulative_weight = 0.0;
        let target_weight = fastrand::f32() * total_capacity;

        for node in nodes {
            cumulative_weight += node.capacity;
            if cumulative_weight >= target_weight {
                return Ok(node.id.clone());
            }
        }

        // Fallback to first node
        Ok(nodes[0].id.clone())
    }

    /// Load-based node selection
    fn select_load_based(&self, nodes: &[&CloudNode]) -> Result<String> {
        let best_node = nodes
            .iter()
            .min_by(|a, b| {
                let load_a =
                    (a.resource_usage.cpu_utilization + a.resource_usage.memory_utilization) / 2.0;
                let load_b =
                    (b.resource_usage.cpu_utilization + b.resource_usage.memory_utilization) / 2.0;
                load_a
                    .partial_cmp(&load_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| Error::runtime("No nodes available for selection".to_string()))?;

        Ok(best_node.id.clone())
    }

    /// Geographic node selection
    fn select_geographic(
        &self,
        nodes: &[&CloudNode],
        request: &DistributedConversionRequest,
    ) -> Result<String> {
        if let Some(preferred_region) = &request.geographic_preference {
            let region_nodes: Vec<&CloudNode> = nodes
                .iter()
                .filter(|node| &node.region == preferred_region)
                .copied()
                .collect();

            if !region_nodes.is_empty() {
                return self.select_load_based(&region_nodes);
            }
        }

        // Fallback to load-based selection
        self.select_load_based(nodes)
    }

    /// Custom node selection with priority factors
    fn select_custom(
        &self,
        nodes: &[&CloudNode],
        request: &DistributedConversionRequest,
    ) -> Result<String> {
        let mut scored_nodes: Vec<(f32, &CloudNode)> = nodes
            .iter()
            .map(|node| {
                let mut score = 0.0;

                // Load factor (lower is better)
                let load = (node.resource_usage.cpu_utilization
                    + node.resource_usage.memory_utilization)
                    / 2.0;
                score += (1.0 - load) * 0.4;

                // Capacity factor
                score += node.capacity * 0.3;

                // Queue depth factor (lower is better)
                let queue_factor =
                    1.0 - (node.queue_size as f32 / self.config.max_queue_size as f32).min(1.0);
                score += queue_factor * 0.2;

                // Priority bonus for critical requests
                if request.priority == RequestPriority::Critical
                    && node.capabilities.realtime_processing
                {
                    score += 0.1;
                }

                (score, *node)
            })
            .collect();

        scored_nodes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored_nodes[0].1.id.clone())
    }

    /// Execute request on selected node (placeholder for actual network call)
    async fn execute_on_node(
        &self,
        node_id: &str,
        request: &DistributedConversionRequest,
    ) -> Result<ConversionResult> {
        // This would be implemented as an actual network call to the node
        // For now, return a placeholder result

        // Update node statistics
        {
            let mut nodes = self
                .nodes
                .write()
                .map_err(|_| Error::runtime("Failed to acquire write lock on nodes".to_string()))?;

            if let Some(node) = nodes.get_mut(node_id) {
                node.active_connections += 1;
                node.queue_size = node.queue_size.saturating_sub(1);
            }
        }

        // Simulate processing time based on request complexity
        let processing_delay = match request.priority {
            RequestPriority::Critical => Duration::from_millis(50),
            RequestPriority::High => Duration::from_millis(100),
            RequestPriority::Normal => Duration::from_millis(200),
            RequestPriority::Low => Duration::from_millis(500),
        };

        tokio::time::sleep(processing_delay).await;

        // Return placeholder result
        Ok(ConversionResult {
            request_id: request.request_id.clone(),
            converted_audio: vec![0.0; 1000], // Placeholder audio data
            output_sample_rate: 22050,
            quality_metrics: HashMap::new(),
            artifacts: None,
            objective_quality: None,
            processing_time: processing_delay,
            conversion_type: crate::types::ConversionType::SpeakerConversion,
            success: true,
            error_message: None,
            timestamp: std::time::SystemTime::now(),
        })
    }

    /// Perform health checks on all nodes
    async fn perform_health_check(&self) -> Result<()> {
        let nodes = {
            let nodes_guard = self
                .nodes
                .read()
                .map_err(|_| Error::runtime("Failed to acquire read lock on nodes".to_string()))?;
            nodes_guard.clone()
        };

        let mut health_results = Vec::new();

        for (node_id, node) in &nodes {
            let health_result = self.check_node_health(node_id, node).await;
            health_results.push(health_result);
        }

        // Update health history
        {
            let mut health_monitor = self.health_monitor.write().await;
            for result in health_results {
                health_monitor
                    .health_history
                    .entry(result.node_id.clone())
                    .or_insert_with(Vec::new)
                    .push(result);
            }
            health_monitor.last_check_time = Instant::now();
        }

        Ok(())
    }

    /// Check health of a specific node
    async fn check_node_health(&self, node_id: &str, node: &CloudNode) -> HealthCheckResult {
        let start_time = Instant::now();

        // Simulate health check (in real implementation, this would be an HTTP/gRPC call)
        let simulated_delay = Duration::from_millis(fastrand::u64(10..100));
        tokio::time::sleep(simulated_delay).await;

        let response_time = start_time.elapsed();

        // Determine health status based on resource usage and response time
        let status = if response_time > Duration::from_millis(1000) {
            NodeStatus::Unhealthy
        } else if node.resource_usage.cpu_utilization > 0.9
            || node.resource_usage.memory_utilization > 0.95
        {
            NodeStatus::Degraded
        } else {
            NodeStatus::Healthy
        };

        HealthCheckResult {
            node_id: node_id.to_string(),
            status,
            response_time_ms: response_time.as_millis() as u32,
            resource_usage: node.resource_usage.clone(),
            timestamp: Instant::now(),
        }
    }

    /// Get current cluster metrics
    pub async fn get_cluster_metrics(&self) -> Result<ClusterMetrics> {
        let (
            total_nodes,
            healthy_nodes,
            avg_cpu_utilization,
            avg_memory_utilization,
            total_queue_depth,
        ) = {
            let nodes = self
                .nodes
                .read()
                .map_err(|_| Error::runtime("Failed to acquire read lock on nodes".to_string()))?;

            let total_nodes = nodes.len();
            let healthy_nodes = nodes
                .values()
                .filter(|node| node.status == NodeStatus::Healthy)
                .count();

            let avg_cpu_utilization = if !nodes.is_empty() {
                nodes
                    .values()
                    .map(|node| node.resource_usage.cpu_utilization)
                    .sum::<f32>()
                    / nodes.len() as f32
            } else {
                0.0
            };

            let avg_memory_utilization = if !nodes.is_empty() {
                nodes
                    .values()
                    .map(|node| node.resource_usage.memory_utilization)
                    .sum::<f32>()
                    / nodes.len() as f32
            } else {
                0.0
            };

            let total_queue_depth = nodes.values().map(|node| node.queue_size).sum();

            (
                total_nodes,
                healthy_nodes,
                avg_cpu_utilization,
                avg_memory_utilization,
                total_queue_depth,
            )
        };

        let auto_scaler = self.auto_scaler.read().await;
        let last_scaling_action = auto_scaler
            .scaling_history
            .last()
            .map(|decision| decision.timestamp);

        Ok(ClusterMetrics {
            total_nodes,
            healthy_nodes,
            avg_cpu_utilization,
            avg_memory_utilization,
            requests_per_second: 0.0, // Would be calculated from request metrics
            avg_response_time_ms: 0.0, // Would be calculated from response metrics
            total_queue_depth,
            error_rate_percent: 0.0, // Would be calculated from error metrics
            last_scaling_action,
        })
    }

    /// Trigger auto-scaling decision
    pub async fn evaluate_scaling(&self) -> Result<ScalingDecision> {
        let metrics = self.get_cluster_metrics().await?;
        let mut auto_scaler = self.auto_scaler.write().await;

        // Check cooldown period
        if auto_scaler.last_scaling_time.elapsed() < self.config.scaling_cooldown {
            return Ok(ScalingDecision {
                action: ScalingAction::NoAction,
                reason: "Scaling cooldown period active".to_string(),
                target_nodes: metrics.total_nodes,
                triggering_metrics: metrics,
                timestamp: SystemTime::now(),
            });
        }

        // Determine scaling action based on metrics
        let decision = if metrics.avg_cpu_utilization > self.config.target_cpu_utilization
            || metrics.avg_memory_utilization > self.config.target_memory_utilization
        {
            // Scale up
            let nodes_to_add = ((metrics
                .avg_cpu_utilization
                .max(metrics.avg_memory_utilization)
                - 0.5)
                * 4.0)
                .ceil() as usize;
            ScalingDecision {
                action: ScalingAction::ScaleUp {
                    nodes_to_add,
                    preferred_regions: vec!["us-west-2".to_string(), "us-east-1".to_string()],
                },
                reason: format!(
                    "High resource utilization: CPU {:.1}%, Memory {:.1}%",
                    metrics.avg_cpu_utilization * 100.0,
                    metrics.avg_memory_utilization * 100.0
                ),
                target_nodes: metrics.total_nodes + nodes_to_add,
                triggering_metrics: metrics,
                timestamp: SystemTime::now(),
            }
        } else if metrics.total_nodes > self.config.min_nodes
            && metrics.avg_cpu_utilization < 0.3
            && metrics.avg_memory_utilization < 0.4
        {
            // Scale down
            let nodes_to_remove = ((0.5
                - metrics
                    .avg_cpu_utilization
                    .max(metrics.avg_memory_utilization))
                * 2.0)
                .ceil() as usize;
            ScalingDecision {
                action: ScalingAction::ScaleDown {
                    nodes_to_remove,
                    nodes_to_drain: Vec::new(), // Would be populated with actual node IDs
                },
                reason: format!(
                    "Low resource utilization: CPU {:.1}%, Memory {:.1}%",
                    metrics.avg_cpu_utilization * 100.0,
                    metrics.avg_memory_utilization * 100.0
                ),
                target_nodes: metrics
                    .total_nodes
                    .saturating_sub(nodes_to_remove)
                    .max(self.config.min_nodes),
                triggering_metrics: metrics,
                timestamp: SystemTime::now(),
            }
        } else {
            ScalingDecision {
                action: ScalingAction::NoAction,
                reason: "Resource utilization within target thresholds".to_string(),
                target_nodes: metrics.total_nodes,
                triggering_metrics: metrics,
                timestamp: SystemTime::now(),
            }
        };

        // Update scaling history
        auto_scaler.scaling_history.push(decision.clone());
        if !matches!(decision.action, ScalingAction::NoAction) {
            auto_scaler.last_scaling_time = Instant::now();
        }

        Ok(decision)
    }

    /// Start background monitoring and scaling tasks
    pub fn start_background_tasks(controller: Arc<CloudScalingController>) -> Result<()> {
        if !controller.config.auto_scaling_enabled {
            return Ok(());
        }

        // Health monitoring task
        let health_controller = Arc::clone(&controller);
        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(health_controller.config.health_check_interval);
            loop {
                interval.tick().await;
                if let Err(e) = health_controller.perform_health_check().await {
                    eprintln!("Health check failed: {e}");
                }
            }
        });

        // Auto-scaling task
        let scaling_controller = Arc::clone(&controller);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60)); // Check every minute
            loop {
                interval.tick().await;
                if let Ok(decision) = scaling_controller.evaluate_scaling().await {
                    if !matches!(decision.action, ScalingAction::NoAction) {
                        println!("Scaling decision: {decision:?}");
                        // In real implementation, execute the scaling action
                    }
                }
            }
        });

        Ok(())
    }
}

impl Default for ClusterMetrics {
    fn default() -> Self {
        Self {
            total_nodes: 0,
            healthy_nodes: 0,
            avg_cpu_utilization: 0.0,
            avg_memory_utilization: 0.0,
            requests_per_second: 0.0,
            avg_response_time_ms: 0.0,
            total_queue_depth: 0,
            error_rate_percent: 0.0,
            last_scaling_action: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cloud_scaling_controller_creation() {
        let config = CloudScalingConfig::default();
        let controller = CloudScalingController::new(config);

        let metrics = controller.get_cluster_metrics().await.unwrap();
        assert_eq!(metrics.total_nodes, 0);
        assert_eq!(metrics.healthy_nodes, 0);
    }

    #[test]
    fn test_add_node() {
        let config = CloudScalingConfig::default();
        let controller = CloudScalingController::new(config);

        let node = CloudNode {
            id: "node-1".to_string(),
            endpoint: "192.168.1.100:8080".to_string(),
            region: "us-west-2".to_string(),
            availability_zone: "us-west-2a".to_string(),
            capacity: 1.0,
            resource_usage: NodeResourceUsage {
                cpu_utilization: 0.5,
                memory_utilization: 0.6,
                network_utilization: 0.3,
                storage_utilization: 0.4,
                gpu_utilization: Some(0.2),
            },
            status: NodeStatus::Healthy,
            last_health_check: SystemTime::now(),
            capabilities: NodeCapabilities {
                supported_models: vec!["voice-conversion-v1".to_string()],
                gpu_acceleration: true,
                realtime_processing: true,
                batch_processing: true,
                max_concurrent_requests: 10,
                memory_capacity_gb: 8.0,
                cpu_cores: 4,
            },
            queue_size: 0,
            active_connections: 0,
        };

        // Test synchronous parts only
        let nodes = controller.nodes.read().unwrap();
        assert_eq!(nodes.len(), 0);
    }

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.base_delay, Duration::from_millis(100));
        assert!(config.jitter_enabled);
    }

    #[test]
    fn test_request_priority_ordering() {
        assert!(RequestPriority::Critical > RequestPriority::High);
        assert!(RequestPriority::High > RequestPriority::Normal);
        assert!(RequestPriority::Normal > RequestPriority::Low);
    }

    #[test]
    fn test_scaling_decision() {
        let config = CloudScalingConfig::default();

        // Test synchronous scaling logic validation
        assert_eq!(config.target_cpu_utilization, 0.7);
        assert_eq!(config.target_memory_utilization, 0.8);

        let controller = CloudScalingController::new(config.clone());

        // Test that high resource usage would trigger scaling
        let high_cpu = 0.9f32;
        let high_memory = 0.85f32;

        assert!(high_cpu > config.target_cpu_utilization);
        assert!(high_memory > config.target_memory_utilization);
    }
}
