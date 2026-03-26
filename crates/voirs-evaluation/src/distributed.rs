//! Distributed evaluation system for parallel processing across multiple nodes
//!
//! This module provides a framework for distributing evaluation tasks across multiple
//! computing nodes to improve performance and scalability for large-scale evaluations.

use crate::traits::{EvaluationResult, QualityScore};
use crate::EvaluationError;
use scirs2_core::random::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, RwLock};
use tokio::time::{timeout, Duration};
use uuid::Uuid;
use voirs_sdk::AudioBuffer;

/// Configuration for distributed evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Maximum number of worker nodes
    pub max_workers: usize,
    /// Minimum number of worker nodes
    pub min_workers: usize,
    /// Task timeout in seconds
    pub task_timeout_seconds: u64,
    /// Heartbeat interval in seconds
    pub heartbeat_interval_seconds: u64,
    /// Maximum retries for failed tasks
    pub max_retries: u32,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Fault tolerance configuration
    pub fault_tolerance: FaultToleranceConfig,
    /// Auto-scaling configuration
    pub auto_scaling: AutoScalingConfig,
    /// Edge computing configuration
    pub edge_computing: EdgeComputingConfig,
    /// Cluster configuration
    pub cluster_config: ClusterConfig,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            max_workers: 10,
            min_workers: 1,
            task_timeout_seconds: 300,
            heartbeat_interval_seconds: 30,
            max_retries: 3,
            load_balancing: LoadBalancingStrategy::RoundRobin,
            fault_tolerance: FaultToleranceConfig::default(),
            auto_scaling: AutoScalingConfig::default(),
            edge_computing: EdgeComputingConfig::default(),
            cluster_config: ClusterConfig::default(),
        }
    }
}

/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    /// Enable auto-scaling
    pub enabled: bool,
    /// Target CPU utilization percentage (0-100)
    pub target_cpu_utilization: f32,
    /// Target memory utilization percentage (0-100)
    pub target_memory_utilization: f32,
    /// Target queue depth before scaling up
    pub target_queue_depth: usize,
    /// Scale up threshold (consecutive checks before scaling up)
    pub scale_up_threshold: u32,
    /// Scale down threshold (consecutive checks before scaling down)
    pub scale_down_threshold: u32,
    /// Cooldown period in seconds after scaling
    pub cooldown_seconds: u64,
    /// Scale up increment (number of workers to add)
    pub scale_up_increment: usize,
    /// Scale down decrement (number of workers to remove)
    pub scale_down_decrement: usize,
}

impl Default for AutoScalingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            target_cpu_utilization: 70.0,
            target_memory_utilization: 80.0,
            target_queue_depth: 20,
            scale_up_threshold: 3,
            scale_down_threshold: 5,
            cooldown_seconds: 300,
            scale_up_increment: 2,
            scale_down_decrement: 1,
        }
    }
}

/// Edge computing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeComputingConfig {
    /// Enable edge computing optimizations
    pub enabled: bool,
    /// Enable bandwidth-aware task assignment
    pub bandwidth_aware: bool,
    /// Enable local result caching at edge nodes
    pub edge_caching: bool,
    /// Maximum bandwidth per edge node (MB/s)
    pub max_bandwidth_mbps: f32,
    /// Enable task compression for network transfer
    pub compress_transfers: bool,
    /// Enable edge-to-edge communication
    pub edge_to_edge_enabled: bool,
    /// Latency threshold for edge classification (ms)
    pub edge_latency_threshold_ms: u64,
}

impl Default for EdgeComputingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bandwidth_aware: true,
            edge_caching: true,
            max_bandwidth_mbps: 100.0,
            compress_transfers: true,
            edge_to_edge_enabled: false,
            edge_latency_threshold_ms: 50,
        }
    }
}

/// Cluster configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Enable cluster mode
    pub enabled: bool,
    /// Cluster name
    pub cluster_name: String,
    /// Enable consensus protocol
    pub consensus_enabled: bool,
    /// Consensus algorithm
    pub consensus_algorithm: ConsensusAlgorithm,
    /// Enable data replication
    pub replication_enabled: bool,
    /// Replication factor
    pub replication_factor: usize,
    /// Enable partition tolerance
    pub partition_tolerance: bool,
    /// Cluster discovery method
    pub discovery_method: ClusterDiscoveryMethod,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cluster_name: "voirs-eval-cluster".to_string(),
            consensus_enabled: false,
            consensus_algorithm: ConsensusAlgorithm::Raft,
            replication_enabled: false,
            replication_factor: 3,
            partition_tolerance: true,
            discovery_method: ClusterDiscoveryMethod::Static,
        }
    }
}

/// Consensus algorithms for cluster coordination
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConsensusAlgorithm {
    /// Raft consensus algorithm
    Raft,
    /// Paxos consensus algorithm
    Paxos,
    /// Simple leader election
    LeaderElection,
}

/// Cluster discovery methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ClusterDiscoveryMethod {
    /// Static configuration
    Static,
    /// DNS-based discovery
    Dns,
    /// Kubernetes service discovery
    Kubernetes,
    /// Consul service discovery
    Consul,
}

/// Load balancing strategies for distributing tasks
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Least-loaded worker distribution
    LeastLoaded,
    /// Random distribution
    Random,
    /// Weighted distribution based on worker capacity
    Weighted,
    /// Latency-aware distribution (for edge computing)
    LatencyAware,
    /// Bandwidth-aware distribution
    BandwidthAware,
    /// Adaptive distribution based on performance
    Adaptive,
}

/// Fault tolerance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceConfig {
    /// Enable automatic task redistribution on failure
    pub auto_redistribute: bool,
    /// Enable worker health monitoring
    pub health_monitoring: bool,
    /// Maximum node failures before system shutdown
    pub max_node_failures: u32,
    /// Enable task result validation
    pub result_validation: bool,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            auto_redistribute: true,
            health_monitoring: true,
            max_node_failures: 3,
            result_validation: true,
        }
    }
}

/// Unique identifier for evaluation tasks
pub type TaskId = Uuid;

/// Unique identifier for worker nodes
pub type WorkerId = Uuid;

/// Evaluation task that can be distributed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationTask {
    /// Unique task identifier
    pub id: TaskId,
    /// Task type
    pub task_type: TaskType,
    /// Audio data to evaluate
    pub audio_data: Vec<u8>, // Serialized audio buffer
    /// Reference audio data (optional)
    pub reference_data: Option<Vec<u8>>,
    /// Evaluation parameters
    pub parameters: TaskParameters,
    /// Task priority
    pub priority: TaskPriority,
    /// Maximum execution time
    pub max_execution_time: Duration,
    /// Number of retry attempts
    pub retry_count: u32,
}

/// Types of evaluation tasks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TaskType {
    /// Quality metric evaluation
    QualityMetrics,
    /// Pronunciation assessment
    PronunciationAssessment,
    /// Comparative analysis
    ComparativeAnalysis,
    /// Perceptual evaluation
    PerceptualEvaluation,
    /// Statistical analysis
    StatisticalAnalysis,
    /// Custom evaluation task
    Custom(String),
}

/// Parameters for evaluation tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskParameters {
    /// Metrics to compute
    pub metrics: Vec<String>,
    /// Language for evaluation
    pub language: Option<String>,
    /// Sample rate
    pub sample_rate: Option<u32>,
    /// Number of channels
    pub channels: Option<u16>,
    /// Additional custom parameters
    pub custom_params: HashMap<String, String>,
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TaskPriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Result of an evaluation task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    /// Task identifier
    pub task_id: TaskId,
    /// Worker that completed the task
    pub worker_id: WorkerId,
    /// Execution result
    pub result: Result<EvaluationOutput, String>,
    /// Execution time
    pub execution_time: Duration,
    /// Resource usage statistics
    pub resource_usage: ResourceUsage,
    /// Timestamp of completion
    pub completed_at: std::time::SystemTime,
}

/// Output of an evaluation task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationOutput {
    /// Quality scores
    pub quality_scores: HashMap<String, f32>,
    /// Detailed metrics
    pub metrics: HashMap<String, serde_json::Value>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Resource usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU usage percentage
    pub cpu_usage: f32,
    /// Memory usage in MB
    pub memory_usage: f32,
    /// Disk I/O in MB
    pub disk_io: f32,
    /// Network I/O in MB
    pub network_io: f32,
}

/// Information about a worker node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Unique worker identifier
    pub id: WorkerId,
    /// Worker name/address
    pub name: String,
    /// Worker capabilities
    pub capabilities: WorkerCapabilities,
    /// Current status
    pub status: WorkerStatus,
    /// Current load
    pub current_load: f32,
    /// Last heartbeat timestamp
    pub last_heartbeat: std::time::SystemTime,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Network metrics for edge computing
    pub network_metrics: NetworkMetrics,
    /// Worker location (for edge computing)
    pub location: Option<WorkerLocation>,
    /// Is this an edge node
    pub is_edge_node: bool,
}

/// Worker capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerCapabilities {
    /// Maximum concurrent tasks
    pub max_concurrent_tasks: usize,
    /// Supported task types
    pub supported_task_types: Vec<TaskType>,
    /// Available memory in MB
    pub available_memory: f32,
    /// CPU cores
    pub cpu_cores: usize,
    /// Specialized hardware (GPU, etc.)
    pub specialized_hardware: Vec<String>,
}

/// Worker status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerStatus {
    /// Worker is online and available
    Online,
    /// Worker is busy processing tasks
    Busy,
    /// Worker is offline
    Offline,
    /// Worker has failed
    Failed,
    /// Worker is being drained (no new tasks)
    Draining,
}

/// Network metrics for worker nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Average latency to coordinator (ms)
    pub latency_ms: f32,
    /// Available bandwidth (MB/s)
    pub bandwidth_mbps: f32,
    /// Packet loss percentage
    pub packet_loss: f32,
    /// Network jitter (ms)
    pub jitter_ms: f32,
    /// Total data transferred (MB)
    pub total_data_transferred_mb: f32,
}

impl Default for NetworkMetrics {
    fn default() -> Self {
        Self {
            latency_ms: 10.0,
            bandwidth_mbps: 100.0,
            packet_loss: 0.0,
            jitter_ms: 1.0,
            total_data_transferred_mb: 0.0,
        }
    }
}

/// Worker location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerLocation {
    /// Geographic region
    pub region: String,
    /// Data center or availability zone
    pub zone: Option<String>,
    /// Latitude
    pub latitude: Option<f64>,
    /// Longitude
    pub longitude: Option<f64>,
}

/// Performance metrics for workers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total tasks completed
    pub tasks_completed: u64,
    /// Total tasks failed
    pub tasks_failed: u64,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Success rate
    pub success_rate: f32,
    /// Throughput (tasks per second)
    pub throughput: f32,
}

/// Distributed evaluation coordinator
pub struct DistributedEvaluator {
    /// Configuration
    config: DistributedConfig,
    /// Registered workers
    workers: Arc<RwLock<HashMap<WorkerId, WorkerInfo>>>,
    /// Task queue
    task_queue: Arc<RwLock<Vec<EvaluationTask>>>,
    /// Running tasks
    running_tasks: Arc<RwLock<HashMap<TaskId, (WorkerId, std::time::SystemTime)>>>,
    /// Completed tasks
    completed_tasks: Arc<RwLock<HashMap<TaskId, TaskResult>>>,
    /// Task distribution channel
    task_sender: mpsc::UnboundedSender<EvaluationTask>,
    /// Result collection channel
    result_receiver: Arc<RwLock<mpsc::UnboundedReceiver<TaskResult>>>,
    /// Shutdown signal
    shutdown_sender: broadcast::Sender<()>,
    /// Statistics
    stats: Arc<RwLock<SystemStatistics>>,
    /// Auto-scaling state
    scaling_state: Arc<RwLock<AutoScalingState>>,
    /// Cluster state
    cluster_state: Arc<RwLock<ClusterState>>,
}

/// Auto-scaling state tracking
#[derive(Debug, Clone, Default)]
pub struct AutoScalingState {
    /// Last scaling action timestamp
    pub last_scaling_action: Option<std::time::SystemTime>,
    /// Scale up counter (consecutive checks above threshold)
    pub scale_up_counter: u32,
    /// Scale down counter (consecutive checks below threshold)
    pub scale_down_counter: u32,
    /// Current system load percentage
    pub current_system_load: f32,
    /// Scaling recommendations
    pub scaling_recommendation: ScalingRecommendation,
}

/// Scaling recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScalingRecommendation {
    /// Scale up recommended
    ScaleUp,
    /// Scale down recommended
    ScaleDown,
    /// No scaling needed
    #[default]
    NoAction,
}

/// Cluster state for coordination
#[derive(Debug, Clone, Default)]
pub struct ClusterState {
    /// Is this node the leader
    pub is_leader: bool,
    /// Current leader ID
    pub leader_id: Option<WorkerId>,
    /// Cluster members
    pub members: Vec<WorkerId>,
    /// Last election timestamp
    pub last_election: Option<std::time::SystemTime>,
    /// Election term
    pub term: u64,
}

/// System-wide statistics
#[derive(Debug, Clone, Default)]
pub struct SystemStatistics {
    /// Total tasks submitted
    pub tasks_submitted: u64,
    /// Total tasks completed
    pub tasks_completed: u64,
    /// Total tasks failed
    pub tasks_failed: u64,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Average task completion time
    pub avg_completion_time: Duration,
    /// System throughput
    pub system_throughput: f32,
    /// Active workers
    pub active_workers: usize,
    /// Failed workers
    pub failed_workers: usize,
}

impl DistributedEvaluator {
    /// Create a new distributed evaluator
    pub fn new(config: DistributedConfig) -> Self {
        let (task_sender, task_receiver) = mpsc::unbounded_channel();
        let (result_sender, result_receiver) = mpsc::unbounded_channel();
        let (shutdown_sender, _) = broadcast::channel(1);

        let evaluator = Self {
            config: config.clone(),
            workers: Arc::new(RwLock::new(HashMap::new())),
            task_queue: Arc::new(RwLock::new(Vec::new())),
            running_tasks: Arc::new(RwLock::new(HashMap::new())),
            completed_tasks: Arc::new(RwLock::new(HashMap::new())),
            task_sender,
            result_receiver: Arc::new(RwLock::new(result_receiver)),
            shutdown_sender: shutdown_sender.clone(),
            stats: Arc::new(RwLock::new(SystemStatistics::default())),
            scaling_state: Arc::new(RwLock::new(AutoScalingState::default())),
            cluster_state: Arc::new(RwLock::new(ClusterState::default())),
        };

        // Start background task management
        evaluator.start_background_tasks(task_receiver, result_sender);

        // Start auto-scaling monitor if enabled
        if config.auto_scaling.enabled {
            evaluator.start_auto_scaling_monitor();
        }

        // Start cluster management if enabled
        if config.cluster_config.enabled {
            evaluator.start_cluster_management();
        }

        evaluator
    }

    /// Start background task management
    fn start_background_tasks(
        &self,
        mut task_receiver: mpsc::UnboundedReceiver<EvaluationTask>,
        result_sender: mpsc::UnboundedSender<TaskResult>,
    ) {
        let workers = Arc::clone(&self.workers);
        let running_tasks = Arc::clone(&self.running_tasks);
        let completed_tasks = Arc::clone(&self.completed_tasks);
        let config = self.config.clone();
        let stats = Arc::clone(&self.stats);
        let mut shutdown_receiver = self.shutdown_sender.subscribe();

        // Task distribution loop
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    Some(task) = task_receiver.recv() => {
                        Self::distribute_task(
                            task,
                            Arc::clone(&workers),
                            Arc::clone(&running_tasks),
                            result_sender.clone(),
                            config.clone(),
                        ).await;
                    }
                    _ = shutdown_receiver.recv() => {
                        break;
                    }
                }
            }
        });

        // Health monitoring loop
        let workers_monitor = Arc::clone(&self.workers);
        let stats_monitor = Arc::clone(&self.stats);
        let config_monitor = self.config.clone();
        let mut shutdown_monitor = self.shutdown_sender.subscribe();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(
                config_monitor.heartbeat_interval_seconds,
            ));

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        Self::monitor_worker_health(
                            Arc::clone(&workers_monitor),
                            Arc::clone(&stats_monitor),
                        ).await;
                    }
                    _ = shutdown_monitor.recv() => {
                        break;
                    }
                }
            }
        });
    }

    /// Register a new worker node
    pub async fn register_worker(&self, worker_info: WorkerInfo) -> EvaluationResult<()> {
        let mut workers = self.workers.write().await;
        workers.insert(worker_info.id, worker_info);

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.active_workers = workers.len();

        Ok(())
    }

    /// Unregister a worker node
    pub async fn unregister_worker(&self, worker_id: WorkerId) -> EvaluationResult<()> {
        let mut workers = self.workers.write().await;
        workers.remove(&worker_id);

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.active_workers = workers.len();

        Ok(())
    }

    /// Submit a task for evaluation
    pub async fn submit_task(&self, task: EvaluationTask) -> EvaluationResult<TaskId> {
        let task_id = task.id;

        // Add to task queue
        let mut queue = self.task_queue.write().await;
        queue.push(task.clone());

        // Sort by priority
        queue.sort_by_key(|b| std::cmp::Reverse(b.priority));

        // Send to distribution channel
        self.task_sender
            .send(task)
            .map_err(|e| EvaluationError::QualityEvaluationError {
                message: format!("Failed to submit task: {}", e),
                source: None,
            })?;

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.tasks_submitted += 1;

        Ok(task_id)
    }

    /// Get task result
    pub async fn get_result(&self, task_id: TaskId) -> EvaluationResult<Option<TaskResult>> {
        let completed_tasks = self.completed_tasks.read().await;
        Ok(completed_tasks.get(&task_id).cloned())
    }

    /// Wait for task completion
    pub async fn wait_for_completion(&self, task_id: TaskId) -> EvaluationResult<TaskResult> {
        let timeout_duration = Duration::from_secs(self.config.task_timeout_seconds);

        timeout(timeout_duration, async {
            loop {
                if let Some(result) = self.get_result(task_id).await? {
                    return Ok(result);
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
        .await
        .map_err(|_| EvaluationError::QualityEvaluationError {
            message: "Task completion timeout".to_string(),
            source: None,
        })?
    }

    /// Distribute task to an available worker
    async fn distribute_task(
        task: EvaluationTask,
        workers: Arc<RwLock<HashMap<WorkerId, WorkerInfo>>>,
        running_tasks: Arc<RwLock<HashMap<TaskId, (WorkerId, std::time::SystemTime)>>>,
        result_sender: mpsc::UnboundedSender<TaskResult>,
        config: DistributedConfig,
    ) {
        let selected_worker = Self::select_worker(&workers, &task, config.load_balancing).await;

        if let Some(worker_id) = selected_worker {
            // Mark task as running
            let mut running = running_tasks.write().await;
            running.insert(task.id, (worker_id, std::time::SystemTime::now()));

            // Simulate task execution (in real implementation, this would be network call)
            let task_clone = task.clone();
            tokio::spawn(async move {
                let result = Self::execute_task_on_worker(task_clone, worker_id).await;
                let _ = result_sender.send(result);
            });
        }
    }

    /// Select the best worker for a task
    async fn select_worker(
        workers: &Arc<RwLock<HashMap<WorkerId, WorkerInfo>>>,
        task: &EvaluationTask,
        strategy: LoadBalancingStrategy,
    ) -> Option<WorkerId> {
        let workers_read = workers.read().await;
        let available_workers: Vec<_> = workers_read
            .values()
            .filter(|w| w.status == WorkerStatus::Online)
            .filter(|w| {
                w.capabilities
                    .supported_task_types
                    .contains(&task.task_type)
            })
            .collect();

        if available_workers.is_empty() {
            return None;
        }

        match strategy {
            LoadBalancingStrategy::RoundRobin => {
                // Simple round-robin (simplified implementation)
                available_workers.first().map(|w| w.id)
            }
            LoadBalancingStrategy::LeastLoaded => {
                // Select worker with lowest current load
                available_workers
                    .iter()
                    .min_by(|a, b| {
                        a.current_load
                            .partial_cmp(&b.current_load)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|w| w.id)
            }
            LoadBalancingStrategy::Random => {
                // Random selection
                if available_workers.is_empty() {
                    None
                } else {
                    let mut rng = scirs2_core::random::Random::seed(0);
                    let idx = rng.random_range(0..available_workers.len());
                    Some(available_workers[idx].id)
                }
            }
            LoadBalancingStrategy::Weighted => {
                // Weighted by capabilities (simplified)
                available_workers
                    .iter()
                    .max_by_key(|w| w.capabilities.max_concurrent_tasks)
                    .map(|w| w.id)
            }
            LoadBalancingStrategy::LatencyAware => {
                // Select worker with lowest latency
                available_workers
                    .iter()
                    .min_by(|a, b| {
                        a.network_metrics
                            .latency_ms
                            .partial_cmp(&b.network_metrics.latency_ms)
                            .expect("value should be present")
                    })
                    .map(|w| w.id)
            }
            LoadBalancingStrategy::BandwidthAware => {
                // Select worker with highest available bandwidth
                available_workers
                    .iter()
                    .max_by(|a, b| {
                        a.network_metrics
                            .bandwidth_mbps
                            .partial_cmp(&b.network_metrics.bandwidth_mbps)
                            .expect("value should be present")
                    })
                    .map(|w| w.id)
            }
            LoadBalancingStrategy::Adaptive => {
                // Adaptive strategy combining multiple factors
                available_workers
                    .iter()
                    .min_by(|a, b| {
                        let score_a = Self::calculate_worker_score(a);
                        let score_b = Self::calculate_worker_score(b);
                        score_a
                            .partial_cmp(&score_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|w| w.id)
            }
        }
    }

    /// Calculate adaptive worker score (lower is better)
    fn calculate_worker_score(worker: &WorkerInfo) -> f32 {
        let load_weight = 0.4;
        let latency_weight = 0.3;
        let bandwidth_weight = 0.2;
        let success_rate_weight = 0.1;

        let load_score = worker.current_load * 100.0;
        let latency_score = worker.network_metrics.latency_ms;
        let bandwidth_score = 100.0 - worker.network_metrics.bandwidth_mbps.min(100.0);
        let success_rate_score = (1.0 - worker.performance_metrics.success_rate) * 100.0;

        load_weight * load_score
            + latency_weight * latency_score
            + bandwidth_weight * bandwidth_score
            + success_rate_weight * success_rate_score
    }

    /// Execute task on a worker (simulated)
    async fn execute_task_on_worker(task: EvaluationTask, worker_id: WorkerId) -> TaskResult {
        use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let mut rng = Random::seed(seed);
        let start_time = std::time::SystemTime::now();

        // Simulate task execution time
        let execution_time = Duration::from_millis(rng.random::<u64>() % 5000 + 1000);
        tokio::time::sleep(execution_time).await;

        // Simulate task result
        let result = if rng.random::<f32>() > 0.1 {
            // 90% success rate
            Ok(EvaluationOutput {
                quality_scores: {
                    let mut scores = HashMap::new();
                    scores.insert("pesq".to_string(), rng.random::<f32>() * 5.0);
                    scores.insert("stoi".to_string(), rng.random::<f32>());
                    scores
                },
                metrics: HashMap::new(),
                metadata: HashMap::new(),
            })
        } else {
            Err("Simulated task failure".to_string())
        };

        TaskResult {
            task_id: task.id,
            worker_id,
            result,
            execution_time,
            resource_usage: ResourceUsage {
                cpu_usage: rng.random::<f32>() * 100.0,
                memory_usage: rng.random::<f32>() * 1024.0,
                disk_io: rng.random::<f32>() * 100.0,
                network_io: rng.random::<f32>() * 50.0,
            },
            completed_at: start_time,
        }
    }

    /// Monitor worker health
    async fn monitor_worker_health(
        workers: Arc<RwLock<HashMap<WorkerId, WorkerInfo>>>,
        stats: Arc<RwLock<SystemStatistics>>,
    ) {
        let mut workers_write = workers.write().await;
        let mut failed_count = 0;

        for worker in workers_write.values_mut() {
            let now = std::time::SystemTime::now();
            let time_since_heartbeat = now
                .duration_since(worker.last_heartbeat)
                .unwrap_or_default();

            if time_since_heartbeat > Duration::from_secs(60) {
                worker.status = WorkerStatus::Failed;
                failed_count += 1;
            }
        }

        // Update statistics
        let mut stats_write = stats.write().await;
        stats_write.failed_workers = failed_count;
        stats_write.active_workers = workers_write.len() - failed_count;
    }

    /// Get system statistics
    pub async fn get_statistics(&self) -> SystemStatistics {
        self.stats.read().await.clone()
    }

    /// Get worker information
    pub async fn get_workers(&self) -> Vec<WorkerInfo> {
        let workers = self.workers.read().await;
        workers.values().cloned().collect()
    }

    /// Start auto-scaling monitor
    fn start_auto_scaling_monitor(&self) {
        let workers = Arc::clone(&self.workers);
        let task_queue = Arc::clone(&self.task_queue);
        let scaling_state = Arc::clone(&self.scaling_state);
        let config = self.config.clone();
        let mut shutdown_receiver = self.shutdown_sender.subscribe();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        Self::evaluate_scaling_needs(
                            Arc::clone(&workers),
                            Arc::clone(&task_queue),
                            Arc::clone(&scaling_state),
                            &config.auto_scaling,
                            config.min_workers,
                            config.max_workers,
                        ).await;
                    }
                    _ = shutdown_receiver.recv() => {
                        break;
                    }
                }
            }
        });
    }

    /// Evaluate scaling needs and update recommendations
    async fn evaluate_scaling_needs(
        workers: Arc<RwLock<HashMap<WorkerId, WorkerInfo>>>,
        task_queue: Arc<RwLock<Vec<EvaluationTask>>>,
        scaling_state: Arc<RwLock<AutoScalingState>>,
        config: &AutoScalingConfig,
        min_workers: usize,
        max_workers: usize,
    ) {
        let workers_read = workers.read().await;
        let queue_read = task_queue.read().await;
        let mut state = scaling_state.write().await;

        // Calculate system metrics
        let active_workers = workers_read
            .values()
            .filter(|w| w.status == WorkerStatus::Online)
            .count();
        let queue_depth = queue_read.len();

        // Calculate average CPU and memory utilization
        let total_cpu: f32 = workers_read.values().map(|w| w.current_load).sum();
        let avg_cpu = if active_workers > 0 {
            total_cpu / active_workers as f32 * 100.0
        } else {
            0.0
        };

        let total_memory: f32 = workers_read
            .values()
            .map(|w| {
                (w.capabilities.available_memory - w.performance_metrics.throughput)
                    / w.capabilities.available_memory
                    * 100.0
            })
            .sum();
        let avg_memory = if active_workers > 0 {
            total_memory / active_workers as f32
        } else {
            0.0
        };

        state.current_system_load = (avg_cpu + avg_memory) / 2.0;

        // Check cooldown period
        let in_cooldown = if let Some(last_action) = state.last_scaling_action {
            let elapsed = std::time::SystemTime::now()
                .duration_since(last_action)
                .unwrap_or_default();
            elapsed < Duration::from_secs(config.cooldown_seconds)
        } else {
            false
        };

        if in_cooldown {
            return;
        }

        // Evaluate scale up conditions
        let should_scale_up = active_workers < max_workers
            && (avg_cpu > config.target_cpu_utilization
                || avg_memory > config.target_memory_utilization
                || queue_depth > config.target_queue_depth);

        // Evaluate scale down conditions
        let should_scale_down = active_workers > min_workers
            && avg_cpu < config.target_cpu_utilization * 0.5
            && avg_memory < config.target_memory_utilization * 0.5
            && queue_depth < config.target_queue_depth / 2;

        // Update counters and recommendations
        if should_scale_up {
            state.scale_up_counter += 1;
            state.scale_down_counter = 0;

            if state.scale_up_counter >= config.scale_up_threshold {
                state.scaling_recommendation = ScalingRecommendation::ScaleUp;
                state.scale_up_counter = 0;
                state.last_scaling_action = Some(std::time::SystemTime::now());
            }
        } else if should_scale_down {
            state.scale_down_counter += 1;
            state.scale_up_counter = 0;

            if state.scale_down_counter >= config.scale_down_threshold {
                state.scaling_recommendation = ScalingRecommendation::ScaleDown;
                state.scale_down_counter = 0;
                state.last_scaling_action = Some(std::time::SystemTime::now());
            }
        } else {
            state.scale_up_counter = 0;
            state.scale_down_counter = 0;
            state.scaling_recommendation = ScalingRecommendation::NoAction;
        }
    }

    /// Get auto-scaling recommendation
    pub async fn get_scaling_recommendation(&self) -> ScalingRecommendation {
        let state = self.scaling_state.read().await;
        state.scaling_recommendation
    }

    /// Get auto-scaling state
    pub async fn get_scaling_state(&self) -> AutoScalingState {
        self.scaling_state.read().await.clone()
    }

    /// Start cluster management
    fn start_cluster_management(&self) {
        let cluster_state = Arc::clone(&self.cluster_state);
        let config = self.config.cluster_config.clone();
        let mut shutdown_receiver = self.shutdown_sender.subscribe();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        Self::perform_cluster_maintenance(
                            Arc::clone(&cluster_state),
                            &config,
                        ).await;
                    }
                    _ = shutdown_receiver.recv() => {
                        break;
                    }
                }
            }
        });
    }

    /// Perform cluster maintenance
    async fn perform_cluster_maintenance(
        cluster_state: Arc<RwLock<ClusterState>>,
        config: &ClusterConfig,
    ) {
        if !config.consensus_enabled {
            return;
        }

        let mut state = cluster_state.write().await;

        // Simple leader election logic (simplified Raft-like)
        if state.leader_id.is_none() {
            // Trigger election
            state.term += 1;
            state.is_leader = true; // In a real implementation, this would be voted on
            state.leader_id = Some(Uuid::new_v4());
            state.last_election = Some(std::time::SystemTime::now());
        }
    }

    /// Get cluster state
    pub async fn get_cluster_state(&self) -> ClusterState {
        self.cluster_state.read().await.clone()
    }

    /// Check if this node is the cluster leader
    pub async fn is_cluster_leader(&self) -> bool {
        let state = self.cluster_state.read().await;
        state.is_leader
    }

    /// Shutdown the distributed evaluator
    pub async fn shutdown(&self) -> EvaluationResult<()> {
        let _ = self.shutdown_sender.send(());
        Ok(())
    }
}

/// Create a new evaluation task
pub fn create_evaluation_task(
    task_type: TaskType,
    audio: &AudioBuffer,
    reference: Option<&AudioBuffer>,
    parameters: TaskParameters,
) -> EvaluationTask {
    EvaluationTask {
        id: Uuid::new_v4(),
        task_type,
        audio_data: serialize_audio_buffer(audio),
        reference_data: reference.map(serialize_audio_buffer),
        parameters,
        priority: TaskPriority::Normal,
        max_execution_time: Duration::from_secs(300),
        retry_count: 0,
    }
}

/// Serialize audio buffer for transmission
fn serialize_audio_buffer(audio: &AudioBuffer) -> Vec<u8> {
    // bincode 2 serde integration
    oxicode::serde::encode_to_vec(audio, oxicode::config::standard()).unwrap_or_default()
}

/// Deserialize audio buffer from transmission
pub fn deserialize_audio_buffer(data: &[u8]) -> Option<AudioBuffer> {
    oxicode::serde::decode_from_slice(data, oxicode::config::standard())
        .ok()
        .map(|(v, _)| v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use voirs_sdk::AudioBuffer;

    #[test]
    fn test_distributed_config_default() {
        let config = DistributedConfig::default();
        assert_eq!(config.max_workers, 10);
        assert_eq!(config.task_timeout_seconds, 300);
        assert!(matches!(
            config.load_balancing,
            LoadBalancingStrategy::RoundRobin
        ));
    }

    #[test]
    fn test_task_creation() {
        let samples = vec![0.1, 0.2, -0.1, -0.2];
        let audio = AudioBuffer::new(samples, 16000, 1);

        let parameters = TaskParameters {
            metrics: vec!["pesq".to_string(), "stoi".to_string()],
            language: Some("en".to_string()),
            sample_rate: Some(16000),
            channels: Some(1),
            custom_params: HashMap::new(),
        };

        let task = create_evaluation_task(TaskType::QualityMetrics, &audio, None, parameters);

        assert!(matches!(task.task_type, TaskType::QualityMetrics));
        assert_eq!(task.priority, TaskPriority::Normal);
        assert!(!task.audio_data.is_empty());
    }

    #[tokio::test]
    async fn test_distributed_evaluator_creation() {
        let config = DistributedConfig::default();
        let evaluator = DistributedEvaluator::new(config);

        let stats = evaluator.get_statistics().await;
        assert_eq!(stats.active_workers, 0);
        assert_eq!(stats.tasks_submitted, 0);
    }

    #[tokio::test]
    async fn test_worker_registration() {
        let config = DistributedConfig::default();
        let evaluator = DistributedEvaluator::new(config);

        let worker_info = WorkerInfo {
            id: Uuid::new_v4(),
            name: "test-worker".to_string(),
            capabilities: WorkerCapabilities {
                max_concurrent_tasks: 4,
                supported_task_types: vec![TaskType::QualityMetrics],
                available_memory: 1024.0,
                cpu_cores: 4,
                specialized_hardware: vec![],
            },
            status: WorkerStatus::Online,
            current_load: 0.0,
            last_heartbeat: std::time::SystemTime::now(),
            performance_metrics: PerformanceMetrics {
                tasks_completed: 0,
                tasks_failed: 0,
                avg_execution_time: Duration::from_secs(0),
                success_rate: 0.0,
                throughput: 0.0,
            },
            network_metrics: NetworkMetrics::default(),
            location: None,
            is_edge_node: false,
        };

        evaluator.register_worker(worker_info).await.unwrap();

        let workers = evaluator.get_workers().await;
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].name, "test-worker");
    }

    #[tokio::test]
    async fn test_task_submission() {
        let config = DistributedConfig::default();
        let evaluator = DistributedEvaluator::new(config);

        let samples = vec![0.1, 0.2, -0.1, -0.2];
        let audio = AudioBuffer::new(samples, 16000, 1);

        let parameters = TaskParameters {
            metrics: vec!["pesq".to_string()],
            language: Some("en".to_string()),
            sample_rate: Some(16000),
            channels: Some(1),
            custom_params: HashMap::new(),
        };

        let task = create_evaluation_task(TaskType::QualityMetrics, &audio, None, parameters);

        let task_id = evaluator.submit_task(task).await.unwrap();

        // Check that task was submitted
        let stats = evaluator.get_statistics().await;
        assert_eq!(stats.tasks_submitted, 1);

        // Task should be in queue but not completed yet (no workers)
        let result = evaluator.get_result(task_id).await.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_task_priority_ordering() {
        let high = TaskPriority::High;
        let low = TaskPriority::Low;
        let critical = TaskPriority::Critical;

        assert!(critical > high);
        assert!(high > low);
    }

    #[test]
    fn test_audio_serialization() {
        let samples = vec![0.1, 0.2, -0.1, -0.2];
        let audio = AudioBuffer::new(samples.clone(), 16000, 1);

        let serialized = serialize_audio_buffer(&audio);
        assert!(!serialized.is_empty());

        let deserialized = deserialize_audio_buffer(&serialized);
        assert!(deserialized.is_some());

        let restored_audio = deserialized.unwrap();
        assert_eq!(restored_audio.sample_rate(), 16000);
        assert_eq!(restored_audio.channels(), 1);
    }

    #[tokio::test]
    async fn test_auto_scaling_config() {
        let config = AutoScalingConfig::default();
        assert!(config.enabled);
        assert_eq!(config.target_cpu_utilization, 70.0);
        assert_eq!(config.target_memory_utilization, 80.0);
        assert_eq!(config.scale_up_threshold, 3);
        assert_eq!(config.scale_down_threshold, 5);
    }

    #[tokio::test]
    async fn test_auto_scaling_state() {
        let config = DistributedConfig {
            auto_scaling: AutoScalingConfig {
                enabled: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let evaluator = DistributedEvaluator::new(config);

        let state = evaluator.get_scaling_state().await;
        assert_eq!(state.scale_up_counter, 0);
        assert_eq!(state.scale_down_counter, 0);
        assert_eq!(
            state.scaling_recommendation,
            ScalingRecommendation::NoAction
        );
    }

    #[tokio::test]
    async fn test_cluster_state() {
        let config = DistributedConfig {
            cluster_config: ClusterConfig {
                enabled: true,
                consensus_enabled: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let evaluator = DistributedEvaluator::new(config);

        // Wait for cluster initialization
        tokio::time::sleep(Duration::from_millis(100)).await;

        let state = evaluator.get_cluster_state().await;
        // After initialization, the term should be >= 1 after leader election
        assert!(state.term >= 1);
        assert!(state.leader_id.is_some());
    }

    #[tokio::test]
    async fn test_edge_computing_config() {
        let config = EdgeComputingConfig::default();
        assert!(!config.enabled); // Disabled by default
        assert!(config.bandwidth_aware);
        assert!(config.edge_caching);
        assert_eq!(config.max_bandwidth_mbps, 100.0);
    }

    #[tokio::test]
    async fn test_network_metrics() {
        let metrics = NetworkMetrics::default();
        assert_eq!(metrics.latency_ms, 10.0);
        assert_eq!(metrics.bandwidth_mbps, 100.0);
        assert_eq!(metrics.packet_loss, 0.0);
    }

    #[tokio::test]
    async fn test_worker_location() {
        let location = WorkerLocation {
            region: "us-east-1".to_string(),
            zone: Some("us-east-1a".to_string()),
            latitude: Some(40.7128),
            longitude: Some(-74.0060),
        };

        assert_eq!(location.region, "us-east-1");
        assert_eq!(location.zone, Some("us-east-1a".to_string()));
    }

    #[tokio::test]
    async fn test_latency_aware_load_balancing() {
        let config = DistributedConfig {
            load_balancing: LoadBalancingStrategy::LatencyAware,
            ..Default::default()
        };
        let evaluator = DistributedEvaluator::new(config);

        // Create workers with different latencies
        let worker1 = WorkerInfo {
            id: Uuid::new_v4(),
            name: "high-latency".to_string(),
            capabilities: WorkerCapabilities {
                max_concurrent_tasks: 4,
                supported_task_types: vec![TaskType::QualityMetrics],
                available_memory: 1024.0,
                cpu_cores: 4,
                specialized_hardware: vec![],
            },
            status: WorkerStatus::Online,
            current_load: 0.5,
            last_heartbeat: std::time::SystemTime::now(),
            performance_metrics: PerformanceMetrics {
                tasks_completed: 10,
                tasks_failed: 0,
                avg_execution_time: Duration::from_secs(2),
                success_rate: 1.0,
                throughput: 5.0,
            },
            network_metrics: NetworkMetrics {
                latency_ms: 50.0,
                bandwidth_mbps: 100.0,
                packet_loss: 0.0,
                jitter_ms: 1.0,
                total_data_transferred_mb: 100.0,
            },
            location: None,
            is_edge_node: false,
        };

        let worker2 = WorkerInfo {
            id: Uuid::new_v4(),
            name: "low-latency".to_string(),
            capabilities: WorkerCapabilities {
                max_concurrent_tasks: 4,
                supported_task_types: vec![TaskType::QualityMetrics],
                available_memory: 1024.0,
                cpu_cores: 4,
                specialized_hardware: vec![],
            },
            status: WorkerStatus::Online,
            current_load: 0.5,
            last_heartbeat: std::time::SystemTime::now(),
            performance_metrics: PerformanceMetrics {
                tasks_completed: 10,
                tasks_failed: 0,
                avg_execution_time: Duration::from_secs(2),
                success_rate: 1.0,
                throughput: 5.0,
            },
            network_metrics: NetworkMetrics {
                latency_ms: 10.0, // Lower latency
                bandwidth_mbps: 100.0,
                packet_loss: 0.0,
                jitter_ms: 1.0,
                total_data_transferred_mb: 100.0,
            },
            location: None,
            is_edge_node: false,
        };

        evaluator.register_worker(worker1).await.unwrap();
        evaluator.register_worker(worker2).await.unwrap();

        let workers = evaluator.get_workers().await;
        assert_eq!(workers.len(), 2);
    }

    #[tokio::test]
    async fn test_bandwidth_aware_load_balancing() {
        let config = DistributedConfig {
            load_balancing: LoadBalancingStrategy::BandwidthAware,
            ..Default::default()
        };
        let evaluator = DistributedEvaluator::new(config);

        let worker = WorkerInfo {
            id: Uuid::new_v4(),
            name: "high-bandwidth".to_string(),
            capabilities: WorkerCapabilities {
                max_concurrent_tasks: 4,
                supported_task_types: vec![TaskType::QualityMetrics],
                available_memory: 1024.0,
                cpu_cores: 4,
                specialized_hardware: vec![],
            },
            status: WorkerStatus::Online,
            current_load: 0.5,
            last_heartbeat: std::time::SystemTime::now(),
            performance_metrics: PerformanceMetrics {
                tasks_completed: 10,
                tasks_failed: 0,
                avg_execution_time: Duration::from_secs(2),
                success_rate: 1.0,
                throughput: 5.0,
            },
            network_metrics: NetworkMetrics {
                latency_ms: 10.0,
                bandwidth_mbps: 1000.0, // High bandwidth
                packet_loss: 0.0,
                jitter_ms: 1.0,
                total_data_transferred_mb: 100.0,
            },
            location: None,
            is_edge_node: false,
        };

        evaluator.register_worker(worker).await.unwrap();

        let workers = evaluator.get_workers().await;
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].network_metrics.bandwidth_mbps, 1000.0);
    }

    #[tokio::test]
    async fn test_adaptive_load_balancing() {
        let config = DistributedConfig {
            load_balancing: LoadBalancingStrategy::Adaptive,
            ..Default::default()
        };
        let evaluator = DistributedEvaluator::new(config);

        let worker = WorkerInfo {
            id: Uuid::new_v4(),
            name: "adaptive-worker".to_string(),
            capabilities: WorkerCapabilities {
                max_concurrent_tasks: 4,
                supported_task_types: vec![TaskType::QualityMetrics],
                available_memory: 1024.0,
                cpu_cores: 4,
                specialized_hardware: vec![],
            },
            status: WorkerStatus::Online,
            current_load: 0.3,
            last_heartbeat: std::time::SystemTime::now(),
            performance_metrics: PerformanceMetrics {
                tasks_completed: 100,
                tasks_failed: 2,
                avg_execution_time: Duration::from_secs(1),
                success_rate: 0.98,
                throughput: 10.0,
            },
            network_metrics: NetworkMetrics {
                latency_ms: 15.0,
                bandwidth_mbps: 500.0,
                packet_loss: 0.1,
                jitter_ms: 2.0,
                total_data_transferred_mb: 1000.0,
            },
            location: Some(WorkerLocation {
                region: "us-west-2".to_string(),
                zone: Some("us-west-2a".to_string()),
                latitude: Some(45.5231),
                longitude: Some(-122.6765),
            }),
            is_edge_node: true,
        };

        evaluator.register_worker(worker).await.unwrap();

        let workers = evaluator.get_workers().await;
        assert_eq!(workers.len(), 1);
        assert!(workers[0].is_edge_node);
    }

    #[test]
    fn test_scaling_recommendation() {
        assert_eq!(
            ScalingRecommendation::default(),
            ScalingRecommendation::NoAction
        );
    }

    #[test]
    fn test_consensus_algorithm() {
        let config = ClusterConfig::default();
        assert!(matches!(
            config.consensus_algorithm,
            ConsensusAlgorithm::Raft
        ));
    }

    #[test]
    fn test_cluster_discovery_method() {
        let config = ClusterConfig::default();
        assert!(matches!(
            config.discovery_method,
            ClusterDiscoveryMethod::Static
        ));
    }
}
