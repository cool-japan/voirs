//! Distributed evaluation system for parallel processing across multiple nodes
//!
//! This module provides a framework for distributing evaluation tasks across multiple
//! computing nodes to improve performance and scalability for large-scale evaluations.

use crate::traits::{EvaluationResult, QualityScore};
use crate::EvaluationError;
use rand::prelude::*;
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
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            max_workers: 10,
            task_timeout_seconds: 300,
            heartbeat_interval_seconds: 30,
            max_retries: 3,
            load_balancing: LoadBalancingStrategy::RoundRobin,
            fault_tolerance: FaultToleranceConfig::default(),
        }
    }
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
            config,
            workers: Arc::new(RwLock::new(HashMap::new())),
            task_queue: Arc::new(RwLock::new(Vec::new())),
            running_tasks: Arc::new(RwLock::new(HashMap::new())),
            completed_tasks: Arc::new(RwLock::new(HashMap::new())),
            task_sender,
            result_receiver: Arc::new(RwLock::new(result_receiver)),
            shutdown_sender,
            stats: Arc::new(RwLock::new(SystemStatistics::default())),
        };

        // Start background task management
        evaluator.start_background_tasks(task_receiver, result_sender);

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
        queue.sort_by(|a, b| b.priority.cmp(&a.priority));

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
                    .min_by(|a, b| a.current_load.partial_cmp(&b.current_load).unwrap())
                    .map(|w| w.id)
            }
            LoadBalancingStrategy::Random => {
                // Random selection
                available_workers
                    .choose(&mut rand::thread_rng())
                    .map(|w| w.id)
            }
            LoadBalancingStrategy::Weighted => {
                // Weighted by capabilities (simplified)
                available_workers
                    .iter()
                    .max_by_key(|w| w.capabilities.max_concurrent_tasks)
                    .map(|w| w.id)
            }
        }
    }

    /// Execute task on a worker (simulated)
    async fn execute_task_on_worker(task: EvaluationTask, worker_id: WorkerId) -> TaskResult {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let mut rng = StdRng::seed_from_u64(seed);
        let start_time = std::time::SystemTime::now();

        // Simulate task execution time
        let execution_time = Duration::from_millis(rng.r#gen::<u64>() % 5000 + 1000);
        tokio::time::sleep(execution_time).await;

        // Simulate task result
        let result = if rng.r#gen::<f32>() > 0.1 {
            // 90% success rate
            Ok(EvaluationOutput {
                quality_scores: {
                    let mut scores = HashMap::new();
                    scores.insert("pesq".to_string(), rng.r#gen::<f32>() * 5.0);
                    scores.insert("stoi".to_string(), rng.r#gen::<f32>());
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
                cpu_usage: rng.r#gen::<f32>() * 100.0,
                memory_usage: rng.r#gen::<f32>() * 1024.0,
                disk_io: rng.r#gen::<f32>() * 100.0,
                network_io: rng.r#gen::<f32>() * 50.0,
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
    // Simplified serialization - in real implementation would use proper serialization
    bincode::serialize(audio).unwrap_or_default()
}

/// Deserialize audio buffer from transmission
pub fn deserialize_audio_buffer(data: &[u8]) -> Option<AudioBuffer> {
    bincode::deserialize(data).ok()
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
}
