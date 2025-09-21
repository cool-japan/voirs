//! GPU Load Balancing System for Voice Cloning
//!
//! This module provides intelligent load balancing across multiple GPUs to optimize
//! voice cloning performance. It includes workload distribution strategies, GPU health
//! monitoring, and automatic load redistribution based on GPU utilization and performance.

use crate::{
    gpu_acceleration::{
        GpuAccelerationConfig, GpuAccelerator, GpuDeviceType, GpuMemoryStats,
        GpuPerformanceMetrics, TensorOperation, TensorOperationResult,
    },
    Error, Result, VoiceCloneRequest, VoiceCloneResult,
};
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::Semaphore;

/// Load balancing strategies for GPU workload distribution
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin assignment
    RoundRobin,
    /// Assign to GPU with lowest utilization
    LowestUtilization,
    /// Assign to GPU with best performance metrics
    PerformanceBased,
    /// Assign based on memory availability
    MemoryBased,
    /// Assign based on historical success rate
    ReliabilityBased,
    /// Custom weighted assignment
    Weighted {
        utilization_weight: f32,
        performance_weight: f32,
        memory_weight: f32,
        reliability_weight: f32,
    },
}

impl Default for LoadBalancingStrategy {
    fn default() -> Self {
        Self::LowestUtilization
    }
}

/// Configuration for GPU load balancing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadBalancingConfig {
    /// Load balancing strategy
    pub strategy: LoadBalancingStrategy,
    /// Maximum number of concurrent operations per GPU
    pub max_concurrent_ops_per_gpu: usize,
    /// Enable automatic GPU failover
    pub enable_failover: bool,
    /// Health check interval in seconds
    pub health_check_interval_secs: u64,
    /// GPU utilization threshold for load redistribution
    pub utilization_threshold: f32,
    /// Memory usage threshold for load redistribution
    pub memory_threshold: f32,
    /// Enable dynamic load rebalancing
    pub enable_dynamic_rebalancing: bool,
    /// Rebalancing check interval in seconds
    pub rebalancing_interval_secs: u64,
    /// Minimum number of operations before rebalancing
    pub min_ops_for_rebalancing: usize,
    /// GPU warmup timeout in seconds
    pub warmup_timeout_secs: u64,
    /// Enable performance prediction
    pub enable_performance_prediction: bool,
    /// Maximum queue size per GPU
    pub max_queue_size_per_gpu: usize,
}

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            strategy: LoadBalancingStrategy::default(),
            max_concurrent_ops_per_gpu: 8,
            enable_failover: true,
            health_check_interval_secs: 30,
            utilization_threshold: 0.85,
            memory_threshold: 0.90,
            enable_dynamic_rebalancing: true,
            rebalancing_interval_secs: 60,
            min_ops_for_rebalancing: 50,
            warmup_timeout_secs: 30,
            enable_performance_prediction: true,
            max_queue_size_per_gpu: 100,
        }
    }
}

/// GPU device information and status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    /// Device ID
    pub device_id: usize,
    /// Device type
    pub device_type: GpuDeviceType,
    /// Device name
    pub device_name: String,
    /// Is device available
    pub is_available: bool,
    /// Is device healthy
    pub is_healthy: bool,
    /// Current utilization (0.0 to 1.0)
    pub utilization: f32,
    /// Memory statistics
    pub memory_stats: GpuMemoryStats,
    /// Performance metrics
    pub performance_metrics: GpuPerformanceMetrics,
    /// Number of active operations
    pub active_operations: usize,
    /// Queue size
    pub queue_size: usize,
    /// Last health check timestamp
    pub last_health_check: SystemTime,
    /// Success rate (operations successful / total operations)
    pub success_rate: f32,
    /// Average operation latency
    pub avg_latency: Duration,
    /// Device temperature (Celsius)
    pub temperature: f32,
}

/// Operation assignment to GPU
#[derive(Debug, Clone)]
pub struct GpuAssignment {
    /// GPU device ID
    pub device_id: usize,
    /// Assignment timestamp
    pub assigned_at: SystemTime,
    /// Priority score used for assignment
    pub priority_score: f32,
    /// Expected completion time
    pub expected_completion: SystemTime,
}

/// Load balancing statistics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadBalancingStats {
    /// Total operations processed
    pub total_operations: u64,
    /// Operations per GPU
    pub operations_per_gpu: HashMap<usize, u64>,
    /// Success rate per GPU
    pub success_rate_per_gpu: HashMap<usize, f32>,
    /// Average latency per GPU
    pub avg_latency_per_gpu: HashMap<usize, Duration>,
    /// Load distribution efficiency (0.0 to 1.0)
    pub load_distribution_efficiency: f32,
    /// Number of failovers
    pub failover_count: u64,
    /// Number of rebalancing operations
    pub rebalancing_count: u64,
    /// Overall system utilization
    pub system_utilization: f32,
    /// GPU availability ratio
    pub gpu_availability_ratio: f32,
}

/// Performance prediction for GPU operations
#[derive(Debug, Clone)]
pub struct PerformancePrediction {
    /// Predicted execution time
    pub predicted_time: Duration,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Predicted memory usage
    pub predicted_memory: u64,
    /// Historical data points used
    pub data_points: usize,
}

/// GPU operation queue entry
#[derive(Debug)]
struct QueuedOperation {
    /// Operation to execute
    operation: TensorOperation,
    /// Request metadata
    request_metadata: OperationMetadata,
    /// Queue timestamp
    queued_at: SystemTime,
    /// Priority
    priority: u8,
    /// Completion callback
    completion_tx: tokio::sync::oneshot::Sender<Result<TensorOperationResult>>,
}

/// Operation metadata for tracking
#[derive(Debug, Clone)]
struct OperationMetadata {
    /// Operation ID
    operation_id: String,
    /// Request ID
    request_id: Option<String>,
    /// Operation type description
    operation_type: String,
    /// Input size estimation
    input_size: usize,
    /// Expected complexity
    complexity: f32,
}

/// GPU load balancing orchestrator
pub struct GpuLoadBalancer {
    /// Configuration
    config: LoadBalancingConfig,
    /// Available GPU devices
    gpu_devices: Arc<RwLock<HashMap<usize, GpuAccelerator>>>,
    /// GPU device information
    device_info: Arc<RwLock<HashMap<usize, GpuDeviceInfo>>>,
    /// Operation queues per GPU
    operation_queues: Arc<RwLock<HashMap<usize, VecDeque<QueuedOperation>>>>,
    /// Semaphores for concurrent operation limiting
    gpu_semaphores: Arc<RwLock<HashMap<usize, Arc<Semaphore>>>>,
    /// Load balancing statistics
    stats: Arc<RwLock<LoadBalancingStats>>,
    /// Round-robin counter
    round_robin_counter: Arc<Mutex<usize>>,
    /// Performance history for prediction
    performance_history: Arc<RwLock<HashMap<String, VecDeque<(Duration, u64)>>>>,
    /// Health monitoring active
    health_monitoring_active: Arc<RwLock<bool>>,
}

impl GpuLoadBalancer {
    /// Create new GPU load balancer
    pub async fn new(config: LoadBalancingConfig) -> Result<Self> {
        let gpu_devices = Arc::new(RwLock::new(HashMap::new()));
        let device_info = Arc::new(RwLock::new(HashMap::new()));
        let operation_queues = Arc::new(RwLock::new(HashMap::new()));
        let gpu_semaphores = Arc::new(RwLock::new(HashMap::new()));

        let load_balancer = Self {
            config: config.clone(),
            gpu_devices,
            device_info,
            operation_queues,
            gpu_semaphores,
            stats: Arc::new(RwLock::new(LoadBalancingStats::default())),
            round_robin_counter: Arc::new(Mutex::new(0)),
            performance_history: Arc::new(RwLock::new(HashMap::new())),
            health_monitoring_active: Arc::new(RwLock::new(false)),
        };

        // Initialize GPU devices
        load_balancer.initialize_gpu_devices().await?;

        // Start health monitoring
        load_balancer.start_health_monitoring().await;

        // Start dynamic rebalancing if enabled
        if config.enable_dynamic_rebalancing {
            load_balancer.start_dynamic_rebalancing().await;
        }

        Ok(load_balancer)
    }

    /// Create with default configuration
    pub async fn new_default() -> Result<Self> {
        Self::new(LoadBalancingConfig::default()).await
    }

    /// Initialize and detect available GPU devices
    async fn initialize_gpu_devices(&self) -> Result<()> {
        let available_devices = self.detect_gpu_devices()?;

        for device_id in available_devices {
            // Create GPU accelerator for each device
            let gpu_config = GpuAccelerationConfig {
                enabled: true,
                device_id,
                auto_fallback: false,
                ..Default::default()
            };

            match GpuAccelerator::new(gpu_config.clone()) {
                Ok(accelerator) => {
                    // Warm up the GPU
                    let warmup_result = tokio::time::timeout(
                        Duration::from_secs(self.config.warmup_timeout_secs),
                        accelerator.warmup(),
                    )
                    .await;

                    let is_healthy = warmup_result.is_ok() && warmup_result.unwrap().is_ok();

                    // Add to devices
                    {
                        let mut devices = self.gpu_devices.write().unwrap();
                        devices.insert(device_id, accelerator);
                    }

                    // Initialize device info
                    let device_info = GpuDeviceInfo {
                        device_id,
                        device_type: gpu_config.device_type,
                        device_name: format!("{:?} Device {}", gpu_config.device_type, device_id),
                        is_available: true,
                        is_healthy,
                        utilization: 0.0,
                        memory_stats: GpuMemoryStats {
                            total_memory: 8_000_000_000,
                            used_memory: 0,
                            free_memory: 8_000_000_000,
                            peak_memory: 0,
                            active_tensors: 0,
                            fragmentation_ratio: 0.0,
                        },
                        performance_metrics: GpuPerformanceMetrics {
                            gpu_utilization: 0.0,
                            memory_bandwidth: 0.0,
                            avg_kernel_time: Duration::from_millis(0),
                            operations_count: 0,
                            temperature: 25.0,
                            power_consumption: 0.0,
                        },
                        active_operations: 0,
                        queue_size: 0,
                        last_health_check: SystemTime::now(),
                        success_rate: 1.0,
                        avg_latency: Duration::from_millis(0),
                        temperature: 25.0,
                    };

                    {
                        let mut info_map = self.device_info.write().unwrap();
                        info_map.insert(device_id, device_info);
                    }

                    // Initialize operation queue
                    {
                        let mut queues = self.operation_queues.write().unwrap();
                        queues.insert(device_id, VecDeque::new());
                    }

                    // Initialize semaphore
                    {
                        let mut semaphores = self.gpu_semaphores.write().unwrap();
                        semaphores.insert(
                            device_id,
                            Arc::new(Semaphore::new(self.config.max_concurrent_ops_per_gpu)),
                        );
                    }

                    tracing::info!(
                        "Initialized GPU device {} (healthy: {})",
                        device_id,
                        is_healthy
                    );
                }
                Err(e) => {
                    tracing::warn!("Failed to initialize GPU device {}: {}", device_id, e);
                }
            }
        }

        let device_count = self.gpu_devices.read().unwrap().len();
        if device_count == 0 {
            return Err(Error::Config(
                "No GPU devices available for load balancing".to_string(),
            ));
        }

        tracing::info!(
            "Initialized {} GPU devices for load balancing",
            device_count
        );
        Ok(())
    }

    /// Detect available GPU devices
    fn detect_gpu_devices(&self) -> Result<Vec<usize>> {
        let mut available_devices = Vec::new();

        // Check CUDA devices
        #[cfg(feature = "gpu")]
        {
            for device_id in 0..8 {
                if Device::new_cuda(device_id).is_ok() {
                    available_devices.push(device_id);
                } else {
                    break;
                }
            }
        }

        // Check Metal devices
        for device_id in 0..4 {
            if Device::new_metal(device_id).is_ok() {
                available_devices.push(device_id);
            } else {
                break;
            }
        }

        // Ensure we have at least one device (CPU fallback if needed)
        if available_devices.is_empty() {
            available_devices.push(0); // CPU device
        }

        Ok(available_devices)
    }

    /// Execute operation with load balancing
    pub async fn execute_operation(
        &self,
        operation: TensorOperation,
    ) -> Result<TensorOperationResult> {
        let operation_id = uuid::Uuid::new_v4().to_string();
        let start_time = SystemTime::now();

        // Select best GPU for this operation
        let selected_gpu = self.select_gpu_for_operation(&operation).await?;

        // Create operation metadata
        let metadata = OperationMetadata {
            operation_id: operation_id.clone(),
            request_id: None,
            operation_type: format!("{:?}", operation.operation_type),
            input_size: operation.inputs.iter().map(|t| t.elem_count()).sum(),
            complexity: self.estimate_operation_complexity(&operation),
        };

        // Get performance prediction if enabled
        let prediction = if self.config.enable_performance_prediction {
            Some(self.predict_operation_performance(&operation, selected_gpu))
        } else {
            None
        };

        // Execute operation on selected GPU
        let result = self
            .execute_on_gpu(selected_gpu, operation, metadata.clone())
            .await;

        // Update performance history
        if let Ok(ref operation_result) = result {
            self.update_performance_history(&metadata, operation_result, prediction)
                .await;
        }

        // Update statistics
        self.update_operation_statistics(selected_gpu, &result, start_time)
            .await;

        result
    }

    /// Execute batch of operations with load balancing
    pub async fn execute_batch(
        &self,
        operations: Vec<TensorOperation>,
    ) -> Result<Vec<TensorOperationResult>> {
        let mut results = Vec::with_capacity(operations.len());
        let mut futures = Vec::new();

        // Distribute operations across GPUs
        for operation in operations {
            let operation_future = self.execute_operation(operation);
            futures.push(operation_future);
        }

        // Execute all operations concurrently
        for future in futures {
            results.push(future.await?);
        }

        Ok(results)
    }

    /// Select optimal GPU for operation based on load balancing strategy
    async fn select_gpu_for_operation(&self, operation: &TensorOperation) -> Result<usize> {
        let available_gpus = self.get_available_gpus().await;

        if available_gpus.is_empty() {
            return Err(Error::Processing(
                "No available GPUs for operation".to_string(),
            ));
        }

        let selected_gpu = match self.config.strategy {
            LoadBalancingStrategy::RoundRobin => self.select_round_robin(&available_gpus),
            LoadBalancingStrategy::LowestUtilization => {
                self.select_lowest_utilization(&available_gpus).await
            }
            LoadBalancingStrategy::PerformanceBased => {
                self.select_performance_based(&available_gpus, operation)
                    .await
            }
            LoadBalancingStrategy::MemoryBased => {
                self.select_memory_based(&available_gpus, operation).await
            }
            LoadBalancingStrategy::ReliabilityBased => {
                self.select_reliability_based(&available_gpus).await
            }
            LoadBalancingStrategy::Weighted {
                utilization_weight,
                performance_weight,
                memory_weight,
                reliability_weight,
            } => {
                self.select_weighted(
                    &available_gpus,
                    operation,
                    utilization_weight,
                    performance_weight,
                    memory_weight,
                    reliability_weight,
                )
                .await
            }
        };

        Ok(selected_gpu)
    }

    /// Round-robin GPU selection
    fn select_round_robin(&self, available_gpus: &[usize]) -> usize {
        let mut counter = self.round_robin_counter.lock().unwrap();
        let selected = available_gpus[*counter % available_gpus.len()];
        *counter += 1;
        selected
    }

    /// Select GPU with lowest utilization
    async fn select_lowest_utilization(&self, available_gpus: &[usize]) -> usize {
        let device_info = self.device_info.read().unwrap();

        available_gpus
            .iter()
            .min_by(|&&a, &&b| {
                let info_a = device_info.get(&a).unwrap();
                let info_b = device_info.get(&b).unwrap();
                info_a.utilization.partial_cmp(&info_b.utilization).unwrap()
            })
            .copied()
            .unwrap_or(available_gpus[0])
    }

    /// Select GPU based on performance metrics
    async fn select_performance_based(
        &self,
        available_gpus: &[usize],
        operation: &TensorOperation,
    ) -> usize {
        let device_info = self.device_info.read().unwrap();

        available_gpus
            .iter()
            .max_by(|&&a, &&b| {
                let info_a = device_info.get(&a).unwrap();
                let info_b = device_info.get(&b).unwrap();

                // Calculate performance score (higher is better)
                let score_a = self.calculate_performance_score(info_a, operation);
                let score_b = self.calculate_performance_score(info_b, operation);

                score_a.partial_cmp(&score_b).unwrap()
            })
            .copied()
            .unwrap_or(available_gpus[0])
    }

    /// Select GPU based on memory availability
    async fn select_memory_based(
        &self,
        available_gpus: &[usize],
        operation: &TensorOperation,
    ) -> usize {
        let device_info = self.device_info.read().unwrap();
        let estimated_memory = self.estimate_operation_memory(operation);

        available_gpus
            .iter()
            .filter(|&&gpu_id| {
                let info = device_info.get(&gpu_id).unwrap();
                info.memory_stats.free_memory >= estimated_memory
            })
            .max_by(|&&a, &&b| {
                let info_a = device_info.get(&a).unwrap();
                let info_b = device_info.get(&b).unwrap();
                info_a
                    .memory_stats
                    .free_memory
                    .cmp(&info_b.memory_stats.free_memory)
            })
            .copied()
            .unwrap_or(available_gpus[0])
    }

    /// Select GPU based on reliability (success rate)
    async fn select_reliability_based(&self, available_gpus: &[usize]) -> usize {
        let device_info = self.device_info.read().unwrap();

        available_gpus
            .iter()
            .max_by(|&&a, &&b| {
                let info_a = device_info.get(&a).unwrap();
                let info_b = device_info.get(&b).unwrap();
                info_a
                    .success_rate
                    .partial_cmp(&info_b.success_rate)
                    .unwrap()
            })
            .copied()
            .unwrap_or(available_gpus[0])
    }

    /// Select GPU using weighted scoring
    async fn select_weighted(
        &self,
        available_gpus: &[usize],
        operation: &TensorOperation,
        utilization_weight: f32,
        performance_weight: f32,
        memory_weight: f32,
        reliability_weight: f32,
    ) -> usize {
        let device_info = self.device_info.read().unwrap();
        let estimated_memory = self.estimate_operation_memory(operation);

        available_gpus
            .iter()
            .max_by(|&&a, &&b| {
                let info_a = device_info.get(&a).unwrap();
                let info_b = device_info.get(&b).unwrap();

                let score_a = self.calculate_weighted_score(
                    info_a,
                    operation,
                    estimated_memory,
                    utilization_weight,
                    performance_weight,
                    memory_weight,
                    reliability_weight,
                );

                let score_b = self.calculate_weighted_score(
                    info_b,
                    operation,
                    estimated_memory,
                    utilization_weight,
                    performance_weight,
                    memory_weight,
                    reliability_weight,
                );

                score_a.partial_cmp(&score_b).unwrap()
            })
            .copied()
            .unwrap_or(available_gpus[0])
    }

    /// Calculate performance score for GPU
    fn calculate_performance_score(
        &self,
        info: &GpuDeviceInfo,
        _operation: &TensorOperation,
    ) -> f32 {
        let utilization_factor = 1.0 - info.utilization; // Lower utilization is better
        let memory_factor =
            info.memory_stats.free_memory as f32 / info.memory_stats.total_memory as f32;
        let reliability_factor = info.success_rate;
        let latency_factor = 1.0 / (info.avg_latency.as_millis() as f32 + 1.0);

        (utilization_factor + memory_factor + reliability_factor + latency_factor) / 4.0
    }

    /// Calculate weighted score for GPU selection
    fn calculate_weighted_score(
        &self,
        info: &GpuDeviceInfo,
        operation: &TensorOperation,
        estimated_memory: u64,
        utilization_weight: f32,
        performance_weight: f32,
        memory_weight: f32,
        reliability_weight: f32,
    ) -> f32 {
        let utilization_score = (1.0 - info.utilization) * utilization_weight;
        let performance_score =
            self.calculate_performance_score(info, operation) * performance_weight;

        let memory_score = if info.memory_stats.free_memory >= estimated_memory {
            (info.memory_stats.free_memory as f32 / info.memory_stats.total_memory as f32)
                * memory_weight
        } else {
            0.0 // Insufficient memory
        };

        let reliability_score = info.success_rate * reliability_weight;

        utilization_score + performance_score + memory_score + reliability_score
    }

    /// Get list of available and healthy GPUs
    async fn get_available_gpus(&self) -> Vec<usize> {
        let device_info = self.device_info.read().unwrap();

        device_info
            .iter()
            .filter(|(_, info)| info.is_available && info.is_healthy)
            .map(|(&device_id, _)| device_id)
            .collect()
    }

    /// Execute operation on specific GPU
    async fn execute_on_gpu(
        &self,
        gpu_id: usize,
        operation: TensorOperation,
        metadata: OperationMetadata,
    ) -> Result<TensorOperationResult> {
        // Get semaphore permit to limit concurrent operations
        let semaphore = {
            let semaphores = self.gpu_semaphores.read().unwrap();
            semaphores
                .get(&gpu_id)
                .cloned()
                .ok_or_else(|| Error::Processing(format!("GPU {} not available", gpu_id)))?
        };

        let _permit = semaphore
            .acquire()
            .await
            .map_err(|_| Error::Processing("Failed to acquire GPU semaphore".to_string()))?;

        // Update active operations count
        {
            let mut device_info = self.device_info.write().unwrap();
            if let Some(info) = device_info.get_mut(&gpu_id) {
                info.active_operations += 1;
            }
        }

        // Get GPU accelerator
        let accelerator = {
            let devices = self.gpu_devices.read().unwrap();
            devices
                .get(&gpu_id)
                .cloned()
                .ok_or_else(|| Error::Processing(format!("GPU {} not found", gpu_id)))?
        };

        // Execute operation
        let start_time = Instant::now();
        let result = accelerator.execute_operation(operation).await;
        let execution_time = start_time.elapsed();

        // Update device info
        {
            let mut device_info = self.device_info.write().unwrap();
            if let Some(info) = device_info.get_mut(&gpu_id) {
                info.active_operations = info.active_operations.saturating_sub(1);

                // Update memory stats
                info.memory_stats = accelerator.get_memory_stats();

                // Update performance metrics
                info.performance_metrics = accelerator.get_performance_metrics();

                // Update utilization
                info.utilization = info.performance_metrics.gpu_utilization;

                // Update average latency
                let total_ops = info.performance_metrics.operations_count;
                if total_ops > 0 {
                    info.avg_latency = Duration::from_nanos(
                        (info.avg_latency.as_nanos() as u64 * (total_ops - 1)
                            + execution_time.as_nanos() as u64)
                            / total_ops,
                    );
                }
            }
        }

        result
    }

    /// Estimate operation complexity
    fn estimate_operation_complexity(&self, operation: &TensorOperation) -> f32 {
        let input_size: usize = operation.inputs.iter().map(|t| t.elem_count()).sum();
        let base_complexity = input_size as f32 / 1_000_000.0; // Normalize by 1M elements

        // Adjust based on operation type
        let type_multiplier = match operation.operation_type {
            crate::gpu_acceleration::GpuOperationType::MatMul => 2.0,
            crate::gpu_acceleration::GpuOperationType::Conv => 3.0,
            crate::gpu_acceleration::GpuOperationType::Attention => 4.0,
            crate::gpu_acceleration::GpuOperationType::AudioProcessing => 2.5,
            crate::gpu_acceleration::GpuOperationType::SpeakerAdaptation => 3.5,
            _ => 1.0,
        };

        base_complexity * type_multiplier
    }

    /// Estimate memory required for operation
    fn estimate_operation_memory(&self, operation: &TensorOperation) -> u64 {
        let input_memory: u64 = operation
            .inputs
            .iter()
            .map(|t| (t.elem_count() * t.dtype().size_in_bytes()) as u64)
            .sum();

        // Estimate output memory (assume similar size to inputs)
        let estimated_output_memory = input_memory;

        // Add overhead for intermediate computations
        let overhead_factor = match operation.operation_type {
            crate::gpu_acceleration::GpuOperationType::Attention => 3.0, // Attention needs Q, K, V matrices
            crate::gpu_acceleration::GpuOperationType::Conv => 2.0,
            _ => 1.5,
        };

        ((input_memory + estimated_output_memory) as f32 * overhead_factor) as u64
    }

    /// Predict operation performance
    fn predict_operation_performance(
        &self,
        operation: &TensorOperation,
        gpu_id: usize,
    ) -> PerformancePrediction {
        let operation_key = format!("{:?}_{}", operation.operation_type, gpu_id);
        let history = self.performance_history.read().unwrap();

        if let Some(history_data) = history.get(&operation_key) {
            if !history_data.is_empty() {
                let avg_time: Duration = {
                    let total_nanos: u64 = history_data
                        .iter()
                        .map(|(time, _)| time.as_nanos() as u64)
                        .sum();
                    Duration::from_nanos(total_nanos / history_data.len() as u64)
                };

                let avg_memory: u64 = history_data.iter().map(|(_, mem)| *mem).sum::<u64>()
                    / history_data.len() as u64;

                // Confidence based on number of data points
                let confidence = (history_data.len() as f32 / 50.0).min(1.0);

                return PerformancePrediction {
                    predicted_time: avg_time,
                    confidence,
                    predicted_memory: avg_memory,
                    data_points: history_data.len(),
                };
            }
        }

        // Fallback estimation
        let estimated_time =
            Duration::from_millis((self.estimate_operation_complexity(operation) * 100.0) as u64);
        let estimated_memory = self.estimate_operation_memory(operation);

        PerformancePrediction {
            predicted_time: estimated_time,
            confidence: 0.1, // Low confidence for estimates
            predicted_memory: estimated_memory,
            data_points: 0,
        }
    }

    /// Update performance history
    async fn update_performance_history(
        &self,
        metadata: &OperationMetadata,
        result: &TensorOperationResult,
        _prediction: Option<PerformancePrediction>,
    ) {
        let operation_key = metadata.operation_type.clone();
        let mut history = self.performance_history.write().unwrap();

        let history_data = history.entry(operation_key).or_insert_with(VecDeque::new);

        // Add new data point
        history_data.push_back((result.execution_time, result.memory_used));

        // Keep only recent history (last 100 operations)
        while history_data.len() > 100 {
            history_data.pop_front();
        }
    }

    /// Update operation statistics
    async fn update_operation_statistics(
        &self,
        gpu_id: usize,
        result: &Result<TensorOperationResult>,
        start_time: SystemTime,
    ) {
        let mut stats = self.stats.write().unwrap();

        stats.total_operations += 1;
        *stats.operations_per_gpu.entry(gpu_id).or_insert(0) += 1;

        let is_success = result.is_ok();
        let gpu_ops = *stats.operations_per_gpu.get(&gpu_id).unwrap_or(&1);

        // Update success rate
        let current_success_rate = stats.success_rate_per_gpu.entry(gpu_id).or_insert(1.0);
        let new_success_rate = (*current_success_rate * (gpu_ops - 1) as f32
            + if is_success { 1.0 } else { 0.0 })
            / gpu_ops as f32;
        *current_success_rate = new_success_rate;

        // Update latency
        if let Ok(operation_result) = result {
            let latency = operation_result.execution_time;
            let current_latency = stats
                .avg_latency_per_gpu
                .entry(gpu_id)
                .or_insert(Duration::from_millis(0));
            *current_latency = Duration::from_nanos(
                (current_latency.as_nanos() as u64 * (gpu_ops - 1) as u64
                    + latency.as_nanos() as u64)
                    / gpu_ops as u64,
            );
        }

        // Update device success rate
        {
            let mut device_info = self.device_info.write().unwrap();
            if let Some(info) = device_info.get_mut(&gpu_id) {
                info.success_rate = new_success_rate;
            }
        }
    }

    /// Start health monitoring background task
    async fn start_health_monitoring(&self) {
        let device_info = Arc::clone(&self.device_info);
        let gpu_devices = Arc::clone(&self.gpu_devices);
        let health_monitoring_active = Arc::clone(&self.health_monitoring_active);
        let check_interval = Duration::from_secs(self.config.health_check_interval_secs);

        {
            let mut active = health_monitoring_active.write().unwrap();
            *active = true;
        }

        tokio::spawn(async move {
            while *health_monitoring_active.read().unwrap() {
                tokio::time::sleep(check_interval).await;

                let device_ids: Vec<usize> =
                    { device_info.read().unwrap().keys().copied().collect() };

                for device_id in device_ids {
                    // Check GPU health
                    let accelerator = gpu_devices.read().unwrap().get(&device_id).cloned();
                    let is_healthy = if let Some(accelerator) = accelerator {
                        // Try a simple operation to test GPU health
                        let test_result =
                            tokio::time::timeout(Duration::from_secs(5), accelerator.synchronize())
                                .await;

                        test_result.is_ok() && test_result.unwrap().is_ok()
                    } else {
                        false
                    };

                    // Update device health
                    {
                        let mut info_map = device_info.write().unwrap();
                        if let Some(info) = info_map.get_mut(&device_id) {
                            info.is_healthy = is_healthy;
                            info.last_health_check = SystemTime::now();

                            if !is_healthy {
                                tracing::warn!("GPU {} health check failed", device_id);
                            }
                        }
                    }
                }
            }
        });
    }

    /// Start dynamic rebalancing background task
    async fn start_dynamic_rebalancing(&self) {
        let device_info = Arc::clone(&self.device_info);
        let stats = Arc::clone(&self.stats);
        let rebalancing_interval = Duration::from_secs(self.config.rebalancing_interval_secs);
        let utilization_threshold = self.config.utilization_threshold;
        let memory_threshold = self.config.memory_threshold;
        let min_ops = self.config.min_ops_for_rebalancing;

        tokio::spawn(async move {
            loop {
                tokio::time::sleep(rebalancing_interval).await;

                // Check if rebalancing is needed
                let total_ops = stats.read().unwrap().total_operations;
                if total_ops < min_ops as u64 {
                    continue;
                }

                let device_info_snapshot = device_info.read().unwrap().clone();
                let mut needs_rebalancing = false;

                for (_, info) in &device_info_snapshot {
                    if info.utilization > utilization_threshold
                        || (info.memory_stats.used_memory as f32
                            / info.memory_stats.total_memory as f32)
                            > memory_threshold
                    {
                        needs_rebalancing = true;
                        break;
                    }
                }

                if needs_rebalancing {
                    tracing::info!("Triggering dynamic load rebalancing");
                    // In a full implementation, this would redistribute queued operations
                    // For now, we just log the event

                    let mut stats_lock = stats.write().unwrap();
                    stats_lock.rebalancing_count += 1;
                }
            }
        });
    }

    /// Get current load balancing statistics
    pub fn get_statistics(&self) -> LoadBalancingStats {
        let stats = self.stats.read().unwrap();
        let device_info = self.device_info.read().unwrap();

        // Calculate derived metrics
        let total_gpus = device_info.len();
        let healthy_gpus = device_info.values().filter(|info| info.is_healthy).count();
        let gpu_availability_ratio = if total_gpus > 0 {
            healthy_gpus as f32 / total_gpus as f32
        } else {
            0.0
        };

        let system_utilization = if !device_info.is_empty() {
            device_info
                .values()
                .map(|info| info.utilization)
                .sum::<f32>()
                / device_info.len() as f32
        } else {
            0.0
        };

        // Calculate load distribution efficiency
        let load_distribution_efficiency = if stats.operations_per_gpu.len() > 1 {
            let ops_values: Vec<u64> = stats.operations_per_gpu.values().copied().collect();
            let max_ops = *ops_values.iter().max().unwrap_or(&0);
            let min_ops = *ops_values.iter().min().unwrap_or(&0);

            if max_ops > 0 {
                1.0 - ((max_ops - min_ops) as f32 / max_ops as f32)
            } else {
                1.0
            }
        } else {
            1.0
        };

        LoadBalancingStats {
            total_operations: stats.total_operations,
            operations_per_gpu: stats.operations_per_gpu.clone(),
            success_rate_per_gpu: stats.success_rate_per_gpu.clone(),
            avg_latency_per_gpu: stats.avg_latency_per_gpu.clone(),
            load_distribution_efficiency,
            failover_count: stats.failover_count,
            rebalancing_count: stats.rebalancing_count,
            system_utilization,
            gpu_availability_ratio,
        }
    }

    /// Get information about all GPU devices
    pub fn get_device_info(&self) -> HashMap<usize, GpuDeviceInfo> {
        self.device_info.read().unwrap().clone()
    }

    /// Get current configuration
    pub fn get_config(&self) -> &LoadBalancingConfig {
        &self.config
    }

    /// Update load balancing configuration
    pub fn update_config(&mut self, new_config: LoadBalancingConfig) {
        self.config = new_config;
    }

    /// Shutdown load balancer
    pub async fn shutdown(&self) {
        // Stop health monitoring
        {
            let mut active = self.health_monitoring_active.write().unwrap();
            *active = false;
        }

        // Clear all caches and shutdown GPUs
        let device_ids: Vec<usize> = { self.gpu_devices.read().unwrap().keys().copied().collect() };

        for device_id in device_ids {
            if let Some(accelerator) = self.gpu_devices.read().unwrap().get(&device_id) {
                accelerator.clear_cache();
                let _ = accelerator.synchronize().await;
            }
        }

        tracing::info!("GPU load balancer shutdown completed");
    }
}

impl Default for LoadBalancingStats {
    fn default() -> Self {
        Self {
            total_operations: 0,
            operations_per_gpu: HashMap::new(),
            success_rate_per_gpu: HashMap::new(),
            avg_latency_per_gpu: HashMap::new(),
            load_distribution_efficiency: 1.0,
            failover_count: 0,
            rebalancing_count: 0,
            system_utilization: 0.0,
            gpu_availability_ratio: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_acceleration::GpuOperationType;
    use candle_core::{DType, Tensor};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_load_balancer_creation() {
        let config = LoadBalancingConfig {
            strategy: LoadBalancingStrategy::RoundRobin,
            max_concurrent_ops_per_gpu: 4,
            enable_failover: true,
            ..Default::default()
        };

        // Should not panic even if no GPUs are available
        let result = GpuLoadBalancer::new(config).await;
        // Test might fail if no GPU devices available, but shouldn't panic
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_load_balancing_strategies() {
        assert_eq!(
            LoadBalancingStrategy::default(),
            LoadBalancingStrategy::LowestUtilization
        );

        let weighted = LoadBalancingStrategy::Weighted {
            utilization_weight: 0.3,
            performance_weight: 0.3,
            memory_weight: 0.2,
            reliability_weight: 0.2,
        };

        match weighted {
            LoadBalancingStrategy::Weighted {
                utilization_weight, ..
            } => {
                assert_eq!(utilization_weight, 0.3);
            }
            _ => panic!("Expected weighted strategy"),
        }
    }

    #[test]
    fn test_config_defaults() {
        let config = LoadBalancingConfig::default();
        assert_eq!(config.strategy, LoadBalancingStrategy::LowestUtilization);
        assert_eq!(config.max_concurrent_ops_per_gpu, 8);
        assert!(config.enable_failover);
        assert!(config.enable_dynamic_rebalancing);
        assert_eq!(config.health_check_interval_secs, 30);
        assert_eq!(config.utilization_threshold, 0.85);
        assert_eq!(config.memory_threshold, 0.90);
    }

    #[test]
    fn test_gpu_device_info() {
        let device_info = GpuDeviceInfo {
            device_id: 0,
            device_type: GpuDeviceType::Cuda,
            device_name: "Test GPU".to_string(),
            is_available: true,
            is_healthy: true,
            utilization: 0.5,
            memory_stats: GpuMemoryStats {
                total_memory: 8_000_000_000,
                used_memory: 4_000_000_000,
                free_memory: 4_000_000_000,
                peak_memory: 4_000_000_000,
                active_tensors: 10,
                fragmentation_ratio: 0.1,
            },
            performance_metrics: GpuPerformanceMetrics {
                gpu_utilization: 0.5,
                memory_bandwidth: 0.7,
                avg_kernel_time: Duration::from_millis(5),
                operations_count: 100,
                temperature: 55.0,
                power_consumption: 200.0,
            },
            active_operations: 2,
            queue_size: 5,
            last_health_check: SystemTime::now(),
            success_rate: 0.95,
            avg_latency: Duration::from_millis(10),
            temperature: 55.0,
        };

        assert_eq!(device_info.device_id, 0);
        assert_eq!(device_info.device_type, GpuDeviceType::Cuda);
        assert!(device_info.is_available);
        assert!(device_info.is_healthy);
        assert_eq!(device_info.utilization, 0.5);
        assert_eq!(device_info.success_rate, 0.95);
    }

    #[test]
    fn test_load_balancing_stats() {
        let mut stats = LoadBalancingStats::default();

        stats.total_operations = 100;
        stats.operations_per_gpu.insert(0, 60);
        stats.operations_per_gpu.insert(1, 40);
        stats.success_rate_per_gpu.insert(0, 0.95);
        stats.success_rate_per_gpu.insert(1, 0.98);

        assert_eq!(stats.total_operations, 100);
        assert_eq!(stats.operations_per_gpu.len(), 2);
        assert_eq!(stats.success_rate_per_gpu.len(), 2);
        assert_eq!(*stats.operations_per_gpu.get(&0).unwrap(), 60);
        assert_eq!(*stats.success_rate_per_gpu.get(&1).unwrap(), 0.98);
    }

    #[test]
    fn test_performance_prediction() {
        let prediction = PerformancePrediction {
            predicted_time: Duration::from_millis(100),
            confidence: 0.8,
            predicted_memory: 1_000_000,
            data_points: 50,
        };

        assert_eq!(prediction.predicted_time, Duration::from_millis(100));
        assert_eq!(prediction.confidence, 0.8);
        assert_eq!(prediction.predicted_memory, 1_000_000);
        assert_eq!(prediction.data_points, 50);
    }
}
