//! Processing optimization for multi-modal analysis

use serde::{Deserialize, Serialize};

/// Processing optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingOptimization {
    /// Enable GPU acceleration
    pub gpu_acceleration: bool,
    /// Number of processing threads
    pub num_threads: usize,
    /// Memory optimization
    pub memory_optimization: MemoryOptimization,
    /// Batch processing configuration
    pub batch_processing: BatchProcessingConfig,
}

/// Memory optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimization {
    /// Enable memory pooling
    pub enable_pooling: bool,
    /// Maximum memory usage (MB)
    pub max_memory_mb: usize,
    /// Garbage collection frequency
    pub gc_frequency: usize,
    /// Use memory mapping
    pub use_memory_mapping: bool,
}

/// Batch processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessingConfig {
    /// Batch size
    pub batch_size: usize,
    /// Processing pipeline depth
    pub pipeline_depth: usize,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round robin
    RoundRobin,
    /// Least loaded
    LeastLoaded,
    /// Random
    Random,
    /// Weighted
    Weighted,
}

/// GPU acceleration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUAccelerationConfig {
    /// Enable GPU acceleration
    pub enabled: bool,
    /// Preferred GPU device
    pub device_id: Option<usize>,
    /// Memory allocation strategy
    pub memory_strategy: GPUMemoryStrategy,
    /// Batch size for GPU processing
    pub gpu_batch_size: usize,
}

/// GPU memory allocation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GPUMemoryStrategy {
    /// Conservative memory usage
    Conservative,
    /// Aggressive memory usage
    Aggressive,
    /// Balanced memory usage
    Balanced,
    /// Dynamic memory allocation
    Dynamic,
}

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitoring {
    /// Enable performance monitoring
    pub enabled: bool,
    /// Monitoring interval (seconds)
    pub monitoring_interval: f32,
    /// Metrics to collect
    pub metrics: Vec<PerformanceMetric>,
    /// Performance thresholds
    pub thresholds: PerformanceThresholds,
}

/// Performance metrics to monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceMetric {
    /// CPU usage percentage
    CPUUsage,
    /// Memory usage (MB)
    MemoryUsage,
    /// GPU usage percentage
    GPUUsage,
    /// Processing throughput (items/second)
    Throughput,
    /// Processing latency (milliseconds)
    Latency,
    /// Queue depth
    QueueDepth,
}

/// Performance thresholds for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    /// Maximum CPU usage before optimization
    pub max_cpu_usage: f32,
    /// Maximum memory usage before optimization (MB)
    pub max_memory_usage: usize,
    /// Maximum latency before optimization (ms)
    pub max_latency: f32,
    /// Minimum throughput before optimization
    pub min_throughput: f32,
}

/// Optimization strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStrategy {
    /// Optimization approach
    pub approach: OptimizationApproach,
    /// Adaptation parameters
    pub adaptation: AdaptationParameters,
    /// Optimization objectives
    pub objectives: Vec<OptimizationObjective>,
}

/// Optimization approaches
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationApproach {
    /// Static optimization
    Static,
    /// Dynamic optimization
    Dynamic,
    /// Adaptive optimization
    Adaptive,
    /// Hybrid optimization
    Hybrid,
}

/// Adaptation parameters for dynamic optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationParameters {
    /// Learning rate for adaptation
    pub learning_rate: f32,
    /// Adaptation window size
    pub window_size: usize,
    /// Minimum adaptation threshold
    pub min_threshold: f32,
    /// Maximum adaptation rate
    pub max_adaptation_rate: f32,
}

/// Optimization objectives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationObjective {
    /// Minimize processing time
    MinimizeLatency,
    /// Maximize throughput
    MaximizeThroughput,
    /// Minimize memory usage
    MinimizeMemory,
    /// Minimize energy consumption
    MinimizeEnergy,
    /// Maximize quality
    MaximizeQuality,
}

/// Resource utilization information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU utilization percentage
    pub cpu_usage: f32,
    /// Memory usage in MB
    pub memory_usage: usize,
    /// GPU utilization percentage
    pub gpu_usage: Option<f32>,
    /// Network bandwidth usage
    pub network_usage: Option<f32>,
    /// Disk I/O usage
    pub disk_usage: Option<f32>,
}

/// Performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Processing throughput (items per second)
    pub throughput: f32,
    /// Average processing latency (milliseconds)
    pub avg_latency: f32,
    /// Peak processing latency (milliseconds)
    pub peak_latency: f32,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
    /// Error rate
    pub error_rate: f32,
    /// Latency sample count for averaging
    #[serde(skip)]
    latency_count: usize,
    /// Total latency sum for averaging
    #[serde(skip)]
    latency_sum: f32,
}

impl ProcessingOptimization {
    /// Create new processing optimization configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable GPU acceleration
    pub fn with_gpu_acceleration(mut self) -> Self {
        self.gpu_acceleration = true;
        self
    }

    /// Set number of threads
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_processing.batch_size = batch_size;
        self
    }

    /// Set maximum memory usage
    pub fn with_max_memory(mut self, max_memory_mb: usize) -> Self {
        self.memory_optimization.max_memory_mb = max_memory_mb;
        self
    }
}

impl MemoryOptimization {
    /// Create new memory optimization configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable memory pooling
    pub fn with_pooling(mut self) -> Self {
        self.enable_pooling = true;
        self
    }

    /// Enable memory mapping
    pub fn with_memory_mapping(mut self) -> Self {
        self.use_memory_mapping = true;
        self
    }
}

impl BatchProcessingConfig {
    /// Create new batch processing configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set load balancing strategy
    pub fn with_load_balancing(mut self, strategy: LoadBalancingStrategy) -> Self {
        self.load_balancing = strategy;
        self
    }
}

impl PerformanceStats {
    /// Create new performance statistics
    pub fn new() -> Self {
        Self {
            throughput: 0.0,
            avg_latency: 0.0,
            peak_latency: 0.0,
            resource_utilization: ResourceUtilization::new(),
            error_rate: 0.0,
            latency_count: 0,
            latency_sum: 0.0,
        }
    }

    /// Update throughput
    pub fn update_throughput(&mut self, throughput: f32) {
        self.throughput = throughput;
    }

    /// Update latency
    pub fn update_latency(&mut self, latency: f32) {
        self.latency_count += 1;
        self.latency_sum += latency;
        self.avg_latency = self.latency_sum / self.latency_count as f32;
        if latency > self.peak_latency {
            self.peak_latency = latency;
        }
    }

    /// Update resource utilization
    pub fn update_resource_utilization(&mut self, utilization: ResourceUtilization) {
        self.resource_utilization = utilization;
    }

    /// Check if performance is within acceptable limits
    pub fn is_within_limits(&self, thresholds: &PerformanceThresholds) -> bool {
        self.resource_utilization.cpu_usage <= thresholds.max_cpu_usage
            && self.resource_utilization.memory_usage <= thresholds.max_memory_usage
            && self.avg_latency <= thresholds.max_latency
            && self.throughput >= thresholds.min_throughput
    }
}

impl ResourceUtilization {
    /// Create new resource utilization
    pub fn new() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0,
            gpu_usage: None,
            network_usage: None,
            disk_usage: None,
        }
    }

    /// Update CPU usage
    pub fn update_cpu_usage(&mut self, usage: f32) {
        self.cpu_usage = usage;
    }

    /// Update memory usage
    pub fn update_memory_usage(&mut self, usage: usize) {
        self.memory_usage = usage;
    }

    /// Update GPU usage
    pub fn update_gpu_usage(&mut self, usage: f32) {
        self.gpu_usage = Some(usage);
    }
}

impl Default for ProcessingOptimization {
    fn default() -> Self {
        Self {
            gpu_acceleration: false,
            num_threads: num_cpus::get(),
            memory_optimization: MemoryOptimization::default(),
            batch_processing: BatchProcessingConfig::default(),
        }
    }
}

impl Default for MemoryOptimization {
    fn default() -> Self {
        Self {
            enable_pooling: true,
            max_memory_mb: 1024,
            gc_frequency: 100,
            use_memory_mapping: false,
        }
    }
}

impl Default for BatchProcessingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            pipeline_depth: 4,
            load_balancing: LoadBalancingStrategy::LeastLoaded,
        }
    }
}

impl Default for GPUAccelerationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            device_id: None,
            memory_strategy: GPUMemoryStrategy::Balanced,
            gpu_batch_size: 64,
        }
    }
}

impl Default for PerformanceMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            monitoring_interval: 1.0,
            metrics: vec![
                PerformanceMetric::CPUUsage,
                PerformanceMetric::MemoryUsage,
                PerformanceMetric::Throughput,
                PerformanceMetric::Latency,
            ],
            thresholds: PerformanceThresholds::default(),
        }
    }
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_cpu_usage: 80.0,
            max_memory_usage: 2048,
            max_latency: 100.0,
            min_throughput: 10.0,
        }
    }
}

impl Default for OptimizationStrategy {
    fn default() -> Self {
        Self {
            approach: OptimizationApproach::Adaptive,
            adaptation: AdaptationParameters::default(),
            objectives: vec![
                OptimizationObjective::MinimizeLatency,
                OptimizationObjective::MaximizeThroughput,
            ],
        }
    }
}

impl Default for AdaptationParameters {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            window_size: 100,
            min_threshold: 0.05,
            max_adaptation_rate: 0.5,
        }
    }
}

impl Default for ResourceUtilization {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self::new()
    }
}
