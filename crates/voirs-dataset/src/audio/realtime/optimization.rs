//! Performance optimization settings for real-time audio processing

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Real-time optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RealTimeOptimization {
    /// CPU optimization settings
    pub cpu_optimization: CPUOptimization,
    /// Memory optimization settings
    pub memory_optimization: MemoryOptimization,
    /// I/O optimization settings
    pub io_optimization: IOOptimization,
}

/// CPU optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPUOptimization {
    /// Enable SIMD optimizations
    pub simd_optimizations: bool,
    /// Thread affinity settings
    pub thread_affinity: ThreadAffinityConfig,
    /// CPU governor settings
    pub cpu_governor: CPUGovernorConfig,
    /// Instruction-level optimizations
    pub instruction_optimizations: InstructionOptimizations,
}

/// Thread affinity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadAffinityConfig {
    /// Enable thread affinity
    pub enabled: bool,
    /// CPU cores to bind to
    pub cpu_cores: Vec<usize>,
    /// Binding strategy
    pub binding_strategy: BindingStrategy,
}

/// Binding strategies for thread affinity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BindingStrategy {
    /// Static binding
    Static,
    /// Dynamic binding
    Dynamic,
    /// Load-balanced binding
    LoadBalanced,
    /// Custom binding
    Custom(String),
}

/// CPU governor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPUGovernorConfig {
    /// Enable CPU governor control
    pub enabled: bool,
    /// Target governor
    pub target_governor: CPUGovernor,
    /// Frequency scaling settings
    pub frequency_scaling: FrequencyScalingConfig,
}

/// CPU governors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CPUGovernor {
    /// Performance governor
    Performance,
    /// Powersave governor
    Powersave,
    /// Ondemand governor
    Ondemand,
    /// Conservative governor
    Conservative,
    /// Custom governor
    Custom(String),
}

/// Frequency scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyScalingConfig {
    /// Minimum frequency
    pub min_frequency: Option<u64>,
    /// Maximum frequency
    pub max_frequency: Option<u64>,
    /// Scaling factor
    pub scaling_factor: f32,
}

/// Instruction-level optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstructionOptimizations {
    /// Enable vectorization
    pub vectorization: bool,
    /// Enable loop unrolling
    pub loop_unrolling: bool,
    /// Enable function inlining
    pub function_inlining: bool,
    /// Enable branch prediction optimizations
    pub branch_prediction: bool,
}

/// Memory optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimization {
    /// Memory pool configuration
    pub memory_pools: MemoryPoolConfig,
    /// Garbage collection tuning
    pub gc_tuning: GCTuning,
    /// Cache optimization
    pub cache_optimization: CacheOptimizationConfig,
    /// Memory allocation strategy
    pub allocation_strategy: AllocationStrategy,
}

/// Memory pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolConfig {
    /// Enable memory pooling
    pub enabled: bool,
    /// Pool sizes
    pub pool_sizes: Vec<usize>,
    /// Maximum pool size
    pub max_pool_size: usize,
    /// Pool allocation strategy
    pub allocation_strategy: PoolAllocationStrategy,
}

/// Pool allocation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoolAllocationStrategy {
    /// First-fit allocation
    FirstFit,
    /// Best-fit allocation
    BestFit,
    /// Worst-fit allocation
    WorstFit,
    /// Custom allocation
    Custom(String),
}

/// Garbage collection tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCTuning {
    /// GC algorithm
    pub algorithm: GCAlgorithm,
    /// GC frequency
    pub frequency: GCFrequency,
    /// Memory pressure thresholds
    pub pressure_thresholds: PressureThresholds,
}

/// Garbage collection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GCAlgorithm {
    /// Mark and sweep
    MarkAndSweep,
    /// Generational GC
    Generational,
    /// Incremental GC
    Incremental,
    /// Concurrent GC
    Concurrent,
    /// Custom GC
    Custom(String),
}

/// Garbage collection frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GCFrequency {
    /// Automatic frequency
    Automatic,
    /// Fixed interval
    FixedInterval(Duration),
    /// Memory pressure triggered
    MemoryPressure,
    /// Custom frequency
    Custom(String),
}

/// Memory pressure thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PressureThresholds {
    /// Low pressure threshold (%)
    pub low_pressure: f32,
    /// Medium pressure threshold (%)
    pub medium_pressure: f32,
    /// High pressure threshold (%)
    pub high_pressure: f32,
    /// Critical pressure threshold (%)
    pub critical_pressure: f32,
}

/// Cache optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheOptimizationConfig {
    /// Enable cache optimization
    pub enabled: bool,
    /// Cache line alignment
    pub cache_line_alignment: bool,
    /// Prefetching strategy
    pub prefetching_strategy: PrefetchingStrategy,
    /// Cache-friendly data structures
    pub cache_friendly_structures: bool,
}

/// Prefetching strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrefetchingStrategy {
    /// No prefetching
    None,
    /// Sequential prefetching
    Sequential,
    /// Adaptive prefetching
    Adaptive,
    /// Custom prefetching
    Custom(String),
}

/// Memory allocation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Standard allocator
    Standard,
    /// Pool allocator
    Pool,
    /// Stack allocator
    Stack,
    /// Custom allocator
    Custom(String),
}

/// I/O optimization settings
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IOOptimization {
    /// Buffer size optimization
    pub buffer_optimization: BufferOptimizationConfig,
    /// Async I/O configuration
    pub async_io: AsyncIOConfig,
    /// Network optimization
    pub network_optimization: NetworkOptimizationConfig,
    /// Disk I/O optimization
    pub disk_optimization: DiskOptimizationConfig,
}

/// Buffer optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferOptimizationConfig {
    /// Enable buffer optimization
    pub enabled: bool,
    /// Optimal buffer sizes
    pub optimal_sizes: Vec<usize>,
    /// Buffer alignment
    pub alignment: usize,
    /// Zero-copy operations
    pub zero_copy: bool,
}

/// Async I/O configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncIOConfig {
    /// Enable async I/O
    pub enabled: bool,
    /// Number of async workers
    pub num_workers: usize,
    /// Queue depth
    pub queue_depth: usize,
    /// Batch size
    pub batch_size: usize,
}

/// Network optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkOptimizationConfig {
    /// TCP optimization
    pub tcp_optimization: TCPOptimization,
    /// UDP optimization
    pub udp_optimization: UDPOptimization,
    /// Bandwidth optimization
    pub bandwidth_optimization: BandwidthOptimization,
}

/// TCP optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCPOptimization {
    /// TCP no delay
    pub no_delay: bool,
    /// TCP keep alive
    pub keep_alive: bool,
    /// TCP window scaling
    pub window_scaling: bool,
    /// TCP congestion control
    pub congestion_control: String,
}

/// UDP optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UDPOptimization {
    /// UDP buffer size
    pub buffer_size: usize,
    /// UDP packet size
    pub packet_size: usize,
    /// UDP timeout
    pub timeout: Duration,
}

/// Bandwidth optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthOptimization {
    /// Bandwidth limiting
    pub bandwidth_limiting: bool,
    /// Adaptive bitrate
    pub adaptive_bitrate: bool,
    /// Compression
    pub compression: bool,
    /// Traffic shaping
    pub traffic_shaping: bool,
}

/// Disk I/O optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DiskOptimizationConfig {
    /// Read-ahead optimization
    pub read_ahead: ReadAheadConfig,
    /// Write optimization
    pub write_optimization: WriteOptimizationConfig,
    /// File system optimization
    pub filesystem_optimization: FilesystemOptimizationConfig,
}

/// Read-ahead configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadAheadConfig {
    /// Enable read-ahead
    pub enabled: bool,
    /// Read-ahead size
    pub size: usize,
    /// Read-ahead strategy
    pub strategy: ReadAheadStrategy,
}

/// Read-ahead strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReadAheadStrategy {
    /// Sequential read-ahead
    Sequential,
    /// Random read-ahead
    Random,
    /// Adaptive read-ahead
    Adaptive,
    /// Custom read-ahead
    Custom(String),
}

/// Write optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteOptimizationConfig {
    /// Write batching
    pub batching: bool,
    /// Write caching
    pub caching: bool,
    /// Sync strategy
    pub sync_strategy: SyncStrategy,
}

/// Sync strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncStrategy {
    /// Immediate sync
    Immediate,
    /// Periodic sync
    Periodic(Duration),
    /// Threshold-based sync
    Threshold(usize),
    /// Custom sync
    Custom(String),
}

/// Filesystem optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesystemOptimizationConfig {
    /// Direct I/O
    pub direct_io: bool,
    /// Memory mapping
    pub memory_mapping: bool,
    /// File preallocation
    pub preallocation: bool,
}

impl Default for CPUOptimization {
    fn default() -> Self {
        Self {
            simd_optimizations: true,
            thread_affinity: ThreadAffinityConfig::default(),
            cpu_governor: CPUGovernorConfig::default(),
            instruction_optimizations: InstructionOptimizations::default(),
        }
    }
}

impl Default for ThreadAffinityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cpu_cores: Vec::new(),
            binding_strategy: BindingStrategy::LoadBalanced,
        }
    }
}

impl Default for CPUGovernorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            target_governor: CPUGovernor::Performance,
            frequency_scaling: FrequencyScalingConfig::default(),
        }
    }
}

impl Default for FrequencyScalingConfig {
    fn default() -> Self {
        Self {
            min_frequency: None,
            max_frequency: None,
            scaling_factor: 1.0,
        }
    }
}

impl Default for InstructionOptimizations {
    fn default() -> Self {
        Self {
            vectorization: true,
            loop_unrolling: true,
            function_inlining: true,
            branch_prediction: true,
        }
    }
}

impl Default for MemoryOptimization {
    fn default() -> Self {
        Self {
            memory_pools: MemoryPoolConfig::default(),
            gc_tuning: GCTuning::default(),
            cache_optimization: CacheOptimizationConfig::default(),
            allocation_strategy: AllocationStrategy::Standard,
        }
    }
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            pool_sizes: vec![1024, 4096, 16384],
            max_pool_size: 1024 * 1024,
            allocation_strategy: PoolAllocationStrategy::BestFit,
        }
    }
}

impl Default for GCTuning {
    fn default() -> Self {
        Self {
            algorithm: GCAlgorithm::Incremental,
            frequency: GCFrequency::MemoryPressure,
            pressure_thresholds: PressureThresholds::default(),
        }
    }
}

impl Default for PressureThresholds {
    fn default() -> Self {
        Self {
            low_pressure: 50.0,
            medium_pressure: 70.0,
            high_pressure: 85.0,
            critical_pressure: 95.0,
        }
    }
}

impl Default for CacheOptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cache_line_alignment: true,
            prefetching_strategy: PrefetchingStrategy::Sequential,
            cache_friendly_structures: true,
        }
    }
}

impl Default for BufferOptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            optimal_sizes: vec![4096, 8192, 16384],
            alignment: 64,
            zero_copy: true,
        }
    }
}

impl Default for AsyncIOConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            num_workers: num_cpus::get(),
            queue_depth: 32,
            batch_size: 8,
        }
    }
}

impl Default for TCPOptimization {
    fn default() -> Self {
        Self {
            no_delay: true,
            keep_alive: true,
            window_scaling: true,
            congestion_control: "cubic".to_string(),
        }
    }
}

impl Default for UDPOptimization {
    fn default() -> Self {
        Self {
            buffer_size: 65536,
            packet_size: 1500,
            timeout: Duration::from_millis(100),
        }
    }
}

impl Default for BandwidthOptimization {
    fn default() -> Self {
        Self {
            bandwidth_limiting: false,
            adaptive_bitrate: true,
            compression: true,
            traffic_shaping: false,
        }
    }
}

impl Default for ReadAheadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            size: 8192,
            strategy: ReadAheadStrategy::Sequential,
        }
    }
}

impl Default for WriteOptimizationConfig {
    fn default() -> Self {
        Self {
            batching: true,
            caching: true,
            sync_strategy: SyncStrategy::Periodic(Duration::from_secs(1)),
        }
    }
}

impl Default for FilesystemOptimizationConfig {
    fn default() -> Self {
        Self {
            direct_io: false,
            memory_mapping: true,
            preallocation: true,
        }
    }
}
