//! Runtime configuration for acoustic models
//!
//! This module defines configuration structures for runtime behavior,
//! including backend selection, memory management, and performance tuning.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::{Result, AcousticError};
use super::{DeviceConfig, CacheConfig, DeviceType};

/// Runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Backend selection
    pub backend: BackendConfig,
    /// Device configuration
    pub device: DeviceConfig,
    /// Memory management
    pub memory: MemoryConfig,
    /// Performance tuning
    pub performance: PerformanceConfig,
    /// Caching configuration
    pub cache: CacheConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
    /// Debugging options
    pub debug: DebugConfig,
}

impl RuntimeConfig {
    /// Create new runtime configuration
    pub fn new() -> Self {
        Self {
            backend: BackendConfig::default(),
            device: DeviceConfig::default(),
            memory: MemoryConfig::default(),
            performance: PerformanceConfig::default(),
            cache: CacheConfig::default(),
            logging: LoggingConfig::default(),
            debug: DebugConfig::default(),
        }
    }
    
    /// Create configuration for CPU execution
    pub fn cpu() -> Self {
        let mut config = Self::new();
        config.backend.preferred_backend = BackendType::Candle;
        config.device = DeviceConfig::cpu();
        config.performance.num_threads = Some(std::thread::available_parallelism().map(|p| p.get() as u32).unwrap_or(4));
        config
    }
    
    /// Create configuration for GPU execution
    pub fn gpu() -> Self {
        let mut config = Self::new();
        config.backend.preferred_backend = BackendType::Candle;
        config.device = DeviceConfig::cuda(Some(0));
        config.memory.enable_memory_pool = true;
        config.performance.use_optimized_kernels = true;
        config
    }
    
    /// Create configuration for fast inference
    pub fn fast() -> Self {
        let mut config = Self::new();
        config.backend.preferred_backend = BackendType::Onnx;
        config.performance.optimization_level = OptimizationLevel::Fast;
        config.memory.enable_memory_pool = true;
        config.cache.enabled = true;
        config
    }
    
    /// Validate runtime configuration
    pub fn validate(&self) -> Result<()> {
        self.backend.validate()?;
        self.device.validate()?;
        self.memory.validate()?;
        self.performance.validate()?;
        self.logging.validate()?;
        self.debug.validate()?;
        Ok(())
    }
    
    /// Merge with another runtime configuration
    pub fn merge(&mut self, other: &RuntimeConfig) {
        self.backend.merge(&other.backend);
        self.device.merge(&other.device);
        self.memory.merge(&other.memory);
        self.performance.merge(&other.performance);
        self.logging.merge(&other.logging);
        self.debug.merge(&other.debug);
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// Preferred backend type
    pub preferred_backend: BackendType,
    /// Fallback backends in order of preference
    pub fallback_backends: Vec<BackendType>,
    /// Backend-specific options
    pub backend_options: HashMap<BackendType, BackendOptions>,
}

impl BackendConfig {
    /// Create new backend configuration
    pub fn new() -> Self {
        Self {
            preferred_backend: BackendType::Candle,
            fallback_backends: vec![BackendType::Onnx],
            backend_options: HashMap::new(),
        }
    }
    
    /// Set preferred backend
    pub fn with_backend(mut self, backend: BackendType) -> Self {
        self.preferred_backend = backend;
        self
    }
    
    /// Add fallback backend
    pub fn with_fallback(mut self, backend: BackendType) -> Self {
        self.fallback_backends.push(backend);
        self
    }
    
    /// Set backend options
    pub fn with_options(mut self, backend: BackendType, options: BackendOptions) -> Self {
        self.backend_options.insert(backend, options);
        self
    }
    
    /// Validate backend configuration
    pub fn validate(&self) -> Result<()> {
        for options in self.backend_options.values() {
            options.validate()?;
        }
        Ok(())
    }
    
    /// Merge with another backend configuration
    pub fn merge(&mut self, other: &BackendConfig) {
        self.preferred_backend = other.preferred_backend;
        self.fallback_backends = other.fallback_backends.clone();
        self.backend_options.extend(other.backend_options.clone());
    }
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Supported backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendType {
    /// Candle backend
    Candle,
    /// ONNX Runtime backend
    Onnx,
    /// Custom backend
    Custom(u32),
}

impl BackendType {
    /// Get string representation
    pub fn as_str(&self) -> &str {
        match self {
            BackendType::Candle => "candle",
            BackendType::Onnx => "onnx",
            BackendType::Custom(_) => "custom",
        }
    }
}

/// Backend-specific options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendOptions {
    /// Candle-specific options
    pub candle: Option<CandleOptions>,
    /// ONNX-specific options
    pub onnx: Option<OnnxOptions>,
    /// Custom options
    pub custom: Option<HashMap<String, serde_json::Value>>,
}

impl BackendOptions {
    /// Create new backend options
    pub fn new() -> Self {
        Self {
            candle: None,
            onnx: None,
            custom: None,
        }
    }
    
    /// Create Candle options
    pub fn candle(options: CandleOptions) -> Self {
        Self {
            candle: Some(options),
            onnx: None,
            custom: None,
        }
    }
    
    /// Create ONNX options
    pub fn onnx(options: OnnxOptions) -> Self {
        Self {
            candle: None,
            onnx: Some(options),
            custom: None,
        }
    }
    
    /// Validate backend options
    pub fn validate(&self) -> Result<()> {
        if let Some(candle) = &self.candle {
            candle.validate()?;
        }
        if let Some(onnx) = &self.onnx {
            onnx.validate()?;
        }
        Ok(())
    }
}

impl Default for BackendOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Candle backend options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleOptions {
    /// Use metal performance shaders on macOS
    pub use_metal_performance_shaders: bool,
    /// Enable CUDA graphs
    pub enable_cuda_graphs: bool,
    /// CUDA memory pool size in MB
    pub cuda_memory_pool_mb: Option<u32>,
    /// Use optimized attention kernels
    pub use_optimized_attention: bool,
}

impl CandleOptions {
    /// Create new Candle options
    pub fn new() -> Self {
        Self {
            use_metal_performance_shaders: true,
            enable_cuda_graphs: false,
            cuda_memory_pool_mb: None,
            use_optimized_attention: true,
        }
    }
    
    /// Enable CUDA optimizations
    pub fn cuda_optimized() -> Self {
        Self {
            use_metal_performance_shaders: false,
            enable_cuda_graphs: true,
            cuda_memory_pool_mb: Some(1024),
            use_optimized_attention: true,
        }
    }
    
    /// Enable Metal optimizations
    pub fn metal_optimized() -> Self {
        Self {
            use_metal_performance_shaders: true,
            enable_cuda_graphs: false,
            cuda_memory_pool_mb: None,
            use_optimized_attention: true,
        }
    }
    
    /// Validate Candle options
    pub fn validate(&self) -> Result<()> {
        if let Some(pool_size) = self.cuda_memory_pool_mb {
            if pool_size == 0 {
                return Err(AcousticError::ConfigError("CUDA memory pool size must be > 0".to_string()));
            }
        }
        Ok(())
    }
}

impl Default for CandleOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// ONNX backend options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxOptions {
    /// Execution providers in order of preference
    pub execution_providers: Vec<String>,
    /// Session options
    pub session_options: OnnxSessionOptions,
    /// Graph optimization level
    pub graph_optimization_level: OnnxOptimizationLevel,
}

impl OnnxOptions {
    /// Create new ONNX options
    pub fn new() -> Self {
        Self {
            execution_providers: vec!["CPUExecutionProvider".to_string()],
            session_options: OnnxSessionOptions::default(),
            graph_optimization_level: OnnxOptimizationLevel::Basic,
        }
    }
    
    /// Create CUDA-optimized options
    pub fn cuda() -> Self {
        Self {
            execution_providers: vec![
                "CUDAExecutionProvider".to_string(),
                "CPUExecutionProvider".to_string(),
            ],
            session_options: OnnxSessionOptions::default(),
            graph_optimization_level: OnnxOptimizationLevel::All,
        }
    }
    
    /// Create TensorRT-optimized options
    pub fn tensorrt() -> Self {
        Self {
            execution_providers: vec![
                "TensorrtExecutionProvider".to_string(),
                "CUDAExecutionProvider".to_string(),
                "CPUExecutionProvider".to_string(),
            ],
            session_options: OnnxSessionOptions::default(),
            graph_optimization_level: OnnxOptimizationLevel::All,
        }
    }
    
    /// Validate ONNX options
    pub fn validate(&self) -> Result<()> {
        if self.execution_providers.is_empty() {
            return Err(AcousticError::ConfigError("At least one execution provider must be specified".to_string()));
        }
        self.session_options.validate()?;
        Ok(())
    }
}

impl Default for OnnxOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// ONNX session options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxSessionOptions {
    /// Number of threads for intra-op parallelism
    pub intra_op_num_threads: Option<u32>,
    /// Number of threads for inter-op parallelism
    pub inter_op_num_threads: Option<u32>,
    /// Enable memory pattern optimization
    pub enable_mem_pattern: bool,
    /// Enable CPU memory arena
    pub enable_cpu_mem_arena: bool,
    /// Memory arena size in MB
    pub memory_arena_size_mb: Option<u32>,
}

impl OnnxSessionOptions {
    /// Create new ONNX session options
    pub fn new() -> Self {
        Self {
            intra_op_num_threads: None,
            inter_op_num_threads: None,
            enable_mem_pattern: true,
            enable_cpu_mem_arena: true,
            memory_arena_size_mb: None,
        }
    }
    
    /// Validate ONNX session options
    pub fn validate(&self) -> Result<()> {
        if let Some(threads) = self.intra_op_num_threads {
            if threads == 0 {
                return Err(AcousticError::ConfigError("Intra-op threads must be > 0".to_string()));
            }
        }
        if let Some(threads) = self.inter_op_num_threads {
            if threads == 0 {
                return Err(AcousticError::ConfigError("Inter-op threads must be > 0".to_string()));
            }
        }
        if let Some(arena_size) = self.memory_arena_size_mb {
            if arena_size == 0 {
                return Err(AcousticError::ConfigError("Memory arena size must be > 0".to_string()));
            }
        }
        Ok(())
    }
}

impl Default for OnnxSessionOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// ONNX optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OnnxOptimizationLevel {
    /// No optimization
    None,
    /// Basic optimization
    Basic,
    /// Extended optimization
    Extended,
    /// All optimizations
    All,
}

impl OnnxOptimizationLevel {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            OnnxOptimizationLevel::None => "none",
            OnnxOptimizationLevel::Basic => "basic",
            OnnxOptimizationLevel::Extended => "extended",
            OnnxOptimizationLevel::All => "all",
        }
    }
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Enable memory pool
    pub enable_memory_pool: bool,
    /// Memory pool size in MB
    pub memory_pool_size_mb: Option<u32>,
    /// Enable memory mapping for model loading
    pub enable_memory_mapping: bool,
    /// Memory cleanup interval in seconds
    pub cleanup_interval_seconds: u32,
    /// Memory pressure threshold (0.0 to 1.0)
    pub pressure_threshold: f32,
    /// Enable garbage collection
    pub enable_gc: bool,
}

impl MemoryConfig {
    /// Create new memory configuration
    pub fn new() -> Self {
        Self {
            enable_memory_pool: false,
            memory_pool_size_mb: None,
            enable_memory_mapping: true,
            cleanup_interval_seconds: 60,
            pressure_threshold: 0.8,
            enable_gc: true,
        }
    }
    
    /// Create memory-optimized configuration
    pub fn optimized() -> Self {
        Self {
            enable_memory_pool: true,
            memory_pool_size_mb: Some(1024),
            enable_memory_mapping: true,
            cleanup_interval_seconds: 30,
            pressure_threshold: 0.7,
            enable_gc: true,
        }
    }
    
    /// Validate memory configuration
    pub fn validate(&self) -> Result<()> {
        if let Some(pool_size) = self.memory_pool_size_mb {
            if pool_size == 0 {
                return Err(AcousticError::ConfigError("Memory pool size must be > 0".to_string()));
            }
        }
        if self.cleanup_interval_seconds == 0 {
            return Err(AcousticError::ConfigError("Cleanup interval must be > 0".to_string()));
        }
        if self.pressure_threshold < 0.0 || self.pressure_threshold > 1.0 {
            return Err(AcousticError::ConfigError("Pressure threshold must be between 0.0 and 1.0".to_string()));
        }
        Ok(())
    }
    
    /// Merge with another memory configuration
    pub fn merge(&mut self, other: &MemoryConfig) {
        self.enable_memory_pool = other.enable_memory_pool;
        if other.memory_pool_size_mb.is_some() {
            self.memory_pool_size_mb = other.memory_pool_size_mb;
        }
        self.enable_memory_mapping = other.enable_memory_mapping;
        self.cleanup_interval_seconds = other.cleanup_interval_seconds;
        self.pressure_threshold = other.pressure_threshold;
        self.enable_gc = other.enable_gc;
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Number of threads for CPU operations
    pub num_threads: Option<u32>,
    /// Use optimized kernels
    pub use_optimized_kernels: bool,
    /// Enable kernel fusion
    pub enable_kernel_fusion: bool,
    /// Batch size for batch operations
    pub batch_size: Option<u32>,
    /// Enable async inference
    pub enable_async_inference: bool,
    /// Inference timeout in milliseconds
    pub inference_timeout_ms: Option<u32>,
}

impl PerformanceConfig {
    /// Create new performance configuration
    pub fn new() -> Self {
        Self {
            optimization_level: OptimizationLevel::Balanced,
            num_threads: None,
            use_optimized_kernels: true,
            enable_kernel_fusion: false,
            batch_size: None,
            enable_async_inference: true,
            inference_timeout_ms: None,
        }
    }
    
    /// Create performance-optimized configuration
    pub fn optimized() -> Self {
        Self {
            optimization_level: OptimizationLevel::Fast,
            num_threads: Some(std::thread::available_parallelism().map(|p| p.get() as u32).unwrap_or(4)),
            use_optimized_kernels: true,
            enable_kernel_fusion: true,
            batch_size: Some(32),
            enable_async_inference: true,
            inference_timeout_ms: Some(5000),
        }
    }
    
    /// Validate performance configuration
    pub fn validate(&self) -> Result<()> {
        if let Some(threads) = self.num_threads {
            if threads == 0 {
                return Err(AcousticError::ConfigError("Number of threads must be > 0".to_string()));
            }
        }
        if let Some(batch_size) = self.batch_size {
            if batch_size == 0 {
                return Err(AcousticError::ConfigError("Batch size must be > 0".to_string()));
            }
        }
        if let Some(timeout) = self.inference_timeout_ms {
            if timeout == 0 {
                return Err(AcousticError::ConfigError("Inference timeout must be > 0".to_string()));
            }
        }
        Ok(())
    }
    
    /// Merge with another performance configuration
    pub fn merge(&mut self, other: &PerformanceConfig) {
        self.optimization_level = other.optimization_level;
        if other.num_threads.is_some() {
            self.num_threads = other.num_threads;
        }
        self.use_optimized_kernels = other.use_optimized_kernels;
        self.enable_kernel_fusion = other.enable_kernel_fusion;
        if other.batch_size.is_some() {
            self.batch_size = other.batch_size;
        }
        self.enable_async_inference = other.enable_async_inference;
        if other.inference_timeout_ms.is_some() {
            self.inference_timeout_ms = other.inference_timeout_ms;
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Fast optimization
    Fast,
    /// Balanced optimization
    Balanced,
    /// Maximum optimization
    Maximum,
}

impl OptimizationLevel {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            OptimizationLevel::None => "none",
            OptimizationLevel::Fast => "fast",
            OptimizationLevel::Balanced => "balanced",
            OptimizationLevel::Maximum => "maximum",
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Enable logging
    pub enabled: bool,
    /// Log level
    pub level: LogLevel,
    /// Log to file
    pub log_to_file: bool,
    /// Log file path
    pub log_file_path: Option<PathBuf>,
    /// Log to console
    pub log_to_console: bool,
    /// Enable structured logging
    pub structured_logging: bool,
    /// Log performance metrics
    pub log_performance: bool,
}

impl LoggingConfig {
    /// Create new logging configuration
    pub fn new() -> Self {
        Self {
            enabled: true,
            level: LogLevel::Info,
            log_to_file: false,
            log_file_path: None,
            log_to_console: true,
            structured_logging: false,
            log_performance: false,
        }
    }
    
    /// Create debug logging configuration
    pub fn debug() -> Self {
        Self {
            enabled: true,
            level: LogLevel::Debug,
            log_to_file: true,
            log_file_path: Some(PathBuf::from("voirs-acoustic-debug.log")),
            log_to_console: true,
            structured_logging: true,
            log_performance: true,
        }
    }
    
    /// Validate logging configuration
    pub fn validate(&self) -> Result<()> {
        if self.log_to_file && self.log_file_path.is_none() {
            return Err(AcousticError::ConfigError("Log file path must be specified when logging to file".to_string()));
        }
        Ok(())
    }
    
    /// Merge with another logging configuration
    pub fn merge(&mut self, other: &LoggingConfig) {
        self.enabled = other.enabled;
        self.level = other.level;
        self.log_to_file = other.log_to_file;
        if other.log_file_path.is_some() {
            self.log_file_path = other.log_file_path.clone();
        }
        self.log_to_console = other.log_to_console;
        self.structured_logging = other.structured_logging;
        self.log_performance = other.log_performance;
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Log levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Error => "error",
            LogLevel::Warn => "warn",
            LogLevel::Info => "info",
            LogLevel::Debug => "debug",
            LogLevel::Trace => "trace",
        }
    }
}

/// Debug configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugConfig {
    /// Enable debug mode
    pub enabled: bool,
    /// Save intermediate results
    pub save_intermediate_results: bool,
    /// Intermediate results directory
    pub intermediate_results_dir: Option<PathBuf>,
    /// Enable profiling
    pub enable_profiling: bool,
    /// Profile output directory
    pub profile_output_dir: Option<PathBuf>,
    /// Validate tensor shapes
    pub validate_tensor_shapes: bool,
    /// Check for NaN values
    pub check_for_nan: bool,
}

impl DebugConfig {
    /// Create new debug configuration
    pub fn new() -> Self {
        Self {
            enabled: false,
            save_intermediate_results: false,
            intermediate_results_dir: None,
            enable_profiling: false,
            profile_output_dir: None,
            validate_tensor_shapes: false,
            check_for_nan: false,
        }
    }
    
    /// Create full debug configuration
    pub fn full_debug() -> Self {
        Self {
            enabled: true,
            save_intermediate_results: true,
            intermediate_results_dir: Some(PathBuf::from("debug/intermediate")),
            enable_profiling: true,
            profile_output_dir: Some(PathBuf::from("debug/profile")),
            validate_tensor_shapes: true,
            check_for_nan: true,
        }
    }
    
    /// Validate debug configuration
    pub fn validate(&self) -> Result<()> {
        if self.save_intermediate_results && self.intermediate_results_dir.is_none() {
            return Err(AcousticError::ConfigError("Intermediate results directory must be specified".to_string()));
        }
        if self.enable_profiling && self.profile_output_dir.is_none() {
            return Err(AcousticError::ConfigError("Profile output directory must be specified".to_string()));
        }
        Ok(())
    }
    
    /// Merge with another debug configuration
    pub fn merge(&mut self, other: &DebugConfig) {
        self.enabled = other.enabled;
        self.save_intermediate_results = other.save_intermediate_results;
        if other.intermediate_results_dir.is_some() {
            self.intermediate_results_dir = other.intermediate_results_dir.clone();
        }
        self.enable_profiling = other.enable_profiling;
        if other.profile_output_dir.is_some() {
            self.profile_output_dir = other.profile_output_dir.clone();
        }
        self.validate_tensor_shapes = other.validate_tensor_shapes;
        self.check_for_nan = other.check_for_nan;
    }
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self::new()
    }
}

// Add the missing validate and merge methods for DeviceConfig
impl DeviceConfig {
    /// Validate device configuration
    pub fn validate(&self) -> Result<()> {
        if let Some(max_memory) = self.max_memory_mb {
            if max_memory == 0 {
                return Err(AcousticError::ConfigError("Max memory must be > 0".to_string()));
            }
        }
        Ok(())
    }
    
    /// Merge with another device configuration
    pub fn merge(&mut self, other: &DeviceConfig) {
        self.device_type = other.device_type;
        if other.device_index.is_some() {
            self.device_index = other.device_index;
        }
        self.mixed_precision = other.mixed_precision;
        if other.max_memory_mb.is_some() {
            self.max_memory_mb = other.max_memory_mb;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_config_validation() {
        let config = RuntimeConfig::new();
        assert!(config.validate().is_ok());
        
        let cpu_config = RuntimeConfig::cpu();
        assert!(cpu_config.validate().is_ok());
        assert_eq!(cpu_config.device.device_type, DeviceType::Cpu);
        
        let gpu_config = RuntimeConfig::gpu();
        assert!(gpu_config.validate().is_ok());
        assert_eq!(gpu_config.device.device_type, DeviceType::Cuda);
        
        let fast_config = RuntimeConfig::fast();
        assert!(fast_config.validate().is_ok());
        assert_eq!(fast_config.backend.preferred_backend, BackendType::Onnx);
    }

    #[test]
    fn test_backend_config() {
        let config = BackendConfig::new()
            .with_backend(BackendType::Onnx)
            .with_fallback(BackendType::Candle);
        
        assert_eq!(config.preferred_backend, BackendType::Onnx);
        assert!(config.fallback_backends.contains(&BackendType::Candle));
    }

    #[test]
    fn test_memory_config_validation() {
        let config = MemoryConfig::new();
        assert!(config.validate().is_ok());
        
        let mut config = MemoryConfig::new();
        config.memory_pool_size_mb = Some(0);
        assert!(config.validate().is_err());
        
        config.memory_pool_size_mb = Some(1024);
        config.pressure_threshold = 1.5;
        assert!(config.validate().is_err());
        
        config.pressure_threshold = 0.8;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_performance_config_validation() {
        let config = PerformanceConfig::new();
        assert!(config.validate().is_ok());
        
        let mut config = PerformanceConfig::new();
        config.num_threads = Some(0);
        assert!(config.validate().is_err());
        
        config.num_threads = Some(4);
        config.batch_size = Some(0);
        assert!(config.validate().is_err());
        
        config.batch_size = Some(32);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_logging_config_validation() {
        let config = LoggingConfig::new();
        assert!(config.validate().is_ok());
        
        let mut config = LoggingConfig::new();
        config.log_to_file = true;
        config.log_file_path = None;
        assert!(config.validate().is_err());
        
        config.log_file_path = Some(PathBuf::from("test.log"));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_debug_config_validation() {
        let config = DebugConfig::new();
        assert!(config.validate().is_ok());
        
        let mut config = DebugConfig::new();
        config.save_intermediate_results = true;
        config.intermediate_results_dir = None;
        assert!(config.validate().is_err());
        
        config.intermediate_results_dir = Some(PathBuf::from("debug"));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_candle_options() {
        let cuda_opts = CandleOptions::cuda_optimized();
        assert!(cuda_opts.enable_cuda_graphs);
        assert!(cuda_opts.cuda_memory_pool_mb.is_some());
        
        let metal_opts = CandleOptions::metal_optimized();
        assert!(metal_opts.use_metal_performance_shaders);
        assert!(!metal_opts.enable_cuda_graphs);
    }

    #[test]
    fn test_onnx_options() {
        let cuda_opts = OnnxOptions::cuda();
        assert!(cuda_opts.execution_providers.contains(&"CUDAExecutionProvider".to_string()));
        assert_eq!(cuda_opts.graph_optimization_level, OnnxOptimizationLevel::All);
        
        let tensorrt_opts = OnnxOptions::tensorrt();
        assert!(tensorrt_opts.execution_providers.contains(&"TensorrtExecutionProvider".to_string()));
    }

    #[test]
    fn test_enum_string_representations() {
        assert_eq!(BackendType::Candle.as_str(), "candle");
        assert_eq!(BackendType::Onnx.as_str(), "onnx");
        assert_eq!(OptimizationLevel::Fast.as_str(), "fast");
        assert_eq!(LogLevel::Info.as_str(), "info");
        assert_eq!(OnnxOptimizationLevel::All.as_str(), "all");
    }
}