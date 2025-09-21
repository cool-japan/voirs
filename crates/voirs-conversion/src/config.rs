//! Configuration for voice conversion

use crate::types::ConversionType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for voice conversion
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConversionConfig {
    /// Default conversion type
    pub default_conversion_type: ConversionType,
    /// Output sample rate
    pub output_sample_rate: u32,
    /// Enable real-time processing
    pub enable_realtime: bool,
    /// Buffer size for real-time processing
    pub buffer_size: usize,
    /// Quality level (0.0 to 1.0)
    pub quality_level: f32,
    /// Use GPU acceleration
    pub use_gpu: bool,
    /// GPU device ID (for multi-GPU systems)
    pub gpu_device_id: usize,
    /// Enable mixed precision (FP16) for GPU
    pub use_mixed_precision: bool,
    /// Memory optimization level (0-3)
    pub memory_optimization_level: u8,
    /// Enable model quantization
    pub enable_quantization: bool,
    /// Quantization bits (8, 16)
    pub quantization_bits: u8,
    /// Maximum memory usage in MB (0 = unlimited)
    pub max_memory_mb: usize,
    /// Enable memory pooling
    pub enable_memory_pooling: bool,
    /// Thread pool size for CPU processing
    pub cpu_thread_count: usize,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Cache size for models in MB
    pub model_cache_size_mb: usize,
    /// Enable asynchronous processing
    pub enable_async_processing: bool,
    /// Batch size for processing
    pub batch_size: usize,
    /// Enable pipeline parallelism
    pub enable_pipeline_parallelism: bool,
    /// Custom performance parameters
    pub custom_params: HashMap<String, f32>,
}

impl ConversionConfig {
    /// Validate configuration
    pub fn validate(&self) -> crate::Result<()> {
        if self.output_sample_rate == 0 {
            return Err(crate::Error::validation(
                "Sample rate must be positive".to_string(),
            ));
        }
        if self.buffer_size == 0 {
            return Err(crate::Error::validation(
                "Buffer size must be positive".to_string(),
            ));
        }
        if self.quality_level < 0.0 || self.quality_level > 1.0 {
            return Err(crate::Error::validation(
                "Quality level must be between 0.0 and 1.0".to_string(),
            ));
        }
        if self.memory_optimization_level > 3 {
            return Err(crate::Error::validation(
                "Memory optimization level must be 0-3".to_string(),
            ));
        }
        if self.quantization_bits != 8 && self.quantization_bits != 16 {
            return Err(crate::Error::validation(
                "Quantization bits must be 8 or 16".to_string(),
            ));
        }
        if self.cpu_thread_count == 0 {
            return Err(crate::Error::validation(
                "CPU thread count must be positive".to_string(),
            ));
        }
        if self.batch_size == 0 {
            return Err(crate::Error::validation(
                "Batch size must be positive".to_string(),
            ));
        }
        Ok(())
    }

    /// Optimize configuration for low latency
    pub fn optimize_for_latency(mut self) -> Self {
        self.buffer_size = 256;
        self.quality_level = 0.6;
        self.memory_optimization_level = 2;
        self.enable_quantization = true;
        self.quantization_bits = 8;
        self.batch_size = 1;
        self.enable_pipeline_parallelism = false;
        self.use_mixed_precision = true;
        self
    }

    /// Optimize configuration for quality
    pub fn optimize_for_quality(mut self) -> Self {
        self.buffer_size = 2048;
        self.quality_level = 1.0;
        self.memory_optimization_level = 0;
        self.enable_quantization = false;
        self.use_mixed_precision = false;
        self.batch_size = 4;
        self.enable_pipeline_parallelism = true;
        self
    }

    /// Optimize configuration for memory usage
    pub fn optimize_for_memory(mut self) -> Self {
        self.memory_optimization_level = 3;
        self.enable_quantization = true;
        self.quantization_bits = 8;
        self.max_memory_mb = 256;
        self.enable_memory_pooling = true;
        self.model_cache_size_mb = 128;
        self.batch_size = 1;
        self.use_mixed_precision = true;
        self
    }

    /// Optimize configuration for GPU processing
    pub fn optimize_for_gpu(mut self) -> Self {
        self.use_gpu = true;
        self.use_mixed_precision = true;
        self.batch_size = 8;
        self.enable_async_processing = true;
        self.enable_pipeline_parallelism = true;
        self.memory_optimization_level = 1;
        self
    }

    /// Configure for specific hardware
    pub fn configure_for_hardware(mut self, hardware_info: &HardwareInfo) -> Self {
        // Adjust settings based on available hardware
        if hardware_info.has_gpu {
            self.use_gpu = true;
            self.gpu_device_id = hardware_info.best_gpu_id;

            if hardware_info.gpu_memory_gb >= 4.0 {
                self.batch_size = 8;
                self.use_mixed_precision = true;
            } else {
                self.batch_size = 2;
                self.enable_quantization = true;
            }
        }

        // CPU configuration
        self.cpu_thread_count = hardware_info.cpu_cores.min(8); // Cap at 8 threads

        // Memory configuration
        let available_memory_mb = (hardware_info.available_memory_gb * 1024.0) as usize;
        self.max_memory_mb = (available_memory_mb / 4).max(512); // Use 25% of available memory
        self.model_cache_size_mb = (available_memory_mb / 8).clamp(128, 1024);

        // Enable SIMD based on CPU features
        self.enable_simd = hardware_info.cpu_features.has_avx2;

        self
    }

    /// Estimate memory usage
    pub fn estimate_memory_usage_mb(&self) -> usize {
        let mut memory_usage = 0;

        // Base memory for buffers
        memory_usage += (self.buffer_size * self.batch_size * 4) / (1024 * 1024); // 4 bytes per f32

        // Model cache
        memory_usage += self.model_cache_size_mb;

        // Additional overhead based on optimization level
        memory_usage += match self.memory_optimization_level {
            0 => 512, // No optimization, more memory for quality
            1 => 256, // Balanced
            2 => 128, // Optimized
            3 => 64,  // Maximum optimization
            _ => 256,
        };

        memory_usage
    }

    /// Check if configuration is valid
    pub fn is_valid(&self) -> bool {
        self.output_sample_rate > 0
            && self.buffer_size > 0
            && self.quality_level >= 0.0
            && self.quality_level <= 1.0
            && self.memory_optimization_level <= 3
            && (self.quantization_bits == 8 || self.quantization_bits == 16)
            && self.cpu_thread_count > 0
            && self.batch_size > 0
    }
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self {
            default_conversion_type: ConversionType::SpeakerConversion,
            output_sample_rate: 22050,
            enable_realtime: true,
            buffer_size: 1024,
            quality_level: 0.8,
            use_gpu: false,
            gpu_device_id: 0,
            use_mixed_precision: false,
            memory_optimization_level: 1,
            enable_quantization: false,
            quantization_bits: 8,
            max_memory_mb: 0,
            enable_memory_pooling: true,
            cpu_thread_count: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            enable_simd: true,
            model_cache_size_mb: 512,
            enable_async_processing: true,
            batch_size: 1,
            enable_pipeline_parallelism: true,
            custom_params: HashMap::new(),
        }
    }
}

/// Hardware information for configuration optimization
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// Available memory in GB
    pub available_memory_gb: f32,
    /// Whether GPU is available
    pub has_gpu: bool,
    /// GPU memory in GB
    pub gpu_memory_gb: f32,
    /// Best GPU device ID
    pub best_gpu_id: usize,
    /// Whether GPU supports mixed precision
    pub supports_mixed_precision: bool,
    /// CPU architecture features
    pub cpu_features: CpuFeatures,
}

/// CPU feature detection
#[derive(Debug, Clone, Default)]
pub struct CpuFeatures {
    /// AVX support
    pub has_avx: bool,
    /// AVX2 support  
    pub has_avx2: bool,
    /// FMA support
    pub has_fma: bool,
    /// SSE4.1 support
    pub has_sse41: bool,
}

impl HardwareInfo {
    /// Detect current hardware capabilities
    pub fn detect() -> Self {
        Self {
            cpu_cores: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            available_memory_gb: Self::get_available_memory_gb(),
            has_gpu: Self::detect_gpu(),
            gpu_memory_gb: Self::get_gpu_memory_gb(),
            best_gpu_id: 0,
            supports_mixed_precision: Self::check_mixed_precision_support(),
            cpu_features: CpuFeatures::detect(),
        }
    }

    fn get_available_memory_gb() -> f32 {
        // Simple estimate - in practice would use system APIs
        8.0 // Default to 8GB
    }

    fn detect_gpu() -> bool {
        // In practice would check for CUDA/OpenCL availability
        cfg!(feature = "gpu")
    }

    fn get_gpu_memory_gb() -> f32 {
        // In practice would query GPU memory
        if Self::detect_gpu() {
            4.0
        } else {
            0.0
        }
    }

    fn check_mixed_precision_support() -> bool {
        // In practice would check GPU capabilities
        Self::detect_gpu()
    }
}

impl CpuFeatures {
    /// Detect CPU features
    pub fn detect() -> Self {
        Self {
            has_avx: Self::has_avx(),
            has_avx2: Self::has_avx2(),
            has_fma: Self::has_fma(),
            has_sse41: Self::has_sse41(),
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn has_avx() -> bool {
        is_x86_feature_detected!("avx")
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn has_avx() -> bool {
        false
    }

    #[cfg(target_arch = "x86_64")]
    fn has_avx2() -> bool {
        is_x86_feature_detected!("avx2")
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn has_avx2() -> bool {
        false
    }

    #[cfg(target_arch = "x86_64")]
    fn has_fma() -> bool {
        is_x86_feature_detected!("fma")
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn has_fma() -> bool {
        false
    }

    #[cfg(target_arch = "x86_64")]
    fn has_sse41() -> bool {
        is_x86_feature_detected!("sse4.1")
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn has_sse41() -> bool {
        false
    }
}

/// Performance optimization presets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformancePreset {
    /// Maximum quality, highest resource usage
    MaxQuality,
    /// Balanced quality and performance
    Balanced,
    /// Low latency, optimized for real-time
    LowLatency,
    /// Memory optimized for resource-constrained environments
    MemoryOptimized,
    /// GPU optimized for high-performance systems
    GpuOptimized,
    /// Custom configuration
    Custom,
}

/// Builder for ConversionConfig
#[derive(Debug, Clone)]
pub struct ConversionConfigBuilder {
    config: ConversionConfig,
}

impl ConversionConfigBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: ConversionConfig::default(),
        }
    }

    /// Create builder with performance preset
    pub fn with_preset(preset: PerformancePreset) -> Self {
        let mut builder = Self::new();
        builder.config = match preset {
            PerformancePreset::MaxQuality => ConversionConfig::default().optimize_for_quality(),
            PerformancePreset::Balanced => ConversionConfig::default(),
            PerformancePreset::LowLatency => ConversionConfig::default().optimize_for_latency(),
            PerformancePreset::MemoryOptimized => ConversionConfig::default().optimize_for_memory(),
            PerformancePreset::GpuOptimized => ConversionConfig::default().optimize_for_gpu(),
            PerformancePreset::Custom => ConversionConfig::default(),
        };
        builder
    }

    /// Auto-configure based on hardware
    pub fn auto_configure(mut self) -> Self {
        let hardware_info = HardwareInfo::detect();
        self.config = self.config.configure_for_hardware(&hardware_info);
        self
    }

    /// Set default conversion type
    pub fn default_conversion_type(mut self, conversion_type: ConversionType) -> Self {
        self.config.default_conversion_type = conversion_type;
        self
    }

    /// Set output sample rate
    pub fn output_sample_rate(mut self, sample_rate: u32) -> Self {
        self.config.output_sample_rate = sample_rate;
        self
    }

    /// Enable real-time processing
    pub fn enable_realtime(mut self, enable: bool) -> Self {
        self.config.enable_realtime = enable;
        self
    }

    /// Set buffer size
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.config.buffer_size = size;
        self
    }

    /// Set quality level
    pub fn quality_level(mut self, level: f32) -> Self {
        self.config.quality_level = level.clamp(0.0, 1.0);
        self
    }

    /// Enable GPU acceleration
    pub fn use_gpu(mut self, enable: bool) -> Self {
        self.config.use_gpu = enable;
        self
    }

    /// Set GPU device ID
    pub fn gpu_device_id(mut self, device_id: usize) -> Self {
        self.config.gpu_device_id = device_id;
        self
    }

    /// Enable mixed precision
    pub fn mixed_precision(mut self, enable: bool) -> Self {
        self.config.use_mixed_precision = enable;
        self
    }

    /// Set memory optimization level
    pub fn memory_optimization_level(mut self, level: u8) -> Self {
        self.config.memory_optimization_level = level.min(3);
        self
    }

    /// Enable quantization
    pub fn quantization(mut self, enable: bool, bits: u8) -> Self {
        self.config.enable_quantization = enable;
        self.config.quantization_bits = if bits == 16 { 16 } else { 8 };
        self
    }

    /// Set maximum memory usage
    pub fn max_memory_mb(mut self, memory_mb: usize) -> Self {
        self.config.max_memory_mb = memory_mb;
        self
    }

    /// Set CPU thread count
    pub fn cpu_threads(mut self, count: usize) -> Self {
        self.config.cpu_thread_count = count.max(1);
        self
    }

    /// Enable SIMD optimizations
    pub fn simd(mut self, enable: bool) -> Self {
        self.config.enable_simd = enable;
        self
    }

    /// Set batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size.max(1);
        self
    }

    /// Add custom parameter
    pub fn custom_param(mut self, key: String, value: f32) -> Self {
        self.config.custom_params.insert(key, value);
        self
    }

    /// Build configuration
    pub fn build(self) -> crate::Result<ConversionConfig> {
        self.config.validate()?;
        Ok(self.config)
    }

    /// Build without validation
    pub fn build_unchecked(self) -> ConversionConfig {
        self.config
    }
}

impl Default for ConversionConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let config = ConversionConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid_config = config.clone();
        invalid_config.output_sample_rate = 0;
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_performance_presets() {
        let latency_config = ConversionConfigBuilder::with_preset(PerformancePreset::LowLatency)
            .build()
            .unwrap();
        assert!(latency_config.buffer_size < ConversionConfig::default().buffer_size);

        let quality_config = ConversionConfigBuilder::with_preset(PerformancePreset::MaxQuality)
            .build()
            .unwrap();
        assert!(quality_config.quality_level >= 1.0);
    }

    #[test]
    fn test_hardware_detection() {
        let hardware = HardwareInfo::detect();
        assert!(hardware.cpu_cores > 0);
        assert!(hardware.available_memory_gb > 0.0);
    }

    #[test]
    fn test_memory_estimation() {
        let config = ConversionConfig::default();
        let memory_usage = config.estimate_memory_usage_mb();
        assert!(memory_usage > 0);
    }
}
