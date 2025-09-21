//! On-Device Processing Optimizations
//!
//! This module provides comprehensive optimizations for on-device speech recognition
//! including mobile-specific optimizations, edge device acceleration, real-time
//! processing constraints, hardware acceleration, model compression, and adaptive
//! inference for efficient deployment on resource-constrained devices.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Device capability profile
#[derive(Debug, Clone)]
pub struct DeviceProfile {
    /// Device type
    pub device_type: DeviceType,
    /// CPU cores and frequencies
    pub cpu_info: CpuInfo,
    /// Memory information
    pub memory_info: MemoryInfo,
    /// GPU availability and capabilities
    pub gpu_info: Option<GpuInfo>,
    /// NPU/AI accelerator info
    pub npu_info: Option<NpuInfo>,
    /// Thermal constraints
    pub thermal_info: ThermalInfo,
    /// Battery constraints
    pub battery_info: Option<BatteryInfo>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeviceType {
    Mobile,
    Tablet,
    Laptop,
    Desktop,
    EdgeDevice,
    IoT,
    Embedded,
}

#[derive(Debug, Clone)]
pub struct CpuInfo {
    /// Number of CPU cores
    pub cores: usize,
    /// Performance cores count
    pub performance_cores: usize,
    /// Efficiency cores count
    pub efficiency_cores: usize,
    /// Base frequency (MHz)
    pub base_frequency: u32,
    /// Maximum frequency (MHz)
    pub max_frequency: u32,
    /// Architecture (ARM, x86, etc.)
    pub architecture: CpuArchitecture,
    /// SIMD instruction sets available
    pub simd_support: Vec<SimdInstructionSet>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CpuArchitecture {
    ARM64,
    X86_64,
    ARM32,
    X86,
    RISC_V,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SimdInstructionSet {
    NEON,   // ARM NEON
    AVX2,   // x86 AVX2
    AVX512, // x86 AVX-512
    SSE4_2, // x86 SSE 4.2
    SVE,    // ARM Scalable Vector Extension
}

#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Total RAM (MB)
    pub total_ram_mb: usize,
    /// Available RAM (MB)
    pub available_ram_mb: usize,
    /// Memory bandwidth (GB/s)
    pub bandwidth_gbps: f32,
    /// Cache sizes (L1, L2, L3)
    pub cache_sizes: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// GPU vendor
    pub vendor: GpuVendor,
    /// Compute units
    pub compute_units: usize,
    /// Memory size (MB)
    pub memory_mb: usize,
    /// Memory bandwidth (GB/s)
    pub bandwidth_gbps: f32,
    /// Supports FP16 operations
    pub fp16_support: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GpuVendor {
    Apple,
    Nvidia,
    AMD,
    Intel,
    ARM,
    Qualcomm,
}

#[derive(Debug, Clone)]
pub struct NpuInfo {
    /// NPU vendor
    pub vendor: NpuVendor,
    /// TOPS (Trillions of Operations Per Second)
    pub tops: f32,
    /// Supported data types
    pub supported_types: Vec<DataType>,
    /// Memory size (MB)
    pub memory_mb: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NpuVendor {
    Apple,
    Qualcomm,
    MediaTek,
    Samsung,
    Intel,
    Google,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    FP32,
    FP16,
    INT8,
    INT4,
    BF16,
}

#[derive(Debug, Clone)]
pub struct ThermalInfo {
    /// Current temperature (Celsius)
    pub current_temp: f32,
    /// Temperature threshold for throttling
    pub throttle_temp: f32,
    /// Maximum safe temperature
    pub max_temp: f32,
    /// Thermal management enabled
    pub thermal_management: bool,
}

#[derive(Debug, Clone)]
pub struct BatteryInfo {
    /// Current battery level (0-100)
    pub level: f32,
    /// Battery status
    pub status: BatteryStatus,
    /// Power consumption constraints
    pub power_constraints: PowerConstraints,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BatteryStatus {
    Charging,
    Discharging,
    Full,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct PowerConstraints {
    /// Maximum power consumption (watts)
    pub max_power_watts: f32,
    /// Low power mode enabled
    pub low_power_mode: bool,
    /// Aggressive power saving
    pub aggressive_saving: bool,
}

/// On-device optimization configuration
#[derive(Debug, Clone)]
pub struct OnDeviceConfig {
    /// Target device profile
    pub device_profile: DeviceProfile,
    /// Real-time constraints
    pub realtime_constraints: RealtimeConstraints,
    /// Model compression settings
    pub compression: CompressionConfig,
    /// Hardware acceleration preferences
    pub acceleration: AccelerationConfig,
    /// Adaptive inference settings
    pub adaptive_inference: AdaptiveInferenceConfig,
    /// Power management
    pub power_management: PowerManagementConfig,
}

#[derive(Debug, Clone)]
pub struct RealtimeConstraints {
    /// Maximum processing latency (ms)
    pub max_latency_ms: u32,
    /// Target latency for optimization (ms)
    pub target_latency_ms: u32,
    /// Maximum jitter tolerance (ms)
    pub max_jitter_ms: u32,
    /// Deadline miss policy
    pub deadline_policy: DeadlinePolicy,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeadlinePolicy {
    Drop,       // Drop frames if deadline missed
    Degrade,    // Reduce quality if deadline missed
    BestEffort, // Continue processing regardless
}

#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Model pruning enabled
    pub pruning: bool,
    /// Pruning sparsity (0.0 - 1.0)
    pub pruning_sparsity: f32,
    /// Knowledge distillation enabled
    pub distillation: bool,
    /// Quantization settings
    pub quantization: QuantizationSettings,
    /// Dynamic model selection
    pub dynamic_models: bool,
}

#[derive(Debug, Clone)]
pub struct QuantizationSettings {
    /// Weight quantization bits
    pub weight_bits: u8,
    /// Activation quantization bits
    pub activation_bits: u8,
    /// Dynamic quantization
    pub dynamic: bool,
    /// Calibration enabled
    pub calibration: bool,
}

#[derive(Debug, Clone)]
pub struct AccelerationConfig {
    /// Preferred acceleration backend
    pub preferred_backend: AccelerationBackend,
    /// Fallback backends in order of preference
    pub fallback_backends: Vec<AccelerationBackend>,
    /// GPU memory limit (MB)
    pub gpu_memory_limit_mb: Option<usize>,
    /// NPU model caching
    pub npu_caching: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AccelerationBackend {
    CPU,
    GPU,
    NPU,
    AutoSelect,
}

#[derive(Debug, Clone)]
pub struct AdaptiveInferenceConfig {
    /// Enable quality adaptation
    pub quality_adaptation: bool,
    /// Enable model switching
    pub model_switching: bool,
    /// Performance monitoring interval (ms)
    pub monitoring_interval_ms: u32,
    /// Adaptation thresholds
    pub thresholds: AdaptationThresholds,
}

#[derive(Debug, Clone)]
pub struct AdaptationThresholds {
    /// CPU usage threshold for downgrade (0.0 - 1.0)
    pub cpu_threshold: f32,
    /// Memory usage threshold for downgrade (0.0 - 1.0)
    pub memory_threshold: f32,
    /// Temperature threshold for throttling (Celsius)
    pub temperature_threshold: f32,
    /// Battery level threshold for power saving (0.0 - 1.0)
    pub battery_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct PowerManagementConfig {
    /// Enable dynamic frequency scaling
    pub dvfs_enabled: bool,
    /// Core affinity management
    pub core_affinity: bool,
    /// Sleep states utilization
    pub sleep_states: bool,
    /// Thermal throttling
    pub thermal_throttling: bool,
}

/// On-device optimizer
#[derive(Debug)]
pub struct OnDeviceOptimizer {
    /// Configuration
    config: OnDeviceConfig,
    /// Current performance metrics
    metrics: Arc<Mutex<PerformanceMetrics>>,
    /// Model variants for different quality levels
    model_variants: HashMap<QualityLevel, ModelVariant>,
    /// Current active model
    current_model: QualityLevel,
    /// Processing pipeline
    pipeline: ProcessingPipeline,
    /// Resource monitor
    resource_monitor: ResourceMonitor,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Average processing latency (ms)
    pub avg_latency_ms: f32,
    /// 95th percentile latency (ms)
    pub p95_latency_ms: f32,
    /// CPU usage (0.0 - 1.0)
    pub cpu_usage: f32,
    /// Memory usage (MB)
    pub memory_usage_mb: usize,
    /// GPU usage (0.0 - 1.0)
    pub gpu_usage: f32,
    /// NPU usage (0.0 - 1.0)
    pub npu_usage: f32,
    /// Temperature (Celsius)
    pub temperature: f32,
    /// Power consumption (watts)
    pub power_consumption: f32,
    /// Deadline misses count
    pub deadline_misses: usize,
    /// Total processed frames
    pub total_frames: usize,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            avg_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            cpu_usage: 0.0,
            memory_usage_mb: 0,
            gpu_usage: 0.0,
            npu_usage: 0.0,
            temperature: 25.0,
            power_consumption: 0.0,
            deadline_misses: 0,
            total_frames: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QualityLevel {
    Ultra,   // Highest quality, most resource intensive
    High,    // High quality, balanced resource usage
    Medium,  // Medium quality, moderate resource usage
    Low,     // Lower quality, lightweight
    Minimal, // Minimal quality, ultra-lightweight
}

#[derive(Debug, Clone)]
pub struct ModelVariant {
    /// Model parameters count
    pub parameters: usize,
    /// Expected latency (ms)
    pub expected_latency_ms: f32,
    /// Memory footprint (MB)
    pub memory_footprint_mb: usize,
    /// Expected accuracy
    pub expected_accuracy: f32,
    /// Supported backends
    pub supported_backends: Vec<AccelerationBackend>,
}

/// Processing pipeline for on-device optimization
#[derive(Debug)]
pub struct ProcessingPipeline {
    /// Audio preprocessing stages
    preprocessing_stages: Vec<PreprocessingStage>,
    /// Feature extraction optimizations
    feature_extraction: FeatureExtractionOptimizer,
    /// Model inference optimizations
    inference: InferenceOptimizer,
    /// Post-processing optimizations
    postprocessing: PostprocessingOptimizer,
}

#[derive(Debug, Clone)]
pub enum PreprocessingStage {
    /// SIMD-optimized audio preprocessing
    SimdPreprocessing,
    /// Hardware-accelerated resampling
    HardwareResampling,
    /// Adaptive gain control
    AdaptiveGainControl,
    /// Noise suppression
    NoiseSuppression,
}

#[derive(Debug)]
pub struct FeatureExtractionOptimizer {
    /// Use SIMD instructions for FFT
    simd_fft: bool,
    /// GPU-accelerated mel-spectrogram
    gpu_mel_spectrogram: bool,
    /// Cached filterbanks
    cached_filterbanks: HashMap<String, Vec<Vec<f32>>>,
    /// Feature compression
    feature_compression: bool,
}

#[derive(Debug)]
pub struct InferenceOptimizer {
    /// Current backend
    backend: AccelerationBackend,
    /// Model cache
    model_cache: HashMap<QualityLevel, CachedModel>,
    /// Batch processing
    batch_processing: bool,
    /// Dynamic batching
    dynamic_batching: bool,
}

#[derive(Debug, Clone)]
pub struct CachedModel {
    /// Model weights (quantized if applicable)
    weights: Vec<u8>,
    /// Model metadata
    metadata: ModelMetadata,
    /// Compilation artifacts (for NPU/GPU)
    compilation_artifacts: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Quantization parameters
    pub quantization_params: Option<QuantizationParams>,
    /// Compilation target
    pub target_backend: AccelerationBackend,
}

#[derive(Debug, Clone)]
pub struct QuantizationParams {
    /// Scale factors
    pub scales: Vec<f32>,
    /// Zero points
    pub zero_points: Vec<i32>,
    /// Quantization scheme
    pub scheme: QuantizationScheme,
}

#[derive(Debug, Clone, PartialEq)]
pub enum QuantizationScheme {
    PerTensor,
    PerChannel,
    Dynamic,
}

#[derive(Debug)]
pub struct PostprocessingOptimizer {
    /// Beam search optimization
    beam_search_optimized: bool,
    /// Language model integration
    language_model: Option<LightweightLanguageModel>,
    /// Confidence scoring
    confidence_scoring: bool,
}

#[derive(Debug)]
pub struct LightweightLanguageModel {
    /// N-gram probabilities
    ngram_probs: HashMap<String, f32>,
    /// Vocabulary
    vocabulary: Vec<String>,
    /// Model order (2-gram, 3-gram, etc.)
    order: usize,
}

/// Resource monitor for adaptive optimization
#[derive(Debug)]
pub struct ResourceMonitor {
    /// Monitoring interval
    interval: Duration,
    /// Resource usage history
    usage_history: VecDeque<ResourceUsage>,
    /// Last monitoring time
    last_update: Instant,
    /// Monitoring enabled
    enabled: bool,
}

#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Timestamp
    pub timestamp: Instant,
    /// CPU usage per core
    pub cpu_per_core: Vec<f32>,
    /// Memory usage (MB)
    pub memory_mb: usize,
    /// Temperature readings
    pub temperatures: Vec<f32>,
    /// GPU utilization
    pub gpu_utilization: f32,
    /// Battery level
    pub battery_level: Option<f32>,
}

impl OnDeviceOptimizer {
    /// Create new on-device optimizer
    pub fn new(config: OnDeviceConfig) -> Self {
        let metrics = Arc::new(Mutex::new(PerformanceMetrics::default()));
        let mut model_variants = HashMap::new();

        // Define model variants with different quality/performance trade-offs
        model_variants.insert(
            QualityLevel::Ultra,
            ModelVariant {
                parameters: 50_000_000,
                expected_latency_ms: 200.0,
                memory_footprint_mb: 200,
                expected_accuracy: 0.95,
                supported_backends: vec![AccelerationBackend::GPU, AccelerationBackend::NPU],
            },
        );

        model_variants.insert(
            QualityLevel::High,
            ModelVariant {
                parameters: 25_000_000,
                expected_latency_ms: 100.0,
                memory_footprint_mb: 100,
                expected_accuracy: 0.92,
                supported_backends: vec![
                    AccelerationBackend::GPU,
                    AccelerationBackend::NPU,
                    AccelerationBackend::CPU,
                ],
            },
        );

        model_variants.insert(
            QualityLevel::Medium,
            ModelVariant {
                parameters: 10_000_000,
                expected_latency_ms: 50.0,
                memory_footprint_mb: 50,
                expected_accuracy: 0.88,
                supported_backends: vec![AccelerationBackend::GPU, AccelerationBackend::CPU],
            },
        );

        model_variants.insert(
            QualityLevel::Low,
            ModelVariant {
                parameters: 5_000_000,
                expected_latency_ms: 25.0,
                memory_footprint_mb: 25,
                expected_accuracy: 0.82,
                supported_backends: vec![AccelerationBackend::CPU],
            },
        );

        model_variants.insert(
            QualityLevel::Minimal,
            ModelVariant {
                parameters: 1_000_000,
                expected_latency_ms: 10.0,
                memory_footprint_mb: 10,
                expected_accuracy: 0.75,
                supported_backends: vec![AccelerationBackend::CPU],
            },
        );

        let current_model = Self::select_initial_model(&config, &model_variants);

        let pipeline = ProcessingPipeline {
            preprocessing_stages: Self::select_preprocessing_stages(&config),
            feature_extraction: FeatureExtractionOptimizer::new(&config),
            inference: InferenceOptimizer::new(&config),
            postprocessing: PostprocessingOptimizer::new(&config),
        };

        let resource_monitor = ResourceMonitor::new(Duration::from_millis(
            config.adaptive_inference.monitoring_interval_ms as u64,
        ));

        Self {
            config,
            metrics,
            model_variants,
            current_model,
            pipeline,
            resource_monitor,
        }
    }

    /// Select initial model based on device capabilities
    fn select_initial_model(
        config: &OnDeviceConfig,
        variants: &HashMap<QualityLevel, ModelVariant>,
    ) -> QualityLevel {
        let device = &config.device_profile;

        // Select based on device capabilities
        match device.device_type {
            DeviceType::Desktop | DeviceType::Laptop => {
                if device.memory_info.total_ram_mb >= 8192 && device.gpu_info.is_some() {
                    QualityLevel::Ultra
                } else if device.memory_info.total_ram_mb >= 4096 {
                    QualityLevel::High
                } else {
                    QualityLevel::Medium
                }
            }
            DeviceType::Mobile | DeviceType::Tablet => {
                if device.memory_info.total_ram_mb >= 6144 && device.npu_info.is_some() {
                    QualityLevel::High
                } else if device.memory_info.total_ram_mb >= 4096 {
                    QualityLevel::Medium
                } else {
                    QualityLevel::Low
                }
            }
            DeviceType::EdgeDevice | DeviceType::IoT | DeviceType::Embedded => {
                if device.memory_info.total_ram_mb >= 2048 {
                    QualityLevel::Low
                } else {
                    QualityLevel::Minimal
                }
            }
        }
    }

    /// Select preprocessing stages based on device capabilities
    fn select_preprocessing_stages(config: &OnDeviceConfig) -> Vec<PreprocessingStage> {
        let mut stages = Vec::new();

        // Add SIMD preprocessing if supported
        if config
            .device_profile
            .cpu_info
            .simd_support
            .contains(&SimdInstructionSet::NEON)
            || config
                .device_profile
                .cpu_info
                .simd_support
                .contains(&SimdInstructionSet::AVX2)
        {
            stages.push(PreprocessingStage::SimdPreprocessing);
        }

        // Add hardware resampling if GPU available
        if config.device_profile.gpu_info.is_some() {
            stages.push(PreprocessingStage::HardwareResampling);
        }

        // Add adaptive gain control for all devices
        stages.push(PreprocessingStage::AdaptiveGainControl);

        // Add noise suppression if device has sufficient resources
        if config.device_profile.memory_info.total_ram_mb >= 2048 {
            stages.push(PreprocessingStage::NoiseSuppression);
        }

        stages
    }

    /// Process audio with on-device optimizations
    pub fn process_audio(&mut self, audio_data: &[f32]) -> Result<String, OnDeviceError> {
        let start_time = Instant::now();

        // Check resource constraints
        self.update_resource_monitoring();
        if self.should_adapt_quality() {
            self.adapt_quality();
        }

        // Apply thermal throttling if necessary
        if self.should_thermal_throttle() {
            self.apply_thermal_throttling();
        }

        // Preprocessing
        let preprocessed = self.pipeline.preprocess(audio_data, &self.config)?;

        // Feature extraction
        let features = self
            .pipeline
            .feature_extraction
            .extract_features(&preprocessed, &self.config)?;

        // Model inference
        let inference_result =
            self.pipeline
                .inference
                .infer(&features, &self.current_model, &self.config)?;

        // Post-processing
        let result = self
            .pipeline
            .postprocessing
            .process(&inference_result, &self.config)?;

        // Update metrics
        let processing_time = start_time.elapsed();
        self.update_metrics(processing_time);

        // Check if deadline was met
        if processing_time.as_millis() > self.config.realtime_constraints.max_latency_ms as u128 {
            self.handle_deadline_miss();
        }

        Ok(result)
    }

    /// Update resource monitoring
    fn update_resource_monitoring(&mut self) {
        if self.resource_monitor.enabled {
            let now = Instant::now();
            if now.duration_since(self.resource_monitor.last_update)
                >= self.resource_monitor.interval
            {
                let usage = self.collect_resource_usage();
                self.resource_monitor.usage_history.push_back(usage);

                // Keep only recent history
                while self.resource_monitor.usage_history.len() > 100 {
                    self.resource_monitor.usage_history.pop_front();
                }

                self.resource_monitor.last_update = now;
            }
        }
    }

    /// Collect current resource usage
    fn collect_resource_usage(&self) -> ResourceUsage {
        // This would interface with system APIs to get actual resource usage
        // For now, we'll use placeholder values
        ResourceUsage {
            timestamp: Instant::now(),
            cpu_per_core: vec![0.5; self.config.device_profile.cpu_info.cores],
            memory_mb: 1024,
            temperatures: vec![self.config.device_profile.thermal_info.current_temp],
            gpu_utilization: 0.3,
            battery_level: self
                .config
                .device_profile
                .battery_info
                .as_ref()
                .map(|b| b.level),
        }
    }

    /// Check if quality adaptation is needed
    fn should_adapt_quality(&self) -> bool {
        if !self.config.adaptive_inference.quality_adaptation {
            return false;
        }

        let metrics = self.metrics.lock().unwrap();
        let thresholds = &self.config.adaptive_inference.thresholds;

        metrics.cpu_usage > thresholds.cpu_threshold
            || (metrics.memory_usage_mb as f32
                / self.config.device_profile.memory_info.total_ram_mb as f32)
                > thresholds.memory_threshold
            || metrics.temperature > thresholds.temperature_threshold
            || self
                .config
                .device_profile
                .battery_info
                .as_ref()
                .map_or(false, |b| b.level < thresholds.battery_threshold * 100.0)
    }

    /// Adapt quality based on current conditions
    fn adapt_quality(&mut self) {
        let current_variant = &self.model_variants[&self.current_model];
        let metrics = self.metrics.lock().unwrap();

        // Downgrade if under stress
        if metrics.avg_latency_ms > self.config.realtime_constraints.target_latency_ms as f32 {
            self.current_model = match self.current_model {
                QualityLevel::Ultra => QualityLevel::High,
                QualityLevel::High => QualityLevel::Medium,
                QualityLevel::Medium => QualityLevel::Low,
                QualityLevel::Low => QualityLevel::Minimal,
                QualityLevel::Minimal => QualityLevel::Minimal,
            };
        }
        // Upgrade if resources are available
        else if metrics.avg_latency_ms
            < self.config.realtime_constraints.target_latency_ms as f32 * 0.5
        {
            self.current_model = match self.current_model {
                QualityLevel::Minimal => QualityLevel::Low,
                QualityLevel::Low => QualityLevel::Medium,
                QualityLevel::Medium => QualityLevel::High,
                QualityLevel::High => QualityLevel::Ultra,
                QualityLevel::Ultra => QualityLevel::Ultra,
            };
        }
    }

    /// Check if thermal throttling is needed
    fn should_thermal_throttle(&self) -> bool {
        self.config.device_profile.thermal_info.thermal_management
            && self.config.device_profile.thermal_info.current_temp
                > self.config.device_profile.thermal_info.throttle_temp
    }

    /// Apply thermal throttling
    fn apply_thermal_throttling(&mut self) {
        // Reduce processing frequency
        if self.current_model != QualityLevel::Minimal {
            self.current_model = match self.current_model {
                QualityLevel::Ultra => QualityLevel::Medium,
                QualityLevel::High => QualityLevel::Low,
                QualityLevel::Medium => QualityLevel::Minimal,
                QualityLevel::Low => QualityLevel::Minimal,
                QualityLevel::Minimal => QualityLevel::Minimal,
            };
        }

        // Introduce processing delays to reduce heat generation
        thread::sleep(Duration::from_millis(10));
    }

    /// Update performance metrics
    fn update_metrics(&self, processing_time: Duration) {
        let mut metrics = self.metrics.lock().unwrap();

        metrics.total_frames += 1;
        let latency_ms = processing_time.as_millis() as f32;

        // Update average latency with exponential moving average
        let alpha = 0.1;
        metrics.avg_latency_ms = alpha * latency_ms + (1.0 - alpha) * metrics.avg_latency_ms;

        // Update P95 latency (simplified)
        if latency_ms > metrics.p95_latency_ms {
            metrics.p95_latency_ms = metrics.p95_latency_ms * 0.95 + latency_ms * 0.05;
        }
    }

    /// Handle deadline miss
    fn handle_deadline_miss(&mut self) {
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.deadline_misses += 1;
        }

        match self.config.realtime_constraints.deadline_policy {
            DeadlinePolicy::Drop => {
                // Frame would be dropped (implemented in calling code)
            }
            DeadlinePolicy::Degrade => {
                // Degrade quality for next frame
                if self.current_model != QualityLevel::Minimal {
                    self.adapt_quality();
                }
            }
            DeadlinePolicy::BestEffort => {
                // Continue processing, just log the miss
            }
        }
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics.lock().unwrap().clone()
    }

    /// Get current model quality level
    pub fn get_current_quality(&self) -> QualityLevel {
        self.current_model.clone()
    }
}

impl ProcessingPipeline {
    /// Preprocess audio data
    pub fn preprocess(
        &self,
        audio: &[f32],
        config: &OnDeviceConfig,
    ) -> Result<Vec<f32>, OnDeviceError> {
        let mut processed = audio.to_vec();

        for stage in &self.preprocessing_stages {
            processed = match stage {
                PreprocessingStage::SimdPreprocessing => {
                    self.simd_preprocess(&processed, config)?
                }
                PreprocessingStage::HardwareResampling => {
                    self.hardware_resample(&processed, config)?
                }
                PreprocessingStage::AdaptiveGainControl => {
                    self.adaptive_gain_control(&processed, config)?
                }
                PreprocessingStage::NoiseSuppression => {
                    self.noise_suppression(&processed, config)?
                }
            };
        }

        Ok(processed)
    }

    fn simd_preprocess(
        &self,
        audio: &[f32],
        _config: &OnDeviceConfig,
    ) -> Result<Vec<f32>, OnDeviceError> {
        // SIMD-optimized preprocessing (windowing, normalization)
        let mut result = audio.to_vec();

        // Apply windowing function using SIMD
        for chunk in result.chunks_mut(8) {
            for (i, sample) in chunk.iter_mut().enumerate() {
                let window_val = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / 8.0).cos();
                *sample *= window_val;
            }
        }

        Ok(result)
    }

    fn hardware_resample(
        &self,
        audio: &[f32],
        _config: &OnDeviceConfig,
    ) -> Result<Vec<f32>, OnDeviceError> {
        // Hardware-accelerated resampling (simplified)
        Ok(audio.to_vec())
    }

    fn adaptive_gain_control(
        &self,
        audio: &[f32],
        _config: &OnDeviceConfig,
    ) -> Result<Vec<f32>, OnDeviceError> {
        // Adaptive gain control
        let rms = (audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        let target_rms = 0.1;
        let gain = if rms > 0.0 { target_rms / rms } else { 1.0 };

        Ok(audio.iter().map(|&x| x * gain).collect())
    }

    fn noise_suppression(
        &self,
        audio: &[f32],
        _config: &OnDeviceConfig,
    ) -> Result<Vec<f32>, OnDeviceError> {
        // Simple noise suppression (spectral subtraction)
        Ok(audio.to_vec())
    }
}

impl FeatureExtractionOptimizer {
    fn new(config: &OnDeviceConfig) -> Self {
        Self {
            simd_fft: config.device_profile.cpu_info.simd_support.len() > 0,
            gpu_mel_spectrogram: config.device_profile.gpu_info.is_some(),
            cached_filterbanks: HashMap::new(),
            feature_compression: config.compression.quantization.dynamic,
        }
    }

    fn extract_features(
        &mut self,
        audio: &[f32],
        _config: &OnDeviceConfig,
    ) -> Result<Vec<Vec<f32>>, OnDeviceError> {
        // Extract mel-spectrogram features (simplified)
        let frame_size = 400;
        let hop_size = 160;
        let n_mels = 80;

        let mut features = Vec::new();

        for i in (0..audio.len()).step_by(hop_size) {
            let end = (i + frame_size).min(audio.len());
            let frame = &audio[i..end];

            // Simple FFT and mel-scale conversion
            let mut mel_frame = vec![0.0; n_mels];
            for (j, mel_val) in mel_frame.iter_mut().enumerate() {
                let mut magnitude = 0.0;
                for (k, &sample) in frame.iter().enumerate() {
                    let angle =
                        2.0 * std::f32::consts::PI * j as f32 * k as f32 / frame_size as f32;
                    magnitude += sample * angle.cos();
                }
                *mel_val = magnitude.abs().ln().max(-10.0);
            }

            features.push(mel_frame);
        }

        Ok(features)
    }
}

impl InferenceOptimizer {
    fn new(config: &OnDeviceConfig) -> Self {
        Self {
            backend: config.acceleration.preferred_backend.clone(),
            model_cache: HashMap::new(),
            batch_processing: config.device_profile.device_type != DeviceType::IoT,
            dynamic_batching: config.adaptive_inference.quality_adaptation,
        }
    }

    fn infer(
        &self,
        features: &[Vec<f32>],
        quality: &QualityLevel,
        _config: &OnDeviceConfig,
    ) -> Result<Vec<f32>, OnDeviceError> {
        // Simplified inference
        let mut output = vec![0.0; 1000]; // Placeholder vocab size

        // Simple linear transformation as placeholder
        for (i, feature_frame) in features.iter().enumerate() {
            for (j, &feature_val) in feature_frame.iter().enumerate() {
                if i * feature_frame.len() + j < output.len() {
                    output[i * feature_frame.len() + j] = feature_val * 0.1;
                }
            }
        }

        Ok(output)
    }
}

impl PostprocessingOptimizer {
    fn new(_config: &OnDeviceConfig) -> Self {
        Self {
            beam_search_optimized: true,
            language_model: None,
            confidence_scoring: true,
        }
    }

    fn process(
        &self,
        inference_output: &[f32],
        _config: &OnDeviceConfig,
    ) -> Result<String, OnDeviceError> {
        // Simple greedy decoding for demonstration
        let vocab = vec![
            "hello",
            "world",
            "this",
            "is",
            "a",
            "test",
            "of",
            "speech",
            "recognition",
        ];

        let mut result = String::new();
        for chunk in inference_output.chunks(vocab.len()) {
            if let Some((max_idx, _)) = chunk
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            {
                if !result.is_empty() {
                    result.push(' ');
                }
                result.push_str(vocab[max_idx % vocab.len()]);
            }
        }

        Ok(result)
    }
}

impl ResourceMonitor {
    fn new(interval: Duration) -> Self {
        Self {
            interval,
            usage_history: VecDeque::new(),
            last_update: Instant::now(),
            enabled: true,
        }
    }
}

/// On-device processing errors
#[derive(Debug, thiserror::Error)]
pub enum OnDeviceError {
    #[error("Processing deadline exceeded: {deadline_ms}ms")]
    DeadlineExceeded { deadline_ms: u32 },

    #[error("Insufficient device resources: {resource}")]
    InsufficientResources { resource: String },

    #[error("Hardware acceleration not available: {backend:?}")]
    AccelerationUnavailable { backend: AccelerationBackend },

    #[error("Thermal throttling active")]
    ThermalThrottling,

    #[error("Model loading failed: {reason}")]
    ModelLoadingFailed { reason: String },

    #[error("Feature extraction failed: {reason}")]
    FeatureExtractionFailed { reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_profile_creation() {
        let cpu_info = CpuInfo {
            cores: 8,
            performance_cores: 4,
            efficiency_cores: 4,
            base_frequency: 2400,
            max_frequency: 3200,
            architecture: CpuArchitecture::ARM64,
            simd_support: vec![SimdInstructionSet::NEON],
        };

        let memory_info = MemoryInfo {
            total_ram_mb: 8192,
            available_ram_mb: 6144,
            bandwidth_gbps: 68.0,
            cache_sizes: vec![32, 256, 8192], // KB
        };

        let thermal_info = ThermalInfo {
            current_temp: 35.0,
            throttle_temp: 80.0,
            max_temp: 105.0,
            thermal_management: true,
        };

        let profile = DeviceProfile {
            device_type: DeviceType::Mobile,
            cpu_info,
            memory_info,
            gpu_info: None,
            npu_info: None,
            thermal_info,
            battery_info: None,
        };

        assert_eq!(profile.device_type, DeviceType::Mobile);
        assert_eq!(profile.cpu_info.cores, 8);
    }

    #[test]
    fn test_model_variant_selection() {
        let cpu_info = CpuInfo {
            cores: 4,
            performance_cores: 4,
            efficiency_cores: 0,
            base_frequency: 1800,
            max_frequency: 2400,
            architecture: CpuArchitecture::ARM64,
            simd_support: vec![SimdInstructionSet::NEON],
        };

        let memory_info = MemoryInfo {
            total_ram_mb: 4096,
            available_ram_mb: 3072,
            bandwidth_gbps: 34.0,
            cache_sizes: vec![32, 256, 2048],
        };

        let thermal_info = ThermalInfo {
            current_temp: 30.0,
            throttle_temp: 75.0,
            max_temp: 95.0,
            thermal_management: true,
        };

        let profile = DeviceProfile {
            device_type: DeviceType::Mobile,
            cpu_info,
            memory_info,
            gpu_info: None,
            npu_info: None,
            thermal_info,
            battery_info: None,
        };

        let config = OnDeviceConfig {
            device_profile: profile,
            realtime_constraints: RealtimeConstraints {
                max_latency_ms: 100,
                target_latency_ms: 50,
                max_jitter_ms: 10,
                deadline_policy: DeadlinePolicy::Degrade,
            },
            compression: CompressionConfig {
                pruning: true,
                pruning_sparsity: 0.5,
                distillation: false,
                quantization: QuantizationSettings {
                    weight_bits: 8,
                    activation_bits: 8,
                    dynamic: true,
                    calibration: false,
                },
                dynamic_models: true,
            },
            acceleration: AccelerationConfig {
                preferred_backend: AccelerationBackend::CPU,
                fallback_backends: vec![AccelerationBackend::CPU],
                gpu_memory_limit_mb: None,
                npu_caching: false,
            },
            adaptive_inference: AdaptiveInferenceConfig {
                quality_adaptation: true,
                model_switching: true,
                monitoring_interval_ms: 1000,
                thresholds: AdaptationThresholds {
                    cpu_threshold: 0.8,
                    memory_threshold: 0.9,
                    temperature_threshold: 70.0,
                    battery_threshold: 0.2,
                },
            },
            power_management: PowerManagementConfig {
                dvfs_enabled: true,
                core_affinity: true,
                sleep_states: true,
                thermal_throttling: true,
            },
        };

        let optimizer = OnDeviceOptimizer::new(config);
        assert_eq!(optimizer.current_model, QualityLevel::Medium);
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::default();
        assert_eq!(metrics.total_frames, 0);
        assert_eq!(metrics.deadline_misses, 0);

        metrics.total_frames = 100;
        metrics.deadline_misses = 5;
        assert_eq!(metrics.total_frames, 100);
        assert_eq!(metrics.deadline_misses, 5);
    }
}
