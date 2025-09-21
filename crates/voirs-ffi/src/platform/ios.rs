//! iOS-specific bindings for VoiRS
//!
//! Provides native bindings for iOS applications, including Swift-compatible
//! interfaces and iOS-specific optimizations.

use crate::error::{FfiError, FfiResult};
use crate::types::{VoirsConfig, VoirsHandle};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_uint, c_void};
use std::ptr;
use voirs_cloning::config::{
    CloningConfig, CloningMethod, MemoryOptimization, ModelArchitecture, ModelConfig,
    ModelPrecision, PerformanceConfig, PreprocessingConfig, QualityAssessmentConfig, QualityMetric,
};
use voirs_cloning::mobile::{MobileCloningConfig, MobileDeviceInfo, MobilePlatform, PowerMode};
use voirs_sdk::VoirsPipeline;

/// Convert VoirsConfig to CloningConfig
fn convert_voirs_config_to_cloning_config(voirs_config: &VoirsConfig) -> CloningConfig {
    CloningConfig {
        default_method: CloningMethod::ZeroShot,
        output_sample_rate: voirs_config.sample_rate,
        quality_level: voirs_config.quality_level,
        use_gpu: voirs_config.use_gpu != 0,
        max_concurrent_operations: voirs_config.max_concurrent_ops as usize,
        model_configs: create_default_model_configs(),
        preprocessing: PreprocessingConfig {
            target_sample_rate: voirs_config.sample_rate,
            normalize_audio: true,
            trim_silence: true,
            silence_threshold: 0.01,
            noise_reduction: false,
            noise_reduction_strength: 0.3,
            segment_long_audio: true,
            max_segment_length: 30.0,
        },
        quality_assessment: QualityAssessmentConfig {
            enabled: true,
            similarity_threshold: 0.8,
            metrics: vec![
                QualityMetric::SpeakerSimilarity,
                QualityMetric::AudioQuality,
            ],
            perceptual_assessment: true,
            verification_threshold: 0.7,
        },
        performance: PerformanceConfig {
            num_threads: Some(4),
            use_simd: true,
            embedding_cache_size: 1000,
            model_cache_size: 100,
            quantization: true,
            quantization_bits: 8,
        },
        enable_cross_lingual: voirs_config.enable_cross_lingual != 0,
    }
}

/// Create default model configurations for different cloning methods
fn create_default_model_configs() -> HashMap<CloningMethod, ModelConfig> {
    let mut configs = HashMap::new();

    configs.insert(
        CloningMethod::ZeroShot,
        ModelConfig {
            model_path: None,
            architecture: ModelArchitecture::SpeakerEncoder,
            embedding_dim: 256,
            parameters: HashMap::new(),
            memory_optimization: MemoryOptimization {
                gradient_checkpointing: false,
                precision: ModelPrecision::Float32,
                batch_size: 1,
                max_memory_mb: Some(512),
            },
        },
    );

    configs.insert(
        CloningMethod::OneShot,
        ModelConfig {
            model_path: None,
            architecture: ModelArchitecture::MetaLearning,
            embedding_dim: 512,
            parameters: HashMap::new(),
            memory_optimization: MemoryOptimization {
                gradient_checkpointing: true,
                precision: ModelPrecision::Float16,
                batch_size: 2,
                max_memory_mb: Some(1024),
            },
        },
    );

    configs
}

/// iOS device information structure for Swift interop
#[repr(C)]
pub struct IOSDeviceInfo {
    pub model: *const c_char,
    pub os_version: *const c_char,
    pub cpu_type: *const c_char,
    pub cpu_cores: c_uint,
    pub ram_gb: c_float,
    pub has_neural_engine: c_int,
    pub has_neon: c_int,
    pub max_cpu_frequency: c_uint,
    pub gpu_type: *const c_char,
}

/// iOS-specific VoiRS configuration
#[repr(C)]
pub struct IOSVoirsConfig {
    pub base_config: VoirsConfig,
    pub power_mode: c_int,             // PowerMode enum as int
    pub adaptive_quality: c_int,       // bool as int
    pub use_metal_acceleration: c_int, // bool as int
    pub thermal_threshold: c_float,
    pub battery_threshold: c_float,
    pub memory_limit_mb: c_uint,
    pub background_processing: c_int, // bool as int
}

/// iOS optimization settings
#[repr(C)]
pub struct IOSOptimizationSettings {
    pub use_neural_engine: c_int,
    pub use_ane_optimization: c_int,
    pub priority_class: c_int, // iOS QoS class
    pub allow_background_refresh: c_int,
    pub network_type_preference: c_int, // WiFi vs Cellular
}

/// Initialize VoiRS for iOS with device-specific optimizations
///
/// # Safety
/// This function is intended to be called from Swift/Objective-C code.
/// The config pointer must be valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn voirs_ios_init(config: *const IOSVoirsConfig) -> VoirsHandle {
    if config.is_null() {
        return ptr::null_mut();
    }

    let ios_config = &*config;

    // Convert iOS config to internal config
    let mobile_config = MobileCloningConfig {
        base_config: convert_voirs_config_to_cloning_config(&ios_config.base_config),
        power_mode: match ios_config.power_mode {
            0 => PowerMode::Performance,
            1 => PowerMode::Balanced,
            2 => PowerMode::PowerSaver,
            3 => PowerMode::UltraLowPower,
            4 => PowerMode::Throttled,
            _ => PowerMode::Balanced,
        },
        adaptive_quality: ios_config.adaptive_quality != 0,
        max_concurrent_operations: if ios_config.background_processing != 0 {
            4
        } else {
            2
        },
        compression_level: 0.4, // iOS-optimized default
        use_quantized_models: true,
        quantization_method: voirs_cloning::quantization::QuantizationMethod::DynamicQuantization,
        enable_neon_optimization: true,
        allow_background_processing: ios_config.background_processing != 0,
        thermal_threshold: ios_config.thermal_threshold,
        battery_threshold: ios_config.battery_threshold,
        memory_limit_mb: ios_config.memory_limit_mb,
        cpu_limit_percent: 30.0, // iOS-conservative default
        enable_model_caching: true,
        max_cached_models: 3, // iOS memory constraints
        cache_strategy: voirs_cloning::mobile::CacheStrategy::LRU,
    };

    // Create pipeline with mobile optimizations
    match VoirsPipeline::new_with_mobile_config(mobile_config) {
        Ok(pipeline) => Box::into_raw(Box::new(pipeline)) as VoirsHandle,
        Err(_) => ptr::null_mut(),
    }
}

/// Get iOS device information
///
/// # Safety
/// The returned pointer must be freed using voirs_ios_free_device_info
#[no_mangle]
pub unsafe extern "C" fn voirs_ios_get_device_info() -> *mut IOSDeviceInfo {
    let device_info = voirs_cloning::mobile::device_detection::detect_ios_device();

    let model = CString::new(device_info.model).unwrap_or_default();
    let cpu_type = CString::new(device_info.architecture).unwrap_or_default();
    let gpu_type = CString::new(device_info.gpu_type).unwrap_or_default();
    let os_version = CString::new("iOS 15.0+").unwrap_or_default();

    let ios_info = IOSDeviceInfo {
        model: model.into_raw(),
        os_version: os_version.into_raw(),
        cpu_type: cpu_type.into_raw(),
        cpu_cores: device_info.cpu_cores,
        ram_gb: device_info.ram_mb as f32 / 1024.0,
        has_neural_engine: if device_info.has_npu { 1 } else { 0 },
        has_neon: if device_info.has_neon { 1 } else { 0 },
        max_cpu_frequency: device_info.max_cpu_frequency,
        gpu_type: gpu_type.into_raw(),
    };

    Box::into_raw(Box::new(ios_info))
}

/// Free iOS device information structure
///
/// # Safety
/// The device_info pointer must have been allocated by voirs_ios_get_device_info
#[no_mangle]
pub unsafe extern "C" fn voirs_ios_free_device_info(device_info: *mut IOSDeviceInfo) {
    if device_info.is_null() {
        return;
    }

    let info = Box::from_raw(device_info);

    // Free string fields
    if !info.model.is_null() {
        let _ = CString::from_raw(info.model as *mut c_char);
    }
    if !info.os_version.is_null() {
        let _ = CString::from_raw(info.os_version as *mut c_char);
    }
    if !info.cpu_type.is_null() {
        let _ = CString::from_raw(info.cpu_type as *mut c_char);
    }
    if !info.gpu_type.is_null() {
        let _ = CString::from_raw(info.gpu_type as *mut c_char);
    }
}

/// Configure iOS-specific optimizations
///
/// # Safety
/// Both handle and settings pointers must be valid
#[no_mangle]
pub unsafe extern "C" fn voirs_ios_configure_optimizations(
    handle: VoirsHandle,
    settings: *const IOSOptimizationSettings,
) -> c_int {
    if handle.is_null() || settings.is_null() {
        return -1; // Error
    }

    let settings = &*settings;

    // Configure iOS-specific optimizations
    // This would typically involve setting iOS QoS classes, neural engine usage, etc.
    // For now, return success
    0
}

/// Enable iOS background processing mode
///
/// # Safety
/// Handle must be a valid VoiRS handle
#[no_mangle]
pub unsafe extern "C" fn voirs_ios_enable_background_mode(
    handle: VoirsHandle,
    enable: c_int,
) -> c_int {
    if handle.is_null() {
        return -1;
    }

    // Implementation would configure background processing capabilities
    // This involves iOS-specific background task management
    0
}

/// Set iOS thermal state callback
///
/// # Safety
/// Handle must be valid, callback can be null to disable
#[no_mangle]
pub unsafe extern "C" fn voirs_ios_set_thermal_callback(
    handle: VoirsHandle,
    callback: Option<extern "C" fn(thermal_state: c_int)>,
) -> c_int {
    if handle.is_null() {
        return -1;
    }

    // Implementation would register thermal state monitoring
    // This would integrate with iOS thermal APIs
    0
}

/// iOS-specific cleanup
///
/// # Safety
/// Handle must be a valid VoiRS handle
#[no_mangle]
pub unsafe extern "C" fn voirs_ios_cleanup(handle: VoirsHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle as *mut VoirsPipeline);
    }
}

/// Get iOS performance statistics
///
/// # Safety
/// Handle must be valid, stats pointer must point to allocated memory
#[no_mangle]
pub unsafe extern "C" fn voirs_ios_get_performance_stats(
    handle: VoirsHandle,
    stats: *mut IOSPerformanceStats,
) -> c_int {
    if handle.is_null() || stats.is_null() {
        return -1;
    }

    // Implementation would collect iOS-specific performance metrics
    // including Metal GPU usage, neural engine utilization, etc.
    0
}

/// iOS performance statistics structure
#[repr(C)]
pub struct IOSPerformanceStats {
    pub cpu_usage_percent: c_float,
    pub memory_usage_mb: c_float,
    pub gpu_usage_percent: c_float,
    pub neural_engine_usage_percent: c_float,
    pub thermal_state: c_int,
    pub battery_usage_rate: c_float,
    pub synthesis_operations: c_uint,
    pub avg_synthesis_time_ms: c_float,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ios_device_info() {
        unsafe {
            let device_info = voirs_ios_get_device_info();
            assert!(!device_info.is_null());

            let info = &*device_info;
            assert!(info.cpu_cores > 0);
            assert!(info.ram_gb > 0.0);

            voirs_ios_free_device_info(device_info);
        }
    }

    #[test]
    fn test_ios_config_conversion() {
        let ios_config = IOSVoirsConfig {
            base_config: VoirsConfig::default(),
            power_mode: 1, // Balanced
            adaptive_quality: 1,
            use_metal_acceleration: 1,
            thermal_threshold: 45.0,
            battery_threshold: 0.2,
            memory_limit_mb: 512,
            background_processing: 0,
        };

        unsafe {
            let handle = voirs_ios_init(&ios_config);
            assert!(!handle.is_null());
            voirs_ios_cleanup(handle);
        }
    }
}
