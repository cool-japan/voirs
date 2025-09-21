//! Extended C API for configuration management.
//!
//! This module provides configuration management functions for synthesis
//! parameters, model settings, and performance tuning through the C API.

use crate::VoirsErrorCode;
use std::os::raw::{c_char, c_float, c_int, c_uint};

/// Synthesis configuration structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VoirsSynthesisConfig {
    /// Speed adjustment factor (0.5 to 2.0, 1.0 = normal)
    pub speed: c_float,
    /// Pitch adjustment in semitones (-12.0 to 12.0)
    pub pitch: c_float,
    /// Volume adjustment (0.0 to 2.0, 1.0 = normal)
    pub volume: c_float,
    /// Sample rate in Hz (8000, 16000, 22050, 44100, 48000)
    pub sample_rate: c_uint,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: c_uint,
    /// Quality level (0 = fast, 1 = balanced, 2 = high quality)
    pub quality: c_uint,
    /// Enable noise reduction (0 = disabled, 1 = enabled)
    pub noise_reduction: c_int,
    /// Enable voice enhancement (0 = disabled, 1 = enabled)
    pub voice_enhancement: c_int,
}

impl Default for VoirsSynthesisConfig {
    fn default() -> Self {
        Self {
            speed: 1.0,
            pitch: 0.0,
            volume: 1.0,
            sample_rate: 22050,
            channels: 1,
            quality: 1,
            noise_reduction: 0,
            voice_enhancement: 0,
        }
    }
}

/// Model configuration structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VoirsModelConfig {
    /// Model type (0 = FastSpeech2, 1 = VITS, 2 = Tacotron2)
    pub model_type: c_uint,
    /// Memory usage mode (0 = low, 1 = balanced, 2 = high)
    pub memory_mode: c_uint,
    /// Processing mode (0 = CPU, 1 = GPU if available)
    pub processing_mode: c_uint,
    /// Batch size for processing (1-32)
    pub batch_size: c_uint,
    /// Enable model caching (0 = disabled, 1 = enabled)
    pub enable_caching: c_int,
    /// Cache size limit in MB
    pub cache_size_mb: c_uint,
}

impl Default for VoirsModelConfig {
    fn default() -> Self {
        Self {
            model_type: 0,
            memory_mode: 1,
            processing_mode: 0,
            batch_size: 1,
            enable_caching: 1,
            cache_size_mb: 256,
        }
    }
}

/// API Performance configuration structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VoirsApiPerformanceConfig {
    /// Thread count (0 = auto, 1-16 = specific count)
    pub thread_count: c_uint,
    /// Enable SIMD optimizations (0 = disabled, 1 = enabled)
    pub enable_simd: c_int,
    /// Prefetch size for audio buffers
    pub prefetch_size: c_uint,
    /// Memory pool size in MB
    pub memory_pool_mb: c_uint,
    /// Enable performance monitoring (0 = disabled, 1 = enabled)
    pub enable_monitoring: c_int,
    /// Monitoring sample interval in ms
    pub monitoring_interval_ms: c_uint,
}

impl Default for VoirsApiPerformanceConfig {
    fn default() -> Self {
        Self {
            thread_count: 0,
            enable_simd: 1,
            prefetch_size: 1024,
            memory_pool_mb: 64,
            enable_monitoring: 0,
            monitoring_interval_ms: 1000,
        }
    }
}

/// Create a default synthesis configuration
#[no_mangle]
pub unsafe extern "C" fn voirs_config_create_synthesis_default() -> VoirsSynthesisConfig {
    VoirsSynthesisConfig::default()
}

/// Create a default model configuration
#[no_mangle]
pub unsafe extern "C" fn voirs_config_create_model_default() -> VoirsModelConfig {
    VoirsModelConfig::default()
}

/// Create a default performance configuration
#[no_mangle]
pub unsafe extern "C" fn voirs_config_create_performance_default() -> VoirsApiPerformanceConfig {
    VoirsApiPerformanceConfig::default()
}

/// Validate synthesis configuration
#[no_mangle]
pub unsafe extern "C" fn voirs_config_validate_synthesis(
    config: *const VoirsSynthesisConfig,
) -> VoirsErrorCode {
    if config.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    let cfg = &*config;

    // Validate speed range
    if cfg.speed < 0.5 || cfg.speed > 2.0 {
        return VoirsErrorCode::InvalidParameter;
    }

    // Validate pitch range
    if cfg.pitch < -12.0 || cfg.pitch > 12.0 {
        return VoirsErrorCode::InvalidParameter;
    }

    // Validate volume range
    if cfg.volume < 0.0 || cfg.volume > 2.0 {
        return VoirsErrorCode::InvalidParameter;
    }

    // Validate sample rate
    match cfg.sample_rate {
        8000 | 16000 | 22050 | 44100 | 48000 => {}
        _ => return VoirsErrorCode::InvalidParameter,
    }

    // Validate channels
    if cfg.channels < 1 || cfg.channels > 2 {
        return VoirsErrorCode::InvalidParameter;
    }

    // Validate quality
    if cfg.quality > 2 {
        return VoirsErrorCode::InvalidParameter;
    }

    VoirsErrorCode::Success
}

/// Validate model configuration
#[no_mangle]
pub unsafe extern "C" fn voirs_config_validate_model(
    config: *const VoirsModelConfig,
) -> VoirsErrorCode {
    if config.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    let cfg = &*config;

    // Validate model type
    if cfg.model_type > 2 {
        return VoirsErrorCode::InvalidParameter;
    }

    // Validate memory mode
    if cfg.memory_mode > 2 {
        return VoirsErrorCode::InvalidParameter;
    }

    // Validate processing mode
    if cfg.processing_mode > 1 {
        return VoirsErrorCode::InvalidParameter;
    }

    // Validate batch size
    if cfg.batch_size < 1 || cfg.batch_size > 32 {
        return VoirsErrorCode::InvalidParameter;
    }

    // Validate cache size
    if cfg.cache_size_mb > 2048 {
        return VoirsErrorCode::InvalidParameter;
    }

    VoirsErrorCode::Success
}

/// Validate performance configuration
#[no_mangle]
pub unsafe extern "C" fn voirs_config_validate_performance(
    config: *const VoirsApiPerformanceConfig,
) -> VoirsErrorCode {
    if config.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    let cfg = &*config;

    // Validate thread count
    if cfg.thread_count > 16 {
        return VoirsErrorCode::InvalidParameter;
    }

    // Validate prefetch size
    if cfg.prefetch_size > 8192 {
        return VoirsErrorCode::InvalidParameter;
    }

    // Validate memory pool size
    if cfg.memory_pool_mb > 1024 {
        return VoirsErrorCode::InvalidParameter;
    }

    // Validate monitoring interval
    if cfg.monitoring_interval_ms < 100 || cfg.monitoring_interval_ms > 10000 {
        return VoirsErrorCode::InvalidParameter;
    }

    VoirsErrorCode::Success
}

/// Get synthesis configuration description
#[no_mangle]
pub unsafe extern "C" fn voirs_config_get_synthesis_info(
    config: *const VoirsSynthesisConfig,
    buffer: *mut c_char,
    buffer_size: c_uint,
) -> VoirsErrorCode {
    if config.is_null() || buffer.is_null() || buffer_size == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let cfg = &*config;

    let info = format!(
        "Speed: {:.2}, Pitch: {:.1}st, Volume: {:.2}, SR: {}Hz, Ch: {}, Quality: {}, NR: {}, VE: {}",
        cfg.speed, cfg.pitch, cfg.volume, cfg.sample_rate, cfg.channels,
        cfg.quality, cfg.noise_reduction, cfg.voice_enhancement
    );

    let info_bytes = info.as_bytes();
    let copy_len = (info_bytes.len()).min(buffer_size as usize - 1);

    std::ptr::copy_nonoverlapping(info_bytes.as_ptr(), buffer as *mut u8, copy_len);

    // Null terminate
    *buffer.add(copy_len) = 0;

    VoirsErrorCode::Success
}

/// Apply synthesis configuration preset
#[no_mangle]
pub unsafe extern "C" fn voirs_config_apply_synthesis_preset(
    config: *mut VoirsSynthesisConfig,
    preset: c_uint, // 0=fast, 1=balanced, 2=quality, 3=low_latency
) -> VoirsErrorCode {
    if config.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    let cfg = &mut *config;

    match preset {
        0 => {
            // Fast preset
            cfg.quality = 0;
            cfg.sample_rate = 16000;
            cfg.noise_reduction = 0;
            cfg.voice_enhancement = 0;
        }
        1 => {
            // Balanced preset
            cfg.quality = 1;
            cfg.sample_rate = 22050;
            cfg.noise_reduction = 1;
            cfg.voice_enhancement = 0;
        }
        2 => {
            // Quality preset
            cfg.quality = 2;
            cfg.sample_rate = 44100;
            cfg.noise_reduction = 1;
            cfg.voice_enhancement = 1;
        }
        3 => {
            // Low latency preset
            cfg.quality = 0;
            cfg.sample_rate = 8000;
            cfg.noise_reduction = 0;
            cfg.voice_enhancement = 0;
        }
        _ => return VoirsErrorCode::InvalidParameter,
    }

    VoirsErrorCode::Success
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesis_config_default() {
        let config = VoirsSynthesisConfig::default();
        assert_eq!(config.speed, 1.0);
        assert_eq!(config.pitch, 0.0);
        assert_eq!(config.volume, 1.0);
        assert_eq!(config.sample_rate, 22050);
        assert_eq!(config.channels, 1);
    }

    #[test]
    fn test_model_config_default() {
        let config = VoirsModelConfig::default();
        assert_eq!(config.model_type, 0);
        assert_eq!(config.memory_mode, 1);
        assert_eq!(config.processing_mode, 0);
        assert_eq!(config.batch_size, 1);
    }

    #[test]
    fn test_performance_config_default() {
        let config = VoirsApiPerformanceConfig::default();
        assert_eq!(config.thread_count, 0);
        assert_eq!(config.enable_simd, 1);
        assert_eq!(config.prefetch_size, 1024);
        assert_eq!(config.memory_pool_mb, 64);
    }

    #[test]
    fn test_synthesis_config_validation() {
        let config = VoirsSynthesisConfig::default();
        unsafe {
            let result = voirs_config_validate_synthesis(&config);
            assert_eq!(result, VoirsErrorCode::Success);
        }
    }

    #[test]
    fn test_synthesis_config_invalid_speed() {
        let mut config = VoirsSynthesisConfig::default();
        config.speed = 3.0; // Invalid speed
        unsafe {
            let result = voirs_config_validate_synthesis(&config);
            assert_eq!(result, VoirsErrorCode::InvalidParameter);
        }
    }

    #[test]
    fn test_synthesis_preset_application() {
        let mut config = VoirsSynthesisConfig::default();
        unsafe {
            let result = voirs_config_apply_synthesis_preset(&mut config, 2); // Quality preset
            assert_eq!(result, VoirsErrorCode::Success);
            assert_eq!(config.quality, 2);
            assert_eq!(config.sample_rate, 44100);
        }
    }

    #[test]
    fn test_config_info_generation() {
        let config = VoirsSynthesisConfig::default();
        let mut buffer = [0u8; 256];
        unsafe {
            let result = voirs_config_get_synthesis_info(
                &config,
                buffer.as_mut_ptr() as *mut c_char,
                buffer.len() as c_uint,
            );
            assert_eq!(result, VoirsErrorCode::Success);

            let info_str = std::ffi::CStr::from_ptr(buffer.as_ptr() as *const c_char);
            let info = info_str.to_str().unwrap();
            assert!(info.contains("Speed: 1.00"));
            assert!(info.contains("SR: 22050Hz"));
        }
    }
}
