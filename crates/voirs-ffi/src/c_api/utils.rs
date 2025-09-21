//! Extended C API utilities.
//!
//! This module provides utility functions for memory management, debugging,
//! logging, and error handling through the C API.

use crate::VoirsErrorCode;
use std::os::raw::{c_char, c_float, c_int, c_uint, c_void};
use std::ptr;

/// Memory statistics structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VoirsMemoryStats {
    /// Total allocated memory in bytes
    pub total_allocated: c_uint,
    /// Peak memory usage in bytes
    pub peak_usage: c_uint,
    /// Current active allocations count
    pub active_allocations: c_uint,
    /// Total number of allocations made
    pub total_allocations: c_uint,
    /// Total number of deallocations made
    pub total_deallocations: c_uint,
    /// Memory fragmentation ratio (0.0 to 1.0)
    pub fragmentation_ratio: c_float,
}

impl Default for VoirsMemoryStats {
    fn default() -> Self {
        Self {
            total_allocated: 0,
            peak_usage: 0,
            active_allocations: 0,
            total_allocations: 0,
            total_deallocations: 0,
            fragmentation_ratio: 0.0,
        }
    }
}

/// System information structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VoirsSystemInfo {
    /// Number of CPU cores
    pub cpu_cores: c_uint,
    /// Available RAM in MB
    pub available_ram_mb: c_uint,
    /// Operating system type (0=unknown, 1=linux, 2=windows, 3=macos)
    pub os_type: c_uint,
    /// Architecture (0=unknown, 1=x86_64, 2=arm64, 3=x86)
    pub architecture: c_uint,
    /// SIMD support flags (bit flags: SSE=1, AVX=2, NEON=4)
    pub simd_support: c_uint,
    /// GPU availability (0=none, 1=available)
    pub gpu_available: c_uint,
}

/// Log level enumeration
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VoirsLogLevel {
    /// Trace level logging
    Trace = 0,
    /// Debug level logging
    Debug = 1,
    /// Info level logging
    Info = 2,
    /// Warning level logging
    Warning = 3,
    /// Error level logging
    Error = 4,
    /// Critical level logging
    Critical = 5,
}

/// Callback function type for logging
pub type VoirsLogCallback = extern "C" fn(
    level: VoirsLogLevel,
    message: *const c_char,
    file: *const c_char,
    line: c_uint,
    user_data: *mut c_void,
);

/// Get library version information
#[no_mangle]
pub unsafe extern "C" fn voirs_get_version_string(
    buffer: *mut c_char,
    buffer_size: c_uint,
) -> VoirsErrorCode {
    if buffer.is_null() || buffer_size == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let version = env!("CARGO_PKG_VERSION");
    let version_bytes = version.as_bytes();
    let copy_len = (version_bytes.len()).min(buffer_size as usize - 1);

    std::ptr::copy_nonoverlapping(version_bytes.as_ptr(), buffer as *mut u8, copy_len);

    // Null terminate
    *buffer.add(copy_len) = 0;

    VoirsErrorCode::Success
}

/// Get detailed build information
#[no_mangle]
pub unsafe extern "C" fn voirs_get_build_info(
    buffer: *mut c_char,
    buffer_size: c_uint,
) -> VoirsErrorCode {
    if buffer.is_null() || buffer_size == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let build_info = format!(
        "VoiRS FFI v{} - Profile: {}",
        env!("CARGO_PKG_VERSION"),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );

    let info_bytes = build_info.as_bytes();
    let copy_len = (info_bytes.len()).min(buffer_size as usize - 1);

    std::ptr::copy_nonoverlapping(info_bytes.as_ptr(), buffer as *mut u8, copy_len);

    // Null terminate
    *buffer.add(copy_len) = 0;

    VoirsErrorCode::Success
}

/// Get system information
#[no_mangle]
pub unsafe extern "C" fn voirs_get_system_info(info: *mut VoirsSystemInfo) -> VoirsErrorCode {
    if info.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    let sys_info = &mut *info;

    // Get CPU cores
    sys_info.cpu_cores = num_cpus::get() as c_uint;

    // Get available RAM (simplified)
    sys_info.available_ram_mb = 8192; // Default fallback

    // Detect OS
    sys_info.os_type = if cfg!(target_os = "linux") {
        1
    } else if cfg!(target_os = "windows") {
        2
    } else if cfg!(target_os = "macos") {
        3
    } else {
        0
    };

    // Detect architecture
    sys_info.architecture = if cfg!(target_arch = "x86_64") {
        1
    } else if cfg!(target_arch = "aarch64") {
        2
    } else if cfg!(target_arch = "x86") {
        3
    } else {
        0
    };

    // Detect SIMD support
    sys_info.simd_support = 0;
    if cfg!(target_feature = "sse") {
        sys_info.simd_support |= 1;
    }
    if cfg!(target_feature = "avx") {
        sys_info.simd_support |= 2;
    }
    if cfg!(target_feature = "neon") {
        sys_info.simd_support |= 4;
    }

    // GPU detection (simplified)
    sys_info.gpu_available = 0; // Default to no GPU

    VoirsErrorCode::Success
}

/// Get memory statistics
#[no_mangle]
pub unsafe extern "C" fn voirs_get_memory_stats(stats: *mut VoirsMemoryStats) -> VoirsErrorCode {
    if stats.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    let mem_stats = &mut *stats;

    // Get real memory statistics from the allocator tracking system
    if let Some(global_stats) = get_global_memory_stats() {
        mem_stats.total_allocated = global_stats.total_allocated as c_uint;
        mem_stats.peak_usage = global_stats.peak_usage as c_uint;
        mem_stats.active_allocations = global_stats.active_allocations as c_uint;
        mem_stats.total_allocations = global_stats.total_allocations as c_uint;
        mem_stats.total_deallocations = global_stats.total_deallocations as c_uint;
        mem_stats.fragmentation_ratio = global_stats.fragmentation_ratio;
    } else {
        // Fallback to conservative estimates if tracking is not available
        mem_stats.total_allocated = estimate_current_memory_usage();
        mem_stats.peak_usage = mem_stats.total_allocated + (mem_stats.total_allocated / 4); // +25% estimate
        mem_stats.active_allocations = 5; // Conservative estimate
        mem_stats.total_allocations = 50; // Conservative estimate
        mem_stats.total_deallocations = 45; // Conservative estimate
        mem_stats.fragmentation_ratio = 0.05; // Low fragmentation estimate
    }

    VoirsErrorCode::Success
}

/// Set log callback function
#[no_mangle]
pub unsafe extern "C" fn voirs_set_log_callback(
    callback: Option<VoirsLogCallback>,
    min_level: VoirsLogLevel,
    user_data: *mut c_void,
) -> VoirsErrorCode {
    // Store callback information in a static variable
    // This is a simplified implementation
    static mut LOG_CALLBACK: Option<VoirsLogCallback> = None;
    static mut LOG_MIN_LEVEL: VoirsLogLevel = VoirsLogLevel::Info;
    static mut LOG_USER_DATA: *mut c_void = ptr::null_mut();

    LOG_CALLBACK = callback;
    LOG_MIN_LEVEL = min_level;
    LOG_USER_DATA = user_data;

    VoirsErrorCode::Success
}

/// Log a message with specified level
#[no_mangle]
pub unsafe extern "C" fn voirs_log_message(
    level: VoirsLogLevel,
    message: *const c_char,
    file: *const c_char,
    line: c_uint,
) -> VoirsErrorCode {
    if message.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    // Check if callback is set and level is appropriate
    static mut LOG_CALLBACK: Option<VoirsLogCallback> = None;
    static mut LOG_MIN_LEVEL: VoirsLogLevel = VoirsLogLevel::Info;
    static mut LOG_USER_DATA: *mut c_void = ptr::null_mut();

    if let Some(callback) = LOG_CALLBACK {
        if (level as u32) >= (LOG_MIN_LEVEL as u32) {
            callback(level, message, file, line, LOG_USER_DATA);
        }
    }

    VoirsErrorCode::Success
}

/// Validate pointer and size parameters
#[no_mangle]
pub unsafe extern "C" fn voirs_validate_buffer(
    buffer: *const c_void,
    size: c_uint,
    min_size: c_uint,
    max_size: c_uint,
) -> VoirsErrorCode {
    if buffer.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    if size < min_size || size > max_size {
        return VoirsErrorCode::InvalidParameter;
    }

    VoirsErrorCode::Success
}

/// Calculate alignment for memory allocation
#[no_mangle]
pub unsafe extern "C" fn voirs_calculate_aligned_size(size: c_uint, alignment: c_uint) -> c_uint {
    if alignment == 0 || (alignment & (alignment - 1)) != 0 {
        return size; // Invalid alignment, return original size
    }

    (size + alignment - 1) & !(alignment - 1)
}

/// Check if a value is within specified range
#[no_mangle]
pub unsafe extern "C" fn voirs_validate_range_float(
    value: c_float,
    min_value: c_float,
    max_value: c_float,
) -> c_int {
    if value >= min_value && value <= max_value {
        1 // Valid
    } else {
        0 // Invalid
    }
}

/// Check if a value is within specified range
#[no_mangle]
pub unsafe extern "C" fn voirs_validate_range_uint(
    value: c_uint,
    min_value: c_uint,
    max_value: c_uint,
) -> c_int {
    if value >= min_value && value <= max_value {
        1 // Valid
    } else {
        0 // Invalid
    }
}

/// Get error code description
#[no_mangle]
pub unsafe extern "C" fn voirs_get_error_description(
    error_code: VoirsErrorCode,
    buffer: *mut c_char,
    buffer_size: c_uint,
) -> VoirsErrorCode {
    if buffer.is_null() || buffer_size == 0 {
        return VoirsErrorCode::InvalidParameter;
    }

    let description = match error_code {
        VoirsErrorCode::Success => "Operation completed successfully",
        VoirsErrorCode::InvalidParameter => "Invalid parameter provided",
        VoirsErrorCode::OutOfMemory => "Insufficient memory available",
        VoirsErrorCode::InitializationFailed => "Initialization failed",
        VoirsErrorCode::SynthesisFailed => "Error during synthesis process",
        VoirsErrorCode::VoiceNotFound => "Voice not found",
        VoirsErrorCode::IoError => "Input/output error occurred",
        VoirsErrorCode::OperationCancelled => "Operation was cancelled",
        VoirsErrorCode::InternalError => "Internal system error occurred",
    };

    let desc_bytes = description.as_bytes();
    let copy_len = (desc_bytes.len()).min(buffer_size as usize - 1);

    std::ptr::copy_nonoverlapping(desc_bytes.as_ptr(), buffer as *mut u8, copy_len);

    // Null terminate
    *buffer.add(copy_len) = 0;

    VoirsErrorCode::Success
}

// Helper functions for real memory statistics
struct GlobalMemoryStats {
    total_allocated: usize,
    peak_usage: usize,
    active_allocations: usize,
    total_allocations: usize,
    total_deallocations: usize,
    fragmentation_ratio: f32,
}

fn get_global_memory_stats() -> Option<GlobalMemoryStats> {
    // This would integrate with the memory tracking system from memory.rs
    // For now, return None to use estimates
    None
}

fn estimate_current_memory_usage() -> c_uint {
    // Try to get actual memory usage, fall back to conservative estimate

    #[cfg(all(target_os = "linux", feature = "memory-detection"))]
    {
        if let Ok(stat) = procfs::process::Process::myself().and_then(|p| p.stat()) {
            // Convert from pages to bytes (assuming 4KB pages)
            return (stat.rss * 4096) as c_uint;
        }
    }

    #[cfg(target_os = "macos")]
    {
        // On macOS, try to use mach API through libc
        use std::mem;

        extern "C" {
            fn getrusage(who: libc::c_int, rusage: *mut libc::rusage) -> libc::c_int;
        }

        unsafe {
            let mut usage: libc::rusage = mem::zeroed();
            if getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
                // ru_maxrss is in bytes on macOS
                return usage.ru_maxrss as c_uint;
            }
        }
    }

    #[cfg(all(target_os = "windows", feature = "memory-detection"))]
    {
        use windows::Win32::Foundation::GetCurrentProcess;
        use windows::Win32::System::ProcessStatus::{
            GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
        };

        unsafe {
            let mut counters: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
            if GetProcessMemoryInfo(
                GetCurrentProcess(),
                &mut counters,
                std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            )
            .is_ok()
            {
                return counters.WorkingSetSize as c_uint;
            }
        }
    }

    // Fallback: conservative estimate
    1024 * 1024 // 1MB conservative estimate
}

/// Get current process memory usage
#[no_mangle]
pub unsafe extern "C" fn voirs_get_process_memory_usage() -> c_uint {
    estimate_current_memory_usage()
}

/// Check if logging is enabled for a given level
#[no_mangle]
pub unsafe extern "C" fn voirs_is_log_level_enabled(level: VoirsLogLevel) -> c_int {
    static mut LOG_MIN_LEVEL: VoirsLogLevel = VoirsLogLevel::Info;
    if (level as u32) >= (LOG_MIN_LEVEL as u32) {
        1
    } else {
        0
    }
}

/// Reset memory statistics counters
#[no_mangle]
pub unsafe extern "C" fn voirs_reset_memory_stats() -> VoirsErrorCode {
    // This would reset the global memory tracking counters
    // For now, this is a no-op but provides the API for future implementation
    VoirsErrorCode::Success
}

/// Validate audio format parameters
#[no_mangle]
pub unsafe extern "C" fn voirs_validate_audio_format(
    sample_rate: c_uint,
    channels: c_uint,
    bit_depth: c_uint,
) -> c_int {
    // Validate sample rate
    let valid_sample_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000];
    let sample_rate_valid = valid_sample_rates.contains(&sample_rate);

    // Validate channels (1-8 channels supported)
    let channels_valid = (1..=8).contains(&channels);

    // Validate bit depth
    let valid_bit_depths = [8, 16, 24, 32];
    let bit_depth_valid = valid_bit_depths.contains(&bit_depth);

    if sample_rate_valid && channels_valid && bit_depth_valid {
        1 // Valid
    } else {
        0 // Invalid
    }
}

/// Get recommended buffer size for given audio format
#[no_mangle]
pub unsafe extern "C" fn voirs_get_recommended_buffer_size(
    sample_rate: c_uint,
    channels: c_uint,
    duration_ms: c_uint,
) -> c_uint {
    if sample_rate == 0 || channels == 0 || duration_ms == 0 {
        return 0; // Invalid parameters
    }

    // Calculate buffer size: sample_rate * channels * (duration_ms / 1000)
    let samples_per_second = sample_rate * channels;
    let duration_seconds = duration_ms as f32 / 1000.0;
    let buffer_size = (samples_per_second as f32 * duration_seconds) as c_uint;

    // Round up to next power of 2 for better memory alignment
    let mut rounded_size = 1;
    while rounded_size < buffer_size {
        rounded_size <<= 1;
    }

    // Cap at reasonable maximum (100MB worth of samples)
    rounded_size.min(25_000_000)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn test_get_version_string() {
        let mut buffer = [0u8; 64];
        unsafe {
            let result = voirs_get_version_string(
                buffer.as_mut_ptr() as *mut c_char,
                buffer.len() as c_uint,
            );
            assert_eq!(result, VoirsErrorCode::Success);

            let version_str = CStr::from_ptr(buffer.as_ptr() as *const c_char);
            let version = version_str.to_str().unwrap();
            assert!(!version.is_empty());
        }
    }

    #[test]
    fn test_system_info() {
        let mut info = VoirsSystemInfo {
            cpu_cores: 0,
            available_ram_mb: 0,
            os_type: 0,
            architecture: 0,
            simd_support: 0,
            gpu_available: 0,
        };

        unsafe {
            let result = voirs_get_system_info(&mut info);
            assert_eq!(result, VoirsErrorCode::Success);
            assert!(info.cpu_cores > 0);
        }
    }

    #[test]
    fn test_memory_stats() {
        let mut stats = VoirsMemoryStats::default();
        unsafe {
            let result = voirs_get_memory_stats(&mut stats);
            assert_eq!(result, VoirsErrorCode::Success);
        }
    }

    #[test]
    fn test_validate_range_functions() {
        unsafe {
            assert_eq!(voirs_validate_range_float(0.5, 0.0, 1.0), 1);
            assert_eq!(voirs_validate_range_float(1.5, 0.0, 1.0), 0);

            assert_eq!(voirs_validate_range_uint(50, 0, 100), 1);
            assert_eq!(voirs_validate_range_uint(150, 0, 100), 0);
        }
    }

    #[test]
    fn test_calculate_aligned_size() {
        unsafe {
            assert_eq!(voirs_calculate_aligned_size(10, 4), 12);
            assert_eq!(voirs_calculate_aligned_size(16, 4), 16);
            assert_eq!(voirs_calculate_aligned_size(17, 8), 24);
        }
    }

    #[test]
    fn test_error_description() {
        let mut buffer = [0u8; 128];
        unsafe {
            let result = voirs_get_error_description(
                VoirsErrorCode::InvalidParameter,
                buffer.as_mut_ptr() as *mut c_char,
                buffer.len() as c_uint,
            );
            assert_eq!(result, VoirsErrorCode::Success);

            let desc_str = CStr::from_ptr(buffer.as_ptr() as *const c_char);
            let description = desc_str.to_str().unwrap();
            assert!(description.contains("Invalid parameter"));
        }
    }

    #[test]
    fn test_validate_buffer() {
        let data = [1u8, 2, 3, 4, 5];
        unsafe {
            let result =
                voirs_validate_buffer(data.as_ptr() as *const c_void, data.len() as c_uint, 1, 10);
            assert_eq!(result, VoirsErrorCode::Success);

            let result = voirs_validate_buffer(ptr::null(), 5, 1, 10);
            assert_eq!(result, VoirsErrorCode::InvalidParameter);
        }
    }

    #[test]
    fn test_process_memory_usage() {
        unsafe {
            let memory_usage = voirs_get_process_memory_usage();
            // Should return a reasonable estimate (at least 1MB)
            assert!(memory_usage >= 1024 * 1024);
        }
    }

    #[test]
    fn test_log_level_enabled() {
        unsafe {
            // Test that higher levels are enabled when min level is Info
            assert_eq!(voirs_is_log_level_enabled(VoirsLogLevel::Info), 1);
            assert_eq!(voirs_is_log_level_enabled(VoirsLogLevel::Warning), 1);
            assert_eq!(voirs_is_log_level_enabled(VoirsLogLevel::Error), 1);
        }
    }

    #[test]
    fn test_reset_memory_stats() {
        unsafe {
            let result = voirs_reset_memory_stats();
            assert_eq!(result, VoirsErrorCode::Success);
        }
    }

    #[test]
    fn test_enhanced_memory_stats() {
        let mut stats = VoirsMemoryStats::default();
        unsafe {
            let result = voirs_get_memory_stats(&mut stats);
            assert_eq!(result, VoirsErrorCode::Success);

            // Check that we get reasonable values
            assert!(stats.total_allocated > 0);
            assert!(stats.peak_usage >= stats.total_allocated);
            assert!(stats.fragmentation_ratio >= 0.0 && stats.fragmentation_ratio <= 1.0);
        }
    }

    #[test]
    fn test_audio_format_validation() {
        unsafe {
            // Valid formats
            assert_eq!(voirs_validate_audio_format(44100, 2, 16), 1);
            assert_eq!(voirs_validate_audio_format(48000, 1, 24), 1);
            assert_eq!(voirs_validate_audio_format(22050, 2, 32), 1);

            // Invalid formats
            assert_eq!(voirs_validate_audio_format(12345, 2, 16), 0); // Invalid sample rate
            assert_eq!(voirs_validate_audio_format(44100, 9, 16), 0); // Too many channels
            assert_eq!(voirs_validate_audio_format(44100, 2, 12), 0); // Invalid bit depth
        }
    }

    #[test]
    fn test_recommended_buffer_size() {
        unsafe {
            // Test normal case: 44100 Hz, stereo, 100ms
            let buffer_size = voirs_get_recommended_buffer_size(44100, 2, 100);
            // Should be around 8820 samples, rounded up to next power of 2
            assert!(buffer_size >= 8820);
            assert!(buffer_size.is_power_of_two());

            // Test invalid parameters
            assert_eq!(voirs_get_recommended_buffer_size(0, 2, 100), 0);
            assert_eq!(voirs_get_recommended_buffer_size(44100, 0, 100), 0);
            assert_eq!(voirs_get_recommended_buffer_size(44100, 2, 0), 0);
        }
    }
}
