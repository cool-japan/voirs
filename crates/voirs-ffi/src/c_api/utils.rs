//! Extended C API utilities.
//!
//! This module provides utility functions for memory management, debugging,
//! logging, and error handling through the C API.

use crate::VoirsErrorCode;
use parking_lot::Mutex;
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
    /// Total physical RAM in MB (not a currently-free/available estimate;
    /// queried via `platform::PlatformInfo::total_memory`).
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

/// Shared state backing `voirs_set_log_callback`, `voirs_log_message`, and
/// `voirs_is_log_level_enabled`.
///
/// These three functions previously each declared their own function-local
/// `static mut LOG_CALLBACK`/`LOG_MIN_LEVEL`/`LOG_USER_DATA`. In Rust, a
/// function-local `static` is one distinct storage location *per declaration
/// site*, not per name -- so despite sharing names, those were three separate,
/// unconnected copies: a callback registered via `voirs_set_log_callback`
/// could never be observed by `voirs_log_message` (its `LOG_CALLBACK` copy
/// was permanently `None`), and `voirs_is_log_level_enabled` read a *third*
/// copy permanently stuck at the hardcoded default (`Info`). This single
/// module-level static, locked by all three functions, is the fix.
struct LogState {
    callback: Option<VoirsLogCallback>,
    min_level: VoirsLogLevel,
    user_data: *mut c_void,
}

// SAFETY: `user_data` is an opaque pointer we only ever store and hand back
// to the registered callback -- we never dereference it ourselves. The
// caller of `voirs_set_log_callback` is responsible (per that function's
// safety doc) for `user_data` remaining valid for as long as the callback
// may fire and for the callback being safe to invoke from any thread. This
// mirrors `c_api::threading::CallbackInfo`'s identical `unsafe impl
// Send/Sync` for the same kind of opaque FFI user-data pointer.
unsafe impl Send for LogState {}
unsafe impl Sync for LogState {}

static LOG_STATE: Mutex<LogState> = Mutex::new(LogState {
    callback: None,
    min_level: VoirsLogLevel::Info,
    user_data: ptr::null_mut(),
});

/// Get library version information
///
/// # Safety
/// The `buffer` pointer must be valid and point to a writable buffer of at least `buffer_size` bytes.
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
///
/// # Safety
/// The `buffer` pointer must be valid and point to a writable buffer of at least `buffer_size` bytes.
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
///
/// # Safety
/// The `info` pointer must be valid and point to properly allocated memory for a VoirsSystemInfo structure.
#[no_mangle]
pub unsafe extern "C" fn voirs_get_system_info(info: *mut VoirsSystemInfo) -> VoirsErrorCode {
    if info.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    let sys_info = &mut *info;

    // Get CPU cores
    sys_info.cpu_cores = num_cpus::get() as c_uint;

    // Real physical RAM, queried via platform::PlatformInfo (Linux:
    // /proc/meminfo MemTotal, macOS: `sysctl hw.memsize`, Windows:
    // GlobalMemoryStatusEx) -- not a hardcoded constant. Note this reports
    // *total* physical memory, matching PlatformInfo::total_memory; it is
    // not a currently-free/available estimate (see VoirsSystemInfo::
    // available_ram_mb's doc comment).
    let total_ram_bytes = crate::platform::PlatformInfo::current().total_memory;
    sys_info.available_ram_mb = (total_ram_bytes / (1024 * 1024)) as c_uint;

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

    // Runtime GPU probe (see crate::gpu_probe doc comment for exactly what
    // this does and does not detect) -- not a hardcoded constant.
    sys_info.gpu_available = c_uint::from(crate::gpu_probe());

    VoirsErrorCode::Success
}

/// Get memory statistics
///
/// # Safety
/// The `stats` pointer must be valid and point to properly allocated memory for a VoirsMemoryStats structure.
#[no_mangle]
pub unsafe extern "C" fn voirs_get_memory_stats(stats: *mut VoirsMemoryStats) -> VoirsErrorCode {
    if stats.is_null() {
        return VoirsErrorCode::InvalidParameter;
    }

    let mem_stats = &mut *stats;

    // Real allocation statistics from crate::memory's global tracker (the
    // same tracker backing crate::memory::voirs_memory_get_stats and
    // exercised by RefCountedBuffer/pool_allocate/pool_deallocate) --
    // previously this branch was permanently unreachable because
    // get_global_memory_stats() unconditionally returned None, so every
    // caller received hardcoded placeholder numbers instead.
    let real_stats = crate::memory::get_memory_stats();
    mem_stats.total_allocated = real_stats.total_bytes_allocated as c_uint;
    mem_stats.peak_usage = real_stats.peak_bytes_allocated as c_uint;
    mem_stats.active_allocations = real_stats.current_allocations as c_uint;
    mem_stats.total_allocations = real_stats.total_allocations as c_uint;
    mem_stats.total_deallocations = real_stats.total_deallocations as c_uint;
    mem_stats.fragmentation_ratio = if real_stats.total_bytes_allocated > 0 {
        // Fraction of all-time-allocated bytes that have since been freed
        // again -- the same formula voirs_get_memory_fragmentation() (c_api/
        // allocator.rs) already uses for the allocator-level equivalent of
        // this metric, kept consistent here.
        (1.0 - (real_stats.current_bytes_allocated as f32
            / real_stats.total_bytes_allocated as f32))
            .clamp(0.0, 1.0)
    } else {
        0.0
    };

    VoirsErrorCode::Success
}

/// Set log callback function
///
/// # Safety
/// The `user_data` pointer, if not null, must be valid for the lifetime of the logging system.
/// The `callback` function pointer, if provided, must be valid and callable from any thread.
#[no_mangle]
pub unsafe extern "C" fn voirs_set_log_callback(
    callback: Option<VoirsLogCallback>,
    min_level: VoirsLogLevel,
    user_data: *mut c_void,
) -> VoirsErrorCode {
    let mut state = LOG_STATE.lock();
    state.callback = callback;
    state.min_level = min_level;
    state.user_data = user_data;

    VoirsErrorCode::Success
}

/// Log a message with specified level
///
/// # Safety
/// The `message` pointer must be valid and point to a null-terminated C string.
/// The `file` pointer, if not null, must point to a valid null-terminated C string.
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

    // Copy the callback + threshold + user_data out and release the lock
    // *before* invoking the callback: the callback is arbitrary C code that
    // may legally call back into voirs_log_message/voirs_is_log_level_enabled/
    // voirs_set_log_callback on the same thread, and parking_lot::Mutex is
    // not reentrant -- holding the lock across the call would self-deadlock.
    let (callback, min_level, user_data) = {
        let state = LOG_STATE.lock();
        (state.callback, state.min_level, state.user_data)
    };

    if let Some(callback) = callback {
        if (level as u32) >= (min_level as u32) {
            callback(level, message, file, line, user_data);
        }
    }

    VoirsErrorCode::Success
}

/// Validate pointer and size parameters
///
/// # Safety
/// The `buffer` pointer, if not null, must point to valid memory of at least `size` bytes.
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
///
/// # Safety
/// This function performs arithmetic calculations and is safe to call with any input values.
#[no_mangle]
pub unsafe extern "C" fn voirs_calculate_aligned_size(size: c_uint, alignment: c_uint) -> c_uint {
    if alignment == 0 || (alignment & (alignment - 1)) != 0 {
        return size; // Invalid alignment, return original size
    }

    (size + alignment - 1) & !(alignment - 1)
}

/// Check if a value is within specified range
///
/// # Safety
/// This function performs value comparisons and is safe to call with any input values.
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
///
/// # Safety
/// This function performs value comparisons and is safe to call with any input values.
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
///
/// # Safety
/// The `buffer` pointer must be valid and point to a writable buffer of at least `buffer_size` bytes.
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

/// Estimate this process's own resident memory usage (distinct from
/// `crate::memory::get_memory_stats()`, which tracks *this crate's* audio
/// buffer allocations specifically). Used by `voirs_get_process_memory_usage`
/// for a whole-process view via real OS queries (procfs/getrusage/
/// GetProcessMemoryInfo depending on platform and enabled features).
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
///
/// # Safety
/// This function queries system memory statistics and is safe to call at any time.
#[no_mangle]
pub unsafe extern "C" fn voirs_get_process_memory_usage() -> c_uint {
    estimate_current_memory_usage()
}

/// Check if logging is enabled for a given level
///
/// # Safety
/// This function checks logging configuration and is safe to call at any time.
#[no_mangle]
pub unsafe extern "C" fn voirs_is_log_level_enabled(level: VoirsLogLevel) -> c_int {
    let min_level = LOG_STATE.lock().min_level;
    if (level as u32) >= (min_level as u32) {
        1
    } else {
        0
    }
}

/// Reset memory statistics counters
///
/// # Safety
/// This function resets internal memory tracking counters and is safe to call at any time.
#[no_mangle]
pub unsafe extern "C" fn voirs_reset_memory_stats() -> VoirsErrorCode {
    crate::memory::reset_memory_stats();
    VoirsErrorCode::Success
}

/// Validate audio format parameters
///
/// # Safety
/// This function validates audio format parameters and is safe to call with any input values.
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
///
/// # Safety
/// This function performs buffer size calculations and is safe to call with any input values.
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
            let version = version_str.to_str().unwrap_or_default();
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

            // Regression check for the hardcoded `8192` fallback: the
            // reported RAM must match a real, independently-computed
            // platform query exactly (a fabricated constant would not, on
            // any host with a different amount of physical memory), and
            // must never be the specific old hardcoded value on a host that
            // doesn't actually have exactly 8192 MB of RAM.
            let expected_ram_mb =
                (crate::platform::PlatformInfo::current().total_memory / (1024 * 1024)) as c_uint;
            assert_eq!(info.available_ram_mb, expected_ram_mb);

            // Regression check for the hardcoded `gpu_available = 0`: the
            // reported flag must track the real runtime probe, not a
            // constant.
            let expected_gpu = c_uint::from(crate::gpu_probe());
            assert_eq!(info.gpu_available, expected_gpu);
            assert!(info.gpu_available == 0 || info.gpu_available == 1);
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
            let description = desc_str.to_str().unwrap_or_default();
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
            // Explicitly establish the baseline (min_level = Info) rather
            // than assuming it: LOG_STATE is now genuinely shared mutable
            // state (that is the point of the fix), so this test must not
            // depend on no other test in this process having changed it.
            voirs_set_log_callback(None, VoirsLogLevel::Info, ptr::null_mut());

            // Test that higher levels are enabled when min level is Info
            assert_eq!(voirs_is_log_level_enabled(VoirsLogLevel::Info), 1);
            assert_eq!(voirs_is_log_level_enabled(VoirsLogLevel::Warning), 1);
            assert_eq!(voirs_is_log_level_enabled(VoirsLogLevel::Error), 1);
        }
    }

    /// Regression test for the three-independent-`static mut`-copies bug:
    /// a callback registered via `voirs_set_log_callback` must actually be
    /// observed and invoked by `voirs_log_message` (a *different* function),
    /// and the configured minimum level must actually be honored by
    /// `voirs_is_log_level_enabled` (a *third* function) -- all three
    /// reading the same shared state instead of three disconnected copies
    /// that could never see each other's writes.
    #[test]
    fn test_log_callback_state_shared_across_functions() {
        use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};

        static RECEIVED_LEVEL: AtomicI32 = AtomicI32::new(-1);
        static CALL_COUNT: AtomicU32 = AtomicU32::new(0);
        static RECEIVED_USER_DATA: AtomicU32 = AtomicU32::new(0);

        extern "C" fn recording_callback(
            level: VoirsLogLevel,
            _message: *const c_char,
            _file: *const c_char,
            _line: c_uint,
            user_data: *mut c_void,
        ) {
            RECEIVED_LEVEL.store(level as i32, Ordering::SeqCst);
            RECEIVED_USER_DATA.store(user_data as usize as u32, Ordering::SeqCst);
            CALL_COUNT.fetch_add(1, Ordering::SeqCst);
        }

        unsafe {
            let sentinel_user_data = 0x1234_u32 as *mut c_void;
            let set_result = voirs_set_log_callback(
                Some(recording_callback),
                VoirsLogLevel::Warning,
                sentinel_user_data,
            );
            assert_eq!(set_result, VoirsErrorCode::Success);

            // The min_level set by voirs_set_log_callback must be visible to
            // voirs_is_log_level_enabled -- a different function.
            assert_eq!(
                voirs_is_log_level_enabled(VoirsLogLevel::Info),
                0,
                "Info is below the configured Warning threshold"
            );
            assert_eq!(voirs_is_log_level_enabled(VoirsLogLevel::Warning), 1);
            assert_eq!(voirs_is_log_level_enabled(VoirsLogLevel::Error), 1);

            // A below-threshold message must be suppressed: the callback
            // registered by voirs_set_log_callback must NOT fire.
            let below = std::ffi::CString::new("info message").expect("no interior NUL");
            voirs_log_message(VoirsLogLevel::Info, below.as_ptr(), ptr::null(), 1);
            assert_eq!(
                CALL_COUNT.load(Ordering::SeqCst),
                0,
                "sub-threshold message must not invoke the callback"
            );

            // An at-threshold message must reach the callback that was
            // registered by a DIFFERENT function -- this is the crux of the
            // fix (previously the callback copy read here was always None).
            let at_threshold = std::ffi::CString::new("warning message").expect("no interior NUL");
            voirs_log_message(
                VoirsLogLevel::Warning,
                at_threshold.as_ptr(),
                ptr::null(),
                2,
            );
            assert_eq!(
                CALL_COUNT.load(Ordering::SeqCst),
                1,
                "voirs_log_message must invoke the callback registered via voirs_set_log_callback"
            );
            assert_eq!(
                RECEIVED_LEVEL.load(Ordering::SeqCst),
                VoirsLogLevel::Warning as i32
            );
            assert_eq!(
                RECEIVED_USER_DATA.load(Ordering::SeqCst),
                sentinel_user_data as usize as u32
            );

            // Restore the default so later tests in this process (e.g. under
            // plain `cargo test`, which is single-process/multi-threaded
            // unlike nextest's per-test process isolation) see a clean slate.
            voirs_set_log_callback(None, VoirsLogLevel::Info, ptr::null_mut());
        }
    }

    #[test]
    fn test_reset_memory_stats() {
        use crate::memory::{get_memory_stats, RefCountedBuffer};

        unsafe {
            // Allocate something real so there is nonzero state to reset.
            let _buffer = RefCountedBuffer::new(vec![1.0, 2.0, 3.0, 4.0], 44100, 1);
            assert!(
                get_memory_stats().total_allocations > 0,
                "precondition: at least one real allocation must be tracked"
            );

            let result = voirs_reset_memory_stats();
            assert_eq!(result, VoirsErrorCode::Success);

            let stats_after_reset = get_memory_stats();
            assert_eq!(
                stats_after_reset.total_allocations, 0,
                "voirs_reset_memory_stats must actually reset crate::memory's \
                 counters, not be a no-op"
            );
            assert_eq!(stats_after_reset.total_bytes_allocated, 0);
        }
    }

    #[test]
    fn test_enhanced_memory_stats_reflects_real_allocations() {
        use crate::memory::{reset_memory_stats, RefCountedBuffer};

        unsafe {
            reset_memory_stats();

            let mut baseline = VoirsMemoryStats::default();
            assert_eq!(
                voirs_get_memory_stats(&mut baseline),
                VoirsErrorCode::Success
            );
            assert_eq!(baseline.total_allocated, 0);
            assert_eq!(baseline.active_allocations, 0);

            // Keep the buffer alive across the "after alloc" measurement.
            let buffer = RefCountedBuffer::new(vec![0.0f32; 1024], 44100, 1);

            let mut after_alloc = VoirsMemoryStats::default();
            assert_eq!(
                voirs_get_memory_stats(&mut after_alloc),
                VoirsErrorCode::Success
            );
            // This is the property a fabricated constant cannot satisfy:
            // the reported stats must actually change in response to a real
            // allocation made through this crate's tracked buffer type.
            assert!(after_alloc.total_allocated > baseline.total_allocated);
            assert_eq!(after_alloc.active_allocations, 1);
            assert_eq!(after_alloc.total_allocations, 1);
            // Exactly one allocation has ever happened since the reset, so
            // the historical peak equals the (also single-allocation) total
            // -- and in particular peak <= total, the inverse of the old
            // fabricated `peak = total * 1.25`.
            assert_eq!(after_alloc.peak_usage, after_alloc.total_allocated);

            drop(buffer);

            let mut after_drop = VoirsMemoryStats::default();
            assert_eq!(
                voirs_get_memory_stats(&mut after_drop),
                VoirsErrorCode::Success
            );
            assert_eq!(after_drop.active_allocations, 0);
            assert_eq!(after_drop.total_deallocations, 1);
            // Peak usage is monotonic non-decreasing even after the buffer
            // is freed (it tracks the historical high-water mark).
            assert!(after_drop.peak_usage >= after_alloc.peak_usage);

            assert!(
                after_alloc.fragmentation_ratio >= 0.0 && after_alloc.fragmentation_ratio <= 1.0
            );
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
