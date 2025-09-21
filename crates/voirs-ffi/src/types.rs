//! FFI-safe type definitions.

use std::os::raw::{c_char, c_float, c_int, c_uint};

/// FFI-safe voice information
#[repr(C)]
#[derive(Debug, Clone)]
pub struct VoirsVoiceInfo {
    pub id: *mut c_char,
    pub name: *mut c_char,
    pub language: *mut c_char,
    pub quality: crate::VoirsQualityLevel,
    pub is_available: c_int, // 0 = false, 1 = true
}

impl Default for VoirsVoiceInfo {
    fn default() -> Self {
        Self {
            id: std::ptr::null_mut(),
            name: std::ptr::null_mut(),
            language: std::ptr::null_mut(),
            quality: crate::VoirsQualityLevel::Medium,
            is_available: 0,
        }
    }
}

/// FFI-safe voice list
#[repr(C)]
#[derive(Debug)]
pub struct VoirsVoiceList {
    pub voices: *mut VoirsVoiceInfo,
    pub count: c_uint,
}

impl Default for VoirsVoiceList {
    fn default() -> Self {
        Self {
            voices: std::ptr::null_mut(),
            count: 0,
        }
    }
}

/// FFI-safe streaming operation handle
pub type VoirsStreamHandle = c_uint;

/// Enhanced audio chunk callback type - called when audio chunk is ready
/// Parameters: pipeline_id, chunk_data, chunk_size, sample_rate, channels, user_data
pub type VoirsAudioChunkCallback = extern "C" fn(
    pipeline_id: c_uint,
    chunk_data: *const c_float,
    chunk_size: c_uint,
    sample_rate: c_uint,
    channels: c_uint,
    user_data: *mut std::ffi::c_void,
);

/// Enhanced progress callback type - called periodically to report synthesis progress
/// Parameters: pipeline_id, progress (0.0-1.0), estimated_remaining_ms, user_data
pub type VoirsProgressCallback = extern "C" fn(
    pipeline_id: c_uint,
    progress: c_float,
    estimated_remaining_ms: c_uint,
    user_data: *mut std::ffi::c_void,
);

/// Error callback type - called when an error occurs during streaming
/// Parameters: pipeline_id, error_code, error_message, user_data
pub type VoirsErrorCallback = extern "C" fn(
    pipeline_id: c_uint,
    error_code: crate::VoirsErrorCode,
    error_message: *const c_char,
    user_data: *mut std::ffi::c_void,
);

/// Completion callback type - called when synthesis is complete
/// Parameters: pipeline_id, total_samples, total_duration_ms, user_data
pub type VoirsCompletionCallback = extern "C" fn(
    pipeline_id: c_uint,
    total_samples: c_uint,
    total_duration_ms: c_uint,
    user_data: *mut std::ffi::c_void,
);

/// Legacy streaming callback function type (for backward compatibility)
/// Parameters: audio_chunk, chunk_size, user_data
/// Returns: 0 to continue, non-zero to stop
pub type VoirsStreamingCallback =
    extern "C" fn(*const c_float, c_uint, *mut std::ffi::c_void) -> c_int;

/// FFI-safe streaming configuration
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VoirsStreamingConfig {
    pub chunk_size: c_uint,           // Size of audio chunks in samples
    pub max_latency_ms: c_uint,       // Maximum acceptable latency
    pub buffer_count: c_uint,         // Number of buffers for streaming
    pub enable_realtime: c_int,       // 1 = enable real-time mode, 0 = disable
    pub enable_progress: c_int,       // 1 = enable progress callbacks, 0 = disable
    pub progress_interval_ms: c_uint, // Progress callback interval
}

impl Default for VoirsStreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1024,          // 1024 samples per chunk
            max_latency_ms: 100,       // 100ms max latency
            buffer_count: 4,           // 4 buffers
            enable_realtime: 1,        // Enable real-time by default
            enable_progress: 1,        // Enable progress by default
            progress_interval_ms: 100, // Progress every 100ms
        }
    }
}

/// FFI-safe performance optimization configuration
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VoirsPerformanceConfig {
    pub enable_simd: c_int,         // 1 = enable SIMD, 0 = disable
    pub enable_batching: c_int,     // 1 = enable batch operations, 0 = disable
    pub cache_line_size: c_uint,    // Cache line size for alignment
    pub prefetch_distance: c_uint,  // Prefetch distance for memory access
    pub parallel_threshold: c_uint, // Minimum size for parallel processing
}

impl Default for VoirsPerformanceConfig {
    fn default() -> Self {
        Self {
            enable_simd: if cfg!(target_feature = "sse2") || cfg!(target_feature = "neon") {
                1
            } else {
                0
            },
            enable_batching: 1,
            cache_line_size: 64, // Common cache line size
            prefetch_distance: 64,
            parallel_threshold: 1024,
        }
    }
}

/// FFI-safe batch operation
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VoirsBatchOperation {
    pub operation_type: c_uint, // 0=mix, 1=scale, 2=convert, etc.
    pub input_buffer1: *const c_float,
    pub input_buffer2: *const c_float,
    pub output_buffer: *mut c_float,
    pub buffer_size: c_uint,
    pub parameter: c_float, // gain, scale factor, etc.
}

/// FFI-safe VoiRS configuration
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VoirsConfig {
    pub sample_rate: c_uint,         // Output sample rate (Hz)
    pub quality_level: c_float,      // Quality level (0.0 to 1.0)
    pub use_gpu: c_int,              // 0 = false, 1 = true
    pub max_concurrent_ops: c_uint,  // Maximum concurrent operations
    pub enable_cross_lingual: c_int, // 0 = false, 1 = true
}

impl Default for VoirsConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,      // Standard sample rate
            quality_level: 0.8,      // High quality by default
            use_gpu: 0,              // CPU by default
            max_concurrent_ops: 4,   // Reasonable default
            enable_cross_lingual: 0, // Disabled by default
        }
    }
}

/// FFI-safe pipeline configuration
#[repr(C)]
#[derive(Debug, Clone)]
pub struct VoirsPipelineConfig {
    pub use_gpu: c_int, // 0 = false, 1 = true
    pub num_threads: c_uint,
    pub cache_dir: *mut c_char,
    pub device: *mut c_char,
}

impl Default for VoirsPipelineConfig {
    fn default() -> Self {
        Self {
            use_gpu: 0,
            num_threads: 0, // 0 = auto-detect
            cache_dir: std::ptr::null_mut(),
            device: std::ptr::null_mut(),
        }
    }
}
