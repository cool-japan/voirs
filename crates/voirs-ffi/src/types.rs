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

/// FFI-safe streaming callback function type
/// Parameters: audio_chunk, chunk_size, user_data
/// Returns: 0 to continue, non-zero to stop
pub type VoirsStreamingCallback = extern "C" fn(*const c_float, c_uint, *mut std::ffi::c_void) -> c_int;

/// FFI-safe progress callback function type  
/// Parameters: progress (0.0-1.0), user_data
pub type VoirsProgressCallback = extern "C" fn(c_float, *mut std::ffi::c_void);

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