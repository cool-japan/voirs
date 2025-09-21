//! # VoiRS FFI (Foreign Function Interface)
//!
//! C-compatible bindings for VoiRS speech synthesis framework.
//! Allows integration with C/C++, Python, and other languages.

use parking_lot::Mutex;
use std::{
    collections::HashMap,
    ffi::{CStr, CString},
    os::raw::{c_char, c_float, c_int, c_uint},
    ptr,
    sync::Arc,
};
use voirs_sdk::{
    audio::AudioBuffer,
    error::{Result, VoirsError},
    types::{AudioFormat, LanguageCode, QualityLevel, SynthesisConfig},
    VoirsPipeline,
};

pub mod c_api;
pub mod config;
pub mod error;
pub mod memory;
pub mod nodejs;
pub mod performance;
pub mod platform;
pub mod python;
pub mod threading;
pub mod types;
pub mod utils;
pub mod wasm;

// Re-export for convenience
pub use c_api::*;
pub use error::*;
pub use performance::*;
pub use types::*;
pub use utils::audio::VoirsAudioAnalysis;

// Export Python module when feature is enabled
#[cfg(feature = "python")]
pub use python::pyo3_bindings::*;

// Export Node.js module when feature is enabled
#[cfg(feature = "nodejs")]
pub use nodejs::napi_bindings::*;

// Export WASM module when feature is enabled
#[cfg(feature = "wasm")]
pub use wasm::wasm_bindings::*;

/// FFI-safe error codes
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VoirsErrorCode {
    Success = 0,
    InvalidParameter = 1,
    InitializationFailed = 2,
    SynthesisFailed = 3,
    VoiceNotFound = 4,
    IoError = 5,
    OutOfMemory = 6,
    OperationCancelled = 7,
    InternalError = 99,
}

/// FFI-safe audio format enum
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VoirsAudioFormat {
    Wav = 0,
    Flac = 1,
    Mp3 = 2,
    Opus = 3,
    Ogg = 4,
}

impl From<AudioFormat> for VoirsAudioFormat {
    fn from(format: AudioFormat) -> Self {
        match format {
            AudioFormat::Wav => Self::Wav,
            AudioFormat::Flac => Self::Flac,
            AudioFormat::Mp3 => Self::Mp3,
            AudioFormat::Opus => Self::Opus,
            AudioFormat::Ogg => Self::Ogg,
        }
    }
}

impl From<VoirsAudioFormat> for AudioFormat {
    fn from(format: VoirsAudioFormat) -> Self {
        match format {
            VoirsAudioFormat::Wav => Self::Wav,
            VoirsAudioFormat::Flac => Self::Flac,
            VoirsAudioFormat::Mp3 => Self::Mp3,
            VoirsAudioFormat::Opus => Self::Opus,
            VoirsAudioFormat::Ogg => Self::Ogg,
        }
    }
}

/// FFI-safe quality level enum
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VoirsQualityLevel {
    Low = 0,
    Medium = 1,
    High = 2,
    Ultra = 3,
}

impl From<QualityLevel> for VoirsQualityLevel {
    fn from(quality: QualityLevel) -> Self {
        match quality {
            QualityLevel::Low => Self::Low,
            QualityLevel::Medium => Self::Medium,
            QualityLevel::High => Self::High,
            QualityLevel::Ultra => Self::Ultra,
        }
    }
}

impl From<VoirsQualityLevel> for QualityLevel {
    fn from(quality: VoirsQualityLevel) -> Self {
        match quality {
            VoirsQualityLevel::Low => Self::Low,
            VoirsQualityLevel::Medium => Self::Medium,
            VoirsQualityLevel::High => Self::High,
            VoirsQualityLevel::Ultra => Self::Ultra,
        }
    }
}

/// FFI-safe synthesis configuration
#[repr(C)]
#[derive(Debug, Clone)]
pub struct VoirsSynthesisConfig {
    pub speaking_rate: c_float,
    pub pitch_shift: c_float,
    pub volume_gain: c_float,
    pub enable_enhancement: c_int, // 0 = false, 1 = true
    pub output_format: VoirsAudioFormat,
    pub sample_rate: c_uint,
    pub quality: VoirsQualityLevel,
}

impl Default for VoirsSynthesisConfig {
    fn default() -> Self {
        let config = SynthesisConfig::default();
        Self {
            speaking_rate: config.speaking_rate,
            pitch_shift: config.pitch_shift,
            volume_gain: config.volume_gain,
            enable_enhancement: if config.enable_enhancement { 1 } else { 0 },
            output_format: config.output_format.into(),
            sample_rate: config.sample_rate,
            quality: config.quality.into(),
        }
    }
}

impl From<VoirsSynthesisConfig> for SynthesisConfig {
    fn from(config: VoirsSynthesisConfig) -> Self {
        Self {
            speaking_rate: config.speaking_rate,
            pitch_shift: config.pitch_shift,
            volume_gain: config.volume_gain,
            enable_enhancement: config.enable_enhancement != 0,
            output_format: config.output_format.into(),
            sample_rate: config.sample_rate,
            quality: config.quality.into(),
            language: LanguageCode::EnUs,  // Default language
            effects: Vec::new(),           // No effects by default
            streaming_chunk_size: None,    // Use default chunk size
            seed: None,                    // No seed by default
            enable_emotion: false,         // No emotion by default
            emotion_type: None,            // No emotion type
            emotion_intensity: 0.7,        // Default intensity
            emotion_preset: None,          // No preset
            auto_emotion_detection: false, // No auto detection
            enable_cloning: false,
            cloning_method: None,
            cloning_quality: 0.85,
            enable_conversion: false,
            conversion_target: None,
            realtime_conversion: false,
            enable_singing: false,
            singing_voice_type: None,
            singing_technique: None,
            musical_key: None,
            tempo: None,
            enable_spatial: false,
            listener_position: None,
            hrtf_enabled: false,
            room_size: None,
            reverb_level: 0.3,
        }
    }
}

/// FFI-safe audio buffer
#[repr(C)]
#[derive(Debug)]
pub struct VoirsAudioBuffer {
    pub samples: *mut c_float,
    pub length: c_uint,
    pub sample_rate: c_uint,
    pub channels: c_uint,
    pub duration: c_float,
}

impl VoirsAudioBuffer {
    /// Create from Rust AudioBuffer
    pub fn from_audio_buffer(audio: AudioBuffer) -> Self {
        let samples = audio.samples().to_vec();
        let length = samples.len() as c_uint;
        let sample_rate = audio.sample_rate();
        let channels = audio.channels();
        let duration = audio.duration();

        // Allocate C-compatible buffer
        let mut c_samples = samples.into_boxed_slice();
        let samples_ptr = c_samples.as_mut_ptr();
        std::mem::forget(c_samples); // Prevent deallocation

        Self {
            samples: samples_ptr,
            length,
            sample_rate,
            channels,
            duration,
        }
    }

    /// Convert to Rust AudioBuffer
    ///
    /// # Safety
    ///
    /// This function is unsafe because it dereferences raw pointers.
    /// The caller must ensure that:
    /// - `self.samples` is a valid pointer to at least `self.length` f32 values
    /// - The memory referenced by `self.samples` remains valid for the duration of this call
    pub unsafe fn to_audio_buffer(&self) -> AudioBuffer {
        let samples = std::slice::from_raw_parts(self.samples, self.length as usize).to_vec();
        AudioBuffer::new(samples, self.sample_rate, self.channels)
    }

    /// Free the audio buffer
    ///
    /// # Safety
    ///
    /// This function is unsafe because it deallocates raw memory.
    /// The caller must ensure that:
    /// - `self.samples` was allocated using the same allocator as used by this library
    /// - This function is called at most once per buffer
    /// - The buffer is not used after calling this function
    pub unsafe fn free(&mut self) {
        if !self.samples.is_null() {
            // Reconstruct the original boxed slice that was forgotten during creation
            // This is safe because we know the samples pointer came from into_boxed_slice()
            let boxed_slice = Box::from_raw(std::slice::from_raw_parts_mut(
                self.samples,
                self.length as usize,
            ));
            // Drop the boxed slice to deallocate the memory
            drop(boxed_slice);
            self.samples = ptr::null_mut();
        }
    }
}

use once_cell::sync::Lazy;

/// Global pipeline manager for FFI
struct PipelineManager {
    pipelines: HashMap<u32, Arc<VoirsPipeline>>,
    placeholder_pipelines: std::collections::HashSet<u32>, // Track placeholder IDs for benchmarking
    next_id: u32,
}

impl PipelineManager {
    fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
            placeholder_pipelines: std::collections::HashSet::new(),
            next_id: 1,
        }
    }

    fn add_pipeline(&mut self, pipeline: VoirsPipeline) -> u32 {
        let id = self.next_id;
        self.pipelines.insert(id, Arc::new(pipeline));
        self.next_id = self.next_id.wrapping_add(1);
        if self.next_id == 0 {
            self.next_id = 1; // Skip 0 as it's reserved for errors
        }
        id
    }

    /// Add a placeholder pipeline for benchmarking (doesn't create actual pipeline)
    fn add_placeholder_pipeline(&mut self) -> u32 {
        let id = self.next_id;
        self.placeholder_pipelines.insert(id);
        self.next_id = self.next_id.wrapping_add(1);
        if self.next_id == 0 {
            self.next_id = 1; // Skip 0 as it's reserved for errors
        }
        id
    }

    fn get_pipeline(&self, id: u32) -> Option<Arc<VoirsPipeline>> {
        self.pipelines.get(&id).cloned()
    }

    fn remove_pipeline(&mut self, id: u32) -> bool {
        // Remove from both real pipelines and placeholders
        let removed_real = self.pipelines.remove(&id).is_some();
        let removed_placeholder = self.placeholder_pipelines.remove(&id);
        removed_real || removed_placeholder
    }

    fn is_valid_pipeline(&self, id: u32) -> bool {
        self.pipelines.contains_key(&id) || self.placeholder_pipelines.contains(&id)
    }

    fn count(&self) -> usize {
        self.pipelines.len() + self.placeholder_pipelines.len()
    }
}

/// Global pipeline manager instance using once_cell for thread-safe initialization
static PIPELINE_MANAGER: Lazy<Mutex<PipelineManager>> =
    Lazy::new(|| Mutex::new(PipelineManager::new()));

thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// Set the last error message for the current thread
pub fn set_last_error(error: String) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = Some(error);
    });
}

/// Clear the last error for the current thread
fn clear_last_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}

/// Get the last error message for the current thread
fn get_last_error() -> Option<String> {
    LAST_ERROR.with(|e| e.borrow().clone())
}

/// Get the global pipeline manager
fn get_pipeline_manager() -> &'static Mutex<PipelineManager> {
    &PIPELINE_MANAGER
}

/// Global tokio runtime for async operations
static TOKIO_RUNTIME: Lazy<Mutex<Option<tokio::runtime::Runtime>>> = Lazy::new(|| Mutex::new(None));

/// Get or create the global tokio runtime
fn get_runtime() -> std::result::Result<tokio::runtime::Handle, VoirsErrorCode> {
    // First try to get the current runtime handle if we're already in a runtime context
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        return Ok(handle);
    }

    // If not in a runtime context, create or get our global runtime
    let mut runtime_guard = TOKIO_RUNTIME.lock();
    if runtime_guard.is_none() {
        let rt =
            tokio::runtime::Runtime::new().map_err(|_| VoirsErrorCode::InitializationFailed)?;
        *runtime_guard = Some(rt);
    }

    match runtime_guard.as_ref() {
        Some(rt) => Ok(rt.handle().clone()),
        None => Err(VoirsErrorCode::InternalError),
    }
}

/// Utility function to convert C string to Rust string
unsafe fn c_str_to_string(c_str: *const c_char) -> Result<String> {
    if c_str.is_null() {
        let err = VoirsError::config_error("Null string pointer");
        set_last_error(format!("{err}"));
        return Err(err);
    }

    CStr::from_ptr(c_str)
        .to_str()
        .map(|s| s.to_string())
        .map_err(|e| {
            let err = VoirsError::config_error(format!("Invalid UTF-8: {e}"));
            set_last_error(format!("{err}"));
            err
        })
}

/// Utility function to convert Rust string to C string
fn string_to_c_str(s: &str) -> *mut c_char {
    match CString::new(s) {
        Ok(c_string) => c_string.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Free a C string allocated by this library
///
/// # Safety
///
/// This function is unsafe because it deallocates raw memory.
/// The caller must ensure that:
/// - `s` was allocated by this library using CString::into_raw() or equivalent
/// - This function is called at most once per string
/// - The string is not used after calling this function
#[no_mangle]
pub unsafe extern "C" fn voirs_free_string(s: *mut c_char) {
    if !s.is_null() {
        let _ = CString::from_raw(s);
    }
}

/// Free an audio buffer allocated by this library
///
/// # Safety
///
/// This function is unsafe because it deallocates raw memory and dereferences raw pointers.
/// The caller must ensure that:
/// - `buffer` was allocated by this library
/// - This function is called at most once per buffer
/// - The buffer is not used after calling this function
#[no_mangle]
pub unsafe extern "C" fn voirs_free_audio_buffer(buffer: *mut VoirsAudioBuffer) {
    if !buffer.is_null() {
        (*buffer).free();
        let _ = Box::from_raw(buffer);
    }
}

/// Convert error code to string description
#[no_mangle]
pub extern "C" fn voirs_error_message(code: VoirsErrorCode) -> *const c_char {
    let message = match code {
        VoirsErrorCode::Success => "Success",
        VoirsErrorCode::InvalidParameter => "Invalid parameter",
        VoirsErrorCode::InitializationFailed => "Initialization failed",
        VoirsErrorCode::SynthesisFailed => "Synthesis failed",
        VoirsErrorCode::VoiceNotFound => "Voice not found",
        VoirsErrorCode::IoError => "I/O error",
        VoirsErrorCode::OutOfMemory => "Out of memory",
        VoirsErrorCode::OperationCancelled => "Operation cancelled",
        VoirsErrorCode::InternalError => "Internal error",
    };

    message.as_ptr() as *const c_char
}

/// Get the last error message for the current thread
#[no_mangle]
pub extern "C" fn voirs_get_last_error() -> *mut c_char {
    match get_last_error() {
        Some(error) => string_to_c_str(&error),
        None => ptr::null_mut(),
    }
}

/// Clear the last error for the current thread
#[no_mangle]
pub extern "C" fn voirs_clear_error() {
    clear_last_error();
}

/// Check if there is a pending error for the current thread
#[no_mangle]
pub extern "C" fn voirs_has_error() -> c_int {
    if get_last_error().is_some() {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_conversion() {
        assert_eq!(VoirsErrorCode::Success as i32, 0);
        assert_eq!(VoirsErrorCode::InvalidParameter as i32, 1);
        assert_eq!(VoirsErrorCode::InitializationFailed as i32, 2);
        assert_eq!(VoirsErrorCode::SynthesisFailed as i32, 3);
        assert_eq!(VoirsErrorCode::VoiceNotFound as i32, 4);
        assert_eq!(VoirsErrorCode::IoError as i32, 5);
        assert_eq!(VoirsErrorCode::OutOfMemory as i32, 6);
        assert_eq!(VoirsErrorCode::InternalError as i32, 99);
    }

    #[test]
    fn test_audio_format_conversion() {
        let format = AudioFormat::Wav;
        let ffi_format: VoirsAudioFormat = format.into();
        let back_format: AudioFormat = ffi_format.into();
        assert_eq!(format, back_format);

        // Test all formats
        let formats = [
            (AudioFormat::Wav, VoirsAudioFormat::Wav),
            (AudioFormat::Flac, VoirsAudioFormat::Flac),
            (AudioFormat::Mp3, VoirsAudioFormat::Mp3),
            (AudioFormat::Opus, VoirsAudioFormat::Opus),
            (AudioFormat::Ogg, VoirsAudioFormat::Ogg),
        ];

        for (original, ffi) in formats {
            let converted: VoirsAudioFormat = original.into();
            assert_eq!(converted, ffi);
            let back: AudioFormat = converted.into();
            assert_eq!(back, original);
        }
    }

    #[test]
    fn test_quality_level_conversion() {
        let qualities = [
            (QualityLevel::Low, VoirsQualityLevel::Low),
            (QualityLevel::Medium, VoirsQualityLevel::Medium),
            (QualityLevel::High, VoirsQualityLevel::High),
            (QualityLevel::Ultra, VoirsQualityLevel::Ultra),
        ];

        for (original, ffi) in qualities {
            let converted: VoirsQualityLevel = original.into();
            assert_eq!(converted, ffi);
            let back: QualityLevel = converted.into();
            assert_eq!(back, original);
        }
    }

    #[test]
    fn test_synthesis_config_conversion() {
        let config = SynthesisConfig::default();
        let ffi_config: VoirsSynthesisConfig = VoirsSynthesisConfig::default();
        let back_config: SynthesisConfig = ffi_config.into();
        assert_eq!(config.speaking_rate, back_config.speaking_rate);
        assert_eq!(config.pitch_shift, back_config.pitch_shift);
        assert_eq!(config.volume_gain, back_config.volume_gain);
        assert_eq!(config.enable_enhancement, back_config.enable_enhancement);
        assert_eq!(config.output_format, back_config.output_format);
        assert_eq!(config.sample_rate, back_config.sample_rate);
        assert_eq!(config.quality, back_config.quality);
    }

    #[test]
    fn test_synthesis_config_default() {
        let config = VoirsSynthesisConfig::default();
        assert_eq!(config.speaking_rate, 1.0);
        assert_eq!(config.pitch_shift, 0.0);
        assert_eq!(config.volume_gain, 0.0);
        assert_eq!(config.enable_enhancement, 1); // true
        assert_eq!(config.output_format, VoirsAudioFormat::Wav);
        assert_eq!(config.sample_rate, 22050);
        assert_eq!(config.quality, VoirsQualityLevel::High);
    }

    #[test]
    fn test_error_message_function() {
        let message = voirs_error_message(VoirsErrorCode::Success);
        let c_str = unsafe { CStr::from_ptr(message) };
        assert_eq!(c_str.to_str().unwrap(), "Success");

        let message = voirs_error_message(VoirsErrorCode::InvalidParameter);
        let c_str = unsafe { CStr::from_ptr(message) };
        assert_eq!(c_str.to_str().unwrap(), "Invalid parameter");

        let message = voirs_error_message(VoirsErrorCode::InternalError);
        let c_str = unsafe { CStr::from_ptr(message) };
        assert_eq!(c_str.to_str().unwrap(), "Internal error");
    }

    #[test]
    fn test_audio_buffer_memory_safety() {
        // Test creating and freeing audio buffer
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let audio = AudioBuffer::new(samples.clone(), 44100, 1);
        let mut ffi_buffer = VoirsAudioBuffer::from_audio_buffer(audio);

        assert_eq!(ffi_buffer.length, 4);
        assert_eq!(ffi_buffer.sample_rate, 44100);
        assert_eq!(ffi_buffer.channels, 1);
        assert!(!ffi_buffer.samples.is_null());

        // Test that samples are correctly stored
        unsafe {
            let samples_slice =
                std::slice::from_raw_parts(ffi_buffer.samples, ffi_buffer.length as usize);
            assert_eq!(samples_slice, &[1.0, 2.0, 3.0, 4.0]);
        }

        // Test freeing memory
        unsafe {
            ffi_buffer.free();
        }
        assert!(ffi_buffer.samples.is_null());
    }

    #[test]
    fn test_string_conversion_utilities() {
        let test_string = "Hello, VoiRS!";
        let c_string = string_to_c_str(test_string);

        assert!(!c_string.is_null());

        unsafe {
            let converted_back = c_str_to_string(c_string);
            assert!(converted_back.is_ok());
            assert_eq!(converted_back.unwrap(), test_string);

            // Free the string
            voirs_free_string(c_string);
        }
    }

    #[test]
    fn test_null_string_handling() {
        unsafe {
            let result = c_str_to_string(std::ptr::null());
            assert!(result.is_err());

            // Test freeing null string (should not crash)
            voirs_free_string(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_voice_info_default() {
        let voice_info = types::VoirsVoiceInfo::default();
        assert!(voice_info.id.is_null());
        assert!(voice_info.name.is_null());
        assert!(voice_info.language.is_null());
        assert_eq!(voice_info.quality, VoirsQualityLevel::Medium);
        assert_eq!(voice_info.is_available, 0);
    }

    #[test]
    fn test_voice_list_default() {
        let voice_list = types::VoirsVoiceList::default();
        assert!(voice_list.voices.is_null());
        assert_eq!(voice_list.count, 0);
    }

    #[test]
    fn test_pipeline_config_default() {
        let config = types::VoirsPipelineConfig::default();
        assert_eq!(config.use_gpu, 0);
        assert_eq!(config.num_threads, 0);
        assert!(config.cache_dir.is_null());
        assert!(config.device.is_null());
    }

    #[test]
    fn test_ffi_struct_sizes() {
        // Ensure structs have reasonable sizes for FFI
        assert!(std::mem::size_of::<VoirsErrorCode>() <= 8);
        assert!(std::mem::size_of::<VoirsAudioFormat>() <= 8);
        assert!(std::mem::size_of::<VoirsQualityLevel>() <= 8);
        assert!(std::mem::size_of::<VoirsSynthesisConfig>() <= 64);
        assert!(std::mem::size_of::<VoirsAudioBuffer>() <= 64);
        assert!(std::mem::size_of::<types::VoirsVoiceInfo>() <= 64);
        assert!(std::mem::size_of::<types::VoirsVoiceList>() <= 16);
        assert!(std::mem::size_of::<types::VoirsPipelineConfig>() <= 32);
    }

    #[test]
    fn test_ffi_struct_alignment() {
        // Ensure structs are properly aligned for FFI
        assert_eq!(
            std::mem::align_of::<VoirsErrorCode>(),
            std::mem::align_of::<i32>()
        );
        assert_eq!(
            std::mem::align_of::<VoirsAudioFormat>(),
            std::mem::align_of::<i32>()
        );
        assert_eq!(
            std::mem::align_of::<VoirsQualityLevel>(),
            std::mem::align_of::<i32>()
        );
    }

    #[test]
    fn test_repr_c_layout() {
        // Test that repr(C) structs have expected field offsets
        use std::mem::offset_of;

        // VoirsSynthesisConfig
        assert_eq!(offset_of!(VoirsSynthesisConfig, speaking_rate), 0);
        assert_eq!(offset_of!(VoirsSynthesisConfig, pitch_shift), 4);
        assert_eq!(offset_of!(VoirsSynthesisConfig, volume_gain), 8);
        assert_eq!(offset_of!(VoirsSynthesisConfig, enable_enhancement), 12);

        // VoirsAudioBuffer
        assert_eq!(offset_of!(VoirsAudioBuffer, samples), 0);
        assert_eq!(offset_of!(VoirsAudioBuffer, length), 8);
        assert_eq!(offset_of!(VoirsAudioBuffer, sample_rate), 12);
        assert_eq!(offset_of!(VoirsAudioBuffer, channels), 16);
        assert_eq!(offset_of!(VoirsAudioBuffer, duration), 20);
    }

    #[test]
    fn test_enhanced_error_handling() {
        // Test that error messages are properly stored and retrieved
        clear_last_error();
        assert!(!voirs_has_error() != 0);

        set_last_error("Test error message".to_string());
        assert!(voirs_has_error() != 0);

        let error_msg = voirs_get_last_error();
        assert!(!error_msg.is_null());

        unsafe {
            let c_str = std::ffi::CStr::from_ptr(error_msg);
            assert_eq!(c_str.to_str().unwrap(), "Test error message");
            voirs_free_string(error_msg);
        }

        voirs_clear_error();
        assert!(voirs_has_error() == 0);
    }

    #[test]
    fn test_memory_management_integration() {
        use crate::memory::{check_memory_leaks, get_memory_stats, reset_memory_stats};

        reset_memory_stats();
        assert!(check_memory_leaks());

        let initial_stats = get_memory_stats();
        assert_eq!(initial_stats.current_allocations, 0);

        // Test memory tracking via our C API
        let stats_json = memory::voirs_memory_get_stats();
        assert!(!stats_json.is_null());

        unsafe {
            let stats_str = std::ffi::CStr::from_ptr(stats_json).to_str().unwrap();
            assert!(stats_str.contains("total_allocations"));
            voirs_free_string(stats_json);
        }

        let leak_check = memory::voirs_memory_check_leaks();
        assert_eq!(leak_check, 1); // Should be leak-free
    }

    #[test]
    fn test_memory_pool_integration() {
        use crate::memory::{pool_allocate, pool_deallocate};

        // Test memory pool allocation
        let buffer1 = pool_allocate(100);
        assert_eq!(buffer1.len(), 100);

        let buffer2 = pool_allocate(100);
        assert_eq!(buffer2.len(), 100);

        // Return to pool
        pool_deallocate(buffer1);
        pool_deallocate(buffer2);

        // Should reuse from pool
        let buffer3 = pool_allocate(100);
        assert_eq!(buffer3.len(), 100);
    }

    #[test]
    fn test_ref_counted_audio_buffer() {
        use crate::memory::RefCountedBuffer;

        let buffer = RefCountedBuffer::new(vec![1.0, 2.0, 3.0, 4.0], 44100, 2);
        assert_eq!(buffer.data(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(buffer.sample_rate(), 44100);
        assert_eq!(buffer.channels(), 2);
        assert_eq!(buffer.ref_count(), 1);

        let buffer2 = buffer.clone();
        assert_eq!(buffer.ref_count(), 2);
        assert_eq!(buffer2.ref_count(), 2);

        drop(buffer2);
        assert_eq!(buffer.ref_count(), 1);
    }
}
