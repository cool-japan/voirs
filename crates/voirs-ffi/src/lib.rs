//! # VoiRS FFI (Foreign Function Interface)
//!
//! C-compatible bindings for VoiRS speech synthesis framework with comprehensive
//! language support for C/C++, Python, Node.js, and WebAssembly.
//!
//! ## Features
//!
//! - **C API**: Complete C-compatible interface for maximum portability
//! - **Python Bindings**: PyO3-based Python integration with NumPy support
//! - **Node.js Bindings**: N-API bindings for JavaScript/TypeScript
//! - **WebAssembly**: WASM bindings for browser-based synthesis
//! - **Zero-Copy Operations**: Efficient memory management across FFI boundaries
//! - **Thread Safety**: Comprehensive threading support with work-stealing schedulers
//! - **Platform Integration**: Native integration with Windows, macOS, and Linux
//!
//! ## Module Organization
//!
//! - [`c_api`]: Core C API functions for synthesis, audio, and voice management
//! - `python`: PyO3 bindings for Python integration (requires `python` feature)
//! - `nodejs`: N-API bindings for Node.js (requires `nodejs` feature)
//! - `wasm`: WebAssembly bindings (requires `wasm` feature)
//! - [`memory`]: Advanced memory management with custom allocators and zero-copy operations
//! - [`threading`]: Thread pools, synchronization primitives, and callback management
//! - [`error`]: Comprehensive error handling with i18n support
//! - [`performance`]: Performance monitoring and optimization utilities
//! - [`platform`]: Platform-specific integrations (Windows, macOS, Linux)
//! - [`utils`]: Utility functions for audio processing, string conversion, and performance analysis
//!
//! ## Quick Start (C API)
//!
//! ```c
//! #include "voirs_ffi.h"
//!
//! // Initialize pipeline
//! VoirsPipelineHandle* pipeline = voirs_create_pipeline();
//!
//! // Synthesize speech
//! VoirsAudioBuffer* buffer = NULL;
//! VoirsErrorCode result = voirs_synthesize(
//!     pipeline,
//!     "Hello, world!",
//!     &buffer
//! );
//!
//! if (result == VOIRS_SUCCESS) {
//!     // Process audio...
//!     voirs_free_audio_buffer(buffer);
//! }
//!
//! voirs_destroy_pipeline(pipeline);
//! ```
//!
//! ## Quick Start (Python)
//!
//! ```python
//! from voirs import VoirsPipeline
//!
//! # Create pipeline
//! pipeline = VoirsPipeline()
//!
//! # Synthesize speech
//! result = pipeline.synthesize("Hello, world!")
//! audio_data = result.audio_data  # NumPy array
//! ```
//!
//! ## Safety
//!
//! All FFI functions are marked as `unsafe` and require careful handling:
//! - Null pointer checks for all pointer parameters
//! - Proper memory management (use provided free functions)
//! - Thread safety guarantees where documented
//! - No undefined behavior when contracts are followed
//!
//! ## Performance
//!
//! The FFI layer is designed for minimal overhead:
//! - Zero-copy operations where possible
//! - Efficient memory pooling
//! - SIMD-optimized audio processing
//! - Work-stealing thread pools for parallelism

// Allow pedantic lints that are acceptable for audio/DSP processing code
#![allow(clippy::cast_precision_loss)] // Acceptable for audio sample conversions
#![allow(clippy::cast_possible_truncation)] // Controlled truncation in audio processing
#![allow(clippy::cast_sign_loss)] // Intentional in index calculations
#![allow(clippy::missing_errors_doc)] // Many internal functions with self-documenting error types
#![allow(clippy::missing_panics_doc)] // Panics are documented where relevant
#![allow(clippy::unused_self)] // Some trait implementations require &self for consistency
#![allow(clippy::must_use_candidate)] // Not all return values need must_use annotation
#![allow(clippy::doc_markdown)] // Technical terms don't all need backticks
#![allow(clippy::unnecessary_wraps)] // Result wrappers maintained for API consistency
#![allow(clippy::float_cmp)] // Exact float comparisons are intentional in some contexts
#![allow(clippy::match_same_arms)] // Pattern matching clarity sometimes requires duplication
#![allow(clippy::module_name_repetitions)] // Type names often repeat module names
#![allow(clippy::struct_excessive_bools)] // Config structs naturally have many boolean flags
#![allow(clippy::too_many_lines)] // Some functions are inherently complex
#![allow(clippy::needless_pass_by_value)] // Some functions designed for ownership transfer
#![allow(clippy::similar_names)] // Many similar variable names in algorithms
#![allow(clippy::unused_async)] // Public API functions may need async for consistency
#![allow(clippy::needless_range_loop)] // Range loops sometimes clearer than iterators
#![allow(clippy::uninlined_format_args)] // Explicit argument names can improve clarity
#![allow(clippy::manual_clamp)] // Manual clamping sometimes clearer
#![allow(clippy::return_self_not_must_use)] // Not all builder methods need must_use
#![allow(clippy::cast_possible_wrap)] // Controlled wrapping in processing code
#![allow(clippy::cast_lossless)] // Explicit casts preferred for clarity
#![allow(clippy::wildcard_imports)] // Prelude imports are convenient and standard
#![allow(clippy::format_push_string)] // Sometimes more readable than alternative
#![allow(clippy::redundant_closure_for_method_calls)] // Closures sometimes needed for type inference
#![allow(clippy::too_many_arguments)] // Some functions naturally need many parameters
#![allow(clippy::field_reassign_with_default)] // Sometimes clearer than builder pattern
#![allow(clippy::trivially_copy_pass_by_ref)] // API consistency more important
#![allow(clippy::await_holding_lock)] // Controlled lock holding in async contexts

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
    VoirsPipeline as SdkPipeline,
};

pub mod c_api;
pub mod config;
pub mod error;
pub mod memory;
#[cfg(feature = "nodejs")]
pub mod nodejs;
pub mod performance;
pub mod platform;
#[cfg(feature = "python")]
pub mod python;
pub mod threading;
pub mod types;
pub mod utils;
#[cfg(feature = "wasm")]
pub mod wasm;

// Re-export for convenience
pub use c_api::*;
pub use error::*;
pub use performance::*;
pub use types::*;
pub use utils::audio::VoirsAudioAnalysis;

// Export Python module when feature is enabled
// Python types are exported directly from the python module
#[cfg(feature = "python")]
pub use python::{PyAudioBuffer, PySynthesisConfig, PyVoiceInfo, VoirsPipeline as PyVoirsPipeline};

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
            let boxed_slice = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
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
    pipelines: HashMap<u32, Arc<SdkPipeline>>,
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

    fn add_pipeline(&mut self, pipeline: SdkPipeline) -> u32 {
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

    fn get_pipeline(&self, id: u32) -> Option<Arc<SdkPipeline>> {
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

    /// Whether `id` refers to a `VOIRS_BENCHMARK_MODE` placeholder handle
    /// rather than a real, synthesis-capable pipeline. Lets call sites give
    /// callers an honest, specific error ("this handle is a benchmark
    /// placeholder") instead of the generic "invalid pipeline ID" when a
    /// caller passes a placeholder handle to an operation that requires a
    /// real pipeline (voice management, synthesis, ...).
    ///
    /// Takes `&self` on an already-held `MutexGuard<PipelineManager>` (see
    /// call sites in `c_api::voice` / `c_api::threading`) rather than
    /// re-locking `PIPELINE_MANAGER` itself, since `parking_lot::Mutex` is
    /// not reentrant and a second `.lock()` from the same thread while the
    /// first guard is still alive would deadlock.
    pub(crate) fn is_placeholder(&self, id: u32) -> bool {
        self.placeholder_pipelines.contains(&id)
    }

    fn count(&self) -> usize {
        self.pipelines.len() + self.placeholder_pipelines.len()
    }
}

/// Build an honest `voirs_get_last_error()` message for a pipeline ID that
/// `PipelineManager::get_pipeline()` failed to resolve, distinguishing a
/// genuinely unknown/destroyed ID from a `VOIRS_BENCHMARK_MODE` placeholder
/// handle (which passes `voirs_is_pipeline_valid()` by design, but backs no
/// real `SdkPipeline` and therefore cannot perform synthesis or voice
/// operations).
///
/// Pure/lock-free by design: callers must determine `is_placeholder` via
/// `PipelineManager::is_placeholder()` while they already hold the
/// `PIPELINE_MANAGER` guard, then pass the result in here (see doc on
/// `is_placeholder` for why this function must not lock the manager itself).
pub(crate) fn invalid_pipeline_message(id: u32, is_placeholder: bool) -> String {
    if is_placeholder {
        format!(
            "Pipeline {id} is a VOIRS_BENCHMARK_MODE placeholder handle: it has no \
             real model loaded and cannot perform synthesis or voice operations. \
             Create a pipeline without VOIRS_BENCHMARK_MODE=1 set for real synthesis."
        )
    } else {
        format!("Invalid pipeline ID: {id}")
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

/// Run `f` (a private `*_impl()`-style fallible operation), and if it
/// returns `Err`, ensure `voirs_get_last_error()` reports a diagnostic for
/// *this* call specifically: `f`'s own [`set_last_error`] call while it ran
/// (e.g. "Invalid UTF-8 in config: ...", or the honest
/// `crate::invalid_pipeline_message`), if any, or `fallback` otherwise.
///
/// Several C API entry points are a thin `pub extern "C" fn` wrapper around
/// exactly such a `*_impl()`. Naively checking "is *any* error currently
/// pending" (`voirs_has_error()`) to decide whether to install `fallback` is
/// unsound two different ways, both fixed here by snapshotting the message
/// immediately before calling `f` and only trusting a *change* in it:
///
/// - **Clobbering `f`'s own specific message**: an unconditional
///   `set_last_error(fallback())` after `f` returns `Err` would silently
///   overwrite whatever specific diagnostic `f` had just set for this exact
///   failure.
/// - **Leaking a stale message from a wholly unrelated earlier call**: `f`
///   may have message-less `Err` paths (e.g. a `pipeline_id == 0` check with
///   nothing pipeline-specific to say). If a *previous, unrelated* call on
///   this thread left a message behind and was never cleared, a naive
///   "is any error pending" check can't tell that message apart from one
///   `f` just set for *this* call -- and would wrongly skip installing
///   `fallback`, leaving `voirs_get_last_error()` reporting old, irrelevant
///   text for a failure it doesn't describe. Comparing the message
///   *before* vs. *after* `f` runs distinguishes "unchanged stale leftover"
///   from "`f` just set something new" correctly in both directions, and
///   (unlike unconditionally clearing at entry) leaves a still-pending
///   message from an earlier call untouched when `f` *succeeds* -- callers
///   are only ever guaranteed a specific error after **this** call failed.
///
/// `fallback` is lazily evaluated (an `impl FnOnce`, not an already-formatted
/// `String`) so building the generic message costs nothing on the common
/// path where a specific one already won.
pub(crate) fn run_with_fallback_error<T, E>(
    f: impl FnOnce() -> std::result::Result<T, E>,
    fallback: impl FnOnce(&E) -> String,
) -> std::result::Result<T, E> {
    let before = get_last_error();
    let result = f();
    if let Err(ref e) = result {
        if get_last_error() == before {
            set_last_error(fallback(e));
        }
    }
    result
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
    // Return pointers to null-terminated C string literals. A plain Rust `&str`
    // literal is NOT null-terminated, so handing its `.as_ptr()` to `CStr::from_ptr`
    // (as C/Python/Node callers do) reads past the end into adjacent memory — UB
    // that surfaces as nondeterministic garbage or panics under the full test suite.
    let message: &CStr = match code {
        VoirsErrorCode::Success => c"Success",
        VoirsErrorCode::InvalidParameter => c"Invalid parameter",
        VoirsErrorCode::InitializationFailed => c"Initialization failed",
        VoirsErrorCode::SynthesisFailed => c"Synthesis failed",
        VoirsErrorCode::VoiceNotFound => c"Voice not found",
        VoirsErrorCode::IoError => c"I/O error",
        VoirsErrorCode::OutOfMemory => c"Out of memory",
        VoirsErrorCode::OperationCancelled => c"Operation cancelled",
        VoirsErrorCode::InternalError => c"Internal error",
    };

    message.as_ptr()
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

/// Cheap, non-initializing runtime probe for GPU availability.
///
/// Shared by the C (`c_api::utils::voirs_get_system_info`), Python
/// (`python::pipeline::VoirsPipeline::is_gpu_available`), and Node.js
/// (`nodejs::is_gpu_available`) bindings so all three language bindings
/// report the same answer instead of drifting independently.
///
/// This intentionally does **not** create a CUDA or Metal device context
/// (context creation can be slow, print vendor diagnostics to stderr, or
/// have other side effects unsuitable for a "just tell me if a GPU is
/// there" query) -- it only inspects the `CUDA_VISIBLE_DEVICES` environment
/// variable using NVIDIA's convention. This makes it CUDA-oriented: on a
/// Metal-only host (e.g. Apple Silicon) with no `CUDA_VISIBLE_DEVICES` set,
/// this returns `false` even though Metal acceleration may be usable
/// through `voirs-acoustic`'s `gpu` feature. Reporting `false` without
/// positive evidence is the honest default for this probe: it never claims
/// GPU availability it cannot back up, unlike the previous
/// `cfg!(feature = "gpu")` (a compile-time constant baked into the binary,
/// identical for every process regardless of actual hardware) or the
/// hardcoded `gpu_available = 0` this replaces.
pub(crate) fn gpu_probe() -> bool {
    gpu_available_from_env(std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref())
}

/// Decide GPU availability from the value of `CUDA_VISIBLE_DEVICES`.
///
/// Heuristic (NVIDIA convention):
/// * unset (`None`) -> `false` (cannot confirm without initializing a device)
/// * empty string -> `false` (all GPUs masked)
/// * exactly `"-1"` -> `false` (all GPUs masked)
/// * any other value (e.g. `"0"`, `"0,1"`) -> `true`
pub(crate) fn gpu_available_from_env(cuda_visible_devices: Option<&str>) -> bool {
    match cuda_visible_devices {
        Some(value) => {
            let trimmed = value.trim();
            !trimmed.is_empty() && trimmed != "-1"
        }
        None => false,
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
        assert_eq!(c_str.to_str().unwrap_or_default(), "Success");

        let message = voirs_error_message(VoirsErrorCode::InvalidParameter);
        let c_str = unsafe { CStr::from_ptr(message) };
        assert_eq!(c_str.to_str().unwrap_or_default(), "Invalid parameter");

        let message = voirs_error_message(VoirsErrorCode::InternalError);
        let c_str = unsafe { CStr::from_ptr(message) };
        assert_eq!(c_str.to_str().unwrap_or_default(), "Internal error");
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

    /// Regression test for the "outer wrapper clobbers a more specific inner
    /// error" bug class: found by empirically observing that
    /// `voirs_create_pipeline()` (`c_api::core`) discarded `create_pipeline_impl`'s
    /// detailed `"Pipeline creation failed: <real voirs_sdk error>"` message,
    /// unconditionally overwriting it with the uninformative
    /// `"Failed to create pipeline: InitializationFailed"` -- the same class
    /// of bug `c_api::voice`'s wrappers already guarded against, but that
    /// guard had never been applied to pipeline creation/destruction.
    #[test]
    fn test_run_with_fallback_error_preserves_specific_message_from_f() {
        clear_last_error();
        let result: std::result::Result<(), &str> = run_with_fallback_error(
            || {
                set_last_error("specific inner diagnostic".to_string());
                Err("boom")
            },
            |_| "generic outer fallback".to_string(),
        );
        assert!(result.is_err());
        assert_eq!(
            get_last_error().as_deref(),
            Some("specific inner diagnostic")
        );
    }

    /// Complementary case: when `f` fails WITHOUT setting its own message,
    /// the fallback must still be installed -- this isn't a no-op, it's
    /// specifically non-clobbering (only skips the fallback when `f` truly
    /// changed the message itself).
    #[test]
    fn test_run_with_fallback_error_installs_fallback_when_f_sets_nothing() {
        clear_last_error();
        let result: std::result::Result<(), &str> =
            run_with_fallback_error(|| Err("boom"), |_| "generic outer fallback".to_string());
        assert!(result.is_err());
        assert_eq!(get_last_error().as_deref(), Some("generic outer fallback"));
    }

    /// Regression test for the companion "stale message from a wholly
    /// unrelated earlier call" bug: naively checking "is any error currently
    /// pending" (instead of snapshotting the message before/after `f` runs)
    /// would mistake a leftover message from a previous, unrelated failed
    /// call for "`f` already handled this one" and skip the fallback --
    /// leaving `voirs_get_last_error()` reporting old, irrelevant text
    /// instead of a diagnostic for the failure that actually just happened.
    #[test]
    fn test_run_with_fallback_error_does_not_mistake_stale_message_for_fs_own() {
        clear_last_error();
        set_last_error("stale message from a totally unrelated earlier call".to_string());

        let result: std::result::Result<(), &str> =
            run_with_fallback_error(|| Err("boom"), |_| "this call's own message".to_string());

        assert!(result.is_err());
        assert_eq!(get_last_error().as_deref(), Some("this call's own message"));
    }

    /// `f` succeeding must never touch a still-pending message left by an
    /// earlier, unrelated call -- `run_with_fallback_error` only ever
    /// installs a message in response to `f`'s own `Err`, matching the
    /// existing sticky-until-explicitly-cleared-or-overwritten contract
    /// `voirs_clear_error()`'s existence as a distinct public API implies.
    #[test]
    fn test_run_with_fallback_error_leaves_pending_message_untouched_on_success() {
        clear_last_error();
        set_last_error("earlier unrelated failure".to_string());

        let result: std::result::Result<i32, &str> =
            run_with_fallback_error(|| Ok(42), |_| "should never be used".to_string());

        assert_eq!(result, Ok(42));
        assert_eq!(
            get_last_error().as_deref(),
            Some("earlier unrelated failure")
        );
    }

    #[test]
    fn test_enhanced_error_handling() {
        // Test that error messages are properly stored and retrieved
        clear_last_error();
        assert_eq!(voirs_has_error(), 0);
        assert!(get_last_error().is_none());

        set_last_error("Test error message".to_string());
        assert!(voirs_has_error() != 0);

        // Use internal Rust API directly (avoids C string conversion issues in some pyo3 contexts)
        let error = get_last_error();
        assert!(error.is_some());
        assert_eq!(error.as_deref(), Some("Test error message"));

        voirs_clear_error();
        assert!(voirs_has_error() == 0);
        assert!(get_last_error().is_none());
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
            let stats_str = std::ffi::CStr::from_ptr(stats_json)
                .to_str()
                .unwrap_or_default();
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

    #[test]
    fn test_gpu_available_from_env_heuristic() {
        // Visible devices -> available.
        assert!(gpu_available_from_env(Some("0")));
        assert!(gpu_available_from_env(Some("0,1")));
        assert!(gpu_available_from_env(Some(" 0 ")));
        // Masked / unset -> unavailable. The probe never claims availability
        // it can't back up with positive evidence.
        assert!(!gpu_available_from_env(Some("")));
        assert!(!gpu_available_from_env(Some("-1")));
        assert!(!gpu_available_from_env(Some(" -1 ")));
        assert!(!gpu_available_from_env(None));
    }

    #[test]
    fn test_pipeline_manager_distinguishes_placeholder_from_real() {
        let mut manager = PipelineManager::new();
        let placeholder_id = manager.add_placeholder_pipeline();

        assert!(manager.is_valid_pipeline(placeholder_id));
        assert!(
            manager.is_placeholder(placeholder_id),
            "a benchmark-mode handle must be reported as a placeholder"
        );
        // A placeholder handle is deliberately absent from `pipelines`, so
        // any real-pipeline lookup honestly fails instead of fabricating one.
        assert!(manager.get_pipeline(placeholder_id).is_none());

        // A never-issued ID is neither valid nor a placeholder.
        assert!(!manager.is_valid_pipeline(9999));
        assert!(!manager.is_placeholder(9999));

        assert_eq!(
            invalid_pipeline_message(placeholder_id, true),
            invalid_pipeline_message(placeholder_id, manager.is_placeholder(placeholder_id))
        );
        assert!(invalid_pipeline_message(placeholder_id, true).contains("VOIRS_BENCHMARK_MODE"));
        assert!(!invalid_pipeline_message(9999, false).contains("VOIRS_BENCHMARK_MODE"));
    }

    #[test]
    fn test_gpu_probe_matches_current_env() {
        // gpu_probe() must be a pure function of CUDA_VISIBLE_DEVICES, not a
        // hardcoded constant: it should agree with gpu_available_from_env()
        // applied to whatever is actually set in this process's environment
        // right now (nextest runs each test in its own process, so no other
        // test can race this env var).
        let expected =
            gpu_available_from_env(std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref());
        assert_eq!(gpu_probe(), expected);
    }
}
