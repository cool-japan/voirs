//! C++ header-only interface for VoiRS evaluation framework
//!
//! This module provides C++ bindings for the VoiRS evaluation framework through
//! a header-only approach. It generates C-compatible functions that can be called
//! from C++ with RAII wrappers.
//!
//! The generated header file (`voirs_evaluation.hpp`) provides:
//! - RAII wrappers for safe resource management
//! - Exception-safe interfaces
//! - STL container support (std::vector, std::string)
//! - Modern C++ (C++17+) features
//!
//! ## Example Usage (C++)
//!
//! ```cpp
//! #include "voirs_evaluation.hpp"
//!
//! int main() {
//!     try {
//!         // Create quality evaluator
//!         voirs::QualityEvaluator evaluator;
//!
//!         // Prepare audio data
//!         std::vector<float> generated(16000, 0.1f);
//!         std::vector<float> reference(16000, 0.12f);
//!
//!         // Evaluate quality
//!         auto result = evaluator.evaluate_quality(
//!             generated, reference, 16000, 1
//!         );
//!
//!         std::cout << "Quality score: " << result.overall_score << std::endl;
//!         std::cout << "PESQ: " << result.pesq << std::endl;
//!
//!         return 0;
//!     } catch (const voirs::Exception& e) {
//!         std::cerr << "Error: " << e.what() << std::endl;
//!         return 1;
//!     }
//! }
//! ```

use crate::prelude::*;
use crate::{EvaluationError, QualityEvaluationConfig};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_uint, c_void};
use std::ptr;
use std::sync::Arc;
use tokio::runtime::Runtime;
use voirs_sdk::AudioBuffer;

/// Opaque handle for C++ QualityEvaluator
#[repr(C)]
pub struct CppQualityEvaluatorHandle {
    _private: [u8; 0],
}

/// Opaque handle for C++ PronunciationEvaluator
#[repr(C)]
pub struct CppPronunciationEvaluatorHandle {
    _private: [u8; 0],
}

/// Opaque handle for C++ ComparativeEvaluator
#[repr(C)]
pub struct CppComparativeEvaluatorHandle {
    _private: [u8; 0],
}

/// C-compatible quality evaluation result
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CppQualityResult {
    /// Overall quality score (0-1)
    pub overall_score: c_float,
    /// PESQ score (1-4.5), -1 if not available
    pub pesq: c_float,
    /// STOI score (0-1), -1 if not available
    pub stoi: c_float,
    /// MCD score (lower is better), -1 if not available
    pub mcd: c_float,
    /// Signal-to-noise ratio, -1 if not available
    pub snr: c_float,
    /// Spectral distortion, -1 if not available
    pub spectral_distortion: c_float,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Success flag (0 = success, non-zero = error)
    pub error_code: c_int,
}

impl Default for CppQualityResult {
    fn default() -> Self {
        Self {
            overall_score: 0.0,
            pesq: -1.0,
            stoi: -1.0,
            mcd: -1.0,
            snr: -1.0,
            spectral_distortion: -1.0,
            processing_time_ms: 0.0,
            error_code: 0,
        }
    }
}

/// C-compatible pronunciation result
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CppPronunciationResult {
    /// Overall pronunciation score (0-1)
    pub overall_score: c_float,
    /// Phoneme accuracy score
    pub phoneme_accuracy: c_float,
    /// Word accuracy score
    pub word_accuracy: c_float,
    /// Fluency score
    pub fluency: c_float,
    /// Prosody score
    pub prosody: c_float,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Success flag
    pub error_code: c_int,
}

impl Default for CppPronunciationResult {
    fn default() -> Self {
        Self {
            overall_score: 0.0,
            phoneme_accuracy: 0.0,
            word_accuracy: 0.0,
            fluency: 0.0,
            prosody: 0.0,
            processing_time_ms: 0.0,
            error_code: 0,
        }
    }
}

/// C-compatible comparison result
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CppComparisonResult {
    /// System A score
    pub system_a_score: c_float,
    /// System B score
    pub system_b_score: c_float,
    /// Winner: 0 = A, 1 = B, 2 = Tie
    pub winner: c_int,
    /// Statistical significance (p-value), -1 if not available
    pub p_value: f64,
    /// Effect size (Cohen's d), -1 if not available
    pub effect_size: f64,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Success flag
    pub error_code: c_int,
}

impl Default for CppComparisonResult {
    fn default() -> Self {
        Self {
            system_a_score: 0.0,
            system_b_score: 0.0,
            winner: 2, // Tie by default
            p_value: -1.0,
            effect_size: -1.0,
            processing_time_ms: 0.0,
            error_code: 0,
        }
    }
}

/// Internal wrapper for quality evaluator with runtime
struct CppQualityEvaluatorWrapper {
    evaluator: Arc<QualityEvaluator>,
    runtime: Runtime,
}

/// Internal wrapper for pronunciation evaluator with runtime
struct CppPronunciationEvaluatorWrapper {
    evaluator: Arc<PronunciationEvaluatorImpl>,
    runtime: Runtime,
}

/// Internal wrapper for comparative evaluator with runtime
struct CppComparativeEvaluatorWrapper {
    evaluator: Arc<ComparativeEvaluatorImpl>,
    runtime: Runtime,
}

// ============================================================================
// Quality Evaluator C API
// ============================================================================

/// Create a new quality evaluator
///
/// # Safety
///
/// The returned handle must be freed with `cpp_quality_evaluator_free`
#[no_mangle]
pub unsafe extern "C" fn cpp_quality_evaluator_new() -> *mut CppQualityEvaluatorHandle {
    let runtime = match Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return ptr::null_mut(),
    };

    let evaluator = match runtime.block_on(QualityEvaluator::new()) {
        Ok(eval) => eval,
        Err(_) => return ptr::null_mut(),
    };

    let wrapper = Box::new(CppQualityEvaluatorWrapper {
        evaluator: Arc::new(evaluator),
        runtime,
    });

    Box::into_raw(wrapper) as *mut CppQualityEvaluatorHandle
}

/// Free a quality evaluator
///
/// # Safety
///
/// The handle must be valid and not already freed
#[no_mangle]
pub unsafe extern "C" fn cpp_quality_evaluator_free(handle: *mut CppQualityEvaluatorHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle as *mut CppQualityEvaluatorWrapper);
    }
}

/// Evaluate quality of generated audio against reference
///
/// # Safety
///
/// - `handle` must be a valid evaluator handle
/// - `generated` and `reference` must be valid pointers to f32 arrays
/// - `length` must match the actual array lengths
/// - `result` must be a valid pointer to `CppQualityResult`
#[no_mangle]
pub unsafe extern "C" fn cpp_quality_evaluator_evaluate(
    handle: *mut CppQualityEvaluatorHandle,
    generated: *const c_float,
    generated_length: c_uint,
    reference: *const c_float,
    reference_length: c_uint,
    sample_rate: c_uint,
    channels: c_uint,
    result: *mut CppQualityResult,
) -> c_int {
    if handle.is_null() || generated.is_null() || result.is_null() {
        return -1;
    }

    let wrapper = &*(handle as *const CppQualityEvaluatorWrapper);
    let start_time = std::time::Instant::now();

    // Convert input arrays
    let generated_samples = std::slice::from_raw_parts(generated, generated_length as usize);
    let generated_buf = AudioBuffer::new(generated_samples.to_vec(), sample_rate, channels);

    let reference_buf = if !reference.is_null() && reference_length > 0 {
        let reference_samples = std::slice::from_raw_parts(reference, reference_length as usize);
        Some(AudioBuffer::new(
            reference_samples.to_vec(),
            sample_rate,
            channels,
        ))
    } else {
        None
    };

    // Evaluate
    let eval_result = wrapper.runtime.block_on(wrapper.evaluator.evaluate_quality(
        &generated_buf,
        reference_buf.as_ref(),
        None,
    ));

    let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;

    match eval_result {
        Ok(quality) => {
            *result = CppQualityResult {
                overall_score: quality.overall_score,
                pesq: quality
                    .component_scores
                    .get("pesq")
                    .copied()
                    .unwrap_or(-1.0),
                stoi: quality
                    .component_scores
                    .get("stoi")
                    .copied()
                    .unwrap_or(-1.0),
                mcd: quality.component_scores.get("mcd").copied().unwrap_or(-1.0),
                snr: quality.component_scores.get("snr").copied().unwrap_or(-1.0),
                spectral_distortion: quality
                    .component_scores
                    .get("spectral_distortion")
                    .copied()
                    .unwrap_or(-1.0),
                processing_time_ms: processing_time,
                error_code: 0,
            };
            0
        }
        Err(_) => {
            *result = CppQualityResult {
                error_code: -2,
                processing_time_ms: processing_time,
                ..Default::default()
            };
            -2
        }
    }
}

// ============================================================================
// Pronunciation Evaluator C API
// ============================================================================

/// Create a new pronunciation evaluator
///
/// # Safety
///
/// The returned handle must be freed with `cpp_pronunciation_evaluator_free`
#[no_mangle]
pub unsafe extern "C" fn cpp_pronunciation_evaluator_new() -> *mut CppPronunciationEvaluatorHandle {
    let runtime = match Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return ptr::null_mut(),
    };

    let evaluator = match runtime.block_on(PronunciationEvaluatorImpl::new()) {
        Ok(eval) => eval,
        Err(_) => return ptr::null_mut(),
    };

    let wrapper = Box::new(CppPronunciationEvaluatorWrapper {
        evaluator: Arc::new(evaluator),
        runtime,
    });

    Box::into_raw(wrapper) as *mut CppPronunciationEvaluatorHandle
}

/// Free a pronunciation evaluator
///
/// # Safety
///
/// The handle must be valid and not already freed
#[no_mangle]
pub unsafe extern "C" fn cpp_pronunciation_evaluator_free(
    handle: *mut CppPronunciationEvaluatorHandle,
) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle as *mut CppPronunciationEvaluatorWrapper);
    }
}

/// Evaluate pronunciation from audio and text
///
/// # Safety
///
/// - `handle` must be a valid evaluator handle
/// - `audio` must be a valid pointer to f32 array
/// - `text` must be a valid null-terminated C string
/// - `result` must be a valid pointer to `CppPronunciationResult`
#[no_mangle]
pub unsafe extern "C" fn cpp_pronunciation_evaluator_evaluate(
    handle: *mut CppPronunciationEvaluatorHandle,
    audio: *const c_float,
    audio_length: c_uint,
    text: *const c_char,
    sample_rate: c_uint,
    channels: c_uint,
    result: *mut CppPronunciationResult,
) -> c_int {
    if handle.is_null() || audio.is_null() || text.is_null() || result.is_null() {
        return -1;
    }

    let wrapper = &*(handle as *const CppPronunciationEvaluatorWrapper);
    let start_time = std::time::Instant::now();

    // Convert input
    let audio_samples = std::slice::from_raw_parts(audio, audio_length as usize);
    let audio_buf = AudioBuffer::new(audio_samples.to_vec(), sample_rate, channels);

    let text_str = match CStr::from_ptr(text).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    // Evaluate
    let eval_result = wrapper.runtime.block_on(
        wrapper
            .evaluator
            .evaluate_pronunciation(&audio_buf, text_str, None),
    );

    let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;

    match eval_result {
        Ok(pron) => {
            // Calculate average phoneme and word accuracy
            let phoneme_accuracy = if !pron.phoneme_scores.is_empty() {
                pron.phoneme_scores.iter().map(|p| p.accuracy).sum::<f32>()
                    / pron.phoneme_scores.len() as f32
            } else {
                0.0
            };

            let word_accuracy = if !pron.word_scores.is_empty() {
                pron.word_scores.iter().map(|w| w.accuracy).sum::<f32>()
                    / pron.word_scores.len() as f32
            } else {
                0.0
            };

            *result = CppPronunciationResult {
                overall_score: pron.overall_score,
                phoneme_accuracy,
                word_accuracy,
                fluency: pron.fluency_score,
                prosody: (pron.stress_accuracy + pron.intonation_accuracy) / 2.0,
                processing_time_ms: processing_time,
                error_code: 0,
            };
            0
        }
        Err(_) => {
            *result = CppPronunciationResult {
                error_code: -2,
                processing_time_ms: processing_time,
                ..Default::default()
            };
            -2
        }
    }
}

// ============================================================================
// Comparative Evaluator C API
// ============================================================================

/// Create a new comparative evaluator
///
/// # Safety
///
/// The returned handle must be freed with `cpp_comparative_evaluator_free`
#[no_mangle]
pub unsafe extern "C" fn cpp_comparative_evaluator_new() -> *mut CppComparativeEvaluatorHandle {
    let runtime = match Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return ptr::null_mut(),
    };

    let evaluator = match runtime.block_on(ComparativeEvaluatorImpl::new()) {
        Ok(eval) => eval,
        Err(_) => return ptr::null_mut(),
    };

    let wrapper = Box::new(CppComparativeEvaluatorWrapper {
        evaluator: Arc::new(evaluator),
        runtime,
    });

    Box::into_raw(wrapper) as *mut CppComparativeEvaluatorHandle
}

/// Free a comparative evaluator
///
/// # Safety
///
/// The handle must be valid and not already freed
#[no_mangle]
pub unsafe extern "C" fn cpp_comparative_evaluator_free(
    handle: *mut CppComparativeEvaluatorHandle,
) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle as *mut CppComparativeEvaluatorWrapper);
    }
}

/// Compare two audio systems
///
/// # Safety
///
/// - `handle` must be a valid evaluator handle
/// - All audio pointers must be valid
/// - `result` must be a valid pointer to `CppComparisonResult`
#[no_mangle]
pub unsafe extern "C" fn cpp_comparative_evaluator_compare(
    handle: *mut CppComparativeEvaluatorHandle,
    system_a: *const c_float,
    system_a_length: c_uint,
    system_b: *const c_float,
    system_b_length: c_uint,
    reference: *const c_float,
    reference_length: c_uint,
    sample_rate: c_uint,
    channels: c_uint,
    result: *mut CppComparisonResult,
) -> c_int {
    if handle.is_null() || system_a.is_null() || system_b.is_null() || result.is_null() {
        return -1;
    }

    let wrapper = &*(handle as *const CppComparativeEvaluatorWrapper);
    let start_time = std::time::Instant::now();

    // Convert inputs
    let system_a_samples = std::slice::from_raw_parts(system_a, system_a_length as usize);
    let system_a_buf = AudioBuffer::new(system_a_samples.to_vec(), sample_rate, channels);

    let system_b_samples = std::slice::from_raw_parts(system_b, system_b_length as usize);
    let system_b_buf = AudioBuffer::new(system_b_samples.to_vec(), sample_rate, channels);

    let reference_buf = if !reference.is_null() && reference_length > 0 {
        let reference_samples = std::slice::from_raw_parts(reference, reference_length as usize);
        Some(AudioBuffer::new(
            reference_samples.to_vec(),
            sample_rate,
            channels,
        ))
    } else {
        None
    };

    // Compare - use trait method
    use crate::traits::ComparativeEvaluator as _;
    let eval_result = wrapper.runtime.block_on(wrapper.evaluator.compare_samples(
        &system_a_buf,
        &system_b_buf,
        None,
    ));

    let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;

    match eval_result {
        Ok(comp) => {
            // Extract system scores from metric comparisons
            let system_a_score = comp
                .metric_comparisons
                .values()
                .map(|m| m.score_a)
                .sum::<f32>()
                / comp.metric_comparisons.len().max(1) as f32;

            let system_b_score = comp
                .metric_comparisons
                .values()
                .map(|m| m.score_b)
                .sum::<f32>()
                / comp.metric_comparisons.len().max(1) as f32;

            let winner = if comp.preference_score < -0.1 {
                0
            } else if comp.preference_score > 0.1 {
                1
            } else {
                2
            };

            // Get average p-value from statistical significance
            let p_value = comp.statistical_significance.values().copied().sum::<f32>()
                / comp.statistical_significance.len().max(1) as f32;

            *result = CppComparisonResult {
                system_a_score,
                system_b_score,
                winner,
                p_value: p_value as f64,
                effect_size: comp.preference_score as f64,
                processing_time_ms: processing_time,
                error_code: 0,
            };
            0
        }
        Err(_) => {
            *result = CppComparisonResult {
                error_code: -2,
                processing_time_ms: processing_time,
                ..Default::default()
            };
            -2
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Get version string
///
/// # Safety
///
/// The returned string is static and does not need to be freed
#[no_mangle]
pub unsafe extern "C" fn cpp_evaluation_version() -> *const c_char {
    static VERSION_CSTRING: once_cell::sync::Lazy<CString> = once_cell::sync::Lazy::new(|| {
        CString::new(crate::VERSION)
            .unwrap_or_else(|_| CString::new("unknown").expect("value should be present"))
    });
    VERSION_CSTRING.as_ptr()
}

/// Get last error message (thread-local)
///
/// # Safety
///
/// The returned string is valid until the next error or thread exit
#[no_mangle]
pub unsafe extern "C" fn cpp_evaluation_last_error() -> *const c_char {
    thread_local! {
        static LAST_ERROR: std::cell::RefCell<Option<CString>> = std::cell::RefCell::new(None);
    }

    LAST_ERROR.with(|error| error.borrow().as_ref().map_or(ptr::null(), |s| s.as_ptr()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpp_quality_result_default() {
        let result = CppQualityResult::default();
        assert_eq!(result.overall_score, 0.0);
        assert_eq!(result.pesq, -1.0);
        assert_eq!(result.error_code, 0);
    }

    #[test]
    fn test_cpp_pronunciation_result_default() {
        let result = CppPronunciationResult::default();
        assert_eq!(result.overall_score, 0.0);
        assert_eq!(result.error_code, 0);
    }

    #[test]
    fn test_cpp_comparison_result_default() {
        let result = CppComparisonResult::default();
        assert_eq!(result.winner, 2); // Tie
        assert_eq!(result.error_code, 0);
    }

    #[test]
    fn test_version_string() {
        unsafe {
            let version = cpp_evaluation_version();
            assert!(!version.is_null());
            let version_str = CStr::from_ptr(version).to_str().unwrap();
            assert!(!version_str.is_empty());
        }
    }

    #[test]
    fn test_quality_evaluator_lifecycle() {
        unsafe {
            let handle = cpp_quality_evaluator_new();
            assert!(!handle.is_null());
            cpp_quality_evaluator_free(handle);
        }
    }

    #[test]
    fn test_pronunciation_evaluator_lifecycle() {
        unsafe {
            let handle = cpp_pronunciation_evaluator_new();
            assert!(!handle.is_null());
            cpp_pronunciation_evaluator_free(handle);
        }
    }

    #[test]
    fn test_comparative_evaluator_lifecycle() {
        unsafe {
            let handle = cpp_comparative_evaluator_new();
            assert!(!handle.is_null());
            cpp_comparative_evaluator_free(handle);
        }
    }
}
