//! MATLAB/Octave bindings for VoiRS evaluation framework
//!
//! This module provides MEX-compatible bindings for MATLAB and GNU Octave, allowing
//! users to evaluate speech synthesis quality directly from MATLAB/Octave scripts.
//!
//! The bindings use a C-compatible interface that can be compiled as MEX files
//! using `mex` command. The functions follow MATLAB conventions for array handling
//! and error reporting.
//!
//! ## Building MEX Files
//!
//! To build the MEX files for MATLAB:
//!
//! ```bash
//! # Build the Rust library
//! cargo build --release --features matlab
//!
//! # Create MEX wrapper (example for Linux)
//! mex -v -largeArrayDims \
//!     -L../target/release \
//!     -lvoirs_evaluation \
//!     evaluate_quality.c
//! ```
//!
//! ## Example Usage (MATLAB)
//!
//! ```matlab
//! % Load audio files
//! [generated, fs] = audioread('generated.wav');
//! [reference, fs] = audioread('reference.wav');
//!
//! % Evaluate quality
//! result = evaluate_quality(generated, reference, fs);
//!
//! fprintf('Quality score: %.3f\n', result.overall_score);
//! fprintf('PESQ: %.3f\n', result.pesq);
//! fprintf('STOI: %.3f\n', result.stoi);
//! ```

use crate::prelude::*;
use crate::EvaluationError;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_int, c_void};
use std::ptr;
use std::sync::Arc;
use tokio::runtime::Runtime;
use voirs_sdk::AudioBuffer;

/// MATLAB mxArray representation (opaque)
#[repr(C)]
pub struct MxArray {
    _private: [u8; 0],
}

/// MATLAB class IDs
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MxClassID {
    Unknown = 0,
    Double = 6,
    Single = 7,
    Int8 = 8,
    UInt8 = 9,
    Int16 = 10,
    UInt16 = 11,
    Int32 = 12,
    UInt32 = 13,
    Int64 = 14,
    UInt64 = 15,
    Struct = 2,
}

/// MATLAB-compatible quality result structure
#[repr(C)]
pub struct MatlabQualityResult {
    /// Overall quality score (0-1)
    pub overall_score: c_double,
    /// PESQ score (1-4.5)
    pub pesq: c_double,
    /// STOI score (0-1)
    pub stoi: c_double,
    /// MCD score (lower is better)
    pub mcd: c_double,
    /// Signal-to-noise ratio
    pub snr: c_double,
    /// Spectral distortion
    pub spectral_distortion: c_double,
    /// Processing time in seconds
    pub processing_time: c_double,
}

/// MATLAB-compatible pronunciation result
#[repr(C)]
pub struct MatlabPronunciationResult {
    /// Overall pronunciation score (0-1)
    pub overall_score: c_double,
    /// Phoneme accuracy
    pub phoneme_accuracy: c_double,
    /// Word accuracy
    pub word_accuracy: c_double,
    /// Fluency score
    pub fluency: c_double,
    /// Prosody score
    pub prosody: c_double,
    /// Processing time in seconds
    pub processing_time: c_double,
}

/// MATLAB-compatible comparison result
#[repr(C)]
pub struct MatlabComparisonResult {
    /// System A score
    pub system_a_score: c_double,
    /// System B score
    pub system_b_score: c_double,
    /// Winner: 0 = A, 1 = B, 2 = Tie
    pub winner: c_int,
    /// Statistical significance (p-value)
    pub p_value: c_double,
    /// Effect size (Cohen's d)
    pub effect_size: c_double,
    /// Processing time in seconds
    pub processing_time: c_double,
}

/// Opaque handle for MATLAB evaluator
#[repr(C)]
pub struct MatlabEvaluatorHandle {
    _private: [u8; 0],
}

/// Internal evaluator wrapper with runtime
struct MatlabEvaluatorWrapper {
    quality_evaluator: Arc<QualityEvaluator>,
    pronunciation_evaluator: Arc<PronunciationEvaluatorImpl>,
    comparative_evaluator: Arc<ComparativeEvaluatorImpl>,
    runtime: Runtime,
}

// ============================================================================
// MATLAB API Functions
// ============================================================================

/// Initialize evaluation system for MATLAB
///
/// # Safety
///
/// Must be called before any other evaluation functions
#[no_mangle]
pub unsafe extern "C" fn matlab_evaluation_init() -> *mut MatlabEvaluatorHandle {
    let runtime = match Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return ptr::null_mut(),
    };

    let quality_evaluator = match runtime.block_on(QualityEvaluator::new()) {
        Ok(eval) => Arc::new(eval),
        Err(_) => return ptr::null_mut(),
    };

    let pronunciation_evaluator = match runtime.block_on(PronunciationEvaluatorImpl::new()) {
        Ok(eval) => Arc::new(eval),
        Err(_) => return ptr::null_mut(),
    };

    let comparative_evaluator = match runtime.block_on(ComparativeEvaluatorImpl::new()) {
        Ok(eval) => Arc::new(eval),
        Err(_) => return ptr::null_mut(),
    };

    let wrapper = Box::new(MatlabEvaluatorWrapper {
        quality_evaluator,
        pronunciation_evaluator,
        comparative_evaluator,
        runtime,
    });

    Box::into_raw(wrapper) as *mut MatlabEvaluatorHandle
}

/// Cleanup evaluation system
///
/// # Safety
///
/// Handle must be valid and not already freed
#[no_mangle]
pub unsafe extern "C" fn matlab_evaluation_cleanup(handle: *mut MatlabEvaluatorHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle as *mut MatlabEvaluatorWrapper);
    }
}

/// Evaluate quality from MATLAB arrays
///
/// # Safety
///
/// - `handle` must be a valid evaluator handle
/// - `generated` and `reference` must be valid double arrays
/// - `result` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn matlab_evaluate_quality(
    handle: *mut MatlabEvaluatorHandle,
    generated: *const c_double,
    generated_length: c_int,
    reference: *const c_double,
    reference_length: c_int,
    sample_rate: c_double,
    result: *mut MatlabQualityResult,
) -> c_int {
    if handle.is_null() || generated.is_null() || result.is_null() {
        return -1;
    }

    let wrapper = &*(handle as *const MatlabEvaluatorWrapper);
    let start_time = std::time::Instant::now();

    // Convert MATLAB double arrays to f32
    let generated_samples: Vec<f32> =
        std::slice::from_raw_parts(generated, generated_length as usize)
            .iter()
            .map(|&x| x as f32)
            .collect();

    let generated_buf = AudioBuffer::new(
        generated_samples,
        sample_rate as u32,
        1, // Assume mono; could be extended for stereo
    );

    let reference_buf = if !reference.is_null() && reference_length > 0 {
        let reference_samples: Vec<f32> =
            std::slice::from_raw_parts(reference, reference_length as usize)
                .iter()
                .map(|&x| x as f32)
                .collect();
        Some(AudioBuffer::new(reference_samples, sample_rate as u32, 1))
    } else {
        None
    };

    // Evaluate
    let eval_result = wrapper
        .runtime
        .block_on(wrapper.quality_evaluator.evaluate_quality(
            &generated_buf,
            reference_buf.as_ref(),
            None,
        ));

    let processing_time = start_time.elapsed().as_secs_f64();

    match eval_result {
        Ok(quality) => {
            *result = MatlabQualityResult {
                overall_score: quality.overall_score as c_double,
                pesq: quality
                    .component_scores
                    .get("pesq")
                    .copied()
                    .unwrap_or(-1.0) as c_double,
                stoi: quality
                    .component_scores
                    .get("stoi")
                    .copied()
                    .unwrap_or(-1.0) as c_double,
                mcd: quality.component_scores.get("mcd").copied().unwrap_or(-1.0) as c_double,
                snr: quality.component_scores.get("snr").copied().unwrap_or(-1.0) as c_double,
                spectral_distortion: quality
                    .component_scores
                    .get("spectral_distortion")
                    .copied()
                    .unwrap_or(-1.0) as c_double,
                processing_time,
            };
            0
        }
        Err(_) => -2,
    }
}

/// Evaluate pronunciation from MATLAB arrays
///
/// # Safety
///
/// - `handle` must be a valid evaluator handle
/// - `audio` must be a valid double array
/// - `text` must be a valid null-terminated C string
/// - `result` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn matlab_evaluate_pronunciation(
    handle: *mut MatlabEvaluatorHandle,
    audio: *const c_double,
    audio_length: c_int,
    text: *const c_char,
    sample_rate: c_double,
    result: *mut MatlabPronunciationResult,
) -> c_int {
    if handle.is_null() || audio.is_null() || text.is_null() || result.is_null() {
        return -1;
    }

    let wrapper = &*(handle as *const MatlabEvaluatorWrapper);
    let start_time = std::time::Instant::now();

    // Convert audio
    let audio_samples: Vec<f32> = std::slice::from_raw_parts(audio, audio_length as usize)
        .iter()
        .map(|&x| x as f32)
        .collect();

    let audio_buf = AudioBuffer::new(audio_samples, sample_rate as u32, 1);

    // Convert text
    let text_str = match CStr::from_ptr(text).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    // Evaluate pronunciation with default config
    let eval_result = wrapper.runtime.block_on(
        wrapper
            .pronunciation_evaluator
            .evaluate_pronunciation(&audio_buf, text_str, None),
    );

    let processing_time = start_time.elapsed().as_secs_f64();

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

            *result = MatlabPronunciationResult {
                overall_score: pron.overall_score as c_double,
                phoneme_accuracy: phoneme_accuracy as c_double,
                word_accuracy: word_accuracy as c_double,
                fluency: pron.fluency_score as c_double,
                prosody: ((pron.stress_accuracy + pron.intonation_accuracy) / 2.0) as c_double,
                processing_time,
            };
            0
        }
        Err(_) => -2,
    }
}

/// Compare two systems from MATLAB arrays
///
/// # Safety
///
/// - `handle` must be a valid evaluator handle
/// - All audio arrays must be valid
/// - `result` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn matlab_compare_systems(
    handle: *mut MatlabEvaluatorHandle,
    system_a: *const c_double,
    system_a_length: c_int,
    system_b: *const c_double,
    system_b_length: c_int,
    reference: *const c_double,
    reference_length: c_int,
    sample_rate: c_double,
    result: *mut MatlabComparisonResult,
) -> c_int {
    if handle.is_null() || system_a.is_null() || system_b.is_null() || result.is_null() {
        return -1;
    }

    let wrapper = &*(handle as *const MatlabEvaluatorWrapper);
    let start_time = std::time::Instant::now();

    // Convert system A
    let system_a_samples: Vec<f32> = std::slice::from_raw_parts(system_a, system_a_length as usize)
        .iter()
        .map(|&x| x as f32)
        .collect();

    let system_a_buf = AudioBuffer::new(system_a_samples, sample_rate as u32, 1);

    // Convert system B
    let system_b_samples: Vec<f32> = std::slice::from_raw_parts(system_b, system_b_length as usize)
        .iter()
        .map(|&x| x as f32)
        .collect();

    let system_b_buf = AudioBuffer::new(system_b_samples, sample_rate as u32, 1);

    // Convert reference (optional)
    let reference_buf = if !reference.is_null() && reference_length > 0 {
        let reference_samples: Vec<f32> =
            std::slice::from_raw_parts(reference, reference_length as usize)
                .iter()
                .map(|&x| x as f32)
                .collect();
        Some(AudioBuffer::new(reference_samples, sample_rate as u32, 1))
    } else {
        None
    };

    // Compare - use trait method
    use crate::traits::ComparativeEvaluator as _;
    let eval_result = wrapper
        .runtime
        .block_on(wrapper.comparative_evaluator.compare_samples(
            &system_a_buf,
            &system_b_buf,
            None,
        ));

    let processing_time = start_time.elapsed().as_secs_f64();

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

            *result = MatlabComparisonResult {
                system_a_score: system_a_score as c_double,
                system_b_score: system_b_score as c_double,
                winner,
                p_value: p_value as f64,
                effect_size: comp.preference_score as f64,
                processing_time,
            };
            0
        }
        Err(_) => -2,
    }
}

/// Get version string for MATLAB
///
/// # Safety
///
/// Returns a static string that does not need to be freed
#[no_mangle]
pub unsafe extern "C" fn matlab_evaluation_version() -> *const c_char {
    static VERSION_CSTRING: once_cell::sync::Lazy<CString> = once_cell::sync::Lazy::new(|| {
        CString::new(crate::VERSION)
            .unwrap_or_else(|_| CString::new("unknown").expect("value should be present"))
    });
    VERSION_CSTRING.as_ptr()
}

// ============================================================================
// Octave-specific Functions
// ============================================================================

/// Octave uses the same API as MATLAB, but we provide type aliases for clarity
pub type OctaveEvaluatorHandle = MatlabEvaluatorHandle;
pub type OctaveQualityResult = MatlabQualityResult;
pub type OctavePronunciationResult = MatlabPronunciationResult;
pub type OctaveComparisonResult = MatlabComparisonResult;

/// Initialize evaluation system for Octave (alias to MATLAB version)
///
/// # Safety
///
/// Must be called before any other evaluation functions
#[no_mangle]
pub unsafe extern "C" fn octave_evaluation_init() -> *mut OctaveEvaluatorHandle {
    matlab_evaluation_init()
}

/// Cleanup evaluation system for Octave (alias to MATLAB version)
///
/// # Safety
///
/// Handle must be valid and not already freed
#[no_mangle]
pub unsafe extern "C" fn octave_evaluation_cleanup(handle: *mut OctaveEvaluatorHandle) {
    matlab_evaluation_cleanup(handle)
}

// ============================================================================
// Helper Functions for MEX File Generation
// ============================================================================

/// Generate MEX wrapper code for quality evaluation
///
/// This function generates C code that can be compiled as a MEX file
#[must_use]
pub fn generate_quality_mex_wrapper() -> String {
    r#"
/* MEX wrapper for VoiRS quality evaluation
 * Usage: result = evaluate_quality(generated, reference, sample_rate)
 */

#include "mex.h"
#include <string.h>

/* External C functions from Rust library */
extern void* matlab_evaluation_init(void);
extern void matlab_evaluation_cleanup(void*);
extern int matlab_evaluate_quality(
    void* handle,
    const double* generated, int generated_length,
    const double* reference, int reference_length,
    double sample_rate,
    void* result
);

typedef struct {
    double overall_score;
    double pesq;
    double stoi;
    double mcd;
    double snr;
    double spectral_distortion;
    double processing_time;
} MatlabQualityResult;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Check arguments */
    if (nrhs < 2 || nrhs > 3) {
        mexErrMsgIdAndTxt("VoiRS:evaluate_quality:nrhs",
            "Two or three inputs required: generated, reference, [sample_rate]");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("VoiRS:evaluate_quality:nlhs",
            "One output required.");
    }

    /* Get inputs */
    double *generated = mxGetPr(prhs[0]);
    int generated_length = (int)mxGetNumberOfElements(prhs[0]);

    double *reference = NULL;
    int reference_length = 0;
    if (nrhs >= 2 && !mxIsEmpty(prhs[1])) {
        reference = mxGetPr(prhs[1]);
        reference_length = (int)mxGetNumberOfElements(prhs[1]);
    }

    double sample_rate = 16000.0;
    if (nrhs >= 3) {
        sample_rate = mxGetScalar(prhs[2]);
    }

    /* Initialize evaluator */
    void *handle = matlab_evaluation_init();
    if (handle == NULL) {
        mexErrMsgIdAndTxt("VoiRS:evaluate_quality:init",
            "Failed to initialize evaluator");
    }

    /* Evaluate */
    MatlabQualityResult result;
    int status = matlab_evaluate_quality(
        handle, generated, generated_length,
        reference, reference_length,
        sample_rate, &result
    );

    /* Cleanup */
    matlab_evaluation_cleanup(handle);

    if (status != 0) {
        mexErrMsgIdAndTxt("VoiRS:evaluate_quality:eval",
            "Evaluation failed");
    }

    /* Create output structure */
    const char *field_names[] = {
        "overall_score", "pesq", "stoi", "mcd",
        "snr", "spectral_distortion", "processing_time"
    };
    plhs[0] = mxCreateStructMatrix(1, 1, 7, field_names);

    mxSetField(plhs[0], 0, "overall_score", mxCreateDoubleScalar(result.overall_score));
    mxSetField(plhs[0], 0, "pesq", mxCreateDoubleScalar(result.pesq));
    mxSetField(plhs[0], 0, "stoi", mxCreateDoubleScalar(result.stoi));
    mxSetField(plhs[0], 0, "mcd", mxCreateDoubleScalar(result.mcd));
    mxSetField(plhs[0], 0, "snr", mxCreateDoubleScalar(result.snr));
    mxSetField(plhs[0], 0, "spectral_distortion", mxCreateDoubleScalar(result.spectral_distortion));
    mxSetField(plhs[0], 0, "processing_time", mxCreateDoubleScalar(result.processing_time));
}
"#
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matlab_evaluator_lifecycle() {
        unsafe {
            let handle = matlab_evaluation_init();
            assert!(!handle.is_null());
            matlab_evaluation_cleanup(handle);
        }
    }

    #[test]
    fn test_octave_evaluator_lifecycle() {
        unsafe {
            let handle = octave_evaluation_init();
            assert!(!handle.is_null());
            octave_evaluation_cleanup(handle);
        }
    }

    #[test]
    fn test_version_string() {
        unsafe {
            let version = matlab_evaluation_version();
            assert!(!version.is_null());
            let version_str = CStr::from_ptr(version).to_str().unwrap();
            assert!(!version_str.is_empty());
        }
    }

    #[test]
    fn test_mex_wrapper_generation() {
        let wrapper = generate_quality_mex_wrapper();
        assert!(wrapper.contains("mexFunction"));
        assert!(wrapper.contains("matlab_evaluate_quality"));
        assert!(wrapper.contains("mxCreateStructMatrix"));
    }

    #[test]
    fn test_matlab_quality_evaluation() {
        unsafe {
            let handle = matlab_evaluation_init();
            assert!(!handle.is_null());

            let generated: Vec<c_double> = (0..16000).map(|i| (i as f64) * 0.0001).collect();
            let reference: Vec<c_double> = (0..16000).map(|i| (i as f64) * 0.00012).collect();
            let mut result = MatlabQualityResult {
                overall_score: 0.0,
                pesq: 0.0,
                stoi: 0.0,
                mcd: 0.0,
                snr: 0.0,
                spectral_distortion: 0.0,
                processing_time: 0.0,
            };

            let status = matlab_evaluate_quality(
                handle,
                generated.as_ptr(),
                generated.len() as c_int,
                reference.as_ptr(),
                reference.len() as c_int,
                16000.0,
                &mut result as *mut MatlabQualityResult,
            );

            matlab_evaluation_cleanup(handle);

            assert_eq!(status, 0);

            // For synthetic test data, scores may be outside normal range
            // Just verify we got a result with valid processing time
            assert!(result.processing_time > 0.0);

            // Verify result structure is populated
            assert!(result.overall_score.is_finite());
        }
    }
}
