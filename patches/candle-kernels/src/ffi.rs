//! FFI stubs for `candle-kernels`.
//!
//! On platforms without CUDA (macOS, or any system where `nvcc` is absent)
//! these functions panic with a descriptive error message rather than
//! requiring external CUDA symbols that would cause linker failures.
//!
//! The signatures are identical to the upstream `candle-kernels` so that any
//! code compiled against this stub will link and run correctly for CPU-only
//! workloads.  Only code paths that actually dispatch to a CUDA device will
//! encounter the runtime panic.

#![allow(clippy::too_many_arguments)]
#![allow(unused_variables)]

use core::ffi::c_void;

/// Mixture-of-Experts GEMM using WMMA (tensor core) instructions.
///
/// # Safety
/// This is a stub implementation that always panics.  CUDA is not available
/// on this platform.
#[no_mangle]
pub unsafe extern "C" fn moe_gemm_wmma(
    _input: *const c_void,
    _weights: *const c_void,
    _sorted_token_ids: *const i32,
    _expert_ids: *const i32,
    _topk_weights: *const f32,
    _output: *mut c_void,
    _expert_counts: *mut i32,
    _expert_offsets: *mut i32,
    _num_experts: i32,
    _topk: i32,
    _size_m: i32,
    _size_n: i32,
    _size_k: i32,
    _dtype: i32,
    _is_prefill: bool,
    _stream: i64,
) {
    panic!(
        "moe_gemm_wmma: CUDA is not available on this platform. \
         GPU-accelerated MoE inference requires a CUDA-capable device and the \
         upstream `candle-kernels` crate compiled with CUDA support."
    );
}

/// Mixture-of-Experts GEMM for GGUF-quantised weights.
///
/// # Safety
/// This is a stub implementation that always panics.  CUDA is not available
/// on this platform.
#[no_mangle]
pub unsafe extern "C" fn moe_gemm_gguf(
    _input: *const f32,
    _weights: *const c_void,
    _sorted_token_ids: *const i32,
    _expert_ids: *const i32,
    _topk_weights: *const f32,
    _output: *mut c_void,
    _num_experts: i32,
    _topk: i32,
    _size_m: i32,
    _size_n: i32,
    _size_k: i32,
    _gguf_dtype: i32,
    _stream: i64,
) {
    panic!(
        "moe_gemm_gguf: CUDA is not available on this platform. \
         GPU-accelerated MoE inference requires a CUDA-capable device and the \
         upstream `candle-kernels` crate compiled with CUDA support."
    );
}

/// Mixture-of-Experts GEMM (prefill path) for GGUF-quantised weights.
///
/// # Safety
/// This is a stub implementation that always panics.  CUDA is not available
/// on this platform.
#[no_mangle]
pub unsafe extern "C" fn moe_gemm_gguf_prefill(
    _input: *const c_void,
    _weights: *const u8,
    _sorted_token_ids: *const i32,
    _expert_ids: *const i32,
    _topk_weights: *const f32,
    _output: *mut c_void,
    _num_experts: i32,
    _topk: i32,
    _size_m: i32,
    _size_n: i32,
    _size_k: i32,
    _input_dtype: i32,
    _gguf_dtype: i32,
    _stream: i64,
) {
    panic!(
        "moe_gemm_gguf_prefill: CUDA is not available on this platform. \
         GPU-accelerated MoE inference requires a CUDA-capable device and the \
         upstream `candle-kernels` crate compiled with CUDA support."
    );
}
