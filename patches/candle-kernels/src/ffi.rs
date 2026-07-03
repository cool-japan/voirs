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

// ---------------------------------------------------------------------------
// candle-core 0.11 quantized "fast path" launchers (fast_mmvq / fast_mmq).
//
// `candle-core` 0.11.0 added two new CUDA-only source files that did not
// exist in 0.10.x: `src/quantized/fast_mmvq.rs` (batch 1..=8 mat-vec) and
// `src/quantized/fast_mmq.rs` (batch >8 tiled mat-mat), both gated behind
// `#[cfg(feature = "cuda")]`. Each declares its own `unsafe extern "C" fn`
// type alias (`PlainLauncher`, `QuantizeLauncher`, `MmqLauncher`) and expects
// `candle_kernels::ffi` to export a matching symbol per quantization dtype.
// The signatures below are copied verbatim from those type aliases so the
// function-pointer coercions at each call site (e.g.
// `let f: PlainLauncher = ffi::launch_mmvq_gguf_q4_0_bf16_plain;`) type-check
// identically to a real CUDA-toolchain build of upstream candle-kernels.
// ---------------------------------------------------------------------------

/// Stub matching `fast_mmvq.rs`'s `PlainLauncher` type alias:
/// `unsafe extern "C" fn(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, b_size, stream)`.
macro_rules! stub_mmvq_plain_launcher {
    ($name:ident) => {
        /// # Safety
        /// This is a stub implementation that always panics. CUDA is not
        /// available on this platform.
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            _vx: *const c_void,
            _vy: *const c_void,
            _dst: *mut c_void,
            _ncols_x: i32,
            _nrows_x: i32,
            _stride_col_y: i32,
            _stride_col_dst: i32,
            _b_size: i32,
            _stream: *mut c_void,
        ) {
            panic!(concat!(
                stringify!($name),
                ": CUDA is not available on this platform. GPU-accelerated \
                 quantized MMVQ inference requires a CUDA-capable device and \
                 the upstream `candle-kernels` crate compiled with CUDA support."
            ));
        }
    };
}

stub_mmvq_plain_launcher!(launch_mmvq_gguf_q4_0_bf16_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q4_1_bf16_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q5_0_bf16_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q5_1_bf16_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q8_0_bf16_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q2_k_bf16_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q3_k_bf16_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q4_k_bf16_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q5_k_bf16_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q6_k_bf16_plain);

stub_mmvq_plain_launcher!(launch_mmvq_gguf_q4_0_f16_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q4_1_f16_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q5_0_f16_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q5_1_f16_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q8_0_f16_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q2_k_f16_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q3_k_f16_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q4_k_f16_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q5_k_f16_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q6_k_f16_plain);

stub_mmvq_plain_launcher!(launch_mmvq_gguf_q4_0_f32_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q4_1_f32_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q5_0_f32_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q5_1_f32_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q8_0_f32_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q2_k_f32_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q3_k_f32_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q4_k_f32_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q5_k_f32_plain);
stub_mmvq_plain_launcher!(launch_mmvq_gguf_q6_k_f32_plain);

/// Stub matching `fast_mmvq.rs`'s activation-quantization call sites:
/// `unsafe extern "C" fn(rhs_ptr, scratch_ptr, k, k_padded, b_size, stream)`.
macro_rules! stub_mmvq_quantize_launcher {
    ($name:ident) => {
        /// # Safety
        /// This is a stub implementation that always panics. CUDA is not
        /// available on this platform.
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            _rhs_ptr: *const c_void,
            _scratch_ptr: *mut c_void,
            _k: i32,
            _k_padded: i32,
            _b_size: i32,
            _stream: *mut c_void,
        ) {
            panic!(concat!(
                stringify!($name),
                ": CUDA is not available on this platform. GPU-accelerated \
                 quantized MMVQ activation quantization requires a CUDA-capable \
                 device and the upstream `candle-kernels` crate compiled with \
                 CUDA support."
            ));
        }
    };
}

stub_mmvq_quantize_launcher!(launch_mmvq_gguf_quantize_q8_1_bf16);
stub_mmvq_quantize_launcher!(launch_mmvq_gguf_quantize_q8_1_f16);
stub_mmvq_quantize_launcher!(launch_mmvq_gguf_quantize_q8_1_f32);

/// Stub matching `fast_mmq.rs`'s `QuantizeLauncher` type alias:
/// `unsafe extern "C" fn(x, ids, vy, type_x, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, stream)`.
///
/// The real upstream symbol names use the GGML `D4`/`DS4`/`D2S6` scale-layout
/// tags verbatim (not snake_case), so `non_snake_case` is allowed here to
/// match the upstream ABI exactly.
macro_rules! stub_mmq_quantize_launcher {
    ($name:ident) => {
        /// # Safety
        /// This is a stub implementation that always panics. CUDA is not
        /// available on this platform.
        #[no_mangle]
        #[allow(non_snake_case)]
        pub unsafe extern "C" fn $name(
            _x: *const c_void,
            _ids: *const i32,
            _vy: *mut c_void,
            _type_x: i32,
            _ne00: i64,
            _s01: i64,
            _s02: i64,
            _s03: i64,
            _ne0: i64,
            _ne1: i64,
            _ne2: i64,
            _ne3: i64,
            _stream: *mut c_void,
        ) {
            panic!(concat!(
                stringify!($name),
                ": CUDA is not available on this platform. GPU-accelerated \
                 quantized MMQ activation quantization requires a CUDA-capable \
                 device and the upstream `candle-kernels` crate compiled with \
                 CUDA support."
            ));
        }
    };
}

stub_mmq_quantize_launcher!(launch_mmq_quantize_q8_1_D4);
stub_mmq_quantize_launcher!(launch_mmq_quantize_q8_1_DS4);
stub_mmq_quantize_launcher!(launch_mmq_quantize_q8_1_D2S6);

/// Stub matching `fast_mmq.rs`'s `MmqLauncher` type alias:
/// `unsafe extern "C" fn(tmp_fixup, x, y, dst, ncols_x, nrows_x, ncols_y, stride_row_x, stride_col_dst, cc, nsm, smpbo, warp_size, stream)`.
macro_rules! stub_mmq_gguf_launcher {
    ($name:ident) => {
        /// # Safety
        /// This is a stub implementation that always panics. CUDA is not
        /// available on this platform.
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            _tmp_fixup: *mut c_void,
            _x: *const c_void,
            _y: *const c_void,
            _dst: *mut c_void,
            _ncols_x: i64,
            _nrows_x: i64,
            _ncols_y: i64,
            _stride_row_x: i64,
            _stride_col_dst: i64,
            _cc: i32,
            _nsm: i32,
            _smpbo: i64,
            _warp_size: i32,
            _stream: *mut c_void,
        ) {
            panic!(concat!(
                stringify!($name),
                ": CUDA is not available on this platform. GPU-accelerated \
                 quantized MMQ (tiled matmul) inference requires a CUDA-capable \
                 device and the upstream `candle-kernels` crate compiled with \
                 CUDA support."
            ));
        }
    };
}

stub_mmq_gguf_launcher!(launch_mmq_gguf_q4_0);
stub_mmq_gguf_launcher!(launch_mmq_gguf_q4_1);
stub_mmq_gguf_launcher!(launch_mmq_gguf_q5_0);
stub_mmq_gguf_launcher!(launch_mmq_gguf_q5_1);
stub_mmq_gguf_launcher!(launch_mmq_gguf_q8_0);
stub_mmq_gguf_launcher!(launch_mmq_gguf_q2_k);
stub_mmq_gguf_launcher!(launch_mmq_gguf_q3_k);
stub_mmq_gguf_launcher!(launch_mmq_gguf_q4_k);
stub_mmq_gguf_launcher!(launch_mmq_gguf_q5_k);
stub_mmq_gguf_launcher!(launch_mmq_gguf_q6_k);
