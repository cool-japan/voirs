//! SIMD-accelerated operations for acoustic processing
//!
//! This module provides SIMD-optimized implementations for CPU-intensive
//! operations in acoustic modeling, including mel computation, matrix operations,
//! and spectral analysis.

use crate::{AcousticError, Result};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub mod audio;
pub mod fft;
pub mod matrix;
pub mod mel;

pub use audio::*;
pub use fft::*;
pub use matrix::*;
pub use mel::*;

/// Fast exponential approximation for audio processing
/// Uses polynomial approximation for speed
#[inline]
fn fast_exp(x: f32) -> f32 {
    // Clamp to reasonable range to avoid overflow
    let x = x.clamp(-10.0, 10.0);

    // Polynomial approximation: exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x3 * x;

    1.0 + x + x2 * 0.5 + x3 * 0.16666667 + x4 * 0.041666667
}

/// Fast logarithm approximation for audio processing
/// Uses bit manipulation and polynomial approximation
#[inline]
fn fast_log(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }

    // Bit manipulation to extract exponent
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let mantissa = f32::from_bits((bits & 0x007FFFFF) | 0x3F800000);

    // Polynomial approximation for log(1 + x) where x is in [0, 1]
    let m = mantissa - 1.0;
    let log_mantissa = m * (1.0 - 0.5 * m + 0.33333333 * m * m);

    exp as f32 * std::f32::consts::LN_2 + log_mantissa
}

/// SIMD vector size for different architectures
#[cfg(target_arch = "x86_64")]
pub const SIMD_WIDTH_F32: usize = 16; // AVX-512: 16 f32 values (fallback to smaller if not available)

#[cfg(target_arch = "aarch64")]
pub const SIMD_WIDTH_F32: usize = 4; // NEON: 4 f32 values

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub const SIMD_WIDTH_F32: usize = 4; // Fallback

/// Advanced SIMD vector sizes
#[cfg(target_arch = "x86_64")]
pub const AVX512_WIDTH_F32: usize = 16; // AVX-512: 16 f32 values

#[cfg(target_arch = "x86_64")]
pub const AVX2_WIDTH_F32: usize = 8; // AVX2: 8 f32 values

#[cfg(target_arch = "x86_64")]
pub const SSE_WIDTH_F32: usize = 4; // SSE: 4 f32 values

#[cfg(target_arch = "aarch64")]
pub const NEON_WIDTH_F32: usize = 4; // NEON: 4 f32 values

/// SIMD acceleration capabilities
#[derive(Debug, Clone, Copy)]
pub struct SimdCapabilities {
    /// AVX2 support (x86_64)
    pub avx2: bool,
    /// AVX-512 support (x86_64)
    pub avx512: bool,
    /// FMA support (x86_64)
    pub fma: bool,
    /// NEON support (AArch64)
    pub neon: bool,
    /// SVE support (AArch64)
    pub sve: bool,
}

impl SimdCapabilities {
    /// Detect SIMD capabilities at runtime
    pub fn detect() -> Self {
        Self {
            #[cfg(target_arch = "x86_64")]
            avx2: is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            avx512: is_x86_feature_detected!("avx512f"),
            #[cfg(target_arch = "x86_64")]
            fma: is_x86_feature_detected!("fma"),
            #[cfg(not(target_arch = "x86_64"))]
            avx2: false,
            #[cfg(not(target_arch = "x86_64"))]
            avx512: false,
            #[cfg(not(target_arch = "x86_64"))]
            fma: false,

            #[cfg(target_arch = "aarch64")]
            neon: std::arch::is_aarch64_feature_detected!("neon"),
            #[cfg(not(target_arch = "aarch64"))]
            neon: false,

            // SVE detection is more complex, defaulting to false for now
            sve: false,
        }
    }

    /// Check if any SIMD acceleration is available
    pub fn has_simd(&self) -> bool {
        self.avx2 || self.avx512 || self.neon || self.sve
    }

    /// Get preferred vector width for f32 operations
    pub fn preferred_f32_width(&self) -> usize {
        if self.avx512 {
            16 // 16 f32 values
        } else if self.avx2 {
            8 // 8 f32 values
        } else if self.neon {
            4 // 4 f32 values
        } else {
            1 // Scalar fallback
        }
    }

    /// Get optimal chunk size for processing
    pub fn optimal_chunk_size(&self) -> usize {
        self.preferred_f32_width() * 64 // Optimize for cache line efficiency
    }

    /// Check if we have advanced features for specific operations
    pub fn has_advanced_features(&self) -> bool {
        self.avx512 || (self.avx2 && self.fma) || self.sve
    }
}

/// SIMD-accelerated operations dispatcher
pub struct SimdDispatcher {
    capabilities: SimdCapabilities,
}

impl SimdDispatcher {
    /// Create new SIMD dispatcher with runtime capability detection
    pub fn new() -> Self {
        Self {
            capabilities: SimdCapabilities::detect(),
        }
    }

    /// Get SIMD capabilities
    pub fn capabilities(&self) -> SimdCapabilities {
        self.capabilities
    }

    /// Vector addition with SIMD acceleration
    pub fn add_f32(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(AcousticError::InputError(
                "Vector lengths must match".to_string(),
            ));
        }

        #[cfg(target_arch = "x86_64")]
        if self.capabilities.avx512 {
            return unsafe { self.add_f32_avx512(a, b, result) };
        }

        #[cfg(target_arch = "x86_64")]
        if self.capabilities.avx2 {
            return unsafe { self.add_f32_avx2(a, b, result) };
        }

        #[cfg(target_arch = "aarch64")]
        if self.capabilities.neon {
            return unsafe { self.add_f32_neon(a, b, result) };
        }

        // Scalar fallback
        for ((a_val, b_val), result_val) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
            *result_val = a_val + b_val;
        }

        Ok(())
    }

    /// Vector multiplication with SIMD acceleration
    pub fn mul_f32(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(AcousticError::InputError(
                "Vector lengths must match".to_string(),
            ));
        }

        #[cfg(target_arch = "x86_64")]
        if self.capabilities.avx512 {
            return unsafe { self.mul_f32_avx512(a, b, result) };
        }

        #[cfg(target_arch = "x86_64")]
        if self.capabilities.avx2 {
            return unsafe { self.mul_f32_avx2(a, b, result) };
        }

        #[cfg(target_arch = "aarch64")]
        if self.capabilities.neon {
            return unsafe { self.mul_f32_neon(a, b, result) };
        }

        // Scalar fallback
        for ((a_val, b_val), result_val) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
            *result_val = a_val * b_val;
        }

        Ok(())
    }

    /// Fused multiply-add with SIMD acceleration
    pub fn fma_f32(&self, a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != c.len() || a.len() != result.len() {
            return Err(AcousticError::InputError(
                "Vector lengths must match".to_string(),
            ));
        }

        #[cfg(target_arch = "x86_64")]
        if self.capabilities.avx512 && self.capabilities.fma {
            return unsafe { self.fma_f32_avx512(a, b, c, result) };
        }

        #[cfg(target_arch = "x86_64")]
        if self.capabilities.fma {
            return unsafe { self.fma_f32_fma(a, b, c, result) };
        }

        // Fallback to separate multiply and add
        for (((a_val, b_val), c_val), result_val) in
            a.iter().zip(b.iter()).zip(c.iter()).zip(result.iter_mut())
        {
            *result_val = a_val.mul_add(*b_val, *c_val);
        }

        Ok(())
    }

    /// Dot product with SIMD acceleration
    pub fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(AcousticError::InputError(
                "Vector lengths must match".to_string(),
            ));
        }

        #[cfg(target_arch = "x86_64")]
        if self.capabilities.avx512 {
            return unsafe { self.dot_product_f32_avx512(a, b) };
        }

        #[cfg(target_arch = "x86_64")]
        if self.capabilities.avx2 {
            return unsafe { self.dot_product_f32_avx2(a, b) };
        }

        #[cfg(target_arch = "aarch64")]
        if self.capabilities.neon {
            return unsafe { self.dot_product_f32_neon(a, b) };
        }

        // Scalar fallback
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
    }

    /// SIMD-optimized exponential function (approximate, for audio processing)
    pub fn exp_f32(&self, input: &[f32], result: &mut [f32]) -> Result<()> {
        if input.len() != result.len() {
            return Err(AcousticError::InputError(
                "Vector lengths must match".to_string(),
            ));
        }

        #[cfg(target_arch = "x86_64")]
        if self.capabilities.avx512 {
            return unsafe { self.exp_f32_avx512(input, result) };
        }

        #[cfg(target_arch = "x86_64")]
        if self.capabilities.avx2 {
            return unsafe { self.exp_f32_avx2(input, result) };
        }

        // Scalar fallback using fast approximation
        for (inp, res) in input.iter().zip(result.iter_mut()) {
            *res = fast_exp(*inp);
        }

        Ok(())
    }

    /// SIMD-optimized logarithm function (approximate, for audio processing)
    pub fn log_f32(&self, input: &[f32], result: &mut [f32]) -> Result<()> {
        if input.len() != result.len() {
            return Err(AcousticError::InputError(
                "Vector lengths must match".to_string(),
            ));
        }

        #[cfg(target_arch = "x86_64")]
        if self.capabilities.avx512 {
            return unsafe { self.log_f32_avx512(input, result) };
        }

        #[cfg(target_arch = "x86_64")]
        if self.capabilities.avx2 {
            return unsafe { self.log_f32_avx2(input, result) };
        }

        // Scalar fallback using fast approximation
        for (inp, res) in input.iter().zip(result.iter_mut()) {
            *res = if *inp > 0.0 {
                fast_log(*inp)
            } else {
                f32::NEG_INFINITY
            };
        }

        Ok(())
    }

    /// SIMD-optimized magnitude calculation for complex numbers
    pub fn magnitude_f32(&self, real: &[f32], imag: &[f32], result: &mut [f32]) -> Result<()> {
        if real.len() != imag.len() || real.len() != result.len() {
            return Err(AcousticError::InputError(
                "Vector lengths must match".to_string(),
            ));
        }

        #[cfg(target_arch = "x86_64")]
        if self.capabilities.avx512 {
            return unsafe { self.magnitude_f32_avx512(real, imag, result) };
        }

        #[cfg(target_arch = "x86_64")]
        if self.capabilities.avx2 {
            return unsafe { self.magnitude_f32_avx2(real, imag, result) };
        }

        // Scalar fallback
        for ((r, i), res) in real.iter().zip(imag.iter()).zip(result.iter_mut()) {
            *res = (r * r + i * i).sqrt();
        }

        Ok(())
    }

    /// SIMD-optimized mel-frequency computation
    pub fn mel_scale_f32(&self, frequencies: &[f32], result: &mut [f32]) -> Result<()> {
        if frequencies.len() != result.len() {
            return Err(AcousticError::InputError(
                "Vector lengths must match".to_string(),
            ));
        }

        #[cfg(target_arch = "x86_64")]
        if self.capabilities.avx512 {
            return unsafe { self.mel_scale_f32_avx512(frequencies, result) };
        }

        #[cfg(target_arch = "x86_64")]
        if self.capabilities.avx2 {
            return unsafe { self.mel_scale_f32_avx2(frequencies, result) };
        }

        // Scalar fallback: mel = 2595 * log10(1 + freq / 700)
        for (freq, res) in frequencies.iter().zip(result.iter_mut()) {
            *res = 2595.0 * (1.0 + freq / 700.0).log10();
        }

        Ok(())
    }

    // Platform-specific implementations

    // AVX-512 implementations
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn add_f32_avx512(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        let len = a.len();
        let simd_len = len - (len % 16);

        // Process 16 elements at a time
        for i in (0..simd_len).step_by(16) {
            let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
            let result_vec = _mm512_add_ps(a_vec, b_vec);
            _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        }

        // Handle remaining elements
        for i in simd_len..len {
            result[i] = a[i] + b[i];
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn mul_f32_avx512(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        let len = a.len();
        let simd_len = len - (len % 16);

        for i in (0..simd_len).step_by(16) {
            let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
            let result_vec = _mm512_mul_ps(a_vec, b_vec);
            _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        }

        for i in simd_len..len {
            result[i] = a[i] * b[i];
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn fma_f32_avx512(
        &self,
        a: &[f32],
        b: &[f32],
        c: &[f32],
        result: &mut [f32],
    ) -> Result<()> {
        let len = a.len();
        let simd_len = len - (len % 16);

        for i in (0..simd_len).step_by(16) {
            let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
            let c_vec = _mm512_loadu_ps(c.as_ptr().add(i));
            let result_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
            _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        }

        for i in simd_len..len {
            result[i] = a[i].mul_add(b[i], c[i]);
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn dot_product_f32_avx512(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let len = a.len();
        let simd_len = len - (len % 16);

        let mut sum_vec = _mm512_setzero_ps();

        for i in (0..simd_len).step_by(16) {
            let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
            let mul_vec = _mm512_mul_ps(a_vec, b_vec);
            sum_vec = _mm512_add_ps(sum_vec, mul_vec);
        }

        // Horizontal sum of AVX-512 vector
        let mut result = _mm512_reduce_add_ps(sum_vec);

        // Add remaining elements
        for i in simd_len..len {
            result += a[i] * b[i];
        }

        Ok(result)
    }

    // AVX2 implementations
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_f32_avx2(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        let len = a.len();
        let simd_len = len - (len % 8);

        // Process 8 elements at a time
        for i in (0..simd_len).step_by(8) {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
            let result_vec = _mm256_add_ps(a_vec, b_vec);
            _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        }

        // Handle remaining elements
        for i in simd_len..len {
            result[i] = a[i] + b[i];
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_f32_avx2(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        let len = a.len();
        let simd_len = len - (len % 8);

        for i in (0..simd_len).step_by(8) {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
            let result_vec = _mm256_mul_ps(a_vec, b_vec);
            _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        }

        for i in simd_len..len {
            result[i] = a[i] * b[i];
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "fma")]
    unsafe fn fma_f32_fma(
        &self,
        a: &[f32],
        b: &[f32],
        c: &[f32],
        result: &mut [f32],
    ) -> Result<()> {
        let len = a.len();
        let simd_len = len - (len % 8);

        for i in (0..simd_len).step_by(8) {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
            let c_vec = _mm256_loadu_ps(c.as_ptr().add(i));
            let result_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
            _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        }

        for i in simd_len..len {
            result[i] = a[i].mul_add(b[i], c[i]);
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_product_f32_avx2(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let len = a.len();
        let simd_len = len - (len % 8);

        let mut sum_vec = _mm256_setzero_ps();

        for i in (0..simd_len).step_by(8) {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
            let mul_vec = _mm256_mul_ps(a_vec, b_vec);
            sum_vec = _mm256_add_ps(sum_vec, mul_vec);
        }

        // Horizontal sum of AVX2 vector
        let sum_high = _mm256_extractf128_ps(sum_vec, 1);
        let sum_low = _mm256_castps256_ps128(sum_vec);
        let sum_quad = _mm_add_ps(sum_low, sum_high);
        let sum_dual = _mm_add_ps(sum_quad, _mm_movehl_ps(sum_quad, sum_quad));
        let sum_single = _mm_add_ss(sum_dual, _mm_shuffle_ps(sum_dual, sum_dual, 1));

        let mut result = _mm_cvtss_f32(sum_single);

        // Add remaining elements
        for i in simd_len..len {
            result += a[i] * b[i];
        }

        Ok(result)
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn add_f32_neon(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        let len = a.len();
        let simd_len = len - (len % 4);

        for i in (0..simd_len).step_by(4) {
            let a_vec = vld1q_f32(a.as_ptr().add(i));
            let b_vec = vld1q_f32(b.as_ptr().add(i));
            let result_vec = vaddq_f32(a_vec, b_vec);
            vst1q_f32(result.as_mut_ptr().add(i), result_vec);
        }

        for i in simd_len..len {
            result[i] = a[i] + b[i];
        }

        Ok(())
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn mul_f32_neon(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        let len = a.len();
        let simd_len = len - (len % 4);

        for i in (0..simd_len).step_by(4) {
            let a_vec = vld1q_f32(a.as_ptr().add(i));
            let b_vec = vld1q_f32(b.as_ptr().add(i));
            let result_vec = vmulq_f32(a_vec, b_vec);
            vst1q_f32(result.as_mut_ptr().add(i), result_vec);
        }

        for i in simd_len..len {
            result[i] = a[i] * b[i];
        }

        Ok(())
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn dot_product_f32_neon(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let len = a.len();
        let simd_len = len - (len % 4);

        let mut sum_vec = vdupq_n_f32(0.0);

        for i in (0..simd_len).step_by(4) {
            let a_vec = vld1q_f32(a.as_ptr().add(i));
            let b_vec = vld1q_f32(b.as_ptr().add(i));
            let mul_vec = vmulq_f32(a_vec, b_vec);
            sum_vec = vaddq_f32(sum_vec, mul_vec);
        }

        // Horizontal sum of NEON vector
        let sum_pair = vadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
        let sum_single = vpadd_f32(sum_pair, sum_pair);
        let mut result = vget_lane_f32(sum_single, 0);

        // Add remaining elements
        for i in simd_len..len {
            result += a[i] * b[i];
        }

        Ok(result)
    }

    // AVX-512 implementations
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn add_f32_avx512(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        let len = a.len();
        let simd_len = len - (len % 16);

        for i in (0..simd_len).step_by(16) {
            let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
            let result_vec = _mm512_add_ps(a_vec, b_vec);
            _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        }

        for i in simd_len..len {
            result[i] = a[i] + b[i];
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn mul_f32_avx512(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        let len = a.len();
        let simd_len = len - (len % 16);

        for i in (0..simd_len).step_by(16) {
            let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
            let result_vec = _mm512_mul_ps(a_vec, b_vec);
            _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        }

        for i in simd_len..len {
            result[i] = a[i] * b[i];
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn fma_f32_avx512(
        &self,
        a: &[f32],
        b: &[f32],
        c: &[f32],
        result: &mut [f32],
    ) -> Result<()> {
        let len = a.len();
        let simd_len = len - (len % 16);

        for i in (0..simd_len).step_by(16) {
            let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
            let c_vec = _mm512_loadu_ps(c.as_ptr().add(i));
            let result_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
            _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        }

        for i in simd_len..len {
            result[i] = a[i].mul_add(b[i], c[i]);
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn dot_product_f32_avx512(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let len = a.len();
        let simd_len = len - (len % 16);

        let mut sum_vec = _mm512_setzero_ps();

        for i in (0..simd_len).step_by(16) {
            let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
            let mul_vec = _mm512_mul_ps(a_vec, b_vec);
            sum_vec = _mm512_add_ps(sum_vec, mul_vec);
        }

        let mut result = _mm512_reduce_add_ps(sum_vec);

        for i in simd_len..len {
            result += a[i] * b[i];
        }

        Ok(result)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn exp_f32_avx512(&self, input: &[f32], result: &mut [f32]) -> Result<()> {
        let len = input.len();
        let simd_len = len - (len % 16);

        let one = _mm512_set1_ps(1.0);
        let half = _mm512_set1_ps(0.5);
        let sixth = _mm512_set1_ps(0.16666667);
        let twentyfourth = _mm512_set1_ps(0.041666667);

        for i in (0..simd_len).step_by(16) {
            let x = _mm512_loadu_ps(input.as_ptr().add(i));
            let x_clamped = _mm512_max_ps(
                _mm512_min_ps(x, _mm512_set1_ps(10.0)),
                _mm512_set1_ps(-10.0),
            );

            let x2 = _mm512_mul_ps(x_clamped, x_clamped);
            let x3 = _mm512_mul_ps(x2, x_clamped);
            let x4 = _mm512_mul_ps(x3, x_clamped);

            let result_vec = _mm512_add_ps(
                _mm512_add_ps(one, x_clamped),
                _mm512_add_ps(
                    _mm512_mul_ps(x2, half),
                    _mm512_add_ps(_mm512_mul_ps(x3, sixth), _mm512_mul_ps(x4, twentyfourth)),
                ),
            );

            _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        }

        for i in simd_len..len {
            result[i] = fast_exp(input[i]);
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn log_f32_avx512(&self, input: &[f32], result: &mut [f32]) -> Result<()> {
        let len = input.len();
        let simd_len = len - (len % 16);

        for i in (0..simd_len).step_by(16) {
            let x = _mm512_loadu_ps(input.as_ptr().add(i));
            let mask = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_GT_OQ);
            let log_result = self.simd_log_approximation_avx512(x);
            let neg_inf = _mm512_set1_ps(f32::NEG_INFINITY);
            let result_vec = _mm512_mask_blend_ps(mask, neg_inf, log_result);
            _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        }

        for i in simd_len..len {
            result[i] = if input[i] > 0.0 {
                fast_log(input[i])
            } else {
                f32::NEG_INFINITY
            };
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn simd_log_approximation_avx512(&self, x: __m512) -> __m512 {
        let one = _mm512_set1_ps(1.0);
        let ln2 = _mm512_set1_ps(std::f32::consts::LN_2);

        let x_bits = _mm512_castps_si512(x);
        let exp_bits = _mm512_and_si512(_mm512_srli_epi32(x_bits, 23), _mm512_set1_epi32(0xFF));
        let exp = _mm512_cvtepi32_ps(_mm512_sub_epi32(exp_bits, _mm512_set1_epi32(127)));

        let mantissa_bits = _mm512_or_si512(
            _mm512_and_si512(x_bits, _mm512_set1_epi32(0x007FFFFF)),
            _mm512_set1_epi32(0x3F800000),
        );
        let mantissa = _mm512_castsi512_ps(mantissa_bits);

        let m = _mm512_sub_ps(mantissa, one);
        let m2 = _mm512_mul_ps(m, m);
        let log_mantissa =
            _mm512_mul_ps(m, _mm512_sub_ps(one, _mm512_mul_ps(_mm512_set1_ps(0.5), m)));
        let log_mantissa = _mm512_add_ps(
            log_mantissa,
            _mm512_mul_ps(_mm512_set1_ps(0.33333333), _mm512_mul_ps(m2, m)),
        );

        _mm512_add_ps(_mm512_mul_ps(exp, ln2), log_mantissa)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn magnitude_f32_avx512(
        &self,
        real: &[f32],
        imag: &[f32],
        result: &mut [f32],
    ) -> Result<()> {
        let len = real.len();
        let simd_len = len - (len % 16);

        for i in (0..simd_len).step_by(16) {
            let r = _mm512_loadu_ps(real.as_ptr().add(i));
            let i_vec = _mm512_loadu_ps(imag.as_ptr().add(i));
            let r2 = _mm512_mul_ps(r, r);
            let i2 = _mm512_mul_ps(i_vec, i_vec);
            let sum = _mm512_add_ps(r2, i2);
            let magnitude = _mm512_sqrt_ps(sum);
            _mm512_storeu_ps(result.as_mut_ptr().add(i), magnitude);
        }

        for i in simd_len..len {
            result[i] = (real[i] * real[i] + imag[i] * imag[i]).sqrt();
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn mel_scale_f32_avx512(&self, frequencies: &[f32], result: &mut [f32]) -> Result<()> {
        let len = frequencies.len();
        let simd_len = len - (len % 16);

        let factor = _mm512_set1_ps(2595.0);
        let divisor = _mm512_set1_ps(700.0);
        let one = _mm512_set1_ps(1.0);

        for i in (0..simd_len).step_by(16) {
            let freq = _mm512_loadu_ps(frequencies.as_ptr().add(i));
            let normalized = _mm512_add_ps(one, _mm512_div_ps(freq, divisor));
            let ln_val = self.simd_log_approximation_avx512(normalized);
            let log10_factor = _mm512_set1_ps(1.0 / std::f32::consts::LN_10);
            let log10_val = _mm512_mul_ps(ln_val, log10_factor);
            let mel = _mm512_mul_ps(factor, log10_val);
            _mm512_storeu_ps(result.as_mut_ptr().add(i), mel);
        }

        for i in simd_len..len {
            result[i] = 2595.0 * (1.0 + frequencies[i] / 700.0).log10();
        }

        Ok(())
    }

    // AVX2 implementations for advanced operations
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn exp_f32_avx2(&self, input: &[f32], result: &mut [f32]) -> Result<()> {
        let len = input.len();
        let simd_len = len - (len % 8);

        let one = _mm256_set1_ps(1.0);
        let half = _mm256_set1_ps(0.5);
        let sixth = _mm256_set1_ps(0.16666667);
        let twentyfourth = _mm256_set1_ps(0.041666667);

        for i in (0..simd_len).step_by(8) {
            let x = _mm256_loadu_ps(input.as_ptr().add(i));
            let x_clamped = _mm256_max_ps(
                _mm256_min_ps(x, _mm256_set1_ps(10.0)),
                _mm256_set1_ps(-10.0),
            );

            let x2 = _mm256_mul_ps(x_clamped, x_clamped);
            let x3 = _mm256_mul_ps(x2, x_clamped);
            let x4 = _mm256_mul_ps(x3, x_clamped);

            let result_vec = _mm256_add_ps(
                _mm256_add_ps(one, x_clamped),
                _mm256_add_ps(
                    _mm256_mul_ps(x2, half),
                    _mm256_add_ps(_mm256_mul_ps(x3, sixth), _mm256_mul_ps(x4, twentyfourth)),
                ),
            );

            _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        }

        for i in simd_len..len {
            result[i] = fast_exp(input[i]);
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn log_f32_avx2(&self, input: &[f32], result: &mut [f32]) -> Result<()> {
        let len = input.len();
        let simd_len = len - (len % 8);

        for i in (0..simd_len).step_by(8) {
            let x = _mm256_loadu_ps(input.as_ptr().add(i));

            // Check for positive values
            let zero = _mm256_setzero_ps();
            let mask = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);

            // Use log approximation for positive values
            let log_result = self.simd_log_approximation_avx2(x);

            // Set negative infinity for non-positive values
            let neg_inf = _mm256_set1_ps(f32::NEG_INFINITY);
            let result_vec = _mm256_blendv_ps(neg_inf, log_result, mask);

            _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        }

        for i in simd_len..len {
            result[i] = if input[i] > 0.0 {
                fast_log(input[i])
            } else {
                f32::NEG_INFINITY
            };
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn simd_log_approximation_avx2(&self, x: __m256) -> __m256 {
        let one = _mm256_set1_ps(1.0);
        let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);

        let x_bits = _mm256_castps_si256(x);
        let exp_bits = _mm256_and_si256(_mm256_srli_epi32(x_bits, 23), _mm256_set1_epi32(0xFF));
        let exp = _mm256_cvtepi32_ps(_mm256_sub_epi32(exp_bits, _mm256_set1_epi32(127)));

        let mantissa_bits = _mm256_or_si256(
            _mm256_and_si256(x_bits, _mm256_set1_epi32(0x007FFFFF)),
            _mm256_set1_epi32(0x3F800000),
        );
        let mantissa = _mm256_castsi256_ps(mantissa_bits);

        let m = _mm256_sub_ps(mantissa, one);
        let m2 = _mm256_mul_ps(m, m);
        let log_mantissa =
            _mm256_mul_ps(m, _mm256_sub_ps(one, _mm256_mul_ps(_mm256_set1_ps(0.5), m)));
        let log_mantissa = _mm256_add_ps(
            log_mantissa,
            _mm256_mul_ps(_mm256_set1_ps(0.33333333), _mm256_mul_ps(m2, m)),
        );

        _mm256_add_ps(_mm256_mul_ps(exp, ln2), log_mantissa)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn magnitude_f32_avx2(
        &self,
        real: &[f32],
        imag: &[f32],
        result: &mut [f32],
    ) -> Result<()> {
        let len = real.len();
        let simd_len = len - (len % 8);

        for i in (0..simd_len).step_by(8) {
            let r = _mm256_loadu_ps(real.as_ptr().add(i));
            let i_vec = _mm256_loadu_ps(imag.as_ptr().add(i));
            let r2 = _mm256_mul_ps(r, r);
            let i2 = _mm256_mul_ps(i_vec, i_vec);
            let sum = _mm256_add_ps(r2, i2);
            let magnitude = _mm256_sqrt_ps(sum);
            _mm256_storeu_ps(result.as_mut_ptr().add(i), magnitude);
        }

        for i in simd_len..len {
            result[i] = (real[i] * real[i] + imag[i] * imag[i]).sqrt();
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn mel_scale_f32_avx2(&self, frequencies: &[f32], result: &mut [f32]) -> Result<()> {
        let len = frequencies.len();
        let simd_len = len - (len % 8);

        let factor = _mm256_set1_ps(2595.0);
        let divisor = _mm256_set1_ps(700.0);
        let one = _mm256_set1_ps(1.0);

        for i in (0..simd_len).step_by(8) {
            let freq = _mm256_loadu_ps(frequencies.as_ptr().add(i));
            let normalized = _mm256_add_ps(one, _mm256_div_ps(freq, divisor));
            let ln_val = self.simd_log_approximation_avx2(normalized);
            let log10_factor = _mm256_set1_ps(1.0 / std::f32::consts::LN_10);
            let log10_val = _mm256_mul_ps(ln_val, log10_factor);
            let mel = _mm256_mul_ps(factor, log10_val);
            _mm256_storeu_ps(result.as_mut_ptr().add(i), mel);
        }

        for i in simd_len..len {
            result[i] = 2595.0 * (1.0 + frequencies[i] / 700.0).log10();
        }

        Ok(())
    }
}

impl Default for SimdDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Global SIMD dispatcher instance
static SIMD: std::sync::OnceLock<SimdDispatcher> = std::sync::OnceLock::new();

/// Get global SIMD dispatcher
pub fn simd() -> &'static SimdDispatcher {
    SIMD.get_or_init(SimdDispatcher::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_capabilities_detection() {
        let capabilities = SimdCapabilities::detect();
        // Just ensure it doesn't crash
        assert!(capabilities.preferred_f32_width() >= 1);
    }

    #[test]
    fn test_simd_dispatcher_creation() {
        let dispatcher = SimdDispatcher::new();
        let caps = dispatcher.capabilities();
        assert!(caps.preferred_f32_width() >= 1);
    }

    #[test]
    fn test_vector_addition() {
        let dispatcher = SimdDispatcher::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut result = vec![0.0; a.len()];

        dispatcher.add_f32(&a, &b, &mut result).unwrap();

        for (i, &val) in result.iter().enumerate() {
            assert_eq!(val, a[i] + b[i]);
        }
    }

    #[test]
    fn test_vector_multiplication() {
        let dispatcher = SimdDispatcher::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let mut result = vec![0.0; a.len()];

        dispatcher.mul_f32(&a, &b, &mut result).unwrap();

        for (i, &val) in result.iter().enumerate() {
            assert_eq!(val, a[i] * b[i]);
        }
    }

    #[test]
    fn test_dot_product() {
        let dispatcher = SimdDispatcher::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];

        let result = dispatcher.dot_product_f32(&a, &b).unwrap();
        assert_eq!(result, 10.0); // 1*1 + 2*1 + 3*1 + 4*1 = 10
    }

    #[test]
    fn test_fused_multiply_add() {
        let dispatcher = SimdDispatcher::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 2.0, 2.0, 2.0];
        let c = vec![1.0, 1.0, 1.0, 1.0];
        let mut result = vec![0.0; a.len()];

        dispatcher.fma_f32(&a, &b, &c, &mut result).unwrap();

        for (i, &val) in result.iter().enumerate() {
            assert_eq!(val, a[i] * b[i] + c[i]);
        }
    }

    #[test]
    fn test_global_simd_dispatcher() {
        let dispatcher1 = simd();
        let dispatcher2 = simd();

        // Should return the same instance
        assert!(std::ptr::eq(dispatcher1, dispatcher2));
    }

    #[test]
    fn test_simd_exp_approximation() {
        let dispatcher = SimdDispatcher::new();
        let input = vec![0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 10.0];
        let mut result = vec![0.0; input.len()];

        dispatcher.exp_f32(&input, &mut result).unwrap();

        // Test some known values with reasonable tolerance
        assert!((result[0] - 1.0).abs() < 0.1); // exp(0) ≈ 1
        assert!((result[1] - std::f32::consts::E).abs() < 0.2); // exp(1) ≈ e
        assert!((result[2] - (1.0 / std::f32::consts::E)).abs() < 0.1); // exp(-1) ≈ 1/e
    }

    #[test]
    fn test_simd_log_approximation() {
        let dispatcher = SimdDispatcher::new();
        let input = vec![1.0, std::f32::consts::E, 10.0, 0.5, 2.0, 0.1, -1.0, 0.0];
        let mut result = vec![0.0; input.len()];

        dispatcher.log_f32(&input, &mut result).unwrap();

        // Test some known values
        assert!(result[0].abs() < 0.1); // log(1) ≈ 0
        assert!((result[1] - 1.0).abs() < 0.2); // log(e) ≈ 1
        assert!((result[2] - 10.0_f32.ln()).abs() < 0.3); // log(10)
        assert_eq!(result[6], f32::NEG_INFINITY); // log(-1) = -∞
        assert_eq!(result[7], f32::NEG_INFINITY); // log(0) = -∞
    }

    #[test]
    fn test_simd_magnitude_calculation() {
        let dispatcher = SimdDispatcher::new();
        let real = vec![3.0, 0.0, 1.0, -4.0, 5.0, 0.0, 2.0, 1.0];
        let imag = vec![4.0, 1.0, 0.0, 3.0, 0.0, -2.0, 2.0, 1.0];
        let mut result = vec![0.0; real.len()];

        dispatcher.magnitude_f32(&real, &imag, &mut result).unwrap();

        // Test some known values
        assert!((result[0] - 5.0).abs() < 0.001); // sqrt(3² + 4²) = 5
        assert!((result[1] - 1.0).abs() < 0.001); // sqrt(0² + 1²) = 1
        assert!((result[2] - 1.0).abs() < 0.001); // sqrt(1² + 0²) = 1
        assert!((result[3] - 5.0).abs() < 0.001); // sqrt((-4)² + 3²) = 5
        assert!((result[6] - 2.828).abs() < 0.01); // sqrt(2² + 2²) ≈ 2.828
    }

    #[test]
    fn test_simd_mel_scale_conversion() {
        let dispatcher = SimdDispatcher::new();
        let frequencies = vec![0.0, 700.0, 1400.0, 2100.0, 4000.0, 8000.0];
        let mut result = vec![0.0; frequencies.len()];

        dispatcher.mel_scale_f32(&frequencies, &mut result).unwrap();

        // Test some known mel scale values using correct mel scale formula: mel = 2595 * log10(1 + freq / 700)
        assert!(result[0].abs() < 0.1); // mel(0) = 0
        assert!((result[1] - 781.17).abs() < 1.0); // mel(700) = 2595 * log10(2) ≈ 781.17
        assert!((result[2] - 1238.13).abs() < 1.0); // mel(1400) ≈ 1238.13

        // Verify mel scale is monotonically increasing
        for i in 1..result.len() {
            assert!(result[i] > result[i - 1]);
        }
    }

    #[test]
    fn test_advanced_simd_capabilities() {
        let capabilities = SimdCapabilities::detect();

        // Test capability detection functions
        assert!(capabilities.preferred_f32_width() >= 1);
        assert!(capabilities.optimal_chunk_size() >= capabilities.preferred_f32_width());

        // Test that capabilities are consistent
        if capabilities.avx512 {
            assert_eq!(capabilities.preferred_f32_width(), 16);
        } else if capabilities.avx2 {
            assert_eq!(capabilities.preferred_f32_width(), 8);
        } else if capabilities.neon {
            assert_eq!(capabilities.preferred_f32_width(), 4);
        }
    }

    #[test]
    fn test_simd_operations_consistency() {
        let dispatcher = SimdDispatcher::new();
        let size = 1000;

        // Generate test data
        let a: Vec<f32> = (0..size).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let b: Vec<f32> = (0..size).map(|i| (i as f32 + 1.0) * 0.02).collect();
        let c: Vec<f32> = (0..size).map(|i| (i as f32 + 1.0) * 0.03).collect();

        // Test vector operations
        let mut add_result = vec![0.0; size];
        let mut mul_result = vec![0.0; size];
        let mut fma_result = vec![0.0; size];

        dispatcher.add_f32(&a, &b, &mut add_result).unwrap();
        dispatcher.mul_f32(&a, &b, &mut mul_result).unwrap();
        dispatcher.fma_f32(&a, &b, &c, &mut fma_result).unwrap();

        // Verify results
        for i in 0..size {
            assert!((add_result[i] - (a[i] + b[i])).abs() < 1e-6);
            assert!((mul_result[i] - (a[i] * b[i])).abs() < 1e-6);
            // FMA operations may have slightly different precision than separate multiply-add
            // due to intermediate rounding differences and SIMD implementations, so use a more relaxed tolerance
            assert!((fma_result[i] - (a[i] * b[i] + c[i])).abs() < 1e-4);
        }

        // Test dot product
        let dot_result = dispatcher.dot_product_f32(&a, &b).unwrap();
        let expected_dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        // Dot product involves accumulation of many operations with different ordering in SIMD,
        // so use a tolerance that accounts for accumulated floating-point precision differences
        let dot_diff = (dot_result - expected_dot).abs();
        let relative_tolerance = expected_dot.abs() * 1e-5; // 0.001% relative tolerance
        let absolute_tolerance = 0.1; // Allow up to 0.1 absolute difference for large sums
        assert!(dot_diff < relative_tolerance.max(absolute_tolerance));
    }

    #[test]
    fn test_simd_performance_with_large_vectors() {
        let dispatcher = SimdDispatcher::new();
        let size = 10000;

        let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.0001).collect();
        let mut exp_result = vec![0.0; size];
        let mut log_result = vec![0.0; size];

        // Test that large vector operations complete successfully
        dispatcher.exp_f32(&input, &mut exp_result).unwrap();

        let positive_input: Vec<f32> = input.iter().map(|&x| x.abs() + 0.001).collect();
        dispatcher
            .log_f32(&positive_input, &mut log_result)
            .unwrap();

        // Verify no NaN or infinite values in normal range
        for &val in &exp_result {
            assert!(val.is_finite());
        }

        for &val in &log_result {
            assert!(val.is_finite());
        }
    }
}
