//! AArch64 SIMD implementations using NEON instructions
//!
//! Provides vectorized implementations for audio processing operations
//! optimized for ARM64 architecture with NEON support.

use std::arch::aarch64::*;

/// NEON implementation of vector addition
#[target_feature(enable = "neon")]
unsafe fn add_f32_neon(a: &[f32], b: &[f32], output: &mut [f32]) {
    let len = a.len().min(b.len()).min(output.len());
    let chunks = len / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.as_ptr().add(offset));
        let vb = vld1q_f32(b.as_ptr().add(offset));
        let result = vaddq_f32(va, vb);
        vst1q_f32(output.as_mut_ptr().add(offset), result);
    }

    // Handle remaining elements
    for i in (chunks * 4)..len {
        output[i] = a[i] + b[i];
    }
}

/// NEON implementation of vector multiplication
#[target_feature(enable = "neon")]
unsafe fn mul_f32_neon(a: &[f32], b: &[f32], output: &mut [f32]) {
    let len = a.len().min(b.len()).min(output.len());
    let chunks = len / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.as_ptr().add(offset));
        let vb = vld1q_f32(b.as_ptr().add(offset));
        let result = vmulq_f32(va, vb);
        vst1q_f32(output.as_mut_ptr().add(offset), result);
    }

    // Handle remaining elements
    for i in (chunks * 4)..len {
        output[i] = a[i] * b[i];
    }
}

/// NEON implementation of scalar multiplication
#[target_feature(enable = "neon")]
unsafe fn mul_scalar_f32_neon(input: &[f32], scalar: f32, output: &mut [f32]) {
    let len = input.len().min(output.len());
    let chunks = len / 4;
    let scalar_vec = vdupq_n_f32(scalar);

    for i in 0..chunks {
        let offset = i * 4;
        let vinput = vld1q_f32(input.as_ptr().add(offset));
        let result = vmulq_f32(vinput, scalar_vec);
        vst1q_f32(output.as_mut_ptr().add(offset), result);
    }

    // Handle remaining elements
    for i in (chunks * 4)..len {
        output[i] = input[i] * scalar;
    }
}

/// NEON implementation of fused multiply-add
#[target_feature(enable = "neon")]
unsafe fn fma_f32_neon(a: &[f32], b: &[f32], c: &[f32], output: &mut [f32]) {
    let len = a.len().min(b.len()).min(c.len()).min(output.len());
    let chunks = len / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.as_ptr().add(offset));
        let vb = vld1q_f32(b.as_ptr().add(offset));
        let vc = vld1q_f32(c.as_ptr().add(offset));
        let result = vfmaq_f32(vc, va, vb); // c + a * b
        vst1q_f32(output.as_mut_ptr().add(offset), result);
    }

    // Handle remaining elements
    for i in (chunks * 4)..len {
        output[i] = a[i] * b[i] + c[i];
    }
}

/// NEON implementation of sum reduction
#[target_feature(enable = "neon")]
unsafe fn sum_f32_neon(input: &[f32]) -> f32 {
    let len = input.len();
    let chunks = len / 4;
    let mut sum_vec = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * 4;
        let vinput = vld1q_f32(input.as_ptr().add(offset));
        sum_vec = vaddq_f32(sum_vec, vinput);
    }

    // Horizontal sum of the vector
    let sum_pair = vadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
    let mut total = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);

    // Add remaining elements
    for &element in input.iter().skip(chunks * 4).take(len - chunks * 4) {
        total += element;
    }

    total
}

/// NEON implementation of dot product
#[target_feature(enable = "neon")]
unsafe fn dot_product_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let chunks = len / 4;
    let mut dot_vec = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.as_ptr().add(offset));
        let vb = vld1q_f32(b.as_ptr().add(offset));
        dot_vec = vfmaq_f32(dot_vec, va, vb);
    }

    // Horizontal sum of the dot product vector
    let sum_pair = vadd_f32(vget_low_f32(dot_vec), vget_high_f32(dot_vec));
    let mut total = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);

    // Add remaining elements
    for i in (chunks * 4)..len {
        total += a[i] * b[i];
    }

    total
}

// Public interface functions that check for CPU feature availability

/// Vectorized addition with runtime feature detection
pub fn add_f32_aarch64(a: &[f32], b: &[f32], output: &mut [f32]) {
    // NEON is always available on AArch64
    unsafe { add_f32_neon(a, b, output) }
}

/// Vectorized multiplication with runtime feature detection
pub fn mul_f32_aarch64(a: &[f32], b: &[f32], output: &mut [f32]) {
    // NEON is always available on AArch64
    unsafe { mul_f32_neon(a, b, output) }
}

/// Vectorized scalar multiplication with runtime feature detection
pub fn mul_scalar_f32_aarch64(input: &[f32], scalar: f32, output: &mut [f32]) {
    // NEON is always available on AArch64
    unsafe { mul_scalar_f32_neon(input, scalar, output) }
}

/// Vectorized fused multiply-add with runtime feature detection
pub fn fma_f32_aarch64(a: &[f32], b: &[f32], c: &[f32], output: &mut [f32]) {
    // NEON is always available on AArch64
    unsafe { fma_f32_neon(a, b, c, output) }
}

/// Vectorized sum reduction with runtime feature detection
pub fn sum_f32_aarch64(input: &[f32]) -> f32 {
    // NEON is always available on AArch64
    unsafe { sum_f32_neon(input) }
}

/// Vectorized dot product with runtime feature detection
pub fn dot_product_f32_aarch64(a: &[f32], b: &[f32]) -> f32 {
    // NEON is always available on AArch64
    unsafe { dot_product_f32_neon(a, b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_f32_aarch64() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 5];

        add_f32_aarch64(&a, &b, &mut output);

        let expected = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(output, expected);
    }

    #[test]
    fn test_mul_f32_aarch64() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 2.0, 2.0, 2.0, 2.0];
        let mut output = vec![0.0; 5];

        mul_f32_aarch64(&a, &b, &mut output);

        let expected = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert_eq!(output, expected);
    }

    #[test]
    fn test_dot_product_f32_aarch64() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];

        let result = dot_product_f32_aarch64(&a, &b);

        assert_eq!(result, 10.0);
    }
}
