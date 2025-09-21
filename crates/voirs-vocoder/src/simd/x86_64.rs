//! x86_64 SIMD implementations using AVX2/AVX-512 instructions
//!
//! Provides vectorized implementations for audio processing operations
//! optimized for x86_64 architecture with AVX2 and AVX-512 support.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

/// AVX2 implementation of vector addition
#[target_feature(enable = "avx2")]
unsafe fn add_f32_avx2(a: &[f32], b: &[f32], output: &mut [f32]) {
    let len = a.len().min(b.len()).min(output.len());
    let chunks = len / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let result = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), result);
    }

    // Handle remaining elements
    for i in (chunks * 8)..len {
        output[i] = a[i] + b[i];
    }
}

/// AVX2 implementation of vector multiplication
#[target_feature(enable = "avx2")]
unsafe fn mul_f32_avx2(a: &[f32], b: &[f32], output: &mut [f32]) {
    let len = a.len().min(b.len()).min(output.len());
    let chunks = len / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let result = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), result);
    }

    // Handle remaining elements
    for i in (chunks * 8)..len {
        output[i] = a[i] * b[i];
    }
}

/// AVX2 implementation of scalar multiplication
#[target_feature(enable = "avx2")]
unsafe fn mul_scalar_f32_avx2(input: &[f32], scalar: f32, output: &mut [f32]) {
    let len = input.len().min(output.len());
    let chunks = len / 8;
    let scalar_vec = _mm256_set1_ps(scalar);

    for i in 0..chunks {
        let offset = i * 8;
        let vinput = _mm256_loadu_ps(input.as_ptr().add(offset));
        let result = _mm256_mul_ps(vinput, scalar_vec);
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), result);
    }

    // Handle remaining elements
    for i in (chunks * 8)..len {
        output[i] = input[i] * scalar;
    }
}

/// AVX2 implementation of fused multiply-add
#[target_feature(enable = "fma")]
unsafe fn fma_f32_avx2(a: &[f32], b: &[f32], c: &[f32], output: &mut [f32]) {
    let len = a.len().min(b.len()).min(c.len()).min(output.len());
    let chunks = len / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let vc = _mm256_loadu_ps(c.as_ptr().add(offset));
        let result = _mm256_fmadd_ps(va, vb, vc);
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), result);
    }

    // Handle remaining elements
    for i in (chunks * 8)..len {
        output[i] = a[i] * b[i] + c[i];
    }
}

/// AVX2 implementation of sum reduction
#[target_feature(enable = "avx2")]
unsafe fn sum_f32_avx2(input: &[f32]) -> f32 {
    let len = input.len();
    let chunks = len / 8;
    let mut sum_vec = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * 8;
        let vinput = _mm256_loadu_ps(input.as_ptr().add(offset));
        sum_vec = _mm256_add_ps(sum_vec, vinput);
    }

    // Horizontal sum of the vector
    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum_vec);
    let mut total = result.iter().sum::<f32>();

    // Add remaining elements
    for i in (chunks * 8)..len {
        total += input[i];
    }

    total
}

/// AVX2 implementation of dot product
#[target_feature(enable = "fma")]
unsafe fn dot_product_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let chunks = len / 8;
    let mut dot_vec = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        dot_vec = _mm256_fmadd_ps(va, vb, dot_vec);
    }

    // Horizontal sum of the dot product vector
    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), dot_vec);
    let mut total = result.iter().sum::<f32>();

    // Add remaining elements
    for i in (chunks * 8)..len {
        total += a[i] * b[i];
    }

    total
}

// Public interface functions that check for CPU feature availability

/// Vectorized addition with runtime feature detection
pub fn add_f32_x86_64(a: &[f32], b: &[f32], output: &mut [f32]) {
    if is_x86_feature_detected!("avx2") {
        unsafe { add_f32_avx2(a, b, output) }
    } else {
        super::generic::add_f32_scalar(a, b, output)
    }
}

/// Vectorized multiplication with runtime feature detection
pub fn mul_f32_x86_64(a: &[f32], b: &[f32], output: &mut [f32]) {
    if is_x86_feature_detected!("avx2") {
        unsafe { mul_f32_avx2(a, b, output) }
    } else {
        super::generic::mul_f32_scalar(a, b, output)
    }
}

/// Vectorized scalar multiplication with runtime feature detection
pub fn mul_scalar_f32_x86_64(input: &[f32], scalar: f32, output: &mut [f32]) {
    if is_x86_feature_detected!("avx2") {
        unsafe { mul_scalar_f32_avx2(input, scalar, output) }
    } else {
        super::generic::mul_scalar_f32_scalar(input, scalar, output)
    }
}

/// Vectorized fused multiply-add with runtime feature detection
pub fn fma_f32_x86_64(a: &[f32], b: &[f32], c: &[f32], output: &mut [f32]) {
    if is_x86_feature_detected!("fma") {
        unsafe { fma_f32_avx2(a, b, c, output) }
    } else {
        super::generic::fma_f32_scalar(a, b, c, output)
    }
}

/// Vectorized sum reduction with runtime feature detection
pub fn sum_f32_x86_64(input: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") {
        unsafe { sum_f32_avx2(input) }
    } else {
        super::generic::sum_f32_scalar(input)
    }
}

/// Vectorized dot product with runtime feature detection
pub fn dot_product_f32_x86_64(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("fma") {
        unsafe { dot_product_f32_avx2(a, b) }
    } else {
        super::generic::dot_product_f32_scalar(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_f32_x86_64() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 9];

        add_f32_x86_64(&a, &b, &mut output);

        let expected = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(output, expected);
    }

    #[test]
    fn test_mul_f32_x86_64() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let mut output = vec![0.0; 9];

        mul_f32_x86_64(&a, &b, &mut output);

        let expected = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
        assert_eq!(output, expected);
    }

    #[test]
    fn test_dot_product_f32_x86_64() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];

        let result = dot_product_f32_x86_64(&a, &b);

        assert_eq!(result, 10.0);
    }
}
