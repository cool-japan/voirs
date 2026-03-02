//! High-performance SIMD operations for speaker embeddings
//!
//! This module provides highly optimized SIMD-accelerated operations for speaker
//! embeddings using scirs2-core's SIMD abstractions. These operations are critical
//! for performance in real-time voice cloning scenarios.
//!
//! # SIMD Optimization Strategy
//!
//! This module leverages scirs2-core's `SimdUnifiedOps` trait which provides
//! hardware-accelerated SIMD operations that automatically detect and use the best
//! available instruction set (AVX2, AVX512, NEON, etc.) at runtime.
//!
//! For embeddings (typically 256-512 dimensions), SIMD can provide 4-8x speedup
//! over scalar operations, which is critical for real-time voice cloning where
//! thousands of similarity computations may be needed per second.

use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Compute cosine similarity between two embeddings using SIMD operations
///
/// # Arguments
/// * `a` - First embedding vector
/// * `b` - Second embedding vector
///
/// # Returns
/// Cosine similarity score in range [-1, 1], or 0.0 if dimensions don't match
///
/// # Performance
/// This function uses scirs2-core's SIMD operations which provide hardware-accelerated
/// computation with automatic platform detection. Typical speedup is 4-8x over scalar code
/// on modern processors (AVX2, AVX512, NEON).
///
/// # Examples
///
/// ```
/// use voirs_cloning::embedding::simd_ops::simd_cosine_similarity;
///
/// let emb1 = vec![1.0, 0.0, 0.0];
/// let emb2 = vec![1.0, 0.0, 0.0];
/// let similarity = simd_cosine_similarity(&emb1, &emb2);
/// assert!((similarity - 1.0).abs() < 1e-6);
/// ```
pub fn simd_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    // Convert slices to ArrayView1 for SIMD operations
    let arr_a = ArrayView1::from(a);
    let arr_b = ArrayView1::from(b);

    // Use scirs2-core's SIMD dot product
    let dot_product = f32::simd_dot(&arr_a, &arr_b);

    // Use scirs2-core's SIMD norm computation
    let norm_a = f32::simd_norm(&arr_a);
    let norm_b = f32::simd_norm(&arr_b);

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

/// Compute Euclidean distance between two embeddings using SIMD operations
///
/// # Arguments
/// * `a` - First embedding vector
/// * `b` - Second embedding vector
///
/// # Returns
/// Euclidean distance, or f32::INFINITY if dimensions don't match
///
/// # Performance
/// This function uses scirs2-core's SIMD operations for hardware-accelerated computation.
/// Typical speedup is 4-8x over scalar code.
///
/// # Examples
///
/// ```
/// use voirs_cloning::embedding::simd_ops::simd_euclidean_distance;
///
/// let a = vec![0.0, 0.0, 0.0];
/// let b = vec![3.0, 4.0, 0.0];
/// let dist = simd_euclidean_distance(&a, &b);
/// assert!((dist - 5.0).abs() < 1e-6);
/// ```
pub fn simd_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return f32::INFINITY;
    }

    // Convert slices to ArrayView1 for SIMD operations
    let arr_a = ArrayView1::from(a);
    let arr_b = ArrayView1::from(b);

    // Compute difference using SIMD subtraction
    let diff = f32::simd_sub(&arr_a, &arr_b);

    // Compute norm of difference vector using SIMD
    f32::simd_norm(&diff.view())
}

/// L2 normalize a vector in-place using SIMD operations
///
/// # Arguments
/// * `vec` - Vector to normalize (modified in-place)
///
/// # Performance
/// This function uses scirs2-core's SIMD operations for hardware-accelerated computation.
///
/// # Examples
///
/// ```
/// use voirs_cloning::embedding::simd_ops::simd_normalize_inplace;
///
/// let mut vec = vec![3.0, 4.0, 0.0];
/// simd_normalize_inplace(&mut vec);
/// let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
/// assert!((norm - 1.0).abs() < 1e-6);
/// ```
pub fn simd_normalize_inplace(vec: &mut [f32]) {
    if vec.is_empty() {
        return;
    }

    // Convert to ArrayView1 for SIMD norm computation
    let arr_view = ArrayView1::from(&*vec);
    let norm = f32::simd_norm(&arr_view);

    if norm > 0.0 {
        let inv_norm = 1.0 / norm;

        // Scalar multiply in-place using SIMD
        let arr = Array1::from_vec(vec.to_vec());
        let normalized = f32::simd_scalar_mul(&arr.view(), inv_norm);

        // Copy results back to original vec
        vec.copy_from_slice(normalized.as_slice().expect("Array1 should be contiguous"));
    }
}

/// Weighted average of multiple embeddings using SIMD operations
///
/// # Arguments
/// * `embeddings` - Slice of embedding vectors (all must have same dimension)
/// * `weights` - Weights for each embedding (should sum to 1.0)
///
/// # Returns
/// Weighted average embedding vector, or empty vec if inputs invalid
///
/// # Performance
/// This function uses scirs2-core's SIMD operations for efficient batch processing.
///
/// # Examples
///
/// ```
/// use voirs_cloning::embedding::simd_ops::simd_weighted_average;
///
/// let emb1 = vec![1.0, 0.0];
/// let emb2 = vec![0.0, 1.0];
/// let embeddings = vec![emb1, emb2];
/// let weights = vec![0.5, 0.5];
/// let result = simd_weighted_average(&embeddings, &weights);
/// assert_eq!(result, vec![0.5, 0.5]);
/// ```
pub fn simd_weighted_average(embeddings: &[Vec<f32>], weights: &[f32]) -> Vec<f32> {
    if embeddings.is_empty() || weights.is_empty() || embeddings.len() != weights.len() {
        return Vec::new();
    }

    let dim = embeddings[0].len();

    // Verify all embeddings have same dimension
    if !embeddings.iter().all(|e| e.len() == dim) {
        return Vec::new();
    }

    // Initialize result array using scirs2-core's Array1
    let mut result = Array1::zeros(dim);

    // Compute weighted sum using SIMD operations
    for (embedding, &weight) in embeddings.iter().zip(weights.iter()) {
        let arr = Array1::from_vec(embedding.clone());
        let weighted = f32::simd_scalar_mul(&arr.view(), weight);

        // Add to result using SIMD
        result = f32::simd_add(&result.view(), &weighted.view());
    }

    result.to_vec()
}

/// Compute dot product between two vectors using SIMD operations
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Dot product, or 0.0 if dimensions don't match
///
/// # Performance
/// This function uses scirs2-core's SIMD dot product for hardware acceleration.
///
/// # Examples
///
/// ```
/// use voirs_cloning::embedding::simd_ops::simd_dot_product;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// let dot = simd_dot_product(&a, &b);
/// assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 32
/// ```
pub fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    // Convert to ArrayView1 and use SIMD dot product
    let arr_a = ArrayView1::from(a);
    let arr_b = ArrayView1::from(b);

    f32::simd_dot(&arr_a, &arr_b)
}

/// Compute L2 norm of a vector using SIMD operations
///
/// # Arguments
/// * `vec` - Input vector
///
/// # Returns
/// L2 norm of the vector
///
/// # Performance
/// This function uses scirs2-core's SIMD norm computation for hardware acceleration.
///
/// # Examples
///
/// ```
/// use voirs_cloning::embedding::simd_ops::simd_l2_norm;
///
/// let vec = vec![3.0, 4.0];
/// let norm = simd_l2_norm(&vec);
/// assert!((norm - 5.0).abs() < 1e-6);
/// ```
pub fn simd_l2_norm(vec: &[f32]) -> f32 {
    if vec.is_empty() {
        return 0.0;
    }

    // Convert to ArrayView1 and use SIMD norm
    let arr = ArrayView1::from(vec);
    f32::simd_norm(&arr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = simd_cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);

        let c = vec![1.0, 0.0, 0.0];
        let d = vec![0.0, 1.0, 0.0];
        let sim = simd_cosine_similarity(&c, &d);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_simd_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        let dist = simd_euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_normalize() {
        let mut vec = vec![3.0, 4.0, 0.0];
        simd_normalize_inplace(&mut vec);
        let norm = simd_l2_norm(&vec);
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_weighted_average() {
        let emb1 = vec![1.0, 0.0];
        let emb2 = vec![0.0, 1.0];
        let embeddings = vec![emb1, emb2];
        let weights = vec![0.5, 0.5];
        let result = simd_weighted_average(&embeddings, &weights);
        assert_eq!(result, vec![0.5, 0.5]);
    }

    #[test]
    fn test_simd_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dot = simd_dot_product(&a, &b);
        assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }

    #[test]
    fn test_simd_l2_norm() {
        let vec = vec![3.0, 4.0];
        let norm = simd_l2_norm(&vec);
        assert!((norm - 5.0).abs() < 1e-6);
    }
}
