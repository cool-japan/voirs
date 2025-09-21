//! SIMD-accelerated matrix operations for neural networks
//!
//! This module provides SIMD-optimized matrix operations commonly used
//! in neural acoustic models, including matrix multiplication, convolution,
//! and activation functions.

use super::{simd, SIMD_WIDTH_F32};
use crate::{AcousticError, Result};

/// SIMD-accelerated matrix operations
pub struct SimdMatrix;

impl SimdMatrix {
    /// Matrix multiplication with SIMD acceleration: C = A * B
    /// A: [m, k], B: [k, n], C: [m, n]
    pub fn matmul_f32(a: &[Vec<f32>], b: &[Vec<f32>], result: &mut [Vec<f32>]) -> Result<()> {
        let m = a.len();
        let k = a[0].len();
        let n = b[0].len();

        if b.len() != k {
            return Err(AcousticError::InputError(
                "Matrix dimensions don't match for multiplication".to_string(),
            ));
        }

        if result.len() != m || result[0].len() != n {
            return Err(AcousticError::InputError(
                "Result matrix has incorrect dimensions".to_string(),
            ));
        }

        // Transpose B for better cache locality
        let b_transposed = Self::transpose(b);

        // Perform matrix multiplication with SIMD
        #[allow(clippy::needless_range_loop)]
        for i in 0..m {
            #[allow(clippy::needless_range_loop)]
            for j in 0..n {
                result[i][j] = simd().dot_product_f32(&a[i], &b_transposed[j])?;
            }
        }

        Ok(())
    }

    /// Batch matrix multiplication for multiple matrices
    pub fn batch_matmul_f32(
        a_batch: &[Vec<Vec<f32>>],
        b_batch: &[Vec<Vec<f32>>],
        result_batch: &mut [Vec<Vec<f32>>],
    ) -> Result<()> {
        if a_batch.len() != b_batch.len() || a_batch.len() != result_batch.len() {
            return Err(AcousticError::InputError(
                "Batch sizes must match".to_string(),
            ));
        }

        for (i, ((a, b), result)) in a_batch
            .iter()
            .zip(b_batch.iter())
            .zip(result_batch.iter_mut())
            .enumerate()
        {
            Self::matmul_f32(a, b, result)
                .map_err(|e| AcousticError::InputError(format!("Batch item {i}: {e}")))?;
        }

        Ok(())
    }

    /// Matrix-vector multiplication with SIMD acceleration: y = A * x
    pub fn matvec_f32(matrix: &[Vec<f32>], vector: &[f32], result: &mut [f32]) -> Result<()> {
        let m = matrix.len();
        let n = matrix[0].len();

        if vector.len() != n {
            return Err(AcousticError::InputError(
                "Vector length doesn't match matrix columns".to_string(),
            ));
        }

        if result.len() != m {
            return Err(AcousticError::InputError(
                "Result vector has incorrect length".to_string(),
            ));
        }

        for (i, row) in matrix.iter().enumerate() {
            result[i] = simd().dot_product_f32(row, vector)?;
        }

        Ok(())
    }

    /// 1D convolution with SIMD acceleration
    pub fn conv1d_f32(
        input: &[f32],
        kernel: &[f32],
        result: &mut [f32],
        stride: usize,
        padding: usize,
    ) -> Result<()> {
        let input_len = input.len();
        let kernel_len = kernel.len();
        let output_len = (input_len + 2 * padding - kernel_len) / stride + 1;

        if result.len() != output_len {
            return Err(AcousticError::InputError(
                "Result length doesn't match expected output length".to_string(),
            ));
        }

        // Create padded input
        let mut padded_input = vec![0.0f32; input_len + 2 * padding];
        padded_input[padding..padding + input_len].copy_from_slice(input);

        // Perform convolution
        #[allow(clippy::needless_range_loop)]
        for i in 0..output_len {
            let start_idx = i * stride;
            let end_idx = start_idx + kernel_len;

            if end_idx <= padded_input.len() {
                result[i] = simd().dot_product_f32(kernel, &padded_input[start_idx..end_idx])?;
            }
        }

        Ok(())
    }

    /// Batch 1D convolution for multiple channels
    pub fn batch_conv1d_f32(
        input_batch: &[Vec<f32>],
        kernels: &[Vec<f32>],
        result_batch: &mut [Vec<f32>],
        stride: usize,
        padding: usize,
    ) -> Result<()> {
        if input_batch.len() != kernels.len() || input_batch.len() != result_batch.len() {
            return Err(AcousticError::InputError(
                "Batch sizes must match".to_string(),
            ));
        }

        for ((input, kernel), result) in input_batch
            .iter()
            .zip(kernels.iter())
            .zip(result_batch.iter_mut())
        {
            Self::conv1d_f32(input, kernel, result, stride, padding)?;
        }

        Ok(())
    }

    /// Element-wise operations with SIMD acceleration
    pub fn elementwise_add_f32(
        a: &[Vec<f32>],
        b: &[Vec<f32>],
        result: &mut [Vec<f32>],
    ) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(AcousticError::InputError(
                "Matrix dimensions must match".to_string(),
            ));
        }

        for ((a_row, b_row), result_row) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
            simd().add_f32(a_row, b_row, result_row)?;
        }

        Ok(())
    }

    /// Element-wise multiplication with SIMD acceleration
    pub fn elementwise_mul_f32(
        a: &[Vec<f32>],
        b: &[Vec<f32>],
        result: &mut [Vec<f32>],
    ) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(AcousticError::InputError(
                "Matrix dimensions must match".to_string(),
            ));
        }

        for ((a_row, b_row), result_row) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
            simd().mul_f32(a_row, b_row, result_row)?;
        }

        Ok(())
    }

    /// Apply activation functions with SIMD acceleration
    pub fn apply_activation_f32(
        input: &[Vec<f32>],
        result: &mut [Vec<f32>],
        activation: ActivationFunction,
    ) -> Result<()> {
        if input.len() != result.len() {
            return Err(AcousticError::InputError(
                "Input and result dimensions must match".to_string(),
            ));
        }

        for (input_row, result_row) in input.iter().zip(result.iter_mut()) {
            Self::apply_activation_vector(input_row, result_row, activation)?;
        }

        Ok(())
    }

    /// Apply activation function to a vector with SIMD acceleration
    pub fn apply_activation_vector(
        input: &[f32],
        result: &mut [f32],
        activation: ActivationFunction,
    ) -> Result<()> {
        if input.len() != result.len() {
            return Err(AcousticError::InputError(
                "Input and result lengths must match".to_string(),
            ));
        }

        match activation {
            ActivationFunction::ReLU => {
                Self::relu_simd(input, result)?;
            }
            ActivationFunction::Sigmoid => {
                Self::sigmoid_simd(input, result)?;
            }
            ActivationFunction::Tanh => {
                Self::tanh_simd(input, result)?;
            }
            ActivationFunction::Gelu => {
                Self::gelu_simd(input, result)?;
            }
            ActivationFunction::Swish => {
                Self::swish_simd(input, result)?;
            }
        }

        Ok(())
    }

    /// Softmax activation with SIMD acceleration
    pub fn softmax_f32(input: &[f32], result: &mut [f32]) -> Result<()> {
        if input.len() != result.len() {
            return Err(AcousticError::InputError(
                "Input and result lengths must match".to_string(),
            ));
        }

        // Find maximum for numerical stability
        let max_val = input
            .iter()
            .fold(f32::NEG_INFINITY, |max, &val| max.max(val));

        // Compute exponentials
        let mut exp_sum = 0.0f32;
        for (i, &val) in input.iter().enumerate() {
            result[i] = (val - max_val).exp();
            exp_sum += result[i];
        }

        // Normalize
        if exp_sum > 0.0 {
            for val in result.iter_mut() {
                *val /= exp_sum;
            }
        }

        Ok(())
    }

    /// Layer normalization with SIMD acceleration
    pub fn layer_norm_f32(
        input: &[f32],
        result: &mut [f32],
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<()> {
        if input.len() != result.len() || input.len() != gamma.len() || input.len() != beta.len() {
            return Err(AcousticError::InputError(
                "All arrays must have the same length".to_string(),
            ));
        }

        // Compute mean
        let mean = input.iter().sum::<f32>() / input.len() as f32;

        // Compute variance
        let variance =
            input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / input.len() as f32;

        let std_dev = (variance + eps).sqrt();

        // Apply normalization
        for i in 0..input.len() {
            result[i] = gamma[i] * (input[i] - mean) / std_dev + beta[i];
        }

        Ok(())
    }

    /// Transpose matrix
    pub fn transpose(matrix: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if matrix.is_empty() || matrix[0].is_empty() {
            return Vec::new();
        }

        let rows = matrix.len();
        let cols = matrix[0].len();
        let mut transposed = vec![vec![0.0f32; rows]; cols];

        #[allow(clippy::needless_range_loop)]
        for i in 0..rows {
            #[allow(clippy::needless_range_loop)]
            for j in 0..cols {
                transposed[j][i] = matrix[i][j];
            }
        }

        transposed
    }

    /// Attention mechanism computation with SIMD acceleration
    pub fn attention_f32(
        query: &[Vec<f32>],
        key: &[Vec<f32>],
        value: &[Vec<f32>],
        result: &mut [Vec<f32>],
        scale: f32,
    ) -> Result<()> {
        let seq_len = query.len();
        let d_model = query[0].len();

        // Compute attention scores: Q * K^T
        let key_transposed = Self::transpose(key);
        let mut scores = vec![vec![0.0f32; seq_len]; seq_len];

        #[allow(clippy::needless_range_loop)]
        for i in 0..seq_len {
            #[allow(clippy::needless_range_loop)]
            for j in 0..seq_len {
                scores[i][j] = simd().dot_product_f32(&query[i], &key_transposed[j])? * scale;
            }
        }

        // Apply softmax to each row
        for row in scores.iter_mut() {
            let row_copy = row.clone();
            Self::softmax_f32(&row_copy, row)?;
        }

        // Compute output: Attention_weights * V
        #[allow(clippy::needless_range_loop)]
        for i in 0..seq_len {
            #[allow(clippy::needless_range_loop)]
            for j in 0..d_model {
                let mut sum = 0.0f32;
                #[allow(clippy::needless_range_loop)]
                for k in 0..seq_len {
                    sum += scores[i][k] * value[k][j];
                }
                result[i][j] = sum;
            }
        }

        Ok(())
    }

    // Private SIMD activation implementations

    fn relu_simd(input: &[f32], result: &mut [f32]) -> Result<()> {
        // Process in SIMD-friendly chunks
        for chunk in input
            .chunks(SIMD_WIDTH_F32)
            .zip(result.chunks_mut(SIMD_WIDTH_F32))
        {
            for (&input_val, result_val) in chunk.0.iter().zip(chunk.1.iter_mut()) {
                *result_val = input_val.max(0.0);
            }
        }
        Ok(())
    }

    fn sigmoid_simd(input: &[f32], result: &mut [f32]) -> Result<()> {
        for chunk in input
            .chunks(SIMD_WIDTH_F32)
            .zip(result.chunks_mut(SIMD_WIDTH_F32))
        {
            for (&input_val, result_val) in chunk.0.iter().zip(chunk.1.iter_mut()) {
                *result_val = 1.0 / (1.0 + (-input_val).exp());
            }
        }
        Ok(())
    }

    fn tanh_simd(input: &[f32], result: &mut [f32]) -> Result<()> {
        for chunk in input
            .chunks(SIMD_WIDTH_F32)
            .zip(result.chunks_mut(SIMD_WIDTH_F32))
        {
            for (&input_val, result_val) in chunk.0.iter().zip(chunk.1.iter_mut()) {
                *result_val = input_val.tanh();
            }
        }
        Ok(())
    }

    fn gelu_simd(input: &[f32], result: &mut [f32]) -> Result<()> {
        const SQRT_2_PI: f32 = 0.797_884_6; // sqrt(2/π)

        for chunk in input
            .chunks(SIMD_WIDTH_F32)
            .zip(result.chunks_mut(SIMD_WIDTH_F32))
        {
            for (&input_val, result_val) in chunk.0.iter().zip(chunk.1.iter_mut()) {
                let cdf =
                    0.5 * (1.0 + (SQRT_2_PI * (input_val + 0.044715 * input_val.powi(3))).tanh());
                *result_val = input_val * cdf;
            }
        }
        Ok(())
    }

    fn swish_simd(input: &[f32], result: &mut [f32]) -> Result<()> {
        for chunk in input
            .chunks(SIMD_WIDTH_F32)
            .zip(result.chunks_mut(SIMD_WIDTH_F32))
        {
            for (&input_val, result_val) in chunk.0.iter().zip(chunk.1.iter_mut()) {
                let sigmoid = 1.0 / (1.0 + (-input_val).exp());
                *result_val = input_val * sigmoid;
            }
        }
        Ok(())
    }
}

/// Activation function types
#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Gelu,
    Swish,
}

/// SIMD-optimized linear layer
pub struct SimdLinearLayer {
    /// Weight matrix [output_dim, input_dim]
    weight: Vec<Vec<f32>>,
    /// Bias vector [output_dim]
    bias: Option<Vec<f32>>,
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
}

impl SimdLinearLayer {
    /// Create new linear layer
    pub fn new(weight: Vec<Vec<f32>>, bias: Option<Vec<f32>>) -> Result<Self> {
        let output_dim = weight.len();
        let input_dim = if output_dim > 0 { weight[0].len() } else { 0 };

        if let Some(ref b) = bias {
            if b.len() != output_dim {
                return Err(AcousticError::ConfigError(
                    "Bias length must match output dimension".to_string(),
                ));
            }
        }

        Ok(Self {
            weight,
            bias,
            input_dim,
            output_dim,
        })
    }

    /// Forward pass with SIMD acceleration
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != self.input_dim {
            return Err(AcousticError::InputError(format!(
                "Expected input dimension {}, got {}",
                self.input_dim,
                input.len()
            )));
        }

        if output.len() != self.output_dim {
            return Err(AcousticError::InputError(format!(
                "Expected output dimension {}, got {}",
                self.output_dim,
                output.len()
            )));
        }

        // Compute matrix-vector multiplication
        SimdMatrix::matvec_f32(&self.weight, input, output)?;

        // Add bias if present
        if let Some(ref bias) = self.bias {
            for (out_val, &bias_val) in output.iter_mut().zip(bias.iter()) {
                *out_val += bias_val;
            }
        }

        Ok(())
    }

    /// Batch forward pass
    pub fn forward_batch(
        &self,
        input_batch: &[Vec<f32>],
        output_batch: &mut [Vec<f32>],
    ) -> Result<()> {
        if input_batch.len() != output_batch.len() {
            return Err(AcousticError::InputError(
                "Batch sizes must match".to_string(),
            ));
        }

        for (input, output) in input_batch.iter().zip(output_batch.iter_mut()) {
            self.forward(input, output)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_multiplication() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let b = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let mut result = vec![vec![0.0; 2]; 2];

        SimdMatrix::matmul_f32(&a, &b, &mut result).unwrap();

        // Expected: [[22, 28], [49, 64]]
        assert_eq!(result[0][0], 22.0);
        assert_eq!(result[0][1], 28.0);
        assert_eq!(result[1][0], 49.0);
        assert_eq!(result[1][1], 64.0);
    }

    #[test]
    fn test_matrix_vector_multiplication() {
        let matrix = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let vector = vec![1.0, 2.0, 3.0];
        let mut result = vec![0.0; 2];

        SimdMatrix::matvec_f32(&matrix, &vector, &mut result).unwrap();

        // Expected: [14, 32]
        assert_eq!(result[0], 14.0);
        assert_eq!(result[1], 32.0);
    }

    #[test]
    fn test_conv1d() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0, -1.0];
        let mut result = vec![0.0; 4];

        SimdMatrix::conv1d_f32(&input, &kernel, &mut result, 1, 0).unwrap();

        // Expected: [-1, -1, -1, -1] (convolution: x[n]*1 + x[n+1]*(-1) = x[n] - x[n+1])
        for &val in result.iter() {
            assert_eq!(val, -1.0);
        }
    }

    #[test]
    fn test_relu_activation() {
        let input = vec![vec![-1.0, 0.0, 1.0, 2.0]];
        let mut result = vec![vec![0.0; 4]];

        SimdMatrix::apply_activation_f32(&input, &mut result, ActivationFunction::ReLU).unwrap();

        assert_eq!(result[0], vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_softmax() {
        let input = vec![1.0, 2.0, 3.0];
        let mut result = vec![0.0; 3];

        SimdMatrix::softmax_f32(&input, &mut result).unwrap();

        // Check that probabilities sum to 1
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that largest input produces largest output
        let max_idx = result
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 2);
    }

    #[test]
    fn test_layer_norm() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut result = vec![0.0; 4];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];

        SimdMatrix::layer_norm_f32(&input, &mut result, &gamma, &beta, 1e-6).unwrap();

        // Check that normalized output has mean ≈ 0 and std ≈ 1
        let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
        assert!(mean.abs() < 1e-5);

        let variance: f32 =
            result.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / result.len() as f32;
        assert!((variance.sqrt() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_layer() {
        let weight = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let bias = Some(vec![0.5, -0.5]);
        let layer = SimdLinearLayer::new(weight, bias).unwrap();

        let input = vec![1.0, 2.0];
        let mut output = vec![0.0; 2];

        layer.forward(&input, &mut output).unwrap();

        // Expected: [1*1 + 2*2 + 0.5, 3*1 + 4*2 - 0.5] = [5.5, 10.5]
        assert_eq!(output[0], 5.5);
        assert_eq!(output[1], 10.5);
    }

    #[test]
    fn test_transpose() {
        let matrix = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let transposed = SimdMatrix::transpose(&matrix);

        assert_eq!(transposed.len(), 3);
        assert_eq!(transposed[0].len(), 2);
        assert_eq!(transposed[0], vec![1.0, 4.0]);
        assert_eq!(transposed[1], vec![2.0, 5.0]);
        assert_eq!(transposed[2], vec![3.0, 6.0]);
    }
}
