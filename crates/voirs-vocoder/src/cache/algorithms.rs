//! Cache-friendly algorithms for audio processing operations
//!
//! Provides cache-optimized implementations of common audio processing algorithms.

use super::{CacheConfig, CacheOptimizer};

/// Cache-friendly convolution implementation
pub struct CacheConvolution {
    cache_optimizer: CacheOptimizer,
}

impl CacheConvolution {
    /// Create new cache-optimized convolution
    pub fn new(config: CacheConfig) -> Self {
        Self {
            cache_optimizer: CacheOptimizer::new(config),
        }
    }

    /// Perform cache-friendly 1D convolution
    pub fn convolve_1d(&self, input: &[f32], kernel: &[f32]) -> Vec<f32> {
        if input.is_empty() || kernel.is_empty() {
            return Vec::new();
        }

        let output_len = input.len() + kernel.len() - 1;
        let mut output = vec![0.0f32; output_len];

        let block_size = self.cache_optimizer.optimal_block_size(input.len());

        // Process input in cache-friendly blocks
        for input_block_start in (0..input.len()).step_by(block_size) {
            let input_block_end = (input_block_start + block_size).min(input.len());
            let input_block = &input[input_block_start..input_block_end];

            // For each element in the current input block
            for (i, &input_val) in input_block.iter().enumerate() {
                let input_idx = input_block_start + i;

                // Prefetch next input if available
                self.cache_optimizer.prefetch_sequential(input, input_idx);

                // Apply kernel to current input sample
                for (k, &kernel_val) in kernel.iter().enumerate() {
                    let output_idx = input_idx + k;
                    if output_idx < output.len() {
                        output[output_idx] += input_val * kernel_val;
                    }
                }
            }
        }

        output
    }

    /// Perform cache-friendly 2D convolution
    pub fn convolve_2d(
        &self,
        input: &[f32],
        input_height: usize,
        input_width: usize,
        kernel: &[f32],
        kernel_height: usize,
        kernel_width: usize,
    ) -> Vec<f32> {
        if input.len() != input_height * input_width {
            return Vec::new();
        }
        if kernel.len() != kernel_height * kernel_width {
            return Vec::new();
        }

        let output_height = input_height + kernel_height - 1;
        let output_width = input_width + kernel_width - 1;
        let mut output = vec![0.0f32; output_height * output_width];

        let block_size = self.cache_optimizer.optimal_block_size(input.len());
        let block_height = (block_size as f32).sqrt() as usize;
        let block_width = block_height;

        // Process input in 2D blocks for better cache locality
        for block_r in (0..input_height).step_by(block_height) {
            for block_c in (0..input_width).step_by(block_width) {
                let end_r = (block_r + block_height).min(input_height);
                let end_c = (block_c + block_width).min(input_width);

                // Process current block
                for r in block_r..end_r {
                    for c in block_c..end_c {
                        let input_idx = r * input_width + c;
                        let input_val = input[input_idx];

                        // Prefetch next row if available
                        if r + 1 < input_height {
                            let next_row_idx = (r + 1) * input_width + c;
                            if next_row_idx < input.len() {
                                self.cache_optimizer.prefetch_strided(
                                    input,
                                    input_idx,
                                    input_width,
                                );
                            }
                        }

                        // Apply kernel
                        for kr in 0..kernel_height {
                            for kc in 0..kernel_width {
                                let kernel_idx = kr * kernel_width + kc;
                                let kernel_val = kernel[kernel_idx];

                                let output_r = r + kr;
                                let output_c = c + kc;
                                let output_idx = output_r * output_width + output_c;

                                if output_idx < output.len() {
                                    output[output_idx] += input_val * kernel_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        output
    }
}

impl Default for CacheConvolution {
    fn default() -> Self {
        Self::new(CacheConfig::default())
    }
}

/// Cache-friendly matrix multiplication
pub struct CacheMatMul {
    cache_optimizer: CacheOptimizer,
}

impl CacheMatMul {
    /// Create new cache-optimized matrix multiplication
    pub fn new(config: CacheConfig) -> Self {
        Self {
            cache_optimizer: CacheOptimizer::new(config),
        }
    }

    /// Perform cache-friendly matrix multiplication (C = A × B)
    pub fn multiply(
        &self,
        a: &[f32],
        a_rows: usize,
        a_cols: usize,
        b: &[f32],
        b_rows: usize,
        b_cols: usize,
    ) -> Result<Vec<f32>, String> {
        if a_cols != b_rows {
            return Err("Matrix dimensions don't match for multiplication".to_string());
        }
        if a.len() != a_rows * a_cols || b.len() != b_rows * b_cols {
            return Err("Input array sizes don't match specified dimensions".to_string());
        }

        let mut c = vec![0.0f32; a_rows * b_cols];
        let block_size = self.cache_optimizer.optimal_block_size(a.len() + b.len());

        // Use blocked matrix multiplication for better cache performance
        for i_block in (0..a_rows).step_by(block_size) {
            for j_block in (0..b_cols).step_by(block_size) {
                for k_block in (0..a_cols).step_by(block_size) {
                    let i_end = (i_block + block_size).min(a_rows);
                    let j_end = (j_block + block_size).min(b_cols);
                    let k_end = (k_block + block_size).min(a_cols);

                    // Process current block
                    for i in i_block..i_end {
                        for j in j_block..j_end {
                            let mut sum = 0.0f32;

                            for k in k_block..k_end {
                                let a_idx = i * a_cols + k;
                                let b_idx = k * b_cols + j;

                                // Prefetch next elements
                                if k + 1 < a_cols {
                                    self.cache_optimizer.prefetch_sequential(a, a_idx);
                                    self.cache_optimizer.prefetch_strided(b, b_idx, b_cols);
                                }

                                sum += a[a_idx] * b[b_idx];
                            }

                            let c_idx = i * b_cols + j;
                            c[c_idx] += sum;
                        }
                    }
                }
            }
        }

        Ok(c)
    }
}

impl Default for CacheMatMul {
    fn default() -> Self {
        Self::new(CacheConfig::default())
    }
}

/// Cache-friendly FFT implementation
pub struct CacheFFT {
    cache_optimizer: CacheOptimizer,
}

impl CacheFFT {
    /// Create new cache-optimized FFT
    pub fn new(config: CacheConfig) -> Self {
        Self {
            cache_optimizer: CacheOptimizer::new(config),
        }
    }

    /// Perform cache-friendly complex number multiplication
    pub fn complex_multiply(
        &self,
        a_real: &[f32],
        a_imag: &[f32],
        b_real: &[f32],
        b_imag: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        if a_real.len() != a_imag.len()
            || b_real.len() != b_imag.len()
            || a_real.len() != b_real.len()
        {
            return Err("Array lengths must match".to_string());
        }

        let len = a_real.len();
        let mut c_real = Vec::with_capacity(len);
        let mut c_imag = Vec::with_capacity(len);

        let block_size = self.cache_optimizer.optimal_block_size(len);

        // Process in cache-friendly blocks
        for block_start in (0..len).step_by(block_size) {
            let block_end = (block_start + block_size).min(len);

            for i in block_start..block_end {
                // Prefetch next elements
                self.cache_optimizer.prefetch_sequential(a_real, i);
                self.cache_optimizer.prefetch_sequential(a_imag, i);
                self.cache_optimizer.prefetch_sequential(b_real, i);
                self.cache_optimizer.prefetch_sequential(b_imag, i);

                // Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
                let real_part = a_real[i] * b_real[i] - a_imag[i] * b_imag[i];
                let imag_part = a_real[i] * b_imag[i] + a_imag[i] * b_real[i];

                c_real.push(real_part);
                c_imag.push(imag_part);
            }
        }

        Ok((c_real, c_imag))
    }

    /// Bit-reverse permutation with cache optimization
    pub fn bit_reverse_permutation(&self, data: &mut [f32]) {
        let n = data.len();
        if !n.is_power_of_two() {
            return;
        }

        let log_n = n.trailing_zeros() as usize;
        let block_size = self.cache_optimizer.optimal_block_size(n);

        // Process in blocks to improve cache locality
        for block_start in (0..n).step_by(block_size) {
            let block_end = (block_start + block_size).min(n);

            for i in block_start..block_end {
                let j = self.bit_reverse(i, log_n);

                if i < j {
                    // Prefetch swap locations
                    if i + 1 < data.len() {
                        self.cache_optimizer.prefetch_sequential(data, i);
                    }
                    if j + 1 < data.len() {
                        self.cache_optimizer.prefetch_sequential(data, j);
                    }

                    data.swap(i, j);
                }
            }
        }
    }

    /// Reverse bits of a number
    fn bit_reverse(&self, mut n: usize, bits: usize) -> usize {
        let mut reversed = 0;
        for _ in 0..bits {
            reversed = (reversed << 1) | (n & 1);
            n >>= 1;
        }
        reversed
    }
}

impl Default for CacheFFT {
    fn default() -> Self {
        Self::new(CacheConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_convolution_1d() {
        let conv = CacheConvolution::default();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let kernel = vec![0.5, 0.5];

        let result = conv.convolve_1d(&input, &kernel);
        assert_eq!(result.len(), input.len() + kernel.len() - 1);

        // First element should be 1.0 * 0.5 = 0.5
        assert_eq!(result[0], 0.5);
        // Second element should be 1.0 * 0.5 + 2.0 * 0.5 = 1.5
        assert_eq!(result[1], 1.5);
    }

    #[test]
    fn test_cache_convolution_2d() {
        let conv = CacheConvolution::default();
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let kernel = vec![1.0]; // 1x1 kernel

        let result = conv.convolve_2d(&input, 2, 2, &kernel, 1, 1);
        assert_eq!(result.len(), 2 * 2); // Same size for 1x1 kernel
        assert_eq!(result, input); // Should be identical for 1x1 kernel with value 1.0
    }

    #[test]
    fn test_cache_matmul() {
        let matmul = CacheMatMul::default();

        // 2x2 × 2x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = matmul.multiply(&a, 2, 2, &b, 2, 2).unwrap();
        assert_eq!(result.len(), 4);

        // [1 2] × [5 6] = [19 22]
        // [3 4]   [7 8]   [43 50]
        assert_eq!(result[0], 19.0); // 1*5 + 2*7
        assert_eq!(result[1], 22.0); // 1*6 + 2*8
        assert_eq!(result[2], 43.0); // 3*5 + 4*7
        assert_eq!(result[3], 50.0); // 3*6 + 4*8
    }

    #[test]
    fn test_cache_fft_complex_multiply() {
        let fft = CacheFFT::default();

        let a_real = vec![1.0, 2.0];
        let a_imag = vec![0.0, 1.0];
        let b_real = vec![1.0, 0.0];
        let b_imag = vec![0.0, 1.0];

        let (c_real, c_imag) = fft
            .complex_multiply(&a_real, &a_imag, &b_real, &b_imag)
            .unwrap();

        assert_eq!(c_real.len(), 2);
        assert_eq!(c_imag.len(), 2);

        // (1 + 0i) * (1 + 0i) = 1 + 0i
        assert_eq!(c_real[0], 1.0);
        assert_eq!(c_imag[0], 0.0);

        // (2 + 1i) * (0 + 1i) = -1 + 2i
        assert_eq!(c_real[1], -1.0);
        assert_eq!(c_imag[1], 2.0);
    }

    #[test]
    fn test_bit_reverse_permutation() {
        let fft = CacheFFT::default();
        let mut data = vec![0.0, 1.0, 2.0, 3.0]; // length 4 = 2^2

        fft.bit_reverse_permutation(&mut data);

        // For n=4, bit reversal pattern:
        // 0 (00) -> 0 (00) = 0
        // 1 (01) -> 2 (10) = 2
        // 2 (10) -> 1 (01) = 1
        // 3 (11) -> 3 (11) = 3
        // So [0,1,2,3] becomes [0,2,1,3]
        assert_eq!(data, vec![0.0, 2.0, 1.0, 3.0]);
    }

    #[test]
    fn test_empty_inputs() {
        let conv = CacheConvolution::default();
        let matmul = CacheMatMul::default();

        assert!(conv.convolve_1d(&[], &[1.0]).is_empty());
        assert!(conv.convolve_1d(&[1.0], &[]).is_empty());

        assert!(matmul.multiply(&[], 0, 0, &[], 0, 0).unwrap().is_empty());
    }
}
