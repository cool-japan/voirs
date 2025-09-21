//! SIMD-optimized convolution operations for neural network layers
//!
//! Provides vectorized implementations of convolution operations commonly
//! used in neural vocoder architectures like HiFi-GAN and DiffWave.

/// 1D convolution with SIMD optimization
pub fn conv1d_f32(
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
    stride: usize,
    padding: usize,
) {
    let input_len = input.len();
    let kernel_len = kernel.len();
    let output_len = output.len();

    if kernel_len == 0 || input_len == 0 {
        return;
    }

    // Use input directly if no padding needed, otherwise create padded version
    let (padded_input, padded_len) = if padding > 0 {
        let mut padded = vec![0.0; input_len + 2 * padding];
        padded[padding..padding + input_len].copy_from_slice(input);
        let len = padded.len();
        (Some(padded), len)
    } else {
        (None, input_len)
    };

    // Get reference to the input data to use
    let input_data = if let Some(ref padded) = padded_input {
        padded.as_slice()
    } else {
        input
    };

    #[allow(clippy::needless_range_loop)]
    for i in 0..output_len {
        let start = i * stride;
        if start + kernel_len <= padded_len {
            // Use SIMD optimized dot product for convolution
            let sum = {
                #[cfg(target_arch = "x86_64")]
                {
                    if kernel_len >= 8 && is_x86_feature_detected!("fma") {
                        super::x86_64::dot_product_f32_x86_64(
                            &input_data[start..start + kernel_len],
                            kernel,
                        )
                    } else {
                        super::generic::dot_product_f32_scalar(
                            &input_data[start..start + kernel_len],
                            kernel,
                        )
                    }
                }

                #[cfg(target_arch = "aarch64")]
                {
                    if kernel_len >= 4 {
                        super::aarch64::dot_product_f32_aarch64(
                            &input_data[start..start + kernel_len],
                            kernel,
                        )
                    } else {
                        super::generic::dot_product_f32_scalar(
                            &input_data[start..start + kernel_len],
                            kernel,
                        )
                    }
                }

                #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                {
                    super::generic::dot_product_f32_scalar(
                        &input_data[start..start + kernel_len],
                        kernel,
                    )
                }
            };

            output[i] = sum;
        }
    }
}

/// Depthwise separable convolution (used in mobile-optimized architectures)
pub fn depthwise_conv1d_f32(
    input: &[f32],
    kernels: &[&[f32]], // One kernel per channel
    output: &mut [f32],
    stride: usize,
    padding: usize,
) {
    let channels = kernels.len();
    if channels == 0 || input.is_empty() {
        return;
    }

    let input_per_channel = input.len() / channels;
    let output_per_channel = output.len() / channels;

    for ch in 0..channels {
        let ch_input = &input[ch * input_per_channel..(ch + 1) * input_per_channel];
        let ch_output = &mut output[ch * output_per_channel..(ch + 1) * output_per_channel];

        conv1d_f32(ch_input, kernels[ch], ch_output, stride, padding);
    }
}

/// Transposed convolution (deconvolution) with SIMD optimization
pub fn transpose_conv1d_f32(
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
    stride: usize,
    padding: usize,
) {
    let input_len = input.len();
    let kernel_len = kernel.len();
    let output_len = output.len();

    if kernel_len == 0 || input_len == 0 {
        return;
    }

    // Initialize output to zero
    output.fill(0.0);

    #[allow(clippy::needless_range_loop)]
    for i in 0..input_len {
        let output_start = i * stride;

        if output_start >= padding && output_start - padding + kernel_len <= output_len {
            let out_pos = output_start - padding;

            // Use SIMD optimized scalar multiplication and addition
            #[cfg(target_arch = "x86_64")]
            {
                if kernel_len >= 8 && is_x86_feature_detected!("avx2") {
                    let mut temp = vec![0.0; kernel_len];
                    super::x86_64::mul_scalar_f32_x86_64(kernel, input[i], &mut temp);
                    super::x86_64::add_f32_x86_64(
                        &output[out_pos..out_pos + kernel_len],
                        &temp,
                        &mut output[out_pos..out_pos + kernel_len],
                    );
                } else {
                    for (j, &k) in kernel.iter().enumerate() {
                        if out_pos + j < output_len {
                            output[out_pos + j] += input[i] * k;
                        }
                    }
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                if kernel_len >= 4 {
                    let mut temp = vec![0.0; kernel_len];
                    super::aarch64::mul_scalar_f32_aarch64(kernel, input[i], &mut temp);
                    let slice = &mut output[out_pos..out_pos + kernel_len];
                    let original_slice = slice.to_vec();
                    super::aarch64::add_f32_aarch64(&original_slice, &temp, slice);
                } else {
                    for (j, &k) in kernel.iter().enumerate() {
                        if out_pos + j < output_len {
                            output[out_pos + j] += input[i] * k;
                        }
                    }
                }
            }

            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                for (j, &k) in kernel.iter().enumerate() {
                    if out_pos + j < output_len {
                        output[out_pos + j] += input[i] * k;
                    }
                }
            }
        }
    }
}

/// Multi-channel convolution with batch processing
#[allow(clippy::too_many_arguments)]
pub fn conv1d_multi_channel_f32(
    input: &[f32],      // [batch_size, in_channels, length]
    kernels: &[f32],    // [out_channels, in_channels, kernel_size]
    output: &mut [f32], // [batch_size, out_channels, output_length]
    batch_size: usize,
    in_channels: usize,
    out_channels: usize,
    input_length: usize,
    output_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) {
    for batch in 0..batch_size {
        for out_ch in 0..out_channels {
            let output_offset = batch * out_channels * output_length + out_ch * output_length;
            let out_slice = &mut output[output_offset..output_offset + output_length];

            // Initialize output channel to zero
            out_slice.fill(0.0);

            for in_ch in 0..in_channels {
                let input_offset = batch * in_channels * input_length + in_ch * input_length;
                let input_slice = &input[input_offset..input_offset + input_length];

                let kernel_offset = out_ch * in_channels * kernel_size + in_ch * kernel_size;
                let kernel_slice = &kernels[kernel_offset..kernel_offset + kernel_size];

                // Temporary buffer for this channel's contribution
                let mut temp_output = vec![0.0; output_length];
                conv1d_f32(input_slice, kernel_slice, &mut temp_output, stride, padding);

                // Add to output channel
                #[cfg(target_arch = "x86_64")]
                {
                    if output_length >= 8 && is_x86_feature_detected!("avx2") {
                        super::x86_64::add_f32_x86_64(out_slice, &temp_output, out_slice);
                    } else {
                        for i in 0..output_length {
                            out_slice[i] += temp_output[i];
                        }
                    }
                }

                #[cfg(target_arch = "aarch64")]
                {
                    if output_length >= 4 {
                        let original_slice = out_slice.to_vec();
                        super::aarch64::add_f32_aarch64(&original_slice, &temp_output, out_slice);
                    } else {
                        for i in 0..output_length {
                            out_slice[i] += temp_output[i];
                        }
                    }
                }

                #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                {
                    for i in 0..output_length {
                        out_slice[i] += temp_output[i];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv1d_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0, 0.5];
        let mut output = vec![0.0; 4];

        conv1d_f32(&input, &kernel, &mut output, 1, 0);

        // Expected: [1*1 + 2*0.5, 2*1 + 3*0.5, 3*1 + 4*0.5, 4*1 + 5*0.5]
        let expected = vec![2.0, 3.5, 5.0, 6.5];
        assert_eq!(output, expected);
    }

    #[test]
    fn test_conv1d_with_padding() {
        let input = vec![1.0, 2.0, 3.0];
        let kernel = vec![1.0, 1.0];
        let mut output = vec![0.0; 4]; // Should handle padding

        conv1d_f32(&input, &kernel, &mut output, 1, 1);

        // With padding=1: [0,1,2,3,0] -> [0*1 + 1*1, 1*1 + 2*1, 2*1 + 3*1, 3*1 + 0*1]
        let expected = vec![1.0, 3.0, 5.0, 3.0];
        assert_eq!(output, expected);
    }

    #[test]
    fn test_transpose_conv1d_basic() {
        let input = vec![1.0, 2.0];
        let kernel = vec![1.0, 0.5];
        let mut output = vec![0.0; 3];

        transpose_conv1d_f32(&input, &kernel, &mut output, 1, 0);

        // Expected: input[0] * kernel at pos 0, input[1] * kernel at pos 1
        let expected = vec![1.0, 2.5, 1.0]; // [1*1, 1*0.5 + 2*1, 2*0.5]
        assert_eq!(output, expected);
    }

    #[test]
    fn test_depthwise_conv1d() {
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 2 channels, 2 samples each
        let kernels = [vec![1.0], vec![2.0]]; // Different kernel per channel
        let kernel_refs: Vec<&[f32]> = kernels.iter().map(|k| k.as_slice()).collect();
        let mut output = vec![0.0; 2]; // 2 channels, 1 sample each (no padding, kernel size 1)

        depthwise_conv1d_f32(&input, &kernel_refs, &mut output, 1, 0);

        // Channel 0: [1.0, 2.0] * [1.0] -> [1.0, 2.0] -> take first element for output
        // Channel 1: [3.0, 4.0] * [2.0] -> [6.0, 8.0] -> take first element for output
        let expected = vec![1.0, 6.0];
        assert_eq!(output, expected);
    }
}
