//! Quantization utility functions
//!
//! This module provides utility functions for quantization operations,
//! including tensor manipulations, statistics calculations, and format conversions.

use crate::{AcousticError, Result};
use std::collections::HashMap;

/// Tensor statistics for quantization analysis
#[derive(Debug, Clone)]
pub struct TensorStatistics {
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Number of elements
    pub count: usize,
}

impl TensorStatistics {
    /// Calculate statistics from tensor data
    pub fn from_data(data: &[f32]) -> Result<Self> {
        if data.is_empty() {
            return Err(AcousticError::Processing(
                "Cannot calculate statistics from empty data".to_string(),
            ));
        }

        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean = data.iter().sum::<f32>() / data.len() as f32;

        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std_dev = variance.sqrt();

        Ok(Self {
            min,
            max,
            mean,
            std_dev,
            count: data.len(),
        })
    }

    /// Get the range (max - min)
    pub fn range(&self) -> f32 {
        self.max - self.min
    }

    /// Check if the data is well-distributed for quantization
    pub fn is_well_distributed(&self) -> bool {
        // Consider data well-distributed if standard deviation is reasonable
        // compared to the range
        let normalized_std = self.std_dev / self.range();
        normalized_std > 0.1 && normalized_std < 2.0
    }
}

/// Quantization format utilities
pub struct QuantizationFormat;

impl QuantizationFormat {
    /// Convert quantization precision to bit width
    pub fn precision_to_bits(precision: &crate::quantization::QuantizationPrecision) -> u8 {
        match precision {
            crate::quantization::QuantizationPrecision::Int4 => 4,
            crate::quantization::QuantizationPrecision::Int8 => 8,
            crate::quantization::QuantizationPrecision::Int16 => 16,
            crate::quantization::QuantizationPrecision::Mixed => 8, // Default to 8-bit
        }
    }

    /// Get quantization range for precision
    pub fn precision_to_range(
        precision: &crate::quantization::QuantizationPrecision,
    ) -> (i32, i32) {
        match precision {
            crate::quantization::QuantizationPrecision::Int4 => (-8, 7),
            crate::quantization::QuantizationPrecision::Int8 => (-128, 127),
            crate::quantization::QuantizationPrecision::Int16 => (-32768, 32767),
            crate::quantization::QuantizationPrecision::Mixed => (-128, 127), // Default to 8-bit
        }
    }

    /// Calculate theoretical compression ratio
    pub fn theoretical_compression_ratio(
        precision: &crate::quantization::QuantizationPrecision,
    ) -> f32 {
        let bits = Self::precision_to_bits(precision) as f32;
        32.0 / bits // Compared to FP32
    }
}

/// Tensor manipulation utilities
pub struct TensorUtils;

impl TensorUtils {
    /// Reshape flat tensor data to specified dimensions
    pub fn reshape(data: &[f32], shape: &[usize]) -> Result<Vec<Vec<f32>>> {
        let total_elements: usize = shape.iter().product();
        if data.len() != total_elements {
            return Err(AcousticError::Processing(format!(
                "Data length {} doesn't match shape dimensions {}",
                data.len(),
                total_elements
            )));
        }

        if shape.len() != 2 {
            return Err(AcousticError::Processing(
                "Only 2D reshaping is currently supported".to_string(),
            ));
        }

        let rows = shape[0];
        let cols = shape[1];
        let mut result = Vec::with_capacity(rows);

        for i in 0..rows {
            let start = i * cols;
            let end = start + cols;
            result.push(data[start..end].to_vec());
        }

        Ok(result)
    }

    /// Flatten 2D tensor to 1D
    pub fn flatten(data: &[Vec<f32>]) -> Vec<f32> {
        data.iter().flatten().copied().collect()
    }

    /// Calculate tensor memory usage in bytes
    pub fn memory_usage_bytes<T>(shape: &[usize]) -> usize {
        let element_count: usize = shape.iter().product();
        element_count * std::mem::size_of::<T>()
    }
}

/// Quantization analysis utilities
pub struct QuantizationAnalysis;

impl QuantizationAnalysis {
    /// Analyze quantization error for given parameters
    pub fn analyze_quantization_error(
        original: &[f32],
        params: &crate::quantization::QuantizationParams,
    ) -> Result<QuantizationErrorMetrics> {
        if original.is_empty() {
            return Err(AcousticError::Processing(
                "Cannot analyze empty tensor".to_string(),
            ));
        }

        let quantized = params.quantize_tensor(original);
        let dequantized = params.dequantize_tensor(&quantized);

        let mut mse = 0.0f32;
        let mut mae = 0.0f32;
        let mut max_error = 0.0f32;

        for (orig, deq) in original.iter().zip(dequantized.iter()) {
            let error = orig - deq;
            let abs_error = error.abs();

            mse += error * error;
            mae += abs_error;
            max_error = max_error.max(abs_error);
        }

        let n = original.len() as f32;
        mse /= n;
        mae /= n;

        let rmse = mse.sqrt();
        let snr = if mse > 0.0 {
            let signal_power = original.iter().map(|&x| x * x).sum::<f32>() / n;
            10.0 * (signal_power / mse).log10()
        } else {
            f32::INFINITY
        };

        Ok(QuantizationErrorMetrics {
            mse,
            rmse,
            mae,
            max_error,
            snr_db: snr,
        })
    }

    /// Find optimal quantization parameters for given data
    pub fn find_optimal_params(
        data: &[f32],
        precision: &crate::quantization::QuantizationPrecision,
        symmetric: bool,
    ) -> Result<crate::quantization::QuantizationParams> {
        let stats = TensorStatistics::from_data(data)?;
        let (qmin, qmax) = QuantizationFormat::precision_to_range(precision);

        if symmetric {
            let abs_max = stats.max.abs().max(stats.min.abs());
            let scale = abs_max / (qmax as f32);
            Ok(crate::quantization::QuantizationParams::symmetric(
                scale, qmin, qmax,
            ))
        } else {
            let scale = (stats.max - stats.min) / (qmax - qmin) as f32;
            let zero_point = qmin - (stats.min / scale).round() as i32;
            Ok(crate::quantization::QuantizationParams::asymmetric(
                scale, zero_point, qmin, qmax,
            ))
        }
    }
}

/// Quantization error metrics
#[derive(Debug, Clone)]
pub struct QuantizationErrorMetrics {
    /// Mean squared error
    pub mse: f32,
    /// Root mean squared error
    pub rmse: f32,
    /// Mean absolute error
    pub mae: f32,
    /// Maximum absolute error
    pub max_error: f32,
    /// Signal-to-noise ratio in dB
    pub snr_db: f32,
}

impl QuantizationErrorMetrics {
    /// Check if quantization error is within acceptable bounds
    pub fn is_acceptable(&self, max_snr_loss_db: f32, max_rmse: f32) -> bool {
        self.snr_db >= max_snr_loss_db && self.rmse <= max_rmse
    }
}

/// Batch processing utilities for quantization
pub struct BatchQuantization;

impl BatchQuantization {
    /// Process multiple tensors in batch
    pub fn quantize_batch(
        tensors: &HashMap<String, Vec<f32>>,
        params: &HashMap<String, crate::quantization::QuantizationParams>,
    ) -> Result<HashMap<String, crate::quantization::QuantizedTensor>> {
        let mut results = HashMap::new();

        for (name, data) in tensors {
            if let Some(param) = params.get(name) {
                let quantized_data = param.quantize_tensor(data);
                let shape = vec![data.len()]; // Simple 1D shape
                let tensor = crate::quantization::QuantizedTensor::new(
                    quantized_data,
                    param.clone(),
                    shape,
                    name.clone(),
                );
                results.insert(name.clone(), tensor);
            }
        }

        Ok(results)
    }

    /// Calculate batch compression statistics
    pub fn calculate_batch_stats(
        tensors: &HashMap<String, crate::quantization::QuantizedTensor>,
    ) -> crate::quantization::QuantizationStats {
        let mut original_size = 0;
        let mut quantized_size = 0;
        let quantized_layers = tensors.len();

        for tensor in tensors.values() {
            original_size += tensor.size() * std::mem::size_of::<f32>();
            quantized_size += tensor.memory_usage();
        }

        let compression_ratio = if quantized_size > 0 {
            original_size as f32 / quantized_size as f32
        } else {
            1.0
        };

        crate::quantization::QuantizationStats {
            original_size,
            quantized_size,
            compression_ratio,
            quantized_layers,
            skipped_layers: 0,
            estimated_accuracy: 0.95, // Placeholder
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_statistics() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = TensorStatistics::from_data(&data).unwrap();

        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.count, 5);
        assert_eq!(stats.range(), 4.0);
    }

    #[test]
    fn test_tensor_reshape() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];

        let reshaped = TensorUtils::reshape(&data, &shape).unwrap();
        assert_eq!(reshaped.len(), 2);
        assert_eq!(reshaped[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(reshaped[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_tensor_flatten() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let flattened = TensorUtils::flatten(&data);
        assert_eq!(flattened, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_quantization_format() {
        assert_eq!(
            QuantizationFormat::precision_to_bits(
                &crate::quantization::QuantizationPrecision::Int8
            ),
            8
        );
        assert_eq!(
            QuantizationFormat::precision_to_bits(
                &crate::quantization::QuantizationPrecision::Int16
            ),
            16
        );

        let (qmin, qmax) = QuantizationFormat::precision_to_range(
            &crate::quantization::QuantizationPrecision::Int8,
        );
        assert_eq!(qmin, -128);
        assert_eq!(qmax, 127);
    }

    #[test]
    fn test_memory_usage_calculation() {
        let shape = vec![10, 20];
        let usage = TensorUtils::memory_usage_bytes::<f32>(&shape);
        assert_eq!(usage, 10 * 20 * 4); // 800 bytes for f32
    }
}
