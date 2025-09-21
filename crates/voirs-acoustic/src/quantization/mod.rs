//! Model quantization utilities for compression and optimization
//!
//! This module provides various quantization techniques including post-training quantization (PTQ),
//! quantization-aware training (QAT), and dynamic range calibration for neural acoustic models.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::{AcousticError, Result};

pub mod calibration;
pub mod ptq;
pub mod qat;
pub mod utils;

/// Quantization precision types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationPrecision {
    /// 8-bit integer quantization
    Int8,
    /// 16-bit integer quantization
    Int16,
    /// 4-bit integer quantization (experimental)
    Int4,
    /// Mixed precision (some layers remain FP32)
    Mixed,
}

/// Quantization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMethod {
    /// Post-training quantization
    PostTraining,
    /// Quantization-aware training
    AwareTraining,
    /// Dynamic quantization
    Dynamic,
}

/// Quantization configuration
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Target precision
    pub precision: QuantizationPrecision,
    /// Quantization method
    pub method: QuantizationMethod,
    /// Calibration dataset size
    pub calibration_samples: usize,
    /// Layers to skip quantization
    pub skip_layers: Vec<String>,
    /// Symmetric vs asymmetric quantization
    pub symmetric: bool,
    /// Per-channel vs per-tensor quantization
    pub per_channel: bool,
    /// Target accuracy retention (0.0 to 1.0)
    pub target_accuracy: f32,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            precision: QuantizationPrecision::Int8,
            method: QuantizationMethod::PostTraining,
            calibration_samples: 1000,
            skip_layers: vec!["output".to_string()], // Usually skip output layer
            symmetric: true,
            per_channel: true,
            target_accuracy: 0.95, // 95% accuracy retention
        }
    }
}

/// Quantization parameters for a tensor
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    /// Scale factor
    pub scale: f32,
    /// Zero point
    pub zero_point: i32,
    /// Quantization range
    pub qmin: i32,
    /// Quantization maximum
    pub qmax: i32,
    /// Whether quantization is symmetric
    pub symmetric: bool,
}

impl QuantizationParams {
    /// Create new quantization parameters
    pub fn new(scale: f32, zero_point: i32, qmin: i32, qmax: i32, symmetric: bool) -> Self {
        Self {
            scale,
            zero_point,
            qmin,
            qmax,
            symmetric,
        }
    }

    /// Create symmetric quantization parameters
    pub fn symmetric(scale: f32, qmin: i32, qmax: i32) -> Self {
        Self {
            scale,
            zero_point: 0,
            qmin,
            qmax,
            symmetric: true,
        }
    }

    /// Create asymmetric quantization parameters
    pub fn asymmetric(scale: f32, zero_point: i32, qmin: i32, qmax: i32) -> Self {
        Self {
            scale,
            zero_point,
            qmin,
            qmax,
            symmetric: false,
        }
    }

    /// Quantize a value
    pub fn quantize(&self, value: f32) -> i32 {
        let quantized = (value / self.scale).round() as i32 + self.zero_point;
        quantized.clamp(self.qmin, self.qmax)
    }

    /// Dequantize a value
    pub fn dequantize(&self, quantized: i32) -> f32 {
        (quantized - self.zero_point) as f32 * self.scale
    }

    /// Quantize a tensor
    pub fn quantize_tensor(&self, input: &[f32]) -> Vec<i32> {
        input.iter().map(|&x| self.quantize(x)).collect()
    }

    /// Dequantize a tensor
    pub fn dequantize_tensor(&self, quantized: &[i32]) -> Vec<f32> {
        quantized.iter().map(|&x| self.dequantize(x)).collect()
    }
}

/// Quantized tensor representation
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized data
    pub data: Vec<i32>,
    /// Quantization parameters
    pub params: QuantizationParams,
    /// Original tensor shape
    pub shape: Vec<usize>,
    /// Tensor name/identifier
    pub name: String,
}

impl QuantizedTensor {
    /// Create new quantized tensor
    pub fn new(
        data: Vec<i32>,
        params: QuantizationParams,
        shape: Vec<usize>,
        name: String,
    ) -> Self {
        Self {
            data,
            params,
            shape,
            name,
        }
    }

    /// Dequantize tensor to floating point
    pub fn dequantize(&self) -> Vec<f32> {
        self.params.dequantize_tensor(&self.data)
    }

    /// Get tensor size in elements
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get memory usage in bytes (for quantized representation)
    pub fn memory_usage(&self) -> usize {
        self.data.len() * std::mem::size_of::<i32>()
    }

    /// Get compression ratio compared to FP32
    pub fn compression_ratio(&self) -> f32 {
        let fp32_size = self.size() * std::mem::size_of::<f32>();
        let quantized_size = self.memory_usage();
        fp32_size as f32 / quantized_size as f32
    }
}

/// Model quantizer for applying quantization to acoustic models
pub struct ModelQuantizer {
    /// Quantization configuration
    config: QuantizationConfig,
    /// Quantization parameters per layer
    layer_params: Arc<Mutex<HashMap<String, QuantizationParams>>>,
    /// Calibration data cache
    calibration_cache: Arc<Mutex<HashMap<String, Vec<f32>>>>,
}

impl ModelQuantizer {
    /// Create new model quantizer
    pub fn new(config: QuantizationConfig) -> Self {
        Self {
            config,
            layer_params: Arc::new(Mutex::new(HashMap::new())),
            calibration_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Add calibration data for a layer
    pub fn add_calibration_data(&self, layer_name: String, data: Vec<f32>) -> Result<()> {
        let mut cache = self.calibration_cache.lock().unwrap();
        cache.insert(layer_name, data);
        Ok(())
    }

    /// Calibrate quantization parameters for all layers
    pub fn calibrate(&self) -> Result<()> {
        let cache = self.calibration_cache.lock().unwrap();
        let mut params = self.layer_params.lock().unwrap();

        for (layer_name, data) in cache.iter() {
            if self.config.skip_layers.contains(layer_name) {
                continue;
            }

            let qparams = self.calculate_quantization_params(data)?;
            params.insert(layer_name.clone(), qparams);
        }

        Ok(())
    }

    /// Calculate quantization parameters from calibration data
    fn calculate_quantization_params(&self, data: &[f32]) -> Result<QuantizationParams> {
        if data.is_empty() {
            return Err(AcousticError::Processing(
                "Cannot calculate quantization params from empty data".to_string(),
            ));
        }

        let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let (qmin, qmax) = match self.config.precision {
            QuantizationPrecision::Int8 => (-128, 127),
            QuantizationPrecision::Int16 => (-32768, 32767),
            QuantizationPrecision::Int4 => (-8, 7),
            QuantizationPrecision::Mixed => (-128, 127), // Default to Int8 for mixed
        };

        if self.config.symmetric {
            let abs_max = max_val.abs().max(min_val.abs());
            let scale = abs_max / (qmax as f32);
            Ok(QuantizationParams::symmetric(scale, qmin, qmax))
        } else {
            let scale = (max_val - min_val) / (qmax - qmin) as f32;
            let zero_point = qmin - (min_val / scale).round() as i32;
            Ok(QuantizationParams::asymmetric(
                scale, zero_point, qmin, qmax,
            ))
        }
    }

    /// Quantize a tensor using calibrated parameters
    pub fn quantize_tensor(
        &self,
        layer_name: &str,
        data: &[f32],
        shape: Vec<usize>,
    ) -> Result<QuantizedTensor> {
        let params = self.layer_params.lock().unwrap();
        let qparams = params.get(layer_name).ok_or_else(|| {
            AcousticError::Processing(format!(
                "No quantization parameters found for layer: {layer_name}"
            ))
        })?;

        let quantized_data = qparams.quantize_tensor(data);
        Ok(QuantizedTensor::new(
            quantized_data,
            qparams.clone(),
            shape,
            layer_name.to_string(),
        ))
    }

    /// Get quantization parameters for a layer
    pub fn get_layer_params(&self, layer_name: &str) -> Option<QuantizationParams> {
        self.layer_params.lock().unwrap().get(layer_name).cloned()
    }

    /// Get configuration
    pub fn config(&self) -> &QuantizationConfig {
        &self.config
    }

    /// Get calibration progress
    pub fn calibration_progress(&self) -> f32 {
        let cache = self.calibration_cache.lock().unwrap();
        if cache.is_empty() {
            0.0
        } else {
            let params = self.layer_params.lock().unwrap();
            params.len() as f32 / cache.len() as f32
        }
    }
}

/// Quantization statistics for model analysis
#[derive(Debug, Clone)]
pub struct QuantizationStats {
    /// Total model size before quantization (bytes)
    pub original_size: usize,
    /// Total model size after quantization (bytes)
    pub quantized_size: usize,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Number of quantized layers
    pub quantized_layers: usize,
    /// Number of skipped layers
    pub skipped_layers: usize,
    /// Estimated accuracy retention
    pub estimated_accuracy: f32,
}

impl QuantizationStats {
    /// Calculate compression savings
    pub fn compression_savings(&self) -> f32 {
        if self.original_size == 0 {
            0.0
        } else {
            1.0 - (self.quantized_size as f32 / self.original_size as f32)
        }
    }

    /// Calculate memory savings in MB
    pub fn memory_savings_mb(&self) -> f32 {
        (self.original_size - self.quantized_size) as f32 / (1024.0 * 1024.0)
    }
}

/// Quantization benchmark for performance testing
#[derive(Debug, Clone)]
pub struct QuantizationBenchmark {
    /// Original model inference time (ms)
    pub original_inference_ms: f32,
    /// Quantized model inference time (ms)
    pub quantized_inference_ms: f32,
    /// Speedup factor
    pub speedup: f32,
    /// Accuracy on test set (original)
    pub original_accuracy: f32,
    /// Accuracy on test set (quantized)
    pub quantized_accuracy: f32,
    /// Accuracy degradation
    pub accuracy_degradation: f32,
}

impl QuantizationBenchmark {
    /// Create new benchmark
    pub fn new(
        original_inference_ms: f32,
        quantized_inference_ms: f32,
        original_accuracy: f32,
        quantized_accuracy: f32,
    ) -> Self {
        let speedup = original_inference_ms / quantized_inference_ms;
        let accuracy_degradation = original_accuracy - quantized_accuracy;

        Self {
            original_inference_ms,
            quantized_inference_ms,
            speedup,
            original_accuracy,
            quantized_accuracy,
            accuracy_degradation,
        }
    }

    /// Check if quantization meets quality targets
    pub fn meets_targets(&self, target_speedup: f32, max_accuracy_loss: f32) -> bool {
        self.speedup >= target_speedup && self.accuracy_degradation <= max_accuracy_loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_params() {
        let params = QuantizationParams::symmetric(0.1, -128, 127);
        assert_eq!(params.scale, 0.1);
        assert_eq!(params.zero_point, 0);
        assert!(params.symmetric);

        // Test quantization
        let value = 1.0;
        let quantized = params.quantize(value);
        let dequantized = params.dequantize(quantized);
        assert!((dequantized - value).abs() < 0.1);
    }

    #[test]
    fn test_quantized_tensor() {
        let data = vec![1, 2, 3, 4];
        let params = QuantizationParams::symmetric(0.1, -128, 127);
        let shape = vec![2, 2];
        let tensor = QuantizedTensor::new(data, params, shape, "test".to_string());

        assert_eq!(tensor.size(), 4);
        assert_eq!(tensor.memory_usage(), 16); // 4 * 4 bytes
        assert_eq!(tensor.compression_ratio(), 1.0); // Same as i32 vs f32
    }

    #[test]
    fn test_model_quantizer() {
        let config = QuantizationConfig::default();
        let quantizer = ModelQuantizer::new(config);

        // Add calibration data
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        quantizer
            .add_calibration_data("layer1".to_string(), data.clone())
            .unwrap();

        // Calibrate
        quantizer.calibrate().unwrap();

        // Check parameters were calculated
        assert!(quantizer.get_layer_params("layer1").is_some());

        // Quantize tensor
        let quantized = quantizer.quantize_tensor("layer1", &data, vec![5]).unwrap();
        assert_eq!(quantized.size(), 5);
    }

    #[test]
    fn test_quantization_config() {
        let config = QuantizationConfig::default();
        assert_eq!(config.precision, QuantizationPrecision::Int8);
        assert_eq!(config.method, QuantizationMethod::PostTraining);
        assert_eq!(config.calibration_samples, 1000);
    }

    #[test]
    fn test_quantization_benchmark() {
        let benchmark = QuantizationBenchmark::new(100.0, 50.0, 0.95, 0.92);
        assert_eq!(benchmark.speedup, 2.0);
        assert!((benchmark.accuracy_degradation - 0.03).abs() < 1e-6);
        assert!(benchmark.meets_targets(1.5, 0.05));
        assert!(!benchmark.meets_targets(3.0, 0.02));
    }

    #[test]
    fn test_quantization_stats() {
        let stats = QuantizationStats {
            original_size: 1000,
            quantized_size: 250,
            compression_ratio: 4.0,
            quantized_layers: 8,
            skipped_layers: 2,
            estimated_accuracy: 0.94,
        };

        assert_eq!(stats.compression_savings(), 0.75);
        assert!((stats.memory_savings_mb() - 0.000715).abs() < 0.0001);
    }
}
