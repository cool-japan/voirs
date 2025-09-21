//! Model quantization for edge deployment
//!
//! This module provides INT8 and FP16 quantization support for model deployment on edge devices.
//! Quantization reduces memory usage and computational requirements while maintaining quality.

use candle_core::{Device, Result as CandleResult, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quantization configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Target quantization precision
    pub precision: QuantizationPrecision,
    /// Quantization method to use
    pub method: QuantizationMethod,
    /// Calibration dataset size for post-training quantization
    pub calibration_samples: usize,
    /// Enable dynamic quantization
    pub dynamic_quantization: bool,
    /// Percentile for outlier clipping (0.01 = 99th percentile)
    pub outlier_percentile: f32,
    /// Layer-specific quantization settings
    pub layer_configs: HashMap<String, LayerQuantizationConfig>,
    /// Enable quantization-aware training mode
    pub quantization_aware_training: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            precision: QuantizationPrecision::Int8,
            method: QuantizationMethod::PostTrainingQuantization,
            calibration_samples: 100,
            dynamic_quantization: false,
            outlier_percentile: 0.01,
            layer_configs: HashMap::new(),
            quantization_aware_training: false,
        }
    }
}

impl QuantizationConfig {
    /// Create a new quantization config
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a config optimized for mobile deployment
    pub fn mobile_optimized() -> Self {
        Self {
            precision: QuantizationPrecision::Int8,
            method: QuantizationMethod::PostTrainingQuantization,
            calibration_samples: 50,
            dynamic_quantization: true,
            outlier_percentile: 0.005,
            layer_configs: HashMap::new(),
            quantization_aware_training: false,
        }
    }

    /// Create a config optimized for edge devices with severe memory constraints
    pub fn edge_optimized() -> Self {
        let mut layer_configs = HashMap::new();
        // Quantize embedding layers more aggressively
        layer_configs.insert(
            "embedding".to_string(),
            LayerQuantizationConfig {
                precision: QuantizationPrecision::Int4,
                quantize_weights: true,
                quantize_activations: true,
                symmetric: true,
            },
        );

        Self {
            precision: QuantizationPrecision::Int8,
            method: QuantizationMethod::PostTrainingQuantization,
            calibration_samples: 25,
            dynamic_quantization: true,
            outlier_percentile: 0.001,
            layer_configs,
            quantization_aware_training: false,
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> crate::Result<()> {
        if self.calibration_samples == 0 {
            return Err(crate::Error::Config(
                "Calibration samples must be greater than 0".to_string(),
            ));
        }

        if !(0.0..0.1).contains(&self.outlier_percentile) {
            return Err(crate::Error::Config(
                "Outlier percentile must be between 0.0 and 0.1".to_string(),
            ));
        }

        Ok(())
    }
}

/// Quantization precision levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantizationPrecision {
    /// 4-bit integer (extreme compression)
    Int4,
    /// 8-bit integer (standard quantization)
    Int8,
    /// 16-bit integer
    Int16,
    /// 16-bit floating point (half precision)
    Float16,
    /// Mixed precision (different precisions for different layers)
    Mixed,
}

impl QuantizationPrecision {
    /// Get the bits per parameter for this precision
    pub fn bits_per_param(&self) -> u8 {
        match self {
            QuantizationPrecision::Int4 => 4,
            QuantizationPrecision::Int8 => 8,
            QuantizationPrecision::Int16 => 16,
            QuantizationPrecision::Float16 => 16,
            QuantizationPrecision::Mixed => 8, // Average estimate
        }
    }

    /// Get memory reduction ratio compared to FP32
    pub fn memory_reduction_ratio(&self) -> f32 {
        32.0 / self.bits_per_param() as f32
    }
}

/// Quantization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationMethod {
    /// Post-training quantization (no retraining required)
    PostTrainingQuantization,
    /// Quantization-aware training (requires fine-tuning)
    QuantizationAwareTraining,
    /// Dynamic quantization (runtime quantization)
    DynamicQuantization,
    /// Knowledge distillation with quantization
    KnowledgeDistillation,
}

/// Layer-specific quantization configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerQuantizationConfig {
    /// Precision for this layer
    pub precision: QuantizationPrecision,
    /// Whether to quantize weights
    pub quantize_weights: bool,
    /// Whether to quantize activations
    pub quantize_activations: bool,
    /// Use symmetric quantization
    pub symmetric: bool,
}

impl Default for LayerQuantizationConfig {
    fn default() -> Self {
        Self {
            precision: QuantizationPrecision::Int8,
            quantize_weights: true,
            quantize_activations: true,
            symmetric: false,
        }
    }
}

/// Quantization statistics for calibration
#[derive(Debug, Clone)]
pub struct QuantizationStats {
    /// Minimum value observed
    pub min_val: f32,
    /// Maximum value observed
    pub max_val: f32,
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std: f32,
    /// Number of samples
    pub num_samples: usize,
}

impl QuantizationStats {
    /// Create new empty stats
    pub fn new() -> Self {
        Self {
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            mean: 0.0,
            std: 0.0,
            num_samples: 0,
        }
    }

    /// Update stats with new tensor values
    pub fn update(&mut self, tensor: &Tensor) -> CandleResult<()> {
        let flat_tensor = tensor.flatten_all()?;
        let values: Vec<f32> = flat_tensor.to_vec1()?;

        for &val in &values {
            self.min_val = self.min_val.min(val);
            self.max_val = self.max_val.max(val);
        }

        // Update running statistics
        let old_count = self.num_samples;
        self.num_samples += values.len();

        // Update mean using Welford's algorithm
        let old_mean = self.mean;
        let sum: f32 = values.iter().sum();
        self.mean = (old_mean * old_count as f32 + sum) / self.num_samples as f32;

        // Update standard deviation
        let sum_sq_diff: f32 = values.iter().map(|&x| (x - self.mean).powi(2)).sum();
        let old_sum_sq = self.std.powi(2) * old_count as f32;
        self.std = ((old_sum_sq + sum_sq_diff) / self.num_samples as f32).sqrt();

        Ok(())
    }

    /// Get quantization scale and zero point for given precision
    pub fn get_quantization_params(
        &self,
        precision: QuantizationPrecision,
        symmetric: bool,
    ) -> (f32, i32) {
        let (min_quant, max_quant) = match precision {
            QuantizationPrecision::Int4 => {
                if symmetric {
                    (-8, 7)
                } else {
                    (0, 15)
                }
            }
            QuantizationPrecision::Int8 => {
                if symmetric {
                    (-128, 127)
                } else {
                    (0, 255)
                }
            }
            QuantizationPrecision::Int16 => {
                if symmetric {
                    (-32768, 32767)
                } else {
                    (0, 65535)
                }
            }
            _ => (0, 255), // Fallback to Int8
        };

        if symmetric {
            let abs_max = self.max_val.abs().max(self.min_val.abs());
            let scale = abs_max / max_quant as f32;
            (scale, 0) // Zero point is 0 for symmetric quantization
        } else {
            let scale = (self.max_val - self.min_val) / (max_quant - min_quant) as f32;
            let zero_point = (min_quant as f32 - self.min_val / scale).round() as i32;
            (scale, zero_point.clamp(min_quant, max_quant))
        }
    }
}

impl Default for QuantizationStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Model quantizer
#[derive(Debug)]
pub struct ModelQuantizer {
    /// Quantization configuration
    config: QuantizationConfig,
    /// Statistics collector for calibration
    stats_collector: HashMap<String, QuantizationStats>,
    /// Device for computations
    device: Device,
    /// Whether calibration is active
    calibration_active: bool,
}

impl ModelQuantizer {
    /// Create a new model quantizer
    pub fn new(config: QuantizationConfig, device: Device) -> crate::Result<Self> {
        config.validate()?;

        Ok(Self {
            config,
            stats_collector: HashMap::new(),
            device,
            calibration_active: false,
        })
    }

    /// Get quantization config
    pub fn config(&self) -> &QuantizationConfig {
        &self.config
    }

    /// Start calibration phase
    pub fn start_calibration(&mut self) {
        self.calibration_active = true;
        self.stats_collector.clear();
    }

    /// Finish calibration phase
    pub fn finish_calibration(&mut self) {
        self.calibration_active = false;
    }

    /// Calibrate with a tensor (collect statistics)
    pub fn calibrate(&mut self, layer_name: &str, tensor: &Tensor) -> CandleResult<()> {
        if !self.calibration_active {
            return Ok(());
        }

        let stats = self
            .stats_collector
            .entry(layer_name.to_string())
            .or_insert_with(QuantizationStats::new);

        stats.update(tensor)?;
        Ok(())
    }

    /// Quantize a tensor to specified precision
    pub fn quantize_tensor(
        &self,
        tensor: &Tensor,
        layer_name: &str,
        precision: QuantizationPrecision,
    ) -> CandleResult<QuantizedTensor> {
        let layer_config = self
            .config
            .layer_configs
            .get(layer_name)
            .cloned()
            .unwrap_or_default();

        let stats = self.stats_collector.get(layer_name);

        match precision {
            QuantizationPrecision::Int8 => {
                self.quantize_int8(tensor, stats, layer_config.symmetric)
            }
            QuantizationPrecision::Int4 => {
                self.quantize_int4(tensor, stats, layer_config.symmetric)
            }
            QuantizationPrecision::Float16 => self.quantize_float16(tensor),
            QuantizationPrecision::Int16 => {
                self.quantize_int16(tensor, stats, layer_config.symmetric)
            }
            QuantizationPrecision::Mixed => {
                // Default to Int8 for mixed precision
                self.quantize_int8(tensor, stats, layer_config.symmetric)
            }
        }
    }

    /// Quantize to INT8
    fn quantize_int8(
        &self,
        tensor: &Tensor,
        stats: Option<&QuantizationStats>,
        symmetric: bool,
    ) -> CandleResult<QuantizedTensor> {
        let (scale, zero_point) = if let Some(stats) = stats {
            stats.get_quantization_params(QuantizationPrecision::Int8, symmetric)
        } else {
            // Fallback to dynamic quantization
            self.compute_dynamic_quantization_params(
                tensor,
                QuantizationPrecision::Int8,
                symmetric,
            )?
        };

        let scale_tensor = Tensor::new(&[scale], tensor.device())?.broadcast_as(tensor.shape())?;
        let quantized = if symmetric {
            ((tensor / scale_tensor)?.round()?.clamp(-128.0, 127.0)?)
                .to_dtype(candle_core::DType::I64)?
        } else {
            let zero_tensor =
                Tensor::new(&[zero_point as f64], tensor.device())?.broadcast_as(tensor.shape())?;
            (((tensor / scale_tensor)? + zero_tensor)?
                .round()?
                .clamp(0.0, 255.0)?)
            .to_dtype(candle_core::DType::I64)?
        };

        Ok(QuantizedTensor {
            data: quantized,
            scale,
            zero_point,
            precision: QuantizationPrecision::Int8,
            symmetric,
            original_shape: tensor.shape().clone(),
        })
    }

    /// Quantize to INT4
    fn quantize_int4(
        &self,
        tensor: &Tensor,
        stats: Option<&QuantizationStats>,
        symmetric: bool,
    ) -> CandleResult<QuantizedTensor> {
        let (scale, zero_point) = if let Some(stats) = stats {
            stats.get_quantization_params(QuantizationPrecision::Int4, symmetric)
        } else {
            self.compute_dynamic_quantization_params(
                tensor,
                QuantizationPrecision::Int4,
                symmetric,
            )?
        };

        let scale_tensor = Tensor::new(&[scale], tensor.device())?.broadcast_as(tensor.shape())?;
        let quantized = if symmetric {
            ((tensor / scale_tensor)?.round()?.clamp(-8.0, 7.0)?)
                .to_dtype(candle_core::DType::I64)?
        } else {
            let zero_tensor =
                Tensor::new(&[zero_point as f64], tensor.device())?.broadcast_as(tensor.shape())?;
            (((tensor / scale_tensor)? + zero_tensor)?
                .round()?
                .clamp(0.0, 15.0)?)
            .to_dtype(candle_core::DType::I64)?
        };

        Ok(QuantizedTensor {
            data: quantized,
            scale,
            zero_point,
            precision: QuantizationPrecision::Int4,
            symmetric,
            original_shape: tensor.shape().clone(),
        })
    }

    /// Quantize to INT16
    fn quantize_int16(
        &self,
        tensor: &Tensor,
        stats: Option<&QuantizationStats>,
        symmetric: bool,
    ) -> CandleResult<QuantizedTensor> {
        let (scale, zero_point) = if let Some(stats) = stats {
            stats.get_quantization_params(QuantizationPrecision::Int16, symmetric)
        } else {
            self.compute_dynamic_quantization_params(
                tensor,
                QuantizationPrecision::Int16,
                symmetric,
            )?
        };

        let scale_tensor = Tensor::new(&[scale], tensor.device())?.broadcast_as(tensor.shape())?;
        let quantized = if symmetric {
            ((tensor / scale_tensor)?.round()?.clamp(-32768.0, 32767.0)?)
                .to_dtype(candle_core::DType::I64)?
        } else {
            let zero_tensor =
                Tensor::new(&[zero_point as f64], tensor.device())?.broadcast_as(tensor.shape())?;
            (((tensor / scale_tensor)? + zero_tensor)?
                .round()?
                .clamp(0.0, 65535.0)?)
            .to_dtype(candle_core::DType::I64)?
        };

        Ok(QuantizedTensor {
            data: quantized,
            scale,
            zero_point,
            precision: QuantizationPrecision::Int16,
            symmetric,
            original_shape: tensor.shape().clone(),
        })
    }

    /// Quantize to Float16
    fn quantize_float16(&self, tensor: &Tensor) -> CandleResult<QuantizedTensor> {
        let quantized = tensor.to_dtype(candle_core::DType::F16)?;

        Ok(QuantizedTensor {
            data: quantized,
            scale: 1.0,
            zero_point: 0,
            precision: QuantizationPrecision::Float16,
            symmetric: true,
            original_shape: tensor.shape().clone(),
        })
    }

    /// Compute dynamic quantization parameters
    fn compute_dynamic_quantization_params(
        &self,
        tensor: &Tensor,
        precision: QuantizationPrecision,
        symmetric: bool,
    ) -> CandleResult<(f32, i32)> {
        let min_val = tensor.min(0)?.to_vec0::<f32>()?;
        let max_val = tensor.max(0)?.to_vec0::<f32>()?;

        let mut temp_stats = QuantizationStats::new();
        temp_stats.min_val = min_val;
        temp_stats.max_val = max_val;

        Ok(temp_stats.get_quantization_params(precision, symmetric))
    }

    /// Get quantization statistics summary
    pub fn get_stats_summary(&self) -> HashMap<String, QuantizationStatsSummary> {
        self.stats_collector
            .iter()
            .map(|(layer, stats)| {
                let summary = QuantizationStatsSummary {
                    layer_name: layer.clone(),
                    min_val: stats.min_val,
                    max_val: stats.max_val,
                    mean: stats.mean,
                    std: stats.std,
                    dynamic_range: stats.max_val - stats.min_val,
                    num_samples: stats.num_samples,
                };
                (layer.clone(), summary)
            })
            .collect()
    }

    /// Estimate memory savings from quantization
    pub fn estimate_memory_savings(
        &self,
        original_model_size_mb: f32,
    ) -> QuantizationMemoryAnalysis {
        let reduction_ratio = self.config.precision.memory_reduction_ratio();
        let quantized_size_mb = original_model_size_mb / reduction_ratio;
        let savings_mb = original_model_size_mb - quantized_size_mb;
        let savings_percent = (savings_mb / original_model_size_mb) * 100.0;

        QuantizationMemoryAnalysis {
            original_size_mb: original_model_size_mb,
            quantized_size_mb,
            savings_mb,
            savings_percent,
            compression_ratio: reduction_ratio,
            precision: self.config.precision,
        }
    }
}

/// Quantized tensor representation
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized data
    pub data: Tensor,
    /// Quantization scale
    pub scale: f32,
    /// Zero point for asymmetric quantization
    pub zero_point: i32,
    /// Quantization precision
    pub precision: QuantizationPrecision,
    /// Whether quantization is symmetric
    pub symmetric: bool,
    /// Original tensor shape
    pub original_shape: candle_core::Shape,
}

impl QuantizedTensor {
    /// Dequantize back to float32
    pub fn dequantize(&self) -> CandleResult<Tensor> {
        match self.precision {
            QuantizationPrecision::Float16 => {
                // For FP16, just convert back to F32
                self.data.to_dtype(candle_core::DType::F32)
            }
            _ => {
                // For integer quantization, apply scale and zero point
                let float_data = self.data.to_dtype(candle_core::DType::F32)?;
                let scale_tensor = Tensor::new(&[self.scale], self.data.device())?
                    .broadcast_as(float_data.shape())?;
                if self.symmetric {
                    Ok((&float_data * scale_tensor)?)
                } else {
                    let zero_tensor = Tensor::new(&[self.zero_point as f64], self.data.device())?
                        .broadcast_as(float_data.shape())?;
                    Ok(((&float_data - zero_tensor)? * scale_tensor)?)
                }
            }
        }
    }

    /// Get memory usage of quantized tensor in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        let num_elements = self.data.elem_count();
        let bytes_per_element = match self.precision {
            QuantizationPrecision::Int4 => 1, // Packed 2 elements per byte
            QuantizationPrecision::Int8 => 1,
            QuantizationPrecision::Int16 | QuantizationPrecision::Float16 => 2,
            QuantizationPrecision::Mixed => 1, // Average estimate
        };

        if self.precision == QuantizationPrecision::Int4 {
            (num_elements + 1) / 2 // Ceiling division for packed 4-bit
        } else {
            num_elements * bytes_per_element
        }
    }
}

/// Summary statistics for quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationStatsSummary {
    /// Layer name
    pub layer_name: String,
    /// Minimum value
    pub min_val: f32,
    /// Maximum value
    pub max_val: f32,
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std: f32,
    /// Dynamic range
    pub dynamic_range: f32,
    /// Number of samples used for calibration
    pub num_samples: usize,
}

/// Memory analysis for quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationMemoryAnalysis {
    /// Original model size in MB
    pub original_size_mb: f32,
    /// Quantized model size in MB
    pub quantized_size_mb: f32,
    /// Memory savings in MB
    pub savings_mb: f32,
    /// Percentage savings
    pub savings_percent: f32,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Quantization precision used
    pub precision: QuantizationPrecision,
}

/// Quantization result with performance metrics
#[derive(Debug, Clone)]
pub struct QuantizationResult {
    /// Quantized tensors by layer name
    pub quantized_tensors: HashMap<String, QuantizedTensor>,
    /// Memory analysis
    pub memory_analysis: QuantizationMemoryAnalysis,
    /// Statistics summary
    pub stats_summary: HashMap<String, QuantizationStatsSummary>,
    /// Quantization configuration used
    pub config: QuantizationConfig,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Shape, Tensor};

    #[test]
    fn test_quantization_config_default() {
        let config = QuantizationConfig::default();
        assert_eq!(config.precision, QuantizationPrecision::Int8);
        assert_eq!(config.calibration_samples, 100);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_quantization_config_mobile() {
        let config = QuantizationConfig::mobile_optimized();
        assert_eq!(config.precision, QuantizationPrecision::Int8);
        assert_eq!(config.calibration_samples, 50);
        assert!(config.dynamic_quantization);
    }

    #[test]
    fn test_quantization_config_edge() {
        let config = QuantizationConfig::edge_optimized();
        assert_eq!(config.precision, QuantizationPrecision::Int8);
        assert_eq!(config.calibration_samples, 25);
        assert!(config.layer_configs.contains_key("embedding"));
    }

    #[test]
    fn test_quantization_precision_bits() {
        assert_eq!(QuantizationPrecision::Int4.bits_per_param(), 4);
        assert_eq!(QuantizationPrecision::Int8.bits_per_param(), 8);
        assert_eq!(QuantizationPrecision::Int16.bits_per_param(), 16);
        assert_eq!(QuantizationPrecision::Float16.bits_per_param(), 16);
    }

    #[test]
    fn test_quantization_precision_memory_reduction() {
        // FP32 to INT8 should give 4x reduction
        assert_eq!(QuantizationPrecision::Int8.memory_reduction_ratio(), 4.0);
        // FP32 to INT4 should give 8x reduction
        assert_eq!(QuantizationPrecision::Int4.memory_reduction_ratio(), 8.0);
        // FP32 to FP16 should give 2x reduction
        assert_eq!(QuantizationPrecision::Float16.memory_reduction_ratio(), 2.0);
    }

    #[test]
    fn test_quantization_stats() {
        let device = Device::Cpu;
        let data = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], (5,), &device).unwrap();

        let mut stats = QuantizationStats::new();
        stats.update(&data).unwrap();

        assert_eq!(stats.min_val, 1.0);
        assert_eq!(stats.max_val, 5.0);
        assert_eq!(stats.num_samples, 5);

        let (scale, zero_point) = stats.get_quantization_params(QuantizationPrecision::Int8, false);
        assert!(scale > 0.0);
        assert!(zero_point >= 0 && zero_point <= 255);
    }

    #[test]
    fn test_model_quantizer_creation() {
        let config = QuantizationConfig::default();
        let device = Device::Cpu;

        let quantizer = ModelQuantizer::new(config, device);
        assert!(quantizer.is_ok());
    }

    #[test]
    fn test_quantized_tensor_memory_usage() {
        let device = Device::Cpu;
        let data = Tensor::zeros((100,), DType::I64, &device).unwrap();

        let quantized = QuantizedTensor {
            data,
            scale: 1.0,
            zero_point: 0,
            precision: QuantizationPrecision::Int8,
            symmetric: true,
            original_shape: Shape::from_dims(&[100]),
        };

        // 100 elements * 1 byte per element = 100 bytes
        assert_eq!(quantized.memory_usage_bytes(), 100);
    }

    #[test]
    fn test_quantized_tensor_memory_usage_int4() {
        let device = Device::Cpu;
        let data = Tensor::zeros((100,), DType::I64, &device).unwrap();

        let quantized = QuantizedTensor {
            data,
            scale: 1.0,
            zero_point: 0,
            precision: QuantizationPrecision::Int4,
            symmetric: true,
            original_shape: Shape::from_dims(&[100]),
        };

        // 100 elements, packed 2 per byte = 50 bytes
        assert_eq!(quantized.memory_usage_bytes(), 50);
    }

    #[test]
    fn test_layer_quantization_config_default() {
        let config = LayerQuantizationConfig::default();
        assert_eq!(config.precision, QuantizationPrecision::Int8);
        assert!(config.quantize_weights);
        assert!(config.quantize_activations);
        assert!(!config.symmetric);
    }
}
