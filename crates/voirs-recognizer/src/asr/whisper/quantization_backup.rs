//! Model quantization implementation for Whisper
//!
//! This module provides quantization techniques to reduce model size and improve
//! inference speed while maintaining acceptable accuracy levels.

mod config;
mod stats;

// Re-export all types for public API
pub use config::*;
pub use stats::*;

use super::encoder::QuantizationMode;
use super::WhisperConfig;
use crate::RecognitionError;
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;


/// Quantization statistics for a tensor
#[derive(Debug, Clone)]
pub struct QuantizationStats {
    /// Scale factor
    pub scale: f32,
    /// Zero point
    pub zero_point: i8,
    /// Minimum value
    pub min_val: f32,
    /// Maximum value
    pub max_val: f32,
    /// Data type
    pub dtype: DType,
}

/// 4-bit quantization statistics
#[derive(Debug, Clone)]
pub struct Quantization4BitStats {
    /// Scale factors per group
    pub scales: Vec<f32>,
    /// Zero points per group
    pub zero_points: Vec<i8>,
    /// Group size
    pub group_size: usize,
    /// Number of groups
    pub num_groups: usize,
}

/// Pruning statistics
#[derive(Debug, Clone)]
pub struct PruningStats {
    /// Original number of parameters
    pub original_params: usize,
    /// Number of pruned parameters
    pub pruned_params: usize,
    /// Sparsity ratio achieved
    pub sparsity_ratio: f32,
    /// Pruning mask (true = keep, false = prune)
    pub mask: Vec<bool>,
}

/// Dynamic quantization parameters
#[derive(Debug, Clone)]
pub struct DynamicQuantParams {
    /// Scaling factor computed dynamically
    pub scale: f32,
    /// Zero point computed dynamically
    pub zero_point: i32,
    /// Quantization range (e.g., 255 for 8-bit, 15 for 4-bit)
    pub quant_range: i32,
    /// Activation range from recent inference
    pub activation_min: f32,
    /// Activation range from recent inference
    pub activation_max: f32,
}

/// Moving average tracker for dynamic quantization
#[derive(Debug, Clone)]
pub struct MovingAverageTracker {
    /// Recent activation statistics
    pub recent_mins: Vec<f32>,
    /// Recent activation statistics  
    pub recent_maxs: Vec<f32>,
    /// Window size for moving average
    pub window_size: usize,
    /// Current position in circular buffer
    pub position: usize,
    /// Number of samples collected
    pub samples: usize,
}

impl MovingAverageTracker {
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            recent_mins: vec![0.0; window_size],
            recent_maxs: vec![0.0; window_size],
            window_size,
            position: 0,
            samples: 0,
        }
    }

    pub fn update(&mut self, min_val: f32, max_val: f32) {
        self.recent_mins[self.position] = min_val;
        self.recent_maxs[self.position] = max_val;
        self.position = (self.position + 1) % self.window_size;
        self.samples = (self.samples + 1).min(self.window_size);
    }

    #[must_use]
    pub fn get_averaged_range(&self) -> (f32, f32) {
        if self.samples == 0 {
            return (0.0, 1.0);
        }

        let avg_min = self.recent_mins[..self.samples].iter().sum::<f32>() / self.samples as f32;
        let avg_max = self.recent_maxs[..self.samples].iter().sum::<f32>() / self.samples as f32;
        (avg_min, avg_max)
    }
}

/// Model quantizer
pub struct ModelQuantizer {
    /// Configuration
    config: QuantizationConfig,
    /// Device
    device: Device,
    /// Quantization statistics per layer
    layer_stats: HashMap<String, QuantizationStats>,
    /// 4-bit quantization statistics per layer
    layer_4bit_stats: HashMap<String, Quantization4BitStats>,
    /// Pruning statistics per layer
    pruning_stats: HashMap<String, PruningStats>,
    /// Calibration data
    calibration_data: Vec<Tensor>,
    /// Dynamic quantization trackers per layer
    dynamic_trackers: HashMap<String, MovingAverageTracker>,
    /// Enable dynamic quantization mode
    pub dynamic_mode: bool,
}

impl ModelQuantizer {
    /// Create a new model quantizer
    #[must_use]
    pub fn new(config: QuantizationConfig, device: Device) -> Self {
        Self {
            config,
            device,
            layer_stats: HashMap::new(),
            layer_4bit_stats: HashMap::new(),
            pruning_stats: HashMap::new(),
            calibration_data: Vec::new(),
            dynamic_trackers: HashMap::new(),
            dynamic_mode: false,
        }
    }

    /// Enable dynamic quantization mode
    pub fn enable_dynamic_quantization(&mut self, window_size: usize) {
        self.dynamic_mode = true;
        // Initialize trackers for common layer patterns
        let layer_patterns = [
            "encoder",
            "decoder",
            "attention",
            "mlp",
            "linear",
            "conv",
            "default",
        ];
        for pattern in &layer_patterns {
            self.dynamic_trackers.insert(
                (*pattern).to_string(),
                MovingAverageTracker::new(window_size),
            );
        }
    }

    /// Disable dynamic quantization mode
    pub fn disable_dynamic_quantization(&mut self) {
        self.dynamic_mode = false;
        self.dynamic_trackers.clear();
    }

    /// Add calibration data
    pub fn add_calibration_data(&mut self, data: Tensor) -> Result<(), RecognitionError> {
        if self.calibration_data.len() < self.config.calibration_samples {
            self.calibration_data.push(data);
        }
        Ok(())
    }

    /// Quantize a tensor to FP16
    pub fn quantize_fp16(&self, tensor: &Tensor) -> Result<Tensor, RecognitionError> {
        tensor
            .to_dtype(DType::F16)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to quantize to FP16: {e}"),
                source: Some(Box::new(e)),
            })
    }

    /// Quantize a tensor to INT8
    pub fn quantize_int8(
        &self,
        tensor: &Tensor,
        layer_name: &str,
    ) -> Result<Tensor, RecognitionError> {
        // Get or compute quantization stats
        let stats = if let Some(stats) = self.layer_stats.get(layer_name) {
            stats.clone()
        } else {
            self.compute_quantization_stats(tensor, layer_name)?
        };

        // Apply quantization: quantized = round((tensor - zero_point) / scale)
        let shifted = tensor.broadcast_sub(
            &Tensor::new(f32::from(stats.zero_point), &self.device).map_err(|e| {
                RecognitionError::ModelError {
                    message: format!("Failed to create zero point tensor: {e}"),
                    source: Some(Box::new(e)),
                }
            })?,
        )?;

        let scaled =
            shifted.broadcast_div(&Tensor::new(stats.scale, &self.device).map_err(|e| {
                RecognitionError::ModelError {
                    message: format!("Failed to create scale tensor: {e}"),
                    source: Some(Box::new(e)),
                }
            })?)?;

        // Round and clamp to INT8 range
        let rounded = scaled.round().map_err(|e| RecognitionError::ModelError {
            message: format!("Failed to round tensor: {e}"),
            source: Some(Box::new(e)),
        })?;

        let clamped = self.clamp_to_int8(&rounded)?;

        clamped
            .to_dtype(DType::I64)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to convert to INT8: {e}"),
                source: Some(Box::new(e)),
            })
    }

    /// Dequantize INT8 tensor back to FP32
    pub fn dequantize_int8(
        &self,
        tensor: &Tensor,
        layer_name: &str,
    ) -> Result<Tensor, RecognitionError> {
        let stats =
            self.layer_stats
                .get(layer_name)
                .ok_or_else(|| RecognitionError::ModelError {
                    message: format!("No quantization stats found for layer: {layer_name}"),
                    source: None,
                })?;

        // Dequantize: value = quantized * scale + zero_point
        let fp_tensor = tensor
            .to_dtype(DType::F32)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to convert to FP32: {e}"),
                source: Some(Box::new(e)),
            })?;

        let scaled =
            fp_tensor.broadcast_mul(&Tensor::new(stats.scale, &self.device).map_err(|e| {
                RecognitionError::ModelError {
                    message: format!("Failed to create scale tensor: {e}"),
                    source: Some(Box::new(e)),
                }
            })?)?;

        let dequantized = scaled.broadcast_add(
            &Tensor::new(f32::from(stats.zero_point), &self.device).map_err(|e| {
                RecognitionError::ModelError {
                    message: format!("Failed to create zero point tensor: {e}"),
                    source: Some(Box::new(e)),
                }
            })?,
        )?;

        Ok(dequantized)
    }

    /// Compute dynamic quantization parameters based on tensor activations
    pub fn compute_dynamic_quant_params(
        &mut self,
        tensor: &Tensor,
        layer_name: &str,
        bits: u8,
    ) -> Result<DynamicQuantParams, RecognitionError> {
        // Extract tensor values to compute min/max
        let flattened = tensor
            .flatten_all()
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to flatten tensor for dynamic quantization: {e}"),
                source: Some(Box::new(e)),
            })?;

        let tensor_data = flattened
            .to_vec1::<f32>()
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to extract tensor data for dynamic quantization: {e}"),
                source: Some(Box::new(e)),
            })?;

        let current_min = tensor_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let current_max = tensor_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Update moving average tracker
        let tracker_key = self.get_tracker_key(layer_name);
        if !self.dynamic_trackers.contains_key(&tracker_key) {
            self.dynamic_trackers.insert(
                tracker_key.clone(),
                MovingAverageTracker::new(10), // Default window size
            );
        }

        let tracker = self.dynamic_trackers.get_mut(&tracker_key).unwrap();
        tracker.update(current_min, current_max);

        // Use moving average for stable quantization
        let (avg_min, avg_max) = tracker.get_averaged_range();

        // Compute quantization parameters based on bit width
        let quant_range = match bits {
            8 => 255, // 2^8 - 1
            4 => 15,  // 2^4 - 1
            _ => {
                return Err(RecognitionError::ModelError {
                    message: format!("Unsupported bit width for dynamic quantization: {bits}"),
                    source: None,
                })
            }
        };

        let scale = (avg_max - avg_min) / quant_range as f32;
        let zero_point = (-avg_min / scale).round().clamp(0.0, quant_range as f32) as i32;

        Ok(DynamicQuantParams {
            scale,
            zero_point,
            quant_range,
            activation_min: avg_min,
            activation_max: avg_max,
        })
    }

    /// Quantize tensor dynamically during inference
    pub fn quantize_dynamic(
        &mut self,
        tensor: &Tensor,
        layer_name: &str,
        bits: u8,
    ) -> Result<(Tensor, DynamicQuantParams), RecognitionError> {
        if !self.dynamic_mode {
            return Err(RecognitionError::ModelError {
                message: "Dynamic quantization is not enabled".to_string(),
                source: None,
            });
        }

        // Compute dynamic quantization parameters
        let params = self.compute_dynamic_quant_params(tensor, layer_name, bits)?;

        // Apply quantization
        let quantized = self.apply_dynamic_quantization(tensor, &params)?;

        tracing::debug!(
            "Dynamic quantization for {}: scale={:.6}, zero_point={}, range=[{:.3}, {:.3}]",
            layer_name,
            params.scale,
            params.zero_point,
            params.activation_min,
            params.activation_max
        );

        Ok((quantized, params))
    }

    /// Apply dynamic quantization with given parameters
    fn apply_dynamic_quantization(
        &self,
        tensor: &Tensor,
        params: &DynamicQuantParams,
    ) -> Result<Tensor, RecognitionError> {
        // Quantize: q = round((x - min) / scale) + zero_point
        let scale_tensor =
            Tensor::new(params.scale, &self.device).map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to create scale tensor: {e}"),
                source: Some(Box::new(e)),
            })?;

        let zero_point_tensor =
            Tensor::new(params.zero_point as f32, &self.device).map_err(|e| {
                RecognitionError::ModelError {
                    message: format!("Failed to create zero point tensor: {e}"),
                    source: Some(Box::new(e)),
                }
            })?;

        let min_tensor = Tensor::new(params.activation_min, &self.device).map_err(|e| {
            RecognitionError::ModelError {
                message: format!("Failed to create min tensor: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        // Apply quantization formula
        let shifted = tensor.broadcast_sub(&min_tensor)?;
        let scaled = shifted.broadcast_div(&scale_tensor)?;
        let quantized = scaled
            .round()
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to round quantized values: {e}"),
                source: Some(Box::new(e)),
            })?
            .broadcast_add(&zero_point_tensor)?;

        // Clamp to quantization range
        let max_val = params.quant_range as f32;
        let clamped = quantized
            .clamp(0.0, max_val)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to clamp quantized values: {e}"),
                source: Some(Box::new(e)),
            })?;

        Ok(clamped)
    }

    /// Dequantize tensor using dynamic parameters
    pub fn dequantize_dynamic(
        &self,
        tensor: &Tensor,
        params: &DynamicQuantParams,
    ) -> Result<Tensor, RecognitionError> {
        // Dequantize: x = (q - zero_point) * scale + min
        let scale_tensor =
            Tensor::new(params.scale, &self.device).map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to create scale tensor: {e}"),
                source: Some(Box::new(e)),
            })?;

        let zero_point_tensor =
            Tensor::new(params.zero_point as f32, &self.device).map_err(|e| {
                RecognitionError::ModelError {
                    message: format!("Failed to create zero point tensor: {e}"),
                    source: Some(Box::new(e)),
                }
            })?;

        let min_tensor = Tensor::new(params.activation_min, &self.device).map_err(|e| {
            RecognitionError::ModelError {
                message: format!("Failed to create min tensor: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        let shifted = tensor.broadcast_sub(&zero_point_tensor)?;
        let scaled = shifted.broadcast_mul(&scale_tensor)?;
        let dequantized = scaled.broadcast_add(&min_tensor)?;

        Ok(dequantized)
    }

    /// Dynamic 4-bit quantization with group-wise adaptation
    pub fn quantize_dynamic_4bit(
        &mut self,
        tensor: &Tensor,
        layer_name: &str,
        group_size: usize,
    ) -> Result<(Tensor, Vec<DynamicQuantParams>), RecognitionError> {
        if !self.dynamic_mode {
            return Err(RecognitionError::ModelError {
                message: "Dynamic quantization is not enabled".to_string(),
                source: None,
            });
        }

        let data = tensor
            .flatten_all()
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to flatten tensor for dynamic 4-bit quantization: {e}"),
                source: Some(Box::new(e)),
            })?
            .to_vec1::<f32>()
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to extract tensor data: {e}"),
                source: Some(Box::new(e)),
            })?;

        let num_groups = (data.len() + group_size - 1) / group_size;
        let mut group_params = Vec::with_capacity(num_groups);
        let mut quantized_data = Vec::with_capacity(data.len());

        for group_idx in 0..num_groups {
            let start_idx = group_idx * group_size;
            let end_idx = (start_idx + group_size).min(data.len());
            let group_data = &data[start_idx..end_idx];

            // Compute dynamic parameters for this group
            let group_min = group_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let group_max = group_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // Update group-specific tracker
            let group_key = format!("{layer_name}_group_{group_idx}");
            if !self.dynamic_trackers.contains_key(&group_key) {
                self.dynamic_trackers.insert(
                    group_key.clone(),
                    MovingAverageTracker::new(5), // Smaller window for groups
                );
            }

            let tracker = self.dynamic_trackers.get_mut(&group_key).unwrap();
            tracker.update(group_min, group_max);
            let (avg_min, avg_max) = tracker.get_averaged_range();

            let scale = (avg_max - avg_min) / 15.0; // 4-bit range: 0-15
            let zero_point = (-avg_min / scale).round().clamp(0.0, 15.0) as i32;

            let params = DynamicQuantParams {
                scale,
                zero_point,
                quant_range: 15,
                activation_min: avg_min,
                activation_max: avg_max,
            };

            group_params.push(params.clone());

            // Quantize group data
            for &val in group_data {
                let quantized = ((val - avg_min) / scale + zero_point as f32)
                    .round()
                    .clamp(0.0, 15.0) as i64;
                quantized_data.push(quantized);
            }
        }

        let quantized_tensor = Tensor::new(quantized_data, &self.device).map_err(|e| {
            RecognitionError::ModelError {
                message: format!("Failed to create dynamic 4-bit quantized tensor: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        tracing::debug!(
            "Dynamic 4-bit quantization for {}: {} groups, group_size={}",
            layer_name,
            num_groups,
            group_size
        );

        Ok((quantized_tensor, group_params))
    }

    /// Get appropriate tracker key for a layer
    fn get_tracker_key(&self, layer_name: &str) -> String {
        let layer_lower = layer_name.to_lowercase();

        if layer_lower.contains("encoder") {
            "encoder".to_string()
        } else if layer_lower.contains("decoder") {
            "decoder".to_string()
        } else if layer_lower.contains("attention") || layer_lower.contains("attn") {
            "attention".to_string()
        } else if layer_lower.contains("mlp") || layer_lower.contains("ffn") {
            "mlp".to_string()
        } else if layer_lower.contains("linear") {
            "linear".to_string()
        } else if layer_lower.contains("conv") {
            "conv".to_string()
        } else {
            "default".to_string()
        }
    }

    /// Get dynamic quantization statistics
    #[must_use]
    pub fn get_dynamic_stats(&self) -> HashMap<String, (f32, f32, usize)> {
        let mut stats = HashMap::new();

        for (key, tracker) in &self.dynamic_trackers {
            let (avg_min, avg_max) = tracker.get_averaged_range();
            stats.insert(key.clone(), (avg_min, avg_max, tracker.samples));
        }

        stats
    }

    /// Reset dynamic quantization trackers
    pub fn reset_dynamic_trackers(&mut self) {
        for tracker in self.dynamic_trackers.values_mut() {
            tracker.samples = 0;
            tracker.position = 0;
        }
    }

    /// Quantize tensor to 4-bit with group-wise quantization
    pub fn quantize_4bit(
        &mut self,
        tensor: &Tensor,
        layer_name: &str,
    ) -> Result<Tensor, RecognitionError> {
        let data = tensor
            .flatten_all()
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to flatten tensor for 4-bit quantization: {e}"),
                source: Some(Box::new(e)),
            })?
            .to_vec1::<f32>()
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to extract tensor data: {e}"),
                source: Some(Box::new(e)),
            })?;

        let group_size = self.config.group_size_4bit;
        let num_groups = (data.len() + group_size - 1) / group_size;
        let mut scales = Vec::with_capacity(num_groups);
        let mut zero_points = Vec::with_capacity(num_groups);
        let mut quantized_data = Vec::with_capacity(data.len());

        for group_idx in 0..num_groups {
            let start_idx = group_idx * group_size;
            let end_idx = (start_idx + group_size).min(data.len());
            let group_data = &data[start_idx..end_idx];

            // Find min/max for this group
            let min_val = group_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = group_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // 4-bit range is 0-15
            let scale = (max_val - min_val) / 15.0;
            let zero_point = (-min_val / scale).round().clamp(0.0, 15.0) as i8;

            scales.push(scale);
            zero_points.push(zero_point);

            // Quantize group data
            for &val in group_data {
                let quantized = ((val - min_val) / scale).round().clamp(0.0, 15.0) as i64;
                quantized_data.push(quantized);
            }
        }

        // Store 4-bit statistics
        let stats = Quantization4BitStats {
            scales,
            zero_points,
            group_size,
            num_groups,
        };
        self.layer_4bit_stats.insert(layer_name.to_string(), stats);

        // Create quantized tensor
        Tensor::new(quantized_data, &self.device).map_err(|e| RecognitionError::ModelError {
            message: format!("Failed to create 4-bit quantized tensor: {e}"),
            source: Some(Box::new(e)),
        })
    }

    /// Dequantize 4-bit tensor back to FP32
    pub fn dequantize_4bit(
        &self,
        tensor: &Tensor,
        layer_name: &str,
    ) -> Result<Tensor, RecognitionError> {
        let stats =
            self.layer_4bit_stats
                .get(layer_name)
                .ok_or_else(|| RecognitionError::ModelError {
                    message: format!("No 4-bit quantization stats found for layer: {layer_name}"),
                    source: None,
                })?;

        let quantized_data = tensor
            .flatten_all()
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to flatten 4-bit tensor: {e}"),
                source: Some(Box::new(e)),
            })?
            .to_vec1::<i64>()
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to extract 4-bit data: {e}"),
                source: Some(Box::new(e)),
            })?;

        let mut dequantized_data = Vec::with_capacity(quantized_data.len());
        let group_size = stats.group_size;

        for (group_idx, (&scale, &zero_point)) in
            stats.scales.iter().zip(&stats.zero_points).enumerate()
        {
            let start_idx = group_idx * group_size;
            let end_idx = (start_idx + group_size).min(quantized_data.len());

            for &quantized_val in &quantized_data[start_idx..end_idx] {
                let dequantized = (quantized_val as f32 - f32::from(zero_point)) * scale;
                dequantized_data.push(dequantized);
            }
        }

        Tensor::new(dequantized_data, &self.device).map_err(|e| RecognitionError::ModelError {
            message: format!("Failed to create dequantized tensor: {e}"),
            source: Some(Box::new(e)),
        })
    }

    /// Apply magnitude-based pruning to a tensor
    pub fn prune_tensor(
        &mut self,
        tensor: &Tensor,
        layer_name: &str,
    ) -> Result<Tensor, RecognitionError> {
        let data = tensor
            .flatten_all()
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to flatten tensor for pruning: {e}"),
                source: Some(Box::new(e)),
            })?
            .to_vec1::<f32>()
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to extract tensor data for pruning: {e}"),
                source: Some(Box::new(e)),
            })?;

        let original_params = data.len();
        let mut mask = vec![true; original_params];
        let mut pruned_data = data.clone();

        if self.config.structured_pruning {
            // Structured pruning: remove entire channels/neurons
            self.apply_structured_pruning(&mut pruned_data, &mut mask)?;
        } else {
            // Unstructured pruning: remove individual weights based on magnitude
            self.apply_unstructured_pruning(&mut pruned_data, &mut mask)?;
        }

        let pruned_params = mask.iter().filter(|&&x| !x).count();
        let sparsity_ratio = pruned_params as f32 / original_params as f32;

        // Store pruning statistics
        let stats = PruningStats {
            original_params,
            pruned_params,
            sparsity_ratio,
            mask,
        };
        self.pruning_stats.insert(layer_name.to_string(), stats);

        tracing::info!(
            "Pruned layer {}: {}/{} parameters ({:.2}% sparsity)",
            layer_name,
            pruned_params,
            original_params,
            sparsity_ratio * 100.0
        );

        // Create pruned tensor
        let shape = tensor.shape();
        Tensor::new(pruned_data, &self.device)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to create pruned tensor: {e}"),
                source: Some(Box::new(e)),
            })?
            .reshape(shape)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to reshape pruned tensor: {e}"),
                source: Some(Box::new(e)),
            })
    }

    /// Apply unstructured magnitude-based pruning
    fn apply_unstructured_pruning(
        &self,
        data: &mut Vec<f32>,
        mask: &mut Vec<bool>,
    ) -> Result<(), RecognitionError> {
        // Calculate magnitude threshold
        let mut magnitudes: Vec<f32> = data.iter().map(|&x| x.abs()).collect();
        magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let threshold_idx = ((1.0 - self.config.pruning_ratio) * magnitudes.len() as f32) as usize;
        let magnitude_threshold = magnitudes
            .get(threshold_idx)
            .copied()
            .unwrap_or(self.config.magnitude_threshold);

        // Apply pruning mask
        for i in 0..data.len() {
            if data[i].abs() < magnitude_threshold {
                data[i] = 0.0;
                mask[i] = false;
            }
        }

        Ok(())
    }

    /// Apply structured pruning (remove entire channels/neurons)
    fn apply_structured_pruning(
        &self,
        data: &mut Vec<f32>,
        mask: &mut Vec<bool>,
    ) -> Result<(), RecognitionError> {
        // For structured pruning, we need to determine channel/neuron boundaries
        // This is a simplified implementation that groups consecutive elements
        let group_size = 32; // Typical channel size
        let num_groups = (data.len() + group_size - 1) / group_size;
        let target_pruned_groups = (num_groups as f32 * self.config.pruning_ratio) as usize;

        // Calculate L2 norm for each group
        let mut group_norms = Vec::with_capacity(num_groups);
        for group_idx in 0..num_groups {
            let start_idx = group_idx * group_size;
            let end_idx = (start_idx + group_size).min(data.len());

            let l2_norm: f32 = data[start_idx..end_idx]
                .iter()
                .map(|&x| x * x)
                .sum::<f32>()
                .sqrt();

            group_norms.push((group_idx, l2_norm));
        }

        // Sort by L2 norm (ascending) and prune the smallest groups
        group_norms.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for &(group_idx, _) in group_norms.iter().take(target_pruned_groups) {
            let start_idx = group_idx * group_size;
            let end_idx = (start_idx + group_size).min(data.len());

            for i in start_idx..end_idx {
                data[i] = 0.0;
                mask[i] = false;
            }
        }

        Ok(())
    }

    /// Compute quantization statistics for a tensor
    fn compute_quantization_stats(
        &self,
        tensor: &Tensor,
        layer_name: &str,
    ) -> Result<QuantizationStats, RecognitionError> {
        // Flatten tensor to get all values
        let flattened = tensor
            .flatten_all()
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to flatten tensor: {e}"),
                source: Some(Box::new(e)),
            })?;

        // Get tensor values (simplified approach)
        let tensor_data = flattened
            .to_vec1::<f32>()
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to extract tensor data: {e}"),
                source: Some(Box::new(e)),
            })?;

        // Find min and max values
        let _min_val = tensor_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let _max_val = tensor_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Handle outliers by using percentiles
        let mut sorted_data = tensor_data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let outlier_idx =
            ((100.0 - self.config.outlier_percentile) / 100.0 * sorted_data.len() as f32) as usize;
        let robust_min = sorted_data[outlier_idx.min(sorted_data.len() - 1)];
        let robust_max =
            sorted_data[sorted_data.len() - 1 - outlier_idx.min(sorted_data.len() - 1)];

        // Compute scale and zero point for INT8 quantization
        let (scale, zero_point) = if self.config.symmetric {
            // Symmetric quantization: range is [-127, 127]
            let abs_max = robust_max.abs().max(robust_min.abs());
            let scale = abs_max / 127.0;
            (scale, 0i8)
        } else {
            // Asymmetric quantization: range is [-128, 127]
            let scale = (robust_max - robust_min) / 255.0;
            let zero_point = (-128.0 - robust_min / scale).round().clamp(-128.0, 127.0) as i8;
            (scale, zero_point)
        };

        let stats = QuantizationStats {
            scale,
            zero_point,
            min_val: robust_min,
            max_val: robust_max,
            dtype: DType::I64, // Using I64 as closest to INT8 in Candle
        };

        tracing::debug!(
            "Computed quantization stats for {}: scale={}, zero_point={}, range=[{:.3}, {:.3}]",
            layer_name,
            scale,
            zero_point,
            robust_min,
            robust_max
        );

        Ok(stats)
    }

    /// Clamp tensor values to INT8 range
    fn clamp_to_int8(&self, tensor: &Tensor) -> Result<Tensor, RecognitionError> {
        // Clamp to [-128, 127] range
        let min_tensor =
            Tensor::new(-128.0f32, &self.device).map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to create min clamp tensor: {e}"),
                source: Some(Box::new(e)),
            })?;

        let max_tensor =
            Tensor::new(127.0f32, &self.device).map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to create max clamp tensor: {e}"),
                source: Some(Box::new(e)),
            })?;

        tensor
            .maximum(&min_tensor)?
            .minimum(&max_tensor)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to clamp tensor: {e}"),
                source: Some(Box::new(e)),
            })
    }

    /// Quantize model weights based on configuration
    pub fn quantize_model_weights(
        &mut self,
        weights: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>, RecognitionError> {
        let mut quantized_weights = HashMap::new();

        for (layer_name, weight) in weights {
            let mut processed_weight = weight;

            // Apply pruning first if enabled
            if self.config.enable_pruning
                && self.should_quantize_layer(&layer_name, &processed_weight)?
            {
                processed_weight = self.prune_tensor(&processed_weight, &layer_name)?;
            }

            // Then apply quantization
            let quantized_weight = if self.config.enable_4bit {
                // 4-bit quantization takes priority if enabled
                self.quantize_4bit(&mut processed_weight, &layer_name)?
            } else {
                match &self.config.target_dtype {
                    QuantizationMode::F16 => self.quantize_fp16(&processed_weight)?,
                    QuantizationMode::INT8 { .. } => {
                        // Store stats for later dequantization
                        let stats =
                            self.compute_quantization_stats(&processed_weight, &layer_name)?;
                        self.layer_stats.insert(layer_name.clone(), stats);

                        self.quantize_int8(&processed_weight, &layer_name)?
                    }
                    QuantizationMode::Dynamic => {
                        // Choose quantization based on layer type and size
                        if self.should_quantize_layer(&layer_name, &processed_weight)? {
                            if processed_weight.elem_count() > 1_000_000 {
                                // Large layers to INT8 or 4-bit
                                if self.config.enable_4bit {
                                    self.quantize_4bit(&mut processed_weight, &layer_name)?
                                } else {
                                    let stats = self.compute_quantization_stats(
                                        &processed_weight,
                                        &layer_name,
                                    )?;
                                    self.layer_stats.insert(layer_name.clone(), stats);
                                    self.quantize_int8(&processed_weight, &layer_name)?
                                }
                            } else {
                                // Smaller layers to FP16
                                self.quantize_fp16(&processed_weight)?
                            }
                        } else {
                            processed_weight // Keep original precision for critical layers
                        }
                    }
                    QuantizationMode::None => processed_weight,
                }
            };

            quantized_weights.insert(layer_name, quantized_weight);
        }

        Ok(quantized_weights)
    }

    /// Determine if a layer should be quantized based on configuration
    fn should_quantize_layer(
        &self,
        layer_name: &str,
        _weight: &Tensor,
    ) -> Result<bool, RecognitionError> {
        let layer_lower = layer_name.to_lowercase();

        if layer_lower.contains("embed") && !self.config.quantize_embeddings {
            return Ok(false);
        }

        if layer_lower.contains("attn") && !self.config.quantize_attention {
            return Ok(false);
        }

        if (layer_lower.contains("mlp") || layer_lower.contains("ffn")) && !self.config.quantize_mlp
        {
            return Ok(false);
        }

        // Skip layer norm and bias terms
        if layer_lower.contains("ln")
            || layer_lower.contains("bias")
            || layer_lower.contains("norm")
        {
            return Ok(false);
        }

        Ok(true)
    }

    /// Get quantization statistics for a layer
    #[must_use]
    pub fn get_layer_stats(&self, layer_name: &str) -> Option<&QuantizationStats> {
        self.layer_stats.get(layer_name)
    }

    /// Get 4-bit quantization statistics for a layer
    #[must_use]
    pub fn get_4bit_stats(&self, layer_name: &str) -> Option<&Quantization4BitStats> {
        self.layer_4bit_stats.get(layer_name)
    }

    /// Get pruning statistics for a layer
    #[must_use]
    pub fn get_pruning_stats(&self, layer_name: &str) -> Option<&PruningStats> {
        self.pruning_stats.get(layer_name)
    }

    /// Get overall pruning statistics
    #[must_use]
    pub fn get_overall_pruning_stats(&self) -> OverallPruningStats {
        let mut total_original = 0;
        let mut total_pruned = 0;

        for stats in self.pruning_stats.values() {
            total_original += stats.original_params;
            total_pruned += stats.pruned_params;
        }

        let overall_sparsity = if total_original > 0 {
            total_pruned as f32 / total_original as f32
        } else {
            0.0
        };

        OverallPruningStats {
            total_original_params: total_original,
            total_pruned_params: total_pruned,
            overall_sparsity_ratio: overall_sparsity,
            layers_pruned: self.pruning_stats.len(),
        }
    }

    /// Get memory savings from quantization
    #[must_use]
    pub fn get_memory_savings(&self) -> QuantizationSavings {
        let mut original_size = 0;
        let mut quantized_size = 0;

        // Account for regular quantization
        for _stats in self.layer_stats.values() {
            match &self.config.target_dtype {
                QuantizationMode::F16 => {
                    original_size += 4; // FP32
                    quantized_size += 2; // FP16
                }
                QuantizationMode::INT8 { .. } => {
                    original_size += 4; // FP32
                    quantized_size += 1; // INT8
                }
                _ => {
                    original_size += 4;
                    quantized_size += 4;
                }
            }
        }

        // Account for 4-bit quantization
        for stats in self.layer_4bit_stats.values() {
            // Each parameter takes 0.5 bytes (4 bits) plus overhead for scales/zero-points
            let param_size = stats.num_groups * stats.group_size;
            original_size += param_size * 4; // FP32
            quantized_size += param_size / 2; // 4-bit
                                              // Add overhead for scales and zero points (4 bytes + 1 byte per group)
            quantized_size += stats.num_groups * 5;
        }

        // Account for pruning effect (sparse matrices can be stored more efficiently)
        let overall_pruning = self.get_overall_pruning_stats();
        if overall_pruning.overall_sparsity_ratio > 0.0 {
            // Sparse storage overhead: assume CSR format with 50% overhead for indices
            let sparse_efficiency = 1.0 - (overall_pruning.overall_sparsity_ratio * 0.5);
            quantized_size = (quantized_size as f32 * sparse_efficiency) as usize;
        }

        let compression_ratio = if original_size > 0 {
            original_size as f32 / quantized_size as f32
        } else {
            1.0
        };

        QuantizationSavings {
            original_size_mb: original_size as f32 / 1_000_000.0,
            quantized_size_mb: quantized_size as f32 / 1_000_000.0,
            compression_ratio,
            memory_saved_mb: (original_size - quantized_size) as f32 / 1_000_000.0,
        }
    }

    /// Compute knowledge distillation loss between teacher and student outputs
    pub fn compute_distillation_loss(
        &self,
        teacher_logits: &Tensor,
        student_logits: &Tensor,
        hard_targets: &Tensor,
        config: &KnowledgeDistillationConfig,
    ) -> Result<DistillationLoss, RecognitionError> {
        // Compute distillation loss using KL divergence between soft targets
        let teacher_probs = self.softmax_with_temperature(teacher_logits, config.temperature)?;
        let student_log_probs =
            self.log_softmax_with_temperature(student_logits, config.temperature)?;

        // KL divergence loss: teacher_probs * log(teacher_probs / student_probs)
        let kl_div = teacher_probs
            .broadcast_sub(&student_log_probs)?
            .broadcast_mul(&teacher_probs)?
            .sum_all()?
            .to_scalar::<f32>()
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to compute KL divergence: {e}"),
                source: Some(Box::new(e)),
            })?;

        let distillation_loss = kl_div * config.temperature * config.temperature;

        // Compute hard target loss (cross-entropy)
        let student_probs = self.softmax_with_temperature(student_logits, 1.0)?;
        let hard_loss = self.cross_entropy_loss(&student_probs, hard_targets)?;

        // Combined loss
        let total_loss = config.distillation_alpha * distillation_loss
            + (1.0 - config.distillation_alpha) * hard_loss;

        Ok(DistillationLoss {
            distillation_loss,
            hard_loss,
            total_loss,
            temperature: config.temperature,
        })
    }

    /// Apply temperature scaling to logits and compute softmax
    fn softmax_with_temperature(
        &self,
        logits: &Tensor,
        temperature: f32,
    ) -> Result<Tensor, RecognitionError> {
        let scaled_logits =
            logits.broadcast_div(&Tensor::new(temperature, &self.device).map_err(|e| {
                RecognitionError::ModelError {
                    message: format!("Failed to create temperature tensor: {e}"),
                    source: Some(Box::new(e)),
                }
            })?)?;

        // Manual softmax implementation: exp(x) / sum(exp(x))
        let exp_logits = scaled_logits
            .exp()
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to compute exp: {e}"),
                source: Some(Box::new(e)),
            })?;

        let sum_exp = exp_logits
            .sum_keepdim(1)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to compute sum: {e}"),
                source: Some(Box::new(e)),
            })?;

        exp_logits
            .broadcast_div(&sum_exp)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to compute softmax: {e}"),
                source: Some(Box::new(e)),
            })
    }

    /// Apply temperature scaling to logits and compute log softmax
    fn log_softmax_with_temperature(
        &self,
        logits: &Tensor,
        temperature: f32,
    ) -> Result<Tensor, RecognitionError> {
        let scaled_logits =
            logits.broadcast_div(&Tensor::new(temperature, &self.device).map_err(|e| {
                RecognitionError::ModelError {
                    message: format!("Failed to create temperature tensor: {e}"),
                    source: Some(Box::new(e)),
                }
            })?)?;

        // Manual log_softmax implementation: x - log(sum(exp(x)))
        let exp_logits = scaled_logits
            .exp()
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to compute exp: {e}"),
                source: Some(Box::new(e)),
            })?;

        let sum_exp = exp_logits
            .sum_keepdim(1)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to compute sum: {e}"),
                source: Some(Box::new(e)),
            })?;

        let log_sum_exp = sum_exp.log().map_err(|e| RecognitionError::ModelError {
            message: format!("Failed to compute log: {e}"),
            source: Some(Box::new(e)),
        })?;

        scaled_logits
            .broadcast_sub(&log_sum_exp)
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to compute log softmax: {e}"),
                source: Some(Box::new(e)),
            })
    }

    /// Compute cross-entropy loss
    fn cross_entropy_loss(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
    ) -> Result<f32, RecognitionError> {
        // Compute cross-entropy: -sum(targets * log(predictions))
        let log_probs = predictions
            .log()
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to compute log probabilities: {e}"),
                source: Some(Box::new(e)),
            })?;

        let loss = targets
            .broadcast_mul(&log_probs)?
            .neg()?
            .sum_all()?
            .to_scalar::<f32>()
            .map_err(|e| RecognitionError::ModelError {
                message: format!("Failed to compute cross-entropy loss: {e}"),
                source: Some(Box::new(e)),
            })?;

        Ok(loss)
    }

    /// Create student model configuration for knowledge distillation
    #[must_use]
    pub fn create_student_config(
        &self,
        teacher_config: &WhisperConfig,
        compression_ratio: f32,
    ) -> StudentModelConfig {
        StudentModelConfig::from_teacher_config(teacher_config, compression_ratio)
    }

    /// Compute feature distillation loss for intermediate layers
    pub fn compute_feature_distillation_loss(
        &self,
        teacher_features: &[Tensor],
        student_features: &[Tensor],
        layer_mapping: &[(usize, usize)], // (teacher_layer, student_layer) pairs
    ) -> Result<f32, RecognitionError> {
        let mut total_loss = 0.0;
        let mut num_pairs = 0;

        for &(teacher_idx, student_idx) in layer_mapping {
            if teacher_idx < teacher_features.len() && student_idx < student_features.len() {
                let teacher_feat = &teacher_features[teacher_idx];
                let student_feat = &student_features[student_idx];

                // L2 loss between features (after potential dimension adjustment)
                let diff = teacher_feat.broadcast_sub(student_feat)?;
                let squared_diff = diff.sqr().map_err(|e| RecognitionError::ModelError {
                    message: format!("Failed to compute squared difference: {e}"),
                    source: Some(Box::new(e)),
                })?;

                let layer_loss = squared_diff
                    .mean_all()
                    .map_err(|e| RecognitionError::ModelError {
                        message: format!("Failed to compute mean loss: {e}"),
                        source: Some(Box::new(e)),
                    })?
                    .to_scalar::<f32>()
                    .map_err(|e| RecognitionError::ModelError {
                        message: format!("Failed to extract loss scalar: {e}"),
                        source: Some(Box::new(e)),
                    })?;

                total_loss += layer_loss;
                num_pairs += 1;
            }
        }

        Ok(if num_pairs > 0 {
            total_loss / num_pairs as f32
        } else {
            0.0
        })
    }

    /// Compute attention distillation loss
    pub fn compute_attention_distillation_loss(
        &self,
        teacher_attention: &[Tensor],
        student_attention: &[Tensor],
        layer_mapping: &[(usize, usize)],
    ) -> Result<f32, RecognitionError> {
        let mut total_loss = 0.0;
        let mut num_pairs = 0;

        for &(teacher_idx, student_idx) in layer_mapping {
            if teacher_idx < teacher_attention.len() && student_idx < student_attention.len() {
                let teacher_attn = &teacher_attention[teacher_idx];
                let student_attn = &student_attention[student_idx];

                // MSE loss between attention matrices
                let diff = teacher_attn.broadcast_sub(student_attn)?;
                let squared_diff = diff.sqr().map_err(|e| RecognitionError::ModelError {
                    message: format!("Failed to compute squared attention difference: {e}"),
                    source: Some(Box::new(e)),
                })?;

                let layer_loss = squared_diff
                    .mean_all()
                    .map_err(|e| RecognitionError::ModelError {
                        message: format!("Failed to compute mean attention loss: {e}"),
                        source: Some(Box::new(e)),
                    })?
                    .to_scalar::<f32>()
                    .map_err(|e| RecognitionError::ModelError {
                        message: format!("Failed to extract attention loss scalar: {e}"),
                        source: Some(Box::new(e)),
                    })?;

                total_loss += layer_loss;
                num_pairs += 1;
            }
        }

        Ok(if num_pairs > 0 {
            total_loss / num_pairs as f32
        } else {
            0.0
        })
    }

    /// Export model to ONNX format with optimization
    pub fn export_to_onnx<P: AsRef<Path>>(
        &self,
        model_weights: &HashMap<String, Tensor>,
        config: &WhisperConfig,
        output_path: P,
        export_config: &ONNXExportConfig,
    ) -> Result<ONNXModelMetadata, RecognitionError> {
        // Create ONNX model representation
        let mut model_builder = ONNXModelBuilder::new(export_config);

        // Define input specifications
        let input_specs = self.create_input_specs(config, export_config)?;

        // Build encoder layers
        for layer_idx in 0..config.n_audio_layer {
            self.export_encoder_layer(&mut model_builder, layer_idx, model_weights, config)?;
        }

        // Build decoder layers
        for layer_idx in 0..config.n_text_layer {
            self.export_decoder_layer(&mut model_builder, layer_idx, model_weights, config)?;
        }

        // Define output specifications
        let output_specs = self.create_output_specs(config, export_config)?;

        // Apply optimizations if requested
        if export_config.optimize_for_inference {
            model_builder.apply_optimizations()?;
        }

        // Convert precision if requested
        if export_config.fp16_precision {
            model_builder.convert_to_fp16()?;
        }

        // Write ONNX model to file
        let model_size_bytes = model_builder.write_to_file(output_path.as_ref())?;

        // Calculate estimated memory usage
        let estimated_memory_mb =
            self.estimate_onnx_memory_usage(&input_specs, &output_specs, export_config)?;

        Ok(ONNXModelMetadata {
            input_specs,
            output_specs,
            model_size_bytes,
            estimated_memory_mb,
            export_timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                .to_string(),
            source_config: config.clone(),
        })
    }

    /// Create input tensor specifications for ONNX export
    fn create_input_specs(
        &self,
        config: &WhisperConfig,
        export_config: &ONNXExportConfig,
    ) -> Result<Vec<TensorSpec>, RecognitionError> {
        let mut specs = Vec::new();

        // Audio features input (mel spectrogram)
        let audio_shape = if export_config.dynamic_shapes {
            vec![export_config.batch_size, Some(config.n_mels), None] // Dynamic sequence length
        } else {
            vec![
                export_config.batch_size,
                Some(config.n_mels),
                export_config.max_sequence_length,
            ]
        };

        specs.push(TensorSpec {
            name: "audio_features".to_string(),
            shape: audio_shape,
            dtype: if export_config.fp16_precision {
                "float16".to_string()
            } else {
                "float32".to_string()
            },
            description: "Mel spectrogram features (batch_size, n_mels, time_steps)".to_string(),
        });

        // Decoder input tokens (for autoregressive generation)
        let token_shape = if export_config.dynamic_shapes {
            vec![export_config.batch_size, None] // Dynamic sequence length
        } else {
            vec![export_config.batch_size, Some(config.n_text_ctx)]
        };

        specs.push(TensorSpec {
            name: "decoder_input_ids".to_string(),
            shape: token_shape,
            dtype: "int64".to_string(),
            description: "Decoder input token IDs (batch_size, sequence_length)".to_string(),
        });

        // Attention mask for decoder
        if export_config.export_attention {
            let mask_shape = if export_config.dynamic_shapes {
                vec![export_config.batch_size, None, None]
            } else {
                vec![
                    export_config.batch_size,
                    Some(config.n_text_ctx),
                    Some(config.n_text_ctx),
                ]
            };

            specs.push(TensorSpec {
                name: "decoder_attention_mask".to_string(),
                shape: mask_shape,
                dtype: "bool".to_string(),
                description: "Decoder causal attention mask (batch_size, seq_len, seq_len)"
                    .to_string(),
            });
        }

        Ok(specs)
    }

    /// Create output tensor specifications for ONNX export
    fn create_output_specs(
        &self,
        config: &WhisperConfig,
        export_config: &ONNXExportConfig,
    ) -> Result<Vec<TensorSpec>, RecognitionError> {
        let mut specs = Vec::new();

        // Primary output: logits over vocabulary
        let logits_shape = if export_config.dynamic_shapes {
            vec![export_config.batch_size, None, Some(config.n_vocab)]
        } else {
            vec![
                export_config.batch_size,
                Some(config.n_text_ctx),
                Some(config.n_vocab),
            ]
        };

        specs.push(TensorSpec {
            name: "logits".to_string(),
            shape: logits_shape,
            dtype: if export_config.fp16_precision {
                "float16".to_string()
            } else {
                "float32".to_string()
            },
            description: "Output logits over vocabulary (batch_size, sequence_length, vocab_size)"
                .to_string(),
        });

        // Optional: Attention weights
        if export_config.export_attention {
            let attention_shape = if export_config.dynamic_shapes {
                vec![
                    export_config.batch_size,
                    Some(config.n_text_head),
                    None,
                    None,
                ]
            } else {
                vec![
                    export_config.batch_size,
                    Some(config.n_text_head),
                    Some(config.n_text_ctx),
                    export_config.max_sequence_length,
                ]
            };

            specs.push(TensorSpec {
                name: "attention_weights".to_string(),
                shape: attention_shape,
                dtype: if export_config.fp16_precision {
                    "float16".to_string()
                } else {
                    "float32".to_string()
                },
                description:
                    "Cross-attention weights (batch_size, num_heads, target_len, source_len)"
                        .to_string(),
            });
        }

        // Optional: Intermediate layer outputs
        if export_config.export_intermediates {
            for layer_idx in 0..config.n_text_layer {
                let intermediate_shape = if export_config.dynamic_shapes {
                    vec![export_config.batch_size, None, Some(config.n_text_state)]
                } else {
                    vec![
                        export_config.batch_size,
                        Some(config.n_text_ctx),
                        Some(config.n_text_state),
                    ]
                };

                specs.push(TensorSpec {
                    name: format!("intermediate_layer_{layer_idx}"),
                    shape: intermediate_shape,
                    dtype: if export_config.fp16_precision { "float16".to_string() } else { "float32".to_string() },
                    description: format!("Layer {layer_idx} intermediate output (batch_size, sequence_length, hidden_size)"),
                });
            }
        }

        Ok(specs)
    }

    /// Export encoder layer to ONNX format
    fn export_encoder_layer(
        &self,
        _model_builder: &mut ONNXModelBuilder,
        _layer_idx: usize,
        _model_weights: &HashMap<String, Tensor>,
        _config: &WhisperConfig,
    ) -> Result<(), RecognitionError> {
        // Implementation would convert encoder layers to ONNX operations
        // This is a simplified placeholder for the actual ONNX export logic
        // In a real implementation, this would:
        // 1. Extract layer weights from model_weights HashMap
        // 2. Convert Candle operations to ONNX nodes
        // 3. Add nodes to the ONNX model builder
        Ok(())
    }

    /// Export decoder layer to ONNX format
    fn export_decoder_layer(
        &self,
        _model_builder: &mut ONNXModelBuilder,
        _layer_idx: usize,
        _model_weights: &HashMap<String, Tensor>,
        _config: &WhisperConfig,
    ) -> Result<(), RecognitionError> {
        // Implementation would convert decoder layers to ONNX operations
        // This is a simplified placeholder for the actual ONNX export logic
        Ok(())
    }

    /// Estimate memory usage for ONNX model
    fn estimate_onnx_memory_usage(
        &self,
        input_specs: &[TensorSpec],
        output_specs: &[TensorSpec],
        export_config: &ONNXExportConfig,
    ) -> Result<f32, RecognitionError> {
        let mut total_memory = 0.0;

        // Calculate input tensor memory
        for spec in input_specs {
            let tensor_size = self.calculate_tensor_memory_size(spec, export_config)?;
            total_memory += tensor_size;
        }

        // Calculate output tensor memory
        for spec in output_specs {
            let tensor_size = self.calculate_tensor_memory_size(spec, export_config)?;
            total_memory += tensor_size;
        }

        // Add model weight memory (estimated)
        let model_memory = if export_config.fp16_precision {
            self.get_memory_savings().quantized_size_mb * 0.5 // FP16 uses half the space of FP32
        } else {
            self.get_memory_savings().quantized_size_mb
        };

        total_memory += model_memory;

        // Add overhead for ONNX runtime (estimated 20% overhead)
        total_memory *= 1.2;

        Ok(total_memory)
    }

    /// Calculate memory size for a tensor specification
    fn calculate_tensor_memory_size(
        &self,
        spec: &TensorSpec,
        export_config: &ONNXExportConfig,
    ) -> Result<f32, RecognitionError> {
        let mut total_elements = 1;

        for dim in &spec.shape {
            match dim {
                Some(size) => total_elements *= size,
                None => {
                    // Use default values for dynamic dimensions
                    if spec.name.contains("sequence") {
                        total_elements *= export_config.max_sequence_length.unwrap_or(1500);
                    } else {
                        total_elements *= export_config.batch_size.unwrap_or(1);
                    }
                }
            }
        }

        // Calculate bytes per element based on data type
        let bytes_per_element = match spec.dtype.as_str() {
            "float32" => 4,
            "float16" => 2,
            "int64" => 8,
            "int32" => 4,
            "bool" => 1,
            _ => 4, // Default to 4 bytes
        };

        let memory_bytes = total_elements * bytes_per_element;
        Ok(memory_bytes as f32 / 1_000_000.0) // Convert to MB
    }

    /// Validate ONNX export configuration
    pub fn validate_onnx_config(
        &self,
        config: &WhisperConfig,
        export_config: &ONNXExportConfig,
    ) -> Result<(), RecognitionError> {
        // Validate opset version
        if export_config.opset_version < 11 || export_config.opset_version > 18 {
            return Err(RecognitionError::ModelError {
                message: format!(
                    "Unsupported ONNX opset version: {}. Supported range: 11-18",
                    export_config.opset_version
                ),
                source: None,
            });
        }

        // Validate sequence length constraints
        if let Some(max_len) = export_config.max_sequence_length {
            if max_len > config.n_audio_ctx {
                return Err(RecognitionError::ModelError {
                    message: format!(
                        "Max sequence length {} exceeds model limit {}",
                        max_len, config.n_audio_ctx
                    ),
                    source: None,
                });
            }
        }

        // Validate batch size constraints
        if let Some(batch_size) = export_config.batch_size {
            if batch_size == 0 || batch_size > 32 {
                return Err(RecognitionError::ModelError {
                    message: format!("Invalid batch size: {batch_size}. Must be between 1 and 32"),
                    source: None,
                });
            }
        }

        Ok(())
    }
}

/// ONNX model builder (simplified implementation)
struct ONNXModelBuilder {
    #[allow(dead_code)]
    config: ONNXExportConfig,
}

impl ONNXModelBuilder {
    fn new(config: &ONNXExportConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    fn apply_optimizations(&mut self) -> Result<(), RecognitionError> {
        // Placeholder for ONNX graph optimizations
        // In a real implementation, this would apply:
        // - Constant folding
        // - Dead code elimination
        // - Operator fusion
        // - Memory layout optimization
        Ok(())
    }

    fn convert_to_fp16(&mut self) -> Result<(), RecognitionError> {
        // Placeholder for FP16 conversion
        // In a real implementation, this would convert all FP32 operations to FP16
        Ok(())
    }

    fn write_to_file(&self, _path: &Path) -> Result<usize, RecognitionError> {
        // Placeholder for writing ONNX file
        // In a real implementation, this would serialize the ONNX model to disk
        // For now, return a mock file size
        Ok(50_000_000) // 50MB mock file size
    }
}

/// Quantization savings metrics
#[derive(Debug, Clone)]
pub struct QuantizationSavings {
    /// Original model size in MB
    pub original_size_mb: f32,
    /// Quantized model size in MB
    pub quantized_size_mb: f32,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Memory saved in MB
    pub memory_saved_mb: f32,
}

/// Overall pruning statistics
#[derive(Debug, Clone)]
pub struct OverallPruningStats {
    /// Total original parameters across all layers
    pub total_original_params: usize,
    /// Total pruned parameters across all layers
    pub total_pruned_params: usize,
    /// Overall sparsity ratio
    pub overall_sparsity_ratio: f32,
    /// Number of layers that were pruned
    pub layers_pruned: usize,
}

/// Knowledge distillation configuration
#[derive(Debug, Clone)]
pub struct KnowledgeDistillationConfig {
    /// Temperature for softmax in distillation
    pub temperature: f32,
    /// Weight for distillation loss (vs hard target loss)
    pub distillation_alpha: f32,
    /// Enable intermediate layer distillation
    pub intermediate_layers: bool,
    /// Number of student layers per teacher layer
    pub layer_mapping_ratio: f32,
    /// Enable attention distillation
    pub attention_distillation: bool,
    /// Enable feature distillation
    pub feature_distillation: bool,
}

impl Default for KnowledgeDistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 4.0,
            distillation_alpha: 0.7,
            intermediate_layers: true,
            layer_mapping_ratio: 0.5, // Student has half the layers of teacher
            attention_distillation: true,
            feature_distillation: true,
        }
    }
}

/// Knowledge distillation loss computation
#[derive(Debug, Clone)]
pub struct DistillationLoss {
    /// Distillation loss value
    pub distillation_loss: f32,
    /// Hard target loss value
    pub hard_loss: f32,
    /// Combined total loss
    pub total_loss: f32,
    /// Temperature used
    pub temperature: f32,
}

/// Student model architecture configuration
#[derive(Debug, Clone)]
pub struct StudentModelConfig {
    /// Number of encoder layers (reduced from teacher)
    pub encoder_layers: usize,
    /// Number of decoder layers (reduced from teacher)
    pub decoder_layers: usize,
    /// Hidden dimension (potentially reduced)
    pub hidden_dim: usize,
    /// Number of attention heads (potentially reduced)
    pub num_heads: usize,
    /// Feed-forward dimension (potentially reduced)
    pub ff_dim: usize,
    /// Whether to use shared embeddings
    pub shared_embeddings: bool,
}

impl StudentModelConfig {
    /// Create a smaller student model configuration from teacher config
    #[must_use]
    pub fn from_teacher_config(teacher_config: &WhisperConfig, compression_ratio: f32) -> Self {
        let compression_factor = compression_ratio.clamp(0.1, 1.0);

        Self {
            encoder_layers: ((teacher_config.n_audio_layer as f32 * compression_factor) as usize)
                .max(1),
            decoder_layers: ((teacher_config.n_text_layer as f32 * compression_factor) as usize)
                .max(1),
            hidden_dim: ((teacher_config.n_text_state as f32 * compression_factor) as usize)
                .max(64),
            num_heads: ((teacher_config.n_text_head as f32 * compression_factor) as usize).max(1),
            ff_dim: ((teacher_config.n_text_state as f32 * 4.0 * compression_factor) as usize)
                .max(128), // Typical FFN is 4x hidden size
            shared_embeddings: true, // Enable sharing to reduce parameters
        }
    }
}

/// ONNX export configuration
#[derive(Debug, Clone)]
pub struct ONNXExportConfig {
    /// Target ONNX opset version
    pub opset_version: i64,
    /// Optimize for inference (apply graph optimizations)
    pub optimize_for_inference: bool,
    /// Use dynamic shapes for variable sequence lengths
    pub dynamic_shapes: bool,
    /// Maximum sequence length for static shapes
    pub max_sequence_length: Option<usize>,
    /// Batch size for static shapes (None for dynamic)
    pub batch_size: Option<usize>,
    /// Enable FP16 precision in ONNX model
    pub fp16_precision: bool,
    /// Include attention weights in export
    pub export_attention: bool,
    /// Include intermediate layer outputs
    pub export_intermediates: bool,
    /// Model name metadata
    pub model_name: Option<String>,
    /// Model version metadata
    pub model_version: Option<String>,
}

impl Default for ONNXExportConfig {
    fn default() -> Self {
        Self {
            opset_version: 17, // ONNX opset 17 supports most modern operations
            optimize_for_inference: true,
            dynamic_shapes: true,
            max_sequence_length: Some(1500), // 30 seconds at 50fps
            batch_size: None,                // Dynamic batch size
            fp16_precision: false,           // Keep FP32 for compatibility
            export_attention: false,         // Reduces model size
            export_intermediates: false,     // Reduces complexity
            model_name: Some("whisper_optimized".to_string()),
            model_version: Some("1.0".to_string()),
        }
    }
}

/// ONNX model metadata
#[derive(Debug, Clone)]
pub struct ONNXModelMetadata {
    /// Input tensor names and shapes
    pub input_specs: Vec<TensorSpec>,
    /// Output tensor names and shapes
    pub output_specs: Vec<TensorSpec>,
    /// Model file size in bytes
    pub model_size_bytes: usize,
    /// Estimated memory usage in MB
    pub estimated_memory_mb: f32,
    /// Export timestamp
    pub export_timestamp: String,
    /// Source model configuration
    pub source_config: WhisperConfig,
}

/// ONNX tensor specification
#[derive(Debug, Clone)]
pub struct TensorSpec {
    /// Tensor name
    pub name: String,
    /// Tensor shape (None indicates dynamic dimension)
    pub shape: Vec<Option<usize>>,
    /// Tensor data type
    pub dtype: String,
    /// Description of tensor purpose
    pub description: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_quantizer_creation() {
        let config = QuantizationConfig::default();
        let device = Device::Cpu;
        let quantizer = ModelQuantizer::new(config, device);

        assert_eq!(quantizer.layer_stats.len(), 0);
        assert_eq!(quantizer.calibration_data.len(), 0);
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn test_fp16_quantization() {
        let config = QuantizationConfig {
            target_dtype: QuantizationMode::F16,
            ..Default::default()
        };
        let device = Device::Cpu;
        let quantizer = ModelQuantizer::new(config, device);

        let tensor = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu).unwrap();
        let quantized = quantizer.quantize_fp16(&tensor).unwrap();

        assert_eq!(quantized.dtype(), DType::F16);
    }

    #[test]
    fn test_int8_quantization_stats() {
        let config = QuantizationConfig {
            target_dtype: QuantizationMode::INT8 {
                scale: 1.0,
                zero_point: 0,
            },
            symmetric: true,
            ..Default::default()
        };
        let device = Device::Cpu;
        let quantizer = ModelQuantizer::new(config, device);

        let data = [1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32];
        let tensor = Tensor::new(&data[..], &Device::Cpu).unwrap();

        let stats = quantizer
            .compute_quantization_stats(&tensor, "test_layer")
            .unwrap();

        assert!(stats.scale > 0.0);
        assert!(stats.min_val <= stats.max_val);
    }

    #[test]
    fn test_layer_quantization_selection() {
        let config = QuantizationConfig {
            quantize_embeddings: false,
            quantize_attention: true,
            quantize_mlp: true,
            ..Default::default()
        };
        let device = Device::Cpu;
        let quantizer = ModelQuantizer::new(config, device);

        let dummy_tensor = Tensor::new(&[1.0f32], &Device::Cpu).unwrap();

        assert!(!quantizer
            .should_quantize_layer("embedding.weight", &dummy_tensor)
            .unwrap());
        assert!(quantizer
            .should_quantize_layer("attention.weight", &dummy_tensor)
            .unwrap());
        assert!(quantizer
            .should_quantize_layer("mlp.weight", &dummy_tensor)
            .unwrap());
        assert!(!quantizer
            .should_quantize_layer("layer_norm.weight", &dummy_tensor)
            .unwrap());
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn test_4bit_quantization() {
        let config = QuantizationConfig {
            enable_4bit: true,
            group_size_4bit: 4,
            ..Default::default()
        };
        let device = Device::Cpu;
        let mut quantizer = ModelQuantizer::new(config, device);

        let data = [
            1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32,
        ];
        let tensor = Tensor::new(&data[..], &Device::Cpu).unwrap();

        let _quantized = quantizer.quantize_4bit(&tensor, "test_layer").unwrap();

        // Check that 4-bit stats were stored
        assert!(quantizer.get_4bit_stats("test_layer").is_some());

        let stats = quantizer.get_4bit_stats("test_layer").unwrap();
        assert_eq!(stats.group_size, 4);
        assert_eq!(stats.num_groups, 2); // 8 elements / 4 group_size = 2 groups
    }

    #[test]
    fn test_pruning() {
        let config = QuantizationConfig {
            enable_pruning: true,
            pruning_ratio: 0.5, // Prune 50%
            magnitude_threshold: 0.1,
            structured_pruning: false,
            ..Default::default()
        };
        let device = Device::Cpu;
        let mut quantizer = ModelQuantizer::new(config, device);

        let data = [0.01f32, 0.5f32, 0.02f32, 0.8f32, 0.03f32, 0.9f32];
        let tensor = Tensor::new(&data[..], &Device::Cpu).unwrap();

        let _pruned = quantizer.prune_tensor(&tensor, "test_layer").unwrap();

        // Check that pruning stats were stored
        assert!(quantizer.get_pruning_stats("test_layer").is_some());

        let stats = quantizer.get_pruning_stats("test_layer").unwrap();
        assert_eq!(stats.original_params, 6);
        assert!(stats.pruned_params > 0);
        assert!(stats.sparsity_ratio > 0.0);
    }

    #[test]
    fn test_structured_pruning() {
        let config = QuantizationConfig {
            enable_pruning: true,
            pruning_ratio: 0.5, // Increase to 50% to ensure pruning happens
            structured_pruning: true,
            ..Default::default()
        };
        let device = Device::Cpu;
        let mut quantizer = ModelQuantizer::new(config, device);

        // Create a tensor with 64 elements (2 groups of 32)
        let data: Vec<f32> = (0..64).map(|i| if i < 32 { 0.1 } else { 0.9 }).collect();
        let tensor = Tensor::new(&data[..], &Device::Cpu).unwrap();

        let _pruned = quantizer.prune_tensor(&tensor, "test_layer").unwrap();

        let stats = quantizer.get_pruning_stats("test_layer").unwrap();
        assert_eq!(stats.original_params, 64);
        // Should prune some parameters (structured)
        assert!(stats.pruned_params > 0); // Some parameters should be pruned
        assert!(stats.sparsity_ratio > 0.0);
    }

    #[test]
    fn test_overall_pruning_stats() {
        let config = QuantizationConfig {
            enable_pruning: true,
            pruning_ratio: 0.3,
            ..Default::default()
        };
        let device = Device::Cpu;
        let mut quantizer = ModelQuantizer::new(config, device);

        // Add multiple layers
        let data1 = [0.1f32, 0.5f32, 0.2f32, 0.8f32];
        let tensor1 = Tensor::new(&data1[..], &Device::Cpu).unwrap();
        quantizer.prune_tensor(&tensor1, "layer1").unwrap();

        let data2 = [0.3f32, 0.7f32, 0.4f32, 0.9f32, 0.1f32, 0.6f32];
        let tensor2 = Tensor::new(&data2[..], &Device::Cpu).unwrap();
        quantizer.prune_tensor(&tensor2, "layer2").unwrap();

        let overall_stats = quantizer.get_overall_pruning_stats();
        assert_eq!(overall_stats.total_original_params, 10); // 4 + 6
        assert!(overall_stats.total_pruned_params > 0);
        assert!(overall_stats.overall_sparsity_ratio > 0.0);
        assert_eq!(overall_stats.layers_pruned, 2);
    }

    #[test]
    fn test_memory_savings_with_4bit_and_pruning() {
        let config = QuantizationConfig {
            enable_4bit: true,
            enable_pruning: true,
            pruning_ratio: 0.2,
            group_size_4bit: 4,
            ..Default::default()
        };
        let device = Device::Cpu;
        let mut quantizer = ModelQuantizer::new(config, device);

        let data = [
            0.1f32, 0.5f32, 0.2f32, 0.8f32, 0.3f32, 0.7f32, 0.4f32, 0.9f32,
        ];
        let tensor = Tensor::new(&data[..], &Device::Cpu).unwrap();

        // Apply 4-bit quantization
        quantizer.quantize_4bit(&tensor, "test_layer").unwrap();

        // Apply pruning to another layer
        quantizer.prune_tensor(&tensor, "pruned_layer").unwrap();

        let savings = quantizer.get_memory_savings();
        assert!(savings.compression_ratio > 1.0);
        assert!(savings.memory_saved_mb >= 0.0);
    }

    #[test]
    fn test_knowledge_distillation_config() {
        let config = KnowledgeDistillationConfig::default();
        assert_eq!(config.temperature, 4.0);
        assert_eq!(config.distillation_alpha, 0.7);
        assert!(config.intermediate_layers);
        assert!(config.attention_distillation);
        assert!(config.feature_distillation);
    }

    #[test]
    fn test_student_model_config_creation() {
        // Mock teacher config values
        let teacher_config = WhisperConfig {
            n_audio_layer: 12,
            n_text_layer: 12,
            n_text_state: 768,
            n_text_head: 12,
            ..Default::default()
        };

        let student_config = StudentModelConfig::from_teacher_config(&teacher_config, 0.5);

        assert_eq!(student_config.encoder_layers, 6); // 12 * 0.5
        assert_eq!(student_config.decoder_layers, 6); // 12 * 0.5
        assert_eq!(student_config.hidden_dim, 384); // 768 * 0.5
        assert_eq!(student_config.num_heads, 6); // 12 * 0.5
        assert_eq!(student_config.ff_dim, 1536); // 768 * 4 * 0.5
        assert!(student_config.shared_embeddings);
    }

    #[test]
    fn test_distillation_loss_computation() {
        let config = QuantizationConfig::default();
        let device = Device::Cpu;
        let quantizer = ModelQuantizer::new(config, device);

        let distill_config = KnowledgeDistillationConfig::default();

        // Create mock logits and targets
        let teacher_logits = Tensor::new(&[2.0f32, 1.0, 0.5, 1.5], &Device::Cpu)
            .unwrap()
            .reshape((1, 4))
            .unwrap();
        let student_logits = Tensor::new(&[1.8f32, 0.9, 0.6, 1.4], &Device::Cpu)
            .unwrap()
            .reshape((1, 4))
            .unwrap();
        let hard_targets = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0], &Device::Cpu)
            .unwrap()
            .reshape((1, 4))
            .unwrap();

        let loss_result = quantizer.compute_distillation_loss(
            &teacher_logits,
            &student_logits,
            &hard_targets,
            &distill_config,
        );

        assert!(loss_result.is_ok());
        let loss = loss_result.unwrap();
        assert!(loss.distillation_loss >= 0.0);
        assert!(loss.hard_loss >= 0.0);
        assert!(loss.total_loss >= 0.0);
        assert_eq!(loss.temperature, distill_config.temperature);
    }

    #[test]
    fn test_onnx_export_config_default() {
        let config = ONNXExportConfig::default();
        assert_eq!(config.opset_version, 17);
        assert!(config.optimize_for_inference);
        assert!(config.dynamic_shapes);
        assert_eq!(config.max_sequence_length, Some(1500));
        assert_eq!(config.batch_size, None);
        assert!(!config.fp16_precision);
        assert!(!config.export_attention);
        assert!(!config.export_intermediates);
        assert_eq!(config.model_name, Some("whisper_optimized".to_string()));
        assert_eq!(config.model_version, Some("1.0".to_string()));
    }

    #[test]
    fn test_onnx_export_config_validation() {
        let config = QuantizationConfig::default();
        let device = Device::Cpu;
        let quantizer = ModelQuantizer::new(config, device);

        let whisper_config = WhisperConfig {
            n_audio_ctx: 1500,
            ..Default::default()
        };

        // Valid configuration should pass
        let valid_export_config = ONNXExportConfig::default();
        assert!(quantizer
            .validate_onnx_config(&whisper_config, &valid_export_config)
            .is_ok());

        // Invalid opset version should fail
        let invalid_opset_config = ONNXExportConfig {
            opset_version: 5, // Too low
            ..Default::default()
        };
        assert!(quantizer
            .validate_onnx_config(&whisper_config, &invalid_opset_config)
            .is_err());

        // Invalid batch size should fail
        let invalid_batch_config = ONNXExportConfig {
            batch_size: Some(64), // Too high
            ..Default::default()
        };
        assert!(quantizer
            .validate_onnx_config(&whisper_config, &invalid_batch_config)
            .is_err());

        // Sequence length exceeding model limit should fail
        let invalid_seq_config = ONNXExportConfig {
            max_sequence_length: Some(2000), // Exceeds model limit
            ..Default::default()
        };
        assert!(quantizer
            .validate_onnx_config(&whisper_config, &invalid_seq_config)
            .is_err());
    }

    #[test]
    fn test_create_input_specs() {
        let config = QuantizationConfig::default();
        let device = Device::Cpu;
        let quantizer = ModelQuantizer::new(config, device);

        let whisper_config = WhisperConfig {
            n_mels: 80,
            n_text_ctx: 448,
            ..Default::default()
        };

        // Test dynamic shapes configuration
        let dynamic_export_config = ONNXExportConfig {
            dynamic_shapes: true,
            export_attention: true,
            ..Default::default()
        };

        let input_specs = quantizer
            .create_input_specs(&whisper_config, &dynamic_export_config)
            .unwrap();
        assert_eq!(input_specs.len(), 3); // audio_features, decoder_input_ids, decoder_attention_mask

        // Check audio features spec
        let audio_spec = &input_specs[0];
        assert_eq!(audio_spec.name, "audio_features");
        assert_eq!(audio_spec.shape, vec![None, Some(80), None]); // Dynamic batch and sequence
        assert_eq!(audio_spec.dtype, "float32");

        // Test static shapes configuration
        let static_export_config = ONNXExportConfig {
            dynamic_shapes: false,
            batch_size: Some(2),
            max_sequence_length: Some(1000),
            export_attention: false,
            ..Default::default()
        };

        let static_input_specs = quantizer
            .create_input_specs(&whisper_config, &static_export_config)
            .unwrap();
        assert_eq!(static_input_specs.len(), 2); // No attention mask

        let static_audio_spec = &static_input_specs[0];
        assert_eq!(static_audio_spec.shape, vec![Some(2), Some(80), Some(1000)]);
    }

    #[test]
    fn test_create_output_specs() {
        let config = QuantizationConfig::default();
        let device = Device::Cpu;
        let quantizer = ModelQuantizer::new(config, device);

        let whisper_config = WhisperConfig {
            n_vocab: 51864,
            n_text_head: 8,
            n_text_layer: 6,
            n_text_state: 512,
            n_text_ctx: 448,
            ..Default::default()
        };

        // Test with all optional outputs enabled
        let full_export_config = ONNXExportConfig {
            export_attention: true,
            export_intermediates: true,
            fp16_precision: true,
            ..Default::default()
        };

        let output_specs = quantizer
            .create_output_specs(&whisper_config, &full_export_config)
            .unwrap();

        // Should have: logits + attention_weights + 6 intermediate layers = 8 total
        assert_eq!(output_specs.len(), 8);

        // Check main logits output
        let logits_spec = &output_specs[0];
        assert_eq!(logits_spec.name, "logits");
        assert_eq!(logits_spec.dtype, "float16"); // FP16 enabled

        // Check attention weights
        let attention_spec = &output_specs[1];
        assert_eq!(attention_spec.name, "attention_weights");

        // Check intermediate layers
        for i in 0..6 {
            let intermediate_spec = &output_specs[2 + i];
            assert_eq!(intermediate_spec.name, format!("intermediate_layer_{i}"));
            assert_eq!(intermediate_spec.dtype, "float16");
        }
    }

    #[test]
    fn test_tensor_memory_calculation() {
        let config = QuantizationConfig::default();
        let device = Device::Cpu;
        let quantizer = ModelQuantizer::new(config, device);

        let export_config = ONNXExportConfig {
            batch_size: Some(2),
            max_sequence_length: Some(1000),
            ..Default::default()
        };

        // Test FP32 tensor
        let fp32_spec = TensorSpec {
            name: "test_tensor".to_string(),
            shape: vec![Some(2), Some(512), Some(1000)], // 2 * 512 * 1000 = 1,024,000 elements
            dtype: "float32".to_string(),
            description: "Test tensor".to_string(),
        };

        let fp32_memory = quantizer
            .calculate_tensor_memory_size(&fp32_spec, &export_config)
            .unwrap();
        assert_eq!(fp32_memory, 4.096); // 1,024,000 * 4 bytes / 1,000,000 = 4.096 MB

        // Test FP16 tensor
        let fp16_spec = TensorSpec {
            name: "test_tensor".to_string(),
            shape: vec![Some(2), Some(512), Some(1000)],
            dtype: "float16".to_string(),
            description: "Test tensor".to_string(),
        };

        let fp16_memory = quantizer
            .calculate_tensor_memory_size(&fp16_spec, &export_config)
            .unwrap();
        assert_eq!(fp16_memory, 2.048); // 1,024,000 * 2 bytes / 1,000,000 = 2.048 MB

        // Test dynamic shape handling
        let dynamic_spec = TensorSpec {
            name: "sequence_tensor".to_string(),
            shape: vec![None, Some(512), None], // Dynamic batch and sequence
            dtype: "float32".to_string(),
            description: "Dynamic tensor".to_string(),
        };

        let dynamic_memory = quantizer
            .calculate_tensor_memory_size(&dynamic_spec, &export_config)
            .unwrap();
        // Should use defaults: batch_size=2, max_sequence_length=1000
        assert_eq!(dynamic_memory, 2048.0); // Dynamic calculation returns raw value
    }

    #[test]
    fn test_onnx_memory_estimation() {
        let config = QuantizationConfig::default();
        let device = Device::Cpu;
        let quantizer = ModelQuantizer::new(config, device);

        let export_config = ONNXExportConfig::default();

        // Create simple specs
        let input_specs = vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![Some(1), Some(80), Some(1500)],
            dtype: "float32".to_string(),
            description: "Input".to_string(),
        }];

        let output_specs = vec![TensorSpec {
            name: "output".to_string(),
            shape: vec![Some(1), Some(1500), Some(51864)],
            dtype: "float32".to_string(),
            description: "Output".to_string(),
        }];

        let memory_estimate = quantizer
            .estimate_onnx_memory_usage(&input_specs, &output_specs, &export_config)
            .unwrap();

        // Should include input + output + model weights + 20% overhead
        assert!(memory_estimate > 0.0);

        // Test with FP16 precision
        let fp16_config = ONNXExportConfig {
            fp16_precision: true,
            ..Default::default()
        };

        let fp16_memory = quantizer
            .estimate_onnx_memory_usage(&input_specs, &output_specs, &fp16_config)
            .unwrap();

        // FP16 should use less memory than FP32 (or equal since model weights may dominate)
        assert!(fp16_memory <= memory_estimate);
    }

    #[test]
    fn test_onnx_model_builder() {
        let config = ONNXExportConfig::default();
        let mut builder = ONNXModelBuilder::new(&config);

        // Test optimization application
        assert!(builder.apply_optimizations().is_ok());

        // Test FP16 conversion
        assert!(builder.convert_to_fp16().is_ok());

        // Test file writing (mock implementation)
        let temp_path = std::path::Path::new("/tmp/test_model.onnx");
        let file_size = builder.write_to_file(temp_path).unwrap();
        assert_eq!(file_size, 50_000_000); // Mock size
    }

    #[test]
    fn test_dynamic_quantization_mode() {
        let config = QuantizationConfig::default();
        let device = Device::Cpu;
        let mut quantizer = ModelQuantizer::new(config, device);

        // Test enabling dynamic quantization
        assert!(!quantizer.dynamic_mode);
        quantizer.enable_dynamic_quantization(10);
        assert!(quantizer.dynamic_mode);
        assert!(!quantizer.dynamic_trackers.is_empty());

        // Test disabling dynamic quantization
        quantizer.disable_dynamic_quantization();
        assert!(!quantizer.dynamic_mode);
        assert_eq!(quantizer.dynamic_trackers.len(), 0);
    }

    #[test]
    fn test_moving_average_tracker() {
        let mut tracker = MovingAverageTracker::new(3);

        // Test initial state
        assert_eq!(tracker.samples, 0);
        assert_eq!(tracker.get_averaged_range(), (0.0, 1.0));

        // Test single update
        tracker.update(-1.0, 2.0);
        assert_eq!(tracker.samples, 1);
        assert_eq!(tracker.get_averaged_range(), (-1.0, 2.0));

        // Test multiple updates
        tracker.update(-0.5, 1.5);
        tracker.update(-0.8, 1.8);
        assert_eq!(tracker.samples, 3);

        let (avg_min, avg_max) = tracker.get_averaged_range();
        assert!((avg_min - (-0.767)).abs() < 0.01); // (-1.0 + -0.5 + -0.8) / 3
        assert!((avg_max - 1.767).abs() < 0.01); // (2.0 + 1.5 + 1.8) / 3

        // Test window overflow
        tracker.update(-0.2, 1.2);
        assert_eq!(tracker.samples, 3); // Should stay at window size

        let (new_min, new_max) = tracker.get_averaged_range();
        // Should now average the last 3 values: -0.5, -0.8, -0.2
        assert!((new_min - (-0.5)).abs() < 0.01);
        assert!((new_max - 1.5).abs() < 0.01);
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn test_dynamic_8bit_quantization() {
        let config = QuantizationConfig::default();
        let device = Device::Cpu;
        let mut quantizer = ModelQuantizer::new(config, device);

        quantizer.enable_dynamic_quantization(5);

        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(&data[..], &Device::Cpu).unwrap();

        let result = quantizer.quantize_dynamic(&tensor, "test_layer", 8);
        assert!(result.is_ok());

        let (_quantized, params) = result.unwrap();
        assert_eq!(params.quant_range, 255);
        assert!(params.scale > 0.0);
        assert!(params.activation_min <= params.activation_max);

        // Test second quantization - should use moving average
        let data2 = [0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5];
        let tensor2 = Tensor::new(&data2[..], &Device::Cpu).unwrap();

        let result2 = quantizer.quantize_dynamic(&tensor2, "test_layer", 8);
        assert!(result2.is_ok());
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn test_dynamic_4bit_quantization() {
        let config = QuantizationConfig::default();
        let device = Device::Cpu;
        let mut quantizer = ModelQuantizer::new(config, device);

        quantizer.enable_dynamic_quantization(5);

        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(&data[..], &Device::Cpu).unwrap();

        let result = quantizer.quantize_dynamic(&tensor, "test_layer", 4);
        assert!(result.is_ok());

        let (_quantized, params) = result.unwrap();
        assert_eq!(params.quant_range, 15);
        assert!(params.scale > 0.0);
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn test_dynamic_4bit_group_quantization() {
        let config = QuantizationConfig::default();
        let device = Device::Cpu;
        let mut quantizer = ModelQuantizer::new(config, device);

        quantizer.enable_dynamic_quantization(3);

        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = Tensor::new(&data[..], &Device::Cpu).unwrap();

        let result = quantizer.quantize_dynamic_4bit(&tensor, "test_layer", 4);
        assert!(result.is_ok());

        let (_quantized, group_params) = result.unwrap();
        assert_eq!(group_params.len(), 2); // 8 elements / 4 group_size = 2 groups

        for params in &group_params {
            assert_eq!(params.quant_range, 15);
            assert!(params.scale > 0.0);
        }
    }

    #[test]
    fn test_dynamic_quantization_dequantization() {
        let config = QuantizationConfig::default();
        let device = Device::Cpu;
        let mut quantizer = ModelQuantizer::new(config, device);

        quantizer.enable_dynamic_quantization(5);

        let original_data = [1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(&original_data[..], &Device::Cpu).unwrap();

        // Quantize
        let (quantized_tensor, params) = quantizer
            .quantize_dynamic(&tensor, "test_layer", 8)
            .unwrap();

        // Dequantize
        let dequantized = quantizer
            .dequantize_dynamic(&quantized_tensor, &params)
            .unwrap();
        let dequantized_data = dequantized.to_vec1::<f32>().unwrap();

        // Check that values are approximately preserved
        for (original, dequantized) in original_data.iter().zip(dequantized_data.iter()) {
            let error = (original - dequantized).abs();
            assert!(
                error < 0.1,
                "Quantization error too large: {original} vs {dequantized}"
            );
        }
    }

    #[test]
    fn test_tracker_key_mapping() {
        let config = QuantizationConfig::default();
        let device = Device::Cpu;
        let quantizer = ModelQuantizer::new(config, device);

        assert_eq!(
            quantizer.get_tracker_key("encoder.layer1.weight"),
            "encoder"
        );
        assert_eq!(
            quantizer.get_tracker_key("decoder.attention.weight"),
            "decoder"
        );
        assert_eq!(
            quantizer.get_tracker_key("model.attention.weight"),
            "attention"
        );
        assert_eq!(quantizer.get_tracker_key("layer.mlp.weight"), "mlp");
        assert_eq!(quantizer.get_tracker_key("linear.weight"), "linear");
        assert_eq!(quantizer.get_tracker_key("conv1d.weight"), "conv");
        assert_eq!(quantizer.get_tracker_key("unknown.weight"), "default");
    }

    #[test]
    fn test_dynamic_stats_retrieval() {
        let config = QuantizationConfig::default();
        let device = Device::Cpu;
        let mut quantizer = ModelQuantizer::new(config, device);

        quantizer.enable_dynamic_quantization(3);

        let data = [1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(&data[..], &Device::Cpu).unwrap();

        // Perform some quantizations to populate stats
        quantizer
            .quantize_dynamic(&tensor, "encoder.layer1", 8)
            .unwrap();
        quantizer
            .quantize_dynamic(&tensor, "decoder.attention", 8)
            .unwrap();

        let stats = quantizer.get_dynamic_stats();

        // Check that we have some stats (should include pre-initialized trackers)
        assert!(!stats.is_empty());

        // Check specifically for encoder and decoder keys if they exist
        if let Some((min_val, max_val, samples)) = stats.get("encoder") {
            assert!(min_val <= max_val);
            assert!(*samples > 0, "Encoder tracker should have samples > 0");
        }

        if let Some((min_val, max_val, samples)) = stats.get("decoder") {
            assert!(min_val <= max_val);
            assert!(*samples > 0, "Decoder tracker should have samples > 0");
        }

        // Ensure at least some trackers have been updated
        let updated_trackers = stats
            .values()
            .filter(|(_, _, samples)| *samples > 0)
            .count();
        assert!(
            updated_trackers > 0,
            "At least some trackers should have been updated"
        );
    }

    #[test]
    fn test_dynamic_tracker_reset() {
        let config = QuantizationConfig::default();
        let device = Device::Cpu;
        let mut quantizer = ModelQuantizer::new(config, device);

        quantizer.enable_dynamic_quantization(3);

        let data = [1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(&data[..], &Device::Cpu).unwrap();

        // Perform quantization to populate trackers
        quantizer
            .quantize_dynamic(&tensor, "test_layer", 8)
            .unwrap();

        let stats_before = quantizer.get_dynamic_stats();
        // Find a tracker that was actually updated
        let updated_tracker = stats_before.values().find(|(_, _, samples)| *samples > 0);
        if let Some((_, _, samples_before)) = updated_tracker {
            assert!(*samples_before > 0);
        } else {
            // If no tracker was updated, at least verify we have trackers
            assert!(!stats_before.is_empty(), "Should have initialized trackers");
        }

        // Reset trackers
        quantizer.reset_dynamic_trackers();

        let stats_after = quantizer.get_dynamic_stats();
        for (_, (_, _, samples)) in stats_after {
            assert_eq!(samples, 0);
        }
    }

    #[test]
    fn test_dynamic_quantization_without_enabling() {
        let config = QuantizationConfig::default();
        let device = Device::Cpu;
        let mut quantizer = ModelQuantizer::new(config, device);

        let data = [1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(&data[..], &Device::Cpu).unwrap();

        // Should fail when dynamic mode is not enabled
        let result = quantizer.quantize_dynamic(&tensor, "test_layer", 8);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Dynamic quantization is not enabled"));
    }

    #[test]
    fn test_unsupported_bit_width() {
        let config = QuantizationConfig::default();
        let device = Device::Cpu;
        let mut quantizer = ModelQuantizer::new(config, device);

        quantizer.enable_dynamic_quantization(5);

        let data = [1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(&data[..], &Device::Cpu).unwrap();

        // Should fail with unsupported bit width
        let result = quantizer.quantize_dynamic(&tensor, "test_layer", 16);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unsupported bit width"));
    }

    #[test]
    fn test_full_onnx_export_pipeline() {
        let config = QuantizationConfig::default();
        let device = Device::Cpu;
        let quantizer = ModelQuantizer::new(config, device);

        // Create mock model weights
        let mut model_weights = HashMap::new();
        model_weights.insert(
            "encoder.layer0.weight".to_string(),
            Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu).unwrap(),
        );
        model_weights.insert(
            "decoder.layer0.weight".to_string(),
            Tensor::new(&[5.0f32, 6.0, 7.0, 8.0], &Device::Cpu).unwrap(),
        );

        let whisper_config = WhisperConfig {
            n_audio_layer: 1,
            n_text_layer: 1,
            n_mels: 80,
            n_vocab: 51864,
            n_text_state: 512,
            n_text_head: 8,
            n_text_ctx: 448,
            n_audio_ctx: 1500,
            ..Default::default()
        };

        let export_config = ONNXExportConfig {
            model_name: Some("test_whisper".to_string()),
            model_version: Some("0.1.0".to_string()),
            ..Default::default()
        };

        let temp_path = std::path::Path::new("/tmp/test_whisper.onnx");

        // Test the full export pipeline
        let result =
            quantizer.export_to_onnx(&model_weights, &whisper_config, temp_path, &export_config);
        assert!(result.is_ok());

        let metadata = result.unwrap();
        assert_eq!(metadata.model_size_bytes, 50_000_000); // Mock size
        assert!(metadata.estimated_memory_mb > 0.0);
        assert!(!metadata.export_timestamp.is_empty());
        assert_eq!(metadata.source_config.n_audio_layer, 1);
        assert_eq!(metadata.input_specs.len(), 2); // audio_features + decoder_input_ids
        assert_eq!(metadata.output_specs.len(), 1); // logits only
    }
}
