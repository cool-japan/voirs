//! Quantization-aware training (QAT) implementation
//!
//! This module provides quantization-aware training techniques that simulate
//! quantization during training to achieve better accuracy after quantization.

use std::collections::HashMap;

use super::{QuantizationConfig, QuantizationParams, QuantizationPrecision};
use crate::{AcousticError, Result};

/// Quantization-aware training configuration
#[derive(Debug, Clone)]
pub struct QatConfig {
    /// Base quantization configuration
    pub base: QuantizationConfig,
    /// Learning rate for quantization parameters
    pub quant_lr: f32,
    /// Weight decay for quantization parameters
    pub quant_weight_decay: f32,
    /// Number of warmup epochs before enabling quantization
    pub warmup_epochs: u32,
    /// Gradual quantization schedule
    pub gradual_quantization: bool,
    /// Fake quantization noise factor
    pub noise_factor: f32,
}

impl Default for QatConfig {
    fn default() -> Self {
        Self {
            base: QuantizationConfig::default(),
            quant_lr: 0.001,
            quant_weight_decay: 0.0001,
            warmup_epochs: 5,
            gradual_quantization: true,
            noise_factor: 0.01,
        }
    }
}

/// Fake quantization layer for training
pub struct FakeQuantizeLayer {
    /// Quantization parameters
    params: QuantizationParams,
    /// Whether quantization is enabled
    enabled: bool,
    /// Gradient scaling factor
    gradient_scale: f32,
    /// Noise factor for stochastic quantization
    noise_factor: f32,
}

impl FakeQuantizeLayer {
    /// Create new fake quantization layer
    pub fn new(params: QuantizationParams, noise_factor: f32) -> Self {
        Self {
            params,
            enabled: true,
            gradient_scale: 1.0,
            noise_factor,
        }
    }

    /// Forward pass with fake quantization
    pub fn forward(&self, input: &[f32], training: bool) -> Vec<f32> {
        if !self.enabled {
            return input.to_vec();
        }

        let mut output = Vec::with_capacity(input.len());

        for &value in input {
            let quantized = if training && self.noise_factor > 0.0 {
                // Add stochastic noise during training
                let noise = (fastrand::f32() - 0.5) * self.noise_factor;
                self.params.quantize(value + noise)
            } else {
                self.params.quantize(value)
            };

            let dequantized = self.params.dequantize(quantized);
            output.push(dequantized);
        }

        output
    }

    /// Backward pass (simplified - in practice would compute gradients)
    pub fn backward(&mut self, grad_output: &[f32]) -> Vec<f32> {
        // Straight-through estimator: pass gradients through unchanged
        // In practice, you would implement proper gradient computation
        grad_output
            .iter()
            .map(|&g| g * self.gradient_scale)
            .collect()
    }

    /// Enable/disable quantization
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Update quantization parameters
    pub fn update_params(&mut self, new_params: QuantizationParams) {
        self.params = new_params;
    }

    /// Get current parameters
    pub fn params(&self) -> &QuantizationParams {
        &self.params
    }
}

/// Quantization-aware trainer
pub struct QuantizationAwareTrainer {
    /// Configuration
    config: QatConfig,
    /// Fake quantization layers
    fake_quant_layers: HashMap<String, FakeQuantizeLayer>,
    /// Current epoch
    current_epoch: u32,
    /// Quantization schedule
    quantization_schedule: QuantizationSchedule,
}

/// Quantization schedule for gradual QAT
#[derive(Debug, Clone)]
pub struct QuantizationSchedule {
    /// Total training epochs
    total_epochs: u32,
    /// Warmup epochs
    warmup_epochs: u32,
    /// Current quantization factor (0.0 to 1.0)
    current_factor: f32,
}

impl QuantizationSchedule {
    /// Create new quantization schedule
    pub fn new(total_epochs: u32, warmup_epochs: u32) -> Self {
        Self {
            total_epochs,
            warmup_epochs,
            current_factor: 0.0,
        }
    }

    /// Update schedule for current epoch
    pub fn update(&mut self, epoch: u32) {
        if epoch < self.warmup_epochs {
            self.current_factor = 0.0;
        } else if epoch >= self.total_epochs {
            self.current_factor = 1.0;
        } else {
            // Linear ramp-up from warmup to full quantization
            let ramp_epochs = self.total_epochs - self.warmup_epochs;
            let ramp_progress = (epoch - self.warmup_epochs) as f32 / ramp_epochs as f32;
            self.current_factor = ramp_progress.min(1.0);
        }
    }

    /// Get current quantization factor
    pub fn factor(&self) -> f32 {
        self.current_factor
    }
}

impl QuantizationAwareTrainer {
    /// Create new QAT trainer
    pub fn new(config: QatConfig, total_epochs: u32) -> Self {
        let schedule = QuantizationSchedule::new(total_epochs, config.warmup_epochs);

        Self {
            config,
            fake_quant_layers: HashMap::new(),
            current_epoch: 0,
            quantization_schedule: schedule,
        }
    }

    /// Add layer for quantization-aware training
    pub fn add_layer(&mut self, layer_name: String, params: QuantizationParams) {
        let fake_quant = FakeQuantizeLayer::new(params, self.config.noise_factor);
        self.fake_quant_layers.insert(layer_name, fake_quant);
    }

    /// Forward pass through fake quantization
    pub fn forward(&self, layer_name: &str, input: &[f32], training: bool) -> Result<Vec<f32>> {
        let layer = self.fake_quant_layers.get(layer_name).ok_or_else(|| {
            AcousticError::Processing(format!("No fake quantization layer found: {layer_name}"))
        })?;

        Ok(layer.forward(input, training))
    }

    /// Update epoch and quantization schedule
    pub fn set_epoch(&mut self, epoch: u32) {
        self.current_epoch = epoch;
        self.quantization_schedule.update(epoch);

        // Update quantization factor for all layers
        let factor = self.quantization_schedule.factor();
        for layer in self.fake_quant_layers.values_mut() {
            layer.set_enabled(factor > 0.0);
            layer.gradient_scale = factor;
        }
    }

    /// Get current quantization factor
    pub fn quantization_factor(&self) -> f32 {
        self.quantization_schedule.factor()
    }

    /// Calibrate quantization parameters from training data
    pub fn calibrate_from_training(&mut self, layer_name: &str, activations: &[f32]) -> Result<()> {
        let (qmin, qmax) = self.get_quantization_range();

        if activations.is_empty() {
            return Err(AcousticError::Processing(
                "Cannot calibrate from empty activations".to_string(),
            ));
        }

        let min_val = activations.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = activations
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        let new_params = if self.config.base.symmetric {
            let abs_max = max_val.abs().max(min_val.abs());
            let scale = abs_max / (qmax as f32);
            QuantizationParams::symmetric(scale, qmin, qmax)
        } else {
            let scale = (max_val - min_val) / (qmax - qmin) as f32;
            let zero_point = qmin - (min_val / scale).round() as i32;
            QuantizationParams::asymmetric(scale, zero_point, qmin, qmax)
        };

        if let Some(layer) = self.fake_quant_layers.get_mut(layer_name) {
            layer.update_params(new_params);
        }

        Ok(())
    }

    /// Get quantization range for current precision
    fn get_quantization_range(&self) -> (i32, i32) {
        match self.config.base.precision {
            QuantizationPrecision::Int8 => (-128, 127),
            QuantizationPrecision::Int16 => (-32768, 32767),
            QuantizationPrecision::Int4 => (-8, 7),
            QuantizationPrecision::Mixed => (-128, 127),
        }
    }

    /// Export final quantization parameters
    pub fn export_quantization_params(&self) -> HashMap<String, QuantizationParams> {
        self.fake_quant_layers
            .iter()
            .map(|(name, layer)| (name.clone(), layer.params().clone()))
            .collect()
    }

    /// Get training progress
    pub fn training_progress(&self) -> f32 {
        if self.quantization_schedule.total_epochs == 0 {
            1.0
        } else {
            self.current_epoch as f32 / self.quantization_schedule.total_epochs as f32
        }
    }

    /// Calculate quantization loss (regularization term)
    pub fn quantization_loss(&self, layer_outputs: &HashMap<String, Vec<f32>>) -> f32 {
        let mut total_loss = 0.0;
        let mut count = 0;

        for (layer_name, original_output) in layer_outputs {
            if let Some(layer) = self.fake_quant_layers.get(layer_name) {
                let quantized_output = layer.forward(original_output, false);

                // MSE between original and quantized outputs
                let mse: f32 = original_output
                    .iter()
                    .zip(quantized_output.iter())
                    .map(|(&orig, &quant)| (orig - quant).powi(2))
                    .sum::<f32>()
                    / original_output.len() as f32;

                total_loss += mse;
                count += 1;
            }
        }

        if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        }
    }
}

/// Mixed precision training helper
pub struct MixedPrecisionTrainer {
    /// Layers to quantize
    quantized_layers: Vec<String>,
    /// Layers to keep in FP32
    #[allow(dead_code)]
    fp32_layers: Vec<String>,
    /// QAT trainer for quantized layers
    qat_trainer: QuantizationAwareTrainer,
}

impl MixedPrecisionTrainer {
    /// Create new mixed precision trainer
    pub fn new(
        quantized_layers: Vec<String>,
        fp32_layers: Vec<String>,
        config: QatConfig,
        total_epochs: u32,
    ) -> Self {
        let qat_trainer = QuantizationAwareTrainer::new(config, total_epochs);

        Self {
            quantized_layers,
            fp32_layers,
            qat_trainer,
        }
    }

    /// Process layer based on precision configuration
    pub fn process_layer(
        &self,
        layer_name: &str,
        input: &[f32],
        training: bool,
    ) -> Result<Vec<f32>> {
        if self.quantized_layers.contains(&layer_name.to_string()) {
            self.qat_trainer.forward(layer_name, input, training)
        } else {
            // Keep in FP32
            Ok(input.to_vec())
        }
    }

    /// Update training epoch
    pub fn set_epoch(&mut self, epoch: u32) {
        self.qat_trainer.set_epoch(epoch);
    }

    /// Get QAT trainer reference
    pub fn qat_trainer(&mut self) -> &mut QuantizationAwareTrainer {
        &mut self.qat_trainer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fake_quantize_layer() {
        let params = QuantizationParams::symmetric(0.1, -128, 127);
        let layer = FakeQuantizeLayer::new(params, 0.01);

        let input = vec![1.0, 2.0, 3.0];
        let output = layer.forward(&input, false);

        assert_eq!(output.len(), input.len());
        // Values should be quantized and dequantized
        for (i, &orig) in input.iter().enumerate() {
            assert!((output[i] - orig).abs() < 0.2); // Some quantization error
        }
    }

    #[test]
    fn test_quantization_schedule() {
        let mut schedule = QuantizationSchedule::new(100, 10);

        // Warmup phase
        schedule.update(5);
        assert_eq!(schedule.factor(), 0.0);

        // Ramp-up phase
        schedule.update(55); // Halfway through ramp
        assert!((schedule.factor() - 0.5).abs() < 0.01);

        // Full quantization
        schedule.update(100);
        assert_eq!(schedule.factor(), 1.0);
    }

    #[test]
    fn test_qat_trainer() {
        let config = QatConfig::default();
        let mut trainer = QuantizationAwareTrainer::new(config, 100);

        let params = QuantizationParams::symmetric(0.1, -128, 127);
        trainer.add_layer("layer1".to_string(), params);

        // Test forward pass
        let input = vec![1.0, 2.0, 3.0];
        let output = trainer.forward("layer1", &input, true).unwrap();
        assert_eq!(output.len(), input.len());

        // Test epoch update
        trainer.set_epoch(50);
        assert!(trainer.quantization_factor() > 0.0);
    }

    #[test]
    fn test_mixed_precision_trainer() {
        let quantized = vec!["layer1".to_string()];
        let fp32 = vec!["output".to_string()];
        let config = QatConfig::default();

        let mut trainer = MixedPrecisionTrainer::new(quantized, fp32, config, 100);

        // Add quantization parameters
        let params = QuantizationParams::symmetric(0.1, -128, 127);
        trainer
            .qat_trainer()
            .add_layer("layer1".to_string(), params);

        let input = vec![1.0, 2.0, 3.0];

        // Quantized layer
        let output1 = trainer.process_layer("layer1", &input, true).unwrap();
        assert_eq!(output1.len(), input.len());

        // FP32 layer
        let output2 = trainer.process_layer("output", &input, true).unwrap();
        assert_eq!(output2, input); // Should be unchanged
    }

    #[test]
    fn test_quantization_loss() {
        let config = QatConfig::default();
        let mut trainer = QuantizationAwareTrainer::new(config, 100);

        let params = QuantizationParams::symmetric(0.1, -128, 127);
        trainer.add_layer("layer1".to_string(), params);

        let mut layer_outputs = HashMap::new();
        layer_outputs.insert("layer1".to_string(), vec![1.0, 2.0, 3.0]);

        let loss = trainer.quantization_loss(&layer_outputs);
        assert!(loss >= 0.0); // Loss should be non-negative
    }
}
