//! Quantization configuration types
//!
//! This module contains all configuration structures for quantization,
//! knowledge distillation, ONNX export, and related settings.

use super::super::encoder::QuantizationMode;

/// Quantization configuration
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Target data type
    pub target_dtype: QuantizationMode,
    /// Calibration samples for INT8 quantization
    pub calibration_samples: usize,
    /// Percentile for outlier detection
    pub outlier_percentile: f32,
    /// Symmetric quantization
    pub symmetric: bool,
    /// Per-channel quantization
    pub per_channel: bool,
    /// Quantize embeddings
    pub quantize_embeddings: bool,
    /// Quantize attention weights
    pub quantize_attention: bool,
    /// Quantize MLP weights  
    pub quantize_mlp: bool,
    /// Enable 4-bit quantization
    pub enable_4bit: bool,
    /// Group size for 4-bit quantization
    pub group_size_4bit: usize,
    /// Enable model pruning
    pub enable_pruning: bool,
    /// Pruning sparsity ratio (0.0 to 1.0)
    pub pruning_ratio: f32,
    /// Magnitude-based pruning threshold
    pub magnitude_threshold: f32,
    /// Enable structured pruning (vs unstructured)
    pub structured_pruning: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            target_dtype: QuantizationMode::F16,
            calibration_samples: 100,
            outlier_percentile: 99.9,
            symmetric: true,
            per_channel: false,
            quantize_embeddings: true,
            quantize_attention: true,
            quantize_mlp: true,
            enable_4bit: false,
            group_size_4bit: 128,
            enable_pruning: false,
            pruning_ratio: 0.1,
            magnitude_threshold: 0.01,
            structured_pruning: false,
        }
    }
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
            batch_size: None,                // Dynamic batching
            fp16_precision: false,           // Keep FP32 for broader compatibility
            export_attention: false,         // Reduces model size
            export_intermediates: false,     // Reduces complexity
            model_name: Some("whisper-quantized".to_string()),
            model_version: Some("1.0".to_string()),
        }
    }
}

/// Student model architecture configuration
#[derive(Debug, Clone)]
pub struct StudentModelConfig {
    /// Number of encoder layers
    pub n_encoder_layers: usize,
    /// Number of decoder layers
    pub n_decoder_layers: usize,
    /// Model dimension
    pub n_state: usize,
    /// Feed-forward dimension
    pub ff_dim: usize,
    /// Number of attention heads
    pub n_head: usize,
    /// Share embeddings between encoder and decoder
    pub shared_embeddings: bool,
}

impl StudentModelConfig {
    /// Create student config from teacher config with compression ratio
    pub fn from_teacher_config(
        teacher_n_layers: usize,
        teacher_n_state: usize,
        teacher_ff_dim: usize,
        teacher_n_head: usize,
        compression_ratio: f32,
    ) -> Self {
        let student_layers = ((teacher_n_layers as f32 * compression_ratio) as usize).max(1);
        let student_state = ((teacher_n_state as f32 * compression_ratio) as usize).max(64);
        let student_ff_dim = ((teacher_ff_dim as f32 * compression_ratio) as usize).max(256);
        let student_heads = ((teacher_n_head as f32 * compression_ratio) as usize).max(1);

        Self {
            n_encoder_layers: student_layers,
            n_decoder_layers: student_layers,
            n_state: student_state,
            ff_dim: student_ff_dim,
            n_head: student_heads,
            shared_embeddings: true, // Enable for parameter sharing
        }
    }
}

/// Dynamic quantization parameters
#[derive(Debug, Clone)]
pub struct DynamicQuantParams {
    /// Minimum values observed
    pub min_vals: Vec<f32>,
    /// Maximum values observed
    pub max_vals: Vec<f32>,
    /// Running average of scale factors
    pub scale_history: Vec<f32>,
    /// Number of calibration steps completed
    pub calibration_steps: usize,
    /// Whether calibration is complete
    pub calibrated: bool,
}

impl Default for DynamicQuantParams {
    fn default() -> Self {
        Self {
            min_vals: Vec::new(),
            max_vals: Vec::new(),
            scale_history: Vec::new(),
            calibration_steps: 0,
            calibrated: false,
        }
    }
}
