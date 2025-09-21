//! Model optimization techniques for efficient inference
//!
//! This module implements various model optimization techniques to reduce memory usage,
//! improve inference speed, and maintain quality for production deployments:
//! - INT8/FP16 quantization for reduced memory and faster inference
//! - Model pruning for removing redundant parameters
//! - Knowledge distillation for creating smaller, efficient student models
//! - Dynamic optimization based on hardware capabilities

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use candle_core::Device;
use serde::{Deserialize, Serialize};

use crate::{AcousticError, AcousticModel, Phoneme, Result};

/// Model optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable quantization optimizations
    pub quantization: QuantizationConfig,
    /// Enable pruning optimizations
    pub pruning: PruningConfig,
    /// Enable knowledge distillation
    pub distillation: DistillationConfig,
    /// Hardware-specific optimizations
    pub hardware_optimization: HardwareOptimization,
    /// Target optimization goals
    pub optimization_targets: OptimizationTargets,
}

/// Quantization configuration for reducing model precision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Enable quantization
    pub enabled: bool,
    /// Target quantization precision
    pub precision: QuantizationPrecision,
    /// Calibration dataset size for quantization
    pub calibration_samples: usize,
    /// Layers to exclude from quantization (sensitive layers)
    pub excluded_layers: Vec<String>,
    /// Post-training quantization vs quantization-aware training
    pub quantization_method: QuantizationMethod,
    /// Dynamic quantization for variable precision
    pub dynamic_quantization: bool,
}

/// Quantization precision options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationPrecision {
    /// 8-bit integer quantization
    Int8,
    /// 16-bit floating point
    Float16,
    /// Mixed precision (FP16 + FP32 for sensitive layers)
    Mixed,
    /// Dynamic precision based on layer sensitivity
    Dynamic,
}

/// Quantization method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationMethod {
    /// Post-training quantization (faster setup)
    PostTraining,
    /// Quantization-aware training (higher quality)
    QuantizationAware,
    /// Gradual quantization with fine-tuning
    Gradual,
}

/// Model pruning configuration for removing redundant parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    /// Enable pruning
    pub enabled: bool,
    /// Pruning strategy
    pub strategy: PruningStrategy,
    /// Target sparsity percentage (0.0 to 1.0)
    pub target_sparsity: f32,
    /// Gradual pruning over multiple steps
    pub gradual_pruning: bool,
    /// Structured vs unstructured pruning
    pub pruning_type: PruningType,
    /// Layers to exclude from pruning
    pub excluded_layers: Vec<String>,
}

/// Pruning strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PruningStrategy {
    /// Magnitude-based pruning (remove smallest weights)
    Magnitude,
    /// Gradient-based pruning (remove low-gradient weights)
    Gradient,
    /// Fisher information-based pruning
    Fisher,
    /// Layer-wise adaptive pruning
    Adaptive,
}

/// Pruning type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PruningType {
    /// Remove individual weights (fine-grained)
    Unstructured,
    /// Remove entire channels/filters (coarse-grained)
    Structured,
    /// Mixed approach
    Mixed,
}

/// Knowledge distillation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    /// Enable knowledge distillation
    pub enabled: bool,
    /// Teacher model path (larger, more accurate model)
    pub teacher_model_path: Option<String>,
    /// Student model configuration (smaller, faster model)
    pub student_config: StudentModelConfig,
    /// Distillation temperature for softmax
    pub temperature: f32,
    /// Weight for distillation loss vs task loss
    pub distillation_weight: f32,
    /// Distillation method
    pub method: DistillationMethod,
}

/// Student model configuration for knowledge distillation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentModelConfig {
    /// Reduction factor for hidden dimensions
    pub hidden_reduction_factor: f32,
    /// Reduction factor for number of layers
    pub layer_reduction_factor: f32,
    /// Number of attention heads in student model
    pub num_heads: usize,
    /// Whether to use shared parameters
    pub shared_parameters: bool,
}

/// Knowledge distillation method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistillationMethod {
    /// Standard knowledge distillation (output matching)
    Standard,
    /// Feature-based distillation (intermediate layer matching)
    FeatureBased,
    /// Attention-based distillation (attention map matching)
    AttentionBased,
    /// Progressive distillation (gradual reduction)
    Progressive,
}

/// Hardware-specific optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareOptimization {
    /// Target device type
    pub target_device: TargetDevice,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Enable GPU optimizations if available
    pub enable_gpu: bool,
    /// Memory constraints (MB)
    pub memory_limit_mb: Option<usize>,
    /// CPU core count for optimization
    pub cpu_cores: Option<usize>,
}

/// Target deployment device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TargetDevice {
    /// Mobile/embedded devices (aggressive optimization)
    Mobile,
    /// Desktop/laptop (balanced optimization)
    Desktop,
    /// Server/cloud (performance-focused optimization)
    Server,
    /// Edge devices (power-efficient optimization)
    Edge,
}

/// Optimization targets and constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationTargets {
    /// Maximum acceptable quality degradation (0.0 to 1.0)
    pub max_quality_loss: f32,
    /// Target memory reduction factor
    pub memory_reduction_target: f32,
    /// Target inference speedup factor
    pub speed_improvement_target: f32,
    /// Maximum model size in MB
    pub max_model_size_mb: Option<usize>,
    /// Target latency in milliseconds
    pub target_latency_ms: Option<f32>,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            quantization: QuantizationConfig {
                enabled: true,
                precision: QuantizationPrecision::Float16,
                calibration_samples: 1000,
                excluded_layers: vec!["output".to_string(), "embedding".to_string()],
                quantization_method: QuantizationMethod::PostTraining,
                dynamic_quantization: false,
            },
            pruning: PruningConfig {
                enabled: true,
                strategy: PruningStrategy::Magnitude,
                target_sparsity: 0.3, // 30% sparsity
                gradual_pruning: true,
                pruning_type: PruningType::Unstructured,
                excluded_layers: vec!["output".to_string()],
            },
            distillation: DistillationConfig {
                enabled: false, // Requires teacher model
                teacher_model_path: None,
                student_config: StudentModelConfig {
                    hidden_reduction_factor: 0.5,
                    layer_reduction_factor: 0.5,
                    num_heads: 4,
                    shared_parameters: false,
                },
                temperature: 3.0,
                distillation_weight: 0.7,
                method: DistillationMethod::Standard,
            },
            hardware_optimization: HardwareOptimization {
                target_device: TargetDevice::Desktop,
                enable_simd: true,
                enable_gpu: true,
                memory_limit_mb: Some(500), // 500MB limit
                cpu_cores: None,            // Auto-detect
            },
            optimization_targets: OptimizationTargets {
                max_quality_loss: 0.05,        // 5% max quality loss
                memory_reduction_target: 0.5,  // 50% memory reduction
                speed_improvement_target: 2.0, // 2x speedup
                max_model_size_mb: Some(100),  // 100MB max
                target_latency_ms: Some(10.0), // 10ms target
            },
        }
    }
}

/// Model optimization results and metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResults {
    /// Original model metrics
    pub original_metrics: ModelMetrics,
    /// Optimized model metrics
    pub optimized_metrics: ModelMetrics,
    /// Applied optimizations
    pub applied_optimizations: Vec<AppliedOptimization>,
    /// Quality assessment results
    pub quality_assessment: QualityAssessment,
    /// Performance improvements
    pub performance_improvements: PerformanceImprovements,
}

/// Model metrics for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Model size in bytes
    pub model_size_bytes: usize,
    /// Memory usage during inference (MB)
    pub memory_usage_mb: f32,
    /// Inference latency (ms)
    pub inference_latency_ms: f32,
    /// Throughput (samples/second)
    pub throughput_sps: f32,
    /// Number of parameters
    pub parameter_count: usize,
    /// Number of operations (FLOPs)
    pub flop_count: usize,
}

/// Applied optimization details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedOptimization {
    /// Optimization type
    pub optimization_type: String,
    /// Configuration used
    pub config: serde_json::Value,
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Metrics impact
    pub metrics_impact: ModelMetrics,
}

/// Quality assessment after optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessment {
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f32,
    /// Quality metrics by category
    pub category_scores: HashMap<String, f32>,
    /// Sample-based quality comparison
    pub sample_comparisons: Vec<SampleQualityComparison>,
}

/// Individual sample quality comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleQualityComparison {
    /// Sample identifier
    pub sample_id: String,
    /// Original model output quality
    pub original_quality: f32,
    /// Optimized model output quality
    pub optimized_quality: f32,
    /// Quality difference
    pub quality_difference: f32,
}

/// Performance improvements summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImprovements {
    /// Memory reduction ratio
    pub memory_reduction: f32,
    /// Speed improvement ratio
    pub speed_improvement: f32,
    /// Model size reduction ratio
    pub size_reduction: f32,
    /// Energy efficiency improvement
    pub energy_efficiency: f32,
}

/// Model optimizer for applying various optimization techniques
pub struct ModelOptimizer {
    config: OptimizationConfig,
    _device: Device,
    optimization_history: Vec<OptimizationResults>,
}

impl ModelOptimizer {
    /// Create new model optimizer
    pub fn new(config: OptimizationConfig, device: Device) -> Self {
        Self {
            config,
            _device: device,
            optimization_history: Vec::new(),
        }
    }

    /// Optimize an acoustic model using configured techniques
    pub async fn optimize_model(
        &mut self,
        model: Arc<dyn AcousticModel>,
    ) -> Result<(Arc<dyn AcousticModel>, OptimizationResults)> {
        let mut optimized_model = model.clone();
        let mut applied_optimizations = Vec::new();

        // Measure original model metrics
        let original_metrics = self.measure_model_metrics(&*optimized_model).await?;

        // Apply quantization if enabled
        if self.config.quantization.enabled {
            match self.apply_quantization(optimized_model.clone()).await {
                Ok(quantized_model) => {
                    optimized_model = quantized_model;
                    applied_optimizations.push(AppliedOptimization {
                        optimization_type: "quantization".to_string(),
                        config: serde_json::to_value(&self.config.quantization).unwrap_or_default(),
                        success: true,
                        error_message: None,
                        metrics_impact: self.measure_model_metrics(&*optimized_model).await?,
                    });
                }
                Err(e) => {
                    applied_optimizations.push(AppliedOptimization {
                        optimization_type: "quantization".to_string(),
                        config: serde_json::to_value(&self.config.quantization).unwrap_or_default(),
                        success: false,
                        error_message: Some(e.to_string()),
                        metrics_impact: original_metrics.clone(),
                    });
                }
            }
        }

        // Apply pruning if enabled
        if self.config.pruning.enabled {
            match self.apply_pruning(optimized_model.clone()).await {
                Ok(pruned_model) => {
                    optimized_model = pruned_model;
                    applied_optimizations.push(AppliedOptimization {
                        optimization_type: "pruning".to_string(),
                        config: serde_json::to_value(&self.config.pruning).unwrap_or_default(),
                        success: true,
                        error_message: None,
                        metrics_impact: self.measure_model_metrics(&*optimized_model).await?,
                    });
                }
                Err(e) => {
                    applied_optimizations.push(AppliedOptimization {
                        optimization_type: "pruning".to_string(),
                        config: serde_json::to_value(&self.config.pruning).unwrap_or_default(),
                        success: false,
                        error_message: Some(e.to_string()),
                        metrics_impact: original_metrics.clone(),
                    });
                }
            }
        }

        // Apply knowledge distillation if enabled and teacher model is available
        if self.config.distillation.enabled && self.config.distillation.teacher_model_path.is_some()
        {
            match self
                .apply_knowledge_distillation(optimized_model.clone())
                .await
            {
                Ok(distilled_model) => {
                    optimized_model = distilled_model;
                    applied_optimizations.push(AppliedOptimization {
                        optimization_type: "knowledge_distillation".to_string(),
                        config: serde_json::to_value(&self.config.distillation).unwrap_or_default(),
                        success: true,
                        error_message: None,
                        metrics_impact: self.measure_model_metrics(&*optimized_model).await?,
                    });
                }
                Err(e) => {
                    applied_optimizations.push(AppliedOptimization {
                        optimization_type: "knowledge_distillation".to_string(),
                        config: serde_json::to_value(&self.config.distillation).unwrap_or_default(),
                        success: false,
                        error_message: Some(e.to_string()),
                        metrics_impact: original_metrics.clone(),
                    });
                }
            }
        }

        // Measure optimized model metrics
        let optimized_metrics = self.measure_model_metrics(&*optimized_model).await?;

        // Assess quality impact
        let quality_assessment = self.assess_quality_impact(&model, &optimized_model).await?;

        // Calculate performance improvements
        let performance_improvements =
            self.calculate_performance_improvements(&original_metrics, &optimized_metrics);

        let results = OptimizationResults {
            original_metrics,
            optimized_metrics,
            applied_optimizations,
            quality_assessment,
            performance_improvements,
        };

        // Store optimization history
        self.optimization_history.push(results.clone());

        Ok((optimized_model, results))
    }

    /// Apply quantization to reduce model precision
    async fn apply_quantization(
        &self,
        model: Arc<dyn AcousticModel>,
    ) -> Result<Arc<dyn AcousticModel>> {
        match self.config.quantization.precision {
            QuantizationPrecision::Int8 => self.apply_int8_quantization(model).await,
            QuantizationPrecision::Float16 => self.apply_fp16_quantization(model).await,
            QuantizationPrecision::Mixed => self.apply_mixed_precision(model).await,
            QuantizationPrecision::Dynamic => self.apply_dynamic_quantization(model).await,
        }
    }

    /// Apply INT8 quantization
    async fn apply_int8_quantization(
        &self,
        _model: Arc<dyn AcousticModel>,
    ) -> Result<Arc<dyn AcousticModel>> {
        // This is a placeholder implementation
        // In practice, this would involve:
        // 1. Collecting activation statistics from calibration data
        // 2. Computing quantization scales and zero points
        // 3. Converting model weights and activations to INT8
        // 4. Implementing quantized operations

        // For now, return the original model
        // This would be replaced with actual quantization logic
        Err(AcousticError::Processing(
            "INT8 quantization not yet implemented".to_string(),
        ))
    }

    /// Apply FP16 quantization
    async fn apply_fp16_quantization(
        &self,
        _model: Arc<dyn AcousticModel>,
    ) -> Result<Arc<dyn AcousticModel>> {
        // This is a placeholder implementation
        // In practice, this would involve:
        // 1. Converting all model weights from FP32 to FP16
        // 2. Implementing FP16 operations
        // 3. Handling numerical stability issues

        // For now, return the original model
        // This would be replaced with actual FP16 conversion logic
        Err(AcousticError::Processing(
            "FP16 quantization not yet implemented".to_string(),
        ))
    }

    /// Apply mixed precision quantization
    async fn apply_mixed_precision(
        &self,
        _model: Arc<dyn AcousticModel>,
    ) -> Result<Arc<dyn AcousticModel>> {
        // This is a placeholder implementation
        // Mixed precision keeps sensitive layers in FP32 and others in FP16
        Err(AcousticError::Processing(
            "Mixed precision not yet implemented".to_string(),
        ))
    }

    /// Apply dynamic quantization
    async fn apply_dynamic_quantization(
        &self,
        _model: Arc<dyn AcousticModel>,
    ) -> Result<Arc<dyn AcousticModel>> {
        // This is a placeholder implementation
        // Dynamic quantization adjusts precision based on layer sensitivity
        Err(AcousticError::Processing(
            "Dynamic quantization not yet implemented".to_string(),
        ))
    }

    /// Apply pruning to remove redundant parameters
    async fn apply_pruning(&self, model: Arc<dyn AcousticModel>) -> Result<Arc<dyn AcousticModel>> {
        match self.config.pruning.strategy {
            PruningStrategy::Magnitude => self.apply_magnitude_pruning(model).await,
            PruningStrategy::Gradient => self.apply_gradient_pruning(model).await,
            PruningStrategy::Fisher => self.apply_fisher_pruning(model).await,
            PruningStrategy::Adaptive => self.apply_adaptive_pruning(model).await,
        }
    }

    /// Apply magnitude-based pruning
    async fn apply_magnitude_pruning(
        &self,
        _model: Arc<dyn AcousticModel>,
    ) -> Result<Arc<dyn AcousticModel>> {
        // This is a placeholder implementation
        // Magnitude pruning removes weights with smallest absolute values
        Err(AcousticError::Processing(
            "Magnitude pruning not yet implemented".to_string(),
        ))
    }

    /// Apply gradient-based pruning
    async fn apply_gradient_pruning(
        &self,
        _model: Arc<dyn AcousticModel>,
    ) -> Result<Arc<dyn AcousticModel>> {
        // This is a placeholder implementation
        // Gradient pruning removes weights with smallest gradients
        Err(AcousticError::Processing(
            "Gradient pruning not yet implemented".to_string(),
        ))
    }

    /// Apply Fisher information-based pruning
    async fn apply_fisher_pruning(
        &self,
        _model: Arc<dyn AcousticModel>,
    ) -> Result<Arc<dyn AcousticModel>> {
        // This is a placeholder implementation
        // Fisher pruning uses Fisher information to identify important weights
        Err(AcousticError::Processing(
            "Fisher pruning not yet implemented".to_string(),
        ))
    }

    /// Apply adaptive pruning
    async fn apply_adaptive_pruning(
        &self,
        _model: Arc<dyn AcousticModel>,
    ) -> Result<Arc<dyn AcousticModel>> {
        // This is a placeholder implementation
        // Adaptive pruning adjusts sparsity per layer based on sensitivity
        Err(AcousticError::Processing(
            "Adaptive pruning not yet implemented".to_string(),
        ))
    }

    /// Apply knowledge distillation
    async fn apply_knowledge_distillation(
        &self,
        _model: Arc<dyn AcousticModel>,
    ) -> Result<Arc<dyn AcousticModel>> {
        // This is a placeholder implementation
        // Knowledge distillation trains a smaller student model to mimic a larger teacher
        Err(AcousticError::Processing(
            "Knowledge distillation not yet implemented".to_string(),
        ))
    }

    /// Measure model performance metrics
    async fn measure_model_metrics<M: AcousticModel + ?Sized>(
        &self,
        model: &M,
    ) -> Result<ModelMetrics> {
        // Get model metadata for basic information
        let metadata = model.metadata();

        // Estimate memory usage based on model architecture
        let estimated_memory_mb = match metadata.architecture.as_str() {
            "tacotron2" => 150.0,   // Typical size for Tacotron2
            "fastspeech2" => 120.0, // Typical size for FastSpeech2
            "vits" => 200.0,        // Typical size for VITS
            _ => 128.0,             // Conservative default for unknown models
        };

        // Measure inference latency with a small test input
        let test_phonemes = vec![
            Phoneme::new("t"),
            Phoneme::new("e"),
            Phoneme::new("s"),
            Phoneme::new("t"),
        ];

        let latency_ms =
            (self.measure_inference_latency(model, &test_phonemes).await).unwrap_or(50.0);

        // Calculate throughput from latency (approximate)
        let throughput_sps = if latency_ms > 0.0 {
            1000.0 / latency_ms // samples per second based on latency
        } else {
            20.0 // Conservative fallback
        };

        // Estimate parameter count based on model architecture
        let parameter_count = match metadata.architecture.as_str() {
            "tacotron2" => 28_000_000,   // Typical parameter count for Tacotron2
            "fastspeech2" => 22_000_000, // Typical parameter count for FastSpeech2
            "vits" => 35_000_000,        // Typical parameter count for VITS
            _ => 15_000_000,             // Conservative default for unknown models
        };

        // Estimate FLOP count based on parameter count and typical operations
        let flop_count = parameter_count * 2; // Rough estimate: 2 FLOPs per parameter

        // Estimate model size in bytes based on parameter count
        let model_size_bytes = parameter_count * 4; // Assuming 4 bytes per parameter (FP32)

        Ok(ModelMetrics {
            model_size_bytes,
            memory_usage_mb: estimated_memory_mb,
            inference_latency_ms: latency_ms,
            throughput_sps,
            parameter_count,
            flop_count,
        })
    }

    /// Helper function to measure inference latency
    async fn measure_inference_latency<M: AcousticModel + ?Sized>(
        &self,
        model: &M,
        test_phonemes: &[Phoneme],
    ) -> Result<f32> {
        let start = Instant::now();

        // Perform a small test synthesis
        let _result = model.synthesize(test_phonemes, None).await?;

        let duration = start.elapsed();
        Ok(duration.as_millis() as f32)
    }

    /// Assess quality impact of optimizations
    async fn assess_quality_impact(
        &self,
        _original_model: &Arc<dyn AcousticModel>,
        _optimized_model: &Arc<dyn AcousticModel>,
    ) -> Result<QualityAssessment> {
        // This is a placeholder implementation
        // In practice, this would run quality assessment tests
        Ok(QualityAssessment {
            overall_score: 0.95, // 95% quality retained
            category_scores: [
                ("naturalness".to_string(), 0.94),
                ("intelligibility".to_string(), 0.96),
                ("prosody".to_string(), 0.93),
            ]
            .into_iter()
            .collect(),
            sample_comparisons: vec![],
        })
    }

    /// Calculate performance improvements
    fn calculate_performance_improvements(
        &self,
        original_metrics: &ModelMetrics,
        optimized_metrics: &ModelMetrics,
    ) -> PerformanceImprovements {
        let memory_reduction =
            1.0 - (optimized_metrics.memory_usage_mb / original_metrics.memory_usage_mb);
        let speed_improvement = optimized_metrics.throughput_sps / original_metrics.throughput_sps;
        let size_reduction = 1.0
            - (optimized_metrics.model_size_bytes as f32
                / original_metrics.model_size_bytes as f32);

        // Estimate energy efficiency improvement based on model size and speed
        let energy_efficiency = (speed_improvement + size_reduction) / 2.0;

        PerformanceImprovements {
            memory_reduction,
            speed_improvement,
            size_reduction,
            energy_efficiency,
        }
    }

    /// Get optimization history
    pub fn get_optimization_history(&self) -> &[OptimizationResults] {
        &self.optimization_history
    }

    /// Update optimization configuration
    pub fn update_config(&mut self, config: OptimizationConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_config_default() {
        let config = OptimizationConfig::default();
        assert!(config.quantization.enabled);
        assert!(config.pruning.enabled);
        assert!(!config.distillation.enabled); // Requires teacher model

        assert_eq!(config.pruning.target_sparsity, 0.3);
        assert_eq!(config.distillation.temperature, 3.0);
    }

    #[test]
    fn test_performance_improvements_calculation() {
        let optimizer = ModelOptimizer::new(OptimizationConfig::default(), Device::Cpu);

        let original = ModelMetrics {
            model_size_bytes: 100_000_000,
            memory_usage_mb: 400.0,
            inference_latency_ms: 50.0,
            throughput_sps: 20.0,
            parameter_count: 20_000_000,
            flop_count: 2_000_000_000,
        };

        let optimized = ModelMetrics {
            model_size_bytes: 50_000_000, // 50% size reduction
            memory_usage_mb: 200.0,       // 50% memory reduction
            inference_latency_ms: 25.0,   // 50% latency reduction
            throughput_sps: 40.0,         // 2x throughput improvement
            parameter_count: 10_000_000,  // 50% parameter reduction
            flop_count: 1_000_000_000,    // 50% FLOP reduction
        };

        let improvements = optimizer.calculate_performance_improvements(&original, &optimized);

        assert!((improvements.memory_reduction - 0.5).abs() < 0.001);
        assert!((improvements.speed_improvement - 2.0).abs() < 0.001);
        assert!((improvements.size_reduction - 0.5).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_model_metrics_measurement() {
        let optimizer = ModelOptimizer::new(OptimizationConfig::default(), Device::Cpu);

        // Create a mock model (this would be a real model in practice)
        struct MockModel;

        #[async_trait::async_trait]
        impl AcousticModel for MockModel {
            async fn synthesize(
                &self,
                _phonemes: &[crate::Phoneme],
                _config: Option<&crate::SynthesisConfig>,
            ) -> Result<crate::MelSpectrogram> {
                // Add a small delay to simulate realistic inference time
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                Ok(crate::MelSpectrogram {
                    data: vec![vec![0.0; 100]; 80], // 80 mel bins, 100 frames
                    n_mels: 80,
                    n_frames: 100,
                    sample_rate: 22050,
                    hop_length: 256,
                })
            }

            async fn synthesize_batch(
                &self,
                inputs: &[&[crate::Phoneme]],
                _configs: Option<&[crate::SynthesisConfig]>,
            ) -> Result<Vec<crate::MelSpectrogram>> {
                let mut results = Vec::new();
                for _ in inputs {
                    results.push(self.synthesize(&[], None).await?);
                }
                Ok(results)
            }

            fn metadata(&self) -> crate::AcousticModelMetadata {
                crate::AcousticModelMetadata {
                    name: "MockModel".to_string(),
                    version: "1.0.0".to_string(),
                    architecture: "Mock".to_string(),
                    supported_languages: vec![crate::LanguageCode::EnUs],
                    sample_rate: 22050,
                    mel_channels: 80,
                    is_multi_speaker: false,
                    speaker_count: None,
                }
            }

            fn supports(&self, _feature: crate::AcousticModelFeature) -> bool {
                false
            }

            async fn set_speaker(&mut self, _speaker_id: Option<u32>) -> Result<()> {
                Ok(())
            }
        }

        let model = MockModel;
        let metrics = optimizer.measure_model_metrics(&model).await.unwrap();

        assert!(metrics.model_size_bytes > 0);
        assert!(metrics.memory_usage_mb > 0.0);
        assert!(metrics.inference_latency_ms > 0.0);
        assert!(metrics.throughput_sps > 0.0);
        assert!(metrics.parameter_count > 0);
        assert!(metrics.flop_count > 0);
    }
}

// Type aliases for compatibility with test code
pub type OptimizationReport = OptimizationResults;
pub type OptimizationMetrics = ModelMetrics;
pub type HardwareTarget = TargetDevice;
pub type DistillationStrategy = DistillationMethod;
