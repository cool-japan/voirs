//! Integration module for advanced model optimization
//!
//! This module provides high-level interfaces to apply various optimization
//! techniques to ASR models, with automatic configuration and evaluation.

use super::advanced_optimization::{
    AdvancedOptimizationConfig, KnowledgeDistillationOptimizer, MixedPrecisionOptimizer,
    PerformanceMeasurement, ProgressivePruningOptimizer,
};
use crate::RecognitionError;
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tracing::info;

/// Comprehensive optimization pipeline
#[derive(Debug)]
pub struct OptimizationPipeline {
    /// Advanced optimization configuration
    config: AdvancedOptimizationConfig,
    /// Knowledge distillation optimizer
    kd_optimizer: Option<KnowledgeDistillationOptimizer>,
    /// Progressive pruning optimizer
    pruning_optimizer: Option<ProgressivePruningOptimizer>,
    /// Mixed-precision optimizer
    mp_optimizer: Option<MixedPrecisionOptimizer>,
    /// Device
    device: Device,
    /// Optimization results
    results: OptimizationResults,
}

/// Optimization pipeline results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResults {
    /// Original model statistics
    pub original_stats: ModelStats,
    /// Optimized model statistics
    pub optimized_stats: ModelStats,
    /// Knowledge distillation results
    pub distillation_results: Option<DistillationResults>,
    /// Pruning results
    pub pruning_results: Option<PruningResults>,
    /// Mixed-precision results
    pub mixed_precision_results: Option<MixedPrecisionResults>,
    /// Overall optimization summary
    pub summary: OptimizationSummary,
}

/// Model statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStats {
    /// Number of parameters
    pub num_parameters: usize,
    /// Model size in MB
    pub model_size_mb: f32,
    /// Inference time in milliseconds
    pub inference_time_ms: f32,
    /// Memory usage in MB
    pub memory_usage_mb: f32,
    /// Accuracy score
    pub accuracy: f32,
    /// Real-time factor
    pub rtf: f32,
}

/// Knowledge distillation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationResults {
    /// Final distillation loss
    pub final_loss: f32,
    /// Knowledge transfer efficiency
    pub transfer_efficiency: f32,
    /// Best temperature found
    pub optimal_temperature: f32,
    /// Accuracy retention
    pub accuracy_retention: f32,
}

/// Pruning results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningResults {
    /// Final sparsity achieved
    pub final_sparsity: f32,
    /// Model size reduction
    pub size_reduction: f32,
    /// Inference speedup
    pub speedup: f32,
    /// Accuracy retention
    pub accuracy_retention: f32,
    /// Number of pruning steps
    pub pruning_steps: usize,
}

/// Mixed-precision results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionResults {
    /// Precision distribution
    pub precision_distribution: HashMap<String, usize>, // DType name -> count
    /// Estimated speedup
    pub estimated_speedup: f32,
    /// Memory reduction
    pub memory_reduction: f32,
    /// Accuracy retention
    pub accuracy_retention: f32,
}

/// Overall optimization summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSummary {
    /// Total optimization time in seconds
    pub optimization_time_s: f32,
    /// Overall speedup achieved
    pub overall_speedup: f32,
    /// Overall memory reduction
    pub overall_memory_reduction: f32,
    /// Overall model size reduction
    pub overall_size_reduction: f32,
    /// Final accuracy retention
    pub final_accuracy_retention: f32,
    /// Optimization techniques applied
    pub techniques_applied: Vec<String>,
    /// Whether optimization meets targets
    pub meets_targets: bool,
}

impl OptimizationPipeline {
    /// Create new optimization pipeline
    pub fn new(config: AdvancedOptimizationConfig, device: Device) -> Self {
        Self {
            kd_optimizer: if config.enable_knowledge_distillation {
                Some(KnowledgeDistillationOptimizer::new(
                    config.clone(),
                    device.clone(),
                ))
            } else {
                None
            },
            pruning_optimizer: if config.enable_progressive_pruning {
                Some(ProgressivePruningOptimizer::new(
                    config.clone(),
                    device.clone(),
                ))
            } else {
                None
            },
            mp_optimizer: if config.enable_mixed_precision {
                Some(MixedPrecisionOptimizer::new(config.clone(), device.clone()))
            } else {
                None
            },
            config,
            device,
            results: OptimizationResults::default(),
        }
    }

    /// Run comprehensive optimization pipeline
    pub async fn optimize_model(
        &mut self,
        model_layers: &mut HashMap<String, Tensor>,
        teacher_layers: Option<HashMap<String, Tensor>>,
        validation_fn: impl Fn(&HashMap<String, Tensor>) -> Result<ModelStats, RecognitionError> + Copy,
    ) -> Result<OptimizationResults, RecognitionError> {
        let start_time = Instant::now();
        info!("Starting comprehensive model optimization pipeline");

        // Measure original model statistics
        let original_stats = validation_fn(model_layers)?;
        info!(
            "Original model: {:.1}MB, {:.1}ms inference, {:.3} accuracy",
            original_stats.model_size_mb, original_stats.inference_time_ms, original_stats.accuracy
        );

        let mut techniques_applied = Vec::new();

        // Step 1: Knowledge Distillation (if enabled and teacher provided)
        let distillation_results = if self.config.enable_knowledge_distillation {
            if let (Some(teacher), Some(kd_optimizer)) =
                (teacher_layers, self.kd_optimizer.as_mut())
            {
                info!("Applying knowledge distillation");
                kd_optimizer.set_teacher_layers(teacher);
                kd_optimizer.set_student_layers(model_layers.clone());

                let results = Self::apply_knowledge_distillation_static(
                    &self.device,
                    kd_optimizer,
                    model_layers,
                    validation_fn,
                )
                .await?;
                techniques_applied.push("Knowledge Distillation".to_string());
                Some(results)
            } else {
                None
            }
        } else {
            None
        };

        // Step 2: Progressive Pruning (if enabled)
        let pruning_results = if self.config.enable_progressive_pruning {
            if let Some(pruning_optimizer) = self.pruning_optimizer.as_mut() {
                info!("Applying progressive pruning");
                let results = Self::apply_progressive_pruning_static(
                    &self.device,
                    pruning_optimizer,
                    model_layers,
                    validation_fn,
                )
                .await?;
                techniques_applied.push("Progressive Pruning".to_string());
                Some(results)
            } else {
                None
            }
        } else {
            None
        };

        // Step 3: Mixed-Precision Optimization (if enabled)
        let mixed_precision_results = if self.config.enable_mixed_precision {
            if let Some(mp_optimizer) = self.mp_optimizer.as_mut() {
                info!("Applying mixed-precision optimization");
                let results = Self::apply_mixed_precision_static(
                    &self.device,
                    mp_optimizer,
                    model_layers,
                    validation_fn,
                )
                .await?;
                techniques_applied.push("Mixed-Precision".to_string());
                Some(results)
            } else {
                None
            }
        } else {
            None
        };

        // Measure final optimized model statistics
        let optimized_stats = validation_fn(model_layers)?;
        let optimization_time = start_time.elapsed().as_secs_f32();

        // Compute overall metrics
        let overall_speedup = original_stats.inference_time_ms / optimized_stats.inference_time_ms;
        let overall_memory_reduction = (original_stats.memory_usage_mb
            - optimized_stats.memory_usage_mb)
            / original_stats.memory_usage_mb;
        let overall_size_reduction = (original_stats.model_size_mb - optimized_stats.model_size_mb)
            / original_stats.model_size_mb;
        let final_accuracy_retention = optimized_stats.accuracy / original_stats.accuracy;

        let meets_targets = optimized_stats.rtf <= self.config.performance_budget
            && final_accuracy_retention >= self.config.accuracy_budget;

        let summary = OptimizationSummary {
            optimization_time_s: optimization_time,
            overall_speedup,
            overall_memory_reduction,
            overall_size_reduction,
            final_accuracy_retention,
            techniques_applied,
            meets_targets,
        };

        let results = OptimizationResults {
            original_stats,
            optimized_stats,
            distillation_results,
            pruning_results,
            mixed_precision_results,
            summary,
        };

        self.results = results.clone();

        info!("Optimization completed in {:.1}s: {:.2}x speedup, {:.1}% memory reduction, {:.1}% accuracy retention",
              optimization_time, overall_speedup, overall_memory_reduction * 100.0, final_accuracy_retention * 100.0);

        Ok(results)
    }

    /// Apply knowledge distillation
    async fn apply_knowledge_distillation_static(
        device: &Device,
        kd_optimizer: &mut KnowledgeDistillationOptimizer,
        model_layers: &mut HashMap<String, Tensor>,
        validation_fn: impl Fn(&HashMap<String, Tensor>) -> Result<ModelStats, RecognitionError>,
    ) -> Result<DistillationResults, RecognitionError> {
        // Analyze temperature sensitivity
        let temperatures = vec![1.0, 2.0, 4.0, 8.0, 16.0];
        let validation_data: Vec<Tensor> = vec![
            Tensor::randn(0.0, 1.0, (1, 512), device)?,
            Tensor::randn(0.0, 1.0, (1, 512), device)?,
            Tensor::randn(0.0, 1.0, (1, 512), device)?,
        ];

        kd_optimizer
            .analyze_temperature_sensitivity(temperatures, &validation_data)
            .await?;

        // Perform intermediate layer distillation
        let layer_losses = kd_optimizer.distill_intermediate_layers()?;
        let final_loss = layer_losses.values().sum::<f32>() / layer_losses.len() as f32;

        // Get stats after distillation
        let stats = kd_optimizer.get_stats();
        let optimal_temperature = stats
            .temperature_sensitivity
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|&(temp, _)| temp)
            .unwrap_or(4.0);

        // Measure final accuracy
        let final_stats = validation_fn(model_layers)?;
        let initial_accuracy = 1.0; // Assume teacher has 100% accuracy
        let accuracy_retention = final_stats.accuracy / initial_accuracy;

        let transfer_efficiency = stats.transfer_efficiency;

        Ok(DistillationResults {
            final_loss,
            transfer_efficiency,
            optimal_temperature,
            accuracy_retention,
        })
    }

    /// Apply progressive pruning
    async fn apply_progressive_pruning_static(
        _device: &Device,
        pruning_optimizer: &mut ProgressivePruningOptimizer,
        model_layers: &mut HashMap<String, Tensor>,
        validation_fn: impl Fn(&HashMap<String, Tensor>) -> Result<ModelStats, RecognitionError>,
    ) -> Result<PruningResults, RecognitionError> {
        // Compute layer importance scores
        pruning_optimizer.compute_layer_importance(model_layers)?;

        let initial_stats = validation_fn(model_layers)?;
        let mut all_step_results = Vec::new();

        // Execute progressive pruning steps
        let (current_step, total_steps) = pruning_optimizer.get_progress();
        for _step in current_step..total_steps {
            let step_result = pruning_optimizer.execute_pruning_step(model_layers, |layers| {
                validation_fn(layers).map(|stats| stats.accuracy)
            })?;
            all_step_results.push(step_result);
        }

        let final_stats = validation_fn(model_layers)?;

        // Compute overall results
        let final_sparsity = if let Some(last_step) = all_step_results.last() {
            last_step.pruning_ratio
        } else {
            0.0
        };

        let size_reduction =
            (initial_stats.model_size_mb - final_stats.model_size_mb) / initial_stats.model_size_mb;
        let speedup = initial_stats.inference_time_ms / final_stats.inference_time_ms;
        let accuracy_retention = final_stats.accuracy / initial_stats.accuracy;

        Ok(PruningResults {
            final_sparsity,
            size_reduction,
            speedup,
            accuracy_retention,
            pruning_steps: all_step_results.len(),
        })
    }

    /// Apply mixed-precision optimization
    async fn apply_mixed_precision_static(
        _device: &Device,
        mp_optimizer: &mut MixedPrecisionOptimizer,
        model_layers: &mut HashMap<String, Tensor>,
        validation_fn: impl Fn(&HashMap<String, Tensor>) -> Result<ModelStats, RecognitionError>,
    ) -> Result<MixedPrecisionResults, RecognitionError> {
        let initial_stats = validation_fn(model_layers)?;

        // Perform automatic precision selection
        mp_optimizer.auto_select_precisions(model_layers, |layers| {
            validation_fn(layers).map(|stats| PerformanceMeasurement {
                inference_time_ms: stats.inference_time_ms,
                memory_usage_mb: stats.memory_usage_mb,
                accuracy: stats.accuracy,
                model_size_mb: stats.model_size_mb,
            })
        })?;

        // Apply mixed-precision configuration
        let mp_stats = mp_optimizer.apply_mixed_precision(model_layers)?;

        let final_stats = validation_fn(model_layers)?;

        // Create precision distribution
        let mut precision_distribution = HashMap::new();
        precision_distribution.insert("FP32".to_string(), mp_stats.fp32_layers);
        precision_distribution.insert("FP16".to_string(), mp_stats.fp16_layers);
        precision_distribution.insert("INT8".to_string(), mp_stats.int8_layers);

        let memory_reduction = (initial_stats.memory_usage_mb - final_stats.memory_usage_mb)
            / initial_stats.memory_usage_mb;
        let accuracy_retention = final_stats.accuracy / initial_stats.accuracy;

        Ok(MixedPrecisionResults {
            precision_distribution,
            estimated_speedup: mp_stats.estimated_speedup,
            memory_reduction,
            accuracy_retention,
        })
    }

    /// Generate optimization report
    pub fn generate_report(&self) -> String {
        let results = &self.results;
        let mut report = String::new();

        report.push_str("# Model Optimization Report\n\n");

        // Summary
        report.push_str("## Summary\n");
        report.push_str(&format!(
            "- **Overall Speedup**: {:.2}x\n",
            results.summary.overall_speedup
        ));
        report.push_str(&format!(
            "- **Memory Reduction**: {:.1}%\n",
            results.summary.overall_memory_reduction * 100.0
        ));
        report.push_str(&format!(
            "- **Model Size Reduction**: {:.1}%\n",
            results.summary.overall_size_reduction * 100.0
        ));
        report.push_str(&format!(
            "- **Accuracy Retention**: {:.1}%\n",
            results.summary.final_accuracy_retention * 100.0
        ));
        report.push_str(&format!(
            "- **Optimization Time**: {:.1}s\n",
            results.summary.optimization_time_s
        ));
        report.push_str(&format!(
            "- **Meets Targets**: {}\n\n",
            if results.summary.meets_targets {
                "✅ Yes"
            } else {
                "❌ No"
            }
        ));

        // Original vs Optimized
        report.push_str("## Model Comparison\n");
        report.push_str("| Metric | Original | Optimized | Improvement |\n");
        report.push_str("|--------|----------|-----------|-------------|\n");

        let size_improvement = (results.original_stats.model_size_mb
            - results.optimized_stats.model_size_mb)
            / results.original_stats.model_size_mb
            * 100.0;
        let speed_improvement = (results.original_stats.inference_time_ms
            - results.optimized_stats.inference_time_ms)
            / results.original_stats.inference_time_ms
            * 100.0;
        let memory_improvement = (results.original_stats.memory_usage_mb
            - results.optimized_stats.memory_usage_mb)
            / results.original_stats.memory_usage_mb
            * 100.0;

        report.push_str(&format!(
            "| Model Size (MB) | {:.1} | {:.1} | {:.1}% |\n",
            results.original_stats.model_size_mb,
            results.optimized_stats.model_size_mb,
            size_improvement
        ));
        report.push_str(&format!(
            "| Inference Time (ms) | {:.1} | {:.1} | {:.1}% |\n",
            results.original_stats.inference_time_ms,
            results.optimized_stats.inference_time_ms,
            speed_improvement
        ));
        report.push_str(&format!(
            "| Memory Usage (MB) | {:.1} | {:.1} | {:.1}% |\n",
            results.original_stats.memory_usage_mb,
            results.optimized_stats.memory_usage_mb,
            memory_improvement
        ));
        report.push_str(&format!(
            "| Accuracy | {:.3} | {:.3} | {:.1}% |\n\n",
            results.original_stats.accuracy,
            results.optimized_stats.accuracy,
            (results.optimized_stats.accuracy - results.original_stats.accuracy)
                / results.original_stats.accuracy
                * 100.0
        ));

        // Techniques Applied
        report.push_str("## Optimization Techniques Applied\n");
        for technique in &results.summary.techniques_applied {
            report.push_str(&format!("- {}\n", technique));
        }
        report.push_str("\n");

        // Detailed Results
        if let Some(distillation) = &results.distillation_results {
            report.push_str("### Knowledge Distillation Results\n");
            report.push_str(&format!("- Final Loss: {:.6}\n", distillation.final_loss));
            report.push_str(&format!(
                "- Transfer Efficiency: {:.3}\n",
                distillation.transfer_efficiency
            ));
            report.push_str(&format!(
                "- Optimal Temperature: {:.1}\n",
                distillation.optimal_temperature
            ));
            report.push_str(&format!(
                "- Accuracy Retention: {:.1}%\n\n",
                distillation.accuracy_retention * 100.0
            ));
        }

        if let Some(pruning) = &results.pruning_results {
            report.push_str("### Progressive Pruning Results\n");
            report.push_str(&format!(
                "- Final Sparsity: {:.1}%\n",
                pruning.final_sparsity * 100.0
            ));
            report.push_str(&format!(
                "- Size Reduction: {:.1}%\n",
                pruning.size_reduction * 100.0
            ));
            report.push_str(&format!("- Speedup: {:.2}x\n", pruning.speedup));
            report.push_str(&format!("- Pruning Steps: {}\n", pruning.pruning_steps));
            report.push_str(&format!(
                "- Accuracy Retention: {:.1}%\n\n",
                pruning.accuracy_retention * 100.0
            ));
        }

        if let Some(mixed_precision) = &results.mixed_precision_results {
            report.push_str("### Mixed-Precision Results\n");
            report.push_str(&format!(
                "- Estimated Speedup: {:.2}x\n",
                mixed_precision.estimated_speedup
            ));
            report.push_str(&format!(
                "- Memory Reduction: {:.1}%\n",
                mixed_precision.memory_reduction * 100.0
            ));
            report.push_str(&format!(
                "- Accuracy Retention: {:.1}%\n",
                mixed_precision.accuracy_retention * 100.0
            ));
            report.push_str("- Precision Distribution:\n");
            for (precision, count) in &mixed_precision.precision_distribution {
                report.push_str(&format!("  - {}: {} layers\n", precision, count));
            }
            report.push_str("\n");
        }

        report
    }

    /// Get optimization results
    pub fn get_results(&self) -> &OptimizationResults {
        &self.results
    }
}

impl Default for OptimizationResults {
    fn default() -> Self {
        Self {
            original_stats: ModelStats::default(),
            optimized_stats: ModelStats::default(),
            distillation_results: None,
            pruning_results: None,
            mixed_precision_results: None,
            summary: OptimizationSummary::default(),
        }
    }
}

impl Default for ModelStats {
    fn default() -> Self {
        Self {
            num_parameters: 0,
            model_size_mb: 0.0,
            inference_time_ms: 0.0,
            memory_usage_mb: 0.0,
            accuracy: 0.0,
            rtf: 0.0,
        }
    }
}

impl Default for OptimizationSummary {
    fn default() -> Self {
        Self {
            optimization_time_s: 0.0,
            overall_speedup: 1.0,
            overall_memory_reduction: 0.0,
            overall_size_reduction: 0.0,
            final_accuracy_retention: 1.0,
            techniques_applied: Vec::new(),
            meets_targets: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[tokio::test]
    async fn test_optimization_pipeline_creation() {
        let config = AdvancedOptimizationConfig::default();
        let device = Device::Cpu;
        let pipeline = OptimizationPipeline::new(config, device);

        assert!(pipeline.mp_optimizer.is_some());
        assert!(pipeline.kd_optimizer.is_none()); // Disabled by default
        assert!(pipeline.pruning_optimizer.is_none()); // Disabled by default
    }

    #[test]
    fn test_optimization_results_default() {
        let results = OptimizationResults::default();
        assert_eq!(results.original_stats.num_parameters, 0);
        assert_eq!(results.optimized_stats.num_parameters, 0);
        assert!(results.distillation_results.is_none());
        assert!(results.pruning_results.is_none());
        assert!(results.mixed_precision_results.is_none());
    }

    #[test]
    fn test_report_generation() {
        let mut results = OptimizationResults::default();
        results.summary.overall_speedup = 1.5;
        results.summary.overall_memory_reduction = 0.2;
        results.summary.final_accuracy_retention = 0.98;
        results.summary.techniques_applied = vec!["Mixed-Precision".to_string()];
        results.summary.meets_targets = true;

        let mut pipeline =
            OptimizationPipeline::new(AdvancedOptimizationConfig::default(), Device::Cpu);
        pipeline.results = results;

        let report = pipeline.generate_report();
        assert!(report.contains("# Model Optimization Report"));
        assert!(report.contains("1.50x"));
        assert!(report.contains("20.0%"));
        assert!(report.contains("Mixed-Precision"));
    }
}
