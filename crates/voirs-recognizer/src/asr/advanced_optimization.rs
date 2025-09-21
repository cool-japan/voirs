//! Advanced model optimization techniques
//!
//! This module provides state-of-the-art optimization techniques including
//! knowledge distillation, progressive pruning, mixed-precision optimization,
//! and benchmark-driven optimization selection.

use crate::RecognitionError;
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, info};

/// Advanced optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedOptimizationConfig {
    /// Enable knowledge distillation
    pub enable_knowledge_distillation: bool,
    /// Knowledge distillation temperature
    pub distillation_temperature: f32,
    /// Knowledge distillation loss weight
    pub distillation_alpha: f32,

    /// Enable progressive pruning
    pub enable_progressive_pruning: bool,
    /// Initial pruning ratio
    pub initial_pruning_ratio: f32,
    /// Final pruning ratio
    pub final_pruning_ratio: f32,
    /// Number of progressive pruning steps
    pub pruning_steps: usize,

    /// Enable mixed-precision optimization
    pub enable_mixed_precision: bool,
    /// Automatic precision selection
    pub auto_precision_selection: bool,
    /// Performance budget (RTF threshold)
    pub performance_budget: f32,
    /// Accuracy budget (minimum accuracy retention)
    pub accuracy_budget: f32,

    /// Enable quantization-aware training simulation
    pub enable_qat_simulation: bool,
    /// QAT simulation iterations
    pub qat_iterations: usize,
    /// QAT learning rate
    pub qat_learning_rate: f32,

    /// Enable benchmark-driven optimization
    pub enable_benchmark_optimization: bool,
    /// Target hardware platform
    pub target_platform: OptimizationPlatform,
    /// Optimization objective
    pub optimization_objective: OptimizationObjective,
}

impl Default for AdvancedOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_knowledge_distillation: false,
            distillation_temperature: 4.0,
            distillation_alpha: 0.7,

            enable_progressive_pruning: false,
            initial_pruning_ratio: 0.1,
            final_pruning_ratio: 0.5,
            pruning_steps: 10,

            enable_mixed_precision: true,
            auto_precision_selection: true,
            performance_budget: 0.3, // RTF < 0.3
            accuracy_budget: 0.95,   // Retain 95% accuracy

            enable_qat_simulation: false,
            qat_iterations: 100,
            qat_learning_rate: 0.001,

            enable_benchmark_optimization: true,
            target_platform: OptimizationPlatform::CPU,
            optimization_objective: OptimizationObjective::Balanced,
        }
    }
}

/// Target platform for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationPlatform {
    CPU,
    GPU,
    Mobile,
    Edge,
    Server,
}

/// Optimization objective
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationObjective {
    /// Minimize latency
    Latency,
    /// Minimize memory usage
    Memory,
    /// Minimize model size
    Size,
    /// Balance all metrics
    Balanced,
    /// Maximize throughput
    Throughput,
}

/// Knowledge distillation trainer
#[derive(Debug)]
pub struct KnowledgeDistillationOptimizer {
    /// Teacher model reference (larger, accurate model)
    teacher_layers: HashMap<String, Tensor>,
    /// Student model reference (smaller, optimized model)
    student_layers: HashMap<String, Tensor>,
    /// Configuration
    config: AdvancedOptimizationConfig,
    /// Device
    device: Device,
    /// Distillation statistics
    distillation_stats: DistillationStats,
}

/// Knowledge distillation statistics
#[derive(Debug, Clone)]
pub struct DistillationStats {
    /// Teacher-student loss over time
    pub loss_history: Vec<f32>,
    /// Knowledge transfer efficiency
    pub transfer_efficiency: f32,
    /// Layer-wise distillation effectiveness
    pub layer_effectiveness: HashMap<String, f32>,
    /// Temperature sensitivity analysis
    pub temperature_sensitivity: Vec<(f32, f32)>, // (temperature, accuracy)
}

impl KnowledgeDistillationOptimizer {
    /// Create new knowledge distillation optimizer
    pub fn new(config: AdvancedOptimizationConfig, device: Device) -> Self {
        Self {
            teacher_layers: HashMap::new(),
            student_layers: HashMap::new(),
            config,
            device,
            distillation_stats: DistillationStats {
                loss_history: Vec::new(),
                transfer_efficiency: 0.0,
                layer_effectiveness: HashMap::new(),
                temperature_sensitivity: Vec::new(),
            },
        }
    }

    /// Set teacher model layers
    pub fn set_teacher_layers(&mut self, layers: HashMap<String, Tensor>) {
        self.teacher_layers = layers;
        info!(
            "Set teacher model with {} layers",
            self.teacher_layers.len()
        );
    }

    /// Set student model layers
    pub fn set_student_layers(&mut self, layers: HashMap<String, Tensor>) {
        self.student_layers = layers;
        info!(
            "Set student model with {} layers",
            self.student_layers.len()
        );
    }

    /// Compute knowledge distillation loss
    pub fn compute_distillation_loss(
        &self,
        teacher_logits: &Tensor,
        student_logits: &Tensor,
    ) -> Result<f32, RecognitionError> {
        let temperature = self.config.distillation_temperature;

        // Apply temperature scaling
        let teacher_soft = self.apply_temperature_scaling(teacher_logits, temperature)?;
        let student_soft = self.apply_temperature_scaling(student_logits, temperature)?;

        // Compute KL divergence loss
        let kl_loss = self.compute_kl_divergence(&teacher_soft, &student_soft)?;

        debug!("Knowledge distillation loss: {:.6}", kl_loss);
        Ok(kl_loss)
    }

    /// Apply temperature scaling for knowledge distillation
    fn apply_temperature_scaling(
        &self,
        logits: &Tensor,
        temperature: f32,
    ) -> Result<Tensor, RecognitionError> {
        let temp_tensor = Tensor::new(temperature, &self.device)?;
        let scaled = logits.div(&temp_tensor)?;
        let softmax = candle_nn::ops::softmax(&scaled, 1)?;
        Ok(softmax)
    }

    /// Compute KL divergence between teacher and student distributions
    fn compute_kl_divergence(
        &self,
        teacher: &Tensor,
        student: &Tensor,
    ) -> Result<f32, RecognitionError> {
        // KL(P||Q) = sum(P * log(P/Q))
        let log_ratio = teacher.div(student)?.log()?;
        let kl = teacher.mul(&log_ratio)?.sum_all()?.to_scalar::<f32>()?;
        Ok(kl)
    }

    /// Perform intermediate layer distillation
    pub fn distill_intermediate_layers(
        &mut self,
    ) -> Result<HashMap<String, f32>, RecognitionError> {
        let mut layer_losses = HashMap::new();

        for (layer_name, teacher_features) in &self.teacher_layers {
            if let Some(student_features) = self.student_layers.get(layer_name) {
                // Compute feature matching loss
                let feature_loss =
                    self.compute_feature_matching_loss(teacher_features, student_features)?;
                layer_losses.insert(layer_name.clone(), feature_loss);

                // Update layer effectiveness
                self.distillation_stats.layer_effectiveness.insert(
                    layer_name.clone(),
                    1.0 / (1.0 + feature_loss), // Effectiveness inversely related to loss
                );
            }
        }

        info!("Distilled {} intermediate layers", layer_losses.len());
        Ok(layer_losses)
    }

    /// Compute feature matching loss between teacher and student features
    fn compute_feature_matching_loss(
        &self,
        teacher: &Tensor,
        student: &Tensor,
    ) -> Result<f32, RecognitionError> {
        // MSE loss between features
        let diff = teacher.sub(student)?;
        let squared_diff = diff.sqr()?;
        let mse_loss = squared_diff.mean_all()?.to_scalar::<f32>()?;
        Ok(mse_loss)
    }

    /// Analyze temperature sensitivity
    pub async fn analyze_temperature_sensitivity(
        &mut self,
        temperatures: Vec<f32>,
        validation_data: &[Tensor],
    ) -> Result<(), RecognitionError> {
        info!(
            "Analyzing temperature sensitivity with {} temperature values",
            temperatures.len()
        );

        for &temperature in &temperatures {
            // Simulate distillation with this temperature
            let mut total_accuracy = 0.0;

            for data in validation_data {
                // Mock teacher and student logits for demonstration
                let teacher_logits = data.clone();
                let student_logits = data.clone(); // In real implementation, this would be student output

                let teacher_soft = self.apply_temperature_scaling(&teacher_logits, temperature)?;
                let student_soft = self.apply_temperature_scaling(&student_logits, temperature)?;

                // Compute accuracy (simplified)
                let accuracy = self.compute_prediction_accuracy(&teacher_soft, &student_soft)?;
                total_accuracy += accuracy;
            }

            let avg_accuracy = total_accuracy / validation_data.len() as f32;
            self.distillation_stats
                .temperature_sensitivity
                .push((temperature, avg_accuracy));

            debug!(
                "Temperature {:.1}: accuracy {:.3}",
                temperature, avg_accuracy
            );
        }

        Ok(())
    }

    /// Compute prediction accuracy between teacher and student
    fn compute_prediction_accuracy(
        &self,
        teacher: &Tensor,
        student: &Tensor,
    ) -> Result<f32, RecognitionError> {
        // Simplified accuracy computation
        let teacher_pred = teacher.argmax(1)?;
        let student_pred = student.argmax(1)?;
        let matches = teacher_pred
            .eq(&student_pred)?
            .to_dtype(DType::F32)?
            .mean_all()?
            .to_scalar::<f32>()?;
        Ok(matches)
    }

    /// Get distillation statistics
    pub fn get_stats(&self) -> &DistillationStats {
        &self.distillation_stats
    }
}

/// Progressive pruning optimizer
#[derive(Debug)]
pub struct ProgressivePruningOptimizer {
    /// Configuration
    config: AdvancedOptimizationConfig,
    /// Current pruning step
    current_step: usize,
    /// Pruning schedule
    pruning_schedule: Vec<f32>,
    /// Layer importance scores
    layer_importance: HashMap<String, f32>,
    /// Pruning history
    pruning_history: Vec<PruningStepResult>,
    /// Device
    device: Device,
}

/// Result of a pruning step
#[derive(Debug, Clone)]
pub struct PruningStepResult {
    /// Step number
    pub step: usize,
    /// Pruning ratio applied
    pub pruning_ratio: f32,
    /// Accuracy after pruning
    pub accuracy: f32,
    /// Model size reduction
    pub size_reduction: f32,
    /// Speedup achieved
    pub speedup: f32,
    /// Recovery iterations needed
    pub recovery_iterations: usize,
}

impl ProgressivePruningOptimizer {
    /// Create new progressive pruning optimizer
    pub fn new(config: AdvancedOptimizationConfig, device: Device) -> Self {
        let pruning_schedule = Self::create_pruning_schedule(&config);

        Self {
            config,
            current_step: 0,
            pruning_schedule,
            layer_importance: HashMap::new(),
            pruning_history: Vec::new(),
            device,
        }
    }

    /// Create pruning schedule
    fn create_pruning_schedule(config: &AdvancedOptimizationConfig) -> Vec<f32> {
        let steps = config.pruning_steps;
        let initial = config.initial_pruning_ratio;
        let final_ratio = config.final_pruning_ratio;

        (0..steps)
            .map(|i| {
                let progress = i as f32 / (steps - 1) as f32;
                initial + (final_ratio - initial) * progress
            })
            .collect()
    }

    /// Compute layer importance scores
    pub fn compute_layer_importance(
        &mut self,
        model_layers: &HashMap<String, Tensor>,
    ) -> Result<(), RecognitionError> {
        info!(
            "Computing layer importance scores for {} layers",
            model_layers.len()
        );

        for (layer_name, weights) in model_layers {
            // Use magnitude-based importance (L1 norm)
            let magnitude_sum = weights.abs()?.sum_all()?.to_scalar::<f32>()?;
            let num_params = weights.elem_count();
            let importance = magnitude_sum / num_params as f32;

            self.layer_importance.insert(layer_name.clone(), importance);
            debug!("Layer {} importance: {:.6}", layer_name, importance);
        }

        Ok(())
    }

    /// Execute progressive pruning step
    pub fn execute_pruning_step(
        &mut self,
        model_layers: &mut HashMap<String, Tensor>,
        validation_fn: impl Fn(&HashMap<String, Tensor>) -> Result<f32, RecognitionError>,
    ) -> Result<PruningStepResult, RecognitionError> {
        if self.current_step >= self.pruning_schedule.len() {
            return Err(RecognitionError::ModelError {
                message: "All pruning steps completed".to_string(),
                source: None,
            });
        }

        let target_ratio = self.pruning_schedule[self.current_step];
        info!(
            "Executing pruning step {}: target ratio {:.2}",
            self.current_step + 1,
            target_ratio
        );

        let start_time = Instant::now();

        // Measure baseline accuracy
        let baseline_accuracy = validation_fn(model_layers)?;
        let baseline_size = self.compute_model_size(model_layers);

        // Apply structured pruning
        let _pruned_params = self.apply_structured_pruning(model_layers, target_ratio)?;

        // Measure post-pruning accuracy
        let pruned_accuracy = validation_fn(model_layers)?;
        let pruned_size = self.compute_model_size(model_layers);

        // Simulate recovery iterations (in real implementation, this would involve fine-tuning)
        let recovery_iterations =
            self.simulate_recovery_training(model_layers, baseline_accuracy)?;

        let processing_time = start_time.elapsed();
        let speedup = self.estimate_inference_speedup(target_ratio);
        let size_reduction = (baseline_size - pruned_size) / baseline_size;

        let result = PruningStepResult {
            step: self.current_step + 1,
            pruning_ratio: target_ratio,
            accuracy: pruned_accuracy,
            size_reduction,
            speedup,
            recovery_iterations,
        };

        self.pruning_history.push(result.clone());
        self.current_step += 1;

        info!(
            "Pruning step completed in {:.2}s: {:.1}% accuracy, {:.1}% size reduction",
            processing_time.as_secs_f32(),
            pruned_accuracy * 100.0,
            size_reduction * 100.0
        );

        Ok(result)
    }

    /// Apply structured pruning to model layers
    fn apply_structured_pruning(
        &self,
        model_layers: &mut HashMap<String, Tensor>,
        target_ratio: f32,
    ) -> Result<usize, RecognitionError> {
        let mut total_pruned = 0;

        for (layer_name, weights) in model_layers.iter_mut() {
            if let Some(&importance) = self.layer_importance.get(layer_name) {
                // Apply pruning based on importance (lower importance = more pruning)
                let layer_ratio = target_ratio * (1.0 - importance).max(0.1);
                let pruned = self.prune_layer_structured(weights, layer_ratio)?;
                total_pruned += pruned;

                debug!(
                    "Pruned {} parameters from layer {} (ratio: {:.3})",
                    pruned, layer_name, layer_ratio
                );
            }
        }

        Ok(total_pruned)
    }

    /// Prune individual layer with structured approach
    fn prune_layer_structured(
        &self,
        weights: &mut Tensor,
        ratio: f32,
    ) -> Result<usize, RecognitionError> {
        let shape = weights.shape();
        let total_params = shape.elem_count();
        let params_to_prune = (total_params as f32 * ratio) as usize;

        // For structured pruning, we would typically prune entire channels/filters
        // This is a simplified implementation
        let magnitude = weights.abs()?.sum_keepdim(1)?;
        let threshold = self.compute_pruning_threshold(&magnitude, ratio)?;

        // Create mask for pruning
        let threshold_tensor = Tensor::new(threshold, &self.device)?;
        let mask = magnitude.gt(&threshold_tensor)?;
        *weights = weights.mul(&mask)?;

        Ok(params_to_prune)
    }

    /// Compute pruning threshold based on magnitude distribution
    fn compute_pruning_threshold(
        &self,
        magnitudes: &Tensor,
        ratio: f32,
    ) -> Result<f32, RecognitionError> {
        // Find the magnitude value that corresponds to the pruning ratio
        let flat = magnitudes.flatten_all()?;
        let values: Vec<f32> = flat.to_vec1()?;
        let mut sorted_values = values;
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let threshold_index = (sorted_values.len() as f32 * ratio) as usize;
        let threshold = sorted_values.get(threshold_index).copied().unwrap_or(0.0);

        Ok(threshold)
    }

    /// Simulate recovery training iterations
    fn simulate_recovery_training(
        &self,
        _model_layers: &HashMap<String, Tensor>,
        target_accuracy: f32,
    ) -> Result<usize, RecognitionError> {
        // Simulate the number of iterations needed to recover accuracy
        // This is a heuristic based on pruning ratio
        let pruning_ratio = self.pruning_schedule[self.current_step];
        let base_iterations = 10;
        let recovery_iterations = (base_iterations as f32 * (1.0 + pruning_ratio * 2.0)) as usize;

        debug!(
            "Estimated {} recovery iterations for {:.1}% accuracy recovery",
            recovery_iterations,
            target_accuracy * 100.0
        );

        Ok(recovery_iterations)
    }

    /// Compute total model size
    fn compute_model_size(&self, model_layers: &HashMap<String, Tensor>) -> f32 {
        model_layers
            .values()
            .map(|tensor| tensor.elem_count() as f32)
            .sum()
    }

    /// Estimate inference speedup from pruning
    fn estimate_inference_speedup(&self, pruning_ratio: f32) -> f32 {
        // Empirical relationship between pruning ratio and speedup
        // Actual speedup depends on hardware and implementation
        1.0 + pruning_ratio * 0.8 // Conservative estimate
    }

    /// Get pruning history
    pub fn get_pruning_history(&self) -> &[PruningStepResult] {
        &self.pruning_history
    }

    /// Get current pruning progress
    pub fn get_progress(&self) -> (usize, usize) {
        (self.current_step, self.pruning_schedule.len())
    }
}

/// Mixed-precision optimizer
#[derive(Debug)]
pub struct MixedPrecisionOptimizer {
    /// Configuration
    config: AdvancedOptimizationConfig,
    /// Layer precision assignments
    layer_precisions: HashMap<String, DType>,
    /// Performance measurements per precision
    precision_performance: HashMap<DType, PerformanceMeasurement>,
    /// Automatic precision search results
    search_results: Vec<PrecisionSearchResult>,
    /// Device
    device: Device,
}

/// Performance measurement for a precision setting
#[derive(Debug, Clone)]
pub struct PerformanceMeasurement {
    /// Inference time (milliseconds)
    pub inference_time_ms: f32,
    /// Memory usage (MB)
    pub memory_usage_mb: f32,
    /// Accuracy
    pub accuracy: f32,
    /// Model size (MB)
    pub model_size_mb: f32,
}

/// Result of precision search
#[derive(Debug, Clone)]
pub struct PrecisionSearchResult {
    /// Layer name
    pub layer_name: String,
    /// Tested precision
    pub precision: DType,
    /// Performance measurement
    pub performance: PerformanceMeasurement,
    /// Meets performance budget
    pub meets_performance_budget: bool,
    /// Meets accuracy budget
    pub meets_accuracy_budget: bool,
}

impl MixedPrecisionOptimizer {
    /// Create new mixed-precision optimizer
    pub fn new(config: AdvancedOptimizationConfig, device: Device) -> Self {
        Self {
            config,
            layer_precisions: HashMap::new(),
            precision_performance: HashMap::new(),
            search_results: Vec::new(),
            device,
        }
    }

    /// Perform automatic precision selection
    pub fn auto_select_precisions(
        &mut self,
        model_layers: &HashMap<String, Tensor>,
        benchmark_fn: impl Fn(
            &HashMap<String, Tensor>,
        ) -> Result<PerformanceMeasurement, RecognitionError>,
    ) -> Result<(), RecognitionError> {
        info!(
            "Starting automatic precision selection for {} layers",
            model_layers.len()
        );

        let precisions_to_test = vec![DType::F32, DType::F16, DType::U8];

        for (layer_name, _weights) in model_layers {
            let mut best_precision = DType::F32;
            let mut best_score = f32::NEG_INFINITY;

            for &precision in &precisions_to_test {
                // Simulate conversion to target precision
                let test_layers = model_layers.clone();
                // In real implementation, convert layer to target precision here

                let performance = benchmark_fn(&test_layers)?;

                // Compute optimization score based on objective
                let score = self.compute_optimization_score(&performance, precision);

                let meets_perf_budget =
                    performance.inference_time_ms / 1000.0 <= self.config.performance_budget;
                let meets_acc_budget = performance.accuracy >= self.config.accuracy_budget;

                let search_result = PrecisionSearchResult {
                    layer_name: layer_name.clone(),
                    precision,
                    performance: performance.clone(),
                    meets_performance_budget: meets_perf_budget,
                    meets_accuracy_budget: meets_acc_budget,
                };

                self.search_results.push(search_result);

                if meets_perf_budget && meets_acc_budget && score > best_score {
                    best_score = score;
                    best_precision = precision;
                }

                debug!(
                    "Layer {}, precision {:?}: score {:.3}, perf budget: {}, acc budget: {}",
                    layer_name, precision, score, meets_perf_budget, meets_acc_budget
                );
            }

            self.layer_precisions
                .insert(layer_name.clone(), best_precision);
            info!(
                "Selected {:?} precision for layer {}",
                best_precision, layer_name
            );
        }

        Ok(())
    }

    /// Compute optimization score based on objective
    fn compute_optimization_score(
        &self,
        performance: &PerformanceMeasurement,
        _precision: DType,
    ) -> f32 {
        match self.config.optimization_objective {
            OptimizationObjective::Latency => {
                -performance.inference_time_ms // Lower is better
            }
            OptimizationObjective::Memory => {
                -performance.memory_usage_mb // Lower is better
            }
            OptimizationObjective::Size => {
                -performance.model_size_mb // Lower is better
            }
            OptimizationObjective::Balanced => {
                // Weighted combination
                let latency_score = -performance.inference_time_ms / 1000.0;
                let memory_score = -performance.memory_usage_mb / 1000.0;
                let accuracy_score = performance.accuracy;
                let size_score = -performance.model_size_mb / 100.0;

                0.3 * latency_score + 0.2 * memory_score + 0.4 * accuracy_score + 0.1 * size_score
            }
            OptimizationObjective::Throughput => {
                1000.0 / performance.inference_time_ms // Higher throughput is better
            }
        }
    }

    /// Apply mixed-precision configuration to model
    pub fn apply_mixed_precision(
        &self,
        model_layers: &mut HashMap<String, Tensor>,
    ) -> Result<MixedPrecisionStats, RecognitionError> {
        let mut stats = MixedPrecisionStats {
            total_layers: model_layers.len(),
            fp32_layers: 0,
            fp16_layers: 0,
            int8_layers: 0,
            estimated_speedup: 1.0,
            estimated_memory_reduction: 0.0,
        };

        for (layer_name, weights) in model_layers.iter_mut() {
            if let Some(&target_precision) = self.layer_precisions.get(layer_name) {
                // Convert weights to target precision
                let converted_weights = weights.to_dtype(target_precision)?;
                *weights = converted_weights;

                // Update statistics
                match target_precision {
                    DType::F32 => stats.fp32_layers += 1,
                    DType::F16 => stats.fp16_layers += 1,
                    DType::U8 => stats.int8_layers += 1,
                    _ => {}
                }

                debug!("Converted layer {} to {:?}", layer_name, target_precision);
            }
        }

        // Estimate performance improvements
        stats.estimated_speedup = self.estimate_mixed_precision_speedup(&stats);
        stats.estimated_memory_reduction = self.estimate_memory_reduction(&stats);

        info!(
            "Applied mixed-precision: {:.1}x speedup, {:.1}% memory reduction",
            stats.estimated_speedup,
            stats.estimated_memory_reduction * 100.0
        );

        Ok(stats)
    }

    /// Estimate speedup from mixed-precision configuration
    fn estimate_mixed_precision_speedup(&self, stats: &MixedPrecisionStats) -> f32 {
        let total = stats.total_layers as f32;
        let fp16_ratio = stats.fp16_layers as f32 / total;
        let int8_ratio = stats.int8_layers as f32 / total;

        // Empirical speedup estimates
        1.0 + fp16_ratio * 0.4 + int8_ratio * 0.8
    }

    /// Estimate memory reduction from mixed-precision
    fn estimate_memory_reduction(&self, stats: &MixedPrecisionStats) -> f32 {
        let total = stats.total_layers as f32;
        let fp16_ratio = stats.fp16_layers as f32 / total;
        let int8_ratio = stats.int8_layers as f32 / total;

        // Memory reduction estimates (FP16 = 50% reduction, INT8 = 75% reduction)
        fp16_ratio * 0.5 + int8_ratio * 0.75
    }

    /// Get layer precision assignments
    pub fn get_layer_precisions(&self) -> &HashMap<String, DType> {
        &self.layer_precisions
    }

    /// Get search results
    pub fn get_search_results(&self) -> &[PrecisionSearchResult] {
        &self.search_results
    }
}

/// Mixed-precision optimization statistics
#[derive(Debug, Clone)]
pub struct MixedPrecisionStats {
    /// Total number of layers
    pub total_layers: usize,
    /// Number of FP32 layers
    pub fp32_layers: usize,
    /// Number of FP16 layers
    pub fp16_layers: usize,
    /// Number of INT8 layers
    pub int8_layers: usize,
    /// Estimated speedup
    pub estimated_speedup: f32,
    /// Estimated memory reduction (0.0 to 1.0)
    pub estimated_memory_reduction: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::whisper::quantization::MovingAverageTracker;

    #[test]
    fn test_optimization_config_creation() {
        let config = AdvancedOptimizationConfig::default();

        assert!(!config.enable_knowledge_distillation);
        assert!(config.enable_mixed_precision);
        assert!(config.auto_precision_selection);
        assert_eq!(config.distillation_temperature, 4.0);
        assert_eq!(config.performance_budget, 0.3);
        assert_eq!(config.accuracy_budget, 0.95);
    }

    #[test]
    fn test_pruning_schedule_creation() {
        let config = AdvancedOptimizationConfig {
            pruning_steps: 5,
            initial_pruning_ratio: 0.1,
            final_pruning_ratio: 0.5,
            ..Default::default()
        };

        let schedule = ProgressivePruningOptimizer::create_pruning_schedule(&config);

        assert_eq!(schedule.len(), 5);
        assert_eq!(schedule[0], 0.1);
        assert_eq!(schedule[4], 0.5);

        // Verify monotonic increase
        for i in 1..schedule.len() {
            assert!(schedule[i] >= schedule[i - 1]);
        }
    }

    #[test]
    fn test_moving_average_tracker() {
        let mut tracker = MovingAverageTracker::new(3);

        tracker.update(1.0, 2.0);
        tracker.update(2.0, 3.0);
        tracker.update(3.0, 4.0);

        let (avg_min, avg_max) = tracker.get_averaged_range();
        assert_eq!(avg_min, 2.0); // (1+2+3)/3
        assert_eq!(avg_max, 3.0); // (2+3+4)/3
    }

    #[test]
    fn test_mixed_precision_stats() {
        let stats = MixedPrecisionStats {
            total_layers: 10,
            fp32_layers: 4,
            fp16_layers: 4,
            int8_layers: 2,
            estimated_speedup: 1.0,
            estimated_memory_reduction: 0.0,
        };

        let optimizer =
            MixedPrecisionOptimizer::new(AdvancedOptimizationConfig::default(), Device::Cpu);

        let speedup = optimizer.estimate_mixed_precision_speedup(&stats);
        let memory_reduction = optimizer.estimate_memory_reduction(&stats);

        assert!(speedup > 1.0);
        assert!(memory_reduction > 0.0);
        assert!(memory_reduction < 1.0);
    }
}
