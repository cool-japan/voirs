//! Performance optimizer for automated tuning and recommendations
//!
//! This module analyzes performance metrics and applies optimizations automatically
//! or provides detailed recommendations for manual optimization.

use super::{
    GpuMetrics, MemoryMetrics, OptimizationCategory, OptimizationRecommendation,
    PerformanceMetrics, SynthesisMetrics, SystemMetrics,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Performance optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Enable automatic optimizations
    pub auto_optimize: bool,
    /// Minimum improvement threshold to trigger optimization
    pub min_improvement_threshold: f64,
    /// Maximum optimization attempts per session
    pub max_optimization_attempts: u32,
    /// Optimization target priority
    pub optimization_targets: Vec<OptimizationTarget>,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
    /// Optimization aggressiveness (1-10)
    pub aggressiveness: u8,
}

/// Optimization targets with priorities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationTarget {
    /// Target category
    pub category: OptimizationCategory,
    /// Priority weight (0.0-1.0)
    pub weight: f64,
    /// Enable automatic optimization for this target
    pub auto_optimize: bool,
}

/// Resource constraints for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Maximum memory usage in MB
    pub max_memory_mb: Option<u64>,
    /// Maximum CPU usage percentage
    pub max_cpu_percent: Option<f64>,
    /// Maximum GPU memory usage in MB
    pub max_gpu_memory_mb: Option<u64>,
    /// Minimum disk space in MB
    pub min_disk_space_mb: Option<u64>,
    /// Target latency in milliseconds
    pub target_latency_ms: Option<f64>,
}

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Applied optimizations
    pub applied: Vec<AppliedOptimization>,
    /// Performance improvement achieved
    pub improvement: PerformanceImprovement,
    /// Time taken to apply optimizations
    pub optimization_time: Duration,
    /// Whether optimization was successful
    pub success: bool,
    /// Error message if optimization failed
    pub error: Option<String>,
}

/// Details of an applied optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedOptimization {
    /// Optimization category
    pub category: OptimizationCategory,
    /// Description of what was changed
    pub description: String,
    /// Previous value/setting
    pub previous_value: String,
    /// New value/setting
    pub new_value: String,
    /// Expected improvement
    pub expected_improvement: f64,
    /// Actual improvement measured
    pub actual_improvement: Option<f64>,
}

/// Performance improvement metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImprovement {
    /// CPU usage improvement (positive = better)
    pub cpu_improvement: f64,
    /// Memory usage improvement (positive = better)
    pub memory_improvement: f64,
    /// Synthesis speed improvement (positive = better)
    pub speed_improvement: f64,
    /// Overall performance score improvement
    pub overall_improvement: f64,
    /// Real-time factor improvement
    pub rtf_improvement: f64,
}

/// Performance optimizer
pub struct PerformanceOptimizer {
    /// Optimizer configuration
    config: OptimizerConfig,
    /// Applied optimizations history
    optimization_history: Vec<OptimizationResult>,
    /// Current optimization settings
    current_settings: HashMap<String, String>,
    /// Performance baseline
    baseline_metrics: Option<PerformanceMetrics>,
}

impl PerformanceOptimizer {
    /// Create a new performance optimizer
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            optimization_history: Vec::new(),
            current_settings: HashMap::new(),
            baseline_metrics: None,
        }
    }

    /// Set performance baseline
    pub fn set_baseline(&mut self, metrics: PerformanceMetrics) {
        self.baseline_metrics = Some(metrics);
    }

    /// Analyze metrics and generate optimization recommendations
    pub async fn analyze_and_recommend(
        &self,
        current_metrics: &PerformanceMetrics,
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Analyze each optimization target
        for target in &self.config.optimization_targets {
            match target.category {
                OptimizationCategory::Memory => {
                    self.analyze_memory_optimization(
                        &mut recommendations,
                        &current_metrics.memory,
                        &current_metrics.system,
                        target.weight,
                    )
                    .await;
                }
                OptimizationCategory::Cpu => {
                    self.analyze_cpu_optimization(
                        &mut recommendations,
                        &current_metrics.system,
                        &current_metrics.synthesis,
                        target.weight,
                    )
                    .await;
                }
                OptimizationCategory::Gpu => {
                    if let Some(ref gpu_metrics) = current_metrics.gpu {
                        self.analyze_gpu_optimization(
                            &mut recommendations,
                            gpu_metrics,
                            &current_metrics.synthesis,
                            target.weight,
                        )
                        .await;
                    }
                }
                OptimizationCategory::Parallelization => {
                    self.analyze_parallelization_optimization(
                        &mut recommendations,
                        &current_metrics.synthesis,
                        &current_metrics.system,
                        target.weight,
                    )
                    .await;
                }
                OptimizationCategory::Caching => {
                    self.analyze_caching_optimization(
                        &mut recommendations,
                        &current_metrics.memory,
                        &current_metrics.synthesis,
                        target.weight,
                    )
                    .await;
                }
                OptimizationCategory::ModelOptimization => {
                    self.analyze_model_optimization(
                        &mut recommendations,
                        &current_metrics.synthesis,
                        &current_metrics.system,
                        target.weight,
                    )
                    .await;
                }
                _ => {
                    // Handle other categories
                    self.analyze_general_optimization(
                        &mut recommendations,
                        current_metrics,
                        &target.category,
                        target.weight,
                    )
                    .await;
                }
            }
        }

        // Sort by priority and filter by improvement threshold
        recommendations.sort_by(|a, b| {
            b.priority.cmp(&a.priority).then(
                b.performance_impact
                    .partial_cmp(&a.performance_impact)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });

        recommendations
            .into_iter()
            .filter(|r| r.performance_impact >= self.config.min_improvement_threshold)
            .take(10) // Limit to top 10 recommendations
            .collect()
    }

    /// Apply automatic optimizations
    pub async fn apply_optimizations(
        &mut self,
        recommendations: &[OptimizationRecommendation],
    ) -> OptimizationResult {
        let start_time = std::time::Instant::now();
        let mut applied = Vec::new();
        let mut total_improvement = 0.0;

        for recommendation in recommendations {
            // Check if auto-optimization is enabled for this category
            let auto_optimize = self
                .config
                .optimization_targets
                .iter()
                .find(|t| t.category == recommendation.category)
                .map(|t| t.auto_optimize)
                .unwrap_or(false);

            if !auto_optimize || !self.config.auto_optimize {
                continue;
            }

            if let Ok(optimization) = self.apply_single_optimization(recommendation).await {
                total_improvement += optimization.expected_improvement;
                applied.push(optimization);
            }
        }

        let optimization_time = start_time.elapsed();

        OptimizationResult {
            applied,
            improvement: PerformanceImprovement {
                cpu_improvement: total_improvement * 0.3,
                memory_improvement: total_improvement * 0.2,
                speed_improvement: total_improvement * 0.4,
                overall_improvement: total_improvement,
                rtf_improvement: total_improvement * 0.5,
            },
            optimization_time,
            success: total_improvement > 0.0,
            error: None,
        }
    }

    /// Apply a single optimization
    async fn apply_single_optimization(
        &mut self,
        recommendation: &OptimizationRecommendation,
    ) -> Result<AppliedOptimization, Box<dyn std::error::Error>> {
        let (setting_key, previous_value, new_value) = match recommendation.category {
            OptimizationCategory::Memory => self.apply_memory_optimization(recommendation).await?,
            OptimizationCategory::Cpu => self.apply_cpu_optimization(recommendation).await?,
            OptimizationCategory::Gpu => self.apply_gpu_optimization(recommendation).await?,
            OptimizationCategory::Parallelization => {
                self.apply_parallelization_optimization(recommendation)
                    .await?
            }
            OptimizationCategory::Caching => {
                self.apply_caching_optimization(recommendation).await?
            }
            OptimizationCategory::ModelOptimization => {
                self.apply_model_optimization(recommendation).await?
            }
            _ => {
                return Err("Optimization category not implemented".into());
            }
        };

        // Update current settings
        self.current_settings
            .insert(setting_key.clone(), new_value.clone());

        Ok(AppliedOptimization {
            category: recommendation.category.clone(),
            description: recommendation.recommendation.clone(),
            previous_value,
            new_value,
            expected_improvement: recommendation.performance_impact,
            actual_improvement: None, // Will be measured later
        })
    }

    /// Memory optimization analysis
    async fn analyze_memory_optimization(
        &self,
        recommendations: &mut Vec<OptimizationRecommendation>,
        memory: &MemoryMetrics,
        system: &SystemMetrics,
        weight: f64,
    ) {
        let memory_usage_percent = (memory.heap_used as f64
            / (system.memory_used + system.memory_available) as f64)
            * 100.0;

        if memory_usage_percent > 75.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Memory,
                priority: (8.0 * weight) as u8,
                description: format!("High memory usage: {:.1}%", memory_usage_percent),
                recommendation:
                    "Enable memory optimization: reduce batch size, enable streaming processing"
                        .to_string(),
                expected_improvement: format!("{:.0}% memory reduction", 30.0 * weight),
                difficulty: 2,
                performance_impact: 0.3 * weight,
            });
        }

        if memory.fragmentation_percent > 20.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Memory,
                priority: (6.0 * weight) as u8,
                description: format!("Memory fragmentation: {:.1}%", memory.fragmentation_percent),
                recommendation: "Enable memory pool allocation".to_string(),
                expected_improvement: format!("{:.0}% fragmentation reduction", 50.0 * weight),
                difficulty: 3,
                performance_impact: 0.15 * weight,
            });
        }

        if memory.cache_hit_rate < 60.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Caching,
                priority: (7.0 * weight) as u8,
                description: format!("Low cache hit rate: {:.1}%", memory.cache_hit_rate),
                recommendation: "Increase cache size and enable aggressive caching".to_string(),
                expected_improvement: format!(
                    "{:.0}% cache performance improvement",
                    40.0 * weight
                ),
                difficulty: 2,
                performance_impact: 0.25 * weight,
            });
        }
    }

    /// CPU optimization analysis
    async fn analyze_cpu_optimization(
        &self,
        recommendations: &mut Vec<OptimizationRecommendation>,
        system: &SystemMetrics,
        synthesis: &SynthesisMetrics,
        weight: f64,
    ) {
        if system.cpu_usage > 90.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Cpu,
                priority: (9.0 * weight) as u8,
                description: format!("CPU overload: {:.1}%", system.cpu_usage),
                recommendation:
                    "Reduce parallel threads, enable GPU acceleration, or use lower quality"
                        .to_string(),
                expected_improvement: format!("{:.0}% CPU usage reduction", 40.0 * weight),
                difficulty: 2,
                performance_impact: 0.4 * weight,
            });
        } else if system.cpu_usage < 30.0 && synthesis.real_time_factor > 2.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Parallelization,
                priority: (6.0 * weight) as u8,
                description: format!("CPU underutilization: {:.1}%", system.cpu_usage),
                recommendation: "Increase parallel processing threads".to_string(),
                expected_improvement: format!("{:.0}% throughput increase", 30.0 * weight),
                difficulty: 1,
                performance_impact: 0.3 * weight,
            });
        }

        if synthesis.real_time_factor < 1.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::ModelOptimization,
                priority: (8.0 * weight) as u8,
                description: format!(
                    "Poor real-time performance: {:.2}x RTF",
                    synthesis.real_time_factor
                ),
                recommendation: "Use quantized models or enable GPU acceleration".to_string(),
                expected_improvement: "Achieve real-time synthesis".to_string(),
                difficulty: 4,
                performance_impact: 0.5 * weight,
            });
        }
    }

    /// GPU optimization analysis
    async fn analyze_gpu_optimization(
        &self,
        recommendations: &mut Vec<OptimizationRecommendation>,
        gpu: &GpuMetrics,
        synthesis: &SynthesisMetrics,
        weight: f64,
    ) {
        if gpu.utilization < 40.0 && synthesis.real_time_factor < 2.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Gpu,
                priority: (7.0 * weight) as u8,
                description: format!("Low GPU utilization: {:.1}%", gpu.utilization),
                recommendation: "Increase batch size or use larger models".to_string(),
                expected_improvement: format!("{:.0}% GPU utilization improvement", 50.0 * weight),
                difficulty: 2,
                performance_impact: 0.3 * weight,
            });
        }

        let gpu_memory_usage = (gpu.memory_used as f64 / gpu.memory_total as f64) * 100.0;
        if gpu_memory_usage > 85.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Gpu,
                priority: (8.0 * weight) as u8,
                description: format!("High GPU memory usage: {:.1}%", gpu_memory_usage),
                recommendation: "Reduce batch size or enable model quantization".to_string(),
                expected_improvement: format!("{:.0}% GPU memory reduction", 30.0 * weight),
                difficulty: 3,
                performance_impact: 0.2 * weight,
            });
        }

        if gpu.temperature > 80.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Gpu,
                priority: (9.0 * weight) as u8,
                description: format!("High GPU temperature: {:.1}°C", gpu.temperature),
                recommendation: "Reduce GPU workload or improve cooling".to_string(),
                expected_improvement: "Prevent thermal throttling".to_string(),
                difficulty: 4,
                performance_impact: 0.25 * weight,
            });
        }
    }

    /// Parallelization optimization analysis
    async fn analyze_parallelization_optimization(
        &self,
        recommendations: &mut Vec<OptimizationRecommendation>,
        synthesis: &SynthesisMetrics,
        system: &SystemMetrics,
        weight: f64,
    ) {
        let available_cores = system.thread_count;
        let estimated_utilization = system.cpu_usage / 100.0;

        if estimated_utilization < 0.6 && synthesis.queue_depth > 5 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Parallelization,
                priority: (7.0 * weight) as u8,
                description: format!(
                    "Suboptimal parallelization: {} cores available",
                    available_cores
                ),
                recommendation: format!(
                    "Increase worker threads to {}",
                    (available_cores as f64 * 0.8) as usize
                ),
                expected_improvement: format!("{:.0}% throughput improvement", 40.0 * weight),
                difficulty: 2,
                performance_impact: 0.35 * weight,
            });
        }

        if synthesis.avg_synthesis_time_ms > 1000.0 && synthesis.queue_depth < 2 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Parallelization,
                priority: (6.0 * weight) as u8,
                description: "Sequential processing detected".to_string(),
                recommendation: "Enable batch processing for improved efficiency".to_string(),
                expected_improvement: format!("{:.0}% processing time reduction", 25.0 * weight),
                difficulty: 2,
                performance_impact: 0.25 * weight,
            });
        }
    }

    /// Caching optimization analysis
    async fn analyze_caching_optimization(
        &self,
        recommendations: &mut Vec<OptimizationRecommendation>,
        memory: &MemoryMetrics,
        synthesis: &SynthesisMetrics,
        weight: f64,
    ) {
        if memory.cache_hit_rate < 70.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Caching,
                priority: (7.0 * weight) as u8,
                description: format!(
                    "Low cache efficiency: {:.1}% hit rate",
                    memory.cache_hit_rate
                ),
                recommendation: "Increase cache size and implement model preloading".to_string(),
                expected_improvement: format!("{:.0}% cache hit rate improvement", 30.0 * weight),
                difficulty: 3,
                performance_impact: 0.3 * weight,
            });
        }

        if synthesis.avg_synthesis_time_ms > 500.0 && memory.cache_hit_rate > 80.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Caching,
                priority: (5.0 * weight) as u8,
                description: "Opportunity for aggressive caching".to_string(),
                recommendation: "Enable result caching for repeated synthesis".to_string(),
                expected_improvement: format!("{:.0}% synthesis speed improvement", 50.0 * weight),
                difficulty: 3,
                performance_impact: 0.4 * weight,
            });
        }
    }

    /// Model optimization analysis
    async fn analyze_model_optimization(
        &self,
        recommendations: &mut Vec<OptimizationRecommendation>,
        synthesis: &SynthesisMetrics,
        system: &SystemMetrics,
        weight: f64,
    ) {
        if synthesis.real_time_factor < 1.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::ModelOptimization,
                priority: (9.0 * weight) as u8,
                description: format!(
                    "Below real-time performance: {:.2}x",
                    synthesis.real_time_factor
                ),
                recommendation: "Use quantized models (INT8/FP16) for faster inference".to_string(),
                expected_improvement: format!("{:.0}x speed improvement", 2.0 * weight),
                difficulty: 4,
                performance_impact: 0.6 * weight,
            });
        }

        if synthesis.memory_per_operation_mb > 1000.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::ModelOptimization,
                priority: (7.0 * weight) as u8,
                description: format!(
                    "High memory per operation: {:.0} MB",
                    synthesis.memory_per_operation_mb
                ),
                recommendation: "Use model pruning or distillation for smaller footprint"
                    .to_string(),
                expected_improvement: format!("{:.0}% memory reduction", 40.0 * weight),
                difficulty: 5,
                performance_impact: 0.3 * weight,
            });
        }

        let memory_usage_percent =
            (synthesis.memory_per_operation_mb * synthesis.queue_depth as f64 * 1024.0 * 1024.0)
                / (system.memory_used + system.memory_available) as f64
                * 100.0;

        if memory_usage_percent > 50.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::ModelOptimization,
                priority: (8.0 * weight) as u8,
                description: "Models consuming excessive memory".to_string(),
                recommendation: "Use smaller model variants or enable model streaming".to_string(),
                expected_improvement: format!(
                    "{:.0}% memory efficiency improvement",
                    35.0 * weight
                ),
                difficulty: 4,
                performance_impact: 0.35 * weight,
            });
        }
    }

    /// General optimization analysis for other categories
    async fn analyze_general_optimization(
        &self,
        recommendations: &mut Vec<OptimizationRecommendation>,
        metrics: &PerformanceMetrics,
        category: &OptimizationCategory,
        weight: f64,
    ) {
        match category {
            OptimizationCategory::Io => {
                if metrics.system.disk_read_bps > 50_000_000
                    || metrics.system.disk_write_bps > 50_000_000
                {
                    recommendations.push(OptimizationRecommendation {
                        category: OptimizationCategory::Io,
                        priority: (6.0 * weight) as u8,
                        description: "High disk I/O detected".to_string(),
                        recommendation: "Use SSD storage or enable I/O optimization".to_string(),
                        expected_improvement: format!(
                            "{:.0}% I/O performance improvement",
                            40.0 * weight
                        ),
                        difficulty: 3,
                        performance_impact: 0.3 * weight,
                    });
                }
            }
            OptimizationCategory::Network => {
                if metrics.system.network_bps > 100_000_000 {
                    recommendations.push(OptimizationRecommendation {
                        category: OptimizationCategory::Network,
                        priority: (5.0 * weight) as u8,
                        description: "High network usage detected".to_string(),
                        recommendation: "Enable compression or local caching".to_string(),
                        expected_improvement: format!(
                            "{:.0}% network efficiency improvement",
                            25.0 * weight
                        ),
                        difficulty: 3,
                        performance_impact: 0.2 * weight,
                    });
                }
            }
            OptimizationCategory::Configuration => {
                recommendations.push(OptimizationRecommendation {
                    category: OptimizationCategory::Configuration,
                    priority: (4.0 * weight) as u8,
                    description: "Configuration optimization available".to_string(),
                    recommendation: "Review and optimize system configuration".to_string(),
                    expected_improvement: format!("{:.0}% overall improvement", 15.0 * weight),
                    difficulty: 2,
                    performance_impact: 0.15 * weight,
                });
            }
            OptimizationCategory::ResourceAllocation => {
                if metrics.synthesis.queue_depth > 10 {
                    recommendations.push(OptimizationRecommendation {
                        category: OptimizationCategory::ResourceAllocation,
                        priority: (7.0 * weight) as u8,
                        description: format!("High queue depth: {}", metrics.synthesis.queue_depth),
                        recommendation: "Optimize resource allocation and scheduling".to_string(),
                        expected_improvement: format!("{:.0}% latency reduction", 30.0 * weight),
                        difficulty: 3,
                        performance_impact: 0.25 * weight,
                    });
                }
            }
            _ => {} // Already handled in specific methods
        }
    }

    /// Apply memory optimization
    async fn apply_memory_optimization(
        &mut self,
        recommendation: &OptimizationRecommendation,
    ) -> Result<(String, String, String), Box<dyn std::error::Error>> {
        if recommendation.description.contains("High memory usage") {
            let key = "batch_size".to_string();
            let previous = self
                .current_settings
                .get(&key)
                .unwrap_or(&"32".to_string())
                .clone();
            let new_value = (previous.parse::<u32>()? / 2).to_string();
            Ok((key, previous, new_value))
        } else if recommendation.description.contains("fragmentation") {
            let key = "memory_pool".to_string();
            let previous = self
                .current_settings
                .get(&key)
                .unwrap_or(&"false".to_string())
                .clone();
            let new_value = "true".to_string();
            Ok((key, previous, new_value))
        } else {
            Err("Unknown memory optimization".into())
        }
    }

    /// Apply CPU optimization
    async fn apply_cpu_optimization(
        &mut self,
        recommendation: &OptimizationRecommendation,
    ) -> Result<(String, String, String), Box<dyn std::error::Error>> {
        if recommendation.description.contains("CPU overload") {
            let key = "max_threads".to_string();
            let previous = self
                .current_settings
                .get(&key)
                .unwrap_or(&"8".to_string())
                .clone();
            let new_value = (previous.parse::<u32>()?.saturating_sub(2)).to_string();
            Ok((key, previous, new_value))
        } else {
            Err("Unknown CPU optimization".into())
        }
    }

    /// Apply GPU optimization
    async fn apply_gpu_optimization(
        &mut self,
        recommendation: &OptimizationRecommendation,
    ) -> Result<(String, String, String), Box<dyn std::error::Error>> {
        if recommendation.description.contains("Low GPU utilization") {
            let key = "gpu_batch_size".to_string();
            let previous = self
                .current_settings
                .get(&key)
                .unwrap_or(&"16".to_string())
                .clone();
            let new_value = (previous.parse::<u32>()? * 2).to_string();
            Ok((key, previous, new_value))
        } else if recommendation.description.contains("High GPU memory") {
            let key = "gpu_batch_size".to_string();
            let previous = self
                .current_settings
                .get(&key)
                .unwrap_or(&"16".to_string())
                .clone();
            let new_value = (previous.parse::<u32>()? / 2).to_string();
            Ok((key, previous, new_value))
        } else {
            Err("Unknown GPU optimization".into())
        }
    }

    /// Apply parallelization optimization
    async fn apply_parallelization_optimization(
        &mut self,
        recommendation: &OptimizationRecommendation,
    ) -> Result<(String, String, String), Box<dyn std::error::Error>> {
        if recommendation.recommendation.contains("worker threads") {
            let key = "worker_threads".to_string();
            let previous = self
                .current_settings
                .get(&key)
                .unwrap_or(&"4".to_string())
                .clone();
            let new_value = (previous.parse::<u32>()? + 2).to_string();
            Ok((key, previous, new_value))
        } else {
            Err("Unknown parallelization optimization".into())
        }
    }

    /// Apply caching optimization
    async fn apply_caching_optimization(
        &mut self,
        recommendation: &OptimizationRecommendation,
    ) -> Result<(String, String, String), Box<dyn std::error::Error>> {
        if recommendation.description.contains("cache") {
            let key = "cache_size_mb".to_string();
            let previous = self
                .current_settings
                .get(&key)
                .unwrap_or(&"256".to_string())
                .clone();
            let new_value = (previous.parse::<u32>()? * 2).to_string();
            Ok((key, previous, new_value))
        } else {
            Err("Unknown caching optimization".into())
        }
    }

    /// Apply model optimization
    async fn apply_model_optimization(
        &mut self,
        recommendation: &OptimizationRecommendation,
    ) -> Result<(String, String, String), Box<dyn std::error::Error>> {
        if recommendation.recommendation.contains("quantized") {
            let key = "model_precision".to_string();
            let previous = self
                .current_settings
                .get(&key)
                .unwrap_or(&"fp32".to_string())
                .clone();
            let new_value = "fp16".to_string();
            Ok((key, previous, new_value))
        } else {
            Err("Unknown model optimization".into())
        }
    }

    /// Get optimization history
    pub fn get_optimization_history(&self) -> &[OptimizationResult] {
        &self.optimization_history
    }

    /// Get current settings
    pub fn get_current_settings(&self) -> &HashMap<String, String> {
        &self.current_settings
    }

    /// Update configuration
    pub fn update_config(&mut self, config: OptimizerConfig) {
        self.config = config;
    }
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            auto_optimize: false,
            min_improvement_threshold: 0.1,
            max_optimization_attempts: 5,
            optimization_targets: vec![
                OptimizationTarget {
                    category: OptimizationCategory::Memory,
                    weight: 0.8,
                    auto_optimize: true,
                },
                OptimizationTarget {
                    category: OptimizationCategory::Cpu,
                    weight: 0.9,
                    auto_optimize: true,
                },
                OptimizationTarget {
                    category: OptimizationCategory::Gpu,
                    weight: 0.7,
                    auto_optimize: false,
                },
            ],
            resource_constraints: ResourceConstraints::default(),
            aggressiveness: 5,
        }
    }
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_memory_mb: Some(8192),
            max_cpu_percent: Some(80.0),
            max_gpu_memory_mb: Some(6144),
            min_disk_space_mb: Some(1024),
            target_latency_ms: Some(500.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_config_default() {
        let config = OptimizerConfig::default();
        assert!(!config.auto_optimize);
        assert_eq!(config.aggressiveness, 5);
        assert!(config.optimization_targets.len() > 0);
    }

    #[tokio::test]
    async fn test_performance_optimizer_creation() {
        let config = OptimizerConfig::default();
        let optimizer = PerformanceOptimizer::new(config);
        assert_eq!(optimizer.optimization_history.len(), 0);
        assert_eq!(optimizer.current_settings.len(), 0);
    }

    #[tokio::test]
    async fn test_memory_optimization_analysis() {
        let config = OptimizerConfig::default();
        let optimizer = PerformanceOptimizer::new(config);

        let mut recommendations = Vec::new();
        let memory = MemoryMetrics {
            heap_used: 8_000_000_000, // 8GB
            fragmentation_percent: 25.0,
            cache_hit_rate: 50.0,
            ..Default::default()
        };
        let system = SystemMetrics {
            memory_used: 8_000_000_000,
            memory_available: 2_000_000_000,
            ..Default::default()
        };

        optimizer
            .analyze_memory_optimization(&mut recommendations, &memory, &system, 1.0)
            .await;

        assert!(recommendations.len() > 0);
        assert!(recommendations
            .iter()
            .any(|r| r.category == OptimizationCategory::Memory));
    }

    #[tokio::test]
    async fn test_cpu_optimization_analysis() {
        let config = OptimizerConfig::default();
        let optimizer = PerformanceOptimizer::new(config);

        let mut recommendations = Vec::new();
        let system = SystemMetrics {
            cpu_usage: 95.0, // High CPU usage
            ..Default::default()
        };
        let synthesis = SynthesisMetrics {
            real_time_factor: 0.5, // Poor RTF
            ..Default::default()
        };

        optimizer
            .analyze_cpu_optimization(&mut recommendations, &system, &synthesis, 1.0)
            .await;

        assert!(recommendations.len() > 0);
        assert!(recommendations
            .iter()
            .any(|r| r.category == OptimizationCategory::Cpu
                || r.category == OptimizationCategory::ModelOptimization));
    }

    #[tokio::test]
    async fn test_optimization_recommendation_sorting() {
        let config = OptimizerConfig::default();
        let optimizer = PerformanceOptimizer::new(config);

        let metrics = PerformanceMetrics {
            system: SystemMetrics {
                cpu_usage: 95.0,
                memory_used: 8_000_000_000,
                memory_available: 2_000_000_000,
                ..Default::default()
            },
            memory: MemoryMetrics {
                heap_used: 8_000_000_000,
                cache_hit_rate: 50.0,
                ..Default::default()
            },
            synthesis: SynthesisMetrics {
                real_time_factor: 0.5,
                ..Default::default()
            },
            ..Default::default()
        };

        let recommendations = optimizer.analyze_and_recommend(&metrics).await;

        assert!(!recommendations.is_empty());

        // Check that recommendations are sorted by priority
        for i in 1..recommendations.len() {
            assert!(recommendations[i - 1].priority >= recommendations[i].priority);
        }
    }
}
