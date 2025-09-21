//! Real-time optimization and self-tuning for G2P systems.
//!
//! This module provides advanced optimization capabilities including
//! dynamic parameter tuning, performance monitoring, and automatic
//! model adaptation based on real-world usage patterns.

use crate::{G2pError, LanguageCode, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

/// Real-time performance optimizer for G2P systems
pub struct RealTimeOptimizer {
    /// Language being optimized
    pub language: LanguageCode,
    /// Performance metrics collector
    pub metrics_collector: Arc<Mutex<MetricsCollector>>,
    /// Optimization strategies
    pub optimization_strategies: Vec<Box<dyn OptimizationStrategy>>,
    /// Current optimization state
    pub optimization_state: OptimizationState,
    /// Optimization history
    pub optimization_history: VecDeque<OptimizationEntry>,
    /// Configuration
    pub config: OptimizerConfig,
}

/// Performance metrics collector
#[derive(Debug, Clone, Default)]
pub struct MetricsCollector {
    /// Latency measurements
    pub latency_measurements: VecDeque<Duration>,
    /// Accuracy measurements
    pub accuracy_measurements: VecDeque<f32>,
    /// Throughput measurements
    pub throughput_measurements: VecDeque<f32>,
    /// Error rates
    pub error_rates: VecDeque<f32>,
    /// Memory usage measurements
    pub memory_usage: VecDeque<u64>,
    /// CPU usage measurements
    pub cpu_usage: VecDeque<f32>,
    /// Quality scores
    pub quality_scores: VecDeque<f32>,
    /// User satisfaction scores
    pub user_satisfaction: VecDeque<f32>,
}

/// Optimization strategy trait
pub trait OptimizationStrategy: Send + Sync {
    /// Strategy name
    fn name(&self) -> &str;

    /// Analyze current performance metrics
    fn analyze_metrics(&self, metrics: &MetricsCollector) -> OptimizationRecommendation;

    /// Apply optimization based on recommendation
    fn apply_optimization(&self, recommendation: &OptimizationRecommendation) -> Result<()>;

    /// Check if strategy is applicable for current conditions
    fn is_applicable(&self, metrics: &MetricsCollector) -> bool;

    /// Get strategy priority (higher = more important)
    fn priority(&self) -> u32;
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Strategy that generated this recommendation
    pub strategy_name: String,
    /// Optimization type
    pub optimization_type: OptimizationType,
    /// Target parameters to adjust
    pub target_parameters: HashMap<String, ParameterAdjustment>,
    /// Expected improvement
    pub expected_improvement: ExpectedImprovement,
    /// Confidence in recommendation
    pub confidence: f32,
    /// Priority level
    pub priority: u32,
}

/// Types of optimizations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationType {
    /// Improve processing latency
    LatencyOptimization,
    /// Improve accuracy
    AccuracyOptimization,
    /// Improve throughput
    ThroughputOptimization,
    /// Reduce memory usage
    MemoryOptimization,
    /// Reduce CPU usage
    CpuOptimization,
    /// Improve overall quality
    QualityOptimization,
    /// Balance multiple metrics
    BalancedOptimization,
}

/// Parameter adjustment specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterAdjustment {
    /// Parameter name
    pub parameter_name: String,
    /// Current value
    pub current_value: f32,
    /// Recommended new value
    pub recommended_value: f32,
    /// Adjustment type
    pub adjustment_type: AdjustmentType,
    /// Confidence in adjustment
    pub confidence: f32,
}

/// Types of parameter adjustments
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AdjustmentType {
    /// Increase parameter value
    Increase,
    /// Decrease parameter value
    Decrease,
    /// Set to specific value
    SetValue,
    /// Fine-tune within range
    FineTune,
}

/// Expected improvement from optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedImprovement {
    /// Expected latency improvement (%)
    pub latency_improvement: f32,
    /// Expected accuracy improvement (%)
    pub accuracy_improvement: f32,
    /// Expected throughput improvement (%)
    pub throughput_improvement: f32,
    /// Expected memory reduction (%)
    pub memory_reduction: f32,
    /// Expected CPU reduction (%)
    pub cpu_reduction: f32,
    /// Overall quality improvement
    pub quality_improvement: f32,
}

/// Current optimization state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationState {
    /// Current parameters
    pub current_parameters: HashMap<String, f32>,
    /// Last optimization timestamp
    pub last_optimization: Option<SystemTime>,
    /// Active optimizations
    pub active_optimizations: Vec<String>,
    /// Optimization effectiveness
    pub effectiveness_scores: HashMap<String, f32>,
    /// Stability indicator
    pub stability_score: f32,
}

/// Optimization entry for history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationEntry {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Applied optimization
    pub optimization: OptimizationRecommendation,
    /// Performance before optimization
    pub performance_before: PerformanceSnapshot,
    /// Performance after optimization
    pub performance_after: Option<PerformanceSnapshot>,
    /// Optimization success
    pub success: bool,
    /// Notes or error messages
    pub notes: String,
}

/// Performance snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    /// Average latency
    pub avg_latency_ms: f32,
    /// Average accuracy
    pub avg_accuracy: f32,
    /// Average throughput
    pub avg_throughput: f32,
    /// Average memory usage
    pub avg_memory_mb: f32,
    /// Average CPU usage
    pub avg_cpu_percent: f32,
    /// Overall quality score
    pub quality_score: f32,
    /// Snapshot timestamp
    pub timestamp: SystemTime,
}

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Enable automatic optimization
    pub enable_auto_optimization: bool,
    /// Optimization interval
    pub optimization_interval: Duration,
    /// Minimum data points before optimization
    pub min_data_points: usize,
    /// Maximum optimization attempts per interval
    pub max_optimizations_per_interval: usize,
    /// Performance improvement threshold
    pub improvement_threshold: f32,
    /// Rollback threshold for failed optimizations
    pub rollback_threshold: f32,
    /// Enable conservative optimization mode
    pub conservative_mode: bool,
}

/// Latency optimization strategy
pub struct LatencyOptimizationStrategy {
    /// Target latency threshold
    pub target_latency_ms: f32,
    /// Aggressive optimization flag
    pub aggressive_mode: bool,
}

/// Accuracy optimization strategy
pub struct AccuracyOptimizationStrategy {
    /// Target accuracy threshold
    pub target_accuracy: f32,
    /// Quality vs speed trade-off
    pub quality_priority: f32,
}

/// Memory optimization strategy
pub struct MemoryOptimizationStrategy {
    /// Target memory usage
    pub target_memory_mb: f32,
    /// Enable aggressive garbage collection
    pub aggressive_gc: bool,
}

/// Adaptive load balancing strategy
pub struct AdaptiveLoadBalancingStrategy {
    /// Load balancing thresholds
    pub load_thresholds: HashMap<String, f32>,
    /// Backend performance tracking
    pub backend_performance: HashMap<String, PerformanceSnapshot>,
}

/// Dynamic caching optimization
pub struct DynamicCachingStrategy {
    /// Cache hit rate threshold
    pub target_hit_rate: f32,
    /// Memory budget for caching
    pub memory_budget_mb: f32,
    /// Cache eviction policy
    pub eviction_policy: String,
}

/// Batch processing optimizer
pub struct BatchProcessingOptimizer {
    /// Optimal batch sizes for different workloads
    pub optimal_batch_sizes: HashMap<String, usize>,
    /// Throughput measurements
    pub throughput_history: VecDeque<(usize, f32)>, // (batch_size, throughput)
    /// Current optimization target
    pub optimization_target: BatchOptimizationTarget,
}

/// Batch optimization targets
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BatchOptimizationTarget {
    /// Maximize throughput
    MaxThroughput,
    /// Minimize latency
    MinLatency,
    /// Balance throughput and latency
    Balanced,
    /// Minimize memory usage
    MinMemory,
}

/// Quality-aware optimization system
pub struct QualityAwareOptimizer {
    /// Quality metrics tracking
    pub quality_metrics: QualityMetrics,
    /// Quality thresholds
    pub quality_thresholds: HashMap<String, f32>,
    /// Quality improvement strategies
    pub improvement_strategies: Vec<QualityImprovementStrategy>,
}

/// Quality metrics tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Phoneme-level accuracy
    pub phoneme_accuracy: VecDeque<f32>,
    /// Word-level accuracy
    pub word_accuracy: VecDeque<f32>,
    /// Pronunciation naturalness
    pub naturalness_scores: VecDeque<f32>,
    /// User satisfaction ratings
    pub user_satisfaction: VecDeque<f32>,
    /// Error distribution
    pub error_patterns: HashMap<String, usize>,
}

/// Quality improvement strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityImprovementStrategy {
    /// Strategy name
    pub name: String,
    /// Target quality aspect
    pub target_aspect: String,
    /// Improvement actions
    pub actions: Vec<String>,
    /// Expected improvement
    pub expected_improvement: f32,
    /// Implementation complexity
    pub complexity: u32,
}

// Implementation for RealTimeOptimizer
impl RealTimeOptimizer {
    /// Create new real-time optimizer
    pub fn new(language: LanguageCode, config: OptimizerConfig) -> Self {
        Self {
            language,
            metrics_collector: Arc::new(Mutex::new(MetricsCollector::new())),
            optimization_strategies: Vec::new(),
            optimization_state: OptimizationState::default(),
            optimization_history: VecDeque::new(),
            config,
        }
    }

    /// Add optimization strategy
    pub fn add_strategy(&mut self, strategy: Box<dyn OptimizationStrategy>) {
        self.optimization_strategies.push(strategy);

        // Sort strategies by priority
        self.optimization_strategies
            .sort_by_key(|b| std::cmp::Reverse(b.priority()));
    }

    /// Record performance measurement
    pub fn record_performance(&self, latency: Duration, accuracy: f32, throughput: f32) {
        let mut collector = self.metrics_collector.lock().unwrap();
        collector.add_measurement(latency, accuracy, throughput);
    }

    /// Run optimization cycle
    pub fn optimize(&mut self) -> Result<Vec<OptimizationRecommendation>> {
        let metrics = self.metrics_collector.lock().unwrap().clone();

        // Check if we have enough data
        if metrics.latency_measurements.len() < self.config.min_data_points {
            return Ok(Vec::new());
        }

        let mut recommendations = Vec::new();

        // Get recommendations from all applicable strategies
        for strategy in &self.optimization_strategies {
            if strategy.is_applicable(&metrics) {
                let recommendation = strategy.analyze_metrics(&metrics);
                recommendations.push(recommendation);
            }
        }

        // Sort by priority and confidence
        recommendations.sort_by(|a, b| {
            let priority_cmp = b.priority.cmp(&a.priority);
            if priority_cmp == std::cmp::Ordering::Equal {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                priority_cmp
            }
        });

        // Apply top recommendations if auto-optimization is enabled
        if self.config.enable_auto_optimization {
            let max_optimizations = self.config.max_optimizations_per_interval;
            for recommendation in recommendations.iter().take(max_optimizations) {
                if let Err(e) = self.apply_optimization(recommendation) {
                    eprintln!("Failed to apply optimization: {e}");
                }
            }
        }

        Ok(recommendations)
    }

    /// Apply specific optimization
    fn apply_optimization(&mut self, recommendation: &OptimizationRecommendation) -> Result<()> {
        let performance_before = self.capture_performance_snapshot();

        // Find and apply the strategy
        let strategy_name = &recommendation.strategy_name;
        let strategy = self
            .optimization_strategies
            .iter()
            .find(|s| s.name() == strategy_name)
            .ok_or_else(|| G2pError::ConfigError(format!("Strategy not found: {strategy_name}")))?;

        let result = strategy.apply_optimization(recommendation);
        let success = result.is_ok();

        // Record optimization entry
        let entry = OptimizationEntry {
            timestamp: SystemTime::now(),
            optimization: recommendation.clone(),
            performance_before,
            performance_after: None, // Will be updated later
            success,
            notes: if success {
                "Applied successfully".to_string()
            } else {
                format!("Failed: {result:?}")
            },
        };

        self.optimization_history.push_back(entry);

        // Keep history bounded
        if self.optimization_history.len() > 1000 {
            self.optimization_history.pop_front();
        }

        result
    }

    /// Capture current performance snapshot
    fn capture_performance_snapshot(&self) -> PerformanceSnapshot {
        let metrics = self.metrics_collector.lock().unwrap();

        PerformanceSnapshot {
            avg_latency_ms: metrics.average_latency().as_millis() as f32,
            avg_accuracy: metrics.average_accuracy(),
            avg_throughput: metrics.average_throughput(),
            avg_memory_mb: metrics.average_memory() as f32 / (1024.0 * 1024.0),
            avg_cpu_percent: metrics.average_cpu(),
            quality_score: metrics.average_quality(),
            timestamp: SystemTime::now(),
        }
    }

    /// Get optimization effectiveness report
    pub fn get_effectiveness_report(&self) -> OptimizationEffectivenessReport {
        let mut strategy_effectiveness = HashMap::new();
        let mut total_improvements = 0;
        let mut successful_optimizations = 0;

        for entry in &self.optimization_history {
            let strategy_name = &entry.optimization.strategy_name;

            if entry.success {
                successful_optimizations += 1;

                if let Some(after) = &entry.performance_after {
                    let improvement = self.calculate_improvement(&entry.performance_before, after);

                    let effectiveness_entry = strategy_effectiveness
                        .entry(strategy_name.clone())
                        .or_insert(StrategyEffectiveness::default());

                    effectiveness_entry.total_applications += 1;
                    effectiveness_entry.successful_applications += 1;
                    effectiveness_entry.average_improvement += improvement;

                    if improvement > 0.0 {
                        total_improvements += 1;
                    }
                }
            } else {
                let effectiveness_entry = strategy_effectiveness
                    .entry(strategy_name.clone())
                    .or_insert(StrategyEffectiveness::default());

                effectiveness_entry.total_applications += 1;
            }
        }

        // Calculate averages
        for effectiveness in strategy_effectiveness.values_mut() {
            if effectiveness.successful_applications > 0 {
                effectiveness.average_improvement /= effectiveness.successful_applications as f32;
            }
        }

        OptimizationEffectivenessReport {
            total_optimizations: self.optimization_history.len(),
            successful_optimizations,
            total_improvements,
            strategy_effectiveness,
            overall_success_rate: if self.optimization_history.is_empty() {
                0.0
            } else {
                successful_optimizations as f32 / self.optimization_history.len() as f32
            },
        }
    }

    /// Calculate improvement between two performance snapshots
    fn calculate_improvement(
        &self,
        before: &PerformanceSnapshot,
        after: &PerformanceSnapshot,
    ) -> f32 {
        let latency_improvement =
            (before.avg_latency_ms - after.avg_latency_ms) / before.avg_latency_ms;
        let accuracy_improvement = (after.avg_accuracy - before.avg_accuracy) / before.avg_accuracy;
        let throughput_improvement =
            (after.avg_throughput - before.avg_throughput) / before.avg_throughput;

        // Weighted average of improvements
        (latency_improvement * 0.3 + accuracy_improvement * 0.4 + throughput_improvement * 0.3)
            * 100.0
    }
}

/// Optimization effectiveness report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationEffectivenessReport {
    /// Total number of optimizations attempted
    pub total_optimizations: usize,
    /// Number of successful optimizations
    pub successful_optimizations: usize,
    /// Number of optimizations that resulted in improvements
    pub total_improvements: usize,
    /// Per-strategy effectiveness
    pub strategy_effectiveness: HashMap<String, StrategyEffectiveness>,
    /// Overall success rate
    pub overall_success_rate: f32,
}

/// Strategy effectiveness metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StrategyEffectiveness {
    /// Total times strategy was applied
    pub total_applications: usize,
    /// Successful applications
    pub successful_applications: usize,
    /// Average improvement achieved
    pub average_improvement: f32,
}

// Implementation for MetricsCollector
impl MetricsCollector {
    /// Create new metrics collector
    pub fn new() -> Self {
        Self {
            latency_measurements: VecDeque::new(),
            accuracy_measurements: VecDeque::new(),
            throughput_measurements: VecDeque::new(),
            error_rates: VecDeque::new(),
            memory_usage: VecDeque::new(),
            cpu_usage: VecDeque::new(),
            quality_scores: VecDeque::new(),
            user_satisfaction: VecDeque::new(),
        }
    }

    /// Add measurement
    pub fn add_measurement(&mut self, latency: Duration, accuracy: f32, throughput: f32) {
        self.latency_measurements.push_back(latency);
        self.accuracy_measurements.push_back(accuracy);
        self.throughput_measurements.push_back(throughput);

        // Keep collections bounded
        const MAX_MEASUREMENTS: usize = 1000;

        if self.latency_measurements.len() > MAX_MEASUREMENTS {
            self.latency_measurements.pop_front();
        }
        if self.accuracy_measurements.len() > MAX_MEASUREMENTS {
            self.accuracy_measurements.pop_front();
        }
        if self.throughput_measurements.len() > MAX_MEASUREMENTS {
            self.throughput_measurements.pop_front();
        }
    }

    /// Calculate average latency
    pub fn average_latency(&self) -> Duration {
        if self.latency_measurements.is_empty() {
            return Duration::from_millis(0);
        }

        let total: Duration = self.latency_measurements.iter().sum();
        total / self.latency_measurements.len() as u32
    }

    /// Calculate average accuracy
    pub fn average_accuracy(&self) -> f32 {
        if self.accuracy_measurements.is_empty() {
            return 0.0;
        }

        self.accuracy_measurements.iter().sum::<f32>() / self.accuracy_measurements.len() as f32
    }

    /// Calculate average throughput
    pub fn average_throughput(&self) -> f32 {
        if self.throughput_measurements.is_empty() {
            return 0.0;
        }

        self.throughput_measurements.iter().sum::<f32>() / self.throughput_measurements.len() as f32
    }

    /// Calculate average memory usage
    pub fn average_memory(&self) -> u64 {
        if self.memory_usage.is_empty() {
            return 0;
        }

        self.memory_usage.iter().sum::<u64>() / self.memory_usage.len() as u64
    }

    /// Calculate average CPU usage
    pub fn average_cpu(&self) -> f32 {
        if self.cpu_usage.is_empty() {
            return 0.0;
        }

        self.cpu_usage.iter().sum::<f32>() / self.cpu_usage.len() as f32
    }

    /// Calculate average quality score
    pub fn average_quality(&self) -> f32 {
        if self.quality_scores.is_empty() {
            return 0.0;
        }

        self.quality_scores.iter().sum::<f32>() / self.quality_scores.len() as f32
    }
}

// Implementation for LatencyOptimizationStrategy
impl OptimizationStrategy for LatencyOptimizationStrategy {
    fn name(&self) -> &str {
        "LatencyOptimization"
    }

    fn analyze_metrics(&self, metrics: &MetricsCollector) -> OptimizationRecommendation {
        let avg_latency_ms = metrics.average_latency().as_millis() as f32;
        let target_latency_ms = self.target_latency_ms;

        let mut parameter_adjustments = HashMap::new();

        if avg_latency_ms > target_latency_ms {
            // Suggest reducing model complexity
            parameter_adjustments.insert(
                "model_complexity".to_string(),
                ParameterAdjustment {
                    parameter_name: "model_complexity".to_string(),
                    current_value: 1.0,
                    recommended_value: 0.8,
                    adjustment_type: AdjustmentType::Decrease,
                    confidence: 0.7,
                },
            );

            // Suggest increasing cache size
            parameter_adjustments.insert(
                "cache_size".to_string(),
                ParameterAdjustment {
                    parameter_name: "cache_size".to_string(),
                    current_value: 1000.0,
                    recommended_value: 2000.0,
                    adjustment_type: AdjustmentType::Increase,
                    confidence: 0.8,
                },
            );
        }

        let expected_improvement = ExpectedImprovement {
            latency_improvement: if avg_latency_ms > target_latency_ms {
                20.0
            } else {
                0.0
            },
            accuracy_improvement: -5.0, // Trade-off
            throughput_improvement: 15.0,
            memory_reduction: -10.0, // May use more memory for caching
            cpu_reduction: 10.0,
            quality_improvement: -5.0, // Trade-off
        };

        OptimizationRecommendation {
            strategy_name: self.name().to_string(),
            optimization_type: OptimizationType::LatencyOptimization,
            target_parameters: parameter_adjustments,
            expected_improvement,
            confidence: 0.75,
            priority: 3,
        }
    }

    fn apply_optimization(&self, _recommendation: &OptimizationRecommendation) -> Result<()> {
        // In practice, this would apply the actual optimizations
        Ok(())
    }

    fn is_applicable(&self, metrics: &MetricsCollector) -> bool {
        !metrics.latency_measurements.is_empty()
    }

    fn priority(&self) -> u32 {
        3
    }
}

// Implementation for AccuracyOptimizationStrategy
impl OptimizationStrategy for AccuracyOptimizationStrategy {
    fn name(&self) -> &str {
        "AccuracyOptimization"
    }

    fn analyze_metrics(&self, metrics: &MetricsCollector) -> OptimizationRecommendation {
        let avg_accuracy = metrics.average_accuracy();
        let target_accuracy = self.target_accuracy;

        let mut parameter_adjustments = HashMap::new();

        if avg_accuracy < target_accuracy {
            // Suggest increasing model complexity
            parameter_adjustments.insert(
                "model_complexity".to_string(),
                ParameterAdjustment {
                    parameter_name: "model_complexity".to_string(),
                    current_value: 0.8,
                    recommended_value: 1.0,
                    adjustment_type: AdjustmentType::Increase,
                    confidence: 0.8,
                },
            );

            // Suggest enabling ensemble methods
            parameter_adjustments.insert(
                "ensemble_enabled".to_string(),
                ParameterAdjustment {
                    parameter_name: "ensemble_enabled".to_string(),
                    current_value: 0.0,
                    recommended_value: 1.0,
                    adjustment_type: AdjustmentType::SetValue,
                    confidence: 0.9,
                },
            );
        }

        let expected_improvement = ExpectedImprovement {
            latency_improvement: -15.0, // Trade-off
            accuracy_improvement: if avg_accuracy < target_accuracy {
                10.0
            } else {
                0.0
            },
            throughput_improvement: -10.0, // Trade-off
            memory_reduction: -20.0,       // May use more memory
            cpu_reduction: -15.0,          // Trade-off
            quality_improvement: 15.0,
        };

        OptimizationRecommendation {
            strategy_name: self.name().to_string(),
            optimization_type: OptimizationType::AccuracyOptimization,
            target_parameters: parameter_adjustments,
            expected_improvement,
            confidence: 0.8,
            priority: 4,
        }
    }

    fn apply_optimization(&self, _recommendation: &OptimizationRecommendation) -> Result<()> {
        // In practice, this would apply the actual optimizations
        Ok(())
    }

    fn is_applicable(&self, metrics: &MetricsCollector) -> bool {
        !metrics.accuracy_measurements.is_empty()
    }

    fn priority(&self) -> u32 {
        4
    }
}

// Default implementations
impl Default for OptimizationState {
    fn default() -> Self {
        Self {
            current_parameters: HashMap::new(),
            last_optimization: None,
            active_optimizations: Vec::new(),
            effectiveness_scores: HashMap::new(),
            stability_score: 1.0,
        }
    }
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            enable_auto_optimization: false,
            optimization_interval: Duration::from_secs(300), // 5 minutes
            min_data_points: 100,
            max_optimizations_per_interval: 3,
            improvement_threshold: 5.0,
            rollback_threshold: -10.0,
            conservative_mode: true,
        }
    }
}

impl Default for ExpectedImprovement {
    fn default() -> Self {
        Self {
            latency_improvement: 0.0,
            accuracy_improvement: 0.0,
            throughput_improvement: 0.0,
            memory_reduction: 0.0,
            cpu_reduction: 0.0,
            quality_improvement: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new();

        collector.add_measurement(Duration::from_millis(100), 0.8, 50.0);
        collector.add_measurement(Duration::from_millis(120), 0.85, 45.0);
        collector.add_measurement(Duration::from_millis(110), 0.82, 48.0);

        assert_eq!(collector.latency_measurements.len(), 3);
        assert!((collector.average_accuracy() - 0.823).abs() < 0.01);
        assert!((collector.average_throughput() - 47.67).abs() < 0.1);
    }

    #[test]
    fn test_real_time_optimizer() {
        let config = OptimizerConfig::default();
        let mut optimizer = RealTimeOptimizer::new(LanguageCode::EnUs, config);

        // Add a strategy
        let latency_strategy = Box::new(LatencyOptimizationStrategy {
            target_latency_ms: 50.0,
            aggressive_mode: false,
        });
        optimizer.add_strategy(latency_strategy);

        // Record some performance data
        optimizer.record_performance(Duration::from_millis(100), 0.8, 50.0);
        optimizer.record_performance(Duration::from_millis(120), 0.85, 45.0);

        assert_eq!(optimizer.optimization_strategies.len(), 1);
    }

    #[test]
    fn test_latency_optimization_strategy() {
        let strategy = LatencyOptimizationStrategy {
            target_latency_ms: 50.0,
            aggressive_mode: false,
        };

        let mut metrics = MetricsCollector::new();
        metrics.add_measurement(Duration::from_millis(100), 0.8, 50.0);

        assert!(strategy.is_applicable(&metrics));

        let recommendation = strategy.analyze_metrics(&metrics);
        assert_eq!(
            recommendation.optimization_type,
            OptimizationType::LatencyOptimization
        );
        assert!(recommendation.expected_improvement.latency_improvement > 0.0);
    }

    #[test]
    fn test_accuracy_optimization_strategy() {
        let strategy = AccuracyOptimizationStrategy {
            target_accuracy: 0.9,
            quality_priority: 0.8,
        };

        let mut metrics = MetricsCollector::new();
        metrics.add_measurement(Duration::from_millis(100), 0.7, 50.0); // Low accuracy

        assert!(strategy.is_applicable(&metrics));

        let recommendation = strategy.analyze_metrics(&metrics);
        assert_eq!(
            recommendation.optimization_type,
            OptimizationType::AccuracyOptimization
        );
        assert!(recommendation.expected_improvement.accuracy_improvement > 0.0);
    }

    #[test]
    fn test_optimization_recommendation() {
        let recommendation = OptimizationRecommendation {
            strategy_name: "TestStrategy".to_string(),
            optimization_type: OptimizationType::LatencyOptimization,
            target_parameters: HashMap::new(),
            expected_improvement: ExpectedImprovement::default(),
            confidence: 0.8,
            priority: 3,
        };

        assert_eq!(recommendation.strategy_name, "TestStrategy");
        assert_eq!(
            recommendation.optimization_type,
            OptimizationType::LatencyOptimization
        );
        assert_eq!(recommendation.confidence, 0.8);
    }

    #[test]
    fn test_performance_snapshot() {
        let snapshot = PerformanceSnapshot {
            avg_latency_ms: 100.0,
            avg_accuracy: 0.85,
            avg_throughput: 50.0,
            avg_memory_mb: 256.0,
            avg_cpu_percent: 45.0,
            quality_score: 0.8,
            timestamp: SystemTime::now(),
        };

        assert!(snapshot.avg_latency_ms > 0.0);
        assert!(snapshot.avg_accuracy > 0.0);
        assert!(snapshot.quality_score > 0.0);
    }
}
