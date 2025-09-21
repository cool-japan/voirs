//! Feature-specific optimization paths for vocoder operations
//!
//! Provides specialized optimization strategies for different vocoder features including:
//! - Emotion-specific optimizations
//! - Voice conversion optimizations
//! - Spatial audio optimizations
//! - Singing voice optimizations
//! - Real-time streaming optimizations

use crate::{
    adaptive_quality::{AdaptiveConfig, PrecisionMode},
    performance::PerformanceMetrics,
    Result, VocoderError, VocoderFeature,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Feature-specific optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureOptimizationConfig {
    /// Feature type being optimized
    pub feature_type: VocoderFeature,
    /// Optimization strategy
    pub strategy: OptimizationStrategy,
    /// Performance targets
    pub targets: PerformanceTargets,
    /// Resource constraints
    pub constraints: ResourceConstraints,
    /// Quality preferences
    pub quality_preferences: QualityPreferences,
}

/// Optimization strategies for different features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    /// Optimize for minimum latency
    LatencyFirst,
    /// Optimize for best quality
    QualityFirst,
    /// Balance latency and quality
    Balanced,
    /// Optimize for minimum resource usage
    ResourceEfficient,
    /// Custom optimization with weights
    Custom {
        latency_weight: f32,
        quality_weight: f32,
        resource_weight: f32,
    },
}

/// Performance targets for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    /// Target latency in milliseconds
    pub target_latency_ms: f32,
    /// Target quality score (0.0-1.0)
    pub target_quality: f32,
    /// Target real-time factor
    pub target_rtf: f32,
    /// Target memory usage in MB
    pub target_memory_mb: f32,
    /// Target CPU usage percentage
    pub target_cpu_usage: f32,
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            target_latency_ms: 50.0,
            target_quality: 0.8,
            target_rtf: 0.5,
            target_memory_mb: 512.0,
            target_cpu_usage: 70.0,
        }
    }
}

/// Resource constraints for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Maximum allowed memory usage in MB
    pub max_memory_mb: f32,
    /// Maximum allowed CPU usage percentage
    pub max_cpu_usage: f32,
    /// Maximum allowed latency in milliseconds
    pub max_latency_ms: f32,
    /// Available compute units (cores, threads, etc.)
    pub available_compute_units: usize,
    /// GPU memory available in MB (if applicable)
    pub gpu_memory_mb: Option<f32>,
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_memory_mb: 2048.0,
            max_cpu_usage: 90.0,
            max_latency_ms: 100.0,
            available_compute_units: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4),
            gpu_memory_mb: None,
        }
    }
}

/// Quality preferences for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityPreferences {
    /// Minimum acceptable quality score
    pub min_quality: f32,
    /// Quality vs speed preference (0.0=speed, 1.0=quality)
    pub quality_preference: f32,
    /// Enable advanced processing features
    pub enable_advanced_features: bool,
    /// Quality consistency priority
    pub consistency_priority: f32,
}

impl Default for QualityPreferences {
    fn default() -> Self {
        Self {
            min_quality: 0.6,
            quality_preference: 0.7,
            enable_advanced_features: true,
            consistency_priority: 0.8,
        }
    }
}

/// Feature-specific optimization manager
pub struct FeatureOptimizer {
    /// Optimization configurations by feature
    feature_configs: HashMap<VocoderFeature, FeatureOptimizationConfig>,
    /// Performance history
    performance_history: HashMap<VocoderFeature, Vec<PerformanceMetrics>>,
    /// Optimization statistics
    optimization_stats: OptimizationStats,
}

/// Optimization statistics
#[derive(Debug, Clone, Default)]
pub struct OptimizationStats {
    /// Total optimizations performed
    pub total_optimizations: u64,
    /// Successful optimizations
    pub successful_optimizations: u64,
    /// Average latency improvement
    pub avg_latency_improvement_ms: f32,
    /// Average quality improvement
    pub avg_quality_improvement: f32,
    /// Average resource usage reduction
    pub avg_resource_reduction: f32,
    /// Last optimization timestamp
    pub last_optimization: Option<Instant>,
}

impl Default for FeatureOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureOptimizer {
    /// Create new feature optimizer
    pub fn new() -> Self {
        Self {
            feature_configs: HashMap::new(),
            performance_history: HashMap::new(),
            optimization_stats: OptimizationStats::default(),
        }
    }

    /// Add optimization configuration for a feature
    pub fn add_feature_config(&mut self, config: FeatureOptimizationConfig) {
        self.feature_configs.insert(config.feature_type, config);
    }

    /// Optimize feature configuration based on performance metrics
    pub fn optimize_feature(
        &mut self,
        feature: &VocoderFeature,
        current_metrics: &PerformanceMetrics,
    ) -> Result<AdaptiveConfig> {
        let config = self
            .feature_configs
            .get(feature)
            .ok_or_else(|| {
                VocoderError::ConfigError(format!(
                    "No optimization config for feature: {feature:?}"
                ))
            })?
            .clone();

        // Record performance metrics
        self.record_performance(*feature, current_metrics.clone());

        // Generate optimized configuration
        let optimized_config = self.generate_optimized_config(&config, current_metrics)?;

        // Update optimization statistics
        self.update_optimization_stats(current_metrics, &optimized_config);

        Ok(optimized_config)
    }

    /// Generate optimized configuration based on strategy and metrics
    fn generate_optimized_config(
        &self,
        feature_config: &FeatureOptimizationConfig,
        metrics: &PerformanceMetrics,
    ) -> Result<AdaptiveConfig> {
        let mut config = AdaptiveConfig::default();

        match &feature_config.strategy {
            OptimizationStrategy::LatencyFirst => {
                self.optimize_for_latency(&mut config, feature_config, metrics)?;
            }
            OptimizationStrategy::QualityFirst => {
                self.optimize_for_quality(&mut config, feature_config, metrics)?;
            }
            OptimizationStrategy::Balanced => {
                self.optimize_balanced(&mut config, feature_config, metrics)?;
            }
            OptimizationStrategy::ResourceEfficient => {
                self.optimize_for_resources(&mut config, feature_config, metrics)?;
            }
            OptimizationStrategy::Custom {
                latency_weight,
                quality_weight,
                resource_weight,
            } => {
                self.optimize_custom(
                    &mut config,
                    feature_config,
                    metrics,
                    *latency_weight,
                    *quality_weight,
                    *resource_weight,
                )?;
            }
        }

        Ok(config)
    }

    /// Optimize configuration for minimum latency
    fn optimize_for_latency(
        &self,
        config: &mut AdaptiveConfig,
        feature_config: &FeatureOptimizationConfig,
        metrics: &PerformanceMetrics,
    ) -> Result<()> {
        // Reduce quality for better latency
        config.quality_level = 0.6;
        config.precision_mode = PrecisionMode::Low;
        config.batch_size = 8; // Smaller batches for lower latency
        config.enable_expensive_processing = false;
        config.upsampling_factor = 0.8; // Reduce upsampling

        // Adjust based on current performance
        if metrics.latency_ms > feature_config.targets.target_latency_ms {
            config.quality_level *= 0.8;
            config.batch_size = config.batch_size.max(4);
            config.num_streams = config.num_streams.min(2);
        }

        Ok(())
    }

    /// Optimize configuration for best quality
    fn optimize_for_quality(
        &self,
        config: &mut AdaptiveConfig,
        feature_config: &FeatureOptimizationConfig,
        _metrics: &PerformanceMetrics,
    ) -> Result<()> {
        // Maximize quality settings
        config.quality_level = 1.0;
        config.precision_mode = PrecisionMode::Ultra;
        config.batch_size = 32; // Larger batches for better quality
        config.enable_expensive_processing = true;
        config.upsampling_factor = 1.2; // Enhanced upsampling
        config.noise_suppression = 0.5; // Strong noise suppression

        // Ensure minimum quality is maintained
        if config.quality_level < feature_config.quality_preferences.min_quality {
            config.quality_level = feature_config.quality_preferences.min_quality;
        }

        Ok(())
    }

    /// Optimize configuration with balanced approach
    fn optimize_balanced(
        &self,
        config: &mut AdaptiveConfig,
        feature_config: &FeatureOptimizationConfig,
        metrics: &PerformanceMetrics,
    ) -> Result<()> {
        // Balance quality and performance
        config.quality_level = 0.8;
        config.precision_mode = PrecisionMode::Medium;
        config.batch_size = 16;
        config.enable_expensive_processing = true;
        config.upsampling_factor = 1.0;
        config.noise_suppression = 0.3;

        // Adjust based on current performance
        let latency_ratio = metrics.latency_ms / feature_config.targets.target_latency_ms;
        let quality_ratio =
            metrics.quality.mos_estimate / (feature_config.targets.target_quality * 5.0);

        if latency_ratio > 1.2 {
            // Latency too high, reduce quality
            config.quality_level *= 0.9;
            config.batch_size = config.batch_size.max(8);
        } else if quality_ratio < 0.8 {
            // Quality too low, increase processing
            config.quality_level = (config.quality_level * 1.1).min(1.0);
            config.enable_expensive_processing = true;
        }

        Ok(())
    }

    /// Optimize configuration for resource efficiency
    fn optimize_for_resources(
        &self,
        config: &mut AdaptiveConfig,
        feature_config: &FeatureOptimizationConfig,
        metrics: &PerformanceMetrics,
    ) -> Result<()> {
        // Minimize resource usage
        config.quality_level = feature_config.quality_preferences.min_quality;
        config.precision_mode = PrecisionMode::Low;
        config.batch_size = 4; // Small batches
        config.num_streams = 1; // Single stream
        config.enable_expensive_processing = false;
        config.upsampling_factor = 0.7;
        config.noise_suppression = 0.1;
        config.compression_ratio = 1.2; // Slight compression

        // Adjust based on resource usage
        if metrics.memory_usage_mb > feature_config.constraints.max_memory_mb * 0.8 {
            config.batch_size = config.batch_size.max(2);
            config.quality_level *= 0.9;
        }

        if metrics.cpu_usage > feature_config.constraints.max_cpu_usage * 0.8 {
            config.num_streams = 1;
            config.enable_expensive_processing = false;
        }

        Ok(())
    }

    /// Optimize configuration with custom weights
    fn optimize_custom(
        &self,
        config: &mut AdaptiveConfig,
        feature_config: &FeatureOptimizationConfig,
        metrics: &PerformanceMetrics,
        latency_weight: f32,
        quality_weight: f32,
        resource_weight: f32,
    ) -> Result<()> {
        // Start with balanced configuration
        self.optimize_balanced(config, feature_config, metrics)?;

        // Apply custom weights
        let total_weight = latency_weight + quality_weight + resource_weight;
        let norm_latency = latency_weight / total_weight;
        let norm_quality = quality_weight / total_weight;
        let norm_resource = resource_weight / total_weight;

        // Adjust quality level based on weights
        let base_quality = config.quality_level;
        config.quality_level =
            base_quality * (norm_quality * 1.5 + norm_latency * 0.5 + norm_resource * 0.3);
        config.quality_level = config.quality_level.clamp(0.1, 1.0);

        // Adjust batch size based on weights
        let base_batch = config.batch_size as f32;
        let adjusted_batch =
            base_batch * (norm_quality * 1.3 + norm_latency * 0.7 + norm_resource * 0.5);
        config.batch_size = (adjusted_batch as usize).clamp(1, 64);

        // Adjust precision mode based on weights
        if norm_quality > 0.6 {
            config.precision_mode = PrecisionMode::High;
        } else if norm_latency > 0.6 || norm_resource > 0.6 {
            config.precision_mode = PrecisionMode::Low;
        } else {
            config.precision_mode = PrecisionMode::Medium;
        }

        Ok(())
    }

    /// Record performance metrics for a feature
    fn record_performance(&mut self, feature: VocoderFeature, metrics: PerformanceMetrics) {
        let history = self.performance_history.entry(feature).or_default();
        history.push(metrics);

        // Keep only recent history (last 100 measurements)
        if history.len() > 100 {
            history.remove(0);
        }
    }

    /// Update optimization statistics
    fn update_optimization_stats(
        &mut self,
        old_metrics: &PerformanceMetrics,
        new_config: &AdaptiveConfig,
    ) {
        self.optimization_stats.total_optimizations += 1;
        self.optimization_stats.last_optimization = Some(Instant::now());

        // Estimate improvements (would be measured in practice)
        let estimated_latency_improvement = old_metrics.latency_ms * 0.1; // Assume 10% improvement
        let estimated_quality_improvement = new_config.quality_level * 0.05; // Assume 5% improvement
        let estimated_resource_reduction = 0.08; // Assume 8% reduction

        self.optimization_stats.avg_latency_improvement_ms =
            (self.optimization_stats.avg_latency_improvement_ms
                * (self.optimization_stats.total_optimizations - 1) as f32
                + estimated_latency_improvement)
                / self.optimization_stats.total_optimizations as f32;

        self.optimization_stats.avg_quality_improvement =
            (self.optimization_stats.avg_quality_improvement
                * (self.optimization_stats.total_optimizations - 1) as f32
                + estimated_quality_improvement)
                / self.optimization_stats.total_optimizations as f32;

        self.optimization_stats.avg_resource_reduction =
            (self.optimization_stats.avg_resource_reduction
                * (self.optimization_stats.total_optimizations - 1) as f32
                + estimated_resource_reduction)
                / self.optimization_stats.total_optimizations as f32;

        self.optimization_stats.successful_optimizations += 1;
    }

    /// Get optimization statistics
    pub fn get_optimization_stats(&self) -> &OptimizationStats {
        &self.optimization_stats
    }

    /// Get performance history for a feature
    pub fn get_performance_history(
        &self,
        feature: &VocoderFeature,
    ) -> Option<&Vec<PerformanceMetrics>> {
        self.performance_history.get(feature)
    }

    /// Create feature-specific optimization for emotion processing
    pub fn create_emotion_optimization(
        quality_preference: f32,
        latency_target_ms: f32,
    ) -> FeatureOptimizationConfig {
        FeatureOptimizationConfig {
            feature_type: VocoderFeature::Emotion,
            strategy: if quality_preference > 0.8 {
                OptimizationStrategy::QualityFirst
            } else if latency_target_ms < 30.0 {
                OptimizationStrategy::LatencyFirst
            } else {
                OptimizationStrategy::Balanced
            },
            targets: PerformanceTargets {
                target_latency_ms: latency_target_ms,
                target_quality: quality_preference,
                target_rtf: 0.4,
                target_memory_mb: 256.0,
                target_cpu_usage: 60.0,
            },
            constraints: ResourceConstraints::default(),
            quality_preferences: QualityPreferences {
                min_quality: quality_preference * 0.8,
                quality_preference,
                enable_advanced_features: quality_preference > 0.7,
                consistency_priority: 0.9, // Emotion requires consistency
            },
        }
    }

    /// Create feature-specific optimization for voice conversion
    pub fn create_voice_conversion_optimization(
        conversion_quality: f32,
        real_time_factor: f32,
    ) -> FeatureOptimizationConfig {
        FeatureOptimizationConfig {
            feature_type: VocoderFeature::VoiceConversion,
            strategy: OptimizationStrategy::Custom {
                latency_weight: 0.3,
                quality_weight: 0.5,
                resource_weight: 0.2,
            },
            targets: PerformanceTargets {
                target_latency_ms: 80.0,
                target_quality: conversion_quality,
                target_rtf: real_time_factor,
                target_memory_mb: 512.0,
                target_cpu_usage: 75.0,
            },
            constraints: ResourceConstraints::default(),
            quality_preferences: QualityPreferences {
                min_quality: 0.7, // Voice conversion needs good quality
                quality_preference: conversion_quality,
                enable_advanced_features: true,
                consistency_priority: 0.8,
            },
        }
    }

    /// Create feature-specific optimization for spatial audio
    pub fn create_spatial_optimization(
        spatial_precision: f32,
        head_tracking_latency_ms: f32,
    ) -> FeatureOptimizationConfig {
        FeatureOptimizationConfig {
            feature_type: VocoderFeature::Spatial,
            strategy: if head_tracking_latency_ms < 20.0 {
                OptimizationStrategy::LatencyFirst
            } else {
                OptimizationStrategy::Balanced
            },
            targets: PerformanceTargets {
                target_latency_ms: head_tracking_latency_ms,
                target_quality: spatial_precision,
                target_rtf: 0.6,
                target_memory_mb: 384.0,
                target_cpu_usage: 70.0,
            },
            constraints: ResourceConstraints::default(),
            quality_preferences: QualityPreferences {
                min_quality: 0.6,
                quality_preference: spatial_precision,
                enable_advanced_features: spatial_precision > 0.8,
                consistency_priority: 0.95, // Spatial audio requires high consistency
            },
        }
    }
}
