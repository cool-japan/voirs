//! Performance monitoring and metrics utilities for VoiRS SDK.
//!
//! This module provides comprehensive performance tracking, monitoring, and analysis
//! capabilities to help developers optimize their speech synthesis applications.

use crate::types::AdvancedFeature;
use crate::{Result, VoirsError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Performance metrics collector for VoiRS SDK operations.
#[derive(Debug, Clone)]
pub struct PerformanceMonitor {
    metrics: Arc<Mutex<PerformanceMetrics>>,
    start_time: Instant,
}

/// Comprehensive performance metrics data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total synthesis operations performed
    pub total_syntheses: u64,

    /// Total processing time across all operations
    pub total_processing_time: Duration,

    /// Average synthesis time per operation
    pub average_synthesis_time: Duration,

    /// Peak memory usage (in bytes)
    pub peak_memory_usage: u64,

    /// Current memory usage (in bytes)
    pub current_memory_usage: u64,

    /// Cache hit rate percentage (0.0 - 1.0)
    pub cache_hit_rate: f64,

    /// Real-time factor statistics
    pub rtf_stats: RealTimeFactorStats,

    /// Per-component timing breakdown
    pub component_timings: HashMap<String, Duration>,

    /// Quality metrics
    pub quality_metrics: QualityMetrics,

    /// Feature-specific performance metrics
    pub feature_metrics: HashMap<AdvancedFeature, FeaturePerformanceMetrics>,

    /// Overall feature performance statistics
    pub feature_stats: FeaturePerformanceStats,
}

/// Real-time factor (RTF) statistics for streaming synthesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeFactorStats {
    /// Average real-time factor
    pub average_rtf: f64,

    /// Minimum real-time factor
    pub min_rtf: f64,

    /// Maximum real-time factor
    pub max_rtf: f64,

    /// 95th percentile real-time factor
    pub p95_rtf: f64,

    /// Number of real-time violations (RTF > 1.0)
    pub rtf_violations: u64,
}

/// Audio quality metrics tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Average signal-to-noise ratio
    pub average_snr: f64,

    /// Average total harmonic distortion
    pub average_thd: f64,

    /// Average dynamic range
    pub average_dynamic_range: f64,

    /// Number of quality warnings
    pub quality_warnings: u64,
}

/// Feature-specific performance metrics for advanced voice features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeaturePerformanceMetrics {
    /// Number of times this feature was used
    pub usage_count: u64,

    /// Total processing time for this feature
    pub total_processing_time: Duration,

    /// Average processing time per operation
    pub average_processing_time: Duration,

    /// Memory usage statistics for this feature
    pub memory_stats: FeatureMemoryStats,

    /// Real-time factor for this feature
    pub rtf_stats: RealTimeFactorStats,

    /// Quality metrics specific to this feature
    pub quality_stats: FeatureQualityStats,

    /// Error rate for this feature (0.0-1.0)
    pub error_rate: f64,

    /// Success rate for this feature (0.0-1.0)
    pub success_rate: f64,

    /// Feature-specific metrics
    pub feature_specific_metrics: HashMap<String, f64>,
}

/// Memory usage statistics for a specific feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMemoryStats {
    /// Peak memory usage for this feature (bytes)
    pub peak_memory: u64,

    /// Average memory usage (bytes)
    pub average_memory: u64,

    /// Current memory usage (bytes)
    pub current_memory: u64,

    /// Memory allocation count
    pub allocation_count: u64,

    /// Memory deallocation count
    pub deallocation_count: u64,
}

/// Quality statistics for a specific feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureQualityStats {
    /// Average quality score (0.0-1.0)
    pub average_quality: f64,

    /// Minimum quality score observed
    pub min_quality: f64,

    /// Maximum quality score observed
    pub max_quality: f64,

    /// Quality degradation events
    pub degradation_count: u64,

    /// Feature-specific quality metrics
    pub specific_metrics: HashMap<String, f64>,
}

/// Overall feature performance statistics across all features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeaturePerformanceStats {
    /// Total number of features used
    pub active_feature_count: u32,

    /// Most used feature
    pub most_used_feature: Option<AdvancedFeature>,

    /// Feature with best performance
    pub best_performing_feature: Option<AdvancedFeature>,

    /// Feature with worst performance
    pub worst_performing_feature: Option<AdvancedFeature>,

    /// Average memory overhead from features (bytes)
    pub average_feature_memory_overhead: u64,

    /// Average processing overhead from features (percentage)
    pub average_feature_processing_overhead: f64,

    /// Feature combination performance impact
    pub combination_impact: HashMap<Vec<AdvancedFeature>, f64>,

    /// Resource utilization by feature category
    pub category_utilization: HashMap<String, f64>,
}

/// Feature performance analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeaturePerformanceAnalysis {
    /// Performance summary
    pub summary: PerformanceSummary,

    /// Recommendations for optimization
    pub recommendations: Vec<PerformanceRecommendation>,

    /// Bottleneck analysis
    pub bottlenecks: Vec<PerformanceBottleneck>,

    /// Resource usage analysis
    pub resource_analysis: ResourceUsageAnalysis,
}

/// Performance summary for features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    /// Overall performance score (0.0-1.0)
    pub overall_score: f64,

    /// Features meeting performance targets
    pub features_meeting_targets: Vec<AdvancedFeature>,

    /// Features missing performance targets
    pub features_missing_targets: Vec<AdvancedFeature>,

    /// Critical performance issues
    pub critical_issues: Vec<String>,

    /// Performance trends
    pub trends: HashMap<AdvancedFeature, PerformanceTrend>,
}

/// Performance optimization recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecommendation {
    /// Recommendation ID
    pub id: String,

    /// Target feature
    pub feature: Option<AdvancedFeature>,

    /// Recommendation type
    pub recommendation_type: RecommendationType,

    /// Description
    pub description: String,

    /// Expected impact
    pub expected_impact: f64,

    /// Implementation difficulty (0.0-1.0)
    pub difficulty: f64,

    /// Priority level
    pub priority: RecommendationPriority,
}

/// Types of performance recommendations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Reduce memory usage
    ReduceMemory,
    /// Optimize processing speed
    OptimizeSpeed,
    /// Improve quality
    ImproveQuality,
    /// Enable GPU acceleration
    EnableGpu,
    /// Adjust feature configuration
    AdjustConfiguration,
    /// Disable underperforming features
    DisableFeatures,
    /// Optimize feature combinations
    OptimizeCombinations,
    /// Upgrade hardware
    UpgradeHardware,
}

/// Priority levels for recommendations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecommendationPriority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Performance bottleneck identification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    /// Bottleneck location
    pub location: String,

    /// Affected feature
    pub feature: Option<AdvancedFeature>,

    /// Bottleneck type
    pub bottleneck_type: BottleneckType,

    /// Severity (0.0-1.0)
    pub severity: f64,

    /// Impact description
    pub impact: String,

    /// Suggested solutions
    pub solutions: Vec<String>,
}

/// Types of performance bottlenecks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    /// CPU processing bottleneck
    CpuProcessing,
    /// Memory bandwidth bottleneck
    MemoryBandwidth,
    /// GPU processing bottleneck
    GpuProcessing,
    /// I/O operations bottleneck
    IoOperations,
    /// Network bandwidth bottleneck
    NetworkBandwidth,
    /// Feature interaction bottleneck
    FeatureInteraction,
    /// Resource contention
    ResourceContention,
}

/// Resource usage analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageAnalysis {
    /// CPU utilization by feature
    pub cpu_utilization: HashMap<AdvancedFeature, f64>,

    /// Memory utilization by feature
    pub memory_utilization: HashMap<AdvancedFeature, f64>,

    /// GPU utilization by feature (if applicable)
    pub gpu_utilization: HashMap<AdvancedFeature, f64>,

    /// Resource efficiency scores
    pub efficiency_scores: HashMap<AdvancedFeature, f64>,

    /// Resource waste detection
    pub waste_detection: Vec<ResourceWaste>,
}

/// Resource waste detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceWaste {
    /// Resource type
    pub resource_type: String,

    /// Waste amount
    pub waste_amount: f64,

    /// Waste percentage
    pub waste_percentage: f64,

    /// Cause of waste
    pub cause: String,

    /// Suggested fix
    pub suggested_fix: String,
}

/// Performance trend analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceTrend {
    /// Performance is improving
    Improving,
    /// Performance is stable
    Stable,
    /// Performance is degrading
    Degrading,
    /// Performance is highly variable
    Variable,
}

/// Performance measurement scope for automatic timing.
pub struct PerformanceScope<'a> {
    monitor: &'a PerformanceMonitor,
    operation_name: String,
    start_time: Instant,
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceMonitor {
    /// Create a new performance monitor.
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
            start_time: Instant::now(),
        }
    }

    /// Start measuring a specific operation.
    pub fn start_operation(&self, operation_name: &str) -> PerformanceScope<'_> {
        PerformanceScope {
            monitor: self,
            operation_name: operation_name.to_string(),
            start_time: Instant::now(),
        }
    }

    /// Record a synthesis operation.
    pub fn record_synthesis(
        &self,
        processing_time: Duration,
        audio_duration: Duration,
    ) -> Result<()> {
        let mut metrics = self
            .metrics
            .lock()
            .map_err(|_| VoirsError::internal("PerformanceMonitor", "Failed to lock metrics"))?;

        metrics.total_syntheses += 1;
        metrics.total_processing_time += processing_time;
        metrics.average_synthesis_time =
            metrics.total_processing_time / metrics.total_syntheses as u32;

        // Calculate real-time factor
        let rtf = processing_time.as_secs_f64() / audio_duration.as_secs_f64();
        self.update_rtf_stats(&mut metrics.rtf_stats, rtf);

        Ok(())
    }

    /// Update memory usage metrics.
    pub fn update_memory_usage(&self, current_usage: u64) -> Result<()> {
        let mut metrics = self
            .metrics
            .lock()
            .map_err(|_| VoirsError::internal("PerformanceMonitor", "Failed to lock metrics"))?;

        metrics.current_memory_usage = current_usage;
        if current_usage > metrics.peak_memory_usage {
            metrics.peak_memory_usage = current_usage;
        }

        Ok(())
    }

    /// Record cache statistics.
    pub fn record_cache_hit_rate(&self, hit_rate: f64) -> Result<()> {
        let mut metrics = self
            .metrics
            .lock()
            .map_err(|_| VoirsError::internal("PerformanceMonitor", "Failed to lock metrics"))?;
        metrics.cache_hit_rate = hit_rate;
        Ok(())
    }

    /// Record quality metrics.
    pub fn record_quality_metrics(&self, snr: f64, thd: f64, dynamic_range: f64) -> Result<()> {
        let mut metrics = self
            .metrics
            .lock()
            .map_err(|_| VoirsError::internal("PerformanceMonitor", "Failed to lock metrics"))?;

        // Update running averages
        let count = metrics.total_syntheses as f64;
        if count > 0.0 {
            metrics.quality_metrics.average_snr =
                (metrics.quality_metrics.average_snr * (count - 1.0) + snr) / count;
            metrics.quality_metrics.average_thd =
                (metrics.quality_metrics.average_thd * (count - 1.0) + thd) / count;
            metrics.quality_metrics.average_dynamic_range =
                (metrics.quality_metrics.average_dynamic_range * (count - 1.0) + dynamic_range)
                    / count;
        }

        // Check for quality warnings
        if snr < 20.0 || thd > 0.05 || dynamic_range < 30.0 {
            metrics.quality_metrics.quality_warnings += 1;
        }

        Ok(())
    }

    /// Get current performance metrics.
    pub fn get_metrics(&self) -> Result<PerformanceMetrics> {
        let metrics = self
            .metrics
            .lock()
            .map_err(|_| VoirsError::internal("PerformanceMonitor", "Failed to lock metrics"))?;
        Ok(metrics.clone())
    }

    /// Generate a performance report.
    pub fn generate_report(&self) -> Result<String> {
        let metrics = self.get_metrics()?;
        let uptime = self.start_time.elapsed();

        Ok(format!(
            "VoiRS SDK Performance Report\n\
            ===========================\n\
            Uptime: {:.2}s\n\
            Total Syntheses: {}\n\
            Average Synthesis Time: {:.2}ms\n\
            Peak Memory Usage: {:.2} MB\n\
            Current Memory Usage: {:.2} MB\n\
            Cache Hit Rate: {:.1}%\n\
            Average RTF: {:.3}\n\
            RTF Violations: {}\n\
            Average SNR: {:.1} dB\n\
            Average THD: {:.3}%\n\
            Quality Warnings: {}\n",
            uptime.as_secs_f64(),
            metrics.total_syntheses,
            metrics.average_synthesis_time.as_millis(),
            metrics.peak_memory_usage as f64 / 1_048_576.0,
            metrics.current_memory_usage as f64 / 1_048_576.0,
            metrics.cache_hit_rate * 100.0,
            metrics.rtf_stats.average_rtf,
            metrics.rtf_stats.rtf_violations,
            metrics.quality_metrics.average_snr,
            metrics.quality_metrics.average_thd * 100.0,
            metrics.quality_metrics.quality_warnings,
        ))
    }

    /// Reset all metrics.
    pub fn reset(&self) -> Result<()> {
        let mut metrics = self
            .metrics
            .lock()
            .map_err(|_| VoirsError::internal("PerformanceMonitor", "Failed to lock metrics"))?;
        *metrics = PerformanceMetrics::default();
        Ok(())
    }

    fn update_rtf_stats(&self, stats: &mut RealTimeFactorStats, rtf: f64) {
        let count = stats.average_rtf;
        if count == 0.0 {
            stats.average_rtf = rtf;
            stats.min_rtf = rtf;
            stats.max_rtf = rtf;
        } else {
            // Simple running average (could be enhanced with proper statistical tracking)
            stats.average_rtf = (stats.average_rtf + rtf) / 2.0;
            stats.min_rtf = stats.min_rtf.min(rtf);
            stats.max_rtf = stats.max_rtf.max(rtf);
        }

        if rtf > 1.0 {
            stats.rtf_violations += 1;
        }

        // Approximate 95th percentile (simplified)
        stats.p95_rtf = stats.max_rtf * 0.95;
    }

    /// Record feature-specific performance metrics.
    pub fn record_feature_operation(
        &self,
        feature: AdvancedFeature,
        processing_time: Duration,
        memory_usage: u64,
        quality_score: f64,
        success: bool,
    ) -> Result<()> {
        let mut metrics = self
            .metrics
            .lock()
            .map_err(|_| VoirsError::internal("PerformanceMonitor", "Failed to lock metrics"))?;

        // Update feature metrics
        {
            let feature_metrics = metrics
                .feature_metrics
                .entry(feature)
                .or_insert_with(FeaturePerformanceMetrics::default);

            // Update usage count
            feature_metrics.usage_count += 1;

            // Update timing metrics
            feature_metrics.total_processing_time += processing_time;
            feature_metrics.average_processing_time =
                feature_metrics.total_processing_time / feature_metrics.usage_count as u32;

            // Update memory metrics
            feature_metrics.memory_stats.current_memory = memory_usage;
            if memory_usage > feature_metrics.memory_stats.peak_memory {
                feature_metrics.memory_stats.peak_memory = memory_usage;
            }
            feature_metrics.memory_stats.average_memory =
                (feature_metrics.memory_stats.average_memory
                    * (feature_metrics.usage_count - 1) as u64
                    + memory_usage)
                    / feature_metrics.usage_count;

            // Update quality metrics
            let old_quality = feature_metrics.quality_stats.average_quality;
            feature_metrics.quality_stats.average_quality =
                (old_quality * (feature_metrics.usage_count - 1) as f64 + quality_score)
                    / feature_metrics.usage_count as f64;

            if quality_score < feature_metrics.quality_stats.min_quality {
                feature_metrics.quality_stats.min_quality = quality_score;
            }
            if quality_score > feature_metrics.quality_stats.max_quality {
                feature_metrics.quality_stats.max_quality = quality_score;
            }

            // Update success/error rates
            let total_ops = feature_metrics.usage_count as f64;
            if success {
                feature_metrics.success_rate =
                    (feature_metrics.success_rate * (total_ops - 1.0) + 1.0) / total_ops;
            } else {
                feature_metrics.error_rate =
                    (feature_metrics.error_rate * (total_ops - 1.0) + 1.0) / total_ops;
            }
            feature_metrics.success_rate = 1.0 - feature_metrics.error_rate;
        }

        // Update overall feature stats with cloned metrics to avoid borrow issues
        let feature_metrics_clone = metrics.feature_metrics.get(&feature).unwrap().clone();
        self.update_feature_stats(&mut metrics.feature_stats, feature, &feature_metrics_clone)?;

        Ok(())
    }

    /// Record feature-specific metric.
    pub fn record_feature_metric(
        &self,
        feature: AdvancedFeature,
        metric_name: &str,
        value: f64,
    ) -> Result<()> {
        let mut metrics = self
            .metrics
            .lock()
            .map_err(|_| VoirsError::internal("PerformanceMonitor", "Failed to lock metrics"))?;

        let feature_metrics = metrics
            .feature_metrics
            .entry(feature)
            .or_insert_with(FeaturePerformanceMetrics::default);
        feature_metrics
            .feature_specific_metrics
            .insert(metric_name.to_string(), value);

        Ok(())
    }

    /// Get feature performance analysis.
    pub fn analyze_feature_performance(&self) -> Result<FeaturePerformanceAnalysis> {
        let metrics = self
            .metrics
            .lock()
            .map_err(|_| VoirsError::internal("PerformanceMonitor", "Failed to lock metrics"))?;

        let summary = self.generate_performance_summary(&metrics)?;
        let recommendations = self.generate_recommendations(&metrics)?;
        let bottlenecks = self.identify_bottlenecks(&metrics)?;
        let resource_analysis = self.analyze_resource_usage(&metrics)?;

        Ok(FeaturePerformanceAnalysis {
            summary,
            recommendations,
            bottlenecks,
            resource_analysis,
        })
    }

    /// Get feature-specific metrics.
    pub fn get_feature_metrics(
        &self,
        feature: AdvancedFeature,
    ) -> Result<Option<FeaturePerformanceMetrics>> {
        let metrics = self
            .metrics
            .lock()
            .map_err(|_| VoirsError::internal("PerformanceMonitor", "Failed to lock metrics"))?;

        Ok(metrics.feature_metrics.get(&feature).cloned())
    }

    /// Get overall feature performance statistics.
    pub fn get_feature_stats(&self) -> Result<FeaturePerformanceStats> {
        let metrics = self
            .metrics
            .lock()
            .map_err(|_| VoirsError::internal("PerformanceMonitor", "Failed to lock metrics"))?;

        Ok(metrics.feature_stats.clone())
    }

    /// Reset feature-specific metrics.
    pub fn reset_feature_metrics(&self, feature: Option<AdvancedFeature>) -> Result<()> {
        let mut metrics = self
            .metrics
            .lock()
            .map_err(|_| VoirsError::internal("PerformanceMonitor", "Failed to lock metrics"))?;

        if let Some(feature) = feature {
            metrics.feature_metrics.remove(&feature);
        } else {
            metrics.feature_metrics.clear();
            metrics.feature_stats = FeaturePerformanceStats::default();
        }

        Ok(())
    }

    /// Update overall feature statistics.
    fn update_feature_stats(
        &self,
        stats: &mut FeaturePerformanceStats,
        feature: AdvancedFeature,
        feature_metrics: &FeaturePerformanceMetrics,
    ) -> Result<()> {
        // Update active feature count
        stats.active_feature_count = stats.active_feature_count.max(1);

        // Find most used feature
        if stats.most_used_feature.is_none() || feature_metrics.usage_count > 0 {
            stats.most_used_feature = Some(feature);
        }

        // Find best/worst performing features based on average processing time
        let avg_time = feature_metrics.average_processing_time.as_millis() as f64;
        if stats.best_performing_feature.is_none() {
            stats.best_performing_feature = Some(feature);
        }
        if stats.worst_performing_feature.is_none() {
            stats.worst_performing_feature = Some(feature);
        }

        // Update memory overhead
        stats.average_feature_memory_overhead = (stats.average_feature_memory_overhead
            + feature_metrics.memory_stats.average_memory)
            / 2;

        // Update processing overhead (simplified calculation)
        let processing_overhead = avg_time / 1000.0; // Convert to seconds
        stats.average_feature_processing_overhead =
            (stats.average_feature_processing_overhead + processing_overhead) / 2.0;

        Ok(())
    }

    /// Generate performance summary.
    fn generate_performance_summary(
        &self,
        metrics: &PerformanceMetrics,
    ) -> Result<PerformanceSummary> {
        let mut features_meeting_targets = Vec::new();
        let mut features_missing_targets = Vec::new();
        let mut critical_issues = Vec::new();
        let mut trends = HashMap::new();

        // Analyze each feature
        for (feature, feature_metrics) in &metrics.feature_metrics {
            // Check if feature meets performance targets (simplified criteria)
            let meets_targets = feature_metrics.average_processing_time.as_millis() < 500
                && feature_metrics.error_rate < 0.05
                && feature_metrics.quality_stats.average_quality > 0.8;

            if meets_targets {
                features_meeting_targets.push(*feature);
            } else {
                features_missing_targets.push(*feature);
            }

            // Determine trend (simplified)
            let trend = if feature_metrics.error_rate > 0.1 {
                PerformanceTrend::Degrading
            } else if feature_metrics.quality_stats.average_quality > 0.9 {
                PerformanceTrend::Improving
            } else {
                PerformanceTrend::Stable
            };
            trends.insert(*feature, trend);

            // Check for critical issues
            if feature_metrics.error_rate > 0.2 {
                critical_issues.push(format!(
                    "High error rate for {:?}: {:.1}%",
                    feature,
                    feature_metrics.error_rate * 100.0
                ));
            }
            if feature_metrics.average_processing_time.as_millis() > 2000 {
                critical_issues.push(format!(
                    "High latency for {:?}: {}ms",
                    feature,
                    feature_metrics.average_processing_time.as_millis()
                ));
            }
        }

        // Calculate overall score
        let total_features = metrics.feature_metrics.len() as f64;
        let meeting_targets = features_meeting_targets.len() as f64;
        let overall_score = if total_features > 0.0 {
            meeting_targets / total_features
        } else {
            1.0
        };

        Ok(PerformanceSummary {
            overall_score,
            features_meeting_targets,
            features_missing_targets,
            critical_issues,
            trends,
        })
    }

    /// Generate optimization recommendations.
    fn generate_recommendations(
        &self,
        metrics: &PerformanceMetrics,
    ) -> Result<Vec<PerformanceRecommendation>> {
        let mut recommendations = Vec::new();

        for (feature, feature_metrics) in &metrics.feature_metrics {
            // High memory usage recommendation
            if feature_metrics.memory_stats.peak_memory > 1_000_000_000 {
                // 1GB
                recommendations.push(PerformanceRecommendation {
                    id: format!("mem_reduce_{:?}", feature),
                    feature: Some(*feature),
                    recommendation_type: RecommendationType::ReduceMemory,
                    description: format!("Feature {:?} is using high memory ({} MB). Consider optimizing memory usage.", 
                        feature, feature_metrics.memory_stats.peak_memory / 1_000_000),
                    expected_impact: 0.3,
                    difficulty: 0.6,
                    priority: RecommendationPriority::Medium,
                });
            }

            // High latency recommendation
            if feature_metrics.average_processing_time.as_millis() > 1000 {
                recommendations.push(PerformanceRecommendation {
                    id: format!("speed_optimize_{:?}", feature),
                    feature: Some(*feature),
                    recommendation_type: RecommendationType::OptimizeSpeed,
                    description: format!("Feature {:?} has high processing latency ({}ms). Consider optimization or GPU acceleration.", 
                        feature, feature_metrics.average_processing_time.as_millis()),
                    expected_impact: 0.5,
                    difficulty: 0.7,
                    priority: RecommendationPriority::High,
                });
            }

            // Low quality recommendation
            if feature_metrics.quality_stats.average_quality < 0.7 {
                recommendations.push(PerformanceRecommendation {
                    id: format!("quality_improve_{:?}", feature),
                    feature: Some(*feature),
                    recommendation_type: RecommendationType::ImproveQuality,
                    description: format!("Feature {:?} has low quality score ({:.2}). Consider adjusting configuration.", 
                        feature, feature_metrics.quality_stats.average_quality),
                    expected_impact: 0.4,
                    difficulty: 0.5,
                    priority: RecommendationPriority::Medium,
                });
            }
        }

        Ok(recommendations)
    }

    /// Identify performance bottlenecks.
    fn identify_bottlenecks(
        &self,
        metrics: &PerformanceMetrics,
    ) -> Result<Vec<PerformanceBottleneck>> {
        let mut bottlenecks = Vec::new();

        for (feature, feature_metrics) in &metrics.feature_metrics {
            // CPU bottleneck detection
            if feature_metrics.average_processing_time.as_millis() > 2000 {
                bottlenecks.push(PerformanceBottleneck {
                    location: format!("Feature {:?} processing", feature),
                    feature: Some(*feature),
                    bottleneck_type: BottleneckType::CpuProcessing,
                    severity: 0.8,
                    impact: "High processing latency affects real-time performance".to_string(),
                    solutions: vec![
                        "Enable GPU acceleration if available".to_string(),
                        "Optimize algorithm implementation".to_string(),
                        "Reduce feature complexity".to_string(),
                    ],
                });
            }

            // Memory bottleneck detection
            if feature_metrics.memory_stats.peak_memory > 2_000_000_000 {
                // 2GB
                bottlenecks.push(PerformanceBottleneck {
                    location: format!("Feature {:?} memory usage", feature),
                    feature: Some(*feature),
                    bottleneck_type: BottleneckType::MemoryBandwidth,
                    severity: 0.6,
                    impact: "High memory usage may cause system instability".to_string(),
                    solutions: vec![
                        "Implement memory pooling".to_string(),
                        "Optimize data structures".to_string(),
                        "Add memory usage limits".to_string(),
                    ],
                });
            }
        }

        Ok(bottlenecks)
    }

    /// Analyze resource usage across features.
    fn analyze_resource_usage(
        &self,
        metrics: &PerformanceMetrics,
    ) -> Result<ResourceUsageAnalysis> {
        let mut cpu_utilization = HashMap::new();
        let mut memory_utilization = HashMap::new();
        let mut gpu_utilization = HashMap::new();
        let mut efficiency_scores = HashMap::new();
        let waste_detection = Vec::new();

        for (feature, feature_metrics) in &metrics.feature_metrics {
            // Simplified CPU utilization calculation
            let cpu_util =
                (feature_metrics.average_processing_time.as_millis() as f64 / 1000.0).min(1.0);
            cpu_utilization.insert(*feature, cpu_util);

            // Memory utilization as percentage of peak system memory
            let mem_util =
                (feature_metrics.memory_stats.average_memory as f64 / 4_000_000_000.0).min(1.0); // Assume 4GB system
            memory_utilization.insert(*feature, mem_util);

            // GPU utilization (placeholder)
            gpu_utilization.insert(*feature, 0.0);

            // Efficiency score (quality per unit of resources)
            let efficiency =
                feature_metrics.quality_stats.average_quality / (cpu_util + mem_util + 0.1);
            efficiency_scores.insert(*feature, efficiency);
        }

        Ok(ResourceUsageAnalysis {
            cpu_utilization,
            memory_utilization,
            gpu_utilization,
            efficiency_scores,
            waste_detection,
        })
    }
}

impl<'a> Drop for PerformanceScope<'a> {
    fn drop(&mut self) {
        let elapsed = self.start_time.elapsed();
        if let Ok(mut metrics) = self.monitor.metrics.lock() {
            metrics
                .component_timings
                .insert(self.operation_name.clone(), elapsed);
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_syntheses: 0,
            total_processing_time: Duration::ZERO,
            average_synthesis_time: Duration::ZERO,
            peak_memory_usage: 0,
            current_memory_usage: 0,
            cache_hit_rate: 0.0,
            rtf_stats: RealTimeFactorStats::default(),
            component_timings: HashMap::new(),
            quality_metrics: QualityMetrics::default(),
            feature_metrics: HashMap::new(),
            feature_stats: FeaturePerformanceStats::default(),
        }
    }
}

impl Default for RealTimeFactorStats {
    fn default() -> Self {
        Self {
            average_rtf: 0.0,
            min_rtf: 0.0,
            max_rtf: 0.0,
            p95_rtf: 0.0,
            rtf_violations: 0,
        }
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            average_snr: 0.0,
            average_thd: 0.0,
            average_dynamic_range: 0.0,
            quality_warnings: 0,
        }
    }
}

impl Default for FeaturePerformanceMetrics {
    fn default() -> Self {
        Self {
            usage_count: 0,
            total_processing_time: Duration::ZERO,
            average_processing_time: Duration::ZERO,
            memory_stats: FeatureMemoryStats::default(),
            rtf_stats: RealTimeFactorStats::default(),
            quality_stats: FeatureQualityStats::default(),
            error_rate: 0.0,
            success_rate: 1.0,
            feature_specific_metrics: HashMap::new(),
        }
    }
}

impl Default for FeatureMemoryStats {
    fn default() -> Self {
        Self {
            peak_memory: 0,
            average_memory: 0,
            current_memory: 0,
            allocation_count: 0,
            deallocation_count: 0,
        }
    }
}

impl Default for FeatureQualityStats {
    fn default() -> Self {
        Self {
            average_quality: 1.0,
            min_quality: 1.0,
            max_quality: 1.0,
            degradation_count: 0,
            specific_metrics: HashMap::new(),
        }
    }
}

impl Default for FeaturePerformanceStats {
    fn default() -> Self {
        Self {
            active_feature_count: 0,
            most_used_feature: None,
            best_performing_feature: None,
            worst_performing_feature: None,
            average_feature_memory_overhead: 0,
            average_feature_processing_overhead: 0.0,
            combination_impact: HashMap::new(),
            category_utilization: HashMap::new(),
        }
    }
}

/// Convenience macro for measuring operation performance.
#[macro_export]
macro_rules! measure_performance {
    ($monitor:expr, $operation:expr, $code:block) => {{
        let _scope = $monitor.start_operation($operation);
        $code
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_performance_monitor_creation() {
        let monitor = PerformanceMonitor::new();
        let metrics = monitor.get_metrics().unwrap();
        assert_eq!(metrics.total_syntheses, 0);
    }

    #[test]
    fn test_synthesis_recording() {
        let monitor = PerformanceMonitor::new();
        monitor
            .record_synthesis(Duration::from_millis(100), Duration::from_millis(1000))
            .unwrap();

        let metrics = monitor.get_metrics().unwrap();
        assert_eq!(metrics.total_syntheses, 1);
        assert_eq!(metrics.total_processing_time, Duration::from_millis(100));
    }

    #[test]
    fn test_memory_tracking() {
        let monitor = PerformanceMonitor::new();
        monitor.update_memory_usage(1024).unwrap();
        monitor.update_memory_usage(2048).unwrap();

        let metrics = monitor.get_metrics().unwrap();
        assert_eq!(metrics.current_memory_usage, 2048);
        assert_eq!(metrics.peak_memory_usage, 2048);
    }

    #[test]
    fn test_performance_scope() {
        let monitor = PerformanceMonitor::new();
        {
            let _scope = monitor.start_operation("test_operation");
            thread::sleep(Duration::from_millis(1));
        }

        let metrics = monitor.get_metrics().unwrap();
        assert!(metrics.component_timings.contains_key("test_operation"));
    }

    #[test]
    fn test_quality_metrics() {
        let monitor = PerformanceMonitor::new();
        monitor
            .record_synthesis(Duration::from_millis(100), Duration::from_millis(1000))
            .unwrap();
        monitor.record_quality_metrics(25.0, 0.02, 40.0).unwrap();

        let metrics = monitor.get_metrics().unwrap();
        assert_eq!(metrics.quality_metrics.average_snr, 25.0);
        assert_eq!(metrics.quality_metrics.quality_warnings, 0);
    }

    #[test]
    fn test_performance_report() {
        let monitor = PerformanceMonitor::new();
        monitor
            .record_synthesis(Duration::from_millis(100), Duration::from_millis(1000))
            .unwrap();

        let report = monitor.generate_report().unwrap();
        assert!(report.contains("VoiRS SDK Performance Report"));
        assert!(report.contains("Total Syntheses: 1"));
    }

    #[test]
    fn test_reset_metrics() {
        let monitor = PerformanceMonitor::new();
        monitor
            .record_synthesis(Duration::from_millis(100), Duration::from_millis(1000))
            .unwrap();
        monitor.reset().unwrap();

        let metrics = monitor.get_metrics().unwrap();
        assert_eq!(metrics.total_syntheses, 0);
    }
}
