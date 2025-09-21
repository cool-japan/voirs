//! Adaptive quality control system for vocoders
//!
//! Provides dynamic quality adjustment based on:
//! - Real-time performance metrics
//! - Quality feedback loops
//! - System resource constraints
//! - User preferences and targets

use crate::metrics::{QualityCalculator, QualityConfig, QualityMetrics};
use crate::performance::{PerformanceMetrics, PerformanceThresholds};
use crate::{AudioBuffer, Result};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Quality target specification
#[derive(Debug, Clone)]
pub struct QualityTarget {
    /// Target MOS score (1.0-5.0)
    pub target_mos: f32,
    /// Minimum acceptable MOS
    pub min_mos: f32,
    /// Maximum acceptable latency (ms)
    pub max_latency_ms: f32,
    /// Target real-time factor
    pub target_rtf: f32,
    /// Quality vs speed preference (0.0=speed, 1.0=quality)
    pub quality_preference: f32,
}

impl Default for QualityTarget {
    fn default() -> Self {
        Self {
            target_mos: 4.0,
            min_mos: 3.0,
            max_latency_ms: 100.0,
            target_rtf: 0.5,
            quality_preference: 0.7,
        }
    }
}

/// Adaptive configuration parameters
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Model quality level (0.0-1.0)
    pub quality_level: f32,
    /// Processing batch size
    pub batch_size: usize,
    /// Number of parallel streams
    pub num_streams: usize,
    /// Enable/disable expensive processing
    pub enable_expensive_processing: bool,
    /// Upsampling quality factor
    pub upsampling_factor: f32,
    /// Noise suppression strength
    pub noise_suppression: f32,
    /// Dynamic range compression
    pub compression_ratio: f32,
    /// Processing precision mode
    pub precision_mode: PrecisionMode,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            quality_level: 0.8,
            batch_size: 16,
            num_streams: 4,
            enable_expensive_processing: true,
            upsampling_factor: 1.0,
            noise_suppression: 0.3,
            compression_ratio: 1.0,
            precision_mode: PrecisionMode::High,
        }
    }
}

/// Processing precision modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionMode {
    /// Low precision, high speed (e.g., FP16)
    Low,
    /// Medium precision, balanced (e.g., mixed precision)
    Medium,
    /// High precision, high quality (e.g., FP32)
    High,
    /// Ultra precision for critical applications (e.g., FP64)
    Ultra,
}

impl PrecisionMode {
    /// Get relative speed multiplier
    pub fn speed_multiplier(&self) -> f32 {
        match self {
            PrecisionMode::Low => 2.0,
            PrecisionMode::Medium => 1.5,
            PrecisionMode::High => 1.0,
            PrecisionMode::Ultra => 0.5,
        }
    }

    /// Get relative quality multiplier
    pub fn quality_multiplier(&self) -> f32 {
        match self {
            PrecisionMode::Low => 0.8,
            PrecisionMode::Medium => 0.9,
            PrecisionMode::High => 1.0,
            PrecisionMode::Ultra => 1.1,
        }
    }
}

/// Quality adjustment decision
#[derive(Debug, Clone)]
pub struct QualityAdjustment {
    /// New configuration to apply
    pub config: AdaptiveConfig,
    /// Reason for the adjustment
    pub reason: String,
    /// Expected impact on quality
    pub quality_impact: f32,
    /// Expected impact on performance
    pub performance_impact: f32,
    /// Confidence in this adjustment (0.0-1.0)
    pub confidence: f32,
}

/// Quality control history entry
#[derive(Debug, Clone)]
pub struct QualityHistoryEntry {
    /// Timestamp
    pub timestamp: Instant,
    /// Configuration used
    pub config: AdaptiveConfig,
    /// Measured quality metrics
    pub quality: QualityMetrics,
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// User satisfaction (if available)
    pub user_satisfaction: Option<f32>,
}

/// Adaptive quality controller
pub struct AdaptiveQualityController {
    /// Current configuration
    current_config: AdaptiveConfig,
    /// Quality targets
    quality_target: QualityTarget,
    /// Performance thresholds
    performance_thresholds: PerformanceThresholds,
    /// History of adjustments and results
    history: VecDeque<QualityHistoryEntry>,
    /// Maximum history size
    max_history: usize,
    /// Quality calculator for evaluation
    quality_calculator: QualityCalculator,
    /// Last adjustment time
    last_adjustment: Instant,
    /// Minimum time between adjustments
    adjustment_cooldown: Duration,
    /// Learning rate for adaptive adjustments
    learning_rate: f32,
    /// Performance weight in decision making
    performance_weight: f32,
    /// Quality weight in decision making
    quality_weight: f32,
}

impl AdaptiveQualityController {
    /// Create new adaptive quality controller
    pub fn new(
        initial_config: AdaptiveConfig,
        quality_target: QualityTarget,
        performance_thresholds: PerformanceThresholds,
    ) -> Self {
        let cooldown_duration = Duration::from_secs(10);
        Self {
            current_config: initial_config,
            quality_target,
            performance_thresholds,
            history: VecDeque::new(),
            max_history: 100,
            quality_calculator: QualityCalculator::new(QualityConfig::default()),
            last_adjustment: Instant::now() - cooldown_duration - Duration::from_secs(1), // Allow immediate adjustments
            adjustment_cooldown: cooldown_duration, // Wait 10s between adjustments
            learning_rate: 0.1,
            performance_weight: 0.4,
            quality_weight: 0.6,
        }
    }

    /// Update controller with new performance and quality metrics
    pub fn update_metrics(
        &mut self,
        quality: QualityMetrics,
        performance: PerformanceMetrics,
        user_satisfaction: Option<f32>,
    ) {
        // Add to history
        let entry = QualityHistoryEntry {
            timestamp: Instant::now(),
            config: self.current_config.clone(),
            quality,
            performance,
            user_satisfaction,
        };

        self.history.push_back(entry);
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Determine if quality adjustment is needed
    pub fn should_adjust(&self) -> bool {
        if self.last_adjustment.elapsed() < self.adjustment_cooldown {
            return false; // Too soon to adjust again
        }

        if self.history.is_empty() {
            return false; // No data to make decisions
        }

        let recent_entry = &self.history[self.history.len() - 1];

        // Check if quality is below target
        let quality_below_target =
            recent_entry.quality.mos_estimate < self.quality_target.target_mos;

        // Check if performance is above thresholds
        let performance_issues = !recent_entry
            .performance
            .is_acceptable(&self.performance_thresholds);

        // Check if user satisfaction is low (if available)
        let user_dissatisfied = if let Some(satisfaction) = recent_entry.user_satisfaction {
            satisfaction < 0.6 // Below 60% satisfaction
        } else {
            false
        };

        // Weighted decision making using configured weights
        let quality_factor = if quality_below_target { 1.0 } else { 0.0 };
        let performance_factor = if performance_issues { 1.0 } else { 0.0 };
        let user_factor = if user_dissatisfied { 1.0 } else { 0.0 };

        let weighted_score = quality_factor * self.quality_weight
            + performance_factor * self.performance_weight
            + user_factor * (1.0 - self.quality_weight - self.performance_weight).max(0.0);

        weighted_score > 0.5 // Adjust if weighted score indicates need
    }

    /// Generate quality adjustment recommendation
    pub fn recommend_adjustment(&self) -> Option<QualityAdjustment> {
        if !self.should_adjust() {
            return None;
        }

        let recent_entry = &self.history[self.history.len() - 1];
        let mut new_config = self.current_config.clone();
        let mut reasons = Vec::new();
        let mut quality_impact = 0.0;
        let mut performance_impact = 0.0;

        // Analyze current situation
        let quality_score = recent_entry.quality.mos_estimate;
        let target_score = self.quality_target.target_mos;
        let performance_score = recent_entry
            .performance
            .performance_score(&self.performance_thresholds);

        // Quality too low - increase quality settings
        if quality_score < target_score {
            let quality_gap = target_score - quality_score;

            if quality_gap > 0.5 {
                // Significant quality issues - major adjustments scaled by learning rate
                let major_adjustment = 0.2 * (1.0 + self.learning_rate);
                new_config.quality_level = (new_config.quality_level + major_adjustment).min(1.0);
                new_config.precision_mode = match new_config.precision_mode {
                    PrecisionMode::Low => PrecisionMode::Medium,
                    PrecisionMode::Medium => PrecisionMode::High,
                    PrecisionMode::High => PrecisionMode::Ultra,
                    PrecisionMode::Ultra => PrecisionMode::Ultra,
                };
                new_config.enable_expensive_processing = true;
                new_config.upsampling_factor =
                    (new_config.upsampling_factor + major_adjustment).min(2.0);

                reasons.push("Significant quality improvement needed".to_string());
                quality_impact = 0.3 * (1.0 + self.learning_rate);
                performance_impact = -0.2 * (1.0 + self.learning_rate);
            } else {
                // Minor quality improvements scaled by learning rate
                let minor_adjustment = 0.1 * (1.0 + self.learning_rate);
                new_config.quality_level = (new_config.quality_level + minor_adjustment).min(1.0);
                new_config.noise_suppression =
                    (new_config.noise_suppression + minor_adjustment).min(1.0);

                reasons.push("Minor quality adjustment".to_string());
                quality_impact = 0.1 * (1.0 + self.learning_rate);
                performance_impact = -0.05 * (1.0 + self.learning_rate);
            }
        }

        // Performance issues - reduce computational load
        if performance_score < 0.5 {
            if recent_entry.performance.latency_ms > self.performance_thresholds.max_latency_ms {
                // Latency issues - reduce processing time
                new_config.batch_size = (new_config.batch_size / 2).max(1);
                new_config.num_streams = (new_config.num_streams / 2).max(1);

                if new_config.precision_mode != PrecisionMode::Low {
                    new_config.precision_mode = match new_config.precision_mode {
                        PrecisionMode::Ultra => PrecisionMode::High,
                        PrecisionMode::High => PrecisionMode::Medium,
                        PrecisionMode::Medium => PrecisionMode::Low,
                        PrecisionMode::Low => PrecisionMode::Low,
                    };
                }

                reasons.push("Reducing latency".to_string());
                quality_impact = -0.1;
                performance_impact = 0.3;
            }

            if recent_entry.performance.cpu_usage > self.performance_thresholds.max_cpu_usage {
                // CPU usage too high
                new_config.enable_expensive_processing = false;
                new_config.quality_level = (new_config.quality_level - 0.1).max(0.1);

                reasons.push("Reducing CPU usage".to_string());
                quality_impact = -0.15;
                performance_impact = 0.2;
            }
        }

        // User satisfaction adjustments
        if let Some(satisfaction) = recent_entry.user_satisfaction {
            if satisfaction < 0.4 {
                // Very low satisfaction - prioritize quality over performance
                new_config.quality_level = (new_config.quality_level + 0.3).min(1.0);
                new_config.precision_mode = PrecisionMode::High;
                new_config.enable_expensive_processing = true;

                reasons.push("Improving user satisfaction".to_string());
                quality_impact += 0.2;
                performance_impact -= 0.15;
            }
        }

        // Apply quality preference weighting
        let quality_preference_factor = self.quality_target.quality_preference;
        if quality_preference_factor > 0.8 {
            // Strong preference for quality
            new_config.quality_level = (new_config.quality_level + 0.1).min(1.0);
            quality_impact += 0.1;
            performance_impact -= 0.05;
        } else if quality_preference_factor < 0.3 {
            // Strong preference for speed
            new_config.quality_level = (new_config.quality_level - 0.1).max(0.1);
            quality_impact -= 0.1;
            performance_impact += 0.05;
        }

        // Calculate confidence based on history consistency
        let confidence = self.calculate_adjustment_confidence(&new_config);

        if reasons.is_empty() {
            return None;
        }

        Some(QualityAdjustment {
            config: new_config,
            reason: reasons.join("; "),
            quality_impact,
            performance_impact,
            confidence,
        })
    }

    /// Apply quality adjustment
    pub fn apply_adjustment(&mut self, adjustment: QualityAdjustment) {
        self.current_config = adjustment.config;
        self.last_adjustment = Instant::now();

        tracing::info!(
            "Applied quality adjustment: {} (quality impact: {:.2}, performance impact: {:.2}, confidence: {:.2})",
            adjustment.reason,
            adjustment.quality_impact,
            adjustment.performance_impact,
            adjustment.confidence
        );
    }

    /// Get current configuration
    pub fn current_config(&self) -> &AdaptiveConfig {
        &self.current_config
    }

    /// Get adaptation statistics
    pub fn get_adaptation_stats(&self) -> AdaptationStats {
        let mut total_adjustments = 0;
        let mut quality_improvements = 0;
        let mut performance_improvements = 0;
        let mut avg_quality = 0.0;
        let mut avg_performance = 0.0;

        for entry in &self.history {
            avg_quality += entry.quality.mos_estimate;
            avg_performance += entry
                .performance
                .performance_score(&self.performance_thresholds);
        }

        if !self.history.is_empty() {
            avg_quality /= self.history.len() as f32;
            avg_performance /= self.history.len() as f32;
        }

        // Count improvements by comparing consecutive entries
        for i in 1..self.history.len() {
            let prev = &self.history[i - 1];
            let curr = &self.history[i];

            if curr.quality.mos_estimate > prev.quality.mos_estimate {
                quality_improvements += 1;
            }

            if curr
                .performance
                .performance_score(&self.performance_thresholds)
                > prev
                    .performance
                    .performance_score(&self.performance_thresholds)
            {
                performance_improvements += 1;
            }

            total_adjustments += 1;
        }

        AdaptationStats {
            total_adjustments,
            quality_improvements,
            performance_improvements,
            avg_quality_score: avg_quality,
            avg_performance_score: avg_performance,
            history_size: self.history.len(),
            quality_target_achievement: if avg_quality > 0.0 {
                (avg_quality / self.quality_target.target_mos).min(1.0)
            } else {
                0.0
            },
        }
    }

    /// Evaluate audio quality using the built-in quality calculator
    pub fn evaluate_audio_quality(
        &mut self,
        reference: &AudioBuffer,
        degraded: &AudioBuffer,
    ) -> Result<QualityMetrics> {
        self.quality_calculator
            .calculate_metrics(reference, degraded)
    }

    /// Quick quality estimate for audio without reference
    pub fn estimate_audio_quality(&self, audio_data: &[f32], _sample_rate: u32) -> f32 {
        // Quick quality estimation based on simple metrics
        if audio_data.is_empty() {
            return 0.0;
        }

        // Calculate basic quality indicators
        let rms = (audio_data.iter().map(|x| x * x).sum::<f32>() / audio_data.len() as f32).sqrt();
        let peak = audio_data.iter().map(|x| x.abs()).fold(0.0, f32::max);

        // Dynamic range estimation
        let dynamic_range = if peak > 0.0 {
            20.0 * (peak / (rms + 1e-10)).log10()
        } else {
            0.0
        };

        // Clip detection
        let clip_count = audio_data.iter().filter(|&&x| x.abs() > 0.95).count();
        let clip_ratio = clip_count as f32 / audio_data.len() as f32;

        // Basic quality score combining multiple factors
        let level_score = (rms * 10.0).clamp(0.0, 1.0);
        let dynamic_score = (dynamic_range / 30.0).clamp(0.0, 1.0);
        let clip_penalty = (1.0 - clip_ratio * 5.0).clamp(0.0, 1.0);

        // Combine scores with appropriate weights
        let quality_score =
            (level_score * 0.3 + dynamic_score * 0.4 + clip_penalty * 0.3) * 4.0 + 1.0;
        quality_score.clamp(1.0, 5.0)
    }

    /// Calculate confidence in an adjustment decision
    fn calculate_adjustment_confidence(&self, _new_config: &AdaptiveConfig) -> f32 {
        // Confidence based on:
        // 1. Amount of historical data
        // 2. Consistency of recent trends
        // 3. Similarity to previous successful adjustments

        let history_confidence = (self.history.len() as f32 / self.max_history as f32).min(1.0);

        // Check trend consistency in recent history
        let recent_entries: Vec<_> = if self.history.len() >= 5 {
            self.history.iter().skip(self.history.len() - 5).collect()
        } else {
            self.history.iter().collect()
        };

        let trend_confidence = if recent_entries.len() >= 2 {
            let mut consistent_trends = 0;
            let mut total_comparisons = 0;

            for i in 1..recent_entries.len() {
                let prev_quality = recent_entries[i - 1].quality.mos_estimate;
                let curr_quality = recent_entries[i].quality.mos_estimate;

                // Check if quality trend is consistent with target
                if (curr_quality > prev_quality && curr_quality < self.quality_target.target_mos)
                    || (curr_quality < prev_quality
                        && curr_quality > self.quality_target.target_mos)
                {
                    consistent_trends += 1;
                }
                total_comparisons += 1;
            }

            if total_comparisons > 0 {
                consistent_trends as f32 / total_comparisons as f32
            } else {
                0.5
            }
        } else {
            0.5 // Neutral confidence with insufficient data
        };

        // Combine confidences
        (history_confidence * 0.4 + trend_confidence * 0.6).clamp(0.1, 0.95)
    }
}

/// Adaptation statistics
#[derive(Debug, Clone)]
pub struct AdaptationStats {
    /// Total number of adjustments made
    pub total_adjustments: usize,
    /// Number of quality improvements
    pub quality_improvements: usize,
    /// Number of performance improvements
    pub performance_improvements: usize,
    /// Average quality score achieved
    pub avg_quality_score: f32,
    /// Average performance score achieved
    pub avg_performance_score: f32,
    /// Size of history data
    pub history_size: usize,
    /// How well quality targets are being achieved (0.0-1.0)
    pub quality_target_achievement: f32,
}

impl AdaptationStats {
    /// Check if adaptation is working well
    pub fn is_adapting_well(&self) -> bool {
        self.quality_target_achievement > 0.8
            && self.avg_performance_score > 0.6
            && self.history_size >= 10
    }

    /// Get improvement rate
    pub fn improvement_rate(&self) -> f32 {
        if self.total_adjustments > 0 {
            (self.quality_improvements + self.performance_improvements) as f32
                / (self.total_adjustments * 2) as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::QualityMetrics;
    use crate::performance::{PerformanceMetrics, PerformanceThresholds};

    fn create_test_quality_metrics(mos: f32) -> QualityMetrics {
        QualityMetrics {
            pesq: None,
            stoi: None,
            si_sdr: None,
            mos_prediction: Some(mos),
            snr: 20.0,
            thd_n: 1.0,
            lsd: 0.5,
            mcd: None,
            psnr: 30.0,
            spectral_convergence: 0.1,
            mos_estimate: mos,
        }
    }

    fn create_test_performance_metrics(latency: f32, cpu: f32, rtf: f32) -> PerformanceMetrics {
        let quality = create_test_quality_metrics(3.5);
        PerformanceMetrics::new(latency, cpu, 100.0, rtf, quality, 0, 0.9)
    }

    #[test]
    fn test_adaptive_controller_creation() {
        let config = AdaptiveConfig::default();
        let target = QualityTarget::default();
        let thresholds = PerformanceThresholds::default();

        let _controller = AdaptiveQualityController::new(config, target, thresholds);
    }

    #[test]
    fn test_should_adjust_logic() {
        let config = AdaptiveConfig::default();
        let target = QualityTarget::default();
        let thresholds = PerformanceThresholds::default();

        let mut controller = AdaptiveQualityController::new(config, target, thresholds);

        // No history - should not adjust
        assert!(!controller.should_adjust());

        // Add low quality metrics - should adjust
        let quality = create_test_quality_metrics(2.0); // Below target of 4.0
        let performance = create_test_performance_metrics(50.0, 30.0, 0.3);

        controller.update_metrics(quality, performance, None);

        // Should want to adjust due to low quality
        assert!(controller.should_adjust());
    }

    #[test]
    fn test_quality_adjustment_recommendation() {
        let config = AdaptiveConfig::default();
        let target = QualityTarget::default();
        let thresholds = PerformanceThresholds::default();

        let mut controller = AdaptiveQualityController::new(config, target, thresholds);

        // Add low quality metrics
        let quality = create_test_quality_metrics(2.5);
        let performance = create_test_performance_metrics(50.0, 30.0, 0.3);

        controller.update_metrics(quality, performance, None);

        let adjustment = controller.recommend_adjustment();
        assert!(adjustment.is_some());

        let adj = adjustment.unwrap();
        assert!(adj.quality_impact > 0.0); // Should improve quality
        assert!(adj.confidence > 0.0);
    }

    #[test]
    fn test_precision_mode_multipliers() {
        assert_eq!(PrecisionMode::Low.speed_multiplier(), 2.0);
        assert_eq!(PrecisionMode::High.speed_multiplier(), 1.0);
        assert_eq!(PrecisionMode::Ultra.quality_multiplier(), 1.1);
        assert_eq!(PrecisionMode::Low.quality_multiplier(), 0.8);
    }

    #[test]
    fn test_adaptation_stats() {
        let config = AdaptiveConfig::default();
        let target = QualityTarget::default();
        let thresholds = PerformanceThresholds::default();

        let mut controller = AdaptiveQualityController::new(config, target, thresholds);

        // Add some metrics
        for i in 0..5 {
            let quality = create_test_quality_metrics(3.0 + i as f32 * 0.2);
            let performance = create_test_performance_metrics(50.0, 30.0, 0.3);
            controller.update_metrics(quality, performance, None);
        }

        let stats = controller.get_adaptation_stats();
        assert_eq!(stats.history_size, 5);
        assert!(stats.avg_quality_score > 3.0);
    }
}
