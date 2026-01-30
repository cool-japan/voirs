//! Adaptive quality control system for VoiRS SDK.
//!
//! This module provides intelligent, runtime-adaptive quality control that automatically
//! adjusts synthesis parameters based on system load, performance metrics, and constraints.
//!
//! # Features
//!
//! - **Dynamic Quality Adjustment**: Automatically scales quality based on system load
//! - **Real-time Monitoring**: Continuous monitoring of system resources and performance
//! - **Predictive Optimization**: ML-based prediction of optimal quality settings
//! - **SLA Compliance**: Ensures synthesis meets latency and quality targets
//! - **Graceful Degradation**: Smooth quality transitions under load
//!
//! # Example
//!
//! ```no_run
//! use voirs_sdk::adaptive::{AdaptiveController, AdaptiveConfig, QualityTarget};
//!
//! #[tokio::main]
//! async fn main() -> voirs_sdk::Result<()> {
//!     let config = AdaptiveConfig::default()
//!         .with_target_latency(100) // 100ms max latency
//!         .with_min_quality(QualityTarget::Medium)
//!         .with_adaptation_speed(0.7);
//!
//!     let controller = AdaptiveController::new(config);
//!
//!     // Controller automatically adjusts quality based on system load
//!     let quality_level = controller.get_recommended_quality().await?;
//!
//!     Ok(())
//! }
//! ```

use super::predictor::{PredictionInput, QualityPredictor, TextComplexityAnalyzer, TrainingSample};
use crate::types::QualityLevel;
use crate::{Result, VoirsError};
use chrono::Timelike;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Adaptive quality control configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Target maximum latency in milliseconds
    pub target_latency_ms: u64,

    /// Minimum acceptable quality level
    pub min_quality: QualityTarget,

    /// Maximum quality level to use
    pub max_quality: QualityTarget,

    /// Adaptation speed (0.0-1.0): how quickly to adjust quality
    /// - 0.0: No adaptation (fixed quality)
    /// - 0.5: Moderate adaptation
    /// - 1.0: Aggressive adaptation
    pub adaptation_speed: f32,

    /// CPU usage threshold for quality reduction (0.0-1.0)
    pub cpu_threshold: f32,

    /// Memory usage threshold for quality reduction (0.0-1.0)
    pub memory_threshold: f32,

    /// RTF threshold for quality adjustment
    pub rtf_threshold: f32,

    /// Enable predictive optimization using historical data
    pub enable_prediction: bool,

    /// Sliding window size for performance metrics
    pub metrics_window_size: usize,

    /// Minimum time between quality changes (stability)
    pub min_change_interval: Duration,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            target_latency_ms: 100,
            min_quality: QualityTarget::Medium,
            max_quality: QualityTarget::High,
            adaptation_speed: 0.5,
            cpu_threshold: 0.75,
            memory_threshold: 0.80,
            rtf_threshold: 0.8,
            enable_prediction: true,
            metrics_window_size: 100,
            min_change_interval: Duration::from_secs(2),
        }
    }
}

impl AdaptiveConfig {
    /// Create configuration with specific target latency.
    pub fn with_target_latency(mut self, latency_ms: u64) -> Self {
        self.target_latency_ms = latency_ms;
        self
    }

    /// Set minimum quality level.
    pub fn with_min_quality(mut self, quality: QualityTarget) -> Self {
        self.min_quality = quality;
        self
    }

    /// Set maximum quality level.
    pub fn with_max_quality(mut self, quality: QualityTarget) -> Self {
        self.max_quality = quality;
        self
    }

    /// Set adaptation speed.
    pub fn with_adaptation_speed(mut self, speed: f32) -> Self {
        self.adaptation_speed = speed.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable predictive optimization.
    pub fn with_prediction(mut self, enable: bool) -> Self {
        self.enable_prediction = enable;
        self
    }
}

/// Quality target for adaptive control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QualityTarget {
    /// Lowest quality, fastest processing
    Low,

    /// Balanced quality and performance
    Medium,

    /// High quality with acceptable performance
    High,

    /// Maximum quality, slowest processing
    VeryHigh,

    /// Custom quality level (0-100)
    Custom(u8),
}

impl QualityTarget {
    /// Convert to VoiRS QualityLevel.
    pub fn to_quality_level(&self) -> QualityLevel {
        match self {
            QualityTarget::Low => QualityLevel::Low,
            QualityTarget::Medium => QualityLevel::Medium,
            QualityTarget::High => QualityLevel::High,
            QualityTarget::VeryHigh => QualityLevel::High,
            QualityTarget::Custom(level) => {
                if *level < 30 {
                    QualityLevel::Low
                } else if *level < 70 {
                    QualityLevel::Medium
                } else {
                    QualityLevel::High
                }
            }
        }
    }

    /// Get numeric score for comparison (0-100).
    pub fn score(&self) -> u8 {
        match self {
            QualityTarget::Low => 25,
            QualityTarget::Medium => 50,
            QualityTarget::High => 75,
            QualityTarget::VeryHigh => 100,
            QualityTarget::Custom(level) => *level,
        }
    }
}

/// System resource metrics for adaptive control.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// Current CPU usage (0.0-1.0)
    pub cpu_usage: f32,

    /// Current memory usage (0.0-1.0)
    pub memory_usage: f32,

    /// Current real-time factor
    pub current_rtf: f32,

    /// Recent synthesis latency
    pub recent_latency_ms: u64,

    /// Timestamp of measurement (not serialized)
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
}

/// Performance history entry for predictive optimization.
#[derive(Debug, Clone)]
struct PerformanceEntry {
    quality_level: QualityTarget,
    system_metrics: SystemMetrics,
    synthesis_time_ms: u64,
    success: bool,
}

/// Internal state for adaptive controller.
#[derive(Debug)]
struct AdaptiveState {
    config: AdaptiveConfig,
    current_quality: QualityTarget,
    last_adjustment: Instant,
    performance_history: Vec<PerformanceEntry>,
    current_metrics: Option<SystemMetrics>,
    adjustment_count: u64,
    /// ML-based quality predictor (when prediction enabled)
    predictor: Option<QualityPredictor>,
}

/// Adaptive quality controller.
///
/// Automatically adjusts synthesis quality based on system load and performance.
#[derive(Debug, Clone)]
pub struct AdaptiveController {
    state: Arc<RwLock<AdaptiveState>>,
}

impl AdaptiveController {
    /// Create a new adaptive controller with the given configuration.
    pub fn new(config: AdaptiveConfig) -> Self {
        let initial_quality = config.max_quality;
        let predictor = if config.enable_prediction {
            Some(QualityPredictor::new().with_history_size(config.metrics_window_size, 10))
        } else {
            None
        };

        Self {
            state: Arc::new(RwLock::new(AdaptiveState {
                config,
                current_quality: initial_quality,
                last_adjustment: Instant::now(),
                performance_history: Vec::new(),
                current_metrics: None,
                adjustment_count: 0,
                predictor,
            })),
        }
    }

    /// Get the current recommended quality level.
    pub async fn get_recommended_quality(&self) -> Result<QualityTarget> {
        let state = self.state.read().await;
        Ok(state.current_quality)
    }

    /// Update system metrics and potentially adjust quality.
    pub async fn update_metrics(&self, metrics: SystemMetrics) -> Result<Option<QualityTarget>> {
        let mut state = self.state.write().await;

        // Check if enough time has passed since last adjustment
        let elapsed = state.last_adjustment.elapsed();
        if elapsed < state.config.min_change_interval {
            state.current_metrics = Some(metrics);
            return Ok(None);
        }

        // Calculate stress score (0.0-1.0)
        let stress_score = Self::calculate_stress_score(&metrics, &state.config);

        // Determine if quality adjustment is needed
        let new_quality = if stress_score > 0.7 {
            // System under stress, reduce quality
            Self::reduce_quality(state.current_quality, stress_score, &state.config)
        } else if stress_score < 0.3 {
            // System has headroom, can increase quality
            Self::increase_quality(state.current_quality, &state.config)
        } else {
            // System in acceptable range, maintain quality
            state.current_quality
        };

        // Apply change if different
        if new_quality != state.current_quality {
            tracing::info!(
                "Adaptive quality change: {:?} -> {:?} (stress: {:.2})",
                state.current_quality,
                new_quality,
                stress_score
            );

            state.current_quality = new_quality;
            state.last_adjustment = Instant::now();
            state.adjustment_count += 1;
            state.current_metrics = Some(metrics);

            Ok(Some(new_quality))
        } else {
            state.current_metrics = Some(metrics);
            Ok(None)
        }
    }

    /// Record performance data for predictive optimization.
    pub async fn record_performance(
        &self,
        quality: QualityTarget,
        synthesis_time_ms: u64,
        success: bool,
    ) -> Result<()> {
        let mut state = self.state.write().await;

        if let Some(metrics) = state.current_metrics.clone() {
            let entry = PerformanceEntry {
                quality_level: quality,
                system_metrics: metrics.clone(),
                synthesis_time_ms,
                success,
            };

            state.performance_history.push(entry);

            // Train the ML predictor if enabled
            if let Some(ref mut predictor) = state.predictor {
                let training_sample = TrainingSample {
                    input: PredictionInput {
                        cpu_usage: metrics.cpu_usage,
                        memory_usage: metrics.memory_usage,
                        text_complexity: 0.5, // Default, can be updated with actual complexity
                        time_of_day: chrono::Local::now().hour() as u8,
                        recent_rtf: metrics.current_rtf,
                    },
                    quality,
                    synthesis_time_ms,
                    success,
                    measured_rtf: metrics.current_rtf,
                };

                predictor.train(training_sample).await?;
            }

            // Limit history size
            let max_size = state.config.metrics_window_size;
            let current_len = state.performance_history.len();
            if current_len > max_size {
                state.performance_history.drain(0..current_len - max_size);
            }
        }

        Ok(())
    }

    /// Record performance with text complexity for better ML prediction.
    pub async fn record_performance_with_text(
        &self,
        quality: QualityTarget,
        synthesis_time_ms: u64,
        success: bool,
        text: &str,
    ) -> Result<()> {
        let mut state = self.state.write().await;

        if let Some(metrics) = state.current_metrics.clone() {
            let entry = PerformanceEntry {
                quality_level: quality,
                system_metrics: metrics.clone(),
                synthesis_time_ms,
                success,
            };

            state.performance_history.push(entry);

            // Train the ML predictor with text complexity
            if let Some(ref mut predictor) = state.predictor {
                let text_complexity = TextComplexityAnalyzer::analyze(text);
                let training_sample = TrainingSample {
                    input: PredictionInput {
                        cpu_usage: metrics.cpu_usage,
                        memory_usage: metrics.memory_usage,
                        text_complexity,
                        time_of_day: chrono::Local::now().hour() as u8,
                        recent_rtf: metrics.current_rtf,
                    },
                    quality,
                    synthesis_time_ms,
                    success,
                    measured_rtf: metrics.current_rtf,
                };

                predictor.train(training_sample).await?;
            }

            // Limit history size
            let max_size = state.config.metrics_window_size;
            let current_len = state.performance_history.len();
            if current_len > max_size {
                state.performance_history.drain(0..current_len - max_size);
            }
        }

        Ok(())
    }

    /// Get ML-predicted quality for given text and current system state.
    pub async fn get_predicted_quality(&self, text: &str) -> Result<Option<QualityTarget>> {
        let state = self.state.read().await;

        if let (Some(ref predictor), Some(ref metrics)) = (&state.predictor, &state.current_metrics)
        {
            let text_complexity = TextComplexityAnalyzer::analyze(text);
            let input = PredictionInput::from_metrics(metrics, text_complexity);

            let prediction = predictor.predict(&input).await?;

            // Only use prediction if confidence is high enough
            if prediction.confidence > 0.5 {
                Ok(Some(prediction.quality))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Get statistics from the ML predictor.
    pub async fn get_predictor_stats(&self) -> Result<Option<super::predictor::PredictorStats>> {
        let state = self.state.read().await;

        if let Some(ref predictor) = state.predictor {
            Ok(Some(predictor.get_stats()))
        } else {
            Ok(None)
        }
    }

    /// Get statistics about quality adaptations.
    pub async fn get_adaptation_stats(&self) -> Result<AdaptationStats> {
        let state = self.state.read().await;

        let history_entries = state.performance_history.len();
        let successful_syntheses = state
            .performance_history
            .iter()
            .filter(|e| e.success)
            .count();

        let avg_synthesis_time = if history_entries > 0 {
            state
                .performance_history
                .iter()
                .map(|e| e.synthesis_time_ms as f64)
                .sum::<f64>()
                / history_entries as f64
        } else {
            0.0
        };

        Ok(AdaptationStats {
            total_adjustments: state.adjustment_count,
            current_quality: state.current_quality,
            history_entries,
            success_rate: if history_entries > 0 {
                successful_syntheses as f64 / history_entries as f64
            } else {
                0.0
            },
            avg_synthesis_time_ms: avg_synthesis_time,
        })
    }

    /// Reset the adaptive controller to initial state.
    pub async fn reset(&self) -> Result<()> {
        let mut state = self.state.write().await;
        state.current_quality = state.config.max_quality;
        state.last_adjustment = Instant::now();
        state.performance_history.clear();
        state.current_metrics = None;
        state.adjustment_count = 0;

        // Reset ML predictor if enabled
        if let Some(ref mut predictor) = state.predictor {
            predictor.reset();
        }

        Ok(())
    }

    // Internal helper methods

    fn calculate_stress_score(metrics: &SystemMetrics, config: &AdaptiveConfig) -> f32 {
        // Weighted combination of stress factors
        let cpu_stress = (metrics.cpu_usage / config.cpu_threshold).min(1.0);
        let memory_stress = (metrics.memory_usage / config.memory_threshold).min(1.0);
        let rtf_stress = (metrics.current_rtf / config.rtf_threshold).min(1.0);

        // Weights: CPU(40%), Memory(30%), RTF(30%)
        cpu_stress * 0.4 + memory_stress * 0.3 + rtf_stress * 0.3
    }

    fn reduce_quality(
        current: QualityTarget,
        stress: f32,
        config: &AdaptiveConfig,
    ) -> QualityTarget {
        let current_score = current.score();
        let min_score = config.min_quality.score();

        // Calculate reduction based on stress and adaptation speed
        let reduction = (stress * config.adaptation_speed * 25.0) as u8;
        let new_score = current_score.saturating_sub(reduction).max(min_score);

        QualityTarget::Custom(new_score)
    }

    fn increase_quality(current: QualityTarget, config: &AdaptiveConfig) -> QualityTarget {
        let current_score = current.score();
        let max_score = config.max_quality.score();

        // Gradual increase (smaller steps than decrease for stability)
        let increase = (config.adaptation_speed * 10.0) as u8;
        let new_score = current_score.saturating_add(increase).min(max_score);

        QualityTarget::Custom(new_score)
    }
}

/// Statistics about quality adaptations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationStats {
    /// Total number of quality adjustments made
    pub total_adjustments: u64,

    /// Current quality level
    pub current_quality: QualityTarget,

    /// Number of performance history entries
    pub history_entries: usize,

    /// Success rate of recent syntheses
    pub success_rate: f64,

    /// Average synthesis time in recent history
    pub avg_synthesis_time_ms: f64,
}

/// Helper function to get current system metrics.
///
/// This is a simplified implementation. In production, you would use
/// system-specific APIs for more accurate measurements.
pub async fn get_system_metrics() -> Result<SystemMetrics> {
    Ok(SystemMetrics {
        cpu_usage: estimate_cpu_usage(),
        memory_usage: estimate_memory_usage(),
        current_rtf: 0.5, // Would be measured from actual synthesis
        recent_latency_ms: 50,
        timestamp: Instant::now(),
    })
}

fn estimate_cpu_usage() -> f32 {
    // Simplified estimation - in production would use sysinfo or similar
    num_cpus::get() as f32 * 0.5
}

fn estimate_memory_usage() -> f32 {
    // Simplified estimation - in production would use sysinfo or similar
    0.4 // 40% memory usage
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_target_ordering() {
        assert!(QualityTarget::Low < QualityTarget::Medium);
        assert!(QualityTarget::Medium < QualityTarget::High);
        assert!(QualityTarget::High < QualityTarget::VeryHigh);
    }

    #[test]
    fn test_quality_target_scores() {
        assert_eq!(QualityTarget::Low.score(), 25);
        assert_eq!(QualityTarget::Medium.score(), 50);
        assert_eq!(QualityTarget::High.score(), 75);
        assert_eq!(QualityTarget::VeryHigh.score(), 100);
        assert_eq!(QualityTarget::Custom(80).score(), 80);
    }

    #[test]
    fn test_adaptive_config_builder() {
        let config = AdaptiveConfig::default()
            .with_target_latency(150)
            .with_min_quality(QualityTarget::Low)
            .with_adaptation_speed(0.8);

        assert_eq!(config.target_latency_ms, 150);
        assert_eq!(config.min_quality, QualityTarget::Low);
        assert_eq!(config.adaptation_speed, 0.8);
    }

    #[tokio::test]
    async fn test_adaptive_controller_creation() {
        let config = AdaptiveConfig::default();
        let controller = AdaptiveController::new(config.clone());

        let quality = controller.get_recommended_quality().await.unwrap();
        assert_eq!(quality, config.max_quality);
    }

    #[tokio::test]
    async fn test_quality_reduction_under_stress() {
        let config = AdaptiveConfig::default()
            .with_min_quality(QualityTarget::Low)
            .with_adaptation_speed(1.0);

        let controller = AdaptiveController::new(config);

        // Simulate high stress
        let metrics = SystemMetrics {
            cpu_usage: 0.95,
            memory_usage: 0.90,
            current_rtf: 1.2,
            recent_latency_ms: 200,
            timestamp: Instant::now(),
        };

        // Wait for min_change_interval
        tokio::time::sleep(Duration::from_secs(3)).await;

        let result = controller.update_metrics(metrics).await.unwrap();
        assert!(result.is_some());

        if let Some(new_quality) = result {
            // Quality should have decreased
            assert!(new_quality.score() < QualityTarget::High.score());
        }
    }

    #[tokio::test]
    async fn test_performance_recording() {
        let controller = AdaptiveController::new(AdaptiveConfig::default());

        // Record some successful syntheses
        for _ in 0..5 {
            controller
                .record_performance(QualityTarget::High, 100, true)
                .await
                .unwrap();
        }

        let stats = controller.get_adaptation_stats().await.unwrap();
        assert_eq!(stats.history_entries, 0); // No metrics provided yet
    }

    #[tokio::test]
    async fn test_controller_reset() {
        let config = AdaptiveConfig::default();
        let controller = AdaptiveController::new(config.clone());

        controller.reset().await.unwrap();

        let quality = controller.get_recommended_quality().await.unwrap();
        assert_eq!(quality, config.max_quality);
    }
}
