//! Latency optimization for streaming synthesis
//!
//! This module provides adaptive algorithms for minimizing latency in real-time
//! synthesis while maintaining quality through predictive processing and
//! dynamic parameter adjustment.

use crate::{Result, SynthesisConfig};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Instant;

/// Latency optimization strategy
#[derive(Debug, Clone, Copy)]
pub enum LatencyStrategy {
    /// Minimize latency at any cost
    UltraLow,
    /// Balance latency and quality
    Balanced,
    /// Prioritize quality over latency
    HighQuality,
    /// Adaptive based on input characteristics
    Adaptive,
}

impl Default for LatencyStrategy {
    fn default() -> Self {
        Self::Balanced
    }
}

/// Latency optimizer configuration
#[derive(Debug, Clone)]
pub struct LatencyOptimizerConfig {
    /// Optimization strategy
    pub strategy: LatencyStrategy,
    /// Target latency in milliseconds
    pub target_latency_ms: f32,
    /// Maximum acceptable latency
    pub max_latency_ms: f32,
    /// Quality tolerance (how much quality reduction is acceptable)
    pub quality_tolerance: f32,
    /// Look-ahead window size for prediction
    pub lookahead_frames: usize,
    /// Enable predictive synthesis
    pub enable_prediction: bool,
    /// Adaptive chunk sizing
    pub adaptive_chunks: bool,
    /// Performance history window
    pub history_window: usize,
}

impl Default for LatencyOptimizerConfig {
    fn default() -> Self {
        Self {
            strategy: LatencyStrategy::Balanced,
            target_latency_ms: 30.0,
            max_latency_ms: 50.0,
            quality_tolerance: 0.15, // 15% quality reduction acceptable
            lookahead_frames: 64,
            enable_prediction: true,
            adaptive_chunks: true,
            history_window: 50,
        }
    }
}

/// Performance measurement for latency optimization
#[derive(Debug, Clone)]
pub struct PerformanceMeasurement {
    /// Timestamp of measurement
    pub timestamp: Instant,
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
    /// Input complexity (estimated)
    pub input_complexity: f32,
    /// Quality score achieved
    pub quality_score: f32,
    /// Chunk size used
    pub chunk_size: usize,
    /// Configuration used
    pub config_hash: u64,
}

/// Predictive performance model
#[derive(Debug)]
pub struct PerformancePredictor {
    /// Historical measurements
    measurements: VecDeque<PerformanceMeasurement>,
    /// Maximum history size
    max_history: usize,
    /// Learned weights for prediction
    complexity_weight: f32,
    chunk_weight: f32,
    config_weight: f32,
}

impl PerformancePredictor {
    /// Create new performance predictor
    pub fn new(max_history: usize) -> Self {
        Self {
            measurements: VecDeque::with_capacity(max_history),
            max_history,
            complexity_weight: 0.5,
            chunk_weight: 0.3,
            config_weight: 0.2,
        }
    }

    /// Add performance measurement
    pub fn add_measurement(&mut self, measurement: PerformanceMeasurement) {
        if self.measurements.len() >= self.max_history {
            self.measurements.pop_front();
        }

        self.measurements.push_back(measurement);
        self.update_weights();
    }

    /// Predict processing time for given parameters
    pub fn predict_processing_time(
        &self,
        input_complexity: f32,
        chunk_size: usize,
        config_hash: u64,
    ) -> Option<f32> {
        if self.measurements.is_empty() {
            return None;
        }

        // Simple linear prediction based on recent measurements
        let recent_measurements: Vec<_> = self.measurements.iter().rev().take(10).collect();

        if recent_measurements.is_empty() {
            return None;
        }

        // Find similar measurements
        let mut similar_times = Vec::new();
        for measurement in &recent_measurements {
            let complexity_diff = (measurement.input_complexity - input_complexity).abs();
            let chunk_diff =
                (measurement.chunk_size as f32 - chunk_size as f32).abs() / chunk_size as f32;
            let config_match = if measurement.config_hash == config_hash {
                1.0
            } else {
                0.0
            };

            let similarity = 1.0
                - (complexity_diff * self.complexity_weight
                    + chunk_diff * self.chunk_weight
                    + (1.0 - config_match) * self.config_weight);

            if similarity > 0.5 {
                similar_times.push(measurement.processing_time_ms);
            }
        }

        if similar_times.is_empty() {
            // Fallback to average of recent measurements
            let avg = recent_measurements
                .iter()
                .map(|m| m.processing_time_ms)
                .sum::<f32>()
                / recent_measurements.len() as f32;
            Some(avg)
        } else {
            // Weighted average of similar measurements
            let avg = similar_times.iter().sum::<f32>() / similar_times.len() as f32;
            Some(avg)
        }
    }

    /// Predict quality score for given configuration
    pub fn predict_quality(&self, config_hash: u64) -> Option<f32> {
        let matching_measurements: Vec<_> = self
            .measurements
            .iter()
            .filter(|m| m.config_hash == config_hash)
            .collect();

        if matching_measurements.is_empty() {
            return None;
        }

        let avg_quality = matching_measurements
            .iter()
            .map(|m| m.quality_score)
            .sum::<f32>()
            / matching_measurements.len() as f32;

        Some(avg_quality)
    }

    fn update_weights(&mut self) {
        // Simple online learning to adjust prediction weights
        // This could be replaced with more sophisticated ML algorithms
        if self.measurements.len() < 10 {
            return;
        }

        // Analyze prediction accuracy and adjust weights
        let start_idx = self.measurements.len().saturating_sub(5);
        let recent: Vec<_> = self.measurements.iter().skip(start_idx).cloned().collect();
        let mut complexity_errors = Vec::new();
        let mut chunk_errors = Vec::new();

        for (i, measurement) in recent.iter().enumerate() {
            if i == 0 {
                continue;
            }

            let prev = &recent[i - 1];
            let predicted_complex =
                prev.processing_time_ms * (measurement.input_complexity / prev.input_complexity);
            let predicted_chunk =
                prev.processing_time_ms * (measurement.chunk_size as f32 / prev.chunk_size as f32);

            complexity_errors.push((predicted_complex - measurement.processing_time_ms).abs());
            chunk_errors.push((predicted_chunk - measurement.processing_time_ms).abs());
        }

        if !complexity_errors.is_empty() && !chunk_errors.is_empty() {
            let avg_complexity_error =
                complexity_errors.iter().sum::<f32>() / complexity_errors.len() as f32;
            let avg_chunk_error = chunk_errors.iter().sum::<f32>() / chunk_errors.len() as f32;

            // Adjust weights based on error rates (lower error = higher weight)
            let total_error = avg_complexity_error + avg_chunk_error;
            if total_error > 0.0 {
                self.complexity_weight =
                    0.8 * self.complexity_weight + 0.2 * (1.0 - avg_complexity_error / total_error);
                self.chunk_weight =
                    0.8 * self.chunk_weight + 0.2 * (1.0 - avg_chunk_error / total_error);

                // Normalize weights
                let weight_sum = self.complexity_weight + self.chunk_weight + self.config_weight;
                self.complexity_weight /= weight_sum;
                self.chunk_weight /= weight_sum;
                self.config_weight /= weight_sum;
            }
        }
    }
}

/// Latency optimizer for streaming synthesis
pub struct LatencyOptimizer {
    /// Configuration
    config: LatencyOptimizerConfig,
    /// Performance predictor
    predictor: PerformancePredictor,
    /// Current optimal chunk size
    optimal_chunk_size: usize,
    /// Quality-latency trade-off curve
    quality_curve: Vec<(f32, f32)>, // (quality_reduction, latency_reduction)
    /// Recent latency measurements
    recent_latencies: VecDeque<f32>,
}

impl LatencyOptimizer {
    /// Create new latency optimizer
    pub fn new(config: LatencyOptimizerConfig) -> Self {
        let predictor = PerformancePredictor::new(config.history_window);

        Self {
            config,
            predictor,
            optimal_chunk_size: 256, // Default chunk size
            quality_curve: Self::default_quality_curve(),
            recent_latencies: VecDeque::with_capacity(20),
        }
    }

    /// Optimize synthesis configuration for latency
    pub fn optimize_config(
        &mut self,
        base_config: &SynthesisConfig,
        input_complexity: f32,
        target_latency_ms: Option<f32>,
    ) -> Result<SynthesisConfig> {
        let target = target_latency_ms.unwrap_or(self.config.target_latency_ms);

        let mut optimized_config = base_config.clone();

        match self.config.strategy {
            LatencyStrategy::UltraLow => {
                self.apply_ultra_low_latency(&mut optimized_config)?;
            }
            LatencyStrategy::Balanced => {
                self.apply_balanced_optimization(&mut optimized_config, target, input_complexity)?;
            }
            LatencyStrategy::HighQuality => {
                self.apply_quality_first(&mut optimized_config, target)?;
            }
            LatencyStrategy::Adaptive => {
                self.apply_adaptive_optimization(&mut optimized_config, target, input_complexity)?;
            }
        }

        Ok(optimized_config)
    }

    /// Calculate optimal chunk size for current conditions
    pub fn calculate_optimal_chunk_size(
        &self,
        input_complexity: f32,
        target_latency_ms: f32,
    ) -> usize {
        if !self.config.adaptive_chunks {
            return self.optimal_chunk_size;
        }

        // Start with base chunk size and adjust based on predicted performance
        let base_chunk_size = self.optimal_chunk_size;
        let mut test_sizes = vec![base_chunk_size / 2, base_chunk_size, base_chunk_size * 2];

        // Filter out unreasonable sizes
        test_sizes.retain(|&size| (64..=1024).contains(&size));

        let mut best_size = base_chunk_size;
        let mut best_score = f32::NEG_INFINITY;

        for &chunk_size in &test_sizes {
            if let Some(predicted_time) = self.predictor.predict_processing_time(
                input_complexity,
                chunk_size,
                0, // Config hash would be calculated from actual config
            ) {
                // Score based on how close we get to target latency
                let latency_score =
                    1.0 - (predicted_time - target_latency_ms).abs() / target_latency_ms;

                // Prefer smaller chunks for lower latency, but with diminishing returns
                let size_score = 1.0 / (1.0 + (chunk_size as f32 / 256.0).ln());

                let total_score = 0.7 * latency_score + 0.3 * size_score;

                if total_score > best_score {
                    best_score = total_score;
                    best_size = chunk_size;
                }
            }
        }

        best_size
    }

    /// Add performance measurement for learning
    pub fn add_measurement(
        &mut self,
        processing_time_ms: f32,
        input_complexity: f32,
        quality_score: f32,
        chunk_size: usize,
        config: &SynthesisConfig,
    ) {
        let measurement = PerformanceMeasurement {
            timestamp: Instant::now(),
            processing_time_ms,
            input_complexity,
            quality_score,
            chunk_size,
            config_hash: self.calculate_config_hash(config),
        };

        self.predictor.add_measurement(measurement);

        // Update recent latency tracking
        if self.recent_latencies.len() >= 20 {
            self.recent_latencies.pop_front();
        }
        self.recent_latencies.push_back(processing_time_ms);

        // Update optimal chunk size based on recent performance
        if self.config.adaptive_chunks {
            self.update_optimal_chunk_size();
        }
    }

    /// Check if current performance meets latency requirements
    pub fn is_meeting_latency_target(&self) -> bool {
        if self.recent_latencies.is_empty() {
            return true;
        }

        let avg_latency =
            self.recent_latencies.iter().sum::<f32>() / self.recent_latencies.len() as f32;
        avg_latency <= self.config.max_latency_ms
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> LatencyStats {
        let avg_latency = if self.recent_latencies.is_empty() {
            0.0
        } else {
            self.recent_latencies.iter().sum::<f32>() / self.recent_latencies.len() as f32
        };

        let latency_variance = if self.recent_latencies.len() > 1 {
            let variance = self
                .recent_latencies
                .iter()
                .map(|&x| (x - avg_latency).powi(2))
                .sum::<f32>()
                / (self.recent_latencies.len() - 1) as f32;
            variance.sqrt()
        } else {
            0.0
        };

        LatencyStats {
            avg_latency_ms: avg_latency,
            latency_variance,
            target_latency_ms: self.config.target_latency_ms,
            optimal_chunk_size: self.optimal_chunk_size,
            measurements_count: self.predictor.measurements.len(),
            is_meeting_target: self.is_meeting_latency_target(),
        }
    }

    // Private helper methods

    fn apply_ultra_low_latency(&self, config: &mut SynthesisConfig) -> Result<()> {
        // Minimize processing time for maximum speed
        config.speed = config.speed.max(1.5); // Increase speed for lower latency
        Ok(())
    }

    fn apply_balanced_optimization(
        &self,
        config: &mut SynthesisConfig,
        target_latency_ms: f32,
        input_complexity: f32,
    ) -> Result<()> {
        // Find optimal quality-latency trade-off
        let quality_reduction =
            self.calculate_required_quality_reduction(target_latency_ms, input_complexity);

        if quality_reduction > 0.0 {
            // Increase speed to reduce latency
            let speed_increase = 1.0 + quality_reduction * 0.5;
            config.speed *= speed_increase;
        }

        Ok(())
    }

    fn apply_quality_first(
        &self,
        config: &mut SynthesisConfig,
        target_latency_ms: f32,
    ) -> Result<()> {
        // Only make minimal adjustments to meet hard latency constraints
        let current_avg = if self.recent_latencies.is_empty() {
            target_latency_ms
        } else {
            self.recent_latencies.iter().sum::<f32>() / self.recent_latencies.len() as f32
        };

        if current_avg > self.config.max_latency_ms {
            // Only slightly increase speed if we're exceeding max latency
            let reduction = ((current_avg - self.config.max_latency_ms) / current_avg).min(0.1);
            config.speed *= 1.0 + reduction * 0.2;
        }

        Ok(())
    }

    fn apply_adaptive_optimization(
        &self,
        config: &mut SynthesisConfig,
        target_latency_ms: f32,
        input_complexity: f32,
    ) -> Result<()> {
        // Choose strategy based on current performance and input characteristics
        let avg_latency = if self.recent_latencies.is_empty() {
            target_latency_ms
        } else {
            self.recent_latencies.iter().sum::<f32>() / self.recent_latencies.len() as f32
        };

        let latency_pressure = avg_latency / target_latency_ms;
        let complexity_factor = input_complexity; // Normalized 0-1

        if latency_pressure > 1.3 || complexity_factor > 0.8 {
            // High latency pressure or complex input - prioritize speed
            self.apply_ultra_low_latency(config)?;
        } else if latency_pressure > 1.1 || complexity_factor > 0.6 {
            // Moderate pressure - use balanced approach
            self.apply_balanced_optimization(config, target_latency_ms, input_complexity)?;
        } else {
            // Low pressure - maintain quality
            self.apply_quality_first(config, target_latency_ms)?;
        }

        Ok(())
    }

    fn calculate_required_quality_reduction(
        &self,
        target_latency_ms: f32,
        input_complexity: f32,
    ) -> f32 {
        // Estimate how much quality reduction is needed to meet target latency
        let current_predicted = self
            .predictor
            .predict_processing_time(input_complexity, self.optimal_chunk_size, 0)
            .unwrap_or(target_latency_ms);

        if current_predicted <= target_latency_ms {
            return 0.0;
        }

        // Simple linear approximation - could be made more sophisticated
        let overshoot = (current_predicted - target_latency_ms) / target_latency_ms;

        // Look up quality-latency curve to find required reduction
        for &(quality_reduction, latency_reduction) in &self.quality_curve {
            if latency_reduction >= overshoot {
                return quality_reduction;
            }
        }

        // If we can't meet target even with maximum quality reduction
        self.config.quality_tolerance
    }

    fn update_optimal_chunk_size(&mut self) {
        if self.recent_latencies.len() < 5 {
            return;
        }

        let avg_latency =
            self.recent_latencies.iter().sum::<f32>() / self.recent_latencies.len() as f32;

        // Simple heuristic: if latency is too high, reduce chunk size
        if avg_latency > self.config.target_latency_ms * 1.2 {
            self.optimal_chunk_size = (self.optimal_chunk_size * 3 / 4).max(64);
        } else if avg_latency < self.config.target_latency_ms * 0.8 {
            // If latency is comfortably low, we can afford larger chunks for better quality
            self.optimal_chunk_size = (self.optimal_chunk_size * 5 / 4).min(1024);
        }
    }

    fn calculate_config_hash(&self, config: &SynthesisConfig) -> u64 {
        // Simple hash of key configuration parameters
        // In practice, this should be a proper hash function
        let mut hash = 0u64;
        hash ^= (config.speed * 1000.0) as u64;
        hash ^= (config.pitch_shift * 1000.0) as u64;
        hash ^= (config.energy * 1000.0) as u64;
        if let Some(speaker_id) = config.speaker_id {
            hash ^= (speaker_id as u64) << 16;
        }
        hash
    }

    fn default_quality_curve() -> Vec<(f32, f32)> {
        // Default quality-latency trade-off curve
        // (quality_reduction, latency_reduction)
        vec![
            (0.0, 0.0),  // No reduction
            (0.1, 0.15), // 10% quality reduction -> 15% latency reduction
            (0.2, 0.30), // 20% quality reduction -> 30% latency reduction
            (0.3, 0.45), // 30% quality reduction -> 45% latency reduction
            (0.4, 0.60), // 40% quality reduction -> 60% latency reduction
            (0.5, 0.75), // 50% quality reduction -> 75% latency reduction
        ]
    }
}

/// Latency performance statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LatencyStats {
    /// Average latency in milliseconds
    pub avg_latency_ms: f32,
    /// Latency variance (standard deviation)
    pub latency_variance: f32,
    /// Target latency
    pub target_latency_ms: f32,
    /// Current optimal chunk size
    pub optimal_chunk_size: usize,
    /// Number of measurements collected
    pub measurements_count: usize,
    /// Whether we're meeting the latency target
    pub is_meeting_target: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_predictor() {
        let mut predictor = PerformancePredictor::new(10);

        let measurement = PerformanceMeasurement {
            timestamp: Instant::now(),
            processing_time_ms: 25.0,
            input_complexity: 0.5,
            quality_score: 0.9,
            chunk_size: 256,
            config_hash: 12345,
        };

        predictor.add_measurement(measurement);

        let predicted = predictor.predict_processing_time(0.5, 256, 12345);
        assert!(predicted.is_some());
        assert!((predicted.unwrap() - 25.0).abs() < 1.0);
    }

    #[test]
    fn test_latency_optimizer_config() {
        let config = LatencyOptimizerConfig::default();
        let mut optimizer = LatencyOptimizer::new(config);

        let base_config = SynthesisConfig {
            speed: 1.0,
            pitch_shift: 0.0,
            energy: 1.0,
            speaker_id: None,
            seed: None,
            emotion: None,
            voice_style: None,
        };

        let optimized = optimizer.optimize_config(&base_config, 0.5, Some(30.0));
        assert!(optimized.is_ok());
    }

    #[test]
    fn test_chunk_size_calculation() {
        let config = LatencyOptimizerConfig::default();
        let optimizer = LatencyOptimizer::new(config);

        let chunk_size = optimizer.calculate_optimal_chunk_size(0.5, 30.0);
        assert!((64..=1024).contains(&chunk_size));
    }

    #[test]
    fn test_latency_stats() {
        let config = LatencyOptimizerConfig::default();
        let mut optimizer = LatencyOptimizer::new(config);

        // Add some measurements
        for i in 0..5 {
            optimizer.add_measurement(25.0 + i as f32, 0.5, 0.9, 256, &SynthesisConfig::default());
        }

        let stats = optimizer.get_stats();
        assert!(stats.avg_latency_ms > 0.0);
        assert_eq!(stats.measurements_count, 5);
    }
}
