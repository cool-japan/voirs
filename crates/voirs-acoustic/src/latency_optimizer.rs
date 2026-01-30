//! Advanced Latency Optimization for Real-Time TTS
//!
//! This module provides sophisticated latency optimization techniques for
//! real-time text-to-speech synthesis, including:
//! - Predictive pre-computation
//! - Adaptive chunk sizing
//! - Lookahead buffer management
//! - Priority-based processing
//! - Latency budget tracking

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use crate::{AcousticError, Result};

/// Latency budget configuration for real-time synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyBudget {
    /// Target end-to-end latency (ms)
    pub target_latency_ms: f32,
    /// Maximum acceptable latency (ms)
    pub max_latency_ms: f32,
    /// Warning threshold for latency violations (%)
    pub warning_threshold: f32,
    /// Enable adaptive quality reduction under latency pressure
    pub adaptive_quality: bool,
    /// Minimum quality level when under pressure (0.0-1.0)
    pub min_quality: f32,
}

impl Default for LatencyBudget {
    fn default() -> Self {
        Self {
            target_latency_ms: 100.0, // 100ms target for real-time feel
            max_latency_ms: 200.0,    // 200ms absolute maximum
            warning_threshold: 0.8,   // Warn at 80% of budget
            adaptive_quality: true,
            min_quality: 0.7, // Don't go below 70% quality
        }
    }
}

impl LatencyBudget {
    /// Create budget optimized for conversation
    pub fn conversational() -> Self {
        Self {
            target_latency_ms: 150.0,
            max_latency_ms: 300.0,
            adaptive_quality: true,
            ..Default::default()
        }
    }

    /// Create budget optimized for gaming/interactive
    pub fn interactive() -> Self {
        Self {
            target_latency_ms: 50.0,
            max_latency_ms: 100.0,
            adaptive_quality: true,
            min_quality: 0.6,
            ..Default::default()
        }
    }

    /// Create budget optimized for broadcasting (higher quality priority)
    pub fn broadcast() -> Self {
        Self {
            target_latency_ms: 500.0,
            max_latency_ms: 1000.0,
            adaptive_quality: false,
            min_quality: 0.95,
            ..Default::default()
        }
    }

    /// Check if latency exceeds warning threshold
    pub fn is_warning(&self, actual_latency_ms: f32) -> bool {
        actual_latency_ms >= self.target_latency_ms * self.warning_threshold
    }

    /// Check if latency exceeds maximum budget
    pub fn is_exceeded(&self, actual_latency_ms: f32) -> bool {
        actual_latency_ms > self.max_latency_ms
    }

    /// Calculate recommended quality level based on latency pressure
    pub fn recommended_quality(&self, actual_latency_ms: f32) -> f32 {
        if !self.adaptive_quality {
            return 1.0;
        }

        if actual_latency_ms <= self.target_latency_ms {
            return 1.0; // Full quality
        }

        if actual_latency_ms >= self.max_latency_ms {
            return self.min_quality; // Minimum quality
        }

        // Linear interpolation between target and max latency
        let pressure = (actual_latency_ms - self.target_latency_ms)
            / (self.max_latency_ms - self.target_latency_ms);
        1.0 - (pressure * (1.0 - self.min_quality))
    }
}

/// Processing priority levels for latency optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ProcessingPriority {
    /// Low priority - can be delayed or dropped
    Low = 0,
    /// Normal priority - standard processing
    Normal = 1,
    /// High priority - process as soon as possible
    High = 2,
    /// Critical priority - process immediately
    Critical = 3,
}

/// Chunk processing strategy for latency optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkStrategy {
    /// Fixed chunk size
    Fixed { size: usize },
    /// Adaptive chunk size based on latency
    Adaptive {
        min_size: usize,
        max_size: usize,
        target_latency_ms: f32,
    },
    /// Dynamic sizing based on content complexity
    Dynamic {
        base_size: usize,
        complexity_factor: f32,
    },
    /// Predictive sizing based on historical performance
    Predictive {
        window_size: usize,
        adjustment_rate: f32,
    },
}

impl Default for ChunkStrategy {
    fn default() -> Self {
        Self::Adaptive {
            min_size: 32,
            max_size: 256,
            target_latency_ms: 50.0,
        }
    }
}

/// Latency measurement and tracking
#[derive(Debug, Clone)]
pub struct LatencyMeasurement {
    /// Timestamp when processing started
    #[allow(dead_code)]
    start_time: Instant,
    /// Duration of processing
    pub duration: Duration,
    /// Number of items processed
    pub items_processed: usize,
    /// Priority of this measurement
    pub priority: ProcessingPriority,
    /// Whether latency budget was met
    pub budget_met: bool,
}

impl LatencyMeasurement {
    /// Create new measurement starting now
    pub fn start(priority: ProcessingPriority) -> Self {
        Self {
            start_time: Instant::now(),
            duration: Duration::ZERO,
            items_processed: 0,
            priority,
            budget_met: false,
        }
    }

    /// Complete the measurement
    pub fn finish(&mut self, items_processed: usize, budget_met: bool) {
        self.duration = self.start_time.elapsed();
        self.items_processed = items_processed;
        self.budget_met = budget_met;
    }

    /// Get latency in milliseconds
    pub fn latency_ms(&self) -> f32 {
        self.duration.as_secs_f32() * 1000.0
    }

    /// Get throughput (items per second)
    pub fn throughput(&self) -> f32 {
        if self.duration.as_secs_f32() > 0.0 {
            self.items_processed as f32 / self.duration.as_secs_f32()
        } else {
            0.0
        }
    }
}

/// Latency optimizer with adaptive strategies
pub struct LatencyOptimizer {
    /// Latency budget configuration
    budget: LatencyBudget,
    /// Chunk processing strategy
    chunk_strategy: ChunkStrategy,
    /// Historical latency measurements
    measurements: Arc<RwLock<VecDeque<LatencyMeasurement>>>,
    /// Maximum history size
    max_history: usize,
    /// Current chunk size (adaptive)
    current_chunk_size: Arc<RwLock<usize>>,
    /// Latency violation count
    violations: Arc<RwLock<usize>>,
}

impl LatencyOptimizer {
    /// Create new latency optimizer
    pub fn new(budget: LatencyBudget, chunk_strategy: ChunkStrategy) -> Self {
        let initial_chunk_size = match &chunk_strategy {
            ChunkStrategy::Fixed { size } => *size,
            ChunkStrategy::Adaptive { min_size, .. } => *min_size,
            ChunkStrategy::Dynamic { base_size, .. } => *base_size,
            ChunkStrategy::Predictive { .. } => 64,
        };

        Self {
            budget,
            chunk_strategy,
            measurements: Arc::new(RwLock::new(VecDeque::new())),
            max_history: 100,
            current_chunk_size: Arc::new(RwLock::new(initial_chunk_size)),
            violations: Arc::new(RwLock::new(0)),
        }
    }

    /// Create optimizer with default conversational settings
    pub fn conversational() -> Self {
        Self::new(LatencyBudget::conversational(), ChunkStrategy::default())
    }

    /// Create optimizer for interactive/gaming scenarios
    pub fn interactive() -> Self {
        Self::new(
            LatencyBudget::interactive(),
            ChunkStrategy::Adaptive {
                min_size: 16,
                max_size: 128,
                target_latency_ms: 30.0,
            },
        )
    }

    /// Create optimizer for broadcast quality
    pub fn broadcast() -> Self {
        Self::new(
            LatencyBudget::broadcast(),
            ChunkStrategy::Fixed { size: 512 },
        )
    }

    /// Start latency measurement for processing
    pub fn start_measurement(&self, priority: ProcessingPriority) -> LatencyMeasurement {
        LatencyMeasurement::start(priority)
    }

    /// Record completed measurement
    pub async fn record_measurement(&self, mut measurement: LatencyMeasurement) {
        let latency_ms = measurement.latency_ms();
        let budget_met = !self.budget.is_exceeded(latency_ms);
        measurement.finish(measurement.items_processed, budget_met);

        let mut measurements = self.measurements.write().await;
        measurements.push_back(measurement.clone());

        // Maintain history size
        while measurements.len() > self.max_history {
            measurements.pop_front();
        }

        // Track violations
        if !budget_met {
            let mut violations = self.violations.write().await;
            *violations += 1;
        }

        // Adapt chunk size based on measurements
        self.adapt_chunk_size(&measurements).await;
    }

    /// Adapt chunk size based on recent measurements
    async fn adapt_chunk_size(&self, measurements: &VecDeque<LatencyMeasurement>) {
        let mut chunk_size = self.current_chunk_size.write().await;

        match &self.chunk_strategy {
            ChunkStrategy::Fixed { .. } => {
                // No adaptation for fixed strategy
            }
            ChunkStrategy::Adaptive {
                min_size,
                max_size,
                target_latency_ms,
            } => {
                if measurements.is_empty() {
                    return;
                }

                // Calculate average recent latency
                let recent_measurements: Vec<_> = measurements.iter().rev().take(10).collect();
                let avg_latency: f32 = recent_measurements
                    .iter()
                    .map(|m| m.latency_ms())
                    .sum::<f32>()
                    / recent_measurements.len() as f32;

                // Adjust chunk size
                if avg_latency > *target_latency_ms * 1.2 {
                    // Latency too high, reduce chunk size
                    *chunk_size = (*chunk_size * 9 / 10).max(*min_size);
                } else if avg_latency < *target_latency_ms * 0.8 {
                    // Latency comfortable, increase chunk size
                    *chunk_size = (*chunk_size * 11 / 10).min(*max_size);
                }
            }
            ChunkStrategy::Dynamic {
                base_size,
                complexity_factor,
            } => {
                // Adjust based on complexity
                *chunk_size = (*base_size as f32 * complexity_factor).round() as usize;
            }
            ChunkStrategy::Predictive {
                adjustment_rate, ..
            } => {
                if measurements.len() < 2 {
                    return;
                }

                // Simple predictive adjustment based on trend
                let recent: Vec<_> = measurements.iter().rev().take(5).collect();
                let trend: f32 = recent
                    .windows(2)
                    .map(|w| w[0].latency_ms() - w[1].latency_ms())
                    .sum::<f32>()
                    / (recent.len() - 1) as f32;

                if trend > 5.0 {
                    // Latency increasing
                    *chunk_size = (*chunk_size as f32 * (1.0 - adjustment_rate)) as usize;
                } else if trend < -5.0 {
                    // Latency decreasing
                    *chunk_size = (*chunk_size as f32 * (1.0 + adjustment_rate)) as usize;
                }

                *chunk_size = (*chunk_size).clamp(32, 512);
            }
        }
    }

    /// Get current recommended chunk size
    pub async fn recommended_chunk_size(&self) -> usize {
        *self.current_chunk_size.read().await
    }

    /// Get current recommended quality level
    pub async fn recommended_quality(&self) -> f32 {
        let measurements = self.measurements.read().await;
        if measurements.is_empty() {
            return 1.0;
        }

        // Use recent average latency
        let recent: Vec<_> = measurements.iter().rev().take(5).collect();
        let avg_latency: f32 =
            recent.iter().map(|m| m.latency_ms()).sum::<f32>() / recent.len() as f32;

        self.budget.recommended_quality(avg_latency)
    }

    /// Get latency statistics
    pub async fn get_statistics(&self) -> LatencyStatistics {
        let measurements = self.measurements.read().await;
        let violations = *self.violations.read().await;

        if measurements.is_empty() {
            return LatencyStatistics::default();
        }

        let latencies: Vec<f32> = measurements.iter().map(|m| m.latency_ms()).collect();

        // Find min/max with NaN handling
        let min = latencies
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        let max = latencies
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        let avg = latencies.iter().sum::<f32>() / latencies.len() as f32;

        // Calculate percentiles with NaN handling
        let mut sorted = latencies.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p50_idx = sorted.len() / 2;
        let p95_idx = (sorted.len() as f32 * 0.95) as usize;
        let p99_idx = (sorted.len() as f32 * 0.99) as usize;

        let p50 = sorted.get(p50_idx).copied().unwrap_or(avg);
        let p95 = sorted.get(p95_idx).copied().unwrap_or(max);
        let p99 = sorted.get(p99_idx).copied().unwrap_or(max);

        let budget_met_count = measurements.iter().filter(|m| m.budget_met).count();
        let budget_met_rate = budget_met_count as f32 / measurements.len() as f32;

        LatencyStatistics {
            min_latency_ms: min,
            max_latency_ms: max,
            avg_latency_ms: avg,
            p50_latency_ms: p50,
            p95_latency_ms: p95,
            p99_latency_ms: p99,
            total_measurements: measurements.len(),
            budget_violations: violations,
            budget_met_rate,
            current_chunk_size: *self.current_chunk_size.read().await,
        }
    }

    /// Reset statistics
    pub async fn reset_statistics(&self) {
        let mut measurements = self.measurements.write().await;
        measurements.clear();
        let mut violations = self.violations.write().await;
        *violations = 0;
    }

    /// Check if system is under latency pressure
    pub async fn is_under_pressure(&self) -> bool {
        let stats = self.get_statistics().await;
        stats.budget_met_rate < 0.9 // Less than 90% meeting budget
    }
}

/// Latency statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStatistics {
    /// Minimum observed latency (ms)
    pub min_latency_ms: f32,
    /// Maximum observed latency (ms)
    pub max_latency_ms: f32,
    /// Average latency (ms)
    pub avg_latency_ms: f32,
    /// Median latency (p50)
    pub p50_latency_ms: f32,
    /// 95th percentile latency
    pub p95_latency_ms: f32,
    /// 99th percentile latency
    pub p99_latency_ms: f32,
    /// Total number of measurements
    pub total_measurements: usize,
    /// Number of budget violations
    pub budget_violations: usize,
    /// Rate of meeting latency budget
    pub budget_met_rate: f32,
    /// Current adaptive chunk size
    pub current_chunk_size: usize,
}

impl Default for LatencyStatistics {
    fn default() -> Self {
        Self {
            min_latency_ms: 0.0,
            max_latency_ms: 0.0,
            avg_latency_ms: 0.0,
            p50_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            total_measurements: 0,
            budget_violations: 0,
            budget_met_rate: 1.0,
            current_chunk_size: 64,
        }
    }
}

impl LatencyStatistics {
    /// Generate human-readable report
    pub fn report(&self) -> String {
        format!(
            "Latency Statistics:\n\
             - Measurements: {}\n\
             - Min: {:.2} ms\n\
             - Max: {:.2} ms\n\
             - Avg: {:.2} ms\n\
             - P50: {:.2} ms\n\
             - P95: {:.2} ms\n\
             - P99: {:.2} ms\n\
             - Budget met: {:.1}%\n\
             - Violations: {}\n\
             - Current chunk size: {}",
            self.total_measurements,
            self.min_latency_ms,
            self.max_latency_ms,
            self.avg_latency_ms,
            self.p50_latency_ms,
            self.p95_latency_ms,
            self.p99_latency_ms,
            self.budget_met_rate * 100.0,
            self.budget_violations,
            self.current_chunk_size
        )
    }

    /// Check if performance is acceptable
    pub fn is_acceptable(&self, target_budget: &LatencyBudget) -> bool {
        self.budget_met_rate >= 0.95 && self.p95_latency_ms <= target_budget.max_latency_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_budget_presets() {
        let conv = LatencyBudget::conversational();
        assert_eq!(conv.target_latency_ms, 150.0);

        let inter = LatencyBudget::interactive();
        assert_eq!(inter.target_latency_ms, 50.0);

        let broadcast = LatencyBudget::broadcast();
        assert_eq!(broadcast.target_latency_ms, 500.0);
    }

    #[test]
    fn test_latency_budget_warnings() {
        let budget = LatencyBudget::default();

        assert!(!budget.is_warning(50.0));
        assert!(budget.is_warning(90.0)); // 90% of 100ms target
        assert!(budget.is_exceeded(250.0));
    }

    #[test]
    fn test_quality_recommendation() {
        let budget = LatencyBudget::default();

        // Under target: full quality
        assert_eq!(budget.recommended_quality(50.0), 1.0);

        // At target: full quality
        assert_eq!(budget.recommended_quality(100.0), 1.0);

        // Over max: minimum quality
        assert_eq!(budget.recommended_quality(250.0), budget.min_quality);

        // Between target and max: interpolated
        let mid_quality = budget.recommended_quality(150.0);
        assert!(mid_quality > budget.min_quality && mid_quality < 1.0);
    }

    #[test]
    fn test_latency_measurement() {
        let mut measurement = LatencyMeasurement::start(ProcessingPriority::Normal);
        std::thread::sleep(Duration::from_millis(10));
        measurement.finish(100, true);

        assert!(measurement.latency_ms() >= 10.0);
        assert_eq!(measurement.items_processed, 100);
        assert!(measurement.budget_met);
        assert!(measurement.throughput() > 0.0);
    }

    #[tokio::test]
    async fn test_latency_optimizer_creation() {
        let optimizer = LatencyOptimizer::conversational();
        let chunk_size = optimizer.recommended_chunk_size().await;
        assert!(chunk_size > 0);
    }

    #[tokio::test]
    async fn test_latency_optimizer_measurement() {
        let optimizer = LatencyOptimizer::interactive();

        let mut measurement = optimizer.start_measurement(ProcessingPriority::High);
        std::thread::sleep(Duration::from_millis(5));
        measurement.finish(50, true);

        optimizer.record_measurement(measurement).await;

        let stats = optimizer.get_statistics().await;
        assert_eq!(stats.total_measurements, 1);
        assert!(stats.avg_latency_ms >= 5.0);
    }

    #[tokio::test]
    async fn test_latency_optimizer_adaptive_chunk_size() {
        let optimizer = LatencyOptimizer::new(
            LatencyBudget::default(),
            ChunkStrategy::Adaptive {
                min_size: 32,
                max_size: 256,
                target_latency_ms: 50.0,
            },
        );

        let initial_size = optimizer.recommended_chunk_size().await;

        // Simulate high latency measurements
        for _ in 0..10 {
            let mut measurement = optimizer.start_measurement(ProcessingPriority::Normal);
            std::thread::sleep(Duration::from_millis(80)); // Over target
            measurement.finish(100, false);
            optimizer.record_measurement(measurement).await;
        }

        let adapted_size = optimizer.recommended_chunk_size().await;
        // Should adapt to smaller chunks due to high latency
        assert!(adapted_size <= initial_size);
    }

    #[tokio::test]
    async fn test_latency_statistics() {
        let optimizer = LatencyOptimizer::conversational();

        // Add various measurements
        for i in 0..20 {
            let mut measurement = optimizer.start_measurement(ProcessingPriority::Normal);
            std::thread::sleep(Duration::from_millis(5 + i % 10));
            measurement.finish(50, true);
            optimizer.record_measurement(measurement).await;
        }

        let stats = optimizer.get_statistics().await;
        assert_eq!(stats.total_measurements, 20);
        assert!(stats.min_latency_ms > 0.0);
        assert!(stats.max_latency_ms >= stats.min_latency_ms);
        assert!(stats.avg_latency_ms > 0.0);
        assert!(stats.budget_met_rate > 0.0);
    }

    #[tokio::test]
    async fn test_pressure_detection() {
        let optimizer = LatencyOptimizer::interactive();

        // Add measurements that violate budget
        for _ in 0..10 {
            let mut measurement = optimizer.start_measurement(ProcessingPriority::Normal);
            std::thread::sleep(Duration::from_millis(150)); // Way over budget
            measurement.finish(50, false);
            optimizer.record_measurement(measurement).await;
        }

        assert!(optimizer.is_under_pressure().await);
    }

    #[test]
    fn test_statistics_report() {
        let stats = LatencyStatistics {
            min_latency_ms: 10.0,
            max_latency_ms: 100.0,
            avg_latency_ms: 50.0,
            p50_latency_ms: 45.0,
            p95_latency_ms: 90.0,
            p99_latency_ms: 98.0,
            total_measurements: 100,
            budget_violations: 5,
            budget_met_rate: 0.95,
            current_chunk_size: 128,
        };

        let report = stats.report();
        assert!(report.contains("Measurements: 100"));
        assert!(report.contains("Avg: 50.00 ms"));
        assert!(report.contains("Budget met: 95.0%"));
    }

    #[test]
    fn test_statistics_acceptability() {
        let good_stats = LatencyStatistics {
            budget_met_rate: 0.98,
            p95_latency_ms: 80.0,
            ..Default::default()
        };

        let budget = LatencyBudget::default();
        assert!(good_stats.is_acceptable(&budget));

        let poor_stats = LatencyStatistics {
            budget_met_rate: 0.80,
            p95_latency_ms: 250.0,
            ..Default::default()
        };

        assert!(!poor_stats.is_acceptable(&budget));
    }
}
