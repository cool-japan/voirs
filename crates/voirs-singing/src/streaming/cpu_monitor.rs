//! Real-time CPU monitoring for adaptive quality scaling
//!
//! Provides lightweight CPU usage monitoring suitable for real-time audio processing.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Real-time CPU usage monitor
///
/// Tracks CPU usage with minimal overhead using sampling and exponential smoothing.
pub struct CpuMonitor {
    /// Last measurement timestamp
    last_measure: Arc<std::sync::Mutex<Instant>>,

    /// Smoothed CPU usage estimate (0.0-1.0), stored as fixed-point (multiply by 10000)
    smoothed_usage: Arc<AtomicU64>,

    /// Exponential smoothing factor (0.0-1.0), stored as fixed-point
    smoothing_factor: f32,

    /// Measurement interval
    measure_interval: Duration,

    /// CPU usage samples for statistics
    samples: Arc<std::sync::Mutex<Vec<f32>>>,

    /// Maximum samples to keep
    max_samples: usize,
}

impl CpuMonitor {
    /// Create a new CPU monitor
    ///
    /// # Arguments
    /// * `smoothing_factor` - Exponential smoothing factor (0.0-1.0, higher = less smoothing)
    /// * `measure_interval` - Time between measurements
    pub fn new(smoothing_factor: f32, measure_interval: Duration) -> Self {
        Self {
            last_measure: Arc::new(std::sync::Mutex::new(Instant::now())),
            smoothed_usage: Arc::new(AtomicU64::new(0)),
            smoothing_factor: smoothing_factor.clamp(0.0, 1.0),
            measure_interval,
            samples: Arc::new(std::sync::Mutex::new(Vec::with_capacity(100))),
            max_samples: 100,
        }
    }

    /// Record CPU time spent on synthesis
    ///
    /// Returns current smoothed CPU usage estimate (0.0-1.0)
    pub fn record_synthesis_time(&self, elapsed: Duration, frame_duration: Duration) -> f32 {
        if let Ok(mut last_measure) = self.last_measure.try_lock() {
            let now = Instant::now();
            if now.duration_since(*last_measure) >= self.measure_interval {
                // Calculate CPU usage for this frame
                let usage = (elapsed.as_secs_f32() / frame_duration.as_secs_f32()).clamp(0.0, 1.0);

                // Update smoothed estimate
                let old_usage = self.get_usage();
                let new_usage =
                    old_usage * (1.0 - self.smoothing_factor) + usage * self.smoothing_factor;

                // Store as fixed-point
                let fixed_point = (new_usage * 10000.0) as u64;
                self.smoothed_usage.store(fixed_point, Ordering::Relaxed);

                // Record sample
                if let Ok(mut samples) = self.samples.try_lock() {
                    samples.push(usage);
                    if samples.len() > self.max_samples {
                        samples.remove(0);
                    }
                }

                *last_measure = now;
                new_usage
            } else {
                self.get_usage()
            }
        } else {
            self.get_usage()
        }
    }

    /// Get current smoothed CPU usage (0.0-1.0)
    pub fn get_usage(&self) -> f32 {
        let fixed_point = self.smoothed_usage.load(Ordering::Relaxed);
        (fixed_point as f32) / 10000.0
    }

    /// Get CPU usage statistics
    pub fn get_stats(&self) -> CpuStats {
        let current_usage = self.get_usage();

        let (peak, avg, min) = if let Ok(samples) = self.samples.lock() {
            let peak = samples.iter().copied().fold(0.0f32, f32::max);
            let min = samples.iter().copied().fold(1.0f32, f32::min);
            let sum: f32 = samples.iter().sum();
            let avg = if !samples.is_empty() {
                sum / samples.len() as f32
            } else {
                0.0
            };
            (peak, avg, min)
        } else {
            (current_usage, current_usage, current_usage)
        };

        CpuStats {
            current_usage,
            peak_usage: peak,
            avg_usage: avg,
            min_usage: min,
            sample_count: self.max_samples,
        }
    }

    /// Reset statistics
    pub fn reset(&self) {
        self.smoothed_usage.store(0, Ordering::Relaxed);
        if let Ok(mut samples) = self.samples.lock() {
            samples.clear();
        }
    }

    /// Check if CPU usage is above threshold
    pub fn is_overloaded(&self, threshold: f32) -> bool {
        self.get_usage() > threshold
    }
}

/// CPU usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuStats {
    /// Current smoothed CPU usage (0.0-1.0)
    pub current_usage: f32,

    /// Peak CPU usage observed
    pub peak_usage: f32,

    /// Average CPU usage
    pub avg_usage: f32,

    /// Minimum CPU usage
    pub min_usage: f32,

    /// Number of samples used for statistics
    pub sample_count: usize,
}

impl Default for CpuStats {
    fn default() -> Self {
        Self {
            current_usage: 0.0,
            peak_usage: 0.0,
            avg_usage: 0.0,
            min_usage: 1.0,
            sample_count: 0,
        }
    }
}

/// System load estimator for adaptive quality scaling
pub struct SystemLoadEstimator {
    /// CPU monitor
    cpu_monitor: CpuMonitor,

    /// Memory pressure indicator (0.0-1.0)
    memory_pressure: Arc<AtomicU64>,

    /// Load history for trend analysis
    load_history: Arc<std::sync::Mutex<Vec<f32>>>,
}

impl SystemLoadEstimator {
    /// Create a new system load estimator
    pub fn new() -> Self {
        Self {
            cpu_monitor: CpuMonitor::new(0.2, Duration::from_millis(50)),
            memory_pressure: Arc::new(AtomicU64::new(0)),
            load_history: Arc::new(std::sync::Mutex::new(Vec::with_capacity(20))),
        }
    }

    /// Update load estimate with synthesis timing
    pub fn update(&self, synthesis_time: Duration, frame_duration: Duration) -> f32 {
        let cpu_usage = self
            .cpu_monitor
            .record_synthesis_time(synthesis_time, frame_duration);

        // Get memory pressure (simplified - would use system APIs in production)
        let memory_pressure = (self.memory_pressure.load(Ordering::Relaxed) as f32) / 10000.0;

        // Combined load is weighted sum of CPU and memory
        let combined_load = cpu_usage * 0.8 + memory_pressure * 0.2;

        // Record in history
        if let Ok(mut history) = self.load_history.try_lock() {
            history.push(combined_load);
            if history.len() > 20 {
                history.remove(0);
            }
        }

        combined_load
    }

    /// Get current system load estimate (0.0-1.0)
    pub fn get_load(&self) -> f32 {
        let cpu_usage = self.cpu_monitor.get_usage();
        let memory_pressure = (self.memory_pressure.load(Ordering::Relaxed) as f32) / 10000.0;
        cpu_usage * 0.8 + memory_pressure * 0.2
    }

    /// Get load trend (positive = increasing, negative = decreasing)
    pub fn get_trend(&self) -> f32 {
        if let Ok(history) = self.load_history.lock() {
            if history.len() < 2 {
                return 0.0;
            }

            // Simple linear regression for trend
            let n = history.len() as f32;
            let sum_x: f32 = (0..history.len()).map(|i| i as f32).sum();
            let sum_y: f32 = history.iter().sum();
            let sum_xy: f32 = history.iter().enumerate().map(|(i, &y)| i as f32 * y).sum();
            let sum_x2: f32 = (0..history.len()).map(|i| (i * i) as f32).sum();

            (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        } else {
            0.0
        }
    }

    /// Set memory pressure (0.0-1.0)
    pub fn set_memory_pressure(&self, pressure: f32) {
        let fixed_point = (pressure.clamp(0.0, 1.0) * 10000.0) as u64;
        self.memory_pressure.store(fixed_point, Ordering::Relaxed);
    }

    /// Get CPU statistics
    pub fn get_cpu_stats(&self) -> CpuStats {
        self.cpu_monitor.get_stats()
    }
}

impl Default for SystemLoadEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_cpu_monitor_creation() {
        let monitor = CpuMonitor::new(0.2, Duration::from_millis(100));
        assert_eq!(monitor.get_usage(), 0.0);
    }

    #[test]
    fn test_cpu_monitor_record() {
        let monitor = CpuMonitor::new(0.5, Duration::from_millis(10));

        // Simulate 50% CPU usage
        let synthesis_time = Duration::from_millis(5);
        let frame_duration = Duration::from_millis(10);

        thread::sleep(Duration::from_millis(15));
        let usage = monitor.record_synthesis_time(synthesis_time, frame_duration);

        assert!(usage >= 0.0 && usage <= 1.0);
    }

    #[test]
    fn test_cpu_monitor_smoothing() {
        let monitor = CpuMonitor::new(0.3, Duration::from_millis(10));

        // Record multiple samples
        for _ in 0..5 {
            thread::sleep(Duration::from_millis(15));
            monitor.record_synthesis_time(Duration::from_millis(7), Duration::from_millis(10));
        }

        let usage = monitor.get_usage();
        assert!(usage > 0.0);
    }

    #[test]
    fn test_cpu_monitor_stats() {
        let monitor = CpuMonitor::new(0.5, Duration::from_millis(10));

        // Record some samples
        for i in 1..=10 {
            thread::sleep(Duration::from_millis(15));
            let time = Duration::from_millis(i);
            monitor.record_synthesis_time(time, Duration::from_millis(10));
        }

        let stats = monitor.get_stats();
        assert!(stats.current_usage >= 0.0);
        assert!(stats.peak_usage >= stats.avg_usage);
        assert!(stats.avg_usage >= stats.min_usage);
    }

    #[test]
    fn test_cpu_monitor_overload() {
        let monitor = CpuMonitor::new(0.8, Duration::from_millis(10));

        // Record multiple high CPU usage samples to build up smoothed estimate
        for _ in 0..5 {
            thread::sleep(Duration::from_millis(15));
            monitor.record_synthesis_time(Duration::from_millis(9), Duration::from_millis(10));
        }

        let usage = monitor.get_usage();
        assert!(usage > 0.5, "CPU usage should be > 0.5, got {}", usage);
        assert!(monitor.is_overloaded(0.5));
        assert!(!monitor.is_overloaded(0.99));
    }

    #[test]
    fn test_cpu_monitor_reset() {
        let monitor = CpuMonitor::new(0.5, Duration::from_millis(10));

        thread::sleep(Duration::from_millis(15));
        monitor.record_synthesis_time(Duration::from_millis(5), Duration::from_millis(10));

        monitor.reset();
        assert_eq!(monitor.get_usage(), 0.0);
    }

    #[test]
    fn test_system_load_estimator() {
        let estimator = SystemLoadEstimator::new();

        thread::sleep(Duration::from_millis(60));
        let load = estimator.update(Duration::from_millis(5), Duration::from_millis(10));

        assert!(load >= 0.0 && load <= 1.0);
    }

    #[test]
    fn test_system_load_with_memory_pressure() {
        let estimator = SystemLoadEstimator::new();

        estimator.set_memory_pressure(0.5);

        thread::sleep(Duration::from_millis(60));
        let load = estimator.update(Duration::from_millis(3), Duration::from_millis(10));

        // Load should reflect both CPU and memory pressure
        assert!(load > 0.0);
    }

    #[test]
    fn test_system_load_trend() {
        let estimator = SystemLoadEstimator::new();

        // Create increasing load trend
        for i in 1..=10 {
            thread::sleep(Duration::from_millis(60));
            let time = Duration::from_millis(i);
            estimator.update(time, Duration::from_millis(10));
        }

        let trend = estimator.get_trend();
        assert!(trend > 0.0); // Positive trend (increasing load)
    }

    #[test]
    fn test_cpu_stats_default() {
        let stats = CpuStats::default();
        assert_eq!(stats.current_usage, 0.0);
        assert_eq!(stats.peak_usage, 0.0);
        assert_eq!(stats.min_usage, 1.0);
    }
}
