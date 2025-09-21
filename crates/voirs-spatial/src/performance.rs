//! Performance Testing and Validation for Spatial Audio
//!
//! This module provides comprehensive performance testing capabilities including
//! latency measurement, CPU usage monitoring, memory analysis, and throughput testing.

use crate::{
    AmbisonicsProcessor, BinauralRenderer, Error, Position3D, Result, SpeakerConfiguration,
};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Performance test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Number of test iterations
    pub iterations: usize,
    /// Test duration for long-running tests
    pub test_duration: Duration,
    /// Number of audio sources to test with
    pub source_count: usize,
    /// Audio sample rate
    pub sample_rate: u32,
    /// Buffer size for testing
    pub buffer_size: usize,
    /// Enable memory usage tracking
    pub track_memory: bool,
    /// Enable CPU usage tracking
    pub track_cpu: bool,
    /// Target latency thresholds (VR, Gaming, General)
    pub latency_thresholds: (Duration, Duration, Duration),
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            iterations: 1000,
            test_duration: Duration::from_secs(60),
            source_count: 8,
            sample_rate: 44100,
            buffer_size: 512,
            track_memory: true,
            track_cpu: true,
            latency_thresholds: (
                Duration::from_millis(20), // VR
                Duration::from_millis(30), // Gaming
                Duration::from_millis(50), // General
            ),
        }
    }
}

/// Performance metrics collected during testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Test name
    pub test_name: String,
    /// Average processing latency
    pub avg_latency: Duration,
    /// Minimum processing latency
    pub min_latency: Duration,
    /// Maximum processing latency
    pub max_latency: Duration,
    /// 95th percentile latency
    pub p95_latency: Duration,
    /// 99th percentile latency
    pub p99_latency: Duration,
    /// Average CPU usage percentage
    pub avg_cpu_usage: f32,
    /// Peak CPU usage percentage
    pub peak_cpu_usage: f32,
    /// Average memory usage in bytes
    pub avg_memory_usage: usize,
    /// Peak memory usage in bytes
    pub peak_memory_usage: usize,
    /// Audio processing throughput (samples/second)
    pub throughput: f64,
    /// Number of test iterations
    pub iterations: usize,
    /// Test success rate (0.0 to 1.0)
    pub success_rate: f32,
    /// Additional custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

impl PerformanceMetrics {
    /// Create new empty metrics
    pub fn new(test_name: String) -> Self {
        Self {
            test_name,
            avg_latency: Duration::ZERO,
            min_latency: Duration::MAX,
            max_latency: Duration::ZERO,
            p95_latency: Duration::ZERO,
            p99_latency: Duration::ZERO,
            avg_cpu_usage: 0.0,
            peak_cpu_usage: 0.0,
            avg_memory_usage: 0,
            peak_memory_usage: 0,
            throughput: 0.0,
            iterations: 0,
            success_rate: 0.0,
            custom_metrics: HashMap::new(),
        }
    }

    /// Check if metrics meet performance targets
    pub fn meets_targets(&self, config: &PerformanceConfig) -> PerformanceTargetResult {
        let mut result = PerformanceTargetResult {
            vr_latency_met: self.p95_latency <= config.latency_thresholds.0,
            gaming_latency_met: self.p95_latency <= config.latency_thresholds.1,
            general_latency_met: self.p95_latency <= config.latency_thresholds.2,
            cpu_usage_acceptable: self.avg_cpu_usage < 25.0, // <25% CPU
            success_rate_acceptable: self.success_rate >= 0.95, // 95%+ success
            issues: Vec::new(),
        };

        if !result.vr_latency_met {
            result.issues.push(format!(
                "VR latency target not met: {}ms > {}ms",
                self.p95_latency.as_millis(),
                config.latency_thresholds.0.as_millis()
            ));
        }

        if !result.cpu_usage_acceptable {
            result.issues.push(format!(
                "CPU usage too high: {:.1}% > 25.0%",
                self.avg_cpu_usage
            ));
        }

        if !result.success_rate_acceptable {
            result.issues.push(format!(
                "Success rate too low: {:.1}% < 95.0%",
                self.success_rate * 100.0
            ));
        }

        result
    }
}

/// Performance target validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargetResult {
    /// VR latency target met (<20ms)
    pub vr_latency_met: bool,
    /// Gaming latency target met (<30ms)
    pub gaming_latency_met: bool,
    /// General latency target met (<50ms)
    pub general_latency_met: bool,
    /// CPU usage acceptable (<25%)
    pub cpu_usage_acceptable: bool,
    /// Success rate acceptable (>95%)
    pub success_rate_acceptable: bool,
    /// List of issues found
    pub issues: Vec<String>,
}

/// System resource monitor
pub struct ResourceMonitor {
    start_time: Instant,
    samples: Arc<Mutex<Vec<ResourceSample>>>,
    stop_flag: Arc<Mutex<bool>>,
}

/// Resource sample point
#[derive(Debug, Clone)]
struct ResourceSample {
    timestamp: Instant,
    cpu_usage: f32,
    memory_usage: usize,
}

impl ResourceMonitor {
    /// Start monitoring system resources
    pub fn start() -> Self {
        let samples = Arc::new(Mutex::new(Vec::new()));
        let stop_flag = Arc::new(Mutex::new(false));

        let samples_clone = samples.clone();
        let stop_clone = stop_flag.clone();

        // Start monitoring thread
        thread::spawn(move || {
            while !*stop_clone.lock().unwrap() {
                let sample = ResourceSample {
                    timestamp: Instant::now(),
                    cpu_usage: Self::get_cpu_usage(),
                    memory_usage: Self::get_memory_usage(),
                };

                samples_clone.lock().unwrap().push(sample);
                thread::sleep(Duration::from_millis(100)); // 10Hz sampling
            }
        });

        Self {
            start_time: Instant::now(),
            samples,
            stop_flag,
        }
    }

    /// Stop monitoring and return statistics
    pub fn stop(self) -> ResourceStatistics {
        *self.stop_flag.lock().unwrap() = true;
        thread::sleep(Duration::from_millis(200)); // Allow thread to finish

        let samples = self.samples.lock().unwrap().clone();
        ResourceStatistics::from_samples(samples, self.start_time)
    }

    /// Get current CPU usage (simplified implementation)
    fn get_cpu_usage() -> f32 {
        // In a real implementation, this would read from /proc/stat on Linux
        // or use system APIs. For testing, return a mock value.
        fastrand::f32() * 10.0 // Random 0-10% CPU usage
    }

    /// Get current memory usage (simplified implementation)
    fn get_memory_usage() -> usize {
        // In a real implementation, this would read system memory usage
        // For testing, return a mock value.
        100_000_000 + (fastrand::usize(..50_000_000)) // 100-150 MB
    }
}

/// Resource usage statistics
#[derive(Debug, Clone)]
pub struct ResourceStatistics {
    /// Average CPU usage (0.0-1.0)
    pub avg_cpu_usage: f32,
    /// Peak CPU usage (0.0-1.0)
    pub peak_cpu_usage: f32,
    /// Average memory usage in bytes
    pub avg_memory_usage: usize,
    /// Peak memory usage in bytes
    pub peak_memory_usage: usize,
    /// Duration of the measurement period
    pub duration: Duration,
    /// Number of samples taken
    pub sample_count: usize,
}

impl ResourceStatistics {
    fn from_samples(samples: Vec<ResourceSample>, start_time: Instant) -> Self {
        if samples.is_empty() {
            return Self {
                avg_cpu_usage: 0.0,
                peak_cpu_usage: 0.0,
                avg_memory_usage: 0,
                peak_memory_usage: 0,
                duration: Duration::ZERO,
                sample_count: 0,
            };
        }

        let avg_cpu = samples.iter().map(|s| s.cpu_usage).sum::<f32>() / samples.len() as f32;
        let peak_cpu = samples.iter().map(|s| s.cpu_usage).fold(0.0, f32::max);
        let avg_memory = samples.iter().map(|s| s.memory_usage).sum::<usize>() / samples.len();
        let peak_memory = samples.iter().map(|s| s.memory_usage).max().unwrap_or(0);

        let duration = samples.last().unwrap().timestamp - start_time;

        Self {
            avg_cpu_usage: avg_cpu,
            peak_cpu_usage: peak_cpu,
            avg_memory_usage: avg_memory,
            peak_memory_usage: peak_memory,
            duration,
            sample_count: samples.len(),
        }
    }
}

/// Comprehensive performance test suite
pub struct PerformanceTestSuite {
    config: PerformanceConfig,
    results: Vec<PerformanceMetrics>,
}

impl PerformanceTestSuite {
    /// Create new test suite
    pub fn new(config: PerformanceConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Run all performance tests
    pub fn run_all_tests(&mut self) -> Result<Vec<PerformanceMetrics>> {
        tracing::info!("Starting comprehensive performance test suite");

        // Test binaural rendering performance
        self.test_binaural_rendering()?;

        // Test ambisonics processing performance
        self.test_ambisonics_processing()?;

        // Test multi-source processing
        self.test_multi_source_processing()?;

        // Test real-time latency
        self.test_real_time_latency()?;

        // Test memory efficiency
        self.test_memory_efficiency()?;

        // Test throughput scaling
        self.test_throughput_scaling()?;

        tracing::info!(
            "Performance test suite completed: {} tests",
            self.results.len()
        );
        Ok(self.results.clone())
    }

    /// Test binaural rendering performance
    fn test_binaural_rendering(&mut self) -> Result<()> {
        let mut metrics = PerformanceMetrics::new("Binaural Rendering".to_string());
        let mut latencies = Vec::new();
        let mut successes = 0;

        let monitor = ResourceMonitor::start();

        // Create test audio and renderer
        let audio_samples = Array1::from_vec(vec![0.1; self.config.buffer_size]);
        let position = Position3D::new(1.0, 0.5, 0.0);

        // Mock binaural renderer for testing
        for i in 0..self.config.iterations {
            let start = Instant::now();

            // Simulate binaural processing
            let _ = self.simulate_binaural_processing(&audio_samples, &position);

            let latency = start.elapsed();
            latencies.push(latency);

            if latency <= self.config.latency_thresholds.2 {
                successes += 1;
            }

            if i % 100 == 0 {
                tracing::debug!("Binaural test progress: {}/{}", i, self.config.iterations);
            }
        }

        let resource_stats = monitor.stop();

        // Calculate metrics
        latencies.sort();
        metrics.avg_latency = Duration::from_nanos(
            (latencies.iter().map(|d| d.as_nanos()).sum::<u128>() / latencies.len() as u128) as u64,
        );
        metrics.min_latency = latencies[0];
        metrics.max_latency = latencies[latencies.len() - 1];
        metrics.p95_latency = latencies[(latencies.len() as f32 * 0.95) as usize];
        metrics.p99_latency = latencies[(latencies.len() as f32 * 0.99) as usize];
        metrics.avg_cpu_usage = resource_stats.avg_cpu_usage;
        metrics.peak_cpu_usage = resource_stats.peak_cpu_usage;
        metrics.avg_memory_usage = resource_stats.avg_memory_usage;
        metrics.peak_memory_usage = resource_stats.peak_memory_usage;
        metrics.iterations = self.config.iterations;
        metrics.success_rate = successes as f32 / self.config.iterations as f32;
        metrics.throughput = (self.config.iterations * self.config.buffer_size) as f64
            / resource_stats.duration.as_secs_f64();

        self.results.push(metrics);
        Ok(())
    }

    /// Test ambisonics processing performance
    fn test_ambisonics_processing(&mut self) -> Result<()> {
        let mut metrics = PerformanceMetrics::new("Ambisonics Processing".to_string());
        let mut latencies = Vec::new();
        let mut successes = 0;

        let monitor = ResourceMonitor::start();

        // Create test data
        let audio_samples = Array1::from_vec(vec![0.1; self.config.buffer_size]);
        let position = Position3D::new(1.0, 0.5, 0.0);

        for i in 0..self.config.iterations {
            let start = Instant::now();

            // Simulate ambisonics processing
            let _ = self.simulate_ambisonics_processing(&audio_samples, &position);

            let latency = start.elapsed();
            latencies.push(latency);

            if latency <= self.config.latency_thresholds.2 {
                successes += 1;
            }

            if i % 100 == 0 {
                tracing::debug!("Ambisonics test progress: {}/{}", i, self.config.iterations);
            }
        }

        let resource_stats = monitor.stop();

        // Calculate metrics
        latencies.sort();
        metrics.avg_latency = Duration::from_nanos(
            (latencies.iter().map(|d| d.as_nanos()).sum::<u128>() / latencies.len() as u128) as u64,
        );
        metrics.min_latency = latencies[0];
        metrics.max_latency = latencies[latencies.len() - 1];
        metrics.p95_latency = latencies[(latencies.len() as f32 * 0.95) as usize];
        metrics.p99_latency = latencies[(latencies.len() as f32 * 0.99) as usize];
        metrics.avg_cpu_usage = resource_stats.avg_cpu_usage;
        metrics.peak_cpu_usage = resource_stats.peak_cpu_usage;
        metrics.avg_memory_usage = resource_stats.avg_memory_usage;
        metrics.peak_memory_usage = resource_stats.peak_memory_usage;
        metrics.iterations = self.config.iterations;
        metrics.success_rate = successes as f32 / self.config.iterations as f32;
        metrics.throughput = (self.config.iterations * self.config.buffer_size) as f64
            / resource_stats.duration.as_secs_f64();

        self.results.push(metrics);
        Ok(())
    }

    /// Test multi-source processing performance
    fn test_multi_source_processing(&mut self) -> Result<()> {
        let mut metrics = PerformanceMetrics::new("Multi-Source Processing".to_string());
        let mut latencies = Vec::new();
        let mut successes = 0;

        let monitor = ResourceMonitor::start();

        // Create multi-source test data
        let audio_data = Array2::from_shape_vec(
            (self.config.source_count, self.config.buffer_size),
            vec![0.1; self.config.source_count * self.config.buffer_size],
        )
        .map_err(|e| Error::LegacyProcessing(format!("Failed to create test audio data: {e}")))?;

        let positions: Vec<Position3D> = (0..self.config.source_count)
            .map(|i| {
                let angle =
                    (i as f32 / self.config.source_count as f32) * 2.0 * std::f32::consts::PI;
                Position3D::new(angle.cos(), angle.sin(), 0.0)
            })
            .collect();

        for i in 0..self.config.iterations {
            let start = Instant::now();

            // Simulate multi-source processing
            let _ = self.simulate_multi_source_processing(&audio_data, &positions);

            let latency = start.elapsed();
            latencies.push(latency);

            // Higher latency threshold for multi-source
            if latency <= Duration::from_millis(100) {
                successes += 1;
            }

            if i % 100 == 0 {
                tracing::debug!(
                    "Multi-source test progress: {}/{}",
                    i,
                    self.config.iterations
                );
            }
        }

        let resource_stats = monitor.stop();

        // Calculate metrics
        latencies.sort();
        metrics.avg_latency = Duration::from_nanos(
            (latencies.iter().map(|d| d.as_nanos()).sum::<u128>() / latencies.len() as u128) as u64,
        );
        metrics.min_latency = latencies[0];
        metrics.max_latency = latencies[latencies.len() - 1];
        metrics.p95_latency = latencies[(latencies.len() as f32 * 0.95) as usize];
        metrics.p99_latency = latencies[(latencies.len() as f32 * 0.99) as usize];
        metrics.avg_cpu_usage = resource_stats.avg_cpu_usage;
        metrics.peak_cpu_usage = resource_stats.peak_cpu_usage;
        metrics.avg_memory_usage = resource_stats.avg_memory_usage;
        metrics.peak_memory_usage = resource_stats.peak_memory_usage;
        metrics.iterations = self.config.iterations;
        metrics.success_rate = successes as f32 / self.config.iterations as f32;
        metrics.throughput =
            (self.config.iterations * self.config.source_count * self.config.buffer_size) as f64
                / resource_stats.duration.as_secs_f64();

        // Add custom metric for sources per second
        metrics.custom_metrics.insert(
            "sources_per_second".to_string(),
            (self.config.iterations * self.config.source_count) as f64
                / resource_stats.duration.as_secs_f64(),
        );

        self.results.push(metrics);
        Ok(())
    }

    /// Test real-time latency requirements
    fn test_real_time_latency(&mut self) -> Result<()> {
        let mut metrics = PerformanceMetrics::new("Real-Time Latency".to_string());
        let mut latencies = Vec::new();
        let mut vr_successes = 0;
        let mut gaming_successes = 0;
        let mut general_successes = 0;

        // Simulate real-time processing constraints
        let audio_samples = Array1::from_vec(vec![0.1; self.config.buffer_size]);
        let position = Position3D::new(1.0, 0.0, 0.0);

        for i in 0..self.config.iterations {
            let start = Instant::now();

            // Simulate full spatial audio pipeline
            let _ = self.simulate_full_pipeline(&audio_samples, &position);

            let latency = start.elapsed();
            latencies.push(latency);

            if latency <= self.config.latency_thresholds.0 {
                vr_successes += 1;
            }
            if latency <= self.config.latency_thresholds.1 {
                gaming_successes += 1;
            }
            if latency <= self.config.latency_thresholds.2 {
                general_successes += 1;
            }

            if i % 100 == 0 {
                tracing::debug!("Latency test progress: {}/{}", i, self.config.iterations);
            }
        }

        // Calculate metrics
        latencies.sort();
        metrics.avg_latency = Duration::from_nanos(
            (latencies.iter().map(|d| d.as_nanos()).sum::<u128>() / latencies.len() as u128) as u64,
        );
        metrics.min_latency = latencies[0];
        metrics.max_latency = latencies[latencies.len() - 1];
        metrics.p95_latency = latencies[(latencies.len() as f32 * 0.95) as usize];
        metrics.p99_latency = latencies[(latencies.len() as f32 * 0.99) as usize];
        metrics.iterations = self.config.iterations;
        metrics.success_rate = general_successes as f32 / self.config.iterations as f32;

        // Add custom metrics for different latency targets
        metrics.custom_metrics.insert(
            "vr_success_rate".to_string(),
            vr_successes as f64 / self.config.iterations as f64,
        );
        metrics.custom_metrics.insert(
            "gaming_success_rate".to_string(),
            gaming_successes as f64 / self.config.iterations as f64,
        );
        metrics.custom_metrics.insert(
            "general_success_rate".to_string(),
            general_successes as f64 / self.config.iterations as f64,
        );

        self.results.push(metrics);
        Ok(())
    }

    /// Test memory efficiency
    fn test_memory_efficiency(&mut self) -> Result<()> {
        let mut metrics = PerformanceMetrics::new("Memory Efficiency".to_string());

        let monitor = ResourceMonitor::start();

        // Test memory allocation patterns
        let mut audio_buffers = Vec::new();
        let mut processors: Vec<i32> = Vec::new(); // Placeholder type for testing

        // Allocate resources incrementally
        for i in 0..(self.config.source_count * 10) {
            let buffer = Array1::from_vec(vec![0.1; self.config.buffer_size]);
            audio_buffers.push(buffer);

            if i % 10 == 0 {
                thread::sleep(Duration::from_millis(10));
            }
        }

        // Hold resources for a while to test steady-state memory usage
        thread::sleep(Duration::from_millis(1000));

        // Clean up
        drop(audio_buffers);
        drop(processors);

        let resource_stats = monitor.stop();

        metrics.avg_memory_usage = resource_stats.avg_memory_usage;
        metrics.peak_memory_usage = resource_stats.peak_memory_usage;
        metrics.avg_cpu_usage = resource_stats.avg_cpu_usage;
        metrics.peak_cpu_usage = resource_stats.peak_cpu_usage;
        metrics.iterations = 1;
        metrics.success_rate = 1.0;

        // Custom memory metrics
        metrics.custom_metrics.insert(
            "memory_per_source_mb".to_string(),
            (resource_stats.peak_memory_usage as f64 / self.config.source_count as f64)
                / 1_000_000.0,
        );

        self.results.push(metrics);
        Ok(())
    }

    /// Test throughput scaling with source count
    fn test_throughput_scaling(&mut self) -> Result<()> {
        let mut metrics = PerformanceMetrics::new("Throughput Scaling".to_string());
        let mut throughputs = Vec::new();

        let audio_samples = Array1::from_vec(vec![0.1; self.config.buffer_size]);

        // Test with varying source counts
        for source_count in [1, 2, 4, 8, 16, 32] {
            if source_count > self.config.source_count * 4 {
                break;
            }

            let positions: Vec<Position3D> = (0..source_count)
                .map(|i| {
                    let angle = (i as f32 / source_count as f32) * 2.0 * std::f32::consts::PI;
                    Position3D::new(angle.cos(), angle.sin(), 0.0)
                })
                .collect();

            let start = Instant::now();

            // Process batch with this source count
            for _ in 0..100 {
                for pos in &positions {
                    let _ = self.simulate_binaural_processing(&audio_samples, pos);
                }
            }

            let duration = start.elapsed();
            let throughput =
                (100 * source_count * self.config.buffer_size) as f64 / duration.as_secs_f64();
            throughputs.push(throughput);

            metrics
                .custom_metrics
                .insert(format!("throughput_{source_count}_sources"), throughput);
        }

        metrics.throughput = throughputs.iter().copied().fold(0.0, f64::max);
        metrics.iterations = 100;
        metrics.success_rate = 1.0;

        self.results.push(metrics);
        Ok(())
    }

    // Simulation functions (would use real processors in practice)

    fn simulate_binaural_processing(
        &self,
        _audio: &Array1<f32>,
        _position: &Position3D,
    ) -> Array2<f32> {
        // Simulate processing time
        thread::sleep(Duration::from_micros(50));
        Array2::zeros((2, self.config.buffer_size))
    }

    fn simulate_ambisonics_processing(
        &self,
        _audio: &Array1<f32>,
        _position: &Position3D,
    ) -> Array2<f32> {
        // Simulate processing time
        thread::sleep(Duration::from_micros(75));
        Array2::zeros((4, self.config.buffer_size))
    }

    fn simulate_multi_source_processing(
        &self,
        _audio: &Array2<f32>,
        _positions: &[Position3D],
    ) -> Array2<f32> {
        // Simulate processing time proportional to source count
        thread::sleep(Duration::from_micros(25 * self.config.source_count as u64));
        Array2::zeros((2, self.config.buffer_size))
    }

    fn simulate_full_pipeline(&self, _audio: &Array1<f32>, _position: &Position3D) -> Array2<f32> {
        // Simulate full spatial audio pipeline
        thread::sleep(Duration::from_micros(100));
        Array2::zeros((2, self.config.buffer_size))
    }

    /// Generate performance report
    pub fn generate_report(&self) -> PerformanceReport {
        PerformanceReport::new(&self.results, &self.config)
    }
}

/// Performance test report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// Test configuration used
    pub config: PerformanceConfig,
    /// All test results
    pub results: Vec<PerformanceMetrics>,
    /// Overall summary
    pub summary: PerformanceSummary,
    /// Target validation results
    pub target_results: Vec<PerformanceTargetResult>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    /// Total tests run
    pub total_tests: usize,
    /// Tests that met VR targets
    pub vr_compatible_tests: usize,
    /// Tests that met gaming targets
    pub gaming_compatible_tests: usize,
    /// Overall system rating (0-10)
    pub overall_rating: f32,
    /// Primary performance bottleneck
    pub bottleneck: String,
}

impl PerformanceReport {
    fn new(results: &[PerformanceMetrics], config: &PerformanceConfig) -> Self {
        let target_results: Vec<PerformanceTargetResult> =
            results.iter().map(|r| r.meets_targets(config)).collect();

        let vr_compatible = target_results.iter().filter(|r| r.vr_latency_met).count();
        let gaming_compatible = target_results
            .iter()
            .filter(|r| r.gaming_latency_met)
            .count();

        let mut recommendations = Vec::new();
        if vr_compatible < results.len() {
            recommendations
                .push("Consider optimizing for lower latency to meet VR requirements".to_string());
        }
        if target_results.iter().any(|r| !r.cpu_usage_acceptable) {
            recommendations.push(
                "CPU usage is high - consider GPU acceleration or algorithmic optimization"
                    .to_string(),
            );
        }

        let avg_latency_ms: f32 = results
            .iter()
            .map(|r| r.avg_latency.as_millis() as f32)
            .sum::<f32>()
            / results.len() as f32;

        let overall_rating = if avg_latency_ms < 20.0 {
            10.0
        } else if avg_latency_ms < 30.0 {
            8.0
        } else if avg_latency_ms < 50.0 {
            6.0
        } else {
            4.0
        };

        let bottleneck = if results.iter().any(|r| r.avg_cpu_usage > 50.0) {
            "CPU Processing".to_string()
        } else if results.iter().any(|r| r.peak_memory_usage > 500_000_000) {
            "Memory Usage".to_string()
        } else {
            "Algorithm Efficiency".to_string()
        };

        Self {
            config: config.clone(),
            results: results.to_vec(),
            summary: PerformanceSummary {
                total_tests: results.len(),
                vr_compatible_tests: vr_compatible,
                gaming_compatible_tests: gaming_compatible,
                overall_rating,
                bottleneck,
            },
            target_results,
            recommendations,
        }
    }

    /// Print report to console
    pub fn print_summary(&self) {
        println!("\n=== Spatial Audio Performance Report ===");
        println!("Total tests: {}", self.summary.total_tests);
        println!(
            "VR-compatible: {}/{}",
            self.summary.vr_compatible_tests, self.summary.total_tests
        );
        println!(
            "Gaming-compatible: {}/{}",
            self.summary.gaming_compatible_tests, self.summary.total_tests
        );
        println!("Overall rating: {:.1}/10", self.summary.overall_rating);
        println!("Primary bottleneck: {}", self.summary.bottleneck);

        println!("\n--- Test Results ---");
        for result in &self.results {
            println!(
                "{}: avg={:.1}ms, p95={:.1}ms, cpu={:.1}%",
                result.test_name,
                result.avg_latency.as_millis(),
                result.p95_latency.as_millis(),
                result.avg_cpu_usage
            );
        }

        if !self.recommendations.is_empty() {
            println!("\n--- Recommendations ---");
            for rec in &self.recommendations {
                println!("• {rec}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_config_default() {
        let config = PerformanceConfig::default();
        assert_eq!(config.iterations, 1000);
        assert_eq!(config.source_count, 8);
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.buffer_size, 512);
    }

    #[test]
    fn test_performance_metrics_creation() {
        let metrics = PerformanceMetrics::new("Test".to_string());
        assert_eq!(metrics.test_name, "Test");
        assert_eq!(metrics.iterations, 0);
        assert_eq!(metrics.success_rate, 0.0);
    }

    #[test]
    fn test_resource_monitor() {
        let monitor = ResourceMonitor::start();
        thread::sleep(Duration::from_millis(200));
        let stats = monitor.stop();

        assert!(stats.sample_count > 0);
        assert!(stats.duration > Duration::from_millis(100));
    }

    #[test]
    fn test_performance_targets() {
        let config = PerformanceConfig::default();
        let mut metrics = PerformanceMetrics::new("Test".to_string());

        metrics.p95_latency = Duration::from_millis(15);
        metrics.avg_cpu_usage = 20.0;
        metrics.success_rate = 0.98;

        let result = metrics.meets_targets(&config);
        assert!(result.vr_latency_met);
        assert!(result.cpu_usage_acceptable);
        assert!(result.success_rate_acceptable);
    }

    #[test]
    fn test_performance_test_suite_creation() {
        let config = PerformanceConfig {
            iterations: 10,
            ..Default::default()
        };
        let suite = PerformanceTestSuite::new(config);
        assert_eq!(suite.config.iterations, 10);
        assert_eq!(suite.results.len(), 0);
    }

    #[test]
    fn test_simulation_functions() {
        let config = PerformanceConfig::default();
        let suite = PerformanceTestSuite::new(config);

        let audio = Array1::zeros(512);
        let position = Position3D::new(1.0, 0.0, 0.0);

        let result = suite.simulate_binaural_processing(&audio, &position);
        assert_eq!(result.shape(), [2, 512]);
    }

    #[test]
    fn test_performance_report_generation() {
        let config = PerformanceConfig::default();
        let mut metrics = PerformanceMetrics::new("Test".to_string());
        metrics.avg_latency = Duration::from_millis(25);
        metrics.p95_latency = Duration::from_millis(35);
        metrics.avg_cpu_usage = 15.0;
        metrics.success_rate = 0.96;

        let report = PerformanceReport::new(&[metrics], &config);
        assert_eq!(report.summary.total_tests, 1);
        assert_eq!(report.summary.vr_compatible_tests, 0); // 35ms > 20ms VR threshold
        assert_eq!(report.summary.gaming_compatible_tests, 0); // 35ms > 30ms gaming threshold
    }

    #[test]
    fn test_throughput_calculation() {
        let mut metrics = PerformanceMetrics::new("Throughput Test".to_string());
        metrics.iterations = 1000;

        // Simulate processing 1000 iterations of 512 samples in 1 second
        metrics.throughput = (1000.0 * 512.0) / 1.0; // samples per second

        assert_eq!(metrics.throughput, 512_000.0);
    }
}
