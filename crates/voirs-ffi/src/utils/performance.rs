//! Performance monitoring and optimization utilities for FFI.

use std::os::raw::{c_char, c_float, c_uint};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// Note: super::* import removed as it causes unused import warning
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

/// Performance benchmark result
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BenchmarkResult {
    pub mean_duration_us: f64,       // Mean execution time in microseconds
    pub min_duration_us: f64,        // Minimum execution time in microseconds
    pub max_duration_us: f64,        // Maximum execution time in microseconds
    pub std_deviation_us: f64,       // Standard deviation in microseconds
    pub throughput_ops_per_sec: f64, // Operations per second
    pub samples_processed: u64,      // Total samples processed
    pub total_iterations: u32,       // Number of benchmark iterations
}

impl Default for BenchmarkResult {
    fn default() -> Self {
        Self {
            mean_duration_us: 0.0,
            min_duration_us: 0.0,
            max_duration_us: 0.0,
            std_deviation_us: 0.0,
            throughput_ops_per_sec: 0.0,
            samples_processed: 0,
            total_iterations: 0,
        }
    }
}

/// Audio processing benchmark suite
pub struct AudioBenchmark {
    durations: Vec<Duration>,
    samples_per_iteration: usize,
}

impl Default for AudioBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioBenchmark {
    pub fn new() -> Self {
        Self {
            durations: Vec::new(),
            samples_per_iteration: 0,
        }
    }

    /// Benchmark audio processing function
    pub fn benchmark_audio_function<F>(
        &mut self,
        samples: &[f32],
        iterations: u32,
        mut function: F,
    ) -> BenchmarkResult
    where
        F: FnMut(&[f32]),
    {
        self.durations.clear();
        self.samples_per_iteration = samples.len();

        // Warmup iterations
        for _ in 0..5 {
            function(samples);
        }

        // Actual benchmark iterations
        for _ in 0..iterations {
            let start = Instant::now();
            function(samples);
            let duration = start.elapsed();
            self.durations.push(duration);
        }

        self.calculate_results(iterations)
    }

    /// Benchmark mutable audio processing function
    pub fn benchmark_audio_function_mut<F>(
        &mut self,
        samples: &[f32],
        iterations: u32,
        mut function: F,
    ) -> BenchmarkResult
    where
        F: FnMut(&mut [f32]),
    {
        self.durations.clear();
        self.samples_per_iteration = samples.len();

        // Warmup iterations
        for _ in 0..5 {
            let mut test_samples = samples.to_vec();
            function(&mut test_samples);
        }

        // Actual benchmark iterations
        for _ in 0..iterations {
            let mut test_samples = samples.to_vec();
            let start = Instant::now();
            function(&mut test_samples);
            let duration = start.elapsed();
            self.durations.push(duration);
        }

        self.calculate_results(iterations)
    }

    fn calculate_results(&self, iterations: u32) -> BenchmarkResult {
        if self.durations.is_empty() {
            return BenchmarkResult::default();
        }

        let durations_us: Vec<f64> = self
            .durations
            .iter()
            .map(|d| d.as_secs_f64() * 1_000_000.0)
            .collect();

        let min_duration = durations_us.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_duration = durations_us
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean_duration = durations_us.iter().sum::<f64>() / durations_us.len() as f64;

        let variance = durations_us
            .iter()
            .map(|x| (x - mean_duration).powi(2))
            .sum::<f64>()
            / durations_us.len() as f64;
        let std_deviation = variance.sqrt();

        let throughput = if mean_duration > 0.0 {
            1_000_000.0 / mean_duration // operations per second
        } else {
            0.0
        };

        BenchmarkResult {
            mean_duration_us: mean_duration,
            min_duration_us: min_duration,
            max_duration_us: max_duration,
            std_deviation_us: std_deviation,
            throughput_ops_per_sec: throughput,
            samples_processed: (self.samples_per_iteration * iterations as usize) as u64,
            total_iterations: iterations,
        }
    }
}

/// Memory allocation pattern analyzer
pub struct MemoryPatternAnalyzer {
    allocation_sizes: VecDeque<usize>,
    allocation_times: VecDeque<Instant>,
    max_history: usize,
}

impl MemoryPatternAnalyzer {
    pub fn new(max_history: usize) -> Self {
        Self {
            allocation_sizes: VecDeque::new(),
            allocation_times: VecDeque::new(),
            max_history,
        }
    }

    pub fn record_allocation(&mut self, size: usize) {
        let now = Instant::now();

        self.allocation_sizes.push_back(size);
        self.allocation_times.push_back(now);

        // Maintain history limit
        while self.allocation_sizes.len() > self.max_history {
            self.allocation_sizes.pop_front();
            self.allocation_times.pop_front();
        }
    }

    pub fn analyze_patterns(&self) -> MemoryPatternStats {
        if self.allocation_sizes.is_empty() {
            return MemoryPatternStats::default();
        }

        let sizes: Vec<usize> = self.allocation_sizes.iter().cloned().collect();
        let total_allocations = sizes.len();
        let total_bytes: usize = sizes.iter().sum();
        let average_size = total_bytes as f64 / total_allocations as f64;

        let mut size_histogram = std::collections::HashMap::new();
        for &size in &sizes {
            let bucket = Self::size_to_bucket(size);
            *size_histogram.entry(bucket).or_insert(0) += 1;
        }

        let allocation_rate = if self.allocation_times.len() >= 2 {
            if let (Some(last_time), Some(first_time)) =
                (self.allocation_times.back(), self.allocation_times.front())
            {
                let time_span = last_time.duration_since(*first_time);
                if time_span.as_secs_f64() > 0.0 {
                    total_allocations as f64 / time_span.as_secs_f64()
                } else {
                    0.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        };

        MemoryPatternStats {
            total_allocations,
            total_bytes,
            average_allocation_size: average_size,
            allocation_rate_per_sec: allocation_rate,
            size_distribution: size_histogram,
        }
    }

    fn size_to_bucket(size: usize) -> String {
        match size {
            0..=1024 => "0-1KB".to_string(),
            1025..=4096 => "1-4KB".to_string(),
            4097..=16384 => "4-16KB".to_string(),
            16385..=65536 => "16-64KB".to_string(),
            65537..=262144 => "64-256KB".to_string(),
            262145..=1048576 => "256KB-1MB".to_string(),
            _ => "1MB+".to_string(),
        }
    }
}

/// Memory allocation pattern statistics
#[derive(Debug, Clone)]
pub struct MemoryPatternStats {
    pub total_allocations: usize,
    pub total_bytes: usize,
    pub average_allocation_size: f64,
    pub allocation_rate_per_sec: f64,
    pub size_distribution: std::collections::HashMap<String, usize>,
}

impl Default for MemoryPatternStats {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            total_bytes: 0,
            average_allocation_size: 0.0,
            allocation_rate_per_sec: 0.0,
            size_distribution: std::collections::HashMap::new(),
        }
    }
}

/// FFI call overhead measurement utility
pub struct FFIOverheadAnalyzer {
    call_durations: Vec<Duration>,
    data_sizes: Vec<usize>,
}

/// Real-time performance monitoring system
pub struct RealTimePerformanceMonitor {
    audio_processing_times: VecDeque<f64>,
    memory_usage_samples: VecDeque<usize>,
    cpu_usage_samples: VecDeque<f64>,
    start_time: Instant,
    sample_interval: Duration,
    max_samples: usize,
}

impl RealTimePerformanceMonitor {
    pub fn new(sample_interval_ms: u64, max_samples: usize) -> Self {
        Self {
            audio_processing_times: VecDeque::new(),
            memory_usage_samples: VecDeque::new(),
            cpu_usage_samples: VecDeque::new(),
            start_time: Instant::now(),
            sample_interval: Duration::from_millis(sample_interval_ms),
            max_samples,
        }
    }

    pub fn record_audio_processing_time(&mut self, duration_ms: f64) {
        self.audio_processing_times.push_back(duration_ms);
        let max_samples = self.max_samples;
        Self::maintain_sample_limit(&mut self.audio_processing_times, max_samples);
    }

    pub fn record_memory_usage(&mut self, bytes: usize) {
        self.memory_usage_samples.push_back(bytes);
        let max_samples = self.max_samples;
        Self::maintain_sample_limit(&mut self.memory_usage_samples, max_samples);
    }

    pub fn record_cpu_usage(&mut self, cpu_percent: f64) {
        self.cpu_usage_samples.push_back(cpu_percent);
        let max_samples = self.max_samples;
        Self::maintain_sample_limit(&mut self.cpu_usage_samples, max_samples);
    }

    fn maintain_sample_limit<T>(samples: &mut VecDeque<T>, max_samples: usize) {
        while samples.len() > max_samples {
            samples.pop_front();
        }
    }

    pub fn get_performance_summary(&self) -> PerformanceSummary {
        let audio_times: Vec<f64> = self.audio_processing_times.iter().cloned().collect();
        let memory_samples: Vec<usize> = self.memory_usage_samples.iter().cloned().collect();
        let cpu_samples: Vec<f64> = self.cpu_usage_samples.iter().cloned().collect();

        let audio_stats = self.calculate_stats(&audio_times);
        let memory_stats = self.calculate_memory_stats(&memory_samples);
        let cpu_stats = self.calculate_stats(&cpu_samples);

        PerformanceSummary {
            uptime_seconds: self.start_time.elapsed().as_secs(),
            audio_processing_avg_ms: audio_stats.mean,
            audio_processing_max_ms: audio_stats.max,
            memory_usage_avg_mb: memory_stats.mean / (1024.0 * 1024.0),
            memory_usage_peak_mb: memory_stats.max / (1024.0 * 1024.0),
            cpu_usage_avg_percent: cpu_stats.mean,
            cpu_usage_peak_percent: cpu_stats.max,
            total_samples_collected: self.audio_processing_times.len()
                + self.memory_usage_samples.len()
                + self.cpu_usage_samples.len(),
        }
    }

    fn calculate_stats(&self, values: &[f64]) -> StatsSummary {
        if values.is_empty() {
            return StatsSummary {
                mean: 0.0,
                max: 0.0,
                min: 0.0,
                std_dev: 0.0,
            };
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);

        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        StatsSummary {
            mean,
            max,
            min,
            std_dev,
        }
    }

    fn calculate_memory_stats(&self, values: &[usize]) -> StatsSummary {
        if values.is_empty() {
            return StatsSummary {
                mean: 0.0,
                max: 0.0,
                min: 0.0,
                std_dev: 0.0,
            };
        }

        let float_values: Vec<f64> = values.iter().map(|&x| x as f64).collect();
        self.calculate_stats(&float_values)
    }
}

/// Statistics summary structure
#[derive(Debug, Clone, Copy)]
pub struct StatsSummary {
    pub mean: f64,
    pub max: f64,
    pub min: f64,
    pub std_dev: f64,
}

/// Real-time performance summary
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PerformanceSummary {
    pub uptime_seconds: u64,
    pub audio_processing_avg_ms: f64,
    pub audio_processing_max_ms: f64,
    pub memory_usage_avg_mb: f64,
    pub memory_usage_peak_mb: f64,
    pub cpu_usage_avg_percent: f64,
    pub cpu_usage_peak_percent: f64,
    pub total_samples_collected: usize,
}

impl Default for PerformanceSummary {
    fn default() -> Self {
        Self {
            uptime_seconds: 0,
            audio_processing_avg_ms: 0.0,
            audio_processing_max_ms: 0.0,
            memory_usage_avg_mb: 0.0,
            memory_usage_peak_mb: 0.0,
            cpu_usage_avg_percent: 0.0,
            cpu_usage_peak_percent: 0.0,
            total_samples_collected: 0,
        }
    }
}

/// Performance regression detector
pub struct PerformanceRegressionDetector {
    baseline_metrics: Vec<f64>,
    current_window: VecDeque<f64>,
    window_size: usize,
    regression_threshold: f64, // Percentage increase considered regression
}

impl PerformanceRegressionDetector {
    pub fn new(window_size: usize, regression_threshold_percent: f64) -> Self {
        Self {
            baseline_metrics: Vec::new(),
            current_window: VecDeque::new(),
            window_size,
            regression_threshold: regression_threshold_percent / 100.0,
        }
    }

    pub fn set_baseline(&mut self, baseline_values: Vec<f64>) {
        self.baseline_metrics = baseline_values;
    }

    pub fn add_measurement(&mut self, value: f64) {
        self.current_window.push_back(value);
        if self.current_window.len() > self.window_size {
            self.current_window.pop_front();
        }
    }

    pub fn check_for_regression(&self) -> RegressionResult {
        if self.baseline_metrics.is_empty() || self.current_window.is_empty() {
            return RegressionResult {
                has_regression: false,
                baseline_avg: 0.0,
                current_avg: 0.0,
                regression_percent: 0.0,
            };
        }

        let baseline_avg =
            self.baseline_metrics.iter().sum::<f64>() / self.baseline_metrics.len() as f64;
        let current_avg =
            self.current_window.iter().sum::<f64>() / self.current_window.len() as f64;

        let regression_percent = if baseline_avg > 0.0 {
            (current_avg - baseline_avg) / baseline_avg
        } else {
            0.0
        };

        let has_regression = regression_percent > self.regression_threshold;

        RegressionResult {
            has_regression,
            baseline_avg,
            current_avg,
            regression_percent,
        }
    }
}

/// Performance regression detection result
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RegressionResult {
    pub has_regression: bool,
    pub baseline_avg: f64,
    pub current_avg: f64,
    pub regression_percent: f64,
}

impl Default for FFIOverheadAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl FFIOverheadAnalyzer {
    pub fn new() -> Self {
        Self {
            call_durations: Vec::new(),
            data_sizes: Vec::new(),
        }
    }

    /// Measure FFI call overhead for a given function
    pub fn measure_ffi_call<F, T>(
        &mut self,
        data_size: usize,
        iterations: u32,
        mut ffi_function: F,
    ) -> FFIOverheadStats
    where
        F: FnMut() -> T,
    {
        self.call_durations.clear();
        self.data_sizes.clear();

        // Warmup
        for _ in 0..5 {
            ffi_function();
        }

        // Measure actual calls
        for _ in 0..iterations {
            let start = Instant::now();
            ffi_function();
            let duration = start.elapsed();

            self.call_durations.push(duration);
            self.data_sizes.push(data_size);
        }

        self.calculate_overhead_stats(iterations)
    }

    fn calculate_overhead_stats(&self, iterations: u32) -> FFIOverheadStats {
        if self.call_durations.is_empty() {
            return FFIOverheadStats::default();
        }

        let durations_ns: Vec<u64> = self
            .call_durations
            .iter()
            .map(|d| d.as_nanos() as u64)
            .collect();

        let min_overhead = durations_ns.iter().min().copied().unwrap_or(0);
        let max_overhead = durations_ns.iter().max().copied().unwrap_or(0);
        let mean_overhead = durations_ns.iter().sum::<u64>() / durations_ns.len() as u64;

        let total_data_processed: usize = self.data_sizes.iter().sum();
        let throughput_mbps = if mean_overhead > 0 {
            (total_data_processed as f64 * 1000.0) / (mean_overhead as f64 * iterations as f64)
        } else {
            0.0
        };

        FFIOverheadStats {
            mean_overhead_ns: mean_overhead,
            min_overhead_ns: min_overhead,
            max_overhead_ns: max_overhead,
            throughput_mb_per_sec: throughput_mbps,
            total_iterations: iterations,
            avg_data_size_bytes: total_data_processed / iterations as usize,
        }
    }
}

/// FFI call overhead statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FFIOverheadStats {
    pub mean_overhead_ns: u64,      // Mean overhead in nanoseconds
    pub min_overhead_ns: u64,       // Minimum overhead in nanoseconds
    pub max_overhead_ns: u64,       // Maximum overhead in nanoseconds
    pub throughput_mb_per_sec: f64, // Data throughput in MB/s
    pub total_iterations: u32,      // Number of iterations tested
    pub avg_data_size_bytes: usize, // Average data size per call
}

impl Default for FFIOverheadStats {
    fn default() -> Self {
        Self {
            mean_overhead_ns: 0,
            min_overhead_ns: 0,
            max_overhead_ns: 0,
            throughput_mb_per_sec: 0.0,
            total_iterations: 0,
            avg_data_size_bytes: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::audio::calculate_rms;

    #[test]
    fn test_audio_benchmark() {
        let samples = vec![0.5f32; 1000];
        let mut benchmark = AudioBenchmark::new();

        let result = benchmark.benchmark_audio_function(&samples, 100, |samples| {
            // Simulate some audio processing work with more computation
            let _rms = calculate_rms(samples);
            // Add more work to ensure measurable timing
            let mut sum = 0.0;
            for &sample in samples {
                sum += sample * sample * sample; // Cubic operation for more work
            }
            std::hint::black_box(sum); // Prevent optimization
        });

        assert!(result.total_iterations == 100);
        assert!(result.samples_processed == 100000);
        assert!(result.mean_duration_us > 0.0);
        assert!(result.throughput_ops_per_sec > 0.0);
    }

    #[test]
    fn test_memory_pattern_analyzer() {
        let mut analyzer = MemoryPatternAnalyzer::new(100);

        // Simulate allocation pattern
        analyzer.record_allocation(1024);
        analyzer.record_allocation(2048);
        analyzer.record_allocation(512);

        let stats = analyzer.analyze_patterns();
        assert_eq!(stats.total_allocations, 3);
        assert_eq!(stats.total_bytes, 3584);
        assert!((stats.average_allocation_size - 1194.67).abs() < 0.1);
    }

    #[test]
    fn test_ffi_overhead_analyzer() {
        let mut analyzer = FFIOverheadAnalyzer::new();

        let stats = analyzer.measure_ffi_call(1000, 5, || {
            // Simulate FFI call overhead
            std::thread::sleep(Duration::from_nanos(100));
            42
        });

        assert_eq!(stats.total_iterations, 5);
        assert_eq!(stats.avg_data_size_bytes, 1000);
        assert!(stats.mean_overhead_ns > 0);
    }

    #[test]
    fn test_real_time_performance_monitor() {
        let mut monitor = RealTimePerformanceMonitor::new(100, 10);

        // Record some sample data
        monitor.record_audio_processing_time(15.5);
        monitor.record_audio_processing_time(12.3);
        monitor.record_memory_usage(1024 * 1024); // 1MB
        monitor.record_cpu_usage(45.0);

        let summary = monitor.get_performance_summary();
        assert!(summary.audio_processing_avg_ms > 0.0);
        assert!(summary.memory_usage_avg_mb > 0.0);
        assert!(summary.cpu_usage_avg_percent > 0.0);
        assert_eq!(summary.total_samples_collected, 4);
    }

    #[test]
    fn test_performance_regression_detector() {
        let mut detector = PerformanceRegressionDetector::new(5, 20.0); // 20% threshold

        // Set baseline (good performance)
        detector.set_baseline(vec![10.0, 11.0, 9.5, 10.5, 10.2]);

        // Add measurements that show no regression
        detector.add_measurement(10.1);
        detector.add_measurement(10.8);
        let result = detector.check_for_regression();
        assert!(!result.has_regression);

        // Add measurements that show regression
        detector.add_measurement(15.0); // 50% increase
        detector.add_measurement(14.5);
        detector.add_measurement(16.0);
        let result = detector.check_for_regression();
        assert!(result.has_regression);
        assert!(result.regression_percent > 0.2); // More than 20% regression
    }

    #[test]
    fn test_performance_monitor_sample_limit() {
        let mut monitor = RealTimePerformanceMonitor::new(100, 3); // Max 3 samples

        // Add more samples than the limit
        monitor.record_audio_processing_time(10.0);
        monitor.record_audio_processing_time(20.0);
        monitor.record_audio_processing_time(30.0);
        monitor.record_audio_processing_time(40.0); // Should evict first sample

        let summary = monitor.get_performance_summary();
        // Should average the last 3 samples: (20 + 30 + 40) / 3 = 30
        assert!((summary.audio_processing_avg_ms - 30.0).abs() < 0.1);
    }
}

#[test]
fn test_spectral_rolloff() {
    use crate::utils::audio;

    // Test with a simple signal
    let samples = vec![1.0, 0.5, 0.2, 0.1, 0.05, 0.01];
    let rolloff = audio::calculate_spectral_rolloff(&samples, 44100);
    assert!(rolloff > 0.0);

    // Test with empty input
    assert_eq!(audio::calculate_spectral_rolloff(&[], 44100), 0.0);
}

#[test]
fn test_spectral_flux() {
    use crate::utils::audio;

    // Test with a changing signal
    let samples = vec![1.0; 1024]; // Constant signal should have low flux
    let flux = audio::calculate_spectral_flux(&samples, 44100);
    assert!(flux >= 0.0);

    // Test with insufficient samples
    let short_samples = vec![1.0; 100];
    assert_eq!(audio::calculate_spectral_flux(&short_samples, 44100), 0.0);
}

#[test]
fn test_brightness() {
    use crate::utils::audio;

    // Test with a signal that has both low and high frequencies
    let samples = vec![1.0, 0.5, 0.2, 0.1, 0.05, 0.01];
    let brightness = audio::calculate_brightness(&samples, 44100);
    assert!(brightness >= 0.0);

    // Test with empty input
    assert_eq!(audio::calculate_brightness(&[], 44100), 0.0);
}

#[test]
fn test_dynamic_compression() {
    use crate::utils::audio;

    let mut samples = vec![0.1, 0.8, 0.9, 0.2, 0.7]; // Mix of low and high amplitude
    let original_samples = samples.clone();

    audio::apply_dynamic_compression(&mut samples, 0.5, 2.0, 0.1, 0.1);

    // The high amplitude samples should be compressed more than low amplitude ones
    assert!(samples[1] <= original_samples[1]); // 0.8 should be compressed
    assert!(samples[2] <= original_samples[2]); // 0.9 should be compressed
    assert!(samples[0] == original_samples[0]); // 0.1 should be unchanged (below threshold)
}

#[test]
fn test_multiband_eq() {
    use crate::utils::audio;

    let mut samples = vec![1.0f32, 0.5, -0.5, -1.0, 0.2, -0.2];
    let original_samples = samples.clone();

    audio::apply_multiband_eq(&mut samples, 1.0, 1.0, 1.0);

    // With gains of 1.0, the output should be similar to input
    for (i, &sample) in samples.iter().enumerate() {
        assert!((sample - original_samples[i]).abs() < 0.5); // Allow some filtering artifacts
    }
}

#[test]
fn test_ffi_spectral_rolloff() {
    use crate::VoirsAudioBuffer;

    let samples = [1.0f32, 0.5, 0.2, 0.1, 0.05, 0.01];
    let buffer = VoirsAudioBuffer {
        samples: samples.as_ptr() as *mut f32,
        length: samples.len() as u32,
        sample_rate: 44100,
        channels: 1,
        duration: samples.len() as f32 / 44100.0,
    };

    let rolloff = unsafe { crate::utils::audio::voirs_audio_calculate_spectral_rolloff(&buffer) };
    assert!(rolloff > 0.0);
}

#[test]
fn test_ffi_spectral_flux() {
    use crate::VoirsAudioBuffer;

    let samples = vec![1.0f32; 1024]; // Constant signal
    let buffer = VoirsAudioBuffer {
        samples: samples.as_ptr() as *mut f32,
        length: samples.len() as u32,
        sample_rate: 44100,
        channels: 1,
        duration: samples.len() as f32 / 44100.0,
    };

    let flux = unsafe { crate::utils::audio::voirs_audio_calculate_spectral_flux(&buffer) };
    assert!(flux >= 0.0);
}

#[test]
fn test_ffi_brightness() {
    use crate::VoirsAudioBuffer;

    let samples = [1.0f32, 0.5, 0.2, 0.1, 0.05, 0.01];
    let buffer = VoirsAudioBuffer {
        samples: samples.as_ptr() as *mut f32,
        length: samples.len() as u32,
        sample_rate: 44100,
        channels: 1,
        duration: samples.len() as f32 / 44100.0,
    };

    let brightness = unsafe { crate::utils::audio::voirs_audio_calculate_brightness(&buffer) };
    assert!(brightness >= 0.0);
}

#[test]
fn test_ffi_compression() {
    use crate::VoirsAudioBuffer;

    let mut samples = vec![0.1f32, 0.8, 0.9, 0.2, 0.7];
    let mut buffer = VoirsAudioBuffer {
        samples: samples.as_mut_ptr(),
        length: samples.len() as u32,
        sample_rate: 44100,
        channels: 1,
        duration: samples.len() as f32 / 44100.0,
    };

    let result = unsafe {
        crate::utils::audio::voirs_audio_apply_compression(&mut buffer, 0.5, 2.0, 0.1, 0.1)
    };
    assert_eq!(result, crate::VoirsErrorCode::Success);

    // High amplitude samples should be compressed
    assert!(samples[1] <= 0.8); // 0.8 should be compressed
    assert!(samples[2] <= 0.9); // 0.9 should be compressed
}

#[test]
fn test_ffi_multiband_eq() {
    use crate::VoirsAudioBuffer;

    let mut samples = vec![1.0f32, 0.5, -0.5, -1.0, 0.2, -0.2];
    let mut buffer = VoirsAudioBuffer {
        samples: samples.as_mut_ptr(),
        length: samples.len() as u32,
        sample_rate: 44100,
        channels: 1,
        duration: samples.len() as f32 / 44100.0,
    };

    let result =
        unsafe { crate::utils::audio::voirs_audio_apply_multiband_eq(&mut buffer, 1.0, 1.0, 1.0) };
    assert_eq!(result, crate::VoirsErrorCode::Success);

    // With gains of 1.0, the samples should be modified but not drastically
    for &sample in &samples {
        assert!(sample.abs() <= 2.0); // Should be reasonably bounded
    }
}
