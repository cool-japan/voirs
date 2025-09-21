//! Performance benchmarking and optimization for Whisper models
//!
//! This module provides comprehensive benchmarking tools, performance profiling,
//! and optimization suggestions for production Whisper deployments.

use crate::RecognitionError;
// use super::super::whisper_pure::PureRustWhisper;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use voirs_sdk::{AudioBuffer, LanguageCode};

/// Comprehensive benchmarking suite for Whisper models
pub struct WhisperBenchmark {
    config: BenchmarkConfig,
    results: Arc<RwLock<BenchmarkResults>>,
    #[allow(dead_code)]
    profiler: Arc<RwLock<PerformanceProfiler>>,
}

/// Benchmark configuration parameters
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // Configuration struct with related boolean settings
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    pub warmup_iterations: u32,
    /// Number of benchmark iterations
    pub benchmark_iterations: u32,
    /// Test audio durations in seconds
    pub test_durations: Vec<f32>,
    /// Languages to test
    pub test_languages: Vec<LanguageCode>,
    /// Batch sizes to test
    pub batch_sizes: Vec<usize>,
    /// Enable memory profiling
    pub profile_memory: bool,
    /// Enable detailed timing
    pub detailed_timing: bool,
    /// Enable throughput testing
    pub throughput_testing: bool,
    /// Enable latency testing
    pub latency_testing: bool,
    /// Target performance thresholds
    pub performance_targets: PerformanceTargets,
}

/// Performance targets for validation
#[derive(Debug, Clone)]
pub struct PerformanceTargets {
    /// Maximum Real-Time Factor (RTF)
    pub max_rtf: f32,
    /// Maximum processing latency in milliseconds
    pub max_latency_ms: u32,
    /// Minimum throughput in hours/hour
    pub min_throughput: f32,
    /// Maximum memory usage in MB
    pub max_memory_mb: f32,
    /// Minimum accuracy (if available)
    pub min_accuracy: Option<f32>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 5,
            benchmark_iterations: 20,
            test_durations: vec![5.0, 10.0, 30.0, 60.0],
            test_languages: vec![LanguageCode::EnUs, LanguageCode::ZhCn, LanguageCode::EsEs],
            batch_sizes: vec![1, 4, 8],
            profile_memory: true,
            detailed_timing: true,
            throughput_testing: true,
            latency_testing: true,
            performance_targets: PerformanceTargets::default(),
        }
    }
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            max_rtf: 0.5,
            max_latency_ms: 200,
            min_throughput: 10.0,
            max_memory_mb: 4096.0,
            min_accuracy: Some(0.85),
        }
    }
}

/// Comprehensive benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Timestamp when the benchmark was executed
    pub timestamp: SystemTime,
    /// Summary of the configuration used for benchmarking
    pub config_summary: String,
    /// Overall performance metrics
    pub overall_performance: OverallPerformance,
    /// Component-specific benchmark results
    pub component_benchmarks: ComponentBenchmarks,
    /// Detailed latency analysis results
    pub latency_analysis: LatencyAnalysis,
    /// Throughput analysis results
    pub throughput_analysis: ThroughputAnalysis,
    /// Memory usage analysis results
    pub memory_analysis: MemoryAnalysis,
    /// Suggested optimizations based on benchmark results
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
    /// Optional comparison with baseline performance
    pub comparison_baseline: Option<BaselineComparison>,
}

/// Overall performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallPerformance {
    /// Average real-time factor across all test samples
    pub average_rtf: f32,
    /// Median real-time factor
    pub median_rtf: f32,
    /// 95th percentile real-time factor
    pub p95_rtf: f32,
    /// 99th percentile real-time factor
    pub p99_rtf: f32,
    /// Average processing latency in milliseconds
    pub average_latency_ms: u32,
    /// Throughput measured in hours of audio processed per hour
    pub throughput_hours_per_hour: f32,
    /// Peak memory usage in megabytes
    pub peak_memory_mb: f32,
    /// Whether the performance meets target requirements
    pub meets_targets: bool,
    /// Overall performance score from 0-100
    pub performance_score: f32,
}

/// Component-specific benchmarks
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComponentBenchmarks {
    /// Audio preprocessing performance metrics
    pub audio_processing: ComponentPerformance,
    /// Encoder component performance
    pub encoder: ComponentPerformance,
    /// Decoder component performance
    pub decoder: ComponentPerformance,
    /// Tokenizer performance metrics
    pub tokenizer: ComponentPerformance,
    /// End-to-end pipeline performance
    pub end_to_end: ComponentPerformance,
}

/// Individual component performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentPerformance {
    /// Average execution time in milliseconds
    pub average_time_ms: f32,
    /// Minimum execution time observed
    pub min_time_ms: f32,
    /// Maximum execution time observed
    pub max_time_ms: f32,
    /// Standard deviation of execution times
    pub std_dev_ms: f32,
    /// Memory usage in megabytes
    pub memory_usage_mb: f32,
    /// Number of benchmark iterations performed
    pub iterations: u32,
    /// Bottleneck score (0-1, higher means more of a bottleneck)
    pub bottleneck_score: f32,
}

/// Latency analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyAnalysis {
    /// Time to first token output in milliseconds
    pub first_token_latency_ms: f32,
    /// Streaming processing latency in milliseconds
    pub streaming_latency_ms: f32,
    /// Latency by batch size (batch_size -> latency_ms)
    pub batch_latency_ms: HashMap<usize, f32>,
    /// Latency by language (language -> latency_ms)
    pub language_latency_ms: HashMap<String, f32>,
    /// Audio length impact on latency (duration, latency) pairs
    pub audio_length_impact: Vec<(f32, f32)>,
}

/// Throughput analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputAnalysis {
    /// Processing speed for single audio stream in real-time factor
    pub single_stream_throughput: f32,
    /// Maximum number of streams that can be processed in parallel
    pub max_parallel_streams: u32,
    /// Optimal batch size for maximum throughput
    pub optimal_batch_size: usize,
    /// Throughput measurements for different batch sizes
    pub throughput_vs_batch_size: Vec<(usize, f32)>,
    /// Average CPU utilization percentage during processing
    pub cpu_utilization: f32,
    /// Average GPU utilization percentage during processing (if available)
    pub gpu_utilization: Option<f32>,
}

/// Memory analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAnalysis {
    /// Peak memory usage in megabytes during processing
    pub peak_usage_mb: f32,
    /// Average memory usage in megabytes during processing
    pub average_usage_mb: f32,
    /// Memory efficiency in MB per hour of audio processed
    pub memory_efficiency: f32,
    /// Memory growth rate in MB per hour of operation
    pub memory_growth_rate: f32,
    /// Cache hit rate as a percentage (0.0 to 1.0)
    pub cache_hit_rate: f32,
    /// Memory fragmentation level as a percentage (0.0 to 1.0)
    pub fragmentation_level: f32,
}

/// Optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    /// Category of optimization (memory, computation, etc.)
    pub category: OptimizationCategory,
    /// Priority level for implementing this optimization
    pub priority: OptimizationPriority,
    /// Detailed description of the optimization
    pub description: String,
    /// Estimated performance improvement (e.g., "15% faster", "30% less memory")
    pub estimated_improvement: String,
    /// Estimated effort required to implement the optimization
    pub implementation_effort: ImplementationEffort,
    /// References to specific code locations that need modification
    pub code_references: Vec<String>,
}

/// Optimization categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationCategory {
    /// Memory usage optimizations
    Memory,
    /// Computational efficiency optimizations
    Computation,
    /// Input/output performance optimizations
    IO,
    /// Concurrency and parallelization optimizations
    Concurrency,
    /// Algorithm-level optimizations
    Algorithm,
    /// Hardware-specific optimizations
    Hardware,
}

/// Optimization priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationPriority {
    /// Critical optimizations that should be implemented immediately
    Critical,
    /// High priority optimizations with significant impact
    High,
    /// Medium priority optimizations with moderate impact
    Medium,
    /// Low priority optimizations with minor impact
    Low,
}

/// Implementation effort estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    /// Trivial effort (< 1 hour)
    Trivial,
    /// Simple effort (1-4 hours)
    Simple,
    /// Moderate effort (1-3 days)
    Moderate,
    /// Complex effort (1-2 weeks)
    Complex,
    /// Major effort (> 2 weeks)
    Major,
}

/// Baseline comparison for performance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    /// Version identifier for the baseline being compared against
    pub baseline_version: String,
    /// Real-time factor improvement as percentage (positive = better)
    pub rtf_improvement: f32,
    /// Latency improvement as percentage (positive = better)
    pub latency_improvement: f32,
    /// Memory usage improvement as percentage (positive = better)
    pub memory_improvement: f32,
    /// Throughput improvement as percentage (positive = better)
    pub throughput_improvement: f32,
    /// Whether any performance regression was detected
    pub regression_detected: bool,
}

/// Performance profiler for detailed timing
pub struct PerformanceProfiler {
    timing_stack: Vec<TimingEntry>,
    completed_timings: HashMap<String, Vec<Duration>>,
    #[allow(dead_code)]
    memory_snapshots: Vec<MemorySnapshot>,
    #[allow(dead_code)]
    cpu_samples: Vec<CpuSample>,
}

/// Individual timing entry
#[derive(Debug, Clone)]
struct TimingEntry {
    name: String,
    start_time: Instant,
    #[allow(dead_code)]
    memory_start: f32,
}

/// Memory usage snapshot
#[derive(Debug, Clone)]
struct MemorySnapshot {
    #[allow(dead_code)]
    timestamp: Instant,
    #[allow(dead_code)]
    total_mb: f32,
    #[allow(dead_code)]
    component: String,
}

/// CPU usage sample
#[derive(Debug, Clone)]
struct CpuSample {
    #[allow(dead_code)]
    timestamp: Instant,
    #[allow(dead_code)]
    usage_percent: f32,
    #[allow(dead_code)]
    threads: u32,
}

impl WhisperBenchmark {
    #[must_use]
    /// Creates a new benchmark suite with the specified configuration
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: Arc::new(RwLock::new(BenchmarkResults::new())),
            profiler: Arc::new(RwLock::new(PerformanceProfiler::new())),
        }
    }

    /// Run comprehensive benchmark suite
    ///
    /// # Errors
    /// Returns `RecognitionError` if benchmark execution fails
    pub async fn run_full_benchmark(
        &self,
        _model: &(),
    ) -> Result<BenchmarkResults, RecognitionError> {
        let start_time = Instant::now();

        // Warmup phase
        self.warmup_phase(&()).await?;

        // Component benchmarks
        let component_results = self.benchmark_components(&()).await?;

        // End-to-end benchmarks
        let _e2e_results = self.benchmark_end_to_end(&()).await?;

        // Latency analysis
        let latency_results = self.analyze_latency(&()).await?;

        // Throughput analysis
        let throughput_results = self.analyze_throughput(&()).await?;

        // Memory analysis
        let memory_results = self.analyze_memory(&()).await?;

        // Generate optimization suggestions
        let optimizations = self
            .generate_optimizations(&component_results, &memory_results)
            .await;

        // Compile final results
        let mut results = self.results.write().await;
        results.component_benchmarks = component_results;
        results.latency_analysis = latency_results;
        results.throughput_analysis = throughput_results;
        results.memory_analysis = memory_results;
        results.optimization_suggestions = optimizations;
        results.overall_performance = self.calculate_overall_performance(&results).await;

        let total_time = start_time.elapsed();
        println!("Benchmark completed in {:.2}s", total_time.as_secs_f32());

        Ok(results.clone())
    }

    /// Quick performance check
    ///
    /// # Errors
    /// Returns `RecognitionError` if quick benchmark execution fails
    pub async fn quick_benchmark(
        &self,
        _model: &(),
    ) -> Result<OverallPerformance, RecognitionError> {
        // Single test with 10-second audio
        let test_audio = self.generate_test_audio(10.0, 16000).await?;

        let mut timings = Vec::new();
        let mut memory_usage = Vec::new();

        for _ in 0..5 {
            let start = Instant::now();
            let initial_memory = self.get_memory_usage().await;

            // let _result = model.transcribe(&test_audio).await?;
            tokio::time::sleep(std::time::Duration::from_millis(100)).await; // Placeholder

            let duration = start.elapsed();
            let final_memory = self.get_memory_usage().await;

            timings.push(duration);
            memory_usage.push(final_memory - initial_memory);
        }

        let rtfs: Vec<f32> = timings
            .iter()
            .map(|d| d.as_secs_f32() / test_audio.duration())
            .collect();

        #[allow(clippy::cast_precision_loss)]
        // Acceptable precision loss in performance calculations
        let average_rtf = rtfs.iter().sum::<f32>() / rtfs.len() as f32;
        #[allow(clippy::cast_precision_loss)]
        // Acceptable precision loss in performance calculations
        let average_memory = memory_usage.iter().sum::<f32>() / memory_usage.len() as f32;

        Ok(OverallPerformance {
            average_rtf,
            median_rtf: Self::calculate_median(&rtfs),
            p95_rtf: Self::calculate_percentile(&rtfs, 0.95),
            p99_rtf: Self::calculate_percentile(&rtfs, 0.99),
            average_latency_ms: (timings.iter().sum::<Duration>().as_millis()
                / timings.len() as u128)
                .try_into()
                .unwrap_or(u32::MAX),
            throughput_hours_per_hour: 3600.0 / average_rtf / 3600.0,
            peak_memory_mb: average_memory,
            meets_targets: average_rtf <= self.config.performance_targets.max_rtf,
            performance_score: self
                .calculate_performance_score(average_rtf, average_memory)
                .await,
        })
    }

    /// Continuous monitoring mode
    ///
    /// # Errors
    /// Returns `RecognitionError` if benchmark execution fails
    pub async fn start_monitoring(&self, _model: &()) -> Result<(), RecognitionError> {
        let mut interval = tokio::time::interval(Duration::from_secs(60));

        loop {
            interval.tick().await;

            let quick_result = self.quick_benchmark(&()).await?;

            // Check for performance regressions
            if !quick_result.meets_targets {
                println!(
                    "Performance regression detected! RTF: {:.3}, Target: {:.3}",
                    quick_result.average_rtf, self.config.performance_targets.max_rtf
                );
            }

            // Log performance metrics
            self.log_performance_metrics(&quick_result).await;
        }
    }

    // Internal implementation methods

    async fn warmup_phase(&self, _model: &()) -> Result<(), RecognitionError> {
        let _test_audio = self.generate_test_audio(5.0, 16000).await?;

        for _ in 0..self.config.warmup_iterations {
            // let _ = model.transcribe(&test_audio).await?;
            tokio::time::sleep(std::time::Duration::from_millis(50)).await; // Placeholder
        }

        Ok(())
    }

    async fn benchmark_components(
        &self,
        _model: &(),
    ) -> Result<ComponentBenchmarks, RecognitionError> {
        // This would benchmark individual components
        // For now, return placeholder data
        Ok(ComponentBenchmarks {
            audio_processing: ComponentPerformance::default(),
            encoder: ComponentPerformance::default(),
            decoder: ComponentPerformance::default(),
            tokenizer: ComponentPerformance::default(),
            end_to_end: ComponentPerformance::default(),
        })
    }

    async fn benchmark_end_to_end(
        &self,
        _model: &(),
    ) -> Result<ComponentPerformance, RecognitionError> {
        let mut timings = Vec::new();
        let mut memory_usage = Vec::new();

        for duration in &self.config.test_durations {
            let _test_audio = self.generate_test_audio(*duration, 16000).await?;

            for _ in 0..self.config.benchmark_iterations {
                let start = Instant::now();
                let initial_memory = self.get_memory_usage().await;

                // let _ = model.transcribe(&test_audio).await?;
                tokio::time::sleep(std::time::Duration::from_millis(50)).await; // Placeholder

                let elapsed = start.elapsed();
                let final_memory = self.get_memory_usage().await;

                #[allow(clippy::cast_precision_loss)]
                // Acceptable precision loss in timing measurements
                timings.push(elapsed.as_millis() as f32);
                memory_usage.push(final_memory - initial_memory);
            }
        }

        Ok(ComponentPerformance {
            #[allow(clippy::cast_precision_loss)] // Acceptable precision loss in performance calculations
            average_time_ms: timings.iter().sum::<f32>() / timings.len() as f32,
            min_time_ms: timings.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            max_time_ms: timings.iter().fold(0.0, |a, &b| a.max(b)),
            std_dev_ms: Self::calculate_std_dev(&timings),
            #[allow(clippy::cast_precision_loss)] // Acceptable precision loss in memory calculations
            memory_usage_mb: memory_usage.iter().sum::<f32>() / memory_usage.len() as f32,
            iterations: timings.len().try_into().unwrap_or(u32::MAX),
            bottleneck_score: 0.5, // Placeholder
        })
    }

    async fn analyze_latency(&self, _model: &()) -> Result<LatencyAnalysis, RecognitionError> {
        // Placeholder implementation
        Ok(LatencyAnalysis {
            first_token_latency_ms: 50.0,
            streaming_latency_ms: 100.0,
            batch_latency_ms: HashMap::new(),
            language_latency_ms: HashMap::new(),
            audio_length_impact: vec![(5.0, 50.0), (10.0, 75.0), (30.0, 150.0)],
        })
    }

    async fn analyze_throughput(
        &self,
        _model: &(),
    ) -> Result<ThroughputAnalysis, RecognitionError> {
        // Placeholder implementation
        Ok(ThroughputAnalysis {
            single_stream_throughput: 10.0,
            max_parallel_streams: 4,
            optimal_batch_size: 4,
            throughput_vs_batch_size: vec![(1, 2.5), (2, 4.8), (4, 8.5), (8, 12.0)],
            cpu_utilization: 75.0,
            gpu_utilization: Some(85.0),
        })
    }

    async fn analyze_memory(&self, _model: &()) -> Result<MemoryAnalysis, RecognitionError> {
        // Placeholder implementation
        Ok(MemoryAnalysis {
            peak_usage_mb: 2048.0,
            average_usage_mb: 1536.0,
            memory_efficiency: 256.0,
            memory_growth_rate: 0.1,
            cache_hit_rate: 0.85,
            fragmentation_level: 0.15,
        })
    }

    async fn generate_optimizations(
        &self,
        _component_results: &ComponentBenchmarks,
        _memory_results: &MemoryAnalysis,
    ) -> Vec<OptimizationSuggestion> {
        vec![
            OptimizationSuggestion {
                category: OptimizationCategory::Memory,
                priority: OptimizationPriority::High,
                description: "Implement tensor pooling to reduce allocation overhead".to_string(),
                estimated_improvement: "15-25% memory reduction".to_string(),
                implementation_effort: ImplementationEffort::Moderate,
                code_references: vec!["whisper/memory_manager.rs".to_string()],
            },
            OptimizationSuggestion {
                category: OptimizationCategory::Computation,
                priority: OptimizationPriority::Medium,
                description: "Use Flash Attention for longer sequences".to_string(),
                estimated_improvement: "20-30% speed improvement for long audio".to_string(),
                implementation_effort: ImplementationEffort::Complex,
                code_references: vec!["whisper/attention.rs".to_string()],
            },
        ]
    }

    async fn calculate_overall_performance(
        &self,
        _results: &BenchmarkResults,
    ) -> OverallPerformance {
        // Placeholder implementation
        OverallPerformance {
            average_rtf: 0.4,
            median_rtf: 0.38,
            p95_rtf: 0.65,
            p99_rtf: 0.85,
            average_latency_ms: 120,
            throughput_hours_per_hour: 9.0,
            peak_memory_mb: 2048.0,
            meets_targets: true,
            performance_score: 87.5,
        }
    }

    async fn generate_test_audio(
        &self,
        duration: f32,
        sample_rate: u32,
    ) -> Result<AudioBuffer, RecognitionError> {
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )] // Acceptable precision loss in audio generation
        {
            let samples_count = (duration * sample_rate as f32) as usize;
            let samples: Vec<f32> = (0..samples_count)
                .map(|i| {
                    (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.1
                })
                .collect();

            Ok(AudioBuffer::new(samples, sample_rate, 1))
        }
    }

    async fn get_memory_usage(&self) -> f32 {
        // Placeholder - would integrate with actual memory monitoring
        1024.0
    }

    async fn calculate_performance_score(&self, rtf: f32, memory_mb: f32) -> f32 {
        let rtf_score = (1.0 - (rtf / self.config.performance_targets.max_rtf).min(1.0)) * 50.0;
        let memory_score =
            (1.0 - (memory_mb / self.config.performance_targets.max_memory_mb).min(1.0)) * 50.0;
        rtf_score + memory_score
    }

    async fn log_performance_metrics(&self, performance: &OverallPerformance) {
        println!(
            "Performance Update - RTF: {:.3}, Latency: {}ms, Memory: {:.1}MB, Score: {:.1}",
            performance.average_rtf,
            performance.average_latency_ms,
            performance.peak_memory_mb,
            performance.performance_score
        );
    }

    fn calculate_median(values: &[f32]) -> f32 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let len = sorted.len();
        if len % 2 == 0 {
            (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
        } else {
            sorted[len / 2]
        }
    }

    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn calculate_percentile(values: &[f32], percentile: f32) -> f32 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Acceptable for percentile calculation
        let index = (percentile * (sorted.len() - 1) as f32).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    }

    fn calculate_std_dev(values: &[f32]) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        // Acceptable precision loss for statistical calculations
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        #[allow(clippy::cast_precision_loss)]
        // Acceptable precision loss for statistical calculations
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        variance.sqrt()
    }
}

impl BenchmarkResults {
    fn new() -> Self {
        Self {
            timestamp: SystemTime::now(),
            config_summary: "Default benchmark configuration".to_string(),
            overall_performance: OverallPerformance::default(),
            component_benchmarks: ComponentBenchmarks::default(),
            latency_analysis: LatencyAnalysis::default(),
            throughput_analysis: ThroughputAnalysis::default(),
            memory_analysis: MemoryAnalysis::default(),
            optimization_suggestions: Vec::new(),
            comparison_baseline: None,
        }
    }
}

impl Default for OverallPerformance {
    fn default() -> Self {
        Self {
            average_rtf: 0.0,
            median_rtf: 0.0,
            p95_rtf: 0.0,
            p99_rtf: 0.0,
            average_latency_ms: 0,
            throughput_hours_per_hour: 0.0,
            peak_memory_mb: 0.0,
            meets_targets: false,
            performance_score: 0.0,
        }
    }
}

impl Default for ComponentPerformance {
    fn default() -> Self {
        Self {
            average_time_ms: 0.0,
            min_time_ms: 0.0,
            max_time_ms: 0.0,
            std_dev_ms: 0.0,
            memory_usage_mb: 0.0,
            iterations: 0,
            bottleneck_score: 0.0,
        }
    }
}

impl Default for LatencyAnalysis {
    fn default() -> Self {
        Self {
            first_token_latency_ms: 0.0,
            streaming_latency_ms: 0.0,
            batch_latency_ms: HashMap::new(),
            language_latency_ms: HashMap::new(),
            audio_length_impact: Vec::new(),
        }
    }
}

impl Default for ThroughputAnalysis {
    fn default() -> Self {
        Self {
            single_stream_throughput: 0.0,
            max_parallel_streams: 0,
            optimal_batch_size: 1,
            throughput_vs_batch_size: Vec::new(),
            cpu_utilization: 0.0,
            gpu_utilization: None,
        }
    }
}

impl Default for MemoryAnalysis {
    fn default() -> Self {
        Self {
            peak_usage_mb: 0.0,
            average_usage_mb: 0.0,
            memory_efficiency: 0.0,
            memory_growth_rate: 0.0,
            cache_hit_rate: 0.0,
            fragmentation_level: 0.0,
        }
    }
}

impl PerformanceProfiler {
    fn new() -> Self {
        Self {
            timing_stack: Vec::new(),
            completed_timings: HashMap::new(),
            memory_snapshots: Vec::new(),
            cpu_samples: Vec::new(),
        }
    }

    /// Starts timing a named operation
    pub fn start_timing(&mut self, name: String) {
        self.timing_stack.push(TimingEntry {
            name,
            start_time: Instant::now(),
            memory_start: 0.0, // Would integrate with actual memory monitoring
        });
    }

    /// Ends timing the most recently started operation and returns its duration
    pub fn end_timing(&mut self) -> Option<Duration> {
        if let Some(entry) = self.timing_stack.pop() {
            let duration = entry.start_time.elapsed();
            self.completed_timings
                .entry(entry.name)
                .or_default()
                .push(duration);
            Some(duration)
        } else {
            None
        }
    }

    #[must_use]
    /// Gets timing statistics (min, max, average) for a named operation
    pub fn get_timing_stats(&self, name: &str) -> Option<(Duration, Duration, Duration)> {
        if let Some(timings) = self.completed_timings.get(name) {
            let min = timings.iter().min().copied()?;
            let max = timings.iter().max().copied()?;
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let avg = Duration::from_nanos(
                (timings
                    .iter()
                    .map(std::time::Duration::as_nanos)
                    .sum::<u128>()
                    / timings.len() as u128) as u64,
            );
            Some((min, max, avg))
        } else {
            None
        }
    }
}
