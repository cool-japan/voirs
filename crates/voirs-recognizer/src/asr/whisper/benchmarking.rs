//! Performance benchmarking and optimization for Whisper models
//!
//! This module provides comprehensive benchmarking tools, performance profiling,
//! and optimization suggestions for production Whisper deployments.

use crate::traits::{ASRConfig, ASRModel};
use crate::RecognitionError;
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
    /// Latency by batch size (`batch_size` -> `latency_ms`)
    pub batch_latency_ms: HashMap<usize, f32>,
    /// Latency by language (language -> `latency_ms`)
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
    pub async fn run_full_benchmark<M: ASRModel + ?Sized>(
        &self,
        model: &M,
    ) -> Result<BenchmarkResults, RecognitionError> {
        let start_time = Instant::now();

        // Warmup phase
        self.warmup_phase(model).await?;

        // Component benchmarks
        let component_results = self.benchmark_components(model).await?;

        // End-to-end benchmarks
        let _e2e_results = self.benchmark_end_to_end(model).await?;

        // Latency analysis
        let latency_results = self.analyze_latency(model).await?;

        // Throughput analysis
        let throughput_results = self.analyze_throughput(model).await?;

        // Memory analysis
        let memory_results = self.analyze_memory(model).await?;

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
        let overall = self.calculate_overall_performance(&results);
        results.overall_performance = overall;

        let total_time = start_time.elapsed();
        println!("Benchmark completed in {:.2}s", total_time.as_secs_f32());

        Ok(results.clone())
    }

    /// Quick performance check
    ///
    /// # Errors
    /// Returns `RecognitionError` if quick benchmark execution fails
    pub async fn quick_benchmark<M: ASRModel + ?Sized>(
        &self,
        model: &M,
    ) -> Result<OverallPerformance, RecognitionError> {
        // Single test with 10-second audio
        let test_audio = self.generate_test_audio(10.0, 16000).await?;
        let asr_config = ASRConfig::default();

        let mut timings = Vec::new();
        let mut memory_deltas = Vec::new();

        for _ in 0..5 {
            let initial_memory = Self::current_rss_mb();
            let start = Instant::now();

            // Real work: a full transcription of the generated audio.
            model.transcribe(&test_audio, Some(&asr_config)).await?;

            let duration = start.elapsed();
            if let (Some(before), Some(after)) = (initial_memory, Self::current_rss_mb()) {
                memory_deltas.push(after - before);
            }

            timings.push(duration);
        }

        let rtfs: Vec<f32> = timings
            .iter()
            .map(|d| d.as_secs_f32() / test_audio.duration())
            .collect();

        #[allow(clippy::cast_precision_loss)]
        let average_rtf = rtfs.iter().sum::<f32>() / rtfs.len() as f32;
        // Peak resident memory observed during the run, or 0.0 when this platform
        // does not expose a resident-set size (see `current_rss_mb`).
        let peak_memory_mb = Self::current_rss_mb().unwrap_or(0.0);

        Ok(OverallPerformance {
            average_rtf,
            median_rtf: Self::calculate_median(&rtfs),
            p95_rtf: Self::calculate_percentile(&rtfs, 0.95),
            p99_rtf: Self::calculate_percentile(&rtfs, 0.99),
            average_latency_ms: (timings.iter().sum::<Duration>().as_millis()
                / timings.len() as u128)
                .try_into()
                .unwrap_or(u32::MAX),
            throughput_hours_per_hour: if average_rtf > 0.0 {
                1.0 / average_rtf
            } else {
                0.0
            },
            peak_memory_mb,
            meets_targets: average_rtf <= self.config.performance_targets.max_rtf,
            performance_score: self.calculate_performance_score(average_rtf, peak_memory_mb),
        })
    }

    /// Continuous monitoring mode
    ///
    /// # Errors
    /// Returns `RecognitionError` if benchmark execution fails
    pub async fn start_monitoring<M: ASRModel + ?Sized>(
        &self,
        model: &M,
    ) -> Result<(), RecognitionError> {
        let mut interval = tokio::time::interval(Duration::from_secs(60));

        loop {
            interval.tick().await;

            let quick_result = self.quick_benchmark(model).await?;

            // Check for performance regressions
            if !quick_result.meets_targets {
                println!(
                    "Performance regression detected! RTF: {:.3}, Target: {:.3}",
                    quick_result.average_rtf, self.config.performance_targets.max_rtf
                );
            }

            // Log performance metrics
            Self::log_performance_metrics(&quick_result);
        }
    }

    // Internal implementation methods

    /// Run the model a few times so lazily-initialised buffers are hot before timing.
    async fn warmup_phase<M: ASRModel + ?Sized>(&self, model: &M) -> Result<(), RecognitionError> {
        let test_audio = self.generate_test_audio(5.0, 16000).await?;
        let asr_config = ASRConfig::default();

        for _ in 0..self.config.warmup_iterations {
            model.transcribe(&test_audio, Some(&asr_config)).await?;
        }

        Ok(())
    }

    /// Benchmark the pipeline stages that can be timed through the public trait.
    ///
    /// Only `end_to_end` is measured directly. The per-stage entries stay at their
    /// defaults because [`ASRModel`] exposes no hook to time the audio front end,
    /// encoder, decoder and tokenizer separately; reporting invented per-stage numbers
    /// would be worse than reporting none.
    async fn benchmark_components<M: ASRModel + ?Sized>(
        &self,
        model: &M,
    ) -> Result<ComponentBenchmarks, RecognitionError> {
        let end_to_end = self.benchmark_end_to_end(model).await?;

        Ok(ComponentBenchmarks {
            audio_processing: ComponentPerformance::default(),
            encoder: ComponentPerformance::default(),
            decoder: ComponentPerformance::default(),
            tokenizer: ComponentPerformance::default(),
            end_to_end,
        })
    }

    /// Time real transcriptions across every configured audio duration.
    async fn benchmark_end_to_end<M: ASRModel + ?Sized>(
        &self,
        model: &M,
    ) -> Result<ComponentPerformance, RecognitionError> {
        let mut timings = Vec::new();
        let mut memory_samples = Vec::new();
        let asr_config = ASRConfig::default();

        for duration in &self.config.test_durations {
            let test_audio = self.generate_test_audio(*duration, 16000).await?;

            for _ in 0..self.config.benchmark_iterations {
                let start = Instant::now();
                model.transcribe(&test_audio, Some(&asr_config)).await?;
                let elapsed = start.elapsed();

                #[allow(clippy::cast_precision_loss)]
                timings.push(elapsed.as_secs_f64() as f32 * 1000.0);
                if let Some(rss) = Self::current_rss_mb() {
                    memory_samples.push(rss);
                }
            }
        }

        if timings.is_empty() {
            return Err(RecognitionError::ConfigurationError {
                message: "Benchmark produced no samples: test_durations and \
                          benchmark_iterations must both be non-empty"
                    .to_string(),
            });
        }

        #[allow(clippy::cast_precision_loss)]
        let average_time_ms = timings.iter().sum::<f32>() / timings.len() as f32;
        // 0.0 == not measured on this platform.
        #[allow(clippy::cast_precision_loss)]
        let memory_usage_mb = if memory_samples.is_empty() {
            0.0
        } else {
            memory_samples.iter().sum::<f32>() / memory_samples.len() as f32
        };
        let std_dev_ms = Self::calculate_std_dev(&timings);

        Ok(ComponentPerformance {
            average_time_ms,
            min_time_ms: timings.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            max_time_ms: timings.iter().fold(0.0, |a, &b| a.max(b)),
            std_dev_ms,
            memory_usage_mb,
            iterations: timings.len().try_into().unwrap_or(u32::MAX),
            // Relative dispersion of the measured timings: a stage whose runtime varies
            // a lot is the one worth investigating.
            bottleneck_score: if average_time_ms > 0.0 {
                (std_dev_ms / average_time_ms).min(1.0)
            } else {
                0.0
            },
        })
    }

    /// Measure real latency: time to the first completed transcription, and how
    /// transcription time scales with audio length.
    async fn analyze_latency<M: ASRModel + ?Sized>(
        &self,
        model: &M,
    ) -> Result<LatencyAnalysis, RecognitionError> {
        let asr_config = ASRConfig::default();

        // Shortest configured clip stands in for the minimum-work request.
        let shortest = self
            .config
            .test_durations
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let shortest = if shortest.is_finite() { shortest } else { 1.0 };

        let short_audio = self.generate_test_audio(shortest, 16000).await?;
        let start = Instant::now();
        model.transcribe(&short_audio, Some(&asr_config)).await?;
        #[allow(clippy::cast_precision_loss)]
        let first_token_latency_ms = start.elapsed().as_secs_f64() as f32 * 1000.0;

        // Real measurement of the runtime-vs-length relationship.
        let mut audio_length_impact = Vec::new();
        for duration in &self.config.test_durations {
            let audio = self.generate_test_audio(*duration, 16000).await?;
            let start = Instant::now();
            model.transcribe(&audio, Some(&asr_config)).await?;
            #[allow(clippy::cast_precision_loss)]
            audio_length_impact.push((*duration, start.elapsed().as_secs_f64() as f32 * 1000.0));
        }

        // Per-language latency, measured with the same audio so the difference is the
        // model's language handling rather than the input.
        let mut language_latency_ms = HashMap::new();
        for language in &self.config.test_languages {
            let config = ASRConfig {
                language: Some(*language),
                ..ASRConfig::default()
            };
            let start = Instant::now();
            if model.transcribe(&short_audio, Some(&config)).await.is_ok() {
                #[allow(clippy::cast_precision_loss)]
                language_latency_ms.insert(
                    format!("{language:?}"),
                    start.elapsed().as_secs_f64() as f32 * 1000.0,
                );
            }
        }

        // Batch latency: total wall time for N concurrent-sized workloads run back to
        // back, divided by N.
        let mut batch_latency_ms = HashMap::new();
        for batch_size in &self.config.batch_sizes {
            if *batch_size == 0 {
                continue;
            }
            let start = Instant::now();
            for _ in 0..*batch_size {
                model.transcribe(&short_audio, Some(&asr_config)).await?;
            }
            #[allow(clippy::cast_precision_loss)]
            let per_item = start.elapsed().as_secs_f64() as f32 * 1000.0 / *batch_size as f32;
            batch_latency_ms.insert(*batch_size, per_item);
        }

        Ok(LatencyAnalysis {
            first_token_latency_ms,
            // Streaming latency equals the shortest clip's end-to-end time: this
            // implementation has no partial-result hook to measure separately.
            streaming_latency_ms: first_token_latency_ms,
            batch_latency_ms,
            language_latency_ms,
            audio_length_impact,
        })
    }

    /// Measure real throughput: audio seconds processed per wall-clock second.
    async fn analyze_throughput<M: ASRModel + ?Sized>(
        &self,
        model: &M,
    ) -> Result<ThroughputAnalysis, RecognitionError> {
        let asr_config = ASRConfig::default();
        let audio = self.generate_test_audio(5.0, 16000).await?;

        let start = Instant::now();
        let iterations = self.config.benchmark_iterations.max(1);
        for _ in 0..iterations {
            model.transcribe(&audio, Some(&asr_config)).await?;
        }
        let elapsed = start.elapsed().as_secs_f32();

        #[allow(clippy::cast_precision_loss)]
        let processed_seconds = audio.duration() * iterations as f32;
        let single_stream_throughput = if elapsed > 0.0 {
            processed_seconds / elapsed
        } else {
            0.0
        };

        // Real per-batch-size measurement.
        let mut throughput_vs_batch_size = Vec::new();
        let mut best = (1_usize, 0.0_f32);
        for batch_size in &self.config.batch_sizes {
            if *batch_size == 0 {
                continue;
            }
            let start = Instant::now();
            for _ in 0..*batch_size {
                model.transcribe(&audio, Some(&asr_config)).await?;
            }
            let batch_elapsed = start.elapsed().as_secs_f32();
            #[allow(clippy::cast_precision_loss)]
            let rate = if batch_elapsed > 0.0 {
                audio.duration() * *batch_size as f32 / batch_elapsed
            } else {
                0.0
            };
            throughput_vs_batch_size.push((*batch_size, rate));
            if rate > best.1 {
                best = (*batch_size, rate);
            }
        }

        Ok(ThroughputAnalysis {
            single_stream_throughput,
            // Parallelism is bounded by the real core count of this machine.
            max_parallel_streams: u32::try_from(num_cpus::get()).unwrap_or(u32::MAX),
            optimal_batch_size: best.0,
            throughput_vs_batch_size,
            // CPU/GPU utilisation sampling is not available through this API; 0.0 and
            // None mean "not measured" rather than "idle".
            cpu_utilization: 0.0,
            gpu_utilization: None,
        })
    }

    /// Measure real resident-memory behaviour across repeated transcriptions.
    ///
    /// When the platform exposes no resident-set size, every field is left at `0.0`,
    /// which means "not measured" — no figure is invented.
    async fn analyze_memory<M: ASRModel + ?Sized>(
        &self,
        model: &M,
    ) -> Result<MemoryAnalysis, RecognitionError> {
        let Some(baseline) = Self::current_rss_mb() else {
            tracing::warn!(
                "Resident-set size is unavailable on this platform; memory analysis \
                            reports zeros rather than estimates"
            );
            return Ok(MemoryAnalysis::default());
        };

        let asr_config = ASRConfig::default();
        let audio = self.generate_test_audio(10.0, 16000).await?;
        let mut samples = vec![baseline];

        for _ in 0..self.config.benchmark_iterations.max(1) {
            model.transcribe(&audio, Some(&asr_config)).await?;
            if let Some(rss) = Self::current_rss_mb() {
                samples.push(rss);
            }
        }

        let peak_usage_mb = samples.iter().copied().fold(0.0_f32, f32::max);
        #[allow(clippy::cast_precision_loss)]
        let average_usage_mb = samples.iter().sum::<f32>() / samples.len() as f32;
        let last = samples.last().copied().unwrap_or(baseline);
        let memory_growth_rate = if baseline > 0.0 {
            (last - baseline) / baseline
        } else {
            0.0
        };

        Ok(MemoryAnalysis {
            peak_usage_mb,
            average_usage_mb,
            // Audio seconds processed per MiB of peak resident memory.
            memory_efficiency: if peak_usage_mb > 0.0 {
                #[allow(clippy::cast_precision_loss)]
                {
                    audio.duration() * self.config.benchmark_iterations.max(1) as f32
                        / peak_usage_mb
                }
            } else {
                0.0
            },
            memory_growth_rate,
            // Cache and fragmentation instrumentation is not available here; 0.0 means
            // "not measured".
            cache_hit_rate: 0.0,
            fragmentation_level: 0.0,
        })
    }

    /// Suggestions derived from the measurements that were actually taken.
    async fn generate_optimizations(
        &self,
        component_results: &ComponentBenchmarks,
        memory_results: &MemoryAnalysis,
    ) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();
        let targets = &self.config.performance_targets;

        if memory_results.peak_usage_mb > targets.max_memory_mb {
            suggestions.push(OptimizationSuggestion {
                category: OptimizationCategory::Memory,
                priority: OptimizationPriority::High,
                description: format!(
                    "Peak resident memory was {:.1} MB against a {:.1} MB target; enable tensor \
                     pooling or a smaller model variant",
                    memory_results.peak_usage_mb, targets.max_memory_mb
                ),
                estimated_improvement: "Brings peak usage under the configured target".to_string(),
                implementation_effort: ImplementationEffort::Moderate,
                code_references: vec!["whisper/memory_manager.rs".to_string()],
            });
        }

        if memory_results.memory_growth_rate > 0.25 {
            suggestions.push(OptimizationSuggestion {
                category: OptimizationCategory::Memory,
                priority: OptimizationPriority::High,
                description: format!(
                    "Resident memory grew {:.0}% across the run, which suggests state is \
                     retained between transcriptions",
                    memory_results.memory_growth_rate * 100.0
                ),
                estimated_improvement: "Removes per-request growth".to_string(),
                implementation_effort: ImplementationEffort::Moderate,
                code_references: vec!["whisper/memory_manager.rs".to_string()],
            });
        }

        let e2e = &component_results.end_to_end;
        if e2e.bottleneck_score > 0.3 {
            suggestions.push(OptimizationSuggestion {
                category: OptimizationCategory::Computation,
                priority: OptimizationPriority::Medium,
                description: format!(
                    "End-to-end timing varied by {:.0}% of its mean ({:.1} ms +/- {:.1} ms), so \
                     latency is not predictable",
                    e2e.bottleneck_score * 100.0,
                    e2e.average_time_ms,
                    e2e.std_dev_ms
                ),
                estimated_improvement: "More consistent per-request latency".to_string(),
                implementation_effort: ImplementationEffort::Moderate,
                code_references: vec!["whisper/attention.rs".to_string()],
            });
        }

        suggestions
    }

    /// Overall performance, computed from the measurements already collected.
    fn calculate_overall_performance(&self, results: &BenchmarkResults) -> OverallPerformance {
        let e2e = &results.component_benchmarks.end_to_end;
        let latency = &results.latency_analysis;

        // RTF per measured (audio_seconds, elapsed_ms) pair.
        let rtfs: Vec<f32> = latency
            .audio_length_impact
            .iter()
            .filter(|(seconds, _)| *seconds > 0.0)
            .map(|(seconds, elapsed_ms)| elapsed_ms / 1000.0 / seconds)
            .collect();

        #[allow(clippy::cast_precision_loss)]
        let average_rtf = if rtfs.is_empty() {
            0.0
        } else {
            rtfs.iter().sum::<f32>() / rtfs.len() as f32
        };

        OverallPerformance {
            average_rtf,
            median_rtf: if rtfs.is_empty() {
                0.0
            } else {
                Self::calculate_median(&rtfs)
            },
            p95_rtf: if rtfs.is_empty() {
                0.0
            } else {
                Self::calculate_percentile(&rtfs, 0.95)
            },
            p99_rtf: if rtfs.is_empty() {
                0.0
            } else {
                Self::calculate_percentile(&rtfs, 0.99)
            },
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            average_latency_ms: e2e.average_time_ms.max(0.0) as u32,
            throughput_hours_per_hour: results.throughput_analysis.single_stream_throughput,
            peak_memory_mb: results.memory_analysis.peak_usage_mb,
            meets_targets: average_rtf > 0.0
                && average_rtf <= self.config.performance_targets.max_rtf,
            performance_score: self
                .calculate_performance_score(average_rtf, results.memory_analysis.peak_usage_mb),
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

    /// Real resident-set size of this process in mebibytes.
    ///
    /// Returns `None` on platforms where VoiRS cannot obtain it without linking a C
    /// library, so callers can report "not measured" instead of a made-up figure.
    #[must_use]
    #[cfg(target_os = "linux")]
    pub fn current_rss_mb() -> Option<f32> {
        // /proc/self/statm: total, resident, shared, ... in pages.
        let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
        let resident_pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
        // 4 KiB is the page size on every Linux target VoiRS supports.
        #[allow(clippy::cast_precision_loss)]
        Some(resident_pages as f32 * 4.0 / 1024.0)
    }

    /// Real resident-set size of this process in mebibytes.
    ///
    /// Reads the kernel's own accounting via `ps`, which needs no C bindings.
    #[must_use]
    #[cfg(target_os = "macos")]
    pub fn current_rss_mb() -> Option<f32> {
        // `ps` reports RSS in kibibytes.
        let output = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p"])
            .arg(std::process::id().to_string())
            .output()
            .ok()?;
        let kib: f32 = String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse()
            .ok()?;
        Some(kib / 1024.0)
    }

    /// Real resident-set size of this process in mebibytes.
    ///
    /// Always `None` on this platform: VoiRS reports "not measured" rather than a
    /// made-up figure.
    #[must_use]
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    pub fn current_rss_mb() -> Option<f32> {
        None
    }

    fn calculate_performance_score(&self, rtf: f32, memory_mb: f32) -> f32 {
        let rtf_score = (1.0 - (rtf / self.config.performance_targets.max_rtf).min(1.0)) * 50.0;
        let memory_score =
            (1.0 - (memory_mb / self.config.performance_targets.max_memory_mb).min(1.0)) * 50.0;
        rtf_score + memory_score
    }

    fn log_performance_metrics(performance: &OverallPerformance) {
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
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let len = sorted.len();
        if len.is_multiple_of(2) {
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
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{
        ASRFeature, ASRMetadata, AudioStream, RecognitionResult, Transcript, TranscriptStream,
    };
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A model whose transcription cost is proportional to the audio length, so the
    /// benchmark's measurements have something real to observe.
    struct CountingModel {
        calls: AtomicUsize,
        micros_per_audio_second: u64,
    }

    impl CountingModel {
        fn new(micros_per_audio_second: u64) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                micros_per_audio_second,
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl ASRModel for CountingModel {
        async fn transcribe(
            &self,
            audio: &AudioBuffer,
            _config: Option<&ASRConfig>,
        ) -> RecognitionResult<Transcript> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let micros = (audio.duration() * self.micros_per_audio_second as f32) as u64;
            tokio::time::sleep(Duration::from_micros(micros)).await;
            Ok(Transcript {
                text: format!("{:.1}s", audio.duration()),
                language: LanguageCode::EnUs,
                confidence: 1.0,
                word_timestamps: Vec::new(),
                sentence_boundaries: Vec::new(),
                processing_duration: None,
            })
        }

        async fn transcribe_streaming(
            &self,
            _audio_stream: AudioStream,
            _config: Option<&ASRConfig>,
        ) -> RecognitionResult<TranscriptStream> {
            unimplemented!("not exercised by the benchmark tests")
        }

        fn supported_languages(&self) -> Vec<LanguageCode> {
            vec![LanguageCode::EnUs]
        }

        fn metadata(&self) -> ASRMetadata {
            ASRMetadata {
                name: "CountingModel".to_string(),
                version: "0".to_string(),
                description: "test double".to_string(),
                supported_languages: self.supported_languages(),
                architecture: "test".to_string(),
                model_size_mb: 0.0,
                inference_speed: 0.0,
                wer_benchmarks: HashMap::new(),
                supported_features: Vec::new(),
            }
        }

        fn supports_feature(&self, _feature: ASRFeature) -> bool {
            false
        }
    }

    fn fast_config() -> BenchmarkConfig {
        BenchmarkConfig {
            warmup_iterations: 1,
            benchmark_iterations: 2,
            test_durations: vec![0.5, 1.0],
            test_languages: vec![LanguageCode::EnUs],
            batch_sizes: vec![1, 2],
            ..BenchmarkConfig::default()
        }
    }

    /// Regression test for the `_model: &()` placeholder API: the benchmark must
    /// actually call the model it was given.
    #[tokio::test]
    async fn quick_benchmark_calls_the_real_model() {
        let model = CountingModel::new(200);
        let benchmark = WhisperBenchmark::new(fast_config());

        let result = benchmark.quick_benchmark(&model).await.unwrap();

        assert_eq!(
            model.calls(),
            5,
            "quick_benchmark must run 5 transcriptions"
        );
        assert!(
            result.average_rtf > 0.0,
            "RTF must be measured, not asserted"
        );
        assert!(result.average_rtf.is_finite());
        assert!(result.average_latency_ms > 0);
    }

    /// A slower model must produce a measurably larger RTF: the numbers track the
    /// model rather than being constants.
    #[tokio::test]
    async fn measured_rtf_tracks_model_speed() {
        let benchmark = WhisperBenchmark::new(fast_config());

        let fast = benchmark
            .quick_benchmark(&CountingModel::new(500))
            .await
            .unwrap();
        let slow = benchmark
            .quick_benchmark(&CountingModel::new(20_000))
            .await
            .unwrap();

        assert!(
            slow.average_rtf > fast.average_rtf * 2.0,
            "slow model RTF {} was not clearly above fast model RTF {}",
            slow.average_rtf,
            fast.average_rtf
        );
    }

    /// The full benchmark must exercise the model across every configured stage.
    #[tokio::test]
    async fn full_benchmark_measures_every_stage() {
        // 20 ms of work per audio-second keeps every measurement well above the
        // ~1 ms resolution of the async timer, so the length/time relationship is
        // observable rather than lost in timer noise.
        let model = CountingModel::new(20_000);
        let benchmark = WhisperBenchmark::new(fast_config());

        let results = benchmark.run_full_benchmark(&model).await.unwrap();

        assert!(model.calls() > 10, "only {} calls made", model.calls());

        let e2e = &results.component_benchmarks.end_to_end;
        assert_eq!(e2e.iterations, 4, "2 durations x 2 iterations");
        assert!(e2e.average_time_ms > 0.0);
        assert!(e2e.max_time_ms >= e2e.min_time_ms);

        // Latency vs audio length must have one real sample per configured duration.
        assert_eq!(results.latency_analysis.audio_length_impact.len(), 2);
        let (short_secs, short_ms) = results.latency_analysis.audio_length_impact[0];
        let (long_secs, long_ms) = results.latency_analysis.audio_length_impact[1];
        assert!(short_secs < long_secs);
        assert!(
            long_ms > short_ms,
            "1.0s clip ({long_ms} ms) should take longer than 0.5s clip ({short_ms} ms)"
        );

        // Throughput must be a real audio-seconds-per-second figure.
        assert!(results.throughput_analysis.single_stream_throughput > 0.0);
        assert_eq!(
            results.throughput_analysis.throughput_vs_batch_size.len(),
            2
        );
        assert!(results.throughput_analysis.max_parallel_streams >= 1);

        // Overall RTF must be derived from the measured pairs above.
        assert!(results.overall_performance.average_rtf > 0.0);
        assert!(results.overall_performance.average_rtf.is_finite());
    }

    /// Resident-set size must either be a real, plausible figure or an honest `None` —
    /// never the old hardcoded 1024.0.
    #[test]
    fn rss_is_real_or_absent() {
        match WhisperBenchmark::current_rss_mb() {
            Some(mb) => {
                assert!(mb > 0.0, "RSS must be positive, got {mb}");
                assert!(mb < 1_000_000.0, "implausible RSS {mb} MB");
                assert!(
                    (mb - 1024.0).abs() > f32::EPSILON,
                    "RSS is suspiciously exactly the old placeholder value"
                );
            }
            None => {
                // Acceptable only on platforms VoiRS cannot measure without C bindings.
                #[cfg(any(target_os = "linux", target_os = "macos"))]
                panic!("RSS should be measurable on this platform");
            }
        }
    }

    /// Memory analysis must report zeros (meaning "not measured") rather than the old
    /// 2048.0 / 1536.0 / 0.85 constants.
    #[tokio::test]
    async fn memory_analysis_is_measured_not_asserted() {
        let benchmark = WhisperBenchmark::new(fast_config());
        let analysis = benchmark
            .analyze_memory(&CountingModel::new(100))
            .await
            .unwrap();

        assert_eq!(
            analysis.cache_hit_rate, 0.0,
            "cache hit rate must not be invented"
        );
        assert_eq!(analysis.fragmentation_level, 0.0);
        if WhisperBenchmark::current_rss_mb().is_some() {
            assert!(analysis.peak_usage_mb > 0.0);
            assert!(analysis.average_usage_mb > 0.0);
            assert!(analysis.peak_usage_mb >= analysis.average_usage_mb);
            assert!(
                (analysis.peak_usage_mb - 2048.0).abs() > f32::EPSILON,
                "peak memory is suspiciously the old placeholder value"
            );
        } else {
            assert_eq!(analysis.peak_usage_mb, 0.0);
        }
    }

    /// Optimisation suggestions must follow from the measurements, so a model that is
    /// well within its targets produces no memory complaint.
    #[tokio::test]
    async fn suggestions_follow_from_measurements() {
        let mut config = fast_config();
        config.performance_targets.max_memory_mb = f32::MAX;
        let benchmark = WhisperBenchmark::new(config);

        let components = ComponentBenchmarks::default();
        let memory = MemoryAnalysis {
            peak_usage_mb: 10.0,
            memory_growth_rate: 0.0,
            ..MemoryAnalysis::default()
        };
        let suggestions = benchmark.generate_optimizations(&components, &memory).await;
        assert!(
            suggestions.is_empty(),
            "no suggestion should fire when everything is within target: {suggestions:?}"
        );

        let over_budget = MemoryAnalysis {
            peak_usage_mb: 10.0,
            memory_growth_rate: 2.0,
            ..MemoryAnalysis::default()
        };
        let suggestions = benchmark
            .generate_optimizations(&components, &over_budget)
            .await;
        assert_eq!(suggestions.len(), 1);
        assert!(suggestions[0].description.contains("grew"));
    }
}
