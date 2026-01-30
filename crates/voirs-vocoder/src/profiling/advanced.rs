//! Advanced Performance Profiling System
//!
//! Provides comprehensive profiling capabilities for production environments,
//! including detailed latency tracking, throughput analysis, and performance
//! regression detection.

use crate::{AudioBuffer, MelSpectrogram, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Advanced performance profiler with detailed metrics tracking
pub struct AdvancedProfiler {
    /// Configuration for profiling
    config: ProfilerConfig,
    /// Accumulated metrics
    metrics: Arc<RwLock<ProfilerMetrics>>,
    /// Historical performance data for regression detection
    history: Arc<RwLock<PerformanceHistory>>,
}

/// Configuration for the profiler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Enable detailed latency breakdown
    pub detailed_latency: bool,
    /// Enable memory usage tracking
    pub track_memory: bool,
    /// Enable throughput analysis
    pub track_throughput: bool,
    /// Enable regression detection
    pub detect_regressions: bool,
    /// Maximum history size for regression detection
    pub max_history_size: usize,
    /// Sampling rate for profiling (1 = all operations, 10 = every 10th, etc.)
    pub sampling_rate: usize,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            detailed_latency: true,
            track_memory: true,
            track_throughput: true,
            detect_regressions: true,
            max_history_size: 1000,
            sampling_rate: 1,
        }
    }
}

/// Comprehensive profiling metrics
#[derive(Debug, Clone, Default)]
pub struct ProfilerMetrics {
    /// Total number of operations profiled
    pub total_operations: u64,
    /// Total processing time
    pub total_time: Duration,
    /// Minimum latency observed
    pub min_latency: Option<Duration>,
    /// Maximum latency observed
    pub max_latency: Option<Duration>,
    /// Average latency
    pub avg_latency: Duration,
    /// 50th percentile (median)
    pub p50_latency: Duration,
    /// 95th percentile
    pub p95_latency: Duration,
    /// 99th percentile
    pub p99_latency: Duration,
    /// Real-time factor statistics
    pub rtf_stats: RtfStatistics,
    /// Throughput statistics
    pub throughput_stats: ThroughputStatistics,
    /// Memory usage statistics (if enabled)
    pub memory_stats: Option<MemoryStatistics>,
    /// Latency breakdown by stage
    pub latency_breakdown: LatencyBreakdown,
}

/// Real-time factor statistics
#[derive(Debug, Clone, Default)]
pub struct RtfStatistics {
    /// Minimum RTF observed
    pub min_rtf: f32,
    /// Maximum RTF observed
    pub max_rtf: f32,
    /// Average RTF
    pub avg_rtf: f32,
    /// Percentage of operations meeting real-time constraints
    pub realtime_percentage: f32,
}

/// Throughput statistics
#[derive(Debug, Clone, Default)]
pub struct ThroughputStatistics {
    /// Total audio seconds processed
    pub total_audio_seconds: f32,
    /// Total mel frames processed
    pub total_frames: u64,
    /// Average throughput in frames per second
    pub avg_frames_per_second: f32,
    /// Average throughput in audio seconds per wall-clock second
    pub avg_audio_seconds_per_second: f32,
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStatistics {
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Average memory usage in bytes
    pub avg_memory_bytes: usize,
    /// Current memory usage in bytes
    pub current_memory_bytes: usize,
}

/// Latency breakdown by processing stage
#[derive(Debug, Clone, Default)]
pub struct LatencyBreakdown {
    /// Time spent in preprocessing
    pub preprocessing: Duration,
    /// Time spent in model inference
    pub inference: Duration,
    /// Time spent in postprocessing
    pub postprocessing: Duration,
    /// Time spent in I/O operations
    pub io_operations: Duration,
    /// Time spent waiting (synchronization, etc.)
    pub waiting: Duration,
}

/// Performance history for regression detection
#[derive(Debug, Clone, Default)]
struct PerformanceHistory {
    /// Recent latency measurements
    latencies: VecDeque<Duration>,
    /// Recent RTF measurements
    rtfs: VecDeque<f32>,
    /// Recent throughput measurements
    throughputs: VecDeque<f32>,
    /// Baseline performance metrics
    baseline: Option<BaselineMetrics>,
}

/// Baseline performance metrics for comparison
#[derive(Debug, Clone)]
struct BaselineMetrics {
    /// Baseline average latency
    avg_latency: Duration,
    /// Baseline average RTF
    avg_rtf: f32,
    /// Baseline average throughput
    avg_throughput: f32,
}

impl AdvancedProfiler {
    /// Create a new profiler with default configuration
    pub fn new() -> Self {
        Self::with_config(ProfilerConfig::default())
    }

    /// Create a new profiler with custom configuration
    pub fn with_config(config: ProfilerConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(RwLock::new(ProfilerMetrics::default())),
            history: Arc::new(RwLock::new(PerformanceHistory::default())),
        }
    }

    /// Start profiling a vocoding operation
    pub fn start_operation(&self) -> ProfilingSession {
        ProfilingSession {
            start_time: Instant::now(),
            profiler: self.metrics.clone(),
            history: self.history.clone(),
            config: self.config.clone(),
            stage_times: LatencyBreakdown::default(),
            current_stage_start: None,
        }
    }

    /// Get current metrics snapshot
    pub fn get_metrics(&self) -> ProfilerMetrics {
        self.metrics.read().clone()
    }

    /// Reset all metrics
    pub fn reset(&self) {
        *self.metrics.write() = ProfilerMetrics::default();
        *self.history.write() = PerformanceHistory::default();
    }

    /// Set performance baseline for regression detection
    pub fn set_baseline(&self) {
        let metrics = self.metrics.read();
        let mut history = self.history.write();

        history.baseline = Some(BaselineMetrics {
            avg_latency: metrics.avg_latency,
            avg_rtf: metrics.rtf_stats.avg_rtf,
            avg_throughput: metrics.throughput_stats.avg_frames_per_second,
        });
    }

    /// Check for performance regressions
    pub fn detect_regression(&self, threshold_percent: f32) -> Option<RegressionReport> {
        let metrics = self.metrics.read();
        let history = self.history.read();

        let baseline = history.baseline.as_ref()?;

        let mut regressions = Vec::new();

        // Check latency regression
        let latency_increase = (metrics.avg_latency.as_secs_f32()
            - baseline.avg_latency.as_secs_f32())
            / baseline.avg_latency.as_secs_f32()
            * 100.0;

        if latency_increase > threshold_percent {
            regressions.push(RegressionType::Latency {
                baseline: baseline.avg_latency,
                current: metrics.avg_latency,
                increase_percent: latency_increase,
            });
        }

        // Check RTF regression
        let rtf_increase =
            (metrics.rtf_stats.avg_rtf - baseline.avg_rtf) / baseline.avg_rtf * 100.0;

        if rtf_increase > threshold_percent {
            regressions.push(RegressionType::Rtf {
                baseline: baseline.avg_rtf,
                current: metrics.rtf_stats.avg_rtf,
                increase_percent: rtf_increase,
            });
        }

        // Check throughput regression
        let throughput_decrease = (baseline.avg_throughput
            - metrics.throughput_stats.avg_frames_per_second)
            / baseline.avg_throughput
            * 100.0;

        if throughput_decrease > threshold_percent {
            regressions.push(RegressionType::Throughput {
                baseline: baseline.avg_throughput,
                current: metrics.throughput_stats.avg_frames_per_second,
                decrease_percent: throughput_decrease,
            });
        }

        if !regressions.is_empty() {
            Some(RegressionReport {
                detected_at: Instant::now(),
                regressions,
                threshold_percent,
            })
        } else {
            None
        }
    }

    /// Generate a comprehensive performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let metrics = self.metrics.read();

        PerformanceReport {
            total_operations: metrics.total_operations,
            latency_summary: LatencySummary {
                min: metrics.min_latency.unwrap_or(Duration::ZERO),
                max: metrics.max_latency.unwrap_or(Duration::ZERO),
                avg: metrics.avg_latency,
                p50: metrics.p50_latency,
                p95: metrics.p95_latency,
                p99: metrics.p99_latency,
            },
            rtf_summary: metrics.rtf_stats.clone(),
            throughput_summary: metrics.throughput_stats.clone(),
            memory_summary: metrics.memory_stats.clone(),
            latency_breakdown: metrics.latency_breakdown.clone(),
        }
    }
}

impl Default for AdvancedProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Active profiling session for a single operation
pub struct ProfilingSession {
    start_time: Instant,
    profiler: Arc<RwLock<ProfilerMetrics>>,
    history: Arc<RwLock<PerformanceHistory>>,
    config: ProfilerConfig,
    stage_times: LatencyBreakdown,
    current_stage_start: Option<Instant>,
}

impl ProfilingSession {
    /// Start timing a specific processing stage
    pub fn start_stage(&mut self, _stage: ProcessingStage) {
        self.current_stage_start = Some(Instant::now());
    }

    /// End timing a specific processing stage
    pub fn end_stage(&mut self, stage: ProcessingStage) {
        if let Some(stage_start) = self.current_stage_start.take() {
            let duration = stage_start.elapsed();
            match stage {
                ProcessingStage::Preprocessing => self.stage_times.preprocessing += duration,
                ProcessingStage::Inference => self.stage_times.inference += duration,
                ProcessingStage::Postprocessing => self.stage_times.postprocessing += duration,
                ProcessingStage::IoOperations => self.stage_times.io_operations += duration,
                ProcessingStage::Waiting => self.stage_times.waiting += duration,
            }
        }
    }

    /// Complete the profiling session
    pub fn complete(self, mel: &MelSpectrogram, audio: &AudioBuffer) {
        let total_latency = self.start_time.elapsed();
        let audio_duration = audio.duration();
        let rtf = if audio_duration > 0.0 {
            total_latency.as_secs_f32() / audio_duration
        } else {
            0.0
        };

        let frame_count = mel.n_frames as u64;

        // Update metrics
        let mut metrics = self.profiler.write();
        metrics.total_operations += 1;
        metrics.total_time += total_latency;

        // Update latency statistics
        if metrics.min_latency.is_none() || total_latency < metrics.min_latency.unwrap() {
            metrics.min_latency = Some(total_latency);
        }
        if metrics.max_latency.is_none() || total_latency > metrics.max_latency.unwrap() {
            metrics.max_latency = Some(total_latency);
        }

        // Update average latency
        metrics.avg_latency = metrics.total_time / metrics.total_operations as u32;

        // Update RTF statistics
        if metrics.rtf_stats.min_rtf == 0.0 || rtf < metrics.rtf_stats.min_rtf {
            metrics.rtf_stats.min_rtf = rtf;
        }
        if rtf > metrics.rtf_stats.max_rtf {
            metrics.rtf_stats.max_rtf = rtf;
        }
        metrics.rtf_stats.avg_rtf =
            ((metrics.rtf_stats.avg_rtf * (metrics.total_operations - 1) as f32) + rtf)
                / metrics.total_operations as f32;

        if rtf < 1.0 {
            let realtime_count = (metrics.rtf_stats.realtime_percentage
                * (metrics.total_operations - 1) as f32
                / 100.0)
                + 1.0;
            metrics.rtf_stats.realtime_percentage =
                (realtime_count / metrics.total_operations as f32) * 100.0;
        }

        // Update throughput statistics
        metrics.throughput_stats.total_frames += frame_count;
        metrics.throughput_stats.total_audio_seconds += audio_duration;

        if metrics.total_time.as_secs_f32() > 0.0 {
            metrics.throughput_stats.avg_frames_per_second =
                metrics.throughput_stats.total_frames as f32 / metrics.total_time.as_secs_f32();
            metrics.throughput_stats.avg_audio_seconds_per_second =
                metrics.throughput_stats.total_audio_seconds / metrics.total_time.as_secs_f32();
        }

        // Update latency breakdown
        if self.config.detailed_latency {
            metrics.latency_breakdown.preprocessing = Duration::from_secs_f32(
                (metrics.latency_breakdown.preprocessing.as_secs_f32()
                    * (metrics.total_operations - 1) as f32
                    + self.stage_times.preprocessing.as_secs_f32())
                    / metrics.total_operations as f32,
            );
            metrics.latency_breakdown.inference = Duration::from_secs_f32(
                (metrics.latency_breakdown.inference.as_secs_f32()
                    * (metrics.total_operations - 1) as f32
                    + self.stage_times.inference.as_secs_f32())
                    / metrics.total_operations as f32,
            );
            metrics.latency_breakdown.postprocessing = Duration::from_secs_f32(
                (metrics.latency_breakdown.postprocessing.as_secs_f32()
                    * (metrics.total_operations - 1) as f32
                    + self.stage_times.postprocessing.as_secs_f32())
                    / metrics.total_operations as f32,
            );
        }

        drop(metrics);

        // Update history for regression detection
        if self.config.detect_regressions {
            let mut history = self.history.write();

            if history.latencies.len() >= self.config.max_history_size {
                history.latencies.pop_front();
            }
            history.latencies.push_back(total_latency);

            if history.rtfs.len() >= self.config.max_history_size {
                history.rtfs.pop_front();
            }
            history.rtfs.push_back(rtf);

            let throughput = if total_latency.as_secs_f32() > 0.0 {
                frame_count as f32 / total_latency.as_secs_f32()
            } else {
                0.0
            };

            if history.throughputs.len() >= self.config.max_history_size {
                history.throughputs.pop_front();
            }
            history.throughputs.push_back(throughput);
        }
    }
}

/// Processing stages for detailed profiling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessingStage {
    /// Preprocessing stage (mel normalization, etc.)
    Preprocessing,
    /// Model inference stage
    Inference,
    /// Postprocessing stage (audio effects, normalization, etc.)
    Postprocessing,
    /// I/O operations (file reading/writing, etc.)
    IoOperations,
    /// Waiting time (locks, synchronization, etc.)
    Waiting,
}

/// Type of performance regression detected
#[derive(Debug, Clone)]
pub enum RegressionType {
    /// Latency regression
    Latency {
        baseline: Duration,
        current: Duration,
        increase_percent: f32,
    },
    /// RTF regression
    Rtf {
        baseline: f32,
        current: f32,
        increase_percent: f32,
    },
    /// Throughput regression
    Throughput {
        baseline: f32,
        current: f32,
        decrease_percent: f32,
    },
}

/// Regression detection report
#[derive(Debug, Clone)]
pub struct RegressionReport {
    /// When the regression was detected
    pub detected_at: Instant,
    /// List of detected regressions
    pub regressions: Vec<RegressionType>,
    /// Threshold used for detection
    pub threshold_percent: f32,
}

/// Comprehensive performance report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Total number of operations profiled
    pub total_operations: u64,
    /// Latency summary
    pub latency_summary: LatencySummary,
    /// RTF summary
    pub rtf_summary: RtfStatistics,
    /// Throughput summary
    pub throughput_summary: ThroughputStatistics,
    /// Memory summary (if available)
    pub memory_summary: Option<MemoryStatistics>,
    /// Latency breakdown by stage
    pub latency_breakdown: LatencyBreakdown,
}

/// Latency summary statistics
#[derive(Debug, Clone)]
pub struct LatencySummary {
    /// Minimum latency
    pub min: Duration,
    /// Maximum latency
    pub max: Duration,
    /// Average latency
    pub avg: Duration,
    /// 50th percentile
    pub p50: Duration,
    /// 95th percentile
    pub p95: Duration,
    /// 99th percentile
    pub p99: Duration,
}

impl std::fmt::Display for PerformanceReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Performance Report")?;
        writeln!(f, "==================")?;
        writeln!(f, "Total Operations: {}", self.total_operations)?;
        writeln!(f)?;
        writeln!(f, "Latency Statistics:")?;
        writeln!(
            f,
            "  Min: {:.2} ms",
            self.latency_summary.min.as_secs_f32() * 1000.0
        )?;
        writeln!(
            f,
            "  Max: {:.2} ms",
            self.latency_summary.max.as_secs_f32() * 1000.0
        )?;
        writeln!(
            f,
            "  Avg: {:.2} ms",
            self.latency_summary.avg.as_secs_f32() * 1000.0
        )?;
        writeln!(
            f,
            "  P50: {:.2} ms",
            self.latency_summary.p50.as_secs_f32() * 1000.0
        )?;
        writeln!(
            f,
            "  P95: {:.2} ms",
            self.latency_summary.p95.as_secs_f32() * 1000.0
        )?;
        writeln!(
            f,
            "  P99: {:.2} ms",
            self.latency_summary.p99.as_secs_f32() * 1000.0
        )?;
        writeln!(f)?;
        writeln!(f, "RTF Statistics:")?;
        writeln!(f, "  Min: {:.3}x", self.rtf_summary.min_rtf)?;
        writeln!(f, "  Max: {:.3}x", self.rtf_summary.max_rtf)?;
        writeln!(f, "  Avg: {:.3}x", self.rtf_summary.avg_rtf)?;
        writeln!(
            f,
            "  Real-time %: {:.1}%",
            self.rtf_summary.realtime_percentage
        )?;
        writeln!(f)?;
        writeln!(f, "Throughput Statistics:")?;
        writeln!(
            f,
            "  Total Frames: {}",
            self.throughput_summary.total_frames
        )?;
        writeln!(
            f,
            "  Total Audio: {:.1} s",
            self.throughput_summary.total_audio_seconds
        )?;
        writeln!(
            f,
            "  Avg Frames/s: {:.1}",
            self.throughput_summary.avg_frames_per_second
        )?;
        writeln!(
            f,
            "  Avg Audio/s: {:.2}x",
            self.throughput_summary.avg_audio_seconds_per_second
        )?;
        writeln!(f)?;
        writeln!(f, "Latency Breakdown:")?;
        writeln!(
            f,
            "  Preprocessing: {:.2} ms",
            self.latency_breakdown.preprocessing.as_secs_f32() * 1000.0
        )?;
        writeln!(
            f,
            "  Inference: {:.2} ms",
            self.latency_breakdown.inference.as_secs_f32() * 1000.0
        )?;
        writeln!(
            f,
            "  Postprocessing: {:.2} ms",
            self.latency_breakdown.postprocessing.as_secs_f32() * 1000.0
        )?;

        if let Some(ref memory) = self.memory_summary {
            writeln!(f)?;
            writeln!(f, "Memory Statistics:")?;
            writeln!(
                f,
                "  Peak: {:.2} MB",
                memory.peak_memory_bytes as f32 / 1024.0 / 1024.0
            )?;
            writeln!(
                f,
                "  Avg: {:.2} MB",
                memory.avg_memory_bytes as f32 / 1024.0 / 1024.0
            )?;
            writeln!(
                f,
                "  Current: {:.2} MB",
                memory.current_memory_bytes as f32 / 1024.0 / 1024.0
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioBuffer;

    #[test]
    fn test_profiler_creation() {
        let profiler = AdvancedProfiler::new();
        let metrics = profiler.get_metrics();
        assert_eq!(metrics.total_operations, 0);
    }

    #[test]
    fn test_profiling_session() {
        let profiler = AdvancedProfiler::new();
        let mel = MelSpectrogram::new(vec![vec![0.0; 100]; 80], 22050, 256);
        let audio = AudioBuffer::from_samples(vec![0.0; 22050], 22050.0);

        let session = profiler.start_operation();
        session.complete(&mel, &audio);

        let metrics = profiler.get_metrics();
        assert_eq!(metrics.total_operations, 1);
    }

    #[test]
    fn test_stage_timing() {
        let profiler = AdvancedProfiler::new();
        let mel = MelSpectrogram::new(vec![vec![0.0; 100]; 80], 22050, 256);
        let audio = AudioBuffer::from_samples(vec![0.0; 22050], 22050.0);

        let mut session = profiler.start_operation();
        session.start_stage(ProcessingStage::Preprocessing);
        std::thread::sleep(Duration::from_millis(1));
        session.end_stage(ProcessingStage::Preprocessing);

        session.complete(&mel, &audio);

        let metrics = profiler.get_metrics();
        assert!(metrics.latency_breakdown.preprocessing.as_millis() >= 1);
    }

    #[test]
    fn test_regression_detection() {
        let profiler = AdvancedProfiler::new();
        let mel = MelSpectrogram::new(vec![vec![0.0; 100]; 80], 22050, 256);
        let audio = AudioBuffer::from_samples(vec![0.0; 22050], 22050.0);

        // Establish baseline
        for _ in 0..10 {
            let session = profiler.start_operation();
            session.complete(&mel, &audio);
        }

        profiler.set_baseline();

        // Should not detect regression with normal performance
        assert!(profiler.detect_regression(10.0).is_none());
    }

    #[test]
    fn test_performance_report() {
        let profiler = AdvancedProfiler::new();
        let mel = MelSpectrogram::new(vec![vec![0.0; 100]; 80], 22050, 256);
        let audio = AudioBuffer::from_samples(vec![0.0; 22050], 22050.0);

        let session = profiler.start_operation();
        session.complete(&mel, &audio);

        let report = profiler.generate_report();
        assert_eq!(report.total_operations, 1);

        // Test Display implementation
        let report_str = format!("{}", report);
        assert!(report_str.contains("Performance Report"));
        assert!(report_str.contains("Latency Statistics"));
    }
}
