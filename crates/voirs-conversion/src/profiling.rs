//! Comprehensive Performance Profiling and Bottleneck Analysis System
//!
//! This module provides detailed performance profiling capabilities for the VoiRS conversion system,
//! including bottleneck identification, memory analysis, and optimization recommendations.

use crate::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Comprehensive performance profiler for voice conversion operations
#[derive(Debug)]
pub struct ConversionProfiler {
    /// Profiling sessions with detailed timing information
    sessions: Arc<RwLock<HashMap<String, ProfilingSession>>>,
    /// Global performance metrics
    global_metrics: Arc<RwLock<GlobalMetrics>>,
    /// Configuration for profiling behavior
    config: ProfilingConfig,
    /// Bottleneck analyzer for identifying performance issues
    bottleneck_analyzer: BottleneckAnalyzer,
}

/// Configuration for profiling behavior
#[derive(Debug, Clone)]
pub struct ProfilingConfig {
    /// Maximum number of sessions to keep in memory
    pub max_sessions: usize,
    /// Enable detailed memory tracking
    pub enable_memory_tracking: bool,
    /// Enable CPU usage tracking
    pub enable_cpu_tracking: bool,
    /// Enable bottleneck analysis
    pub enable_bottleneck_analysis: bool,
    /// Sampling interval for continuous monitoring
    pub sampling_interval: Duration,
    /// Maximum number of detailed samples per session
    pub max_samples_per_session: usize,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            max_sessions: 100,
            enable_memory_tracking: true,
            enable_cpu_tracking: true,
            enable_bottleneck_analysis: true,
            sampling_interval: Duration::from_millis(10),
            max_samples_per_session: 1000,
        }
    }
}

/// Detailed profiling session for a single conversion operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingSession {
    /// Unique session identifier
    pub session_id: String,
    /// Session start time
    pub start_time: SystemTime,
    /// Session end time (if completed)
    pub end_time: Option<SystemTime>,
    /// Conversion type being profiled
    pub conversion_type: ConversionType,
    /// Audio characteristics
    pub audio_info: AudioInfo,
    /// Detailed timing measurements
    pub timing_data: TimingData,
    /// Memory usage throughout the session
    pub memory_data: MemoryData,
    /// CPU usage information
    pub cpu_data: CpuData,
    /// Identified bottlenecks
    pub bottlenecks: Vec<BottleneckInfo>,
    /// Performance score (0.0 = worst, 1.0 = best)
    pub performance_score: f64,
}

/// Audio characteristics for profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioInfo {
    /// Audio length in samples
    pub length_samples: usize,
    /// Sample rate
    pub sample_rate: f32,
    /// Number of channels
    pub channels: usize,
    /// Audio duration in seconds
    pub duration_seconds: f64,
    /// Peak amplitude
    pub peak_amplitude: f32,
    /// RMS level
    pub rms_level: f32,
}

/// Detailed timing measurements for different stages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingData {
    /// Total processing time
    pub total_duration: Duration,
    /// Time spent in preprocessing
    pub preprocessing_time: Duration,
    /// Time spent in core conversion
    pub conversion_time: Duration,
    /// Time spent in postprocessing
    pub postprocessing_time: Duration,
    /// Time spent in model loading/initialization
    pub model_init_time: Duration,
    /// Time spent in quality assessment
    pub quality_assessment_time: Duration,
    /// Detailed stage timings
    pub stage_timings: BTreeMap<String, StageTimingInfo>,
}

/// Timing information for a specific processing stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageTimingInfo {
    /// Stage name
    pub name: String,
    /// Time spent in this stage
    pub duration: Duration,
    /// Number of times this stage was executed
    pub execution_count: usize,
    /// Average time per execution
    pub average_duration: Duration,
    /// Minimum execution time
    pub min_duration: Duration,
    /// Maximum execution time
    pub max_duration: Duration,
}

/// Memory usage data throughout the conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryData {
    /// Peak memory usage in bytes
    pub peak_memory: usize,
    /// Memory usage at start
    pub initial_memory: usize,
    /// Memory usage at end
    pub final_memory: usize,
    /// Number of allocations
    pub allocation_count: usize,
    /// Number of deallocations
    pub deallocation_count: usize,
    /// Total allocated bytes
    pub total_allocated: usize,
    /// Memory usage samples over time
    pub memory_samples: VecDeque<MemorySample>,
}

/// Single memory usage sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySample {
    /// Timestamp of the sample
    pub timestamp: SystemTime,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Active allocations count
    pub active_allocations: usize,
}

/// CPU usage data during conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuData {
    /// Average CPU usage percentage
    pub average_cpu_usage: f64,
    /// Peak CPU usage percentage
    pub peak_cpu_usage: f64,
    /// CPU usage samples over time
    pub cpu_samples: VecDeque<CpuSample>,
    /// Number of threads used
    pub thread_count: usize,
}

/// Single CPU usage sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuSample {
    /// Timestamp of the sample
    pub timestamp: SystemTime,
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage at this time
    pub memory_usage: usize,
}

/// Global performance metrics across all sessions
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct GlobalMetrics {
    /// Total number of conversions profiled
    pub total_conversions: usize,
    /// Average processing time across all conversions
    pub average_processing_time: Duration,
    /// Average memory usage
    pub average_memory_usage: usize,
    /// Average CPU usage
    pub average_cpu_usage: f64,
    /// Performance metrics by conversion type
    pub metrics_by_type: HashMap<ConversionType, ConversionTypeMetrics>,
    /// Historical performance trends
    pub performance_trends: VecDeque<TrendDataPoint>,
}

/// Performance metrics for a specific conversion type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionTypeMetrics {
    /// Number of conversions of this type
    pub conversion_count: usize,
    /// Average processing time
    pub average_time: Duration,
    /// Average memory usage
    pub average_memory: usize,
    /// Average performance score
    pub average_score: f64,
    /// Most common bottlenecks
    pub common_bottlenecks: HashMap<String, usize>,
}

/// Performance trend data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendDataPoint {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Average processing time at this time
    pub processing_time: Duration,
    /// Average memory usage
    pub memory_usage: usize,
    /// Number of conversions in this time window
    pub conversion_count: usize,
}

impl Default for ConversionProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversionProfiler {
    /// Create a new profiler with default configuration
    pub fn new() -> Self {
        Self::with_config(ProfilingConfig::default())
    }

    /// Create a new profiler with custom configuration
    pub fn with_config(config: ProfilingConfig) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            global_metrics: Arc::new(RwLock::new(GlobalMetrics::default())),
            bottleneck_analyzer: BottleneckAnalyzer::new(),
            config,
        }
    }

    /// Start a new profiling session
    pub fn start_session(&self, conversion_type: ConversionType, audio_info: AudioInfo) -> String {
        let session_id = format!(
            "session_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );

        let session = ProfilingSession {
            session_id: session_id.clone(),
            start_time: SystemTime::now(),
            end_time: None,
            conversion_type,
            audio_info,
            timing_data: TimingData::new(),
            memory_data: MemoryData::new(),
            cpu_data: CpuData::new(),
            bottlenecks: Vec::new(),
            performance_score: 0.0,
        };

        let mut sessions = self.sessions.write().unwrap();
        sessions.insert(session_id.clone(), session);

        // Limit session count
        if sessions.len() > self.config.max_sessions {
            let oldest_session = sessions
                .keys()
                .min_by_key(|&k| sessions[k].start_time)
                .cloned()
                .unwrap();
            sessions.remove(&oldest_session);
        }

        session_id
    }

    /// End a profiling session and perform analysis
    pub fn end_session(&self, session_id: &str) -> Result<ProfilingReport> {
        let mut sessions = self.sessions.write().unwrap();

        if let Some(session) = sessions.get_mut(session_id) {
            session.end_time = Some(SystemTime::now());

            // Perform bottleneck analysis
            if self.config.enable_bottleneck_analysis {
                session.bottlenecks = self.bottleneck_analyzer.analyze_session(session);
            }

            // Calculate performance score
            session.performance_score = self.calculate_performance_score(session);

            // Update global metrics
            self.update_global_metrics(session);

            // Generate report
            let report = ProfilingReport::from_session(session);
            Ok(report)
        } else {
            Err(Error::processing(format!("Session {session_id} not found")))
        }
    }

    /// Record timing for a specific stage
    pub fn record_stage_timing(&self, session_id: &str, stage_name: &str, duration: Duration) {
        let mut sessions = self.sessions.write().unwrap();

        if let Some(session) = sessions.get_mut(session_id) {
            let stage_info = session
                .timing_data
                .stage_timings
                .entry(stage_name.to_string())
                .or_insert_with(|| StageTimingInfo::new(stage_name));

            stage_info.update_timing(duration);
        }
    }

    /// Record memory usage sample
    pub fn record_memory_sample(
        &self,
        session_id: &str,
        memory_usage: usize,
        active_allocations: usize,
    ) {
        if !self.config.enable_memory_tracking {
            return;
        }

        let mut sessions = self.sessions.write().unwrap();

        if let Some(session) = sessions.get_mut(session_id) {
            let sample = MemorySample {
                timestamp: SystemTime::now(),
                memory_usage,
                active_allocations,
            };

            session.memory_data.memory_samples.push_back(sample);

            // Update peak memory
            if memory_usage > session.memory_data.peak_memory {
                session.memory_data.peak_memory = memory_usage;
            }

            // Limit sample count
            if session.memory_data.memory_samples.len() > self.config.max_samples_per_session {
                session.memory_data.memory_samples.pop_front();
            }
        }
    }

    /// Record CPU usage sample
    pub fn record_cpu_sample(&self, session_id: &str, cpu_usage: f64, memory_usage: usize) {
        if !self.config.enable_cpu_tracking {
            return;
        }

        let mut sessions = self.sessions.write().unwrap();

        if let Some(session) = sessions.get_mut(session_id) {
            let sample = CpuSample {
                timestamp: SystemTime::now(),
                cpu_usage,
                memory_usage,
            };

            session.cpu_data.cpu_samples.push_back(sample);

            // Update peak CPU usage
            if cpu_usage > session.cpu_data.peak_cpu_usage {
                session.cpu_data.peak_cpu_usage = cpu_usage;
            }

            // Calculate running average
            let total_usage: f64 = session
                .cpu_data
                .cpu_samples
                .iter()
                .map(|s| s.cpu_usage)
                .sum();
            session.cpu_data.average_cpu_usage =
                total_usage / session.cpu_data.cpu_samples.len() as f64;

            // Limit sample count
            if session.cpu_data.cpu_samples.len() > self.config.max_samples_per_session {
                session.cpu_data.cpu_samples.pop_front();
            }
        }
    }

    /// Get global performance metrics
    pub fn get_global_metrics(&self) -> GlobalMetrics {
        self.global_metrics.read().unwrap().clone()
    }

    /// Get detailed report for a specific session
    pub fn get_session_report(&self, session_id: &str) -> Result<ProfilingReport> {
        let sessions = self.sessions.read().unwrap();

        if let Some(session) = sessions.get(session_id) {
            Ok(ProfilingReport::from_session(session))
        } else {
            Err(Error::processing(format!("Session {session_id} not found")))
        }
    }

    /// Get performance trends over time
    pub fn get_performance_trends(&self) -> Vec<TrendDataPoint> {
        let metrics = self.global_metrics.read().unwrap();
        metrics.performance_trends.iter().cloned().collect()
    }

    /// Calculate performance score for a session
    fn calculate_performance_score(&self, session: &ProfilingSession) -> f64 {
        let mut score = 1.0;

        // Factor in processing time relative to audio duration
        let processing_efficiency =
            session.audio_info.duration_seconds / session.timing_data.total_duration.as_secs_f64();

        if processing_efficiency < 1.0 {
            // Real-time factor penalty
            score *= processing_efficiency;
        }

        // Factor in memory efficiency
        let memory_per_second =
            session.memory_data.peak_memory as f64 / session.audio_info.duration_seconds;

        if memory_per_second > 1_000_000.0 {
            // > 1MB per second
            score *= 0.8;
        }

        // Factor in CPU efficiency
        if session.cpu_data.average_cpu_usage > 80.0 {
            score *= 0.7;
        }

        // Factor in bottlenecks
        let bottleneck_penalty = session.bottlenecks.len() as f64 * 0.1;
        score *= (1.0 - bottleneck_penalty).max(0.1);

        score.clamp(0.0, 1.0)
    }

    /// Update global metrics with session data
    fn update_global_metrics(&self, session: &ProfilingSession) {
        let mut metrics = self.global_metrics.write().unwrap();

        metrics.total_conversions += 1;

        // Update running averages
        let n = metrics.total_conversions as f64;
        let new_weight = 1.0 / n;
        let old_weight = (n - 1.0) / n;

        metrics.average_processing_time = Duration::from_nanos(
            (metrics.average_processing_time.as_nanos() as f64 * old_weight
                + session.timing_data.total_duration.as_nanos() as f64 * new_weight)
                as u64,
        );

        metrics.average_memory_usage = (metrics.average_memory_usage as f64 * old_weight
            + session.memory_data.peak_memory as f64 * new_weight)
            as usize;

        metrics.average_cpu_usage = metrics.average_cpu_usage * old_weight
            + session.cpu_data.average_cpu_usage * new_weight;

        // Update conversion type metrics
        let type_metrics = metrics
            .metrics_by_type
            .entry(session.conversion_type.clone())
            .or_insert_with(ConversionTypeMetrics::new);

        type_metrics.update_with_session(session);

        // Add trend data point
        let trend_point = TrendDataPoint {
            timestamp: SystemTime::now(),
            processing_time: session.timing_data.total_duration,
            memory_usage: session.memory_data.peak_memory,
            conversion_count: 1,
        };

        metrics.performance_trends.push_back(trend_point);

        // Limit trend data
        if metrics.performance_trends.len() > 1000 {
            metrics.performance_trends.pop_front();
        }
    }
}

/// Bottleneck analyzer for identifying performance issues
#[derive(Debug)]
pub struct BottleneckAnalyzer {
    /// Thresholds for identifying bottlenecks
    thresholds: BottleneckThresholds,
}

/// Thresholds for bottleneck detection
#[derive(Debug, Clone)]
pub struct BottleneckThresholds {
    /// Maximum acceptable processing time as multiple of audio duration
    pub max_processing_ratio: f64,
    /// Maximum acceptable memory usage per second of audio
    pub max_memory_per_second: usize,
    /// Maximum acceptable CPU usage percentage
    pub max_cpu_usage: f64,
    /// Maximum acceptable time for any single stage
    pub max_stage_time_ratio: f64,
}

impl Default for BottleneckThresholds {
    fn default() -> Self {
        Self {
            max_processing_ratio: 1.0,         // Real-time or faster
            max_memory_per_second: 10_000_000, // 10MB per second
            max_cpu_usage: 80.0,
            max_stage_time_ratio: 0.5, // No stage should take >50% of total time
        }
    }
}

/// Information about an identified bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckInfo {
    /// Type of bottleneck
    pub bottleneck_type: BottleneckType,
    /// Severity of the bottleneck (0.0 = minor, 1.0 = severe)
    pub severity: f64,
    /// Description of the bottleneck
    pub description: String,
    /// Recommended actions to address the bottleneck
    pub recommendations: Vec<String>,
    /// Affected component or stage
    pub affected_component: String,
    /// Measured value that triggered the bottleneck
    pub measured_value: f64,
    /// Expected or threshold value
    pub threshold_value: f64,
}

/// Types of bottlenecks that can be identified
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BottleneckType {
    /// Processing is too slow
    ProcessingSpeed,
    /// Memory usage is too high
    MemoryUsage,
    /// CPU usage is too high
    CpuUsage,
    /// Specific stage is taking too long
    StagePerformance,
    /// Memory allocation patterns are inefficient
    MemoryAllocation,
    /// Poor cache utilization
    CacheEfficiency,
    /// I/O operations are slow
    IoPerformance,
}

impl Default for BottleneckAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl BottleneckAnalyzer {
    /// Create a new bottleneck analyzer
    pub fn new() -> Self {
        Self {
            thresholds: BottleneckThresholds::default(),
        }
    }

    /// Create analyzer with custom thresholds
    pub fn with_thresholds(thresholds: BottleneckThresholds) -> Self {
        Self { thresholds }
    }

    /// Analyze a session for bottlenecks
    pub fn analyze_session(&self, session: &ProfilingSession) -> Vec<BottleneckInfo> {
        let mut bottlenecks = Vec::new();

        // Check processing speed
        bottlenecks.extend(self.check_processing_speed(session));

        // Check memory usage
        bottlenecks.extend(self.check_memory_usage(session));

        // Check CPU usage
        bottlenecks.extend(self.check_cpu_usage(session));

        // Check stage performance
        bottlenecks.extend(self.check_stage_performance(session));

        // Check memory allocation patterns
        bottlenecks.extend(self.check_memory_allocation(session));

        bottlenecks
    }

    fn check_processing_speed(&self, session: &ProfilingSession) -> Vec<BottleneckInfo> {
        let mut bottlenecks = Vec::new();

        let processing_ratio =
            session.timing_data.total_duration.as_secs_f64() / session.audio_info.duration_seconds;

        if processing_ratio > self.thresholds.max_processing_ratio {
            let severity = ((processing_ratio - self.thresholds.max_processing_ratio)
                / self.thresholds.max_processing_ratio)
                .min(1.0);

            bottlenecks.push(BottleneckInfo {
                bottleneck_type: BottleneckType::ProcessingSpeed,
                severity,
                description: format!(
                    "Processing is {:.2}x slower than real-time (target: {:.2}x)",
                    processing_ratio, self.thresholds.max_processing_ratio
                ),
                recommendations: vec![
                    "Consider using GPU acceleration".to_string(),
                    "Optimize algorithm parameters for speed".to_string(),
                    "Use lower quality settings for real-time applications".to_string(),
                ],
                affected_component: "Overall Processing".to_string(),
                measured_value: processing_ratio,
                threshold_value: self.thresholds.max_processing_ratio,
            });
        }

        bottlenecks
    }

    fn check_memory_usage(&self, session: &ProfilingSession) -> Vec<BottleneckInfo> {
        let mut bottlenecks = Vec::new();

        let memory_per_second =
            session.memory_data.peak_memory as f64 / session.audio_info.duration_seconds;

        if memory_per_second > self.thresholds.max_memory_per_second as f64 {
            let severity = ((memory_per_second - self.thresholds.max_memory_per_second as f64)
                / self.thresholds.max_memory_per_second as f64)
                .min(1.0);

            bottlenecks.push(BottleneckInfo {
                bottleneck_type: BottleneckType::MemoryUsage,
                severity,
                description: format!(
                    "Memory usage is {:.2} MB per second (target: {:.2} MB/s)",
                    memory_per_second / 1_000_000.0,
                    self.thresholds.max_memory_per_second as f64 / 1_000_000.0
                ),
                recommendations: vec![
                    "Enable memory pooling".to_string(),
                    "Reduce buffer sizes".to_string(),
                    "Use streaming processing for large files".to_string(),
                ],
                affected_component: "Memory Management".to_string(),
                measured_value: memory_per_second,
                threshold_value: self.thresholds.max_memory_per_second as f64,
            });
        }

        bottlenecks
    }

    fn check_cpu_usage(&self, session: &ProfilingSession) -> Vec<BottleneckInfo> {
        let mut bottlenecks = Vec::new();

        if session.cpu_data.average_cpu_usage > self.thresholds.max_cpu_usage {
            let severity = ((session.cpu_data.average_cpu_usage - self.thresholds.max_cpu_usage)
                / self.thresholds.max_cpu_usage)
                .min(1.0);

            bottlenecks.push(BottleneckInfo {
                bottleneck_type: BottleneckType::CpuUsage,
                severity,
                description: format!(
                    "CPU usage is {:.1}% (target: {:.1}%)",
                    session.cpu_data.average_cpu_usage, self.thresholds.max_cpu_usage
                ),
                recommendations: vec![
                    "Use multi-threading for parallel processing".to_string(),
                    "Optimize algorithms for CPU efficiency".to_string(),
                    "Consider using SIMD instructions".to_string(),
                ],
                affected_component: "CPU Processing".to_string(),
                measured_value: session.cpu_data.average_cpu_usage,
                threshold_value: self.thresholds.max_cpu_usage,
            });
        }

        bottlenecks
    }

    fn check_stage_performance(&self, session: &ProfilingSession) -> Vec<BottleneckInfo> {
        let mut bottlenecks = Vec::new();

        let total_time = session.timing_data.total_duration.as_secs_f64();

        for (stage_name, stage_info) in &session.timing_data.stage_timings {
            let stage_ratio = stage_info.duration.as_secs_f64() / total_time;

            if stage_ratio > self.thresholds.max_stage_time_ratio {
                let severity = ((stage_ratio - self.thresholds.max_stage_time_ratio)
                    / self.thresholds.max_stage_time_ratio)
                    .min(1.0);

                bottlenecks.push(BottleneckInfo {
                    bottleneck_type: BottleneckType::StagePerformance,
                    severity,
                    description: format!(
                        "Stage '{}' takes {:.1}% of total processing time (target: {:.1}%)",
                        stage_name,
                        stage_ratio * 100.0,
                        self.thresholds.max_stage_time_ratio * 100.0
                    ),
                    recommendations: vec![
                        format!("Optimize {} algorithm", stage_name),
                        "Consider caching results for this stage".to_string(),
                        "Profile individual operations within this stage".to_string(),
                    ],
                    affected_component: stage_name.clone(),
                    measured_value: stage_ratio,
                    threshold_value: self.thresholds.max_stage_time_ratio,
                });
            }
        }

        bottlenecks
    }

    fn check_memory_allocation(&self, session: &ProfilingSession) -> Vec<BottleneckInfo> {
        let mut bottlenecks = Vec::new();

        // Check for excessive allocations
        let total_allocations = session.memory_data.allocation_count;
        let allocation_rate = total_allocations as f64 / session.audio_info.duration_seconds;

        if allocation_rate > 1000.0 {
            // More than 1000 allocations per second
            bottlenecks.push(BottleneckInfo {
                bottleneck_type: BottleneckType::MemoryAllocation,
                severity: (allocation_rate / 10000.0).min(1.0),
                description: format!(
                    "High allocation rate: {allocation_rate:.0} allocations per second"
                ),
                recommendations: vec![
                    "Use object pooling to reduce allocations".to_string(),
                    "Pre-allocate buffers when possible".to_string(),
                    "Consider using stack allocation for small objects".to_string(),
                ],
                affected_component: "Memory Allocator".to_string(),
                measured_value: allocation_rate,
                threshold_value: 1000.0,
            });
        }

        // Check for memory leaks
        let net_allocations = session.memory_data.allocation_count as i64
            - session.memory_data.deallocation_count as i64;

        if net_allocations > 100 {
            bottlenecks.push(BottleneckInfo {
                bottleneck_type: BottleneckType::MemoryAllocation,
                severity: (net_allocations as f64 / 1000.0).min(1.0),
                description: format!("Potential memory leak: {net_allocations} net allocations"),
                recommendations: vec![
                    "Check for unreleased resources".to_string(),
                    "Ensure proper cleanup in error paths".to_string(),
                    "Use RAII patterns for resource management".to_string(),
                ],
                affected_component: "Memory Management".to_string(),
                measured_value: net_allocations as f64,
                threshold_value: 100.0,
            });
        }

        bottlenecks
    }
}

/// Comprehensive profiling report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingReport {
    /// Session information
    pub session_info: SessionInfo,
    /// Performance summary
    pub performance_summary: PerformanceSummary,
    /// Detailed timing breakdown
    pub timing_breakdown: TimingBreakdown,
    /// Memory analysis
    pub memory_analysis: MemoryAnalysis,
    /// CPU analysis
    pub cpu_analysis: CpuAnalysis,
    /// Identified bottlenecks
    pub bottlenecks: Vec<BottleneckInfo>,
    /// Overall recommendations
    pub recommendations: Vec<String>,
    /// Performance score
    pub performance_score: f64,
}

/// Session information summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub conversion_type: ConversionType,
    pub audio_duration: f64,
    pub audio_samples: usize,
    pub sample_rate: f32,
    pub channels: usize,
    pub start_time: SystemTime,
    pub end_time: Option<SystemTime>,
}

/// Performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_processing_time: Duration,
    pub real_time_factor: f64,
    pub peak_memory_mb: f64,
    pub average_cpu_usage: f64,
    pub peak_cpu_usage: f64,
    pub bottleneck_count: usize,
    pub performance_grade: String,
}

/// Detailed timing breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingBreakdown {
    pub preprocessing_percentage: f64,
    pub conversion_percentage: f64,
    pub postprocessing_percentage: f64,
    pub model_init_percentage: f64,
    pub quality_assessment_percentage: f64,
    pub slowest_stage: String,
    pub fastest_stage: String,
}

/// Memory usage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAnalysis {
    pub peak_memory_mb: f64,
    pub average_memory_mb: f64,
    pub memory_efficiency_score: f64,
    pub allocation_count: usize,
    pub deallocation_count: usize,
    pub potential_leaks: bool,
    pub memory_growth_rate: f64,
}

/// CPU usage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuAnalysis {
    pub average_usage: f64,
    pub peak_usage: f64,
    pub cpu_efficiency_score: f64,
    pub thread_utilization: f64,
    pub cpu_intensive_stages: Vec<String>,
}

impl ProfilingReport {
    /// Create a report from a profiling session
    pub fn from_session(session: &ProfilingSession) -> Self {
        let session_info = SessionInfo {
            session_id: session.session_id.clone(),
            conversion_type: session.conversion_type.clone(),
            audio_duration: session.audio_info.duration_seconds,
            audio_samples: session.audio_info.length_samples,
            sample_rate: session.audio_info.sample_rate,
            channels: session.audio_info.channels,
            start_time: session.start_time,
            end_time: session.end_time,
        };

        let real_time_factor =
            session.timing_data.total_duration.as_secs_f64() / session.audio_info.duration_seconds;

        let performance_summary = PerformanceSummary {
            total_processing_time: session.timing_data.total_duration,
            real_time_factor,
            peak_memory_mb: session.memory_data.peak_memory as f64 / 1_000_000.0,
            average_cpu_usage: session.cpu_data.average_cpu_usage,
            peak_cpu_usage: session.cpu_data.peak_cpu_usage,
            bottleneck_count: session.bottlenecks.len(),
            performance_grade: Self::calculate_grade(session.performance_score),
        };

        let total_time = session.timing_data.total_duration.as_secs_f64();
        let timing_breakdown = TimingBreakdown {
            preprocessing_percentage: session.timing_data.preprocessing_time.as_secs_f64()
                / total_time
                * 100.0,
            conversion_percentage: session.timing_data.conversion_time.as_secs_f64() / total_time
                * 100.0,
            postprocessing_percentage: session.timing_data.postprocessing_time.as_secs_f64()
                / total_time
                * 100.0,
            model_init_percentage: session.timing_data.model_init_time.as_secs_f64() / total_time
                * 100.0,
            quality_assessment_percentage: session
                .timing_data
                .quality_assessment_time
                .as_secs_f64()
                / total_time
                * 100.0,
            slowest_stage: Self::find_slowest_stage(&session.timing_data),
            fastest_stage: Self::find_fastest_stage(&session.timing_data),
        };

        let memory_analysis = MemoryAnalysis {
            peak_memory_mb: session.memory_data.peak_memory as f64 / 1_000_000.0,
            average_memory_mb: Self::calculate_average_memory(&session.memory_data) / 1_000_000.0,
            memory_efficiency_score: Self::calculate_memory_efficiency(
                &session.memory_data,
                session.audio_info.duration_seconds,
            ),
            allocation_count: session.memory_data.allocation_count,
            deallocation_count: session.memory_data.deallocation_count,
            potential_leaks: session.memory_data.allocation_count
                > session.memory_data.deallocation_count + 10,
            memory_growth_rate: Self::calculate_memory_growth_rate(&session.memory_data),
        };

        let cpu_analysis = CpuAnalysis {
            average_usage: session.cpu_data.average_cpu_usage,
            peak_usage: session.cpu_data.peak_cpu_usage,
            cpu_efficiency_score: Self::calculate_cpu_efficiency(&session.cpu_data),
            thread_utilization: session.cpu_data.thread_count as f64 / num_cpus::get() as f64,
            cpu_intensive_stages: Self::find_cpu_intensive_stages(&session.timing_data),
        };

        let recommendations = Self::generate_recommendations(session);

        Self {
            session_info,
            performance_summary,
            timing_breakdown,
            memory_analysis,
            cpu_analysis,
            bottlenecks: session.bottlenecks.clone(),
            recommendations,
            performance_score: session.performance_score,
        }
    }

    fn calculate_grade(score: f64) -> String {
        if score >= 0.9 {
            "A".to_string()
        } else if score >= 0.8 {
            "B".to_string()
        } else if score >= 0.7 {
            "C".to_string()
        } else if score >= 0.6 {
            "D".to_string()
        } else {
            "F".to_string()
        }
    }

    fn find_slowest_stage(timing_data: &TimingData) -> String {
        timing_data
            .stage_timings
            .iter()
            .max_by_key(|(_, info)| info.duration)
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| "Unknown".to_string())
    }

    fn find_fastest_stage(timing_data: &TimingData) -> String {
        timing_data
            .stage_timings
            .iter()
            .min_by_key(|(_, info)| info.duration)
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| "Unknown".to_string())
    }

    fn calculate_average_memory(memory_data: &MemoryData) -> f64 {
        if memory_data.memory_samples.is_empty() {
            memory_data.peak_memory as f64
        } else {
            let sum: usize = memory_data
                .memory_samples
                .iter()
                .map(|s| s.memory_usage)
                .sum();
            sum as f64 / memory_data.memory_samples.len() as f64
        }
    }

    fn calculate_memory_efficiency(memory_data: &MemoryData, duration: f64) -> f64 {
        let memory_per_second = memory_data.peak_memory as f64 / duration;
        let efficiency = 1.0 - (memory_per_second / 10_000_000.0).min(1.0); // 10MB/s as baseline
        efficiency.max(0.0)
    }

    fn calculate_memory_growth_rate(memory_data: &MemoryData) -> f64 {
        if memory_data.memory_samples.len() < 2 {
            return 0.0;
        }

        let first = memory_data.memory_samples.front().unwrap().memory_usage as f64;
        let last = memory_data.memory_samples.back().unwrap().memory_usage as f64;
        let growth = (last - first) / first;
        growth.clamp(-1.0, 10.0) // Clamp between -100% and 1000%
    }

    fn calculate_cpu_efficiency(cpu_data: &CpuData) -> f64 {
        let efficiency = 1.0 - (cpu_data.average_cpu_usage / 100.0);
        efficiency.max(0.0)
    }

    fn find_cpu_intensive_stages(timing_data: &TimingData) -> Vec<String> {
        let total_time = timing_data.total_duration.as_secs_f64();
        timing_data
            .stage_timings
            .iter()
            .filter(|(_, info)| info.duration.as_secs_f64() / total_time > 0.2) // >20% of total time
            .map(|(name, _)| name.clone())
            .collect()
    }

    fn generate_recommendations(session: &ProfilingSession) -> Vec<String> {
        let mut recommendations = Vec::new();

        let rtf =
            session.timing_data.total_duration.as_secs_f64() / session.audio_info.duration_seconds;
        if rtf > 1.0 {
            recommendations
                .push("Consider enabling GPU acceleration for faster processing".to_string());
            recommendations
                .push("Use lower quality settings for real-time applications".to_string());
        }

        let memory_per_sec =
            session.memory_data.peak_memory as f64 / session.audio_info.duration_seconds;
        if memory_per_sec > 10_000_000.0 {
            recommendations.push("Enable memory pooling to reduce memory usage".to_string());
            recommendations.push("Use streaming processing for large audio files".to_string());
        }

        if session.cpu_data.average_cpu_usage > 80.0 {
            recommendations.push("Enable multi-threading for better CPU utilization".to_string());
            recommendations.push("Consider using SIMD optimizations".to_string());
        }

        if session.bottlenecks.len() > 3 {
            recommendations
                .push("Address identified bottlenecks to improve overall performance".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Performance is good - no major optimizations needed".to_string());
        }

        recommendations
    }

    /// Export report to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(Error::Serialization)
    }

    /// Export report to human-readable text
    pub fn to_text(&self) -> String {
        let mut report = String::new();

        report.push_str("=== VoiRS Conversion Performance Report ===\n\n");

        // Session Info
        report.push_str(&format!("Session ID: {}\n", self.session_info.session_id));
        report.push_str(&format!(
            "Conversion Type: {:?}\n",
            self.session_info.conversion_type
        ));
        report.push_str(&format!(
            "Audio Duration: {:.2}s\n",
            self.session_info.audio_duration
        ));
        report.push_str(&format!(
            "Sample Rate: {:.0} Hz\n",
            self.session_info.sample_rate
        ));
        report.push_str(&format!("Channels: {}\n\n", self.session_info.channels));

        // Performance Summary
        report.push_str("=== Performance Summary ===\n");
        report.push_str(&format!(
            "Processing Time: {:.2}s\n",
            self.performance_summary.total_processing_time.as_secs_f64()
        ));
        report.push_str(&format!(
            "Real-time Factor: {:.2}x\n",
            self.performance_summary.real_time_factor
        ));
        report.push_str(&format!(
            "Peak Memory: {:.1} MB\n",
            self.performance_summary.peak_memory_mb
        ));
        report.push_str(&format!(
            "Average CPU: {:.1}%\n",
            self.performance_summary.average_cpu_usage
        ));
        report.push_str(&format!(
            "Performance Grade: {}\n",
            self.performance_summary.performance_grade
        ));
        report.push_str(&format!(
            "Performance Score: {:.2}\n\n",
            self.performance_score
        ));

        // Timing Breakdown
        report.push_str("=== Timing Breakdown ===\n");
        report.push_str(&format!(
            "Preprocessing: {:.1}%\n",
            self.timing_breakdown.preprocessing_percentage
        ));
        report.push_str(&format!(
            "Conversion: {:.1}%\n",
            self.timing_breakdown.conversion_percentage
        ));
        report.push_str(&format!(
            "Postprocessing: {:.1}%\n",
            self.timing_breakdown.postprocessing_percentage
        ));
        report.push_str(&format!(
            "Model Init: {:.1}%\n",
            self.timing_breakdown.model_init_percentage
        ));
        report.push_str(&format!(
            "Quality Assessment: {:.1}%\n",
            self.timing_breakdown.quality_assessment_percentage
        ));
        report.push_str(&format!(
            "Slowest Stage: {}\n\n",
            self.timing_breakdown.slowest_stage
        ));

        // Bottlenecks
        if !self.bottlenecks.is_empty() {
            report.push_str("=== Identified Bottlenecks ===\n");
            for bottleneck in &self.bottlenecks {
                report.push_str(&format!(
                    "- {} (Severity: {:.2})\n",
                    bottleneck.description, bottleneck.severity
                ));
                report.push_str(&format!(
                    "  Component: {component}\n",
                    component = bottleneck.affected_component
                ));
                for rec in &bottleneck.recommendations {
                    report.push_str(&format!("  → {rec}\n"));
                }
                report.push('\n');
            }
        }

        // Recommendations
        report.push_str("=== Recommendations ===\n");
        for (i, rec) in self.recommendations.iter().enumerate() {
            report.push_str(&format!("{}. {}\n", i + 1, rec));
        }

        report
    }
}

// Implementation details for the data structures
impl Default for TimingData {
    fn default() -> Self {
        Self::new()
    }
}

impl TimingData {
    pub fn new() -> Self {
        Self {
            total_duration: Duration::from_millis(0),
            preprocessing_time: Duration::from_millis(0),
            conversion_time: Duration::from_millis(0),
            postprocessing_time: Duration::from_millis(0),
            model_init_time: Duration::from_millis(0),
            quality_assessment_time: Duration::from_millis(0),
            stage_timings: BTreeMap::new(),
        }
    }
}

impl Default for MemoryData {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryData {
    pub fn new() -> Self {
        Self {
            peak_memory: 0,
            initial_memory: 0,
            final_memory: 0,
            allocation_count: 0,
            deallocation_count: 0,
            total_allocated: 0,
            memory_samples: VecDeque::new(),
        }
    }
}

impl Default for CpuData {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuData {
    pub fn new() -> Self {
        Self {
            average_cpu_usage: 0.0,
            peak_cpu_usage: 0.0,
            cpu_samples: VecDeque::new(),
            thread_count: 1,
        }
    }
}

impl StageTimingInfo {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            duration: Duration::from_millis(0),
            execution_count: 0,
            average_duration: Duration::from_millis(0),
            min_duration: Duration::from_secs(u64::MAX),
            max_duration: Duration::from_millis(0),
        }
    }

    pub fn update_timing(&mut self, new_duration: Duration) {
        self.execution_count += 1;
        self.duration += new_duration;

        if new_duration < self.min_duration {
            self.min_duration = new_duration;
        }
        if new_duration > self.max_duration {
            self.max_duration = new_duration;
        }

        self.average_duration = self.duration / self.execution_count as u32;
    }
}

impl Default for ConversionTypeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversionTypeMetrics {
    pub fn new() -> Self {
        Self {
            conversion_count: 0,
            average_time: Duration::from_millis(0),
            average_memory: 0,
            average_score: 0.0,
            common_bottlenecks: HashMap::new(),
        }
    }

    pub fn update_with_session(&mut self, session: &ProfilingSession) {
        self.conversion_count += 1;

        let n = self.conversion_count as f64;
        let new_weight = 1.0 / n;
        let old_weight = (n - 1.0) / n;

        self.average_time = Duration::from_nanos(
            (self.average_time.as_nanos() as f64 * old_weight
                + session.timing_data.total_duration.as_nanos() as f64 * new_weight)
                as u64,
        );

        self.average_memory = (self.average_memory as f64 * old_weight
            + session.memory_data.peak_memory as f64 * new_weight)
            as usize;

        self.average_score =
            self.average_score * old_weight + session.performance_score * new_weight;

        // Update common bottlenecks
        for bottleneck in &session.bottlenecks {
            *self
                .common_bottlenecks
                .entry(bottleneck.affected_component.clone())
                .or_insert(0) += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_session_lifecycle() {
        let profiler = ConversionProfiler::new();

        let audio_info = AudioInfo {
            length_samples: 44100,
            sample_rate: 44100.0,
            channels: 1,
            duration_seconds: 1.0,
            peak_amplitude: 0.8,
            rms_level: 0.2,
        };

        let session_id = profiler.start_session(ConversionType::PitchShift, audio_info);

        // Simulate some processing
        profiler.record_stage_timing(&session_id, "preprocessing", Duration::from_millis(50));
        profiler.record_stage_timing(&session_id, "conversion", Duration::from_millis(200));
        profiler.record_memory_sample(&session_id, 1_000_000, 10);
        profiler.record_cpu_sample(&session_id, 75.0, 1_000_000);

        let report = profiler.end_session(&session_id).unwrap();

        assert!(!report.session_info.session_id.is_empty());
        assert_eq!(
            report.session_info.conversion_type,
            ConversionType::PitchShift
        );
        assert!(report.performance_score > 0.0);
    }

    #[test]
    fn test_bottleneck_analyzer() {
        let analyzer = BottleneckAnalyzer::new();

        let mut session = ProfilingSession {
            session_id: "test".to_string(),
            start_time: SystemTime::now(),
            end_time: None,
            conversion_type: ConversionType::PitchShift,
            audio_info: AudioInfo {
                length_samples: 44100,
                sample_rate: 44100.0,
                channels: 1,
                duration_seconds: 1.0,
                peak_amplitude: 0.8,
                rms_level: 0.2,
            },
            timing_data: TimingData::new(),
            memory_data: MemoryData::new(),
            cpu_data: CpuData::new(),
            bottlenecks: Vec::new(),
            performance_score: 0.0,
        };

        // Create a slow processing scenario
        session.timing_data.total_duration = Duration::from_secs(5); // 5x slower than real-time
        session.cpu_data.average_cpu_usage = 90.0; // High CPU usage

        let bottlenecks = analyzer.analyze_session(&session);

        assert!(!bottlenecks.is_empty());
        assert!(bottlenecks
            .iter()
            .any(|b| b.bottleneck_type == BottleneckType::ProcessingSpeed));
        assert!(bottlenecks
            .iter()
            .any(|b| b.bottleneck_type == BottleneckType::CpuUsage));
    }

    #[test]
    fn test_profiling_report_generation() {
        let mut session = ProfilingSession {
            session_id: "test_report".to_string(),
            start_time: SystemTime::now(),
            end_time: Some(SystemTime::now()),
            conversion_type: ConversionType::SpeedTransformation,
            audio_info: AudioInfo {
                length_samples: 88200,
                sample_rate: 44100.0,
                channels: 2,
                duration_seconds: 2.0,
                peak_amplitude: 0.9,
                rms_level: 0.3,
            },
            timing_data: TimingData::new(),
            memory_data: MemoryData::new(),
            cpu_data: CpuData::new(),
            bottlenecks: Vec::new(),
            performance_score: 0.85,
        };

        session.timing_data.total_duration = Duration::from_millis(1500);
        session.memory_data.peak_memory = 2_000_000;
        session.cpu_data.average_cpu_usage = 45.0;

        let report = ProfilingReport::from_session(&session);

        assert_eq!(
            report.session_info.conversion_type,
            ConversionType::SpeedTransformation
        );
        assert_eq!(report.performance_summary.performance_grade, "B");
        assert!(report.performance_summary.real_time_factor < 1.0); // Faster than real-time

        let text_report = report.to_text();
        assert!(text_report.contains("Performance Report"));
        assert!(text_report.contains("Real-time Factor"));

        let json_report = report.to_json().unwrap();
        assert!(json_report.contains("session_info"));
    }

    #[test]
    fn test_stage_timing_info() {
        let mut stage = StageTimingInfo::new("test_stage");

        stage.update_timing(Duration::from_millis(100));
        stage.update_timing(Duration::from_millis(200));
        stage.update_timing(Duration::from_millis(50));

        assert_eq!(stage.execution_count, 3);
        assert_eq!(stage.min_duration, Duration::from_millis(50));
        assert_eq!(stage.max_duration, Duration::from_millis(200));
        // (100+200+50)/3 = 350/3 ≈ 116.666ms
        let expected_avg = Duration::from_millis(350) / 3;
        assert_eq!(stage.average_duration, expected_avg);
    }
}
