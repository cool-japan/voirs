//! Comprehensive performance profiling and analysis for VoiRS SDK.
//!
//! This module provides detailed performance profiling capabilities including:
//! - Stage-by-stage timing analysis for synthesis pipeline
//! - Memory allocation tracking and profiling
//! - Bottleneck detection and recommendations
//! - Performance comparison and regression detection
//! - Real-time performance monitoring
//! - Historical performance tracking
//!
//! # Examples
//!
//! ```no_run
//! use voirs_sdk::prelude::*;
//! use voirs_sdk::profiling::{Profiler, ProfilerConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let pipeline = VoirsPipelineBuilder::new().build().await?;
//!
//!     // Create profiler
//!     let profiler = Profiler::new(ProfilerConfig::default());
//!
//!     // Start profiling session
//!     let session = profiler.start_session("synthesis_test").await;
//!
//!     // Perform synthesis with profiling
//!     let audio = pipeline.synthesize("Hello, world!").await?;
//!
//!     // End session and get report
//!     let report = profiler.end_session(session).await?;
//!
//!     // Display performance report
//!     println!("{}", report.summary());
//!     println!("Bottlenecks: {:?}", report.bottlenecks());
//!
//!     Ok(())
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

fn instant_now() -> Instant {
    Instant::now()
}

mod bottleneck_detector;
mod comparator;
mod memory_profiler;
mod pipeline_profiler;
mod report_generator;

pub use bottleneck_detector::{Bottleneck, BottleneckDetector, BottleneckSeverity};
pub use comparator::{ComparisonResult, PerformanceComparator, RegressionDetector};
pub use memory_profiler::{AllocationTracker, MemoryProfiler, MemorySnapshot};
pub use pipeline_profiler::{PipelineProfiler, PipelineStage, StageMetrics};
pub use report_generator::{PerformanceReport, ReportFormat, ReportGenerator};

/// Configuration for the performance profiler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Enable detailed timing profiling
    pub enable_timing: bool,

    /// Enable memory profiling
    pub enable_memory: bool,

    /// Enable bottleneck detection
    pub enable_bottleneck_detection: bool,

    /// Maximum number of sessions to keep in history
    pub max_history_size: usize,

    /// Sampling interval for real-time monitoring (milliseconds)
    pub sampling_interval_ms: u64,

    /// Enable automatic report generation
    pub auto_generate_reports: bool,

    /// Report output directory
    pub report_output_dir: Option<std::path::PathBuf>,

    /// Enable comparison with baseline
    pub enable_baseline_comparison: bool,

    /// Threshold for regression detection (percentage)
    pub regression_threshold_percent: f64,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enable_timing: true,
            enable_memory: true,
            enable_bottleneck_detection: true,
            max_history_size: 100,
            sampling_interval_ms: 100,
            auto_generate_reports: false,
            report_output_dir: None,
            enable_baseline_comparison: false,
            regression_threshold_percent: 10.0,
        }
    }
}

/// A profiling session tracking performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSession {
    /// Unique session identifier
    pub id: String,

    /// Session name
    pub name: String,

    /// Start time (not serialized)
    #[serde(skip, default = "instant_now")]
    pub start_time: Instant,

    /// End time (if session is complete, not serialized)
    #[serde(skip, default)]
    pub end_time: Option<Instant>,

    /// Total duration in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<Duration>,

    /// Stage metrics
    pub stage_metrics: HashMap<String, StageMetrics>,

    /// Memory snapshots
    pub memory_snapshots: Vec<MemorySnapshot>,

    /// Detected bottlenecks
    pub bottlenecks: Vec<Bottleneck>,

    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl Default for ProfileSession {
    fn default() -> Self {
        Self::new("")
    }
}

impl ProfileSession {
    /// Create a new profiling session.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.into(),
            start_time: Instant::now(),
            end_time: None,
            duration: None,
            stage_metrics: HashMap::new(),
            memory_snapshots: Vec::new(),
            bottlenecks: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Mark session as complete.
    pub fn complete(&mut self) {
        let end_time = Instant::now();
        self.duration = Some(end_time.duration_since(self.start_time));
        self.end_time = Some(end_time);
    }

    /// Check if session is complete.
    pub fn is_complete(&self) -> bool {
        self.end_time.is_some()
    }

    /// Get session duration.
    pub fn duration(&self) -> Option<Duration> {
        self.duration
    }

    /// Add stage metrics.
    pub fn add_stage_metrics(&mut self, stage: String, metrics: StageMetrics) {
        self.stage_metrics.insert(stage, metrics);
    }

    /// Add memory snapshot.
    pub fn add_memory_snapshot(&mut self, snapshot: MemorySnapshot) {
        self.memory_snapshots.push(snapshot);
    }

    /// Add detected bottleneck.
    pub fn add_bottleneck(&mut self, bottleneck: Bottleneck) {
        self.bottlenecks.push(bottleneck);
    }

    /// Add custom metadata.
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
}

/// Main performance profiler for VoiRS SDK.
pub struct Profiler {
    config: ProfilerConfig,
    sessions: Arc<RwLock<VecDeque<ProfileSession>>>,
    pipeline_profiler: Arc<PipelineProfiler>,
    memory_profiler: Arc<MemoryProfiler>,
    bottleneck_detector: Arc<BottleneckDetector>,
    report_generator: Arc<ReportGenerator>,
    comparator: Arc<PerformanceComparator>,
    baseline_session: Arc<RwLock<Option<ProfileSession>>>,
}

impl Profiler {
    /// Create a new profiler with the given configuration.
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            config: config.clone(),
            sessions: Arc::new(RwLock::new(VecDeque::new())),
            pipeline_profiler: Arc::new(PipelineProfiler::new()),
            memory_profiler: Arc::new(MemoryProfiler::new()),
            bottleneck_detector: Arc::new(BottleneckDetector::new(
                config.regression_threshold_percent,
            )),
            report_generator: Arc::new(ReportGenerator::new(config.clone())),
            comparator: Arc::new(PerformanceComparator::new()),
            baseline_session: Arc::new(RwLock::new(None)),
        }
    }

    /// Start a new profiling session.
    pub async fn start_session(&self, name: impl Into<String>) -> ProfileSession {
        let session = ProfileSession::new(name);
        info!(
            "Started profiling session: {} ({})",
            session.name, session.id
        );
        session
    }

    /// End a profiling session and analyze results.
    pub async fn end_session(
        &self,
        mut session: ProfileSession,
    ) -> Result<PerformanceReport, crate::VoirsError> {
        session.complete();
        info!(
            "Ended profiling session: {} (duration: {:?})",
            session.name,
            session.duration()
        );

        // Detect bottlenecks if enabled
        if self.config.enable_bottleneck_detection {
            let bottlenecks = self.bottleneck_detector.detect(&session).await;
            for bottleneck in bottlenecks {
                session.add_bottleneck(bottleneck);
            }
        }

        // Store session in history
        let mut sessions = self.sessions.write().await;
        if sessions.len() >= self.config.max_history_size {
            sessions.pop_front();
        }
        sessions.push_back(session.clone());
        drop(sessions);

        // Compare with baseline if enabled
        let comparison = if self.config.enable_baseline_comparison {
            let baseline = self.baseline_session.read().await;
            if let Some(baseline_session) = baseline.as_ref() {
                Some(self.comparator.compare(&session, baseline_session).await)
            } else {
                None
            }
        } else {
            None
        };

        // Generate performance report
        let report = self.report_generator.generate(&session, comparison).await?;

        // Auto-save report if enabled
        if self.config.auto_generate_reports {
            if let Some(output_dir) = &self.config.report_output_dir {
                let report_path = output_dir.join(format!("{}.json", session.id));
                self.report_generator
                    .save_report(&report, &report_path)
                    .await?;
            }
        }

        Ok(report)
    }

    /// Set baseline session for comparison.
    pub async fn set_baseline(&self, session: ProfileSession) {
        let mut baseline = self.baseline_session.write().await;
        *baseline = Some(session);
        info!("Baseline session set for comparison");
    }

    /// Get all profiling sessions.
    pub async fn get_sessions(&self) -> Vec<ProfileSession> {
        self.sessions.read().await.iter().cloned().collect()
    }

    /// Clear session history.
    pub async fn clear_history(&self) {
        let mut sessions = self.sessions.write().await;
        sessions.clear();
        info!("Cleared profiling session history");
    }

    /// Get pipeline profiler for detailed stage profiling.
    pub fn pipeline_profiler(&self) -> Arc<PipelineProfiler> {
        self.pipeline_profiler.clone()
    }

    /// Get memory profiler for memory analysis.
    pub fn memory_profiler(&self) -> Arc<MemoryProfiler> {
        self.memory_profiler.clone()
    }

    /// Get bottleneck detector.
    pub fn bottleneck_detector(&self) -> Arc<BottleneckDetector> {
        self.bottleneck_detector.clone()
    }

    /// Generate aggregate report from multiple sessions.
    pub async fn generate_aggregate_report(
        &self,
        sessions: &[ProfileSession],
    ) -> Result<PerformanceReport, crate::VoirsError> {
        self.report_generator.generate_aggregate(sessions).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_config_default() {
        let config = ProfilerConfig::default();
        assert!(config.enable_timing);
        assert!(config.enable_memory);
        assert!(config.enable_bottleneck_detection);
        assert_eq!(config.max_history_size, 100);
    }

    #[test]
    fn test_profile_session_creation() {
        let session = ProfileSession::new("test_session");
        assert_eq!(session.name, "test_session");
        assert!(!session.is_complete());
        assert!(session.duration().is_none());
    }

    #[test]
    fn test_profile_session_completion() {
        let mut session = ProfileSession::new("test_session");
        session.complete();
        assert!(session.is_complete());
        assert!(session.duration().is_some());
    }

    #[test]
    fn test_profile_session_metadata() {
        let mut session = ProfileSession::new("test_session");
        session.add_metadata("key1", "value1");
        session.add_metadata("key2", "value2");
        assert_eq!(session.metadata.len(), 2);
        assert_eq!(session.metadata.get("key1"), Some(&"value1".to_string()));
    }

    #[tokio::test]
    async fn test_profiler_creation() {
        let config = ProfilerConfig::default();
        let profiler = Profiler::new(config);

        let sessions = profiler.get_sessions().await;
        assert_eq!(sessions.len(), 0);
    }

    #[tokio::test]
    async fn test_profiler_session_management() {
        let profiler = Profiler::new(ProfilerConfig::default());

        let session = profiler.start_session("test").await;
        assert_eq!(session.name, "test");

        // Note: We can't call end_session in this test without proper report generation
        // which requires the full implementation of sub-modules
    }

    #[tokio::test]
    async fn test_profiler_clear_history() {
        let profiler = Profiler::new(ProfilerConfig::default());
        profiler.clear_history().await;

        let sessions = profiler.get_sessions().await;
        assert_eq!(sessions.len(), 0);
    }
}
