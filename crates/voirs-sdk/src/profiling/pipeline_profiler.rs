//! Pipeline stage profiling and timing analysis.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Represents a stage in the synthesis pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PipelineStage {
    /// Text preprocessing stage
    TextPreprocessing,
    /// G2P (Grapheme-to-Phoneme) conversion
    G2pConversion,
    /// Acoustic model inference
    AcousticModel,
    /// Vocoder synthesis
    Vocoder,
    /// Post-processing and enhancement
    PostProcessing,
    /// Audio encoding/formatting
    AudioEncoding,
    /// Complete end-to-end pipeline
    FullPipeline,
}

impl PipelineStage {
    /// Get human-readable name for the stage.
    pub fn name(&self) -> &'static str {
        match self {
            Self::TextPreprocessing => "Text Preprocessing",
            Self::G2pConversion => "G2P Conversion",
            Self::AcousticModel => "Acoustic Model",
            Self::Vocoder => "Vocoder",
            Self::PostProcessing => "Post-Processing",
            Self::AudioEncoding => "Audio Encoding",
            Self::FullPipeline => "Full Pipeline",
        }
    }

    /// Get typical expected duration percentage of total pipeline.
    pub fn expected_percentage(&self) -> f64 {
        match self {
            Self::TextPreprocessing => 2.0,
            Self::G2pConversion => 5.0,
            Self::AcousticModel => 35.0,
            Self::Vocoder => 50.0,
            Self::PostProcessing => 5.0,
            Self::AudioEncoding => 3.0,
            Self::FullPipeline => 100.0,
        }
    }
}

/// Metrics for a specific pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageMetrics {
    /// Stage identifier
    pub stage: PipelineStage,

    /// Number of times this stage was executed
    pub execution_count: usize,

    /// Total time spent in this stage
    pub total_duration: Duration,

    /// Minimum execution time
    pub min_duration: Duration,

    /// Maximum execution time
    pub max_duration: Duration,

    /// Average execution time
    pub avg_duration: Duration,

    /// Standard deviation of execution times
    pub std_deviation: Duration,

    /// Percentage of total pipeline time
    pub percentage_of_total: f64,

    /// Input size (e.g., text length, phoneme count)
    pub avg_input_size: f64,

    /// Output size (e.g., mel spectrogram frames, audio samples)
    pub avg_output_size: f64,

    /// Throughput (output per second)
    pub throughput: f64,
}

impl StageMetrics {
    /// Create new stage metrics.
    pub fn new(stage: PipelineStage) -> Self {
        Self {
            stage,
            execution_count: 0,
            total_duration: Duration::from_secs(0),
            min_duration: Duration::from_secs(u64::MAX),
            max_duration: Duration::from_secs(0),
            avg_duration: Duration::from_secs(0),
            std_deviation: Duration::from_secs(0),
            percentage_of_total: 0.0,
            avg_input_size: 0.0,
            avg_output_size: 0.0,
            throughput: 0.0,
        }
    }

    /// Record a new execution.
    pub fn record_execution(&mut self, duration: Duration, input_size: usize, output_size: usize) {
        self.execution_count += 1;
        self.total_duration += duration;

        if duration < self.min_duration {
            self.min_duration = duration;
        }
        if duration > self.max_duration {
            self.max_duration = duration;
        }

        // Update rolling averages
        let count = self.execution_count as f64;
        let new_input = input_size as f64;
        let new_output = output_size as f64;

        self.avg_input_size = (self.avg_input_size * (count - 1.0) + new_input) / count;
        self.avg_output_size = (self.avg_output_size * (count - 1.0) + new_output) / count;

        // Update average duration
        self.avg_duration = self.total_duration / self.execution_count.try_into().unwrap_or(1);

        // Update throughput (output per second)
        if duration.as_secs_f64() > 0.0 {
            self.throughput = self.avg_output_size / self.avg_duration.as_secs_f64();
        }
    }

    /// Calculate standard deviation of execution times.
    pub fn calculate_std_deviation(&mut self, durations: &[Duration]) {
        if durations.len() < 2 {
            return;
        }

        let mean = self.avg_duration.as_secs_f64();
        let variance: f64 = durations
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - mean;
                diff * diff
            })
            .sum::<f64>()
            / durations.len() as f64;

        self.std_deviation = Duration::from_secs_f64(variance.sqrt());
    }

    /// Check if this stage is a potential bottleneck.
    pub fn is_bottleneck(&self) -> bool {
        // A stage is considered a bottleneck if it takes significantly more
        // time than expected based on typical distributions
        let expected = self.stage.expected_percentage();
        self.percentage_of_total > expected * 1.5 // 50% more than expected
    }
}

/// Pipeline profiler for tracking stage-by-stage performance.
pub struct PipelineProfiler {
    stage_metrics: Arc<RwLock<HashMap<PipelineStage, StageMetrics>>>,
    active_timings: Arc<RwLock<HashMap<String, (PipelineStage, Instant)>>>,
}

impl PipelineProfiler {
    /// Create a new pipeline profiler.
    pub fn new() -> Self {
        Self {
            stage_metrics: Arc::new(RwLock::new(HashMap::new())),
            active_timings: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start timing a pipeline stage.
    pub async fn start_stage(&self, stage: PipelineStage) -> String {
        let timing_id = uuid::Uuid::new_v4().to_string();
        let mut active = self.active_timings.write().await;
        active.insert(timing_id.clone(), (stage, Instant::now()));
        timing_id
    }

    /// End timing a pipeline stage and record metrics.
    pub async fn end_stage(
        &self,
        timing_id: &str,
        input_size: usize,
        output_size: usize,
    ) -> Option<Duration> {
        let mut active = self.active_timings.write().await;
        if let Some((stage, start_time)) = active.remove(timing_id) {
            let duration = start_time.elapsed();

            // Update metrics
            let mut metrics = self.stage_metrics.write().await;
            let stage_metrics = metrics
                .entry(stage)
                .or_insert_with(|| StageMetrics::new(stage));
            stage_metrics.record_execution(duration, input_size, output_size);

            Some(duration)
        } else {
            None
        }
    }

    /// Get metrics for a specific stage.
    pub async fn get_stage_metrics(&self, stage: PipelineStage) -> Option<StageMetrics> {
        let metrics = self.stage_metrics.read().await;
        metrics.get(&stage).cloned()
    }

    /// Get all stage metrics.
    pub async fn get_all_metrics(&self) -> HashMap<PipelineStage, StageMetrics> {
        self.stage_metrics.read().await.clone()
    }

    /// Calculate percentage of total time for each stage.
    pub async fn calculate_percentages(&self) {
        let mut metrics = self.stage_metrics.write().await;

        // Calculate total duration across all stages
        let total_duration: Duration = metrics.values().map(|m| m.total_duration).sum();

        if total_duration.as_secs_f64() > 0.0 {
            for stage_metrics in metrics.values_mut() {
                stage_metrics.percentage_of_total = (stage_metrics.total_duration.as_secs_f64()
                    / total_duration.as_secs_f64())
                    * 100.0;
            }
        }
    }

    /// Reset all metrics.
    pub async fn reset(&self) {
        let mut metrics = self.stage_metrics.write().await;
        metrics.clear();
        let mut active = self.active_timings.write().await;
        active.clear();
    }

    /// Get summary of pipeline performance.
    pub async fn get_summary(&self) -> PipelineSummary {
        let metrics = self.stage_metrics.read().await;

        let total_duration: Duration = metrics.values().map(|m| m.total_duration).sum();

        let total_executions: usize = metrics.values().map(|m| m.execution_count).sum();

        let bottlenecks: Vec<PipelineStage> = metrics
            .values()
            .filter(|m| m.is_bottleneck())
            .map(|m| m.stage)
            .collect();

        PipelineSummary {
            total_duration,
            total_executions,
            stage_count: metrics.len(),
            bottlenecks,
            avg_pipeline_duration: if total_executions > 0 {
                total_duration / total_executions.try_into().unwrap_or(1)
            } else {
                Duration::from_secs(0)
            },
        }
    }
}

impl Default for PipelineProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of pipeline performance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSummary {
    /// Total time spent across all stages
    pub total_duration: Duration,

    /// Total number of pipeline executions
    pub total_executions: usize,

    /// Number of unique stages profiled
    pub stage_count: usize,

    /// Detected bottleneck stages
    pub bottlenecks: Vec<PipelineStage>,

    /// Average time per complete pipeline execution
    pub avg_pipeline_duration: Duration,
}

impl fmt::Display for PipelineSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Pipeline Performance Summary:\n\
             - Total Executions: {}\n\
             - Total Duration: {:?}\n\
             - Avg Duration: {:?}\n\
             - Stages Profiled: {}\n\
             - Bottlenecks: {:?}",
            self.total_executions,
            self.total_duration,
            self.avg_pipeline_duration,
            self.stage_count,
            self.bottlenecks
                .iter()
                .map(|s| s.name())
                .collect::<Vec<_>>()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_stage_names() {
        assert_eq!(
            PipelineStage::TextPreprocessing.name(),
            "Text Preprocessing"
        );
        assert_eq!(PipelineStage::G2pConversion.name(), "G2P Conversion");
        assert_eq!(PipelineStage::AcousticModel.name(), "Acoustic Model");
    }

    #[test]
    fn test_stage_metrics_creation() {
        let metrics = StageMetrics::new(PipelineStage::Vocoder);
        assert_eq!(metrics.stage, PipelineStage::Vocoder);
        assert_eq!(metrics.execution_count, 0);
    }

    #[test]
    fn test_stage_metrics_recording() {
        let mut metrics = StageMetrics::new(PipelineStage::Vocoder);
        metrics.record_execution(Duration::from_millis(100), 100, 1000);

        assert_eq!(metrics.execution_count, 1);
        assert_eq!(metrics.total_duration, Duration::from_millis(100));
        assert_eq!(metrics.avg_input_size, 100.0);
        assert_eq!(metrics.avg_output_size, 1000.0);
    }

    #[tokio::test]
    async fn test_pipeline_profiler_creation() {
        let profiler = PipelineProfiler::new();
        let metrics = profiler.get_all_metrics().await;
        assert_eq!(metrics.len(), 0);
    }

    #[tokio::test]
    async fn test_pipeline_profiler_timing() {
        let profiler = PipelineProfiler::new();

        let timing_id = profiler.start_stage(PipelineStage::Vocoder).await;
        tokio::time::sleep(Duration::from_millis(10)).await;
        let duration = profiler.end_stage(&timing_id, 100, 1000).await;

        assert!(duration.is_some());
        assert!(duration.unwrap() >= Duration::from_millis(10));

        let metrics = profiler.get_stage_metrics(PipelineStage::Vocoder).await;
        assert!(metrics.is_some());
        assert_eq!(metrics.unwrap().execution_count, 1);
    }

    #[tokio::test]
    async fn test_pipeline_profiler_reset() {
        let profiler = PipelineProfiler::new();

        let timing_id = profiler.start_stage(PipelineStage::Vocoder).await;
        profiler.end_stage(&timing_id, 100, 1000).await;

        profiler.reset().await;

        let metrics = profiler.get_all_metrics().await;
        assert_eq!(metrics.len(), 0);
    }

    #[tokio::test]
    async fn test_pipeline_summary() {
        let profiler = PipelineProfiler::new();

        let timing_id = profiler.start_stage(PipelineStage::Vocoder).await;
        tokio::time::sleep(Duration::from_millis(10)).await;
        profiler.end_stage(&timing_id, 100, 1000).await;

        let summary = profiler.get_summary().await;
        assert_eq!(summary.total_executions, 1);
        assert_eq!(summary.stage_count, 1);
    }
}
