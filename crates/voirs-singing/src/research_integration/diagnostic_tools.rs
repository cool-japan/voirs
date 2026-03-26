// Diagnostic Tools - Phase 5 Feature
//
// Pipeline visualization, bottleneck identification, performance profiling,
// and comprehensive diagnostics for production systems.

use crate::Error;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Pipeline stage for visualization and analysis
#[derive(Debug, Clone)]
pub struct PipelineStage {
    pub id: String,
    pub name: String,
    pub stage_type: StageType,
    pub dependencies: Vec<String>,
    pub execution_time: Duration,
    pub memory_usage: usize,
    pub success_rate: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StageType {
    Input,
    Processing,
    Synthesis,
    PostProcessing,
    Output,
}

/// Pipeline visualization and analysis tool
#[derive(Debug, Clone)]
pub struct PipelineVisualizer {
    stages: Arc<RwLock<HashMap<String, PipelineStage>>>,
    execution_history: Arc<RwLock<Vec<ExecutionTrace>>>,
}

#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    pub timestamp: Instant,
    pub stage_id: String,
    pub duration: Duration,
    pub success: bool,
}

impl PipelineVisualizer {
    pub fn new() -> Self {
        Self {
            stages: Arc::new(RwLock::new(HashMap::new())),
            execution_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a pipeline stage
    pub fn register_stage(&self, stage: PipelineStage) -> Result<(), Error> {
        let mut stages = self
            .stages
            .write()
            .map_err(|_| Error::Processing("Failed to acquire stages lock".into()))?;
        stages.insert(stage.id.clone(), stage);
        Ok(())
    }

    /// Record stage execution
    pub fn record_execution(
        &self,
        stage_id: &str,
        duration: Duration,
        success: bool,
    ) -> Result<(), Error> {
        let mut history = self
            .execution_history
            .write()
            .map_err(|_| Error::Processing("Failed to acquire history lock".into()))?;

        history.push(ExecutionTrace {
            timestamp: Instant::now(),
            stage_id: stage_id.to_string(),
            duration,
            success,
        });

        // Update stage statistics
        let mut stages = self
            .stages
            .write()
            .map_err(|_| Error::Processing("Failed to acquire stages lock".into()))?;

        if let Some(stage) = stages.get_mut(stage_id) {
            stage.execution_time = duration;
        }

        Ok(())
    }

    /// Generate ASCII pipeline visualization
    pub fn visualize_ascii(&self) -> Result<String, Error> {
        let stages = self
            .stages
            .read()
            .map_err(|_| Error::Processing("Failed to read stages".into()))?;

        let mut output = String::from("Pipeline Visualization:\n\n");

        // Find input stages (no dependencies)
        let mut current_level: Vec<String> = stages
            .values()
            .filter(|s| s.dependencies.is_empty())
            .map(|s| s.id.clone())
            .collect();

        let mut level = 0;
        let mut processed = std::collections::HashSet::new();

        while !current_level.is_empty() {
            output.push_str(&format!("Level {}:\n", level));

            for stage_id in &current_level {
                if let Some(stage) = stages.get(stage_id) {
                    output.push_str(&format!(
                        "  [{}] {} ({:?}ms)\n",
                        stage.id,
                        stage.name,
                        stage.execution_time.as_millis()
                    ));
                    processed.insert(stage_id.clone());
                }
            }

            output.push('\n');

            // Find next level stages
            let mut next_level = Vec::new();
            for stage in stages.values() {
                if !processed.contains(&stage.id)
                    && stage.dependencies.iter().all(|dep| processed.contains(dep))
                {
                    next_level.push(stage.id.clone());
                }
            }

            current_level = next_level;
            level += 1;
        }

        Ok(output)
    }

    /// Generate Graphviz DOT format
    pub fn export_dot(&self) -> Result<String, Error> {
        let stages = self
            .stages
            .read()
            .map_err(|_| Error::Processing("Failed to read stages".into()))?;

        let mut dot = String::from("digraph Pipeline {\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  node [shape=box];\n\n");

        // Add nodes
        for stage in stages.values() {
            let color = match stage.stage_type {
                StageType::Input => "lightblue",
                StageType::Processing => "lightgreen",
                StageType::Synthesis => "yellow",
                StageType::PostProcessing => "orange",
                StageType::Output => "lightpink",
            };

            dot.push_str(&format!(
                "  \"{}\" [label=\"{}\\n{:?}ms\", style=filled, fillcolor={}];\n",
                stage.id,
                stage.name,
                stage.execution_time.as_millis(),
                color
            ));
        }

        dot.push('\n');

        // Add edges
        for stage in stages.values() {
            for dep in &stage.dependencies {
                dot.push_str(&format!("  \"{}\" -> \"{}\";\n", dep, stage.id));
            }
        }

        dot.push_str("}\n");
        Ok(dot)
    }

    /// Get pipeline statistics
    pub fn get_statistics(&self) -> Result<PipelineStatistics, Error> {
        let stages = self
            .stages
            .read()
            .map_err(|_| Error::Processing("Failed to read stages".into()))?;
        let history = self
            .execution_history
            .read()
            .map_err(|_| Error::Processing("Failed to read history".into()))?;

        let total_stages = stages.len();
        let total_executions = history.len();
        let successful_executions = history.iter().filter(|e| e.success).count();

        let total_time: Duration = stages.values().map(|s| s.execution_time).sum();

        let avg_stage_time = if total_stages > 0 {
            total_time / total_stages as u32
        } else {
            Duration::ZERO
        };

        Ok(PipelineStatistics {
            total_stages,
            total_executions,
            successful_executions,
            failed_executions: total_executions - successful_executions,
            total_pipeline_time: total_time,
            average_stage_time: avg_stage_time,
        })
    }

    /// Clear execution history
    pub fn clear_history(&self) -> Result<(), Error> {
        self.execution_history
            .write()
            .map_err(|_| Error::Processing("Failed to acquire history lock".into()))?
            .clear();
        Ok(())
    }
}

impl Default for PipelineVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct PipelineStatistics {
    pub total_stages: usize,
    pub total_executions: usize,
    pub successful_executions: usize,
    pub failed_executions: usize,
    pub total_pipeline_time: Duration,
    pub average_stage_time: Duration,
}

/// Bottleneck identification system
#[derive(Debug, Clone)]
pub struct BottleneckDetector {
    stage_metrics: Arc<RwLock<HashMap<String, StageMetrics>>>,
}

#[derive(Debug, Clone)]
pub struct StageMetrics {
    pub stage_id: String,
    pub total_time: Duration,
    pub execution_count: usize,
    pub average_time: Duration,
    pub max_time: Duration,
    pub min_time: Duration,
}

impl BottleneckDetector {
    pub fn new() -> Self {
        Self {
            stage_metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Record stage execution time
    pub fn record_stage_time(&self, stage_id: &str, duration: Duration) -> Result<(), Error> {
        let mut metrics = self
            .stage_metrics
            .write()
            .map_err(|_| Error::Processing("Failed to acquire metrics lock".into()))?;

        let stage_metrics = metrics.entry(stage_id.to_string()).or_insert(StageMetrics {
            stage_id: stage_id.to_string(),
            total_time: Duration::ZERO,
            execution_count: 0,
            average_time: Duration::ZERO,
            max_time: Duration::ZERO,
            min_time: Duration::MAX,
        });

        stage_metrics.total_time += duration;
        stage_metrics.execution_count += 1;
        stage_metrics.average_time =
            stage_metrics.total_time / stage_metrics.execution_count as u32;
        stage_metrics.max_time = stage_metrics.max_time.max(duration);
        stage_metrics.min_time = stage_metrics.min_time.min(duration);

        Ok(())
    }

    /// Identify bottlenecks (stages taking >80% of average time)
    pub fn identify_bottlenecks(&self) -> Result<Vec<BottleneckReport>, Error> {
        let metrics = self
            .stage_metrics
            .read()
            .map_err(|_| Error::Processing("Failed to read metrics".into()))?;

        if metrics.is_empty() {
            return Ok(Vec::new());
        }

        let total_avg_time: Duration = metrics.values().map(|m| m.average_time).sum();
        let overall_avg = total_avg_time / metrics.len() as u32;

        let threshold = overall_avg.mul_f64(1.8); // 80% above average

        let mut bottlenecks: Vec<_> = metrics
            .values()
            .filter(|m| m.average_time > threshold)
            .map(|m| {
                let percentage =
                    (m.average_time.as_secs_f64() / total_avg_time.as_secs_f64()) * 100.0;
                BottleneckReport {
                    stage_id: m.stage_id.clone(),
                    average_time: m.average_time,
                    percentage_of_total: percentage,
                    execution_count: m.execution_count,
                    severity: if m.average_time > threshold.mul_f64(2.0) {
                        BottleneckSeverity::Critical
                    } else if m.average_time > threshold.mul_f64(1.5) {
                        BottleneckSeverity::High
                    } else {
                        BottleneckSeverity::Medium
                    },
                }
            })
            .collect();

        bottlenecks.sort_by_key(|b| std::cmp::Reverse(b.average_time));

        Ok(bottlenecks)
    }

    /// Get all stage metrics
    pub fn get_all_metrics(&self) -> Result<Vec<StageMetrics>, Error> {
        let metrics = self
            .stage_metrics
            .read()
            .map_err(|_| Error::Processing("Failed to read metrics".into()))?;
        Ok(metrics.values().cloned().collect())
    }

    /// Reset all metrics
    pub fn reset(&self) -> Result<(), Error> {
        self.stage_metrics
            .write()
            .map_err(|_| Error::Processing("Failed to acquire metrics lock".into()))?
            .clear();
        Ok(())
    }
}

impl Default for BottleneckDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct BottleneckReport {
    pub stage_id: String,
    pub average_time: Duration,
    pub percentage_of_total: f64,
    pub execution_count: usize,
    pub severity: BottleneckSeverity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BottleneckSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Performance profiler
#[derive(Debug, Clone)]
pub struct PerformanceProfiler {
    profiles: Arc<RwLock<HashMap<String, ProfileData>>>,
}

#[derive(Debug, Clone)]
pub struct ProfileData {
    pub operation: String,
    pub samples: Vec<Duration>,
    pub cpu_samples: Vec<f64>,
    pub memory_samples: Vec<usize>,
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            profiles: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start profiling an operation
    pub fn start_profile(&self, operation: &str) -> ProfileSession {
        ProfileSession {
            operation: operation.to_string(),
            start_time: Instant::now(),
            profiler: self.clone(),
        }
    }

    fn record_sample(
        &self,
        operation: &str,
        duration: Duration,
        cpu_usage: f64,
        memory_usage: usize,
    ) -> Result<(), Error> {
        let mut profiles = self
            .profiles
            .write()
            .map_err(|_| Error::Processing("Failed to acquire profiles lock".into()))?;

        let profile = profiles
            .entry(operation.to_string())
            .or_insert(ProfileData {
                operation: operation.to_string(),
                samples: Vec::new(),
                cpu_samples: Vec::new(),
                memory_samples: Vec::new(),
            });

        profile.samples.push(duration);
        profile.cpu_samples.push(cpu_usage);
        profile.memory_samples.push(memory_usage);

        Ok(())
    }

    /// Get profiling report
    pub fn get_report(&self, operation: &str) -> Result<ProfilingReport, Error> {
        let profiles = self
            .profiles
            .read()
            .map_err(|_| Error::Processing("Failed to read profiles".into()))?;

        let profile = profiles.get(operation).ok_or_else(|| {
            Error::Processing(format!("Profile not found for operation: {}", operation))
        })?;

        if profile.samples.is_empty() {
            return Ok(ProfilingReport::default());
        }

        let total_time: Duration = profile.samples.iter().sum();
        let avg_time = total_time / profile.samples.len() as u32;

        let mut sorted_samples = profile.samples.clone();
        sorted_samples.sort();

        let min_time = sorted_samples[0];
        let max_time = sorted_samples[sorted_samples.len() - 1];
        let p50 = sorted_samples[sorted_samples.len() / 2];
        let p95 = sorted_samples[sorted_samples.len() * 95 / 100];
        let p99 = sorted_samples[sorted_samples.len() * 99 / 100];

        let avg_cpu = if !profile.cpu_samples.is_empty() {
            profile.cpu_samples.iter().sum::<f64>() / profile.cpu_samples.len() as f64
        } else {
            0.0
        };

        let avg_memory = if !profile.memory_samples.is_empty() {
            profile.memory_samples.iter().sum::<usize>() / profile.memory_samples.len()
        } else {
            0
        };

        Ok(ProfilingReport {
            operation: operation.to_string(),
            sample_count: profile.samples.len(),
            total_time,
            average_time: avg_time,
            min_time,
            max_time,
            p50_time: p50,
            p95_time: p95,
            p99_time: p99,
            average_cpu_usage: avg_cpu,
            average_memory_usage: avg_memory,
        })
    }

    /// Get all profiling reports
    pub fn get_all_reports(&self) -> Result<Vec<ProfilingReport>, Error> {
        let profiles = self
            .profiles
            .read()
            .map_err(|_| Error::Processing("Failed to read profiles".into()))?;

        profiles.keys().map(|op| self.get_report(op)).collect()
    }

    /// Clear all profiling data
    pub fn clear(&self) -> Result<(), Error> {
        self.profiles
            .write()
            .map_err(|_| Error::Processing("Failed to acquire profiles lock".into()))?
            .clear();
        Ok(())
    }
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ProfileSession {
    operation: String,
    start_time: Instant,
    profiler: PerformanceProfiler,
}

impl ProfileSession {
    /// End profiling session
    pub fn end(self, cpu_usage: f64, memory_usage: usize) -> Result<(), Error> {
        let duration = self.start_time.elapsed();
        self.profiler
            .record_sample(&self.operation, duration, cpu_usage, memory_usage)
    }
}

#[derive(Debug, Clone)]
pub struct ProfilingReport {
    pub operation: String,
    pub sample_count: usize,
    pub total_time: Duration,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub p50_time: Duration,
    pub p95_time: Duration,
    pub p99_time: Duration,
    pub average_cpu_usage: f64,
    pub average_memory_usage: usize,
}

impl Default for ProfilingReport {
    fn default() -> Self {
        Self {
            operation: String::new(),
            sample_count: 0,
            total_time: Duration::ZERO,
            average_time: Duration::ZERO,
            min_time: Duration::ZERO,
            max_time: Duration::ZERO,
            p50_time: Duration::ZERO,
            p95_time: Duration::ZERO,
            p99_time: Duration::ZERO,
            average_cpu_usage: 0.0,
            average_memory_usage: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_visualizer_register_stage() {
        let visualizer = PipelineVisualizer::new();

        let stage = PipelineStage {
            id: "stage1".to_string(),
            name: "Input Stage".to_string(),
            stage_type: StageType::Input,
            dependencies: vec![],
            execution_time: Duration::from_millis(100),
            memory_usage: 1024,
            success_rate: 0.99,
        };

        visualizer.register_stage(stage).unwrap();

        let stats = visualizer.get_statistics().unwrap();
        assert_eq!(stats.total_stages, 1);
    }

    #[test]
    fn test_pipeline_visualizer_execution_tracking() {
        let visualizer = PipelineVisualizer::new();

        visualizer
            .record_execution("stage1", Duration::from_millis(50), true)
            .unwrap();
        visualizer
            .record_execution("stage1", Duration::from_millis(60), true)
            .unwrap();

        let stats = visualizer.get_statistics().unwrap();
        assert_eq!(stats.total_executions, 2);
        assert_eq!(stats.successful_executions, 2);
    }

    #[test]
    fn test_pipeline_visualizer_ascii_output() {
        let visualizer = PipelineVisualizer::new();

        let stage = PipelineStage {
            id: "input".to_string(),
            name: "Input".to_string(),
            stage_type: StageType::Input,
            dependencies: vec![],
            execution_time: Duration::from_millis(10),
            memory_usage: 512,
            success_rate: 1.0,
        };

        visualizer.register_stage(stage).unwrap();

        let ascii = visualizer.visualize_ascii().unwrap();
        assert!(ascii.contains("Pipeline Visualization"));
        assert!(ascii.contains("Input"));
    }

    #[test]
    fn test_pipeline_visualizer_dot_export() {
        let visualizer = PipelineVisualizer::new();

        let stage1 = PipelineStage {
            id: "s1".to_string(),
            name: "Stage 1".to_string(),
            stage_type: StageType::Processing,
            dependencies: vec![],
            execution_time: Duration::from_millis(20),
            memory_usage: 1024,
            success_rate: 0.95,
        };

        let stage2 = PipelineStage {
            id: "s2".to_string(),
            name: "Stage 2".to_string(),
            stage_type: StageType::Synthesis,
            dependencies: vec!["s1".to_string()],
            execution_time: Duration::from_millis(30),
            memory_usage: 2048,
            success_rate: 0.98,
        };

        visualizer.register_stage(stage1).unwrap();
        visualizer.register_stage(stage2).unwrap();

        let dot = visualizer.export_dot().unwrap();
        assert!(dot.contains("digraph Pipeline"));
        assert!(dot.contains("s1"));
        assert!(dot.contains("s2"));
        assert!(dot.contains("->"));
    }

    #[test]
    fn test_bottleneck_detector_recording() {
        let detector = BottleneckDetector::new();

        detector
            .record_stage_time("fast_stage", Duration::from_millis(10))
            .unwrap();
        detector
            .record_stage_time("slow_stage", Duration::from_millis(100))
            .unwrap();

        let metrics = detector.get_all_metrics().unwrap();
        assert_eq!(metrics.len(), 2);
    }

    #[test]
    fn test_bottleneck_detector_identification() {
        let detector = BottleneckDetector::new();

        // Normal stage
        for _ in 0..10 {
            detector
                .record_stage_time("normal", Duration::from_millis(10))
                .unwrap();
        }

        // Bottleneck stage
        for _ in 0..10 {
            detector
                .record_stage_time("bottleneck", Duration::from_millis(100))
                .unwrap();
        }

        let bottlenecks = detector.identify_bottlenecks().unwrap();
        assert!(!bottlenecks.is_empty());
        assert_eq!(bottlenecks[0].stage_id, "bottleneck");
    }

    #[test]
    fn test_performance_profiler_session() {
        let profiler = PerformanceProfiler::new();

        let session = profiler.start_profile("test_op");
        std::thread::sleep(Duration::from_millis(10));
        session.end(50.0, 1024 * 1024).unwrap();

        let report = profiler.get_report("test_op").unwrap();
        assert_eq!(report.sample_count, 1);
        assert!(report.average_time >= Duration::from_millis(10));
    }

    #[test]
    fn test_performance_profiler_statistics() {
        let profiler = PerformanceProfiler::new();

        for i in 0..5 {
            let session = profiler.start_profile("benchmark");
            std::thread::sleep(Duration::from_millis(10 + i));
            session
                .end(50.0 + i as f64, (1024 * (i + 1)) as usize)
                .unwrap();
        }

        let report = profiler.get_report("benchmark").unwrap();
        assert_eq!(report.sample_count, 5);
        assert!(report.min_time < report.max_time);
        assert!(report.p50_time > Duration::ZERO);
    }

    #[test]
    fn test_performance_profiler_multiple_operations() {
        let profiler = PerformanceProfiler::new();

        profiler.start_profile("op1").end(25.0, 1024).unwrap();
        profiler.start_profile("op2").end(75.0, 2048).unwrap();

        let reports = profiler.get_all_reports().unwrap();
        assert_eq!(reports.len(), 2);
    }

    #[test]
    fn test_bottleneck_severity_levels() {
        let detector = BottleneckDetector::new();

        // Add stages with varying performance
        detector
            .record_stage_time("fast", Duration::from_millis(5))
            .unwrap();
        detector
            .record_stage_time("medium", Duration::from_millis(50))
            .unwrap();
        detector
            .record_stage_time("slow", Duration::from_millis(200))
            .unwrap();
        detector
            .record_stage_time("critical", Duration::from_millis(500))
            .unwrap();

        let bottlenecks = detector.identify_bottlenecks().unwrap();

        if !bottlenecks.is_empty() {
            // Most severe bottleneck should be first
            assert_eq!(bottlenecks[0].stage_id, "critical");
        }
    }

    #[test]
    fn test_pipeline_statistics_comprehensive() {
        let visualizer = PipelineVisualizer::new();

        for i in 0..5 {
            let stage = PipelineStage {
                id: format!("stage{}", i),
                name: format!("Stage {}", i),
                stage_type: StageType::Processing,
                dependencies: if i > 0 {
                    vec![format!("stage{}", i - 1)]
                } else {
                    vec![]
                },
                execution_time: Duration::from_millis(10 * (i + 1) as u64),
                memory_usage: 1024 * (i + 1),
                success_rate: 0.95,
            };
            visualizer.register_stage(stage).unwrap();
        }

        let stats = visualizer.get_statistics().unwrap();
        assert_eq!(stats.total_stages, 5);
        assert!(stats.total_pipeline_time > Duration::ZERO);
    }
}
