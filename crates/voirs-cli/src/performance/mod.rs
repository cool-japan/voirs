//! Performance profiling and optimization module
//!
//! This module provides comprehensive performance monitoring, profiling, and optimization
//! recommendations for VoiRS synthesis operations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

pub mod memory_optimizer;
pub mod metrics;
pub mod monitor;
pub mod optimizer;
pub mod profiler;
pub mod streaming_optimizer;

/// Performance metrics collector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// System resource usage
    pub system: SystemMetrics,
    /// Synthesis operation metrics
    pub synthesis: SynthesisMetrics,
    /// Memory usage tracking
    pub memory: MemoryMetrics,
    /// GPU utilization if available
    pub gpu: Option<GpuMetrics>,
    /// Timestamp of metrics collection
    pub timestamp: u64,
}

/// System-level performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// CPU usage percentage (0.0 - 100.0)
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_used: u64,
    /// Available memory in bytes
    pub memory_available: u64,
    /// Disk I/O read bytes per second
    pub disk_read_bps: u64,
    /// Disk I/O write bytes per second
    pub disk_write_bps: u64,
    /// Network I/O bytes per second
    pub network_bps: u64,
    /// Number of active threads
    pub thread_count: usize,
    /// Load average (Unix systems)
    pub load_average: Option<f64>,
}

/// Synthesis operation performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisMetrics {
    /// Total synthesis operations
    pub total_operations: u64,
    /// Successful operations
    pub successful_operations: u64,
    /// Failed operations
    pub failed_operations: u64,
    /// Average synthesis time in milliseconds
    pub avg_synthesis_time_ms: f64,
    /// Total audio duration generated in seconds
    pub total_audio_duration: f64,
    /// Real-time factor (audio_duration / synthesis_time)
    pub real_time_factor: f64,
    /// Throughput (characters per second)
    pub throughput_chars_per_sec: f64,
    /// Queue depth (pending operations)
    pub queue_depth: usize,
    /// Memory usage per operation in MB
    pub memory_per_operation_mb: f64,
}

/// Memory usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Heap memory usage in bytes
    pub heap_used: u64,
    /// Peak memory usage in bytes
    pub peak_usage: u64,
    /// Memory allocations per second
    pub allocations_per_sec: f64,
    /// Memory deallocations per second
    pub deallocations_per_sec: f64,
    /// Garbage collection events
    pub gc_events: u64,
    /// Memory fragmentation percentage
    pub fragmentation_percent: f64,
    /// Cache hit rate percentage
    pub cache_hit_rate: f64,
}

/// GPU utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    /// GPU usage percentage
    pub utilization: f64,
    /// GPU memory used in bytes
    pub memory_used: u64,
    /// GPU memory total in bytes
    pub memory_total: u64,
    /// GPU temperature in Celsius
    pub temperature: f64,
    /// GPU power consumption in watts
    pub power_consumption: f64,
    /// GPU compute units active
    pub compute_units_active: usize,
    /// GPU memory bandwidth utilization
    pub memory_bandwidth_util: f64,
}

/// Performance optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Recommendation category
    pub category: OptimizationCategory,
    /// Priority level (1-10, 10 being highest)
    pub priority: u8,
    /// Description of the issue
    pub description: String,
    /// Recommended action
    pub recommendation: String,
    /// Expected improvement
    pub expected_improvement: String,
    /// Implementation difficulty (1-5, 5 being hardest)
    pub difficulty: u8,
    /// Performance impact estimate
    pub performance_impact: f64,
}

/// Categories of performance optimizations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationCategory {
    /// Memory usage optimizations
    Memory,
    /// CPU utilization improvements
    Cpu,
    /// GPU acceleration opportunities
    Gpu,
    /// I/O operation optimizations
    Io,
    /// Network efficiency improvements
    Network,
    /// Caching strategy enhancements
    Caching,
    /// Parallel processing optimizations
    Parallelization,
    /// Model optimization suggestions
    ModelOptimization,
    /// Configuration tuning
    Configuration,
    /// System resource allocation
    ResourceAllocation,
}

/// Performance profiler for monitoring and analysis
pub struct PerformanceProfiler {
    /// Metrics storage
    metrics_history: Arc<RwLock<Vec<PerformanceMetrics>>>,
    /// Operation timings
    operation_timings: Arc<RwLock<HashMap<String, Vec<Duration>>>>,
    /// Optimization recommendations cache
    recommendations: Arc<RwLock<Vec<OptimizationRecommendation>>>,
    /// Profiler start time
    start_time: Instant,
    /// Is profiling enabled
    enabled: bool,
    /// Maximum history size
    max_history_size: usize,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new(enabled: bool, max_history_size: usize) -> Self {
        Self {
            metrics_history: Arc::new(RwLock::new(Vec::with_capacity(max_history_size))),
            operation_timings: Arc::new(RwLock::new(HashMap::new())),
            recommendations: Arc::new(RwLock::new(Vec::new())),
            start_time: Instant::now(),
            enabled,
            max_history_size,
        }
    }

    /// Record performance metrics
    pub async fn record_metrics(&self, metrics: PerformanceMetrics) {
        if !self.enabled {
            return;
        }

        let mut history = self.metrics_history.write().await;

        // Maintain maximum history size
        if history.len() >= self.max_history_size {
            history.remove(0);
        }

        history.push(metrics);
    }

    /// Start timing an operation
    pub async fn start_operation(&self, operation_name: &str) -> OperationTimer {
        if !self.enabled {
            return OperationTimer::disabled();
        }

        OperationTimer::new(
            operation_name.to_string(),
            self.operation_timings.clone(),
            Instant::now(),
        )
    }

    /// Get performance metrics summary
    pub async fn get_metrics_summary(&self) -> Option<PerformanceMetrics> {
        if !self.enabled {
            return None;
        }

        let history = self.metrics_history.read().await;
        history.last().cloned()
    }

    /// Get operation timing statistics
    pub async fn get_timing_stats(&self, operation_name: &str) -> Option<TimingStats> {
        if !self.enabled {
            return None;
        }

        let timings = self.operation_timings.read().await;
        timings
            .get(operation_name)
            .map(|durations| TimingStats::from_durations(durations))
    }

    /// Generate optimization recommendations
    pub async fn generate_recommendations(&self) -> Vec<OptimizationRecommendation> {
        if !self.enabled {
            return Vec::new();
        }

        let mut recommendations = Vec::new();
        let history = self.metrics_history.read().await;

        if history.is_empty() {
            return recommendations;
        }

        // Analyze recent metrics for optimization opportunities
        let recent_metrics = &history[history.len().saturating_sub(10)..];

        // Memory optimization checks
        self.check_memory_optimizations(&mut recommendations, recent_metrics)
            .await;

        // CPU optimization checks
        self.check_cpu_optimizations(&mut recommendations, recent_metrics)
            .await;

        // GPU optimization checks
        self.check_gpu_optimizations(&mut recommendations, recent_metrics)
            .await;

        // I/O optimization checks
        self.check_io_optimizations(&mut recommendations, recent_metrics)
            .await;

        // Sort by priority
        recommendations.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Cache recommendations
        let mut cached_recommendations = self.recommendations.write().await;
        *cached_recommendations = recommendations.clone();

        recommendations
    }

    /// Check for memory optimization opportunities
    async fn check_memory_optimizations(
        &self,
        recommendations: &mut Vec<OptimizationRecommendation>,
        metrics: &[PerformanceMetrics],
    ) {
        let avg_memory_usage = metrics
            .iter()
            .map(|m| m.memory.heap_used as f64)
            .sum::<f64>()
            / metrics.len() as f64;

        let total_memory = metrics
            .iter()
            .map(|m| m.system.memory_used + m.system.memory_available)
            .max()
            .unwrap_or(0) as f64;

        let memory_usage_percent = (avg_memory_usage / total_memory) * 100.0;

        if memory_usage_percent > 80.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Memory,
                priority: 9,
                description: format!("High memory usage detected: {:.1}%", memory_usage_percent),
                recommendation: "Consider enabling memory optimization flags, reducing batch sizes, or using streaming processing for large texts".to_string(),
                expected_improvement: "20-40% reduction in memory usage".to_string(),
                difficulty: 2,
                performance_impact: 0.3,
            });
        }

        // Check for memory fragmentation
        let avg_fragmentation = metrics
            .iter()
            .map(|m| m.memory.fragmentation_percent)
            .sum::<f64>()
            / metrics.len() as f64;

        if avg_fragmentation > 15.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Memory,
                priority: 6,
                description: format!("Memory fragmentation detected: {:.1}%", avg_fragmentation),
                recommendation:
                    "Enable memory pool allocation or restart the application periodically"
                        .to_string(),
                expected_improvement: "10-20% improvement in memory efficiency".to_string(),
                difficulty: 3,
                performance_impact: 0.15,
            });
        }

        // Check cache hit rate
        let avg_cache_hit_rate =
            metrics.iter().map(|m| m.memory.cache_hit_rate).sum::<f64>() / metrics.len() as f64;

        if avg_cache_hit_rate < 70.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Caching,
                priority: 7,
                description: format!("Low cache hit rate: {:.1}%", avg_cache_hit_rate),
                recommendation: "Increase cache size, implement more aggressive caching, or use model preloading".to_string(),
                expected_improvement: "15-30% improvement in synthesis speed".to_string(),
                difficulty: 3,
                performance_impact: 0.25,
            });
        }
    }

    /// Check for CPU optimization opportunities
    async fn check_cpu_optimizations(
        &self,
        recommendations: &mut Vec<OptimizationRecommendation>,
        metrics: &[PerformanceMetrics],
    ) {
        let avg_cpu_usage =
            metrics.iter().map(|m| m.system.cpu_usage).sum::<f64>() / metrics.len() as f64;

        if avg_cpu_usage > 90.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Cpu,
                priority: 8,
                description: format!("High CPU usage detected: {:.1}%", avg_cpu_usage),
                recommendation: "Enable GPU acceleration, reduce parallel processing threads, or use lower quality settings".to_string(),
                expected_improvement: "30-50% reduction in CPU usage".to_string(),
                difficulty: 2,
                performance_impact: 0.4,
            });
        } else if avg_cpu_usage < 30.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Parallelization,
                priority: 5,
                description: format!("Low CPU utilization: {:.1}%", avg_cpu_usage),
                recommendation: "Increase parallel processing threads or batch size to better utilize available CPU cores".to_string(),
                expected_improvement: "20-40% improvement in throughput".to_string(),
                difficulty: 2,
                performance_impact: 0.3,
            });
        }

        // Check real-time factor
        let avg_rtf = metrics
            .iter()
            .map(|m| m.synthesis.real_time_factor)
            .sum::<f64>()
            / metrics.len() as f64;

        if avg_rtf < 1.0 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::ModelOptimization,
                priority: 8,
                description: format!("Poor real-time factor: {:.2}x", avg_rtf),
                recommendation: "Use quantized models, enable GPU acceleration, or reduce quality settings for real-time applications".to_string(),
                expected_improvement: "Achieve real-time synthesis (>1.0x RTF)".to_string(),
                difficulty: 4,
                performance_impact: 0.5,
            });
        }
    }

    /// Check for GPU optimization opportunities
    async fn check_gpu_optimizations(
        &self,
        recommendations: &mut Vec<OptimizationRecommendation>,
        metrics: &[PerformanceMetrics],
    ) {
        let gpu_available = metrics.iter().any(|m| m.gpu.is_some());

        if !gpu_available {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Gpu,
                priority: 6,
                description: "GPU acceleration not detected".to_string(),
                recommendation: "Enable GPU acceleration if available, or consider using cloud GPU instances for large workloads".to_string(),
                expected_improvement: "2-10x improvement in synthesis speed".to_string(),
                difficulty: 3,
                performance_impact: 0.8,
            });
            return;
        }

        // Analyze GPU utilization
        let gpu_metrics: Vec<&GpuMetrics> = metrics.iter().filter_map(|m| m.gpu.as_ref()).collect();

        if !gpu_metrics.is_empty() {
            let avg_gpu_utilization =
                gpu_metrics.iter().map(|g| g.utilization).sum::<f64>() / gpu_metrics.len() as f64;

            if avg_gpu_utilization < 30.0 {
                recommendations.push(OptimizationRecommendation {
                    category: OptimizationCategory::Gpu,
                    priority: 7,
                    description: format!("Low GPU utilization: {:.1}%", avg_gpu_utilization),
                    recommendation: "Increase batch size, use larger models, or enable more GPU-accelerated features".to_string(),
                    expected_improvement: "Better GPU utilization and potentially faster processing".to_string(),
                    difficulty: 2,
                    performance_impact: 0.3,
                });
            }

            // Check GPU memory usage
            let avg_gpu_memory_usage = gpu_metrics
                .iter()
                .map(|g| (g.memory_used as f64 / g.memory_total as f64) * 100.0)
                .sum::<f64>()
                / gpu_metrics.len() as f64;

            if avg_gpu_memory_usage > 85.0 {
                recommendations.push(OptimizationRecommendation {
                    category: OptimizationCategory::Gpu,
                    priority: 8,
                    description: format!("High GPU memory usage: {:.1}%", avg_gpu_memory_usage),
                    recommendation: "Reduce batch size, use model quantization, or enable gradient checkpointing".to_string(),
                    expected_improvement: "Prevent GPU memory overflow and improve stability".to_string(),
                    difficulty: 3,
                    performance_impact: 0.2,
                });
            }
        }
    }

    /// Check for I/O optimization opportunities
    async fn check_io_optimizations(
        &self,
        recommendations: &mut Vec<OptimizationRecommendation>,
        metrics: &[PerformanceMetrics],
    ) {
        let avg_disk_read =
            metrics.iter().map(|m| m.system.disk_read_bps).sum::<u64>() / metrics.len() as u64;

        let avg_disk_write =
            metrics.iter().map(|m| m.system.disk_write_bps).sum::<u64>() / metrics.len() as u64;

        // Check for high I/O usage
        if avg_disk_read > 100_000_000 || avg_disk_write > 100_000_000 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Io,
                priority: 6,
                description: format!(
                    "High disk I/O: {:.1} MB/s read, {:.1} MB/s write",
                    avg_disk_read as f64 / 1_000_000.0,
                    avg_disk_write as f64 / 1_000_000.0
                ),
                recommendation:
                    "Use SSD storage, enable I/O caching, or process files in memory when possible"
                        .to_string(),
                expected_improvement: "20-50% reduction in I/O bottlenecks".to_string(),
                difficulty: 3,
                performance_impact: 0.3,
            });
        }

        // Check operation queue depth
        let avg_queue_depth = metrics
            .iter()
            .map(|m| m.synthesis.queue_depth)
            .sum::<usize>()
            / metrics.len();

        if avg_queue_depth > 10 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::ResourceAllocation,
                priority: 7,
                description: format!("High operation queue depth: {}", avg_queue_depth),
                recommendation: "Increase worker threads, enable parallel processing, or optimize resource allocation".to_string(),
                expected_improvement: "Reduced latency and better throughput".to_string(),
                difficulty: 2,
                performance_impact: 0.25,
            });
        }
    }

    /// Get profiler uptime
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Enable or disable profiling
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Clear metrics history
    pub async fn clear_history(&self) {
        let mut history = self.metrics_history.write().await;
        history.clear();

        let mut timings = self.operation_timings.write().await;
        timings.clear();

        let mut recommendations = self.recommendations.write().await;
        recommendations.clear();
    }
}

/// Timer for tracking operation performance
pub struct OperationTimer {
    operation_name: String,
    timings_store: Option<Arc<RwLock<HashMap<String, Vec<Duration>>>>>,
    start_time: Instant,
}

impl OperationTimer {
    fn new(
        operation_name: String,
        timings_store: Arc<RwLock<HashMap<String, Vec<Duration>>>>,
        start_time: Instant,
    ) -> Self {
        Self {
            operation_name,
            timings_store: Some(timings_store),
            start_time,
        }
    }

    fn disabled() -> Self {
        Self {
            operation_name: String::new(),
            timings_store: None,
            start_time: Instant::now(),
        }
    }

    /// Stop timing and record the duration
    pub async fn stop(self) -> Duration {
        let duration = self.start_time.elapsed();

        if let Some(timings_store) = self.timings_store {
            let mut timings = timings_store.write().await;
            timings
                .entry(self.operation_name)
                .or_insert_with(Vec::new)
                .push(duration);
        }

        duration
    }
}

/// Statistical summary of operation timings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStats {
    /// Number of samples
    pub count: usize,
    /// Average duration
    pub average: Duration,
    /// Minimum duration
    pub minimum: Duration,
    /// Maximum duration
    pub maximum: Duration,
    /// 95th percentile
    pub p95: Duration,
    /// 99th percentile
    pub p99: Duration,
    /// Standard deviation
    pub std_dev: Duration,
}

impl TimingStats {
    fn from_durations(durations: &[Duration]) -> Self {
        if durations.is_empty() {
            return Self {
                count: 0,
                average: Duration::ZERO,
                minimum: Duration::ZERO,
                maximum: Duration::ZERO,
                p95: Duration::ZERO,
                p99: Duration::ZERO,
                std_dev: Duration::ZERO,
            };
        }

        let mut sorted = durations.to_vec();
        sorted.sort();

        let count = sorted.len();
        let sum: Duration = sorted.iter().sum();
        let average = sum / count as u32;

        let minimum = sorted[0];
        let maximum = sorted[count - 1];

        let p95_index = (count as f64 * 0.95) as usize;
        let p99_index = (count as f64 * 0.99) as usize;
        let p95 = sorted[p95_index.min(count - 1)];
        let p99 = sorted[p99_index.min(count - 1)];

        // Calculate standard deviation
        let variance: f64 = durations
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - average.as_secs_f64();
                diff * diff
            })
            .sum::<f64>()
            / count as f64;

        let std_dev = Duration::from_secs_f64(variance.sqrt());

        Self {
            count,
            average,
            minimum,
            maximum,
            p95,
            p99,
            std_dev,
        }
    }
}

/// Default implementation for performance metrics
impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            system: SystemMetrics::default(),
            synthesis: SynthesisMetrics::default(),
            memory: MemoryMetrics::default(),
            gpu: None,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_used: 0,
            memory_available: 0,
            disk_read_bps: 0,
            disk_write_bps: 0,
            network_bps: 0,
            thread_count: 0,
            load_average: None,
        }
    }
}

impl Default for SynthesisMetrics {
    fn default() -> Self {
        Self {
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            avg_synthesis_time_ms: 0.0,
            total_audio_duration: 0.0,
            real_time_factor: 0.0,
            throughput_chars_per_sec: 0.0,
            queue_depth: 0,
            memory_per_operation_mb: 0.0,
        }
    }
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        Self {
            heap_used: 0,
            peak_usage: 0,
            allocations_per_sec: 0.0,
            deallocations_per_sec: 0.0,
            gc_events: 0,
            fragmentation_percent: 0.0,
            cache_hit_rate: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_performance_profiler_creation() {
        let profiler = PerformanceProfiler::new(true, 100);
        assert!(profiler.enabled);
        assert_eq!(profiler.max_history_size, 100);
    }

    #[tokio::test]
    async fn test_metrics_recording() {
        let profiler = PerformanceProfiler::new(true, 10);
        let metrics = PerformanceMetrics::default();

        profiler.record_metrics(metrics.clone()).await;

        let summary = profiler.get_metrics_summary().await;
        assert!(summary.is_some());
    }

    #[tokio::test]
    async fn test_operation_timing() {
        let profiler = PerformanceProfiler::new(true, 10);

        let timer = profiler.start_operation("test_operation").await;
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        let duration = timer.stop().await;

        assert!(duration >= Duration::from_millis(10));

        let stats = profiler.get_timing_stats("test_operation").await;
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().count, 1);
    }

    #[tokio::test]
    async fn test_timing_stats_calculation() {
        let durations = vec![
            Duration::from_millis(100),
            Duration::from_millis(200),
            Duration::from_millis(300),
            Duration::from_millis(400),
            Duration::from_millis(500),
        ];

        let stats = TimingStats::from_durations(&durations);

        assert_eq!(stats.count, 5);
        assert_eq!(stats.average, Duration::from_millis(300));
        assert_eq!(stats.minimum, Duration::from_millis(100));
        assert_eq!(stats.maximum, Duration::from_millis(500));
    }

    #[tokio::test]
    async fn test_recommendations_generation() {
        let profiler = PerformanceProfiler::new(true, 10);

        // Add some metrics that should trigger recommendations
        let mut metrics = PerformanceMetrics::default();
        metrics.system.cpu_usage = 95.0; // High CPU usage
        metrics.memory.cache_hit_rate = 50.0; // Low cache hit rate

        profiler.record_metrics(metrics).await;

        let recommendations = profiler.generate_recommendations().await;
        assert!(!recommendations.is_empty());

        // Should have CPU and caching recommendations
        assert!(recommendations
            .iter()
            .any(|r| r.category == OptimizationCategory::Cpu));
        assert!(recommendations
            .iter()
            .any(|r| r.category == OptimizationCategory::Caching));
    }

    #[test]
    fn test_optimization_category_serialization() {
        let category = OptimizationCategory::Memory;
        let serialized = serde_json::to_string(&category).unwrap();
        let deserialized: OptimizationCategory = serde_json::from_str(&serialized).unwrap();
        assert_eq!(category, deserialized);
    }

    #[tokio::test]
    async fn test_disabled_profiler() {
        let profiler = PerformanceProfiler::new(false, 10);
        let metrics = PerformanceMetrics::default();

        profiler.record_metrics(metrics).await;

        let summary = profiler.get_metrics_summary().await;
        assert!(summary.is_none());

        let recommendations = profiler.generate_recommendations().await;
        assert!(recommendations.is_empty());
    }
}
