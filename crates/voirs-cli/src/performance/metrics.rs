//! Performance metrics collection and analysis
//!
//! This module provides comprehensive metrics collection, aggregation, and analysis
//! for performance monitoring and optimization of VoiRS synthesis operations.

use super::{GpuMetrics, MemoryMetrics, PerformanceMetrics, SynthesisMetrics, SystemMetrics};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

/// Metrics aggregation window types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricsWindow {
    /// Last 1 minute
    OneMinute,
    /// Last 5 minutes
    FiveMinutes,
    /// Last 15 minutes
    FifteenMinutes,
    /// Last 1 hour
    OneHour,
    /// Last 24 hours
    TwentyFourHours,
    /// All time
    AllTime,
}

/// Aggregated performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    /// Time window for aggregation
    pub window: MetricsWindow,
    /// Number of samples included
    pub sample_count: usize,
    /// Time range of samples
    pub time_range: TimeRange,
    /// System metrics summary
    pub system: SystemSummary,
    /// Synthesis metrics summary
    pub synthesis: SynthesisSummary,
    /// Memory metrics summary
    pub memory: MemorySummary,
    /// GPU metrics summary (if available)
    pub gpu: Option<GpuSummary>,
    /// Performance trends
    pub trends: PerformanceTrends,
}

/// Time range for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start timestamp
    pub start: u64,
    /// End timestamp
    pub end: u64,
    /// Duration in seconds
    pub duration_seconds: u64,
}

/// System metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemSummary {
    /// CPU usage statistics
    pub cpu: StatisticsSummary,
    /// Memory usage statistics
    pub memory_used: StatisticsSummary,
    /// Memory available statistics
    pub memory_available: StatisticsSummary,
    /// Disk I/O read statistics
    pub disk_read: StatisticsSummary,
    /// Disk I/O write statistics
    pub disk_write: StatisticsSummary,
    /// Network I/O statistics
    pub network: StatisticsSummary,
    /// Thread count statistics
    pub thread_count: StatisticsSummary,
    /// Load average statistics (Unix only)
    pub load_average: Option<StatisticsSummary>,
}

/// Synthesis metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisSummary {
    /// Total operations in window
    pub total_operations: u64,
    /// Success rate percentage
    pub success_rate: f64,
    /// Synthesis time statistics
    pub synthesis_time: StatisticsSummary,
    /// Real-time factor statistics
    pub real_time_factor: StatisticsSummary,
    /// Throughput statistics (chars/sec)
    pub throughput: StatisticsSummary,
    /// Queue depth statistics
    pub queue_depth: StatisticsSummary,
    /// Memory per operation statistics
    pub memory_per_operation: StatisticsSummary,
    /// Total audio duration generated
    pub total_audio_duration: f64,
}

/// Memory metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySummary {
    /// Heap usage statistics
    pub heap_used: StatisticsSummary,
    /// Peak usage statistics
    pub peak_usage: StatisticsSummary,
    /// Allocation rate statistics
    pub allocation_rate: StatisticsSummary,
    /// Deallocation rate statistics
    pub deallocation_rate: StatisticsSummary,
    /// Fragmentation statistics
    pub fragmentation: StatisticsSummary,
    /// Cache hit rate statistics
    pub cache_hit_rate: StatisticsSummary,
    /// GC events count
    pub gc_events: u64,
}

/// GPU metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSummary {
    /// GPU utilization statistics
    pub utilization: StatisticsSummary,
    /// GPU memory usage statistics
    pub memory_used: StatisticsSummary,
    /// GPU memory usage percentage statistics
    pub memory_usage_percent: StatisticsSummary,
    /// GPU temperature statistics
    pub temperature: StatisticsSummary,
    /// GPU power consumption statistics
    pub power_consumption: StatisticsSummary,
    /// Compute units active statistics
    pub compute_units: StatisticsSummary,
    /// Memory bandwidth utilization statistics
    pub memory_bandwidth: StatisticsSummary,
}

/// Statistical summary for a metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticsSummary {
    /// Average value
    pub average: f64,
    /// Minimum value
    pub minimum: f64,
    /// Maximum value
    pub maximum: f64,
    /// Standard deviation
    pub std_deviation: f64,
    /// 50th percentile (median)
    pub p50: f64,
    /// 90th percentile
    pub p90: f64,
    /// 95th percentile
    pub p95: f64,
    /// 99th percentile
    pub p99: f64,
    /// Sample count
    pub count: usize,
}

/// Performance trends analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    /// CPU usage trend (positive = increasing)
    pub cpu_trend: TrendDirection,
    /// Memory usage trend
    pub memory_trend: TrendDirection,
    /// Synthesis performance trend
    pub synthesis_performance_trend: TrendDirection,
    /// Queue depth trend
    pub queue_depth_trend: TrendDirection,
    /// Error rate trend
    pub error_rate_trend: TrendDirection,
    /// Overall performance score trend
    pub overall_trend: TrendDirection,
}

/// Trend direction indicator
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Strongly improving
    StronglyImproving,
    /// Improving
    Improving,
    /// Stable
    Stable,
    /// Degrading
    Degrading,
    /// Strongly degrading
    StronglyDegrading,
    /// Insufficient data
    Unknown,
}

/// Performance metrics collector and analyzer
pub struct MetricsCollector {
    /// Raw metrics storage
    raw_metrics: Arc<RwLock<VecDeque<PerformanceMetrics>>>,
    /// Aggregated metrics cache
    aggregated_cache: Arc<RwLock<HashMap<MetricsWindow, AggregatedMetrics>>>,
    /// Maximum raw metrics to store
    max_raw_metrics: usize,
    /// Last cache update time
    last_cache_update: Arc<RwLock<Instant>>,
    /// Cache validity duration
    cache_validity: Duration,
    /// Metrics collection start time
    start_time: Instant,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new(max_raw_metrics: usize, cache_validity: Duration) -> Self {
        Self {
            raw_metrics: Arc::new(RwLock::new(VecDeque::with_capacity(max_raw_metrics))),
            aggregated_cache: Arc::new(RwLock::new(HashMap::new())),
            max_raw_metrics,
            last_cache_update: Arc::new(RwLock::new(Instant::now())),
            cache_validity,
            start_time: Instant::now(),
        }
    }

    /// Add a new performance metrics sample
    pub async fn add_metrics(&self, metrics: PerformanceMetrics) {
        let mut raw_metrics = self.raw_metrics.write().await;

        // Maintain maximum size
        if raw_metrics.len() >= self.max_raw_metrics {
            raw_metrics.pop_front();
        }

        raw_metrics.push_back(metrics);

        // Invalidate cache
        self.invalidate_cache().await;
    }

    /// Get aggregated metrics for a specific window
    pub async fn get_aggregated_metrics(&self, window: MetricsWindow) -> Option<AggregatedMetrics> {
        // Check cache first
        if let Some(cached) = self.get_cached_metrics(window).await {
            return Some(cached);
        }

        // Generate new aggregated metrics
        let aggregated = self.generate_aggregated_metrics(window).await?;

        // Cache the result
        self.cache_metrics(window, aggregated.clone()).await;

        Some(aggregated)
    }

    /// Get performance trends for a specific window
    pub async fn get_performance_trends(&self, window: MetricsWindow) -> Option<PerformanceTrends> {
        let aggregated = self.get_aggregated_metrics(window).await?;
        Some(aggregated.trends)
    }

    /// Get real-time metrics (latest sample)
    pub async fn get_latest_metrics(&self) -> Option<PerformanceMetrics> {
        let raw_metrics = self.raw_metrics.read().await;
        raw_metrics.back().cloned()
    }

    /// Get metrics history for a specific time range
    pub async fn get_metrics_history(
        &self,
        start_time: u64,
        end_time: u64,
    ) -> Vec<PerformanceMetrics> {
        let raw_metrics = self.raw_metrics.read().await;

        raw_metrics
            .iter()
            .filter(|m| m.timestamp >= start_time && m.timestamp <= end_time)
            .cloned()
            .collect()
    }

    /// Generate performance report
    pub async fn generate_performance_report(&self) -> PerformanceReport {
        let mut report = PerformanceReport {
            generation_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            uptime_seconds: self.start_time.elapsed().as_secs(),
            windows: HashMap::new(),
            summary: ReportSummary::default(),
        };

        // Generate metrics for all windows
        for &window in &[
            MetricsWindow::OneMinute,
            MetricsWindow::FiveMinutes,
            MetricsWindow::FifteenMinutes,
            MetricsWindow::OneHour,
            MetricsWindow::TwentyFourHours,
        ] {
            if let Some(metrics) = self.get_aggregated_metrics(window).await {
                report.windows.insert(window, metrics);
            }
        }

        // Generate summary
        report.summary = self.generate_report_summary(&report.windows).await;

        report
    }

    /// Check cached metrics
    async fn get_cached_metrics(&self, window: MetricsWindow) -> Option<AggregatedMetrics> {
        let cache = self.aggregated_cache.read().await;
        let last_update = *self.last_cache_update.read().await;

        if last_update.elapsed() < self.cache_validity {
            cache.get(&window).cloned()
        } else {
            None
        }
    }

    /// Cache aggregated metrics
    async fn cache_metrics(&self, window: MetricsWindow, metrics: AggregatedMetrics) {
        let mut cache = self.aggregated_cache.write().await;
        cache.insert(window, metrics);

        let mut last_update = self.last_cache_update.write().await;
        *last_update = Instant::now();
    }

    /// Invalidate cache
    async fn invalidate_cache(&self) {
        let mut cache = self.aggregated_cache.write().await;
        cache.clear();
    }

    /// Generate aggregated metrics for a window
    async fn generate_aggregated_metrics(
        &self,
        window: MetricsWindow,
    ) -> Option<AggregatedMetrics> {
        let raw_metrics = self.raw_metrics.read().await;

        if raw_metrics.is_empty() {
            return None;
        }

        let window_duration = self.get_window_duration(window);
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let cutoff_time = current_time.saturating_sub(window_duration);

        // Filter metrics within the window
        let window_metrics: Vec<&PerformanceMetrics> = raw_metrics
            .iter()
            .filter(|m| m.timestamp >= cutoff_time)
            .collect();

        if window_metrics.is_empty() {
            return None;
        }

        let sample_count = window_metrics.len();
        let start_time = window_metrics.first()?.timestamp;
        let end_time = window_metrics.last()?.timestamp;

        let time_range = TimeRange {
            start: start_time,
            end: end_time,
            duration_seconds: end_time - start_time,
        };

        // Aggregate system metrics
        let system = self.aggregate_system_metrics(&window_metrics);

        // Aggregate synthesis metrics
        let synthesis = self.aggregate_synthesis_metrics(&window_metrics);

        // Aggregate memory metrics
        let memory = self.aggregate_memory_metrics(&window_metrics);

        // Aggregate GPU metrics if available
        let gpu = self.aggregate_gpu_metrics(&window_metrics);

        // Calculate trends
        let trends = self.calculate_trends(&window_metrics);

        Some(AggregatedMetrics {
            window,
            sample_count,
            time_range,
            system,
            synthesis,
            memory,
            gpu,
            trends,
        })
    }

    /// Get window duration in seconds
    fn get_window_duration(&self, window: MetricsWindow) -> u64 {
        match window {
            MetricsWindow::OneMinute => 60,
            MetricsWindow::FiveMinutes => 300,
            MetricsWindow::FifteenMinutes => 900,
            MetricsWindow::OneHour => 3600,
            MetricsWindow::TwentyFourHours => 86400,
            MetricsWindow::AllTime => u64::MAX,
        }
    }

    /// Aggregate system metrics
    fn aggregate_system_metrics(&self, metrics: &[&PerformanceMetrics]) -> SystemSummary {
        let cpu_values: Vec<f64> = metrics.iter().map(|m| m.system.cpu_usage).collect();
        let memory_used_values: Vec<f64> = metrics
            .iter()
            .map(|m| m.system.memory_used as f64)
            .collect();
        let memory_available_values: Vec<f64> = metrics
            .iter()
            .map(|m| m.system.memory_available as f64)
            .collect();
        let disk_read_values: Vec<f64> = metrics
            .iter()
            .map(|m| m.system.disk_read_bps as f64)
            .collect();
        let disk_write_values: Vec<f64> = metrics
            .iter()
            .map(|m| m.system.disk_write_bps as f64)
            .collect();
        let network_values: Vec<f64> = metrics
            .iter()
            .map(|m| m.system.network_bps as f64)
            .collect();
        let thread_count_values: Vec<f64> = metrics
            .iter()
            .map(|m| m.system.thread_count as f64)
            .collect();

        let load_average_values: Vec<f64> = metrics
            .iter()
            .filter_map(|m| m.system.load_average)
            .collect();

        SystemSummary {
            cpu: StatisticsSummary::from_values(&cpu_values),
            memory_used: StatisticsSummary::from_values(&memory_used_values),
            memory_available: StatisticsSummary::from_values(&memory_available_values),
            disk_read: StatisticsSummary::from_values(&disk_read_values),
            disk_write: StatisticsSummary::from_values(&disk_write_values),
            network: StatisticsSummary::from_values(&network_values),
            thread_count: StatisticsSummary::from_values(&thread_count_values),
            load_average: if load_average_values.is_empty() {
                None
            } else {
                Some(StatisticsSummary::from_values(&load_average_values))
            },
        }
    }

    /// Aggregate synthesis metrics
    fn aggregate_synthesis_metrics(&self, metrics: &[&PerformanceMetrics]) -> SynthesisSummary {
        let total_operations: u64 = metrics.iter().map(|m| m.synthesis.total_operations).sum();
        let successful_operations: u64 = metrics
            .iter()
            .map(|m| m.synthesis.successful_operations)
            .sum();
        let success_rate = if total_operations > 0 {
            (successful_operations as f64 / total_operations as f64) * 100.0
        } else {
            0.0
        };

        let synthesis_time_values: Vec<f64> = metrics
            .iter()
            .map(|m| m.synthesis.avg_synthesis_time_ms)
            .collect();
        let rtf_values: Vec<f64> = metrics
            .iter()
            .map(|m| m.synthesis.real_time_factor)
            .collect();
        let throughput_values: Vec<f64> = metrics
            .iter()
            .map(|m| m.synthesis.throughput_chars_per_sec)
            .collect();
        let queue_depth_values: Vec<f64> = metrics
            .iter()
            .map(|m| m.synthesis.queue_depth as f64)
            .collect();
        let memory_per_op_values: Vec<f64> = metrics
            .iter()
            .map(|m| m.synthesis.memory_per_operation_mb)
            .collect();

        let total_audio_duration: f64 = metrics
            .iter()
            .map(|m| m.synthesis.total_audio_duration)
            .sum();

        SynthesisSummary {
            total_operations,
            success_rate,
            synthesis_time: StatisticsSummary::from_values(&synthesis_time_values),
            real_time_factor: StatisticsSummary::from_values(&rtf_values),
            throughput: StatisticsSummary::from_values(&throughput_values),
            queue_depth: StatisticsSummary::from_values(&queue_depth_values),
            memory_per_operation: StatisticsSummary::from_values(&memory_per_op_values),
            total_audio_duration,
        }
    }

    /// Aggregate memory metrics
    fn aggregate_memory_metrics(&self, metrics: &[&PerformanceMetrics]) -> MemorySummary {
        let heap_used_values: Vec<f64> =
            metrics.iter().map(|m| m.memory.heap_used as f64).collect();
        let peak_usage_values: Vec<f64> =
            metrics.iter().map(|m| m.memory.peak_usage as f64).collect();
        let allocation_rate_values: Vec<f64> = metrics
            .iter()
            .map(|m| m.memory.allocations_per_sec)
            .collect();
        let deallocation_rate_values: Vec<f64> = metrics
            .iter()
            .map(|m| m.memory.deallocations_per_sec)
            .collect();
        let fragmentation_values: Vec<f64> = metrics
            .iter()
            .map(|m| m.memory.fragmentation_percent)
            .collect();
        let cache_hit_rate_values: Vec<f64> =
            metrics.iter().map(|m| m.memory.cache_hit_rate).collect();

        let gc_events: u64 = metrics.iter().map(|m| m.memory.gc_events).sum();

        MemorySummary {
            heap_used: StatisticsSummary::from_values(&heap_used_values),
            peak_usage: StatisticsSummary::from_values(&peak_usage_values),
            allocation_rate: StatisticsSummary::from_values(&allocation_rate_values),
            deallocation_rate: StatisticsSummary::from_values(&deallocation_rate_values),
            fragmentation: StatisticsSummary::from_values(&fragmentation_values),
            cache_hit_rate: StatisticsSummary::from_values(&cache_hit_rate_values),
            gc_events,
        }
    }

    /// Aggregate GPU metrics
    fn aggregate_gpu_metrics(&self, metrics: &[&PerformanceMetrics]) -> Option<GpuSummary> {
        let gpu_metrics: Vec<&GpuMetrics> = metrics.iter().filter_map(|m| m.gpu.as_ref()).collect();

        if gpu_metrics.is_empty() {
            return None;
        }

        let utilization_values: Vec<f64> = gpu_metrics.iter().map(|g| g.utilization).collect();
        let memory_used_values: Vec<f64> =
            gpu_metrics.iter().map(|g| g.memory_used as f64).collect();
        let memory_usage_percent_values: Vec<f64> = gpu_metrics
            .iter()
            .map(|g| (g.memory_used as f64 / g.memory_total as f64) * 100.0)
            .collect();
        let temperature_values: Vec<f64> = gpu_metrics.iter().map(|g| g.temperature).collect();
        let power_values: Vec<f64> = gpu_metrics.iter().map(|g| g.power_consumption).collect();
        let compute_units_values: Vec<f64> = gpu_metrics
            .iter()
            .map(|g| g.compute_units_active as f64)
            .collect();
        let bandwidth_values: Vec<f64> = gpu_metrics
            .iter()
            .map(|g| g.memory_bandwidth_util)
            .collect();

        Some(GpuSummary {
            utilization: StatisticsSummary::from_values(&utilization_values),
            memory_used: StatisticsSummary::from_values(&memory_used_values),
            memory_usage_percent: StatisticsSummary::from_values(&memory_usage_percent_values),
            temperature: StatisticsSummary::from_values(&temperature_values),
            power_consumption: StatisticsSummary::from_values(&power_values),
            compute_units: StatisticsSummary::from_values(&compute_units_values),
            memory_bandwidth: StatisticsSummary::from_values(&bandwidth_values),
        })
    }

    /// Calculate performance trends
    fn calculate_trends(&self, metrics: &[&PerformanceMetrics]) -> PerformanceTrends {
        if metrics.len() < 2 {
            return PerformanceTrends {
                cpu_trend: TrendDirection::Unknown,
                memory_trend: TrendDirection::Unknown,
                synthesis_performance_trend: TrendDirection::Unknown,
                queue_depth_trend: TrendDirection::Unknown,
                error_rate_trend: TrendDirection::Unknown,
                overall_trend: TrendDirection::Unknown,
            };
        }

        let cpu_values: Vec<f64> = metrics.iter().map(|m| m.system.cpu_usage).collect();
        let memory_values: Vec<f64> = metrics
            .iter()
            .map(|m| m.system.memory_used as f64)
            .collect();
        let rtf_values: Vec<f64> = metrics
            .iter()
            .map(|m| m.synthesis.real_time_factor)
            .collect();
        let queue_values: Vec<f64> = metrics
            .iter()
            .map(|m| m.synthesis.queue_depth as f64)
            .collect();
        let error_rate_values: Vec<f64> = metrics
            .iter()
            .map(|m| {
                if m.synthesis.total_operations > 0 {
                    (m.synthesis.failed_operations as f64 / m.synthesis.total_operations as f64)
                        * 100.0
                } else {
                    0.0
                }
            })
            .collect();

        PerformanceTrends {
            cpu_trend: self.calculate_trend_direction(&cpu_values, false),
            memory_trend: self.calculate_trend_direction(&memory_values, false),
            synthesis_performance_trend: self.calculate_trend_direction(&rtf_values, true),
            queue_depth_trend: self.calculate_trend_direction(&queue_values, false),
            error_rate_trend: self.calculate_trend_direction(&error_rate_values, false),
            overall_trend: self.calculate_overall_trend(
                &cpu_values,
                &memory_values,
                &rtf_values,
                &error_rate_values,
            ),
        }
    }

    /// Calculate trend direction for a series of values
    fn calculate_trend_direction(&self, values: &[f64], higher_is_better: bool) -> TrendDirection {
        if values.len() < 2 {
            return TrendDirection::Unknown;
        }

        // Simple linear regression to find trend
        let n = values.len() as f64;
        let x_values: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();

        let sum_x: f64 = x_values.iter().sum();
        let sum_y: f64 = values.iter().sum();
        let sum_xy: f64 = x_values.iter().zip(values.iter()).map(|(x, y)| x * y).sum();
        let sum_xx: f64 = x_values.iter().map(|x| x * x).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);

        // Determine relative change magnitude
        let avg = sum_y / n;
        let relative_slope = if avg != 0.0 { slope / avg } else { 0.0 };

        let threshold_strong = 0.1; // 10% change
        let threshold_weak = 0.02; // 2% change

        let improving = if higher_is_better {
            slope > 0.0
        } else {
            slope < 0.0
        };
        let abs_slope = relative_slope.abs();

        if improving {
            if abs_slope > threshold_strong {
                TrendDirection::StronglyImproving
            } else if abs_slope > threshold_weak {
                TrendDirection::Improving
            } else {
                TrendDirection::Stable
            }
        } else {
            if abs_slope > threshold_strong {
                TrendDirection::StronglyDegrading
            } else if abs_slope > threshold_weak {
                TrendDirection::Degrading
            } else {
                TrendDirection::Stable
            }
        }
    }

    /// Calculate overall trend based on multiple metrics
    fn calculate_overall_trend(
        &self,
        cpu: &[f64],
        memory: &[f64],
        rtf: &[f64],
        error_rate: &[f64],
    ) -> TrendDirection {
        let cpu_trend = self.calculate_trend_direction(cpu, false);
        let memory_trend = self.calculate_trend_direction(memory, false);
        let rtf_trend = self.calculate_trend_direction(rtf, true);
        let error_trend = self.calculate_trend_direction(error_rate, false);

        // Weight the trends (RTF is most important for synthesis performance)
        let trends = vec![
            (cpu_trend, 0.2),
            (memory_trend, 0.2),
            (rtf_trend, 0.4),
            (error_trend, 0.2),
        ];

        let mut score = 0.0;
        for (trend, weight) in trends {
            let trend_score = match trend {
                TrendDirection::StronglyImproving => 2.0,
                TrendDirection::Improving => 1.0,
                TrendDirection::Stable => 0.0,
                TrendDirection::Degrading => -1.0,
                TrendDirection::StronglyDegrading => -2.0,
                TrendDirection::Unknown => 0.0,
            };
            score += trend_score * weight;
        }

        if score > 1.0 {
            TrendDirection::StronglyImproving
        } else if score > 0.3 {
            TrendDirection::Improving
        } else if score > -0.3 {
            TrendDirection::Stable
        } else if score > -1.0 {
            TrendDirection::Degrading
        } else {
            TrendDirection::StronglyDegrading
        }
    }

    /// Generate report summary
    async fn generate_report_summary(
        &self,
        windows: &HashMap<MetricsWindow, AggregatedMetrics>,
    ) -> ReportSummary {
        let mut summary = ReportSummary::default();

        if let Some(current) = windows.get(&MetricsWindow::OneMinute) {
            summary.current_cpu_usage = current.system.cpu.average;
            summary.current_memory_usage = current.system.memory_used.average;
            summary.current_rtf = current.synthesis.real_time_factor.average;
            summary.current_success_rate = current.synthesis.success_rate;
        }

        if let Some(hourly) = windows.get(&MetricsWindow::OneHour) {
            summary.hourly_operations = hourly.synthesis.total_operations;
            summary.hourly_audio_duration = hourly.synthesis.total_audio_duration;
        }

        if let Some(daily) = windows.get(&MetricsWindow::TwentyFourHours) {
            summary.daily_operations = daily.synthesis.total_operations;
            summary.daily_audio_duration = daily.synthesis.total_audio_duration;
        }

        // Find best and worst performing windows
        let mut best_rtf = 0.0;
        let mut worst_rtf = f64::INFINITY;

        for metrics in windows.values() {
            if metrics.synthesis.real_time_factor.average > best_rtf {
                best_rtf = metrics.synthesis.real_time_factor.average;
                summary.best_performance_window = Some(metrics.window);
            }
            if metrics.synthesis.real_time_factor.average < worst_rtf {
                worst_rtf = metrics.synthesis.real_time_factor.average;
                summary.worst_performance_window = Some(metrics.window);
            }
        }

        summary
    }

    /// Clear all metrics
    pub async fn clear_metrics(&self) {
        let mut raw_metrics = self.raw_metrics.write().await;
        raw_metrics.clear();

        self.invalidate_cache().await;
    }

    /// Get metrics count
    pub async fn metrics_count(&self) -> usize {
        let raw_metrics = self.raw_metrics.read().await;
        raw_metrics.len()
    }
}

/// Performance report structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// Report generation timestamp
    pub generation_time: u64,
    /// System uptime in seconds
    pub uptime_seconds: u64,
    /// Metrics for different time windows
    pub windows: HashMap<MetricsWindow, AggregatedMetrics>,
    /// High-level summary
    pub summary: ReportSummary,
}

/// Report summary with key metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReportSummary {
    /// Current CPU usage percentage
    pub current_cpu_usage: f64,
    /// Current memory usage in bytes
    pub current_memory_usage: f64,
    /// Current real-time factor
    pub current_rtf: f64,
    /// Current success rate percentage
    pub current_success_rate: f64,
    /// Operations in the last hour
    pub hourly_operations: u64,
    /// Audio duration generated in the last hour
    pub hourly_audio_duration: f64,
    /// Operations in the last 24 hours
    pub daily_operations: u64,
    /// Audio duration generated in the last 24 hours
    pub daily_audio_duration: f64,
    /// Best performing time window
    pub best_performance_window: Option<MetricsWindow>,
    /// Worst performing time window
    pub worst_performance_window: Option<MetricsWindow>,
}

impl StatisticsSummary {
    /// Create statistics summary from a vector of values
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let count = sorted.len();
        let sum: f64 = sorted.iter().sum();
        let average = sum / count as f64;
        let minimum = sorted[0];
        let maximum = sorted[count - 1];

        // Calculate percentiles
        let p50 = percentile(&sorted, 50.0);
        let p90 = percentile(&sorted, 90.0);
        let p95 = percentile(&sorted, 95.0);
        let p99 = percentile(&sorted, 99.0);

        // Calculate standard deviation
        let variance: f64 =
            values.iter().map(|v| (v - average).powi(2)).sum::<f64>() / count as f64;
        let std_deviation = variance.sqrt();

        Self {
            average,
            minimum,
            maximum,
            std_deviation,
            p50,
            p90,
            p95,
            p99,
            count,
        }
    }
}

impl Default for StatisticsSummary {
    fn default() -> Self {
        Self {
            average: 0.0,
            minimum: 0.0,
            maximum: 0.0,
            std_deviation: 0.0,
            p50: 0.0,
            p90: 0.0,
            p95: 0.0,
            p99: 0.0,
            count: 0,
        }
    }
}

/// Calculate percentile value
fn percentile(sorted_values: &[f64], percentile: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }

    let index = (percentile / 100.0) * (sorted_values.len() - 1) as f64;
    let lower_index = index.floor() as usize;
    let upper_index = index.ceil() as usize;

    if lower_index == upper_index {
        sorted_values[lower_index]
    } else {
        let lower_value = sorted_values[lower_index];
        let upper_value = sorted_values[upper_index];
        let fraction = index - lower_index as f64;
        lower_value + fraction * (upper_value - lower_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new(1000, Duration::from_secs(60));
        assert_eq!(collector.metrics_count().await, 0);
    }

    #[tokio::test]
    async fn test_add_metrics() {
        let collector = MetricsCollector::new(1000, Duration::from_secs(60));
        let metrics = PerformanceMetrics::default();

        collector.add_metrics(metrics).await;
        assert_eq!(collector.metrics_count().await, 1);

        let latest = collector.get_latest_metrics().await;
        assert!(latest.is_some());
    }

    #[tokio::test]
    async fn test_statistics_summary() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let summary = StatisticsSummary::from_values(&values);

        assert_eq!(summary.count, 5);
        assert_eq!(summary.average, 3.0);
        assert_eq!(summary.minimum, 1.0);
        assert_eq!(summary.maximum, 5.0);
        assert_eq!(summary.p50, 3.0);
    }

    #[test]
    fn test_percentile_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(percentile(&values, 0.0), 1.0);
        assert_eq!(percentile(&values, 50.0), 3.0);
        assert_eq!(percentile(&values, 100.0), 5.0);
    }

    #[tokio::test]
    async fn test_aggregated_metrics_generation() {
        let collector = MetricsCollector::new(1000, Duration::from_secs(60));

        // Add some test metrics
        for i in 0..5 {
            let mut metrics = PerformanceMetrics::default();
            metrics.system.cpu_usage = (i as f64) * 10.0;
            metrics.timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            collector.add_metrics(metrics).await;
        }

        let aggregated = collector
            .get_aggregated_metrics(MetricsWindow::OneMinute)
            .await;
        assert!(aggregated.is_some());

        let aggregated = aggregated.unwrap();
        assert_eq!(aggregated.sample_count, 5);
        assert_eq!(aggregated.window, MetricsWindow::OneMinute);
    }

    #[tokio::test]
    async fn test_trend_calculation() {
        let collector = MetricsCollector::new(1000, Duration::from_secs(60));

        // Add metrics with improving trend
        for i in 0..10 {
            let mut metrics = PerformanceMetrics::default();
            metrics.synthesis.real_time_factor = 1.0 + (i as f64) * 0.1; // Improving RTF
            metrics.timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            collector.add_metrics(metrics).await;
        }

        let trends = collector
            .get_performance_trends(MetricsWindow::OneMinute)
            .await;
        assert!(trends.is_some());

        let trends = trends.unwrap();
        assert!(matches!(
            trends.synthesis_performance_trend,
            TrendDirection::Improving | TrendDirection::StronglyImproving
        ));
    }

    #[test]
    fn test_window_duration() {
        let collector = MetricsCollector::new(1000, Duration::from_secs(60));

        assert_eq!(collector.get_window_duration(MetricsWindow::OneMinute), 60);
        assert_eq!(
            collector.get_window_duration(MetricsWindow::FiveMinutes),
            300
        );
        assert_eq!(collector.get_window_duration(MetricsWindow::OneHour), 3600);
    }

    #[tokio::test]
    async fn test_performance_report_generation() {
        let collector = MetricsCollector::new(1000, Duration::from_secs(60));

        // Add some metrics
        let mut metrics = PerformanceMetrics::default();
        metrics.synthesis.total_operations = 100;
        metrics.synthesis.successful_operations = 95;
        collector.add_metrics(metrics).await;

        let report = collector.generate_performance_report().await;
        assert!(report.generation_time > 0);
        // uptime_seconds is unsigned, so always >= 0 - removing redundant check
    }
}
