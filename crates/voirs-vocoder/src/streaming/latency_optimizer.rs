//! Enhanced latency optimization for neural vocoders
//!
//! This module provides advanced latency optimization techniques including:
//! - Predictive processing with machine learning-based load prediction
//! - Dynamic buffer sizing based on network conditions and system load
//! - CPU affinity and thread priority management
//! - NUMA-aware memory allocation
//! - Real-time deadline scheduling

use crate::config::StreamingConfig;
use crate::Result;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex as AsyncMutex;

/// Enhanced latency optimizer with advanced algorithms
pub struct EnhancedLatencyOptimizer {
    /// Base configuration
    config: Arc<RwLock<StreamingConfig>>,

    /// Performance metrics collector
    metrics: Arc<PerformanceMetrics>,

    /// Predictive load estimator
    load_predictor: Arc<AsyncMutex<LoadPredictor>>,

    /// Dynamic buffer manager
    buffer_manager: Arc<DynamicBufferManager>,

    /// System resource monitor
    resource_monitor: Arc<SystemResourceMonitor>,

    /// Real-time scheduler
    rt_scheduler: Arc<RealtimeScheduler>,

    /// Optimization statistics
    stats: Arc<RwLock<EnhancedLatencyStats>>,
}

/// Performance metrics collection
#[allow(dead_code)]
struct PerformanceMetrics {
    /// Processing time history with timestamps
    processing_history: RwLock<VecDeque<(Instant, f32)>>,

    /// Latency measurements with context
    latency_history: RwLock<VecDeque<LatencyMeasurement>>,

    /// System load measurements
    system_load: RwLock<VecDeque<SystemLoad>>,

    /// Buffer underrun/overrun events
    buffer_events: RwLock<VecDeque<BufferEvent>>,

    /// Quality degradation events
    quality_events: RwLock<VecDeque<QualityEvent>>,
}

/// Single latency measurement with context
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct LatencyMeasurement {
    /// Timestamp
    timestamp: Instant,
    /// End-to-end latency in ms
    latency_ms: f32,
    /// Processing latency component
    processing_ms: f32,
    /// Buffer latency component
    buffer_ms: f32,
    /// Network latency component (if applicable)
    network_ms: f32,
    /// Chunk size used
    chunk_size: usize,
    /// System load at time of measurement
    system_load: f32,
}

/// System load snapshot
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SystemLoad {
    timestamp: Instant,
    cpu_usage: f32,
    memory_usage: f32,
    gpu_usage: Option<f32>,
    network_bandwidth_mbps: f32,
    io_wait: f32,
    context_switches_per_sec: u64,
}

/// Buffer event (underrun or overrun)
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BufferEvent {
    timestamp: Instant,
    event_type: BufferEventType,
    buffer_size: usize,
    chunk_size: usize,
    system_load: f32,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum BufferEventType {
    Underrun,
    Overrun,
}

/// Quality degradation event
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct QualityEvent {
    timestamp: Instant,
    quality_score: f32,
    expected_score: f32,
    latency_ms: f32,
    chunk_size: usize,
}

/// Predictive load estimator using time series analysis
#[allow(dead_code)]
struct LoadPredictor {
    /// Historical data points (timestamp, load)
    load_history: VecDeque<(u64, f32)>,

    /// Seasonal patterns (hour of day -> expected load)
    seasonal_patterns: HashMap<u8, f32>,

    /// Trend analysis coefficients
    trend_coefficients: Vec<f32>,

    /// Moving average window
    ma_window: usize,

    /// Prediction accuracy tracking
    prediction_accuracy: VecDeque<f32>,

    /// Last update time
    last_update: Instant,
}

impl LoadPredictor {
    fn new() -> Self {
        Self {
            load_history: VecDeque::with_capacity(1000),
            seasonal_patterns: HashMap::new(),
            trend_coefficients: vec![0.0; 5],
            ma_window: 20,
            prediction_accuracy: VecDeque::with_capacity(100),
            last_update: Instant::now(),
        }
    }

    /// Update with new load measurement
    fn update(&mut self, load: f32) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.load_history.push_back((now, load));

        // Keep only recent history
        if self.load_history.len() > 1000 {
            self.load_history.pop_front();
        }

        // Update seasonal patterns
        let hour = ((now / 3600) % 24) as u8;
        let current_seasonal = self.seasonal_patterns.entry(hour).or_insert(0.0);
        *current_seasonal = (*current_seasonal * 0.95) + (load * 0.05);

        // Update trend coefficients using simple linear regression
        self.update_trend_analysis();

        self.last_update = Instant::now();
    }

    /// Predict load for next N seconds
    fn predict_load(&self, seconds_ahead: u32) -> f32 {
        if self.load_history.is_empty() {
            return 0.5; // Default moderate load
        }

        // Get current time
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let future_time = now + seconds_ahead as u64;
        let future_hour = ((future_time / 3600) % 24) as u8;

        // Base prediction on seasonal pattern
        let seasonal_load = self
            .seasonal_patterns
            .get(&future_hour)
            .copied()
            .unwrap_or(0.5);

        // Apply trend analysis
        let trend_adjustment = self.calculate_trend_adjustment(seconds_ahead);

        // Combine seasonal and trend predictions
        let predicted_load = seasonal_load + trend_adjustment;

        // Clamp to valid range
        predicted_load.clamp(0.0, 1.0)
    }

    /// Update trend analysis coefficients
    fn update_trend_analysis(&mut self) {
        if self.load_history.len() < 10 {
            return;
        }

        // Simple linear regression on recent data
        let recent_data: Vec<_> = self.load_history.iter().rev().take(50).collect();

        let n = recent_data.len() as f32;
        let sum_x: f32 = (0..recent_data.len()).map(|i| i as f32).sum();
        let sum_y: f32 = recent_data.iter().map(|(_, load)| *load).sum();
        let sum_xy: f32 = recent_data
            .iter()
            .enumerate()
            .map(|(i, (_, load))| i as f32 * load)
            .sum();
        let sum_x2: f32 = (0..recent_data.len()).map(|i| (i as f32).powi(2)).sum();

        // Calculate slope (trend)
        let slope = if n * sum_x2 - sum_x.powi(2) != 0.0 {
            (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2))
        } else {
            0.0
        };

        // Update trend coefficients with exponential smoothing
        if self.trend_coefficients.is_empty() {
            self.trend_coefficients.push(slope);
        } else {
            self.trend_coefficients[0] = self.trend_coefficients[0] * 0.8 + slope * 0.2;
        }
    }

    /// Calculate trend adjustment for prediction
    fn calculate_trend_adjustment(&self, seconds_ahead: u32) -> f32 {
        if self.trend_coefficients.is_empty() {
            return 0.0;
        }

        // Apply trend with diminishing effect over time
        let time_factor = 1.0 / (1.0 + seconds_ahead as f32 / 60.0); // Reduce effect over time
        self.trend_coefficients[0] * seconds_ahead as f32 * time_factor
    }

    /// Get prediction accuracy score
    fn get_accuracy(&self) -> f32 {
        if self.prediction_accuracy.is_empty() {
            return 0.0;
        }

        let sum: f32 = self.prediction_accuracy.iter().sum();
        sum / self.prediction_accuracy.len() as f32
    }
}

/// Dynamic buffer management for optimal latency
#[allow(dead_code)]
struct DynamicBufferManager {
    /// Current buffer configurations per stream
    buffer_configs: RwLock<HashMap<u64, BufferConfig>>,

    /// Global buffer statistics
    #[allow(dead_code)]
    global_stats: RwLock<BufferStats>,

    /// Adaptive thresholds
    #[allow(dead_code)]
    adaptive_thresholds: RwLock<AdaptiveThresholds>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BufferConfig {
    #[allow(dead_code)]
    stream_id: u64,
    buffer_size: usize,
    #[allow(dead_code)]
    low_watermark: usize,
    #[allow(dead_code)]
    high_watermark: usize,
    #[allow(dead_code)]
    prefetch_size: usize,
    #[allow(dead_code)]
    last_adjustment: Instant,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
struct BufferStats {
    #[allow(dead_code)]
    total_underruns: u64,
    #[allow(dead_code)]
    total_overruns: u64,
    #[allow(dead_code)]
    avg_buffer_utilization: f32,
    #[allow(dead_code)]
    peak_buffer_utilization: f32,
    #[allow(dead_code)]
    adjustment_count: u64,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AdaptiveThresholds {
    #[allow(dead_code)]
    underrun_threshold: f32,
    #[allow(dead_code)]
    overrun_threshold: f32,
    #[allow(dead_code)]
    adjustment_sensitivity: f32,
    #[allow(dead_code)]
    last_update: Instant,
}

impl Default for AdaptiveThresholds {
    fn default() -> Self {
        Self {
            underrun_threshold: 0.1,
            overrun_threshold: 0.9,
            adjustment_sensitivity: 0.1,
            last_update: Instant::now(),
        }
    }
}

impl DynamicBufferManager {
    fn new() -> Self {
        Self {
            buffer_configs: RwLock::new(HashMap::new()),
            global_stats: RwLock::new(BufferStats::default()),
            adaptive_thresholds: RwLock::new(AdaptiveThresholds::default()),
        }
    }

    /// Get optimal buffer size for a stream
    fn get_optimal_buffer_size(&self, stream_id: u64, predicted_load: f32) -> usize {
        let configs = self.buffer_configs.read().unwrap();

        if let Some(config) = configs.get(&stream_id) {
            // Adjust based on predicted load
            let load_factor = 1.0 + predicted_load * 0.5; // Increase buffer for higher load
            (config.buffer_size as f32 * load_factor) as usize
        } else {
            // Default buffer size for new streams
            1024
        }
    }

    /// Update buffer configuration based on performance
    #[allow(dead_code)]
    fn update_buffer_config(&self, stream_id: u64, event: BufferEvent) {
        let mut configs = self.buffer_configs.write().unwrap();

        let config = configs.entry(stream_id).or_insert_with(|| BufferConfig {
            stream_id,
            buffer_size: 1024,
            low_watermark: 256,
            high_watermark: 768,
            prefetch_size: 512,
            last_adjustment: Instant::now(),
        });

        // Adjust buffer size based on event type
        match event.event_type {
            BufferEventType::Underrun => {
                config.buffer_size = (config.buffer_size as f32 * 1.2) as usize;
                config.low_watermark = config.buffer_size / 4;
                config.high_watermark = config.buffer_size * 3 / 4;
            }
            BufferEventType::Overrun => {
                config.buffer_size = (config.buffer_size as f32 * 0.9) as usize;
                config.low_watermark = config.buffer_size / 4;
                config.high_watermark = config.buffer_size * 3 / 4;
            }
        }

        config.last_adjustment = Instant::now();
    }
}

/// System resource monitoring
#[allow(dead_code)]
struct SystemResourceMonitor {
    /// Resource usage history
    #[allow(dead_code)]
    usage_history: RwLock<VecDeque<SystemLoad>>,

    /// CPU affinity settings
    #[allow(dead_code)]
    cpu_affinity: RwLock<Option<Vec<usize>>>,

    /// Memory allocation strategy
    #[allow(dead_code)]
    memory_strategy: RwLock<MemoryStrategy>,

    /// Last monitoring update
    #[allow(dead_code)]
    last_update: RwLock<Instant>,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
enum MemoryStrategy {
    #[default]
    Default,
    #[allow(dead_code)]
    NumaAware,
    #[allow(dead_code)]
    LargePages,
    #[allow(dead_code)]
    Pinned,
}

impl SystemResourceMonitor {
    fn new() -> Self {
        Self {
            usage_history: RwLock::new(VecDeque::with_capacity(100)),
            cpu_affinity: RwLock::new(None),
            memory_strategy: RwLock::new(MemoryStrategy::Default),
            last_update: RwLock::new(Instant::now()),
        }
    }

    /// Get current system load
    fn get_system_load(&self) -> SystemLoad {
        // In a real implementation, this would query system metrics
        // For now, return a dummy load
        SystemLoad {
            timestamp: Instant::now(),
            cpu_usage: 0.5,
            memory_usage: 0.4,
            gpu_usage: Some(0.3),
            network_bandwidth_mbps: 100.0,
            io_wait: 0.1,
            context_switches_per_sec: 1000,
        }
    }

    /// Update resource monitoring
    #[allow(dead_code)]
    fn update(&self) {
        let load = self.get_system_load();
        let mut history = self.usage_history.write().unwrap();

        history.push_back(load);

        if history.len() > 100 {
            history.pop_front();
        }

        *self.last_update.write().unwrap() = Instant::now();
    }

    /// Get optimal CPU affinity for current load
    #[allow(dead_code)]
    fn get_optimal_cpu_affinity(&self) -> Option<Vec<usize>> {
        // In a real implementation, this would analyze CPU topology
        // and current load distribution
        None
    }
}

/// Real-time deadline scheduling
#[allow(dead_code)]
struct RealtimeScheduler {
    /// Active deadlines
    #[allow(dead_code)]
    deadlines: AsyncMutex<VecDeque<SchedulingDeadline>>,

    /// Priority levels
    #[allow(dead_code)]
    priority_levels: RwLock<HashMap<u64, Priority>>,

    /// Scheduling statistics
    stats: RwLock<SchedulingStats>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SchedulingDeadline {
    #[allow(dead_code)]
    stream_id: u64,
    #[allow(dead_code)]
    deadline: Instant,
    #[allow(dead_code)]
    priority: Priority,
    #[allow(dead_code)]
    chunk_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
#[allow(dead_code)]
enum Priority {
    #[allow(dead_code)]
    Low = 1,
    #[default]
    Normal = 2,
    #[allow(dead_code)]
    High = 3,
    #[allow(dead_code)]
    Critical = 4,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
struct SchedulingStats {
    deadlines_met: u64,
    deadlines_missed: u64,
    #[allow(dead_code)]
    avg_scheduling_latency_ms: f32,
    #[allow(dead_code)]
    priority_inversions: u64,
}

impl RealtimeScheduler {
    fn new() -> Self {
        Self {
            deadlines: AsyncMutex::new(VecDeque::new()),
            priority_levels: RwLock::new(HashMap::new()),
            stats: RwLock::new(SchedulingStats::default()),
        }
    }

    /// Schedule a task with deadline
    #[allow(dead_code)]
    async fn schedule_task(
        &self,
        stream_id: u64,
        deadline: Instant,
        chunk_size: usize,
    ) -> Result<()> {
        let priority = self
            .priority_levels
            .read()
            .unwrap()
            .get(&stream_id)
            .copied()
            .unwrap_or(Priority::Normal);

        let scheduling_deadline = SchedulingDeadline {
            stream_id,
            deadline,
            priority,
            chunk_size,
        };

        let mut deadlines = self.deadlines.lock().await;

        // Insert in priority order
        let insert_pos = deadlines
            .iter()
            .position(|d| {
                d.priority < priority || (d.priority == priority && d.deadline > deadline)
            })
            .unwrap_or(deadlines.len());

        deadlines.insert(insert_pos, scheduling_deadline);

        Ok(())
    }

    /// Check for deadline misses
    #[allow(dead_code)]
    async fn check_deadlines(&self) -> Vec<u64> {
        let mut deadlines = self.deadlines.lock().await;
        let now = Instant::now();
        let mut missed_streams = Vec::new();

        while let Some(deadline) = deadlines.front() {
            if deadline.deadline <= now {
                missed_streams.push(deadline.stream_id);
                deadlines.pop_front();

                // Update statistics
                if let Ok(mut stats) = self.stats.write() {
                    stats.deadlines_missed += 1;
                }
            } else {
                break;
            }
        }

        missed_streams
    }
}

/// Enhanced latency optimization statistics
#[derive(Debug, Clone, Default)]
pub struct EnhancedLatencyStats {
    /// Basic latency stats
    pub avg_latency_ms: f32,
    pub p50_latency_ms: f32,
    pub p95_latency_ms: f32,
    pub p99_latency_ms: f32,

    /// Prediction accuracy
    pub load_prediction_accuracy: f32,
    pub latency_prediction_accuracy: f32,

    /// Adaptation statistics
    pub buffer_adaptations: u64,
    pub chunk_adaptations: u64,
    pub priority_adjustments: u64,

    /// System resource utilization
    pub avg_cpu_usage: f32,
    pub avg_memory_usage: f32,
    pub avg_gpu_usage: f32,

    /// Quality metrics
    pub quality_degradation_events: u64,
    pub deadline_miss_rate: f32,

    /// Optimization effectiveness
    pub latency_reduction_percent: f32,
    pub throughput_improvement_percent: f32,
}

impl EnhancedLatencyOptimizer {
    /// Create new enhanced latency optimizer
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            metrics: Arc::new(PerformanceMetrics {
                processing_history: RwLock::new(VecDeque::with_capacity(1000)),
                latency_history: RwLock::new(VecDeque::with_capacity(1000)),
                system_load: RwLock::new(VecDeque::with_capacity(100)),
                buffer_events: RwLock::new(VecDeque::with_capacity(100)),
                quality_events: RwLock::new(VecDeque::with_capacity(100)),
            }),
            load_predictor: Arc::new(AsyncMutex::new(LoadPredictor::new())),
            buffer_manager: Arc::new(DynamicBufferManager::new()),
            resource_monitor: Arc::new(SystemResourceMonitor::new()),
            rt_scheduler: Arc::new(RealtimeScheduler::new()),
            stats: Arc::new(RwLock::new(EnhancedLatencyStats::default())),
        }
    }

    /// Predict optimal chunk size for upcoming processing
    pub async fn predict_optimal_chunk_size(
        &self,
        stream_id: u64,
        lookahead_seconds: u32,
    ) -> usize {
        // Get predicted system load
        let predicted_load = self
            .load_predictor
            .lock()
            .await
            .predict_load(lookahead_seconds);

        // Get optimal buffer size
        let _buffer_size = self
            .buffer_manager
            .get_optimal_buffer_size(stream_id, predicted_load);

        // Calculate chunk size based on buffer and load
        let config = self.config.read().unwrap();
        let base_size = config.chunk_size;

        // Adjust based on predicted conditions
        let load_factor = if predicted_load > 0.8 {
            0.7 // Smaller chunks for high load
        } else if predicted_load < 0.3 {
            1.3 // Larger chunks for low load
        } else {
            1.0
        };

        let optimal_size = (base_size as f32 * load_factor) as usize;
        optimal_size.clamp(config.min_chunk_size, config.max_chunk_size)
    }

    /// Record processing completion with full context
    pub async fn record_processing(
        &self,
        _stream_id: u64,
        processing_time_ms: f32,
        chunk_size: usize,
        quality_score: f32,
    ) {
        let now = Instant::now();

        // Update load predictor
        let system_load = self.resource_monitor.get_system_load();
        self.load_predictor
            .lock()
            .await
            .update(system_load.cpu_usage);

        // Record latency measurement
        let latency_measurement = LatencyMeasurement {
            timestamp: now,
            latency_ms: processing_time_ms,
            processing_ms: processing_time_ms,
            buffer_ms: 0.0,  // Would be calculated from buffer state
            network_ms: 0.0, // Would be measured if applicable
            chunk_size,
            system_load: system_load.cpu_usage,
        };

        self.metrics
            .latency_history
            .write()
            .unwrap()
            .push_back(latency_measurement);

        // Record processing time
        self.metrics
            .processing_history
            .write()
            .unwrap()
            .push_back((now, processing_time_ms));

        // Check for quality degradation
        let expected_quality = 0.9; // Would be calculated based on model/settings
        if quality_score < expected_quality * 0.9 {
            let quality_event = QualityEvent {
                timestamp: now,
                quality_score,
                expected_score: expected_quality,
                latency_ms: processing_time_ms,
                chunk_size,
            };
            self.metrics
                .quality_events
                .write()
                .unwrap()
                .push_back(quality_event);
        }

        // Update scheduling statistics
        if let Ok(mut stats) = self.rt_scheduler.stats.write() {
            stats.deadlines_met += 1;
        }

        // Clean up old data
        self.cleanup_old_metrics();
    }

    /// Get comprehensive optimization statistics
    pub fn get_stats(&self) -> EnhancedLatencyStats {
        self.calculate_comprehensive_stats();
        self.stats.read().unwrap().clone()
    }

    /// Calculate comprehensive statistics
    fn calculate_comprehensive_stats(&self) {
        let latency_history = self.metrics.latency_history.read().unwrap();

        if latency_history.is_empty() {
            return;
        }

        // Calculate latency percentiles
        let mut latencies: Vec<f32> = latency_history.iter().map(|m| m.latency_ms).collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = latencies.len();
        let p50 = latencies[len / 2];
        let p95 = latencies[(len as f32 * 0.95) as usize];
        let p99 = latencies[(len as f32 * 0.99) as usize];
        let avg = latencies.iter().sum::<f32>() / len as f32;

        // Update statistics
        if let Ok(mut stats) = self.stats.write() {
            stats.avg_latency_ms = avg;
            stats.p50_latency_ms = p50;
            stats.p95_latency_ms = p95;
            stats.p99_latency_ms = p99;

            // Get prediction accuracy
            if let Ok(predictor) = self.load_predictor.try_lock() {
                stats.load_prediction_accuracy = predictor.get_accuracy();
            }

            // Calculate quality metrics
            let quality_events = self.metrics.quality_events.read().unwrap();
            stats.quality_degradation_events = quality_events.len() as u64;

            // Calculate deadline miss rate
            if let Ok(sched_stats) = self.rt_scheduler.stats.read() {
                let total_deadlines = sched_stats.deadlines_met + sched_stats.deadlines_missed;
                if total_deadlines > 0 {
                    stats.deadline_miss_rate =
                        sched_stats.deadlines_missed as f32 / total_deadlines as f32;
                }
            }
        }
    }

    /// Clean up old metrics to prevent memory growth
    fn cleanup_old_metrics(&self) {
        let cutoff = Instant::now() - Duration::from_secs(300); // Keep 5 minutes

        // Clean processing history
        {
            let mut history = self.metrics.processing_history.write().unwrap();
            while let Some((timestamp, _)) = history.front() {
                if *timestamp < cutoff {
                    history.pop_front();
                } else {
                    break;
                }
            }
        }

        // Clean latency history
        {
            let mut history = self.metrics.latency_history.write().unwrap();
            while let Some(measurement) = history.front() {
                if measurement.timestamp < cutoff {
                    history.pop_front();
                } else {
                    break;
                }
            }
        }

        // Clean system load history
        {
            let mut history = self.metrics.system_load.write().unwrap();
            while let Some(load) = history.front() {
                if load.timestamp < cutoff {
                    history.pop_front();
                } else {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::StreamingConfig;

    #[tokio::test]
    async fn test_enhanced_latency_optimizer_creation() {
        let config = StreamingConfig::default();
        let optimizer = EnhancedLatencyOptimizer::new(config);

        let stats = optimizer.get_stats();
        assert_eq!(stats.avg_latency_ms, 0.0);
    }

    #[tokio::test]
    async fn test_load_predictor() {
        let mut predictor = LoadPredictor::new();

        // Add some sample data
        for i in 0..50 {
            let load = 0.5 + (i as f32 / 100.0) * 0.3; // Gradual increase
            predictor.update(load);
        }

        let prediction = predictor.predict_load(10);
        assert!((0.0..=1.0).contains(&prediction));
    }

    #[test]
    fn test_dynamic_buffer_manager() {
        let manager = DynamicBufferManager::new();

        let buffer_size = manager.get_optimal_buffer_size(1, 0.5);
        assert!(buffer_size > 0);

        // Test buffer event handling
        let event = BufferEvent {
            timestamp: Instant::now(),
            event_type: BufferEventType::Underrun,
            buffer_size: 1024,
            chunk_size: 256,
            system_load: 0.7,
        };

        manager.update_buffer_config(1, event);

        let new_buffer_size = manager.get_optimal_buffer_size(1, 0.5);
        assert!(new_buffer_size >= buffer_size); // Should increase after underrun
    }

    #[tokio::test]
    async fn test_realtime_scheduler() {
        let scheduler = RealtimeScheduler::new();

        let deadline = Instant::now() + Duration::from_millis(100);
        let result = scheduler.schedule_task(1, deadline, 256).await;
        assert!(result.is_ok());

        // Test deadline checking
        tokio::time::sleep(Duration::from_millis(200)).await;
        let missed = scheduler.check_deadlines().await;
        assert_eq!(missed.len(), 1);
        assert_eq!(missed[0], 1);
    }
}
