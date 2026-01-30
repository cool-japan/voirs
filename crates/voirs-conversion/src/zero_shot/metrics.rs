//! Metrics and performance tracking for zero-shot conversion

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Zero-shot conversion metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroShotMetrics {
    /// Number of successful conversions
    pub successful_conversions: u64,

    /// Number of failed conversions
    pub failed_conversions: u64,

    /// Average processing time (ms)
    pub avg_processing_time: f32,

    /// Average quality score
    pub avg_quality_score: f32,

    /// Cache hit rate
    pub cache_hit_rate: f32,

    /// Reference database utilization
    pub db_utilization: f32,

    /// Performance metrics
    pub performance: PerformanceMetrics,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// CPU usage percentage
    pub cpu_usage: f32,

    /// Memory usage (MB)
    pub memory_usage: f32,

    /// GPU usage percentage
    pub gpu_usage: Option<f32>,

    /// Real-time factor
    pub real_time_factor: f32,

    /// Throughput (conversions per second)
    pub throughput: f32,
}

/// Cached conversion result
#[derive(Debug, Clone)]
pub struct CachedConversion {
    /// Conversion result
    pub result: Vec<f32>,

    /// Quality score
    pub quality_score: f32,

    /// Processing time
    pub processing_time: Duration,

    /// Cache timestamp
    pub timestamp: Instant,

    /// Usage count
    pub usage_count: u32,
}

impl Default for ZeroShotMetrics {
    fn default() -> Self {
        Self {
            successful_conversions: 0,
            failed_conversions: 0,
            avg_processing_time: 0.0,
            avg_quality_score: 0.0,
            cache_hit_rate: 0.0,
            db_utilization: 0.0,
            performance: PerformanceMetrics {
                cpu_usage: 0.0,
                memory_usage: 0.0,
                gpu_usage: None,
                real_time_factor: 1.0,
                throughput: 0.0,
            },
        }
    }
}
