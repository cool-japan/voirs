//! Quality settings, adaptive quality, and performance monitoring

use super::audio::AudioQualitySettings;
use super::network::NetworkThresholds;
use super::spatial::SpatialQualitySettings;
use super::types::*;
use serde::{Deserialize, Serialize};

/// Quality settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySettings {
    /// Audio quality preferences
    pub audio_quality: AudioQualitySettings,

    /// Spatial quality preferences
    pub spatial_quality: SpatialQualitySettings,

    /// Adaptive quality settings
    pub adaptive_quality: AdaptiveQualitySettings,

    /// Performance monitoring
    pub performance_monitoring: PerformanceMonitoringSettings,
}

/// Adaptive quality settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveQualitySettings {
    /// Enable adaptive quality
    pub enabled: bool,

    /// Quality adaptation algorithm
    pub algorithm: QualityAdaptationAlgorithm,

    /// Adaptation speed
    pub adaptation_speed: AdaptationSpeed,

    /// Quality bounds
    pub quality_bounds: QualityBounds,

    /// Network condition thresholds
    pub network_thresholds: NetworkThresholds,
}

/// Quality bounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityBounds {
    /// Minimum quality level
    pub min_quality: f32,

    /// Maximum quality level
    pub max_quality: f32,

    /// Quality step size
    pub step_size: f32,
}

/// Performance monitoring settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitoringSettings {
    /// Enable monitoring
    pub enabled: bool,

    /// Monitoring interval (ms)
    pub interval: u32,

    /// Metrics to collect
    pub metrics: Vec<PerformanceMetric>,

    /// History retention (seconds)
    pub history_retention: u32,
}

/// Quality information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityInfo {
    /// Quality score (0.0-1.0)
    pub quality_score: f32,

    /// Estimated latency (ms)
    pub estimated_latency: f32,

    /// Packet loss percentage
    pub packet_loss: f32,

    /// Jitter (ms)
    pub jitter: f32,

    /// Signal-to-noise ratio (dB)
    pub snr: f32,
}

/// Performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// CPU usage average (%)
    pub avg_cpu_usage: f32,

    /// Peak CPU usage (%)
    pub peak_cpu_usage: f32,

    /// Memory usage average (MB)
    pub avg_memory_usage: f32,

    /// Peak memory usage (MB)
    pub peak_memory_usage: f32,

    /// Audio processing time (ms)
    pub audio_processing_time: f32,

    /// Network processing time (ms)
    pub network_processing_time: f32,
}

// Default implementations

impl Default for QualitySettings {
    fn default() -> Self {
        Self {
            audio_quality: AudioQualitySettings::default(),
            spatial_quality: SpatialQualitySettings::default(),
            adaptive_quality: AdaptiveQualitySettings::default(),
            performance_monitoring: PerformanceMonitoringSettings::default(),
        }
    }
}

impl Default for AdaptiveQualitySettings {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: QualityAdaptationAlgorithm::Hybrid,
            adaptation_speed: AdaptationSpeed::Medium,
            quality_bounds: QualityBounds::default(),
            network_thresholds: NetworkThresholds::default(),
        }
    }
}

impl Default for QualityBounds {
    fn default() -> Self {
        Self {
            min_quality: 0.3,
            max_quality: 1.0,
            step_size: 0.1,
        }
    }
}

impl Default for PerformanceMonitoringSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: 1000, // 1 second
            metrics: vec![
                PerformanceMetric::AudioLatency,
                PerformanceMetric::NetworkLatency,
                PerformanceMetric::PacketLoss,
                PerformanceMetric::AudioQuality,
            ],
            history_retention: 300, // 5 minutes
        }
    }
}
