//! Advanced Streaming Synthesis (Phase 3)
//!
//! This module implements ultra-low latency streaming synthesis with:
//! - Zero-copy circular buffer architecture
//! - Lock-free ring buffer for audio chunks
//! - Predictive pre-synthesis for upcoming notes
//! - Adaptive quality scaling based on system load
//! - Real-time CPU monitoring and dynamic quality adjustment
//!
//! Target metrics:
//! - Streaming latency: <10ms (down from 50ms)
//! - CPU overhead: 40% reduction in streaming mode
//! - Quality maintenance: MOS 4.0+ even at lowest quality
//! - Jitter: <1ms variance in frame timing

pub mod adaptive_quality;
pub mod cpu_monitor;
pub mod lock_free_ring_buffer;
pub mod predictive_synthesis;
pub mod zero_copy_streaming;

// Re-export public types
pub use adaptive_quality::*;
pub use cpu_monitor::*;
pub use lock_free_ring_buffer::*;
pub use predictive_synthesis::*;
pub use zero_copy_streaming::*;

use crate::{Error, MusicalNote, MusicalScore, VoiceCharacteristics};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Advanced streaming synthesis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedStreamingConfig {
    /// Target latency in milliseconds (default: 10ms)
    pub target_latency_ms: u32,

    /// Ring buffer size in samples
    pub ring_buffer_size: usize,

    /// Number of audio chunks to pre-allocate
    pub chunk_count: usize,

    /// Samples per chunk
    pub chunk_size: usize,

    /// Enable predictive synthesis
    pub enable_predictive: bool,

    /// Look-ahead time for predictive synthesis (seconds)
    pub lookahead_time: f32,

    /// Enable adaptive quality scaling
    pub enable_adaptive_quality: bool,

    /// CPU usage threshold for quality scaling (0.0-1.0)
    pub cpu_threshold: f32,

    /// Minimum quality level (0.0-1.0)
    pub min_quality: f32,

    /// Maximum quality level (0.0-1.0)
    pub max_quality: f32,

    /// Sample rate
    pub sample_rate: u32,
}

impl Default for AdvancedStreamingConfig {
    fn default() -> Self {
        Self {
            target_latency_ms: 10,
            ring_buffer_size: 8192,
            chunk_count: 16,
            chunk_size: 256,
            enable_predictive: true,
            lookahead_time: 0.5,
            enable_adaptive_quality: true,
            cpu_threshold: 0.75,
            min_quality: 0.6,
            max_quality: 1.0,
            sample_rate: 48000,
        }
    }
}

/// Streaming performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamingMetrics {
    /// Current latency in milliseconds
    pub current_latency_ms: f32,

    /// Average latency over last 100 frames
    pub avg_latency_ms: f32,

    /// Peak latency observed
    pub peak_latency_ms: f32,

    /// Latency jitter (standard deviation)
    pub jitter_ms: f32,

    /// Current CPU usage (0.0-1.0)
    pub cpu_usage: f32,

    /// Current quality level (0.0-1.0)
    pub quality_level: f32,

    /// Number of quality adjustments
    pub quality_adjustments: u64,

    /// Number of buffer underruns
    pub buffer_underruns: u64,

    /// Number of frames synthesized
    pub frames_synthesized: u64,

    /// Total samples produced
    pub total_samples: u64,

    /// Predictive synthesis cache hits
    pub predictive_hits: u64,

    /// Predictive synthesis cache misses
    pub predictive_misses: u64,
}

impl StreamingMetrics {
    /// Calculate predictive hit rate
    pub fn predictive_hit_rate(&self) -> f32 {
        let total = self.predictive_hits + self.predictive_misses;
        if total == 0 {
            0.0
        } else {
            self.predictive_hits as f32 / total as f32
        }
    }

    /// Check if latency target is met
    pub fn is_latency_target_met(&self, target_ms: f32) -> bool {
        self.avg_latency_ms <= target_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_streaming_config_default() {
        let config = AdvancedStreamingConfig::default();
        assert_eq!(config.target_latency_ms, 10);
        assert_eq!(config.sample_rate, 48000);
        assert!(config.enable_predictive);
        assert!(config.enable_adaptive_quality);
    }

    #[test]
    fn test_streaming_metrics_predictive_hit_rate() {
        let mut metrics = StreamingMetrics::default();
        metrics.predictive_hits = 80;
        metrics.predictive_misses = 20;

        let hit_rate = metrics.predictive_hit_rate();
        assert!((hit_rate - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_streaming_metrics_zero_predictive() {
        let metrics = StreamingMetrics::default();
        assert_eq!(metrics.predictive_hit_rate(), 0.0);
    }

    #[test]
    fn test_streaming_metrics_latency_target() {
        let mut metrics = StreamingMetrics::default();
        metrics.avg_latency_ms = 8.5;

        assert!(metrics.is_latency_target_met(10.0));
        assert!(!metrics.is_latency_target_met(5.0));
    }
}
