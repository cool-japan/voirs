//! Configuration structures for real-time emotion adaptation and streaming
//!
//! This module provides configuration types for:
//! - Real-time emotion adaptation settings
//! - Streaming emotion control parameters
//! - Performance and quality tuning options
//!
//! These configurations control how emotion adaptation responds to input,
//! manages transitions, and handles streaming synthesis sessions.

use crate::interpolation::{InterpolationConfig, InterpolationMethod};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};

/// Configuration for real-time emotion adaptation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RealtimeEmotionConfig {
    /// Update frequency in Hz
    pub update_frequency: f32,
    /// Buffer size for emotion history
    pub history_buffer_size: usize,
    /// Minimum time between emotion changes in milliseconds
    pub min_change_interval_ms: u64,
    /// Smoothing factor for emotion changes (0.0 = immediate, 1.0 = very smooth)
    pub smoothing_factor: f32,
    /// Enable adaptive response based on audio characteristics
    pub enable_audio_adaptation: bool,
    /// Enable external emotion signal input
    pub enable_external_input: bool,
    /// Maximum emotion change rate per second
    pub max_change_rate: f32,
    /// Interpolation configuration
    pub interpolation: InterpolationConfig,
}

impl RealtimeEmotionConfig {
    /// Create default configuration optimized for real-time use
    pub fn new() -> Self {
        Self::default()
    }

    /// Create configuration optimized for low latency
    pub fn low_latency() -> Self {
        Self {
            update_frequency: 60.0,
            history_buffer_size: 10,
            min_change_interval_ms: 50,
            smoothing_factor: 0.3,
            max_change_rate: 5.0,
            interpolation: InterpolationConfig {
                method: InterpolationMethod::Linear,
                transition_duration_ms: 200,
                change_threshold: 0.05,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create configuration optimized for smooth transitions
    pub fn smooth_transitions() -> Self {
        Self {
            update_frequency: 30.0,
            history_buffer_size: 20,
            min_change_interval_ms: 100,
            smoothing_factor: 0.7,
            max_change_rate: 2.0,
            interpolation: InterpolationConfig {
                method: InterpolationMethod::EaseInOut,
                transition_duration_ms: 1000,
                change_threshold: 0.1,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.update_frequency <= 0.0 || self.update_frequency > 1000.0 {
            return Err(Error::Config(
                "Update frequency must be between 0 and 1000 Hz".to_string(),
            ));
        }

        if self.smoothing_factor < 0.0 || self.smoothing_factor > 1.0 {
            return Err(Error::Config(
                "Smoothing factor must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.max_change_rate <= 0.0 {
            return Err(Error::Config(
                "Max change rate must be positive".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for RealtimeEmotionConfig {
    fn default() -> Self {
        Self {
            update_frequency: 30.0,
            history_buffer_size: 15,
            min_change_interval_ms: 100,
            smoothing_factor: 0.5,
            enable_audio_adaptation: true,
            enable_external_input: true,
            max_change_rate: 3.0,
            interpolation: InterpolationConfig::default(),
        }
    }
}

/// Configuration for streaming emotion control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Audio buffer size in samples
    pub buffer_size: usize,
    /// Maximum concurrent streaming sessions
    pub max_concurrent_sessions: usize,
    /// Stream update interval in milliseconds
    pub stream_update_interval_ms: u64,
    /// Enable adaptive quality based on network conditions
    pub adaptive_quality: bool,
    /// Audio chunk size for processing
    pub chunk_size: usize,
    /// Sample rate for streaming
    pub sample_rate: f32,
    /// Enable emotion interpolation across chunks
    pub enable_chunk_interpolation: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 4096,
            max_concurrent_sessions: 10,
            stream_update_interval_ms: 50,
            adaptive_quality: true,
            chunk_size: 1024,
            sample_rate: 16000.0,
            enable_chunk_interpolation: true,
        }
    }
}
