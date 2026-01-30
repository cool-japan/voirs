//! Performance metrics for real-time emotion adaptation
//!
//! This module provides comprehensive metrics tracking for both:
//! - Emotion adaptation performance (update frequency, transitions)
//! - Streaming synthesis quality (latency, sessions, throughput)

use std::time::Instant;

/// Performance metrics for emotion adaptation
#[derive(Debug, Clone)]
pub struct AdaptationMetrics {
    /// Number of emotion updates
    pub update_count: u64,
    /// Average update frequency
    pub avg_update_frequency: f32,
    /// Number of transitions started
    pub transition_count: u64,
    /// Average transition duration
    pub avg_transition_duration: f32,
    /// Last update time
    pub last_update: Option<Instant>,
}

impl AdaptationMetrics {
    /// Create a new `AdaptationMetrics` instance with default values
    ///
    /// # Returns
    /// A new metrics instance with all counters set to zero
    pub fn new() -> Self {
        Self {
            update_count: 0,
            avg_update_frequency: 0.0,
            transition_count: 0,
            avg_transition_duration: 0.0,
            last_update: None,
        }
    }

    /// Update metrics with a new timestamp
    ///
    /// # Arguments
    /// * `now` - Current timestamp for computing update frequency
    pub fn update(&mut self, now: Instant) {
        self.update_count += 1;

        if let Some(last) = self.last_update {
            let delta = now.duration_since(last).as_secs_f32();
            let frequency = 1.0 / delta;

            // Exponential moving average
            let alpha = 0.1;
            self.avg_update_frequency =
                alpha * frequency + (1.0 - alpha) * self.avg_update_frequency;
        }

        self.last_update = Some(now);
    }
}

impl Default for AdaptationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Streaming performance metrics
#[derive(Debug, Clone)]
pub struct StreamingMetrics {
    /// Total audio chunks processed
    pub chunks_processed: u64,
    /// Average processing latency per chunk
    pub avg_processing_latency_ms: f32,
    /// Active session count
    pub active_sessions: u32,
    /// Total sessions created
    pub total_sessions_created: u64,
    /// Emotion updates per second
    pub emotion_updates_per_sec: f32,
    /// Buffer underruns count
    pub buffer_underruns: u64,
    /// Last metrics update time
    pub last_update: Instant,
}

impl Default for StreamingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingMetrics {
    /// Create a new `StreamingMetrics` instance with default values
    ///
    /// # Returns
    /// A new metrics instance with all counters set to zero and current timestamp
    pub fn new() -> Self {
        Self {
            chunks_processed: 0,
            avg_processing_latency_ms: 0.0,
            active_sessions: 0,
            total_sessions_created: 0,
            emotion_updates_per_sec: 0.0,
            buffer_underruns: 0,
            last_update: Instant::now(),
        }
    }
}
