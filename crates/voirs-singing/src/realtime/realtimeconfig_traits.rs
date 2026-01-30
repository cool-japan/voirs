//! # RealtimeConfig - Trait Implementations
//!
//! This module contains trait implementations for `RealtimeConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::RealtimeConfig;

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            target_latency: 50.0,
            buffer_size: 512,
            sample_rate: 44100,
            thread_count: 2,
            low_latency_mode: true,
            precompute_buffer: 2048,
            quality_vs_speed: 0.7,
            realtime_effects: true,
            voice_switch_latency: 100.0,
            ultra_low_latency_mode: false,
            midi_controller_support: false,
            expression_pedal_support: false,
            loop_station_enabled: false,
        }
    }
}
