//! Real-time emotion adaptation and control system
//!
//! This module provides real-time emotion adaptation capabilities that can:
//! - Respond to audio input characteristics
//! - Adapt to external emotion signals
//! - Provide low-latency emotion updates
//! - Maintain emotion consistency across streaming synthesis
//!
//! # Module Structure
//!
//! - `config`: Configuration types for real-time and streaming emotion control
//! - `signal`: External emotion signal types and management
//! - `adapter`: Core real-time emotion adaptation engine
//! - `streaming`: Streaming emotion controller for real-time synthesis
//! - `metrics`: Performance metrics and monitoring
//! - `utils`: Audio analysis and processing utilities
//! - `tests`: Test suite for real-time functionality

pub mod adapter;
pub mod config;
pub mod metrics;
pub mod signal;
pub mod streaming;
pub mod utils;

#[cfg(test)]
mod tests;

// Re-export main types for convenient access
pub use adapter::RealtimeEmotionAdapter;
pub use config::{RealtimeEmotionConfig, StreamingConfig};
pub use metrics::{AdaptationMetrics, StreamingMetrics};
pub use signal::{AudioCharacteristics, EmotionSignal};
pub use streaming::{StreamingEmotionController, StreamingSession};

// Re-export utility functions for internal use
pub(crate) use utils::*;
