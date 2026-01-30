//! Audio-visual synchronization types and state management

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Visual synchronization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualSyncSettings {
    /// Audio-visual latency compensation (ms)
    pub latency_compensation: f32,

    /// Synchronization tolerance (ms)
    pub sync_tolerance: f32,

    /// Visual prediction lookahead (ms)
    pub prediction_lookahead: f32,

    /// Frame rate synchronization
    pub frame_sync: bool,

    /// V-sync enable
    pub vsync: bool,
}

/// Visual synchronization state
pub(crate) struct VisualSyncState {
    /// Audio timeline reference
    pub(crate) audio_timeline: Instant,

    /// Visual timeline reference
    pub(crate) visual_timeline: Instant,

    /// Measured latency offset
    pub(crate) measured_latency: Duration,

    /// Frame timing buffer
    pub(crate) frame_timing: Vec<FrameTiming>,

    /// Sync quality metrics
    pub(crate) sync_quality: SyncQualityMetrics,
}

/// Frame timing information
#[derive(Debug, Clone)]
pub(crate) struct FrameTiming {
    /// Frame timestamp
    pub(crate) timestamp: Instant,

    /// Frame duration
    pub(crate) duration: Duration,

    /// Audio-visual offset
    pub(crate) av_offset: Duration,

    /// Sync quality score
    pub(crate) quality_score: f32,
}

/// Synchronization quality metrics
#[derive(Debug, Clone)]
pub(crate) struct SyncQualityMetrics {
    /// Average sync error (ms)
    pub(crate) avg_sync_error: f32,

    /// Maximum sync error (ms)
    pub(crate) max_sync_error: f32,

    /// Sync stability (coefficient of variation)
    pub(crate) sync_stability: f32,

    /// Frames in sync (percentage)
    pub(crate) frames_in_sync: f32,
}

// Default implementations

impl Default for VisualSyncSettings {
    fn default() -> Self {
        Self {
            latency_compensation: 10.0, // 10ms default compensation
            sync_tolerance: 5.0,
            prediction_lookahead: 50.0,
            frame_sync: true,
            vsync: true,
        }
    }
}

impl VisualSyncState {
    pub(crate) fn new() -> Self {
        let now = Instant::now();
        Self {
            audio_timeline: now,
            visual_timeline: now,
            measured_latency: Duration::from_millis(10),
            frame_timing: Vec::new(),
            sync_quality: SyncQualityMetrics {
                avg_sync_error: 0.0,
                max_sync_error: 0.0,
                sync_stability: 1.0,
                frames_in_sync: 100.0,
            },
        }
    }
}
