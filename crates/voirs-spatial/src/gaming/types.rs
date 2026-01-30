//! Core types and enums for gaming audio integration

use crate::position::SoundSource;
use crate::types::Position3D;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Gaming engine types supported by VoiRS
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GameEngine {
    /// Unity 3D engine
    Unity,
    /// Unreal Engine 4/5
    Unreal,
    /// Godot engine
    Godot,
    /// Custom C/C++ engine
    Custom,
    /// Web-based engine (Three.js, Babylon.js, etc.)
    WebEngine,
    /// Console-specific engines
    Console,
    /// PlayStation 4/5 specific
    PlayStation,
    /// Xbox One/Series X|S specific
    Xbox,
    /// Nintendo Switch specific
    NintendoSwitch,
}

/// Gaming audio source handle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GameAudioSource {
    /// Unique source ID
    pub id: u32,
    /// Source category for mixing
    pub category: AudioCategory,
    /// Priority level (0 = highest, 255 = lowest)
    pub priority: u8,
}

/// Audio categories for gaming
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioCategory {
    /// Player character sounds
    Player,
    /// Non-player character sounds
    Npc,
    /// Environmental sounds
    Environment,
    /// Music and soundtrack
    Music,
    /// Sound effects
    Sfx,
    /// UI sounds
    Ui,
    /// Voice and dialogue
    Voice,
    /// Ambient sounds
    Ambient,
}

/// Distance attenuation settings for gaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttenuationSettings {
    /// Minimum distance (no attenuation)
    pub min_distance: f32,
    /// Maximum distance (silence)
    pub max_distance: f32,
    /// Attenuation curve (linear, logarithmic, exponential)
    pub curve: AttenuationCurve,
    /// Attenuation factor
    pub factor: f32,
}

impl Default for AttenuationSettings {
    fn default() -> Self {
        Self {
            min_distance: 1.0,
            max_distance: 100.0,
            curve: AttenuationCurve::Logarithmic,
            factor: 1.0,
        }
    }
}

/// Distance attenuation curves
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttenuationCurve {
    /// Linear attenuation curve
    Linear,
    /// Logarithmic attenuation curve
    Logarithmic,
    /// Exponential attenuation curve
    Exponential,
    /// Custom attenuation curve
    Custom,
}

/// Gaming performance metrics
#[derive(Debug, Clone, Default)]
pub struct GamingMetrics {
    /// Audio processing time per frame (ms)
    pub audio_time_ms: f32,
    /// Number of active sources
    pub active_sources: u32,
    /// Memory usage (MB)
    pub memory_usage_mb: f32,
    /// CPU usage percentage
    pub cpu_usage_percent: f32,
    /// GPU usage percentage (if available)
    pub gpu_usage_percent: Option<f32>,
    /// Dropped frames count
    pub dropped_frames: u32,
    /// Average FPS
    pub fps: f32,
    /// Audio latency (ms)
    pub latency_ms: f32,
}

/// Internal audio source data
#[derive(Debug, Clone)]
pub(super) struct GameAudioSourceData {
    /// Source handle
    pub handle: GameAudioSource,
    /// Spatial source
    pub spatial_source: SoundSource,
    /// Audio buffer
    pub audio_data: Vec<f32>,
    /// Playback state
    pub state: PlaybackState,
    /// Volume level
    pub volume: f32,
    /// Loop settings
    pub looping: bool,
    /// Distance attenuation settings
    pub attenuation: AttenuationSettings,
}

/// Playback state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PlaybackState {
    Stopped,
    Playing,
    Paused,
    Finished,
}

/// Frame timing helper
#[derive(Debug)]
pub(super) struct FrameTimer {
    pub last_frame: Instant,
    pub frame_count: u32,
    pub fps_accumulator: f32,
    pub fps_timer: Instant,
}

impl Default for FrameTimer {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            last_frame: now,
            frame_count: 0,
            fps_accumulator: 0.0,
            fps_timer: now,
        }
    }
}
