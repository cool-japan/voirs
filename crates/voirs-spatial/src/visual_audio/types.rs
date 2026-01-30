//! Core types for visual audio integration

use crate::{Position3D, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Visual display interface for spatial audio cues
pub trait VisualDisplay: Send + Sync {
    /// Render a visual effect at the specified position
    fn render_effect(&mut self, effect: &VisualEffect) -> Result<()>;

    /// Clear all visual effects
    fn clear_all(&mut self) -> Result<()>;

    /// Update display refresh
    fn update(&mut self) -> Result<()>;

    /// Check if the display is ready
    fn is_ready(&self) -> bool;

    /// Get display capabilities
    fn capabilities(&self) -> VisualDisplayCapabilities;

    /// Get display identifier
    fn display_id(&self) -> String;
}

/// Visual effect for spatial audio indication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualEffect {
    /// Effect identifier
    pub id: String,

    /// Effect name
    pub name: String,

    /// Visual elements composing this effect
    pub elements: Vec<VisualElement>,

    /// Effect duration
    pub duration: Duration,

    /// Whether effect should loop
    pub looping: bool,

    /// Effect priority (higher numbers take precedence)
    pub priority: u8,

    /// 3D position for the visual effect
    pub position: Position3D,

    /// Associated audio source
    pub audio_source_id: Option<String>,
}

/// Individual visual element within an effect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualElement {
    /// Start time within the effect
    pub start_time: Duration,

    /// Element duration
    pub duration: Duration,

    /// Visual element type
    pub element_type: VisualElementType,

    /// Color information
    pub color: ColorRGBA,

    /// Intensity/brightness (0.0 to 1.0)
    pub intensity: f32,

    /// Size/scale factor
    pub size: f32,

    /// Animation parameters
    pub animation: Option<AnimationParams>,

    /// Distance-based attenuation
    pub distance_attenuation: f32,
}

/// Types of visual elements
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualElementType {
    /// Point light indicator
    PointLight,

    /// Directional light beam
    DirectionalLight,

    /// Particle effect
    ParticleEffect,

    /// 3D geometric shape
    Shape(ShapeType),

    /// Text/label display
    Text,

    /// Progress bar/meter
    ProgressBar,

    /// Waveform visualization
    Waveform,

    /// Frequency spectrum display
    Spectrum,

    /// Custom visual effect
    Custom(String),
}

/// Shape types for visual elements
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeType {
    /// Sphere
    Sphere,

    /// Cube
    Cube,

    /// Cylinder
    Cylinder,

    /// Cone (directional indicator)
    Cone,

    /// Ring/circle
    Ring,

    /// Arrow (directional)
    Arrow,
}

/// Color representation with alpha
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ColorRGBA {
    /// Red component (0.0-1.0)
    pub r: f32,

    /// Green component (0.0-1.0)
    pub g: f32,

    /// Blue component (0.0-1.0)
    pub b: f32,

    /// Alpha/transparency (0.0-1.0)
    pub a: f32,
}

/// Animation parameters for visual elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationParams {
    /// Animation type
    pub animation_type: AnimationType,

    /// Animation speed multiplier
    pub speed: f32,

    /// Animation amplitude
    pub amplitude: f32,

    /// Phase offset
    pub phase: f32,

    /// Easing function
    pub easing: EasingFunction,
}

/// Animation types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnimationType {
    /// Static (no animation)
    Static,

    /// Pulsing intensity
    Pulse,

    /// Smooth fade in/out
    Fade,

    /// Rotation around axis
    Rotate,

    /// Scaling up/down
    Scale,

    /// Position oscillation
    Oscillate,

    /// Spiral movement
    Spiral,

    /// Custom animation
    Custom(String),
}

/// Easing functions for smooth animation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EasingFunction {
    /// Linear interpolation
    Linear,

    /// Ease in (slow start)
    EaseIn,

    /// Ease out (slow end)
    EaseOut,

    /// Ease in-out (slow start and end)
    EaseInOut,

    /// Bounce effect
    Bounce,

    /// Elastic effect
    Elastic,
}

/// Visual display capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualDisplayCapabilities {
    /// Maximum simultaneous effects
    pub max_concurrent_effects: usize,

    /// Supported visual element types
    pub supported_elements: Vec<VisualElementType>,

    /// Color depth (bits per channel)
    pub color_depth: u8,

    /// Refresh rate (Hz)
    pub refresh_rate: f32,

    /// 3D positioning support
    pub spatial_support: bool,

    /// Animation support
    pub animation_support: bool,

    /// Transparency/alpha support
    pub alpha_support: bool,

    /// Display resolution
    pub resolution: (u32, u32),

    /// Field of view (degrees)
    pub field_of_view: Option<f32>,

    /// Device-specific features
    pub features: HashMap<String, String>,
}

/// Direction zones for color coding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DirectionZone {
    /// Front (±30 degrees)
    Front,

    /// Left side (30-150 degrees)
    Left,

    /// Back (150-210 degrees)
    Back,

    /// Right side (210-330 degrees)
    Right,

    /// Above listener
    Above,

    /// Below listener
    Below,
}

/// Performance metrics for visual audio processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualAudioMetrics {
    /// Visual processing latency (ms)
    pub processing_latency: f32,

    /// Audio-visual synchronization accuracy (ms RMS error)
    pub sync_accuracy: f32,

    /// Active effects count
    pub active_effects: usize,

    /// Frame rate (FPS)
    pub frame_rate: f32,

    /// GPU utilization percentage
    pub gpu_utilization: f32,

    /// Effect cache hit rate
    pub cache_hit_rate: f32,

    /// User satisfaction rating
    pub user_satisfaction: f32,

    /// Resource usage statistics
    pub resource_usage: VisualResourceUsage,
}

/// Visual resource usage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualResourceUsage {
    /// CPU usage percentage
    pub cpu_usage: f32,

    /// GPU memory usage (MB)
    pub gpu_memory_usage: f32,

    /// System memory usage (MB)
    pub system_memory_usage: f32,

    /// Effect library size
    pub effect_library_size: usize,

    /// Active display count
    pub active_displays: usize,

    /// Render queue size
    pub render_queue_size: usize,
}

// Helper structs for internal processing

#[derive(Debug)]
pub(crate) struct VisualEvent {
    pub(crate) source_id: String,
    pub(crate) event_type: VisualEventType,
    pub(crate) intensity: f32,
    pub(crate) color_hint: Option<ColorRGBA>,
    pub(crate) timestamp: Instant,
}

#[derive(Debug)]
pub(crate) struct SpatialVisualEvent {
    pub(crate) base_event: VisualEvent,
    pub(crate) position: Position3D,
    pub(crate) distance: f32,
    pub(crate) attenuation: f32,
    pub(crate) direction_zone: DirectionZone,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum VisualEventType {
    FrequencyBand(FrequencyBand),
    Onset,
    Beat,
    Downbeat,
    SpectralChange,
    Silence,
    Custom(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrequencyBand {
    Low,
    Mid,
    High,
}

// Default implementations

impl Default for VisualAudioMetrics {
    fn default() -> Self {
        Self {
            processing_latency: 0.0,
            sync_accuracy: 0.0,
            active_effects: 0,
            frame_rate: 0.0,
            gpu_utilization: 0.0,
            cache_hit_rate: 0.0,
            user_satisfaction: 5.0,
            resource_usage: VisualResourceUsage::default(),
        }
    }
}

impl Default for VisualResourceUsage {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            gpu_memory_usage: 0.0,
            system_memory_usage: 0.0,
            effect_library_size: 0,
            active_displays: 0,
            render_queue_size: 0,
        }
    }
}
