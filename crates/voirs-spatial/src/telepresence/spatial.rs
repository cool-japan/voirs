//! Spatial telepresence, head tracking, and presence indicators

use super::audio::FrequencyFilterSettings;
use super::types::*;
use crate::Position3D;
use serde::{Deserialize, Serialize};

/// Spatial telepresence settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialTelepresenceSettings {
    /// Enable spatial audio
    pub spatial_enabled: bool,

    /// Spatial quality level
    pub spatial_quality: SpatialQualityLevel,

    /// Head tracking integration
    pub head_tracking: HeadTrackingSettings,

    /// Environmental awareness
    pub environmental_awareness: EnvironmentalAwarenessSettings,

    /// Presence indicators
    pub presence_indicators: PresenceIndicatorSettings,
}

/// Head tracking settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadTrackingSettings {
    /// Enable head tracking
    pub enabled: bool,

    /// Tracking source
    pub tracking_source: TrackingSource,

    /// Prediction settings
    pub prediction: TrackingPredictionSettings,

    /// Smoothing settings
    pub smoothing: TrackingSmoothingSettings,
}

/// Tracking prediction settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingPredictionSettings {
    /// Enable prediction
    pub enabled: bool,

    /// Prediction horizon (ms)
    pub horizon: f32,

    /// Prediction algorithm
    pub algorithm: PredictionAlgorithm,

    /// Confidence threshold
    pub confidence_threshold: f32,
}

/// Tracking smoothing settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingSmoothingSettings {
    /// Position smoothing factor
    pub position_smoothing: f32,

    /// Orientation smoothing factor
    pub orientation_smoothing: f32,

    /// Velocity smoothing factor
    pub velocity_smoothing: f32,

    /// Jitter reduction
    pub jitter_reduction: f32,
}

/// Environmental awareness settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalAwarenessSettings {
    /// Enable environmental audio
    pub enabled: bool,

    /// Ambient sound sharing
    pub ambient_sharing: AmbientSharingSettings,

    /// Background noise handling
    pub background_noise: BackgroundNoiseSettings,

    /// Acoustic echo from environment
    pub acoustic_echo: AcousticEchoSettings,
}

/// Ambient sound sharing settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientSharingSettings {
    /// Enable ambient sharing
    pub enabled: bool,

    /// Ambient level (0.0-1.0)
    pub level: f32,

    /// Frequency filtering
    pub frequency_filtering: FrequencyFilterSettings,

    /// Spatial ambient processing
    pub spatial_processing: bool,
}

/// Background noise settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundNoiseSettings {
    /// Noise suppression level
    pub suppression_level: f32,

    /// Adaptive suppression
    pub adaptive_suppression: bool,

    /// Noise gate threshold
    pub gate_threshold: f32,

    /// Noise profiling
    pub noise_profiling: NoiseProfilingSettings,
}

/// Noise profiling settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseProfilingSettings {
    /// Enable automatic profiling
    pub enabled: bool,

    /// Profiling duration (seconds)
    pub duration: f32,

    /// Update interval (seconds)
    pub update_interval: f32,

    /// Profile adaptation rate
    pub adaptation_rate: f32,
}

/// Acoustic echo settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticEchoSettings {
    /// Echo detection sensitivity
    pub detection_sensitivity: f32,

    /// Echo suppression strength
    pub suppression_strength: f32,

    /// Echo path modeling
    pub path_modeling: bool,

    /// Nonlinear echo processing
    pub nonlinear_processing: bool,
}

/// Presence indicator settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresenceIndicatorSettings {
    /// Enable presence indicators
    pub enabled: bool,

    /// Visual indicators
    pub visual_indicators: VisualPresenceSettings,

    /// Audio indicators
    pub audio_indicators: AudioPresenceSettings,

    /// Breathing room detection
    pub breathing_room: BreathingRoomSettings,
}

/// Visual presence settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualPresenceSettings {
    /// Show speaking indicator
    pub speaking_indicator: bool,

    /// Show position indicator
    pub position_indicator: bool,

    /// Show attention indicator
    pub attention_indicator: bool,

    /// Indicator style
    pub indicator_style: IndicatorStyle,
}

/// Audio presence settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioPresenceSettings {
    /// Spatial breathing sounds
    pub breathing_sounds: bool,

    /// Footstep simulation
    pub footsteps: bool,

    /// Cloth/movement sounds
    pub movement_sounds: bool,

    /// Presence audio level
    pub presence_level: f32,
}

/// Breathing room settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreathingRoomSettings {
    /// Enable breathing room
    pub enabled: bool,

    /// Personal space radius (meters)
    pub personal_space: f32,

    /// Comfort distance (meters)
    pub comfort_distance: f32,

    /// Audio adjustments for proximity
    pub proximity_adjustments: ProximityAdjustments,
}

/// Proximity-based audio adjustments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProximityAdjustments {
    /// Volume adjustment for close proximity
    pub volume_adjustment: f32,

    /// Frequency response adjustment
    pub frequency_adjustment: super::audio::FrequencyResponseAdjustment,

    /// Reverb adjustment
    pub reverb_adjustment: f32,

    /// Intimacy enhancement
    pub intimacy_enhancement: bool,
}

/// User position in room
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPosition {
    /// User identifier
    pub user_id: String,

    /// Position in room
    pub position: Position3D,

    /// Orientation
    pub orientation: Orientation,

    /// Movement constraints
    pub movement_constraints: MovementConstraints,
}

/// Spatial audio information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialAudioInfo {
    /// Source position
    pub position: Position3D,

    /// Source orientation
    pub orientation: Orientation,

    /// Velocity (for Doppler)
    pub velocity: Option<Velocity>,

    /// Distance from listener
    pub distance: f32,

    /// Spatial quality
    pub spatial_quality: SpatialQualityLevel,
}

/// Spatial quality settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialQualitySettings {
    /// HRTF quality level
    pub hrtf_quality: HrtfQualityLevel,

    /// Room simulation quality
    pub room_quality: RoomQualityLevel,

    /// Distance modeling precision
    pub distance_precision: DistancePrecisionLevel,

    /// Update rate (Hz)
    pub update_rate: f32,
}

/// Spatial audio statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialAudioStats {
    /// Position updates received
    pub position_updates: u64,

    /// Spatial processing accuracy
    pub spatial_accuracy: f32,

    /// HRTF processing efficiency
    pub hrtf_efficiency: f32,

    /// Room simulation performance
    pub room_sim_performance: f32,

    /// Distance calculations performed
    pub distance_calculations: u64,
}

// Default implementations

impl Default for SpatialTelepresenceSettings {
    fn default() -> Self {
        Self {
            spatial_enabled: true,
            spatial_quality: SpatialQualityLevel::Full3D,
            head_tracking: HeadTrackingSettings::default(),
            environmental_awareness: EnvironmentalAwarenessSettings::default(),
            presence_indicators: PresenceIndicatorSettings::default(),
        }
    }
}

impl Default for HeadTrackingSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            tracking_source: TrackingSource::VRHeadset,
            prediction: TrackingPredictionSettings::default(),
            smoothing: TrackingSmoothingSettings::default(),
        }
    }
}

impl Default for TrackingPredictionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            horizon: 50.0, // 50ms prediction
            algorithm: PredictionAlgorithm::Kalman,
            confidence_threshold: 0.7,
        }
    }
}

impl Default for TrackingSmoothingSettings {
    fn default() -> Self {
        Self {
            position_smoothing: 0.8,
            orientation_smoothing: 0.85,
            velocity_smoothing: 0.7,
            jitter_reduction: 0.9,
        }
    }
}

impl Default for EnvironmentalAwarenessSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            ambient_sharing: AmbientSharingSettings::default(),
            background_noise: BackgroundNoiseSettings::default(),
            acoustic_echo: AcousticEchoSettings::default(),
        }
    }
}

impl Default for AmbientSharingSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            level: 0.3,
            frequency_filtering: FrequencyFilterSettings::default(),
            spatial_processing: true,
        }
    }
}

impl Default for BackgroundNoiseSettings {
    fn default() -> Self {
        Self {
            suppression_level: 0.8,
            adaptive_suppression: true,
            gate_threshold: -40.0, // -40 dB
            noise_profiling: NoiseProfilingSettings::default(),
        }
    }
}

impl Default for NoiseProfilingSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            duration: 5.0,         // 5 seconds
            update_interval: 30.0, // 30 seconds
            adaptation_rate: 0.1,
        }
    }
}

impl Default for AcousticEchoSettings {
    fn default() -> Self {
        Self {
            detection_sensitivity: 0.7,
            suppression_strength: 0.8,
            path_modeling: true,
            nonlinear_processing: true,
        }
    }
}

impl Default for PresenceIndicatorSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            visual_indicators: VisualPresenceSettings::default(),
            audio_indicators: AudioPresenceSettings::default(),
            breathing_room: BreathingRoomSettings::default(),
        }
    }
}

impl Default for VisualPresenceSettings {
    fn default() -> Self {
        Self {
            speaking_indicator: true,
            position_indicator: true,
            attention_indicator: false,
            indicator_style: IndicatorStyle::Standard,
        }
    }
}

impl Default for AudioPresenceSettings {
    fn default() -> Self {
        Self {
            breathing_sounds: false,
            footsteps: false,
            movement_sounds: false,
            presence_level: 0.2,
        }
    }
}

impl Default for BreathingRoomSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            personal_space: 1.0,   // 1 meter
            comfort_distance: 2.0, // 2 meters
            proximity_adjustments: ProximityAdjustments::default(),
        }
    }
}

impl Default for ProximityAdjustments {
    fn default() -> Self {
        Self {
            volume_adjustment: -3.0, // -3 dB for close proximity
            frequency_adjustment: super::audio::FrequencyResponseAdjustment::default(),
            reverb_adjustment: -0.2,
            intimacy_enhancement: true,
        }
    }
}

impl Default for SpatialQualitySettings {
    fn default() -> Self {
        Self {
            hrtf_quality: HrtfQualityLevel::High,
            room_quality: RoomQualityLevel::Standard,
            distance_precision: DistancePrecisionLevel::High,
            update_rate: 90.0, // 90 Hz
        }
    }
}
