//! Core types for position tracking and listener/source management

use crate::types::Position3D;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Listener in 3D space with orientation and movement tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Listener {
    /// Current position
    position: Position3D,
    /// Orientation (yaw, pitch, roll) in radians
    orientation: (f32, f32, f32),
    /// Velocity vector
    velocity: Position3D,
    /// Head radius for HRTF calculations
    head_radius: f32,
    /// Inter-aural distance
    interaural_distance: f32,
    /// Movement tracking
    movement_history: Vec<PositionSnapshot>,
    /// Last update time
    #[serde(skip)]
    last_update: Option<Instant>,
}

/// Sound source in 3D space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoundSource {
    /// Unique identifier
    pub id: String,
    /// Current position
    position: Position3D,
    /// Velocity vector
    velocity: Position3D,
    /// Source orientation (for directional sources)
    orientation: Option<(f32, f32, f32)>,
    /// Source type
    source_type: SourceType,
    /// Attenuation parameters
    attenuation: AttenuationParams,
    /// Directivity pattern
    directivity: Option<DirectivityPattern>,
    /// Movement tracking
    movement_history: Vec<PositionSnapshot>,
    /// Active state
    is_active: bool,
    /// Last update time
    #[serde(skip)]
    last_update: Option<Instant>,
}

/// Position snapshot for movement tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSnapshot {
    /// Position at this time
    pub position: Position3D,
    /// Timestamp
    pub timestamp: f64, // Serializable timestamp
    /// Velocity at this time
    pub velocity: Position3D,
}

/// Source type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SourceType {
    /// Point source (omnidirectional)
    Point,
    /// Directional source
    Directional,
    /// Area source
    Area {
        /// Width of the area source in meters
        width: f32,
        /// Height of the area source in meters
        height: f32,
    },
    /// Line source
    Line {
        /// Length of the line source in meters
        length: f32,
    },
    /// Ambient source (environment)
    Ambient,
}

/// Attenuation parameters for sound sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttenuationParams {
    /// Reference distance (distance at which attenuation begins)
    pub reference_distance: f32,
    /// Maximum distance (beyond which sound is inaudible)
    pub max_distance: f32,
    /// Rolloff factor (how quickly sound attenuates)
    pub rolloff_factor: f32,
    /// Attenuation model
    pub model: AttenuationModel,
}

/// Attenuation model types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AttenuationModel {
    /// No attenuation
    None,
    /// Linear attenuation
    Linear,
    /// Inverse distance law
    Inverse,
    /// Inverse square law
    InverseSquare,
    /// Exponential attenuation
    Exponential,
    /// Custom curve
    Custom,
}

/// Directivity pattern for directional sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectivityPattern {
    /// Front gain (0 degrees)
    pub front_gain: f32,
    /// Back gain (180 degrees)
    pub back_gain: f32,
    /// Side gain (90/270 degrees)
    pub side_gain: f32,
    /// Directivity index (sharpness of pattern)
    pub directivity_index: f32,
    /// Frequency-dependent directivity
    pub frequency_response: Vec<FrequencyGain>,
}

/// Frequency-dependent gain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyGain {
    /// Frequency in Hz
    pub frequency: f32,
    /// Gain multiplier
    pub gain: f32,
}

/// Orientation snapshot for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrientationSnapshot {
    /// Orientation at this time (yaw, pitch, roll)
    pub orientation: (f32, f32, f32),
    /// Timestamp
    pub timestamp: f64,
    /// Angular velocity (rad/s)
    pub angular_velocity: (f32, f32, f32),
}

/// 3D bounding box for occlusion detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Box3D {
    /// Minimum corner
    pub min: Position3D,
    /// Maximum corner
    pub max: Position3D,
    /// Material ID
    pub material_id: String,
}

/// Navigation modes for different use cases
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NavigationMode {
    /// Free 6DOF movement
    FreeFlight,
    /// Walking/ground-based movement
    Walking,
    /// Seated experience with head tracking only
    Seated,
    /// Teleport-based movement
    Teleport,
    /// Vehicle/third-person movement
    Vehicle,
}

/// Comfort settings for VR movement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComfortSettings {
    /// Motion sickness reduction (0.0 = disabled, 1.0 = maximum)
    pub motion_sickness_reduction: f32,
    /// Snap turning enabled
    pub snap_turn: bool,
    /// Snap turn degrees
    pub snap_turn_degrees: f32,
    /// Vignetting during movement
    pub movement_vignetting: bool,
    /// Ground reference enabled
    pub ground_reference: bool,
    /// Movement speed multiplier
    pub speed_multiplier: f32,
}

/// Movement constraints and boundaries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MovementConstraints {
    /// Boundary box for movement
    pub boundary: Option<Box3D>,
    /// Maximum movement speed (m/s)
    pub max_speed: f32,
    /// Maximum acceleration (m/s²)
    pub max_acceleration: f32,
    /// Ground height constraint
    pub ground_height: Option<f32>,
    /// Ceiling height constraint
    pub ceiling_height: Option<f32>,
}

/// Movement performance metrics
#[derive(Debug, Clone, Default)]
pub struct MovementMetrics {
    /// Total distance traveled
    pub total_distance: f32,
    /// Average speed
    pub average_speed: f32,
    /// Peak speed
    pub peak_speed: f32,
    /// Movement duration
    pub movement_duration: Duration,
    /// Number of position updates
    pub update_count: usize,
    /// Prediction accuracy (when available)
    pub prediction_accuracy: f32,
}

/// Supported VR/AR platforms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlatformType {
    /// Generic 6DOF tracking
    Generic,
    /// Oculus/Meta platforms
    Oculus,
    /// SteamVR/OpenVR
    SteamVR,
    /// Apple ARKit
    ARKit,
    /// Google ARCore
    ARCore,
    /// Microsoft Mixed Reality
    WMR,
    /// Custom platform
    Custom,
}

/// Platform-specific tracking data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformData {
    /// Device ID or name
    pub device_id: String,
    /// Platform-specific pose data
    pub pose_data: Vec<f32>,
    /// Tracking confidence (0.0 = lost, 1.0 = perfect)
    pub tracking_confidence: f32,
    /// Platform timestamp
    pub platform_timestamp: u64,
    /// Additional platform-specific properties
    pub properties: std::collections::HashMap<String, String>,
}

/// Calibration data for spatial audio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationData {
    /// Head circumference for HRTF adjustment
    pub head_circumference: Option<f32>,
    /// Inter-pupillary distance
    pub ipd: Option<f32>,
    /// Height offset from tracking origin
    pub height_offset: f32,
    /// Forward offset from tracking origin
    pub forward_offset: f32,
    /// Custom HRTF profile if available
    pub custom_hrtf_profile: Option<String>,
}

// Implementations

impl Listener {
    /// Create new listener at origin
    pub fn new() -> Self {
        Self {
            position: Position3D::default(),
            orientation: (0.0, 0.0, 0.0),
            velocity: Position3D::default(),
            head_radius: 0.0875,        // ~8.75cm average head radius
            interaural_distance: 0.175, // ~17.5cm average interaural distance
            movement_history: Vec::new(),
            last_update: None,
        }
    }

    /// Create listener at specific position
    pub fn at_position(position: Position3D) -> Self {
        let mut listener = Self::new();
        listener.position = position;
        listener
    }

    /// Get current position
    pub fn position(&self) -> Position3D {
        self.position
    }

    /// Set position and update velocity
    pub fn set_position(&mut self, position: Position3D) {
        let now = Instant::now();

        // Calculate velocity if we have a previous update
        if let Some(last_time) = self.last_update {
            let time_delta = now.duration_since(last_time).as_secs_f32();
            if time_delta > 0.0 {
                self.velocity = Position3D::new(
                    (position.x - self.position.x) / time_delta,
                    (position.y - self.position.y) / time_delta,
                    (position.z - self.position.z) / time_delta,
                );
            }
        }

        // Add to movement history
        self.movement_history.push(PositionSnapshot {
            position: self.position,
            timestamp: now.elapsed().as_secs_f64(),
            velocity: self.velocity,
        });

        // Limit history size
        if self.movement_history.len() > 100 {
            self.movement_history.remove(0);
        }

        self.position = position;
        self.last_update = Some(now);
    }

    /// Get current orientation
    pub fn orientation(&self) -> (f32, f32, f32) {
        self.orientation
    }

    /// Set orientation
    pub fn set_orientation(&mut self, orientation: (f32, f32, f32)) {
        self.orientation = orientation;
    }

    /// Get current velocity
    pub fn velocity(&self) -> Position3D {
        self.velocity
    }

    /// Get head radius
    pub fn head_radius(&self) -> f32 {
        self.head_radius
    }

    /// Set head radius
    pub fn set_head_radius(&mut self, radius: f32) {
        self.head_radius = radius;
    }

    /// Get interaural distance
    pub fn interaural_distance(&self) -> f32 {
        self.interaural_distance
    }

    /// Set interaural distance
    pub fn set_interaural_distance(&mut self, distance: f32) {
        self.interaural_distance = distance;
    }

    /// Get movement history
    pub fn movement_history(&self) -> &[PositionSnapshot] {
        &self.movement_history
    }

    /// Predict future position based on current velocity
    pub fn predict_position(&self, time_ahead: Duration) -> Position3D {
        let delta_time = time_ahead.as_secs_f32();
        Position3D::new(
            self.position.x + self.velocity.x * delta_time,
            self.position.y + self.velocity.y * delta_time,
            self.position.z + self.velocity.z * delta_time,
        )
    }

    /// Calculate left ear position
    pub fn left_ear_position(&self) -> Position3D {
        let (yaw, _pitch, _roll) = self.orientation;
        let offset_x = -self.interaural_distance / 2.0 * yaw.cos();
        let offset_z = -self.interaural_distance / 2.0 * yaw.sin();

        Position3D::new(
            self.position.x + offset_x,
            self.position.y,
            self.position.z + offset_z,
        )
    }

    /// Calculate right ear position
    pub fn right_ear_position(&self) -> Position3D {
        let (yaw, _pitch, _roll) = self.orientation;
        let offset_x = self.interaural_distance / 2.0 * yaw.cos();
        let offset_z = self.interaural_distance / 2.0 * yaw.sin();

        Position3D::new(
            self.position.x + offset_x,
            self.position.y,
            self.position.z + offset_z,
        )
    }
}

impl Default for Listener {
    fn default() -> Self {
        Self::new()
    }
}

impl SoundSource {
    /// Create new point source
    pub fn new_point(id: String, position: Position3D) -> Self {
        Self {
            id,
            position,
            velocity: Position3D::default(),
            orientation: None,
            source_type: SourceType::Point,
            attenuation: AttenuationParams::default(),
            directivity: None,
            movement_history: Vec::new(),
            is_active: true,
            last_update: None,
        }
    }

    /// Create new directional source
    pub fn new_directional(
        id: String,
        position: Position3D,
        orientation: (f32, f32, f32),
        directivity: DirectivityPattern,
    ) -> Self {
        Self {
            id,
            position,
            velocity: Position3D::default(),
            orientation: Some(orientation),
            source_type: SourceType::Directional,
            attenuation: AttenuationParams::default(),
            directivity: Some(directivity),
            movement_history: Vec::new(),
            is_active: true,
            last_update: None,
        }
    }

    /// Get current position
    pub fn position(&self) -> Position3D {
        self.position
    }

    /// Set position and update velocity
    pub fn set_position(&mut self, position: Position3D) {
        let now = Instant::now();

        // Calculate velocity if we have a previous update
        if let Some(last_time) = self.last_update {
            let time_delta = now.duration_since(last_time).as_secs_f32();
            if time_delta > 0.0 {
                self.velocity = Position3D::new(
                    (position.x - self.position.x) / time_delta,
                    (position.y - self.position.y) / time_delta,
                    (position.z - self.position.z) / time_delta,
                );
            }
        }

        // Add to movement history
        self.movement_history.push(PositionSnapshot {
            position: self.position,
            timestamp: now.elapsed().as_secs_f64(),
            velocity: self.velocity,
        });

        // Limit history size
        if self.movement_history.len() > 100 {
            self.movement_history.remove(0);
        }

        self.position = position;
        self.last_update = Some(now);
    }

    /// Get current velocity
    pub fn velocity(&self) -> Position3D {
        self.velocity
    }

    /// Get orientation
    pub fn orientation(&self) -> Option<(f32, f32, f32)> {
        self.orientation
    }

    /// Set orientation
    pub fn set_orientation(&mut self, orientation: (f32, f32, f32)) {
        self.orientation = Some(orientation);
    }

    /// Get source type
    pub fn source_type(&self) -> SourceType {
        self.source_type
    }

    /// Get attenuation parameters
    pub fn attenuation(&self) -> &AttenuationParams {
        &self.attenuation
    }

    /// Set attenuation parameters
    pub fn set_attenuation(&mut self, attenuation: AttenuationParams) {
        self.attenuation = attenuation;
    }

    /// Get directivity pattern
    pub fn directivity(&self) -> Option<&DirectivityPattern> {
        self.directivity.as_ref()
    }

    /// Set directivity pattern
    pub fn set_directivity(&mut self, directivity: DirectivityPattern) {
        self.directivity = Some(directivity);
    }

    /// Check if source is active
    pub fn is_active(&self) -> bool {
        self.is_active
    }

    /// Set active state
    pub fn set_active(&mut self, active: bool) {
        self.is_active = active;
    }

    /// Calculate gain based on direction for directional sources
    pub fn calculate_directional_gain(&self, listener_position: Position3D) -> f32 {
        if let (Some(orientation), Some(directivity)) = (&self.orientation, &self.directivity) {
            // Calculate direction from source to listener
            let direction = Position3D::new(
                listener_position.x - self.position.x,
                listener_position.y - self.position.y,
                listener_position.z - self.position.z,
            );

            // Normalize direction vector
            let distance = self.position.distance_to(&listener_position);
            if distance == 0.0 {
                return 1.0;
            }

            let normalized_dir = Position3D::new(
                direction.x / distance,
                direction.y / distance,
                direction.z / distance,
            );

            // Calculate angle between source orientation and listener direction
            let (yaw, _pitch, _roll) = *orientation;
            let source_forward = Position3D::new(yaw.cos(), 0.0, yaw.sin());

            // Dot product for angle calculation
            let dot_product =
                source_forward.x * normalized_dir.x + source_forward.z * normalized_dir.z;
            let angle = dot_product.acos();

            // Interpolate gain based on angle
            let angle_degrees = angle.to_degrees();
            if angle_degrees <= 45.0 {
                directivity.front_gain
            } else if angle_degrees <= 135.0 {
                directivity.side_gain
            } else {
                directivity.back_gain
            }
        } else {
            1.0 // Omnidirectional
        }
    }

    /// Predict future position based on current velocity
    pub fn predict_position(&self, time_ahead: Duration) -> Position3D {
        let delta_time = time_ahead.as_secs_f32();
        Position3D::new(
            self.position.x + self.velocity.x * delta_time,
            self.position.y + self.velocity.y * delta_time,
            self.position.z + self.velocity.z * delta_time,
        )
    }
}

impl Default for AttenuationParams {
    fn default() -> Self {
        Self {
            reference_distance: 1.0,
            max_distance: 100.0,
            rolloff_factor: 1.0,
            model: AttenuationModel::Inverse,
        }
    }
}

impl DirectivityPattern {
    /// Create omnidirectional pattern
    pub fn omnidirectional() -> Self {
        Self {
            front_gain: 1.0,
            back_gain: 1.0,
            side_gain: 1.0,
            directivity_index: 0.0,
            frequency_response: Vec::new(),
        }
    }

    /// Create cardioid pattern
    pub fn cardioid() -> Self {
        Self {
            front_gain: 1.0,
            back_gain: 0.0,
            side_gain: 0.5,
            directivity_index: 3.0,
            frequency_response: Vec::new(),
        }
    }

    /// Create hypercardioid pattern
    pub fn hypercardioid() -> Self {
        Self {
            front_gain: 1.0,
            back_gain: 0.25,
            side_gain: 0.375,
            directivity_index: 6.0,
            frequency_response: Vec::new(),
        }
    }
}

impl Default for ComfortSettings {
    fn default() -> Self {
        Self {
            motion_sickness_reduction: 0.3,
            snap_turn: false,
            snap_turn_degrees: 30.0,
            movement_vignetting: false,
            ground_reference: true,
            speed_multiplier: 1.0,
        }
    }
}

impl Default for MovementConstraints {
    fn default() -> Self {
        Self {
            boundary: None,
            max_speed: 10.0,        // 10 m/s max speed
            max_acceleration: 20.0, // 20 m/s² max acceleration
            ground_height: Some(0.0),
            ceiling_height: None,
        }
    }
}
