//! Position tracking and listener/source management

pub mod advanced_prediction;

use crate::types::Position3D;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
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

/// Occlusion calculation method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OcclusionMethod {
    /// Simple line-of-sight check
    LineOfSight,
    /// Ray casting with multiple rays
    RayCasting,
    /// Fresnel zone checking
    FresnelZone,
    /// Wave diffraction modeling
    Diffraction,
}

/// Material properties for occlusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcclusionMaterial {
    /// Material name
    pub name: String,
    /// Transmission coefficient (0.0 = full block, 1.0 = no block)
    pub transmission: f32,
    /// High-frequency absorption
    pub high_freq_absorption: f32,
    /// Low-frequency absorption
    pub low_freq_absorption: f32,
    /// Scattering coefficient
    pub scattering: f32,
}

/// Occlusion result
#[derive(Debug, Clone)]
pub struct OcclusionResult {
    /// Is source occluded
    pub is_occluded: bool,
    /// Transmission factor (0.0 = fully blocked, 1.0 = no blocking)
    pub transmission_factor: f32,
    /// High-frequency attenuation
    pub high_freq_attenuation: f32,
    /// Low-frequency attenuation
    pub low_freq_attenuation: f32,
    /// Diffraction paths (if any)
    pub diffraction_paths: Vec<DiffractionPath>,
}

/// Sound diffraction path around obstacle
#[derive(Debug, Clone)]
pub struct DiffractionPath {
    /// Path around obstacle
    pub path: Vec<Position3D>,
    /// Path length
    pub length: f32,
    /// Attenuation factor
    pub attenuation: f32,
    /// Delay in samples
    pub delay_samples: usize,
}

/// Advanced head tracking with prediction and smoothing
#[derive(Debug, Clone)]
pub struct HeadTracker {
    /// Position history for prediction
    position_history: VecDeque<PositionSnapshot>,
    /// Orientation history for prediction
    orientation_history: VecDeque<OrientationSnapshot>,
    /// Maximum history size
    max_history_size: usize,
    /// Prediction lookahead time
    prediction_time: Duration,
    /// Velocity smoothing factor (0.0 = no smoothing, 1.0 = maximum smoothing)
    velocity_smoothing: f32,
    /// Orientation smoothing factor
    orientation_smoothing: f32,
    /// Motion prediction enabled
    enable_prediction: bool,
    /// Latency compensation in seconds
    latency_compensation: f32,
}

/// Dynamic source manager for spatial audio
pub struct SpatialSourceManager {
    /// Active sound sources
    sources: std::collections::HashMap<String, SoundSource>,
    /// Spatial awareness grid for optimization
    spatial_grid: SpatialGrid,
    /// Occlusion detector
    occlusion_detector: OcclusionDetector,
    /// Maximum number of concurrent sources
    max_sources: usize,
    /// Distance-based culling threshold
    culling_distance: f32,
    /// Update frequency for spatial calculations
    update_frequency: f32,
}

/// Spatial grid for efficient proximity queries
pub struct SpatialGrid {
    /// Grid cell size in meters
    cell_size: f32,
    /// Grid dimensions
    grid_size: (usize, usize, usize),
    /// Grid cells containing source IDs
    cells: Vec<Vec<Vec<Vec<String>>>>,
    /// Grid bounds
    bounds: (Position3D, Position3D),
}

/// Occlusion and obstruction detector
pub struct OcclusionDetector {
    /// Occlusion geometry (simple box obstacles for now)
    obstacles: Vec<Box3D>,
    /// Occlusion calculation method
    method: OcclusionMethod,
    /// Material properties for obstacles
    materials: std::collections::HashMap<String, OcclusionMaterial>,
}

/// Movement detection and prediction
pub struct MovementTracker {
    /// Position history size
    history_size: usize,
    /// Prediction lookahead time
    prediction_time: Duration,
    /// Velocity smoothing factor
    smoothing_factor: f32,
}

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

impl MovementTracker {
    /// Create new movement tracker
    pub fn new() -> Self {
        Self {
            history_size: 10,
            prediction_time: Duration::from_millis(50),
            smoothing_factor: 0.1,
        }
    }

    /// Track movement for a listener
    pub fn track_listener(&self, _listener: &mut Listener) {
        // Movement tracking is handled in set_position
        // This method could be used for additional processing
    }

    /// Track movement for a sound source
    pub fn track_source(&self, _source: &mut SoundSource) {
        // Movement tracking is handled in set_position
        // This method could be used for additional processing
    }

    /// Predict collision or intersection
    pub fn predict_intersection(
        &self,
        source: &SoundSource,
        listener: &Listener,
        time_ahead: Duration,
    ) -> Option<Position3D> {
        let source_future = source.predict_position(time_ahead);
        let listener_future = listener.predict_position(time_ahead);

        let distance = source_future.distance_to(&listener_future);

        // If they'll be close, return the midpoint
        if distance < 2.0 {
            Some(Position3D::new(
                (source_future.x + listener_future.x) / 2.0,
                (source_future.y + listener_future.y) / 2.0,
                (source_future.z + listener_future.z) / 2.0,
            ))
        } else {
            None
        }
    }
}

impl Default for MovementTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl HeadTracker {
    /// Create new head tracker
    pub fn new() -> Self {
        Self {
            position_history: VecDeque::new(),
            orientation_history: VecDeque::new(),
            max_history_size: 50,
            prediction_time: Duration::from_millis(20),
            velocity_smoothing: 0.3,
            orientation_smoothing: 0.4,
            enable_prediction: true,
            latency_compensation: 0.015, // 15ms typical VR latency
        }
    }

    /// Update head position with smoothing and prediction
    pub fn update_position(&mut self, position: Position3D, timestamp: Instant) {
        let timestamp_f64 = timestamp.elapsed().as_secs_f64();

        // Add to history
        self.position_history.push_back(PositionSnapshot {
            position,
            timestamp: timestamp_f64,
            velocity: self.calculate_velocity(&position, timestamp_f64),
        });

        // Limit history size
        while self.position_history.len() > self.max_history_size {
            self.position_history.pop_front();
        }
    }

    /// Update head position with explicit timestamp (for testing)
    pub fn update_position_with_time(&mut self, position: Position3D, timestamp_secs: f64) {
        // Add to history
        self.position_history.push_back(PositionSnapshot {
            position,
            timestamp: timestamp_secs,
            velocity: self.calculate_velocity_for_time(&position, timestamp_secs),
        });

        // Limit history size
        while self.position_history.len() > self.max_history_size {
            self.position_history.pop_front();
        }
    }

    /// Update head orientation with smoothing
    pub fn update_orientation(&mut self, orientation: (f32, f32, f32), timestamp: Instant) {
        let timestamp_f64 = timestamp.elapsed().as_secs_f64();

        // Calculate angular velocity
        let angular_velocity = self.calculate_angular_velocity(&orientation, timestamp_f64);

        self.orientation_history.push_back(OrientationSnapshot {
            orientation,
            timestamp: timestamp_f64,
            angular_velocity,
        });

        // Limit history size
        while self.orientation_history.len() > self.max_history_size {
            self.orientation_history.pop_front();
        }
    }

    /// Update head orientation with explicit timestamp (for testing)
    pub fn update_orientation_with_time(
        &mut self,
        orientation: (f32, f32, f32),
        timestamp_secs: f64,
    ) {
        // Calculate angular velocity
        let angular_velocity =
            self.calculate_angular_velocity_for_time(&orientation, timestamp_secs);

        self.orientation_history.push_back(OrientationSnapshot {
            orientation,
            timestamp: timestamp_secs,
            angular_velocity,
        });

        // Limit history size
        while self.orientation_history.len() > self.max_history_size {
            self.orientation_history.pop_front();
        }
    }

    /// Predict future head position
    pub fn predict_position(&self, time_ahead: Duration) -> Option<Position3D> {
        if !self.enable_prediction || self.position_history.len() < 2 {
            return self.position_history.back().map(|s| s.position);
        }

        let latest = self.position_history.back()?;
        let prediction_time = time_ahead.as_secs_f32() + self.latency_compensation;

        // Linear prediction with velocity
        Some(Position3D::new(
            latest.position.x + latest.velocity.x * prediction_time,
            latest.position.y + latest.velocity.y * prediction_time,
            latest.position.z + latest.velocity.z * prediction_time,
        ))
    }

    /// Predict future head orientation
    pub fn predict_orientation(&self, time_ahead: Duration) -> Option<(f32, f32, f32)> {
        if !self.enable_prediction || self.orientation_history.len() < 2 {
            return self.orientation_history.back().map(|s| s.orientation);
        }

        let latest = self.orientation_history.back()?;
        let prediction_time = time_ahead.as_secs_f32() + self.latency_compensation;

        // Predict orientation with angular velocity
        let predicted_yaw = latest.orientation.0 + latest.angular_velocity.0 * prediction_time;
        let predicted_pitch = latest.orientation.1 + latest.angular_velocity.1 * prediction_time;
        let predicted_roll = latest.orientation.2 + latest.angular_velocity.2 * prediction_time;

        Some((predicted_yaw, predicted_pitch, predicted_roll))
    }

    /// Calculate smoothed velocity
    fn calculate_velocity(&self, position: &Position3D, timestamp: f64) -> Position3D {
        if self.position_history.len() < 2 {
            return Position3D::default();
        }

        let prev = &self.position_history[self.position_history.len() - 1];
        let dt = timestamp - prev.timestamp;

        if dt <= 0.0 {
            return prev.velocity;
        }

        // Calculate raw velocity
        let raw_velocity = Position3D::new(
            (position.x - prev.position.x) / dt as f32,
            (position.y - prev.position.y) / dt as f32,
            (position.z - prev.position.z) / dt as f32,
        );

        // Apply smoothing
        Position3D::new(
            prev.velocity.x * self.velocity_smoothing
                + raw_velocity.x * (1.0 - self.velocity_smoothing),
            prev.velocity.y * self.velocity_smoothing
                + raw_velocity.y * (1.0 - self.velocity_smoothing),
            prev.velocity.z * self.velocity_smoothing
                + raw_velocity.z * (1.0 - self.velocity_smoothing),
        )
    }

    /// Calculate smoothed velocity with explicit timestamp
    fn calculate_velocity_for_time(&self, position: &Position3D, timestamp: f64) -> Position3D {
        if self.position_history.is_empty() {
            return Position3D::default();
        }

        let prev = &self.position_history[self.position_history.len() - 1];
        let dt = timestamp - prev.timestamp;

        if dt <= 0.0 {
            return prev.velocity;
        }

        // Calculate raw velocity
        let raw_velocity = Position3D::new(
            (position.x - prev.position.x) / dt as f32,
            (position.y - prev.position.y) / dt as f32,
            (position.z - prev.position.z) / dt as f32,
        );

        // Apply smoothing if previous velocity exists
        if self.position_history.len() > 1 {
            Position3D::new(
                prev.velocity.x * self.velocity_smoothing
                    + raw_velocity.x * (1.0 - self.velocity_smoothing),
                prev.velocity.y * self.velocity_smoothing
                    + raw_velocity.y * (1.0 - self.velocity_smoothing),
                prev.velocity.z * self.velocity_smoothing
                    + raw_velocity.z * (1.0 - self.velocity_smoothing),
            )
        } else {
            raw_velocity
        }
    }

    /// Calculate angular velocity
    fn calculate_angular_velocity(
        &self,
        orientation: &(f32, f32, f32),
        timestamp: f64,
    ) -> (f32, f32, f32) {
        if self.orientation_history.len() < 2 {
            return (0.0, 0.0, 0.0);
        }

        let prev = &self.orientation_history[self.orientation_history.len() - 1];
        let dt = timestamp - prev.timestamp;

        if dt <= 0.0 {
            return prev.angular_velocity;
        }

        // Calculate raw angular velocity with wrap-around handling
        let raw_angular_velocity = (
            self.angle_difference(orientation.0, prev.orientation.0) / dt as f32,
            self.angle_difference(orientation.1, prev.orientation.1) / dt as f32,
            self.angle_difference(orientation.2, prev.orientation.2) / dt as f32,
        );

        // Apply smoothing
        (
            prev.angular_velocity.0 * self.orientation_smoothing
                + raw_angular_velocity.0 * (1.0 - self.orientation_smoothing),
            prev.angular_velocity.1 * self.orientation_smoothing
                + raw_angular_velocity.1 * (1.0 - self.orientation_smoothing),
            prev.angular_velocity.2 * self.orientation_smoothing
                + raw_angular_velocity.2 * (1.0 - self.orientation_smoothing),
        )
    }

    /// Calculate angular velocity with explicit timestamp
    fn calculate_angular_velocity_for_time(
        &self,
        orientation: &(f32, f32, f32),
        timestamp: f64,
    ) -> (f32, f32, f32) {
        if self.orientation_history.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let prev = &self.orientation_history[self.orientation_history.len() - 1];
        let dt = timestamp - prev.timestamp;

        if dt <= 0.0 {
            return prev.angular_velocity;
        }

        // Calculate raw angular velocity with wrap-around handling
        let raw_angular_velocity = (
            self.angle_difference(orientation.0, prev.orientation.0) / dt as f32,
            self.angle_difference(orientation.1, prev.orientation.1) / dt as f32,
            self.angle_difference(orientation.2, prev.orientation.2) / dt as f32,
        );

        // Apply smoothing if previous velocity exists
        if self.orientation_history.len() > 1 {
            (
                prev.angular_velocity.0 * self.orientation_smoothing
                    + raw_angular_velocity.0 * (1.0 - self.orientation_smoothing),
                prev.angular_velocity.1 * self.orientation_smoothing
                    + raw_angular_velocity.1 * (1.0 - self.orientation_smoothing),
                prev.angular_velocity.2 * self.orientation_smoothing
                    + raw_angular_velocity.2 * (1.0 - self.orientation_smoothing),
            )
        } else {
            raw_angular_velocity
        }
    }

    /// Calculate angle difference with wrap-around
    fn angle_difference(&self, angle1: f32, angle2: f32) -> f32 {
        let diff = angle1 - angle2;
        if diff > std::f32::consts::PI {
            diff - 2.0 * std::f32::consts::PI
        } else if diff < -std::f32::consts::PI {
            diff + 2.0 * std::f32::consts::PI
        } else {
            diff
        }
    }
    /// Configure tracking parameters
    pub fn configure(
        &mut self,
        max_history: usize,
        prediction_ms: u64,
        velocity_smoothing: f32,
        orientation_smoothing: f32,
    ) {
        self.max_history_size = max_history;
        self.prediction_time = Duration::from_millis(prediction_ms);
        self.velocity_smoothing = velocity_smoothing.clamp(0.0, 1.0);
        self.orientation_smoothing = orientation_smoothing.clamp(0.0, 1.0);
    }

    /// Enable/disable prediction
    pub fn set_prediction_enabled(&mut self, enabled: bool) {
        self.enable_prediction = enabled;
    }

    /// Set latency compensation
    pub fn set_latency_compensation(&mut self, latency_ms: f32) {
        self.latency_compensation = latency_ms / 1000.0;
    }

    /// Get current position (latest)
    pub fn current_position(&self) -> Option<Position3D> {
        self.position_history.back().map(|s| s.position)
    }

    /// Get current orientation (latest)
    pub fn current_orientation(&self) -> Option<(f32, f32, f32)> {
        self.orientation_history.back().map(|s| s.orientation)
    }

    /// Get smoothed velocity
    pub fn current_velocity(&self) -> Option<Position3D> {
        self.position_history.back().map(|s| s.velocity)
    }

    /// Get angular velocity
    pub fn current_angular_velocity(&self) -> Option<(f32, f32, f32)> {
        self.orientation_history.back().map(|s| s.angular_velocity)
    }

    /// Clear history
    pub fn reset(&mut self) {
        self.position_history.clear();
        self.orientation_history.clear();
    }

    /// Get prediction quality score (0.0 = poor, 1.0 = excellent)
    pub fn prediction_quality(&self) -> f32 {
        if self.position_history.len() < 3 {
            return 0.0;
        }

        // Calculate velocity consistency
        let mut velocity_consistency = 0.0;
        let mut count = 0;

        for i in 0..self.position_history.len().saturating_sub(1) {
            let v1 = self.position_history[i].velocity;
            let v2 = self.position_history[i + 1].velocity;

            let vel_diff =
                ((v1.x - v2.x).powi(2) + (v1.y - v2.y).powi(2) + (v1.z - v2.z).powi(2)).sqrt();
            velocity_consistency += (-vel_diff * 0.1).exp();
            count += 1;
        }

        if count > 0 {
            velocity_consistency / count as f32
        } else {
            0.0
        }
    }
}

/// Real-time listener movement and navigation system
#[derive(Debug, Clone)]
pub struct ListenerMovementSystem {
    /// Current listener reference
    listener: Listener,
    /// Head tracking system
    head_tracker: HeadTracker,
    /// Movement prediction enabled
    enable_movement_prediction: bool,
    /// Navigation mode
    navigation_mode: NavigationMode,
    /// Comfort settings for VR
    comfort_settings: ComfortSettings,
    /// Movement constraints
    movement_constraints: MovementConstraints,
    /// Performance metrics
    movement_metrics: MovementMetrics,
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

/// VR/AR platform integration support
#[derive(Debug, Clone)]
pub struct PlatformIntegration {
    /// Platform type
    platform_type: PlatformType,
    /// Platform-specific tracking data
    platform_data: PlatformData,
    /// Calibration data
    calibration: CalibrationData,
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

impl ListenerMovementSystem {
    /// Create new listener movement system
    pub fn new() -> Self {
        Self {
            listener: Listener::new(),
            head_tracker: HeadTracker::new(),
            enable_movement_prediction: true,
            navigation_mode: NavigationMode::FreeFlight,
            comfort_settings: ComfortSettings::default(),
            movement_constraints: MovementConstraints::default(),
            movement_metrics: MovementMetrics::default(),
        }
    }

    /// Create system with specific navigation mode
    pub fn with_navigation_mode(mode: NavigationMode) -> Self {
        let mut system = Self::new();
        system.navigation_mode = mode;

        // Adjust defaults based on mode
        match mode {
            NavigationMode::Seated => {
                system.movement_constraints.max_speed = 0.0;
                system.enable_movement_prediction = false;
            }
            NavigationMode::Walking => {
                system.movement_constraints.max_speed = 5.0; // Walking speed
                system.comfort_settings.ground_reference = true;
            }
            NavigationMode::Vehicle => {
                system.movement_constraints.max_speed = 50.0; // Vehicle speed
                system.comfort_settings.motion_sickness_reduction = 0.7;
            }
            _ => {}
        }

        system
    }

    /// Update listener position with platform data
    pub fn update_position_from_platform(
        &mut self,
        position: Position3D,
        platform_data: Option<PlatformData>,
    ) -> crate::Result<()> {
        let now = Instant::now();

        // Apply movement constraints
        let constrained_position = self.apply_movement_constraints(position)?;

        // Calculate distance before updating position
        let current_pos = self.listener.position();
        let distance = ((constrained_position.x - current_pos.x).powi(2)
            + (constrained_position.y - current_pos.y).powi(2)
            + (constrained_position.z - current_pos.z).powi(2))
        .sqrt();

        // Update head tracker
        self.head_tracker.update_position(constrained_position, now);

        // Update listener
        self.listener.set_position(constrained_position);

        // Update metrics with calculated distance
        self.movement_metrics.total_distance += distance;
        self.movement_metrics.update_count += 1;

        // Process platform-specific data
        if let Some(data) = platform_data {
            self.process_platform_data(data)?;
        }

        Ok(())
    }

    /// Update orientation with platform data
    pub fn update_orientation_from_platform(
        &mut self,
        orientation: (f32, f32, f32),
        platform_data: Option<PlatformData>,
    ) -> crate::Result<()> {
        let now = Instant::now();

        // Apply comfort settings (snap turn, etc.)
        let adjusted_orientation = self.apply_comfort_adjustments(orientation);

        // Update head tracker
        self.head_tracker
            .update_orientation(adjusted_orientation, now);

        // Update listener
        self.listener.set_orientation(adjusted_orientation);

        // Process platform-specific data
        if let Some(data) = platform_data {
            self.process_platform_data(data)?;
        }

        Ok(())
    }

    /// Get predicted listener position for latency compensation
    pub fn predict_position(&self, lookahead: Duration) -> Option<Position3D> {
        if !self.enable_movement_prediction {
            return Some(self.listener.position());
        }

        self.head_tracker.predict_position(lookahead)
    }

    /// Get predicted listener orientation
    pub fn predict_orientation(&self, lookahead: Duration) -> Option<(f32, f32, f32)> {
        if !self.enable_movement_prediction {
            return Some(self.listener.orientation());
        }

        self.head_tracker.predict_orientation(lookahead)
    }

    /// Get listener reference
    pub fn listener(&self) -> &Listener {
        &self.listener
    }

    /// Get mutable listener reference
    pub fn listener_mut(&mut self) -> &mut Listener {
        &mut self.listener
    }

    /// Get head tracker reference
    pub fn head_tracker(&self) -> &HeadTracker {
        &self.head_tracker
    }

    /// Configure comfort settings
    pub fn set_comfort_settings(&mut self, settings: ComfortSettings) {
        self.comfort_settings = settings;
    }

    /// Configure movement constraints
    pub fn set_movement_constraints(&mut self, constraints: MovementConstraints) {
        self.movement_constraints = constraints;
    }

    /// Get movement metrics
    pub fn movement_metrics(&self) -> &MovementMetrics {
        &self.movement_metrics
    }

    /// Reset movement metrics
    pub fn reset_metrics(&mut self) {
        self.movement_metrics = MovementMetrics::default();
    }

    /// Apply movement constraints
    fn apply_movement_constraints(&self, mut position: Position3D) -> crate::Result<Position3D> {
        // Apply boundary constraints
        if let Some(boundary) = &self.movement_constraints.boundary {
            position.x = position.x.clamp(boundary.min.x, boundary.max.x);
            position.y = position.y.clamp(boundary.min.y, boundary.max.y);
            position.z = position.z.clamp(boundary.min.z, boundary.max.z);
        }

        // Apply ground height constraint
        if let Some(ground_height) = self.movement_constraints.ground_height {
            position.y = position.y.max(ground_height);
        }

        // Apply ceiling height constraint
        if let Some(ceiling_height) = self.movement_constraints.ceiling_height {
            position.y = position.y.min(ceiling_height);
        }

        // Check speed constraints (would need velocity calculation)
        // This would be implemented with proper velocity tracking

        Ok(position)
    }

    /// Apply comfort adjustments to orientation
    fn apply_comfort_adjustments(&self, orientation: (f32, f32, f32)) -> (f32, f32, f32) {
        let mut adjusted = orientation;

        // Apply snap turning
        if self.comfort_settings.snap_turn {
            let snap_radians = self.comfort_settings.snap_turn_degrees.to_radians();
            adjusted.0 = (adjusted.0 / snap_radians).round() * snap_radians;
        }

        adjusted
    }

    /// Process platform-specific tracking data
    fn process_platform_data(&mut self, _data: PlatformData) -> crate::Result<()> {
        // This would be implemented for specific platform integrations
        // For now, just validate the data
        Ok(())
    }
}

impl PlatformIntegration {
    /// Create platform integration for specific platform
    pub fn new(platform: PlatformType) -> Self {
        Self {
            platform_type: platform,
            platform_data: PlatformData {
                device_id: String::new(),
                pose_data: Vec::new(),
                tracking_confidence: 0.0,
                platform_timestamp: 0,
                properties: std::collections::HashMap::new(),
            },
            calibration: CalibrationData {
                head_circumference: None,
                ipd: None,
                height_offset: 0.0,
                forward_offset: 0.0,
                custom_hrtf_profile: None,
            },
        }
    }

    /// Update platform tracking data
    pub fn update_tracking_data(&mut self, data: PlatformData) {
        self.platform_data = data;
    }

    /// Get tracking confidence
    pub fn tracking_confidence(&self) -> f32 {
        self.platform_data.tracking_confidence
    }

    /// Configure calibration
    pub fn set_calibration(&mut self, calibration: CalibrationData) {
        self.calibration = calibration;
    }

    /// Get platform type
    pub fn platform_type(&self) -> PlatformType {
        self.platform_type
    }
}

impl Default for ListenerMovementSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PlatformIntegration {
    fn default() -> Self {
        Self::new(PlatformType::Generic)
    }
}

impl SpatialSourceManager {
    /// Create new spatial source manager
    pub fn new(bounds: (Position3D, Position3D), cell_size: f32) -> Self {
        Self {
            sources: std::collections::HashMap::new(),
            spatial_grid: SpatialGrid::new(bounds, cell_size),
            occlusion_detector: OcclusionDetector::new(),
            max_sources: 64,
            culling_distance: 100.0,
            update_frequency: 60.0,
        }
    }

    /// Add source to manager
    pub fn add_source(&mut self, source: SoundSource) -> crate::Result<()> {
        if self.sources.len() >= self.max_sources {
            return Err(crate::Error::LegacyPosition(
                "Maximum sources exceeded".to_string(),
            ));
        }

        let source_id = source.id.clone();
        let position = source.position();

        // Add to spatial grid
        self.spatial_grid.add_source(&source_id, position);

        // Add to sources
        self.sources.insert(source_id, source);

        Ok(())
    }

    /// Remove source from manager
    pub fn remove_source(&mut self, source_id: &str) -> Option<SoundSource> {
        if let Some(source) = self.sources.remove(source_id) {
            self.spatial_grid.remove_source(source_id);
            Some(source)
        } else {
            None
        }
    }

    /// Update source position
    pub fn update_source_position(
        &mut self,
        source_id: &str,
        position: Position3D,
    ) -> crate::Result<()> {
        if let Some(source) = self.sources.get_mut(source_id) {
            let old_position = source.position();
            source.set_position(position);

            // Update spatial grid
            self.spatial_grid
                .move_source(source_id, old_position, position);

            Ok(())
        } else {
            Err(crate::Error::LegacyPosition(format!(
                "Source not found: {source_id}"
            )))
        }
    }

    /// Get nearby sources for listener
    pub fn get_nearby_sources(
        &self,
        listener_position: Position3D,
        radius: f32,
    ) -> Vec<&SoundSource> {
        let nearby_ids = self.spatial_grid.query_sphere(listener_position, radius);
        nearby_ids
            .iter()
            .filter_map(|id| self.sources.get(id))
            .collect()
    }

    /// Check occlusion for source
    pub fn check_occlusion(
        &self,
        source_position: Position3D,
        listener_position: Position3D,
    ) -> OcclusionResult {
        self.occlusion_detector
            .check_occlusion(source_position, listener_position)
    }

    /// Cull distant sources
    pub fn cull_distant_sources(&mut self, listener_position: Position3D) {
        let culling_distance_sq = self.culling_distance * self.culling_distance;
        let distant_sources: Vec<String> = self
            .sources
            .iter()
            .filter(|(_, source)| {
                let distance_sq = listener_position.distance_to(&source.position()).powi(2);
                distance_sq > culling_distance_sq
            })
            .map(|(id, _)| id.clone())
            .collect();

        for source_id in distant_sources {
            self.remove_source(&source_id);
        }
    }

    /// Get all active sources
    pub fn get_active_sources(&self) -> Vec<&SoundSource> {
        self.sources.values().filter(|s| s.is_active()).collect()
    }
}

impl SpatialGrid {
    /// Create new spatial grid
    pub fn new(bounds: (Position3D, Position3D), cell_size: f32) -> Self {
        let (min_bounds, max_bounds) = bounds;
        let width = ((max_bounds.x - min_bounds.x) / cell_size).ceil() as usize;
        let height = ((max_bounds.y - min_bounds.y) / cell_size).ceil() as usize;
        let depth = ((max_bounds.z - min_bounds.z) / cell_size).ceil() as usize;

        let mut cells = Vec::with_capacity(width);
        for _ in 0..width {
            let mut column = Vec::with_capacity(height);
            for _ in 0..height {
                let mut layer = Vec::with_capacity(depth);
                for _ in 0..depth {
                    layer.push(Vec::new());
                }
                column.push(layer);
            }
            cells.push(column);
        }

        Self {
            cell_size,
            grid_size: (width, height, depth),
            cells,
            bounds,
        }
    }

    /// Add source to grid
    pub fn add_source(&mut self, source_id: &str, position: Position3D) {
        if let Some((x, y, z)) = self.position_to_cell(position) {
            self.cells[x][y][z].push(source_id.to_string());
        }
    }

    /// Remove source from grid
    pub fn remove_source(&mut self, source_id: &str) {
        for x in 0..self.grid_size.0 {
            for y in 0..self.grid_size.1 {
                for z in 0..self.grid_size.2 {
                    self.cells[x][y][z].retain(|id| id != source_id);
                }
            }
        }
    }

    /// Move source in grid
    pub fn move_source(
        &mut self,
        source_id: &str,
        old_position: Position3D,
        new_position: Position3D,
    ) {
        let old_cell = self.position_to_cell(old_position);
        let new_cell = self.position_to_cell(new_position);

        if old_cell != new_cell {
            // Remove from old cell
            if let Some((x, y, z)) = old_cell {
                self.cells[x][y][z].retain(|id| id != source_id);
            }

            // Add to new cell
            if let Some((x, y, z)) = new_cell {
                self.cells[x][y][z].push(source_id.to_string());
            }
        }
    }

    /// Query sources within sphere
    pub fn query_sphere(&self, center: Position3D, radius: f32) -> Vec<String> {
        let mut results = Vec::new();
        let radius_cells = (radius / self.cell_size).ceil() as i32;

        if let Some((cx, cy, cz)) = self.position_to_cell(center) {
            let cx = cx as i32;
            let cy = cy as i32;
            let cz = cz as i32;

            for dx in -radius_cells..=radius_cells {
                for dy in -radius_cells..=radius_cells {
                    for dz in -radius_cells..=radius_cells {
                        let x = cx + dx;
                        let y = cy + dy;
                        let z = cz + dz;

                        if x >= 0
                            && y >= 0
                            && z >= 0
                            && (x as usize) < self.grid_size.0
                            && (y as usize) < self.grid_size.1
                            && (z as usize) < self.grid_size.2
                        {
                            results.extend(
                                self.cells[x as usize][y as usize][z as usize]
                                    .iter()
                                    .cloned(),
                            );
                        }
                    }
                }
            }
        }

        results
    }

    /// Convert position to grid cell coordinates
    fn position_to_cell(&self, position: Position3D) -> Option<(usize, usize, usize)> {
        let (min_bounds, _) = self.bounds;

        let x = ((position.x - min_bounds.x) / self.cell_size) as usize;
        let y = ((position.y - min_bounds.y) / self.cell_size) as usize;
        let z = ((position.z - min_bounds.z) / self.cell_size) as usize;

        if x < self.grid_size.0 && y < self.grid_size.1 && z < self.grid_size.2 {
            Some((x, y, z))
        } else {
            None
        }
    }
}

impl OcclusionDetector {
    /// Create new occlusion detector
    pub fn new() -> Self {
        Self {
            obstacles: Vec::new(),
            method: OcclusionMethod::LineOfSight,
            materials: std::collections::HashMap::new(),
        }
    }

    /// Add obstacle
    pub fn add_obstacle(&mut self, obstacle: Box3D) {
        self.obstacles.push(obstacle);
    }

    /// Add material
    pub fn add_material(&mut self, material: OcclusionMaterial) {
        self.materials.insert(material.name.clone(), material);
    }

    /// Check occlusion between source and listener
    pub fn check_occlusion(&self, source: Position3D, listener: Position3D) -> OcclusionResult {
        match self.method {
            OcclusionMethod::LineOfSight => self.line_of_sight_check(source, listener),
            OcclusionMethod::RayCasting => self.ray_casting_check(source, listener),
            OcclusionMethod::FresnelZone => self.fresnel_zone_check(source, listener),
            OcclusionMethod::Diffraction => self.diffraction_check(source, listener),
        }
    }

    /// Simple line-of-sight occlusion check
    fn line_of_sight_check(&self, source: Position3D, listener: Position3D) -> OcclusionResult {
        for obstacle in &self.obstacles {
            if self.line_intersects_box(source, listener, obstacle) {
                let material = self
                    .materials
                    .get(&obstacle.material_id)
                    .cloned()
                    .unwrap_or_else(OcclusionMaterial::default);

                return OcclusionResult {
                    is_occluded: true,
                    transmission_factor: material.transmission,
                    high_freq_attenuation: material.high_freq_absorption,
                    low_freq_attenuation: material.low_freq_absorption,
                    diffraction_paths: Vec::new(),
                };
            }
        }

        OcclusionResult {
            is_occluded: false,
            transmission_factor: 1.0,
            high_freq_attenuation: 1.0,
            low_freq_attenuation: 1.0,
            diffraction_paths: Vec::new(),
        }
    }

    /// Ray casting occlusion check with multiple rays
    fn ray_casting_check(&self, source: Position3D, listener: Position3D) -> OcclusionResult {
        // For now, delegate to line-of-sight
        // In a full implementation, this would cast multiple rays
        self.line_of_sight_check(source, listener)
    }

    /// Fresnel zone occlusion check
    fn fresnel_zone_check(&self, source: Position3D, listener: Position3D) -> OcclusionResult {
        // Simplified implementation - delegate to line-of-sight
        // A full implementation would check Fresnel zone clearance
        self.line_of_sight_check(source, listener)
    }

    /// Diffraction-based occlusion check
    fn diffraction_check(&self, source: Position3D, listener: Position3D) -> OcclusionResult {
        let mut result = self.line_of_sight_check(source, listener);

        // If occluded, try to find diffraction paths
        if result.is_occluded {
            result.diffraction_paths = self.find_diffraction_paths(source, listener);

            // If diffraction paths exist, adjust transmission
            if !result.diffraction_paths.is_empty() {
                result.transmission_factor = result.transmission_factor.max(0.1);
                // Some sound gets through
            }
        }

        result
    }

    /// Check if line intersects with 3D box
    fn line_intersects_box(&self, start: Position3D, end: Position3D, box3d: &Box3D) -> bool {
        // Simplified box-line intersection test
        let dir = Position3D::new(end.x - start.x, end.y - start.y, end.z - start.z);
        let length = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt();

        if length == 0.0 {
            return false;
        }

        let dir_norm = Position3D::new(dir.x / length, dir.y / length, dir.z / length);

        // Ray-box intersection using slab method
        let inv_dir = Position3D::new(
            if dir_norm.x != 0.0 {
                1.0 / dir_norm.x
            } else {
                f32::INFINITY
            },
            if dir_norm.y != 0.0 {
                1.0 / dir_norm.y
            } else {
                f32::INFINITY
            },
            if dir_norm.z != 0.0 {
                1.0 / dir_norm.z
            } else {
                f32::INFINITY
            },
        );

        let t1 = (box3d.min.x - start.x) * inv_dir.x;
        let t2 = (box3d.max.x - start.x) * inv_dir.x;
        let t3 = (box3d.min.y - start.y) * inv_dir.y;
        let t4 = (box3d.max.y - start.y) * inv_dir.y;
        let t5 = (box3d.min.z - start.z) * inv_dir.z;
        let t6 = (box3d.max.z - start.z) * inv_dir.z;

        let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
        let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

        tmax >= 0.0 && tmin <= tmax && tmin <= length
    }

    /// Find diffraction paths around obstacles
    fn find_diffraction_paths(
        &self,
        _source: Position3D,
        _listener: Position3D,
    ) -> Vec<DiffractionPath> {
        // Simplified implementation - returns empty for now
        // A full implementation would find paths around obstacle edges
        Vec::new()
    }
}

impl Default for HeadTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for OcclusionDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for OcclusionMaterial {
    fn default() -> Self {
        Self {
            name: "Default".to_string(),
            transmission: 0.1,
            high_freq_absorption: 0.8,
            low_freq_absorption: 0.3,
            scattering: 0.2,
        }
    }
}

/// Doppler effect processor for moving sources and listeners
#[derive(Debug, Clone)]
pub struct DopplerProcessor {
    /// Speed of sound (m/s)
    speed_of_sound: f32,
    /// Sample rate
    sample_rate: f32,
    /// Maximum Doppler shift factor (safety limit)
    max_doppler_factor: f32,
    /// Smoothing factor for Doppler transitions
    smoothing_factor: f32,
}

/// Dynamic source manager for handling moving sources
#[derive(Debug, Clone)]
pub struct DynamicSourceManager {
    /// Dynamic sources
    sources: std::collections::HashMap<String, DynamicSource>,
    /// Doppler processor
    doppler_processor: DopplerProcessor,
    /// Motion predictor
    motion_predictor: MotionPredictor,
}

/// Dynamic source with motion tracking and prediction
#[derive(Debug, Clone)]
pub struct DynamicSource {
    /// Base source
    pub base_source: SoundSource,
    /// Velocity tracking
    pub velocity: Position3D,
    /// Acceleration tracking
    pub acceleration: Position3D,
    /// Motion history for prediction
    pub motion_history: VecDeque<MotionSnapshot>,
    /// Current Doppler factor
    pub doppler_factor: f32,
    /// Smoothed Doppler factor
    pub smoothed_doppler_factor: f32,
    /// Last Doppler update time
    pub last_doppler_update: Option<Instant>,
}

/// Motion snapshot for tracking and prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionSnapshot {
    /// Position at this time
    pub position: Position3D,
    /// Velocity at this time
    pub velocity: Position3D,
    /// Acceleration at this time  
    pub acceleration: Position3D,
    /// Timestamp
    pub timestamp: f64,
}

/// Motion predictor for anticipating source movement
#[derive(Debug, Clone)]
pub struct MotionPredictor {
    /// Prediction time horizon (seconds)
    prediction_horizon: f32,
    /// Minimum samples needed for prediction
    min_samples: usize,
    /// Maximum history size
    max_history_size: usize,
}

impl DopplerProcessor {
    /// Create new Doppler processor
    pub fn new(sample_rate: f32) -> Self {
        Self {
            speed_of_sound: 343.0, // m/s at 20°C
            sample_rate,
            max_doppler_factor: 2.0, // Safety limit to prevent extreme shifts
            smoothing_factor: 0.95,  // Smooth Doppler transitions
        }
    }

    /// Create Doppler processor with custom speed of sound
    pub fn with_speed_of_sound(sample_rate: f32, speed_of_sound: f32) -> Self {
        Self {
            speed_of_sound,
            sample_rate,
            max_doppler_factor: 2.0,
            smoothing_factor: 0.95,
        }
    }

    /// Calculate Doppler factor for source and listener
    pub fn calculate_doppler_factor(
        &self,
        source_position: Position3D,
        source_velocity: Position3D,
        listener_position: Position3D,
        listener_velocity: Position3D,
    ) -> f32 {
        // Vector from source to listener
        let source_to_listener = Position3D::new(
            listener_position.x - source_position.x,
            listener_position.y - source_position.y,
            listener_position.z - source_position.z,
        );

        let distance = source_to_listener.magnitude();
        if distance < 0.001 {
            return 1.0; // Avoid division by zero
        }

        // Unit vector from source to listener
        let direction = source_to_listener.normalized();

        // Radial velocities (positive means approaching for source, receding for listener)
        let source_radial_velocity = source_velocity.dot(&direction); // Positive = towards listener
        let listener_radial_velocity = -listener_velocity.dot(&direction); // Positive = towards source

        // Classic Doppler formula: f' = f * (v + vr) / (v - vs)
        // where vr is listener radial velocity, vs is source radial velocity
        // Positive source velocity means approaching (should increase frequency)
        let numerator = self.speed_of_sound + listener_radial_velocity;
        let denominator = self.speed_of_sound - source_radial_velocity;

        if denominator.abs() < 0.001 {
            return 1.0; // Avoid division by zero
        }

        let doppler_factor = numerator / denominator;

        // Clamp to reasonable limits
        doppler_factor.clamp(1.0 / self.max_doppler_factor, self.max_doppler_factor)
    }

    /// Apply Doppler effect to audio buffer
    pub fn process_doppler_effect(
        &self,
        input: &[f32],
        output: &mut [f32],
        doppler_factor: f32,
    ) -> crate::Result<()> {
        if input.len() != output.len() {
            return Err(crate::Error::LegacyProcessing(
                "Input and output buffers must have the same length".to_string(),
            ));
        }

        if (doppler_factor - 1.0).abs() < 0.001 {
            // No significant Doppler effect, just copy
            output.copy_from_slice(input);
            return Ok(());
        }

        // Simple pitch shifting using linear interpolation
        // In production, would use more sophisticated algorithms like PSOLA or phase vocoder
        let pitch_ratio = doppler_factor;
        let mut read_pos = 0.0;

        for i in 0..output.len() {
            let read_index = read_pos as usize;
            let read_frac = read_pos - read_index as f32;

            if read_index + 1 < input.len() {
                // Linear interpolation
                let sample1 = input[read_index];
                let sample2 = input[read_index + 1];
                output[i] = sample1 + read_frac * (sample2 - sample1);
            } else if read_index < input.len() {
                output[i] = input[read_index];
            } else {
                output[i] = 0.0;
            }

            read_pos += pitch_ratio;
            if read_pos >= input.len() as f32 {
                // Fill remaining samples with zero
                output[i + 1..].fill(0.0);
                break;
            }
        }

        Ok(())
    }

    /// Smooth Doppler factor transitions
    pub fn smooth_doppler_factor(&self, current: f32, target: f32) -> f32 {
        current * self.smoothing_factor + target * (1.0 - self.smoothing_factor)
    }

    /// Set speed of sound (useful for different atmospheric conditions)
    pub fn set_speed_of_sound(&mut self, speed: f32) {
        self.speed_of_sound = speed;
    }

    /// Get current speed of sound
    pub fn speed_of_sound(&self) -> f32 {
        self.speed_of_sound
    }
}

impl DynamicSourceManager {
    /// Create new dynamic source manager
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sources: std::collections::HashMap::new(),
            doppler_processor: DopplerProcessor::new(sample_rate),
            motion_predictor: MotionPredictor::new(),
        }
    }

    /// Add dynamic source
    pub fn add_source(&mut self, source: SoundSource) -> crate::Result<()> {
        let dynamic_source = DynamicSource::new(source);
        self.sources
            .insert(dynamic_source.base_source.id.clone(), dynamic_source);
        Ok(())
    }

    /// Remove dynamic source
    pub fn remove_source(&mut self, source_id: &str) -> Option<DynamicSource> {
        self.sources.remove(source_id)
    }

    /// Update source motion
    pub fn update_source_motion(
        &mut self,
        source_id: &str,
        position: Position3D,
        velocity: Position3D,
        acceleration: Position3D,
    ) -> crate::Result<()> {
        if let Some(source) = self.sources.get_mut(source_id) {
            source.update_motion(position, velocity, acceleration);
            Ok(())
        } else {
            Err(crate::Error::LegacyPosition(format!(
                "Source '{source_id}' not found"
            )))
        }
    }

    /// Process all dynamic sources with Doppler effects
    pub async fn process_dynamic_sources(
        &mut self,
        listener_position: Position3D,
        listener_velocity: Position3D,
    ) -> crate::Result<()> {
        for source in self.sources.values_mut() {
            // Update Doppler factor
            let doppler_factor = self.doppler_processor.calculate_doppler_factor(
                source.base_source.position(),
                source.velocity,
                listener_position,
                listener_velocity,
            );

            // Smooth the Doppler factor
            source.smoothed_doppler_factor = self
                .doppler_processor
                .smooth_doppler_factor(source.smoothed_doppler_factor, doppler_factor);

            source.doppler_factor = doppler_factor;
            source.last_doppler_update = Some(Instant::now());
        }

        Ok(())
    }

    /// Get all dynamic sources
    pub fn sources(&self) -> &std::collections::HashMap<String, DynamicSource> {
        &self.sources
    }

    /// Get dynamic source by ID
    pub fn get_source(&self, source_id: &str) -> Option<&DynamicSource> {
        self.sources.get(source_id)
    }

    /// Get mutable dynamic source by ID
    pub fn get_source_mut(&mut self, source_id: &str) -> Option<&mut DynamicSource> {
        self.sources.get_mut(source_id)
    }

    /// Predict source positions for latency compensation
    pub fn predict_source_positions(
        &self,
        prediction_time: Duration,
    ) -> std::collections::HashMap<String, Position3D> {
        let mut predictions = std::collections::HashMap::new();

        for (id, source) in &self.sources {
            if let Some(predicted_pos) = self
                .motion_predictor
                .predict_position(&source.motion_history, prediction_time)
            {
                predictions.insert(id.clone(), predicted_pos);
            }
        }

        predictions
    }
}

impl DynamicSource {
    /// Create new dynamic source
    pub fn new(base_source: SoundSource) -> Self {
        Self {
            base_source,
            velocity: Position3D::default(),
            acceleration: Position3D::default(),
            motion_history: VecDeque::with_capacity(100),
            doppler_factor: 1.0,
            smoothed_doppler_factor: 1.0,
            last_doppler_update: None,
        }
    }

    /// Update motion parameters
    pub fn update_motion(
        &mut self,
        position: Position3D,
        velocity: Position3D,
        acceleration: Position3D,
    ) {
        self.velocity = velocity;
        self.acceleration = acceleration;

        // Update position in base source
        self.base_source.set_position(position);

        // Add to motion history
        let snapshot = MotionSnapshot {
            position,
            velocity,
            acceleration,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
        };

        self.motion_history.push_back(snapshot);

        // Keep history size manageable
        while self.motion_history.len() > 100 {
            self.motion_history.pop_front();
        }
    }

    /// Get predicted position after given time
    pub fn predict_position(&self, time_delta: Duration) -> Position3D {
        let dt = time_delta.as_secs_f32();
        let current_pos = self.base_source.position();

        // Simple kinematic prediction: pos = pos0 + v*t + 0.5*a*t^2
        Position3D::new(
            current_pos.x + self.velocity.x * dt + 0.5 * self.acceleration.x * dt * dt,
            current_pos.y + self.velocity.y * dt + 0.5 * self.acceleration.y * dt * dt,
            current_pos.z + self.velocity.z * dt + 0.5 * self.acceleration.z * dt * dt,
        )
    }

    /// Check if source is moving significantly
    pub fn is_moving(&self) -> bool {
        self.velocity.magnitude() > 0.1 // m/s threshold
    }
}

impl MotionPredictor {
    /// Create new motion predictor
    pub fn new() -> Self {
        Self {
            prediction_horizon: 0.1, // 100ms prediction
            min_samples: 3,
            max_history_size: 50,
        }
    }

    /// Predict position based on motion history
    pub fn predict_position(
        &self,
        history: &VecDeque<MotionSnapshot>,
        prediction_time: Duration,
    ) -> Option<Position3D> {
        if history.len() < self.min_samples {
            return None;
        }

        // Get latest motion data
        let latest = history.back()?;
        let dt = prediction_time.as_secs_f32();

        // Simple kinematic prediction
        Some(Position3D::new(
            latest.position.x + latest.velocity.x * dt + 0.5 * latest.acceleration.x * dt * dt,
            latest.position.y + latest.velocity.y * dt + 0.5 * latest.acceleration.y * dt * dt,
            latest.position.z + latest.velocity.z * dt + 0.5 * latest.acceleration.z * dt * dt,
        ))
    }

    /// Set prediction horizon
    pub fn set_prediction_horizon(&mut self, horizon: f32) {
        self.prediction_horizon = horizon;
    }

    /// Get prediction horizon
    pub fn prediction_horizon(&self) -> f32 {
        self.prediction_horizon
    }
}

impl Default for MotionPredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_listener_creation() {
        let listener = Listener::new();
        assert_eq!(listener.position(), Position3D::default());
        assert_eq!(listener.orientation(), (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_listener_position_update() {
        let mut listener = Listener::new();
        let new_pos = Position3D::new(1.0, 2.0, 3.0);
        listener.set_position(new_pos);
        assert_eq!(listener.position(), new_pos);
    }

    #[test]
    fn test_sound_source_creation() {
        let source = SoundSource::new_point("test".to_string(), Position3D::new(5.0, 0.0, 0.0));
        assert_eq!(source.position(), Position3D::new(5.0, 0.0, 0.0));
        assert_eq!(source.source_type(), SourceType::Point);
        assert!(source.is_active());
    }

    #[test]
    fn test_ear_positions() {
        let mut listener = Listener::new();
        listener.set_interaural_distance(0.2);

        let left_ear = listener.left_ear_position();
        let right_ear = listener.right_ear_position();

        // With zero orientation, ears should be on x-axis
        assert_eq!(left_ear.x, -0.1);
        assert_eq!(right_ear.x, 0.1);
    }

    #[test]
    fn test_directivity_patterns() {
        let omni = DirectivityPattern::omnidirectional();
        assert_eq!(omni.front_gain, 1.0);
        assert_eq!(omni.back_gain, 1.0);

        let cardioid = DirectivityPattern::cardioid();
        assert_eq!(cardioid.front_gain, 1.0);
        assert_eq!(cardioid.back_gain, 0.0);
    }

    #[test]
    fn test_position_prediction() {
        let mut source = SoundSource::new_point("test".to_string(), Position3D::default());

        // Simulate movement
        source.set_position(Position3D::new(1.0, 0.0, 0.0));
        std::thread::sleep(Duration::from_millis(10));
        source.set_position(Position3D::new(2.0, 0.0, 0.0));

        // Should have positive velocity in x direction
        assert!(source.velocity().x > 0.0);

        // Predicted position should be further along x axis
        let predicted = source.predict_position(Duration::from_secs(1));
        assert!(predicted.x > 2.0);
    }

    #[test]
    fn test_head_tracker() {
        let mut tracker = HeadTracker::new();

        // Add position updates with explicit timestamps for predictable testing
        tracker.update_position_with_time(Position3D::new(0.0, 0.0, 0.0), 0.0);
        tracker.update_position_with_time(Position3D::new(1.0, 0.0, 0.0), 0.1); // 0.1s later
        tracker.update_position_with_time(Position3D::new(2.0, 0.0, 0.0), 0.2); // 0.2s later

        // Test prediction
        let predicted = tracker.predict_position(Duration::from_millis(100));
        assert!(predicted.is_some());

        let predicted_pos = predicted.unwrap();
        assert!(predicted_pos.x > 2.0); // Should be ahead of current position
    }

    #[test]
    fn test_head_tracker_orientation() {
        let mut tracker = HeadTracker::new();

        // Add orientation updates with explicit timestamps
        tracker.update_orientation_with_time((0.0, 0.0, 0.0), 0.0);
        tracker.update_orientation_with_time((0.1, 0.0, 0.0), 0.1); // 0.1s later
        tracker.update_orientation_with_time((0.2, 0.0, 0.0), 0.2); // 0.2s later

        // Test orientation prediction
        let predicted = tracker.predict_orientation(Duration::from_millis(100));
        assert!(predicted.is_some());

        let predicted_orient = predicted.unwrap();
        assert!(predicted_orient.0 > 0.2); // Should be ahead of current orientation
    }

    #[test]
    fn test_spatial_source_manager() {
        let bounds = (
            Position3D::new(-10.0, -10.0, -10.0),
            Position3D::new(10.0, 10.0, 10.0),
        );
        let mut manager = SpatialSourceManager::new(bounds, 2.0);

        // Add sources
        let source1 = SoundSource::new_point("source1".to_string(), Position3D::new(1.0, 0.0, 0.0));
        let source2 = SoundSource::new_point("source2".to_string(), Position3D::new(5.0, 0.0, 0.0));

        assert!(manager.add_source(source1).is_ok());
        assert!(manager.add_source(source2).is_ok());

        // Test nearby sources query
        let listener_pos = Position3D::new(0.0, 0.0, 0.0);
        let nearby = manager.get_nearby_sources(listener_pos, 3.0);
        assert!(!nearby.is_empty());

        // Test source removal
        let removed = manager.remove_source("source1");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, "source1");
    }

    #[test]
    fn test_spatial_grid() {
        let bounds = (
            Position3D::new(-10.0, -10.0, -10.0),
            Position3D::new(10.0, 10.0, 10.0),
        );
        let mut grid = SpatialGrid::new(bounds, 1.0); // Smaller cell size

        // Add sources to grid with more separation
        grid.add_source("source1", Position3D::new(1.0, 1.0, 1.0));
        grid.add_source("source2", Position3D::new(8.0, 8.0, 8.0));

        // Query nearby sources with smaller radius
        let nearby = grid.query_sphere(Position3D::new(0.0, 0.0, 0.0), 2.5);
        assert!(nearby.contains(&"source1".to_string()));
        assert!(!nearby.contains(&"source2".to_string()));

        // Test source movement
        grid.move_source(
            "source1",
            Position3D::new(1.0, 1.0, 1.0),
            Position3D::new(9.0, 9.0, 9.0),
        );

        let nearby_after_move = grid.query_sphere(Position3D::new(0.0, 0.0, 0.0), 2.5);
        assert!(!nearby_after_move.contains(&"source1".to_string()));
    }

    #[test]
    fn test_occlusion_detector() {
        let mut detector = OcclusionDetector::new();

        // Add obstacle
        let obstacle = Box3D {
            min: Position3D::new(-1.0, -1.0, -1.0),
            max: Position3D::new(1.0, 1.0, 1.0),
            material_id: "wall".to_string(),
        };
        detector.add_obstacle(obstacle);

        // Add material
        let material = OcclusionMaterial {
            name: "wall".to_string(),
            transmission: 0.2,
            high_freq_absorption: 0.8,
            low_freq_absorption: 0.4,
            scattering: 0.3,
        };
        detector.add_material(material);

        // Test occlusion (line passes through obstacle)
        let source = Position3D::new(-5.0, 0.0, 0.0);
        let listener = Position3D::new(5.0, 0.0, 0.0);
        let result = detector.check_occlusion(source, listener);

        assert!(result.is_occluded);
        assert_eq!(result.transmission_factor, 0.2);

        // Test no occlusion (line doesn't pass through obstacle)
        let source_clear = Position3D::new(-5.0, 5.0, 0.0);
        let listener_clear = Position3D::new(5.0, 5.0, 0.0);
        let result_clear = detector.check_occlusion(source_clear, listener_clear);

        assert!(!result_clear.is_occluded);
        assert_eq!(result_clear.transmission_factor, 1.0);
    }

    #[test]
    fn test_box3d_intersection() {
        let detector = OcclusionDetector::new();

        let box3d = Box3D {
            min: Position3D::new(-1.0, -1.0, -1.0),
            max: Position3D::new(1.0, 1.0, 1.0),
            material_id: "test".to_string(),
        };

        // Line passes through box
        let start = Position3D::new(-2.0, 0.0, 0.0);
        let end = Position3D::new(2.0, 0.0, 0.0);
        assert!(detector.line_intersects_box(start, end, &box3d));

        // Line doesn't intersect box
        let start_miss = Position3D::new(-2.0, 2.0, 0.0);
        let end_miss = Position3D::new(2.0, 2.0, 0.0);
        assert!(!detector.line_intersects_box(start_miss, end_miss, &box3d));
    }

    #[test]
    fn test_angle_difference() {
        let tracker = HeadTracker::new();

        // Test normal angle difference
        let diff1 = tracker.angle_difference(0.5, 0.0);
        assert!((diff1 - 0.5).abs() < 0.001);

        // Test wrap-around with PI values
        let diff2 = tracker.angle_difference(0.1, 2.0 * std::f32::consts::PI - 0.1);
        assert!(diff2.abs() < 0.5); // Should be small after wrap-around

        // Test wrap-around (crossing zero)
        let diff3 = tracker.angle_difference(-0.1, 0.1);
        assert!((diff3 + 0.2).abs() < 0.001);

        // Test wrap-around near PI
        let diff4 =
            tracker.angle_difference(std::f32::consts::PI - 0.1, -std::f32::consts::PI + 0.1);
        assert!(diff4.abs() < 0.5); // Should wrap around to small difference
    }

    #[test]
    fn test_source_manager_culling() {
        let bounds = (
            Position3D::new(-50.0, -50.0, -50.0),
            Position3D::new(50.0, 50.0, 50.0),
        );
        let mut manager = SpatialSourceManager::new(bounds, 5.0);

        // Set a small culling distance for testing
        manager.culling_distance = 10.0;

        // Add sources at different distances
        let near_source =
            SoundSource::new_point("near".to_string(), Position3D::new(2.0, 0.0, 0.0));
        let far_source = SoundSource::new_point("far".to_string(), Position3D::new(20.0, 0.0, 0.0));

        manager.add_source(near_source).unwrap();
        manager.add_source(far_source).unwrap();

        assert_eq!(manager.sources.len(), 2);

        // Cull distant sources
        let listener_pos = Position3D::new(0.0, 0.0, 0.0);
        manager.cull_distant_sources(listener_pos);

        // Far source should be culled
        assert_eq!(manager.sources.len(), 1);
        assert!(manager.sources.contains_key("near"));
        assert!(!manager.sources.contains_key("far"));
    }

    #[test]
    fn test_doppler_processor_creation() {
        let doppler = DopplerProcessor::new(44100.0);
        assert_eq!(doppler.speed_of_sound(), 343.0);
    }

    #[test]
    fn test_doppler_factor_calculation() {
        let doppler = DopplerProcessor::new(44100.0);

        // Stationary source and listener
        let source_pos = Position3D::new(0.0, 0.0, 0.0);
        let source_vel = Position3D::new(0.0, 0.0, 0.0);
        let listener_pos = Position3D::new(10.0, 0.0, 0.0);
        let listener_vel = Position3D::new(0.0, 0.0, 0.0);

        let factor =
            doppler.calculate_doppler_factor(source_pos, source_vel, listener_pos, listener_vel);
        assert!((factor - 1.0).abs() < 0.001); // Should be no Doppler effect

        // Source approaching listener
        let approaching_vel = Position3D::new(10.0, 0.0, 0.0); // 10 m/s towards listener
        let factor_approaching = doppler.calculate_doppler_factor(
            source_pos,
            approaching_vel,
            listener_pos,
            listener_vel,
        );
        assert!(factor_approaching > 1.0); // Higher frequency when approaching

        // Source receding from listener
        let receding_vel = Position3D::new(-10.0, 0.0, 0.0); // 10 m/s away from listener
        let factor_receding =
            doppler.calculate_doppler_factor(source_pos, receding_vel, listener_pos, listener_vel);
        assert!(factor_receding < 1.0); // Lower frequency when receding
    }

    #[test]
    fn test_dynamic_source_manager() {
        let mut manager = DynamicSourceManager::new(44100.0);

        // Create a sound source
        let source =
            SoundSource::new_point("test_source".to_string(), Position3D::new(0.0, 0.0, 0.0));

        // Add to manager
        assert!(manager.add_source(source).is_ok());
        assert_eq!(manager.sources().len(), 1);

        // Update motion
        let position = Position3D::new(1.0, 0.0, 0.0);
        let velocity = Position3D::new(5.0, 0.0, 0.0);
        let acceleration = Position3D::new(0.0, 0.0, 0.0);

        assert!(manager
            .update_source_motion("test_source", position, velocity, acceleration)
            .is_ok());

        // Check motion was updated
        let source = manager.get_source("test_source").unwrap();
        assert_eq!(source.velocity, velocity);
    }

    #[tokio::test]
    async fn test_dynamic_source_processing() {
        let mut manager = DynamicSourceManager::new(44100.0);

        // Add a moving source
        let source =
            SoundSource::new_point("moving_source".to_string(), Position3D::new(0.0, 0.0, 0.0));
        manager.add_source(source).unwrap();

        // Update with motion
        manager
            .update_source_motion(
                "moving_source",
                Position3D::new(1.0, 0.0, 0.0),
                Position3D::new(10.0, 0.0, 0.0), // Moving at 10 m/s
                Position3D::new(0.0, 0.0, 0.0),
            )
            .unwrap();

        // Process with listener
        let listener_pos = Position3D::new(10.0, 0.0, 0.0);
        let listener_vel = Position3D::new(0.0, 0.0, 0.0);

        assert!(manager
            .process_dynamic_sources(listener_pos, listener_vel)
            .await
            .is_ok());

        // Check Doppler factor was calculated
        let source = manager.get_source("moving_source").unwrap();
        assert!(source.doppler_factor > 1.0); // Should be higher frequency (approaching)
    }

    #[test]
    fn test_doppler_audio_processing() {
        let doppler = DopplerProcessor::new(44100.0);

        // Create test audio (sine wave)
        let input: Vec<f32> = (0..1000)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI / 44.1).sin())
            .collect();
        let mut output = vec![0.0; 1000];

        // Apply no Doppler effect
        assert!(doppler
            .process_doppler_effect(&input, &mut output, 1.0)
            .is_ok());

        // Should be identical
        for i in 0..input.len() {
            assert!((input[i] - output[i]).abs() < 0.001);
        }

        // Apply Doppler effect
        assert!(doppler
            .process_doppler_effect(&input, &mut output, 1.1)
            .is_ok());

        // Output should be different (higher frequency)
        let mut differences = 0;
        for i in 0..input.len() {
            if (input[i] - output[i]).abs() > 0.001 {
                differences += 1;
            }
        }
        assert!(differences > 100); // Should have many differences
    }

    #[test]
    fn test_motion_prediction() {
        let predictor = MotionPredictor::new();
        let mut history = VecDeque::new();

        // Add motion snapshots
        for i in 0..5 {
            let snapshot = MotionSnapshot {
                position: Position3D::new(i as f32, 0.0, 0.0),
                velocity: Position3D::new(1.0, 0.0, 0.0),
                acceleration: Position3D::new(0.0, 0.0, 0.0),
                timestamp: i as f64 * 0.1,
            };
            history.push_back(snapshot);
        }

        // Predict position
        let prediction_time = Duration::from_millis(100);
        let predicted = predictor
            .predict_position(&history, prediction_time)
            .unwrap();

        // Should predict ahead
        assert!(predicted.x > 4.0); // Last position was 4.0, should be ahead
    }

    #[test]
    fn test_dynamic_source_motion_history() {
        let source = SoundSource::new_point("test".to_string(), Position3D::default());
        let mut dynamic_source = DynamicSource::new(source);

        // Update motion several times
        for i in 0..10 {
            let pos = Position3D::new(i as f32, 0.0, 0.0);
            let vel = Position3D::new(1.0, 0.0, 0.0);
            let acc = Position3D::new(0.0, 0.0, 0.0);
            dynamic_source.update_motion(pos, vel, acc);
        }

        // Should have motion history
        assert_eq!(dynamic_source.motion_history.len(), 10);

        // Test prediction
        let predicted = dynamic_source.predict_position(Duration::from_millis(100));
        assert!(predicted.x > 9.0); // Should predict ahead of last position
        assert!(dynamic_source.is_moving()); // Should detect movement
    }

    #[test]
    fn test_enhanced_head_tracker_configuration() {
        let mut tracker = HeadTracker::new();

        // Test configuration
        tracker.configure(30, 50, 0.5, 0.6);
        tracker.set_prediction_enabled(true);
        tracker.set_latency_compensation(20.0);

        // Add some position data
        for i in 0..5 {
            let pos = Position3D::new(i as f32 * 0.1, 0.0, 0.0);
            tracker.update_position_with_time(pos, i as f64 * 0.1);
        }

        // Test current position
        assert!(tracker.current_position().is_some());
        assert!(tracker.current_velocity().is_some());

        // Test prediction quality
        let quality = tracker.prediction_quality();
        assert!(quality >= 0.0 && quality <= 1.0);

        // Test reset
        tracker.reset();
        assert!(tracker.current_position().is_none());
    }

    #[test]
    fn test_listener_movement_system() {
        let mut system = ListenerMovementSystem::new();

        // Test initial state
        assert_eq!(system.navigation_mode, NavigationMode::FreeFlight);
        assert!(system.enable_movement_prediction);

        // Test position update
        let position = Position3D::new(1.0, 0.0, 0.0);
        let result = system.update_position_from_platform(position, None);
        assert!(result.is_ok());

        assert_eq!(system.listener().position(), position);

        // Test prediction
        let predicted = system.predict_position(Duration::from_millis(100));
        assert!(predicted.is_some());

        // Test metrics
        let metrics = system.movement_metrics();
        assert!(metrics.update_count > 0);
    }

    #[test]
    fn test_navigation_modes() {
        // Test seated mode
        let seated_system = ListenerMovementSystem::with_navigation_mode(NavigationMode::Seated);
        assert_eq!(seated_system.movement_constraints.max_speed, 0.0);
        assert!(!seated_system.enable_movement_prediction);

        // Test walking mode
        let walking_system = ListenerMovementSystem::with_navigation_mode(NavigationMode::Walking);
        assert_eq!(walking_system.movement_constraints.max_speed, 5.0);
        assert!(walking_system.comfort_settings.ground_reference);

        // Test vehicle mode
        let vehicle_system = ListenerMovementSystem::with_navigation_mode(NavigationMode::Vehicle);
        assert_eq!(vehicle_system.movement_constraints.max_speed, 50.0);
        assert_eq!(
            vehicle_system.comfort_settings.motion_sickness_reduction,
            0.7
        );
    }

    #[test]
    fn test_movement_constraints() {
        let mut system = ListenerMovementSystem::new();

        // Set boundary constraints
        let boundary = Box3D {
            min: Position3D::new(-5.0, 0.0, -5.0),
            max: Position3D::new(5.0, 3.0, 5.0),
            material_id: "boundary".to_string(),
        };

        let constraints = MovementConstraints {
            boundary: Some(boundary),
            max_speed: 2.0,
            max_acceleration: 5.0,
            ground_height: Some(0.0),
            ceiling_height: Some(3.0),
        };

        system.set_movement_constraints(constraints);

        // Test position constraint
        let out_of_bounds = Position3D::new(10.0, -1.0, 0.0);
        let result = system.update_position_from_platform(out_of_bounds, None);
        assert!(result.is_ok());

        // Position should be constrained
        let actual_pos = system.listener().position();
        assert!(actual_pos.x <= 5.0);
        assert!(actual_pos.y >= 0.0);
    }

    #[test]
    fn test_comfort_settings() {
        let mut system = ListenerMovementSystem::new();

        let comfort = ComfortSettings {
            motion_sickness_reduction: 0.8,
            snap_turn: true,
            snap_turn_degrees: 45.0,
            movement_vignetting: true,
            ground_reference: true,
            speed_multiplier: 0.8,
        };

        system.set_comfort_settings(comfort.clone());

        // Test orientation snap
        let orientation = (0.1, 0.0, 0.0); // Small rotation
        let result = system.update_orientation_from_platform(orientation, None);
        assert!(result.is_ok());

        // Should be snapped to nearest 45-degree increment
        let actual_orientation = system.listener().orientation();
        assert!(actual_orientation.0 % 45.0f32.to_radians() < 0.01);
    }

    #[test]
    fn test_platform_integration() {
        let mut integration = PlatformIntegration::new(PlatformType::Oculus);
        assert_eq!(integration.platform_type(), PlatformType::Oculus);

        // Test platform data update
        let data = PlatformData {
            device_id: "Oculus Quest 2".to_string(),
            pose_data: vec![1.0, 0.0, 0.0, 0.0], // Quaternion
            tracking_confidence: 0.95,
            platform_timestamp: 123456,
            properties: std::collections::HashMap::new(),
        };

        integration.update_tracking_data(data.clone());
        assert_eq!(integration.tracking_confidence(), 0.95);

        // Test calibration
        let calibration = CalibrationData {
            head_circumference: Some(58.5),
            ipd: Some(63.0),
            height_offset: 1.75,
            forward_offset: 0.0,
            custom_hrtf_profile: None,
        };

        integration.set_calibration(calibration);
    }

    #[test]
    fn test_movement_metrics() {
        let mut system = ListenerMovementSystem::new();

        // Move in a pattern
        for i in 0..10 {
            let pos = Position3D::new(i as f32 * 0.5, 0.0, 0.0);
            system.update_position_from_platform(pos, None).unwrap();
        }

        let metrics = system.movement_metrics();
        assert!(metrics.total_distance > 0.0);
        assert!(metrics.update_count == 10);

        // Reset and verify
        system.reset_metrics();
        let new_metrics = system.movement_metrics();
        assert_eq!(new_metrics.total_distance, 0.0);
        assert_eq!(new_metrics.update_count, 0);
    }

    #[test]
    fn test_platform_types() {
        // Test all platform types can be created
        let platforms = [
            PlatformType::Generic,
            PlatformType::Oculus,
            PlatformType::SteamVR,
            PlatformType::ARKit,
            PlatformType::ARCore,
            PlatformType::WMR,
            PlatformType::Custom,
        ];

        for platform in platforms.iter() {
            let integration = PlatformIntegration::new(*platform);
            assert_eq!(integration.platform_type(), *platform);
        }
    }
}
