//! Head tracking and movement tracking systems

use super::types::{
    Box3D, CalibrationData, ComfortSettings, Listener, MovementConstraints, MovementMetrics,
    NavigationMode, OrientationSnapshot, PlatformData, PlatformType, PositionSnapshot,
};
use crate::types::Position3D;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

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

/// Movement detection and prediction
#[derive(Debug, Clone)]
pub struct MovementTracker {
    /// Position history size
    history_size: usize,
    /// Prediction lookahead time
    prediction_time: Duration,
    /// Velocity smoothing factor
    smoothing_factor: f32,
}

/// Real-time listener movement and navigation system
#[derive(Debug, Clone)]
pub struct ListenerMovementSystem {
    /// Current listener reference
    listener: Listener,
    /// Head tracking system
    head_tracker: HeadTracker,
    /// Movement prediction enabled
    pub enable_movement_prediction: bool,
    /// Navigation mode
    pub navigation_mode: NavigationMode,
    /// Comfort settings for VR
    pub comfort_settings: ComfortSettings,
    /// Movement constraints
    pub movement_constraints: MovementConstraints,
    /// Performance metrics
    movement_metrics: MovementMetrics,
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
    pub fn angle_difference(&self, angle1: f32, angle2: f32) -> f32 {
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

    /// Get position history for advanced prediction
    pub fn position_history(&self) -> &[PositionSnapshot] {
        // Convert VecDeque to slice using make_contiguous
        let (first, _) = self.position_history.as_slices();
        first
    }
}

impl Default for HeadTracker {
    fn default() -> Self {
        Self::new()
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
    pub fn track_source(&self, _source: &mut super::types::SoundSource) {
        // Movement tracking is handled in set_position
        // This method could be used for additional processing
    }

    /// Predict collision or intersection
    pub fn predict_intersection(
        &self,
        source: &super::types::SoundSource,
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

impl Default for ListenerMovementSystem {
    fn default() -> Self {
        Self::new()
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

impl Default for PlatformIntegration {
    fn default() -> Self {
        Self::new(PlatformType::Generic)
    }
}
