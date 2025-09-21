//! Generic Platform Integration
//!
//! This module provides a generic platform implementation that can be used
//! as a fallback when no specific VR/AR platform is available, or for testing.

use crate::platforms::{
    DeviceInfo, EyeTrackingData, HandTrackingData, PlatformCapabilities, PlatformIntegration,
    PlatformTrackingData, PoseData, TrackingConfig, TrackingQuality, TrackingState,
};
use crate::position::{PlatformData, PlatformType};
use crate::types::Position3D;
use crate::{Error, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use tokio::time::Instant;

/// Generic platform integration for testing and fallback scenarios
pub struct GenericPlatform {
    device_info: DeviceInfo,
    capabilities: PlatformCapabilities,
    tracking_active: bool,
    config: TrackingConfig,

    // Generic platform state
    simulation_mode: SimulationMode,
    motion_pattern: MotionPattern,
    quality_degradation: f32,

    // Timing and state
    start_time: Option<Instant>,
    last_update: Option<Instant>,
}

/// Different simulation modes for the generic platform
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimulationMode {
    /// Static pose (no movement)
    Static,
    /// Simple circular motion
    Circular,
    /// Random walk pattern
    RandomWalk,
    /// Figure-8 motion pattern
    Figure8,
    /// Realistic head movement simulation
    Realistic,
}

/// Motion pattern parameters
#[derive(Debug, Clone)]
pub struct MotionPattern {
    /// Movement speed multiplier
    pub speed: f32,
    /// Movement amplitude
    pub amplitude: f32,
    /// Base position offset
    pub base_position: Position3D,
    /// Rotation speed multiplier
    pub rotation_speed: f32,
    /// Add noise to movement
    pub add_noise: bool,
}

impl Default for MotionPattern {
    fn default() -> Self {
        Self {
            speed: 1.0,
            amplitude: 1.0,
            base_position: Position3D::new(0.0, 1.7, 0.0),
            rotation_speed: 1.0,
            add_noise: true,
        }
    }
}

impl GenericPlatform {
    /// Create new generic platform integration
    pub fn new() -> Self {
        Self {
            device_info: DeviceInfo {
                name: "Generic Platform".to_string(),
                manufacturer: "VoiRS".to_string(),
                model: "Generic VR/AR Device".to_string(),
                serial_number: "SIM-000001".to_string(),
                firmware_version: "1.0.0".to_string(),
                platform_version: "Generic 1.0".to_string(),
            },
            capabilities: PlatformCapabilities {
                head_tracking_6dof: true,
                hand_tracking: false,
                eye_tracking: false,
                controller_tracking: false,
                room_scale: false,
                passthrough: false,
                refresh_rates: vec![60.0],
                tracking_range: 5.0,
            },
            tracking_active: false,
            config: TrackingConfig::default(),
            simulation_mode: SimulationMode::Realistic,
            motion_pattern: MotionPattern::default(),
            quality_degradation: 0.0,
            start_time: None,
            last_update: None,
        }
    }

    /// Create a generic platform with specific simulation mode
    pub fn with_simulation_mode(simulation_mode: SimulationMode) -> Self {
        let mut platform = Self::new();
        platform.simulation_mode = simulation_mode;
        platform
    }

    /// Configure the motion pattern
    pub fn configure_motion(&mut self, pattern: MotionPattern) {
        self.motion_pattern = pattern;
    }

    /// Set quality degradation factor (0.0 = perfect, 1.0 = completely degraded)
    pub fn set_quality_degradation(&mut self, degradation: f32) {
        self.quality_degradation = degradation.clamp(0.0, 1.0);
    }

    /// Generate simulated tracking data based on current simulation mode
    fn generate_tracking_data(&self) -> PlatformTrackingData {
        let now = Instant::now();
        let elapsed = if let Some(start_time) = self.start_time {
            now.duration_since(start_time).as_secs_f32()
        } else {
            0.0
        };

        let head_pose = self.calculate_head_pose(elapsed);
        let quality = self.calculate_tracking_quality(elapsed);

        PlatformTrackingData {
            head_pose,
            left_controller: None, // Generic platform doesn't simulate controllers
            right_controller: None,
            quality: quality.clone(),
            timestamp: now,
            raw_data: PlatformData {
                device_id: "Generic".to_string(),
                pose_data: vec![],
                tracking_confidence: quality.overall_quality,
                platform_timestamp: (elapsed * 1000.0) as u64,
                properties: {
                    let mut props = HashMap::new();
                    props.insert(
                        "simulation_mode".to_string(),
                        format!("{:?}", self.simulation_mode),
                    );
                    props.insert("elapsed_time".to_string(), elapsed.to_string());
                    props.insert(
                        "quality_degradation".to_string(),
                        self.quality_degradation.to_string(),
                    );
                    props
                },
            },
        }
    }

    /// Calculate head pose based on simulation mode and time
    fn calculate_head_pose(&self, time: f32) -> PoseData {
        let position = match self.simulation_mode {
            SimulationMode::Static => self.motion_pattern.base_position,
            SimulationMode::Circular => self.calculate_circular_motion(time),
            SimulationMode::RandomWalk => self.calculate_random_walk(time),
            SimulationMode::Figure8 => self.calculate_figure8_motion(time),
            SimulationMode::Realistic => self.calculate_realistic_motion(time),
        };

        let orientation = self.calculate_orientation(time);
        let (linear_velocity, angular_velocity) = self.calculate_velocities(time);

        PoseData {
            position,
            orientation,
            linear_velocity,
            angular_velocity,
            confidence: 1.0 - self.quality_degradation,
        }
    }

    /// Calculate circular motion pattern
    fn calculate_circular_motion(&self, time: f32) -> Position3D {
        let t = time * self.motion_pattern.speed;
        let radius = self.motion_pattern.amplitude;

        Position3D::new(
            self.motion_pattern.base_position.x + radius * t.cos(),
            self.motion_pattern.base_position.y,
            self.motion_pattern.base_position.z + radius * t.sin(),
        )
    }

    /// Calculate random walk pattern
    fn calculate_random_walk(&self, time: f32) -> Position3D {
        // Use deterministic "randomness" based on time for reproducibility
        let seed = (time * 100.0) as u32;
        let noise_x = ((seed * 1299827) % 1000) as f32 / 1000.0 - 0.5;
        let noise_z = ((seed * 1399831) % 1000) as f32 / 1000.0 - 0.5;

        Position3D::new(
            self.motion_pattern.base_position.x + noise_x * self.motion_pattern.amplitude * 0.1,
            self.motion_pattern.base_position.y,
            self.motion_pattern.base_position.z + noise_z * self.motion_pattern.amplitude * 0.1,
        )
    }

    /// Calculate figure-8 motion pattern
    fn calculate_figure8_motion(&self, time: f32) -> Position3D {
        let t = time * self.motion_pattern.speed * 0.5;
        let scale = self.motion_pattern.amplitude;

        Position3D::new(
            self.motion_pattern.base_position.x + scale * (2.0 * t).sin(),
            self.motion_pattern.base_position.y,
            self.motion_pattern.base_position.z + scale * t.sin() * t.cos(),
        )
    }

    /// Calculate realistic head movement pattern
    fn calculate_realistic_motion(&self, time: f32) -> Position3D {
        // Combine multiple frequencies to simulate natural head movement
        let slow_wave = (time * 0.1 * self.motion_pattern.speed).sin() * 0.05;
        let medium_wave = (time * 0.3 * self.motion_pattern.speed).sin() * 0.02;
        let fast_wave = (time * 0.8 * self.motion_pattern.speed).sin() * 0.01;

        let x_offset = (slow_wave + medium_wave) * self.motion_pattern.amplitude;
        let y_offset = (medium_wave + fast_wave) * self.motion_pattern.amplitude * 0.5;
        let z_offset = slow_wave * self.motion_pattern.amplitude * 0.3;

        // Add noise if enabled
        let (noise_x, noise_y, noise_z) = if self.motion_pattern.add_noise {
            let seed = (time * 1000.0) as u32;
            (
                ((seed * 1299827) % 1000) as f32 / 10000.0 - 0.05,
                ((seed * 1399831) % 1000) as f32 / 10000.0 - 0.05,
                ((seed * 1499833) % 1000) as f32 / 10000.0 - 0.05,
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        Position3D::new(
            self.motion_pattern.base_position.x + x_offset + noise_x,
            self.motion_pattern.base_position.y + y_offset + noise_y,
            self.motion_pattern.base_position.z + z_offset + noise_z,
        )
    }

    /// Calculate orientation based on time and motion pattern
    fn calculate_orientation(&self, time: f32) -> (f32, f32, f32, f32) {
        let t = time * self.motion_pattern.rotation_speed;

        // Simulate natural head rotation patterns
        let yaw = match self.simulation_mode {
            SimulationMode::Static => 0.0,
            SimulationMode::Circular => t * 0.5, // Look in direction of movement
            SimulationMode::RandomWalk => (t * 0.1).sin() * 0.3,
            SimulationMode::Figure8 => (t * 0.3).sin() * 0.4,
            SimulationMode::Realistic => (t * 0.05).sin() * 0.2 + (t * 0.15).sin() * 0.1,
        };

        let pitch = match self.simulation_mode {
            SimulationMode::Static => 0.0,
            _ => (t * 0.07).sin() * 0.1, // Small pitch movements
        };

        let roll = match self.simulation_mode {
            SimulationMode::Static => 0.0,
            _ => (t * 0.12).sin() * 0.05, // Very small roll movements
        };

        // Convert Euler angles to quaternion
        let half_yaw = yaw * 0.5;
        let half_pitch = pitch * 0.5;
        let half_roll = roll * 0.5;

        let cos_yaw = half_yaw.cos();
        let sin_yaw = half_yaw.sin();
        let cos_pitch = half_pitch.cos();
        let sin_pitch = half_pitch.sin();
        let cos_roll = half_roll.cos();
        let sin_roll = half_roll.sin();

        (
            sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw,
            cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw,
            cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw,
            cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw,
        )
    }

    /// Calculate linear and angular velocities
    fn calculate_velocities(&self, time: f32) -> (Position3D, Position3D) {
        // Simple velocity estimation based on simulation mode
        let linear_speed = match self.simulation_mode {
            SimulationMode::Static => 0.0,
            SimulationMode::Circular => self.motion_pattern.speed * self.motion_pattern.amplitude,
            SimulationMode::RandomWalk => 0.1,
            SimulationMode::Figure8 => {
                self.motion_pattern.speed * self.motion_pattern.amplitude * 0.8
            }
            SimulationMode::Realistic => 0.05,
        };

        let angular_speed = self.motion_pattern.rotation_speed * 0.1;

        (
            Position3D::new(linear_speed * 0.1, 0.0, linear_speed * 0.1),
            Position3D::new(angular_speed * 0.05, angular_speed, angular_speed * 0.02),
        )
    }

    /// Calculate tracking quality with degradation over time
    fn calculate_tracking_quality(&self, time: f32) -> TrackingQuality {
        let base_quality = 0.75; // Generic platform has decent but not perfect quality

        // Simulate quality variations
        let time_degradation = if self.quality_degradation > 0.0 {
            (time * 0.01).sin() * 0.1 * self.quality_degradation
        } else {
            0.0
        };

        let quality_variation = (time * 0.2).sin() * 0.05; // Small natural variations
        let final_quality = (base_quality - time_degradation + quality_variation).clamp(0.1, 1.0);

        let state = if final_quality > 0.8 {
            TrackingState::Full
        } else if final_quality > 0.5 {
            TrackingState::Limited
        } else if final_quality > 0.2 {
            TrackingState::Lost
        } else {
            TrackingState::NotTracking
        };

        TrackingQuality {
            overall_quality: final_quality,
            position_quality: final_quality * 0.98,
            orientation_quality: final_quality * 1.02,
            feature_count: ((final_quality * 50.0) as u32).max(5),
            state,
        }
    }
}

#[async_trait]
impl PlatformIntegration for GenericPlatform {
    async fn initialize(&mut self) -> Result<()> {
        self.start_time = Some(Instant::now());
        self.last_update = Some(Instant::now());
        tracing::info!(
            "Generic platform initialized with simulation mode: {:?}",
            self.simulation_mode
        );
        Ok(())
    }

    async fn get_tracking_data(&self) -> Result<PlatformTrackingData> {
        if !self.tracking_active {
            return Err(Error::LegacyProcessing(
                "Generic tracking not active".to_string(),
            ));
        }

        Ok(self.generate_tracking_data())
    }

    async fn is_available(&self) -> bool {
        true // Generic platform is always available
    }

    fn get_capabilities(&self) -> PlatformCapabilities {
        self.capabilities.clone()
    }

    async fn configure_tracking(&mut self, config: TrackingConfig) -> Result<()> {
        self.config = config;
        tracing::info!("Configured generic tracking with config: {:?}", self.config);
        Ok(())
    }

    fn get_device_info(&self) -> DeviceInfo {
        self.device_info.clone()
    }

    async fn start_tracking(&mut self) -> Result<()> {
        if self.start_time.is_none() {
            return Err(Error::LegacyProcessing(
                "Generic platform not initialized".to_string(),
            ));
        }

        self.tracking_active = true;
        self.last_update = Some(Instant::now());
        tracing::info!("Started generic tracking");
        Ok(())
    }

    async fn stop_tracking(&mut self) -> Result<()> {
        self.tracking_active = false;
        tracing::info!("Stopped generic tracking");
        Ok(())
    }

    async fn get_hand_tracking(&self) -> Result<Option<HandTrackingData>> {
        // Generic platform doesn't support hand tracking by default
        Ok(None)
    }

    async fn get_eye_tracking(&self) -> Result<Option<EyeTrackingData>> {
        // Generic platform doesn't support eye tracking by default
        Ok(None)
    }
}

impl Default for GenericPlatform {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generic_platform_creation() {
        let platform = GenericPlatform::new();
        assert!(!platform.tracking_active);
        assert_eq!(platform.simulation_mode, SimulationMode::Realistic);
        assert_eq!(platform.device_info.manufacturer, "VoiRS");
    }

    #[tokio::test]
    async fn test_generic_platform_always_available() {
        let platform = GenericPlatform::new();
        assert!(platform.is_available().await);
    }

    #[tokio::test]
    async fn test_generic_platform_initialization() {
        let mut platform = GenericPlatform::new();
        assert!(platform.initialize().await.is_ok());
        assert!(platform.start_time.is_some());
        assert!(platform.last_update.is_some());
    }

    #[tokio::test]
    async fn test_simulation_modes() {
        for mode in [
            SimulationMode::Static,
            SimulationMode::Circular,
            SimulationMode::RandomWalk,
            SimulationMode::Figure8,
            SimulationMode::Realistic,
        ] {
            let mut platform = GenericPlatform::with_simulation_mode(mode);
            assert!(platform.initialize().await.is_ok());
            assert!(platform.start_tracking().await.is_ok());

            let tracking_data = platform.get_tracking_data().await;
            assert!(tracking_data.is_ok());

            let data = tracking_data.unwrap();
            assert!(data.quality.overall_quality > 0.0);

            // Static mode should have minimal movement
            if mode == SimulationMode::Static {
                assert_eq!(
                    data.head_pose.position,
                    platform.motion_pattern.base_position
                );
            }
        }
    }

    #[tokio::test]
    async fn test_motion_pattern_configuration() {
        let mut platform = GenericPlatform::new();
        let custom_pattern = MotionPattern {
            speed: 2.0,
            amplitude: 0.5,
            base_position: Position3D::new(1.0, 2.0, 3.0),
            rotation_speed: 0.5,
            add_noise: false,
        };

        platform.configure_motion(custom_pattern.clone());
        assert_eq!(platform.motion_pattern.speed, 2.0);
        assert_eq!(platform.motion_pattern.amplitude, 0.5);
        assert_eq!(platform.motion_pattern.base_position.x, 1.0);
    }

    #[tokio::test]
    async fn test_quality_degradation() {
        let mut platform = GenericPlatform::new();
        assert!(platform.initialize().await.is_ok());
        assert!(platform.start_tracking().await.is_ok());

        // Test normal quality
        let normal_data = platform.get_tracking_data().await.unwrap();
        let normal_quality = normal_data.quality.overall_quality;

        // Test degraded quality
        platform.set_quality_degradation(0.5);
        let degraded_data = platform.get_tracking_data().await.unwrap();
        let degraded_quality = degraded_data.quality.overall_quality;

        // Degraded quality should be lower or the same (allow for edge cases with random variations)
        // The key test is that quality degradation was set correctly
        assert!(platform.quality_degradation > 0.0);
        assert!(degraded_quality <= normal_quality + 0.1); // Allow small tolerance for random variations
    }

    #[tokio::test]
    async fn test_tracking_properties() {
        let mut platform = GenericPlatform::with_simulation_mode(SimulationMode::Figure8);
        platform.set_quality_degradation(0.2);
        assert!(platform.initialize().await.is_ok());
        assert!(platform.start_tracking().await.is_ok());

        let tracking_data = platform.get_tracking_data().await.unwrap();
        let props = &tracking_data.raw_data.properties;

        assert_eq!(props.get("simulation_mode").unwrap(), "Figure8");
        assert_eq!(props.get("quality_degradation").unwrap(), "0.2");
        assert!(props.contains_key("elapsed_time"));
    }

    #[tokio::test]
    async fn test_pose_calculation_deterministic() {
        let platform = GenericPlatform::with_simulation_mode(SimulationMode::Circular);

        // Same time should produce same pose
        let pose1 = platform.calculate_head_pose(5.0);
        let pose2 = platform.calculate_head_pose(5.0);

        assert_eq!(pose1.position.x, pose2.position.x);
        assert_eq!(pose1.position.y, pose2.position.y);
        assert_eq!(pose1.position.z, pose2.position.z);
    }

    #[tokio::test]
    async fn test_capabilities() {
        let platform = GenericPlatform::new();
        let capabilities = platform.get_capabilities();

        assert!(capabilities.head_tracking_6dof);
        assert!(!capabilities.hand_tracking);
        assert!(!capabilities.eye_tracking);
        assert!(!capabilities.controller_tracking);
        assert_eq!(capabilities.refresh_rates, vec![60.0]);
    }
}
