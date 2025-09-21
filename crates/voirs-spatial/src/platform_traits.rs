//! VR/AR Platform Integration Module
//!
//! This module provides comprehensive integration with major VR/AR platforms,
//! including Oculus/Meta, SteamVR, ARKit, ARCore, and Windows Mixed Reality.

use crate::position::{CalibrationData, PlatformData, PlatformType};
use crate::types::Position3D;
use crate::{Error, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::Instant;

// Platform-specific implementations
pub mod oculus;
pub mod steamvr;
pub mod arkit;
pub mod arcore;
pub mod wmr;
pub mod generic;

// Re-export platform implementations
pub use oculus::OculusPlatform;
#[cfg(feature = "steamvr")]
pub use steamvr::SteamVRPlatform;
pub use arkit::ARKitPlatform;
pub use arcore::ARCorePlatform;
pub use wmr::WMRPlatform;
pub use generic::GenericPlatform;

/// Platform integration trait for VR/AR platforms
#[async_trait]
pub trait PlatformIntegration: Send + Sync {
    /// Initialize the platform integration
    async fn initialize(&mut self) -> Result<()>;

    /// Get current tracking data
    async fn get_tracking_data(&self) -> Result<PlatformTrackingData>;

    /// Check if platform is available
    async fn is_available(&self) -> bool;

    /// Get platform capabilities
    fn get_capabilities(&self) -> PlatformCapabilities;

    /// Set tracking configuration
    async fn configure_tracking(&mut self, config: TrackingConfig) -> Result<()>;

    /// Get device information
    fn get_device_info(&self) -> DeviceInfo;

    /// Start tracking session
    async fn start_tracking(&mut self) -> Result<()>;

    /// Stop tracking session  
    async fn stop_tracking(&mut self) -> Result<()>;

    /// Get hand tracking data if available
    async fn get_hand_tracking(&self) -> Result<Option<HandTrackingData>>;

    /// Get eye tracking data if available
    async fn get_eye_tracking(&self) -> Result<Option<EyeTrackingData>>;
}

/// Complete tracking data from platform
#[derive(Debug, Clone)]
pub struct PlatformTrackingData {
    /// Head pose (position + orientation)
    pub head_pose: PoseData,
    /// Left controller pose (if available)
    pub left_controller: Option<PoseData>,
    /// Right controller pose (if available)
    pub right_controller: Option<PoseData>,
    /// Tracking quality metrics
    pub quality: TrackingQuality,
    /// Platform timestamp
    pub timestamp: Instant,
    /// Raw platform data
    pub raw_data: PlatformData,
}

/// Pose data (position + orientation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseData {
    /// Position in 3D space
    pub position: Position3D,
    /// Orientation quaternion (x, y, z, w)
    pub orientation: (f32, f32, f32, f32),
    /// Linear velocity
    pub linear_velocity: Position3D,
    /// Angular velocity
    pub angular_velocity: Position3D,
    /// Pose confidence (0.0 - 1.0)
    pub confidence: f32,
}

/// Tracking quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingQuality {
    /// Overall tracking quality (0.0 - 1.0)
    pub overall_quality: f32,
    /// Position tracking quality
    pub position_quality: f32,
    /// Orientation tracking quality
    pub orientation_quality: f32,
    /// Number of tracking features
    pub feature_count: u32,
    /// Tracking state
    pub state: TrackingState,
}

/// Tracking state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TrackingState {
    /// Not tracking
    NotTracking,
    /// Limited tracking (3DOF)
    Limited,
    /// Full tracking (6DOF)
    Full,
    /// Tracking lost
    Lost,
}

/// Platform capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCapabilities {
    /// Supports 6DOF head tracking
    pub head_tracking_6dof: bool,
    /// Supports hand tracking
    pub hand_tracking: bool,
    /// Supports eye tracking
    pub eye_tracking: bool,
    /// Supports controller tracking
    pub controller_tracking: bool,
    /// Supports room scale tracking
    pub room_scale: bool,
    /// Supports passthrough/AR
    pub passthrough: bool,
    /// Supported refresh rates
    pub refresh_rates: Vec<f32>,
    /// Maximum tracking range
    pub tracking_range: f32,
}

/// Tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingConfig {
    /// Enable prediction for latency compensation
    pub enable_prediction: bool,
    /// Prediction time (milliseconds)
    pub prediction_time_ms: f32,
    /// Smoothing factor for position (0.0 - 1.0)
    pub position_smoothing: f32,
    /// Smoothing factor for orientation (0.0 - 1.0)
    pub orientation_smoothing: f32,
    /// Enable hand tracking
    pub enable_hand_tracking: bool,
    /// Enable eye tracking
    pub enable_eye_tracking: bool,
    /// Target refresh rate
    pub target_refresh_rate: f32,
}

/// Device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Device name
    pub name: String,
    /// Manufacturer
    pub manufacturer: String,
    /// Model number
    pub model: String,
    /// Serial number
    pub serial_number: String,
    /// Firmware version
    pub firmware_version: String,
    /// Platform version/SDK version
    pub platform_version: String,
}

/// Hand tracking data
#[derive(Debug, Clone)]
pub struct HandTrackingData {
    /// Left hand data
    pub left_hand: Option<HandData>,
    /// Right hand data
    pub right_hand: Option<HandData>,
    /// Timestamp
    pub timestamp: Instant,
}

/// Individual hand data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandData {
    /// Joint positions (25 joints)
    pub joints: Vec<Position3D>,
    /// Joint orientations (quaternions)
    pub orientations: Vec<(f32, f32, f32, f32)>,
    /// Tracking confidence per joint
    pub confidences: Vec<f32>,
    /// Hand gesture recognition
    pub gesture: HandGesture,
    /// Is hand tracked
    pub is_tracked: bool,
}

/// Hand gesture enumeration
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum HandGesture {
    /// Unknown gesture
    Unknown,
    /// Open hand
    Open,
    /// Closed fist
    Fist,
    /// Pointing gesture
    Point,
    /// Pinch gesture
    Pinch,
    /// Thumb up
    ThumbUp,
    /// Peace sign
    Peace,
    /// OK gesture
    Ok,
}

/// Eye tracking data
#[derive(Debug, Clone)]
pub struct EyeTrackingData {
    /// Left eye data
    pub left_eye: EyeData,
    /// Right eye data
    pub right_eye: EyeData,
    /// Combined gaze direction
    pub combined_gaze: Position3D,
    /// Pupil dilation
    pub pupil_dilation: f32,
    /// Eye tracking confidence
    pub confidence: f32,
    /// Timestamp
    pub timestamp: Instant,
}

/// Individual eye data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EyeData {
    /// Eye position in world space
    pub position: Position3D,
    /// Gaze direction vector
    pub gaze_direction: Position3D,
    /// Eye openness (0.0 - 1.0)
    pub openness: f32,
    /// Pupil position in eye
    pub pupil_position: (f32, f32),
    /// Eye tracking confidence
    pub confidence: f32,
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self {
            enable_prediction: true,
            prediction_time_ms: 15.0, // 15ms for VR
            position_smoothing: 0.1,
            orientation_smoothing: 0.05,
            enable_hand_tracking: false,
            enable_eye_tracking: false,
            target_refresh_rate: 90.0,
        }
    }
}

impl PoseData {
    /// Create new pose data
    pub fn new(position: Position3D, orientation: (f32, f32, f32, f32)) -> Self {
        Self {
            position,
            orientation,
            linear_velocity: Position3D::new(0.0, 0.0, 0.0),
            angular_velocity: Position3D::new(0.0, 0.0, 0.0),
            confidence: 1.0,
        }
    }

    /// Get orientation as euler angles (yaw, pitch, roll)
    pub fn get_euler_angles(&self) -> (f32, f32, f32) {
        let (x, y, z, w) = self.orientation;

        // Roll (x-axis rotation)
        let sinr_cosp = 2.0 * (w * x + y * z);
        let cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
        let roll = sinr_cosp.atan2(cosr_cosp);

        // Pitch (y-axis rotation)
        let sinp = 2.0 * (w * y - z * x);
        let pitch = if sinp.abs() >= 1.0 {
            std::f32::consts::PI / 2.0 * sinp.signum()
        } else {
            sinp.asin()
        };

        // Yaw (z-axis rotation)
        let siny_cosp = 2.0 * (w * z + x * y);
        let cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        let yaw = siny_cosp.atan2(cosy_cosp);

        (yaw, pitch, roll)
    }
}

/// Platform factory for creating platform integrations
pub struct PlatformFactory;

impl PlatformFactory {
    /// Create platform integration based on platform type
    pub fn create_platform(platform_type: PlatformType) -> Result<Box<dyn PlatformIntegration>> {
        match platform_type {
            PlatformType::Oculus => Ok(Box::new(OculusPlatform::new())),
            #[cfg(feature = "steamvr")]
            PlatformType::SteamVR => Ok(Box::new(SteamVRPlatform::new())),
            #[cfg(not(feature = "steamvr"))]
            PlatformType::SteamVR => Err(Error::LegacyConfig(
                "SteamVR support not compiled in".to_string(),
            )),
            PlatformType::ARKit => Ok(Box::new(ARKitPlatform::new())),
            PlatformType::ARCore => Ok(Box::new(ARCorePlatform::new())),
            PlatformType::WMR => Ok(Box::new(WMRPlatform::new())),
            PlatformType::Generic => Ok(Box::new(GenericPlatform::new())),
            PlatformType::Custom => Err(Error::LegacyConfig(
                "Custom platform requires specific implementation".to_string(),
            )),
        }
    }

    /// Auto-detect available platforms
    pub async fn detect_platforms() -> Vec<PlatformType> {
        let mut available = Vec::new();

        // Check each platform type
        let platforms = [
            PlatformType::Oculus,
            #[cfg(feature = "steamvr")]
            PlatformType::SteamVR,
            PlatformType::ARKit,
            PlatformType::ARCore,
            PlatformType::WMR,
        ];

        for platform_type in platforms {
            if let Ok(mut platform) = Self::create_platform(platform_type) {
                if platform.is_available().await {
                    available.push(platform_type);
                }
            }
        }

        // Generic platform is always available as fallback
        available.push(PlatformType::Generic);

        available
    }

    /// Create the best available platform
    pub async fn create_best_available() -> Result<Box<dyn PlatformIntegration>> {
        let available_platforms = Self::detect_platforms().await;
        
        // Priority order: SteamVR > Oculus > WMR > ARKit > ARCore > Generic
        let priority_order = [
            #[cfg(feature = "steamvr")]
            PlatformType::SteamVR,
            PlatformType::Oculus,
            PlatformType::WMR,
            PlatformType::ARKit,
            PlatformType::ARCore,
            PlatformType::Generic,
        ];

        for platform_type in priority_order {
            if available_platforms.contains(&platform_type) {
                return Self::create_platform(platform_type);
            }
        }

        // Fallback to generic platform
        Self::create_platform(PlatformType::Generic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_platform_factory() {
        let platform = PlatformFactory::create_platform(PlatformType::Generic).unwrap();
        let capabilities = platform.get_capabilities();
        assert!(capabilities.head_tracking_6dof);
    }

    #[tokio::test]
    async fn test_platform_detection() {
        let platforms = PlatformFactory::detect_platforms().await;
        // Generic platform should always be available
        assert!(platforms.contains(&PlatformType::Generic));
    }

    #[tokio::test]
    async fn test_best_available_platform() {
        let platform = PlatformFactory::create_best_available().await;
        assert!(platform.is_ok());
    }

    #[tokio::test]
    async fn test_tracking_config_default() {
        let config = TrackingConfig::default();
        assert!(config.enable_prediction);
        assert_eq!(config.prediction_time_ms, 15.0);
        assert_eq!(config.target_refresh_rate, 90.0);
    }

    #[tokio::test]
    async fn test_pose_data() {
        let pose = PoseData::new(Position3D::new(1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0));

        let euler = pose.get_euler_angles();
        assert_eq!(euler, (0.0, 0.0, 0.0)); // Identity quaternion

        assert_eq!(pose.position.x, 1.0);
        assert_eq!(pose.position.y, 2.0);
        assert_eq!(pose.position.z, 3.0);
    }

    #[tokio::test]
    async fn test_platform_capabilities() {
        let capabilities = PlatformCapabilities {
            head_tracking_6dof: true,
            hand_tracking: false,
            eye_tracking: false,
            controller_tracking: true,
            room_scale: true,
            passthrough: false,
            refresh_rates: vec![90.0, 120.0],
            tracking_range: 10.0,
        };

        assert!(capabilities.head_tracking_6dof);
        assert!(capabilities.controller_tracking);
        assert!(capabilities.room_scale);
        assert!(!capabilities.hand_tracking);
        assert!(!capabilities.eye_tracking);
        assert!(!capabilities.passthrough);
    }
}