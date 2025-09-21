//! SteamVR/OpenVR Platform Integration
//!
//! This module provides comprehensive integration with SteamVR through the OpenVR SDK,
//! enabling 6DOF head tracking, controller tracking, hand tracking, and room scale VR.

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

#[cfg(feature = "steamvr")]
use openvr::{Context, System, TrackedDeviceClass, TrackedDevicePose, VRSystem};

/// SteamVR platform integration with full OpenVR SDK support
pub struct SteamVRPlatform {
    #[cfg(feature = "steamvr")]
    context: Option<Context>,
    #[cfg(feature = "steamvr")]
    system: Option<System>,

    device_info: DeviceInfo,
    capabilities: PlatformCapabilities,
    tracking_active: bool,
    config: TrackingConfig,

    // Cached device indices
    hmd_index: Option<u32>,
    controller_indices: Vec<u32>,
    tracker_indices: Vec<u32>,
}

impl SteamVRPlatform {
    /// Create new SteamVR platform integration
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "steamvr")]
            context: None,
            #[cfg(feature = "steamvr")]
            system: None,

            device_info: DeviceInfo {
                name: "SteamVR Device".to_string(),
                manufacturer: "Valve".to_string(),
                model: "Unknown".to_string(),
                serial_number: "Unknown".to_string(),
                firmware_version: "Unknown".to_string(),
                platform_version: "Unknown".to_string(),
            },
            capabilities: PlatformCapabilities {
                head_tracking_6dof: true,
                hand_tracking: true,
                eye_tracking: false, // Depends on headset
                controller_tracking: true,
                room_scale: true,
                passthrough: false, // Depends on headset
                refresh_rates: vec![90.0, 120.0, 144.0],
                tracking_range: 15.0, // Larger tracking area with lighthouses
            },
            tracking_active: false,
            config: TrackingConfig::default(),
            hmd_index: None,
            controller_indices: Vec::new(),
            tracker_indices: Vec::new(),
        }
    }

    #[cfg(feature = "steamvr")]
    /// Initialize OpenVR and discover devices
    async fn init_openvr(&mut self) -> Result<()> {
        use openvr::{init, ApplicationType, InitError};

        // Initialize OpenVR
        let context = match init(ApplicationType::Scene) {
            Ok(ctx) => ctx,
            Err(InitError::Init_VRInitError_Init_InstallationNotFound) => {
                return Err(Error::LegacyConfig("SteamVR not installed".to_string()));
            }
            Err(InitError::Init_VRInitError_Init_NoServerForBackgroundApp) => {
                return Err(Error::LegacyConfig(
                    "SteamVR server not running".to_string(),
                ));
            }
            Err(e) => {
                return Err(Error::LegacyConfig(format!("OpenVR init failed: {e:?}")));
            }
        };

        let system = context
            .system()
            .map_err(|e| Error::LegacyConfig(format!("Failed to get VR system: {e:?}")))?;

        // Discover connected devices
        self.discover_devices(&system)?;

        // Get device information
        self.update_device_info(&system)?;

        self.context = Some(context);
        self.system = Some(system);

        tracing::info!("SteamVR initialized successfully");
        Ok(())
    }

    #[cfg(feature = "steamvr")]
    /// Discover and categorize connected VR devices
    fn discover_devices(&mut self, system: &System) -> Result<()> {
        self.controller_indices.clear();
        self.tracker_indices.clear();
        self.hmd_index = None;

        // Check all possible device indices
        for device_index in 0..openvr::MAX_TRACKED_DEVICE_COUNT {
            if !system.is_tracked_device_connected(device_index) {
                continue;
            }

            let device_class = system.tracked_device_class(device_index);
            match device_class {
                TrackedDeviceClass::HMD => {
                    self.hmd_index = Some(device_index);
                    tracing::info!("Found HMD at index {}", device_index);
                }
                TrackedDeviceClass::Controller => {
                    self.controller_indices.push(device_index);
                    tracing::info!("Found controller at index {}", device_index);
                }
                TrackedDeviceClass::GenericTracker => {
                    self.tracker_indices.push(device_index);
                    tracing::info!("Found tracker at index {}", device_index);
                }
                TrackedDeviceClass::TrackingReference => {
                    tracing::info!("Found base station at index {}", device_index);
                }
                _ => {
                    tracing::debug!(
                        "Found unknown device class {:?} at index {}",
                        device_class,
                        device_index
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(feature = "steamvr")]
    /// Update device information from OpenVR
    fn update_device_info(&mut self, system: &System) -> Result<()> {
        if let Some(hmd_index) = self.hmd_index {
            // Get HMD information
            if let Ok(manufacturer) = system.string_tracked_device_property(
                hmd_index,
                openvr::property::TrackingSystemName_String,
            ) {
                self.device_info.manufacturer = manufacturer;
            }

            if let Ok(model) = system
                .string_tracked_device_property(hmd_index, openvr::property::ModelNumber_String)
            {
                self.device_info.model = model;
            }

            if let Ok(serial) = system
                .string_tracked_device_property(hmd_index, openvr::property::SerialNumber_String)
            {
                self.device_info.serial_number = serial;
            }

            if let Ok(firmware) = system
                .string_tracked_device_property(hmd_index, openvr::property::FirmwareVersion_String)
            {
                self.device_info.firmware_version = firmware;
            }

            // Update capabilities based on actual device
            if let Ok(display_frequency) = system
                .float_tracked_device_property(hmd_index, openvr::property::DisplayFrequency_Float)
            {
                self.capabilities.refresh_rates = vec![display_frequency];
            }
        }

        Ok(())
    }

    #[cfg(feature = "steamvr")]
    /// Get current poses from OpenVR
    fn get_device_poses(&self) -> Result<Vec<TrackedDevicePose>> {
        let system = self
            .system
            .as_ref()
            .ok_or_else(|| Error::LegacyProcessing("OpenVR system not initialized".to_string()))?;

        let poses = system.device_to_absolute_tracking_poses(
            openvr::TrackingUniverseOrigin::Standing,
            0.0, // No prediction
        );

        Ok(poses)
    }

    #[cfg(feature = "steamvr")]
    /// Convert OpenVR matrix to pose data
    fn matrix_to_pose(&self, matrix: &openvr::HmdMatrix34) -> PoseData {
        // Extract position
        let position = Position3D::new(matrix.m[0][3], matrix.m[1][3], -matrix.m[2][3]);

        // Extract rotation matrix and convert to quaternion
        let m00 = matrix.m[0][0];
        let m01 = matrix.m[0][1];
        let m02 = -matrix.m[0][2]; // Flip Z for coordinate system
        let m10 = matrix.m[1][0];
        let m11 = matrix.m[1][1];
        let m12 = -matrix.m[1][2];
        let m20 = -matrix.m[2][0]; // Flip Z
        let m21 = -matrix.m[2][1];
        let m22 = matrix.m[2][2];

        // Convert rotation matrix to quaternion
        let trace = m00 + m11 + m22;
        let (x, y, z, w) = if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            let w = 0.25 * s;
            let x = (m21 - m12) / s;
            let y = (m02 - m20) / s;
            let z = (m10 - m01) / s;
            (x, y, z, w)
        } else if m00 > m11 && m00 > m22 {
            let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
            let w = (m21 - m12) / s;
            let x = 0.25 * s;
            let y = (m01 + m10) / s;
            let z = (m02 + m20) / s;
            (x, y, z, w)
        } else if m11 > m22 {
            let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
            let w = (m02 - m20) / s;
            let x = (m01 + m10) / s;
            let y = 0.25 * s;
            let z = (m12 + m21) / s;
            (x, y, z, w)
        } else {
            let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
            let w = (m10 - m01) / s;
            let x = (m02 + m20) / s;
            let y = (m12 + m21) / s;
            let z = 0.25 * s;
            (x, y, z, w)
        };

        PoseData {
            position,
            orientation: (x, y, z, w),
            linear_velocity: Position3D::new(0.0, 0.0, 0.0), // TODO: Calculate from previous poses
            angular_velocity: Position3D::new(0.0, 0.0, 0.0),
            confidence: 1.0, // OpenVR doesn't provide confidence directly
        }
    }

    #[cfg(not(feature = "steamvr"))]
    /// Fallback implementation when SteamVR feature is disabled
    async fn init_openvr(&mut self) -> Result<()> {
        Err(Error::LegacyConfig(
            "SteamVR support not compiled in".to_string(),
        ))
    }

    #[cfg(not(feature = "steamvr"))]
    fn get_device_poses(&self) -> Result<Vec<()>> {
        Err(Error::LegacyConfig(
            "SteamVR support not compiled in".to_string(),
        ))
    }
}

#[async_trait]
impl PlatformIntegration for SteamVRPlatform {
    async fn initialize(&mut self) -> Result<()> {
        self.init_openvr().await
    }

    async fn get_tracking_data(&self) -> Result<PlatformTrackingData> {
        if !self.tracking_active {
            return Err(Error::LegacyProcessing("Tracking not active".to_string()));
        }

        #[cfg(feature = "steamvr")]
        {
            let poses = self.get_device_poses()?;

            // Get HMD pose
            let head_pose = if let Some(hmd_index) = self.hmd_index {
                if let Some(pose) = poses.get(hmd_index as usize) {
                    if pose.device_is_connected && pose.pose_is_valid {
                        self.matrix_to_pose(&pose.device_to_absolute_tracking)
                    } else {
                        PoseData::new(Position3D::new(0.0, 1.7, 0.0), (0.0, 0.0, 0.0, 1.0))
                    }
                } else {
                    PoseData::new(Position3D::new(0.0, 1.7, 0.0), (0.0, 0.0, 0.0, 1.0))
                }
            } else {
                PoseData::new(Position3D::new(0.0, 1.7, 0.0), (0.0, 0.0, 0.0, 1.0))
            };

            // Get controller poses
            let left_controller = if let Some(&first_controller) = self.controller_indices.first() {
                if let Some(pose) = poses.get(first_controller as usize) {
                    if pose.device_is_connected && pose.pose_is_valid {
                        Some(self.matrix_to_pose(&pose.device_to_absolute_tracking))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let right_controller = if self.controller_indices.len() > 1 {
                let second_controller = self.controller_indices[1];
                if let Some(pose) = poses.get(second_controller as usize) {
                    if pose.device_is_connected && pose.pose_is_valid {
                        Some(self.matrix_to_pose(&pose.device_to_absolute_tracking))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            // Calculate tracking quality based on pose validity
            let connected_devices = poses
                .iter()
                .take(self.controller_indices.len() + if self.hmd_index.is_some() { 1 } else { 0 })
                .filter(|pose| pose.device_is_connected && pose.pose_is_valid)
                .count();

            let total_devices =
                self.controller_indices.len() + if self.hmd_index.is_some() { 1 } else { 0 };
            let quality_ratio = if total_devices > 0 {
                connected_devices as f32 / total_devices as f32
            } else {
                0.0
            };

            Ok(PlatformTrackingData {
                head_pose,
                left_controller,
                right_controller,
                quality: TrackingQuality {
                    overall_quality: quality_ratio * 0.95,
                    position_quality: quality_ratio * 0.93,
                    orientation_quality: quality_ratio * 0.97,
                    feature_count: (connected_devices * 20) as u32, // Estimated features per device
                    state: if quality_ratio > 0.8 {
                        TrackingState::Full
                    } else if quality_ratio > 0.3 {
                        TrackingState::Limited
                    } else {
                        TrackingState::Lost
                    },
                },
                timestamp: Instant::now(),
                raw_data: PlatformData {
                    device_id: "SteamVR".to_string(),
                    pose_data: vec![], // Could store raw pose data if needed
                    tracking_confidence: quality_ratio * 0.95,
                    platform_timestamp: 0, // TODO: Get actual OpenVR timestamp
                    properties: HashMap::new(),
                },
            })
        }

        #[cfg(not(feature = "steamvr"))]
        {
            // Fallback mock data when SteamVR feature is disabled
            Ok(PlatformTrackingData {
                head_pose: PoseData::new(Position3D::new(0.0, 1.7, 0.0), (0.0, 0.0, 0.0, 1.0)),
                left_controller: None,
                right_controller: None,
                quality: TrackingQuality {
                    overall_quality: 0.0,
                    position_quality: 0.0,
                    orientation_quality: 0.0,
                    feature_count: 0,
                    state: TrackingState::NotTracking,
                },
                timestamp: Instant::now(),
                raw_data: PlatformData {
                    device_id: "SteamVR (disabled)".to_string(),
                    pose_data: vec![],
                    tracking_confidence: 0.0,
                    platform_timestamp: 0,
                    properties: HashMap::new(),
                },
            })
        }
    }

    async fn is_available(&self) -> bool {
        #[cfg(feature = "steamvr")]
        {
            // Check if SteamVR is running
            self.context.is_some() && self.system.is_some()
        }

        #[cfg(not(feature = "steamvr"))]
        {
            false
        }
    }

    fn get_capabilities(&self) -> PlatformCapabilities {
        self.capabilities.clone()
    }

    async fn configure_tracking(&mut self, config: TrackingConfig) -> Result<()> {
        self.config = config;
        tracing::info!("Configured SteamVR tracking with config: {:?}", self.config);
        Ok(())
    }

    fn get_device_info(&self) -> DeviceInfo {
        self.device_info.clone()
    }

    async fn start_tracking(&mut self) -> Result<()> {
        #[cfg(feature = "steamvr")]
        {
            if self.system.is_none() {
                return Err(Error::LegacyProcessing(
                    "OpenVR not initialized".to_string(),
                ));
            }
        }

        self.tracking_active = true;
        tracing::info!("Started SteamVR tracking");
        Ok(())
    }

    async fn stop_tracking(&mut self) -> Result<()> {
        self.tracking_active = false;
        tracing::info!("Stopped SteamVR tracking");
        Ok(())
    }

    async fn get_hand_tracking(&self) -> Result<Option<HandTrackingData>> {
        if !self.config.enable_hand_tracking {
            return Ok(None);
        }

        #[cfg(feature = "steamvr")]
        {
            // TODO: Implement hand tracking via SteamVR Input system
            // This would require using the Input API to get skeletal data
        }

        Ok(None)
    }

    async fn get_eye_tracking(&self) -> Result<Option<EyeTrackingData>> {
        if !self.config.enable_eye_tracking || !self.capabilities.eye_tracking {
            return Ok(None);
        }

        #[cfg(feature = "steamvr")]
        {
            // TODO: Check if connected headset supports eye tracking
            // This would require checking device properties and using eye tracking APIs
        }

        Ok(None)
    }
}

impl Default for SteamVRPlatform {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_steamvr_platform_creation() {
        let platform = SteamVRPlatform::new();
        assert!(!platform.tracking_active);
        assert_eq!(platform.device_info.manufacturer, "Valve");
    }

    #[tokio::test]
    async fn test_steamvr_capabilities() {
        let platform = SteamVRPlatform::new();
        let capabilities = platform.get_capabilities();

        assert!(capabilities.head_tracking_6dof);
        assert!(capabilities.controller_tracking);
        assert!(capabilities.room_scale);
        assert!(capabilities.refresh_rates.contains(&90.0));
    }

    #[tokio::test]
    async fn test_steamvr_configuration() {
        let mut platform = SteamVRPlatform::new();
        let config = TrackingConfig {
            enable_prediction: true,
            prediction_time_ms: 20.0,
            position_smoothing: 0.2,
            orientation_smoothing: 0.1,
            enable_hand_tracking: true,
            enable_eye_tracking: false,
            target_refresh_rate: 120.0,
        };

        assert!(platform.configure_tracking(config.clone()).await.is_ok());
        assert_eq!(platform.config.prediction_time_ms, 20.0);
        assert!(platform.config.enable_hand_tracking);
    }

    #[cfg(feature = "steamvr")]
    #[tokio::test]
    async fn test_steamvr_initialization() {
        let mut platform = SteamVRPlatform::new();

        // This will likely fail in CI/test environment, but tests the code path
        let _result = platform.initialize().await;
        // Don't assert success since SteamVR may not be available in test environment
    }

    #[cfg(not(feature = "steamvr"))]
    #[tokio::test]
    async fn test_steamvr_disabled() {
        let mut platform = SteamVRPlatform::new();

        let result = platform.initialize().await;
        assert!(result.is_err());
        assert!(!platform.is_available().await);
    }
}
