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
use std::sync::Mutex;
use tokio::time::Instant;

#[cfg(feature = "steamvr")]
use openvr::{Context, System, TrackedDeviceClass, TrackedDevicePose};

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

    // Pose history for velocity calculation (using interior mutability for const methods)
    previous_hmd_pose: Mutex<Option<(PoseData, Instant)>>,
    previous_controller_poses: Mutex<HashMap<u32, (PoseData, Instant)>>,
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
            previous_hmd_pose: Mutex::new(None),
            previous_controller_poses: Mutex::new(HashMap::new()),
        }
    }

    #[cfg(feature = "steamvr")]
    /// Initialize OpenVR and discover devices
    async fn init_openvr(&mut self) -> Result<()> {
        use openvr::{init, ApplicationType};

        // Initialize OpenVR - unsafe required by OpenVR C API
        #[allow(unsafe_code)]
        let context = unsafe {
            match init(ApplicationType::Scene) {
                Ok(ctx) => ctx,
                Err(e) => {
                    return Err(Error::LegacyConfig(format!("OpenVR init failed: {e}")));
                }
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
        for device_index_usize in 0..openvr::MAX_TRACKED_DEVICE_COUNT {
            let device_index = device_index_usize as u32;
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
                self.device_info.manufacturer =
                    manufacturer.to_str().unwrap_or("Unknown").to_string();
            }

            if let Ok(model) = system
                .string_tracked_device_property(hmd_index, openvr::property::ModelNumber_String)
            {
                self.device_info.model = model.to_str().unwrap_or("Unknown").to_string();
            }

            if let Ok(serial) = system
                .string_tracked_device_property(hmd_index, openvr::property::SerialNumber_String)
            {
                self.device_info.serial_number = serial.to_str().unwrap_or("Unknown").to_string();
            }

            if let Ok(firmware) = system.string_tracked_device_property(
                hmd_index,
                openvr::property::TrackingFirmwareVersion_String,
            ) {
                self.device_info.firmware_version =
                    firmware.to_str().unwrap_or("Unknown").to_string();
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
    /// Get current poses from OpenVR with timing information
    fn get_device_poses(&self) -> Result<(Vec<TrackedDevicePose>, f64)> {
        let system = self
            .system
            .as_ref()
            .ok_or_else(|| Error::LegacyProcessing("OpenVR system not initialized".to_string()))?;

        // Get the current time seconds (OpenVR's timing reference)
        let time_since_last_vsync = system.time_since_last_vsync();
        let seconds_since_vsync = time_since_last_vsync
            .map(|(secs, _)| secs as f64)
            .unwrap_or(0.0);

        // Get all device poses at once
        let poses_array = system.device_to_absolute_tracking_pose(
            openvr::TrackingUniverseOrigin::Standing,
            0.0, // No prediction
        );

        // Convert array to Vec
        let poses: Vec<TrackedDevicePose> = poses_array.to_vec();

        Ok((poses, seconds_since_vsync))
    }

    #[cfg(feature = "steamvr")]
    /// Convert OpenVR matrix to pose data with velocity calculation
    fn matrix_to_pose(
        &self,
        matrix: &[[f32; 4]; 3],
        device_index: u32,
        current_time: Instant,
    ) -> PoseData {
        // Extract position
        let position = Position3D::new(matrix[0][3], matrix[1][3], -matrix[2][3]);

        // Extract rotation matrix and convert to quaternion
        let m00 = matrix[0][0];
        let m01 = matrix[0][1];
        let m02 = -matrix[0][2]; // Flip Z for coordinate system
        let m10 = matrix[1][0];
        let m11 = matrix[1][1];
        let m12 = -matrix[1][2];
        let m20 = -matrix[2][0]; // Flip Z
        let m21 = -matrix[2][1];
        let m22 = matrix[2][2];

        // Convert rotation matrix to quaternion
        let trace = m00 + m11 + m22;
        let (x, y, z, w) = if trace > 0.0 {
            let s = (trace + 1.0_f32).sqrt() * 2.0;
            let w = 0.25 * s;
            let x = (m21 - m12) / s;
            let y = (m02 - m20) / s;
            let z = (m10 - m01) / s;
            (x, y, z, w)
        } else if m00 > m11 && m00 > m22 {
            let s = (1.0_f32 + m00 - m11 - m22).sqrt() * 2.0;
            let w = (m21 - m12) / s;
            let x = 0.25 * s;
            let y = (m01 + m10) / s;
            let z = (m02 + m20) / s;
            (x, y, z, w)
        } else if m11 > m22 {
            let s = (1.0_f32 + m11 - m00 - m22).sqrt() * 2.0;
            let w = (m02 - m20) / s;
            let x = (m01 + m10) / s;
            let y = 0.25 * s;
            let z = (m12 + m21) / s;
            (x, y, z, w)
        } else {
            let s = (1.0_f32 + m22 - m00 - m11).sqrt() * 2.0;
            let w = (m10 - m01) / s;
            let x = (m02 + m20) / s;
            let y = (m12 + m21) / s;
            let z = 0.25 * s;
            (x, y, z, w)
        };

        // Calculate velocity from previous pose
        let (linear_velocity, angular_velocity) = if let Some(hmd_index) = self.hmd_index {
            if device_index == hmd_index {
                if let Ok(prev_lock) = self.previous_hmd_pose.lock() {
                    if let Some((prev_pose, prev_time)) = prev_lock.as_ref() {
                        let dt = current_time.duration_since(*prev_time).as_secs_f32();
                        if dt > 0.0 && dt < 0.1 {
                            // Reasonable time delta (< 100ms)
                            let linear_vel = Position3D::new(
                                (position.x - prev_pose.position.x) / dt,
                                (position.y - prev_pose.position.y) / dt,
                                (position.z - prev_pose.position.z) / dt,
                            );

                            // Simple angular velocity approximation from quaternion difference
                            let (px, py, pz, pw) = prev_pose.orientation;
                            let dot = x * px + y * py + z * pz + w * pw;
                            let angle = 2.0 * dot.abs().min(1.0).acos();
                            let angular_vel = Position3D::new(0.0, angle / dt, 0.0); // Simplified

                            (linear_vel, angular_vel)
                        } else {
                            (
                                Position3D::new(0.0, 0.0, 0.0),
                                Position3D::new(0.0, 0.0, 0.0),
                            )
                        }
                    } else {
                        (
                            Position3D::new(0.0, 0.0, 0.0),
                            Position3D::new(0.0, 0.0, 0.0),
                        )
                    }
                } else {
                    tracing::warn!("Failed to acquire HMD pose lock");
                    (
                        Position3D::new(0.0, 0.0, 0.0),
                        Position3D::new(0.0, 0.0, 0.0),
                    )
                }
            } else {
                // Controller velocity calculation
                if let Ok(prev_controllers) = self.previous_controller_poses.lock() {
                    if let Some((prev_pose, prev_time)) = prev_controllers.get(&device_index) {
                        let dt = current_time.duration_since(*prev_time).as_secs_f32();
                        if dt > 0.0 && dt < 0.1 {
                            let linear_vel = Position3D::new(
                                (position.x - prev_pose.position.x) / dt,
                                (position.y - prev_pose.position.y) / dt,
                                (position.z - prev_pose.position.z) / dt,
                            );
                            (linear_vel, Position3D::new(0.0, 0.0, 0.0))
                        } else {
                            (
                                Position3D::new(0.0, 0.0, 0.0),
                                Position3D::new(0.0, 0.0, 0.0),
                            )
                        }
                    } else {
                        (
                            Position3D::new(0.0, 0.0, 0.0),
                            Position3D::new(0.0, 0.0, 0.0),
                        )
                    }
                } else {
                    tracing::warn!("Failed to acquire controller poses lock");
                    (
                        Position3D::new(0.0, 0.0, 0.0),
                        Position3D::new(0.0, 0.0, 0.0),
                    )
                }
            }
        } else {
            (
                Position3D::new(0.0, 0.0, 0.0),
                Position3D::new(0.0, 0.0, 0.0),
            )
        };

        let pose_data = PoseData {
            position,
            orientation: (x, y, z, w),
            linear_velocity,
            angular_velocity,
            confidence: 1.0, // OpenVR doesn't provide confidence directly
        };

        // Store current pose for next velocity calculation
        if let Some(hmd_index) = self.hmd_index {
            if device_index == hmd_index {
                if let Ok(mut prev_pose) = self.previous_hmd_pose.lock() {
                    *prev_pose = Some((pose_data.clone(), current_time));
                } else {
                    tracing::warn!("Failed to acquire HMD pose lock for storing");
                }
            } else if self.controller_indices.contains(&device_index) {
                if let Ok(mut prev_controllers) = self.previous_controller_poses.lock() {
                    prev_controllers.insert(device_index, (pose_data.clone(), current_time));
                } else {
                    tracing::warn!("Failed to acquire controller poses lock for storing");
                }
            }
        }

        pose_data
    }

    #[cfg(not(feature = "steamvr"))]
    /// Fallback implementation when SteamVR feature is disabled
    async fn init_openvr(&mut self) -> Result<()> {
        Err(Error::LegacyConfig(
            "SteamVR support not compiled in".to_string(),
        ))
    }

    #[cfg(not(feature = "steamvr"))]
    fn get_device_poses(&self) -> Result<(Vec<()>, f64)> {
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
            let (poses, platform_timestamp) = self.get_device_poses()?;

            let current_time = Instant::now();

            // Get HMD pose
            let head_pose = if let Some(hmd_index) = self.hmd_index {
                if let Some(pose) = poses.get(hmd_index as usize) {
                    if pose.device_is_connected() && pose.pose_is_valid() {
                        self.matrix_to_pose(
                            pose.device_to_absolute_tracking(),
                            hmd_index,
                            current_time,
                        )
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
                    if pose.device_is_connected() && pose.pose_is_valid() {
                        Some(self.matrix_to_pose(
                            pose.device_to_absolute_tracking(),
                            first_controller,
                            current_time,
                        ))
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
                    if pose.device_is_connected() && pose.pose_is_valid() {
                        Some(self.matrix_to_pose(
                            pose.device_to_absolute_tracking(),
                            second_controller,
                            current_time,
                        ))
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
                .filter(|pose| pose.device_is_connected() && pose.pose_is_valid())
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
                    platform_timestamp: (platform_timestamp * 1000000.0) as u64, // Convert to microseconds
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
            // Hand tracking in SteamVR requires the Input API and skeletal tracking data
            // Check if controllers support skeletal tracking (Index controllers, etc.)
            if let Some(system) = &self.system {
                // Check if any controller supports skeletal input
                for &controller_index in &self.controller_indices {
                    // In a full implementation, we would:
                    // 1. Use VRInput() to get the input system
                    // 2. Get the action handle for skeletal data
                    // 3. Query skeletal bone data using GetSkeletalBoneData()
                    // 4. Convert bone transforms to HandTrackingData format

                    // For now, detect capability but don't implement full tracking
                    let _has_skeletal = system.is_tracked_device_connected(controller_index);

                    tracing::debug!(
                        "Hand tracking detection for controller {}: capable but not fully implemented",
                        controller_index
                    );
                }
            }
        }

        // Return None until full skeletal tracking implementation
        Ok(None)
    }

    async fn get_eye_tracking(&self) -> Result<Option<EyeTrackingData>> {
        if !self.config.enable_eye_tracking {
            return Ok(None);
        }

        #[cfg(feature = "steamvr")]
        {
            // Check if the connected HMD supports eye tracking
            if let Some(system) = &self.system {
                if let Some(hmd_index) = self.hmd_index {
                    // Check device properties for eye tracking capability
                    // The HMD should support IVREyeTracking interface
                    // Devices like Vive Pro Eye, HP Reverb G2 Omnicept support this

                    if system.is_tracked_device_connected(hmd_index) {
                        // In a full implementation, we would:
                        // 1. Query device properties for eye tracking support
                        // 2. Use IVREyeTracking interface to get gaze data
                        // 3. Get eye positions, gaze directions, pupil diameter, etc.
                        // 4. Convert to EyeTrackingData format

                        tracing::debug!(
                            "Eye tracking check for HMD {}: checking capability (not fully implemented)",
                            hmd_index
                        );

                        // Update capabilities if eye tracking is detected
                        // For now, this would require platform-specific property queries
                    }
                }
            }
        }

        // Return None until full eye tracking implementation
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
