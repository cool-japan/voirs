//! Spatial audio effect implementation for 3D positioning.

use crate::{
    audio::AudioBuffer,
    error::Result,
    plugins::{AudioEffect, ParameterDefinition, ParameterType, ParameterValue, VoirsPlugin},
    VoirsError,
};
use async_trait::async_trait;
use std::{collections::HashMap, sync::RwLock};

/// Spatial audio effect plugin for 3D positioning
pub struct SpatialAudioEffect {
    /// Azimuth angle in degrees (-180 to 180)
    pub azimuth: RwLock<f32>,

    /// Elevation angle in degrees (-90 to 90)
    pub elevation: RwLock<f32>,

    /// Distance in meters
    pub distance: RwLock<f32>,

    /// Room size parameter (0.0 - 1.0)
    pub room_size: RwLock<f32>,
}

impl SpatialAudioEffect {
    pub fn new() -> Self {
        Self {
            azimuth: RwLock::new(0.0),
            elevation: RwLock::new(0.0),
            distance: RwLock::new(1.0),
            room_size: RwLock::new(0.5),
        }
    }

    fn calculate_stereo_pan(&self) -> (f32, f32) {
        let azimuth = *self.azimuth.read().unwrap_or_else(|e| e.into_inner());
        let distance = *self.distance.read().unwrap_or_else(|e| e.into_inner());

        // Simple stereo panning based on azimuth
        let angle_rad = azimuth.to_radians();
        let left =
            ((angle_rad + std::f32::consts::PI / 2.0) / std::f32::consts::PI).clamp(0.0, 1.0);
        let right = 1.0 - left;

        // Apply distance attenuation
        let attenuation = 1.0 / (1.0 + distance);

        (left * attenuation, right * attenuation)
    }
}

impl Default for SpatialAudioEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl VoirsPlugin for SpatialAudioEffect {
    fn name(&self) -> &str {
        "Spatial Audio"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "3D spatial audio positioning effect"
    }

    fn author(&self) -> &str {
        "VoiRS Team"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait]
impl AudioEffect for SpatialAudioEffect {
    async fn process_audio(&self, audio: &AudioBuffer) -> Result<AudioBuffer> {
        let (left_gain, right_gain) = self.calculate_stereo_pan();

        let mut processed = audio.clone();
        let samples = processed.samples_mut();

        // For simplicity, apply equal panning to all samples
        // In a real implementation, this would use HRTF filters
        for sample in samples.iter_mut() {
            let input = *sample;
            // Blend left and right gains
            *sample = input * ((left_gain + right_gain) / 2.0);
        }

        Ok(processed)
    }

    fn get_parameters(&self) -> HashMap<String, ParameterValue> {
        let mut params = HashMap::new();
        params.insert(
            "azimuth".to_string(),
            ParameterValue::Float(*self.azimuth.read().unwrap_or_else(|e| e.into_inner())),
        );
        params.insert(
            "elevation".to_string(),
            ParameterValue::Float(*self.elevation.read().unwrap_or_else(|e| e.into_inner())),
        );
        params.insert(
            "distance".to_string(),
            ParameterValue::Float(*self.distance.read().unwrap_or_else(|e| e.into_inner())),
        );
        params.insert(
            "room_size".to_string(),
            ParameterValue::Float(*self.room_size.read().unwrap_or_else(|e| e.into_inner())),
        );
        params
    }

    fn set_parameter(&self, name: &str, value: ParameterValue) -> Result<()> {
        match name {
            "azimuth" => {
                if let Some(v) = value.as_f32() {
                    *self.azimuth.write().unwrap_or_else(|e| e.into_inner()) =
                        v.clamp(-180.0, 180.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid azimuth parameter type",
                    ))
                }
            }
            "elevation" => {
                if let Some(v) = value.as_f32() {
                    *self.elevation.write().unwrap_or_else(|e| e.into_inner()) =
                        v.clamp(-90.0, 90.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid elevation parameter type",
                    ))
                }
            }
            "distance" => {
                if let Some(v) = value.as_f32() {
                    *self.distance.write().unwrap_or_else(|e| e.into_inner()) =
                        v.clamp(0.1, 100.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid distance parameter type",
                    ))
                }
            }
            "room_size" => {
                if let Some(v) = value.as_f32() {
                    *self.room_size.write().unwrap_or_else(|e| e.into_inner()) =
                        v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid room_size parameter type",
                    ))
                }
            }
            _ => Err(VoirsError::internal(
                "plugins",
                format!("Unknown parameter: {name}"),
            )),
        }
    }

    fn get_parameter_definition(&self, name: &str) -> Option<ParameterDefinition> {
        match name {
            "azimuth" => Some(ParameterDefinition {
                name: "azimuth".to_string(),
                description: "Horizontal angle in degrees".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.0),
                min_value: Some(ParameterValue::Float(-180.0)),
                max_value: Some(ParameterValue::Float(180.0)),
                step_size: Some(1.0),
                realtime_safe: true,
            }),
            "elevation" => Some(ParameterDefinition {
                name: "elevation".to_string(),
                description: "Vertical angle in degrees".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.0),
                min_value: Some(ParameterValue::Float(-90.0)),
                max_value: Some(ParameterValue::Float(90.0)),
                step_size: Some(1.0),
                realtime_safe: true,
            }),
            "distance" => Some(ParameterDefinition {
                name: "distance".to_string(),
                description: "Distance in meters".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(1.0),
                min_value: Some(ParameterValue::Float(0.1)),
                max_value: Some(ParameterValue::Float(100.0)),
                step_size: Some(0.1),
                realtime_safe: true,
            }),
            "room_size" => Some(ParameterDefinition {
                name: "room_size".to_string(),
                description: "Virtual room size".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.5),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(1.0)),
                step_size: Some(0.01),
                realtime_safe: false,
            }),
            _ => None,
        }
    }
}
