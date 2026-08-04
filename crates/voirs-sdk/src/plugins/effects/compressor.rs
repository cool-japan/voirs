//! Dynamic range compressor effect implementation.

use crate::{
    audio::AudioBuffer,
    error::Result,
    plugins::{AudioEffect, ParameterDefinition, ParameterType, ParameterValue, VoirsPlugin},
    VoirsError,
};
use async_trait::async_trait;
use std::{collections::HashMap, sync::RwLock};

/// Compressor effect plugin with proper envelope following
pub struct CompressorEffect {
    /// Threshold in dB
    pub threshold: RwLock<f32>,

    /// Compression ratio
    pub ratio: RwLock<f32>,

    /// Attack time in milliseconds
    pub attack: RwLock<f32>,

    /// Release time in milliseconds
    pub release: RwLock<f32>,

    /// Makeup gain in dB
    pub makeup_gain: RwLock<f32>,

    /// Current envelope follower state
    envelope: RwLock<f32>,

    /// Sample rate for time calculations
    sample_rate: RwLock<Option<u32>>,
}

impl CompressorEffect {
    pub fn new() -> Self {
        Self {
            threshold: RwLock::new(-20.0),
            ratio: RwLock::new(4.0),
            attack: RwLock::new(5.0),
            release: RwLock::new(50.0),
            makeup_gain: RwLock::new(0.0),
            envelope: RwLock::new(0.0),
            sample_rate: RwLock::new(None),
        }
    }

    fn calculate_gain_reduction(&self, input_level_db: f32) -> f32 {
        let threshold = *self.threshold.read().unwrap_or_else(|e| e.into_inner());
        let ratio = *self.ratio.read().unwrap_or_else(|e| e.into_inner());

        if input_level_db <= threshold {
            0.0 // No compression below threshold
        } else {
            let excess_db = input_level_db - threshold;
            excess_db - (excess_db / ratio)
        }
    }
}

impl Default for CompressorEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl VoirsPlugin for CompressorEffect {
    fn name(&self) -> &str {
        "Compressor"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "Dynamic range compressor for controlling audio dynamics"
    }

    fn author(&self) -> &str {
        "VoiRS Team"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait]
impl AudioEffect for CompressorEffect {
    async fn process_audio(&self, audio: &AudioBuffer) -> Result<AudioBuffer> {
        let sample_rate = audio.sample_rate();

        // Update sample rate if changed
        let current_sample_rate = *self
            .sample_rate
            .read()
            .unwrap_or_else(|e| e.into_inner());
        if current_sample_rate != Some(sample_rate) {
            *self.sample_rate.write().unwrap_or_else(|e| e.into_inner()) = Some(sample_rate);
        }

        let mut processed = audio.clone();
        let samples = processed.samples_mut();

        let attack = *self.attack.read().unwrap_or_else(|e| e.into_inner());
        let release = *self.release.read().unwrap_or_else(|e| e.into_inner());
        let makeup_gain = *self.makeup_gain.read().unwrap_or_else(|e| e.into_inner());

        // Calculate attack and release coefficients
        let attack_coeff = (-1.0 / (sample_rate as f32 * attack / 1000.0)).exp();
        let release_coeff = (-1.0 / (sample_rate as f32 * release / 1000.0)).exp();

        let mut envelope = *self.envelope.read().unwrap_or_else(|e| e.into_inner());
        let makeup_gain_linear = 10_f32.powf(makeup_gain / 20.0);

        for sample in samples.iter_mut() {
            let input = *sample;

            // Convert to dB (with floor to avoid log(0))
            let input_level_db = 20.0 * (input.abs().max(0.00001)).log10();

            // Calculate gain reduction
            let gain_reduction_db = self.calculate_gain_reduction(input_level_db);

            // Apply envelope follower
            let target_gain = 10_f32.powf(-gain_reduction_db / 20.0);
            let coeff = if target_gain < envelope {
                attack_coeff
            } else {
                release_coeff
            };
            envelope = target_gain + coeff * (envelope - target_gain);

            // Apply compression and makeup gain
            *sample = input * envelope * makeup_gain_linear;

            // Prevent clipping
            *sample = sample.clamp(-1.0, 1.0);
        }

        *self.envelope.write().unwrap_or_else(|e| e.into_inner()) = envelope;

        Ok(processed)
    }

    fn get_parameters(&self) -> HashMap<String, ParameterValue> {
        let mut params = HashMap::new();
        params.insert(
            "threshold".to_string(),
            ParameterValue::Float(*self.threshold.read().unwrap_or_else(|e| e.into_inner())),
        );
        params.insert(
            "ratio".to_string(),
            ParameterValue::Float(*self.ratio.read().unwrap_or_else(|e| e.into_inner())),
        );
        params.insert(
            "attack".to_string(),
            ParameterValue::Float(*self.attack.read().unwrap_or_else(|e| e.into_inner())),
        );
        params.insert(
            "release".to_string(),
            ParameterValue::Float(*self.release.read().unwrap_or_else(|e| e.into_inner())),
        );
        params.insert(
            "makeup_gain".to_string(),
            ParameterValue::Float(*self.makeup_gain.read().unwrap_or_else(|e| e.into_inner())),
        );
        params
    }

    fn set_parameter(&self, name: &str, value: ParameterValue) -> Result<()> {
        match name {
            "threshold" => {
                if let Some(v) = value.as_f32() {
                    *self.threshold.write().unwrap_or_else(|e| e.into_inner()) =
                        v.clamp(-60.0, 0.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid threshold parameter type",
                    ))
                }
            }
            "ratio" => {
                if let Some(v) = value.as_f32() {
                    *self.ratio.write().unwrap_or_else(|e| e.into_inner()) = v.clamp(1.0, 20.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid ratio parameter type",
                    ))
                }
            }
            "attack" => {
                if let Some(v) = value.as_f32() {
                    *self.attack.write().unwrap_or_else(|e| e.into_inner()) =
                        v.clamp(0.1, 100.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid attack parameter type",
                    ))
                }
            }
            "release" => {
                if let Some(v) = value.as_f32() {
                    *self.release.write().unwrap_or_else(|e| e.into_inner()) =
                        v.clamp(10.0, 1000.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid release parameter type",
                    ))
                }
            }
            "makeup_gain" => {
                if let Some(v) = value.as_f32() {
                    *self.makeup_gain.write().unwrap_or_else(|e| e.into_inner()) =
                        v.clamp(0.0, 30.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid makeup_gain parameter type",
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
            "threshold" => Some(ParameterDefinition {
                name: "threshold".to_string(),
                description: "Compression threshold in dB".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(-20.0),
                min_value: Some(ParameterValue::Float(-60.0)),
                max_value: Some(ParameterValue::Float(0.0)),
                step_size: Some(0.1),
                realtime_safe: true,
            }),
            "ratio" => Some(ParameterDefinition {
                name: "ratio".to_string(),
                description: "Compression ratio".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(4.0),
                min_value: Some(ParameterValue::Float(1.0)),
                max_value: Some(ParameterValue::Float(20.0)),
                step_size: Some(0.1),
                realtime_safe: true,
            }),
            "attack" => Some(ParameterDefinition {
                name: "attack".to_string(),
                description: "Attack time in milliseconds".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(5.0),
                min_value: Some(ParameterValue::Float(0.1)),
                max_value: Some(ParameterValue::Float(100.0)),
                step_size: Some(0.1),
                realtime_safe: true,
            }),
            "release" => Some(ParameterDefinition {
                name: "release".to_string(),
                description: "Release time in milliseconds".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(50.0),
                min_value: Some(ParameterValue::Float(10.0)),
                max_value: Some(ParameterValue::Float(1000.0)),
                step_size: Some(1.0),
                realtime_safe: true,
            }),
            "makeup_gain" => Some(ParameterDefinition {
                name: "makeup_gain".to_string(),
                description: "Makeup gain in dB".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.0),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(30.0)),
                step_size: Some(0.1),
                realtime_safe: true,
            }),
            _ => None,
        }
    }
}
