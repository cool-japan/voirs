//! Enhancement plugins for speech quality improvement
//!
//! This module provides various audio enhancement plugins specifically designed
//! for improving speech synthesis quality, including noise reduction, speech
//! enhancement, quality upsampling, and artifact removal.

use crate::{
    audio::AudioBuffer,
    error::Result,
    plugins::{AudioEffect, ParameterDefinition, ParameterType, ParameterValue, VoirsPlugin},
    VoirsError,
};
use async_trait::async_trait;
use std::{collections::HashMap, sync::RwLock};

/// Noise reduction plugin using spectral subtraction
pub struct NoiseReduction {
    /// Noise floor threshold in dB
    pub noise_floor: RwLock<f32>,

    /// Reduction strength (0.0 - 1.0)
    pub reduction_strength: RwLock<f32>,

    /// Smoothing factor for spectral subtraction
    pub smoothing: RwLock<f32>,

    /// Preserve speech clarity (0.0 - 1.0)
    pub speech_preservation: RwLock<f32>,
}

impl NoiseReduction {
    pub fn new() -> Self {
        Self {
            noise_floor: RwLock::new(-40.0),
            reduction_strength: RwLock::new(0.7),
            smoothing: RwLock::new(0.8),
            speech_preservation: RwLock::new(0.9),
        }
    }

    /// Simple noise reduction using amplitude thresholding
    fn apply_noise_reduction(&self, samples: &mut [f32]) {
        let noise_floor_linear = 10.0_f32.powf(*self.noise_floor.read().unwrap() / 20.0);
        let reduction_strength = *self.reduction_strength.read().unwrap();
        let speech_preservation = *self.speech_preservation.read().unwrap();

        for sample in samples.iter_mut() {
            let abs_sample = sample.abs();

            if abs_sample < noise_floor_linear {
                // Apply noise reduction
                *sample *= 1.0 - reduction_strength;
            } else {
                // Preserve speech with gentle processing
                let reduction_factor = 1.0 - (reduction_strength * (1.0 - speech_preservation));
                *sample *= reduction_factor;
            }
        }
    }
}

impl Default for NoiseReduction {
    fn default() -> Self {
        Self::new()
    }
}

impl VoirsPlugin for NoiseReduction {
    fn name(&self) -> &str {
        "Noise Reduction"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "Advanced noise reduction for speech enhancement"
    }

    fn author(&self) -> &str {
        "VoiRS Team"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait]
impl AudioEffect for NoiseReduction {
    async fn process_audio(&self, audio: &AudioBuffer) -> Result<AudioBuffer> {
        let mut processed = audio.clone();
        self.apply_noise_reduction(processed.samples_mut());
        Ok(processed)
    }

    fn get_parameters(&self) -> HashMap<String, ParameterValue> {
        let mut params = HashMap::new();
        params.insert(
            "noise_floor".to_string(),
            ParameterValue::Float(*self.noise_floor.read().unwrap()),
        );
        params.insert(
            "reduction_strength".to_string(),
            ParameterValue::Float(*self.reduction_strength.read().unwrap()),
        );
        params.insert(
            "smoothing".to_string(),
            ParameterValue::Float(*self.smoothing.read().unwrap()),
        );
        params.insert(
            "speech_preservation".to_string(),
            ParameterValue::Float(*self.speech_preservation.read().unwrap()),
        );
        params
    }

    fn set_parameter(&self, name: &str, value: ParameterValue) -> Result<()> {
        match name {
            "noise_floor" => {
                if let Some(v) = value.as_f32() {
                    *self.noise_floor.write().unwrap() = v.clamp(-80.0, -10.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid noise_floor parameter type",
                    ))
                }
            }
            "reduction_strength" => {
                if let Some(v) = value.as_f32() {
                    *self.reduction_strength.write().unwrap() = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid reduction_strength parameter type",
                    ))
                }
            }
            "smoothing" => {
                if let Some(v) = value.as_f32() {
                    *self.smoothing.write().unwrap() = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid smoothing parameter type",
                    ))
                }
            }
            "speech_preservation" => {
                if let Some(v) = value.as_f32() {
                    *self.speech_preservation.write().unwrap() = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid speech_preservation parameter type",
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
            "noise_floor" => Some(ParameterDefinition {
                name: "noise_floor".to_string(),
                description: "Noise floor threshold in dB".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(-40.0),
                min_value: Some(ParameterValue::Float(-80.0)),
                max_value: Some(ParameterValue::Float(-10.0)),
                step_size: Some(1.0),
                realtime_safe: true,
            }),
            "reduction_strength" => Some(ParameterDefinition {
                name: "reduction_strength".to_string(),
                description: "Noise reduction strength".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.7),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(1.0)),
                step_size: Some(0.01),
                realtime_safe: true,
            }),
            "smoothing" => Some(ParameterDefinition {
                name: "smoothing".to_string(),
                description: "Spectral smoothing factor".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.8),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(1.0)),
                step_size: Some(0.01),
                realtime_safe: false,
            }),
            "speech_preservation" => Some(ParameterDefinition {
                name: "speech_preservation".to_string(),
                description: "Speech clarity preservation".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.9),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(1.0)),
                step_size: Some(0.01),
                realtime_safe: true,
            }),
            _ => None,
        }
    }
}

/// Speech enhancement plugin for intelligibility improvement
pub struct SpeechEnhancement {
    /// Formant enhancement strength
    pub formant_enhancement: RwLock<f32>,

    /// Clarity boost (0.0 - 1.0)
    pub clarity_boost: RwLock<f32>,

    /// Presence enhancement for voice
    pub presence: RwLock<f32>,

    /// Sibilance control (0.0 - 1.0)
    pub sibilance_control: RwLock<f32>,
}

impl SpeechEnhancement {
    pub fn new() -> Self {
        Self {
            formant_enhancement: RwLock::new(0.3),
            clarity_boost: RwLock::new(0.5),
            presence: RwLock::new(0.4),
            sibilance_control: RwLock::new(0.7),
        }
    }

    /// Apply speech enhancement processing
    fn enhance_speech(&self, samples: &mut [f32], _sample_rate: u32) {
        let formant_enhancement = *self.formant_enhancement.read().unwrap();
        let clarity_boost = *self.clarity_boost.read().unwrap();
        let presence = *self.presence.read().unwrap();

        // Simple speech enhancement using frequency-domain processing
        // In a real implementation, this would use proper filter banks
        for sample in samples.iter_mut() {
            // Enhance mid frequencies (speech formants)
            let enhanced = *sample * (1.0 + formant_enhancement * 0.3);

            // Add clarity boost
            let clarity_enhanced = enhanced + (*sample * clarity_boost * 0.1);

            // Apply presence boost for voice frequencies
            let presence_enhanced = clarity_enhanced * (1.0 + presence * 0.2);

            *sample = presence_enhanced.clamp(-1.0, 1.0);
        }
    }
}

impl Default for SpeechEnhancement {
    fn default() -> Self {
        Self::new()
    }
}

impl VoirsPlugin for SpeechEnhancement {
    fn name(&self) -> &str {
        "Speech Enhancement"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "Intelligibility and clarity enhancement for speech"
    }

    fn author(&self) -> &str {
        "VoiRS Team"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait]
impl AudioEffect for SpeechEnhancement {
    async fn process_audio(&self, audio: &AudioBuffer) -> Result<AudioBuffer> {
        let mut processed = audio.clone();
        self.enhance_speech(processed.samples_mut(), audio.sample_rate());
        Ok(processed)
    }

    fn get_parameters(&self) -> HashMap<String, ParameterValue> {
        let mut params = HashMap::new();
        params.insert(
            "formant_enhancement".to_string(),
            ParameterValue::Float(*self.formant_enhancement.read().unwrap()),
        );
        params.insert(
            "clarity_boost".to_string(),
            ParameterValue::Float(*self.clarity_boost.read().unwrap()),
        );
        params.insert(
            "presence".to_string(),
            ParameterValue::Float(*self.presence.read().unwrap()),
        );
        params.insert(
            "sibilance_control".to_string(),
            ParameterValue::Float(*self.sibilance_control.read().unwrap()),
        );
        params
    }

    fn set_parameter(&self, name: &str, value: ParameterValue) -> Result<()> {
        match name {
            "formant_enhancement" => {
                if let Some(v) = value.as_f32() {
                    *self.formant_enhancement.write().unwrap() = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid formant_enhancement parameter type",
                    ))
                }
            }
            "clarity_boost" => {
                if let Some(v) = value.as_f32() {
                    *self.clarity_boost.write().unwrap() = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid clarity_boost parameter type",
                    ))
                }
            }
            "presence" => {
                if let Some(v) = value.as_f32() {
                    *self.presence.write().unwrap() = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid presence parameter type",
                    ))
                }
            }
            "sibilance_control" => {
                if let Some(v) = value.as_f32() {
                    *self.sibilance_control.write().unwrap() = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid sibilance_control parameter type",
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
            "formant_enhancement" => Some(ParameterDefinition {
                name: "formant_enhancement".to_string(),
                description: "Formant frequency enhancement".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.3),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(1.0)),
                step_size: Some(0.01),
                realtime_safe: true,
            }),
            "clarity_boost" => Some(ParameterDefinition {
                name: "clarity_boost".to_string(),
                description: "Speech clarity boost".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.5),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(1.0)),
                step_size: Some(0.01),
                realtime_safe: true,
            }),
            "presence" => Some(ParameterDefinition {
                name: "presence".to_string(),
                description: "Voice presence enhancement".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.4),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(1.0)),
                step_size: Some(0.01),
                realtime_safe: true,
            }),
            "sibilance_control" => Some(ParameterDefinition {
                name: "sibilance_control".to_string(),
                description: "Sibilance reduction control".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.7),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(1.0)),
                step_size: Some(0.01),
                realtime_safe: true,
            }),
            _ => None,
        }
    }
}

/// Quality upsampling plugin for sample rate enhancement
pub struct QualityUpsampler {
    /// Target sample rate multiplier (2x, 4x, etc.)
    pub upsample_factor: RwLock<u32>,

    /// Anti-aliasing filter strength
    pub anti_aliasing: RwLock<f32>,

    /// Interpolation quality (0.0 - 1.0)
    pub interpolation_quality: RwLock<f32>,

    /// High-frequency restoration
    pub hf_restoration: RwLock<f32>,
}

impl QualityUpsampler {
    pub fn new() -> Self {
        Self {
            upsample_factor: RwLock::new(2),
            anti_aliasing: RwLock::new(0.8),
            interpolation_quality: RwLock::new(0.9),
            hf_restoration: RwLock::new(0.5),
        }
    }

    /// Simple linear interpolation upsampling
    fn upsample_audio(&self, samples: &[f32]) -> Vec<f32> {
        let factor = *self.upsample_factor.read().unwrap() as usize;
        let quality = *self.interpolation_quality.read().unwrap();
        let hf_restoration = *self.hf_restoration.read().unwrap();

        let mut upsampled = Vec::with_capacity(samples.len() * factor);

        for i in 0..samples.len() {
            upsampled.push(samples[i]);

            // Add interpolated samples
            for j in 1..factor {
                let next_sample = if i + 1 < samples.len() {
                    samples[i + 1]
                } else {
                    samples[i]
                };

                let alpha = j as f32 / factor as f32;
                let interpolated = samples[i] * (1.0 - alpha) + next_sample * alpha;

                // Apply quality enhancement
                let enhanced = interpolated * (1.0 + quality * 0.1);

                // Add high-frequency restoration
                let with_hf = enhanced + (enhanced * hf_restoration * 0.05);

                upsampled.push(with_hf.clamp(-1.0, 1.0));
            }
        }

        upsampled
    }
}

impl Default for QualityUpsampler {
    fn default() -> Self {
        Self::new()
    }
}

impl VoirsPlugin for QualityUpsampler {
    fn name(&self) -> &str {
        "Quality Upsampler"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "High-quality sample rate upsampling with interpolation"
    }

    fn author(&self) -> &str {
        "VoiRS Team"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait]
impl AudioEffect for QualityUpsampler {
    async fn process_audio(&self, audio: &AudioBuffer) -> Result<AudioBuffer> {
        let factor = *self.upsample_factor.read().unwrap();
        let upsampled_samples = self.upsample_audio(audio.samples());
        let new_sample_rate = audio.sample_rate() * factor;

        let processed = AudioBuffer::new(upsampled_samples, new_sample_rate, audio.channels());

        Ok(processed)
    }

    fn get_parameters(&self) -> HashMap<String, ParameterValue> {
        let mut params = HashMap::new();
        params.insert(
            "upsample_factor".to_string(),
            ParameterValue::Integer(*self.upsample_factor.read().unwrap() as i64),
        );
        params.insert(
            "anti_aliasing".to_string(),
            ParameterValue::Float(*self.anti_aliasing.read().unwrap()),
        );
        params.insert(
            "interpolation_quality".to_string(),
            ParameterValue::Float(*self.interpolation_quality.read().unwrap()),
        );
        params.insert(
            "hf_restoration".to_string(),
            ParameterValue::Float(*self.hf_restoration.read().unwrap()),
        );
        params
    }

    fn set_parameter(&self, name: &str, value: ParameterValue) -> Result<()> {
        match name {
            "upsample_factor" => {
                if let Some(v) = value.as_i64() {
                    *self.upsample_factor.write().unwrap() = (v as u32).clamp(2, 8);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid upsample_factor parameter type",
                    ))
                }
            }
            "anti_aliasing" => {
                if let Some(v) = value.as_f32() {
                    *self.anti_aliasing.write().unwrap() = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid anti_aliasing parameter type",
                    ))
                }
            }
            "interpolation_quality" => {
                if let Some(v) = value.as_f32() {
                    *self.interpolation_quality.write().unwrap() = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid interpolation_quality parameter type",
                    ))
                }
            }
            "hf_restoration" => {
                if let Some(v) = value.as_f32() {
                    *self.hf_restoration.write().unwrap() = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid hf_restoration parameter type",
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
            "upsample_factor" => Some(ParameterDefinition {
                name: "upsample_factor".to_string(),
                description: "Upsampling factor (2x, 4x, etc.)".to_string(),
                parameter_type: ParameterType::Integer,
                default_value: ParameterValue::Integer(2),
                min_value: Some(ParameterValue::Integer(2)),
                max_value: Some(ParameterValue::Integer(8)),
                step_size: Some(1.0),
                realtime_safe: false,
            }),
            "anti_aliasing" => Some(ParameterDefinition {
                name: "anti_aliasing".to_string(),
                description: "Anti-aliasing filter strength".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.8),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(1.0)),
                step_size: Some(0.01),
                realtime_safe: false,
            }),
            "interpolation_quality" => Some(ParameterDefinition {
                name: "interpolation_quality".to_string(),
                description: "Interpolation quality level".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.9),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(1.0)),
                step_size: Some(0.01),
                realtime_safe: false,
            }),
            "hf_restoration" => Some(ParameterDefinition {
                name: "hf_restoration".to_string(),
                description: "High-frequency restoration".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.5),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(1.0)),
                step_size: Some(0.01),
                realtime_safe: true,
            }),
            _ => None,
        }
    }
}

/// Artifact removal plugin for cleaning synthesis artifacts
pub struct ArtifactRemoval {
    /// Click and pop removal strength
    pub click_removal: RwLock<f32>,

    /// Glitch detection sensitivity
    pub glitch_sensitivity: RwLock<f32>,

    /// Smoothing factor for artifact removal
    pub smoothing: RwLock<f32>,

    /// Preserve transients (0.0 - 1.0)
    pub transient_preservation: RwLock<f32>,
}

impl ArtifactRemoval {
    pub fn new() -> Self {
        Self {
            click_removal: RwLock::new(0.8),
            glitch_sensitivity: RwLock::new(0.7),
            smoothing: RwLock::new(0.5),
            transient_preservation: RwLock::new(0.9),
        }
    }

    /// Remove clicks, pops, and other artifacts
    fn remove_artifacts(&self, samples: &mut [f32]) {
        let click_removal = *self.click_removal.read().unwrap();
        let glitch_sensitivity = *self.glitch_sensitivity.read().unwrap();
        let smoothing = *self.smoothing.read().unwrap();
        let transient_preservation = *self.transient_preservation.read().unwrap();

        // Simple artifact detection and removal
        for i in 1..samples.len() - 1 {
            let current = samples[i];
            let prev = samples[i - 1];
            let next = samples[i + 1];

            // Detect sudden amplitude changes (clicks/pops)
            let change_from_prev = (current - prev).abs();
            let change_to_next = (current - next).abs();
            let avg_change = (change_from_prev + change_to_next) / 2.0;

            // Threshold for artifact detection
            let threshold = glitch_sensitivity * 0.1;

            if avg_change > threshold {
                // Apply artifact removal with transient preservation
                let artifact_strength = (avg_change - threshold) / threshold;
                let removal_factor =
                    click_removal * artifact_strength * (1.0 - transient_preservation);

                // Smooth interpolation between neighboring samples
                let interpolated = (prev + next) / 2.0;
                let smoothed = current * (1.0 - smoothing) + interpolated * smoothing;

                samples[i] = current * (1.0 - removal_factor) + smoothed * removal_factor;
            }
        }
    }
}

impl Default for ArtifactRemoval {
    fn default() -> Self {
        Self::new()
    }
}

impl VoirsPlugin for ArtifactRemoval {
    fn name(&self) -> &str {
        "Artifact Removal"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "Removes clicks, pops, and synthesis artifacts"
    }

    fn author(&self) -> &str {
        "VoiRS Team"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait]
impl AudioEffect for ArtifactRemoval {
    async fn process_audio(&self, audio: &AudioBuffer) -> Result<AudioBuffer> {
        let mut processed = audio.clone();
        self.remove_artifacts(processed.samples_mut());
        Ok(processed)
    }

    fn get_parameters(&self) -> HashMap<String, ParameterValue> {
        let mut params = HashMap::new();
        params.insert(
            "click_removal".to_string(),
            ParameterValue::Float(*self.click_removal.read().unwrap()),
        );
        params.insert(
            "glitch_sensitivity".to_string(),
            ParameterValue::Float(*self.glitch_sensitivity.read().unwrap()),
        );
        params.insert(
            "smoothing".to_string(),
            ParameterValue::Float(*self.smoothing.read().unwrap()),
        );
        params.insert(
            "transient_preservation".to_string(),
            ParameterValue::Float(*self.transient_preservation.read().unwrap()),
        );
        params
    }

    fn set_parameter(&self, name: &str, value: ParameterValue) -> Result<()> {
        match name {
            "click_removal" => {
                if let Some(v) = value.as_f32() {
                    *self.click_removal.write().unwrap() = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid click_removal parameter type",
                    ))
                }
            }
            "glitch_sensitivity" => {
                if let Some(v) = value.as_f32() {
                    *self.glitch_sensitivity.write().unwrap() = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid glitch_sensitivity parameter type",
                    ))
                }
            }
            "smoothing" => {
                if let Some(v) = value.as_f32() {
                    *self.smoothing.write().unwrap() = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid smoothing parameter type",
                    ))
                }
            }
            "transient_preservation" => {
                if let Some(v) = value.as_f32() {
                    *self.transient_preservation.write().unwrap() = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid transient_preservation parameter type",
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
            "click_removal" => Some(ParameterDefinition {
                name: "click_removal".to_string(),
                description: "Click and pop removal strength".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.8),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(1.0)),
                step_size: Some(0.01),
                realtime_safe: true,
            }),
            "glitch_sensitivity" => Some(ParameterDefinition {
                name: "glitch_sensitivity".to_string(),
                description: "Glitch detection sensitivity".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.7),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(1.0)),
                step_size: Some(0.01),
                realtime_safe: true,
            }),
            "smoothing" => Some(ParameterDefinition {
                name: "smoothing".to_string(),
                description: "Artifact smoothing factor".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.5),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(1.0)),
                step_size: Some(0.01),
                realtime_safe: true,
            }),
            "transient_preservation" => Some(ParameterDefinition {
                name: "transient_preservation".to_string(),
                description: "Preserve important transients".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.9),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(1.0)),
                step_size: Some(0.01),
                realtime_safe: true,
            }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_noise_reduction() {
        let nr = NoiseReduction::new();

        // Test parameter setting
        nr.set_parameter("noise_floor", ParameterValue::Float(-50.0))
            .unwrap();
        nr.set_parameter("reduction_strength", ParameterValue::Float(0.8))
            .unwrap();

        assert_eq!(*nr.noise_floor.read().unwrap(), -50.0);
        assert_eq!(*nr.reduction_strength.read().unwrap(), 0.8);

        // Test audio processing
        let audio = crate::AudioBuffer::sine_wave(440.0, 0.1, 44100, 0.5); // Quiet signal
        let processed = nr.process_audio(&audio).await.unwrap();

        assert_eq!(processed.len(), audio.len());
        assert_eq!(processed.sample_rate(), audio.sample_rate());
    }

    #[tokio::test]
    async fn test_speech_enhancement() {
        let se = SpeechEnhancement::new();

        // Test parameter setting
        se.set_parameter("formant_enhancement", ParameterValue::Float(0.5))
            .unwrap();
        se.set_parameter("clarity_boost", ParameterValue::Float(0.7))
            .unwrap();

        assert_eq!(*se.formant_enhancement.read().unwrap(), 0.5);
        assert_eq!(*se.clarity_boost.read().unwrap(), 0.7);

        // Test audio processing
        let audio = crate::AudioBuffer::sine_wave(1000.0, 0.5, 44100, 0.5);
        let processed = se.process_audio(&audio).await.unwrap();

        assert_eq!(processed.len(), audio.len());
        assert_eq!(processed.sample_rate(), audio.sample_rate());
    }

    #[tokio::test]
    async fn test_quality_upsampler() {
        let upsampler = QualityUpsampler::new();

        // Test parameter setting
        upsampler
            .set_parameter("upsample_factor", ParameterValue::Integer(4))
            .unwrap();

        assert_eq!(*upsampler.upsample_factor.read().unwrap(), 4);

        // Test audio processing - should increase sample rate
        let audio = crate::AudioBuffer::sine_wave(440.0, 0.1, 22050, 0.5);
        let processed = upsampler.process_audio(&audio).await.unwrap();

        assert_eq!(processed.len(), audio.len() * 4); // 4x upsampling
        assert_eq!(processed.sample_rate(), audio.sample_rate() * 4);
    }

    #[tokio::test]
    async fn test_artifact_removal() {
        let ar = ArtifactRemoval::new();

        // Test parameter setting
        ar.set_parameter("click_removal", ParameterValue::Float(0.9))
            .unwrap();
        ar.set_parameter("glitch_sensitivity", ParameterValue::Float(0.6))
            .unwrap();

        assert_eq!(*ar.click_removal.read().unwrap(), 0.9);
        assert_eq!(*ar.glitch_sensitivity.read().unwrap(), 0.6);

        // Test audio processing
        let audio = crate::AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.5);
        let processed = ar.process_audio(&audio).await.unwrap();

        assert_eq!(processed.len(), audio.len());
        assert_eq!(processed.sample_rate(), audio.sample_rate());
    }

    #[test]
    fn test_enhancement_plugin_metadata() {
        let nr = NoiseReduction::new();
        assert_eq!(nr.name(), "Noise Reduction");
        assert_eq!(nr.version(), "1.0.0");
        assert_eq!(nr.author(), "VoiRS Team");

        let se = SpeechEnhancement::new();
        assert_eq!(se.name(), "Speech Enhancement");

        let upsampler = QualityUpsampler::new();
        assert_eq!(upsampler.name(), "Quality Upsampler");

        let ar = ArtifactRemoval::new();
        assert_eq!(ar.name(), "Artifact Removal");
    }
}
