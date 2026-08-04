//! Delay effect implementation with feedback and damping.

use crate::{
    audio::AudioBuffer,
    error::Result,
    plugins::{AudioEffect, ParameterDefinition, ParameterType, ParameterValue, VoirsPlugin},
    VoirsError,
};
use async_trait::async_trait;
use std::{collections::HashMap, sync::RwLock};

/// Delay effect plugin with feedback and damping
pub struct DelayEffect {
    /// Delay time in milliseconds
    pub delay_ms: RwLock<f32>,

    /// Feedback amount (0.0 - 0.95)
    pub feedback: RwLock<f32>,

    /// Wet/dry mix (0.0 = dry, 1.0 = wet)
    pub mix: RwLock<f32>,

    /// High-frequency damping (0.0 - 1.0)
    pub damping: RwLock<f32>,

    /// Delay buffer
    delay_buffer: RwLock<Vec<f32>>,

    /// Current position in delay buffer
    buffer_position: RwLock<usize>,
}

impl DelayEffect {
    pub fn new() -> Self {
        Self {
            delay_ms: RwLock::new(250.0),
            feedback: RwLock::new(0.4),
            mix: RwLock::new(0.3),
            damping: RwLock::new(0.2),
            delay_buffer: RwLock::new(Vec::new()),
            buffer_position: RwLock::new(0),
        }
    }

    fn initialize_buffer(&self, sample_rate: u32) {
        let max_delay_samples = (sample_rate as f32 * 2.0) as usize; // 2 seconds max delay
        let mut buffer = self.delay_buffer.write().unwrap_or_else(|e| e.into_inner());
        if buffer.len() != max_delay_samples {
            *buffer = vec![0.0; max_delay_samples];
            *self
                .buffer_position
                .write()
                .unwrap_or_else(|e| e.into_inner()) = 0;
        }
    }
}

impl Default for DelayEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl VoirsPlugin for DelayEffect {
    fn name(&self) -> &str {
        "Delay"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "High-quality delay effect with feedback and damping"
    }

    fn author(&self) -> &str {
        "VoiRS Team"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait]
impl AudioEffect for DelayEffect {
    async fn process_audio(&self, audio: &AudioBuffer) -> Result<AudioBuffer> {
        self.initialize_buffer(audio.sample_rate());

        let delay_samples = (*self.delay_ms.read().unwrap_or_else(|e| e.into_inner())
            * audio.sample_rate() as f32
            / 1000.0) as usize;
        let feedback = *self.feedback.read().unwrap_or_else(|e| e.into_inner());
        let mix = *self.mix.read().unwrap_or_else(|e| e.into_inner());
        let damping = *self.damping.read().unwrap_or_else(|e| e.into_inner());

        let mut processed = audio.clone();
        let samples = processed.samples_mut();
        let mut buffer = self.delay_buffer.write().unwrap_or_else(|e| e.into_inner());
        let mut pos = *self
            .buffer_position
            .read()
            .unwrap_or_else(|e| e.into_inner());

        for sample in samples.iter_mut() {
            let delay_pos = if pos >= delay_samples {
                pos - delay_samples
            } else {
                buffer.len() + pos - delay_samples
            };

            let delayed_sample = buffer[delay_pos];

            // Apply damping to high frequencies (simple lowpass)
            let damped_delayed = delayed_sample * (1.0 - damping);

            // Mix delayed signal back into buffer with feedback
            buffer[pos] = *sample + damped_delayed * feedback;

            // Mix dry and wet signals
            *sample = *sample * (1.0 - mix) + damped_delayed * mix;

            pos = (pos + 1) % buffer.len();
        }

        *self
            .buffer_position
            .write()
            .unwrap_or_else(|e| e.into_inner()) = pos;
        Ok(processed)
    }

    fn get_parameters(&self) -> HashMap<String, ParameterValue> {
        let mut params = HashMap::new();
        params.insert(
            "delay_ms".to_string(),
            ParameterValue::Float(*self.delay_ms.read().unwrap_or_else(|e| e.into_inner())),
        );
        params.insert(
            "feedback".to_string(),
            ParameterValue::Float(*self.feedback.read().unwrap_or_else(|e| e.into_inner())),
        );
        params.insert(
            "mix".to_string(),
            ParameterValue::Float(*self.mix.read().unwrap_or_else(|e| e.into_inner())),
        );
        params.insert(
            "damping".to_string(),
            ParameterValue::Float(*self.damping.read().unwrap_or_else(|e| e.into_inner())),
        );
        params
    }

    fn set_parameter(&self, name: &str, value: ParameterValue) -> Result<()> {
        match name {
            "delay_ms" => {
                if let Some(v) = value.as_f32() {
                    *self.delay_ms.write().unwrap_or_else(|e| e.into_inner()) =
                        v.clamp(1.0, 2000.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid delay_ms parameter type",
                    ))
                }
            }
            "feedback" => {
                if let Some(v) = value.as_f32() {
                    *self.feedback.write().unwrap_or_else(|e| e.into_inner()) = v.clamp(0.0, 0.95);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid feedback parameter type",
                    ))
                }
            }
            "mix" => {
                if let Some(v) = value.as_f32() {
                    *self.mix.write().unwrap_or_else(|e| e.into_inner()) = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid mix parameter type",
                    ))
                }
            }
            "damping" => {
                if let Some(v) = value.as_f32() {
                    *self.damping.write().unwrap_or_else(|e| e.into_inner()) = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid damping parameter type",
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
            "delay_ms" => Some(ParameterDefinition {
                name: "delay_ms".to_string(),
                description: "Delay time in milliseconds".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(250.0),
                min_value: Some(ParameterValue::Float(1.0)),
                max_value: Some(ParameterValue::Float(2000.0)),
                step_size: Some(1.0),
                realtime_safe: false,
            }),
            "feedback" => Some(ParameterDefinition {
                name: "feedback".to_string(),
                description: "Feedback amount".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.4),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(0.95)),
                step_size: Some(0.01),
                realtime_safe: true,
            }),
            "mix" => Some(ParameterDefinition {
                name: "mix".to_string(),
                description: "Wet/dry mix level".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.3),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(1.0)),
                step_size: Some(0.01),
                realtime_safe: true,
            }),
            "damping" => Some(ParameterDefinition {
                name: "damping".to_string(),
                description: "High-frequency damping".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.2),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(1.0)),
                step_size: Some(0.01),
                realtime_safe: true,
            }),
            _ => None,
        }
    }

    fn get_latency_samples(&self) -> usize {
        // Delay effect adds latency equal to the delay time
        let delay_ms = *self.delay_ms.read().unwrap_or_else(|e| e.into_inner());
        (delay_ms * 44.1) as usize // Assume 44.1 kHz for estimation
    }
}
