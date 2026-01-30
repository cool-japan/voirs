//! Reverb effect implementation using Freeverb-style algorithm.

use super::filters::{AllpassFilter, CombFilter};
use crate::{
    audio::AudioBuffer,
    error::Result,
    plugins::{AudioEffect, ParameterDefinition, ParameterType, ParameterValue, VoirsPlugin},
    VoirsError,
};
use async_trait::async_trait;
use std::{collections::HashMap, sync::RwLock};

/// Reverb effect plugin using Freeverb-style algorithm
pub struct ReverbEffect {
    /// Wet/dry mix (0.0 = dry, 1.0 = wet)
    pub mix: RwLock<f32>,

    /// Room size (0.0 - 1.0)
    pub room_size: RwLock<f32>,

    /// Damping factor (0.0 - 1.0)
    pub damping: RwLock<f32>,

    /// Decay time in seconds
    pub decay_time: RwLock<f32>,

    /// Comb filter delay lines
    comb_filters: RwLock<Vec<CombFilter>>,

    /// All-pass filter delay lines
    allpass_filters: RwLock<Vec<AllpassFilter>>,

    /// Sample rate for proper initialization
    sample_rate: RwLock<Option<u32>>,
}

impl ReverbEffect {
    pub fn new() -> Self {
        Self {
            mix: RwLock::new(0.3),
            room_size: RwLock::new(0.5),
            damping: RwLock::new(0.5),
            decay_time: RwLock::new(2.0),
            comb_filters: RwLock::new(Vec::new()),
            allpass_filters: RwLock::new(Vec::new()),
            sample_rate: RwLock::new(None),
        }
    }

    fn initialize_filters(&self, sample_rate: u32) {
        let mut comb_filters = self.comb_filters.write().unwrap();
        let mut allpass_filters = self.allpass_filters.write().unwrap();

        // Freeverb comb filter delay lengths (in samples at 44.1kHz)
        let comb_tunings = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
        let allpass_tunings = [556, 441, 341, 225];

        // Scale for current sample rate
        let scale = sample_rate as f32 / 44100.0;

        if comb_filters.is_empty() {
            comb_filters.clear();
            for &tuning in &comb_tunings {
                let size = (tuning as f32 * scale) as usize;
                comb_filters.push(CombFilter::new(size));
            }
        }

        if allpass_filters.is_empty() {
            allpass_filters.clear();
            for &tuning in &allpass_tunings {
                let size = (tuning as f32 * scale) as usize;
                allpass_filters.push(AllpassFilter::new(size));
            }
        }

        *self.sample_rate.write().unwrap() = Some(sample_rate);
    }

    fn update_parameters(&self) {
        let room_size = *self.room_size.read().unwrap();
        let damping = *self.damping.read().unwrap();
        let decay = *self.decay_time.read().unwrap();

        // Calculate feedback based on room size and decay time
        let feedback = 0.28 + (room_size * 0.7) * (decay / 10.0).min(1.0);

        let mut comb_filters = self.comb_filters.write().unwrap();
        for filter in comb_filters.iter_mut() {
            filter.set_feedback(feedback);
            filter.set_damp(damping);
        }
    }
}

impl Default for ReverbEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl VoirsPlugin for ReverbEffect {
    fn name(&self) -> &str {
        "Reverb"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "High-quality reverb effect for spatial audio enhancement"
    }

    fn author(&self) -> &str {
        "VoiRS Team"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait]
impl AudioEffect for ReverbEffect {
    async fn process_audio(&self, audio: &AudioBuffer) -> Result<AudioBuffer> {
        // Initialize filters if needed
        let current_sample_rate = *self.sample_rate.read().unwrap();
        if current_sample_rate.is_none() || current_sample_rate.unwrap() != audio.sample_rate() {
            self.initialize_filters(audio.sample_rate());
        }

        // Update parameters
        self.update_parameters();

        let mut processed = audio.clone();
        let samples = processed.samples_mut();
        let mix = *self.mix.read().unwrap();

        let mut comb_filters = self.comb_filters.write().unwrap();
        let mut allpass_filters = self.allpass_filters.write().unwrap();

        for sample in samples.iter_mut() {
            let input = *sample;

            // Process through comb filters (parallel)
            let mut comb_output = 0.0;
            for filter in comb_filters.iter_mut() {
                comb_output += filter.process(input);
            }

            // Process through allpass filters (series)
            let mut allpass_output = comb_output;
            for filter in allpass_filters.iter_mut() {
                allpass_output = filter.process(allpass_output);
            }

            // Mix dry and wet signals
            *sample = input * (1.0 - mix) + allpass_output * mix;

            // Ensure no clipping
            *sample = sample.clamp(-1.0, 1.0);
        }

        Ok(processed)
    }

    fn get_parameters(&self) -> HashMap<String, ParameterValue> {
        let mut params = HashMap::new();
        params.insert(
            "mix".to_string(),
            ParameterValue::Float(*self.mix.read().unwrap()),
        );
        params.insert(
            "room_size".to_string(),
            ParameterValue::Float(*self.room_size.read().unwrap()),
        );
        params.insert(
            "damping".to_string(),
            ParameterValue::Float(*self.damping.read().unwrap()),
        );
        params.insert(
            "decay_time".to_string(),
            ParameterValue::Float(*self.decay_time.read().unwrap()),
        );
        params
    }

    fn set_parameter(&self, name: &str, value: ParameterValue) -> Result<()> {
        match name {
            "mix" => {
                if let Some(v) = value.as_f32() {
                    *self.mix.write().unwrap() = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid mix parameter type",
                    ))
                }
            }
            "room_size" => {
                if let Some(v) = value.as_f32() {
                    *self.room_size.write().unwrap() = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid room_size parameter type",
                    ))
                }
            }
            "damping" => {
                if let Some(v) = value.as_f32() {
                    *self.damping.write().unwrap() = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid damping parameter type",
                    ))
                }
            }
            "decay_time" => {
                if let Some(v) = value.as_f32() {
                    *self.decay_time.write().unwrap() = v.clamp(0.1, 10.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid decay_time parameter type",
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
            "damping" => Some(ParameterDefinition {
                name: "damping".to_string(),
                description: "High frequency damping".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.5),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(1.0)),
                step_size: Some(0.01),
                realtime_safe: true,
            }),
            "decay_time" => Some(ParameterDefinition {
                name: "decay_time".to_string(),
                description: "Reverb decay time in seconds".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(2.0),
                min_value: Some(ParameterValue::Float(0.1)),
                max_value: Some(ParameterValue::Float(10.0)),
                step_size: Some(0.1),
                realtime_safe: false,
            }),
            _ => None,
        }
    }

    fn get_latency_samples(&self) -> usize {
        // Reverb typically adds some latency
        512
    }
}
