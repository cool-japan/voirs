//! 3-band equalizer effect implementation.

use super::filters::BiquadFilter;
use crate::{
    audio::AudioBuffer,
    error::Result,
    plugins::{AudioEffect, ParameterDefinition, ParameterType, ParameterValue, VoirsPlugin},
    VoirsError,
};
use async_trait::async_trait;
use std::{collections::HashMap, sync::RwLock};

/// 3-band equalizer effect for frequency shaping
pub struct EqualizerEffect {
    /// Low frequency gain (dB)
    pub low_gain: RwLock<f32>,

    /// Mid frequency gain (dB)
    pub mid_gain: RwLock<f32>,

    /// High frequency gain (dB)
    pub high_gain: RwLock<f32>,

    /// Low frequency cutoff (Hz)
    pub low_freq: RwLock<f32>,

    /// High frequency cutoff (Hz)
    pub high_freq: RwLock<f32>,

    /// Low shelf filter
    low_filter: RwLock<BiquadFilter>,

    /// Mid peaking filter
    mid_filter: RwLock<BiquadFilter>,

    /// High shelf filter
    high_filter: RwLock<BiquadFilter>,

    /// Current sample rate
    sample_rate: RwLock<Option<u32>>,
}

impl EqualizerEffect {
    pub fn new() -> Self {
        Self {
            low_gain: RwLock::new(0.0),
            mid_gain: RwLock::new(0.0),
            high_gain: RwLock::new(0.0),
            low_freq: RwLock::new(200.0),
            high_freq: RwLock::new(2000.0),
            low_filter: RwLock::new(BiquadFilter::new()),
            mid_filter: RwLock::new(BiquadFilter::new()),
            high_filter: RwLock::new(BiquadFilter::new()),
            sample_rate: RwLock::new(None),
        }
    }

    fn update_filters(&self, sample_rate: u32) {
        let low_gain = *self.low_gain.read().unwrap_or_else(|e| e.into_inner());
        let mid_gain = *self.mid_gain.read().unwrap_or_else(|e| e.into_inner());
        let high_gain = *self.high_gain.read().unwrap_or_else(|e| e.into_inner());
        let low_freq = *self.low_freq.read().unwrap_or_else(|e| e.into_inner());
        let high_freq = *self.high_freq.read().unwrap_or_else(|e| e.into_inner());

        // Update filter coefficients
        self.low_filter
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .set_low_shelf(low_freq, low_gain, sample_rate as f32);

        // Mid frequency is between low and high frequencies
        let mid_freq = (low_freq * high_freq).sqrt(); // Geometric mean
        self.mid_filter
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .set_peaking(mid_freq, mid_gain, 0.7, sample_rate as f32);

        self.high_filter
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .set_high_shelf(high_freq, high_gain, sample_rate as f32);

        *self.sample_rate.write().unwrap_or_else(|e| e.into_inner()) = Some(sample_rate);
    }
}

impl Default for EqualizerEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl VoirsPlugin for EqualizerEffect {
    fn name(&self) -> &str {
        "Equalizer"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "3-band equalizer for frequency shaping"
    }

    fn author(&self) -> &str {
        "VoiRS Team"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait]
impl AudioEffect for EqualizerEffect {
    async fn process_audio(&self, audio: &AudioBuffer) -> Result<AudioBuffer> {
        // Update filters if sample rate changed
        let current_sample_rate = *self.sample_rate.read().unwrap_or_else(|e| e.into_inner());
        if current_sample_rate != Some(audio.sample_rate()) {
            self.update_filters(audio.sample_rate());
        }

        let mut processed = audio.clone();
        let samples = processed.samples_mut();

        let mut low_filter = self.low_filter.write().unwrap_or_else(|e| e.into_inner());
        let mut mid_filter = self.mid_filter.write().unwrap_or_else(|e| e.into_inner());
        let mut high_filter = self.high_filter.write().unwrap_or_else(|e| e.into_inner());

        for sample in samples.iter_mut() {
            let input = *sample;

            // Process through each EQ band in series
            let low_output = low_filter.process(input);
            let mid_output = mid_filter.process(low_output);
            let high_output = high_filter.process(mid_output);

            *sample = high_output.clamp(-1.0, 1.0);
        }

        Ok(processed)
    }

    fn get_parameters(&self) -> HashMap<String, ParameterValue> {
        let mut params = HashMap::new();
        params.insert(
            "low_gain".to_string(),
            ParameterValue::Float(*self.low_gain.read().unwrap_or_else(|e| e.into_inner())),
        );
        params.insert(
            "mid_gain".to_string(),
            ParameterValue::Float(*self.mid_gain.read().unwrap_or_else(|e| e.into_inner())),
        );
        params.insert(
            "high_gain".to_string(),
            ParameterValue::Float(*self.high_gain.read().unwrap_or_else(|e| e.into_inner())),
        );
        params.insert(
            "low_freq".to_string(),
            ParameterValue::Float(*self.low_freq.read().unwrap_or_else(|e| e.into_inner())),
        );
        params.insert(
            "high_freq".to_string(),
            ParameterValue::Float(*self.high_freq.read().unwrap_or_else(|e| e.into_inner())),
        );
        params
    }

    fn set_parameter(&self, name: &str, value: ParameterValue) -> Result<()> {
        match name {
            "low_gain" => {
                if let Some(v) = value.as_f32() {
                    *self.low_gain.write().unwrap_or_else(|e| e.into_inner()) =
                        v.clamp(-20.0, 20.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid low_gain parameter type",
                    ))
                }
            }
            "mid_gain" => {
                if let Some(v) = value.as_f32() {
                    *self.mid_gain.write().unwrap_or_else(|e| e.into_inner()) =
                        v.clamp(-20.0, 20.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid mid_gain parameter type",
                    ))
                }
            }
            "high_gain" => {
                if let Some(v) = value.as_f32() {
                    *self.high_gain.write().unwrap_or_else(|e| e.into_inner()) =
                        v.clamp(-20.0, 20.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid high_gain parameter type",
                    ))
                }
            }
            "low_freq" => {
                if let Some(v) = value.as_f32() {
                    *self.low_freq.write().unwrap_or_else(|e| e.into_inner()) =
                        v.clamp(20.0, 20000.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid low_freq parameter type",
                    ))
                }
            }
            "high_freq" => {
                if let Some(v) = value.as_f32() {
                    *self.high_freq.write().unwrap_or_else(|e| e.into_inner()) =
                        v.clamp(20.0, 20000.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid high_freq parameter type",
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
            "low_gain" => Some(ParameterDefinition {
                name: "low_gain".to_string(),
                description: "Low frequency gain in dB".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.0),
                min_value: Some(ParameterValue::Float(-20.0)),
                max_value: Some(ParameterValue::Float(20.0)),
                step_size: Some(0.1),
                realtime_safe: true,
            }),
            "mid_gain" => Some(ParameterDefinition {
                name: "mid_gain".to_string(),
                description: "Mid frequency gain in dB".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.0),
                min_value: Some(ParameterValue::Float(-20.0)),
                max_value: Some(ParameterValue::Float(20.0)),
                step_size: Some(0.1),
                realtime_safe: true,
            }),
            "high_gain" => Some(ParameterDefinition {
                name: "high_gain".to_string(),
                description: "High frequency gain in dB".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.0),
                min_value: Some(ParameterValue::Float(-20.0)),
                max_value: Some(ParameterValue::Float(20.0)),
                step_size: Some(0.1),
                realtime_safe: true,
            }),
            "low_freq" => Some(ParameterDefinition {
                name: "low_freq".to_string(),
                description: "Low/mid crossover frequency in Hz".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(200.0),
                min_value: Some(ParameterValue::Float(20.0)),
                max_value: Some(ParameterValue::Float(20000.0)),
                step_size: Some(10.0),
                realtime_safe: false,
            }),
            "high_freq" => Some(ParameterDefinition {
                name: "high_freq".to_string(),
                description: "Mid/high crossover frequency in Hz".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(2000.0),
                min_value: Some(ParameterValue::Float(20.0)),
                max_value: Some(ParameterValue::Float(20000.0)),
                step_size: Some(10.0),
                realtime_safe: false,
            }),
            _ => None,
        }
    }
}
