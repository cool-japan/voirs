//! Built-in audio effects plugins.

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

/// Comb filter for reverb
struct CombFilter {
    buffer: Vec<f32>,
    index: usize,
    feedback: f32,
    filter_state: f32,
    damp1: f32,
    damp2: f32,
}

impl CombFilter {
    fn new(size: usize) -> Self {
        Self {
            buffer: vec![0.0; size],
            index: 0,
            feedback: 0.5,
            filter_state: 0.0,
            damp1: 0.5,
            damp2: 0.5,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let output = self.buffer[self.index];
        self.filter_state = (output * self.damp2) + (self.filter_state * self.damp1);
        self.buffer[self.index] = input + (self.filter_state * self.feedback);

        self.index = (self.index + 1) % self.buffer.len();
        output
    }

    fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback;
    }

    fn set_damp(&mut self, damp: f32) {
        self.damp1 = damp;
        self.damp2 = 1.0 - damp;
    }
}

/// All-pass filter for reverb
struct AllpassFilter {
    buffer: Vec<f32>,
    index: usize,
    feedback: f32,
}

impl AllpassFilter {
    fn new(size: usize) -> Self {
        Self {
            buffer: vec![0.0; size],
            index: 0,
            feedback: 0.5,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let delayed = self.buffer[self.index];
        let output = -input + delayed;
        self.buffer[self.index] = input + (delayed * self.feedback);

        self.index = (self.index + 1) % self.buffer.len();
        output
    }
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

/// Biquad filter for EQ bands
#[derive(Clone)]
struct BiquadFilter {
    // Filter coefficients
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,

    // Filter memory
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl BiquadFilter {
    fn new() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn set_low_shelf(&mut self, freq: f32, gain_db: f32, sample_rate: f32) {
        let gain = 10.0_f32.powf(gain_db / 40.0); // Divide by 40 for shelf
        let omega = 2.0 * std::f32::consts::PI * freq / sample_rate;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let alpha = sin_omega / 2.0 * ((gain + 1.0 / gain) * (1.0 / 0.7 - 1.0) + 2.0).sqrt();

        let a = gain.sqrt();

        let b0 = a * ((a + 1.0) - (a - 1.0) * cos_omega + 2.0 * a.sqrt() * alpha);
        let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_omega);
        let b2 = a * ((a + 1.0) - (a - 1.0) * cos_omega - 2.0 * a.sqrt() * alpha);
        let a0 = (a + 1.0) + (a - 1.0) * cos_omega + 2.0 * a.sqrt() * alpha;
        let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_omega);
        let a2 = (a + 1.0) + (a - 1.0) * cos_omega - 2.0 * a.sqrt() * alpha;

        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;
    }

    fn set_high_shelf(&mut self, freq: f32, gain_db: f32, sample_rate: f32) {
        let gain = 10.0_f32.powf(gain_db / 40.0); // Divide by 40 for shelf
        let omega = 2.0 * std::f32::consts::PI * freq / sample_rate;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let alpha = sin_omega / 2.0 * ((gain + 1.0 / gain) * (1.0 / 0.7 - 1.0) + 2.0).sqrt();

        let a = gain.sqrt();

        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_omega + 2.0 * a.sqrt() * alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_omega);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_omega - 2.0 * a.sqrt() * alpha);
        let a0 = (a + 1.0) - (a - 1.0) * cos_omega + 2.0 * a.sqrt() * alpha;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_omega);
        let a2 = (a + 1.0) - (a - 1.0) * cos_omega - 2.0 * a.sqrt() * alpha;

        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;
    }

    fn set_peaking(&mut self, freq: f32, gain_db: f32, q: f32, sample_rate: f32) {
        let gain = 10.0_f32.powf(gain_db / 20.0);
        let omega = 2.0 * std::f32::consts::PI * freq / sample_rate;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let alpha = sin_omega / (2.0 * q);

        let b0 = 1.0 + alpha * gain;
        let b1 = -2.0 * cos_omega;
        let b2 = 1.0 - alpha * gain;
        let a0 = 1.0 + alpha / gain;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha / gain;

        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;
    }

    fn process(&mut self, input: f32) -> f32 {
        let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;

        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Equalizer effect plugin with proper biquad filtering
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
        let low_gain = *self.low_gain.read().unwrap();
        let mid_gain = *self.mid_gain.read().unwrap();
        let high_gain = *self.high_gain.read().unwrap();
        let low_freq = *self.low_freq.read().unwrap();
        let high_freq = *self.high_freq.read().unwrap();

        // Update filter coefficients
        self.low_filter
            .write()
            .unwrap()
            .set_low_shelf(low_freq, low_gain, sample_rate as f32);

        // Mid frequency is between low and high frequencies
        let mid_freq = (low_freq * high_freq).sqrt(); // Geometric mean
        self.mid_filter
            .write()
            .unwrap()
            .set_peaking(mid_freq, mid_gain, 0.7, sample_rate as f32);

        self.high_filter
            .write()
            .unwrap()
            .set_high_shelf(high_freq, high_gain, sample_rate as f32);

        *self.sample_rate.write().unwrap() = Some(sample_rate);
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
        let current_sample_rate = *self.sample_rate.read().unwrap();
        if current_sample_rate.is_none() || current_sample_rate.unwrap() != audio.sample_rate() {
            self.update_filters(audio.sample_rate());
        }

        let mut processed = audio.clone();
        let samples = processed.samples_mut();

        let mut low_filter = self.low_filter.write().unwrap();
        let mut mid_filter = self.mid_filter.write().unwrap();
        let mut high_filter = self.high_filter.write().unwrap();

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
            ParameterValue::Float(*self.low_gain.read().unwrap()),
        );
        params.insert(
            "mid_gain".to_string(),
            ParameterValue::Float(*self.mid_gain.read().unwrap()),
        );
        params.insert(
            "high_gain".to_string(),
            ParameterValue::Float(*self.high_gain.read().unwrap()),
        );
        params.insert(
            "low_freq".to_string(),
            ParameterValue::Float(*self.low_freq.read().unwrap()),
        );
        params.insert(
            "high_freq".to_string(),
            ParameterValue::Float(*self.high_freq.read().unwrap()),
        );
        params
    }

    fn set_parameter(&self, name: &str, value: ParameterValue) -> Result<()> {
        match name {
            "low_gain" => {
                if let Some(v) = value.as_f32() {
                    *self.low_gain.write().unwrap() = v.clamp(-20.0, 20.0);
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
                    *self.mid_gain.write().unwrap() = v.clamp(-20.0, 20.0);
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
                    *self.high_gain.write().unwrap() = v.clamp(-20.0, 20.0);
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
                    *self.low_freq.write().unwrap() = v.clamp(20.0, 20000.0);
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
                    *self.high_freq.write().unwrap() = v.clamp(20.0, 20000.0);
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

/// Compressor effect plugin with proper envelope following
pub struct CompressorEffect {
    /// Threshold in dB
    pub threshold: RwLock<f32>,

    /// Compression ratio
    pub ratio: RwLock<f32>,

    /// Attack time in milliseconds
    pub attack_ms: RwLock<f32>,

    /// Release time in milliseconds
    pub release_ms: RwLock<f32>,

    /// Makeup gain in dB
    pub makeup_gain: RwLock<f32>,

    /// Envelope follower state
    envelope: RwLock<f32>,

    /// Attack coefficient
    attack_coeff: RwLock<f32>,

    /// Release coefficient
    release_coeff: RwLock<f32>,

    /// Sample rate for coefficient calculation
    sample_rate: RwLock<Option<u32>>,
}

impl CompressorEffect {
    pub fn new() -> Self {
        Self {
            threshold: RwLock::new(-12.0),
            ratio: RwLock::new(4.0),
            attack_ms: RwLock::new(10.0),
            release_ms: RwLock::new(100.0),
            makeup_gain: RwLock::new(0.0),
            envelope: RwLock::new(0.0),
            attack_coeff: RwLock::new(0.0),
            release_coeff: RwLock::new(0.0),
            sample_rate: RwLock::new(None),
        }
    }

    fn calculate_coefficients(&self, sample_rate: u32) {
        let attack_ms = *self.attack_ms.read().unwrap();
        let release_ms = *self.release_ms.read().unwrap();

        // Calculate attack and release coefficients
        // Time constants for exponential decay: coeff = exp(-1 / (time_ms * sample_rate / 1000))
        let attack_coeff = (-1.0 / (attack_ms * sample_rate as f32 / 1000.0)).exp();
        let release_coeff = (-1.0 / (release_ms * sample_rate as f32 / 1000.0)).exp();

        *self.attack_coeff.write().unwrap() = attack_coeff;
        *self.release_coeff.write().unwrap() = release_coeff;
        *self.sample_rate.write().unwrap() = Some(sample_rate);
    }

    fn calculate_gain_reduction(&self, input_level_db: f32) -> f32 {
        let threshold = *self.threshold.read().unwrap();
        let ratio = *self.ratio.read().unwrap();

        if input_level_db <= threshold {
            // Below threshold - no compression
            0.0
        } else {
            // Above threshold - apply compression
            let excess_db = input_level_db - threshold;
            let compressed_excess = excess_db / ratio;
            let gain_reduction = excess_db - compressed_excess;
            -gain_reduction // Negative because we're reducing gain
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
        "Dynamic range compressor for level control"
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
        // Update coefficients if sample rate changed
        let current_sample_rate = *self.sample_rate.read().unwrap();
        if current_sample_rate.is_none() || current_sample_rate.unwrap() != audio.sample_rate() {
            self.calculate_coefficients(audio.sample_rate());
        }

        let makeup_linear = 10.0_f32.powf(*self.makeup_gain.read().unwrap() / 20.0);
        let attack_coeff = *self.attack_coeff.read().unwrap();
        let release_coeff = *self.release_coeff.read().unwrap();

        let mut processed = audio.clone();
        let samples = processed.samples_mut();
        let mut envelope = *self.envelope.read().unwrap();

        for sample in samples.iter_mut() {
            let input = *sample;
            let input_level = input.abs();

            // Convert to dB (with minimum to avoid log(0))
            let input_level_db = if input_level > 0.000001 {
                20.0 * input_level.log10()
            } else {
                -120.0 // Very quiet
            };

            // Calculate desired gain reduction
            let target_gain_reduction_db = self.calculate_gain_reduction(input_level_db);

            // Envelope following with attack/release
            let coeff = if target_gain_reduction_db < envelope {
                attack_coeff // Attack (gain reduction increasing)
            } else {
                release_coeff // Release (gain reduction decreasing)
            };

            envelope = target_gain_reduction_db + (envelope - target_gain_reduction_db) * coeff;

            // Convert gain reduction from dB to linear
            let gain_linear = 10.0_f32.powf(envelope / 20.0);

            // Apply compression and makeup gain
            *sample = input * gain_linear * makeup_linear;
            *sample = sample.clamp(-1.0, 1.0);
        }

        // Store envelope state
        *self.envelope.write().unwrap() = envelope;

        Ok(processed)
    }

    fn get_parameters(&self) -> HashMap<String, ParameterValue> {
        let mut params = HashMap::new();
        params.insert(
            "threshold".to_string(),
            ParameterValue::Float(*self.threshold.read().unwrap()),
        );
        params.insert(
            "ratio".to_string(),
            ParameterValue::Float(*self.ratio.read().unwrap()),
        );
        params.insert(
            "attack_ms".to_string(),
            ParameterValue::Float(*self.attack_ms.read().unwrap()),
        );
        params.insert(
            "release_ms".to_string(),
            ParameterValue::Float(*self.release_ms.read().unwrap()),
        );
        params.insert(
            "makeup_gain".to_string(),
            ParameterValue::Float(*self.makeup_gain.read().unwrap()),
        );
        params
    }

    fn set_parameter(&self, name: &str, value: ParameterValue) -> Result<()> {
        match name {
            "threshold" => {
                if let Some(v) = value.as_f32() {
                    *self.threshold.write().unwrap() = v.clamp(-60.0, 0.0);
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
                    *self.ratio.write().unwrap() = v.clamp(1.0, 20.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid ratio parameter type",
                    ))
                }
            }
            "attack_ms" => {
                if let Some(v) = value.as_f32() {
                    *self.attack_ms.write().unwrap() = v.clamp(0.1, 1000.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid attack_ms parameter type",
                    ))
                }
            }
            "release_ms" => {
                if let Some(v) = value.as_f32() {
                    *self.release_ms.write().unwrap() = v.clamp(1.0, 5000.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid release_ms parameter type",
                    ))
                }
            }
            "makeup_gain" => {
                if let Some(v) = value.as_f32() {
                    *self.makeup_gain.write().unwrap() = v.clamp(-20.0, 20.0);
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
                default_value: ParameterValue::Float(-12.0),
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
            "attack_ms" => Some(ParameterDefinition {
                name: "attack_ms".to_string(),
                description: "Attack time in milliseconds".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(10.0),
                min_value: Some(ParameterValue::Float(0.1)),
                max_value: Some(ParameterValue::Float(1000.0)),
                step_size: Some(0.1),
                realtime_safe: false,
            }),
            "release_ms" => Some(ParameterDefinition {
                name: "release_ms".to_string(),
                description: "Release time in milliseconds".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(100.0),
                min_value: Some(ParameterValue::Float(1.0)),
                max_value: Some(ParameterValue::Float(5000.0)),
                step_size: Some(1.0),
                realtime_safe: false,
            }),
            "makeup_gain" => Some(ParameterDefinition {
                name: "makeup_gain".to_string(),
                description: "Makeup gain in dB".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.0),
                min_value: Some(ParameterValue::Float(-20.0)),
                max_value: Some(ParameterValue::Float(20.0)),
                step_size: Some(0.1),
                realtime_safe: true,
            }),
            _ => None,
        }
    }
}

/// Delay effect plugin
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
        let mut buffer = self.delay_buffer.write().unwrap();
        if buffer.len() != max_delay_samples {
            *buffer = vec![0.0; max_delay_samples];
            *self.buffer_position.write().unwrap() = 0;
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

        let delay_samples =
            (*self.delay_ms.read().unwrap() * audio.sample_rate() as f32 / 1000.0) as usize;
        let feedback = *self.feedback.read().unwrap();
        let mix = *self.mix.read().unwrap();
        let damping = *self.damping.read().unwrap();

        let mut processed = audio.clone();
        let samples = processed.samples_mut();
        let mut buffer = self.delay_buffer.write().unwrap();
        let mut pos = *self.buffer_position.read().unwrap();

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

        *self.buffer_position.write().unwrap() = pos;
        Ok(processed)
    }

    fn get_parameters(&self) -> HashMap<String, ParameterValue> {
        let mut params = HashMap::new();
        params.insert(
            "delay_ms".to_string(),
            ParameterValue::Float(*self.delay_ms.read().unwrap()),
        );
        params.insert(
            "feedback".to_string(),
            ParameterValue::Float(*self.feedback.read().unwrap()),
        );
        params.insert(
            "mix".to_string(),
            ParameterValue::Float(*self.mix.read().unwrap()),
        );
        params.insert(
            "damping".to_string(),
            ParameterValue::Float(*self.damping.read().unwrap()),
        );
        params
    }

    fn set_parameter(&self, name: &str, value: ParameterValue) -> Result<()> {
        match name {
            "delay_ms" => {
                if let Some(v) = value.as_f32() {
                    *self.delay_ms.write().unwrap() = v.clamp(1.0, 2000.0);
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
                    *self.feedback.write().unwrap() = v.clamp(0.0, 0.95);
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
                    *self.mix.write().unwrap() = v.clamp(0.0, 1.0);
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
                    *self.damping.write().unwrap() = v.clamp(0.0, 1.0);
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
        let delay_ms = *self.delay_ms.read().unwrap();
        (delay_ms * 44.1) as usize // Assume 44.1 kHz for estimation
    }
}

/// Spatial audio effect plugin for 3D positioning
pub struct SpatialAudioEffect {
    /// Azimuth angle in degrees (-180 to 180)
    pub azimuth: RwLock<f32>,

    /// Elevation angle in degrees (-90 to 90)
    pub elevation: RwLock<f32>,

    /// Distance from listener (0.1 to 100.0)
    pub distance: RwLock<f32>,

    /// Room size for reverb (0.0 to 1.0)
    pub room_size: RwLock<f32>,

    /// Head-related transfer function intensity
    pub hrtf_intensity: RwLock<f32>,
}

impl SpatialAudioEffect {
    pub fn new() -> Self {
        Self {
            azimuth: RwLock::new(0.0),
            elevation: RwLock::new(0.0),
            distance: RwLock::new(1.0),
            room_size: RwLock::new(0.3),
            hrtf_intensity: RwLock::new(0.7),
        }
    }

    /// Calculate HRTF-based gain for left and right channels
    fn calculate_stereo_gains(&self) -> (f32, f32) {
        let azimuth = *self.azimuth.read().unwrap();
        let distance = *self.distance.read().unwrap();
        let hrtf_intensity = *self.hrtf_intensity.read().unwrap();

        // Simple panning based on azimuth
        let pan_radians = azimuth.to_radians();
        let left_gain = ((pan_radians + std::f32::consts::PI / 2.0).cos() + 1.0) / 2.0;
        let right_gain = ((pan_radians - std::f32::consts::PI / 2.0).cos() + 1.0) / 2.0;

        // Apply distance attenuation
        let distance_attenuation = 1.0 / (1.0 + distance * 0.5);

        // Apply HRTF intensity
        let left_final = left_gain * distance_attenuation * hrtf_intensity + (1.0 - hrtf_intensity);
        let right_final =
            right_gain * distance_attenuation * hrtf_intensity + (1.0 - hrtf_intensity);

        (left_final, right_final)
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
        "3D spatial audio positioning with HRTF simulation"
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
        let (left_gain, right_gain) = self.calculate_stereo_gains();

        // For mono input, create stereo output
        let mut processed = if audio.channels() == 1 {
            let mut stereo_samples = Vec::with_capacity(audio.samples().len() * 2);

            // Interleave mono to stereo with spatial gains
            for &sample in audio.samples() {
                stereo_samples.push(sample * left_gain);
                stereo_samples.push(sample * right_gain);
            }

            AudioBuffer::new(
                stereo_samples,
                audio.sample_rate(),
                2, // Stereo output
            )
        } else {
            // For stereo input, apply spatial processing
            let mut processed = audio.clone();
            let samples = processed.samples_mut();

            for chunk in samples.chunks_mut(2) {
                if chunk.len() == 2 {
                    chunk[0] *= left_gain; // Left channel
                    chunk[1] *= right_gain; // Right channel
                }
            }

            processed
        };

        // Apply simple room reverb based on room size
        let room_size = *self.room_size.read().unwrap();
        if room_size > 0.1 {
            let reverb_gain = room_size * 0.3;
            let reverb_delay = (room_size * 0.05 * audio.sample_rate() as f32) as usize;

            let samples = processed.samples_mut();
            if samples.len() > reverb_delay {
                for i in reverb_delay..samples.len() {
                    let reverb_sample = samples[i - reverb_delay] * reverb_gain;
                    samples[i] += reverb_sample;
                    samples[i] = samples[i].clamp(-1.0, 1.0);
                }
            }
        }

        Ok(processed)
    }

    fn get_parameters(&self) -> HashMap<String, ParameterValue> {
        let mut params = HashMap::new();
        params.insert(
            "azimuth".to_string(),
            ParameterValue::Float(*self.azimuth.read().unwrap()),
        );
        params.insert(
            "elevation".to_string(),
            ParameterValue::Float(*self.elevation.read().unwrap()),
        );
        params.insert(
            "distance".to_string(),
            ParameterValue::Float(*self.distance.read().unwrap()),
        );
        params.insert(
            "room_size".to_string(),
            ParameterValue::Float(*self.room_size.read().unwrap()),
        );
        params.insert(
            "hrtf_intensity".to_string(),
            ParameterValue::Float(*self.hrtf_intensity.read().unwrap()),
        );
        params
    }

    fn set_parameter(&self, name: &str, value: ParameterValue) -> Result<()> {
        match name {
            "azimuth" => {
                if let Some(v) = value.as_f32() {
                    *self.azimuth.write().unwrap() = v.clamp(-180.0, 180.0);
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
                    *self.elevation.write().unwrap() = v.clamp(-90.0, 90.0);
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
                    *self.distance.write().unwrap() = v.clamp(0.1, 100.0);
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
                    *self.room_size.write().unwrap() = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid room_size parameter type",
                    ))
                }
            }
            "hrtf_intensity" => {
                if let Some(v) = value.as_f32() {
                    *self.hrtf_intensity.write().unwrap() = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal(
                        "plugins",
                        "Invalid hrtf_intensity parameter type",
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
                description: "Distance from listener".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(1.0),
                min_value: Some(ParameterValue::Float(0.1)),
                max_value: Some(ParameterValue::Float(100.0)),
                step_size: Some(0.1),
                realtime_safe: true,
            }),
            "room_size" => Some(ParameterDefinition {
                name: "room_size".to_string(),
                description: "Virtual room size for reverb".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.3),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(1.0)),
                step_size: Some(0.01),
                realtime_safe: false,
            }),
            "hrtf_intensity" => Some(ParameterDefinition {
                name: "hrtf_intensity".to_string(),
                description: "HRTF processing intensity".to_string(),
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

    fn get_latency_samples(&self) -> usize {
        // Spatial processing has minimal latency
        64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_reverb_effect() {
        let reverb = ReverbEffect::new();

        // Test parameter setting
        reverb
            .set_parameter("mix", ParameterValue::Float(0.5))
            .unwrap();
        assert_eq!(*reverb.mix.read().unwrap(), 0.5);

        // Test audio processing
        let audio = crate::AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);
        let processed = reverb.process_audio(&audio).await.unwrap();

        assert_eq!(processed.len(), audio.len());
        assert_eq!(processed.sample_rate(), audio.sample_rate());
    }

    #[tokio::test]
    async fn test_equalizer_effect() {
        let eq = EqualizerEffect::new();

        // Test parameter setting
        eq.set_parameter("low_gain", ParameterValue::Float(3.0))
            .unwrap();
        eq.set_parameter("mid_gain", ParameterValue::Float(-2.0))
            .unwrap();
        eq.set_parameter("high_gain", ParameterValue::Float(1.0))
            .unwrap();

        assert_eq!(*eq.low_gain.read().unwrap(), 3.0);
        assert_eq!(*eq.mid_gain.read().unwrap(), -2.0);
        assert_eq!(*eq.high_gain.read().unwrap(), 1.0);

        // Test audio processing
        let audio = crate::AudioBuffer::sine_wave(1000.0, 0.5, 44100, 0.3);
        let processed = eq.process_audio(&audio).await.unwrap();

        assert_eq!(processed.len(), audio.len());
    }

    #[tokio::test]
    async fn test_compressor_effect() {
        let comp = CompressorEffect::new();

        // Test parameter setting
        comp.set_parameter("threshold", ParameterValue::Float(-18.0))
            .unwrap();
        comp.set_parameter("ratio", ParameterValue::Float(6.0))
            .unwrap();

        assert_eq!(*comp.threshold.read().unwrap(), -18.0);
        assert_eq!(*comp.ratio.read().unwrap(), 6.0);

        // Test audio processing with loud signal
        let audio = crate::AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.9); // Loud signal
        let processed = comp.process_audio(&audio).await.unwrap();

        // Should have reduced the peaks
        let original_peak = audio.samples().iter().map(|&s| s.abs()).fold(0.0, f32::max);
        let processed_peak = processed
            .samples()
            .iter()
            .map(|&s| s.abs())
            .fold(0.0, f32::max);

        assert!(processed_peak <= original_peak);
    }

    #[test]
    fn test_parameter_definitions() {
        let reverb = ReverbEffect::new();
        let mix_def = reverb.get_parameter_definition("mix").unwrap();

        assert_eq!(mix_def.name, "mix");
        assert_eq!(mix_def.parameter_type, ParameterType::Float);
        assert!(mix_def.realtime_safe);

        let room_def = reverb.get_parameter_definition("room_size").unwrap();
        assert!(!room_def.realtime_safe); // Room size changes aren't real-time safe
    }

    #[tokio::test]
    async fn test_delay_effect() {
        let delay = DelayEffect::new();

        // Test parameter setting
        delay
            .set_parameter("delay_ms", ParameterValue::Float(500.0))
            .unwrap();
        delay
            .set_parameter("feedback", ParameterValue::Float(0.6))
            .unwrap();

        assert_eq!(*delay.delay_ms.read().unwrap(), 500.0);
        assert_eq!(*delay.feedback.read().unwrap(), 0.6);

        // Test audio processing
        let audio = crate::AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.5);
        let processed = delay.process_audio(&audio).await.unwrap();

        assert_eq!(processed.len(), audio.len());
        assert_eq!(processed.sample_rate(), audio.sample_rate());
    }

    #[tokio::test]
    async fn test_spatial_audio_effect() {
        let spatial = SpatialAudioEffect::new();

        // Test parameter setting
        spatial
            .set_parameter("azimuth", ParameterValue::Float(45.0))
            .unwrap();
        spatial
            .set_parameter("distance", ParameterValue::Float(2.0))
            .unwrap();

        assert_eq!(*spatial.azimuth.read().unwrap(), 45.0);
        assert_eq!(*spatial.distance.read().unwrap(), 2.0);

        // Test mono to stereo conversion
        let mono_audio = crate::AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.5);
        let processed = spatial.process_audio(&mono_audio).await.unwrap();

        // Should convert mono to stereo
        assert_eq!(processed.channels(), 2);
        assert_eq!(processed.len(), mono_audio.len() * 2);
    }

    #[test]
    fn test_plugin_metadata() {
        let reverb = ReverbEffect::new();
        assert_eq!(reverb.name(), "Reverb");
        assert_eq!(reverb.version(), "1.0.0");
        assert_eq!(reverb.author(), "VoiRS Team");

        let eq = EqualizerEffect::new();
        assert_eq!(eq.name(), "Equalizer");

        let comp = CompressorEffect::new();
        assert_eq!(comp.name(), "Compressor");

        let delay = DelayEffect::new();
        assert_eq!(delay.name(), "Delay");

        let spatial = SpatialAudioEffect::new();
        assert_eq!(spatial.name(), "Spatial Audio");
    }
}
