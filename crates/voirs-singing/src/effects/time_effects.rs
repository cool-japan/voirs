//! Time-based audio effects for singing synthesis

use super::core::SingingEffect;
use super::filters::{AllPassFilter, DelayLine, HighPassFilter, LowPassFilter};
use super::helpers::LFO;
use std::collections::HashMap;

/// Reverb effect
#[derive(Debug, Clone)]
pub struct ReverbEffect {
    name: String,
    parameters: HashMap<String, f32>,
    delay_lines: Vec<DelayLine>,
    all_pass_filters: Vec<AllPassFilter>,
    low_pass_filter: LowPassFilter,
    high_pass_filter: HighPassFilter,
}

impl ReverbEffect {
    /// Creates a new reverb effect for simulating acoustic spaces.
    ///
    /// # Arguments
    ///
    /// * `parameters` - Effect parameters including room size, damping, wet/dry mix, and width
    ///
    /// # Returns
    ///
    /// A new `ReverbEffect` instance with Schroeder reverb topology using delay lines and all-pass filters.
    pub fn new(mut parameters: HashMap<String, f32>) -> Self {
        let mut effect = Self {
            name: "reverb".to_string(),
            parameters: HashMap::new(),
            delay_lines: Vec::new(),
            all_pass_filters: Vec::new(),
            low_pass_filter: LowPassFilter::new(8000.0, 0.7),
            high_pass_filter: HighPassFilter::new(200.0, 0.7),
        };

        // Initialize default parameters
        effect.parameters.insert("room_size".to_string(), 0.7);
        effect.parameters.insert("damping".to_string(), 0.5);
        effect.parameters.insert("wet".to_string(), 0.3);
        effect.parameters.insert("dry".to_string(), 0.7);
        effect.parameters.insert("width".to_string(), 1.0);

        // Override with provided parameters
        for (key, value) in parameters.drain() {
            effect.parameters.insert(key, value);
        }

        effect.initialize_delay_lines();
        effect
    }

    fn initialize_delay_lines(&mut self) {
        // Schroeder reverb topology - simplified
        let delay_times = vec![1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116]; // Sample delays

        for &delay_time in &delay_times {
            self.delay_lines
                .push(DelayLine::new(delay_time, delay_time as f32, 0.8));
        }

        // All-pass filters for diffusion
        let allpass_delays = vec![225, 556, 441, 341];
        for &delay_time in &allpass_delays {
            self.all_pass_filters
                .push(AllPassFilter::new(delay_time, 0.7));
        }
    }
}

impl SingingEffect for ReverbEffect {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(&mut self, audio: &mut [f32], sample_rate: f32) -> crate::Result<()> {
        let wet = *self.parameters.get("wet").unwrap_or(&0.3);
        let dry = *self.parameters.get("dry").unwrap_or(&0.7);
        let room_size = *self.parameters.get("room_size").unwrap_or(&0.7);

        for sample in audio.iter_mut() {
            let input = *sample;

            // High-pass filter input
            let filtered_input = self.high_pass_filter.process(input, sample_rate);

            // Process through delay lines
            let mut reverb_sum = 0.0;
            for delay_line in &mut self.delay_lines {
                reverb_sum += delay_line.process(filtered_input);
            }

            // Process through all-pass filters for diffusion
            let mut diffused = reverb_sum;
            for all_pass in &mut self.all_pass_filters {
                diffused = all_pass.process(diffused);
            }

            // Low-pass filter for damping
            let damped = self.low_pass_filter.process(diffused, sample_rate);

            // Mix dry and wet signals
            *sample = input * dry + damped * wet * room_size;
        }

        Ok(())
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> crate::Result<()> {
        self.parameters.insert(name.to_string(), value);

        // Update filter parameters if needed
        if name == "damping" {
            let freq = 20000.0 * (1.0 - value);
            self.low_pass_filter.set_cutoff(freq);
        }

        Ok(())
    }

    fn get_parameter(&self, name: &str) -> Option<f32> {
        self.parameters.get(name).copied()
    }

    fn get_parameters(&self) -> HashMap<String, f32> {
        self.parameters.clone()
    }

    fn reset(&mut self) {
        for delay_line in &mut self.delay_lines {
            delay_line.clear();
        }
        for all_pass in &mut self.all_pass_filters {
            all_pass.clear();
        }
        self.low_pass_filter.reset();
        self.high_pass_filter.reset();
    }

    fn clone_effect(&self) -> Box<dyn SingingEffect> {
        Box::new(self.clone())
    }
}

/// Chorus effect
#[derive(Debug, Clone)]
pub struct ChorusEffect {
    name: String,
    parameters: HashMap<String, f32>,
    delay_line: DelayLine,
    lfo: LFO,
    sample_rate: f32,
}

impl ChorusEffect {
    /// Creates a new chorus effect for thickening vocal texture.
    ///
    /// # Arguments
    ///
    /// * `parameters` - Effect parameters including rate, depth, mix, and feedback
    ///
    /// # Returns
    ///
    /// A new `ChorusEffect` instance with modulated delay line and LFO.
    pub fn new(mut parameters: HashMap<String, f32>) -> Self {
        let mut effect = Self {
            name: "chorus".to_string(),
            parameters: HashMap::new(),
            delay_line: DelayLine::new(1024, 512.0, 0.5),
            lfo: LFO::new(0.5, 0.5, 44100.0),
            sample_rate: 44100.0,
        };

        // Initialize default parameters
        effect.parameters.insert("rate".to_string(), 0.5);
        effect.parameters.insert("depth".to_string(), 0.5);
        effect.parameters.insert("mix".to_string(), 0.5);
        effect.parameters.insert("feedback".to_string(), 0.3);

        // Override with provided parameters
        for (key, value) in parameters.drain() {
            effect.parameters.insert(key, value);
        }

        effect
    }
}

impl SingingEffect for ChorusEffect {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(&mut self, audio: &mut [f32], sample_rate: f32) -> crate::Result<()> {
        self.sample_rate = sample_rate;

        let mix = *self.parameters.get("mix").unwrap_or(&0.5);
        let depth = *self.parameters.get("depth").unwrap_or(&0.5);

        for sample in audio.iter_mut() {
            let lfo_value = self.lfo.process();
            let delay_samples = 20.0 + lfo_value * depth * 10.0; // Variable delay

            self.delay_line.set_delay(delay_samples);
            let delayed = self.delay_line.process(*sample);

            *sample = *sample * (1.0 - mix) + delayed * mix;
        }

        Ok(())
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> crate::Result<()> {
        self.parameters.insert(name.to_string(), value);

        match name {
            "rate" => self.lfo.set_frequency(value),
            "depth" => {
                self.lfo.set_amplitude(value);
            }
            "feedback" => {
                self.delay_line.set_feedback(value);
            }
            _ => {}
        }

        Ok(())
    }

    fn get_parameter(&self, name: &str) -> Option<f32> {
        self.parameters.get(name).copied()
    }

    fn get_parameters(&self) -> HashMap<String, f32> {
        self.parameters.clone()
    }

    fn reset(&mut self) {
        self.delay_line.clear();
        self.lfo.reset();
    }

    fn clone_effect(&self) -> Box<dyn SingingEffect> {
        Box::new(self.clone())
    }
}

/// Vibrato effect
#[derive(Debug, Clone)]
pub struct VibratoEffect {
    name: String,
    parameters: HashMap<String, f32>,
    lfo: LFO,
    delay_line: DelayLine,
    sample_rate: f32,
}

impl VibratoEffect {
    /// Creates a new vibrato effect for pitch modulation.
    ///
    /// # Arguments
    ///
    /// * `parameters` - Effect parameters including rate, depth, and intensity
    ///
    /// # Returns
    ///
    /// A new `VibratoEffect` instance with LFO-controlled delay line for pitch variation.
    pub fn new(mut parameters: HashMap<String, f32>) -> Self {
        let mut effect = Self {
            name: "vibrato".to_string(),
            parameters: HashMap::new(),
            lfo: LFO::new(6.0, 0.02, 44100.0),
            delay_line: DelayLine::new(1024, 512.0, 0.0), // No feedback for vibrato
            sample_rate: 44100.0,
        };

        // Initialize default parameters
        effect.parameters.insert("rate".to_string(), 6.0);
        effect.parameters.insert("depth".to_string(), 0.02);
        effect.parameters.insert("intensity".to_string(), 1.0);

        // Override with provided parameters
        for (key, value) in parameters.drain() {
            effect.parameters.insert(key, value);
        }

        effect
    }
}

impl SingingEffect for VibratoEffect {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(&mut self, audio: &mut [f32], sample_rate: f32) -> crate::Result<()> {
        self.sample_rate = sample_rate;

        let intensity = *self.parameters.get("intensity").unwrap_or(&1.0);

        for sample in audio.iter_mut() {
            let lfo_value = self.lfo.process();
            let delay_samples = 10.0 + lfo_value * 5.0; // Variable delay for pitch modulation

            self.delay_line.set_delay(delay_samples);
            let delayed = self.delay_line.process(*sample);

            *sample = *sample * (1.0 - intensity) + delayed * intensity;
        }

        Ok(())
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> crate::Result<()> {
        self.parameters.insert(name.to_string(), value);

        match name {
            "rate" => self.lfo.set_frequency(value),
            "depth" => self.lfo.set_amplitude(value),
            _ => {}
        }

        Ok(())
    }

    fn get_parameter(&self, name: &str) -> Option<f32> {
        self.parameters.get(name).copied()
    }

    fn get_parameters(&self) -> HashMap<String, f32> {
        self.parameters.clone()
    }

    fn reset(&mut self) {
        self.delay_line.clear();
        self.lfo.reset();
    }

    fn clone_effect(&self) -> Box<dyn SingingEffect> {
        Box::new(self.clone())
    }
}
