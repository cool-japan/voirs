//! Spatial audio effects for stereo enhancement and ambience.
//!
//! This module implements reverb, stereo width control, and spatial positioning
//! for enhanced audio experience.

use super::{AudioEffect, EffectParameter};
use crate::{AudioBuffer, Result};
use std::f32::consts::PI;

/// Simple all-pass filter for reverb
#[derive(Debug, Clone)]
struct AllPassFilter {
    buffer: Vec<f32>,
    index: usize,
    gain: f32,
}

impl AllPassFilter {
    pub fn new(delay_samples: usize, gain: f32) -> Self {
        Self {
            buffer: vec![0.0; delay_samples.max(1)],
            index: 0,
            gain,
        }
    }

    pub fn process(&mut self, input: f32) -> f32 {
        let delayed = self.buffer[self.index];
        let output = -self.gain * input + delayed;

        self.buffer[self.index] = input + self.gain * delayed;
        self.index = (self.index + 1) % self.buffer.len();

        output
    }

    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.index = 0;
    }
}

/// Simple delay line for reverb
#[derive(Debug, Clone)]
struct DelayLine {
    buffer: Vec<f32>,
    index: usize,
}

impl DelayLine {
    pub fn new(delay_samples: usize) -> Self {
        Self {
            buffer: vec![0.0; delay_samples.max(1)],
            index: 0,
        }
    }

    pub fn process(&mut self, input: f32, delay_samples: usize) -> f32 {
        let read_index = (self.index + self.buffer.len() - delay_samples.min(self.buffer.len()))
            % self.buffer.len();
        let output = self.buffer[read_index];

        self.buffer[self.index] = input;
        self.index = (self.index + 1) % self.buffer.len();

        output
    }

    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.index = 0;
    }
}

/// Schroeder reverb implementation
pub struct Reverb {
    enabled: bool,
    room_size: EffectParameter, // Room size (0.0 - 1.0)
    damping: EffectParameter,   // High frequency damping
    wet_level: EffectParameter, // Wet signal level
    dry_level: EffectParameter, // Dry signal level
    width: EffectParameter,     // Stereo width

    // Reverb components
    all_pass_filters: Vec<AllPassFilter>,
    comb_filters: Vec<CombFilter>,
    sample_rate: u32,
}

#[derive(Debug, Clone)]
struct CombFilter {
    buffer: Vec<f32>,
    index: usize,
    feedback: f32,
    damping: f32,
    filter_store: f32,
}

impl CombFilter {
    pub fn new(delay_samples: usize, feedback: f32, damping: f32) -> Self {
        Self {
            buffer: vec![0.0; delay_samples.max(1)],
            index: 0,
            feedback,
            damping,
            filter_store: 0.0,
        }
    }

    pub fn process(&mut self, input: f32) -> f32 {
        let output = self.buffer[self.index];

        // One-pole low-pass filter for damping
        self.filter_store = output * (1.0 - self.damping) + self.filter_store * self.damping;

        self.buffer[self.index] = input + self.filter_store * self.feedback;
        self.index = (self.index + 1) % self.buffer.len();

        output
    }

    pub fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback;
    }

    pub fn set_damping(&mut self, damping: f32) {
        self.damping = damping;
    }

    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.index = 0;
        self.filter_store = 0.0;
    }
}

impl Reverb {
    pub fn new(sample_rate: u32) -> Self {
        let mut reverb = Self {
            enabled: true,
            room_size: EffectParameter::new("room_size", 0.5, 0.0, 1.0),
            damping: EffectParameter::new("damping", 0.5, 0.0, 1.0),
            wet_level: EffectParameter::new("wet_level", 0.3, 0.0, 1.0),
            dry_level: EffectParameter::new("dry_level", 0.7, 0.0, 1.0),
            width: EffectParameter::new("width", 1.0, 0.0, 1.0),

            all_pass_filters: Vec::new(),
            comb_filters: Vec::new(),
            sample_rate,
        };

        reverb.initialize_filters();
        reverb
    }

    fn initialize_filters(&mut self) {
        let sample_rate = self.sample_rate as f32;

        // Comb filter delays (in samples) - tuned for natural reverb
        let comb_delays = [
            (sample_rate * 0.0297) as usize,
            (sample_rate * 0.0371) as usize,
            (sample_rate * 0.0411) as usize,
            (sample_rate * 0.0437) as usize,
            (sample_rate * 0.005) as usize,
            (sample_rate * 0.0017) as usize,
            (sample_rate * 0.0083) as usize,
            (sample_rate * 0.0109) as usize,
        ];

        // All-pass filter delays
        let allpass_delays = [
            (sample_rate * 0.005) as usize,
            (sample_rate * 0.0017) as usize,
            (sample_rate * 0.0083) as usize,
            (sample_rate * 0.0109) as usize,
        ];

        // Initialize comb filters
        for &delay in &comb_delays {
            self.comb_filters.push(CombFilter::new(delay, 0.84, 0.2));
        }

        // Initialize all-pass filters
        for &delay in &allpass_delays {
            self.all_pass_filters.push(AllPassFilter::new(delay, 0.5));
        }

        self.update_parameters();
    }

    fn update_parameters(&mut self) {
        let room_scale = self.room_size.value * 0.28 + 0.7;
        let damping = self.damping.value * 0.4;

        for comb in &mut self.comb_filters {
            comb.set_feedback(room_scale);
            comb.set_damping(damping);
        }
    }

    pub fn set_room_size(&mut self, size: f32) {
        self.room_size.set_value(size);
        self.update_parameters();
    }

    pub fn set_damping(&mut self, damping: f32) {
        self.damping.set_value(damping);
        self.update_parameters();
    }

    pub fn set_wet_level(&mut self, level: f32) {
        self.wet_level.set_value(level);
    }

    pub fn set_dry_level(&mut self, level: f32) {
        self.dry_level.set_value(level);
    }
}

impl AudioEffect for Reverb {
    fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let channels = audio.channels() as usize;
        let samples = audio.samples_mut();

        if channels == 1 {
            // Mono processing
            for sample in samples.iter_mut() {
                let input = *sample;

                // Process through comb filters
                let mut comb_sum = 0.0;
                for comb in &mut self.comb_filters {
                    comb_sum += comb.process(input);
                }

                // Process through all-pass filters
                let mut wet = comb_sum;
                for allpass in &mut self.all_pass_filters {
                    wet = allpass.process(wet);
                }

                // Mix dry and wet signals
                *sample = input * self.dry_level.value + wet * self.wet_level.value;
            }
        } else {
            // Stereo processing
            for i in (0..samples.len()).step_by(2) {
                let left = samples[i];
                let right = samples[i + 1];

                // Mix to mono for reverb input
                let mono = (left + right) * 0.5;

                // Process through comb filters
                let mut comb_sum = 0.0;
                for comb in &mut self.comb_filters {
                    comb_sum += comb.process(mono);
                }

                // Process through all-pass filters
                let mut wet = comb_sum;
                for allpass in &mut self.all_pass_filters {
                    wet = allpass.process(wet);
                }

                // Create stereo wet signal
                let wet_left = wet * self.wet_level.value;
                let wet_right = wet * self.wet_level.value * self.width.value;

                // Mix with dry signal
                samples[i] = left * self.dry_level.value + wet_left;
                samples[i + 1] = right * self.dry_level.value + wet_right;
            }
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Reverb"
    }

    fn reset(&mut self) {
        for allpass in &mut self.all_pass_filters {
            allpass.reset();
        }
        for comb in &mut self.comb_filters {
            comb.reset();
        }
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// Stereo width controller for spatial enhancement
pub struct StereoWidth {
    enabled: bool,
    width: EffectParameter, // Stereo width (0.0 = mono, 1.0 = normal, >1.0 = wide)
    bass_mono: EffectParameter, // Frequency below which to keep mono

    // High-pass filter for bass mono
    highpass_left: super::frequency::BiquadFilter,
    highpass_right: super::frequency::BiquadFilter,
    sample_rate: u32,
}

impl StereoWidth {
    pub fn new(sample_rate: u32) -> Self {
        let mut width_processor = Self {
            enabled: true,
            width: EffectParameter::new("width", 1.0, 0.0, 2.0),
            bass_mono: EffectParameter::new("bass_mono", 120.0, 60.0, 500.0),

            highpass_left: super::frequency::BiquadFilter::new(),
            highpass_right: super::frequency::BiquadFilter::new(),
            sample_rate,
        };

        width_processor.update_filters();
        width_processor
    }

    fn update_filters(&mut self) {
        let sample_rate = self.sample_rate as f32;
        let crossover_freq = self.bass_mono.value;

        self.highpass_left
            .design_highpass(sample_rate, crossover_freq, 0.7);
        self.highpass_right
            .design_highpass(sample_rate, crossover_freq, 0.7);
    }

    pub fn set_width(&mut self, width: f32) {
        self.width.set_value(width);
    }

    pub fn set_bass_mono_frequency(&mut self, frequency: f32) {
        self.bass_mono.set_value(frequency);
        self.update_filters();
    }
}

impl AudioEffect for StereoWidth {
    fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if !self.enabled || audio.channels() != 2 {
            return Ok(());
        }

        let samples = audio.samples_mut();
        let width = self.width.value;

        for i in (0..samples.len()).step_by(2) {
            let left = samples[i];
            let right = samples[i + 1];

            // Calculate mid and side signals
            let mid = (left + right) * 0.5;
            let side = (left - right) * 0.5;

            // Apply width to side signal
            let wide_side = side * width;

            // Filter for bass mono
            let high_left = self.highpass_left.process(left);
            let high_right = self.highpass_right.process(right);
            let low_left = left - high_left;
            let low_right = right - high_right;
            let low_mono = (low_left + low_right) * 0.5;

            // Reconstruct stereo signal
            let new_left = mid + wide_side;
            let new_right = mid - wide_side;

            // Combine with bass mono
            samples[i] = new_left * 0.7 + low_mono * 0.3;
            samples[i + 1] = new_right * 0.7 + low_mono * 0.3;
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "StereoWidth"
    }

    fn reset(&mut self) {
        self.highpass_left.reset();
        self.highpass_right.reset();
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// 3D positioning effect (basic implementation)
pub struct SpatialPositioner {
    enabled: bool,
    azimuth: EffectParameter,   // Horizontal angle (-180 to 180 degrees)
    elevation: EffectParameter, // Vertical angle (-90 to 90 degrees)
    distance: EffectParameter,  // Distance (0.0 to 1.0)

    // HRTF simulation (simplified)
    delay_left: DelayLine,
    delay_right: DelayLine,
    sample_rate: u32,
}

impl SpatialPositioner {
    pub fn new(sample_rate: u32) -> Self {
        let max_delay_samples = (sample_rate as f32 * 0.001) as usize; // 1ms max delay

        Self {
            enabled: true,
            azimuth: EffectParameter::new("azimuth", 0.0, -180.0, 180.0),
            elevation: EffectParameter::new("elevation", 0.0, -90.0, 90.0),
            distance: EffectParameter::new("distance", 0.5, 0.0, 1.0),

            delay_left: DelayLine::new(max_delay_samples),
            delay_right: DelayLine::new(max_delay_samples),
            sample_rate,
        }
    }

    fn calculate_hrtf_parameters(&self) -> (f32, f32, usize, usize) {
        let azimuth_rad = self.azimuth.value * PI / 180.0;
        let distance_factor = 1.0 - self.distance.value * 0.5; // Closer = louder

        // Simple HRTF simulation
        let left_gain = if azimuth_rad < 0.0 {
            distance_factor
        } else {
            distance_factor * (1.0 - azimuth_rad.abs() / PI)
        };

        let right_gain = if azimuth_rad > 0.0 {
            distance_factor
        } else {
            distance_factor * (1.0 - azimuth_rad.abs() / PI)
        };

        // Calculate interaural time difference (ITD)
        let head_radius = 0.0875; // ~8.75cm average head radius
        let sound_speed = 343.0; // m/s
        let itd_seconds = head_radius * azimuth_rad.sin().abs() / sound_speed;
        let itd_samples = (itd_seconds * self.sample_rate as f32) as usize;

        let (left_delay, right_delay) = if azimuth_rad > 0.0 {
            (itd_samples, 0) // Sound from right, delay left ear
        } else {
            (0, itd_samples) // Sound from left, delay right ear
        };

        (left_gain, right_gain, left_delay, right_delay)
    }

    pub fn set_position(&mut self, azimuth: f32, elevation: f32, distance: f32) {
        self.azimuth.set_value(azimuth);
        self.elevation.set_value(elevation);
        self.distance.set_value(distance);
    }
}

impl AudioEffect for SpatialPositioner {
    fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let channels = audio.channels() as usize;
        let (left_gain, right_gain, left_delay, right_delay) = self.calculate_hrtf_parameters();
        let samples = audio.samples_mut();

        if channels == 1 {
            // Convert mono to positioned stereo
            let mono_samples: Vec<f32> = samples.to_vec();
            let mut stereo_samples = Vec::with_capacity(mono_samples.len() * 2);

            for &mono_sample in mono_samples.iter() {
                let delayed_left = self.delay_left.process(mono_sample, left_delay);
                let delayed_right = self.delay_right.process(mono_sample, right_delay);

                stereo_samples.push(delayed_left * left_gain);
                stereo_samples.push(delayed_right * right_gain);
            }

            // Update audio buffer to stereo
            *audio = AudioBuffer::new(stereo_samples, audio.sample_rate(), 2);
        } else if channels == 2 {
            // Process stereo input
            for i in (0..samples.len()).step_by(2) {
                let input = (samples[i] + samples[i + 1]) * 0.5; // Mix to mono first

                let delayed_left = self.delay_left.process(input, left_delay);
                let delayed_right = self.delay_right.process(input, right_delay);

                samples[i] = delayed_left * left_gain;
                samples[i + 1] = delayed_right * right_gain;
            }
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "SpatialPositioner"
    }

    fn reset(&mut self) {
        self.delay_left.reset();
        self.delay_right.reset();
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}
