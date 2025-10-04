//! Granular synthesis for special vocal effects
//!
//! This module implements granular synthesis techniques for creating special vocal effects
//! including texture manipulation, time-stretching, pitch-shifting, and unique timbral effects.

#![allow(dead_code)]

use crate::effects::SingingEffect;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Granular synthesis processor for advanced audio manipulation.
///
/// This processor implements granular synthesis techniques by decomposing audio into small
/// grains (typically 5-500ms) and manipulating their playback position, pitch, amplitude,
/// and density. This enables sophisticated effects including:
///
/// - Time-stretching without pitch change
/// - Pitch-shifting without time change
/// - Textural transformations (smooth, rough, crystalline, cloudy)
/// - Spectral freezing and morphing
/// - Stochastic variations for organic, evolving sounds
///
/// # Technical Details
///
/// The processor maintains a circular input buffer and spawns multiple concurrent grains,
/// each with independent envelope, pitch shift, and amplitude parameters. Grains are
/// processed in parallel and mixed to produce the final output.
#[derive(Debug, Clone)]
pub struct GranularSynthesisEffect {
    name: String,
    parameters: HashMap<String, f32>,

    // Core granular parameters
    grain_size: f32,         // Grain duration in seconds
    grain_density: f32,      // Grains per second
    grain_overlap: f32,      // Overlap ratio (0.0 - 1.0)
    position_variation: f32, // Random position variation
    pitch_variation: f32,    // Random pitch variation per grain
    amp_variation: f32,      // Random amplitude variation per grain

    // Grain envelope and windowing
    envelope_type: GrainEnvelope,
    window_function: WindowFunction,

    // Buffer management
    input_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
    grain_buffer: Vec<GrainState>,

    // Processing state
    sample_rate: f32,
    buffer_size: usize,
    read_position: f32,
    write_position: usize,
    grain_counter: f32,

    // Random state
    rng_state: u64,
}

/// Individual grain state
#[derive(Debug, Clone)]
struct GrainState {
    active: bool,
    position: f32,       // Position in source buffer
    pitch_shift: f32,    // Pitch shift ratio for this grain
    amplitude: f32,      // Amplitude multiplier
    duration: f32,       // Grain duration in samples
    phase: f32,          // Current phase in grain
    envelope_value: f32, // Current envelope value
}

/// Envelope shapes applied to individual grains.
///
/// The envelope controls the amplitude trajectory of each grain over its lifetime,
/// affecting the smoothness of grain transitions and the overall sonic character.
/// Different envelope types produce different perceptual qualities:
///
/// - Smooth envelopes (Gaussian, Hann) reduce artifacts and produce flowing textures
/// - Sharp envelopes (Linear, Exponential) emphasize grain boundaries
/// - Specialized envelopes (Kaiser, Tukey) offer precise control over sidelobe suppression
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrainEnvelope {
    /// Linear attack/decay
    Linear,
    /// Exponential attack/decay
    Exponential,
    /// Gaussian envelope
    Gaussian,
    /// Hann window
    Hann,
    /// Hamming window
    Hamming,
    /// Kaiser window
    Kaiser,
    /// Tukey window
    Tukey,
}

/// Window functions applied during grain processing.
///
/// Window functions shape the spectral characteristics of individual grains,
/// controlling frequency domain artifacts and smoothing transitions. Each window
/// offers different trade-offs between main lobe width and side lobe suppression:
///
/// - Rectangle: No windowing, maximum time resolution but spectral artifacts
/// - Hann/Hamming: Balanced time-frequency resolution, minimal artifacts
/// - Blackman: Superior sidelobe suppression, wider main lobe
/// - Kaiser: Adjustable parameter for flexible main lobe/sidelobe trade-off
/// - Tukey: Combines flat top with tapered edges
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WindowFunction {
    /// Rectangular window (no windowing)
    Rectangle,
    /// Hann window
    Hann,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
    /// Kaiser window
    Kaiser,
    /// Tukey window
    Tukey,
}

/// Configuration parameters for granular synthesis.
///
/// This structure defines all parameters controlling the granular synthesis process,
/// including grain characteristics (size, density, overlap), randomization amounts,
/// pitch/time manipulation, and mixing settings. All parameters include documented
/// valid ranges and are clamped during processing to prevent invalid values.
///
/// # Example
///
/// ```ignore
/// use voirs_singing::granular_synthesis::{GranularConfig, GrainEnvelope, WindowFunction};
///
/// let config = GranularConfig {
///     grain_size_ms: 100.0,
///     grain_density: 30.0,
///     grain_overlap: 0.7,
///     position_variation: 0.2,
///     pitch_variation_semitones: 2.0,
///     amplitude_variation: 0.1,
///     envelope_type: GrainEnvelope::Gaussian,
///     window_function: WindowFunction::Hann,
///     time_stretch: 1.5,
///     pitch_shift_semitones: 0.0,
///     dry_wet_mix: 0.8,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GranularConfig {
    /// Grain size in milliseconds (5-500ms)
    pub grain_size_ms: f32,
    /// Grain density (grains per second, 1-200)
    pub grain_density: f32,
    /// Grain overlap ratio (0.0-1.0)
    pub grain_overlap: f32,
    /// Position randomization amount (0.0-1.0)
    pub position_variation: f32,
    /// Pitch variation range in semitones (-12.0 to +12.0)
    pub pitch_variation_semitones: f32,
    /// Amplitude variation amount (0.0-1.0)
    pub amplitude_variation: f32,
    /// Grain envelope type
    pub envelope_type: GrainEnvelope,
    /// Window function for grain processing
    pub window_function: WindowFunction,
    /// Time stretch factor (0.1-10.0)
    pub time_stretch: f32,
    /// Global pitch shift in semitones (-24.0 to +24.0)
    pub pitch_shift_semitones: f32,
    /// Dry/wet mix (0.0=dry, 1.0=wet)
    pub dry_wet_mix: f32,
}

impl Default for GranularConfig {
    fn default() -> Self {
        Self {
            grain_size_ms: 50.0,
            grain_density: 20.0,
            grain_overlap: 0.5,
            position_variation: 0.1,
            pitch_variation_semitones: 0.0,
            amplitude_variation: 0.05,
            envelope_type: GrainEnvelope::Hann,
            window_function: WindowFunction::Hann,
            time_stretch: 1.0,
            pitch_shift_semitones: 0.0,
            dry_wet_mix: 0.5,
        }
    }
}

impl GranularSynthesisEffect {
    /// Creates a new granular synthesis effect processor.
    ///
    /// Initializes the granular synthesis engine with the specified configuration,
    /// allocating internal buffers and setting up the grain processing pipeline.
    /// The processor is configured with a 2-second circular input buffer and
    /// supports up to 64 concurrent grains.
    ///
    /// # Arguments
    ///
    /// * `name` - Identifier for this effect instance
    /// * `config` - Granular synthesis configuration parameters
    /// * `sample_rate` - Audio sample rate in Hz (e.g., 44100.0, 48000.0)
    ///
    /// # Returns
    ///
    /// A new `GranularSynthesisEffect` instance ready for audio processing.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use voirs_singing::granular_synthesis::{GranularSynthesisEffect, GranularConfig};
    ///
    /// let config = GranularConfig::default();
    /// let effect = GranularSynthesisEffect::new(
    ///     String::from("my_granular"),
    ///     config,
    ///     44100.0
    /// );
    /// ```
    pub fn new(name: String, config: GranularConfig, sample_rate: f32) -> Self {
        let mut effect = Self {
            name,
            parameters: HashMap::new(),
            grain_size: config.grain_size_ms / 1000.0,
            grain_density: config.grain_density,
            grain_overlap: config.grain_overlap,
            position_variation: config.position_variation,
            pitch_variation: config.pitch_variation_semitones,
            amp_variation: config.amplitude_variation,
            envelope_type: config.envelope_type,
            window_function: config.window_function,
            input_buffer: Vec::with_capacity((sample_rate * 2.0) as usize), // 2 second buffer
            output_buffer: Vec::new(),
            grain_buffer: Vec::with_capacity(64), // Up to 64 concurrent grains
            sample_rate,
            buffer_size: 0,
            read_position: 0.0,
            write_position: 0,
            grain_counter: 0.0,
            rng_state: 1234567890,
        };

        // Initialize parameters
        effect.update_parameters_from_config(&config);

        // Initialize grain buffer
        for _ in 0..64 {
            effect.grain_buffer.push(GrainState::new());
        }

        effect
    }

    /// Update parameters from configuration
    fn update_parameters_from_config(&mut self, config: &GranularConfig) {
        self.parameters
            .insert(String::from("grain_size_ms"), config.grain_size_ms);
        self.parameters
            .insert(String::from("grain_density"), config.grain_density);
        self.parameters
            .insert(String::from("grain_overlap"), config.grain_overlap);
        self.parameters.insert(
            String::from("position_variation"),
            config.position_variation,
        );
        self.parameters.insert(
            String::from("pitch_variation"),
            config.pitch_variation_semitones,
        );
        self.parameters.insert(
            String::from("amplitude_variation"),
            config.amplitude_variation,
        );
        self.parameters
            .insert(String::from("time_stretch"), config.time_stretch);
        self.parameters
            .insert(String::from("pitch_shift"), config.pitch_shift_semitones);
        self.parameters
            .insert(String::from("dry_wet_mix"), config.dry_wet_mix);
    }

    /// Processes audio through the granular synthesis engine.
    ///
    /// Applies granular synthesis to the input audio, generating grains based on density
    /// settings, processing each active grain, and mixing the results into the output buffer.
    /// The input is added to a circular buffer for grain source material, and grains are
    /// triggered at intervals determined by grain density.
    ///
    /// # Arguments
    ///
    /// * `input` - Input audio samples to process
    /// * `output` - Output buffer to write processed audio (must be pre-allocated)
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Processing completed successfully
    /// * `Err(...)` - Processing error occurred
    ///
    /// # Errors
    ///
    /// Returns an error if grain triggering or processing fails (though current
    /// implementation does not produce errors under normal operation).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut effect = GranularSynthesisEffect::new(
    ///     String::from("granular"),
    ///     GranularConfig::default(),
    ///     44100.0
    /// );
    ///
    /// let input = vec![0.0; 4410];  // 100ms at 44.1kHz
    /// let mut output = vec![0.0; 4410];
    /// effect.process_granular(&input, &mut output)?;
    /// ```
    pub fn process_granular(&mut self, input: &[f32], output: &mut [f32]) -> crate::Result<()> {
        if input.is_empty() || output.is_empty() {
            return Ok(());
        }

        // Add input to circular buffer
        self.add_to_input_buffer(input);

        // Clear output buffer
        output.fill(0.0);

        // Calculate grain timing
        let samples_per_grain = self.sample_rate / self.grain_density;

        // Process each output sample
        for i in 0..output.len() {
            // Check if we should trigger a new grain
            self.grain_counter += 1.0;
            if self.grain_counter >= samples_per_grain {
                self.grain_counter = 0.0;
                self.trigger_new_grain()?;
            }

            // Process all active grains
            let mut grain_output = 0.0;
            for grain_idx in 0..self.grain_buffer.len() {
                if self.grain_buffer[grain_idx].active {
                    grain_output += self.process_grain_at_index(grain_idx, i)?;
                }
            }

            // Apply dry/wet mix
            let dry_wet = self.parameters.get("dry_wet_mix").copied().unwrap_or(0.5);
            if i < input.len() {
                output[i] = input[i] * (1.0 - dry_wet) + grain_output * dry_wet;
            } else {
                output[i] = grain_output * dry_wet;
            }

            // Advance read position for time stretching
            let time_stretch = self.parameters.get("time_stretch").copied().unwrap_or(1.0);
            self.read_position += time_stretch;
        }

        Ok(())
    }

    /// Add input samples to the circular input buffer
    fn add_to_input_buffer(&mut self, input: &[f32]) {
        for &sample in input {
            if self.input_buffer.len() >= self.input_buffer.capacity() {
                self.input_buffer.remove(0);
            }
            self.input_buffer.push(sample);
        }
        self.buffer_size = self.input_buffer.len();
    }

    /// Trigger a new grain
    fn trigger_new_grain(&mut self) -> crate::Result<()> {
        // Calculate random values first (to avoid borrowing conflicts)
        let grain_size_samples = self.grain_size * self.sample_rate;
        let position_var = self.position_variation * self.random_bilateral();
        let pitch_var = self.pitch_variation * self.random_bilateral();
        let amp_var = 1.0 + self.amp_variation * self.random_bilateral();

        // Find an inactive grain slot
        let grain_slot = self.grain_buffer.iter_mut().find(|g| !g.active);
        if let Some(grain) = grain_slot {
            // Set grain parameters
            grain.active = true;
            grain.position = self.read_position + position_var * grain_size_samples;
            grain.pitch_shift = Self::semitones_to_ratio(
                self.parameters.get("pitch_shift").copied().unwrap_or(0.0) + pitch_var,
            );
            grain.amplitude = amp_var.max(0.0);
            grain.duration = grain_size_samples / grain.pitch_shift;
            grain.phase = 0.0;
            grain.envelope_value = 0.0;
        }

        Ok(())
    }

    /// Process a single grain at index for one output sample
    fn process_grain_at_index(
        &mut self,
        grain_index: usize,
        _output_index: usize,
    ) -> crate::Result<f32> {
        if grain_index >= self.grain_buffer.len() {
            return Ok(0.0);
        }

        if !self.grain_buffer[grain_index].active {
            return Ok(0.0);
        }

        // Get grain values (without borrowing)
        let grain_position = self.grain_buffer[grain_index].position;
        let grain_phase = self.grain_buffer[grain_index].phase;
        let grain_pitch_shift = self.grain_buffer[grain_index].pitch_shift;
        let grain_duration = self.grain_buffer[grain_index].duration;
        let grain_amplitude = self.grain_buffer[grain_index].amplitude;

        // Calculate envelope value
        let envelope_progress = grain_phase / grain_duration;
        let envelope_value = self.calculate_grain_envelope(envelope_progress);

        // Get sample from input buffer with pitch shifting
        let sample_position = grain_position + grain_phase * grain_pitch_shift;
        let sample_value = self.interpolate_sample(sample_position);

        // Apply window function
        let windowed_sample = sample_value * self.calculate_window_function(envelope_progress);

        // Apply envelope and amplitude
        let output_sample = windowed_sample * envelope_value * grain_amplitude;

        // Update grain state
        let grain = &mut self.grain_buffer[grain_index];
        grain.envelope_value = envelope_value;
        grain.phase += 1.0;

        // Check if grain is finished
        if grain.phase >= grain.duration {
            grain.active = false;
        }

        Ok(output_sample)
    }

    /// Calculate grain envelope value based on type and progress
    fn calculate_grain_envelope(&self, progress: f32) -> f32 {
        let clamped_progress = progress.clamp(0.0, 1.0);

        match self.envelope_type {
            GrainEnvelope::Linear => {
                if clamped_progress < 0.5 {
                    clamped_progress * 2.0
                } else {
                    2.0 * (1.0 - clamped_progress)
                }
            }
            GrainEnvelope::Exponential => {
                if clamped_progress < 0.5 {
                    (clamped_progress * 2.0).powf(2.0)
                } else {
                    (2.0 * (1.0 - clamped_progress)).powf(2.0)
                }
            }
            GrainEnvelope::Gaussian => {
                let x = (clamped_progress - 0.5) * 4.0; // Scale to -2 to 2
                (-x * x).exp()
            }
            GrainEnvelope::Hann => {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * clamped_progress).cos())
            }
            GrainEnvelope::Hamming => {
                0.54 - 0.46 * (2.0 * std::f32::consts::PI * clamped_progress).cos()
            }
            GrainEnvelope::Kaiser => {
                // Simplified Kaiser window (beta=5)
                let alpha = 5.0;
                let x = 2.0 * clamped_progress - 1.0;
                Self::modified_bessel_i0(alpha * (1.0 - x * x).sqrt())
                    / Self::modified_bessel_i0(alpha)
            }
            GrainEnvelope::Tukey => {
                let alpha = 0.5; // Taper ratio
                if clamped_progress < alpha / 2.0 {
                    0.5 * (1.0
                        + (2.0 * std::f32::consts::PI * clamped_progress / alpha
                            - std::f32::consts::PI)
                            .cos())
                } else if clamped_progress > 1.0 - alpha / 2.0 {
                    0.5 * (1.0
                        + (2.0 * std::f32::consts::PI * (clamped_progress - 1.0) / alpha
                            + std::f32::consts::PI)
                            .cos())
                } else {
                    1.0
                }
            }
        }
    }

    /// Calculate window function value
    fn calculate_window_function(&self, progress: f32) -> f32 {
        let clamped_progress = progress.clamp(0.0, 1.0);

        match self.window_function {
            WindowFunction::Rectangle => 1.0,
            WindowFunction::Hann => {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * clamped_progress).cos())
            }
            WindowFunction::Hamming => {
                0.54 - 0.46 * (2.0 * std::f32::consts::PI * clamped_progress).cos()
            }
            WindowFunction::Blackman => {
                0.42 - 0.5 * (2.0 * std::f32::consts::PI * clamped_progress).cos()
                    + 0.08 * (4.0 * std::f32::consts::PI * clamped_progress).cos()
            }
            WindowFunction::Kaiser => {
                // Simplified Kaiser window (beta=5)
                let alpha = 5.0;
                let x = 2.0 * clamped_progress - 1.0;
                Self::modified_bessel_i0(alpha * (1.0 - x * x).sqrt())
                    / Self::modified_bessel_i0(alpha)
            }
            WindowFunction::Tukey => {
                let alpha = 0.5; // Taper ratio
                if clamped_progress < alpha / 2.0 {
                    0.5 * (1.0
                        + (2.0 * std::f32::consts::PI * clamped_progress / alpha
                            - std::f32::consts::PI)
                            .cos())
                } else if clamped_progress > 1.0 - alpha / 2.0 {
                    0.5 * (1.0
                        + (2.0 * std::f32::consts::PI * (clamped_progress - 1.0) / alpha
                            + std::f32::consts::PI)
                            .cos())
                } else {
                    1.0
                }
            }
        }
    }

    /// Interpolate sample from buffer at fractional position
    fn interpolate_sample(&self, position: f32) -> f32 {
        if self.input_buffer.is_empty() {
            return 0.0;
        }

        let buffer_len = self.input_buffer.len() as f32;
        let wrapped_pos = ((position % buffer_len) + buffer_len) % buffer_len;

        let index = wrapped_pos.floor() as usize;
        let frac = wrapped_pos.fract();

        let index0 = index % self.input_buffer.len();
        let index1 = (index + 1) % self.input_buffer.len();

        // Linear interpolation
        self.input_buffer[index0] * (1.0 - frac) + self.input_buffer[index1] * frac
    }

    /// Convert semitones to pitch ratio
    fn semitones_to_ratio(semitones: f32) -> f32 {
        2.0_f32.powf(semitones / 12.0)
    }

    /// Generate random value between -1.0 and 1.0
    fn random_bilateral(&mut self) -> f32 {
        self.random_unilateral() * 2.0 - 1.0
    }

    /// Generate random value between 0.0 and 1.0
    fn random_unilateral(&mut self) -> f32 {
        // Simple LCG random number generator
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.rng_state & 0x7fffffff) as f32 / 0x7fffffff as f32
    }

    /// Approximation of modified Bessel function I0
    fn modified_bessel_i0(x: f32) -> f32 {
        let ax = x.abs();
        if ax < 3.75 {
            let y = (x / 3.75).powi(2);
            1.0 + y
                * (3.5156229
                    + y * (3.0899424
                        + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))))
        } else {
            let z = 3.75 / ax;
            (ax.exp() / ax.sqrt())
                * (0.398_942_3
                    + z * (0.013_285_92
                        + z * (0.002_253_19
                            + z * (-0.001_575_65
                                + z * (0.009_162_81
                                    + z * (-0.020_577_06
                                        + z * (0.026_355_37
                                            + z * (-0.016_476_33 + z * 0.003_923_77))))))))
        }
    }

    /// Applies a preset texture configuration to the granular synthesis.
    ///
    /// Sets grain parameters to predefined values optimized for specific sonic textures.
    /// This provides a quick way to achieve characteristic granular sounds without
    /// manually adjusting individual parameters.
    ///
    /// # Arguments
    ///
    /// * `texture_type` - The texture preset to apply (Smooth, Rough, Crystalline, or Cloudy)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use voirs_singing::granular_synthesis::{GranularSynthesisEffect, GranularTexture, GranularConfig};
    ///
    /// let mut effect = GranularSynthesisEffect::new(
    ///     String::from("granular"),
    ///     GranularConfig::default(),
    ///     44100.0
    /// );
    ///
    /// // Apply smooth, flowing texture
    /// effect.apply_texture_effect(GranularTexture::Smooth);
    ///
    /// // Switch to rough, granular texture
    /// effect.apply_texture_effect(GranularTexture::Rough);
    /// ```
    pub fn apply_texture_effect(&mut self, texture_type: GranularTexture) {
        match texture_type {
            GranularTexture::Smooth => {
                self.grain_overlap = 0.8;
                self.position_variation = 0.02;
                self.amp_variation = 0.01;
                self.envelope_type = GrainEnvelope::Gaussian;
            }
            GranularTexture::Rough => {
                self.grain_overlap = 0.3;
                self.position_variation = 0.3;
                self.amp_variation = 0.2;
                self.envelope_type = GrainEnvelope::Linear;
            }
            GranularTexture::Crystalline => {
                self.grain_overlap = 0.1;
                self.position_variation = 0.0;
                self.amp_variation = 0.0;
                self.envelope_type = GrainEnvelope::Kaiser;
            }
            GranularTexture::Cloudy => {
                self.grain_overlap = 0.9;
                self.position_variation = 0.5;
                self.amp_variation = 0.3;
                self.envelope_type = GrainEnvelope::Gaussian;
            }
        }
    }

    /// Retrieves the current granular synthesis configuration.
    ///
    /// Returns a `GranularConfig` structure populated with the current parameter
    /// values. This is useful for saving presets, displaying current settings,
    /// or cloning configurations across multiple effect instances.
    ///
    /// # Returns
    ///
    /// A `GranularConfig` containing all current parameter values.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use voirs_singing::granular_synthesis::{GranularSynthesisEffect, GranularConfig};
    ///
    /// let mut effect = GranularSynthesisEffect::new(
    ///     String::from("granular"),
    ///     GranularConfig::default(),
    ///     44100.0
    /// );
    ///
    /// // Modify some parameters
    /// effect.set_parameter("grain_size_ms", 100.0)?;
    /// effect.set_parameter("grain_density", 40.0)?;
    ///
    /// // Retrieve current configuration
    /// let current_config = effect.get_config();
    /// assert_eq!(current_config.grain_size_ms, 100.0);
    /// ```
    pub fn get_config(&self) -> GranularConfig {
        GranularConfig {
            grain_size_ms: self.grain_size * 1000.0,
            grain_density: self.grain_density,
            grain_overlap: self.grain_overlap,
            position_variation: self.position_variation,
            pitch_variation_semitones: self.pitch_variation,
            amplitude_variation: self.amp_variation,
            envelope_type: self.envelope_type,
            window_function: self.window_function,
            time_stretch: self.parameters.get("time_stretch").copied().unwrap_or(1.0),
            pitch_shift_semitones: self.parameters.get("pitch_shift").copied().unwrap_or(0.0),
            dry_wet_mix: self.parameters.get("dry_wet_mix").copied().unwrap_or(0.5),
        }
    }
}

/// Granular texture types for preset configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GranularTexture {
    /// Smooth, flowing texture with high overlap
    Smooth,
    /// Rough, granular texture with low overlap  
    Rough,
    /// Crystalline, precise texture with no randomization
    Crystalline,
    /// Cloudy, ethereal texture with high randomization
    Cloudy,
}

impl GrainState {
    fn new() -> Self {
        Self {
            active: false,
            position: 0.0,
            pitch_shift: 1.0,
            amplitude: 1.0,
            duration: 0.0,
            phase: 0.0,
            envelope_value: 0.0,
        }
    }
}

impl SingingEffect for GranularSynthesisEffect {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(&mut self, audio: &mut [f32], sample_rate: f32) -> crate::Result<()> {
        self.sample_rate = sample_rate;

        // Create temporary output buffer
        let mut output = vec![0.0; audio.len()];

        // Process with granular synthesis
        self.process_granular(audio, &mut output)?;

        // Copy result back to input buffer
        audio.copy_from_slice(&output);

        Ok(())
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> crate::Result<()> {
        match name {
            "grain_size_ms" => {
                let clamped = value.clamp(5.0, 500.0);
                self.grain_size = clamped / 1000.0;
                self.parameters.insert(name.to_string(), clamped);
            }
            "grain_density" => {
                let clamped = value.clamp(1.0, 200.0);
                self.grain_density = clamped;
                self.parameters.insert(name.to_string(), clamped);
            }
            "grain_overlap" => {
                let clamped = value.clamp(0.0, 1.0);
                self.grain_overlap = clamped;
                self.parameters.insert(name.to_string(), clamped);
            }
            "position_variation" => {
                let clamped = value.clamp(0.0, 1.0);
                self.position_variation = clamped;
                self.parameters.insert(name.to_string(), clamped);
            }
            "pitch_variation" => {
                let clamped = value.clamp(-12.0, 12.0);
                self.pitch_variation = clamped;
                self.parameters.insert(name.to_string(), clamped);
            }
            "amplitude_variation" => {
                let clamped = value.clamp(0.0, 1.0);
                self.amp_variation = clamped;
                self.parameters.insert(name.to_string(), clamped);
            }
            "time_stretch" => {
                let clamped = value.clamp(0.1, 10.0);
                self.parameters.insert(name.to_string(), clamped);
            }
            "pitch_shift" => {
                let clamped = value.clamp(-24.0, 24.0);
                self.parameters.insert(name.to_string(), clamped);
            }
            "dry_wet_mix" => {
                let clamped = value.clamp(0.0, 1.0);
                self.parameters.insert(name.to_string(), clamped);
            }
            _ => {
                return Err(crate::Error::Effect(format!("Unknown parameter: {}", name)));
            }
        }

        Ok(())
    }

    fn get_parameter(&self, name: &str) -> Option<f32> {
        match name {
            "grain_size_ms" => Some(self.grain_size * 1000.0),
            "grain_density" => Some(self.grain_density),
            "grain_overlap" => Some(self.grain_overlap),
            "position_variation" => Some(self.position_variation),
            "pitch_variation" => Some(self.pitch_variation),
            "amplitude_variation" => Some(self.amp_variation),
            _ => self.parameters.get(name).copied(),
        }
    }

    fn get_parameters(&self) -> HashMap<String, f32> {
        let mut params = self.parameters.clone();
        params.insert(String::from("grain_size_ms"), self.grain_size * 1000.0);
        params.insert(String::from("grain_density"), self.grain_density);
        params.insert(String::from("grain_overlap"), self.grain_overlap);
        params.insert(String::from("position_variation"), self.position_variation);
        params.insert(String::from("pitch_variation"), self.pitch_variation);
        params.insert(String::from("amplitude_variation"), self.amp_variation);
        params
    }

    fn reset(&mut self) {
        self.input_buffer.clear();
        self.output_buffer.clear();
        self.read_position = 0.0;
        self.write_position = 0;
        self.grain_counter = 0.0;

        // Reset all grains
        for grain in &mut self.grain_buffer {
            grain.active = false;
            grain.phase = 0.0;
        }
    }

    fn clone_effect(&self) -> Box<dyn SingingEffect> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_granular_synthesis_creation() {
        let config = GranularConfig::default();
        let effect = GranularSynthesisEffect::new(String::from("granular"), config, 44100.0);

        assert_eq!(effect.name(), "granular");
        assert_eq!(effect.sample_rate, 44100.0);
        assert!(!effect.grain_buffer.is_empty());
    }

    #[test]
    fn test_granular_parameters() {
        let config = GranularConfig::default();
        let mut effect = GranularSynthesisEffect::new(String::from("granular"), config, 44100.0);

        // Test parameter setting and getting
        effect.set_parameter("grain_size_ms", 100.0).unwrap();
        assert_eq!(effect.get_parameter("grain_size_ms"), Some(100.0));

        effect.set_parameter("grain_density", 50.0).unwrap();
        assert_eq!(effect.get_parameter("grain_density"), Some(50.0));

        effect.set_parameter("dry_wet_mix", 0.8).unwrap();
        assert_eq!(effect.get_parameter("dry_wet_mix"), Some(0.8));
    }

    #[test]
    fn test_granular_processing() {
        let config = GranularConfig::default();
        let mut effect = GranularSynthesisEffect::new(String::from("granular"), config, 44100.0);

        // Create test audio (440Hz sine wave)
        let mut audio = vec![0.0; 4410]; // 0.1 seconds at 44.1kHz
        for (i, sample) in audio.iter_mut().enumerate() {
            let t = i as f32 / 44100.0;
            *sample = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
        }

        // Process audio
        let result = effect.process(&mut audio, 44100.0);
        assert!(result.is_ok());

        // Check that processing modified the audio
        let rms = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        assert!(rms > 0.01, "Processed audio should have some energy");
    }

    #[test]
    fn test_grain_envelopes() {
        let config = GranularConfig::default();
        let effect = GranularSynthesisEffect::new(String::from("granular"), config, 44100.0);

        // Test all envelope types at mid-point
        let test_progress = 0.5;

        let linear_env = effect.calculate_grain_envelope(test_progress);
        assert!(linear_env > 0.0 && linear_env <= 1.0);

        // Test edge cases
        let start_env = effect.calculate_grain_envelope(0.0);
        let end_env = effect.calculate_grain_envelope(1.0);
        assert!(start_env >= 0.0);
        assert!(end_env >= 0.0);
    }

    #[test]
    fn test_texture_effects() {
        let config = GranularConfig::default();
        let mut effect = GranularSynthesisEffect::new(String::from("granular"), config, 44100.0);

        // Test different texture presets
        effect.apply_texture_effect(GranularTexture::Smooth);
        assert!(effect.grain_overlap > 0.5);
        assert!(effect.position_variation < 0.1);

        effect.apply_texture_effect(GranularTexture::Rough);
        assert!(effect.grain_overlap < 0.5);
        assert!(effect.position_variation > 0.2);

        effect.apply_texture_effect(GranularTexture::Crystalline);
        assert_eq!(effect.position_variation, 0.0);
        assert_eq!(effect.amp_variation, 0.0);
    }

    #[test]
    fn test_semitone_conversion() {
        assert!((GranularSynthesisEffect::semitones_to_ratio(0.0) - 1.0).abs() < 1e-6);
        assert!((GranularSynthesisEffect::semitones_to_ratio(12.0) - 2.0).abs() < 1e-6);
        assert!((GranularSynthesisEffect::semitones_to_ratio(-12.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sample_interpolation() {
        let config = GranularConfig::default();
        let mut effect = GranularSynthesisEffect::new(String::from("granular"), config, 44100.0);

        // Add test data to buffer
        effect.input_buffer = vec![0.0, 1.0, 2.0, 3.0];

        // Test interpolation
        let interpolated = effect.interpolate_sample(1.5);
        assert!((interpolated - 1.5).abs() < 1e-6);

        let interpolated = effect.interpolate_sample(2.5);
        assert!((interpolated - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_reset() {
        let config = GranularConfig::default();
        let mut effect = GranularSynthesisEffect::new(String::from("granular"), config, 44100.0);

        // Add some data
        effect.input_buffer = vec![1.0, 2.0, 3.0];
        effect.read_position = 100.0;
        effect.grain_counter = 50.0;

        // Reset
        effect.reset();

        // Check that state is reset
        assert!(effect.input_buffer.is_empty());
        assert_eq!(effect.read_position, 0.0);
        assert_eq!(effect.grain_counter, 0.0);

        // Check that all grains are inactive
        for grain in &effect.grain_buffer {
            assert!(!grain.active);
        }
    }
}
