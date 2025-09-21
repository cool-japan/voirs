//! Core physical modeling components for vocal tract synthesis
//!
//! This module contains the fundamental structures and implementations for physical modeling
//! of the vocal tract, including the main VocalTractModel, GlottalModel, and DelayLine components.

use crate::effects::SingingEffect;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Formant target for articulatory control
#[derive(Debug, Clone, Copy)]
pub struct FormantTarget {
    pub frequency: f32, // Target formant frequency (Hz)
    pub bandwidth: f32, // Target formant bandwidth (Hz)
    pub amplitude: f32, // Target formant amplitude (dB)
}

/// Physical vocal tract model
#[derive(Debug, Clone)]
pub struct VocalTractModel {
    pub name: String,
    pub parameters: HashMap<String, f32>,

    // Vocal tract geometry
    pub tract_length: f32,       // Total vocal tract length (cm)
    pub area_function: Vec<f32>, // Cross-sectional area function
    pub num_sections: usize,     // Number of tube sections

    // Acoustic properties
    pub sound_speed: f32, // Speed of sound (cm/s)
    pub sample_rate: f32, // Sampling rate

    // Tube model state
    pub delay_lines: Vec<DelayLine>,  // Delay lines for each section
    pub reflectances: Vec<f32>,       // Reflection coefficients
    pub junction_pressures: Vec<f32>, // Pressures at junctions
    pub velocities: Vec<f32>,         // Velocities in sections

    // Excitation source
    pub glottal_model: GlottalModel, // Glottal pulse generator

    // Articulatory parameters
    pub tongue_position: f32, // Tongue position (0-1)
    pub tongue_shape: f32,    // Tongue curvature
    pub jaw_opening: f32,     // Jaw aperture (0-1)
    pub lip_rounding: f32,    // Lip rounding (0-1)
    pub velum_opening: f32,   // Nasal coupling (0-1)

    // Formant control
    pub formant_targets: Vec<FormantTarget>,
}

/// Glottal excitation model
#[derive(Debug, Clone)]
pub struct GlottalModel {
    /// Fundamental frequency
    pub f0: f32,
    /// Glottal pulse shape parameters
    pub open_quotient: f32, // Fraction of period glottis is open
    pub speed_quotient: f32,  // Speed of opening vs closing
    pub pulse_amplitude: f32, // Pulse amplitude
    pub spectral_tilt: f32,   // High frequency rolloff
    pub shimmer: f32,         // Amplitude variation
    pub jitter: f32,          // Frequency variation

    /// Internal state
    pub phase: f32, // Current phase in glottal cycle
    pub last_pulse_amplitude: f32, // Previous pulse for shimmer
    pub rng_state: u64,            // Random state for jitter/shimmer
}

/// Delay line for acoustic tube modeling
#[derive(Debug, Clone)]
pub struct DelayLine {
    buffer: Vec<f32>,
    size: usize,
    read_pos: usize,
    write_pos: usize,
}

/// Physical modeling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalModelConfig {
    /// Vocal tract length in cm (typical: 17.5cm)
    pub tract_length: f32,
    /// Number of tube sections (8-44 typical)
    pub num_sections: usize,
    /// Fundamental frequency in Hz
    pub fundamental_frequency: f32,
    /// Glottal pulse parameters
    pub open_quotient: f32,
    pub speed_quotient: f32,
    pub spectral_tilt: f32,
    /// Articulatory parameters (0-1)
    pub tongue_position: f32,
    pub tongue_shape: f32,
    pub jaw_opening: f32,
    pub lip_rounding: f32,
    pub velum_opening: f32,
    /// Voice quality parameters
    pub breathiness: f32,
    pub roughness: f32,
    /// Control parameters
    pub dry_wet_mix: f32,
}

impl Default for PhysicalModelConfig {
    fn default() -> Self {
        Self {
            tract_length: 17.5,           // Adult male average
            num_sections: 20,             // Good balance of quality vs efficiency
            fundamental_frequency: 220.0, // A3
            open_quotient: 0.6,           // Typical value
            speed_quotient: 2.0,          // Faster closing than opening
            spectral_tilt: -12.0,         // dB/octave
            tongue_position: 0.5,         // Neutral position
            tongue_shape: 0.0,            // No curvature
            jaw_opening: 0.7,             // Moderate opening
            lip_rounding: 0.0,            // No rounding
            velum_opening: 0.0,           // Oral sounds (no nasal coupling)
            breathiness: 0.1,             // Slight breathiness
            roughness: 0.05,              // Minimal roughness
            dry_wet_mix: 1.0,             // Full physical model
        }
    }
}

/// Vowel presets for quick configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VowelPreset {
    /// /a/ as in "father"
    OpenCentral,
    /// /e/ as in "bet"
    MidFront,
    /// /i/ as in "beat"
    CloseFront,
    /// /o/ as in "boat"
    MidBack,
    /// /u/ as in "boot"
    CloseBack,
    /// Schwa /ə/ - neutral vowel
    Schwa,
}

impl VocalTractModel {
    /// Create new physical vocal tract model
    pub fn new(name: String, config: PhysicalModelConfig, sample_rate: f32) -> crate::Result<Self> {
        let sound_speed = 35000.0; // cm/s at body temperature
        let section_length = config.tract_length / config.num_sections as f32;
        let delay_samples = (section_length * sample_rate / sound_speed / 2.0).round() as usize;

        // Initialize delay lines for each section
        let mut delay_lines = Vec::with_capacity(config.num_sections);
        for _ in 0..config.num_sections {
            delay_lines.push(DelayLine::new(delay_samples.max(1)));
        }

        // Initialize area function (neutral vocal tract)
        let area_function = Self::create_neutral_area_function(config.num_sections);

        // Calculate reflection coefficients
        let reflectances = Self::calculate_reflectances(&area_function);

        // Initialize formant targets for neutral vowel
        let formant_targets = vec![
            FormantTarget {
                frequency: 500.0,
                bandwidth: 50.0,
                amplitude: 0.0,
            }, // F1
            FormantTarget {
                frequency: 1500.0,
                bandwidth: 70.0,
                amplitude: -6.0,
            }, // F2
            FormantTarget {
                frequency: 2500.0,
                bandwidth: 110.0,
                amplitude: -12.0,
            }, // F3
        ];

        // Initialize parameters map
        let mut parameters = HashMap::new();
        parameters.insert(
            "fundamental_frequency".to_string(),
            config.fundamental_frequency,
        );
        parameters.insert("tongue_position".to_string(), config.tongue_position);
        parameters.insert("tongue_shape".to_string(), config.tongue_shape);
        parameters.insert("jaw_opening".to_string(), config.jaw_opening);
        parameters.insert("lip_rounding".to_string(), config.lip_rounding);
        parameters.insert("velum_opening".to_string(), config.velum_opening);
        parameters.insert("breathiness".to_string(), config.breathiness);
        parameters.insert("roughness".to_string(), config.roughness);

        // Initialize glottal model
        let glottal_model = GlottalModel::new(
            config.fundamental_frequency,
            config.open_quotient,
            config.speed_quotient,
            config.spectral_tilt,
        );

        Ok(Self {
            name,
            parameters,
            tract_length: config.tract_length,
            area_function,
            num_sections: config.num_sections,
            sound_speed,
            sample_rate,
            delay_lines,
            reflectances,
            junction_pressures: vec![0.0; config.num_sections + 1],
            velocities: vec![0.0; config.num_sections],
            glottal_model,
            tongue_position: config.tongue_position,
            tongue_shape: config.tongue_shape,
            jaw_opening: config.jaw_opening,
            lip_rounding: config.lip_rounding,
            velum_opening: config.velum_opening,
            formant_targets,
        })
    }

    /// Get model name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Create neutral area function (uniform tube with slight constriction at glottis)
    pub fn create_neutral_area_function(num_sections: usize) -> Vec<f32> {
        let mut areas = Vec::with_capacity(num_sections);

        for i in 0..num_sections {
            let position = i as f32 / (num_sections - 1) as f32; // 0 to 1 from glottis to lips

            // Start with base area that increases from glottis to lips
            let base_area = 1.0 + 2.0 * position; // 1.0 to 3.0 cm²

            // Add slight constriction in pharynx region (around 20% from glottis)
            let pharynx_position = 0.2;
            let pharynx_constriction = if (position - pharynx_position).abs() < 0.1 {
                0.8 // 20% reduction
            } else {
                1.0
            };

            areas.push(base_area * pharynx_constriction);
        }

        areas
    }

    /// Calculate reflection coefficients from area function
    pub fn calculate_reflectances(areas: &[f32]) -> Vec<f32> {
        let mut reflectances = Vec::with_capacity(areas.len());

        for i in 0..areas.len() {
            let r = if i == areas.len() - 1 {
                // Lip termination - radiation load approximation
                -0.95 // Most energy radiated, some reflection
            } else {
                // Junction between sections
                let area1 = areas[i];
                let area2 = areas[i + 1];
                let denominator = area2 + area1;
                if denominator.abs() > 1e-10 {
                    (area2 - area1) / denominator
                } else {
                    0.0 // Avoid division by zero
                }
            };

            let safe_r = if r.is_finite() { r } else { 0.0 };
            reflectances.push(safe_r.clamp(-0.99, 0.99)); // Prevent instability
        }

        reflectances
    }

    /// Set vowel preset
    pub fn set_vowel_preset(&mut self, preset: VowelPreset) {
        match preset {
            VowelPreset::OpenCentral => {
                // /a/ - open central vowel
                self.tongue_position = 0.5;
                self.tongue_shape = 0.0;
                self.jaw_opening = 1.0;
                self.lip_rounding = 0.0;
                self.velum_opening = 0.0;
            }
            VowelPreset::MidFront => {
                // /e/ - mid front vowel
                self.tongue_position = 0.7;
                self.tongue_shape = 0.3;
                self.jaw_opening = 0.7;
                self.lip_rounding = 0.0;
                self.velum_opening = 0.0;
            }
            VowelPreset::CloseFront => {
                // /i/ - close front vowel
                self.tongue_position = 0.9;
                self.tongue_shape = 0.5;
                self.jaw_opening = 0.3;
                self.lip_rounding = 0.0;
                self.velum_opening = 0.0;
            }
            VowelPreset::MidBack => {
                // /o/ - mid back vowel
                self.tongue_position = 0.3;
                self.tongue_shape = -0.3;
                self.jaw_opening = 0.6;
                self.lip_rounding = 0.7;
                self.velum_opening = 0.0;
            }
            VowelPreset::CloseBack => {
                // /u/ - close back vowel
                self.tongue_position = 0.1;
                self.tongue_shape = -0.5;
                self.jaw_opening = 0.3;
                self.lip_rounding = 1.0;
                self.velum_opening = 0.0;
            }
            VowelPreset::Schwa => {
                // /ə/ - neutral vowel
                self.tongue_position = 0.5;
                self.tongue_shape = 0.0;
                self.jaw_opening = 0.5;
                self.lip_rounding = 0.0;
                self.velum_opening = 0.0;
            }
        }

        self.update_articulation();
    }

    /// Update area function based on articulatory parameters
    pub fn update_articulation(&mut self) {
        for i in 0..self.num_sections {
            let position = i as f32 / (self.num_sections - 1) as f32;

            // Base area
            let mut area = 1.0 + 2.0 * position;

            // Tongue influence (affects different regions differently)
            let tongue_influence =
                Self::tongue_area_influence(position, self.tongue_position, self.tongue_shape);
            area *= tongue_influence;

            // Jaw opening affects overall constriction
            area *= 0.3 + 0.7 * self.jaw_opening;

            // Lip rounding affects lip area (last few sections)
            if position > 0.8 {
                let lip_factor = 1.0 - 0.5 * self.lip_rounding;
                area *= lip_factor;
            }

            self.area_function[i] = area.max(0.1); // Minimum area to prevent closure
        }

        // Update reflection coefficients
        self.reflectances = Self::calculate_reflectances(&self.area_function);
    }

    /// Calculate tongue influence on area function
    fn tongue_area_influence(position: f32, tongue_pos: f32, tongue_shape: f32) -> f32 {
        let tongue_center = tongue_pos;
        let tongue_width = 0.3; // Width of tongue influence

        // Distance from tongue center
        let distance = (position - tongue_center).abs();

        if distance < tongue_width {
            // Tongue constriction
            let constriction_factor = 1.0 - (tongue_width - distance) / tongue_width;
            let shape_factor = 1.0 + tongue_shape;
            0.2 + 0.8 * constriction_factor * shape_factor
        } else {
            1.0
        }
    }

    /// Set parameter by name
    pub fn set_parameter(&mut self, name: &str, value: f32) -> crate::Result<()> {
        let clamped_value = match name {
            "fundamental_frequency" => value.clamp(50.0, 1000.0),
            "tongue_position" => value.clamp(0.0, 1.0),
            "tongue_shape" => value.clamp(-1.0, 1.0),
            "jaw_opening" => value.clamp(0.0, 1.0),
            "lip_rounding" => value.clamp(0.0, 1.0),
            "velum_opening" => value.clamp(0.0, 1.0),
            "breathiness" => value.clamp(0.0, 1.0),
            "roughness" => value.clamp(0.0, 1.0),
            _ => return Err(crate::Error::Processing("Unknown parameter".to_string())),
        };

        self.parameters.insert(name.to_string(), clamped_value);

        // Update internal state
        match name {
            "fundamental_frequency" => self.glottal_model.f0 = clamped_value,
            "tongue_position" => {
                self.tongue_position = clamped_value;
                self.update_articulation();
            }
            "tongue_shape" => {
                self.tongue_shape = clamped_value;
                self.update_articulation();
            }
            "jaw_opening" => {
                self.jaw_opening = clamped_value;
                self.update_articulation();
            }
            "lip_rounding" => {
                self.lip_rounding = clamped_value;
                self.update_articulation();
            }
            "velum_opening" => {
                self.velum_opening = clamped_value;
                self.update_articulation();
            }
            _ => {}
        }

        Ok(())
    }

    /// Get parameter by name
    pub fn get_parameter(&self, name: &str) -> Option<f32> {
        self.parameters.get(name).copied()
    }

    /// Process audio through physical model
    pub fn process_physical_model(&mut self, audio: &mut [f32]) -> crate::Result<()> {
        for sample in audio.iter_mut() {
            // Generate glottal excitation
            let excitation = self.glottal_model.generate_sample();

            // Process through vocal tract tube model
            let output = self.process_tube_model(excitation);

            *sample += output;
        }

        Ok(())
    }

    /// Process single sample through tube model using Kelly-Lochbaum algorithm
    fn process_tube_model(&mut self, excitation: f32) -> f32 {
        // Kelly-Lochbaum algorithm implementation
        let mut forward_waves = vec![0.0; self.num_sections + 1];
        let mut backward_waves = vec![0.0; self.num_sections + 1];

        // Glottal excitation
        forward_waves[0] = excitation;

        // Forward pass
        for i in 0..self.num_sections {
            let delayed_backward = self.delay_lines[i].read_sample();
            backward_waves[i] = delayed_backward;

            // Scattering at junction
            let r = self.reflectances[i];
            let forward_transmitted = (1.0 + r) * forward_waves[i] + r * delayed_backward;
            let backward_reflected = r * forward_waves[i] + (r - 1.0) * delayed_backward;

            forward_waves[i + 1] = forward_transmitted;
            self.delay_lines[i].write_sample(backward_reflected);
        }

        // Output is the forward wave at the lips
        let output = forward_waves[self.num_sections];

        // Ensure output is finite
        if output.is_finite() {
            output
        } else {
            0.0
        }
    }

    /// Reset model state
    pub fn reset(&mut self) {
        // Clear delay lines
        for delay_line in &mut self.delay_lines {
            delay_line.clear();
        }

        // Reset pressures and velocities
        self.junction_pressures.fill(0.0);
        self.velocities.fill(0.0);

        // Reset glottal model
        self.glottal_model.reset();
    }
}

impl GlottalModel {
    /// Create new glottal model
    pub fn new(f0: f32, open_quotient: f32, speed_quotient: f32, spectral_tilt: f32) -> Self {
        Self {
            f0,
            open_quotient,
            speed_quotient,
            pulse_amplitude: 1.0,
            spectral_tilt,
            shimmer: 0.05, // 5% amplitude variation
            jitter: 0.01,  // 1% frequency variation
            phase: 0.0,
            last_pulse_amplitude: 1.0,
            rng_state: 12345, // Fixed seed for reproducibility
        }
    }

    /// Generate single sample using Liljencrants-Fant model
    pub fn generate_sample(&mut self) -> f32 {
        let period = 1.0 / self.f0;
        let sample_period = 1.0 / 44100.0; // Assume 44.1kHz for now

        // Add jitter
        let jittered_f0 = self.f0 * (1.0 + self.jitter * self.random_gaussian());
        let phase_increment = jittered_f0 * sample_period;

        self.phase += phase_increment;

        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        // Generate Liljencrants-Fant glottal pulse
        let pulse = if self.phase < self.open_quotient {
            // Opening phase
            let denominator = self.open_quotient.max(0.01); // Prevent division by zero
            let t = self.phase / denominator;
            let opening = 0.5 * (1.0 - (std::f32::consts::PI * t).cos());
            opening
        } else {
            // Closing phase
            let denominator = (1.0 - self.open_quotient).max(0.01); // Prevent division by zero
            let t = (self.phase - self.open_quotient) / denominator;
            let closing = 0.5 * (1.0 + (std::f32::consts::PI * t * self.speed_quotient).cos());
            closing.max(0.0)
        };

        // Add shimmer
        let shimmer_factor = 1.0 + self.shimmer * self.random_gaussian();
        let amplitude = self.pulse_amplitude * shimmer_factor;
        self.last_pulse_amplitude = amplitude;

        // Apply spectral tilt (simple first-order high-pass approximation)
        let result = pulse * amplitude;

        // Ensure result is finite to prevent NaN propagation
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }

    /// Simple random Gaussian generator (Box-Muller approximation)
    fn random_gaussian(&mut self) -> f32 {
        // Linear congruential generator for reproducible randomness
        self.rng_state = (self.rng_state.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fffffff;
        let uniform = (self.rng_state as f32) / (0x7fffffff as f32);

        // Convert to Gaussian (simplified Box-Muller)
        (uniform - 0.5) * 4.0 // Approximate Gaussian with std dev ≈ 1
    }

    /// Reset glottal model state
    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.last_pulse_amplitude = 1.0;
        self.rng_state = 12345;
    }
}

impl DelayLine {
    /// Create new delay line
    pub fn new(size: usize) -> Self {
        Self {
            buffer: vec![0.0; size.max(1)],
            size: size.max(1),
            read_pos: 0,
            write_pos: 0,
        }
    }

    /// Read sample from delay line
    pub fn read_sample(&mut self) -> f32 {
        let sample = self.buffer[self.read_pos];
        self.read_pos = (self.read_pos + 1) % self.size;
        sample
    }

    /// Write sample to delay line
    pub fn write_sample(&mut self, sample: f32) {
        self.buffer[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % self.size;
    }

    /// Clear delay line
    pub fn clear(&mut self) {
        self.buffer.fill(0.0);
        self.read_pos = 0;
        self.write_pos = 0;
    }
}

impl SingingEffect for VocalTractModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(&mut self, audio: &mut [f32], _sample_rate: f32) -> crate::Result<()> {
        self.process_physical_model(audio)
    }

    fn set_parameter(&mut self, name: &str, value: f32) -> crate::Result<()> {
        self.set_parameter(name, value)
    }

    fn get_parameter(&self, name: &str) -> Option<f32> {
        self.get_parameter(name)
    }

    fn get_parameters(&self) -> HashMap<String, f32> {
        self.parameters.clone()
    }

    fn reset(&mut self) {
        self.reset()
    }

    fn clone_effect(&self) -> Box<dyn SingingEffect> {
        Box::new(self.clone())
    }
}
