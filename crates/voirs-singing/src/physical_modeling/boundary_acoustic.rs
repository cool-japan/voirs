//! Boundary condition and acoustic propagation modeling
//!
//! This module contains advanced boundary condition handling and acoustic propagation
//! models for enhanced realism in vocal tract simulation.

use super::advanced_physics::Complex32;

/// Advanced boundary condition modeling
#[derive(Debug, Clone)]
pub struct BoundaryConditionModel {
    /// Lip radiation impedance (frequency-dependent)
    pub lip_radiation_impedance: Vec<Complex32>,
    /// Nostril radiation (for nasal sounds)
    pub nostril_impedance: Vec<Complex32>,
    /// Glottal impedance (time-varying)
    pub glottal_impedance_history: Vec<f32>,
    /// Wall impedance (acoustic properties of tissues)
    pub wall_acoustic_impedance: Vec<Complex32>,
    /// Subglottal impedance
    pub subglottal_impedance: Complex32,
    /// Frequency-dependent boundary losses
    pub boundary_loss_coefficients: Vec<f32>,
}

/// Advanced acoustic propagation with dispersion and absorption
#[derive(Debug, Clone)]
pub struct AcousticPropagationModel {
    /// Frequency-dependent attenuation
    pub frequency_attenuation: Vec<f32>,
    /// Dispersion relation parameters
    pub dispersion_coefficients: Vec<f32>,
    /// Modal decomposition for complex geometries
    pub acoustic_modes: Vec<AcousticMode>,
    /// Scattering parameters
    pub scattering_matrix: Vec<Vec<Complex32>>,
    /// Diffraction effects at constrictions
    pub diffraction_corrections: Vec<f32>,
    /// Multi-path propagation
    pub propagation_paths: Vec<PropagationPath>,
}

/// Acoustic mode for modal analysis
#[derive(Debug, Clone)]
pub struct AcousticMode {
    /// Mode frequency
    pub frequency: f32,
    /// Mode shape (spatial distribution)
    pub mode_shape: Vec<f32>,
    /// Modal damping
    pub damping: f32,
    /// Modal excitation coefficient
    pub excitation_coeff: f32,
}

/// Propagation path for multi-path analysis
#[derive(Debug, Clone)]
pub struct PropagationPath {
    /// Path length
    pub length: f32,
    /// Path delay
    pub delay: f32,
    /// Path attenuation
    pub attenuation: f32,
    /// Path phase shift
    pub phase_shift: f32,
}

impl BoundaryConditionModel {
    pub fn new() -> crate::Result<Self> {
        // Initialize frequency-dependent lip radiation impedance
        let mut lip_impedance = Vec::new();
        for i in 0..1000 {
            let freq = i as f32 * 20.0; // 0-20kHz
            let ka = freq * 0.005 / 343.0; // k*a for lip radius ≈ 5mm

            // Radiation impedance for circular piston in infinite baffle
            let real_part = if ka < 0.5 {
                ka * ka / 2.0
            } else {
                1.0 - 0.7 / (ka * ka)
            };

            let imag_part = if ka < 1.0 {
                ka * (8.0 / (3.0 * std::f32::consts::PI))
            } else {
                ka / (ka * ka + 1.0)
            };

            lip_impedance.push(Complex32::new(real_part, imag_part));
        }

        // Initialize nostril impedance (smaller aperture)
        let mut nostril_impedance = Vec::new();
        for i in 0..1000 {
            let freq = i as f32 * 20.0;
            let ka = freq * 0.002 / 343.0; // Smaller nostril radius

            let real_part = ka * ka / 4.0; // Reduced radiation
            let imag_part = ka * (4.0 / (3.0 * std::f32::consts::PI));

            nostril_impedance.push(Complex32::new(real_part, imag_part));
        }

        // Initialize wall impedance for tissue
        let mut wall_impedance = Vec::new();
        for _ in 0..20 {
            // Typical tissue impedance
            wall_impedance.push(Complex32::new(1000.0, 100.0));
        }

        Ok(Self {
            lip_radiation_impedance: lip_impedance,
            nostril_impedance,
            glottal_impedance_history: vec![0.0; 100],
            wall_acoustic_impedance: wall_impedance,
            subglottal_impedance: Complex32::new(100.0, 50.0),
            boundary_loss_coefficients: (0..1000).map(|i| 0.001 * (i as f32).sqrt()).collect(),
        })
    }

    pub fn update_glottal_impedance(&mut self, glottal_area: f32, velocity: f32) {
        // Model time-varying glottal impedance based on area and flow
        let impedance = if glottal_area > 0.001 {
            velocity / glottal_area // Flow-dependent resistance
        } else {
            1000.0 // High impedance when closed
        };

        // Add to history
        self.glottal_impedance_history.rotate_right(1);
        self.glottal_impedance_history[0] = impedance;
    }

    pub fn get_lip_reflection_coefficient(&self, frequency: f32) -> Complex32 {
        let freq_index = (frequency / 20.0).min(999.0) as usize;
        let impedance = self.lip_radiation_impedance[freq_index];

        // Reflection coefficient: (Z - Z0) / (Z + Z0) where Z0 = 1.0 (normalized)
        let z_plus_one = Complex32::new(impedance.real + 1.0, impedance.imag);
        let z_minus_one = Complex32::new(impedance.real - 1.0, impedance.imag);

        // Simplified complex division
        let denom = z_plus_one.real * z_plus_one.real + z_plus_one.imag * z_plus_one.imag;
        if denom > 1e-10 {
            Complex32::new(
                (z_minus_one.real * z_plus_one.real + z_minus_one.imag * z_plus_one.imag) / denom,
                (z_minus_one.imag * z_plus_one.real - z_minus_one.real * z_plus_one.imag) / denom,
            )
        } else {
            Complex32::new(-1.0, 0.0) // Perfect reflection if impedance is zero
        }
    }

    pub fn get_boundary_loss(&self, frequency: f32) -> f32 {
        let freq_index = (frequency / 20.0).min(999.0) as usize;
        self.boundary_loss_coefficients[freq_index]
    }
}

impl AcousticPropagationModel {
    pub fn new() -> crate::Result<Self> {
        Ok(Self {
            frequency_attenuation: (0..1000).map(|i| 0.001 * (i as f32)).collect(),
            dispersion_coefficients: vec![1.0, 0.001, 1e-6],
            acoustic_modes: vec![AcousticMode {
                frequency: 500.0,
                mode_shape: vec![1.0; 20],
                damping: 0.01,
                excitation_coeff: 1.0,
            }],
            scattering_matrix: vec![vec![Complex32::new(1.0, 0.0); 10]; 10],
            diffraction_corrections: vec![1.0; 20],
            propagation_paths: vec![PropagationPath {
                length: 0.175, // 17.5cm
                delay: 0.0005, // seconds
                attenuation: 0.99,
                phase_shift: 0.0,
            }],
        })
    }

    pub fn calculate_modal_response(&self, excitation_spectrum: &[f32]) -> Vec<f32> {
        let mut response = vec![0.0; excitation_spectrum.len()];

        for mode in &self.acoustic_modes {
            for (i, &excitation) in excitation_spectrum.iter().enumerate() {
                let frequency = i as f32 * 20.0; // 20Hz per bin

                // Modal response function (simplified)
                let freq_ratio = frequency / mode.frequency;
                let q_factor = 1.0 / (2.0 * mode.damping);

                let denominator =
                    (1.0 - freq_ratio * freq_ratio).powi(2) + (freq_ratio / q_factor).powi(2);

                let modal_gain = mode.excitation_coeff / denominator.sqrt();
                response[i] += excitation * modal_gain;
            }
        }

        response
    }

    pub fn apply_frequency_attenuation(&self, spectrum: &mut [f32]) {
        for (i, sample) in spectrum.iter_mut().enumerate() {
            if i < self.frequency_attenuation.len() {
                *sample *= (1.0 - self.frequency_attenuation[i]).max(0.0);
            }
        }
    }

    pub fn calculate_dispersion_delay(&self, frequency: f32) -> f32 {
        // Calculate frequency-dependent group delay
        let omega = 2.0 * std::f32::consts::PI * frequency;

        // Dispersion relation: k = ω/c + α*ω² + β*ω³
        let mut delay = 1.0 / 343.0; // Base delay (1/c)

        for (n, &coeff) in self.dispersion_coefficients.iter().enumerate() {
            delay += coeff * omega.powi(n as i32 + 1);
        }

        delay
    }

    pub fn apply_diffraction_correction(&self, section: usize, area_ratio: f32) -> f32 {
        if section < self.diffraction_corrections.len() {
            // Diffraction correction based on area change
            let base_correction = self.diffraction_corrections[section];
            let area_correction = if area_ratio < 0.5 {
                // Strong constriction - significant diffraction
                0.8 + 0.2 * area_ratio / 0.5
            } else {
                // Mild constriction - minimal diffraction
                0.95 + 0.05 * (area_ratio - 0.5) / 0.5
            };

            base_correction * area_correction
        } else {
            1.0
        }
    }

    pub fn calculate_multipath_response(&self, input_signal: &[f32], sample_rate: f32) -> Vec<f32> {
        let mut output = vec![0.0; input_signal.len()];

        for path in &self.propagation_paths {
            let delay_samples = (path.delay * sample_rate) as usize;

            for i in delay_samples..input_signal.len() {
                let delayed_index = i - delay_samples;
                let delayed_sample = input_signal[delayed_index] * path.attenuation;

                // Apply phase shift (simplified as a small delay)
                let phase_delay_samples =
                    (path.phase_shift / (2.0 * std::f32::consts::PI) * sample_rate) as usize;
                if delayed_index >= phase_delay_samples {
                    output[i] += delayed_sample;
                }
            }
        }

        output
    }
}

impl AcousticMode {
    pub fn new(frequency: f32, num_sections: usize) -> Self {
        // Generate mode shape (simplified as sinusoidal)
        let mut mode_shape = Vec::with_capacity(num_sections);
        for i in 0..num_sections {
            let position = i as f32 / (num_sections - 1) as f32;
            let shape = (std::f32::consts::PI * position).sin();
            mode_shape.push(shape);
        }

        Self {
            frequency,
            mode_shape,
            damping: 0.01, // 1% damping
            excitation_coeff: 1.0,
        }
    }

    pub fn get_modal_amplitude(&self, excitation_frequency: f32) -> f32 {
        let freq_ratio = excitation_frequency / self.frequency;
        let q_factor = 1.0 / (2.0 * self.damping);

        let denominator = (1.0 - freq_ratio * freq_ratio).powi(2) + (freq_ratio / q_factor).powi(2);

        self.excitation_coeff / denominator.sqrt()
    }
}

impl PropagationPath {
    pub fn new(length: f32, sound_speed: f32) -> Self {
        let delay = length / sound_speed;
        let attenuation = (-0.1 * length).exp(); // Simple exponential decay

        Self {
            length,
            delay,
            attenuation,
            phase_shift: 0.0,
        }
    }

    pub fn update_for_frequency(&mut self, frequency: f32, sound_speed: f32) {
        // Update phase shift based on frequency
        let wavelength = sound_speed / frequency;
        let phase_cycles = self.length / wavelength;
        self.phase_shift = 2.0 * std::f32::consts::PI * (phase_cycles % 1.0);

        // Update frequency-dependent attenuation
        let alpha = 0.001 * frequency.sqrt(); // Frequency-dependent absorption
        self.attenuation = (-alpha * self.length).exp();
    }
}
