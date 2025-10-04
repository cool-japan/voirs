//! Advanced physics modeling components for enhanced vocal tract simulation
//!
//! This module contains sophisticated physics models including turbulence, wall vibration,
//! thermal effects, and nonlinear dynamics for highly realistic vocal tract simulation.

/// Complex number for viscoelastic modeling
#[derive(Debug, Clone, Copy)]
pub struct Complex32 {
    /// Real component
    pub real: f32,
    /// Imaginary component
    pub imag: f32,
}

impl Complex32 {
    /// Create new complex number
    ///
    /// # Arguments
    /// * `real` - Real component
    /// * `imag` - Imaginary component
    ///
    /// # Returns
    /// New complex number
    pub fn new(real: f32, imag: f32) -> Self {
        Self { real, imag }
    }

    /// Calculate magnitude (absolute value) of complex number
    ///
    /// # Returns
    /// Magnitude as sqrt(real² + imag²)
    pub fn magnitude(&self) -> f32 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }

    /// Calculate phase angle of complex number
    ///
    /// # Returns
    /// Phase angle in radians
    pub fn phase(&self) -> f32 {
        self.imag.atan2(self.real)
    }
}

/// Turbulence modeling for realistic flow simulation
#[derive(Debug, Clone)]
pub struct TurbulenceModel {
    /// Reynolds number for flow characterization (dimensionless)
    pub reynolds_number: f32,
    /// Turbulence intensity from laminar (0) to fully turbulent (1)
    pub turbulence_intensity: f32,
    /// Turbulent kinetic energy per section in m²/s²
    pub kinetic_energy: Vec<f32>,
    /// Turbulence dissipation rate per section in m²/s³
    pub dissipation_rate: Vec<f32>,
    /// Velocity fluctuation history for statistical analysis
    pub velocity_history: Vec<Vec<f32>>,
    /// Eddy viscosity per section in m²/s
    pub eddy_viscosity: Vec<f32>,
    /// Large Eddy Simulation filter width in meters
    pub les_filter_width: f32,
    /// Smagorinsky constant for subgrid-scale model (typically ~0.18)
    pub sgs_model_constant: f32,
}

/// Wall vibration modeling for tissue compliance
#[derive(Debug, Clone)]
pub struct WallVibrationModel {
    /// Wall mass per unit area in kg/m²
    pub wall_mass: Vec<f32>,
    /// Wall stiffness (spring constant per unit area) in N/m³
    pub wall_stiffness: Vec<f32>,
    /// Wall damping coefficient in N·s/m³
    pub wall_damping: Vec<f32>,
    /// Current wall displacement in meters
    pub wall_displacement: Vec<f32>,
    /// Wall velocity in m/s
    pub wall_velocity: Vec<f32>,
    /// Wall acceleration in m/s²
    pub wall_acceleration: Vec<f32>,
    /// Coupling coefficient with acoustic field (dimensionless)
    pub fluid_structure_coupling: f32,
    /// Tissue viscoelastic properties (complex modulus)
    pub viscoelastic_modulus: Vec<Complex32>,
    /// Wall thickness per section in meters
    pub wall_thickness: Vec<f32>,
}

/// Thermal effects modeling for temperature-dependent acoustics
#[derive(Debug, Clone)]
pub struct ThermalModel {
    /// Temperature distribution along vocal tract in °C
    pub temperature_profile: Vec<f32>,
    /// Thermal conductivity in W/(m·K)
    pub thermal_conductivity: f32,
    /// Specific heat capacity in J/(kg·K)
    pub specific_heat: f32,
    /// Density-temperature coefficient in 1/K
    pub density_temperature_coeff: f32,
    /// Sound speed temperature coefficient in m/(s·K)
    pub sound_speed_temp_coeff: f32,
    /// Viscosity-temperature power law exponent (dimensionless)
    pub viscosity_temperature_coeff: f32,
    /// Heat transfer coefficient with walls in W/(m²·K)
    pub heat_transfer_coeff: f32,
    /// Thermal boundary layer thickness per section in meters
    pub thermal_boundary_layer: Vec<f32>,
}

/// Nonlinear dynamics modeling for large amplitude effects
#[derive(Debug, Clone)]
pub struct NonlinearDynamicsModel {
    /// Nonlinear convective acceleration terms in m/s²
    pub convective_acceleration: Vec<f32>,
    /// Pressure-dependent compressibility factor (dimensionless)
    pub compressibility_factor: Vec<f32>,
    /// Nonlinear wave steepening factor (dimensionless)
    pub wave_steepening_factor: f32,
    /// Shock formation threshold (Mach number)
    pub shock_threshold: f32,
    /// Harmonic distortion coefficients for 2nd-5th harmonics
    pub harmonic_distortion: Vec<f32>,
    /// Amplitude-dependent damping per section (dimensionless)
    pub amplitude_damping: Vec<f32>,
    /// Frequency-dependent nonlinear shift in Hz
    pub nonlinear_frequency_shift: Vec<f32>,
}

/// Physics accuracy levels for performance vs quality tradeoff
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsAccuracyLevel {
    /// Basic physical modeling (fastest)
    Basic,
    /// Intermediate accuracy with key nonlinear effects
    Intermediate,
    /// High accuracy with most physical effects
    High,
    /// Maximum accuracy with all physical effects (slowest)
    Maximum,
}

/// Real-time adaptable physics parameters
#[derive(Debug, Clone)]
pub struct AdaptivePhysicsParameters {
    /// Dynamic quality adjustment strategy based on CPU load
    pub quality_adaptation: QualityAdaptation,
    /// Frequency-dependent resolution (bins per frequency band)
    pub frequency_resolution: Vec<usize>,
    /// Spatial resolution (number of sections per region)
    pub spatial_resolution: Vec<usize>,
    /// Temporal resolution (time step size) in seconds
    pub temporal_resolution: f32,
    /// Error-based mesh refinement strategy
    pub mesh_refinement: MeshRefinement,
}

/// Quality adaptation strategy
#[derive(Debug, Clone)]
pub struct QualityAdaptation {
    /// CPU load threshold for quality reduction (0-1)
    pub cpu_threshold: f32,
    /// Memory usage threshold in bytes
    pub memory_threshold: usize,
    /// Maximum acceptable latency in milliseconds
    pub max_latency_ms: f32,
    /// Progressive quality reduction steps
    pub quality_steps: Vec<PhysicsAccuracyLevel>,
}

/// Mesh refinement for adaptive resolution
#[derive(Debug, Clone)]
pub struct MeshRefinement {
    /// Error estimation method selection
    pub error_estimator: ErrorEstimator,
    /// Error threshold above which mesh is refined
    pub refinement_threshold: f32,
    /// Error threshold below which mesh is coarsened
    pub coarsening_threshold: f32,
    /// Maximum allowed refinement level (0 = base mesh)
    pub max_refinement_level: usize,
}

/// Error estimation methods for adaptive meshing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorEstimator {
    /// Gradient-based error estimation
    Gradient,
    /// Residual-based error estimation
    Residual,
    /// Adjoint-based error estimation
    Adjoint,
    /// Feature-based error estimation
    Feature,
}

impl TurbulenceModel {
    /// Create new turbulence model for specified number of sections
    ///
    /// # Arguments
    /// * `num_sections` - Number of vocal tract sections to model
    ///
    /// # Returns
    /// New turbulence model with default k-ε parameters
    pub fn new(num_sections: usize) -> crate::Result<Self> {
        Ok(Self {
            reynolds_number: 3000.0,
            turbulence_intensity: 0.05,
            kinetic_energy: vec![0.0; num_sections],
            dissipation_rate: vec![0.001; num_sections],
            velocity_history: vec![vec![0.0; 10]; num_sections],
            eddy_viscosity: vec![1e-5; num_sections],
            les_filter_width: 0.1,
            sgs_model_constant: 0.18,
        })
    }

    /// Update turbulence state using k-ε model
    ///
    /// # Arguments
    /// * `velocities` - Flow velocities in each section (m/s)
    /// * `areas` - Cross-sectional areas in each section (m²)
    /// * `dt` - Time step in seconds
    pub fn update_turbulence(&mut self, velocities: &[f32], areas: &[f32], dt: f32) {
        let len = self.kinetic_energy.len().min(velocities.len());
        for (i, &velocity) in velocities.iter().enumerate().take(len) {
            let area = areas.get(i).unwrap_or(&1.0);

            // Update Reynolds number based on local conditions
            let _local_reynolds = velocity * area.sqrt() / 1e-5; // kinematic viscosity ≈ 1e-5 m²/s

            // Update turbulent kinetic energy using k-ε model
            let production = self.eddy_viscosity[i] * velocity * velocity / area;
            let dissipation = self.dissipation_rate[i];

            self.kinetic_energy[i] += dt * (production - dissipation);
            self.kinetic_energy[i] = self.kinetic_energy[i].max(0.0);

            // Update dissipation rate
            let c_epsilon1 = 1.44;
            let c_epsilon2 = 1.92;
            let time_scale = self.kinetic_energy[i] / dissipation.max(1e-10);

            let dissipation_production = c_epsilon1 * production / time_scale;
            let dissipation_destruction = c_epsilon2 * dissipation / time_scale;

            self.dissipation_rate[i] += dt * (dissipation_production - dissipation_destruction);
            self.dissipation_rate[i] = self.dissipation_rate[i].max(1e-10);

            // Update eddy viscosity
            let c_mu = 0.09;
            self.eddy_viscosity[i] =
                c_mu * self.kinetic_energy[i] * self.kinetic_energy[i] / self.dissipation_rate[i];
        }
    }
}

impl WallVibrationModel {
    /// Create new wall vibration model with typical tissue properties
    ///
    /// # Arguments
    /// * `num_sections` - Number of vocal tract sections to model
    ///
    /// # Returns
    /// New wall vibration model with default tissue parameters
    pub fn new(num_sections: usize) -> crate::Result<Self> {
        Ok(Self {
            wall_mass: vec![0.001; num_sections],       // kg/m²
            wall_stiffness: vec![1000.0; num_sections], // N/m³
            wall_damping: vec![0.1; num_sections],
            wall_displacement: vec![0.0; num_sections],
            wall_velocity: vec![0.0; num_sections],
            wall_acceleration: vec![0.0; num_sections],
            fluid_structure_coupling: 0.1,
            viscoelastic_modulus: vec![Complex32::new(1000.0, 100.0); num_sections],
            wall_thickness: vec![0.001; num_sections], // 1mm
        })
    }

    /// Update wall dynamics using spring-mass-damper model
    ///
    /// # Arguments
    /// * `pressures` - Acoustic pressures at each section (Pa)
    /// * `dt` - Time step in seconds
    pub fn update_wall_dynamics(&mut self, pressures: &[f32], dt: f32) {
        let len = self.wall_displacement.len().min(pressures.len());
        for (i, &pressure) in pressures.iter().enumerate().take(len) {
            // Calculate forces on wall element
            let pressure_force = pressure * self.fluid_structure_coupling;
            let spring_force = -self.wall_stiffness[i] * self.wall_displacement[i];
            let damping_force = -self.wall_damping[i] * self.wall_velocity[i];

            let total_force = pressure_force + spring_force + damping_force;

            // Update wall motion (F = ma)
            self.wall_acceleration[i] = total_force / self.wall_mass[i];
            self.wall_velocity[i] += self.wall_acceleration[i] * dt;
            self.wall_displacement[i] += self.wall_velocity[i] * dt;

            // Apply displacement limits to prevent numerical instability
            self.wall_displacement[i] = self.wall_displacement[i].clamp(-0.01, 0.01);
        }
    }
}

impl ThermalModel {
    /// Create new thermal model with body temperature conditions
    ///
    /// # Arguments
    /// * `num_sections` - Number of vocal tract sections to model
    ///
    /// # Returns
    /// New thermal model initialized to 37°C body temperature
    pub fn new(num_sections: usize) -> crate::Result<Self> {
        Ok(Self {
            temperature_profile: vec![37.0; num_sections], // Body temperature in °C
            thermal_conductivity: 0.6,                     // W/(m·K) for air
            specific_heat: 1005.0,                         // J/(kg·K) for air
            density_temperature_coeff: -0.00366,           // 1/K
            sound_speed_temp_coeff: 0.6,                   // m/(s·K)
            viscosity_temperature_coeff: 0.7,              // Power law exponent
            heat_transfer_coeff: 10.0,                     // W/(m²·K)
            thermal_boundary_layer: vec![1e-4; num_sections], // 0.1mm
        })
    }

    /// Update temperature distribution and thermal effects
    ///
    /// # Arguments
    /// * `velocities` - Flow velocities in each section (m/s)
    /// * `dt` - Time step in seconds
    pub fn update_temperature_effects(&mut self, velocities: &[f32], dt: f32) {
        let len = self.temperature_profile.len().min(velocities.len());
        for (i, &velocity) in velocities.iter().enumerate().take(len) {
            // Heat generation due to viscous dissipation
            let viscous_heating = velocity * velocity * 1e-5; // Simplified

            // Heat transfer to walls
            let wall_heat_loss = self.heat_transfer_coeff *
                (self.temperature_profile[i] - 20.0) / // Assume 20°C wall temperature
                self.thermal_boundary_layer[i];

            // Temperature change
            let temp_change =
                dt * (viscous_heating - wall_heat_loss) / (1.225 * self.specific_heat); // Air density ≈ 1.225 kg/m³

            self.temperature_profile[i] += temp_change;

            // Clamp temperature to reasonable range
            self.temperature_profile[i] = self.temperature_profile[i].clamp(15.0, 50.0);

            // Update boundary layer thickness based on flow conditions
            let reynolds_local = velocity * 0.01 / 1e-5; // Local Reynolds number
            if reynolds_local > 0.0 {
                self.thermal_boundary_layer[i] = 0.01 / reynolds_local.sqrt();
            }
        }
    }

    /// Get temperature-dependent sound speed correction factor
    ///
    /// # Arguments
    /// * `section` - Vocal tract section index
    ///
    /// # Returns
    /// Sound speed correction factor (1.0 = no correction)
    pub fn get_sound_speed_correction(&self, section: usize) -> f32 {
        if section < self.temperature_profile.len() {
            let temp_celsius = self.temperature_profile[section];
            let temp_kelvin = temp_celsius + 273.15;
            // Sound speed correction: c = c0 * sqrt(T/T0)
            (temp_kelvin / 310.15).sqrt() // T0 = 37°C = 310.15K
        } else {
            1.0
        }
    }

    /// Get temperature-dependent density correction factor
    ///
    /// # Arguments
    /// * `section` - Vocal tract section index
    ///
    /// # Returns
    /// Density correction factor (1.0 = no correction)
    pub fn get_density_correction(&self, section: usize) -> f32 {
        if section < self.temperature_profile.len() {
            let temp_celsius = self.temperature_profile[section];
            let temp_kelvin = temp_celsius + 273.15;
            // Density correction: ρ = ρ0 * T0/T
            310.15 / temp_kelvin
        } else {
            1.0
        }
    }
}

impl NonlinearDynamicsModel {
    /// Create new nonlinear dynamics model
    ///
    /// # Arguments
    /// * `num_sections` - Number of vocal tract sections to model
    ///
    /// # Returns
    /// New nonlinear dynamics model with default parameters
    pub fn new(num_sections: usize) -> crate::Result<Self> {
        Ok(Self {
            convective_acceleration: vec![0.0; num_sections],
            compressibility_factor: vec![1.0; num_sections],
            wave_steepening_factor: 0.1,
            shock_threshold: 0.3,              // Mach number
            harmonic_distortion: vec![0.0; 5], // Up to 5th harmonic
            amplitude_damping: vec![0.0; num_sections],
            nonlinear_frequency_shift: vec![0.0; num_sections],
        })
    }

    /// Update nonlinear effects including convection and shock formation
    ///
    /// # Arguments
    /// * `velocities` - Flow velocities in each section (m/s)
    /// * `pressures` - Acoustic pressures in each section (Pa)
    /// * `dt` - Time step in seconds
    pub fn update_nonlinear_effects(&mut self, velocities: &[f32], pressures: &[f32], dt: f32) {
        for i in 0..self.convective_acceleration.len().min(velocities.len()) {
            let velocity = velocities[i];
            let pressure = pressures.get(i).unwrap_or(&0.0);

            // Convective acceleration (u ∂u/∂x)
            if i > 0 && i < velocities.len() - 1 {
                let velocity_gradient = (velocities[i + 1] - velocities[i - 1]) / 2.0;
                self.convective_acceleration[i] = velocity * velocity_gradient;
            }

            // Pressure-dependent compressibility
            let mach_number = velocity / 343.0; // Assume sound speed = 343 m/s
            self.compressibility_factor[i] = 1.0 + 0.2 * mach_number * mach_number;

            // Check for shock formation
            if mach_number.abs() > self.shock_threshold {
                // Apply shock dissipation
                self.amplitude_damping[i] = 0.1 * (mach_number.abs() - self.shock_threshold);
            } else {
                self.amplitude_damping[i] *= 0.95; // Decay
            }

            // Nonlinear frequency shift due to amplitude
            let amplitude = velocity.abs();
            self.nonlinear_frequency_shift[i] = 0.01 * amplitude * amplitude;
        }

        // Update harmonic distortion coefficients
        let fundamental_amplitude =
            velocities.iter().map(|v| v.abs()).sum::<f32>() / velocities.len() as f32;

        for (harmonic, coeff) in self.harmonic_distortion.iter_mut().enumerate() {
            let harmonic_order = harmonic + 2; // 2nd, 3rd, 4th, 5th harmonics
            *coeff = 0.01 * fundamental_amplitude.powi(harmonic_order as i32);
        }
    }

    /// Apply nonlinear damping correction to velocity
    ///
    /// # Arguments
    /// * `velocity` - Input velocity (m/s)
    /// * `section` - Vocal tract section index
    ///
    /// # Returns
    /// Corrected velocity with nonlinear damping applied
    pub fn apply_nonlinear_correction(&self, velocity: f32, section: usize) -> f32 {
        if section < self.amplitude_damping.len() {
            let damping = self.amplitude_damping[section];
            velocity * (1.0 - damping)
        } else {
            velocity
        }
    }

    /// Get harmonic distortion coefficients
    ///
    /// # Returns
    /// Slice of harmonic coefficients (2nd-5th harmonics)
    pub fn get_harmonic_content(&self) -> &[f32] {
        &self.harmonic_distortion
    }
}
