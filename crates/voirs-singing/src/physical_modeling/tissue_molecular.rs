//! Tissue mechanics and molecular dynamics modeling
//!
//! This module contains models for tissue mechanics and molecular-scale effects
//! for highly detailed vocal tract simulation.

/// Molecular dynamics effects for micro-scale accuracy
#[derive(Debug, Clone)]
pub struct MolecularDynamicsModel {
    /// Air molecule mean free path in meters
    pub mean_free_path: f32,
    /// Knudsen number for rarefaction effects (dimensionless)
    pub knudsen_number: f32,
    /// Molecular collision frequency in Hz
    pub collision_frequency: f32,
    /// Gas kinetic corrections per section (dimensionless)
    pub kinetic_corrections: Vec<f32>,
    /// Slip boundary condition coefficients per section
    pub slip_coefficients: Vec<f32>,
    /// Temperature jump at walls per section in K
    pub temperature_jump: Vec<f32>,
}

/// Tissue mechanics for realistic vocal tract behavior
#[derive(Debug, Clone)]
pub struct TissueMechanicsModel {
    /// Tissue elastic modulus (Young's modulus) per section in Pa
    pub elastic_modulus: Vec<f32>,
    /// Poisson's ratio per section (dimensionless, ~0.45 for tissue)
    pub poisson_ratio: Vec<f32>,
    /// Tissue density per section in kg/m³
    pub tissue_density: Vec<f32>,
    /// Viscoelastic relaxation times per section in seconds
    pub relaxation_times: Vec<f32>,
    /// Muscle fiber orientation as 3D unit vectors per section
    pub fiber_orientation: Vec<[f32; 3]>,
    /// Muscle activation levels per section (0-1)
    pub muscle_activation: Vec<f32>,
    /// Collagen fiber stiffness per section in Pa
    pub collagen_stiffness: Vec<f32>,
    /// Elastin fiber elastic modulus per section in Pa
    pub elastin_properties: Vec<f32>,
}

impl MolecularDynamicsModel {
    /// Create new molecular dynamics model
    ///
    /// # Arguments
    /// * `num_sections` - Number of vocal tract sections
    ///
    /// # Returns
    /// New model initialized at standard temperature and pressure
    pub fn new(num_sections: usize) -> crate::Result<Self> {
        Ok(Self {
            mean_free_path: 6.8e-8,   // meters at STP
            knudsen_number: 1e-4,     // Continuum regime
            collision_frequency: 5e9, // Hz
            kinetic_corrections: vec![1.0; num_sections],
            slip_coefficients: vec![0.0; num_sections],
            temperature_jump: vec![0.0; num_sections],
        })
    }

    /// Update molecular-scale effects based on local conditions
    ///
    /// # Arguments
    /// * `pressures` - Acoustic pressures per section (Pa)
    /// * `temperatures` - Temperatures per section (°C)
    /// * `characteristic_length` - Characteristic length scale in meters
    pub fn update_molecular_effects(
        &mut self,
        pressures: &[f32],
        temperatures: &[f32],
        characteristic_length: f32,
    ) {
        // Update Knudsen number based on local conditions
        self.knudsen_number = self.mean_free_path / characteristic_length;

        let len = self.kinetic_corrections.len().min(pressures.len());
        for (i, &pressure) in pressures.iter().enumerate().take(len) {
            let temperature = temperatures.get(i).unwrap_or(&300.0); // Default 300K

            // Update mean free path based on local pressure and temperature
            let local_mean_free_path = self.mean_free_path
                * (temperature / 273.15)
                * (101325.0 / pressure.abs().max(1000.0));

            let local_knudsen = local_mean_free_path / characteristic_length;

            // Gas kinetic corrections for slip flow regime
            if local_knudsen > 0.001 {
                // Slip flow regime - apply kinetic corrections
                self.kinetic_corrections[i] = 1.0 + 2.0 * local_knudsen;
                self.slip_coefficients[i] = 1.146 * local_knudsen;
                self.temperature_jump[i] = 2.18 * local_knudsen * temperature;
            } else {
                // Continuum regime
                self.kinetic_corrections[i] = 1.0;
                self.slip_coefficients[i] = 0.0;
                self.temperature_jump[i] = 0.0;
            }
        }

        // Update molecular collision frequency
        self.collision_frequency = 5e9 * (temperatures.first().unwrap_or(&300.0) / 300.0).sqrt();
    }

    /// Get viscosity correction factor for molecular effects
    ///
    /// # Arguments
    /// * `section` - Vocal tract section index
    ///
    /// # Returns
    /// Viscosity correction factor (1.0 = no correction)
    pub fn get_viscosity_correction(&self, section: usize) -> f32 {
        if section < self.kinetic_corrections.len() {
            self.kinetic_corrections[section]
        } else {
            1.0
        }
    }

    /// Calculate slip velocity at wall boundary
    ///
    /// # Arguments
    /// * `section` - Vocal tract section index
    /// * `wall_velocity_gradient` - Velocity gradient at wall (1/s)
    ///
    /// # Returns
    /// Slip velocity in m/s
    pub fn get_slip_velocity(&self, section: usize, wall_velocity_gradient: f32) -> f32 {
        if section < self.slip_coefficients.len() {
            self.slip_coefficients[section] * wall_velocity_gradient * self.mean_free_path
        } else {
            0.0
        }
    }

    /// Apply rarefaction effects to velocity
    ///
    /// # Arguments
    /// * `velocity` - Input velocity (m/s)
    /// * `section` - Vocal tract section index
    ///
    /// # Returns
    /// Corrected velocity with rarefaction effects
    pub fn apply_rarefaction_effects(&self, velocity: f32, section: usize) -> f32 {
        if section < self.kinetic_corrections.len() {
            velocity * self.kinetic_corrections[section]
        } else {
            velocity
        }
    }
}

impl TissueMechanicsModel {
    /// Create new tissue mechanics model with typical properties
    ///
    /// # Arguments
    /// * `num_sections` - Number of vocal tract sections
    ///
    /// # Returns
    /// New model with soft tissue mechanical properties
    pub fn new(num_sections: usize) -> crate::Result<Self> {
        // Initialize with typical vocal tract tissue properties
        let mut fiber_orientation = Vec::with_capacity(num_sections);
        for i in 0..num_sections {
            // Longitudinal fiber orientation by default
            let angle = (i as f32 / num_sections as f32) * 0.2; // Slight variation
            fiber_orientation.push([angle.cos(), angle.sin(), 0.0]);
        }

        Ok(Self {
            elastic_modulus: vec![1e6; num_sections], // 1 MPa - typical soft tissue
            poisson_ratio: vec![0.45; num_sections],  // Nearly incompressible
            tissue_density: vec![1050.0; num_sections], // kg/m³ - typical tissue density
            relaxation_times: vec![0.1; num_sections], // 100ms relaxation
            fiber_orientation,
            muscle_activation: vec![0.1; num_sections], // 10% baseline activation
            collagen_stiffness: vec![1e8; num_sections], // 100 MPa - collagen is much stiffer
            elastin_properties: vec![1e5; num_sections], // 100 kPa - elastin is softer
        })
    }

    /// Update tissue mechanics state based on deformation
    ///
    /// # Arguments
    /// * `strains` - Strain values per section (dimensionless)
    /// * `strain_rates` - Strain rate per section (1/s)
    /// * `dt` - Time step in seconds
    pub fn update_tissue_mechanics(&mut self, strains: &[f32], strain_rates: &[f32], dt: f32) {
        let len = self.elastic_modulus.len().min(strains.len());
        for (i, &strain) in strains.iter().enumerate().take(len) {
            let strain_rate = strain_rates.get(i).unwrap_or(&0.0);

            // Update muscle activation based on strain rate (active response)
            let activation_response = 0.1 * strain_rate.abs().tanh();
            self.muscle_activation[i] +=
                dt * (activation_response - self.muscle_activation[i]) / 0.1;
            self.muscle_activation[i] = self.muscle_activation[i].clamp(0.0, 1.0);

            // Update effective elastic modulus based on activation
            let base_modulus = 1e6; // Passive tissue modulus
            let active_modulus = 5e6; // Active muscle modulus
            self.elastic_modulus[i] =
                base_modulus + self.muscle_activation[i] * (active_modulus - base_modulus);

            // Strain-dependent modulus (tissue stiffening at high strain)
            if strain.abs() > 0.1 {
                let stiffening_factor = 1.0 + 10.0 * (strain.abs() - 0.1);
                self.elastic_modulus[i] *= stiffening_factor;
            }
        }
    }

    /// Calculate total stress from strain and strain rate
    ///
    /// # Arguments
    /// * `strain` - Strain value (dimensionless)
    /// * `strain_rate` - Strain rate (1/s)
    /// * `section` - Vocal tract section index
    ///
    /// # Returns
    /// Total stress in Pa
    pub fn calculate_stress(&self, strain: f32, strain_rate: f32, section: usize) -> f32 {
        if section >= self.elastic_modulus.len() {
            return 0.0;
        }

        // Elastic stress component
        let elastic_stress = self.elastic_modulus[section] * strain;

        // Viscous stress component (strain rate dependent)
        let viscosity = self.elastic_modulus[section] * self.relaxation_times[section];
        let viscous_stress = viscosity * strain_rate;

        // Muscle active stress
        let active_stress = self.muscle_activation[section] * 1e5 * strain.signum(); // 100 kPa active stress

        elastic_stress + viscous_stress + active_stress
    }

    /// Get effective tissue stiffness combining all components
    ///
    /// # Arguments
    /// * `section` - Vocal tract section index
    ///
    /// # Returns
    /// Effective stiffness (normalized)
    pub fn get_effective_stiffness(&self, section: usize) -> f32 {
        if section < self.elastic_modulus.len() {
            // Combine collagen and elastin contributions
            let collagen_contribution = 0.3 * self.collagen_stiffness[section];
            let elastin_contribution = 0.7 * self.elastin_properties[section];
            let muscle_contribution = self.muscle_activation[section] * 1e6;

            (collagen_contribution + elastin_contribution + muscle_contribution) / 1e6
        // Normalize
        } else {
            1.0
        }
    }

    /// Calculate anisotropic stress response based on fiber orientation
    ///
    /// # Arguments
    /// * `stress_tensor` - Stress tensor [σxx, σyy, σzz, τxy, τxz, τyz]
    /// * `section` - Vocal tract section index
    ///
    /// # Returns
    /// Modified stress tensor with anisotropic effects
    pub fn calculate_anisotropic_response(
        &self,
        stress_tensor: [f32; 6],
        section: usize,
    ) -> [f32; 6] {
        // Simplified anisotropic response based on fiber orientation
        // stress_tensor = [σxx, σyy, σzz, τxy, τxz, τyz]

        if section >= self.fiber_orientation.len() {
            return stress_tensor;
        }

        let fiber_dir = self.fiber_orientation[section];
        let fiber_stiffness_ratio = 3.0; // Fibers are 3x stiffer in fiber direction

        // Project stress onto fiber direction and apply anisotropic scaling
        let fiber_stress = stress_tensor[0] * fiber_dir[0] * fiber_dir[0]
            + stress_tensor[1] * fiber_dir[1] * fiber_dir[1]
            + stress_tensor[2] * fiber_dir[2] * fiber_dir[2];

        let mut anisotropic_stress = stress_tensor;

        // Enhance stress in fiber direction
        anisotropic_stress[0] *= 1.0 + (fiber_stiffness_ratio - 1.0) * fiber_dir[0] * fiber_dir[0];
        anisotropic_stress[1] *= 1.0 + (fiber_stiffness_ratio - 1.0) * fiber_dir[1] * fiber_dir[1];
        anisotropic_stress[2] *= 1.0 + (fiber_stiffness_ratio - 1.0) * fiber_dir[2] * fiber_dir[2];

        anisotropic_stress
    }

    /// Update fiber orientation due to large deformations
    ///
    /// # Arguments
    /// * `strain_tensor` - Strain tensor [εxx, εyy, εzz, γxy, γxz, γyz]
    /// * `section` - Vocal tract section index
    pub fn update_fiber_orientation(&mut self, strain_tensor: [f32; 6], section: usize) {
        // Fiber reorientation due to large deformations
        if section >= self.fiber_orientation.len() {
            return;
        }

        let strain_magnitude =
            (strain_tensor[0].powi(2) + strain_tensor[1].powi(2) + strain_tensor[2].powi(2)).sqrt();

        if strain_magnitude > 0.05 {
            // 5% strain threshold
            // Simple reorientation model - fibers tend to align with principal strain direction
            let principal_strain_dir = if strain_tensor[0].abs() > strain_tensor[1].abs() {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };

            // Gradually reorient fibers
            let reorientation_rate = 0.1 * strain_magnitude;
            for (i, &dir) in principal_strain_dir.iter().enumerate() {
                self.fiber_orientation[section][i] +=
                    reorientation_rate * (dir - self.fiber_orientation[section][i]);
            }

            // Normalize fiber direction
            let norm = (self.fiber_orientation[section][0].powi(2)
                + self.fiber_orientation[section][1].powi(2)
                + self.fiber_orientation[section][2].powi(2))
            .sqrt();

            if norm > 0.0 {
                for val in &mut self.fiber_orientation[section] {
                    *val /= norm;
                }
            }
        }
    }

    /// Calculate viscoelastic response from strain history
    ///
    /// # Arguments
    /// * `strain_history` - Historical strain values
    /// * `dt` - Time step in seconds
    /// * `section` - Vocal tract section index
    ///
    /// # Returns
    /// Viscoelastic stress in Pa
    pub fn get_viscoelastic_response(
        &self,
        strain_history: &[f32],
        dt: f32,
        section: usize,
    ) -> f32 {
        if section >= self.relaxation_times.len() || strain_history.is_empty() {
            return 0.0;
        }

        let relaxation_time = self.relaxation_times[section];
        let mut stress = 0.0;

        // Prony series approximation for viscoelastic response
        for (i, &strain) in strain_history.iter().enumerate() {
            let time = i as f32 * dt;
            let relaxation_function = (-time / relaxation_time).exp();
            stress += strain * relaxation_function;
        }

        stress * self.elastic_modulus[section] / strain_history.len() as f32
    }
}
