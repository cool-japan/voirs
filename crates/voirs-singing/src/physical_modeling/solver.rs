//! Multi-scale physics solver for enhanced accuracy
//!
//! This module contains the multi-scale physics solver that coordinates
//! molecular, tissue, and acoustic scale simulations.

use super::boundary_acoustic::AcousticPropagationModel;
use super::tissue_molecular::{MolecularDynamicsModel, TissueMechanicsModel};
use scirs2_core::Complex;

/// Multi-scale physics solver for enhanced accuracy
#[derive(Debug, Clone)]
pub struct MultiScalePhysicsSolver {
    /// Microscale molecular dynamics effects model
    pub molecular_effects: MolecularDynamicsModel,
    /// Mesoscale tissue mechanics model
    pub tissue_mechanics: TissueMechanicsModel,
    /// Macroscale acoustic wave propagation model
    pub acoustic_propagation: AcousticPropagationModel,
    /// Current time step size in seconds
    pub time_step: f32,
    /// Adaptive time stepping enabled flag
    pub adaptive_stepping: bool,
    /// Error tolerance for adaptive time stepping
    pub error_tolerance: f32,
    /// Maximum allowed time step size in seconds
    pub max_time_step: f32,
    /// Minimum allowed time step size in seconds
    pub min_time_step: f32,
}

impl MultiScalePhysicsSolver {
    /// Create new multi-scale physics solver
    ///
    /// # Arguments
    /// * `num_sections` - Number of vocal tract sections
    ///
    /// # Returns
    /// New solver with molecular, tissue, and acoustic models
    pub fn new(num_sections: usize) -> crate::Result<Self> {
        Ok(Self {
            molecular_effects: MolecularDynamicsModel::new(num_sections)?,
            tissue_mechanics: TissueMechanicsModel::new(num_sections)?,
            acoustic_propagation: AcousticPropagationModel::new()?,
            time_step: 1e-5, // 10 microseconds
            adaptive_stepping: true,
            error_tolerance: 1e-6,
            max_time_step: 1e-4, // 100 microseconds
            min_time_step: 1e-7, // 0.1 microseconds
        })
    }

    /// Solve coupled multi-scale physics equations
    ///
    /// # Arguments
    /// * `pressures` - Acoustic pressures (Pa) - modified in-place
    /// * `velocities` - Flow velocities (m/s) - modified in-place
    /// * `temperatures` - Temperature distribution (°C)
    /// * `areas` - Cross-sectional areas (m²)
    ///
    /// # Returns
    /// Ok on success, error if solver fails
    pub fn solve_coupled_physics(
        &mut self,
        pressures: &mut [f32],
        velocities: &mut [f32],
        temperatures: &[f32],
        areas: &[f32],
    ) -> crate::Result<()> {
        let dt = if self.adaptive_stepping {
            self.calculate_adaptive_time_step(velocities, pressures)
        } else {
            self.time_step
        };

        // Update molecular effects
        self.molecular_effects.update_molecular_effects(
            pressures,
            temperatures,
            self.calculate_characteristic_length(areas),
        );

        // Calculate strain and strain rates for tissue mechanics
        let strains = self.calculate_strains(areas);
        let strain_rates = self.calculate_strain_rates(&strains, dt);

        // Update tissue mechanics
        self.tissue_mechanics
            .update_tissue_mechanics(&strains, &strain_rates, dt);

        // Apply multi-scale coupling effects
        self.apply_scale_coupling(pressures, velocities, &strains, dt)?;

        Ok(())
    }

    fn calculate_adaptive_time_step(&self, velocities: &[f32], pressures: &[f32]) -> f32 {
        // Calculate CFL condition for stability
        let max_velocity = velocities.iter().map(|v| v.abs()).fold(0.0, f32::max);
        let sound_speed = 343.0; // m/s

        let cfl_dt = if max_velocity > 0.0 {
            0.1 / (max_velocity + sound_speed) // CFL number = 0.1
        } else {
            self.max_time_step
        };

        // Calculate dt based on pressure gradients
        let max_pressure_gradient = self.calculate_max_pressure_gradient(pressures);
        let pressure_dt = if max_pressure_gradient > 0.0 {
            self.error_tolerance / max_pressure_gradient
        } else {
            self.max_time_step
        };

        // Take minimum of constraints
        let adaptive_dt = cfl_dt.min(pressure_dt);
        adaptive_dt.clamp(self.min_time_step, self.max_time_step)
    }

    fn calculate_max_pressure_gradient(&self, pressures: &[f32]) -> f32 {
        let mut max_gradient: f32 = 0.0;

        for i in 1..pressures.len() {
            let gradient = (pressures[i] - pressures[i - 1]).abs();
            max_gradient = max_gradient.max(gradient);
        }

        max_gradient
    }

    fn calculate_characteristic_length(&self, areas: &[f32]) -> f32 {
        if areas.is_empty() {
            return 0.01; // Default 1cm
        }

        // Hydraulic diameter: 4*Area/Perimeter ≈ 2*sqrt(Area/π) for circular
        let avg_area = areas.iter().sum::<f32>() / areas.len() as f32;
        2.0 * (avg_area / std::f32::consts::PI).sqrt()
    }

    fn calculate_strains(&self, areas: &[f32]) -> Vec<f32> {
        let mut strains = Vec::with_capacity(areas.len());

        // Reference area (assumed neutral configuration)
        let reference_areas: Vec<f32> = (0..areas.len())
            .map(|i| 1.0 + 2.0 * (i as f32 / areas.len() as f32))
            .collect();

        for (i, &area) in areas.iter().enumerate() {
            let ref_area = reference_areas.get(i).unwrap_or(&1.0);
            let strain = (area - ref_area) / ref_area;
            strains.push(strain);
        }

        strains
    }

    fn calculate_strain_rates(&self, strains: &[f32], dt: f32) -> Vec<f32> {
        // For first timestep, assume zero strain rate
        // In practice, this would use strain history
        vec![0.0; strains.len()]
    }

    fn apply_scale_coupling(
        &mut self,
        pressures: &mut [f32],
        velocities: &mut [f32],
        strains: &[f32],
        dt: f32,
    ) -> crate::Result<()> {
        // Apply molecular corrections to velocities
        for (i, velocity) in velocities.iter_mut().enumerate() {
            let molecular_correction = self.molecular_effects.get_viscosity_correction(i);
            *velocity *= molecular_correction;

            // Apply rarefaction effects
            *velocity = self
                .molecular_effects
                .apply_rarefaction_effects(*velocity, i);
        }

        // Apply tissue mechanics effects to pressures
        for i in 0..pressures.len().min(strains.len()) {
            let strain = strains[i];
            let strain_rate = 0.0; // Simplified - would use actual strain rate history

            let tissue_stress = self
                .tissue_mechanics
                .calculate_stress(strain, strain_rate, i);
            let pressure_correction = tissue_stress * 0.001; // Scale factor

            pressures[i] += pressure_correction;
        }

        // Apply acoustic coupling effects
        self.apply_acoustic_coupling(pressures, velocities, dt)?;

        Ok(())
    }

    /// Apply frequency-dependent acoustic coupling to the pressure field.
    ///
    /// The pressure signal is transformed to the frequency domain with a real
    /// FFT ([`scirs2_fft::rfft`]), the acoustic propagation model's
    /// frequency-dependent attenuation is applied to the per-bin **magnitudes**
    /// while the original **phase** of each complex bin is preserved, and the
    /// modified complex spectrum is transformed back with the inverse real FFT
    /// ([`scirs2_fft::irfft`]).
    ///
    /// Preserving phase makes the forward → (modify) → inverse round-trip
    /// mathematically correct: with an identity coupling (no attenuation) the
    /// reconstructed signal equals the input up to numerical precision, instead
    /// of the previous magnitude-as-time-sample corruption that discarded all
    /// phase information.
    fn apply_acoustic_coupling(
        &mut self,
        pressures: &mut [f32],
        velocities: &mut [f32],
        dt: f32,
    ) -> crate::Result<()> {
        let _ = (velocities, dt);

        let n = pressures.len();
        if n == 0 {
            return Ok(());
        }

        // Forward transform: real signal -> `n / 2 + 1` complex bins.
        let spectrum = self.fft_transform(pressures)?;

        // Extract per-bin magnitudes for the attenuation model, which operates
        // on a real magnitude spectrum in-place.
        let mut magnitudes: Vec<f32> = spectrum.iter().map(|bin| bin.norm() as f32).collect();
        self.acoustic_propagation
            .apply_frequency_attenuation(&mut magnitudes);

        // Rebuild the complex spectrum from the attenuated magnitudes while
        // keeping the original phase of every bin (re-scale each bin by the
        // ratio of new to old magnitude). Bins with (near-)zero magnitude have
        // undefined phase and are left at zero.
        let scaled: Vec<Complex<f64>> = spectrum
            .iter()
            .zip(magnitudes.iter())
            .map(|(bin, &new_mag)| {
                let old_mag = bin.norm();
                if old_mag > f64::EPSILON {
                    let ratio = new_mag as f64 / old_mag;
                    Complex::new(bin.re * ratio, bin.im * ratio)
                } else {
                    Complex::new(0.0, 0.0)
                }
            })
            .collect();

        // Inverse transform back to the time domain (phase-preserving).
        let reconstructed = scirs2_fft::irfft(&scaled, Some(n)).map_err(|e| {
            crate::Error::Processing(format!("acoustic coupling inverse FFT failed: {e}"))
        })?;
        for (slot, value) in pressures.iter_mut().zip(reconstructed.iter()) {
            *slot = *value as f32;
        }

        Ok(())
    }

    /// Forward real FFT of a real-valued time-domain signal.
    ///
    /// Returns the `signal.len() / 2 + 1` non-redundant complex frequency bins
    /// using [`scirs2_fft::rfft`], preserving both magnitude and phase (the
    /// previous O(N²) DFT discarded phase by returning only magnitudes). An
    /// empty input yields an empty spectrum.
    fn fft_transform(&self, signal: &[f32]) -> crate::Result<Vec<Complex<f64>>> {
        if signal.is_empty() {
            return Ok(Vec::new());
        }
        scirs2_fft::rfft(signal, Some(signal.len())).map_err(|e| {
            crate::Error::Processing(format!("acoustic coupling forward FFT failed: {e}"))
        })
    }

    /// Estimate total computational cost for simulation
    ///
    /// # Arguments
    /// * `num_sections` - Number of vocal tract sections
    /// * `time_duration` - Simulation duration in seconds
    ///
    /// # Returns
    /// Estimated cost in relative units
    pub fn estimate_computational_cost(&self, num_sections: usize, time_duration: f32) -> f32 {
        let num_time_steps = time_duration / self.time_step;

        // Cost estimates (relative units)
        let molecular_cost = 10.0 * num_sections as f32; // Molecular dynamics
        let tissue_cost = 50.0 * num_sections as f32; // Tissue mechanics
        let acoustic_cost = 100.0 * num_sections as f32; // Acoustic propagation

        let total_cost_per_step = molecular_cost + tissue_cost + acoustic_cost;
        total_cost_per_step * num_time_steps
    }

    /// Optimize solver performance for target real-time factor
    ///
    /// # Arguments
    /// * `target_real_time_factor` - Target RTF (1.0 = real-time)
    pub fn optimize_performance(&mut self, target_real_time_factor: f32) {
        // Adjust time step for performance
        if target_real_time_factor < 1.0 {
            // Need to run faster than real-time - increase time step
            self.time_step = (self.time_step * 1.1).min(self.max_time_step);
        } else if target_real_time_factor > 2.0 {
            // Can afford more accuracy - decrease time step
            self.time_step = (self.time_step * 0.9).max(self.min_time_step);
        }

        // Adjust error tolerance
        if target_real_time_factor < 0.5 {
            self.error_tolerance *= 2.0; // Relax accuracy for speed
        } else if target_real_time_factor > 5.0 {
            self.error_tolerance *= 0.5; // Increase accuracy when possible
        }
    }

    /// Get current solver status and stability information
    ///
    /// # Returns
    /// Solver status with time step and stability information
    pub fn get_solver_status(&self) -> SolverStatus {
        SolverStatus {
            current_time_step: self.time_step,
            adaptive_stepping_enabled: self.adaptive_stepping,
            error_tolerance: self.error_tolerance,
            stability_margin: self.calculate_stability_margin(),
        }
    }

    fn calculate_stability_margin(&self) -> f32 {
        // Simplified stability analysis
        let cfl_margin = self.max_time_step / self.time_step;
        let error_margin = 1.0 / self.error_tolerance;

        cfl_margin.min(error_margin)
    }
}

/// Solver status information
#[derive(Debug, Clone)]
pub struct SolverStatus {
    /// Current time step size in seconds
    pub current_time_step: f32,
    /// Adaptive time stepping enabled flag
    pub adaptive_stepping_enabled: bool,
    /// Current error tolerance
    pub error_tolerance: f32,
    /// Stability margin (>1.0 = stable)
    pub stability_margin: f32,
}

impl SolverStatus {
    /// Check if solver is numerically stable
    ///
    /// # Returns
    /// True if stability margin > 1.0
    pub fn is_stable(&self) -> bool {
        self.stability_margin > 1.0
    }

    /// Get current performance level based on time step size
    ///
    /// # Returns
    /// Performance level (Fast/Balanced/HighAccuracy)
    pub fn performance_level(&self) -> PerformanceLevel {
        if self.current_time_step >= 1e-4 {
            PerformanceLevel::Fast
        } else if self.current_time_step >= 1e-5 {
            PerformanceLevel::Balanced
        } else {
            PerformanceLevel::HighAccuracy
        }
    }
}

/// Performance level indicators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceLevel {
    /// Fast mode with larger time steps
    Fast,
    /// Balanced mode with moderate time steps
    Balanced,
    /// High accuracy mode with small time steps
    HighAccuracy,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a deterministic test signal (sum of two sinusoids plus a DC
    /// offset) so that the round-trip exercises non-trivial phase.
    fn test_signal(n: usize) -> Vec<f32> {
        use std::f32::consts::PI;
        (0..n)
            .map(|i| {
                let t = i as f32;
                0.25 + (2.0 * PI * 5.0 * t / n as f32).sin()
                    + 0.5 * (2.0 * PI * 11.0 * t / n as f32 + 0.7).cos()
            })
            .collect()
    }

    #[test]
    fn test_acoustic_coupling_identity_roundtrip_preserves_phase() {
        let mut solver = MultiScalePhysicsSolver::new(4).expect("solver construction");

        // Identity coupling: zero attenuation across all bins so the
        // magnitude is unchanged and only the FFT round-trip is exercised.
        solver.acoustic_propagation.frequency_attenuation = vec![0.0; 4096];

        let n = 256;
        let input = test_signal(n);
        let mut pressures = input.clone();
        let mut velocities = vec![0.0f32; n];

        solver
            .apply_acoustic_coupling(&mut pressures, &mut velocities, 1e-4)
            .expect("acoustic coupling");

        // Forward rfft -> irfft with identity coupling must reconstruct the
        // input within numerical tolerance, proving phase is preserved (the
        // previous magnitude-as-time-sample code could not pass this).
        let mse: f64 = input
            .iter()
            .zip(pressures.iter())
            .map(|(&a, &b)| {
                let d = (a - b) as f64;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let rms = mse.sqrt();
        assert!(
            rms < 1e-4,
            "identity round-trip RMS error too large: {rms} (phase not preserved?)"
        );
    }

    #[test]
    fn test_fft_transform_bin_count_and_spectral_peak() {
        let solver = MultiScalePhysicsSolver::new(2).expect("solver construction");

        // A pure tone at exactly bin 8 of a 64-point FFT.
        let n = 64usize;
        let bin = 8usize;
        use std::f32::consts::PI;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * bin as f32 * i as f32 / n as f32).sin())
            .collect();

        let spectrum = solver.fft_transform(&signal).expect("forward fft");

        // rfft returns n / 2 + 1 non-redundant complex bins.
        assert_eq!(spectrum.len(), n / 2 + 1);

        // The magnitude must peak at the tone's bin.
        let peak = spectrum
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.norm().partial_cmp(&b.1.norm()).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(peak, bin, "spectral peak not at the tone's bin");
    }

    #[test]
    fn test_fft_transform_empty_signal() {
        let solver = MultiScalePhysicsSolver::new(1).expect("solver construction");
        let spectrum = solver.fft_transform(&[]).expect("empty fft");
        assert!(spectrum.is_empty());
    }
}
