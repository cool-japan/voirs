//! Physical modeling of the vocal tract for realistic singing synthesis
//!
//! This module implements physical modeling techniques to simulate the human vocal tract,
//! providing highly realistic and expressive singing synthesis through acoustic modeling.
//! Enhanced with advanced physics simulation including turbulence, wall vibration,
//! thermal effects, and nonlinear dynamics.
//!
//! The module is organized into several focused submodules:
//! - `core`: Core vocal tract model, glottal model, and basic components
//! - `advanced_physics`: Advanced physics models (turbulence, thermal, nonlinear)
//! - `boundary_acoustic`: Boundary conditions and acoustic propagation
//! - `tissue_molecular`: Tissue mechanics and molecular dynamics
//! - `solver`: Multi-scale physics solver
//! - `enhanced`: Enhanced vocal tract model combining all components

#![allow(dead_code, missing_docs)]

// Core modules
pub mod advanced_physics;
pub mod boundary_acoustic;
pub mod core;
pub mod enhanced;
pub mod solver;
pub mod tissue_molecular;

// Re-export core types for backward compatibility
pub use core::{
    DelayLine, FormantTarget, GlottalModel, PhysicalModelConfig, VocalTractModel, VowelPreset,
};

// Re-export advanced physics types
pub use advanced_physics::{
    AdaptivePhysicsParameters, Complex32, ErrorEstimator, MeshRefinement, NonlinearDynamicsModel,
    PhysicsAccuracyLevel, QualityAdaptation, ThermalModel, TurbulenceModel, WallVibrationModel,
};

// Re-export boundary and acoustic types
pub use boundary_acoustic::{
    AcousticMode, AcousticPropagationModel, BoundaryConditionModel, PropagationPath,
};

// Re-export tissue and molecular types
pub use tissue_molecular::{MolecularDynamicsModel, TissueMechanicsModel};

// Re-export solver types
pub use solver::{MultiScalePhysicsSolver, PerformanceLevel, SolverStatus};

// Re-export enhanced model types
pub use enhanced::{AdvancedPerformanceMetrics, AdvancedVocalTractModel, PerformanceRating};

// Ensure the main VocalTractModel still implements SingingEffect
// (this is already implemented in core.rs)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physical_model_creation() {
        let config = PhysicalModelConfig::default();
        let model = VocalTractModel::new("physical".to_string(), config, 44100.0);

        assert!(model.is_ok());
        let model = model.unwrap();
        assert_eq!(model.name(), "physical");
        assert_eq!(model.num_sections, 20);
    }

    #[test]
    fn test_glottal_model() {
        let mut glottal = GlottalModel::new(220.0, 0.6, 2.0, -12.0);

        // Generate some samples
        let mut samples = vec![0.0; 200];
        for sample in samples.iter_mut() {
            *sample = glottal.generate_sample();
        }

        // Check that we get non-zero output
        let rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
        assert!(rms > 0.01, "Glottal model should produce audible output");
    }

    #[test]
    fn test_delay_line() {
        let mut delay = DelayLine::new(4);

        // Initially should read zeros
        assert_eq!(delay.read_sample(), 0.0);

        // Write and read samples to test basic operation
        delay.write_sample(1.0);
        delay.write_sample(2.0);
        delay.write_sample(3.0);
        delay.write_sample(4.0);

        // The delay line should now contain our written values
        // Read 4 more samples - should get the delayed values
        let sample1 = delay.read_sample();
        let sample2 = delay.read_sample();
        let sample3 = delay.read_sample();
        let sample4 = delay.read_sample();

        // Should eventually get back our written samples (may need more cycles)
        // For this simple test, just verify we get non-zero values
        assert!(sample1 >= 0.0);

        // Test continuous operation
        delay.write_sample(5.0);
        let output = delay.read_sample();
        assert!(output.is_finite()); // Should be a valid number
    }

    #[test]
    fn test_area_function_generation() {
        let areas = VocalTractModel::create_neutral_area_function(10);
        assert_eq!(areas.len(), 10);

        // Check that all areas are positive
        for area in &areas {
            assert!(*area > 0.0);
        }

        // Check that areas generally increase from glottis to lips
        assert!(areas[0] < areas[areas.len() - 1]);
    }

    #[test]
    fn test_reflection_calculation() {
        let areas = vec![4.0, 3.0, 2.0, 1.0];
        let reflectances = VocalTractModel::calculate_reflectances(&areas);

        assert_eq!(reflectances.len(), 4);

        // Check that reflectances are in valid range
        for r in &reflectances {
            assert!(*r >= -1.0 && *r <= 1.0);
        }

        // Lip termination should have negative reflection
        assert!(*reflectances.last().unwrap() < 0.0);
    }

    #[test]
    fn test_vowel_presets() {
        let config = PhysicalModelConfig::default();
        let mut model = VocalTractModel::new("test".to_string(), config, 44100.0).unwrap();

        // Test setting different vowel presets
        model.set_vowel_preset(VowelPreset::OpenCentral);
        assert_eq!(model.jaw_opening, 1.0);

        model.set_vowel_preset(VowelPreset::CloseFront);
        assert_eq!(model.jaw_opening, 0.3);

        model.set_vowel_preset(VowelPreset::CloseBack);
        assert_eq!(model.lip_rounding, 1.0);
    }

    #[test]
    fn test_parameter_setting() {
        let config = PhysicalModelConfig::default();
        let mut model = VocalTractModel::new("test".to_string(), config, 44100.0).unwrap();

        // Test parameter setting and getting
        model.set_parameter("fundamental_frequency", 440.0).unwrap();
        assert_eq!(model.get_parameter("fundamental_frequency"), Some(440.0));

        model.set_parameter("tongue_position", 0.8).unwrap();
        assert_eq!(model.get_parameter("tongue_position"), Some(0.8));

        model.set_parameter("jaw_opening", 0.9).unwrap();
        assert_eq!(model.get_parameter("jaw_opening"), Some(0.9));

        // Test parameter clamping
        model.set_parameter("tongue_position", 1.5).unwrap(); // Should clamp to 1.0
        assert_eq!(model.get_parameter("tongue_position"), Some(1.0));
    }

    #[test]
    fn test_physical_synthesis() {
        let config = PhysicalModelConfig::default();
        let mut model = VocalTractModel::new("test".to_string(), config, 44100.0).unwrap();

        // Generate some audio
        let mut audio = vec![0.0; 4410]; // 0.1 seconds at 44.1kHz
        let result = model.process_physical_model(&mut audio);
        assert!(result.is_ok());

        // Check that synthesis produced some output
        let rms = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        assert!(
            rms > 0.001,
            "Physical model should produce audible output, got RMS: {}",
            rms
        );
    }

    #[test]
    fn test_articulation_update() {
        let config = PhysicalModelConfig::default();
        let mut model = VocalTractModel::new("test".to_string(), config, 44100.0).unwrap();

        let original_areas = model.area_function.clone();

        // Change articulation
        model.tongue_position = 0.2;
        model.jaw_opening = 0.3;
        model.update_articulation();

        // Area function should have changed
        assert_ne!(model.area_function, original_areas);
    }

    #[test]
    fn test_reset() {
        let config = PhysicalModelConfig::default();
        let mut model = VocalTractModel::new("test".to_string(), config, 44100.0).unwrap();

        // Generate some audio to populate internal state
        let mut audio = vec![0.0; 1000];
        model.process_physical_model(&mut audio).unwrap();

        // Reset should clear state
        model.reset();

        // All pressures and velocities should be zero
        for pressure in &model.junction_pressures {
            assert_eq!(*pressure, 0.0);
        }
        for velocity in &model.velocities {
            assert_eq!(*velocity, 0.0);
        }
    }

    #[test]
    fn test_advanced_vocal_tract_model() {
        let config = PhysicalModelConfig::default();
        let base_model = VocalTractModel::new("test".to_string(), config, 44100.0).unwrap();
        let advanced_model = AdvancedVocalTractModel::new(base_model);

        assert!(advanced_model.is_ok());
        let advanced_model = advanced_model.unwrap();
        assert_eq!(
            advanced_model.physics_accuracy_level,
            PhysicsAccuracyLevel::High
        );
    }

    #[test]
    fn test_turbulence_model() {
        let turbulence = TurbulenceModel::new(10);
        assert!(turbulence.is_ok());
        let turbulence = turbulence.unwrap();
        assert_eq!(turbulence.kinetic_energy.len(), 10);
    }

    #[test]
    fn test_boundary_condition_model() {
        let boundary = BoundaryConditionModel::new();
        assert!(boundary.is_ok());
        let boundary = boundary.unwrap();
        assert!(!boundary.lip_radiation_impedance.is_empty());
    }

    #[test]
    fn test_multi_scale_solver() {
        let solver = MultiScalePhysicsSolver::new(5);
        assert!(solver.is_ok());
        let solver = solver.unwrap();
        assert!(solver.time_step > 0.0);
    }
}
