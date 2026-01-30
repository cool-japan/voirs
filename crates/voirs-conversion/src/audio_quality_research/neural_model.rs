//! Neural quality prediction model

use std::collections::HashMap;

/// Neural quality prediction model
#[derive(Debug, Clone)]
pub struct NeuralQualityModel {
    /// Model weights for different features
    pub(crate) weights: HashMap<&'static str, f32>,
    /// Feature scaling parameters
    pub(crate) feature_scales: HashMap<&'static str, (f32, f32)>, // (mean, std)
    /// Model architecture parameters
    pub(crate) hidden_layers: Vec<usize>,
}

impl Default for NeuralQualityModel {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert("spectral_distortion", -0.7);
        weights.insert("temporal_coherence", 0.6);
        weights.insert("loudness_deviation", -0.5);
        weights.insert("harmonic_preservation", 0.8);
        weights.insert("noise_level", -0.4);

        let mut feature_scales = HashMap::new();
        feature_scales.insert("spectral_distortion", (0.15, 0.08));
        feature_scales.insert("temporal_coherence", (0.85, 0.12));
        feature_scales.insert("loudness_deviation", (0.1, 0.05));

        Self {
            weights,
            feature_scales,
            hidden_layers: vec![64, 32, 16],
        }
    }
}
