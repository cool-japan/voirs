//! Universal voice models for zero-shot learning

use super::database::SpeakerEmbedding;
use crate::Result;
use scirs2_core::Complex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Universal voice model for zero-shot learning
pub struct UniversalVoiceModel {
    /// Model parameters
    parameters: Arc<RwLock<ModelParameters>>,

    /// Feature extractors
    feature_extractors: HashMap<String, Box<dyn FeatureExtractor>>,

    /// Voice generators
    voice_generators: HashMap<String, Box<dyn VoiceGenerator>>,

    /// Model metadata
    metadata: ModelMetadata,
}

/// Model parameters
#[derive(Debug, Clone)]
pub struct ModelParameters {
    /// Embedding dimension
    pub embedding_dim: usize,

    /// Hidden layer sizes
    pub hidden_sizes: Vec<usize>,

    /// Activation functions
    pub activations: Vec<String>,

    /// Dropout rates
    pub dropout_rates: Vec<f32>,

    /// Model weights (simplified representation)
    pub weights: Vec<Vec<f32>>,

    /// Bias terms
    pub biases: Vec<Vec<f32>>,
}

/// Feature extractor trait
pub trait FeatureExtractor: Send + Sync {
    /// Extract features from audio
    fn extract_features(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<f32>>;

    /// Get feature dimension
    fn feature_dim(&self) -> usize;

    /// Get extractor name
    fn name(&self) -> &str;
}

/// Voice generator trait
pub trait VoiceGenerator: Send + Sync {
    /// Generate voice from features
    fn generate_voice(
        &self,
        features: &[f32],
        target_embedding: &SpeakerEmbedding,
    ) -> Result<Vec<f32>>;

    /// Get generator name
    fn name(&self) -> &str;

    /// Check if real-time capable
    fn is_realtime(&self) -> bool;
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,

    /// Model version
    pub version: String,

    /// Training information
    pub training_info: TrainingInfo,

    /// Performance benchmarks
    pub benchmarks: Vec<BenchmarkResult>,

    /// Supported features
    pub supported_features: Vec<String>,
}

/// Training information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingInfo {
    /// Training dataset size
    pub dataset_size: usize,

    /// Number of speakers
    pub num_speakers: usize,

    /// Training languages
    pub languages: Vec<String>,

    /// Training duration (hours)
    pub training_duration: f32,

    /// Model architecture
    pub architecture: String,
}

/// Benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: String,

    /// Score
    pub score: f32,

    /// Metric type
    pub metric_type: String,

    /// Test conditions
    pub conditions: HashMap<String, String>,

    /// Timestamp
    #[serde(
        skip_serializing,
        skip_deserializing,
        default = "std::time::Instant::now"
    )]
    pub timestamp: Instant,
}

/// Adapted model for neural adaptation
pub struct AdaptedModel {
    parameters: ModelParameters,
}

impl Default for UniversalVoiceModel {
    fn default() -> Self {
        Self::new()
    }
}

impl UniversalVoiceModel {
    /// Creates a new universal voice model with default parameters.
    ///
    /// Initializes the model with:
    /// - 256-dimensional embeddings
    /// - Three hidden layers (512, 256, 128 units)
    /// - ReLU and Tanh activations
    /// - Transformer-based architecture
    /// - Support for zero-shot learning and style transfer
    ///
    /// # Returns
    ///
    /// A new [`UniversalVoiceModel`] instance with default configuration.
    pub fn new() -> Self {
        Self {
            parameters: Arc::new(RwLock::new(ModelParameters {
                embedding_dim: 256,
                hidden_sizes: vec![512, 256, 128],
                activations: vec!["relu".to_string(), "relu".to_string(), "tanh".to_string()],
                dropout_rates: vec![0.1, 0.1, 0.0],
                weights: vec![vec![0.0; 512]; 3],
                biases: vec![vec![0.0; 512]; 3],
            })),
            feature_extractors: HashMap::new(),
            voice_generators: HashMap::new(),
            metadata: ModelMetadata {
                name: "UniversalVoiceModel".to_string(),
                version: "1.0.0".to_string(),
                training_info: TrainingInfo {
                    dataset_size: 10000,
                    num_speakers: 1000,
                    languages: vec!["en".to_string(), "es".to_string(), "fr".to_string()],
                    training_duration: 100.0,
                    architecture: "Transformer".to_string(),
                },
                benchmarks: Vec::new(),
                supported_features: vec!["zero_shot".to_string(), "style_transfer".to_string()],
            },
        }
    }
}

impl Default for AdaptedModel {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptedModel {
    /// Creates a new adapted model with default parameters.
    ///
    /// Initializes the model with:
    /// - 256-dimensional embeddings
    /// - Three hidden layers (512, 256, 128 units)
    /// - ReLU and Tanh activations
    /// - Zero-initialized weights and biases
    ///
    /// # Returns
    ///
    /// A new [`AdaptedModel`] instance ready for fine-tuning and audio generation.
    pub fn new() -> Self {
        Self {
            parameters: ModelParameters {
                embedding_dim: 256,
                hidden_sizes: vec![512, 256, 128],
                activations: vec!["relu".to_string(), "relu".to_string(), "tanh".to_string()],
                dropout_rates: vec![0.1, 0.1, 0.0],
                weights: vec![vec![0.0; 512]; 3],
                biases: vec![vec![0.0; 512]; 3],
            },
        }
    }

    /// Generates converted audio from a source waveform using the neural voice model.
    pub fn generate_audio(&self, source_audio: &[f32], _sample_rate: u32) -> Result<Vec<f32>> {
        let n = source_audio.len();
        if n < 512 {
            return Ok(source_audio.to_vec());
        }

        let fft_len = n.next_power_of_two();
        let mut padded: Vec<f64> = source_audio.iter().map(|&x| x as f64).collect();
        padded.resize(fft_len, 0.0);

        let mut spectrum =
            scirs2_fft::rfft(&padded, Some(fft_len)).map_err(|e| crate::Error::Processing {
                operation: "generate_audio_rfft".to_string(),
                message: format!("rfft failed: {e}"),
                context: None,
                recovery_suggestions: Box::new(Vec::new()),
            })?;

        let n_bins = fft_len / 2 + 1;
        let weights_layer = if self.parameters.weights.is_empty() {
            &[][..]
        } else {
            &self.parameters.weights[0]
        };
        let biases_layer = if self.parameters.biases.is_empty() {
            &[][..]
        } else {
            &self.parameters.biases[0]
        };

        let n_coeff = weights_layer.len().min(n_bins.min(256));

        for k in 0..n_bins {
            if k < n_coeff {
                let warp_gain = (weights_layer[k] * 0.5 + 1.0).clamp(0.1, 3.0) as f64;
                spectrum[k] = Complex::new(spectrum[k].re * warp_gain, spectrum[k].im * warp_gain);
            }
        }

        let bias_len = biases_layer.len().min(n_bins.min(256));
        for k in 0..bias_len {
            let phase_shift = (biases_layer[k] * 0.1).clamp(-PI / 4.0, PI / 4.0) as f64;
            let rotation = Complex::new(phase_shift.cos(), phase_shift.sin());
            spectrum[k] = Complex::new(
                spectrum[k].re * rotation.re - spectrum[k].im * rotation.im,
                spectrum[k].re * rotation.im + spectrum[k].im * rotation.re,
            );
        }

        let time_domain =
            scirs2_fft::irfft(&spectrum, Some(fft_len)).map_err(|e| crate::Error::Processing {
                operation: "generate_audio_irfft".to_string(),
                message: format!("irfft failed: {e}"),
                context: None,
                recovery_suggestions: Box::new(Vec::new()),
            })?;

        let output: Vec<f32> = time_domain
            .iter()
            .take(n)
            .map(|&x| (x as f32).clamp(-0.95, 0.95))
            .collect();

        Ok(output)
    }
}

impl Default for BenchmarkResult {
    fn default() -> Self {
        Self {
            name: String::new(),
            score: 0.0,
            metric_type: String::new(),
            conditions: HashMap::new(),
            timestamp: Instant::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_audio_length_preserved() {
        let model = AdaptedModel::new();
        let source: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        let generated = model.generate_audio(&source, 22050).unwrap();
        assert_eq!(generated.len(), source.len());
    }

    #[test]
    fn test_generate_audio_not_identity() {
        let mut model = AdaptedModel::new();
        model.parameters.weights[0][0] = 0.5;
        let source: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let generated = model.generate_audio(&source, 22050).unwrap();
        let diff: f32 = source
            .iter()
            .zip(generated.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "output should differ from input when weight != 0"
        );
    }

    #[test]
    fn test_generate_audio_short_input() {
        let model = AdaptedModel::new();
        let source: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let generated = model.generate_audio(&source, 22050).unwrap();
        assert_eq!(generated, source);
    }
}
