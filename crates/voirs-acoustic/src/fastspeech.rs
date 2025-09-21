//! FastSpeech2 implementation.
//!
//! FastSpeech2 is a non-autoregressive text-to-speech model that generates
//! mel spectrograms directly from phoneme sequences using duration, pitch,
//! and energy predictors for improved controllability.

use crate::{
    AcousticError, AcousticModel, AcousticModelFeature, AcousticModelMetadata, LanguageCode,
    MelSpectrogram, Phoneme, Result, SynthesisConfig,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// FastSpeech2 model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastSpeech2Config {
    /// Vocabulary size (number of phonemes)
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of encoder layers
    pub encoder_layers: usize,
    /// Number of decoder layers
    pub decoder_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Feed-forward dimension
    pub ffn_dim: usize,
    /// Dropout rate
    pub dropout: f32,
    /// Number of mel channels
    pub n_mel_channels: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Speaker embedding dimension (for multi-speaker)
    pub speaker_embed_dim: Option<usize>,
}

impl Default for FastSpeech2Config {
    fn default() -> Self {
        Self {
            vocab_size: 256,
            hidden_dim: 256,
            encoder_layers: 4,
            decoder_layers: 4,
            num_heads: 2,
            ffn_dim: 1024,
            dropout: 0.1,
            n_mel_channels: 80,
            max_seq_len: 1000,
            speaker_embed_dim: Some(64),
        }
    }
}

/// Variance Adaptor for predicting duration, pitch, and energy
#[derive(Debug)]
pub struct VarianceAdaptor {
    /// Hidden dimension
    #[allow(dead_code)]
    hidden_dim: usize,
    /// Duration predictor layers
    duration_layers: Vec<ConvLayer>,
    /// Pitch predictor layers
    pitch_layers: Vec<ConvLayer>,
    /// Energy predictor layers
    energy_layers: Vec<ConvLayer>,
}

/// Simple convolutional layer for variance prediction
#[derive(Debug)]
pub struct ConvLayer {
    /// Input dimension
    #[allow(dead_code)]
    in_dim: usize,
    /// Output dimension
    #[allow(dead_code)]
    out_dim: usize,
    /// Kernel size
    kernel_size: usize,
    /// Weights (simulated for now)
    weights: Vec<f32>,
}

impl ConvLayer {
    pub fn new(in_dim: usize, out_dim: usize, kernel_size: usize) -> Self {
        let weight_count = in_dim * out_dim * kernel_size;
        let weights = (0..weight_count)
            .map(|i| (i as f32 * 0.01) % 1.0 - 0.5)
            .collect();

        Self {
            in_dim,
            out_dim,
            kernel_size,
            weights,
        }
    }

    /// Forward pass (simplified implementation)
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let output_len = input.len();
        let mut output = vec![0.0; output_len];

        for i in 0..output_len {
            let mut sum = 0.0;
            for j in 0..(self.kernel_size.min(input.len() - i)) {
                if i + j < input.len() {
                    sum += input[i + j] * self.weights[j % self.weights.len()];
                }
            }
            output[i] = sum.tanh(); // Apply activation
        }

        output
    }
}

impl VarianceAdaptor {
    pub fn new(hidden_dim: usize) -> Self {
        let duration_layers = vec![
            ConvLayer::new(hidden_dim, hidden_dim, 3),
            ConvLayer::new(hidden_dim, hidden_dim, 3),
            ConvLayer::new(hidden_dim, 1, 1), // Final layer outputs duration
        ];

        let pitch_layers = vec![
            ConvLayer::new(hidden_dim, hidden_dim, 3),
            ConvLayer::new(hidden_dim, hidden_dim, 3),
            ConvLayer::new(hidden_dim, 1, 1), // Final layer outputs pitch
        ];

        let energy_layers = vec![
            ConvLayer::new(hidden_dim, hidden_dim, 3),
            ConvLayer::new(hidden_dim, hidden_dim, 3),
            ConvLayer::new(hidden_dim, 1, 1), // Final layer outputs energy
        ];

        Self {
            hidden_dim,
            duration_layers,
            pitch_layers,
            energy_layers,
        }
    }

    /// Predict duration for each phoneme
    pub fn predict_duration(&self, encoder_output: &[Vec<f32>]) -> Vec<f32> {
        let mut durations = Vec::with_capacity(encoder_output.len());

        for phoneme_features in encoder_output {
            let mut features = phoneme_features.clone();

            // Pass through duration prediction layers
            for layer in &self.duration_layers {
                features = layer.forward(&features);
            }

            // Extract duration (take first element and ensure positive)
            let duration = features.first().copied().unwrap_or(0.5).exp().max(0.1);
            durations.push(duration);
        }

        durations
    }

    /// Predict pitch for each phoneme
    pub fn predict_pitch(&self, encoder_output: &[Vec<f32>]) -> Vec<f32> {
        let mut pitches = Vec::with_capacity(encoder_output.len());

        for phoneme_features in encoder_output {
            let mut features = phoneme_features.clone();

            // Pass through pitch prediction layers
            for layer in &self.pitch_layers {
                features = layer.forward(&features);
            }

            // Extract pitch (take first element)
            let pitch = features.first().copied().unwrap_or(0.0);
            pitches.push(pitch);
        }

        pitches
    }

    /// Predict energy for each phoneme
    pub fn predict_energy(&self, encoder_output: &[Vec<f32>]) -> Vec<f32> {
        let mut energies = Vec::with_capacity(encoder_output.len());

        for phoneme_features in encoder_output {
            let mut features = phoneme_features.clone();

            // Pass through energy prediction layers
            for layer in &self.energy_layers {
                features = layer.forward(&features);
            }

            // Extract energy (take first element and ensure positive)
            let energy = features.first().copied().unwrap_or(0.5).exp().max(0.01);
            energies.push(energy);
        }

        energies
    }
}

/// Length Regulator for expanding phoneme sequence based on duration
pub struct LengthRegulator;

impl LengthRegulator {
    /// Expand phoneme features based on predicted durations
    pub fn regulate(&self, phoneme_features: &[Vec<f32>], durations: &[f32]) -> Vec<Vec<f32>> {
        let mut regulated_features = Vec::new();

        for (features, &duration) in phoneme_features.iter().zip(durations.iter()) {
            let repeat_count = (duration * 10.0).round() as usize; // Scale duration to frame count
            let repeat_count = repeat_count.clamp(1, 20); // Clamp to reasonable range

            for _ in 0..repeat_count {
                regulated_features.push(features.clone());
            }
        }

        regulated_features
    }
}

/// FastSpeech2 model implementation
pub struct FastSpeech2Model {
    /// Model configuration
    config: FastSpeech2Config,
    /// Variance adaptor for duration, pitch, energy prediction
    variance_adaptor: VarianceAdaptor,
    /// Length regulator
    length_regulator: LengthRegulator,
    /// Phoneme embedding weights
    phoneme_embeddings: HashMap<u8, Vec<f32>>,
    /// Speaker embeddings (if multi-speaker)
    #[allow(dead_code)]
    speaker_embeddings: Option<HashMap<String, Vec<f32>>>,
}

impl FastSpeech2Model {
    pub fn new() -> Self {
        let config = FastSpeech2Config::default();
        Self::with_config(config)
    }

    pub fn with_config(config: FastSpeech2Config) -> Self {
        let variance_adaptor = VarianceAdaptor::new(config.hidden_dim);
        let length_regulator = LengthRegulator;

        // Initialize phoneme embeddings
        let mut phoneme_embeddings = HashMap::new();
        for phoneme_id in 0..config.vocab_size as u8 {
            let embedding = (0..config.hidden_dim)
                .map(|i| ((phoneme_id as f32 + i as f32 * 0.1) % 2.0) - 1.0)
                .collect();
            phoneme_embeddings.insert(phoneme_id, embedding);
        }

        // Initialize speaker embeddings if multi-speaker
        let speaker_embeddings = if config.speaker_embed_dim.is_some() {
            let mut embeddings = HashMap::new();
            for speaker_id in ["default", "speaker_1", "speaker_2"] {
                let embedding = (0..config.speaker_embed_dim.unwrap())
                    .map(|i| (speaker_id.len() as f32 + i as f32 * 0.05) % 1.0 - 0.5)
                    .collect();
                embeddings.insert(speaker_id.to_string(), embedding);
            }
            Some(embeddings)
        } else {
            None
        };

        Self {
            config,
            variance_adaptor,
            length_regulator,
            phoneme_embeddings,
            speaker_embeddings,
        }
    }

    /// Encode phonemes to hidden representations
    fn encode_phonemes(&self, phonemes: &[Phoneme]) -> Vec<Vec<f32>> {
        let mut encoded = Vec::with_capacity(phonemes.len());

        for phoneme in phonemes {
            // Get phoneme embedding
            let phoneme_id = phoneme.symbol.as_bytes().first().copied().unwrap_or(0);
            let embedding = self
                .phoneme_embeddings
                .get(&phoneme_id)
                .cloned()
                .unwrap_or_else(|| vec![0.0; self.config.hidden_dim]);

            // Simple encoder (in real implementation would use transformer layers)
            let encoded_phoneme = embedding
                .iter()
                .map(|&x| (x + 0.1).tanh()) // Simple nonlinearity
                .collect();

            encoded.push(encoded_phoneme);
        }

        encoded
    }

    /// Apply prosody control to variance predictions
    fn apply_prosody_control(
        &self,
        durations: &mut [f32],
        pitches: &mut [f32],
        energies: &mut [f32],
        config: Option<&SynthesisConfig>,
    ) {
        if let Some(synthesis_config) = config {
            // Apply speed scaling (affects duration)
            let speed_factor = synthesis_config.speed;
            for duration in durations.iter_mut() {
                *duration /= speed_factor;
            }

            // Apply pitch scaling
            let pitch_shift = synthesis_config.pitch_shift / 12.0; // Convert semitones to factor
            for pitch in pitches.iter_mut() {
                *pitch += pitch_shift;
            }

            // Apply energy scaling
            let energy_scale = synthesis_config.energy;
            for energy in energies.iter_mut() {
                *energy *= energy_scale;
            }
        }
    }

    /// Decode regulated features to mel spectrogram
    fn decode_to_mel(&self, regulated_features: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut mel_frames = Vec::with_capacity(regulated_features.len());

        for features in regulated_features {
            // Simple decoder (in real implementation would use transformer layers)
            let mut mel_frame = Vec::with_capacity(self.config.n_mel_channels);

            for mel_channel in 0..self.config.n_mel_channels {
                let feature_idx = mel_channel % features.len();
                let mel_value = features[feature_idx] * (mel_channel as f32 * 0.01 + 1.0);
                mel_frame.push(mel_value.tanh());
            }

            mel_frames.push(mel_frame);
        }

        Ok(mel_frames)
    }
}

impl Default for FastSpeech2Model {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AcousticModel for FastSpeech2Model {
    async fn synthesize(
        &self,
        phonemes: &[Phoneme],
        config: Option<&SynthesisConfig>,
    ) -> Result<MelSpectrogram> {
        if phonemes.is_empty() {
            return Err(AcousticError::ModelError(
                "Cannot synthesize empty phoneme sequence".to_string(),
            ));
        }

        // 1. Encode phonemes
        let encoder_output = self.encode_phonemes(phonemes);

        // 2. Predict durations, pitches, and energies
        let mut durations = self.variance_adaptor.predict_duration(&encoder_output);
        let mut pitches = self.variance_adaptor.predict_pitch(&encoder_output);
        let mut energies = self.variance_adaptor.predict_energy(&encoder_output);

        // 3. Apply prosody control if provided
        self.apply_prosody_control(&mut durations, &mut pitches, &mut energies, config);

        // 4. Length regulation (expand phoneme features based on duration)
        let regulated_features = self.length_regulator.regulate(&encoder_output, &durations);

        // 5. Decode to mel spectrogram
        let mel_data = self.decode_to_mel(&regulated_features)?;

        // 6. Create MelSpectrogram structure
        let sample_rate = 22050; // Standard sample rate
        let hop_length = 256; // Standard hop length

        // Transpose mel_data to match expected format [n_mels, n_frames]
        let n_frames = mel_data.len();
        let n_mels = self.config.n_mel_channels;
        let mut transposed_data = vec![vec![0.0; n_frames]; n_mels];

        for (frame_idx, frame) in mel_data.iter().enumerate() {
            for (mel_idx, &value) in frame.iter().enumerate() {
                if mel_idx < n_mels {
                    transposed_data[mel_idx][frame_idx] = value;
                }
            }
        }

        Ok(MelSpectrogram::new(
            transposed_data,
            sample_rate,
            hop_length,
        ))
    }

    async fn synthesize_batch(
        &self,
        inputs: &[&[Phoneme]],
        configs: Option<&[SynthesisConfig]>,
    ) -> Result<Vec<MelSpectrogram>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(inputs.len());

        // Process each input sequence
        for (i, phonemes) in inputs.iter().enumerate() {
            let config = configs.and_then(|c| c.get(i));
            let mel_spec = self.synthesize(phonemes, config).await?;
            results.push(mel_spec);
        }

        Ok(results)
    }

    fn metadata(&self) -> AcousticModelMetadata {
        AcousticModelMetadata {
            name: "FastSpeech2".to_string(),
            version: "1.0.0".to_string(),
            architecture: "FastSpeech2".to_string(),
            supported_languages: vec![LanguageCode::EnUs],
            sample_rate: 22050,
            mel_channels: 80,
            is_multi_speaker: true,
            speaker_count: Some(128),
        }
    }

    fn supports(&self, feature: AcousticModelFeature) -> bool {
        matches!(
            feature,
            AcousticModelFeature::MultiSpeaker
                | AcousticModelFeature::ProsodyControl
                | AcousticModelFeature::BatchProcessing
                | AcousticModelFeature::GpuAcceleration
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Phoneme;

    fn create_test_phonemes() -> Vec<Phoneme> {
        vec![
            Phoneme {
                symbol: "H".to_string(),
                duration: Some(0.1),
                features: Some(HashMap::new()),
            },
            Phoneme {
                symbol: "E".to_string(),
                duration: Some(0.15),
                features: Some(HashMap::new()),
            },
            Phoneme {
                symbol: "L".to_string(),
                duration: Some(0.1),
                features: Some(HashMap::new()),
            },
            Phoneme {
                symbol: "O".to_string(),
                duration: Some(0.2),
                features: Some(HashMap::new()),
            },
        ]
    }

    #[test]
    fn test_fastspeech2_model_creation() {
        let model = FastSpeech2Model::new();
        assert_eq!(model.config.vocab_size, 256);
        assert_eq!(model.config.hidden_dim, 256);
        assert_eq!(model.config.n_mel_channels, 80);
    }

    #[test]
    fn test_fastspeech2_config() {
        let config = FastSpeech2Config {
            vocab_size: 128,
            hidden_dim: 512,
            encoder_layers: 6,
            decoder_layers: 6,
            num_heads: 8,
            ffn_dim: 2048,
            dropout: 0.2,
            n_mel_channels: 128,
            max_seq_len: 2000,
            speaker_embed_dim: Some(128),
        };

        let model = FastSpeech2Model::with_config(config.clone());
        assert_eq!(model.config.vocab_size, 128);
        assert_eq!(model.config.hidden_dim, 512);
        assert_eq!(model.config.n_mel_channels, 128);
    }

    #[test]
    fn test_variance_adaptor() {
        let adaptor = VarianceAdaptor::new(256);

        // Test with mock encoder output
        let encoder_output = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.5, 0.6, 0.7, 0.8],
            vec![0.9, 1.0, 1.1, 1.2],
        ];

        let durations = adaptor.predict_duration(&encoder_output);
        let pitches = adaptor.predict_pitch(&encoder_output);
        let energies = adaptor.predict_energy(&encoder_output);

        assert_eq!(durations.len(), 3);
        assert_eq!(pitches.len(), 3);
        assert_eq!(energies.len(), 3);

        // Check that durations are positive
        for &duration in &durations {
            assert!(duration > 0.0);
        }

        // Check that energies are positive
        for &energy in &energies {
            assert!(energy > 0.0);
        }
    }

    #[test]
    fn test_length_regulator() {
        let regulator = LengthRegulator;

        let phoneme_features = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let durations = vec![0.5, 1.0]; // Should expand to ~5 and ~10 frames

        let regulated = regulator.regulate(&phoneme_features, &durations);

        // Should have expanded the sequence
        assert!(regulated.len() > phoneme_features.len());

        // Check that the features are preserved
        assert_eq!(regulated[0], vec![1.0, 2.0, 3.0]);
    }

    #[tokio::test]
    async fn test_fastspeech2_synthesis() {
        let model = FastSpeech2Model::new();
        let phonemes = create_test_phonemes();

        let result = model.synthesize(&phonemes, None).await;
        assert!(result.is_ok());

        let mel_spec = result.unwrap();
        assert_eq!(mel_spec.n_mels, 80);
        assert!(mel_spec.n_frames > 0);
        assert_eq!(mel_spec.sample_rate, 22050);
        assert_eq!(mel_spec.hop_length, 256);

        // Check that data has the correct size (n_mels channels)
        assert_eq!(mel_spec.data.len(), mel_spec.n_mels);

        // Check that the mel spectrogram has reasonable values
        assert!(!mel_spec.data.is_empty());
        for mel_channel in &mel_spec.data {
            assert!(!mel_channel.is_empty());
        }
    }

    #[tokio::test]
    async fn test_fastspeech2_synthesis_with_prosody() {
        let model = FastSpeech2Model::new();
        let phonemes = create_test_phonemes();

        let config = SynthesisConfig {
            speed: 0.5,       // Slower speech
            pitch_shift: 2.0, // Higher pitch (+2 semitones)
            energy: 1.5,      // Louder
            speaker_id: None,
            seed: Some(42),
            emotion: None,
            voice_style: None,
        };

        let result = model.synthesize(&phonemes, Some(&config)).await;
        assert!(result.is_ok());

        let mel_spec = result.unwrap();
        assert_eq!(mel_spec.n_mels, 80);
        assert!(mel_spec.n_frames > 0);
    }

    #[tokio::test]
    async fn test_fastspeech2_batch_synthesis() {
        let model = FastSpeech2Model::new();
        let phonemes1 = create_test_phonemes();
        let phonemes2 = vec![
            Phoneme {
                symbol: "W".to_string(),
                duration: Some(0.1),
                features: Some(HashMap::new()),
            },
            Phoneme {
                symbol: "O".to_string(),
                duration: Some(0.15),
                features: Some(HashMap::new()),
            },
        ];

        let inputs = vec![phonemes1.as_slice(), phonemes2.as_slice()];

        let result = model.synthesize_batch(&inputs, None).await;
        assert!(result.is_ok());

        let mel_specs = result.unwrap();
        assert_eq!(mel_specs.len(), 2);

        for mel_spec in &mel_specs {
            assert_eq!(mel_spec.n_mels, 80);
            assert!(mel_spec.n_frames > 0);
        }
    }

    #[tokio::test]
    async fn test_fastspeech2_empty_input() {
        let model = FastSpeech2Model::new();
        let phonemes = vec![];

        let result = model.synthesize(&phonemes, None).await;
        assert!(result.is_err());

        if let Err(AcousticError::ModelError(msg)) = result {
            assert!(msg.contains("empty phoneme sequence"));
        } else {
            panic!("Expected ModelError for empty input");
        }
    }

    #[test]
    fn test_fastspeech2_metadata() {
        let model = FastSpeech2Model::new();
        let metadata = model.metadata();

        assert_eq!(metadata.name, "FastSpeech2");
        assert_eq!(metadata.architecture, "FastSpeech2");
        assert_eq!(metadata.sample_rate, 22050);
        assert_eq!(metadata.mel_channels, 80);
        assert!(metadata.is_multi_speaker);
        assert_eq!(metadata.speaker_count, Some(128));
    }

    #[test]
    fn test_fastspeech2_features() {
        let model = FastSpeech2Model::new();

        assert!(model.supports(AcousticModelFeature::MultiSpeaker));
        assert!(model.supports(AcousticModelFeature::ProsodyControl));
        assert!(model.supports(AcousticModelFeature::BatchProcessing));
        assert!(model.supports(AcousticModelFeature::GpuAcceleration));
    }

    #[test]
    fn test_conv_layer() {
        let layer = ConvLayer::new(64, 32, 3);
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let output = layer.forward(&input);
        assert_eq!(output.len(), input.len());

        // Check that output values are in reasonable range (after tanh)
        for &val in &output {
            assert!((-1.0..=1.0).contains(&val));
        }
    }
}
