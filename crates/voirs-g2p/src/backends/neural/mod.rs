//! Neural network-based G2P backend implementation.
//!
//! This module provides LSTM-based neural network implementations for grapheme-to-phoneme
//! conversion with attention mechanisms and advanced training capabilities.

pub mod core;
pub mod training;

pub use core::{AttentionLayer, BeamCandidate, EnhancedDecoder, SimpleDecoder, SimpleEncoder};
pub use training::{LstmConfig, LstmTrainer, ModelMetadata, TrainingBatch, TrainingState};

use crate::{G2pError, Phoneme, Result};
use candle_core::Device;
use tracing::{debug, info};

/// Neural G2P backend using LSTM encoder-decoder with attention
pub struct NeuralG2pBackend {
    encoder: SimpleEncoder,
    decoder: SimpleDecoder,
    device: Device,
    config: LstmConfig,
}

impl NeuralG2pBackend {
    /// Create a new neural G2P backend
    pub fn new(config: LstmConfig) -> Result<Self> {
        let device = Device::Cpu; // Default to CPU for compatibility

        info!("Initializing Neural G2P backend with device: {:?}", device);

        // Create a dummy trainer to get access to model loading
        let trainer = LstmTrainer::new(device.clone(), config.clone());

        // Load or create models (for now, create new ones)
        let (encoder, decoder) = trainer
            .load_model(std::path::Path::new("/tmp/neural_g2p_model.bin"))
            .or_else(|_| {
                debug!("Model not found, creating new neural network models");
                // Create new models with default parameters
                let varmap = candle_nn::VarMap::new();
                let vb =
                    candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

                let encoder = SimpleEncoder::new(
                    config.vocab_size,
                    128,
                    config.hidden_size,
                    vb.pp("encoder"),
                )
                .map_err(|e| G2pError::ModelError(format!("Failed to create encoder: {e}")))?;

                let decoder = SimpleDecoder::new(
                    config.phoneme_vocab_size,
                    128,
                    config.hidden_size,
                    config.phoneme_vocab_size,
                    vb.pp("decoder"),
                )
                .map_err(|e| G2pError::ModelError(format!("Failed to create decoder: {e}")))?;

                Ok::<(SimpleEncoder, SimpleDecoder), G2pError>((encoder, decoder))
            })?;

        Ok(Self {
            encoder,
            decoder,
            device,
            config,
        })
    }

    /// Convert text to phonemes using neural network
    pub fn text_to_phonemes(&self, text: &str) -> Result<Vec<Phoneme>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        debug!("Converting text to phonemes: {}", text);

        // Convert text to character indices
        let char_indices: Vec<usize> = text
            .chars()
            .map(|c| (c as usize) % self.config.vocab_size)
            .collect();

        if char_indices.is_empty() {
            return Ok(Vec::new());
        }

        // Create input tensor with proper shape for embedding
        // Convert to one-hot encoding matrix [seq_len, vocab_size]
        let seq_len = char_indices.len();
        let mut one_hot = vec![0.0f32; seq_len * self.config.vocab_size];

        for (i, &idx) in char_indices.iter().enumerate() {
            let offset = i * self.config.vocab_size + idx;
            if offset < one_hot.len() {
                one_hot[offset] = 1.0;
            }
        }

        let input_tensor = candle_core::Tensor::new(one_hot.as_slice(), &self.device)
            .map_err(|e| G2pError::ModelError(format!("Failed to create input tensor: {e}")))?
            .reshape(&[seq_len, self.config.vocab_size])
            .map_err(|e| G2pError::ModelError(format!("Failed to reshape input tensor: {e}")))?;

        // Encode input
        let encoded = self
            .encoder
            .forward(&input_tensor)
            .map_err(|e| G2pError::ModelError(format!("Encoder forward failed: {e}")))?;

        // Decode to phoneme sequence
        let phoneme_indices = self
            .decoder
            .decode_sequence(&encoded, text.len() * 2)
            .map_err(|e| {
                G2pError::ModelError(format!("Decoder sequence generation failed: {e}"))
            })?;

        // Convert indices back to phonemes
        let phonemes: Vec<Phoneme> = phoneme_indices
            .into_iter()
            .map(|idx| {
                // Map index back to phoneme symbol (simplified mapping)
                let symbol = match idx % 44 {
                    0 => "AH",
                    1 => "AA",
                    2 => "AE",
                    3 => "AO",
                    4 => "AW",
                    5 => "AY",
                    6 => "EH",
                    7 => "ER",
                    8 => "EY",
                    9 => "IH",
                    10 => "IY",
                    11 => "OW",
                    12 => "OY",
                    13 => "UH",
                    14 => "UW",
                    15 => "B",
                    16 => "CH",
                    17 => "D",
                    18 => "DH",
                    19 => "F",
                    20 => "G",
                    21 => "HH",
                    22 => "JH",
                    23 => "K",
                    24 => "L",
                    25 => "M",
                    26 => "N",
                    27 => "NG",
                    28 => "P",
                    29 => "R",
                    30 => "S",
                    31 => "SH",
                    32 => "T",
                    33 => "TH",
                    34 => "V",
                    35 => "W",
                    36 => "Y",
                    37 => "Z",
                    38 => "ZH",
                    39 => "SIL", // Silence
                    40 => "SP",  // Short pause
                    _ => "AH",   // Default fallback
                };

                Phoneme {
                    symbol: symbol.to_string(),
                    ipa_symbol: Some(symbol.to_string()),
                    language_notation: None,
                    stress: 0,
                    syllable_position: crate::SyllablePosition::Standalone,
                    duration_ms: None,
                    confidence: 0.8, // Neural network confidence
                    phonetic_features: None,
                    custom_features: None,
                    is_word_boundary: false,
                    is_syllable_boundary: false,
                }
            })
            .collect();

        debug!("Generated {} phonemes for text: {}", phonemes.len(), text);
        Ok(phonemes)
    }

    /// Train the neural network with provided dataset
    pub async fn train(&mut self, dataset: &crate::models::TrainingDataset) -> Result<()> {
        info!(
            "Starting neural G2P training with {} examples",
            dataset.examples.len()
        );

        let mut trainer = LstmTrainer::new(self.device.clone(), self.config.clone());

        // Train the model
        let (new_encoder, new_decoder) = trainer
            .train_model(
                dataset, None, // No validation set for now
                10,   // epochs
                32,   // batch_size
            )
            .await?;

        // Update our models
        self.encoder = new_encoder;
        self.decoder = new_decoder;

        info!("Neural G2P training completed successfully");
        Ok(())
    }

    /// Get training statistics
    pub fn get_stats(&self) -> std::collections::HashMap<String, f32> {
        let mut stats = std::collections::HashMap::new();
        stats.insert("vocab_size".to_string(), self.config.vocab_size as f32);
        stats.insert("hidden_size".to_string(), self.config.hidden_size as f32);
        stats.insert(
            "device".to_string(),
            if matches!(self.device, Device::Cpu) {
                0.0
            } else {
                1.0
            },
        );
        stats
    }
}

impl Default for NeuralG2pBackend {
    fn default() -> Self {
        Self::new(LstmConfig::default()).expect("Failed to create default neural G2P backend")
    }
}

#[async_trait::async_trait]
impl crate::G2p for NeuralG2pBackend {
    async fn to_phonemes(
        &self,
        text: &str,
        _lang: Option<crate::LanguageCode>,
    ) -> Result<Vec<Phoneme>> {
        self.text_to_phonemes(text)
    }

    fn supported_languages(&self) -> Vec<crate::LanguageCode> {
        // For now, support English
        vec![crate::LanguageCode::EnUs]
    }

    fn metadata(&self) -> crate::G2pMetadata {
        crate::G2pMetadata {
            name: "Neural G2P Backend".to_string(),
            version: "1.0.0".to_string(),
            description: "LSTM-based neural network G2P backend".to_string(),
            supported_languages: self.supported_languages(),
            accuracy_scores: std::collections::HashMap::new(),
        }
    }
}
