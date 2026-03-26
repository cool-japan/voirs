//! OxiONNX backend for voice cloning.
//!
//! This module provides ONNX-based speaker encoding and voice cloning,
//! enabling high-performance inference using pre-trained voice cloning models.

use crate::{Error, Result};
use oxionnx::{OptLevel, Session, Tensor};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};
use tracing::{debug, info, warn};

/// Default speaker embedding dimensionality
pub const ONNX_SPEAKER_EMBEDDING_DIM: usize = 256;

// ---------------------------------------------------------------------------
// OnnxSpeakerEncoder
// ---------------------------------------------------------------------------

/// Configuration for the ONNX speaker encoder
#[derive(Debug, Clone)]
pub struct OnnxSpeakerEncoderConfig {
    /// Path to the ONNX model file
    pub model_path: PathBuf,

    /// ONNX graph optimization level
    pub opt_level: OptLevel,

    /// Enable per-node profiling during inference
    pub enable_profiling: bool,

    /// Enable memory pool for buffer reuse
    pub enable_memory_pool: bool,

    /// Number of mel frequency bins (default: 80)
    pub mel_bins: usize,

    /// Speaker embedding dimensionality (default: 256)
    pub embedding_dim: usize,
}

impl Default for OnnxSpeakerEncoderConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            opt_level: OptLevel::All,
            enable_profiling: false,
            enable_memory_pool: true,
            mel_bins: 80,
            embedding_dim: ONNX_SPEAKER_EMBEDDING_DIM,
        }
    }
}

/// Information about the loaded ONNX speaker encoder model
#[derive(Debug, Clone)]
pub struct SpeakerEncoderModelInfo {
    /// Model file path
    pub model_path: PathBuf,

    /// Input tensor names
    pub input_names: Vec<String>,

    /// Output tensor names
    pub output_names: Vec<String>,

    /// Number of mel frequency bins
    pub mel_bins: usize,

    /// Speaker embedding dimensionality
    pub embedding_dim: usize,
}

/// ONNX-based speaker encoder
///
/// Extracts speaker embeddings from mel spectrograms.
/// Input shape: `[1, mel_bins, T]`
/// Output shape: `[1, embedding_dim]`
pub struct OnnxSpeakerEncoder {
    /// OxiONNX inference session
    session: Arc<RwLock<Session>>,

    /// Encoder configuration
    config: OnnxSpeakerEncoderConfig,

    /// Model information
    model_info: SpeakerEncoderModelInfo,
}

impl OnnxSpeakerEncoder {
    /// Create a new ONNX speaker encoder from the given configuration.
    pub fn new(config: OnnxSpeakerEncoderConfig) -> Result<Self> {
        info!(
            "Initializing OxiONNX speaker encoder from {:?}",
            config.model_path
        );

        let mut builder = Session::builder()
            .with_optimization_level(config.opt_level)
            .with_memory_pool(config.enable_memory_pool);
        if config.enable_profiling {
            builder = builder.with_profiling();
        }
        let session = builder
            .load(&config.model_path)
            .map_err(|e| Error::Model(format!("Failed to load ONNX speaker encoder: {}", e)))?;

        let input_names: Vec<String> = session.input_names().to_vec();
        let output_names: Vec<String> = session.output_names().to_vec();

        debug!("Speaker encoder inputs: {:?}", input_names);
        debug!("Speaker encoder outputs: {:?}", output_names);

        let model_info = SpeakerEncoderModelInfo {
            model_path: config.model_path.clone(),
            input_names,
            output_names,
            mel_bins: config.mel_bins,
            embedding_dim: config.embedding_dim,
        };

        info!("OxiONNX speaker encoder loaded successfully");

        Ok(Self {
            session: Arc::new(RwLock::new(session)),
            config,
            model_info,
        })
    }

    /// Load a speaker encoder from a model file path with default configuration.
    pub fn from_path(model_path: impl AsRef<Path>) -> Result<Self> {
        let config = OnnxSpeakerEncoderConfig {
            model_path: model_path.as_ref().to_path_buf(),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Extract a speaker embedding from a mel spectrogram.
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram data with shape `[mel_bins, T]` in row-major order.
    /// * `time_steps` - Number of time frames.
    ///
    /// # Returns
    /// Speaker embedding vector of length `embedding_dim`.
    pub fn encode_speaker(&self, mel: &[f32], time_steps: usize) -> Result<Vec<f32>> {
        let mel_bins = self.config.mel_bins;
        let expected_len = mel_bins * time_steps;
        if mel.len() != expected_len {
            return Err(Error::Embedding(format!(
                "Mel spectrogram size mismatch: expected {} ({}x{}), got {}",
                expected_len,
                mel_bins,
                time_steps,
                mel.len()
            )));
        }

        let input_tensor = Tensor::new(mel.to_vec(), vec![1, mel_bins, time_steps]);

        let input_name = self
            .model_info
            .input_names
            .first()
            .ok_or_else(|| Error::Embedding("Model has no input names".to_string()))?;

        let mut owned_inputs: HashMap<String, Tensor> = HashMap::new();
        owned_inputs.insert(input_name.clone(), input_tensor);
        let ref_inputs: HashMap<&str, Tensor> = owned_inputs
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();

        let session = self
            .session
            .read()
            .map_err(|e| Error::Embedding(format!("Failed to acquire session read lock: {}", e)))?;

        let outputs = session
            .run(&ref_inputs)
            .map_err(|e| Error::Embedding(format!("Speaker encoding inference failed: {}", e)))?;

        let output_name = self
            .model_info
            .output_names
            .first()
            .ok_or_else(|| Error::Embedding("Model has no output names".to_string()))?;

        let output_tensor = outputs.get(output_name).ok_or_else(|| {
            Error::Embedding(format!("Output '{}' not found in results", output_name))
        })?;

        let embedding = output_tensor.data.clone();

        // Normalize embedding to unit length
        let norm: f32 = embedding.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let normalized = if norm > 1e-8 {
            embedding.iter().map(|&v| v / norm).collect()
        } else {
            warn!("Speaker embedding has near-zero norm; returning unnormalized");
            embedding
        };

        Ok(normalized)
    }

    /// Return information about the loaded model.
    pub fn model_info(&self) -> &SpeakerEncoderModelInfo {
        &self.model_info
    }

    /// Return the encoder configuration.
    pub fn config(&self) -> &OnnxSpeakerEncoderConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// OnnxVoiceCloner
// ---------------------------------------------------------------------------

/// Configuration for the ONNX voice cloner
#[derive(Debug, Clone)]
pub struct OnnxVoiceClonerConfig {
    /// Path to the ONNX model file
    pub model_path: PathBuf,

    /// ONNX graph optimization level
    pub opt_level: OptLevel,

    /// Enable per-node profiling during inference
    pub enable_profiling: bool,

    /// Enable memory pool for buffer reuse
    pub enable_memory_pool: bool,

    /// Speaker embedding dimensionality (default: 256)
    pub embedding_dim: usize,

    /// Number of mel frequency bins in the output (default: 80)
    pub mel_bins: usize,

    /// Maximum input sequence length
    pub max_sequence_length: usize,
}

impl Default for OnnxVoiceClonerConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            opt_level: OptLevel::All,
            enable_profiling: false,
            enable_memory_pool: true,
            embedding_dim: ONNX_SPEAKER_EMBEDDING_DIM,
            mel_bins: 80,
            max_sequence_length: 512,
        }
    }
}

/// Result of voice cloning synthesis
#[derive(Debug, Clone)]
pub struct VoiceCloningResult {
    /// Generated mel spectrogram in row-major `[mel_bins, T]` layout
    pub mel: Vec<f32>,

    /// Number of mel frequency bins
    pub mel_bins: usize,

    /// Number of generated time frames
    pub time_steps: usize,
}

/// Information about the loaded ONNX voice cloner model
#[derive(Debug, Clone)]
pub struct VoiceClonerModelInfo {
    /// Model file path
    pub model_path: PathBuf,

    /// Input tensor names
    pub input_names: Vec<String>,

    /// Output tensor names
    pub output_names: Vec<String>,

    /// Speaker embedding dimensionality
    pub embedding_dim: usize,

    /// Output mel frequency bins
    pub mel_bins: usize,

    /// Maximum input sequence length
    pub max_sequence_length: usize,
}

/// ONNX-based voice cloner
///
/// Synthesizes mel spectrograms conditioned on text token ids and a speaker embedding.
/// Inputs:
///   - `text_ids`: `[1, seq_len]` (token ids cast to f32)
///   - `speaker_embedding`: `[1, embedding_dim]`
///
/// Output:
///   - `mel`: `[1, mel_bins, T]`
pub struct OnnxVoiceCloner {
    /// OxiONNX inference session
    session: Arc<RwLock<Session>>,

    /// Cloner configuration
    config: OnnxVoiceClonerConfig,

    /// Model information
    model_info: VoiceClonerModelInfo,
}

impl OnnxVoiceCloner {
    /// Create a new ONNX voice cloner from the given configuration.
    pub fn new(config: OnnxVoiceClonerConfig) -> Result<Self> {
        info!(
            "Initializing OxiONNX voice cloner from {:?}",
            config.model_path
        );

        let mut builder = Session::builder()
            .with_optimization_level(config.opt_level)
            .with_memory_pool(config.enable_memory_pool);
        if config.enable_profiling {
            builder = builder.with_profiling();
        }
        let session = builder
            .load(&config.model_path)
            .map_err(|e| Error::Model(format!("Failed to load ONNX voice cloner model: {}", e)))?;

        let input_names: Vec<String> = session.input_names().to_vec();
        let output_names: Vec<String> = session.output_names().to_vec();

        debug!("Voice cloner inputs: {:?}", input_names);
        debug!("Voice cloner outputs: {:?}", output_names);

        let model_info = VoiceClonerModelInfo {
            model_path: config.model_path.clone(),
            input_names,
            output_names,
            embedding_dim: config.embedding_dim,
            mel_bins: config.mel_bins,
            max_sequence_length: config.max_sequence_length,
        };

        info!("OxiONNX voice cloner loaded successfully");

        Ok(Self {
            session: Arc::new(RwLock::new(session)),
            config,
            model_info,
        })
    }

    /// Load a voice cloner from a model file path with default configuration.
    pub fn from_path(model_path: impl AsRef<Path>) -> Result<Self> {
        let config = OnnxVoiceClonerConfig {
            model_path: model_path.as_ref().to_path_buf(),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Synthesize a mel spectrogram from text token ids and a speaker embedding.
    ///
    /// # Arguments
    /// * `text_ids` - Token ids as i64 values with shape `[seq_len]`.
    ///   These are cast to f32 internally (oxionnx uses f32 tensors).
    /// * `speaker_embedding` - Speaker embedding vector of length `embedding_dim`.
    ///
    /// # Returns
    /// A [`VoiceCloningResult`] containing the generated mel spectrogram.
    pub fn clone_voice(
        &self,
        text_ids: &[i64],
        speaker_embedding: &[f32],
    ) -> Result<VoiceCloningResult> {
        if text_ids.is_empty() {
            return Err(Error::Processing(
                "Text token ids must not be empty".to_string(),
            ));
        }
        if text_ids.len() > self.config.max_sequence_length {
            return Err(Error::Processing(format!(
                "Text sequence length {} exceeds maximum {}",
                text_ids.len(),
                self.config.max_sequence_length
            )));
        }
        if speaker_embedding.len() != self.config.embedding_dim {
            return Err(Error::Embedding(format!(
                "Speaker embedding dimension mismatch: expected {}, got {}",
                self.config.embedding_dim,
                speaker_embedding.len()
            )));
        }

        let seq_len = text_ids.len();
        // Cast i64 token ids to f32 (oxionnx uses f32 tensors)
        let text_f32: Vec<f32> = text_ids.iter().map(|&id| id as f32).collect();
        let text_tensor = Tensor::new(text_f32, vec![1, seq_len]);
        let speaker_tensor = Tensor::new(
            speaker_embedding.to_vec(),
            vec![1, self.config.embedding_dim],
        );

        let session = self.session.read().map_err(|e| {
            Error::Processing(format!("Failed to acquire session read lock: {}", e))
        })?;

        // Build inputs from model input names
        let names = &self.model_info.input_names;
        let text_name = names
            .first()
            .cloned()
            .unwrap_or_else(|| "text_ids".to_string());
        let speaker_name = names
            .get(1)
            .cloned()
            .unwrap_or_else(|| "speaker_embedding".to_string());

        let mut owned_inputs: HashMap<String, Tensor> = HashMap::new();
        owned_inputs.insert(text_name, text_tensor);
        owned_inputs.insert(speaker_name, speaker_tensor);
        let ref_inputs: HashMap<&str, Tensor> = owned_inputs
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();

        let outputs = session
            .run(&ref_inputs)
            .map_err(|e| Error::Processing(format!("Voice cloning inference failed: {}", e)))?;

        let output_name = self
            .model_info
            .output_names
            .first()
            .ok_or_else(|| Error::Processing("Model has no output names".to_string()))?;

        let output_tensor = outputs.get(output_name).ok_or_else(|| {
            Error::Processing(format!("Output '{}' not found in results", output_name))
        })?;

        let mel_data = output_tensor.data.clone();
        let mel_bins = self.config.mel_bins;
        let time_steps = mel_data.len().checked_div(mel_bins).unwrap_or(0);

        Ok(VoiceCloningResult {
            mel: mel_data,
            mel_bins,
            time_steps,
        })
    }

    /// Return information about the loaded model.
    pub fn model_info(&self) -> &VoiceClonerModelInfo {
        &self.model_info
    }

    /// Return the cloner configuration.
    pub fn config(&self) -> &OnnxVoiceClonerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_speaker_encoder_config() {
        let config = OnnxSpeakerEncoderConfig::default();
        assert_eq!(config.mel_bins, 80);
        assert_eq!(config.embedding_dim, ONNX_SPEAKER_EMBEDDING_DIM);
        assert!(config.enable_memory_pool);
        assert!(!config.enable_profiling);
    }

    #[test]
    fn test_default_voice_cloner_config() {
        let config = OnnxVoiceClonerConfig::default();
        assert_eq!(config.mel_bins, 80);
        assert_eq!(config.embedding_dim, ONNX_SPEAKER_EMBEDDING_DIM);
        assert_eq!(config.max_sequence_length, 512);
        assert!(config.enable_memory_pool);
    }

    #[test]
    fn test_speaker_embedding_dim() {
        assert_eq!(ONNX_SPEAKER_EMBEDDING_DIM, 256);
    }

    #[test]
    fn test_voice_cloning_result_fields() {
        let result = VoiceCloningResult {
            mel: vec![0.0; 80 * 100],
            mel_bins: 80,
            time_steps: 100,
        };
        assert_eq!(result.mel.len(), result.mel_bins * result.time_steps);
    }

    #[test]
    fn test_speaker_encoder_model_info() {
        let info = SpeakerEncoderModelInfo {
            model_path: PathBuf::from("encoder.onnx"),
            input_names: vec!["mel_input".to_string()],
            output_names: vec!["embedding".to_string()],
            mel_bins: 80,
            embedding_dim: 256,
        };
        assert_eq!(info.embedding_dim, 256);
        assert_eq!(info.input_names.len(), 1);
    }

    #[test]
    fn test_voice_cloner_model_info() {
        let info = VoiceClonerModelInfo {
            model_path: PathBuf::from("cloner.onnx"),
            input_names: vec!["text_ids".to_string(), "speaker_embedding".to_string()],
            output_names: vec!["mel_output".to_string()],
            embedding_dim: 256,
            mel_bins: 80,
            max_sequence_length: 512,
        };
        assert_eq!(info.input_names.len(), 2);
        assert_eq!(info.max_sequence_length, 512);
    }
}
