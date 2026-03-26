//! OxiONNX backend for voice conversion.
//!
//! This module provides an ONNX-based voice conversion pipeline consisting of
//! three sub-models: content encoder, speaker encoder, and decoder.

use crate::{Error, Result};
use oxionnx::{OptLevel, Session, Tensor};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};
use tracing::{debug, info, warn};

/// Default speaker embedding dimensionality
pub const VC_SPEAKER_EMBEDDING_DIM: usize = 256;

/// Default content feature dimensionality
pub const VC_CONTENT_FEATURE_DIM: usize = 512;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the ONNX voice converter
#[derive(Debug, Clone)]
pub struct OnnxVoiceConverterConfig {
    /// Path to the content encoder ONNX model
    pub content_encoder_path: PathBuf,

    /// Path to the speaker encoder ONNX model
    pub speaker_encoder_path: PathBuf,

    /// Path to the decoder ONNX model
    pub decoder_path: PathBuf,

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

    /// Content feature dimensionality (default: 512)
    pub content_dim: usize,
}

impl Default for OnnxVoiceConverterConfig {
    fn default() -> Self {
        Self {
            content_encoder_path: PathBuf::new(),
            speaker_encoder_path: PathBuf::new(),
            decoder_path: PathBuf::new(),
            opt_level: OptLevel::All,
            enable_profiling: false,
            enable_memory_pool: true,
            mel_bins: 80,
            embedding_dim: VC_SPEAKER_EMBEDDING_DIM,
            content_dim: VC_CONTENT_FEATURE_DIM,
        }
    }
}

// ---------------------------------------------------------------------------
// Result / info types
// ---------------------------------------------------------------------------

/// Result of voice conversion
#[derive(Debug, Clone)]
pub struct VoiceConversionResult {
    /// Converted mel spectrogram in row-major `[mel_bins, T]` layout
    pub mel: Vec<f32>,

    /// Number of mel frequency bins
    pub mel_bins: usize,

    /// Number of generated time frames
    pub time_steps: usize,

    /// Content features extracted from the source (for diagnostics)
    pub content_features_dim: usize,
}

/// Information about the loaded ONNX voice conversion pipeline
#[derive(Debug, Clone)]
pub struct VoiceConverterModelInfo {
    /// Content encoder model path
    pub content_encoder_path: PathBuf,

    /// Speaker encoder model path
    pub speaker_encoder_path: PathBuf,

    /// Decoder model path
    pub decoder_path: PathBuf,

    /// Content encoder input names
    pub content_encoder_inputs: Vec<String>,

    /// Content encoder output names
    pub content_encoder_outputs: Vec<String>,

    /// Speaker encoder input names
    pub speaker_encoder_inputs: Vec<String>,

    /// Speaker encoder output names
    pub speaker_encoder_outputs: Vec<String>,

    /// Decoder input names
    pub decoder_inputs: Vec<String>,

    /// Decoder output names
    pub decoder_outputs: Vec<String>,

    /// Number of mel frequency bins
    pub mel_bins: usize,

    /// Speaker embedding dimensionality
    pub embedding_dim: usize,

    /// Content feature dimensionality
    pub content_dim: usize,
}

// ---------------------------------------------------------------------------
// OnnxVoiceConverter
// ---------------------------------------------------------------------------

/// ONNX-based voice converter
///
/// Converts a source speaker's mel spectrogram to sound like a target speaker.
/// The pipeline consists of three stages:
///
/// 1. **Content encoder** `source_mel [1, 80, T] -> content_features [1, T, D]`
/// 2. **Speaker encoder** `target_mel [1, 80, T] -> speaker_embedding [1, 256]`
/// 3. **Decoder** `content_features [1, T, D] + speaker_embedding [1, 256] -> converted_mel [1, 80, T]`
pub struct OnnxVoiceConverter {
    /// Content encoder session
    content_encoder: Arc<RwLock<Session>>,

    /// Speaker encoder session
    speaker_encoder: Arc<RwLock<Session>>,

    /// Decoder session
    decoder: Arc<RwLock<Session>>,

    /// Converter configuration
    config: OnnxVoiceConverterConfig,

    /// Model information
    model_info: VoiceConverterModelInfo,
}

impl OnnxVoiceConverter {
    /// Create a new ONNX voice converter from the given configuration.
    pub fn new(config: OnnxVoiceConverterConfig) -> Result<Self> {
        info!("Initializing OxiONNX voice conversion pipeline");

        // Load content encoder
        let content_encoder = Self::load_session(
            &config.content_encoder_path,
            config.opt_level,
            config.enable_memory_pool,
            config.enable_profiling,
            "content encoder",
        )?;

        // Load speaker encoder
        let speaker_encoder = Self::load_session(
            &config.speaker_encoder_path,
            config.opt_level,
            config.enable_memory_pool,
            config.enable_profiling,
            "speaker encoder",
        )?;

        // Load decoder
        let decoder = Self::load_session(
            &config.decoder_path,
            config.opt_level,
            config.enable_memory_pool,
            config.enable_profiling,
            "decoder",
        )?;

        let model_info = VoiceConverterModelInfo {
            content_encoder_path: config.content_encoder_path.clone(),
            speaker_encoder_path: config.speaker_encoder_path.clone(),
            decoder_path: config.decoder_path.clone(),
            content_encoder_inputs: content_encoder.input_names().to_vec(),
            content_encoder_outputs: content_encoder.output_names().to_vec(),
            speaker_encoder_inputs: speaker_encoder.input_names().to_vec(),
            speaker_encoder_outputs: speaker_encoder.output_names().to_vec(),
            decoder_inputs: decoder.input_names().to_vec(),
            decoder_outputs: decoder.output_names().to_vec(),
            mel_bins: config.mel_bins,
            embedding_dim: config.embedding_dim,
            content_dim: config.content_dim,
        };

        info!("OxiONNX voice conversion pipeline loaded successfully");

        Ok(Self {
            content_encoder: Arc::new(RwLock::new(content_encoder)),
            speaker_encoder: Arc::new(RwLock::new(speaker_encoder)),
            decoder: Arc::new(RwLock::new(decoder)),
            config,
            model_info,
        })
    }

    /// Convert a source mel spectrogram to sound like the target speaker.
    ///
    /// # Arguments
    /// * `source_mel` - Source mel spectrogram `[mel_bins, T_src]` in row-major order.
    /// * `source_time_steps` - Number of time frames in the source.
    /// * `target_mel` - Target speaker's reference mel spectrogram `[mel_bins, T_tgt]`.
    /// * `target_time_steps` - Number of time frames in the target reference.
    ///
    /// # Returns
    /// A [`VoiceConversionResult`] with the converted mel spectrogram.
    pub fn convert(
        &self,
        source_mel: &[f32],
        source_time_steps: usize,
        target_mel: &[f32],
        target_time_steps: usize,
    ) -> Result<VoiceConversionResult> {
        let mel_bins = self.config.mel_bins;

        // Validate source
        let src_expected = mel_bins * source_time_steps;
        if source_mel.len() != src_expected {
            return Err(Error::model(format!(
                "Source mel size mismatch: expected {} ({}x{}), got {}",
                src_expected,
                mel_bins,
                source_time_steps,
                source_mel.len()
            )));
        }

        // Validate target
        let tgt_expected = mel_bins * target_time_steps;
        if target_mel.len() != tgt_expected {
            return Err(Error::model(format!(
                "Target mel size mismatch: expected {} ({}x{}), got {}",
                tgt_expected,
                mel_bins,
                target_time_steps,
                target_mel.len()
            )));
        }

        // Stage 1: Extract content features from source
        let content_features = self.encode_content(source_mel, source_time_steps)?;

        // Stage 2: Extract speaker embedding from target
        let speaker_embedding = self.encode_speaker(target_mel, target_time_steps)?;

        // Stage 3: Decode to converted mel
        let converted = self.decode(&content_features, source_time_steps, &speaker_embedding)?;

        let out_time_steps = converted.len().checked_div(mel_bins).unwrap_or(0);

        Ok(VoiceConversionResult {
            mel: converted,
            mel_bins,
            time_steps: out_time_steps,
            content_features_dim: self.config.content_dim,
        })
    }

    /// Return information about the loaded models.
    pub fn model_info(&self) -> &VoiceConverterModelInfo {
        &self.model_info
    }

    /// Return the converter configuration.
    pub fn config(&self) -> &OnnxVoiceConverterConfig {
        &self.config
    }

    // ---- private helpers ----

    /// Load a single ONNX session with the given parameters.
    fn load_session(
        path: &Path,
        opt_level: OptLevel,
        enable_memory_pool: bool,
        enable_profiling: bool,
        label: &str,
    ) -> Result<Session> {
        info!("Loading {} from {:?}", label, path);
        let mut builder = Session::builder()
            .with_optimization_level(opt_level)
            .with_memory_pool(enable_memory_pool);
        if enable_profiling {
            builder = builder.with_profiling();
        }
        builder
            .load(path)
            .map_err(|e| Error::model(format!("Failed to load ONNX {} model: {}", label, e)))
    }

    /// Run the content encoder: `source_mel [1, mel_bins, T] -> content_features [flat]`
    fn encode_content(&self, source_mel: &[f32], time_steps: usize) -> Result<Vec<f32>> {
        let mel_bins = self.config.mel_bins;
        let input_tensor = Tensor::new(source_mel.to_vec(), vec![1, mel_bins, time_steps]);

        let input_name = self
            .model_info
            .content_encoder_inputs
            .first()
            .cloned()
            .unwrap_or_else(|| "source_mel".to_string());

        let mut owned_inputs: HashMap<String, Tensor> = HashMap::new();
        owned_inputs.insert(input_name, input_tensor);
        let ref_inputs: HashMap<&str, Tensor> = owned_inputs
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();

        let session = self
            .content_encoder
            .read()
            .map_err(|e| Error::model(format!("Failed to acquire content encoder lock: {}", e)))?;

        let outputs = session
            .run(&ref_inputs)
            .map_err(|e| Error::model(format!("Content encoder inference failed: {}", e)))?;

        let output_name = self
            .model_info
            .content_encoder_outputs
            .first()
            .cloned()
            .unwrap_or_else(|| "content_features".to_string());

        let out = outputs.get(&output_name).ok_or_else(|| {
            Error::model("Content encoder produced no matching output".to_string())
        })?;

        Ok(out.data.clone())
    }

    /// Run the speaker encoder: `target_mel [1, mel_bins, T] -> speaker_embedding [flat]`
    fn encode_speaker(&self, target_mel: &[f32], time_steps: usize) -> Result<Vec<f32>> {
        let mel_bins = self.config.mel_bins;
        let input_tensor = Tensor::new(target_mel.to_vec(), vec![1, mel_bins, time_steps]);

        let input_name = self
            .model_info
            .speaker_encoder_inputs
            .first()
            .cloned()
            .unwrap_or_else(|| "target_mel".to_string());

        let mut owned_inputs: HashMap<String, Tensor> = HashMap::new();
        owned_inputs.insert(input_name, input_tensor);
        let ref_inputs: HashMap<&str, Tensor> = owned_inputs
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();

        let session = self
            .speaker_encoder
            .read()
            .map_err(|e| Error::model(format!("Failed to acquire speaker encoder lock: {}", e)))?;

        let outputs = session
            .run(&ref_inputs)
            .map_err(|e| Error::model(format!("Speaker encoder inference failed: {}", e)))?;

        let output_name = self
            .model_info
            .speaker_encoder_outputs
            .first()
            .cloned()
            .unwrap_or_else(|| "speaker_embedding".to_string());

        let out = outputs.get(&output_name).ok_or_else(|| {
            Error::model("Speaker encoder produced no matching output".to_string())
        })?;

        let embedding = out.data.clone();

        // L2-normalize the embedding
        let norm: f32 = embedding.iter().map(|&v| v * v).sum::<f32>().sqrt();
        if norm > 1e-8 {
            Ok(embedding.iter().map(|&v| v / norm).collect())
        } else {
            warn!("Speaker embedding has near-zero norm; returning unnormalized");
            Ok(embedding)
        }
    }

    /// Run the decoder: `content_features + speaker_embedding -> converted_mel`
    fn decode(
        &self,
        content_features: &[f32],
        content_time_steps: usize,
        speaker_embedding: &[f32],
    ) -> Result<Vec<f32>> {
        let content_dim = self.config.content_dim;
        let embedding_dim = self.config.embedding_dim;

        let content_tensor = Tensor::new(
            content_features.to_vec(),
            vec![1, content_time_steps, content_dim],
        );
        let speaker_tensor = Tensor::new(speaker_embedding.to_vec(), vec![1, embedding_dim]);

        let decoder_inputs = &self.model_info.decoder_inputs;
        let content_name = decoder_inputs
            .first()
            .cloned()
            .unwrap_or_else(|| "content_features".to_string());
        let speaker_name = decoder_inputs
            .get(1)
            .cloned()
            .unwrap_or_else(|| "speaker_embedding".to_string());

        let mut owned_inputs: HashMap<String, Tensor> = HashMap::new();
        owned_inputs.insert(content_name, content_tensor);
        owned_inputs.insert(speaker_name, speaker_tensor);
        let ref_inputs: HashMap<&str, Tensor> = owned_inputs
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();

        let session = self
            .decoder
            .read()
            .map_err(|e| Error::model(format!("Failed to acquire decoder lock: {}", e)))?;

        let outputs = session
            .run(&ref_inputs)
            .map_err(|e| Error::model(format!("Decoder inference failed: {}", e)))?;

        let output_name = self
            .model_info
            .decoder_outputs
            .first()
            .cloned()
            .unwrap_or_else(|| "converted_mel".to_string());

        let out = outputs
            .get(&output_name)
            .ok_or_else(|| Error::model("Decoder produced no matching output".to_string()))?;

        Ok(out.data.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OnnxVoiceConverterConfig::default();
        assert_eq!(config.mel_bins, 80);
        assert_eq!(config.embedding_dim, VC_SPEAKER_EMBEDDING_DIM);
        assert_eq!(config.content_dim, VC_CONTENT_FEATURE_DIM);
        assert!(config.enable_memory_pool);
        assert!(!config.enable_profiling);
    }

    #[test]
    fn test_constants() {
        assert_eq!(VC_SPEAKER_EMBEDDING_DIM, 256);
        assert_eq!(VC_CONTENT_FEATURE_DIM, 512);
    }

    #[test]
    fn test_voice_conversion_result_fields() {
        let result = VoiceConversionResult {
            mel: vec![0.0; 80 * 50],
            mel_bins: 80,
            time_steps: 50,
            content_features_dim: 512,
        };
        assert_eq!(result.mel.len(), result.mel_bins * result.time_steps);
        assert_eq!(result.content_features_dim, 512);
    }

    #[test]
    fn test_model_info_fields() {
        let info = VoiceConverterModelInfo {
            content_encoder_path: PathBuf::from("content.onnx"),
            speaker_encoder_path: PathBuf::from("speaker.onnx"),
            decoder_path: PathBuf::from("decoder.onnx"),
            content_encoder_inputs: vec!["source_mel".to_string()],
            content_encoder_outputs: vec!["content_features".to_string()],
            speaker_encoder_inputs: vec!["target_mel".to_string()],
            speaker_encoder_outputs: vec!["speaker_embedding".to_string()],
            decoder_inputs: vec![
                "content_features".to_string(),
                "speaker_embedding".to_string(),
            ],
            decoder_outputs: vec!["converted_mel".to_string()],
            mel_bins: 80,
            embedding_dim: 256,
            content_dim: 512,
        };
        assert_eq!(info.decoder_inputs.len(), 2);
        assert_eq!(info.mel_bins, 80);
    }
}
