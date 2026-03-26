//! OxiONNX backend for singing synthesis models.
//!
//! This module provides ONNX-based implementations for singing synthesis,
//! including DiffSinger and generic singing model backends. Models are loaded
//! and executed via the oxionnx pure-Rust inference engine.
//!
//! # Architecture
//!
//! The singing ONNX backend supports a two-stage pipeline:
//!
//! 1. **Acoustic model** (DiffSinger): Converts phoneme IDs, note IDs,
//!    note durations, and F0 contours into mel spectrograms.
//! 2. **Vocoder** (optional): Converts mel spectrograms into audio waveforms.
//!
//! When no vocoder is provided, the acoustic model output (mel spectrogram)
//! is returned directly for downstream processing.
//!
//! # Example
//!
//! ```rust,ignore
//! use voirs_singing::backends::onnx::{OnnxDiffSinger, OnnxSingingConfig};
//! use std::path::PathBuf;
//!
//! # async fn example() -> voirs_singing::Result<()> {
//! let config = OnnxSingingConfig {
//!     acoustic_model_path: PathBuf::from("diffsinger_acoustic.onnx"),
//!     vocoder_model_path: Some(PathBuf::from("nsf_hifigan.onnx")),
//!     ..OnnxSingingConfig::default()
//! };
//!
//! let singer = OnnxDiffSinger::new(config).await?;
//!
//! let phoneme_ids = vec![0, 5, 12, 7, 3];
//! let note_ids = vec![60, 62, 64, 65, 67];
//! let durations = vec![0.25, 0.5, 0.25, 0.5, 1.0];
//! let f0 = vec![261.63, 293.66, 329.63, 349.23, 392.0];
//!
//! let audio = singer.synthesize(&phoneme_ids, &note_ids, &durations, &f0)?;
//! println!("Generated {} audio samples", audio.len());
//! # Ok(())
//! # }
//! ```

use crate::{Error, Result};
use oxionnx::{ModelInfo, NodeProfile, OptLevel, Session, Tensor};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for an ONNX-based singing synthesis model.
#[derive(Debug, Clone)]
pub struct OnnxSingingConfig {
    /// Path to the acoustic (DiffSinger) ONNX model.
    pub acoustic_model_path: PathBuf,

    /// Optional path to the vocoder ONNX model (e.g. NSF-HiFiGAN).
    /// When `None`, only mel spectrograms are produced.
    pub vocoder_model_path: Option<PathBuf>,

    /// Output audio sample rate in Hz.
    pub sample_rate: u32,

    /// Number of mel frequency bins expected by the acoustic model.
    pub n_mels: usize,

    /// Hop length of the STFT used during training.
    pub hop_length: usize,

    /// ONNX graph optimization level.
    pub opt_level: OptLevel,

    /// Enable per-node profiling during inference.
    pub enable_profiling: bool,

    /// Enable the memory buffer pool for activation reuse.
    pub enable_memory_pool: bool,

    /// Number of CPU threads (0 = auto-detect).
    pub num_threads: usize,

    /// Audio clipping threshold for the vocoder output.
    pub clip_threshold: f32,

    /// Maximum sequence length supported by the model.
    pub max_sequence_length: usize,
}

impl Default for OnnxSingingConfig {
    fn default() -> Self {
        Self {
            acoustic_model_path: PathBuf::new(),
            vocoder_model_path: None,
            sample_rate: 24000,
            n_mels: 80,
            hop_length: 256,
            opt_level: OptLevel::All,
            enable_profiling: false,
            enable_memory_pool: false,
            num_threads: 0,
            clip_threshold: 0.99,
            max_sequence_length: 2048,
        }
    }
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

/// Metadata extracted from a loaded ONNX singing model.
#[derive(Debug, Clone)]
pub struct OnnxSingingMetadata {
    /// Friendly model name (derived from the file stem).
    pub name: String,

    /// Model version string.
    pub version: String,

    /// Model architecture descriptor.
    pub architecture: String,

    /// Input tensor names declared in the ONNX graph.
    pub input_names: Vec<String>,

    /// Output tensor names declared in the ONNX graph.
    pub output_names: Vec<String>,

    /// Number of mel bins the model produces.
    pub n_mels: usize,

    /// Expected sample rate of the output audio.
    pub sample_rate: u32,
}

// ---------------------------------------------------------------------------
// OnnxDiffSinger
// ---------------------------------------------------------------------------

/// ONNX-based DiffSinger model for singing voice synthesis.
///
/// DiffSinger is a diffusion-based singing synthesis model that converts
/// phoneme sequences, note information, durations, and F0 contours into
/// mel spectrograms which can then be vocoded into audio.
pub struct OnnxDiffSinger {
    /// Acoustic model session.
    acoustic_session: Arc<RwLock<Session>>,

    /// Optional vocoder session.
    vocoder_session: Option<Arc<RwLock<Session>>>,

    /// User-supplied configuration.
    config: OnnxSingingConfig,

    /// Metadata extracted from the acoustic model.
    acoustic_metadata: OnnxSingingMetadata,

    /// Metadata extracted from the vocoder model (if loaded).
    vocoder_metadata: Option<OnnxSingingMetadata>,
}

impl OnnxDiffSinger {
    /// Load an ONNX DiffSinger model from disk.
    ///
    /// This loads the acoustic model and, optionally, the vocoder model.
    pub async fn new(config: OnnxSingingConfig) -> Result<Self> {
        info!(
            "Initializing OxiONNX DiffSinger from {:?}",
            config.acoustic_model_path
        );

        // --- Acoustic model ---
        let acoustic_session =
            Self::load_session(&config.acoustic_model_path, &config, "DiffSinger acoustic")?;
        let acoustic_metadata =
            Self::extract_metadata(&acoustic_session, &config.acoustic_model_path, &config)?;

        info!(
            "Acoustic model loaded: {} (inputs: {:?}, outputs: {:?})",
            acoustic_metadata.name, acoustic_metadata.input_names, acoustic_metadata.output_names
        );

        // --- Optional vocoder ---
        let (vocoder_session, vocoder_metadata) =
            if let Some(ref voc_path) = config.vocoder_model_path {
                info!("Loading vocoder from {:?}", voc_path);
                let session = Self::load_session(voc_path, &config, "vocoder")?;
                let meta = Self::extract_metadata(&session, voc_path, &config)?;
                info!(
                    "Vocoder loaded: {} (inputs: {:?}, outputs: {:?})",
                    meta.name, meta.input_names, meta.output_names
                );
                (Some(Arc::new(RwLock::new(session))), Some(meta))
            } else {
                (None, None)
            };

        Ok(Self {
            acoustic_session: Arc::new(RwLock::new(acoustic_session)),
            vocoder_session,
            config,
            acoustic_metadata,
            vocoder_metadata,
        })
    }

    // ------------------------------------------------------------------
    // Session loading helpers
    // ------------------------------------------------------------------

    fn load_session(path: &Path, config: &OnnxSingingConfig, label: &str) -> Result<Session> {
        let mut builder = Session::builder()
            .with_optimization_level(config.opt_level)
            .with_memory_pool(config.enable_memory_pool);
        if config.enable_profiling {
            builder = builder.with_profiling();
        }
        builder.load(path).map_err(|e| {
            Error::Model(format!(
                "Failed to load ONNX {label} model from {path:?}: {e}"
            ))
        })
    }

    fn extract_metadata(
        session: &Session,
        model_path: &Path,
        config: &OnnxSingingConfig,
    ) -> Result<OnnxSingingMetadata> {
        let input_names: Vec<String> = session.input_names().to_vec();
        let output_names: Vec<String> = session.output_names().to_vec();

        debug!("ONNX model inputs: {:?}", input_names);
        debug!("ONNX model outputs: {:?}", output_names);

        let name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(OnnxSingingMetadata {
            name,
            version: "1.0.0".to_string(),
            architecture: "DiffSinger".to_string(),
            input_names,
            output_names,
            n_mels: config.n_mels,
            sample_rate: config.sample_rate,
        })
    }

    // ------------------------------------------------------------------
    // Public inference API
    // ------------------------------------------------------------------

    /// Synthesize a mel spectrogram from musical input.
    ///
    /// # Arguments
    ///
    /// * `phoneme_ids` - Phoneme indices for the lyric sequence.
    /// * `note_ids`    - MIDI note numbers for each phoneme.
    /// * `durations`   - Duration in seconds for each phoneme/note.
    /// * `f0`          - Fundamental frequency contour in Hz for each frame.
    ///
    /// # Returns
    ///
    /// A flat `Vec<f32>` representing the mel spectrogram in row-major order
    /// with shape `[n_mels, T]` where `T` is the number of time frames.
    pub fn synthesize_mel(
        &self,
        phoneme_ids: &[i32],
        note_ids: &[i32],
        durations: &[f32],
        f0: &[f32],
    ) -> Result<Vec<f32>> {
        let seq_len = phoneme_ids.len();
        if note_ids.len() != seq_len || durations.len() != seq_len {
            return Err(Error::Validation(format!(
                "Input length mismatch: phoneme_ids={}, note_ids={}, durations={}",
                seq_len,
                note_ids.len(),
                durations.len()
            )));
        }
        if seq_len > self.config.max_sequence_length {
            return Err(Error::Validation(format!(
                "Sequence length {} exceeds maximum {}",
                seq_len, self.config.max_sequence_length
            )));
        }

        // Build input tensors
        let mut inputs: HashMap<&str, Tensor> = HashMap::new();

        // phoneme_ids [1, seq_len] (stored as f32 for oxionnx)
        let phoneme_data: Vec<f32> = phoneme_ids.iter().map(|&v| v as f32).collect();
        inputs.insert("phoneme_ids", Tensor::new(phoneme_data, vec![1, seq_len]));

        // note_ids [1, seq_len]
        let note_data: Vec<f32> = note_ids.iter().map(|&v| v as f32).collect();
        inputs.insert("note_ids", Tensor::new(note_data, vec![1, seq_len]));

        // note_durations [1, seq_len]
        inputs.insert(
            "note_durations",
            Tensor::new(durations.to_vec(), vec![1, seq_len]),
        );

        // f0 [1, f0_len]
        let f0_len = f0.len();
        inputs.insert("f0", Tensor::new(f0.to_vec(), vec![1, f0_len]));

        // Run acoustic inference
        let session = self
            .acoustic_session
            .read()
            .map_err(|_| Error::Processing("Acoustic session RwLock poisoned".to_string()))?;

        let outputs = session
            .run(&inputs)
            .map_err(|e| Error::Processing(format!("DiffSinger acoustic inference failed: {e}")))?;

        // Extract mel from first output
        let mel_tensor = Self::first_output(&outputs, "acoustic")?;

        debug!(
            "Acoustic output shape: {:?}, {} elements",
            mel_tensor.shape,
            mel_tensor.data.len()
        );

        Ok(mel_tensor.data.clone())
    }

    /// Synthesize audio from a mel spectrogram using the loaded vocoder.
    ///
    /// Returns an error if no vocoder was loaded.
    pub fn synthesize_audio(&self, mel_data: &[f32], mel_frames: usize) -> Result<Vec<f32>> {
        let vocoder_session = self
            .vocoder_session
            .as_ref()
            .ok_or_else(|| Error::Config("No vocoder model loaded".to_string()))?;

        let n_mels = self.config.n_mels;
        let expected_len = n_mels * mel_frames;
        if mel_data.len() != expected_len {
            return Err(Error::Validation(format!(
                "Mel data length {} does not match expected {} (n_mels={}, frames={})",
                mel_data.len(),
                expected_len,
                n_mels,
                mel_frames
            )));
        }

        // Build vocoder input: mel [1, n_mels, T]
        let mut inputs: HashMap<&str, Tensor> = HashMap::new();
        inputs.insert(
            "mel",
            Tensor::new(mel_data.to_vec(), vec![1, n_mels, mel_frames]),
        );

        let session = vocoder_session
            .read()
            .map_err(|_| Error::Processing("Vocoder session RwLock poisoned".to_string()))?;

        let outputs = session
            .run(&inputs)
            .map_err(|e| Error::Processing(format!("Vocoder inference failed: {e}")))?;

        let audio_tensor = Self::first_output(&outputs, "vocoder")?;

        debug!(
            "Vocoder output shape: {:?}, {} samples",
            audio_tensor.shape,
            audio_tensor.data.len()
        );

        // Clip audio to configured threshold
        let clip = self.config.clip_threshold;
        let audio: Vec<f32> = audio_tensor
            .data
            .iter()
            .map(|&s| s.clamp(-clip, clip))
            .collect();

        Ok(audio)
    }

    /// Full pipeline: phoneme/note/duration/F0 -> audio.
    ///
    /// If no vocoder is loaded, returns the raw mel spectrogram data.
    pub fn synthesize(
        &self,
        phoneme_ids: &[i32],
        note_ids: &[i32],
        durations: &[f32],
        f0: &[f32],
    ) -> Result<Vec<f32>> {
        let mel = self.synthesize_mel(phoneme_ids, note_ids, durations, f0)?;

        if self.vocoder_session.is_some() {
            // Determine mel frames from the acoustic output shape.
            // The acoustic model outputs [1, n_mels, T] or [n_mels, T].
            let n_mels = self.config.n_mels;
            if n_mels == 0 {
                return Err(Error::Config("n_mels is 0".to_string()));
            }
            let mel_frames = mel.len() / n_mels;
            if mel_frames == 0 {
                return Err(Error::Processing(
                    "Acoustic model produced empty mel spectrogram".to_string(),
                ));
            }
            self.synthesize_audio(&mel, mel_frames)
        } else {
            Ok(mel)
        }
    }

    // ------------------------------------------------------------------
    // Streaming synthesis
    // ------------------------------------------------------------------

    /// Synthesize audio in streaming chunks for lower latency.
    ///
    /// Splits the input sequence into overlapping chunks and processes them
    /// sequentially, concatenating the output.
    pub fn synthesize_streaming(
        &self,
        phoneme_ids: &[i32],
        note_ids: &[i32],
        durations: &[f32],
        f0: &[f32],
        chunk_size: usize,
        overlap: usize,
    ) -> Result<Vec<f32>> {
        let seq_len = phoneme_ids.len();
        if seq_len == 0 {
            return Ok(Vec::new());
        }
        if chunk_size == 0 {
            return Err(Error::Validation("chunk_size must be > 0".to_string()));
        }

        let step = chunk_size.saturating_sub(overlap).max(1);
        let mut all_audio = Vec::new();

        let mut offset = 0usize;
        while offset < seq_len {
            let end = (offset + chunk_size).min(seq_len);
            let chunk_phonemes = &phoneme_ids[offset..end];
            let chunk_notes = &note_ids[offset..end];
            let chunk_durations = &durations[offset..end];

            // F0 may have a different length; map proportionally
            let f0_start = (offset * f0.len()).checked_div(seq_len).unwrap_or(0);
            let f0_end = (end * f0.len())
                .checked_div(seq_len)
                .unwrap_or(0)
                .min(f0.len());
            let chunk_f0 = &f0[f0_start..f0_end];

            let chunk_audio =
                self.synthesize(chunk_phonemes, chunk_notes, chunk_durations, chunk_f0)?;

            // For the first chunk, take everything. For subsequent chunks,
            // skip the overlap region to avoid duplication.
            if offset == 0 {
                all_audio.extend_from_slice(&chunk_audio);
            } else {
                let skip_samples = if !chunk_audio.is_empty() {
                    let chunk_len = end - offset;
                    (overlap * chunk_audio.len())
                        .checked_div(chunk_len)
                        .unwrap_or(0)
                } else {
                    0
                };
                if skip_samples < chunk_audio.len() {
                    all_audio.extend_from_slice(&chunk_audio[skip_samples..]);
                }
            }

            offset += step;
        }

        Ok(all_audio)
    }

    // ------------------------------------------------------------------
    // Introspection
    // ------------------------------------------------------------------

    /// Return oxionnx `ModelInfo` for the acoustic model.
    pub fn acoustic_model_info(&self) -> Result<ModelInfo> {
        let session = self
            .acoustic_session
            .read()
            .map_err(|_| Error::Processing("Acoustic session RwLock poisoned".to_string()))?;
        Ok(session.model_info())
    }

    /// Return oxionnx `ModelInfo` for the vocoder model.
    pub fn vocoder_model_info(&self) -> Result<Option<ModelInfo>> {
        match &self.vocoder_session {
            Some(sess) => {
                let session = sess.read().map_err(|_| {
                    Error::Processing("Vocoder session RwLock poisoned".to_string())
                })?;
                Ok(Some(session.model_info()))
            }
            None => Ok(None),
        }
    }

    /// Return profiling results from the acoustic model session.
    pub fn acoustic_profiling_results(&self) -> Result<Option<Vec<NodeProfile>>> {
        let session = self
            .acoustic_session
            .read()
            .map_err(|_| Error::Processing("Acoustic session RwLock poisoned".to_string()))?;
        Ok(session.profiling_results())
    }

    /// Return profiling results from the vocoder model session.
    pub fn vocoder_profiling_results(&self) -> Result<Option<Vec<NodeProfile>>> {
        match &self.vocoder_session {
            Some(sess) => {
                let session = sess.read().map_err(|_| {
                    Error::Processing("Vocoder session RwLock poisoned".to_string())
                })?;
                Ok(session.profiling_results())
            }
            None => Ok(None),
        }
    }

    /// Estimated memory usage in bytes for both models.
    pub fn estimated_memory_bytes(&self) -> Result<usize> {
        let acoustic_bytes = {
            let session = self
                .acoustic_session
                .read()
                .map_err(|_| Error::Processing("Acoustic session RwLock poisoned".to_string()))?;
            session.estimated_memory_bytes().unwrap_or(0)
        };
        let vocoder_bytes = match &self.vocoder_session {
            Some(sess) => {
                let session = sess.read().map_err(|_| {
                    Error::Processing("Vocoder session RwLock poisoned".to_string())
                })?;
                session.estimated_memory_bytes().unwrap_or(0)
            }
            None => 0,
        };
        Ok(acoustic_bytes + vocoder_bytes)
    }

    /// Return the acoustic model metadata.
    pub fn acoustic_metadata(&self) -> &OnnxSingingMetadata {
        &self.acoustic_metadata
    }

    /// Return the vocoder model metadata, if loaded.
    pub fn vocoder_metadata(&self) -> Option<&OnnxSingingMetadata> {
        self.vocoder_metadata.as_ref()
    }

    /// Return the configuration used to create this model.
    pub fn config(&self) -> &OnnxSingingConfig {
        &self.config
    }

    /// Whether a vocoder is available for end-to-end synthesis.
    pub fn has_vocoder(&self) -> bool {
        self.vocoder_session.is_some()
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Extract the first output tensor from a run result.
    fn first_output<'a>(outputs: &'a HashMap<String, Tensor>, label: &str) -> Result<&'a Tensor> {
        if outputs.is_empty() {
            return Err(Error::Processing(format!(
                "No outputs received from ONNX {label} model"
            )));
        }
        outputs
            .values()
            .next()
            .ok_or_else(|| Error::Processing(format!("No output tensor found from {label} model")))
    }
}

// ---------------------------------------------------------------------------
// OnnxSingingModel — generic ONNX singing model
// ---------------------------------------------------------------------------

/// A generic ONNX-based singing model for architectures other than DiffSinger.
///
/// This wrapper loads a single ONNX model and exposes flexible input/output
/// access so that it can serve different singing model architectures (e.g.
/// VISinger, ACE, NNSVS).
pub struct OnnxSingingModel {
    /// ONNX session.
    session: Arc<RwLock<Session>>,

    /// Model metadata.
    metadata: OnnxSingingMetadata,

    /// Configuration.
    config: OnnxSingingModelConfig,
}

/// Configuration for a generic ONNX singing model.
#[derive(Debug, Clone)]
pub struct OnnxSingingModelConfig {
    /// Path to the ONNX model file.
    pub model_path: PathBuf,

    /// Sample rate of the model output.
    pub sample_rate: u32,

    /// Number of mel bins (if the model produces mel spectrograms).
    pub n_mels: usize,

    /// ONNX graph optimization level.
    pub opt_level: OptLevel,

    /// Enable per-node profiling.
    pub enable_profiling: bool,

    /// Enable memory buffer pool.
    pub enable_memory_pool: bool,

    /// Maximum expected sequence length.
    pub max_sequence_length: usize,
}

impl Default for OnnxSingingModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            sample_rate: 24000,
            n_mels: 80,
            opt_level: OptLevel::All,
            enable_profiling: false,
            enable_memory_pool: false,
            max_sequence_length: 2048,
        }
    }
}

impl OnnxSingingModel {
    /// Load a generic ONNX singing model.
    pub async fn new(config: OnnxSingingModelConfig) -> Result<Self> {
        info!(
            "Initializing generic OxiONNX singing model from {:?}",
            config.model_path
        );

        let mut builder = Session::builder()
            .with_optimization_level(config.opt_level)
            .with_memory_pool(config.enable_memory_pool);
        if config.enable_profiling {
            builder = builder.with_profiling();
        }
        let session = builder.load(&config.model_path).map_err(|e| {
            Error::Model(format!(
                "Failed to load ONNX singing model from {:?}: {e}",
                config.model_path
            ))
        })?;

        let input_names: Vec<String> = session.input_names().to_vec();
        let output_names: Vec<String> = session.output_names().to_vec();

        let name = config
            .model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let metadata = OnnxSingingMetadata {
            name: name.clone(),
            version: "1.0.0".to_string(),
            architecture: "Generic".to_string(),
            input_names,
            output_names,
            n_mels: config.n_mels,
            sample_rate: config.sample_rate,
        };

        info!("Generic singing model loaded: {}", name);

        Ok(Self {
            session: Arc::new(RwLock::new(session)),
            metadata,
            config,
        })
    }

    /// Run inference with arbitrary named inputs.
    ///
    /// The caller is responsible for constructing the correct input tensors
    /// matching the model's expected inputs.
    pub fn run(&self, inputs: &HashMap<&str, Tensor>) -> Result<HashMap<String, Tensor>> {
        let session = self
            .session
            .read()
            .map_err(|_| Error::Processing("Session RwLock poisoned".to_string()))?;

        session
            .run(inputs)
            .map_err(|e| Error::Processing(format!("ONNX singing model inference failed: {e}")))
    }

    /// Convenience method: run with a single phoneme + note sequence.
    ///
    /// Creates standard DiffSinger-like inputs and runs the model.
    pub fn run_simple(
        &self,
        phoneme_ids: &[f32],
        note_ids: &[f32],
        durations: &[f32],
    ) -> Result<HashMap<String, Tensor>> {
        let seq_len = phoneme_ids.len();
        if note_ids.len() != seq_len || durations.len() != seq_len {
            return Err(Error::Validation(format!(
                "Input length mismatch: phoneme_ids={}, note_ids={}, durations={}",
                seq_len,
                note_ids.len(),
                durations.len()
            )));
        }
        if seq_len > self.config.max_sequence_length {
            return Err(Error::Validation(format!(
                "Sequence length {} exceeds maximum {}",
                seq_len, self.config.max_sequence_length
            )));
        }

        let mut inputs: HashMap<&str, Tensor> = HashMap::new();
        inputs.insert(
            "phoneme_ids",
            Tensor::new(phoneme_ids.to_vec(), vec![1, seq_len]),
        );
        inputs.insert("note_ids", Tensor::new(note_ids.to_vec(), vec![1, seq_len]));
        inputs.insert(
            "durations",
            Tensor::new(durations.to_vec(), vec![1, seq_len]),
        );

        self.run(&inputs)
    }

    /// Return the model metadata.
    pub fn metadata(&self) -> &OnnxSingingMetadata {
        &self.metadata
    }

    /// Return the configuration.
    pub fn config(&self) -> &OnnxSingingModelConfig {
        &self.config
    }

    /// Return oxionnx `ModelInfo`.
    pub fn model_info(&self) -> Result<ModelInfo> {
        let session = self
            .session
            .read()
            .map_err(|_| Error::Processing("Session RwLock poisoned".to_string()))?;
        Ok(session.model_info())
    }

    /// Return profiling results if profiling was enabled.
    pub fn profiling_results(&self) -> Result<Option<Vec<NodeProfile>>> {
        let session = self
            .session
            .read()
            .map_err(|_| Error::Processing("Session RwLock poisoned".to_string()))?;
        Ok(session.profiling_results())
    }

    /// Estimated memory usage in bytes.
    pub fn estimated_memory_bytes(&self) -> Result<usize> {
        let session = self
            .session
            .read()
            .map_err(|_| Error::Processing("Session RwLock poisoned".to_string()))?;
        Ok(session.estimated_memory_bytes().unwrap_or(0))
    }

    /// Declared input tensor names.
    pub fn input_names(&self) -> &[String] {
        &self.metadata.input_names
    }

    /// Declared output tensor names.
    pub fn output_names(&self) -> &[String] {
        &self.metadata.output_names
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OnnxSingingConfig::default();
        assert_eq!(config.sample_rate, 24000);
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.hop_length, 256);
        assert!(!config.enable_profiling);
        assert!(!config.enable_memory_pool);
        assert_eq!(config.max_sequence_length, 2048);
    }

    #[test]
    fn test_default_model_config() {
        let config = OnnxSingingModelConfig::default();
        assert_eq!(config.sample_rate, 24000);
        assert_eq!(config.n_mels, 80);
        assert!(!config.enable_profiling);
        assert!(!config.enable_memory_pool);
        assert_eq!(config.max_sequence_length, 2048);
    }

    #[test]
    fn test_config_custom() {
        let config = OnnxSingingConfig {
            acoustic_model_path: PathBuf::from("/tmp/acoustic.onnx"),
            vocoder_model_path: Some(PathBuf::from("/tmp/vocoder.onnx")),
            sample_rate: 48000,
            n_mels: 128,
            hop_length: 512,
            opt_level: OptLevel::Basic,
            enable_profiling: true,
            enable_memory_pool: true,
            num_threads: 4,
            clip_threshold: 0.95,
            max_sequence_length: 4096,
        };
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.n_mels, 128);
        assert_eq!(config.hop_length, 512);
        assert!(config.enable_profiling);
        assert!(config.enable_memory_pool);
        assert_eq!(config.num_threads, 4);
        assert_eq!(config.max_sequence_length, 4096);
    }

    #[test]
    fn test_metadata_clone() {
        let meta = OnnxSingingMetadata {
            name: "test-model".to_string(),
            version: "1.0.0".to_string(),
            architecture: "DiffSinger".to_string(),
            input_names: vec!["phoneme_ids".to_string(), "note_ids".to_string()],
            output_names: vec!["mel".to_string()],
            n_mels: 80,
            sample_rate: 24000,
        };
        let cloned = meta.clone();
        assert_eq!(cloned.name, "test-model");
        assert_eq!(cloned.input_names.len(), 2);
        assert_eq!(cloned.output_names.len(), 1);
    }
}
