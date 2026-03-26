//! OxiONNX backend for grapheme-to-phoneme conversion.
//!
//! This module provides an ONNX-based G2P model that converts text into phoneme
//! sequences using a neural sequence-to-sequence model. The model takes grapheme
//! (character) IDs as input and produces phoneme IDs as output, which are then
//! decoded back to phoneme strings using a vocabulary mapping.

use crate::{G2pError, Result};
use oxionnx::{OptLevel, Session, Tensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tracing::{debug, info, warn};

/// Configuration for the ONNX G2P backend.
#[derive(Debug, Clone)]
pub struct OnnxG2pConfig {
    /// Path to the ONNX model file.
    pub model_path: PathBuf,

    /// Grapheme vocabulary: maps each character to its integer ID.
    pub grapheme_vocab: HashMap<char, u32>,

    /// Phoneme vocabulary: maps each integer ID back to its phoneme string.
    pub phoneme_vocab: HashMap<u32, String>,

    /// Maximum input sequence length in characters (default: 256).
    pub max_input_length: usize,

    /// Maximum output sequence length in phoneme tokens (default: 512).
    pub max_output_length: usize,

    /// ONNX graph optimization level (default: All).
    pub opt_level: OptLevel,

    /// Enable per-node profiling during inference.
    pub enable_profiling: bool,

    /// Enable memory pool for buffer reuse.
    pub enable_memory_pool: bool,
}

impl Default for OnnxG2pConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            grapheme_vocab: HashMap::new(),
            phoneme_vocab: HashMap::new(),
            max_input_length: 256,
            max_output_length: 512,
            opt_level: OptLevel::All,
            enable_profiling: false,
            enable_memory_pool: false,
        }
    }
}

/// ONNX-based grapheme-to-phoneme converter.
///
/// Converts input text to phoneme sequences using a pre-trained neural
/// sequence-to-sequence model exported to ONNX format.
///
/// ## Model I/O
///
/// - **Input**: `grapheme_ids` tensor of shape `[1, seq_len]` (float32 character IDs).
/// - **Output**: `phoneme_ids` tensor of shape `[1, out_seq_len]` (float32 phoneme IDs).
///
/// ## Vocabulary
///
/// The `grapheme_vocab` maps characters to integer IDs for the model input.
/// The `phoneme_vocab` maps integer IDs from the model output to phoneme strings.
/// Characters not found in the vocabulary are skipped with a warning.
pub struct OnnxG2p {
    /// The loaded ONNX session.
    session: Arc<RwLock<Session>>,

    /// Configuration snapshot.
    config: OnnxG2pConfig,
}

impl OnnxG2p {
    /// Create a new ONNX G2P backend by loading the model from disk.
    pub fn new(config: OnnxG2pConfig) -> Result<Self> {
        info!("Loading ONNX G2P model from {:?}", config.model_path);

        let session = load_session(&config.model_path, &config)?;

        info!("ONNX G2P model loaded successfully");
        debug!(
            "Config: grapheme_vocab_size={}, phoneme_vocab_size={}, max_input={}, max_output={}",
            config.grapheme_vocab.len(),
            config.phoneme_vocab.len(),
            config.max_input_length,
            config.max_output_length,
        );

        Ok(Self {
            session: Arc::new(RwLock::new(session)),
            config,
        })
    }

    /// Convert text to a phoneme string.
    ///
    /// Characters are mapped to IDs via `grapheme_vocab`, fed through the ONNX
    /// model, and the resulting phoneme IDs are decoded via `phoneme_vocab`.
    /// Individual phonemes are joined with spaces.
    ///
    /// # Arguments
    /// * `text` - The input text to convert.
    ///
    /// # Returns
    /// A space-separated phoneme string.
    pub fn convert(&self, text: &str) -> Result<String> {
        if text.is_empty() {
            return Err(G2pError::InvalidInput(
                "Input text must not be empty".to_string(),
            ));
        }

        let grapheme_ids = self.text_to_ids(text)?;
        let phoneme_ids = self.convert_to_ids(&grapheme_ids)?;
        let phonemes = self.ids_to_phonemes(&phoneme_ids);

        Ok(phonemes.join(" "))
    }

    /// Convert raw grapheme IDs through the ONNX model to phoneme IDs.
    ///
    /// # Arguments
    /// * `grapheme_ids` - A slice of character IDs matching the model vocabulary.
    ///
    /// # Returns
    /// A vector of phoneme IDs produced by the model.
    pub fn convert_to_ids(&self, grapheme_ids: &[u32]) -> Result<Vec<u32>> {
        if grapheme_ids.is_empty() {
            return Err(G2pError::InvalidInput(
                "Grapheme IDs must not be empty".to_string(),
            ));
        }

        // Truncate to max input length
        let ids = if grapheme_ids.len() > self.config.max_input_length {
            warn!(
                "Input length {} exceeds max_input_length {}, truncating",
                grapheme_ids.len(),
                self.config.max_input_length
            );
            &grapheme_ids[..self.config.max_input_length]
        } else {
            grapheme_ids
        };

        let seq_len = ids.len();

        // Build input tensor [1, seq_len] (Tensor stores f32, so cast)
        let float_ids: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
        let input = Tensor::new(float_ids, vec![1, seq_len]);

        let mut inputs = std::collections::HashMap::new();
        inputs.insert("grapheme_ids", input);

        // Run inference
        let session = self
            .session
            .read()
            .map_err(|e| G2pError::ModelError(format!("Session lock poisoned: {e}")))?;

        let outputs = session.run(&inputs).map_err(|e| G2pError::BackendError {
            backend: "onnx".to_string(),
            message: format!("ONNX inference failed: {e}"),
        })?;

        // Extract phoneme IDs
        let phoneme_tensor = outputs
            .get("phoneme_ids")
            .ok_or_else(|| G2pError::BackendError {
                backend: "onnx".to_string(),
                message: "Missing 'phoneme_ids' output from ONNX model".to_string(),
            })?;

        // Convert f32 data back to u32 IDs, truncate to max output length
        let max_out = self.config.max_output_length.min(phoneme_tensor.data.len());
        let result: Vec<u32> = phoneme_tensor.data[..max_out]
            .iter()
            .map(|&v| v.round().max(0.0) as u32)
            .collect();

        debug!(
            "G2P ONNX: input_len={}, output_len={}",
            seq_len,
            result.len()
        );

        Ok(result)
    }

    /// Map text characters to grapheme IDs using the configured vocabulary.
    /// Unknown characters are skipped with a trace-level warning.
    fn text_to_ids(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::with_capacity(text.len());

        for ch in text.chars() {
            if let Some(&id) = self.config.grapheme_vocab.get(&ch) {
                ids.push(id);
            } else {
                debug!("Character '{}' not in grapheme vocabulary, skipping", ch);
            }
        }

        if ids.is_empty() {
            return Err(G2pError::InvalidInput(
                "No valid graphemes found in input text".to_string(),
            ));
        }

        Ok(ids)
    }

    /// Decode phoneme IDs to phoneme strings using the configured vocabulary.
    /// Unknown IDs are represented as `<unk:ID>`.
    fn ids_to_phonemes(&self, ids: &[u32]) -> Vec<String> {
        ids.iter()
            .map(|&id| {
                self.config
                    .phoneme_vocab
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| format!("<unk:{id}>"))
            })
            .collect()
    }

    /// Return the grapheme vocabulary size.
    pub fn grapheme_vocab_size(&self) -> usize {
        self.config.grapheme_vocab.len()
    }

    /// Return the phoneme vocabulary size.
    pub fn phoneme_vocab_size(&self) -> usize {
        self.config.phoneme_vocab.len()
    }
}

/// Load an ONNX session from disk with the supplied configuration.
fn load_session(path: &Path, config: &OnnxG2pConfig) -> Result<Session> {
    let mut builder = Session::builder()
        .with_optimization_level(config.opt_level)
        .with_memory_pool(config.enable_memory_pool);
    if config.enable_profiling {
        builder = builder.with_profiling();
    }
    builder.load(path).map_err(|e| {
        G2pError::ModelError(format!(
            "Failed to load ONNX model from {}: {e}",
            path.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OnnxG2pConfig::default();
        assert_eq!(config.max_input_length, 256);
        assert_eq!(config.max_output_length, 512);
        assert!(config.grapheme_vocab.is_empty());
        assert!(config.phoneme_vocab.is_empty());
        assert!(!config.enable_profiling);
        assert!(!config.enable_memory_pool);
    }

    #[test]
    fn test_ids_to_phonemes() {
        let mut phoneme_vocab = HashMap::new();
        phoneme_vocab.insert(0, "p".to_string());
        phoneme_vocab.insert(1, "ae".to_string());
        phoneme_vocab.insert(2, "t".to_string());

        let config = OnnxG2pConfig {
            phoneme_vocab,
            ..Default::default()
        };

        // We cannot call ids_to_phonemes directly since it requires OnnxG2p,
        // but we can test the logic inline.
        let ids = [0u32, 1, 2, 99];
        let result: Vec<String> = ids
            .iter()
            .map(|&id| {
                config
                    .phoneme_vocab
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| format!("<unk:{id}>"))
            })
            .collect();

        assert_eq!(result, vec!["p", "ae", "t", "<unk:99>"]);
    }

    #[test]
    fn test_text_to_ids_logic() {
        let mut grapheme_vocab = HashMap::new();
        grapheme_vocab.insert('h', 0);
        grapheme_vocab.insert('e', 1);
        grapheme_vocab.insert('l', 2);
        grapheme_vocab.insert('o', 3);

        let text = "hello";
        let mut ids = Vec::new();
        for ch in text.chars() {
            if let Some(&id) = grapheme_vocab.get(&ch) {
                ids.push(id);
            }
        }
        assert_eq!(ids, vec![0, 1, 2, 2, 3]);
    }
}
