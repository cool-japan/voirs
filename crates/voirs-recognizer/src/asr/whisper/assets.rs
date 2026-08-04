//! Real pretrained assets for the pure-Rust Whisper implementation.
//!
//! The Whisper architecture in this crate is only usable with real trained parameters
//! and the real byte-pair vocabulary that produced them. This module locates those
//! assets on disk and turns them into a `candle` [`VarBuilder`] and a token map.
//!
//! Nothing here downloads anything: VoiRS never fetches weights implicitly. Export or
//! download the assets yourself, for example from `openai/whisper-tiny` on the
//! Hugging Face Hub, and point [`WhisperAssets`] at them.

use crate::RecognitionError;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Filesystem locations of the real pretrained assets a Whisper model needs.
///
/// # Example
///
/// ```no_run
/// use voirs_recognizer::asr::whisper::{WhisperAssets, WhisperConfig};
///
/// let config = WhisperConfig::tiny().with_assets(WhisperAssets::new(
///     "whisper-tiny/model.safetensors",
///     "whisper-tiny/vocab.json",
/// ));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperAssets {
    /// `safetensors` checkpoint holding the encoder and decoder parameters.
    pub weights: PathBuf,
    /// Byte-pair vocabulary (`vocab.json`) that the checkpoint was trained with.
    pub vocab: PathBuf,
}

impl WhisperAssets {
    /// Point at a checkpoint and its matching vocabulary.
    #[must_use]
    pub fn new(weights: impl Into<PathBuf>, vocab: impl Into<PathBuf>) -> Self {
        Self {
            weights: weights.into(),
            vocab: vocab.into(),
        }
    }

    /// Assume the conventional Hugging Face layout inside `dir`:
    /// `model.safetensors` and `vocab.json`.
    #[must_use]
    pub fn from_dir(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        Self {
            weights: dir.join("model.safetensors"),
            vocab: dir.join("vocab.json"),
        }
    }

    /// Verify that both files really exist and are readable.
    ///
    /// # Errors
    /// Returns [`RecognitionError::ModelLoadError`] naming the first missing file.
    pub fn validate(&self) -> Result<(), RecognitionError> {
        for (label, path) in [("weights", &self.weights), ("vocabulary", &self.vocab)] {
            if !path.is_file() {
                return Err(RecognitionError::ModelLoadError {
                    message: format!(
                        "Whisper {label} not found at {}. VoiRS does not download model assets: \
                         fetch openai/whisper-<size> yourself and set WhisperConfig::assets.",
                        path.display()
                    ),
                    source: None,
                });
            }
        }
        Ok(())
    }
}

/// The error returned by every constructor that needs assets it was not given.
#[must_use]
pub fn missing_assets_error(model_size: &str) -> RecognitionError {
    RecognitionError::ModelLoadError {
        message: format!(
            "No pretrained weights configured for the '{model_size}' Whisper model. This \
             implementation refuses to run with untrained parameters, because a zero- or \
             random-initialised network produces text that looks like a transcript but is \
             noise. Call WhisperConfig::with_assets(WhisperAssets::from_dir(..)) with a real \
             checkpoint, or use OnnxWhisper (`onnx` feature) with an exported graph."
        ),
        source: None,
    }
}

/// Build a [`VarBuilder`] backed by the real tensors in the configured checkpoint.
///
/// The whole checkpoint is read into memory (no `unsafe` memory mapping) and handed to
/// `candle` so that every layer constructed from the returned builder is initialised
/// from trained parameters.
///
/// # Errors
/// Returns [`RecognitionError::ModelLoadError`] when no assets are configured, when a
/// file is missing, or when the checkpoint cannot be parsed.
pub fn var_builder_from_assets(
    assets: Option<&WhisperAssets>,
    model_size: &str,
    device: &Device,
) -> Result<VarBuilder<'static>, RecognitionError> {
    let assets = assets.ok_or_else(|| missing_assets_error(model_size))?;
    assets.validate()?;

    let tensors = candle_core::safetensors::load(&assets.weights, device).map_err(|e| {
        RecognitionError::ModelLoadError {
            message: format!(
                "Failed to load Whisper checkpoint {}: {e}",
                assets.weights.display()
            ),
            source: Some(Box::new(e)),
        }
    })?;

    if tensors.is_empty() {
        return Err(RecognitionError::ModelLoadError {
            message: format!(
                "Whisper checkpoint {} contains no tensors",
                assets.weights.display()
            ),
            source: None,
        });
    }

    tracing::info!(
        "Loaded {} tensors from Whisper checkpoint {}",
        tensors.len(),
        assets.weights.display()
    );

    Ok(VarBuilder::from_tensors(tensors, DType::F32, device))
}

/// Read a Hugging Face `vocab.json` into a token-string to token-id map.
///
/// # Errors
/// Returns [`RecognitionError::ModelLoadError`] when the file cannot be read or is not a
/// JSON object of string keys to integer ids.
pub fn load_vocab(path: &Path) -> Result<HashMap<String, u32>, RecognitionError> {
    let bytes = std::fs::read(path).map_err(|e| RecognitionError::ModelLoadError {
        message: format!("Failed to read Whisper vocabulary {}: {e}", path.display()),
        source: Some(Box::new(e)),
    })?;

    let json: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| RecognitionError::ModelLoadError {
            message: format!(
                "Whisper vocabulary {} is not valid JSON: {e}",
                path.display()
            ),
            source: Some(Box::new(e)),
        })?;

    let object = json
        .as_object()
        .ok_or_else(|| RecognitionError::ModelLoadError {
            message: format!(
                "Whisper vocabulary {} is not a JSON object of token -> id",
                path.display()
            ),
            source: None,
        })?;

    let mut vocab = HashMap::with_capacity(object.len());
    for (token, id) in object {
        let id = id
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| RecognitionError::ModelLoadError {
                message: format!(
                    "Whisper vocabulary {}: token '{token}' has a non-integer id",
                    path.display()
                ),
                source: None,
            })?;
        vocab.insert(token.clone(), id);
    }

    if vocab.is_empty() {
        return Err(RecognitionError::ModelLoadError {
            message: format!("Whisper vocabulary {} is empty", path.display()),
            source: None,
        });
    }

    Ok(vocab)
}

/// The GPT-2 byte-to-unicode table used by Whisper's byte-level BPE.
///
/// Byte-level BPE maps every raw byte to a printable Unicode code point so that the
/// vocabulary contains no control characters. Decoding a token string therefore means
/// mapping each character back to the byte it stands for.
#[must_use]
pub fn byte_decoder() -> HashMap<char, u8> {
    let mut byte_to_char: Vec<(u8, char)> = Vec::with_capacity(256);
    let mut printable: Vec<u8> = Vec::new();

    // The three printable ASCII/Latin-1 runs map to themselves.
    printable.extend(b'!'..=b'~');
    printable.extend(0xA1_u8..=0xAC);
    printable.extend(0xAE_u8..=0xFF);

    let mut next_spare = 0_u32;
    for byte in 0..=255_u16 {
        let byte = byte as u8;
        if printable.contains(&byte) {
            byte_to_char.push((byte, char::from(byte)));
        } else {
            // Non-printable bytes are shifted into the 256.. range.
            let code = 256 + next_spare;
            next_spare += 1;
            if let Some(ch) = char::from_u32(code) {
                byte_to_char.push((byte, ch));
            }
        }
    }

    byte_to_char.into_iter().map(|(b, c)| (c, b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn missing_assets_are_reported_by_path() {
        let dir = tempfile::tempdir().unwrap();
        let assets = WhisperAssets::from_dir(dir.path());

        let err = assets.validate().unwrap_err();
        match err {
            RecognitionError::ModelLoadError { message, .. } => {
                assert!(message.contains("model.safetensors"), "unexpected: {message}");
                assert!(message.contains("does not download"), "unexpected: {message}");
            }
            other => panic!("expected ModelLoadError, got {other:?}"),
        }
    }

    #[test]
    fn var_builder_without_assets_fails_closed() {
        // `VarBuilder` is not `Debug`, so `unwrap_err()` is unavailable here.
        let Err(err) = var_builder_from_assets(None, "tiny", &Device::Cpu) else {
            panic!("building a VarBuilder without assets must fail closed");
        };
        assert!(
            err.to_string()
                .contains("refuses to run with untrained parameters"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn vocab_round_trips_real_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vocab.json");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(r#"{"hello":0,"\u0120world":1,"!":2}"#.as_bytes())
            .unwrap();
        drop(file);

        let vocab = load_vocab(&path).unwrap();
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab["hello"], 0);
        assert_eq!(vocab["Ġworld"], 1);
    }

    #[test]
    fn vocab_rejects_malformed_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vocab.json");
        std::fs::write(&path, b"[1, 2, 3]").unwrap();
        assert!(load_vocab(&path).is_err());

        std::fs::write(&path, b"{}").unwrap();
        assert!(load_vocab(&path).is_err());

        std::fs::write(&path, br#"{"a":"not-an-id"}"#).unwrap();
        assert!(load_vocab(&path).is_err());
    }

    #[test]
    fn byte_decoder_covers_every_byte() {
        let decoder = byte_decoder();
        assert_eq!(decoder.len(), 256, "every byte must have a distinct symbol");

        // Printable ASCII maps to itself.
        assert_eq!(decoder[&'a'], b'a');
        assert_eq!(decoder[&'~'], b'~');
        // The space byte is shifted into the private run, as GPT-2 specifies.
        assert_eq!(decoder[&'\u{0120}'], b' ');
    }
}
