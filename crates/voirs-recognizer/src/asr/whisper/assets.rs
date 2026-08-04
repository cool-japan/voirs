//! Real pretrained assets for the pure-Rust Whisper implementation.
//!
//! The Whisper architecture in this crate is only usable with real trained parameters
//! and the real byte-pair vocabulary that produced them. This module locates those
//! assets on disk and turns them into a `candle` [`VarBuilder`] and a token map.
//!
//! Nothing here downloads anything: VoiRS never fetches weights implicitly. Provide the
//! assets yourself and point [`WhisperAssets`] at them.
//!
//! # Required tensor naming
//!
//! The encoder and decoder in this crate read the OpenAI-style parameter names
//! ([`WhisperLayout::Voirs`]):
//!
//! ```text
//! encoder.positional_embedding      decoder.positional_embedding
//! encoder.conv1.weight              decoder.token_embedding.weight
//! encoder.blocks.N.attn.query.…     decoder.blocks.N.cross_attn.…
//! encoder.blocks.N.mlp.c_fc.…       decoder.ln.weight
//! encoder.ln_post.weight
//! ```
//!
//! The `openai/whisper-*` checkpoints published on the Hugging Face Hub use the
//! `transformers` naming instead (`model.encoder.layers.N.self_attn.q_proj.weight`,
//! `model.decoder.embed_tokens.weight`, `…fc1.weight`) and are **not** loadable here —
//! [`check_layout`] detects them and fails closed with a message saying so, rather than
//! leaving layers silently uninitialised.
//!
//! To run a stock Hugging Face checkpoint today, use `OnnxWhisper` (feature `onnx`) with
//! an ONNX export, or `candle_transformers::models::whisper`, which is already a
//! dependency of this crate and reads that layout as published.

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

/// Environment variable naming a directory of real Whisper assets.
pub const ASSETS_ENV_VAR: &str = "VOIRS_WHISPER_ASSETS";

/// Discover real assets from the [`ASSETS_ENV_VAR`] environment variable.
///
/// The variable must name a directory laid out like a Hugging Face checkout, holding
/// `model.safetensors` and `vocab.json`. Returns `None` when the variable is unset or
/// the directory does not really contain both files, so callers can fall back to their
/// own configuration instead of failing on a typo.
///
/// This is also how the crate's model-dependent tests opt in to running against a real
/// checkpoint; without it they assert the fail-closed behaviour instead.
#[must_use]
pub fn assets_from_env() -> Option<WhisperAssets> {
    let dir = std::env::var_os(ASSETS_ENV_VAR)?;
    let assets = WhisperAssets::from_dir(dir);
    assets.validate().ok().map(|()| assets)
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
    // Reject a checkpoint written in a naming scheme these modules cannot read *before*
    // building layers from it, so the caller gets a precise diagnostic instead of a bare
    // "cannot find tensor" from deep inside layer construction.
    check_layout(&assets.weights)?;

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

/// Parameter-naming schemes published Whisper checkpoints use.
///
/// The three differ only in how they spell their tensors; the architecture is the same.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhisperLayout {
    /// What these modules read: `encoder.blocks.N.attn.query.weight`,
    /// `encoder.ln_post.weight`, `decoder.token_embedding.weight`, `…mlp.c_fc.weight`.
    Voirs,
    /// The `transformers` layout published as `openai/whisper-*` on the Hugging Face
    /// Hub: `model.encoder.layers.N.self_attn.q_proj.weight`,
    /// `model.decoder.embed_tokens.weight`, `…fc1.weight`.
    HuggingFace,
    /// The original OpenAI release: `encoder.blocks.N.attn.query.weight` with the MLP as
    /// an indexed `Sequential` (`…mlp.0.weight`, `…mlp.2.weight`).
    OpenAi,
    /// Nothing recognisable.
    Unknown,
}

impl WhisperLayout {
    /// Identify the layout from the tensor names a checkpoint really declares.
    #[must_use]
    pub fn detect(names: &[&str]) -> Self {
        let has = |needle: &str| names.iter().any(|name| name.contains(needle));

        if has("model.encoder.") || has("model.decoder.") || has(".self_attn.q_proj") {
            return Self::HuggingFace;
        }
        // The MLP spelling is the only thing that really separates the VoiRS naming from
        // the original OpenAI release: both use `blocks.N.attn.query` and `ln_post`, but
        // OpenAI stores the MLP as an indexed `Sequential`. Discriminate on it first, so
        // a name the two share can never decide the answer.
        if has(".mlp.c_fc") || has(".mlp.c_proj") {
            return Self::Voirs;
        }
        if has(".mlp.0.") || has(".mlp.2.") {
            return Self::OpenAi;
        }
        if has("ln_post") || has(".attn.query") || has("token_embedding") {
            return Self::Voirs;
        }
        Self::Unknown
    }

    /// Human-readable name used in diagnostics.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Voirs => "VoiRS/OpenAI-style",
            Self::HuggingFace => "Hugging Face `transformers`",
            Self::OpenAi => "original OpenAI release",
            Self::Unknown => "unrecognised",
        }
    }
}

/// Verify that a checkpoint is written in the naming scheme these modules read.
///
/// # Errors
/// Returns [`RecognitionError::ModelLoadError`] when the header cannot be parsed, or
/// when the checkpoint uses a different naming scheme — with a message naming both the
/// detected scheme and the working alternatives, instead of letting layer construction
/// fail later with a bare "cannot find tensor".
pub fn check_layout(weights: &Path) -> Result<(), RecognitionError> {
    let header = crate::asr::weights::SafetensorsHeader::read(weights)?;
    let names: Vec<&str> = header.tensors.keys().map(String::as_str).collect();

    match WhisperLayout::detect(&names) {
        WhisperLayout::Voirs => Ok(()),
        other => Err(RecognitionError::ModelLoadError {
            message: format!(
                "{} is a {} checkpoint ({} tensors), but this pure-Rust Whisper reads the \
                 {} naming scheme (`encoder.blocks.N.attn.query.weight`, \
                 `decoder.token_embedding.weight`, `...mlp.c_fc.weight`). Loading it would \
                 silently leave layers uninitialised, so it is refused. Either convert the \
                 tensor names, or use a backend that reads this checkpoint directly: \
                 `OnnxWhisper` (feature `onnx`) with an ONNX export, or \
                 `candle_transformers::models::whisper`, which reads the Hugging Face layout \
                 as published.",
                weights.display(),
                other.label(),
                header.tensors.len(),
                WhisperLayout::Voirs.label(),
            ),
            source: None,
        }),
    }
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
                assert!(
                    message.contains("model.safetensors"),
                    "unexpected: {message}"
                );
                assert!(
                    message.contains("does not download"),
                    "unexpected: {message}"
                );
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

    /// Build a real (header-only) safetensors file declaring the given tensor names.
    fn write_named_checkpoint(dir: &std::path::Path, names: &[&str]) -> PathBuf {
        let mut header = serde_json::Map::new();
        let mut offset = 0_u64;
        for name in names {
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": [2_usize, 2],
                    "data_offsets": [offset, offset + 16],
                }),
            );
            offset += 16;
        }
        let header_bytes = serde_json::to_vec(&header).expect("serialise");
        let path = dir.join("model.safetensors");
        let mut file = std::fs::File::create(&path).expect("create");
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .expect("write len");
        file.write_all(&header_bytes).expect("write header");
        file.write_all(&vec![0_u8; offset as usize])
            .expect("write payload");
        path
    }

    #[test]
    fn layout_detection_tells_the_published_schemes_apart() {
        assert_eq!(
            WhisperLayout::detect(&[
                "model.encoder.layers.0.self_attn.q_proj.weight",
                "model.decoder.embed_tokens.weight",
            ]),
            WhisperLayout::HuggingFace
        );
        assert_eq!(
            WhisperLayout::detect(&[
                "encoder.blocks.0.attn.query.weight",
                "encoder.blocks.0.mlp.c_fc.weight",
                "encoder.ln_post.weight",
            ]),
            WhisperLayout::Voirs
        );
        assert_eq!(
            WhisperLayout::detect(&[
                "encoder.blocks.0.attn.query.weight",
                "encoder.blocks.0.mlp.0.weight",
            ]),
            WhisperLayout::OpenAi
        );
        // Regression guard: the OpenAI release also carries `ln_post` and
        // `token_embedding`, so those shared names must not outvote the MLP spelling.
        assert_eq!(
            WhisperLayout::detect(&[
                "encoder.ln_post.weight",
                "encoder.blocks.0.attn.query.weight",
                "encoder.blocks.0.mlp.0.weight",
                "encoder.blocks.0.mlp.2.weight",
                "decoder.token_embedding.weight",
            ]),
            WhisperLayout::OpenAi,
            "a full OpenAI checkpoint must not be mistaken for the VoiRS layout"
        );
        // ...and a full VoiRS checkpoint is still recognised with all of them present.
        assert_eq!(
            WhisperLayout::detect(&[
                "encoder.ln_post.weight",
                "encoder.blocks.0.attn.query.weight",
                "encoder.blocks.0.mlp.c_fc.weight",
                "encoder.blocks.0.mlp.c_proj.weight",
                "decoder.token_embedding.weight",
            ]),
            WhisperLayout::Voirs
        );
        assert_eq!(
            WhisperLayout::detect(&["something.completely.unrelated"]),
            WhisperLayout::Unknown
        );
    }

    #[test]
    fn hugging_face_checkpoints_are_refused_with_an_actionable_message() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_named_checkpoint(
            dir.path(),
            &[
                "model.encoder.conv1.weight",
                "model.encoder.layers.0.self_attn.q_proj.weight",
                "model.decoder.embed_tokens.weight",
            ],
        );

        let err = check_layout(&path).unwrap_err();
        let rendered = err.to_string();
        assert!(rendered.contains("Hugging Face"), "unexpected: {rendered}");
        // The message must name a route that actually works today.
        assert!(rendered.contains("OnnxWhisper"), "unexpected: {rendered}");
        assert!(
            rendered.contains("candle_transformers"),
            "unexpected: {rendered}"
        );
        // And it must not pretend the file was loaded.
        assert!(rendered.contains("refused"), "unexpected: {rendered}");
    }

    #[test]
    fn matching_checkpoints_pass_the_layout_check() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_named_checkpoint(
            dir.path(),
            &[
                "encoder.positional_embedding",
                "encoder.blocks.0.attn.query.weight",
                "encoder.blocks.0.mlp.c_fc.weight",
                "encoder.ln_post.weight",
                "decoder.token_embedding.weight",
            ],
        );
        check_layout(&path).expect("a VoiRS-layout checkpoint must be accepted");
    }

    #[test]
    fn var_builder_rejects_a_mismatched_checkpoint_before_building_layers() {
        let dir = tempfile::tempdir().unwrap();
        write_named_checkpoint(dir.path(), &["model.encoder.conv1.weight"]);
        std::fs::write(dir.path().join("vocab.json"), br#"{"a":0}"#).unwrap();

        let assets = WhisperAssets::from_dir(dir.path());
        let Err(err) = var_builder_from_assets(Some(&assets), "tiny", &Device::Cpu) else {
            panic!("a mismatched checkpoint must not produce a VarBuilder");
        };
        assert!(err.to_string().contains("Hugging Face"), "{err}");
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
