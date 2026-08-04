//! Real inspection of pretrained weight files.
//!
//! VoiRS never fabricates model weights. Backends that need pretrained parameters use
//! this module to locate a real `safetensors` file and to parse its header for real,
//! so that they can report precise, truthful diagnostics — the tensors that are
//! present, their shapes, the real parameter count — or fail closed with a typed error
//! when the weights are missing or malformed.
//!
//! Only the header is read (a few kilobytes), never the whole tensor payload, so
//! validation is cheap even for multi-gigabyte checkpoints.

use crate::RecognitionError;
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Largest `safetensors` JSON header VoiRS will parse (100 MB).
///
/// Real checkpoint headers are a few hundred kilobytes at most; a larger value means
/// the file is not a `safetensors` container (or is hostile) and is rejected.
const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024;

/// Conventional file names searched when a directory is supplied instead of a file.
const CANDIDATE_FILE_NAMES: &[&str] = &["model.safetensors", "pytorch_model.safetensors"];

/// Description of a single tensor as declared by the `safetensors` header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDescriptor {
    /// `safetensors` dtype string, e.g. `F32`, `F16`, `BF16`, `I64`.
    pub dtype: String,
    /// Tensor shape, outermost dimension first.
    pub shape: Vec<usize>,
    /// Byte range of this tensor within the payload region.
    pub data_offsets: (u64, u64),
}

impl TensorDescriptor {
    /// Number of scalar elements in this tensor.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.shape.iter().copied().product::<usize>()
    }

    /// Size of one element in bytes, or `None` for an unrecognised dtype.
    #[must_use]
    pub fn element_size(&self) -> Option<usize> {
        Some(match self.dtype.as_str() {
            "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1,
            "U16" | "I16" | "F16" | "BF16" => 2,
            "U32" | "I32" | "F32" => 4,
            "U64" | "I64" | "F64" => 8,
            _ => return None,
        })
    }
}

/// Parsed `safetensors` header plus the real on-disk facts about the file.
#[derive(Debug, Clone)]
pub struct SafetensorsHeader {
    /// Path the header was read from.
    pub path: PathBuf,
    /// Declared tensors, keyed by name.
    pub tensors: BTreeMap<String, TensorDescriptor>,
    /// Free-form `__metadata__` entries, if any.
    pub metadata: BTreeMap<String, String>,
    /// Real total file size in bytes.
    pub file_size_bytes: u64,
    /// Byte offset at which the tensor payload region begins (`8 + header_len`).
    payload_start: u64,
}

impl SafetensorsHeader {
    /// Resolve `path` to a `safetensors` file and parse its header.
    ///
    /// If `path` is a directory, the conventional checkpoint file names are tried in
    /// order.
    ///
    /// # Errors
    /// Returns [`RecognitionError::ModelLoadError`] when the path does not resolve to a
    /// readable file, when the file is not a valid `safetensors` container, or when the
    /// declared tensor offsets do not fit the real file length.
    pub fn read(path: impl AsRef<Path>) -> Result<Self, RecognitionError> {
        let resolved = resolve_weights_path(path.as_ref())?;
        let mut file = std::fs::File::open(&resolved).map_err(|e| load_error(&resolved, &e))?;
        let file_size_bytes = file
            .metadata()
            .map_err(|e| load_error(&resolved, &e))?
            .len();

        if file_size_bytes < 8 {
            return Err(RecognitionError::ModelLoadError {
                message: format!(
                    "{} is only {file_size_bytes} bytes: too small to be a safetensors file",
                    resolved.display()
                ),
                source: None,
            });
        }

        let mut len_bytes = [0_u8; 8];
        file.read_exact(&mut len_bytes)
            .map_err(|e| load_error(&resolved, &e))?;
        let header_len = u64::from_le_bytes(len_bytes);

        if header_len == 0 || header_len > MAX_HEADER_BYTES || header_len > file_size_bytes - 8 {
            return Err(RecognitionError::ModelLoadError {
                message: format!(
                    "{} declares a {header_len}-byte safetensors header, which does not fit a \
                     {file_size_bytes}-byte file: not a safetensors checkpoint",
                    resolved.display()
                ),
                source: None,
            });
        }

        // `header_len` is bounded by MAX_HEADER_BYTES above, so this cast cannot wrap on
        // any platform with a 32-bit-or-wider usize.
        let mut header_bytes = vec![0_u8; usize::try_from(header_len).unwrap_or(usize::MAX)];
        file.seek(SeekFrom::Start(8))
            .map_err(|e| load_error(&resolved, &e))?;
        file.read_exact(&mut header_bytes)
            .map_err(|e| load_error(&resolved, &e))?;

        let json: serde_json::Value = serde_json::from_slice(&header_bytes).map_err(|e| {
            RecognitionError::ModelLoadError {
                message: format!(
                    "{} has a safetensors header that is not valid JSON: {e}",
                    resolved.display()
                ),
                source: Some(Box::new(e)),
            }
        })?;
        let object = json
            .as_object()
            .ok_or_else(|| RecognitionError::ModelLoadError {
                message: format!(
                    "{} has a safetensors header that is not a JSON object",
                    resolved.display()
                ),
                source: None,
            })?;

        let payload_len = file_size_bytes - 8 - header_len;
        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();

        for (name, value) in object {
            if name == "__metadata__" {
                if let Some(entries) = value.as_object() {
                    for (key, item) in entries {
                        if let Some(text) = item.as_str() {
                            metadata.insert(key.clone(), text.to_string());
                        }
                    }
                }
                continue;
            }

            let descriptor = parse_tensor_descriptor(&resolved, name, value)?;
            if descriptor.data_offsets.1 > payload_len
                || descriptor.data_offsets.0 > descriptor.data_offsets.1
            {
                return Err(RecognitionError::ModelLoadError {
                    message: format!(
                        "{}: tensor '{name}' declares byte range {}..{} which exceeds the \
                         {payload_len}-byte payload",
                        resolved.display(),
                        descriptor.data_offsets.0,
                        descriptor.data_offsets.1
                    ),
                    source: None,
                });
            }
            tensors.insert(name.clone(), descriptor);
        }

        if tensors.is_empty() {
            return Err(RecognitionError::ModelLoadError {
                message: format!("{} declares no tensors", resolved.display()),
                source: None,
            });
        }

        Ok(Self {
            path: resolved,
            tensors,
            metadata,
            file_size_bytes,
            payload_start: 8 + header_len,
        })
    }

    /// Total number of scalar parameters across all declared tensors.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.tensors
            .values()
            .map(TensorDescriptor::element_count)
            .sum()
    }

    /// Real size of the checkpoint in mebibytes.
    #[must_use]
    pub fn size_mb(&self) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        {
            self.file_size_bytes as f32 / (1024.0 * 1024.0)
        }
    }

    /// Names from `required` that the checkpoint does not declare.
    #[must_use]
    pub fn missing_tensors(&self, required: &[&str]) -> Vec<String> {
        required
            .iter()
            .filter(|name| !self.tensors.contains_key(**name))
            .map(|name| (*name).to_string())
            .collect()
    }

    /// Whether any declared tensor name starts with `prefix`.
    #[must_use]
    pub fn has_prefix(&self, prefix: &str) -> bool {
        self.tensors.keys().any(|name| name.starts_with(prefix))
    }

    /// Read one named tensor's real bytes from the file and convert them to `f32`.
    ///
    /// Supports the `F32`, `F16` and `BF16` dtypes, which cover every published
    /// speech checkpoint VoiRS targets.
    ///
    /// # Errors
    /// Returns [`RecognitionError::ModelLoadError`] when the tensor is absent, stored in
    /// an unsupported dtype, or when its declared byte range does not match its shape.
    pub fn read_tensor_f32(&self, name: &str) -> Result<Vec<f32>, RecognitionError> {
        let descriptor =
            self.tensors
                .get(name)
                .ok_or_else(|| RecognitionError::ModelLoadError {
                    message: format!("{}: checkpoint has no tensor '{name}'", self.path.display()),
                    source: None,
                })?;

        let element_size =
            descriptor
                .element_size()
                .ok_or_else(|| RecognitionError::ModelLoadError {
                    message: format!(
                        "{}: tensor '{name}' has unsupported dtype '{}'",
                        self.path.display(),
                        descriptor.dtype
                    ),
                    source: None,
                })?;

        let elements = descriptor.element_count();
        let declared = descriptor.data_offsets.1 - descriptor.data_offsets.0;
        if declared != (elements * element_size) as u64 {
            return Err(RecognitionError::ModelLoadError {
                message: format!(
                    "{}: tensor '{name}' declares {declared} bytes but its shape {:?} of dtype \
                     {} needs {}",
                    self.path.display(),
                    descriptor.shape,
                    descriptor.dtype,
                    elements * element_size
                ),
                source: None,
            });
        }

        let mut file = std::fs::File::open(&self.path).map_err(|e| load_error(&self.path, &e))?;
        file.seek(SeekFrom::Start(
            self.payload_start + descriptor.data_offsets.0,
        ))
        .map_err(|e| load_error(&self.path, &e))?;
        let mut raw = vec![0_u8; elements * element_size];
        file.read_exact(&mut raw)
            .map_err(|e| load_error(&self.path, &e))?;

        let values = match descriptor.dtype.as_str() {
            "F32" => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            "F16" => raw
                .chunks_exact(2)
                .map(|c| f32::from(half::f16::from_le_bytes([c[0], c[1]])))
                .collect(),
            "BF16" => raw
                .chunks_exact(2)
                .map(|c| f32::from(half::bf16::from_le_bytes([c[0], c[1]])))
                .collect(),
            other => {
                return Err(RecognitionError::ModelLoadError {
                    message: format!(
                        "{}: tensor '{name}' has dtype '{other}', which is not a float type \
                         VoiRS can load",
                        self.path.display()
                    ),
                    source: None,
                })
            }
        };

        Ok(values)
    }

    /// Read a named tensor as a row-major `rows` x `cols` matrix.
    ///
    /// # Errors
    /// Returns [`RecognitionError::ModelLoadError`] when the tensor is absent or its
    /// real shape differs from `[rows, cols]`.
    pub fn read_matrix(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<Vec<f32>>, RecognitionError> {
        self.expect_shape(name, &[rows, cols])?;
        let flat = self.read_tensor_f32(name)?;
        Ok(flat.chunks_exact(cols).map(<[f32]>::to_vec).collect())
    }

    /// Read a named tensor as a vector of exactly `len` elements.
    ///
    /// # Errors
    /// Returns [`RecognitionError::ModelLoadError`] when the tensor is absent or its
    /// real shape differs from `[len]`.
    pub fn read_vector(&self, name: &str, len: usize) -> Result<Vec<f32>, RecognitionError> {
        self.expect_shape(name, &[len])?;
        self.read_tensor_f32(name)
    }

    /// Verify that a tensor's declared shape matches what the caller expects.
    ///
    /// # Errors
    /// Returns [`RecognitionError::ModelLoadError`] on absence or shape mismatch.
    pub fn expect_shape(&self, name: &str, shape: &[usize]) -> Result<(), RecognitionError> {
        let descriptor =
            self.tensors
                .get(name)
                .ok_or_else(|| RecognitionError::ModelLoadError {
                    message: format!("{}: checkpoint has no tensor '{name}'", self.path.display()),
                    source: None,
                })?;
        if descriptor.shape != shape {
            return Err(RecognitionError::ModelLoadError {
                message: format!(
                    "{}: tensor '{name}' has shape {:?}, expected {shape:?}",
                    self.path.display(),
                    descriptor.shape
                ),
                source: None,
            });
        }
        Ok(())
    }
}

/// Resolve a user-supplied path to a concrete weights file.
///
/// # Errors
/// Returns [`RecognitionError::ModelLoadError`] when the path is missing, or when a
/// supplied directory contains none of the conventional checkpoint file names.
pub fn resolve_weights_path(path: &Path) -> Result<PathBuf, RecognitionError> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    if path.is_dir() {
        for name in CANDIDATE_FILE_NAMES {
            let candidate = path.join(name);
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
        return Err(RecognitionError::ModelLoadError {
            message: format!(
                "{} contains none of the expected checkpoint files ({})",
                path.display(),
                CANDIDATE_FILE_NAMES.join(", ")
            ),
            source: None,
        });
    }
    Err(RecognitionError::ModelLoadError {
        message: format!("Model weights not found at {}", path.display()),
        source: None,
    })
}

fn parse_tensor_descriptor(
    path: &Path,
    name: &str,
    value: &serde_json::Value,
) -> Result<TensorDescriptor, RecognitionError> {
    let malformed = |detail: &str| RecognitionError::ModelLoadError {
        message: format!("{}: tensor '{name}' {detail}", path.display()),
        source: None,
    };

    let entry = value
        .as_object()
        .ok_or_else(|| malformed("is not a JSON object"))?;
    let dtype = entry
        .get("dtype")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| malformed("has no 'dtype' string"))?
        .to_string();

    let shape = entry
        .get("shape")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| malformed("has no 'shape' array"))?
        .iter()
        .map(|dim| {
            dim.as_u64()
                .and_then(|d| usize::try_from(d).ok())
                .ok_or_else(|| malformed("has a non-integer dimension"))
        })
        .collect::<Result<Vec<usize>, RecognitionError>>()?;

    let offsets = entry
        .get("data_offsets")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| malformed("has no 'data_offsets' array"))?;
    if offsets.len() != 2 {
        return Err(malformed("has a 'data_offsets' array that is not a pair"));
    }
    let start = offsets[0]
        .as_u64()
        .ok_or_else(|| malformed("has a non-integer start offset"))?;
    let end = offsets[1]
        .as_u64()
        .ok_or_else(|| malformed("has a non-integer end offset"))?;

    Ok(TensorDescriptor {
        dtype,
        shape,
        data_offsets: (start, end),
    })
}

fn load_error(path: &Path, error: &std::io::Error) -> RecognitionError {
    RecognitionError::ModelLoadError {
        message: format!("Failed to read {}: {error}", path.display()),
        source: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a real (tiny) safetensors file on disk.
    fn write_safetensors(dir: &Path, name: &str, tensors: &[(&str, &str, Vec<usize>)]) -> PathBuf {
        let mut header = serde_json::Map::new();
        let mut offset = 0_u64;
        for (tensor_name, dtype, shape) in tensors {
            let elements: usize = shape.iter().product();
            let elem_size = match *dtype {
                "F16" | "BF16" => 2_u64,
                "F64" | "I64" => 8,
                _ => 4,
            };
            let bytes = elements as u64 * elem_size;
            header.insert(
                (*tensor_name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }

        let header_bytes = serde_json::to_vec(&header).unwrap();
        let path = dir.join(name);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&vec![0_u8; offset as usize]).unwrap();
        file.flush().unwrap();
        path
    }

    #[test]
    fn reads_real_header_and_counts_real_parameters() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_safetensors(
            dir.path(),
            "model.safetensors",
            &[
                ("encoder.conv1.weight", "F32", vec![8, 4, 3]),
                ("encoder.conv1.bias", "F32", vec![8]),
                ("lm_head.weight", "F16", vec![32, 8]),
            ],
        );

        let header = SafetensorsHeader::read(&path).unwrap();
        assert_eq!(header.tensors.len(), 3);
        // 8*4*3 + 8 + 32*8 = 96 + 8 + 256
        assert_eq!(header.parameter_count(), 360);
        assert_eq!(
            header.tensors["encoder.conv1.weight"].shape,
            vec![8_usize, 4, 3]
        );
        assert_eq!(header.tensors["lm_head.weight"].element_size(), Some(2));
        assert!(header.has_prefix("encoder."));
        assert!(!header.has_prefix("decoder."));
        assert!(header.missing_tensors(&["encoder.conv1.bias"]).is_empty());
        assert_eq!(
            header.missing_tensors(&["nope.weight"]),
            vec!["nope.weight".to_string()]
        );
        assert!(header.size_mb() > 0.0);
    }

    #[test]
    fn resolves_directory_to_conventional_file() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors(dir.path(), "model.safetensors", &[("w", "F32", vec![2, 2])]);

        let header = SafetensorsHeader::read(dir.path()).unwrap();
        assert_eq!(header.parameter_count(), 4);
        assert!(header.path.ends_with("model.safetensors"));
    }

    #[test]
    fn rejects_missing_path() {
        let dir = tempfile::tempdir().unwrap();
        let err = SafetensorsHeader::read(dir.path().join("absent.safetensors")).unwrap_err();
        assert!(matches!(err, RecognitionError::ModelLoadError { .. }));
    }

    #[test]
    fn rejects_non_safetensors_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("junk.safetensors");
        std::fs::write(&path, b"this is definitely not a checkpoint").unwrap();

        let err = SafetensorsHeader::read(&path).unwrap_err();
        match err {
            RecognitionError::ModelLoadError { message, .. } => {
                assert!(message.contains("safetensors"), "unexpected: {message}");
            }
            other => panic!("expected ModelLoadError, got {other:?}"),
        }
    }

    #[test]
    fn rejects_offsets_past_end_of_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.safetensors");
        let header = serde_json::json!({
            "w": { "dtype": "F32", "shape": [1024], "data_offsets": [0, 4096] }
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        // Payload deliberately far too short for the declared range.
        file.write_all(&[0_u8; 16]).unwrap();
        drop(file);

        let err = SafetensorsHeader::read(&path).unwrap_err();
        match err {
            RecognitionError::ModelLoadError { message, .. } => {
                assert!(message.contains("exceeds"), "unexpected: {message}");
            }
            other => panic!("expected ModelLoadError, got {other:?}"),
        }
    }

    #[test]
    fn rejects_empty_tensor_map() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.safetensors");
        let header_bytes = b"{}".to_vec();
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        drop(file);

        assert!(SafetensorsHeader::read(&path).is_err());
    }
}
