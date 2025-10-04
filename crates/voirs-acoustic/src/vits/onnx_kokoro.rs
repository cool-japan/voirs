//! Kokoro ONNX model loader - Multilingual TTS (Chinese, Japanese, English)
//!
//! Based on Kokoro-82M by hexgrad: https://huggingface.co/hexgrad/Kokoro-82M
//! Supports 54 voices across 8 languages with IPA phoneme input.

use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "onnx")]
use ort::{inputs, session::Session, value::Value};

use crate::{AcousticError, Result};

/// Kokoro ONNX inference model for multilingual TTS
#[cfg(feature = "onnx")]
pub struct KokoroOnnxInference {
    session: Session,
    vocab: HashMap<String, i64>,
    voices: Vec<f32>,
    voice_dim: usize,
    sample_rate: u32,
}

#[cfg(feature = "onnx")]
impl KokoroOnnxInference {
    /// Load Kokoro model from ONNX file
    ///
    /// # Arguments
    /// - `model_path`: Path to kokoro-v1.0.onnx or similar
    /// - `vocab`: Phoneme vocabulary (IPA → token ID mapping)
    /// - `voices`: Voice embeddings as flat f32 array
    /// - `voice_dim`: Dimension of each voice embedding (default: 256)
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        vocab: HashMap<String, i64>,
        voices: Vec<f32>,
        voice_dim: usize,
    ) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| {
                AcousticError::ModelError(format!("Failed to create session builder: {}", e))
            })?
            .commit_from_file(model_path)
            .map_err(|e| AcousticError::ModelError(format!("Failed to load ONNX model: {}", e)))?;

        Ok(Self {
            session,
            vocab,
            voices,
            voice_dim,
            sample_rate: 24000, // Kokoro uses 24kHz
        })
    }

    /// Load vocabulary from config.json
    pub fn load_vocab_from_json<P: AsRef<Path>>(config_path: P) -> Result<HashMap<String, i64>> {
        use std::fs::File;
        use std::io::BufReader;

        let file = File::open(config_path)
            .map_err(|e| AcousticError::ModelError(format!("Failed to open config.json: {}", e)))?;
        let reader = BufReader::new(file);

        let config: serde_json::Value = serde_json::from_reader(reader).map_err(|e| {
            AcousticError::ModelError(format!("Failed to parse config.json: {}", e))
        })?;

        let vocab_obj = config
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                AcousticError::ModelError("No 'vocab' field in config.json".to_string())
            })?;

        let mut vocab = HashMap::new();
        for (phoneme, id) in vocab_obj {
            let id_val = id.as_i64().ok_or_else(|| {
                AcousticError::ModelError(format!("Invalid ID for phoneme: {}", phoneme))
            })?;
            vocab.insert(phoneme.clone(), id_val);
        }

        Ok(vocab)
    }

    /// Load voice embeddings from voices.bin file (flat f32 binary)
    pub fn load_voices_from_bin<P: AsRef<Path>>(voices_path: P) -> Result<Vec<f32>> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(voices_path)
            .map_err(|e| AcousticError::ModelError(format!("Failed to open voices.bin: {}", e)))?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| AcousticError::ModelError(format!("Failed to read voices.bin: {}", e)))?;

        // Convert bytes to f32 (little-endian)
        let floats: Vec<f32> = buffer
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(floats)
    }

    /// Load voice embeddings from NPZ file (numpy archive format)
    ///
    /// This loads all voice arrays from a .npz file (e.g., voices-v1.0.bin),
    /// averages them over the first dimension if needed, and returns a flat Vec<f32>
    /// suitable for use with get_voice_embedding().
    ///
    /// # Arguments
    /// * `npz_path` - Path to the .npz file containing voice embeddings
    /// * `voice_dim` - Expected dimension of each voice embedding (typically 256)
    ///
    /// # Returns
    /// Flat vector of voice embeddings, concatenated in alphabetical order by voice name
    pub fn load_voices_from_npz<P: AsRef<Path>>(npz_path: P, voice_dim: usize) -> Result<Vec<f32>> {
        use numrs2::io::list_npz_arrays;
        use numrs2::io::load_npz_array;
        use std::fs::File;

        // List all arrays in the NPZ file
        let file = File::open(npz_path.as_ref())
            .map_err(|e| AcousticError::ModelError(format!("Failed to open NPZ file: {}", e)))?;

        let mut array_names = list_npz_arrays(file)
            .map_err(|e| AcousticError::ModelError(format!("Failed to list NPZ arrays: {}", e)))?;

        // Sort alphabetically to ensure consistent ordering
        array_names.sort();

        tracing::info!("Found {} voice arrays in NPZ file", array_names.len());

        // Load and flatten all voice arrays
        let mut all_voices = Vec::new();

        for voice_name in &array_names {
            let file = File::open(npz_path.as_ref()).map_err(|e| {
                AcousticError::ModelError(format!("Failed to open NPZ file: {}", e))
            })?;

            let voice_array = load_npz_array::<f32, _>(file, voice_name).map_err(|e| {
                AcousticError::ModelError(format!("Failed to load voice '{}': {}", voice_name, e))
            })?;

            let shape = voice_array.shape();
            let data = voice_array.to_vec();

            // Handle different shapes:
            // - (510, 1, 256): Average over first dimension to get (256,)
            // - (256,): Use directly
            let voice_embedding = if shape.len() == 3 {
                // Shape is (N, 1, voice_dim) - average over first dimension
                let n = shape[0];
                if shape[1] != 1 || shape[2] != voice_dim {
                    return Err(AcousticError::ModelError(format!(
                        "Voice '{}' has unexpected shape {:?}, expected (N, 1, {})",
                        voice_name, shape, voice_dim
                    )));
                }

                // Average over first dimension
                let mut averaged = vec![0.0f32; voice_dim];
                for i in 0..n {
                    for (j, avg_val) in averaged.iter_mut().enumerate().take(voice_dim) {
                        let idx = i * voice_dim + j; // shape is (N, 1, voice_dim) flattened
                        *avg_val += data[idx];
                    }
                }
                for val in &mut averaged {
                    *val /= n as f32;
                }
                averaged
            } else if shape.len() == 1 && shape[0] == voice_dim {
                // Already the right shape
                data
            } else {
                return Err(AcousticError::ModelError(format!(
                    "Voice '{}' has unsupported shape {:?}, expected (N, 1, {}) or ({})",
                    voice_name, shape, voice_dim, voice_dim
                )));
            };

            all_voices.extend(voice_embedding);
        }

        tracing::info!(
            "Loaded {} voices with {} total floats",
            array_names.len(),
            all_voices.len()
        );

        Ok(all_voices)
    }

    /// Convenience constructor that loads from standard Kokoro file structure
    ///
    /// Supports two voice file formats:
    /// 1. `voices_averaged.bin`: Flat binary file (preferred, created by Python conversion)
    /// 2. `voices-v1.0.bin`: NumPy .npz archive (original format, loaded via numrs2)
    pub fn from_kokoro_files<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let model_path = model_dir.join("kokoro-v1.0.onnx");
        let config_path = model_dir.join("config.json");

        let vocab = Self::load_vocab_from_json(&config_path)?;

        // Try to load voices from different formats in order of preference
        let voices = {
            // 1. Try voices_averaged.bin (flat binary, fastest)
            let averaged_bin = model_dir.join("voices_averaged.bin");
            if averaged_bin.exists() {
                tracing::info!("Loading voices from voices_averaged.bin");
                Self::load_voices_from_bin(&averaged_bin)?
            }
            // 2. Try voices-v1.0.bin (NPZ format, requires averaging)
            else {
                let npz_path = model_dir.join("voices-v1.0.bin");
                if npz_path.exists() {
                    tracing::info!("Loading voices from voices-v1.0.bin (NPZ format)");
                    Self::load_voices_from_npz(&npz_path, 256)?
                } else {
                    return Err(AcousticError::ModelError(
                        "No voice file found. Expected voices_averaged.bin or voices-v1.0.bin"
                            .to_string(),
                    ));
                }
            }
        };

        Self::new(model_path, vocab, voices, 256)
    }

    /// Tokenize IPA phonemes to token IDs
    pub fn tokenize(&self, phonemes: &str) -> Result<Vec<i64>> {
        let mut tokens = Vec::new();

        // Split into characters (handle multi-byte UTF-8)
        let mut chars = phonemes.chars().peekable();

        while let Some(ch) = chars.next() {
            let ch_str = ch.to_string();

            // Try to match longer sequences first (e.g., "ː" stress marks)
            let mut matched = false;

            // Check for two-character sequences
            if let Some(&next_ch) = chars.peek() {
                let two_char = format!("{}{}", ch, next_ch);
                if let Some(&token_id) = self.vocab.get(&two_char) {
                    tokens.push(token_id);
                    chars.next(); // Consume the peeked character
                    matched = true;
                }
            }

            // Fall back to single character
            if !matched {
                if let Some(&token_id) = self.vocab.get(&ch_str) {
                    tokens.push(token_id);
                } else {
                    eprintln!("⚠️  Unknown phoneme: '{}' (U+{:04X})", ch, ch as u32);
                    // Use space token as fallback
                    if let Some(&space_token) = self.vocab.get(" ") {
                        tokens.push(space_token);
                    }
                }
            }
        }

        Ok(tokens)
    }

    /// Get voice embedding by index
    pub fn get_voice_embedding(&self, voice_idx: usize) -> Result<Vec<f32>> {
        let start_idx = voice_idx * self.voice_dim;
        let end_idx = start_idx + self.voice_dim;

        if end_idx > self.voices.len() {
            return Err(AcousticError::ModelError(format!(
                "Voice index {} out of bounds (max: {})",
                voice_idx,
                self.voices.len() / self.voice_dim
            )));
        }

        Ok(self.voices[start_idx..end_idx].to_vec())
    }

    /// Synthesize audio from IPA phonemes
    ///
    /// # Arguments
    /// - `phonemes`: IPA phoneme string (e.g., "nɪˈhaʊ")
    /// - `voice_idx`: Voice index (0-53 for v1.0)
    /// - `speed`: Speech speed (default: 1.0)
    ///
    /// # Returns
    /// - Audio samples as Vec<f32> at 24kHz sample rate
    pub fn synthesize(&mut self, phonemes: &str, voice_idx: usize, speed: f32) -> Result<Vec<f32>> {
        self.synthesize_with_options(phonemes, voice_idx, speed, true)
    }

    /// Synthesize audio with options
    ///
    /// # Arguments
    /// - `phonemes`: IPA phoneme string
    /// - `voice_idx`: Voice index (0-53 for v1.0)
    /// - `speed`: Speech speed (default: 1.0)
    /// - `trim_silence`: Whether to trim leading/trailing silence
    pub fn synthesize_with_options(
        &mut self,
        phonemes: &str,
        voice_idx: usize,
        speed: f32,
        trim_silence: bool,
    ) -> Result<Vec<f32>> {
        // Tokenize phonemes
        let token_ids = self.tokenize(phonemes)?;

        // Get voice embedding
        let voice_embedding = self.get_voice_embedding(voice_idx)?;

        // Create input tensors
        let tokens_shape = vec![1, token_ids.len()];
        let tokens_tensor = Value::from_array((tokens_shape, token_ids)).map_err(|e| {
            AcousticError::ModelError(format!("Failed to create tokens tensor: {}", e))
        })?;

        let voice_shape = vec![1, self.voice_dim];
        let voice_tensor = Value::from_array((voice_shape, voice_embedding)).map_err(|e| {
            AcousticError::ModelError(format!("Failed to create voice tensor: {}", e))
        })?;

        let speed_tensor = Value::from_array((vec![1], vec![speed])).map_err(|e| {
            AcousticError::ModelError(format!("Failed to create speed tensor: {}", e))
        })?;

        // Run inference
        let inputs_vec = inputs![
            "tokens" => tokens_tensor,
            "style" => voice_tensor,
            "speed" => speed_tensor
        ];

        let outputs = self
            .session
            .run(inputs_vec)
            .map_err(|e| AcousticError::ModelError(format!("Inference failed: {}", e)))?;

        // Extract audio output (first output)
        let audio_tensor = outputs
            .iter()
            .next()
            .ok_or_else(|| AcousticError::ModelError("No output from model".to_string()))?
            .1;

        // Convert to Vec<f32>
        let (_, audio_slice) = audio_tensor
            .try_extract_tensor::<f32>()
            .map_err(|e| AcousticError::ModelError(format!("Failed to extract audio: {}", e)))?;

        let mut audio_data: Vec<f32> = audio_slice.to_vec();

        // Trim silence if requested
        if trim_silence {
            audio_data = Self::trim_silence(&audio_data, 0.005);
        }

        Ok(audio_data)
    }

    /// Synthesize audio with trailing silence trimming only
    ///
    /// This is useful when you want to keep the natural beginning but remove
    /// excessive silence at the end.
    pub fn synthesize_trim_end(
        &mut self,
        phonemes: &str,
        voice_idx: usize,
        speed: f32,
    ) -> Result<Vec<f32>> {
        // Tokenize phonemes
        let token_ids = self.tokenize(phonemes)?;

        // Get voice embedding
        let voice_embedding = self.get_voice_embedding(voice_idx)?;

        // Create input tensors
        let tokens_shape = vec![1, token_ids.len()];
        let tokens_tensor = Value::from_array((tokens_shape, token_ids)).map_err(|e| {
            AcousticError::ModelError(format!("Failed to create tokens tensor: {}", e))
        })?;

        let voice_shape = vec![1, self.voice_dim];
        let voice_tensor = Value::from_array((voice_shape, voice_embedding)).map_err(|e| {
            AcousticError::ModelError(format!("Failed to create voice tensor: {}", e))
        })?;

        let speed_tensor = Value::from_array((vec![1], vec![speed])).map_err(|e| {
            AcousticError::ModelError(format!("Failed to create speed tensor: {}", e))
        })?;

        // Run inference
        let inputs_vec = inputs![
            "tokens" => tokens_tensor,
            "style" => voice_tensor,
            "speed" => speed_tensor
        ];

        let outputs = self
            .session
            .run(inputs_vec)
            .map_err(|e| AcousticError::ModelError(format!("Inference failed: {}", e)))?;

        // Extract audio output (first output)
        let audio_tensor = outputs
            .iter()
            .next()
            .ok_or_else(|| AcousticError::ModelError("No output from model".to_string()))?
            .1;

        // Convert to Vec<f32>
        let (_, audio_slice) = audio_tensor
            .try_extract_tensor::<f32>()
            .map_err(|e| AcousticError::ModelError(format!("Failed to extract audio: {}", e)))?;

        let audio_data: Vec<f32> = audio_slice.to_vec();

        // Trim only trailing silence
        let trimmed = Self::trim_trailing_silence(&audio_data, 0.01);

        Ok(trimmed)
    }

    /// Trim leading and trailing silence from audio
    fn trim_silence(audio: &[f32], threshold: f32) -> Vec<f32> {
        // Find first sample above threshold
        let start = audio.iter().position(|&s| s.abs() > threshold).unwrap_or(0);

        // Find last sample above threshold
        let end = audio
            .iter()
            .rposition(|&s| s.abs() > threshold)
            .unwrap_or(audio.len() - 1)
            + 1;

        if start < end {
            audio[start..end].to_vec()
        } else {
            audio.to_vec()
        }
    }

    /// Trim only trailing silence (keep leading silence)
    fn trim_trailing_silence(audio: &[f32], threshold: f32) -> Vec<f32> {
        // Find last sample above threshold
        let end = audio
            .iter()
            .rposition(|&s| s.abs() > threshold)
            .unwrap_or(audio.len() - 1);

        // Keep 20ms (480 samples at 24kHz) after last significant sample
        let end_with_tail = (end + 480).min(audio.len());

        audio[..end_with_tail].to_vec()
    }

    /// Get sample rate (24000 Hz for Kokoro)
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

#[cfg(not(feature = "onnx"))]
pub struct KokoroOnnxInference;

#[cfg(not(feature = "onnx"))]
impl KokoroOnnxInference {
    pub fn from_kokoro_files<P: AsRef<Path>>(_model_dir: P) -> Result<Self> {
        Err(AcousticError::ModelError(
            "ONNX feature not enabled. Enable with --features onnx".to_string(),
        ))
    }

    pub fn synthesize(&self, _phonemes: &str, _voice_idx: usize, _speed: f32) -> Result<Vec<f32>> {
        Err(AcousticError::ModelError(
            "ONNX feature not enabled".to_string(),
        ))
    }

    pub fn sample_rate(&self) -> u32 {
        24000
    }
}
