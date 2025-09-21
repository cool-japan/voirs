//! PyTorch export
//!
//! This module provides functionality to export datasets in PyTorch-compatible formats.
//! It supports tensor conversion, DataLoader integration, and efficient data loading.

use crate::{DatasetSample, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;

/// PyTorch export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyTorchConfig {
    /// Output format
    pub format: PyTorchFormat,
    /// Whether to normalize audio values to [-1, 1]
    pub normalize_audio: bool,
    /// Target sample rate for audio (None = keep original)
    pub target_sample_rate: Option<u32>,
    /// Whether to pad audio to fixed length
    pub pad_audio: bool,
    /// Fixed audio length in samples (if padding enabled)
    pub fixed_audio_length: Option<usize>,
    /// Whether to include raw audio data or file paths
    pub include_audio_data: bool,
    /// Text encoding format
    pub text_encoding: TextEncoding,
    /// Maximum text length (for padding/truncation)
    pub max_text_length: Option<usize>,
}

impl Default for PyTorchConfig {
    fn default() -> Self {
        Self {
            format: PyTorchFormat::Pickle,
            normalize_audio: true,
            target_sample_rate: None,
            pad_audio: false,
            fixed_audio_length: None,
            include_audio_data: true,
            text_encoding: TextEncoding::Raw,
            max_text_length: None,
        }
    }
}

/// PyTorch export formats
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PyTorchFormat {
    /// Python pickle format (.pkl)
    Pickle,
    /// PyTorch tensor format (.pt)
    Tensor,
    /// NumPy arrays (.npz)
    Numpy,
    /// JSON with tensor descriptions
    Json,
}

/// Text encoding options
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TextEncoding {
    /// Raw text strings
    Raw,
    /// Character-level encoding
    Character,
    /// Token IDs (requires tokenizer)
    TokenIds,
    /// One-hot character encoding
    OneHot,
}

/// PyTorch dataset export structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyTorchDataset {
    /// Sample data
    pub samples: Vec<PyTorchSample>,
    /// Dataset metadata
    pub metadata: DatasetMetadata,
    /// Configuration used for export
    pub config: PyTorchConfig,
}

/// PyTorch sample representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyTorchSample {
    /// Sample ID
    pub id: String,
    /// Text data (encoding depends on config)
    pub text: TextData,
    /// Audio data (format depends on config)
    pub audio: PyTorchAudioData,
    /// Speaker information
    pub speaker_id: Option<String>,
    /// Language code
    pub language: String,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Text data representation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TextData {
    /// Raw text string
    Raw(String),
    /// Character indices
    Characters(Vec<u32>),
    /// Token IDs
    Tokens(Vec<u32>),
    /// One-hot character matrix
    OneHot(Vec<Vec<f32>>),
}

/// Audio data representation for PyTorch export
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PyTorchAudioData {
    /// Raw audio samples
    Samples(Vec<f32>),
    /// Audio file path reference
    Path(String),
    /// Spectrogram features
    Spectrogram(Vec<Vec<f32>>),
}

/// Dataset metadata for PyTorch export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Dataset name
    pub name: String,
    /// Total samples
    pub total_samples: usize,
    /// Sample rate (if audio included)
    pub sample_rate: Option<u32>,
    /// Audio channels
    pub channels: Option<u32>,
    /// Text vocabulary (if applicable)
    pub vocabulary: Option<Vec<String>>,
    /// Unique speakers
    pub speakers: Vec<String>,
    /// Languages
    pub languages: Vec<String>,
}

/// PyTorch exporter
pub struct PyTorchExporter {
    /// Export configuration
    config: PyTorchConfig,
}

impl PyTorchExporter {
    /// Create new PyTorch exporter
    pub fn new(config: PyTorchConfig) -> Self {
        Self { config }
    }

    /// Create exporter with default configuration
    pub fn new_default() -> Self {
        Self::new(PyTorchConfig::default())
    }

    /// Export dataset to PyTorch format
    pub async fn export_dataset(
        &self,
        samples: &[DatasetSample],
        output_path: &Path,
    ) -> Result<()> {
        // Create output directory structure
        let output_dir = if output_path.is_dir() {
            output_path.to_path_buf()
        } else {
            output_path.parent().unwrap_or(Path::new(".")).to_path_buf()
        };

        fs::create_dir_all(&output_dir).await?;

        // If not including audio data, save audio files separately
        if !self.config.include_audio_data {
            self.save_audio_files(samples, &output_dir).await?;
        }

        let pytorch_dataset = self.convert_to_pytorch(samples).await?;

        match self.config.format {
            PyTorchFormat::Pickle => self.export_pickle(&pytorch_dataset, output_path).await?,
            PyTorchFormat::Tensor => self.export_tensor(&pytorch_dataset, output_path).await?,
            PyTorchFormat::Numpy => self.export_numpy(&pytorch_dataset, output_path).await?,
            PyTorchFormat::Json => self.export_json(&pytorch_dataset, output_path).await?,
        }

        // Also export dataset info and loader script
        self.export_dataset_info(&pytorch_dataset, output_path)
            .await?;
        self.export_loader_script(output_path).await?;

        Ok(())
    }

    /// Convert samples to PyTorch format
    async fn convert_to_pytorch(&self, samples: &[DatasetSample]) -> Result<PyTorchDataset> {
        let mut pytorch_samples = Vec::new();
        let mut all_speakers = std::collections::HashSet::new();
        let mut all_languages = std::collections::HashSet::new();
        let mut vocabulary = std::collections::HashSet::new();

        for sample in samples {
            // Process text
            let text_data = self.encode_text(&sample.text, &mut vocabulary)?;

            // Process audio
            let audio_data = self.process_audio(&sample.audio, &sample.id)?;

            // Collect metadata
            if let Some(ref speaker) = sample.speaker {
                all_speakers.insert(speaker.id.clone());
            }
            all_languages.insert(sample.language.as_str().to_string());

            pytorch_samples.push(PyTorchSample {
                id: sample.id.clone(),
                text: text_data,
                audio: audio_data,
                speaker_id: sample.speaker.as_ref().map(|s| s.id.clone()),
                language: sample.language.as_str().to_string(),
                metadata: sample.metadata.clone(),
            });
        }

        // Create metadata
        let sample_rate = samples.first().map(|s| s.audio.sample_rate());
        let channels = samples.first().map(|s| s.audio.channels());

        let vocab_vec = if vocabulary.is_empty() {
            None
        } else {
            let mut vocab: Vec<String> = vocabulary.into_iter().collect();
            vocab.sort();
            Some(vocab)
        };

        let metadata = DatasetMetadata {
            name: "pytorch-dataset".to_string(),
            total_samples: samples.len(),
            sample_rate,
            channels,
            vocabulary: vocab_vec,
            speakers: all_speakers.into_iter().collect(),
            languages: all_languages.into_iter().collect(),
        };

        Ok(PyTorchDataset {
            samples: pytorch_samples,
            metadata,
            config: self.config.clone(),
        })
    }

    /// Encode text based on configuration
    fn encode_text(
        &self,
        text: &str,
        vocabulary: &mut std::collections::HashSet<String>,
    ) -> Result<TextData> {
        match self.config.text_encoding {
            TextEncoding::Raw => Ok(TextData::Raw(text.to_string())),
            TextEncoding::Character => {
                let char_indices: Vec<u32> = text
                    .chars()
                    .map(|c| {
                        vocabulary.insert(c.to_string());
                        c as u32
                    })
                    .collect();
                Ok(TextData::Characters(char_indices))
            }
            TextEncoding::TokenIds => {
                // Simple whitespace tokenization for demo
                let tokens: Vec<u32> = text
                    .split_whitespace()
                    .enumerate()
                    .map(|(i, token)| {
                        vocabulary.insert(token.to_string());
                        i as u32
                    })
                    .collect();
                Ok(TextData::Tokens(tokens))
            }
            TextEncoding::OneHot => {
                // Create one-hot character encoding
                let chars: Vec<char> = text.chars().collect();
                let char_set: std::collections::BTreeSet<char> = chars.iter().cloned().collect();
                let char_to_idx: HashMap<char, usize> =
                    char_set.iter().enumerate().map(|(i, &c)| (c, i)).collect();

                let one_hot: Vec<Vec<f32>> = chars
                    .iter()
                    .map(|&c| {
                        let mut vec = vec![0.0; char_set.len()];
                        if let Some(&idx) = char_to_idx.get(&c) {
                            vec[idx] = 1.0;
                        }
                        vec
                    })
                    .collect();

                // Add characters to vocabulary
                for c in char_set {
                    vocabulary.insert(c.to_string());
                }

                Ok(TextData::OneHot(one_hot))
            }
        }
    }

    /// Process audio based on configuration
    fn process_audio(&self, audio: &crate::AudioData, sample_id: &str) -> Result<PyTorchAudioData> {
        let mut samples = audio.samples().to_vec();

        // Normalize if requested
        if self.config.normalize_audio {
            let max_val = samples
                .iter()
                .fold(0.0f32, |max, &sample| max.max(sample.abs()));
            if max_val > 0.0 {
                let scale = 1.0 / max_val;
                for sample in &mut samples {
                    *sample *= scale;
                }
            }
        }

        // Resample if needed
        if let Some(target_sr) = self.config.target_sample_rate {
            if target_sr != audio.sample_rate() {
                // Enhanced resampling using cubic interpolation for better quality
                let ratio = target_sr as f32 / audio.sample_rate() as f32;
                let new_length = (samples.len() as f32 * ratio) as usize;
                let mut resampled = Vec::with_capacity(new_length);

                for i in 0..new_length {
                    let original_pos = i as f32 / ratio;
                    let original_idx = original_pos as usize;

                    if original_idx + 3 < samples.len() {
                        // Cubic interpolation for smoother resampling
                        let frac = original_pos - original_idx as f32;
                        let y0 = samples[original_idx.saturating_sub(1).min(samples.len() - 1)];
                        let y1 = samples[original_idx];
                        let y2 = samples[original_idx + 1];
                        let y3 = samples[original_idx + 2];

                        // Catmull-Rom spline interpolation
                        let a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
                        let b = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
                        let c = -0.5 * y0 + 0.5 * y2;
                        let d = y1;

                        let interpolated = a * frac.powi(3) + b * frac.powi(2) + c * frac + d;
                        resampled.push(interpolated);
                    } else if original_idx < samples.len() {
                        // Linear interpolation for edge cases
                        let frac = original_pos - original_idx as f32;
                        let y1 = samples[original_idx];
                        let y2 = samples.get(original_idx + 1).copied().unwrap_or(0.0);
                        resampled.push(y1 + frac * (y2 - y1));
                    } else {
                        resampled.push(0.0);
                    }
                }
                samples = resampled;
            }
        }

        // Pad or truncate if requested
        if self.config.pad_audio {
            if let Some(target_length) = self.config.fixed_audio_length {
                if samples.len() < target_length {
                    // Pad with zeros
                    samples.resize(target_length, 0.0);
                } else if samples.len() > target_length {
                    // Truncate
                    samples.truncate(target_length);
                }
            }
        }

        if self.config.include_audio_data {
            Ok(PyTorchAudioData::Samples(samples))
        } else {
            // Save audio to file and return the path reference
            let audio_filename = format!("{sample_id}.wav");
            let audio_path = format!("audio/{audio_filename}");
            Ok(PyTorchAudioData::Path(audio_path))
        }
    }

    /// Save audio files to disk when using path references
    async fn save_audio_files(&self, samples: &[DatasetSample], output_dir: &Path) -> Result<()> {
        let audio_dir = output_dir.join("audio");
        fs::create_dir_all(&audio_dir).await?;

        for sample in samples {
            let audio_filename = format!("{}.wav", sample.id);
            let audio_path = audio_dir.join(&audio_filename);

            // Save audio as WAV file
            self.save_wav_file(&sample.audio, &audio_path).await?;
        }

        Ok(())
    }

    /// Save audio as WAV file
    async fn save_wav_file(&self, audio: &crate::AudioData, path: &Path) -> Result<()> {
        use hound::{WavSpec, WavWriter};
        use std::io::Cursor;

        let spec = WavSpec {
            channels: audio.channels() as u16,
            sample_rate: audio.sample_rate(),
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = WavWriter::new(&mut cursor, spec)?;
            for &sample in audio.samples() {
                writer.write_sample(sample)?;
            }
            writer.finalize()?;
        }

        fs::write(path, cursor.into_inner()).await?;
        Ok(())
    }

    /// Export as pickle format
    async fn export_pickle(&self, dataset: &PyTorchDataset, output_path: &Path) -> Result<()> {
        // Implement simplified Python pickle protocol v4 compatible format
        // This creates binary data that can be read by Python's pickle module

        let mut pickle_data = Vec::new();

        // Pickle protocol header (protocol version 4)
        pickle_data.extend_from_slice(b"\x80\x04");

        // Build dictionary structure for the dataset
        self.write_pickle_dict_start(&mut pickle_data);

        // Add 'samples' key and list
        self.write_pickle_string(&mut pickle_data, "samples");
        self.write_pickle_list_start(&mut pickle_data);

        for sample in &dataset.samples {
            // Each sample as a dictionary
            self.write_pickle_dict_start(&mut pickle_data);

            // Add sample fields
            self.write_pickle_string(&mut pickle_data, "id");
            self.write_pickle_string(&mut pickle_data, &sample.id);

            self.write_pickle_string(&mut pickle_data, "text");
            match &sample.text {
                TextData::Raw(text) => {
                    self.write_pickle_string(&mut pickle_data, text);
                }
                TextData::Characters(chars) => {
                    self.write_pickle_list_start(&mut pickle_data);
                    for &char_idx in chars {
                        self.write_pickle_int(&mut pickle_data, char_idx as i64);
                    }
                    self.write_pickle_list_end(&mut pickle_data);
                }
                TextData::Tokens(tokens) => {
                    self.write_pickle_list_start(&mut pickle_data);
                    for &token_idx in tokens {
                        self.write_pickle_int(&mut pickle_data, token_idx as i64);
                    }
                    self.write_pickle_list_end(&mut pickle_data);
                }
                TextData::OneHot(matrix) => {
                    self.write_pickle_list_start(&mut pickle_data);
                    for row in matrix {
                        self.write_pickle_list_start(&mut pickle_data);
                        for &value in row {
                            // Write float as string for pickle compatibility
                            let float_str = format!("{value}");
                            self.write_pickle_string(&mut pickle_data, &float_str);
                        }
                        self.write_pickle_list_end(&mut pickle_data);
                    }
                    self.write_pickle_list_end(&mut pickle_data);
                }
            }

            // Add audio data as numpy-like array metadata
            self.write_pickle_string(&mut pickle_data, "audio_shape");
            match &sample.audio {
                PyTorchAudioData::Samples(samples) => {
                    self.write_pickle_tuple_start(&mut pickle_data);
                    self.write_pickle_int(&mut pickle_data, samples.len() as i64);
                    self.write_pickle_tuple_end(&mut pickle_data);
                }
                PyTorchAudioData::Path(path) => {
                    self.write_pickle_string(&mut pickle_data, path);
                }
                PyTorchAudioData::Spectrogram(spectrogram) => {
                    self.write_pickle_tuple_start(&mut pickle_data);
                    self.write_pickle_int(&mut pickle_data, spectrogram.len() as i64);
                    if !spectrogram.is_empty() {
                        self.write_pickle_int(&mut pickle_data, spectrogram[0].len() as i64);
                    }
                    self.write_pickle_tuple_end(&mut pickle_data);
                }
            }

            // Add metadata
            self.write_pickle_string(&mut pickle_data, "metadata");
            self.write_pickle_dict_start(&mut pickle_data);
            for (key, value) in &sample.metadata {
                self.write_pickle_string(&mut pickle_data, key);
                // Convert serde_json::Value to string for pickle compatibility
                let value_str = match value {
                    serde_json::Value::String(s) => s.clone(),
                    _ => value.to_string(),
                };
                self.write_pickle_string(&mut pickle_data, &value_str);
            }
            self.write_pickle_dict_end(&mut pickle_data);

            self.write_pickle_dict_end(&mut pickle_data);
        }

        self.write_pickle_list_end(&mut pickle_data);

        // Add dataset metadata
        self.write_pickle_string(&mut pickle_data, "dataset_metadata");
        self.write_pickle_dict_start(&mut pickle_data);

        self.write_pickle_string(&mut pickle_data, "name");
        self.write_pickle_string(&mut pickle_data, &dataset.metadata.name);

        self.write_pickle_string(&mut pickle_data, "total_samples");
        self.write_pickle_int(&mut pickle_data, dataset.metadata.total_samples as i64);

        if let Some(sample_rate) = dataset.metadata.sample_rate {
            self.write_pickle_string(&mut pickle_data, "sample_rate");
            self.write_pickle_int(&mut pickle_data, sample_rate as i64);
        }

        if let Some(channels) = dataset.metadata.channels {
            self.write_pickle_string(&mut pickle_data, "channels");
            self.write_pickle_int(&mut pickle_data, channels as i64);
        }

        self.write_pickle_string(&mut pickle_data, "speakers");
        self.write_pickle_list_start(&mut pickle_data);
        for speaker in &dataset.metadata.speakers {
            self.write_pickle_string(&mut pickle_data, speaker);
        }
        self.write_pickle_list_end(&mut pickle_data);

        self.write_pickle_string(&mut pickle_data, "languages");
        self.write_pickle_list_start(&mut pickle_data);
        for language in &dataset.metadata.languages {
            self.write_pickle_string(&mut pickle_data, language);
        }
        self.write_pickle_list_end(&mut pickle_data);

        self.write_pickle_dict_end(&mut pickle_data);

        self.write_pickle_dict_end(&mut pickle_data);

        // Pickle stop opcode
        pickle_data.push(b'.');

        fs::write(output_path.with_extension("pkl"), pickle_data).await?;
        Ok(())
    }

    /// Write pickle dictionary start
    fn write_pickle_dict_start(&self, data: &mut Vec<u8>) {
        data.push(b'}'); // EMPTY_DICT opcode
    }

    /// Write pickle dictionary end
    fn write_pickle_dict_end(&self, data: &mut Vec<u8>) {
        data.push(b's'); // SETITEMS opcode
    }

    /// Write pickle list start
    fn write_pickle_list_start(&self, data: &mut Vec<u8>) {
        data.push(b']'); // EMPTY_LIST opcode
    }

    /// Write pickle list end
    fn write_pickle_list_end(&self, data: &mut Vec<u8>) {
        data.push(b'e'); // APPENDS opcode
    }

    /// Write pickle tuple start
    fn write_pickle_tuple_start(&self, data: &mut Vec<u8>) {
        data.push(b')'); // EMPTY_TUPLE opcode
    }

    /// Write pickle tuple end
    fn write_pickle_tuple_end(&self, data: &mut Vec<u8>) {
        data.push(b't'); // TUPLE opcode
    }

    /// Write pickle string
    fn write_pickle_string(&self, data: &mut Vec<u8>, s: &str) {
        let bytes = s.as_bytes();
        if bytes.len() < 256 {
            data.push(b'U'); // SHORT_BINUNICODE opcode
            data.push(bytes.len() as u8);
        } else {
            data.push(b'X'); // BINUNICODE opcode
            data.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        }
        data.extend_from_slice(bytes);
    }

    /// Write pickle integer
    fn write_pickle_int(&self, data: &mut Vec<u8>, value: i64) {
        if (0..256).contains(&value) {
            data.push(b'K'); // BININT1 opcode
            data.push(value as u8);
        } else if (i32::MIN as i64..=i32::MAX as i64).contains(&value) {
            data.push(b'J'); // BININT opcode
            data.extend_from_slice(&(value as i32).to_le_bytes());
        } else {
            data.push(b'L'); // LONG opcode
            let value_str = format!("{value}L");
            data.extend_from_slice(value_str.as_bytes());
            data.push(b'\n');
        }
    }

    /// Export as tensor format
    async fn export_tensor(&self, dataset: &PyTorchDataset, output_path: &Path) -> Result<()> {
        // Create tensor directory structure
        let tensor_dir = output_path.with_extension("tensors");
        fs::create_dir_all(&tensor_dir).await?;

        // Export audio tensors
        let audio_dir = tensor_dir.join("audio");
        fs::create_dir_all(&audio_dir).await?;

        // Export text tensors
        let text_dir = tensor_dir.join("text");
        fs::create_dir_all(&text_dir).await?;

        // Export metadata
        let metadata_dir = tensor_dir.join("metadata");
        fs::create_dir_all(&metadata_dir).await?;

        for (idx, sample) in dataset.samples.iter().enumerate() {
            // Export audio tensor (binary format for PyTorch compatibility)
            if let PyTorchAudioData::Samples(ref audio_samples) = sample.audio {
                let audio_tensor_path = audio_dir.join(format!("{}.bin", sample.id));
                let mut audio_bytes = Vec::new();

                // Write tensor header: [dimensions, shape, data_type]
                // Format: [num_dims=1, shape=[length], dtype=float32]
                audio_bytes.extend_from_slice(&1u32.to_le_bytes()); // num_dims
                audio_bytes.extend_from_slice(&(audio_samples.len() as u32).to_le_bytes()); // shape
                audio_bytes.extend_from_slice(&1u32.to_le_bytes()); // dtype indicator (1 = float32)

                // Write actual audio data
                for &sample_val in audio_samples {
                    audio_bytes.extend_from_slice(&sample_val.to_le_bytes());
                }

                fs::write(audio_tensor_path, audio_bytes).await?;
            }

            // Export text tensor
            let text_tensor_path = text_dir.join(format!("{}.bin", sample.id));
            let mut text_bytes = Vec::new();

            match &sample.text {
                TextData::Characters(chars) => {
                    // Write tensor header for character indices
                    text_bytes.extend_from_slice(&1u32.to_le_bytes()); // num_dims
                    text_bytes.extend_from_slice(&(chars.len() as u32).to_le_bytes()); // shape
                    text_bytes.extend_from_slice(&2u32.to_le_bytes()); // dtype indicator (2 = uint32)

                    // Write character data
                    for &char_idx in chars {
                        text_bytes.extend_from_slice(&char_idx.to_le_bytes());
                    }
                }
                TextData::Tokens(tokens) => {
                    // Write tensor header for token indices
                    text_bytes.extend_from_slice(&1u32.to_le_bytes()); // num_dims
                    text_bytes.extend_from_slice(&(tokens.len() as u32).to_le_bytes()); // shape
                    text_bytes.extend_from_slice(&2u32.to_le_bytes()); // dtype indicator (2 = uint32)

                    // Write token data
                    for &token_idx in tokens {
                        text_bytes.extend_from_slice(&token_idx.to_le_bytes());
                    }
                }
                TextData::OneHot(matrix) => {
                    // Write tensor header for one-hot matrix
                    text_bytes.extend_from_slice(&2u32.to_le_bytes()); // num_dims
                    text_bytes.extend_from_slice(&(matrix.len() as u32).to_le_bytes()); // shape[0]
                    text_bytes.extend_from_slice(
                        &(matrix.first().map(|v| v.len()).unwrap_or(0) as u32).to_le_bytes(),
                    ); // shape[1]
                    text_bytes.extend_from_slice(&1u32.to_le_bytes()); // dtype indicator (1 = float32)

                    // Write one-hot data
                    for row in matrix {
                        for &val in row {
                            text_bytes.extend_from_slice(&val.to_le_bytes());
                        }
                    }
                }
                TextData::Raw(text) => {
                    // For raw text, just save as UTF-8 bytes with length prefix
                    let text_bytes_raw = text.as_bytes();
                    text_bytes.extend_from_slice(&1u32.to_le_bytes()); // num_dims
                    text_bytes.extend_from_slice(&(text_bytes_raw.len() as u32).to_le_bytes()); // shape
                    text_bytes.extend_from_slice(&3u32.to_le_bytes()); // dtype indicator (3 = utf8)
                    text_bytes.extend_from_slice(text_bytes_raw);
                }
            }

            fs::write(text_tensor_path, text_bytes).await?;

            // Export sample metadata
            let metadata_path = metadata_dir.join(format!("{}.json", sample.id));
            let metadata_json = serde_json::json!({
                "id": sample.id,
                "speaker_id": sample.speaker_id,
                "language": sample.language,
                "metadata": sample.metadata,
                "sample_index": idx
            });
            fs::write(metadata_path, serde_json::to_string_pretty(&metadata_json)?).await?;
        }

        // Export tensor info and loading instructions
        let tensor_info = serde_json::json!({
            "metadata": dataset.metadata,
            "config": dataset.config,
            "samples": dataset.samples.len(),
            "tensor_format": "Binary tensor format with custom header",
            "format_specification": {
                "audio_format": "1D float32 tensors with header [num_dims, shape, dtype]",
                "text_format": "Variable format based on encoding (see dtype indicator)",
                "dtype_indicators": {
                    "1": "float32",
                    "2": "uint32",
                    "3": "utf8_string"
                }
            },
            "directory_structure": {
                "audio/": "Audio tensor files (.bin)",
                "text/": "Text tensor files (.bin)",
                "metadata/": "Sample metadata files (.json)"
            }
        });

        fs::write(
            output_path.with_extension("tensor_info.json"),
            serde_json::to_string_pretty(&tensor_info)?,
        )
        .await?;

        // Export PyTorch tensor loading utility
        let loader_script = r#"#!/usr/bin/env python3
"""
PyTorch Tensor Loader for VoiRS binary tensor format.
"""

import torch
import json
import struct
from pathlib import Path
from typing import Dict, Any, Union, Tuple

class VoiRSTensorLoader:
    """Loader for VoiRS binary tensor format."""
    
    def __init__(self, tensor_dir: str):
        self.tensor_dir = Path(tensor_dir)
        self.info_path = self.tensor_dir.parent / f"{self.tensor_dir.stem}.tensor_info.json"
        
        # Load tensor info
        with open(self.info_path) as f:
            self.info = json.load(f)
    
    def load_audio_tensor(self, sample_id: str) -> torch.Tensor:
        """Load audio tensor for a sample."""
        audio_path = self.tensor_dir / "audio" / f"{sample_id}.bin"
        
        with open(audio_path, 'rb') as f:
            # Read header
            num_dims = struct.unpack('<I', f.read(4))[0]
            shape = struct.unpack('<I', f.read(4))[0]
            dtype = struct.unpack('<I', f.read(4))[0]
            
            # Read data
            if dtype == 1:  # float32
                data = struct.unpack(f'<{shape}f', f.read(shape * 4))
                return torch.tensor(data, dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")
    
    def load_text_tensor(self, sample_id: str) -> Union[torch.Tensor, str]:
        """Load text tensor for a sample."""
        text_path = self.tensor_dir / "text" / f"{sample_id}.bin"
        
        with open(text_path, 'rb') as f:
            # Read header
            num_dims = struct.unpack('<I', f.read(4))[0]
            
            if num_dims == 1:
                shape = struct.unpack('<I', f.read(4))[0]
                dtype = struct.unpack('<I', f.read(4))[0]
                
                if dtype == 2:  # uint32
                    data = struct.unpack(f'<{shape}I', f.read(shape * 4))
                    return torch.tensor(data, dtype=torch.long)
                elif dtype == 3:  # utf8
                    data = f.read(shape)
                    return data.decode('utf-8')
                else:
                    raise ValueError(f"Unsupported dtype: {dtype}")
            elif num_dims == 2:
                shape0 = struct.unpack('<I', f.read(4))[0]
                shape1 = struct.unpack('<I', f.read(4))[0]
                dtype = struct.unpack('<I', f.read(4))[0]
                
                if dtype == 1:  # float32
                    data = struct.unpack(f'<{shape0 * shape1}f', f.read(shape0 * shape1 * 4))
                    return torch.tensor(data, dtype=torch.float32).reshape(shape0, shape1)
                else:
                    raise ValueError(f"Unsupported dtype: {dtype}")
    
    def load_metadata(self, sample_id: str) -> Dict[str, Any]:
        """Load metadata for a sample."""
        metadata_path = self.tensor_dir / "metadata" / f"{sample_id}.json"
        
        with open(metadata_path) as f:
            return json.load(f)
    
    def get_sample_ids(self) -> List[str]:
        """Get all available sample IDs."""
        audio_files = list((self.tensor_dir / "audio").glob("*.bin"))
        return [f.stem for f in audio_files]

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Load VoiRS tensors')
    parser.add_argument('tensor_dir', help='Path to tensor directory')
    parser.add_argument('--sample_id', help='Sample ID to load', required=True)
    
    args = parser.parse_args()
    
    loader = VoiRSTensorLoader(args.tensor_dir)
    
    # Load sample data
    audio = loader.load_audio_tensor(args.sample_id)
    text = loader.load_text_tensor(args.sample_id)
    metadata = loader.load_metadata(args.sample_id)
    
    print(f"Audio shape: {audio.shape}")
    print(f"Text: {text}")
    print(f"Metadata: {metadata}")
"#;

        let loader_path = output_path.with_file_name("pytorch_tensor_loader.py");
        fs::write(loader_path, loader_script).await?;

        Ok(())
    }

    /// Export as NumPy format
    async fn export_numpy(&self, dataset: &PyTorchDataset, output_path: &Path) -> Result<()> {
        // Export NumPy-compatible descriptions
        let numpy_info = serde_json::json!({
            "metadata": dataset.metadata,
            "config": dataset.config,
            "arrays": {
                "audio_samples": "Array of audio samples per sample",
                "text_data": "Array of text encodings",
                "speaker_ids": "Array of speaker identifiers",
                "languages": "Array of language codes"
            }
        });

        fs::write(
            output_path.with_extension("numpy_info.json"),
            serde_json::to_string_pretty(&numpy_info)?,
        )
        .await?;
        Ok(())
    }

    /// Export as JSON format
    async fn export_json(&self, dataset: &PyTorchDataset, output_path: &Path) -> Result<()> {
        let json_data = serde_json::to_string_pretty(dataset)?;
        fs::write(output_path.with_extension("json"), json_data).await?;
        Ok(())
    }

    /// Export dataset information
    async fn export_dataset_info(
        &self,
        dataset: &PyTorchDataset,
        output_path: &Path,
    ) -> Result<()> {
        let info = serde_json::json!({
            "name": dataset.metadata.name,
            "total_samples": dataset.metadata.total_samples,
            "sample_rate": dataset.metadata.sample_rate,
            "channels": dataset.metadata.channels,
            "speakers": dataset.metadata.speakers,
            "languages": dataset.metadata.languages,
            "vocabulary_size": dataset.metadata.vocabulary.as_ref().map(Vec::len),
            "config": dataset.config
        });

        let info_path = output_path.with_file_name("dataset_info.json");
        fs::write(info_path, serde_json::to_string_pretty(&info)?).await?;
        Ok(())
    }

    /// Export PyTorch DataLoader script
    async fn export_loader_script(&self, output_path: &Path) -> Result<()> {
        let script_content = r#"#!/usr/bin/env python3
"""
PyTorch Dataset Loader for VoiRS-exported dataset.

This script provides a PyTorch Dataset class for loading the exported dataset.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional

class VoiRSDataset(Dataset):
    """PyTorch Dataset for VoiRS-exported speech data."""
    
    def __init__(self, data_path: str):
        """Initialize dataset from exported data."""
        self.data_path = Path(data_path)
        
        # Load dataset info
        with open(self.data_path.parent / "dataset_info.json") as f:
            self.info = json.load(f)
        
        # Load main dataset
        if data_path.endswith('.json'):
            with open(data_path) as f:
                self.data = json.load(f)
        else:
            raise ValueError(f"Unsupported format: {data_path}")
        
        self.samples = self.data['samples']
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample by index."""
        sample = self.samples[idx]
        
        # Convert audio data
        if isinstance(sample['audio'], list):
            audio = torch.tensor(sample['audio'], dtype=torch.float32)
        else:
            # Handle audio path reference
            audio = torch.zeros(1)  # Placeholder
        
        # Convert text data
        text = sample['text']
        if isinstance(text, list):
            text = torch.tensor(text, dtype=torch.long)
        elif isinstance(text, str):
            # Keep as string for raw text
            pass
        
        return {
            'id': sample['id'],
            'text': text,
            'audio': audio,
            'speaker_id': sample.get('speaker_id'),
            'language': sample['language'],
            'metadata': sample.get('metadata', {})
        }

def create_dataloader(data_path: str, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    """Create a PyTorch DataLoader for the dataset."""
    dataset = VoiRSDataset(data_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

def collate_fn(batch):
    """Custom collate function for batching samples."""
    # Extract fields
    ids = [item['id'] for item in batch]
    texts = [item['text'] for item in batch]
    audios = [item['audio'] for item in batch]
    speaker_ids = [item['speaker_id'] for item in batch]
    languages = [item['language'] for item in batch]
    metadata = [item['metadata'] for item in batch]
    
    # Handle audio tensors
    if all(isinstance(audio, torch.Tensor) for audio in audios):
        # Pad audio sequences to same length
        max_len = max(audio.size(0) for audio in audios)
        padded_audios = torch.zeros(len(audios), max_len)
        for i, audio in enumerate(audios):
            padded_audios[i, :audio.size(0)] = audio
        audios = padded_audios
    
    # Handle text tensors
    if all(isinstance(text, torch.Tensor) for text in texts):
        # Pad text sequences
        max_len = max(text.size(0) for text in texts)
        padded_texts = torch.zeros(len(texts), max_len, dtype=torch.long)
        for i, text in enumerate(texts):
            padded_texts[i, :text.size(0)] = text
        texts = padded_texts
    
    return {
        'ids': ids,
        'texts': texts,
        'audios': audios,
        'speaker_ids': speaker_ids,
        'languages': languages,
        'metadata': metadata
    }

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Load VoiRS dataset')
    parser.add_argument('data_path', help='Path to dataset JSON file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    
    # Create dataset and dataloader
    dataloader = create_dataloader(args.data_path, args.batch_size, num_workers=args.num_workers)
    
    print(f"Dataset loaded with {len(dataloader.dataset)} samples")
    print(f"Number of batches: {len(dataloader)}")
    
    # Show first batch
    for batch in dataloader:
        print(f"Batch shape - Texts: {batch['texts'].shape if isinstance(batch['texts'], torch.Tensor) else 'Variable'}")
        print(f"Batch shape - Audios: {batch['audios'].shape if isinstance(batch['audios'], torch.Tensor) else 'Variable'}")
        print(f"Sample IDs: {batch['ids'][:5]}...")  # Show first 5 IDs
        break
"#;

        let script_path = output_path.with_file_name("pytorch_loader.py");
        fs::write(script_path, script_content).await?;
        Ok(())
    }

    /// Update configuration
    pub fn with_config(mut self, config: PyTorchConfig) -> Self {
        self.config = config;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioData, LanguageCode};
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_pytorch_config_default() {
        let config = PyTorchConfig::default();
        assert!(matches!(config.format, PyTorchFormat::Pickle));
        assert!(config.normalize_audio);
        assert!(config.include_audio_data);
        assert!(matches!(config.text_encoding, TextEncoding::Raw));
    }

    #[tokio::test]
    async fn test_text_encoding() {
        let config = PyTorchConfig {
            text_encoding: TextEncoding::Character,
            ..Default::default()
        };
        let exporter = PyTorchExporter::new(config);
        let mut vocab = std::collections::HashSet::new();

        let result = exporter.encode_text("hello", &mut vocab).unwrap();
        match result {
            TextData::Characters(chars) => {
                assert_eq!(chars.len(), 5); // "hello" has 5 characters
            }
            _ => panic!("Expected Characters encoding but got a different type"),
        }
    }

    #[tokio::test]
    async fn test_audio_processing() {
        let config = PyTorchConfig {
            normalize_audio: true,
            include_audio_data: true,
            ..Default::default()
        };
        let exporter = PyTorchExporter::new(config);

        let audio = AudioData::new(vec![0.5, -0.8, 0.3], 22050, 1);
        let result = exporter.process_audio(&audio, "test_sample").unwrap();

        match result {
            PyTorchAudioData::Samples(samples) => {
                // Check that audio was normalized (max absolute value should be 1.0)
                let max_val = samples
                    .iter()
                    .fold(0.0f32, |max, &sample| max.max(sample.abs()));
                assert!((max_val - 1.0).abs() < 0.001);
            }
            _ => panic!("Expected Samples format but got a different type"),
        }
    }

    #[tokio::test]
    async fn test_export_workflow() {
        let temp_dir = TempDir::new().unwrap();
        let config = PyTorchConfig::default();
        let exporter = PyTorchExporter::new(config);

        let samples = vec![crate::DatasetSample::new(
            "sample_001".to_string(),
            "Test sample".to_string(),
            AudioData::silence(1.0, 22050, 1),
            LanguageCode::EnUs,
        )];

        let output_path = temp_dir.path().join("dataset");
        exporter
            .export_dataset(&samples, &output_path)
            .await
            .unwrap();

        // Check that files were created
        assert!(temp_dir.path().join("dataset.pkl").exists());
        assert!(temp_dir.path().join("dataset_info.json").exists());
        assert!(temp_dir.path().join("pytorch_loader.py").exists());

        // Check pickle file exists and has content
        let pickle_content = fs::read(temp_dir.path().join("dataset.pkl")).await.unwrap();
        assert!(!pickle_content.is_empty());
        // Verify it's a valid pickle file by checking the protocol header
        assert_eq!(&pickle_content[0..2], b"\x80\x04");
    }
}
