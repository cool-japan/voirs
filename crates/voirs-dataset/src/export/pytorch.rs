//! PyTorch export
//!
//! This module provides functionality to export datasets in PyTorch-compatible formats.
//! It supports tensor conversion, DataLoader integration, and efficient data loading.

use crate::{DatasetSample, DatasetError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
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
    pub fn default() -> Self {
        Self::new(PyTorchConfig::default())
    }

    /// Export dataset to PyTorch format
    pub async fn export_dataset(
        &self,
        samples: &[DatasetSample],
        output_path: &Path,
    ) -> Result<()> {
        let pytorch_dataset = self.convert_to_pytorch(samples).await?;

        match self.config.format {
            PyTorchFormat::Pickle => self.export_pickle(&pytorch_dataset, output_path).await?,
            PyTorchFormat::Tensor => self.export_tensor(&pytorch_dataset, output_path).await?,
            PyTorchFormat::Numpy => self.export_numpy(&pytorch_dataset, output_path).await?,
            PyTorchFormat::Json => self.export_json(&pytorch_dataset, output_path).await?,
        }

        // Also export dataset info and loader script
        self.export_dataset_info(&pytorch_dataset, output_path).await?;
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
            let audio_data = self.process_audio(&sample.audio)?;

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
    fn encode_text(&self, text: &str, vocabulary: &mut std::collections::HashSet<String>) -> Result<TextData> {
        match self.config.text_encoding {
            TextEncoding::Raw => Ok(TextData::Raw(text.to_string())),
            TextEncoding::Character => {
                let char_indices: Vec<u32> = text.chars()
                    .map(|c| {
                        vocabulary.insert(c.to_string());
                        c as u32
                    })
                    .collect();
                Ok(TextData::Characters(char_indices))
            },
            TextEncoding::TokenIds => {
                // Simple whitespace tokenization for demo
                let tokens: Vec<u32> = text.split_whitespace()
                    .enumerate()
                    .map(|(i, token)| {
                        vocabulary.insert(token.to_string());
                        i as u32
                    })
                    .collect();
                Ok(TextData::Tokens(tokens))
            },
            TextEncoding::OneHot => {
                // Create one-hot character encoding
                let chars: Vec<char> = text.chars().collect();
                let char_set: std::collections::BTreeSet<char> = chars.iter().cloned().collect();
                let char_to_idx: HashMap<char, usize> = char_set.iter()
                    .enumerate()
                    .map(|(i, &c)| (c, i))
                    .collect();

                let one_hot: Vec<Vec<f32>> = chars.iter()
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
            },
        }
    }

    /// Process audio based on configuration
    fn process_audio(&self, audio: &crate::AudioData) -> Result<PyTorchAudioData> {
        let mut samples = audio.samples().to_vec();

        // Normalize if requested
        if self.config.normalize_audio {
            let max_val = samples.iter().fold(0.0f32, |max, &sample| max.max(sample.abs()));
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
                // Simple resampling (linear interpolation)
                let ratio = target_sr as f32 / audio.sample_rate() as f32;
                let new_length = (samples.len() as f32 * ratio) as usize;
                let mut resampled = Vec::with_capacity(new_length);

                for i in 0..new_length {
                    let original_idx = (i as f32 / ratio) as usize;
                    if original_idx < samples.len() {
                        resampled.push(samples[original_idx]);
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
            // Return placeholder path - in real implementation, save to file
            Ok(PyTorchAudioData::Path("audio/placeholder.wav".to_string()))
        }
    }

    /// Export as pickle format
    async fn export_pickle(&self, dataset: &PyTorchDataset, output_path: &Path) -> Result<()> {
        // Export as JSON for now (Python pickle requires Python)
        let json_data = serde_json::to_string_pretty(dataset)?;
        fs::write(output_path.with_extension("json"), json_data).await?;
        Ok(())
    }

    /// Export as tensor format
    async fn export_tensor(&self, dataset: &PyTorchDataset, output_path: &Path) -> Result<()> {
        // Export tensor descriptions as JSON
        let tensor_info = serde_json::json!({
            "metadata": dataset.metadata,
            "config": dataset.config,
            "samples": dataset.samples.len(),
            "tensor_format": "Individual tensors per sample"
        });

        fs::write(output_path.with_extension("tensor_info.json"), serde_json::to_string_pretty(&tensor_info)?).await?;
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

        fs::write(output_path.with_extension("numpy_info.json"), serde_json::to_string_pretty(&numpy_info)?).await?;
        Ok(())
    }

    /// Export as JSON format
    async fn export_json(&self, dataset: &PyTorchDataset, output_path: &Path) -> Result<()> {
        let json_data = serde_json::to_string_pretty(dataset)?;
        fs::write(output_path.with_extension("json"), json_data).await?;
        Ok(())
    }

    /// Export dataset information
    async fn export_dataset_info(&self, dataset: &PyTorchDataset, output_path: &Path) -> Result<()> {
        let info = serde_json::json!({
            "name": dataset.metadata.name,
            "total_samples": dataset.metadata.total_samples,
            "sample_rate": dataset.metadata.sample_rate,
            "channels": dataset.metadata.channels,
            "speakers": dataset.metadata.speakers,
            "languages": dataset.metadata.languages,
            "vocabulary_size": dataset.metadata.vocabulary.as_ref().map(|v| v.len()),
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
    use crate::{AudioData, LanguageCode, SpeakerInfo};
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
            },
            _ => panic!("Expected Characters encoding"),
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
        let result = exporter.process_audio(&audio).unwrap();

        match result {
            PyTorchAudioData::Samples(samples) => {
                // Check that audio was normalized (max absolute value should be 1.0)
                let max_val = samples.iter().fold(0.0f32, |max, &sample| max.max(sample.abs()));
                assert!((max_val - 1.0).abs() < 0.001);
            },
            _ => panic!("Expected Samples format"),
        }
    }

    #[tokio::test]
    async fn test_export_workflow() {
        let temp_dir = TempDir::new().unwrap();
        let config = PyTorchConfig::default();
        let exporter = PyTorchExporter::new(config);

        let samples = vec![
            crate::DatasetSample::new(
                "sample_001".to_string(),
                "Test sample".to_string(),
                AudioData::silence(1.0, 22050, 1),
                LanguageCode::EnUs,
            ),
        ];

        let output_path = temp_dir.path().join("dataset");
        exporter.export_dataset(&samples, &output_path).await.unwrap();

        // Check that files were created
        assert!(temp_dir.path().join("dataset.json").exists());
        assert!(temp_dir.path().join("dataset_info.json").exists());
        assert!(temp_dir.path().join("pytorch_loader.py").exists());

        // Check JSON content
        let json_content = fs::read_to_string(temp_dir.path().join("dataset.json")).await.unwrap();
        assert!(json_content.contains("sample_001"));
        assert!(json_content.contains("Test sample"));
    }
}
