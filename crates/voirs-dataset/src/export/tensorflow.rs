//! TensorFlow export
//!
//! This module provides functionality to export datasets in TensorFlow-compatible formats.
//! It supports tf.data.Dataset creation, TFRecord format export, and pipeline optimization.

use crate::{DatasetSample, DatasetError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;

/// TensorFlow export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorFlowConfig {
    /// Output format
    pub format: TensorFlowFormat,
    /// Whether to create TFRecord files
    pub create_tfrecords: bool,
    /// Whether to normalize audio values to [-1, 1]
    pub normalize_audio: bool,
    /// Target sample rate for audio (None = keep original)
    pub target_sample_rate: Option<u32>,
    /// Whether to pad sequences to fixed length
    pub pad_sequences: bool,
    /// Fixed audio length in samples (if padding enabled)
    pub fixed_audio_length: Option<usize>,
    /// Fixed text length in characters/tokens (if padding enabled)
    pub fixed_text_length: Option<usize>,
    /// Text encoding format
    pub text_encoding: TextEncoding,
    /// Whether to include raw audio data or file paths
    pub include_audio_data: bool,
    /// Compression type for TFRecords
    pub compression_type: CompressionType,
}

impl Default for TensorFlowConfig {
    fn default() -> Self {
        Self {
            format: TensorFlowFormat::TfData,
            create_tfrecords: true,
            normalize_audio: true,
            target_sample_rate: None,
            pad_sequences: false,
            fixed_audio_length: None,
            fixed_text_length: None,
            text_encoding: TextEncoding::Utf8,
            include_audio_data: true,
            compression_type: CompressionType::Gzip,
        }
    }
}

/// TensorFlow export formats
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TensorFlowFormat {
    /// tf.data.Dataset with JSON metadata
    TfData,
    /// TFRecord format
    TfRecord,
    /// SavedModel format
    SavedModel,
    /// JSON with tensor specifications
    Json,
}

/// Text encoding options for TensorFlow
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TextEncoding {
    /// UTF-8 raw text
    Utf8,
    /// Byte-level encoding
    Bytes,
    /// Character indices
    Characters,
    /// Subword tokens (BPE-style)
    Subwords,
}

/// Compression types for TFRecord
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CompressionType {
    /// No compression
    None,
    /// GZIP compression
    Gzip,
    /// ZLIB compression
    Zlib,
}

/// TensorFlow dataset specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorFlowDataset {
    /// Dataset samples in TF format
    pub samples: Vec<TfSample>,
    /// Feature specifications
    pub feature_spec: FeatureSpec,
    /// Dataset metadata
    pub metadata: TfDatasetMetadata,
    /// Configuration used for export
    pub config: TensorFlowConfig,
}

/// TensorFlow sample representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TfSample {
    /// Sample features
    pub features: HashMap<String, TfFeature>,
}

/// TensorFlow feature value
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TfFeature {
    /// Bytes feature
    Bytes(Vec<u8>),
    /// Float list feature
    FloatList(Vec<f32>),
    /// Int64 list feature
    Int64List(Vec<i64>),
    /// String feature
    String(String),
}

/// Feature specification for TensorFlow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSpec {
    /// Audio feature specification
    pub audio: TfFeatureSpec,
    /// Text feature specification
    pub text: TfFeatureSpec,
    /// Speaker ID feature specification
    pub speaker_id: TfFeatureSpec,
    /// Language feature specification
    pub language: TfFeatureSpec,
    /// Additional features
    pub additional: HashMap<String, TfFeatureSpec>,
}

/// Individual feature specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TfFeatureSpec {
    /// Feature type
    pub dtype: String,
    /// Feature shape (None for variable length)
    pub shape: Option<Vec<Option<usize>>>,
    /// Default value
    pub default_value: Option<serde_json::Value>,
    /// Description
    pub description: String,
}

/// TensorFlow dataset metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TfDatasetMetadata {
    /// Dataset name
    pub name: String,
    /// Total samples
    pub total_samples: usize,
    /// Sample rate (if audio included)
    pub sample_rate: Option<u32>,
    /// Audio channels
    pub channels: Option<u32>,
    /// Text vocabulary size
    pub vocab_size: Option<usize>,
    /// Unique speakers
    pub speakers: Vec<String>,
    /// Languages
    pub languages: Vec<String>,
    /// Splits information
    pub splits: HashMap<String, SplitInfo>,
}

/// Split information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitInfo {
    /// Number of examples
    pub num_examples: usize,
    /// File shards
    pub shards: Vec<String>,
}

/// TensorFlow exporter
pub struct TensorFlowExporter {
    /// Export configuration
    config: TensorFlowConfig,
}

impl TensorFlowExporter {
    /// Create new TensorFlow exporter
    pub fn new(config: TensorFlowConfig) -> Self {
        Self { config }
    }

    /// Create exporter with default configuration
    pub fn default() -> Self {
        Self::new(TensorFlowConfig::default())
    }

    /// Export dataset to TensorFlow format
    pub async fn export_dataset(
        &self,
        samples: &[DatasetSample],
        output_dir: &Path,
    ) -> Result<()> {
        // Create output directory
        fs::create_dir_all(output_dir).await?;

        let tf_dataset = self.convert_to_tensorflow(samples).await?;

        match self.config.format {
            TensorFlowFormat::TfData => self.export_tfdata(&tf_dataset, output_dir).await?,
            TensorFlowFormat::TfRecord => self.export_tfrecord(&tf_dataset, output_dir).await?,
            TensorFlowFormat::SavedModel => self.export_savedmodel(&tf_dataset, output_dir).await?,
            TensorFlowFormat::Json => self.export_json(&tf_dataset, output_dir).await?,
        }

        // Export dataset info and loader script
        self.export_dataset_info(&tf_dataset, output_dir).await?;
        self.export_loader_script(output_dir).await?;

        Ok(())
    }

    /// Convert samples to TensorFlow format
    async fn convert_to_tensorflow(&self, samples: &[DatasetSample]) -> Result<TensorFlowDataset> {
        let mut tf_samples = Vec::new();
        let mut all_speakers = std::collections::HashSet::new();
        let mut all_languages = std::collections::HashSet::new();
        let mut vocab_size = 0;

        for sample in samples {
            let mut features = HashMap::new();

            // Process audio
            let audio_feature = self.process_audio_feature(&sample.audio)?;
            features.insert("audio".to_string(), audio_feature);

            // Process text
            let (text_feature, text_vocab_size) = self.process_text_feature(&sample.text)?;
            features.insert("text".to_string(), text_feature);
            vocab_size = vocab_size.max(text_vocab_size);

            // Add speaker ID
            if let Some(ref speaker) = sample.speaker {
                features.insert("speaker_id".to_string(), TfFeature::String(speaker.id.clone()));
                all_speakers.insert(speaker.id.clone());
            } else {
                features.insert("speaker_id".to_string(), TfFeature::String("unknown".to_string()));
            }

            // Add language
            let language = sample.language.as_str();
            features.insert("language".to_string(), TfFeature::String(language.to_string()));
            all_languages.insert(language.to_string());

            // Add sample ID
            features.insert("id".to_string(), TfFeature::String(sample.id.clone()));

            // Add metadata as JSON string
            if !sample.metadata.is_empty() {
                let metadata_json = serde_json::to_string(&sample.metadata)?;
                features.insert("metadata".to_string(), TfFeature::String(metadata_json));
            }

            tf_samples.push(TfSample { features });
        }

        // Create feature specification
        let feature_spec = FeatureSpec {
            audio: TfFeatureSpec {
                dtype: "float32".to_string(),
                shape: if self.config.fixed_audio_length.is_some() {
                    Some(vec![self.config.fixed_audio_length])
                } else {
                    Some(vec![None])
                },
                default_value: None,
                description: "Audio waveform samples".to_string(),
            },
            text: TfFeatureSpec {
                dtype: match self.config.text_encoding {
                    TextEncoding::Utf8 => "string".to_string(),
                    TextEncoding::Bytes => "uint8".to_string(),
                    TextEncoding::Characters | TextEncoding::Subwords => "int64".to_string(),
                },
                shape: if self.config.fixed_text_length.is_some() {
                    Some(vec![self.config.fixed_text_length])
                } else {
                    Some(vec![None])
                },
                default_value: None,
                description: "Text transcription".to_string(),
            },
            speaker_id: TfFeatureSpec {
                dtype: "string".to_string(),
                shape: Some(vec![]),
                default_value: Some(serde_json::Value::String("unknown".to_string())),
                description: "Speaker identifier".to_string(),
            },
            language: TfFeatureSpec {
                dtype: "string".to_string(),
                shape: Some(vec![]),
                default_value: None,
                description: "Language code".to_string(),
            },
            additional: HashMap::new(),
        };

        // Create metadata
        let sample_rate = samples.first().map(|s| s.audio.sample_rate());
        let channels = samples.first().map(|s| s.audio.channels());

        let mut splits = HashMap::new();
        splits.insert("train".to_string(), SplitInfo {
            num_examples: samples.len(),
            shards: vec!["train-00000-of-00001.tfrecord".to_string()],
        });

        let metadata = TfDatasetMetadata {
            name: "tensorflow-dataset".to_string(),
            total_samples: samples.len(),
            sample_rate,
            channels,
            vocab_size: if vocab_size > 0 { Some(vocab_size) } else { None },
            speakers: all_speakers.into_iter().collect(),
            languages: all_languages.into_iter().collect(),
            splits,
        };

        Ok(TensorFlowDataset {
            samples: tf_samples,
            feature_spec,
            metadata,
            config: self.config.clone(),
        })
    }

    /// Process audio feature
    fn process_audio_feature(&self, audio: &crate::AudioData) -> Result<TfFeature> {
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
                // Simple resampling
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
        if self.config.pad_sequences {
            if let Some(target_length) = self.config.fixed_audio_length {
                if samples.len() < target_length {
                    samples.resize(target_length, 0.0);
                } else if samples.len() > target_length {
                    samples.truncate(target_length);
                }
            }
        }

        Ok(TfFeature::FloatList(samples))
    }

    /// Process text feature
    fn process_text_feature(&self, text: &str) -> Result<(TfFeature, usize)> {
        match self.config.text_encoding {
            TextEncoding::Utf8 => {
                Ok((TfFeature::String(text.to_string()), 0))
            },
            TextEncoding::Bytes => {
                let bytes = text.as_bytes().to_vec();
                Ok((TfFeature::Bytes(bytes), 256)) // ASCII/UTF-8 byte range
            },
            TextEncoding::Characters => {
                let char_indices: Vec<i64> = text.chars()
                    .map(|c| c as u32 as i64)
                    .collect();
                let max_char = char_indices.iter().max().copied().unwrap_or(0) as usize;
                Ok((TfFeature::Int64List(char_indices), max_char + 1))
            },
            TextEncoding::Subwords => {
                // Simple whitespace tokenization for demo
                let tokens: Vec<i64> = text.split_whitespace()
                    .enumerate()
                    .map(|(i, _)| i as i64)
                    .collect();
                let vocab_size = text.split_whitespace().count();
                Ok((TfFeature::Int64List(tokens), vocab_size))
            },
        }
    }

    /// Export as tf.data format
    async fn export_tfdata(&self, dataset: &TensorFlowDataset, output_dir: &Path) -> Result<()> {
        // Export as JSON which can be loaded by tf.data
        let json_data = serde_json::to_string_pretty(dataset)?;
        fs::write(output_dir.join("dataset.json"), json_data).await?;

        // Export feature spec separately
        let spec_json = serde_json::to_string_pretty(&dataset.feature_spec)?;
        fs::write(output_dir.join("feature_spec.json"), spec_json).await?;

        Ok(())
    }

    /// Export as TFRecord format
    async fn export_tfrecord(&self, dataset: &TensorFlowDataset, output_dir: &Path) -> Result<()> {
        // For now, export TFRecord descriptions as JSON
        // In a real implementation, this would create actual TFRecord files
        let tfrecord_info = serde_json::json!({
            "format": "TFRecord",
            "compression": format!("{:?}", self.config.compression_type),
            "feature_spec": dataset.feature_spec,
            "metadata": dataset.metadata,
            "total_examples": dataset.samples.len(),
            "shard_pattern": "train-{shard:05d}-of-{total_shards:05d}.tfrecord"
        });

        fs::write(output_dir.join("tfrecord_info.json"), serde_json::to_string_pretty(&tfrecord_info)?).await?;

        // Export sample data in JSON format for conversion
        let samples_json = serde_json::to_string_pretty(&dataset.samples)?;
        fs::write(output_dir.join("samples.json"), samples_json).await?;

        Ok(())
    }

    /// Export as SavedModel format
    async fn export_savedmodel(&self, dataset: &TensorFlowDataset, output_dir: &Path) -> Result<()> {
        let savedmodel_info = serde_json::json!({
            "format": "SavedModel",
            "signature_def": {
                "inputs": {
                    "audio": dataset.feature_spec.audio,
                    "text": dataset.feature_spec.text,
                    "speaker_id": dataset.feature_spec.speaker_id,
                    "language": dataset.feature_spec.language
                }
            },
            "metadata": dataset.metadata
        });

        fs::write(output_dir.join("savedmodel_info.json"), serde_json::to_string_pretty(&savedmodel_info)?).await?;
        Ok(())
    }

    /// Export as JSON format
    async fn export_json(&self, dataset: &TensorFlowDataset, output_dir: &Path) -> Result<()> {
        let json_data = serde_json::to_string_pretty(dataset)?;
        fs::write(output_dir.join("dataset.json"), json_data).await?;
        Ok(())
    }

    /// Export dataset information
    async fn export_dataset_info(&self, dataset: &TensorFlowDataset, output_dir: &Path) -> Result<()> {
        let info = serde_json::json!({
            "name": dataset.metadata.name,
            "total_samples": dataset.metadata.total_samples,
            "sample_rate": dataset.metadata.sample_rate,
            "channels": dataset.metadata.channels,
            "vocab_size": dataset.metadata.vocab_size,
            "speakers": dataset.metadata.speakers,
            "languages": dataset.metadata.languages,
            "feature_spec": dataset.feature_spec,
            "config": dataset.config
        });

        fs::write(output_dir.join("dataset_info.json"), serde_json::to_string_pretty(&info)?).await?;
        Ok(())
    }

    /// Export TensorFlow DataLoader script
    async fn export_loader_script(&self, output_dir: &Path) -> Result<()> {
        let script_content = r#"#!/usr/bin/env python3
"""
TensorFlow Dataset Loader for VoiRS-exported dataset.

This script provides TensorFlow tf.data.Dataset functionality for loading the exported dataset.
"""

import json
import tensorflow as tf
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional

class VoiRSDataset:
    """TensorFlow Dataset for VoiRS-exported speech data."""
    
    def __init__(self, data_dir: str):
        """Initialize dataset from exported data."""
        self.data_dir = Path(data_dir)
        
        # Load dataset info
        with open(self.data_dir / "dataset_info.json") as f:
            self.info = json.load(f)
        
        # Load feature spec
        with open(self.data_dir / "feature_spec.json") as f:
            self.feature_spec = json.load(f)
        
        # Load main dataset
        with open(self.data_dir / "dataset.json") as f:
            self.data = json.load(f)
        
        self.samples = self.data['samples']
    
    def to_tf_dataset(self) -> tf.data.Dataset:
        """Convert to tf.data.Dataset."""
        def generator():
            for sample in self.samples:
                features = sample['features']
                
                # Convert features to appropriate TensorFlow types
                tf_features = {}
                
                # Audio feature
                if 'audio' in features:
                    audio_data = features['audio']
                    if isinstance(audio_data, list):
                        tf_features['audio'] = tf.constant(audio_data, dtype=tf.float32)
                
                # Text feature
                if 'text' in features:
                    text_data = features['text']
                    if isinstance(text_data, str):
                        tf_features['text'] = tf.constant(text_data)
                    elif isinstance(text_data, list):
                        tf_features['text'] = tf.constant(text_data, dtype=tf.int64)
                
                # String features
                for key in ['speaker_id', 'language', 'id']:
                    if key in features:
                        tf_features[key] = tf.constant(features[key])
                
                yield tf_features
        
        # Define output signature
        output_signature = {}
        
        # Audio signature
        if self.feature_spec['audio']['shape'] and self.feature_spec['audio']['shape'][0] is not None:
            audio_shape = tuple(self.feature_spec['audio']['shape'])
        else:
            audio_shape = (None,)
        output_signature['audio'] = tf.TensorSpec(shape=audio_shape, dtype=tf.float32)
        
        # Text signature  
        if self.feature_spec['text']['dtype'] == 'string':
            output_signature['text'] = tf.TensorSpec(shape=(), dtype=tf.string)
        else:
            if self.feature_spec['text']['shape'] and self.feature_spec['text']['shape'][0] is not None:
                text_shape = tuple(self.feature_spec['text']['shape'])
            else:
                text_shape = (None,)
            output_signature['text'] = tf.TensorSpec(shape=text_shape, dtype=tf.int64)
        
        # String features
        for key in ['speaker_id', 'language', 'id']:
            output_signature[key] = tf.TensorSpec(shape=(), dtype=tf.string)
        
        return tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
    
    def create_input_pipeline(self, batch_size: int = 32, shuffle: bool = True, 
                            buffer_size: int = 1000) -> tf.data.Dataset:
        """Create optimized input pipeline."""
        dataset = self.to_tf_dataset()
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        
        # Pad sequences if needed
        if self.info['config'].get('pad_sequences', False):
            padded_shapes = {}
            padding_values = {}
            
            # Audio padding
            if self.info['config'].get('fixed_audio_length'):
                padded_shapes['audio'] = (self.info['config']['fixed_audio_length'],)
                padding_values['audio'] = tf.constant(0.0, dtype=tf.float32)
            else:
                padded_shapes['audio'] = (None,)
                padding_values['audio'] = tf.constant(0.0, dtype=tf.float32)
            
            # Text padding
            if self.info['config'].get('fixed_text_length'):
                if self.feature_spec['text']['dtype'] == 'string':
                    padded_shapes['text'] = ()
                    padding_values['text'] = tf.constant('', dtype=tf.string)
                else:
                    padded_shapes['text'] = (self.info['config']['fixed_text_length'],)
                    padding_values['text'] = tf.constant(0, dtype=tf.int64)
            else:
                if self.feature_spec['text']['dtype'] == 'string':
                    padded_shapes['text'] = ()
                    padding_values['text'] = tf.constant('', dtype=tf.string)
                else:
                    padded_shapes['text'] = (None,)
                    padding_values['text'] = tf.constant(0, dtype=tf.int64)
            
            # String features don't need padding
            for key in ['speaker_id', 'language', 'id']:
                padded_shapes[key] = ()
                padding_values[key] = tf.constant('', dtype=tf.string)
            
            dataset = dataset.padded_batch(
                batch_size,
                padded_shapes=padded_shapes,
                padding_values=padding_values
            )
        else:
            dataset = dataset.batch(batch_size)
        
        return dataset.prefetch(tf.data.AUTOTUNE)

def load_dataset(data_dir: str, batch_size: int = 32, shuffle: bool = True) -> tf.data.Dataset:
    """Load VoiRS dataset as tf.data.Dataset."""
    dataset = VoiRSDataset(data_dir)
    return dataset.create_input_pipeline(batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Load VoiRS dataset with TensorFlow')
    parser.add_argument('data_dir', help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--no_shuffle', action='store_true', help='Disable shuffling')
    
    args = parser.parse_args()
    
    # Create dataset
    dataset = load_dataset(args.data_dir, args.batch_size, shuffle=not args.no_shuffle)
    
    print(f"Dataset loaded from {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    
    # Show first batch
    for batch in dataset.take(1):
        print("Batch structure:")
        for key, value in batch.items():
            print(f"  {key}: {value.shape} ({value.dtype})")
        break
"#;

        fs::write(output_dir.join("tensorflow_loader.py"), script_content).await?;
        Ok(())
    }

    /// Update configuration
    pub fn with_config(mut self, config: TensorFlowConfig) -> Self {
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
    async fn test_tensorflow_config_default() {
        let config = TensorFlowConfig::default();
        assert!(matches!(config.format, TensorFlowFormat::TfData));
        assert!(config.create_tfrecords);
        assert!(config.normalize_audio);
        assert!(matches!(config.text_encoding, TextEncoding::Utf8));
    }

    #[tokio::test]
    async fn test_text_encoding() {
        let config = TensorFlowConfig {
            text_encoding: TextEncoding::Characters,
            ..Default::default()
        };
        let exporter = TensorFlowExporter::new(config);

        let (result, vocab_size) = exporter.process_text_feature("hello").unwrap();
        match result {
            TfFeature::Int64List(chars) => {
                assert_eq!(chars.len(), 5); // "hello" has 5 characters
                assert!(vocab_size > 0);
            },
            _ => panic!("Expected Int64List for character encoding"),
        }
    }

    #[tokio::test]
    async fn test_audio_processing() {
        let config = TensorFlowConfig {
            normalize_audio: true,
            ..Default::default()
        };
        let exporter = TensorFlowExporter::new(config);

        let audio = AudioData::new(vec![0.5, -0.8, 0.3], 22050, 1);
        let result = exporter.process_audio_feature(&audio).unwrap();

        match result {
            TfFeature::FloatList(samples) => {
                // Check that audio was normalized
                let max_val = samples.iter().fold(0.0f32, |max, &sample| max.max(sample.abs()));
                assert!((max_val - 1.0).abs() < 0.001);
            },
            _ => panic!("Expected FloatList for audio"),
        }
    }

    #[tokio::test]
    async fn test_export_workflow() {
        let temp_dir = TempDir::new().unwrap();
        let config = TensorFlowConfig::default();
        let exporter = TensorFlowExporter::new(config);

        let samples = vec![
            crate::DatasetSample::new(
                "sample_001".to_string(),
                "Test sample".to_string(),
                AudioData::silence(1.0, 22050, 1),
                LanguageCode::EnUs,
            ),
        ];

        exporter.export_dataset(&samples, temp_dir.path()).await.unwrap();

        // Check that files were created
        assert!(temp_dir.path().join("dataset.json").exists());
        assert!(temp_dir.path().join("feature_spec.json").exists());
        assert!(temp_dir.path().join("dataset_info.json").exists());
        assert!(temp_dir.path().join("tensorflow_loader.py").exists());

        // Check JSON content
        let json_content = fs::read_to_string(temp_dir.path().join("dataset.json")).await.unwrap();
        assert!(json_content.contains("sample_001"));
        assert!(json_content.contains("Test sample"));
    }
}
