//! Training data loader for vocoder training
//!
//! This module provides data loading capabilities for neural vocoder training.
//! It supports the LJSpeech dataset format and can be extended to support
//! other TTS datasets.
//!
//! # Features
//!
//! - Async dataset loading with error handling
//! - Real-time mel spectrogram extraction using FFT
//! - Efficient batch generation with sample wraparound
//! - Support for LJSpeech format with automatic validation
//!
//! # Example
//!
//! ```no_run
//! use voirs_cli::commands::train::data_loader::VocoderDataLoader;
//!
//! #[tokio::main]
//! async fn main() -> voirs_sdk::Result<()> {
//!     let mut loader = VocoderDataLoader::load("./data/LJSpeech-1.1").await?;
//!
//!     println!("Loaded {} samples", loader.len());
//!
//!     let batch = loader.get_batch(4)?;
//!     println!("Batch contains {} audio samples and {} mel spectrograms",
//!              batch.audio.len(), batch.mels.len());
//!
//!     Ok(())
//! }
//! ```

use std::path::{Path, PathBuf};
use voirs_dataset::{
    loaders::LjSpeechLoader,
    processing::features::{extract_mel_spectrogram, MelSpectrogramConfig},
    traits::Dataset,
    DatasetSample,
};
use voirs_sdk::Result;

/// Vocoder training data loader
pub struct VocoderDataLoader {
    /// Dataset samples
    samples: Vec<DatasetSample>,
    /// Mel spectrogram configuration
    mel_config: MelSpectrogramConfig,
    /// Current index for iteration
    current_index: usize,
}

impl VocoderDataLoader {
    /// Load dataset from directory
    pub async fn load<P: AsRef<Path>>(data_dir: P) -> Result<Self> {
        // Try to load as LJSpeech dataset
        let is_valid = LjSpeechLoader::is_valid_dataset(data_dir.as_ref());

        let dataset = if is_valid {
            LjSpeechLoader::load(data_dir).await.map_err(|e| {
                voirs_sdk::VoirsError::config_error(format!("Failed to load dataset: {}", e))
            })?
        } else {
            return Err(voirs_sdk::VoirsError::config_error(format!(
                "Unsupported dataset format at {:?}. Currently only LJSpeech is supported.",
                data_dir.as_ref()
            )));
        };

        // Get all samples
        let num_samples = dataset.len();
        let mut samples = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            match dataset.get(i).await {
                Ok(sample) => samples.push(sample),
                Err(e) => {
                    // Log warning and continue with other samples
                    eprintln!("Warning: Failed to load sample {}: {}", i, e);
                }
            }
        }

        if samples.is_empty() {
            return Err(voirs_sdk::VoirsError::config_error(
                "No valid samples found in dataset".to_string(),
            ));
        }

        Ok(Self {
            samples,
            mel_config: MelSpectrogramConfig::default(),
            current_index: 0,
        })
    }

    /// Get total number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get batch of audio samples with mel spectrograms
    pub fn get_batch(&mut self, batch_size: usize) -> Result<VocoderBatch> {
        let mut batch_audio = Vec::new();
        let mut batch_mels = Vec::new();

        for _ in 0..batch_size {
            if self.current_index >= self.samples.len() {
                // Wrap around to beginning (epoch completed)
                self.current_index = 0;
            }

            let sample = &self.samples[self.current_index];
            self.current_index += 1;

            // Extract mel spectrogram
            let mel_result = extract_mel_spectrogram(
                &sample.audio,
                self.mel_config.n_mels,
                self.mel_config.n_fft,
                self.mel_config.hop_length,
            )
            .map_err(|e| {
                voirs_sdk::VoirsError::config_error(format!(
                    "Failed to extract mel spectrogram: {}",
                    e
                ))
            })?;

            // Convert to Vec<Vec<f32>> (frames x mels)
            let mel_matrix = mel_result.as_matrix();

            batch_audio.push(sample.audio.samples().to_vec());
            batch_mels.push(mel_matrix);
        }

        Ok(VocoderBatch {
            audio: batch_audio,
            mels: batch_mels,
        })
    }

    /// Reset iterator to beginning
    pub fn reset(&mut self) {
        self.current_index = 0;
    }

    /// Get current index in dataset
    pub fn current_index(&self) -> usize {
        self.current_index
    }

    /// Set current index in dataset
    pub fn set_index(&mut self, index: usize) {
        self.current_index = index.min(self.samples.len());
    }
}

/// Batch of vocoder training data
pub struct VocoderBatch {
    /// Audio waveforms (batch_size x samples)
    pub audio: Vec<Vec<f32>>,
    /// Mel spectrograms (batch_size x frames x n_mels)
    pub mels: Vec<Vec<Vec<f32>>>,
}

impl VocoderBatch {
    /// Get batch size
    pub fn len(&self) -> usize {
        self.audio.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.audio.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn resolve_ljspeech_path() -> PathBuf {
        env::var("LJSPEECH_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                std::env::temp_dir()
                    .join("voirs")
                    .join("datasets")
                    .join("LJSpeech-1.1")
            })
    }

    #[tokio::test]
    async fn test_vocoder_data_loader_basic() {
        // Test with LJSpeech dataset if available
        let ljspeech_path = resolve_ljspeech_path();

        if !ljspeech_path.exists() {
            eprintln!(
                "Skipping test: LJSpeech dataset not found at {}",
                ljspeech_path.display()
            );
            return;
        }

        let loader = VocoderDataLoader::load(&ljspeech_path).await;
        assert!(loader.is_ok(), "Failed to load dataset");

        let loader = loader.unwrap();
        assert!(loader.len() > 0, "Dataset should not be empty");
        assert!(!loader.is_empty(), "Dataset should not be empty");
    }

    #[tokio::test]
    async fn test_batch_generation() {
        let ljspeech_path = resolve_ljspeech_path();

        if !ljspeech_path.exists() {
            eprintln!("Skipping test: LJSpeech dataset not found");
            return;
        }

        let mut loader = VocoderDataLoader::load(&ljspeech_path).await.unwrap();
        let batch_size = 4;
        let batch = loader.get_batch(batch_size).unwrap();

        assert_eq!(batch.len(), batch_size, "Batch size should match");
        assert_eq!(
            batch.audio.len(),
            batch_size,
            "Audio batch size should match"
        );
        assert_eq!(batch.mels.len(), batch_size, "Mel batch size should match");

        // Verify mel spectrogram dimensions
        for mel in &batch.mels {
            assert!(!mel.is_empty(), "Mel spectrogram should not be empty");
            assert!(mel[0].len() > 0, "Mel spectrogram should have features");
        }
    }

    #[tokio::test]
    async fn test_batch_wraparound() {
        let ljspeech_path = resolve_ljspeech_path();

        if !ljspeech_path.exists() {
            eprintln!("Skipping test: LJSpeech dataset not found");
            return;
        }

        let mut loader = VocoderDataLoader::load(&ljspeech_path).await.unwrap();
        let total_samples = loader.len();

        // Consume all samples plus some to test wraparound
        let batch_size = 4;
        let num_batches = (total_samples / batch_size) + 2;

        for i in 0..num_batches {
            let batch = loader.get_batch(batch_size);
            assert!(batch.is_ok(), "Batch generation failed at iteration {}", i);
            assert_eq!(batch.unwrap().len(), batch_size);
        }
    }

    #[test]
    fn test_vocoder_batch_properties() {
        let batch = VocoderBatch {
            audio: vec![vec![0.0; 100]; 4],
            mels: vec![vec![vec![0.0; 80]; 10]; 4],
        };

        assert_eq!(batch.len(), 4);
        assert!(!batch.is_empty());

        let empty_batch = VocoderBatch {
            audio: vec![],
            mels: vec![],
        };

        assert_eq!(empty_batch.len(), 0);
        assert!(empty_batch.is_empty());
    }

    #[tokio::test]
    async fn test_invalid_dataset_path() {
        let invalid_path = "/nonexistent/path/to/dataset";
        let result = VocoderDataLoader::load(invalid_path).await;

        assert!(result.is_err(), "Should fail with invalid path");
    }

    #[tokio::test]
    async fn test_mel_spectrogram_shape() {
        let ljspeech_path = resolve_ljspeech_path();

        if !ljspeech_path.exists() {
            eprintln!("Skipping test: LJSpeech dataset not found");
            return;
        }

        let mut loader = VocoderDataLoader::load(&ljspeech_path).await.unwrap();
        let batch = loader.get_batch(1).unwrap();

        assert_eq!(batch.mels.len(), 1);

        let mel = &batch.mels[0];
        assert!(!mel.is_empty(), "Mel spectrogram should have frames");

        // Check each frame has correct number of mel bins (80)
        for frame in mel {
            assert_eq!(frame.len(), 80, "Each frame should have 80 mel bins");
        }
    }
}
