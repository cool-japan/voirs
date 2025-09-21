//! Dataset implementations for various speech synthesis datasets
//!
//! This module provides implementations for popular speech synthesis datasets
//! including LJSpeech, VCTK, JVS, and others.

pub mod custom;
pub mod dummy;
pub mod jvs;
pub mod ljspeech;
pub mod vctk;

use crate::{DatasetError, DatasetSample, Result};
use std::path::Path;
use tokio::fs;

/// Detected dataset type based on directory structure analysis
#[derive(Debug, Clone, PartialEq, Eq)]
enum DetectedDatasetType {
    /// LJSpeech dataset (single speaker English)
    LjSpeech,
    /// VCTK dataset (multi-speaker English)
    Vctk,
    /// JVS dataset (multi-speaker Japanese)
    Jvs,
    /// Custom dataset format
    Custom,
    /// Unknown or unsupported dataset format
    Unknown,
}

/// Dataset loader trait
pub trait DatasetLoader {
    /// Load dataset from path
    fn load(&self, path: &Path) -> Result<Box<dyn crate::traits::Dataset<Sample = DatasetSample>>>;

    /// Get dataset name
    fn name(&self) -> &'static str;

    /// Get supported file extensions
    fn extensions(&self) -> &'static [&'static str];
}

/// Dataset registry for automatic dataset detection
pub struct DatasetRegistry {
    loaders: Vec<Box<dyn DatasetLoader>>,
}

impl DatasetRegistry {
    /// Create new dataset registry
    pub fn new() -> Self {
        Self {
            loaders: Vec::new(),
        }
    }

    /// Register dataset loader
    pub fn register<T: DatasetLoader + 'static>(&mut self, loader: T) {
        self.loaders.push(Box::new(loader));
    }

    /// Auto-detect and load dataset
    pub async fn auto_load<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<Box<dyn crate::traits::Dataset<Sample = DatasetSample>>> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(DatasetError::LoadError(format!(
                "Dataset path does not exist: {path:?}"
            )));
        }

        if !path.is_dir() {
            return Err(DatasetError::LoadError(format!(
                "Dataset path must be a directory: {path:?}"
            )));
        }

        // Try to detect dataset type based on directory structure
        let dataset_type = self.detect_dataset_type(path).await?;

        match dataset_type {
            DetectedDatasetType::LjSpeech => {
                let dataset = ljspeech::LjSpeechDataset::load(path).await?;
                Ok(Box::new(dataset))
            }
            DetectedDatasetType::Vctk => {
                let dataset = vctk::VctkDataset::new(path).await?;
                Ok(Box::new(dataset))
            }
            DetectedDatasetType::Jvs => {
                let dataset = jvs::JvsDataset::load(path).await?;
                Ok(Box::new(dataset))
            }
            DetectedDatasetType::Custom => {
                let dataset = custom::CustomDataset::new(path).await?;
                Ok(Box::new(dataset))
            }
            DetectedDatasetType::Unknown => Err(DatasetError::LoadError(format!(
                "Unable to detect dataset type for path: {path:?}. \
                     Please load the dataset using the specific dataset loader."
            ))),
        }
    }

    /// Detect dataset type based on directory structure
    async fn detect_dataset_type<P: AsRef<Path>>(&self, path: P) -> Result<DetectedDatasetType> {
        let path = path.as_ref();

        // Check for LJSpeech dataset
        if self.is_ljspeech_dataset(path).await? {
            return Ok(DetectedDatasetType::LjSpeech);
        }

        // Check for VCTK dataset
        if self.is_vctk_dataset(path).await? {
            return Ok(DetectedDatasetType::Vctk);
        }

        // Check for JVS dataset
        if self.is_jvs_dataset(path).await? {
            return Ok(DetectedDatasetType::Jvs);
        }

        // Check for custom dataset structure
        if self.is_custom_dataset(path).await? {
            return Ok(DetectedDatasetType::Custom);
        }

        Ok(DetectedDatasetType::Unknown)
    }

    /// Check if path contains LJSpeech dataset
    async fn is_ljspeech_dataset<P: AsRef<Path>>(&self, path: P) -> Result<bool> {
        let path = path.as_ref();

        // LJSpeech characteristics:
        // - metadata.csv file
        // - wavs/ directory
        let metadata_file = path.join("metadata.csv");
        let wavs_dir = path.join("wavs");

        Ok(metadata_file.exists() && wavs_dir.exists() && wavs_dir.is_dir())
    }

    /// Check if path contains VCTK dataset
    async fn is_vctk_dataset<P: AsRef<Path>>(&self, path: P) -> Result<bool> {
        let path = path.as_ref();

        // VCTK characteristics:
        // - wav48/ or wav48_silence_trimmed/ directory
        // - txt/ directory
        // - speaker-info.txt file
        let wav48_dir = path.join("wav48");
        let wav48_trimmed_dir = path.join("wav48_silence_trimmed");
        let txt_dir = path.join("txt");
        let speaker_info = path.join("speaker-info.txt");

        let has_wav_dir = wav48_dir.exists() || wav48_trimmed_dir.exists();
        let has_txt_dir = txt_dir.exists() && txt_dir.is_dir();
        let has_speaker_info = speaker_info.exists();

        Ok(has_wav_dir && has_txt_dir && has_speaker_info)
    }

    /// Check if path contains JVS dataset
    async fn is_jvs_dataset<P: AsRef<Path>>(&self, path: P) -> Result<bool> {
        let path = path.as_ref();

        // JVS characteristics:
        // - Speaker directories (jvs001, jvs002, etc.)
        // - Each speaker dir has parallel001, nonpara30, etc.
        let mut entries = fs::read_dir(path)
            .await
            .map_err(|e| DatasetError::LoadError(format!("Failed to read directory: {e}")))?;

        let mut jvs_speaker_count = 0;
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| DatasetError::LoadError(format!("Failed to read directory entry: {e}")))?
        {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("jvs") && name.len() == 6 {
                // Check if it's a valid JVS speaker directory
                let speaker_path = entry.path();
                if speaker_path.is_dir() {
                    let parallel_dir = speaker_path.join("parallel001");
                    let nonpara_dir = speaker_path.join("nonpara30");
                    if parallel_dir.exists() || nonpara_dir.exists() {
                        jvs_speaker_count += 1;
                    }
                }
            }
        }

        // Consider it JVS if we found at least 2 valid speaker directories
        Ok(jvs_speaker_count >= 2)
    }

    /// Check if path contains custom dataset
    async fn is_custom_dataset<P: AsRef<Path>>(&self, path: P) -> Result<bool> {
        let path = path.as_ref();

        // Custom dataset characteristics:
        // - Contains audio files (.wav, .flac, .mp3)
        // - May have manifest.json or similar metadata file
        let mut entries = fs::read_dir(path)
            .await
            .map_err(|e| DatasetError::LoadError(format!("Failed to read directory: {e}")))?;

        let mut has_audio = false;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| DatasetError::LoadError(format!("Failed to read directory entry: {e}")))?
        {
            let name = entry.file_name().to_string_lossy().to_string();
            let name_lower = name.to_lowercase();

            // Check for audio files
            if name_lower.ends_with(".wav")
                || name_lower.ends_with(".flac")
                || name_lower.ends_with(".mp3")
            {
                has_audio = true;
            }
        }

        // Custom dataset if it has audio files (metadata is optional)
        Ok(has_audio)
    }
}

impl Default for DatasetRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio::fs;

    async fn create_ljspeech_structure(base_dir: &Path) -> Result<()> {
        // Create wavs directory
        fs::create_dir_all(base_dir.join("wavs")).await?;

        // Create metadata.csv
        let metadata_content = "LJ001-0001|Printing, in the only sense with which we are at present concerned.|Printing, in the only sense with which we are at present concerned.\n";
        fs::write(base_dir.join("metadata.csv"), metadata_content).await?;

        // Create some dummy wav files
        fs::write(
            base_dir.join("wavs").join("LJ001-0001.wav"),
            b"dummy wav content",
        )
        .await?;

        Ok(())
    }

    async fn create_vctk_structure(base_dir: &Path) -> Result<()> {
        // Create required directories
        fs::create_dir_all(base_dir.join("wav48")).await?;
        fs::create_dir_all(base_dir.join("txt")).await?;
        fs::create_dir_all(base_dir.join("txt").join("p225")).await?;

        // Create speaker-info.txt
        let speaker_info =
            "ID AGE GENDER ACCENTS REGION COMMENTS\np225 23 F English Southern England\n";
        fs::write(base_dir.join("speaker-info.txt"), speaker_info).await?;

        // Create some dummy files
        fs::write(
            base_dir.join("txt").join("p225").join("p225_001.txt"),
            "Please call Stella.",
        )
        .await?;

        Ok(())
    }

    async fn create_jvs_structure(base_dir: &Path) -> Result<()> {
        // Create speaker directories
        for speaker_id in ["jvs001", "jvs002", "jvs003"] {
            let speaker_dir = base_dir.join(speaker_id);
            fs::create_dir_all(&speaker_dir).await?;

            // Create subdirectories
            fs::create_dir_all(speaker_dir.join("parallel001")).await?;
            fs::create_dir_all(speaker_dir.join("nonpara30")).await?;

            // Create some dummy files
            fs::create_dir_all(speaker_dir.join("parallel001").join("wav24kHz16bit")).await?;
            fs::write(
                speaker_dir
                    .join("parallel001")
                    .join("wav24kHz16bit")
                    .join("BASIC5000_0001.wav"),
                b"dummy",
            )
            .await?;
        }

        Ok(())
    }

    async fn create_custom_structure(base_dir: &Path) -> Result<()> {
        // Create some audio files
        fs::write(base_dir.join("audio1.wav"), b"dummy wav content").await?;
        fs::write(base_dir.join("audio2.flac"), b"dummy flac content").await?;
        fs::write(base_dir.join("audio3.mp3"), b"dummy mp3 content").await?;

        // Optionally add manifest
        let manifest = r#"{"name": "custom-dataset", "samples": []}"#;
        fs::write(base_dir.join("manifest.json"), manifest).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_ljspeech_detection() {
        let temp_dir = TempDir::new().unwrap();
        let dataset_path = temp_dir.path();

        create_ljspeech_structure(dataset_path).await.unwrap();

        let registry = DatasetRegistry::new();
        let detected = registry.detect_dataset_type(dataset_path).await.unwrap();

        assert_eq!(detected, DetectedDatasetType::LjSpeech);
        assert!(registry.is_ljspeech_dataset(dataset_path).await.unwrap());
    }

    #[tokio::test]
    async fn test_vctk_detection() {
        let temp_dir = TempDir::new().unwrap();
        let dataset_path = temp_dir.path();

        create_vctk_structure(dataset_path).await.unwrap();

        let registry = DatasetRegistry::new();
        let detected = registry.detect_dataset_type(dataset_path).await.unwrap();

        assert_eq!(detected, DetectedDatasetType::Vctk);
        assert!(registry.is_vctk_dataset(dataset_path).await.unwrap());
    }

    #[tokio::test]
    async fn test_jvs_detection() {
        let temp_dir = TempDir::new().unwrap();
        let dataset_path = temp_dir.path();

        create_jvs_structure(dataset_path).await.unwrap();

        let registry = DatasetRegistry::new();
        let detected = registry.detect_dataset_type(dataset_path).await.unwrap();

        assert_eq!(detected, DetectedDatasetType::Jvs);
        assert!(registry.is_jvs_dataset(dataset_path).await.unwrap());
    }

    #[tokio::test]
    async fn test_custom_detection() {
        let temp_dir = TempDir::new().unwrap();
        let dataset_path = temp_dir.path();

        create_custom_structure(dataset_path).await.unwrap();

        let registry = DatasetRegistry::new();
        let detected = registry.detect_dataset_type(dataset_path).await.unwrap();

        assert_eq!(detected, DetectedDatasetType::Custom);
        assert!(registry.is_custom_dataset(dataset_path).await.unwrap());
    }

    #[tokio::test]
    async fn test_unknown_detection() {
        let temp_dir = TempDir::new().unwrap();
        let dataset_path = temp_dir.path();

        // Create directory with no recognizable structure
        fs::write(dataset_path.join("some_file.txt"), "not a dataset")
            .await
            .unwrap();

        let registry = DatasetRegistry::new();
        let detected = registry.detect_dataset_type(dataset_path).await.unwrap();

        assert_eq!(detected, DetectedDatasetType::Unknown);
    }

    #[tokio::test]
    async fn test_detection_priority() {
        let temp_dir = TempDir::new().unwrap();
        let dataset_path = temp_dir.path();

        // Create LJSpeech structure first
        create_ljspeech_structure(dataset_path).await.unwrap();

        let registry = DatasetRegistry::new();
        let detected = registry.detect_dataset_type(dataset_path).await.unwrap();

        // Should detect as LJSpeech, not custom (even though it has audio files)
        assert_eq!(detected, DetectedDatasetType::LjSpeech);
    }

    #[tokio::test]
    async fn test_vctk_with_trimmed_wav() {
        let temp_dir = TempDir::new().unwrap();
        let dataset_path = temp_dir.path();

        // Create VCTK structure with wav48_silence_trimmed instead of wav48
        fs::create_dir_all(dataset_path.join("wav48_silence_trimmed"))
            .await
            .unwrap();
        fs::create_dir_all(dataset_path.join("txt")).await.unwrap();

        let speaker_info =
            "ID AGE GENDER ACCENTS REGION COMMENTS\np225 23 F English Southern England\n";
        fs::write(dataset_path.join("speaker-info.txt"), speaker_info)
            .await
            .unwrap();

        let registry = DatasetRegistry::new();
        let detected = registry.detect_dataset_type(dataset_path).await.unwrap();

        assert_eq!(detected, DetectedDatasetType::Vctk);
    }

    #[tokio::test]
    async fn test_insufficient_jvs_speakers() {
        let temp_dir = TempDir::new().unwrap();
        let dataset_path = temp_dir.path();

        // Create only one JVS speaker (should not be detected as JVS)
        let speaker_dir = dataset_path.join("jvs001");
        fs::create_dir_all(&speaker_dir).await.unwrap();
        fs::create_dir_all(speaker_dir.join("parallel001"))
            .await
            .unwrap();

        let registry = DatasetRegistry::new();
        let detected = registry.detect_dataset_type(dataset_path).await.unwrap();

        // Should not detect as JVS with only one speaker
        assert_ne!(detected, DetectedDatasetType::Jvs);
    }

    #[tokio::test]
    async fn test_mixed_audio_formats() {
        let temp_dir = TempDir::new().unwrap();
        let dataset_path = temp_dir.path();

        // Create directory with mixed audio formats
        fs::write(dataset_path.join("sample1.wav"), b"wav content")
            .await
            .unwrap();
        fs::write(dataset_path.join("sample2.flac"), b"flac content")
            .await
            .unwrap();
        fs::write(dataset_path.join("sample3.mp3"), b"mp3 content")
            .await
            .unwrap();
        fs::write(dataset_path.join("readme.txt"), "not audio")
            .await
            .unwrap();

        let registry = DatasetRegistry::new();
        let detected = registry.detect_dataset_type(dataset_path).await.unwrap();

        assert_eq!(detected, DetectedDatasetType::Custom);
    }

    #[tokio::test]
    async fn test_nonexistent_path() {
        let registry = DatasetRegistry::new();
        let nonexistent_path = Path::new("/nonexistent/path");

        let result = registry.auto_load(nonexistent_path).await;
        assert!(result.is_err());

        if let Err(DatasetError::LoadError(msg)) = result {
            assert!(msg.contains("does not exist"));
        } else {
            panic!("Expected LoadError for nonexistent path");
        }
    }

    #[tokio::test]
    async fn test_file_instead_of_directory() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("not_a_directory.txt");
        fs::write(&file_path, "content").await.unwrap();

        let registry = DatasetRegistry::new();
        let result = registry.auto_load(&file_path).await;
        assert!(result.is_err());

        if let Err(DatasetError::LoadError(msg)) = result {
            assert!(msg.contains("must be a directory"));
        } else {
            panic!("Expected LoadError for file path");
        }
    }
}
