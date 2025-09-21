//! Dataset loaders for various formats.
//!
//! This module provides simple loader functions that wrap the underlying dataset implementations
//! for easy use in applications. These loaders provide a consistent interface for loading
//! different dataset formats.

use crate::datasets::{custom, jvs, ljspeech, vctk};
use crate::traits::Dataset;
use crate::{DatasetError, DatasetSample, Result};
use std::path::Path;
use std::sync::Arc;

/// LJSpeech dataset loader
pub struct LjSpeechLoader;

impl LjSpeechLoader {
    /// Load LJSpeech dataset from the given path
    pub async fn load<P: AsRef<Path>>(path: P) -> Result<Arc<dyn Dataset<Sample = DatasetSample>>> {
        let dataset = ljspeech::LjSpeechDataset::load(path).await?;
        Ok(Arc::new(dataset))
    }

    /// Check if the given path contains a valid LJSpeech dataset
    pub fn is_valid_dataset<P: AsRef<Path>>(path: P) -> bool {
        let path = path.as_ref();
        let metadata_file = path.join("metadata.csv");
        let wavs_dir = path.join("wavs");
        metadata_file.exists() && wavs_dir.exists() && wavs_dir.is_dir()
    }
}

/// VCTK dataset loader
pub struct VctkLoader;

impl VctkLoader {
    /// Load VCTK dataset from the given path
    pub async fn load<P: AsRef<Path>>(path: P) -> Result<Arc<dyn Dataset<Sample = DatasetSample>>> {
        let dataset = vctk::VctkDataset::new(path).await?;
        Ok(Arc::new(dataset))
    }

    /// Load VCTK dataset with custom configuration
    pub async fn load_with_config(
        config: vctk::VctkConfig,
    ) -> Result<Arc<dyn Dataset<Sample = DatasetSample>>> {
        let dataset = vctk::VctkDataset::from_config(config).await?;
        Ok(Arc::new(dataset))
    }

    /// Check if the given path contains a valid VCTK dataset
    pub fn is_valid_dataset<P: AsRef<Path>>(path: P) -> bool {
        let path = path.as_ref();
        let wav48_dir = path.join("wav48");
        let wav48_trimmed_dir = path.join("wav48_silence_trimmed");
        let txt_dir = path.join("txt");
        let speaker_info = path.join("speaker-info.txt");

        let has_wav_dir = wav48_dir.exists() || wav48_trimmed_dir.exists();
        let has_txt_dir = txt_dir.exists() && txt_dir.is_dir();
        let has_speaker_info = speaker_info.exists();

        has_wav_dir && has_txt_dir && has_speaker_info
    }
}

/// JVS dataset loader
pub struct JvsLoader;

impl JvsLoader {
    /// Load JVS dataset from the given path
    pub async fn load<P: AsRef<Path>>(path: P) -> Result<Arc<dyn Dataset<Sample = DatasetSample>>> {
        let dataset = jvs::JvsDataset::load(path).await?;
        Ok(Arc::new(dataset))
    }

    /// Check if the given path contains a valid JVS dataset
    pub fn is_valid_dataset<P: AsRef<Path>>(path: P) -> bool {
        let path = path.as_ref();
        // Look for JVS speaker directories (jvs001, jvs002, etc.)
        if let Ok(entries) = std::fs::read_dir(path) {
            let jvs_dirs: Vec<_> = entries
                .filter_map(|entry| entry.ok())
                .filter(|entry| {
                    let name = entry.file_name().to_string_lossy().to_string();
                    name.starts_with("jvs") && name.len() == 6 && entry.path().is_dir()
                })
                .collect();
            jvs_dirs.len() >= 2
        } else {
            false
        }
    }
}

/// LibriTTS dataset loader
pub struct LibriTtsLoader;

impl LibriTtsLoader {
    /// Load LibriTTS dataset from the given path
    pub async fn load<P: AsRef<Path>>(path: P) -> Result<Arc<dyn Dataset<Sample = DatasetSample>>> {
        let config = custom::CustomConfig {
            root_dir: path.as_ref().to_path_buf(),
            ..Default::default()
        };
        let dataset = custom::CustomDataset::from_config(config).await?;
        Ok(Arc::new(dataset))
    }

    /// Load specific LibriTTS subset (e.g., "train-clean-100", "dev-clean")
    pub async fn load_subset<P: AsRef<Path>>(
        base_path: P,
        subset: &str,
    ) -> Result<Arc<dyn Dataset<Sample = DatasetSample>>> {
        let subset_path = base_path.as_ref().join(subset);
        let config = custom::CustomConfig {
            root_dir: subset_path,
            ..Default::default()
        };
        let dataset = custom::CustomDataset::from_config(config).await?;
        Ok(Arc::new(dataset))
    }

    /// Check if the given path contains a valid LibriTTS dataset
    pub fn is_valid_dataset<P: AsRef<Path>>(path: P) -> bool {
        let path = path.as_ref();

        // Check for LibriTTS directory structure (speaker_id/book_id/*.wav)
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.filter_map(|e| e.ok()) {
                if entry.path().is_dir() {
                    let speaker_dir = entry.path();
                    if let Ok(book_entries) = std::fs::read_dir(&speaker_dir) {
                        for book_entry in book_entries.filter_map(|e| e.ok()) {
                            if book_entry.path().is_dir() {
                                // Check for .wav files in book directory
                                if let Ok(audio_entries) = std::fs::read_dir(book_entry.path()) {
                                    let wav_files = audio_entries
                                        .filter_map(|e| e.ok())
                                        .filter(|e| {
                                            e.path().extension().and_then(|ext| ext.to_str())
                                                == Some("wav")
                                        })
                                        .count();
                                    if wav_files > 0 {
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        false
    }
}

/// Common Voice dataset loader
pub struct CommonVoiceLoader;

impl CommonVoiceLoader {
    /// Load Common Voice dataset from the given path
    pub async fn load<P: AsRef<Path>>(path: P) -> Result<Arc<dyn Dataset<Sample = DatasetSample>>> {
        let config = custom::CustomConfig {
            root_dir: path.as_ref().to_path_buf(),
            transcript_format: custom::TranscriptFormat::Csv,
            ..Default::default()
        };
        let dataset = custom::CustomDataset::from_config(config).await?;
        Ok(Arc::new(dataset))
    }

    /// Load Common Voice with specific language
    pub async fn load_language<P: AsRef<Path>>(
        path: P,
        language: &str,
    ) -> Result<Arc<dyn Dataset<Sample = DatasetSample>>> {
        let lang_path = path.as_ref().join(language);
        let config = custom::CustomConfig {
            root_dir: lang_path,
            transcript_format: custom::TranscriptFormat::Csv,
            ..Default::default()
        };
        let dataset = custom::CustomDataset::from_config(config).await?;
        Ok(Arc::new(dataset))
    }

    /// Check if the given path contains a valid Common Voice dataset
    pub fn is_valid_dataset<P: AsRef<Path>>(path: P) -> bool {
        let path = path.as_ref();
        let clips_dir = path.join("clips");
        let validated_tsv = path.join("validated.tsv");
        let train_tsv = path.join("train.tsv");
        let test_tsv = path.join("test.tsv");

        clips_dir.exists()
            && clips_dir.is_dir()
            && (validated_tsv.exists() || train_tsv.exists() || test_tsv.exists())
    }
}

/// CSV dataset loader
pub struct CsvLoader;

impl CsvLoader {
    /// Load dataset from CSV file with default column mapping
    pub async fn load<P: AsRef<Path>>(
        csv_path: P,
    ) -> Result<Arc<dyn Dataset<Sample = DatasetSample>>> {
        let config = custom::CustomConfig {
            root_dir: csv_path
                .as_ref()
                .parent()
                .unwrap_or(Path::new("."))
                .to_path_buf(),
            transcript_format: custom::TranscriptFormat::Csv,
            csv_config: custom::CsvConfig::default(),
            ..Default::default()
        };

        let dataset = custom::CustomDataset::from_config(config).await?;
        Ok(Arc::new(dataset))
    }

    /// Load dataset from CSV with custom column mapping
    pub async fn load_with_columns<P: AsRef<Path>>(
        csv_path: P,
        audio_column: &str,
        text_column: &str,
        speaker_column: Option<&str>,
    ) -> Result<Arc<dyn Dataset<Sample = DatasetSample>>> {
        let csv_config = custom::CsvConfig {
            audio_column: audio_column.to_string(),
            text_column: text_column.to_string(),
            speaker_column: speaker_column.map(ToString::to_string),
            ..Default::default()
        };

        let config = custom::CustomConfig {
            root_dir: csv_path
                .as_ref()
                .parent()
                .unwrap_or(Path::new("."))
                .to_path_buf(),
            transcript_format: custom::TranscriptFormat::Csv,
            csv_config,
            ..Default::default()
        };

        let dataset = custom::CustomDataset::from_config(config).await?;
        Ok(Arc::new(dataset))
    }
}

/// JSON dataset loader
pub struct JsonLoader;

impl JsonLoader {
    /// Load dataset from JSON file
    pub async fn load<P: AsRef<Path>>(
        json_path: P,
    ) -> Result<Arc<dyn Dataset<Sample = DatasetSample>>> {
        let config = custom::CustomConfig {
            root_dir: json_path
                .as_ref()
                .parent()
                .unwrap_or(Path::new("."))
                .to_path_buf(),
            transcript_format: custom::TranscriptFormat::Json,
            ..Default::default()
        };

        let dataset = custom::CustomDataset::from_config(config).await?;
        Ok(Arc::new(dataset))
    }

    /// Load dataset from JSON Lines (.jsonl) file
    pub async fn load_jsonl<P: AsRef<Path>>(
        jsonl_path: P,
    ) -> Result<Arc<dyn Dataset<Sample = DatasetSample>>> {
        let config = custom::CustomConfig {
            root_dir: jsonl_path
                .as_ref()
                .parent()
                .unwrap_or(Path::new("."))
                .to_path_buf(),
            transcript_format: custom::TranscriptFormat::JsonLines,
            ..Default::default()
        };

        let dataset = custom::CustomDataset::from_config(config).await?;
        Ok(Arc::new(dataset))
    }
}

/// Custom dataset loader
pub struct CustomLoader;

impl CustomLoader {
    /// Load custom dataset from the given path
    pub async fn load<P: AsRef<Path>>(path: P) -> Result<Arc<dyn Dataset<Sample = DatasetSample>>> {
        let dataset = custom::CustomDataset::new(path).await?;
        Ok(Arc::new(dataset))
    }

    /// Load custom dataset with configuration
    pub async fn load_with_config(
        config: custom::CustomConfig,
    ) -> Result<Arc<dyn Dataset<Sample = DatasetSample>>> {
        let dataset = custom::CustomDataset::from_config(config).await?;
        Ok(Arc::new(dataset))
    }

    /// Check if the given path contains audio files (basic custom dataset validation)
    pub fn is_valid_dataset<P: AsRef<Path>>(path: P) -> bool {
        let path = path.as_ref();
        if let Ok(entries) = std::fs::read_dir(path) {
            let audio_files: Vec<_> = entries
                .filter_map(|entry| entry.ok())
                .filter(|entry| {
                    let name = entry
                        .file_name()
                        .to_string_lossy()
                        .to_string()
                        .to_lowercase();
                    name.ends_with(".wav") || name.ends_with(".flac") || name.ends_with(".mp3")
                })
                .collect();
            !audio_files.is_empty()
        } else {
            false
        }
    }
}

/// Auto-detection loader that tries to identify the dataset type
pub struct AutoLoader;

impl AutoLoader {
    /// Automatically detect and load dataset from the given path
    pub async fn load<P: AsRef<Path>>(path: P) -> Result<Arc<dyn Dataset<Sample = DatasetSample>>> {
        let path = path.as_ref();

        // Try different dataset types in order of specificity
        if LjSpeechLoader::is_valid_dataset(path) {
            return LjSpeechLoader::load(path).await;
        }

        if VctkLoader::is_valid_dataset(path) {
            return VctkLoader::load(path).await;
        }

        if JvsLoader::is_valid_dataset(path) {
            return JvsLoader::load(path).await;
        }

        if LibriTtsLoader::is_valid_dataset(path) {
            return LibriTtsLoader::load(path).await;
        }

        if CommonVoiceLoader::is_valid_dataset(path) {
            return CommonVoiceLoader::load(path).await;
        }

        // Check for specific file formats
        if path.is_file() {
            if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
                match extension.to_lowercase().as_str() {
                    "csv" => return CsvLoader::load(path).await,
                    "json" => return JsonLoader::load(path).await,
                    "jsonl" => return JsonLoader::load_jsonl(path).await,
                    _ => {}
                }
            }
        }

        if CustomLoader::is_valid_dataset(path) {
            return CustomLoader::load(path).await;
        }

        Err(DatasetError::LoadError(format!(
            "Unable to detect dataset type for path: {path:?}. \
             The directory does not match any known dataset format."
        )))
    }

    /// Detect the dataset type without loading it
    pub fn detect_type<P: AsRef<Path>>(path: P) -> Option<DatasetType> {
        let path = path.as_ref();

        if LjSpeechLoader::is_valid_dataset(path) {
            Some(DatasetType::LjSpeech)
        } else if VctkLoader::is_valid_dataset(path) {
            Some(DatasetType::Vctk)
        } else if JvsLoader::is_valid_dataset(path) {
            Some(DatasetType::Jvs)
        } else if LibriTtsLoader::is_valid_dataset(path) {
            Some(DatasetType::LibriTts)
        } else if CommonVoiceLoader::is_valid_dataset(path) {
            Some(DatasetType::CommonVoice)
        } else if path.is_file() {
            if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
                match extension.to_lowercase().as_str() {
                    "csv" => Some(DatasetType::Csv),
                    "json" => Some(DatasetType::Json),
                    "jsonl" => Some(DatasetType::JsonLines),
                    _ => None,
                }
            } else {
                None
            }
        } else if CustomLoader::is_valid_dataset(path) {
            Some(DatasetType::Custom)
        } else {
            None
        }
    }
}

/// Supported dataset types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetType {
    /// LJSpeech dataset (single speaker English)
    LjSpeech,
    /// VCTK dataset (multi-speaker English)
    Vctk,
    /// JVS dataset (multi-speaker Japanese)
    Jvs,
    /// LibriTTS dataset (multi-speaker English)
    LibriTts,
    /// Common Voice dataset (multilingual crowdsourced)
    CommonVoice,
    /// CSV format dataset
    Csv,
    /// JSON format dataset
    Json,
    /// JSON Lines format dataset
    JsonLines,
    /// Custom dataset format
    Custom,
}

impl DatasetType {
    /// Get the name of the dataset type
    pub fn name(&self) -> &'static str {
        match self {
            DatasetType::LjSpeech => "LJSpeech",
            DatasetType::Vctk => "VCTK",
            DatasetType::Jvs => "JVS",
            DatasetType::LibriTts => "LibriTTS",
            DatasetType::CommonVoice => "Common Voice",
            DatasetType::Csv => "CSV",
            DatasetType::Json => "JSON",
            DatasetType::JsonLines => "JSON Lines",
            DatasetType::Custom => "Custom",
        }
    }

    /// Get the description of the dataset type
    pub fn description(&self) -> &'static str {
        match self {
            DatasetType::LjSpeech => "Single speaker English speech synthesis dataset",
            DatasetType::Vctk => "Multi-speaker English speech corpus with various accents",
            DatasetType::Jvs => "Multi-speaker Japanese versatile speech corpus",
            DatasetType::LibriTts => {
                "Large-scale multi-speaker English corpus from LibriVox audiobooks"
            }
            DatasetType::CommonVoice => "Mozilla's multilingual crowdsourced speech dataset",
            DatasetType::Csv => "Dataset defined in CSV format with configurable columns",
            DatasetType::Json => "Dataset defined in JSON format",
            DatasetType::JsonLines => {
                "Dataset defined in JSON Lines format (one JSON object per line)"
            }
            DatasetType::Custom => "Custom dataset with flexible structure",
        }
    }
}

impl std::fmt::Display for DatasetType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
