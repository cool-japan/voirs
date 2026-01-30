//! Mozilla CommonVoice dataset loader and processor
//!
//! CommonVoice is a multilingual speech corpus made available by Mozilla.
//! It contains crowd-sourced voice recordings in many languages.
//!
//! Dataset structure:
//! ```text
//! cv-corpus-{version}-{date}/{language}/
//! ├── clips/
//! │   ├── {clip_id}.mp3
//! │   └── ...
//! ├── train.tsv
//! ├── dev.tsv
//! ├── test.tsv
//! ├── validated.tsv
//! ├── invalidated.tsv
//! └── other.tsv
//! ```
//!
//! # Example
//!
//! ```no_run
//! use voirs_dataset::datasets::commonvoice::CommonVoiceDataset;
//! use std::path::Path;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let dataset = CommonVoiceDataset::load(
//!         Path::new("/path/to/cv-corpus/en"),
//!         "train"
//!     ).await?;
//!
//!     println!("Loaded {} samples", dataset.len());
//!     Ok(())
//! }
//! ```

use crate::{
    traits::{Dataset, DatasetMetadata},
    AudioData, DatasetError, DatasetSample, DatasetStatistics, DurationStatistics, LanguageCode,
    LengthStatistics, QualityMetrics, Result, SpeakerInfo, ValidationReport,
};
use async_trait::async_trait;
use csv::ReaderBuilder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::fs;
use tracing::{debug, info, warn};

/// CommonVoice split types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CommonVoiceSplit {
    /// Training split
    Train,
    /// Development/validation split
    Dev,
    /// Test split
    Test,
    /// Validated clips
    Validated,
    /// Invalidated clips
    Invalidated,
    /// Other clips
    Other,
}

impl CommonVoiceSplit {
    /// Get TSV filename for this split
    pub fn tsv_name(&self) -> &'static str {
        match self {
            CommonVoiceSplit::Train => "train.tsv",
            CommonVoiceSplit::Dev => "dev.tsv",
            CommonVoiceSplit::Test => "test.tsv",
            CommonVoiceSplit::Validated => "validated.tsv",
            CommonVoiceSplit::Invalidated => "invalidated.tsv",
            CommonVoiceSplit::Other => "other.tsv",
        }
    }

    /// Parse split name
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "train" => Ok(CommonVoiceSplit::Train),
            "dev" | "development" => Ok(CommonVoiceSplit::Dev),
            "test" => Ok(CommonVoiceSplit::Test),
            "validated" => Ok(CommonVoiceSplit::Validated),
            "invalidated" => Ok(CommonVoiceSplit::Invalidated),
            "other" => Ok(CommonVoiceSplit::Other),
            _ => Err(DatasetError::FormatError(format!(
                "Unknown CommonVoice split: {s}"
            ))),
        }
    }
}

/// CommonVoice TSV row structure
#[derive(Debug, Clone, Deserialize)]
struct CommonVoiceTsvRow {
    /// Client ID (speaker identifier)
    client_id: String,
    /// Path to audio clip (relative to clips/ directory)
    path: String,
    /// Sentence text
    sentence: String,
    /// Number of upvotes
    #[serde(default)]
    up_votes: u32,
    /// Number of downvotes
    #[serde(default)]
    down_votes: u32,
    /// Age of speaker (optional)
    #[serde(default)]
    age: String,
    /// Gender of speaker (optional)
    #[serde(default)]
    gender: String,
    /// Accent of speaker (optional)
    #[serde(default)]
    accent: String,
    /// Locale/language code
    #[serde(default)]
    locale: String,
}

/// CommonVoice dataset entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonVoiceEntry {
    /// Client/speaker ID
    pub client_id: String,
    /// Audio file path
    pub audio_path: PathBuf,
    /// Sentence text
    pub sentence: String,
    /// Number of upvotes
    pub up_votes: u32,
    /// Number of downvotes
    pub down_votes: u32,
    /// Age (if provided)
    pub age: Option<String>,
    /// Gender (if provided)
    pub gender: Option<String>,
    /// Accent (if provided)
    pub accent: Option<String>,
    /// Language locale
    pub locale: String,
    /// Dataset split
    pub split: CommonVoiceSplit,
}

impl CommonVoiceEntry {
    /// Get unique ID for this entry
    pub fn id(&self) -> String {
        format!(
            "cv_{}_{}",
            self.client_id,
            self.audio_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
        )
    }

    /// Calculate vote-based quality score (0.0 - 1.0)
    pub fn vote_quality(&self) -> f32 {
        let total_votes = self.up_votes + self.down_votes;
        if total_votes == 0 {
            0.5 // Unknown quality
        } else {
            self.up_votes as f32 / total_votes as f32
        }
    }

    /// Check if this entry has sufficient validation (at least 2 upvotes)
    pub fn is_validated(&self) -> bool {
        self.up_votes >= 2
    }
}

/// CommonVoice dataset implementation
#[derive(Debug, Clone)]
pub struct CommonVoiceDataset {
    /// Dataset root directory (language-specific directory)
    root: PathBuf,
    /// All entries in the dataset
    entries: Vec<CommonVoiceEntry>,
    /// Speaker metadata cache (thread-safe)
    speaker_cache: Arc<Mutex<HashMap<String, SpeakerInfo>>>,
    /// Language code
    language: LanguageCode,
    /// Dataset split
    split: CommonVoiceSplit,
    /// Dataset metadata
    metadata: DatasetMetadata,
}

impl CommonVoiceDataset {
    /// Load CommonVoice dataset from directory
    ///
    /// # Arguments
    ///
    /// * `root` - Language-specific root directory (e.g., cv-corpus/en/)
    /// * `split` - Split name ("train", "dev", "test", etc.)
    pub async fn load<P: AsRef<Path>>(root: P, split: &str) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        info!("Loading CommonVoice dataset from: {root:?}, split: {split}");

        if !root.exists() {
            return Err(DatasetError::LoadError(format!(
                "Dataset root does not exist: {root:?}"
            )));
        }

        let split_enum = CommonVoiceSplit::parse(split)?;
        let tsv_path = root.join(split_enum.tsv_name());

        if !tsv_path.exists() {
            return Err(DatasetError::LoadError(format!(
                "Split file does not exist: {tsv_path:?}"
            )));
        }

        info!("Loading TSV from: {tsv_path:?}");
        let entries = Self::load_tsv(&tsv_path, &root, split_enum).await?;

        if entries.is_empty() {
            return Err(DatasetError::LoadError(
                "No valid entries found in CommonVoice dataset".to_string(),
            ));
        }

        // Detect language from first entry or directory name
        let language = Self::detect_language(&entries, &root)?;

        info!(
            "Loaded {} entries from CommonVoice dataset (language: {:?})",
            entries.len(),
            language
        );

        // Create metadata
        let total_samples = entries.len();
        let total_duration = 0.0; // Would need to load all audio to calculate
        let speakers: Vec<String> = entries
            .iter()
            .map(|e| e.client_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let num_speakers = speakers.len();

        let metadata = DatasetMetadata {
            name: "CommonVoice".to_string(),
            version: "11.0".to_string(),
            description: Some(format!(
                "Mozilla CommonVoice multilingual corpus ({:?} - {})",
                language,
                split_enum.tsv_name()
            )),
            total_samples,
            total_duration,
            languages: vec![language.as_str().to_string()],
            speakers,
            license: Some("CC0 1.0".to_string()),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert(
                    "num_speakers".to_string(),
                    serde_json::Value::Number(num_speakers.into()),
                );
                meta
            },
        };

        Ok(Self {
            root,
            entries,
            speaker_cache: Arc::new(Mutex::new(HashMap::new())),
            language,
            split: split_enum,
            metadata,
        })
    }

    /// Load entries from TSV file
    async fn load_tsv(
        tsv_path: &Path,
        root: &Path,
        split: CommonVoiceSplit,
    ) -> Result<Vec<CommonVoiceEntry>> {
        let content = fs::read_to_string(tsv_path).await.map_err(|e| {
            DatasetError::IoError(std::io::Error::new(
                e.kind(),
                format!("Failed to read TSV file {tsv_path:?}: {e}"),
            ))
        })?;

        let mut reader = ReaderBuilder::new()
            .delimiter(b'\t')
            .from_reader(content.as_bytes());

        let mut entries = Vec::new();
        let clips_dir = root.join("clips");

        for (line_num, result) in reader.deserialize().enumerate() {
            match result {
                Ok(row) => {
                    let row: CommonVoiceTsvRow = row;

                    // Construct audio path
                    let audio_path = clips_dir.join(&row.path);

                    // Only include entries where audio file exists
                    if !audio_path.exists() {
                        debug!("Audio file not found, skipping: {audio_path:?}");
                        continue;
                    }

                    entries.push(CommonVoiceEntry {
                        client_id: row.client_id,
                        audio_path,
                        sentence: row.sentence,
                        up_votes: row.up_votes,
                        down_votes: row.down_votes,
                        age: if row.age.is_empty() {
                            None
                        } else {
                            Some(row.age)
                        },
                        gender: if row.gender.is_empty() {
                            None
                        } else {
                            Some(row.gender)
                        },
                        accent: if row.accent.is_empty() {
                            None
                        } else {
                            Some(row.accent)
                        },
                        locale: row.locale,
                        split,
                    });
                }
                Err(e) => {
                    warn!("Failed to parse TSV row {line_num}: {e}");
                }
            }
        }

        Ok(entries)
    }

    /// Detect language from entries or directory name
    fn detect_language(entries: &[CommonVoiceEntry], root: &Path) -> Result<LanguageCode> {
        // Try to get language from first entry's locale
        if let Some(entry) = entries.first() {
            if !entry.locale.is_empty() {
                return Self::parse_language_code(&entry.locale);
            }
        }

        // Try to detect from directory name
        if let Some(dir_name) = root.file_name().and_then(|n| n.to_str()) {
            // CommonVoice uses ISO 639-1 codes (e.g., "en", "ja", "fr")
            match dir_name {
                "en" => return Ok(LanguageCode::EnUs),
                "ja" => return Ok(LanguageCode::Ja),
                "zh-CN" => return Ok(LanguageCode::ZhCn),
                "ko" => return Ok(LanguageCode::Ko),
                "de" => return Ok(LanguageCode::De),
                "fr" => return Ok(LanguageCode::Fr),
                "es" => return Ok(LanguageCode::Es),
                _ => {}
            }
        }

        // Default to English if cannot detect
        warn!("Could not detect language, defaulting to en-US");
        Ok(LanguageCode::EnUs)
    }

    /// Parse language code string to LanguageCode enum
    fn parse_language_code(code: &str) -> Result<LanguageCode> {
        match code.to_lowercase().as_str() {
            "en" | "en-us" | "en_us" => Ok(LanguageCode::EnUs),
            "en-gb" | "en_gb" => Ok(LanguageCode::EnGb),
            "ja" | "ja-jp" | "ja_jp" => Ok(LanguageCode::Ja),
            "zh" | "zh-cn" | "zh_cn" | "cmn" => Ok(LanguageCode::ZhCn),
            "ko" | "ko-kr" | "ko_kr" => Ok(LanguageCode::Ko),
            "de" | "de-de" | "de_de" => Ok(LanguageCode::De),
            "fr" | "fr-fr" | "fr_fr" => Ok(LanguageCode::Fr),
            "es" | "es-es" | "es_es" => Ok(LanguageCode::Es),
            _ => {
                warn!("Unknown language code: {code}, defaulting to en-US");
                Ok(LanguageCode::EnUs)
            }
        }
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get entry by index
    pub fn get_entry(&self, index: usize) -> Result<&CommonVoiceEntry> {
        self.entries
            .get(index)
            .ok_or(DatasetError::IndexError(index))
    }

    /// Load audio from entry (supports MP3 files)
    fn load_audio(&self, entry: &CommonVoiceEntry) -> Result<AudioData> {
        // CommonVoice uses MP3 files
        let ext = entry
            .audio_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        match ext.to_lowercase().as_str() {
            "mp3" => self.load_mp3_audio(entry),
            "wav" => self.load_wav_audio(entry),
            _ => Err(DatasetError::FormatError(format!(
                "Unsupported audio format: {ext}"
            ))),
        }
    }

    /// Load WAV audio file
    fn load_wav_audio(&self, entry: &CommonVoiceEntry) -> Result<AudioData> {
        let reader = hound::WavReader::open(&entry.audio_path).map_err(|e| {
            DatasetError::AudioError(format!(
                "Failed to open WAV file {:?}: {}",
                entry.audio_path, e
            ))
        })?;

        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let channels = spec.channels as u32;

        let samples: Result<Vec<f32>> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .into_samples::<f32>()
                .map(|s| {
                    s.map_err(|e| DatasetError::AudioError(format!("Failed to read sample: {e}")))
                })
                .collect(),
            hound::SampleFormat::Int => {
                let bits = spec.bits_per_sample;
                let max_value = 2.0_f32.powi(bits as i32 - 1);
                reader
                    .into_samples::<i32>()
                    .map(|s| {
                        s.map(|sample| sample as f32 / max_value).map_err(|e| {
                            DatasetError::AudioError(format!("Failed to read sample: {e}"))
                        })
                    })
                    .collect()
            }
        };

        let samples = samples?;
        Ok(AudioData::new(samples, sample_rate, channels))
    }

    /// Load MP3 audio file
    fn load_mp3_audio(&self, entry: &CommonVoiceEntry) -> Result<AudioData> {
        let file_data = std::fs::read(&entry.audio_path).map_err(|e| {
            DatasetError::IoError(std::io::Error::new(
                e.kind(),
                format!("Failed to read MP3 file {:?}: {}", entry.audio_path, e),
            ))
        })?;

        let mut decoder = minimp3::Decoder::new(&file_data[..]);
        let mut all_samples = Vec::new();
        let mut sample_rate = 0u32;
        let mut channels = 0u32;

        loop {
            match decoder.next_frame() {
                Ok(frame) => {
                    sample_rate = frame.sample_rate as u32;
                    channels = frame.channels as u32;

                    // Convert i16 samples to f32 [-1.0, 1.0]
                    for sample in frame.data {
                        all_samples.push(sample as f32 / 32768.0);
                    }
                }
                Err(minimp3::Error::Eof) => break,
                Err(e) => {
                    return Err(DatasetError::AudioError(format!(
                        "Failed to decode MP3 frame: {e:?}"
                    )))
                }
            }
        }

        if all_samples.is_empty() {
            return Err(DatasetError::AudioError(format!(
                "No audio data in MP3 file: {:?}",
                entry.audio_path
            )));
        }

        Ok(AudioData::new(all_samples, sample_rate, channels))
    }

    /// Get or create speaker info for a client ID
    fn get_speaker_info(&self, entry: &CommonVoiceEntry) -> SpeakerInfo {
        let mut cache = self.speaker_cache.lock().unwrap();

        if let Some(info) = cache.get(&entry.client_id) {
            return info.clone();
        }

        // Create speaker info from CommonVoice metadata
        let mut metadata = HashMap::new();
        if !entry.locale.is_empty() {
            metadata.insert("locale".to_string(), entry.locale.clone());
        }

        let age = entry.age.as_ref().and_then(|a| {
            if a.is_empty() {
                None
            } else {
                // CommonVoice provides age ranges like "twenties", "thirties"
                // Try to extract approximate age
                match a.to_lowercase().as_str() {
                    "teens" => Some(15),
                    "twenties" => Some(25),
                    "thirties" => Some(35),
                    "fourties" => Some(45),
                    "fifties" => Some(55),
                    "sixties" => Some(65),
                    "seventies" => Some(75),
                    "eighties" => Some(85),
                    "nineties" => Some(95),
                    _ => a.parse().ok(),
                }
            }
        });

        let info = SpeakerInfo {
            id: entry.client_id.clone(),
            name: Some(format!("Client {}", entry.client_id)),
            gender: entry.gender.clone(),
            age,
            accent: entry.accent.clone(),
            metadata,
        };

        cache.insert(entry.client_id.clone(), info.clone());
        info
    }

    /// Load sample by index
    pub async fn load_sample(&self, index: usize) -> Result<DatasetSample> {
        let entry = self.get_entry(index)?.clone();

        // Load audio
        let audio = self.load_audio(&entry)?;

        // Get speaker info
        let speaker = self.get_speaker_info(&entry);

        // Calculate quality metrics from voting
        let vote_quality = entry.vote_quality();
        let quality = QualityMetrics {
            snr: Some(20.0 + (vote_quality * 20.0)), // 20-40 dB SNR based on votes
            clipping: Some((1.0 - vote_quality) * 0.1), // Less clipping for higher quality
            dynamic_range: Some(35.0),
            spectral_quality: Some(vote_quality),
            overall_quality: Some(vote_quality),
        };

        // Create metadata
        let mut metadata = HashMap::new();
        metadata.insert(
            "up_votes".to_string(),
            serde_json::Value::Number(entry.up_votes.into()),
        );
        metadata.insert(
            "down_votes".to_string(),
            serde_json::Value::Number(entry.down_votes.into()),
        );
        metadata.insert(
            "locale".to_string(),
            serde_json::Value::String(entry.locale.clone()),
        );
        metadata.insert(
            "split".to_string(),
            serde_json::Value::String(format!("{:?}", entry.split)),
        );

        Ok(DatasetSample {
            id: entry.id(),
            text: entry.sentence,
            audio,
            speaker: Some(speaker),
            language: self.language,
            quality,
            phonemes: None,
            metadata,
        })
    }

    /// Get all unique client/speaker IDs
    pub fn client_ids(&self) -> Vec<String> {
        let mut clients: Vec<String> = self
            .entries
            .iter()
            .map(|e| e.client_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        clients.sort();
        clients
    }

    /// Get number of unique speakers
    pub fn num_speakers(&self) -> usize {
        self.entries
            .iter()
            .map(|e| e.client_id.as_str())
            .collect::<std::collections::HashSet<_>>()
            .len()
    }

    /// Filter entries by validation status
    pub fn validated_entries(&self) -> Vec<&CommonVoiceEntry> {
        self.entries.iter().filter(|e| e.is_validated()).collect()
    }

    /// Get statistics about voting
    pub fn vote_statistics(&self) -> VoteStatistics {
        let total = self.entries.len();
        let validated = self.validated_entries().len();
        let avg_upvotes =
            self.entries.iter().map(|e| e.up_votes).sum::<u32>() as f32 / total.max(1) as f32;
        let avg_downvotes =
            self.entries.iter().map(|e| e.down_votes).sum::<u32>() as f32 / total.max(1) as f32;

        VoteStatistics {
            total_entries: total,
            validated_entries: validated,
            average_upvotes: avg_upvotes,
            average_downvotes: avg_downvotes,
        }
    }
}

/// Vote statistics for CommonVoice dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteStatistics {
    /// Total number of entries
    pub total_entries: usize,
    /// Number of validated entries (>=2 upvotes)
    pub validated_entries: usize,
    /// Average upvotes per entry
    pub average_upvotes: f32,
    /// Average downvotes per entry
    pub average_downvotes: f32,
}

// Implement Dataset trait for CommonVoiceDataset
#[async_trait]
impl Dataset for CommonVoiceDataset {
    type Sample = crate::DatasetSample;

    fn len(&self) -> usize {
        self.entries.len()
    }

    async fn get(&self, index: usize) -> Result<Self::Sample> {
        self.load_sample(index).await
    }

    fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    async fn statistics(&self) -> Result<DatasetStatistics> {
        let mut language_distribution = HashMap::new();
        language_distribution.insert(self.language, self.len());

        let mut speaker_distribution = HashMap::new();
        for entry in &self.entries {
            *speaker_distribution
                .entry(entry.client_id.clone())
                .or_insert(0) += 1;
        }

        Ok(DatasetStatistics {
            total_items: self.len(),
            total_duration: 0.0, // Expensive to calculate
            average_duration: 0.0,
            language_distribution,
            speaker_distribution,
            text_length_stats: LengthStatistics {
                min: 0,
                max: 0,
                mean: 0.0,
                median: 0,
                std_dev: 0.0,
            },
            duration_stats: DurationStatistics {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
            },
        })
    }

    async fn validate(&self) -> Result<ValidationReport> {
        let validated = self.validated_entries().len();
        Ok(ValidationReport {
            is_valid: true,
            errors: vec![],
            warnings: if validated < self.len() {
                vec![format!(
                    "Only {} out of {} entries have sufficient validation (>=2 upvotes)",
                    validated,
                    self.len()
                )]
            } else {
                vec![]
            },
            items_validated: self.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_tsv_name() {
        assert_eq!(CommonVoiceSplit::Train.tsv_name(), "train.tsv");
        assert_eq!(CommonVoiceSplit::Dev.tsv_name(), "dev.tsv");
        assert_eq!(CommonVoiceSplit::Validated.tsv_name(), "validated.tsv");
    }

    #[test]
    fn test_split_from_str() {
        assert!(matches!(
            CommonVoiceSplit::parse("train").unwrap(),
            CommonVoiceSplit::Train
        ));
        assert!(matches!(
            CommonVoiceSplit::parse("dev").unwrap(),
            CommonVoiceSplit::Dev
        ));
        assert!(matches!(
            CommonVoiceSplit::parse("development").unwrap(),
            CommonVoiceSplit::Dev
        ));
        assert!(CommonVoiceSplit::parse("invalid").is_err());
    }

    #[test]
    fn test_entry_vote_quality() {
        let entry = CommonVoiceEntry {
            client_id: "test".to_string(),
            audio_path: PathBuf::from("test.mp3"),
            sentence: "Test sentence".to_string(),
            up_votes: 3,
            down_votes: 1,
            age: None,
            gender: None,
            accent: None,
            locale: "en".to_string(),
            split: CommonVoiceSplit::Train,
        };

        assert_eq!(entry.vote_quality(), 0.75); // 3/(3+1) = 0.75
    }

    #[test]
    fn test_entry_is_validated() {
        let mut entry = CommonVoiceEntry {
            client_id: "test".to_string(),
            audio_path: PathBuf::from("test.mp3"),
            sentence: "Test".to_string(),
            up_votes: 1,
            down_votes: 0,
            age: None,
            gender: None,
            accent: None,
            locale: "en".to_string(),
            split: CommonVoiceSplit::Train,
        };

        assert!(!entry.is_validated()); // Only 1 upvote

        entry.up_votes = 2;
        assert!(entry.is_validated()); // 2 upvotes
    }

    #[test]
    fn test_parse_language_code() {
        assert!(matches!(
            CommonVoiceDataset::parse_language_code("en").unwrap(),
            LanguageCode::EnUs
        ));
        assert!(matches!(
            CommonVoiceDataset::parse_language_code("ja").unwrap(),
            LanguageCode::Ja
        ));
        assert!(matches!(
            CommonVoiceDataset::parse_language_code("zh-CN").unwrap(),
            LanguageCode::ZhCn
        ));
    }

    #[test]
    fn test_entry_id_generation() {
        let entry = CommonVoiceEntry {
            client_id: "abc123".to_string(),
            audio_path: PathBuf::from("/path/to/common_voice_en_12345.mp3"),
            sentence: "Test".to_string(),
            up_votes: 2,
            down_votes: 0,
            age: None,
            gender: None,
            accent: None,
            locale: "en".to_string(),
            split: CommonVoiceSplit::Train,
        };

        let id = entry.id();
        assert!(id.starts_with("cv_abc123_"));
        assert!(id.contains("common_voice_en_12345"));
    }
}
