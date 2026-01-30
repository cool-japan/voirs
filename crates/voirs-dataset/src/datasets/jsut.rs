//! JSUT (Japanese Speech corpus of Saruwatari-lab, University of Tokyo) dataset loader
//!
//! JSUT is a free Japanese speech corpus for TTS research, consisting of clean studio
//! recordings by a single female speaker. The dataset contains multiple subsets with
//! various types of speech content.
//!
//! # Dataset Structure
//!
//! ```text
//! jsut_ver1.1/
//! ├── basic5000/           # 5000 basic sentences
//! │   ├── wav/
//! │   │   ├── BASIC5000_0001.wav
//! │   │   └── ...
//! │   └── transcript_utf8.txt
//! ├── onomatopeia300/      # 300 onomatopoeia sentences
//! │   ├── wav/
//! │   └── transcript_utf8.txt
//! ├── loanword128/         # 128 sentences with loanwords
//! ├── countersuffix26/     # 26 sentences with counter suffixes
//! ├── precedent130/        # 130 sentences from precedent
//! ├── repeat500/           # 500 repeated sentences
//! ├── travel1000/          # 1000 travel domain sentences
//! ├── voiceactress100/     # 100 voice actress sentences
//! └── utparaphrase512/     # 512 paraphrased sentences
//! ```
//!
//! # Example Usage
//!
//! ```no_run
//! use voirs_dataset::datasets::jsut::{JsutDataset, JsutSubset};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Load the basic5000 subset
//! let dataset = JsutDataset::new("path/to/jsut_ver1.1", Some(JsutSubset::Basic5000)).await?;
//!
//! // Load all subsets
//! let full_dataset = JsutDataset::new("path/to/jsut_ver1.1", None).await?;
//!
//! println!("Dataset has {} utterances", dataset.len());
//! # Ok(())
//! # }
//! ```

use crate::{
    traits::{Dataset, DatasetMetadata},
    AudioData, DatasetError, DatasetSample, DatasetStatistics, DurationStatistics, LanguageCode,
    LengthStatistics, Result, SpeakerInfo, ValidationReport,
};
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// JSUT dataset subsets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JsutSubset {
    /// 5000 basic sentences (main corpus)
    Basic5000,
    /// 300 onomatopoeia sentences
    Onomatopeia300,
    /// 128 sentences with loanwords
    Loanword128,
    /// 26 sentences with counter suffixes
    Countersuffix26,
    /// 130 sentences from precedent
    Precedent130,
    /// 500 repeated sentences
    Repeat500,
    /// 1000 travel domain sentences
    Travel1000,
    /// 100 voice actress sentences
    Voiceactress100,
    /// 512 paraphrased sentences
    Utparaphrase512,
}

impl JsutSubset {
    /// Get the directory name for this subset
    pub fn dir_name(&self) -> &'static str {
        match self {
            Self::Basic5000 => "basic5000",
            Self::Onomatopeia300 => "onomatopeia300",
            Self::Loanword128 => "loanword128",
            Self::Countersuffix26 => "countersuffix26",
            Self::Precedent130 => "precedent130",
            Self::Repeat500 => "repeat500",
            Self::Travel1000 => "travel1000",
            Self::Voiceactress100 => "voiceactress100",
            Self::Utparaphrase512 => "utparaphrase512",
        }
    }

    /// Get all available subsets
    pub fn all() -> Vec<Self> {
        vec![
            Self::Basic5000,
            Self::Onomatopeia300,
            Self::Loanword128,
            Self::Countersuffix26,
            Self::Precedent130,
            Self::Repeat500,
            Self::Travel1000,
            Self::Voiceactress100,
            Self::Utparaphrase512,
        ]
    }

    /// Parse subset from directory name
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "basic5000" => Some(Self::Basic5000),
            "onomatopeia300" => Some(Self::Onomatopeia300),
            "loanword128" => Some(Self::Loanword128),
            "countersuffix26" => Some(Self::Countersuffix26),
            "precedent130" => Some(Self::Precedent130),
            "repeat500" => Some(Self::Repeat500),
            "travel1000" => Some(Self::Travel1000),
            "voiceactress100" => Some(Self::Voiceactress100),
            "utparaphrase512" => Some(Self::Utparaphrase512),
            _ => None,
        }
    }
}

/// Single entry in the JSUT dataset
#[derive(Debug, Clone)]
pub struct JsutEntry {
    /// Unique identifier (e.g., "BASIC5000_0001")
    pub id: String,
    /// Path to the audio file
    pub audio_path: PathBuf,
    /// Transcription text
    pub text: String,
    /// Subset this entry belongs to
    pub subset: JsutSubset,
}

/// JSUT dataset loader
///
/// Loads the JSUT (Japanese Speech corpus of Saruwatari-lab, University of Tokyo) dataset.
/// This is a single-speaker Japanese TTS corpus with multiple subsets.
#[derive(Debug, Clone)]
pub struct JsutDataset {
    root: PathBuf,
    entries: Vec<JsutEntry>,
    speaker_cache: Arc<Mutex<HashMap<String, SpeakerInfo>>>,
    subset_filter: Option<JsutSubset>,
    metadata: DatasetMetadata,
}

impl JsutDataset {
    /// Create a new JSUT dataset loader
    ///
    /// # Arguments
    ///
    /// * `root` - Root directory of the JSUT dataset (e.g., "jsut_ver1.1/")
    /// * `subset` - Optional subset to load. If None, loads all subsets.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use voirs_dataset::datasets::jsut::{JsutDataset, JsutSubset};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// // Load only basic5000 subset
    /// let dataset = JsutDataset::new("jsut_ver1.1", Some(JsutSubset::Basic5000)).await?;
    ///
    /// // Load all subsets
    /// let full = JsutDataset::new("jsut_ver1.1", None).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new<P: AsRef<Path>>(root: P, subset: Option<JsutSubset>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();

        if !root.exists() {
            return Err(DatasetError::LoadError(format!(
                "JSUT dataset directory not found: {}",
                root.display()
            )));
        }

        let subsets_to_load = if let Some(s) = subset {
            vec![s]
        } else {
            JsutSubset::all()
        };

        let mut all_entries = Vec::new();

        for subset in subsets_to_load {
            let subset_dir = root.join(subset.dir_name());
            if !subset_dir.exists() {
                tracing::warn!(
                    "Subset directory not found: {} (skipping)",
                    subset_dir.display()
                );
                continue;
            }

            let transcript_path = subset_dir.join("transcript_utf8.txt");
            if !transcript_path.exists() {
                tracing::warn!(
                    "Transcript file not found: {} (skipping)",
                    transcript_path.display()
                );
                continue;
            }

            let entries = Self::load_subset(&subset_dir, subset).await?;
            all_entries.extend(entries);
        }

        if all_entries.is_empty() {
            return Err(DatasetError::LoadError(
                "No valid entries found in JSUT dataset".to_string(),
            ));
        }

        let speaker_cache = Arc::new(Mutex::new(HashMap::new()));

        // Create metadata
        let metadata = DatasetMetadata {
            name: "JSUT".to_string(),
            version: "1.1".to_string(),
            description: Some(
                "Japanese Speech corpus of Saruwatari-lab, University of Tokyo".to_string(),
            ),
            total_samples: all_entries.len(),
            total_duration: 0.0, // Expensive to calculate
            languages: vec!["ja".to_string()],
            speakers: vec!["jsut_speaker".to_string()],
            license: Some("CC BY-SA 4.0".to_string()),
            metadata: HashMap::new(),
        };

        Ok(Self {
            root,
            entries: all_entries,
            speaker_cache,
            subset_filter: subset,
            metadata,
        })
    }

    /// Load a single subset
    async fn load_subset(subset_dir: &Path, subset: JsutSubset) -> Result<Vec<JsutEntry>> {
        let transcript_path = subset_dir.join("transcript_utf8.txt");
        let wav_dir = subset_dir.join("wav");

        if !wav_dir.exists() {
            return Err(DatasetError::LoadError(format!(
                "WAV directory not found: {}",
                wav_dir.display()
            )));
        }

        let transcript_content =
            tokio::fs::read_to_string(&transcript_path)
                .await
                .map_err(|e| {
                    DatasetError::LoadError(format!(
                        "Failed to read transcript file {}: {}",
                        transcript_path.display(),
                        e
                    ))
                })?;

        let mut entries = Vec::new();

        for line in transcript_content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Format: "BASIC5000_0001:これは例文です。"
            let parts: Vec<&str> = line.splitn(2, ':').collect();
            if parts.len() != 2 {
                tracing::warn!("Invalid transcript line format: {}", line);
                continue;
            }

            let id = parts[0].trim().to_string();
            let text = parts[1].trim().to_string();

            let audio_path = wav_dir.join(format!("{}.wav", id));
            if !audio_path.exists() {
                tracing::warn!("Audio file not found: {} (skipping)", audio_path.display());
                continue;
            }

            entries.push(JsutEntry {
                id,
                audio_path,
                text,
                subset,
            });
        }

        Ok(entries)
    }

    /// Get an entry by index
    fn get_entry(&self, index: usize) -> Result<&JsutEntry> {
        self.entries
            .get(index)
            .ok_or(DatasetError::IndexError(index))
    }

    /// Load a sample from an entry
    async fn load_sample(&self, index: usize) -> Result<DatasetSample> {
        let entry = self.get_entry(index)?.clone();

        // Load audio file
        let audio_data = tokio::fs::read(&entry.audio_path).await.map_err(|e| {
            DatasetError::LoadError(format!(
                "Failed to read audio file {}: {}",
                entry.audio_path.display(),
                e
            ))
        })?;

        // Parse WAV file
        let mut reader = hound::WavReader::new(std::io::Cursor::new(&audio_data)).map_err(|e| {
            DatasetError::FormatError(format!(
                "Failed to parse WAV file {}: {}",
                entry.audio_path.display(),
                e
            ))
        })?;

        let spec = reader.spec();
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.unwrap_or(0) as f32 / i16::MAX as f32)
            .collect();

        // Create AudioData
        let mut audio = AudioData::new(samples, spec.sample_rate, spec.channels as u32);
        audio.add_metadata("subset".to_string(), entry.subset.dir_name().to_string());
        audio.add_metadata("dataset".to_string(), "jsut".to_string());
        audio.add_metadata("version".to_string(), "1.1".to_string());

        // Get or create speaker info
        let speaker_id = "jsut_speaker".to_string();
        let speaker_info = {
            let mut cache = self.speaker_cache.lock().map_err(|e| {
                DatasetError::LoadError(format!("Failed to lock speaker cache: {}", e))
            })?;
            cache
                .entry(speaker_id.clone())
                .or_insert_with(|| SpeakerInfo {
                    id: speaker_id.clone(),
                    name: Some("JSUT Speaker".to_string()),
                    gender: Some("female".to_string()),
                    age: None,
                    accent: None,
                    metadata: HashMap::new(),
                })
                .clone()
        };

        Ok(DatasetSample {
            id: entry.id,
            text: entry.text,
            audio,
            speaker: Some(speaker_info),
            language: LanguageCode::Ja,
            quality: crate::QualityMetrics::default(),
            phonemes: None,
            metadata: HashMap::new(),
        })
    }

    /// Get the dataset metadata
    pub fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the dataset is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get all unique speaker IDs
    pub fn speakers(&self) -> Vec<String> {
        vec!["jsut_speaker".to_string()]
    }

    /// Filter entries by subset
    pub fn filter_subset(&self, subset: JsutSubset) -> Vec<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.subset == subset)
            .map(|(i, _)| i)
            .collect()
    }
}

#[async_trait]
impl Dataset for JsutDataset {
    type Sample = DatasetSample;

    fn len(&self) -> usize {
        self.entries.len()
    }

    async fn get(&self, index: usize) -> Result<Self::Sample> {
        self.load_sample(index).await
    }

    fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    async fn statistics(&self) -> Result<crate::DatasetStatistics> {
        use crate::{DatasetStatistics, DurationStatistics, LengthStatistics};

        let mut language_distribution = HashMap::new();
        language_distribution.insert(LanguageCode::Ja, self.len());

        let mut speaker_distribution = HashMap::new();
        speaker_distribution.insert("jsut_speaker".to_string(), self.len());

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

    async fn validate(&self) -> Result<crate::ValidationReport> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        for (i, entry) in self.entries.iter().enumerate() {
            // Check if audio file exists
            if !entry.audio_path.exists() {
                errors.push(format!(
                    "Entry {}: Audio file not found: {}",
                    entry.id,
                    entry.audio_path.display()
                ));
                continue;
            }

            // Check if text is not empty
            if entry.text.trim().is_empty() {
                warnings.push(format!("Entry {}: Empty transcription text", entry.id));
            }

            // Try to load the audio file
            if let Err(e) = self.load_sample(i).await {
                errors.push(format!("Entry {}: Failed to load audio: {}", entry.id, e));
            }
        }

        let is_valid = errors.is_empty();

        Ok(crate::ValidationReport {
            is_valid,
            errors,
            warnings,
            items_validated: self.entries.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subset_dir_names() {
        assert_eq!(JsutSubset::Basic5000.dir_name(), "basic5000");
        assert_eq!(JsutSubset::Onomatopeia300.dir_name(), "onomatopeia300");
        assert_eq!(JsutSubset::Travel1000.dir_name(), "travel1000");
    }

    #[test]
    fn test_subset_parse() {
        assert_eq!(JsutSubset::parse("basic5000"), Some(JsutSubset::Basic5000));
        assert_eq!(
            JsutSubset::parse("onomatopeia300"),
            Some(JsutSubset::Onomatopeia300)
        );
        assert_eq!(JsutSubset::parse("invalid"), None);
    }

    #[test]
    fn test_all_subsets() {
        let all = JsutSubset::all();
        assert_eq!(all.len(), 9);
        assert!(all.contains(&JsutSubset::Basic5000));
        assert!(all.contains(&JsutSubset::Travel1000));
    }

    #[tokio::test]
    async fn test_jsut_dataset_not_found() {
        let result = JsutDataset::new("/nonexistent/path", None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_jsut_entry_creation() {
        let entry = JsutEntry {
            id: "BASIC5000_0001".to_string(),
            audio_path: PathBuf::from("test.wav"),
            text: "これはテストです。".to_string(),
            subset: JsutSubset::Basic5000,
        };

        assert_eq!(entry.id, "BASIC5000_0001");
        assert_eq!(entry.subset, JsutSubset::Basic5000);
        assert!(!entry.text.is_empty());
    }

    #[tokio::test]
    async fn test_jsut_metadata() {
        // Create a temporary directory structure
        let temp_dir = std::env::temp_dir().join(format!("jsut_test_{}", fastrand::u64(..)));
        let basic_dir = temp_dir.join("basic5000");
        let wav_dir = basic_dir.join("wav");

        std::fs::create_dir_all(&wav_dir).unwrap();

        // Create a transcript file
        let transcript_path = basic_dir.join("transcript_utf8.txt");
        std::fs::write(&transcript_path, "BASIC5000_0001:テストです。\n").unwrap();

        // Create a dummy WAV file
        let wav_path = wav_dir.join("BASIC5000_0001.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
        for _ in 0..24000 {
            writer.write_sample(0i16).unwrap();
        }
        writer.finalize().unwrap();

        let dataset = JsutDataset::new(&temp_dir, Some(JsutSubset::Basic5000))
            .await
            .unwrap();

        let metadata = dataset.metadata();
        assert_eq!(metadata.name, "JSUT");
        assert_eq!(metadata.version, "1.1");
        assert_eq!(metadata.languages, vec!["ja".to_string()]);
        assert_eq!(metadata.speakers.len(), 1);
        assert_eq!(metadata.speakers[0], "jsut_speaker");

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[tokio::test]
    async fn test_jsut_speakers() {
        let temp_dir = std::env::temp_dir().join(format!("jsut_test_{}", fastrand::u64(..)));
        let basic_dir = temp_dir.join("basic5000");
        let wav_dir = basic_dir.join("wav");

        std::fs::create_dir_all(&wav_dir).unwrap();

        let transcript_path = basic_dir.join("transcript_utf8.txt");
        std::fs::write(&transcript_path, "BASIC5000_0001:テスト\n").unwrap();

        let wav_path = wav_dir.join("BASIC5000_0001.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
        for _ in 0..1000 {
            writer.write_sample(0i16).unwrap();
        }
        writer.finalize().unwrap();

        let dataset = JsutDataset::new(&temp_dir, None).await.unwrap();

        let speakers = dataset.speakers();
        assert_eq!(speakers.len(), 1);
        assert_eq!(speakers[0], "jsut_speaker");

        std::fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[tokio::test]
    async fn test_jsut_filter_subset() {
        let temp_dir = std::env::temp_dir().join(format!("jsut_test_{}", fastrand::u64(..)));

        // Create two subsets
        for subset_name in &["basic5000", "travel1000"] {
            let subset_dir = temp_dir.join(subset_name);
            let wav_dir = subset_dir.join("wav");
            std::fs::create_dir_all(&wav_dir).unwrap();

            let transcript_path = subset_dir.join("transcript_utf8.txt");
            let prefix = if *subset_name == "basic5000" {
                "BASIC5000"
            } else {
                "TRAVEL1000"
            };
            std::fs::write(&transcript_path, format!("{}_0001:テスト\n", prefix)).unwrap();

            let wav_path = wav_dir.join(format!("{}_0001.wav", prefix));
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 24000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
            for _ in 0..1000 {
                writer.write_sample(0i16).unwrap();
            }
            writer.finalize().unwrap();
        }

        let dataset = JsutDataset::new(&temp_dir, None).await.unwrap();
        assert_eq!(dataset.len(), 2);

        let basic_indices = dataset.filter_subset(JsutSubset::Basic5000);
        assert_eq!(basic_indices.len(), 1);

        let travel_indices = dataset.filter_subset(JsutSubset::Travel1000);
        assert_eq!(travel_indices.len(), 1);

        std::fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[tokio::test]
    async fn test_jsut_load_sample() {
        let temp_dir = std::env::temp_dir().join(format!("jsut_test_{}", fastrand::u64(..)));
        let basic_dir = temp_dir.join("basic5000");
        let wav_dir = basic_dir.join("wav");

        std::fs::create_dir_all(&wav_dir).unwrap();

        let transcript_path = basic_dir.join("transcript_utf8.txt");
        std::fs::write(&transcript_path, "BASIC5000_0001:これはテストです。\n").unwrap();

        let wav_path = wav_dir.join("BASIC5000_0001.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
        for i in 0..24000 {
            writer.write_sample((i % 1000) as i16).unwrap();
        }
        writer.finalize().unwrap();

        let dataset = JsutDataset::new(&temp_dir, None).await.unwrap();
        let sample = dataset.get(0).await.unwrap();

        assert_eq!(sample.id, "BASIC5000_0001");
        assert_eq!(sample.text, "これはテストです。");
        assert_eq!(sample.audio.sample_rate(), 24000);
        assert_eq!(sample.audio.samples().len(), 24000);
        assert_eq!(sample.language, LanguageCode::Ja);
        assert!(sample.speaker.is_some());

        std::fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[tokio::test]
    async fn test_jsut_dataset_trait_len() {
        let temp_dir = std::env::temp_dir().join(format!("jsut_test_{}", fastrand::u64(..)));
        let basic_dir = temp_dir.join("basic5000");
        let wav_dir = basic_dir.join("wav");

        std::fs::create_dir_all(&wav_dir).unwrap();

        let transcript_path = basic_dir.join("transcript_utf8.txt");
        std::fs::write(
            &transcript_path,
            "BASIC5000_0001:テスト1\nBASIC5000_0002:テスト2\n",
        )
        .unwrap();

        for id in &["BASIC5000_0001", "BASIC5000_0002"] {
            let wav_path = wav_dir.join(format!("{}.wav", id));
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 24000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
            for _ in 0..1000 {
                writer.write_sample(0i16).unwrap();
            }
            writer.finalize().unwrap();
        }

        let dataset = JsutDataset::new(&temp_dir, None).await.unwrap();
        assert_eq!(dataset.len(), 2);
        assert!(!dataset.is_empty());

        std::fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[tokio::test]
    async fn test_jsut_statistics() {
        let temp_dir = std::env::temp_dir().join(format!("jsut_test_{}", fastrand::u64(..)));
        let basic_dir = temp_dir.join("basic5000");
        let wav_dir = basic_dir.join("wav");

        std::fs::create_dir_all(&wav_dir).unwrap();

        let transcript_path = basic_dir.join("transcript_utf8.txt");
        std::fs::write(
            &transcript_path,
            "BASIC5000_0001:短い\nBASIC5000_0002:これは少し長いテキストです\n",
        )
        .unwrap();

        for id in &["BASIC5000_0001", "BASIC5000_0002"] {
            let wav_path = wav_dir.join(format!("{}.wav", id));
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 24000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
            for _ in 0..24000 {
                writer.write_sample(0i16).unwrap();
            }
            writer.finalize().unwrap();
        }

        let dataset = JsutDataset::new(&temp_dir, None).await.unwrap();
        let stats = dataset.statistics().await.unwrap();

        assert_eq!(stats.total_items, 2);
        assert_eq!(stats.speaker_distribution.len(), 1);
        assert_eq!(stats.language_distribution.get(&LanguageCode::Ja), Some(&2));

        std::fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[tokio::test]
    async fn test_jsut_validation() {
        let temp_dir = std::env::temp_dir().join(format!("jsut_test_{}", fastrand::u64(..)));
        let basic_dir = temp_dir.join("basic5000");
        let wav_dir = basic_dir.join("wav");

        std::fs::create_dir_all(&wav_dir).unwrap();

        let transcript_path = basic_dir.join("transcript_utf8.txt");
        std::fs::write(&transcript_path, "BASIC5000_0001:テスト\n").unwrap();

        let wav_path = wav_dir.join("BASIC5000_0001.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
        for _ in 0..1000 {
            writer.write_sample(0i16).unwrap();
        }
        writer.finalize().unwrap();

        let dataset = JsutDataset::new(&temp_dir, None).await.unwrap();
        let report = dataset.validate().await.unwrap();

        assert!(report.is_valid);
        assert_eq!(report.items_validated, 1);
        assert!(report.errors.is_empty());

        std::fs::remove_dir_all(&temp_dir).unwrap();
    }
}
