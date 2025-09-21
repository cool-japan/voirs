//! JVS (Japanese Versatile Speech) dataset implementation
//!
//! This module provides loading and processing capabilities for the JVS dataset,
//! a multi-speaker Japanese speech corpus.

use crate::splits::{DatasetSplits, SplitConfig};
use crate::traits::{Dataset, DatasetMetadata};
use crate::{
    AudioData, DatasetError, DatasetSample, DatasetStatistics, LanguageCode, Phoneme,
    QualityMetrics, Result, SpeakerInfo, ValidationReport,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// JVS dataset URL
#[allow(dead_code)]
const JVS_URL: &str =
    "https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus";

/// JVS speaker metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JvsSpeakerInfo {
    /// Speaker ID (e.g., "jvs001")
    pub speaker_id: String,
    /// Gender (M/F)
    pub gender: String,
    /// Age
    pub age: Option<u32>,
    /// Region/dialect
    pub region: Option<String>,
    /// Recording environment
    pub environment: Option<String>,
    /// Available sentence types
    pub sentence_types: Vec<String>,
}

/// JVS sentence types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JvsSentenceType {
    /// Parallel sentences (same text across speakers)
    Parallel,
    /// Non-parallel sentences (speaker-specific)
    NonParallel,
    /// Whisper speech
    Whisper,
    /// Falsetto speech
    Falsetto,
    /// Reading-style speech
    Reading,
}

impl JvsSentenceType {
    pub fn from_path(path: &str) -> Option<Self> {
        if path.contains("parallel100") {
            Some(Self::Parallel)
        } else if path.contains("nonpara30") {
            Some(Self::NonParallel)
        } else if path.contains("whisper10") {
            Some(Self::Whisper)
        } else if path.contains("falsetto10") {
            Some(Self::Falsetto)
        } else if path.contains("reading") {
            Some(Self::Reading)
        } else {
            None
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Parallel => "parallel",
            Self::NonParallel => "nonparallel",
            Self::Whisper => "whisper",
            Self::Falsetto => "falsetto",
            Self::Reading => "reading",
        }
    }
}

/// JVS dataset loader
pub struct JvsDataset {
    /// Dataset metadata
    metadata: DatasetMetadata,
    /// Dataset samples
    samples: Vec<DatasetSample>,
    /// Base path to dataset
    #[allow(dead_code)]
    base_path: PathBuf,
    /// Speaker information
    speakers: HashMap<String, JvsSpeakerInfo>,
}

impl JvsDataset {
    /// Load JVS dataset from directory
    pub async fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let base_path = path.as_ref().to_path_buf();

        // Check if dataset exists
        if !base_path.exists() {
            return Err(DatasetError::LoadError(format!(
                "JVS dataset not found at {base_path:?}"
            )));
        }

        // Load speaker metadata
        let speakers = Self::load_speaker_metadata(&base_path)?;

        // Load all samples
        let mut samples = Vec::new();

        for speaker_info in speakers.values() {
            let speaker_samples = Self::load_speaker_samples(&base_path, speaker_info).await?;
            samples.extend(speaker_samples);
        }

        tracing::info!("Loaded {} samples from JVS dataset", samples.len());

        // Calculate total duration
        let total_duration: f32 = samples.iter().map(|s| s.audio.duration()).sum();

        // Create metadata
        let metadata = DatasetMetadata {
            name: "JVS".to_string(),
            version: "1.1".to_string(),
            description: Some("Japanese Versatile Speech corpus for multi-speaker TTS".to_string()),
            total_samples: samples.len(),
            total_duration,
            languages: vec!["ja".to_string()],
            speakers: speakers.keys().cloned().collect(),
            license: Some("CC BY-SA 4.0".to_string()),
            metadata: HashMap::new(),
        };

        Ok(Self {
            metadata,
            samples,
            base_path,
            speakers,
        })
    }

    /// Load speaker metadata from filesystem structure
    fn load_speaker_metadata(base_path: &Path) -> Result<HashMap<String, JvsSpeakerInfo>> {
        let mut speakers = HashMap::new();

        // JVS dataset typically has directories like jvs001, jvs002, etc.
        for entry in std::fs::read_dir(base_path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("jvs") && name.len() == 6 {
                        let speaker_id = name.to_string();

                        // Infer metadata from speaker ID and available directories
                        let speaker_info = Self::infer_speaker_info(&speaker_id, &path)?;
                        speakers.insert(speaker_id.clone(), speaker_info);
                    }
                }
            }
        }

        Ok(speakers)
    }

    /// Infer speaker information from directory structure
    fn infer_speaker_info(speaker_id: &str, speaker_path: &Path) -> Result<JvsSpeakerInfo> {
        // Extract speaker number for metadata inference
        let speaker_num: u32 = speaker_id[3..]
            .parse()
            .map_err(|_| DatasetError::LoadError(format!("Invalid speaker ID: {speaker_id}")))?;

        // Basic metadata inference (this would ideally come from a metadata file)
        let gender = if speaker_num <= 50 { "M" } else { "F" }.to_string();
        let age = Some(20 + (speaker_num % 40)); // Rough age estimate

        // Check available sentence types
        let mut sentence_types = Vec::new();
        if speaker_path.join("parallel100").exists() {
            sentence_types.push("parallel".to_string());
        }
        if speaker_path.join("nonpara30").exists() {
            sentence_types.push("nonparallel".to_string());
        }
        if speaker_path.join("whisper10").exists() {
            sentence_types.push("whisper".to_string());
        }
        if speaker_path.join("falsetto10").exists() {
            sentence_types.push("falsetto".to_string());
        }

        Ok(JvsSpeakerInfo {
            speaker_id: speaker_id.to_string(),
            gender,
            age,
            region: Some("Japan".to_string()),
            environment: Some("studio".to_string()),
            sentence_types,
        })
    }

    /// Load samples for a specific speaker
    async fn load_speaker_samples(
        base_path: &Path,
        speaker_info: &JvsSpeakerInfo,
    ) -> Result<Vec<DatasetSample>> {
        let mut samples = Vec::new();
        let speaker_path = base_path.join(&speaker_info.speaker_id);

        // Iterate through all subdirectories (sentence types)
        #[allow(clippy::redundant_closure_for_method_calls)]
        for entry in WalkDir::new(&speaker_path)
            .max_depth(3)
            .into_iter()
            .filter_map(|result| result.ok())
        {
            let path = entry.path();

            // Look for WAV files
            if path.extension().and_then(|ext| ext.to_str()) == Some("wav") {
                if let Some(sample) = Self::load_sample(path, speaker_info, base_path).await? {
                    samples.push(sample);
                }
            }
        }

        Ok(samples)
    }

    /// Load a single sample from a WAV file
    async fn load_sample(
        audio_path: &Path,
        speaker_info: &JvsSpeakerInfo,
        base_path: &Path,
    ) -> Result<Option<DatasetSample>> {
        // Extract sample information from path
        let relative_path = audio_path
            .strip_prefix(base_path)
            .map_err(|_| DatasetError::LoadError("Invalid audio path".to_string()))?;

        let path_str = relative_path.to_string_lossy();
        let sentence_type = JvsSentenceType::from_path(&path_str);

        // Generate sample ID from path
        let sample_id = audio_path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| DatasetError::LoadError("Invalid audio filename".to_string()))?;

        // Try to load corresponding transcript
        let transcript = Self::load_transcript(audio_path).await.unwrap_or_else(|_| {
            // If no transcript file, generate a placeholder
            format!("Sample {sample_id}")
        });

        // Load audio data
        let audio_data = match Self::load_audio_file(audio_path) {
            Ok(audio) => audio,
            Err(e) => {
                tracing::warn!("Failed to load audio file {:?}: {}", audio_path, e);
                return Ok(None);
            }
        };

        // Create speaker info
        let speaker_id = &speaker_info.speaker_id;
        let speaker = SpeakerInfo {
            id: speaker_info.speaker_id.clone(),
            name: Some(format!("JVS Speaker {speaker_id}")),
            gender: Some(speaker_info.gender.clone()),
            age: speaker_info.age,
            accent: speaker_info.region.clone(),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert(
                    "environment".to_string(),
                    speaker_info.environment.clone().unwrap_or_default(),
                );
                if let Some(sentence_type) = sentence_type {
                    meta.insert(
                        "sentence_type".to_string(),
                        sentence_type.as_str().to_string(),
                    );
                }
                meta
            },
        };

        // Create sample
        let sample = DatasetSample {
            id: format!("{speaker_id}_{sample_id}"),
            text: transcript,
            audio: audio_data,
            speaker: Some(speaker),
            language: LanguageCode::Ja,
            quality: QualityMetrics {
                snr: None,
                clipping: None,
                dynamic_range: None,
                spectral_quality: None,
                overall_quality: None,
            },
            phonemes: Self::load_phonemes(audio_path).await.ok(),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert(
                    "audio_path".to_string(),
                    serde_json::Value::String(audio_path.to_string_lossy().to_string()),
                );
                meta.insert(
                    "speaker_id".to_string(),
                    serde_json::Value::String(speaker_info.speaker_id.clone()),
                );
                if let Some(sentence_type) = sentence_type {
                    meta.insert(
                        "sentence_type".to_string(),
                        serde_json::Value::String(sentence_type.as_str().to_string()),
                    );
                }
                meta
            },
        };

        Ok(Some(sample))
    }

    /// Load transcript for audio file
    async fn load_transcript(audio_path: &Path) -> Result<String> {
        // Try different transcript file extensions
        let base_path = audio_path.with_extension("");

        for ext in &["txt", "lab", "transcript"] {
            let transcript_path = base_path.with_extension(ext);
            if transcript_path.exists() {
                let content = tokio::fs::read_to_string(&transcript_path).await?;
                return Ok(content.trim().to_string());
            }
        }

        Err(DatasetError::LoadError(
            "No transcript file found".to_string(),
        ))
    }

    /// Load phonemes from TextGrid file
    async fn load_phonemes(audio_path: &Path) -> Result<Vec<Phoneme>> {
        let base_path = audio_path.with_extension("");

        // Try different TextGrid file extensions
        for ext in &["TextGrid", "textgrid", "tg"] {
            let textgrid_path = base_path.with_extension(ext);
            if textgrid_path.exists() {
                let content = tokio::fs::read_to_string(&textgrid_path).await?;
                return Self::parse_textgrid(&content);
            }
        }

        Err(DatasetError::LoadError(
            "No TextGrid file found".to_string(),
        ))
    }

    /// Parse TextGrid file content into phonemes
    fn parse_textgrid(content: &str) -> Result<Vec<Phoneme>> {
        let mut phonemes = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        let mut i = 0;
        let mut in_intervals = false;
        let mut current_start: Option<f32> = None;
        let mut current_end: Option<f32> = None;

        while i < lines.len() {
            let line = lines[i].trim();

            // Look for interval tier sections
            if line.contains("intervals [") {
                in_intervals = true;
                i += 1;
                continue;
            }

            // Exit interval parsing when we reach another section
            if in_intervals && (line.starts_with("item [") || line.starts_with("size =")) {
                in_intervals = false;
                i += 1;
                continue;
            }

            if in_intervals {
                // Parse xmin (start time)
                if line.starts_with("xmin =") {
                    if let Some(time_str) = line.split('=').nth(1) {
                        current_start = time_str.trim().parse::<f32>().ok();
                    }
                }
                // Parse xmax (end time)
                else if line.starts_with("xmax =") {
                    if let Some(time_str) = line.split('=').nth(1) {
                        current_end = time_str.trim().parse::<f32>().ok();
                    }
                }
                // Parse text (phoneme symbol)
                else if line.starts_with("text =") {
                    if let Some(text_part) = line.split('=').nth(1) {
                        let symbol = text_part.trim().trim_matches('"').trim();

                        // Only add non-empty phonemes
                        if !symbol.is_empty() && symbol != "sp" && symbol != "sil" {
                            if let (Some(start), Some(end)) = (current_start, current_end) {
                                let duration = end - start;

                                // Create phoneme with duration
                                let phoneme = Phoneme {
                                    symbol: symbol.to_string(),
                                    features: None,
                                    duration: Some(duration),
                                };

                                phonemes.push(phoneme);
                            }
                        }
                    }

                    // Reset for next interval
                    current_start = None;
                    current_end = None;
                }
            }

            i += 1;
        }

        if phonemes.is_empty() {
            // Try simpler parsing for lab files or other formats
            Self::parse_simple_phoneme_file(content)
        } else {
            Ok(phonemes)
        }
    }

    /// Parse simple phoneme files (e.g., .lab format)
    fn parse_simple_phoneme_file(content: &str) -> Result<Vec<Phoneme>> {
        let mut phonemes = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();

            // Handle different formats:
            // Format 1: start_time end_time phoneme
            // Format 2: phoneme
            match parts.len() {
                3 => {
                    if let (Ok(start), Ok(end)) = (parts[0].parse::<f32>(), parts[1].parse::<f32>())
                    {
                        let symbol = parts[2].to_string();
                        let duration = end - start;

                        if !symbol.is_empty() && symbol != "sp" && symbol != "sil" {
                            phonemes.push(Phoneme {
                                symbol,
                                features: None,
                                duration: Some(duration),
                            });
                        }
                    }
                }
                1 => {
                    let symbol = parts[0].to_string();
                    if !symbol.is_empty() && symbol != "sp" && symbol != "sil" {
                        phonemes.push(Phoneme {
                            symbol,
                            features: None,
                            duration: None,
                        });
                    }
                }
                _ => {
                    // Skip malformed lines
                    continue;
                }
            }
        }

        if phonemes.is_empty() {
            Err(DatasetError::FormatError(
                "No valid phonemes found in file".to_string(),
            ))
        } else {
            Ok(phonemes)
        }
    }

    /// Load audio file from path
    fn load_audio_file(path: &Path) -> Result<AudioData> {
        crate::audio::io::load_wav(path)
    }

    /// Get speaker information
    pub fn get_speaker_info(&self, speaker_id: &str) -> Option<&JvsSpeakerInfo> {
        self.speakers.get(speaker_id)
    }

    /// Get all speaker IDs
    pub fn speaker_ids(&self) -> Vec<&str> {
        self.speakers.keys().map(String::as_str).collect()
    }

    /// Filter samples by speaker
    pub fn filter_by_speaker(&self, speaker_id: &str) -> Vec<&DatasetSample> {
        self.samples
            .iter()
            .filter(|sample| {
                sample
                    .speaker
                    .as_ref()
                    .map(|s| s.id == speaker_id)
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Filter samples by sentence type
    pub fn filter_by_sentence_type(&self, sentence_type: JvsSentenceType) -> Vec<&DatasetSample> {
        self.samples
            .iter()
            .filter(|sample| {
                sample
                    .metadata
                    .get("sentence_type")
                    .and_then(|v| v.as_str())
                    .map(|s| s == sentence_type.as_str())
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Get parallel sentences (same text across speakers)
    pub fn get_parallel_sentences(&self) -> HashMap<String, Vec<&DatasetSample>> {
        let mut parallel_sentences: HashMap<String, Vec<&DatasetSample>> = HashMap::new();

        for sample in self.filter_by_sentence_type(JvsSentenceType::Parallel) {
            // Extract sentence ID from sample ID (assuming format like "jvs001_BASIC5000_0001")
            if let Some(sentence_id) = sample.id.split('_').nth(2) {
                parallel_sentences
                    .entry(sentence_id.to_string())
                    .or_default()
                    .push(sample);
            }
        }

        parallel_sentences
    }

    /// Create splits with speaker-aware stratification
    pub fn create_speaker_aware_splits(&self, config: SplitConfig) -> Result<DatasetSplits> {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        use std::collections::BTreeMap;

        // Group samples by speaker
        let mut speaker_samples: BTreeMap<String, Vec<usize>> = BTreeMap::new();

        for (index, sample) in self.samples.iter().enumerate() {
            if let Some(speaker) = &sample.speaker {
                speaker_samples
                    .entry(speaker.id.clone())
                    .or_default()
                    .push(index);
            } else {
                // Handle samples without speaker info by assigning to a default speaker
                speaker_samples
                    .entry("unknown".to_string())
                    .or_default()
                    .push(index);
            }
        }

        // Get list of speakers
        let mut speakers: Vec<String> = speaker_samples.keys().cloned().collect();

        // Shuffle speakers based on seed
        if let Some(seed) = config.seed {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            speakers.shuffle(&mut rng);
        } else {
            let mut rng = rand::thread_rng();
            speakers.shuffle(&mut rng);
        }

        // Calculate split sizes based on number of speakers
        let total_speakers = speakers.len();
        let train_speakers = (total_speakers as f32 * config.train_ratio).ceil() as usize;
        let val_speakers = (total_speakers as f32 * config.val_ratio).ceil() as usize;
        let test_speakers = total_speakers - train_speakers - val_speakers;

        // Ensure we don't have negative test speakers
        let _test_speakers = test_speakers.max(0);

        // Split speakers into sets
        let mut speaker_iter = speakers.into_iter();
        let train_speakers: Vec<String> = speaker_iter.by_ref().take(train_speakers).collect();
        let val_speakers: Vec<String> = speaker_iter.by_ref().take(val_speakers).collect();
        let test_speakers: Vec<String> = speaker_iter.collect();

        // Collect sample indices for each split
        let mut train_indices = Vec::new();
        let mut val_indices = Vec::new();
        let mut test_indices = Vec::new();

        for speaker in train_speakers {
            if let Some(indices) = speaker_samples.get(&speaker) {
                train_indices.extend(indices);
            }
        }

        for speaker in val_speakers {
            if let Some(indices) = speaker_samples.get(&speaker) {
                val_indices.extend(indices);
            }
        }

        for speaker in test_speakers {
            if let Some(indices) = speaker_samples.get(&speaker) {
                test_indices.extend(indices);
            }
        }

        // Create DatasetSplit structures
        let train_samples: Vec<DatasetSample> = train_indices
            .iter()
            .filter_map(|&i| self.samples.get(i).cloned())
            .collect();

        let val_samples: Vec<DatasetSample> = val_indices
            .iter()
            .filter_map(|&i| self.samples.get(i).cloned())
            .collect();

        let test_samples: Vec<DatasetSample> = test_indices
            .iter()
            .filter_map(|&i| self.samples.get(i).cloned())
            .collect();

        use crate::splits::DatasetSplit;

        let train_split = DatasetSplit {
            samples: train_samples,
            indices: train_indices,
        };

        let validation_split = DatasetSplit {
            samples: val_samples,
            indices: val_indices,
        };

        let test_split = DatasetSplit {
            samples: test_samples,
            indices: test_indices,
        };

        Ok(DatasetSplits {
            train: train_split,
            validation: validation_split,
            test: test_split,
            config,
        })
    }
}

#[async_trait]
impl Dataset for JvsDataset {
    type Sample = DatasetSample;

    fn len(&self) -> usize {
        self.samples.len()
    }

    async fn get(&self, index: usize) -> Result<Self::Sample> {
        self.samples
            .get(index)
            .cloned()
            .ok_or_else(|| DatasetError::IndexError(index))
    }

    fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    async fn statistics(&self) -> Result<DatasetStatistics> {
        // Calculate text length statistics
        let text_lengths: Vec<usize> = self
            .samples
            .iter()
            .map(|s| s.text.chars().count())
            .collect();

        let text_length_stats = if text_lengths.is_empty() {
            crate::LengthStatistics {
                min: 0,
                max: 0,
                mean: 0.0,
                median: 0,
                std_dev: 0.0,
            }
        } else {
            let mut sorted_lengths = text_lengths.clone();
            sorted_lengths.sort_unstable();

            let min = sorted_lengths[0];
            let max = sorted_lengths[sorted_lengths.len() - 1];
            let sum: usize = text_lengths.iter().sum();
            let mean = sum as f32 / text_lengths.len() as f32;
            let median = sorted_lengths[sorted_lengths.len() / 2];

            let variance: f32 = text_lengths
                .iter()
                .map(|&x| (x as f32 - mean).powi(2))
                .sum::<f32>()
                / text_lengths.len() as f32;
            let std_dev = variance.sqrt();

            crate::LengthStatistics {
                min,
                max,
                mean,
                median,
                std_dev,
            }
        };

        // Calculate duration statistics
        let durations: Vec<f32> = self.samples.iter().map(|s| s.audio.duration()).collect();

        let duration_stats = if durations.is_empty() {
            crate::DurationStatistics {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
            }
        } else {
            let mut sorted_durations = durations.clone();
            sorted_durations.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let min = sorted_durations[0];
            let max = sorted_durations[sorted_durations.len() - 1];
            let sum: f32 = durations.iter().sum();
            let mean = sum / durations.len() as f32;
            let median = sorted_durations[sorted_durations.len() / 2];

            let variance: f32 =
                durations.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / durations.len() as f32;
            let std_dev = variance.sqrt();

            crate::DurationStatistics {
                min,
                max,
                mean,
                median,
                std_dev,
            }
        };

        // Language and speaker distributions
        let mut language_distribution = HashMap::new();
        let mut speaker_distribution = HashMap::new();

        for sample in &self.samples {
            *language_distribution.entry(sample.language).or_insert(0) += 1;
            if let Some(speaker) = &sample.speaker {
                *speaker_distribution.entry(speaker.id.clone()).or_insert(0) += 1;
            }
        }

        Ok(DatasetStatistics {
            total_items: self.samples.len(),
            total_duration: self.samples.iter().map(|s| s.audio.duration()).sum(),
            average_duration: if self.samples.is_empty() {
                0.0
            } else {
                self.samples.iter().map(|s| s.audio.duration()).sum::<f32>()
                    / self.samples.len() as f32
            },
            language_distribution,
            speaker_distribution,
            text_length_stats,
            duration_stats,
        })
    }

    async fn validate(&self) -> Result<ValidationReport> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        for (i, sample) in self.samples.iter().enumerate() {
            // Check for empty text
            if sample.text.trim().is_empty() {
                errors.push(format!("Sample {i}: Empty text"));
            }

            // Check for very short audio
            let duration = sample.audio.duration();
            if duration < 0.3 {
                warnings.push(format!("Sample {i}: Very short audio ({duration:.3}s)"));
            }

            // Check for very long audio
            if duration > 20.0 {
                warnings.push(format!("Sample {i}: Very long audio ({duration:.1}s)"));
            }

            // Check for empty audio
            if sample.audio.is_empty() {
                errors.push(format!("Sample {i}: Empty audio"));
            }

            // Check for missing speaker info
            if sample.speaker.is_none() {
                warnings.push(format!("Sample {i}: Missing speaker information"));
            }

            // Check language
            if sample.language != LanguageCode::Ja {
                warnings.push(format!(
                    "Sample {i}: Unexpected language (expected Japanese)"
                ));
            }
        }

        Ok(ValidationReport {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            items_validated: self.samples.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_type_from_path() {
        assert_eq!(
            JvsSentenceType::from_path("jvs001/parallel100/wav24kHz16bit/BASIC5000_0001.wav"),
            Some(JvsSentenceType::Parallel)
        );
        assert_eq!(
            JvsSentenceType::from_path("jvs002/nonpara30/wav24kHz16bit/BASIC5000_1001.wav"),
            Some(JvsSentenceType::NonParallel)
        );
        assert_eq!(
            JvsSentenceType::from_path("jvs003/whisper10/wav24kHz16bit/BASIC5000_2001.wav"),
            Some(JvsSentenceType::Whisper)
        );
        assert_eq!(
            JvsSentenceType::from_path("jvs004/falsetto10/wav24kHz16bit/BASIC5000_3001.wav"),
            Some(JvsSentenceType::Falsetto)
        );
        assert_eq!(
            JvsSentenceType::from_path("jvs005/reading/wav24kHz16bit/VOICEACTRESS100_001.wav"),
            Some(JvsSentenceType::Reading)
        );
        assert_eq!(JvsSentenceType::from_path("unknown/path/file.wav"), None);
    }

    #[test]
    fn test_sentence_type_as_str() {
        assert_eq!(JvsSentenceType::Parallel.as_str(), "parallel");
        assert_eq!(JvsSentenceType::NonParallel.as_str(), "nonparallel");
        assert_eq!(JvsSentenceType::Whisper.as_str(), "whisper");
        assert_eq!(JvsSentenceType::Falsetto.as_str(), "falsetto");
        assert_eq!(JvsSentenceType::Reading.as_str(), "reading");
    }

    // Helper function to check approximate equality for floats
    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_parse_textgrid_simple() {
        let textgrid_content = r#"
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0 
xmax = 2.5 
tiers? <exists> 
size = 1 
item []: 
    item [1]:
        class = "IntervalTier" 
        name = "phonemes" 
        xmin = 0 
        xmax = 2.5 
        intervals: size = 3 
        intervals [1]:
            xmin = 0 
            xmax = 0.5 
            text = "a" 
        intervals [2]:
            xmin = 0.5 
            xmax = 1.2 
            text = "i" 
        intervals [3]:
            xmin = 1.2 
            xmax = 2.5 
            text = "u" 
"#;

        let phonemes = JvsDataset::parse_textgrid(textgrid_content).unwrap();
        assert_eq!(phonemes.len(), 3);

        assert_eq!(phonemes[0].symbol, "a");
        assert!(approx_eq(phonemes[0].duration.unwrap(), 0.5, 1e-6));

        assert_eq!(phonemes[1].symbol, "i");
        assert!(approx_eq(phonemes[1].duration.unwrap(), 0.7, 1e-6));

        assert_eq!(phonemes[2].symbol, "u");
        assert!(approx_eq(phonemes[2].duration.unwrap(), 1.3, 1e-6));
    }

    #[test]
    fn test_parse_textgrid_with_silence() {
        let textgrid_content = r#"
intervals [1]:
    xmin = 0 
    xmax = 0.2 
    text = "sil" 
intervals [2]:
    xmin = 0.2 
    xmax = 0.8 
    text = "ka" 
intervals [3]:
    xmin = 0.8 
    xmax = 1.0 
    text = "sp" 
intervals [4]:
    xmin = 1.0 
    xmax = 1.5 
    text = "ta" 
"#;

        let phonemes = JvsDataset::parse_textgrid(textgrid_content).unwrap();
        // Should only have "ka" and "ta", silences filtered out
        assert_eq!(phonemes.len(), 2);

        assert_eq!(phonemes[0].symbol, "ka");
        assert_eq!(phonemes[0].duration, Some(0.6));

        assert_eq!(phonemes[1].symbol, "ta");
        assert_eq!(phonemes[1].duration, Some(0.5));
    }

    #[test]
    fn test_parse_simple_phoneme_file_with_timing() {
        let lab_content = "0.0 0.5 a\n0.5 1.2 i\n1.2 2.5 u\n";

        let phonemes = JvsDataset::parse_simple_phoneme_file(lab_content).unwrap();
        assert_eq!(phonemes.len(), 3);

        assert_eq!(phonemes[0].symbol, "a");
        assert!(approx_eq(phonemes[0].duration.unwrap(), 0.5, 1e-6));

        assert_eq!(phonemes[1].symbol, "i");
        assert!(approx_eq(phonemes[1].duration.unwrap(), 0.7, 1e-6));

        assert_eq!(phonemes[2].symbol, "u");
        assert!(approx_eq(phonemes[2].duration.unwrap(), 1.3, 1e-6));
    }

    #[test]
    fn test_parse_simple_phoneme_file_without_timing() {
        let content = "a\ni\nu\ne\no\n";

        let phonemes = JvsDataset::parse_simple_phoneme_file(content).unwrap();
        assert_eq!(phonemes.len(), 5);

        assert_eq!(phonemes[0].symbol, "a");
        assert_eq!(phonemes[0].duration, None);

        assert_eq!(phonemes[1].symbol, "i");
        assert_eq!(phonemes[1].duration, None);

        assert_eq!(phonemes[4].symbol, "o");
        assert_eq!(phonemes[4].duration, None);
    }

    #[test]
    fn test_parse_phoneme_file_with_silences() {
        let content = "0.0 0.1 sil\n0.1 0.6 ka\n0.6 0.7 sp\n0.7 1.2 ta\n1.2 1.3 sil\n";

        let phonemes = JvsDataset::parse_simple_phoneme_file(content).unwrap();
        // Should only have "ka" and "ta", silences filtered out
        assert_eq!(phonemes.len(), 2);

        assert_eq!(phonemes[0].symbol, "ka");
        assert_eq!(phonemes[1].symbol, "ta");
    }

    #[test]
    fn test_parse_empty_phoneme_file() {
        let content = "";
        let result = JvsDataset::parse_simple_phoneme_file(content);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_malformed_phoneme_file() {
        let content = "malformed line\nanother bad line with too many parts a b c d e\n";
        let result = JvsDataset::parse_simple_phoneme_file(content);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_textgrid_fallback_to_simple() {
        // TextGrid content that doesn't match the interval format
        let content = "ka\nta\nna\n";

        let phonemes = JvsDataset::parse_textgrid(content).unwrap();
        assert_eq!(phonemes.len(), 3);
        assert_eq!(phonemes[0].symbol, "ka");
        assert_eq!(phonemes[1].symbol, "ta");
        assert_eq!(phonemes[2].symbol, "na");
    }

    #[test]
    fn test_japanese_phoneme_symbols() {
        let content = "あ\nい\nう\nえ\nお\n";

        let phonemes = JvsDataset::parse_simple_phoneme_file(content).unwrap();
        assert_eq!(phonemes.len(), 5);
        assert_eq!(phonemes[0].symbol, "あ");
        assert_eq!(phonemes[4].symbol, "お");
    }
}
