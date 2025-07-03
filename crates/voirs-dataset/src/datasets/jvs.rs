//! JVS (Japanese Versatile Speech) dataset implementation
//!
//! This module provides loading and processing capabilities for the JVS dataset,
//! a multi-speaker Japanese speech corpus.

use crate::{
    DatasetSample, AudioData, LanguageCode, DatasetError, Result,
    SpeakerInfo, QualityMetrics, DatasetStatistics, ValidationReport,
};
use crate::traits::{Dataset, DatasetMetadata};
use crate::splits::{SplitConfig, SplitStrategy, DatasetSplits, DatasetSplit};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

/// JVS dataset URL
const JVS_URL: &str = "https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus";

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
                "JVS dataset not found at {:?}",
                base_path
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
        let speaker_num: u32 = speaker_id[3..].parse()
            .map_err(|_| DatasetError::LoadError(format!("Invalid speaker ID: {}", speaker_id)))?;
        
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
        for entry in WalkDir::new(&speaker_path)
            .max_depth(3)
            .into_iter()
            .filter_map(|e| e.ok())
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
        let relative_path = audio_path.strip_prefix(base_path)
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
            format!("Sample {}", sample_id)
        });
        
        // Load audio data
        let audio_data = match Self::load_audio_file(audio_path).await {
            Ok(audio) => audio,
            Err(e) => {
                tracing::warn!("Failed to load audio file {:?}: {}", audio_path, e);
                return Ok(None);
            }
        };
        
        // Create speaker info
        let speaker = SpeakerInfo {
            id: speaker_info.speaker_id.clone(),
            name: Some(format!("JVS Speaker {}", speaker_info.speaker_id)),
            gender: Some(speaker_info.gender.clone()),
            age: speaker_info.age,
            accent: speaker_info.region.clone(),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("environment".to_string(), 
                           speaker_info.environment.clone().unwrap_or_default());
                if let Some(sentence_type) = sentence_type {
                    meta.insert("sentence_type".to_string(), sentence_type.as_str().to_string());
                }
                meta
            },
        };
        
        // Create sample
        let sample = DatasetSample {
            id: format!("{}_{}", speaker_info.speaker_id, sample_id),
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
            phonemes: None, // TODO: Add TextGrid parsing
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("audio_path".to_string(), 
                           serde_json::Value::String(audio_path.to_string_lossy().to_string()));
                meta.insert("speaker_id".to_string(), 
                           serde_json::Value::String(speaker_info.speaker_id.clone()));
                if let Some(sentence_type) = sentence_type {
                    meta.insert("sentence_type".to_string(), 
                               serde_json::Value::String(sentence_type.as_str().to_string()));
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
        
        Err(DatasetError::LoadError("No transcript file found".to_string()))
    }
    
    /// Load audio file from path
    async fn load_audio_file(path: &Path) -> Result<AudioData> {
        crate::audio::io::load_wav(path)
    }
    
    /// Get speaker information
    pub fn get_speaker_info(&self, speaker_id: &str) -> Option<&JvsSpeakerInfo> {
        self.speakers.get(speaker_id)
    }
    
    /// Get all speaker IDs
    pub fn speaker_ids(&self) -> Vec<&str> {
        self.speakers.keys().map(|s| s.as_str()).collect()
    }
    
    /// Filter samples by speaker
    pub fn filter_by_speaker(&self, speaker_id: &str) -> Vec<&DatasetSample> {
        self.samples.iter()
            .filter(|sample| {
                sample.speaker.as_ref()
                    .map(|s| s.id == speaker_id)
                    .unwrap_or(false)
            })
            .collect()
    }
    
    /// Filter samples by sentence type
    pub fn filter_by_sentence_type(&self, sentence_type: JvsSentenceType) -> Vec<&DatasetSample> {
        self.samples.iter()
            .filter(|sample| {
                sample.metadata.get("sentence_type")
                    .and_then(|v| v.as_str())
                    .map(|s| s == sentence_type.as_str())
                    .unwrap_or(false)
            })
            .collect()
    }
    
    /// Get parallel sentences (same text across speakers)
    pub fn get_parallel_sentences(&self) -> HashMap<String, Vec<&DatasetSample>> {
        let mut parallel_sentences = HashMap::new();
        
        for sample in self.filter_by_sentence_type(JvsSentenceType::Parallel) {
            // Extract sentence ID from sample ID (assuming format like "jvs001_BASIC5000_0001")
            if let Some(sentence_id) = sample.id.split('_').nth(2) {
                parallel_sentences.entry(sentence_id.to_string())
                    .or_insert_with(Vec::new)
                    .push(sample);
            }
        }
        
        parallel_sentences
    }
    
    /// Create splits with speaker-aware stratification (TODO: Fix type inference issue)
    pub fn create_speaker_aware_splits(&self, _config: SplitConfig) -> Result<DatasetSplits> {
        // TODO: Implement speaker-aware splits - temporarily disabled due to type inference issue
        Err(DatasetError::LoadError("Speaker-aware splits not yet implemented".to_string()))
    }
}

#[async_trait]
impl Dataset for JvsDataset {
    type Sample = DatasetSample;
    
    fn len(&self) -> usize {
        self.samples.len()
    }
    
    async fn get(&self, index: usize) -> Result<Self::Sample> {
        self.samples.get(index)
            .cloned()
            .ok_or_else(|| DatasetError::IndexError(index))
    }
    
    fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }
    
    async fn statistics(&self) -> Result<DatasetStatistics> {
        // Calculate text length statistics
        let text_lengths: Vec<usize> = self.samples.iter()
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
            
            let variance: f32 = text_lengths.iter()
                .map(|&x| (x as f32 - mean).powi(2))
                .sum::<f32>() / text_lengths.len() as f32;
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
        let durations: Vec<f32> = self.samples.iter()
            .map(|s| s.audio.duration())
            .collect();
        
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
            
            let variance: f32 = durations.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / durations.len() as f32;
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
            average_duration: if self.samples.is_empty() { 0.0 } else { 
                self.samples.iter().map(|s| s.audio.duration()).sum::<f32>() / self.samples.len() as f32
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
                errors.push(format!("Sample {}: Empty text", i));
            }
            
            // Check for very short audio
            let duration = sample.audio.duration();
            if duration < 0.3 {
                warnings.push(format!("Sample {}: Very short audio ({:.3}s)", i, duration));
            }
            
            // Check for very long audio
            if duration > 20.0 {
                warnings.push(format!("Sample {}: Very long audio ({:.1}s)", i, duration));
            }
            
            // Check for empty audio
            if sample.audio.is_empty() {
                errors.push(format!("Sample {}: Empty audio", i));
            }
            
            // Check for missing speaker info
            if sample.speaker.is_none() {
                warnings.push(format!("Sample {}: Missing speaker information", i));
            }
            
            // Check language
            if sample.language != LanguageCode::Ja {
                warnings.push(format!("Sample {}: Unexpected language (expected Japanese)", i));
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
