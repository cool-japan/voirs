//! VCTK (Voice Cloning Toolkit) dataset implementation
//!
//! The VCTK corpus contains speech data uttered by 110 English speakers with various accents.
//! Each speaker reads out approximately 400 sentences, which were selected from a newspaper,
//! the rainbow passage and an elicitation paragraph used for the speech accent archive.
//!
//! Features:
//! - Multi-speaker English corpus (110 speakers)
//! - Various English accents and dialects
//! - Parallel and non-parallel text sets
//! - Speaker demographic information
//! - High-quality audio recordings

use crate::audio::io::load_audio;
use crate::traits::{Dataset, DatasetMetadata, DatasetSample};
use crate::{
    AudioData, DatasetError, DatasetSample as Sample, DatasetStatistics, DurationStatistics,
    LanguageCode, LengthStatistics, QualityMetrics, Result, SpeakerInfo, ValidationReport,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use walkdir::WalkDir;

/// VCTK dataset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VctkConfig {
    /// Root directory containing VCTK corpus
    pub root_dir: PathBuf,
    /// Whether to include only parallel sentences
    pub parallel_only: bool,
    /// Filter by specific speakers (None = all speakers)
    pub speaker_filter: Option<Vec<String>>,
    /// Filter by accent (None = all accents)
    pub accent_filter: Option<Vec<String>>,
    /// Minimum audio duration (seconds)
    pub min_duration: Option<f32>,
    /// Maximum audio duration (seconds)
    pub max_duration: Option<f32>,
    /// Whether to validate audio files on load
    pub validate_audio: bool,
}

impl Default for VctkConfig {
    fn default() -> Self {
        Self {
            root_dir: PathBuf::from("VCTK-Corpus"),
            parallel_only: false,
            speaker_filter: None,
            accent_filter: None,
            min_duration: Some(0.5),
            max_duration: Some(10.0),
            validate_audio: true,
        }
    }
}

/// VCTK speaker demographics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VctkSpeakerInfo {
    /// Speaker ID (e.g., "p225")
    pub id: String,
    /// Speaker age
    pub age: Option<u32>,
    /// Speaker gender
    pub gender: Option<String>,
    /// Accent/region (e.g., "English", "Scottish", etc.)
    pub accent: Option<String>,
    /// Region/country
    pub region: Option<String>,
    /// Comments about speaker
    pub comments: Option<String>,
}

impl From<VctkSpeakerInfo> for SpeakerInfo {
    fn from(vctk_speaker: VctkSpeakerInfo) -> Self {
        let mut metadata = HashMap::new();
        
        if let Some(ref accent) = vctk_speaker.accent {
            metadata.insert("accent".to_string(), accent.clone());
        }
        if let Some(ref region) = vctk_speaker.region {
            metadata.insert("region".to_string(), region.clone());
        }
        if let Some(ref comments) = vctk_speaker.comments {
            metadata.insert("comments".to_string(), comments.clone());
        }
        
        SpeakerInfo {
            id: vctk_speaker.id,
            name: None,
            gender: vctk_speaker.gender,
            age: vctk_speaker.age,
            accent: vctk_speaker.accent,
            metadata,
        }
    }
}

/// VCTK dataset item
#[derive(Debug, Clone)]
pub struct VctkSample {
    /// Sample identifier (e.g., "p225_001")
    pub id: String,
    /// Text content
    pub text: String,
    /// Path to audio file
    pub audio_path: PathBuf,
    /// Speaker information
    pub speaker: VctkSpeakerInfo,
    /// Whether this is a parallel sentence
    pub is_parallel: bool,
    /// Sentence number for parallel sentences
    pub sentence_number: Option<u32>,
    /// Cached audio data
    cached_audio: Option<AudioData>,
}

impl VctkSample {
    /// Load audio data for this sample
    pub async fn load_audio(&mut self) -> Result<&AudioData> {
        if self.cached_audio.is_none() {
            let audio = load_audio(&self.audio_path)?;
            self.cached_audio = Some(audio);
        }
        Ok(self.cached_audio.as_ref().unwrap())
    }
    
    /// Get audio data (load if not cached)
    pub async fn audio(&mut self) -> Result<AudioData> {
        self.load_audio().await.map(|a| a.clone())
    }
    
    /// Convert to standard DatasetSample
    pub async fn to_dataset_sample(mut self) -> Result<Sample> {
        let audio = self.audio().await?;
        
        let quality = QualityMetrics {
            snr: None,
            clipping: None,
            dynamic_range: None,
            spectral_quality: None,
            overall_quality: None,
        };
        
        let mut metadata = HashMap::new();
        metadata.insert("is_parallel".to_string(), serde_json::Value::Bool(self.is_parallel));
        if let Some(sentence_number) = self.sentence_number {
            metadata.insert("sentence_number".to_string(), serde_json::Value::Number(sentence_number.into()));
        }
        metadata.insert("audio_path".to_string(), serde_json::Value::String(self.audio_path.to_string_lossy().to_string()));
        
        Ok(Sample {
            id: self.id,
            text: self.text,
            audio,
            speaker: Some(self.speaker.into()),
            language: LanguageCode::EnGb, // VCTK is primarily British English
            quality,
            phonemes: None,
            metadata,
        })
    }
}

/// VCTK dataset implementation
pub struct VctkDataset {
    /// Dataset configuration
    config: VctkConfig,
    /// List of samples
    samples: Vec<VctkSample>,
    /// Speaker information mapping
    speakers: HashMap<String, VctkSpeakerInfo>,
    /// Dataset metadata
    metadata: DatasetMetadata,
}

impl VctkDataset {
    /// Create new VCTK dataset from directory
    pub async fn new<P: AsRef<Path>>(root_dir: P) -> Result<Self> {
        let config = VctkConfig {
            root_dir: root_dir.as_ref().to_path_buf(),
            ..Default::default()
        };
        Self::from_config(config).await
    }
    
    /// Create VCTK dataset with custom configuration
    pub async fn from_config(config: VctkConfig) -> Result<Self> {
        let mut dataset = VctkDataset {
            config: config.clone(),
            samples: Vec::new(),
            speakers: HashMap::new(),
            metadata: DatasetMetadata {
                name: "VCTK".to_string(),
                version: "0.92".to_string(),
                description: Some("VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit".to_string()),
                total_samples: 0,
                total_duration: 0.0,
                languages: vec!["en-GB".to_string()],
                speakers: Vec::new(),
                license: Some("Open Data Commons Attribution License (ODC-By) v1.0".to_string()),
                metadata: HashMap::new(),
            },
        };
        
        // Load speaker information
        dataset.load_speaker_info().await?;
        
        // Load samples
        dataset.load_samples().await?;
        
        // Update metadata
        dataset.update_metadata().await?;
        
        Ok(dataset)
    }
    
    /// Load speaker information from SPEAKERS.txt
    async fn load_speaker_info(&mut self) -> Result<()> {
        let speakers_file = self.config.root_dir.join("speaker-info.txt");
        
        if !speakers_file.exists() {
            // If speaker info file doesn't exist, create default speaker info
            // from discovered audio directories
            return self.discover_speakers().await;
        }
        
        let content = fs::read_to_string(&speakers_file).await
            .map_err(|e| DatasetError::LoadError(format!("Failed to read speaker info: {}", e)))?;
        
        // Parse speaker information file
        // Format: ID AGE GENDER ACCENTS REGION COMMENTS
        for (_line_num, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }
            
            let id = parts[0].to_string();
            let age = parts.get(1).and_then(|s| s.parse().ok());
            let gender = parts.get(2).map(|s| s.to_string());
            let accent = parts.get(3).map(|s| s.to_string());
            let region = parts.get(4).map(|s| s.to_string());
            let comments = if parts.len() > 5 {
                Some(parts[5..].join(" "))
            } else {
                None
            };
            
            let speaker_info = VctkSpeakerInfo {
                id: id.clone(),
                age,
                gender,
                accent,
                region,
                comments,
            };
            
            self.speakers.insert(id, speaker_info);
        }
        
        Ok(())
    }
    
    /// Discover speakers from audio directory structure
    async fn discover_speakers(&mut self) -> Result<()> {
        let wav48_dir = self.config.root_dir.join("wav48_silence_trimmed");
        if !wav48_dir.exists() {
            return Err(DatasetError::LoadError("VCTK wav48_silence_trimmed directory not found".to_string()));
        }
        
        // Walk through speaker directories
        for entry in WalkDir::new(&wav48_dir).min_depth(1).max_depth(1) {
            let entry = entry.map_err(|e| DatasetError::LoadError(format!("Directory walk error: {}", e)))?;
            
            if entry.file_type().is_dir() {
                if let Some(speaker_id) = entry.file_name().to_str() {
                    if speaker_id.starts_with('p') && speaker_id.len() >= 4 {
                        let speaker_info = VctkSpeakerInfo {
                            id: speaker_id.to_string(),
                            age: None,
                            gender: None,
                            accent: Some("English".to_string()),
                            region: Some("UK".to_string()),
                            comments: None,
                        };
                        
                        self.speakers.insert(speaker_id.to_string(), speaker_info);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Load text transcriptions
    async fn load_transcriptions(&self) -> Result<HashMap<String, String>> {
        let txt_dir = self.config.root_dir.join("txt");
        let mut transcriptions = HashMap::new();
        
        if !txt_dir.exists() {
            return Err(DatasetError::LoadError("VCTK txt directory not found".to_string()));
        }
        
        // Walk through speaker text directories
        for entry in WalkDir::new(&txt_dir).min_depth(2).max_depth(2) {
            let entry = entry.map_err(|e| DatasetError::LoadError(format!("Directory walk error: {}", e)))?;
            
            if entry.file_type().is_file() && 
               entry.path().extension().map_or(false, |ext| ext == "txt") {
                
                let file_path = entry.path();
                let file_stem = file_path.file_stem()
                    .and_then(|s| s.to_str())
                    .ok_or_else(|| DatasetError::FormatError("Invalid text filename".to_string()))?;
                
                let content = fs::read_to_string(file_path).await
                    .map_err(|e| DatasetError::LoadError(format!("Failed to read text file {}: {}", file_path.display(), e)))?;
                
                transcriptions.insert(file_stem.to_string(), content.trim().to_string());
            }
        }
        
        Ok(transcriptions)
    }
    
    /// Load all samples from the dataset
    async fn load_samples(&mut self) -> Result<()> {
        let transcriptions = self.load_transcriptions().await?;
        let wav48_dir = self.config.root_dir.join("wav48_silence_trimmed");
        
        if !wav48_dir.exists() {
            return Err(DatasetError::LoadError("VCTK wav48_silence_trimmed directory not found".to_string()));
        }
        
        // Walk through all audio files
        for entry in WalkDir::new(&wav48_dir).min_depth(2).max_depth(2) {
            let entry = entry.map_err(|e| DatasetError::LoadError(format!("Directory walk error: {}", e)))?;
            
            if entry.file_type().is_file() && 
               entry.path().extension().map_or(false, |ext| ext == "wav") {
                
                let audio_path = entry.path().to_path_buf();
                let file_stem = audio_path.file_stem()
                    .and_then(|s| s.to_str())
                    .ok_or_else(|| DatasetError::FormatError("Invalid audio filename".to_string()))?;
                
                // Extract speaker ID from filename (e.g., "p225_001" -> "p225")
                let parts: Vec<&str> = file_stem.split('_').collect();
                if parts.len() != 2 {
                    continue;
                }
                
                let speaker_id = parts[0];
                let sentence_id = parts[1];
                
                // Check speaker filter
                if let Some(ref filter) = self.config.speaker_filter {
                    if !filter.contains(&speaker_id.to_string()) {
                        continue;
                    }
                }
                
                // Get speaker info
                let speaker_info = self.speakers.get(speaker_id)
                    .ok_or_else(|| DatasetError::LoadError(format!("Unknown speaker: {}", speaker_id)))?
                    .clone();
                
                // Check accent filter
                if let Some(ref filter) = self.config.accent_filter {
                    if let Some(ref accent) = speaker_info.accent {
                        if !filter.contains(accent) {
                            continue;
                        }
                    }
                }
                
                // Get transcription
                let text = transcriptions.get(file_stem)
                    .ok_or_else(|| DatasetError::LoadError(format!("No transcription found for {}", file_stem)))?
                    .clone();
                
                // Determine if this is a parallel sentence
                let sentence_number: Option<u32> = sentence_id.parse().ok();
                let is_parallel = sentence_number.map_or(false, |n| n <= 400); // VCTK has ~400 parallel sentences
                
                // Check parallel filter
                if self.config.parallel_only && !is_parallel {
                    continue;
                }
                
                // Validate audio duration if required
                if self.config.validate_audio {
                    if let Ok(audio) = load_audio(&audio_path) {
                        let duration = audio.duration();
                        
                        if let Some(min_dur) = self.config.min_duration {
                            if duration < min_dur {
                                continue;
                            }
                        }
                        
                        if let Some(max_dur) = self.config.max_duration {
                            if duration > max_dur {
                                continue;
                            }
                        }
                    }
                }
                
                let sample = VctkSample {
                    id: file_stem.to_string(),
                    text,
                    audio_path,
                    speaker: speaker_info,
                    is_parallel,
                    sentence_number,
                    cached_audio: None,
                };
                
                self.samples.push(sample);
            }
        }
        
        Ok(())
    }
    
    /// Update dataset metadata
    async fn update_metadata(&mut self) -> Result<()> {
        self.metadata.total_samples = self.samples.len();
        self.metadata.speakers = self.speakers.keys().cloned().collect();
        
        // Calculate total duration (sample a few files for estimation)
        let mut total_duration = 0.0;
        let sample_count = 10.min(self.samples.len());
        
        for sample in self.samples.iter().take(sample_count) {
            if let Ok(audio) = load_audio(&sample.audio_path) {
                total_duration += audio.duration();
            }
        }
        
        // Estimate total duration
        if sample_count > 0 {
            let avg_duration = total_duration / sample_count as f32;
            self.metadata.total_duration = avg_duration * self.samples.len() as f32;
        }
        
        // Add VCTK-specific metadata
        self.metadata.metadata.insert(
            "parallel_sentences".to_string(),
            serde_json::Value::Number(self.samples.iter().filter(|s| s.is_parallel).count().into()),
        );
        self.metadata.metadata.insert(
            "non_parallel_sentences".to_string(),
            serde_json::Value::Number(self.samples.iter().filter(|s| !s.is_parallel).count().into()),
        );
        
        Ok(())
    }
    
    /// Get samples for a specific speaker
    pub fn get_speaker_samples(&self, speaker_id: &str) -> Vec<&VctkSample> {
        self.samples
            .iter()
            .filter(|sample| sample.speaker.id == speaker_id)
            .collect()
    }
    
    /// Get all parallel sentences across speakers
    pub fn get_parallel_sentences(&self) -> HashMap<u32, Vec<&VctkSample>> {
        let mut parallel_map = HashMap::new();
        
        for sample in &self.samples {
            if let Some(sentence_number) = sample.sentence_number {
                if sample.is_parallel {
                    parallel_map.entry(sentence_number).or_insert_with(Vec::new).push(sample);
                }
            }
        }
        
        parallel_map
    }
    
    /// Get speaker demographics summary
    pub fn speaker_demographics(&self) -> HashMap<String, usize> {
        let mut demographics = HashMap::new();
        
        for speaker in self.speakers.values() {
            if let Some(ref gender) = speaker.gender {
                *demographics.entry(format!("gender_{}", gender)).or_insert(0) += 1;
            }
            if let Some(ref accent) = speaker.accent {
                *demographics.entry(format!("accent_{}", accent)).or_insert(0) += 1;
            }
            if let Some(ref region) = speaker.region {
                *demographics.entry(format!("region_{}", region)).or_insert(0) += 1;
            }
        }
        
        demographics
    }
}

#[async_trait]
impl Dataset for VctkDataset {
    type Sample = Sample;
    
    fn len(&self) -> usize {
        self.samples.len()
    }
    
    async fn get(&self, index: usize) -> Result<Self::Sample> {
        let sample = self.samples
            .get(index)
            .ok_or_else(|| DatasetError::IndexError(index))?
            .clone();
        
        sample.to_dataset_sample().await
    }
    
    fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }
    
    async fn statistics(&self) -> Result<DatasetStatistics> {
        let mut total_duration = 0.0;
        let mut text_lengths = Vec::new();
        let mut durations = Vec::new();
        let mut language_distribution = HashMap::new();
        let mut speaker_distribution = HashMap::new();
        
        // Sample for statistics (to avoid loading all audio files)
        let sample_size = 100.min(self.samples.len());
        
        for sample in self.samples.iter().take(sample_size) {
            text_lengths.push(sample.text.len());
            
            if let Ok(audio) = load_audio(&sample.audio_path) {
                let duration = audio.duration();
                total_duration += duration;
                durations.push(duration);
            }
            
            *language_distribution.entry(LanguageCode::EnGb).or_insert(0) += 1;
            *speaker_distribution.entry(sample.speaker.id.clone()).or_insert(0) += 1;
        }
        
        // Estimate total duration
        if !durations.is_empty() {
            let avg_duration = total_duration / durations.len() as f32;
            total_duration = avg_duration * self.samples.len() as f32;
        }
        
        let text_length_stats = if text_lengths.is_empty() {
            LengthStatistics {
                min: 0,
                max: 0,
                mean: 0.0,
                median: 0,
                std_dev: 0.0,
            }
        } else {
            crate::calculate_length_stats(&text_lengths)
        };
        
        let duration_stats = if durations.is_empty() {
            DurationStatistics {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
            }
        } else {
            crate::calculate_duration_stats(&durations)
        };
        
        Ok(DatasetStatistics {
            total_items: self.samples.len(),
            total_duration,
            average_duration: if self.samples.is_empty() { 0.0 } else { total_duration / self.samples.len() as f32 },
            language_distribution,
            speaker_distribution,
            text_length_stats,
            duration_stats,
        })
    }
    
    async fn validate(&self) -> Result<ValidationReport> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut items_validated = 0;
        
        for (i, sample) in self.samples.iter().enumerate() {
            items_validated += 1;
            
            // Check if text is empty
            if sample.text.trim().is_empty() {
                errors.push(format!("Sample {}: Empty text", i));
            }
            
            // Check if audio file exists
            if !sample.audio_path.exists() {
                errors.push(format!("Sample {}: Audio file not found: {}", i, sample.audio_path.display()));
                continue;
            }
            
            // Validate audio file (sample every 10th file to avoid loading everything)
            if i % 10 == 0 {
                match load_audio(&sample.audio_path) {
                    Ok(audio) => {
                        let duration = audio.duration();
                        if duration < 0.1 {
                            warnings.push(format!("Sample {}: Very short audio ({:.3}s)", i, duration));
                        } else if duration > 30.0 {
                            warnings.push(format!("Sample {}: Very long audio ({:.1}s)", i, duration));
                        }
                        
                        if audio.is_empty() {
                            errors.push(format!("Sample {}: Empty audio", i));
                        }
                    }
                    Err(e) => {
                        errors.push(format!("Sample {}: Audio loading error: {}", i, e));
                    }
                }
            }
        }
        
        Ok(ValidationReport {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            items_validated,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_vctk_config_default() {
        let config = VctkConfig::default();
        assert_eq!(config.root_dir, PathBuf::from("VCTK-Corpus"));
        assert!(!config.parallel_only);
        assert!(config.speaker_filter.is_none());
        assert_eq!(config.min_duration, Some(0.5));
        assert_eq!(config.max_duration, Some(10.0));
        assert!(config.validate_audio);
    }
    
    #[tokio::test]
    async fn test_vctk_speaker_info_conversion() {
        let vctk_speaker = VctkSpeakerInfo {
            id: "p225".to_string(),
            age: Some(23),
            gender: Some("Female".to_string()),
            accent: Some("English".to_string()),
            region: Some("Southern England".to_string()),
            comments: Some("Clear speech".to_string()),
        };
        
        let speaker_info: SpeakerInfo = vctk_speaker.into();
        assert_eq!(speaker_info.id, "p225");
        assert_eq!(speaker_info.age, Some(23));
        assert_eq!(speaker_info.gender, Some("Female".to_string()));
        assert_eq!(speaker_info.accent, Some("English".to_string()));
        assert_eq!(speaker_info.metadata.get("region"), Some(&"Southern England".to_string()));
        assert_eq!(speaker_info.metadata.get("comments"), Some(&"Clear speech".to_string()));
    }
    
    #[tokio::test]
    async fn test_vctk_sample_creation() {
        let speaker = VctkSpeakerInfo {
            id: "p225".to_string(),
            age: Some(23),
            gender: Some("Female".to_string()),
            accent: Some("English".to_string()),
            region: Some("Southern England".to_string()),
            comments: None,
        };
        
        let sample = VctkSample {
            id: "p225_001".to_string(),
            text: "Please call Stella.".to_string(),
            audio_path: PathBuf::from("/path/to/p225_001.wav"),
            speaker,
            is_parallel: true,
            sentence_number: Some(1),
            cached_audio: None,
        };
        
        assert_eq!(sample.id, "p225_001");
        assert_eq!(sample.text, "Please call Stella.");
        assert!(sample.is_parallel);
        assert_eq!(sample.sentence_number, Some(1));
        assert_eq!(sample.speaker.id, "p225");
    }
    
    #[tokio::test]
    async fn test_vctk_speaker_discovery() {
        // This test would require setting up a mock VCTK directory structure
        // For now, we'll just test the config creation
        let config = VctkConfig {
            root_dir: PathBuf::from("/tmp/test-vctk"),
            parallel_only: true,
            speaker_filter: Some(vec!["p225".to_string(), "p226".to_string()]),
            accent_filter: Some(vec!["English".to_string()]),
            min_duration: Some(1.0),
            max_duration: Some(5.0),
            validate_audio: false,
        };
        
        assert_eq!(config.root_dir, PathBuf::from("/tmp/test-vctk"));
        assert!(config.parallel_only);
        assert_eq!(config.speaker_filter, Some(vec!["p225".to_string(), "p226".to_string()]));
        assert_eq!(config.accent_filter, Some(vec!["English".to_string()]));
        assert_eq!(config.min_duration, Some(1.0));
        assert_eq!(config.max_duration, Some(5.0));
        assert!(!config.validate_audio);
    }
}