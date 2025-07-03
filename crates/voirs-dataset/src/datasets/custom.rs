//! Custom dataset implementation
//!
//! A flexible dataset loader that can handle various directory structures and file formats.
//! This loader is designed to work with custom datasets that don't follow standard
//! dataset structures like LJSpeech or VCTK.
//!
//! Features:
//! - Flexible directory structure discovery
//! - Multiple transcript format support (CSV, JSON, TXT, Manifest)
//! - Audio file discovery and validation
//! - Metadata validation and cleaning
//! - Speaker information extraction
//! - Configurable filtering and validation

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

/// Transcript file format types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TranscriptFormat {
    /// CSV format with configurable columns
    Csv,
    /// JSON format with structured data
    Json,
    /// JSON Lines format (one JSON object per line)
    JsonLines,
    /// Plain text files (one per audio file)
    Txt,
    /// Manifest format (audio_path|text|speaker_id format)
    Manifest,
    /// Automatic detection based on file extension
    Auto,
}

impl Default for TranscriptFormat {
    fn default() -> Self {
        Self::Auto
    }
}

/// Directory structure patterns
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DirectoryStructure {
    /// Flat structure: all files in one directory
    Flat,
    /// Speaker-based: speaker_id/audio_files
    BySpeaker,
    /// Category-based: category/audio_files
    ByCategory,
    /// Nested: category/speaker/audio_files
    Nested,
    /// Custom pattern with placeholders
    Custom(String),
    /// Auto-detect structure
    Auto,
}

impl Default for DirectoryStructure {
    fn default() -> Self {
        Self::Auto
    }
}

/// CSV column mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsvConfig {
    /// Column index or name for audio file path
    pub audio_column: String,
    /// Column index or name for text transcript
    pub text_column: String,
    /// Column index or name for speaker ID (optional)
    pub speaker_column: Option<String>,
    /// Column index or name for language (optional)
    pub language_column: Option<String>,
    /// Whether the CSV has headers
    pub has_headers: bool,
    /// CSV delimiter character
    pub delimiter: char,
}

impl Default for CsvConfig {
    fn default() -> Self {
        Self {
            audio_column: "audio_path".to_string(),
            text_column: "text".to_string(),
            speaker_column: Some("speaker_id".to_string()),
            language_column: Some("language".to_string()),
            has_headers: true,
            delimiter: ',',
        }
    }
}

/// Custom dataset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomConfig {
    /// Root directory containing the dataset
    pub root_dir: PathBuf,
    /// Directory structure pattern
    pub structure: DirectoryStructure,
    /// Transcript file format
    pub transcript_format: TranscriptFormat,
    /// Path to transcript file(s) relative to root_dir
    pub transcript_path: Option<PathBuf>,
    /// CSV configuration (used when transcript_format is Csv)
    pub csv_config: CsvConfig,
    /// Audio file extensions to include
    pub audio_extensions: Vec<String>,
    /// Default language code
    pub default_language: LanguageCode,
    /// Whether to recursively search for audio files
    pub recursive_search: bool,
    /// Maximum directory depth for recursive search
    pub max_depth: Option<usize>,
    /// Minimum audio duration (seconds)
    pub min_duration: Option<f32>,
    /// Maximum audio duration (seconds)
    pub max_duration: Option<f32>,
    /// Filter by specific speakers
    pub speaker_filter: Option<Vec<String>>,
    /// Whether to validate audio files during loading
    pub validate_audio: bool,
    /// Whether to clean and normalize text
    pub clean_text: bool,
}

impl Default for CustomConfig {
    fn default() -> Self {
        Self {
            root_dir: PathBuf::from("."),
            structure: DirectoryStructure::Auto,
            transcript_format: TranscriptFormat::Auto,
            transcript_path: None,
            csv_config: CsvConfig::default(),
            audio_extensions: vec!["wav".to_string(), "flac".to_string(), "mp3".to_string(), "ogg".to_string()],
            default_language: LanguageCode::EnUs,
            recursive_search: true,
            max_depth: Some(5),
            min_duration: Some(0.1),
            max_duration: Some(30.0),
            speaker_filter: None,
            validate_audio: true,
            clean_text: true,
        }
    }
}

/// Custom dataset sample
#[derive(Debug, Clone)]
pub struct CustomSample {
    /// Sample identifier
    pub id: String,
    /// Text content
    pub text: String,
    /// Path to audio file
    pub audio_path: PathBuf,
    /// Speaker information (if available)
    pub speaker: Option<SpeakerInfo>,
    /// Language
    pub language: LanguageCode,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Cached audio data
    cached_audio: Option<AudioData>,
}

impl CustomSample {
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
        
        Ok(Sample {
            id: self.id,
            text: self.text,
            audio,
            speaker: self.speaker,
            language: self.language,
            quality,
            phonemes: None,
            metadata: self.metadata,
        })
    }
}

/// Custom dataset implementation
pub struct CustomDataset {
    /// Dataset configuration
    config: CustomConfig,
    /// List of samples
    samples: Vec<CustomSample>,
    /// Dataset metadata
    metadata: DatasetMetadata,
}

impl CustomDataset {
    /// Create new custom dataset from directory
    pub async fn new<P: AsRef<Path>>(root_dir: P) -> Result<Self> {
        let config = CustomConfig {
            root_dir: root_dir.as_ref().to_path_buf(),
            ..Default::default()
        };
        Self::from_config(config).await
    }
    
    /// Create custom dataset with configuration
    pub async fn from_config(config: CustomConfig) -> Result<Self> {
        let mut dataset = CustomDataset {
            config: config.clone(),
            samples: Vec::new(),
            metadata: DatasetMetadata {
                name: "Custom".to_string(),
                version: "1.0".to_string(),
                description: Some("Custom dataset with flexible structure".to_string()),
                total_samples: 0,
                total_duration: 0.0,
                languages: vec![config.default_language.as_str().to_string()],
                speakers: Vec::new(),
                license: None,
                metadata: HashMap::new(),
            },
        };
        
        // Load samples based on configuration
        dataset.load_samples().await?;
        
        // Update metadata
        dataset.update_metadata().await?;
        
        Ok(dataset)
    }
    
    /// Load samples from the dataset
    async fn load_samples(&mut self) -> Result<()> {
        match self.config.transcript_format {
            TranscriptFormat::Auto => self.auto_detect_and_load().await?,
            TranscriptFormat::Csv => self.load_from_csv().await?,
            TranscriptFormat::Json => self.load_from_json().await?,
            TranscriptFormat::JsonLines => self.load_from_jsonlines().await?,
            TranscriptFormat::Txt => self.load_from_txt_files().await?,
            TranscriptFormat::Manifest => self.load_from_manifest().await?,
        }
        
        // Apply filters
        self.apply_filters().await?;
        
        Ok(())
    }
    
    /// Auto-detect transcript format and load
    async fn auto_detect_and_load(&mut self) -> Result<()> {
        // Look for common transcript files
        let root = &self.config.root_dir;
        
        // Check for CSV files
        let mut read_dir = fs::read_dir(root).await.map_err(|e| DatasetError::LoadError(format!("Failed to read directory: {}", e)))?;
        while let Some(entry) = read_dir.next_entry().await.map_err(|e| DatasetError::LoadError(format!("Directory entry error: {}", e)))? {
            let path = entry.path();
            
            if let Some(ext) = path.extension() {
                match ext.to_str() {
                    Some("csv") => {
                        self.config.transcript_path = Some(path.strip_prefix(root).unwrap().to_path_buf());
                        return self.load_from_csv().await;
                    }
                    Some("json") => {
                        self.config.transcript_path = Some(path.strip_prefix(root).unwrap().to_path_buf());
                        return self.load_from_json().await;
                    }
                    Some("jsonl") | Some("jsonlines") => {
                        self.config.transcript_path = Some(path.strip_prefix(root).unwrap().to_path_buf());
                        return self.load_from_jsonlines().await;
                    }
                    Some("txt") if path.file_name().unwrap().to_str().unwrap().contains("manifest") => {
                        self.config.transcript_path = Some(path.strip_prefix(root).unwrap().to_path_buf());
                        return self.load_from_manifest().await;
                    }
                    _ => {}
                }
            }
        }
        
        // Fall back to TXT files (one per audio file)
        self.load_from_txt_files().await
    }
    
    /// Load from CSV file
    async fn load_from_csv(&mut self) -> Result<()> {
        let csv_path = if let Some(ref path) = self.config.transcript_path {
            self.config.root_dir.join(path)
        } else {
            return Err(DatasetError::LoadError("CSV path not specified".to_string()));
        };
        
        let content = fs::read_to_string(&csv_path).await
            .map_err(|e| DatasetError::LoadError(format!("Failed to read CSV file: {}", e)))?;
        
        let mut csv_reader = csv::ReaderBuilder::new()
            .delimiter(self.config.csv_config.delimiter as u8)
            .has_headers(self.config.csv_config.has_headers)
            .from_reader(content.as_bytes());
        
        let headers = if self.config.csv_config.has_headers {
            csv_reader.headers()?.clone()
        } else {
            csv::StringRecord::new()
        };
        
        for (row_idx, result) in csv_reader.records().enumerate() {
            let record = result.map_err(|e| DatasetError::CsvError(e))?;
            
            // Extract fields based on configuration
            let audio_path = self.get_csv_field(&record, &headers, &self.config.csv_config.audio_column)?;
            let text = self.get_csv_field(&record, &headers, &self.config.csv_config.text_column)?;
            
            let speaker_id = if let Some(ref col) = self.config.csv_config.speaker_column {
                self.get_csv_field(&record, &headers, col).ok()
            } else {
                None
            };
            
            let language = if let Some(ref col) = self.config.csv_config.language_column {
                self.get_csv_field(&record, &headers, col)
                    .ok()
                    .and_then(|lang| self.parse_language(&lang))
                    .unwrap_or(self.config.default_language)
            } else {
                self.config.default_language
            };
            
            // Resolve audio path
            let full_audio_path = if Path::new(&audio_path).is_absolute() {
                PathBuf::from(audio_path)
            } else {
                self.config.root_dir.join(&audio_path)
            };
            
            if !full_audio_path.exists() {
                continue; // Skip missing audio files
            }
            
            // Create speaker info if available
            let speaker = speaker_id.map(|id| SpeakerInfo {
                id: id.clone(),
                name: None,
                gender: None,
                age: None,
                accent: None,
                metadata: HashMap::new(),
            });
            
            let sample = CustomSample {
                id: format!("sample_{:06}", row_idx),
                text: if self.config.clean_text { self.clean_text(&text) } else { text },
                audio_path: full_audio_path,
                speaker,
                language,
                metadata: HashMap::new(),
                cached_audio: None,
            };
            
            self.samples.push(sample);
        }
        
        Ok(())
    }
    
    /// Load from JSON file
    async fn load_from_json(&mut self) -> Result<()> {
        let json_path = if let Some(ref path) = self.config.transcript_path {
            self.config.root_dir.join(path)
        } else {
            return Err(DatasetError::LoadError("JSON path not specified".to_string()));
        };
        
        let content = fs::read_to_string(&json_path).await
            .map_err(|e| DatasetError::LoadError(format!("Failed to read JSON file: {}", e)))?;
        
        let data: serde_json::Value = serde_json::from_str(&content)?;
        
        // Handle different JSON structures
        let items = match &data {
            serde_json::Value::Array(arr) => arr,
            serde_json::Value::Object(obj) => {
                // Look for common array keys
                if let Some(serde_json::Value::Array(arr)) = obj.get("data").or_else(|| obj.get("samples")).or_else(|| obj.get("items")) {
                    arr
                } else {
                    return Err(DatasetError::FormatError("JSON object must contain 'data', 'samples', or 'items' array".to_string()));
                }
            }
            _ => return Err(DatasetError::FormatError("JSON must be array or object with array field".to_string())),
        };
        
        for (idx, item) in items.iter().enumerate() {
            let obj = item.as_object()
                .ok_or_else(|| DatasetError::FormatError("JSON items must be objects".to_string()))?;
            
            let audio_path = obj.get("audio_path").or_else(|| obj.get("path")).or_else(|| obj.get("file"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| DatasetError::FormatError("JSON item missing audio_path field".to_string()))?;
            
            let text = obj.get("text").or_else(|| obj.get("transcript")).or_else(|| obj.get("transcription"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| DatasetError::FormatError("JSON item missing text field".to_string()))?;
            
            let speaker_id = obj.get("speaker_id").or_else(|| obj.get("speaker"))
                .and_then(|v| v.as_str());
            
            let language = obj.get("language").or_else(|| obj.get("lang"))
                .and_then(|v| v.as_str())
                .and_then(|lang| self.parse_language(lang))
                .unwrap_or(self.config.default_language);
            
            // Resolve audio path
            let full_audio_path = if Path::new(audio_path).is_absolute() {
                PathBuf::from(audio_path)
            } else {
                self.config.root_dir.join(audio_path)
            };
            
            if !full_audio_path.exists() {
                continue; // Skip missing audio files
            }
            
            // Create speaker info if available
            let speaker = speaker_id.map(|id| SpeakerInfo {
                id: id.to_string(),
                name: None,
                gender: None,
                age: None,
                accent: None,
                metadata: HashMap::new(),
            });
            
            // Extract additional metadata
            let mut metadata = HashMap::new();
            for (key, value) in obj {
                if !["audio_path", "path", "file", "text", "transcript", "transcription", "speaker_id", "speaker", "language", "lang"].contains(&key.as_str()) {
                    metadata.insert(key.clone(), value.clone());
                }
            }
            
            let sample = CustomSample {
                id: obj.get("id").and_then(|v| v.as_str()).unwrap_or(&format!("sample_{:06}", idx)).to_string(),
                text: if self.config.clean_text { self.clean_text(text) } else { text.to_string() },
                audio_path: full_audio_path,
                speaker,
                language,
                metadata,
                cached_audio: None,
            };
            
            self.samples.push(sample);
        }
        
        Ok(())
    }
    
    /// Load from JSON Lines file
    async fn load_from_jsonlines(&mut self) -> Result<()> {
        let jsonl_path = if let Some(ref path) = self.config.transcript_path {
            self.config.root_dir.join(path)
        } else {
            return Err(DatasetError::LoadError("JSON Lines path not specified".to_string()));
        };
        
        let content = fs::read_to_string(&jsonl_path).await
            .map_err(|e| DatasetError::LoadError(format!("Failed to read JSON Lines file: {}", e)))?;
        
        for (idx, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            
            let obj: serde_json::Map<String, serde_json::Value> = serde_json::from_str(line)?;
            
            let audio_path = obj.get("audio_path").or_else(|| obj.get("path")).or_else(|| obj.get("file"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| DatasetError::FormatError("JSON Lines item missing audio_path field".to_string()))?;
            
            let text = obj.get("text").or_else(|| obj.get("transcript")).or_else(|| obj.get("transcription"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| DatasetError::FormatError("JSON Lines item missing text field".to_string()))?;
            
            let speaker_id = obj.get("speaker_id").or_else(|| obj.get("speaker"))
                .and_then(|v| v.as_str());
            
            let language = obj.get("language").or_else(|| obj.get("lang"))
                .and_then(|v| v.as_str())
                .and_then(|lang| self.parse_language(lang))
                .unwrap_or(self.config.default_language);
            
            // Resolve audio path
            let full_audio_path = if Path::new(audio_path).is_absolute() {
                PathBuf::from(audio_path)
            } else {
                self.config.root_dir.join(audio_path)
            };
            
            if !full_audio_path.exists() {
                continue; // Skip missing audio files
            }
            
            // Create speaker info if available
            let speaker = speaker_id.map(|id| SpeakerInfo {
                id: id.to_string(),
                name: None,
                gender: None,
                age: None,
                accent: None,
                metadata: HashMap::new(),
            });
            
            // Extract additional metadata
            let mut metadata = HashMap::new();
            for (key, value) in &obj {
                if !["audio_path", "path", "file", "text", "transcript", "transcription", "speaker_id", "speaker", "language", "lang"].contains(&key.as_str()) {
                    metadata.insert(key.clone(), value.clone());
                }
            }
            
            let sample = CustomSample {
                id: obj.get("id").and_then(|v| v.as_str()).unwrap_or(&format!("sample_{:06}", idx)).to_string(),
                text: if self.config.clean_text { self.clean_text(text) } else { text.to_string() },
                audio_path: full_audio_path,
                speaker,
                language,
                metadata,
                cached_audio: None,
            };
            
            self.samples.push(sample);
        }
        
        Ok(())
    }
    
    /// Load from individual text files
    async fn load_from_txt_files(&mut self) -> Result<()> {
        let audio_files = self.discover_audio_files().await?;
        
        for audio_path in audio_files {
            // Look for corresponding text file
            let txt_path = audio_path.with_extension("txt");
            
            if !txt_path.exists() {
                continue; // Skip if no corresponding text file
            }
            
            let text = fs::read_to_string(&txt_path).await
                .map_err(|e| DatasetError::LoadError(format!("Failed to read text file {}: {}", txt_path.display(), e)))?;
            
            let text = text.trim();
            if text.is_empty() {
                continue;
            }
            
            // Extract sample ID from filename
            let id = audio_path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            
            // Try to extract speaker ID from path or filename
            let speaker_id = self.extract_speaker_from_path(&audio_path);
            let speaker = speaker_id.map(|id| SpeakerInfo {
                id,
                name: None,
                gender: None,
                age: None,
                accent: None,
                metadata: HashMap::new(),
            });
            
            let sample = CustomSample {
                id,
                text: if self.config.clean_text { self.clean_text(text) } else { text.to_string() },
                audio_path,
                speaker,
                language: self.config.default_language,
                metadata: HashMap::new(),
                cached_audio: None,
            };
            
            self.samples.push(sample);
        }
        
        Ok(())
    }
    
    /// Load from manifest file (audio_path|text|speaker_id format)
    async fn load_from_manifest(&mut self) -> Result<()> {
        let manifest_path = if let Some(ref path) = self.config.transcript_path {
            self.config.root_dir.join(path)
        } else {
            return Err(DatasetError::LoadError("Manifest path not specified".to_string()));
        };
        
        let content = fs::read_to_string(&manifest_path).await
            .map_err(|e| DatasetError::LoadError(format!("Failed to read manifest file: {}", e)))?;
        
        for (idx, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            
            let parts: Vec<&str> = line.split('|').collect();
            if parts.len() < 2 {
                continue; // Skip malformed lines
            }
            
            let audio_path = parts[0].trim();
            let text = parts[1].trim();
            let speaker_id = if parts.len() > 2 && !parts[2].trim().is_empty() {
                Some(parts[2].trim().to_string())
            } else {
                None
            };
            
            // Resolve audio path
            let full_audio_path = if Path::new(audio_path).is_absolute() {
                PathBuf::from(audio_path)
            } else {
                self.config.root_dir.join(audio_path)
            };
            
            if !full_audio_path.exists() {
                continue; // Skip missing audio files
            }
            
            // Create speaker info if available
            let speaker = speaker_id.map(|id| SpeakerInfo {
                id,
                name: None,
                gender: None,
                age: None,
                accent: None,
                metadata: HashMap::new(),
            });
            
            let sample = CustomSample {
                id: format!("sample_{:06}", idx),
                text: if self.config.clean_text { self.clean_text(text) } else { text.to_string() },
                audio_path: full_audio_path,
                speaker,
                language: self.config.default_language,
                metadata: HashMap::new(),
                cached_audio: None,
            };
            
            self.samples.push(sample);
        }
        
        Ok(())
    }
    
    /// Discover audio files in the dataset
    async fn discover_audio_files(&self) -> Result<Vec<PathBuf>> {
        let mut audio_files = Vec::new();
        let max_depth = self.config.max_depth.unwrap_or(usize::MAX);
        
        let walker = if self.config.recursive_search {
            WalkDir::new(&self.config.root_dir).max_depth(max_depth)
        } else {
            WalkDir::new(&self.config.root_dir).max_depth(1)
        };
        
        for entry in walker {
            let entry = entry.map_err(|e| DatasetError::LoadError(format!("Directory walk error: {}", e)))?;
            
            if entry.file_type().is_file() {
                if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
                    if self.config.audio_extensions.contains(&ext.to_lowercase()) {
                        audio_files.push(entry.path().to_path_buf());
                    }
                }
            }
        }
        
        Ok(audio_files)
    }
    
    /// Apply filters to loaded samples
    async fn apply_filters(&mut self) -> Result<()> {
        let mut filtered_samples = Vec::new();
        
        for mut sample in std::mem::take(&mut self.samples) {
            // Apply speaker filter
            if let Some(ref filter) = self.config.speaker_filter {
                if let Some(ref speaker) = sample.speaker {
                    if !filter.contains(&speaker.id) {
                        continue;
                    }
                }
            }
            
            // Apply duration filter if validation is enabled
            if self.config.validate_audio {
                if let Ok(audio) = load_audio(&sample.audio_path) {
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
            
            filtered_samples.push(sample);
        }
        
        self.samples = filtered_samples;
        Ok(())
    }
    
    /// Get CSV field by column name or index
    fn get_csv_field(&self, record: &csv::StringRecord, headers: &csv::StringRecord, column: &str) -> Result<String> {
        // Try as column name first
        if self.config.csv_config.has_headers {
            for (i, header) in headers.iter().enumerate() {
                if header == column {
                    return record.get(i)
                        .ok_or_else(|| DatasetError::FormatError(format!("Column {} not found in record", column)))
                        .map(|s| s.to_string());
                }
            }
        }
        
        // Try as column index
        if let Ok(index) = column.parse::<usize>() {
            return record.get(index)
                .ok_or_else(|| DatasetError::FormatError(format!("Column index {} out of bounds", index)))
                .map(|s| s.to_string());
        }
        
        Err(DatasetError::FormatError(format!("Column {} not found", column)))
    }
    
    /// Parse language string to LanguageCode
    fn parse_language(&self, lang_str: &str) -> Option<LanguageCode> {
        match lang_str.to_lowercase().as_str() {
            "en" | "en-us" | "english" => Some(LanguageCode::EnUs),
            "en-gb" | "en-uk" | "british" => Some(LanguageCode::EnGb),
            "ja" | "japanese" => Some(LanguageCode::Ja),
            "zh" | "zh-cn" | "chinese" | "mandarin" => Some(LanguageCode::ZhCn),
            "ko" | "korean" => Some(LanguageCode::Ko),
            "de" | "german" => Some(LanguageCode::De),
            "fr" | "french" => Some(LanguageCode::Fr),
            "es" | "spanish" => Some(LanguageCode::Es),
            _ => None,
        }
    }
    
    /// Extract speaker ID from file path
    fn extract_speaker_from_path(&self, path: &Path) -> Option<String> {
        // Try different patterns to extract speaker ID
        
        // Pattern: /speaker_id/filename
        if let Some(parent) = path.parent() {
            if let Some(speaker_dir) = parent.file_name().and_then(|s| s.to_str()) {
                // Check if this looks like a speaker directory (reasonable length and not generic names)
                if speaker_dir.len() >= 2 && speaker_dir.len() <= 20 
                    && !speaker_dir.eq_ignore_ascii_case("dataset") 
                    && !speaker_dir.eq_ignore_ascii_case("data") 
                    && !speaker_dir.eq_ignore_ascii_case("audio") {
                    return Some(speaker_dir.to_string());
                }
            }
        }
        
        // Pattern: speaker_id_filename
        if let Some(filename) = path.file_stem().and_then(|s| s.to_str()) {
            let parts: Vec<&str> = filename.split('_').collect();
            if parts.len() >= 2 {
                let speaker_candidate = parts[0];
                // Check if this looks like a speaker ID (reasonable length and not generic names)
                if speaker_candidate.len() >= 2 && speaker_candidate.len() <= 20
                    && !speaker_candidate.eq_ignore_ascii_case("dataset")
                    && !speaker_candidate.eq_ignore_ascii_case("data")
                    && !speaker_candidate.eq_ignore_ascii_case("audio") {
                    return Some(speaker_candidate.to_string());
                }
            }
        }
        
        None
    }
    
    /// Clean and normalize text
    fn clean_text(&self, text: &str) -> String {
        text.trim()
            .chars()
            .filter(|c| !c.is_control() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
    
    /// Update dataset metadata
    async fn update_metadata(&mut self) -> Result<()> {
        self.metadata.total_samples = self.samples.len();
        
        // Collect unique speakers
        let mut speakers = std::collections::HashSet::new();
        for sample in &self.samples {
            if let Some(ref speaker) = sample.speaker {
                speakers.insert(speaker.id.clone());
            }
        }
        self.metadata.speakers = speakers.into_iter().collect();
        
        // Collect unique languages
        let mut languages = std::collections::HashSet::new();
        for sample in &self.samples {
            languages.insert(sample.language.as_str().to_string());
        }
        self.metadata.languages = languages.into_iter().collect();
        
        // Estimate total duration (sample a few files)
        let mut total_duration = 0.0;
        let sample_count = 10.min(self.samples.len());
        
        for sample in self.samples.iter().take(sample_count) {
            if let Ok(audio) = load_audio(&sample.audio_path) {
                total_duration += audio.duration();
            }
        }
        
        if sample_count > 0 {
            let avg_duration = total_duration / sample_count as f32;
            self.metadata.total_duration = avg_duration * self.samples.len() as f32;
        }
        
        Ok(())
    }
    
    /// Get configuration
    pub fn config(&self) -> &CustomConfig {
        &self.config
    }
    
    /// Get samples by speaker
    pub fn get_speaker_samples(&self, speaker_id: &str) -> Vec<&CustomSample> {
        self.samples
            .iter()
            .filter(|sample| {
                sample.speaker.as_ref().map_or(false, |s| s.id == speaker_id)
            })
            .collect()
    }
}

#[async_trait]
impl Dataset for CustomDataset {
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
            
            *language_distribution.entry(sample.language).or_insert(0) += 1;
            
            if let Some(ref speaker) = sample.speaker {
                *speaker_distribution.entry(speaker.id.clone()).or_insert(0) += 1;
            }
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
    async fn test_custom_config_default() {
        let config = CustomConfig::default();
        assert_eq!(config.root_dir, PathBuf::from("."));
        assert_eq!(config.structure, DirectoryStructure::Auto);
        assert_eq!(config.transcript_format, TranscriptFormat::Auto);
        assert!(config.recursive_search);
        assert_eq!(config.max_depth, Some(5));
        assert_eq!(config.default_language, LanguageCode::EnUs);
        assert!(config.validate_audio);
        assert!(config.clean_text);
    }
    
    #[tokio::test]
    async fn test_csv_config_default() {
        let config = CsvConfig::default();
        assert_eq!(config.audio_column, "audio_path");
        assert_eq!(config.text_column, "text");
        assert_eq!(config.speaker_column, Some("speaker_id".to_string()));
        assert_eq!(config.language_column, Some("language".to_string()));
        assert!(config.has_headers);
        assert_eq!(config.delimiter, ',');
    }
    
    #[tokio::test]
    async fn test_language_parsing() {
        let config = CustomConfig::default();
        let dataset = CustomDataset {
            config,
            samples: Vec::new(),
            metadata: DatasetMetadata {
                name: "test".to_string(),
                version: "1.0".to_string(),
                description: None,
                total_samples: 0,
                total_duration: 0.0,
                languages: Vec::new(),
                speakers: Vec::new(),
                license: None,
                metadata: HashMap::new(),
            },
        };
        
        assert_eq!(dataset.parse_language("en"), Some(LanguageCode::EnUs));
        assert_eq!(dataset.parse_language("en-GB"), Some(LanguageCode::EnGb));
        assert_eq!(dataset.parse_language("ja"), Some(LanguageCode::Ja));
        assert_eq!(dataset.parse_language("unknown"), None);
    }
    
    #[tokio::test]
    async fn test_text_cleaning() {
        let config = CustomConfig::default();
        let dataset = CustomDataset {
            config,
            samples: Vec::new(),
            metadata: DatasetMetadata {
                name: "test".to_string(),
                version: "1.0".to_string(),
                description: None,
                total_samples: 0,
                total_duration: 0.0,
                languages: Vec::new(),
                speakers: Vec::new(),
                license: None,
                metadata: HashMap::new(),
            },
        };
        
        let cleaned = dataset.clean_text("  Hello,   world!  \n\t  ");
        assert_eq!(cleaned, "Hello, world!");
        
        let cleaned = dataset.clean_text("Text\x00with\x01control\x02chars");
        assert_eq!(cleaned, "Textwithcontrolchars");
    }
    
    #[tokio::test]
    async fn test_speaker_extraction() {
        let config = CustomConfig::default();
        let dataset = CustomDataset {
            config,
            samples: Vec::new(),
            metadata: DatasetMetadata {
                name: "test".to_string(),
                version: "1.0".to_string(),
                description: None,
                total_samples: 0,
                total_duration: 0.0,
                languages: Vec::new(),
                speakers: Vec::new(),
                license: None,
                metadata: HashMap::new(),
            },
        };
        
        // Test speaker extraction from path
        let path = PathBuf::from("/dataset/speaker1/audio.wav");
        assert_eq!(dataset.extract_speaker_from_path(&path), Some("speaker1".to_string()));
        
        // Test speaker extraction from filename
        let path = PathBuf::from("/dataset/speaker1_001.wav");
        assert_eq!(dataset.extract_speaker_from_path(&path), Some("speaker1".to_string()));
    }
}