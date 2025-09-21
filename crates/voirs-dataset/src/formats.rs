//! Dataset format definitions and utilities.
//!
//! This module provides comprehensive support for various dataset formats including
//! metadata formats, audio file validation, phoneme representations, and alignment formats.

use crate::{AudioFormat, DatasetError, LanguageCode, Phoneme, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Dataset metadata format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub name: String,
    pub version: String,
    pub language: LanguageCode,
    pub description: Option<String>,
    pub speaker_count: Option<usize>,
    pub total_duration: Option<f32>,
    pub license: Option<String>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    pub tags: Vec<String>,
    pub statistics: Option<HashMap<String, serde_json::Value>>,
}

impl DatasetMetadata {
    /// Create new dataset metadata
    pub fn new(name: String, version: String, language: LanguageCode) -> Self {
        Self {
            name,
            version,
            language,
            description: None,
            speaker_count: None,
            total_duration: None,
            license: None,
            created_at: None,
            updated_at: None,
            tags: Vec::new(),
            statistics: None,
        }
    }

    /// Add tag to metadata
    pub fn add_tag<S: Into<String>>(mut self, tag: S) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set description
    pub fn with_description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set license
    pub fn with_license<S: Into<String>>(mut self, license: S) -> Self {
        self.license = Some(license.into());
        self
    }
}

/// Dataset manifest entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    pub id: String,
    pub text: String,
    pub audio_path: String,
    pub phonemes_path: Option<String>,
    pub alignment_path: Option<String>,
    pub speaker_id: Option<String>,
    pub duration: Option<f32>,
    pub quality_score: Option<f32>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ManifestEntry {
    /// Create new manifest entry
    pub fn new<S: Into<String>>(id: S, text: S, audio_path: S) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            audio_path: audio_path.into(),
            phonemes_path: None,
            alignment_path: None,
            speaker_id: None,
            duration: None,
            quality_score: None,
            metadata: HashMap::new(),
        }
    }

    /// Set phonemes path
    pub fn with_phonemes_path<S: Into<String>>(mut self, path: S) -> Self {
        self.phonemes_path = Some(path.into());
        self
    }

    /// Set alignment path
    pub fn with_alignment_path<S: Into<String>>(mut self, path: S) -> Self {
        self.alignment_path = Some(path.into());
        self
    }

    /// Set speaker ID
    pub fn with_speaker_id<S: Into<String>>(mut self, speaker_id: S) -> Self {
        self.speaker_id = Some(speaker_id.into());
        self
    }
}

/// Supported metadata file formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataFormat {
    Json,
    Yaml,
    Csv,
    Toml,
}

impl MetadataFormat {
    /// Detect format from file extension
    pub fn from_extension<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("json") => Ok(MetadataFormat::Json),
            Some("yaml") | Some("yml") => Ok(MetadataFormat::Yaml),
            Some("csv") => Ok(MetadataFormat::Csv),
            Some("toml") => Ok(MetadataFormat::Toml),
            _ => Err(DatasetError::FormatError(format!(
                "Unsupported metadata format: {ext:?}",
                ext = path.extension()
            ))),
        }
    }

    /// Get file extension for format
    pub fn extension(&self) -> &'static str {
        match self {
            MetadataFormat::Json => "json",
            MetadataFormat::Yaml => "yaml",
            MetadataFormat::Csv => "csv",
            MetadataFormat::Toml => "toml",
        }
    }
}

/// Audio file format validator
pub struct AudioFormatValidator;

impl AudioFormatValidator {
    /// Validate audio file format
    pub fn validate<P: AsRef<Path>>(path: P) -> Result<AudioFormat> {
        let path = path.as_ref();

        // Check if file exists
        if !path.exists() {
            return Err(DatasetError::AudioError(format!(
                "Audio file does not exist: {path:?}"
            )));
        }

        // Detect format from extension
        let format = match path.extension().and_then(|ext| ext.to_str()) {
            Some("wav") => AudioFormat::Wav,
            Some("flac") => AudioFormat::Flac,
            Some("mp3") => AudioFormat::Mp3,
            Some("ogg") => AudioFormat::Ogg,
            _ => {
                return Err(DatasetError::FormatError(format!(
                    "Unsupported audio format: {ext:?}",
                    ext = path.extension()
                )))
            }
        };

        // Basic file header validation
        match format {
            AudioFormat::Wav => Self::validate_wav_header(path)?,
            AudioFormat::Flac => Self::validate_flac_header(path)?,
            AudioFormat::Mp3 => Self::validate_mp3_header(path)?,
            AudioFormat::Ogg => Self::validate_ogg_header(path)?,
            AudioFormat::Opus => Self::validate_opus_header(path)?,
        }

        Ok(format)
    }

    /// Validate WAV file header
    fn validate_wav_header<P: AsRef<Path>>(path: P) -> Result<()> {
        let data = std::fs::read(path)
            .map_err(|e| DatasetError::AudioError(format!("Failed to read WAV file: {e}")))?;

        if data.len() < 12 {
            return Err(DatasetError::FormatError("WAV file too short".to_string()));
        }

        // Check RIFF header
        if &data[0..4] != b"RIFF" {
            return Err(DatasetError::FormatError(
                "Invalid WAV RIFF header".to_string(),
            ));
        }

        // Check WAVE format
        if &data[8..12] != b"WAVE" {
            return Err(DatasetError::FormatError(
                "Invalid WAV format header".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate FLAC file header
    fn validate_flac_header<P: AsRef<Path>>(path: P) -> Result<()> {
        let data = std::fs::read(path)
            .map_err(|e| DatasetError::AudioError(format!("Failed to read FLAC file: {e}")))?;

        if data.len() < 4 {
            return Err(DatasetError::FormatError("FLAC file too short".to_string()));
        }

        // Check FLAC header
        if &data[0..4] != b"fLaC" {
            return Err(DatasetError::FormatError("Invalid FLAC header".to_string()));
        }

        Ok(())
    }

    /// Validate MP3 file header
    fn validate_mp3_header<P: AsRef<Path>>(path: P) -> Result<()> {
        let data = std::fs::read(path)
            .map_err(|e| DatasetError::AudioError(format!("Failed to read MP3 file: {e}")))?;

        if data.len() < 3 {
            return Err(DatasetError::FormatError("MP3 file too short".to_string()));
        }

        // Check for MP3 frame sync or ID3 tag
        if (data[0] == 0xFF && (data[1] & 0xE0) == 0xE0) || // MP3 frame sync
           &data[0..3] == b"ID3"
        {
            // ID3 tag
            Ok(())
        } else {
            Err(DatasetError::FormatError("Invalid MP3 header".to_string()))
        }
    }

    /// Validate OGG file header
    fn validate_ogg_header<P: AsRef<Path>>(path: P) -> Result<()> {
        let data = std::fs::read(path)
            .map_err(|e| DatasetError::AudioError(format!("Failed to read OGG file: {e}")))?;

        if data.len() < 4 {
            return Err(DatasetError::FormatError("OGG file too short".to_string()));
        }

        // Check OGG header
        if &data[0..4] != b"OggS" {
            return Err(DatasetError::FormatError("Invalid OGG header".to_string()));
        }

        Ok(())
    }

    fn validate_opus_header<P: AsRef<Path>>(path: P) -> Result<()> {
        let data = std::fs::read(path)
            .map_err(|e| DatasetError::AudioError(format!("Failed to read OPUS file: {e}")))?;

        if data.len() < 16 {
            return Err(DatasetError::FormatError("OPUS file too short".to_string()));
        }

        // Check OGG header first (OPUS is in OGG container)
        if &data[0..4] != b"OggS" {
            return Err(DatasetError::FormatError(
                "Invalid OPUS OGG header".to_string(),
            ));
        }

        // Check for OPUS identification header
        if data.len() >= 16 && &data[8..16] == b"OpusHead" {
            Ok(())
        } else {
            Err(DatasetError::FormatError(
                "OPUS identification header not found".to_string(),
            ))
        }
    }
}

/// Phoneme file format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhonemeFormat {
    /// International Phonetic Alphabet (IPA)
    Ipa,
    /// ARPABET phoneme set
    Arpabet,
    /// Custom phoneme set
    Custom,
    /// Praat TextGrid format
    TextGrid,
}

/// Phoneme sequence representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeSequence {
    pub phonemes: Vec<Phoneme>,
    pub format: String,
    pub language: LanguageCode,
    pub total_duration: Option<f32>,
}

impl PhonemeSequence {
    /// Create new phoneme sequence
    pub fn new(format: String, language: LanguageCode) -> Self {
        Self {
            phonemes: Vec::new(),
            format,
            language,
            total_duration: None,
        }
    }

    /// Add phoneme to sequence
    pub fn add_phoneme(&mut self, phoneme: Phoneme) {
        self.phonemes.push(phoneme);
    }

    /// Calculate total duration
    pub fn calculate_duration(&mut self) {
        self.total_duration = Some(self.phonemes.iter().filter_map(|p| p.duration).sum());
    }

    /// Load from file
    pub fn load_from_file<P: AsRef<Path>>(path: P, format: PhonemeFormat) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| DatasetError::AudioError(format!("Failed to read phoneme file: {e}")))?;

        match format {
            PhonemeFormat::Ipa => Self::parse_ipa(&content),
            PhonemeFormat::Arpabet => Self::parse_arpabet(&content),
            PhonemeFormat::TextGrid => Self::parse_textgrid(&content),
            PhonemeFormat::Custom => Self::parse_custom(&content),
        }
    }

    /// Parse IPA format
    fn parse_ipa(content: &str) -> Result<Self> {
        let mut sequence = Self::new("IPA".to_string(), LanguageCode::EnUs);

        for line in content.lines() {
            let line = line.trim();
            if !line.is_empty() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if !parts.is_empty() {
                    let symbol = parts[0].to_string();
                    let duration = if parts.len() > 1 {
                        parts[1].parse::<f32>().ok()
                    } else {
                        None
                    };

                    let mut phoneme = Phoneme::new(symbol);
                    phoneme.duration = duration;
                    sequence.add_phoneme(phoneme);
                }
            }
        }

        sequence.calculate_duration();
        Ok(sequence)
    }

    /// Parse ARPABET format
    fn parse_arpabet(content: &str) -> Result<Self> {
        let mut sequence = Self::new("ARPABET".to_string(), LanguageCode::EnUs);

        for line in content.lines() {
            let line = line.trim();
            if !line.is_empty() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if !parts.is_empty() {
                    let symbol = parts[0].to_string();
                    let duration = if parts.len() > 1 {
                        parts[1].parse::<f32>().ok()
                    } else {
                        None
                    };

                    let mut phoneme = Phoneme::new(symbol);
                    phoneme.duration = duration;
                    sequence.add_phoneme(phoneme);
                }
            }
        }

        sequence.calculate_duration();
        Ok(sequence)
    }

    /// Parse TextGrid format (simplified)
    fn parse_textgrid(content: &str) -> Result<Self> {
        let mut sequence = Self::new("TextGrid".to_string(), LanguageCode::EnUs);

        let lines: Vec<&str> = content.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i].trim();
            if line.contains("text =") {
                // Extract phoneme symbol
                let symbol = line.split('"').nth(1).unwrap_or("").trim().to_string();

                if !symbol.is_empty() && symbol != "sil" {
                    let phoneme = Phoneme::new(symbol);
                    sequence.add_phoneme(phoneme);
                }
            }
            i += 1;
        }

        Ok(sequence)
    }

    /// Parse custom format (JSON)
    fn parse_custom(content: &str) -> Result<Self> {
        serde_json::from_str(content).map_err(|e| {
            DatasetError::FormatError(format!("Failed to parse custom phoneme format: {e}"))
        })
    }
}

/// Alignment file format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentFormat {
    /// Montreal Forced Alignment (MFA) format
    Mfa,
    /// Praat TextGrid format
    TextGrid,
    /// Lab file format
    Lab,
    /// Custom JSON format
    Json,
}

/// Time alignment entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentEntry {
    pub start_time: f32,
    pub end_time: f32,
    pub text: String,
    pub phoneme: Option<String>,
    pub confidence: Option<f32>,
}

impl AlignmentEntry {
    /// Create new alignment entry
    pub fn new(start_time: f32, end_time: f32, text: String) -> Self {
        Self {
            start_time,
            end_time,
            text,
            phoneme: None,
            confidence: None,
        }
    }

    /// Get duration
    pub fn duration(&self) -> f32 {
        self.end_time - self.start_time
    }
}

/// Alignment sequence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentSequence {
    pub entries: Vec<AlignmentEntry>,
    pub format: String,
    pub total_duration: f32,
}

impl AlignmentSequence {
    /// Create new alignment sequence
    pub fn new(format: String) -> Self {
        Self {
            entries: Vec::new(),
            format,
            total_duration: 0.0,
        }
    }

    /// Add alignment entry
    pub fn add_entry(&mut self, entry: AlignmentEntry) {
        self.total_duration = self.total_duration.max(entry.end_time);
        self.entries.push(entry);
    }

    /// Load from file
    pub fn load_from_file<P: AsRef<Path>>(path: P, format: AlignmentFormat) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| DatasetError::AudioError(format!("Failed to read alignment file: {e}")))?;

        match format {
            AlignmentFormat::Mfa => Self::parse_mfa(&content),
            AlignmentFormat::TextGrid => Self::parse_textgrid_alignment(&content),
            AlignmentFormat::Lab => Self::parse_lab(&content),
            AlignmentFormat::Json => Self::parse_json(&content),
        }
    }

    /// Parse MFA format
    fn parse_mfa(content: &str) -> Result<Self> {
        let mut sequence = Self::new("MFA".to_string());

        for line in content.lines() {
            let line = line.trim();
            if !line.is_empty() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    let start_time = parts[0].parse::<f32>().map_err(|_| {
                        DatasetError::FormatError("Invalid start time in MFA format".to_string())
                    })?;
                    let end_time = parts[1].parse::<f32>().map_err(|_| {
                        DatasetError::FormatError("Invalid end time in MFA format".to_string())
                    })?;
                    let text = parts[2..].join(" ");

                    let entry = AlignmentEntry::new(start_time, end_time, text);
                    sequence.add_entry(entry);
                }
            }
        }

        Ok(sequence)
    }

    /// Parse TextGrid alignment format
    fn parse_textgrid_alignment(content: &str) -> Result<Self> {
        let mut sequence = Self::new("TextGrid".to_string());

        let lines: Vec<&str> = content.lines().collect();
        let mut i = 0;
        let mut current_start = 0.0;
        let mut current_end = 0.0;
        #[allow(unused_assignments)]
        let mut current_text = String::new();

        while i < lines.len() {
            let line = lines[i].trim();

            if line.contains("xmin =") {
                if let Some(time_str) = line.split('=').nth(1) {
                    current_start = time_str.trim().parse::<f32>().map_err(|_| {
                        DatasetError::FormatError("Invalid xmin in TextGrid".to_string())
                    })?;
                }
            } else if line.contains("xmax =") {
                if let Some(time_str) = line.split('=').nth(1) {
                    current_end = time_str.trim().parse::<f32>().map_err(|_| {
                        DatasetError::FormatError("Invalid xmax in TextGrid".to_string())
                    })?;
                }
            } else if line.contains("text =") {
                current_text = line.split('"').nth(1).unwrap_or("").trim().to_string();

                if !current_text.is_empty() {
                    let entry =
                        AlignmentEntry::new(current_start, current_end, current_text.clone());
                    sequence.add_entry(entry);
                }
            }

            i += 1;
        }

        Ok(sequence)
    }

    /// Parse Lab format
    fn parse_lab(content: &str) -> Result<Self> {
        let mut sequence = Self::new("Lab".to_string());

        for line in content.lines() {
            let line = line.trim();
            if !line.is_empty() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    let start_time = parts[0].parse::<f32>().map_err(|_| {
                        DatasetError::FormatError("Invalid start time in Lab format".to_string())
                    })?;
                    let end_time = parts[1].parse::<f32>().map_err(|_| {
                        DatasetError::FormatError("Invalid end time in Lab format".to_string())
                    })?;
                    let text = parts[2..].join(" ");

                    let entry = AlignmentEntry::new(start_time, end_time, text);
                    sequence.add_entry(entry);
                }
            }
        }

        Ok(sequence)
    }

    /// Parse JSON format
    fn parse_json(content: &str) -> Result<Self> {
        serde_json::from_str(content).map_err(|e| {
            DatasetError::FormatError(format!("Failed to parse JSON alignment format: {e}"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_dataset_metadata_creation() {
        let metadata = DatasetMetadata::new(
            "test-dataset".to_string(),
            "1.0.0".to_string(),
            LanguageCode::EnUs,
        );

        assert_eq!(metadata.name, "test-dataset");
        assert_eq!(metadata.version, "1.0.0");
        assert_eq!(metadata.language, LanguageCode::EnUs);
        assert!(metadata.description.is_none());
        assert!(metadata.tags.is_empty());
    }

    #[test]
    fn test_dataset_metadata_builder_pattern() {
        let metadata = DatasetMetadata::new(
            "test-dataset".to_string(),
            "1.0.0".to_string(),
            LanguageCode::EnUs,
        )
        .with_description("A test dataset")
        .with_license("MIT")
        .add_tag("speech")
        .add_tag("synthesis");

        assert_eq!(metadata.description.as_deref(), Some("A test dataset"));
        assert_eq!(metadata.license.as_deref(), Some("MIT"));
        assert_eq!(metadata.tags, vec!["speech", "synthesis"]);
    }

    #[test]
    fn test_manifest_entry_creation() {
        let entry = ManifestEntry::new("sample-001", "Hello world", "audio/sample-001.wav");

        assert_eq!(entry.id, "sample-001");
        assert_eq!(entry.text, "Hello world");
        assert_eq!(entry.audio_path, "audio/sample-001.wav");
        assert!(entry.phonemes_path.is_none());
        assert!(entry.speaker_id.is_none());
    }

    #[test]
    fn test_manifest_entry_builder_pattern() {
        let entry = ManifestEntry::new("sample-001", "Hello world", "audio/sample-001.wav")
            .with_phonemes_path("phonemes/sample-001.txt")
            .with_alignment_path("alignments/sample-001.json")
            .with_speaker_id("speaker-01");

        assert_eq!(
            entry.phonemes_path.as_deref(),
            Some("phonemes/sample-001.txt")
        );
        assert_eq!(
            entry.alignment_path.as_deref(),
            Some("alignments/sample-001.json")
        );
        assert_eq!(entry.speaker_id.as_deref(), Some("speaker-01"));
    }

    #[test]
    fn test_metadata_format_detection() {
        assert_eq!(
            MetadataFormat::from_extension("test.json").unwrap(),
            MetadataFormat::Json
        );
        assert_eq!(
            MetadataFormat::from_extension("test.yaml").unwrap(),
            MetadataFormat::Yaml
        );
        assert_eq!(
            MetadataFormat::from_extension("test.yml").unwrap(),
            MetadataFormat::Yaml
        );
        assert_eq!(
            MetadataFormat::from_extension("test.csv").unwrap(),
            MetadataFormat::Csv
        );
        assert_eq!(
            MetadataFormat::from_extension("test.toml").unwrap(),
            MetadataFormat::Toml
        );

        assert!(MetadataFormat::from_extension("test.txt").is_err());
    }

    #[test]
    fn test_metadata_format_extension() {
        assert_eq!(MetadataFormat::Json.extension(), "json");
        assert_eq!(MetadataFormat::Yaml.extension(), "yaml");
        assert_eq!(MetadataFormat::Csv.extension(), "csv");
        assert_eq!(MetadataFormat::Toml.extension(), "toml");
    }

    #[test]
    fn test_wav_header_validation() -> Result<()> {
        let mut file = NamedTempFile::new().unwrap();

        // Write valid WAV header
        file.write_all(b"RIFF").unwrap();
        file.write_all(&[0; 4]).unwrap(); // File size (placeholder)
        file.write_all(b"WAVE").unwrap();
        file.flush().unwrap();

        assert!(AudioFormatValidator::validate_wav_header(file.path()).is_ok());

        // Test invalid header
        let mut invalid_file = NamedTempFile::new().unwrap();
        invalid_file.write_all(b"INVALID_HEADER").unwrap();
        invalid_file.flush().unwrap();

        assert!(AudioFormatValidator::validate_wav_header(invalid_file.path()).is_err());

        Ok(())
    }

    #[test]
    fn test_flac_header_validation() -> Result<()> {
        let mut file = NamedTempFile::new().unwrap();

        // Write valid FLAC header
        file.write_all(b"fLaC").unwrap();
        file.flush().unwrap();

        assert!(AudioFormatValidator::validate_flac_header(file.path()).is_ok());

        // Test invalid header
        let mut invalid_file = NamedTempFile::new().unwrap();
        invalid_file.write_all(b"INVALID").unwrap();
        invalid_file.flush().unwrap();

        assert!(AudioFormatValidator::validate_flac_header(invalid_file.path()).is_err());

        Ok(())
    }

    #[test]
    fn test_mp3_header_validation() -> Result<()> {
        let mut file = NamedTempFile::new().unwrap();

        // Write valid MP3 header (ID3 tag)
        file.write_all(b"ID3").unwrap();
        file.flush().unwrap();

        assert!(AudioFormatValidator::validate_mp3_header(file.path()).is_ok());

        // Test MP3 frame sync
        let mut sync_file = NamedTempFile::new().unwrap();
        sync_file.write_all(&[0xFF, 0xFB, 0x00]).unwrap(); // Valid frame sync with third byte
        sync_file.flush().unwrap();

        assert!(AudioFormatValidator::validate_mp3_header(sync_file.path()).is_ok());

        // Test invalid header
        let mut invalid_file = NamedTempFile::new().unwrap();
        invalid_file.write_all(b"INVALID").unwrap();
        invalid_file.flush().unwrap();

        assert!(AudioFormatValidator::validate_mp3_header(invalid_file.path()).is_err());

        Ok(())
    }

    #[test]
    fn test_ogg_header_validation() -> Result<()> {
        let mut file = NamedTempFile::new().unwrap();

        // Write valid OGG header
        file.write_all(b"OggS").unwrap();
        file.flush().unwrap();

        assert!(AudioFormatValidator::validate_ogg_header(file.path()).is_ok());

        // Test invalid header
        let mut invalid_file = NamedTempFile::new().unwrap();
        invalid_file.write_all(b"INVALID").unwrap();
        invalid_file.flush().unwrap();

        assert!(AudioFormatValidator::validate_ogg_header(invalid_file.path()).is_err());

        Ok(())
    }

    #[test]
    fn test_phoneme_sequence_creation() {
        let mut sequence = PhonemeSequence::new("IPA".to_string(), LanguageCode::EnUs);

        let mut phoneme1 = Phoneme::new("h");
        phoneme1.duration = Some(0.1);
        sequence.add_phoneme(phoneme1);

        let mut phoneme2 = Phoneme::new("ɛ");
        phoneme2.duration = Some(0.15);
        sequence.add_phoneme(phoneme2);

        sequence.calculate_duration();

        assert_eq!(sequence.phonemes.len(), 2);
        assert_eq!(sequence.total_duration, Some(0.25));
    }

    #[test]
    fn test_phoneme_ipa_parsing() -> Result<()> {
        let content = "h 0.1\nɛ 0.15\nl 0.12\no 0.18";
        let sequence = PhonemeSequence::parse_ipa(content)?;

        assert_eq!(sequence.phonemes.len(), 4);
        assert_eq!(sequence.phonemes[0].symbol, "h");
        assert_eq!(sequence.phonemes[0].duration, Some(0.1));
        assert_eq!(sequence.phonemes[1].symbol, "ɛ");
        assert_eq!(sequence.phonemes[1].duration, Some(0.15));

        Ok(())
    }

    #[test]
    fn test_alignment_entry_creation() {
        let entry = AlignmentEntry::new(0.0, 0.5, "hello".to_string());

        assert_eq!(entry.start_time, 0.0);
        assert_eq!(entry.end_time, 0.5);
        assert_eq!(entry.text, "hello");
        assert_eq!(entry.duration(), 0.5);
    }

    #[test]
    fn test_alignment_sequence_creation() {
        let mut sequence = AlignmentSequence::new("MFA".to_string());

        let entry1 = AlignmentEntry::new(0.0, 0.5, "hello".to_string());
        let entry2 = AlignmentEntry::new(0.5, 1.0, "world".to_string());

        sequence.add_entry(entry1);
        sequence.add_entry(entry2);

        assert_eq!(sequence.entries.len(), 2);
        assert_eq!(sequence.total_duration, 1.0);
    }

    #[test]
    fn test_mfa_alignment_parsing() -> Result<()> {
        let content = "0.0 0.5 hello\n0.5 1.0 world\n1.0 1.5 test";
        let sequence = AlignmentSequence::parse_mfa(content)?;

        assert_eq!(sequence.entries.len(), 3);
        assert_eq!(sequence.entries[0].start_time, 0.0);
        assert_eq!(sequence.entries[0].end_time, 0.5);
        assert_eq!(sequence.entries[0].text, "hello");
        assert_eq!(sequence.entries[2].text, "test");

        Ok(())
    }

    #[test]
    fn test_lab_alignment_parsing() -> Result<()> {
        let content = "0.0 0.5 hello\n0.5 1.0 world test\n1.0 1.5 final";
        let sequence = AlignmentSequence::parse_lab(content)?;

        assert_eq!(sequence.entries.len(), 3);
        assert_eq!(sequence.entries[1].text, "world test");
        assert_eq!(sequence.entries[2].text, "final");

        Ok(())
    }
}
