//! Dataset format definitions and utilities.

// TODO: Implement format support:
// - Metadata file formats (JSON, YAML, CSV)
// - Audio file format validation
// - Phoneme file formats
// - Alignment file formats

use serde::{Deserialize, Serialize};
use crate::LanguageCode;

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
}

/// Dataset manifest entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    pub id: String,
    pub text: String,
    pub audio_path: String,
    pub phonemes_path: Option<String>,
    pub speaker_id: Option<String>,
    pub duration: Option<f32>,
}