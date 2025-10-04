//! Voice management and voice banking

#![allow(dead_code)]

use crate::types::{Expression, VoiceCharacteristics, VoiceType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Voice manager for handling voice collections
pub struct VoiceManager {
    voices: HashMap<String, VoiceCharacteristics>,
    voice_bank_path: Option<String>,
}

/// Voice controller for voice manipulation
pub struct VoiceController {
    current_voice: VoiceCharacteristics,
}

/// Voice bank metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceBankMetadata {
    /// Name of the voice bank
    pub name: String,
    /// Version identifier (e.g., "1.0.0")
    pub version: String,
    /// Author or creator of the voice bank
    pub author: String,
    /// Textual description of the voice bank contents and purpose
    pub description: String,
    /// Primary language code (e.g., "en", "ja", "es")
    pub language: String,
    /// Musical genre or category (e.g., "classical", "pop", "mixed")
    pub genre: String,
    /// Timestamp when the voice bank was created
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Timestamp of the last modification
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Total number of voices in the bank
    pub voice_count: usize,
    /// Searchable tags for categorization and filtering
    pub tags: Vec<String>,
}

/// Individual voice metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceMetadata {
    /// Internal identifier for the voice
    pub name: String,
    /// Human-friendly display name for the voice
    pub display_name: String,
    /// Textual description of voice characteristics and intended use
    pub description: String,
    /// Primary singing style category
    pub style: SingingStyle,
    /// Technical difficulty level for singers
    pub difficulty_level: DifficultyLevel,
    /// Recommended tempo range in BPM (min, max)
    pub recommended_tempo_range: (f32, f32),
    /// Vocal range as note names (lowest, highest)
    pub vocal_range_notes: (String, String),
    /// Supported language codes for this voice
    pub languages: Vec<String>,
    /// Searchable tags for categorization and filtering
    pub tags: Vec<String>,
    /// Timestamp when the voice metadata was created
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Singing style categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SingingStyle {
    /// Classical art song style with traditional technique and phrasing
    Classical,
    /// Contemporary popular music style with minimal vibrato and commercial appeal
    Pop,
    /// Jazz style with swing phrasing, improvisation, and expressive vibrato
    Jazz,
    /// Operatic style with powerful projection, dramatic expression, and extensive vibrato
    Opera,
    /// Traditional folk music style with natural, unadorned vocal quality
    Folk,
    /// Gospel music style with emotional intensity, melisma, and spiritual expression
    Gospel,
    /// Rock music style with raw power, edge, and aggressive vocal techniques
    Rock,
    /// Country music style with twang, storytelling phrasing, and emotional authenticity
    Country,
    /// Musical theatre style with clear diction, character portrayal, and theatrical expression
    Musical,
    /// Experimental and avant-garde vocal techniques exploring extended vocal possibilities
    Experimental,
}

/// Voice difficulty level for singers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DifficultyLevel {
    /// Suitable for novice singers with limited technique and range
    Beginner,
    /// Suitable for developing singers with moderate technical skills
    Intermediate,
    /// Suitable for skilled singers with strong technique and extended range
    Advanced,
    /// Suitable for expert singers with professional-level control and artistry
    Professional,
    /// Suitable for virtuoso singers with exceptional mastery and vocal command
    Master,
}

/// Voice bank for storing voice collections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceBank {
    /// Metadata describing the voice bank
    pub metadata: VoiceBankMetadata,
    /// Collection of voice entries indexed by voice name
    pub voices: HashMap<String, VoiceEntry>,
}

/// Voice entry in a voice bank
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceEntry {
    /// Metadata describing the voice properties and usage
    pub metadata: VoiceMetadata,
    /// Acoustic and synthesis characteristics of the voice
    pub characteristics: VoiceCharacteristics,
}

impl VoiceManager {
    /// Create a new voice manager
    pub fn new() -> Self {
        Self {
            voices: HashMap::new(),
            voice_bank_path: None,
        }
    }

    /// Create voice manager with default voice banks
    pub fn with_defaults() -> Self {
        let mut manager = Self::new();
        manager.load_default_voices();
        manager
    }

    /// Load default built-in voices
    pub fn load_default_voices(&mut self) {
        // Classical voices
        self.voices
            .insert("soprano".to_string(), Self::create_soprano_voice());
        self.voices
            .insert("alto".to_string(), Self::create_alto_voice());
        self.voices
            .insert("tenor".to_string(), Self::create_tenor_voice());
        self.voices
            .insert("bass".to_string(), Self::create_bass_voice());

        // Popular music voices
        self.voices
            .insert("pop_female".to_string(), Self::create_pop_female_voice());
        self.voices
            .insert("pop_male".to_string(), Self::create_pop_male_voice());

        // Jazz voices
        self.voices
            .insert("jazz_sultry".to_string(), Self::create_jazz_sultry_voice());
        self.voices.insert(
            "jazz_crooner".to_string(),
            Self::create_jazz_crooner_voice(),
        );
    }

    /// Set voice bank path
    pub fn set_voice_bank_path(&mut self, path: String) {
        self.voice_bank_path = Some(path);
    }

    /// Load a voice from the specified path
    pub async fn load_voice(&self, path: &str) -> crate::Result<VoiceCharacteristics> {
        let content = fs::read_to_string(path).map_err(crate::Error::Io)?;
        let voice: VoiceCharacteristics = serde_json::from_str(&content)
            .map_err(|e| crate::Error::Format(format!("Failed to deserialize voice: {e}")))?;
        Ok(voice)
    }

    /// Save a voice to the specified path
    pub async fn save_voice(&self, voice: &VoiceCharacteristics, path: &str) -> crate::Result<()> {
        let content = serde_json::to_string_pretty(voice)
            .map_err(|e| crate::Error::Format(format!("Failed to serialize voice: {e}")))?;
        fs::write(path, content).map_err(crate::Error::Io)?;
        Ok(())
    }

    /// Load a voice bank from file
    pub async fn load_voice_bank(&mut self, path: &str) -> crate::Result<()> {
        let content = fs::read_to_string(path).map_err(crate::Error::Io)?;
        let voice_bank: VoiceBank = serde_json::from_str(&content)
            .map_err(|e| crate::Error::Format(format!("Failed to deserialize voice bank: {e}")))?;

        // Add voices from the bank to the manager
        for (name, entry) in voice_bank.voices {
            self.voices.insert(name, entry.characteristics);
        }

        self.voice_bank_path = Some(path.to_string());
        Ok(())
    }

    /// Save current voices as a voice bank
    pub async fn save_voice_bank(
        &self,
        path: &str,
        metadata: VoiceBankMetadata,
    ) -> crate::Result<()> {
        let mut voice_entries = HashMap::new();

        for (name, characteristics) in &self.voices {
            let voice_metadata = VoiceMetadata {
                name: name.clone(),
                display_name: name.clone(),
                description: format!("Voice: {name}"),
                style: Self::infer_singing_style(characteristics),
                difficulty_level: DifficultyLevel::Intermediate,
                recommended_tempo_range: (60.0, 180.0),
                vocal_range_notes: Self::get_vocal_range_notes(characteristics),
                languages: vec!["en".to_string()],
                tags: vec![],
                created_at: chrono::Utc::now(),
            };

            voice_entries.insert(
                name.clone(),
                VoiceEntry {
                    metadata: voice_metadata,
                    characteristics: characteristics.clone(),
                },
            );
        }

        let voice_bank = VoiceBank {
            metadata,
            voices: voice_entries,
        };

        let content = serde_json::to_string_pretty(&voice_bank)
            .map_err(|e| crate::Error::Format(format!("Failed to serialize voice bank: {e}")))?;
        fs::write(path, content).map_err(crate::Error::Io)?;

        Ok(())
    }

    /// List all available voices
    pub async fn list_voices(&self) -> crate::Result<Vec<String>> {
        Ok(self.voices.keys().cloned().collect())
    }

    /// Add a voice with the given name
    pub async fn add_voice(
        &mut self,
        name: String,
        voice: VoiceCharacteristics,
    ) -> crate::Result<()> {
        self.voices.insert(name, voice);
        Ok(())
    }

    /// Remove a voice by name
    pub async fn remove_voice(&mut self, name: &str) -> crate::Result<()> {
        self.voices.remove(name);
        Ok(())
    }

    /// Get a voice by name
    pub async fn get_voice(&self, name: &str) -> Option<&VoiceCharacteristics> {
        self.voices.get(name)
    }

    /// Filter voices by singing style
    pub async fn get_voices_by_style(
        &self,
        style: SingingStyle,
    ) -> Vec<(String, &VoiceCharacteristics)> {
        self.voices
            .iter()
            .filter(|(_, voice)| Self::infer_singing_style(voice) == style)
            .map(|(name, voice)| (name.clone(), voice))
            .collect()
    }

    /// Filter voices by voice type
    pub async fn get_voices_by_type(
        &self,
        voice_type: VoiceType,
    ) -> Vec<(String, &VoiceCharacteristics)> {
        self.voices
            .iter()
            .filter(|(_, voice)| voice.voice_type == voice_type)
            .map(|(name, voice)| (name.clone(), voice))
            .collect()
    }

    /// Create predefined voice characteristics
    fn create_soprano_voice() -> VoiceCharacteristics {
        let mut characteristics = VoiceCharacteristics::default();
        characteristics.voice_type = VoiceType::Soprano;
        characteristics.range = VoiceType::Soprano.frequency_range();
        characteristics.f0_mean = VoiceType::Soprano.f0_mean();
        characteristics.f0_std = 75.0;
        characteristics.vibrato_frequency = 6.5;
        characteristics.vibrato_depth = 0.4;
        characteristics.breath_capacity = 12.0;
        characteristics.vocal_power = 0.9;
        characteristics
    }

    fn create_alto_voice() -> VoiceCharacteristics {
        let mut characteristics = VoiceCharacteristics::default();
        characteristics.voice_type = VoiceType::Alto;
        characteristics.range = VoiceType::Alto.frequency_range();
        characteristics.f0_mean = VoiceType::Alto.f0_mean();
        characteristics.f0_std = 60.0;
        characteristics.vibrato_frequency = 5.8;
        characteristics.vibrato_depth = 0.35;
        characteristics.breath_capacity = 10.0;
        characteristics.vocal_power = 0.85;
        characteristics
    }

    fn create_tenor_voice() -> VoiceCharacteristics {
        let mut characteristics = VoiceCharacteristics::default();
        characteristics.voice_type = VoiceType::Tenor;
        characteristics.range = VoiceType::Tenor.frequency_range();
        characteristics.f0_mean = VoiceType::Tenor.f0_mean();
        characteristics.f0_std = 55.0;
        characteristics.vibrato_frequency = 6.0;
        characteristics.vibrato_depth = 0.3;
        characteristics.breath_capacity = 11.0;
        characteristics.vocal_power = 0.8;
        characteristics
    }

    fn create_bass_voice() -> VoiceCharacteristics {
        let mut characteristics = VoiceCharacteristics::default();
        characteristics.voice_type = VoiceType::Bass;
        characteristics.range = VoiceType::Bass.frequency_range();
        characteristics.f0_mean = VoiceType::Bass.f0_mean();
        characteristics.f0_std = 45.0;
        characteristics.vibrato_frequency = 5.5;
        characteristics.vibrato_depth = 0.25;
        characteristics.breath_capacity = 14.0;
        characteristics.vocal_power = 0.95;
        characteristics
    }

    fn create_pop_female_voice() -> VoiceCharacteristics {
        let mut characteristics = VoiceCharacteristics::default();
        characteristics.voice_type = VoiceType::MezzoSoprano;
        characteristics.range = (220.0, 880.0); // A3 to A5
        characteristics.f0_mean = 440.0; // A4
        characteristics.f0_std = 80.0;
        characteristics.vibrato_frequency = 5.0;
        characteristics.vibrato_depth = 0.2; // Less vibrato for pop style
        characteristics.breath_capacity = 8.0;
        characteristics.vocal_power = 0.75;
        characteristics
    }

    fn create_pop_male_voice() -> VoiceCharacteristics {
        let mut characteristics = VoiceCharacteristics::default();
        characteristics.voice_type = VoiceType::Baritone;
        characteristics.range = (98.0, 392.0); // G2 to G4
        characteristics.f0_mean = 196.0; // G3
        characteristics.f0_std = 50.0;
        characteristics.vibrato_frequency = 4.5;
        characteristics.vibrato_depth = 0.15;
        characteristics.breath_capacity = 9.0;
        characteristics.vocal_power = 0.7;
        characteristics
    }

    fn create_jazz_sultry_voice() -> VoiceCharacteristics {
        let mut characteristics = VoiceCharacteristics::default();
        characteristics.voice_type = VoiceType::Alto;
        characteristics.range = (175.0, 698.5); // F3 to F5
        characteristics.f0_mean = 330.0; // E4
        characteristics.f0_std = 70.0;
        characteristics.vibrato_frequency = 4.0;
        characteristics.vibrato_depth = 0.5; // More expressive vibrato
        characteristics.breath_capacity = 10.0;
        characteristics.vocal_power = 0.6; // More intimate
        characteristics
    }

    fn create_jazz_crooner_voice() -> VoiceCharacteristics {
        let mut characteristics = VoiceCharacteristics::default();
        characteristics.voice_type = VoiceType::Baritone;
        characteristics.range = (87.3, 349.2); // F2 to F4
        characteristics.f0_mean = 174.6; // F3
        characteristics.f0_std = 45.0;
        characteristics.vibrato_frequency = 3.5;
        characteristics.vibrato_depth = 0.4;
        characteristics.breath_capacity = 12.0;
        characteristics.vocal_power = 0.65;
        characteristics
    }

    fn infer_singing_style(characteristics: &VoiceCharacteristics) -> SingingStyle {
        // Simple heuristic to infer singing style from voice characteristics
        // Check jazz characteristics first (low frequency + moderate depth)
        if characteristics.vibrato_frequency <= 4.0 && characteristics.vibrato_depth >= 0.35 {
            return SingingStyle::Jazz;
        }

        // Check pop characteristics (very low vibrato)
        if characteristics.vibrato_depth <= 0.25 {
            return SingingStyle::Pop;
        }

        // Check classical/opera characteristics (high vibrato depth)
        if characteristics.vibrato_depth > 0.35 {
            match characteristics.voice_type {
                VoiceType::Soprano | VoiceType::MezzoSoprano | VoiceType::Alto => {
                    SingingStyle::Classical
                }
                VoiceType::Tenor | VoiceType::Baritone | VoiceType::Bass => SingingStyle::Opera,
            }
        } else {
            SingingStyle::Classical
        }
    }

    fn get_vocal_range_notes(characteristics: &VoiceCharacteristics) -> (String, String) {
        // Convert frequency range to note names (simplified)
        let low_note = Self::frequency_to_note_name(characteristics.range.0);
        let high_note = Self::frequency_to_note_name(characteristics.range.1);
        (low_note, high_note)
    }

    fn frequency_to_note_name(frequency: f32) -> String {
        let notes = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        let a4 = 440.0;
        let semitones_from_a4 = (12.0 * (frequency / a4).log2()).round() as i32;
        let semitone_index = ((semitones_from_a4 + 9) % 12) as usize;
        let octave = 4 + (semitones_from_a4 + 9) / 12;
        format!("{}{octave}", notes[semitone_index])
    }
}

impl VoiceController {
    /// Create a new voice controller with the given voice
    pub fn new(voice: VoiceCharacteristics) -> Self {
        Self {
            current_voice: voice,
        }
    }

    /// Get the current voice characteristics
    pub fn get_voice(&self) -> &VoiceCharacteristics {
        &self.current_voice
    }

    /// Set the current voice characteristics
    pub fn set_voice(&mut self, voice: VoiceCharacteristics) {
        self.current_voice = voice;
    }
}

impl VoiceBank {
    /// Create a new voice bank with metadata
    pub fn new(metadata: VoiceBankMetadata) -> Self {
        Self {
            metadata,
            voices: HashMap::new(),
        }
    }

    /// Create a default voice bank
    pub fn default_bank() -> Self {
        let metadata = VoiceBankMetadata {
            name: "Default Voice Bank".to_string(),
            version: "1.0.0".to_string(),
            author: "VoiRS".to_string(),
            description: "Built-in voice collection for singing synthesis".to_string(),
            language: "en".to_string(),
            genre: "mixed".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            voice_count: 0,
            tags: vec!["default", "singing", "classical", "pop", "jazz"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        };

        Self::new(metadata)
    }

    /// Add a voice entry to the bank
    pub fn add_voice_entry(&mut self, name: String, entry: VoiceEntry) {
        self.voices.insert(name, entry);
        self.metadata.voice_count = self.voices.len();
        self.metadata.updated_at = chrono::Utc::now();
    }

    /// Add a voice with metadata to the bank
    pub fn add_voice(
        &mut self,
        name: String,
        characteristics: VoiceCharacteristics,
        metadata: VoiceMetadata,
    ) {
        let entry = VoiceEntry {
            metadata,
            characteristics,
        };
        self.add_voice_entry(name, entry);
    }

    /// Get a voice entry from the bank by name
    pub fn get_voice_entry(&self, name: &str) -> Option<&VoiceEntry> {
        self.voices.get(name)
    }

    /// Get voice characteristics from the bank by name
    pub fn get_voice(&self, name: &str) -> Option<&VoiceCharacteristics> {
        self.voices.get(name).map(|entry| &entry.characteristics)
    }

    /// Get voice metadata by name
    pub fn get_voice_metadata(&self, name: &str) -> Option<&VoiceMetadata> {
        self.voices.get(name).map(|entry| &entry.metadata)
    }

    /// List all voices in the bank
    pub fn list_voices(&self) -> Vec<String> {
        self.voices.keys().cloned().collect()
    }

    /// Filter voices by singing style
    pub fn filter_by_style(&self, style: SingingStyle) -> Vec<(String, &VoiceEntry)> {
        self.voices
            .iter()
            .filter(|(_, entry)| entry.metadata.style == style)
            .map(|(name, entry)| (name.clone(), entry))
            .collect()
    }

    /// Filter voices by voice type
    pub fn filter_by_voice_type(&self, voice_type: VoiceType) -> Vec<(String, &VoiceEntry)> {
        self.voices
            .iter()
            .filter(|(_, entry)| entry.characteristics.voice_type == voice_type)
            .map(|(name, entry)| (name.clone(), entry))
            .collect()
    }

    /// Filter voices by difficulty level
    pub fn filter_by_difficulty(&self, difficulty: DifficultyLevel) -> Vec<(String, &VoiceEntry)> {
        self.voices
            .iter()
            .filter(|(_, entry)| entry.metadata.difficulty_level == difficulty)
            .map(|(name, entry)| (name.clone(), entry))
            .collect()
    }

    /// Get voices suitable for a tempo range
    pub fn filter_by_tempo_range(
        &self,
        min_tempo: f32,
        max_tempo: f32,
    ) -> Vec<(String, &VoiceEntry)> {
        self.voices
            .iter()
            .filter(|(_, entry)| {
                let (voice_min, voice_max) = entry.metadata.recommended_tempo_range;
                !(max_tempo < voice_min || min_tempo > voice_max) // Ranges overlap
            })
            .map(|(name, entry)| (name.clone(), entry))
            .collect()
    }

    /// Remove a voice from the bank
    pub fn remove_voice(&mut self, name: &str) -> Option<VoiceEntry> {
        let result = self.voices.remove(name);
        self.metadata.voice_count = self.voices.len();
        self.metadata.updated_at = chrono::Utc::now();
        result
    }

    /// Get bank statistics
    pub fn get_statistics(&self) -> VoiceBankStatistics {
        let mut style_counts = HashMap::new();
        let mut voice_type_counts = HashMap::new();
        let mut difficulty_counts = HashMap::new();
        let mut total_range = (f32::MAX, f32::MIN);

        for entry in self.voices.values() {
            // Count styles
            *style_counts.entry(entry.metadata.style).or_insert(0) += 1;

            // Count voice types
            *voice_type_counts
                .entry(entry.characteristics.voice_type)
                .or_insert(0) += 1;

            // Count difficulty levels
            *difficulty_counts
                .entry(entry.metadata.difficulty_level)
                .or_insert(0) += 1;

            // Calculate total range
            let (low, high) = entry.characteristics.range;
            total_range.0 = total_range.0.min(low);
            total_range.1 = total_range.1.max(high);
        }

        VoiceBankStatistics {
            total_voices: self.voices.len(),
            style_distribution: style_counts,
            voice_type_distribution: voice_type_counts,
            difficulty_distribution: difficulty_counts,
            frequency_range: if total_range.0 != f32::MAX {
                Some(total_range)
            } else {
                None
            },
        }
    }
}

/// Voice bank statistics
#[derive(Debug, Clone)]
pub struct VoiceBankStatistics {
    /// Total number of voices in the bank
    pub total_voices: usize,
    /// Count of voices per singing style
    pub style_distribution: HashMap<SingingStyle, usize>,
    /// Count of voices per voice type
    pub voice_type_distribution: HashMap<VoiceType, usize>,
    /// Count of voices per difficulty level
    pub difficulty_distribution: HashMap<DifficultyLevel, usize>,
    /// Overall frequency range covering all voices in Hz (min, max), None if bank is empty
    pub frequency_range: Option<(f32, f32)>,
}

/// Implementation of default metadata creation
impl Default for VoiceBankMetadata {
    fn default() -> Self {
        Self {
            name: "Unnamed Voice Bank".to_string(),
            version: "1.0.0".to_string(),
            author: "Unknown".to_string(),
            description: "A collection of singing voices".to_string(),
            language: "en".to_string(),
            genre: "mixed".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            voice_count: 0,
            tags: Vec::new(),
        }
    }
}

impl Default for VoiceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for VoiceBank {
    fn default() -> Self {
        Self::default_bank()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[test]
    fn test_voice_manager_creation() {
        let manager = VoiceManager::new();
        assert_eq!(manager.voices.len(), 0);
        assert!(manager.voice_bank_path.is_none());
    }

    #[test]
    fn test_voice_manager_with_defaults() {
        let manager = VoiceManager::with_defaults();
        assert_eq!(manager.voices.len(), 8); // 4 classical + 2 pop + 2 jazz

        // Test that all expected voices are present
        assert!(manager.voices.contains_key("soprano"));
        assert!(manager.voices.contains_key("alto"));
        assert!(manager.voices.contains_key("tenor"));
        assert!(manager.voices.contains_key("bass"));
        assert!(manager.voices.contains_key("pop_female"));
        assert!(manager.voices.contains_key("pop_male"));
        assert!(manager.voices.contains_key("jazz_sultry"));
        assert!(manager.voices.contains_key("jazz_crooner"));
    }

    #[tokio::test]
    async fn test_voice_manager_operations() {
        let mut manager = VoiceManager::new();
        let test_voice = VoiceCharacteristics::default();

        // Test adding voice
        assert!(manager
            .add_voice("test_voice".to_string(), test_voice.clone())
            .await
            .is_ok());
        assert_eq!(manager.voices.len(), 1);

        // Test getting voice
        let retrieved_voice = manager.get_voice("test_voice").await;
        assert!(retrieved_voice.is_some());

        // Test listing voices
        let voice_list = manager.list_voices().await.unwrap();
        assert_eq!(voice_list.len(), 1);
        assert!(voice_list.contains(&"test_voice".to_string()));

        // Test removing voice
        assert!(manager.remove_voice("test_voice").await.is_ok());
        assert_eq!(manager.voices.len(), 0);
    }

    #[tokio::test]
    async fn test_voice_filtering() {
        let manager = VoiceManager::with_defaults();

        // Test filtering by voice type
        let sopranos = manager.get_voices_by_type(VoiceType::Soprano).await;
        assert_eq!(sopranos.len(), 1);
        assert_eq!(sopranos[0].0, "soprano");

        // Test filtering by style
        let classical_voices = manager.get_voices_by_style(SingingStyle::Classical).await;
        assert!(!classical_voices.is_empty());

        let pop_voices = manager.get_voices_by_style(SingingStyle::Pop).await;
        assert!(!pop_voices.is_empty());

        let jazz_voices = manager.get_voices_by_style(SingingStyle::Jazz).await;
        assert!(!jazz_voices.is_empty());
    }

    #[test]
    fn test_voice_bank_creation() {
        let bank = VoiceBank::default();
        assert_eq!(bank.metadata.name, "Default Voice Bank");
        assert_eq!(bank.voices.len(), 0);
        assert_eq!(bank.metadata.voice_count, 0);
    }

    #[test]
    fn test_voice_bank_operations() {
        let mut bank = VoiceBank::default();

        let voice_characteristics = VoiceCharacteristics::default();
        let voice_metadata = VoiceMetadata {
            name: "test_voice".to_string(),
            display_name: "Test Voice".to_string(),
            description: "A test voice".to_string(),
            style: SingingStyle::Pop,
            difficulty_level: DifficultyLevel::Beginner,
            recommended_tempo_range: (60.0, 120.0),
            vocal_range_notes: ("C3".to_string(), "C5".to_string()),
            languages: vec!["en".to_string()],
            tags: vec!["test".to_string()],
            created_at: chrono::Utc::now(),
        };

        // Test adding voice
        bank.add_voice(
            "test_voice".to_string(),
            voice_characteristics,
            voice_metadata,
        );
        assert_eq!(bank.voices.len(), 1);
        assert_eq!(bank.metadata.voice_count, 1);

        // Test getting voice
        let retrieved_voice = bank.get_voice("test_voice");
        assert!(retrieved_voice.is_some());

        let retrieved_metadata = bank.get_voice_metadata("test_voice");
        assert!(retrieved_metadata.is_some());
        assert_eq!(retrieved_metadata.unwrap().style, SingingStyle::Pop);

        // Test listing voices
        let voice_list = bank.list_voices();
        assert_eq!(voice_list.len(), 1);
        assert!(voice_list.contains(&"test_voice".to_string()));

        // Test filtering
        let pop_voices = bank.filter_by_style(SingingStyle::Pop);
        assert_eq!(pop_voices.len(), 1);

        let beginner_voices = bank.filter_by_difficulty(DifficultyLevel::Beginner);
        assert_eq!(beginner_voices.len(), 1);

        let tempo_filtered = bank.filter_by_tempo_range(80.0, 100.0);
        assert_eq!(tempo_filtered.len(), 1);

        // Test statistics
        let stats = bank.get_statistics();
        assert_eq!(stats.total_voices, 1);
        assert_eq!(stats.style_distribution[&SingingStyle::Pop], 1);

        // Test removing voice
        let removed = bank.remove_voice("test_voice");
        assert!(removed.is_some());
        assert_eq!(bank.voices.len(), 0);
        assert_eq!(bank.metadata.voice_count, 0);
    }

    #[test]
    fn test_voice_characteristics_creation() {
        let soprano = VoiceManager::create_soprano_voice();
        assert_eq!(soprano.voice_type, VoiceType::Soprano);
        assert_eq!(soprano.range, VoiceType::Soprano.frequency_range());
        assert_eq!(soprano.f0_mean, VoiceType::Soprano.f0_mean());

        let bass = VoiceManager::create_bass_voice();
        assert_eq!(bass.voice_type, VoiceType::Bass);
        assert!(bass.f0_mean < soprano.f0_mean); // Bass should be lower than soprano
    }

    #[test]
    fn test_singing_style_inference() {
        let classical_voice = VoiceManager::create_soprano_voice();
        let inferred_style = VoiceManager::infer_singing_style(&classical_voice);
        assert_eq!(inferred_style, SingingStyle::Classical);

        let pop_voice = VoiceManager::create_pop_female_voice();
        let inferred_style = VoiceManager::infer_singing_style(&pop_voice);
        assert_eq!(inferred_style, SingingStyle::Pop);
    }

    #[test]
    fn test_frequency_to_note_conversion() {
        // Test A4 = 440 Hz
        let note = VoiceManager::frequency_to_note_name(440.0);
        assert_eq!(note, "A4");

        // Test C4 ≈ 261.63 Hz
        let note = VoiceManager::frequency_to_note_name(261.63);
        assert_eq!(note, "C4");

        // Test higher octaves
        let note = VoiceManager::frequency_to_note_name(880.0); // A5
        assert_eq!(note, "A5");
    }

    #[test]
    fn test_voice_controller() {
        let voice = VoiceCharacteristics::default();
        let mut controller = VoiceController::new(voice.clone());

        assert_eq!(controller.get_voice().voice_type, voice.voice_type);

        let new_voice = VoiceManager::create_soprano_voice();
        controller.set_voice(new_voice.clone());
        assert_eq!(controller.get_voice().voice_type, new_voice.voice_type);
    }

    #[test]
    fn test_voice_metadata_serialization() {
        let metadata = VoiceMetadata {
            name: "test".to_string(),
            display_name: "Test Voice".to_string(),
            description: "Test description".to_string(),
            style: SingingStyle::Jazz,
            difficulty_level: DifficultyLevel::Advanced,
            recommended_tempo_range: (80.0, 160.0),
            vocal_range_notes: ("F3".to_string(), "F5".to_string()),
            languages: vec!["en".to_string(), "fr".to_string()],
            tags: vec!["jazz", "smooth"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            created_at: chrono::Utc::now(),
        };

        // Test serialization
        let serialized = serde_json::to_string(&metadata).unwrap();
        assert!(!serialized.is_empty());

        // Test deserialization
        let deserialized: VoiceMetadata = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.name, metadata.name);
        assert_eq!(deserialized.style, metadata.style);
        assert_eq!(deserialized.difficulty_level, metadata.difficulty_level);
    }
}
