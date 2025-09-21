//! Speaker control and voice characteristics management.

pub mod characteristics;
pub mod cloning;
pub mod emotion;
pub mod multi;

pub use characteristics::*;
pub use cloning::*;
pub use emotion::*;
pub use multi::*;

use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Speaker identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpeakerId(pub u32);

impl SpeakerId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn id(&self) -> u32 {
        self.0
    }
}

impl From<u32> for SpeakerId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

/// Speaker metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerMetadata {
    /// Speaker identifier
    pub id: SpeakerId,
    /// Speaker name
    pub name: String,
    /// Speaker description
    pub description: Option<String>,
    /// Speaker characteristics
    pub characteristics: VoiceCharacteristics,
    /// Default emotion setting
    pub default_emotion: EmotionConfig,
    /// Supported languages
    pub supported_languages: Vec<crate::LanguageCode>,
}

/// Speaker embedding vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerEmbedding {
    /// Speaker ID
    pub speaker_id: SpeakerId,
    /// Embedding vector
    pub embedding: Vec<f32>,
    /// Embedding dimension
    pub dimension: usize,
}

impl SpeakerEmbedding {
    /// Create new speaker embedding
    pub fn new(speaker_id: SpeakerId, embedding: Vec<f32>) -> Self {
        let dimension = embedding.len();
        Self {
            speaker_id,
            embedding,
            dimension,
        }
    }

    /// Get embedding vector
    pub fn vector(&self) -> &[f32] {
        &self.embedding
    }

    /// Interpolate with another embedding
    pub fn interpolate(&self, other: &SpeakerEmbedding, alpha: f32) -> Result<SpeakerEmbedding> {
        if self.dimension != other.dimension {
            return Err(crate::AcousticError::InputError(
                "Speaker embeddings must have the same dimension".to_string(),
            ));
        }

        let mut interpolated = Vec::with_capacity(self.dimension);
        for (a, b) in self.embedding.iter().zip(other.embedding.iter()) {
            interpolated.push(a * (1.0 - alpha) + b * alpha);
        }

        Ok(SpeakerEmbedding::new(
            SpeakerId::new(0), // Combined speaker ID
            interpolated,
        ))
    }
}

/// Speaker registry for managing multiple speakers
#[derive(Debug, Clone)]
pub struct SpeakerRegistry {
    /// Speaker metadata
    speakers: HashMap<SpeakerId, SpeakerMetadata>,
    /// Speaker embeddings
    embeddings: HashMap<SpeakerId, SpeakerEmbedding>,
}

impl SpeakerRegistry {
    /// Create new speaker registry
    pub fn new() -> Self {
        Self {
            speakers: HashMap::new(),
            embeddings: HashMap::new(),
        }
    }

    /// Register a speaker
    pub fn register_speaker(&mut self, metadata: SpeakerMetadata, embedding: SpeakerEmbedding) {
        let speaker_id = metadata.id;
        self.speakers.insert(speaker_id, metadata);
        self.embeddings.insert(speaker_id, embedding);
    }

    /// Get speaker metadata
    pub fn get_speaker(&self, speaker_id: SpeakerId) -> Option<&SpeakerMetadata> {
        self.speakers.get(&speaker_id)
    }

    /// Get speaker embedding
    pub fn get_embedding(&self, speaker_id: SpeakerId) -> Option<&SpeakerEmbedding> {
        self.embeddings.get(&speaker_id)
    }

    /// List all speakers
    pub fn list_speakers(&self) -> Vec<SpeakerId> {
        self.speakers.keys().cloned().collect()
    }

    /// Find speakers by characteristics
    pub fn find_speakers_by_characteristics(
        &self,
        filter: &VoiceCharacteristics,
    ) -> Vec<SpeakerId> {
        self.speakers
            .iter()
            .filter(|(_, metadata)| metadata.characteristics.matches(filter))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Create interpolated speaker embedding
    pub fn interpolate_speakers(
        &self,
        speaker1: SpeakerId,
        speaker2: SpeakerId,
        alpha: f32,
    ) -> Result<SpeakerEmbedding> {
        let embedding1 = self.get_embedding(speaker1).ok_or_else(|| {
            crate::AcousticError::InputError(format!("Speaker {} not found", speaker1.id()))
        })?;
        let embedding2 = self.get_embedding(speaker2).ok_or_else(|| {
            crate::AcousticError::InputError(format!("Speaker {} not found", speaker2.id()))
        })?;

        embedding1.interpolate(embedding2, alpha)
    }
}

impl Default for SpeakerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Speaker embedding lookup table for efficient speaker conditioning
#[derive(Debug, Clone)]
pub struct SpeakerEmbeddingTable {
    /// Fast lookup table: SpeakerId -> embedding index
    lookup_table: HashMap<SpeakerId, usize>,
    /// Packed embedding matrix [num_speakers, embedding_dim]
    embedding_matrix: Vec<f32>,
    /// Embedding dimension
    embedding_dim: usize,
    /// Number of speakers
    num_speakers: usize,
}

impl SpeakerEmbeddingTable {
    /// Create new embedding table
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            lookup_table: HashMap::new(),
            embedding_matrix: Vec::new(),
            embedding_dim,
            num_speakers: 0,
        }
    }

    /// Add speaker embedding to lookup table
    pub fn add_speaker(
        &mut self,
        speaker_id: SpeakerId,
        embedding: &SpeakerEmbedding,
    ) -> Result<()> {
        if embedding.dimension != self.embedding_dim {
            return Err(crate::AcousticError::InputError(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.embedding_dim, embedding.dimension
            )));
        }

        // Check if speaker already exists
        if self.lookup_table.contains_key(&speaker_id) {
            return Err(crate::AcousticError::InputError(format!(
                "Speaker {} already exists in lookup table",
                speaker_id.id()
            )));
        }

        // Add to lookup table
        let index = self.num_speakers;
        self.lookup_table.insert(speaker_id, index);

        // Add to embedding matrix
        self.embedding_matrix
            .extend_from_slice(&embedding.embedding);
        self.num_speakers += 1;

        Ok(())
    }

    /// Get speaker embedding by ID
    pub fn get_embedding(&self, speaker_id: SpeakerId) -> Option<&[f32]> {
        let index = *self.lookup_table.get(&speaker_id)?;
        let start = index * self.embedding_dim;
        let end = start + self.embedding_dim;
        self.embedding_matrix.get(start..end)
    }

    /// Get speaker embedding index for model input
    pub fn get_speaker_index(&self, speaker_id: SpeakerId) -> Option<usize> {
        self.lookup_table.get(&speaker_id).copied()
    }

    /// Get all speaker IDs
    pub fn speaker_ids(&self) -> Vec<SpeakerId> {
        self.lookup_table.keys().copied().collect()
    }

    /// Get embedding matrix as slice for model input
    pub fn embedding_matrix(&self) -> &[f32] {
        &self.embedding_matrix
    }

    /// Get dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.num_speakers, self.embedding_dim)
    }
}

/// Voice style control for dynamic speaker conditioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceStyleControl {
    /// Base speaker ID
    pub base_speaker: SpeakerId,
    /// Emotion configuration
    pub emotion: EmotionConfig,
    /// Age modification factor (-1.0 to 1.0, 0.0 = no change)
    pub age_factor: f32,
    /// Gender modification factor (-1.0 = more masculine, 1.0 = more feminine)
    pub gender_factor: f32,
    /// Speech speed factor (0.5 = half speed, 2.0 = double speed)
    pub speed_factor: f32,
    /// Pitch shift in semitones (-12 to 12)
    pub pitch_shift: f32,
    /// Energy/volume factor (0.0 to 2.0)
    pub energy_factor: f32,
    /// Voice breathiness (0.0 = clear, 1.0 = breathy)
    pub breathiness: f32,
    /// Voice roughness (0.0 = smooth, 1.0 = rough)
    pub roughness: f32,
}

impl VoiceStyleControl {
    /// Create new voice style control
    pub fn new(base_speaker: SpeakerId) -> Self {
        Self {
            base_speaker,
            emotion: EmotionConfig::default(),
            age_factor: 0.0,
            gender_factor: 0.0,
            speed_factor: 1.0,
            pitch_shift: 0.0,
            energy_factor: 1.0,
            breathiness: 0.0,
            roughness: 0.0,
        }
    }

    /// Set emotion
    pub fn with_emotion(mut self, emotion: EmotionConfig) -> Self {
        self.emotion = emotion;
        self
    }

    /// Set age factor
    pub fn with_age_factor(mut self, factor: f32) -> Self {
        self.age_factor = factor.clamp(-1.0, 1.0);
        self
    }

    /// Set gender factor
    pub fn with_gender_factor(mut self, factor: f32) -> Self {
        self.gender_factor = factor.clamp(-1.0, 1.0);
        self
    }

    /// Set speed factor
    pub fn with_speed_factor(mut self, factor: f32) -> Self {
        self.speed_factor = factor.max(0.1);
        self
    }

    /// Set pitch shift
    pub fn with_pitch_shift(mut self, semitones: f32) -> Self {
        self.pitch_shift = semitones.clamp(-12.0, 12.0);
        self
    }

    /// Set energy factor
    pub fn with_energy_factor(mut self, factor: f32) -> Self {
        self.energy_factor = factor.max(0.0);
        self
    }

    /// Set breathiness
    pub fn with_breathiness(mut self, breathiness: f32) -> Self {
        self.breathiness = breathiness.clamp(0.0, 1.0);
        self
    }

    /// Set roughness
    pub fn with_roughness(mut self, roughness: f32) -> Self {
        self.roughness = roughness.clamp(0.0, 1.0);
        self
    }

    /// Apply style to speaker embedding
    pub fn apply_to_embedding(
        &self,
        base_embedding: &SpeakerEmbedding,
    ) -> Result<SpeakerEmbedding> {
        let mut modified_embedding = base_embedding.embedding.clone();

        // Apply age factor (affects higher frequencies in embedding space)
        if self.age_factor != 0.0 {
            let embedding_len = modified_embedding.len();
            for (i, value) in modified_embedding.iter_mut().enumerate() {
                let age_mod = 1.0 + self.age_factor * 0.1 * (i as f32 / embedding_len as f32);
                *value *= age_mod;
            }
        }

        // Apply gender factor (affects formant-related components)
        if self.gender_factor != 0.0 {
            let gender_scale = 1.0 + self.gender_factor * 0.15;
            for value in modified_embedding.iter_mut() {
                *value *= gender_scale;
            }
        }

        // Apply emotion modifications (simplified version)
        let emotion_factor = self.emotion.intensity.as_f32();
        if emotion_factor > 0.0 {
            for value in modified_embedding.iter_mut() {
                *value += emotion_factor * 0.1 * value.signum();
            }
        }

        Ok(SpeakerEmbedding::new(self.base_speaker, modified_embedding))
    }
}

impl Default for VoiceStyleControl {
    fn default() -> Self {
        Self::new(SpeakerId::new(0))
    }
}

/// Multi-speaker model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSpeakerConfig {
    /// Speaker embedding table
    pub embedding_table: Option<String>, // Path to embedding table file
    /// Whether to use speaker embeddings
    pub use_speaker_embeddings: bool,
    /// Default speaker ID when none specified
    pub default_speaker: Option<SpeakerId>,
    /// Maximum number of supported speakers
    pub max_speakers: usize,
    /// Embedding dimension
    pub embedding_dimension: usize,
    /// Whether to support speaker interpolation
    pub support_interpolation: bool,
}

impl MultiSpeakerConfig {
    /// Create new multi-speaker configuration
    pub fn new(max_speakers: usize, embedding_dimension: usize) -> Self {
        Self {
            embedding_table: None,
            use_speaker_embeddings: true,
            default_speaker: None,
            max_speakers,
            embedding_dimension,
            support_interpolation: true,
        }
    }

    /// Set embedding table path
    pub fn with_embedding_table(mut self, path: String) -> Self {
        self.embedding_table = Some(path);
        self
    }

    /// Set default speaker
    pub fn with_default_speaker(mut self, speaker_id: SpeakerId) -> Self {
        self.default_speaker = Some(speaker_id);
        self
    }

    /// Disable speaker embeddings
    pub fn without_speaker_embeddings(mut self) -> Self {
        self.use_speaker_embeddings = false;
        self
    }
}

impl Default for MultiSpeakerConfig {
    fn default() -> Self {
        Self::new(256, 256) // Default: 256 speakers, 256-dim embeddings
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LanguageCode;

    #[test]
    fn test_speaker_id_creation() {
        let id = SpeakerId::new(42);
        assert_eq!(id.id(), 42);

        let id2: SpeakerId = 42.into();
        assert_eq!(id, id2);
    }

    #[test]
    fn test_speaker_embedding_creation() {
        let embedding = SpeakerEmbedding::new(SpeakerId::new(1), vec![1.0, 2.0, 3.0]);
        assert_eq!(embedding.speaker_id, SpeakerId::new(1));
        assert_eq!(embedding.dimension, 3);
        assert_eq!(embedding.vector(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_speaker_embedding_interpolation() {
        let emb1 = SpeakerEmbedding::new(SpeakerId::new(1), vec![1.0, 2.0, 3.0]);
        let emb2 = SpeakerEmbedding::new(SpeakerId::new(2), vec![4.0, 5.0, 6.0]);

        let interpolated = emb1.interpolate(&emb2, 0.5).unwrap();
        assert_eq!(interpolated.vector(), &[2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_speaker_embedding_interpolation_dimension_mismatch() {
        let emb1 = SpeakerEmbedding::new(SpeakerId::new(1), vec![1.0, 2.0]);
        let emb2 = SpeakerEmbedding::new(SpeakerId::new(2), vec![4.0, 5.0, 6.0]);

        assert!(emb1.interpolate(&emb2, 0.5).is_err());
    }

    #[test]
    fn test_speaker_registry() {
        let mut registry = SpeakerRegistry::new();

        let metadata = SpeakerMetadata {
            id: SpeakerId::new(1),
            name: "Test Speaker".to_string(),
            description: Some("A test speaker".to_string()),
            characteristics: VoiceCharacteristics::default(),
            default_emotion: EmotionConfig::default(),
            supported_languages: vec![LanguageCode::EnUs],
        };

        let embedding = SpeakerEmbedding::new(SpeakerId::new(1), vec![1.0, 2.0, 3.0]);

        registry.register_speaker(metadata, embedding);

        assert!(registry.get_speaker(SpeakerId::new(1)).is_some());
        assert!(registry.get_embedding(SpeakerId::new(1)).is_some());
        assert_eq!(registry.list_speakers(), vec![SpeakerId::new(1)]);
    }

    #[test]
    fn test_speaker_embedding_table() {
        let mut table = SpeakerEmbeddingTable::new(3);

        let speaker1 = SpeakerId::new(1);
        let speaker2 = SpeakerId::new(2);

        let embedding1 = SpeakerEmbedding::new(speaker1, vec![1.0, 2.0, 3.0]);
        let embedding2 = SpeakerEmbedding::new(speaker2, vec![4.0, 5.0, 6.0]);

        // Add speakers
        assert!(table.add_speaker(speaker1, &embedding1).is_ok());
        assert!(table.add_speaker(speaker2, &embedding2).is_ok());

        // Test retrieval
        assert_eq!(
            table.get_embedding(speaker1),
            Some([1.0, 2.0, 3.0].as_slice())
        );
        assert_eq!(
            table.get_embedding(speaker2),
            Some([4.0, 5.0, 6.0].as_slice())
        );

        // Test indices
        assert_eq!(table.get_speaker_index(speaker1), Some(0));
        assert_eq!(table.get_speaker_index(speaker2), Some(1));

        // Test dimensions
        assert_eq!(table.dimensions(), (2, 3));

        // Test speaker IDs
        let mut speaker_ids = table.speaker_ids();
        speaker_ids.sort_by_key(|id| id.id());
        assert_eq!(speaker_ids, vec![speaker1, speaker2]);
    }

    #[test]
    fn test_speaker_embedding_table_dimension_mismatch() {
        let mut table = SpeakerEmbeddingTable::new(3);
        let speaker = SpeakerId::new(1);
        let wrong_embedding = SpeakerEmbedding::new(speaker, vec![1.0, 2.0]); // Wrong dimension

        assert!(table.add_speaker(speaker, &wrong_embedding).is_err());
    }

    #[test]
    fn test_speaker_embedding_table_duplicate_speaker() {
        let mut table = SpeakerEmbeddingTable::new(3);
        let speaker = SpeakerId::new(1);
        let embedding1 = SpeakerEmbedding::new(speaker, vec![1.0, 2.0, 3.0]);
        let embedding2 = SpeakerEmbedding::new(speaker, vec![4.0, 5.0, 6.0]);

        assert!(table.add_speaker(speaker, &embedding1).is_ok());
        assert!(table.add_speaker(speaker, &embedding2).is_err());
    }

    #[test]
    fn test_voice_style_control() {
        let speaker_id = SpeakerId::new(1);
        let style = VoiceStyleControl::new(speaker_id)
            .with_age_factor(0.5)
            .with_gender_factor(-0.3)
            .with_speed_factor(1.2)
            .with_pitch_shift(2.0)
            .with_energy_factor(1.5)
            .with_breathiness(0.2)
            .with_roughness(0.1);

        assert_eq!(style.base_speaker, speaker_id);
        assert_eq!(style.age_factor, 0.5);
        assert_eq!(style.gender_factor, -0.3);
        assert_eq!(style.speed_factor, 1.2);
        assert_eq!(style.pitch_shift, 2.0);
        assert_eq!(style.energy_factor, 1.5);
        assert_eq!(style.breathiness, 0.2);
        assert_eq!(style.roughness, 0.1);
    }

    #[test]
    fn test_voice_style_control_clamping() {
        let speaker_id = SpeakerId::new(1);
        let style = VoiceStyleControl::new(speaker_id)
            .with_age_factor(2.0) // Should be clamped to 1.0
            .with_gender_factor(-2.0) // Should be clamped to -1.0
            .with_speed_factor(-0.5) // Should be clamped to 0.1
            .with_pitch_shift(20.0) // Should be clamped to 12.0
            .with_breathiness(2.0) // Should be clamped to 1.0
            .with_roughness(-1.0); // Should be clamped to 0.0

        assert_eq!(style.age_factor, 1.0);
        assert_eq!(style.gender_factor, -1.0);
        assert_eq!(style.speed_factor, 0.1);
        assert_eq!(style.pitch_shift, 12.0);
        assert_eq!(style.breathiness, 1.0);
        assert_eq!(style.roughness, 0.0);
    }

    #[test]
    fn test_voice_style_control_apply_to_embedding() {
        let speaker_id = SpeakerId::new(1);
        let base_embedding = SpeakerEmbedding::new(speaker_id, vec![1.0, 2.0, 3.0]);

        let style = VoiceStyleControl::new(speaker_id)
            .with_age_factor(0.5)
            .with_gender_factor(0.2);

        let modified = style.apply_to_embedding(&base_embedding).unwrap();

        // Check that embedding was modified (should be different from original)
        assert_ne!(modified.vector(), base_embedding.vector());

        // Check that dimension is preserved
        assert_eq!(modified.dimension, base_embedding.dimension);
    }

    #[test]
    fn test_multi_speaker_config() {
        let config = MultiSpeakerConfig::new(100, 128)
            .with_embedding_table("embeddings.bin".to_string())
            .with_default_speaker(SpeakerId::new(1));

        assert_eq!(config.max_speakers, 100);
        assert_eq!(config.embedding_dimension, 128);
        assert_eq!(config.embedding_table, Some("embeddings.bin".to_string()));
        assert_eq!(config.default_speaker, Some(SpeakerId::new(1)));
        assert!(config.use_speaker_embeddings);
        assert!(config.support_interpolation);
    }

    #[test]
    fn test_multi_speaker_config_without_embeddings() {
        let config = MultiSpeakerConfig::new(50, 64).without_speaker_embeddings();

        assert!(!config.use_speaker_embeddings);
    }
}
