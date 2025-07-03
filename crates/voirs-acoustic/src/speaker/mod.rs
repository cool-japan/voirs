//! Speaker control and voice characteristics management.

pub mod multi;
pub mod emotion;
pub mod characteristics;

pub use multi::*;
pub use emotion::*;
pub use characteristics::*;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::Result;

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
                "Speaker embeddings must have the same dimension".to_string()
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
    pub fn find_speakers_by_characteristics(&self, filter: &VoiceCharacteristics) -> Vec<SpeakerId> {
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
        alpha: f32
    ) -> Result<SpeakerEmbedding> {
        let embedding1 = self.get_embedding(speaker1)
            .ok_or_else(|| crate::AcousticError::InputError(
                format!("Speaker {} not found", speaker1.id())
            ))?;
        let embedding2 = self.get_embedding(speaker2)
            .ok_or_else(|| crate::AcousticError::InputError(
                format!("Speaker {} not found", speaker2.id())
            ))?;
        
        embedding1.interpolate(embedding2, alpha)
    }
}

impl Default for SpeakerRegistry {
    fn default() -> Self {
        Self::new()
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
}