//! Multi-speaker support for acoustic models.

use super::{SpeakerEmbedding, SpeakerId};
use crate::{AcousticError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Multi-speaker model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSpeakerConfig {
    /// Number of speakers supported
    pub speaker_count: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Whether to use speaker embedding lookup table
    pub use_lookup_table: bool,
    /// Default speaker ID
    pub default_speaker: Option<SpeakerId>,
}

impl Default for MultiSpeakerConfig {
    fn default() -> Self {
        Self {
            speaker_count: 100,
            embedding_dim: 256,
            use_lookup_table: true,
            default_speaker: Some(SpeakerId::new(0)),
        }
    }
}

/// Multi-speaker acoustic model wrapper
pub struct MultiSpeakerModel {
    /// Model configuration
    config: MultiSpeakerConfig,
    /// Speaker embedding lookup table
    speaker_embeddings: HashMap<SpeakerId, SpeakerEmbedding>,
    /// Current active speaker
    current_speaker: Option<SpeakerId>,
}

impl MultiSpeakerModel {
    /// Create new multi-speaker model
    pub fn new(config: MultiSpeakerConfig) -> Self {
        Self {
            config,
            speaker_embeddings: HashMap::new(),
            current_speaker: None,
        }
    }

    /// Add speaker embedding
    pub fn add_speaker(
        &mut self,
        speaker_id: SpeakerId,
        embedding: SpeakerEmbedding,
    ) -> Result<()> {
        if embedding.dimension != self.config.embedding_dim {
            return Err(AcousticError::ConfigError(format!(
                "Speaker embedding dimension {} does not match model dimension {}",
                embedding.dimension, self.config.embedding_dim
            )));
        }

        self.speaker_embeddings.insert(speaker_id, embedding);
        Ok(())
    }

    /// Remove speaker
    pub fn remove_speaker(&mut self, speaker_id: SpeakerId) -> Result<()> {
        if !self.speaker_embeddings.contains_key(&speaker_id) {
            return Err(AcousticError::InputError(format!(
                "Speaker {} not found",
                speaker_id.id()
            )));
        }

        self.speaker_embeddings.remove(&speaker_id);

        // Clear current speaker if it was removed
        if self.current_speaker == Some(speaker_id) {
            self.current_speaker = None;
        }

        Ok(())
    }

    /// Set current speaker
    pub fn set_current_speaker(&mut self, speaker_id: SpeakerId) -> Result<()> {
        if !self.speaker_embeddings.contains_key(&speaker_id) {
            return Err(AcousticError::InputError(format!(
                "Speaker {} not found",
                speaker_id.id()
            )));
        }

        self.current_speaker = Some(speaker_id);
        Ok(())
    }

    /// Get current speaker
    pub fn get_current_speaker(&self) -> Option<SpeakerId> {
        self.current_speaker.or(self.config.default_speaker)
    }

    /// Get speaker embedding
    pub fn get_speaker_embedding(&self, speaker_id: SpeakerId) -> Result<&SpeakerEmbedding> {
        self.speaker_embeddings.get(&speaker_id).ok_or_else(|| {
            AcousticError::InputError(format!("Speaker {} not found", speaker_id.id()))
        })
    }

    /// Get current speaker embedding
    pub fn get_current_speaker_embedding(&self) -> Result<&SpeakerEmbedding> {
        let speaker_id = self
            .get_current_speaker()
            .ok_or_else(|| AcousticError::ConfigError("No current speaker set".to_string()))?;

        self.get_speaker_embedding(speaker_id)
    }

    /// List all speakers
    pub fn list_speakers(&self) -> Vec<SpeakerId> {
        self.speaker_embeddings.keys().cloned().collect()
    }

    /// Get speaker count
    pub fn speaker_count(&self) -> usize {
        self.speaker_embeddings.len()
    }

    /// Create voice morphing between two speakers
    pub fn morph_speakers(
        &self,
        speaker1: SpeakerId,
        speaker2: SpeakerId,
        alpha: f32,
    ) -> Result<SpeakerEmbedding> {
        let embedding1 = self.get_speaker_embedding(speaker1)?;
        let embedding2 = self.get_speaker_embedding(speaker2)?;

        embedding1.interpolate(embedding2, alpha)
    }

    /// Find most similar speaker to given embedding
    pub fn find_similar_speaker(&self, target_embedding: &SpeakerEmbedding) -> Result<SpeakerId> {
        if self.speaker_embeddings.is_empty() {
            return Err(AcousticError::ConfigError(
                "No speakers registered".to_string(),
            ));
        }

        let mut best_speaker = None;
        let mut best_similarity = f32::NEG_INFINITY;

        for (speaker_id, embedding) in &self.speaker_embeddings {
            let similarity = self.compute_similarity(target_embedding, embedding);
            if similarity > best_similarity {
                best_similarity = similarity;
                best_speaker = Some(*speaker_id);
            }
        }

        best_speaker.ok_or_else(|| {
            AcousticError::InferenceError("Failed to find similar speaker".to_string())
        })
    }

    /// Compute cosine similarity between two embeddings
    fn compute_similarity(
        &self,
        embedding1: &SpeakerEmbedding,
        embedding2: &SpeakerEmbedding,
    ) -> f32 {
        if embedding1.dimension != embedding2.dimension {
            return 0.0;
        }

        let mut dot_product = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;

        for (a, b) in embedding1.vector().iter().zip(embedding2.vector().iter()) {
            dot_product += a * b;
            norm1 += a * a;
            norm2 += b * b;
        }

        if norm1 == 0.0 || norm2 == 0.0 {
            return 0.0;
        }

        dot_product / (norm1.sqrt() * norm2.sqrt())
    }

    /// Generate random speaker embedding (for testing)
    pub fn generate_random_embedding(&self, speaker_id: SpeakerId) -> SpeakerEmbedding {
        let mut embedding = Vec::with_capacity(self.config.embedding_dim);
        for _ in 0..self.config.embedding_dim {
            embedding.push(fastrand::f32() * 2.0 - 1.0); // Random between -1 and 1
        }

        SpeakerEmbedding::new(speaker_id, embedding)
    }

    /// Initialize with default speakers
    pub fn initialize_default_speakers(&mut self) -> Result<()> {
        // Create a few default speakers with different characteristics
        let default_speakers = [
            (0, "Default Speaker"),
            (1, "Male Speaker"),
            (2, "Female Speaker"),
            (3, "Child Speaker"),
            (4, "Elderly Speaker"),
        ];

        for (id, _name) in default_speakers {
            let speaker_id = SpeakerId::new(id);
            let embedding = self.generate_random_embedding(speaker_id);
            self.add_speaker(speaker_id, embedding)?;
        }

        // Set default speaker
        self.current_speaker = Some(SpeakerId::new(0));

        Ok(())
    }
}

impl Default for MultiSpeakerModel {
    fn default() -> Self {
        Self::new(MultiSpeakerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_speaker_config() {
        let config = MultiSpeakerConfig::default();
        assert_eq!(config.speaker_count, 100);
        assert_eq!(config.embedding_dim, 256);
        assert!(config.use_lookup_table);
        assert_eq!(config.default_speaker, Some(SpeakerId::new(0)));
    }

    #[test]
    fn test_multi_speaker_model_creation() {
        let model = MultiSpeakerModel::default();
        assert_eq!(model.speaker_count(), 0);
        assert_eq!(model.get_current_speaker(), Some(SpeakerId::new(0)));
    }

    #[test]
    fn test_add_speaker() {
        let mut model = MultiSpeakerModel::default();
        let speaker_id = SpeakerId::new(1);
        let embedding = SpeakerEmbedding::new(speaker_id, vec![0.0; 256]);

        assert!(model.add_speaker(speaker_id, embedding).is_ok());
        assert_eq!(model.speaker_count(), 1);
        assert!(model.list_speakers().contains(&speaker_id));
    }

    #[test]
    fn test_add_speaker_wrong_dimension() {
        let mut model = MultiSpeakerModel::default();
        let speaker_id = SpeakerId::new(1);
        let embedding = SpeakerEmbedding::new(speaker_id, vec![0.0; 128]); // Wrong dimension

        assert!(model.add_speaker(speaker_id, embedding).is_err());
    }

    #[test]
    fn test_set_current_speaker() {
        let mut model = MultiSpeakerModel::default();
        let speaker_id = SpeakerId::new(1);
        let embedding = SpeakerEmbedding::new(speaker_id, vec![0.0; 256]);

        model.add_speaker(speaker_id, embedding).unwrap();
        assert!(model.set_current_speaker(speaker_id).is_ok());
        assert_eq!(model.get_current_speaker(), Some(speaker_id));
    }

    #[test]
    fn test_set_nonexistent_speaker() {
        let mut model = MultiSpeakerModel::default();
        let speaker_id = SpeakerId::new(999);

        assert!(model.set_current_speaker(speaker_id).is_err());
    }

    #[test]
    fn test_remove_speaker() {
        let mut model = MultiSpeakerModel::default();
        let speaker_id = SpeakerId::new(1);
        let embedding = SpeakerEmbedding::new(speaker_id, vec![0.0; 256]);

        model.add_speaker(speaker_id, embedding).unwrap();
        model.set_current_speaker(speaker_id).unwrap();

        assert!(model.remove_speaker(speaker_id).is_ok());
        assert_eq!(model.speaker_count(), 0);
        assert_eq!(model.get_current_speaker(), Some(SpeakerId::new(0))); // Should fall back to default
    }

    #[test]
    fn test_morph_speakers() {
        let mut model = MultiSpeakerModel::default();

        let speaker1 = SpeakerId::new(1);
        let speaker2 = SpeakerId::new(2);

        let embedding1 = SpeakerEmbedding::new(speaker1, vec![1.0; 256]);
        let embedding2 = SpeakerEmbedding::new(speaker2, vec![2.0; 256]);

        model.add_speaker(speaker1, embedding1).unwrap();
        model.add_speaker(speaker2, embedding2).unwrap();

        let morphed = model.morph_speakers(speaker1, speaker2, 0.5).unwrap();
        assert_eq!(morphed.vector()[0], 1.5); // Average of 1.0 and 2.0
    }

    #[test]
    fn test_compute_similarity() {
        let model = MultiSpeakerModel::default();

        // Use smaller vectors for this test (not connected to model dimension)
        let embedding1 = SpeakerEmbedding::new(SpeakerId::new(1), vec![1.0, 0.0, 0.0]);
        let embedding2 = SpeakerEmbedding::new(SpeakerId::new(2), vec![1.0, 0.0, 0.0]);
        let embedding3 = SpeakerEmbedding::new(SpeakerId::new(3), vec![0.0, 1.0, 0.0]);

        let similarity12 = model.compute_similarity(&embedding1, &embedding2);
        let similarity13 = model.compute_similarity(&embedding1, &embedding3);

        assert!((similarity12 - 1.0).abs() < 1e-6); // Should be 1.0 (identical)
        assert!((similarity13 - 0.0).abs() < 1e-6); // Should be 0.0 (orthogonal)
    }

    #[test]
    fn test_initialize_default_speakers() {
        let mut model = MultiSpeakerModel::default();
        assert!(model.initialize_default_speakers().is_ok());
        assert_eq!(model.speaker_count(), 5);
        assert_eq!(model.get_current_speaker(), Some(SpeakerId::new(0)));
    }

    #[test]
    fn test_find_similar_speaker() {
        let mut model = MultiSpeakerModel::default();

        let speaker1 = SpeakerId::new(1);
        let speaker2 = SpeakerId::new(2);

        // Use correct dimension (256)
        let mut embedding1_vec = vec![0.0; 256];
        embedding1_vec[0] = 1.0; // Set first dimension
        let embedding1 = SpeakerEmbedding::new(speaker1, embedding1_vec);

        let mut embedding2_vec = vec![0.0; 256];
        embedding2_vec[1] = 1.0; // Set second dimension
        let embedding2 = SpeakerEmbedding::new(speaker2, embedding2_vec);

        model.add_speaker(speaker1, embedding1).unwrap();
        model.add_speaker(speaker2, embedding2).unwrap();

        let mut target_vec = vec![0.0; 256];
        target_vec[0] = 0.9; // Closer to first speaker
        target_vec[1] = 0.1;
        let target = SpeakerEmbedding::new(SpeakerId::new(999), target_vec);
        let similar = model.find_similar_speaker(&target).unwrap();

        assert_eq!(similar, speaker1); // Should be closer to speaker1
    }
}
