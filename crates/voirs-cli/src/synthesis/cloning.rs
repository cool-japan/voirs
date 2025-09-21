use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use voirs_sdk::types::SynthesisConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloningMethod {
    FineTuning,
    SpeakerEmbedding,
    ZeroShot,
    FewShot,
    Adaptive,
    Neural,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceProfile {
    pub id: String,
    pub name: String,
    pub embedding: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    pub duration_samples: u64,
    pub quality_score: f32,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloningConfig {
    pub method: CloningMethod,
    pub target_voice_profile: VoiceProfile,
    pub similarity_threshold: f32,
    pub adaptation_rate: f32,
    pub quality_threshold: f32,
    pub max_training_iterations: u32,
    pub use_speaker_verification: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationConfig {
    pub learning_rate: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub batch_size: u32,
    pub gradient_clipping: f32,
    pub convergence_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerEmbeddingConfig {
    pub embedding_dimension: u32,
    pub network_depth: u32,
    pub attention_heads: u32,
    pub dropout_rate: f32,
    pub normalization: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCloningConfig {
    pub base_config: SynthesisConfig,
    pub cloning_config: CloningConfig,
    pub adaptation_config: Option<AdaptationConfig>,
    pub embedding_config: Option<SpeakerEmbeddingConfig>,
    pub reference_audio_paths: Vec<String>,
    pub output_quality_target: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloningProgress {
    pub current_iteration: u32,
    pub total_iterations: u32,
    pub current_loss: f32,
    pub best_loss: f32,
    pub similarity_score: f32,
    pub quality_score: f32,
    pub eta_seconds: u32,
}

pub struct VoiceCloner {
    voice_profiles: HashMap<String, VoiceProfile>,
    active_cloning_sessions: HashMap<String, CloningProgress>,
    embedding_cache: HashMap<String, Vec<f32>>,
    quality_assessor: QualityAssessor,
}

struct QualityAssessor {
    similarity_threshold: f32,
    quality_threshold: f32,
}

impl VoiceCloner {
    pub fn new() -> Self {
        Self {
            voice_profiles: HashMap::new(),
            active_cloning_sessions: HashMap::new(),
            embedding_cache: HashMap::new(),
            quality_assessor: QualityAssessor {
                similarity_threshold: 0.8,
                quality_threshold: 0.7,
            },
        }
    }

    pub fn add_voice_profile(&mut self, profile: VoiceProfile) -> Result<(), String> {
        if profile.embedding.is_empty() {
            return Err("Voice profile embedding cannot be empty".to_string());
        }

        if profile.quality_score < self.quality_assessor.quality_threshold {
            return Err("Voice profile quality score below threshold".to_string());
        }

        self.voice_profiles.insert(profile.id.clone(), profile);
        Ok(())
    }

    pub fn create_voice_profile_from_audio(
        &mut self,
        id: String,
        name: String,
        audio_data: &[f32],
        sample_rate: u32,
        channels: u16,
    ) -> Result<VoiceProfile, String> {
        if audio_data.is_empty() {
            return Err("Audio data cannot be empty".to_string());
        }

        let embedding = self.extract_speaker_embedding(audio_data, sample_rate, channels)?;
        let quality_score = self.assess_audio_quality(audio_data, sample_rate);

        let profile = VoiceProfile {
            id: id.clone(),
            name,
            embedding,
            sample_rate,
            channels,
            duration_samples: audio_data.len() as u64,
            quality_score,
            metadata: HashMap::new(),
        };

        self.embedding_cache
            .insert(id.clone(), profile.embedding.clone());
        Ok(profile)
    }

    fn extract_speaker_embedding(
        &self,
        audio_data: &[f32],
        sample_rate: u32,
        channels: u16,
    ) -> Result<Vec<f32>, String> {
        // Simplified speaker embedding extraction
        // In a real implementation, this would use a pre-trained speaker encoder

        let frame_size = (sample_rate as usize / 100) * channels as usize; // 10ms frames
        let mut embeddings = Vec::new();

        for chunk in audio_data.chunks(frame_size) {
            let mean = chunk.iter().sum::<f32>() / chunk.len() as f32;
            let variance =
                chunk.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / chunk.len() as f32;
            let energy = chunk.iter().map(|x| x.powi(2)).sum::<f32>();

            embeddings.push(mean);
            embeddings.push(variance.sqrt());
            embeddings.push(energy.ln().max(-10.0));
        }

        if embeddings.len() < 128 {
            embeddings.resize(128, 0.0);
        } else {
            embeddings.truncate(128);
        }

        Ok(embeddings)
    }

    fn assess_audio_quality(&self, audio_data: &[f32], _sample_rate: u32) -> f32 {
        let rms =
            (audio_data.iter().map(|x| x.powi(2)).sum::<f32>() / audio_data.len() as f32).sqrt();
        let peak = audio_data.iter().map(|x| x.abs()).fold(0.0, f32::max);
        let dynamic_range = if peak > 0.0 {
            20.0 * (peak / rms).log10()
        } else {
            0.0
        };

        (dynamic_range / 60.0).clamp(0.0, 1.0)
    }

    pub fn calculate_voice_similarity(
        &self,
        profile1: &VoiceProfile,
        profile2: &VoiceProfile,
    ) -> f32 {
        if profile1.embedding.len() != profile2.embedding.len() {
            return 0.0;
        }

        let dot_product: f32 = profile1
            .embedding
            .iter()
            .zip(profile2.embedding.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm1: f32 = profile1
            .embedding
            .iter()
            .map(|x| x.powi(2))
            .sum::<f32>()
            .sqrt();
        let norm2: f32 = profile2
            .embedding
            .iter()
            .map(|x| x.powi(2))
            .sum::<f32>()
            .sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return 0.0;
        }

        (dot_product / (norm1 * norm2)).clamp(-1.0, 1.0)
    }

    pub fn start_cloning_session(
        &mut self,
        session_id: String,
        target_profile: &VoiceProfile,
        config: &CloningConfig,
    ) -> Result<(), String> {
        let total_iterations = config.max_training_iterations;

        let progress = CloningProgress {
            current_iteration: 0,
            total_iterations,
            current_loss: 1.0,
            best_loss: 1.0,
            similarity_score: 0.0,
            quality_score: 0.0,
            eta_seconds: total_iterations * 10, // Estimate 10 seconds per iteration
        };

        self.active_cloning_sessions.insert(session_id, progress);
        Ok(())
    }

    pub fn update_cloning_progress(
        &mut self,
        session_id: &str,
        iteration: u32,
        loss: f32,
        similarity_score: f32,
        quality_score: f32,
    ) -> Result<(), String> {
        if let Some(progress) = self.active_cloning_sessions.get_mut(session_id) {
            progress.current_iteration = iteration;
            progress.current_loss = loss;
            progress.similarity_score = similarity_score;
            progress.quality_score = quality_score;

            if loss < progress.best_loss {
                progress.best_loss = loss;
            }

            let remaining_iterations = progress.total_iterations.saturating_sub(iteration);
            progress.eta_seconds = remaining_iterations * 10;

            Ok(())
        } else {
            Err("Cloning session not found".to_string())
        }
    }

    pub fn create_cloning_synthesis_config(
        &self,
        base_config: SynthesisConfig,
        cloning_config: CloningConfig,
        reference_audio_paths: Vec<String>,
    ) -> VoiceCloningConfig {
        VoiceCloningConfig {
            base_config,
            cloning_config,
            adaptation_config: Some(AdaptationConfig::default()),
            embedding_config: Some(SpeakerEmbeddingConfig::default()),
            reference_audio_paths,
            output_quality_target: 0.8,
        }
    }

    pub fn get_cloning_progress(&self, session_id: &str) -> Option<&CloningProgress> {
        self.active_cloning_sessions.get(session_id)
    }

    pub fn is_cloning_complete(&self, session_id: &str) -> bool {
        if let Some(progress) = self.active_cloning_sessions.get(session_id) {
            progress.current_iteration >= progress.total_iterations
                || progress.similarity_score >= self.quality_assessor.similarity_threshold
        } else {
            false
        }
    }

    pub fn get_voice_profile(&self, profile_id: &str) -> Option<&VoiceProfile> {
        self.voice_profiles.get(profile_id)
    }

    pub fn list_voice_profiles(&self) -> Vec<&VoiceProfile> {
        self.voice_profiles.values().collect()
    }

    pub fn remove_voice_profile(&mut self, profile_id: &str) -> Option<VoiceProfile> {
        self.embedding_cache.remove(profile_id);
        self.voice_profiles.remove(profile_id)
    }

    pub fn clear_completed_sessions(&mut self) {
        let completed_sessions: Vec<String> = self
            .active_cloning_sessions
            .iter()
            .filter(|(id, _)| self.is_cloning_complete(id))
            .map(|(id, _)| id.clone())
            .collect();

        for session_id in completed_sessions {
            self.active_cloning_sessions.remove(&session_id);
        }
    }
}

impl Default for VoiceCloner {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CloningConfig {
    fn default() -> Self {
        Self {
            method: CloningMethod::SpeakerEmbedding,
            target_voice_profile: VoiceProfile {
                id: "default".to_string(),
                name: "Default Voice".to_string(),
                embedding: vec![0.0; 128],
                sample_rate: 22050,
                channels: 1,
                duration_samples: 0,
                quality_score: 0.5,
                metadata: HashMap::new(),
            },
            similarity_threshold: 0.8,
            adaptation_rate: 0.01,
            quality_threshold: 0.7,
            max_training_iterations: 100,
            use_speaker_verification: true,
        }
    }
}

impl Default for AdaptationConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            momentum: 0.9,
            weight_decay: 0.0001,
            batch_size: 32,
            gradient_clipping: 1.0,
            convergence_threshold: 1e-6,
        }
    }
}

impl Default for SpeakerEmbeddingConfig {
    fn default() -> Self {
        Self {
            embedding_dimension: 128,
            network_depth: 4,
            attention_heads: 8,
            dropout_rate: 0.1,
            normalization: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_cloner_creation() {
        let cloner = VoiceCloner::new();
        assert!(cloner.voice_profiles.is_empty());
        assert!(cloner.active_cloning_sessions.is_empty());
    }

    #[test]
    fn test_voice_profile_creation() {
        let mut cloner = VoiceCloner::new();
        let audio_data = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let profile = cloner.create_voice_profile_from_audio(
            "test_id".to_string(),
            "Test Voice".to_string(),
            &audio_data,
            22050,
            1,
        );

        assert!(profile.is_ok());
        let profile = profile.unwrap();
        assert_eq!(profile.id, "test_id");
        assert_eq!(profile.name, "Test Voice");
        assert_eq!(profile.embedding.len(), 128);
    }

    #[test]
    fn test_voice_similarity_calculation() {
        let cloner = VoiceCloner::new();

        let profile1 = VoiceProfile {
            id: "voice1".to_string(),
            name: "Voice 1".to_string(),
            embedding: vec![1.0, 0.0, 0.0, 1.0],
            sample_rate: 22050,
            channels: 1,
            duration_samples: 1000,
            quality_score: 0.8,
            metadata: HashMap::new(),
        };

        let profile2 = VoiceProfile {
            id: "voice2".to_string(),
            name: "Voice 2".to_string(),
            embedding: vec![1.0, 0.0, 0.0, 1.0],
            sample_rate: 22050,
            channels: 1,
            duration_samples: 1000,
            quality_score: 0.8,
            metadata: HashMap::new(),
        };

        let similarity = cloner.calculate_voice_similarity(&profile1, &profile2);
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cloning_session_management() {
        let mut cloner = VoiceCloner::new();
        let profile = VoiceProfile {
            id: "target".to_string(),
            name: "Target Voice".to_string(),
            embedding: vec![0.0; 128],
            sample_rate: 22050,
            channels: 1,
            duration_samples: 1000,
            quality_score: 0.8,
            metadata: HashMap::new(),
        };

        let config = CloningConfig::default();
        let session_id = "session1".to_string();

        assert!(cloner
            .start_cloning_session(session_id.clone(), &profile, &config)
            .is_ok());
        assert!(cloner.get_cloning_progress(&session_id).is_some());
        assert!(!cloner.is_cloning_complete(&session_id));
    }

    #[test]
    fn test_cloning_progress_update() {
        let mut cloner = VoiceCloner::new();
        let profile = VoiceProfile {
            id: "target".to_string(),
            name: "Target Voice".to_string(),
            embedding: vec![0.0; 128],
            sample_rate: 22050,
            channels: 1,
            duration_samples: 1000,
            quality_score: 0.8,
            metadata: HashMap::new(),
        };

        let config = CloningConfig::default();
        let session_id = "session1".to_string();

        cloner
            .start_cloning_session(session_id.clone(), &profile, &config)
            .unwrap();

        assert!(cloner
            .update_cloning_progress(&session_id, 10, 0.5, 0.7, 0.8)
            .is_ok());

        let progress = cloner.get_cloning_progress(&session_id).unwrap();
        assert_eq!(progress.current_iteration, 10);
        assert_eq!(progress.current_loss, 0.5);
        assert_eq!(progress.similarity_score, 0.7);
        assert_eq!(progress.quality_score, 0.8);
    }

    #[test]
    fn test_config_serialization() {
        let config = CloningConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: CloningConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            deserialized.similarity_threshold,
            config.similarity_threshold
        );
        assert_eq!(
            deserialized.max_training_iterations,
            config.max_training_iterations
        );
    }
}
