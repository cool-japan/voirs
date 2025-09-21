//! Voice cloning and speaker adaptation functionality.

use super::{SpeakerEmbedding, SpeakerId};
use crate::{AcousticError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Voice cloning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCloningConfig {
    /// Target speaker embedding dimension
    pub embedding_dimension: usize,
    /// Number of reference audio samples required
    pub min_reference_samples: usize,
    /// Maximum number of reference samples to process
    pub max_reference_samples: usize,
    /// Minimum duration per reference sample (in seconds)
    pub min_sample_duration: f32,
    /// Maximum duration per reference sample (in seconds)
    pub max_sample_duration: f32,
    /// Quality threshold for reference samples (0.0-1.0)
    pub quality_threshold: f32,
    /// Whether to enable cross-language adaptation
    pub enable_cross_language: bool,
    /// Adaptation learning rate
    pub adaptation_learning_rate: f32,
    /// Number of adaptation iterations
    pub adaptation_iterations: usize,
}

impl Default for VoiceCloningConfig {
    fn default() -> Self {
        Self {
            embedding_dimension: 256,
            min_reference_samples: 3,
            max_reference_samples: 20,
            min_sample_duration: 2.0,
            max_sample_duration: 10.0,
            quality_threshold: 0.7,
            enable_cross_language: true,
            adaptation_learning_rate: 0.001,
            adaptation_iterations: 100,
        }
    }
}

/// Audio reference sample for voice cloning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioReference {
    /// Unique identifier for this reference
    pub id: String,
    /// Audio file path
    pub file_path: String,
    /// Duration in seconds
    pub duration: f32,
    /// Sample rate
    pub sample_rate: u32,
    /// Audio quality score (0.0-1.0)
    pub quality_score: f32,
    /// Transcription (if available)
    pub transcription: Option<String>,
    /// Language code
    pub language: String,
    /// Extracted audio features
    pub features: Option<AudioFeatures>,
}

/// Audio features extracted from reference samples
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFeatures {
    /// Mel-frequency cepstral coefficients (MFCCs)
    pub mfccs: Vec<Vec<f32>>,
    /// Fundamental frequency (F0) contour
    pub f0_contour: Vec<f32>,
    /// Spectral centroid
    pub spectral_centroid: Vec<f32>,
    /// Spectral rolloff
    pub spectral_rolloff: Vec<f32>,
    /// Zero crossing rate
    pub zero_crossing_rate: Vec<f32>,
    /// Energy/RMS
    pub energy: Vec<f32>,
    /// Voice activity detection (VAD) segments
    pub vad_segments: Vec<(f32, f32)>, // (start_time, end_time)
}

/// Voice cloning quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloningQualityMetrics {
    /// Overall cloning quality score (0.0-1.0)
    pub overall_quality: f32,
    /// Speaker similarity score (0.0-1.0)
    pub speaker_similarity: f32,
    /// Audio clarity score (0.0-1.0)
    pub audio_clarity: f32,
    /// Prosody preservation score (0.0-1.0)
    pub prosody_preservation: f32,
    /// Naturalness score (0.0-1.0)
    pub naturalness: f32,
    /// Cross-language adaptation score (0.0-1.0, if applicable)
    pub cross_language_score: Option<f32>,
}

/// Speaker verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerVerificationResult {
    /// Whether verification passed
    pub verified: bool,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Cosine similarity score
    pub cosine_similarity: f32,
    /// Euclidean distance
    pub euclidean_distance: f32,
    /// Verification threshold used
    pub threshold: f32,
}

/// Few-shot speaker adaptation algorithm
pub struct FewShotSpeakerAdaptation {
    config: VoiceCloningConfig,
    base_model_embeddings: HashMap<SpeakerId, SpeakerEmbedding>,
}

impl FewShotSpeakerAdaptation {
    /// Create new few-shot adaptation instance
    pub fn new(config: VoiceCloningConfig) -> Self {
        Self {
            config,
            base_model_embeddings: HashMap::new(),
        }
    }

    /// Add base model speaker embedding
    pub fn add_base_speaker(&mut self, speaker_id: SpeakerId, embedding: SpeakerEmbedding) {
        self.base_model_embeddings.insert(speaker_id, embedding);
    }

    /// Extract speaker embedding from reference audio samples
    pub fn extract_speaker_embedding(
        &self,
        references: &[AudioReference],
    ) -> Result<SpeakerEmbedding> {
        if references.len() < self.config.min_reference_samples {
            return Err(AcousticError::InputError(format!(
                "Need at least {} reference samples, got {}",
                self.config.min_reference_samples,
                references.len()
            )));
        }

        // Validate reference samples
        for reference in references {
            if reference.quality_score < self.config.quality_threshold {
                return Err(AcousticError::InputError(format!(
                    "Reference sample {} has quality score {:.2}, minimum required: {:.2}",
                    reference.id, reference.quality_score, self.config.quality_threshold
                )));
            }

            if reference.duration < self.config.min_sample_duration
                || reference.duration > self.config.max_sample_duration
            {
                return Err(AcousticError::InputError(format!(
                    "Reference sample {} duration {:.2}s is outside valid range [{:.2}s, {:.2}s]",
                    reference.id,
                    reference.duration,
                    self.config.min_sample_duration,
                    self.config.max_sample_duration
                )));
            }
        }

        // Extract features from reference samples
        let mut aggregated_features = self.extract_aggregated_features(references)?;

        // Perform few-shot adaptation
        let adapted_embedding = self.adapt_embedding(&mut aggregated_features)?;

        Ok(adapted_embedding)
    }

    /// Extract aggregated features from multiple reference samples
    fn extract_aggregated_features(&self, references: &[AudioReference]) -> Result<Vec<f32>> {
        let mut aggregated_features = vec![0.0; self.config.embedding_dimension];

        for reference in references {
            let features = self.extract_single_sample_features(reference)?;

            // Weighted aggregation based on quality score
            let weight = reference.quality_score;
            for (i, feature) in features.iter().enumerate() {
                if i < aggregated_features.len() {
                    aggregated_features[i] += feature * weight;
                }
            }
        }

        // Normalize by sum of weights
        let total_weight: f32 = references.iter().map(|r| r.quality_score).sum();
        if total_weight > 0.0 {
            for feature in aggregated_features.iter_mut() {
                *feature /= total_weight;
            }
        }

        Ok(aggregated_features)
    }

    /// Extract features from a single reference sample
    fn extract_single_sample_features(&self, reference: &AudioReference) -> Result<Vec<f32>> {
        // This is a simplified implementation
        // In a real implementation, this would involve:
        // 1. Loading the audio file
        // 2. Extracting acoustic features (MFCCs, F0, etc.)
        // 3. Converting to speaker embedding space
        // 4. Applying voice activity detection
        // 5. Temporal aggregation

        let mut features = Vec::with_capacity(self.config.embedding_dimension);

        // Generate deterministic features based on reference properties
        let mut seed = reference.file_path.len() as u64;
        seed = seed.wrapping_mul(reference.duration as u64);
        seed = seed.wrapping_mul((reference.quality_score * 1000.0) as u64);

        // Simple LCG for deterministic random features
        let mut rng_state = seed;
        for _ in 0..self.config.embedding_dimension {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let normalized = (rng_state as f32) / (u32::MAX as f32);
            features.push(normalized * 2.0 - 1.0); // Range [-1, 1]
        }

        // Apply quality-based scaling
        for feature in features.iter_mut() {
            *feature *= reference.quality_score;
        }

        Ok(features)
    }

    /// Adapt embedding using few-shot learning
    fn adapt_embedding(&self, features: &mut [f32]) -> Result<SpeakerEmbedding> {
        // Find the most similar base speaker
        let (best_base_id, best_similarity) = self.find_most_similar_base_speaker(features)?;

        // Perform adaptation iterations
        for _ in 0..self.config.adaptation_iterations {
            self.adaptation_step(features, best_base_id, best_similarity)?;
        }

        // Create final embedding
        let speaker_id = SpeakerId::new(0); // Will be assigned later
        Ok(SpeakerEmbedding::new(speaker_id, features.to_owned()))
    }

    /// Find the most similar base speaker for adaptation
    fn find_most_similar_base_speaker(&self, features: &[f32]) -> Result<(SpeakerId, f32)> {
        let mut best_similarity = -1.0;
        let mut best_speaker_id = None;

        for (speaker_id, embedding) in &self.base_model_embeddings {
            let similarity = self.calculate_cosine_similarity(features, embedding.vector());
            if similarity > best_similarity {
                best_similarity = similarity;
                best_speaker_id = Some(*speaker_id);
            }
        }

        if let Some(speaker_id) = best_speaker_id {
            Ok((speaker_id, best_similarity))
        } else {
            Err(AcousticError::InputError(
                "No base speakers available for adaptation".to_string(),
            ))
        }
    }

    /// Perform one adaptation step
    fn adaptation_step(
        &self,
        features: &mut [f32],
        base_speaker_id: SpeakerId,
        similarity: f32,
    ) -> Result<()> {
        if let Some(base_embedding) = self.base_model_embeddings.get(&base_speaker_id) {
            let base_vector = base_embedding.vector();

            // Adaptive learning rate based on similarity
            let adaptive_lr = self.config.adaptation_learning_rate * (1.0 - similarity);

            // Update features towards base speaker while preserving uniqueness
            for (i, feature) in features.iter_mut().enumerate() {
                if i < base_vector.len() {
                    let delta = base_vector[i] - *feature;
                    *feature += adaptive_lr * delta;
                }
            }
        }

        Ok(())
    }

    /// Calculate cosine similarity between two vectors
    fn calculate_cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }
}

/// Speaker verification system
pub struct SpeakerVerifier {
    verification_threshold: f32,
    #[allow(dead_code)]
    embedding_dimension: usize,
}

impl SpeakerVerifier {
    /// Create new speaker verifier
    pub fn new(embedding_dimension: usize, verification_threshold: f32) -> Self {
        Self {
            verification_threshold,
            embedding_dimension,
        }
    }

    /// Verify speaker identity
    pub fn verify_speaker(
        &self,
        reference_embedding: &SpeakerEmbedding,
        test_embedding: &SpeakerEmbedding,
    ) -> Result<SpeakerVerificationResult> {
        if reference_embedding.dimension != test_embedding.dimension {
            return Err(AcousticError::InputError(
                "Speaker embeddings must have the same dimension".to_string(),
            ));
        }

        let cosine_similarity =
            self.calculate_cosine_similarity(reference_embedding.vector(), test_embedding.vector());

        let euclidean_distance = self
            .calculate_euclidean_distance(reference_embedding.vector(), test_embedding.vector());

        let verified = cosine_similarity >= self.verification_threshold;
        let confidence = if verified {
            cosine_similarity
        } else {
            1.0 - cosine_similarity
        };

        Ok(SpeakerVerificationResult {
            verified,
            confidence,
            cosine_similarity,
            euclidean_distance,
            threshold: self.verification_threshold,
        })
    }

    /// Calculate cosine similarity between two embeddings
    fn calculate_cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    /// Calculate Euclidean distance between two embeddings
    fn calculate_euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// Voice cloning quality assessor
pub struct VoiceCloningQualityAssessor {
    config: VoiceCloningConfig,
}

impl VoiceCloningQualityAssessor {
    /// Create new quality assessor
    pub fn new(config: VoiceCloningConfig) -> Self {
        Self { config }
    }

    /// Assess cloning quality
    pub fn assess_quality(
        &self,
        original_references: &[AudioReference],
        cloned_embedding: &SpeakerEmbedding,
        synthesized_audio_path: Option<&str>,
    ) -> Result<CloningQualityMetrics> {
        let speaker_similarity =
            self.calculate_speaker_similarity(original_references, cloned_embedding)?;
        let audio_clarity =
            self.calculate_audio_clarity(original_references, synthesized_audio_path)?;
        let prosody_preservation = self.calculate_prosody_preservation(original_references)?;
        let naturalness =
            self.calculate_naturalness(original_references, synthesized_audio_path)?;
        let cross_language_score = if self.config.enable_cross_language {
            Some(self.calculate_cross_language_score(original_references)?)
        } else {
            None
        };

        // Calculate overall quality as weighted average
        let overall_quality = (speaker_similarity * 0.3
            + audio_clarity * 0.25
            + prosody_preservation * 0.25
            + naturalness * 0.2)
            * 0.9
            + cross_language_score.unwrap_or(0.8) * 0.1;

        Ok(CloningQualityMetrics {
            overall_quality,
            speaker_similarity,
            audio_clarity,
            prosody_preservation,
            naturalness,
            cross_language_score,
        })
    }

    /// Calculate speaker similarity score
    fn calculate_speaker_similarity(
        &self,
        references: &[AudioReference],
        cloned_embedding: &SpeakerEmbedding,
    ) -> Result<f32> {
        let mut total_similarity = 0.0;
        let mut count = 0;

        for reference in references {
            // Extract reference embedding (simplified)
            let ref_features = self.extract_reference_features(reference)?;
            let ref_embedding = SpeakerEmbedding::new(SpeakerId::new(0), ref_features);

            // Calculate similarity
            let similarity =
                self.calculate_cosine_similarity(ref_embedding.vector(), cloned_embedding.vector());

            total_similarity += similarity;
            count += 1;
        }

        Ok(if count > 0 {
            total_similarity / count as f32
        } else {
            0.0
        })
    }

    /// Calculate audio clarity score
    fn calculate_audio_clarity(
        &self,
        references: &[AudioReference],
        _synthesized_audio_path: Option<&str>,
    ) -> Result<f32> {
        // Calculate average quality score from references
        let avg_quality: f32 =
            references.iter().map(|r| r.quality_score).sum::<f32>() / references.len() as f32;

        // For synthesized audio, in a real implementation we would:
        // 1. Analyze SNR (Signal-to-Noise Ratio)
        // 2. Measure spectral clarity
        // 3. Detect artifacts
        // 4. Check for distortion

        Ok(avg_quality.min(1.0))
    }

    /// Calculate prosody preservation score
    fn calculate_prosody_preservation(&self, references: &[AudioReference]) -> Result<f32> {
        // In a real implementation, this would analyze:
        // 1. F0 contour preservation
        // 2. Rhythm pattern consistency
        // 3. Stress pattern maintenance
        // 4. Intonation similarity

        // For now, use a heuristic based on reference quality and diversity
        let quality_score: f32 =
            references.iter().map(|r| r.quality_score).sum::<f32>() / references.len() as f32;
        let diversity_bonus = if references.len() > 5 { 0.1 } else { 0.0 };

        Ok((quality_score + diversity_bonus).min(1.0))
    }

    /// Calculate naturalness score
    fn calculate_naturalness(
        &self,
        references: &[AudioReference],
        _synthesized_audio_path: Option<&str>,
    ) -> Result<f32> {
        // In a real implementation, this would:
        // 1. Analyze human-likeness of speech patterns
        // 2. Check for robotic artifacts
        // 3. Measure emotional expressiveness
        // 4. Evaluate pronunciation naturalness

        // Heuristic based on reference quality and sample count
        let quality_score: f32 =
            references.iter().map(|r| r.quality_score).sum::<f32>() / references.len() as f32;
        let sample_count_factor = (references.len() as f32 / 10.0).min(1.0);

        Ok(quality_score * 0.7 + sample_count_factor * 0.3)
    }

    /// Calculate cross-language adaptation score
    fn calculate_cross_language_score(&self, references: &[AudioReference]) -> Result<f32> {
        // Check language diversity
        let mut languages = std::collections::HashSet::new();
        for reference in references {
            languages.insert(&reference.language);
        }

        let language_diversity = languages.len() as f32;
        let diversity_score = (language_diversity / 3.0).min(1.0); // Bonus for multilingual

        // Quality-weighted score
        let quality_score: f32 =
            references.iter().map(|r| r.quality_score).sum::<f32>() / references.len() as f32;

        Ok(quality_score * 0.8 + diversity_score * 0.2)
    }

    /// Extract features from reference for similarity calculation
    fn extract_reference_features(&self, reference: &AudioReference) -> Result<Vec<f32>> {
        // Simplified feature extraction
        let mut features = Vec::with_capacity(self.config.embedding_dimension);

        // Generate deterministic features based on reference properties
        let mut seed = reference.file_path.len() as u64;
        seed = seed.wrapping_mul(reference.duration as u64);
        seed = seed.wrapping_mul((reference.quality_score * 1000.0) as u64);

        let mut rng_state = seed;
        for _ in 0..self.config.embedding_dimension {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let normalized = (rng_state as f32) / (u32::MAX as f32);
            features.push(normalized * 2.0 - 1.0);
        }

        Ok(features)
    }

    /// Calculate cosine similarity between two vectors
    fn calculate_cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }
}

/// Cross-language speaker adapter
pub struct CrossLanguageSpeakerAdapter {
    #[allow(dead_code)]
    config: VoiceCloningConfig,
    language_models: HashMap<String, Vec<f32>>, // Language-specific adaptation models
}

impl CrossLanguageSpeakerAdapter {
    /// Create new cross-language adapter
    pub fn new(config: VoiceCloningConfig) -> Self {
        Self {
            config,
            language_models: HashMap::new(),
        }
    }

    /// Add language-specific adaptation model
    pub fn add_language_model(&mut self, language: String, model: Vec<f32>) {
        self.language_models.insert(language, model);
    }

    /// Adapt speaker embedding for cross-language synthesis
    pub fn adapt_for_language(
        &self,
        base_embedding: &SpeakerEmbedding,
        target_language: &str,
    ) -> Result<SpeakerEmbedding> {
        let language_model = self.language_models.get(target_language).ok_or_else(|| {
            AcousticError::InputError(format!(
                "No adaptation model for language: {target_language}"
            ))
        })?;

        let mut adapted_embedding = base_embedding.embedding.clone();

        // Apply language-specific adaptation
        for (i, adaptation_factor) in language_model.iter().enumerate() {
            if i < adapted_embedding.len() {
                adapted_embedding[i] *= 1.0 + adaptation_factor * 0.1;
            }
        }

        Ok(SpeakerEmbedding::new(
            base_embedding.speaker_id,
            adapted_embedding,
        ))
    }

    /// Analyze cross-language compatibility
    pub fn analyze_compatibility(
        &self,
        references: &[AudioReference],
        target_language: &str,
    ) -> Result<f32> {
        // Calculate compatibility score based on:
        // 1. Language diversity in references
        // 2. Phonetic similarity between languages
        // 3. Quality of reference samples

        let mut source_languages = std::collections::HashSet::new();
        let mut total_quality = 0.0;

        for reference in references {
            source_languages.insert(&reference.language);
            total_quality += reference.quality_score;
        }

        let avg_quality = total_quality / references.len() as f32;
        let language_diversity = source_languages.len() as f32;

        // Simple heuristic for compatibility
        let diversity_factor = (language_diversity / 3.0).min(1.0);
        let quality_factor = avg_quality;

        // Check if target language is among source languages
        let same_language_bonus = if source_languages.iter().any(|&lang| lang == target_language) {
            0.3
        } else {
            0.0
        };

        Ok((diversity_factor * 0.4 + quality_factor * 0.4 + same_language_bonus).min(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_cloning_config_default() {
        let config = VoiceCloningConfig::default();
        assert_eq!(config.embedding_dimension, 256);
        assert_eq!(config.min_reference_samples, 3);
        assert_eq!(config.max_reference_samples, 20);
        assert_eq!(config.quality_threshold, 0.7);
        assert!(config.enable_cross_language);
    }

    #[test]
    fn test_audio_reference_creation() {
        let reference = AudioReference {
            id: "ref1".to_string(),
            file_path: "test.wav".to_string(),
            duration: 5.0,
            sample_rate: 22050,
            quality_score: 0.8,
            transcription: Some("Hello world".to_string()),
            language: "en-US".to_string(),
            features: None,
        };

        assert_eq!(reference.id, "ref1");
        assert_eq!(reference.duration, 5.0);
        assert_eq!(reference.quality_score, 0.8);
    }

    #[test]
    fn test_few_shot_adaptation_creation() {
        let config = VoiceCloningConfig::default();
        let adapter = FewShotSpeakerAdaptation::new(config);

        // Test that adapter is created successfully
        assert_eq!(adapter.config.embedding_dimension, 256);
    }

    #[test]
    fn test_speaker_verifier_creation() {
        let verifier = SpeakerVerifier::new(256, 0.8);
        assert_eq!(verifier.embedding_dimension, 256);
        assert_eq!(verifier.verification_threshold, 0.8);
    }

    #[test]
    fn test_speaker_verification() {
        let verifier = SpeakerVerifier::new(3, 0.8);

        let emb1 = SpeakerEmbedding::new(SpeakerId::new(1), vec![1.0, 0.0, 0.0]);
        let emb2 = SpeakerEmbedding::new(SpeakerId::new(2), vec![0.9, 0.1, 0.1]);

        let result = verifier.verify_speaker(&emb1, &emb2).unwrap();
        assert!(result.cosine_similarity > 0.8);
        assert!(result.euclidean_distance < 0.5);
    }

    #[test]
    fn test_cosine_similarity_calculation() {
        let verifier = SpeakerVerifier::new(3, 0.8);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = verifier.calculate_cosine_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        let similarity2 = verifier.calculate_cosine_similarity(&a, &c);
        assert!((similarity2 - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_quality_assessor_creation() {
        let config = VoiceCloningConfig::default();
        let assessor = VoiceCloningQualityAssessor::new(config);
        assert_eq!(assessor.config.embedding_dimension, 256);
    }

    #[test]
    fn test_cross_language_adapter_creation() {
        let config = VoiceCloningConfig::default();
        let mut adapter = CrossLanguageSpeakerAdapter::new(config);

        // Add a language model
        let model = vec![0.1; 256];
        adapter.add_language_model("en-US".to_string(), model);

        assert!(adapter.language_models.contains_key("en-US"));
    }

    #[test]
    fn test_extract_speaker_embedding_insufficient_samples() {
        let config = VoiceCloningConfig::default();
        let adapter = FewShotSpeakerAdaptation::new(config);

        let references = vec![AudioReference {
            id: "ref1".to_string(),
            file_path: "test1.wav".to_string(),
            duration: 5.0,
            sample_rate: 22050,
            quality_score: 0.8,
            transcription: None,
            language: "en-US".to_string(),
            features: None,
        }];

        let result = adapter.extract_speaker_embedding(&references);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_speaker_embedding_low_quality() {
        let config = VoiceCloningConfig::default();
        let adapter = FewShotSpeakerAdaptation::new(config);

        let references = vec![
            AudioReference {
                id: "ref1".to_string(),
                file_path: "test1.wav".to_string(),
                duration: 5.0,
                sample_rate: 22050,
                quality_score: 0.5, // Below threshold
                transcription: None,
                language: "en-US".to_string(),
                features: None,
            },
            AudioReference {
                id: "ref2".to_string(),
                file_path: "test2.wav".to_string(),
                duration: 5.0,
                sample_rate: 22050,
                quality_score: 0.5,
                transcription: None,
                language: "en-US".to_string(),
                features: None,
            },
            AudioReference {
                id: "ref3".to_string(),
                file_path: "test3.wav".to_string(),
                duration: 5.0,
                sample_rate: 22050,
                quality_score: 0.5,
                transcription: None,
                language: "en-US".to_string(),
                features: None,
            },
        ];

        let result = adapter.extract_speaker_embedding(&references);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_speaker_embedding_success() {
        let config = VoiceCloningConfig::default();
        let mut adapter = FewShotSpeakerAdaptation::new(config);

        // Add a base speaker for adaptation
        let base_embedding = SpeakerEmbedding::new(SpeakerId::new(1), vec![0.5; 256]);
        adapter.add_base_speaker(SpeakerId::new(1), base_embedding);

        let references = vec![
            AudioReference {
                id: "ref1".to_string(),
                file_path: "test1.wav".to_string(),
                duration: 5.0,
                sample_rate: 22050,
                quality_score: 0.8,
                transcription: None,
                language: "en-US".to_string(),
                features: None,
            },
            AudioReference {
                id: "ref2".to_string(),
                file_path: "test2.wav".to_string(),
                duration: 5.0,
                sample_rate: 22050,
                quality_score: 0.8,
                transcription: None,
                language: "en-US".to_string(),
                features: None,
            },
            AudioReference {
                id: "ref3".to_string(),
                file_path: "test3.wav".to_string(),
                duration: 5.0,
                sample_rate: 22050,
                quality_score: 0.8,
                transcription: None,
                language: "en-US".to_string(),
                features: None,
            },
        ];

        let result = adapter.extract_speaker_embedding(&references);
        assert!(result.is_ok());

        let embedding = result.unwrap();
        assert_eq!(embedding.dimension, 256);
    }
}
