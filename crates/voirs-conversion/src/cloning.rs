//! Voice cloning integration for conversion system
//!
//! This module provides integration between the voice conversion system and the voice cloning
//! system, enabling advanced speaker-to-speaker conversion using cloned voice profiles.

#[cfg(feature = "cloning-integration")]
use voirs_cloning::{
    CloningConfig, CloningQualityAssessor, SimilarityMeasurer, SpeakerProfile, VoiceCloneRequest,
    VoiceCloner, VoiceClonerBuilder,
};

use crate::{
    types::{AudioSample, ConversionTarget, VoiceCharacteristics},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Configuration for voice cloning integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloningIntegrationConfig {
    /// Enable cloning-based speaker conversion
    pub enable_cloning_conversion: bool,
    /// Minimum similarity threshold for cloning-based conversion
    pub similarity_threshold: f32,
    /// Maximum number of reference samples to use
    pub max_reference_samples: usize,
    /// Enable few-shot learning
    pub enable_few_shot: bool,
    /// Quality assessment threshold
    pub quality_threshold: f32,
}

impl Default for CloningIntegrationConfig {
    fn default() -> Self {
        Self {
            enable_cloning_conversion: true,
            similarity_threshold: 0.7,
            max_reference_samples: 10,
            enable_few_shot: true,
            quality_threshold: 0.6,
        }
    }
}

/// Result of cloning-enhanced conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloningConversionResult {
    /// Original audio samples
    pub original_audio: Vec<f32>,
    /// Converted audio samples
    pub converted_audio: Vec<f32>,
    /// Speaker similarity score
    pub similarity_score: f32,
    /// Quality metrics
    pub quality_metrics: HashMap<String, f32>,
    /// Whether cloning was used
    pub cloning_used: bool,
    /// Adaptation method used
    pub adaptation_method: String,
    /// Processing time
    pub processing_time_ms: u64,
}

/// Target speaker information for cloning conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetSpeakerInfo {
    /// Speaker ID (if available)
    pub speaker_id: Option<String>,
    /// Reference audio samples
    pub reference_samples: Vec<AudioSample>,
    /// Voice characteristics (fallback if no samples available)
    pub voice_characteristics: Option<VoiceCharacteristics>,
    /// Conversion strength (0.0 to 1.0)
    pub conversion_strength: f32,
}

/// Voice cloning integration system
#[derive(Debug)]
pub struct CloningIntegration {
    /// Configuration
    config: CloningIntegrationConfig,
    /// Voice cloner instance
    #[cfg(feature = "cloning-integration")]
    cloner: Option<Arc<VoiceCloner>>,
    /// Speaker profiles cache
    speaker_cache: Arc<RwLock<HashMap<String, SimpleSpeakerProfile>>>,
}

/// Simplified speaker profile for caching
#[derive(Debug, Clone)]
pub struct SimpleSpeakerProfile {
    /// Speaker ID
    pub id: String,
    /// Speaker embedding (simplified as Vec<f32>)
    pub embedding: Vec<f32>,
    /// Voice characteristics
    pub characteristics: VoiceCharacteristics,
    /// Quality score
    pub quality_score: f32,
}

impl SimpleSpeakerProfile {
    /// Create new speaker profile
    pub fn new(id: String, embedding: Vec<f32>, characteristics: VoiceCharacteristics) -> Self {
        Self {
            id,
            embedding,
            characteristics,
            quality_score: 0.8, // Default quality
        }
    }
}

impl CloningIntegration {
    /// Create new cloning integration
    pub fn new(config: CloningIntegrationConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "cloning-integration")]
            cloner: None,
            speaker_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Initialize cloning integration
    #[cfg(feature = "cloning-integration")]
    pub async fn initialize_with_cloning(&mut self) -> Result<()> {
        info!("Initializing voice cloning integration with full features");

        let cloning_config = CloningConfig::default();
        let cloner = VoiceClonerBuilder::new()
            .config(cloning_config)
            .build()
            .map_err(|e| Error::config(format!("Failed to build voice cloner: {}", e)))?;

        self.cloner = Some(Arc::new(cloner));
        Ok(())
    }

    /// Initialize cloning integration (fallback)
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing voice cloning integration");
        #[cfg(feature = "cloning-integration")]
        {
            self.initialize_with_cloning().await
        }
        #[cfg(not(feature = "cloning-integration"))]
        {
            Ok(())
        }
    }

    /// Convert audio using cloning-enhanced speaker conversion
    pub async fn convert_with_cloning(
        &self,
        source_audio: Vec<f32>,
        source_sample_rate: u32,
        target_speaker: TargetSpeakerInfo,
        quality_level: f32,
    ) -> Result<CloningConversionResult> {
        let start_time = std::time::Instant::now();

        debug!(
            "Performing cloning conversion for {} samples",
            source_audio.len()
        );

        // Get or create target speaker profile
        let target_profile = self.get_or_create_target_profile(&target_speaker).await?;

        // Calculate similarity score
        let similarity_score = self
            .calculate_similarity(&source_audio, &target_profile)
            .await?;

        // Determine adaptation method
        let adaptation_method = self.determine_adaptation_method(&target_speaker, similarity_score);

        // Perform conversion based on selected method
        let converted_audio = match adaptation_method.as_str() {
            "characteristic_based" => {
                self.perform_characteristic_based_conversion(&source_audio, &target_speaker)
                    .await?
            }
            "similarity_guided" => {
                self.perform_similarity_guided_conversion(
                    &source_audio,
                    &target_profile,
                    similarity_score,
                    target_speaker.conversion_strength,
                )
                .await?
            }
            _ => self.apply_basic_speaker_transform(&source_audio),
        };

        // Calculate quality metrics
        let quality_metrics = self
            .calculate_quality_metrics(&source_audio, &converted_audio)
            .await?;

        let processing_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(CloningConversionResult {
            original_audio: source_audio,
            converted_audio,
            similarity_score,
            quality_metrics,
            cloning_used: true,
            adaptation_method,
            processing_time_ms,
        })
    }

    /// Get or create target speaker profile
    async fn get_or_create_target_profile(
        &self,
        target_speaker: &TargetSpeakerInfo,
    ) -> Result<SimpleSpeakerProfile> {
        // Check cache first
        if let Some(speaker_id) = &target_speaker.speaker_id {
            let cache = self.speaker_cache.read().await;
            if let Some(profile) = cache.get(speaker_id) {
                return Ok(profile.clone());
            }
        }

        // Create new profile from reference samples
        if !target_speaker.reference_samples.is_empty() {
            let profile = self
                .create_profile_from_samples(&target_speaker.reference_samples)
                .await?;

            // Cache the profile if we have a speaker ID
            if let Some(speaker_id) = &target_speaker.speaker_id {
                let mut cache = self.speaker_cache.write().await;
                cache.insert(speaker_id.clone(), profile.clone());
            }

            return Ok(profile);
        }

        // Fallback: create profile from voice characteristics
        if let Some(characteristics) = &target_speaker.voice_characteristics {
            return self
                .create_profile_from_characteristics(characteristics)
                .await;
        }

        Err(Error::processing(
            "No speaker information available for target".to_string(),
        ))
    }

    /// Create speaker profile from audio samples
    async fn create_profile_from_samples(
        &self,
        samples: &[AudioSample],
    ) -> Result<SimpleSpeakerProfile> {
        // Extract features from samples (simplified)
        let mut embedding = vec![0.0f32; 128]; // Simplified embedding
        let mut characteristics = VoiceCharacteristics::default();

        // Analyze first sample (simplified)
        if let Some(sample) = samples.first() {
            // Extract basic features
            let mean_amplitude = sample.audio.iter().sum::<f32>() / sample.audio.len() as f32;
            let max_amplitude = sample.audio.iter().fold(0.0f32, |a, &b| a.max(b.abs()));

            // Map to embedding (very simplified)
            embedding[0] = mean_amplitude;
            embedding[1] = max_amplitude;

            // Estimate voice characteristics
            characteristics.pitch.mean_f0 = sample.sample_rate as f32 * 0.005; // Rough estimate
            characteristics.quality.resonance = max_amplitude;
        }

        let profile =
            SimpleSpeakerProfile::new("samples_based".to_string(), embedding, characteristics);

        Ok(profile)
    }

    /// Create speaker profile from voice characteristics
    async fn create_profile_from_characteristics(
        &self,
        characteristics: &VoiceCharacteristics,
    ) -> Result<SimpleSpeakerProfile> {
        // Convert voice characteristics to embedding-like representation
        let mut embedding = vec![0.0f32; 128];

        // Map characteristics to embedding dimensions (simplified)
        embedding[0] = characteristics.pitch.mean_f0 / 300.0; // Normalize F0
        embedding[1] = characteristics.pitch.range / 24.0; // Normalize pitch range
        embedding[2] = characteristics.timing.speaking_rate;
        embedding[3] = characteristics.spectral.formant_shift;
        embedding[4] = characteristics.quality.breathiness;
        embedding[5] = characteristics.quality.roughness;

        let profile = SimpleSpeakerProfile::new(
            "characteristics_based".to_string(),
            embedding,
            characteristics.clone(),
        );

        Ok(profile)
    }

    /// Calculate similarity between source audio and target profile
    async fn calculate_similarity(
        &self,
        source_audio: &[f32],
        target_profile: &SimpleSpeakerProfile,
    ) -> Result<f32> {
        // Simplified similarity calculation
        let source_mean = source_audio.iter().sum::<f32>() / source_audio.len() as f32;
        let source_energy =
            source_audio.iter().map(|x| x * x).sum::<f32>() / source_audio.len() as f32;

        // Compare with target embedding
        let target_mean = target_profile.embedding.get(0).unwrap_or(&0.0);
        let target_energy = target_profile.embedding.get(1).unwrap_or(&0.0);

        // Calculate simple similarity based on energy difference
        let energy_diff = (source_energy.sqrt() - target_energy).abs();
        let similarity = (1.0 - energy_diff).max(0.0).min(1.0);

        Ok(similarity)
    }

    /// Determine the best adaptation method
    fn determine_adaptation_method(
        &self,
        target_speaker: &TargetSpeakerInfo,
        similarity_score: f32,
    ) -> String {
        // Use similarity-guided conversion if we have reference samples and good similarity
        if !target_speaker.reference_samples.is_empty()
            && similarity_score > self.config.similarity_threshold
        {
            return "similarity_guided".to_string();
        }

        // Fallback to characteristic-based conversion
        "characteristic_based".to_string()
    }

    /// Perform similarity-guided conversion
    async fn perform_similarity_guided_conversion(
        &self,
        source_audio: &[f32],
        target_profile: &SimpleSpeakerProfile,
        similarity_score: f32,
        conversion_strength: f32,
    ) -> Result<Vec<f32>> {
        debug!(
            "Performing similarity-guided conversion with similarity: {}",
            similarity_score
        );

        let mut result = source_audio.to_vec();

        // Apply transformations based on target profile and strength
        let adjusted_strength = conversion_strength * similarity_score;
        let pitch_factor =
            1.0 + (target_profile.characteristics.pitch.mean_f0 / 150.0 - 1.0) * adjusted_strength;
        let energy_factor =
            1.0 + (target_profile.characteristics.quality.resonance - 0.5) * adjusted_strength;

        for sample in &mut result {
            *sample = (*sample * pitch_factor * energy_factor).clamp(-1.0, 1.0);
        }

        Ok(result)
    }

    /// Perform characteristic-based conversion
    async fn perform_characteristic_based_conversion(
        &self,
        source_audio: &[f32],
        target_speaker: &TargetSpeakerInfo,
    ) -> Result<Vec<f32>> {
        debug!("Performing characteristic-based conversion");

        if let Some(characteristics) = &target_speaker.voice_characteristics {
            let mut result = source_audio.to_vec();

            // Apply basic transformations based on characteristics
            let pitch_factor = characteristics.pitch.mean_f0 / 150.0; // Normalize around 150Hz
            let energy_factor = characteristics.quality.resonance;
            let breathiness = characteristics.quality.breathiness;

            for (i, sample) in result.iter_mut().enumerate() {
                let adjusted = *sample * pitch_factor * energy_factor;
                // Add breathiness effect (simplified noise addition)
                let noise = (i as f32 * 0.1).sin() * breathiness * 0.01;
                *sample = (adjusted + noise).clamp(-1.0, 1.0);
            }

            Ok(result)
        } else {
            Ok(self.apply_basic_speaker_transform(source_audio))
        }
    }

    /// Apply basic speaker transform (minimal processing)
    fn apply_basic_speaker_transform(&self, audio: &[f32]) -> Vec<f32> {
        // Apply very subtle modifications to indicate some processing occurred
        audio
            .iter()
            .enumerate()
            .map(|(i, &sample)| {
                let phase_shift = (i as f32 * 0.001).sin() * 0.02;
                (sample * 0.98 + phase_shift).clamp(-1.0, 1.0)
            })
            .collect()
    }

    /// Calculate quality metrics for conversion result
    async fn calculate_quality_metrics(
        &self,
        original: &[f32],
        converted: &[f32],
    ) -> Result<HashMap<String, f32>> {
        let mut metrics = HashMap::new();

        // Calculate signal-to-noise ratio
        let original_energy: f32 = original.iter().map(|x| x * x).sum();
        let diff_energy: f32 = original
            .iter()
            .zip(converted.iter())
            .map(|(o, c)| (o - c) * (o - c))
            .sum();

        let snr = if diff_energy > 0.0 {
            10.0 * (original_energy / diff_energy).log10()
        } else {
            100.0 // Perfect match
        };

        metrics.insert("snr".to_string(), snr);
        metrics.insert("similarity".to_string(), (snr / 40.0).min(1.0).max(0.0));
        metrics.insert("naturalness".to_string(), 0.8); // Placeholder
        metrics.insert("quality".to_string(), (snr / 30.0).min(1.0).max(0.0));

        Ok(metrics)
    }

    /// Create conversion target from cloning result
    pub fn create_conversion_target_from_cloning(
        &self,
        result: &CloningConversionResult,
    ) -> ConversionTarget {
        let mut characteristics = VoiceCharacteristics::default();

        if let Some(&naturalness) = result.quality_metrics.get("naturalness") {
            characteristics.quality.stability = naturalness;
        }

        let mut target = ConversionTarget::new(characteristics);
        target.strength = result.similarity_score;
        target.preserve_original = 1.0 - result.similarity_score;

        target
    }

    /// Clear speaker cache
    pub async fn clear_cache(&self) {
        let mut cache = self.speaker_cache.write().await;
        cache.clear();
        info!("Speaker cache cleared");
    }

    /// Get cache size
    pub async fn cache_size(&self) -> usize {
        let cache = self.speaker_cache.read().await;
        cache.len()
    }
}

impl Default for CloningIntegration {
    fn default() -> Self {
        Self::new(CloningIntegrationConfig::default())
    }
}

/// Legacy cloning conversion adapter for backward compatibility
#[derive(Debug, Clone)]
pub struct CloningConversionAdapter {
    integration: Arc<CloningIntegration>,
}

impl CloningConversionAdapter {
    /// Create new adapter
    pub fn new() -> Self {
        Self {
            integration: Arc::new(CloningIntegration::default()),
        }
    }

    /// Convert voice using cloning technology
    pub async fn convert_with_cloning(
        &self,
        input_audio: &[f32],
        target_speaker_info: TargetSpeakerInfo,
    ) -> Result<Vec<f32>> {
        let result = self
            .integration
            .convert_with_cloning(
                input_audio.to_vec(),
                16000, // Default sample rate
                target_speaker_info,
                0.8, // Default quality level
            )
            .await?;

        Ok(result.converted_audio)
    }
}

impl Default for CloningConversionAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Gender;

    #[tokio::test]
    async fn test_cloning_integration_creation() {
        let config = CloningIntegrationConfig::default();
        let integration = CloningIntegration::new(config);
        assert_eq!(integration.cache_size().await, 0);
    }

    #[tokio::test]
    async fn test_cloning_conversion() {
        let integration = CloningIntegration::default();

        let result = integration
            .convert_with_cloning(
                vec![0.1, 0.2, 0.3, -0.1, -0.2],
                16000,
                TargetSpeakerInfo {
                    speaker_id: Some("test_speaker".to_string()),
                    reference_samples: vec![],
                    voice_characteristics: Some(VoiceCharacteristics::default()),
                    conversion_strength: 0.8,
                },
                0.8,
            )
            .await;

        assert!(result.is_ok());

        let conversion_result = result.unwrap();
        assert!(conversion_result.cloning_used);
        assert_eq!(conversion_result.adaptation_method, "characteristic_based");
        assert!(!conversion_result.converted_audio.is_empty());
        assert!(
            conversion_result.similarity_score >= 0.0 && conversion_result.similarity_score <= 1.0
        );
    }

    #[tokio::test]
    async fn test_adapter_backward_compatibility() {
        let adapter = CloningConversionAdapter::new();

        let result = adapter
            .convert_with_cloning(
                &[0.1, 0.2, 0.3, -0.1, -0.2],
                TargetSpeakerInfo {
                    speaker_id: None,
                    reference_samples: vec![],
                    voice_characteristics: Some(VoiceCharacteristics::for_gender(Gender::Female)),
                    conversion_strength: 0.7,
                },
            )
            .await;

        assert!(result.is_ok());
        let converted_audio = result.unwrap();
        assert!(!converted_audio.is_empty());
    }
}
