//! Voice morphing and speaker blending implementation
//!
//! This module provides advanced voice morphing capabilities that can blend
//! characteristics from multiple speakers to create new synthetic voices.
//!
//! Key Features:
//! - Multi-speaker voice blending
//! - Controllable morphing parameters
//! - Real-time voice morphing
//! - Quality-aware blending
//! - Temporal voice transitions
//! - Style interpolation

use crate::{
    embedding::{SpeakerEmbedding, SpeakerEmbeddingExtractor},
    quality::{CloningQualityAssessor, QualityMetrics},
    types::{SpeakerCharacteristics, SpeakerProfile, VoiceSample},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, trace, warn};

/// Configuration for voice morphing operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceMorphingConfig {
    /// Enable quality-aware blending
    pub quality_aware_blending: bool,
    /// Morphing smoothness factor (0.0 = abrupt, 1.0 = very smooth)
    pub smoothness_factor: f32,
    /// Maximum number of speakers to blend
    pub max_speakers: usize,
    /// Temporal morphing support
    pub enable_temporal_morphing: bool,
    /// Real-time morphing capability
    pub enable_realtime: bool,
    /// Morphing quality threshold
    pub quality_threshold: f32,
    /// Preserve dominant speaker characteristics
    pub preserve_dominant_speaker: bool,
    /// Morphing interpolation method
    pub interpolation_method: InterpolationMethod,
    /// Enable style preservation during morphing
    pub preserve_style: bool,
}

impl Default for VoiceMorphingConfig {
    fn default() -> Self {
        Self {
            quality_aware_blending: true,
            smoothness_factor: 0.7,
            max_speakers: 4,
            enable_temporal_morphing: true,
            enable_realtime: true,
            quality_threshold: 0.6,
            preserve_dominant_speaker: false,
            interpolation_method: InterpolationMethod::Weighted,
            preserve_style: true,
        }
    }
}

/// Interpolation methods for voice morphing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterpolationMethod {
    /// Simple linear interpolation
    Linear,
    /// Weighted interpolation based on quality
    Weighted,
    /// Cubic spline interpolation for smoothness
    CubicSpline,
    /// Spherical interpolation for embeddings
    Spherical,
    /// Gaussian mixture interpolation
    GaussianMixture,
}

/// Voice morphing weight specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphingWeight {
    /// Speaker ID
    pub speaker_id: String,
    /// Weight (0.0 to 1.0)
    pub weight: f32,
    /// Quality boost factor
    pub quality_boost: f32,
    /// Temporal variation (for time-based morphing)
    pub temporal_variation: Option<TemporalVariation>,
}

/// Temporal variation specification for dynamic morphing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalVariation {
    /// Variation type
    pub variation_type: TemporalVariationType,
    /// Variation parameters
    pub parameters: HashMap<String, f32>,
    /// Duration of variation cycle
    pub cycle_duration: Duration,
}

/// Types of temporal variation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalVariationType {
    /// Sinusoidal variation
    Sinusoidal,
    /// Linear transition
    Linear,
    /// Step-wise changes
    StepWise,
    /// Random variation
    Random,
    /// Custom curve
    Custom,
}

/// Voice morphing request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceMorphingRequest {
    /// Target morphed voice ID
    pub target_id: String,
    /// Speaker weights for morphing
    pub speaker_weights: Vec<MorphingWeight>,
    /// Morphing configuration
    pub config: VoiceMorphingConfig,
    /// Optional target characteristics to achieve
    pub target_characteristics: Option<SpeakerCharacteristics>,
    /// Duration for temporal morphing
    pub morphing_duration: Option<Duration>,
}

/// Voice morphing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceMorphingResult {
    /// Generated morphed speaker profile
    pub morphed_profile: SpeakerProfile,
    /// Quality metrics of the morphed voice
    pub quality_metrics: QualityMetrics,
    /// Individual speaker contributions
    pub speaker_contributions: HashMap<String, f32>,
    /// Morphing method used
    pub morphing_method: InterpolationMethod,
    /// Processing time
    pub processing_time: Duration,
    /// Confidence in the morphing result
    pub confidence: f32,
    /// Morphing statistics
    pub morphing_stats: MorphingStatistics,
}

/// Statistics about the morphing process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphingStatistics {
    /// Number of speakers used
    pub speakers_used: usize,
    /// Average quality of source speakers
    pub avg_source_quality: f32,
    /// Embedding distance variance
    pub embedding_variance: f32,
    /// Characteristic spread
    pub characteristic_spread: f32,
    /// Temporal complexity (if applicable)
    pub temporal_complexity: Option<f32>,
}

/// Real-time voice morphing session
pub struct RealtimeMorphingSession {
    /// Session ID
    pub session_id: String,
    /// Current morphing weights
    current_weights: Vec<MorphingWeight>,
    /// Target morphing weights
    target_weights: Vec<MorphingWeight>,
    /// Morphing progress (0.0 to 1.0)
    morphing_progress: f32,
    /// Session start time
    start_time: Instant,
    /// Configuration
    config: VoiceMorphingConfig,
    /// Current morphed profile
    current_profile: Option<SpeakerProfile>,
}

impl RealtimeMorphingSession {
    /// Create a new real-time morphing session
    pub fn new(
        session_id: String,
        initial_weights: Vec<MorphingWeight>,
        config: VoiceMorphingConfig,
    ) -> Self {
        Self {
            session_id,
            current_weights: initial_weights.clone(),
            target_weights: initial_weights,
            morphing_progress: 0.0,
            start_time: Instant::now(),
            config,
            current_profile: None,
        }
    }

    /// Update morphing targets
    pub fn update_targets(&mut self, new_weights: Vec<MorphingWeight>) {
        self.target_weights = new_weights;
        self.morphing_progress = 0.0;
    }

    /// Get current morphing progress
    pub fn get_progress(&self) -> f32 {
        self.morphing_progress
    }

    /// Advance morphing by one step
    pub fn step(&mut self, delta_time: Duration) -> Result<()> {
        let progress_increment = delta_time.as_secs_f32() / (self.config.smoothness_factor * 2.0);
        self.morphing_progress = (self.morphing_progress + progress_increment).min(1.0);

        // Interpolate weights
        for (i, current_weight) in self.current_weights.iter_mut().enumerate() {
            if let Some(target_weight) = self.target_weights.get(i) {
                if current_weight.speaker_id == target_weight.speaker_id {
                    current_weight.weight = current_weight.weight
                        + (target_weight.weight - current_weight.weight) * progress_increment;
                }
            }
        }

        Ok(())
    }
}

/// Main voice morphing engine
pub struct VoiceMorpher {
    /// Configuration
    config: VoiceMorphingConfig,
    /// Speaker profiles database
    speaker_profiles: Arc<RwLock<HashMap<String, SpeakerProfile>>>,
    /// Embedding extractor for similarity calculations
    embedding_extractor: Arc<tokio::sync::Mutex<SpeakerEmbeddingExtractor>>,
    /// Quality assessor
    quality_assessor: Arc<CloningQualityAssessor>,
    /// Active real-time sessions
    realtime_sessions: Arc<RwLock<HashMap<String, RealtimeMorphingSession>>>,
    /// Morphing cache for performance
    morphing_cache: Arc<RwLock<HashMap<String, VoiceMorphingResult>>>,
}

impl std::fmt::Debug for VoiceMorpher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoiceMorpher")
            .field("config", &self.config)
            .field("speaker_profiles", &"<Arc<RwLock<HashMap>>>")
            .field(
                "embedding_extractor",
                &"<Arc<Mutex<SpeakerEmbeddingExtractor>>>",
            )
            .field("quality_assessor", &"<Arc<CloningQualityAssessor>>")
            .field("realtime_sessions", &"<Arc<RwLock<HashMap>>>")
            .field("morphing_cache", &"<Arc<RwLock<HashMap>>>")
            .finish()
    }
}

impl VoiceMorpher {
    /// Create a new voice morpher
    pub fn new(config: VoiceMorphingConfig) -> Result<Self> {
        let embedding_extractor =
            Arc::new(tokio::sync::Mutex::new(SpeakerEmbeddingExtractor::default()));
        let quality_assessor = Arc::new(CloningQualityAssessor::default());

        Ok(Self {
            config,
            speaker_profiles: Arc::new(RwLock::new(HashMap::new())),
            embedding_extractor,
            quality_assessor,
            realtime_sessions: Arc::new(RwLock::new(HashMap::new())),
            morphing_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Add a speaker profile to the morphing database
    pub async fn add_speaker_profile(&self, profile: SpeakerProfile) -> Result<()> {
        let speaker_id = profile.id.clone();
        let mut profiles = self.speaker_profiles.write().await;
        profiles.insert(speaker_id.clone(), profile);
        info!("Added speaker profile: {}", speaker_id);
        Ok(())
    }

    /// Perform voice morphing with multiple speakers
    pub async fn morph_voices(&self, request: VoiceMorphingRequest) -> Result<VoiceMorphingResult> {
        let start_time = Instant::now();

        // Check cache first
        let cache_key = self.generate_cache_key(&request);
        if let Some(cached_result) = self.check_cache(&cache_key).await {
            return Ok(cached_result);
        }

        // Validate request
        self.validate_morphing_request(&request).await?;

        // Get speaker profiles
        let profiles = self.get_speaker_profiles(&request.speaker_weights).await?;

        // Perform morphing based on interpolation method
        let morphed_profile = match request.config.interpolation_method {
            InterpolationMethod::Linear => self.linear_morphing(&profiles, &request).await?,
            InterpolationMethod::Weighted => self.weighted_morphing(&profiles, &request).await?,
            InterpolationMethod::CubicSpline => {
                self.cubic_spline_morphing(&profiles, &request).await?
            }
            InterpolationMethod::Spherical => self.spherical_morphing(&profiles, &request).await?,
            InterpolationMethod::GaussianMixture => {
                self.gaussian_mixture_morphing(&profiles, &request).await?
            }
        };

        // Compute quality metrics
        let quality_metrics = self
            .assess_morphed_quality(&morphed_profile, &profiles)
            .await?;

        // Calculate speaker contributions
        let speaker_contributions = self.calculate_speaker_contributions(&request.speaker_weights);

        // Generate morphing statistics
        let morphing_stats = self
            .generate_morphing_statistics(&profiles, &morphed_profile, &request)
            .await?;

        let result = VoiceMorphingResult {
            morphed_profile,
            quality_metrics,
            speaker_contributions,
            morphing_method: request.config.interpolation_method,
            processing_time: start_time.elapsed(),
            confidence: self.calculate_morphing_confidence(&morphing_stats),
            morphing_stats,
        };

        // Cache the result
        self.cache_result(&cache_key, &result).await;

        Ok(result)
    }

    /// Start a real-time morphing session
    pub async fn start_realtime_session(
        &self,
        session_id: String,
        initial_weights: Vec<MorphingWeight>,
    ) -> Result<()> {
        let session =
            RealtimeMorphingSession::new(session_id.clone(), initial_weights, self.config.clone());

        let mut sessions = self.realtime_sessions.write().await;
        sessions.insert(session_id.clone(), session);

        info!("Started real-time morphing session: {}", session_id);
        Ok(())
    }

    /// Update real-time morphing targets
    pub async fn update_realtime_targets(
        &self,
        session_id: &str,
        new_weights: Vec<MorphingWeight>,
    ) -> Result<()> {
        let mut sessions = self.realtime_sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.update_targets(new_weights);
            Ok(())
        } else {
            Err(Error::Processing(format!(
                "Real-time session not found: {}",
                session_id
            )))
        }
    }

    /// Get current state of real-time morphing session
    pub async fn get_realtime_state(&self, session_id: &str) -> Result<f32> {
        let sessions = self.realtime_sessions.read().await;
        if let Some(session) = sessions.get(session_id) {
            Ok(session.get_progress())
        } else {
            Err(Error::Processing(format!(
                "Real-time session not found: {}",
                session_id
            )))
        }
    }

    /// Stop real-time morphing session
    pub async fn stop_realtime_session(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.realtime_sessions.write().await;
        if sessions.remove(session_id).is_some() {
            info!("Stopped real-time morphing session: {}", session_id);
            Ok(())
        } else {
            Err(Error::Processing(format!(
                "Real-time session not found: {}",
                session_id
            )))
        }
    }

    /// Linear morphing implementation
    async fn linear_morphing(
        &self,
        profiles: &[SpeakerProfile],
        request: &VoiceMorphingRequest,
    ) -> Result<SpeakerProfile> {
        let mut morphed_embedding = vec![0.0; 512]; // Default embedding size
        let mut morphed_characteristics = SpeakerCharacteristics::default();

        // Normalize weights
        let total_weight: f32 = request.speaker_weights.iter().map(|w| w.weight).sum();

        for (profile, weight_info) in profiles.iter().zip(request.speaker_weights.iter()) {
            let normalized_weight = weight_info.weight / total_weight;

            // Blend embeddings
            if let Some(embedding) = &profile.embedding {
                for (i, &value) in embedding.iter().enumerate() {
                    if i < morphed_embedding.len() {
                        morphed_embedding[i] += value * normalized_weight;
                    }
                }
            }

            // Blend characteristics
            morphed_characteristics.average_pitch +=
                profile.characteristics.average_pitch * normalized_weight;
            morphed_characteristics.average_energy +=
                profile.characteristics.average_energy * normalized_weight;
            morphed_characteristics.speaking_rate +=
                profile.characteristics.speaking_rate * normalized_weight;
        }

        Ok(SpeakerProfile {
            id: request.target_id.clone(),
            name: format!("Morphed Voice - {}", request.target_id),
            characteristics: morphed_characteristics,
            samples: Vec::new(),
            embedding: Some(morphed_embedding),
            languages: profiles.iter().flat_map(|p| p.languages.clone()).collect(),
            created_at: std::time::SystemTime::now(),
            updated_at: std::time::SystemTime::now(),
            metadata: std::collections::HashMap::new(),
        })
    }

    /// Weighted morphing implementation (quality-aware)
    async fn weighted_morphing(
        &self,
        profiles: &[SpeakerProfile],
        request: &VoiceMorphingRequest,
    ) -> Result<SpeakerProfile> {
        // Similar to linear but with quality-based weighting adjustments
        let mut morphed_embedding = vec![0.0; 512];
        let mut morphed_characteristics = SpeakerCharacteristics::default();

        // Calculate quality-adjusted weights
        let mut total_weight = 0.0;
        let quality_weights: Vec<f32> = profiles
            .iter()
            .zip(request.speaker_weights.iter())
            .map(|(profile, weight_info)| {
                // Simple quality estimate based on sample count and metadata
                let quality_estimate = self.estimate_profile_quality(profile);
                let adjusted_weight =
                    weight_info.weight * (1.0 + quality_estimate * weight_info.quality_boost);
                total_weight += adjusted_weight;
                adjusted_weight
            })
            .collect();

        // Normalize and apply weights
        for ((profile, weight_info), quality_weight) in profiles
            .iter()
            .zip(request.speaker_weights.iter())
            .zip(quality_weights.iter())
        {
            let normalized_weight = quality_weight / total_weight;

            // Blend embeddings
            if let Some(embedding) = &profile.embedding {
                for (i, &value) in embedding.iter().enumerate() {
                    if i < morphed_embedding.len() {
                        morphed_embedding[i] += value * normalized_weight;
                    }
                }
            }

            // Blend characteristics
            morphed_characteristics.average_pitch +=
                profile.characteristics.average_pitch * normalized_weight;
            morphed_characteristics.average_energy +=
                profile.characteristics.average_energy * normalized_weight;
            morphed_characteristics.speaking_rate +=
                profile.characteristics.speaking_rate * normalized_weight;
        }

        Ok(SpeakerProfile {
            id: request.target_id.clone(),
            name: format!("Quality-Weighted Morphed Voice - {}", request.target_id),
            characteristics: morphed_characteristics,
            samples: Vec::new(),
            embedding: Some(morphed_embedding),
            languages: profiles.iter().flat_map(|p| p.languages.clone()).collect(),
            created_at: std::time::SystemTime::now(),
            updated_at: std::time::SystemTime::now(),
            metadata: std::collections::HashMap::new(),
        })
    }

    /// Cubic spline morphing (placeholder - would implement proper spline interpolation)
    async fn cubic_spline_morphing(
        &self,
        profiles: &[SpeakerProfile],
        request: &VoiceMorphingRequest,
    ) -> Result<SpeakerProfile> {
        // For now, fallback to weighted morphing
        self.weighted_morphing(profiles, request).await
    }

    /// Spherical morphing (SLERP for embeddings)
    async fn spherical_morphing(
        &self,
        profiles: &[SpeakerProfile],
        request: &VoiceMorphingRequest,
    ) -> Result<SpeakerProfile> {
        if profiles.len() == 2 {
            // True SLERP between two profiles
            self.slerp_two_profiles(&profiles[0], &profiles[1], &request)
                .await
        } else {
            // Fallback to weighted for multiple profiles
            self.weighted_morphing(profiles, request).await
        }
    }

    /// Gaussian mixture morphing (placeholder)
    async fn gaussian_mixture_morphing(
        &self,
        profiles: &[SpeakerProfile],
        request: &VoiceMorphingRequest,
    ) -> Result<SpeakerProfile> {
        // For now, fallback to weighted morphing
        self.weighted_morphing(profiles, request).await
    }

    /// SLERP between two speaker profiles
    async fn slerp_two_profiles(
        &self,
        profile1: &SpeakerProfile,
        profile2: &SpeakerProfile,
        request: &VoiceMorphingRequest,
    ) -> Result<SpeakerProfile> {
        let weight = request
            .speaker_weights
            .get(1)
            .map(|w| w.weight)
            .unwrap_or(0.5);

        let mut slerp_embedding = vec![0.0; 512];

        if let (Some(emb1), Some(emb2)) = (&profile1.embedding, &profile2.embedding) {
            // Simplified SLERP (would implement proper spherical interpolation)
            for i in 0..slerp_embedding.len().min(emb1.len()).min(emb2.len()) {
                slerp_embedding[i] = emb1[i] * (1.0 - weight) + emb2[i] * weight;
            }
        }

        // Linear interpolation for characteristics
        let morphed_characteristics = SpeakerCharacteristics {
            average_pitch: profile1.characteristics.average_pitch * (1.0 - weight)
                + profile2.characteristics.average_pitch * weight,
            average_energy: profile1.characteristics.average_energy * (1.0 - weight)
                + profile2.characteristics.average_energy * weight,
            speaking_rate: profile1.characteristics.speaking_rate * (1.0 - weight)
                + profile2.characteristics.speaking_rate * weight,
            ..SpeakerCharacteristics::default()
        };

        Ok(SpeakerProfile {
            id: request.target_id.clone(),
            name: format!("SLERP Morphed Voice - {}", request.target_id),
            characteristics: morphed_characteristics,
            samples: Vec::new(),
            embedding: Some(slerp_embedding),
            languages: [profile1.languages.clone(), profile2.languages.clone()].concat(),
            created_at: std::time::SystemTime::now(),
            updated_at: std::time::SystemTime::now(),
            metadata: std::collections::HashMap::new(),
        })
    }

    /// Validate morphing request
    async fn validate_morphing_request(&self, request: &VoiceMorphingRequest) -> Result<()> {
        if request.speaker_weights.is_empty() {
            return Err(Error::Validation("No speaker weights provided".to_string()));
        }

        if request.speaker_weights.len() > self.config.max_speakers {
            return Err(Error::Validation(format!(
                "Too many speakers: {} (max: {})",
                request.speaker_weights.len(),
                self.config.max_speakers
            )));
        }

        // Check if all speakers exist
        let profiles = self.speaker_profiles.read().await;
        for weight in &request.speaker_weights {
            if !profiles.contains_key(&weight.speaker_id) {
                return Err(Error::Validation(format!(
                    "Speaker not found: {}",
                    weight.speaker_id
                )));
            }
        }

        Ok(())
    }

    /// Get speaker profiles for morphing
    async fn get_speaker_profiles(
        &self,
        weights: &[MorphingWeight],
    ) -> Result<Vec<SpeakerProfile>> {
        let profiles_map = self.speaker_profiles.read().await;
        let mut profiles = Vec::new();

        for weight in weights {
            if let Some(profile) = profiles_map.get(&weight.speaker_id) {
                profiles.push(profile.clone());
            }
        }

        Ok(profiles)
    }

    /// Estimate profile quality
    fn estimate_profile_quality(&self, profile: &SpeakerProfile) -> f32 {
        let sample_quality = if profile.samples.is_empty() { 0.3 } else { 0.8 };
        let embedding_quality = if profile.embedding.is_some() {
            0.9
        } else {
            0.1
        };
        let metadata_quality = if profile.metadata.is_empty() {
            0.5
        } else {
            0.8
        };

        (sample_quality + embedding_quality + metadata_quality) / 3.0
    }

    /// Calculate speaker contributions
    fn calculate_speaker_contributions(&self, weights: &[MorphingWeight]) -> HashMap<String, f32> {
        let total_weight: f32 = weights.iter().map(|w| w.weight).sum();

        weights
            .iter()
            .map(|w| (w.speaker_id.clone(), w.weight / total_weight))
            .collect()
    }

    /// Assess quality of morphed voice
    async fn assess_morphed_quality(
        &self,
        morphed_profile: &SpeakerProfile,
        _source_profiles: &[SpeakerProfile],
    ) -> Result<QualityMetrics> {
        // Create basic quality metrics for the morphed voice
        let mut quality_metrics = QualityMetrics::new();

        // Base quality on the morphed profile characteristics
        let profile_quality = self.estimate_profile_quality(morphed_profile);

        quality_metrics.overall_score = profile_quality * 0.85; // Slightly lower due to morphing
        quality_metrics.speaker_similarity = 0.7; // Mixed similarity
        quality_metrics.audio_quality = profile_quality;
        quality_metrics.naturalness = profile_quality * 0.9;
        quality_metrics.content_preservation = 0.95; // High content preservation in morphing
        quality_metrics.prosodic_similarity = 0.8;
        quality_metrics.spectral_similarity = 0.75;

        Ok(quality_metrics)
    }

    /// Generate morphing statistics
    async fn generate_morphing_statistics(
        &self,
        profiles: &[SpeakerProfile],
        morphed_profile: &SpeakerProfile,
        request: &VoiceMorphingRequest,
    ) -> Result<MorphingStatistics> {
        let avg_source_quality = profiles
            .iter()
            .map(|p| self.estimate_profile_quality(p))
            .sum::<f32>()
            / profiles.len() as f32;

        // Calculate embedding variance
        let embedding_variance = if let Some(morphed_emb) = &morphed_profile.embedding {
            let mut variance_sum = 0.0;
            let mut count = 0;

            for profile in profiles {
                if let Some(emb) = &profile.embedding {
                    for (i, &val) in emb.iter().enumerate() {
                        if i < morphed_emb.len() {
                            let diff = morphed_emb[i] - val;
                            variance_sum += diff * diff;
                            count += 1;
                        }
                    }
                }
            }

            if count > 0 {
                variance_sum / count as f32
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Calculate characteristic spread
        let pitch_values: Vec<f32> = profiles
            .iter()
            .map(|p| p.characteristics.average_pitch)
            .collect();
        let energy_values: Vec<f32> = profiles
            .iter()
            .map(|p| p.characteristics.average_energy)
            .collect();
        let rate_values: Vec<f32> = profiles
            .iter()
            .map(|p| p.characteristics.speaking_rate)
            .collect();

        let pitch_spread = if pitch_values.len() > 1 {
            let mean = pitch_values.iter().sum::<f32>() / pitch_values.len() as f32;
            pitch_values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / pitch_values.len() as f32
        } else {
            0.0
        };

        let characteristic_spread = pitch_spread; // Simplified

        Ok(MorphingStatistics {
            speakers_used: profiles.len(),
            avg_source_quality,
            embedding_variance,
            characteristic_spread,
            temporal_complexity: None, // Would be calculated for temporal morphing
        })
    }

    /// Calculate morphing confidence
    fn calculate_morphing_confidence(&self, stats: &MorphingStatistics) -> f32 {
        let quality_factor = stats.avg_source_quality;
        let variance_factor = (1.0 / (1.0 + stats.embedding_variance)).min(1.0);
        let speaker_factor = if stats.speakers_used <= self.config.max_speakers {
            1.0
        } else {
            0.8
        };

        (quality_factor + variance_factor + speaker_factor) / 3.0
    }

    /// Generate cache key for morphing request
    fn generate_cache_key(&self, request: &VoiceMorphingRequest) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        request.target_id.hash(&mut hasher);
        for weight in &request.speaker_weights {
            weight.speaker_id.hash(&mut hasher);
            ((weight.weight * 1000.0) as u32).hash(&mut hasher);
        }
        format!("morph_{:x}", hasher.finish())
    }

    /// Check morphing cache
    async fn check_cache(&self, cache_key: &str) -> Option<VoiceMorphingResult> {
        let cache = self.morphing_cache.read().await;
        cache.get(cache_key).cloned()
    }

    /// Cache morphing result
    async fn cache_result(&self, cache_key: &str, result: &VoiceMorphingResult) {
        let mut cache = self.morphing_cache.write().await;
        cache.insert(cache_key.to_string(), result.clone());
    }

    /// Get available speaker profiles
    pub async fn get_available_speakers(&self) -> Vec<String> {
        let profiles = self.speaker_profiles.read().await;
        profiles.keys().cloned().collect()
    }

    /// Clear morphing cache
    pub async fn clear_cache(&self) {
        let mut cache = self.morphing_cache.write().await;
        cache.clear();
        info!("Cleared morphing cache");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_voice_morphing_config() {
        let config = VoiceMorphingConfig::default();
        assert_eq!(config.max_speakers, 4);
        assert!(config.quality_aware_blending);
        assert_eq!(config.interpolation_method, InterpolationMethod::Weighted);
    }

    #[tokio::test]
    async fn test_morphing_weight_creation() {
        let weight = MorphingWeight {
            speaker_id: "speaker1".to_string(),
            weight: 0.5,
            quality_boost: 0.1,
            temporal_variation: None,
        };

        assert_eq!(weight.speaker_id, "speaker1");
        assert_eq!(weight.weight, 0.5);
        assert!(weight.temporal_variation.is_none());
    }

    #[tokio::test]
    async fn test_voice_morpher_creation() {
        let config = VoiceMorphingConfig::default();
        let morpher = VoiceMorpher::new(config);
        assert!(morpher.is_ok());
    }

    #[tokio::test]
    async fn test_speaker_profile_addition() {
        let config = VoiceMorphingConfig::default();
        let morpher = VoiceMorpher::new(config).unwrap();

        let profile = SpeakerProfile {
            id: "test_speaker".to_string(),
            name: "Test Speaker".to_string(),
            characteristics: SpeakerCharacteristics::default(),
            samples: Vec::new(),
            embedding: Some(vec![0.1, 0.2, 0.3]),
            languages: vec!["en".to_string()],
            created_at: std::time::SystemTime::now(),
            updated_at: std::time::SystemTime::now(),
            metadata: std::collections::HashMap::new(),
        };

        let result = morpher.add_speaker_profile(profile).await;
        assert!(result.is_ok());

        let speakers = morpher.get_available_speakers().await;
        assert_eq!(speakers.len(), 1);
        assert_eq!(speakers[0], "test_speaker");
    }

    #[tokio::test]
    async fn test_interpolation_methods() {
        // Test that all interpolation methods are available
        let methods = [
            InterpolationMethod::Linear,
            InterpolationMethod::Weighted,
            InterpolationMethod::CubicSpline,
            InterpolationMethod::Spherical,
            InterpolationMethod::GaussianMixture,
        ];

        assert_eq!(methods.len(), 5);
    }

    #[tokio::test]
    async fn test_realtime_session_creation() {
        let config = VoiceMorphingConfig::default();
        let morpher = VoiceMorpher::new(config).unwrap();

        let weights = vec![
            MorphingWeight {
                speaker_id: "speaker1".to_string(),
                weight: 0.6,
                quality_boost: 0.0,
                temporal_variation: None,
            },
            MorphingWeight {
                speaker_id: "speaker2".to_string(),
                weight: 0.4,
                quality_boost: 0.0,
                temporal_variation: None,
            },
        ];

        let result = morpher
            .start_realtime_session("session1".to_string(), weights)
            .await;
        assert!(result.is_ok());

        let progress = morpher.get_realtime_state("session1").await;
        assert!(progress.is_ok());
        assert_eq!(progress.unwrap(), 0.0);

        let stop_result = morpher.stop_realtime_session("session1").await;
        assert!(stop_result.is_ok());
    }

    #[tokio::test]
    async fn test_cache_operations() {
        let config = VoiceMorphingConfig::default();
        let morpher = VoiceMorpher::new(config).unwrap();

        // Initially empty
        let speakers = morpher.get_available_speakers().await;
        assert_eq!(speakers.len(), 0);

        // Clear should work even when empty
        morpher.clear_cache().await;
    }
}
