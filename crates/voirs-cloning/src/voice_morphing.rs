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

    /// Cubic-spline morphing.
    ///
    /// Treats each speaker embedding as a control point of a *natural cubic
    /// spline* positioned at uniformly spaced knots in `[0, 1]`, then evaluates
    /// the spline element-wise across the embedding dimensions at the morph
    /// parameter `t`. The parameter is the weight-weighted centroid of the knot
    /// positions, so a single dominant speaker maps to its own knot; in
    /// particular the result reduces to the first/last embedding at
    /// `t = 0`/`t = 1`. With two speakers the natural spline collapses to
    /// (correct) linear interpolation, while three or more speakers yield a
    /// genuinely smooth `C²` blend. The required second derivatives are obtained
    /// by solving the tridiagonal natural-spline system with the Thomas
    /// algorithm (see [`cubic_spline_interpolate`]).
    async fn cubic_spline_morphing(
        &self,
        profiles: &[SpeakerProfile],
        request: &VoiceMorphingRequest,
    ) -> Result<SpeakerProfile> {
        // A spline needs at least two control points; otherwise defer to the
        // quality-weighted blend, which is the correct answer for one speaker.
        if profiles.len() < 2 {
            return self.weighted_morphing(profiles, request).await;
        }

        let knots = uniform_knots(profiles.len());
        let normalized = normalize_weights(&request.speaker_weights);
        // Morph parameter: the weight-weighted centroid of the knot positions.
        let morph_t = knots
            .iter()
            .zip(normalized.iter())
            .map(|(&knot, &weight)| knot * weight)
            .sum::<f32>()
            .clamp(0.0, 1.0);

        let embeddings: Vec<Vec<f32>> = profiles
            .iter()
            .map(|p| p.embedding.clone().unwrap_or_default())
            .collect();
        let morphed_embedding = if embeddings.iter().all(|e| !e.is_empty()) {
            cubic_spline_interpolate_vector(&knots, &embeddings, morph_t)
        } else {
            vec![0.0; 512]
        };

        let pitch: Vec<f32> = profiles
            .iter()
            .map(|p| p.characteristics.average_pitch)
            .collect();
        let energy: Vec<f32> = profiles
            .iter()
            .map(|p| p.characteristics.average_energy)
            .collect();
        let rate: Vec<f32> = profiles
            .iter()
            .map(|p| p.characteristics.speaking_rate)
            .collect();

        let morphed_characteristics = SpeakerCharacteristics {
            average_pitch: cubic_spline_interpolate(&knots, &pitch, morph_t),
            average_energy: cubic_spline_interpolate(&knots, &energy, morph_t),
            speaking_rate: cubic_spline_interpolate(&knots, &rate, morph_t),
            ..SpeakerCharacteristics::default()
        };

        Ok(SpeakerProfile {
            id: request.target_id.clone(),
            name: format!("Cubic-Spline Morphed Voice - {}", request.target_id),
            characteristics: morphed_characteristics,
            samples: Vec::new(),
            embedding: Some(morphed_embedding),
            languages: profiles.iter().flat_map(|p| p.languages.clone()).collect(),
            created_at: std::time::SystemTime::now(),
            updated_at: std::time::SystemTime::now(),
            metadata: std::collections::HashMap::new(),
        })
    }

    /// Spherical morphing (SLERP for embeddings)
    async fn spherical_morphing(
        &self,
        profiles: &[SpeakerProfile],
        request: &VoiceMorphingRequest,
    ) -> Result<SpeakerProfile> {
        if profiles.len() == 2 {
            // True SLERP between two profiles
            self.slerp_two_profiles(&profiles[0], &profiles[1], request)
                .await
        } else {
            // Fallback to weighted for multiple profiles
            self.weighted_morphing(profiles, request).await
        }
    }

    /// Gaussian-mixture morphing.
    ///
    /// Models each speaker as an isotropic Gaussian component whose mean is its
    /// embedding and whose mixing weight is the (normalized) morph weight. As no
    /// covariance is tracked on the profiles, the per-component precision
    /// (`1/σ²`) is derived from the estimated profile quality: a higher-quality
    /// speaker forms a tighter Gaussian and therefore attracts the result more
    /// strongly, while equal qualities correspond to a shared identity
    /// covariance. The morphed embedding is the mixture expectation — the
    /// precision-weighted (responsibility-weighted) mean — which is continuous
    /// in the mixing weights and reduces to a single speaker's embedding when
    /// that speaker holds all the weight (see [`gaussian_mixture_blend`]).
    async fn gaussian_mixture_morphing(
        &self,
        profiles: &[SpeakerProfile],
        request: &VoiceMorphingRequest,
    ) -> Result<SpeakerProfile> {
        let mixing = normalize_weights(&request.speaker_weights);
        // Component precisions (inverse variance). With no covariance tracked we
        // map estimated quality to precision; equal qualities => identity
        // covariance and the blend collapses to the plain mixture expectation.
        let precisions: Vec<f32> = profiles
            .iter()
            .map(|p| 0.5 + self.estimate_profile_quality(p))
            .collect();

        let embeddings: Vec<Vec<f32>> = profiles
            .iter()
            .map(|p| p.embedding.clone().unwrap_or_default())
            .collect();
        let morphed_embedding = if embeddings.iter().all(|e| !e.is_empty()) {
            gaussian_mixture_blend(&embeddings, &mixing, &precisions)
        } else {
            vec![0.0; 512]
        };

        let pitch: Vec<f32> = profiles
            .iter()
            .map(|p| p.characteristics.average_pitch)
            .collect();
        let energy: Vec<f32> = profiles
            .iter()
            .map(|p| p.characteristics.average_energy)
            .collect();
        let rate: Vec<f32> = profiles
            .iter()
            .map(|p| p.characteristics.speaking_rate)
            .collect();

        let morphed_characteristics = SpeakerCharacteristics {
            average_pitch: gaussian_mixture_blend_scalar(&pitch, &mixing, &precisions),
            average_energy: gaussian_mixture_blend_scalar(&energy, &mixing, &precisions),
            speaking_rate: gaussian_mixture_blend_scalar(&rate, &mixing, &precisions),
            ..SpeakerCharacteristics::default()
        };

        Ok(SpeakerProfile {
            id: request.target_id.clone(),
            name: format!("Gaussian-Mixture Morphed Voice - {}", request.target_id),
            characteristics: morphed_characteristics,
            samples: Vec::new(),
            embedding: Some(morphed_embedding),
            languages: profiles.iter().flat_map(|p| p.languages.clone()).collect(),
            created_at: std::time::SystemTime::now(),
            updated_at: std::time::SystemTime::now(),
            metadata: std::collections::HashMap::new(),
        })
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

        // True spherical linear interpolation of the embedding directions
        // (see [`slerp_vectors`]); magnitudes are interpolated linearly so the
        // endpoints are reproduced exactly.
        let slerp_embedding = match (&profile1.embedding, &profile2.embedding) {
            (Some(emb1), Some(emb2)) => slerp_vectors(emb1, emb2, weight),
            _ => vec![0.0; 512],
        };

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

// ===========================================================================
// Interpolation math
//
// The routines below perform the heavy lifting for the smooth morphing
// methods. They operate purely on `f32`/`Vec` data (the speaker embeddings and
// the scalar voice characteristics) and pull in no external math crates, in
// line with the SciRS2 policy.
// ===========================================================================

/// Dot product of two vectors over their shared leading dimensions.
fn vector_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Euclidean (L2) norm of a vector.
fn vector_norm(v: &[f32]) -> f32 {
    vector_dot(v, v).sqrt()
}

/// Normalize `v` to unit length, returning `fallback` when `v` is ~zero.
fn normalize_or(v: &[f32], fallback: &[f32]) -> Vec<f32> {
    let norm = vector_norm(v);
    if norm < f32::EPSILON {
        fallback.to_vec()
    } else {
        let inv = 1.0 / norm;
        v.iter().map(|&x| x * inv).collect()
    }
}

/// Construct a unit vector orthogonal to the (already unit-length) `unit`.
///
/// Used by [`slerp_vectors`] to resolve the otherwise-undefined great circle
/// between two antiparallel vectors. The canonical basis axis least aligned
/// with `unit` is projected orthogonal to `unit` (Gram-Schmidt) and renormalized.
fn orthonormal_companion(unit: &[f32]) -> Vec<f32> {
    let dim = unit.len();
    let mut companion = vec![0.0f32; dim];
    if dim == 0 {
        return companion;
    }

    // Pick the canonical basis vector least aligned with `unit`.
    let mut min_index = 0usize;
    let mut min_abs = f32::INFINITY;
    for (i, &value) in unit.iter().enumerate() {
        let magnitude = value.abs();
        if magnitude < min_abs {
            min_abs = magnitude;
            min_index = i;
        }
    }
    companion[min_index] = 1.0;

    // Remove the projection onto `unit` so the companion becomes orthogonal.
    let projection = unit[min_index];
    for (c, &u) in companion.iter_mut().zip(unit.iter()) {
        *c -= projection * u;
    }

    let norm = vector_norm(&companion);
    if norm < f32::EPSILON {
        companion[min_index] = 1.0;
        companion
    } else {
        let inv = 1.0 / norm;
        for c in companion.iter_mut() {
            *c *= inv;
        }
        companion
    }
}

/// Spherical linear interpolation (SLERP) between two embedding vectors.
///
/// Directions are interpolated on the unit hypersphere with the canonical
/// identity `slerp(â, b̂, t) = sin((1−t)·Ω)/sin(Ω)·â + sin(t·Ω)/sin(Ω)·b̂`,
/// where `Ω = acos(â·b̂)`. The vector magnitude is interpolated linearly and
/// reapplied, so unit-norm inputs yield a unit-norm result and the endpoints
/// are reproduced exactly (`t = 0 → a`, `t = 1 → b`). Two degenerate
/// configurations are handled explicitly:
/// * **near-parallel** (`Ω ≈ 0`): a normalized linear blend, the numerically
///   stable limit of SLERP;
/// * **near-antiparallel** (`Ω ≈ π`): rotation from `â` through an arbitrary
///   orthonormal companion direction by `π·t`, keeping the path continuous and
///   unit-norm.
fn slerp_vectors(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    let dim = a.len().min(b.len());
    if dim == 0 {
        return Vec::new();
    }
    let a = &a[..dim];
    let b = &b[..dim];

    let norm_a = vector_norm(a);
    let norm_b = vector_norm(b);
    let target_magnitude = norm_a * (1.0 - t) + norm_b * t;

    // Without a well-defined direction for one side, fall back to a plain lerp.
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x * (1.0 - t) + y * t)
            .collect();
    }

    let inv_a = 1.0 / norm_a;
    let inv_b = 1.0 / norm_b;
    let a_hat: Vec<f32> = a.iter().map(|&x| x * inv_a).collect();
    let b_hat: Vec<f32> = b.iter().map(|&x| x * inv_b).collect();

    let cos_omega = vector_dot(&a_hat, &b_hat).clamp(-1.0, 1.0);

    let direction: Vec<f32> = if cos_omega > 1.0 - 1e-6 {
        // Near-parallel: normalized linear interpolation (stable SLERP limit).
        let lerp: Vec<f32> = a_hat
            .iter()
            .zip(b_hat.iter())
            .map(|(&x, &y)| x * (1.0 - t) + y * t)
            .collect();
        normalize_or(&lerp, &a_hat)
    } else if cos_omega < -1.0 + 1e-6 {
        // Near-antiparallel: rotate through an orthonormal companion direction.
        let perp = orthonormal_companion(&a_hat);
        let angle = std::f32::consts::PI * t;
        let (sin_angle, cos_angle) = angle.sin_cos();
        a_hat
            .iter()
            .zip(perp.iter())
            .map(|(&x, &p)| cos_angle * x + sin_angle * p)
            .collect()
    } else {
        let omega = cos_omega.acos();
        let sin_omega = omega.sin();
        let w_a = ((1.0 - t) * omega).sin() / sin_omega;
        let w_b = (t * omega).sin() / sin_omega;
        a_hat
            .iter()
            .zip(b_hat.iter())
            .map(|(&x, &y)| w_a * x + w_b * y)
            .collect()
    };

    direction.iter().map(|&d| d * target_magnitude).collect()
}

/// Normalize morph weights into mixing proportions that sum to one.
///
/// Negative weights are clamped to zero; if the total is non-positive a uniform
/// distribution is returned so downstream blends never divide by zero.
fn normalize_weights(weights: &[MorphingWeight]) -> Vec<f32> {
    let total: f32 = weights.iter().map(|w| w.weight.max(0.0)).sum();
    if total > f32::EPSILON {
        weights.iter().map(|w| w.weight.max(0.0) / total).collect()
    } else if weights.is_empty() {
        Vec::new()
    } else {
        vec![1.0 / weights.len() as f32; weights.len()]
    }
}

/// Evenly spaced spline knots covering `[0, 1]` for `n` control points.
fn uniform_knots(n: usize) -> Vec<f32> {
    match n {
        0 => Vec::new(),
        1 => vec![0.0],
        _ => (0..n).map(|i| i as f32 / (n - 1) as f32).collect(),
    }
}

/// Solve for the natural-cubic-spline second derivatives (the "moments").
///
/// Given interval widths `h` and knot values `values`, this assembles the
/// symmetric tridiagonal system tying consecutive moments together and solves
/// it with the Thomas algorithm. The first and last moments are pinned to zero
/// (the *natural* boundary condition), so a series of fewer than three points
/// trivially has zero moments and the spline degrades to a straight line.
fn solve_natural_spline_moments(h: &[f32], values: &[f32]) -> Vec<f32> {
    let n = values.len();
    let mut moments = vec![0.0f32; n];
    if n < 3 {
        return moments;
    }

    let interior = n - 2;
    let mut sub = vec![0.0f32; interior];
    let mut diag = vec![0.0f32; interior];
    let mut sup = vec![0.0f32; interior];
    let mut rhs = vec![0.0f32; interior];

    for k in 0..interior {
        let i = k + 1;
        sub[k] = h[i - 1];
        diag[k] = 2.0 * (h[i - 1] + h[i]);
        sup[k] = h[i];
        rhs[k] =
            6.0 * ((values[i + 1] - values[i]) / h[i] - (values[i] - values[i - 1]) / h[i - 1]);
    }

    // Thomas algorithm: forward elimination then back substitution.
    for k in 1..interior {
        let factor = sub[k] / diag[k - 1];
        diag[k] -= factor * sup[k - 1];
        rhs[k] -= factor * rhs[k - 1];
    }

    let mut solution = vec![0.0f32; interior];
    solution[interior - 1] = rhs[interior - 1] / diag[interior - 1];
    for k in (0..interior - 1).rev() {
        solution[k] = (rhs[k] - sup[k] * solution[k + 1]) / diag[k];
    }

    for (k, &value) in solution.iter().enumerate() {
        moments[k + 1] = value;
    }
    moments
}

/// Evaluate a one-dimensional natural cubic spline at parameter `t`.
///
/// `knots` must be strictly increasing and the same length as `values`. Because
/// the natural cubic spline interpolates every knot exactly, evaluating at the
/// first/last knot returns the first/last value — this is what guarantees the
/// morph reduces to its endpoints. Evaluation outside the knot span is clamped.
fn cubic_spline_interpolate(knots: &[f32], values: &[f32], t: f32) -> f32 {
    let n = knots.len();
    match n {
        0 => return 0.0,
        1 => return values.first().copied().unwrap_or(0.0),
        _ => {}
    }

    let h: Vec<f32> = knots
        .windows(2)
        .map(|w| {
            let width = w[1] - w[0];
            if width.abs() < f32::EPSILON {
                f32::EPSILON
            } else {
                width
            }
        })
        .collect();

    let moments = solve_natural_spline_moments(&h, values);

    let t_clamped = t.clamp(knots[0], knots[n - 1]);
    let mut i = 0;
    while i < n - 2 && t_clamped > knots[i + 1] {
        i += 1;
    }

    let hi = h[i];
    let lower = (knots[i + 1] - t_clamped) / hi;
    let upper = (t_clamped - knots[i]) / hi;
    lower * values[i]
        + upper * values[i + 1]
        + ((lower.powi(3) - lower) * moments[i] + (upper.powi(3) - upper) * moments[i + 1])
            * hi
            * hi
            / 6.0
}

/// Element-wise natural cubic spline interpolation of vector control points.
///
/// Each `points[k]` is a control-point vector positioned at `knots[k]`; the
/// spline is built and evaluated independently per dimension at `t`.
fn cubic_spline_interpolate_vector(knots: &[f32], points: &[Vec<f32>], t: f32) -> Vec<f32> {
    if points.is_empty() {
        return Vec::new();
    }
    let dim = points.iter().map(|p| p.len()).min().unwrap_or(0);
    let n = points.len();
    let mut result = vec![0.0f32; dim];
    let mut column = vec![0.0f32; n];
    for (d, slot) in result.iter_mut().enumerate() {
        for (k, point) in points.iter().enumerate() {
            column[k] = point[d];
        }
        *slot = cubic_spline_interpolate(knots, &column, t);
    }
    result
}

/// Responsibility-weighted Gaussian-mixture blend of vector control points.
///
/// Each control point `points[k]` is the mean `μ_k` of an isotropic Gaussian
/// component with mixing weight `mixing[k]` and precision `precisions[k]`
/// (`= 1/σ_k²`). The blended embedding is the mixture expectation: for a shared
/// covariance this is the convex combination `Σ π_k μ_k`, and for differing
/// covariances it becomes the precision-weighted (responsibility-weighted) mean
/// `Σ (π_k·prec_k) μ_k / Σ (π_k·prec_k)`. With equal precisions (identity
/// covariance) it reduces exactly to the interpolated-mixture expectation; it is
/// continuous in the mixing weights, strictly monotonic along the `μ_0 → μ_1`
/// segment as a single weight is swept, and reproduces a component mean whenever
/// that component carries all the mixing weight.
fn gaussian_mixture_blend(points: &[Vec<f32>], mixing: &[f32], precisions: &[f32]) -> Vec<f32> {
    let dim = points.iter().map(|p| p.len()).min().unwrap_or(0);
    let mut result = vec![0.0f32; dim];
    let mut total = 0.0f32;

    for ((point, &weight), &precision) in points.iter().zip(mixing.iter()).zip(precisions.iter()) {
        let effective = weight.max(0.0) * precision.max(f32::EPSILON);
        if effective <= 0.0 {
            continue;
        }
        total += effective;
        for (slot, &value) in result.iter_mut().zip(point.iter()) {
            *slot += effective * value;
        }
    }

    if total > f32::EPSILON {
        let inv = 1.0 / total;
        for slot in result.iter_mut() {
            *slot *= inv;
        }
    }
    result
}

/// Scalar form of [`gaussian_mixture_blend`] for the voice characteristics.
fn gaussian_mixture_blend_scalar(values: &[f32], mixing: &[f32], precisions: &[f32]) -> f32 {
    let mut accumulated = 0.0f32;
    let mut total = 0.0f32;
    for ((&value, &weight), &precision) in values.iter().zip(mixing.iter()).zip(precisions.iter()) {
        let effective = weight.max(0.0) * precision.max(f32::EPSILON);
        accumulated += effective * value;
        total += effective;
    }
    if total > f32::EPSILON {
        accumulated / total
    } else if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
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

    #[test]
    fn test_interpolators_reduce_to_endpoints() {
        let source = vec![0.2_f32, -0.5, 0.7, 1.0];
        let target = vec![-0.3_f32, 0.8, 0.1, -0.4];
        let points = vec![source.clone(), target.clone()];

        // Cubic spline: two control points at knots {0, 1}.
        let knots = uniform_knots(2);
        let spline_start = cubic_spline_interpolate_vector(&knots, &points, 0.0);
        let spline_end = cubic_spline_interpolate_vector(&knots, &points, 1.0);

        // Gaussian mixture: all mixing weight on a single component.
        let precisions = vec![1.0_f32, 1.0];
        let gmm_start = gaussian_mixture_blend(&points, &[1.0, 0.0], &precisions);
        let gmm_end = gaussian_mixture_blend(&points, &[0.0, 1.0], &precisions);

        // SLERP at the parameter endpoints.
        let slerp_start = slerp_vectors(&source, &target, 0.0);
        let slerp_end = slerp_vectors(&source, &target, 1.0);

        for d in 0..source.len() {
            assert!((spline_start[d] - source[d]).abs() < 1e-5);
            assert!((spline_end[d] - target[d]).abs() < 1e-5);
            assert!((gmm_start[d] - source[d]).abs() < 1e-5);
            assert!((gmm_end[d] - target[d]).abs() < 1e-5);
            assert!((slerp_start[d] - source[d]).abs() < 1e-4);
            assert!((slerp_end[d] - target[d]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_cubic_spline_collinear_is_linear() {
        // Three collinear control points p_k = base + k * delta. A natural cubic
        // spline through collinear data must coincide with the straight line —
        // i.e. plain linear interpolation between the endpoints.
        let base = [1.0_f32, -2.0, 0.5];
        let delta = [0.4_f32, 0.25, -0.6];
        let points: Vec<Vec<f32>> = (0..3)
            .map(|k| {
                base.iter()
                    .zip(delta.iter())
                    .map(|(&b, &d)| b + k as f32 * d)
                    .collect()
            })
            .collect();
        let knots = uniform_knots(3); // {0.0, 0.5, 1.0}
        let first = points[0].clone();
        let last = points[2].clone();

        for step in 0..=10 {
            let t = step as f32 / 10.0;
            let spline = cubic_spline_interpolate_vector(&knots, &points, t);
            for d in 0..base.len() {
                let linear = first[d] * (1.0 - t) + last[d] * t;
                assert!(
                    (spline[d] - linear).abs() < 1e-4,
                    "spline deviates from line at t={t}, dim={d}: {} vs {}",
                    spline[d],
                    linear
                );
            }
        }
    }

    #[test]
    fn test_slerp_orthogonal_unit_vectors_midpoint() {
        // Two orthogonal unit vectors; their SLERP midpoint must stay on the
        // unit sphere and sit at equal angles (45°) from each input.
        let a = vec![1.0_f32, 0.0, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0, 0.0];
        let mid = slerp_vectors(&a, &b, 0.5);

        let norm = vector_norm(&mid);
        assert!((norm - 1.0).abs() < 1e-5, "midpoint not unit norm: {norm}");

        let angle_a = vector_dot(&mid, &a).clamp(-1.0, 1.0).acos();
        let angle_b = vector_dot(&mid, &b).clamp(-1.0, 1.0).acos();
        assert!(
            (angle_a - angle_b).abs() < 1e-5,
            "not equidistant from inputs"
        );
        let quarter = std::f32::consts::FRAC_PI_4;
        assert!(
            (angle_a - quarter).abs() < 1e-4,
            "expected 45° to each input, got {angle_a}"
        );
    }

    #[test]
    fn test_gaussian_mixture_blend_monotonic_continuous() {
        // Sweep the morph factor and verify the blend marches monotonically and
        // continuously from the source mean toward the target mean. Unequal
        // precisions exercise the responsibility-weighted (non-trivial) path.
        let source = vec![0.0_f32, 0.0, 0.0];
        let target = vec![2.0_f32, -4.0, 1.0];
        let points = vec![source.clone(), target.clone()];
        let precisions = vec![1.3_f32, 0.7];

        let direction: Vec<f32> = target
            .iter()
            .zip(source.iter())
            .map(|(&b, &a)| b - a)
            .collect();

        let mut previous_projection = f32::NEG_INFINITY;
        let mut previous_point: Option<Vec<f32>> = None;
        let steps = 50;
        for step in 0..=steps {
            let t = step as f32 / steps as f32;
            let blend = gaussian_mixture_blend(&points, &[1.0 - t, t], &precisions);

            // Monotonic progression along the source→target axis.
            let projection = vector_dot(&blend, &direction);
            assert!(
                projection >= previous_projection - 1e-6,
                "projection decreased at t={t}"
            );
            previous_projection = projection;

            // Continuity: consecutive samples stay close for small steps.
            if let Some(prev) = &previous_point {
                let delta: f32 = blend
                    .iter()
                    .zip(prev.iter())
                    .map(|(&x, &y)| (x - y).abs())
                    .sum();
                assert!(delta < 0.75, "discontinuous jump at t={t}: {delta}");
            }
            previous_point = Some(blend);
        }

        // Endpoints recovered exactly.
        let at_zero = gaussian_mixture_blend(&points, &[1.0, 0.0], &precisions);
        let at_one = gaussian_mixture_blend(&points, &[0.0, 1.0], &precisions);
        for d in 0..source.len() {
            assert!((at_zero[d] - source[d]).abs() < 1e-6);
            assert!((at_one[d] - target[d]).abs() < 1e-6);
        }
    }
}
