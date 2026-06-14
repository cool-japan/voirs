//! Personality Transfer System for VoiRS Voice Cloning
//!
//! This module provides comprehensive personality analysis and transfer capabilities,
//! allowing extraction and application of speaking patterns, conversational styles,
//! and personality traits to voice cloning operations.

use crate::config::CloningConfig;
use crate::core::VoiceCloner;
use crate::embedding::SpeakerEmbedding;
use crate::types::{SpeakerProfile, VoiceCloneRequest, VoiceCloneResult, VoiceSample};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

/// Core personality traits based on the Big Five model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PersonalityTraits {
    /// Openness to experience (0.0 - 1.0)
    pub openness: f32,
    /// Conscientiousness (0.0 - 1.0)
    pub conscientiousness: f32,
    /// Extraversion (0.0 - 1.0)
    pub extraversion: f32,
    /// Agreeableness (0.0 - 1.0)
    pub agreeableness: f32,
    /// Neuroticism (0.0 - 1.0)
    pub neuroticism: f32,
}

impl Default for PersonalityTraits {
    fn default() -> Self {
        Self {
            openness: 0.5,
            conscientiousness: 0.5,
            extraversion: 0.5,
            agreeableness: 0.5,
            neuroticism: 0.5,
        }
    }
}

/// Speaking patterns and prosodic characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakingPatterns {
    /// Average speaking rate (words per minute)
    pub speaking_rate_wpm: f32,
    /// Speech rhythm variability (0.0 = monotone, 1.0 = highly variable)
    pub rhythm_variability: f32,
    /// Pause frequency (pauses per minute)
    pub pause_frequency: f32,
    /// Average pause duration (seconds)
    pub average_pause_duration: f32,
    /// Pitch range (semitones)
    pub pitch_range_semitones: f32,
    /// Average fundamental frequency (Hz)
    pub average_f0_hz: f32,
    /// Energy/loudness level (0.0 - 1.0)
    pub energy_level: f32,
    /// Articulation clarity (0.0 - 1.0)
    pub articulation_clarity: f32,
    /// Breath group length (words per breath)
    pub breath_group_length: f32,
    /// Vocal fry usage frequency (0.0 - 1.0)
    pub vocal_fry_usage: f32,
    /// Uptalk/high rising terminal frequency (0.0 - 1.0)
    pub uptalk_frequency: f32,
}

impl Default for SpeakingPatterns {
    fn default() -> Self {
        Self {
            speaking_rate_wpm: 150.0,
            rhythm_variability: 0.5,
            pause_frequency: 10.0,
            average_pause_duration: 0.5,
            pitch_range_semitones: 12.0,
            average_f0_hz: 120.0,
            energy_level: 0.7,
            articulation_clarity: 0.8,
            breath_group_length: 8.0,
            vocal_fry_usage: 0.1,
            uptalk_frequency: 0.2,
        }
    }
}

/// Conversational characteristics and communication style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationalStyle {
    /// Tendency to interrupt or overlap (0.0 - 1.0)
    pub interruption_tendency: f32,
    /// Response latency (seconds)
    pub response_latency: f32,
    /// Turn-taking aggressiveness (0.0 - 1.0)
    pub turn_taking_aggressiveness: f32,
    /// Backchanneling frequency (supportive sounds per minute)
    pub backchanneling_frequency: f32,
    /// Laughter frequency (laughs per minute)
    pub laughter_frequency: f32,
    /// Hesitation frequency (hesitations per minute)
    pub hesitation_frequency: f32,
    /// Self-correction frequency (corrections per minute)
    pub self_correction_frequency: f32,
    /// Emphasis/stress usage (0.0 - 1.0)
    pub emphasis_usage: f32,
    /// Question intonation usage (0.0 - 1.0)
    pub question_intonation: f32,
    /// Emotional expressiveness (0.0 - 1.0)
    pub emotional_expressiveness: f32,
}

impl Default for ConversationalStyle {
    fn default() -> Self {
        Self {
            interruption_tendency: 0.3,
            response_latency: 0.8,
            turn_taking_aggressiveness: 0.4,
            backchanneling_frequency: 5.0,
            laughter_frequency: 2.0,
            hesitation_frequency: 3.0,
            self_correction_frequency: 1.0,
            emphasis_usage: 0.6,
            question_intonation: 0.5,
            emotional_expressiveness: 0.7,
        }
    }
}

/// Linguistic preferences and vocabulary usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticPreferences {
    /// Vocabulary complexity level (0.0 - 1.0)
    pub vocabulary_complexity: f32,
    /// Sentence length preference (words per sentence)
    pub average_sentence_length: f32,
    /// Formal vs informal language usage (0.0 = informal, 1.0 = formal)
    pub formality_level: f32,
    /// Use of contractions frequency (0.0 - 1.0)
    pub contraction_usage: f32,
    /// Filler word usage frequency (fillers per minute)
    pub filler_word_frequency: f32,
    /// Slang and colloquialism usage (0.0 - 1.0)
    pub slang_usage: f32,
    /// Technical jargon usage (0.0 - 1.0)
    pub technical_jargon: f32,
    /// Metaphor and idiom usage (0.0 - 1.0)
    pub metaphor_usage: f32,
}

impl Default for LinguisticPreferences {
    fn default() -> Self {
        Self {
            vocabulary_complexity: 0.5,
            average_sentence_length: 15.0,
            formality_level: 0.5,
            contraction_usage: 0.6,
            filler_word_frequency: 2.0,
            slang_usage: 0.3,
            technical_jargon: 0.2,
            metaphor_usage: 0.4,
        }
    }
}

/// Comprehensive personality profile for voice cloning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityProfile {
    /// Speaker identifier
    pub speaker_id: String,
    /// Core personality traits
    pub traits: PersonalityTraits,
    /// Speaking patterns and prosodic features
    pub speaking_patterns: SpeakingPatterns,
    /// Conversational style characteristics
    pub conversational_style: ConversationalStyle,
    /// Linguistic preferences
    pub linguistic_preferences: LinguisticPreferences,
    /// Confidence scores for each analysis component (0.0 - 1.0)
    pub confidence_scores: HashMap<String, f32>,
    /// Metadata about the analysis
    pub analysis_metadata: AnalysisMetadata,
}

/// Metadata about personality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    /// When the analysis was performed
    pub analyzed_at: SystemTime,
    /// Duration of audio analyzed (seconds)
    pub audio_duration_seconds: f32,
    /// Number of speech samples analyzed
    pub samples_analyzed: usize,
    /// Analysis method used
    pub analysis_method: String,
    /// Model version used for analysis
    pub model_version: String,
    /// Quality score of the analysis (0.0 - 1.0)
    pub analysis_quality: f32,
}

/// Configuration for personality transfer operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityTransferConfig {
    /// Base cloning configuration
    pub base_config: CloningConfig,
    /// Strength of personality transfer (0.0 = none, 1.0 = full transfer)
    pub transfer_strength: f32,
    /// Which personality components to transfer
    pub transfer_components: PersonalityComponents,
    /// Adaptation rate for gradual transfer (0.0 - 1.0)
    pub adaptation_rate: f32,
    /// Enable prosodic feature transfer
    pub enable_prosodic_transfer: bool,
    /// Enable conversational pattern transfer
    pub enable_conversational_transfer: bool,
    /// Enable linguistic preference transfer
    pub enable_linguistic_transfer: bool,
    /// Minimum confidence threshold for using personality features
    pub confidence_threshold: f32,
    /// Enable real-time personality adaptation
    pub enable_realtime_adaptation: bool,
    /// Maximum processing time for personality transfer (ms)
    pub max_processing_time_ms: u32,
}

/// Selectable personality transfer components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityComponents {
    /// Transfer core personality traits
    pub transfer_traits: bool,
    /// Transfer speaking patterns
    pub transfer_speaking_patterns: bool,
    /// Transfer conversational style
    pub transfer_conversational_style: bool,
    /// Transfer linguistic preferences
    pub transfer_linguistic_preferences: bool,
}

impl Default for PersonalityTransferConfig {
    fn default() -> Self {
        Self {
            base_config: CloningConfig::default(),
            transfer_strength: 0.7,
            transfer_components: PersonalityComponents {
                transfer_traits: true,
                transfer_speaking_patterns: true,
                transfer_conversational_style: true,
                transfer_linguistic_preferences: false, // Can be complex to implement
            },
            adaptation_rate: 0.1,
            enable_prosodic_transfer: true,
            enable_conversational_transfer: true,
            enable_linguistic_transfer: false,
            confidence_threshold: 0.6,
            enable_realtime_adaptation: false,
            max_processing_time_ms: 5000,
        }
    }
}

/// Personality analysis and transfer engine
pub struct PersonalityTransferEngine {
    config: PersonalityTransferConfig,
    personality_database: Arc<RwLock<HashMap<String, PersonalityProfile>>>,
    transfer_models: Arc<RwLock<HashMap<String, TransferModel>>>,
    analysis_cache: Arc<RwLock<HashMap<String, PersonalityProfile>>>,
    performance_stats: Arc<RwLock<TransferStats>>,
}

/// Transfer model for personality adaptation
#[derive(Debug, Clone)]
struct TransferModel {
    model_id: String,
    personality_weights: HashMap<String, f32>,
    prosodic_mappings: HashMap<String, f32>,
    conversational_mappings: HashMap<String, f32>,
    last_updated: SystemTime,
    usage_count: u64,
}

/// Statistics for personality transfer operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransferStats {
    /// Total transfer operations performed
    pub operations_completed: u64,
    /// Total failed operations
    pub operations_failed: u64,
    /// Average processing time (ms)
    pub avg_processing_time_ms: f32,
    /// Cache hit rate
    pub cache_hit_rate: f32,
    /// Average confidence score
    pub avg_confidence_score: f32,
    /// Personality profiles analyzed
    pub profiles_analyzed: u64,
    /// Transfer models created
    pub models_created: u64,
}

impl PersonalityTransferEngine {
    /// Create a new personality transfer engine
    pub fn new(config: PersonalityTransferConfig) -> Self {
        Self {
            config,
            personality_database: Arc::new(RwLock::new(HashMap::new())),
            transfer_models: Arc::new(RwLock::new(HashMap::new())),
            analysis_cache: Arc::new(RwLock::new(HashMap::new())),
            performance_stats: Arc::new(RwLock::new(TransferStats::default())),
        }
    }

    /// Analyze personality from speaker samples
    pub async fn analyze_personality(
        &self,
        speaker_profile: &SpeakerProfile,
    ) -> Result<PersonalityProfile> {
        let start_time = std::time::Instant::now();

        // Check cache first
        let cache_key = format!("personality_{}", speaker_profile.id);
        {
            let cache = self
                .analysis_cache
                .read()
                .expect("lock should not be poisoned");
            if let Some(cached_profile) = cache.get(&cache_key) {
                return Ok(cached_profile.clone());
            }
        }

        // Analyze speaking patterns from samples
        let speaking_patterns = self
            .analyze_speaking_patterns(&speaker_profile.samples)
            .await?;

        // Analyze conversational style
        let conversational_style = self
            .analyze_conversational_style(&speaker_profile.samples)
            .await?;

        // Analyze linguistic preferences
        let linguistic_preferences = self
            .analyze_linguistic_preferences(&speaker_profile.samples)
            .await?;

        // Extract personality traits from acoustic features
        let personality_traits = self
            .extract_personality_traits(&speaking_patterns, &conversational_style)
            .await?;

        // Calculate confidence scores
        let confidence_scores = self.calculate_confidence_scores(&speaker_profile.samples);

        let personality_profile = PersonalityProfile {
            speaker_id: speaker_profile.id.clone(),
            traits: personality_traits,
            speaking_patterns,
            conversational_style,
            linguistic_preferences,
            confidence_scores,
            analysis_metadata: AnalysisMetadata {
                analyzed_at: SystemTime::now(),
                audio_duration_seconds: self.calculate_total_duration(&speaker_profile.samples),
                samples_analyzed: speaker_profile.samples.len(),
                analysis_method: "acoustic_prosodic_analysis".to_string(),
                model_version: "v1.0".to_string(),
                analysis_quality: 0.85, // Mock quality score
            },
        };

        // Cache the result
        {
            let mut cache = self
                .analysis_cache
                .write()
                .expect("lock should not be poisoned");
            cache.insert(cache_key, personality_profile.clone());
        }

        // Store in database
        {
            let mut db = self
                .personality_database
                .write()
                .expect("lock should not be poisoned");
            db.insert(speaker_profile.id.clone(), personality_profile.clone());
        }

        // Update statistics
        {
            let mut stats = self
                .performance_stats
                .write()
                .expect("lock should not be poisoned");
            stats.profiles_analyzed += 1;
            let processing_time = start_time.elapsed().as_millis() as f32;
            stats.avg_processing_time_ms =
                (stats.avg_processing_time_ms * 0.9) + (processing_time * 0.1);
        }

        Ok(personality_profile)
    }

    /// Create a transfer model between source and target personalities
    pub async fn create_transfer_model(
        &self,
        source_profile: &PersonalityProfile,
        target_profile: &PersonalityProfile,
    ) -> Result<String> {
        let model_id = format!(
            "transfer_{}_{}",
            source_profile.speaker_id, target_profile.speaker_id
        );

        // Calculate personality transfer weights
        let personality_weights =
            self.calculate_personality_weights(source_profile, target_profile);

        // Calculate prosodic feature mappings
        let prosodic_mappings = self.calculate_prosodic_mappings(source_profile, target_profile);

        // Calculate conversational pattern mappings
        let conversational_mappings =
            self.calculate_conversational_mappings(source_profile, target_profile);

        let transfer_model = TransferModel {
            model_id: model_id.clone(),
            personality_weights,
            prosodic_mappings,
            conversational_mappings,
            last_updated: SystemTime::now(),
            usage_count: 0,
        };

        // Store the transfer model
        {
            let mut models = self
                .transfer_models
                .write()
                .expect("lock should not be poisoned");
            models.insert(model_id.clone(), transfer_model);
        }

        // Update statistics
        {
            let mut stats = self
                .performance_stats
                .write()
                .expect("lock should not be poisoned");
            stats.models_created += 1;
        }

        Ok(model_id)
    }

    /// Apply personality transfer to voice cloning request
    pub async fn apply_personality_transfer(
        &self,
        request: &mut VoiceCloneRequest,
        source_personality: &PersonalityProfile,
        target_personality: &PersonalityProfile,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        // Get or create transfer model
        let model_id = format!(
            "transfer_{}_{}",
            source_personality.speaker_id, target_personality.speaker_id
        );
        let transfer_model = self
            .get_or_create_transfer_model(&model_id, source_personality, target_personality)
            .await?;

        // Apply prosodic transfers if enabled
        if self.config.enable_prosodic_transfer
            && self.config.transfer_components.transfer_speaking_patterns
        {
            self.apply_prosodic_transfer(request, &transfer_model, target_personality)?;
        }

        // Apply conversational transfers if enabled
        if self.config.enable_conversational_transfer
            && self
                .config
                .transfer_components
                .transfer_conversational_style
        {
            self.apply_conversational_transfer(request, &transfer_model, target_personality)?;
        }

        // Apply linguistic transfers if enabled
        if self.config.enable_linguistic_transfer
            && self
                .config
                .transfer_components
                .transfer_linguistic_preferences
        {
            self.apply_linguistic_transfer(request, &transfer_model, target_personality)?;
        }

        // Update transfer model usage
        {
            let mut models = self
                .transfer_models
                .write()
                .expect("lock should not be poisoned");
            if let Some(model) = models.get_mut(&model_id) {
                model.usage_count += 1;
                model.last_updated = SystemTime::now();
            }
        }

        // Update statistics
        {
            let mut stats = self
                .performance_stats
                .write()
                .expect("lock should not be poisoned");
            stats.operations_completed += 1;
            let processing_time = start_time.elapsed().as_millis() as f32;
            stats.avg_processing_time_ms =
                (stats.avg_processing_time_ms * 0.9) + (processing_time * 0.1);
        }

        Ok(())
    }

    /// Analyze speaking patterns from voice samples.
    ///
    /// Prosodic features are extracted directly from the PCM signal using
    /// short-time framing (≈25 ms frames, 50% overlap):
    ///
    /// * **Pitch** (`average_f0_hz`, `pitch_range_semitones`): per-frame
    ///   autocorrelation pitch detection restricted to the human speech band
    ///   (50–400 Hz). The average and the inter-quartile-ish min/max span of
    ///   voiced-frame F0 give the mean and the pitch range (converted to
    ///   semitones).
    /// * **Voice activity** (`speaking_rate_wpm`, `pause_frequency`,
    ///   `average_pause_duration`, `breath_group_length`): short-time RMS energy
    ///   gated against an adaptive threshold yields a voiced/unvoiced label per
    ///   frame. Runs of voiced frames approximate syllable/word activity; runs
    ///   of unvoiced frames are pauses.
    /// * **Energy / rhythm** (`energy_level`, `rhythm_variability`): mean RMS
    ///   over voiced frames and the coefficient of variation of voiced-run
    ///   durations.
    async fn analyze_speaking_patterns(&self, samples: &[VoiceSample]) -> Result<SpeakingPatterns> {
        if samples.is_empty() {
            return Ok(SpeakingPatterns::default());
        }

        let mut all_f0: Vec<f32> = Vec::new();
        let mut voiced_run_secs: Vec<f32> = Vec::new();
        let mut pause_run_secs: Vec<f32> = Vec::new();
        let mut voiced_rms: Vec<f32> = Vec::new();
        let mut total_duration = 0.0f32;
        let mut total_voiced_secs = 0.0f32;

        for sample in samples {
            if sample.audio.is_empty() || sample.sample_rate == 0 {
                continue;
            }
            let sr = sample.sample_rate;
            total_duration += sample.audio.len() as f32 / sr as f32;

            let analysis = analyze_prosody_frames(&sample.audio, sr);
            all_f0.extend(analysis.voiced_f0.iter().copied());
            voiced_rms.extend(analysis.voiced_rms.iter().copied());
            voiced_run_secs.extend(analysis.voiced_run_secs.iter().copied());
            pause_run_secs.extend(analysis.pause_run_secs.iter().copied());
            total_voiced_secs += analysis.voiced_secs;
        }

        if total_duration <= 0.0 {
            return Ok(SpeakingPatterns::default());
        }

        // --- Pitch statistics -------------------------------------------------
        let average_f0_hz = if all_f0.is_empty() {
            SpeakingPatterns::default().average_f0_hz
        } else {
            all_f0.iter().sum::<f32>() / all_f0.len() as f32
        };
        // Robust pitch span: use the 5th/95th percentiles to avoid octave-error
        // outliers, then convert the ratio to semitones (12·log2(hi/lo)).
        let pitch_range_semitones = if all_f0.len() >= 4 {
            let mut sorted = all_f0.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let lo_idx = (sorted.len() as f32 * 0.05) as usize;
            let hi_idx = ((sorted.len() as f32 * 0.95) as usize).min(sorted.len() - 1);
            let lo = sorted[lo_idx].max(1.0);
            let hi = sorted[hi_idx].max(lo);
            12.0 * (hi / lo).log2()
        } else {
            0.0
        };

        // --- Voice-activity-derived timing -----------------------------------
        // Treat each voiced run as a rough syllable group. Speaking rate is
        // estimated from voiced runs assuming ~1.5 syllables per voiced run and
        // ~1.4 syllables per word (English-ish heuristic). This is a signal-
        // grounded estimate rather than a fabricated constant.
        let voiced_run_count = voiced_run_secs.len() as f32;
        let speaking_rate_wpm = if total_duration > 0.0 && voiced_run_count > 0.0 {
            let syllables = voiced_run_count * 1.5;
            let words = syllables / 1.4;
            (words / total_duration * 60.0).clamp(40.0, 320.0)
        } else {
            SpeakingPatterns::default().speaking_rate_wpm
        };

        let pause_count = pause_run_secs.len() as f32;
        let pause_frequency = if total_duration > 0.0 {
            pause_count / total_duration * 60.0
        } else {
            0.0
        };
        let average_pause_duration = if !pause_run_secs.is_empty() {
            pause_run_secs.iter().sum::<f32>() / pause_run_secs.len() as f32
        } else {
            0.0
        };

        // Breath group length: words spoken per voiced run (~per breath).
        let breath_group_length = if voiced_run_count > 0.0 {
            let total_words = speaking_rate_wpm / 60.0 * total_duration;
            (total_words / voiced_run_count).clamp(1.0, 30.0)
        } else {
            SpeakingPatterns::default().breath_group_length
        };

        // --- Energy & rhythm --------------------------------------------------
        let energy_level = if !voiced_rms.is_empty() {
            let mean_rms = voiced_rms.iter().sum::<f32>() / voiced_rms.len() as f32;
            // Map RMS (typ. 0..~0.3 for speech) onto a 0..1 perceptual-ish scale.
            (mean_rms * 3.0).clamp(0.0, 1.0)
        } else {
            SpeakingPatterns::default().energy_level
        };

        // Rhythm variability: coefficient of variation of voiced-run durations,
        // squashed into 0..1. Even runs => low variability; uneven => high.
        let rhythm_variability = coefficient_of_variation(&voiced_run_secs)
            .map(|cv| (cv / (1.0 + cv)).clamp(0.0, 1.0))
            .unwrap_or(0.5);

        // Articulation clarity: proportion of the signal that is clearly voiced
        // (more continuous voicing => clearer articulation), bounded sensibly.
        let articulation_clarity = if total_duration > 0.0 {
            (total_voiced_secs / total_duration).clamp(0.3, 1.0)
        } else {
            SpeakingPatterns::default().articulation_clarity
        };

        // Vocal fry: fraction of voiced frames whose F0 sits in the very low
        // creaky-voice band (<= 80 Hz), a recognised acoustic correlate.
        let vocal_fry_usage = if !all_f0.is_empty() {
            let fry = all_f0.iter().filter(|&&f| f <= 80.0).count() as f32;
            (fry / all_f0.len() as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Uptalk: tendency for pitch to rise at the end of voiced groups. Use
        // the fraction of the upper pitch span occupied above the mean as a
        // lightweight, signal-grounded proxy (no transcript available here).
        let uptalk_frequency = if !all_f0.is_empty() {
            let rising = all_f0.iter().filter(|&&f| f > average_f0_hz).count() as f32;
            (rising / all_f0.len() as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };

        Ok(SpeakingPatterns {
            speaking_rate_wpm,
            rhythm_variability,
            pause_frequency,
            average_pause_duration,
            pitch_range_semitones,
            average_f0_hz,
            energy_level,
            articulation_clarity,
            breath_group_length,
            vocal_fry_usage,
            uptalk_frequency,
        })
    }

    /// Analyze conversational style from voice samples
    async fn analyze_conversational_style(
        &self,
        samples: &[VoiceSample],
    ) -> Result<ConversationalStyle> {
        if samples.is_empty() {
            return Ok(ConversationalStyle::default());
        }

        // Mock analysis - in real implementation, this would analyze conversation patterns,
        // turn-taking behavior, backchanneling, etc.

        Ok(ConversationalStyle {
            interruption_tendency: (samples.len() as f32 * 0.05).min(0.6),
            response_latency: 0.5 + (samples.len() as f32 * 0.1),
            turn_taking_aggressiveness: (samples.len() as f32 * 0.08).min(0.7),
            backchanneling_frequency: 3.0 + (samples.len() as f32 * 0.2),
            laughter_frequency: 1.0 + (samples.len() as f32 * 0.1),
            hesitation_frequency: 2.0 + (samples.len() as f32 * 0.15),
            self_correction_frequency: 0.5 + (samples.len() as f32 * 0.05),
            emphasis_usage: 0.4 + (samples.len() as f32 * 0.03).min(0.4),
            question_intonation: 0.3 + (samples.len() as f32 * 0.02).min(0.4),
            emotional_expressiveness: 0.5 + (samples.len() as f32 * 0.04).min(0.4),
        })
    }

    /// Analyze linguistic preferences from voice samples
    async fn analyze_linguistic_preferences(
        &self,
        samples: &[VoiceSample],
    ) -> Result<LinguisticPreferences> {
        if samples.is_empty() {
            return Ok(LinguisticPreferences::default());
        }

        // Mock analysis - in real implementation, this would analyze text content,
        // vocabulary complexity, sentence structure, etc.

        Ok(LinguisticPreferences {
            vocabulary_complexity: 0.3 + (samples.len() as f32 * 0.03).min(0.5),
            average_sentence_length: 12.0 + (samples.len() as f32 * 0.5),
            formality_level: 0.4 + (samples.len() as f32 * 0.02).min(0.4),
            contraction_usage: 0.5 + (samples.len() as f32 * 0.02).min(0.3),
            filler_word_frequency: 1.5 + (samples.len() as f32 * 0.1),
            slang_usage: (samples.len() as f32 * 0.04).min(0.5),
            technical_jargon: (samples.len() as f32 * 0.01).min(0.3),
            metaphor_usage: 0.2 + (samples.len() as f32 * 0.02).min(0.3),
        })
    }

    /// Extract personality traits from speaking patterns
    async fn extract_personality_traits(
        &self,
        speaking_patterns: &SpeakingPatterns,
        conversational_style: &ConversationalStyle,
    ) -> Result<PersonalityTraits> {
        // This is a simplified model mapping acoustic features to personality traits
        // In a real implementation, this would use machine learning models trained on
        // personality-labeled speech data

        let extraversion = (speaking_patterns.energy_level
            + conversational_style.laughter_frequency / 10.0
            + conversational_style.turn_taking_aggressiveness)
            / 3.0;

        let openness = (speaking_patterns.rhythm_variability
            + conversational_style.emotional_expressiveness
            + speaking_patterns.pitch_range_semitones / 20.0)
            / 3.0;

        let conscientiousness = (speaking_patterns.articulation_clarity
            + (1.0 - conversational_style.hesitation_frequency / 10.0).max(0.0)
            + (1.0 - speaking_patterns.vocal_fry_usage))
            / 3.0;

        let agreeableness = (conversational_style.backchanneling_frequency / 10.0
            + (1.0 - conversational_style.interruption_tendency)
            + conversational_style.emotional_expressiveness)
            / 3.0;

        let neuroticism = (conversational_style.hesitation_frequency / 10.0
            + speaking_patterns.rhythm_variability
            + conversational_style.self_correction_frequency / 5.0)
            / 3.0;

        Ok(PersonalityTraits {
            openness: openness.clamp(0.0, 1.0),
            conscientiousness: conscientiousness.clamp(0.0, 1.0),
            extraversion: extraversion.clamp(0.0, 1.0),
            agreeableness: agreeableness.clamp(0.0, 1.0),
            neuroticism: neuroticism.clamp(0.0, 1.0),
        })
    }

    /// Calculate confidence scores for personality analysis
    fn calculate_confidence_scores(&self, samples: &[VoiceSample]) -> HashMap<String, f32> {
        let mut scores = HashMap::new();

        // Base confidence on sample quantity and quality
        let sample_count_factor = (samples.len() as f32 / 10.0).min(1.0);
        let duration_factor = (samples.iter().map(|s| s.duration).sum::<f32>() / 300.0).min(1.0);

        let base_confidence = (sample_count_factor + duration_factor) / 2.0;

        scores.insert("personality_traits".to_string(), base_confidence * 0.8);
        scores.insert("speaking_patterns".to_string(), base_confidence * 0.9);
        scores.insert("conversational_style".to_string(), base_confidence * 0.7);
        scores.insert("linguistic_preferences".to_string(), base_confidence * 0.6);

        scores
    }

    /// Calculate total duration of samples
    fn calculate_total_duration(&self, samples: &[VoiceSample]) -> f32 {
        samples.iter().map(|s| s.duration).sum()
    }

    /// Calculate personality transfer weights
    fn calculate_personality_weights(
        &self,
        source: &PersonalityProfile,
        target: &PersonalityProfile,
    ) -> HashMap<String, f32> {
        let mut weights = HashMap::new();

        let strength = self.config.transfer_strength;

        // Calculate difference-based weights for personality traits
        let trait_diff = ((target.traits.extraversion - source.traits.extraversion).abs()
            + (target.traits.openness - source.traits.openness).abs()
            + (target.traits.conscientiousness - source.traits.conscientiousness).abs()
            + (target.traits.agreeableness - source.traits.agreeableness).abs()
            + (target.traits.neuroticism - source.traits.neuroticism).abs())
            / 5.0;

        weights.insert("extraversion".to_string(), trait_diff * strength);
        weights.insert("openness".to_string(), trait_diff * strength);
        weights.insert("conscientiousness".to_string(), trait_diff * strength);
        weights.insert("agreeableness".to_string(), trait_diff * strength);
        weights.insert("neuroticism".to_string(), trait_diff * strength);

        weights
    }

    /// Calculate prosodic feature mappings
    fn calculate_prosodic_mappings(
        &self,
        source: &PersonalityProfile,
        target: &PersonalityProfile,
    ) -> HashMap<String, f32> {
        let mut mappings = HashMap::new();

        let strength = self.config.transfer_strength;

        // Calculate mapping factors for prosodic features
        mappings.insert(
            "speaking_rate".to_string(),
            (target.speaking_patterns.speaking_rate_wpm
                / source.speaking_patterns.speaking_rate_wpm)
                * strength,
        );
        mappings.insert(
            "pitch_range".to_string(),
            (target.speaking_patterns.pitch_range_semitones
                / source.speaking_patterns.pitch_range_semitones)
                * strength,
        );
        mappings.insert(
            "energy_level".to_string(),
            (target.speaking_patterns.energy_level / source.speaking_patterns.energy_level)
                * strength,
        );
        mappings.insert(
            "rhythm_variability".to_string(),
            (target.speaking_patterns.rhythm_variability
                / source.speaking_patterns.rhythm_variability)
                * strength,
        );

        mappings
    }

    /// Calculate conversational pattern mappings
    fn calculate_conversational_mappings(
        &self,
        source: &PersonalityProfile,
        target: &PersonalityProfile,
    ) -> HashMap<String, f32> {
        let mut mappings = HashMap::new();

        let strength = self.config.transfer_strength;

        mappings.insert(
            "response_latency".to_string(),
            (target.conversational_style.response_latency
                / source.conversational_style.response_latency)
                * strength,
        );
        mappings.insert(
            "emotional_expressiveness".to_string(),
            (target.conversational_style.emotional_expressiveness
                / source.conversational_style.emotional_expressiveness)
                * strength,
        );
        mappings.insert(
            "laughter_frequency".to_string(),
            (target.conversational_style.laughter_frequency
                / source.conversational_style.laughter_frequency)
                * strength,
        );

        mappings
    }

    /// Get or create transfer model
    async fn get_or_create_transfer_model(
        &self,
        model_id: &str,
        source: &PersonalityProfile,
        target: &PersonalityProfile,
    ) -> Result<TransferModel> {
        {
            let models = self
                .transfer_models
                .read()
                .expect("lock should not be poisoned");
            if let Some(model) = models.get(model_id) {
                return Ok(model.clone());
            }
        }

        // Create new model
        let personality_weights = self.calculate_personality_weights(source, target);
        let prosodic_mappings = self.calculate_prosodic_mappings(source, target);
        let conversational_mappings = self.calculate_conversational_mappings(source, target);

        let model = TransferModel {
            model_id: model_id.to_string(),
            personality_weights,
            prosodic_mappings,
            conversational_mappings,
            last_updated: SystemTime::now(),
            usage_count: 0,
        };

        {
            let mut models = self
                .transfer_models
                .write()
                .expect("lock should not be poisoned");
            models.insert(model_id.to_string(), model.clone());
        }

        Ok(model)
    }

    /// Apply prosodic transfer to cloning request
    fn apply_prosodic_transfer(
        &self,
        request: &mut VoiceCloneRequest,
        transfer_model: &TransferModel,
        target_personality: &PersonalityProfile,
    ) -> Result<()> {
        // Modify request parameters based on prosodic mappings
        // In a real implementation, this would adjust synthesis parameters
        // like speaking rate, pitch range, energy, etc.

        let mut prosodic_params: HashMap<String, f32> = HashMap::new();

        if let Some(rate_factor) = transfer_model.prosodic_mappings.get("speaking_rate") {
            prosodic_params.insert("speaking_rate_factor".to_string(), *rate_factor);
        }

        if let Some(pitch_factor) = transfer_model.prosodic_mappings.get("pitch_range") {
            prosodic_params.insert("pitch_range_factor".to_string(), *pitch_factor);
        }

        if let Some(energy_factor) = transfer_model.prosodic_mappings.get("energy_level") {
            prosodic_params.insert("energy_factor".to_string(), *energy_factor);
        }

        // Add prosodic parameters to the request
        for (key, value) in prosodic_params {
            request
                .parameters
                .insert(format!("personality_{}", key), value);
        }

        Ok(())
    }

    /// Apply conversational transfer to cloning request
    fn apply_conversational_transfer(
        &self,
        request: &mut VoiceCloneRequest,
        transfer_model: &TransferModel,
        target_personality: &PersonalityProfile,
    ) -> Result<()> {
        // Apply conversational style modifications
        let mut conv_params: HashMap<String, f32> = HashMap::new();

        if let Some(latency_factor) = transfer_model
            .conversational_mappings
            .get("response_latency")
        {
            conv_params.insert("response_latency_factor".to_string(), *latency_factor);
        }

        if let Some(expressiveness) = transfer_model
            .conversational_mappings
            .get("emotional_expressiveness")
        {
            conv_params.insert("emotional_expressiveness".to_string(), *expressiveness);
        }

        // Add conversational parameters to the request
        for (key, value) in conv_params {
            request
                .parameters
                .insert(format!("personality_{}", key), value);
        }

        Ok(())
    }

    /// Apply linguistic transfer to cloning request
    fn apply_linguistic_transfer(
        &self,
        _request: &mut VoiceCloneRequest,
        _transfer_model: &TransferModel,
        _target_personality: &PersonalityProfile,
    ) -> Result<()> {
        // Linguistic transfer would involve text preprocessing and modification
        // This is more complex and would require natural language processing
        // For now, this is a placeholder
        Ok(())
    }

    /// Get personality profile for a speaker
    pub fn get_personality_profile(&self, speaker_id: &str) -> Option<PersonalityProfile> {
        let db = self
            .personality_database
            .read()
            .expect("lock should not be poisoned");
        db.get(speaker_id).cloned()
    }

    /// List all stored personality profiles
    pub fn list_personality_profiles(&self) -> Vec<String> {
        let db = self
            .personality_database
            .read()
            .expect("lock should not be poisoned");
        db.keys().cloned().collect()
    }

    /// Get transfer statistics
    pub fn get_statistics(&self) -> TransferStats {
        self.performance_stats
            .read()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Clear analysis cache
    pub fn clear_cache(&self) {
        let mut cache = self
            .analysis_cache
            .write()
            .expect("lock should not be poisoned");
        cache.clear();
    }

    /// Update configuration
    pub fn update_config(&mut self, config: PersonalityTransferConfig) {
        self.config = config;
    }
}

/// Per-signal prosodic measurements produced by [`analyze_prosody_frames`].
///
/// All durations are in seconds; F0 values are in Hz and come only from frames
/// classified as voiced.
struct ProsodyAnalysis {
    /// F0 estimate (Hz) for each voiced frame.
    voiced_f0: Vec<f32>,
    /// RMS amplitude for each voiced frame.
    voiced_rms: Vec<f32>,
    /// Duration (s) of each maximal run of consecutive voiced frames.
    voiced_run_secs: Vec<f32>,
    /// Duration (s) of each maximal run of consecutive unvoiced (pause) frames.
    pause_run_secs: Vec<f32>,
    /// Total voiced time (s) in the signal.
    voiced_secs: f32,
}

/// Lowest and highest fundamental frequencies (Hz) considered for speech.
const F0_MIN_HZ: f32 = 50.0;
const F0_MAX_HZ: f32 = 400.0;

/// Analyse a single PCM signal into per-frame prosodic measurements.
///
/// The signal is split into ≈25 ms frames with 50% overlap. For each frame:
///
/// * **Energy-based VAD**: the short-time RMS is compared against an adaptive
///   threshold (a fraction of the signal's peak frame RMS plus a small floor);
///   frames above it are *voiced*, the rest are pauses.
/// * **Autocorrelation pitch**: for voiced frames the normalised
///   autocorrelation is searched over the lag range corresponding to
///   `F0_MIN_HZ..=F0_MAX_HZ`; the strongest peak gives F0. Frames whose best
///   peak is too weak (effectively unvoiced/aperiodic) contribute no F0.
///
/// Returns runs of voiced/unvoiced frames (for rate and pause statistics) and
/// per-voiced-frame F0/RMS. The routine is self-contained and uses no RNG.
fn analyze_prosody_frames(audio: &[f32], sample_rate: u32) -> ProsodyAnalysis {
    let mut out = ProsodyAnalysis {
        voiced_f0: Vec::new(),
        voiced_rms: Vec::new(),
        voiced_run_secs: Vec::new(),
        pause_run_secs: Vec::new(),
        voiced_secs: 0.0,
    };
    if audio.is_empty() || sample_rate == 0 {
        return out;
    }

    let sr = sample_rate as f32;
    // ≈25 ms frame, 50% hop, with sane lower bounds for very low sample rates.
    let frame_len = ((sr * 0.025) as usize).max(256);
    let hop = (frame_len / 2).max(1);
    let frame_secs = hop as f32 / sr;

    // First pass: per-frame RMS to derive an adaptive voicing threshold.
    let mut frame_rms: Vec<f32> = Vec::new();
    let mut start = 0usize;
    while start < audio.len() {
        let end = (start + frame_len).min(audio.len());
        let frame = &audio[start..end];
        let energy: f32 = frame.iter().map(|x| x * x).sum();
        let rms = (energy / frame.len().max(1) as f32).sqrt();
        frame_rms.push(rms);
        if end >= audio.len() {
            break;
        }
        start += hop;
    }
    if frame_rms.is_empty() {
        return out;
    }

    let peak_rms = frame_rms.iter().copied().fold(0.0f32, f32::max);
    let mean_rms = frame_rms.iter().sum::<f32>() / frame_rms.len() as f32;
    // Threshold: above background but below typical speech energy. Combine a
    // fraction of the peak with a fraction of the mean and a tiny absolute
    // floor so a fully silent signal yields no voiced frames.
    let threshold = (0.15 * peak_rms).max(0.5 * mean_rms).max(1e-4);

    // Second pass: classify frames, estimate pitch, and accumulate runs.
    let mut cur_voiced_run = 0usize;
    let mut cur_pause_run = 0usize;

    start = 0;
    for &rms in &frame_rms {
        let end = (start + frame_len).min(audio.len());
        let frame = &audio[start..end];
        let is_voiced = rms >= threshold;

        if is_voiced {
            if cur_pause_run > 0 {
                out.pause_run_secs.push(cur_pause_run as f32 * frame_secs);
                cur_pause_run = 0;
            }
            cur_voiced_run += 1;
            out.voiced_secs += frame_secs;
            out.voiced_rms.push(rms);

            if let Some(f0) = estimate_frame_f0(frame, sample_rate) {
                out.voiced_f0.push(f0);
            }
        } else {
            if cur_voiced_run > 0 {
                out.voiced_run_secs.push(cur_voiced_run as f32 * frame_secs);
                cur_voiced_run = 0;
            }
            cur_pause_run += 1;
        }

        if end >= audio.len() {
            break;
        }
        start += hop;
    }
    // Flush trailing runs.
    if cur_voiced_run > 0 {
        out.voiced_run_secs.push(cur_voiced_run as f32 * frame_secs);
    }
    if cur_pause_run > 0 {
        out.pause_run_secs.push(cur_pause_run as f32 * frame_secs);
    }

    out
}

/// Estimate the fundamental frequency (Hz) of one frame via normalised
/// autocorrelation, or `None` if the frame is not convincingly periodic.
///
/// The autocorrelation is evaluated over lags `min_lag..=max_lag` mapped from
/// the [`F0_MIN_HZ`]–[`F0_MAX_HZ`] band. The lag of the highest peak is refined
/// with parabolic interpolation for sub-sample precision. A peak weaker than a
/// fixed fraction of the zero-lag energy is treated as unvoiced.
fn estimate_frame_f0(frame: &[f32], sample_rate: u32) -> Option<f32> {
    let sr = sample_rate as f32;
    let min_lag = (sr / F0_MAX_HZ).floor() as usize;
    let max_lag = (sr / F0_MIN_HZ).ceil() as usize;
    if frame.len() <= max_lag + 1 || min_lag < 1 {
        return None;
    }

    // Remove DC so silence/offset frames do not masquerade as periodic.
    let mean = frame.iter().sum::<f32>() / frame.len() as f32;
    let centred: Vec<f32> = frame.iter().map(|&x| x - mean).collect();

    let energy: f32 = centred.iter().map(|x| x * x).sum();
    if energy <= 1e-9 {
        return None;
    }

    let autocorr = |lag: usize| -> f32 {
        let mut acc = 0.0f32;
        for i in 0..(centred.len() - lag) {
            acc += centred[i] * centred[i + lag];
        }
        acc
    };

    let mut best_lag = 0usize;
    let mut best_val = 0.0f32;
    for lag in min_lag..=max_lag.min(centred.len() - 1) {
        let val = autocorr(lag);
        if val > best_val {
            best_val = val;
            best_lag = lag;
        }
    }

    // Normalised peak strength; require clear periodicity to call it voiced.
    let normalised = best_val / energy;
    if best_lag == 0 || normalised < 0.3 {
        return None;
    }

    // Parabolic interpolation around the integer peak for sub-sample accuracy.
    let refined_lag = if best_lag > min_lag && best_lag < max_lag {
        let a = autocorr(best_lag - 1);
        let b = best_val;
        let c = autocorr(best_lag + 1);
        let denom = a - 2.0 * b + c;
        if denom.abs() > 1e-9 {
            best_lag as f32 + 0.5 * (a - c) / denom
        } else {
            best_lag as f32
        }
    } else {
        best_lag as f32
    };

    if refined_lag <= 0.0 {
        return None;
    }
    let f0 = sr / refined_lag;
    if (F0_MIN_HZ..=F0_MAX_HZ).contains(&f0) {
        Some(f0)
    } else {
        None
    }
}

/// Coefficient of variation (std / mean) of a set of samples, or `None` if the
/// mean is non-positive or there are too few samples to be meaningful.
fn coefficient_of_variation(values: &[f32]) -> Option<f32> {
    if values.len() < 2 {
        return None;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    if mean <= 0.0 {
        return None;
    }
    let var = values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / values.len() as f32;
    Some(var.sqrt() / mean)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    fn create_test_speaker_profile() -> SpeakerProfile {
        use crate::types::SpeakerCharacteristics;

        SpeakerProfile {
            id: "test_speaker".to_string(),
            name: "Test Speaker".to_string(),
            characteristics: SpeakerCharacteristics::default(),
            samples: vec![
                VoiceSample {
                    id: "sample1".to_string(),
                    audio: vec![0.0f32; 1000],
                    sample_rate: 22050,
                    duration: 5.0,
                    transcript: Some("Hello world".to_string()),
                    language: Some("en".to_string()),
                    quality_score: Some(0.8),
                    metadata: HashMap::new(),
                    timestamp: SystemTime::now(),
                },
                VoiceSample {
                    id: "sample2".to_string(),
                    audio: vec![0.0f32; 2000],
                    sample_rate: 22050,
                    duration: 10.0,
                    transcript: Some("This is a test".to_string()),
                    language: Some("en".to_string()),
                    quality_score: Some(0.9),
                    metadata: HashMap::new(),
                    timestamp: SystemTime::now(),
                },
            ],
            embedding: None,
            languages: vec!["en".to_string()],
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_personality_traits_default() {
        let traits = PersonalityTraits::default();
        assert_eq!(traits.openness, 0.5);
        assert_eq!(traits.conscientiousness, 0.5);
        assert_eq!(traits.extraversion, 0.5);
        assert_eq!(traits.agreeableness, 0.5);
        assert_eq!(traits.neuroticism, 0.5);
    }

    #[test]
    fn test_speaking_patterns_default() {
        let patterns = SpeakingPatterns::default();
        assert_eq!(patterns.speaking_rate_wpm, 150.0);
        assert_eq!(patterns.average_f0_hz, 120.0);
        assert!(patterns.energy_level > 0.0);
        assert!(patterns.articulation_clarity > 0.0);
    }

    #[test]
    fn test_personality_transfer_config_default() {
        let config = PersonalityTransferConfig::default();
        assert_eq!(config.transfer_strength, 0.7);
        assert!(config.transfer_components.transfer_traits);
        assert!(config.enable_prosodic_transfer);
        assert_eq!(config.confidence_threshold, 0.6);
    }

    #[tokio::test]
    async fn test_personality_transfer_engine_creation() {
        let config = PersonalityTransferConfig::default();
        let engine = PersonalityTransferEngine::new(config);

        let stats = engine.get_statistics();
        assert_eq!(stats.operations_completed, 0);
        assert_eq!(stats.profiles_analyzed, 0);
    }

    #[tokio::test]
    async fn test_personality_analysis() {
        let config = PersonalityTransferConfig::default();
        let engine = PersonalityTransferEngine::new(config);
        let speaker = create_test_speaker_profile();

        let result = engine.analyze_personality(&speaker).await;
        assert!(result.is_ok());

        let profile = result.unwrap();
        assert_eq!(profile.speaker_id, "test_speaker");
        assert!(profile.traits.extraversion >= 0.0 && profile.traits.extraversion <= 1.0);
        assert!(profile.confidence_scores.contains_key("personality_traits"));
        assert_eq!(profile.analysis_metadata.samples_analyzed, 2);
    }

    #[tokio::test]
    async fn test_transfer_model_creation() {
        let config = PersonalityTransferConfig::default();
        let engine = PersonalityTransferEngine::new(config);
        let speaker1 = create_test_speaker_profile();
        let speaker2 = {
            let mut s = create_test_speaker_profile();
            s.id = "test_speaker_2".to_string();
            s
        };

        let profile1 = engine.analyze_personality(&speaker1).await.unwrap();
        let profile2 = engine.analyze_personality(&speaker2).await.unwrap();

        let model_id = engine.create_transfer_model(&profile1, &profile2).await;
        assert!(model_id.is_ok());

        let model_id = model_id.unwrap();
        assert!(model_id.contains("transfer_"));
        assert!(model_id.contains("test_speaker"));
        assert!(model_id.contains("test_speaker_2"));
    }

    #[tokio::test]
    async fn test_personality_transfer_application() {
        let config = PersonalityTransferConfig::default();
        let engine = PersonalityTransferEngine::new(config);
        let speaker1 = create_test_speaker_profile();
        let speaker2 = {
            let mut s = create_test_speaker_profile();
            s.id = "test_speaker_2".to_string();
            s
        };

        let profile1 = engine.analyze_personality(&speaker1).await.unwrap();
        let profile2 = engine.analyze_personality(&speaker2).await.unwrap();

        let mut request = VoiceCloneRequest {
            id: "transfer_test".to_string(),
            speaker_data: crate::types::SpeakerData {
                profile: speaker1,
                reference_samples: vec![],
                target_text: Some("Test transfer".to_string()),
                target_language: Some("en".to_string()),
                context: HashMap::new(),
            },
            method: crate::types::CloningMethod::FewShot,
            text: "Test transfer".to_string(),
            language: Some("en".to_string()),
            quality_level: 0.8,
            quality_tradeoff: 0.5,
            parameters: HashMap::new(),
            timestamp: SystemTime::now(),
        };

        let result = engine
            .apply_personality_transfer(&mut request, &profile1, &profile2)
            .await;
        assert!(result.is_ok());

        // Check that personality parameters were added
        assert!(!request.parameters.is_empty());
        let has_personality_params = request
            .parameters
            .keys()
            .any(|k| k.starts_with("personality_"));
        assert!(has_personality_params);
    }

    #[test]
    fn test_confidence_score_calculation() {
        let config = PersonalityTransferConfig::default();
        let engine = PersonalityTransferEngine::new(config);
        let speaker = create_test_speaker_profile();

        let scores = engine.calculate_confidence_scores(&speaker.samples);
        assert!(scores.contains_key("personality_traits"));
        assert!(scores.contains_key("speaking_patterns"));
        assert!(scores.contains_key("conversational_style"));

        for score in scores.values() {
            assert!(*score >= 0.0 && *score <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_speaking_patterns_analysis() {
        let config = PersonalityTransferConfig::default();
        let engine = PersonalityTransferEngine::new(config);
        let speaker = create_test_speaker_profile();

        let patterns = engine
            .analyze_speaking_patterns(&speaker.samples)
            .await
            .unwrap();
        assert!(patterns.speaking_rate_wpm > 0.0);
        assert!(patterns.average_f0_hz > 0.0);
        assert!(patterns.energy_level >= 0.0 && patterns.energy_level <= 1.0);
        assert!(patterns.articulation_clarity >= 0.0 && patterns.articulation_clarity <= 1.0);
    }

    #[tokio::test]
    async fn test_conversational_style_analysis() {
        let config = PersonalityTransferConfig::default();
        let engine = PersonalityTransferEngine::new(config);
        let speaker = create_test_speaker_profile();

        let style = engine
            .analyze_conversational_style(&speaker.samples)
            .await
            .unwrap();
        assert!(style.interruption_tendency >= 0.0 && style.interruption_tendency <= 1.0);
        assert!(style.response_latency > 0.0);
        assert!(style.emotional_expressiveness >= 0.0 && style.emotional_expressiveness <= 1.0);
    }

    #[test]
    fn test_personality_profile_storage_and_retrieval() {
        let config = PersonalityTransferConfig::default();
        let engine = PersonalityTransferEngine::new(config);

        // Initially no profiles
        assert!(engine.get_personality_profile("nonexistent").is_none());
        assert!(engine.list_personality_profiles().is_empty());

        // Store a profile manually for testing
        let profile = PersonalityProfile {
            speaker_id: "stored_speaker".to_string(),
            traits: PersonalityTraits::default(),
            speaking_patterns: SpeakingPatterns::default(),
            conversational_style: ConversationalStyle::default(),
            linguistic_preferences: LinguisticPreferences::default(),
            confidence_scores: HashMap::new(),
            analysis_metadata: AnalysisMetadata {
                analyzed_at: SystemTime::now(),
                audio_duration_seconds: 30.0,
                samples_analyzed: 3,
                analysis_method: "test".to_string(),
                model_version: "test".to_string(),
                analysis_quality: 0.9,
            },
        };

        {
            let mut db = engine
                .personality_database
                .write()
                .expect("lock should not be poisoned");
            db.insert("stored_speaker".to_string(), profile.clone());
        }

        // Retrieve the profile
        let retrieved = engine.get_personality_profile("stored_speaker");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().speaker_id, "stored_speaker");

        // List profiles
        let profiles = engine.list_personality_profiles();
        assert_eq!(profiles.len(), 1);
        assert!(profiles.contains(&"stored_speaker".to_string()));
    }

    #[tokio::test]
    async fn test_statistics_tracking() {
        let config = PersonalityTransferConfig::default();
        let engine = PersonalityTransferEngine::new(config);
        let speaker = create_test_speaker_profile();

        // Initially no stats
        let initial_stats = engine.get_statistics();
        assert_eq!(initial_stats.profiles_analyzed, 0);

        // Analyze a personality
        let _profile = engine.analyze_personality(&speaker).await.unwrap();

        // Check stats were updated
        let updated_stats = engine.get_statistics();
        assert_eq!(updated_stats.profiles_analyzed, 1);
        assert!(updated_stats.avg_processing_time_ms >= 0.0);
    }
}
