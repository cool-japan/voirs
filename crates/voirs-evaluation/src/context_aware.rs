// Copyright (c) 2025 VoiRS Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Context-aware evaluation system for speech synthesis quality assessment.
//!
//! This module provides comprehensive context-aware evaluation capabilities that adjust
//! quality assessment based on various contextual factors including speaking style,
//! emotional context, formality level, domain, environment, and speaker characteristics.
//!
//! # Features
//!
//! - **Speaking Style Context**: Evaluation adapted for reading, conversation, presentation, etc.
//! - **Emotional Context**: Context-sensitive assessment for different emotional states
//! - **Formality Level**: Adjustment for formal vs informal speech quality expectations
//! - **Domain Context**: Domain-specific evaluation (news, podcast, audiobook, etc.)
//! - **Environmental Context**: Quality assessment considering background conditions
//! - **Speaker Context**: Personalized evaluation based on speaker characteristics
//! - **Temporal Context**: Time-aware evaluation with pacing and duration considerations
//! - **Adaptive Weights**: Automatic metric weighting based on context
//!
//! # Example
//!
//! ```no_run
//! use voirs_evaluation::context_aware::*;
//! use voirs_sdk::AudioBuffer;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create context-aware evaluator
//! let config = ContextAwareConfig::default();
//! let evaluator = ContextAwareEvaluator::new(config)?;
//!
//! // Define evaluation context
//! let context = EvaluationContext::builder()
//!     .speaking_style(SpeakingStyle::Conversation)
//!     .emotional_context(EmotionalContext::Neutral)
//!     .formality_level(FormalityLevel::Informal)
//!     .domain(Domain::Podcast)
//!     .build();
//!
//! // Create sample audio
//! let sample_rate = 16000;
//! let duration = 3.0;
//! let samples: Vec<f32> = (0..(sample_rate as f32 * duration) as usize)
//!     .map(|i| (i as f32 * 0.01).sin() * 0.1)
//!     .collect();
//! let audio = AudioBuffer::new(samples, sample_rate, 1);
//!
//! // Perform context-aware evaluation
//! let result = evaluator.evaluate(&audio, &audio, &context)?;
//!
//! println!("Overall Quality: {:.2}", result.overall_quality);
//! println!("Context Score: {:.2}", result.context_appropriateness);
//! # Ok(())
//! # }
//! ```

use crate::EvaluationError;
use scirs2_core::numeric::Float;
use scirs2_core::random::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use voirs_sdk::AudioBuffer;

/// Speaking style context categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpeakingStyle {
    /// Reading text (audiobooks, news articles)
    Reading,
    /// Natural conversation (dialogues, interviews)
    Conversation,
    /// Formal presentation (lectures, speeches)
    Presentation,
    /// Narration (storytelling, documentaries)
    Narration,
    /// Customer service interaction
    CustomerService,
    /// Broadcasting (radio, TV)
    Broadcasting,
    /// Command and control (voice assistants)
    Command,
}

/// Emotional context of the speech
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmotionalContext {
    /// Neutral emotional state
    Neutral,
    /// Happy/joyful state
    Happy,
    /// Sad/melancholic state
    Sad,
    /// Angry/frustrated state
    Angry,
    /// Excited/enthusiastic state
    Excited,
    /// Calm/relaxed state
    Calm,
    /// Fearful/anxious state
    Fearful,
    /// Surprised state
    Surprised,
}

/// Formality level of the speech
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FormalityLevel {
    /// Highly formal (official speeches, legal)
    Formal,
    /// Semi-formal (business, presentations)
    SemiFormal,
    /// Informal (casual conversation)
    Informal,
    /// Very casual (friends, family)
    Casual,
}

/// Domain context for evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Domain {
    /// News broadcasting
    News,
    /// Podcast content
    Podcast,
    /// Audiobook narration
    Audiobook,
    /// Customer support
    CustomerSupport,
    /// Voice assistant
    VoiceAssistant,
    /// Gaming
    Gaming,
    /// Education
    Education,
    /// Entertainment
    Entertainment,
}

/// Environmental context
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnvironmentalContext {
    /// Quiet indoor environment
    QuietIndoor,
    /// Noisy indoor environment
    NoisyIndoor,
    /// Outdoor environment
    Outdoor,
    /// Studio recording quality
    Studio,
    /// Vehicle environment
    Vehicle,
}

/// Speaker characteristic context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerContext {
    /// Estimated age group
    pub age_group: Option<String>,
    /// Estimated gender
    pub gender: Option<String>,
    /// Detected accent
    pub accent: Option<String>,
    /// Speaking rate (words per minute)
    pub speaking_rate: Option<f32>,
    /// Voice characteristics
    pub voice_characteristics: HashMap<String, f32>,
}

impl Default for SpeakerContext {
    fn default() -> Self {
        Self {
            age_group: None,
            gender: None,
            accent: None,
            speaking_rate: None,
            voice_characteristics: HashMap::new(),
        }
    }
}

/// Temporal context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalContext {
    /// Duration of the speech in seconds
    pub duration: f32,
    /// Average pacing (pauses per minute)
    pub pacing: Option<f32>,
    /// Time of day context (for circadian rhythm consideration)
    pub time_of_day: Option<String>,
    /// Timestamp
    pub timestamp: Option<i64>,
}

impl Default for TemporalContext {
    fn default() -> Self {
        Self {
            duration: 0.0,
            pacing: None,
            time_of_day: None,
            timestamp: None,
        }
    }
}

/// Complete evaluation context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationContext {
    /// Speaking style
    pub speaking_style: SpeakingStyle,
    /// Emotional context
    pub emotional_context: EmotionalContext,
    /// Formality level
    pub formality_level: FormalityLevel,
    /// Domain
    pub domain: Domain,
    /// Environmental context
    pub environmental_context: EnvironmentalContext,
    /// Speaker context
    pub speaker_context: SpeakerContext,
    /// Temporal context
    pub temporal_context: TemporalContext,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl EvaluationContext {
    /// Creates a new builder for EvaluationContext
    pub fn builder() -> EvaluationContextBuilder {
        EvaluationContextBuilder::default()
    }
}

impl Default for EvaluationContext {
    fn default() -> Self {
        Self {
            speaking_style: SpeakingStyle::Reading,
            emotional_context: EmotionalContext::Neutral,
            formality_level: FormalityLevel::SemiFormal,
            domain: Domain::Audiobook,
            environmental_context: EnvironmentalContext::Studio,
            speaker_context: SpeakerContext::default(),
            temporal_context: TemporalContext::default(),
            metadata: HashMap::new(),
        }
    }
}

/// Builder for EvaluationContext
#[derive(Debug, Default)]
pub struct EvaluationContextBuilder {
    speaking_style: Option<SpeakingStyle>,
    emotional_context: Option<EmotionalContext>,
    formality_level: Option<FormalityLevel>,
    domain: Option<Domain>,
    environmental_context: Option<EnvironmentalContext>,
    speaker_context: Option<SpeakerContext>,
    temporal_context: Option<TemporalContext>,
    metadata: HashMap<String, String>,
}

impl EvaluationContextBuilder {
    /// Sets the speaking style
    pub fn speaking_style(mut self, style: SpeakingStyle) -> Self {
        self.speaking_style = Some(style);
        self
    }

    /// Sets the emotional context
    pub fn emotional_context(mut self, emotion: EmotionalContext) -> Self {
        self.emotional_context = Some(emotion);
        self
    }

    /// Sets the formality level
    pub fn formality_level(mut self, level: FormalityLevel) -> Self {
        self.formality_level = Some(level);
        self
    }

    /// Sets the domain
    pub fn domain(mut self, domain: Domain) -> Self {
        self.domain = Some(domain);
        self
    }

    /// Sets the environmental context
    pub fn environmental_context(mut self, env: EnvironmentalContext) -> Self {
        self.environmental_context = Some(env);
        self
    }

    /// Sets the speaker context
    pub fn speaker_context(mut self, speaker: SpeakerContext) -> Self {
        self.speaker_context = Some(speaker);
        self
    }

    /// Sets the temporal context
    pub fn temporal_context(mut self, temporal: TemporalContext) -> Self {
        self.temporal_context = Some(temporal);
        self
    }

    /// Adds metadata
    pub fn metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Builds the EvaluationContext
    pub fn build(self) -> EvaluationContext {
        EvaluationContext {
            speaking_style: self.speaking_style.unwrap_or(SpeakingStyle::Reading),
            emotional_context: self.emotional_context.unwrap_or(EmotionalContext::Neutral),
            formality_level: self.formality_level.unwrap_or(FormalityLevel::SemiFormal),
            domain: self.domain.unwrap_or(Domain::Audiobook),
            environmental_context: self
                .environmental_context
                .unwrap_or(EnvironmentalContext::Studio),
            speaker_context: self.speaker_context.unwrap_or_default(),
            temporal_context: self.temporal_context.unwrap_or_default(),
            metadata: self.metadata,
        }
    }
}

/// Context-specific metric weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMetricWeights {
    /// Weight for intelligibility (0.0-1.0)
    pub intelligibility_weight: f32,
    /// Weight for naturalness (0.0-1.0)
    pub naturalness_weight: f32,
    /// Weight for prosody (0.0-1.0)
    pub prosody_weight: f32,
    /// Weight for emotion accuracy (0.0-1.0)
    pub emotion_weight: f32,
    /// Weight for formality match (0.0-1.0)
    pub formality_weight: f32,
    /// Weight for speaking rate (0.0-1.0)
    pub speaking_rate_weight: f32,
}

impl Default for ContextMetricWeights {
    fn default() -> Self {
        Self {
            intelligibility_weight: 0.3,
            naturalness_weight: 0.3,
            prosody_weight: 0.2,
            emotion_weight: 0.1,
            formality_weight: 0.05,
            speaking_rate_weight: 0.05,
        }
    }
}

/// Context-aware evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextAwareResult {
    /// Overall quality score (0.0-1.0)
    pub overall_quality: f32,
    /// Context appropriateness score (0.0-1.0)
    pub context_appropriateness: f32,
    /// Intelligibility score for the context (0.0-1.0)
    pub intelligibility_score: f32,
    /// Naturalness score for the context (0.0-1.0)
    pub naturalness_score: f32,
    /// Prosody quality score (0.0-1.0)
    pub prosody_score: f32,
    /// Emotion accuracy score (0.0-1.0)
    pub emotion_score: f32,
    /// Formality match score (0.0-1.0)
    pub formality_score: f32,
    /// Speaking rate appropriateness (0.0-1.0)
    pub speaking_rate_score: f32,
    /// Context-specific metric weights used
    pub weights: ContextMetricWeights,
    /// Detailed breakdown per context dimension
    pub context_breakdown: HashMap<String, f32>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Configuration for context-aware evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextAwareConfig {
    /// Enable automatic weight adaptation
    pub auto_adapt_weights: bool,
    /// Minimum quality threshold (0.0-1.0)
    pub quality_threshold: f32,
    /// Enable detailed context analysis
    pub detailed_analysis: bool,
    /// Enable recommendation generation
    pub generate_recommendations: bool,
}

impl Default for ContextAwareConfig {
    fn default() -> Self {
        Self {
            auto_adapt_weights: true,
            quality_threshold: 0.7,
            detailed_analysis: true,
            generate_recommendations: true,
        }
    }
}

/// Context-aware evaluator
pub struct ContextAwareEvaluator {
    config: ContextAwareConfig,
}

impl ContextAwareEvaluator {
    /// Creates a new context-aware evaluator
    pub fn new(config: ContextAwareConfig) -> Result<Self, EvaluationError> {
        Ok(Self { config })
    }

    /// Evaluates audio with context awareness
    pub fn evaluate(
        &self,
        reference: &AudioBuffer,
        synthesized: &AudioBuffer,
        context: &EvaluationContext,
    ) -> Result<ContextAwareResult, EvaluationError> {
        // Validate inputs
        if reference.sample_rate() != synthesized.sample_rate() {
            return Err(EvaluationError::AudioProcessingError {
                message: "Sample rates must match".to_string(),
                source: None,
            });
        }

        // Get context-adapted weights
        let weights = if self.config.auto_adapt_weights {
            self.adapt_weights(context)
        } else {
            ContextMetricWeights::default()
        };

        // Evaluate individual metrics
        let intelligibility_score =
            self.evaluate_intelligibility(reference, synthesized, context)?;
        let naturalness_score = self.evaluate_naturalness(reference, synthesized, context)?;
        let prosody_score = self.evaluate_prosody(reference, synthesized, context)?;
        let emotion_score = self.evaluate_emotion(reference, synthesized, context)?;
        let formality_score = self.evaluate_formality(reference, synthesized, context)?;
        let speaking_rate_score = self.evaluate_speaking_rate(reference, synthesized, context)?;

        // Calculate overall quality using weighted combination
        let overall_quality = intelligibility_score * weights.intelligibility_weight
            + naturalness_score * weights.naturalness_weight
            + prosody_score * weights.prosody_weight
            + emotion_score * weights.emotion_weight
            + formality_score * weights.formality_weight
            + speaking_rate_score * weights.speaking_rate_weight;

        // Calculate context appropriateness
        let context_appropriateness = self.calculate_context_appropriateness(
            intelligibility_score,
            naturalness_score,
            prosody_score,
            emotion_score,
            formality_score,
            speaking_rate_score,
            context,
        );

        // Generate context breakdown
        let mut context_breakdown = HashMap::new();
        context_breakdown.insert(
            "speaking_style".to_string(),
            self.evaluate_speaking_style_match(context),
        );
        context_breakdown.insert("emotional_match".to_string(), emotion_score);
        context_breakdown.insert("formality_match".to_string(), formality_score);
        context_breakdown.insert(
            "domain_appropriateness".to_string(),
            self.evaluate_domain_appropriateness(context),
        );
        context_breakdown.insert(
            "environmental_quality".to_string(),
            self.evaluate_environmental_quality(synthesized, context)?,
        );

        // Generate recommendations
        let recommendations = if self.config.generate_recommendations {
            self.generate_recommendations(
                &weights,
                intelligibility_score,
                naturalness_score,
                prosody_score,
                emotion_score,
                formality_score,
                speaking_rate_score,
                context,
            )
        } else {
            Vec::new()
        };

        Ok(ContextAwareResult {
            overall_quality,
            context_appropriateness,
            intelligibility_score,
            naturalness_score,
            prosody_score,
            emotion_score,
            formality_score,
            speaking_rate_score,
            weights,
            context_breakdown,
            recommendations,
        })
    }

    /// Adapts metric weights based on context
    fn adapt_weights(&self, context: &EvaluationContext) -> ContextMetricWeights {
        let mut weights = ContextMetricWeights::default();

        // Adapt based on speaking style
        match context.speaking_style {
            SpeakingStyle::Reading | SpeakingStyle::Broadcasting => {
                weights.intelligibility_weight = 0.4;
                weights.prosody_weight = 0.25;
                weights.naturalness_weight = 0.25;
            }
            SpeakingStyle::Conversation | SpeakingStyle::CustomerService => {
                weights.naturalness_weight = 0.4;
                weights.emotion_weight = 0.2;
                weights.intelligibility_weight = 0.25;
            }
            SpeakingStyle::Presentation => {
                weights.prosody_weight = 0.35;
                weights.formality_weight = 0.15;
                weights.intelligibility_weight = 0.3;
            }
            SpeakingStyle::Narration => {
                weights.emotion_weight = 0.25;
                weights.prosody_weight = 0.3;
                weights.naturalness_weight = 0.25;
            }
            SpeakingStyle::Command => {
                weights.intelligibility_weight = 0.5;
                weights.naturalness_weight = 0.2;
                weights.speaking_rate_weight = 0.15;
            }
        }

        // Adjust based on formality level
        match context.formality_level {
            FormalityLevel::Formal => {
                weights.formality_weight += 0.1;
                weights.prosody_weight += 0.05;
            }
            FormalityLevel::Casual => {
                weights.naturalness_weight += 0.1;
                weights.emotion_weight += 0.05;
            }
            _ => {}
        }

        // Normalize weights to sum to 1.0
        let total = weights.intelligibility_weight
            + weights.naturalness_weight
            + weights.prosody_weight
            + weights.emotion_weight
            + weights.formality_weight
            + weights.speaking_rate_weight;

        if total > 0.0 {
            weights.intelligibility_weight /= total;
            weights.naturalness_weight /= total;
            weights.prosody_weight /= total;
            weights.emotion_weight /= total;
            weights.formality_weight /= total;
            weights.speaking_rate_weight /= total;
        }

        weights
    }

    /// Evaluates intelligibility in context
    fn evaluate_intelligibility(
        &self,
        reference: &AudioBuffer,
        synthesized: &AudioBuffer,
        _context: &EvaluationContext,
    ) -> Result<f32, EvaluationError> {
        // Calculate signal correlation as a proxy for intelligibility
        let ref_samples = reference.samples();
        let syn_samples = synthesized.samples();

        let min_len = ref_samples.len().min(syn_samples.len());
        if min_len == 0 {
            return Ok(0.0);
        }

        let correlation =
            self.calculate_correlation(&ref_samples[..min_len], &syn_samples[..min_len]);
        Ok((correlation + 1.0) / 2.0) // Map from [-1, 1] to [0, 1]
    }

    /// Evaluates naturalness in context
    fn evaluate_naturalness(
        &self,
        reference: &AudioBuffer,
        synthesized: &AudioBuffer,
        _context: &EvaluationContext,
    ) -> Result<f32, EvaluationError> {
        // Calculate spectral similarity as a proxy for naturalness
        let score = self.calculate_spectral_similarity(reference, synthesized)?;
        Ok(score.clamp(0.0, 1.0))
    }

    /// Evaluates prosody quality in context
    fn evaluate_prosody(
        &self,
        reference: &AudioBuffer,
        synthesized: &AudioBuffer,
        _context: &EvaluationContext,
    ) -> Result<f32, EvaluationError> {
        // Calculate pitch and energy variation similarity
        let score = self.calculate_prosody_similarity(reference, synthesized)?;
        Ok(score.clamp(0.0, 1.0))
    }

    /// Evaluates emotion accuracy in context
    fn evaluate_emotion(
        &self,
        _reference: &AudioBuffer,
        synthesized: &AudioBuffer,
        context: &EvaluationContext,
    ) -> Result<f32, EvaluationError> {
        // Analyze emotional characteristics in the synthesized speech
        let detected_emotion = self.detect_emotion(synthesized)?;

        // Compare with expected emotion from context
        let target_emotion = context.emotional_context;
        let match_score = self.calculate_emotion_match(detected_emotion, target_emotion);

        Ok(match_score)
    }

    /// Evaluates formality level match
    fn evaluate_formality(
        &self,
        _reference: &AudioBuffer,
        synthesized: &AudioBuffer,
        context: &EvaluationContext,
    ) -> Result<f32, EvaluationError> {
        // Analyze formality indicators (speaking rate, pitch variation, etc.)
        let detected_formality = self.detect_formality(synthesized)?;

        // Compare with expected formality
        let match_score = match (detected_formality, context.formality_level) {
            (f1, f2) if f1 == f2 => 1.0,
            (FormalityLevel::Formal, FormalityLevel::SemiFormal)
            | (FormalityLevel::SemiFormal, FormalityLevel::Formal) => 0.8,
            (FormalityLevel::SemiFormal, FormalityLevel::Informal)
            | (FormalityLevel::Informal, FormalityLevel::SemiFormal) => 0.7,
            (FormalityLevel::Informal, FormalityLevel::Casual)
            | (FormalityLevel::Casual, FormalityLevel::Informal) => 0.8,
            _ => 0.5,
        };

        Ok(match_score)
    }

    /// Evaluates speaking rate appropriateness
    fn evaluate_speaking_rate(
        &self,
        reference: &AudioBuffer,
        synthesized: &AudioBuffer,
        context: &EvaluationContext,
    ) -> Result<f32, EvaluationError> {
        let ref_rate = self.estimate_speaking_rate(reference)?;
        let syn_rate = self.estimate_speaking_rate(synthesized)?;

        // Get expected rate range for context
        let (min_rate, max_rate) = self.get_expected_rate_range(context);

        // Calculate how well both rates fit the expected range
        let ref_fit = if ref_rate >= min_rate && ref_rate <= max_rate {
            1.0
        } else {
            let distance = if ref_rate < min_rate {
                min_rate - ref_rate
            } else {
                ref_rate - max_rate
            };
            (1.0 - (distance / 100.0).min(1.0)).max(0.0)
        };

        let syn_fit = if syn_rate >= min_rate && syn_rate <= max_rate {
            1.0
        } else {
            let distance = if syn_rate < min_rate {
                min_rate - syn_rate
            } else {
                syn_rate - max_rate
            };
            (1.0 - (distance / 100.0).min(1.0)).max(0.0)
        };

        // Also consider similarity to reference
        let rate_diff = (ref_rate - syn_rate).abs();
        let similarity = (1.0 - (rate_diff / 100.0).min(1.0)).max(0.0);

        Ok((ref_fit * 0.3 + syn_fit * 0.4 + similarity * 0.3).clamp(0.0, 1.0))
    }

    /// Calculates context appropriateness score
    fn calculate_context_appropriateness(
        &self,
        intelligibility: f32,
        naturalness: f32,
        prosody: f32,
        emotion: f32,
        formality: f32,
        speaking_rate: f32,
        context: &EvaluationContext,
    ) -> f32 {
        let mut appropriateness = 0.0;
        let mut weight_sum = 0.0;

        // Speaking style appropriateness
        match context.speaking_style {
            SpeakingStyle::Reading | SpeakingStyle::Broadcasting => {
                appropriateness += intelligibility * 0.4 + prosody * 0.3;
                weight_sum += 0.7;
            }
            SpeakingStyle::Conversation => {
                appropriateness += naturalness * 0.4 + emotion * 0.3;
                weight_sum += 0.7;
            }
            SpeakingStyle::Presentation => {
                appropriateness += prosody * 0.3 + formality * 0.25 + intelligibility * 0.2;
                weight_sum += 0.75;
            }
            SpeakingStyle::Narration => {
                appropriateness += emotion * 0.3 + prosody * 0.25 + naturalness * 0.2;
                weight_sum += 0.75;
            }
            SpeakingStyle::Command | SpeakingStyle::CustomerService => {
                appropriateness += intelligibility * 0.4 + speaking_rate * 0.2;
                weight_sum += 0.6;
            }
        }

        // Normalize
        let result = if weight_sum > 0.0 {
            appropriateness / weight_sum
        } else {
            0.5
        };
        result.clamp(0.0, 1.0)
    }

    /// Evaluates speaking style match
    fn evaluate_speaking_style_match(&self, context: &EvaluationContext) -> f32 {
        // Placeholder: In production, this would analyze acoustic features
        // For now, return a score based on context consistency
        match context.speaking_style {
            SpeakingStyle::Reading
                if matches!(context.domain, Domain::Audiobook | Domain::News) =>
            {
                0.9
            }
            SpeakingStyle::Conversation
                if matches!(context.domain, Domain::Podcast | Domain::CustomerSupport) =>
            {
                0.9
            }
            SpeakingStyle::Presentation if matches!(context.domain, Domain::Education) => 0.9,
            SpeakingStyle::Broadcasting
                if matches!(context.domain, Domain::News | Domain::Entertainment) =>
            {
                0.9
            }
            _ => 0.7,
        }
    }

    /// Evaluates domain appropriateness
    fn evaluate_domain_appropriateness(&self, context: &EvaluationContext) -> f32 {
        // Evaluate how well the context fits the domain
        let style_domain_match = match (context.speaking_style, context.domain) {
            (SpeakingStyle::Reading, Domain::Audiobook | Domain::News) => 1.0,
            (SpeakingStyle::Conversation, Domain::Podcast | Domain::CustomerSupport) => 1.0,
            (SpeakingStyle::Command, Domain::VoiceAssistant) => 1.0,
            (SpeakingStyle::Presentation, Domain::Education) => 1.0,
            (SpeakingStyle::Narration, Domain::Entertainment | Domain::Education) => 1.0,
            _ => 0.7,
        };

        let formality_domain_match = match (context.formality_level, context.domain) {
            (FormalityLevel::Formal, Domain::News | Domain::Education) => 1.0,
            (FormalityLevel::Casual, Domain::Podcast | Domain::Entertainment) => 1.0,
            (FormalityLevel::SemiFormal, Domain::CustomerSupport | Domain::VoiceAssistant) => 1.0,
            _ => 0.8,
        };

        (style_domain_match + formality_domain_match) / 2.0
    }

    /// Evaluates environmental quality
    fn evaluate_environmental_quality(
        &self,
        audio: &AudioBuffer,
        context: &EvaluationContext,
    ) -> Result<f32, EvaluationError> {
        // Analyze noise level
        let noise_level = self.estimate_noise_level(audio)?;

        // Expected noise levels for different environments
        let expected_noise = match context.environmental_context {
            EnvironmentalContext::Studio => 0.01,
            EnvironmentalContext::QuietIndoor => 0.05,
            EnvironmentalContext::NoisyIndoor => 0.15,
            EnvironmentalContext::Outdoor => 0.25,
            EnvironmentalContext::Vehicle => 0.35,
        };

        // Calculate quality based on how close to expected noise
        let noise_diff = (noise_level - expected_noise).abs();
        let quality = (1.0 - (noise_diff * 5.0).min(1.0)).max(0.0);

        Ok(quality)
    }

    /// Generates improvement recommendations
    fn generate_recommendations(
        &self,
        weights: &ContextMetricWeights,
        intelligibility: f32,
        naturalness: f32,
        prosody: f32,
        emotion: f32,
        formality: f32,
        speaking_rate: f32,
        context: &EvaluationContext,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        let threshold = self.config.quality_threshold;

        if intelligibility < threshold && weights.intelligibility_weight > 0.2 {
            recommendations.push(format!(
                "Improve intelligibility (current: {:.2}, threshold: {:.2}). Consider clearer articulation.",
                intelligibility, threshold
            ));
        }

        if naturalness < threshold && weights.naturalness_weight > 0.2 {
            recommendations.push(format!(
                "Enhance naturalness (current: {:.2}, threshold: {:.2}). Consider more natural prosody patterns.",
                naturalness, threshold
            ));
        }

        if prosody < threshold && weights.prosody_weight > 0.15 {
            recommendations.push(format!(
                "Improve prosody (current: {:.2}, threshold: {:.2}). Adjust pitch and timing variations.",
                prosody, threshold
            ));
        }

        if emotion < threshold
            && matches!(
                context.emotional_context,
                EmotionalContext::Happy
                    | EmotionalContext::Excited
                    | EmotionalContext::Sad
                    | EmotionalContext::Angry
            )
        {
            recommendations.push(format!(
                "Enhance emotional expression for {:?} context (current: {:.2}, threshold: {:.2}).",
                context.emotional_context, emotion, threshold
            ));
        }

        if formality < threshold && weights.formality_weight > 0.05 {
            recommendations.push(format!(
                "Adjust formality to match {:?} level (current: {:.2}, threshold: {:.2}).",
                context.formality_level, formality, threshold
            ));
        }

        if speaking_rate < threshold {
            let (min_rate, max_rate) = self.get_expected_rate_range(context);
            recommendations.push(format!(
                "Adjust speaking rate to {:.0}-{:.0} words/min for {:?} context.",
                min_rate, max_rate, context.speaking_style
            ));
        }

        if recommendations.is_empty() {
            recommendations
                .push("Overall quality meets expectations for the given context.".to_string());
        }

        recommendations
    }

    // Helper methods

    fn calculate_correlation(&self, signal1: &[f32], signal2: &[f32]) -> f32 {
        let n = signal1.len().min(signal2.len());
        if n == 0 {
            return 0.0;
        }

        let mean1: f32 = signal1.iter().take(n).sum::<f32>() / n as f32;
        let mean2: f32 = signal2.iter().take(n).sum::<f32>() / n as f32;

        let mut numerator = 0.0;
        let mut denom1 = 0.0;
        let mut denom2 = 0.0;

        for i in 0..n {
            let diff1 = signal1[i] - mean1;
            let diff2 = signal2[i] - mean2;
            numerator += diff1 * diff2;
            denom1 += diff1 * diff1;
            denom2 += diff2 * diff2;
        }

        let denominator = (denom1 * denom2).sqrt();
        if denominator > 1e-10 {
            (numerator / denominator).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    fn calculate_spectral_similarity(
        &self,
        reference: &AudioBuffer,
        synthesized: &AudioBuffer,
    ) -> Result<f32, EvaluationError> {
        // Calculate RMS energy as a simple spectral similarity proxy
        let ref_rms = self.calculate_rms(reference.samples());
        let syn_rms = self.calculate_rms(synthesized.samples());

        let diff = (ref_rms - syn_rms).abs();
        let similarity = (1.0 - diff).max(0.0);

        Ok(similarity)
    }

    fn calculate_prosody_similarity(
        &self,
        reference: &AudioBuffer,
        synthesized: &AudioBuffer,
    ) -> Result<f32, EvaluationError> {
        // Calculate energy envelope correlation as prosody proxy
        let ref_envelope = self.calculate_energy_envelope(reference.samples());
        let syn_envelope = self.calculate_energy_envelope(synthesized.samples());

        let correlation = self.calculate_correlation(&ref_envelope, &syn_envelope);
        Ok((correlation + 1.0) / 2.0)
    }

    fn detect_emotion(&self, audio: &AudioBuffer) -> Result<EmotionalContext, EvaluationError> {
        // Placeholder: analyze energy and pitch variation to detect emotion
        let energy = self.calculate_rms(audio.samples());
        let variation = self.calculate_variation(audio.samples());

        // Simple heuristic emotion detection
        let emotion = if energy > 0.3 && variation > 0.2 {
            EmotionalContext::Excited
        } else if energy < 0.1 {
            EmotionalContext::Calm
        } else if variation > 0.25 {
            EmotionalContext::Happy
        } else if variation < 0.05 {
            EmotionalContext::Sad
        } else {
            EmotionalContext::Neutral
        };

        Ok(emotion)
    }

    fn calculate_emotion_match(
        &self,
        detected: EmotionalContext,
        expected: EmotionalContext,
    ) -> f32 {
        if detected == expected {
            1.0
        } else {
            // Partial credit for similar emotions
            match (detected, expected) {
                (EmotionalContext::Happy, EmotionalContext::Excited)
                | (EmotionalContext::Excited, EmotionalContext::Happy) => 0.8,
                (EmotionalContext::Calm, EmotionalContext::Neutral)
                | (EmotionalContext::Neutral, EmotionalContext::Calm) => 0.8,
                (EmotionalContext::Sad, EmotionalContext::Fearful)
                | (EmotionalContext::Fearful, EmotionalContext::Sad) => 0.7,
                _ => 0.5,
            }
        }
    }

    fn detect_formality(&self, audio: &AudioBuffer) -> Result<FormalityLevel, EvaluationError> {
        // Placeholder: analyze speaking rate and variation
        let rate = self.estimate_speaking_rate(audio)?;
        let variation = self.calculate_variation(audio.samples());

        // Simple heuristic formality detection
        let formality = if rate < 120.0 && variation < 0.15 {
            FormalityLevel::Formal
        } else if rate < 150.0 {
            FormalityLevel::SemiFormal
        } else if rate < 180.0 {
            FormalityLevel::Informal
        } else {
            FormalityLevel::Casual
        };

        Ok(formality)
    }

    fn estimate_speaking_rate(&self, audio: &AudioBuffer) -> Result<f32, EvaluationError> {
        // Placeholder: estimate words per minute
        // In production, this would use more sophisticated speech analysis
        let duration = audio.samples().len() as f32 / audio.sample_rate() as f32;
        if duration < 0.1 {
            return Ok(150.0); // Default rate
        }

        // Estimate based on energy peaks (syllables)
        let energy_envelope = self.calculate_energy_envelope(audio.samples());
        let peaks = self.count_peaks(&energy_envelope, 0.1);

        // Rough estimate: syllables per second * 60 * 0.7 (syllables to words ratio)
        let syllables_per_sec = peaks as f32 / duration;
        let words_per_min = syllables_per_sec * 60.0 * 0.7;

        Ok(words_per_min.clamp(60.0, 300.0))
    }

    fn get_expected_rate_range(&self, context: &EvaluationContext) -> (f32, f32) {
        match context.speaking_style {
            SpeakingStyle::Reading | SpeakingStyle::Broadcasting => (140.0, 180.0),
            SpeakingStyle::Conversation => (120.0, 160.0),
            SpeakingStyle::Presentation => (110.0, 150.0),
            SpeakingStyle::Narration => (130.0, 170.0),
            SpeakingStyle::Command => (100.0, 130.0),
            SpeakingStyle::CustomerService => (120.0, 160.0),
        }
    }

    fn estimate_noise_level(&self, audio: &AudioBuffer) -> Result<f32, EvaluationError> {
        // Calculate noise level using minimum energy frames
        let frame_size = 512;
        let samples = audio.samples();

        let mut min_energy = f32::MAX;
        for chunk in samples.chunks(frame_size) {
            let energy = self.calculate_rms(chunk);
            if energy < min_energy && energy > 1e-10 {
                min_energy = energy;
            }
        }

        Ok(min_energy.min(1.0))
    }

    fn calculate_rms(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: f32 = samples.iter().map(|&s| s * s).sum();
        (sum / samples.len() as f32).sqrt()
    }

    fn calculate_variation(&self, samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }

        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        let variance: f32 =
            samples.iter().map(|&s| (s - mean).powi(2)).sum::<f32>() / samples.len() as f32;

        variance.sqrt()
    }

    fn calculate_energy_envelope(&self, samples: &[f32]) -> Vec<f32> {
        let window_size = 256;
        let hop_size = 128;
        let mut envelope = Vec::new();

        for i in (0..samples.len()).step_by(hop_size) {
            let end = (i + window_size).min(samples.len());
            let window = &samples[i..end];
            let energy = self.calculate_rms(window);
            envelope.push(energy);
        }

        envelope
    }

    fn count_peaks(&self, signal: &[f32], threshold: f32) -> usize {
        let mut peaks = 0;
        for i in 1..signal.len().saturating_sub(1) {
            if signal[i] > threshold && signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
                peaks += 1;
            }
        }
        peaks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_audio(duration_secs: f32, sample_rate: u32) -> AudioBuffer {
        let num_samples = (duration_secs * sample_rate as f32) as usize;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        AudioBuffer::new(samples, sample_rate, 1)
    }

    #[test]
    fn test_context_aware_evaluator_creation() {
        let config = ContextAwareConfig::default();
        let evaluator = ContextAwareEvaluator::new(config);
        assert!(evaluator.is_ok());
    }

    #[test]
    fn test_evaluation_context_builder() {
        let context = EvaluationContext::builder()
            .speaking_style(SpeakingStyle::Conversation)
            .emotional_context(EmotionalContext::Happy)
            .formality_level(FormalityLevel::Informal)
            .domain(Domain::Podcast)
            .build();

        assert_eq!(context.speaking_style, SpeakingStyle::Conversation);
        assert_eq!(context.emotional_context, EmotionalContext::Happy);
        assert_eq!(context.formality_level, FormalityLevel::Informal);
        assert_eq!(context.domain, Domain::Podcast);
    }

    #[test]
    fn test_basic_evaluation() {
        let config = ContextAwareConfig::default();
        let evaluator = ContextAwareEvaluator::new(config).unwrap();

        let audio = create_test_audio(3.0, 16000);
        let context = EvaluationContext::default();

        let result = evaluator.evaluate(&audio, &audio, &context);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.overall_quality >= 0.0 && result.overall_quality <= 1.0);
        assert!(result.context_appropriateness >= 0.0 && result.context_appropriateness <= 1.0);
    }

    #[test]
    fn test_weight_adaptation() {
        let config = ContextAwareConfig {
            auto_adapt_weights: true,
            ..Default::default()
        };
        let evaluator = ContextAwareEvaluator::new(config).unwrap();

        let context_reading = EvaluationContext::builder()
            .speaking_style(SpeakingStyle::Reading)
            .build();

        let context_conversation = EvaluationContext::builder()
            .speaking_style(SpeakingStyle::Conversation)
            .build();

        let weights_reading = evaluator.adapt_weights(&context_reading);
        let weights_conversation = evaluator.adapt_weights(&context_conversation);

        // Reading should prioritize intelligibility
        assert!(
            weights_reading.intelligibility_weight > weights_conversation.intelligibility_weight
        );

        // Conversation should prioritize naturalness
        assert!(weights_conversation.naturalness_weight > weights_reading.naturalness_weight);
    }

    #[test]
    fn test_recommendation_generation() {
        let config = ContextAwareConfig {
            generate_recommendations: true,
            quality_threshold: 0.8,
            ..Default::default()
        };
        let evaluator = ContextAwareEvaluator::new(config).unwrap();

        let weights = ContextMetricWeights::default();
        let context = EvaluationContext::default();

        // Low scores should generate recommendations
        let recommendations =
            evaluator.generate_recommendations(&weights, 0.6, 0.5, 0.7, 0.8, 0.9, 0.8, &context);

        assert!(!recommendations.is_empty());
    }

    #[test]
    fn test_speaking_rate_estimation() {
        let config = ContextAwareConfig::default();
        let evaluator = ContextAwareEvaluator::new(config).unwrap();

        let audio = create_test_audio(3.0, 16000);
        let rate = evaluator.estimate_speaking_rate(&audio);

        assert!(rate.is_ok());
        let rate = rate.unwrap();
        assert!(rate >= 60.0 && rate <= 300.0);
    }

    #[test]
    fn test_emotion_detection() {
        let config = ContextAwareConfig::default();
        let evaluator = ContextAwareEvaluator::new(config).unwrap();

        let audio = create_test_audio(3.0, 16000);
        let emotion = evaluator.detect_emotion(&audio);

        assert!(emotion.is_ok());
    }

    #[test]
    fn test_formality_detection() {
        let config = ContextAwareConfig::default();
        let evaluator = ContextAwareEvaluator::new(config).unwrap();

        let audio = create_test_audio(3.0, 16000);
        let formality = evaluator.detect_formality(&audio);

        assert!(formality.is_ok());
    }

    #[test]
    fn test_context_appropriateness_calculation() {
        let config = ContextAwareConfig::default();
        let evaluator = ContextAwareEvaluator::new(config).unwrap();

        let context = EvaluationContext::builder()
            .speaking_style(SpeakingStyle::Conversation)
            .build();

        let score =
            evaluator.calculate_context_appropriateness(0.9, 0.8, 0.7, 0.8, 0.9, 0.85, &context);

        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_different_speaking_styles() {
        let config = ContextAwareConfig::default();
        let evaluator = ContextAwareEvaluator::new(config).unwrap();

        let audio = create_test_audio(3.0, 16000);
        let styles = vec![
            SpeakingStyle::Reading,
            SpeakingStyle::Conversation,
            SpeakingStyle::Presentation,
            SpeakingStyle::Narration,
        ];

        for style in styles {
            let context = EvaluationContext::builder().speaking_style(style).build();
            let result = evaluator.evaluate(&audio, &audio, &context);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_sample_rate_mismatch() {
        let config = ContextAwareConfig::default();
        let evaluator = ContextAwareEvaluator::new(config).unwrap();

        let audio1 = create_test_audio(3.0, 16000);
        let audio2 = create_test_audio(3.0, 22050);
        let context = EvaluationContext::default();

        let result = evaluator.evaluate(&audio1, &audio2, &context);
        assert!(result.is_err());
    }
}
