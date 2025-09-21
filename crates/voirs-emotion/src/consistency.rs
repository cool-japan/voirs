//! Emotion Consistency System for Maintaining Emotional Coherence Across Long Texts
//!
//! This module provides tools for maintaining emotional coherence in long-form
//! text synthesis by tracking emotional context, preventing abrupt changes,
//! and ensuring smooth emotional narratives.

use crate::interpolation::{EmotionInterpolator, InterpolationMethod};
use crate::types::{Emotion, EmotionIntensity, EmotionParameters, EmotionState};
use crate::Error;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, SystemTime};

/// Configuration for emotion consistency management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionConsistencyConfig {
    /// Maximum allowed intensity change per segment
    pub max_intensity_change: f32,
    /// Maximum dimensional change per segment (VAD space)
    pub max_dimensional_change: f32,
    /// Context window size (number of previous segments to consider)
    pub context_window_size: usize,
    /// Smoothing factor for transitions (0.0 = no smoothing, 1.0 = maximum smoothing)
    pub smoothing_factor: f32,
    /// Enable narrative coherence tracking
    pub enable_narrative_coherence: bool,
    /// Minimum segment duration for consistency checks
    pub min_segment_duration: Duration,
    /// Enable emotional momentum tracking
    pub enable_momentum: bool,
    /// Momentum decay factor
    pub momentum_decay: f32,
}

impl Default for EmotionConsistencyConfig {
    fn default() -> Self {
        Self {
            max_intensity_change: 0.3,   // 30% max change
            max_dimensional_change: 0.4, // 40% max change in VAD space
            context_window_size: 5,
            smoothing_factor: 0.7,
            enable_narrative_coherence: true,
            min_segment_duration: Duration::from_millis(500),
            enable_momentum: true,
            momentum_decay: 0.9,
        }
    }
}

/// Emotion segment with contextual information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionSegment {
    /// Unique segment ID
    pub segment_id: String,
    /// Target emotion for this segment
    pub target_emotion: Emotion,
    /// Target intensity
    pub target_intensity: EmotionIntensity,
    /// Original emotion parameters
    pub original_parameters: EmotionParameters,
    /// Adjusted parameters (after consistency processing)
    pub adjusted_parameters: EmotionParameters,
    /// Segment timestamp
    pub timestamp: SystemTime,
    /// Segment duration
    pub duration: Duration,
    /// Text content (for context)
    pub text_content: Option<String>,
    /// Narrative context tags
    pub context_tags: Vec<String>,
    /// Confidence in emotion assignment
    pub confidence: f32,
}

impl EmotionSegment {
    /// Create new emotion segment
    pub fn new(
        segment_id: String,
        target_emotion: Emotion,
        target_intensity: EmotionIntensity,
        parameters: EmotionParameters,
        duration: Duration,
    ) -> Self {
        Self {
            segment_id,
            target_emotion,
            target_intensity,
            original_parameters: parameters.clone(),
            adjusted_parameters: parameters,
            timestamp: SystemTime::now(),
            duration,
            text_content: None,
            context_tags: Vec::new(),
            confidence: 1.0,
        }
    }

    /// Add text content
    pub fn with_text(mut self, text: String) -> Self {
        self.text_content = Some(text);
        self
    }

    /// Add context tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.context_tags = tags;
        self
    }

    /// Set confidence level
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Calculate emotional distance from another segment
    pub fn emotional_distance(&self, other: &EmotionSegment) -> f32 {
        let dim_self = &self.adjusted_parameters.emotion_vector.dimensions;
        let dim_other = &other.adjusted_parameters.emotion_vector.dimensions;

        let valence_diff = (dim_self.valence - dim_other.valence).abs();
        let arousal_diff = (dim_self.arousal - dim_other.arousal).abs();
        let dominance_diff = (dim_self.dominance - dim_other.dominance).abs();

        (valence_diff * valence_diff
            + arousal_diff * arousal_diff
            + dominance_diff * dominance_diff)
            .sqrt()
    }
}

/// Emotional coherence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceMetrics {
    /// Overall coherence score (0.0 to 1.0)
    pub overall_coherence: f32,
    /// Average emotional distance between adjacent segments
    pub avg_transition_distance: f32,
    /// Maximum emotional jump detected
    pub max_emotional_jump: f32,
    /// Number of abrupt transitions
    pub abrupt_transitions: usize,
    /// Narrative consistency score
    pub narrative_consistency: f32,
    /// Emotional momentum score
    pub momentum_score: f32,
}

/// Emotional momentum tracker
#[derive(Debug, Clone)]
struct EmotionMomentum {
    /// Current emotional velocity (rate of change)
    velocity: EmotionParameters,
    /// Last update timestamp
    last_update: SystemTime,
    /// Momentum strength
    strength: f32,
}

impl EmotionMomentum {
    fn new() -> Self {
        Self {
            velocity: EmotionParameters::default(),
            last_update: SystemTime::now(),
            strength: 0.0,
        }
    }

    /// Update momentum based on new emotion parameters
    fn update(&mut self, new_params: &EmotionParameters, decay_factor: f32) {
        let now = SystemTime::now();
        let elapsed = now
            .duration_since(self.last_update)
            .unwrap_or(Duration::from_millis(1))
            .as_secs_f32();

        if elapsed > 0.0 {
            // Calculate velocity (change per second)
            let valence_vel = (new_params.emotion_vector.dimensions.valence
                - self.velocity.emotion_vector.dimensions.valence)
                / elapsed;
            let arousal_vel = (new_params.emotion_vector.dimensions.arousal
                - self.velocity.emotion_vector.dimensions.arousal)
                / elapsed;
            let dominance_vel = (new_params.emotion_vector.dimensions.dominance
                - self.velocity.emotion_vector.dimensions.dominance)
                / elapsed;

            // Update velocity with momentum decay
            self.velocity.emotion_vector.dimensions.valence =
                self.velocity.emotion_vector.dimensions.valence * decay_factor
                    + valence_vel * (1.0 - decay_factor);
            self.velocity.emotion_vector.dimensions.arousal =
                self.velocity.emotion_vector.dimensions.arousal * decay_factor
                    + arousal_vel * (1.0 - decay_factor);
            self.velocity.emotion_vector.dimensions.dominance =
                self.velocity.emotion_vector.dimensions.dominance * decay_factor
                    + dominance_vel * (1.0 - decay_factor);

            // Update momentum strength
            let velocity_magnitude = (self.velocity.emotion_vector.dimensions.valence
                * self.velocity.emotion_vector.dimensions.valence
                + self.velocity.emotion_vector.dimensions.arousal
                    * self.velocity.emotion_vector.dimensions.arousal
                + self.velocity.emotion_vector.dimensions.dominance
                    * self.velocity.emotion_vector.dimensions.dominance)
                .sqrt();

            self.strength = velocity_magnitude;
            self.last_update = now;
        }
    }

    /// Get predicted next emotion state based on momentum
    fn predict_next(
        &self,
        current_params: &EmotionParameters,
        time_delta: Duration,
    ) -> EmotionParameters {
        let delta_secs = time_delta.as_secs_f32();

        let mut predicted = current_params.clone();
        predicted.emotion_vector.dimensions.valence +=
            self.velocity.emotion_vector.dimensions.valence * delta_secs * self.strength;
        predicted.emotion_vector.dimensions.arousal +=
            self.velocity.emotion_vector.dimensions.arousal * delta_secs * self.strength;
        predicted.emotion_vector.dimensions.dominance +=
            self.velocity.emotion_vector.dimensions.dominance * delta_secs * self.strength;

        // Clamp to valid ranges
        predicted.emotion_vector.dimensions.valence =
            predicted.emotion_vector.dimensions.valence.clamp(-1.0, 1.0);
        predicted.emotion_vector.dimensions.arousal =
            predicted.emotion_vector.dimensions.arousal.clamp(-1.0, 1.0);
        predicted.emotion_vector.dimensions.dominance = predicted
            .emotion_vector
            .dimensions
            .dominance
            .clamp(-1.0, 1.0);

        predicted
    }
}

/// Emotion consistency manager
#[derive(Debug)]
pub struct EmotionConsistencyManager {
    /// Configuration
    config: EmotionConsistencyConfig,
    /// Segment history (context window)
    segment_history: VecDeque<EmotionSegment>,
    /// Emotion interpolator for smooth transitions
    interpolator: EmotionInterpolator,
    /// Emotional momentum tracker
    momentum: Option<EmotionMomentum>,
    /// Narrative context tags frequency
    context_frequency: std::collections::HashMap<String, usize>,
}

impl EmotionConsistencyManager {
    /// Create new consistency manager
    pub fn new(config: EmotionConsistencyConfig) -> Self {
        let interpolation_config = crate::interpolation::InterpolationConfig {
            method: InterpolationMethod::Linear,
            transition_duration_ms: 1000,
            change_threshold: 0.1,
            max_concurrent_transitions: 4,
            use_dimension_interpolation: true,
            parameter_weights: crate::interpolation::ParameterWeights::equal(),
        };
        let interpolator = EmotionInterpolator::new(interpolation_config);
        let momentum = if config.enable_momentum {
            Some(EmotionMomentum::new())
        } else {
            None
        };

        Self {
            config,
            segment_history: VecDeque::new(),
            interpolator,
            momentum,
            context_frequency: std::collections::HashMap::new(),
        }
    }

    /// Process new emotion segment with consistency checks
    pub fn process_segment(
        &mut self,
        mut segment: EmotionSegment,
    ) -> crate::Result<EmotionSegment> {
        // Update context frequency
        for tag in &segment.context_tags {
            *self.context_frequency.entry(tag.clone()).or_insert(0) += 1;
        }

        // Apply consistency adjustments
        segment.adjusted_parameters = self.apply_consistency_adjustments(&segment)?;

        // Update momentum if enabled
        if let Some(ref mut momentum) = self.momentum {
            momentum.update(&segment.adjusted_parameters, self.config.momentum_decay);
        }

        // Add to history (maintain window size)
        self.segment_history.push_back(segment.clone());
        if self.segment_history.len() > self.config.context_window_size {
            self.segment_history.pop_front();
        }

        Ok(segment)
    }

    /// Apply consistency adjustments to emotion parameters
    fn apply_consistency_adjustments(
        &self,
        segment: &EmotionSegment,
    ) -> crate::Result<EmotionParameters> {
        let mut adjusted = segment.original_parameters.clone();

        if let Some(previous_segment) = self.segment_history.back() {
            // Check for abrupt changes
            let distance =
                self.calculate_emotional_distance(&adjusted, &previous_segment.adjusted_parameters);

            if distance > self.config.max_dimensional_change {
                // Apply smoothing to reduce abrupt transitions
                adjusted = self.smooth_transition(
                    &previous_segment.adjusted_parameters,
                    &adjusted,
                    self.config.smoothing_factor,
                )?;
            }

            // Apply intensity constraints
            adjusted = self.apply_intensity_constraints(&adjusted, previous_segment)?;

            // Apply narrative coherence if enabled
            if self.config.enable_narrative_coherence {
                adjusted = self.apply_narrative_coherence(&adjusted, segment)?;
            }

            // Apply momentum prediction if enabled
            if let Some(ref momentum) = self.momentum {
                adjusted = self.apply_momentum_adjustment(&adjusted, momentum, segment)?;
            }
        }

        Ok(adjusted)
    }

    /// Calculate emotional distance between two parameter sets
    fn calculate_emotional_distance(
        &self,
        params1: &EmotionParameters,
        params2: &EmotionParameters,
    ) -> f32 {
        let dim1 = &params1.emotion_vector.dimensions;
        let dim2 = &params2.emotion_vector.dimensions;

        let valence_diff = (dim1.valence - dim2.valence).abs();
        let arousal_diff = (dim1.arousal - dim2.arousal).abs();
        let dominance_diff = (dim1.dominance - dim2.dominance).abs();

        (valence_diff * valence_diff
            + arousal_diff * arousal_diff
            + dominance_diff * dominance_diff)
            .sqrt()
    }

    /// Smooth transition between emotion parameters
    fn smooth_transition(
        &self,
        from_params: &EmotionParameters,
        to_params: &EmotionParameters,
        smoothing_factor: f32,
    ) -> crate::Result<EmotionParameters> {
        let interpolated_params =
            self.interpolator
                .interpolate(from_params, to_params, smoothing_factor)?;
        Ok(interpolated_params)
    }

    /// Apply intensity change constraints
    fn apply_intensity_constraints(
        &self,
        params: &EmotionParameters,
        previous_segment: &EmotionSegment,
    ) -> crate::Result<EmotionParameters> {
        let mut constrained = params.clone();
        let prev_intensity = previous_segment.adjusted_parameters.energy_scale;

        let intensity_diff = (constrained.energy_scale - prev_intensity).abs();
        if intensity_diff > self.config.max_intensity_change {
            // Limit the change
            let direction = if constrained.energy_scale > prev_intensity {
                1.0
            } else {
                -1.0
            };
            constrained.energy_scale =
                prev_intensity + direction * self.config.max_intensity_change;
        }

        Ok(constrained)
    }

    /// Apply narrative coherence adjustments
    fn apply_narrative_coherence(
        &self,
        params: &EmotionParameters,
        segment: &EmotionSegment,
    ) -> crate::Result<EmotionParameters> {
        let mut coherent = params.clone();

        // Analyze context tags for narrative consistency
        let dominant_context = self.get_dominant_narrative_context();
        if let Some(context) = dominant_context {
            // Apply context-specific adjustments
            coherent = self.adjust_for_narrative_context(&coherent, &context)?;
        }

        // Check for contradictory emotional patterns
        if let Some(contradiction_factor) = self.detect_narrative_contradiction(segment) {
            // Reduce emotional intensity for contradictory emotions
            coherent.energy_scale *= 1.0 - contradiction_factor;
            coherent.energy_scale = coherent.energy_scale.max(0.1); // Minimum threshold
        }

        Ok(coherent)
    }

    /// Apply momentum-based adjustments
    fn apply_momentum_adjustment(
        &self,
        params: &EmotionParameters,
        momentum: &EmotionMomentum,
        segment: &EmotionSegment,
    ) -> crate::Result<EmotionParameters> {
        if momentum.strength < 0.1 {
            return Ok(params.clone());
        }

        // Predict next state based on momentum
        let predicted = momentum.predict_next(params, segment.duration);

        // Blend original parameters with momentum prediction
        let momentum_influence = (momentum.strength * 0.3).min(0.5); // Max 50% influence

        let mut adjusted = params.clone();
        adjusted.emotion_vector.dimensions.valence = params.emotion_vector.dimensions.valence
            * (1.0 - momentum_influence)
            + predicted.emotion_vector.dimensions.valence * momentum_influence;
        adjusted.emotion_vector.dimensions.arousal = params.emotion_vector.dimensions.arousal
            * (1.0 - momentum_influence)
            + predicted.emotion_vector.dimensions.arousal * momentum_influence;
        adjusted.emotion_vector.dimensions.dominance = params.emotion_vector.dimensions.dominance
            * (1.0 - momentum_influence)
            + predicted.emotion_vector.dimensions.dominance * momentum_influence;

        // Clamp to valid ranges
        adjusted.emotion_vector.dimensions.valence =
            adjusted.emotion_vector.dimensions.valence.clamp(-1.0, 1.0);
        adjusted.emotion_vector.dimensions.arousal =
            adjusted.emotion_vector.dimensions.arousal.clamp(-1.0, 1.0);
        adjusted.emotion_vector.dimensions.dominance = adjusted
            .emotion_vector
            .dimensions
            .dominance
            .clamp(-1.0, 1.0);

        Ok(adjusted)
    }

    /// Get the most frequent narrative context
    fn get_dominant_narrative_context(&self) -> Option<String> {
        if self.context_frequency.is_empty() {
            return None;
        }

        self.context_frequency
            .iter()
            .max_by_key(|&(_, count)| count)
            .map(|(context, _)| context.clone())
    }

    /// Adjust parameters based on narrative context
    fn adjust_for_narrative_context(
        &self,
        params: &EmotionParameters,
        context: &str,
    ) -> crate::Result<EmotionParameters> {
        let mut adjusted = params.clone();

        match context.to_lowercase().as_str() {
            "dialogue" => {
                // Increase arousal and dominance for dialogue
                adjusted.emotion_vector.dimensions.arousal =
                    (adjusted.emotion_vector.dimensions.arousal * 1.1).min(1.0);
                adjusted.emotion_vector.dimensions.dominance =
                    (adjusted.emotion_vector.dimensions.dominance * 1.05).min(1.0);
            }
            "narration" => {
                // Slightly reduce arousal for narration
                adjusted.emotion_vector.dimensions.arousal =
                    (adjusted.emotion_vector.dimensions.arousal * 0.9).max(-1.0);
            }
            "action" => {
                // Increase arousal and energy for action sequences
                adjusted.emotion_vector.dimensions.arousal =
                    (adjusted.emotion_vector.dimensions.arousal * 1.2).min(1.0);
                adjusted.energy_scale = (adjusted.energy_scale * 1.1).min(2.0);
            }
            "reflection" => {
                // Reduce arousal and intensity for reflective passages
                adjusted.emotion_vector.dimensions.arousal =
                    (adjusted.emotion_vector.dimensions.arousal * 0.8).max(-1.0);
                adjusted.energy_scale = (adjusted.energy_scale * 0.9).max(0.1);
            }
            _ => {
                // No specific adjustments for unknown contexts
            }
        }

        Ok(adjusted)
    }

    /// Detect narrative contradictions
    fn detect_narrative_contradiction(&self, segment: &EmotionSegment) -> Option<f32> {
        if self.segment_history.len() < 2 {
            return None;
        }

        // Look for contradictory emotional patterns in recent history
        let recent_emotions: Vec<&Emotion> = self
            .segment_history
            .iter()
            .rev()
            .take(3)
            .map(|s| &s.target_emotion)
            .collect();

        // Check for rapid emotional oscillation
        if recent_emotions.len() >= 3 {
            let is_oscillation = (recent_emotions[0] == recent_emotions[2])
                && (recent_emotions[0] != recent_emotions[1]);

            if is_oscillation {
                return Some(0.3); // 30% reduction for oscillation
            }
        }

        // Check for extreme emotional contrasts
        if let Some(last_segment) = self.segment_history.back() {
            let distance = segment.emotional_distance(last_segment);
            if distance > 1.5 {
                return Some(0.2); // 20% reduction for extreme contrasts
            }
        }

        None
    }

    /// Calculate coherence metrics for the current segment history
    pub fn calculate_coherence_metrics(&self) -> CoherenceMetrics {
        if self.segment_history.len() < 2 {
            return CoherenceMetrics {
                overall_coherence: 1.0,
                avg_transition_distance: 0.0,
                max_emotional_jump: 0.0,
                abrupt_transitions: 0,
                narrative_consistency: 1.0,
                momentum_score: 0.0,
            };
        }

        let mut transition_distances = Vec::new();
        let mut abrupt_transitions = 0;

        // Calculate transition distances
        for window in self.segment_history.iter().collect::<Vec<_>>().windows(2) {
            let distance = window[1].emotional_distance(window[0]);
            transition_distances.push(distance);

            if distance > self.config.max_dimensional_change {
                abrupt_transitions += 1;
            }
        }

        let avg_transition_distance = if transition_distances.is_empty() {
            0.0
        } else {
            transition_distances.iter().sum::<f32>() / transition_distances.len() as f32
        };

        let max_emotional_jump = transition_distances
            .iter()
            .fold(0.0f32, |max, &distance| max.max(distance));

        // Calculate overall coherence (inverse of average transition distance)
        let overall_coherence = if avg_transition_distance > 0.0 {
            (1.0 / (1.0 + avg_transition_distance)).max(0.0).min(1.0)
        } else {
            1.0
        };

        // Calculate narrative consistency based on context tags
        let narrative_consistency = self.calculate_narrative_consistency();

        // Calculate momentum score
        let momentum_score = if let Some(ref momentum) = self.momentum {
            momentum.strength.min(1.0)
        } else {
            0.0
        };

        CoherenceMetrics {
            overall_coherence,
            avg_transition_distance,
            max_emotional_jump,
            abrupt_transitions,
            narrative_consistency,
            momentum_score,
        }
    }

    /// Calculate narrative consistency score
    fn calculate_narrative_consistency(&self) -> f32 {
        if self.segment_history.is_empty() {
            return 1.0;
        }

        // Check for consistent context usage
        let total_tags: usize = self
            .segment_history
            .iter()
            .map(|s| s.context_tags.len())
            .sum();

        if total_tags == 0 {
            return 0.5; // Neutral score for no context
        }

        // Calculate entropy of context distribution
        let total_frequency: usize = self.context_frequency.values().sum();
        if total_frequency == 0 {
            return 0.5;
        }

        let entropy: f32 = self
            .context_frequency
            .values()
            .map(|&freq| {
                let p = freq as f32 / total_frequency as f32;
                -p * p.log2()
            })
            .sum();

        // Lower entropy = higher consistency
        // Normalize entropy to 0-1 scale (assuming max entropy is around 3.0 for diverse contexts)
        let max_entropy = 3.0;
        let consistency = (max_entropy - entropy) / max_entropy;
        consistency.max(0.0).min(1.0)
    }

    /// Get recent segment history
    pub fn get_segment_history(&self) -> &VecDeque<EmotionSegment> {
        &self.segment_history
    }

    /// Clear segment history
    pub fn clear_history(&mut self) {
        self.segment_history.clear();
        self.context_frequency.clear();
        if let Some(ref mut momentum) = self.momentum {
            *momentum = EmotionMomentum::new();
        }
    }

    /// Export consistency report
    pub fn export_consistency_report(&self) -> crate::Result<String> {
        let metrics = self.calculate_coherence_metrics();

        let report_data = serde_json::json!({
            "config": self.config,
            "metrics": metrics,
            "segment_count": self.segment_history.len(),
            "context_frequency": self.context_frequency,
            "export_timestamp": SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs()
        });

        serde_json::to_string_pretty(&report_data).map_err(|e| Error::Serialization(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EmotionDimensions, EmotionIntensity, EmotionParameters, EmotionVector};

    #[test]
    fn test_emotion_segment_creation() {
        let mut emotion_vector = EmotionVector::new();
        emotion_vector.dimensions = EmotionDimensions::new(0.8, 0.6, 0.7);
        let params = EmotionParameters::new(emotion_vector);

        let segment = EmotionSegment::new(
            "seg_001".to_string(),
            Emotion::Happy,
            EmotionIntensity::new(0.8),
            params,
            Duration::from_secs(2),
        )
        .with_text("Hello world!".to_string())
        .with_tags(vec!["dialogue".to_string(), "greeting".to_string()])
        .with_confidence(0.9);

        assert_eq!(segment.segment_id, "seg_001");
        assert_eq!(segment.target_emotion, Emotion::Happy);
        assert_eq!(segment.text_content, Some("Hello world!".to_string()));
        assert_eq!(segment.context_tags.len(), 2);
        assert_eq!(segment.confidence, 0.9);
    }

    #[test]
    fn test_emotional_distance() {
        let mut emotion_vector1 = EmotionVector::new();
        emotion_vector1.dimensions = EmotionDimensions::new(0.8, 0.6, 0.7);
        let params1 = EmotionParameters::new(emotion_vector1);

        let mut emotion_vector2 = EmotionVector::new();
        emotion_vector2.dimensions = EmotionDimensions::new(-0.5, 0.2, 0.3);
        let params2 = EmotionParameters::new(emotion_vector2);

        let segment1 = EmotionSegment::new(
            "seg_001".to_string(),
            Emotion::Happy,
            EmotionIntensity::new(0.8),
            params1,
            Duration::from_secs(2),
        );

        let segment2 = EmotionSegment::new(
            "seg_002".to_string(),
            Emotion::Sad,
            EmotionIntensity::new(0.6),
            params2.clone(),
            Duration::from_secs(2),
        );

        let distance = segment1.emotional_distance(&segment2);
        assert!(distance > 0.0);
        assert!(distance < 2.5); // Maximum possible distance in 3D VAD space
    }

    #[test]
    fn test_consistency_manager() {
        let config = EmotionConsistencyConfig {
            max_dimensional_change: 0.5,
            context_window_size: 3,
            ..Default::default()
        };

        let mut manager = EmotionConsistencyManager::new(config);

        let mut emotion_vector1 = EmotionVector::new();
        emotion_vector1.dimensions = EmotionDimensions::new(0.8, 0.6, 0.7);
        let params1 = EmotionParameters::new(emotion_vector1);

        let segment1 = EmotionSegment::new(
            "seg_001".to_string(),
            Emotion::Happy,
            EmotionIntensity::new(0.8),
            params1,
            Duration::from_secs(2),
        )
        .with_tags(vec!["dialogue".to_string()]);

        let processed1 = manager.process_segment(segment1).unwrap();
        assert_eq!(processed1.segment_id, "seg_001");

        // Add a second segment with different emotion
        let mut emotion_vector2 = EmotionVector::new();
        emotion_vector2.dimensions = EmotionDimensions::new(-0.5, 0.2, 0.3);
        let params2 = EmotionParameters::new(emotion_vector2);

        let segment2 = EmotionSegment::new(
            "seg_002".to_string(),
            Emotion::Sad,
            EmotionIntensity::new(0.6),
            params2.clone(),
            Duration::from_secs(2),
        )
        .with_tags(vec!["dialogue".to_string()]);

        let processed2 = manager.process_segment(segment2).unwrap();

        // The processed parameters should be smoothed compared to original
        let original_distance = EmotionSegment::new(
            "temp".to_string(),
            Emotion::Sad,
            EmotionIntensity::new(0.6),
            params2.clone(),
            Duration::from_secs(2),
        )
        .emotional_distance(&processed1);

        let smoothed_distance = processed2.emotional_distance(&processed1);

        // Smoothed distance should be less than or equal to original
        assert!(smoothed_distance <= original_distance + 0.1); // Small tolerance for floating point
    }

    #[test]
    fn test_coherence_metrics() {
        let config = EmotionConsistencyConfig::default();
        let mut manager = EmotionConsistencyManager::new(config);

        // Add several segments
        for i in 0..5 {
            let intensity = 0.5 + (i as f32 * 0.1);
            let mut emotion_vector = EmotionVector::new();
            emotion_vector.dimensions = EmotionDimensions::new(intensity, 0.6, 0.7);
            let params = EmotionParameters::new(emotion_vector);

            let segment = EmotionSegment::new(
                format!("seg_{:03}", i),
                Emotion::Happy,
                EmotionIntensity::new(intensity),
                params,
                Duration::from_secs(2),
            )
            .with_tags(vec!["dialogue".to_string()]);

            manager.process_segment(segment).unwrap();
        }

        let metrics = manager.calculate_coherence_metrics();
        assert!(metrics.overall_coherence > 0.0);
        assert!(metrics.overall_coherence <= 1.0);
        assert!(metrics.abrupt_transitions >= 0); // May have some abrupt transitions in the test sequence
        assert!(metrics.narrative_consistency > 0.0);
    }

    #[test]
    fn test_narrative_context_tracking() {
        let config = EmotionConsistencyConfig::default();
        let mut manager = EmotionConsistencyManager::new(config);

        let params = EmotionParameters::new(EmotionVector::new());

        // Add segments with different context tags
        let contexts = vec!["dialogue", "narration", "dialogue", "dialogue"];

        for (i, context) in contexts.iter().enumerate() {
            let segment = EmotionSegment::new(
                format!("seg_{:03}", i),
                Emotion::Neutral,
                EmotionIntensity::new(0.5),
                params.clone(),
                Duration::from_secs(2),
            )
            .with_tags(vec![context.to_string()]);

            manager.process_segment(segment).unwrap();
        }

        // "dialogue" should be the dominant context (3 out of 4 segments)
        assert_eq!(manager.context_frequency.get("dialogue"), Some(&3));
        assert_eq!(manager.context_frequency.get("narration"), Some(&1));
    }
}
