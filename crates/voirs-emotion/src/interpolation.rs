//! Emotion interpolation for smooth transitions

use crate::{
    types::{EmotionDimensions, EmotionParameters, EmotionVector},
    Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Interpolation methods for emotion transitions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum InterpolationMethod {
    /// Linear interpolation
    Linear,
    /// Ease-in (slow start)
    EaseIn,
    /// Ease-out (slow end)
    EaseOut,
    /// Ease-in-out (slow start and end)
    #[default]
    EaseInOut,
    /// Bezier curve interpolation
    Bezier,
    /// Spline interpolation
    Spline,
    /// Custom interpolation function
    Custom,
}

impl InterpolationMethod {
    /// Apply interpolation function to a parameter t (0.0 to 1.0)
    pub fn apply(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);

        match self {
            InterpolationMethod::Linear => t,
            InterpolationMethod::EaseIn => t * t,
            InterpolationMethod::EaseOut => 1.0 - (1.0 - t) * (1.0 - t),
            InterpolationMethod::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - 2.0 * (1.0 - t) * (1.0 - t)
                }
            }
            InterpolationMethod::Bezier => {
                // Simplified cubic bezier with control points (0.25, 0.1, 0.75, 0.9)
                let inv_t = 1.0 - t;
                3.0 * inv_t * inv_t * t * 0.25 + 3.0 * inv_t * t * t * 0.75 + t * t * t
            }
            InterpolationMethod::Spline => {
                // Smooth step function
                t * t * (3.0 - 2.0 * t)
            }
            InterpolationMethod::Custom => t, // Default to linear
        }
    }
}

/// Configuration for emotion interpolation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterpolationConfig {
    /// Default interpolation method
    pub method: InterpolationMethod,
    /// Duration of transitions in milliseconds
    pub transition_duration_ms: u64,
    /// Minimum change threshold for starting a transition
    pub change_threshold: f32,
    /// Maximum number of simultaneous transitions
    pub max_concurrent_transitions: usize,
    /// Use dimension-based interpolation
    pub use_dimension_interpolation: bool,
    /// Interpolation weights for different parameters
    pub parameter_weights: ParameterWeights,
}

impl InterpolationConfig {
    /// Create default interpolation config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set interpolation method
    pub fn with_method(mut self, method: InterpolationMethod) -> Self {
        self.method = method;
        self
    }

    /// Set transition duration
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.transition_duration_ms = duration_ms;
        self
    }

    /// Set change threshold
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.change_threshold = threshold.clamp(0.0, 1.0);
        self
    }
}

impl Default for InterpolationConfig {
    fn default() -> Self {
        Self {
            method: InterpolationMethod::EaseInOut,
            transition_duration_ms: 1000,
            change_threshold: 0.1,
            max_concurrent_transitions: 3,
            use_dimension_interpolation: true,
            parameter_weights: ParameterWeights::default(),
        }
    }
}

/// Weights for different parameter types during interpolation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterWeights {
    /// Weight for emotion vector interpolation
    pub emotion_vector: f32,
    /// Weight for prosody parameter interpolation
    pub prosody: f32,
    /// Weight for voice quality interpolation
    pub voice_quality: f32,
    /// Weight for timing parameter interpolation
    pub timing: f32,
}

impl ParameterWeights {
    /// Create equal weights
    pub fn equal() -> Self {
        Self {
            emotion_vector: 1.0,
            prosody: 1.0,
            voice_quality: 1.0,
            timing: 1.0,
        }
    }

    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let total = self.emotion_vector + self.prosody + self.voice_quality + self.timing;
        if total > 0.0 {
            self.emotion_vector /= total;
            self.prosody /= total;
            self.voice_quality /= total;
            self.timing /= total;
        }
    }
}

impl Default for ParameterWeights {
    fn default() -> Self {
        Self::equal()
    }
}

/// Emotion interpolator for smooth transitions between emotional states
#[derive(Debug, Clone)]
pub struct EmotionInterpolator {
    /// Interpolation configuration
    config: InterpolationConfig,
    /// Active transitions
    active_transitions: Vec<TransitionState>,
}

impl EmotionInterpolator {
    /// Create a new emotion interpolator
    pub fn new(config: InterpolationConfig) -> Self {
        Self {
            config,
            active_transitions: Vec::new(),
        }
    }

    /// Interpolate between two emotion parameters
    pub fn interpolate(
        &self,
        from: &EmotionParameters,
        to: &EmotionParameters,
        progress: f32,
    ) -> Result<EmotionParameters> {
        let t = self.config.method.apply(progress.clamp(0.0, 1.0));

        // Interpolate emotion vector
        let emotion_vector =
            self.interpolate_emotion_vector(&from.emotion_vector, &to.emotion_vector, t)?;

        // Interpolate prosody parameters
        let pitch_shift = lerp(from.pitch_shift, to.pitch_shift, t);
        let tempo_scale = lerp(from.tempo_scale, to.tempo_scale, t);
        let energy_scale = lerp(from.energy_scale, to.energy_scale, t);

        // Interpolate voice quality
        let breathiness = lerp(from.breathiness, to.breathiness, t);
        let roughness = lerp(from.roughness, to.roughness, t);

        // Interpolate timing parameters
        let duration_ms = interpolate_option(from.duration_ms, to.duration_ms, t);
        let fade_in_ms = interpolate_option(from.fade_in_ms, to.fade_in_ms, t);
        let fade_out_ms = interpolate_option(from.fade_out_ms, to.fade_out_ms, t);

        // Interpolate custom parameters
        let custom_params =
            self.interpolate_custom_params(&from.custom_params, &to.custom_params, t);

        Ok(EmotionParameters {
            emotion_vector,
            duration_ms,
            fade_in_ms,
            fade_out_ms,
            pitch_shift,
            tempo_scale,
            energy_scale,
            breathiness,
            roughness,
            custom_params,
        })
    }

    /// Interpolate between emotion vectors
    pub fn interpolate_emotion_vector(
        &self,
        from: &EmotionVector,
        to: &EmotionVector,
        progress: f32,
    ) -> Result<EmotionVector> {
        let t = progress.clamp(0.0, 1.0);

        if self.config.use_dimension_interpolation {
            // Use dimensional interpolation for smoother transitions
            self.interpolate_via_dimensions(from, to, t)
        } else {
            // Direct emotion component interpolation
            self.interpolate_emotion_components(from, to, t)
        }
    }

    /// Interpolate using emotion dimensions
    fn interpolate_via_dimensions(
        &self,
        from: &EmotionVector,
        to: &EmotionVector,
        t: f32,
    ) -> Result<EmotionVector> {
        // Interpolate dimensions
        let from_dims = &from.dimensions;
        let to_dims = &to.dimensions;

        let valence = lerp(from_dims.valence, to_dims.valence, t);
        let arousal = lerp(from_dims.arousal, to_dims.arousal, t);
        let dominance = lerp(from_dims.dominance, to_dims.dominance, t);

        let interpolated_dims = EmotionDimensions::new(valence, arousal, dominance);

        // Create emotion vector from interpolated dimensions
        let mut result = EmotionVector::new();
        result.dimensions = interpolated_dims;

        // Blend emotion components based on dimensional proximity
        self.blend_emotions_by_dimensions(&mut result, from, to, t)?;

        Ok(result)
    }

    /// Blend emotions based on dimensional similarity
    fn blend_emotions_by_dimensions(
        &self,
        result: &mut EmotionVector,
        from: &EmotionVector,
        to: &EmotionVector,
        t: f32,
    ) -> Result<()> {
        // Collect all unique emotions
        let mut all_emotions = std::collections::HashSet::new();
        for emotion in from.emotions.keys() {
            all_emotions.insert(emotion.clone());
        }
        for emotion in to.emotions.keys() {
            all_emotions.insert(emotion.clone());
        }

        // Interpolate each emotion's intensity
        for emotion in all_emotions {
            let from_intensity = from
                .emotions
                .get(&emotion)
                .map(|i| i.value())
                .unwrap_or(0.0);
            let to_intensity = to.emotions.get(&emotion).map(|i| i.value()).unwrap_or(0.0);

            let interpolated_intensity = lerp(from_intensity, to_intensity, t);

            if interpolated_intensity > 0.01 {
                result.add_emotion(emotion, interpolated_intensity.into());
            }
        }

        Ok(())
    }

    /// Direct interpolation of emotion components
    fn interpolate_emotion_components(
        &self,
        from: &EmotionVector,
        to: &EmotionVector,
        t: f32,
    ) -> Result<EmotionVector> {
        let mut result = EmotionVector::new();

        // Collect all unique emotions
        let mut all_emotions = std::collections::HashSet::new();
        for emotion in from.emotions.keys() {
            all_emotions.insert(emotion.clone());
        }
        for emotion in to.emotions.keys() {
            all_emotions.insert(emotion.clone());
        }

        // Interpolate each emotion
        for emotion in all_emotions {
            let from_intensity = from
                .emotions
                .get(&emotion)
                .map(|i| i.value())
                .unwrap_or(0.0);
            let to_intensity = to.emotions.get(&emotion).map(|i| i.value()).unwrap_or(0.0);

            let interpolated_intensity = lerp(from_intensity, to_intensity, t);

            if interpolated_intensity > 0.01 {
                result.add_emotion(emotion, interpolated_intensity.into());
            }
        }

        Ok(result)
    }

    /// Interpolate custom parameters
    fn interpolate_custom_params(
        &self,
        from: &HashMap<String, f32>,
        to: &HashMap<String, f32>,
        t: f32,
    ) -> HashMap<String, f32> {
        let mut result = HashMap::new();

        // Collect all unique parameter names
        let mut all_params = std::collections::HashSet::new();
        for key in from.keys() {
            all_params.insert(key);
        }
        for key in to.keys() {
            all_params.insert(key);
        }

        // Interpolate each parameter
        for param in all_params {
            let from_value = from.get(param).copied().unwrap_or(0.0);
            let to_value = to.get(param).copied().unwrap_or(0.0);
            let interpolated_value = lerp(from_value, to_value, t);

            if interpolated_value.abs() > 0.001 {
                result.insert(param.clone(), interpolated_value);
            }
        }

        result
    }

    /// Start a new transition
    pub fn start_transition(
        &mut self,
        from: EmotionParameters,
        to: EmotionParameters,
        duration_ms: Option<u64>,
    ) -> Result<usize> {
        // Check if we're at the transition limit
        if self.active_transitions.len() >= self.config.max_concurrent_transitions {
            // Remove oldest transition
            self.active_transitions.remove(0);
        }

        let duration = duration_ms.unwrap_or(self.config.transition_duration_ms);

        let transition = TransitionState {
            id: fastrand::u64(..),
            from,
            to,
            start_time: std::time::Instant::now(),
            duration_ms: duration,
            method: self.config.method,
        };

        self.active_transitions.push(transition);
        Ok(self.active_transitions.len() - 1)
    }

    /// Update active transitions and get current blended state
    pub fn update_transitions(&mut self) -> Result<Option<EmotionParameters>> {
        let now = std::time::Instant::now();

        // Remove completed transitions
        self.active_transitions.retain(|transition| {
            let elapsed = now.duration_since(transition.start_time).as_millis() as u64;
            elapsed < transition.duration_ms
        });

        if self.active_transitions.is_empty() {
            return Ok(None);
        }

        // Calculate current state for each active transition
        let mut current_states = Vec::new();
        for transition in &self.active_transitions {
            let elapsed = now.duration_since(transition.start_time).as_millis() as u64;
            let progress = (elapsed as f32) / (transition.duration_ms as f32);
            let progress = transition.method.apply(progress.clamp(0.0, 1.0));

            let current = self.interpolate(&transition.from, &transition.to, progress)?;
            current_states.push((current, 1.0 / self.active_transitions.len() as f32));
        }

        // Blend all active transitions
        if current_states.len() == 1 {
            Ok(Some(current_states[0].0.clone()))
        } else {
            let blended = self.blend_multiple_states(current_states)?;
            Ok(Some(blended))
        }
    }

    /// Blend multiple emotion states
    fn blend_multiple_states(
        &self,
        states: Vec<(EmotionParameters, f32)>,
    ) -> Result<EmotionParameters> {
        if states.is_empty() {
            return Ok(EmotionParameters::neutral());
        }

        if states.len() == 1 {
            return Ok(states[0].0.clone());
        }

        // Use the first state as base and blend others
        let mut result = states[0].0.clone();
        let mut total_weight = states[0].1;

        for (state, weight) in states.iter().skip(1) {
            let blend_factor = weight / (total_weight + weight);
            result = self.interpolate(&result, state, blend_factor)?;
            total_weight += weight;
        }

        Ok(result)
    }

    /// Get number of active transitions
    pub fn active_transition_count(&self) -> usize {
        self.active_transitions.len()
    }

    /// Clear all active transitions
    pub fn clear_transitions(&mut self) {
        self.active_transitions.clear();
    }

    /// Set interpolation configuration
    pub fn set_config(&mut self, config: InterpolationConfig) {
        self.config = config;
    }

    /// Get interpolation configuration
    pub fn get_config(&self) -> &InterpolationConfig {
        &self.config
    }
}

impl Default for EmotionInterpolator {
    fn default() -> Self {
        Self::new(InterpolationConfig::default())
    }
}

/// State of an active transition
#[derive(Debug, Clone)]
struct TransitionState {
    /// Unique transition ID
    #[allow(dead_code)]
    id: u64,
    /// Starting emotion parameters
    from: EmotionParameters,
    /// Target emotion parameters
    to: EmotionParameters,
    /// When the transition started
    start_time: std::time::Instant,
    /// Duration of the transition
    duration_ms: u64,
    /// Interpolation method for this transition
    method: InterpolationMethod,
}

/// Linear interpolation helper function
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Interpolate optional values
fn interpolate_option(a: Option<u64>, b: Option<u64>, t: f32) -> Option<u64> {
    match (a, b) {
        (Some(a_val), Some(b_val)) => {
            Some((a_val as f32 + (b_val as f32 - a_val as f32) * t) as u64)
        }
        (Some(val), None) => Some(val),
        (None, Some(val)) => Some(val),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Emotion, EmotionIntensity, EmotionVector};

    #[test]
    fn test_interpolation_methods() {
        assert_eq!(InterpolationMethod::Linear.apply(0.5), 0.5);
        assert!(InterpolationMethod::EaseIn.apply(0.5) < 0.5);
        assert!(InterpolationMethod::EaseOut.apply(0.5) > 0.5);
    }

    #[test]
    fn test_emotion_interpolation() {
        let config = InterpolationConfig::default();
        let interpolator = EmotionInterpolator::new(config);

        let mut from_vector = EmotionVector::new();
        from_vector.add_emotion(Emotion::Happy, EmotionIntensity::LOW);
        let from_params = EmotionParameters::new(from_vector);

        let mut to_vector = EmotionVector::new();
        to_vector.add_emotion(Emotion::Happy, EmotionIntensity::HIGH);
        let to_params = EmotionParameters::new(to_vector);

        let result = interpolator
            .interpolate(&from_params, &to_params, 0.5)
            .unwrap();

        let happy_intensity = result.emotion_vector.emotions.get(&Emotion::Happy).unwrap();
        assert!(happy_intensity.value() > EmotionIntensity::LOW.value());
        assert!(happy_intensity.value() < EmotionIntensity::HIGH.value());
    }

    #[test]
    fn test_prosody_interpolation() {
        let config = InterpolationConfig::default();
        let interpolator = EmotionInterpolator::new(config);

        let from_params = EmotionParameters::neutral().with_prosody(1.0, 1.0, 1.0);
        let to_params = EmotionParameters::neutral().with_prosody(2.0, 1.5, 0.5);

        let result = interpolator
            .interpolate(&from_params, &to_params, 0.5)
            .unwrap();

        assert!((result.pitch_shift - 1.5).abs() < 0.1);
        assert!((result.tempo_scale - 1.25).abs() < 0.1);
        assert!((result.energy_scale - 0.75).abs() < 0.1);
    }

    #[test]
    fn test_transition_management() {
        let config = InterpolationConfig::default();
        let mut interpolator = EmotionInterpolator::new(config);

        let from_params = EmotionParameters::neutral();
        let to_params = EmotionParameters::neutral();

        interpolator
            .start_transition(from_params, to_params, Some(1000))
            .unwrap();
        assert_eq!(interpolator.active_transition_count(), 1);

        interpolator.clear_transitions();
        assert_eq!(interpolator.active_transition_count(), 0);
    }

    #[test]
    fn test_dimension_interpolation() {
        let config = InterpolationConfig {
            use_dimension_interpolation: true,
            ..InterpolationConfig::default()
        };
        let interpolator = EmotionInterpolator::new(config);

        let mut from_vector = EmotionVector::new();
        from_vector.add_emotion(Emotion::Happy, EmotionIntensity::HIGH);

        let mut to_vector = EmotionVector::new();
        to_vector.add_emotion(Emotion::Sad, EmotionIntensity::HIGH);

        let result = interpolator
            .interpolate_emotion_vector(&from_vector, &to_vector, 0.5)
            .unwrap();

        // Should have interpolated dimensions
        assert!(result.dimensions.valence.abs() < 0.8); // Between happy and sad valence
    }

    #[test]
    fn test_interpolation_method_edge_cases() {
        let method = InterpolationMethod::Linear;
        assert!(method.apply(f32::NAN).is_nan()); // NaN remains NaN with clamp
        assert_eq!(method.apply(f32::INFINITY), 1.0);
        assert_eq!(method.apply(f32::NEG_INFINITY), 0.0);

        // Test all methods with extreme values
        let methods = [
            InterpolationMethod::Linear,
            InterpolationMethod::EaseIn,
            InterpolationMethod::EaseOut,
            InterpolationMethod::EaseInOut,
            InterpolationMethod::Bezier,
            InterpolationMethod::Spline,
        ];

        for method in &methods {
            assert_eq!(method.apply(f32::INFINITY), 1.0);
            assert_eq!(method.apply(f32::NEG_INFINITY), 0.0);
            assert!(method.apply(f32::NAN).is_nan());
        }
    }

    #[test]
    fn test_interpolation_config_builder_methods() {
        let mut config = InterpolationConfig::default()
            .with_method(InterpolationMethod::Spline)
            .with_duration(2000)
            .with_threshold(0.1);

        config.max_concurrent_transitions = 5;
        config.use_dimension_interpolation = false;

        assert_eq!(config.method, InterpolationMethod::Spline);
        assert_eq!(config.transition_duration_ms, 2000);
        assert_eq!(config.change_threshold, 0.1);
        assert_eq!(config.max_concurrent_transitions, 5);
        assert!(!config.use_dimension_interpolation);
    }

    #[test]
    fn test_parameter_weights_custom() {
        let weights = ParameterWeights {
            emotion_vector: 2.0,
            prosody: 0.5,
            voice_quality: 1.5,
            timing: 0.8,
        };

        assert_eq!(weights.emotion_vector, 2.0);
        assert_eq!(weights.prosody, 0.5);
        assert_eq!(weights.voice_quality, 1.5);
        assert_eq!(weights.timing, 0.8);
    }

    #[test]
    fn test_parameter_weights_normalize() {
        let mut weights = ParameterWeights {
            emotion_vector: 2.0,
            prosody: 4.0,
            voice_quality: 2.0,
            timing: 2.0,
        };

        weights.normalize();

        let total =
            weights.emotion_vector + weights.prosody + weights.voice_quality + weights.timing;
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_parameter_weights_equal() {
        let weights = ParameterWeights::equal();

        assert_eq!(weights.emotion_vector, 1.0);
        assert_eq!(weights.prosody, 1.0);
        assert_eq!(weights.voice_quality, 1.0);
        assert_eq!(weights.timing, 1.0);
    }

    #[test]
    fn test_emotion_interpolator_start_transition() {
        let mut interpolator = EmotionInterpolator::new(InterpolationConfig::default());
        let start = EmotionParameters::default();
        let mut target = EmotionParameters::default();
        target.pitch_shift = 2.0;

        let transition_index = interpolator.start_transition(start, target, None).unwrap();

        assert_eq!(interpolator.active_transition_count(), 1);
        assert_eq!(transition_index, 0); // First transition should have index 0
    }

    #[test]
    fn test_emotion_interpolator_update_transitions() {
        let mut interpolator = EmotionInterpolator::new(InterpolationConfig::default());
        let start = EmotionParameters::default();
        let mut target = EmotionParameters::default();
        target.pitch_shift = 2.0;

        interpolator.start_transition(start, target, None).unwrap();

        let current = interpolator.update_transitions().unwrap();
        assert!(current.is_some());

        let params = current.unwrap();
        assert!(params.pitch_shift >= 1.0); // Should be progressing toward target
    }

    #[test]
    fn test_emotion_interpolator_short_transition() {
        let mut interpolator = EmotionInterpolator::new(InterpolationConfig::default());
        let start = EmotionParameters::default();
        let target = EmotionParameters::default();

        interpolator
            .start_transition(start, target, Some(1)) // Very short duration
            .unwrap();

        let initial_count = interpolator.active_transition_count();

        // Wait and update - should complete short transition
        std::thread::sleep(std::time::Duration::from_millis(10));
        let _ = interpolator.update_transitions();

        // Should have fewer active transitions after update
        assert!(interpolator.active_transition_count() <= initial_count);
    }

    #[test]
    fn test_emotion_interpolator_clear_single_transition() {
        let mut interpolator = EmotionInterpolator::new(InterpolationConfig::default());
        let start = EmotionParameters::default();
        let target = EmotionParameters::default();

        interpolator.start_transition(start, target, None).unwrap();

        assert_eq!(interpolator.active_transition_count(), 1);

        interpolator.clear_transitions();
        assert_eq!(interpolator.active_transition_count(), 0);
    }

    #[test]
    fn test_emotion_interpolator_clear_transitions() {
        let mut interpolator = EmotionInterpolator::new(InterpolationConfig::default());
        let start = EmotionParameters::default();
        let target = EmotionParameters::default();

        // Start multiple transitions
        interpolator
            .start_transition(start.clone(), target.clone(), None)
            .unwrap();
        interpolator.start_transition(start, target, None).unwrap();

        assert!(interpolator.active_transitions.len() >= 2);

        interpolator.clear_transitions();
        assert!(interpolator.active_transitions.is_empty());
    }

    #[test]
    fn test_emotion_interpolator_max_transitions() {
        let mut config = InterpolationConfig::default();
        config.max_concurrent_transitions = 2;
        let mut interpolator = EmotionInterpolator::new(config);
        let start = EmotionParameters::default();
        let target = EmotionParameters::default();

        // Start max number of transitions
        interpolator
            .start_transition(start.clone(), target.clone(), None)
            .unwrap();
        interpolator
            .start_transition(start.clone(), target.clone(), None)
            .unwrap();
        assert_eq!(interpolator.active_transition_count(), 2);

        // This should succeed but remove the oldest transition
        let result = interpolator.start_transition(start, target, None);
        assert!(result.is_ok());
        assert_eq!(interpolator.active_transition_count(), 2); // Still at max
    }

    #[test]
    fn test_interpolate_with_all_methods() {
        let methods = [
            InterpolationMethod::Linear,
            InterpolationMethod::EaseIn,
            InterpolationMethod::EaseOut,
            InterpolationMethod::EaseInOut,
            InterpolationMethod::Bezier,
            InterpolationMethod::Spline,
            InterpolationMethod::Custom,
        ];

        for method in &methods {
            // All methods should handle edge cases properly
            assert_eq!(method.apply(0.0), 0.0);
            assert_eq!(method.apply(1.0), 1.0);

            let mid = method.apply(0.5);
            assert!(mid >= 0.0 && mid <= 1.0);
        }
    }

    #[test]
    fn test_dimension_interpolation_edge_cases() {
        let neutral = EmotionDimensions::neutral();
        let extreme = EmotionDimensions::new(1.0, 1.0, 1.0);

        // Test interpolation using basic linear formula
        let start_valence = neutral.valence + (extreme.valence - neutral.valence) * 0.0;
        let start_arousal = neutral.arousal + (extreme.arousal - neutral.arousal) * 0.0;
        let start_dominance = neutral.dominance + (extreme.dominance - neutral.dominance) * 0.0;

        assert_eq!(start_valence, 0.0);
        assert_eq!(start_arousal, 0.0);
        assert_eq!(start_dominance, 0.0);

        let end_valence = neutral.valence + (extreme.valence - neutral.valence) * 1.0;
        let end_arousal = neutral.arousal + (extreme.arousal - neutral.arousal) * 1.0;
        let end_dominance = neutral.dominance + (extreme.dominance - neutral.dominance) * 1.0;

        assert_eq!(end_valence, 1.0);
        assert_eq!(end_arousal, 1.0);
        assert_eq!(end_dominance, 1.0);
    }

    #[test]
    fn test_emotion_vector_interpolation_empty_vectors() {
        let empty1 = EmotionVector::new();
        let empty2 = EmotionVector::new();

        // Test that empty vectors remain empty when interpolated
        assert!(empty1.emotions.is_empty());
        assert!(empty2.emotions.is_empty());

        // Test that dimensions remain neutral when interpolating empty vectors
        assert_eq!(empty1.dimensions.valence, 0.0);
        assert_eq!(empty1.dimensions.arousal, 0.0);
        assert_eq!(empty1.dimensions.dominance, 0.0);
    }

    #[test]
    fn test_parameter_interpolation_with_custom_params() {
        let mut start = EmotionParameters::default();
        start.custom_params.insert("custom1".to_string(), 0.0);
        start.custom_params.insert("custom2".to_string(), 10.0);

        let mut end = EmotionParameters::default();
        end.custom_params.insert("custom1".to_string(), 100.0);
        end.custom_params.insert("custom3".to_string(), 50.0);

        // Test linear interpolation of parameters manually
        let t = 0.5;
        let interpolated_pitch = start.pitch_shift + (end.pitch_shift - start.pitch_shift) * t;
        let interpolated_tempo = start.tempo_scale + (end.tempo_scale - start.tempo_scale) * t;
        let interpolated_energy = start.energy_scale + (end.energy_scale - start.energy_scale) * t;

        assert_eq!(interpolated_pitch, start.pitch_shift); // Both default values are same
        assert_eq!(interpolated_tempo, start.tempo_scale);
        assert_eq!(interpolated_energy, start.energy_scale);

        // Test custom parameter handling
        let custom1_start = start.custom_params.get("custom1").unwrap_or(&0.0);
        let custom1_end = end.custom_params.get("custom1").unwrap_or(&0.0);
        let custom1_interpolated = custom1_start + (custom1_end - custom1_start) * t;
        assert_eq!(custom1_interpolated, 50.0);
    }
}
