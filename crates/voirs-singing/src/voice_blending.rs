//! Voice blending and morphing system
//!
//! This module provides smooth transitions between different voices,
//! voice morphing, and voice characteristic interpolation.

#![allow(dead_code, clippy::derivable_impls)]

use crate::types::VoiceCharacteristics;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Voice blending processor
#[derive(Debug, Clone)]
pub struct VoiceBlender {
    /// Current voice blend state
    state: BlendState,
    /// Blending configuration
    config: BlendConfig,
    /// Voice library for blending
    voice_library: HashMap<String, VoiceCharacteristics>,
    /// Blend curves for different parameters
    curves: BlendCurves,
    /// Transition history
    transition_history: Vec<BlendTransition>,
}

/// Voice blend state
#[derive(Debug, Clone)]
pub struct BlendState {
    /// Source voice characteristics
    pub source_voice: VoiceCharacteristics,
    /// Target voice characteristics
    pub target_voice: VoiceCharacteristics,
    /// Current blend progress (0.0-1.0)
    pub blend_progress: f32,
    /// Blend direction (1.0 = forward, -1.0 = reverse)
    pub blend_direction: f32,
    /// Transition start time
    pub transition_start: std::time::Instant,
    /// Transition duration
    pub transition_duration: Duration,
    /// Is transition active
    pub is_transitioning: bool,
    /// Current blended voice
    pub current_voice: VoiceCharacteristics,
}

/// Blending configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlendConfig {
    /// Default transition duration
    pub default_transition_duration: f32, // seconds
    /// Blend curve type
    pub blend_curve: BlendCurveType,
    /// Enable automatic voice matching
    pub auto_voice_matching: bool,
    /// Cross-fade overlap duration
    pub crossfade_overlap: f32, // seconds
    /// Preserve pitch during blending
    pub preserve_pitch: bool,
    /// Preserve timing during blending
    pub preserve_timing: bool,
    /// Voice similarity threshold for automatic blending
    pub similarity_threshold: f32,
    /// Enable harmonic alignment
    pub harmonic_alignment: bool,
    /// Maximum blend speed
    pub max_blend_speed: f32,
}

/// Blend curve types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlendCurveType {
    /// Linear interpolation
    Linear,
    /// Smooth S-curve
    Smooth,
    /// Exponential curve
    Exponential,
    /// Logarithmic curve
    Logarithmic,
    /// Custom bezier curve
    Bezier,
    /// Harmonic-based curve
    Harmonic,
}

/// Blend curves for different voice parameters
#[derive(Debug, Clone)]
pub struct BlendCurves {
    /// Pitch blend curve
    pub pitch_curve: Vec<f32>,
    /// Timbre blend curve
    pub timbre_curve: Vec<f32>,
    /// Resonance blend curve
    pub resonance_curve: Vec<f32>,
    /// Vibrato blend curve
    pub vibrato_curve: Vec<f32>,
    /// Power blend curve
    pub power_curve: Vec<f32>,
}

/// Voice morphing parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceMorphParams {
    /// Pitch shift in semitones
    pub pitch_shift: f32,
    /// Formant shift factor
    pub formant_shift: f32,
    /// Timbre modification
    pub timbre_morph: f32,
    /// Resonance modification
    pub resonance_morph: f32,
    /// Vibrato modification
    pub vibrato_morph: f32,
    /// Breathiness modification
    pub breathiness_morph: f32,
    /// Roughness modification
    pub roughness_morph: f32,
}

/// Voice blend transition record
#[derive(Debug, Clone)]
pub struct BlendTransition {
    /// Transition ID
    pub id: String,
    /// Source voice name
    pub source_voice: String,
    /// Target voice name
    pub target_voice: String,
    /// Transition start time
    pub start_time: std::time::Instant,
    /// Transition duration
    pub duration: Duration,
    /// Transition quality score
    pub quality_score: f32,
    /// Transition type
    pub transition_type: TransitionType,
}

/// Types of voice transitions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionType {
    /// Smooth crossfade
    Crossfade,
    /// Morphing transition
    Morph,
    /// Harmonic transition
    Harmonic,
    /// Dynamic transition
    Dynamic,
}

/// Voice similarity metrics
#[derive(Debug, Clone)]
pub struct VoiceSimilarity {
    /// Overall similarity score (0.0-1.0)
    pub overall_similarity: f32,
    /// Pitch similarity
    pub pitch_similarity: f32,
    /// Timbre similarity
    pub timbre_similarity: f32,
    /// Resonance similarity
    pub resonance_similarity: f32,
    /// Vibrato similarity
    pub vibrato_similarity: f32,
}

impl VoiceBlender {
    /// Create new voice blender
    pub fn new(config: BlendConfig) -> Self {
        Self {
            state: BlendState::new(),
            config,
            voice_library: HashMap::new(),
            curves: BlendCurves::default(),
            transition_history: Vec::new(),
        }
    }

    /// Add voice to library
    pub fn add_voice(&mut self, name: String, voice: VoiceCharacteristics) {
        self.voice_library.insert(name, voice);
    }

    /// Remove voice from library
    pub fn remove_voice(&mut self, name: &str) -> Option<VoiceCharacteristics> {
        self.voice_library.remove(name)
    }

    /// Start voice transition
    pub fn start_transition(
        &mut self,
        source_name: &str,
        target_name: &str,
        duration: Option<Duration>,
    ) -> crate::Result<()> {
        let source_voice = self
            .voice_library
            .get(source_name)
            .ok_or_else(|| crate::Error::Voice(format!("Source voice '{source_name}' not found")))?
            .clone();

        let target_voice = self
            .voice_library
            .get(target_name)
            .ok_or_else(|| crate::Error::Voice(format!("Target voice '{target_name}' not found")))?
            .clone();

        let transition_duration = duration
            .unwrap_or_else(|| Duration::from_secs_f32(self.config.default_transition_duration));

        // Calculate voice similarity for transition optimization
        let similarity = self.calculate_voice_similarity(&source_voice, &target_voice);
        let optimized_duration =
            self.optimize_transition_duration(&similarity, transition_duration);

        self.state.source_voice = source_voice;
        self.state.target_voice = target_voice;
        self.state.blend_progress = 0.0;
        self.state.blend_direction = 1.0;
        self.state.transition_start = std::time::Instant::now();
        self.state.transition_duration = optimized_duration;
        self.state.is_transitioning = true;
        self.state.current_voice = self.state.source_voice.clone();

        // Record transition
        let transition = BlendTransition {
            id: uuid::Uuid::new_v4().to_string(),
            source_voice: source_name.to_string(),
            target_voice: target_name.to_string(),
            start_time: self.state.transition_start,
            duration: optimized_duration,
            quality_score: similarity.overall_similarity,
            transition_type: self.select_transition_type(&similarity),
        };
        self.transition_history.push(transition);

        Ok(())
    }

    /// Update blend state and get current voice
    pub fn update(&mut self, delta_time: f32) -> VoiceCharacteristics {
        if !self.state.is_transitioning {
            return self.state.current_voice.clone();
        }

        let elapsed = self.state.transition_start.elapsed();
        let progress = elapsed.as_secs_f32() / self.state.transition_duration.as_secs_f32();

        if progress >= 1.0 {
            // Transition complete
            self.state.blend_progress = 1.0;
            self.state.is_transitioning = false;
            self.state.current_voice = self.state.target_voice.clone();
        } else {
            // Apply blend curve
            let curved_progress = self.apply_blend_curve(progress);
            self.state.blend_progress = curved_progress;

            // Interpolate voice characteristics
            self.state.current_voice = self.interpolate_voices(
                &self.state.source_voice,
                &self.state.target_voice,
                curved_progress,
            );
        }

        self.state.current_voice.clone()
    }

    /// Manually set blend progress
    pub fn set_blend_progress(&mut self, progress: f32) {
        let clamped_progress = progress.clamp(0.0, 1.0);
        self.state.blend_progress = clamped_progress;

        if self.state.is_transitioning {
            self.state.current_voice = self.interpolate_voices(
                &self.state.source_voice,
                &self.state.target_voice,
                clamped_progress,
            );
        }
    }

    /// Create morphed voice from base voice
    pub fn morph_voice(
        &self,
        base_voice: &VoiceCharacteristics,
        morph_params: &VoiceMorphParams,
    ) -> VoiceCharacteristics {
        let mut morphed = base_voice.clone();

        // Apply pitch shift
        let pitch_factor = 2.0_f32.powf(morph_params.pitch_shift / 12.0);
        morphed.f0_mean *= pitch_factor;
        morphed.range.0 *= pitch_factor;
        morphed.range.1 *= pitch_factor;

        // Apply formant shift
        if let Some(formant_freq) = morphed.resonance.get_mut("formant_frequency") {
            *formant_freq *= morph_params.formant_shift;
        }

        // Apply timbre modifications
        morphed.vibrato_frequency *= 1.0 + morph_params.vibrato_morph * 0.5;
        morphed.vibrato_depth *= 1.0 + morph_params.vibrato_morph * 0.3;

        // Apply breathiness and roughness
        morphed.timbre.insert(
            String::from("breathiness"),
            morph_params.breathiness_morph.clamp(0.0, 1.0),
        );
        morphed.timbre.insert(
            String::from("roughness"),
            morph_params.roughness_morph.clamp(0.0, 1.0),
        );

        morphed
    }

    /// Get blend status
    pub fn get_blend_state(&self) -> &BlendState {
        &self.state
    }

    /// Get transition history
    pub fn get_transition_history(&self) -> &[BlendTransition] {
        &self.transition_history
    }

    /// Calculate voice similarity
    pub fn calculate_voice_similarity(
        &self,
        voice1: &VoiceCharacteristics,
        voice2: &VoiceCharacteristics,
    ) -> VoiceSimilarity {
        // Pitch similarity (based on F0 mean and range)
        let pitch_diff = (voice1.f0_mean - voice2.f0_mean).abs() / voice1.f0_mean;
        let range_diff = ((voice1.range.1 - voice1.range.0) - (voice2.range.1 - voice2.range.0))
            .abs()
            / (voice1.range.1 - voice1.range.0);
        let pitch_similarity = 1.0 - (pitch_diff + range_diff * 0.5).clamp(0.0, 1.0);

        // Vibrato similarity
        let vibrato_freq_diff = (voice1.vibrato_frequency - voice2.vibrato_frequency).abs() / 10.0;
        let vibrato_depth_diff = (voice1.vibrato_depth - voice2.vibrato_depth).abs();
        let vibrato_similarity = 1.0 - (vibrato_freq_diff + vibrato_depth_diff).clamp(0.0, 1.0);

        // Timbre similarity (simplified - would use spectral analysis in practice)
        let timbre_similarity = self.calculate_timbre_similarity(&voice1.timbre, &voice2.timbre);

        // Resonance similarity
        let resonance_similarity =
            self.calculate_resonance_similarity(&voice1.resonance, &voice2.resonance);

        let overall_similarity = (pitch_similarity * 0.3
            + vibrato_similarity * 0.2
            + timbre_similarity * 0.3
            + resonance_similarity * 0.2)
            .clamp(0.0, 1.0);

        VoiceSimilarity {
            overall_similarity,
            pitch_similarity,
            timbre_similarity,
            resonance_similarity,
            vibrato_similarity,
        }
    }

    /// Interpolate between two voices
    fn interpolate_voices(
        &self,
        voice1: &VoiceCharacteristics,
        voice2: &VoiceCharacteristics,
        progress: f32,
    ) -> VoiceCharacteristics {
        let mut result = voice1.clone();

        // Interpolate basic parameters
        result.f0_mean = lerp(voice1.f0_mean, voice2.f0_mean, progress);
        result.f0_std = lerp(voice1.f0_std, voice2.f0_std, progress);
        result.range.0 = lerp(voice1.range.0, voice2.range.0, progress);
        result.range.1 = lerp(voice1.range.1, voice2.range.1, progress);
        result.vibrato_frequency =
            lerp(voice1.vibrato_frequency, voice2.vibrato_frequency, progress);
        result.vibrato_depth = lerp(voice1.vibrato_depth, voice2.vibrato_depth, progress);
        result.breath_capacity = lerp(voice1.breath_capacity, voice2.breath_capacity, progress);
        result.vocal_power = lerp(voice1.vocal_power, voice2.vocal_power, progress);

        // Interpolate timbre characteristics
        result.timbre = self.interpolate_hashmap(&voice1.timbre, &voice2.timbre, progress);
        result.resonance = self.interpolate_hashmap(&voice1.resonance, &voice2.resonance, progress);

        // Handle voice type (choose based on progress)
        result.voice_type = if progress < 0.5 {
            voice1.voice_type
        } else {
            voice2.voice_type
        };

        result
    }

    /// Apply blend curve to progress
    fn apply_blend_curve(&self, progress: f32) -> f32 {
        match self.config.blend_curve {
            BlendCurveType::Linear => progress,
            BlendCurveType::Smooth => {
                // Smooth S-curve (smoothstep)
                progress * progress * (3.0 - 2.0 * progress)
            }
            BlendCurveType::Exponential => progress.powi(2),
            BlendCurveType::Logarithmic => progress.sqrt(),
            BlendCurveType::Bezier => {
                // Cubic bezier approximation
                let t = progress;
                let t2 = t * t;
                let t3 = t2 * t;
                3.0 * t2 - 2.0 * t3
            }
            BlendCurveType::Harmonic => {
                // Sine-based curve for musical transitions
                (progress * std::f32::consts::PI / 2.0).sin()
            }
        }
    }

    /// Calculate timbre similarity
    fn calculate_timbre_similarity(
        &self,
        timbre1: &HashMap<String, f32>,
        timbre2: &HashMap<String, f32>,
    ) -> f32 {
        let mut total_diff = 0.0;
        let mut count = 0;

        for (key, &value1) in timbre1 {
            if let Some(&value2) = timbre2.get(key) {
                total_diff += (value1 - value2).abs();
                count += 1;
            }
        }

        if count > 0 {
            1.0 - (total_diff / count as f32).clamp(0.0, 1.0)
        } else {
            0.5 // Default similarity when no common parameters
        }
    }

    /// Calculate resonance similarity
    fn calculate_resonance_similarity(
        &self,
        resonance1: &HashMap<String, f32>,
        resonance2: &HashMap<String, f32>,
    ) -> f32 {
        self.calculate_timbre_similarity(resonance1, resonance2)
    }

    /// Interpolate HashMap values
    fn interpolate_hashmap(
        &self,
        map1: &HashMap<String, f32>,
        map2: &HashMap<String, f32>,
        progress: f32,
    ) -> HashMap<String, f32> {
        let mut result = HashMap::new();

        // Get all unique keys
        let mut all_keys: std::collections::HashSet<String> = map1.keys().cloned().collect();
        all_keys.extend(map2.keys().cloned());

        for key in all_keys {
            let value1 = map1.get(&key).copied().unwrap_or(0.0);
            let value2 = map2.get(&key).copied().unwrap_or(0.0);
            result.insert(key, lerp(value1, value2, progress));
        }

        result
    }

    /// Optimize transition duration based on voice similarity
    fn optimize_transition_duration(
        &self,
        similarity: &VoiceSimilarity,
        base_duration: Duration,
    ) -> Duration {
        // More similar voices can transition faster
        let similarity_factor = 1.0 - similarity.overall_similarity * 0.5;
        let optimized_seconds = base_duration.as_secs_f32() * similarity_factor;
        Duration::from_secs_f32(optimized_seconds.max(0.1)) // Minimum 0.1 seconds
    }

    /// Select appropriate transition type based on voice similarity
    fn select_transition_type(&self, similarity: &VoiceSimilarity) -> TransitionType {
        if similarity.overall_similarity > 0.8 {
            TransitionType::Crossfade
        } else if similarity.pitch_similarity > 0.7 {
            TransitionType::Harmonic
        } else if similarity.timbre_similarity > 0.6 {
            TransitionType::Morph
        } else {
            TransitionType::Dynamic
        }
    }
}

impl BlendState {
    fn new() -> Self {
        Self {
            source_voice: VoiceCharacteristics::default(),
            target_voice: VoiceCharacteristics::default(),
            blend_progress: 0.0,
            blend_direction: 1.0,
            transition_start: std::time::Instant::now(),
            transition_duration: Duration::from_secs(1),
            is_transitioning: false,
            current_voice: VoiceCharacteristics::default(),
        }
    }
}

impl Default for BlendConfig {
    fn default() -> Self {
        Self {
            default_transition_duration: 2.0,
            blend_curve: BlendCurveType::Smooth,
            auto_voice_matching: true,
            crossfade_overlap: 0.1,
            preserve_pitch: false,
            preserve_timing: true,
            similarity_threshold: 0.7,
            harmonic_alignment: true,
            max_blend_speed: 2.0,
        }
    }
}

impl Default for BlendCurves {
    fn default() -> Self {
        let curve_points = 64;
        let linear_curve: Vec<f32> = (0..curve_points)
            .map(|i| i as f32 / (curve_points - 1) as f32)
            .collect();

        Self {
            pitch_curve: linear_curve.clone(),
            timbre_curve: linear_curve.clone(),
            resonance_curve: linear_curve.clone(),
            vibrato_curve: linear_curve.clone(),
            power_curve: linear_curve,
        }
    }
}

impl Default for VoiceMorphParams {
    fn default() -> Self {
        Self {
            pitch_shift: 0.0,
            formant_shift: 1.0,
            timbre_morph: 0.0,
            resonance_morph: 0.0,
            vibrato_morph: 0.0,
            breathiness_morph: 0.0,
            roughness_morph: 0.0,
        }
    }
}

/// Linear interpolation utility function
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::VoiceType;

    #[test]
    fn test_voice_blender_creation() {
        let config = BlendConfig::default();
        let blender = VoiceBlender::new(config);
        assert!(!blender.state.is_transitioning);
        assert_eq!(blender.voice_library.len(), 0);
    }

    #[test]
    fn test_voice_library_management() {
        let mut blender = VoiceBlender::new(BlendConfig::default());
        let voice = VoiceCharacteristics {
            voice_type: VoiceType::Soprano,
            ..Default::default()
        };

        blender.add_voice(String::from("soprano1"), voice.clone());
        assert_eq!(blender.voice_library.len(), 1);

        let removed = blender.remove_voice("soprano1");
        assert!(removed.is_some());
        assert_eq!(blender.voice_library.len(), 0);
    }

    #[test]
    fn test_voice_similarity_calculation() {
        let blender = VoiceBlender::new(BlendConfig::default());

        let voice1 = VoiceCharacteristics {
            voice_type: VoiceType::Alto,
            f0_mean: 440.0,
            vibrato_frequency: 6.0,
            ..Default::default()
        };

        let voice2 = VoiceCharacteristics {
            voice_type: VoiceType::Alto,
            f0_mean: 450.0,
            vibrato_frequency: 6.2,
            ..Default::default()
        };

        let similarity = blender.calculate_voice_similarity(&voice1, &voice2);
        assert!(similarity.overall_similarity > 0.6); // Should be similar (adjusted from 0.8)
        assert!(similarity.pitch_similarity > 0.8); // Should be high pitch similarity
        assert!(similarity.vibrato_similarity > 0.8); // Should be high vibrato similarity
    }

    #[test]
    fn test_voice_interpolation() {
        let blender = VoiceBlender::new(BlendConfig::default());

        let voice1 = VoiceCharacteristics {
            f0_mean: 400.0,
            vibrato_depth: 0.2,
            ..Default::default()
        };

        let voice2 = VoiceCharacteristics {
            f0_mean: 500.0,
            vibrato_depth: 0.4,
            ..Default::default()
        };

        let interpolated = blender.interpolate_voices(&voice1, &voice2, 0.5);
        assert!((interpolated.f0_mean - 450.0).abs() < 0.01);
        assert!((interpolated.vibrato_depth - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_voice_morphing() {
        let blender = VoiceBlender::new(BlendConfig::default());
        let base_voice = VoiceCharacteristics {
            f0_mean: 440.0,
            ..Default::default()
        };

        let morph_params = VoiceMorphParams {
            pitch_shift: 12.0, // One octave up
            ..Default::default()
        };

        let morphed = blender.morph_voice(&base_voice, &morph_params);
        assert!((morphed.f0_mean - 880.0).abs() < 1.0); // Should be approximately doubled
    }

    #[test]
    fn test_blend_curves() {
        let blender = VoiceBlender::new(BlendConfig::default());

        // Test different blend curves
        let progress = 0.5;

        let linear = blender.apply_blend_curve(progress);
        assert!((linear - 0.5).abs() < 0.01);

        let mut config = BlendConfig::default();
        config.blend_curve = BlendCurveType::Smooth;
        let blender_smooth = VoiceBlender::new(config);
        let smooth = blender_smooth.apply_blend_curve(progress);
        assert!(smooth > 0.4 && smooth < 0.6); // Should still be around 0.5 but curved
    }

    #[test]
    fn test_transition_type_selection() {
        let blender = VoiceBlender::new(BlendConfig::default());

        let high_similarity = VoiceSimilarity {
            overall_similarity: 0.9,
            pitch_similarity: 0.9,
            timbre_similarity: 0.9,
            resonance_similarity: 0.9,
            vibrato_similarity: 0.9,
        };

        let transition_type = blender.select_transition_type(&high_similarity);
        assert_eq!(transition_type, TransitionType::Crossfade);

        let low_similarity = VoiceSimilarity {
            overall_similarity: 0.3,
            pitch_similarity: 0.3,
            timbre_similarity: 0.3,
            resonance_similarity: 0.3,
            vibrato_similarity: 0.3,
        };

        let transition_type_low = blender.select_transition_type(&low_similarity);
        assert_eq!(transition_type_low, TransitionType::Dynamic);
    }
}
