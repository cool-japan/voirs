//! Advanced Emotion Morphing System
//!
//! This module provides sophisticated emotion morphing capabilities for real-time
//! emotion gradient control and smooth emotion trajectory synthesis.
//!
//! ## Features
//!
//! - **Multi-dimensional Blending**: Blend multiple emotions simultaneously
//! - **Trajectory Control**: Define complex emotion evolution paths
//! - **Gradient Morphing**: Smooth transitions along emotion gradients
//! - **Bezier Curves**: Natural emotion trajectories using Bezier interpolation
//! - **Real-time Control**: Dynamic emotion morphing during synthesis

use crate::{
    types::{Emotion, EmotionDimensions, EmotionIntensity, EmotionVector},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Emotion morphing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionMorphConfig {
    /// Smoothing factor for transitions (0.0-1.0)
    pub smoothing: f32,
    /// Interpolation method
    pub interpolation: MorphInterpolation,
    /// Whether to preserve energy during morphing
    pub preserve_energy: bool,
    /// Maximum morphing rate (emotions/second)
    pub max_morph_rate: f32,
}

impl Default for EmotionMorphConfig {
    fn default() -> Self {
        Self {
            smoothing: 0.3,
            interpolation: MorphInterpolation::Cubic,
            preserve_energy: true,
            max_morph_rate: 2.0,
        }
    }
}

/// Interpolation method for morphing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MorphInterpolation {
    /// Linear interpolation
    Linear,
    /// Cubic (Hermite) interpolation
    Cubic,
    /// Bezier curve interpolation
    Bezier,
    /// Cosine interpolation
    Cosine,
}

/// Emotion trajectory defining emotion evolution over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionTrajectory {
    /// Keyframes defining emotion states at specific times
    keyframes: Vec<EmotionKeyframe>,
    /// Total duration of the trajectory (seconds)
    duration: f32,
}

/// Emotion keyframe at a specific time point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionKeyframe {
    /// Time position (0.0-1.0, normalized to trajectory duration)
    pub time: f32,
    /// Emotion state at this keyframe
    pub emotion: EmotionVector,
    /// Optional easing function for transition from this keyframe
    pub easing: Option<EasingFunction>,
}

/// Easing function for smooth transitions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EasingFunction {
    /// Linear easing (constant velocity)
    Linear,
    /// Ease in (accelerating from zero)
    EaseIn,
    /// Ease out (decelerating to zero)
    EaseOut,
    /// Ease in-out (accelerating then decelerating)
    EaseInOut,
    /// Elastic easing (spring-like overshoot)
    Elastic,
    /// Bounce easing
    Bounce,
}

impl EmotionTrajectory {
    /// Create a new emotion trajectory
    pub fn new(duration: f32) -> Self {
        Self {
            keyframes: Vec::new(),
            duration,
        }
    }

    /// Add a keyframe to the trajectory
    pub fn add_keyframe(&mut self, time: f32, emotion: EmotionVector) {
        self.add_keyframe_with_easing(time, emotion, None);
    }

    /// Add a keyframe with easing function
    pub fn add_keyframe_with_easing(
        &mut self,
        time: f32,
        emotion: EmotionVector,
        easing: Option<EasingFunction>,
    ) {
        let normalized_time = (time / self.duration).clamp(0.0, 1.0);

        self.keyframes.push(EmotionKeyframe {
            time: normalized_time,
            emotion,
            easing,
        });

        // Keep keyframes sorted by time
        self.keyframes.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Sample the trajectory at a specific time
    pub fn sample(&self, time: f32) -> EmotionVector {
        if self.keyframes.is_empty() {
            return EmotionVector::new();
        }

        let normalized_time = (time / self.duration).clamp(0.0, 1.0);

        // Find surrounding keyframes
        let mut prev_keyframe = &self.keyframes[0];
        let mut next_keyframe = &self.keyframes[0];

        for keyframe in &self.keyframes {
            if keyframe.time <= normalized_time {
                prev_keyframe = keyframe;
            }
            if keyframe.time >= normalized_time {
                next_keyframe = keyframe;
                break;
            }
        }

        // If we're exactly on a keyframe, return it
        if (prev_keyframe.time - normalized_time).abs() < 1e-6 {
            return prev_keyframe.emotion.clone();
        }

        // Interpolate between keyframes
        let segment_duration = next_keyframe.time - prev_keyframe.time;
        if segment_duration < 1e-6 {
            return prev_keyframe.emotion.clone();
        }

        let t = (normalized_time - prev_keyframe.time) / segment_duration;

        // Apply easing function
        let eased_t = if let Some(easing) = prev_keyframe.easing {
            apply_easing(t, easing)
        } else {
            t
        };

        // Interpolate emotions
        self.interpolate_emotions(&prev_keyframe.emotion, &next_keyframe.emotion, eased_t)
    }

    /// Interpolate between two emotion vectors
    fn interpolate_emotions(
        &self,
        from: &EmotionVector,
        to: &EmotionVector,
        t: f32,
    ) -> EmotionVector {
        let mut result = EmotionVector::new();

        // Get all unique emotions from both vectors
        let mut all_emotions: Vec<Emotion> = Vec::new();

        // Collect from 'from' vector
        for emotion in from.emotions.keys() {
            if !all_emotions.contains(emotion) {
                all_emotions.push(emotion.clone());
            }
        }

        // Collect from 'to' vector
        for emotion in to.emotions.keys() {
            if !all_emotions.contains(emotion) {
                all_emotions.push(emotion.clone());
            }
        }

        // Interpolate each emotion
        for emotion in all_emotions {
            let from_val = from
                .emotions
                .get(&emotion)
                .map(|i| i.value())
                .unwrap_or(0.0);
            let to_val = to.emotions.get(&emotion).map(|i| i.value()).unwrap_or(0.0);

            let interpolated = from_val * (1.0 - t) + to_val * t;
            result.add_emotion(emotion, EmotionIntensity::new(interpolated));
        }

        result
    }

    /// Get the duration of the trajectory
    pub fn duration(&self) -> f32 {
        self.duration
    }

    /// Get the number of keyframes
    pub fn num_keyframes(&self) -> usize {
        self.keyframes.len()
    }
}

/// Apply easing function to interpolation parameter
fn apply_easing(t: f32, easing: EasingFunction) -> f32 {
    let t = t.clamp(0.0, 1.0);

    match easing {
        EasingFunction::Linear => t,
        EasingFunction::EaseIn => t * t,
        EasingFunction::EaseOut => t * (2.0 - t),
        EasingFunction::EaseInOut => {
            if t < 0.5 {
                2.0 * t * t
            } else {
                -1.0 + (4.0 - 2.0 * t) * t
            }
        }
        EasingFunction::Elastic => {
            if t == 0.0 || t == 1.0 {
                t
            } else {
                let p = 0.3;
                let s = p / 4.0;
                let post = 2.0f32.powf(10.0 * (t - 1.0));
                let post_sin = ((t - 1.0 - s) * (2.0 * std::f32::consts::PI) / p).sin();
                -(post * post_sin) + 1.0
            }
        }
        EasingFunction::Bounce => {
            let t = 1.0 - t;
            let result = if t < 1.0 / 2.75 {
                7.5625 * t * t
            } else if t < 2.0 / 2.75 {
                let t = t - (1.5 / 2.75);
                7.5625 * t * t + 0.75
            } else if t < 2.5 / 2.75 {
                let t = t - (2.25 / 2.75);
                7.5625 * t * t + 0.9375
            } else {
                let t = t - (2.625 / 2.75);
                7.5625 * t * t + 0.984375
            };
            1.0 - result
        }
    }
}

/// Multi-emotion blender for complex morphing
#[derive(Debug)]
pub struct EmotionBlender {
    /// Configuration
    config: EmotionMorphConfig,
    /// Current blend state
    current_blend: EmotionVector,
}

impl EmotionBlender {
    /// Create a new emotion blender
    pub fn new(config: EmotionMorphConfig) -> Self {
        Self {
            config,
            current_blend: EmotionVector::new(),
        }
    }

    /// Blend multiple emotions with individual weights
    pub fn blend(&mut self, emotions: &[(Emotion, f32)]) -> EmotionVector {
        let mut result = EmotionVector::new();

        // Normalize weights
        let total_weight: f32 = emotions.iter().map(|(_, w)| w).sum();

        if total_weight > 0.0 {
            for &(ref emotion, weight) in emotions {
                let normalized_weight = weight / total_weight;
                result.add_emotion(emotion.clone(), EmotionIntensity::new(normalized_weight));
            }
        }

        // Apply smoothing if configured
        if self.config.smoothing > 0.0 {
            result = self.smooth_blend(&result);
        }

        self.current_blend = result.clone();
        result
    }

    /// Apply smoothing to blend transition
    fn smooth_blend(&self, target: &EmotionVector) -> EmotionVector {
        let mut result = EmotionVector::new();
        let alpha = 1.0 - self.config.smoothing;

        // Get all unique emotions
        let mut all_emotions: Vec<Emotion> = Vec::new();

        for emotion in self.current_blend.emotions.keys() {
            if !all_emotions.contains(emotion) {
                all_emotions.push(emotion.clone());
            }
        }

        for emotion in target.emotions.keys() {
            if !all_emotions.contains(emotion) {
                all_emotions.push(emotion.clone());
            }
        }

        // Smooth each emotion
        for emotion in all_emotions {
            let current_val = self
                .current_blend
                .emotions
                .get(&emotion)
                .map(|i| i.value())
                .unwrap_or(0.0);
            let target_val = target
                .emotions
                .get(&emotion)
                .map(|i| i.value())
                .unwrap_or(0.0);

            let smoothed = current_val * (1.0 - alpha) + target_val * alpha;
            result.add_emotion(emotion, EmotionIntensity::new(smoothed));
        }

        result
    }

    /// Get current blend state
    pub fn current_blend(&self) -> &EmotionVector {
        &self.current_blend
    }
}

/// Bezier curve for smooth emotion transitions
#[derive(Debug, Clone)]
pub struct EmotionBezierCurve {
    /// Start emotion
    start: EmotionVector,
    /// Control point 1
    control1: EmotionVector,
    /// Control point 2
    control2: EmotionVector,
    /// End emotion
    end: EmotionVector,
}

impl EmotionBezierCurve {
    /// Create a new Bezier curve
    pub fn new(
        start: EmotionVector,
        control1: EmotionVector,
        control2: EmotionVector,
        end: EmotionVector,
    ) -> Self {
        Self {
            start,
            control1,
            control2,
            end,
        }
    }

    /// Sample the curve at parameter t (0.0-1.0)
    pub fn sample(&self, t: f32) -> EmotionVector {
        let t = t.clamp(0.0, 1.0);
        let t2 = t * t;
        let t3 = t2 * t;

        let one_minus_t = 1.0 - t;
        let one_minus_t2 = one_minus_t * one_minus_t;
        let one_minus_t3 = one_minus_t2 * one_minus_t;

        // Cubic Bezier formula: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
        let b0 = one_minus_t3;
        let b1 = 3.0 * one_minus_t2 * t;
        let b2 = 3.0 * one_minus_t * t2;
        let b3 = t3;

        self.blend_emotions(
            [&self.start, &self.control1, &self.control2, &self.end],
            [b0, b1, b2, b3],
        )
    }

    /// Blend emotions using Bezier weights
    fn blend_emotions(&self, emotions: [&EmotionVector; 4], weights: [f32; 4]) -> EmotionVector {
        let mut result = EmotionVector::new();
        let [e0, e1, e2, e3] = emotions;
        let [w0, w1, w2, w3] = weights;

        // Collect all unique emotions
        let mut all_emotions: Vec<Emotion> = Vec::new();
        for vector in &[e0, e1, e2, e3] {
            for emotion in vector.emotions.keys() {
                if !all_emotions.contains(emotion) {
                    all_emotions.push(emotion.clone());
                }
            }
        }

        // Blend each emotion
        for emotion in all_emotions {
            let v0 = e0.emotions.get(&emotion).map(|i| i.value()).unwrap_or(0.0);
            let v1 = e1.emotions.get(&emotion).map(|i| i.value()).unwrap_or(0.0);
            let v2 = e2.emotions.get(&emotion).map(|i| i.value()).unwrap_or(0.0);
            let v3 = e3.emotions.get(&emotion).map(|i| i.value()).unwrap_or(0.0);

            let blended = w0 * v0 + w1 * v1 + w2 * v2 + w3 * v3;
            result.add_emotion(emotion, EmotionIntensity::new(blended));
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_trajectory() {
        let mut trajectory = EmotionTrajectory::new(2.0);

        let mut emotion1 = EmotionVector::new();
        emotion1.add_emotion(Emotion::Happy, EmotionIntensity::new(1.0));

        let mut emotion2 = EmotionVector::new();
        emotion2.add_emotion(Emotion::Sad, EmotionIntensity::new(1.0));

        trajectory.add_keyframe(0.0, emotion1);
        trajectory.add_keyframe(2.0, emotion2);

        assert_eq!(trajectory.num_keyframes(), 2);

        let mid_emotion = trajectory.sample(1.0);
        assert!(
            mid_emotion
                .emotions
                .get(&Emotion::Happy)
                .map(|i| i.value())
                .unwrap_or(0.0)
                > 0.0
        );
        assert!(
            mid_emotion
                .emotions
                .get(&Emotion::Sad)
                .map(|i| i.value())
                .unwrap_or(0.0)
                > 0.0
        );
    }

    #[test]
    fn test_easing_functions() {
        let linear = apply_easing(0.5, EasingFunction::Linear);
        assert!((linear - 0.5).abs() < 1e-5);

        let ease_in = apply_easing(0.5, EasingFunction::EaseIn);
        assert!(ease_in < 0.5); // Should be slower at start

        let ease_out = apply_easing(0.5, EasingFunction::EaseOut);
        assert!(ease_out > 0.5); // Should be faster at start
    }

    #[test]
    fn test_emotion_blender() {
        let mut config = EmotionMorphConfig::default();
        config.smoothing = 0.0; // Disable smoothing for this test
        let mut blender = EmotionBlender::new(config);

        let emotions = vec![(Emotion::Happy, 0.6), (Emotion::Excited, 0.4)];

        let blend = blender.blend(&emotions);

        assert!(
            (blend
                .emotions
                .get(&Emotion::Happy)
                .map(|i| i.value())
                .unwrap_or(0.0)
                - 0.6)
                .abs()
                < 1e-5
        );
        assert!(
            (blend
                .emotions
                .get(&Emotion::Excited)
                .map(|i| i.value())
                .unwrap_or(0.0)
                - 0.4)
                .abs()
                < 1e-5
        );
    }

    #[test]
    fn test_bezier_curve() {
        let mut start = EmotionVector::new();
        start.add_emotion(Emotion::Neutral, EmotionIntensity::new(1.0));

        let mut end = EmotionVector::new();
        end.add_emotion(Emotion::Happy, EmotionIntensity::new(1.0));

        let curve = EmotionBezierCurve::new(start.clone(), start.clone(), end.clone(), end.clone());

        let sample_start = curve.sample(0.0);
        let sample_mid = curve.sample(0.5);
        let sample_end = curve.sample(1.0);

        assert!(
            (sample_start
                .emotions
                .get(&Emotion::Neutral)
                .map(|i| i.value())
                .unwrap_or(0.0)
                - 1.0)
                .abs()
                < 1e-5
        );
        assert!(
            (sample_end
                .emotions
                .get(&Emotion::Happy)
                .map(|i| i.value())
                .unwrap_or(0.0)
                - 1.0)
                .abs()
                < 1e-5
        );
    }

    #[test]
    fn test_trajectory_with_easing() {
        let mut trajectory = EmotionTrajectory::new(1.0);

        let mut emotion1 = EmotionVector::new();
        emotion1.add_emotion(Emotion::Calm, EmotionIntensity::new(1.0));

        let mut emotion2 = EmotionVector::new();
        emotion2.add_emotion(Emotion::Excited, EmotionIntensity::new(1.0));

        trajectory.add_keyframe_with_easing(0.0, emotion1, Some(EasingFunction::EaseIn));
        trajectory.add_keyframe(1.0, emotion2);

        let early = trajectory.sample(0.25);
        let late = trajectory.sample(0.75);

        // With ease-in, early should have more Calm, late more Excited
        let early_calm = early
            .emotions
            .get(&Emotion::Calm)
            .map(|i| i.value())
            .unwrap_or(0.0);
        let late_calm = late
            .emotions
            .get(&Emotion::Calm)
            .map(|i| i.value())
            .unwrap_or(0.0);
        assert!(early_calm > late_calm);
    }
}
