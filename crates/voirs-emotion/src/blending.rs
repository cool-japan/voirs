//! Advanced Emotion Blending
//!
//! This module provides sophisticated emotion blending capabilities that allow
//! combining multiple emotions with customizable weights and blending modes.
//!
//! ## Features
//!
//! - **Multi-Emotion Blending**: Combine multiple emotions simultaneously
//! - **Weighted Blending**: Control the contribution of each emotion
//! - **Blending Modes**: Different algorithms for combining emotions
//! - **Normalization**: Automatic weight normalization
//! - **Transition Smoothing**: Smooth transitions between emotion blends
//!
//! ## Example Usage
//!
//! ```rust
//! use voirs_emotion::blending::{EmotionBlender, EmotionBlend, BlendMode};
//! use voirs_emotion::types::Emotion;
//!
//! // Create a blend of happy and excited emotions
//! let mut blend = EmotionBlend::new();
//! blend.add_emotion(Emotion::Happy, 0.7);
//! blend.add_emotion(Emotion::Excited, 0.3);
//!
//! // Blend the emotions
//! let blender = EmotionBlender::new(BlendMode::Weighted);
//! let result = blender.blend(&blend)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::{
    types::{Emotion, EmotionDimensions, EmotionIntensity, EmotionVector},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Blending mode for combining emotions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlendMode {
    /// Weighted average based on emotion weights
    Weighted,
    /// Maximum intensity across all emotions
    Maximum,
    /// Minimum intensity across all emotions
    Minimum,
    /// Additive blending (sum with clamping)
    Additive,
    /// Multiplicative blending
    Multiplicative,
    /// Harmonic mean blending
    Harmonic,
}

/// A collection of emotions with associated weights for blending
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionBlend {
    /// Map of emotions to their weights
    emotions: HashMap<Emotion, f32>,
    /// Whether to normalize weights automatically
    auto_normalize: bool,
}

impl EmotionBlend {
    /// Create a new empty emotion blend
    pub fn new() -> Self {
        Self {
            emotions: HashMap::new(),
            auto_normalize: true,
        }
    }

    /// Create a blend with auto-normalization disabled
    pub fn new_unnormalized() -> Self {
        Self {
            emotions: HashMap::new(),
            auto_normalize: false,
        }
    }

    /// Add an emotion with a weight
    pub fn add_emotion(&mut self, emotion: Emotion, weight: f32) {
        self.emotions.insert(emotion, weight.max(0.0));
    }

    /// Remove an emotion from the blend
    pub fn remove_emotion(&mut self, emotion: &Emotion) {
        self.emotions.remove(emotion);
    }

    /// Get the weight of an emotion
    pub fn get_weight(&self, emotion: &Emotion) -> Option<f32> {
        self.emotions.get(emotion).copied()
    }

    /// Get all emotions in the blend
    pub fn emotions(&self) -> Vec<Emotion> {
        self.emotions.keys().cloned().collect()
    }

    /// Get the number of emotions in the blend
    pub fn len(&self) -> usize {
        self.emotions.len()
    }

    /// Check if the blend is empty
    pub fn is_empty(&self) -> bool {
        self.emotions.is_empty()
    }

    /// Normalize weights so they sum to 1.0
    pub fn normalize(&mut self) {
        let total: f32 = self.emotions.values().sum();
        if total > 0.0 {
            for weight in self.emotions.values_mut() {
                *weight /= total;
            }
        }
    }

    /// Get normalized weights (without modifying the blend)
    pub fn normalized_weights(&self) -> HashMap<Emotion, f32> {
        let total: f32 = self.emotions.values().sum();
        if total > 0.0 {
            self.emotions
                .iter()
                .map(|(e, w)| (e.clone(), w / total))
                .collect()
        } else {
            self.emotions.clone()
        }
    }

    /// Clear all emotions from the blend
    pub fn clear(&mut self) {
        self.emotions.clear();
    }
}

impl Default for EmotionBlend {
    fn default() -> Self {
        Self::new()
    }
}

/// Emotion blender that combines multiple emotions using various algorithms
pub struct EmotionBlender {
    mode: BlendMode,
}

impl EmotionBlender {
    /// Create a new emotion blender with the specified mode
    pub fn new(mode: BlendMode) -> Self {
        Self { mode }
    }

    /// Blend multiple emotions into a single emotion vector
    pub fn blend(&self, blend: &EmotionBlend) -> Result<EmotionVector> {
        if blend.is_empty() {
            return Ok(EmotionVector::new());
        }

        match self.mode {
            BlendMode::Weighted => self.blend_weighted(blend),
            BlendMode::Maximum => self.blend_maximum(blend),
            BlendMode::Minimum => self.blend_minimum(blend),
            BlendMode::Additive => self.blend_additive(blend),
            BlendMode::Multiplicative => self.blend_multiplicative(blend),
            BlendMode::Harmonic => self.blend_harmonic(blend),
        }
    }

    /// Blend using weighted average
    fn blend_weighted(&self, blend: &EmotionBlend) -> Result<EmotionVector> {
        let weights = if blend.auto_normalize {
            blend.normalized_weights()
        } else {
            blend.emotions.clone()
        };

        let mut result = EmotionVector::new();

        // Blend each emotion with its weight
        for (emotion, weight) in weights {
            result.add_emotion(emotion.clone(), EmotionIntensity::new(weight));
        }

        Ok(result)
    }

    /// Blend using maximum intensity
    fn blend_maximum(&self, blend: &EmotionBlend) -> Result<EmotionVector> {
        let mut result = EmotionVector::new();

        // Find the emotion with maximum weight
        if let Some((max_emotion, max_weight)) = blend
            .emotions
            .iter()
            .max_by(|(_, w1), (_, w2)| w1.partial_cmp(w2).unwrap_or(std::cmp::Ordering::Equal))
        {
            result.add_emotion(max_emotion.clone(), EmotionIntensity::new(*max_weight));
        }

        Ok(result)
    }

    /// Blend using minimum intensity
    fn blend_minimum(&self, blend: &EmotionBlend) -> Result<EmotionVector> {
        let mut result = EmotionVector::new();

        // Find the emotion with minimum weight
        if let Some((min_emotion, min_weight)) = blend
            .emotions
            .iter()
            .min_by(|(_, w1), (_, w2)| w1.partial_cmp(w2).unwrap_or(std::cmp::Ordering::Equal))
        {
            result.add_emotion(min_emotion.clone(), EmotionIntensity::new(*min_weight));
        }

        Ok(result)
    }

    /// Blend using additive combination
    fn blend_additive(&self, blend: &EmotionBlend) -> Result<EmotionVector> {
        let mut result = EmotionVector::new();

        // Add all emotions, clamping total intensity to 1.0
        for (emotion, weight) in &blend.emotions {
            let clamped_weight = weight.min(1.0);
            result.add_emotion(emotion.clone(), EmotionIntensity::new(clamped_weight));
        }

        Ok(result)
    }

    /// Blend using multiplicative combination
    fn blend_multiplicative(&self, blend: &EmotionBlend) -> Result<EmotionVector> {
        let mut result = EmotionVector::new();

        // Multiply intensities (geometric mean)
        for (emotion, weight) in &blend.emotions {
            let adjusted_weight = weight.powf(1.0 / blend.len() as f32);
            result.add_emotion(emotion.clone(), EmotionIntensity::new(adjusted_weight));
        }

        Ok(result)
    }

    /// Blend using harmonic mean
    fn blend_harmonic(&self, blend: &EmotionBlend) -> Result<EmotionVector> {
        let mut result = EmotionVector::new();

        if blend.is_empty() {
            return Ok(result);
        }

        // Calculate harmonic mean
        let reciprocal_sum: f32 = blend
            .emotions
            .values()
            .map(|w| if *w > 0.0 { 1.0 / w } else { 0.0 })
            .sum();

        let harmonic_mean = if reciprocal_sum > 0.0 {
            blend.len() as f32 / reciprocal_sum
        } else {
            0.0
        };

        // Apply harmonic mean intensity to all emotions
        for emotion in blend.emotions.keys() {
            result.add_emotion(emotion.clone(), EmotionIntensity::new(harmonic_mean));
        }

        Ok(result)
    }

    /// Get the current blending mode
    pub fn mode(&self) -> BlendMode {
        self.mode
    }

    /// Change the blending mode
    pub fn set_mode(&mut self, mode: BlendMode) {
        self.mode = mode;
    }
}

impl Default for EmotionBlender {
    fn default() -> Self {
        Self::new(BlendMode::Weighted)
    }
}

/// Builder for creating complex emotion blends
pub struct EmotionBlendBuilder {
    blend: EmotionBlend,
}

impl EmotionBlendBuilder {
    /// Create a new emotion blend builder
    pub fn new() -> Self {
        Self {
            blend: EmotionBlend::new(),
        }
    }

    /// Add an emotion with a weight
    pub fn with_emotion(mut self, emotion: Emotion, weight: f32) -> Self {
        self.blend.add_emotion(emotion, weight);
        self
    }

    /// Set auto-normalization
    pub fn auto_normalize(mut self, enable: bool) -> Self {
        self.blend.auto_normalize = enable;
        self
    }

    /// Build the emotion blend
    pub fn build(self) -> EmotionBlend {
        self.blend
    }
}

impl Default for EmotionBlendBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_blend_creation() {
        let mut blend = EmotionBlend::new();
        blend.add_emotion(Emotion::Happy, 0.7);
        blend.add_emotion(Emotion::Excited, 0.3);

        assert_eq!(blend.len(), 2);
        assert!(!blend.is_empty());
        assert_eq!(blend.get_weight(&Emotion::Happy), Some(0.7));
        assert_eq!(blend.get_weight(&Emotion::Excited), Some(0.3));
    }

    #[test]
    fn test_emotion_blend_normalization() {
        let mut blend = EmotionBlend::new();
        blend.add_emotion(Emotion::Happy, 2.0);
        blend.add_emotion(Emotion::Sad, 1.0);

        blend.normalize();

        assert!((blend.get_weight(&Emotion::Happy).unwrap() - 0.666).abs() < 0.01);
        assert!((blend.get_weight(&Emotion::Sad).unwrap() - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_weighted_blending() -> Result<()> {
        let mut blend = EmotionBlend::new();
        blend.add_emotion(Emotion::Happy, 0.7);
        blend.add_emotion(Emotion::Calm, 0.3);

        let blender = EmotionBlender::new(BlendMode::Weighted);
        let result = blender.blend(&blend)?;

        // Result should have non-default dimensions due to blending
        assert!(result.dimensions.valence != 0.0 || result.dimensions.arousal != 0.0);

        Ok(())
    }

    #[test]
    fn test_maximum_blending() -> Result<()> {
        let mut blend = EmotionBlend::new();
        blend.add_emotion(Emotion::Happy, 0.3);
        blend.add_emotion(Emotion::Excited, 0.8);

        let blender = EmotionBlender::new(BlendMode::Maximum);
        let result = blender.blend(&blend)?;

        // Should select the emotion with maximum weight
        let dominant = result.dominant_emotion();
        assert!(dominant.is_some());

        Ok(())
    }

    #[test]
    fn test_minimum_blending() -> Result<()> {
        let mut blend = EmotionBlend::new();
        blend.add_emotion(Emotion::Happy, 0.8);
        blend.add_emotion(Emotion::Calm, 0.3);

        let blender = EmotionBlender::new(BlendMode::Minimum);
        let result = blender.blend(&blend)?;

        // Should select the emotion with minimum weight
        let dominant = result.dominant_emotion();
        assert!(dominant.is_some());

        Ok(())
    }

    #[test]
    fn test_additive_blending() -> Result<()> {
        let mut blend = EmotionBlend::new();
        blend.add_emotion(Emotion::Happy, 0.5);
        blend.add_emotion(Emotion::Excited, 0.6);

        let blender = EmotionBlender::new(BlendMode::Additive);
        let result = blender.blend(&blend)?;

        // Result should have non-zero dimensions
        assert!(result.dimensions.arousal > 0.0);

        Ok(())
    }

    #[test]
    fn test_empty_blend() -> Result<()> {
        let blend = EmotionBlend::new();
        let blender = EmotionBlender::new(BlendMode::Weighted);
        let result = blender.blend(&blend)?;

        // Empty blend should have no dominant emotion
        assert!(result.dominant_emotion().is_none());

        Ok(())
    }

    #[test]
    fn test_blend_builder() {
        let blend = EmotionBlendBuilder::new()
            .with_emotion(Emotion::Happy, 0.6)
            .with_emotion(Emotion::Excited, 0.4)
            .auto_normalize(true)
            .build();

        assert_eq!(blend.len(), 2);
        assert_eq!(blend.get_weight(&Emotion::Happy), Some(0.6));
    }

    #[test]
    fn test_harmonic_blending() -> Result<()> {
        let mut blend = EmotionBlend::new();
        blend.add_emotion(Emotion::Happy, 0.5);
        blend.add_emotion(Emotion::Calm, 0.5);

        let blender = EmotionBlender::new(BlendMode::Harmonic);
        let result = blender.blend(&blend)?;

        // Harmonic mean should be applied and produce a valid result
        assert!(result.dominant_emotion().is_some());

        Ok(())
    }

    #[test]
    fn test_remove_emotion() {
        let mut blend = EmotionBlend::new();
        blend.add_emotion(Emotion::Happy, 0.5);
        blend.add_emotion(Emotion::Sad, 0.5);

        blend.remove_emotion(&Emotion::Sad);

        assert_eq!(blend.len(), 1);
        assert_eq!(blend.get_weight(&Emotion::Sad), None);
    }

    #[test]
    fn test_clear_blend() {
        let mut blend = EmotionBlend::new();
        blend.add_emotion(Emotion::Happy, 0.5);
        blend.add_emotion(Emotion::Sad, 0.5);

        blend.clear();

        assert!(blend.is_empty());
        assert_eq!(blend.len(), 0);
    }
}
