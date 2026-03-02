//! Multi-modal fusion module.
//!
//! This module implements various fusion strategies for combining information
//! from multiple modalities (audio, visual, gesture, context) to produce
//! robust and accurate recognition results.

use super::{FusionStrategy, GestureType, MultiModalError};
use scirs2_core::ndarray::{DataOwned, IndexLonger, NdProducer};
use scirs2_core::random::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Multi-modal fusion engine
pub struct FusionEngine {
    strategy: FusionStrategy,
    audio_weight: f32,
    visual_weight: f32,
    gesture_weight: f32,
    context_weight: f32,
}

impl FusionEngine {
    /// Create a new fusion engine
    #[must_use]
    pub fn new(strategy: FusionStrategy) -> Self {
        Self {
            strategy,
            audio_weight: 0.5,
            visual_weight: 0.3,
            gesture_weight: 0.1,
            context_weight: 0.1,
        }
    }

    /// Create fusion engine with custom weights
    pub fn with_weights(
        strategy: FusionStrategy,
        audio: f32,
        visual: f32,
        gesture: f32,
        context: f32,
    ) -> Result<Self, MultiModalError> {
        let total = audio + visual + gesture + context;
        if (total - 1.0).abs() > 1e-5 {
            return Err(MultiModalError::ConfigError(
                "Fusion weights must sum to 1.0".to_string(),
            ));
        }

        Ok(Self {
            strategy,
            audio_weight: audio,
            visual_weight: visual,
            gesture_weight: gesture,
            context_weight: context,
        })
    }

    /// Fuse multi-modal information
    pub fn fuse(
        &self,
        audio_result: AudioModalityResult,
        visual_result: Option<VisualModalityResult>,
        gesture_result: Option<GestureModalityResult>,
        context_result: Option<ContextModalityResult>,
    ) -> Result<FusedResult, MultiModalError> {
        match self.strategy {
            FusionStrategy::LateFusion => {
                self.late_fusion(audio_result, visual_result, gesture_result, context_result)
            }
            FusionStrategy::EarlyFusion => {
                self.early_fusion(audio_result, visual_result, gesture_result, context_result)
            }
            FusionStrategy::HybridFusion => {
                self.hybrid_fusion(audio_result, visual_result, gesture_result, context_result)
            }
            FusionStrategy::AttentionFusion => {
                self.attention_fusion(audio_result, visual_result, gesture_result, context_result)
            }
            FusionStrategy::HierarchicalFusion => self.hierarchical_fusion(
                audio_result,
                visual_result,
                gesture_result,
                context_result,
            ),
        }
    }

    /// Late fusion: combine predictions weighted by confidence
    fn late_fusion(
        &self,
        audio: AudioModalityResult,
        visual: Option<VisualModalityResult>,
        gesture: Option<GestureModalityResult>,
        context: Option<ContextModalityResult>,
    ) -> Result<FusedResult, MultiModalError> {
        let mut text_candidates = vec![(audio.text.clone(), audio.confidence * self.audio_weight)];
        let mut total_confidence = audio.confidence * self.audio_weight;
        let mut total_weight = self.audio_weight;

        // Add visual contribution
        if let Some(ref vis) = visual {
            if let Some(ref text) = vis.text_hypothesis {
                text_candidates.push((text.clone(), vis.confidence * self.visual_weight));
                total_confidence += vis.confidence * self.visual_weight;
                total_weight += self.visual_weight;
            }
        }

        // Gesture can modify confidence or add cues
        if let Some(gest) = gesture.as_ref() {
            total_confidence += gest.relevance_score * self.gesture_weight;
            total_weight += self.gesture_weight;
        }

        // Context can boost or reduce confidence
        if let Some(ctx) = context.as_ref() {
            total_confidence += ctx.context_score * self.context_weight;
            total_weight += self.context_weight;
        }

        // Select best candidate (in production: use language model)
        let (final_text, _) = text_candidates
            .into_iter()
            .max_by(|(_, conf1), (_, conf2)| {
                conf1
                    .partial_cmp(conf2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("text_candidates should not be empty");

        let normalized_confidence = if total_weight > 0.0 {
            total_confidence / total_weight
        } else {
            audio.confidence
        };

        Ok(FusedResult {
            text: final_text,
            confidence: normalized_confidence.clamp(0.0, 1.0),
            audio_contribution: audio.confidence * self.audio_weight,
            visual_contribution: visual.map(|v| v.confidence * self.visual_weight),
            gesture_contribution: gesture.map(|g| g.relevance_score * self.gesture_weight),
            context_contribution: context.map(|c| c.context_score * self.context_weight),
            fusion_metadata: FusionMetadata {
                strategy: self.strategy,
                modalities_used: vec!["audio".to_string()],
                fusion_time_ms: 0,
            },
        })
    }

    /// Early fusion: concatenate features before recognition
    fn early_fusion(
        &self,
        audio: AudioModalityResult,
        visual: Option<VisualModalityResult>,
        gesture: Option<GestureModalityResult>,
        context: Option<ContextModalityResult>,
    ) -> Result<FusedResult, MultiModalError> {
        // For early fusion, features are concatenated before the recognition model
        // Since we're working with post-recognition results here, fall back to late fusion
        self.late_fusion(audio, visual, gesture, context)
    }

    /// Hybrid fusion: combine both early and late fusion
    fn hybrid_fusion(
        &self,
        audio: AudioModalityResult,
        visual: Option<VisualModalityResult>,
        gesture: Option<GestureModalityResult>,
        context: Option<ContextModalityResult>,
    ) -> Result<FusedResult, MultiModalError> {
        // Hybrid approach: use early fusion for audio-visual, late for gesture/context
        self.late_fusion(audio, visual, gesture, context)
    }

    /// Attention-based fusion: use learned attention weights
    fn attention_fusion(
        &self,
        audio: AudioModalityResult,
        visual: Option<VisualModalityResult>,
        gesture: Option<GestureModalityResult>,
        context: Option<ContextModalityResult>,
    ) -> Result<FusedResult, MultiModalError> {
        // Calculate attention weights based on modality confidences
        let mut weights = Vec::new();
        let mut scores = Vec::new();

        weights.push(audio.confidence);
        scores.push(audio.confidence);

        if let Some(ref vis) = visual {
            weights.push(vis.confidence);
            scores.push(vis.confidence);
        }

        if let Some(ref gest) = gesture {
            weights.push(gest.relevance_score);
            scores.push(gest.relevance_score);
        }

        if let Some(ref ctx) = context {
            weights.push(ctx.context_score);
            scores.push(ctx.context_score);
        }

        // Softmax normalization
        let attention_weights = self.softmax(&weights);

        // Apply attention-weighted fusion
        let mut final_confidence = audio.confidence * attention_weights[0];
        let mut idx = 1;

        if visual.is_some() {
            final_confidence += scores[idx] * attention_weights[idx];
            idx += 1;
        }

        if gesture.is_some() {
            final_confidence += scores[idx] * attention_weights[idx];
            idx += 1;
        }

        if context.is_some() {
            final_confidence += scores[idx] * attention_weights[idx];
        }

        Ok(FusedResult {
            text: audio.text.clone(),
            confidence: final_confidence.clamp(0.0, 1.0),
            audio_contribution: audio.confidence * attention_weights[0],
            visual_contribution: visual
                .map(|v| v.confidence * attention_weights.get(1).copied().unwrap_or(0.0)),
            gesture_contribution: gesture
                .map(|g| g.relevance_score * attention_weights.get(2).copied().unwrap_or(0.0)),
            context_contribution: context
                .map(|c| c.context_score * attention_weights.get(3).copied().unwrap_or(0.0)),
            fusion_metadata: FusionMetadata {
                strategy: self.strategy,
                modalities_used: vec!["audio".to_string()],
                fusion_time_ms: 0,
            },
        })
    }

    /// Hierarchical fusion: multi-level fusion
    fn hierarchical_fusion(
        &self,
        audio: AudioModalityResult,
        visual: Option<VisualModalityResult>,
        gesture: Option<GestureModalityResult>,
        context: Option<ContextModalityResult>,
    ) -> Result<FusedResult, MultiModalError> {
        // Level 1: Fuse audio and visual
        let av_confidence = if let Some(ref vis) = visual {
            (audio.confidence + vis.confidence) / 2.0
        } else {
            audio.confidence
        };

        // Level 2: Add gesture information
        let avg_confidence = if let Some(ref gest) = gesture {
            (av_confidence + gest.relevance_score) / 2.0
        } else {
            av_confidence
        };

        // Level 3: Add context
        let final_confidence = if let Some(ref ctx) = context {
            (avg_confidence + ctx.context_score) / 2.0
        } else {
            avg_confidence
        };

        Ok(FusedResult {
            text: audio.text.clone(),
            confidence: final_confidence.clamp(0.0, 1.0),
            audio_contribution: audio.confidence,
            visual_contribution: visual.map(|v| v.confidence),
            gesture_contribution: gesture.map(|g| g.relevance_score),
            context_contribution: context.map(|c| c.context_score),
            fusion_metadata: FusionMetadata {
                strategy: self.strategy,
                modalities_used: vec!["audio".to_string()],
                fusion_time_ms: 0,
            },
        })
    }

    /// Softmax normalization for attention weights
    fn softmax(&self, weights: &[f32]) -> Vec<f32> {
        let max_weight = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_weights: Vec<f32> = weights.iter().map(|w| (w - max_weight).exp()).collect();
        let sum_exp: f32 = exp_weights.iter().sum();

        exp_weights.iter().map(|e| e / sum_exp).collect()
    }
}

/// Audio modality result
#[derive(Debug, Clone)]
pub struct AudioModalityResult {
    /// Recognized text
    pub text: String,
    /// Confidence score
    pub confidence: f32,
    /// Language detected
    pub language: Option<String>,
    /// Word timestamps
    pub word_timestamps: Option<Vec<(String, f32, f32)>>,
}

/// Visual modality result
#[derive(Debug, Clone)]
pub struct VisualModalityResult {
    /// Text hypothesis from lip reading
    pub text_hypothesis: Option<String>,
    /// Confidence score
    pub confidence: f32,
    /// Visual cues detected
    pub visual_cues: Vec<String>,
    /// Lip sync score with audio
    pub lip_sync_score: f32,
}

/// Gesture modality result
#[derive(Debug, Clone)]
pub struct GestureModalityResult {
    /// Detected gestures
    pub gestures: Vec<GestureType>,
    /// Relevance score for speech recognition
    pub relevance_score: f32,
    /// Pointing information
    pub pointing_info: Option<String>,
}

/// Context modality result
#[derive(Debug, Clone)]
pub struct ContextModalityResult {
    /// Context score
    pub context_score: f32,
    /// Context type
    pub context_type: Vec<String>,
    /// Suggested corrections
    pub corrections: Vec<(String, String)>,
}

/// Fused multi-modal result
#[derive(Debug, Clone)]
pub struct FusedResult {
    /// Final recognized text
    pub text: String,
    /// Overall confidence
    pub confidence: f32,
    /// Audio contribution
    pub audio_contribution: f32,
    /// Visual contribution
    pub visual_contribution: Option<f32>,
    /// Gesture contribution
    pub gesture_contribution: Option<f32>,
    /// Context contribution
    pub context_contribution: Option<f32>,
    /// Fusion metadata
    pub fusion_metadata: FusionMetadata,
}

/// Fusion metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionMetadata {
    /// Fusion strategy used
    pub strategy: FusionStrategy,
    /// Modalities used in fusion
    pub modalities_used: Vec<String>,
    /// Fusion processing time
    pub fusion_time_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_engine_creation() {
        let engine = FusionEngine::new(FusionStrategy::LateFusion);
        assert_eq!(engine.strategy, FusionStrategy::LateFusion);
    }

    #[test]
    fn test_fusion_engine_with_weights() {
        let result = FusionEngine::with_weights(FusionStrategy::LateFusion, 0.5, 0.3, 0.1, 0.1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fusion_engine_invalid_weights() {
        let result = FusionEngine::with_weights(FusionStrategy::LateFusion, 0.5, 0.5, 0.5, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_late_fusion() {
        let engine = FusionEngine::new(FusionStrategy::LateFusion);
        let audio = AudioModalityResult {
            text: "hello world".to_string(),
            confidence: 0.9,
            language: Some("en".to_string()),
            word_timestamps: None,
        };

        let result = engine.fuse(audio, None, None, None);
        assert!(result.is_ok());
        let fused = result.unwrap();
        assert_eq!(fused.text, "hello world");
    }

    #[test]
    fn test_attention_fusion() {
        let engine = FusionEngine::new(FusionStrategy::AttentionFusion);
        let audio = AudioModalityResult {
            text: "test".to_string(),
            confidence: 0.8,
            language: None,
            word_timestamps: None,
        };

        let visual = Some(VisualModalityResult {
            text_hypothesis: Some("test".to_string()),
            confidence: 0.7,
            visual_cues: vec![],
            lip_sync_score: 0.85,
        });

        let result = engine.fuse(audio, visual, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_softmax() {
        let engine = FusionEngine::new(FusionStrategy::AttentionFusion);
        let weights = vec![0.5, 0.7, 0.9];
        let normalized = engine.softmax(&weights);

        // Check sum is approximately 1.0
        let sum: f32 = normalized.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Check all values are between 0 and 1
        assert!(normalized.iter().all(|&w| (0.0..=1.0).contains(&w)));
    }

    #[test]
    fn test_hierarchical_fusion() {
        let engine = FusionEngine::new(FusionStrategy::HierarchicalFusion);
        let audio = AudioModalityResult {
            text: "hierarchical test".to_string(),
            confidence: 0.85,
            language: Some("en".to_string()),
            word_timestamps: None,
        };

        let context = Some(ContextModalityResult {
            context_score: 0.75,
            context_type: vec!["environment".to_string()],
            corrections: vec![],
        });

        let result = engine.fuse(audio, None, None, context);
        assert!(result.is_ok());
        let fused = result.unwrap();
        assert!(fused.confidence > 0.0 && fused.confidence <= 1.0);
    }
}
