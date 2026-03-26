// Copyright (c) 2025 VoiRS Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Conversational quality assessment for dialogue and interactive speech systems.
//!
//! This module provides comprehensive evaluation capabilities for conversational AI
//! and dialogue systems, assessing turn-taking, response quality, coherence, and
//! natural conversation flow.
//!
//! # Features
//!
//! - **Turn-Taking Analysis**: Evaluation of appropriate turn transitions and timing
//! - **Response Quality**: Assessment of response relevance and appropriateness
//! - **Coherence Evaluation**: Measurement of dialogue coherence and consistency
//! - **Conversation Flow**: Analysis of natural conversation progression
//! - **Emotional Appropriateness**: Evaluation of emotional responses in dialogue
//! - **Interruption Handling**: Assessment of interruption management
//! - **Backchanneling**: Analysis of conversational acknowledgments
//! - **Engagement Metrics**: Measurement of user engagement levels
//!
//! # Example
//!
//! ```no_run
//! use voirs_evaluation::conversational::*;
//! use voirs_sdk::AudioBuffer;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create conversational evaluator
//! let config = ConversationalConfig::default();
//! let evaluator = ConversationalEvaluator::new(config)?;
//!
//! // Create dialogue turn
//! let turn = DialogueTurn::new(
//!     "user",
//!     "How's the weather today?",
//!     0.0,
//!     2.0,
//! );
//!
//! // Create sample audio
//! let sample_rate = 16000;
//! let samples: Vec<f32> = (0..32000)
//!     .map(|i| (i as f32 * 0.01).sin() * 0.1)
//!     .collect();
//! let audio = AudioBuffer::new(samples, sample_rate, 1);
//!
//! // Evaluate turn
//! let result = evaluator.evaluate_turn(&turn, &audio, None)?;
//!
//! println!("Turn Quality: {:.2}", result.overall_quality);
//! println!("Response Appropriateness: {:.2}", result.response_appropriateness);
//! # Ok(())
//! # }
//! ```

use crate::EvaluationError;
use scirs2_core::numeric::Float;
use scirs2_core::random::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use voirs_sdk::AudioBuffer;

/// Speaker role in conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpeakerRole {
    /// User/Human speaker
    User,
    /// System/AI assistant
    System,
    /// Third party in multi-party conversation
    ThirdParty,
}

/// Dialogue turn type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TurnType {
    /// Initiating a new topic
    Initiation,
    /// Responding to previous turn
    Response,
    /// Follow-up question
    FollowUp,
    /// Acknowledgment/backchannel
    Acknowledgment,
    /// Clarification request
    Clarification,
    /// Topic change
    TopicChange,
    /// Closing/ending
    Closing,
}

/// Emotional tone of a dialogue turn
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmotionalTone {
    /// Neutral tone
    Neutral,
    /// Friendly/warm tone
    Friendly,
    /// Professional/formal tone
    Professional,
    /// Empathetic tone
    Empathetic,
    /// Apologetic tone
    Apologetic,
    /// Assertive tone
    Assertive,
    /// Enthusiastic tone
    Enthusiastic,
}

/// Represents a single dialogue turn
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueTurn {
    /// Speaker identifier
    pub speaker_id: String,
    /// Speaker role
    pub role: SpeakerRole,
    /// Turn text content
    pub text: String,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Turn type
    pub turn_type: TurnType,
    /// Emotional tone
    pub emotional_tone: EmotionalTone,
    /// Turn metadata
    pub metadata: HashMap<String, String>,
}

impl DialogueTurn {
    /// Creates a new dialogue turn
    pub fn new(speaker_id: &str, text: &str, start_time: f32, end_time: f32) -> Self {
        let role = if speaker_id.to_lowercase().contains("user") {
            SpeakerRole::User
        } else {
            SpeakerRole::System
        };

        Self {
            speaker_id: speaker_id.to_string(),
            role,
            text: text.to_string(),
            start_time,
            end_time,
            turn_type: TurnType::Response,
            emotional_tone: EmotionalTone::Neutral,
            metadata: HashMap::new(),
        }
    }

    /// Gets the duration of the turn
    pub fn duration(&self) -> f32 {
        self.end_time - self.start_time
    }

    /// Sets the turn type
    pub fn with_turn_type(mut self, turn_type: TurnType) -> Self {
        self.turn_type = turn_type;
        self
    }

    /// Sets the emotional tone
    pub fn with_emotional_tone(mut self, tone: EmotionalTone) -> Self {
        self.emotional_tone = tone;
        self
    }

    /// Adds metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Represents a complete dialogue/conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dialogue {
    /// Dialogue identifier
    pub dialogue_id: String,
    /// List of turns
    pub turns: Vec<DialogueTurn>,
    /// Dialogue metadata
    pub metadata: HashMap<String, String>,
}

impl Dialogue {
    /// Creates a new dialogue
    pub fn new(dialogue_id: &str) -> Self {
        Self {
            dialogue_id: dialogue_id.to_string(),
            turns: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Adds a turn to the dialogue
    pub fn add_turn(&mut self, turn: DialogueTurn) {
        self.turns.push(turn);
    }

    /// Gets the total duration
    pub fn total_duration(&self) -> f32 {
        self.turns.last().map(|t| t.end_time).unwrap_or(0.0)
            - self.turns.first().map(|t| t.start_time).unwrap_or(0.0)
    }

    /// Gets the number of turns
    pub fn num_turns(&self) -> usize {
        self.turns.len()
    }
}

/// Turn-level evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnEvaluationResult {
    /// Overall turn quality (0.0-1.0)
    pub overall_quality: f32,
    /// Response appropriateness score (0.0-1.0)
    pub response_appropriateness: f32,
    /// Timing quality (0.0-1.0)
    pub timing_quality: f32,
    /// Audio quality (0.0-1.0)
    pub audio_quality: f32,
    /// Emotional appropriateness (0.0-1.0)
    pub emotional_appropriateness: f32,
    /// Natural fluency (0.0-1.0)
    pub fluency: f32,
    /// Issues detected
    pub issues: Vec<String>,
}

/// Dialogue-level evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationalEvaluationResult {
    /// Overall dialogue quality (0.0-1.0)
    pub overall_quality: f32,
    /// Coherence score (0.0-1.0)
    pub coherence: f32,
    /// Conversation flow quality (0.0-1.0)
    pub flow_quality: f32,
    /// Turn-taking quality (0.0-1.0)
    pub turn_taking_quality: f32,
    /// Average response quality (0.0-1.0)
    pub avg_response_quality: f32,
    /// Engagement score (0.0-1.0)
    pub engagement_score: f32,
    /// Natural conversation score (0.0-1.0)
    pub naturalness: f32,
    /// Emotional consistency (0.0-1.0)
    pub emotional_consistency: f32,
    /// Turn-level results
    pub turn_results: Vec<TurnEvaluationResult>,
    /// Dialogue statistics
    pub statistics: ConversationStatistics,
    /// Issues detected
    pub issues: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Conversation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationStatistics {
    /// Total number of turns
    pub total_turns: usize,
    /// Number of user turns
    pub user_turns: usize,
    /// Number of system turns
    pub system_turns: usize,
    /// Average turn duration
    pub avg_turn_duration: f32,
    /// Average response time (gap between turns)
    pub avg_response_time: f32,
    /// Number of interruptions
    pub num_interruptions: usize,
    /// Number of overlaps
    pub num_overlaps: usize,
    /// Total dialogue duration
    pub total_duration: f32,
}

/// Configuration for conversational evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationalConfig {
    /// Enable turn-taking analysis
    pub analyze_turn_taking: bool,
    /// Enable coherence evaluation
    pub analyze_coherence: bool,
    /// Enable emotional analysis
    pub analyze_emotions: bool,
    /// Enable engagement metrics
    pub analyze_engagement: bool,
    /// Maximum acceptable response time (seconds)
    pub max_response_time: f32,
    /// Minimum acceptable response time (seconds)
    pub min_response_time: f32,
    /// Generate recommendations
    pub generate_recommendations: bool,
}

impl Default for ConversationalConfig {
    fn default() -> Self {
        Self {
            analyze_turn_taking: true,
            analyze_coherence: true,
            analyze_emotions: true,
            analyze_engagement: true,
            max_response_time: 2.0,
            min_response_time: 0.2,
            generate_recommendations: true,
        }
    }
}

/// Conversational quality evaluator
pub struct ConversationalEvaluator {
    config: ConversationalConfig,
}

impl ConversationalEvaluator {
    /// Creates a new conversational evaluator
    pub fn new(config: ConversationalConfig) -> Result<Self, EvaluationError> {
        Ok(Self { config })
    }

    /// Evaluates a single dialogue turn
    pub fn evaluate_turn(
        &self,
        turn: &DialogueTurn,
        audio: &AudioBuffer,
        previous_turn: Option<&DialogueTurn>,
    ) -> Result<TurnEvaluationResult, EvaluationError> {
        // Evaluate audio quality
        let audio_quality = self.evaluate_audio_quality(audio)?;

        // Evaluate timing
        let timing_quality = if let Some(prev) = previous_turn {
            self.evaluate_timing_quality(turn, prev)?
        } else {
            1.0
        };

        // Evaluate response appropriateness
        let response_appropriateness = if let Some(prev) = previous_turn {
            self.evaluate_response_appropriateness(turn, prev)?
        } else {
            0.8 // Default for initiating turns
        };

        // Evaluate emotional appropriateness
        let emotional_appropriateness =
            self.evaluate_emotional_appropriateness(turn, previous_turn)?;

        // Evaluate fluency
        let fluency = self.evaluate_fluency(audio)?;

        // Calculate overall quality
        let overall_quality = (audio_quality * 0.25
            + timing_quality * 0.2
            + response_appropriateness * 0.25
            + emotional_appropriateness * 0.15
            + fluency * 0.15)
            .clamp(0.0, 1.0);

        // Detect issues
        let issues = self.detect_turn_issues(
            turn,
            audio_quality,
            timing_quality,
            response_appropriateness,
        );

        Ok(TurnEvaluationResult {
            overall_quality,
            response_appropriateness,
            timing_quality,
            audio_quality,
            emotional_appropriateness,
            fluency,
            issues,
        })
    }

    /// Evaluates a complete dialogue
    pub fn evaluate_dialogue(
        &self,
        dialogue: &Dialogue,
        audio_turns: &[AudioBuffer],
    ) -> Result<ConversationalEvaluationResult, EvaluationError> {
        if dialogue.turns.len() != audio_turns.len() {
            return Err(EvaluationError::ProcessingError {
                message: "Number of turns and audio buffers must match".to_string(),
                source: None,
            });
        }

        // Evaluate individual turns
        let mut turn_results = Vec::new();
        for (i, (turn, audio)) in dialogue.turns.iter().zip(audio_turns.iter()).enumerate() {
            let previous_turn = if i > 0 {
                Some(&dialogue.turns[i - 1])
            } else {
                None
            };
            let result = self.evaluate_turn(turn, audio, previous_turn)?;
            turn_results.push(result);
        }

        // Calculate dialogue-level metrics
        let coherence = if self.config.analyze_coherence {
            self.evaluate_coherence(&dialogue.turns)?
        } else {
            0.8
        };

        let flow_quality = if self.config.analyze_turn_taking {
            self.evaluate_flow_quality(&dialogue.turns)?
        } else {
            0.8
        };

        let turn_taking_quality = if self.config.analyze_turn_taking {
            self.evaluate_turn_taking(&dialogue.turns)?
        } else {
            0.8
        };

        let engagement_score = if self.config.analyze_engagement {
            self.evaluate_engagement(&dialogue.turns)?
        } else {
            0.7
        };

        let emotional_consistency = if self.config.analyze_emotions {
            self.evaluate_emotional_consistency(&dialogue.turns)?
        } else {
            0.8
        };

        // Calculate average response quality
        let avg_response_quality = if !turn_results.is_empty() {
            turn_results
                .iter()
                .map(|r| r.response_appropriateness)
                .sum::<f32>()
                / turn_results.len() as f32
        } else {
            0.0
        };

        // Calculate naturalness
        let naturalness = if !turn_results.is_empty() {
            turn_results.iter().map(|r| r.fluency).sum::<f32>() / turn_results.len() as f32
        } else {
            0.0
        };

        // Calculate overall quality
        let overall_quality = (coherence * 0.2
            + flow_quality * 0.2
            + turn_taking_quality * 0.15
            + avg_response_quality * 0.2
            + engagement_score * 0.1
            + naturalness * 0.1
            + emotional_consistency * 0.05)
            .clamp(0.0, 1.0);

        // Calculate statistics
        let statistics = self.calculate_statistics(dialogue);

        // Detect dialogue-level issues
        let issues = self.detect_dialogue_issues(dialogue, &turn_results, &statistics);

        // Generate recommendations
        let recommendations = if self.config.generate_recommendations {
            self.generate_recommendations(
                &turn_results,
                &statistics,
                coherence,
                flow_quality,
                engagement_score,
            )
        } else {
            Vec::new()
        };

        Ok(ConversationalEvaluationResult {
            overall_quality,
            coherence,
            flow_quality,
            turn_taking_quality,
            avg_response_quality,
            engagement_score,
            naturalness,
            emotional_consistency,
            turn_results,
            statistics,
            issues,
            recommendations,
        })
    }

    // Helper methods

    fn evaluate_audio_quality(&self, audio: &AudioBuffer) -> Result<f32, EvaluationError> {
        // Calculate RMS energy and SNR estimate
        let samples = audio.samples();
        let rms = self.calculate_rms(samples);

        // Simple quality heuristic based on energy level
        let quality = if rms > 0.01 && rms < 0.5 {
            1.0 - (rms - 0.1).abs() * 2.0
        } else if rms <= 0.01 {
            0.5
        } else {
            0.7
        };

        Ok(quality.clamp(0.0, 1.0))
    }

    fn evaluate_timing_quality(
        &self,
        current: &DialogueTurn,
        previous: &DialogueTurn,
    ) -> Result<f32, EvaluationError> {
        let gap = current.start_time - previous.end_time;

        // Evaluate based on configured acceptable range
        let quality =
            if gap >= self.config.min_response_time && gap <= self.config.max_response_time {
                1.0
            } else if gap < self.config.min_response_time {
                // Too fast (interruption)
                (gap / self.config.min_response_time).max(0.3)
            } else {
                // Too slow (awkward pause)
                let excess = gap - self.config.max_response_time;
                (1.0 - (excess / 3.0).min(0.7)).max(0.3)
            };

        Ok(quality)
    }

    fn evaluate_response_appropriateness(
        &self,
        current: &DialogueTurn,
        previous: &DialogueTurn,
    ) -> Result<f32, EvaluationError> {
        // Evaluate based on turn type transitions
        let transition_quality = match (previous.turn_type, current.turn_type) {
            (TurnType::Initiation, TurnType::Response) => 1.0,
            (TurnType::Response, TurnType::FollowUp) => 0.9,
            (TurnType::Clarification, TurnType::Response) => 1.0,
            (TurnType::FollowUp, TurnType::Response) => 0.9,
            (_, TurnType::Acknowledgment) => 0.8,
            (_, TurnType::TopicChange) => 0.7,
            _ => 0.6,
        };

        // Consider text length appropriateness
        let length_quality = if current.text.len() > 10 && current.text.len() < 500 {
            1.0
        } else if current.text.is_empty() {
            0.3
        } else {
            0.7
        };

        Ok((transition_quality * 0.7 + length_quality * 0.3).clamp(0.0, 1.0))
    }

    fn evaluate_emotional_appropriateness(
        &self,
        current: &DialogueTurn,
        previous: Option<&DialogueTurn>,
    ) -> Result<f32, EvaluationError> {
        if let Some(prev) = previous {
            // Evaluate emotional tone consistency
            let consistency = match (prev.emotional_tone, current.emotional_tone) {
                (a, b) if a == b => 1.0,
                (EmotionalTone::Professional, EmotionalTone::Neutral)
                | (EmotionalTone::Neutral, EmotionalTone::Professional) => 0.9,
                (EmotionalTone::Friendly, EmotionalTone::Empathetic)
                | (EmotionalTone::Empathetic, EmotionalTone::Friendly) => 0.9,
                _ => 0.7,
            };
            Ok(consistency)
        } else {
            Ok(0.8)
        }
    }

    fn evaluate_fluency(&self, audio: &AudioBuffer) -> Result<f32, EvaluationError> {
        // Calculate variation in energy envelope as fluency proxy
        let samples = audio.samples();
        let envelope = self.calculate_energy_envelope(samples);

        let variation = self.calculate_variation(&envelope);

        // Moderate variation indicates good fluency
        let fluency = if variation > 0.05 && variation < 0.3 {
            1.0 - (variation - 0.15).abs() * 2.0
        } else {
            0.6
        };

        Ok(fluency.clamp(0.0, 1.0))
    }

    fn evaluate_coherence(&self, turns: &[DialogueTurn]) -> Result<f32, EvaluationError> {
        if turns.len() < 2 {
            return Ok(1.0);
        }

        // Analyze turn type sequences for coherence
        let mut coherence_sum = 0.0;
        let mut count = 0;

        for i in 1..turns.len() {
            let prev = &turns[i - 1];
            let curr = &turns[i];

            // Check role alternation (good for conversation)
            let role_score = if prev.role != curr.role { 1.0 } else { 0.5 };

            // Check appropriate turn type transitions
            let type_score = match (prev.turn_type, curr.turn_type) {
                (TurnType::Initiation, TurnType::Response) => 1.0,
                (TurnType::Response, TurnType::FollowUp) => 0.9,
                (TurnType::Clarification, TurnType::Response) => 1.0,
                _ => 0.7,
            };

            coherence_sum += role_score * 0.4 + type_score * 0.6;
            count += 1;
        }

        let coherence = if count > 0 {
            coherence_sum / count as f32
        } else {
            0.8
        };

        Ok(coherence.clamp(0.0, 1.0))
    }

    fn evaluate_flow_quality(&self, turns: &[DialogueTurn]) -> Result<f32, EvaluationError> {
        if turns.len() < 2 {
            return Ok(1.0);
        }

        // Analyze timing gaps between turns
        let mut gap_qualities = Vec::new();

        for i in 1..turns.len() {
            let gap = turns[i].start_time - turns[i - 1].end_time;

            let quality =
                if gap >= self.config.min_response_time && gap <= self.config.max_response_time {
                    1.0
                } else if gap < 0.0 {
                    0.3 // Overlap/interruption
                } else if gap < self.config.min_response_time {
                    gap / self.config.min_response_time
                } else {
                    let excess = gap - self.config.max_response_time;
                    (1.0 - (excess / 3.0).min(0.7)).max(0.3)
                };

            gap_qualities.push(quality);
        }

        let flow_quality = if !gap_qualities.is_empty() {
            gap_qualities.iter().sum::<f32>() / gap_qualities.len() as f32
        } else {
            0.8
        };

        Ok(flow_quality.clamp(0.0, 1.0))
    }

    fn evaluate_turn_taking(&self, turns: &[DialogueTurn]) -> Result<f32, EvaluationError> {
        if turns.len() < 2 {
            return Ok(1.0);
        }

        // Count proper role alternations
        let mut proper_alternations = 0;
        let mut total_transitions = 0;

        for i in 1..turns.len() {
            if turns[i].role != turns[i - 1].role {
                proper_alternations += 1;
            }
            total_transitions += 1;
        }

        let alternation_rate = if total_transitions > 0 {
            proper_alternations as f32 / total_transitions as f32
        } else {
            1.0
        };

        Ok(alternation_rate.clamp(0.0, 1.0))
    }

    fn evaluate_engagement(&self, turns: &[DialogueTurn]) -> Result<f32, EvaluationError> {
        if turns.is_empty() {
            return Ok(0.0);
        }

        // Analyze turn frequency and diversity
        let avg_turn_length: f32 =
            turns.iter().map(|t| t.duration()).sum::<f32>() / turns.len() as f32;

        // Good engagement: moderate turn length (3-15 seconds)
        let length_score = if avg_turn_length > 3.0 && avg_turn_length < 15.0 {
            1.0
        } else if avg_turn_length < 3.0 {
            avg_turn_length / 3.0
        } else {
            (1.0 - ((avg_turn_length - 15.0) / 20.0).min(0.7)).max(0.3)
        };

        // Count turn type diversity
        let unique_types: std::collections::HashSet<_> =
            turns.iter().map(|t| t.turn_type).collect();
        let diversity_score = (unique_types.len() as f32 / 7.0).min(1.0);

        Ok((length_score * 0.6 + diversity_score * 0.4).clamp(0.0, 1.0))
    }

    fn evaluate_emotional_consistency(
        &self,
        turns: &[DialogueTurn],
    ) -> Result<f32, EvaluationError> {
        if turns.len() < 2 {
            return Ok(1.0);
        }

        // Analyze emotional tone changes
        let mut consistent_transitions = 0;
        let mut total_transitions = 0;

        for i in 1..turns.len() {
            let prev_tone = turns[i - 1].emotional_tone;
            let curr_tone = turns[i].emotional_tone;

            // Appropriate transitions get higher scores
            let is_consistent = match (prev_tone, curr_tone) {
                (a, b) if a == b => true,
                (EmotionalTone::Neutral, _) | (_, EmotionalTone::Neutral) => true,
                (EmotionalTone::Professional, EmotionalTone::Friendly)
                | (EmotionalTone::Friendly, EmotionalTone::Professional) => true,
                _ => false,
            };

            if is_consistent {
                consistent_transitions += 1;
            }
            total_transitions += 1;
        }

        let consistency = if total_transitions > 0 {
            consistent_transitions as f32 / total_transitions as f32
        } else {
            1.0
        };

        Ok(consistency.clamp(0.0, 1.0))
    }

    fn calculate_statistics(&self, dialogue: &Dialogue) -> ConversationStatistics {
        let total_turns = dialogue.turns.len();
        let user_turns = dialogue
            .turns
            .iter()
            .filter(|t| matches!(t.role, SpeakerRole::User))
            .count();
        let system_turns = total_turns - user_turns;

        let avg_turn_duration = if !dialogue.turns.is_empty() {
            dialogue.turns.iter().map(|t| t.duration()).sum::<f32>() / dialogue.turns.len() as f32
        } else {
            0.0
        };

        let mut response_times = Vec::new();
        let mut num_interruptions = 0;
        let mut num_overlaps = 0;

        for i in 1..dialogue.turns.len() {
            let gap = dialogue.turns[i].start_time - dialogue.turns[i - 1].end_time;
            if gap >= 0.0 {
                response_times.push(gap);
            } else {
                if gap < -0.5 {
                    num_interruptions += 1;
                } else {
                    num_overlaps += 1;
                }
            }
        }

        let avg_response_time = if !response_times.is_empty() {
            response_times.iter().sum::<f32>() / response_times.len() as f32
        } else {
            0.0
        };

        let total_duration = dialogue.total_duration();

        ConversationStatistics {
            total_turns,
            user_turns,
            system_turns,
            avg_turn_duration,
            avg_response_time,
            num_interruptions,
            num_overlaps,
            total_duration,
        }
    }

    fn detect_turn_issues(
        &self,
        turn: &DialogueTurn,
        audio_quality: f32,
        timing_quality: f32,
        response_appropriateness: f32,
    ) -> Vec<String> {
        let mut issues = Vec::new();

        if audio_quality < 0.6 {
            issues.push(format!(
                "Low audio quality detected in turn by {} (score: {:.2})",
                turn.speaker_id, audio_quality
            ));
        }

        if timing_quality < 0.5 {
            issues.push(format!(
                "Timing issue detected in turn by {} (score: {:.2})",
                turn.speaker_id, timing_quality
            ));
        }

        if response_appropriateness < 0.5 {
            issues.push(format!(
                "Response appropriateness issue in turn by {} (score: {:.2})",
                turn.speaker_id, response_appropriateness
            ));
        }

        if turn.text.is_empty() {
            issues.push(format!("Empty text content in turn by {}", turn.speaker_id));
        }

        issues
    }

    fn detect_dialogue_issues(
        &self,
        dialogue: &Dialogue,
        turn_results: &[TurnEvaluationResult],
        statistics: &ConversationStatistics,
    ) -> Vec<String> {
        let mut issues = Vec::new();

        if statistics.num_interruptions > dialogue.turns.len() / 4 {
            issues.push(format!(
                "High number of interruptions detected ({} out of {} turns)",
                statistics.num_interruptions, statistics.total_turns
            ));
        }

        if statistics.avg_response_time > self.config.max_response_time * 1.5 {
            issues.push(format!(
                "Average response time ({:.2}s) exceeds acceptable range",
                statistics.avg_response_time
            ));
        }

        let low_quality_turns = turn_results
            .iter()
            .filter(|r| r.overall_quality < 0.5)
            .count();
        if low_quality_turns > dialogue.turns.len() / 3 {
            issues.push(format!(
                "High proportion of low-quality turns ({}/{})",
                low_quality_turns,
                dialogue.turns.len()
            ));
        }

        if statistics.user_turns == 0 || statistics.system_turns == 0 {
            issues.push("Dialogue lacks proper turn alternation between roles".to_string());
        }

        issues
    }

    fn generate_recommendations(
        &self,
        turn_results: &[TurnEvaluationResult],
        statistics: &ConversationStatistics,
        coherence: f32,
        flow_quality: f32,
        engagement_score: f32,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if coherence < 0.7 {
            recommendations.push(format!(
                "Improve dialogue coherence (current: {:.2}). Ensure logical turn transitions and topic continuity.",
                coherence
            ));
        }

        if flow_quality < 0.7 {
            recommendations.push(format!(
                "Enhance conversation flow (current: {:.2}). Adjust response timing to {:.1}s-{:.1}s range.",
                flow_quality, self.config.min_response_time, self.config.max_response_time
            ));
        }

        if engagement_score < 0.6 {
            recommendations.push(format!(
                "Increase engagement (current: {:.2}). Vary turn types and maintain appropriate turn lengths.",
                engagement_score
            ));
        }

        if statistics.avg_response_time > self.config.max_response_time {
            recommendations.push(format!(
                "Reduce response latency (current: {:.2}s, target: <{:.1}s)",
                statistics.avg_response_time, self.config.max_response_time
            ));
        }

        let avg_turn_quality = if !turn_results.is_empty() {
            turn_results.iter().map(|r| r.overall_quality).sum::<f32>() / turn_results.len() as f32
        } else {
            0.0
        };

        if avg_turn_quality < 0.7 {
            recommendations.push(format!(
                "Improve overall turn quality (current: {:.2}). Focus on audio quality and response appropriateness.",
                avg_turn_quality
            ));
        }

        if recommendations.is_empty() {
            recommendations.push(
                "Conversational quality meets expectations. Continue current practices."
                    .to_string(),
            );
        }

        recommendations
    }

    // Utility methods

    fn calculate_rms(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: f32 = samples.iter().map(|&s| s * s).sum();
        (sum / samples.len() as f32).sqrt()
    }

    fn calculate_variation(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
        let variance: f32 =
            values.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;

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
    fn test_conversational_evaluator_creation() {
        let config = ConversationalConfig::default();
        let evaluator = ConversationalEvaluator::new(config);
        assert!(evaluator.is_ok());
    }

    #[test]
    fn test_dialogue_turn_creation() {
        let turn = DialogueTurn::new("user1", "Hello, how are you?", 0.0, 2.0);
        assert_eq!(turn.speaker_id, "user1");
        assert_eq!(turn.text, "Hello, how are you?");
        assert_eq!(turn.duration(), 2.0);
        assert_eq!(turn.role, SpeakerRole::User);
    }

    #[test]
    fn test_dialogue_creation() {
        let mut dialogue = Dialogue::new("test_dialogue");
        dialogue.add_turn(DialogueTurn::new("user", "Hi", 0.0, 1.0));
        dialogue.add_turn(DialogueTurn::new("system", "Hello!", 1.5, 2.5));

        assert_eq!(dialogue.num_turns(), 2);
        assert_eq!(dialogue.total_duration(), 2.5);
    }

    #[test]
    fn test_turn_evaluation() {
        let config = ConversationalConfig::default();
        let evaluator = ConversationalEvaluator::new(config).unwrap();

        let turn = DialogueTurn::new("user", "How's the weather?", 0.0, 2.0);
        let audio = create_test_audio(2.0, 16000);

        let result = evaluator.evaluate_turn(&turn, &audio, None);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.overall_quality >= 0.0 && result.overall_quality <= 1.0);
    }

    #[test]
    fn test_dialogue_evaluation() {
        let config = ConversationalConfig::default();
        let evaluator = ConversationalEvaluator::new(config).unwrap();

        let mut dialogue = Dialogue::new("test");
        dialogue.add_turn(DialogueTurn::new("user", "Hello", 0.0, 1.0));
        dialogue.add_turn(DialogueTurn::new("system", "Hi there!", 1.5, 2.5));

        let audio1 = create_test_audio(1.0, 16000);
        let audio2 = create_test_audio(1.0, 16000);
        let audio_turns = vec![audio1, audio2];

        let result = evaluator.evaluate_dialogue(&dialogue, &audio_turns);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.overall_quality >= 0.0 && result.overall_quality <= 1.0);
        assert_eq!(result.statistics.total_turns, 2);
    }

    #[test]
    fn test_timing_quality_evaluation() {
        let config = ConversationalConfig::default();
        let evaluator = ConversationalEvaluator::new(config).unwrap();

        let turn1 = DialogueTurn::new("user", "Question?", 0.0, 1.0);
        let turn2 = DialogueTurn::new("system", "Answer", 1.5, 2.5);

        let quality = evaluator.evaluate_timing_quality(&turn2, &turn1).unwrap();
        assert!(quality > 0.8); // 0.5s gap is within acceptable range
    }

    #[test]
    fn test_coherence_evaluation() {
        let config = ConversationalConfig::default();
        let evaluator = ConversationalEvaluator::new(config).unwrap();

        let turns = vec![
            DialogueTurn::new("user", "Hello", 0.0, 1.0).with_turn_type(TurnType::Initiation),
            DialogueTurn::new("system", "Hi", 1.5, 2.0).with_turn_type(TurnType::Response),
        ];

        let coherence = evaluator.evaluate_coherence(&turns).unwrap();
        assert!(coherence > 0.7);
    }

    #[test]
    fn test_conversation_statistics() {
        let config = ConversationalConfig::default();
        let evaluator = ConversationalEvaluator::new(config).unwrap();

        let mut dialogue = Dialogue::new("test");
        dialogue.add_turn(DialogueTurn::new("user", "Q1", 0.0, 1.0));
        dialogue.add_turn(DialogueTurn::new("system", "A1", 1.5, 2.5));
        dialogue.add_turn(DialogueTurn::new("user", "Q2", 3.0, 4.0));

        let stats = evaluator.calculate_statistics(&dialogue);
        assert_eq!(stats.total_turns, 3);
        assert_eq!(stats.user_turns, 2);
        assert_eq!(stats.system_turns, 1);
    }

    #[test]
    fn test_turn_type_transitions() {
        let turn1 =
            DialogueTurn::new("user", "Hello", 0.0, 1.0).with_turn_type(TurnType::Initiation);
        let turn2 = DialogueTurn::new("system", "Hi", 1.5, 2.0).with_turn_type(TurnType::Response);

        assert_eq!(turn1.turn_type, TurnType::Initiation);
        assert_eq!(turn2.turn_type, TurnType::Response);
    }

    #[test]
    fn test_emotional_tone() {
        let turn = DialogueTurn::new("system", "I'm sorry to hear that", 0.0, 2.0)
            .with_emotional_tone(EmotionalTone::Empathetic);

        assert_eq!(turn.emotional_tone, EmotionalTone::Empathetic);
    }

    #[test]
    fn test_mismatched_audio_turns() {
        let config = ConversationalConfig::default();
        let evaluator = ConversationalEvaluator::new(config).unwrap();

        let mut dialogue = Dialogue::new("test");
        dialogue.add_turn(DialogueTurn::new("user", "Hello", 0.0, 1.0));

        let audio_turns = vec![]; // Mismatch: 1 turn, 0 audio

        let result = evaluator.evaluate_dialogue(&dialogue, &audio_turns);
        assert!(result.is_err());
    }
}
