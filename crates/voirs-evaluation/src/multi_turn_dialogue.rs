// Copyright (c) 2025 VoiRS Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-turn dialogue evaluation for extended conversations and dialogue systems.
//!
//! This module provides comprehensive evaluation capabilities for multi-turn dialogues,
//! analyzing conversation coherence, context tracking, topic progression, and overall
//! dialogue quality across multiple turns.
//!
//! # Features
//!
//! - **Context Tracking**: Evaluation of context maintenance across dialogue turns
//! - **Topic Coherence**: Analysis of topic progression and coherence
//! - **Dialogue State**: Assessment of dialogue state management
//! - **Long-term Quality**: Measurement of conversation quality over extended interactions
//! - **Turn Sequence Analysis**: Analysis of turn sequences and patterns
//! - **Progression Evaluation**: Evaluation of dialogue progression towards goals
//! - **Consistency Checking**: Verification of consistent information across turns
//! - **Memory Evaluation**: Assessment of conversation memory and recall
//!
//! # Example
//!
//! ```rust
//! use voirs_evaluation::multi_turn_dialogue::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create multi-turn dialogue evaluator
//! let config = MultiTurnConfig::default();
//! let evaluator = MultiTurnDialogueEvaluator::new(config)?;
//!
//! // Create a multi-turn dialogue
//! let mut dialogue = MultiTurnDialogue::new("booking_conversation");
//! dialogue.add_turn(Turn::new("user", "I want to book a flight", 0));
//! dialogue.add_turn(Turn::new("system", "Where would you like to go?", 1));
//! dialogue.add_turn(Turn::new("user", "New York", 2));
//! dialogue.add_turn(Turn::new("system", "When would you like to travel?", 3));
//!
//! // Evaluate the dialogue
//! let result = evaluator.evaluate(&dialogue)?;
//!
//! println!("Overall Quality: {:.2}", result.overall_quality);
//! println!("Context Coherence: {:.2}", result.context_coherence);
//! println!("Topic Consistency: {:.2}", result.topic_consistency);
//! # Ok(())
//! # }
//! ```

use crate::EvaluationError;
use scirs2_core::numeric::Float;
use scirs2_core::random::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Represents a single dialogue turn in a multi-turn conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    /// Speaker identifier
    pub speaker: String,
    /// Turn content
    pub content: String,
    /// Turn index/position
    pub turn_index: usize,
    /// Intent/goal of this turn
    pub intent: Option<String>,
    /// Entities mentioned in this turn
    pub entities: HashMap<String, String>,
    /// Timestamp
    pub timestamp: Option<i64>,
}

impl Turn {
    /// Creates a new turn
    pub fn new(speaker: &str, content: &str, turn_index: usize) -> Self {
        Self {
            speaker: speaker.to_string(),
            content: content.to_string(),
            turn_index,
            intent: None,
            entities: HashMap::new(),
            timestamp: None,
        }
    }

    /// Sets the intent
    pub fn with_intent(mut self, intent: &str) -> Self {
        self.intent = Some(intent.to_string());
        self
    }

    /// Adds an entity
    pub fn with_entity(mut self, key: String, value: String) -> Self {
        self.entities.insert(key, value);
        self
    }
}

/// Represents a complete multi-turn dialogue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTurnDialogue {
    /// Dialogue identifier
    pub dialogue_id: String,
    /// List of turns in order
    pub turns: Vec<Turn>,
    /// Dialogue goal/task
    pub goal: Option<String>,
    /// Topics covered
    pub topics: Vec<String>,
    /// Dialogue metadata
    pub metadata: HashMap<String, String>,
}

impl MultiTurnDialogue {
    /// Creates a new multi-turn dialogue
    pub fn new(dialogue_id: &str) -> Self {
        Self {
            dialogue_id: dialogue_id.to_string(),
            turns: Vec::new(),
            goal: None,
            topics: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Adds a turn to the dialogue
    pub fn add_turn(&mut self, turn: Turn) {
        self.turns.push(turn);
    }

    /// Sets the dialogue goal
    pub fn with_goal(mut self, goal: &str) -> Self {
        self.goal = Some(goal.to_string());
        self
    }

    /// Adds a topic
    pub fn add_topic(&mut self, topic: &str) {
        self.topics.push(topic.to_string());
    }

    /// Gets the number of turns
    pub fn num_turns(&self) -> usize {
        self.turns.len()
    }
}

/// Multi-turn dialogue evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTurnEvaluationResult {
    /// Overall dialogue quality (0.0-1.0)
    pub overall_quality: f32,
    /// Context coherence across turns (0.0-1.0)
    pub context_coherence: f32,
    /// Topic consistency score (0.0-1.0)
    pub topic_consistency: f32,
    /// Information consistency (0.0-1.0)
    pub information_consistency: f32,
    /// Dialogue progression quality (0.0-1.0)
    pub progression_quality: f32,
    /// Context tracking accuracy (0.0-1.0)
    pub context_tracking: f32,
    /// Memory and recall score (0.0-1.0)
    pub memory_score: f32,
    /// Goal achievement score (0.0-1.0)
    pub goal_achievement: f32,
    /// Turn-by-turn analysis
    pub turn_analysis: Vec<TurnAnalysis>,
    /// Topic transitions
    pub topic_transitions: Vec<TopicTransition>,
    /// Context violations detected
    pub context_violations: Vec<ContextViolation>,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Dialogue statistics
    pub statistics: DialogueStatistics,
}

/// Analysis of a single turn in context of the dialogue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnAnalysis {
    /// Turn index
    pub turn_index: usize,
    /// Relevance to previous context (0.0-1.0)
    pub context_relevance: f32,
    /// Consistency with dialogue history (0.0-1.0)
    pub consistency: f32,
    /// Information contribution (0.0-1.0)
    pub information_contribution: f32,
    /// Goal progress (0.0-1.0)
    pub goal_progress: f32,
}

/// Represents a topic transition in the dialogue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicTransition {
    /// Turn index where transition occurs
    pub turn_index: usize,
    /// Previous topic
    pub from_topic: String,
    /// New topic
    pub to_topic: String,
    /// Transition quality (0.0-1.0)
    pub transition_quality: f32,
    /// Whether transition is appropriate
    pub is_appropriate: bool,
}

/// Represents a context violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextViolation {
    /// Turn index where violation occurs
    pub turn_index: usize,
    /// Type of violation
    pub violation_type: ViolationType,
    /// Description
    pub description: String,
    /// Severity (0.0-1.0)
    pub severity: f32,
}

/// Type of context violation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationType {
    /// Contradicting previous information
    Contradiction,
    /// Ignoring context
    ContextIgnored,
    /// Missing required information
    MissingInformation,
    /// Irrelevant response
    Irrelevant,
    /// Incorrect reference
    IncorrectReference,
}

/// Dialogue statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueStatistics {
    /// Total number of turns
    pub total_turns: usize,
    /// Average turn length (words)
    pub avg_turn_length: f32,
    /// Number of topic changes
    pub num_topic_changes: usize,
    /// Number of speakers
    pub num_speakers: usize,
    /// Dialogue duration (if available)
    pub duration: Option<f32>,
    /// Context violations count
    pub num_violations: usize,
}

/// Configuration for multi-turn dialogue evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTurnConfig {
    /// Enable context coherence analysis
    pub analyze_context_coherence: bool,
    /// Enable topic consistency analysis
    pub analyze_topic_consistency: bool,
    /// Enable information consistency checking
    pub check_information_consistency: bool,
    /// Enable goal achievement analysis
    pub analyze_goal_achievement: bool,
    /// Minimum context coherence threshold
    pub coherence_threshold: f32,
    /// Maximum topic drift allowed
    pub max_topic_drift: f32,
    /// Generate recommendations
    pub generate_recommendations: bool,
}

impl Default for MultiTurnConfig {
    fn default() -> Self {
        Self {
            analyze_context_coherence: true,
            analyze_topic_consistency: true,
            check_information_consistency: true,
            analyze_goal_achievement: true,
            coherence_threshold: 0.7,
            max_topic_drift: 0.3,
            generate_recommendations: true,
        }
    }
}

/// Multi-turn dialogue evaluator
pub struct MultiTurnDialogueEvaluator {
    config: MultiTurnConfig,
}

impl MultiTurnDialogueEvaluator {
    /// Creates a new multi-turn dialogue evaluator
    pub fn new(config: MultiTurnConfig) -> Result<Self, EvaluationError> {
        Ok(Self { config })
    }

    /// Evaluates a multi-turn dialogue
    pub fn evaluate(
        &self,
        dialogue: &MultiTurnDialogue,
    ) -> Result<MultiTurnEvaluationResult, EvaluationError> {
        if dialogue.turns.is_empty() {
            return Err(EvaluationError::ProcessingError {
                message: "Cannot evaluate empty dialogue".to_string(),
                source: None,
            });
        }

        // Analyze individual turns in context
        let turn_analysis = self.analyze_turns(dialogue)?;

        // Evaluate context coherence
        let context_coherence = if self.config.analyze_context_coherence {
            self.evaluate_context_coherence(&dialogue.turns, &turn_analysis)?
        } else {
            0.8
        };

        // Evaluate topic consistency
        let (topic_consistency, topic_transitions) = if self.config.analyze_topic_consistency {
            self.evaluate_topic_consistency(&dialogue.turns)?
        } else {
            (0.8, Vec::new())
        };

        // Check information consistency
        let (information_consistency, context_violations) =
            if self.config.check_information_consistency {
                self.check_information_consistency(&dialogue.turns)?
            } else {
                (0.9, Vec::new())
            };

        // Evaluate dialogue progression
        let progression_quality = self.evaluate_progression(&dialogue.turns, &turn_analysis)?;

        // Evaluate context tracking
        let context_tracking = self.evaluate_context_tracking(&dialogue.turns)?;

        // Evaluate memory and recall
        let memory_score = self.evaluate_memory(&dialogue.turns)?;

        // Evaluate goal achievement
        let goal_achievement = if self.config.analyze_goal_achievement {
            self.evaluate_goal_achievement(dialogue)?
        } else {
            0.7
        };

        // Calculate overall quality
        let overall_quality = self.calculate_overall_quality(
            context_coherence,
            topic_consistency,
            information_consistency,
            progression_quality,
            context_tracking,
            memory_score,
            goal_achievement,
        );

        // Calculate statistics
        let statistics = self.calculate_statistics(dialogue, &context_violations);

        // Generate recommendations
        let recommendations = if self.config.generate_recommendations {
            self.generate_recommendations(
                &turn_analysis,
                &context_violations,
                context_coherence,
                topic_consistency,
                information_consistency,
            )
        } else {
            Vec::new()
        };

        Ok(MultiTurnEvaluationResult {
            overall_quality,
            context_coherence,
            topic_consistency,
            information_consistency,
            progression_quality,
            context_tracking,
            memory_score,
            goal_achievement,
            turn_analysis,
            topic_transitions,
            context_violations,
            recommendations,
            statistics,
        })
    }

    /// Analyzes individual turns in the context of the dialogue
    fn analyze_turns(
        &self,
        dialogue: &MultiTurnDialogue,
    ) -> Result<Vec<TurnAnalysis>, EvaluationError> {
        let mut analyses = Vec::new();

        for (i, turn) in dialogue.turns.iter().enumerate() {
            let context_relevance = if i > 0 {
                self.calculate_context_relevance(turn, &dialogue.turns[..i])?
            } else {
                1.0 // First turn has no context
            };

            let consistency = if i > 0 {
                self.calculate_turn_consistency(turn, &dialogue.turns[..i])?
            } else {
                1.0
            };

            let information_contribution =
                self.calculate_information_contribution(turn, &dialogue.turns[..i])?;

            let goal_progress = dialogue
                .goal
                .as_ref()
                .map(|goal| self.calculate_goal_progress(turn, goal))
                .unwrap_or(0.5);

            analyses.push(TurnAnalysis {
                turn_index: i,
                context_relevance,
                consistency,
                information_contribution,
                goal_progress,
            });
        }

        Ok(analyses)
    }

    /// Evaluates context coherence across the dialogue
    fn evaluate_context_coherence(
        &self,
        turns: &[Turn],
        analyses: &[TurnAnalysis],
    ) -> Result<f32, EvaluationError> {
        if analyses.is_empty() {
            return Ok(1.0);
        }

        let avg_relevance =
            analyses.iter().map(|a| a.context_relevance).sum::<f32>() / analyses.len() as f32;

        let avg_consistency =
            analyses.iter().map(|a| a.consistency).sum::<f32>() / analyses.len() as f32;

        Ok((avg_relevance * 0.6 + avg_consistency * 0.4).clamp(0.0, 1.0))
    }

    /// Evaluates topic consistency across the dialogue
    fn evaluate_topic_consistency(
        &self,
        turns: &[Turn],
    ) -> Result<(f32, Vec<TopicTransition>), EvaluationError> {
        let mut transitions = Vec::new();
        let mut current_topic = self.extract_topic(&turns[0]);

        for i in 1..turns.len() {
            let new_topic = self.extract_topic(&turns[i]);

            if new_topic != current_topic {
                let transition_quality = self.evaluate_topic_transition(
                    &current_topic,
                    &new_topic,
                    &turns[i - 1],
                    &turns[i],
                );

                transitions.push(TopicTransition {
                    turn_index: i,
                    from_topic: current_topic.clone(),
                    to_topic: new_topic.clone(),
                    transition_quality,
                    is_appropriate: transition_quality > 0.6,
                });

                current_topic = new_topic;
            }
        }

        // Calculate overall topic consistency
        let consistency = if !transitions.is_empty() {
            let avg_quality = transitions
                .iter()
                .map(|t| t.transition_quality)
                .sum::<f32>()
                / transitions.len() as f32;

            let appropriate_ratio = transitions.iter().filter(|t| t.is_appropriate).count() as f32
                / transitions.len() as f32;

            (avg_quality * 0.5 + appropriate_ratio * 0.5).clamp(0.0, 1.0)
        } else {
            1.0 // No transitions means perfect consistency
        };

        Ok((consistency, transitions))
    }

    /// Checks information consistency across the dialogue
    fn check_information_consistency(
        &self,
        turns: &[Turn],
    ) -> Result<(f32, Vec<ContextViolation>), EvaluationError> {
        let mut violations = Vec::new();
        let mut mentioned_entities: HashMap<String, String> = HashMap::new();

        for (i, turn) in turns.iter().enumerate() {
            // Check for contradictions with previous information
            for (entity_key, entity_value) in &turn.entities {
                if let Some(previous_value) = mentioned_entities.get(entity_key) {
                    if previous_value != entity_value {
                        violations.push(ContextViolation {
                            turn_index: i,
                            violation_type: ViolationType::Contradiction,
                            description: format!(
                                "Entity '{}' changed from '{}' to '{}'",
                                entity_key, previous_value, entity_value
                            ),
                            severity: 0.8,
                        });
                    }
                } else {
                    mentioned_entities.insert(entity_key.clone(), entity_value.clone());
                }
            }

            // Check for context relevance
            if i > 0 {
                let relevance = self
                    .calculate_context_relevance(turn, &turns[..i])
                    .unwrap_or(0.5);
                if relevance < self.config.coherence_threshold {
                    violations.push(ContextViolation {
                        turn_index: i,
                        violation_type: ViolationType::Irrelevant,
                        description: format!("Turn has low context relevance: {:.2}", relevance),
                        severity: 1.0 - relevance,
                    });
                }
            }
        }

        // Calculate consistency score
        let consistency = if turns.len() > 1 {
            let violation_penalty = violations.len() as f32 / turns.len() as f32;
            (1.0 - violation_penalty).max(0.0)
        } else {
            1.0
        };

        Ok((consistency, violations))
    }

    /// Evaluates dialogue progression quality
    fn evaluate_progression(
        &self,
        turns: &[Turn],
        analyses: &[TurnAnalysis],
    ) -> Result<f32, EvaluationError> {
        if analyses.len() < 2 {
            return Ok(1.0);
        }

        // Check if information contribution decreases over time (good progression)
        let first_half = &analyses[..analyses.len() / 2];
        let second_half = &analyses[analyses.len() / 2..];

        let first_avg = first_half
            .iter()
            .map(|a| a.information_contribution)
            .sum::<f32>()
            / first_half.len() as f32;

        let second_avg = second_half
            .iter()
            .map(|a| a.information_contribution)
            .sum::<f32>()
            / second_half.len() as f32;

        // Good progression: consistent or slightly decreasing contribution (dialogue converging)
        let progression = if second_avg >= first_avg * 0.7 && second_avg <= first_avg * 1.2 {
            0.9
        } else if second_avg < first_avg * 0.5 {
            0.7 // Dialogue losing steam
        } else {
            0.8
        };

        Ok(progression)
    }

    /// Evaluates context tracking accuracy
    fn evaluate_context_tracking(&self, turns: &[Turn]) -> Result<f32, EvaluationError> {
        if turns.len() < 2 {
            return Ok(1.0);
        }

        let mut tracking_scores = Vec::new();

        for i in 1..turns.len() {
            let current = &turns[i];
            let context = &turns[..i];

            // Check if current turn references or builds on context appropriately
            let score = self.calculate_context_reference_score(current, context);
            tracking_scores.push(score);
        }

        if !tracking_scores.is_empty() {
            Ok(tracking_scores.iter().sum::<f32>() / tracking_scores.len() as f32)
        } else {
            Ok(1.0)
        }
    }

    /// Evaluates memory and recall
    fn evaluate_memory(&self, turns: &[Turn]) -> Result<f32, EvaluationError> {
        if turns.len() < 3 {
            return Ok(1.0);
        }

        let mut memory_scores = Vec::new();

        // Check if later turns recall information from earlier turns
        for i in 2..turns.len() {
            let current_turn = &turns[i];
            let early_context = &turns[..i.saturating_sub(2)]; // Context excluding immediate previous

            if !early_context.is_empty() {
                let recall_score = self.calculate_recall_score(current_turn, early_context);
                memory_scores.push(recall_score);
            }
        }

        if !memory_scores.is_empty() {
            Ok(memory_scores.iter().sum::<f32>() / memory_scores.len() as f32)
        } else {
            Ok(0.7)
        }
    }

    /// Evaluates goal achievement
    fn evaluate_goal_achievement(
        &self,
        dialogue: &MultiTurnDialogue,
    ) -> Result<f32, EvaluationError> {
        if let Some(goal) = &dialogue.goal {
            // Check if final turns indicate goal completion
            let last_few = dialogue.turns.len().saturating_sub(3);
            let final_turns = &dialogue.turns[last_few..];

            let goal_indicators = final_turns
                .iter()
                .filter(|t| self.contains_goal_completion(t, goal))
                .count();

            let achievement = (goal_indicators as f32 / 3.0).min(1.0);
            Ok(achievement.max(0.3)) // Minimum score for attempting
        } else {
            Ok(0.5) // No goal specified
        }
    }

    // Helper methods

    fn calculate_context_relevance(
        &self,
        turn: &Turn,
        context: &[Turn],
    ) -> Result<f32, EvaluationError> {
        if context.is_empty() {
            return Ok(1.0);
        }

        let turn_words: HashSet<_> = self.tokenize(&turn.content).into_iter().collect();
        let mut context_words = HashSet::new();

        for ctx_turn in context.iter().rev().take(3) {
            // Last 3 turns
            context_words.extend(self.tokenize(&ctx_turn.content));
        }

        let overlap = turn_words.intersection(&context_words).count();
        let relevance = if !turn_words.is_empty() {
            overlap as f32 / turn_words.len() as f32
        } else {
            0.5
        };

        Ok(relevance.clamp(0.0, 1.0))
    }

    fn calculate_turn_consistency(
        &self,
        turn: &Turn,
        context: &[Turn],
    ) -> Result<f32, EvaluationError> {
        // Check if entities are consistent with context
        let mut consistency = 1.0;

        for (key, value) in &turn.entities {
            for ctx_turn in context.iter().rev() {
                if let Some(ctx_value) = ctx_turn.entities.get(key) {
                    if ctx_value != value {
                        consistency -= 0.2; // Penalty for inconsistency
                    }
                    break;
                }
            }
        }

        Ok(consistency.max(0.0))
    }

    fn calculate_information_contribution(
        &self,
        turn: &Turn,
        context: &[Turn],
    ) -> Result<f32, EvaluationError> {
        let turn_words: HashSet<_> = self.tokenize(&turn.content).into_iter().collect();
        let mut context_words = HashSet::new();

        for ctx_turn in context {
            context_words.extend(self.tokenize(&ctx_turn.content));
        }

        // New information = words not in context
        let new_words = turn_words.difference(&context_words).count();
        let contribution = if !turn_words.is_empty() {
            new_words as f32 / turn_words.len() as f32
        } else {
            0.0
        };

        Ok(contribution.clamp(0.0, 1.0))
    }

    fn calculate_goal_progress(&self, turn: &Turn, goal: &str) -> f32 {
        let turn_words: HashSet<_> = self.tokenize(&turn.content).into_iter().collect();
        let goal_words: HashSet<_> = self.tokenize(goal).into_iter().collect();

        let overlap = turn_words.intersection(&goal_words).count();
        if !goal_words.is_empty() {
            overlap as f32 / goal_words.len() as f32
        } else {
            0.5
        }
    }

    fn extract_topic(&self, turn: &Turn) -> String {
        // Extract main topic from turn (simplified: use first noun-like word or intent)
        if let Some(intent) = &turn.intent {
            return intent.clone();
        }

        let words = self.tokenize(&turn.content);
        words
            .iter()
            .find(|w| w.len() > 4)
            .cloned()
            .unwrap_or_else(|| "general".to_string())
    }

    fn evaluate_topic_transition(
        &self,
        from: &str,
        to: &str,
        prev_turn: &Turn,
        curr_turn: &Turn,
    ) -> f32 {
        // Evaluate quality of topic transition
        if from == to {
            return 1.0;
        }

        // Check if transition is signaled
        let transition_words = vec!["now", "next", "also", "regarding", "about"];
        let curr_words = self.tokenize(&curr_turn.content);

        let has_signal = curr_words
            .iter()
            .any(|w| transition_words.contains(&w.as_str()));

        if has_signal {
            0.9
        } else {
            0.6
        }
    }

    fn calculate_context_reference_score(&self, turn: &Turn, context: &[Turn]) -> f32 {
        // Check for pronouns and references that require context
        let pronouns = vec!["it", "that", "this", "they", "them", "there"];
        let turn_words = self.tokenize(&turn.content);

        let has_references = turn_words.iter().any(|w| pronouns.contains(&w.as_str()));

        if has_references {
            // Good if there's context to refer to
            if !context.is_empty() {
                0.9
            } else {
                0.3 // Bad if no context
            }
        } else {
            0.7 // Neutral if no references
        }
    }

    fn calculate_recall_score(&self, turn: &Turn, early_context: &[Turn]) -> f32 {
        let turn_words: HashSet<_> = self.tokenize(&turn.content).into_iter().collect();
        let mut early_words = HashSet::new();

        for ctx_turn in early_context {
            early_words.extend(self.tokenize(&ctx_turn.content));
        }

        let recalls = turn_words.intersection(&early_words).count();
        if !turn_words.is_empty() {
            (recalls as f32 / turn_words.len() as f32).min(1.0)
        } else {
            0.0
        }
    }

    fn contains_goal_completion(&self, turn: &Turn, goal: &str) -> bool {
        let completion_words = vec!["done", "complete", "finished", "booked", "confirmed"];
        let turn_words = self.tokenize(&turn.content);
        let goal_words: HashSet<_> = self.tokenize(goal).into_iter().collect();

        let has_completion = turn_words
            .iter()
            .any(|w| completion_words.contains(&w.as_str()));

        let has_goal_words = turn_words.iter().any(|w| goal_words.contains(w));

        has_completion && has_goal_words
    }

    fn calculate_overall_quality(
        &self,
        context_coherence: f32,
        topic_consistency: f32,
        information_consistency: f32,
        progression_quality: f32,
        context_tracking: f32,
        memory_score: f32,
        goal_achievement: f32,
    ) -> f32 {
        (context_coherence * 0.25
            + topic_consistency * 0.2
            + information_consistency * 0.2
            + progression_quality * 0.15
            + context_tracking * 0.1
            + memory_score * 0.05
            + goal_achievement * 0.05)
            .clamp(0.0, 1.0)
    }

    fn calculate_statistics(
        &self,
        dialogue: &MultiTurnDialogue,
        violations: &[ContextViolation],
    ) -> DialogueStatistics {
        let total_turns = dialogue.turns.len();

        let avg_turn_length = if !dialogue.turns.is_empty() {
            dialogue
                .turns
                .iter()
                .map(|t| self.tokenize(&t.content).len() as f32)
                .sum::<f32>()
                / dialogue.turns.len() as f32
        } else {
            0.0
        };

        // Count topic changes (simplified)
        let mut topics = HashSet::new();
        for turn in &dialogue.turns {
            topics.insert(self.extract_topic(turn));
        }
        let num_topic_changes = topics.len().saturating_sub(1);

        let num_speakers = dialogue
            .turns
            .iter()
            .map(|t| &t.speaker)
            .collect::<HashSet<_>>()
            .len();

        DialogueStatistics {
            total_turns,
            avg_turn_length,
            num_topic_changes,
            num_speakers,
            duration: None,
            num_violations: violations.len(),
        }
    }

    fn generate_recommendations(
        &self,
        analyses: &[TurnAnalysis],
        violations: &[ContextViolation],
        context_coherence: f32,
        topic_consistency: f32,
        information_consistency: f32,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if context_coherence < self.config.coherence_threshold {
            recommendations.push(format!(
                "Improve context coherence (current: {:.2}, threshold: {:.2}). Ensure turns build on previous context.",
                context_coherence, self.config.coherence_threshold
            ));
        }

        if topic_consistency < 0.7 {
            recommendations.push(format!(
                "Enhance topic consistency (current: {:.2}). Signal topic transitions more clearly.",
                topic_consistency
            ));
        }

        if information_consistency < 0.8 {
            recommendations.push(format!(
                "Improve information consistency (current: {:.2}). Avoid contradictions across turns.",
                information_consistency
            ));
        }

        if !violations.is_empty() {
            let high_severity = violations.iter().filter(|v| v.severity > 0.7).count();
            if high_severity > 0 {
                recommendations.push(format!(
                    "Address {} high-severity context violations. Review consistency of entities and references.",
                    high_severity
                ));
            }
        }

        let low_contribution_turns = analyses
            .iter()
            .filter(|a| a.information_contribution < 0.3)
            .count();
        if low_contribution_turns > analyses.len() / 3 {
            recommendations.push(format!(
                "{} turns have low information contribution. Ensure each turn adds value.",
                low_contribution_turns
            ));
        }

        if recommendations.is_empty() {
            recommendations.push("Multi-turn dialogue quality meets expectations.".to_string());
        }

        recommendations
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_turn_evaluator_creation() {
        let config = MultiTurnConfig::default();
        let evaluator = MultiTurnDialogueEvaluator::new(config);
        assert!(evaluator.is_ok());
    }

    #[test]
    fn test_turn_creation() {
        let turn = Turn::new("user", "Hello", 0);
        assert_eq!(turn.speaker, "user");
        assert_eq!(turn.content, "Hello");
        assert_eq!(turn.turn_index, 0);
    }

    #[test]
    fn test_dialogue_creation() {
        let mut dialogue = MultiTurnDialogue::new("test");
        dialogue.add_turn(Turn::new("user", "Hi", 0));
        dialogue.add_turn(Turn::new("system", "Hello", 1));

        assert_eq!(dialogue.num_turns(), 2);
    }

    #[test]
    fn test_dialogue_evaluation() {
        let config = MultiTurnConfig::default();
        let evaluator = MultiTurnDialogueEvaluator::new(config).unwrap();

        let mut dialogue = MultiTurnDialogue::new("test");
        dialogue.add_turn(Turn::new("user", "I want to book a flight", 0));
        dialogue.add_turn(Turn::new("system", "Where would you like to go?", 1));
        dialogue.add_turn(Turn::new("user", "New York", 2));

        let result = evaluator.evaluate(&dialogue);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.overall_quality >= 0.0 && result.overall_quality <= 1.0);
    }

    #[test]
    fn test_empty_dialogue() {
        let config = MultiTurnConfig::default();
        let evaluator = MultiTurnDialogueEvaluator::new(config).unwrap();

        let dialogue = MultiTurnDialogue::new("empty");
        let result = evaluator.evaluate(&dialogue);
        assert!(result.is_err());
    }

    #[test]
    fn test_context_coherence() {
        let config = MultiTurnConfig::default();
        let evaluator = MultiTurnDialogueEvaluator::new(config).unwrap();

        let turns = vec![
            Turn::new("user", "I need a hotel", 0),
            Turn::new("system", "Where would you like to stay?", 1),
            Turn::new("user", "In Paris", 2),
        ];

        let analyses = evaluator
            .analyze_turns(&MultiTurnDialogue {
                dialogue_id: "test".to_string(),
                turns: turns.clone(),
                goal: None,
                topics: Vec::new(),
                metadata: HashMap::new(),
            })
            .unwrap();

        let coherence = evaluator
            .evaluate_context_coherence(&turns, &analyses)
            .unwrap();
        assert!(coherence > 0.5);
    }

    #[test]
    fn test_topic_extraction() {
        let config = MultiTurnConfig::default();
        let evaluator = MultiTurnDialogueEvaluator::new(config).unwrap();

        let turn = Turn::new("user", "I want to book a flight", 0).with_intent("book_flight");

        let topic = evaluator.extract_topic(&turn);
        assert_eq!(topic, "book_flight");
    }

    #[test]
    fn test_information_contribution() {
        let config = MultiTurnConfig::default();
        let evaluator = MultiTurnDialogueEvaluator::new(config).unwrap();

        let context = vec![Turn::new("user", "Hello there", 0)];
        let turn = Turn::new("user", "I need help with booking", 1);

        let contribution = evaluator
            .calculate_information_contribution(&turn, &context)
            .unwrap();

        assert!(contribution > 0.5); // Most words are new
    }

    #[test]
    fn test_context_violations() {
        let config = MultiTurnConfig::default();
        let evaluator = MultiTurnDialogueEvaluator::new(config).unwrap();

        let mut turn1 = Turn::new("user", "I'm going to Paris", 0);
        turn1
            .entities
            .insert("destination".to_string(), "Paris".to_string());

        let mut turn2 = Turn::new("user", "Actually to London", 1);
        turn2
            .entities
            .insert("destination".to_string(), "London".to_string());

        let (consistency, violations) = evaluator
            .check_information_consistency(&vec![turn1, turn2])
            .unwrap();

        assert!(!violations.is_empty()); // Should detect contradiction
        assert!(consistency < 1.0);
    }

    #[test]
    fn test_goal_achievement() {
        let config = MultiTurnConfig::default();
        let evaluator = MultiTurnDialogueEvaluator::new(config).unwrap();

        let mut dialogue = MultiTurnDialogue::new("booking");
        dialogue.goal = Some("book a flight".to_string());
        dialogue.add_turn(Turn::new("user", "I want to book a flight", 0));
        dialogue.add_turn(Turn::new("system", "Flight booked successfully", 1));

        let achievement = evaluator.evaluate_goal_achievement(&dialogue).unwrap();
        assert!(achievement > 0.3);
    }
}
