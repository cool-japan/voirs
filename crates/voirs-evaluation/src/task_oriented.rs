//! # Task-Oriented Evaluation
//!
//! Evaluation metrics for task-oriented dialogue systems and goal-driven interactions.
//! This module provides comprehensive assessment of task completion, user goals,
//! and dialogue effectiveness in task-oriented scenarios.

use crate::traits::EvaluationResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Task-oriented evaluation error types
#[derive(Debug, Error)]
pub enum TaskOrientedError {
    #[error("Invalid task definition: {0}")]
    InvalidTask(String),
    #[error("Dialogue analysis failed: {0}")]
    DialogueAnalysisFailed(String),
    #[error("Missing required slot: {0}")]
    MissingSlot(String),
    #[error("Task evaluation error: {0}")]
    EvaluationError(String),
}

/// Task type for categorizing different dialogue tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    /// Information seeking (e.g., weather, directions)
    InformationSeeking,
    /// Transaction (e.g., booking, purchasing)
    Transaction,
    /// Troubleshooting (e.g., technical support)
    Troubleshooting,
    /// Recommendation (e.g., restaurant, product suggestions)
    Recommendation,
    /// Navigation (e.g., finding location, menu navigation)
    Navigation,
    /// Configuration (e.g., settings, preferences)
    Configuration,
    /// Other task types
    Other,
}

/// Task completion status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Task completed successfully
    Completed,
    /// Task partially completed
    PartiallyCompleted,
    /// Task failed
    Failed,
    /// Task abandoned by user
    Abandoned,
    /// Task in progress
    InProgress,
}

/// Slot filling status for information gathering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotFillingStatus {
    /// Slot name
    pub slot_name: String,
    /// Whether slot is required
    pub required: bool,
    /// Whether slot is filled
    pub filled: bool,
    /// Confidence of slot value (0.0-1.0)
    pub confidence: f32,
    /// Number of turns to fill this slot
    pub turns_to_fill: u32,
}

/// Task definition with goals and constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDefinition {
    /// Task identifier
    pub task_id: String,
    /// Task type
    pub task_type: TaskType,
    /// Required slots to complete task
    pub required_slots: Vec<String>,
    /// Optional slots
    pub optional_slots: Vec<String>,
    /// Success criteria
    pub success_criteria: Vec<String>,
    /// Maximum expected turns
    pub max_expected_turns: u32,
}

/// Dialogue turn in task-oriented context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskTurn {
    /// Turn index
    pub turn_index: u32,
    /// Speaker (user or system)
    pub speaker: Speaker,
    /// Utterance text
    pub text: String,
    /// Extracted slots from this turn
    pub extracted_slots: HashMap<String, String>,
    /// Intent detected
    pub intent: Option<String>,
    /// Action taken by system
    pub action: Option<SystemAction>,
}

/// Speaker in dialogue
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Speaker {
    /// User/customer
    User,
    /// System/agent
    System,
}

/// System action in task-oriented dialogue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemAction {
    /// Request information from user
    RequestSlot(String),
    /// Confirm slot value
    ConfirmSlot(String),
    /// Provide information
    Inform(Vec<String>),
    /// Offer options
    Offer(Vec<String>),
    /// Acknowledge user input
    Acknowledge,
    /// Request clarification
    Clarify,
    /// Complete task
    Complete,
    /// Reject or fail
    Reject(String),
}

/// Task-oriented evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskOrientedScore {
    /// Overall task success rate (0.0-1.0)
    pub task_success_rate: f32,
    /// Task completion status
    pub task_status: TaskStatus,
    /// Slot filling accuracy (0.0-1.0)
    pub slot_filling_accuracy: f32,
    /// Dialogue efficiency (turns used / expected turns)
    pub dialogue_efficiency: f32,
    /// User goal achievement (0.0-1.0)
    pub goal_achievement: f32,
    /// Intent recognition accuracy (0.0-1.0)
    pub intent_accuracy: f32,
    /// Error recovery rate (0.0-1.0)
    pub error_recovery_rate: f32,
    /// Slot filling details
    pub slot_status: Vec<SlotFillingStatus>,
    /// Number of turns used
    pub turns_used: u32,
    /// Number of errors/corrections
    pub error_count: u32,
    /// Detailed breakdown
    pub details: TaskEvaluationDetails,
}

/// Detailed task evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEvaluationDetails {
    /// Required slots filled
    pub required_slots_filled: u32,
    /// Total required slots
    pub total_required_slots: u32,
    /// Optional slots filled
    pub optional_slots_filled: u32,
    /// Total optional slots
    pub total_optional_slots: u32,
    /// Success criteria met
    pub criteria_met: Vec<String>,
    /// Success criteria failed
    pub criteria_failed: Vec<String>,
    /// Average slot fill time (turns)
    pub avg_slot_fill_time: f32,
    /// Clarification requests made
    pub clarification_requests: u32,
    /// Confirmation requests made
    pub confirmation_requests: u32,
}

/// Task-oriented evaluation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskOrientedConfig {
    /// Minimum slot confidence threshold
    pub min_slot_confidence: f32,
    /// Maximum acceptable turns for task
    pub max_acceptable_turns: u32,
    /// Penalty for exceeding turn limit
    pub turn_limit_penalty: f32,
    /// Weight for task success (0.0-1.0)
    pub task_success_weight: f32,
    /// Weight for efficiency (0.0-1.0)
    pub efficiency_weight: f32,
    /// Weight for slot accuracy (0.0-1.0)
    pub slot_accuracy_weight: f32,
}

impl Default for TaskOrientedConfig {
    fn default() -> Self {
        Self {
            min_slot_confidence: 0.7,
            max_acceptable_turns: 20,
            turn_limit_penalty: 0.1,
            task_success_weight: 0.5,
            efficiency_weight: 0.3,
            slot_accuracy_weight: 0.2,
        }
    }
}

/// Task-oriented evaluator for dialogue systems
pub struct TaskOrientedEvaluator {
    config: TaskOrientedConfig,
}

impl TaskOrientedEvaluator {
    /// Create a new task-oriented evaluator
    pub fn new(config: TaskOrientedConfig) -> Self {
        Self { config }
    }

    /// Evaluate a task-oriented dialogue
    pub fn evaluate_dialogue(
        &self,
        task: &TaskDefinition,
        turns: &[TaskTurn],
    ) -> EvaluationResult<TaskOrientedScore> {
        // Analyze slot filling
        let slot_status = self.analyze_slot_filling(task, turns)?;
        let slot_filling_accuracy = self.calculate_slot_accuracy(&slot_status);

        // Determine task status
        let task_status = self.determine_task_status(task, &slot_status);
        let task_success_rate = if task_status == TaskStatus::Completed {
            1.0
        } else if task_status == TaskStatus::PartiallyCompleted {
            0.5
        } else {
            0.0
        };

        // Calculate efficiency
        let turns_used = turns.len() as u32;
        let dialogue_efficiency = if turns_used <= task.max_expected_turns {
            1.0
        } else {
            (task.max_expected_turns as f32 / turns_used as f32)
                * (1.0 - self.config.turn_limit_penalty)
        };

        // Analyze intents and actions
        let intent_accuracy = self.calculate_intent_accuracy(turns);
        let error_count = self.count_errors(turns);
        let error_recovery_rate = self.calculate_error_recovery_rate(turns, error_count);

        // Calculate goal achievement
        let goal_achievement = self.calculate_goal_achievement(task, &slot_status, task_status);

        // Create detailed breakdown
        let details = self.create_evaluation_details(task, &slot_status, turns);

        Ok(TaskOrientedScore {
            task_success_rate,
            task_status,
            slot_filling_accuracy,
            dialogue_efficiency,
            goal_achievement,
            intent_accuracy,
            error_recovery_rate,
            slot_status,
            turns_used,
            error_count,
            details,
        })
    }

    /// Analyze slot filling progress
    fn analyze_slot_filling(
        &self,
        task: &TaskDefinition,
        turns: &[TaskTurn],
    ) -> EvaluationResult<Vec<SlotFillingStatus>> {
        let mut slot_status = Vec::new();
        let mut filled_slots: HashMap<String, (u32, f32)> = HashMap::new();

        // Track when each slot was filled
        for (turn_idx, turn) in turns.iter().enumerate() {
            for (slot_name, _value) in &turn.extracted_slots {
                filled_slots
                    .entry(slot_name.clone())
                    .or_insert((turn_idx as u32 + 1, 0.8));
            }
        }

        // Check required slots
        for slot_name in &task.required_slots {
            let (turns_to_fill, confidence) =
                filled_slots.get(slot_name).copied().unwrap_or((0, 0.0));
            slot_status.push(SlotFillingStatus {
                slot_name: slot_name.clone(),
                required: true,
                filled: filled_slots.contains_key(slot_name),
                confidence,
                turns_to_fill,
            });
        }

        // Check optional slots
        for slot_name in &task.optional_slots {
            let (turns_to_fill, confidence) =
                filled_slots.get(slot_name).copied().unwrap_or((0, 0.0));
            slot_status.push(SlotFillingStatus {
                slot_name: slot_name.clone(),
                required: false,
                filled: filled_slots.contains_key(slot_name),
                confidence,
                turns_to_fill,
            });
        }

        Ok(slot_status)
    }

    /// Calculate slot filling accuracy
    fn calculate_slot_accuracy(&self, slot_status: &[SlotFillingStatus]) -> f32 {
        if slot_status.is_empty() {
            return 1.0;
        }

        let mut total_accuracy = 0.0;
        let mut count = 0;

        for slot in slot_status {
            if slot.filled && slot.confidence >= self.config.min_slot_confidence {
                total_accuracy += slot.confidence;
                count += 1;
            } else if slot.required {
                // Required slots that aren't filled contribute 0 to accuracy
                count += 1;
            }
            // Optional unfilled slots don't count toward accuracy calculation
        }

        if count > 0 {
            total_accuracy / count as f32
        } else {
            0.0
        }
    }

    /// Determine overall task status
    fn determine_task_status(
        &self,
        task: &TaskDefinition,
        slot_status: &[SlotFillingStatus],
    ) -> TaskStatus {
        let required_filled = slot_status
            .iter()
            .filter(|s| s.required && s.filled && s.confidence >= self.config.min_slot_confidence)
            .count();

        if required_filled == task.required_slots.len() {
            TaskStatus::Completed
        } else if required_filled > 0 {
            TaskStatus::PartiallyCompleted
        } else {
            TaskStatus::Failed
        }
    }

    /// Calculate intent recognition accuracy
    fn calculate_intent_accuracy(&self, turns: &[TaskTurn]) -> f32 {
        let user_turns: Vec<_> = turns
            .iter()
            .filter(|t| t.speaker == Speaker::User)
            .collect();
        if user_turns.is_empty() {
            return 1.0;
        }

        let recognized = user_turns.iter().filter(|t| t.intent.is_some()).count();
        recognized as f32 / user_turns.len() as f32
    }

    /// Count dialogue errors
    fn count_errors(&self, turns: &[TaskTurn]) -> u32 {
        turns
            .iter()
            .filter(|t| {
                if let Some(SystemAction::Reject(_)) = t.action {
                    true
                } else if let Some(SystemAction::Clarify) = t.action {
                    true
                } else {
                    false
                }
            })
            .count() as u32
    }

    /// Calculate error recovery rate
    fn calculate_error_recovery_rate(&self, turns: &[TaskTurn], error_count: u32) -> f32 {
        if error_count == 0 {
            return 1.0;
        }

        let mut recovered = 0u32;
        let mut in_error = false;

        for turn in turns {
            if let Some(action) = &turn.action {
                match action {
                    SystemAction::Reject(_) | SystemAction::Clarify => {
                        in_error = true;
                    }
                    SystemAction::Acknowledge | SystemAction::Complete => {
                        if in_error {
                            recovered += 1;
                            in_error = false;
                        }
                    }
                    _ => {}
                }
            }
        }

        recovered as f32 / error_count as f32
    }

    /// Calculate goal achievement score
    fn calculate_goal_achievement(
        &self,
        task: &TaskDefinition,
        slot_status: &[SlotFillingStatus],
        task_status: TaskStatus,
    ) -> f32 {
        let slot_completion = slot_status
            .iter()
            .filter(|s| s.required && s.filled)
            .count() as f32
            / task.required_slots.len().max(1) as f32;

        let status_score = match task_status {
            TaskStatus::Completed => 1.0,
            TaskStatus::PartiallyCompleted => 0.5,
            _ => 0.0,
        };

        (slot_completion + status_score) / 2.0
    }

    /// Create detailed evaluation breakdown
    fn create_evaluation_details(
        &self,
        task: &TaskDefinition,
        slot_status: &[SlotFillingStatus],
        turns: &[TaskTurn],
    ) -> TaskEvaluationDetails {
        let required_slots_filled = slot_status
            .iter()
            .filter(|s| s.required && s.filled)
            .count() as u32;

        let optional_slots_filled = slot_status
            .iter()
            .filter(|s| !s.required && s.filled)
            .count() as u32;

        let avg_slot_fill_time = if !slot_status.is_empty() {
            slot_status
                .iter()
                .filter(|s| s.filled)
                .map(|s| s.turns_to_fill as f32)
                .sum::<f32>()
                / slot_status.iter().filter(|s| s.filled).count().max(1) as f32
        } else {
            0.0
        };

        let clarification_requests = turns
            .iter()
            .filter(|t| matches!(t.action, Some(SystemAction::Clarify)))
            .count() as u32;

        let confirmation_requests = turns
            .iter()
            .filter(|t| matches!(t.action, Some(SystemAction::ConfirmSlot(_))))
            .count() as u32;

        TaskEvaluationDetails {
            required_slots_filled,
            total_required_slots: task.required_slots.len() as u32,
            optional_slots_filled,
            total_optional_slots: task.optional_slots.len() as u32,
            criteria_met: task.success_criteria.clone(),
            criteria_failed: Vec::new(),
            avg_slot_fill_time,
            clarification_requests,
            confirmation_requests,
        }
    }

    /// Batch evaluate multiple dialogues
    pub fn evaluate_batch(
        &self,
        tasks: &[(TaskDefinition, Vec<TaskTurn>)],
    ) -> EvaluationResult<Vec<TaskOrientedScore>> {
        tasks
            .iter()
            .map(|(task, turns)| self.evaluate_dialogue(task, turns))
            .collect()
    }

    /// Calculate aggregate metrics across multiple dialogues
    pub fn calculate_aggregate_metrics(
        &self,
        scores: &[TaskOrientedScore],
    ) -> AggregateTaskMetrics {
        if scores.is_empty() {
            return AggregateTaskMetrics::default();
        }

        let avg_success_rate =
            scores.iter().map(|s| s.task_success_rate).sum::<f32>() / scores.len() as f32;

        let avg_efficiency =
            scores.iter().map(|s| s.dialogue_efficiency).sum::<f32>() / scores.len() as f32;

        let avg_slot_accuracy =
            scores.iter().map(|s| s.slot_filling_accuracy).sum::<f32>() / scores.len() as f32;

        let avg_goal_achievement =
            scores.iter().map(|s| s.goal_achievement).sum::<f32>() / scores.len() as f32;

        let completed_tasks = scores
            .iter()
            .filter(|s| s.task_status == TaskStatus::Completed)
            .count();

        let avg_turns =
            scores.iter().map(|s| s.turns_used as f32).sum::<f32>() / scores.len() as f32;

        AggregateTaskMetrics {
            avg_success_rate,
            avg_efficiency,
            avg_slot_accuracy,
            avg_goal_achievement,
            total_tasks: scores.len(),
            completed_tasks,
            avg_turns,
        }
    }
}

/// Aggregate metrics across multiple task evaluations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregateTaskMetrics {
    /// Average task success rate
    pub avg_success_rate: f32,
    /// Average dialogue efficiency
    pub avg_efficiency: f32,
    /// Average slot filling accuracy
    pub avg_slot_accuracy: f32,
    /// Average goal achievement
    pub avg_goal_achievement: f32,
    /// Total number of tasks evaluated
    pub total_tasks: usize,
    /// Number of completed tasks
    pub completed_tasks: usize,
    /// Average number of turns
    pub avg_turns: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_oriented_evaluator_creation() {
        let config = TaskOrientedConfig::default();
        let evaluator = TaskOrientedEvaluator::new(config);
        assert_eq!(evaluator.config.min_slot_confidence, 0.7);
    }

    #[test]
    fn test_successful_task_completion() {
        let config = TaskOrientedConfig::default();
        let evaluator = TaskOrientedEvaluator::new(config);

        let task = TaskDefinition {
            task_id: "booking_001".to_string(),
            task_type: TaskType::Transaction,
            required_slots: vec!["date".to_string(), "time".to_string(), "guests".to_string()],
            optional_slots: vec!["preferences".to_string()],
            success_criteria: vec!["all_slots_filled".to_string()],
            max_expected_turns: 10,
        };

        let turns = vec![
            TaskTurn {
                turn_index: 0,
                speaker: Speaker::User,
                text: "I want to book a table".to_string(),
                extracted_slots: HashMap::new(),
                intent: Some("book_table".to_string()),
                action: None,
            },
            TaskTurn {
                turn_index: 1,
                speaker: Speaker::System,
                text: "For which date?".to_string(),
                extracted_slots: HashMap::new(),
                intent: None,
                action: Some(SystemAction::RequestSlot("date".to_string())),
            },
            TaskTurn {
                turn_index: 2,
                speaker: Speaker::User,
                text: "Tomorrow at 7pm for 4 people".to_string(),
                extracted_slots: {
                    let mut map = HashMap::new();
                    map.insert("date".to_string(), "tomorrow".to_string());
                    map.insert("time".to_string(), "7pm".to_string());
                    map.insert("guests".to_string(), "4".to_string());
                    map
                },
                intent: Some("provide_details".to_string()),
                action: None,
            },
            TaskTurn {
                turn_index: 3,
                speaker: Speaker::System,
                text: "Booking confirmed".to_string(),
                extracted_slots: HashMap::new(),
                intent: None,
                action: Some(SystemAction::Complete),
            },
        ];

        let result = evaluator.evaluate_dialogue(&task, &turns).unwrap();

        assert_eq!(result.task_status, TaskStatus::Completed);
        assert!(result.task_success_rate > 0.9);
        assert_eq!(result.turns_used, 4);
        assert!(result.slot_filling_accuracy > 0.7);
    }

    #[test]
    fn test_partial_task_completion() {
        let config = TaskOrientedConfig::default();
        let evaluator = TaskOrientedEvaluator::new(config);

        let task = TaskDefinition {
            task_id: "info_001".to_string(),
            task_type: TaskType::InformationSeeking,
            required_slots: vec!["location".to_string(), "date".to_string()],
            optional_slots: vec![],
            success_criteria: vec![],
            max_expected_turns: 5,
        };

        let turns = vec![
            TaskTurn {
                turn_index: 0,
                speaker: Speaker::User,
                text: "What's the weather?".to_string(),
                extracted_slots: HashMap::new(),
                intent: Some("weather_query".to_string()),
                action: None,
            },
            TaskTurn {
                turn_index: 1,
                speaker: Speaker::System,
                text: "Where?".to_string(),
                extracted_slots: HashMap::new(),
                intent: None,
                action: Some(SystemAction::RequestSlot("location".to_string())),
            },
            TaskTurn {
                turn_index: 2,
                speaker: Speaker::User,
                text: "London".to_string(),
                extracted_slots: {
                    let mut map = HashMap::new();
                    map.insert("location".to_string(), "London".to_string());
                    map
                },
                intent: Some("provide_location".to_string()),
                action: None,
            },
        ];

        let result = evaluator.evaluate_dialogue(&task, &turns).unwrap();

        assert_eq!(result.task_status, TaskStatus::PartiallyCompleted);
        assert!(result.task_success_rate < 1.0);
        assert_eq!(result.details.required_slots_filled, 1);
        assert_eq!(result.details.total_required_slots, 2);
    }

    #[test]
    fn test_slot_filling_analysis() {
        let config = TaskOrientedConfig::default();
        let evaluator = TaskOrientedEvaluator::new(config);

        let task = TaskDefinition {
            task_id: "test_001".to_string(),
            task_type: TaskType::Transaction,
            required_slots: vec!["item".to_string(), "quantity".to_string()],
            optional_slots: vec!["color".to_string()],
            success_criteria: vec![],
            max_expected_turns: 5,
        };

        let turns = vec![TaskTurn {
            turn_index: 0,
            speaker: Speaker::User,
            text: "I want 5 red shirts".to_string(),
            extracted_slots: {
                let mut map = HashMap::new();
                map.insert("item".to_string(), "shirts".to_string());
                map.insert("quantity".to_string(), "5".to_string());
                map.insert("color".to_string(), "red".to_string());
                map
            },
            intent: Some("purchase".to_string()),
            action: None,
        }];

        let slot_status = evaluator.analyze_slot_filling(&task, &turns).unwrap();

        assert_eq!(slot_status.len(), 3);
        assert_eq!(slot_status.iter().filter(|s| s.filled).count(), 3);
    }

    #[test]
    fn test_dialogue_efficiency() {
        let config = TaskOrientedConfig::default();
        let evaluator = TaskOrientedEvaluator::new(config);

        let task = TaskDefinition {
            task_id: "efficiency_test".to_string(),
            task_type: TaskType::InformationSeeking,
            required_slots: vec!["query".to_string()],
            optional_slots: vec![],
            success_criteria: vec![],
            max_expected_turns: 3,
        };

        // Efficient dialogue (2 turns)
        let efficient_turns = vec![
            TaskTurn {
                turn_index: 0,
                speaker: Speaker::User,
                text: "What time is it?".to_string(),
                extracted_slots: {
                    let mut map = HashMap::new();
                    map.insert("query".to_string(), "current_time".to_string());
                    map
                },
                intent: Some("time_query".to_string()),
                action: None,
            },
            TaskTurn {
                turn_index: 1,
                speaker: Speaker::System,
                text: "It's 3:00 PM".to_string(),
                extracted_slots: HashMap::new(),
                intent: None,
                action: Some(SystemAction::Inform(vec!["time".to_string()])),
            },
        ];

        let result = evaluator
            .evaluate_dialogue(&task, &efficient_turns)
            .unwrap();
        assert!(result.dialogue_efficiency >= 0.9);
    }

    #[test]
    fn test_error_recovery() {
        let config = TaskOrientedConfig::default();
        let evaluator = TaskOrientedEvaluator::new(config);

        let turns = vec![
            TaskTurn {
                turn_index: 0,
                speaker: Speaker::User,
                text: "unclear request".to_string(),
                extracted_slots: HashMap::new(),
                intent: None,
                action: None,
            },
            TaskTurn {
                turn_index: 1,
                speaker: Speaker::System,
                text: "Could you clarify?".to_string(),
                extracted_slots: HashMap::new(),
                intent: None,
                action: Some(SystemAction::Clarify),
            },
            TaskTurn {
                turn_index: 2,
                speaker: Speaker::User,
                text: "I need help".to_string(),
                extracted_slots: HashMap::new(),
                intent: Some("help_request".to_string()),
                action: None,
            },
            TaskTurn {
                turn_index: 3,
                speaker: Speaker::System,
                text: "Here's how I can help".to_string(),
                extracted_slots: HashMap::new(),
                intent: None,
                action: Some(SystemAction::Acknowledge),
            },
        ];

        let error_count = evaluator.count_errors(&turns);
        assert_eq!(error_count, 1);

        let recovery_rate = evaluator.calculate_error_recovery_rate(&turns, error_count);
        assert!(recovery_rate > 0.8);
    }

    #[test]
    fn test_aggregate_metrics() {
        let config = TaskOrientedConfig::default();
        let evaluator = TaskOrientedEvaluator::new(config);

        let scores = vec![
            TaskOrientedScore {
                task_success_rate: 1.0,
                task_status: TaskStatus::Completed,
                slot_filling_accuracy: 0.9,
                dialogue_efficiency: 0.95,
                goal_achievement: 1.0,
                intent_accuracy: 0.85,
                error_recovery_rate: 1.0,
                slot_status: vec![],
                turns_used: 4,
                error_count: 0,
                details: TaskEvaluationDetails {
                    required_slots_filled: 3,
                    total_required_slots: 3,
                    optional_slots_filled: 0,
                    total_optional_slots: 0,
                    criteria_met: vec![],
                    criteria_failed: vec![],
                    avg_slot_fill_time: 2.0,
                    clarification_requests: 0,
                    confirmation_requests: 1,
                },
            },
            TaskOrientedScore {
                task_success_rate: 0.5,
                task_status: TaskStatus::PartiallyCompleted,
                slot_filling_accuracy: 0.6,
                dialogue_efficiency: 0.7,
                goal_achievement: 0.5,
                intent_accuracy: 0.75,
                error_recovery_rate: 0.8,
                slot_status: vec![],
                turns_used: 8,
                error_count: 2,
                details: TaskEvaluationDetails {
                    required_slots_filled: 2,
                    total_required_slots: 3,
                    optional_slots_filled: 0,
                    total_optional_slots: 1,
                    criteria_met: vec![],
                    criteria_failed: vec![],
                    avg_slot_fill_time: 3.5,
                    clarification_requests: 2,
                    confirmation_requests: 0,
                },
            },
        ];

        let aggregate = evaluator.calculate_aggregate_metrics(&scores);

        assert_eq!(aggregate.total_tasks, 2);
        assert_eq!(aggregate.completed_tasks, 1);
        assert!((aggregate.avg_success_rate - 0.75).abs() < 0.01);
        assert_eq!(aggregate.avg_turns, 6.0);
    }
}
