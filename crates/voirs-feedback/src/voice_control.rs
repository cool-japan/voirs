//! Voice Control Support Module
//!
//! Provides voice command recognition and control for hands-free accessibility.
//! Integrates with speech recognition and natural language processing for
//! intuitive voice-based interaction with the feedback system.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

/// Voice control errors
#[derive(Error, Debug)]
#[allow(missing_docs)]
pub enum VoiceControlError {
    /// Recognition failed
    #[error("Voice recognition failed: {message}")]
    RecognitionFailed { message: String },

    /// Command not recognized
    #[error("Command not recognized: {input}")]
    CommandNotRecognized { input: String },

    /// The command was recognized but no real handler has been registered
    /// for it, so there is no action to actually dispatch.
    #[error("No handler registered for command '{command_id}'")]
    HandlerNotRegistered {
        /// The command ID that has no registered handler.
        command_id: String,
    },

    /// A registered handler ran but reported a real failure.
    #[error("Handler for command '{command_id}' failed: {message}")]
    HandlerFailed {
        /// The command ID whose handler failed.
        command_id: String,
        /// The real failure reason reported by the handler.
        message: String,
    },

    /// Ambiguous command
    #[error("Ambiguous command: {input} - matches: {matches:?}")]
    AmbiguousCommand { input: String, matches: Vec<String> },

    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    /// Intent parsing failed
    #[error("Intent parsing failed: {message}")]
    IntentParsingFailed { message: String },
}

/// Result type for voice control operations
pub type VoiceControlResult<T> = Result<T, VoiceControlError>;

/// Voice command definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCommand {
    /// Command ID
    pub command_id: String,
    /// Command name
    pub name: String,
    /// Trigger phrases
    pub trigger_phrases: Vec<String>,
    /// Command category
    pub category: CommandCategory,
    /// Required parameters
    pub parameters: Vec<CommandParameter>,
    /// Description
    pub description: String,
    /// Enabled state
    pub enabled: bool,
    /// Confirmation required
    pub requires_confirmation: bool,
}

/// Command category
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum CommandCategory {
    /// Navigation commands
    Navigation,
    /// Playback control
    Playback,
    /// Settings adjustment
    Settings,
    /// Content interaction
    Content,
    /// System control
    System,
    /// Custom category
    Custom { name: String },
}

/// Command parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: ParameterType,
    /// Required flag
    pub required: bool,
    /// Default value
    pub default_value: Option<String>,
}

/// Parameter type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum ParameterType {
    /// String value
    String,
    /// Number value
    Number,
    /// Boolean value
    Boolean,
    /// Enum from predefined values
    Enum { values: Vec<String> },
}

/// Recognized intent from voice input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceIntent {
    /// Intent ID
    pub intent_id: String,
    /// Matched command
    pub command_id: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Extracted parameters
    pub parameters: HashMap<String, String>,
    /// Original input
    pub original_input: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Voice command execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandExecution {
    /// Execution ID
    pub execution_id: String,
    /// Command ID
    pub command_id: String,
    /// Success status
    pub success: bool,
    /// Result message
    pub message: String,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Timestamp
    pub executed_at: DateTime<Utc>,
}

/// Voice control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceControlConfig {
    /// Enable voice control
    pub enabled: bool,
    /// Recognition language
    pub language: String,
    /// Minimum confidence threshold
    pub min_confidence: f32,
    /// Enable wake word detection
    pub wake_word_enabled: bool,
    /// Wake words
    pub wake_words: Vec<String>,
    /// Enable continuous listening
    pub continuous_listening: bool,
    /// Timeout for command completion (seconds)
    pub command_timeout: u64,
    /// Enable command history
    pub history_enabled: bool,
}

impl Default for VoiceControlConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            language: "en-US".to_string(),
            min_confidence: 0.7,
            wake_word_enabled: true,
            wake_words: vec!["hey voirs".to_string(), "voice command".to_string()],
            continuous_listening: false,
            command_timeout: 5,
            history_enabled: true,
        }
    }
}

/// A real, callable action a command dispatches to. Handlers receive the
/// recognized [`VoiceIntent`] (including its extracted parameters) and
/// report their own genuine success or failure -- there is no synthetic
/// "always succeeds" fallback anywhere in the dispatch path.
pub type CommandHandler = dyn Fn(&VoiceIntent) -> VoiceControlResult<String> + Send + Sync;

/// Voice control manager
pub struct VoiceControlManager {
    /// Configuration
    config: Arc<RwLock<VoiceControlConfig>>,
    /// Registered commands
    commands: Arc<RwLock<HashMap<String, VoiceCommand>>>,
    /// Command history
    history: Arc<RwLock<Vec<VoiceIntent>>>,
    /// Execution log
    executions: Arc<RwLock<Vec<CommandExecution>>>,
    /// Command statistics
    stats: Arc<RwLock<VoiceControlStats>>,
    /// Real handlers dispatched to by [`VoiceControlManager::execute_command`],
    /// keyed by [`VoiceCommand::command_id`]. A command with no registered
    /// handler here has nothing wired up to actually run, and
    /// `execute_command` reports that honestly instead of a fabricated
    /// success.
    handlers: Arc<RwLock<HashMap<String, Arc<CommandHandler>>>>,
}

/// Voice control statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceControlStats {
    /// Total commands recognized
    pub total_commands: usize,
    /// Successful executions
    pub successful_executions: usize,
    /// Failed executions
    pub failed_executions: usize,
    /// Average confidence score
    pub avg_confidence: f32,
    /// Commands by category
    pub commands_by_category: HashMap<String, usize>,
    /// Most used commands
    pub most_used_commands: Vec<(String, usize)>,
}

impl Default for VoiceControlStats {
    fn default() -> Self {
        Self {
            total_commands: 0,
            successful_executions: 0,
            failed_executions: 0,
            avg_confidence: 0.0,
            commands_by_category: HashMap::new(),
            most_used_commands: Vec::new(),
        }
    }
}

impl VoiceControlManager {
    /// Create new voice control manager
    #[must_use]
    pub fn new(config: VoiceControlConfig) -> Self {
        // Build default commands synchronously during construction
        let mut default_command_map = HashMap::new();
        for command in Self::build_default_commands() {
            default_command_map.insert(command.command_id.clone(), command);
        }

        Self {
            config: Arc::new(RwLock::new(config)),
            commands: Arc::new(RwLock::new(default_command_map)),
            history: Arc::new(RwLock::new(Vec::new())),
            executions: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(VoiceControlStats::default())),
            handlers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a real handler that `execute_command` will invoke whenever a
    /// recognized intent's `command_id` matches. The handler receives the
    /// full [`VoiceIntent`] and returns either `Ok(message)` describing what
    /// it really did, or `Err` if the action genuinely failed -- both are
    /// propagated into the resulting [`CommandExecution`] as-is.
    ///
    /// Registering a handler for a `command_id` that has no corresponding
    /// [`VoiceCommand`] is allowed (the handler simply will never be
    /// reached, since `recognize_intent` can never produce that
    /// `command_id`), but is not an error here.
    pub async fn register_handler<F>(&self, command_id: impl Into<String>, handler: F)
    where
        F: Fn(&VoiceIntent) -> VoiceControlResult<String> + Send + Sync + 'static,
    {
        let mut handlers = self.handlers.write().await;
        handlers.insert(command_id.into(), Arc::new(handler));
    }

    /// Remove a previously registered handler, if any.
    pub async fn unregister_handler(&self, command_id: &str) {
        let mut handlers = self.handlers.write().await;
        handlers.remove(command_id);
    }

    /// Build the set of default voice commands
    fn build_default_commands() -> Vec<VoiceCommand> {
        vec![
            VoiceCommand {
                command_id: "play".to_string(),
                name: "Play".to_string(),
                trigger_phrases: vec!["play".to_string(), "start playback".to_string()],
                category: CommandCategory::Playback,
                parameters: vec![],
                description: "Start playback".to_string(),
                enabled: true,
                requires_confirmation: false,
            },
            VoiceCommand {
                command_id: "pause".to_string(),
                name: "Pause".to_string(),
                trigger_phrases: vec!["pause".to_string(), "stop playback".to_string()],
                category: CommandCategory::Playback,
                parameters: vec![],
                description: "Pause playback".to_string(),
                enabled: true,
                requires_confirmation: false,
            },
            VoiceCommand {
                command_id: "next".to_string(),
                name: "Next".to_string(),
                trigger_phrases: vec![
                    "next".to_string(),
                    "skip".to_string(),
                    "next item".to_string(),
                ],
                category: CommandCategory::Navigation,
                parameters: vec![],
                description: "Go to next item".to_string(),
                enabled: true,
                requires_confirmation: false,
            },
            VoiceCommand {
                command_id: "previous".to_string(),
                name: "Previous".to_string(),
                trigger_phrases: vec!["previous".to_string(), "go back".to_string()],
                category: CommandCategory::Navigation,
                parameters: vec![],
                description: "Go to previous item".to_string(),
                enabled: true,
                requires_confirmation: false,
            },
            VoiceCommand {
                command_id: "help".to_string(),
                name: "Help".to_string(),
                trigger_phrases: vec![
                    "help".to_string(),
                    "show help".to_string(),
                    "what can I say".to_string(),
                ],
                category: CommandCategory::System,
                parameters: vec![],
                description: "Show help information".to_string(),
                enabled: true,
                requires_confirmation: false,
            },
        ]
    }

    /// Register a voice command
    pub async fn register_command(&self, command: VoiceCommand) -> VoiceControlResult<()> {
        if command.trigger_phrases.is_empty() {
            return Err(VoiceControlError::ConfigError {
                message: "Command must have at least one trigger phrase".to_string(),
            });
        }

        let mut commands = self.commands.write().await;
        commands.insert(command.command_id.clone(), command);

        Ok(())
    }

    /// Process voice input and recognize intent
    pub async fn recognize_intent(&self, input: &str) -> VoiceControlResult<VoiceIntent> {
        let config = self.config.read().await;
        if !config.enabled {
            return Err(VoiceControlError::ConfigError {
                message: "Voice control is disabled".to_string(),
            });
        }
        drop(config);

        let commands = self.commands.read().await;
        let input_lower = input.to_lowercase();

        // Find matching commands
        let mut matches: Vec<(String, f32)> = Vec::new();

        for (command_id, command) in commands.iter() {
            if !command.enabled {
                continue;
            }

            for trigger in &command.trigger_phrases {
                let confidence = self.calculate_similarity(&input_lower, &trigger.to_lowercase());
                if confidence > 0.5 {
                    matches.push((command_id.clone(), confidence));
                    break;
                }
            }
        }

        if matches.is_empty() {
            return Err(VoiceControlError::CommandNotRecognized {
                input: input.to_string(),
            });
        }

        // Sort by confidence
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (best_command_id, confidence) = matches[0].clone();

        // Check minimum confidence threshold
        let config = self.config.read().await;
        if confidence < config.min_confidence {
            return Err(VoiceControlError::CommandNotRecognized {
                input: input.to_string(),
            });
        }
        drop(config);

        // Extract parameters. `best_command_id` was drawn directly from
        // `commands` above, so this lookup should never miss -- but a real,
        // typed error is returned instead of panicking if that invariant is
        // ever violated.
        let command = commands.get(&best_command_id).ok_or_else(|| {
            VoiceControlError::CommandNotRecognized {
                input: input.to_string(),
            }
        })?;
        let parameters = self.extract_parameters(input, command);

        let intent = VoiceIntent {
            intent_id: uuid::Uuid::new_v4().to_string(),
            command_id: best_command_id,
            confidence,
            parameters,
            original_input: input.to_string(),
            timestamp: Utc::now(),
        };

        // Store in history
        if self.config.read().await.history_enabled {
            let mut history = self.history.write().await;
            history.push(intent.clone());
        }

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.total_commands += 1;
        stats.avg_confidence = (stats.avg_confidence * (stats.total_commands - 1) as f32
            + confidence)
            / stats.total_commands as f32;

        let category_name = format!("{:?}", command.category);
        *stats.commands_by_category.entry(category_name).or_insert(0) += 1;

        Ok(intent)
    }

    /// Execute a voice command.
    ///
    /// Dispatches to the real handler registered for `intent.command_id` via
    /// [`VoiceControlManager::register_handler`], propagating whatever that
    /// handler genuinely reports. If no handler is registered, this reports
    /// a real failure (`success: false`) rather than fabricating success --
    /// consistent with how a missing required parameter is already reported
    /// below.
    pub async fn execute_command(
        &self,
        intent: &VoiceIntent,
    ) -> VoiceControlResult<CommandExecution> {
        let start_time = std::time::Instant::now();

        let command = {
            let commands = self.commands.read().await;
            commands.get(&intent.command_id).cloned().ok_or_else(|| {
                VoiceControlError::CommandNotRecognized {
                    input: intent.command_id.clone(),
                }
            })?
        };

        // Validate required parameters
        for param in &command.parameters {
            if param.required && !intent.parameters.contains_key(&param.name) {
                let execution = CommandExecution {
                    execution_id: uuid::Uuid::new_v4().to_string(),
                    command_id: intent.command_id.clone(),
                    success: false,
                    message: format!("Missing required parameter: {}", param.name),
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    executed_at: Utc::now(),
                };

                self.log_execution(execution.clone()).await;
                return Ok(execution);
            }
        }

        // Look up the real handler registered for this command, if any, then
        // drop the lock before invoking it (handlers are arbitrary user code
        // and must not run while holding our internal lock).
        let handler = {
            let handlers = self.handlers.read().await;
            handlers.get(&intent.command_id).cloned()
        };

        let (success, message) = match handler {
            Some(handler) => match handler(intent) {
                Ok(real_message) => (true, real_message),
                Err(e) => (false, e.to_string()),
            },
            None => (
                false,
                VoiceControlError::HandlerNotRegistered {
                    command_id: intent.command_id.clone(),
                }
                .to_string(),
            ),
        };

        let execution = CommandExecution {
            execution_id: uuid::Uuid::new_v4().to_string(),
            command_id: intent.command_id.clone(),
            success,
            message,
            execution_time_ms: start_time.elapsed().as_millis() as u64,
            executed_at: Utc::now(),
        };

        self.log_execution(execution.clone()).await;

        Ok(execution)
    }

    /// Get command by ID
    pub async fn get_command(&self, command_id: &str) -> Option<VoiceCommand> {
        let commands = self.commands.read().await;
        commands.get(command_id).cloned()
    }

    /// List all commands by category
    pub async fn list_commands_by_category(&self, category: CommandCategory) -> Vec<VoiceCommand> {
        let commands = self.commands.read().await;
        commands
            .values()
            .filter(|cmd| cmd.category == category && cmd.enabled)
            .cloned()
            .collect()
    }

    /// Get command history
    pub async fn get_history(&self, limit: Option<usize>) -> Vec<VoiceIntent> {
        let history = self.history.read().await;
        let limit = limit.unwrap_or(history.len());
        history.iter().rev().take(limit).cloned().collect()
    }

    /// Get execution log
    pub async fn get_executions(&self, limit: Option<usize>) -> Vec<CommandExecution> {
        let executions = self.executions.read().await;
        let limit = limit.unwrap_or(executions.len());
        executions.iter().rev().take(limit).cloned().collect()
    }

    /// Get statistics
    pub async fn get_statistics(&self) -> VoiceControlStats {
        self.stats.read().await.clone()
    }

    /// Update configuration
    pub async fn update_config(&self, config: VoiceControlConfig) {
        let mut current_config = self.config.write().await;
        *current_config = config;
    }

    /// Clear history
    pub async fn clear_history(&self) {
        let mut history = self.history.write().await;
        history.clear();
    }

    // Private helper methods

    fn calculate_similarity(&self, input: &str, trigger: &str) -> f32 {
        // Intent-focused similarity that checks what fraction of trigger keywords
        // appear in the user's input. This allows natural phrasing like
        // "please play the audio" to match the "play" trigger with high confidence.
        let input_words: Vec<&str> = input.split_whitespace().collect();
        let trigger_words: Vec<&str> = trigger.split_whitespace().collect();

        if input_words.is_empty() || trigger_words.is_empty() {
            return 0.0;
        }

        // Recall metric: fraction of trigger words that appear in the input.
        // A trigger like "play" appearing anywhere in the input means the user
        // likely intends that command, regardless of other words present.
        let trigger_matches = trigger_words
            .iter()
            .filter(|&&tw| {
                input_words
                    .iter()
                    .any(|&iw| iw == tw || iw.contains(tw) || tw.contains(iw))
            })
            .count();

        let trigger_recall = trigger_matches as f32 / trigger_words.len() as f32;

        // Precision metric: fraction of input words that match trigger words.
        // Used to reduce false positives for very short trigger phrases.
        let input_matches = input_words
            .iter()
            .filter(|&&iw| {
                trigger_words
                    .iter()
                    .any(|&tw| iw == tw || iw.contains(tw) || tw.contains(iw))
            })
            .count();
        let input_precision = input_matches as f32 / input_words.len() as f32;

        // Combine recall and precision using a weighted geometric mean.
        // Recall is weighted higher (3:1) because we prioritize detecting intent
        // over exact phrase matching.
        let recall_weight = 3.0_f32;
        let precision_weight = 1.0_f32;
        let total_weight = recall_weight + precision_weight;

        (trigger_recall.powf(recall_weight / total_weight))
            * (input_precision.powf(precision_weight / total_weight))
    }

    fn extract_parameters(&self, input: &str, _command: &VoiceCommand) -> HashMap<String, String> {
        // Simplified parameter extraction
        // In a real implementation, this would use NLP to extract entities
        HashMap::new()
    }

    async fn log_execution(&self, execution: CommandExecution) {
        let mut executions = self.executions.write().await;
        executions.push(execution.clone());

        let mut stats = self.stats.write().await;
        if execution.success {
            stats.successful_executions += 1;
        } else {
            stats.failed_executions += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_register_command() {
        let manager = VoiceControlManager::new(VoiceControlConfig::default());

        let command = VoiceCommand {
            command_id: "test_cmd".to_string(),
            name: "Test Command".to_string(),
            trigger_phrases: vec!["test".to_string()],
            category: CommandCategory::Custom {
                name: "Test".to_string(),
            },
            parameters: vec![],
            description: "Test".to_string(),
            enabled: true,
            requires_confirmation: false,
        };

        manager.register_command(command).await.unwrap();

        let retrieved = manager.get_command("test_cmd").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "Test Command");
    }

    #[tokio::test]
    async fn test_recognize_intent_exact_match() {
        let manager = VoiceControlManager::new(VoiceControlConfig::default());

        let intent = manager.recognize_intent("play").await.unwrap();
        assert_eq!(intent.command_id, "play");
        assert!(intent.confidence > 0.7);
    }

    #[tokio::test]
    async fn test_recognize_intent_fuzzy_match() {
        let manager = VoiceControlManager::new(VoiceControlConfig::default());

        let intent = manager
            .recognize_intent("please play the audio")
            .await
            .unwrap();
        assert_eq!(intent.command_id, "play");
    }

    #[tokio::test]
    async fn test_recognize_intent_not_found() {
        let manager = VoiceControlManager::new(VoiceControlConfig::default());

        let result = manager.recognize_intent("invalid command xyz").await;
        assert!(result.is_err());
    }

    /// With no handler registered, `execute_command` must report a real,
    /// honest failure instead of fabricating success -- there is genuinely
    /// nothing wired up to run.
    #[tokio::test]
    async fn test_execute_command_without_handler_is_honest_failure() {
        let manager = VoiceControlManager::new(VoiceControlConfig::default());

        let intent = manager.recognize_intent("play").await.unwrap();
        let execution = manager.execute_command(&intent).await.unwrap();

        assert!(!execution.success);
        assert!(execution.message.contains("No handler registered"));

        // The failure must also be reflected in real statistics, not hidden.
        let stats = manager.get_statistics().await;
        assert_eq!(stats.failed_executions, 1);
        assert_eq!(stats.successful_executions, 0);
    }

    /// A registered handler must be genuinely invoked and its real result
    /// (not a constant) must flow through to the `CommandExecution`.
    #[tokio::test]
    async fn test_execute_command_dispatches_to_real_handler() {
        let manager = VoiceControlManager::new(VoiceControlConfig::default());
        let play_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let counter = play_count.clone();
        manager
            .register_handler("play", move |_intent| {
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok("playback actually started".to_string())
            })
            .await;

        let intent = manager.recognize_intent("play").await.unwrap();
        let execution = manager.execute_command(&intent).await.unwrap();

        assert!(execution.success);
        assert_eq!(execution.message, "playback actually started");
        assert_eq!(
            play_count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the real handler must actually run exactly once"
        );

        // A different, unregistered command must still honestly fail --
        // registering one handler does not fabricate success for others.
        let pause_intent = manager.recognize_intent("pause").await.unwrap();
        let pause_execution = manager.execute_command(&pause_intent).await.unwrap();
        assert!(!pause_execution.success);
    }

    /// A handler that reports a genuine failure must have that failure
    /// propagate through `execute_command`, not get silently upgraded to
    /// success.
    #[tokio::test]
    async fn test_execute_command_propagates_real_handler_failure() {
        let manager = VoiceControlManager::new(VoiceControlConfig::default());

        manager
            .register_handler("play", |_intent| {
                Err(VoiceControlError::ConfigError {
                    message: "audio device unavailable".to_string(),
                })
            })
            .await;

        let intent = manager.recognize_intent("play").await.unwrap();
        let execution = manager.execute_command(&intent).await.unwrap();

        assert!(!execution.success);
        assert!(execution.message.contains("audio device unavailable"));
    }

    #[tokio::test]
    async fn test_command_history() {
        let manager = VoiceControlManager::new(VoiceControlConfig::default());

        manager.recognize_intent("play").await.unwrap();
        manager.recognize_intent("pause").await.unwrap();

        let history = manager.get_history(None).await;
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].command_id, "pause"); // Most recent first
        assert_eq!(history[1].command_id, "play");
    }

    #[tokio::test]
    async fn test_list_commands_by_category() {
        let manager = VoiceControlManager::new(VoiceControlConfig::default());

        let commands = manager
            .list_commands_by_category(CommandCategory::Playback)
            .await;
        assert!(!commands.is_empty());
        assert!(commands
            .iter()
            .all(|cmd| cmd.category == CommandCategory::Playback));
    }

    #[tokio::test]
    async fn test_statistics() {
        let manager = VoiceControlManager::new(VoiceControlConfig::default());

        manager.recognize_intent("play").await.unwrap();
        manager.recognize_intent("pause").await.unwrap();

        let stats = manager.get_statistics().await;
        assert_eq!(stats.total_commands, 2);
        assert!(stats.avg_confidence > 0.0);
    }

    #[tokio::test]
    async fn test_clear_history() {
        let manager = VoiceControlManager::new(VoiceControlConfig::default());

        manager.recognize_intent("play").await.unwrap();
        assert_eq!(manager.get_history(None).await.len(), 1);

        manager.clear_history().await;
        assert_eq!(manager.get_history(None).await.len(), 0);
    }
}
