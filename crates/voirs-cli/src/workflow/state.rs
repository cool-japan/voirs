//! Workflow State Management
//!
//! Manages persistent state for workflow execution, enabling resume capability.

use super::executor::StepResult;
use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Execution state enum
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecutionState {
    /// Not started
    Pending,
    /// Currently running
    Running,
    /// Completed successfully
    Completed,
    /// Failed
    Failed,
    /// Stopped by user
    Stopped,
    /// Paused (for future use)
    Paused,
}

/// Workflow state for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowState {
    /// Workflow name
    pub workflow_name: String,
    /// Current execution state
    pub state: ExecutionState,
    /// Current variables
    pub variables: HashMap<String, serde_json::Value>,
    /// Completed steps
    pub completed_steps: HashMap<String, StepResult>,
    /// Skipped steps
    pub skipped_steps: Vec<String>,
    /// Current step being executed
    pub current_step: Option<String>,
    /// Total retries performed
    pub total_retries: usize,
    /// Last update timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl WorkflowState {
    /// Create new workflow state
    pub fn new(workflow_name: String) -> Self {
        Self {
            workflow_name,
            state: ExecutionState::Pending,
            variables: HashMap::new(),
            completed_steps: HashMap::new(),
            skipped_steps: Vec::new(),
            current_step: None,
            total_retries: 0,
            last_updated: chrono::Utc::now(),
        }
    }

    /// Mark as running
    pub fn mark_running(&mut self) {
        self.state = ExecutionState::Running;
        self.last_updated = chrono::Utc::now();
    }

    /// Mark as completed
    pub fn mark_completed(&mut self) {
        self.state = ExecutionState::Completed;
        self.last_updated = chrono::Utc::now();
    }

    /// Mark as failed
    pub fn mark_failed(&mut self) {
        self.state = ExecutionState::Failed;
        self.last_updated = chrono::Utc::now();
    }

    /// Update current step
    pub fn set_current_step(&mut self, step_name: Option<String>) {
        self.current_step = step_name;
        self.last_updated = chrono::Utc::now();
    }

    /// Check if can be resumed
    pub fn can_resume(&self) -> bool {
        matches!(
            self.state,
            ExecutionState::Failed | ExecutionState::Stopped | ExecutionState::Paused
        )
    }
}

/// State manager for persistence
pub struct StateManager {
    storage_dir: PathBuf,
}

impl StateManager {
    /// Create new state manager
    pub fn new(storage_dir: PathBuf) -> Self {
        Self { storage_dir }
    }

    /// Save workflow state
    pub async fn save(&self, workflow_name: &str, state: &WorkflowState) -> Result<()> {
        tokio::fs::create_dir_all(&self.storage_dir).await?;

        let filename = format!("{}.state.json", workflow_name);
        let path = self.storage_dir.join(filename);

        let json = serde_json::to_string_pretty(state)?;
        tokio::fs::write(path, json).await?;

        Ok(())
    }

    /// Load workflow state
    pub async fn load(&self, workflow_name: &str) -> Result<WorkflowState> {
        let filename = format!("{}.state.json", workflow_name);
        let path = self.storage_dir.join(filename);

        let json = tokio::fs::read_to_string(path).await?;
        let state = serde_json::from_str(&json)?;

        Ok(state)
    }

    /// Delete workflow state
    pub async fn delete(&self, workflow_name: &str) -> Result<()> {
        let filename = format!("{}.state.json", workflow_name);
        let path = self.storage_dir.join(filename);

        if path.exists() {
            tokio::fs::remove_file(path).await?;
        }

        Ok(())
    }

    /// List all saved states
    pub async fn list_states(&self) -> Result<Vec<String>> {
        let mut states = Vec::new();

        if !self.storage_dir.exists() {
            return Ok(states);
        }

        let mut entries = tokio::fs::read_dir(&self.storage_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if let Some(filename) = path.file_name() {
                if let Some(name_str) = filename.to_str() {
                    if name_str.ends_with(".state.json") {
                        let workflow_name = name_str.strip_suffix(".state.json").unwrap();
                        states.push(workflow_name.to_string());
                    }
                }
            }
        }

        Ok(states)
    }

    /// Check if state exists
    pub async fn exists(&self, workflow_name: &str) -> bool {
        let filename = format!("{}.state.json", workflow_name);
        let path = self.storage_dir.join(filename);
        path.exists()
    }

    /// Get state info without loading full state
    pub async fn get_info(&self, workflow_name: &str) -> Result<StateInfo> {
        let state = self.load(workflow_name).await?;

        Ok(StateInfo {
            workflow_name: state.workflow_name,
            state: state.state,
            completed_steps: state.completed_steps.len(),
            total_steps: state.completed_steps.len() + state.skipped_steps.len(),
            current_step: state.current_step,
            last_updated: state.last_updated,
        })
    }
}

/// State information summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateInfo {
    /// Workflow name
    pub workflow_name: String,
    /// Execution state
    pub state: ExecutionState,
    /// Number of completed steps
    pub completed_steps: usize,
    /// Total steps
    pub total_steps: usize,
    /// Current step
    pub current_step: Option<String>,
    /// Last update time
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_workflow_state_creation() {
        let state = WorkflowState::new("test-workflow".to_string());
        assert_eq!(state.workflow_name, "test-workflow");
        assert_eq!(state.state, ExecutionState::Pending);
        assert_eq!(state.completed_steps.len(), 0);
    }

    #[test]
    fn test_workflow_state_transitions() {
        let mut state = WorkflowState::new("test".to_string());

        assert_eq!(state.state, ExecutionState::Pending);

        state.mark_running();
        assert_eq!(state.state, ExecutionState::Running);

        state.mark_completed();
        assert_eq!(state.state, ExecutionState::Completed);
    }

    #[test]
    fn test_workflow_state_can_resume() {
        let mut state = WorkflowState::new("test".to_string());

        state.mark_failed();
        assert!(state.can_resume());

        state.state = ExecutionState::Stopped;
        assert!(state.can_resume());

        state.state = ExecutionState::Paused;
        assert!(state.can_resume());

        state.mark_completed();
        assert!(!state.can_resume());
    }

    #[tokio::test]
    async fn test_state_manager_creation() {
        let temp_dir = env::temp_dir().join("voirs_state_test");
        let manager = StateManager::new(temp_dir);
        // Just verify creation works
        assert!(true);
    }

    #[tokio::test]
    async fn test_state_manager_save_and_load() {
        let temp_dir = env::temp_dir().join("voirs_state_test_2");
        let manager = StateManager::new(temp_dir);

        let mut state = WorkflowState::new("test-workflow".to_string());
        state.mark_running();
        state
            .variables
            .insert("key1".to_string(), serde_json::json!("value1"));

        manager.save("test-workflow", &state).await.unwrap();

        let loaded_state = manager.load("test-workflow").await.unwrap();
        assert_eq!(loaded_state.workflow_name, "test-workflow");
        assert_eq!(loaded_state.state, ExecutionState::Running);
        assert_eq!(loaded_state.variables.len(), 1);
    }

    #[tokio::test]
    async fn test_state_manager_exists() {
        let temp_dir = env::temp_dir().join("voirs_state_test_3");
        let manager = StateManager::new(temp_dir);

        assert!(!manager.exists("nonexistent").await);

        let state = WorkflowState::new("exists-test".to_string());
        manager.save("exists-test", &state).await.unwrap();

        assert!(manager.exists("exists-test").await);
    }

    #[tokio::test]
    async fn test_state_manager_delete() {
        let temp_dir = env::temp_dir().join("voirs_state_test_4");
        let manager = StateManager::new(temp_dir);

        let state = WorkflowState::new("delete-test".to_string());
        manager.save("delete-test", &state).await.unwrap();

        assert!(manager.exists("delete-test").await);

        manager.delete("delete-test").await.unwrap();

        assert!(!manager.exists("delete-test").await);
    }

    #[tokio::test]
    async fn test_state_manager_list_states() {
        let temp_dir = env::temp_dir().join("voirs_state_test_5");
        let manager = StateManager::new(temp_dir);

        let state1 = WorkflowState::new("workflow1".to_string());
        let state2 = WorkflowState::new("workflow2".to_string());

        manager.save("workflow1", &state1).await.unwrap();
        manager.save("workflow2", &state2).await.unwrap();

        let states = manager.list_states().await.unwrap();
        assert!(states.len() >= 2);
        assert!(
            states.contains(&"workflow1".to_string()) || states.contains(&"workflow2".to_string())
        );
    }
}
