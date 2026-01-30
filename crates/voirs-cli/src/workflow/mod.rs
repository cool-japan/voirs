//! Workflow Automation System
//!
//! This module provides a sophisticated workflow engine for defining and executing
//! complex multi-step synthesis pipelines with conditional logic, error recovery,
//! and composition capabilities.
//!
//! # Features
//!
//! - **Declarative Workflow Definition**: Define workflows using YAML/JSON/TOML
//! - **Conditional Execution**: Branch based on quality metrics, file size, duration
//! - **Error Recovery**: Automatic retry with exponential backoff, fallback strategies
//! - **Workflow Composition**: Combine workflows, create sub-workflows
//! - **State Management**: Persistent workflow state, resume capability
//! - **Validation**: Pre-execution workflow validation and optimization
//! - **Parallel Execution**: Run independent steps in parallel
//! - **Variable Substitution**: Dynamic parameter substitution
//!
//! # Example Workflow
//!
//! ```yaml
//! name: "Multi-Voice Production Pipeline"
//! version: "1.0"
//! variables:
//!   quality: "high"
//!   voices: ["en-US-neural-1", "en-US-neural-2"]
//!
//! steps:
//!   - name: "synthesis"
//!     type: "synthesize"
//!     parameters:
//!       text: "${input_text}"
//!       voice: "${voice}"
//!       quality: "${quality}"
//!     for_each: "${voices}"
//!     retry:
//!       max_attempts: 3
//!       backoff: "exponential"
//!
//!   - name: "quality_check"
//!     type: "validate"
//!     condition: "${synthesis.success}"
//!     parameters:
//!       min_mos: 4.0
//!       max_duration_diff: 0.1
//!
//!   - name: "fallback"
//!     type: "synthesize"
//!     condition: "${quality_check.failed}"
//!     parameters:
//!       voice: "fallback-voice"
//!       quality: "medium"
//! ```

pub mod definition;
pub mod engine;
pub mod executor;
pub mod retry;
pub mod state;
pub mod validation;
pub mod variables;

pub use definition::{
    Condition, ConditionOperator, RetryStrategy, Step, StepDependency, StepType, Variable,
    Workflow, WorkflowMetadata,
};
pub use engine::WorkflowEngine;
pub use executor::{ExecutionContext, ExecutionResult, StepExecutor, StepResult};
pub use retry::{BackoffStrategy, RetryConfig, RetryManager};
pub use state::{ExecutionState, StateManager, WorkflowState};
pub use validation::{ValidationError, ValidationResult, WorkflowValidator};
pub use variables::{VariableResolver, VariableScope};

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Workflow execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStats {
    /// Total steps executed
    pub total_steps: usize,
    /// Successful steps
    pub successful_steps: usize,
    /// Failed steps
    pub failed_steps: usize,
    /// Skipped steps (due to conditions)
    pub skipped_steps: usize,
    /// Total execution time in milliseconds
    pub total_duration_ms: u64,
    /// Average step duration in milliseconds
    pub avg_step_duration_ms: u64,
    /// Number of retries performed
    pub total_retries: usize,
}

impl WorkflowStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self {
            total_steps: 0,
            successful_steps: 0,
            failed_steps: 0,
            skipped_steps: 0,
            total_duration_ms: 0,
            avg_step_duration_ms: 0,
            total_retries: 0,
        }
    }

    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.successful_steps as f64 / self.total_steps as f64
        }
    }

    /// Check if workflow completed successfully
    pub fn is_successful(&self) -> bool {
        self.failed_steps == 0 && self.total_steps > 0
    }
}

impl Default for WorkflowStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Workflow registry for managing multiple workflows
pub struct WorkflowRegistry {
    workflows: Arc<RwLock<HashMap<String, Workflow>>>,
    storage_dir: PathBuf,
}

impl WorkflowRegistry {
    /// Create new registry
    pub fn new(storage_dir: PathBuf) -> Self {
        Self {
            workflows: Arc::new(RwLock::new(HashMap::new())),
            storage_dir,
        }
    }

    /// Register a workflow
    pub async fn register(&self, workflow: Workflow) -> Result<()> {
        let mut workflows = self.workflows.write().await;
        let name = workflow.metadata.name.clone();
        workflows.insert(name, workflow);
        Ok(())
    }

    /// Get a workflow by name
    pub async fn get(&self, name: &str) -> Option<Workflow> {
        let workflows = self.workflows.read().await;
        workflows.get(name).cloned()
    }

    /// List all registered workflows
    pub async fn list(&self) -> Vec<String> {
        let workflows = self.workflows.read().await;
        workflows.keys().cloned().collect()
    }

    /// Remove a workflow
    pub async fn remove(&self, name: &str) -> Result<()> {
        let mut workflows = self.workflows.write().await;
        workflows.remove(name);
        Ok(())
    }

    /// Load workflows from directory
    pub async fn load_from_directory(&self) -> Result<usize> {
        let mut count = 0;
        if !self.storage_dir.exists() {
            tokio::fs::create_dir_all(&self.storage_dir).await?;
            return Ok(0);
        }

        let mut entries = tokio::fs::read_dir(&self.storage_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path
                .extension()
                .is_some_and(|ext| ext == "yaml" || ext == "json")
            {
                if let Ok(workflow) = Workflow::load_from_file(&path).await {
                    self.register(workflow).await?;
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    /// Save all workflows to directory
    pub async fn save_all(&self) -> Result<usize> {
        tokio::fs::create_dir_all(&self.storage_dir).await?;

        let workflows = self.workflows.read().await;
        let mut count = 0;

        for workflow in workflows.values() {
            let filename = format!("{}.yaml", workflow.metadata.name);
            let path = self.storage_dir.join(filename);
            workflow.save_to_file(&path).await?;
            count += 1;
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_workflow_stats_creation() {
        let stats = WorkflowStats::new();
        assert_eq!(stats.total_steps, 0);
        assert_eq!(stats.successful_steps, 0);
        assert_eq!(stats.failed_steps, 0);
        assert_eq!(stats.success_rate(), 0.0);
        assert!(!stats.is_successful());
    }

    #[test]
    fn test_workflow_stats_success_rate() {
        let stats = WorkflowStats {
            total_steps: 10,
            successful_steps: 8,
            failed_steps: 2,
            skipped_steps: 0,
            total_duration_ms: 1000,
            avg_step_duration_ms: 100,
            total_retries: 3,
        };

        assert_eq!(stats.success_rate(), 0.8);
        assert!(!stats.is_successful()); // Has failures
    }

    #[test]
    fn test_workflow_stats_full_success() {
        let stats = WorkflowStats {
            total_steps: 5,
            successful_steps: 5,
            failed_steps: 0,
            skipped_steps: 0,
            total_duration_ms: 500,
            avg_step_duration_ms: 100,
            total_retries: 0,
        };

        assert_eq!(stats.success_rate(), 1.0);
        assert!(stats.is_successful());
    }

    #[tokio::test]
    async fn test_workflow_registry_creation() {
        let temp_dir = env::temp_dir().join("voirs_workflow_test");
        let registry = WorkflowRegistry::new(temp_dir);

        let workflows = registry.list().await;
        assert_eq!(workflows.len(), 0);
    }

    #[tokio::test]
    async fn test_workflow_registry_register_and_get() {
        let temp_dir = env::temp_dir().join("voirs_workflow_test_2");
        let registry = WorkflowRegistry::new(temp_dir);

        let workflow = Workflow::new("test-workflow", "1.0", "Test workflow");
        registry.register(workflow.clone()).await.unwrap();

        let retrieved = registry.get("test-workflow").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().metadata.name, "test-workflow");
    }

    #[tokio::test]
    async fn test_workflow_registry_list() {
        let temp_dir = env::temp_dir().join("voirs_workflow_test_3");
        let registry = WorkflowRegistry::new(temp_dir);

        let workflow1 = Workflow::new("workflow-1", "1.0", "First workflow");
        let workflow2 = Workflow::new("workflow-2", "1.0", "Second workflow");

        registry.register(workflow1).await.unwrap();
        registry.register(workflow2).await.unwrap();

        let workflows = registry.list().await;
        assert_eq!(workflows.len(), 2);
        assert!(workflows.contains(&"workflow-1".to_string()));
        assert!(workflows.contains(&"workflow-2".to_string()));
    }

    #[tokio::test]
    async fn test_workflow_registry_remove() {
        let temp_dir = env::temp_dir().join("voirs_workflow_test_4");
        let registry = WorkflowRegistry::new(temp_dir);

        let workflow = Workflow::new("removable-workflow", "1.0", "Test removal");
        registry.register(workflow).await.unwrap();

        assert!(registry.get("removable-workflow").await.is_some());

        registry.remove("removable-workflow").await.unwrap();

        assert!(registry.get("removable-workflow").await.is_none());
    }
}
