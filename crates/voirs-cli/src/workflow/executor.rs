//! Step Executor
//!
//! Executes individual workflow steps with retry logic and state management.

use super::{
    definition::{Step, StepType, Workflow},
    retry::RetryManager,
    state::WorkflowState,
    WorkflowStats,
};
use crate::error::CliError;

type Result<T> = std::result::Result<T, CliError>;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Execution context for a workflow
#[derive(Clone)]
pub struct ExecutionContext {
    /// The workflow being executed
    workflow: Workflow,
    /// Current variables
    variables: HashMap<String, serde_json::Value>,
    /// Completed steps and their results
    completed: HashMap<String, StepResult>,
    /// Skipped steps
    skipped: Vec<String>,
    /// Total retries performed
    retries: usize,
}

impl ExecutionContext {
    /// Create new execution context
    pub fn new(workflow: Workflow) -> Self {
        // Initialize variables from workflow definition
        let mut variables = HashMap::new();
        for (key, value) in &workflow.variables {
            let json_value = match value {
                super::definition::Variable::String(s) => serde_json::Value::String(s.clone()),
                super::definition::Variable::Number(n) => serde_json::json!(n),
                super::definition::Variable::Boolean(b) => serde_json::Value::Bool(*b),
                super::definition::Variable::Array(arr) => serde_json::Value::Array(arr.clone()),
                super::definition::Variable::Object(obj) => {
                    serde_json::Value::Object(serde_json::Map::from_iter(obj.clone()))
                }
            };
            variables.insert(key.clone(), json_value);
        }

        Self {
            workflow,
            variables,
            completed: HashMap::new(),
            skipped: Vec::new(),
            retries: 0,
        }
    }

    /// Get workflow reference
    pub fn workflow(&self) -> &Workflow {
        &self.workflow
    }

    /// Get current variables
    pub fn get_variables(&self) -> HashMap<String, serde_json::Value> {
        self.variables.clone()
    }

    /// Set a variable
    pub fn set_variable(&mut self, name: String, value: serde_json::Value) {
        self.variables.insert(name, value);
    }

    /// Record step completion
    pub fn complete_step(&mut self, name: &str, result: StepResult) {
        self.completed.insert(name.to_string(), result);
    }

    /// Record step skip
    pub fn skip_step(&mut self, name: &str, reason: &str) {
        self.skipped.push(name.to_string());
        tracing::info!("Skipping step '{}': {}", name, reason);
    }

    /// Get completed steps
    pub fn completed_steps(&self) -> &HashMap<String, StepResult> {
        &self.completed
    }

    /// Get skipped steps
    pub fn skipped_steps(&self) -> &[String] {
        &self.skipped
    }

    /// Increment retry counter
    pub fn increment_retries(&mut self) {
        self.retries += 1;
    }

    /// Get total retries
    pub fn total_retries(&self) -> usize {
        self.retries
    }

    /// Resume from saved state
    pub fn resume_from_state(&mut self, state: WorkflowState) {
        self.variables = state.variables;
        self.completed = state.completed_steps;
        self.skipped = state.skipped_steps;
        self.retries = state.total_retries;
    }

    /// Get current state
    pub fn get_state(&self) -> WorkflowState {
        WorkflowState {
            workflow_name: self.workflow.metadata.name.clone(),
            state: super::state::ExecutionState::Running,
            variables: self.variables.clone(),
            completed_steps: self.completed.clone(),
            skipped_steps: self.skipped.clone(),
            current_step: None,
            total_retries: self.retries,
            last_updated: chrono::Utc::now(),
        }
    }
}

/// Result of a step execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    /// Step name
    pub step_name: String,
    /// Success status
    pub success: bool,
    /// Result message
    pub message: String,
    /// Output data
    pub output: HashMap<String, serde_json::Value>,
    /// Execution duration in milliseconds
    pub duration_ms: u64,
    /// Number of retry attempts
    pub attempts: usize,
}

impl StepResult {
    /// Create success result
    pub fn success(step_name: String, message: String, duration_ms: u64) -> Self {
        Self {
            step_name,
            success: true,
            message,
            output: HashMap::new(),
            duration_ms,
            attempts: 1,
        }
    }

    /// Create failure result
    pub fn failure(step_name: String, message: String, duration_ms: u64) -> Self {
        Self {
            step_name,
            success: false,
            message,
            output: HashMap::new(),
            duration_ms,
            attempts: 1,
        }
    }

    /// Add output data
    pub fn with_output(mut self, key: String, value: serde_json::Value) -> Self {
        self.output.insert(key, value);
        self
    }

    /// Set attempt count
    pub fn with_attempts(mut self, attempts: usize) -> Self {
        self.attempts = attempts;
        self
    }
}

/// Step executor
pub struct StepExecutor {
    retry_manager: RetryManager,
}

impl StepExecutor {
    /// Create new step executor
    pub fn new() -> Self {
        Self {
            retry_manager: RetryManager::new(),
        }
    }

    /// Execute a step
    pub async fn execute_step(
        &self,
        step: &Step,
        context: &mut ExecutionContext,
    ) -> Result<StepResult> {
        let start_time = Instant::now();

        // Handle for-each loop
        if let Some(ref for_each_var) = step.for_each {
            return self.execute_for_each(step, for_each_var, context).await;
        }

        // Execute with retry if configured
        if let Some(ref retry_strategy) = step.retry {
            let mut attempts = 0;
            loop {
                attempts += 1;
                match self.execute_step_once(step, context).await {
                    Ok(result) => {
                        let duration = start_time.elapsed().as_millis() as u64;
                        context.complete_step(&step.name, result.clone().with_attempts(attempts));
                        return Ok(result.with_attempts(attempts));
                    }
                    Err(e) if attempts < retry_strategy.max_attempts => {
                        context.increment_retries();
                        let delay = self.retry_manager.calculate_delay(retry_strategy, attempts);
                        tokio::time::sleep(tokio::time::Duration::from_millis(delay)).await;
                        tracing::warn!(
                            "Step '{}' failed (attempt {}), retrying: {}",
                            step.name,
                            attempts,
                            e
                        );
                        continue;
                    }
                    Err(e) => {
                        let duration = start_time.elapsed().as_millis() as u64;
                        let result = StepResult::failure(
                            step.name.clone(),
                            format!("Error: {}", e),
                            duration,
                        )
                        .with_attempts(attempts);
                        context.complete_step(&step.name, result.clone());
                        return Ok(result);
                    }
                }
            }
        } else {
            let result = self.execute_step_once(step, context).await;
            let duration = start_time.elapsed().as_millis() as u64;

            match result {
                Ok(mut result) => {
                    result.duration_ms = duration;
                    context.complete_step(&step.name, result.clone());
                    Ok(result)
                }
                Err(e) => {
                    let result =
                        StepResult::failure(step.name.clone(), format!("Error: {}", e), duration);
                    context.complete_step(&step.name, result.clone());
                    Ok(result)
                }
            }
        }
    }

    /// Execute step once (without retry)
    async fn execute_step_once(
        &self,
        step: &Step,
        context: &ExecutionContext,
    ) -> Result<StepResult> {
        let start_time = Instant::now();

        // Resolve parameters with variable substitution
        let resolved_params =
            self.resolve_parameters(&step.parameters, &context.get_variables())?;

        // Execute based on step type
        let result = match step.step_type {
            StepType::Synthesize => self.execute_synthesize(step, &resolved_params).await,
            StepType::Validate => self.execute_validate(step, &resolved_params).await,
            StepType::FileOp => self.execute_file_op(step, &resolved_params).await,
            StepType::Command => self.execute_command(step, &resolved_params).await,
            StepType::Script => self.execute_script(step, &resolved_params).await,
            StepType::Branch => self.execute_branch(step, &resolved_params).await,
            StepType::Loop => self.execute_loop(step, &resolved_params).await,
            StepType::Workflow => self.execute_subworkflow(step, &resolved_params).await,
            StepType::Wait => self.execute_wait(step, &resolved_params).await,
            StepType::Notify => self.execute_notify(step, &resolved_params).await,
        }?;

        let duration = start_time.elapsed().as_millis() as u64;

        Ok(StepResult::success(step.name.clone(), result, duration))
    }

    /// Execute for-each loop
    async fn execute_for_each(
        &self,
        step: &Step,
        for_each_var: &str,
        context: &mut ExecutionContext,
    ) -> Result<StepResult> {
        let variables = context.get_variables();

        // Resolve for-each variable
        let var_name = for_each_var
            .strip_prefix("${")
            .and_then(|s| s.strip_suffix('}'))
            .unwrap_or(for_each_var);

        let items = variables
            .get(var_name)
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                CliError::Workflow(format!(
                    "For-each variable '{}' not found or not an array",
                    var_name
                ))
            })?;

        let start_time = Instant::now();
        let mut all_results = Vec::new();

        for (idx, item) in items.iter().enumerate() {
            // Create new step with current item as variable
            let mut step_clone = step.clone();
            step_clone.for_each = None;
            step_clone.name = format!("{}[{}]", step.name, idx);

            // Set loop variable
            context.set_variable(format!("{}_item", var_name), item.clone());
            context.set_variable(format!("{}_index", var_name), serde_json::json!(idx));

            let result = self.execute_step_once(&step_clone, context).await?;
            all_results.push(result);
        }

        let duration = start_time.elapsed().as_millis() as u64;
        let success = all_results.iter().all(|r| r.success);

        Ok(StepResult {
            step_name: step.name.clone(),
            success,
            message: format!("Executed {} iterations", all_results.len()),
            output: HashMap::new(),
            duration_ms: duration,
            attempts: 1,
        })
    }

    /// Resolve parameters with variable substitution
    fn resolve_parameters(
        &self,
        params: &HashMap<String, serde_json::Value>,
        variables: &HashMap<String, serde_json::Value>,
    ) -> Result<HashMap<String, serde_json::Value>> {
        let mut resolved = HashMap::new();

        for (key, value) in params {
            let resolved_value = Self::resolve_value(value, variables);
            resolved.insert(key.clone(), resolved_value);
        }

        Ok(resolved)
    }

    /// Resolve a single value with variable substitution
    fn resolve_value(
        value: &serde_json::Value,
        variables: &HashMap<String, serde_json::Value>,
    ) -> serde_json::Value {
        match value {
            serde_json::Value::String(s) => {
                if let Some(var_name) = s.strip_prefix("${").and_then(|s| s.strip_suffix('}')) {
                    variables
                        .get(var_name)
                        .cloned()
                        .unwrap_or(serde_json::Value::Null)
                } else {
                    value.clone()
                }
            }
            serde_json::Value::Array(arr) => serde_json::Value::Array(
                arr.iter()
                    .map(|v| Self::resolve_value(v, variables))
                    .collect(),
            ),
            serde_json::Value::Object(obj) => serde_json::Value::Object(
                obj.iter()
                    .map(|(k, v)| (k.clone(), Self::resolve_value(v, variables)))
                    .collect(),
            ),
            _ => value.clone(),
        }
    }

    // Step type implementations (placeholders for actual implementation)

    async fn execute_synthesize(
        &self,
        _step: &Step,
        _params: &HashMap<String, serde_json::Value>,
    ) -> Result<String> {
        Ok("Synthesis completed".to_string())
    }

    async fn execute_validate(
        &self,
        _step: &Step,
        _params: &HashMap<String, serde_json::Value>,
    ) -> Result<String> {
        Ok("Validation passed".to_string())
    }

    async fn execute_file_op(
        &self,
        _step: &Step,
        _params: &HashMap<String, serde_json::Value>,
    ) -> Result<String> {
        Ok("File operation completed".to_string())
    }

    async fn execute_command(
        &self,
        _step: &Step,
        _params: &HashMap<String, serde_json::Value>,
    ) -> Result<String> {
        Ok("Command executed".to_string())
    }

    async fn execute_script(
        &self,
        _step: &Step,
        _params: &HashMap<String, serde_json::Value>,
    ) -> Result<String> {
        Ok("Script executed".to_string())
    }

    async fn execute_branch(
        &self,
        _step: &Step,
        _params: &HashMap<String, serde_json::Value>,
    ) -> Result<String> {
        Ok("Branch evaluated".to_string())
    }

    async fn execute_loop(
        &self,
        _step: &Step,
        _params: &HashMap<String, serde_json::Value>,
    ) -> Result<String> {
        Ok("Loop completed".to_string())
    }

    async fn execute_subworkflow(
        &self,
        _step: &Step,
        _params: &HashMap<String, serde_json::Value>,
    ) -> Result<String> {
        Ok("Sub-workflow completed".to_string())
    }

    async fn execute_wait(
        &self,
        _step: &Step,
        params: &HashMap<String, serde_json::Value>,
    ) -> Result<String> {
        if let Some(duration) = params.get("duration_ms") {
            if let Some(ms) = duration.as_u64() {
                tokio::time::sleep(tokio::time::Duration::from_millis(ms)).await;
            }
        }
        Ok("Wait completed".to_string())
    }

    async fn execute_notify(
        &self,
        _step: &Step,
        _params: &HashMap<String, serde_json::Value>,
    ) -> Result<String> {
        Ok("Notification sent".to_string())
    }
}

impl Default for StepExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Overall workflow execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Workflow name
    pub workflow_name: String,
    /// Success status
    pub success: bool,
    /// Result message
    pub message: String,
    /// Execution statistics
    pub stats: WorkflowStats,
}

impl ExecutionResult {
    /// Create success result
    pub fn success(workflow_name: String, message: String, stats: WorkflowStats) -> Self {
        Self {
            workflow_name,
            success: true,
            message,
            stats,
        }
    }

    /// Create failure result
    pub fn failure(workflow_name: String, message: String, stats: WorkflowStats) -> Self {
        Self {
            workflow_name,
            success: false,
            message,
            stats,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::definition::StepType;

    #[test]
    fn test_execution_context_creation() {
        let workflow = Workflow::new("test", "1.0", "Test workflow");
        let context = ExecutionContext::new(workflow);

        assert_eq!(context.completed_steps().len(), 0);
        assert_eq!(context.skipped_steps().len(), 0);
        assert_eq!(context.total_retries(), 0);
    }

    #[test]
    fn test_execution_context_variables() {
        let mut workflow = Workflow::new("test", "1.0", "Test workflow");
        workflow.add_variable(
            "test_var".to_string(),
            super::super::definition::Variable::String("test_value".to_string()),
        );

        let context = ExecutionContext::new(workflow);
        let variables = context.get_variables();

        assert_eq!(variables.len(), 1);
        assert_eq!(
            variables
                .get("test_var")
                .unwrap()
                .as_str()
                .unwrap_or_default(),
            "test_value"
        );
    }

    #[test]
    fn test_step_result_creation() {
        let result = StepResult::success("step1".to_string(), "Success".to_string(), 100);

        assert!(result.success);
        assert_eq!(result.step_name, "step1");
        assert_eq!(result.duration_ms, 100);
    }

    #[test]
    fn test_step_result_with_output() {
        let result = StepResult::success("step1".to_string(), "Success".to_string(), 100)
            .with_output("key1".to_string(), serde_json::json!("value1"));

        assert_eq!(result.output.len(), 1);
        assert_eq!(
            result
                .output
                .get("key1")
                .unwrap()
                .as_str()
                .unwrap_or_default(),
            "value1"
        );
    }

    #[tokio::test]
    async fn test_step_executor_creation() {
        let _executor = StepExecutor::new();
        // Verify creation works without panic
    }

    #[test]
    fn test_execution_result_success() {
        let stats = WorkflowStats::new();
        let result = ExecutionResult::success("test".to_string(), "Done".to_string(), stats);

        assert!(result.success);
        assert_eq!(result.workflow_name, "test");
    }

    #[test]
    fn test_execution_result_failure() {
        let stats = WorkflowStats::new();
        let result = ExecutionResult::failure("test".to_string(), "Failed".to_string(), stats);

        assert!(!result.success);
        assert_eq!(result.message, "Failed");
    }
}
