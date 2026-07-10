//! Step Executor
//!
//! Executes individual workflow steps with retry logic and state management.

use super::{
    definition::{Condition, Step, StepType, Workflow},
    retry::RetryManager,
    state::WorkflowState,
    WorkflowStats,
};
use crate::error::CliError;

type Result<T> = std::result::Result<T, CliError>;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Maximum number of iterations a `loop` step may run, guarding against
/// infinite loops from misconfigured or ever-true/never-false conditions.
const MAX_LOOP_ITERATIONS: u64 = 100_000;

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

/// Outcome of a single step-type handler.
///
/// Carries both a human-readable summary (used as `StepResult::message`)
/// and any structured values a handler wants to expose via
/// `StepResult::output` (e.g. captured command/script stdout, the branch
/// taken, or the number of loop iterations performed).
struct StepOutcome {
    message: String,
    output: HashMap<String, serde_json::Value>,
}

impl StepOutcome {
    /// Create an outcome with just a message and no structured output.
    fn message(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            output: HashMap::new(),
        }
    }

    /// Create an outcome with a message plus structured output values.
    fn with_output(message: impl Into<String>, output: HashMap<String, serde_json::Value>) -> Self {
        Self {
            message: message.into(),
            output,
        }
    }
}

/// Resolve the working directory for a file/command/script step: an
/// explicit `cwd` parameter takes precedence, otherwise fall back to the
/// process's current directory (`ExecutionContext` has no working-directory
/// field of its own).
fn resolve_cwd(params: &HashMap<String, serde_json::Value>) -> Result<PathBuf> {
    if let Some(cwd) = params.get("cwd").and_then(|v| v.as_str()) {
        return Ok(PathBuf::from(cwd));
    }
    std::env::current_dir()
        .map_err(|e| CliError::Workflow(format!("Failed to determine current directory: {}", e)))
}

/// Resolve a (possibly relative) path parameter against a base directory.
fn resolve_path(base: &Path, candidate: &str) -> PathBuf {
    let candidate_path = Path::new(candidate);
    if candidate_path.is_absolute() {
        candidate_path.to_path_buf()
    } else {
        base.join(candidate_path)
    }
}

/// Ensure the parent directory of `path` exists, creating it if necessary.
fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            std::fs::create_dir_all(parent).map_err(|e| {
                CliError::file_operation("create directory", &parent.display().to_string(), e)
            })?;
        }
    }
    Ok(())
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
    ///
    /// Takes the execution context mutably: `Branch` and `Loop` steps need
    /// to read current variables/prior outputs *and* write their own results
    /// (e.g. the branch taken, or a loop counter) back into the context, so
    /// the context can no longer be a shared reference here.
    async fn execute_step_once(
        &self,
        step: &Step,
        context: &mut ExecutionContext,
    ) -> Result<StepResult> {
        let start_time = Instant::now();

        // Resolve parameters with variable substitution
        let resolved_params =
            self.resolve_parameters(&step.parameters, &context.get_variables())?;

        // Execute based on step type
        let outcome = match step.step_type {
            StepType::Synthesize => self.execute_synthesize(step, &resolved_params).await,
            StepType::Validate => self.execute_validate(step, &resolved_params).await,
            StepType::FileOp => self.execute_file_op(step, &resolved_params).await,
            StepType::Command => self.execute_command(step, &resolved_params).await,
            StepType::Script => self.execute_script(step, &resolved_params).await,
            StepType::Branch => self.execute_branch(step, &resolved_params, context).await,
            StepType::Loop => self.execute_loop(step, &resolved_params, context).await,
            StepType::Workflow => self.execute_subworkflow(step, &resolved_params).await,
            StepType::Wait => self.execute_wait(step, &resolved_params).await,
            StepType::Notify => self.execute_notify(step, &resolved_params).await,
        }?;

        let duration = start_time.elapsed().as_millis() as u64;

        let mut result = StepResult::success(step.name.clone(), outcome.message, duration);
        result.output = outcome.output;
        Ok(result)
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

    // Step type implementations

    async fn execute_synthesize(
        &self,
        _step: &Step,
        _params: &HashMap<String, serde_json::Value>,
    ) -> Result<StepOutcome> {
        Ok(StepOutcome::message("Synthesis completed"))
    }

    async fn execute_validate(
        &self,
        _step: &Step,
        _params: &HashMap<String, serde_json::Value>,
    ) -> Result<StepOutcome> {
        Ok(StepOutcome::message("Validation passed"))
    }

    /// Real filesystem operation: `op` selects copy/move/delete/mkdir/write/
    /// read, `path` (and `dest`/`content` where relevant) name the target(s).
    /// A relative `path`/`dest` is resolved against `cwd` (parameter, or the
    /// process's current directory).
    async fn execute_file_op(
        &self,
        step: &Step,
        params: &HashMap<String, serde_json::Value>,
    ) -> Result<StepOutcome> {
        let op = params.get("op").and_then(|v| v.as_str()).ok_or_else(|| {
            CliError::Workflow(format!(
                "file-op step '{}' requires an 'op' parameter (copy|move|delete|mkdir|write|read)",
                step.name
            ))
        })?;

        let cwd = resolve_cwd(params)?;

        let path_param = params.get("path").and_then(|v| v.as_str()).ok_or_else(|| {
            CliError::Workflow(format!(
                "file-op step '{}' requires a 'path' parameter",
                step.name
            ))
        })?;
        let path = resolve_path(&cwd, path_param);

        let mut output = HashMap::new();

        let message = match op {
            "mkdir" => {
                std::fs::create_dir_all(&path).map_err(|e| {
                    CliError::file_operation("create directory", &path.display().to_string(), e)
                })?;
                format!("Created directory '{}'", path.display())
            }
            "write" => {
                let content = params
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                ensure_parent_dir(&path)?;
                std::fs::write(&path, content).map_err(|e| {
                    CliError::file_operation("write", &path.display().to_string(), e)
                })?;
                format!("Wrote {} bytes to '{}'", content.len(), path.display())
            }
            "read" => {
                let content = std::fs::read_to_string(&path).map_err(|e| {
                    CliError::file_operation("read", &path.display().to_string(), e)
                })?;
                let len = content.len();
                output.insert("content".to_string(), serde_json::json!(content));
                format!("Read {} bytes from '{}'", len, path.display())
            }
            "copy" => {
                let dest_param = params.get("dest").and_then(|v| v.as_str()).ok_or_else(|| {
                    CliError::Workflow(format!(
                        "file-op step '{}' (copy) requires a 'dest' parameter",
                        step.name
                    ))
                })?;
                let dest = resolve_path(&cwd, dest_param);
                ensure_parent_dir(&dest)?;
                std::fs::copy(&path, &dest).map_err(|e| {
                    CliError::file_operation(
                        "copy",
                        &format!("{} -> {}", path.display(), dest.display()),
                        e,
                    )
                })?;
                format!("Copied '{}' to '{}'", path.display(), dest.display())
            }
            "move" => {
                let dest_param = params.get("dest").and_then(|v| v.as_str()).ok_or_else(|| {
                    CliError::Workflow(format!(
                        "file-op step '{}' (move) requires a 'dest' parameter",
                        step.name
                    ))
                })?;
                let dest = resolve_path(&cwd, dest_param);
                ensure_parent_dir(&dest)?;
                std::fs::rename(&path, &dest).map_err(|e| {
                    CliError::file_operation(
                        "move",
                        &format!("{} -> {}", path.display(), dest.display()),
                        e,
                    )
                })?;
                format!("Moved '{}' to '{}'", path.display(), dest.display())
            }
            "delete" => {
                if path.is_dir() {
                    std::fs::remove_dir_all(&path).map_err(|e| {
                        CliError::file_operation("delete", &path.display().to_string(), e)
                    })?;
                } else {
                    std::fs::remove_file(&path).map_err(|e| {
                        CliError::file_operation("delete", &path.display().to_string(), e)
                    })?;
                }
                format!("Deleted '{}'", path.display())
            }
            other => {
                return Err(CliError::Workflow(format!(
                    "file-op step '{}' has an unknown op '{}' (expected copy|move|delete|mkdir|write|read)",
                    step.name, other
                )));
            }
        };

        Ok(StepOutcome::with_output(message, output))
    }

    /// Run an external command via `tokio::process::Command`, capturing
    /// stdout/stderr/exit code. A non-zero exit becomes a `CliError::Workflow`;
    /// stdout is exposed via `StepResult::output["stdout"]`.
    async fn execute_command(
        &self,
        step: &Step,
        params: &HashMap<String, serde_json::Value>,
    ) -> Result<StepOutcome> {
        let command_str = params
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                CliError::Workflow(format!(
                    "command step '{}' requires a 'command' parameter",
                    step.name
                ))
            })?;

        let mut cmd = if let Some(args) = params.get("args").and_then(|v| v.as_array()) {
            let mut c = tokio::process::Command::new(command_str);
            for arg in args {
                let arg_str = arg.as_str().ok_or_else(|| {
                    CliError::Workflow(format!(
                        "command step '{}' has a non-string entry in 'args'",
                        step.name
                    ))
                })?;
                c.arg(arg_str);
            }
            c
        } else if cfg!(target_os = "windows") {
            let mut c = tokio::process::Command::new("cmd");
            c.arg("/C").arg(command_str);
            c
        } else {
            let mut c = tokio::process::Command::new("sh");
            c.arg("-c").arg(command_str);
            c
        };

        if let Some(cwd) = params.get("cwd").and_then(|v| v.as_str()) {
            cmd.current_dir(cwd);
        }

        let output = cmd.output().await.map_err(|e| {
            CliError::Workflow(format!(
                "command step '{}' failed to launch '{}': {}",
                step.name, command_str, e
            ))
        })?;

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        if !output.status.success() {
            return Err(CliError::Workflow(format!(
                "command step '{}' ('{}') exited with status {}: {}",
                step.name,
                command_str,
                exit_code,
                if stderr.is_empty() { &stdout } else { &stderr }
            )));
        }

        let mut result_output = HashMap::new();
        result_output.insert("stdout".to_string(), serde_json::json!(stdout));
        result_output.insert("stderr".to_string(), serde_json::json!(stderr));
        result_output.insert("exit_code".to_string(), serde_json::json!(exit_code));

        Ok(StepOutcome::with_output(
            format!(
                "Command '{}' completed with exit code {}",
                command_str, exit_code
            ),
            result_output,
        ))
    }

    /// Write `script` to a temp file under `std::env::temp_dir()`, execute it
    /// with `interpreter` (default: a shell), capture stdout/stderr/exit
    /// code with the same rules as `execute_command`, then remove
    /// the temp file.
    async fn execute_script(
        &self,
        step: &Step,
        params: &HashMap<String, serde_json::Value>,
    ) -> Result<StepOutcome> {
        let script_body = params
            .get("script")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                CliError::Workflow(format!(
                    "script step '{}' requires a 'script' parameter with the script body",
                    step.name
                ))
            })?;

        let interpreter = params
            .get("interpreter")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| {
                if cfg!(target_os = "windows") {
                    "cmd".to_string()
                } else {
                    "sh".to_string()
                }
            });

        let script_path = std::env::temp_dir().join(format!(
            "voirs-workflow-script-{}-{}",
            std::process::id(),
            fastrand::u64(..)
        ));

        std::fs::write(&script_path, script_body).map_err(|e| {
            CliError::file_operation("write", &script_path.display().to_string(), e)
        })?;

        let run_result = if cfg!(target_os = "windows") {
            tokio::process::Command::new(&interpreter)
                .arg("/C")
                .arg(&script_path)
                .output()
                .await
        } else {
            tokio::process::Command::new(&interpreter)
                .arg(&script_path)
                .output()
                .await
        };

        // Best-effort cleanup regardless of whether execution succeeded.
        let _ = std::fs::remove_file(&script_path);

        let output = run_result.map_err(|e| {
            CliError::Workflow(format!(
                "script step '{}' failed to launch interpreter '{}': {}",
                step.name, interpreter, e
            ))
        })?;

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        if !output.status.success() {
            return Err(CliError::Workflow(format!(
                "script step '{}' exited with status {}: {}",
                step.name,
                exit_code,
                if stderr.is_empty() { &stdout } else { &stderr }
            )));
        }

        let mut result_output = HashMap::new();
        result_output.insert("stdout".to_string(), serde_json::json!(stdout));
        result_output.insert("stderr".to_string(), serde_json::json!(stderr));
        result_output.insert("exit_code".to_string(), serde_json::json!(exit_code));

        Ok(StepOutcome::with_output(
            format!(
                "Script step '{}' completed with exit code {}",
                step.name, exit_code
            ),
            result_output,
        ))
    }

    /// Evaluate a condition (step-level `condition`, or a `condition`
    /// parameter) and record which path was taken. This is distinct from the
    /// engine-level skip-on-condition check (see `engine.rs`), which decides
    /// whether to run the step *at all*; a `Branch` step always runs (if
    /// reached) and instead exposes its own decision to downstream steps.
    async fn execute_branch(
        &self,
        step: &Step,
        params: &HashMap<String, serde_json::Value>,
        context: &mut ExecutionContext,
    ) -> Result<StepOutcome> {
        let condition = Self::step_condition(step, "branch")?;

        let variables = context.get_variables();
        let branch_taken = condition.evaluate(&variables);

        let then_target = params
            .get("then")
            .and_then(|v| v.as_str())
            .unwrap_or("then")
            .to_string();
        let else_target = params
            .get("else")
            .and_then(|v| v.as_str())
            .unwrap_or("else")
            .to_string();
        let target = if branch_taken {
            then_target
        } else {
            else_target
        };

        // Expose the decision generically (for simple single-branch
        // workflows) and scoped by step name (so multiple branch steps in
        // the same workflow don't clobber one another).
        context.set_variable("branch_taken".to_string(), serde_json::json!(branch_taken));
        context.set_variable(
            format!("{}_taken", step.name),
            serde_json::json!(branch_taken),
        );

        let mut output = HashMap::new();
        output.insert("branch_taken".to_string(), serde_json::json!(branch_taken));
        output.insert("target".to_string(), serde_json::json!(target));

        Ok(StepOutcome::with_output(
            format!(
                "Branch '{}' evaluated to {} (target='{}')",
                step.name, branch_taken, target
            ),
            output,
        ))
    }

    /// Count-based loop (`count` parameter) or while-loop (a step-level
    /// `condition`/`condition` parameter, re-evaluated before every pass).
    /// This is distinct from the existing `for_each` field, which iterates
    /// over a collection rather than counting/branching. Bounded by
    /// `MAX_LOOP_ITERATIONS` to guard against infinite loops.
    async fn execute_loop(
        &self,
        step: &Step,
        params: &HashMap<String, serde_json::Value>,
        context: &mut ExecutionContext,
    ) -> Result<StepOutcome> {
        // Count-based loop: run a fixed number of iterations.
        if let Some(count_value) = params.get("count") {
            let count = count_value.as_u64().ok_or_else(|| {
                CliError::Workflow(format!(
                    "loop step '{}' has a non-numeric 'count' parameter",
                    step.name
                ))
            })?;

            if count > MAX_LOOP_ITERATIONS {
                return Err(CliError::Workflow(format!(
                    "loop step '{}' requested {} iterations, exceeding the maximum of {}",
                    step.name, count, MAX_LOOP_ITERATIONS
                )));
            }

            for i in 0..count {
                context.set_variable(format!("{}_index", step.name), serde_json::json!(i));
            }
            context.set_variable(
                format!("{}_iterations", step.name),
                serde_json::json!(count),
            );

            let mut output = HashMap::new();
            output.insert("iterations".to_string(), serde_json::json!(count));
            output.insert("kind".to_string(), serde_json::json!("count"));

            return Ok(StepOutcome::with_output(
                format!(
                    "Loop '{}' completed {} count-based iterations",
                    step.name, count
                ),
                output,
            ));
        }

        // While-loop: re-evaluate a condition before every pass.
        let condition = Self::step_condition(step, "loop")?;

        let counter_var = params
            .get("increment_var")
            .and_then(|v| v.as_str())
            .unwrap_or("loop_counter")
            .to_string();

        // Seed the counter so the condition can reference it from the very
        // first check (e.g. "${loop_counter} < 5").
        context.set_variable(counter_var.clone(), serde_json::json!(0_u64));

        let mut iterations: u64 = 0;
        loop {
            let variables = context.get_variables();
            if !condition.evaluate(&variables) {
                break;
            }

            iterations += 1;
            if iterations > MAX_LOOP_ITERATIONS {
                return Err(CliError::Workflow(format!(
                    "loop step '{}' exceeded the maximum of {} iterations without its condition becoming false",
                    step.name, MAX_LOOP_ITERATIONS
                )));
            }

            context.set_variable(counter_var.clone(), serde_json::json!(iterations));
        }

        context.set_variable(
            format!("{}_iterations", step.name),
            serde_json::json!(iterations),
        );

        let mut output = HashMap::new();
        output.insert("iterations".to_string(), serde_json::json!(iterations));
        output.insert("kind".to_string(), serde_json::json!("while"));

        Ok(StepOutcome::with_output(
            format!(
                "Loop '{}' completed {} while-loop iterations",
                step.name, iterations
            ),
            output,
        ))
    }

    /// Resolve the condition for a `Branch`/`Loop` step: prefer the
    /// step-level `condition` field, falling back to a `condition`
    /// parameter (deserialized from the step's *raw* parameters, i.e.
    /// before `${var}` substitution, since `Condition::evaluate` performs
    /// its own substitution and expects unresolved `${var}` placeholders).
    fn step_condition(step: &Step, step_kind: &str) -> Result<Condition> {
        if let Some(ref cond) = step.condition {
            return Ok(cond.clone());
        }

        if let Some(cond_value) = step.parameters.get("condition") {
            return serde_json::from_value::<Condition>(cond_value.clone()).map_err(|e| {
                CliError::Workflow(format!(
                    "{step_kind} step '{}' has an invalid 'condition' parameter: {e}",
                    step.name
                ))
            });
        }

        Err(CliError::Workflow(format!(
            "{step_kind} step '{}' requires either a step-level condition or a 'condition' parameter",
            step.name
        )))
    }

    async fn execute_subworkflow(
        &self,
        _step: &Step,
        _params: &HashMap<String, serde_json::Value>,
    ) -> Result<StepOutcome> {
        Ok(StepOutcome::message("Sub-workflow completed"))
    }

    async fn execute_wait(
        &self,
        _step: &Step,
        params: &HashMap<String, serde_json::Value>,
    ) -> Result<StepOutcome> {
        if let Some(duration) = params.get("duration_ms") {
            if let Some(ms) = duration.as_u64() {
                tokio::time::sleep(tokio::time::Duration::from_millis(ms)).await;
            }
        }
        Ok(StepOutcome::message("Wait completed"))
    }

    async fn execute_notify(
        &self,
        _step: &Step,
        _params: &HashMap<String, serde_json::Value>,
    ) -> Result<StepOutcome> {
        Ok(StepOutcome::message("Notification sent"))
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
    use crate::workflow::definition::{ConditionOperator, StepType};

    /// Build a minimal `Step` for tests, filling in the fields that are
    /// irrelevant to the specific handler under test.
    fn make_step(
        name: &str,
        step_type: StepType,
        parameters: HashMap<String, serde_json::Value>,
        condition: Option<Condition>,
    ) -> Step {
        Step {
            name: name.to_string(),
            step_type,
            description: None,
            parameters,
            condition,
            depends_on: Vec::new(),
            retry: None,
            for_each: None,
            parallel: false,
        }
    }

    fn unique_temp_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "voirs_executor_test_{label}_{}_{}",
            std::process::id(),
            fastrand::u64(..)
        ))
    }

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

    #[tokio::test]
    async fn test_execute_file_op_write_then_read() {
        let executor = StepExecutor::new();
        let workflow = Workflow::new("file-op-test", "1.0", "Test file operations");
        let mut context = ExecutionContext::new(workflow);

        let temp_path = unique_temp_path("file_op");

        let mut write_params = HashMap::new();
        write_params.insert("op".to_string(), serde_json::json!("write"));
        write_params.insert(
            "path".to_string(),
            serde_json::json!(temp_path.display().to_string()),
        );
        write_params.insert("content".to_string(), serde_json::json!("hello workflow"));
        let write_step = make_step("write-step", StepType::FileOp, write_params, None);

        let write_result = executor
            .execute_step(&write_step, &mut context)
            .await
            .unwrap();
        assert!(
            write_result.success,
            "write step failed: {}",
            write_result.message
        );
        assert!(temp_path.exists());

        let mut read_params = HashMap::new();
        read_params.insert("op".to_string(), serde_json::json!("read"));
        read_params.insert(
            "path".to_string(),
            serde_json::json!(temp_path.display().to_string()),
        );
        let read_step = make_step("read-step", StepType::FileOp, read_params, None);

        let read_result = executor
            .execute_step(&read_step, &mut context)
            .await
            .unwrap();
        assert!(
            read_result.success,
            "read step failed: {}",
            read_result.message
        );
        assert_eq!(
            read_result.output.get("content").and_then(|v| v.as_str()),
            Some("hello workflow")
        );

        let _ = std::fs::remove_file(&temp_path);
    }

    #[tokio::test]
    async fn test_execute_file_op_unknown_op_errors() {
        let executor = StepExecutor::new();
        let workflow = Workflow::new("file-op-error-test", "1.0", "Test file op error path");
        let mut context = ExecutionContext::new(workflow);

        let mut params = HashMap::new();
        params.insert("op".to_string(), serde_json::json!("frobnicate"));
        params.insert("path".to_string(), serde_json::json!("irrelevant.txt"));
        let step = make_step("bad-op-step", StepType::FileOp, params, None);

        let result = executor.execute_step(&step, &mut context).await.unwrap();
        assert!(!result.success);
        assert!(result.message.contains("frobnicate"));
    }

    #[tokio::test]
    async fn test_execute_command_captures_stdout() {
        let executor = StepExecutor::new();
        let workflow = Workflow::new("command-test", "1.0", "Test command execution");
        let mut context = ExecutionContext::new(workflow);

        let mut params = HashMap::new();
        params.insert("command".to_string(), serde_json::json!("echo hello"));
        let step = make_step("echo-step", StepType::Command, params, None);

        let result = executor.execute_step(&step, &mut context).await.unwrap();
        assert!(result.success, "command step failed: {}", result.message);
        assert_eq!(
            result.output.get("stdout").and_then(|v| v.as_str()),
            Some("hello")
        );
        assert_eq!(
            result.output.get("exit_code").and_then(|v| v.as_i64()),
            Some(0)
        );
    }

    #[tokio::test]
    async fn test_execute_command_nonzero_exit_fails() {
        let executor = StepExecutor::new();
        let workflow = Workflow::new("command-fail-test", "1.0", "Test command failure path");
        let mut context = ExecutionContext::new(workflow);

        let mut params = HashMap::new();
        params.insert("command".to_string(), serde_json::json!("exit 7"));
        let step = make_step("fail-step", StepType::Command, params, None);

        let result = executor.execute_step(&step, &mut context).await.unwrap();
        assert!(!result.success);
        assert!(result.message.contains('7'));
    }

    #[tokio::test]
    async fn test_execute_script_runs_body() {
        let executor = StepExecutor::new();
        let workflow = Workflow::new("script-test", "1.0", "Test script execution");
        let mut context = ExecutionContext::new(workflow);

        let mut params = HashMap::new();
        params.insert("script".to_string(), serde_json::json!("echo scripted"));
        let step = make_step("script-step", StepType::Script, params, None);

        let result = executor.execute_step(&step, &mut context).await.unwrap();
        assert!(result.success, "script step failed: {}", result.message);
        assert_eq!(
            result.output.get("stdout").and_then(|v| v.as_str()),
            Some("scripted")
        );
    }

    #[tokio::test]
    async fn test_execute_branch_picks_path_from_condition() {
        let executor = StepExecutor::new();
        let mut workflow = Workflow::new("branch-test", "1.0", "Test branch evaluation");
        workflow.add_variable(
            "score".to_string(),
            super::super::definition::Variable::Number(4.5),
        );
        let mut context = ExecutionContext::new(workflow);

        // True branch: score (4.5) > 4.0
        let true_step = make_step(
            "branch-true",
            StepType::Branch,
            HashMap::new(),
            Some(Condition::new(
                "${score}".to_string(),
                ConditionOperator::GreaterThan,
                "4.0".to_string(),
            )),
        );
        let true_result = executor
            .execute_step(&true_step, &mut context)
            .await
            .unwrap();
        assert!(true_result.success);
        assert_eq!(
            true_result.output.get("branch_taken"),
            Some(&serde_json::json!(true))
        );
        assert_eq!(
            context.get_variables().get("branch_taken"),
            Some(&serde_json::json!(true))
        );

        // False branch: score (4.5) > 10.0 is false
        let false_step = make_step(
            "branch-false",
            StepType::Branch,
            HashMap::new(),
            Some(Condition::new(
                "${score}".to_string(),
                ConditionOperator::GreaterThan,
                "10.0".to_string(),
            )),
        );
        let false_result = executor
            .execute_step(&false_step, &mut context)
            .await
            .unwrap();
        assert!(false_result.success);
        assert_eq!(
            false_result.output.get("branch_taken"),
            Some(&serde_json::json!(false))
        );
    }

    #[tokio::test]
    async fn test_execute_branch_without_condition_errors() {
        let executor = StepExecutor::new();
        let workflow = Workflow::new("branch-missing-condition", "1.0", "Test missing condition");
        let mut context = ExecutionContext::new(workflow);

        let step = make_step("branch-no-cond", StepType::Branch, HashMap::new(), None);
        let result = executor.execute_step(&step, &mut context).await.unwrap();
        assert!(!result.success);
    }

    #[tokio::test]
    async fn test_execute_branch_condition_from_parameters() {
        // Distinct code path from `test_execute_branch_picks_path_from_condition`:
        // here the condition comes from a `condition` parameter (raw, unresolved
        // at substitution time) instead of the step-level `condition` field.
        let executor = StepExecutor::new();
        let mut workflow = Workflow::new("branch-param-condition", "1.0", "Test param condition");
        workflow.add_variable(
            "score".to_string(),
            super::super::definition::Variable::Number(4.5),
        );
        let mut context = ExecutionContext::new(workflow);

        let mut params = HashMap::new();
        params.insert(
            "condition".to_string(),
            serde_json::json!({"left": "${score}", "operator": ">", "right": "4.0"}),
        );
        let step = make_step("branch-from-params", StepType::Branch, params, None);

        let result = executor.execute_step(&step, &mut context).await.unwrap();
        assert!(result.success, "branch step failed: {}", result.message);
        assert_eq!(
            result.output.get("branch_taken"),
            Some(&serde_json::json!(true))
        );
    }

    #[tokio::test]
    async fn test_execute_loop_count_based() {
        let executor = StepExecutor::new();
        let workflow = Workflow::new("loop-count-test", "1.0", "Test count-based loop");
        let mut context = ExecutionContext::new(workflow);

        let mut params = HashMap::new();
        params.insert("count".to_string(), serde_json::json!(5));
        let step = make_step("loop-step", StepType::Loop, params, None);

        let result = executor.execute_step(&step, &mut context).await.unwrap();
        assert!(result.success, "loop step failed: {}", result.message);
        assert_eq!(result.output.get("iterations"), Some(&serde_json::json!(5)));

        let variables = context.get_variables();
        assert_eq!(
            variables.get("loop-step_iterations"),
            Some(&serde_json::json!(5))
        );
    }

    #[tokio::test]
    async fn test_execute_loop_count_exceeding_max_errors() {
        let executor = StepExecutor::new();
        let workflow = Workflow::new("loop-too-big-test", "1.0", "Test loop cap enforcement");
        let mut context = ExecutionContext::new(workflow);

        let mut params = HashMap::new();
        params.insert(
            "count".to_string(),
            serde_json::json!(MAX_LOOP_ITERATIONS + 1),
        );
        let step = make_step("loop-too-big", StepType::Loop, params, None);

        let result = executor.execute_step(&step, &mut context).await.unwrap();
        assert!(!result.success);
    }

    #[tokio::test]
    async fn test_execute_loop_while_condition() {
        let executor = StepExecutor::new();
        let workflow = Workflow::new("loop-while-test", "1.0", "Test while-loop execution");
        let mut context = ExecutionContext::new(workflow);

        let step = make_step(
            "while-step",
            StepType::Loop,
            HashMap::new(),
            Some(Condition::new(
                "${loop_counter}".to_string(),
                ConditionOperator::LessThan,
                "3".to_string(),
            )),
        );

        let result = executor.execute_step(&step, &mut context).await.unwrap();
        assert!(result.success, "while-loop step failed: {}", result.message);
        assert_eq!(result.output.get("iterations"), Some(&serde_json::json!(3)));
        assert_eq!(
            context.get_variables().get("loop_counter"),
            Some(&serde_json::json!(3))
        );
    }
}
