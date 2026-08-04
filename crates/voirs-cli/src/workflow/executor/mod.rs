//! Step Executor
//!
//! Executes individual workflow steps with retry logic and state management.

use super::{
    definition::{Condition, Step, StepType, Workflow},
    retry::RetryManager,
    state::WorkflowState,
    validation::WorkflowValidator,
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

/// Maximum sub-workflow nesting depth (`Workflow`-type steps invoking other
/// workflow files). Combined with `ExecutionContext::subworkflow_stack`'s
/// cycle detection, this guards against both accidental cycles that
/// round-trip through more than one file and simple runaway nesting.
const MAX_SUBWORKFLOW_DEPTH: usize = 16;

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
    /// Canonicalized paths of sub-workflow files currently being executed,
    /// outermost first. Used by `execute_subworkflow` to detect cycles
    /// (a workflow transitively including itself) and to enforce
    /// `MAX_SUBWORKFLOW_DEPTH`. Empty for the top-level workflow, which has
    /// no source file of its own from `ExecutionContext`'s point of view.
    subworkflow_stack: Vec<PathBuf>,
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
            subworkflow_stack: Vec::new(),
        }
    }

    /// Create a child execution context for a sub-workflow invoked from
    /// `path` (already canonicalized by the caller). Fails if `path` is
    /// already on the stack (a cycle) or the stack is already at
    /// `MAX_SUBWORKFLOW_DEPTH`, before doing any work on the sub-workflow
    /// itself.
    fn child_for_subworkflow(&self, sub_workflow: Workflow, path: &Path) -> Result<Self> {
        if self.subworkflow_stack.iter().any(|p| p == path) {
            let chain = self
                .subworkflow_stack
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(" -> ");
            return Err(CliError::Workflow(format!(
                "sub-workflow cycle detected: '{}' is already being executed (chain: {} -> {})",
                path.display(),
                chain,
                path.display()
            )));
        }
        if self.subworkflow_stack.len() >= MAX_SUBWORKFLOW_DEPTH {
            return Err(CliError::Workflow(format!(
                "sub-workflow nesting exceeded the maximum depth of {}",
                MAX_SUBWORKFLOW_DEPTH
            )));
        }

        let mut child = Self::new(sub_workflow);
        child.subworkflow_stack = self.subworkflow_stack.clone();
        child.subworkflow_stack.push(path.to_path_buf());
        Ok(child)
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
            StepType::Validate => self.execute_validate(step, &resolved_params, context).await,
            StepType::FileOp => self.execute_file_op(step, &resolved_params).await,
            StepType::Command => self.execute_command(step, &resolved_params).await,
            StepType::Script => self.execute_script(step, &resolved_params).await,
            StepType::Branch => self.execute_branch(step, &resolved_params, context).await,
            StepType::Loop => self.execute_loop(step, &resolved_params, context).await,
            StepType::Workflow => {
                self.execute_subworkflow(step, &resolved_params, context)
                    .await
            }
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

    /// Run real text-to-speech synthesis through `VoirsPipeline` and write
    /// the resulting audio to `output`. `text` and `output` are required;
    /// `voice`, `quality`, `gpu`, `rate`, `pitch`, and `volume` are optional
    /// overrides mirroring the CLI's `synthesize` command.
    ///
    /// `test_mode` (default `false`) maps directly to
    /// `VoirsPipelineBuilder::with_test_mode`: when set, model
    /// auto-download and validation are skipped, so synthesis falls back to
    /// the SDK's built-in dummy G2P/acoustic/vocoder models whenever real
    /// weights aren't already cached locally. This is meant for CI/smoke-test
    /// workflows that need to exercise the synthesize step without network
    /// access or downloaded models -- the audio produced in that case is
    /// *not* production-quality speech, and workflow authors should not set
    /// `test_mode: true` outside of testing.
    async fn execute_synthesize(
        &self,
        step: &Step,
        params: &HashMap<String, serde_json::Value>,
    ) -> Result<StepOutcome> {
        let text = params
            .get("text")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                CliError::Workflow(format!(
                    "synthesize step '{}' requires a non-empty 'text' parameter",
                    step.name
                ))
            })?;

        let output_param = params
            .get("output")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                CliError::Workflow(format!(
                    "synthesize step '{}' requires an 'output' parameter (output audio file path)",
                    step.name
                ))
            })?;

        let cwd = resolve_cwd(params)?;
        let output_path = resolve_path(&cwd, output_param);

        let quality = match params.get("quality").and_then(|v| v.as_str()) {
            Some(quality_str) => quality_str
                .parse::<voirs_sdk::QualityLevel>()
                .map_err(|e| {
                    CliError::Workflow(format!(
                        "synthesize step '{}' has an invalid 'quality' parameter: {e}",
                        step.name
                    ))
                })?,
            None => voirs_sdk::QualityLevel::default(),
        };

        let mut builder = voirs_sdk::VoirsPipeline::builder().with_quality(quality);
        if let Some(voice) = params.get("voice").and_then(|v| v.as_str()) {
            builder = builder.with_voice(voice);
        }
        if let Some(gpu) = params.get("gpu").and_then(|v| v.as_bool()) {
            builder = builder.with_gpu_acceleration(gpu);
        }
        if let Some(test_mode) = params.get("test_mode").and_then(|v| v.as_bool()) {
            builder = builder.with_test_mode(test_mode);
        }

        let pipeline = builder.build().await?;

        let mut synth_config = voirs_sdk::SynthesisConfig {
            quality,
            ..Default::default()
        };
        if let Some(rate) = params.get("rate").and_then(|v| v.as_f64()) {
            synth_config.speaking_rate = rate as f32;
        }
        if let Some(pitch) = params.get("pitch").and_then(|v| v.as_f64()) {
            synth_config.pitch_shift = pitch as f32;
        }
        if let Some(volume) = params.get("volume").and_then(|v| v.as_f64()) {
            synth_config.volume_gain = volume as f32;
        }

        let audio = pipeline.synthesize_with_config(text, &synth_config).await?;

        ensure_parent_dir(&output_path)?;
        let format = crate::utils::format_from_extension(&output_path).unwrap_or_default();
        audio.save(&output_path, format)?;

        let mut output = HashMap::new();
        output.insert(
            "output_path".to_string(),
            serde_json::json!(output_path.display().to_string()),
        );
        output.insert(
            "duration_seconds".to_string(),
            serde_json::json!(audio.duration()),
        );
        output.insert(
            "sample_rate".to_string(),
            serde_json::json!(audio.sample_rate()),
        );

        Ok(StepOutcome::with_output(
            format!(
                "Synthesized {} character(s) to '{}' ({:.2}s audio)",
                text.chars().count(),
                output_path.display(),
                audio.duration()
            ),
            output,
        ))
    }

    /// Validate referenced files/state against the checks specified in
    /// `params`: file existence and size bounds (`path`, `min_size_bytes`,
    /// `max_size_bytes`), WAV audio decodability and duration bounds
    /// (`audio`/extension-inferred, `min_duration_secs`, `max_duration_secs`),
    /// exact format/extension (`format`), JSON structural validity plus
    /// required top-level keys (`required_fields`), and/or a condition
    /// (step-level `condition` or a `condition` parameter, evaluated via the
    /// same `Condition::evaluate` infrastructure `execute_branch` uses). At
    /// least one check must be requested -- a validate step with nothing to
    /// check is a configuration error, not a pass.
    async fn execute_validate(
        &self,
        step: &Step,
        params: &HashMap<String, serde_json::Value>,
        context: &mut ExecutionContext,
    ) -> Result<StepOutcome> {
        let mut checks_performed: Vec<&'static str> = Vec::new();
        let mut output = HashMap::new();

        if let Some(path_param) = params.get("path").and_then(|v| v.as_str()) {
            let cwd = resolve_cwd(params)?;
            let path = resolve_path(&cwd, path_param);

            if !path.exists() {
                return Err(CliError::Workflow(format!(
                    "validate step '{}': path '{}' does not exist",
                    step.name,
                    path.display()
                )));
            }
            checks_performed.push("exists");

            let metadata = std::fs::metadata(&path)
                .map_err(|e| CliError::file_operation("stat", &path.display().to_string(), e))?;
            let size = metadata.len();
            output.insert("size_bytes".to_string(), serde_json::json!(size));

            if let Some(min_size) = params.get("min_size_bytes").and_then(|v| v.as_u64()) {
                if size < min_size {
                    return Err(CliError::Workflow(format!(
                        "validate step '{}': '{}' is {} byte(s), below the required minimum of {}",
                        step.name,
                        path.display(),
                        size,
                        min_size
                    )));
                }
                checks_performed.push("min_size_bytes");
            }
            if let Some(max_size) = params.get("max_size_bytes").and_then(|v| v.as_u64()) {
                if size > max_size {
                    return Err(CliError::Workflow(format!(
                        "validate step '{}': '{}' is {} byte(s), above the allowed maximum of {}",
                        step.name,
                        path.display(),
                        size,
                        max_size
                    )));
                }
                checks_performed.push("max_size_bytes");
            }

            let extension = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_lowercase());

            if let Some(expected_format) = params.get("format").and_then(|v| v.as_str()) {
                if extension.as_deref() != Some(expected_format.to_lowercase().as_str()) {
                    return Err(CliError::Workflow(format!(
                        "validate step '{}': '{}' does not have the expected '{}' extension",
                        step.name,
                        path.display(),
                        expected_format
                    )));
                }
                checks_performed.push("format");
            }

            let want_audio_check = params
                .get("audio")
                .and_then(|v| v.as_bool())
                .unwrap_or(matches!(extension.as_deref(), Some("wav")));
            if want_audio_check {
                if extension.as_deref() != Some("wav") {
                    return Err(CliError::Workflow(format!(
                        "validate step '{}': audio decodability check requested for '{}', but only WAV is currently supported for decode validation",
                        step.name, path.display()
                    )));
                }

                let reader = hound::WavReader::open(&path).map_err(|e| {
                    CliError::Workflow(format!(
                        "validate step '{}': '{}' is not decodable as WAV audio: {}",
                        step.name,
                        path.display(),
                        e
                    ))
                })?;
                let spec = reader.spec();
                let duration_secs = if spec.sample_rate > 0 {
                    reader.duration() as f64 / f64::from(spec.sample_rate)
                } else {
                    0.0
                };
                output.insert(
                    "audio_duration_secs".to_string(),
                    serde_json::json!(duration_secs),
                );
                output.insert(
                    "audio_sample_rate".to_string(),
                    serde_json::json!(spec.sample_rate),
                );
                output.insert(
                    "audio_channels".to_string(),
                    serde_json::json!(spec.channels),
                );
                checks_performed.push("audio_decodable");

                if let Some(min_dur) = params.get("min_duration_secs").and_then(|v| v.as_f64()) {
                    if duration_secs < min_dur {
                        return Err(CliError::Workflow(format!(
                            "validate step '{}': '{}' is {:.3}s, below the required minimum of {:.3}s",
                            step.name, path.display(), duration_secs, min_dur
                        )));
                    }
                    checks_performed.push("min_duration_secs");
                }
                if let Some(max_dur) = params.get("max_duration_secs").and_then(|v| v.as_f64()) {
                    if duration_secs > max_dur {
                        return Err(CliError::Workflow(format!(
                            "validate step '{}': '{}' is {:.3}s, above the allowed maximum of {:.3}s",
                            step.name, path.display(), duration_secs, max_dur
                        )));
                    }
                    checks_performed.push("max_duration_secs");
                }
            }

            if let Some(required_fields) = params.get("required_fields").and_then(|v| v.as_array())
            {
                let content = std::fs::read_to_string(&path).map_err(|e| {
                    CliError::file_operation("read", &path.display().to_string(), e)
                })?;
                let json: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
                    CliError::Workflow(format!(
                        "validate step '{}': '{}' is not valid JSON: {}",
                        step.name,
                        path.display(),
                        e
                    ))
                })?;
                checks_performed.push("json_parses");

                for field in required_fields {
                    let field_name = field.as_str().ok_or_else(|| {
                        CliError::Workflow(format!(
                            "validate step '{}' has a non-string entry in 'required_fields'",
                            step.name
                        ))
                    })?;
                    if json.get(field_name).is_none() {
                        return Err(CliError::Workflow(format!(
                            "validate step '{}': '{}' is missing required field '{}'",
                            step.name,
                            path.display(),
                            field_name
                        )));
                    }
                }
                checks_performed.push("required_fields");
            }
        }

        if step.condition.is_some() || params.contains_key("condition") {
            let condition = Self::step_condition(step, "validate")?;
            let variables = context.get_variables();
            if !condition.evaluate(&variables) {
                return Err(CliError::Workflow(format!(
                    "validate step '{}': condition was not satisfied",
                    step.name
                )));
            }
            checks_performed.push("condition");
        }

        if checks_performed.is_empty() {
            return Err(CliError::Workflow(format!(
                "validate step '{}' requires at least one check: a 'path' parameter (with optional \
                 size/format/audio/required_fields checks) and/or a condition",
                step.name
            )));
        }

        output.insert(
            "checks_performed".to_string(),
            serde_json::json!(checks_performed),
        );

        Ok(StepOutcome::with_output(
            format!(
                "Validation passed for step '{}' ({} check(s): {})",
                step.name,
                checks_performed.len(),
                checks_performed.join(", ")
            ),
            output,
        ))
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

    /// Load and recursively execute a referenced workflow file. The step's
    /// `workflow_file` (or `path`) parameter names the sub-workflow file
    /// (relative paths resolve against `cwd`, matching the other
    /// file-touching step types); an optional `inputs` object seeds/overrides
    /// the child workflow's variables with values already resolved against
    /// the *parent's* variables (standard `${var}` substitution already ran
    /// before this handler is called).
    ///
    /// Guarded against cycles and excessive nesting via
    /// `ExecutionContext::child_for_subworkflow`, which tracks the
    /// canonicalized chain of sub-workflow files currently executing.
    ///
    /// Runs the sub-workflow's steps sequentially in dependency order
    /// (honoring `depends_on` and per-step `condition`s, see
    /// `run_workflow_steps`) rather than delegating to `WorkflowEngine`:
    /// `WorkflowEngine::execute` mints a brand-new `ExecutionContext` per
    /// call, which would sever the ancestor chain this function needs for
    /// cycle detection across more than one level of nesting.
    async fn execute_subworkflow(
        &self,
        step: &Step,
        params: &HashMap<String, serde_json::Value>,
        context: &mut ExecutionContext,
    ) -> Result<StepOutcome> {
        let path_param = params
            .get("workflow_file")
            .or_else(|| params.get("path"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                CliError::Workflow(format!(
                    "workflow step '{}' requires a 'workflow_file' (or 'path') parameter naming \
                     the sub-workflow file",
                    step.name
                ))
            })?;

        let cwd = resolve_cwd(params)?;
        let resolved_path = resolve_path(&cwd, path_param);
        let canonical_path = std::fs::canonicalize(&resolved_path).map_err(|e| {
            CliError::file_operation("resolve", &resolved_path.display().to_string(), e)
        })?;

        let sub_workflow = Workflow::load_from_file(&canonical_path).await?;

        let validation = WorkflowValidator::new().validate(&sub_workflow)?;
        if !validation.valid {
            let messages = validation
                .errors
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            return Err(CliError::Workflow(format!(
                "workflow step '{}': sub-workflow '{}' failed validation: {}",
                step.name,
                canonical_path.display(),
                messages
            )));
        }

        let mut child_context =
            context.child_for_subworkflow(sub_workflow.clone(), &canonical_path)?;

        if let Some(inputs) = params.get("inputs").and_then(|v| v.as_object()) {
            for (key, value) in inputs {
                child_context.set_variable(key.clone(), value.clone());
            }
        }

        let (succeeded, failed) = self
            .run_workflow_steps(&sub_workflow, &mut child_context)
            .await?;

        let mut output = HashMap::new();
        output.insert(
            "workflow_name".to_string(),
            serde_json::json!(sub_workflow.metadata.name),
        );
        output.insert(
            "workflow_file".to_string(),
            serde_json::json!(canonical_path.display().to_string()),
        );
        output.insert("steps_succeeded".to_string(), serde_json::json!(succeeded));
        output.insert("steps_failed".to_string(), serde_json::json!(failed));

        Ok(StepOutcome::with_output(
            format!(
                "Sub-workflow '{}' ({}) completed: {} succeeded, {} failed",
                sub_workflow.metadata.name,
                canonical_path.display(),
                succeeded,
                failed
            ),
            output,
        ))
    }

    /// Execute every step of `workflow` sequentially, in dependency order,
    /// evaluating each step's condition first and skipping it if unmet.
    ///
    /// This mirrors the skip-on-condition check `WorkflowEngine::execute_steps`
    /// performs before dispatching a step (that check lives in the engine's
    /// per-step task body, not in `execute_step`/`execute_step_once`, so it
    /// has to be reproduced here rather than inherited for free). Unlike the
    /// engine, this does not run independent steps in parallel and does not
    /// persist state to a `StateManager`; it exists purely to give
    /// `execute_subworkflow` a self-contained, depth/cycle-guarded runner
    /// built on the same `ExecutionContext`/`StepExecutor` machinery instead
    /// of on `WorkflowEngine` (see `execute_subworkflow`'s doc comment for
    /// why). Returns `(steps_succeeded, steps_failed)`.
    async fn run_workflow_steps(
        &self,
        workflow: &Workflow,
        context: &mut ExecutionContext,
    ) -> Result<(usize, usize)> {
        let mut pending: Vec<&Step> = workflow.steps.iter().collect();
        let mut succeeded = 0usize;
        let mut failed = 0usize;

        while !pending.is_empty() {
            let ready: Vec<&Step> = pending
                .iter()
                .copied()
                .filter(|step| {
                    step.depends_on.iter().all(|dep| {
                        context.completed_steps().contains_key(&dep.step_name)
                            || context.skipped_steps().contains(&dep.step_name)
                    })
                })
                .collect();

            if ready.is_empty() {
                return Err(CliError::Workflow(format!(
                    "sub-workflow '{}' has unsatisfiable step dependencies",
                    workflow.metadata.name
                )));
            }

            for step in ready {
                if let Some(ref condition) = step.condition {
                    let variables = context.get_variables();
                    if !condition.evaluate(&variables) {
                        context.skip_step(&step.name, "Condition not met");
                        continue;
                    }
                }

                // `execute_step` transitively (via `execute_subworkflow`)
                // calls back into `run_workflow_steps`, so this call must be
                // boxed: an async fn that recurses into itself without
                // indirection would require an infinitely-sized future.
                let result: StepResult = Box::pin(self.execute_step(step, context)).await?;
                if result.success {
                    succeeded += 1;
                } else {
                    failed += 1;
                    if !workflow.config.continue_on_error {
                        return Err(CliError::Workflow(format!(
                            "sub-workflow '{}' step '{}' failed: {}",
                            workflow.metadata.name, step.name, result.message
                        )));
                    }
                }
            }

            pending.retain(|step| {
                !context.completed_steps().contains_key(&step.name)
                    && !context.skipped_steps().contains(&step.name)
            });
        }

        Ok((succeeded, failed))
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

    /// Emit a workflow notification. A structured summary is always printed
    /// to stdout (real local delivery, not a stand-in for a "real" channel),
    /// and if `webhook_url` (or `url`) is set, the same information is also
    /// POSTed as JSON to that URL via `reqwest`. A configured webhook that
    /// can't be reached, or that responds with a non-2xx status, is a hard
    /// error: this step type never reports success for a delivery that
    /// didn't happen.
    async fn execute_notify(
        &self,
        step: &Step,
        params: &HashMap<String, serde_json::Value>,
    ) -> Result<StepOutcome> {
        let message = params
            .get("message")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                CliError::Workflow(format!(
                    "notify step '{}' requires a 'message' parameter",
                    step.name
                ))
            })?;

        let channel = params.get("channel").and_then(|v| v.as_str());
        let level = params
            .get("level")
            .and_then(|v| v.as_str())
            .unwrap_or("info");
        let timestamp = chrono::Utc::now();

        // Always emit a local, structured notification. This is genuine
        // delivery to the workflow's own output stream, not a placeholder.
        println!(
            "[notify:{}] {} - {}{}",
            level,
            timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
            message,
            channel
                .map(|c| format!(" (channel: {c})"))
                .unwrap_or_default(),
        );

        let mut output = HashMap::new();
        output.insert("message".to_string(), serde_json::json!(message));
        output.insert("level".to_string(), serde_json::json!(level));
        output.insert("delivered_stdout".to_string(), serde_json::json!(true));
        if let Some(channel) = channel {
            output.insert("channel".to_string(), serde_json::json!(channel));
        }

        let webhook_url = params
            .get("webhook_url")
            .or_else(|| params.get("url"))
            .and_then(|v| v.as_str());

        let Some(webhook_url) = webhook_url else {
            return Ok(StepOutcome::with_output(
                format!("Notification '{}' delivered to stdout", step.name),
                output,
            ));
        };

        // Install the pure-Rust rustls CryptoProvider before any TLS
        // handshake (reqwest is built with `rustls-no-provider`).
        // Once-guarded; safe to call on every notify step.
        voirs_sdk::ensure_crypto_provider();

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .user_agent(concat!("voirs-cli/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| CliError::NetworkError(format!("failed to build HTTP client: {e}")))?;

        let mut payload = serde_json::json!({
            "step": step.name,
            "message": message,
            "level": level,
            "timestamp": timestamp.to_rfc3339(),
        });
        if let Some(obj) = payload.as_object_mut() {
            if let Some(channel) = channel {
                obj.insert("channel".to_string(), serde_json::json!(channel));
            }
            if let Some(extra) = params.get("payload").and_then(|v| v.as_object()) {
                for (key, value) in extra {
                    obj.insert(key.clone(), value.clone());
                }
            }
        }

        let response = client
            .post(webhook_url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| {
                CliError::NetworkError(format!(
                    "notify step '{}' failed to reach webhook '{}': {}",
                    step.name, webhook_url, e
                ))
            })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(CliError::NetworkError(format!(
                "notify step '{}': webhook '{}' responded with status {}: {}",
                step.name, webhook_url, status, body
            )));
        }

        output.insert("webhook_url".to_string(), serde_json::json!(webhook_url));
        output.insert(
            "webhook_status".to_string(),
            serde_json::json!(status.as_u16()),
        );
        output.insert("delivered_webhook".to_string(), serde_json::json!(true));

        Ok(StepOutcome::with_output(
            format!(
                "Notification '{}' delivered to stdout and webhook '{}' ({})",
                step.name, webhook_url, status
            ),
            output,
        ))
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
mod tests;
