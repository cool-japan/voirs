//! Workflow automation command implementations.
//!
//! This module provides commands for managing and executing declarative workflows
//! for complex multi-step synthesis pipelines.

use crate::error::{CliError, Result};
use crate::workflow::{
    ExecutionState, StateManager, Workflow, WorkflowEngine, WorkflowRegistry, WorkflowValidator,
};
use clap::Subcommand;
use std::path::PathBuf;

/// Resolve the default VoiRS working directory (`<cwd>/.voirs/<subdir>`)
/// used for workflow state/registry storage when the user did not pass an
/// explicit `--state-dir`/`--registry-dir`.
///
/// Propagates a typed `CliError` instead of panicking if the current
/// directory cannot be determined (e.g. it was deleted out from under the
/// running process, or the process lacks permission to stat it).
fn default_voirs_dir(subdir: &str) -> Result<PathBuf> {
    let cwd = std::env::current_dir()
        .map_err(|e| CliError::Workflow(format!("Failed to determine current directory: {}", e)))?;
    Ok(cwd.join(".voirs").join(subdir))
}

/// Workflow automation commands
#[derive(Subcommand)]
pub enum WorkflowCommands {
    /// Execute a workflow from a definition file
    Execute {
        /// Path to workflow definition file (YAML or JSON)
        workflow_file: PathBuf,

        /// Override workflow variables (key=value format)
        #[arg(short = 'v', long = "var", value_parser = parse_key_val)]
        variables: Vec<(String, String)>,

        /// Maximum number of parallel steps
        #[arg(long, default_value = "4")]
        max_parallel: usize,

        /// Resume from previous execution if available
        #[arg(long)]
        resume: bool,

        /// Storage directory for workflow state
        #[arg(long)]
        state_dir: Option<PathBuf>,
    },

    /// Validate a workflow definition without executing
    Validate {
        /// Path to workflow definition file (YAML or JSON)
        workflow_file: PathBuf,

        /// Show detailed validation results
        #[arg(long)]
        detailed: bool,

        /// Output format (text, json, yaml)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// List all registered workflows
    List {
        /// Workflow registry directory
        #[arg(long)]
        registry_dir: Option<PathBuf>,

        /// Show detailed information
        #[arg(long)]
        detailed: bool,
    },

    /// Show status of a workflow execution
    Status {
        /// Workflow name
        workflow_name: String,

        /// Storage directory for workflow state
        #[arg(long)]
        state_dir: Option<PathBuf>,

        /// Output format (text, json, yaml)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Resume a failed or stopped workflow
    Resume {
        /// Workflow name to resume
        workflow_name: String,

        /// Storage directory for workflow state
        #[arg(long)]
        state_dir: Option<PathBuf>,

        /// Maximum number of parallel steps
        #[arg(long, default_value = "4")]
        max_parallel: usize,
    },

    /// Stop a running workflow
    Stop {
        /// Workflow name to stop
        workflow_name: String,

        /// Storage directory for workflow state
        #[arg(long)]
        state_dir: Option<PathBuf>,

        /// Force stop without saving state
        #[arg(long)]
        force: bool,
    },
}

/// Parse a key-value pair for variable overrides
fn parse_key_val(s: &str) -> std::result::Result<(String, String), String> {
    let pos = s
        .find('=')
        .ok_or_else(|| format!("Invalid KEY=value: no `=` found in `{}`", s))?;
    Ok((s[..pos].to_string(), s[pos + 1..].to_string()))
}

/// Execute a workflow from a definition file
pub async fn run_workflow_execute(
    workflow_file: PathBuf,
    variables: Vec<(String, String)>,
    max_parallel: usize,
    resume: bool,
    state_dir: Option<PathBuf>,
) -> Result<()> {
    println!("Loading workflow from: {}", workflow_file.display());

    // Load workflow definition
    let mut workflow = Workflow::load_from_file(&workflow_file).await?;

    // Apply variable overrides
    for (key, value) in variables {
        workflow
            .variables
            .insert(key.clone(), crate::workflow::Variable::String(value));
    }

    println!("Workflow: {}", workflow.metadata.name);
    println!("Version: {}", workflow.metadata.version);
    if !workflow.metadata.description.is_empty() {
        println!("Description: {}", workflow.metadata.description);
    }
    println!("Steps: {}", workflow.steps.len());
    println!();

    // Determine state directory
    let state_dir = match state_dir {
        Some(dir) => dir,
        None => default_voirs_dir("workflow_state")?,
    };

    // Check if we should resume
    if resume {
        let state_manager = StateManager::new(state_dir.clone());
        if state_manager.exists(&workflow.metadata.name).await {
            println!("Resuming workflow from previous state...");
            println!();
        }
    }

    // Create workflow engine
    let engine = WorkflowEngine::new(state_dir, max_parallel);

    // Execute workflow
    println!("Executing workflow...");
    println!("─────────────────────────────────────────────────");

    let result = engine.execute(workflow).await?;

    println!("─────────────────────────────────────────────────");
    println!();
    println!("Workflow execution completed!");
    println!();
    println!("Statistics:");
    println!("  Total steps: {}", result.stats.total_steps);
    println!("  Successful: {}", result.stats.successful_steps);
    println!("  Failed: {}", result.stats.failed_steps);
    println!("  Skipped: {}", result.stats.skipped_steps);
    println!("  Total duration: {}ms", result.stats.total_duration_ms);
    println!(
        "  Average step duration: {}ms",
        result.stats.avg_step_duration_ms
    );
    println!("  Total retries: {}", result.stats.total_retries);
    println!(
        "  Success rate: {:.1}%",
        result.stats.success_rate() * 100.0
    );

    if result.stats.is_successful() {
        println!();
        println!("✓ All steps completed successfully");
    } else {
        println!();
        println!("✗ Some steps failed");
        return Err(crate::error::CliError::Workflow(
            "Workflow execution had failures".to_string(),
        ));
    }

    Ok(())
}

/// Validate a workflow definition
pub async fn run_workflow_validate(
    workflow_file: PathBuf,
    detailed: bool,
    format: String,
) -> Result<()> {
    println!("Validating workflow: {}", workflow_file.display());
    println!();

    // Load workflow definition
    let workflow = Workflow::load_from_file(&workflow_file).await?;

    // Create validator
    let validator = WorkflowValidator::new();

    // Validate workflow
    let result = validator.validate(&workflow)?;

    // Output results based on format
    match format.as_str() {
        "json" => {
            let json = serde_json::to_string_pretty(&result)?;
            println!("{}", json);
        }
        "yaml" => {
            let yaml = serde_yaml::to_string(&result).map_err(|e| {
                crate::error::CliError::SerializationError(format!(
                    "Failed to serialize to YAML: {}",
                    e
                ))
            })?;
            println!("{}", yaml);
        }
        _ => {
            // Text format
            if result.valid {
                println!("✓ Workflow validation passed");
            } else {
                println!("✗ Workflow validation failed");
            }
            println!();

            if result.has_errors() {
                println!("Errors:");
                for error in &result.errors {
                    println!("  - {}", error);
                }
                println!();
            }

            if result.has_warnings() {
                println!("Warnings:");
                for warning in &result.warnings {
                    println!("  - {}", warning);
                }
                println!();
            }

            if detailed && result.valid {
                println!("Workflow Details:");
                println!("  Name: {}", workflow.metadata.name);
                println!("  Version: {}", workflow.metadata.version);
                if !workflow.metadata.description.is_empty() {
                    println!("  Description: {}", workflow.metadata.description);
                }
                println!("  Steps: {}", workflow.steps.len());
                println!("  Variables: {}", workflow.variables.len());
                println!("  Max parallel: {}", workflow.config.max_parallel);
            }
        }
    }

    if !result.valid {
        return Err(crate::error::CliError::Workflow(
            "Workflow validation failed".to_string(),
        ));
    }

    Ok(())
}

/// List all registered workflows
pub async fn run_workflow_list(registry_dir: Option<PathBuf>, detailed: bool) -> Result<()> {
    // Determine registry directory
    let registry_dir = match registry_dir {
        Some(dir) => dir,
        None => default_voirs_dir("workflows")?,
    };

    let registry = WorkflowRegistry::new(registry_dir.clone());

    // Load workflows from directory
    let count = registry.load_from_directory().await?;

    if count == 0 {
        println!("No workflows found in: {}", registry_dir.display());
        println!();
        println!("Create workflow definitions in this directory or specify a different path with --registry-dir");
        return Ok(());
    }

    println!("Registered workflows ({} found):", count);
    println!();

    let workflow_names = registry.list().await;

    for name in workflow_names {
        if let Some(workflow) = registry.get(&name).await {
            println!("  • {}", workflow.metadata.name);
            if detailed {
                println!("    Version: {}", workflow.metadata.version);
                if !workflow.metadata.description.is_empty() {
                    println!("    Description: {}", workflow.metadata.description);
                }
                println!("    Steps: {}", workflow.steps.len());
                println!("    Variables: {}", workflow.variables.len());
                println!();
            }
        }
    }

    Ok(())
}

/// Show status of a workflow execution
pub async fn run_workflow_status(
    workflow_name: String,
    state_dir: Option<PathBuf>,
    format: String,
) -> Result<()> {
    // Determine state directory
    let state_dir = match state_dir {
        Some(dir) => dir,
        None => default_voirs_dir("workflow_state")?,
    };

    let state_manager = StateManager::new(state_dir);

    // Check if state exists
    if !state_manager.exists(&workflow_name).await {
        return Err(crate::error::CliError::Workflow(format!(
            "No state found for workflow: {}",
            workflow_name
        )));
    }

    // Load state
    let state = state_manager.load(&workflow_name).await?;

    // Output based on format
    match format.as_str() {
        "json" => {
            let json = serde_json::to_string_pretty(&state)?;
            println!("{}", json);
        }
        "yaml" => {
            let yaml = serde_yaml::to_string(&state).map_err(|e| {
                crate::error::CliError::SerializationError(format!(
                    "Failed to serialize to YAML: {}",
                    e
                ))
            })?;
            println!("{}", yaml);
        }
        _ => {
            // Text format
            println!("Workflow: {}", state.workflow_name);
            println!("State: {:?}", state.state);
            println!();

            println!("Progress:");
            println!("  Completed steps: {}", state.completed_steps.len());
            println!("  Skipped steps: {}", state.skipped_steps.len());
            if let Some(ref current) = state.current_step {
                println!("  Current step: {}", current);
            }
            println!("  Total retries: {}", state.total_retries);
            println!();

            println!("Variables: {}", state.variables.len());
            for (key, value) in &state.variables {
                println!("  {}: {:?}", key, value);
            }
            println!();

            println!(
                "Last updated: {}",
                state.last_updated.format("%Y-%m-%d %H:%M:%S UTC")
            );
            println!();

            if state.can_resume() {
                println!("✓ This workflow can be resumed");
            } else {
                println!(
                    "✗ This workflow cannot be resumed (state: {:?})",
                    state.state
                );
            }
        }
    }

    Ok(())
}

/// Resume a failed or stopped workflow
pub async fn run_workflow_resume(
    workflow_name: String,
    state_dir: Option<PathBuf>,
    max_parallel: usize,
) -> Result<()> {
    // Determine state directory
    let state_dir = match state_dir {
        Some(dir) => dir,
        None => default_voirs_dir("workflow_state")?,
    };

    let state_manager = StateManager::new(state_dir.clone());

    // Check if state exists
    if !state_manager.exists(&workflow_name).await {
        return Err(crate::error::CliError::Workflow(format!(
            "No state found for workflow: {}",
            workflow_name
        )));
    }

    // Load state
    let state = state_manager.load(&workflow_name).await?;

    // Check if workflow can be resumed
    if !state.can_resume() {
        return Err(crate::error::CliError::Workflow(format!(
            "Workflow '{}' cannot be resumed (current state: {:?})",
            workflow_name, state.state
        )));
    }

    println!("Resuming workflow: {}", workflow_name);
    println!("Current state: {:?}", state.state);
    println!("Completed steps: {}", state.completed_steps.len());
    println!();

    // Note: Resume functionality would require loading the original workflow definition
    // and passing the existing state to the engine. This is a simplified version.
    println!("⚠ Resume functionality requires the original workflow definition file");
    println!("  Use: voirs workflow execute <workflow-file> --resume");

    Ok(())
}

/// Stop a running workflow
pub async fn run_workflow_stop(
    workflow_name: String,
    state_dir: Option<PathBuf>,
    force: bool,
) -> Result<()> {
    // Determine state directory
    let state_dir = match state_dir {
        Some(dir) => dir,
        None => default_voirs_dir("workflow_state")?,
    };

    let state_manager = StateManager::new(state_dir);

    // Check if state exists
    if !state_manager.exists(&workflow_name).await {
        return Err(crate::error::CliError::Workflow(format!(
            "No state found for workflow: {}",
            workflow_name
        )));
    }

    // Load state
    let mut state = state_manager.load(&workflow_name).await?;

    // Check if workflow is running
    if state.state != ExecutionState::Running {
        println!(
            "Workflow '{}' is not running (state: {:?})",
            workflow_name, state.state
        );
        return Ok(());
    }

    if force {
        // Force stop - delete state
        state_manager.delete(&workflow_name).await?;
        println!(
            "✓ Workflow '{}' forcefully stopped (state deleted)",
            workflow_name
        );
    } else {
        // Graceful stop - mark as stopped
        state.state = ExecutionState::Stopped;
        state.last_updated = chrono::Utc::now();
        state_manager.save(&workflow_name, &state).await?;
        println!("✓ Workflow '{}' stopped gracefully", workflow_name);
        println!("  State saved for potential resume");
    }

    Ok(())
}
