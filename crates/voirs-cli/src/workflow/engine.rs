//! Workflow Execution Engine
//!
//! The engine orchestrates workflow execution, managing dependencies,
//! parallel execution, and state persistence.

use super::{
    definition::{Step, Workflow},
    executor::{ExecutionContext, ExecutionResult, StepExecutor},
    state::{ExecutionState, StateManager},
    validation::WorkflowValidator,
    WorkflowStats,
};
use crate::error::CliError;

type Result<T> = std::result::Result<T, CliError>;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{RwLock, Semaphore};

/// Workflow execution engine
pub struct WorkflowEngine {
    /// Step executor
    executor: Arc<StepExecutor>,
    /// State manager for persistence
    state_manager: Arc<StateManager>,
    /// Maximum parallel executions
    max_parallel: usize,
    /// Workflow validator
    validator: WorkflowValidator,
}

impl WorkflowEngine {
    /// Create a new workflow engine
    pub fn new(state_dir: PathBuf, max_parallel: usize) -> Self {
        Self {
            executor: Arc::new(StepExecutor::new()),
            state_manager: Arc::new(StateManager::new(state_dir)),
            max_parallel,
            validator: WorkflowValidator::new(),
        }
    }

    /// Execute a workflow
    pub async fn execute(&self, workflow: Workflow) -> Result<ExecutionResult> {
        // Validate workflow
        self.validator.validate(&workflow)?;

        let start_time = Instant::now();
        let workflow_name = workflow.metadata.name.clone();

        // Initialize execution context
        let mut context = ExecutionContext::new(workflow.clone());

        // Load previous state if resuming
        if workflow.config.save_state {
            if let Ok(state) = self.state_manager.load(&workflow_name).await {
                context.resume_from_state(state);
            }
        }

        // Execute steps
        let result = self.execute_steps(&workflow, &mut context).await;

        // Save final state if configured
        if workflow.config.save_state {
            let state = context.get_state();
            let _ = self.state_manager.save(&workflow_name, &state).await;
        }

        let duration = start_time.elapsed();

        // Collect statistics
        let stats = WorkflowStats {
            total_steps: context.completed_steps().len(),
            successful_steps: context
                .completed_steps()
                .iter()
                .filter(|(_, r)| r.success)
                .count(),
            failed_steps: context
                .completed_steps()
                .iter()
                .filter(|(_, r)| !r.success)
                .count(),
            skipped_steps: context.skipped_steps().len(),
            total_duration_ms: duration.as_millis() as u64,
            avg_step_duration_ms: if !context.completed_steps().is_empty() {
                duration.as_millis() as u64 / context.completed_steps().len() as u64
            } else {
                0
            },
            total_retries: context.total_retries(),
        };

        match result {
            Ok(_) => Ok(ExecutionResult::success(
                workflow_name,
                "Workflow completed successfully".to_string(),
                stats,
            )),
            Err(e) => Ok(ExecutionResult::failure(
                workflow_name,
                format!("Workflow failed: {}", e),
                stats,
            )),
        }
    }

    /// Execute workflow steps
    async fn execute_steps(
        &self,
        workflow: &Workflow,
        context: &mut ExecutionContext,
    ) -> Result<()> {
        let semaphore = Arc::new(Semaphore::new(
            workflow.config.max_parallel.min(self.max_parallel),
        ));

        // Build dependency graph
        let dep_graph = self.build_dependency_graph(workflow)?;

        // Track completed steps
        let completed = Arc::new(RwLock::new(HashSet::new()));

        // Execute steps respecting dependencies
        let mut pending_steps: Vec<&Step> = workflow.steps.iter().collect();

        while !pending_steps.is_empty() {
            // Find steps ready to execute (dependencies satisfied)
            let ready_steps: Vec<&Step> = pending_steps
                .iter()
                .filter(|step| {
                    // Check if already completed or skipped
                    if context.completed_steps().contains_key(&step.name) {
                        return false;
                    }

                    // Check dependencies
                    // Check if dependencies are satisfied using blocking read
                    let completed_set = {
                        let guard = completed.blocking_read();
                        guard.clone()
                    };
                    self.dependencies_satisfied(step, &completed_set)
                })
                .copied()
                .collect();

            if ready_steps.is_empty() {
                // Check if we're waiting for parallel tasks
                if context.completed_steps().len() + context.skipped_steps().len()
                    < workflow.steps.len()
                {
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    continue;
                }
                break;
            }

            // Execute ready steps
            let mut tasks = Vec::new();

            for step in ready_steps {
                let step_clone = step.clone();
                let context_clone = Arc::new(RwLock::new(context.clone()));
                let executor = self.executor.clone();
                let semaphore = semaphore.clone();
                let completed = completed.clone();
                let state_manager = self.state_manager.clone();
                let workflow_name = workflow.metadata.name.clone();
                let save_state = workflow.config.save_state;

                let task = tokio::spawn(async move {
                    let _permit = semaphore
                        .acquire()
                        .await
                        .expect("semaphore should not be closed");

                    let mut ctx = context_clone.write().await;

                    // Evaluate condition if present
                    if let Some(ref condition) = step_clone.condition {
                        let variables = ctx.get_variables();
                        if !condition.evaluate(&variables) {
                            ctx.skip_step(&step_clone.name, "Condition not met");
                            return Ok(());
                        }
                    }

                    // Execute step
                    let result = executor.execute_step(&step_clone, &mut ctx).await?;

                    if result.success {
                        completed.write().await.insert(step_clone.name.clone());
                    } else if !ctx.workflow().config.continue_on_error {
                        return Err(CliError::Workflow(format!(
                            "Step '{}' failed: {}",
                            step_clone.name, result.message
                        )));
                    }

                    // Save state after each step if configured
                    if save_state {
                        let state = ctx.get_state();
                        let _ = state_manager.save(&workflow_name, &state).await;
                    }

                    Ok::<(), CliError>(())
                });

                tasks.push(task);
            }

            // Wait for all tasks to complete
            for task in tasks {
                task.await
                    .map_err(|e| CliError::Workflow(format!("Task failed: {}", e)))??;
            }

            // Remove completed steps from pending
            pending_steps.retain(|step| {
                !context.completed_steps().contains_key(&step.name)
                    && !context.skipped_steps().contains(&step.name)
            });
        }

        Ok(())
    }

    /// Build dependency graph
    fn build_dependency_graph(&self, workflow: &Workflow) -> Result<HashMap<String, Vec<String>>> {
        let mut graph: HashMap<String, Vec<String>> = HashMap::new();

        for step in &workflow.steps {
            let deps: Vec<String> = step
                .depends_on
                .iter()
                .map(|d| d.step_name.clone())
                .collect();
            graph.insert(step.name.clone(), deps);
        }

        // Detect cycles
        let mut visited = HashSet::new();
        let mut recursion_stack = HashSet::new();

        for step in &workflow.steps {
            if Self::has_cycle(&step.name, &graph, &mut visited, &mut recursion_stack) {
                return Err(CliError::Workflow(format!(
                    "Circular dependency detected involving step '{}'",
                    step.name
                )));
            }
        }

        Ok(graph)
    }

    /// Check for cycles in dependency graph
    fn has_cycle(
        node: &str,
        graph: &HashMap<String, Vec<String>>,
        visited: &mut HashSet<String>,
        recursion_stack: &mut HashSet<String>,
    ) -> bool {
        if recursion_stack.contains(node) {
            return true;
        }

        if visited.contains(node) {
            return false;
        }

        visited.insert(node.to_string());
        recursion_stack.insert(node.to_string());

        if let Some(neighbors) = graph.get(node) {
            for neighbor in neighbors {
                if Self::has_cycle(neighbor, graph, visited, recursion_stack) {
                    return true;
                }
            }
        }

        recursion_stack.remove(node);
        false
    }

    /// Check if all dependencies are satisfied
    fn dependencies_satisfied(&self, step: &Step, completed: &HashSet<String>) -> bool {
        step.depends_on
            .iter()
            .all(|dep| completed.contains(&dep.step_name))
    }

    /// Stop a running workflow
    pub async fn stop(&self, workflow_name: &str) -> Result<()> {
        // Load current state
        if let Ok(state) = self.state_manager.load(workflow_name).await {
            if state.state == ExecutionState::Running {
                let mut updated_state = state;
                updated_state.state = ExecutionState::Stopped;
                self.state_manager
                    .save(workflow_name, &updated_state)
                    .await?;
            }
        }

        Ok(())
    }

    /// Resume a stopped or failed workflow
    pub async fn resume(&self, workflow: Workflow) -> Result<ExecutionResult> {
        // Similar to execute but loads state first
        self.execute(workflow).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::definition::{StepDependency, StepType};
    use std::env;

    fn create_test_workflow() -> Workflow {
        let mut workflow = Workflow::new("test-workflow", "1.0", "Test workflow");

        let step1 = Step {
            name: "step1".to_string(),
            step_type: StepType::Command,
            description: None,
            parameters: HashMap::new(),
            condition: None,
            depends_on: Vec::new(),
            retry: None,
            for_each: None,
            parallel: false,
        };

        workflow.add_step(step1);
        workflow
    }

    #[tokio::test]
    async fn test_engine_creation() {
        let temp_dir = env::temp_dir().join("voirs_engine_test");
        let engine = WorkflowEngine::new(temp_dir, 4);
        assert_eq!(engine.max_parallel, 4);
    }

    #[test]
    fn test_dependency_graph_building() {
        let temp_dir = env::temp_dir().join("voirs_engine_test_2");
        let engine = WorkflowEngine::new(temp_dir, 4);

        let mut workflow = Workflow::new("test", "1.0", "Test");

        let step1 = Step {
            name: "step1".to_string(),
            step_type: StepType::Command,
            description: None,
            parameters: HashMap::new(),
            condition: None,
            depends_on: Vec::new(),
            retry: None,
            for_each: None,
            parallel: false,
        };

        let step2 = Step {
            name: "step2".to_string(),
            step_type: StepType::Command,
            description: None,
            parameters: HashMap::new(),
            condition: None,
            depends_on: vec![StepDependency {
                step_name: "step1".to_string(),
                must_succeed: true,
            }],
            retry: None,
            for_each: None,
            parallel: false,
        };

        workflow.add_step(step1);
        workflow.add_step(step2);

        let graph = engine.build_dependency_graph(&workflow).unwrap();
        assert_eq!(graph.len(), 2);
        assert_eq!(graph.get("step2").unwrap().len(), 1);
    }

    #[test]
    fn test_circular_dependency_detection() {
        let temp_dir = env::temp_dir().join("voirs_engine_test_3");
        let engine = WorkflowEngine::new(temp_dir, 4);

        let mut workflow = Workflow::new("test", "1.0", "Test");

        let step1 = Step {
            name: "step1".to_string(),
            step_type: StepType::Command,
            description: None,
            parameters: HashMap::new(),
            condition: None,
            depends_on: vec![StepDependency {
                step_name: "step2".to_string(),
                must_succeed: true,
            }],
            retry: None,
            for_each: None,
            parallel: false,
        };

        let step2 = Step {
            name: "step2".to_string(),
            step_type: StepType::Command,
            description: None,
            parameters: HashMap::new(),
            condition: None,
            depends_on: vec![StepDependency {
                step_name: "step1".to_string(),
                must_succeed: true,
            }],
            retry: None,
            for_each: None,
            parallel: false,
        };

        workflow.add_step(step1);
        workflow.add_step(step2);

        let result = engine.build_dependency_graph(&workflow);
        assert!(result.is_err());
    }

    #[test]
    fn test_dependencies_satisfied() {
        let temp_dir = env::temp_dir().join("voirs_engine_test_4");
        let engine = WorkflowEngine::new(temp_dir, 4);

        let step = Step {
            name: "step2".to_string(),
            step_type: StepType::Command,
            description: None,
            parameters: HashMap::new(),
            condition: None,
            depends_on: vec![StepDependency {
                step_name: "step1".to_string(),
                must_succeed: true,
            }],
            retry: None,
            for_each: None,
            parallel: false,
        };

        let mut completed = HashSet::new();
        assert!(!engine.dependencies_satisfied(&step, &completed));

        completed.insert("step1".to_string());
        assert!(engine.dependencies_satisfied(&step, &completed));
    }
}
