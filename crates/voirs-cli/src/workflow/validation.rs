//! Workflow Validation
//!
//! Validates workflow definitions for correctness and safety.

use super::definition::Workflow;
use crate::error::CliError;

type Result<T> = std::result::Result<T, CliError>;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Validation error types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationError {
    /// Duplicate step name
    DuplicateStepName(String),
    /// Missing dependency
    MissingDependency { step: String, dependency: String },
    /// Circular dependency
    CircularDependency(Vec<String>),
    /// Invalid condition
    InvalidCondition { step: String, reason: String },
    /// Invalid parameter
    InvalidParameter {
        step: String,
        parameter: String,
        reason: String,
    },
    /// Empty workflow
    EmptyWorkflow,
    /// Invalid for-each variable
    InvalidForEachVariable { step: String, variable: String },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateStepName(name) => write!(f, "Duplicate step name: {}", name),
            Self::MissingDependency { step, dependency } => {
                write!(
                    f,
                    "Step '{}' depends on non-existent step '{}'",
                    step, dependency
                )
            }
            Self::CircularDependency(cycle) => {
                write!(f, "Circular dependency detected: {}", cycle.join(" -> "))
            }
            Self::InvalidCondition { step, reason } => {
                write!(f, "Invalid condition in step '{}': {}", step, reason)
            }
            Self::InvalidParameter {
                step,
                parameter,
                reason,
            } => write!(
                f,
                "Invalid parameter '{}' in step '{}': {}",
                parameter, step, reason
            ),
            Self::EmptyWorkflow => write!(f, "Workflow has no steps"),
            Self::InvalidForEachVariable { step, variable } => {
                write!(
                    f,
                    "Invalid for-each variable '{}' in step '{}'",
                    variable, step
                )
            }
        }
    }
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether validation passed
    pub valid: bool,
    /// List of errors
    pub errors: Vec<ValidationError>,
    /// List of warnings
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Create successful validation result
    pub fn success() -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Create failed validation result
    pub fn failure(errors: Vec<ValidationError>) -> Self {
        Self {
            valid: false,
            errors,
            warnings: Vec::new(),
        }
    }

    /// Add a warning
    pub fn with_warning(mut self, warning: String) -> Self {
        self.warnings.push(warning);
        self
    }

    /// Check if has errors
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Check if has warnings
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

/// Workflow validator
pub struct WorkflowValidator {
    /// Maximum allowed steps
    max_steps: usize,
    /// Maximum dependency depth
    max_dependency_depth: usize,
}

impl WorkflowValidator {
    /// Create new validator with defaults
    pub fn new() -> Self {
        Self {
            max_steps: 1000,
            max_dependency_depth: 100,
        }
    }

    /// Create validator with custom limits
    pub fn with_limits(max_steps: usize, max_dependency_depth: usize) -> Self {
        Self {
            max_steps,
            max_dependency_depth,
        }
    }

    /// Validate a workflow
    pub fn validate(&self, workflow: &Workflow) -> Result<ValidationResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check if workflow is empty
        if workflow.steps.is_empty() {
            errors.push(ValidationError::EmptyWorkflow);
            return Ok(ValidationResult::failure(errors));
        }

        // Check step count
        if workflow.steps.len() > self.max_steps {
            warnings.push(format!(
                "Workflow has {} steps, which exceeds recommended limit of {}",
                workflow.steps.len(),
                self.max_steps
            ));
        }

        // Check for duplicate step names
        let mut step_names = HashSet::new();
        for step in &workflow.steps {
            if !step_names.insert(&step.name) {
                errors.push(ValidationError::DuplicateStepName(step.name.clone()));
            }
        }

        // Build step map for dependency checking
        let step_map: HashMap<&String, &super::definition::Step> =
            workflow.steps.iter().map(|s| (&s.name, s)).collect();

        // Check dependencies exist
        for step in &workflow.steps {
            for dep in &step.depends_on {
                if !step_map.contains_key(&dep.step_name) {
                    errors.push(ValidationError::MissingDependency {
                        step: step.name.clone(),
                        dependency: dep.step_name.clone(),
                    });
                }
            }
        }

        // Check for circular dependencies
        if let Some(cycle) = self.detect_cycles(workflow) {
            errors.push(ValidationError::CircularDependency(cycle));
        }

        // Check dependency depth
        for step in &workflow.steps {
            let depth = Self::calculate_dependency_depth(&step.name, workflow, &mut HashSet::new());
            if depth > self.max_dependency_depth {
                warnings.push(format!(
                    "Step '{}' has dependency depth of {}, which exceeds recommended limit of {}",
                    step.name, depth, self.max_dependency_depth
                ));
            }
        }

        // Validate conditions
        for step in &workflow.steps {
            if let Some(ref condition) = step.condition {
                // Basic validation of condition syntax
                if condition.left.is_empty() || condition.right.is_empty() {
                    errors.push(ValidationError::InvalidCondition {
                        step: step.name.clone(),
                        reason: "Condition operands cannot be empty".to_string(),
                    });
                }
            }
        }

        // Validate for-each loops
        for step in &workflow.steps {
            if let Some(ref for_each_var) = step.for_each {
                // Check if variable is properly formatted
                if !for_each_var.starts_with("${") || !for_each_var.ends_with('}') {
                    errors.push(ValidationError::InvalidForEachVariable {
                        step: step.name.clone(),
                        variable: for_each_var.clone(),
                    });
                }
            }
        }

        if errors.is_empty() {
            let mut result = ValidationResult::success();
            result.warnings = warnings;
            Ok(result)
        } else {
            let mut result = ValidationResult::failure(errors);
            result.warnings = warnings;
            Ok(result)
        }
    }

    /// Detect circular dependencies
    fn detect_cycles(&self, workflow: &Workflow) -> Option<Vec<String>> {
        let mut visited = HashSet::new();
        let mut recursion_stack = Vec::new();

        for step in &workflow.steps {
            if Self::has_cycle_dfs(&step.name, workflow, &mut visited, &mut recursion_stack) {
                return Some(recursion_stack);
            }
        }

        None
    }

    /// DFS-based cycle detection
    fn has_cycle_dfs(
        node: &str,
        workflow: &Workflow,
        visited: &mut HashSet<String>,
        recursion_stack: &mut Vec<String>,
    ) -> bool {
        if recursion_stack.iter().any(|s| s == node) {
            recursion_stack.push(node.to_string());
            return true;
        }

        if visited.contains(node) {
            return false;
        }

        visited.insert(node.to_string());
        recursion_stack.push(node.to_string());

        // Find step and check dependencies
        if let Some(step) = workflow.steps.iter().find(|s| s.name == node) {
            for dep in &step.depends_on {
                if Self::has_cycle_dfs(&dep.step_name, workflow, visited, recursion_stack) {
                    return true;
                }
            }
        }

        recursion_stack.pop();
        false
    }

    /// Calculate dependency depth
    fn calculate_dependency_depth(
        step_name: &str,
        workflow: &Workflow,
        visited: &mut HashSet<String>,
    ) -> usize {
        if visited.contains(step_name) {
            return 0;
        }

        visited.insert(step_name.to_string());

        let step = workflow.steps.iter().find(|s| s.name == step_name);

        if let Some(step) = step {
            if step.depends_on.is_empty() {
                return 1;
            }

            let max_dep_depth = step
                .depends_on
                .iter()
                .map(|dep| Self::calculate_dependency_depth(&dep.step_name, workflow, visited))
                .max()
                .unwrap_or(0);

            max_dep_depth + 1
        } else {
            0
        }
    }
}

impl Default for WorkflowValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::definition::{Step, StepDependency, StepType};
    use std::collections::HashMap;

    #[test]
    fn test_validator_creation() {
        let validator = WorkflowValidator::new();
        assert_eq!(validator.max_steps, 1000);
    }

    #[test]
    fn test_validate_empty_workflow() {
        let validator = WorkflowValidator::new();
        let workflow = Workflow::new("test", "1.0", "Test");

        let result = validator.validate(&workflow).unwrap();
        assert!(!result.valid);
        assert!(result.has_errors());
    }

    #[test]
    fn test_validate_valid_workflow() {
        let validator = WorkflowValidator::new();
        let mut workflow = Workflow::new("test", "1.0", "Test");

        let step = Step {
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

        workflow.add_step(step);

        let result = validator.validate(&workflow).unwrap();
        assert!(result.valid);
        assert!(!result.has_errors());
    }

    #[test]
    fn test_validate_duplicate_step_names() {
        let validator = WorkflowValidator::new();
        let mut workflow = Workflow::new("test", "1.0", "Test");

        let step1 = Step {
            name: "duplicate".to_string(),
            step_type: StepType::Command,
            description: None,
            parameters: HashMap::new(),
            condition: None,
            depends_on: Vec::new(),
            retry: None,
            for_each: None,
            parallel: false,
        };

        workflow.add_step(step1.clone());
        workflow.add_step(step1);

        let result = validator.validate(&workflow).unwrap();
        assert!(!result.valid);
        assert!(result.has_errors());
    }

    #[test]
    fn test_validate_missing_dependency() {
        let validator = WorkflowValidator::new();
        let mut workflow = Workflow::new("test", "1.0", "Test");

        let step = Step {
            name: "step1".to_string(),
            step_type: StepType::Command,
            description: None,
            parameters: HashMap::new(),
            condition: None,
            depends_on: vec![StepDependency {
                step_name: "nonexistent".to_string(),
                must_succeed: true,
            }],
            retry: None,
            for_each: None,
            parallel: false,
        };

        workflow.add_step(step);

        let result = validator.validate(&workflow).unwrap();
        assert!(!result.valid);
        assert!(result.has_errors());
    }

    #[test]
    fn test_validation_result_success() {
        let result = ValidationResult::success();
        assert!(result.valid);
        assert!(!result.has_errors());
    }

    #[test]
    fn test_validation_result_with_warning() {
        let result = ValidationResult::success().with_warning("Test warning".to_string());
        assert!(result.valid);
        assert!(result.has_warnings());
        assert_eq!(result.warnings.len(), 1);
    }
}
