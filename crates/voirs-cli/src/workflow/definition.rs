//! Workflow Definition Types
//!
//! This module defines the data structures for workflow definitions,
//! including steps, conditions, retries, and metadata.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::error::CliError;

type Result<T> = std::result::Result<T, CliError>;

/// Workflow definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    /// Workflow metadata
    pub metadata: WorkflowMetadata,
    /// Global variables
    #[serde(default)]
    pub variables: HashMap<String, Variable>,
    /// Workflow steps
    pub steps: Vec<Step>,
    /// Workflow-level configuration
    #[serde(default)]
    pub config: WorkflowConfig,
}

impl Workflow {
    /// Create a new workflow
    pub fn new(name: &str, version: &str, description: &str) -> Self {
        Self {
            metadata: WorkflowMetadata {
                name: name.to_string(),
                version: version.to_string(),
                description: description.to_string(),
                author: None,
                tags: Vec::new(),
            },
            variables: HashMap::new(),
            steps: Vec::new(),
            config: WorkflowConfig::default(),
        }
    }

    /// Load workflow from YAML file
    pub async fn load_from_file(path: &Path) -> Result<Self> {
        let content = tokio::fs::read_to_string(path).await?;

        if path.extension().is_some_and(|ext| ext == "json") {
            Ok(serde_json::from_str(&content)?)
        } else {
            // Assume YAML for .yaml, .yml, or no extension
            Ok(serde_yaml::from_str(&content).map_err(|e| {
                CliError::SerializationError(format!("Failed to parse YAML: {}", e))
            })?)
        }
    }

    /// Save workflow to file
    pub async fn save_to_file(&self, path: &Path) -> Result<()> {
        let content = if path.extension().is_some_and(|ext| ext == "json") {
            serde_json::to_string_pretty(self)?
        } else {
            serde_yaml::to_string(self).map_err(|e| {
                CliError::SerializationError(format!("Failed to serialize to YAML: {}", e))
            })?
        };

        tokio::fs::write(path, content).await?;
        Ok(())
    }

    /// Add a step to the workflow
    pub fn add_step(&mut self, step: Step) {
        self.steps.push(step);
    }

    /// Add a variable
    pub fn add_variable(&mut self, name: String, value: Variable) {
        self.variables.insert(name, value);
    }

    /// Get step by name
    pub fn get_step(&self, name: &str) -> Option<&Step> {
        self.steps.iter().find(|s| s.name == name)
    }

    /// Validate workflow structure
    pub fn validate(&self) -> Result<()> {
        // Check for duplicate step names
        let mut step_names = std::collections::HashSet::new();
        for step in &self.steps {
            if !step_names.insert(&step.name) {
                return Err(CliError::Workflow(format!(
                    "Duplicate step name: {}",
                    step.name
                )));
            }
        }

        // Check dependencies exist
        for step in &self.steps {
            for dep in &step.depends_on {
                if !self.steps.iter().any(|s| s.name == dep.step_name) {
                    return Err(CliError::Workflow(format!(
                        "Step '{}' depends on non-existent step '{}'",
                        step.name, dep.step_name
                    )));
                }
            }
        }

        Ok(())
    }
}

/// Workflow metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowMetadata {
    /// Workflow name
    pub name: String,
    /// Version string
    pub version: String,
    /// Human-readable description
    pub description: String,
    /// Optional author information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    /// Tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Workflow configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowConfig {
    /// Maximum parallel steps
    #[serde(default = "default_max_parallel")]
    pub max_parallel: usize,
    /// Timeout in seconds (0 = no timeout)
    #[serde(default)]
    pub timeout_seconds: u64,
    /// Whether to continue on error
    #[serde(default)]
    pub continue_on_error: bool,
    /// Whether to save state for resumption
    #[serde(default = "default_true")]
    pub save_state: bool,
}

fn default_max_parallel() -> usize {
    4
}

fn default_true() -> bool {
    true
}

impl Default for WorkflowConfig {
    fn default() -> Self {
        Self {
            max_parallel: default_max_parallel(),
            timeout_seconds: 0,
            continue_on_error: false,
            save_state: true,
        }
    }
}

/// Workflow step definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    /// Unique step name
    pub name: String,
    /// Step type
    #[serde(rename = "type")]
    pub step_type: StepType,
    /// Optional description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Step parameters
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
    /// Execution condition
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<Condition>,
    /// Dependencies on other steps
    #[serde(default)]
    pub depends_on: Vec<StepDependency>,
    /// Retry strategy
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetryStrategy>,
    /// For-each loop variable
    #[serde(skip_serializing_if = "Option::is_none")]
    pub for_each: Option<String>,
    /// Whether to run in parallel with other steps
    #[serde(default)]
    pub parallel: bool,
}

/// Step type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum StepType {
    /// Synthesis step
    Synthesize,
    /// Quality validation step
    Validate,
    /// File operation step
    FileOp,
    /// Command execution step
    Command,
    /// Script execution step
    Script,
    /// Conditional branch step
    Branch,
    /// Loop step
    Loop,
    /// Sub-workflow invocation
    Workflow,
    /// Wait/delay step
    Wait,
    /// Notification step
    Notify,
}

/// Step dependency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepDependency {
    /// Name of the step to depend on
    pub step_name: String,
    /// Whether the dependency must succeed
    #[serde(default = "default_true")]
    pub must_succeed: bool,
}

/// Condition definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    /// Left operand (variable or value)
    pub left: String,
    /// Operator
    pub operator: ConditionOperator,
    /// Right operand (variable or value)
    pub right: String,
}

impl Condition {
    /// Create a new condition
    pub fn new(left: String, operator: ConditionOperator, right: String) -> Self {
        Self {
            left,
            operator,
            right,
        }
    }

    /// Evaluate condition with variable substitution
    pub fn evaluate(&self, variables: &HashMap<String, serde_json::Value>) -> bool {
        let left_val = self.resolve_value(&self.left, variables);
        let right_val = self.resolve_value(&self.right, variables);

        match self.operator {
            ConditionOperator::Equals => left_val == right_val,
            ConditionOperator::NotEquals => left_val != right_val,
            ConditionOperator::GreaterThan => {
                self.compare_numeric(&left_val, &right_val, |a, b| a > b)
            }
            ConditionOperator::LessThan => {
                self.compare_numeric(&left_val, &right_val, |a, b| a < b)
            }
            ConditionOperator::GreaterOrEqual => {
                self.compare_numeric(&left_val, &right_val, |a, b| a >= b)
            }
            ConditionOperator::LessOrEqual => {
                self.compare_numeric(&left_val, &right_val, |a, b| a <= b)
            }
            ConditionOperator::Contains => {
                if let (Some(left_str), Some(right_str)) = (left_val.as_str(), right_val.as_str()) {
                    left_str.contains(right_str)
                } else {
                    false
                }
            }
            ConditionOperator::Matches => {
                // Regex match (simplified for now)
                if let (Some(left_str), Some(right_str)) = (left_val.as_str(), right_val.as_str()) {
                    regex::Regex::new(right_str)
                        .map(|re| re.is_match(left_str))
                        .unwrap_or(false)
                } else {
                    false
                }
            }
        }
    }

    fn resolve_value(
        &self,
        value: &str,
        variables: &HashMap<String, serde_json::Value>,
    ) -> serde_json::Value {
        // Check if it's a variable reference ${var}
        if let Some(var_name) = value.strip_prefix("${").and_then(|s| s.strip_suffix('}')) {
            variables
                .get(var_name)
                .cloned()
                .unwrap_or(serde_json::Value::Null)
        } else {
            // Try to parse as JSON value
            serde_json::from_str(value)
                .unwrap_or_else(|_| serde_json::Value::String(value.to_string()))
        }
    }

    fn compare_numeric<F>(&self, left: &serde_json::Value, right: &serde_json::Value, op: F) -> bool
    where
        F: Fn(f64, f64) -> bool,
    {
        match (left.as_f64(), right.as_f64()) {
            (Some(l), Some(r)) => op(l, r),
            _ => false,
        }
    }
}

/// Condition operators
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ConditionOperator {
    /// Equality
    #[serde(rename = "==")]
    Equals,
    /// Inequality
    #[serde(rename = "!=")]
    NotEquals,
    /// Greater than
    #[serde(rename = ">")]
    GreaterThan,
    /// Less than
    #[serde(rename = "<")]
    LessThan,
    /// Greater or equal
    #[serde(rename = ">=")]
    GreaterOrEqual,
    /// Less or equal
    #[serde(rename = "<=")]
    LessOrEqual,
    /// String contains
    Contains,
    /// Regex match
    Matches,
}

/// Retry strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryStrategy {
    /// Maximum retry attempts
    pub max_attempts: usize,
    /// Backoff strategy
    pub backoff: BackoffType,
    /// Initial delay in milliseconds
    #[serde(default = "default_initial_delay")]
    pub initial_delay_ms: u64,
    /// Maximum delay in milliseconds
    #[serde(default = "default_max_delay")]
    pub max_delay_ms: u64,
    /// Multiplier for exponential backoff
    #[serde(default = "default_backoff_multiplier")]
    pub backoff_multiplier: f64,
}

fn default_initial_delay() -> u64 {
    1000
}

fn default_max_delay() -> u64 {
    60_000
}

fn default_backoff_multiplier() -> f64 {
    2.0
}

impl Default for RetryStrategy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            backoff: BackoffType::Exponential,
            initial_delay_ms: default_initial_delay(),
            max_delay_ms: default_max_delay(),
            backoff_multiplier: default_backoff_multiplier(),
        }
    }
}

/// Backoff types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum BackoffType {
    /// Fixed delay
    Fixed,
    /// Linear increase
    Linear,
    /// Exponential backoff
    Exponential,
    /// Exponential with jitter
    ExponentialJitter,
}

/// Variable value types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Variable {
    /// String value
    String(String),
    /// Numeric value
    Number(f64),
    /// Boolean value
    Boolean(bool),
    /// Array of values
    Array(Vec<serde_json::Value>),
    /// Object/map
    Object(HashMap<String, serde_json::Value>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_creation() {
        let workflow = Workflow::new("test", "1.0", "Test workflow");
        assert_eq!(workflow.metadata.name, "test");
        assert_eq!(workflow.metadata.version, "1.0");
        assert_eq!(workflow.steps.len(), 0);
    }

    #[test]
    fn test_workflow_add_step() {
        let mut workflow = Workflow::new("test", "1.0", "Test workflow");

        let step = Step {
            name: "step1".to_string(),
            step_type: StepType::Synthesize,
            description: None,
            parameters: HashMap::new(),
            condition: None,
            depends_on: Vec::new(),
            retry: None,
            for_each: None,
            parallel: false,
        };

        workflow.add_step(step);
        assert_eq!(workflow.steps.len(), 1);
        assert_eq!(workflow.steps[0].name, "step1");
    }

    #[test]
    fn test_workflow_validation_duplicate_names() {
        let mut workflow = Workflow::new("test", "1.0", "Test workflow");

        let step1 = Step {
            name: "duplicate".to_string(),
            step_type: StepType::Synthesize,
            description: None,
            parameters: HashMap::new(),
            condition: None,
            depends_on: Vec::new(),
            retry: None,
            for_each: None,
            parallel: false,
        };

        let step2 = Step {
            name: "duplicate".to_string(),
            step_type: StepType::Validate,
            description: None,
            parameters: HashMap::new(),
            condition: None,
            depends_on: Vec::new(),
            retry: None,
            for_each: None,
            parallel: false,
        };

        workflow.add_step(step1);
        workflow.add_step(step2);

        assert!(workflow.validate().is_err());
    }

    #[test]
    fn test_condition_evaluation_equals() {
        let condition = Condition::new(
            "${status}".to_string(),
            ConditionOperator::Equals,
            "success".to_string(),
        );

        let mut variables = HashMap::new();
        variables.insert(
            "status".to_string(),
            serde_json::Value::String("success".to_string()),
        );

        assert!(condition.evaluate(&variables));
    }

    #[test]
    fn test_condition_evaluation_greater_than() {
        let condition = Condition::new(
            "${score}".to_string(),
            ConditionOperator::GreaterThan,
            "4.0".to_string(),
        );

        let mut variables = HashMap::new();
        variables.insert("score".to_string(), serde_json::json!(4.5));

        assert!(condition.evaluate(&variables));
    }

    #[test]
    fn test_condition_evaluation_contains() {
        let condition = Condition::new(
            "${output}".to_string(),
            ConditionOperator::Contains,
            "error".to_string(),
        );

        let mut variables = HashMap::new();
        variables.insert(
            "output".to_string(),
            serde_json::Value::String("An error occurred".to_string()),
        );

        assert!(condition.evaluate(&variables));
    }

    #[test]
    fn test_retry_strategy_defaults() {
        let retry = RetryStrategy::default();
        assert_eq!(retry.max_attempts, 3);
        assert_eq!(retry.backoff, BackoffType::Exponential);
        assert_eq!(retry.initial_delay_ms, 1000);
    }

    #[test]
    fn test_workflow_config_defaults() {
        let config = WorkflowConfig::default();
        assert_eq!(config.max_parallel, 4);
        assert_eq!(config.timeout_seconds, 0);
        assert!(!config.continue_on_error);
        assert!(config.save_state);
    }
}
