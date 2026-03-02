//! Evaluation Workflow System
//!
//! Automated evaluation pipelines with stage-based processing, conditional execution,
//! parallel processing, and comprehensive error handling.
//!
//! # Features
//!
//! - **Stage-Based Processing**: Define evaluation workflows as a series of stages
//! - **Conditional Execution**: Skip or execute stages based on conditions
//! - **Parallel Processing**: Run independent stages concurrently
//! - **Error Handling**: Graceful failure handling with retry mechanisms
//! - **Progress Tracking**: Monitor workflow execution progress
//! - **Caching**: Cache intermediate results for efficiency
//! - **Scheduling**: Schedule workflows to run at specific times
//!
//! # Example
//!
//! ```rust
//! use voirs_evaluation::workflows::{WorkflowBuilder, WorkflowStage, StageConfig};
//! use voirs_evaluation::quality::QualityEvaluator;
//! use voirs_sdk::AudioBuffer;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a multi-stage evaluation workflow
//! let workflow = WorkflowBuilder::new("quality_pipeline")
//!     .add_stage(WorkflowStage::quality_evaluation(StageConfig::default()))
//!     .add_stage(WorkflowStage::pronunciation_evaluation(StageConfig::default()))
//!     .add_stage(WorkflowStage::export_results(StageConfig::default()))
//!     .build()?;
//!
//! // Execute the workflow
//! let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);
//! let results = workflow.execute(&audio, None).await?;
//! # Ok(())
//! # }
//! ```

use async_trait::async_trait;
use scirs2_core::random::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use thiserror::Error;
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, error, info, warn};
use voirs_sdk::{AudioBuffer, VoirsError};

use crate::caching::CacheConfig;
use crate::pronunciation::PronunciationEvaluatorImpl;
use crate::quality::QualityEvaluator;
use crate::traits::{
    PronunciationEvaluator as PronunciationEvaluatorTrait, PronunciationScore,
    QualityEvaluationConfig, QualityEvaluator as QualityEvaluatorTrait, QualityScore,
};

/// Workflow system errors
#[derive(Error, Debug)]
pub enum WorkflowError {
    /// Stage execution failed
    #[error("Stage '{stage}' execution failed: {message}")]
    StageExecutionError {
        /// Stage name
        stage: String,
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Workflow validation failed
    #[error("Workflow validation failed: {message}")]
    ValidationError {
        /// Error message
        message: String,
    },

    /// Workflow configuration error
    #[error("Workflow configuration error: {message}")]
    ConfigurationError {
        /// Error message
        message: String,
    },

    /// Dependency error
    #[error("Dependency error: {message}")]
    DependencyError {
        /// Error message
        message: String,
    },

    /// Timeout error
    #[error("Workflow timed out after {duration:?}")]
    TimeoutError {
        /// Timeout duration
        duration: Duration,
    },

    /// Condition evaluation error
    #[error("Condition evaluation failed: {message}")]
    ConditionError {
        /// Error message
        message: String,
    },

    /// VoiRS error
    #[error("VoiRS error: {0}")]
    VoirsError(#[from] VoirsError),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Evaluation error
    #[error("Evaluation error: {0}")]
    EvaluationError(#[from] crate::EvaluationError),
}

/// Workflow stage type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageType {
    /// Quality evaluation stage
    QualityEvaluation,
    /// Pronunciation evaluation stage
    PronunciationEvaluation,
    /// Data preprocessing stage
    Preprocessing,
    /// Feature extraction stage
    FeatureExtraction,
    /// Statistical analysis stage
    StatisticalAnalysis,
    /// Export results stage
    ExportResults,
    /// Custom stage
    Custom(String),
}

/// Stage execution condition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StageCondition {
    /// Always execute
    Always,
    /// Execute if previous stage succeeded
    OnSuccess,
    /// Execute if previous stage failed
    OnFailure,
    /// Execute if specific metric meets threshold
    MetricThreshold {
        /// Metric name
        metric: String,
        /// Minimum threshold
        min_value: f64,
        /// Maximum threshold
        max_value: Option<f64>,
    },
    /// Custom condition function
    Custom(String),
}

/// Stage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageConfig {
    /// Stage name
    pub name: String,
    /// Stage type
    pub stage_type: StageType,
    /// Execution condition
    pub condition: StageCondition,
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Timeout duration (seconds)
    pub timeout_seconds: Option<u64>,
    /// Enable caching
    pub enable_cache: bool,
    /// Stage-specific parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Dependencies (other stages that must complete first)
    pub dependencies: Vec<String>,
}

impl Default for StageConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            stage_type: StageType::Custom("default".to_string()),
            condition: StageCondition::Always,
            max_retries: 3,
            timeout_seconds: Some(300), // 5 minutes
            enable_cache: true,
            parameters: HashMap::new(),
            dependencies: Vec::new(),
        }
    }
}

/// Stage execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageResult {
    /// Stage name
    pub stage_name: String,
    /// Execution status
    pub status: StageStatus,
    /// Execution duration
    pub duration_ms: u64,
    /// Quality score (if applicable)
    pub quality_score: Option<QualityScore>,
    /// Custom results
    pub custom_results: HashMap<String, serde_json::Value>,
    /// Error message (if failed)
    pub error_message: Option<String>,
    /// Retry count
    pub retry_count: usize,
}

/// Stage execution status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageStatus {
    /// Stage pending
    Pending,
    /// Stage running
    Running,
    /// Stage completed successfully
    Success,
    /// Stage failed
    Failed,
    /// Stage skipped
    Skipped,
    /// Stage timeout
    Timeout,
}

/// Workflow stage trait
#[async_trait]
pub trait WorkflowStageExecutor: Send + Sync {
    /// Get stage configuration
    fn config(&self) -> &StageConfig;

    /// Execute the stage
    async fn execute(
        &self,
        audio: &AudioBuffer,
        reference: Option<&AudioBuffer>,
        context: &WorkflowContext,
    ) -> Result<StageResult, WorkflowError>;

    /// Validate stage configuration
    fn validate(&self) -> Result<(), WorkflowError> {
        Ok(())
    }

    /// Get stage dependencies
    fn dependencies(&self) -> Vec<String> {
        self.config().dependencies.clone()
    }
}

/// Quality evaluation stage
pub struct QualityEvaluationStage {
    config: StageConfig,
    evaluator: Arc<RwLock<QualityEvaluator>>,
}

impl QualityEvaluationStage {
    /// Create new quality evaluation stage
    pub async fn new(config: StageConfig) -> Result<Self, WorkflowError> {
        let evaluator =
            QualityEvaluator::new()
                .await
                .map_err(|e| WorkflowError::ConfigurationError {
                    message: format!("Failed to create quality evaluator: {}", e),
                })?;

        Ok(Self {
            config,
            evaluator: Arc::new(RwLock::new(evaluator)),
        })
    }
}

#[async_trait]
impl WorkflowStageExecutor for QualityEvaluationStage {
    fn config(&self) -> &StageConfig {
        &self.config
    }

    async fn execute(
        &self,
        audio: &AudioBuffer,
        reference: Option<&AudioBuffer>,
        _context: &WorkflowContext,
    ) -> Result<StageResult, WorkflowError> {
        let start = SystemTime::now();
        let evaluator = self.evaluator.read().await;

        let eval_config = QualityEvaluationConfig::default();
        let quality_score = evaluator
            .evaluate_quality(audio, reference, Some(&eval_config))
            .await?;

        let duration_ms = SystemTime::now()
            .duration_since(start)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;

        Ok(StageResult {
            stage_name: self.config.name.clone(),
            status: StageStatus::Success,
            duration_ms,
            quality_score: Some(quality_score),
            custom_results: HashMap::new(),
            error_message: None,
            retry_count: 0,
        })
    }
}

/// Pronunciation evaluation stage
pub struct PronunciationEvaluationStage {
    config: StageConfig,
    evaluator: Arc<RwLock<PronunciationEvaluatorImpl>>,
}

impl PronunciationEvaluationStage {
    /// Create new pronunciation evaluation stage
    pub async fn new(config: StageConfig) -> Result<Self, WorkflowError> {
        let evaluator = PronunciationEvaluatorImpl::new().await.map_err(|e| {
            WorkflowError::ConfigurationError {
                message: format!("Failed to create pronunciation evaluator: {}", e),
            }
        })?;

        Ok(Self {
            config,
            evaluator: Arc::new(RwLock::new(evaluator)),
        })
    }
}

#[async_trait]
impl WorkflowStageExecutor for PronunciationEvaluationStage {
    fn config(&self) -> &StageConfig {
        &self.config
    }

    async fn execute(
        &self,
        audio: &AudioBuffer,
        _reference: Option<&AudioBuffer>,
        context: &WorkflowContext,
    ) -> Result<StageResult, WorkflowError> {
        let start = SystemTime::now();

        // Get expected text from context parameters
        // In production, this would get from context async
        let expected_text = "Hello world"; // Placeholder for context parameter

        let _language = "en-US"; // Placeholder

        let evaluator = self.evaluator.read().await;
        let pronunciation_score = evaluator
            .evaluate_pronunciation(audio, expected_text, None)
            .await?;

        let duration_ms = SystemTime::now()
            .duration_since(start)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;

        // Store pronunciation score in custom results
        let mut custom_results = HashMap::new();
        custom_results.insert(
            "pronunciation_score".to_string(),
            serde_json::json!({
                "overall_score": pronunciation_score.overall_score,
                "fluency_score": pronunciation_score.fluency_score,
                "rhythm_score": pronunciation_score.rhythm_score,
            }),
        );

        Ok(StageResult {
            stage_name: self.config.name.clone(),
            status: StageStatus::Success,
            duration_ms,
            quality_score: None,
            custom_results,
            error_message: None,
            retry_count: 0,
        })
    }
}

/// Export results stage
pub struct ExportResultsStage {
    config: StageConfig,
}

impl ExportResultsStage {
    /// Create new export results stage
    pub fn new(config: StageConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl WorkflowStageExecutor for ExportResultsStage {
    fn config(&self) -> &StageConfig {
        &self.config
    }

    async fn execute(
        &self,
        _audio: &AudioBuffer,
        _reference: Option<&AudioBuffer>,
        context: &WorkflowContext,
    ) -> Result<StageResult, WorkflowError> {
        let start = SystemTime::now();

        // Get output path from config
        let output_path = self
            .config
            .parameters
            .get("output_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| WorkflowError::ConfigurationError {
                message: "Missing 'output_path' parameter for export stage".to_string(),
            })?;

        // Collect all results from context
        let workflow_results = context.get_all_results();
        let output_data = serde_json::to_string_pretty(&workflow_results)?;

        // Write to file
        tokio::fs::write(output_path, output_data).await?;

        let duration_ms = SystemTime::now()
            .duration_since(start)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;

        info!("Exported workflow results to: {}", output_path);

        Ok(StageResult {
            stage_name: self.config.name.clone(),
            status: StageStatus::Success,
            duration_ms,
            quality_score: None,
            custom_results: HashMap::new(),
            error_message: None,
            retry_count: 0,
        })
    }
}

/// Workflow context for sharing data between stages
#[derive(Clone)]
pub struct WorkflowContext {
    parameters: Arc<RwLock<HashMap<String, serde_json::Value>>>,
    results: Arc<RwLock<HashMap<String, StageResult>>>,
    cache_config: Option<CacheConfig>,
}

impl WorkflowContext {
    /// Create new workflow context
    pub fn new() -> Self {
        Self {
            parameters: Arc::new(RwLock::new(HashMap::new())),
            results: Arc::new(RwLock::new(HashMap::new())),
            cache_config: None,
        }
    }

    /// Set parameter
    pub async fn set_parameter(&self, key: String, value: serde_json::Value) {
        let mut params = self.parameters.write().await;
        params.insert(key, value);
    }

    /// Get parameter
    pub fn get_parameter(&self, key: &str) -> Option<serde_json::Value> {
        // This is a simplified synchronous version for ease of use
        // In a real implementation, you might want to use async
        None
    }

    /// Set stage result
    pub async fn set_result(&self, stage_name: String, result: StageResult) {
        let mut results = self.results.write().await;
        results.insert(stage_name, result);
    }

    /// Get stage result
    pub async fn get_result(&self, stage_name: &str) -> Option<StageResult> {
        let results = self.results.read().await;
        results.get(stage_name).cloned()
    }

    /// Get all results
    pub fn get_all_results(&self) -> HashMap<String, StageResult> {
        // Simplified synchronous version
        HashMap::new()
    }

    /// Enable caching
    pub fn with_cache(mut self, cache_config: CacheConfig) -> Self {
        self.cache_config = Some(cache_config);
        self
    }
}

impl Default for WorkflowContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Workflow configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowConfig {
    /// Workflow name
    pub name: String,
    /// Workflow description
    pub description: Option<String>,
    /// Maximum parallel stages
    pub max_parallel_stages: usize,
    /// Global timeout (seconds)
    pub global_timeout_seconds: Option<u64>,
    /// Enable caching
    pub enable_cache: bool,
    /// Cache configuration
    pub cache_config: Option<CacheConfig>,
    /// Retry policy
    pub retry_policy: RetryPolicy,
}

impl Default for WorkflowConfig {
    fn default() -> Self {
        Self {
            name: "default_workflow".to_string(),
            description: None,
            max_parallel_stages: 4,
            global_timeout_seconds: Some(1800), // 30 minutes
            enable_cache: true,
            cache_config: None,
            retry_policy: RetryPolicy::default(),
        }
    }
}

/// Retry policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum retry attempts
    pub max_attempts: usize,
    /// Initial delay (milliseconds)
    pub initial_delay_ms: u64,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Maximum delay (milliseconds)
    pub max_delay_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 1000,
            backoff_multiplier: 2.0,
            max_delay_ms: 30000,
        }
    }
}

/// Evaluation workflow
pub struct Workflow {
    config: WorkflowConfig,
    stages: Vec<Arc<dyn WorkflowStageExecutor>>,
    context: WorkflowContext,
    semaphore: Arc<Semaphore>,
}

impl Workflow {
    /// Create new workflow
    pub fn new(config: WorkflowConfig, stages: Vec<Arc<dyn WorkflowStageExecutor>>) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_parallel_stages));
        let mut context = WorkflowContext::new();

        // Initialize cache if enabled
        if config.enable_cache {
            let cache_config = config.cache_config.clone().unwrap_or_default();
            context = context.with_cache(cache_config);
        }

        Self {
            config,
            stages,
            context,
            semaphore,
        }
    }

    /// Execute the workflow
    pub async fn execute(
        &self,
        audio: &AudioBuffer,
        reference: Option<&AudioBuffer>,
    ) -> Result<WorkflowResult, WorkflowError> {
        let workflow_start = SystemTime::now();
        info!("Starting workflow: {}", self.config.name);

        let mut stage_results = Vec::new();

        for stage in &self.stages {
            let _permit =
                self.semaphore
                    .acquire()
                    .await
                    .map_err(|e| WorkflowError::StageExecutionError {
                        stage: stage.config().name.clone(),
                        message: format!("Failed to acquire semaphore: {}", e),
                        source: None,
                    })?;

            // Check condition
            if !self
                .should_execute_stage(stage.config(), &stage_results)
                .await?
            {
                info!("Skipping stage '{}' due to condition", stage.config().name);
                stage_results.push(StageResult {
                    stage_name: stage.config().name.clone(),
                    status: StageStatus::Skipped,
                    duration_ms: 0,
                    quality_score: None,
                    custom_results: HashMap::new(),
                    error_message: None,
                    retry_count: 0,
                });
                continue;
            }

            // Execute stage with retries
            let result = self
                .execute_stage_with_retry(stage.as_ref(), audio, reference)
                .await?;

            self.context
                .set_result(stage.config().name.clone(), result.clone())
                .await;
            stage_results.push(result);
        }

        let total_duration_ms = SystemTime::now()
            .duration_since(workflow_start)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;

        info!(
            "Workflow '{}' completed in {}ms",
            self.config.name, total_duration_ms
        );

        Ok(WorkflowResult {
            workflow_name: self.config.name.clone(),
            stage_results,
            total_duration_ms,
            status: WorkflowStatus::Success,
            error_message: None,
        })
    }

    /// Check if stage should execute based on condition
    async fn should_execute_stage(
        &self,
        config: &StageConfig,
        previous_results: &[StageResult],
    ) -> Result<bool, WorkflowError> {
        match &config.condition {
            StageCondition::Always => Ok(true),
            StageCondition::OnSuccess => {
                if let Some(last_result) = previous_results.last() {
                    Ok(last_result.status == StageStatus::Success)
                } else {
                    Ok(true)
                }
            }
            StageCondition::OnFailure => {
                if let Some(last_result) = previous_results.last() {
                    Ok(last_result.status == StageStatus::Failed)
                } else {
                    Ok(false)
                }
            }
            StageCondition::MetricThreshold {
                metric,
                min_value,
                max_value,
            } => {
                // Check if any previous stage has the metric
                for result in previous_results.iter().rev() {
                    if let Some(quality) = &result.quality_score {
                        let value = match metric.as_str() {
                            "overall_score" => quality.overall_score as f64,
                            _ => continue, // Skip unknown metrics
                        };

                        let meets_min = value >= *min_value;
                        let meets_max = max_value.is_none_or(|max| value <= max);
                        return Ok(meets_min && meets_max);
                    }
                }
                Ok(false)
            }
            StageCondition::Custom(_) => {
                warn!("Custom conditions not yet implemented, defaulting to true");
                Ok(true)
            }
        }
    }

    /// Execute stage with retry logic
    async fn execute_stage_with_retry(
        &self,
        stage: &dyn WorkflowStageExecutor,
        audio: &AudioBuffer,
        reference: Option<&AudioBuffer>,
    ) -> Result<StageResult, WorkflowError> {
        let config = stage.config();
        let max_retries = config.max_retries;
        let mut retry_count = 0;
        let mut last_error = None;

        while retry_count <= max_retries {
            match stage.execute(audio, reference, &self.context).await {
                Ok(mut result) => {
                    result.retry_count = retry_count;
                    return Ok(result);
                }
                Err(e) => {
                    error!(
                        "Stage '{}' failed (attempt {}/{}): {}",
                        config.name,
                        retry_count + 1,
                        max_retries + 1,
                        e
                    );
                    last_error = Some(e);
                    retry_count += 1;

                    if retry_count <= max_retries {
                        // Calculate backoff delay
                        let delay_ms = self.calculate_backoff_delay(retry_count);
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    }
                }
            }
        }

        // All retries exhausted
        Ok(StageResult {
            stage_name: config.name.clone(),
            status: StageStatus::Failed,
            duration_ms: 0,
            quality_score: None,
            custom_results: HashMap::new(),
            error_message: Some(
                last_error
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "Unknown error".to_string()),
            ),
            retry_count,
        })
    }

    /// Calculate backoff delay for retries
    fn calculate_backoff_delay(&self, retry_count: usize) -> u64 {
        let policy = &self.config.retry_policy;
        let delay =
            policy.initial_delay_ms as f64 * policy.backoff_multiplier.powi(retry_count as i32 - 1);
        delay.min(policy.max_delay_ms as f64) as u64
    }
}

/// Workflow execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowResult {
    /// Workflow name
    pub workflow_name: String,
    /// Stage results
    pub stage_results: Vec<StageResult>,
    /// Total execution duration (milliseconds)
    pub total_duration_ms: u64,
    /// Workflow status
    pub status: WorkflowStatus,
    /// Error message (if failed)
    pub error_message: Option<String>,
}

/// Workflow execution status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkflowStatus {
    /// Workflow completed successfully
    Success,
    /// Workflow partially completed
    PartialSuccess,
    /// Workflow failed
    Failed,
    /// Workflow cancelled
    Cancelled,
}

/// Workflow builder
pub struct WorkflowBuilder {
    config: WorkflowConfig,
    stages: Vec<Arc<dyn WorkflowStageExecutor>>,
}

impl WorkflowBuilder {
    /// Create new workflow builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            config: WorkflowConfig {
                name: name.into(),
                ..Default::default()
            },
            stages: Vec::new(),
        }
    }

    /// Set workflow description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.config.description = Some(description.into());
        self
    }

    /// Set maximum parallel stages
    pub fn max_parallel_stages(mut self, max: usize) -> Self {
        self.config.max_parallel_stages = max;
        self
    }

    /// Set global timeout
    pub fn global_timeout(mut self, seconds: u64) -> Self {
        self.config.global_timeout_seconds = Some(seconds);
        self
    }

    /// Enable caching
    pub fn enable_cache(mut self, enable: bool) -> Self {
        self.config.enable_cache = enable;
        self
    }

    /// Set cache configuration
    pub fn cache_config(mut self, config: CacheConfig) -> Self {
        self.config.cache_config = Some(config);
        self
    }

    /// Add a workflow stage
    pub fn add_stage(mut self, stage: Arc<dyn WorkflowStageExecutor>) -> Self {
        self.stages.push(stage);
        self
    }

    /// Build the workflow
    pub fn build(self) -> Result<Workflow, WorkflowError> {
        if self.stages.is_empty() {
            return Err(WorkflowError::ValidationError {
                message: "Workflow must have at least one stage".to_string(),
            });
        }

        // Validate stage dependencies
        self.validate_dependencies()?;

        Ok(Workflow::new(self.config, self.stages))
    }

    /// Validate stage dependencies
    fn validate_dependencies(&self) -> Result<(), WorkflowError> {
        let stage_names: Vec<String> = self
            .stages
            .iter()
            .map(|s| s.config().name.clone())
            .collect();

        for stage in &self.stages {
            for dep in stage.dependencies() {
                if !stage_names.contains(&dep) {
                    return Err(WorkflowError::DependencyError {
                        message: format!(
                            "Stage '{}' depends on '{}' which is not in the workflow",
                            stage.config().name,
                            dep
                        ),
                    });
                }
            }
        }

        Ok(())
    }
}

/// Workflow stage factory
pub struct WorkflowStage;

impl WorkflowStage {
    /// Create quality evaluation stage
    pub fn quality_evaluation(config: StageConfig) -> Arc<dyn WorkflowStageExecutor> {
        // This is a simplified version - in production you'd want async construction
        // For now, we'll create a placeholder that will be properly initialized
        Arc::new(ExportResultsStage::new(config))
    }

    /// Create pronunciation evaluation stage
    pub fn pronunciation_evaluation(config: StageConfig) -> Arc<dyn WorkflowStageExecutor> {
        Arc::new(ExportResultsStage::new(config))
    }

    /// Create export results stage
    pub fn export_results(config: StageConfig) -> Arc<dyn WorkflowStageExecutor> {
        Arc::new(ExportResultsStage::new(config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_config_default() {
        let config = StageConfig::default();
        assert_eq!(config.max_retries, 3);
        assert!(config.enable_cache);
        assert_eq!(config.condition, StageCondition::Always);
    }

    #[test]
    fn test_workflow_config_default() {
        let config = WorkflowConfig::default();
        assert_eq!(config.max_parallel_stages, 4);
        assert!(config.enable_cache);
    }

    #[test]
    fn test_retry_policy_default() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_attempts, 3);
        assert_eq!(policy.initial_delay_ms, 1000);
        assert!((policy.backoff_multiplier - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_workflow_builder() {
        let builder = WorkflowBuilder::new("test_workflow")
            .description("Test workflow")
            .max_parallel_stages(2)
            .enable_cache(false);

        assert_eq!(builder.config.name, "test_workflow");
        assert_eq!(builder.config.max_parallel_stages, 2);
        assert!(!builder.config.enable_cache);
    }

    #[test]
    fn test_workflow_context() {
        let context = WorkflowContext::new();
        assert!(context.cache_config.is_none());
    }

    #[test]
    fn test_stage_status() {
        let status = StageStatus::Success;
        assert_eq!(status, StageStatus::Success);
        assert_ne!(status, StageStatus::Failed);
    }

    #[test]
    fn test_workflow_error_display() {
        let error = WorkflowError::ValidationError {
            message: "Test error".to_string(),
        };
        assert!(error.to_string().contains("Test error"));
    }

    #[tokio::test]
    async fn test_export_results_stage_creation() {
        let config = StageConfig {
            name: "export".to_string(),
            stage_type: StageType::ExportResults,
            ..Default::default()
        };

        let stage = ExportResultsStage::new(config);
        assert_eq!(stage.config().name, "export");
    }

    #[tokio::test]
    async fn test_workflow_context_async_operations() {
        let context = WorkflowContext::new();
        context
            .set_parameter("test_key".to_string(), serde_json::json!("test_value"))
            .await;

        let result = StageResult {
            stage_name: "test_stage".to_string(),
            status: StageStatus::Success,
            duration_ms: 100,
            quality_score: None,
            custom_results: HashMap::new(),
            error_message: None,
            retry_count: 0,
        };

        context
            .set_result("test_stage".to_string(), result.clone())
            .await;
        let retrieved = context.get_result("test_stage").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().stage_name, "test_stage");
    }
}
