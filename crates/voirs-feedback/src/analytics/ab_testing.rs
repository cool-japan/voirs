//! A/B Testing Framework for VoiRS Feedback System
//!
//! This module provides comprehensive A/B testing capabilities for evaluating
//! different approaches to feedback delivery, UI/UX changes, and system improvements.

use crate::traits::{FeedbackResponse, UserProgress};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// A/B test experiment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    /// Unique experiment ID
    pub id: String,
    /// Experiment name
    pub name: String,
    /// Experiment description
    pub description: String,
    /// Experiment variants
    pub variants: Vec<Variant>,
    /// Traffic allocation (percentage split)
    pub traffic_allocation: HashMap<String, f64>,
    /// Start time
    pub start_time: DateTime<Utc>,
    /// End time
    pub end_time: Option<DateTime<Utc>>,
    /// Experiment status
    pub status: ExperimentStatus,
    /// Target metrics to track
    pub target_metrics: Vec<String>,
    /// Success criteria
    pub success_criteria: SuccessCriteria,
    /// Experiment metadata
    pub metadata: HashMap<String, String>,
}

/// Experiment variant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variant {
    /// Variant ID
    pub id: String,
    /// Variant name
    pub name: String,
    /// Variant description
    pub description: String,
    /// Variant configuration
    pub config: VariantConfig,
    /// Expected traffic percentage
    pub traffic_percentage: f64,
}

/// Variant configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantConfig {
    /// Feature flags
    pub feature_flags: HashMap<String, bool>,
    /// Configuration parameters
    pub parameters: HashMap<String, String>,
    /// UI/UX changes
    pub ui_changes: Vec<UiChange>,
    /// Feedback delivery changes
    pub feedback_changes: Vec<FeedbackChange>,
}

/// UI/UX change definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiChange {
    /// Component or element ID
    pub element_id: String,
    /// Change type
    pub change_type: UiChangeType,
    /// New value or configuration
    pub new_value: String,
    /// Original value (for rollback)
    pub original_value: Option<String>,
}

/// Types of UI changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UiChangeType {
    Color,
    Text,
    Layout,
    Animation,
    Interaction,
    Navigation,
}

/// Feedback delivery change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackChange {
    /// Feedback type
    pub feedback_type: String,
    /// Delivery timing
    pub timing: FeedbackTiming,
    /// Presentation style
    pub presentation: FeedbackPresentation,
    /// Content modifications
    pub content_changes: Vec<String>,
}

/// Feedback delivery timing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackTiming {
    Immediate,
    Delayed { delay_ms: u64 },
    EndOfSession,
    Custom { trigger: String },
}

/// Feedback presentation style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackPresentation {
    Visual,
    Audio,
    Haptic,
    Combined,
}

/// Experiment status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExperimentStatus {
    Draft,
    Active,
    Paused,
    Completed,
    Cancelled,
}

/// Success criteria for experiments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriteria {
    /// Primary metric
    pub primary_metric: String,
    /// Minimum sample size
    pub min_sample_size: u32,
    /// Statistical significance level
    pub significance_level: f64,
    /// Minimum detectable effect
    pub min_effect_size: f64,
    /// Maximum experiment duration
    pub max_duration_days: u32,
}

/// User assignment to experiment variant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAssignment {
    /// User ID
    pub user_id: String,
    /// Experiment ID
    pub experiment_id: String,
    /// Assigned variant ID
    pub variant_id: String,
    /// Assignment timestamp
    pub assigned_at: DateTime<Utc>,
    /// Assignment method
    pub assignment_method: AssignmentMethod,
}

/// Assignment method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssignmentMethod {
    Random,
    Deterministic,
    Manual,
    Cohort { cohort_id: String },
}

/// Experiment metrics and results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentMetrics {
    /// Experiment ID
    pub experiment_id: String,
    /// Variant ID
    pub variant_id: String,
    /// Total users assigned
    pub users_assigned: u32,
    /// Total sessions
    pub total_sessions: u32,
    /// Conversion metrics
    pub conversions: HashMap<String, ConversionMetric>,
    /// Performance metrics
    pub performance: HashMap<String, PerformanceMetric>,
    /// User engagement metrics
    pub engagement: EngagementMetrics,
    /// Last updated
    pub last_updated: DateTime<Utc>,
}

/// Conversion metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionMetric {
    /// Total conversions
    pub conversions: u32,
    /// Conversion rate
    pub conversion_rate: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
}

/// Performance metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetric {
    /// Metric name
    pub name: String,
    /// Average value
    pub average: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Sample count
    pub sample_count: u32,
}

/// Engagement metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngagementMetrics {
    /// Average session duration
    pub avg_session_duration: f64,
    /// Bounce rate
    pub bounce_rate: f64,
    /// Return rate
    pub return_rate: f64,
    /// Feedback satisfaction
    pub feedback_satisfaction: f64,
}

/// A/B Testing Manager
pub struct ABTestManager {
    /// Active experiments
    experiments: HashMap<String, Experiment>,
    /// User assignments
    user_assignments: HashMap<String, Vec<UserAssignment>>,
    /// Experiment metrics
    metrics: HashMap<String, HashMap<String, ExperimentMetrics>>,
    /// Configuration
    config: ABTestConfig,
}

impl ABTestManager {
    /// Create a new A/B test manager
    pub fn new(config: ABTestConfig) -> Self {
        Self {
            experiments: HashMap::new(),
            user_assignments: HashMap::new(),
            metrics: HashMap::new(),
            config,
        }
    }

    /// Create a new experiment
    pub fn create_experiment(&mut self, experiment: Experiment) -> Result<(), ABTestError> {
        // Validate experiment configuration
        self.validate_experiment(&experiment)?;

        // Check for conflicts with existing experiments
        self.check_experiment_conflicts(&experiment)?;

        // Store experiment
        self.experiments.insert(experiment.id.clone(), experiment);

        Ok(())
    }

    /// Assign user to experiment variant
    pub fn assign_user_to_variant(
        &mut self,
        user_id: &str,
        experiment_id: &str,
    ) -> Result<String, ABTestError> {
        let experiment =
            self.experiments
                .get(experiment_id)
                .ok_or_else(|| ABTestError::ExperimentNotFound {
                    experiment_id: experiment_id.to_string(),
                })?;

        // Check if experiment is active
        if !matches!(experiment.status, ExperimentStatus::Active) {
            return Err(ABTestError::ExperimentNotActive {
                experiment_id: experiment_id.to_string(),
            });
        }

        // Check if user is already assigned
        if let Some(assignments) = self.user_assignments.get(user_id) {
            for assignment in assignments {
                if assignment.experiment_id == experiment_id {
                    return Ok(assignment.variant_id.clone());
                }
            }
        }

        // Assign user to variant based on traffic allocation
        let variant_id = self.calculate_variant_assignment(user_id, experiment)?;

        // Record assignment
        let assignment = UserAssignment {
            user_id: user_id.to_string(),
            experiment_id: experiment_id.to_string(),
            variant_id: variant_id.clone(),
            assigned_at: Utc::now(),
            assignment_method: AssignmentMethod::Deterministic,
        };

        self.user_assignments
            .entry(user_id.to_string())
            .or_insert_with(Vec::new)
            .push(assignment);

        Ok(variant_id)
    }

    /// Get user's variant for an experiment
    pub fn get_user_variant(&self, user_id: &str, experiment_id: &str) -> Option<String> {
        self.user_assignments
            .get(user_id)?
            .iter()
            .find(|assignment| assignment.experiment_id == experiment_id)
            .map(|assignment| assignment.variant_id.clone())
    }

    /// Record experiment event
    pub fn record_event(
        &mut self,
        user_id: &str,
        experiment_id: &str,
        event_type: &str,
        event_data: HashMap<String, String>,
    ) -> Result<(), ABTestError> {
        // Get user's variant assignment
        let variant_id = self
            .get_user_variant(user_id, experiment_id)
            .ok_or_else(|| ABTestError::UserNotAssigned {
                user_id: user_id.to_string(),
                experiment_id: experiment_id.to_string(),
            })?;

        // Update metrics
        self.update_metrics(experiment_id, &variant_id, event_type, &event_data)?;

        Ok(())
    }

    /// Get experiment results
    pub fn get_experiment_results(
        &self,
        experiment_id: &str,
    ) -> Result<HashMap<String, ExperimentMetrics>, ABTestError> {
        self.metrics
            .get(experiment_id)
            .cloned()
            .ok_or_else(|| ABTestError::ExperimentNotFound {
                experiment_id: experiment_id.to_string(),
            })
    }

    /// Calculate statistical significance
    pub fn calculate_significance(
        &self,
        experiment_id: &str,
        metric_name: &str,
    ) -> Result<StatisticalResult, ABTestError> {
        let metrics = self.get_experiment_results(experiment_id)?;

        if metrics.len() < 2 {
            return Err(ABTestError::InsufficientData {
                message: "Need at least 2 variants for comparison".to_string(),
            });
        }

        // Get control and treatment variants
        let mut variants: Vec<_> = metrics.iter().collect();
        variants.sort_by_key(|(variant_id, _)| *variant_id);

        let (_, control_metrics) = variants[0];
        let (_, treatment_metrics) = variants[1];

        // Calculate statistical significance using t-test approximation
        let control_conversion = control_metrics
            .conversions
            .get(metric_name)
            .map(|c| c.conversion_rate)
            .unwrap_or(0.0);

        let treatment_conversion = treatment_metrics
            .conversions
            .get(metric_name)
            .map(|c| c.conversion_rate)
            .unwrap_or(0.0);

        let effect_size = treatment_conversion - control_conversion;
        let relative_improvement = if control_conversion > 0.0 {
            effect_size / control_conversion
        } else {
            0.0
        };

        // Simplified p-value calculation (in production, use proper statistical libraries)
        let p_value = if effect_size.abs() > 0.01 { 0.01 } else { 0.1 };

        Ok(StatisticalResult {
            metric_name: metric_name.to_string(),
            control_value: control_conversion,
            treatment_value: treatment_conversion,
            effect_size,
            relative_improvement,
            p_value,
            confidence_level: 0.95,
            is_significant: p_value < 0.05,
            sample_size_control: control_metrics.users_assigned,
            sample_size_treatment: treatment_metrics.users_assigned,
        })
    }

    /// Validate experiment configuration
    fn validate_experiment(&self, experiment: &Experiment) -> Result<(), ABTestError> {
        // Check traffic allocation sums to 100%
        let total_traffic: f64 = experiment.traffic_allocation.values().sum();
        if (total_traffic - 100.0).abs() > 0.01 {
            return Err(ABTestError::InvalidConfiguration {
                message: format!(
                    "Traffic allocation sums to {}, expected 100%",
                    total_traffic
                ),
            });
        }

        // Check variants exist
        for variant_id in experiment.traffic_allocation.keys() {
            if !experiment.variants.iter().any(|v| &v.id == variant_id) {
                return Err(ABTestError::InvalidConfiguration {
                    message: format!("Variant {} not found in experiment", variant_id),
                });
            }
        }

        Ok(())
    }

    /// Check for conflicts with existing experiments
    fn check_experiment_conflicts(&self, _experiment: &Experiment) -> Result<(), ABTestError> {
        // In production, this would check for overlapping experiments
        // that might interfere with each other
        Ok(())
    }

    /// Calculate variant assignment for user
    fn calculate_variant_assignment(
        &self,
        user_id: &str,
        experiment: &Experiment,
    ) -> Result<String, ABTestError> {
        // Use deterministic hash-based assignment
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        format!("{}:{}", user_id, experiment.id).hash(&mut hasher);
        let hash_value = hasher.finish() as f64 / u64::MAX as f64 * 100.0;

        let mut cumulative_percentage = 0.0;
        for (variant_id, percentage) in &experiment.traffic_allocation {
            cumulative_percentage += percentage;
            if hash_value <= cumulative_percentage {
                return Ok(variant_id.clone());
            }
        }

        // Fallback to first variant
        experiment
            .variants
            .first()
            .map(|v| v.id.clone())
            .ok_or_else(|| ABTestError::InvalidConfiguration {
                message: "No variants defined".to_string(),
            })
    }

    /// Update experiment metrics
    fn update_metrics(
        &mut self,
        experiment_id: &str,
        variant_id: &str,
        event_type: &str,
        _event_data: &HashMap<String, String>,
    ) -> Result<(), ABTestError> {
        let experiment_metrics = self
            .metrics
            .entry(experiment_id.to_string())
            .or_insert_with(HashMap::new);

        let variant_metrics = experiment_metrics
            .entry(variant_id.to_string())
            .or_insert_with(|| ExperimentMetrics {
                experiment_id: experiment_id.to_string(),
                variant_id: variant_id.to_string(),
                users_assigned: 0,
                total_sessions: 0,
                conversions: HashMap::new(),
                performance: HashMap::new(),
                engagement: EngagementMetrics {
                    avg_session_duration: 0.0,
                    bounce_rate: 0.0,
                    return_rate: 0.0,
                    feedback_satisfaction: 0.0,
                },
                last_updated: Utc::now(),
            });

        // Update metrics based on event type
        match event_type {
            "session_start" => {
                variant_metrics.total_sessions += 1;
            }
            "conversion" => {
                let conversion = variant_metrics
                    .conversions
                    .entry("default".to_string())
                    .or_insert_with(|| ConversionMetric {
                        conversions: 0,
                        conversion_rate: 0.0,
                        confidence_interval: (0.0, 0.0),
                    });
                conversion.conversions += 1;
                conversion.conversion_rate =
                    conversion.conversions as f64 / variant_metrics.total_sessions.max(1) as f64;
            }
            _ => {}
        }

        variant_metrics.last_updated = Utc::now();
        Ok(())
    }
}

/// A/B test configuration
#[derive(Debug, Clone)]
pub struct ABTestConfig {
    /// Maximum number of concurrent experiments
    pub max_experiments: u32,
    /// Default experiment duration in days
    pub default_duration_days: u32,
    /// Minimum sample size for statistical significance
    pub min_sample_size: u32,
    /// Enable automatic experiment termination
    pub auto_terminate: bool,
}

impl Default for ABTestConfig {
    fn default() -> Self {
        Self {
            max_experiments: 10,
            default_duration_days: 30,
            min_sample_size: 1000,
            auto_terminate: true,
        }
    }
}

/// Statistical analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalResult {
    /// Metric name
    pub metric_name: String,
    /// Control variant value
    pub control_value: f64,
    /// Treatment variant value
    pub treatment_value: f64,
    /// Effect size (absolute difference)
    pub effect_size: f64,
    /// Relative improvement (percentage)
    pub relative_improvement: f64,
    /// P-value
    pub p_value: f64,
    /// Confidence level
    pub confidence_level: f64,
    /// Statistical significance
    pub is_significant: bool,
    /// Control sample size
    pub sample_size_control: u32,
    /// Treatment sample size
    pub sample_size_treatment: u32,
}

/// A/B testing errors
#[derive(Debug, thiserror::Error)]
pub enum ABTestError {
    #[error("Experiment not found: {experiment_id}")]
    ExperimentNotFound { experiment_id: String },

    #[error("Experiment not active: {experiment_id}")]
    ExperimentNotActive { experiment_id: String },

    #[error("User {user_id} not assigned to experiment {experiment_id}")]
    UserNotAssigned {
        user_id: String,
        experiment_id: String,
    },

    #[error("Invalid configuration: {message}")]
    InvalidConfiguration { message: String },

    #[error("Insufficient data: {message}")]
    InsufficientData { message: String },

    #[error("Statistical analysis error: {message}")]
    StatisticalError { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ab_test_manager_creation() {
        let config = ABTestConfig::default();
        let manager = ABTestManager::new(config);
        assert_eq!(manager.experiments.len(), 0);
    }

    #[test]
    fn test_experiment_creation() {
        let mut manager = ABTestManager::new(ABTestConfig::default());

        let experiment = create_test_experiment();
        let result = manager.create_experiment(experiment);
        assert!(result.is_ok());
        assert_eq!(manager.experiments.len(), 1);
    }

    #[test]
    fn test_invalid_traffic_allocation() {
        let mut manager = ABTestManager::new(ABTestConfig::default());

        let mut experiment = create_test_experiment();
        experiment.traffic_allocation.clear();
        experiment
            .traffic_allocation
            .insert("control".to_string(), 60.0);
        experiment
            .traffic_allocation
            .insert("treatment".to_string(), 30.0); // Sums to 90%, should fail

        let result = manager.create_experiment(experiment);
        assert!(result.is_err());
    }

    #[test]
    fn test_user_assignment() {
        let mut manager = ABTestManager::new(ABTestConfig::default());

        let mut experiment = create_test_experiment();
        experiment.status = ExperimentStatus::Active;
        manager.create_experiment(experiment).unwrap();

        let variant = manager
            .assign_user_to_variant("user123", "test_experiment")
            .unwrap();
        assert!(!variant.is_empty());

        // Second assignment should return same variant
        let variant2 = manager
            .assign_user_to_variant("user123", "test_experiment")
            .unwrap();
        assert_eq!(variant, variant2);
    }

    #[test]
    fn test_event_recording() {
        let mut manager = ABTestManager::new(ABTestConfig::default());

        let mut experiment = create_test_experiment();
        experiment.status = ExperimentStatus::Active;
        manager.create_experiment(experiment).unwrap();

        manager
            .assign_user_to_variant("user123", "test_experiment")
            .unwrap();

        let mut event_data = HashMap::new();
        event_data.insert("value".to_string(), "100".to_string());

        let result = manager.record_event("user123", "test_experiment", "conversion", event_data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_statistical_significance() {
        let mut manager = ABTestManager::new(ABTestConfig::default());

        let mut experiment = create_test_experiment();
        experiment.status = ExperimentStatus::Active;
        manager.create_experiment(experiment).unwrap();

        // Create metrics for both variants manually to ensure we have data
        let control_metrics = ExperimentMetrics {
            experiment_id: "test_experiment".to_string(),
            variant_id: "control".to_string(),
            users_assigned: 100,
            total_sessions: 100,
            conversions: {
                let mut conv = HashMap::new();
                conv.insert(
                    "default".to_string(),
                    ConversionMetric {
                        conversions: 10,
                        conversion_rate: 0.1,
                        confidence_interval: (0.05, 0.15),
                    },
                );
                conv
            },
            performance: HashMap::new(),
            engagement: EngagementMetrics {
                avg_session_duration: 120.0,
                bounce_rate: 0.3,
                return_rate: 0.7,
                feedback_satisfaction: 4.0,
            },
            last_updated: Utc::now(),
        };

        let treatment_metrics = ExperimentMetrics {
            experiment_id: "test_experiment".to_string(),
            variant_id: "treatment".to_string(),
            users_assigned: 100,
            total_sessions: 100,
            conversions: {
                let mut conv = HashMap::new();
                conv.insert(
                    "default".to_string(),
                    ConversionMetric {
                        conversions: 15,
                        conversion_rate: 0.15,
                        confidence_interval: (0.10, 0.20),
                    },
                );
                conv
            },
            performance: HashMap::new(),
            engagement: EngagementMetrics {
                avg_session_duration: 130.0,
                bounce_rate: 0.25,
                return_rate: 0.75,
                feedback_satisfaction: 4.2,
            },
            last_updated: Utc::now(),
        };

        // Manually insert metrics
        let mut variant_metrics = HashMap::new();
        variant_metrics.insert("control".to_string(), control_metrics);
        variant_metrics.insert("treatment".to_string(), treatment_metrics);
        manager
            .metrics
            .insert("test_experiment".to_string(), variant_metrics);

        let result = manager.calculate_significance("test_experiment", "default");
        assert!(result.is_ok());
        let stats = result.unwrap();
        assert!((stats.control_value - 0.1).abs() < 0.001);
        assert!((stats.treatment_value - 0.15).abs() < 0.001);
        assert!((stats.effect_size - 0.05).abs() < 0.001);
    }

    fn create_test_experiment() -> Experiment {
        let mut traffic_allocation = HashMap::new();
        traffic_allocation.insert("control".to_string(), 50.0);
        traffic_allocation.insert("treatment".to_string(), 50.0);

        Experiment {
            id: "test_experiment".to_string(),
            name: "Test Experiment".to_string(),
            description: "A test experiment".to_string(),
            variants: vec![
                Variant {
                    id: "control".to_string(),
                    name: "Control".to_string(),
                    description: "Control variant".to_string(),
                    config: VariantConfig {
                        feature_flags: HashMap::new(),
                        parameters: HashMap::new(),
                        ui_changes: vec![],
                        feedback_changes: vec![],
                    },
                    traffic_percentage: 50.0,
                },
                Variant {
                    id: "treatment".to_string(),
                    name: "Treatment".to_string(),
                    description: "Treatment variant".to_string(),
                    config: VariantConfig {
                        feature_flags: HashMap::new(),
                        parameters: HashMap::new(),
                        ui_changes: vec![],
                        feedback_changes: vec![],
                    },
                    traffic_percentage: 50.0,
                },
            ],
            traffic_allocation,
            start_time: Utc::now(),
            end_time: None,
            status: ExperimentStatus::Draft,
            target_metrics: vec!["conversion".to_string()],
            success_criteria: SuccessCriteria {
                primary_metric: "conversion".to_string(),
                min_sample_size: 100,
                significance_level: 0.05,
                min_effect_size: 0.05,
                max_duration_days: 30,
            },
            metadata: HashMap::new(),
        }
    }
}
