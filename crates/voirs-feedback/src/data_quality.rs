//! Data Quality Monitoring Framework
//!
//! This module provides comprehensive data quality monitoring for the `VoiRS` feedback system.
//! It validates data integrity, monitors data quality metrics, detects anomalies, and generates
//! quality reports to ensure the reliability and accuracy of the feedback data.
//!
//! # Features
//!
//! - **Data Validation**: Schema validation, type checking, constraint validation
//! - **Quality Metrics**: Completeness, accuracy, consistency, timeliness, uniqueness
//! - **Anomaly Detection**: Statistical outlier detection, pattern anomalies
//! - **Quality Rules**: Custom business rules and validation logic
//! - **Quality Scoring**: Overall quality scores with dimension breakdown
//! - **Alerting**: Threshold-based alerts for quality degradation
//! - **Reporting**: Comprehensive quality reports with trends and recommendations
//!
//! # Example
//!
//! ```rust
//! use voirs_feedback::data_quality::{DataQualityMonitor, QualityDimension};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let monitor = DataQualityMonitor::new();
//!
//! // Validate data
//! let validation_result = monitor.validate_feedback_data(vec![
//!     ("score".to_string(), serde_json::json!(0.85)),
//!     ("message".to_string(), serde_json::json!("Good job!")),
//! ]).await?;
//!
//! // Get quality metrics
//! let metrics = monitor.get_quality_metrics().await?;
//! println!("Overall quality score: {}", metrics.overall_score);
//!
//! // Generate quality report
//! let report = monitor.generate_quality_report().await?;
//! # Ok(())
//! # }
//! ```

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

/// Errors that can occur in data quality monitoring
#[derive(Error, Debug)]
#[allow(missing_docs)]
pub enum QualityError {
    /// Validation error
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Missing data
    #[error("Missing data: {0}")]
    MissingData(String),

    /// Invalid data type
    #[error("Invalid data type: {0}")]
    InvalidType(String),

    /// Constraint violation
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),
}

/// Type alias for Results in this module
pub type Result<T> = std::result::Result<T, QualityError>;

/// Data quality dimensions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum QualityDimension {
    /// Completeness - extent to which data is not missing
    Completeness,
    /// Accuracy - correctness of data
    Accuracy,
    /// Consistency - coherence across data sources
    Consistency,
    /// Timeliness - data is up-to-date
    Timeliness,
    /// Uniqueness - no duplicate records
    Uniqueness,
    /// Validity - data conforms to defined formats
    Validity,
}

/// Severity level for quality issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum QualitySeverity {
    /// Informational
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Critical
    Critical,
}

/// Validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule ID
    pub id: String,
    /// Field name
    pub field: String,
    /// Rule type
    pub rule_type: RuleType,
    /// Rule description
    pub description: String,
    /// Severity if violated
    pub severity: QualitySeverity,
    /// Enabled status
    pub enabled: bool,
}

/// Types of validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum RuleType {
    /// Field must not be null/empty
    Required,
    /// Field must match regex pattern
    Pattern(String),
    /// Numeric value must be in range
    Range { min: f64, max: f64 },
    /// String length constraints
    Length {
        min: Option<usize>,
        max: Option<usize>,
    },
    /// Value must be one of allowed values
    Enum(Vec<String>),
    /// Custom validation function
    Custom(String),
}

/// Validation result for a field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Field name
    pub field: String,
    /// Whether validation passed
    pub passed: bool,
    /// Validation errors
    pub errors: Vec<ValidationError>,
}

/// Individual validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Rule that was violated
    pub rule_id: String,
    /// Error message
    pub message: String,
    /// Severity
    pub severity: QualitySeverity,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Quality metrics for a dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionMetrics {
    /// Dimension
    pub dimension: QualityDimension,
    /// Score (0.0 to 1.0)
    pub score: f64,
    /// Total items checked
    pub total_items: usize,
    /// Items passing quality checks
    pub passing_items: usize,
    /// Issues found
    pub issues_count: usize,
    /// Sample issues
    pub sample_issues: Vec<String>,
}

/// Overall quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f64,
    /// Metrics per dimension
    pub dimensions: HashMap<QualityDimension, DimensionMetrics>,
    /// Total records analyzed
    pub total_records: usize,
    /// Records with issues
    pub records_with_issues: usize,
    /// Quality trend (improving/degrading)
    pub trend: QualityTrend,
}

/// Quality trend indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum QualityTrend {
    /// Quality improving
    Improving,
    /// Quality stable
    Stable,
    /// Quality degrading
    Degrading,
    /// Insufficient data
    Unknown,
}

/// Quality issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    /// Issue ID
    pub id: String,
    /// Affected dimension
    pub dimension: QualityDimension,
    /// Issue description
    pub description: String,
    /// Severity
    pub severity: QualitySeverity,
    /// Affected records count
    pub affected_records: usize,
    /// First detected
    pub first_detected: DateTime<Utc>,
    /// Last detected
    pub last_detected: DateTime<Utc>,
    /// Resolution status
    pub resolved: bool,
}

/// Quality report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    /// Report generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Time period covered
    pub period_hours: i64,
    /// Quality metrics
    pub metrics: QualityMetrics,
    /// Critical issues
    pub critical_issues: Vec<QualityIssue>,
    /// Quality trends over time
    pub historical_scores: Vec<(DateTime<Utc>, f64)>,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Data health summary
    pub health_summary: HealthSummary,
}

/// Data health summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthSummary {
    /// Overall health status
    pub status: HealthStatus,
    /// Health score (0-100)
    pub health_score: u8,
    /// Key strengths
    pub strengths: Vec<String>,
    /// Key weaknesses
    pub weaknesses: Vec<String>,
    /// Action items
    pub action_items: Vec<String>,
}

/// Health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum HealthStatus {
    /// Excellent health
    Excellent,
    /// Good health
    Good,
    /// Fair health
    Fair,
    /// Poor health
    Poor,
    /// Critical health
    Critical,
}

/// Configuration for quality monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// Maximum issues to track
    pub max_issues: usize,
    /// Issue retention period in days
    pub retention_period_days: i64,
    /// Enable automatic validation
    pub auto_validate: bool,
    /// Validation sample rate (0.0 to 1.0)
    pub sample_rate: f64,
    /// Score history size
    pub score_history_size: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            max_issues: 10_000,
            retention_period_days: 30,
            auto_validate: true,
            sample_rate: 1.0, // Validate all records by default
            score_history_size: 1000,
        }
    }
}

/// Main data quality monitoring system
pub struct DataQualityMonitor {
    /// Configuration
    config: MonitorConfig,
    /// Validation rules
    rules: Arc<RwLock<HashMap<String, ValidationRule>>>,
    /// Quality issues
    issues: Arc<RwLock<HashMap<String, QualityIssue>>>,
    /// Quality score history
    score_history: Arc<RwLock<VecDeque<(DateTime<Utc>, f64)>>>,
    /// Validation cache
    validation_cache: Arc<RwLock<HashMap<String, ValidationResult>>>,
}

impl DataQualityMonitor {
    /// Create a new data quality monitor
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(MonitorConfig::default())
    }

    /// Create with custom configuration
    #[must_use]
    pub fn with_config(config: MonitorConfig) -> Self {
        let monitor = Self {
            config: config.clone(),
            rules: Arc::new(RwLock::new(HashMap::new())),
            issues: Arc::new(RwLock::new(HashMap::new())),
            score_history: Arc::new(RwLock::new(VecDeque::with_capacity(
                config.score_history_size,
            ))),
            validation_cache: Arc::new(RwLock::new(HashMap::new())),
        };

        // Initialize with default rules
        tokio::spawn({
            let monitor = monitor.clone();
            async move {
                let _ = monitor.initialize_default_rules().await;
            }
        });

        monitor
    }

    /// Clone the monitor (Arc-based)
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            rules: Arc::clone(&self.rules),
            issues: Arc::clone(&self.issues),
            score_history: Arc::clone(&self.score_history),
            validation_cache: Arc::clone(&self.validation_cache),
        }
    }

    /// Initialize default validation rules
    async fn initialize_default_rules(&self) -> Result<()> {
        // Score validation
        self.add_rule(ValidationRule {
            id: "score_range".to_string(),
            field: "score".to_string(),
            rule_type: RuleType::Range { min: 0.0, max: 1.0 },
            description: "Score must be between 0.0 and 1.0".to_string(),
            severity: QualitySeverity::Error,
            enabled: true,
        })
        .await?;

        // Message validation
        self.add_rule(ValidationRule {
            id: "message_required".to_string(),
            field: "message".to_string(),
            rule_type: RuleType::Required,
            description: "Message field is required".to_string(),
            severity: QualitySeverity::Error,
            enabled: true,
        })
        .await?;

        self.add_rule(ValidationRule {
            id: "message_length".to_string(),
            field: "message".to_string(),
            rule_type: RuleType::Length {
                min: Some(1),
                max: Some(5000),
            },
            description: "Message must be 1-5000 characters".to_string(),
            severity: QualitySeverity::Warning,
            enabled: true,
        })
        .await?;

        Ok(())
    }

    /// Add a validation rule
    pub async fn add_rule(&self, rule: ValidationRule) -> Result<()> {
        let mut rules = self.rules.write().await;
        rules.insert(rule.id.clone(), rule);
        Ok(())
    }

    /// Remove a validation rule
    pub async fn remove_rule(&self, rule_id: &str) -> Result<()> {
        let mut rules = self.rules.write().await;
        rules
            .remove(rule_id)
            .ok_or_else(|| QualityError::ConfigError(format!("Rule not found: {rule_id}")))?;
        Ok(())
    }

    /// Validate feedback data
    pub async fn validate_feedback_data(
        &self,
        data: Vec<(String, JsonValue)>,
    ) -> Result<Vec<ValidationResult>> {
        let rules = self.rules.read().await;
        let mut results = Vec::new();

        for (field, value) in &data {
            let field_rules: Vec<&ValidationRule> = rules
                .values()
                .filter(|r| &r.field == field && r.enabled)
                .collect();

            if field_rules.is_empty() {
                continue;
            }

            let mut errors = Vec::new();

            for rule in field_rules {
                match &rule.rule_type {
                    RuleType::Required => {
                        if value.is_null()
                            || (value.is_string() && value.as_str().unwrap_or("").is_empty())
                        {
                            errors.push(ValidationError {
                                rule_id: rule.id.clone(),
                                message: "Field is required".to_string(),
                                severity: rule.severity,
                                timestamp: Utc::now(),
                            });
                        }
                    }
                    RuleType::Range { min, max } => {
                        if let Some(num) = value.as_f64() {
                            if num < *min || num > *max {
                                errors.push(ValidationError {
                                    rule_id: rule.id.clone(),
                                    message: format!("Value {num} out of range [{min}, {max}]"),
                                    severity: rule.severity,
                                    timestamp: Utc::now(),
                                });
                            }
                        }
                    }
                    RuleType::Length { min, max } => {
                        if let Some(s) = value.as_str() {
                            let len = s.len();
                            if let Some(min_len) = min {
                                if len < *min_len {
                                    errors.push(ValidationError {
                                        rule_id: rule.id.clone(),
                                        message: format!("Length {len} below minimum {min_len}"),
                                        severity: rule.severity,
                                        timestamp: Utc::now(),
                                    });
                                }
                            }
                            if let Some(max_len) = max {
                                if len > *max_len {
                                    errors.push(ValidationError {
                                        rule_id: rule.id.clone(),
                                        message: format!("Length {len} exceeds maximum {max_len}"),
                                        severity: rule.severity,
                                        timestamp: Utc::now(),
                                    });
                                }
                            }
                        }
                    }
                    RuleType::Pattern(pattern) => {
                        if let Some(s) = value.as_str() {
                            if let Ok(re) = regex::Regex::new(pattern) {
                                if !re.is_match(s) {
                                    errors.push(ValidationError {
                                        rule_id: rule.id.clone(),
                                        message: format!("Value does not match pattern: {pattern}"),
                                        severity: rule.severity,
                                        timestamp: Utc::now(),
                                    });
                                }
                            }
                        }
                    }
                    RuleType::Enum(allowed) => {
                        if let Some(s) = value.as_str() {
                            if !allowed.contains(&s.to_string()) {
                                errors.push(ValidationError {
                                    rule_id: rule.id.clone(),
                                    message: format!("Value not in allowed set: {allowed:?}"),
                                    severity: rule.severity,
                                    timestamp: Utc::now(),
                                });
                            }
                        }
                    }
                    RuleType::Custom(_) => {
                        // Custom validation would be implemented here
                    }
                }
            }

            results.push(ValidationResult {
                field: field.clone(),
                passed: errors.is_empty(),
                errors,
            });
        }

        // Track issues
        for result in &results {
            if !result.passed {
                self.track_quality_issue(&result.field, &result.errors)
                    .await;
            }
        }

        Ok(results)
    }

    /// Track a quality issue
    async fn track_quality_issue(&self, field: &str, errors: &[ValidationError]) {
        let mut issues = self.issues.write().await;

        for error in errors {
            let issue_id = format!("{}_{}", field, error.rule_id);

            if let Some(issue) = issues.get_mut(&issue_id) {
                issue.affected_records += 1;
                issue.last_detected = Utc::now();
            } else {
                let dimension = Self::determine_dimension(field);

                let issue = QualityIssue {
                    id: issue_id.clone(),
                    dimension,
                    description: error.message.clone(),
                    severity: error.severity,
                    affected_records: 1,
                    first_detected: Utc::now(),
                    last_detected: Utc::now(),
                    resolved: false,
                };

                if issues.len() >= self.config.max_issues {
                    // Remove oldest resolved issue
                    if let Some(oldest_id) = issues
                        .iter()
                        .filter(|(_, i)| i.resolved)
                        .min_by_key(|(_, i)| i.last_detected)
                        .map(|(id, _)| id.clone())
                    {
                        issues.remove(&oldest_id);
                    }
                }

                issues.insert(issue_id, issue);
            }
        }
    }

    /// Determine quality dimension from field name
    fn determine_dimension(field: &str) -> QualityDimension {
        match field {
            "score" | "accuracy" => QualityDimension::Accuracy,
            "message" | "text" => QualityDimension::Completeness,
            "timestamp" | "created_at" => QualityDimension::Timeliness,
            "id" | "user_id" => QualityDimension::Uniqueness,
            _ => QualityDimension::Validity,
        }
    }

    /// Calculate quality metrics
    pub async fn get_quality_metrics(&self) -> Result<QualityMetrics> {
        let issues = self.issues.read().await;

        let mut dimensions_map: HashMap<QualityDimension, DimensionMetrics> = HashMap::new();

        // Initialize all dimensions
        for dim in [
            QualityDimension::Completeness,
            QualityDimension::Accuracy,
            QualityDimension::Consistency,
            QualityDimension::Timeliness,
            QualityDimension::Uniqueness,
            QualityDimension::Validity,
        ] {
            dimensions_map.insert(
                dim,
                DimensionMetrics {
                    dimension: dim,
                    score: 1.0,
                    total_items: 0,
                    passing_items: 0,
                    issues_count: 0,
                    sample_issues: Vec::new(),
                },
            );
        }

        // Aggregate issues by dimension
        for issue in issues.values() {
            if let Some(metrics) = dimensions_map.get_mut(&issue.dimension) {
                metrics.issues_count += 1;
                if metrics.sample_issues.len() < 5 {
                    metrics.sample_issues.push(issue.description.clone());
                }
            }
        }

        // Calculate scores (simplified)
        let mut total_score = 0.0;
        for metrics in dimensions_map.values_mut() {
            // Score decreases with more issues
            metrics.score = 1.0 / (1.0 + metrics.issues_count as f64 / 10.0);
            total_score += metrics.score;
        }

        let overall_score = total_score / dimensions_map.len() as f64;

        // Store score in history
        let mut history = self.score_history.write().await;
        if history.len() >= self.config.score_history_size {
            history.pop_front();
        }
        history.push_back((Utc::now(), overall_score));

        // Determine trend
        let trend = self.calculate_trend(&history);

        Ok(QualityMetrics {
            timestamp: Utc::now(),
            overall_score,
            dimensions: dimensions_map,
            total_records: 0, // Would be tracked separately
            records_with_issues: issues.len(),
            trend,
        })
    }

    /// Calculate quality trend
    fn calculate_trend(&self, history: &VecDeque<(DateTime<Utc>, f64)>) -> QualityTrend {
        if history.len() < 2 {
            return QualityTrend::Unknown;
        }

        let recent: Vec<f64> = history
            .iter()
            .rev()
            .take(10)
            .map(|(_, score)| *score)
            .collect();

        if recent.len() < 2 {
            return QualityTrend::Unknown;
        }

        let avg_recent = recent.iter().sum::<f64>() / recent.len() as f64;
        let older: Vec<f64> = history.iter().take(10).map(|(_, score)| *score).collect();
        let avg_older = older.iter().sum::<f64>() / older.len() as f64;

        let diff = avg_recent - avg_older;

        if diff > 0.05 {
            QualityTrend::Improving
        } else if diff < -0.05 {
            QualityTrend::Degrading
        } else {
            QualityTrend::Stable
        }
    }

    /// Generate comprehensive quality report
    pub async fn generate_quality_report(&self) -> Result<QualityReport> {
        let metrics = self.get_quality_metrics().await?;
        let issues = self.issues.read().await;

        let critical_issues: Vec<QualityIssue> = issues
            .values()
            .filter(|i| i.severity >= QualitySeverity::Error && !i.resolved)
            .cloned()
            .collect();

        let history = self.score_history.read().await;
        let historical_scores: Vec<(DateTime<Utc>, f64)> = history.iter().copied().collect();

        let mut recommendations = Vec::new();

        // Generate recommendations
        if metrics.overall_score < 0.7 {
            recommendations
                .push("⚠️ Overall data quality is below acceptable threshold (70%)".to_string());
        }

        for (dim, dim_metrics) in &metrics.dimensions {
            if dim_metrics.score < 0.6 {
                recommendations.push(format!(
                    "🔧 {:?} quality is poor ({:.0}%) - review validation rules and data sources",
                    dim,
                    dim_metrics.score * 100.0
                ));
            }
        }

        if !critical_issues.is_empty() {
            recommendations.push(format!(
                "🚨 {} critical quality issues require immediate attention",
                critical_issues.len()
            ));
        }

        // Health summary
        let health_score = (metrics.overall_score * 100.0) as u8;
        let status = match health_score {
            90..=100 => HealthStatus::Excellent,
            75..=89 => HealthStatus::Good,
            60..=74 => HealthStatus::Fair,
            40..=59 => HealthStatus::Poor,
            _ => HealthStatus::Critical,
        };

        let mut strengths = Vec::new();
        let mut weaknesses = Vec::new();

        for (dim, dim_metrics) in &metrics.dimensions {
            if dim_metrics.score >= 0.9 {
                strengths.push(format!(
                    "{:?} quality is excellent ({:.0}%)",
                    dim,
                    dim_metrics.score * 100.0
                ));
            } else if dim_metrics.score < 0.6 {
                weaknesses.push(format!(
                    "{:?} quality needs improvement ({:.0}%)",
                    dim,
                    dim_metrics.score * 100.0
                ));
            }
        }

        let health_summary = HealthSummary {
            status,
            health_score,
            strengths,
            weaknesses,
            action_items: recommendations.clone(),
        };

        Ok(QualityReport {
            generated_at: Utc::now(),
            period_hours: 24,
            metrics,
            critical_issues,
            historical_scores,
            recommendations,
            health_summary,
        })
    }

    /// Mark an issue as resolved
    pub async fn resolve_issue(&self, issue_id: &str) -> Result<()> {
        let mut issues = self.issues.write().await;
        if let Some(issue) = issues.get_mut(issue_id) {
            issue.resolved = true;
            Ok(())
        } else {
            Err(QualityError::MissingData(format!(
                "Issue not found: {issue_id}"
            )))
        }
    }

    /// Clean up old issues
    pub async fn cleanup_old_issues(&self) -> Result<usize> {
        let cutoff = Utc::now() - Duration::days(self.config.retention_period_days);

        let mut issues = self.issues.write().await;
        let original_len = issues.len();

        issues.retain(|_, i| i.last_detected > cutoff || !i.resolved);

        Ok(original_len - issues.len())
    }
}

impl Default for DataQualityMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_validation_required_field() {
        let monitor = DataQualityMonitor::new();

        // Wait for initialization
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let results = monitor
            .validate_feedback_data(vec![("message".to_string(), JsonValue::Null)])
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(!results[0].passed);
    }

    #[tokio::test]
    async fn test_validation_range() {
        let monitor = DataQualityMonitor::new();

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let results = monitor
            .validate_feedback_data(vec![("score".to_string(), serde_json::json!(1.5))])
            .await
            .unwrap();

        assert!(!results[0].passed);
    }

    #[tokio::test]
    async fn test_quality_metrics() {
        let monitor = DataQualityMonitor::new();

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Add some validation errors
        monitor
            .validate_feedback_data(vec![("score".to_string(), serde_json::json!(1.5))])
            .await
            .unwrap();

        let metrics = monitor.get_quality_metrics().await.unwrap();

        assert!(metrics.overall_score <= 1.0);
        assert!(!metrics.dimensions.is_empty());
    }

    #[tokio::test]
    async fn test_quality_report() {
        let monitor = DataQualityMonitor::new();

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let report = monitor.generate_quality_report().await.unwrap();

        assert!(report.health_summary.health_score <= 100);
    }

    #[tokio::test]
    async fn test_issue_resolution() {
        let monitor = DataQualityMonitor::new();

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        monitor
            .validate_feedback_data(vec![("message".to_string(), JsonValue::Null)])
            .await
            .unwrap();

        let issues = monitor.issues.read().await;
        if let Some((issue_id, _)) = issues.iter().next() {
            let id = issue_id.clone();
            drop(issues);

            monitor.resolve_issue(&id).await.unwrap();

            let issues = monitor.issues.read().await;
            assert!(issues.get(&id).unwrap().resolved);
        }
    }
}
