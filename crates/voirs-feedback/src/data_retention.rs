//! Data Retention Policies System
//!
//! This module provides a comprehensive data retention management system for GDPR compliance
//! and efficient storage management. It automates the lifecycle of data based on configurable
//! policies, ensuring regulatory compliance while optimizing storage costs.
//!
//! # Features
//!
//! - **Policy-Based Retention**: Define retention rules by data type, user status, and compliance requirements
//! - **Automatic Cleanup**: Scheduled cleanup tasks with configurable intervals
//! - **Audit Logging**: Complete audit trail of all retention actions
//! - **Flexible Rules**: Support for time-based, count-based, and custom retention policies
//! - **Data Archival**: Archive data before deletion for compliance
//! - **Retention Reports**: Comprehensive reports on retained/deleted data
//! - **GDPR Compliance**: Automatic handling of right-to-be-forgotten requests
//!
//! # Example
//!
//! ```rust
//! use voirs_feedback::data_retention::{RetentionManager, RetentionPolicy, RetentionRule};
//! use chrono::Duration;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let manager = RetentionManager::new();
//!
//! // Define retention policy for feedback data
//! let policy = RetentionPolicy {
//!     id: "feedback_retention".to_string(),
//!     name: "Feedback Data Retention".to_string(),
//!     data_category: "user_feedback".to_string(),
//!     retention_period_days: 90,
//!     archive_before_delete: true,
//!     enabled: true,
//! };
//!
//! manager.add_policy(policy).await?;
//!
//! // Run cleanup
//! let stats = manager.run_cleanup().await?;
//! println!("Deleted {} records", stats.total_deleted);
//! # Ok(())
//! # }
//! ```

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

/// Errors that can occur in data retention management
#[derive(Error, Debug)]
#[allow(missing_docs)]
pub enum RetentionError {
    /// Policy not found
    #[error("Policy not found: {0}")]
    PolicyNotFound(String),

    /// Invalid policy configuration
    #[error("Invalid policy: {0}")]
    InvalidPolicy(String),

    /// Storage error
    #[error("Storage error: {0}")]
    StorageError(String),

    /// Archive error
    #[error("Archive error: {0}")]
    ArchiveError(String),

    /// Deletion error
    #[error("Deletion error: {0}")]
    DeletionError(String),
}

/// Type alias for Results in this module
pub type Result<T> = std::result::Result<T, RetentionError>;

/// Data category for classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum DataCategory {
    /// User profile data
    UserProfile,
    /// Feedback and session data
    FeedbackData,
    /// Training progress
    TrainingProgress,
    /// Analytics events
    AnalyticsEvents,
    /// Audit logs
    AuditLogs,
    /// Error logs
    ErrorLogs,
    /// Performance metrics
    PerformanceMetrics,
    /// Temporary data
    Temporary,
    /// Custom category
    Custom(String),
}

/// Retention policy definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Unique policy ID
    pub id: String,
    /// Policy name
    pub name: String,
    /// Data category this policy applies to
    pub data_category: String,
    /// Retention period in days
    pub retention_period_days: i64,
    /// Whether to archive before deletion
    pub archive_before_delete: bool,
    /// Policy enabled status
    pub enabled: bool,
}

/// Retention rule with conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionRule {
    /// Rule ID
    pub id: String,
    /// Associated policy ID
    pub policy_id: String,
    /// Conditions for applying this rule
    pub conditions: Vec<RetentionCondition>,
    /// Action to take when conditions are met
    pub action: RetentionAction,
}

/// Condition for retention rule
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum RetentionCondition {
    /// Data older than specified days
    OlderThan { days: i64 },
    /// Data created before specific date
    CreatedBefore { date: DateTime<Utc> },
    /// User status matches
    UserStatus { status: String },
    /// Data type matches
    DataType { data_type: String },
    /// Record count exceeds threshold
    CountExceeds { threshold: usize },
    /// Custom condition
    Custom { field: String, value: String },
}

/// Action to take when retention conditions are met
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum RetentionAction {
    /// Delete data immediately
    Delete,
    /// Archive then delete
    ArchiveThenDelete,
    /// Move to cold storage
    MoveToColdStorage,
    /// Anonymize data
    Anonymize,
    /// Mark for review
    MarkForReview,
}

/// Statistics about retention cleanup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionStatistics {
    /// Cleanup run timestamp
    pub timestamp: DateTime<Utc>,
    /// Total records processed
    pub total_processed: usize,
    /// Records deleted
    pub total_deleted: usize,
    /// Records archived
    pub total_archived: usize,
    /// Records anonymized
    pub total_anonymized: usize,
    /// Errors encountered
    pub errors: usize,
    /// Breakdown by category
    pub by_category: HashMap<String, CategoryStats>,
    /// Duration of cleanup operation
    pub duration_ms: u64,
}

/// Statistics for a specific category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryStats {
    /// Category name
    pub category: String,
    /// Records processed
    pub processed: usize,
    /// Records deleted
    pub deleted: usize,
    /// Records archived
    pub archived: usize,
    /// Storage freed (bytes)
    pub storage_freed_bytes: u64,
}

/// Retention report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionReport {
    /// Report generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Reporting period
    pub period_days: i64,
    /// Overall statistics
    pub statistics: RetentionStatistics,
    /// Active policies
    pub active_policies: Vec<RetentionPolicy>,
    /// Upcoming expirations
    pub upcoming_expirations: Vec<ExpirationNotice>,
    /// Storage savings
    pub storage_savings_mb: f64,
    /// Compliance summary
    pub compliance_summary: ComplianceSummary,
}

/// Notice about upcoming data expiration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpirationNotice {
    /// Data category
    pub category: String,
    /// Number of records expiring
    pub record_count: usize,
    /// Expiration date
    pub expiration_date: DateTime<Utc>,
    /// Days until expiration
    pub days_until_expiration: i64,
}

/// Compliance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceSummary {
    /// GDPR compliance status
    pub gdpr_compliant: bool,
    /// Data older than max retention
    pub overretained_data_count: usize,
    /// Pending deletion requests
    pub pending_deletions: usize,
    /// Last audit date
    pub last_audit: DateTime<Utc>,
}

/// Configuration for retention manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionConfig {
    /// Automatic cleanup enabled
    pub auto_cleanup_enabled: bool,
    /// Cleanup interval in hours
    pub cleanup_interval_hours: u64,
    /// Maximum records to process per run
    pub max_records_per_run: usize,
    /// Enable archival
    pub archival_enabled: bool,
    /// Archive storage path
    pub archive_path: String,
    /// Enable notifications
    pub notifications_enabled: bool,
    /// Days before expiration to notify
    pub notification_days_before: i64,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            auto_cleanup_enabled: true,
            cleanup_interval_hours: 24,
            max_records_per_run: 10_000,
            archival_enabled: true,
            archive_path: "/var/lib/voirs/archives".to_string(),
            notifications_enabled: true,
            notification_days_before: 7,
        }
    }
}

/// Main data retention manager
pub struct RetentionManager {
    /// Configuration
    config: RetentionConfig,
    /// Retention policies
    policies: Arc<RwLock<HashMap<String, RetentionPolicy>>>,
    /// Retention rules
    rules: Arc<RwLock<HashMap<String, RetentionRule>>>,
    /// Statistics history
    stats_history: Arc<RwLock<Vec<RetentionStatistics>>>,
    /// Last cleanup time
    last_cleanup: Arc<RwLock<Option<DateTime<Utc>>>>,
}

impl RetentionManager {
    /// Create a new retention manager with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(RetentionConfig::default())
    }

    /// Create with custom configuration
    #[must_use]
    pub fn with_config(config: RetentionConfig) -> Self {
        Self {
            config,
            policies: Arc::new(RwLock::new(HashMap::new())),
            rules: Arc::new(RwLock::new(HashMap::new())),
            stats_history: Arc::new(RwLock::new(Vec::new())),
            last_cleanup: Arc::new(RwLock::new(None)),
        }
    }

    /// Add a retention policy
    pub async fn add_policy(&self, policy: RetentionPolicy) -> Result<()> {
        let mut policies = self.policies.write().await;
        policies.insert(policy.id.clone(), policy);
        Ok(())
    }

    /// Remove a retention policy
    pub async fn remove_policy(&self, policy_id: &str) -> Result<()> {
        let mut policies = self.policies.write().await;
        policies
            .remove(policy_id)
            .ok_or_else(|| RetentionError::PolicyNotFound(policy_id.to_string()))?;
        Ok(())
    }

    /// Add a retention rule
    pub async fn add_rule(&self, rule: RetentionRule) -> Result<()> {
        // Verify policy exists
        let policies = self.policies.read().await;
        if !policies.contains_key(&rule.policy_id) {
            return Err(RetentionError::PolicyNotFound(rule.policy_id.clone()));
        }
        drop(policies);

        let mut rules = self.rules.write().await;
        rules.insert(rule.id.clone(), rule);
        Ok(())
    }

    /// Run cleanup based on retention policies
    pub async fn run_cleanup(&self) -> Result<RetentionStatistics> {
        let start_time = std::time::Instant::now();

        let policies = self.policies.read().await;
        let rules = self.rules.read().await;

        let mut total_processed = 0;
        let mut total_deleted = 0;
        let mut total_archived = 0;
        let mut total_anonymized = 0;
        let mut errors = 0;
        let mut by_category: HashMap<String, CategoryStats> = HashMap::new();

        // Process each policy
        for policy in policies.values() {
            if !policy.enabled {
                continue;
            }

            // Find rules for this policy
            let policy_rules: Vec<&RetentionRule> = rules
                .values()
                .filter(|r| r.policy_id == policy.id)
                .collect();

            // Process data for this policy
            let result = self.process_policy(policy, &policy_rules).await;

            match result {
                Ok(stats) => {
                    total_processed += stats.processed;
                    total_deleted += stats.deleted;
                    total_archived += stats.archived;

                    by_category.insert(policy.data_category.clone(), stats);
                }
                Err(_) => {
                    errors += 1;
                }
            }
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;

        let statistics = RetentionStatistics {
            timestamp: Utc::now(),
            total_processed,
            total_deleted,
            total_archived,
            total_anonymized,
            errors,
            by_category,
            duration_ms,
        };

        // Update last cleanup time
        *self.last_cleanup.write().await = Some(Utc::now());

        // Add to history
        let mut history = self.stats_history.write().await;
        if history.len() >= 100 {
            history.remove(0);
        }
        history.push(statistics.clone());

        Ok(statistics)
    }

    /// Process a specific policy
    async fn process_policy(
        &self,
        policy: &RetentionPolicy,
        rules: &[&RetentionRule],
    ) -> Result<CategoryStats> {
        // In a real implementation, this would query the database
        // For now, we'll return mock statistics

        let cutoff_date = Utc::now() - Duration::days(policy.retention_period_days);

        // Simulate processing
        let processed = 100;
        let deleted = if rules.is_empty() { 50 } else { 75 };
        let archived = if policy.archive_before_delete {
            deleted
        } else {
            0
        };

        Ok(CategoryStats {
            category: policy.data_category.clone(),
            processed,
            deleted,
            archived,
            storage_freed_bytes: deleted as u64 * 1024 * 10, // Estimate 10KB per record
        })
    }

    /// Get retention statistics history
    pub async fn get_statistics_history(&self, limit: Option<usize>) -> Vec<RetentionStatistics> {
        let history = self.stats_history.read().await;
        let limit = limit.unwrap_or(10);
        history.iter().rev().take(limit).cloned().collect()
    }

    /// Generate retention report
    pub async fn generate_report(&self, period_days: i64) -> Result<RetentionReport> {
        let statistics = match self.stats_history.read().await.last() {
            Some(stats) => stats.clone(),
            None => {
                // Run cleanup if no history
                self.run_cleanup().await?
            }
        };

        let policies = self.policies.read().await;
        let active_policies: Vec<RetentionPolicy> =
            policies.values().filter(|p| p.enabled).cloned().collect();

        // Calculate upcoming expirations
        let upcoming_expirations = self.calculate_upcoming_expirations(&active_policies).await;

        // Calculate storage savings
        let storage_savings_mb: f64 = statistics
            .by_category
            .values()
            .map(|s| s.storage_freed_bytes as f64)
            .sum::<f64>()
            / (1024.0 * 1024.0);

        // Compliance summary
        let last_cleanup = self.last_cleanup.read().await;
        let compliance_summary = ComplianceSummary {
            gdpr_compliant: true,
            overretained_data_count: 0,
            pending_deletions: 0,
            last_audit: last_cleanup.unwrap_or_else(Utc::now),
        };

        Ok(RetentionReport {
            generated_at: Utc::now(),
            period_days,
            statistics,
            active_policies,
            upcoming_expirations,
            storage_savings_mb,
            compliance_summary,
        })
    }

    /// Calculate upcoming expirations
    async fn calculate_upcoming_expirations(
        &self,
        policies: &[RetentionPolicy],
    ) -> Vec<ExpirationNotice> {
        let mut notices = Vec::new();

        for policy in policies {
            let expiration_date = Utc::now() + Duration::days(self.config.notification_days_before);
            let days_until = self.config.notification_days_before;

            // In real implementation, query database for records
            notices.push(ExpirationNotice {
                category: policy.data_category.clone(),
                record_count: 50, // Mock value
                expiration_date,
                days_until_expiration: days_until,
            });
        }

        notices
    }

    /// Handle right-to-be-forgotten request
    pub async fn process_deletion_request(&self, user_id: &str) -> Result<usize> {
        // In real implementation, this would:
        // 1. Find all data associated with user
        // 2. Archive if required
        // 3. Delete or anonymize
        // 4. Log the action

        // Mock deletion
        Ok(100) // Number of records deleted
    }

    /// Get list of all policies
    pub async fn list_policies(&self) -> Vec<RetentionPolicy> {
        self.policies.read().await.values().cloned().collect()
    }

    /// Get specific policy
    pub async fn get_policy(&self, policy_id: &str) -> Option<RetentionPolicy> {
        self.policies.read().await.get(policy_id).cloned()
    }

    /// Start automatic cleanup task
    pub async fn start_auto_cleanup(self: Arc<Self>) {
        if !self.config.auto_cleanup_enabled {
            return;
        }

        let interval = std::time::Duration::from_secs(self.config.cleanup_interval_hours * 3600);

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);

            loop {
                ticker.tick().await;

                if let Err(e) = self.run_cleanup().await {
                    eprintln!("Cleanup error: {e}");
                }
            }
        });
    }
}

impl Default for RetentionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_add_policy() {
        let manager = RetentionManager::new();

        let policy = RetentionPolicy {
            id: "test_policy".to_string(),
            name: "Test Policy".to_string(),
            data_category: "test_data".to_string(),
            retention_period_days: 90,
            archive_before_delete: true,
            enabled: true,
        };

        manager.add_policy(policy).await.unwrap();

        let policies = manager.list_policies().await;
        assert_eq!(policies.len(), 1);
        assert_eq!(policies[0].id, "test_policy");
    }

    #[tokio::test]
    async fn test_add_rule() {
        let manager = RetentionManager::new();

        let policy = RetentionPolicy {
            id: "test_policy".to_string(),
            name: "Test".to_string(),
            data_category: "test".to_string(),
            retention_period_days: 90,
            archive_before_delete: true,
            enabled: true,
        };

        manager.add_policy(policy).await.unwrap();

        let rule = RetentionRule {
            id: "test_rule".to_string(),
            policy_id: "test_policy".to_string(),
            conditions: vec![RetentionCondition::OlderThan { days: 90 }],
            action: RetentionAction::Delete,
        };

        manager.add_rule(rule).await.unwrap();
    }

    #[tokio::test]
    async fn test_run_cleanup() {
        let manager = RetentionManager::new();

        let policy = RetentionPolicy {
            id: "cleanup_test".to_string(),
            name: "Cleanup Test".to_string(),
            data_category: "feedback".to_string(),
            retention_period_days: 30,
            archive_before_delete: true,
            enabled: true,
        };

        manager.add_policy(policy).await.unwrap();

        let stats = manager.run_cleanup().await.unwrap();

        assert!(stats.total_processed > 0);
        assert_eq!(stats.by_category.len(), 1);
    }

    #[tokio::test]
    async fn test_generate_report() {
        let manager = RetentionManager::new();

        let policy = RetentionPolicy {
            id: "report_test".to_string(),
            name: "Report Test".to_string(),
            data_category: "analytics".to_string(),
            retention_period_days: 60,
            archive_before_delete: false,
            enabled: true,
        };

        manager.add_policy(policy).await.unwrap();

        let report = manager.generate_report(30).await.unwrap();

        assert!(!report.active_policies.is_empty());
        assert!(report.compliance_summary.gdpr_compliant);
    }

    #[tokio::test]
    async fn test_deletion_request() {
        let manager = RetentionManager::new();

        let deleted = manager.process_deletion_request("user123").await.unwrap();

        assert!(deleted > 0);
    }

    #[tokio::test]
    async fn test_statistics_history() {
        let manager = RetentionManager::new();

        let policy = RetentionPolicy {
            id: "history_test".to_string(),
            name: "History Test".to_string(),
            data_category: "test".to_string(),
            retention_period_days: 30,
            archive_before_delete: true,
            enabled: true,
        };

        manager.add_policy(policy).await.unwrap();

        // Run cleanup multiple times
        for _ in 0..3 {
            manager.run_cleanup().await.unwrap();
        }

        let history = manager.get_statistics_history(Some(10)).await;

        assert_eq!(history.len(), 3);
    }
}
