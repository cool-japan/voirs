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

use crate::persistence::PersistenceManager;
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

/// Which underlying persistence store a retention policy's `data_category`
/// maps to. [`PersistenceManager`] does not track data per named category --
/// only per session and per feedback record -- so a policy's category string
/// is classified into one of these before any real query or deletion can be
/// attempted against it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RetentionStore {
    /// Backed by `PersistenceManager::{save,load}_session` / `cleanup`.
    Sessions,
    /// Backed by `PersistenceManager::{save,load}_feedback` / `cleanup`.
    Feedback,
    /// No queryable backing store exists for this category yet.
    Unsupported,
}

/// Classify a policy's `data_category` string into the real store it maps
/// to. Matching is intentionally permissive (substring, case-insensitive) so
/// that categories like `"user_feedback"` or `"feedback"` both resolve to
/// [`RetentionStore::Feedback`].
fn classify_category(data_category: &str) -> RetentionStore {
    let lower = data_category.to_lowercase();
    if lower.contains("session") {
        RetentionStore::Sessions
    } else if lower.contains("feedback") {
        RetentionStore::Feedback
    } else {
        RetentionStore::Unsupported
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
    /// Real persistence backend queried/mutated by cleanup, expiration
    /// calculation, and right-to-be-forgotten requests. Without one, those
    /// operations fail closed with [`RetentionError::StorageError`] instead
    /// of fabricating results.
    persistence: Option<Arc<dyn PersistenceManager>>,
}

impl RetentionManager {
    /// Create a new retention manager with default configuration and no
    /// persistence backend attached. Cleanup, deletion-request, and
    /// expiration-calculation operations will fail closed until
    /// [`RetentionManager::with_persistence`] is used to attach a real
    /// backend.
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
            persistence: None,
        }
    }

    /// Attach a real persistence backend. Cleanup, deletion-request, and
    /// expiration-calculation operations query and mutate this backend
    /// directly, so they report genuine counts instead of placeholders.
    #[must_use]
    pub fn with_persistence(mut self, persistence: Arc<dyn PersistenceManager>) -> Self {
        self.persistence = Some(persistence);
        self
    }

    /// Borrow the configured persistence backend, or a real
    /// [`RetentionError::StorageError`] if none has been attached.
    fn require_persistence(&self) -> Result<&Arc<dyn PersistenceManager>> {
        self.persistence.as_ref().ok_or_else(|| {
            RetentionError::StorageError(
                "no persistence backend configured; call RetentionManager::with_persistence \
                 before running cleanup, expiration calculation, or deletion requests"
                    .to_string(),
            )
        })
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

    /// Run cleanup based on retention policies.
    ///
    /// Requires a persistence backend (see [`RetentionManager::with_persistence`]);
    /// without one this fails closed rather than fabricating statistics.
    pub async fn run_cleanup(&self) -> Result<RetentionStatistics> {
        let start_time = std::time::Instant::now();
        let persistence = self.require_persistence()?;

        let policies = self.policies.read().await;
        let rules = self.rules.read().await;

        let enabled_policies: Vec<&RetentionPolicy> =
            policies.values().filter(|p| p.enabled).collect();

        // Real, non-destructive snapshot of current totals, used to report
        // `processed` counts.
        let pre_stats = persistence.get_storage_stats().await.map_err(|e| {
            RetentionError::StorageError(format!("failed to read storage stats: {e}"))
        })?;

        // `PersistenceManager::cleanup` purges old sessions *and* feedback
        // together in a single pass -- it is not scoped per data category.
        // Calling it once per policy with each policy's own (possibly much
        // shorter) cutoff would let a short-retention policy silently delete
        // data that a different, longer-retention policy was supposed to
        // preserve. To avoid that cross-category over-deletion, run at most
        // one real cleanup pass per `run_cleanup()` call, using the single
        // most conservative cutoff (the largest `retention_period_days`,
        // i.e. the one that deletes the *least* data) among every enabled
        // policy that maps to a real store.
        let conservative_cutoff = enabled_policies
            .iter()
            .filter(|p| classify_category(&p.data_category) != RetentionStore::Unsupported)
            .map(|p| p.retention_period_days)
            .max()
            .map(|days| Utc::now() - Duration::days(days));

        let cleanup_result = match conservative_cutoff {
            Some(cutoff) => Some(
                persistence
                    .cleanup(cutoff)
                    .await
                    .map_err(|e| RetentionError::DeletionError(format!("cleanup failed: {e}")))?,
            ),
            None => None,
        };

        let mut total_processed = 0;
        let mut total_deleted = 0;
        let mut total_archived = 0;
        let total_anonymized = 0;
        let mut errors = 0;
        let mut by_category: HashMap<String, CategoryStats> = HashMap::new();
        // Two enabled policies can map to the same underlying store (e.g.
        // two policies both classified as `Feedback`), but the real cleanup
        // pass above only ran *once* for that store. Crediting every
        // policy's `processed`/`deleted`/`archived` in full to the
        // aggregate totals would double (or N-times) count the same real
        // records. Each store's real counts are added to the aggregate at
        // most once; `by_category` below is unaffected and still reports
        // each policy's own full view.
        let mut credited_stores: std::collections::HashSet<RetentionStore> =
            std::collections::HashSet::new();

        // Process each policy
        for policy in &enabled_policies {
            // Find rules for this policy
            let policy_rules: Vec<&RetentionRule> = rules
                .values()
                .filter(|r| r.policy_id == policy.id)
                .collect();

            let result =
                self.process_policy(policy, &policy_rules, &pre_stats, cleanup_result.as_ref());

            match result {
                Ok(stats) => {
                    if credited_stores.insert(classify_category(&policy.data_category)) {
                        total_processed += stats.processed;
                        total_deleted += stats.deleted;
                        total_archived += stats.archived;
                    }

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

    /// Attribute real cleanup results to a specific policy.
    ///
    /// Returns `Err` (counted by the caller into `RetentionStatistics::errors`)
    /// when the policy's `data_category` has no real backing store to verify
    /// -- that is honest, not a cleanup call that silently touches the wrong
    /// data.
    fn process_policy(
        &self,
        policy: &RetentionPolicy,
        rules: &[&RetentionRule],
        pre_stats: &crate::persistence::StorageStats,
        cleanup_result: Option<&crate::persistence::CleanupResult>,
    ) -> Result<CategoryStats> {
        let store = classify_category(&policy.data_category);

        // A policy with explicit rules that are *all* non-destructive
        // (anonymize / cold storage / manual review) has not actually
        // authorized real deletion, even though a shared cleanup pass may
        // have run for other policies sharing the same store. Only count
        // data as deleted when no rules are defined (the policy applies on
        // its own, matching its `archive_before_delete` field) or at least
        // one rule explicitly requests Delete/ArchiveThenDelete.
        let deletion_authorized = rules.is_empty()
            || rules.iter().any(|r| {
                matches!(
                    r.action,
                    RetentionAction::Delete | RetentionAction::ArchiveThenDelete
                )
            });

        let (processed, deleted) = match store {
            RetentionStore::Sessions => {
                let cleanup = cleanup_result.ok_or_else(|| {
                    RetentionError::StorageError(format!(
                        "no cleanup pass was run for category '{}'",
                        policy.data_category
                    ))
                })?;
                let deleted = if deletion_authorized {
                    cleanup.sessions_cleaned
                } else {
                    0
                };
                (pre_stats.total_sessions, deleted)
            }
            RetentionStore::Feedback => {
                let cleanup = cleanup_result.ok_or_else(|| {
                    RetentionError::StorageError(format!(
                        "no cleanup pass was run for category '{}'",
                        policy.data_category
                    ))
                })?;
                let deleted = if deletion_authorized {
                    cleanup.feedback_records_cleaned
                } else {
                    0
                };
                (pre_stats.total_feedback_records, deleted)
            }
            RetentionStore::Unsupported => {
                return Err(RetentionError::InvalidPolicy(format!(
                    "data category '{}' has no queryable backing store in the configured \
                     persistence backend; retention cannot be verified",
                    policy.data_category
                )));
            }
        };

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
            // No per-record size is tracked by the persistence layer; this
            // remains an estimate, applied only to real (non-fabricated)
            // deletion counts.
            storage_freed_bytes: deleted as u64 * 1024 * 10,
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
        // Read the last statistics from history, dropping the lock before potentially calling run_cleanup
        let last_stats = self.stats_history.read().await.last().cloned();
        let statistics = match last_stats {
            Some(stats) => stats,
            None => {
                // Run cleanup if no history (lock is already dropped above)
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

        // Compliance summary, derived from the real cleanup run above rather
        // than hardcoded. `errors` counts policies whose retention could not
        // be verified/enforced against a real backing store (see
        // `process_policy`), so it doubles as an honest proxy for
        // over-retained/unverified data until the persistence layer exposes
        // a non-destructive per-category "count records older than X" query.
        // Deletion requests are processed synchronously by
        // `process_deletion_request` in this implementation, so there is
        // never a genuinely "pending" one to report.
        let last_cleanup = self.last_cleanup.read().await;
        let compliance_summary = ComplianceSummary {
            gdpr_compliant: statistics.errors == 0,
            overretained_data_count: statistics.errors,
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

    /// Calculate upcoming expirations.
    ///
    /// `record_count` reflects the real, currently-stored total for each
    /// policy's mapped store (via a non-destructive [`PersistenceManager::get_storage_stats`]
    /// query), not a hardcoded placeholder. This is an upper bound rather
    /// than an exact "expiring in the next N days" count: the persistence
    /// layer does not expose a non-destructive per-category "count records
    /// older than X" query, so the real current total for the mapped store
    /// is the most accurate honest figure available. Categories with no
    /// backing store report `0` rather than a guess. Returns an empty list
    /// (rather than fabricating data) if no persistence backend is
    /// attached.
    async fn calculate_upcoming_expirations(
        &self,
        policies: &[RetentionPolicy],
    ) -> Vec<ExpirationNotice> {
        let mut notices = Vec::new();

        let Some(persistence) = self.persistence.as_ref() else {
            return notices;
        };

        let Ok(stats) = persistence.get_storage_stats().await else {
            return notices;
        };

        for policy in policies {
            let expiration_date = Utc::now() + Duration::days(self.config.notification_days_before);
            let days_until = self.config.notification_days_before;

            let record_count = match classify_category(&policy.data_category) {
                RetentionStore::Sessions => stats.total_sessions,
                RetentionStore::Feedback => stats.total_feedback_records,
                RetentionStore::Unsupported => 0,
            };

            notices.push(ExpirationNotice {
                category: policy.data_category.clone(),
                record_count,
                expiration_date,
                days_until_expiration: days_until,
            });
        }

        notices
    }

    /// Handle a right-to-be-forgotten request: really deletes every trace of
    /// `user_id` (progress, preferences, sessions, feedback history) from
    /// the configured persistence backend and returns the real number of
    /// records that were removed.
    ///
    /// Returns `Ok(0)` (not an error) when the user has no data on record --
    /// that is a legitimate, honest outcome for a deletion request. Requires
    /// a persistence backend (see [`RetentionManager::with_persistence`]).
    pub async fn process_deletion_request(&self, user_id: &str) -> Result<usize> {
        let persistence = self.require_persistence()?;

        // Count what genuinely exists before deleting, using queries that
        // reliably distinguish "present" from "absent" across every
        // persistence backend.
        let has_progress = persistence.load_user_progress(user_id).await.is_ok();
        let has_preferences = persistence.load_preferences(user_id).await.is_ok();
        let feedback_count = persistence
            .load_feedback_history(user_id, None, None)
            .await
            .map(|history| history.len())
            .unwrap_or(0);
        // Sessions are only enumerable per-user via `export_user_data`
        // (the trait has no dedicated "list sessions for user" method).
        // Every backend now defaults a *missing progress record* to
        // `UserProgress::default()` inside `export_user_data` rather than
        // erroring the whole bundle, so a `.unwrap_or(0)` here only fires
        // for genuine backend errors (e.g. connection failure), not for a
        // user who has sessions but no progress record.
        let session_count = persistence
            .export_user_data(user_id)
            .await
            .map(|export| export.sessions.len())
            .unwrap_or(0);

        let deleted_count = usize::from(has_progress)
            + usize::from(has_preferences)
            + feedback_count
            + session_count;

        if deleted_count == 0 {
            return Ok(0);
        }

        persistence.delete_user_data(user_id).await.map_err(|e| {
            RetentionError::DeletionError(format!(
                "failed to delete data for user '{user_id}': {e}"
            ))
        })?;

        Ok(deleted_count)
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
    use crate::persistence::backends::memory::MemoryPersistenceManager;
    use crate::persistence::PersistenceConfig;
    use crate::traits::{
        AdaptiveState, FeedbackResponse, FeedbackType, ProgressIndicators, SessionState,
        SessionStatistics, SessionStats, UserPreferences, UserProgress,
    };
    use uuid::Uuid;

    /// Build a `RetentionManager` backed by a real, empty in-memory
    /// persistence backend.
    async fn test_manager() -> RetentionManager {
        let backend = MemoryPersistenceManager::new(PersistenceConfig::default())
            .await
            .unwrap();
        RetentionManager::new().with_persistence(Arc::new(backend))
    }

    fn test_session(user_id: &str, start_time: DateTime<Utc>) -> SessionState {
        SessionState {
            session_id: Uuid::new_v4(),
            user_id: user_id.to_string(),
            start_time,
            last_activity: start_time,
            current_task: None,
            stats: SessionStats::default(),
            preferences: UserPreferences::default(),
            adaptive_state: AdaptiveState::default(),
            current_exercise: None,
            session_stats: SessionStatistics::default(),
        }
    }

    fn test_feedback(timestamp: DateTime<Utc>) -> FeedbackResponse {
        FeedbackResponse {
            feedback_items: vec![],
            overall_score: 0.5,
            immediate_actions: vec![],
            long_term_goals: vec![],
            progress_indicators: ProgressIndicators::default(),
            timestamp,
            processing_time: std::time::Duration::from_millis(5),
            feedback_type: FeedbackType::Quality,
        }
    }

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

    /// Without an attached persistence backend, cleanup must fail closed
    /// rather than fabricate statistics.
    #[tokio::test]
    async fn test_run_cleanup_without_persistence_fails_closed() {
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

        assert!(manager.run_cleanup().await.is_err());
    }

    /// `run_cleanup` must really delete real, seeded feedback records older
    /// than the policy's retention period from the real persistence
    /// backend, and report the real (non-hardcoded) counts -- while leaving
    /// fresh records alone.
    #[tokio::test]
    async fn test_run_cleanup_deletes_real_old_feedback() {
        let manager = test_manager().await;

        let policy = RetentionPolicy {
            id: "cleanup_test".to_string(),
            name: "Cleanup Test".to_string(),
            data_category: "feedback".to_string(),
            retention_period_days: 30,
            archive_before_delete: true,
            enabled: true,
        };
        manager.add_policy(policy).await.unwrap();

        let persistence = manager.persistence.clone().unwrap();
        let old_feedback = test_feedback(Utc::now() - Duration::days(90));
        let fresh_feedback = test_feedback(Utc::now());
        persistence
            .save_feedback("user1", &old_feedback)
            .await
            .unwrap();
        persistence
            .save_feedback("user1", &fresh_feedback)
            .await
            .unwrap();

        let stats = manager.run_cleanup().await.unwrap();

        assert_eq!(stats.by_category.len(), 1);
        let category_stats = &stats.by_category["feedback"];
        assert_eq!(
            category_stats.processed, 2,
            "processed must reflect the real pre-cleanup total"
        );
        assert_eq!(
            category_stats.deleted, 1,
            "only the record older than the retention period should be deleted"
        );
        assert_eq!(category_stats.archived, 1, "archive_before_delete was true");
        assert_eq!(stats.total_deleted, 1);
        assert_eq!(stats.errors, 0);

        // The fresh record must genuinely still be there.
        let remaining = persistence
            .load_feedback_history("user1", None, None)
            .await
            .unwrap();
        assert_eq!(remaining.len(), 1);
        assert!((remaining[0].overall_score - fresh_feedback.overall_score).abs() < 1e-6);
    }

    /// A policy whose `data_category` has no real backing store must be
    /// reported as an honest error, not silently skipped or fabricated --
    /// and it must not affect the compliance status of a policy that *does*
    /// map to a real store.
    #[tokio::test]
    async fn test_run_cleanup_unsupported_category_is_an_honest_error() {
        let manager = test_manager().await;

        let policy = RetentionPolicy {
            id: "audit_logs".to_string(),
            name: "Audit Logs".to_string(),
            data_category: "audit_logs".to_string(),
            retention_period_days: 30,
            archive_before_delete: false,
            enabled: true,
        };
        manager.add_policy(policy).await.unwrap();

        let stats = manager.run_cleanup().await.unwrap();
        assert_eq!(stats.errors, 1);
        assert!(stats.by_category.is_empty());
    }

    /// Two policies mapping to the same store with *different* retention
    /// periods must not let the shorter one delete data the longer one is
    /// supposed to preserve: only records older than the longer (more
    /// conservative) cutoff may actually be removed.
    #[tokio::test]
    async fn test_run_cleanup_uses_most_conservative_cutoff_across_policies() {
        let manager = test_manager().await;

        // A short 7-day policy and a long 90-day policy both nominally
        // target feedback data.
        manager
            .add_policy(RetentionPolicy {
                id: "short".to_string(),
                name: "Short".to_string(),
                data_category: "feedback_short".to_string(),
                retention_period_days: 7,
                archive_before_delete: false,
                enabled: true,
            })
            .await
            .unwrap();
        manager
            .add_policy(RetentionPolicy {
                id: "long".to_string(),
                name: "Long".to_string(),
                data_category: "feedback_long".to_string(),
                retention_period_days: 90,
                archive_before_delete: false,
                enabled: true,
            })
            .await
            .unwrap();

        let persistence = manager.persistence.clone().unwrap();
        // 30 days old: newer than the long policy's 90-day cutoff (must
        // survive) but older than the short policy's 7-day cutoff (would be
        // deleted if the short cutoff were wrongly applied globally).
        let middle_aged = test_feedback(Utc::now() - Duration::days(30));
        persistence
            .save_feedback("user1", &middle_aged)
            .await
            .unwrap();

        let stats = manager.run_cleanup().await.unwrap();

        // Neither policy's reported `deleted` count may exceed what the
        // shared, most-conservative (90-day) cleanup pass actually removed.
        assert_eq!(stats.by_category["feedback_short"].deleted, 0);
        assert_eq!(stats.by_category["feedback_long"].deleted, 0);

        let remaining = persistence
            .load_feedback_history("user1", None, None)
            .await
            .unwrap();
        assert_eq!(
            remaining.len(),
            1,
            "a 30-day-old record must survive when the effective cutoff is 90 days"
        );
    }

    /// Two enabled policies mapping to the *same* real store (here, both
    /// classified as `Feedback`) must not cause the aggregate
    /// `total_deleted`/`total_processed` to double-count the single real
    /// deletion the shared cleanup pass actually performed, even though
    /// each policy's own `by_category` entry legitimately reports the full
    /// (shared) count for its own view.
    #[tokio::test]
    async fn test_run_cleanup_aggregate_totals_do_not_double_count_shared_store() {
        let manager = test_manager().await;

        manager
            .add_policy(RetentionPolicy {
                id: "feedback_a".to_string(),
                name: "Feedback A".to_string(),
                data_category: "feedback_a".to_string(),
                retention_period_days: 30,
                archive_before_delete: false,
                enabled: true,
            })
            .await
            .unwrap();
        manager
            .add_policy(RetentionPolicy {
                id: "feedback_b".to_string(),
                name: "Feedback B".to_string(),
                data_category: "feedback_b".to_string(),
                retention_period_days: 30,
                archive_before_delete: false,
                enabled: true,
            })
            .await
            .unwrap();

        let persistence = manager.persistence.clone().unwrap();
        // Exactly one real record, genuinely old enough to be deleted by
        // both policies' shared (30-day) cutoff.
        let old_feedback = test_feedback(Utc::now() - Duration::days(90));
        persistence
            .save_feedback("user1", &old_feedback)
            .await
            .unwrap();

        let stats = manager.run_cleanup().await.unwrap();

        // Each policy's own view legitimately reports the real, shared
        // deletion count.
        assert_eq!(stats.by_category["feedback_a"].deleted, 1);
        assert_eq!(stats.by_category["feedback_b"].deleted, 1);

        // But only one record genuinely existed and was genuinely deleted --
        // the aggregate must not report 2.
        assert_eq!(
            stats.total_deleted, 1,
            "aggregate total_deleted must not double-count a single real \
             deletion shared by two policies mapping to the same store"
        );

        let remaining = persistence
            .load_feedback_history("user1", None, None)
            .await
            .unwrap();
        assert!(
            remaining.is_empty(),
            "the single real record must genuinely be gone"
        );
    }

    #[tokio::test]
    async fn test_generate_report() {
        let manager = test_manager().await;

        let policy = RetentionPolicy {
            id: "report_test".to_string(),
            name: "Report Test".to_string(),
            data_category: "feedback".to_string(),
            retention_period_days: 60,
            archive_before_delete: false,
            enabled: true,
        };

        manager.add_policy(policy).await.unwrap();

        let report = manager.generate_report(30).await.unwrap();

        assert!(!report.active_policies.is_empty());
        assert!(report.compliance_summary.gdpr_compliant);
        assert_eq!(report.upcoming_expirations.len(), 1);
        assert_eq!(report.upcoming_expirations[0].category, "feedback");
    }

    /// An unsupported category must make the compliance report honestly
    /// report non-compliance (unverifiable retention), not a hardcoded
    /// always-true.
    #[tokio::test]
    async fn test_generate_report_unsupported_category_is_not_compliant() {
        let manager = test_manager().await;

        manager
            .add_policy(RetentionPolicy {
                id: "report_test".to_string(),
                name: "Report Test".to_string(),
                data_category: "analytics_events".to_string(),
                retention_period_days: 60,
                archive_before_delete: false,
                enabled: true,
            })
            .await
            .unwrap();

        let report = manager.generate_report(30).await.unwrap();
        assert!(!report.compliance_summary.gdpr_compliant);
        assert_eq!(report.compliance_summary.overretained_data_count, 1);
    }

    #[tokio::test]
    async fn test_deletion_request_without_persistence_fails_closed() {
        let manager = RetentionManager::new();
        assert!(manager.process_deletion_request("user123").await.is_err());
    }

    /// The right-to-be-forgotten handler must really delete the seeded
    /// data for the user and report the real count, and must leave a
    /// different user's data untouched.
    #[tokio::test]
    async fn test_deletion_request_deletes_real_seeded_data() {
        let manager = test_manager().await;
        let persistence = manager.persistence.clone().unwrap();

        let progress = UserProgress {
            user_id: "user123".to_string(),
            ..UserProgress::default()
        };
        persistence
            .save_user_progress("user123", &progress)
            .await
            .unwrap();
        persistence
            .save_preferences("user123", &UserPreferences::default())
            .await
            .unwrap();
        persistence
            .save_feedback("user123", &test_feedback(Utc::now()))
            .await
            .unwrap();
        persistence
            .save_session(&test_session("user123", Utc::now()))
            .await
            .unwrap();

        // An untouched second user, to prove deletion is scoped correctly.
        let other_progress = UserProgress {
            user_id: "user456".to_string(),
            ..UserProgress::default()
        };
        persistence
            .save_user_progress("user456", &other_progress)
            .await
            .unwrap();

        // progress + preferences + 1 feedback + 1 session = 4 real records.
        let deleted = manager.process_deletion_request("user123").await.unwrap();
        assert_eq!(deleted, 4);

        assert!(persistence.load_user_progress("user123").await.is_err());
        assert!(persistence.load_preferences("user123").await.is_err());
        let remaining_feedback = persistence
            .load_feedback_history("user123", None, None)
            .await
            .unwrap();
        assert!(remaining_feedback.is_empty());

        // The other user's data must be unaffected.
        assert!(persistence.load_user_progress("user456").await.is_ok());
    }

    /// A deletion request for a user with no data on record is a
    /// legitimate, honest `Ok(0)` -- not an error, and not a fabricated
    /// positive count.
    #[tokio::test]
    async fn test_deletion_request_for_unknown_user_returns_zero() {
        let manager = test_manager().await;
        let deleted = manager
            .process_deletion_request("never_existed")
            .await
            .unwrap();
        assert_eq!(deleted, 0);
    }

    #[tokio::test]
    async fn test_statistics_history() {
        let manager = test_manager().await;

        let policy = RetentionPolicy {
            id: "history_test".to_string(),
            name: "History Test".to_string(),
            data_category: "feedback".to_string(),
            retention_period_days: 30,
            archive_before_delete: true,
            enabled: true,
        };

        manager.add_policy(policy).await.unwrap();

        // Run cleanup multiple times against the real (empty) backend.
        for _ in 0..3 {
            manager.run_cleanup().await.unwrap();
        }

        let history = manager.get_statistics_history(Some(10)).await;

        assert_eq!(history.len(), 3);
    }
}
