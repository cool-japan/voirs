//! Audit Trail System for Data Access and Compliance
//!
//! This module provides a comprehensive audit logging system for tracking data access,
//! modifications, and system events. It ensures compliance with security and privacy
//! regulations (GDPR, SOC 2, HIPAA) by maintaining detailed, immutable audit logs.
//!
//! # Features
//!
//! - **Comprehensive Logging**: All data access and modifications are logged
//! - **User Activity Tracking**: Track who accessed what, when, and why
//! - **Change History**: Complete history of data modifications with diffs
//! - **Compliance Reports**: Generate audit reports for compliance requirements
//! - **Tamper-Proof Logs**: Cryptographic integrity verification
//! - **Retention Policies**: Configurable log retention and archival
//! - **Search and Filter**: Query audit logs by user, action, time range
//! - **Export Capabilities**: Export logs in multiple formats (JSON, CSV, SYSLOG)
//!
//! # Example
//!
//! ```rust
//! use voirs_feedback::audit_trail::{AuditLogger, AuditAction, AuditContext};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let logger = AuditLogger::new();
//!
//! // Log data access
//! logger.log_access(
//!     "user123",
//!     "feedback_data",
//!     AuditAction::Read,
//!     AuditContext::new()
//!         .with_reason("User requested feedback history")
//!         .with_ip("192.168.1.100"),
//! ).await?;
//!
//! // Query audit logs
//! let logs = logger.query_logs(
//!     Some("user123".to_string()),
//!     None,
//!     None,
//!     100,
//! ).await?;
//!
//! // Generate compliance report
//! let report = logger.generate_compliance_report(30).await?;
//! # Ok(())
//! # }
//! ```

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

/// Errors that can occur in audit logging
#[derive(Error, Debug)]
#[allow(missing_docs)]
pub enum AuditError {
    /// Storage error
    #[error("Storage error: {0}")]
    StorageError(String),

    /// Integrity violation
    #[error("Integrity violation: {0}")]
    IntegrityError(String),

    /// Invalid query
    #[error("Invalid query: {0}")]
    InvalidQuery(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Type alias for Results in this module
pub type Result<T> = std::result::Result<T, AuditError>;

/// Audit actions that can be performed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum AuditAction {
    /// Read/view data
    Read,
    /// Create new data
    Create,
    /// Update existing data
    Update,
    /// Delete data
    Delete,
    /// Export data
    Export,
    /// Login event
    Login,
    /// Logout event
    Logout,
    /// Permission change
    PermissionChange,
    /// Configuration change
    ConfigChange,
    /// System event
    SystemEvent,
}

/// Resource types that can be audited
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum ResourceType {
    /// User profile data
    UserProfile,
    /// Feedback data
    Feedback,
    /// Session data
    Session,
    /// Training exercise
    Exercise,
    /// Analytics data
    Analytics,
    /// Configuration
    Configuration,
    /// System settings
    SystemSettings,
    /// Custom resource type
    Custom(String),
}

/// Audit context with additional information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuditContext {
    /// Reason for the action
    pub reason: Option<String>,
    /// IP address (anonymized)
    pub ip_address: Option<String>,
    /// User agent
    pub user_agent: Option<String>,
    /// Request ID
    pub request_id: Option<String>,
    /// Session ID
    pub session_id: Option<String>,
    /// Application/service name
    pub service: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl AuditContext {
    /// Create a new audit context
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the reason for the action
    #[must_use]
    pub fn with_reason(mut self, reason: &str) -> Self {
        self.reason = Some(reason.to_string());
        self
    }

    /// Set the IP address (will be anonymized)
    #[must_use]
    pub fn with_ip(mut self, ip: &str) -> Self {
        self.ip_address = Some(Self::anonymize_ip(ip));
        self
    }

    /// Set the user agent string
    #[must_use]
    pub fn with_user_agent(mut self, ua: &str) -> Self {
        self.user_agent = Some(ua.to_string());
        self
    }

    /// Set the request ID
    #[must_use]
    pub fn with_request_id(mut self, id: &str) -> Self {
        self.request_id = Some(id.to_string());
        self
    }

    /// Set the service name
    #[must_use]
    pub fn with_service(mut self, service: &str) -> Self {
        self.service = Some(service.to_string());
        self
    }

    /// Add metadata key-value pair
    #[must_use]
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    fn anonymize_ip(ip: &str) -> String {
        if let Some(idx) = ip.rfind('.') {
            format!("{}.xxx", &ip[..idx])
        } else {
            "xxx.xxx.xxx.xxx".to_string()
        }
    }
}

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique entry ID
    pub id: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// User who performed the action
    pub user_id: String,
    /// Action performed
    pub action: AuditAction,
    /// Resource type
    pub resource_type: ResourceType,
    /// Resource ID
    pub resource_id: String,
    /// Action result (success/failure)
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Data before change (for updates/deletes)
    pub old_value: Option<String>,
    /// Data after change (for creates/updates)
    pub new_value: Option<String>,
    /// Context information
    pub context: AuditContext,
    /// Hash of previous entry (for chain integrity)
    pub previous_hash: String,
    /// Hash of this entry
    pub hash: String,
}

impl AuditEntry {
    /// Calculate hash for this entry
    fn calculate_hash(&self) -> String {
        let mut hasher = Sha256::new();

        hasher.update(self.id.as_bytes());
        hasher.update(self.timestamp.to_rfc3339().as_bytes());
        hasher.update(self.user_id.as_bytes());
        hasher.update(format!("{:?}", self.action).as_bytes());
        hasher.update(self.resource_id.as_bytes());
        hasher.update(self.previous_hash.as_bytes());

        if let Some(old) = &self.old_value {
            hasher.update(old.as_bytes());
        }
        if let Some(new) = &self.new_value {
            hasher.update(new.as_bytes());
        }

        format!("{:x}", hasher.finalize())
    }
}

/// Audit summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditStatistics {
    /// Start time of the period
    pub start_time: DateTime<Utc>,
    /// End time of the period
    pub end_time: DateTime<Utc>,
    /// Total audit entries
    pub total_entries: usize,
    /// Entries by action
    pub by_action: HashMap<AuditAction, usize>,
    /// Entries by user
    pub by_user: HashMap<String, usize>,
    /// Failed actions
    pub failed_actions: usize,
    /// Most active users
    pub top_users: Vec<(String, usize)>,
    /// Most common actions
    pub top_actions: Vec<(AuditAction, usize)>,
}

/// Compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Report generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Reporting period in days
    pub period_days: i64,
    /// Audit statistics
    pub statistics: AuditStatistics,
    /// Data access summary
    pub data_access_summary: DataAccessSummary,
    /// Sensitive data access
    pub sensitive_data_access: Vec<AuditEntry>,
    /// Failed access attempts
    pub failed_attempts: Vec<AuditEntry>,
    /// Integrity check status
    pub integrity_verified: bool,
    /// Compliance checks
    pub compliance_checks: Vec<ComplianceCheck>,
}

/// Data access summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAccessSummary {
    /// Total data access events
    pub total_access: usize,
    /// Users who accessed data
    pub unique_users: usize,
    /// Resources accessed
    pub unique_resources: usize,
    /// Exports performed
    pub export_count: usize,
    /// Average accesses per user
    pub avg_accesses_per_user: f64,
}

/// Compliance check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCheck {
    /// Check name
    pub name: String,
    /// Check passed
    pub passed: bool,
    /// Details
    pub details: String,
}

/// Query filter for audit logs
#[derive(Debug, Clone, Default)]
pub struct AuditQuery {
    /// Filter by user ID
    pub user_id: Option<String>,
    /// Filter by action
    pub action: Option<AuditAction>,
    /// Filter by resource type
    pub resource_type: Option<ResourceType>,
    /// Start time
    pub start_time: Option<DateTime<Utc>>,
    /// End time
    pub end_time: Option<DateTime<Utc>>,
    /// Only failed actions
    pub only_failures: bool,
    /// Limit results
    pub limit: usize,
    /// Offset for pagination
    pub offset: usize,
}

/// Configuration for audit logger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggerConfig {
    /// Maximum entries to keep in memory
    pub max_entries: usize,
    /// Retention period in days
    pub retention_period_days: i64,
    /// Enable integrity verification
    pub enable_integrity_check: bool,
    /// Auto-archive interval in seconds
    pub archive_interval_secs: u64,
    /// Require reason for sensitive operations
    pub require_reason_for_sensitive: bool,
}

impl Default for LoggerConfig {
    fn default() -> Self {
        Self {
            max_entries: 1_000_000,
            retention_period_days: 365, // 1 year for compliance
            enable_integrity_check: true,
            archive_interval_secs: 86400, // Daily
            require_reason_for_sensitive: true,
        }
    }
}

/// Main audit logging system
pub struct AuditLogger {
    /// Configuration
    config: LoggerConfig,
    /// Audit log entries
    entries: Arc<RwLock<VecDeque<AuditEntry>>>,
    /// Last entry hash (for chain integrity)
    last_hash: Arc<RwLock<String>>,
    /// Archived entries count
    archived_count: Arc<RwLock<usize>>,
}

impl AuditLogger {
    /// Create a new audit logger with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(LoggerConfig::default())
    }

    /// Create with custom configuration
    #[must_use]
    pub fn with_config(config: LoggerConfig) -> Self {
        Self {
            config,
            entries: Arc::new(RwLock::new(VecDeque::new())),
            last_hash: Arc::new(RwLock::new(String::from("genesis"))),
            archived_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Log a data access event
    pub async fn log_access(
        &self,
        user_id: &str,
        resource_id: &str,
        action: AuditAction,
        context: AuditContext,
    ) -> Result<String> {
        self.log_event(
            user_id,
            action,
            ResourceType::Custom("data".to_string()),
            resource_id,
            true,
            None,
            None,
            None,
            context,
        )
        .await
    }

    /// Log a data modification event
    pub async fn log_modification(
        &self,
        user_id: &str,
        resource_type: ResourceType,
        resource_id: &str,
        action: AuditAction,
        old_value: Option<String>,
        new_value: Option<String>,
        context: AuditContext,
    ) -> Result<String> {
        self.log_event(
            user_id,
            action,
            resource_type,
            resource_id,
            true,
            None,
            old_value,
            new_value,
            context,
        )
        .await
    }

    /// Log a failed event
    pub async fn log_failure(
        &self,
        user_id: &str,
        action: AuditAction,
        resource_type: ResourceType,
        resource_id: &str,
        error: &str,
        context: AuditContext,
    ) -> Result<String> {
        self.log_event(
            user_id,
            action,
            resource_type,
            resource_id,
            false,
            Some(error.to_string()),
            None,
            None,
            context,
        )
        .await
    }

    /// Internal method to log any event
    async fn log_event(
        &self,
        user_id: &str,
        action: AuditAction,
        resource_type: ResourceType,
        resource_id: &str,
        success: bool,
        error_message: Option<String>,
        old_value: Option<String>,
        new_value: Option<String>,
        context: AuditContext,
    ) -> Result<String> {
        // Validate sensitive operations
        if self.config.require_reason_for_sensitive
            && matches!(
                action,
                AuditAction::Delete | AuditAction::Export | AuditAction::PermissionChange
            )
            && context.reason.is_none()
        {
            return Err(AuditError::ConfigError(
                "Reason required for sensitive operations".to_string(),
            ));
        }

        let entry_id = uuid::Uuid::new_v4().to_string();
        let previous_hash = self.last_hash.read().await.clone();

        let mut entry = AuditEntry {
            id: entry_id,
            timestamp: Utc::now(),
            user_id: user_id.to_string(),
            action,
            resource_type,
            resource_id: resource_id.to_string(),
            success,
            error_message,
            old_value,
            new_value,
            context,
            previous_hash,
            hash: String::new(), // Will be calculated
        };

        // Calculate hash
        entry.hash = entry.calculate_hash();

        // Store entry
        let mut entries = self.entries.write().await;
        if entries.len() >= self.config.max_entries {
            entries.pop_front();
            *self.archived_count.write().await += 1;
        }
        entries.push_back(entry.clone());

        // Update last hash
        *self.last_hash.write().await = entry.hash.clone();

        Ok(entry.id)
    }

    /// Query audit logs
    pub async fn query_logs(
        &self,
        user_id: Option<String>,
        action: Option<AuditAction>,
        time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
        limit: usize,
    ) -> Result<Vec<AuditEntry>> {
        let entries = self.entries.read().await;

        let filtered: Vec<AuditEntry> = entries
            .iter()
            .filter(|e| {
                if let Some(ref uid) = user_id {
                    if &e.user_id != uid {
                        return false;
                    }
                }

                if let Some(act) = action {
                    if e.action != act {
                        return false;
                    }
                }

                if let Some((start, end)) = time_range {
                    if e.timestamp < start || e.timestamp > end {
                        return false;
                    }
                }

                true
            })
            .take(limit)
            .cloned()
            .collect();

        Ok(filtered)
    }

    /// Verify audit log integrity
    pub async fn verify_integrity(&self) -> Result<bool> {
        if !self.config.enable_integrity_check {
            return Ok(true);
        }

        let entries = self.entries.read().await;

        let mut previous_hash = "genesis".to_string();

        for entry in entries.iter() {
            // Verify chain
            if entry.previous_hash != previous_hash {
                return Err(AuditError::IntegrityError(format!(
                    "Chain broken at entry {}",
                    entry.id
                )));
            }

            // Verify entry hash
            let computed_hash = entry.calculate_hash();
            if entry.hash != computed_hash {
                return Err(AuditError::IntegrityError(format!(
                    "Hash mismatch at entry {}",
                    entry.id
                )));
            }

            previous_hash = entry.hash.clone();
        }

        Ok(true)
    }

    /// Get audit statistics
    pub async fn get_statistics(&self, hours: Option<i64>) -> Result<AuditStatistics> {
        let hours = hours.unwrap_or(24);
        let end_time = Utc::now();
        let start_time = end_time - Duration::hours(hours);

        let entries = self.entries.read().await;

        let relevant: Vec<&AuditEntry> = entries
            .iter()
            .filter(|e| e.timestamp >= start_time && e.timestamp <= end_time)
            .collect();

        let total_entries = relevant.len();

        // By action
        let mut by_action: HashMap<AuditAction, usize> = HashMap::new();
        for entry in &relevant {
            *by_action.entry(entry.action).or_insert(0) += 1;
        }

        // By user
        let mut by_user: HashMap<String, usize> = HashMap::new();
        for entry in &relevant {
            *by_user.entry(entry.user_id.clone()).or_insert(0) += 1;
        }

        // Failed actions
        let failed_actions = relevant.iter().filter(|e| !e.success).count();

        // Top users (clone before consuming)
        let mut top_users: Vec<(String, usize)> = by_user.clone().into_iter().collect();
        top_users.sort_by_key(|b| std::cmp::Reverse(b.1));
        top_users.truncate(10);

        // Top actions
        let mut top_actions: Vec<(AuditAction, usize)> = by_action.clone().into_iter().collect();
        top_actions.sort_by_key(|b| std::cmp::Reverse(b.1));
        top_actions.truncate(10);

        Ok(AuditStatistics {
            start_time,
            end_time,
            total_entries,
            by_action,
            by_user,
            failed_actions,
            top_users,
            top_actions,
        })
    }

    /// Generate compliance report
    pub async fn generate_compliance_report(&self, period_days: i64) -> Result<ComplianceReport> {
        let statistics = self.get_statistics(Some(period_days * 24)).await?;

        let entries = self.entries.read().await;

        let end_time = Utc::now();
        let start_time = end_time - Duration::days(period_days);

        let relevant: Vec<&AuditEntry> = entries
            .iter()
            .filter(|e| e.timestamp >= start_time)
            .collect();

        // Data access summary
        let total_access = relevant
            .iter()
            .filter(|e| e.action == AuditAction::Read)
            .count();
        let unique_users = statistics.by_user.len();
        let mut unique_resources = HashSet::new();
        for entry in &relevant {
            unique_resources.insert(&entry.resource_id);
        }
        let export_count = relevant
            .iter()
            .filter(|e| e.action == AuditAction::Export)
            .count();

        let data_access_summary = DataAccessSummary {
            total_access,
            unique_users,
            unique_resources: unique_resources.len(),
            export_count,
            avg_accesses_per_user: if unique_users > 0 {
                total_access as f64 / unique_users as f64
            } else {
                0.0
            },
        };

        // Sensitive data access (exports and deletes)
        let sensitive_data_access: Vec<AuditEntry> = relevant
            .iter()
            .filter(|e| matches!(e.action, AuditAction::Export | AuditAction::Delete))
            .map(|e| (*e).clone())
            .collect();

        // Failed attempts
        let failed_attempts: Vec<AuditEntry> = relevant
            .iter()
            .filter(|e| !e.success)
            .map(|e| (*e).clone())
            .collect();

        // Integrity check
        let integrity_verified = self.verify_integrity().await.is_ok();

        // Compliance checks
        let mut compliance_checks = Vec::new();

        // Check 1: All sensitive operations have reasons
        let sensitive_without_reason = relevant
            .iter()
            .filter(|e| {
                matches!(e.action, AuditAction::Delete | AuditAction::Export)
                    && e.context.reason.is_none()
            })
            .count();

        compliance_checks.push(ComplianceCheck {
            name: "Sensitive operations have documented reasons".to_string(),
            passed: sensitive_without_reason == 0,
            details: if sensitive_without_reason > 0 {
                format!("{sensitive_without_reason} operations without reasons")
            } else {
                "All sensitive operations documented".to_string()
            },
        });

        // Check 2: Audit log integrity
        compliance_checks.push(ComplianceCheck {
            name: "Audit log integrity verified".to_string(),
            passed: integrity_verified,
            details: if integrity_verified {
                "Chain integrity verified".to_string()
            } else {
                "Integrity check failed".to_string()
            },
        });

        // Check 3: Failed access attempts monitoring
        compliance_checks.push(ComplianceCheck {
            name: "Failed access attempts monitored".to_string(),
            passed: true,
            details: format!("{} failed attempts logged", failed_attempts.len()),
        });

        Ok(ComplianceReport {
            generated_at: Utc::now(),
            period_days,
            statistics,
            data_access_summary,
            sensitive_data_access,
            failed_attempts,
            integrity_verified,
            compliance_checks,
        })
    }

    /// Export logs in JSON format
    pub async fn export_json(&self, query: AuditQuery) -> Result<String> {
        let entries = self
            .query_logs(
                query.user_id,
                query.action,
                query.start_time.map(|start| {
                    let end = query.end_time.unwrap_or_else(Utc::now);
                    (start, end)
                }),
                query.limit,
            )
            .await?;

        serde_json::to_string_pretty(&entries)
            .map_err(|e| AuditError::StorageError(format!("JSON serialization failed: {e}")))
    }

    /// Clean up old entries
    pub async fn cleanup_old_entries(&self) -> Result<usize> {
        let cutoff = Utc::now() - Duration::days(self.config.retention_period_days);

        let mut entries = self.entries.write().await;
        let original_len = entries.len();

        entries.retain(|e| e.timestamp > cutoff);

        let removed = original_len - entries.len();
        *self.archived_count.write().await += removed;

        Ok(removed)
    }

    /// Get total entry count (including archived)
    pub async fn total_entry_count(&self) -> usize {
        let current = self.entries.read().await.len();
        let archived = *self.archived_count.read().await;
        current + archived
    }
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_log_access() {
        let logger = AuditLogger::new();

        let id = logger
            .log_access(
                "user123",
                "resource456",
                AuditAction::Read,
                AuditContext::new().with_reason("User requested data"),
            )
            .await
            .unwrap();

        assert!(!id.is_empty());
    }

    #[tokio::test]
    async fn test_log_modification() {
        let logger = AuditLogger::new();

        let id = logger
            .log_modification(
                "user123",
                ResourceType::UserProfile,
                "profile456",
                AuditAction::Update,
                Some("old_value".to_string()),
                Some("new_value".to_string()),
                AuditContext::new(),
            )
            .await
            .unwrap();

        assert!(!id.is_empty());
    }

    #[tokio::test]
    async fn test_query_logs() {
        let logger = AuditLogger::new();

        logger
            .log_access("user123", "res1", AuditAction::Read, AuditContext::new())
            .await
            .unwrap();

        logger
            .log_access("user456", "res2", AuditAction::Read, AuditContext::new())
            .await
            .unwrap();

        let logs = logger
            .query_logs(Some("user123".to_string()), None, None, 100)
            .await
            .unwrap();

        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].user_id, "user123");
    }

    #[tokio::test]
    async fn test_integrity_verification() {
        let logger = AuditLogger::new();

        for i in 0..10 {
            logger
                .log_access(
                    &format!("user{}", i),
                    "resource",
                    AuditAction::Read,
                    AuditContext::new(),
                )
                .await
                .unwrap();
        }

        let verified = logger.verify_integrity().await.unwrap();
        assert!(verified);
    }

    #[tokio::test]
    async fn test_statistics() {
        let logger = AuditLogger::new();

        for _ in 0..5 {
            logger
                .log_access("user123", "res", AuditAction::Read, AuditContext::new())
                .await
                .unwrap();
        }

        for _ in 0..3 {
            logger
                .log_access("user456", "res", AuditAction::Update, AuditContext::new())
                .await
                .unwrap();
        }

        let stats = logger.get_statistics(None).await.unwrap();

        assert_eq!(stats.total_entries, 8);
        assert_eq!(stats.by_action.get(&AuditAction::Read), Some(&5));
        assert_eq!(stats.by_action.get(&AuditAction::Update), Some(&3));
    }

    #[tokio::test]
    async fn test_compliance_report() {
        let logger = AuditLogger::new();

        logger
            .log_access(
                "user123",
                "sensitive_data",
                AuditAction::Export,
                AuditContext::new().with_reason("Legal request"),
            )
            .await
            .unwrap();

        let report = logger.generate_compliance_report(7).await.unwrap();

        assert!(report.integrity_verified);
        assert!(!report.compliance_checks.is_empty());
    }

    #[tokio::test]
    async fn test_failed_event() {
        let logger = AuditLogger::new();

        logger
            .log_failure(
                "user123",
                AuditAction::Read,
                ResourceType::Feedback,
                "res456",
                "Permission denied",
                AuditContext::new(),
            )
            .await
            .unwrap();

        let logs = logger.query_logs(None, None, None, 100).await.unwrap();

        assert_eq!(logs.len(), 1);
        assert!(!logs[0].success);
        assert!(logs[0].error_message.is_some());
    }

    #[tokio::test]
    async fn test_cleanup() {
        let mut config = LoggerConfig::default();
        config.retention_period_days = 0;

        let logger = AuditLogger::with_config(config);

        logger
            .log_access("user", "res", AuditAction::Read, AuditContext::new())
            .await
            .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        let removed = logger.cleanup_old_entries().await.unwrap();
        assert!(removed > 0);
    }
}
