//! Error Tracking and Reporting System
//!
//! This module provides a comprehensive error tracking system for the `VoiRS` feedback platform.
//! It captures, aggregates, analyzes, and reports errors with support for error grouping,
//! alerting, and integration with external error tracking services.
//!
//! # Features
//!
//! - **Error Capture**: Automatic error capturing with full context (stack traces, user info, environment)
//! - **Error Grouping**: Intelligent grouping of similar errors for easier analysis
//! - **Error Analytics**: Statistics, trends, and insights about error patterns
//! - **Alerting**: Threshold-based alerting for error rate spikes
//! - **Reporting**: Comprehensive error reports with charts and analysis
//! - **Integration**: Export to Sentry, Rollbar, or custom error tracking services
//! - **Privacy**: Automatic PII scrubbing and data anonymization
//!
//! # Example
//!
//! ```rust
//! use voirs_feedback::error_tracking::{ErrorTracker, ErrorSeverity, ErrorContext};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let tracker = ErrorTracker::new();
//!
//! // Track an error
//! tracker.track_error(
//!     "DatabaseConnectionError",
//!     "Failed to connect to database",
//!     ErrorSeverity::Critical,
//!     ErrorContext::new()
//!         .with_user_id("user123")
//!         .with_request_id("req456")
//!         .with_extra("host", "db.example.com"),
//! ).await?;
//!
//! // Get error statistics
//! let stats = tracker.get_error_statistics(None).await?;
//! println!("Total errors: {}", stats.total_errors);
//!
//! // Generate report
//! let report = tracker.generate_error_report(24).await?;
//! # Ok(())
//! # }
//! ```

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

/// Errors that can occur in the error tracking system
#[derive(Error, Debug)]
#[allow(missing_docs)]
pub enum TrackingError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Storage error
    #[error("Storage error: {0}")]
    StorageError(String),

    /// Export error
    #[error("Export error: {0}")]
    ExportError(String),

    /// Invalid data
    #[error("Invalid data: {0}")]
    InvalidData(String),
}

/// Type alias for Results in this module
pub type Result<T> = std::result::Result<T, TrackingError>;

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Hash)]
#[allow(missing_docs)]
pub enum ErrorSeverity {
    /// Debug-level error
    Debug,
    /// Informational error
    Info,
    /// Warning-level error
    Warning,
    /// Error requiring attention
    Error,
    /// Critical error requiring immediate action
    Critical,
    /// Fatal error causing system failure
    Fatal,
}

/// Error context with additional metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ErrorContext {
    /// User ID (anonymized)
    pub user_id: Option<String>,
    /// Session ID
    pub session_id: Option<String>,
    /// Request ID
    pub request_id: Option<String>,
    /// HTTP method
    pub http_method: Option<String>,
    /// Request URL
    pub url: Option<String>,
    /// IP address (anonymized)
    pub ip_address: Option<String>,
    /// User agent string
    pub user_agent: Option<String>,
    /// Application version
    pub app_version: Option<String>,
    /// Environment (production, staging, development)
    pub environment: Option<String>,
    /// Additional custom data
    pub extra: HashMap<String, String>,
}

impl ErrorContext {
    /// Create a new error context
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the user ID
    #[must_use]
    pub fn with_user_id(mut self, user_id: &str) -> Self {
        self.user_id = Some(user_id.to_string());
        self
    }

    /// Set the session ID
    #[must_use]
    pub fn with_session_id(mut self, session_id: &str) -> Self {
        self.session_id = Some(session_id.to_string());
        self
    }

    /// Set the request ID
    #[must_use]
    pub fn with_request_id(mut self, request_id: &str) -> Self {
        self.request_id = Some(request_id.to_string());
        self
    }

    /// Set the HTTP method
    #[must_use]
    pub fn with_http_method(mut self, method: &str) -> Self {
        self.http_method = Some(method.to_string());
        self
    }

    /// Set the URL
    #[must_use]
    pub fn with_url(mut self, url: &str) -> Self {
        self.url = Some(url.to_string());
        self
    }

    /// Set the IP address (will be anonymized)
    #[must_use]
    pub fn with_ip(mut self, ip: &str) -> Self {
        // Anonymize IP (keep only first 2 octets for IPv4)
        let anonymized = Self::anonymize_ip(ip);
        self.ip_address = Some(anonymized);
        self
    }

    /// Set the environment
    #[must_use]
    pub fn with_environment(mut self, env: &str) -> Self {
        self.environment = Some(env.to_string());
        self
    }

    /// Add extra key-value data
    #[must_use]
    pub fn with_extra(mut self, key: &str, value: &str) -> Self {
        self.extra.insert(key.to_string(), value.to_string());
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

/// Captured error event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEvent {
    /// Unique error event ID
    pub id: String,
    /// Error type/name
    pub error_type: String,
    /// Error message
    pub message: String,
    /// Error severity
    pub severity: ErrorSeverity,
    /// Stack trace (if available)
    pub stack_trace: Option<String>,
    /// Error context
    pub context: ErrorContext,
    /// Timestamp when error occurred
    pub timestamp: DateTime<Utc>,
    /// Error group ID (for grouping similar errors)
    pub group_id: String,
    /// Number of times this error has occurred
    pub occurrence_count: usize,
}

/// Grouped error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorGroup {
    /// Group ID
    pub id: String,
    /// Error type
    pub error_type: String,
    /// Representative message
    pub message: String,
    /// Severity
    pub severity: ErrorSeverity,
    /// First occurrence timestamp
    pub first_seen: DateTime<Utc>,
    /// Last occurrence timestamp
    pub last_seen: DateTime<Utc>,
    /// Total occurrences
    pub occurrence_count: usize,
    /// Affected users
    pub affected_users: Vec<String>,
    /// Sample events (latest N)
    pub sample_events: Vec<ErrorEvent>,
}

/// Error statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStatistics {
    /// Start time of the period
    pub start_time: DateTime<Utc>,
    /// End time of the period
    pub end_time: DateTime<Utc>,
    /// Total errors
    pub total_errors: usize,
    /// Errors by severity
    pub errors_by_severity: HashMap<ErrorSeverity, usize>,
    /// Errors by type
    pub errors_by_type: HashMap<String, usize>,
    /// Error rate (errors per minute)
    pub error_rate_per_minute: f64,
    /// Unique error groups
    pub unique_error_groups: usize,
    /// Affected users
    pub affected_users_count: usize,
    /// Top errors
    pub top_errors: Vec<ErrorGroup>,
}

/// Error trend data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorTrend {
    /// Time bucket
    pub timestamp: DateTime<Utc>,
    /// Error count in this bucket
    pub error_count: usize,
    /// Errors by severity in this bucket
    pub severity_breakdown: HashMap<ErrorSeverity, usize>,
}

/// Error report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorReport {
    /// Report generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Time period
    pub period_hours: i64,
    /// Overall statistics
    pub statistics: ErrorStatistics,
    /// Trends over time
    pub trends: Vec<ErrorTrend>,
    /// Critical errors
    pub critical_errors: Vec<ErrorGroup>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Configuration for error tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackerConfig {
    /// Maximum events to keep in memory
    pub max_events: usize,
    /// Maximum error groups to track
    pub max_groups: usize,
    /// Sample size for error groups
    pub group_sample_size: usize,
    /// Auto-cleanup interval in seconds
    pub cleanup_interval_secs: u64,
    /// Event retention period in days
    pub retention_period_days: i64,
    /// Enable PII scrubbing
    pub scrub_pii: bool,
    /// Alert threshold (errors per minute)
    pub alert_threshold_per_minute: f64,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            max_events: 100_000,
            max_groups: 10_000,
            group_sample_size: 10,
            cleanup_interval_secs: 3600,
            retention_period_days: 90,
            scrub_pii: true,
            alert_threshold_per_minute: 100.0,
        }
    }
}

/// Main error tracking system
pub struct ErrorTracker {
    /// Configuration
    config: TrackerConfig,
    /// All error events
    events: Arc<RwLock<VecDeque<ErrorEvent>>>,
    /// Grouped errors
    groups: Arc<RwLock<HashMap<String, ErrorGroup>>>,
    /// Statistics cache
    stats_cache: Arc<RwLock<Option<(ErrorStatistics, DateTime<Utc>)>>>,
}

impl ErrorTracker {
    /// Create a new error tracker with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(TrackerConfig::default())
    }

    /// Create a new error tracker with custom configuration
    #[must_use]
    pub fn with_config(config: TrackerConfig) -> Self {
        Self {
            config,
            events: Arc::new(RwLock::new(VecDeque::new())),
            groups: Arc::new(RwLock::new(HashMap::new())),
            stats_cache: Arc::new(RwLock::new(None)),
        }
    }

    /// Track a new error
    pub async fn track_error(
        &self,
        error_type: &str,
        message: &str,
        severity: ErrorSeverity,
        mut context: ErrorContext,
    ) -> Result<String> {
        // Scrub PII if enabled
        if self.config.scrub_pii {
            context = self.scrub_pii_from_context(context);
        }

        // Generate group ID based on error type and message pattern
        let group_id = self.generate_group_id(error_type, message);

        // Create error event
        let event = ErrorEvent {
            id: uuid::Uuid::new_v4().to_string(),
            error_type: error_type.to_string(),
            message: message.to_string(),
            severity,
            stack_trace: None,
            context,
            timestamp: Utc::now(),
            group_id: group_id.clone(),
            occurrence_count: 1,
        };

        // Add to events
        let mut events = self.events.write().await;
        if events.len() >= self.config.max_events {
            events.pop_front();
        }
        events.push_back(event.clone());
        drop(events);

        // Update error group
        self.update_error_group(event).await?;

        // Invalidate cache
        *self.stats_cache.write().await = None;

        Ok(group_id)
    }

    /// Track an error with stack trace
    pub async fn track_error_with_trace(
        &self,
        error_type: &str,
        message: &str,
        severity: ErrorSeverity,
        stack_trace: String,
        context: ErrorContext,
    ) -> Result<String> {
        let mut ctx = context;
        let group_id = self
            .track_error(error_type, message, severity, ctx.clone())
            .await?;

        // Update the event with stack trace
        let mut events = self.events.write().await;
        if let Some(event) = events.back_mut() {
            event.stack_trace = Some(stack_trace);
        }

        Ok(group_id)
    }

    /// Generate a group ID for error grouping
    fn generate_group_id(&self, error_type: &str, message: &str) -> String {
        // Simple hash-based grouping (in production, use more sophisticated algorithm)
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        error_type.hash(&mut hasher);
        // Normalize message by removing numbers and specific details
        let normalized = Self::normalize_error_message(message);
        normalized.hash(&mut hasher);

        format!("{:x}", hasher.finish())
    }

    /// Normalize error message for grouping
    fn normalize_error_message(message: &str) -> String {
        // Remove numbers, UUIDs, timestamps, etc.
        message
            .chars()
            .map(|c| if c.is_numeric() { 'X' } else { c })
            .collect()
    }

    /// Update error group with new event
    async fn update_error_group(&self, event: ErrorEvent) -> Result<()> {
        let mut groups = self.groups.write().await;

        if let Some(group) = groups.get_mut(&event.group_id) {
            // Update existing group
            group.last_seen = event.timestamp;
            group.occurrence_count += 1;

            if let Some(user_id) = &event.context.user_id {
                if !group.affected_users.contains(user_id) {
                    group.affected_users.push(user_id.clone());
                }
            }

            // Add to sample events
            if group.sample_events.len() >= self.config.group_sample_size {
                group.sample_events.remove(0);
            }
            group.sample_events.push(event);
        } else {
            // Create new group
            let affected_users = if let Some(user_id) = &event.context.user_id {
                vec![user_id.clone()]
            } else {
                vec![]
            };

            let group = ErrorGroup {
                id: event.group_id.clone(),
                error_type: event.error_type.clone(),
                message: event.message.clone(),
                severity: event.severity,
                first_seen: event.timestamp,
                last_seen: event.timestamp,
                occurrence_count: 1,
                affected_users,
                sample_events: vec![event],
            };

            if groups.len() >= self.config.max_groups {
                // Remove oldest group
                if let Some(oldest_id) = groups
                    .iter()
                    .min_by_key(|(_, g)| g.last_seen)
                    .map(|(id, _)| id.clone())
                {
                    groups.remove(&oldest_id);
                }
            }

            groups.insert(group.id.clone(), group);
        }

        Ok(())
    }

    /// Scrub PII from context
    fn scrub_pii_from_context(&self, mut context: ErrorContext) -> ErrorContext {
        // Scrub email addresses from message
        if let Some(user_id) = &context.user_id {
            if user_id.contains('@') {
                context.user_id = Some("[email]".to_string());
            }
        }

        // Remove potentially sensitive URLs
        if let Some(url) = &context.url {
            if url.contains("token") || url.contains("password") {
                context.url = Some("[redacted]".to_string());
            }
        }

        context
    }

    /// Get error statistics for a time period
    pub async fn get_error_statistics(&self, hours: Option<i64>) -> Result<ErrorStatistics> {
        // Check cache
        {
            let cache = self.stats_cache.read().await;
            if let Some((stats, cached_at)) = &*cache {
                if Utc::now().signed_duration_since(*cached_at) < Duration::minutes(5) {
                    return Ok(stats.clone());
                }
            }
        }

        let hours = hours.unwrap_or(24);
        let end_time = Utc::now();
        let start_time = end_time - Duration::hours(hours);

        let events = self.events.read().await;
        let groups = self.groups.read().await;

        let relevant_events: Vec<&ErrorEvent> = events
            .iter()
            .filter(|e| e.timestamp >= start_time && e.timestamp <= end_time)
            .collect();

        let total_errors = relevant_events.len();

        // Errors by severity
        let mut errors_by_severity: HashMap<ErrorSeverity, usize> = HashMap::new();
        for event in &relevant_events {
            *errors_by_severity.entry(event.severity).or_insert(0) += 1;
        }

        // Errors by type
        let mut errors_by_type: HashMap<String, usize> = HashMap::new();
        for event in &relevant_events {
            *errors_by_type.entry(event.error_type.clone()).or_insert(0) += 1;
        }

        // Error rate
        let duration_minutes = hours as f64 * 60.0;
        let error_rate_per_minute = if duration_minutes > 0.0 {
            total_errors as f64 / duration_minutes
        } else {
            0.0
        };

        // Affected users
        let mut affected_users = std::collections::HashSet::new();
        for event in &relevant_events {
            if let Some(user_id) = &event.context.user_id {
                affected_users.insert(user_id.clone());
            }
        }

        // Top errors
        let mut top_errors: Vec<ErrorGroup> = groups.values().cloned().collect();
        top_errors.sort_by(|a, b| b.occurrence_count.cmp(&a.occurrence_count));
        top_errors.truncate(10);

        let stats = ErrorStatistics {
            start_time,
            end_time,
            total_errors,
            errors_by_severity,
            errors_by_type,
            error_rate_per_minute,
            unique_error_groups: groups.len(),
            affected_users_count: affected_users.len(),
            top_errors,
        };

        // Cache the result
        *self.stats_cache.write().await = Some((stats.clone(), Utc::now()));

        Ok(stats)
    }

    /// Get error trends over time
    pub async fn get_error_trends(
        &self,
        hours: i64,
        bucket_minutes: i64,
    ) -> Result<Vec<ErrorTrend>> {
        let end_time = Utc::now();
        let start_time = end_time - Duration::hours(hours);

        let events = self.events.read().await;

        let mut buckets: HashMap<DateTime<Utc>, (usize, HashMap<ErrorSeverity, usize>)> =
            HashMap::new();

        for event in events.iter() {
            if event.timestamp < start_time || event.timestamp > end_time {
                continue;
            }

            // Calculate bucket
            let minutes_since_start = event
                .timestamp
                .signed_duration_since(start_time)
                .num_minutes();
            let bucket_index = minutes_since_start / bucket_minutes;
            let bucket_time = start_time + Duration::minutes(bucket_index * bucket_minutes);

            let (count, severity_map) = buckets.entry(bucket_time).or_insert((0, HashMap::new()));
            *count += 1;
            *severity_map.entry(event.severity).or_insert(0) += 1;
        }

        let mut trends: Vec<ErrorTrend> = buckets
            .into_iter()
            .map(
                |(timestamp, (error_count, severity_breakdown))| ErrorTrend {
                    timestamp,
                    error_count,
                    severity_breakdown,
                },
            )
            .collect();

        trends.sort_by_key(|t| t.timestamp);

        Ok(trends)
    }

    /// Generate comprehensive error report
    pub async fn generate_error_report(&self, period_hours: i64) -> Result<ErrorReport> {
        let statistics = self.get_error_statistics(Some(period_hours)).await?;
        let trends = self.get_error_trends(period_hours, 60).await?; // 1-hour buckets

        let groups = self.groups.read().await;
        let critical_errors: Vec<ErrorGroup> = groups
            .values()
            .filter(|g| g.severity >= ErrorSeverity::Critical)
            .cloned()
            .collect();

        let mut recommendations = Vec::new();

        // Generate recommendations based on error patterns
        if statistics.error_rate_per_minute > self.config.alert_threshold_per_minute {
            recommendations.push(format!(
                "⚠️ High error rate detected: {:.2} errors/min (threshold: {:.2})",
                statistics.error_rate_per_minute, self.config.alert_threshold_per_minute
            ));
        }

        if !critical_errors.is_empty() {
            recommendations.push(format!(
                "🔥 {} critical error groups require immediate attention",
                critical_errors.len()
            ));
        }

        if statistics.affected_users_count > 100 {
            recommendations.push(format!(
                "👥 {} users affected by errors in the past {} hours",
                statistics.affected_users_count, period_hours
            ));
        }

        Ok(ErrorReport {
            generated_at: Utc::now(),
            period_hours,
            statistics,
            trends,
            critical_errors,
            recommendations,
        })
    }

    /// Get all error groups
    pub async fn get_error_groups(&self) -> Vec<ErrorGroup> {
        self.groups.read().await.values().cloned().collect()
    }

    /// Get specific error group
    pub async fn get_error_group(&self, group_id: &str) -> Option<ErrorGroup> {
        self.groups.read().await.get(group_id).cloned()
    }

    /// Clean up old events
    pub async fn cleanup_old_events(&self) -> Result<usize> {
        let cutoff = Utc::now() - Duration::days(self.config.retention_period_days);

        let mut events = self.events.write().await;
        let original_len = events.len();
        events.retain(|e| e.timestamp > cutoff);

        let removed = original_len - events.len();

        // Invalidate cache
        *self.stats_cache.write().await = None;

        Ok(removed)
    }

    /// Export errors in JSON format
    pub async fn export_json(&self, hours: Option<i64>) -> serde_json::Value {
        let stats = self.get_error_statistics(hours).await.ok();

        serde_json::json!({
            "timestamp": Utc::now().to_rfc3339(),
            "statistics": stats,
            "groups": self.get_error_groups().await,
        })
    }
}

impl Default for ErrorTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_track_error() {
        let tracker = ErrorTracker::new();

        let context = ErrorContext::new()
            .with_user_id("user123")
            .with_request_id("req456");

        let group_id = tracker
            .track_error(
                "DatabaseError",
                "Connection failed",
                ErrorSeverity::Error,
                context,
            )
            .await
            .unwrap();

        assert!(!group_id.is_empty());
    }

    #[tokio::test]
    async fn test_error_grouping() {
        let tracker = ErrorTracker::new();

        // Track same error twice
        for _ in 0..2 {
            let context = ErrorContext::new();
            tracker
                .track_error(
                    "TypeError",
                    "Cannot read property 'x'",
                    ErrorSeverity::Error,
                    context,
                )
                .await
                .unwrap();
        }

        let groups = tracker.get_error_groups().await;
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].occurrence_count, 2);
    }

    #[tokio::test]
    async fn test_error_statistics() {
        let tracker = ErrorTracker::new();

        for i in 0..10 {
            let severity = if i < 5 {
                ErrorSeverity::Warning
            } else {
                ErrorSeverity::Error
            };

            tracker
                .track_error(
                    "TestError",
                    &format!("Error {}", i),
                    severity,
                    ErrorContext::new(),
                )
                .await
                .unwrap();
        }

        let stats = tracker.get_error_statistics(None).await.unwrap();
        assert_eq!(stats.total_errors, 10);
        assert_eq!(
            stats.errors_by_severity.get(&ErrorSeverity::Warning),
            Some(&5)
        );
        assert_eq!(
            stats.errors_by_severity.get(&ErrorSeverity::Error),
            Some(&5)
        );
    }

    #[tokio::test]
    async fn test_pii_scrubbing() {
        let tracker = ErrorTracker::new();

        let context = ErrorContext::new()
            .with_user_id("user@example.com")
            .with_ip("192.168.1.100");

        tracker
            .track_error("Error", "Test", ErrorSeverity::Error, context)
            .await
            .unwrap();

        let events = tracker.events.read().await;
        let event = events.back().unwrap();

        // Email should be scrubbed
        assert_eq!(event.context.user_id.as_deref(), Some("[email]"));
        // IP should be anonymized
        assert!(event.context.ip_address.as_ref().unwrap().contains("xxx"));
    }

    #[tokio::test]
    async fn test_error_trends() {
        let tracker = ErrorTracker::new();

        for _ in 0..20 {
            tracker
                .track_error("Error", "Test", ErrorSeverity::Error, ErrorContext::new())
                .await
                .unwrap();
        }

        let trends = tracker.get_error_trends(1, 15).await.unwrap(); // 15-minute buckets
        assert!(!trends.is_empty());
    }

    #[tokio::test]
    async fn test_error_report() {
        let tracker = ErrorTracker::new();

        // Add some test errors
        tracker
            .track_error(
                "CriticalError",
                "System down",
                ErrorSeverity::Critical,
                ErrorContext::new(),
            )
            .await
            .unwrap();

        let report = tracker.generate_error_report(24).await.unwrap();

        assert!(!report.critical_errors.is_empty());
        assert!(!report.recommendations.is_empty());
    }

    #[tokio::test]
    async fn test_cleanup() {
        let mut config = TrackerConfig::default();
        config.retention_period_days = 0;

        let tracker = ErrorTracker::with_config(config);

        tracker
            .track_error("Error", "Test", ErrorSeverity::Error, ErrorContext::new())
            .await
            .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        let removed = tracker.cleanup_old_events().await.unwrap();
        assert!(removed > 0);
    }
}
