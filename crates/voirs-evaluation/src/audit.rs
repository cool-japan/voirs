//! Audit trail system for compliance, debugging, and security monitoring
//!
//! This module provides comprehensive audit logging capabilities for tracking
//! all evaluation activities, ensuring compliance with regulatory requirements,
//! and enabling forensic analysis of system behavior.

use crate::EvaluationError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// Audit trail configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Enable audit logging
    pub enabled: bool,
    /// Audit log directory
    pub log_directory: PathBuf,
    /// Maximum log file size in bytes (default: 100MB)
    pub max_file_size: usize,
    /// Log rotation policy
    pub rotation_policy: RotationPolicy,
    /// Retention period in days
    pub retention_days: u32,
    /// Include sensitive data in logs
    pub log_sensitive_data: bool,
    /// Encryption configuration for logs
    pub encryption: Option<EncryptionConfig>,
    /// Tamper protection
    pub tamper_protection: bool,
    /// Real-time monitoring
    pub realtime_monitoring: bool,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_directory: PathBuf::from("./audit_logs"),
            max_file_size: 100 * 1024 * 1024, // 100MB
            rotation_policy: RotationPolicy::Daily,
            retention_days: 90,
            log_sensitive_data: false,
            encryption: None,
            tamper_protection: true,
            realtime_monitoring: false,
        }
    }
}

/// Log rotation policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RotationPolicy {
    /// Rotate daily
    Daily,
    /// Rotate weekly
    Weekly,
    /// Rotate monthly
    Monthly,
    /// Rotate when size limit reached
    SizeBased,
    /// No rotation
    None,
}

/// Encryption configuration for audit logs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Encryption algorithm
    pub algorithm: String,
    /// Key ID (reference to key in key management system)
    pub key_id: String,
    /// Enable compression before encryption
    pub compress: bool,
}

/// Audit event representing a logged activity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Unique event ID
    pub event_id: Uuid,
    /// Timestamp of the event
    pub timestamp: SystemTime,
    /// ISO 8601 formatted timestamp
    pub timestamp_iso: String,
    /// Event type/category
    pub event_type: AuditEventType,
    /// Severity level
    pub severity: SeverityLevel,
    /// User or service that triggered the event
    pub actor: Actor,
    /// Resource being accessed/modified
    pub resource: Resource,
    /// Action performed
    pub action: String,
    /// Result of the action
    pub result: ActionResult,
    /// Client information
    pub client_info: ClientInfo,
    /// Additional context data
    pub context: HashMap<String, serde_json::Value>,
    /// Duration of the operation
    pub duration_ms: Option<u64>,
    /// Tamper protection hash
    pub integrity_hash: Option<String>,
}

/// Types of audit events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditEventType {
    /// Authentication events
    Authentication,
    /// Authorization events
    Authorization,
    /// Data access events
    DataAccess,
    /// Data modification events
    DataModification,
    /// Configuration changes
    ConfigurationChange,
    /// Evaluation execution
    EvaluationExecution,
    /// System administration
    SystemAdmin,
    /// Security events
    Security,
    /// Compliance events
    Compliance,
    /// Error/failure events
    Error,
}

/// Severity levels for audit events
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SeverityLevel {
    /// Debug information
    Debug,
    /// Informational
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Critical security event
    Critical,
}

/// Actor information (who performed the action)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Actor {
    /// Actor ID (user ID, service ID, etc.)
    pub id: String,
    /// Actor type
    pub actor_type: ActorType,
    /// Actor name/email
    pub name: Option<String>,
    /// IP address
    pub ip_address: Option<String>,
    /// Session ID
    pub session_id: Option<String>,
    /// Additional attributes
    pub attributes: HashMap<String, String>,
}

/// Types of actors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActorType {
    /// Human user
    User,
    /// Service account
    Service,
    /// System process
    System,
    /// API client
    ApiClient,
    /// Unknown/anonymous
    Unknown,
}

/// Resource being accessed/modified
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    /// Resource ID
    pub id: String,
    /// Resource type
    pub resource_type: ResourceType,
    /// Resource name
    pub name: Option<String>,
    /// Parent resource (if applicable)
    pub parent: Option<String>,
    /// Resource attributes
    pub attributes: HashMap<String, String>,
}

/// Types of resources
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceType {
    /// Audio file
    AudioFile,
    /// Evaluation result
    EvaluationResult,
    /// Model
    Model,
    /// Dataset
    Dataset,
    /// Configuration
    Configuration,
    /// User account
    UserAccount,
    /// API key
    ApiKey,
    /// System resource
    System,
}

/// Result of an action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionResult {
    /// Whether the action succeeded
    pub success: bool,
    /// Status code (HTTP-like)
    pub status_code: u16,
    /// Result message
    pub message: Option<String>,
    /// Error details (if failed)
    pub error_details: Option<String>,
}

/// Client information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    /// User agent
    pub user_agent: Option<String>,
    /// Client application
    pub application: Option<String>,
    /// Client version
    pub version: Option<String>,
    /// Geographic location (country code)
    pub location: Option<String>,
}

/// Audit trail manager
pub struct AuditTrail {
    config: AuditConfig,
    current_file: Arc<Mutex<Option<BufWriter<File>>>>,
    current_file_path: Arc<Mutex<Option<PathBuf>>>,
    current_file_size: Arc<Mutex<usize>>,
    event_buffer: Arc<Mutex<Vec<AuditEvent>>>,
    stats: Arc<Mutex<AuditStatistics>>,
}

/// Audit statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditStatistics {
    /// Total events logged
    pub total_events: u64,
    /// Events by type
    pub events_by_type: HashMap<String, u64>,
    /// Events by severity
    pub events_by_severity: HashMap<String, u64>,
    /// Failed writes
    pub failed_writes: u64,
    /// Buffer overflows
    pub buffer_overflows: u64,
    /// Last event timestamp
    pub last_event_time: Option<SystemTime>,
}

impl AuditTrail {
    /// Create a new audit trail manager
    pub fn new(config: AuditConfig) -> Result<Self, EvaluationError> {
        // Create log directory if it doesn't exist
        if config.enabled {
            std::fs::create_dir_all(&config.log_directory).map_err(|e| {
                EvaluationError::ProcessingError {
                    message: format!("Failed to create audit log directory: {}", e),
                    source: None,
                }
            })?;
        }

        Ok(Self {
            config,
            current_file: Arc::new(Mutex::new(None)),
            current_file_path: Arc::new(Mutex::new(None)),
            current_file_size: Arc::new(Mutex::new(0)),
            event_buffer: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(Mutex::new(AuditStatistics::default())),
        })
    }

    /// Log an audit event
    pub fn log_event(&self, mut event: AuditEvent) -> Result<(), EvaluationError> {
        if !self.config.enabled {
            return Ok(());
        }

        // Add integrity hash if tamper protection is enabled
        if self.config.tamper_protection {
            event.integrity_hash = Some(self.compute_integrity_hash(&event));
        }

        // Update statistics
        self.update_statistics(&event);

        // Serialize event
        let event_json =
            serde_json::to_string(&event).map_err(|e| EvaluationError::ProcessingError {
                message: format!("Failed to serialize audit event: {}", e),
                source: None,
            })?;

        // Write to file
        self.write_event(&event_json)?;

        Ok(())
    }

    /// Log evaluation execution event
    pub fn log_evaluation(
        &self,
        actor: Actor,
        resource: Resource,
        result: ActionResult,
        duration_ms: u64,
        context: HashMap<String, serde_json::Value>,
    ) -> Result<(), EvaluationError> {
        let event = AuditEvent {
            event_id: Uuid::new_v4(),
            timestamp: SystemTime::now(),
            timestamp_iso: Utc::now().to_rfc3339(),
            event_type: AuditEventType::EvaluationExecution,
            severity: if result.success {
                SeverityLevel::Info
            } else {
                SeverityLevel::Error
            },
            actor,
            resource,
            action: "evaluate".to_string(),
            result,
            client_info: ClientInfo {
                user_agent: None,
                application: Some("voirs-evaluation".to_string()),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
                location: None,
            },
            context,
            duration_ms: Some(duration_ms),
            integrity_hash: None,
        };

        self.log_event(event)
    }

    /// Log authentication event
    pub fn log_authentication(
        &self,
        actor: Actor,
        success: bool,
        method: &str,
    ) -> Result<(), EvaluationError> {
        let event = AuditEvent {
            event_id: Uuid::new_v4(),
            timestamp: SystemTime::now(),
            timestamp_iso: Utc::now().to_rfc3339(),
            event_type: AuditEventType::Authentication,
            severity: if success {
                SeverityLevel::Info
            } else {
                SeverityLevel::Warning
            },
            actor,
            resource: Resource {
                id: "authentication".to_string(),
                resource_type: ResourceType::System,
                name: Some("Authentication System".to_string()),
                parent: None,
                attributes: HashMap::new(),
            },
            action: format!("authenticate_{}", method),
            result: ActionResult {
                success,
                status_code: if success { 200 } else { 401 },
                message: Some(
                    if success {
                        "Authentication successful"
                    } else {
                        "Authentication failed"
                    }
                    .to_string(),
                ),
                error_details: None,
            },
            client_info: ClientInfo {
                user_agent: None,
                application: None,
                version: None,
                location: None,
            },
            context: HashMap::from([(
                "auth_method".to_string(),
                serde_json::Value::String(method.to_string()),
            )]),
            duration_ms: None,
            integrity_hash: None,
        };

        self.log_event(event)
    }

    /// Log data access event
    pub fn log_data_access(
        &self,
        actor: Actor,
        resource: Resource,
        action: &str,
        success: bool,
    ) -> Result<(), EvaluationError> {
        let event = AuditEvent {
            event_id: Uuid::new_v4(),
            timestamp: SystemTime::now(),
            timestamp_iso: Utc::now().to_rfc3339(),
            event_type: AuditEventType::DataAccess,
            severity: SeverityLevel::Info,
            actor,
            resource,
            action: action.to_string(),
            result: ActionResult {
                success,
                status_code: if success { 200 } else { 403 },
                message: None,
                error_details: None,
            },
            client_info: ClientInfo {
                user_agent: None,
                application: None,
                version: None,
                location: None,
            },
            context: HashMap::new(),
            duration_ms: None,
            integrity_hash: None,
        };

        self.log_event(event)
    }

    /// Log security event
    pub fn log_security_event(
        &self,
        actor: Actor,
        event_description: &str,
        severity: SeverityLevel,
        details: HashMap<String, serde_json::Value>,
    ) -> Result<(), EvaluationError> {
        let event = AuditEvent {
            event_id: Uuid::new_v4(),
            timestamp: SystemTime::now(),
            timestamp_iso: Utc::now().to_rfc3339(),
            event_type: AuditEventType::Security,
            severity,
            actor,
            resource: Resource {
                id: "security".to_string(),
                resource_type: ResourceType::System,
                name: Some("Security System".to_string()),
                parent: None,
                attributes: HashMap::new(),
            },
            action: event_description.to_string(),
            result: ActionResult {
                success: true,
                status_code: 200,
                message: Some(event_description.to_string()),
                error_details: None,
            },
            client_info: ClientInfo {
                user_agent: None,
                application: None,
                version: None,
                location: None,
            },
            context: details,
            duration_ms: None,
            integrity_hash: None,
        };

        self.log_event(event)
    }

    fn write_event(&self, event_json: &str) -> Result<(), EvaluationError> {
        let mut file_guard = self
            .current_file
            .lock()
            .expect("lock should not be poisoned");
        let mut path_guard = self
            .current_file_path
            .lock()
            .expect("lock should not be poisoned");
        let mut size_guard = self
            .current_file_size
            .lock()
            .expect("lock should not be poisoned");

        // Check if we need to rotate the log file
        let needs_rotation = match self.config.rotation_policy {
            RotationPolicy::SizeBased => *size_guard >= self.config.max_file_size,
            _ => file_guard.is_none(), // Rotate on first write or when file is closed
        };

        if needs_rotation || file_guard.is_none() {
            self.rotate_log_file(&mut file_guard, &mut path_guard, &mut size_guard)?;
        }

        // Write event
        if let Some(ref mut writer) = *file_guard {
            writeln!(writer, "{}", event_json).map_err(|e| EvaluationError::ProcessingError {
                message: format!("Failed to write audit event: {}", e),
                source: None,
            })?;

            *size_guard += event_json.len() + 1; // +1 for newline

            // Flush immediately for critical events
            writer
                .flush()
                .map_err(|e| EvaluationError::ProcessingError {
                    message: format!("Failed to flush audit log: {}", e),
                    source: None,
                })?;
        }

        Ok(())
    }

    fn rotate_log_file(
        &self,
        file_guard: &mut Option<BufWriter<File>>,
        path_guard: &mut Option<PathBuf>,
        size_guard: &mut usize,
    ) -> Result<(), EvaluationError> {
        // Close current file if open
        if let Some(mut writer) = file_guard.take() {
            writer.flush().ok();
        }

        // Generate new log file name
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let filename = format!("audit_{}.jsonl", timestamp);
        let filepath = self.config.log_directory.join(filename);

        // Open new log file
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&filepath)
            .map_err(|e| EvaluationError::ProcessingError {
                message: format!("Failed to open audit log file: {}", e),
                source: None,
            })?;

        *file_guard = Some(BufWriter::new(file));
        *path_guard = Some(filepath);
        *size_guard = 0;

        Ok(())
    }

    fn compute_integrity_hash(&self, event: &AuditEvent) -> String {
        // Simple hash computation (in production, use cryptographic hash)
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        event.event_id.hash(&mut hasher);
        event.action.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    fn update_statistics(&self, event: &AuditEvent) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_events += 1;

            let event_type_key = format!("{:?}", event.event_type);
            *stats.events_by_type.entry(event_type_key).or_insert(0) += 1;

            let severity_key = format!("{:?}", event.severity);
            *stats.events_by_severity.entry(severity_key).or_insert(0) += 1;

            stats.last_event_time = Some(event.timestamp);
        }
    }

    /// Get audit statistics
    pub fn get_statistics(&self) -> AuditStatistics {
        self.stats
            .lock()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Query audit events
    pub fn query_events(&self, query: AuditQuery) -> Result<Vec<AuditEvent>, EvaluationError> {
        let mut results = Vec::new();

        // Read log files in the directory
        let entries = std::fs::read_dir(&self.config.log_directory).map_err(|e| {
            EvaluationError::ProcessingError {
                message: format!("Failed to read audit log directory: {}", e),
                source: None,
            }
        })?;

        for entry in entries.flatten() {
            if let Ok(file_type) = entry.file_type() {
                if file_type.is_file() {
                    if let Some(filename) = entry.file_name().to_str() {
                        if filename.starts_with("audit_") && filename.ends_with(".jsonl") {
                            // Read and parse events from file
                            if let Ok(events) = self.read_events_from_file(&entry.path()) {
                                for event in events {
                                    if query.matches(&event) {
                                        results.push(event);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sort by timestamp (newest first)
        results.sort_by_key(|b| std::cmp::Reverse(b.timestamp));

        // Apply limit
        if let Some(limit) = query.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    fn read_events_from_file(&self, path: &Path) -> Result<Vec<AuditEvent>, EvaluationError> {
        use std::io::{BufRead, BufReader};

        let file = File::open(path).map_err(|e| EvaluationError::ProcessingError {
            message: format!("Failed to open audit log file: {}", e),
            source: None,
        })?;

        let reader = BufReader::new(file);
        let mut events = Vec::new();

        for line_result in reader.lines() {
            if let Ok(line_content) = line_result {
                if let Ok(event) = serde_json::from_str::<AuditEvent>(&line_content) {
                    events.push(event);
                }
            }
        }

        Ok(events)
    }

    /// Flush all pending events
    pub fn flush(&self) -> Result<(), EvaluationError> {
        if let Ok(mut file_guard) = self.current_file.lock() {
            if let Some(ref mut writer) = *file_guard {
                writer
                    .flush()
                    .map_err(|e| EvaluationError::ProcessingError {
                        message: format!("Failed to flush audit log: {}", e),
                        source: None,
                    })?;
            }
        }
        Ok(())
    }

    /// Generate compliance report
    pub fn generate_compliance_report(
        &self,
        start_time: SystemTime,
        end_time: SystemTime,
    ) -> Result<ComplianceReport, EvaluationError> {
        let query = AuditQuery {
            start_time: Some(start_time),
            end_time: Some(end_time),
            event_types: None,
            severity_levels: None,
            actors: None,
            limit: None,
        };

        let events = self.query_events(query)?;

        let total_events = events.len();
        let security_events = events
            .iter()
            .filter(|e| e.event_type == AuditEventType::Security)
            .count();
        let failed_authentications = events
            .iter()
            .filter(|e| e.event_type == AuditEventType::Authentication && !e.result.success)
            .count();
        let data_access_events = events
            .iter()
            .filter(|e| e.event_type == AuditEventType::DataAccess)
            .count();

        Ok(ComplianceReport {
            start_time,
            end_time,
            total_events,
            security_events,
            failed_authentications,
            data_access_events,
            events_by_type: self.count_events_by_type(&events),
            events_by_severity: self.count_events_by_severity(&events),
            top_actors: self.get_top_actors(&events, 10),
            anomalies: self.detect_anomalies(&events),
        })
    }

    fn count_events_by_type(&self, events: &[AuditEvent]) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for event in events {
            let key = format!("{:?}", event.event_type);
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }

    fn count_events_by_severity(&self, events: &[AuditEvent]) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for event in events {
            let key = format!("{:?}", event.severity);
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }

    fn get_top_actors(&self, events: &[AuditEvent], limit: usize) -> Vec<(String, usize)> {
        let mut actor_counts: HashMap<String, usize> = HashMap::new();
        for event in events {
            *actor_counts.entry(event.actor.id.clone()).or_insert(0) += 1;
        }

        let mut sorted: Vec<_> = actor_counts.into_iter().collect();
        sorted.sort_by_key(|b| std::cmp::Reverse(b.1));
        sorted.truncate(limit);
        sorted
    }

    fn detect_anomalies(&self, events: &[AuditEvent]) -> Vec<String> {
        let mut anomalies = Vec::new();

        // Check for rapid failed authentication attempts
        let failed_auth_count = events
            .iter()
            .filter(|e| e.event_type == AuditEventType::Authentication && !e.result.success)
            .count();

        if failed_auth_count > 5 {
            anomalies.push(format!(
                "High number of failed authentication attempts: {}",
                failed_auth_count
            ));
        }

        // Check for critical events
        let critical_count = events
            .iter()
            .filter(|e| e.severity == SeverityLevel::Critical)
            .count();

        if critical_count > 0 {
            anomalies.push(format!(
                "Critical security events detected: {}",
                critical_count
            ));
        }

        anomalies
    }
}

impl Drop for AuditTrail {
    fn drop(&mut self) {
        // Ensure all events are flushed on drop
        let _ = self.flush();
    }
}

/// Audit query for searching events
#[derive(Debug, Clone, Default)]
pub struct AuditQuery {
    /// Start time filter
    pub start_time: Option<SystemTime>,
    /// End time filter
    pub end_time: Option<SystemTime>,
    /// Event type filter
    pub event_types: Option<Vec<AuditEventType>>,
    /// Severity level filter
    pub severity_levels: Option<Vec<SeverityLevel>>,
    /// Actor filter
    pub actors: Option<Vec<String>>,
    /// Maximum number of results
    pub limit: Option<usize>,
}

impl AuditQuery {
    fn matches(&self, event: &AuditEvent) -> bool {
        // Check time range
        if let Some(start) = self.start_time {
            if event.timestamp < start {
                return false;
            }
        }

        if let Some(end) = self.end_time {
            if event.timestamp > end {
                return false;
            }
        }

        // Check event type
        if let Some(ref types) = self.event_types {
            if !types.contains(&event.event_type) {
                return false;
            }
        }

        // Check severity
        if let Some(ref severities) = self.severity_levels {
            if !severities.contains(&event.severity) {
                return false;
            }
        }

        // Check actor
        if let Some(ref actors) = self.actors {
            if !actors.contains(&event.actor.id) {
                return false;
            }
        }

        true
    }
}

/// Compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Report start time
    pub start_time: SystemTime,
    /// Report end time
    pub end_time: SystemTime,
    /// Total events in period
    pub total_events: usize,
    /// Security-related events
    pub security_events: usize,
    /// Failed authentication attempts
    pub failed_authentications: usize,
    /// Data access events
    pub data_access_events: usize,
    /// Events by type
    pub events_by_type: HashMap<String, usize>,
    /// Events by severity
    pub events_by_severity: HashMap<String, usize>,
    /// Top actors by activity
    pub top_actors: Vec<(String, usize)>,
    /// Detected anomalies
    pub anomalies: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_audit_trail_creation() {
        let temp_dir = env::temp_dir().join("voirs_audit_test");
        let config = AuditConfig {
            log_directory: temp_dir,
            ..Default::default()
        };

        let trail = AuditTrail::new(config);
        assert!(trail.is_ok());
    }

    #[test]
    fn test_log_evaluation_event() {
        let temp_dir = env::temp_dir().join("voirs_audit_test_eval");
        let config = AuditConfig {
            log_directory: temp_dir.clone(),
            ..Default::default()
        };

        let trail = AuditTrail::new(config).unwrap();

        let actor = Actor {
            id: "user123".to_string(),
            actor_type: ActorType::User,
            name: Some("Test User".to_string()),
            ip_address: Some("127.0.0.1".to_string()),
            session_id: Some("session123".to_string()),
            attributes: HashMap::new(),
        };

        let resource = Resource {
            id: "audio123".to_string(),
            resource_type: ResourceType::AudioFile,
            name: Some("test.wav".to_string()),
            parent: None,
            attributes: HashMap::new(),
        };

        let result = ActionResult {
            success: true,
            status_code: 200,
            message: Some("Evaluation completed".to_string()),
            error_details: None,
        };

        let log_result = trail.log_evaluation(actor, resource, result, 150, HashMap::new());
        assert!(log_result.is_ok());

        let stats = trail.get_statistics();
        assert_eq!(stats.total_events, 1);

        // Cleanup
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn test_log_authentication_event() {
        let temp_dir = env::temp_dir().join("voirs_audit_test_auth");
        let config = AuditConfig {
            log_directory: temp_dir.clone(),
            ..Default::default()
        };

        let trail = AuditTrail::new(config).unwrap();

        let actor = Actor {
            id: "user456".to_string(),
            actor_type: ActorType::User,
            name: Some("Test User".to_string()),
            ip_address: Some("192.168.1.1".to_string()),
            session_id: None,
            attributes: HashMap::new(),
        };

        let result = trail.log_authentication(actor, true, "api_key");
        assert!(result.is_ok());

        // Cleanup
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn test_audit_statistics() {
        let temp_dir = env::temp_dir().join("voirs_audit_test_stats");
        let config = AuditConfig {
            log_directory: temp_dir.clone(),
            ..Default::default()
        };

        let trail = AuditTrail::new(config).unwrap();

        // Log multiple events
        for i in 0..5 {
            let actor = Actor {
                id: format!("user{}", i),
                actor_type: ActorType::User,
                name: None,
                ip_address: None,
                session_id: None,
                attributes: HashMap::new(),
            };

            trail.log_authentication(actor, i % 2 == 0, "password").ok();
        }

        let stats = trail.get_statistics();
        assert_eq!(stats.total_events, 5);
        assert!(stats.events_by_type.contains_key("Authentication"));

        // Cleanup
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn test_compliance_report() {
        let temp_dir = env::temp_dir().join("voirs_audit_test_compliance");
        let config = AuditConfig {
            log_directory: temp_dir.clone(),
            ..Default::default()
        };

        let trail = AuditTrail::new(config).unwrap();

        let actor = Actor {
            id: "user789".to_string(),
            actor_type: ActorType::User,
            name: None,
            ip_address: None,
            session_id: None,
            attributes: HashMap::new(),
        };

        // Log some events
        trail
            .log_authentication(actor.clone(), true, "api_key")
            .ok();
        trail.log_authentication(actor, false, "password").ok();

        let start_time = SystemTime::now() - Duration::from_secs(3600);
        let end_time = SystemTime::now();

        let report = trail.generate_compliance_report(start_time, end_time);
        assert!(report.is_ok());

        let report = report.unwrap();
        assert!(report.total_events > 0);

        // Cleanup
        std::fs::remove_dir_all(temp_dir).ok();
    }
}
