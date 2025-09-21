//! GDPR Compliance Types and Data Structures
//!
//! This module contains all type definitions, enums, and data structures
//! used throughout the GDPR compliance system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;
use uuid::Uuid;

/// GDPR compliance errors
#[derive(Error, Debug)]
pub enum GdprError {
    /// Data subject not found
    #[error("Data subject {subject_id} not found")]
    DataSubjectNotFound { subject_id: String },

    /// Consent validation failed
    #[error("Consent validation failed: {reason}")]
    ConsentValidationFailed { reason: String },

    /// Data retention policy violation
    #[error("Data retention policy violation: {details}")]
    RetentionPolicyViolation { details: String },

    /// Anonymization failed
    #[error("Data anonymization failed: {message}")]
    AnonymizationFailed { message: String },

    /// Data export failed
    #[error("Data export failed: {message}")]
    DataExportFailed { message: String },

    /// Data deletion failed
    #[error("Data deletion failed: {message}")]
    DataDeletionFailed { message: String },

    /// Privacy policy violation
    #[error("Privacy policy violation: {violation}")]
    PrivacyPolicyViolation { violation: String },

    /// Insufficient consent
    #[error("Insufficient consent for operation: {operation}")]
    InsufficientConsent { operation: String },
}

/// Result type for GDPR operations
pub type GdprResult<T> = Result<T, GdprError>;

/// Data processing purposes under GDPR
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProcessingPurpose {
    /// Essential service functionality
    Essential,
    /// Performance analytics
    Analytics,
    /// Personalization and adaptive learning
    Personalization,
    /// Marketing and communications
    Marketing,
    /// Research and development
    Research,
    /// Legal compliance
    LegalCompliance,
    /// Custom purpose
    Custom(String),
}

/// Legal basis for data processing under GDPR
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LegalBasis {
    /// Consent of the data subject
    Consent,
    /// Necessary for contract performance
    Contract,
    /// Necessary for legal obligation compliance
    LegalObligation,
    /// Necessary to protect vital interests
    VitalInterests,
    /// Necessary for public task performance
    PublicTask,
    /// Legitimate interests pursued by controller
    LegitimateInterests { interest: String },
}

/// Consent status and details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRecord {
    /// Unique consent ID
    pub consent_id: Uuid,
    /// Data subject ID
    pub subject_id: String,
    /// Processing purpose
    pub purpose: ProcessingPurpose,
    /// Legal basis for processing
    pub legal_basis: LegalBasis,
    /// Consent given status
    pub consent_given: bool,
    /// Timestamp when consent was recorded
    pub recorded_at: DateTime<Utc>,
    /// Timestamp when consent was last updated
    pub updated_at: DateTime<Utc>,
    /// Consent expiry date
    pub expires_at: Option<DateTime<Utc>>,
    /// Consent withdrawal timestamp
    pub withdrawn_at: Option<DateTime<Utc>>,
    /// Version of consent form/policy
    pub consent_version: String,
    /// IP address when consent was given
    pub ip_address: Option<String>,
    /// User agent when consent was given
    pub user_agent: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Data subject information and rights management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSubject {
    /// Unique subject identifier
    pub subject_id: String,
    /// Email address (if provided)
    pub email: Option<String>,
    /// Subject's preferred language
    pub language: Option<String>,
    /// Registration timestamp
    pub registered_at: DateTime<Utc>,
    /// Last activity timestamp
    pub last_active_at: DateTime<Utc>,
    /// Consent records
    pub consent_records: Vec<ConsentRecord>,
    /// Data retention settings
    pub retention_settings: RetentionSettings,
    /// Privacy preferences
    pub privacy_preferences: PrivacyPreferences,
    /// Active data requests
    pub active_requests: Vec<DataRequest>,
}

/// Data retention settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionSettings {
    /// Retention period for feedback data
    pub feedback_retention_days: u32,
    /// Retention period for analytics data
    pub analytics_retention_days: u32,
    /// Retention period for progress data
    pub progress_retention_days: u32,
    /// Auto-deletion enabled
    pub auto_deletion_enabled: bool,
    /// Anonymization after retention period
    pub anonymize_after_retention: bool,
    /// Custom retention rules
    pub custom_rules: HashMap<String, u32>,
}

impl Default for RetentionSettings {
    fn default() -> Self {
        Self {
            feedback_retention_days: 365,  // 1 year
            analytics_retention_days: 730, // 2 years
            progress_retention_days: 1095, // 3 years
            auto_deletion_enabled: true,
            anonymize_after_retention: true,
            custom_rules: HashMap::new(),
        }
    }
}

/// Privacy preferences for data subject
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyPreferences {
    /// Allow data processing for analytics
    pub allow_analytics: bool,
    /// Allow data processing for personalization
    pub allow_personalization: bool,
    /// Allow data processing for marketing
    pub allow_marketing: bool,
    /// Allow data processing for research
    pub allow_research: bool,
    /// Allow data sharing with third parties
    pub allow_third_party_sharing: bool,
    /// Allow automated decision making
    pub allow_automated_decisions: bool,
    /// Data minimization preference
    pub data_minimization: bool,
    /// Pseudonymization preference
    pub prefer_pseudonymization: bool,
}

impl Default for PrivacyPreferences {
    fn default() -> Self {
        Self {
            allow_analytics: false,
            allow_personalization: true,
            allow_marketing: false,
            allow_research: false,
            allow_third_party_sharing: false,
            allow_automated_decisions: true,
            data_minimization: true,
            prefer_pseudonymization: true,
        }
    }
}

/// Types of data subject requests under GDPR
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataRequestType {
    /// Request for data access (Article 15)
    Access,
    /// Request for data rectification (Article 16)
    Rectification,
    /// Request for data erasure (Article 17)
    Erasure,
    /// Request to restrict processing (Article 18)
    RestrictProcessing,
    /// Request for data portability (Article 20)
    DataPortability,
    /// Object to processing (Article 21)
    ObjectToProcessing,
    /// Withdraw consent
    WithdrawConsent,
}

/// Status of data subject requests
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RequestStatus {
    /// Request received and pending review
    Pending,
    /// Request is being processed
    Processing,
    /// Request completed successfully
    Completed,
    /// Request rejected with reason
    Rejected { reason: String },
    /// Request requires additional information
    RequiresInfo { details: String },
}

/// Data subject request record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRequest {
    /// Unique request ID
    pub request_id: Uuid,
    /// Data subject ID
    pub subject_id: String,
    /// Request type
    pub request_type: DataRequestType,
    /// Request status
    pub status: RequestStatus,
    /// Request description
    pub description: String,
    /// Request timestamp
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,
    /// Expected completion date
    pub expected_completion: DateTime<Utc>,
    /// Request response/result
    pub response: Option<String>,
    /// Associated data
    pub associated_data: HashMap<String, String>,
}

/// Data processing activity record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingActivity {
    /// Activity ID
    pub activity_id: Uuid,
    /// Data subject ID
    pub subject_id: String,
    /// Processing purpose
    pub purpose: ProcessingPurpose,
    /// Legal basis
    pub legal_basis: LegalBasis,
    /// Data categories processed
    pub data_categories: Vec<String>,
    /// Processing timestamp
    pub processed_at: DateTime<Utc>,
    /// Data source
    pub data_source: String,
    /// Data destination
    pub data_destination: String,
    /// Retention period
    pub retention_period: Option<chrono::Duration>,
    /// Security measures applied
    pub security_measures: Vec<String>,
}

/// Data breach information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataBreach {
    /// Breach ID
    pub breach_id: Uuid,
    /// Breach discovery timestamp
    pub discovered_at: DateTime<Utc>,
    /// Breach occurrence timestamp
    pub occurred_at: DateTime<Utc>,
    /// Affected data subjects
    pub affected_subjects: Vec<String>,
    /// Breach description
    pub description: String,
    /// Breach severity
    pub severity: BreachSeverity,
    /// Data categories affected
    pub affected_data_categories: Vec<String>,
    /// Containment measures taken
    pub containment_measures: Vec<String>,
    /// Notification status
    pub notification_status: NotificationStatus,
    /// Regulatory reporting required
    pub regulatory_reporting_required: bool,
    /// Breach response timeline
    pub response_timeline: Vec<BreachEvent>,
}

/// Severity levels for data breaches
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BreachSeverity {
    /// Low risk to data subjects
    Low,
    /// Medium risk to data subjects
    Medium,
    /// High risk to data subjects
    High,
    /// Critical risk requiring immediate action
    Critical,
}

/// Notification status for data breaches
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NotificationStatus {
    /// No notification required
    NotRequired,
    /// Notification pending
    Pending,
    /// Supervisory authority notified
    AuthorityNotified { notified_at: DateTime<Utc> },
    /// Data subjects notified
    SubjectsNotified { notified_at: DateTime<Utc> },
    /// Both authority and subjects notified
    FullyNotified {
        authority_at: DateTime<Utc>,
        subjects_at: DateTime<Utc>,
    },
}

/// Data breach event with timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreachEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event description
    pub description: String,
    /// Event type
    pub event_type: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Subject data export structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectDataExport {
    /// Subject ID
    pub subject_id: String,
    /// Export timestamp
    pub exported_at: DateTime<Utc>,
    /// Subject information
    pub subject_info: DataSubject,
    /// Feedback data
    pub feedback_data: Vec<crate::traits::FeedbackResponse>,
    /// Progress data
    pub progress_data: Vec<crate::traits::UserProgress>,
    /// Analytics data
    pub analytics_data: Vec<AnalyticsEntry>,
    /// Processing activities
    pub processing_activities: Vec<ProcessingActivity>,
    /// Consent history
    pub consent_history: Vec<ConsentRecord>,
}

/// Analytics entry for export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsEntry {
    /// Entry ID
    pub id: Uuid,
    /// Event type
    pub event_type: String,
    /// Event data
    pub data: HashMap<String, String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Data retention violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionViolation {
    /// Subject ID
    pub subject_id: String,
    /// Data type that violates retention
    pub data_type: String,
    /// Data creation timestamp
    pub data_created_at: DateTime<Utc>,
    /// Retention deadline
    pub retention_deadline: DateTime<Utc>,
    /// Violation severity
    pub severity: ViolationSeverity,
}

/// Severity levels for retention violations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Minor violation
    Minor,
    /// Major violation
    Major,
    /// Critical violation requiring immediate attention
    Critical,
}

/// Data retention compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionReport {
    /// Report ID
    pub report_id: Uuid,
    /// Report generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Report period start
    pub period_start: DateTime<Utc>,
    /// Report period end
    pub period_end: DateTime<Utc>,
    /// Total violations found
    pub total_violations: u32,
    /// Violations by severity
    pub violations_by_severity: HashMap<ViolationSeverity, u32>,
    /// Compliance percentage
    pub compliance_percentage: f64,
}

/// Comprehensive compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Report ID
    pub report_id: Uuid,
    /// Report generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Report period start
    pub period_start: DateTime<Utc>,
    /// Report period end
    pub period_end: DateTime<Utc>,
    /// Total data subjects
    pub total_subjects: u32,
    /// Active consent records
    pub active_consents: u32,
    /// Processed data requests
    pub processed_requests: u32,
    /// Data breaches
    pub data_breaches: u32,
    /// Retention violations
    pub retention_violations: u32,
    /// Overall compliance score
    pub compliance_score: f64,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}
