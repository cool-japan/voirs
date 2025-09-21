//! GDPR Compliance Module
//!
//! This module provides comprehensive GDPR (General Data Protection Regulation)
//! compliance functionality for the VoiRS feedback system, including:
//!
//! - Data subject rights management
//! - Consent tracking and validation
//! - Data retention policy enforcement
//! - Privacy-preserving analytics
//! - Data breach reporting
//! - Compliance monitoring and reporting
//!
//! The module is organized into several submodules:
//! - [`types`] - Core data types and structures
//! - [`traits`] - GDPR compliance traits and interfaces
//! - [`manager`] - Main GDPR compliance manager implementation
//! - [`encryption`] - Encryption and privacy utilities
//! - [`retention`] - Data retention management

pub mod encryption;
pub mod manager;
pub mod retention;
pub mod traits;
pub mod types;

// Re-export commonly used types and traits
pub use encryption::{DifferentialPrivacy, GdprEncryption, PrivacyPreservingAnalytics};
pub use manager::GdprComplianceManager;
pub use retention::{
    CleanupStatus, CleanupTask, CustomRetentionRule, DataRetentionManager, RetentionAction,
    RetentionComplianceReport, RetentionComplianceStatus, RetentionCondition,
    RetentionEnforcementResult, RetentionPolicyConfig,
};
pub use traits::GdprCompliance;
pub use types::{
    AnalyticsEntry, BreachEvent, BreachSeverity, ComplianceReport, ConsentRecord, DataBreach,
    DataRequest, DataRequestType, DataSubject, GdprError, GdprResult, LegalBasis,
    NotificationStatus, PrivacyPreferences, ProcessingActivity, ProcessingPurpose, RequestStatus,
    RetentionReport, RetentionSettings, RetentionViolation, SubjectDataExport, ViolationSeverity,
};

/// Create a new GDPR compliance manager with default settings
pub fn create_gdpr_manager() -> GdprComplianceManager {
    GdprComplianceManager::new()
}

/// Create a new data retention manager
pub fn create_retention_manager() -> DataRetentionManager {
    DataRetentionManager::new()
}

/// Create GDPR encryption utilities
pub fn create_gdpr_encryption() -> GdprEncryption {
    GdprEncryption::new()
}

/// Create privacy-preserving analytics with default epsilon
pub fn create_privacy_analytics() -> PrivacyPreservingAnalytics {
    PrivacyPreservingAnalytics::new(1.0)
}

/// Create privacy-preserving analytics with custom epsilon
pub fn create_privacy_analytics_with_epsilon(epsilon: f64) -> PrivacyPreservingAnalytics {
    PrivacyPreservingAnalytics::new(epsilon)
}
