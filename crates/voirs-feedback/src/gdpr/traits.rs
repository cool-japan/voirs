//! GDPR Compliance Traits
//!
//! This module defines the core traits for GDPR compliance functionality.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use super::types::*;

/// Core GDPR compliance trait defining required functionality
#[async_trait]
pub trait GdprCompliance: Send + Sync {
    /// Register a new data subject
    async fn register_data_subject(&self, subject: DataSubject) -> GdprResult<()>;

    /// Record consent for a processing purpose
    async fn record_consent(&self, subject_id: &str, consent: ConsentRecord) -> GdprResult<()>;

    /// Withdraw consent for a processing purpose
    async fn withdraw_consent(
        &self,
        subject_id: &str,
        purpose: ProcessingPurpose,
    ) -> GdprResult<()>;

    /// Check if processing is allowed for a purpose
    async fn is_processing_allowed(
        &self,
        subject_id: &str,
        purpose: ProcessingPurpose,
    ) -> GdprResult<bool>;

    /// Create a data subject request
    async fn create_data_request(&self, request: DataRequest) -> GdprResult<Uuid>;

    /// Process a data subject request
    async fn process_data_request(&self, request_id: Uuid) -> GdprResult<()>;

    /// Export all data for a subject (data portability)
    async fn export_subject_data(&self, subject_id: &str) -> GdprResult<SubjectDataExport>;

    /// Anonymize data for a subject
    async fn anonymize_subject_data(&self, subject_id: &str) -> GdprResult<()>;

    /// Delete all data for a subject (right to erasure)
    async fn delete_subject_data(&self, subject_id: &str) -> GdprResult<()>;

    /// Record processing activity
    async fn record_processing_activity(&self, activity: ProcessingActivity) -> GdprResult<()>;

    /// Get processing activities for a subject
    async fn get_processing_activities(
        &self,
        subject_id: &str,
    ) -> GdprResult<Vec<ProcessingActivity>>;

    /// Check data retention compliance
    async fn check_retention_compliance(&self) -> GdprResult<Vec<RetentionViolation>>;

    /// Apply data retention policies
    async fn apply_retention_policies(&self) -> GdprResult<RetentionReport>;

    /// Report a data breach
    async fn report_data_breach(&self, breach: DataBreach) -> GdprResult<Uuid>;

    /// Get compliance report
    async fn get_compliance_report(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> GdprResult<ComplianceReport>;
}
