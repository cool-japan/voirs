//! GDPR Compliance Manager Implementation
//!
//! This module contains the main GdprComplianceManager that implements
//! the core GDPR compliance functionality.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use super::traits::GdprCompliance;
use super::types::*;
use crate::traits::{FeedbackResponse, UserProgress};

/// Main GDPR compliance manager
#[derive(Debug)]
pub struct GdprComplianceManager {
    /// Data subjects registry
    subjects: Arc<RwLock<HashMap<String, DataSubject>>>,
    /// Processing activities log
    activities: Arc<RwLock<Vec<ProcessingActivity>>>,
    /// Data requests registry
    requests: Arc<RwLock<HashMap<Uuid, DataRequest>>>,
    /// Data breaches registry
    breaches: Arc<RwLock<HashMap<Uuid, DataBreach>>>,
}

impl GdprComplianceManager {
    /// Create a new GDPR compliance manager
    pub fn new() -> Self {
        Self {
            subjects: Arc::new(RwLock::new(HashMap::new())),
            activities: Arc::new(RwLock::new(Vec::new())),
            requests: Arc::new(RwLock::new(HashMap::new())),
            breaches: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Anonymize string data
    fn anonymize_string(&self, input: &str) -> String {
        if input.len() <= 3 {
            "*".repeat(input.len())
        } else {
            format!("{}***", &input[..2])
        }
    }

    /// Check if consent is valid and not expired
    fn is_consent_valid(&self, consent: &ConsentRecord) -> bool {
        consent.consent_given
            && consent.withdrawn_at.is_none()
            && consent.expires_at.map_or(true, |exp| exp > Utc::now())
    }
}

#[async_trait]
impl GdprCompliance for GdprComplianceManager {
    async fn register_data_subject(&self, subject: DataSubject) -> GdprResult<()> {
        let mut subjects = self.subjects.write().await;
        subjects.insert(subject.subject_id.clone(), subject);
        Ok(())
    }

    async fn record_consent(&self, subject_id: &str, consent: ConsentRecord) -> GdprResult<()> {
        let mut subjects = self.subjects.write().await;

        if let Some(subject) = subjects.get_mut(subject_id) {
            // Remove any existing consent for the same purpose
            subject
                .consent_records
                .retain(|c| c.purpose != consent.purpose);
            subject.consent_records.push(consent);
            Ok(())
        } else {
            Err(GdprError::DataSubjectNotFound {
                subject_id: subject_id.to_string(),
            })
        }
    }

    async fn withdraw_consent(
        &self,
        subject_id: &str,
        purpose: ProcessingPurpose,
    ) -> GdprResult<()> {
        let mut subjects = self.subjects.write().await;

        if let Some(subject) = subjects.get_mut(subject_id) {
            for consent in &mut subject.consent_records {
                if consent.purpose == purpose && consent.consent_given {
                    consent.withdrawn_at = Some(Utc::now());
                    consent.consent_given = false;
                    consent.updated_at = Utc::now();
                    return Ok(());
                }
            }

            Err(GdprError::ConsentValidationFailed {
                reason: format!("No active consent found for purpose {:?}", purpose),
            })
        } else {
            Err(GdprError::DataSubjectNotFound {
                subject_id: subject_id.to_string(),
            })
        }
    }

    async fn is_processing_allowed(
        &self,
        subject_id: &str,
        purpose: ProcessingPurpose,
    ) -> GdprResult<bool> {
        let subjects = self.subjects.read().await;

        if let Some(subject) = subjects.get(subject_id) {
            // Check for valid consent
            for consent in &subject.consent_records {
                if consent.purpose == purpose && self.is_consent_valid(consent) {
                    return Ok(true);
                }
            }

            // Check privacy preferences for implicit consent
            match purpose {
                ProcessingPurpose::Essential => Ok(true), // Always allowed for essential functionality
                ProcessingPurpose::Analytics => Ok(subject.privacy_preferences.allow_analytics),
                ProcessingPurpose::Personalization => {
                    Ok(subject.privacy_preferences.allow_personalization)
                }
                ProcessingPurpose::Marketing => Ok(subject.privacy_preferences.allow_marketing),
                ProcessingPurpose::Research => Ok(subject.privacy_preferences.allow_research),
                _ => Ok(false),
            }
        } else {
            Err(GdprError::DataSubjectNotFound {
                subject_id: subject_id.to_string(),
            })
        }
    }

    async fn create_data_request(&self, request: DataRequest) -> GdprResult<Uuid> {
        let mut requests = self.requests.write().await;
        let request_id = request.request_id;
        requests.insert(request_id, request);
        Ok(request_id)
    }

    async fn process_data_request(&self, request_id: Uuid) -> GdprResult<()> {
        let mut requests = self.requests.write().await;

        if let Some(request) = requests.get_mut(&request_id) {
            request.status = RequestStatus::Processing;

            // Simulate processing based on request type
            match request.request_type {
                DataRequestType::Access => {
                    request.response = Some(String::from("Data export prepared"));
                }
                DataRequestType::Erasure => {
                    request.response = Some(String::from("Data deletion completed"));
                }
                DataRequestType::Rectification => {
                    request.response = Some(String::from("Data rectification completed"));
                }
                DataRequestType::DataPortability => {
                    request.response = Some(String::from("Data portability export prepared"));
                }
                _ => {
                    request.response = Some(String::from("Request processed"));
                }
            }

            request.status = RequestStatus::Completed;
            Ok(())
        } else {
            Err(GdprError::DataExportFailed {
                message: format!("Request {} not found", request_id),
            })
        }
    }

    async fn export_subject_data(&self, subject_id: &str) -> GdprResult<SubjectDataExport> {
        let subjects = self.subjects.read().await;
        let activities = self.activities.read().await;

        if let Some(subject) = subjects.get(subject_id) {
            Ok(SubjectDataExport {
                subject_id: subject_id.to_string(),
                exported_at: Utc::now(),
                subject_info: subject.clone(),
                feedback_data: Vec::new(), // Would be populated from actual data stores
                progress_data: Vec::new(), // Would be populated from actual data stores
                analytics_data: Vec::new(), // Would be populated from actual data stores
                processing_activities: activities
                    .iter()
                    .filter(|a| a.subject_id == subject_id)
                    .cloned()
                    .collect(),
                consent_history: subject.consent_records.clone(),
            })
        } else {
            Err(GdprError::DataSubjectNotFound {
                subject_id: subject_id.to_string(),
            })
        }
    }

    async fn anonymize_subject_data(&self, subject_id: &str) -> GdprResult<()> {
        let mut subjects = self.subjects.write().await;

        if let Some(subject) = subjects.get_mut(subject_id) {
            // Anonymize personal identifiers
            subject.email = subject.email.as_ref().map(|e| self.anonymize_string(e));

            // Clear non-essential metadata
            for consent in &mut subject.consent_records {
                consent.ip_address = None;
                consent.user_agent = None;
                consent.metadata.clear();
            }

            Ok(())
        } else {
            Err(GdprError::DataSubjectNotFound {
                subject_id: subject_id.to_string(),
            })
        }
    }

    async fn delete_subject_data(&self, subject_id: &str) -> GdprResult<()> {
        let mut subjects = self.subjects.write().await;
        let mut activities = self.activities.write().await;
        let mut requests = self.requests.write().await;

        // Remove subject
        if subjects.remove(subject_id).is_none() {
            return Err(GdprError::DataSubjectNotFound {
                subject_id: subject_id.to_string(),
            });
        }

        // Remove related activities
        activities.retain(|a| a.subject_id != subject_id);

        // Remove related requests
        requests.retain(|_, r| r.subject_id != subject_id);

        Ok(())
    }

    async fn record_processing_activity(&self, activity: ProcessingActivity) -> GdprResult<()> {
        let mut activities = self.activities.write().await;
        activities.push(activity);
        Ok(())
    }

    async fn get_processing_activities(
        &self,
        subject_id: &str,
    ) -> GdprResult<Vec<ProcessingActivity>> {
        let activities = self.activities.read().await;
        Ok(activities
            .iter()
            .filter(|a| a.subject_id == subject_id)
            .cloned()
            .collect())
    }

    async fn check_retention_compliance(&self) -> GdprResult<Vec<RetentionViolation>> {
        // Mock implementation - would check actual data stores
        Ok(vec![])
    }

    async fn apply_retention_policies(&self) -> GdprResult<RetentionReport> {
        // Mock implementation - would apply actual retention policies
        Ok(RetentionReport {
            report_id: Uuid::new_v4(),
            generated_at: Utc::now(),
            period_start: Utc::now(),
            period_end: Utc::now(),
            total_violations: 0,
            violations_by_severity: HashMap::new(),
            compliance_percentage: 100.0,
        })
    }

    async fn report_data_breach(&self, breach: DataBreach) -> GdprResult<Uuid> {
        let mut breaches = self.breaches.write().await;
        let breach_id = breach.breach_id;
        breaches.insert(breach_id, breach);
        Ok(breach_id)
    }

    async fn get_compliance_report(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> GdprResult<ComplianceReport> {
        let subjects = self.subjects.read().await;
        let activities = self.activities.read().await;
        let requests = self.requests.read().await;
        let breaches = self.breaches.read().await;

        let total_subjects = subjects.len() as u32;
        let active_consents = subjects
            .values()
            .flat_map(|s| &s.consent_records)
            .filter(|c| self.is_consent_valid(c))
            .count() as u32;

        let processed_requests = requests
            .values()
            .filter(|r| {
                r.created_at >= from && r.created_at <= to && r.status == RequestStatus::Completed
            })
            .count() as u32;

        let data_breaches = breaches
            .values()
            .filter(|b| b.discovered_at >= from && b.discovered_at <= to)
            .count() as u32;

        Ok(ComplianceReport {
            report_id: Uuid::new_v4(),
            generated_at: Utc::now(),
            period_start: from,
            period_end: to,
            total_subjects,
            active_consents,
            processed_requests,
            data_breaches,
            retention_violations: 0, // Would be calculated from actual data
            compliance_score: 95.0,  // Would be calculated based on various factors
            recommendations: vec![
                String::from("Review and update privacy policies regularly"),
                String::from("Implement automated consent management"),
                String::from("Enhance data retention monitoring"),
            ],
        })
    }
}

impl Default for GdprComplianceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration as ChronoDuration;

    #[tokio::test]
    async fn test_gdpr_manager_creation() {
        let manager = GdprComplianceManager::new();
        let report = manager
            .get_compliance_report(Utc::now() - ChronoDuration::days(30), Utc::now())
            .await
            .unwrap();
        assert_eq!(report.total_subjects, 0);
    }

    #[tokio::test]
    async fn test_data_subject_registration() {
        let manager = GdprComplianceManager::new();

        let subject = DataSubject {
            subject_id: String::from("test_user_001"),
            email: Some(String::from("test@example.com")),
            language: Some(String::from("en")),
            registered_at: Utc::now(),
            last_active_at: Utc::now(),
            consent_records: Vec::new(),
            retention_settings: RetentionSettings::default(),
            privacy_preferences: PrivacyPreferences::default(),
            active_requests: Vec::new(),
        };

        let result = manager.register_data_subject(subject).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_consent_management() {
        let manager = GdprComplianceManager::new();
        let subject_id = "test_user_002";

        // Register subject first
        let subject = DataSubject {
            subject_id: subject_id.to_string(),
            email: Some(String::from("test2@example.com")),
            language: Some(String::from("en")),
            registered_at: Utc::now(),
            last_active_at: Utc::now(),
            consent_records: Vec::new(),
            retention_settings: RetentionSettings::default(),
            privacy_preferences: PrivacyPreferences::default(),
            active_requests: Vec::new(),
        };

        manager.register_data_subject(subject).await.unwrap();

        // Record consent
        let consent = ConsentRecord {
            consent_id: Uuid::new_v4(),
            subject_id: subject_id.to_string(),
            purpose: ProcessingPurpose::Analytics,
            legal_basis: LegalBasis::Consent,
            consent_given: true,
            recorded_at: Utc::now(),
            updated_at: Utc::now(),
            expires_at: Some(Utc::now() + ChronoDuration::days(365)),
            withdrawn_at: None,
            consent_version: String::from("1.0"),
            ip_address: Some(String::from("192.168.1.1")),
            user_agent: Some(String::from("Mozilla/5.0")),
            metadata: HashMap::new(),
        };

        let result = manager.record_consent(subject_id, consent).await;
        assert!(result.is_ok());

        // Check if processing is allowed
        let allowed = manager
            .is_processing_allowed(subject_id, ProcessingPurpose::Analytics)
            .await
            .unwrap();
        assert!(allowed);

        // Withdraw consent
        let withdraw_result = manager
            .withdraw_consent(subject_id, ProcessingPurpose::Analytics)
            .await;
        assert!(withdraw_result.is_ok());

        // Check if processing is no longer allowed
        let not_allowed = manager
            .is_processing_allowed(subject_id, ProcessingPurpose::Analytics)
            .await
            .unwrap();
        assert!(!not_allowed);
    }

    #[tokio::test]
    async fn test_data_request_processing() {
        let manager = GdprComplianceManager::new();

        let request = DataRequest {
            request_id: Uuid::new_v4(),
            subject_id: String::from("test_user_003"),
            request_type: DataRequestType::Access,
            status: RequestStatus::Pending,
            description: String::from("Request access to all my data"),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            expected_completion: Utc::now() + ChronoDuration::days(30),
            response: None,
            associated_data: HashMap::new(),
        };

        let request_id = manager.create_data_request(request).await.unwrap();
        let process_result = manager.process_data_request(request_id).await;
        assert!(process_result.is_ok());
    }

    #[tokio::test]
    async fn test_data_anonymization() {
        let manager = GdprComplianceManager::new();
        let subject_id = "test_user_004";

        let subject = DataSubject {
            subject_id: subject_id.to_string(),
            email: Some(String::from("sensitive@example.com")),
            language: Some(String::from("en")),
            registered_at: Utc::now(),
            last_active_at: Utc::now(),
            consent_records: Vec::new(),
            retention_settings: RetentionSettings::default(),
            privacy_preferences: PrivacyPreferences::default(),
            active_requests: Vec::new(),
        };

        manager.register_data_subject(subject).await.unwrap();

        let anonymize_result = manager.anonymize_subject_data(subject_id).await;
        assert!(anonymize_result.is_ok());
    }

    #[tokio::test]
    async fn test_data_deletion() {
        let manager = GdprComplianceManager::new();
        let subject_id = "test_user_005";

        let subject = DataSubject {
            subject_id: subject_id.to_string(),
            email: Some(String::from("delete@example.com")),
            language: Some(String::from("en")),
            registered_at: Utc::now(),
            last_active_at: Utc::now(),
            consent_records: Vec::new(),
            retention_settings: RetentionSettings::default(),
            privacy_preferences: PrivacyPreferences::default(),
            active_requests: Vec::new(),
        };

        manager.register_data_subject(subject).await.unwrap();

        let delete_result = manager.delete_subject_data(subject_id).await;
        assert!(delete_result.is_ok());

        // Verify deletion
        let export_result = manager.export_subject_data(subject_id).await;
        assert!(export_result.is_err());
    }

    #[tokio::test]
    async fn test_retention_compliance() {
        let manager = GdprComplianceManager::new();

        let violations = manager.check_retention_compliance().await.unwrap();
        assert!(violations.is_empty()); // No subjects, no violations
    }

    #[tokio::test]
    async fn test_processing_activity_recording() {
        let manager = GdprComplianceManager::new();

        let activity = ProcessingActivity {
            activity_id: Uuid::new_v4(),
            subject_id: String::from("test_user_006"),
            purpose: ProcessingPurpose::Analytics,
            legal_basis: LegalBasis::Consent,
            data_categories: vec![String::from("feedback"), String::from("progress")],
            processed_at: Utc::now(),
            data_source: String::from("feedback_system"),
            data_destination: String::from("analytics_db"),
            retention_period: Some(ChronoDuration::days(365)),
            security_measures: vec![String::from("encryption"), String::from("access_control")],
        };

        let result = manager.record_processing_activity(activity).await;
        assert!(result.is_ok());

        let activities = manager
            .get_processing_activities("test_user_006")
            .await
            .unwrap();
        assert_eq!(activities.len(), 1);
    }

    #[tokio::test]
    async fn test_data_breach_reporting() {
        let manager = GdprComplianceManager::new();

        let breach = DataBreach {
            breach_id: Uuid::new_v4(),
            discovered_at: Utc::now(),
            occurred_at: Utc::now() - ChronoDuration::hours(2),
            affected_subjects: vec![String::from("user001"), String::from("user002")],
            description: String::from("Unauthorized access to user data"),
            severity: BreachSeverity::High,
            affected_data_categories: vec![String::from("email"), String::from("feedback")],
            containment_measures: vec![
                String::from("access_revoked"),
                String::from("passwords_reset"),
            ],
            notification_status: NotificationStatus::Pending,
            regulatory_reporting_required: true,
            response_timeline: Vec::new(),
        };

        let result = manager.report_data_breach(breach).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_compliance_report_generation() {
        let manager = GdprComplianceManager::new();

        let from = Utc::now() - ChronoDuration::days(30);
        let to = Utc::now();

        let report = manager.get_compliance_report(from, to).await.unwrap();
        assert_eq!(report.total_subjects, 0);
        assert_eq!(report.compliance_score, 95.0);
    }
}
