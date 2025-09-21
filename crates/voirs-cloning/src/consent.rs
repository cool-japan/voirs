//! Consent Management System for Voice Cloning
//!
//! This module provides comprehensive consent management capabilities to ensure ethical
//! use of voice cloning technology. It includes consent verification, digital signatures,
//! usage restrictions, and compliance tracking.

use crate::{usage_tracking::CloningOperationType, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// Comprehensive consent record for voice cloning operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRecord {
    /// Unique identifier for this consent record
    pub consent_id: Uuid,

    /// Identity of the person providing consent
    pub subject_identity: SubjectIdentity,

    /// Type and scope of consent granted
    pub consent_type: ConsentType,

    /// Specific permissions granted
    pub permissions: ConsentPermissions,

    /// Consent verification information
    pub verification: ConsentVerification,

    /// Usage restrictions and limitations
    pub restrictions: UsageRestrictions,

    /// Timestamps for consent lifecycle
    pub timestamps: ConsentTimestamps,

    /// Legal and compliance information
    pub legal_info: LegalInformation,

    /// Current status of the consent
    pub status: ConsentStatus,

    /// Digital signatures and cryptographic proofs
    pub signatures: Vec<DigitalSignature>,

    /// Metadata and additional context
    pub metadata: HashMap<String, String>,

    /// Usage tracking
    pub times_used: usize,
}

/// Identity information for the consent subject
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectIdentity {
    /// Primary identifier (anonymized or pseudonymized)
    pub subject_id: String,

    /// Verification method used
    pub verification_method: IdentityVerificationMethod,

    /// Verification status
    pub verification_status: VerificationStatus,

    /// Biometric identifiers (hashed)
    pub biometric_hash: Option<String>,

    /// Legal name (encrypted)
    pub encrypted_name: Option<Vec<u8>>,

    /// Contact information (encrypted)
    pub encrypted_contact: Option<Vec<u8>>,
}

/// Methods for verifying subject identity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IdentityVerificationMethod {
    /// Voice biometric verification
    VoiceBiometric,
    /// Government-issued ID verification
    GovernmentId,
    /// Digital signature verification
    DigitalSignature,
    /// Multi-factor authentication
    MultiFactorAuth,
    /// Notarized verification
    NotarizedVerification,
    /// Witnessed verification
    WitnessedVerification,
}

/// Verification status of the subject's identity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationStatus {
    Unverified,
    Pending,
    Verified,
    Failed,
    Expired,
}

/// Types of consent that can be granted
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsentType {
    /// Full unrestricted consent
    FullConsent,
    /// Limited consent with specific restrictions
    LimitedConsent,
    /// Research-only consent
    ResearchOnly,
    /// Commercial use consent
    CommercialUse,
    /// Personal use consent
    PersonalUse,
    /// Educational use consent
    EducationalUse,
    /// Temporary consent with expiration
    TemporaryConsent,
    /// Voice cloning specific consent
    VoiceCloning,
}

/// Specific permissions granted in the consent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentPermissions {
    /// Allow voice model training
    pub allow_training: bool,
    /// Allow voice synthesis
    pub allow_synthesis: bool,
    /// Allow voice adaptation
    pub allow_adaptation: bool,
    /// Allow commercial use
    pub allow_commercial: bool,
    /// Allow research use
    pub allow_research: bool,
    /// Allow distribution/sharing
    pub allow_distribution: bool,
    /// Allow modification/editing
    pub allow_modification: bool,
    /// Allow long-term storage
    pub allow_storage: bool,
    /// Specific use cases permitted
    pub permitted_use_cases: HashSet<String>,
    /// Specific applications permitted
    pub permitted_applications: HashSet<String>,
}

/// Consent verification details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentVerification {
    /// Method used for verification
    pub method: ConsentVerificationMethod,
    /// Verification status
    pub status: ConsentVerificationStatus,
    /// Cryptographic proof of consent
    pub cryptographic_proof: Option<String>,
    /// Witness information
    pub witnesses: Vec<WitnessInfo>,
    /// Third-party verification
    pub third_party_verification: Option<ThirdPartyVerification>,
    /// Biometric verification
    pub biometric_verification: Option<BiometricVerification>,
}

/// Methods for verifying consent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsentVerificationMethod {
    /// Digital signature
    DigitalSignature,
    /// Biometric verification
    BiometricAuth,
    /// Notarized consent
    NotarizedConsent,
    /// Witnessed consent
    WitnessedConsent,
    /// Video recorded consent
    VideoRecording,
    /// Multi-step verification
    MultiStepVerification,
}

/// Status of consent verification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConsentVerificationStatus {
    Pending,
    Verified,
    Failed,
    Expired,
    Revoked,
    Invalid,
}

impl ConsentVerificationStatus {
    /// Check if the verification status is valid
    pub fn is_valid(&self) -> bool {
        matches!(self, ConsentVerificationStatus::Verified)
    }
}

/// Witness information for consent verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WitnessInfo {
    pub witness_id: String,
    pub witness_name: Option<String>,
    pub witness_signature: Option<String>,
    pub witness_timestamp: SystemTime,
    pub witness_contact: Option<String>,
}

/// Third-party verification information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThirdPartyVerification {
    pub verifier_name: String,
    pub verifier_id: String,
    pub verification_method: String,
    pub verification_timestamp: SystemTime,
    pub verification_certificate: Option<String>,
}

/// Biometric verification information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiometricVerification {
    pub biometric_type: BiometricType,
    pub verification_score: f64,
    pub verification_threshold: f64,
    pub verification_timestamp: SystemTime,
    pub biometric_hash: String,
}

/// Types of biometric verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BiometricType {
    Voice,
    Fingerprint,
    FacialRecognition,
    Iris,
    Combined,
}

/// Usage restrictions and limitations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRestrictions {
    /// Geographical restrictions
    pub geographical_restrictions: Option<GeographicalRestrictions>,
    /// Temporal restrictions
    pub temporal_restrictions: Option<TemporalRestrictions>,
    /// Usage frequency limits
    pub frequency_limits: Option<FrequencyLimits>,
    /// Content restrictions
    pub content_restrictions: Option<ContentRestrictions>,
    /// Distribution restrictions
    pub distribution_restrictions: Option<DistributionRestrictions>,
    /// Purpose restrictions
    pub purpose_restrictions: HashSet<String>,
    /// Prohibited uses
    pub prohibited_uses: HashSet<String>,
}

/// Geographical usage restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeographicalRestrictions {
    pub allowed_countries: Option<HashSet<String>>,
    pub prohibited_countries: Option<HashSet<String>>,
    pub allowed_regions: Option<HashSet<String>>,
    pub prohibited_regions: Option<HashSet<String>>,
}

/// Temporal usage restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalRestrictions {
    pub valid_from: Option<SystemTime>,
    pub valid_until: Option<SystemTime>,
    pub allowed_hours: Option<Vec<u8>>, // 0-23
    pub allowed_days: Option<Vec<u8>>,  // 0-6, Sunday=0
    pub time_zone: Option<String>,
}

/// Usage frequency limitations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyLimits {
    pub max_uses_per_day: Option<u32>,
    pub max_uses_per_week: Option<u32>,
    pub max_uses_per_month: Option<u32>,
    pub max_total_uses: Option<u32>,
    pub cooldown_period: Option<Duration>,
}

/// Content-based restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentRestrictions {
    pub prohibited_words: Option<HashSet<String>>,
    pub prohibited_phrases: Option<HashSet<String>>,
    pub prohibited_topics: Option<HashSet<String>>,
    pub content_rating_limits: Option<Vec<String>>,
    pub language_restrictions: Option<HashSet<String>>,
}

/// Distribution restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionRestrictions {
    pub allow_public_distribution: bool,
    pub allow_commercial_distribution: bool,
    pub allowed_platforms: Option<HashSet<String>>,
    pub prohibited_platforms: Option<HashSet<String>>,
    pub require_attribution: bool,
    pub attribution_text: Option<String>,
}

/// Consent lifecycle timestamps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentTimestamps {
    pub created_at: SystemTime,
    pub granted_at: Option<SystemTime>,
    pub expires_at: Option<SystemTime>,
    pub last_modified: SystemTime,
    pub last_verified: Option<SystemTime>,
    pub revoked_at: Option<SystemTime>,
}

/// Legal and compliance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalInformation {
    pub jurisdiction: String,
    pub applicable_laws: Vec<String>,
    pub compliance_standards: Vec<String>,
    pub privacy_policy_version: String,
    pub terms_of_service_version: String,
    pub legal_basis: LegalBasis,
    pub data_controller: String,
    pub data_processor: Option<String>,
}

/// Legal basis for processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LegalBasis {
    Consent,
    Contract,
    LegalObligation,
    VitalInterests,
    PublicTask,
    LegitimateInterests,
}

/// Current status of consent
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConsentStatus {
    Draft,
    PendingVerification,
    Active,
    Expired,
    Revoked,
    Suspended,
    Terminated,
}

/// Digital signature information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigitalSignature {
    pub signature_id: Uuid,
    pub signer_identity: String,
    pub signature_algorithm: String,
    pub signature_value: Vec<u8>,
    pub certificate: Option<Vec<u8>>,
    pub timestamp: SystemTime,
    pub signature_type: SignatureType,
}

/// Types of digital signatures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignatureType {
    SubjectSignature,
    WitnessSignature,
    NotarySignature,
    OrganizationSignature,
    SystemSignature,
}

/// Consent management system
pub struct ConsentManager {
    /// Storage for consent records
    consent_store: HashMap<Uuid, ConsentRecord>,
    /// Index by subject ID
    subject_index: HashMap<String, HashSet<Uuid>>,
    /// Verification providers
    verification_providers: HashMap<String, Box<dyn ConsentVerificationProvider>>,
    /// Cryptographic signing service
    signing_service: Option<Box<dyn DigitalSigningService>>,
    /// Audit logger
    audit_logger: Option<Box<dyn ConsentAuditLogger>>,
}

/// Trait for consent verification providers
pub trait ConsentVerificationProvider: Send + Sync {
    fn verify_consent(&self, consent: &ConsentRecord) -> Result<ConsentVerificationStatus>;
    fn get_provider_name(&self) -> &str;
    fn supports_method(&self, method: &ConsentVerificationMethod) -> bool;
}

/// Trait for digital signing services
pub trait DigitalSigningService: Send + Sync {
    fn sign_consent(&self, consent: &ConsentRecord) -> Result<DigitalSignature>;
    fn verify_signature(&self, signature: &DigitalSignature, data: &[u8]) -> Result<bool>;
    fn get_certificate(&self, signature_id: &Uuid) -> Result<Option<Vec<u8>>>;
}

/// Trait for consent audit logging
pub trait ConsentAuditLogger: Send + Sync {
    fn log_consent_action(&self, action: ConsentAuditAction) -> Result<()>;
    fn log_access(&self, access: ConsentAccessLog) -> Result<()>;
    fn log_violation(&self, violation: ConsentViolationLog) -> Result<()>;
}

/// Consent audit actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentAuditAction {
    pub action_id: Uuid,
    pub consent_id: Uuid,
    pub action_type: ConsentActionType,
    pub actor: String,
    pub timestamp: SystemTime,
    pub details: HashMap<String, String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
}

/// Types of consent actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsentActionType {
    ConsentCreated,
    ConsentGranted,
    ConsentModified,
    ConsentRevoked,
    ConsentExpired,
    ConsentVerified,
    ConsentAccessed,
    ConsentViolated,
}

/// Consent access logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentAccessLog {
    pub access_id: Uuid,
    pub consent_id: Uuid,
    pub accessor: String,
    pub access_type: ConsentAccessType,
    pub timestamp: SystemTime,
    pub purpose: String,
    pub ip_address: Option<String>,
    pub result: AccessResult,
}

/// Types of consent access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsentAccessType {
    Read,
    Verify,
    Use,
    Modify,
    Export,
}

/// Result of consent access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessResult {
    Allowed,
    Denied,
    Restricted,
}

/// Consent violation logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentViolationLog {
    pub violation_id: Uuid,
    pub consent_id: Uuid,
    pub violation_type: ConsentViolationType,
    pub severity: ViolationSeverity,
    pub description: String,
    pub timestamp: SystemTime,
    pub detected_by: String,
    pub remedial_action: Option<String>,
}

/// Types of consent violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsentViolationType {
    UnauthorizedUse,
    ExceededPermissions,
    ExpiredConsent,
    RevokedConsent,
    GeographicalViolation,
    TemporalViolation,
    FrequencyViolation,
    ContentViolation,
    DistributionViolation,
}

/// Severity levels for violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl ConsentManager {
    /// Create a new consent manager
    pub fn new() -> Self {
        ConsentManager {
            consent_store: HashMap::new(),
            subject_index: HashMap::new(),
            verification_providers: HashMap::new(),
            signing_service: None,
            audit_logger: None,
        }
    }

    /// Register a verification provider
    pub fn register_verification_provider(
        &mut self,
        name: String,
        provider: Box<dyn ConsentVerificationProvider>,
    ) {
        self.verification_providers.insert(name, provider);
    }

    /// Set digital signing service
    pub fn set_signing_service(&mut self, service: Box<dyn DigitalSigningService>) {
        self.signing_service = Some(service);
    }

    /// Set audit logger
    pub fn set_audit_logger(&mut self, logger: Box<dyn ConsentAuditLogger>) {
        self.audit_logger = Some(logger);
    }

    /// Create a new consent record
    pub fn create_consent(&mut self, subject_identity: SubjectIdentity) -> Result<Uuid> {
        let consent_id = Uuid::new_v4();
        let now = SystemTime::now();

        let consent = ConsentRecord {
            consent_id,
            subject_identity: subject_identity.clone(),
            consent_type: ConsentType::LimitedConsent,
            permissions: ConsentPermissions::default(),
            verification: ConsentVerification {
                method: ConsentVerificationMethod::DigitalSignature,
                status: ConsentVerificationStatus::Pending,
                cryptographic_proof: None,
                witnesses: Vec::new(),
                third_party_verification: None,
                biometric_verification: None,
            },
            restrictions: UsageRestrictions::default(),
            timestamps: ConsentTimestamps {
                created_at: now,
                granted_at: None,
                expires_at: None,
                last_modified: now,
                last_verified: None,
                revoked_at: None,
            },
            legal_info: LegalInformation::default(),
            status: ConsentStatus::Draft,
            signatures: Vec::new(),
            metadata: HashMap::new(),
            times_used: 0,
        };

        // Index by subject ID
        self.subject_index
            .entry(subject_identity.subject_id)
            .or_insert_with(HashSet::new)
            .insert(consent_id);

        // Store consent
        self.consent_store.insert(consent_id, consent);

        // Log audit action
        if let Some(ref logger) = self.audit_logger {
            let action = ConsentAuditAction {
                action_id: Uuid::new_v4(),
                consent_id,
                action_type: ConsentActionType::ConsentCreated,
                actor: "system".to_string(),
                timestamp: now,
                details: HashMap::new(),
                ip_address: None,
                user_agent: None,
            };
            let _ = logger.log_consent_action(action);
        }

        Ok(consent_id)
    }

    /// Grant consent with specified permissions
    pub fn grant_consent(
        &mut self,
        consent_id: Uuid,
        consent_type: ConsentType,
        permissions: ConsentPermissions,
        restrictions: Option<UsageRestrictions>,
    ) -> Result<()> {
        let consent = self
            .consent_store
            .get_mut(&consent_id)
            .ok_or_else(|| Error::Validation("Consent record not found".to_string()))?;

        let now = SystemTime::now();

        consent.consent_type = consent_type;
        consent.permissions = permissions;
        if let Some(restrictions) = restrictions {
            consent.restrictions = restrictions;
        }
        consent.status = ConsentStatus::PendingVerification;
        consent.timestamps.granted_at = Some(now);
        consent.timestamps.last_modified = now;

        // Log audit action
        if let Some(ref logger) = self.audit_logger {
            let action = ConsentAuditAction {
                action_id: Uuid::new_v4(),
                consent_id,
                action_type: ConsentActionType::ConsentGranted,
                actor: consent.subject_identity.subject_id.clone(),
                timestamp: now,
                details: HashMap::new(),
                ip_address: None,
                user_agent: None,
            };
            let _ = logger.log_consent_action(action);
        }

        Ok(())
    }

    /// Verify consent using configured providers
    pub async fn verify_consent(
        &mut self,
        consent_id: &Uuid,
        context: &ConsentUsageContext,
    ) -> Result<ConsentVerificationStatus> {
        let consent = self
            .consent_store
            .get_mut(consent_id)
            .ok_or_else(|| Error::Validation("Consent record not found".to_string()))?;

        // Check if consent has expired
        if let Some(expires_at) = consent.timestamps.expires_at {
            if SystemTime::now() > expires_at {
                consent.status = ConsentStatus::Expired;
                return Ok(ConsentVerificationStatus::Expired);
            }
        }

        // Verify geographical restrictions
        if let Some(geo_restrictions) = &consent.restrictions.geographical_restrictions {
            if let Some(location) = &context.location {
                if let Some(ref prohibited_countries) = geo_restrictions.prohibited_countries {
                    if prohibited_countries.contains(location) {
                        return Ok(ConsentVerificationStatus::Invalid);
                    }
                }
            }
        }

        // Verify purpose restrictions
        if let Some(purpose) = context.additional_context.get("purpose") {
            if consent.restrictions.prohibited_uses.contains(purpose) {
                return Ok(ConsentVerificationStatus::Invalid);
            }
        }

        // Find appropriate verification provider
        let provider = self
            .verification_providers
            .values()
            .find(|p| p.supports_method(&consent.verification.method))
            .ok_or_else(|| {
                Error::Verification("No suitable verification provider found".to_string())
            })?;

        // Verify consent
        let status = provider.verify_consent(consent)?;

        let now = SystemTime::now();
        consent.verification.status = status.clone();
        consent.timestamps.last_verified = Some(now);
        consent.timestamps.last_modified = now;

        // Update overall consent status based on verification
        consent.status = match status {
            ConsentVerificationStatus::Verified => {
                consent.times_used += 1;
                ConsentStatus::Active
            }
            ConsentVerificationStatus::Failed => ConsentStatus::Suspended,
            ConsentVerificationStatus::Pending => ConsentStatus::PendingVerification,
            _ => consent.status.clone(),
        };

        // Log audit action
        if let Some(ref logger) = self.audit_logger {
            let action = ConsentAuditAction {
                action_id: Uuid::new_v4(),
                consent_id: *consent_id,
                action_type: ConsentActionType::ConsentVerified,
                actor: "system".to_string(),
                timestamp: now,
                details: [("status".to_string(), format!("{:?}", status))]
                    .into_iter()
                    .collect(),
                ip_address: None,
                user_agent: None,
            };
            let _ = logger.log_consent_action(action);
        }

        Ok(status)
    }

    /// Revoke consent
    pub fn revoke_consent(&mut self, consent_id: Uuid, reason: String) -> Result<()> {
        let consent = self
            .consent_store
            .get_mut(&consent_id)
            .ok_or_else(|| Error::Validation("Consent record not found".to_string()))?;

        let now = SystemTime::now();
        consent.status = ConsentStatus::Revoked;
        consent.timestamps.revoked_at = Some(now);
        consent.timestamps.last_modified = now;
        consent
            .metadata
            .insert("revocation_reason".to_string(), reason);

        // Log audit action
        if let Some(ref logger) = self.audit_logger {
            let action = ConsentAuditAction {
                action_id: Uuid::new_v4(),
                consent_id,
                action_type: ConsentActionType::ConsentRevoked,
                actor: consent.subject_identity.subject_id.clone(),
                timestamp: now,
                details: HashMap::new(),
                ip_address: None,
                user_agent: None,
            };
            let _ = logger.log_consent_action(action);
        }

        Ok(())
    }

    /// Check if consent allows a specific use
    pub fn check_consent_for_use(
        &self,
        consent_id: Uuid,
        use_case: &str,
        context: &ConsentUsageContext,
    ) -> Result<ConsentUsageResult> {
        let consent = self
            .consent_store
            .get(&consent_id)
            .ok_or_else(|| Error::Validation("Consent record not found".to_string()))?;

        // Check consent status
        match consent.status {
            ConsentStatus::Active => {}
            ConsentStatus::Expired => {
                return Ok(ConsentUsageResult::Denied("Consent expired".to_string()))
            }
            ConsentStatus::Revoked => {
                return Ok(ConsentUsageResult::Denied("Consent revoked".to_string()))
            }
            ConsentStatus::Suspended => {
                return Ok(ConsentUsageResult::Denied("Consent suspended".to_string()))
            }
            _ => return Ok(ConsentUsageResult::Denied("Consent not active".to_string())),
        }

        // Check expiration
        if let Some(expires_at) = consent.timestamps.expires_at {
            if SystemTime::now() > expires_at {
                return Ok(ConsentUsageResult::Denied("Consent expired".to_string()));
            }
        }

        // Check permissions
        let usage_allowed = match use_case {
            "training" => consent.permissions.allow_training,
            "synthesis" => consent.permissions.allow_synthesis,
            "adaptation" => consent.permissions.allow_adaptation,
            "commercial" => consent.permissions.allow_commercial,
            "research" => consent.permissions.allow_research,
            "distribution" => consent.permissions.allow_distribution,
            "modification" => consent.permissions.allow_modification,
            "storage" => consent.permissions.allow_storage,
            _ => consent.permissions.permitted_use_cases.contains(use_case),
        };

        if !usage_allowed {
            return Ok(ConsentUsageResult::Denied(format!(
                "Use case '{}' not permitted",
                use_case
            )));
        }

        // Check restrictions
        if let Some(violation) = self.check_restrictions(consent, context)? {
            return Ok(ConsentUsageResult::Restricted(violation));
        }

        Ok(ConsentUsageResult::Allowed)
    }

    /// Check usage restrictions
    fn check_restrictions(
        &self,
        consent: &ConsentRecord,
        context: &ConsentUsageContext,
    ) -> Result<Option<String>> {
        let restrictions = &consent.restrictions;

        // Check geographical restrictions
        if let Some(ref geo_restrictions) = restrictions.geographical_restrictions {
            if let Some(ref country) = context.country {
                if let Some(ref prohibited) = geo_restrictions.prohibited_countries {
                    if prohibited.contains(country) {
                        return Ok(Some("Usage prohibited in this country".to_string()));
                    }
                }
                if let Some(ref allowed) = geo_restrictions.allowed_countries {
                    if !allowed.contains(country) {
                        return Ok(Some("Usage not allowed in this country".to_string()));
                    }
                }
            }
        }

        // Check temporal restrictions
        if let Some(ref temporal) = restrictions.temporal_restrictions {
            let now = SystemTime::now();

            if let Some(valid_from) = temporal.valid_from {
                if now < valid_from {
                    return Ok(Some("Consent not yet valid".to_string()));
                }
            }

            if let Some(valid_until) = temporal.valid_until {
                if now > valid_until {
                    return Ok(Some("Consent validity period expired".to_string()));
                }
            }
        }

        // Check content restrictions
        if let Some(ref content) = restrictions.content_restrictions {
            if let Some(ref text) = context.content_text {
                if let Some(ref prohibited_words) = content.prohibited_words {
                    for word in prohibited_words {
                        if text.to_lowercase().contains(&word.to_lowercase()) {
                            return Ok(Some(format!("Content contains prohibited word: {}", word)));
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    /// Get consent record
    pub fn get_consent(&self, consent_id: Uuid) -> Option<&ConsentRecord> {
        self.consent_store.get(&consent_id)
    }

    /// Get consent records by subject
    pub fn get_consents_by_subject(&self, subject_id: &str) -> Vec<&ConsentRecord> {
        if let Some(consent_ids) = self.subject_index.get(subject_id) {
            consent_ids
                .iter()
                .filter_map(|id| self.consent_store.get(id))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Export consent record for compliance
    pub fn export_consent(&self, consent_id: Uuid) -> Result<String> {
        let consent = self
            .consent_store
            .get(&consent_id)
            .ok_or_else(|| Error::Validation("Consent record not found".to_string()))?;

        serde_json::to_string_pretty(consent).map_err(|e| Error::Serialization(e))
    }

    /// Get consent statistics for a specific consent
    pub async fn get_consent_statistics(&self, consent_id: &Uuid) -> Result<ConsentStatistics> {
        let consent = self
            .consent_store
            .get(consent_id)
            .ok_or_else(|| Error::Validation("Consent record not found".to_string()))?;

        let mut stats = ConsentStatistics::default();
        stats.total_consents = 1;
        stats.times_used = consent.times_used;

        match consent.status {
            ConsentStatus::Active => stats.active_consents = 1,
            ConsentStatus::Expired => stats.expired_consents = 1,
            ConsentStatus::Revoked => stats.revoked_consents = 1,
            ConsentStatus::PendingVerification => stats.pending_consents = 1,
            _ => {}
        }

        match consent.verification.status {
            ConsentVerificationStatus::Verified => stats.verified_consents = 1,
            ConsentVerificationStatus::Failed => stats.failed_verifications = 1,
            _ => {}
        }

        Ok(stats)
    }
}

/// Context for consent usage checks
#[derive(Debug, Clone)]
pub struct ConsentUsageContext {
    pub use_case: String,
    pub application: Option<String>,
    pub user: Option<String>,
    pub country: Option<String>,
    pub region: Option<String>,
    pub content_text: Option<String>,
    pub timestamp: SystemTime,
    pub ip_address: Option<String>,
    // Additional fields expected by tests
    pub operation_type: CloningOperationType,
    pub user_id: String,
    pub location: Option<String>,
    pub additional_context: HashMap<String, String>,
}

/// Result of consent usage check
#[derive(Debug, Clone)]
pub enum ConsentUsageResult {
    Allowed,
    Denied(String),
    Restricted(String),
}

/// Consent usage statistics
#[derive(Debug, Default, Clone)]
pub struct ConsentStatistics {
    pub total_consents: usize,
    pub active_consents: usize,
    pub expired_consents: usize,
    pub revoked_consents: usize,
    pub pending_consents: usize,
    pub verified_consents: usize,
    pub failed_verifications: usize,
    pub times_used: usize,
}

// Default implementations
impl Default for ConsentPermissions {
    fn default() -> Self {
        ConsentPermissions {
            allow_training: false,
            allow_synthesis: false,
            allow_adaptation: false,
            allow_commercial: false,
            allow_research: false,
            allow_distribution: false,
            allow_modification: false,
            allow_storage: false,
            permitted_use_cases: HashSet::new(),
            permitted_applications: HashSet::new(),
        }
    }
}

impl Default for UsageRestrictions {
    fn default() -> Self {
        UsageRestrictions {
            geographical_restrictions: None,
            temporal_restrictions: None,
            frequency_limits: None,
            content_restrictions: None,
            distribution_restrictions: None,
            purpose_restrictions: HashSet::new(),
            prohibited_uses: HashSet::new(),
        }
    }
}

impl Default for LegalInformation {
    fn default() -> Self {
        LegalInformation {
            jurisdiction: "US".to_string(),
            applicable_laws: vec!["GDPR".to_string(), "CCPA".to_string()],
            compliance_standards: vec!["SOC2".to_string(), "ISO27001".to_string()],
            privacy_policy_version: "1.0".to_string(),
            terms_of_service_version: "1.0".to_string(),
            legal_basis: LegalBasis::Consent,
            data_controller: "VoiRS Corporation".to_string(),
            data_processor: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consent_creation() {
        let mut manager = ConsentManager::new();
        let subject_identity = SubjectIdentity {
            subject_id: "test-subject-001".to_string(),
            verification_method: IdentityVerificationMethod::DigitalSignature,
            verification_status: VerificationStatus::Pending,
            biometric_hash: None,
            encrypted_name: None,
            encrypted_contact: None,
        };

        let consent_id = manager.create_consent(subject_identity).unwrap();
        let consent = manager.get_consent(consent_id).unwrap();

        assert_eq!(consent.status, ConsentStatus::Draft);
        assert_eq!(consent.subject_identity.subject_id, "test-subject-001");
    }

    #[test]
    fn test_consent_granting() {
        let mut manager = ConsentManager::new();
        let subject_identity = SubjectIdentity {
            subject_id: "test-subject-002".to_string(),
            verification_method: IdentityVerificationMethod::DigitalSignature,
            verification_status: VerificationStatus::Verified,
            biometric_hash: None,
            encrypted_name: None,
            encrypted_contact: None,
        };

        let consent_id = manager.create_consent(subject_identity).unwrap();

        let mut permissions = ConsentPermissions::default();
        permissions.allow_research = true;
        permissions.allow_synthesis = true;

        manager
            .grant_consent(consent_id, ConsentType::ResearchOnly, permissions, None)
            .unwrap();

        let consent = manager.get_consent(consent_id).unwrap();
        assert_eq!(consent.status, ConsentStatus::PendingVerification);
        assert!(consent.permissions.allow_research);
        assert!(consent.permissions.allow_synthesis);
        assert!(!consent.permissions.allow_commercial);
    }

    #[test]
    fn test_consent_usage_check() {
        let mut manager = ConsentManager::new();
        let subject_identity = SubjectIdentity {
            subject_id: "test-subject-003".to_string(),
            verification_method: IdentityVerificationMethod::DigitalSignature,
            verification_status: VerificationStatus::Verified,
            biometric_hash: None,
            encrypted_name: None,
            encrypted_contact: None,
        };

        let consent_id = manager.create_consent(subject_identity).unwrap();

        let mut permissions = ConsentPermissions::default();
        permissions.allow_research = true;

        manager
            .grant_consent(consent_id, ConsentType::ResearchOnly, permissions, None)
            .unwrap();

        // Manually set status to Active for testing
        if let Some(consent) = manager.consent_store.get_mut(&consent_id) {
            consent.status = ConsentStatus::Active;
        }

        let context = ConsentUsageContext {
            use_case: "research".to_string(),
            application: None,
            user: None,
            country: None,
            region: None,
            content_text: None,
            timestamp: SystemTime::now(),
            ip_address: None,
            // Additional fields
            operation_type: CloningOperationType::VoiceCloning,
            user_id: "test_user".to_string(),
            location: None,
            additional_context: HashMap::new(),
        };

        let result = manager
            .check_consent_for_use(consent_id, "research", &context)
            .unwrap();
        assert!(matches!(result, ConsentUsageResult::Allowed));

        let result = manager
            .check_consent_for_use(consent_id, "commercial", &context)
            .unwrap();
        assert!(matches!(result, ConsentUsageResult::Denied(_)));
    }
}
