//! # Validation Certificates Generator
//!
//! This module provides functionality for generating cryptographically signed validation
//! certificates for evaluation results that meet specified quality standards.
//!
//! ## Features
//!
//! - Certificate generation for various validation types
//! - Cryptographic signing and verification
//! - Multiple output formats (Text, JSON, XML)
//! - Certificate revocation management
//! - Validity period enforcement
//! - Comprehensive metadata tracking
//!
//! ## Example
//!
//! ```rust
//! use voirs_evaluation::validation_certificates::{
//!     CertificateGenerator, CertificateType, CertificateFormat,
//! };
//! use std::collections::HashMap;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create certificate generator
//! let mut generator = CertificateGenerator::new("VoiRS Evaluation Authority");
//!
//! // Prepare validation results
//! let mut results = HashMap::new();
//! results.insert("pesq_score".to_string(), 4.2);
//! results.insert("stoi_score".to_string(), 0.95);
//!
//! // Generate certificate
//! let certificate = generator.generate_certificate(
//!     CertificateType::QualityValidation,
//!     "test-model-v1",
//!     results,
//!     30, // Valid for 30 days
//! )?;
//!
//! // Export certificate
//! let cert_text = generator.export_certificate(&certificate, CertificateFormat::Text)?;
//! println!("{}", cert_text);
//!
//! // Verify certificate
//! assert!(generator.verify_certificate(&certificate)?);
//! # Ok(())
//! # }
//! ```

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during certificate operations
#[derive(Debug, Error)]
pub enum CertificateError {
    /// Certificate generation failed
    #[error("Failed to generate certificate: {0}")]
    GenerationError(String),

    /// Certificate verification failed
    #[error("Certificate verification failed: {0}")]
    VerificationError(String),

    /// Certificate has expired
    #[error("Certificate has expired on {0}")]
    ExpiredCertificate(DateTime<Utc>),

    /// Certificate has been revoked
    #[error("Certificate has been revoked: {0}")]
    RevokedCertificate(String),

    /// Invalid certificate format
    #[error("Invalid certificate format: {0}")]
    InvalidFormat(String),

    /// Export error
    #[error("Failed to export certificate: {0}")]
    ExportError(String),

    /// Invalid signature
    #[error("Invalid certificate signature")]
    InvalidSignature,

    /// Certificate not found
    #[error("Certificate not found: {0}")]
    NotFound(String),
}

/// Type of validation certificate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CertificateType {
    /// Quality validation certificate
    QualityValidation,
    /// Compliance validation certificate
    ComplianceValidation,
    /// Performance validation certificate
    PerformanceValidation,
    /// Security validation certificate
    SecurityValidation,
    /// Reproducibility validation certificate
    ReproducibilityValidation,
    /// Comprehensive validation (all criteria)
    ComprehensiveValidation,
}

impl std::fmt::Display for CertificateType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QualityValidation => write!(f, "Quality Validation"),
            Self::ComplianceValidation => write!(f, "Compliance Validation"),
            Self::PerformanceValidation => write!(f, "Performance Validation"),
            Self::SecurityValidation => write!(f, "Security Validation"),
            Self::ReproducibilityValidation => write!(f, "Reproducibility Validation"),
            Self::ComprehensiveValidation => write!(f, "Comprehensive Validation"),
        }
    }
}

/// Certificate export format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CertificateFormat {
    /// Human-readable text format
    Text,
    /// JSON format
    Json,
    /// XML format
    Xml,
}

/// Validation certificate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCertificate {
    /// Unique certificate identifier
    pub certificate_id: String,
    /// Type of validation
    pub certificate_type: CertificateType,
    /// Subject of validation (model name, system ID, etc.)
    pub subject: String,
    /// Issuing authority
    pub issuer: String,
    /// Issue date and time
    pub issued_at: DateTime<Utc>,
    /// Expiration date and time
    pub expires_at: DateTime<Utc>,
    /// Validation results and metrics
    pub validation_results: HashMap<String, f64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Cryptographic signature
    pub signature: String,
    /// Certificate version
    pub version: String,
}

impl ValidationCertificate {
    /// Check if the certificate is currently valid (not expired)
    pub fn is_valid(&self) -> bool {
        Utc::now() < self.expires_at
    }

    /// Get remaining validity duration
    pub fn remaining_validity(&self) -> Option<Duration> {
        let now = Utc::now();
        if now < self.expires_at {
            Some(self.expires_at.signed_duration_since(now))
        } else {
            None
        }
    }

    /// Get certificate age
    pub fn age(&self) -> Duration {
        Utc::now().signed_duration_since(self.issued_at)
    }
}

/// Certificate revocation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevocationRecord {
    /// Certificate ID that was revoked
    pub certificate_id: String,
    /// Revocation timestamp
    pub revoked_at: DateTime<Utc>,
    /// Reason for revocation
    pub reason: String,
    /// Revoking authority
    pub revoked_by: String,
}

/// Certificate generator configuration
#[derive(Debug, Clone)]
pub struct CertificateConfig {
    /// Default validity period in days
    pub default_validity_days: i64,
    /// Certificate version
    pub version: String,
    /// Enable signature verification
    pub enable_signature_verification: bool,
    /// Minimum quality threshold for certificate issuance
    pub min_quality_threshold: f64,
}

impl Default for CertificateConfig {
    fn default() -> Self {
        Self {
            default_validity_days: 365,
            version: "1.0".to_string(),
            enable_signature_verification: true,
            min_quality_threshold: 0.8,
        }
    }
}

/// Certificate generator for validation results
pub struct CertificateGenerator {
    /// Issuing authority name
    issuer: String,
    /// Configuration
    config: CertificateConfig,
    /// Certificate registry
    certificates: HashMap<String, ValidationCertificate>,
    /// Revocation list
    revoked_certificates: HashMap<String, RevocationRecord>,
    /// Certificate counter for ID generation
    certificate_counter: u64,
}

impl CertificateGenerator {
    /// Create a new certificate generator
    pub fn new(issuer: impl Into<String>) -> Self {
        Self {
            issuer: issuer.into(),
            config: CertificateConfig::default(),
            certificates: HashMap::new(),
            revoked_certificates: HashMap::new(),
            certificate_counter: 0,
        }
    }

    /// Create a certificate generator with custom configuration
    pub fn with_config(issuer: impl Into<String>, config: CertificateConfig) -> Self {
        Self {
            issuer: issuer.into(),
            config,
            certificates: HashMap::new(),
            revoked_certificates: HashMap::new(),
            certificate_counter: 0,
        }
    }

    /// Generate a validation certificate
    pub fn generate_certificate(
        &mut self,
        certificate_type: CertificateType,
        subject: impl Into<String>,
        validation_results: HashMap<String, f64>,
        validity_days: i64,
    ) -> Result<ValidationCertificate, CertificateError> {
        // Check if results meet minimum threshold
        let overall_score = self.calculate_overall_score(&validation_results);
        if overall_score < self.config.min_quality_threshold {
            return Err(CertificateError::GenerationError(format!(
                "Validation results do not meet minimum threshold: {:.2} < {:.2}",
                overall_score, self.config.min_quality_threshold
            )));
        }

        // Generate certificate ID
        self.certificate_counter += 1;
        let certificate_id = format!(
            "VOIRS-{:04}-{:08X}",
            Utc::now().format("%Y"),
            self.certificate_counter
        );

        let now = Utc::now();
        let expires_at = now + Duration::days(validity_days);

        let subject = subject.into();
        let mut metadata = HashMap::new();
        metadata.insert("overall_score".to_string(), format!("{:.4}", overall_score));
        metadata.insert(
            "validation_count".to_string(),
            validation_results.len().to_string(),
        );

        // Generate signature
        let signature = self.generate_signature(&certificate_id, &subject, &now, &expires_at)?;

        let certificate = ValidationCertificate {
            certificate_id: certificate_id.clone(),
            certificate_type,
            subject,
            issuer: self.issuer.clone(),
            issued_at: now,
            expires_at,
            validation_results,
            metadata,
            signature,
            version: self.config.version.clone(),
        };

        // Store certificate
        self.certificates
            .insert(certificate_id, certificate.clone());

        Ok(certificate)
    }

    /// Verify a certificate's authenticity and validity
    pub fn verify_certificate(
        &self,
        certificate: &ValidationCertificate,
    ) -> Result<bool, CertificateError> {
        // Check if certificate has been revoked
        if self
            .revoked_certificates
            .contains_key(&certificate.certificate_id)
        {
            return Err(CertificateError::RevokedCertificate(
                certificate.certificate_id.clone(),
            ));
        }

        // Check expiration
        if !certificate.is_valid() {
            return Err(CertificateError::ExpiredCertificate(certificate.expires_at));
        }

        // Verify signature
        if self.config.enable_signature_verification {
            let expected_signature = self.generate_signature(
                &certificate.certificate_id,
                &certificate.subject,
                &certificate.issued_at,
                &certificate.expires_at,
            )?;

            if certificate.signature != expected_signature {
                return Err(CertificateError::InvalidSignature);
            }
        }

        Ok(true)
    }

    /// Revoke a certificate
    pub fn revoke_certificate(
        &mut self,
        certificate_id: impl Into<String>,
        reason: impl Into<String>,
    ) -> Result<(), CertificateError> {
        let certificate_id = certificate_id.into();

        // Check if certificate exists
        if !self.certificates.contains_key(&certificate_id) {
            return Err(CertificateError::NotFound(certificate_id));
        }

        // Check if already revoked
        if self.revoked_certificates.contains_key(&certificate_id) {
            return Err(CertificateError::RevokedCertificate(certificate_id));
        }

        // Create revocation record
        let record = RevocationRecord {
            certificate_id: certificate_id.clone(),
            revoked_at: Utc::now(),
            reason: reason.into(),
            revoked_by: self.issuer.clone(),
        };

        self.revoked_certificates.insert(certificate_id, record);
        Ok(())
    }

    /// Get a certificate by ID
    pub fn get_certificate(
        &self,
        certificate_id: &str,
    ) -> Result<&ValidationCertificate, CertificateError> {
        self.certificates
            .get(certificate_id)
            .ok_or_else(|| CertificateError::NotFound(certificate_id.to_string()))
    }

    /// Check if a certificate is revoked
    pub fn is_revoked(&self, certificate_id: &str) -> bool {
        self.revoked_certificates.contains_key(certificate_id)
    }

    /// Get revocation record if certificate is revoked
    pub fn get_revocation_record(&self, certificate_id: &str) -> Option<&RevocationRecord> {
        self.revoked_certificates.get(certificate_id)
    }

    /// Export certificate to specified format
    pub fn export_certificate(
        &self,
        certificate: &ValidationCertificate,
        format: CertificateFormat,
    ) -> Result<String, CertificateError> {
        match format {
            CertificateFormat::Text => self.export_text(certificate),
            CertificateFormat::Json => self.export_json(certificate),
            CertificateFormat::Xml => self.export_xml(certificate),
        }
    }

    /// List all valid (non-revoked, non-expired) certificates
    pub fn list_valid_certificates(&self) -> Vec<&ValidationCertificate> {
        self.certificates
            .values()
            .filter(|cert| {
                cert.is_valid() && !self.revoked_certificates.contains_key(&cert.certificate_id)
            })
            .collect()
    }

    /// Generate cryptographic signature (simplified implementation)
    fn generate_signature(
        &self,
        certificate_id: &str,
        subject: &str,
        issued_at: &DateTime<Utc>,
        expires_at: &DateTime<Utc>,
    ) -> Result<String, CertificateError> {
        // In production, use proper cryptographic signing (e.g., RSA, ECDSA)
        // This is a simplified hash-based signature for demonstration
        let data = format!(
            "{}:{}:{}:{}:{}",
            certificate_id,
            subject,
            self.issuer,
            issued_at.to_rfc3339(),
            expires_at.to_rfc3339()
        );

        // Use MD5 for simple hashing (in production, use SHA-256 or better)
        let digest = md5::compute(data.as_bytes());
        Ok(format!("{:x}", digest))
    }

    /// Calculate overall score from validation results
    fn calculate_overall_score(&self, results: &HashMap<String, f64>) -> f64 {
        if results.is_empty() {
            return 0.0;
        }

        let sum: f64 = results.values().sum();
        sum / results.len() as f64
    }

    /// Export certificate as human-readable text
    fn export_text(&self, certificate: &ValidationCertificate) -> Result<String, CertificateError> {
        let mut output = String::new();
        output.push_str("═══════════════════════════════════════════════════════════════════\n");
        output.push_str("                    VALIDATION CERTIFICATE                         \n");
        output.push_str("═══════════════════════════════════════════════════════════════════\n\n");

        output.push_str(&format!(
            "Certificate ID:    {}\n",
            certificate.certificate_id
        ));
        output.push_str(&format!(
            "Type:              {}\n",
            certificate.certificate_type
        ));
        output.push_str(&format!("Version:           {}\n", certificate.version));
        output.push_str(&format!("Subject:           {}\n", certificate.subject));
        output.push_str(&format!("Issuer:            {}\n", certificate.issuer));
        output.push_str(&format!(
            "Issued:            {}\n",
            certificate.issued_at.format("%Y-%m-%d %H:%M:%S UTC")
        ));
        output.push_str(&format!(
            "Expires:           {}\n",
            certificate.expires_at.format("%Y-%m-%d %H:%M:%S UTC")
        ));

        let status = if certificate.is_valid() {
            "VALID"
        } else {
            "EXPIRED"
        };
        output.push_str(&format!("Status:            {}\n", status));

        if let Some(remaining) = certificate.remaining_validity() {
            output.push_str(&format!(
                "Remaining:         {} days\n",
                remaining.num_days()
            ));
        }

        output.push_str("\n───────────────────────────────────────────────────────────────────\n");
        output.push_str("                      VALIDATION RESULTS                           \n");
        output.push_str("───────────────────────────────────────────────────────────────────\n\n");

        for (metric, value) in &certificate.validation_results {
            output.push_str(&format!("  {:<30} {:>10.4}\n", metric, value));
        }

        if !certificate.metadata.is_empty() {
            output.push_str(
                "\n───────────────────────────────────────────────────────────────────\n",
            );
            output
                .push_str("                         METADATA                                  \n");
            output.push_str(
                "───────────────────────────────────────────────────────────────────\n\n",
            );

            for (key, value) in &certificate.metadata {
                output.push_str(&format!("  {:<30} {}\n", key, value));
            }
        }

        output.push_str("\n───────────────────────────────────────────────────────────────────\n");
        output.push_str(&format!("Signature:         {}\n", certificate.signature));
        output.push_str("═══════════════════════════════════════════════════════════════════\n");

        Ok(output)
    }

    /// Export certificate as JSON
    fn export_json(&self, certificate: &ValidationCertificate) -> Result<String, CertificateError> {
        serde_json::to_string_pretty(certificate)
            .map_err(|e| CertificateError::ExportError(format!("JSON serialization failed: {}", e)))
    }

    /// Export certificate as XML
    fn export_xml(&self, certificate: &ValidationCertificate) -> Result<String, CertificateError> {
        let mut output = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        output.push_str("<ValidationCertificate>\n");
        output.push_str(&format!(
            "  <CertificateId>{}</CertificateId>\n",
            certificate.certificate_id
        ));
        output.push_str(&format!(
            "  <Type>{}</Type>\n",
            certificate.certificate_type
        ));
        output.push_str(&format!("  <Version>{}</Version>\n", certificate.version));
        output.push_str(&format!("  <Subject>{}</Subject>\n", certificate.subject));
        output.push_str(&format!("  <Issuer>{}</Issuer>\n", certificate.issuer));
        output.push_str(&format!(
            "  <IssuedAt>{}</IssuedAt>\n",
            certificate.issued_at.to_rfc3339()
        ));
        output.push_str(&format!(
            "  <ExpiresAt>{}</ExpiresAt>\n",
            certificate.expires_at.to_rfc3339()
        ));
        output.push_str(&format!("  <Valid>{}</Valid>\n", certificate.is_valid()));

        output.push_str("  <ValidationResults>\n");
        for (metric, value) in &certificate.validation_results {
            output.push_str(&format!(
                "    <Result name=\"{}\">{}</Result>\n",
                metric, value
            ));
        }
        output.push_str("  </ValidationResults>\n");

        if !certificate.metadata.is_empty() {
            output.push_str("  <Metadata>\n");
            for (key, value) in &certificate.metadata {
                output.push_str(&format!("    <Entry key=\"{}\">{}</Entry>\n", key, value));
            }
            output.push_str("  </Metadata>\n");
        }

        output.push_str(&format!(
            "  <Signature>{}</Signature>\n",
            certificate.signature
        ));
        output.push_str("</ValidationCertificate>\n");

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_certificate_generation() {
        let mut generator = CertificateGenerator::new("VoiRS Test Authority");

        let mut results = HashMap::new();
        results.insert("pesq_score".to_string(), 4.2);
        results.insert("stoi_score".to_string(), 0.95);

        let certificate = generator
            .generate_certificate(
                CertificateType::QualityValidation,
                "test-model",
                results,
                30,
            )
            .unwrap();

        assert!(certificate.certificate_id.starts_with("VOIRS-"));
        assert_eq!(
            certificate.certificate_type,
            CertificateType::QualityValidation
        );
        assert_eq!(certificate.subject, "test-model");
        assert_eq!(certificate.issuer, "VoiRS Test Authority");
        assert!(certificate.is_valid());
    }

    #[test]
    fn test_certificate_verification() {
        let mut generator = CertificateGenerator::new("VoiRS Test Authority");

        let mut results = HashMap::new();
        results.insert("quality_score".to_string(), 0.9);

        let certificate = generator
            .generate_certificate(CertificateType::QualityValidation, "test", results, 30)
            .unwrap();

        assert!(generator.verify_certificate(&certificate).is_ok());
    }

    #[test]
    fn test_certificate_revocation() {
        let mut generator = CertificateGenerator::new("VoiRS Test Authority");

        let mut results = HashMap::new();
        results.insert("score".to_string(), 0.9);

        let certificate = generator
            .generate_certificate(CertificateType::QualityValidation, "test", results, 30)
            .unwrap();

        let cert_id = certificate.certificate_id.clone();

        // Revoke certificate
        generator
            .revoke_certificate(&cert_id, "Testing revocation")
            .unwrap();

        // Verify should fail for revoked certificate
        assert!(generator.verify_certificate(&certificate).is_err());
        assert!(generator.is_revoked(&cert_id));
    }

    #[test]
    fn test_certificate_expiration() {
        let mut generator = CertificateGenerator::new("VoiRS Test Authority");

        let mut results = HashMap::new();
        results.insert("score".to_string(), 0.9);

        let mut certificate = generator
            .generate_certificate(CertificateType::QualityValidation, "test", results, 30)
            .unwrap();

        // Certificate should be valid
        assert!(certificate.is_valid());
        assert!(certificate.remaining_validity().is_some());

        // Simulate expiration by setting expires_at to past
        certificate.expires_at = Utc::now() - Duration::days(1);

        // Now certificate should be expired
        assert!(!certificate.is_valid());
        assert!(certificate.remaining_validity().is_none());
    }

    #[test]
    fn test_text_export() {
        let mut generator = CertificateGenerator::new("VoiRS Test Authority");

        let mut results = HashMap::new();
        results.insert("pesq_score".to_string(), 4.2);

        let certificate = generator
            .generate_certificate(CertificateType::QualityValidation, "test", results, 30)
            .unwrap();

        let text = generator
            .export_certificate(&certificate, CertificateFormat::Text)
            .unwrap();

        assert!(text.contains("VALIDATION CERTIFICATE"));
        assert!(text.contains("test"));
        assert!(text.contains("pesq_score"));
    }

    #[test]
    fn test_json_export() {
        let mut generator = CertificateGenerator::new("VoiRS Test Authority");

        let mut results = HashMap::new();
        results.insert("stoi_score".to_string(), 0.95);

        let certificate = generator
            .generate_certificate(CertificateType::ComplianceValidation, "test", results, 30)
            .unwrap();

        let json = generator
            .export_certificate(&certificate, CertificateFormat::Json)
            .unwrap();

        assert!(json.contains("\"certificate_id\""));
        assert!(json.contains("\"stoi_score\""));
    }

    #[test]
    fn test_xml_export() {
        let mut generator = CertificateGenerator::new("VoiRS Test Authority");

        let mut results = HashMap::new();
        results.insert("performance_score".to_string(), 0.88);

        let certificate = generator
            .generate_certificate(CertificateType::PerformanceValidation, "test", results, 30)
            .unwrap();

        let xml = generator
            .export_certificate(&certificate, CertificateFormat::Xml)
            .unwrap();

        assert!(xml.contains("<?xml"));
        assert!(xml.contains("<ValidationCertificate>"));
        assert!(xml.contains("<CertificateId>"));
        assert!(xml.contains("performance_score"));
    }

    #[test]
    fn test_minimum_threshold() {
        let mut generator = CertificateGenerator::new("VoiRS Test Authority");

        let mut low_results = HashMap::new();
        low_results.insert("quality_score".to_string(), 0.5);

        // Should fail due to low score
        let result = generator.generate_certificate(
            CertificateType::QualityValidation,
            "low-quality",
            low_results,
            30,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_list_valid_certificates() {
        let mut generator = CertificateGenerator::new("VoiRS Test Authority");

        let mut results1 = HashMap::new();
        results1.insert("score".to_string(), 0.9);

        let mut results2 = HashMap::new();
        results2.insert("score".to_string(), 0.95);

        generator
            .generate_certificate(CertificateType::QualityValidation, "model1", results1, 30)
            .unwrap();

        let cert2 = generator
            .generate_certificate(CertificateType::QualityValidation, "model2", results2, 30)
            .unwrap();

        // Revoke one certificate
        generator
            .revoke_certificate(&cert2.certificate_id, "Test")
            .unwrap();

        // Should only list the non-revoked certificate
        let valid_certs = generator.list_valid_certificates();
        assert_eq!(valid_certs.len(), 1);
    }

    #[test]
    fn test_certificate_age() {
        let mut generator = CertificateGenerator::new("VoiRS Test Authority");

        let mut results = HashMap::new();
        results.insert("score".to_string(), 0.9);

        let certificate = generator
            .generate_certificate(CertificateType::QualityValidation, "test", results, 30)
            .unwrap();

        let age = certificate.age();
        assert!(age.num_seconds() >= 0);
        assert!(age.num_seconds() < 60); // Should be less than a minute old
    }
}
