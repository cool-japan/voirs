// Copyright (c) 2024 VoiRS Contributors
// Licensed under MIT OR Apache-2.0

//! Security audit and compliance framework
//!
//! This module provides comprehensive security auditing and compliance features
//! including SOC 2, ISO 27001, GDPR compliance, audit trails, and vulnerability management.

pub mod audit_trail;
pub mod compliance;
pub mod vulnerability;

use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

/// Security compliance standards
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplianceStandard {
    /// SOC 2 Type II
    Soc2,
    /// ISO 27001
    Iso27001,
    /// GDPR (General Data Protection Regulation)
    Gdpr,
    /// HIPAA (Health Insurance Portability and Accountability Act)
    Hipaa,
    /// PCI DSS (Payment Card Industry Data Security Standard)
    PciDss,
}

/// Security audit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAuditConfig {
    /// Enable security auditing
    pub enabled: bool,
    /// Compliance standards to enforce
    pub compliance_standards: Vec<ComplianceStandard>,
    /// Audit log retention period
    pub audit_log_retention: Duration,
    /// Enable vulnerability scanning
    pub vulnerability_scanning: bool,
    /// Scanning interval
    pub scanning_interval: Duration,
    /// Enable automated remediation
    pub automated_remediation: bool,
}

impl Default for SecurityAuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            compliance_standards: vec![ComplianceStandard::Soc2, ComplianceStandard::Iso27001],
            audit_log_retention: Duration::from_secs(7_776_000), // 90 days
            vulnerability_scanning: true,
            scanning_interval: Duration::from_secs(86400), // Daily
            automated_remediation: false,
        }
    }
}

/// Security audit error
#[derive(Debug, Error)]
pub enum SecurityAuditError {
    /// Compliance violation
    #[error("Compliance violation: {standard:?} - {details}")]
    ComplianceViolation {
        /// Compliance standard that was violated
        standard: ComplianceStandard,
        /// Detailed description of the violation
        details: String,
    },
    /// Audit logging failed
    #[error("Audit logging failed: {0}")]
    AuditLoggingFailed(String),
    /// Vulnerability detected
    #[error("Vulnerability detected: severity {severity}, description: {description}")]
    VulnerabilityDetected {
        /// Severity level of the vulnerability
        severity: VulnerabilitySeverity,
        /// Description of the vulnerability
        description: String,
    },
    /// Scan failed
    #[error("Security scan failed: {0}")]
    ScanFailed(String),
}

/// Security audit result type
pub type Result<T> = std::result::Result<T, SecurityAuditError>;

/// Vulnerability severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum VulnerabilitySeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

impl std::fmt::Display for VulnerabilitySeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "Low"),
            Self::Medium => write!(f, "Medium"),
            Self::High => write!(f, "High"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

/// Security audit metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAuditMetrics {
    /// Total audit events logged
    pub total_audit_events: u64,
    /// Compliance violations detected
    pub compliance_violations: u64,
    /// Vulnerabilities detected
    pub vulnerabilities_detected: u64,
    /// Critical vulnerabilities
    pub critical_vulnerabilities: u64,
    /// High severity vulnerabilities
    pub high_vulnerabilities: u64,
    /// Medium severity vulnerabilities
    pub medium_vulnerabilities: u64,
    /// Low severity vulnerabilities
    pub low_vulnerabilities: u64,
    /// Remediated vulnerabilities
    pub remediated_vulnerabilities: u64,
    /// Last scan time
    #[serde(skip, default)]
    pub last_scan_time: Option<std::time::Instant>,
    /// Compliance score (0-100)
    pub compliance_score: f32,
}

impl Default for SecurityAuditMetrics {
    fn default() -> Self {
        Self {
            total_audit_events: 0,
            compliance_violations: 0,
            vulnerabilities_detected: 0,
            critical_vulnerabilities: 0,
            high_vulnerabilities: 0,
            medium_vulnerabilities: 0,
            low_vulnerabilities: 0,
            remediated_vulnerabilities: 0,
            last_scan_time: None,
            compliance_score: 100.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_audit_config_default() {
        let config = SecurityAuditConfig::default();
        assert!(config.enabled);
        assert!(config.vulnerability_scanning);
        assert_eq!(config.compliance_standards.len(), 2);
    }

    #[test]
    fn test_compliance_standards() {
        assert_eq!(ComplianceStandard::Soc2, ComplianceStandard::Soc2);
        assert_ne!(ComplianceStandard::Soc2, ComplianceStandard::Iso27001);
    }

    #[test]
    fn test_vulnerability_severity_ordering() {
        assert!(VulnerabilitySeverity::Critical > VulnerabilitySeverity::High);
        assert!(VulnerabilitySeverity::High > VulnerabilitySeverity::Medium);
        assert!(VulnerabilitySeverity::Medium > VulnerabilitySeverity::Low);
    }

    #[test]
    fn test_security_metrics_default() {
        let metrics = SecurityAuditMetrics::default();
        assert_eq!(metrics.total_audit_events, 0);
        assert_eq!(metrics.vulnerabilities_detected, 0);
        assert!((metrics.compliance_score - 100.0).abs() < f32::EPSILON);
    }
}
