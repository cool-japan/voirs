//! Industry Standards Compliance Module
//!
//! This module provides comprehensive support for industry standards compliance
//! including ITU-T, ANSI, ISO/IEC, and AES standards for audio quality evaluation.
//!
//! # Supported Standards
//!
//! - **ITU-T P.862**: PESQ (Perceptual Evaluation of Speech Quality)
//! - **ITU-T P.863**: POLQA (Perceptual Objective Listening Quality Assessment)
//! - **ITU-T P.56**: Speech level measurement for voice applications
//! - **ANSI S3.5**: Methods for calculation of the Speech Intelligibility Index
//! - **ISO/IEC 23003-3**: USAC (Unified Speech and Audio Coding)
//! - **AES**: Audio Engineering Society recommended practices
//!
//! # Example
//!
//! ```no_run
//! use voirs_evaluation::standards::*;
//! use voirs_sdk::AudioBuffer;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create standards compliance validator
//! let validator = StandardsValidator::new()?;
//! let audio_data = AudioBuffer::new(vec![0.0f32; 16000], 16000, 1);
//!
//! // Validate ANSI S3.5 compliance
//! let ansi_result = validator.validate_ansi_s3_5(&audio_data, None)?;
//! println!("ANSI S3.5 SII: {:?}", ansi_result.sii);
//!
//! // Validate ISO/IEC 23003-3 compliance
//! let iso_result = validator.validate_iso_23003_3(&audio_data, None)?;
//! println!("ISO/IEC compliance: {}", iso_result.is_compliant);
//! # Ok(())
//! # }
//! ```

pub mod aes_standards;
pub mod ansi_s3_5;
pub mod iso_23003_3;
pub mod itu_t_compliance;
pub mod validator;

pub use aes_standards::{AesRecommendedPractice, AesStandards};
pub use ansi_s3_5::{AnsiS35Evaluator, SpeechIntelligibilityIndex};
pub use iso_23003_3::{IsoUsacValidator, UsacCompliance};
pub use itu_t_compliance::{
    BadgeFormat, CertificationLevel, ComplianceCertificationReport, ComplianceTestResult,
    ItuTComplianceValidator, ItuTStandard, StandardComplianceResult,
};
pub use validator::{ComplianceReport, StandardsValidator};

use thiserror::Error;

/// Standards compliance errors
#[derive(Error, Debug)]
pub enum StandardsError {
    /// Standard not supported
    #[error("Standard not supported: {standard}")]
    UnsupportedStandard {
        /// Standard identifier
        standard: String,
    },

    /// Compliance validation failed
    #[error("Compliance validation failed: {message}")]
    ValidationFailed {
        /// Error message
        message: String,
    },

    /// Invalid audio data
    #[error("Invalid audio data: {message}")]
    InvalidAudioData {
        /// Error message
        message: String,
    },

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Other error
    #[error("Standards error: {0}")]
    Other(String),
}

/// Standard compliance level
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ComplianceLevel {
    /// Fully compliant with standard
    FullyCompliant,
    /// Partially compliant (meets minimum requirements)
    PartiallyCompliant,
    /// Not compliant
    NotCompliant,
}

impl ComplianceLevel {
    /// Check if compliance level is acceptable
    pub fn is_acceptable(&self) -> bool {
        matches!(self, Self::FullyCompliant | Self::PartiallyCompliant)
    }
}
