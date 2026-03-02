//! Standards Compliance Validator
//!
//! This module provides a unified interface for validating audio against
//! multiple industry standards (ITU-T, ANSI, ISO/IEC, AES).

use super::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use voirs_sdk::AudioBuffer;

/// Standards compliance validator
pub struct StandardsValidator {
    /// Sample rate
    sample_rate: u32,
    /// Enabled standards
    enabled_standards: Vec<String>,
}

/// Complete compliance report covering all standards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Overall compliance status
    pub overall_compliant: bool,
    /// Overall compliance level
    pub overall_compliance_level: ComplianceLevel,
    /// ANSI S3.5 result
    pub ansi_s3_5: Option<ansi_s3_5::SpeechIntelligibilityIndex>,
    /// ISO/IEC 23003-3 result
    pub iso_23003_3: Option<iso_23003_3::UsacCompliance>,
    /// AES17 measurements
    pub aes17: Option<aes_standards::Aes17Measurements>,
    /// AES49 loudness
    pub aes49: Option<aes_standards::Aes49Loudness>,
    /// Summary of findings
    pub summary: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

impl StandardsValidator {
    /// Create new standards validator
    pub fn new() -> Result<Self, StandardsError> {
        Self::with_sample_rate(48000)
    }

    /// Create validator with specific sample rate
    pub fn with_sample_rate(sample_rate: u32) -> Result<Self, StandardsError> {
        if sample_rate < 8000 {
            return Err(StandardsError::InvalidAudioData {
                message: "Sample rate must be at least 8000 Hz".to_string(),
            });
        }

        Ok(Self {
            sample_rate,
            enabled_standards: vec![
                "ANSI_S3_5".to_string(),
                "ISO_23003_3".to_string(),
                "AES17".to_string(),
                "AES49".to_string(),
            ],
        })
    }

    /// Enable specific standard
    pub fn enable_standard(&mut self, standard: &str) {
        if !self.enabled_standards.contains(&standard.to_string()) {
            self.enabled_standards.push(standard.to_string());
        }
    }

    /// Disable specific standard
    pub fn disable_standard(&mut self, standard: &str) {
        self.enabled_standards.retain(|s| s != standard);
    }

    /// Validate compliance against all enabled standards
    pub fn validate_all(
        &self,
        audio: &AudioBuffer,
        reference: Option<&AudioBuffer>,
    ) -> Result<ComplianceReport, StandardsError> {
        let mut summary = Vec::new();
        let mut recommendations = Vec::new();

        // Validate ANSI S3.5
        let ansi_s3_5 = if self.enabled_standards.contains(&"ANSI_S3_5".to_string()) {
            match self.validate_ansi_s3_5(audio, reference) {
                Ok(result) => {
                    summary.push(format!(
                        "ANSI S3.5 SII: {:.3} ({:?})",
                        result.sii, result.compliance_level
                    ));
                    if result.sii < 0.75 {
                        recommendations.push(
                            "Improve speech intelligibility by reducing noise or enhancing signal"
                                .to_string(),
                        );
                    }
                    Some(result)
                }
                Err(e) => {
                    summary.push(format!("ANSI S3.5 validation failed: {}", e));
                    None
                }
            }
        } else {
            None
        };

        // Validate ISO/IEC 23003-3
        let iso_23003_3 = if self.enabled_standards.contains(&"ISO_23003_3".to_string()) {
            match self.validate_iso_23003_3(audio, reference) {
                Ok(result) => {
                    summary.push(format!(
                        "ISO/IEC 23003-3: {} ({:?})",
                        if result.is_compliant {
                            "Compliant"
                        } else {
                            "Non-compliant"
                        },
                        result.compliance_level
                    ));
                    if !result.is_compliant {
                        recommendations.extend(result.validation_messages.clone());
                    }
                    Some(result)
                }
                Err(e) => {
                    summary.push(format!("ISO/IEC 23003-3 validation failed: {}", e));
                    None
                }
            }
        } else {
            None
        };

        // Validate AES17
        let aes17 = if self.enabled_standards.contains(&"AES17".to_string()) {
            match self.validate_aes17(audio) {
                Ok(result) => {
                    summary.push(format!(
                        "AES17: DR={:.1}dB, THD+N={:.3}%, SNR={:.1}dB ({:?})",
                        result.dynamic_range_db,
                        result.thd_n_percent,
                        result.snr_db,
                        result.compliance_level
                    ));
                    if !result.notes.is_empty() {
                        recommendations.extend(result.notes.clone());
                    }
                    Some(result)
                }
                Err(e) => {
                    summary.push(format!("AES17 validation failed: {}", e));
                    None
                }
            }
        } else {
            None
        };

        // Validate AES49
        let aes49 = if self.enabled_standards.contains(&"AES49".to_string()) {
            match self.validate_aes49_loudness(audio) {
                Ok(result) => {
                    summary.push(format!(
                        "AES49 Loudness: {:.1} LUFS, Range={:.1} LU ({:?})",
                        result.integrated_loudness_lufs,
                        result.loudness_range_lu,
                        result.compliance_level
                    ));
                    if !result.ebu_r128_compliant {
                        recommendations.push(
                            "Adjust loudness to meet EBU R128 standard (-23 LUFS ±1 LU)"
                                .to_string(),
                        );
                    }
                    Some(result)
                }
                Err(e) => {
                    summary.push(format!("AES49 validation failed: {}", e));
                    None
                }
            }
        } else {
            None
        };

        // Determine overall compliance
        let mut compliant_count = 0;
        let mut total_count = 0;

        if let Some(ref result) = ansi_s3_5 {
            total_count += 1;
            if result.compliance_level.is_acceptable() {
                compliant_count += 1;
            }
        }

        if let Some(ref result) = iso_23003_3 {
            total_count += 1;
            if result.compliance_level.is_acceptable() {
                compliant_count += 1;
            }
        }

        if let Some(ref result) = aes17 {
            total_count += 1;
            if result.compliance_level.is_acceptable() {
                compliant_count += 1;
            }
        }

        if let Some(ref result) = aes49 {
            total_count += 1;
            if result.compliance_level.is_acceptable() {
                compliant_count += 1;
            }
        }

        let overall_compliant = total_count > 0 && compliant_count == total_count;
        let overall_compliance_level = if compliant_count == total_count {
            ComplianceLevel::FullyCompliant
        } else if compliant_count > 0 {
            ComplianceLevel::PartiallyCompliant
        } else {
            ComplianceLevel::NotCompliant
        };

        Ok(ComplianceReport {
            overall_compliant,
            overall_compliance_level,
            ansi_s3_5,
            iso_23003_3,
            aes17,
            aes49,
            summary,
            recommendations,
        })
    }

    /// Validate ANSI S3.5 compliance
    pub fn validate_ansi_s3_5(
        &self,
        speech: &AudioBuffer,
        noise: Option<&AudioBuffer>,
    ) -> Result<ansi_s3_5::SpeechIntelligibilityIndex, StandardsError> {
        let evaluator = ansi_s3_5::AnsiS35Evaluator::new(
            self.sample_rate,
            ansi_s3_5::BandConfiguration::OneThirdOctave,
        )?;

        evaluator.calculate_sii(speech, noise)
    }

    /// Validate ISO/IEC 23003-3 compliance
    pub fn validate_iso_23003_3(
        &self,
        audio: &AudioBuffer,
        reference: Option<&AudioBuffer>,
    ) -> Result<iso_23003_3::UsacCompliance, StandardsError> {
        // Determine bandwidth mode based on sample rate
        let bandwidth_mode = match self.sample_rate {
            8000 => iso_23003_3::UsacBandwidthMode::NarrowBand,
            16000 => iso_23003_3::UsacBandwidthMode::WideBand,
            24000 => iso_23003_3::UsacBandwidthMode::SuperWideBand,
            _ => iso_23003_3::UsacBandwidthMode::FullBand,
        };

        // Assume 64 kbps target bitrate for wideband
        let target_bitrate = match bandwidth_mode {
            iso_23003_3::UsacBandwidthMode::NarrowBand => 24,
            iso_23003_3::UsacBandwidthMode::WideBand => 64,
            iso_23003_3::UsacBandwidthMode::SuperWideBand => 96,
            iso_23003_3::UsacBandwidthMode::FullBand => 128,
        };

        let validator = iso_23003_3::IsoUsacValidator::new(bandwidth_mode, target_bitrate)?;

        validator.validate_compliance(audio, reference)
    }

    /// Validate AES17 compliance
    pub fn validate_aes17(
        &self,
        audio: &AudioBuffer,
    ) -> Result<aes_standards::Aes17Measurements, StandardsError> {
        let aes = aes_standards::AesStandards::new(self.sample_rate)?;
        aes.measure_aes17(audio)
    }

    /// Validate AES49 loudness compliance
    pub fn validate_aes49_loudness(
        &self,
        audio: &AudioBuffer,
    ) -> Result<aes_standards::Aes49Loudness, StandardsError> {
        let aes = aes_standards::AesStandards::new(self.sample_rate)?;
        aes.measure_aes49_loudness(audio)
    }
}

impl Default for StandardsValidator {
    fn default() -> Self {
        Self::new().expect("value should be present")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator = StandardsValidator::new();
        assert!(validator.is_ok());
    }

    #[test]
    fn test_enable_disable_standards() {
        let mut validator = StandardsValidator::new().unwrap();

        validator.disable_standard("AES17");
        assert!(!validator.enabled_standards.contains(&"AES17".to_string()));

        validator.enable_standard("AES17");
        assert!(validator.enabled_standards.contains(&"AES17".to_string()));
    }

    #[test]
    fn test_validate_all() {
        let validator = StandardsValidator::with_sample_rate(48000).unwrap();
        let audio = AudioBuffer::new(vec![0.1; 48000], 48000, 1);

        let result = validator.validate_all(&audio, None);
        assert!(result.is_ok());

        let report = result.unwrap();
        assert!(!report.summary.is_empty());
    }

    #[test]
    fn test_compliance_report_fields() {
        let validator = StandardsValidator::with_sample_rate(48000).unwrap();
        let audio = AudioBuffer::new(vec![0.1; 48000], 48000, 1);

        let report = validator.validate_all(&audio, None).unwrap();

        // Should have at least some standards validated
        let validated_count = [
            report.ansi_s3_5.is_some(),
            report.iso_23003_3.is_some(),
            report.aes17.is_some(),
            report.aes49.is_some(),
        ]
        .iter()
        .filter(|&&x| x)
        .count();

        assert!(validated_count > 0);
        assert!(matches!(
            report.overall_compliance_level,
            ComplianceLevel::FullyCompliant
                | ComplianceLevel::PartiallyCompliant
                | ComplianceLevel::NotCompliant
        ));
    }
}
