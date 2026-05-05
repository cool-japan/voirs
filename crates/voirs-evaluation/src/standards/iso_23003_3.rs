//! ISO/IEC 23003-3 USAC (Unified Speech and Audio Coding) Compliance
//!
//! This module implements validation for ISO/IEC 23003-3:2020 standard,
//! which specifies USAC codec requirements including bitrate ranges,
//! audio bandwidth, and quality constraints.
//!
//! # Standard Reference
//!
//! ISO/IEC 23003-3:2020 - "Unified speech and audio coding"
//!
//! # Compliance Checks
//!
//! - Bitrate compliance (8-384 kbps)
//! - Bandwidth support (NB, WB, SWB, FB)
//! - Delay constraints
//! - Quality metrics (PEAQ, POLQA)
//! - Coding artifacts detection

use super::StandardsError;
use crate::quality::{PESQEvaluator, PolqaBandwidth, PolqaEvaluator};
use serde::{Deserialize, Serialize};
use voirs_sdk::AudioBuffer;

/// ISO/IEC 23003-3 USAC validator
pub struct IsoUsacValidator {
    /// Sample rate
    sample_rate: u32,
    /// Bandwidth mode
    bandwidth_mode: UsacBandwidthMode,
    /// Bitrate (kbps)
    target_bitrate: u32,
}

/// USAC bandwidth modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsacBandwidthMode {
    /// Narrow-band (8 kHz)
    NarrowBand,
    /// Wide-band (16 kHz)
    WideBand,
    /// Super-wideband (24 kHz)
    SuperWideBand,
    /// Full-band (48 kHz)
    FullBand,
}

impl UsacBandwidthMode {
    /// Get sample rate for bandwidth mode
    pub fn sample_rate(&self) -> u32 {
        match self {
            Self::NarrowBand => 8000,
            Self::WideBand => 16000,
            Self::SuperWideBand => 24000,
            Self::FullBand => 48000,
        }
    }

    /// Get bandwidth in Hz
    pub fn bandwidth_hz(&self) -> u32 {
        match self {
            Self::NarrowBand => 4000,
            Self::WideBand => 8000,
            Self::SuperWideBand => 12000,
            Self::FullBand => 20000,
        }
    }
}

/// USAC compliance result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsacCompliance {
    /// Is compliant with ISO/IEC 23003-3
    pub is_compliant: bool,
    /// Compliance level
    pub compliance_level: super::ComplianceLevel,
    /// Bitrate compliance
    pub bitrate_compliant: bool,
    /// Actual bitrate (kbps)
    pub actual_bitrate: u32,
    /// Target bitrate (kbps)
    pub target_bitrate: u32,
    /// Bandwidth compliance
    pub bandwidth_compliant: bool,
    /// Delay compliance
    pub delay_compliant: bool,
    /// Measured delay (ms)
    pub delay_ms: f32,
    /// Quality score (PEAQ ODG or POLQA MOS)
    pub quality_score: f32,
    /// Coding artifacts detected
    pub artifacts_detected: Vec<String>,
    /// Validation messages
    pub validation_messages: Vec<String>,
}

impl IsoUsacValidator {
    /// Create new ISO/IEC 23003-3 validator
    pub fn new(
        bandwidth_mode: UsacBandwidthMode,
        target_bitrate: u32,
    ) -> Result<Self, StandardsError> {
        // Validate bitrate range (8-384 kbps for USAC)
        if target_bitrate < 8 || target_bitrate > 384 {
            return Err(StandardsError::ValidationFailed {
                message: format!(
                    "USAC bitrate must be between 8 and 384 kbps, got {} kbps",
                    target_bitrate
                ),
            });
        }

        let sample_rate = bandwidth_mode.sample_rate();

        Ok(Self {
            sample_rate,
            bandwidth_mode,
            target_bitrate,
        })
    }

    /// Validate USAC compliance
    pub fn validate_compliance(
        &self,
        audio: &AudioBuffer,
        reference: Option<&AudioBuffer>,
    ) -> Result<UsacCompliance, StandardsError> {
        let mut validation_messages = Vec::new();
        let mut artifacts_detected = Vec::new();

        // 1. Validate bitrate compliance
        let actual_bitrate = self.estimate_bitrate(audio)?;
        let bitrate_compliant = self.check_bitrate_compliance(actual_bitrate);

        if !bitrate_compliant {
            validation_messages.push(format!(
                "Bitrate {} kbps outside acceptable range for target {} kbps",
                actual_bitrate, self.target_bitrate
            ));
        }

        // 2. Validate bandwidth compliance
        let bandwidth_compliant = self.check_bandwidth_compliance(audio)?;

        if !bandwidth_compliant {
            validation_messages.push(format!(
                "Audio bandwidth does not match {:?} specification",
                self.bandwidth_mode
            ));
        }

        // 3. Validate delay constraints
        let delay_ms = self.estimate_delay(audio)?;
        let delay_compliant = self.check_delay_compliance(delay_ms);

        if !delay_compliant {
            validation_messages.push(format!(
                "Delay {} ms exceeds maximum allowed for USAC",
                delay_ms
            ));
        }

        // 4. Estimate quality score
        let quality_score = if let Some(ref_audio) = reference {
            self.estimate_quality_score(audio, ref_audio)?
        } else {
            // Use no-reference quality estimation
            self.estimate_nr_quality_score(audio)?
        };

        // 5. Detect coding artifacts
        artifacts_detected.extend(self.detect_artifacts(audio)?);

        // Determine overall compliance
        let is_compliant = bitrate_compliant
            && bandwidth_compliant
            && delay_compliant
            && quality_score >= 3.5
            && artifacts_detected.is_empty();

        let compliance_level = if is_compliant {
            super::ComplianceLevel::FullyCompliant
        } else if bitrate_compliant && bandwidth_compliant {
            super::ComplianceLevel::PartiallyCompliant
        } else {
            super::ComplianceLevel::NotCompliant
        };

        Ok(UsacCompliance {
            is_compliant,
            compliance_level,
            bitrate_compliant,
            actual_bitrate,
            target_bitrate: self.target_bitrate,
            bandwidth_compliant,
            delay_compliant,
            delay_ms,
            quality_score,
            artifacts_detected,
            validation_messages,
        })
    }

    /// Estimate bitrate from audio
    fn estimate_bitrate(&self, audio: &AudioBuffer) -> Result<u32, StandardsError> {
        // Estimate bitrate based on audio characteristics
        // This is a simplified estimation; real bitrate would come from codec metadata

        let samples = audio.samples();
        let duration_sec = samples.len() as f32 / self.sample_rate as f32;

        // Estimate entropy/complexity
        let mut entropy = 0.0f32;
        for window in samples.windows(2) {
            let diff = (window[1] - window[0]).abs();
            entropy += diff;
        }
        entropy /= samples.len() as f32;

        // Map entropy to bitrate estimate (heuristic)
        let estimated_bitrate = (entropy * 200.0 + 32.0).clamp(8.0, 384.0) as u32;

        Ok(estimated_bitrate)
    }

    /// Check bitrate compliance
    fn check_bitrate_compliance(&self, actual_bitrate: u32) -> bool {
        // Allow ±10% tolerance
        let tolerance = (self.target_bitrate as f32 * 0.1) as u32;
        let lower = self.target_bitrate.saturating_sub(tolerance);
        let upper = self.target_bitrate + tolerance;

        actual_bitrate >= lower && actual_bitrate <= upper
    }

    /// Check bandwidth compliance
    fn check_bandwidth_compliance(&self, audio: &AudioBuffer) -> Result<bool, StandardsError> {
        use scirs2_fft::{RealFftPlanner, RealToComplex};

        let samples = audio.samples();
        if samples.is_empty() {
            return Ok(false);
        }

        // Use FFT to check frequency content
        let fft_size = 2048;
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_size);

        let mut buffer: Vec<f32> = samples.iter().take(fft_size).copied().collect();
        buffer.resize(fft_size, 0.0);

        let mut spectrum = vec![scirs2_core::Complex::new(0.0, 0.0); fft_size / 2 + 1];
        fft.process(&mut buffer, &mut spectrum).map_err(|e| StandardsError::Other(e.to_string()))?;

        // Check energy distribution up to expected bandwidth
        let max_freq = self.bandwidth_mode.bandwidth_hz();
        let bin_resolution = self.sample_rate as f32 / fft_size as f32;
        let max_bin = (max_freq as f32 / bin_resolution) as usize;

        // Calculate energy in-band vs out-of-band
        let in_band_energy: f32 = spectrum
            .iter()
            .take(max_bin.min(spectrum.len()))
            .map(|c| c.norm_sqr())
            .sum();
        let total_energy: f32 = spectrum.iter().map(|c| c.norm_sqr()).sum();

        // At least 95% of energy should be in-band
        let in_band_ratio = in_band_energy / total_energy.max(1e-10);
        Ok(in_band_ratio >= 0.95)
    }

    /// Estimate codec delay
    fn estimate_delay(&self, _audio: &AudioBuffer) -> Result<f32, StandardsError> {
        // USAC codec delay depends on configuration
        // Typical values: 20-80 ms
        // This is a placeholder - real delay would be measured or from metadata

        let frame_size_ms = match self.bandwidth_mode {
            UsacBandwidthMode::NarrowBand => 20.0,
            UsacBandwidthMode::WideBand => 20.0,
            UsacBandwidthMode::SuperWideBand => 40.0,
            UsacBandwidthMode::FullBand => 40.0,
        };

        // Add algorithmic delay (lookahead + processing)
        let algorithmic_delay_ms = 20.0;

        Ok(frame_size_ms + algorithmic_delay_ms)
    }

    /// Check delay compliance
    fn check_delay_compliance(&self, delay_ms: f32) -> bool {
        // ISO/IEC 23003-3 recommends delay < 100 ms for interactive applications
        delay_ms < 100.0
    }

    /// Estimate quality score using POLQA (ITU-T P.863) or PESQ (ITU-T P.862)
    ///
    /// This method implements full-reference quality assessment using the appropriate
    /// ITU-T standard based on the bandwidth mode:
    /// - NarrowBand/WideBand: PESQ (P.862) → converted to MOS scale
    /// - SuperWideBand/FullBand: POLQA (P.863) for better accuracy
    fn estimate_quality_score(
        &self,
        degraded: &AudioBuffer,
        reference: &AudioBuffer,
    ) -> Result<f32, StandardsError> {
        // Select appropriate quality metric based on bandwidth
        let quality_score = match self.bandwidth_mode {
            UsacBandwidthMode::NarrowBand | UsacBandwidthMode::WideBand => {
                // Use PESQ for narrow-band and wide-band
                let pesq_evaluator = if self.bandwidth_mode == UsacBandwidthMode::NarrowBand {
                    PESQEvaluator::new_narrowband()
                } else {
                    PESQEvaluator::new_wideband()
                }
                .map_err(|e| StandardsError::ValidationFailed {
                    message: format!("PESQ init failed: {}", e),
                })?;

                // Calculate PESQ score (range: -0.5 to 4.5, typically 1.0-4.5)
                let pesq_score = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async {
                        pesq_evaluator
                            .calculate_pesq(reference, degraded)
                            .await
                            .map_err(|e| StandardsError::ValidationFailed {
                                message: format!("PESQ calculation failed: {}", e),
                            })
                    })
                })?;

                // Convert PESQ to MOS scale (1-5)
                // PESQ ranges from -0.5 to 4.5, we map to 1.0-5.0
                // Formula: MOS = 0.999 + (4.000 / (1 + exp(-1.4945 * PESQ + 4.6607)))
                // Simplified linear mapping for conservative estimate
                let mos = ((pesq_score + 0.5) / 5.0) * 4.0 + 1.0;
                mos.clamp(1.0, 5.0)
            }
            UsacBandwidthMode::SuperWideBand | UsacBandwidthMode::FullBand => {
                // Use POLQA for super-wideband and full-band for better accuracy
                let polqa_bandwidth = if self.bandwidth_mode == UsacBandwidthMode::SuperWideBand {
                    PolqaBandwidth::SuperWideBand
                } else {
                    PolqaBandwidth::FullBand
                };

                let polqa_evaluator = PolqaEvaluator::new(polqa_bandwidth).map_err(|e| {
                    StandardsError::ValidationFailed {
                        message: format!("POLQA init failed: {}", e),
                    }
                })?;

                // Calculate POLQA score (MOS scale 1-5)
                let polqa_score = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async {
                        polqa_evaluator
                            .calculate_polqa(reference, degraded)
                            .await
                            .map_err(|e| StandardsError::ValidationFailed {
                                message: format!("POLQA calculation failed: {}", e),
                            })
                    })
                })?;

                polqa_score.clamp(1.0, 5.0)
            }
        };

        Ok(quality_score)
    }

    /// Estimate no-reference quality score
    fn estimate_nr_quality_score(&self, audio: &AudioBuffer) -> Result<f32, StandardsError> {
        // No-reference quality estimation based on signal characteristics
        let samples = audio.samples();

        // Calculate simple quality indicators
        let mut quality_score: f32 = 5.0;

        // 1. Check for clipping
        let clipping_ratio =
            samples.iter().filter(|&&s| s.abs() > 0.99).count() as f32 / samples.len() as f32;
        if clipping_ratio > 0.01 {
            quality_score -= 1.0;
        }

        // 2. Check for silence/very low levels
        let avg_level: f32 = samples.iter().map(|&s| s.abs()).sum::<f32>() / samples.len() as f32;
        if avg_level < 0.01 {
            quality_score -= 0.5;
        }

        // 3. Check for DC offset
        let dc_offset: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        if dc_offset.abs() > 0.1 {
            quality_score -= 0.3;
        }

        Ok(quality_score.max(1.0))
    }

    /// Detect coding artifacts
    fn detect_artifacts(&self, audio: &AudioBuffer) -> Result<Vec<String>, StandardsError> {
        let mut artifacts = Vec::new();
        let samples = audio.samples();

        // 1. Check for pre-echo artifacts
        if self.detect_pre_echo(samples) {
            artifacts.push("Pre-echo detected".to_string());
        }

        // 2. Check for birdies/tones
        if self.detect_tonal_artifacts(samples)? {
            artifacts.push("Tonal artifacts detected".to_string());
        }

        // 3. Check for bandwidth limitation artifacts
        if self.detect_bandwidth_artifacts(samples)? {
            artifacts.push("Bandwidth limitation artifacts detected".to_string());
        }

        Ok(artifacts)
    }

    /// Detect pre-echo artifacts
    fn detect_pre_echo(&self, samples: &[f32]) -> bool {
        // Simplified pre-echo detection
        // Look for sudden increases in energy before attack transients

        let frame_size = 256;
        for i in 1..samples.len() / frame_size {
            let prev_energy: f32 = samples[(i - 1) * frame_size..i * frame_size]
                .iter()
                .map(|&s| s * s)
                .sum();
            let curr_energy: f32 = samples[i * frame_size..(i + 1) * frame_size]
                .iter()
                .map(|&s| s * s)
                .sum();

            // If energy increases by more than 20 dB, check for pre-echo
            if curr_energy > prev_energy * 100.0 && prev_energy > 1e-6 {
                return true;
            }
        }

        false
    }

    /// Detect tonal artifacts
    fn detect_tonal_artifacts(&self, _samples: &[f32]) -> Result<bool, StandardsError> {
        // Placeholder for spectral analysis to detect coding-induced tones
        // Would use FFT and peak detection
        Ok(false)
    }

    /// Detect bandwidth limitation artifacts
    fn detect_bandwidth_artifacts(&self, _samples: &[f32]) -> Result<bool, StandardsError> {
        // Placeholder for detecting artifacts near bandwidth edges
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usac_validator_creation() {
        let validator = IsoUsacValidator::new(UsacBandwidthMode::WideBand, 64);
        assert!(validator.is_ok());
    }

    #[test]
    fn test_invalid_bitrate() {
        let validator = IsoUsacValidator::new(UsacBandwidthMode::WideBand, 500);
        assert!(validator.is_err());
    }

    #[test]
    fn test_bandwidth_modes() {
        assert_eq!(UsacBandwidthMode::NarrowBand.sample_rate(), 8000);
        assert_eq!(UsacBandwidthMode::WideBand.sample_rate(), 16000);
        assert_eq!(UsacBandwidthMode::SuperWideBand.sample_rate(), 24000);
        assert_eq!(UsacBandwidthMode::FullBand.sample_rate(), 48000);
    }

    #[test]
    fn test_bitrate_compliance() {
        let validator = IsoUsacValidator::new(UsacBandwidthMode::WideBand, 64).unwrap();

        assert!(validator.check_bitrate_compliance(64)); // Exact match
        assert!(validator.check_bitrate_compliance(60)); // Within tolerance
        assert!(validator.check_bitrate_compliance(70)); // Within tolerance
        assert!(!validator.check_bitrate_compliance(50)); // Outside tolerance
        assert!(!validator.check_bitrate_compliance(80)); // Outside tolerance
    }

    #[test]
    fn test_delay_compliance() {
        let validator = IsoUsacValidator::new(UsacBandwidthMode::WideBand, 64).unwrap();

        assert!(validator.check_delay_compliance(50.0));
        assert!(validator.check_delay_compliance(99.0));
        assert!(!validator.check_delay_compliance(100.0));
        assert!(!validator.check_delay_compliance(150.0));
    }

    #[test]
    fn test_compliance_validation() {
        let validator = IsoUsacValidator::new(UsacBandwidthMode::WideBand, 64).unwrap();

        let audio = AudioBuffer::new(vec![0.1; 16000], 16000, 1);

        let result = validator.validate_compliance(&audio, None);
        assert!(result.is_ok());

        let compliance = result.unwrap();
        assert!(compliance.quality_score >= 1.0 && compliance.quality_score <= 5.0);
    }
}
