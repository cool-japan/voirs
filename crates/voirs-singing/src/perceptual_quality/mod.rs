//! Perceptual Quality Testing Framework
//!
//! This module has been refactored into smaller, more manageable components
//! for better maintainability and organization. It provides comprehensive
//! perceptual quality evaluation for singing synthesis including naturalness
//! testing, musical expression validation, voice quality assessment, and
//! performance quality evaluation.

pub mod core;
pub mod naturalness;
pub mod reports;

// Re-export main types and traits
pub use core::{
    ExpressionValidator, PerceptualQualityTester, PerformanceEvaluator, VoiceQualityAssessor,
};

pub use naturalness::NaturalnessTester;

pub use reports::{
    ComprehensiveQualityReport, ExpressionReport, NaturalnessProfile, NaturalnessReport,
    PerformanceReport, QualityMetrics, VoiceQualityProfile, VoiceQualityReport,
};

/// Create a default perceptual quality tester
pub fn create_default_tester() -> PerceptualQualityTester {
    PerceptualQualityTester::default()
}

/// Quality assessment thresholds
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    /// Minimum overall quality score
    pub min_overall: f64,
    /// Minimum naturalness score
    pub min_naturalness: f64,
    /// Minimum expression score
    pub min_expression: f64,
    /// Minimum voice quality score
    pub min_voice_quality: f64,
    /// Minimum performance score
    pub min_performance: f64,
}

impl QualityThresholds {
    /// Conservative quality thresholds for production use
    pub fn conservative() -> Self {
        Self {
            min_overall: 0.8,
            min_naturalness: 0.75,
            min_expression: 0.7,
            min_voice_quality: 0.8,
            min_performance: 0.85,
        }
    }

    /// Balanced quality thresholds for general use
    pub fn balanced() -> Self {
        Self {
            min_overall: 0.7,
            min_naturalness: 0.65,
            min_expression: 0.6,
            min_voice_quality: 0.7,
            min_performance: 0.75,
        }
    }

    /// Relaxed quality thresholds for development/testing
    pub fn relaxed() -> Self {
        Self {
            min_overall: 0.5,
            min_naturalness: 0.5,
            min_expression: 0.4,
            min_voice_quality: 0.5,
            min_performance: 0.6,
        }
    }

    /// Check if a report meets these thresholds
    pub fn check_report(&self, report: &ComprehensiveQualityReport) -> bool {
        report.overall_score >= self.min_overall
            && report.naturalness.overall_score >= self.min_naturalness
            && report.expression.overall_score >= self.min_expression
            && report.voice_quality.overall_score >= self.min_voice_quality
            && report.performance.overall_score >= self.min_performance
    }
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self::balanced()
    }
}

/// Quality evaluation configuration
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// Enable naturalness testing
    pub enable_naturalness: bool,
    /// Enable expression validation
    pub enable_expression: bool,
    /// Enable voice quality assessment
    pub enable_voice_quality: bool,
    /// Enable performance evaluation
    pub enable_performance: bool,
    /// Quality thresholds
    pub thresholds: QualityThresholds,
}

impl EvaluationConfig {
    /// Full evaluation configuration
    pub fn full() -> Self {
        Self {
            enable_naturalness: true,
            enable_expression: true,
            enable_voice_quality: true,
            enable_performance: true,
            thresholds: QualityThresholds::balanced(),
        }
    }

    /// Fast evaluation configuration (performance only)
    pub fn fast() -> Self {
        Self {
            enable_naturalness: false,
            enable_expression: false,
            enable_voice_quality: false,
            enable_performance: true,
            thresholds: QualityThresholds::relaxed(),
        }
    }

    /// Quality-focused evaluation (no performance)
    pub fn quality_focused() -> Self {
        Self {
            enable_naturalness: true,
            enable_expression: true,
            enable_voice_quality: true,
            enable_performance: false,
            thresholds: QualityThresholds::conservative(),
        }
    }
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self::full()
    }
}

/// Quality assessment utilities
pub mod utils {

    /// Validate audio samples for quality assessment
    pub fn validate_audio_samples(audio_samples: &[f32], sample_rate: u32) -> crate::Result<()> {
        if audio_samples.is_empty() {
            return Err(crate::Error::Validation(
                "Audio samples cannot be empty".to_string(),
            ));
        }

        if sample_rate < 8000 || sample_rate > 192000 {
            return Err(crate::Error::Validation(format!(
                "Invalid sample rate: {} Hz",
                sample_rate
            )));
        }

        // Check for reasonable audio length (at least 100ms)
        let min_samples = sample_rate as usize / 10;
        if audio_samples.len() < min_samples {
            return Err(crate::Error::Validation(
                "Audio too short for quality assessment".to_string(),
            ));
        }

        Ok(())
    }

    /// Calculate signal-to-noise ratio
    pub fn calculate_snr(audio_samples: &[f32]) -> f64 {
        let signal_power: f64 = audio_samples
            .iter()
            .map(|&x| (x as f64).powi(2))
            .sum::<f64>()
            / audio_samples.len() as f64;

        let noise_estimate = estimate_noise_floor(audio_samples);

        if noise_estimate > 0.0 {
            10.0 * (signal_power / noise_estimate).log10()
        } else {
            60.0 // Very high SNR if no noise detected
        }
    }

    /// Estimate noise floor from audio samples
    fn estimate_noise_floor(audio_samples: &[f32]) -> f64 {
        // Simple noise floor estimation using minimum energy frames
        let frame_size = 1024;
        let mut frame_energies = Vec::new();

        for chunk in audio_samples.chunks(frame_size) {
            let energy: f64 =
                chunk.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / chunk.len() as f64;
            frame_energies.push(energy);
        }

        if frame_energies.is_empty() {
            return 0.0;
        }

        frame_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Use 10th percentile as noise floor estimate
        let percentile_index = (frame_energies.len() as f64 * 0.1) as usize;
        frame_energies[percentile_index.min(frame_energies.len() - 1)]
    }

    /// Calculate total harmonic distortion
    pub fn calculate_thd(audio_samples: &[f32], sample_rate: u32) -> f64 {
        // Simplified THD calculation
        // In practice, this would need more sophisticated spectral analysis
        let window_size = sample_rate as usize / 10; // 100ms windows
        let mut thd_values = Vec::new();

        for chunk in audio_samples.chunks(window_size) {
            if chunk.len() < window_size {
                continue;
            }

            // Calculate fundamental and harmonics using FFT
            let thd = calculate_chunk_thd(chunk, sample_rate as f32);
            thd_values.push(thd);
        }

        if thd_values.is_empty() {
            0.0
        } else {
            thd_values.iter().sum::<f64>() / thd_values.len() as f64
        }
    }

    fn calculate_chunk_thd(chunk: &[f32], sample_rate: f32) -> f64 {
        // Placeholder THD calculation
        // Real implementation would use FFT to find fundamental and harmonics
        let rms: f64 = chunk.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / chunk.len() as f64;

        rms.sqrt() * 0.01 // Simplified estimate
    }
}
