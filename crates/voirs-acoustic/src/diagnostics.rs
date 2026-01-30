//! Advanced diagnostics and analysis tools for acoustic modeling
//!
//! This module provides sophisticated diagnostic capabilities for:
//! - Error context and debugging
//! - Performance analysis and bottleneck detection
//! - Quality assessment and degradation detection
//! - Resource usage profiling

use crate::{AcousticError, MelSpectrogram, Phoneme, Result};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Diagnostic context for tracking synthesis operations
#[derive(Debug, Clone)]
pub struct DiagnosticContext {
    /// Operation identifier
    pub operation_id: String,
    /// Start timestamp
    pub start_time: Instant,
    /// Input characteristics
    pub input_info: InputInfo,
    /// Intermediate metrics collected during processing
    pub metrics: HashMap<String, f32>,
    /// Warnings encountered
    pub warnings: Vec<String>,
    /// Performance checkpoints
    pub checkpoints: Vec<Checkpoint>,
}

/// Information about input data
#[derive(Debug, Clone)]
pub struct InputInfo {
    /// Number of phonemes
    pub phoneme_count: usize,
    /// Estimated duration in seconds
    pub estimated_duration: f32,
    /// Language if known
    pub language: Option<String>,
    /// Speaker ID if applicable
    pub speaker_id: Option<u32>,
}

/// Performance checkpoint for tracking operation stages
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Checkpoint name
    pub name: String,
    /// Time elapsed since operation start
    pub elapsed: Duration,
    /// Memory usage delta (if available)
    pub memory_delta_mb: Option<f32>,
    /// Additional metrics
    pub metrics: HashMap<String, f32>,
}

impl DiagnosticContext {
    /// Create new diagnostic context
    pub fn new(operation_id: impl Into<String>, phonemes: &[Phoneme]) -> Self {
        let estimated_duration = phonemes.iter().filter_map(|p| p.duration).sum();

        Self {
            operation_id: operation_id.into(),
            start_time: Instant::now(),
            input_info: InputInfo {
                phoneme_count: phonemes.len(),
                estimated_duration,
                language: None,
                speaker_id: None,
            },
            metrics: HashMap::new(),
            warnings: Vec::new(),
            checkpoints: Vec::new(),
        }
    }

    /// Add a metric
    pub fn add_metric(&mut self, key: impl Into<String>, value: f32) {
        self.metrics.insert(key.into(), value);
    }

    /// Add a warning
    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }

    /// Add a checkpoint
    pub fn checkpoint(&mut self, name: impl Into<String>) {
        self.checkpoints.push(Checkpoint {
            name: name.into(),
            elapsed: self.start_time.elapsed(),
            memory_delta_mb: None,
            metrics: HashMap::new(),
        });
    }

    /// Add a checkpoint with metrics
    pub fn checkpoint_with_metrics(
        &mut self,
        name: impl Into<String>,
        metrics: HashMap<String, f32>,
    ) {
        self.checkpoints.push(Checkpoint {
            name: name.into(),
            elapsed: self.start_time.elapsed(),
            memory_delta_mb: None,
            metrics,
        });
    }

    /// Get total elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Generate a diagnostic report
    pub fn report(&self) -> DiagnosticReport {
        DiagnosticReport::from_context(self)
    }
}

/// Comprehensive diagnostic report
#[derive(Debug, Clone)]
pub struct DiagnosticReport {
    /// Operation identifier
    pub operation_id: String,
    /// Total execution time
    pub total_time: Duration,
    /// Real-time factor (if audio duration known)
    pub rtf: Option<f32>,
    /// Input characteristics
    pub input_info: InputInfo,
    /// Performance breakdown by stage
    pub stage_timings: Vec<(String, Duration)>,
    /// Collected metrics
    pub metrics: HashMap<String, f32>,
    /// Warnings
    pub warnings: Vec<String>,
    /// Performance assessment
    pub performance_assessment: PerformanceAssessment,
}

impl DiagnosticReport {
    /// Create report from context
    pub fn from_context(ctx: &DiagnosticContext) -> Self {
        let total_time = ctx.elapsed();

        // Calculate RTF if we have audio duration
        let rtf = if ctx.input_info.estimated_duration > 0.0 {
            Some(total_time.as_secs_f32() / ctx.input_info.estimated_duration)
        } else {
            None
        };

        // Extract stage timings from checkpoints
        let mut stage_timings = Vec::new();
        let mut prev_elapsed = Duration::from_secs(0);
        for checkpoint in &ctx.checkpoints {
            let stage_duration = checkpoint.elapsed - prev_elapsed;
            stage_timings.push((checkpoint.name.clone(), stage_duration));
            prev_elapsed = checkpoint.elapsed;
        }

        // Assess performance
        let performance_assessment = PerformanceAssessment::assess(rtf, total_time, &ctx.warnings);

        Self {
            operation_id: ctx.operation_id.clone(),
            total_time,
            rtf,
            input_info: ctx.input_info.clone(),
            stage_timings,
            metrics: ctx.metrics.clone(),
            warnings: ctx.warnings.clone(),
            performance_assessment,
        }
    }

    /// Format report as human-readable string
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!(
            "=== Diagnostic Report: {} ===\n",
            self.operation_id
        ));
        report.push_str(&format!(
            "Total Time: {:.3}s\n",
            self.total_time.as_secs_f32()
        ));

        if let Some(rtf) = self.rtf {
            report.push_str(&format!("Real-Time Factor: {:.3}x\n", rtf));
        }

        report.push_str(&format!(
            "Input: {} phonemes ({:.2}s estimated)\n",
            self.input_info.phoneme_count, self.input_info.estimated_duration
        ));

        if !self.stage_timings.is_empty() {
            report.push_str("\nStage Timings:\n");
            for (name, duration) in &self.stage_timings {
                let percentage = (duration.as_secs_f32() / self.total_time.as_secs_f32()) * 100.0;
                report.push_str(&format!(
                    "  {}: {:.3}s ({:.1}%)\n",
                    name,
                    duration.as_secs_f32(),
                    percentage
                ));
            }
        }

        if !self.metrics.is_empty() {
            report.push_str("\nMetrics:\n");
            for (key, value) in &self.metrics {
                report.push_str(&format!("  {}: {:.4}\n", key, value));
            }
        }

        if !self.warnings.is_empty() {
            report.push_str("\nWarnings:\n");
            for warning in &self.warnings {
                report.push_str(&format!("  - {}\n", warning));
            }
        }

        report.push_str(&format!(
            "\nPerformance: {}\n",
            self.performance_assessment.summary()
        ));

        if !self.performance_assessment.recommendations.is_empty() {
            report.push_str("\nRecommendations:\n");
            for rec in &self.performance_assessment.recommendations {
                report.push_str(&format!("  • {}\n", rec));
            }
        }

        report
    }
}

/// Performance assessment with recommendations
#[derive(Debug, Clone)]
pub struct PerformanceAssessment {
    /// Overall performance rating
    pub rating: PerformanceRating,
    /// Specific recommendations
    pub recommendations: Vec<String>,
    /// Bottleneck detection
    pub bottlenecks: Vec<String>,
}

/// Performance rating categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceRating {
    /// Excellent performance (RTF < 0.1)
    Excellent,
    /// Good performance (RTF 0.1-0.2)
    Good,
    /// Acceptable performance (RTF 0.2-0.3)
    Acceptable,
    /// Slow performance (RTF 0.3-0.5)
    Slow,
    /// Very slow performance (RTF > 0.5)
    VerySlow,
}

impl PerformanceAssessment {
    /// Assess performance based on metrics
    pub fn assess(rtf: Option<f32>, total_time: Duration, warnings: &[String]) -> Self {
        let rating = if let Some(rtf) = rtf {
            if rtf < 0.1 {
                PerformanceRating::Excellent
            } else if rtf < 0.2 {
                PerformanceRating::Good
            } else if rtf < 0.3 {
                PerformanceRating::Acceptable
            } else if rtf < 0.5 {
                PerformanceRating::Slow
            } else {
                PerformanceRating::VerySlow
            }
        } else {
            // Without RTF, assess based on absolute time for typical utterances
            if total_time < Duration::from_millis(200) {
                PerformanceRating::Excellent
            } else if total_time < Duration::from_millis(500) {
                PerformanceRating::Good
            } else if total_time < Duration::from_secs(1) {
                PerformanceRating::Acceptable
            } else {
                PerformanceRating::Slow
            }
        };

        let mut recommendations = Vec::new();
        let mut bottlenecks = Vec::new();

        // Generate recommendations based on rating
        match rating {
            PerformanceRating::Excellent | PerformanceRating::Good => {
                // No recommendations needed
            }
            PerformanceRating::Acceptable => {
                recommendations.push("Consider enabling GPU acceleration if available".to_string());
                recommendations.push("Check if model quantization is enabled".to_string());
            }
            PerformanceRating::Slow | PerformanceRating::VerySlow => {
                recommendations.push(
                    "Enable GPU acceleration (CUDA/Metal) for significant speedup".to_string(),
                );
                recommendations
                    .push("Use INT8 quantization to reduce compute requirements".to_string());
                recommendations.push("Enable result caching for repeated synthesis".to_string());
                recommendations
                    .push("Consider using streaming synthesis for large texts".to_string());
                bottlenecks.push("Synthesis speed below real-time requirements".to_string());
            }
        }

        // Check for warning-based recommendations
        if warnings.iter().any(|w| w.contains("memory")) {
            recommendations
                .push("High memory usage detected - consider batch size reduction".to_string());
            bottlenecks.push("Memory pressure detected".to_string());
        }

        Self {
            rating,
            recommendations,
            bottlenecks,
        }
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        match self.rating {
            PerformanceRating::Excellent => "Excellent (optimal performance)".to_string(),
            PerformanceRating::Good => "Good (meeting targets)".to_string(),
            PerformanceRating::Acceptable => "Acceptable (room for improvement)".to_string(),
            PerformanceRating::Slow => "Slow (below targets)".to_string(),
            PerformanceRating::VerySlow => {
                "Very Slow (significant optimization needed)".to_string()
            }
        }
    }
}

/// Mel spectrogram quality analyzer
pub struct MelQualityAnalyzer {
    /// Spectral balance thresholds
    spectral_balance_threshold: f32,
    /// Temporal smoothness threshold
    temporal_smoothness_threshold: f32,
    /// Dynamic range thresholds
    dynamic_range_min: f32,
    dynamic_range_max: f32,
}

impl Default for MelQualityAnalyzer {
    fn default() -> Self {
        Self {
            spectral_balance_threshold: 0.3,
            temporal_smoothness_threshold: 0.1,
            dynamic_range_min: 20.0,
            dynamic_range_max: 80.0,
        }
    }
}

impl MelQualityAnalyzer {
    /// Perform comprehensive quality analysis
    pub fn analyze(&self, mel: &MelSpectrogram) -> MelQualityReport {
        let spectral_balance = self.analyze_spectral_balance(mel);
        let temporal_smoothness = self.analyze_temporal_smoothness(mel);
        let dynamic_range = self.analyze_dynamic_range(mel);
        let noise_level = self.estimate_noise_level(mel);

        // Detect quality issues
        let mut issues = Vec::new();

        if spectral_balance.imbalance > self.spectral_balance_threshold {
            issues.push(format!(
                "Spectral imbalance detected: {:.2}",
                spectral_balance.imbalance
            ));
        }

        if temporal_smoothness.roughness > self.temporal_smoothness_threshold {
            issues.push(format!(
                "Temporal roughness detected: {:.2}",
                temporal_smoothness.roughness
            ));
        }

        if dynamic_range.range_db < self.dynamic_range_min {
            issues.push(format!(
                "Low dynamic range: {:.1} dB",
                dynamic_range.range_db
            ));
        } else if dynamic_range.range_db > self.dynamic_range_max {
            issues.push(format!(
                "Excessive dynamic range: {:.1} dB",
                dynamic_range.range_db
            ));
        }

        if noise_level > 0.1 {
            issues.push(format!("High noise level detected: {:.2}", noise_level));
        }

        // Calculate overall quality score (0-100)
        let quality_score = self.calculate_quality_score(
            &spectral_balance,
            &temporal_smoothness,
            &dynamic_range,
            noise_level,
        );

        MelQualityReport {
            quality_score,
            spectral_balance,
            temporal_smoothness,
            dynamic_range,
            noise_level,
            issues,
        }
    }

    fn analyze_spectral_balance(&self, mel: &MelSpectrogram) -> SpectralBalance {
        if mel.n_mels == 0 || mel.n_frames == 0 {
            return SpectralBalance {
                imbalance: 0.0,
                low_energy: 0.0,
                mid_energy: 0.0,
                high_energy: 0.0,
            };
        }

        // Divide spectrum into low, mid, high regions
        let low_end = mel.n_mels / 3;
        let mid_end = 2 * mel.n_mels / 3;

        let mut low_energy = 0.0;
        let mut mid_energy = 0.0;
        let mut high_energy = 0.0;

        for (idx, channel) in mel.data.iter().enumerate() {
            let energy: f32 = channel.iter().map(|v| v * v).sum();
            if idx < low_end {
                low_energy += energy;
            } else if idx < mid_end {
                mid_energy += energy;
            } else {
                high_energy += energy;
            }
        }

        let total_energy = low_energy + mid_energy + high_energy;
        if total_energy > 0.0 {
            low_energy /= total_energy;
            mid_energy /= total_energy;
            high_energy /= total_energy;
        }

        // Calculate imbalance (deviation from equal distribution)
        let ideal = 1.0 / 3.0;
        let imbalance =
            ((low_energy - ideal).abs() + (mid_energy - ideal).abs() + (high_energy - ideal).abs())
                / 2.0;

        SpectralBalance {
            imbalance,
            low_energy,
            mid_energy,
            high_energy,
        }
    }

    fn analyze_temporal_smoothness(&self, mel: &MelSpectrogram) -> TemporalSmoothness {
        if mel.n_frames < 2 {
            return TemporalSmoothness {
                roughness: 0.0,
                mean_variation: 0.0,
                max_variation: 0.0,
            };
        }

        let mut total_variation = 0.0;
        let mut max_variation: f32 = 0.0;
        let mut count = 0;

        for channel in &mel.data {
            for window in channel.windows(2) {
                let variation = (window[1] - window[0]).abs();
                total_variation += variation;
                max_variation = max_variation.max(variation);
                count += 1;
            }
        }

        let mean_variation = if count > 0 {
            total_variation / count as f32
        } else {
            0.0
        };

        // Roughness is a combination of mean and max variation
        let roughness = (mean_variation * 0.7 + max_variation * 0.3).min(1.0);

        TemporalSmoothness {
            roughness,
            mean_variation,
            max_variation,
        }
    }

    fn analyze_dynamic_range(&self, mel: &MelSpectrogram) -> DynamicRange {
        let all_values: Vec<f32> = mel.data.iter().flat_map(|c| c.iter().copied()).collect();

        if all_values.is_empty() {
            return DynamicRange {
                range_db: 0.0,
                min_value: 0.0,
                max_value: 0.0,
            };
        }

        let min_value = all_values.iter().copied().fold(f32::INFINITY, f32::min);
        let max_value = all_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Calculate dynamic range in dB (avoid log of zero)
        let range_db = if max_value > min_value && max_value > 1e-10 {
            20.0 * (max_value / min_value.max(1e-10)).log10()
        } else {
            0.0
        };

        DynamicRange {
            range_db,
            min_value,
            max_value,
        }
    }

    fn estimate_noise_level(&self, mel: &MelSpectrogram) -> f32 {
        if mel.n_frames < 10 {
            return 0.0;
        }

        // Estimate noise by analyzing high-frequency variation
        let mut noise_estimate = 0.0;
        let mut count = 0;

        // Look at last few mel bins (high frequency) for noise
        let high_freq_bins = (mel.n_mels * 3 / 4).max(mel.n_mels.saturating_sub(10))..mel.n_mels;

        for idx in high_freq_bins {
            if idx < mel.data.len() {
                let channel = &mel.data[idx];
                if channel.len() >= 3 {
                    // Calculate second derivative (measure of jitter)
                    for window in channel.windows(3) {
                        let second_deriv = (window[2] - 2.0 * window[1] + window[0]).abs();
                        noise_estimate += second_deriv;
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            noise_estimate / count as f32
        } else {
            0.0
        }
    }

    fn calculate_quality_score(
        &self,
        spectral_balance: &SpectralBalance,
        temporal_smoothness: &TemporalSmoothness,
        dynamic_range: &DynamicRange,
        noise_level: f32,
    ) -> f32 {
        // Calculate component scores (0-1)
        let balance_score = (1.0 - spectral_balance.imbalance / 0.5).clamp(0.0, 1.0);
        let smoothness_score = (1.0 - temporal_smoothness.roughness / 0.2).clamp(0.0, 1.0);

        // Dynamic range score (optimal around 40-60 dB)
        let dr_deviation = (dynamic_range.range_db - 50.0).abs();
        let dr_score = (1.0 - dr_deviation / 30.0).clamp(0.0, 1.0);

        let noise_score = (1.0 - noise_level / 0.2).clamp(0.0, 1.0);

        // Weighted average
        let overall =
            balance_score * 0.25 + smoothness_score * 0.30 + dr_score * 0.25 + noise_score * 0.20;

        (overall * 100.0).clamp(0.0, 100.0)
    }
}

/// Spectral balance analysis
#[derive(Debug, Clone)]
pub struct SpectralBalance {
    /// Overall imbalance measure (0-1, lower is better)
    pub imbalance: f32,
    /// Low frequency energy ratio
    pub low_energy: f32,
    /// Mid frequency energy ratio
    pub mid_energy: f32,
    /// High frequency energy ratio
    pub high_energy: f32,
}

/// Temporal smoothness analysis
#[derive(Debug, Clone)]
pub struct TemporalSmoothness {
    /// Roughness measure (0-1, lower is better)
    pub roughness: f32,
    /// Mean frame-to-frame variation
    pub mean_variation: f32,
    /// Maximum frame-to-frame variation
    pub max_variation: f32,
}

/// Dynamic range analysis
#[derive(Debug, Clone)]
pub struct DynamicRange {
    /// Dynamic range in dB
    pub range_db: f32,
    /// Minimum value
    pub min_value: f32,
    /// Maximum value
    pub max_value: f32,
}

/// Comprehensive mel quality report
#[derive(Debug, Clone)]
pub struct MelQualityReport {
    /// Overall quality score (0-100)
    pub quality_score: f32,
    /// Spectral balance analysis
    pub spectral_balance: SpectralBalance,
    /// Temporal smoothness analysis
    pub temporal_smoothness: TemporalSmoothness,
    /// Dynamic range analysis
    pub dynamic_range: DynamicRange,
    /// Estimated noise level
    pub noise_level: f32,
    /// Detected issues
    pub issues: Vec<String>,
}

impl MelQualityReport {
    /// Get quality grade
    pub fn grade(&self) -> &'static str {
        if self.quality_score >= 90.0 {
            "Excellent"
        } else if self.quality_score >= 80.0 {
            "Good"
        } else if self.quality_score >= 70.0 {
            "Fair"
        } else if self.quality_score >= 60.0 {
            "Poor"
        } else {
            "Very Poor"
        }
    }

    /// Format report as string
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Mel Quality Analysis ===\n");
        report.push_str(&format!(
            "Overall Score: {:.1}/100 ({})\n",
            self.quality_score,
            self.grade()
        ));
        report.push_str("\nSpectral Balance:\n");
        report.push_str(&format!(
            "  Imbalance: {:.3}\n",
            self.spectral_balance.imbalance
        ));
        report.push_str(&format!(
            "  Low/Mid/High: {:.2}/{:.2}/{:.2}\n",
            self.spectral_balance.low_energy,
            self.spectral_balance.mid_energy,
            self.spectral_balance.high_energy
        ));

        report.push_str("\nTemporal Characteristics:\n");
        report.push_str(&format!(
            "  Roughness: {:.3}\n",
            self.temporal_smoothness.roughness
        ));
        report.push_str(&format!(
            "  Mean Variation: {:.4}\n",
            self.temporal_smoothness.mean_variation
        ));
        report.push_str(&format!(
            "  Max Variation: {:.4}\n",
            self.temporal_smoothness.max_variation
        ));

        report.push_str("\nDynamic Range:\n");
        report.push_str(&format!("  Range: {:.1} dB\n", self.dynamic_range.range_db));
        report.push_str(&format!(
            "  Min/Max: {:.4}/{:.4}\n",
            self.dynamic_range.min_value, self.dynamic_range.max_value
        ));

        report.push_str(&format!("\nNoise Level: {:.4}\n", self.noise_level));

        if !self.issues.is_empty() {
            report.push_str("\nDetected Issues:\n");
            for issue in &self.issues {
                report.push_str(&format!("  ⚠ {}\n", issue));
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_context_creation() {
        let mut phonemes = vec![Phoneme::new("HH"), Phoneme::new("AH")];
        // Set durations manually
        phonemes[0].duration = Some(0.08);
        phonemes[1].duration = Some(0.12);

        let ctx = DiagnosticContext::new("test_op", &phonemes);
        assert_eq!(ctx.operation_id, "test_op");
        assert_eq!(ctx.input_info.phoneme_count, 2);
        assert!(ctx.input_info.estimated_duration > 0.0);
    }

    #[test]
    fn test_diagnostic_context_metrics() {
        let phonemes = vec![];
        let mut ctx = DiagnosticContext::new("test", &phonemes);

        ctx.add_metric("test_metric", 42.0);
        assert_eq!(ctx.metrics.get("test_metric"), Some(&42.0));

        ctx.add_warning("test warning");
        assert_eq!(ctx.warnings.len(), 1);

        ctx.checkpoint("stage1");
        assert_eq!(ctx.checkpoints.len(), 1);
    }

    #[test]
    fn test_performance_assessment() {
        let assessment = PerformanceAssessment::assess(Some(0.05), Duration::from_millis(100), &[]);
        assert_eq!(assessment.rating, PerformanceRating::Excellent);

        let slow_assessment = PerformanceAssessment::assess(Some(0.6), Duration::from_secs(2), &[]);
        assert_eq!(slow_assessment.rating, PerformanceRating::VerySlow);
        assert!(!slow_assessment.recommendations.is_empty());
    }

    #[test]
    fn test_mel_quality_analyzer() {
        let mel = MelSpectrogram {
            data: vec![vec![0.1, 0.2, 0.15]; 80],
            sample_rate: 22050,
            hop_length: 256,
            n_mels: 80,
            n_frames: 3,
        };

        let analyzer = MelQualityAnalyzer::default();
        let report = analyzer.analyze(&mel);

        assert!(report.quality_score >= 0.0 && report.quality_score <= 100.0);
        assert!(!report.grade().is_empty());
    }
}
