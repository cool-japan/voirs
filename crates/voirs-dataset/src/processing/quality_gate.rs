//! Dataset quality gate system for enforcing quality standards
//!
//! This module provides a comprehensive quality gate system similar to CI/CD pipelines,
//! allowing you to define and enforce quality criteria on datasets before they are used
//! for training or deployment.
//!
//! # Features
//!
//! - **Multi-dimensional Quality Checks**: Audio quality, text quality, duration, metadata
//! - **Flexible Thresholds**: Define pass/fail/warning levels for each criterion
//! - **Detailed Reporting**: Comprehensive reports showing which samples pass or fail
//! - **Custom Rules**: Define custom quality rules and validators
//! - **Severity Levels**: Blocking errors, warnings, and informational messages
//! - **Batch Validation**: Efficiently validate entire datasets
//!
//! # Examples
//!
//! ```no_run
//! use voirs_dataset::processing::quality_gate::{QualityGate, QualityRule, RuleType};
//! use voirs_dataset::DatasetSample;
//!
//! let mut gate = QualityGate::strict();
//!
//! // Add custom rules
//! gate.add_rule(QualityRule::new(
//!     "min_duration",
//!     RuleType::MinDuration(1.0),
//!     "Samples must be at least 1 second"
//! ));
//!
//! let samples: Vec<DatasetSample> = vec![/* ... */];
//! let report = gate.validate(&samples).unwrap();
//!
//! if report.passed() {
//!     println!("All quality checks passed!");
//! } else {
//!     println!("Quality gate failed: {} errors", report.error_count);
//! }
//! ```

use crate::{DatasetSample, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Severity level for quality issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    /// Informational message (does not fail the gate)
    Info,
    /// Warning (does not fail the gate but should be reviewed)
    Warning,
    /// Error (fails the quality gate)
    Error,
}

/// Type of quality rule to apply
#[derive(Serialize, Deserialize)]
pub enum RuleType {
    /// Minimum audio duration in seconds
    MinDuration(f32),
    /// Maximum audio duration in seconds
    MaxDuration(f32),
    /// Minimum overall quality score (0.0-1.0)
    MinQuality(f32),
    /// Minimum SNR in dB
    MinSnr(f32),
    /// Maximum clipping ratio (0.0-1.0)
    MaxClipping(f32),
    /// Minimum dynamic range in dB
    MinDynamicRange(f32),
    /// Minimum text length in characters
    MinTextLength(usize),
    /// Maximum text length in characters
    MaxTextLength(usize),
    /// Require speaker metadata
    RequireSpeaker,
    /// Require quality metrics
    RequireQualityMetrics,
    /// Text must not be empty
    NonEmptyText,
    /// Text must match pattern (uses regex)
    TextPattern(String),
    /// Sample rate must match
    RequiredSampleRate(u32),
    /// Number of channels must match
    RequiredChannels(u32),
    /// Custom validation function (not serializable)
    #[serde(skip)]
    Custom(Box<dyn Fn(&DatasetSample) -> bool + Send + Sync>),
}

impl std::fmt::Debug for RuleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuleType::MinDuration(v) => write!(f, "MinDuration({:?})", v),
            RuleType::MaxDuration(v) => write!(f, "MaxDuration({:?})", v),
            RuleType::MinQuality(v) => write!(f, "MinQuality({:?})", v),
            RuleType::MinSnr(v) => write!(f, "MinSnr({:?})", v),
            RuleType::MaxClipping(v) => write!(f, "MaxClipping({:?})", v),
            RuleType::MinDynamicRange(v) => write!(f, "MinDynamicRange({:?})", v),
            RuleType::MinTextLength(v) => write!(f, "MinTextLength({:?})", v),
            RuleType::MaxTextLength(v) => write!(f, "MaxTextLength({:?})", v),
            RuleType::RequireSpeaker => write!(f, "RequireSpeaker"),
            RuleType::RequireQualityMetrics => write!(f, "RequireQualityMetrics"),
            RuleType::NonEmptyText => write!(f, "NonEmptyText"),
            RuleType::TextPattern(v) => write!(f, "TextPattern({:?})", v),
            RuleType::RequiredSampleRate(v) => write!(f, "RequiredSampleRate({:?})", v),
            RuleType::RequiredChannels(v) => write!(f, "RequiredChannels({:?})", v),
            RuleType::Custom(_) => write!(f, "Custom(<function>)"),
        }
    }
}

impl Clone for RuleType {
    fn clone(&self) -> Self {
        match self {
            RuleType::MinDuration(v) => RuleType::MinDuration(*v),
            RuleType::MaxDuration(v) => RuleType::MaxDuration(*v),
            RuleType::MinQuality(v) => RuleType::MinQuality(*v),
            RuleType::MinSnr(v) => RuleType::MinSnr(*v),
            RuleType::MaxClipping(v) => RuleType::MaxClipping(*v),
            RuleType::MinDynamicRange(v) => RuleType::MinDynamicRange(*v),
            RuleType::MinTextLength(v) => RuleType::MinTextLength(*v),
            RuleType::MaxTextLength(v) => RuleType::MaxTextLength(*v),
            RuleType::RequireSpeaker => RuleType::RequireSpeaker,
            RuleType::RequireQualityMetrics => RuleType::RequireQualityMetrics,
            RuleType::NonEmptyText => RuleType::NonEmptyText,
            RuleType::TextPattern(v) => RuleType::TextPattern(v.clone()),
            RuleType::RequiredSampleRate(v) => RuleType::RequiredSampleRate(*v),
            RuleType::RequiredChannels(v) => RuleType::RequiredChannels(*v),
            RuleType::Custom(_) => {
                panic!("Cannot clone RuleType::Custom - use a different rule type or create a new rule")
            }
        }
    }
}

/// A single quality rule with associated metadata
#[derive(Clone, Serialize, Deserialize)]
pub struct QualityRule {
    /// Unique identifier for this rule
    pub id: String,
    /// Type of rule
    #[serde(skip_serializing_if = "is_custom")]
    pub rule_type: RuleType,
    /// Human-readable description
    pub description: String,
    /// Severity level if rule is violated
    pub severity: Severity,
    /// Whether this rule is enabled
    pub enabled: bool,
}

fn is_custom(rule_type: &RuleType) -> bool {
    matches!(rule_type, RuleType::Custom(_))
}

impl std::fmt::Debug for QualityRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QualityRule")
            .field("id", &self.id)
            .field("description", &self.description)
            .field("severity", &self.severity)
            .field("enabled", &self.enabled)
            .finish()
    }
}

impl QualityRule {
    /// Create a new quality rule
    pub fn new(id: impl Into<String>, rule_type: RuleType, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            rule_type,
            description: description.into(),
            severity: Severity::Error,
            enabled: true,
        }
    }

    /// Set severity level
    pub fn with_severity(mut self, severity: Severity) -> Self {
        self.severity = severity;
        self
    }

    /// Enable or disable this rule
    pub fn set_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Check if a sample passes this rule
    pub fn check(&self, sample: &DatasetSample) -> bool {
        if !self.enabled {
            return true;
        }

        match &self.rule_type {
            RuleType::MinDuration(min) => sample.audio.duration() >= *min,
            RuleType::MaxDuration(max) => sample.audio.duration() <= *max,
            RuleType::MinQuality(min) => sample.quality.overall_quality.is_some_and(|q| q >= *min),
            RuleType::MinSnr(min) => sample.quality.snr.is_some_and(|s| s >= *min),
            RuleType::MaxClipping(max) => sample.quality.clipping.is_none_or(|c| c <= *max),
            RuleType::MinDynamicRange(min) => {
                sample.quality.dynamic_range.is_some_and(|d| d >= *min)
            }
            RuleType::MinTextLength(min) => sample.text.len() >= *min,
            RuleType::MaxTextLength(max) => sample.text.len() <= *max,
            RuleType::RequireSpeaker => sample.speaker.is_some(),
            RuleType::RequireQualityMetrics => {
                sample.quality.overall_quality.is_some() || sample.quality.snr.is_some()
            }
            RuleType::NonEmptyText => !sample.text.trim().is_empty(),
            RuleType::TextPattern(pattern) => {
                // Simple pattern matching (could use regex crate for more complex patterns)
                sample.text.contains(pattern)
            }
            RuleType::RequiredSampleRate(rate) => sample.audio.sample_rate() == *rate,
            RuleType::RequiredChannels(channels) => sample.audio.channels() == *channels,
            RuleType::Custom(func) => func(sample),
        }
    }
}

/// Issue found during validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    /// Sample ID where issue was found
    pub sample_id: String,
    /// Rule that was violated
    pub rule_id: String,
    /// Description of the rule
    pub rule_description: String,
    /// Severity of the issue
    pub severity: Severity,
    /// Additional context
    pub context: String,
}

/// Statistics about validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStats {
    /// Total samples validated
    pub total_samples: usize,
    /// Samples that passed all rules
    pub passed_samples: usize,
    /// Samples with at least one error
    pub failed_samples: usize,
    /// Samples with only warnings
    pub warning_samples: usize,
    /// Pass rate (0.0-1.0)
    pub pass_rate: f64,
    /// Count by severity
    pub issues_by_severity: HashMap<String, usize>,
    /// Count by rule
    pub issues_by_rule: HashMap<String, usize>,
}

/// Report of quality gate validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGateReport {
    /// All issues found
    pub issues: Vec<QualityIssue>,
    /// Number of errors
    pub error_count: usize,
    /// Number of warnings
    pub warning_count: usize,
    /// Number of info messages
    pub info_count: usize,
    /// Validation statistics
    pub stats: ValidationStats,
    /// IDs of samples that passed
    pub passed_sample_ids: Vec<String>,
    /// IDs of samples that failed
    pub failed_sample_ids: Vec<String>,
}

impl QualityGateReport {
    /// Check if all validation passed (no errors)
    pub fn passed(&self) -> bool {
        self.error_count == 0
    }

    /// Get issues for a specific sample
    pub fn issues_for_sample(&self, sample_id: &str) -> Vec<&QualityIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.sample_id == sample_id)
            .collect()
    }

    /// Get issues by severity
    pub fn issues_by_severity(&self, severity: Severity) -> Vec<&QualityIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.severity == severity)
            .collect()
    }

    /// Print a human-readable summary
    pub fn print(&self) {
        println!("=== Quality Gate Report ===");
        println!("Total samples: {}", self.stats.total_samples);
        println!(
            "Passed: {} ({:.1}%)",
            self.stats.passed_samples,
            self.stats.pass_rate * 100.0
        );
        println!("Failed: {}", self.stats.failed_samples);
        println!("Warnings: {}", self.stats.warning_samples);

        println!("\n--- Issues Summary ---");
        println!("Errors: {}", self.error_count);
        println!("Warnings: {}", self.warning_count);
        println!("Info: {}", self.info_count);

        if !self.stats.issues_by_rule.is_empty() {
            println!("\n--- Top Issues by Rule ---");
            let mut rule_counts: Vec<_> = self.stats.issues_by_rule.iter().collect();
            rule_counts.sort_by_key(|(_, count)| std::cmp::Reverse(**count));

            for (rule_id, count) in rule_counts.iter().take(10) {
                println!("  {}: {} violations", rule_id, count);
            }
        }

        if self.error_count > 0 {
            println!("\n--- Sample Errors (first 10) ---");
            let error_issues: Vec<_> = self.issues_by_severity(Severity::Error);
            for issue in error_issues.iter().take(10) {
                println!(
                    "  [{}] {}: {}",
                    issue.sample_id, issue.rule_id, issue.context
                );
            }

            if error_issues.len() > 10 {
                println!("  ... and {} more errors", error_issues.len() - 10);
            }
        }
    }

    /// Export report to JSON
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Import report from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}

/// Quality gate configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityGateConfig {
    /// Whether to stop validation on first error
    pub fail_fast: bool,
    /// Maximum number of issues to collect
    pub max_issues: Option<usize>,
    /// Whether to include passed samples in report
    pub include_passed: bool,
}

/// Quality gate system for dataset validation
#[derive(Clone, Serialize, Deserialize)]
pub struct QualityGate {
    /// Configuration
    pub config: QualityGateConfig,
    /// Quality rules to enforce
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub rules: Vec<QualityRule>,
}

impl std::fmt::Debug for QualityGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QualityGate")
            .field("config", &self.config)
            .field("rules_count", &self.rules.len())
            .finish()
    }
}

impl Default for QualityGate {
    fn default() -> Self {
        Self::new()
    }
}

impl QualityGate {
    /// Create a new quality gate with no rules
    pub fn new() -> Self {
        Self {
            config: QualityGateConfig::default(),
            rules: Vec::new(),
        }
    }

    /// Create a permissive quality gate (warnings only)
    pub fn permissive() -> Self {
        let mut gate = Self::new();

        gate.add_rule(
            QualityRule::new(
                "non_empty_text",
                RuleType::NonEmptyText,
                "Text must not be empty",
            )
            .with_severity(Severity::Warning),
        );

        gate.add_rule(
            QualityRule::new(
                "min_duration_0.1",
                RuleType::MinDuration(0.1),
                "Audio must be at least 0.1 seconds",
            )
            .with_severity(Severity::Warning),
        );

        gate
    }

    /// Create a standard quality gate with common rules
    pub fn standard() -> Self {
        let mut gate = Self::new();

        // Duration checks
        gate.add_rule(QualityRule::new(
            "min_duration",
            RuleType::MinDuration(0.5),
            "Audio must be at least 0.5 seconds",
        ));

        gate.add_rule(QualityRule::new(
            "max_duration",
            RuleType::MaxDuration(30.0),
            "Audio must not exceed 30 seconds",
        ));

        // Text checks
        gate.add_rule(QualityRule::new(
            "non_empty_text",
            RuleType::NonEmptyText,
            "Text must not be empty",
        ));

        gate.add_rule(QualityRule::new(
            "min_text_length",
            RuleType::MinTextLength(5),
            "Text must be at least 5 characters",
        ));

        // Quality checks (warnings if missing)
        gate.add_rule(
            QualityRule::new(
                "require_quality",
                RuleType::RequireQualityMetrics,
                "Samples should have quality metrics",
            )
            .with_severity(Severity::Warning),
        );

        gate
    }

    /// Create a strict quality gate for production use
    pub fn strict() -> Self {
        let mut gate = Self::standard();

        // Add stricter quality requirements
        gate.add_rule(QualityRule::new(
            "min_quality",
            RuleType::MinQuality(0.7),
            "Overall quality must be at least 0.7",
        ));

        gate.add_rule(QualityRule::new(
            "min_snr",
            RuleType::MinSnr(20.0),
            "SNR must be at least 20 dB",
        ));

        gate.add_rule(QualityRule::new(
            "max_clipping",
            RuleType::MaxClipping(0.05),
            "Clipping must not exceed 5%",
        ));

        gate.add_rule(QualityRule::new(
            "min_dynamic_range",
            RuleType::MinDynamicRange(30.0),
            "Dynamic range must be at least 30 dB",
        ));

        // Require speaker info
        gate.add_rule(QualityRule::new(
            "require_speaker",
            RuleType::RequireSpeaker,
            "Samples must have speaker information",
        ));

        gate
    }

    /// Add a rule to the quality gate
    pub fn add_rule(&mut self, rule: QualityRule) {
        self.rules.push(rule);
    }

    /// Remove a rule by ID
    pub fn remove_rule(&mut self, rule_id: &str) -> Option<QualityRule> {
        if let Some(pos) = self.rules.iter().position(|r| r.id == rule_id) {
            Some(self.rules.remove(pos))
        } else {
            None
        }
    }

    /// Enable or disable a rule
    pub fn set_rule_enabled(&mut self, rule_id: &str, enabled: bool) {
        if let Some(rule) = self.rules.iter_mut().find(|r| r.id == rule_id) {
            rule.enabled = enabled;
        }
    }

    /// Validate a collection of samples
    pub fn validate(&self, samples: &[DatasetSample]) -> Result<QualityGateReport> {
        let mut issues = Vec::new();
        let mut error_count = 0;
        let mut warning_count = 0;
        let mut info_count = 0;
        let mut passed_sample_ids = Vec::new();
        let mut failed_sample_ids = Vec::new();
        let mut sample_has_error: HashMap<String, bool> = HashMap::new();

        let mut issues_by_severity: HashMap<String, usize> = HashMap::new();
        let mut issues_by_rule: HashMap<String, usize> = HashMap::new();

        for sample in samples {
            let mut sample_passed = true;
            let mut sample_has_error_flag = false;

            for rule in &self.rules {
                if !rule.check(sample) {
                    sample_passed = false;

                    let context = self.get_context(sample, &rule.rule_type);

                    let issue = QualityIssue {
                        sample_id: sample.id.clone(),
                        rule_id: rule.id.clone(),
                        rule_description: rule.description.clone(),
                        severity: rule.severity,
                        context,
                    };

                    issues.push(issue);

                    // Update counters
                    match rule.severity {
                        Severity::Error => {
                            error_count += 1;
                            sample_has_error_flag = true;
                        }
                        Severity::Warning => warning_count += 1,
                        Severity::Info => info_count += 1,
                    }

                    // Update statistics
                    *issues_by_severity
                        .entry(format!("{:?}", rule.severity))
                        .or_insert(0) += 1;
                    *issues_by_rule.entry(rule.id.clone()).or_insert(0) += 1;

                    if self.config.fail_fast && rule.severity == Severity::Error {
                        break;
                    }

                    if let Some(max) = self.config.max_issues {
                        if issues.len() >= max {
                            break;
                        }
                    }
                }
            }

            sample_has_error.insert(sample.id.clone(), sample_has_error_flag);

            if sample_passed {
                passed_sample_ids.push(sample.id.clone());
            } else if sample_has_error_flag {
                failed_sample_ids.push(sample.id.clone());
            }

            if self.config.fail_fast && !sample_passed {
                break;
            }

            if let Some(max) = self.config.max_issues {
                if issues.len() >= max {
                    break;
                }
            }
        }

        let total_samples = samples.len();
        let passed_samples = passed_sample_ids.len();
        let failed_samples = failed_sample_ids.len();
        let warning_samples = total_samples - passed_samples - failed_samples;
        let pass_rate = if total_samples > 0 {
            passed_samples as f64 / total_samples as f64
        } else {
            0.0
        };

        let stats = ValidationStats {
            total_samples,
            passed_samples,
            failed_samples,
            warning_samples,
            pass_rate,
            issues_by_severity,
            issues_by_rule,
        };

        Ok(QualityGateReport {
            issues,
            error_count,
            warning_count,
            info_count,
            stats,
            passed_sample_ids,
            failed_sample_ids,
        })
    }

    /// Get context information for an issue
    fn get_context(&self, sample: &DatasetSample, rule_type: &RuleType) -> String {
        match rule_type {
            RuleType::MinDuration(min) => {
                format!("Duration: {:.2}s < {:.2}s", sample.audio.duration(), min)
            }
            RuleType::MaxDuration(max) => {
                format!("Duration: {:.2}s > {:.2}s", sample.audio.duration(), max)
            }
            RuleType::MinQuality(min) => {
                let qual = sample.quality.overall_quality.unwrap_or(0.0);
                format!("Quality: {:.3} < {:.3}", qual, min)
            }
            RuleType::MinSnr(min) => {
                let snr = sample.quality.snr.unwrap_or(0.0);
                format!("SNR: {:.1} dB < {:.1} dB", snr, min)
            }
            RuleType::MaxClipping(max) => {
                let clip = sample.quality.clipping.unwrap_or(0.0);
                format!("Clipping: {:.3} > {:.3}", clip, max)
            }
            RuleType::MinDynamicRange(min) => {
                let dr = sample.quality.dynamic_range.unwrap_or(0.0);
                format!("Dynamic range: {:.1} dB < {:.1} dB", dr, min)
            }
            RuleType::MinTextLength(min) => {
                format!("Text length: {} < {}", sample.text.len(), min)
            }
            RuleType::MaxTextLength(max) => {
                format!("Text length: {} > {}", sample.text.len(), max)
            }
            RuleType::RequireSpeaker => "Speaker information missing".to_string(),
            RuleType::RequireQualityMetrics => "Quality metrics missing".to_string(),
            RuleType::NonEmptyText => "Text is empty".to_string(),
            RuleType::TextPattern(pattern) => {
                format!("Text does not match pattern: {}", pattern)
            }
            RuleType::RequiredSampleRate(rate) => {
                format!("Sample rate: {} != {}", sample.audio.sample_rate(), rate)
            }
            RuleType::RequiredChannels(channels) => {
                format!("Channels: {} != {}", sample.audio.channels(), channels)
            }
            RuleType::Custom(_) => "Custom validation failed".to_string(),
        }
    }

    /// Filter samples to only those that pass the quality gate
    pub fn filter_passing(&self, samples: Vec<DatasetSample>) -> Result<Vec<DatasetSample>> {
        let report = self.validate(&samples)?;
        let passing_ids: std::collections::HashSet<_> = report.passed_sample_ids.iter().collect();

        Ok(samples
            .into_iter()
            .filter(|s| passing_ids.contains(&s.id))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioData, LanguageCode, QualityMetrics, SpeakerInfo};

    fn create_sample(id: &str, duration: f32, text: &str, quality: Option<f32>) -> DatasetSample {
        let audio = AudioData::silence(duration, 22050, 1);
        let qual = QualityMetrics {
            overall_quality: quality,
            snr: Some(25.0),
            clipping: Some(0.01),
            dynamic_range: Some(40.0),
            spectral_quality: Some(0.9),
        };

        DatasetSample::new(id.to_string(), text.to_string(), audio, LanguageCode::EnUs)
            .with_quality(qual)
    }

    #[test]
    fn test_quality_gate_creation() {
        let gate = QualityGate::new();
        assert_eq!(gate.rules.len(), 0);

        let permissive = QualityGate::permissive();
        assert!(!permissive.rules.is_empty());

        let standard = QualityGate::standard();
        assert!(!standard.rules.is_empty());

        let strict = QualityGate::strict();
        assert!(strict.rules.len() > standard.rules.len());
    }

    #[test]
    fn test_rule_checks() {
        let sample = create_sample("test", 2.0, "Hello world", Some(0.8));

        let min_dur = QualityRule::new("test", RuleType::MinDuration(1.0), "Min duration");
        assert!(min_dur.check(&sample));

        let max_dur = QualityRule::new("test", RuleType::MaxDuration(5.0), "Max duration");
        assert!(max_dur.check(&sample));

        let min_qual = QualityRule::new("test", RuleType::MinQuality(0.7), "Min quality");
        assert!(min_qual.check(&sample));

        let non_empty = QualityRule::new("test", RuleType::NonEmptyText, "Non-empty text");
        assert!(non_empty.check(&sample));
    }

    #[test]
    fn test_validation() {
        let mut gate = QualityGate::new();
        gate.add_rule(QualityRule::new(
            "min_duration",
            RuleType::MinDuration(1.0),
            "Minimum duration",
        ));

        let samples = vec![
            create_sample("good", 2.0, "Good sample", Some(0.9)),
            create_sample("short", 0.5, "Too short", Some(0.9)),
        ];

        let report = gate.validate(&samples).unwrap();
        assert_eq!(report.error_count, 1);
        assert_eq!(report.stats.passed_samples, 1);
        assert_eq!(report.stats.failed_samples, 1);
    }

    #[test]
    fn test_multiple_rules() {
        let mut gate = QualityGate::new();
        gate.add_rule(QualityRule::new(
            "min_duration",
            RuleType::MinDuration(1.0),
            "Min duration",
        ));
        gate.add_rule(QualityRule::new(
            "min_quality",
            RuleType::MinQuality(0.8),
            "Min quality",
        ));

        let samples = vec![
            create_sample("good", 2.0, "Good sample", Some(0.9)),
            create_sample("short", 0.5, "Too short", Some(0.9)),
            create_sample("low_quality", 2.0, "Low quality", Some(0.5)),
        ];

        let report = gate.validate(&samples).unwrap();
        assert_eq!(report.error_count, 2);
        assert_eq!(report.stats.passed_samples, 1);
    }

    #[test]
    fn test_severity_levels() {
        let mut gate = QualityGate::new();
        gate.add_rule(
            QualityRule::new("warning_rule", RuleType::MinDuration(1.0), "Warning")
                .with_severity(Severity::Warning),
        );
        gate.add_rule(QualityRule::new(
            "error_rule",
            RuleType::MinQuality(0.8),
            "Error",
        ));

        let sample = create_sample("test", 0.5, "Test", Some(0.5));
        let report = gate.validate(&[sample]).unwrap();

        assert_eq!(report.warning_count, 1);
        assert_eq!(report.error_count, 1);
    }

    #[test]
    fn test_filter_passing() {
        let mut gate = QualityGate::new();
        gate.add_rule(QualityRule::new(
            "min_duration",
            RuleType::MinDuration(1.0),
            "Min duration",
        ));

        let samples = vec![
            create_sample("good1", 2.0, "Good 1", Some(0.9)),
            create_sample("short", 0.5, "Too short", Some(0.9)),
            create_sample("good2", 3.0, "Good 2", Some(0.9)),
        ];

        let passing = gate.filter_passing(samples).unwrap();
        assert_eq!(passing.len(), 2);
    }

    #[test]
    fn test_speaker_requirement() {
        let mut gate = QualityGate::new();
        gate.add_rule(QualityRule::new(
            "require_speaker",
            RuleType::RequireSpeaker,
            "Speaker required",
        ));

        let mut sample_with_speaker = create_sample("with", 2.0, "Test", Some(0.9));
        sample_with_speaker.speaker = Some(SpeakerInfo {
            id: "speaker1".to_string(),
            name: Some("Speaker One".to_string()),
            gender: None,
            age: None,
            accent: None,
            metadata: Default::default(),
        });

        let sample_without_speaker = create_sample("without", 2.0, "Test", Some(0.9));

        let report = gate
            .validate(&[sample_with_speaker, sample_without_speaker])
            .unwrap();
        assert_eq!(report.error_count, 1);
    }

    #[test]
    fn test_rule_enable_disable() {
        let mut gate = QualityGate::new();
        gate.add_rule(QualityRule::new(
            "min_duration",
            RuleType::MinDuration(1.0),
            "Min duration",
        ));

        let sample = create_sample("short", 0.5, "Too short", Some(0.9));

        // Rule enabled - should fail
        let report1 = gate.validate(std::slice::from_ref(&sample)).unwrap();
        assert_eq!(report1.error_count, 1);

        // Disable rule - should pass
        gate.set_rule_enabled("min_duration", false);
        let report2 = gate.validate(&[sample]).unwrap();
        assert_eq!(report2.error_count, 0);
    }

    #[test]
    fn test_report_methods() {
        let mut gate = QualityGate::new();
        gate.add_rule(QualityRule::new(
            "min_quality",
            RuleType::MinQuality(0.8),
            "Min quality",
        ));

        let samples = vec![
            create_sample("good", 2.0, "Good", Some(0.9)),
            create_sample("bad", 2.0, "Bad", Some(0.5)),
        ];

        let report = gate.validate(&samples).unwrap();

        // Report should fail because there's 1 error
        assert!(!report.passed());
        assert_eq!(report.error_count, 1);

        let bad_issues = report.issues_for_sample("bad");
        assert_eq!(bad_issues.len(), 1);

        let errors = report.issues_by_severity(Severity::Error);
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn test_json_serialization() {
        let gate = QualityGate::standard();
        let samples = vec![create_sample("test", 2.0, "Test", Some(0.9))];
        let report = gate.validate(&samples).unwrap();

        let json = report.to_json().unwrap();
        assert!(!json.is_empty());

        let deserialized = QualityGateReport::from_json(&json).unwrap();
        assert_eq!(deserialized.stats.total_samples, report.stats.total_samples);
    }
}
