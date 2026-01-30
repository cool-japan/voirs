//! Quality gate validation system for automated quality assurance
//!
//! This module provides a comprehensive quality gate system that validates
//! evaluation results against predefined quality standards, ensuring that
//! TTS systems meet minimum quality requirements before deployment.

use crate::EvaluationError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Quality gate configuration defining acceptance criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGateConfig {
    /// Metric accuracy requirements
    pub metric_requirements: MetricRequirements,
    /// Performance requirements
    pub performance_requirements: PerformanceRequirements,
    /// Integration requirements
    pub integration_requirements: IntegrationRequirements,
    /// Validation requirements
    pub validation_requirements: ValidationRequirements,
    /// Gate enforcement level
    pub enforcement_level: EnforcementLevel,
    /// Enable automatic rollback on failure
    pub auto_rollback: bool,
}

impl Default for QualityGateConfig {
    fn default() -> Self {
        Self {
            metric_requirements: MetricRequirements::default(),
            performance_requirements: PerformanceRequirements::default(),
            integration_requirements: IntegrationRequirements::default(),
            validation_requirements: ValidationRequirements::default(),
            enforcement_level: EnforcementLevel::Strict,
            auto_rollback: true,
        }
    }
}

/// Metric accuracy requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRequirements {
    /// PESQ correlation with human ratings (target: > 0.9)
    pub pesq_correlation_threshold: f64,
    /// STOI prediction accuracy on test sets (target: > 0.95)
    pub stoi_accuracy_threshold: f64,
    /// MCD calculation precision variance (target: < 0.01 dB)
    pub mcd_precision_threshold: f64,
    /// Statistical test Type I error (target: < 0.05)
    pub type_i_error_threshold: f64,
    /// Minimum MOS score
    pub min_mos_score: f64,
    /// Minimum intelligibility score
    pub min_intelligibility: f64,
}

impl Default for MetricRequirements {
    fn default() -> Self {
        Self {
            pesq_correlation_threshold: 0.9,
            stoi_accuracy_threshold: 0.95,
            mcd_precision_threshold: 0.01,
            type_i_error_threshold: 0.05,
            min_mos_score: 4.0,
            min_intelligibility: 0.95,
        }
    }
}

/// Performance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    /// Real-time factor threshold (target: < 0.1)
    pub max_rtf: f64,
    /// Maximum memory usage in GB (target: < 1GB)
    pub max_memory_gb: f64,
    /// Minimum GPU acceleration speedup (target: > 10x)
    pub min_gpu_speedup: f64,
    /// Minimum parallel efficiency on multi-core (target: > 0.8)
    pub min_parallel_efficiency: f64,
    /// Maximum latency in milliseconds
    pub max_latency_ms: u64,
    /// Minimum throughput (samples per second)
    pub min_throughput: f64,
}

impl Default for PerformanceRequirements {
    fn default() -> Self {
        Self {
            max_rtf: 0.1,
            max_memory_gb: 1.0,
            min_gpu_speedup: 10.0,
            min_parallel_efficiency: 0.8,
            max_latency_ms: 100,
            min_throughput: 22050.0, // At least 1 second of audio per second
        }
    }
}

/// Integration requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationRequirements {
    /// Require zero-config operation
    pub zero_config_required: bool,
    /// Maximum streaming latency (target: < 100ms)
    pub max_streaming_latency_ms: u64,
    /// Require cross-platform consistency
    pub cross_platform_consistency: bool,
    /// Require API backward compatibility
    pub api_backward_compatibility: bool,
    /// Minimum API response time
    pub max_api_response_ms: u64,
}

impl Default for IntegrationRequirements {
    fn default() -> Self {
        Self {
            zero_config_required: true,
            max_streaming_latency_ms: 100,
            cross_platform_consistency: true,
            api_backward_compatibility: true,
            max_api_response_ms: 50,
        }
    }
}

/// Validation requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRequirements {
    /// Reference implementation agreement (target: > 0.99)
    pub reference_agreement_threshold: f64,
    /// Cross-language evaluation accuracy
    pub cross_language_accuracy_threshold: f64,
    /// Edge case robustness threshold
    pub edge_case_robustness_threshold: f64,
    /// Numerical stability threshold
    pub numerical_stability_threshold: f64,
    /// Minimum test coverage
    pub min_test_coverage: f64,
}

impl Default for ValidationRequirements {
    fn default() -> Self {
        Self {
            reference_agreement_threshold: 0.99,
            cross_language_accuracy_threshold: 0.95,
            edge_case_robustness_threshold: 0.98,
            numerical_stability_threshold: 0.999,
            min_test_coverage: 0.9,
        }
    }
}

/// Enforcement level for quality gates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnforcementLevel {
    /// Advisory only - log warnings but don't block
    Advisory,
    /// Moderate - block on critical failures only
    Moderate,
    /// Strict - block on any failure
    Strict,
    /// Custom - use custom enforcement rules
    Custom,
}

/// Quality gate evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGateResult {
    /// Overall gate status
    pub status: GateStatus,
    /// Metric validation results
    pub metric_results: HashMap<String, ValidationResult>,
    /// Performance validation results
    pub performance_results: HashMap<String, ValidationResult>,
    /// Integration validation results
    pub integration_results: HashMap<String, ValidationResult>,
    /// Validation check results
    pub validation_results: HashMap<String, ValidationResult>,
    /// Overall score (0.0-1.0)
    pub overall_score: f64,
    /// Detailed report
    pub report: QualityGateReport,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Gate status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateStatus {
    /// All checks passed
    Passed,
    /// Some checks failed but within advisory level
    Warning,
    /// Critical checks failed
    Failed,
    /// Gate is bypassed
    Bypassed,
}

/// Validation result for individual checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Check name
    pub check_name: String,
    /// Check passed
    pub passed: bool,
    /// Actual value
    pub actual_value: f64,
    /// Expected threshold
    pub threshold: f64,
    /// Severity level
    pub severity: SeverityLevel,
    /// Additional context
    pub message: String,
}

/// Severity level for validation failures
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SeverityLevel {
    /// Informational only
    Info,
    /// Warning - should be addressed
    Warning,
    /// Error - must be fixed
    Error,
    /// Critical - immediate action required
    Critical,
}

/// Detailed quality gate report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGateReport {
    /// Total checks performed
    pub total_checks: usize,
    /// Checks passed
    pub passed_checks: usize,
    /// Checks failed
    pub failed_checks: usize,
    /// Warnings
    pub warnings: Vec<String>,
    /// Errors
    pub errors: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Execution time
    pub execution_time: Duration,
}

/// Quality gate validator
pub struct QualityGateValidator {
    config: QualityGateConfig,
    history: Vec<QualityGateResult>,
}

impl QualityGateValidator {
    /// Create a new quality gate validator
    pub fn new(config: QualityGateConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
        }
    }

    /// Validate evaluation results against quality gates
    pub fn validate(
        &mut self,
        results: &EvaluationMetrics,
    ) -> Result<QualityGateResult, EvaluationError> {
        let start_time = std::time::Instant::now();

        let mut metric_results = HashMap::new();
        let mut performance_results = HashMap::new();
        let mut integration_results = HashMap::new();
        let mut validation_results = HashMap::new();

        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        let mut recommendations = Vec::new();

        // Validate metric requirements
        self.validate_metrics(
            results,
            &mut metric_results,
            &mut warnings,
            &mut errors,
            &mut recommendations,
        );

        // Validate performance requirements
        self.validate_performance(
            results,
            &mut performance_results,
            &mut warnings,
            &mut errors,
            &mut recommendations,
        );

        // Validate integration requirements
        self.validate_integration(
            results,
            &mut integration_results,
            &mut warnings,
            &mut errors,
            &mut recommendations,
        );

        // Validate validation requirements
        self.validate_validation(
            results,
            &mut validation_results,
            &mut warnings,
            &mut errors,
            &mut recommendations,
        );

        // Calculate overall status and score
        let total_checks = metric_results.len()
            + performance_results.len()
            + integration_results.len()
            + validation_results.len();
        let passed_checks = [
            &metric_results,
            &performance_results,
            &integration_results,
            &validation_results,
        ]
        .iter()
        .flat_map(|m| m.values())
        .filter(|v| v.passed)
        .count();
        let failed_checks = total_checks - passed_checks;

        let overall_score = if total_checks > 0 {
            passed_checks as f64 / total_checks as f64
        } else {
            0.0
        };

        let status = self.determine_gate_status(failed_checks, &errors, overall_score);

        let execution_time = start_time.elapsed();

        let result = QualityGateResult {
            status,
            metric_results,
            performance_results,
            integration_results,
            validation_results,
            overall_score,
            report: QualityGateReport {
                total_checks,
                passed_checks,
                failed_checks,
                warnings,
                errors,
                recommendations,
                execution_time,
            },
            timestamp: SystemTime::now(),
        };

        // Store in history
        self.history.push(result.clone());

        Ok(result)
    }

    fn validate_metrics(
        &self,
        results: &EvaluationMetrics,
        metric_results: &mut HashMap<String, ValidationResult>,
        warnings: &mut Vec<String>,
        errors: &mut Vec<String>,
        recommendations: &mut Vec<String>,
    ) {
        // PESQ correlation
        if let Some(pesq_correlation) = results.pesq_correlation {
            let passed =
                pesq_correlation >= self.config.metric_requirements.pesq_correlation_threshold;
            metric_results.insert(
                "pesq_correlation".to_string(),
                ValidationResult {
                    check_name: "PESQ Correlation with Human Ratings".to_string(),
                    passed,
                    actual_value: pesq_correlation,
                    threshold: self.config.metric_requirements.pesq_correlation_threshold,
                    severity: if passed {
                        SeverityLevel::Info
                    } else {
                        SeverityLevel::Error
                    },
                    message: format!(
                        "PESQ correlation: {:.3} (threshold: {:.3})",
                        pesq_correlation,
                        self.config.metric_requirements.pesq_correlation_threshold
                    ),
                },
            );

            if !passed {
                errors.push(format!(
                    "PESQ correlation ({:.3}) below threshold ({:.3})",
                    pesq_correlation, self.config.metric_requirements.pesq_correlation_threshold
                ));
                recommendations.push(
                    "Consider improving acoustic model training or increasing dataset quality"
                        .to_string(),
                );
            }
        }

        // STOI accuracy
        if let Some(stoi_accuracy) = results.stoi_accuracy {
            let passed = stoi_accuracy >= self.config.metric_requirements.stoi_accuracy_threshold;
            metric_results.insert(
                "stoi_accuracy".to_string(),
                ValidationResult {
                    check_name: "STOI Prediction Accuracy".to_string(),
                    passed,
                    actual_value: stoi_accuracy,
                    threshold: self.config.metric_requirements.stoi_accuracy_threshold,
                    severity: if passed {
                        SeverityLevel::Info
                    } else {
                        SeverityLevel::Error
                    },
                    message: format!(
                        "STOI accuracy: {:.3} (threshold: {:.3})",
                        stoi_accuracy, self.config.metric_requirements.stoi_accuracy_threshold
                    ),
                },
            );

            if !passed {
                errors.push(format!(
                    "STOI accuracy ({:.3}) below threshold ({:.3})",
                    stoi_accuracy, self.config.metric_requirements.stoi_accuracy_threshold
                ));
                recommendations.push(
                    "Improve intelligibility by enhancing vocoder quality or reducing artifacts"
                        .to_string(),
                );
            }
        }

        // MCD precision
        if let Some(mcd_variance) = results.mcd_variance {
            let passed = mcd_variance < self.config.metric_requirements.mcd_precision_threshold;
            metric_results.insert(
                "mcd_precision".to_string(),
                ValidationResult {
                    check_name: "MCD Calculation Precision".to_string(),
                    passed,
                    actual_value: mcd_variance,
                    threshold: self.config.metric_requirements.mcd_precision_threshold,
                    severity: if passed {
                        SeverityLevel::Info
                    } else {
                        SeverityLevel::Warning
                    },
                    message: format!(
                        "MCD variance: {:.4} dB (threshold: {:.4} dB)",
                        mcd_variance, self.config.metric_requirements.mcd_precision_threshold
                    ),
                },
            );

            if !passed {
                warnings.push(format!(
                    "MCD variance ({:.4} dB) exceeds threshold ({:.4} dB)",
                    mcd_variance, self.config.metric_requirements.mcd_precision_threshold
                ));
            }
        }

        // MOS score
        if let Some(mos_score) = results.mos_score {
            let passed = mos_score >= self.config.metric_requirements.min_mos_score;
            metric_results.insert(
                "mos_score".to_string(),
                ValidationResult {
                    check_name: "Mean Opinion Score".to_string(),
                    passed,
                    actual_value: mos_score,
                    threshold: self.config.metric_requirements.min_mos_score,
                    severity: if passed {
                        SeverityLevel::Info
                    } else {
                        SeverityLevel::Critical
                    },
                    message: format!(
                        "MOS: {:.2} (threshold: {:.2})",
                        mos_score, self.config.metric_requirements.min_mos_score
                    ),
                },
            );

            if !passed {
                errors.push(format!(
                    "MOS score ({:.2}) below minimum threshold ({:.2})",
                    mos_score, self.config.metric_requirements.min_mos_score
                ));
                recommendations.push("Critical: Overall quality is below acceptable level. Review entire synthesis pipeline".to_string());
            }
        }
    }

    fn validate_performance(
        &self,
        results: &EvaluationMetrics,
        performance_results: &mut HashMap<String, ValidationResult>,
        warnings: &mut Vec<String>,
        errors: &mut Vec<String>,
        recommendations: &mut Vec<String>,
    ) {
        // Real-time factor
        if let Some(rtf) = results.real_time_factor {
            let passed = rtf < self.config.performance_requirements.max_rtf;
            performance_results.insert(
                "real_time_factor".to_string(),
                ValidationResult {
                    check_name: "Real-Time Factor".to_string(),
                    passed,
                    actual_value: rtf,
                    threshold: self.config.performance_requirements.max_rtf,
                    severity: if passed {
                        SeverityLevel::Info
                    } else {
                        SeverityLevel::Error
                    },
                    message: format!(
                        "RTF: {:.3}x (threshold: {:.3}x)",
                        rtf, self.config.performance_requirements.max_rtf
                    ),
                },
            );

            if !passed {
                errors.push(format!(
                    "Real-time factor ({:.3}x) exceeds threshold ({:.3}x)",
                    rtf, self.config.performance_requirements.max_rtf
                ));
                recommendations.push("Optimize inference pipeline, enable GPU acceleration, or use model quantization".to_string());
            }
        }

        // Memory usage
        if let Some(memory_gb) = results.memory_usage_gb {
            let passed = memory_gb < self.config.performance_requirements.max_memory_gb;
            performance_results.insert(
                "memory_usage".to_string(),
                ValidationResult {
                    check_name: "Memory Usage".to_string(),
                    passed,
                    actual_value: memory_gb,
                    threshold: self.config.performance_requirements.max_memory_gb,
                    severity: if passed {
                        SeverityLevel::Info
                    } else {
                        SeverityLevel::Warning
                    },
                    message: format!(
                        "Memory: {:.2} GB (threshold: {:.2} GB)",
                        memory_gb, self.config.performance_requirements.max_memory_gb
                    ),
                },
            );

            if !passed {
                warnings.push(format!(
                    "Memory usage ({:.2} GB) exceeds threshold ({:.2} GB)",
                    memory_gb, self.config.performance_requirements.max_memory_gb
                ));
                recommendations.push("Reduce model size, implement streaming processing, or use memory-efficient data structures".to_string());
            }
        }

        // Latency
        if let Some(latency_ms) = results.latency_ms {
            let passed = latency_ms < self.config.performance_requirements.max_latency_ms;
            performance_results.insert(
                "latency".to_string(),
                ValidationResult {
                    check_name: "Synthesis Latency".to_string(),
                    passed,
                    actual_value: latency_ms as f64,
                    threshold: self.config.performance_requirements.max_latency_ms as f64,
                    severity: if passed {
                        SeverityLevel::Info
                    } else {
                        SeverityLevel::Error
                    },
                    message: format!(
                        "Latency: {} ms (threshold: {} ms)",
                        latency_ms, self.config.performance_requirements.max_latency_ms
                    ),
                },
            );

            if !passed {
                errors.push(format!(
                    "Latency ({} ms) exceeds threshold ({} ms)",
                    latency_ms, self.config.performance_requirements.max_latency_ms
                ));
                recommendations.push("Optimize model architecture, use faster inference backend, or implement speculative decoding".to_string());
            }
        }
    }

    fn validate_integration(
        &self,
        results: &EvaluationMetrics,
        integration_results: &mut HashMap<String, ValidationResult>,
        warnings: &mut Vec<String>,
        _errors: &mut Vec<String>,
        recommendations: &mut Vec<String>,
    ) {
        // Streaming latency
        if let Some(streaming_latency_ms) = results.streaming_latency_ms {
            let passed = streaming_latency_ms
                < self
                    .config
                    .integration_requirements
                    .max_streaming_latency_ms;
            integration_results.insert(
                "streaming_latency".to_string(),
                ValidationResult {
                    check_name: "Streaming Latency".to_string(),
                    passed,
                    actual_value: streaming_latency_ms as f64,
                    threshold: self
                        .config
                        .integration_requirements
                        .max_streaming_latency_ms as f64,
                    severity: if passed {
                        SeverityLevel::Info
                    } else {
                        SeverityLevel::Warning
                    },
                    message: format!(
                        "Streaming latency: {} ms (threshold: {} ms)",
                        streaming_latency_ms,
                        self.config
                            .integration_requirements
                            .max_streaming_latency_ms
                    ),
                },
            );

            if !passed {
                warnings.push(format!(
                    "Streaming latency ({} ms) exceeds threshold ({} ms)",
                    streaming_latency_ms,
                    self.config
                        .integration_requirements
                        .max_streaming_latency_ms
                ));
                recommendations.push(
                    "Optimize streaming buffer sizes and reduce processing overhead".to_string(),
                );
            }
        }

        // Cross-platform consistency
        if let Some(consistency_score) = results.cross_platform_consistency {
            let passed = consistency_score >= 0.99;
            integration_results.insert(
                "cross_platform_consistency".to_string(),
                ValidationResult {
                    check_name: "Cross-Platform Consistency".to_string(),
                    passed,
                    actual_value: consistency_score,
                    threshold: 0.99,
                    severity: if passed {
                        SeverityLevel::Info
                    } else {
                        SeverityLevel::Warning
                    },
                    message: format!("Consistency: {:.4} (threshold: 0.99)", consistency_score),
                },
            );

            if !passed {
                warnings.push(format!(
                    "Cross-platform consistency ({:.4}) below threshold (0.99)",
                    consistency_score
                ));
                recommendations.push(
                    "Review platform-specific implementations and ensure consistent numerics"
                        .to_string(),
                );
            }
        }
    }

    fn validate_validation(
        &self,
        results: &EvaluationMetrics,
        validation_results: &mut HashMap<String, ValidationResult>,
        warnings: &mut Vec<String>,
        errors: &mut Vec<String>,
        recommendations: &mut Vec<String>,
    ) {
        // Reference agreement
        if let Some(reference_agreement) = results.reference_agreement {
            let passed = reference_agreement
                >= self
                    .config
                    .validation_requirements
                    .reference_agreement_threshold;
            validation_results.insert(
                "reference_agreement".to_string(),
                ValidationResult {
                    check_name: "Reference Implementation Agreement".to_string(),
                    passed,
                    actual_value: reference_agreement,
                    threshold: self
                        .config
                        .validation_requirements
                        .reference_agreement_threshold,
                    severity: if passed {
                        SeverityLevel::Info
                    } else {
                        SeverityLevel::Critical
                    },
                    message: format!(
                        "Reference agreement: {:.4} (threshold: {:.4})",
                        reference_agreement,
                        self.config
                            .validation_requirements
                            .reference_agreement_threshold
                    ),
                },
            );

            if !passed {
                errors.push(format!(
                    "Reference agreement ({:.4}) below threshold ({:.4})",
                    reference_agreement,
                    self.config
                        .validation_requirements
                        .reference_agreement_threshold
                ));
                recommendations.push(
                    "Critical: Implementation may have bugs. Validate against reference carefully"
                        .to_string(),
                );
            }
        }

        // Test coverage
        if let Some(test_coverage) = results.test_coverage {
            let passed = test_coverage >= self.config.validation_requirements.min_test_coverage;
            validation_results.insert(
                "test_coverage".to_string(),
                ValidationResult {
                    check_name: "Test Coverage".to_string(),
                    passed,
                    actual_value: test_coverage,
                    threshold: self.config.validation_requirements.min_test_coverage,
                    severity: if passed {
                        SeverityLevel::Info
                    } else {
                        SeverityLevel::Warning
                    },
                    message: format!(
                        "Test coverage: {:.1}% (threshold: {:.1}%)",
                        test_coverage * 100.0,
                        self.config.validation_requirements.min_test_coverage * 100.0
                    ),
                },
            );

            if !passed {
                warnings.push(format!(
                    "Test coverage ({:.1}%) below threshold ({:.1}%)",
                    test_coverage * 100.0,
                    self.config.validation_requirements.min_test_coverage * 100.0
                ));
                recommendations.push("Increase test coverage to ensure reliability".to_string());
            }
        }
    }

    fn determine_gate_status(
        &self,
        failed_checks: usize,
        errors: &[String],
        overall_score: f64,
    ) -> GateStatus {
        match self.config.enforcement_level {
            EnforcementLevel::Advisory => {
                if failed_checks > 0 {
                    GateStatus::Warning
                } else {
                    GateStatus::Passed
                }
            }
            EnforcementLevel::Moderate => {
                if !errors.is_empty() {
                    GateStatus::Failed
                } else if failed_checks > 0 {
                    GateStatus::Warning
                } else {
                    GateStatus::Passed
                }
            }
            EnforcementLevel::Strict => {
                if failed_checks > 0 {
                    GateStatus::Failed
                } else {
                    GateStatus::Passed
                }
            }
            EnforcementLevel::Custom => {
                // Custom logic: pass if overall score > 0.8
                if overall_score >= 0.8 {
                    if failed_checks > 0 {
                        GateStatus::Warning
                    } else {
                        GateStatus::Passed
                    }
                } else {
                    GateStatus::Failed
                }
            }
        }
    }

    /// Get validation history
    pub fn get_history(&self) -> &[QualityGateResult] {
        &self.history
    }

    /// Clear validation history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Generate summary report
    pub fn generate_summary(&self) -> QualityGateSummary {
        if self.history.is_empty() {
            return QualityGateSummary::default();
        }

        let total_validations = self.history.len();
        let passed = self
            .history
            .iter()
            .filter(|r| r.status == GateStatus::Passed)
            .count();
        let warnings = self
            .history
            .iter()
            .filter(|r| r.status == GateStatus::Warning)
            .count();
        let failed = self
            .history
            .iter()
            .filter(|r| r.status == GateStatus::Failed)
            .count();

        let avg_score =
            self.history.iter().map(|r| r.overall_score).sum::<f64>() / total_validations as f64;

        let recent_trend = if self.history.len() >= 5 {
            let recent_avg = self
                .history
                .iter()
                .rev()
                .take(5)
                .map(|r| r.overall_score)
                .sum::<f64>()
                / 5.0;
            let older_avg = self
                .history
                .iter()
                .rev()
                .skip(5)
                .take(5)
                .map(|r| r.overall_score)
                .sum::<f64>()
                / 5.0;

            if recent_avg > older_avg + 0.05 {
                Trend::Improving
            } else if recent_avg < older_avg - 0.05 {
                Trend::Degrading
            } else {
                Trend::Stable
            }
        } else {
            Trend::Insufficient
        };

        QualityGateSummary {
            total_validations,
            passed,
            warnings,
            failed,
            pass_rate: passed as f64 / total_validations as f64,
            average_score: avg_score,
            recent_trend,
            last_validation: self.history.last().map(|r| r.timestamp),
        }
    }
}

impl Default for QualityGateValidator {
    fn default() -> Self {
        Self::new(QualityGateConfig::default())
    }
}

/// Evaluation metrics for quality gate validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    // Metric requirements
    pub pesq_correlation: Option<f64>,
    pub stoi_accuracy: Option<f64>,
    pub mcd_variance: Option<f64>,
    pub type_i_error: Option<f64>,
    pub mos_score: Option<f64>,
    pub intelligibility: Option<f64>,

    // Performance requirements
    pub real_time_factor: Option<f64>,
    pub memory_usage_gb: Option<f64>,
    pub gpu_speedup: Option<f64>,
    pub parallel_efficiency: Option<f64>,
    pub latency_ms: Option<u64>,
    pub throughput: Option<f64>,

    // Integration requirements
    pub streaming_latency_ms: Option<u64>,
    pub cross_platform_consistency: Option<f64>,
    pub api_response_time_ms: Option<u64>,

    // Validation requirements
    pub reference_agreement: Option<f64>,
    pub cross_language_accuracy: Option<f64>,
    pub edge_case_robustness: Option<f64>,
    pub numerical_stability: Option<f64>,
    pub test_coverage: Option<f64>,
}

impl Default for EvaluationMetrics {
    fn default() -> Self {
        Self {
            pesq_correlation: None,
            stoi_accuracy: None,
            mcd_variance: None,
            type_i_error: None,
            mos_score: None,
            intelligibility: None,
            real_time_factor: None,
            memory_usage_gb: None,
            gpu_speedup: None,
            parallel_efficiency: None,
            latency_ms: None,
            throughput: None,
            streaming_latency_ms: None,
            cross_platform_consistency: None,
            api_response_time_ms: None,
            reference_agreement: None,
            cross_language_accuracy: None,
            edge_case_robustness: None,
            numerical_stability: None,
            test_coverage: None,
        }
    }
}

/// Quality gate summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGateSummary {
    pub total_validations: usize,
    pub passed: usize,
    pub warnings: usize,
    pub failed: usize,
    pub pass_rate: f64,
    pub average_score: f64,
    pub recent_trend: Trend,
    pub last_validation: Option<SystemTime>,
}

impl Default for QualityGateSummary {
    fn default() -> Self {
        Self {
            total_validations: 0,
            passed: 0,
            warnings: 0,
            failed: 0,
            pass_rate: 0.0,
            average_score: 0.0,
            recent_trend: Trend::Insufficient,
            last_validation: None,
        }
    }
}

/// Trend indication
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Trend {
    Improving,
    Stable,
    Degrading,
    Insufficient, // Not enough data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_gate_validator_creation() {
        let validator = QualityGateValidator::default();
        assert_eq!(validator.get_history().len(), 0);
    }

    #[test]
    fn test_quality_gate_validation_all_pass() {
        let mut validator = QualityGateValidator::default();

        let metrics = EvaluationMetrics {
            pesq_correlation: Some(0.95),
            stoi_accuracy: Some(0.97),
            mcd_variance: Some(0.005),
            mos_score: Some(4.5),
            real_time_factor: Some(0.05),
            memory_usage_gb: Some(0.5),
            latency_ms: Some(50),
            reference_agreement: Some(0.995),
            test_coverage: Some(0.95),
            ..Default::default()
        };

        let result = validator.validate(&metrics).unwrap();
        assert_eq!(result.status, GateStatus::Passed);
        assert!(result.overall_score > 0.9);
        assert!(result.report.errors.is_empty());
    }

    #[test]
    fn test_quality_gate_validation_failures() {
        let mut validator = QualityGateValidator::default();

        let metrics = EvaluationMetrics {
            pesq_correlation: Some(0.75), // Below threshold
            stoi_accuracy: Some(0.90),    // Below threshold
            mcd_variance: Some(0.02),     // Above threshold
            mos_score: Some(3.5),         // Below threshold
            real_time_factor: Some(0.15), // Above threshold
            ..Default::default()
        };

        let result = validator.validate(&metrics).unwrap();
        assert_eq!(result.status, GateStatus::Failed);
        assert!(!result.report.errors.is_empty());
        assert!(!result.report.recommendations.is_empty());
    }

    #[test]
    fn test_quality_gate_advisory_mode() {
        let config = QualityGateConfig {
            enforcement_level: EnforcementLevel::Advisory,
            ..Default::default()
        };
        let mut validator = QualityGateValidator::new(config);

        let metrics = EvaluationMetrics {
            mos_score: Some(3.5), // Below threshold but advisory mode
            ..Default::default()
        };

        let result = validator.validate(&metrics).unwrap();
        assert_eq!(result.status, GateStatus::Warning);
    }

    #[test]
    fn test_quality_gate_history() {
        let mut validator = QualityGateValidator::default();

        let metrics = EvaluationMetrics {
            mos_score: Some(4.5),
            ..Default::default()
        };

        validator.validate(&metrics).unwrap();
        validator.validate(&metrics).unwrap();

        assert_eq!(validator.get_history().len(), 2);

        let summary = validator.generate_summary();
        assert_eq!(summary.total_validations, 2);
        assert!(summary.pass_rate > 0.0);
    }

    #[test]
    fn test_trend_calculation() {
        let mut validator = QualityGateValidator::default();

        // Simulate improving trend
        for i in 0..10 {
            let score = 0.7 + (i as f64 * 0.03);
            let metrics = EvaluationMetrics {
                mos_score: Some(score * 5.0),
                pesq_correlation: Some(score),
                ..Default::default()
            };
            validator.validate(&metrics).unwrap();
        }

        let summary = validator.generate_summary();
        assert_eq!(summary.recent_trend, Trend::Improving);
    }

    #[test]
    fn test_severity_levels() {
        let critical = ValidationResult {
            check_name: "Test".to_string(),
            passed: false,
            actual_value: 0.5,
            threshold: 0.9,
            severity: SeverityLevel::Critical,
            message: "Test message".to_string(),
        };

        let warning = ValidationResult {
            check_name: "Test".to_string(),
            passed: false,
            actual_value: 0.8,
            threshold: 0.9,
            severity: SeverityLevel::Warning,
            message: "Test message".to_string(),
        };

        assert!(critical.severity > warning.severity);
    }

    #[test]
    fn test_custom_enforcement() {
        let config = QualityGateConfig {
            enforcement_level: EnforcementLevel::Custom,
            ..Default::default()
        };
        let mut validator = QualityGateValidator::new(config);

        // Overall score > 0.8 should pass with warnings
        let metrics = EvaluationMetrics {
            pesq_correlation: Some(0.95),
            stoi_accuracy: Some(0.97),
            mcd_variance: Some(0.008), // Pass (threshold 0.01)
            mos_score: Some(4.5),
            real_time_factor: Some(0.05),
            memory_usage_gb: Some(0.5),
            ..Default::default()
        };

        let result = validator.validate(&metrics).unwrap();
        // With 6 passing checks, overall score should be > 0.8
        assert!(result.overall_score >= 0.8);
        assert!(matches!(
            result.status,
            GateStatus::Warning | GateStatus::Passed
        ));
    }
}
