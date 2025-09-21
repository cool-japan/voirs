//! Evaluation protocol documentation and compliance validation
//!
//! This module provides functionality for creating, validating, and managing
//! evaluation protocols that ensure standardized and reproducible speech synthesis evaluation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Error types for protocol documentation
#[derive(Error, Debug)]
pub enum ProtocolError {
    /// Invalid protocol configuration
    #[error("Invalid protocol configuration: {0}")]
    InvalidConfiguration(String),
    /// Protocol validation failed
    #[error("Protocol validation failed: {0}")]
    ValidationFailed(String),
    /// Missing required protocol component
    #[error("Missing required protocol component: {0}")]
    MissingComponent(String),
}

/// Protocol compliance level
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ComplianceLevel {
    /// Full compliance with all requirements
    Full,
    /// Partial compliance with warnings
    Partial,
    /// Non-compliance with errors
    NonCompliant,
}

/// Quality metric specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSpecification {
    /// Metric name
    pub name: String,
    /// Metric description
    pub description: String,
    /// Expected value range
    pub value_range: (f64, f64),
    /// Minimum threshold for acceptance
    pub threshold: f64,
    /// Metric weight in overall evaluation
    pub weight: f64,
    /// Required or optional
    pub required: bool,
    /// Whether lower values are better (e.g., for distortion metrics)
    pub lower_is_better: bool,
}

/// Audio format specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFormatSpec {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u8,
    /// Bit depth
    pub bit_depth: u8,
    /// Supported file formats
    pub formats: Vec<String>,
}

/// Test dataset specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSpecification {
    /// Dataset name
    pub name: String,
    /// Number of samples
    pub sample_count: usize,
    /// Language requirements
    pub languages: Vec<String>,
    /// Speaker demographics
    pub speaker_demographics: HashMap<String, String>,
    /// Content categories
    pub content_categories: Vec<String>,
}

/// Evaluation protocol configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationProtocol {
    /// Protocol name and version
    pub name: String,
    /// Protocol version
    pub version: String,
    /// Protocol description
    pub description: String,
    /// Required quality metrics
    pub metrics: Vec<MetricSpecification>,
    /// Audio format requirements
    pub audio_format: AudioFormatSpec,
    /// Test dataset specifications
    pub datasets: Vec<DatasetSpecification>,
    /// Evaluation procedures
    pub procedures: Vec<String>,
    /// Compliance requirements
    pub compliance_requirements: HashMap<String, String>,
}

/// Protocol compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Protocol being evaluated
    pub protocol_name: String,
    /// Overall compliance level
    pub compliance_level: ComplianceLevel,
    /// Detailed metric compliance
    pub metric_compliance: HashMap<String, bool>,
    /// Compliance score (0.0 - 1.0)
    pub compliance_score: f64,
    /// Compliance issues and recommendations
    pub issues: Vec<String>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Protocol documentation generator
#[derive(Debug)]
pub struct ProtocolDocumentationGenerator {
    protocols: HashMap<String, EvaluationProtocol>,
}

impl Default for ProtocolDocumentationGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ProtocolDocumentationGenerator {
    /// Create a new protocol documentation generator
    pub fn new() -> Self {
        let mut generator = Self {
            protocols: HashMap::new(),
        };

        // Add standard protocols
        generator.add_itu_t_p862_protocol();
        generator.add_itu_t_p863_protocol();
        generator.add_research_protocol();

        generator
    }

    /// Add ITU-T P.862 (PESQ) protocol
    fn add_itu_t_p862_protocol(&mut self) {
        let protocol = EvaluationProtocol {
            name: "ITU-T P.862 (PESQ)".to_string(),
            version: "2.0".to_string(),
            description: "Perceptual Evaluation of Speech Quality according to ITU-T P.862"
                .to_string(),
            metrics: vec![MetricSpecification {
                name: "PESQ".to_string(),
                description: "Perceptual Evaluation of Speech Quality score".to_string(),
                value_range: (-0.5, 4.5),
                threshold: 1.0,
                weight: 1.0,
                required: true,
                lower_is_better: false,
            }],
            audio_format: AudioFormatSpec {
                sample_rate: 8000,
                channels: 1,
                bit_depth: 16,
                formats: vec!["wav".to_string(), "pcm".to_string()],
            },
            datasets: vec![DatasetSpecification {
                name: "ITU-T P.862 Test Set".to_string(),
                sample_count: 100,
                languages: vec!["en".to_string()],
                speaker_demographics: HashMap::from([
                    ("gender".to_string(), "balanced".to_string()),
                    ("age".to_string(), "adult".to_string()),
                ]),
                content_categories: vec!["sentences".to_string(), "words".to_string()],
            }],
            procedures: vec![
                "Align reference and degraded signals".to_string(),
                "Apply psychoacoustic model".to_string(),
                "Calculate perceptual distance".to_string(),
                "Map to MOS-LQO scale".to_string(),
            ],
            compliance_requirements: HashMap::from([
                ("signal_alignment".to_string(), "required".to_string()),
                ("sample_rate".to_string(), "8kHz or 16kHz".to_string()),
                ("bit_depth".to_string(), "16-bit minimum".to_string()),
            ]),
        };

        self.protocols.insert("ITU-T-P862".to_string(), protocol);
    }

    /// Add ITU-T P.863 (POLQA) protocol
    fn add_itu_t_p863_protocol(&mut self) {
        let protocol = EvaluationProtocol {
            name: "ITU-T P.863 (POLQA)".to_string(),
            version: "3.0".to_string(),
            description:
                "Perceptual Objective Listening Quality Assessment according to ITU-T P.863"
                    .to_string(),
            metrics: vec![MetricSpecification {
                name: "POLQA".to_string(),
                description: "Perceptual Objective Listening Quality Assessment score".to_string(),
                value_range: (1.0, 5.0),
                threshold: 2.0,
                weight: 1.0,
                required: true,
                lower_is_better: false,
            }],
            audio_format: AudioFormatSpec {
                sample_rate: 48000,
                channels: 1,
                bit_depth: 16,
                formats: vec!["wav".to_string()],
            },
            datasets: vec![DatasetSpecification {
                name: "ITU-T P.863 Test Set".to_string(),
                sample_count: 200,
                languages: vec!["en".to_string(), "fr".to_string(), "de".to_string()],
                speaker_demographics: HashMap::from([
                    ("gender".to_string(), "balanced".to_string()),
                    ("age".to_string(), "18-65".to_string()),
                ]),
                content_categories: vec!["speech".to_string(), "music".to_string()],
            }],
            procedures: vec![
                "Temporal alignment of signals".to_string(),
                "Psychoacoustic modeling".to_string(),
                "Cognitive modeling".to_string(),
                "Quality prediction".to_string(),
            ],
            compliance_requirements: HashMap::from([
                ("bandwidth".to_string(), "super-wideband".to_string()),
                ("sample_rate".to_string(), "48kHz".to_string()),
                ("dynamic_range".to_string(), "minimum 16-bit".to_string()),
            ]),
        };

        self.protocols.insert("ITU-T-P863".to_string(), protocol);
    }

    /// Add research protocol for comprehensive evaluation
    fn add_research_protocol(&mut self) {
        let protocol = EvaluationProtocol {
            name: "VoiRS Research Protocol".to_string(),
            version: "1.0".to_string(),
            description: "Comprehensive speech synthesis evaluation protocol for research purposes"
                .to_string(),
            metrics: vec![
                MetricSpecification {
                    name: "PESQ".to_string(),
                    description: "Perceptual Evaluation of Speech Quality".to_string(),
                    value_range: (-0.5, 4.5),
                    threshold: 2.0,
                    weight: 0.3,
                    required: true,
                    lower_is_better: false,
                },
                MetricSpecification {
                    name: "STOI".to_string(),
                    description: "Short-Time Objective Intelligibility".to_string(),
                    value_range: (0.0, 1.0),
                    threshold: 0.7,
                    weight: 0.3,
                    required: true,
                    lower_is_better: false,
                },
                MetricSpecification {
                    name: "MCD".to_string(),
                    description: "Mel-Cepstral Distortion".to_string(),
                    value_range: (0.0, 20.0),
                    threshold: 6.0,
                    weight: 0.2,
                    required: true,
                    lower_is_better: true,
                },
                MetricSpecification {
                    name: "SI-SDR".to_string(),
                    description: "Scale-Invariant Signal-to-Distortion Ratio".to_string(),
                    value_range: (-20.0, 40.0),
                    threshold: 10.0,
                    weight: 0.2,
                    required: false,
                    lower_is_better: false,
                },
            ],
            audio_format: AudioFormatSpec {
                sample_rate: 22050,
                channels: 1,
                bit_depth: 16,
                formats: vec!["wav".to_string(), "flac".to_string()],
            },
            datasets: vec![
                DatasetSpecification {
                    name: "LJSpeech Subset".to_string(),
                    sample_count: 500,
                    languages: vec!["en".to_string()],
                    speaker_demographics: HashMap::from([
                        ("gender".to_string(), "female".to_string()),
                        ("age".to_string(), "adult".to_string()),
                    ]),
                    content_categories: vec!["audiobook".to_string()],
                },
                DatasetSpecification {
                    name: "VCTK Subset".to_string(),
                    sample_count: 1000,
                    languages: vec!["en".to_string()],
                    speaker_demographics: HashMap::from([
                        ("gender".to_string(), "balanced".to_string()),
                        ("age".to_string(), "various".to_string()),
                        ("accent".to_string(), "various".to_string()),
                    ]),
                    content_categories: vec!["newspaper".to_string()],
                },
            ],
            procedures: vec![
                "Audio preprocessing and validation".to_string(),
                "Reference-based quality evaluation".to_string(),
                "Intelligibility assessment".to_string(),
                "Spectral distance measurement".to_string(),
                "Statistical significance testing".to_string(),
                "Cross-validation with human ratings".to_string(),
            ],
            compliance_requirements: HashMap::from([
                ("preprocessing".to_string(), "standardized".to_string()),
                ("reference_alignment".to_string(), "required".to_string()),
                ("statistical_testing".to_string(), "p < 0.05".to_string()),
                (
                    "cross_validation".to_string(),
                    "human correlation > 0.8".to_string(),
                ),
            ]),
        };

        self.protocols
            .insert("VoiRS-Research".to_string(), protocol);
    }

    /// Get available protocols
    pub fn list_protocols(&self) -> Vec<String> {
        self.protocols.keys().cloned().collect()
    }

    /// Get protocol by name
    pub fn get_protocol(&self, name: &str) -> Option<&EvaluationProtocol> {
        self.protocols.get(name)
    }

    /// Add custom protocol
    pub fn add_protocol(
        &mut self,
        name: String,
        protocol: EvaluationProtocol,
    ) -> Result<(), ProtocolError> {
        self.validate_protocol(&protocol)?;
        self.protocols.insert(name, protocol);
        Ok(())
    }

    /// Validate protocol configuration
    pub fn validate_protocol(&self, protocol: &EvaluationProtocol) -> Result<(), ProtocolError> {
        if protocol.name.is_empty() {
            return Err(ProtocolError::InvalidConfiguration(
                "Protocol name cannot be empty".to_string(),
            ));
        }

        if protocol.metrics.is_empty() {
            return Err(ProtocolError::ValidationFailed(
                "Protocol must specify at least one metric".to_string(),
            ));
        }

        // Validate metric weights sum to reasonable value
        let total_weight: f64 = protocol.metrics.iter().map(|m| m.weight).sum();
        if total_weight <= 0.0 || total_weight > 2.0 {
            return Err(ProtocolError::ValidationFailed(
                "Total metric weights must be between 0 and 2".to_string(),
            ));
        }

        // Validate audio format
        if protocol.audio_format.sample_rate == 0 {
            return Err(ProtocolError::InvalidConfiguration(
                "Sample rate must be greater than 0".to_string(),
            ));
        }

        if protocol.audio_format.channels == 0 {
            return Err(ProtocolError::InvalidConfiguration(
                "Channel count must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Generate compliance report for given evaluation results
    pub fn generate_compliance_report(
        &self,
        protocol_name: &str,
        metric_results: &HashMap<String, f64>,
    ) -> Result<ComplianceReport, ProtocolError> {
        let protocol = self.get_protocol(protocol_name).ok_or_else(|| {
            ProtocolError::MissingComponent(format!("Protocol {} not found", protocol_name))
        })?;

        let mut metric_compliance = HashMap::new();
        let mut compliance_score = 0.0;
        let mut total_weight = 0.0;
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        for metric_spec in &protocol.metrics {
            if let Some(&result) = metric_results.get(&metric_spec.name) {
                let compliant = if metric_spec.lower_is_better {
                    result <= metric_spec.threshold
                } else {
                    result >= metric_spec.threshold
                };
                metric_compliance.insert(metric_spec.name.clone(), compliant);

                if compliant {
                    compliance_score += metric_spec.weight;
                } else {
                    let direction = if metric_spec.lower_is_better {
                        "above"
                    } else {
                        "below"
                    };
                    issues.push(format!(
                        "Metric {} ({:.3}) {} threshold ({:.3})",
                        metric_spec.name, result, direction, metric_spec.threshold
                    ));
                    let improvement = if metric_spec.lower_is_better {
                        "reduce"
                    } else {
                        "improve"
                    };
                    recommendations.push(format!(
                        "{} {} performance to meet threshold of {:.3}",
                        improvement.to_uppercase(),
                        metric_spec.name,
                        metric_spec.threshold
                    ));
                }
                total_weight += metric_spec.weight;
            } else if metric_spec.required {
                metric_compliance.insert(metric_spec.name.clone(), false);
                issues.push(format!("Required metric {} not provided", metric_spec.name));
                recommendations.push(format!("Implement {} metric evaluation", metric_spec.name));
                total_weight += metric_spec.weight;
            }
        }

        // Normalize compliance score
        if total_weight > 0.0 {
            compliance_score /= total_weight;
        }

        let compliance_level = match compliance_score {
            s if s >= 0.9 => ComplianceLevel::Full,
            s if s >= 0.7 => ComplianceLevel::Partial,
            _ => ComplianceLevel::NonCompliant,
        };

        Ok(ComplianceReport {
            protocol_name: protocol_name.to_string(),
            compliance_level,
            metric_compliance,
            compliance_score,
            issues,
            recommendations,
        })
    }

    /// Generate markdown documentation for a protocol
    pub fn generate_protocol_documentation(
        &self,
        protocol_name: &str,
    ) -> Result<String, ProtocolError> {
        let protocol = self.get_protocol(protocol_name).ok_or_else(|| {
            ProtocolError::MissingComponent(format!("Protocol {} not found", protocol_name))
        })?;

        let mut doc = String::new();

        // Header
        doc.push_str(&format!(
            "# {} - Version {}\n\n",
            protocol.name, protocol.version
        ));
        doc.push_str(&format!("{}\n\n", protocol.description));

        // Metrics
        doc.push_str("## Quality Metrics\n\n");
        for metric in &protocol.metrics {
            doc.push_str(&format!("### {}\n\n", metric.name));
            doc.push_str(&format!("- **Description**: {}\n", metric.description));
            doc.push_str(&format!(
                "- **Range**: {:.2} to {:.2}\n",
                metric.value_range.0, metric.value_range.1
            ));
            doc.push_str(&format!("- **Threshold**: {:.2}\n", metric.threshold));
            doc.push_str(&format!("- **Weight**: {:.2}\n", metric.weight));
            doc.push_str(&format!(
                "- **Required**: {}\n\n",
                if metric.required { "Yes" } else { "No" }
            ));
        }

        // Audio Format
        doc.push_str("## Audio Format Requirements\n\n");
        doc.push_str(&format!(
            "- **Sample Rate**: {} Hz\n",
            protocol.audio_format.sample_rate
        ));
        doc.push_str(&format!(
            "- **Channels**: {}\n",
            protocol.audio_format.channels
        ));
        doc.push_str(&format!(
            "- **Bit Depth**: {} bits\n",
            protocol.audio_format.bit_depth
        ));
        doc.push_str(&format!(
            "- **Supported Formats**: {}\n\n",
            protocol.audio_format.formats.join(", ")
        ));

        // Datasets
        doc.push_str("## Test Datasets\n\n");
        for dataset in &protocol.datasets {
            doc.push_str(&format!("### {}\n\n", dataset.name));
            doc.push_str(&format!("- **Sample Count**: {}\n", dataset.sample_count));
            doc.push_str(&format!(
                "- **Languages**: {}\n",
                dataset.languages.join(", ")
            ));
            doc.push_str(&format!(
                "- **Content Categories**: {}\n",
                dataset.content_categories.join(", ")
            ));

            if !dataset.speaker_demographics.is_empty() {
                doc.push_str("- **Speaker Demographics**:\n");
                for (key, value) in &dataset.speaker_demographics {
                    doc.push_str(&format!("  - {}: {}\n", key, value));
                }
            }
            doc.push('\n');
        }

        // Procedures
        doc.push_str("## Evaluation Procedures\n\n");
        for (i, procedure) in protocol.procedures.iter().enumerate() {
            doc.push_str(&format!("{}. {}\n", i + 1, procedure));
        }
        doc.push('\n');

        // Compliance Requirements
        doc.push_str("## Compliance Requirements\n\n");
        for (requirement, description) in &protocol.compliance_requirements {
            doc.push_str(&format!("- **{}**: {}\n", requirement, description));
        }

        Ok(doc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_generator_creation() {
        let generator = ProtocolDocumentationGenerator::new();
        let protocols = generator.list_protocols();

        assert!(protocols.contains(&"ITU-T-P862".to_string()));
        assert!(protocols.contains(&"ITU-T-P863".to_string()));
        assert!(protocols.contains(&"VoiRS-Research".to_string()));
    }

    #[test]
    fn test_protocol_validation() {
        let generator = ProtocolDocumentationGenerator::new();

        let valid_protocol = EvaluationProtocol {
            name: "Test Protocol".to_string(),
            version: "1.0".to_string(),
            description: "Test description".to_string(),
            metrics: vec![MetricSpecification {
                name: "Test Metric".to_string(),
                description: "Test metric description".to_string(),
                value_range: (0.0, 1.0),
                threshold: 0.5,
                weight: 1.0,
                required: true,
                lower_is_better: false,
            }],
            audio_format: AudioFormatSpec {
                sample_rate: 16000,
                channels: 1,
                bit_depth: 16,
                formats: vec!["wav".to_string()],
            },
            datasets: vec![],
            procedures: vec![],
            compliance_requirements: HashMap::new(),
        };

        assert!(generator.validate_protocol(&valid_protocol).is_ok());
    }

    #[test]
    fn test_compliance_report_generation() {
        let generator = ProtocolDocumentationGenerator::new();

        let mut metric_results = HashMap::new();
        metric_results.insert("PESQ".to_string(), 2.5);
        metric_results.insert("STOI".to_string(), 0.8);
        metric_results.insert("MCD".to_string(), 5.0);

        let report = generator
            .generate_compliance_report("VoiRS-Research", &metric_results)
            .expect("Should generate compliance report");

        assert_eq!(report.protocol_name, "VoiRS-Research");
        assert!(report.compliance_score > 0.0);
    }

    #[test]
    fn test_protocol_documentation_generation() {
        let generator = ProtocolDocumentationGenerator::new();

        let doc = generator
            .generate_protocol_documentation("ITU-T-P862")
            .expect("Should generate documentation");

        assert!(doc.contains("ITU-T P.862 (PESQ)"));
        assert!(doc.contains("Quality Metrics"));
        assert!(doc.contains("Audio Format Requirements"));
    }

    #[test]
    fn test_invalid_protocol_validation() {
        let generator = ProtocolDocumentationGenerator::new();

        let invalid_protocol = EvaluationProtocol {
            name: "".to_string(), // Empty name should fail
            version: "1.0".to_string(),
            description: "Test".to_string(),
            metrics: vec![],
            audio_format: AudioFormatSpec {
                sample_rate: 0, // Invalid sample rate
                channels: 1,
                bit_depth: 16,
                formats: vec![],
            },
            datasets: vec![],
            procedures: vec![],
            compliance_requirements: HashMap::new(),
        };

        assert!(generator.validate_protocol(&invalid_protocol).is_err());
    }

    #[test]
    fn test_compliance_levels() {
        let generator = ProtocolDocumentationGenerator::new();

        // High compliance score
        let mut high_results = HashMap::new();
        high_results.insert("PESQ".to_string(), 3.0);
        high_results.insert("STOI".to_string(), 0.9);
        high_results.insert("MCD".to_string(), 4.0);

        let high_report = generator
            .generate_compliance_report("VoiRS-Research", &high_results)
            .expect("Should generate report");
        assert_eq!(high_report.compliance_level, ComplianceLevel::Full);

        // Low compliance score
        let mut low_results = HashMap::new();
        low_results.insert("PESQ".to_string(), 1.0);
        low_results.insert("STOI".to_string(), 0.3);
        low_results.insert("MCD".to_string(), 10.0);

        let low_report = generator
            .generate_compliance_report("VoiRS-Research", &low_results)
            .expect("Should generate report");
        assert_eq!(low_report.compliance_level, ComplianceLevel::NonCompliant);
    }
}
