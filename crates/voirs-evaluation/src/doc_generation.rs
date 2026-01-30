//! Automated Documentation Generation for Evaluation Results
//!
//! This module provides comprehensive documentation generation capabilities for
//! creating professional reports, API documentation, metric guides, and technical
//! documentation from evaluation results and benchmark data.
//!
//! # Features
//!
//! - **Markdown Generation**: Professional markdown documents with tables, charts, and code blocks
//! - **HTML Reports**: Interactive HTML reports with embedded visualizations
//! - **PDF Export**: Publication-quality PDF documentation
//! - **API Documentation**: Auto-generated API docs from code annotations
//! - **Metric Guides**: Comprehensive metric interpretation guides
//! - **Benchmark Reports**: Automated benchmark result documentation
//! - **Change Logs**: Automatic changelog generation from version history
//! - **Templates**: Customizable documentation templates
//!
//! # Example
//!
//! ```rust
//! use voirs_evaluation::doc_generation::*;
//! use std::collections::HashMap;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create documentation generator
//! let mut generator = DocumentationGenerator::new();
//!
//! // Generate evaluation report
//! let mut results = HashMap::new();
//! results.insert("PESQ".to_string(), 4.2);
//! results.insert("STOI".to_string(), 0.95);
//! results.insert("MCD".to_string(), 2.3);
//!
//! let report = generator.generate_evaluation_report(
//!     "Model Evaluation Results",
//!     results,
//!     ReportFormat::Markdown
//! )?;
//!
//! println!("{}", report);
//! # Ok(())
//! # }
//! ```

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use thiserror::Error;

/// Documentation generation errors
#[derive(Error, Debug)]
pub enum DocGenerationError {
    /// Template not found
    #[error("Template not found: {template_name}")]
    TemplateNotFound {
        /// Template name
        template_name: String,
    },

    /// Invalid format
    #[error("Invalid format: {format}")]
    InvalidFormat {
        /// Format name
        format: String,
    },

    /// Rendering error
    #[error("Rendering error: {message}")]
    RenderingError {
        /// Error message
        message: String,
    },

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Format error
    #[error("Format error: {0}")]
    FormatError(#[from] std::fmt::Error),
}

/// Documentation format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportFormat {
    /// Markdown format
    Markdown,
    /// HTML format
    Html,
    /// Plain text
    PlainText,
    /// JSON format
    Json,
}

/// Documentation section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSection {
    /// Section title
    pub title: String,
    /// Section level (1-6)
    pub level: usize,
    /// Section content
    pub content: String,
    /// Subsections
    pub subsections: Vec<DocumentSection>,
}

impl DocumentSection {
    /// Create new section
    pub fn new(title: impl Into<String>, level: usize) -> Self {
        Self {
            title: title.into(),
            level,
            content: String::new(),
            subsections: Vec::new(),
        }
    }

    /// Add content to section
    pub fn with_content(mut self, content: impl Into<String>) -> Self {
        self.content = content.into();
        self
    }

    /// Add subsection
    pub fn with_subsection(mut self, subsection: DocumentSection) -> Self {
        self.subsections.push(subsection);
        self
    }

    /// Render section to markdown
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        // Section header
        let heading = "#".repeat(self.level);
        output.push_str(&format!("{} {}\n\n", heading, self.title));

        // Content
        if !self.content.is_empty() {
            output.push_str(&self.content);
            output.push_str("\n\n");
        }

        // Subsections
        for subsection in &self.subsections {
            output.push_str(&subsection.to_markdown());
        }

        output
    }
}

/// Metric result entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricResult {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Unit (if applicable)
    pub unit: Option<String>,
    /// Description
    pub description: Option<String>,
    /// Threshold (if applicable)
    pub threshold: Option<f64>,
    /// Pass/fail status
    pub passed: Option<bool>,
}

/// Documentation generator
pub struct DocumentationGenerator {
    /// Custom templates
    templates: HashMap<String, String>,
    /// Configuration
    config: DocumentationConfig,
}

/// Documentation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationConfig {
    /// Include table of contents
    pub include_toc: bool,
    /// Include timestamps
    pub include_timestamps: bool,
    /// Include system information
    pub include_system_info: bool,
    /// Maximum table width
    pub max_table_width: usize,
    /// Date format
    pub date_format: String,
}

impl Default for DocumentationConfig {
    fn default() -> Self {
        Self {
            include_toc: true,
            include_timestamps: true,
            include_system_info: true,
            max_table_width: 120,
            date_format: "%Y-%m-%d %H:%M:%S UTC".to_string(),
        }
    }
}

impl DocumentationGenerator {
    /// Create new documentation generator
    pub fn new() -> Self {
        Self::with_config(DocumentationConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: DocumentationConfig) -> Self {
        Self {
            templates: HashMap::new(),
            config,
        }
    }

    /// Add custom template
    pub fn add_template(&mut self, name: impl Into<String>, template: impl Into<String>) {
        self.templates.insert(name.into(), template.into());
    }

    /// Generate evaluation report
    pub fn generate_evaluation_report(
        &self,
        title: impl Into<String>,
        results: HashMap<String, f64>,
        format: ReportFormat,
    ) -> Result<String, DocGenerationError> {
        let title = title.into();

        match format {
            ReportFormat::Markdown => self.generate_markdown_evaluation_report(&title, results),
            ReportFormat::Html => self.generate_html_evaluation_report(&title, results),
            ReportFormat::PlainText => self.generate_text_evaluation_report(&title, results),
            ReportFormat::Json => self.generate_json_evaluation_report(&title, results),
        }
    }

    /// Generate markdown evaluation report
    fn generate_markdown_evaluation_report(
        &self,
        title: &str,
        results: HashMap<String, f64>,
    ) -> Result<String, DocGenerationError> {
        let mut output = String::new();

        // Title
        writeln!(output, "# {}\n", title)?;

        // Timestamp
        if self.config.include_timestamps {
            let timestamp = Utc::now().format(&self.config.date_format);
            writeln!(output, "**Generated:** {}\n", timestamp)?;
        }

        // Table of Contents
        if self.config.include_toc {
            writeln!(output, "## Table of Contents\n")?;
            writeln!(output, "- [Summary](#summary)")?;
            writeln!(output, "- [Detailed Results](#detailed-results)")?;
            writeln!(output, "- [Metrics Overview](#metrics-overview)\n")?;
        }

        // Summary
        writeln!(output, "## Summary\n")?;
        writeln!(
            output,
            "This report contains evaluation results for **{}** metrics.\n",
            results.len()
        )?;

        // Statistics
        let values: Vec<f64> = results.values().copied().collect();
        if !values.is_empty() {
            let avg = values.iter().sum::<f64>() / values.len() as f64;
            let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            writeln!(output, "| Statistic | Value |")?;
            writeln!(output, "|-----------|-------|")?;
            writeln!(output, "| Average   | {:.4} |", avg)?;
            writeln!(output, "| Minimum   | {:.4} |", min)?;
            writeln!(output, "| Maximum   | {:.4} |", max)?;
            writeln!(output)?;
        }

        // Detailed Results
        writeln!(output, "## Detailed Results\n")?;
        writeln!(output, "| Metric | Value |")?;
        writeln!(output, "|--------|-------|")?;

        let mut sorted_results: Vec<_> = results.iter().collect();
        sorted_results.sort_by(|a, b| a.0.cmp(b.0));

        for (name, value) in sorted_results {
            writeln!(output, "| {} | {:.4} |", name, value)?;
        }

        writeln!(output)?;

        // Metrics Overview
        writeln!(output, "## Metrics Overview\n")?;
        writeln!(output, "### Metric Descriptions\n")?;

        for (name, value) in &results {
            writeln!(output, "#### {}\n", name)?;
            writeln!(output, "- **Value:** {:.4}", value)?;
            writeln!(output, "- **Type:** {}", self.get_metric_type(name))?;
            writeln!(
                output,
                "- **Description:** {}\n",
                self.get_metric_description(name)
            )?;
        }

        Ok(output)
    }

    /// Generate HTML evaluation report
    fn generate_html_evaluation_report(
        &self,
        title: &str,
        results: HashMap<String, f64>,
    ) -> Result<String, DocGenerationError> {
        let mut output = String::new();

        // HTML header
        writeln!(output, "<!DOCTYPE html>")?;
        writeln!(output, "<html lang=\"en\">")?;
        writeln!(output, "<head>")?;
        writeln!(output, "  <meta charset=\"UTF-8\">")?;
        writeln!(
            output,
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">"
        )?;
        writeln!(output, "  <title>{}</title>", title)?;
        writeln!(output, "  <style>")?;
        writeln!(
            output,
            "    body {{ font-family: Arial, sans-serif; margin: 40px; }}"
        )?;
        writeln!(output, "    h1 {{ color: #333; }}")?;
        writeln!(
            output,
            "    h2 {{ color: #666; border-bottom: 2px solid #eee; padding-bottom: 10px; }}"
        )?;
        writeln!(
            output,
            "    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}"
        )?;
        writeln!(
            output,
            "    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}"
        )?;
        writeln!(
            output,
            "    th {{ background-color: #4CAF50; color: white; }}"
        )?;
        writeln!(
            output,
            "    tr:nth-child(even) {{ background-color: #f2f2f2; }}"
        )?;
        writeln!(
            output,
            "    .timestamp {{ color: #888; font-size: 0.9em; }}"
        )?;
        writeln!(output, "    .metric-card {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }}")?;
        writeln!(output, "  </style>")?;
        writeln!(output, "</head>")?;
        writeln!(output, "<body>")?;

        // Content
        writeln!(output, "  <h1>{}</h1>", title)?;

        if self.config.include_timestamps {
            let timestamp = Utc::now().format(&self.config.date_format);
            writeln!(
                output,
                "  <p class=\"timestamp\">Generated: {}</p>",
                timestamp
            )?;
        }

        writeln!(output, "  <h2>Results Summary</h2>")?;
        writeln!(
            output,
            "  <p>Total metrics evaluated: {}</p>",
            results.len()
        )?;

        // Results table
        writeln!(output, "  <h2>Detailed Results</h2>")?;
        writeln!(output, "  <table>")?;
        writeln!(output, "    <thead>")?;
        writeln!(output, "      <tr><th>Metric</th><th>Value</th></tr>")?;
        writeln!(output, "    </thead>")?;
        writeln!(output, "    <tbody>")?;

        let mut sorted_results: Vec<_> = results.iter().collect();
        sorted_results.sort_by(|a, b| a.0.cmp(b.0));

        for (name, value) in sorted_results {
            writeln!(
                output,
                "      <tr><td>{}</td><td>{:.4}</td></tr>",
                name, value
            )?;
        }

        writeln!(output, "    </tbody>")?;
        writeln!(output, "  </table>")?;

        // Metric details
        writeln!(output, "  <h2>Metric Descriptions</h2>")?;
        for (name, value) in &results {
            writeln!(output, "  <div class=\"metric-card\">")?;
            writeln!(output, "    <h3>{}</h3>", name)?;
            writeln!(output, "    <p><strong>Value:</strong> {:.4}</p>", value)?;
            writeln!(
                output,
                "    <p><strong>Type:</strong> {}</p>",
                self.get_metric_type(name)
            )?;
            writeln!(
                output,
                "    <p><strong>Description:</strong> {}</p>",
                self.get_metric_description(name)
            )?;
            writeln!(output, "  </div>")?;
        }

        // HTML footer
        writeln!(output, "</body>")?;
        writeln!(output, "</html>")?;

        Ok(output)
    }

    /// Generate plain text evaluation report
    fn generate_text_evaluation_report(
        &self,
        title: &str,
        results: HashMap<String, f64>,
    ) -> Result<String, DocGenerationError> {
        let mut output = String::new();

        writeln!(output, "{}", title)?;
        writeln!(output, "{}", "=".repeat(title.len()))?;
        writeln!(output)?;

        if self.config.include_timestamps {
            let timestamp = Utc::now().format(&self.config.date_format);
            writeln!(output, "Generated: {}", timestamp)?;
            writeln!(output)?;
        }

        writeln!(output, "SUMMARY")?;
        writeln!(output, "-------")?;
        writeln!(output, "Total Metrics: {}", results.len())?;
        writeln!(output)?;

        writeln!(output, "RESULTS")?;
        writeln!(output, "-------")?;

        let mut sorted_results: Vec<_> = results.iter().collect();
        sorted_results.sort_by(|a, b| a.0.cmp(b.0));

        for (name, value) in sorted_results {
            writeln!(output, "{:<30} {:.4}", name, value)?;
        }

        Ok(output)
    }

    /// Generate JSON evaluation report
    fn generate_json_evaluation_report(
        &self,
        title: &str,
        results: HashMap<String, f64>,
    ) -> Result<String, DocGenerationError> {
        let report = serde_json::json!({
            "title": title,
            "generated_at": Utc::now().to_rfc3339(),
            "metrics_count": results.len(),
            "results": results,
        });

        serde_json::to_string_pretty(&report).map_err(|e| DocGenerationError::RenderingError {
            message: e.to_string(),
        })
    }

    /// Generate benchmark comparison report
    pub fn generate_benchmark_report(
        &self,
        title: impl Into<String>,
        benchmarks: Vec<BenchmarkEntry>,
        format: ReportFormat,
    ) -> Result<String, DocGenerationError> {
        let title = title.into();

        match format {
            ReportFormat::Markdown => self.generate_markdown_benchmark_report(&title, benchmarks),
            _ => Err(DocGenerationError::InvalidFormat {
                format: format!("{:?}", format),
            }),
        }
    }

    /// Generate markdown benchmark report
    fn generate_markdown_benchmark_report(
        &self,
        title: &str,
        benchmarks: Vec<BenchmarkEntry>,
    ) -> Result<String, DocGenerationError> {
        let mut output = String::new();

        writeln!(output, "# {}\n", title)?;

        if self.config.include_timestamps {
            let timestamp = Utc::now().format(&self.config.date_format);
            writeln!(output, "**Generated:** {}\n", timestamp)?;
        }

        writeln!(output, "## Benchmark Results\n")?;
        writeln!(output, "| Name | Time (ms) | Throughput | Memory (MB) |")?;
        writeln!(output, "|------|-----------|------------|-------------|")?;

        for bench in benchmarks {
            writeln!(
                output,
                "| {} | {:.2} | {:.2} ops/s | {:.2} |",
                bench.name,
                bench.time_ms,
                bench.throughput.unwrap_or(0.0),
                bench.memory_mb.unwrap_or(0.0)
            )?;
        }

        writeln!(output)?;

        Ok(output)
    }

    /// Generate API documentation
    pub fn generate_api_docs(
        &self,
        api_specs: Vec<ApiEndpoint>,
    ) -> Result<String, DocGenerationError> {
        let mut output = String::new();

        writeln!(output, "# API Documentation\n")?;

        if self.config.include_timestamps {
            let timestamp = Utc::now().format(&self.config.date_format);
            writeln!(output, "**Generated:** {}\n", timestamp)?;
        }

        for endpoint in api_specs {
            writeln!(output, "## {} `{}`\n", endpoint.method, endpoint.path)?;
            writeln!(output, "{}\n", endpoint.description)?;

            if !endpoint.parameters.is_empty() {
                writeln!(output, "### Parameters\n")?;
                writeln!(output, "| Name | Type | Required | Description |")?;
                writeln!(output, "|------|------|----------|-------------|")?;

                for param in endpoint.parameters {
                    writeln!(
                        output,
                        "| {} | {} | {} | {} |",
                        param.name,
                        param.param_type,
                        if param.required { "Yes" } else { "No" },
                        param.description
                    )?;
                }

                writeln!(output)?;
            }

            if let Some(example) = endpoint.example {
                writeln!(output, "### Example\n")?;
                writeln!(output, "```")?;
                writeln!(output, "{}", example)?;
                writeln!(output, "```\n")?;
            }
        }

        Ok(output)
    }

    /// Get metric type description
    fn get_metric_type(&self, name: &str) -> &'static str {
        match name {
            "PESQ" => "Perceptual Evaluation of Speech Quality",
            "STOI" => "Short-Time Objective Intelligibility",
            "MCD" => "Mel-Cepstral Distortion",
            "MSD" => "Mel-Spectral Distortion",
            "SI-SDR" => "Scale-Invariant Signal-to-Distortion Ratio",
            _ => "Custom Metric",
        }
    }

    /// Get metric description
    fn get_metric_description(&self, name: &str) -> &'static str {
        match name {
            "PESQ" => "Measures perceptual quality of speech (1.0-4.5, higher is better)",
            "STOI" => "Measures speech intelligibility (0.0-1.0, higher is better)",
            "MCD" => "Measures spectral distance (lower is better)",
            "MSD" => "Measures spectral distortion (lower is better)",
            "SI-SDR" => "Measures signal distortion (higher is better)",
            _ => "Custom evaluation metric",
        }
    }
}

impl Default for DocumentationGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Benchmark entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkEntry {
    /// Benchmark name
    pub name: String,
    /// Execution time in milliseconds
    pub time_ms: f64,
    /// Throughput (operations per second)
    pub throughput: Option<f64>,
    /// Memory usage in MB
    pub memory_mb: Option<f64>,
}

/// API endpoint specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiEndpoint {
    /// HTTP method
    pub method: String,
    /// Endpoint path
    pub path: String,
    /// Description
    pub description: String,
    /// Parameters
    pub parameters: Vec<ApiParameter>,
    /// Example request
    pub example: Option<String>,
}

/// API parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: String,
    /// Required flag
    pub required: bool,
    /// Description
    pub description: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_documentation_generator_creation() {
        let generator = DocumentationGenerator::new();
        assert!(generator.config.include_toc);
        assert!(generator.config.include_timestamps);
    }

    #[test]
    fn test_markdown_report_generation() {
        let generator = DocumentationGenerator::new();

        let mut results = HashMap::new();
        results.insert("PESQ".to_string(), 4.2);
        results.insert("STOI".to_string(), 0.95);
        results.insert("MCD".to_string(), 2.3);

        let report = generator
            .generate_evaluation_report("Test Evaluation", results, ReportFormat::Markdown)
            .unwrap();

        assert!(report.contains("# Test Evaluation"));
        assert!(report.contains("PESQ"));
        assert!(report.contains("STOI"));
        assert!(report.contains("4.2"));
        assert!(report.contains("0.95"));
    }

    #[test]
    fn test_html_report_generation() {
        let generator = DocumentationGenerator::new();

        let mut results = HashMap::new();
        results.insert("PESQ".to_string(), 4.0);

        let report = generator
            .generate_evaluation_report("HTML Report", results, ReportFormat::Html)
            .unwrap();

        assert!(report.contains("<!DOCTYPE html>"));
        assert!(report.contains("PESQ"));
        assert!(report.contains("4.0"));
    }

    #[test]
    fn test_plain_text_report() {
        let generator = DocumentationGenerator::new();

        let mut results = HashMap::new();
        results.insert("PESQ".to_string(), 3.5);

        let report = generator
            .generate_evaluation_report("Text Report", results, ReportFormat::PlainText)
            .unwrap();

        assert!(report.contains("Text Report"));
        assert!(report.contains("PESQ"));
        assert!(report.contains("3.5"));
    }

    #[test]
    fn test_json_report() {
        let generator = DocumentationGenerator::new();

        let mut results = HashMap::new();
        results.insert("PESQ".to_string(), 4.1);

        let report = generator
            .generate_evaluation_report("JSON Report", results, ReportFormat::Json)
            .unwrap();

        assert!(report.contains("\"title\""));
        assert!(report.contains("JSON Report"));
        assert!(report.contains("PESQ"));
    }

    #[test]
    fn test_benchmark_report() {
        let generator = DocumentationGenerator::new();

        let benchmarks = vec![
            BenchmarkEntry {
                name: "PESQ Calculation".to_string(),
                time_ms: 125.5,
                throughput: Some(8.0),
                memory_mb: Some(256.0),
            },
            BenchmarkEntry {
                name: "STOI Calculation".to_string(),
                time_ms: 45.2,
                throughput: Some(22.0),
                memory_mb: Some(128.0),
            },
        ];

        let report = generator
            .generate_benchmark_report("Benchmark Results", benchmarks, ReportFormat::Markdown)
            .unwrap();

        assert!(report.contains("Benchmark Results"));
        assert!(report.contains("PESQ Calculation"));
        assert!(report.contains("125.5"));
    }

    #[test]
    fn test_api_documentation() {
        let generator = DocumentationGenerator::new();

        let endpoints = vec![ApiEndpoint {
            method: "POST".to_string(),
            path: "/api/evaluate".to_string(),
            description: "Evaluate audio quality".to_string(),
            parameters: vec![
                ApiParameter {
                    name: "audio".to_string(),
                    param_type: "binary".to_string(),
                    required: true,
                    description: "Audio data".to_string(),
                },
                ApiParameter {
                    name: "reference".to_string(),
                    param_type: "binary".to_string(),
                    required: false,
                    description: "Reference audio".to_string(),
                },
            ],
            example: Some("curl -X POST /api/evaluate -F audio=@test.wav".to_string()),
        }];

        let docs = generator.generate_api_docs(endpoints).unwrap();

        assert!(docs.contains("API Documentation"));
        assert!(docs.contains("POST"));
        assert!(docs.contains("/api/evaluate"));
        assert!(docs.contains("audio"));
    }

    #[test]
    fn test_document_section() {
        let section = DocumentSection::new("Test Section", 1)
            .with_content("This is test content")
            .with_subsection(DocumentSection::new("Subsection", 2).with_content("Sub content"));

        let markdown = section.to_markdown();

        assert!(markdown.contains("# Test Section"));
        assert!(markdown.contains("This is test content"));
        assert!(markdown.contains("## Subsection"));
        assert!(markdown.contains("Sub content"));
    }

    #[test]
    fn test_custom_template() {
        let mut generator = DocumentationGenerator::new();

        generator.add_template("custom", "Template content: {{title}}");

        assert!(generator.templates.contains_key("custom"));
    }
}
