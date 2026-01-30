//! Benchmark result export and import functionality
//!
//! This module provides utilities for exporting and importing benchmark results
//! in various formats (JSON, CSV, Markdown) for use in CI/CD pipelines, documentation,
//! and performance tracking.

use crate::EvaluationError;
use serde::{Deserialize, Serialize};

/// Performance benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Mean execution time (ms)
    pub mean: f64,
    /// Standard deviation (ms)
    pub std_dev: f64,
    /// Minimum time (ms)
    pub min: f64,
    /// Maximum time (ms)
    pub max: f64,
    /// Number of samples
    pub samples: usize,
}
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Export format for benchmark results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format (machine-readable)
    Json,
    /// CSV format (spreadsheet compatible)
    Csv,
    /// Markdown format (human-readable, documentation-ready)
    Markdown,
    /// HTML format (web-friendly)
    Html,
}

/// Benchmark export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Output format
    pub format: ExportFormat,
    /// Include historical comparison data
    pub include_history: bool,
    /// Include statistical summaries
    pub include_stats: bool,
    /// Sort results by this field
    pub sort_by: Option<String>,
    /// Maximum number of results to export
    pub limit: Option<usize>,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Json,
            include_history: true,
            include_stats: true,
            sort_by: Some("name".to_string()),
            limit: None,
        }
    }
}

/// Benchmark results exporter
pub struct BenchmarkExporter {
    config: ExportConfig,
}

impl BenchmarkExporter {
    /// Create a new exporter with the specified configuration
    #[must_use]
    pub fn new(config: ExportConfig) -> Self {
        Self { config }
    }

    /// Create a new exporter with default configuration
    #[must_use]
    pub fn default() -> Self {
        Self {
            config: ExportConfig::default(),
        }
    }

    /// Export benchmark results to a file
    ///
    /// # Errors
    /// Returns an error if the file cannot be written or the data cannot be serialized
    pub fn export_to_file<P: AsRef<Path>>(
        &self,
        results: &[PerformanceBenchmarkResult],
        path: P,
    ) -> Result<(), EvaluationError> {
        let content = self.export_to_string(results)?;
        fs::write(path.as_ref(), content)
            .map_err(|e| EvaluationError::Io(format!("Failed to write export file: {}", e)))?;
        Ok(())
    }

    /// Export benchmark results to a string
    ///
    /// # Errors
    /// Returns an error if the data cannot be serialized
    pub fn export_to_string(
        &self,
        results: &[PerformanceBenchmarkResult],
    ) -> Result<String, EvaluationError> {
        let mut results = results.to_vec();

        // Apply sorting if configured
        if let Some(sort_field) = &self.config.sort_by {
            results.sort_by(|a, b| match sort_field.as_str() {
                "name" => a.name.cmp(&b.name),
                "mean" => a
                    .mean
                    .partial_cmp(&b.mean)
                    .unwrap_or(std::cmp::Ordering::Equal),
                "std_dev" => a
                    .std_dev
                    .partial_cmp(&b.std_dev)
                    .unwrap_or(std::cmp::Ordering::Equal),
                _ => std::cmp::Ordering::Equal,
            });
        }

        // Apply limit if configured
        if let Some(limit) = self.config.limit {
            results.truncate(limit);
        }

        match self.config.format {
            ExportFormat::Json => self.export_json(&results),
            ExportFormat::Csv => self.export_csv(&results),
            ExportFormat::Markdown => self.export_markdown(&results),
            ExportFormat::Html => self.export_html(&results),
        }
    }

    fn export_json(
        &self,
        results: &[PerformanceBenchmarkResult],
    ) -> Result<String, EvaluationError> {
        serde_json::to_string_pretty(results)
            .map_err(|e| EvaluationError::Other(format!("JSON serialization failed: {}", e)))
    }

    fn export_csv(
        &self,
        results: &[PerformanceBenchmarkResult],
    ) -> Result<String, EvaluationError> {
        let mut csv = String::from("Name,Mean (ms),Std Dev (ms),Min (ms),Max (ms),Samples\n");

        for result in results {
            csv.push_str(&format!(
                "{},{:.3},{:.3},{:.3},{:.3},{}\n",
                result.name, result.mean, result.std_dev, result.min, result.max, result.samples
            ));
        }

        Ok(csv)
    }

    fn export_markdown(
        &self,
        results: &[PerformanceBenchmarkResult],
    ) -> Result<String, EvaluationError> {
        let mut md = String::from("# Benchmark Results\n\n");
        md.push_str("| Name | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | Samples |\n");
        md.push_str("|------|-----------|--------------|----------|----------|----------|\n");

        for result in results {
            md.push_str(&format!(
                "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {} |\n",
                result.name, result.mean, result.std_dev, result.min, result.max, result.samples
            ));
        }

        if self.config.include_stats {
            md.push_str("\n## Summary Statistics\n\n");
            let total_mean: f64 =
                results.iter().map(|r| r.mean).sum::<f64>() / results.len() as f64;
            let total_std: f64 =
                results.iter().map(|r| r.std_dev).sum::<f64>() / results.len() as f64;
            md.push_str(&format!("- **Average Mean**: {:.3} ms\n", total_mean));
            md.push_str(&format!("- **Average Std Dev**: {:.3} ms\n", total_std));
            md.push_str(&format!("- **Total Benchmarks**: {}\n", results.len()));
        }

        Ok(md)
    }

    fn export_html(
        &self,
        results: &[PerformanceBenchmarkResult],
    ) -> Result<String, EvaluationError> {
        let mut html = String::from("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<meta charset=\"utf-8\">\n");
        html.push_str("<title>Benchmark Results</title>\n");
        html.push_str("<style>\n");
        html.push_str("body { font-family: Arial, sans-serif; margin: 20px; }\n");
        html.push_str("table { border-collapse: collapse; width: 100%; }\n");
        html.push_str("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n");
        html.push_str("th { background-color: #4CAF50; color: white; }\n");
        html.push_str("tr:nth-child(even) { background-color: #f2f2f2; }\n");
        html.push_str("</style>\n</head>\n<body>\n");
        html.push_str("<h1>Benchmark Results</h1>\n");
        html.push_str("<table>\n<tr>\n");
        html.push_str("<th>Name</th><th>Mean (ms)</th><th>Std Dev (ms)</th>");
        html.push_str("<th>Min (ms)</th><th>Max (ms)</th><th>Samples</th>\n</tr>\n");

        for result in results {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{}</td></tr>\n",
                result.name, result.mean, result.std_dev, result.min, result.max, result.samples
            ));
        }

        html.push_str("</table>\n");

        if self.config.include_stats {
            html.push_str("<h2>Summary Statistics</h2>\n");
            let total_mean: f64 =
                results.iter().map(|r| r.mean).sum::<f64>() / results.len() as f64;
            let total_std: f64 =
                results.iter().map(|r| r.std_dev).sum::<f64>() / results.len() as f64;
            html.push_str("<ul>\n");
            html.push_str(&format!(
                "<li><strong>Average Mean:</strong> {:.3} ms</li>\n",
                total_mean
            ));
            html.push_str(&format!(
                "<li><strong>Average Std Dev:</strong> {:.3} ms</li>\n",
                total_std
            ));
            html.push_str(&format!(
                "<li><strong>Total Benchmarks:</strong> {}</li>\n",
                results.len()
            ));
            html.push_str("</ul>\n");
        }

        html.push_str("</body>\n</html>");
        Ok(html)
    }

    /// Import benchmark results from a JSON file
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or parsed
    pub fn import_from_file<P: AsRef<Path>>(
        path: P,
    ) -> Result<Vec<PerformanceBenchmarkResult>, EvaluationError> {
        let content = fs::read_to_string(path.as_ref())
            .map_err(|e| EvaluationError::Io(format!("Failed to read import file: {}", e)))?;

        serde_json::from_str(&content)
            .map_err(|e| EvaluationError::Other(format!("JSON deserialization failed: {}", e)))
    }

    /// Compare two sets of benchmark results
    ///
    /// Returns a map of benchmark names to percentage changes (positive = improvement)
    #[must_use]
    pub fn compare_results(
        baseline: &[PerformanceBenchmarkResult],
        current: &[PerformanceBenchmarkResult],
    ) -> HashMap<String, f64> {
        let baseline_map: HashMap<&str, f64> =
            baseline.iter().map(|r| (r.name.as_str(), r.mean)).collect();

        let mut comparison = HashMap::new();

        for result in current {
            if let Some(&baseline_mean) = baseline_map.get(result.name.as_str()) {
                if baseline_mean > 0.0 {
                    // Negative change = improvement (faster)
                    let change = ((result.mean - baseline_mean) / baseline_mean) * 100.0;
                    comparison.insert(result.name.clone(), -change); // Invert so positive = better
                }
            }
        }

        comparison
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    fn create_test_results() -> Vec<PerformanceBenchmarkResult> {
        vec![
            PerformanceBenchmarkResult {
                name: "test1".to_string(),
                mean: 10.5,
                std_dev: 0.5,
                min: 10.0,
                max: 11.0,
                samples: 100,
            },
            PerformanceBenchmarkResult {
                name: "test2".to_string(),
                mean: 20.3,
                std_dev: 1.2,
                min: 18.0,
                max: 22.0,
                samples: 100,
            },
        ]
    }

    #[test]
    fn test_export_json() {
        let exporter = BenchmarkExporter::default();
        let results = create_test_results();
        let json = exporter.export_to_string(&results).unwrap();
        assert!(json.contains("test1"));
        assert!(json.contains("10.5"));
    }

    #[test]
    fn test_export_csv() {
        let config = ExportConfig {
            format: ExportFormat::Csv,
            ..Default::default()
        };
        let exporter = BenchmarkExporter::new(config);
        let results = create_test_results();
        let csv = exporter.export_to_string(&results).unwrap();
        assert!(csv.contains("Name,Mean"));
        assert!(csv.contains("test1,10.500"));
    }

    #[test]
    fn test_export_markdown() {
        let config = ExportConfig {
            format: ExportFormat::Markdown,
            ..Default::default()
        };
        let exporter = BenchmarkExporter::new(config);
        let results = create_test_results();
        let md = exporter.export_to_string(&results).unwrap();
        assert!(md.contains("# Benchmark Results"));
        assert!(md.contains("| test1 |"));
    }

    #[test]
    fn test_export_html() {
        let config = ExportConfig {
            format: ExportFormat::Html,
            ..Default::default()
        };
        let exporter = BenchmarkExporter::new(config);
        let results = create_test_results();
        let html = exporter.export_to_string(&results).unwrap();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<td>test1</td>"));
    }

    #[test]
    fn test_export_import_roundtrip() {
        let temp_path = temp_dir().join("test_benchmark_export.json");
        let exporter = BenchmarkExporter::default();
        let results = create_test_results();

        // Export
        exporter.export_to_file(&results, &temp_path).unwrap();

        // Import
        let imported = BenchmarkExporter::import_from_file(&temp_path).unwrap();

        assert_eq!(results.len(), imported.len());
        assert_eq!(results[0].name, imported[0].name);
        assert!((results[0].mean - imported[0].mean).abs() < 0.001);

        // Cleanup
        let _ = fs::remove_file(&temp_path);
    }

    #[test]
    fn test_compare_results() {
        let baseline = vec![PerformanceBenchmarkResult {
            name: "test1".to_string(),
            mean: 10.0,
            std_dev: 0.5,
            min: 9.5,
            max: 10.5,
            samples: 100,
        }];

        let current = vec![PerformanceBenchmarkResult {
            name: "test1".to_string(),
            mean: 9.0, // 10% faster
            std_dev: 0.5,
            min: 8.5,
            max: 9.5,
            samples: 100,
        }];

        let comparison = BenchmarkExporter::compare_results(&baseline, &current);
        let change = comparison.get("test1").unwrap();
        assert!((change - 10.0).abs() < 0.1); // Should be ~10% improvement
    }

    #[test]
    fn test_sort_by_mean() {
        let config = ExportConfig {
            sort_by: Some("mean".to_string()),
            ..Default::default()
        };
        let exporter = BenchmarkExporter::new(config);
        let results = vec![
            PerformanceBenchmarkResult {
                name: "slow".to_string(),
                mean: 20.0,
                std_dev: 1.0,
                min: 19.0,
                max: 21.0,
                samples: 100,
            },
            PerformanceBenchmarkResult {
                name: "fast".to_string(),
                mean: 10.0,
                std_dev: 0.5,
                min: 9.5,
                max: 10.5,
                samples: 100,
            },
        ];

        let exported = exporter.export_to_string(&results).unwrap();
        let fast_pos = exported.find("fast").unwrap();
        let slow_pos = exported.find("slow").unwrap();
        assert!(fast_pos < slow_pos); // fast should come first when sorted by mean
    }
}
