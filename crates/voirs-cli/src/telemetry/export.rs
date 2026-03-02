//! Telemetry data export functionality

use std::path::Path;
use std::sync::Arc;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::RwLock;

use super::{config::TelemetryConfig, events::TelemetryEvent, TelemetryError};

/// Export format for telemetry data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// JSON format
    Json,

    /// JSON Lines format (one event per line)
    JsonLines,

    /// CSV format
    Csv,

    /// Markdown report format
    Markdown,

    /// HTML report format
    Html,
}

impl ExportFormat {
    /// Get file extension for this format
    pub fn extension(&self) -> &str {
        match self {
            ExportFormat::Json => "json",
            ExportFormat::JsonLines => "jsonl",
            ExportFormat::Csv => "csv",
            ExportFormat::Markdown => "md",
            ExportFormat::Html => "html",
        }
    }
}

impl std::str::FromStr for ExportFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "json" => Ok(ExportFormat::Json),
            "jsonl" | "jsonlines" | "ndjson" => Ok(ExportFormat::JsonLines),
            "csv" => Ok(ExportFormat::Csv),
            "md" | "markdown" => Ok(ExportFormat::Markdown),
            "html" => Ok(ExportFormat::Html),
            _ => Err(format!("Unknown export format: {}", s)),
        }
    }
}

/// Telemetry exporter
pub struct TelemetryExporter {
    config: Arc<RwLock<TelemetryConfig>>,
}

impl TelemetryExporter {
    /// Create a new telemetry exporter
    pub fn new(config: Arc<RwLock<TelemetryConfig>>) -> Self {
        Self { config }
    }

    /// Export telemetry data
    pub async fn export(
        &self,
        events: &[TelemetryEvent],
        format: ExportFormat,
        output: &Path,
    ) -> Result<(), TelemetryError> {
        match format {
            ExportFormat::Json => self.export_json(events, output).await,
            ExportFormat::JsonLines => self.export_jsonlines(events, output).await,
            ExportFormat::Csv => self.export_csv(events, output).await,
            ExportFormat::Markdown => self.export_markdown(events, output).await,
            ExportFormat::Html => self.export_html(events, output).await,
        }
    }

    /// Export as JSON
    async fn export_json(
        &self,
        events: &[TelemetryEvent],
        output: &Path,
    ) -> Result<(), TelemetryError> {
        let json = serde_json::to_string_pretty(events)?;
        fs::write(output, json).await?;
        Ok(())
    }

    /// Export as JSON Lines
    async fn export_jsonlines(
        &self,
        events: &[TelemetryEvent],
        output: &Path,
    ) -> Result<(), TelemetryError> {
        let mut file = fs::File::create(output).await?;

        for event in events {
            let json = serde_json::to_string(event)?;
            file.write_all(json.as_bytes()).await?;
            file.write_all(b"\n").await?;
        }

        file.flush().await?;
        Ok(())
    }

    /// Export as CSV
    async fn export_csv(
        &self,
        events: &[TelemetryEvent],
        output: &Path,
    ) -> Result<(), TelemetryError> {
        let mut content = String::new();
        content.push_str("id,event_type,timestamp,user_id,session_id,metadata\n");

        for event in events {
            let metadata_json = serde_json::to_string(&event.metadata)
                .unwrap_or_else(|_| "{}".to_string())
                .replace('"', "\"\"");

            content.push_str(&format!(
                "\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"\n",
                event.id,
                event.event_type,
                event.timestamp.to_rfc3339(),
                event.user_id.as_deref().unwrap_or(""),
                event.session_id,
                metadata_json
            ));
        }

        fs::write(output, content).await?;
        Ok(())
    }

    /// Export as Markdown report
    async fn export_markdown(
        &self,
        events: &[TelemetryEvent],
        output: &Path,
    ) -> Result<(), TelemetryError> {
        let mut md = String::new();

        // Header
        md.push_str("# VoiRS Telemetry Report\n\n");
        md.push_str(&format!(
            "**Generated**: {}\n\n",
            chrono::Utc::now().to_rfc3339()
        ));
        md.push_str(&format!("**Total Events**: {}\n\n", events.len()));

        // Event count by type
        md.push_str("## Event Summary\n\n");
        let mut event_counts = std::collections::HashMap::new();
        for event in events {
            *event_counts.entry(event.event_type).or_insert(0) += 1;
        }

        md.push_str("| Event Type | Count |\n");
        md.push_str("|------------|-------|\n");
        for (event_type, count) in event_counts.iter() {
            md.push_str(&format!("| {} | {} |\n", event_type, count));
        }
        md.push('\n');

        // Recent events
        md.push_str("## Recent Events\n\n");
        let recent_events = events.iter().rev().take(20);
        md.push_str("| Timestamp | Type | Details |\n");
        md.push_str("|-----------|------|----------|\n");

        for event in recent_events {
            let metadata_str = format!("{:?}", event.metadata);
            let short_metadata = if metadata_str.len() > 50 {
                format!("{}...", &metadata_str[..50])
            } else {
                metadata_str
            };

            md.push_str(&format!(
                "| {} | {} | {} |\n",
                event.timestamp.format("%Y-%m-%d %H:%M:%S"),
                event.event_type,
                short_metadata
            ));
        }

        fs::write(output, md).await?;
        Ok(())
    }

    /// Export as HTML report
    async fn export_html(
        &self,
        events: &[TelemetryEvent],
        output: &Path,
    ) -> Result<(), TelemetryError> {
        let mut html = String::new();

        // HTML header
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<meta charset=\"UTF-8\">\n");
        html.push_str("<title>VoiRS Telemetry Report</title>\n");
        html.push_str("<style>\n");
        html.push_str("body { font-family: Arial, sans-serif; margin: 20px; }\n");
        html.push_str("table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n");
        html.push_str("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n");
        html.push_str("th { background-color: #4CAF50; color: white; }\n");
        html.push_str("tr:nth-child(even) { background-color: #f2f2f2; }\n");
        html.push_str(
            ".summary { background-color: #e7f3e7; padding: 15px; border-radius: 5px; }\n",
        );
        html.push_str("</style>\n");
        html.push_str("</head>\n<body>\n");

        // Content
        html.push_str("<h1>VoiRS Telemetry Report</h1>\n");
        html.push_str(&format!(
            "<div class=\"summary\"><p><strong>Generated:</strong> {}</p>",
            chrono::Utc::now().to_rfc3339()
        ));
        html.push_str(&format!(
            "<p><strong>Total Events:</strong> {}</p></div>\n",
            events.len()
        ));

        // Event summary table
        html.push_str("<h2>Event Summary</h2>\n");
        html.push_str("<table>\n<tr><th>Event Type</th><th>Count</th></tr>\n");

        let mut event_counts = std::collections::HashMap::new();
        for event in events {
            *event_counts.entry(event.event_type).or_insert(0) += 1;
        }

        for (event_type, count) in event_counts.iter() {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{}</td></tr>\n",
                event_type, count
            ));
        }
        html.push_str("</table>\n");

        // Recent events table
        html.push_str("<h2>Recent Events</h2>\n");
        html.push_str(
            "<table>\n<tr><th>Timestamp</th><th>Type</th><th>ID</th><th>Session</th></tr>\n",
        );

        for event in events.iter().rev().take(50) {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>\n",
                event.timestamp.format("%Y-%m-%d %H:%M:%S"),
                event.event_type,
                &event.id[..8],
                &event.session_id[..8]
            ));
        }
        html.push_str("</table>\n");

        // HTML footer
        html.push_str("</body>\n</html>\n");

        fs::write(output, html).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::{config::TelemetryConfig, events::EventType};
    use std::sync::Arc;
    use tokio::sync::RwLock;

    #[test]
    fn test_export_format_extension() {
        assert_eq!(ExportFormat::Json.extension(), "json");
        assert_eq!(ExportFormat::JsonLines.extension(), "jsonl");
        assert_eq!(ExportFormat::Csv.extension(), "csv");
        assert_eq!(ExportFormat::Markdown.extension(), "md");
        assert_eq!(ExportFormat::Html.extension(), "html");
    }

    #[test]
    fn test_export_format_from_str() {
        assert_eq!(
            "json".parse::<ExportFormat>().expect("valid format"),
            ExportFormat::Json
        );
        assert_eq!(
            "jsonl".parse::<ExportFormat>().expect("valid format"),
            ExportFormat::JsonLines
        );
        assert_eq!(
            "csv".parse::<ExportFormat>().expect("valid format"),
            ExportFormat::Csv
        );
        assert_eq!(
            "markdown".parse::<ExportFormat>().expect("valid format"),
            ExportFormat::Markdown
        );
        assert_eq!(
            "html".parse::<ExportFormat>().expect("valid format"),
            ExportFormat::Html
        );
        assert!("invalid".parse::<ExportFormat>().is_err());
    }

    #[tokio::test]
    async fn test_export_json() {
        let temp_file = std::env::temp_dir().join("voirs_export_test.json");
        let config = Arc::new(RwLock::new(TelemetryConfig::default()));
        let exporter = TelemetryExporter::new(config);

        let events = vec![TelemetryEvent::command_executed("test".to_string(), 100)];

        let result = exporter
            .export(&events, ExportFormat::Json, &temp_file)
            .await;
        assert!(result.is_ok());
        assert!(temp_file.exists());

        // Cleanup
        let _ = std::fs::remove_file(temp_file);
    }

    #[tokio::test]
    async fn test_export_jsonlines() {
        let temp_file = std::env::temp_dir().join("voirs_export_test.jsonl");
        let config = Arc::new(RwLock::new(TelemetryConfig::default()));
        let exporter = TelemetryExporter::new(config);

        let events = vec![
            TelemetryEvent::command_executed("test1".to_string(), 100),
            TelemetryEvent::command_executed("test2".to_string(), 200),
        ];

        let result = exporter
            .export(&events, ExportFormat::JsonLines, &temp_file)
            .await;
        assert!(result.is_ok());
        assert!(temp_file.exists());

        // Verify content
        let content = std::fs::read_to_string(&temp_file).unwrap();
        assert_eq!(content.lines().count(), 2);

        // Cleanup
        let _ = std::fs::remove_file(temp_file);
    }

    #[tokio::test]
    async fn test_export_csv() {
        let temp_file = std::env::temp_dir().join("voirs_export_test.csv");
        let config = Arc::new(RwLock::new(TelemetryConfig::default()));
        let exporter = TelemetryExporter::new(config);

        let events = vec![TelemetryEvent::command_executed("test".to_string(), 100)];

        let result = exporter
            .export(&events, ExportFormat::Csv, &temp_file)
            .await;
        assert!(result.is_ok());
        assert!(temp_file.exists());

        // Verify CSV header
        let content = std::fs::read_to_string(&temp_file).unwrap();
        assert!(content.starts_with("id,event_type,timestamp"));

        // Cleanup
        let _ = std::fs::remove_file(temp_file);
    }

    #[tokio::test]
    async fn test_export_markdown() {
        let temp_file = std::env::temp_dir().join("voirs_export_test.md");
        let config = Arc::new(RwLock::new(TelemetryConfig::default()));
        let exporter = TelemetryExporter::new(config);

        let events = vec![TelemetryEvent::command_executed("test".to_string(), 100)];

        let result = exporter
            .export(&events, ExportFormat::Markdown, &temp_file)
            .await;
        assert!(result.is_ok());
        assert!(temp_file.exists());

        // Verify markdown content
        let content = std::fs::read_to_string(&temp_file).unwrap();
        assert!(content.contains("# VoiRS Telemetry Report"));
        assert!(content.contains("## Event Summary"));

        // Cleanup
        let _ = std::fs::remove_file(temp_file);
    }

    #[tokio::test]
    async fn test_export_html() {
        let temp_file = std::env::temp_dir().join("voirs_export_test.html");
        let config = Arc::new(RwLock::new(TelemetryConfig::default()));
        let exporter = TelemetryExporter::new(config);

        let events = vec![TelemetryEvent::command_executed("test".to_string(), 100)];

        let result = exporter
            .export(&events, ExportFormat::Html, &temp_file)
            .await;
        assert!(result.is_ok());
        assert!(temp_file.exists());

        // Verify HTML content
        let content = std::fs::read_to_string(&temp_file).unwrap();
        assert!(content.contains("<!DOCTYPE html>"));
        assert!(content.contains("<title>VoiRS Telemetry Report</title>"));

        // Cleanup
        let _ = std::fs::remove_file(temp_file);
    }
}
