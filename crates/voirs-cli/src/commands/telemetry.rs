//! Telemetry management commands

use clap::Subcommand;
use std::path::{Path, PathBuf};

use crate::telemetry::{
    config::{TelemetryConfig, TelemetryLevel},
    export::ExportFormat,
    privacy::AnonymizationLevel,
    TelemetrySystem,
};

/// Telemetry subcommands
#[derive(Clone, Subcommand)]
pub enum TelemetryCommands {
    /// Enable telemetry collection
    Enable {
        /// Telemetry level (minimal, standard, detailed, debug)
        #[arg(short, long, default_value = "standard")]
        level: String,

        /// Anonymization level (none, low, medium, high)
        #[arg(short, long, default_value = "medium")]
        anonymization: String,
    },

    /// Disable telemetry collection
    Disable,

    /// Show telemetry status and statistics
    Status {
        /// Show detailed statistics
        #[arg(short, long)]
        detailed: bool,
    },

    /// Export telemetry data
    Export {
        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Export format (json, jsonl, csv, markdown, html)
        #[arg(short, long, default_value = "json")]
        format: String,
    },

    /// Clear all telemetry data
    Clear {
        /// Skip confirmation prompt
        #[arg(short, long)]
        yes: bool,
    },

    /// Show telemetry configuration
    Config,

    /// Set telemetry configuration
    SetConfig {
        /// Enable/disable telemetry
        #[arg(long)]
        enabled: Option<bool>,

        /// Telemetry level
        #[arg(long)]
        level: Option<String>,

        /// Anonymization level
        #[arg(long)]
        anonymization: Option<String>,

        /// Remote endpoint URL
        #[arg(long)]
        remote_endpoint: Option<String>,
    },
}

/// Execute telemetry command
pub async fn execute(cmd: TelemetryCommands) -> Result<(), Box<dyn std::error::Error>> {
    match cmd {
        TelemetryCommands::Enable {
            level,
            anonymization,
        } => enable_telemetry(&level, &anonymization).await,
        TelemetryCommands::Disable => disable_telemetry().await,
        TelemetryCommands::Status { detailed } => show_status(detailed).await,
        TelemetryCommands::Export { output, format } => export_data(&output, &format).await,
        TelemetryCommands::Clear { yes } => clear_data(yes).await,
        TelemetryCommands::Config => show_config().await,
        TelemetryCommands::SetConfig {
            enabled,
            level,
            anonymization,
            remote_endpoint,
        } => set_config(enabled, level, anonymization, remote_endpoint).await,
    }
}

/// Enable telemetry
async fn enable_telemetry(
    level_str: &str,
    anonymization_str: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let level: TelemetryLevel = level_str.parse()?;
    let anonymization: AnonymizationLevel = match anonymization_str.to_lowercase().as_str() {
        "none" => AnonymizationLevel::None,
        "low" => AnonymizationLevel::Low,
        "medium" => AnonymizationLevel::Medium,
        "high" => AnonymizationLevel::High,
        _ => return Err(format!("Invalid anonymization level: {}", anonymization_str).into()),
    };

    let config = TelemetryConfig::enabled()
        .with_level(level)
        .with_anonymization(anonymization);

    config.validate()?;
    save_config(&config).await?;

    println!("✅ Telemetry enabled");
    println!("   Level: {}", level);
    println!("   Anonymization: {}", anonymization);
    println!("   Storage: {}", config.storage_path.display());

    Ok(())
}

/// Disable telemetry
async fn disable_telemetry() -> Result<(), Box<dyn std::error::Error>> {
    let config = TelemetryConfig::disabled();
    save_config(&config).await?;

    println!("✅ Telemetry disabled");

    Ok(())
}

/// Show telemetry status
async fn show_status(detailed: bool) -> Result<(), Box<dyn std::error::Error>> {
    let config = load_config().await?;

    println!("Telemetry Status");
    println!("================");
    println!(
        "Enabled: {}",
        if config.enabled { "✅ Yes" } else { "❌ No" }
    );

    if !config.enabled {
        return Ok(());
    }

    println!("Level: {}", config.level);
    println!("Anonymization: {}", config.anonymization);
    println!("Storage: {}", config.storage_path.display());

    if let Some(ref endpoint) = config.remote_endpoint {
        println!("Remote endpoint: {}", endpoint);
    }

    // Show statistics
    let system = TelemetrySystem::new(config).await?;
    let stats = system.get_statistics().await?;

    println!("\nStatistics");
    println!("----------");
    println!("Total events: {}", stats.total_events);
    println!("Synthesis requests: {}", stats.synthesis_requests);
    println!(
        "Average synthesis duration: {:.2}ms",
        stats.avg_synthesis_duration
    );
    println!("Total errors: {}", stats.total_errors);
    println!("Storage size: {}", format_bytes(stats.storage_size_bytes));

    if detailed {
        println!("\nEvents by type:");
        for (event_type, count) in &stats.events_by_type {
            println!("  {}: {}", event_type, count);
        }

        if !stats.most_used_commands.is_empty() {
            println!("\nMost used commands:");
            for (command, count) in stats.most_used_commands.iter().take(5) {
                println!("  {}: {}", command, count);
            }
        }

        if !stats.most_used_voices.is_empty() {
            println!("\nMost used voices:");
            for (voice, count) in stats.most_used_voices.iter().take(5) {
                println!("  {}: {}", voice, count);
            }
        }

        if let (Some(start), Some(end)) = (stats.start_time, stats.end_time) {
            println!("\nTime range:");
            println!("  From: {}", start.format("%Y-%m-%d %H:%M:%S"));
            println!("  To: {}", end.format("%Y-%m-%d %H:%M:%S"));
        }
    }

    Ok(())
}

/// Export telemetry data
async fn export_data(output: &Path, format_str: &str) -> Result<(), Box<dyn std::error::Error>> {
    let config = load_config().await?;

    if !config.enabled {
        println!("⚠️  Telemetry is disabled. No data to export.");
        return Ok(());
    }

    let format: ExportFormat = format_str.parse()?;
    let system = TelemetrySystem::new(config).await?;

    println!("Exporting telemetry data...");
    system.export(format, output).await?;

    println!("✅ Telemetry data exported to: {}", output.display());
    println!("   Format: {}", format_str);

    Ok(())
}

/// Clear telemetry data
async fn clear_data(skip_confirmation: bool) -> Result<(), Box<dyn std::error::Error>> {
    let config = load_config().await?;

    if !skip_confirmation {
        println!("⚠️  This will permanently delete all telemetry data.");
        print!("Are you sure? (y/N): ");
        use std::io::{self, Write};
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !matches!(input.trim().to_lowercase().as_str(), "y" | "yes") {
            println!("Cancelled.");
            return Ok(());
        }
    }

    let system = TelemetrySystem::new(config).await?;
    system.clear_data().await?;

    println!("✅ Telemetry data cleared");

    Ok(())
}

/// Show telemetry configuration
async fn show_config() -> Result<(), Box<dyn std::error::Error>> {
    let config = load_config().await?;

    println!("Telemetry Configuration");
    println!("=======================");
    println!("Enabled: {}", config.enabled);
    println!("Level: {}", config.level);
    println!("Anonymization: {}", config.anonymization);
    println!("Storage path: {}", config.storage_path.display());
    println!("Batch size: {}", config.batch_size);
    println!("Flush interval: {}s", config.flush_interval_secs);

    if let Some(ref endpoint) = config.remote_endpoint {
        println!("Remote endpoint: {}", endpoint);
    } else {
        println!("Remote endpoint: (not configured)");
    }

    Ok(())
}

/// Set telemetry configuration
async fn set_config(
    enabled: Option<bool>,
    level: Option<String>,
    anonymization: Option<String>,
    remote_endpoint: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut config = load_config().await?;

    if let Some(enabled) = enabled {
        config.enabled = enabled;
    }

    if let Some(level_str) = level {
        config.level = level_str.parse()?;
    }

    if let Some(anon_str) = anonymization {
        config.anonymization = match anon_str.to_lowercase().as_str() {
            "none" => AnonymizationLevel::None,
            "low" => AnonymizationLevel::Low,
            "medium" => AnonymizationLevel::Medium,
            "high" => AnonymizationLevel::High,
            _ => return Err(format!("Invalid anonymization level: {}", anon_str).into()),
        };
    }

    if let Some(endpoint) = remote_endpoint {
        if endpoint.is_empty() {
            config.remote_endpoint = None;
        } else {
            config.remote_endpoint = Some(endpoint);
        }
    }

    config.validate()?;
    save_config(&config).await?;

    println!("✅ Telemetry configuration updated");

    Ok(())
}

/// Load telemetry configuration
async fn load_config() -> Result<TelemetryConfig, Box<dyn std::error::Error>> {
    let config_path = get_config_path()?;

    if !config_path.exists() {
        return Ok(TelemetryConfig::default());
    }

    let content = tokio::fs::read_to_string(&config_path).await?;
    let config: TelemetryConfig = serde_json::from_str(&content)?;

    Ok(config)
}

/// Save telemetry configuration
async fn save_config(config: &TelemetryConfig) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = get_config_path()?;

    if let Some(parent) = config_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let content = serde_json::to_string_pretty(config)?;
    tokio::fs::write(&config_path, content).await?;

    Ok(())
}

/// Get telemetry configuration file path
fn get_config_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let config_dir = dirs::config_dir()
        .ok_or("Could not determine config directory")?
        .join("voirs");

    Ok(config_dir.join("telemetry.json"))
}

/// Format bytes for display
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 bytes");
        assert_eq!(format_bytes(500), "500 bytes");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[tokio::test]
    async fn test_get_config_path() {
        let path = get_config_path();
        assert!(path.is_ok());
        assert!(path.unwrap().to_str().unwrap_or_default().contains("voirs"));
    }

    #[tokio::test]
    async fn test_save_and_load_config() {
        let config = TelemetryConfig::enabled();
        let result = save_config(&config).await;
        assert!(result.is_ok());

        let loaded = load_config().await;
        assert!(loaded.is_ok());
        assert!(loaded.unwrap().enabled);
    }
}
