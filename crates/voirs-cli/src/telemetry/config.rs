//! Telemetry configuration

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::privacy::AnonymizationLevel;

/// Telemetry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Enable or disable telemetry
    pub enabled: bool,

    /// Telemetry collection level
    pub level: TelemetryLevel,

    /// Local storage path for telemetry data
    pub storage_path: PathBuf,

    /// Anonymization level for privacy
    pub anonymization: AnonymizationLevel,

    /// Optional remote endpoint for telemetry submission
    pub remote_endpoint: Option<String>,

    /// Batch size for remote submission
    pub batch_size: usize,

    /// Flush interval in seconds
    pub flush_interval_secs: u64,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Opt-in by default
            level: TelemetryLevel::Standard,
            storage_path: Self::default_storage_path(),
            anonymization: AnonymizationLevel::Medium,
            remote_endpoint: None,
            batch_size: 100,
            flush_interval_secs: 300, // 5 minutes
        }
    }
}

impl TelemetryConfig {
    /// Get default storage path
    fn default_storage_path() -> PathBuf {
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("voirs")
            .join("telemetry")
    }

    /// Create config with telemetry enabled
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }

    /// Create config with telemetry disabled
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Set telemetry level
    pub fn with_level(mut self, level: TelemetryLevel) -> Self {
        self.level = level;
        self
    }

    /// Set storage path
    pub fn with_storage_path(mut self, path: PathBuf) -> Self {
        self.storage_path = path;
        self
    }

    /// Set anonymization level
    pub fn with_anonymization(mut self, level: AnonymizationLevel) -> Self {
        self.anonymization = level;
        self
    }

    /// Set remote endpoint
    pub fn with_remote_endpoint(mut self, endpoint: String) -> Self {
        self.remote_endpoint = Some(endpoint);
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.batch_size == 0 {
            return Err("Batch size must be greater than 0".to_string());
        }

        if self.flush_interval_secs == 0 {
            return Err("Flush interval must be greater than 0".to_string());
        }

        if let Some(ref endpoint) = self.remote_endpoint {
            if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
                return Err("Remote endpoint must be a valid HTTP(S) URL".to_string());
            }
        }

        Ok(())
    }
}

/// Telemetry collection level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TelemetryLevel {
    /// Minimal telemetry - only critical errors and basic usage
    Minimal,

    /// Standard telemetry - errors, performance, and command usage
    Standard,

    /// Detailed telemetry - full diagnostic information
    Detailed,

    /// Debug telemetry - includes all events and debug information
    Debug,
}

impl TelemetryLevel {
    /// Check if this level includes command execution events
    pub fn includes_commands(&self) -> bool {
        matches!(
            self,
            TelemetryLevel::Standard | TelemetryLevel::Detailed | TelemetryLevel::Debug
        )
    }

    /// Check if this level includes performance metrics
    pub fn includes_performance(&self) -> bool {
        matches!(
            self,
            TelemetryLevel::Standard | TelemetryLevel::Detailed | TelemetryLevel::Debug
        )
    }

    /// Check if this level includes debug information
    pub fn includes_debug(&self) -> bool {
        matches!(self, TelemetryLevel::Debug)
    }

    /// Check if this level includes errors
    pub fn includes_errors(&self) -> bool {
        true // All levels include errors
    }
}

impl std::fmt::Display for TelemetryLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TelemetryLevel::Minimal => write!(f, "minimal"),
            TelemetryLevel::Standard => write!(f, "standard"),
            TelemetryLevel::Detailed => write!(f, "detailed"),
            TelemetryLevel::Debug => write!(f, "debug"),
        }
    }
}

impl std::str::FromStr for TelemetryLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "minimal" | "min" => Ok(TelemetryLevel::Minimal),
            "standard" | "std" => Ok(TelemetryLevel::Standard),
            "detailed" | "full" => Ok(TelemetryLevel::Detailed),
            "debug" | "dbg" => Ok(TelemetryLevel::Debug),
            _ => Err(format!("Invalid telemetry level: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TelemetryConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.level, TelemetryLevel::Standard);
        assert!(config.remote_endpoint.is_none());
    }

    #[test]
    fn test_enabled_config() {
        let config = TelemetryConfig::enabled();
        assert!(config.enabled);
    }

    #[test]
    fn test_disabled_config() {
        let config = TelemetryConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_builder_pattern() {
        let config = TelemetryConfig::default()
            .with_level(TelemetryLevel::Detailed)
            .with_remote_endpoint("https://telemetry.example.com".to_string());

        assert_eq!(config.level, TelemetryLevel::Detailed);
        assert_eq!(
            config.remote_endpoint,
            Some("https://telemetry.example.com".to_string())
        );
    }

    #[test]
    fn test_config_validation() {
        let config = TelemetryConfig::default();
        assert!(config.validate().is_ok());

        let invalid_config = TelemetryConfig {
            batch_size: 0,
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_telemetry_level_includes() {
        assert!(TelemetryLevel::Minimal.includes_errors());
        assert!(!TelemetryLevel::Minimal.includes_commands());
        assert!(!TelemetryLevel::Minimal.includes_debug());

        assert!(TelemetryLevel::Standard.includes_commands());
        assert!(TelemetryLevel::Standard.includes_performance());
        assert!(!TelemetryLevel::Standard.includes_debug());

        assert!(TelemetryLevel::Debug.includes_debug());
    }

    #[test]
    fn test_telemetry_level_from_str() {
        assert_eq!(
            "minimal".parse::<TelemetryLevel>().unwrap(),
            TelemetryLevel::Minimal
        );
        assert_eq!(
            "standard".parse::<TelemetryLevel>().unwrap(),
            TelemetryLevel::Standard
        );
        assert_eq!(
            "detailed".parse::<TelemetryLevel>().unwrap(),
            TelemetryLevel::Detailed
        );
        assert_eq!(
            "debug".parse::<TelemetryLevel>().unwrap(),
            TelemetryLevel::Debug
        );
        assert!("invalid".parse::<TelemetryLevel>().is_err());
    }

    #[test]
    fn test_telemetry_level_display() {
        assert_eq!(TelemetryLevel::Minimal.to_string(), "minimal");
        assert_eq!(TelemetryLevel::Standard.to_string(), "standard");
        assert_eq!(TelemetryLevel::Detailed.to_string(), "detailed");
        assert_eq!(TelemetryLevel::Debug.to_string(), "debug");
    }
}
