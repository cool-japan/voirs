//! Logging configuration and utilities for VoiRS.

use crate::{
    config::LoggingConfig,
    error::Result,
    VoirsError,
};
use std::{fs, path::Path};
use tracing::{Level, Subscriber};
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Layer, Registry,
};

/// Initialize logging based on configuration
pub fn init_logging(config: &LoggingConfig) -> Result<()> {
    let level = parse_level(&config.level)?;
    
    // Create base filter
    let env_filter = EnvFilter::builder()
        .with_default_directive(level.into())
        .from_env()
        .map_err(|e| VoirsError::config_error(format!("Invalid log filter: {}", e)))?;
    
    // Build simple subscriber
    let console_layer = if config.structured {
        fmt::layer().json().boxed()
    } else {
        fmt::layer().pretty().boxed()
    };
    
    tracing_subscriber::registry()
        .with(env_filter)
        .with(console_layer)
        .init();
    
    tracing::info!("VoiRS logging initialized");
    tracing::debug!("Log level: {}", config.level);
    tracing::debug!("Structured logging: {}", config.structured);
    tracing::debug!("Metrics enabled: {}", config.metrics);
    
    Ok(())
}

/// Parse log level from string
fn parse_level(level_str: &str) -> Result<Level> {
    match level_str.to_lowercase().as_str() {
        "trace" => Ok(Level::TRACE),
        "debug" => Ok(Level::DEBUG),
        "info" => Ok(Level::INFO),
        "warn" | "warning" => Ok(Level::WARN),
        "error" => Ok(Level::ERROR),
        _ => Err(VoirsError::config_error(format!(
            "Invalid log level: {}. Valid levels: trace, debug, info, warn, error",
            level_str
        ))),
    }
}

/// Create console logging layer
fn create_console_layer(
    config: &LoggingConfig,
) -> Result<impl Layer<Registry> + Send + Sync + 'static> {
    let layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_span_events(FmtSpan::CLOSE);
    
    if config.structured {
        Ok(layer.json().boxed())
    } else {
        Ok(layer.pretty().boxed())
    }
}

/// Create file logging layer
fn create_file_layer(
    file_path: &Path,
    config: &LoggingConfig,
) -> Result<impl Layer<Registry> + Send + Sync + 'static> {
    // Ensure parent directory exists
    if let Some(parent) = file_path.parent() {
        fs::create_dir_all(parent)?;
    }
    
    // Create or open log file
    let file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(true)
        .open(file_path)?;
    
    let layer = fmt::layer()
        .with_writer(file)
        .with_target(true)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE);
    
    if config.structured {
        Ok(layer.json().boxed())
    } else {
        Ok(layer.boxed())
    }
}

/// Create metrics logging layer
fn create_metrics_layer() -> Result<impl Layer<Registry> + Send + Sync + 'static> {
    // TODO: Implement metrics collection layer
    // This could integrate with systems like:
    // - OpenTelemetry for distributed tracing
    // - Prometheus for metrics collection
    // - Custom performance counters
    
    Ok(fmt::layer()
        .with_writer(std::io::stderr) // Use stderr for now
        .boxed())
}

/// Performance timing utilities
pub struct PerfTimer {
    start: std::time::Instant,
    name: String,
}

impl PerfTimer {
    /// Start a new performance timer
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        tracing::debug!("Starting timer: {}", name);
        Self {
            start: std::time::Instant::now(),
            name,
        }
    }
    
    /// Record elapsed time and log it
    pub fn elapsed(&self) -> std::time::Duration {
        let elapsed = self.start.elapsed();
        tracing::debug!("Timer '{}': {:.2}ms", self.name, elapsed.as_secs_f64() * 1000.0);
        elapsed
    }
    
    /// Record elapsed time with custom message
    pub fn elapsed_with_message(&self, message: &str) -> std::time::Duration {
        let elapsed = self.start.elapsed();
        tracing::info!("{}: {:.2}ms", message, elapsed.as_secs_f64() * 1000.0);
        elapsed
    }
}

impl Drop for PerfTimer {
    fn drop(&mut self) {
        self.elapsed();
    }
}

/// Memory usage tracking utilities
pub struct MemoryTracker {
    initial_usage: u64,
    name: String,
}

impl MemoryTracker {
    /// Start memory tracking
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let initial_usage = get_memory_usage();
        tracing::debug!("Starting memory tracking: {} (initial: {} KB)", name, initial_usage / 1024);
        Self {
            initial_usage,
            name,
        }
    }
    
    /// Get current memory delta
    pub fn delta(&self) -> i64 {
        let current = get_memory_usage();
        current as i64 - self.initial_usage as i64
    }
    
    /// Log current memory delta
    pub fn log_delta(&self) {
        let delta = self.delta();
        if delta > 0 {
            tracing::debug!("Memory '{}': +{} KB", self.name, delta / 1024);
        } else {
            tracing::debug!("Memory '{}': {} KB", self.name, delta / 1024);
        }
    }
}

impl Drop for MemoryTracker {
    fn drop(&mut self) {
        self.log_delta();
    }
}

/// Get current memory usage (approximate)
fn get_memory_usage() -> u64 {
    // This is a simplified implementation
    // In production, you might want to use more sophisticated memory tracking
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|content| {
                content
                    .lines()
                    .find(|line| line.starts_with("VmRSS:"))
                    .and_then(|line| {
                        line.split_whitespace()
                            .nth(1)
                            .and_then(|s| s.parse::<u64>().ok())
                            .map(|kb| kb * 1024) // Convert to bytes
                    })
            })
            .unwrap_or(0)
    }
    
    #[cfg(not(target_os = "linux"))]
    {
        0 // Placeholder for other platforms
    }
}

/// Macros for convenient logging with context
#[macro_export]
macro_rules! log_synthesis {
    ($level:ident, $text:expr, $($arg:tt)*) => {
        tracing::$level!(
            target: "voirs::synthesis",
            text = %$text,
            $($arg)*
        );
    };
}

#[macro_export]
macro_rules! log_audio {
    ($level:ident, $duration:expr, $sample_rate:expr, $($arg:tt)*) => {
        tracing::$level!(
            target: "voirs::audio",
            duration_sec = %$duration,
            sample_rate = %$sample_rate,
            $($arg)*
        );
    };
}

#[macro_export]
macro_rules! log_model {
    ($level:ident, $model_type:expr, $($arg:tt)*) => {
        tracing::$level!(
            target: "voirs::model",
            model_type = %$model_type,
            $($arg)*
        );
    };
}

/// Specialized loggers for different components
pub mod synthesis {
    use super::*;
    
    pub fn log_start(text: &str) {
        tracing::info!(
            target: "voirs::synthesis",
            text = %text,
            "Starting synthesis"
        );
    }
    
    pub fn log_complete(text: &str, duration_ms: f64, audio_duration_sec: f32) {
        tracing::info!(
            target: "voirs::synthesis",
            text = %text,
            processing_time_ms = %duration_ms,
            audio_duration_sec = %audio_duration_sec,
            real_time_factor = %(audio_duration_sec * 1000.0 / duration_ms as f32),
            "Synthesis complete"
        );
    }
    
    pub fn log_error(text: &str, error: &dyn std::error::Error) {
        tracing::error!(
            target: "voirs::synthesis",
            text = %text,
            error = %error,
            "Synthesis failed"
        );
    }
}

pub mod model {
    use super::*;
    
    pub fn log_load(model_type: &str, path: &Path) {
        tracing::info!(
            target: "voirs::model",
            model_type = %model_type,
            path = %path.display(),
            "Loading model"
        );
    }
    
    pub fn log_load_complete(model_type: &str, duration_ms: f64) {
        tracing::info!(
            target: "voirs::model",
            model_type = %model_type,
            load_time_ms = %duration_ms,
            "Model loaded successfully"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_parse_level() {
        assert_eq!(parse_level("info").unwrap(), Level::INFO);
        assert_eq!(parse_level("DEBUG").unwrap(), Level::DEBUG);
        assert_eq!(parse_level("warn").unwrap(), Level::WARN);
        assert!(parse_level("invalid").is_err());
    }

    #[test]
    fn test_perf_timer() {
        let timer = PerfTimer::new("test");
        std::thread::sleep(std::time::Duration::from_millis(1));
        let elapsed = timer.elapsed();
        assert!(elapsed.as_millis() >= 1);
    }

    #[test]
    fn test_memory_tracker() {
        let tracker = MemoryTracker::new("test");
        // Allocate some memory
        let _data: Vec<u8> = vec![0; 1024 * 1024]; // 1MB
        tracker.log_delta();
        // Memory delta might be 0 on some systems, but at least it shouldn't crash
    }

    #[test]
    fn test_logging_init() {
        let config = LoggingConfig {
            level: "debug".to_string(),
            structured: false,
            file_path: None,
            max_file_size_mb: 10,
            max_files: 5,
            metrics: false,
            module_levels: std::collections::HashMap::new(),
        };
        
        // This might fail if logging is already initialized in other tests
        let _ = init_logging(&config);
    }

    #[test]
    fn test_file_logging() {
        let temp_dir = tempdir().unwrap();
        let log_file = temp_dir.path().join("test.log");
        
        let config = LoggingConfig {
            level: "info".to_string(),
            structured: false,
            file_path: Some(log_file.clone()),
            max_file_size_mb: 10,
            max_files: 5,
            metrics: false,
            module_levels: std::collections::HashMap::new(),
        };
        
        // Test file layer creation
        let result = create_file_layer(&log_file, &config);
        assert!(result.is_ok());
        assert!(log_file.exists());
    }
}