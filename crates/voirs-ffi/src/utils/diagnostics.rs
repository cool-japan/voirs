//! Diagnostic Utilities for VoiRS FFI
//!
//! This module provides comprehensive diagnostic tools for troubleshooting
//! FFI issues, performance analysis, and system health monitoring.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Diagnostic information collector
pub struct DiagnosticCollector {
    /// System information
    system_info: Arc<RwLock<HashMap<String, String>>>,
    /// Performance counters
    counters: Arc<RwLock<HashMap<String, AtomicU64>>>,
    /// Recent errors
    recent_errors: Arc<RwLock<Vec<DiagnosticError>>>,
    /// Start time
    start_time: Instant,
}

/// Diagnostic error entry
#[derive(Debug, Clone)]
pub struct DiagnosticError {
    /// Timestamp when error occurred
    pub timestamp: Duration,
    /// Error code
    pub code: i32,
    /// Error message
    pub message: String,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// System health status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// System is functioning normally
    Healthy,
    /// System has minor issues but is operational
    Degraded,
    /// System has significant issues
    Unhealthy,
    /// System status cannot be determined
    Unknown,
}

impl DiagnosticCollector {
    /// Create a new diagnostic collector
    pub fn new() -> Self {
        Self {
            system_info: Arc::new(RwLock::new(HashMap::new())),
            counters: Arc::new(RwLock::new(HashMap::new())),
            recent_errors: Arc::new(RwLock::new(Vec::new())),
            start_time: Instant::now(),
        }
    }

    /// Add system information
    pub fn add_system_info(&self, key: String, value: String) {
        let mut info = self.system_info.write();
        info.insert(key, value);
    }

    /// Get system information
    pub fn get_system_info(&self, key: &str) -> Option<String> {
        let info = self.system_info.read();
        info.get(key).cloned()
    }

    /// Increment a counter
    pub fn increment_counter(&self, name: &str, value: u64) {
        let counters = self.counters.read();
        if let Some(counter) = counters.get(name) {
            counter.fetch_add(value, Ordering::Relaxed);
        } else {
            drop(counters);
            let mut counters = self.counters.write();
            counters.insert(name.to_string(), AtomicU64::new(value));
        }
    }

    /// Get counter value
    pub fn get_counter(&self, name: &str) -> u64 {
        let counters = self.counters.read();
        counters
            .get(name)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Record a diagnostic error
    pub fn record_error(&self, code: i32, message: String, context: HashMap<String, String>) {
        let mut errors = self.recent_errors.write();
        let timestamp = self.start_time.elapsed();

        errors.push(DiagnosticError {
            timestamp,
            code,
            message,
            context,
        });

        // Keep only last 100 errors
        if errors.len() > 100 {
            errors.remove(0);
        }
    }

    /// Get recent errors
    pub fn get_recent_errors(&self, count: usize) -> Vec<DiagnosticError> {
        let errors = self.recent_errors.read();
        errors.iter().rev().take(count).cloned().collect()
    }

    /// Get uptime
    pub fn get_uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Check system health
    pub fn check_health(&self) -> HealthStatus {
        let errors = self.recent_errors.read();
        let current_uptime = self.start_time.elapsed();

        // Count errors in the last 60 seconds
        let recent_error_count = errors
            .iter()
            .filter(|e| {
                // Only count errors from the last 60 seconds
                current_uptime.saturating_sub(e.timestamp) <= Duration::from_secs(60)
            })
            .count();

        if recent_error_count == 0 {
            HealthStatus::Healthy
        } else if recent_error_count < 5 {
            HealthStatus::Degraded
        } else if recent_error_count < 20 {
            HealthStatus::Unhealthy
        } else {
            HealthStatus::Unknown
        }
    }

    /// Generate diagnostic report
    pub fn generate_report(&self) -> DiagnosticReport {
        let system_info = self.system_info.read().clone();
        let counters: HashMap<String, u64> = self
            .counters
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), v.load(Ordering::Relaxed)))
            .collect();
        let recent_errors = self.get_recent_errors(10);
        let uptime = self.get_uptime();
        let health = self.check_health();

        DiagnosticReport {
            system_info,
            counters,
            recent_errors,
            uptime,
            health,
        }
    }

    /// Clear all diagnostic data
    pub fn clear(&self) {
        self.system_info.write().clear();
        self.counters.write().clear();
        self.recent_errors.write().clear();
    }
}

impl Default for DiagnosticCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive diagnostic report
#[derive(Debug, Clone)]
pub struct DiagnosticReport {
    /// System information
    pub system_info: HashMap<String, String>,
    /// Performance counters
    pub counters: HashMap<String, u64>,
    /// Recent errors
    pub recent_errors: Vec<DiagnosticError>,
    /// System uptime
    pub uptime: Duration,
    /// Health status
    pub health: HealthStatus,
}

impl DiagnosticReport {
    /// Format as human-readable string
    pub fn format(&self) -> String {
        let mut output = String::new();

        output.push_str("=== VoiRS FFI Diagnostic Report ===\n\n");

        output.push_str(&format!("Health Status: {:?}\n", self.health));
        output.push_str(&format!("Uptime: {:?}\n\n", self.uptime));

        output.push_str("System Information:\n");
        for (key, value) in &self.system_info {
            output.push_str(&format!("  {}: {}\n", key, value));
        }

        output.push_str("\nPerformance Counters:\n");
        for (name, value) in &self.counters {
            output.push_str(&format!("  {}: {}\n", name, value));
        }

        output.push_str(&format!(
            "\nRecent Errors ({}):\n",
            self.recent_errors.len()
        ));
        for error in &self.recent_errors {
            output.push_str(&format!(
                "  [{:?}] Code {}: {}\n",
                error.timestamp, error.code, error.message
            ));
            for (key, value) in &error.context {
                output.push_str(&format!("    {}: {}\n", key, value));
            }
        }

        output
    }

    /// Export as JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        #[derive(serde::Serialize)]
        struct JsonReport<'a> {
            health: String,
            uptime_secs: u64,
            system_info: &'a HashMap<String, String>,
            counters: &'a HashMap<String, u64>,
            recent_errors: Vec<JsonError<'a>>,
        }

        #[derive(serde::Serialize)]
        struct JsonError<'a> {
            timestamp_secs: u64,
            code: i32,
            message: &'a str,
            context: &'a HashMap<String, String>,
        }

        let json_errors: Vec<JsonError> = self
            .recent_errors
            .iter()
            .map(|e| JsonError {
                timestamp_secs: e.timestamp.as_secs(),
                code: e.code,
                message: &e.message,
                context: &e.context,
            })
            .collect();

        let report = JsonReport {
            health: format!("{:?}", self.health),
            uptime_secs: self.uptime.as_secs(),
            system_info: &self.system_info,
            counters: &self.counters,
            recent_errors: json_errors,
        };

        serde_json::to_string_pretty(&report)
    }
}

/// Global diagnostic collector instance
static GLOBAL_DIAGNOSTICS: once_cell::sync::Lazy<DiagnosticCollector> =
    once_cell::sync::Lazy::new(DiagnosticCollector::new);

/// Get the global diagnostic collector
pub fn get_global_diagnostics() -> &'static DiagnosticCollector {
    &GLOBAL_DIAGNOSTICS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_collector_creation() {
        let collector = DiagnosticCollector::new();
        assert_eq!(collector.get_counter("test"), 0);
    }

    #[test]
    fn test_system_info() {
        let collector = DiagnosticCollector::new();
        collector.add_system_info("os".to_string(), "Linux".to_string());
        assert_eq!(collector.get_system_info("os"), Some("Linux".to_string()));
    }

    #[test]
    fn test_counter_operations() {
        let collector = DiagnosticCollector::new();
        collector.increment_counter("requests", 1);
        collector.increment_counter("requests", 2);
        assert_eq!(collector.get_counter("requests"), 3);
    }

    #[test]
    fn test_error_recording() {
        let collector = DiagnosticCollector::new();
        let mut context = HashMap::new();
        context.insert("function".to_string(), "test".to_string());

        collector.record_error(1, "Test error".to_string(), context);
        let errors = collector.get_recent_errors(1);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].code, 1);
        assert_eq!(errors[0].message, "Test error");
    }

    #[test]
    fn test_health_check() {
        let collector = DiagnosticCollector::new();
        assert_eq!(collector.check_health(), HealthStatus::Healthy);

        // Add some errors
        for i in 0..3 {
            collector.record_error(i, format!("Error {}", i), HashMap::new());
        }

        // Wait briefly to ensure timestamp difference
        std::thread::sleep(std::time::Duration::from_millis(10));

        let health = collector.check_health();
        assert!(health == HealthStatus::Degraded || health == HealthStatus::Healthy);
    }

    #[test]
    fn test_diagnostic_report() {
        let collector = DiagnosticCollector::new();
        collector.add_system_info("version".to_string(), "0.1.0".to_string());
        collector.increment_counter("total_calls", 42);

        let report = collector.generate_report();
        assert_eq!(
            report.system_info.get("version"),
            Some(&"0.1.0".to_string())
        );
        assert_eq!(report.counters.get("total_calls"), Some(&42));
    }

    #[test]
    fn test_report_formatting() {
        let collector = DiagnosticCollector::new();
        collector.add_system_info("test".to_string(), "value".to_string());
        let report = collector.generate_report();
        let formatted = report.format();
        assert!(formatted.contains("VoiRS FFI Diagnostic Report"));
        assert!(formatted.contains("test: value"));
    }

    #[test]
    fn test_report_json() {
        let collector = DiagnosticCollector::new();
        collector.add_system_info("test".to_string(), "value".to_string());
        let report = collector.generate_report();
        let json = report.to_json().unwrap();
        assert!(json.contains("\"test\": \"value\""));
    }

    #[test]
    fn test_error_limit() {
        let collector = DiagnosticCollector::new();

        // Add more than 100 errors
        for i in 0..150 {
            collector.record_error(i, format!("Error {}", i), HashMap::new());
        }

        let errors = collector.get_recent_errors(200);
        assert_eq!(errors.len(), 100); // Should be capped at 100
    }

    #[test]
    fn test_global_diagnostics() {
        let diagnostics = get_global_diagnostics();
        diagnostics.increment_counter("global_test", 1);
        assert_eq!(diagnostics.get_counter("global_test"), 1);
    }
}
