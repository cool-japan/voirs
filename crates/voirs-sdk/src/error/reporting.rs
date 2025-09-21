//! Error reporting, logging, and diagnostics.
//!
//! This module provides comprehensive error reporting capabilities including:
//! - Structured error logging with context
//! - Error aggregation and statistics
//! - Diagnostic information collection
//! - Error notification and alerting
//! - Performance impact analysis

use super::types::{ErrorSeverity, ErrorWithContext, VoirsError};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
    time::{Duration, SystemTime},
};

/// Error report containing detailed diagnostic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorReport {
    /// Unique identifier for this error report
    pub id: String,
    /// Timestamp when error occurred
    pub timestamp: SystemTime,
    /// Error severity level
    pub severity: ErrorSeverity,
    /// Component where error occurred
    pub component: String,
    /// Operation being performed
    pub operation: String,
    /// Error message
    pub message: String,
    /// Error category
    pub category: ErrorCategory,
    /// System context at time of error
    pub system_context: SystemContext,
    /// Stack trace if available
    pub stack_trace: Option<String>,
    /// Related error reports
    pub related_errors: Vec<String>,
    /// Recovery attempts made
    pub recovery_attempts: u32,
    /// Whether error was recovered
    pub recovered: bool,
    /// Performance impact metrics
    pub performance_impact: Option<PerformanceImpact>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Error category for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// User input or configuration error
    UserError,
    /// System resource limitation
    ResourceError,
    /// External dependency failure
    DependencyError,
    /// Hardware or device issue
    HardwareError,
    /// Network connectivity problem
    NetworkError,
    /// Software bug or logic error
    SoftwareError,
    /// Performance degradation
    PerformanceError,
    /// Security-related error
    SecurityError,
}

/// System context at time of error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemContext {
    /// Available memory in MB
    pub available_memory_mb: u32,
    /// CPU usage percentage
    pub cpu_usage: f32,
    /// GPU memory usage in MB
    pub gpu_memory_mb: Option<u32>,
    /// Active threads count
    pub active_threads: u32,
    /// System load average
    pub load_average: Option<f32>,
    /// Operating system info
    pub os_info: String,
    /// Runtime environment
    pub runtime_info: RuntimeInfo,
}

/// Runtime environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeInfo {
    /// VoiRS SDK version
    pub sdk_version: String,
    /// Rust version
    pub rust_version: String,
    /// Active voice model
    pub active_voice: Option<String>,
    /// Device being used
    pub device: String,
    /// Configuration profile
    pub config_profile: Option<String>,
}

/// Performance impact metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImpact {
    /// Processing time increase in milliseconds
    pub processing_delay_ms: u64,
    /// Memory overhead in MB
    pub memory_overhead_mb: u32,
    /// Quality degradation score (0.0-1.0, higher is worse)
    pub quality_degradation: f32,
    /// Throughput impact percentage
    pub throughput_impact: f32,
}

/// Error statistics and aggregated metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStatistics {
    /// Total number of errors
    pub total_errors: u64,
    /// Errors by severity level
    pub errors_by_severity: HashMap<ErrorSeverity, u64>,
    /// Errors by component
    pub errors_by_component: HashMap<String, u64>,
    /// Errors by category
    pub errors_by_category: HashMap<ErrorCategory, u64>,
    /// Error rate per hour
    pub error_rate_per_hour: f64,
    /// Most common error types
    pub top_error_types: Vec<(String, u64)>,
    /// Average recovery time
    pub average_recovery_time: Duration,
    /// Recovery success rate
    pub recovery_success_rate: f32,
}

/// Error reporter for collecting and reporting errors
pub struct ErrorReporter {
    /// Configuration
    config: ErrorReporterConfig,
    /// Recent error reports
    recent_reports: Arc<Mutex<VecDeque<ErrorReport>>>,
    /// Error statistics
    statistics: Arc<Mutex<ErrorStatistics>>,
    /// Error listeners
    listeners: Vec<Box<dyn ErrorListener + Send + Sync>>,
    /// Start time for calculating error rates
    start_time: SystemTime,
}

/// Configuration for error reporter
#[derive(Debug, Clone)]
pub struct ErrorReporterConfig {
    /// Maximum number of recent reports to keep
    pub max_recent_reports: usize,
    /// Whether to collect stack traces
    pub collect_stack_traces: bool,
    /// Whether to collect system context
    pub collect_system_context: bool,
    /// Minimum severity level to report
    pub min_severity: ErrorSeverity,
    /// Whether to auto-report critical errors
    pub auto_report_critical: bool,
    /// Performance monitoring enabled
    pub performance_monitoring: bool,
}

impl Default for ErrorReporterConfig {
    fn default() -> Self {
        Self {
            max_recent_reports: 1000,
            collect_stack_traces: cfg!(debug_assertions),
            collect_system_context: true,
            min_severity: ErrorSeverity::Warning,
            auto_report_critical: true,
            performance_monitoring: true,
        }
    }
}

/// Trait for error listeners
pub trait ErrorListener {
    /// Handle an error report
    fn on_error(&self, report: &ErrorReport);

    /// Handle error statistics update
    fn on_statistics_update(&self, statistics: &ErrorStatistics);
}

impl ErrorReporter {
    /// Create a new error reporter
    pub fn new(config: ErrorReporterConfig) -> Self {
        Self {
            config,
            recent_reports: Arc::new(Mutex::new(VecDeque::new())),
            statistics: Arc::new(Mutex::new(ErrorStatistics::new())),
            listeners: Vec::new(),
            start_time: SystemTime::now(),
        }
    }

    /// Add an error listener
    pub fn add_listener<L: ErrorListener + Send + Sync + 'static>(&mut self, listener: L) {
        self.listeners.push(Box::new(listener));
    }

    /// Report an error
    pub fn report_error(&self, error: &VoirsError, context: Option<&str>) {
        if error.severity() < self.config.min_severity {
            return;
        }

        let report = self.create_error_report(error, context);

        // Update statistics
        self.update_statistics(&report);

        // Store recent report
        if let Ok(mut recent) = self.recent_reports.lock() {
            recent.push_back(report.clone());

            // Maintain max size
            while recent.len() > self.config.max_recent_reports {
                recent.pop_front();
            }
        }

        // Notify listeners
        for listener in &self.listeners {
            listener.on_error(&report);
        }

        // Auto-report critical errors
        if self.config.auto_report_critical && error.severity() >= ErrorSeverity::Critical {
            self.auto_report_critical_error(&report);
        }

        // Log error
        self.log_error(&report);
    }

    /// Report error with context
    pub fn report_error_with_context(&self, error_with_context: &ErrorWithContext) {
        if error_with_context.error.severity() < self.config.min_severity {
            return;
        }

        let mut report = self.create_error_report(&error_with_context.error, None);

        // Add context information
        report.component = error_with_context.context.component.clone();
        report.operation = error_with_context.context.operation.clone();
        report
            .metadata
            .extend(error_with_context.context.context.clone());

        if let Some(stack_trace) = &error_with_context.context.stack_trace {
            report.stack_trace = Some(stack_trace.clone());
        }

        // Update statistics
        self.update_statistics(&report);

        // Store and notify
        if let Ok(mut recent) = self.recent_reports.lock() {
            recent.push_back(report.clone());
            while recent.len() > self.config.max_recent_reports {
                recent.pop_front();
            }
        }

        for listener in &self.listeners {
            listener.on_error(&report);
        }

        self.log_error(&report);
    }

    /// Create error report from error
    fn create_error_report(&self, error: &VoirsError, context: Option<&str>) -> ErrorReport {
        let id = generate_error_id();
        let timestamp = SystemTime::now();

        ErrorReport {
            id,
            timestamp,
            severity: error.severity(),
            component: error.component().to_string(),
            operation: context.unwrap_or("unknown").to_string(),
            message: error.to_string(),
            category: self.categorize_error(error),
            system_context: if self.config.collect_system_context {
                collect_system_context()
            } else {
                SystemContext::default()
            },
            stack_trace: if self.config.collect_stack_traces {
                Some(std::backtrace::Backtrace::force_capture().to_string())
            } else {
                None
            },
            related_errors: Vec::new(),
            recovery_attempts: 0,
            recovered: false,
            performance_impact: if self.config.performance_monitoring {
                Some(estimate_performance_impact(error))
            } else {
                None
            },
            metadata: HashMap::new(),
        }
    }

    /// Categorize error by type
    fn categorize_error(&self, error: &VoirsError) -> ErrorCategory {
        match error {
            VoirsError::VoiceNotFound { .. }
            | VoirsError::InvalidConfiguration { .. }
            | VoirsError::ConfigError { .. } => ErrorCategory::UserError,

            VoirsError::OutOfMemory { .. }
            | VoirsError::GpuOutOfMemory { .. }
            | VoirsError::ResourceExhausted { .. } => ErrorCategory::ResourceError,

            VoirsError::ModelNotFound { .. }
            | VoirsError::ModelError { .. }
            | VoirsError::DownloadFailed { .. } => ErrorCategory::DependencyError,

            VoirsError::DeviceError { .. }
            | VoirsError::DeviceNotAvailable { .. }
            | VoirsError::UnsupportedDevice { .. } => ErrorCategory::HardwareError,

            VoirsError::NetworkError { .. } | VoirsError::AuthenticationFailed { .. } => {
                ErrorCategory::NetworkError
            }

            VoirsError::PerformanceDegradation { .. }
            | VoirsError::TimeoutError { .. }
            | VoirsError::RealTimeConstraintViolation { .. } => ErrorCategory::PerformanceError,

            VoirsError::InternalError { .. }
            | VoirsError::SynthesisFailed { .. }
            | VoirsError::ComponentSynchronizationFailed { .. } => ErrorCategory::SoftwareError,

            _ => ErrorCategory::SoftwareError,
        }
    }

    /// Update error statistics
    fn update_statistics(&self, report: &ErrorReport) {
        if let Ok(mut stats) = self.statistics.lock() {
            stats.total_errors += 1;

            // Update by severity
            *stats.errors_by_severity.entry(report.severity).or_insert(0) += 1;

            // Update by component
            *stats
                .errors_by_component
                .entry(report.component.clone())
                .or_insert(0) += 1;

            // Update by category
            *stats.errors_by_category.entry(report.category).or_insert(0) += 1;

            // Update error rate based on elapsed time
            let elapsed_hours = self
                .start_time
                .elapsed()
                .unwrap_or(Duration::from_secs(3600))
                .as_secs_f64()
                / 3600.0;
            stats.error_rate_per_hour = if elapsed_hours > 0.0 {
                stats.total_errors as f64 / elapsed_hours
            } else {
                0.0
            };

            // Notify listeners of statistics update
            for listener in &self.listeners {
                listener.on_statistics_update(&stats);
            }
        }
    }

    /// Auto-report critical error
    fn auto_report_critical_error(&self, report: &ErrorReport) {
        tracing::error!(
            "CRITICAL ERROR DETECTED: {} in {} during {}: {}",
            report.severity,
            report.component,
            report.operation,
            report.message
        );

        // In a real implementation, this could send alerts via:
        // - Email notifications
        // - Slack/Teams webhooks
        // - External monitoring systems
        // - SMS alerts
    }

    /// Log error with appropriate level
    fn log_error(&self, report: &ErrorReport) {
        match report.severity {
            ErrorSeverity::Fatal => {
                tracing::error!(
                    error_id = %report.id,
                    component = %report.component,
                    operation = %report.operation,
                    category = ?report.category,
                    "FATAL: {}", report.message
                );
            }
            ErrorSeverity::Critical => {
                tracing::error!(
                    error_id = %report.id,
                    component = %report.component,
                    operation = %report.operation,
                    category = ?report.category,
                    "CRITICAL: {}", report.message
                );
            }
            ErrorSeverity::Error => {
                tracing::error!(
                    error_id = %report.id,
                    component = %report.component,
                    operation = %report.operation,
                    category = ?report.category,
                    "{}", report.message
                );
            }
            ErrorSeverity::Warning => {
                tracing::warn!(
                    error_id = %report.id,
                    component = %report.component,
                    operation = %report.operation,
                    category = ?report.category,
                    "{}", report.message
                );
            }
            ErrorSeverity::Info => {
                tracing::info!(
                    error_id = %report.id,
                    component = %report.component,
                    operation = %report.operation,
                    category = ?report.category,
                    "{}", report.message
                );
            }
        }
    }

    /// Get recent error reports
    pub fn get_recent_reports(&self) -> Vec<ErrorReport> {
        self.recent_reports
            .lock()
            .unwrap()
            .iter()
            .cloned()
            .collect()
    }

    /// Get error statistics
    pub fn get_statistics(&self) -> ErrorStatistics {
        self.statistics.lock().unwrap().clone()
    }

    /// Get reports by component
    pub fn get_reports_by_component(&self, component: &str) -> Vec<ErrorReport> {
        self.recent_reports
            .lock()
            .unwrap()
            .iter()
            .filter(|report| report.component == component)
            .cloned()
            .collect()
    }

    /// Get reports by severity
    pub fn get_reports_by_severity(&self, severity: ErrorSeverity) -> Vec<ErrorReport> {
        self.recent_reports
            .lock()
            .unwrap()
            .iter()
            .filter(|report| report.severity == severity)
            .cloned()
            .collect()
    }

    /// Generate diagnostic report
    pub fn generate_diagnostic_report(&self) -> DiagnosticReport {
        let statistics = self.get_statistics();
        let recent_reports = self.get_recent_reports();
        let system_context = collect_system_context();

        DiagnosticReport {
            timestamp: SystemTime::now(),
            statistics: statistics.clone(),
            recent_critical_errors: recent_reports
                .into_iter()
                .filter(|r| r.severity >= ErrorSeverity::Critical)
                .collect(),
            system_health: assess_system_health(&system_context, &statistics),
            recommendations: generate_recommendations(&statistics),
        }
    }

    /// Clear all reports and statistics
    pub fn clear(&self) {
        if let Ok(mut recent) = self.recent_reports.lock() {
            recent.clear();
        }
        if let Ok(mut stats) = self.statistics.lock() {
            *stats = ErrorStatistics::new();
        }
    }
}

impl Default for ErrorReporter {
    fn default() -> Self {
        Self::new(ErrorReporterConfig::default())
    }
}

/// Diagnostic report containing system health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticReport {
    /// Report timestamp
    pub timestamp: SystemTime,
    /// Error statistics
    pub statistics: ErrorStatistics,
    /// Recent critical errors
    pub recent_critical_errors: Vec<ErrorReport>,
    /// System health assessment
    pub system_health: SystemHealth,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// System health assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    /// Overall health score (0.0-1.0)
    pub overall_score: f32,
    /// Memory health score
    pub memory_score: f32,
    /// Performance health score
    pub performance_score: f32,
    /// Error rate health score
    pub error_rate_score: f32,
    /// Resource utilization score
    pub resource_score: f32,
}

impl ErrorStatistics {
    fn new() -> Self {
        Self {
            total_errors: 0,
            errors_by_severity: HashMap::new(),
            errors_by_component: HashMap::new(),
            errors_by_category: HashMap::new(),
            error_rate_per_hour: 0.0,
            top_error_types: Vec::new(),
            average_recovery_time: Duration::from_secs(0),
            recovery_success_rate: 0.0,
        }
    }
}

impl Default for SystemContext {
    fn default() -> Self {
        Self {
            available_memory_mb: 0,
            cpu_usage: 0.0,
            gpu_memory_mb: None,
            active_threads: 0,
            load_average: None,
            os_info: "unknown".to_string(),
            runtime_info: RuntimeInfo::default(),
        }
    }
}

impl Default for RuntimeInfo {
    fn default() -> Self {
        Self {
            sdk_version: env!("CARGO_PKG_VERSION").to_string(),
            rust_version: "unknown".to_string(),
            active_voice: None,
            device: "cpu".to_string(),
            config_profile: None,
        }
    }
}

/// Console error listener for development
pub struct ConsoleErrorListener;

impl ErrorListener for ConsoleErrorListener {
    fn on_error(&self, report: &ErrorReport) {
        eprintln!(
            "[{}] {} in {}: {}",
            report.severity, report.component, report.operation, report.message
        );
    }

    fn on_statistics_update(&self, statistics: &ErrorStatistics) {
        if statistics.total_errors % 10 == 0 {
            eprintln!("Error statistics: {} total errors", statistics.total_errors);
        }
    }
}

/// File error listener for logging to file
pub struct FileErrorListener {
    file_path: std::path::PathBuf,
}

impl FileErrorListener {
    pub fn new(file_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            file_path: file_path.into(),
        }
    }
}

impl ErrorListener for FileErrorListener {
    fn on_error(&self, report: &ErrorReport) {
        if let Ok(json) = serde_json::to_string(report) {
            let _ = std::fs::write(&self.file_path, format!("{json}\n"));
        }
    }

    fn on_statistics_update(&self, _statistics: &ErrorStatistics) {
        // File listener doesn't handle statistics updates
    }
}

/// Utility functions
fn generate_error_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("err_{timestamp}")
}

fn collect_system_context() -> SystemContext {
    SystemContext {
        available_memory_mb: get_available_memory_mb(),
        cpu_usage: get_cpu_usage(),
        gpu_memory_mb: get_gpu_memory_mb(),
        active_threads: get_active_threads(),
        load_average: get_load_average(),
        os_info: get_os_info(),
        runtime_info: get_runtime_info(),
    }
}

fn get_available_memory_mb() -> u32 {
    #[cfg(target_os = "linux")]
    {
        // Parse /proc/meminfo on Linux
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemAvailable:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u32>() {
                            return kb / 1024; // Convert KB to MB
                        }
                    }
                }
            }
        }
        // Fallback: try to get total memory
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u32>() {
                            return (kb / 1024) * 7 / 10; // Estimate 70% available
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        // Use vm_stat on macOS
        if let Ok(output) = std::process::Command::new("vm_stat").output() {
            if let Ok(vm_stat) = String::from_utf8(output.stdout) {
                let mut page_size = 4096; // Default page size
                let mut free_pages = 0;
                let mut inactive_pages = 0;

                for line in vm_stat.lines() {
                    if line.starts_with("Mach Virtual Memory Statistics:") {
                        // Extract page size if available
                        if line.contains("page size of ") {
                            if let Some(start) = line.find("page size of ") {
                                let page_str = &line[start + 13..];
                                if let Some(end) = page_str.find(' ') {
                                    if let Ok(size) = page_str[..end].parse::<u32>() {
                                        page_size = size;
                                    }
                                }
                            }
                        }
                    } else if line.contains("Pages free:") {
                        if let Some(pages_str) = line.split(':').nth(1) {
                            if let Some(num_str) = pages_str.trim().split('.').next() {
                                if let Ok(pages) = num_str.parse::<u32>() {
                                    free_pages = pages;
                                }
                            }
                        }
                    } else if line.contains("Pages inactive:") {
                        if let Some(pages_str) = line.split(':').nth(1) {
                            if let Some(num_str) = pages_str.trim().split('.').next() {
                                if let Ok(pages) = num_str.parse::<u32>() {
                                    inactive_pages = pages;
                                }
                            }
                        }
                    }
                }

                let available_bytes = (free_pages + inactive_pages) as u64 * page_size as u64;
                return (available_bytes / (1024 * 1024)) as u32; // Convert to MB
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Fallback for Windows - would need Windows API calls
        // For now, return a reasonable default
        return 2048;
    }

    // Fallback for unknown platforms
    1024
}

fn get_cpu_usage() -> f32 {
    #[cfg(target_os = "linux")]
    {
        // Parse /proc/stat on Linux for instant CPU usage approximation
        if let Ok(stat) = std::fs::read_to_string("/proc/stat") {
            if let Some(cpu_line) = stat.lines().next() {
                if cpu_line.starts_with("cpu ") {
                    let parts: Vec<&str> = cpu_line.split_whitespace().collect();
                    if parts.len() >= 5 {
                        // cpu user nice system idle iowait irq softirq...
                        if let (Ok(user), Ok(nice), Ok(system), Ok(idle)) = (
                            parts[1].parse::<u64>(),
                            parts[2].parse::<u64>(),
                            parts[3].parse::<u64>(),
                            parts[4].parse::<u64>(),
                        ) {
                            let active = user + nice + system;
                            let total = active + idle;
                            if total > 0 {
                                return (active as f32 / total as f32) * 100.0;
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        // Use top command to get CPU usage on macOS
        if let Ok(output) = std::process::Command::new("top")
            .args(["-l", "1", "-n", "0"])
            .output()
        {
            if let Ok(top_output) = String::from_utf8(output.stdout) {
                for line in top_output.lines() {
                    if line.contains("CPU usage:") {
                        // Look for pattern like "CPU usage: 12.5% user, 4.2% sys, 83.3% idle"
                        if let Some(user_start) = line.find("% user") {
                            let before_user = &line[..user_start];
                            if let Some(space_pos) = before_user.rfind(' ') {
                                if let Ok(user_pct) = before_user[space_pos + 1..].parse::<f32>() {
                                    if let Some(sys_start) = line.find("% sys") {
                                        let between = &line[user_start + 6..sys_start];
                                        if let Some(sys_space) = between.rfind(' ') {
                                            if let Ok(sys_pct) =
                                                between[sys_space + 1..].parse::<f32>()
                                            {
                                                return user_pct + sys_pct; // Total active CPU
                                            }
                                        }
                                    }
                                    return user_pct; // At least user CPU
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Fallback for Windows - would need Windows API calls
        // Return a reasonable estimate
        return 25.0;
    }

    // Fallback: return low usage estimate
    10.0
}

fn get_gpu_memory_mb() -> Option<u32> {
    // Try to get NVIDIA GPU memory using nvidia-smi
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
    {
        if output.status.success() {
            if let Ok(nvidia_output) = String::from_utf8(output.stdout) {
                if let Some(memory_line) = nvidia_output.lines().next() {
                    if let Ok(memory_mb) = memory_line.trim().parse::<u32>() {
                        return Some(memory_mb);
                    }
                }
            }
        }
    }

    // Try to detect AMD GPU memory (basic attempt)
    #[cfg(target_os = "linux")]
    {
        // Check /sys/class/drm for AMD GPUs
        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("card") && !name.contains('-') {
                        let mem_info_path = path.join("device/mem_info_vram_total");
                        if let Ok(mem_info) = std::fs::read_to_string(mem_info_path) {
                            if let Ok(bytes) = mem_info.trim().parse::<u64>() {
                                return Some((bytes / (1024 * 1024)) as u32);
                            }
                        }
                    }
                }
            }
        }
    }

    // Try to detect integrated GPU memory (estimates)
    #[cfg(target_os = "macos")]
    {
        // On macOS with Apple Silicon, GPU memory is unified with system memory
        if cfg!(target_arch = "aarch64") {
            // Estimate GPU-accessible memory as a portion of system memory
            let system_memory = get_available_memory_mb();
            if system_memory > 8192 {
                return Some(system_memory / 2); // Rough estimate
            }
        }
    }

    // Could not detect GPU memory
    None
}

fn get_active_threads() -> u32 {
    #[cfg(target_os = "linux")]
    {
        // Count threads in /proc/self/status
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("Threads:") {
                    if let Some(threads_str) = line.split_whitespace().nth(1) {
                        if let Ok(threads) = threads_str.parse::<u32>() {
                            return threads;
                        }
                    }
                }
            }
        }

        // Fallback: count entries in /proc/self/task/
        if let Ok(entries) = std::fs::read_dir("/proc/self/task") {
            return entries.count() as u32;
        }
    }

    #[cfg(target_os = "macos")]
    {
        // Use ps command to count threads
        if let Ok(output) = std::process::Command::new("ps")
            .args(["-M", "-p", &std::process::id().to_string()])
            .output()
        {
            if let Ok(ps_output) = String::from_utf8(output.stdout) {
                // Count lines (excluding header)
                let line_count = ps_output.lines().count();
                if line_count > 1 {
                    return (line_count - 1) as u32;
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Fallback for Windows - would need Windows API calls
        // Return reasonable estimate
        return 4;
    }

    // Fallback: estimate based on CPU cores
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
}

fn get_load_average() -> Option<f32> {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        // Read /proc/loadavg on Linux or use uptime on macOS
        #[cfg(target_os = "linux")]
        {
            if let Ok(loadavg) = std::fs::read_to_string("/proc/loadavg") {
                // Format: "0.15 0.20 0.25 1/234 5678"
                if let Some(first_load) = loadavg.split_whitespace().next() {
                    if let Ok(load) = first_load.parse::<f32>() {
                        return Some(load);
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            // Use uptime command on macOS
            if let Ok(output) = std::process::Command::new("uptime").output() {
                if let Ok(uptime_output) = String::from_utf8(output.stdout) {
                    // Look for pattern like "load averages: 1.23 2.34 3.45"
                    if let Some(load_start) = uptime_output.find("load averages:") {
                        let load_part = &uptime_output[load_start + 14..];
                        if let Some(first_load) = load_part.split_whitespace().next() {
                            if let Ok(load) = first_load.parse::<f32>() {
                                return Some(load);
                            }
                        }
                    }
                    // Alternative pattern: "load average: 1.23, 2.34, 3.45"
                    if let Some(load_start) = uptime_output.find("load average:") {
                        let load_part = &uptime_output[load_start + 13..];
                        if let Some(first_load) = load_part.trim().split(',').next() {
                            if let Ok(load) = first_load.trim().parse::<f32>() {
                                return Some(load);
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Windows doesn't have load average concept
        // Could approximate with CPU usage over time, but returning None for now
        return None;
    }

    None
}

fn get_os_info() -> String {
    std::env::consts::OS.to_string()
}

fn get_runtime_info() -> RuntimeInfo {
    RuntimeInfo::default()
}

fn estimate_performance_impact(error: &VoirsError) -> PerformanceImpact {
    match error {
        VoirsError::OutOfMemory { .. } => PerformanceImpact {
            processing_delay_ms: 5000,
            memory_overhead_mb: 100,
            quality_degradation: 0.8,
            throughput_impact: 90.0,
        },
        VoirsError::TimeoutError { .. } => PerformanceImpact {
            processing_delay_ms: 1000,
            memory_overhead_mb: 0,
            quality_degradation: 0.2,
            throughput_impact: 50.0,
        },
        VoirsError::DeviceError { .. } => PerformanceImpact {
            processing_delay_ms: 2000,
            memory_overhead_mb: 50,
            quality_degradation: 0.3,
            throughput_impact: 70.0,
        },
        _ => PerformanceImpact {
            processing_delay_ms: 100,
            memory_overhead_mb: 10,
            quality_degradation: 0.1,
            throughput_impact: 10.0,
        },
    }
}

fn assess_system_health(context: &SystemContext, statistics: &ErrorStatistics) -> SystemHealth {
    let memory_score = if context.available_memory_mb > 1000 {
        1.0
    } else {
        0.5
    };
    let performance_score = 1.0 - (context.cpu_usage / 100.0);
    // Calculate error rate score (lower error rate = higher score)
    let error_rate_score = if statistics.error_rate_per_hour > 0.0 {
        // Score decreases as error rate increases
        (1.0f32 / (1.0f32 + statistics.error_rate_per_hour as f32 * 0.1f32)).max(0.1f32)
    } else {
        1.0f32
    };
    let resource_score = memory_score * performance_score;
    let overall_score =
        (memory_score + performance_score + error_rate_score + resource_score) / 4.0;

    SystemHealth {
        overall_score,
        memory_score,
        performance_score,
        error_rate_score,
        resource_score,
    }
}

fn generate_recommendations(statistics: &ErrorStatistics) -> Vec<String> {
    let mut recommendations = Vec::new();

    if statistics.total_errors > 100 {
        recommendations
            .push("High error count detected. Consider reviewing error patterns.".to_string());
    }

    if let Some(critical_count) = statistics.errors_by_severity.get(&ErrorSeverity::Critical) {
        if *critical_count > 5 {
            recommendations.push(
                "Multiple critical errors detected. Immediate attention required.".to_string(),
            );
        }
    }

    if statistics.recovery_success_rate < 0.8 {
        recommendations.push("Low recovery success rate. Review recovery strategies.".to_string());
    }

    if recommendations.is_empty() {
        recommendations.push("System operating within normal parameters.".to_string());
    }

    recommendations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_reporter() {
        let mut reporter = ErrorReporter::new(ErrorReporterConfig::default());

        // Add console listener
        reporter.add_listener(ConsoleErrorListener);

        // Report an error
        let error = VoirsError::InternalError {
            component: "test".to_string(),
            message: "test error".to_string(),
        };

        reporter.report_error(&error, Some("test_operation"));

        // Check statistics
        let stats = reporter.get_statistics();
        assert_eq!(stats.total_errors, 1);
        assert_eq!(stats.errors_by_component.get("test"), Some(&1));

        // Check recent reports
        let reports = reporter.get_recent_reports();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].component, "test");
    }

    #[test]
    fn test_error_categorization() {
        let reporter = ErrorReporter::default();

        let config_error = VoirsError::ConfigError {
            field: "test".to_string(),
            message: "test".to_string(),
        };
        assert_eq!(
            reporter.categorize_error(&config_error),
            ErrorCategory::UserError
        );

        let memory_error = VoirsError::OutOfMemory {
            message: "test".to_string(),
            requested_mb: 100,
        };
        assert_eq!(
            reporter.categorize_error(&memory_error),
            ErrorCategory::ResourceError
        );
    }

    #[test]
    fn test_diagnostic_report() {
        let reporter = ErrorReporter::default();

        // Report some errors
        reporter.report_error(
            &VoirsError::InternalError {
                component: "test".to_string(),
                message: "test".to_string(),
            },
            None,
        );

        let diagnostic = reporter.generate_diagnostic_report();
        assert_eq!(diagnostic.statistics.total_errors, 1);
        assert!(!diagnostic.recommendations.is_empty());
    }
}
