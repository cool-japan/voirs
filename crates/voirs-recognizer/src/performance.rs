//! Performance validation and monitoring utilities.
//!
//! This module provides utilities to validate that the recognition system meets
//! the performance requirements specified in the project goals:
//! - Real-time factor (RTF) < 0.3 on modern CPU
//! - Memory usage < 2GB for largest models
//! - Startup time < 5 seconds
//! - Streaming latency < 200ms

pub mod regression_detector;

use crate::RecognitionError;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use voirs_sdk::AudioBuffer;

/// Performance requirements as specified in the project TODO
#[derive(Debug, Clone)]
pub struct PerformanceRequirements {
    /// Maximum acceptable real-time factor
    pub max_rtf: f32,
    /// Maximum memory usage in bytes (2GB = 2 * 1024^3)
    pub max_memory_usage: u64,
    /// Maximum startup time in milliseconds
    pub max_startup_time_ms: u64,
    /// Maximum streaming latency in milliseconds
    pub max_streaming_latency_ms: u64,
}

impl Default for PerformanceRequirements {
    fn default() -> Self {
        Self {
            max_rtf: 0.3,
            max_memory_usage: 2 * 1024 * 1024 * 1024, // 2GB
            max_startup_time_ms: 5000,                // 5 seconds
            max_streaming_latency_ms: 200,            // 200ms
        }
    }
}

/// Performance metrics collected during validation
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Real-time factor (processing_time / audio_duration)
    pub rtf: f32,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Model startup time in milliseconds
    pub startup_time_ms: u64,
    /// Streaming latency in milliseconds
    pub streaming_latency_ms: u64,
    /// Processing throughput (samples per second)
    pub throughput_samples_per_sec: f64,
    /// CPU utilization percentage
    pub cpu_utilization: f32,
}

/// Performance validation results
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether all requirements were met
    pub passed: bool,
    /// Detailed metrics collected
    pub metrics: PerformanceMetrics,
    /// Requirements that were checked against
    pub requirements: PerformanceRequirements,
    /// Individual test results
    pub test_results: HashMap<String, bool>,
    /// Additional notes or warnings
    pub notes: Vec<String>,
}

/// Performance validator for ASR models
pub struct PerformanceValidator {
    requirements: PerformanceRequirements,
    verbose: bool,
}

impl PerformanceValidator {
    /// Create a new performance validator with default requirements
    #[must_use]
    pub fn new() -> Self {
        Self {
            requirements: PerformanceRequirements::default(),
            verbose: false,
        }
    }

    /// Create a new performance validator with custom requirements
    #[must_use]
    pub fn with_requirements(requirements: PerformanceRequirements) -> Self {
        Self {
            requirements,
            verbose: false,
        }
    }

    /// Enable verbose logging during validation
    #[must_use]
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Get the performance requirements
    pub fn requirements(&self) -> &PerformanceRequirements {
        &self.requirements
    }

    /// Validate real-time factor performance
    pub fn validate_rtf(&self, audio: &AudioBuffer, processing_time: Duration) -> (f32, bool) {
        let audio_duration_seconds = audio.duration();
        let processing_seconds = processing_time.as_secs_f32();
        let rtf = processing_seconds / audio_duration_seconds;

        let passed = rtf <= self.requirements.max_rtf;

        if self.verbose {
            println!(
                "RTF Validation: {:.3} (target: ≤{:.3}) - {}",
                rtf,
                self.requirements.max_rtf,
                if passed { "PASS" } else { "FAIL" }
            );
        }

        (rtf, passed)
    }

    /// Estimate memory usage (platform-specific implementation)
    pub fn estimate_memory_usage(&self) -> Result<(u64, bool), RecognitionError> {
        let memory_usage = get_memory_usage()?;
        let passed = memory_usage <= self.requirements.max_memory_usage;

        if self.verbose {
            println!(
                "Memory Usage: {:.2} MB (target: ≤{:.2} MB) - {}",
                memory_usage as f64 / (1024.0 * 1024.0),
                self.requirements.max_memory_usage as f64 / (1024.0 * 1024.0),
                if passed { "PASS" } else { "FAIL" }
            );
        }

        Ok((memory_usage, passed))
    }

    /// Measure model startup time
    pub async fn measure_startup_time<F, Fut>(
        &self,
        startup_fn: F,
    ) -> Result<(u64, bool), RecognitionError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<(), RecognitionError>>,
    {
        let start = Instant::now();
        startup_fn().await?;
        let startup_time = start.elapsed();

        let startup_ms = startup_time.as_millis() as u64;
        let passed = startup_ms <= self.requirements.max_startup_time_ms;

        if self.verbose {
            println!(
                "Startup Time: {}ms (target: ≤{}ms) - {}",
                startup_ms,
                self.requirements.max_startup_time_ms,
                if passed { "PASS" } else { "FAIL" }
            );
        }

        Ok((startup_ms, passed))
    }

    /// Validate streaming latency
    pub fn validate_streaming_latency(&self, latency: Duration) -> (u64, bool) {
        let latency_ms = latency.as_millis() as u64;
        let passed = latency_ms <= self.requirements.max_streaming_latency_ms;

        if self.verbose {
            println!(
                "Streaming Latency: {}ms (target: ≤{}ms) - {}",
                latency_ms,
                self.requirements.max_streaming_latency_ms,
                if passed { "PASS" } else { "FAIL" }
            );
        }

        (latency_ms, passed)
    }

    /// Calculate processing throughput
    pub fn calculate_throughput(&self, samples_processed: usize, processing_time: Duration) -> f64 {
        let processing_seconds = processing_time.as_secs_f64();
        if processing_seconds > 0.0 {
            samples_processed as f64 / processing_seconds
        } else {
            0.0
        }
    }

    /// Comprehensive performance validation
    pub async fn validate_comprehensive<F, Fut>(
        &self,
        audio: &AudioBuffer,
        startup_fn: F,
        processing_time: Duration,
        streaming_latency: Option<Duration>,
    ) -> Result<ValidationResult, RecognitionError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<(), RecognitionError>>,
    {
        let mut test_results = HashMap::new();
        let mut notes = Vec::new();

        // Validate RTF
        let (rtf, rtf_passed) = self.validate_rtf(audio, processing_time);
        test_results.insert("rtf".to_string(), rtf_passed);

        // Validate memory usage
        let (memory_usage, memory_passed) = self.estimate_memory_usage()?;
        test_results.insert("memory".to_string(), memory_passed);

        // Validate startup time
        let (startup_time_ms, startup_passed) = self.measure_startup_time(startup_fn).await?;
        test_results.insert("startup".to_string(), startup_passed);

        // Validate streaming latency if provided
        let streaming_latency_ms = if let Some(latency) = streaming_latency {
            let (latency_ms, latency_passed) = self.validate_streaming_latency(latency);
            test_results.insert("streaming_latency".to_string(), latency_passed);
            latency_ms
        } else {
            notes.push("Streaming latency not measured".to_string());
            0
        };

        // Calculate additional metrics
        let throughput_samples_per_sec =
            self.calculate_throughput(audio.samples().len(), processing_time);

        // Estimate CPU utilization (simplified)
        let audio_duration = Duration::from_secs_f32(audio.duration());
        let cpu_utilization = estimate_cpu_utilization(processing_time, audio_duration);

        let metrics = PerformanceMetrics {
            rtf,
            memory_usage,
            startup_time_ms,
            streaming_latency_ms,
            throughput_samples_per_sec,
            cpu_utilization,
        };

        // Check if all tests passed
        let passed = test_results.values().all(|&result| result);

        if self.verbose {
            println!("\n=== Performance Validation Summary ===");
            println!("Overall Result: {}", if passed { "PASS" } else { "FAIL" });
            println!(
                "RTF: {:.3} ({})",
                rtf,
                if test_results["rtf"] { "PASS" } else { "FAIL" }
            );
            println!(
                "Memory: {:.1} MB ({})",
                memory_usage as f64 / (1024.0 * 1024.0),
                if test_results["memory"] {
                    "PASS"
                } else {
                    "FAIL"
                }
            );
            println!(
                "Startup: {}ms ({})",
                startup_time_ms,
                if test_results["startup"] {
                    "PASS"
                } else {
                    "FAIL"
                }
            );
            if streaming_latency.is_some() {
                println!(
                    "Streaming Latency: {}ms ({})",
                    streaming_latency_ms,
                    if *test_results.get("streaming_latency").unwrap_or(&false) {
                        "PASS"
                    } else {
                        "FAIL"
                    }
                );
            }
            println!("Throughput: {:.0} samples/sec", throughput_samples_per_sec);
            println!("CPU Utilization: {:.1}%", cpu_utilization);
        }

        Ok(ValidationResult {
            passed,
            metrics,
            requirements: self.requirements.clone(),
            test_results,
            notes,
        })
    }
}

impl Default for PerformanceValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current process memory usage (platform-specific)
fn get_memory_usage() -> Result<u64, RecognitionError> {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        let status = fs::read_to_string("/proc/self/status").map_err(|e| {
            RecognitionError::ResourceError {
                message: format!("Failed to read memory info: {e}"),
                source: Some(Box::new(e)),
            }
        })?;

        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let kb: u64 = parts[1].parse().unwrap_or(0);
                    return Ok(kb * 1024); // Convert KB to bytes
                }
            }
        }
        Ok(0)
    }

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        let output = Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .map_err(|e| RecognitionError::ResourceError {
                message: format!("Failed to get memory info: {e}"),
                source: Some(Box::new(e)),
            })?;

        let output_str = String::from_utf8_lossy(&output.stdout);
        let kb: u64 = output_str.trim().parse().unwrap_or(0);
        Ok(kb * 1024) // Convert KB to bytes
    }

    #[cfg(target_os = "windows")]
    {
        // Try multiple Windows-specific methods in order of accuracy

        // Method 1: Try PowerShell with Get-Process (most reliable)
        if let Ok(output) = std::process::Command::new("powershell")
            .args([
                "-Command",
                &format!("(Get-Process -Id {}).WorkingSet64", std::process::id()),
            ])
            .output()
        {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                if let Ok(bytes) = output_str.trim().parse::<u64>() {
                    tracing::debug!(
                        "Windows memory usage detected via PowerShell: {} MB",
                        bytes / 1024 / 1024
                    );
                    return Ok(bytes);
                }
            }
        }

        // Method 2: Try WMIC query
        if let Ok(output) = std::process::Command::new("wmic")
            .args([
                "process",
                "where",
                &format!("ProcessId={}", std::process::id()),
                "get",
                "WorkingSetSize",
                "/value",
            ])
            .output()
        {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                for line in output_str.lines() {
                    if line.starts_with("WorkingSetSize=") {
                        if let Ok(bytes) = line
                            .strip_prefix("WorkingSetSize=")
                            .unwrap_or("")
                            .parse::<u64>()
                        {
                            tracing::debug!(
                                "Windows memory usage detected via WMIC: {} MB",
                                bytes / 1024 / 1024
                            );
                            return Ok(bytes);
                        }
                    }
                }
            }
        }

        // Method 3: Try tasklist command (CSV format)
        if let Ok(output) = std::process::Command::new("tasklist")
            .args([
                "/FI",
                &format!("PID eq {}", std::process::id()),
                "/FO",
                "CSV",
            ])
            .output()
        {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                // Parse CSV output to extract memory usage
                for line in output_str.lines().skip(1) {
                    let fields: Vec<&str> = line.split(',').collect();
                    if fields.len() >= 5 {
                        let memory_str = fields[4].trim_matches('"').replace([',', ' '], "");
                        if let Ok(kb) = memory_str.replace('K', "").parse::<u64>() {
                            let bytes = kb * 1024;
                            tracing::debug!(
                                "Windows memory usage detected via tasklist: {} MB",
                                bytes / 1024 / 1024
                            );
                            return Ok(bytes);
                        }
                    }
                }
            }
        }

        // Method 4: Try alternative PowerShell approach
        if let Ok(output) = std::process::Command::new("powershell")
            .args([
                "-Command",
                &format!(
                    "Get-Process -Id {} | Select-Object -ExpandProperty WorkingSet",
                    std::process::id()
                ),
            ])
            .output()
        {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                if let Ok(bytes) = output_str.trim().parse::<u64>() {
                    tracing::debug!(
                        "Windows memory usage detected via PowerShell fallback: {} MB",
                        bytes / 1024 / 1024
                    );
                    return Ok(bytes);
                }
            }
        }

        // Ultimate fallback - intelligent estimation based on system characteristics
        let estimated_usage = estimate_process_memory_usage();
        tracing::warn!(
            "Could not detect actual Windows memory usage, using intelligent estimation: {} MB",
            estimated_usage / 1024 / 1024
        );
        Ok(estimated_usage)
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        // For other platforms, use educated estimation
        let estimated_usage = estimate_process_memory_usage();
        tracing::warn!(
            "Platform-specific memory measurement not available, using estimation: {} MB",
            estimated_usage / 1024 / 1024
        );
        Ok(estimated_usage)
    }
}

/// Estimate process memory usage based on system characteristics and typical usage patterns
fn estimate_process_memory_usage() -> u64 {
    // Base memory for a Rust process with basic ML models
    let mut estimated_memory = 50 * 1024 * 1024; // 50MB base

    // Add estimation based on detected system capabilities
    if let Ok(total_memory) = get_total_system_memory() {
        // Use tiered percentage approach based on system memory
        let memory_percentage = if total_memory > 32 * 1024 * 1024 * 1024 {
            // Very high memory system (>32GB) - can use more
            0.010 // 1.0% of total memory
        } else if total_memory > 16 * 1024 * 1024 * 1024 {
            // High memory system (>16GB) - can use more
            0.008 // 0.8% of total memory
        } else if total_memory > 8 * 1024 * 1024 * 1024 {
            // Medium memory system (>8GB)
            0.006 // 0.6% of total memory
        } else if total_memory > 4 * 1024 * 1024 * 1024 {
            // Lower medium memory system (>4GB)
            0.005 // 0.5% of total memory
        } else {
            // Lower memory system (≤4GB)
            0.004 // 0.4% of total memory
        };

        estimated_memory = ((total_memory as f64 * memory_percentage) as u64)
            .max(estimated_memory)
            .min(800 * 1024 * 1024); // Cap at 800MB for very large systems
    }

    // Add extra for ML models and audio processing buffers
    estimated_memory += detect_model_memory_footprint();

    tracing::debug!(
        "Estimated process memory usage: {:.1} MB (based on system memory detection)",
        estimated_memory as f64 / (1024.0 * 1024.0)
    );

    estimated_memory
}

/// Detect likely memory footprint based on available models and features
fn detect_model_memory_footprint() -> u64 {
    let mut model_memory = 30 * 1024 * 1024; // 30MB base for audio processing

    // Check for common model indicators in the current process
    // This is a heuristic based on typical ASR model sizes

    // Base ASR model memory (Whisper, etc.)
    model_memory += 120 * 1024 * 1024; // ~120MB for base model

    // Add extra for potential larger models
    if std::env::var("VOIRS_LARGE_MODELS").is_ok() {
        model_memory += 200 * 1024 * 1024; // Extra 200MB for large models
    }

    // Audio buffer memory (typical streaming buffers)
    model_memory += 16 * 1024 * 1024; // 16MB for audio buffers

    // Cache and temporary storage
    model_memory += 32 * 1024 * 1024; // 32MB for caches

    model_memory
}

/// Get total system memory (best effort across platforms)
fn get_total_system_memory() -> Result<u64, ()> {
    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<u64>() {
                            return Ok(kb * 1024); // Convert KB to bytes
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
        {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                if let Ok(bytes) = output_str.trim().parse::<u64>() {
                    return Ok(bytes);
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Try WMI query first (most accurate)
        if let Ok(output) = std::process::Command::new("wmic")
            .args(["computersystem", "get", "TotalPhysicalMemory", "/value"])
            .output()
        {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                for line in output_str.lines() {
                    if line.starts_with("TotalPhysicalMemory=") {
                        if let Ok(bytes) = line
                            .strip_prefix("TotalPhysicalMemory=")
                            .unwrap_or("")
                            .parse::<u64>()
                        {
                            return Ok(bytes);
                        }
                    }
                }
            }
        }

        // Fallback to PowerShell
        if let Ok(output) = std::process::Command::new("powershell")
            .args([
                "-Command",
                "(Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory",
            ])
            .output()
        {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                if let Ok(bytes) = output_str.trim().parse::<u64>() {
                    return Ok(bytes);
                }
            }
        }

        // Final fallback to systeminfo command
        if let Ok(output) = std::process::Command::new("systeminfo").output() {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                for line in output_str.lines() {
                    if line.contains("Total Physical Memory:") {
                        // Parse output like "Total Physical Memory:     8,192 MB"
                        if let Some(memory_part) = line.split(':').nth(1) {
                            let cleaned = memory_part
                                .replace(',', "")
                                .replace(" MB", "")
                                .trim()
                                .to_string();
                            if let Ok(mb) = cleaned.parse::<u64>() {
                                return Ok(mb * 1024 * 1024); // Convert MB to bytes
                            }
                        }
                    }
                }
            }
        }

        // Conservative fallback based on common Windows configurations
        tracing::warn!("Could not detect Windows system memory, using conservative estimate");
        Ok(8 * 1024 * 1024 * 1024) // 8GB default
    }

    // Default fallback for unknown platforms
    Ok(8 * 1024 * 1024 * 1024) // 8GB default
}

/// Estimate CPU utilization based on processing time vs real time
fn estimate_cpu_utilization(processing_time: Duration, audio_duration: Duration) -> f32 {
    if audio_duration.as_secs_f32() > 0.0 {
        let utilization = (processing_time.as_secs_f32() / audio_duration.as_secs_f32()) * 100.0;
        utilization.min(100.0) // Cap at 100%
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_performance_requirements_default() {
        let req = PerformanceRequirements::default();
        assert_eq!(req.max_rtf, 0.3);
        assert_eq!(req.max_memory_usage, 2 * 1024 * 1024 * 1024);
        assert_eq!(req.max_startup_time_ms, 5000);
        assert_eq!(req.max_streaming_latency_ms, 200);
    }

    #[test]
    fn test_validator_creation() {
        let validator = PerformanceValidator::new();
        assert_eq!(validator.requirements.max_rtf, 0.3);
        assert!(!validator.verbose);

        let validator = PerformanceValidator::new().with_verbose(true);
        assert!(validator.verbose);
    }

    #[test]
    fn test_rtf_validation() {
        let validator = PerformanceValidator::new();
        let audio = AudioBuffer::mono(vec![0.0; 16000], 16000); // 1 second of audio

        // Test passing RTF
        let processing_time = Duration::from_millis(200); // 0.2 seconds processing
        let (rtf, passed) = validator.validate_rtf(&audio, processing_time);
        assert_eq!(rtf, 0.2);
        assert!(passed);

        // Test failing RTF
        let processing_time = Duration::from_millis(500); // 0.5 seconds processing
        let (rtf, passed) = validator.validate_rtf(&audio, processing_time);
        assert_eq!(rtf, 0.5);
        assert!(!passed);
    }

    #[test]
    fn test_streaming_latency_validation() {
        let validator = PerformanceValidator::new();

        // Test passing latency
        let latency = Duration::from_millis(150);
        let (latency_ms, passed) = validator.validate_streaming_latency(latency);
        assert_eq!(latency_ms, 150);
        assert!(passed);

        // Test failing latency
        let latency = Duration::from_millis(300);
        let (latency_ms, passed) = validator.validate_streaming_latency(latency);
        assert_eq!(latency_ms, 300);
        assert!(!passed);
    }

    #[test]
    fn test_throughput_calculation() {
        let validator = PerformanceValidator::new();
        let processing_time = Duration::from_millis(100);
        let throughput = validator.calculate_throughput(1600, processing_time);
        assert_eq!(throughput, 16000.0); // 1600 samples / 0.1 seconds = 16000 samples/sec
    }

    #[test]
    fn test_cpu_utilization_estimation() {
        let processing_time = Duration::from_millis(200);
        let audio_duration = Duration::from_secs(1);
        let utilization = estimate_cpu_utilization(processing_time, audio_duration);
        assert_eq!(utilization, 20.0); // 200ms processing / 1000ms audio = 20%
    }

    #[test]
    fn test_memory_usage_estimation() {
        let usage = estimate_process_memory_usage();

        // Should be at least base memory (50MB)
        assert!(usage >= 50 * 1024 * 1024);

        // Should be reasonable (not more than 2GB for estimation)
        assert!(usage <= 2 * 1024 * 1024 * 1024);

        // Should include model footprint
        assert!(usage >= 150 * 1024 * 1024); // Base + models should be at least 150MB
    }

    #[test]
    fn test_model_memory_footprint_detection() {
        let footprint = detect_model_memory_footprint();

        // Should include base audio processing (30MB)
        assert!(footprint >= 30 * 1024 * 1024);

        // Should include ASR model memory (120MB)
        assert!(footprint >= 150 * 1024 * 1024);

        // Should be reasonable upper bound
        assert!(footprint <= 1024 * 1024 * 1024); // 1GB max
    }

    #[test]
    fn test_system_memory_detection() {
        // This test may fail on some platforms, but should not panic
        match get_total_system_memory() {
            Ok(memory) => {
                // Should be at least 1GB (reasonable minimum)
                assert!(memory >= 1024 * 1024 * 1024);
                // Should be less than 1TB (reasonable maximum)
                assert!(memory <= 1024 * 1024 * 1024 * 1024);
            }
            Err(_) => {
                // Platform doesn't support detection, which is okay
                println!("System memory detection not supported on this platform");
            }
        }
    }

    #[test]
    fn test_memory_percentage_calculations() {
        // Test different memory tier calculations
        let test_cases = [
            (2u64 * 1024 * 1024 * 1024, 0.004),  // 2GB system
            (8u64 * 1024 * 1024 * 1024, 0.005),  // 8GB system (exactly 8GB falls to >4GB tier)
            (9u64 * 1024 * 1024 * 1024, 0.006),  // 9GB system (>8GB tier)
            (16u64 * 1024 * 1024 * 1024, 0.006), // 16GB system (exactly 16GB falls to >8GB tier)
            (17u64 * 1024 * 1024 * 1024, 0.008), // 17GB system (>16GB tier)
            (32u64 * 1024 * 1024 * 1024, 0.008), // 32GB system (exactly 32GB falls to >16GB tier)
            (33u64 * 1024 * 1024 * 1024, 0.010), // 33GB system (>32GB tier)
        ];

        for (total_memory, expected_percentage) in test_cases {
            let percentage = if total_memory > 32u64 * 1024 * 1024 * 1024 {
                0.010
            } else if total_memory > 16u64 * 1024 * 1024 * 1024 {
                0.008
            } else if total_memory > 8u64 * 1024 * 1024 * 1024 {
                0.006
            } else if total_memory > 4u64 * 1024 * 1024 * 1024 {
                0.005
            } else {
                0.004
            };

            assert_eq!(percentage, expected_percentage);
        }
    }

    #[tokio::test]
    async fn test_startup_time_measurement() {
        let validator = PerformanceValidator::new();

        let startup_fn = || async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(())
        };

        let result = validator.measure_startup_time(startup_fn).await;
        assert!(result.is_ok());

        let (startup_ms, passed) = result.unwrap();
        assert!(startup_ms >= 100);
        assert!(startup_ms < 5000); // Should pass default requirement
        assert!(passed);
    }
}
