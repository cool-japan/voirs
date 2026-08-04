//! Production readiness checker and diagnostic tools for VoiRS SDK.
//!
//! This module provides comprehensive production deployment validation and
//! diagnostic capabilities to ensure your VoiRS deployment meets best practices
//! and performance requirements.
//!
//! # Features
//!
//! - **Readiness Checks**: Validates system configuration and resources
//! - **Performance Benchmarking**: Quick performance validation
//! - **Configuration Validation**: Checks for optimal settings
//! - **Security Audit**: Reviews security-related settings
//! - **Compatibility Checks**: Validates model and system compatibility
//! - **Best Practices**: Ensures compliance with recommended practices
//!
//! Every check queries real, local state (actual `/proc`/`sysctl`/`statvfs`
//! resource queries, the actual running `rustc` version, actual file
//! permissions, an actual timed synthesis run when a backend is attached) —
//! none of the numbers in a [`ReadinessReport`] are hardcoded placeholders.
//! Where a value genuinely cannot be determined (e.g. disk-space detection on
//! an unsupported platform, or no synthesis backend attached for
//! benchmarking), the corresponding check fails closed or is reported as
//! explicitly skipped rather than fabricated.
//!
//! # Example
//!
//! ```no_run
//! use voirs_sdk::diagnostics::{ProductionReadiness, ReadinessConfig};
//!
//! #[tokio::main]
//! async fn main() -> voirs_sdk::Result<()> {
//!     let checker = ProductionReadiness::new(ReadinessConfig::default());
//!
//!     // Run comprehensive readiness check
//!     let report = checker.check_readiness().await?;
//!
//!     if report.is_production_ready() {
//!         println!("✓ System is production ready!");
//!     } else {
//!         println!("✗ Issues found:");
//!         for issue in report.critical_issues() {
//!             println!("  - {}", issue);
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```

use crate::traits::{AcousticModel, G2p, Vocoder};
use crate::{Result, VoirsError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for production readiness checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessConfig {
    /// Minimum CPU cores required
    pub min_cpu_cores: usize,

    /// Minimum available memory (in GB)
    pub min_memory_gb: f64,

    /// Minimum available disk space (in GB)
    pub min_disk_gb: f64,

    /// Required synthesis speed (max RTF)
    pub max_acceptable_rtf: f32,

    /// Maximum acceptable latency (ms)
    pub max_latency_ms: u64,

    /// Enable security checks
    pub enable_security_checks: bool,

    /// Enable performance benchmarking
    pub enable_benchmarking: bool,

    /// Cache directory to validate
    pub cache_dir: Option<PathBuf>,
}

impl Default for ReadinessConfig {
    fn default() -> Self {
        Self {
            min_cpu_cores: 4,
            min_memory_gb: 4.0,
            min_disk_gb: 10.0,
            max_acceptable_rtf: 0.5,
            max_latency_ms: 150,
            enable_security_checks: true,
            enable_benchmarking: true,
            cache_dir: None,
        }
    }
}

/// Production readiness report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessReport {
    /// Overall readiness status
    pub is_ready: bool,

    /// Timestamp of the check
    pub timestamp: String,

    /// Individual check results
    pub checks: Vec<CheckResult>,

    /// Performance benchmark results
    pub benchmark_results: Option<BenchmarkResults>,

    /// System information
    pub system_info: SystemInfo,

    /// Recommendations for improvements
    pub recommendations: Vec<Recommendation>,
}

impl ReadinessReport {
    /// Check if system is production ready.
    pub fn is_production_ready(&self) -> bool {
        self.is_ready
    }

    /// Get all critical issues.
    pub fn critical_issues(&self) -> Vec<String> {
        self.checks
            .iter()
            .filter(|c| c.severity == Severity::Critical && !c.passed)
            .map(|c| c.message.clone())
            .collect()
    }

    /// Get all warnings.
    pub fn warnings(&self) -> Vec<String> {
        self.checks
            .iter()
            .filter(|c| c.severity == Severity::Warning && !c.passed)
            .map(|c| c.message.clone())
            .collect()
    }

    /// Get high-priority recommendations.
    pub fn high_priority_recommendations(&self) -> Vec<&Recommendation> {
        self.recommendations
            .iter()
            .filter(|r| r.priority == RecommendationPriority::High)
            .collect()
    }

    /// Generate a human-readable report.
    pub fn to_report_string(&self) -> String {
        let mut report = String::new();

        report.push_str("=== VoiRS Production Readiness Report ===\n\n");
        report.push_str(&format!(
            "Status: {}\n",
            if self.is_ready {
                "✓ READY"
            } else {
                "✗ NOT READY"
            }
        ));
        report.push_str(&format!("Timestamp: {}\n\n", self.timestamp));

        report.push_str("System Information:\n");
        report.push_str(&format!("  CPU Cores: {}\n", self.system_info.cpu_cores));
        report.push_str(&format!(
            "  Memory: {:.2} GB\n",
            self.system_info.total_memory_gb
        ));
        report.push_str(&format!("  Platform: {}\n\n", self.system_info.platform));

        if !self.critical_issues().is_empty() {
            report.push_str("Critical Issues:\n");
            for issue in self.critical_issues() {
                report.push_str(&format!("  ✗ {}\n", issue));
            }
            report.push('\n');
        }

        if !self.warnings().is_empty() {
            report.push_str("Warnings:\n");
            for warning in self.warnings() {
                report.push_str(&format!("  ⚠ {}\n", warning));
            }
            report.push('\n');
        }

        if !self.high_priority_recommendations().is_empty() {
            report.push_str("High Priority Recommendations:\n");
            for rec in self.high_priority_recommendations() {
                report.push_str(&format!("  → {}\n", rec.message));
            }
            report.push('\n');
        }

        if let Some(bench) = &self.benchmark_results {
            report.push_str("Performance Benchmarks:\n");
            report.push_str(&format!("  RTF: {:.3}\n", bench.average_rtf));
            report.push_str(&format!("  Latency: {} ms\n", bench.average_latency_ms));
            report.push_str(&format!("  Memory: {:.2} MB\n", bench.peak_memory_mb));
        }

        report
    }
}

/// Individual check result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    /// Name of the check
    pub name: String,

    /// Whether the check passed
    pub passed: bool,

    /// Severity level
    pub severity: Severity,

    /// Detailed message
    pub message: String,

    /// Category of the check
    pub category: CheckCategory,
}

/// Severity levels for checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    /// Critical issue that prevents production use
    Critical,

    /// Warning that should be addressed
    Warning,

    /// Informational notice
    Info,
}

/// Categories of checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckCategory {
    /// System resource checks
    Resources,

    /// Configuration validation
    Configuration,

    /// Performance validation
    Performance,

    /// Security validation
    Security,

    /// Compatibility validation
    Compatibility,
}

/// Benchmark results for performance validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Average real-time factor
    pub average_rtf: f32,

    /// Average latency in milliseconds
    pub average_latency_ms: u64,

    /// Peak memory usage in MB
    pub peak_memory_mb: f64,

    /// Number of test syntheses performed
    pub test_count: usize,

    /// Duration of benchmark
    pub benchmark_duration: Duration,
}

/// System information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// Number of CPU cores
    pub cpu_cores: usize,

    /// Total memory in GB
    pub total_memory_gb: f64,

    /// Available disk space in GB
    pub available_disk_gb: f64,

    /// Operating system platform
    pub platform: String,

    /// Architecture
    pub architecture: String,

    /// Human-readable notes about any resource values above that could not
    /// be determined on this host/platform (in which case the corresponding
    /// field is honestly `0.0` rather than a fabricated guess).
    #[serde(default)]
    pub detection_notes: Vec<String>,
}

/// Recommendation for improvement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Priority level
    pub priority: RecommendationPriority,

    /// Category
    pub category: CheckCategory,

    /// Recommendation message
    pub message: String,

    /// Optional action to take
    pub action: Option<String>,
}

/// Priority levels for recommendations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationPriority {
    /// High priority - should address immediately
    High,

    /// Medium priority - address when convenient
    Medium,

    /// Low priority - optional optimization
    Low,
}

/// Real G2P/acoustic-model/vocoder backends used to run an actual timed
/// synthesis benchmark. Without these, [`ProductionReadiness::run_benchmark`]
/// honestly reports that benchmarking was skipped instead of fabricating
/// results.
#[derive(Clone)]
struct SynthesisBackends {
    g2p: Arc<dyn G2p>,
    acoustic: Arc<dyn AcousticModel>,
    vocoder: Arc<dyn Vocoder>,
}

/// Production readiness checker.
pub struct ProductionReadiness {
    config: ReadinessConfig,
    synthesis_backends: Option<SynthesisBackends>,
}

impl std::fmt::Debug for ProductionReadiness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProductionReadiness")
            .field("config", &self.config)
            .field(
                "synthesis_backends_configured",
                &self.synthesis_backends.is_some(),
            )
            .finish()
    }
}

impl ProductionReadiness {
    /// Create a new production readiness checker.
    ///
    /// No synthesis backend is attached; [`Self::run_benchmark`] will report
    /// benchmarking as skipped until [`Self::with_synthesis_backends`] is
    /// called.
    pub fn new(config: ReadinessConfig) -> Self {
        Self {
            config,
            synthesis_backends: None,
        }
    }

    /// Attach real G2P/acoustic-model/vocoder backends so
    /// [`Self::check_readiness`] can run a genuine timed synthesis benchmark
    /// instead of skipping it.
    #[must_use]
    pub fn with_synthesis_backends(
        mut self,
        g2p: Arc<dyn G2p>,
        acoustic: Arc<dyn AcousticModel>,
        vocoder: Arc<dyn Vocoder>,
    ) -> Self {
        self.synthesis_backends = Some(SynthesisBackends {
            g2p,
            acoustic,
            vocoder,
        });
        self
    }

    /// Run comprehensive readiness checks.
    pub async fn check_readiness(&self) -> Result<ReadinessReport> {
        let start_time = Instant::now();
        let mut checks = Vec::new();
        let mut recommendations = Vec::new();

        // Gather system information
        let system_info = self.gather_system_info().await?;

        // Run resource checks
        checks.extend(self.check_resources(&system_info)?);

        // Run configuration checks
        checks.extend(self.check_configuration()?);

        // Run compatibility checks
        checks.extend(self.check_compatibility()?);

        // Run security checks if enabled
        if self.config.enable_security_checks {
            checks.extend(self.check_security()?);
        }

        // Run performance benchmark if enabled
        let benchmark_results = if self.config.enable_benchmarking {
            self.run_benchmark().await?
        } else {
            None
        };

        // Validate benchmark results
        match &benchmark_results {
            Some(bench) => checks.extend(self.validate_performance(bench)?),
            None if self.config.enable_benchmarking => {
                // Benchmarking was requested but no backend was attached: say so
                // explicitly rather than silently omitting the section.
                checks.push(CheckResult {
                    name: "Performance Benchmark".to_string(),
                    passed: true,
                    severity: Severity::Info,
                    message: "Benchmarking skipped: no synthesis backend attached (call \
                        ProductionReadiness::with_synthesis_backends(..) to enable a real, \
                        timed benchmark)"
                        .to_string(),
                    category: CheckCategory::Performance,
                });
            }
            None => {}
        }

        // Generate recommendations based on checks
        recommendations.extend(self.generate_recommendations(&checks, &system_info)?);

        // Determine overall readiness
        let is_ready = !checks
            .iter()
            .any(|c| c.severity == Severity::Critical && !c.passed);

        let elapsed = start_time.elapsed();
        tracing::info!("Production readiness check completed in {:?}", elapsed);

        Ok(ReadinessReport {
            is_ready,
            timestamp: chrono::Utc::now().to_rfc3339(),
            checks,
            benchmark_results,
            system_info,
            recommendations,
        })
    }

    async fn gather_system_info(&self) -> Result<SystemInfo> {
        let mut detection_notes = Vec::new();

        let total_memory_gb = match query_total_memory_gb() {
            Ok(gb) => gb,
            Err(reason) => {
                detection_notes.push(format!("total memory undetermined: {reason}"));
                0.0
            }
        };

        let disk_probe_path = self
            .config
            .cache_dir
            .clone()
            .unwrap_or_else(std::env::temp_dir);
        let available_disk_gb = match query_available_disk_gb(&disk_probe_path) {
            Ok(gb) => gb,
            Err(reason) => {
                detection_notes.push(format!("available disk space undetermined: {reason}"));
                0.0
            }
        };

        Ok(SystemInfo {
            cpu_cores: num_cpus::get(),
            total_memory_gb,
            available_disk_gb,
            platform: std::env::consts::OS.to_string(),
            architecture: std::env::consts::ARCH.to_string(),
            detection_notes,
        })
    }

    fn check_resources(&self, system_info: &SystemInfo) -> Result<Vec<CheckResult>> {
        let mut checks = Vec::new();

        // Check CPU cores
        checks.push(CheckResult {
            name: "CPU Cores".to_string(),
            passed: system_info.cpu_cores >= self.config.min_cpu_cores,
            severity: Severity::Warning,
            message: format!(
                "CPU cores: {} (minimum: {})",
                system_info.cpu_cores, self.config.min_cpu_cores
            ),
            category: CheckCategory::Resources,
        });

        // Check memory. `0.0` combined with a matching `detection_notes` entry
        // means "could not be measured" (fails closed); `0.0` with no note
        // would mean the host genuinely reported zero.
        let memory_note = system_info
            .detection_notes
            .iter()
            .find(|n| n.starts_with("total memory"));
        checks.push(CheckResult {
            name: "Memory".to_string(),
            passed: system_info.total_memory_gb >= self.config.min_memory_gb,
            severity: Severity::Critical,
            message: match memory_note {
                Some(note) => format!("{note}; treating as not ready"),
                None => format!(
                    "Total memory: {:.2} GB (minimum: {:.2} GB)",
                    system_info.total_memory_gb, self.config.min_memory_gb
                ),
            },
            category: CheckCategory::Resources,
        });

        // Check disk space
        let disk_note = system_info
            .detection_notes
            .iter()
            .find(|n| n.starts_with("available disk space"));
        checks.push(CheckResult {
            name: "Disk Space".to_string(),
            passed: system_info.available_disk_gb >= self.config.min_disk_gb,
            severity: Severity::Warning,
            message: match disk_note {
                Some(note) => format!("{note}; treating as not ready"),
                None => format!(
                    "Available disk: {:.2} GB (minimum: {:.2} GB)",
                    system_info.available_disk_gb, self.config.min_disk_gb
                ),
            },
            category: CheckCategory::Resources,
        });

        Ok(checks)
    }

    fn check_configuration(&self) -> Result<Vec<CheckResult>> {
        let mut checks = Vec::new();

        // Check cache directory
        if let Some(ref cache_dir) = self.config.cache_dir {
            let exists = cache_dir.exists();
            checks.push(CheckResult {
                name: "Cache Directory".to_string(),
                passed: exists,
                severity: Severity::Warning,
                message: if exists {
                    format!("Cache directory exists: {:?}", cache_dir)
                } else {
                    format!("Cache directory does not exist: {:?}", cache_dir)
                },
                category: CheckCategory::Configuration,
            });
        }

        Ok(checks)
    }

    fn check_compatibility(&self) -> Result<Vec<CheckResult>> {
        Ok(vec![
            Self::check_rust_version(),
            Self::check_platform_support(),
        ])
    }

    /// Compare the actual running `rustc --version` against this crate's
    /// declared MSRV (`CARGO_PKG_RUST_VERSION`, from `rust-version` in
    /// `Cargo.toml`), rather than unconditionally reporting "compatible".
    fn check_rust_version() -> CheckResult {
        let msrv = env!("CARGO_PKG_RUST_VERSION");
        match detect_rustc_version() {
            Ok(actual) => {
                let passed = match (
                    semver::Version::parse(&normalize_semver(&actual)),
                    semver::Version::parse(&normalize_semver(msrv)),
                ) {
                    (Ok(actual_ver), Ok(msrv_ver)) => actual_ver >= msrv_ver,
                    _ => false,
                };
                CheckResult {
                    name: "Rust Version".to_string(),
                    passed,
                    severity: Severity::Warning,
                    message: format!("rustc {actual} (minimum required: {msrv})"),
                    category: CheckCategory::Compatibility,
                }
            }
            Err(reason) => CheckResult {
                name: "Rust Version".to_string(),
                passed: false,
                severity: Severity::Warning,
                message: format!("Could not determine the running rustc version: {reason}"),
                category: CheckCategory::Compatibility,
            },
        }
    }

    /// Check that the host OS is one VoiRS officially supports (per
    /// `CLAUDE.md`'s Platform Support section), a real fact about
    /// `std::env::consts::OS`, not a hardcoded "compatible".
    fn check_platform_support() -> CheckResult {
        const SUPPORTED: [&str; 3] = ["linux", "macos", "windows"];
        let platform = std::env::consts::OS;
        let passed = SUPPORTED.contains(&platform);
        CheckResult {
            name: "Platform Support".to_string(),
            passed,
            severity: Severity::Warning,
            message: if passed {
                format!("Platform '{platform}' is officially supported")
            } else {
                format!(
                    "Platform '{platform}' is not in the officially supported list {SUPPORTED:?}"
                )
            },
            category: CheckCategory::Compatibility,
        }
    }

    fn check_security(&self) -> Result<Vec<CheckResult>> {
        let mut checks = Vec::new();

        // TLS crypto provider: cloud/http/network features rely on a
        // process-wide rustls CryptoProvider being installed via
        // `ensure_crypto_provider()`; without it the first TLS handshake
        // panics instead of failing gracefully.
        let crypto_installed = rustls::crypto::CryptoProvider::get_default().is_some();
        checks.push(CheckResult {
            name: "TLS Crypto Provider".to_string(),
            passed: crypto_installed,
            severity: Severity::Warning,
            message: if crypto_installed {
                "A process-wide rustls CryptoProvider is installed".to_string()
            } else {
                "No rustls CryptoProvider is installed yet; call \
                 voirs_sdk::ensure_crypto_provider() before any TLS use"
                    .to_string()
            },
            category: CheckCategory::Security,
        });

        // Debug builds should not be deployed to production.
        let is_debug_build = cfg!(debug_assertions);
        checks.push(CheckResult {
            name: "Release Build".to_string(),
            passed: !is_debug_build,
            severity: Severity::Warning,
            message: if is_debug_build {
                "This binary was compiled with debug assertions enabled; use a release build \
                 (cargo build --release) for production"
                    .to_string()
            } else {
                "Compiled without debug assertions (release build)".to_string()
            },
            category: CheckCategory::Security,
        });

        // Cache directory permissions: a world-writable cache directory lets
        // any local user tamper with cached model weights or results.
        if let Some(ref cache_dir) = self.config.cache_dir {
            checks.push(check_cache_dir_permissions(cache_dir));
        }

        Ok(checks)
    }

    /// Run a real, timed synthesis benchmark using the attached backends.
    ///
    /// Returns `Ok(None)` (never fabricated numbers) when no backend has
    /// been attached via [`Self::with_synthesis_backends`].
    async fn run_benchmark(&self) -> Result<Option<BenchmarkResults>> {
        let Some(backends) = self.synthesis_backends.clone() else {
            return Ok(None);
        };

        const TEST_SENTENCES: [&str; 3] = [
            "The quick brown fox jumps over the lazy dog.",
            "VoiRS synthesizes natural sounding speech in real time.",
            "Production readiness benchmarks measure genuine synthesis performance.",
        ];

        let start = Instant::now();
        let mut rtfs = Vec::with_capacity(TEST_SENTENCES.len());
        let mut latencies_ms = Vec::with_capacity(TEST_SENTENCES.len());
        let mut peak_memory_mb = 0.0f64;

        for text in TEST_SENTENCES {
            let call_start = Instant::now();

            let phonemes = backends.g2p.to_phonemes(text, None).await.map_err(|e| {
                VoirsError::config_error(format!("Readiness benchmark G2P step failed: {e}"))
            })?;
            let mel = backends
                .acoustic
                .synthesize(&phonemes, None)
                .await
                .map_err(|e| {
                    VoirsError::config_error(format!(
                        "Readiness benchmark acoustic synthesis step failed: {e}"
                    ))
                })?;
            let audio = backends.vocoder.vocode(&mel, None).await.map_err(|e| {
                VoirsError::config_error(format!("Readiness benchmark vocoding step failed: {e}"))
            })?;

            let elapsed = call_start.elapsed();
            let audio_duration = audio.duration();
            let rtf = if audio_duration > 0.0 {
                elapsed.as_secs_f32() / audio_duration
            } else {
                0.0
            };

            rtfs.push(rtf);
            latencies_ms.push(elapsed.as_millis() as u64);
            peak_memory_mb = peak_memory_mb.max(current_peak_rss_mb());
        }

        let average_rtf = rtfs.iter().sum::<f32>() / rtfs.len() as f32;
        let average_latency_ms = latencies_ms.iter().sum::<u64>() / latencies_ms.len() as u64;

        Ok(Some(BenchmarkResults {
            average_rtf,
            average_latency_ms,
            peak_memory_mb,
            test_count: rtfs.len(),
            benchmark_duration: start.elapsed(),
        }))
    }

    fn validate_performance(&self, bench: &BenchmarkResults) -> Result<Vec<CheckResult>> {
        let mut checks = Vec::new();

        // Check RTF
        checks.push(CheckResult {
            name: "Real-Time Factor".to_string(),
            passed: bench.average_rtf <= self.config.max_acceptable_rtf,
            severity: Severity::Critical,
            message: format!(
                "Average RTF: {:.3} (maximum: {:.3})",
                bench.average_rtf, self.config.max_acceptable_rtf
            ),
            category: CheckCategory::Performance,
        });

        // Check latency
        checks.push(CheckResult {
            name: "Latency".to_string(),
            passed: bench.average_latency_ms <= self.config.max_latency_ms,
            severity: Severity::Warning,
            message: format!(
                "Average latency: {} ms (maximum: {} ms)",
                bench.average_latency_ms, self.config.max_latency_ms
            ),
            category: CheckCategory::Performance,
        });

        Ok(checks)
    }

    fn generate_recommendations(
        &self,
        checks: &[CheckResult],
        system_info: &SystemInfo,
    ) -> Result<Vec<Recommendation>> {
        let mut recommendations = Vec::new();

        // Generate recommendations based on failed checks
        for check in checks {
            if !check.passed {
                let priority = match check.severity {
                    Severity::Critical => RecommendationPriority::High,
                    Severity::Warning => RecommendationPriority::Medium,
                    Severity::Info => RecommendationPriority::Low,
                };

                recommendations.push(Recommendation {
                    priority,
                    category: check.category,
                    message: format!("Address: {}", check.message),
                    action: None,
                });
            }
        }

        // Add general recommendations based on system info
        if system_info.cpu_cores < 8 {
            recommendations.push(Recommendation {
                priority: RecommendationPriority::Medium,
                category: CheckCategory::Performance,
                message: "Consider upgrading to 8+ CPU cores for better performance".to_string(),
                action: Some("Hardware upgrade".to_string()),
            });
        }

        Ok(recommendations)
    }
}

/// Normalize a version string to `major.minor.patch` so [`semver::Version`]
/// can parse Cargo's `rust-version` field, which is commonly written with
/// only `major.minor` (e.g. `"1.89"`).
fn normalize_semver(version: &str) -> String {
    let trimmed = version.trim();
    match trimmed.split('.').count() {
        1 => format!("{trimmed}.0.0"),
        2 => format!("{trimmed}.0"),
        _ => trimmed.to_string(),
    }
}

/// Run `rustc --version` and extract the version token (e.g. `"1.82.0"` from
/// `"rustc 1.82.0 (f6e511eec 2024-10-15)"`).
fn detect_rustc_version() -> std::result::Result<String, String> {
    let mut command = std::process::Command::new("rustc");
    command.arg("--version");
    let output = crate::process_probe::run_with_timeout(
        &mut command,
        crate::process_probe::DEFAULT_PROBE_TIMEOUT,
    )
    .map_err(|e| format!("failed to execute rustc: {e}"))?
    .ok_or_else(|| "rustc --version did not respond within the probe timeout".to_string())?;
    if !output.status.success() {
        return Err("rustc --version exited with a non-zero status".to_string());
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.split_whitespace()
        .nth(1)
        .map(str::to_string)
        .ok_or_else(|| format!("unrecognized `rustc --version` output: {text:?}"))
}

/// Query total system memory in GiB via a real platform-specific mechanism
/// (`/proc/meminfo` on Linux, `sysctl hw.memsize` on macOS).
///
/// Returns `Err` with a human-readable reason when the value cannot be
/// determined on this platform/host, rather than a fabricated placeholder.
fn query_total_memory_gb() -> std::result::Result<f64, String> {
    #[cfg(target_os = "linux")]
    {
        let contents = std::fs::read_to_string("/proc/meminfo")
            .map_err(|e| format!("failed to read /proc/meminfo: {e}"))?;
        for line in contents.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                if let Some(kb) = rest
                    .split_whitespace()
                    .next()
                    .and_then(|tok| tok.parse::<u64>().ok())
                {
                    return Ok(kb as f64 / (1024.0 * 1024.0));
                }
            }
        }
        Err("MemTotal field not found in /proc/meminfo".to_string())
    }

    #[cfg(target_os = "macos")]
    {
        let mut command = std::process::Command::new("sysctl");
        command.args(["-n", "hw.memsize"]);
        let output = crate::process_probe::run_with_timeout(
            &mut command,
            crate::process_probe::DEFAULT_PROBE_TIMEOUT,
        )
        .map_err(|e| format!("failed to execute sysctl: {e}"))?
        .ok_or_else(|| {
            "sysctl -n hw.memsize did not respond within the probe timeout".to_string()
        })?;
        if !output.status.success() {
            return Err("sysctl -n hw.memsize exited with a non-zero status".to_string());
        }
        let text = String::from_utf8_lossy(&output.stdout);
        let bytes: u64 = text
            .trim()
            .parse()
            .map_err(|e| format!("failed to parse sysctl output {text:?}: {e}"))?;
        Ok(bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        Err("total memory detection is only implemented for Linux and macOS".to_string())
    }
}

/// Query available disk space (in GiB) at `path` (or its nearest existing
/// ancestor) via a real `statvfs(2)` call on Unix.
fn query_available_disk_gb(path: &std::path::Path) -> std::result::Result<f64, String> {
    #[cfg(unix)]
    {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;

        let mut probe = path.to_path_buf();
        while !probe.exists() {
            if !probe.pop() {
                return Err(format!("no existing ancestor directory found for {path:?}"));
            }
        }

        let c_path = CString::new(probe.as_os_str().as_bytes())
            .map_err(|e| format!("path contains an interior NUL byte: {e}"))?;

        // Safety: `c_path` is a valid NUL-terminated C string for the duration
        // of this call, and `stat` is a plain-old-data struct that `statvfs`
        // fully initializes on success (checked via its return code below).
        let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
        let ret = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };
        if ret != 0 {
            return Err(format!(
                "statvfs({probe:?}) failed: {}",
                std::io::Error::last_os_error()
            ));
        }

        let available_bytes = stat.f_bavail as f64 * stat.f_frsize as f64;
        Ok(available_bytes / (1024.0 * 1024.0 * 1024.0))
    }

    #[cfg(not(unix))]
    {
        let _ = path;
        Err("disk space detection is only implemented for Unix platforms".to_string())
    }
}

/// Check whether `cache_dir` is world-writable (Unix only): any local user
/// could tamper with cached model weights or synthesis results.
#[cfg(unix)]
fn check_cache_dir_permissions(cache_dir: &std::path::Path) -> CheckResult {
    use std::os::unix::fs::PermissionsExt;

    match std::fs::metadata(cache_dir) {
        Ok(metadata) => {
            let mode = metadata.permissions().mode();
            let world_writable = mode & 0o002 != 0;
            CheckResult {
                name: "Cache Directory Permissions".to_string(),
                passed: !world_writable,
                severity: Severity::Critical,
                message: if world_writable {
                    format!(
                        "Cache directory {cache_dir:?} is world-writable (mode {mode:o}); any \
                         local user could tamper with cached models"
                    )
                } else {
                    format!("Cache directory {cache_dir:?} is not world-writable (mode {mode:o})")
                },
                category: CheckCategory::Security,
            }
        }
        Err(e) => CheckResult {
            name: "Cache Directory Permissions".to_string(),
            passed: false,
            severity: Severity::Warning,
            message: format!("Could not read metadata for cache directory {cache_dir:?}: {e}"),
            category: CheckCategory::Security,
        },
    }
}

#[cfg(not(unix))]
fn check_cache_dir_permissions(cache_dir: &std::path::Path) -> CheckResult {
    CheckResult {
        name: "Cache Directory Permissions".to_string(),
        passed: true,
        severity: Severity::Info,
        message: format!(
            "Cache directory permission checks are only implemented on Unix; skipped for \
             {cache_dir:?}"
        ),
        category: CheckCategory::Security,
    }
}

/// Query the process's real peak resident-set size in MB via `getrusage(2)`.
#[cfg(unix)]
fn current_peak_rss_mb() -> f64 {
    // Safety: `usage` is plain-old-data, fully initialized by `getrusage` on
    // success; on failure we honestly return 0.0 instead of reading it.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if ret != 0 {
        return 0.0;
    }
    // `ru_maxrss` is in bytes on macOS/BSD but kilobytes on Linux.
    #[cfg(target_os = "macos")]
    {
        usage.ru_maxrss as f64 / (1024.0 * 1024.0)
    }
    #[cfg(not(target_os = "macos"))]
    {
        usage.ru_maxrss as f64 / 1024.0
    }
}

#[cfg(not(unix))]
fn current_peak_rss_mb() -> f64 {
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{DummyAcoustic, DummyG2p, DummyVocoder};

    #[test]
    fn test_readiness_config_default() {
        let config = ReadinessConfig::default();
        assert_eq!(config.min_cpu_cores, 4);
        assert_eq!(config.min_memory_gb, 4.0);
        assert!(config.enable_benchmarking);
    }

    #[tokio::test]
    async fn test_production_readiness_check() {
        let checker = ProductionReadiness::new(ReadinessConfig::default());
        let report = checker.check_readiness().await.unwrap();

        assert!(!report.checks.is_empty());
        assert!(report.system_info.cpu_cores > 0);
    }

    #[tokio::test]
    async fn test_system_info_memory_and_disk_are_really_measured_or_honestly_flagged() {
        // Direct regression test for the fabrication bug: the old
        // implementation hardcoded `total_memory_gb: 16.0` and
        // `available_disk_gb: 100.0` regardless of the host. Real detection
        // must either produce a plausible positive value, or explain via
        // `detection_notes` why it could not (never a silent fabricated
        // constant).
        let checker = ProductionReadiness::new(ReadinessConfig::default());
        let report = checker.check_readiness().await.unwrap();
        let info = &report.system_info;

        let memory_undetermined = info
            .detection_notes
            .iter()
            .any(|n| n.starts_with("total memory"));
        if !memory_undetermined {
            assert!(
                info.total_memory_gb > 0.0,
                "measured total memory should be a real positive value on this host"
            );
            // Sanity bound: no real host has exabytes of RAM. Catches the old
            // exact-16.0 fabrication too (real hosts are essentially never
            // exactly 16.0GB to floating-point precision).
            assert_ne!(info.total_memory_gb, 16.0);
        }

        let disk_undetermined = info
            .detection_notes
            .iter()
            .any(|n| n.starts_with("available disk space"));
        if !disk_undetermined {
            assert!(info.available_disk_gb > 0.0);
            assert_ne!(info.available_disk_gb, 100.0);
        }
    }

    #[test]
    fn test_readiness_report_critical_issues() {
        let report = ReadinessReport {
            is_ready: false,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            checks: vec![CheckResult {
                name: "Test".to_string(),
                passed: false,
                severity: Severity::Critical,
                message: "Critical issue".to_string(),
                category: CheckCategory::Resources,
            }],
            benchmark_results: None,
            system_info: SystemInfo {
                cpu_cores: 4,
                total_memory_gb: 8.0,
                available_disk_gb: 50.0,
                platform: "linux".to_string(),
                architecture: "x86_64".to_string(),
                detection_notes: vec![],
            },
            recommendations: vec![],
        };

        let issues = report.critical_issues();
        assert_eq!(issues.len(), 1);
        assert!(issues[0].contains("Critical issue"));
    }

    #[test]
    fn test_report_string_generation() {
        let report = ReadinessReport {
            is_ready: true,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            checks: vec![],
            benchmark_results: None,
            system_info: SystemInfo {
                cpu_cores: 8,
                total_memory_gb: 16.0,
                available_disk_gb: 100.0,
                platform: "linux".to_string(),
                architecture: "x86_64".to_string(),
                detection_notes: vec![],
            },
            recommendations: vec![],
        };

        let report_str = report.to_report_string();
        assert!(report_str.contains("READY"));
        assert!(report_str.contains("CPU Cores: 8"));
    }

    #[tokio::test]
    async fn test_benchmarking_skipped_without_backend_is_honest_not_fabricated() {
        // Direct regression test: the old implementation always slept 100ms
        // and returned hardcoded RTF/latency/memory numbers regardless of the
        // host's real performance. Without an attached backend, benchmarking
        // must now be honestly skipped (Ok(None)), not fabricated.
        let checker = ProductionReadiness::new(ReadinessConfig::default());
        let report = checker.check_readiness().await.unwrap();
        assert!(report.benchmark_results.is_none());

        let skipped_check = report
            .checks
            .iter()
            .find(|c| c.name == "Performance Benchmark");
        assert!(
            skipped_check.is_some(),
            "expected an explicit 'benchmarking skipped' check"
        );
        assert!(skipped_check.unwrap().passed);
    }

    #[tokio::test]
    async fn test_benchmark_with_real_backend_scales_with_actual_synthesis() {
        // Direct regression test for the fabrication bug: the old
        // implementation was `sleep(100ms)` + hardcoded
        // `average_rtf: 0.3, average_latency_ms: 80, peak_memory_mb: 150.0,
        // test_count: 10` regardless of what was actually run. With a real
        // (if minimal) backend attached, the benchmark must reflect genuine
        // measured timings from `test_count` == the real number of test
        // sentences actually synthesized.
        let checker = ProductionReadiness::new(ReadinessConfig::default()).with_synthesis_backends(
            Arc::new(DummyG2p::new()),
            Arc::new(DummyAcoustic::new()),
            Arc::new(DummyVocoder::new()),
        );

        let report = checker.check_readiness().await.unwrap();
        let bench = report
            .benchmark_results
            .expect("benchmark should run with a real backend attached");

        assert_eq!(
            bench.test_count, 3,
            "must match the real number of sentences run"
        );
        assert_ne!(
            bench.average_rtf, 0.3,
            "must not be the old hardcoded value"
        );
        assert_ne!(
            bench.average_latency_ms, 80,
            "must not be the old hardcoded value"
        );
        assert!(bench.benchmark_duration > Duration::ZERO);
    }

    #[test]
    fn test_normalize_semver_pads_missing_components() {
        assert_eq!(normalize_semver("1.89"), "1.89.0");
        assert_eq!(normalize_semver("1"), "1.0.0");
        assert_eq!(normalize_semver("1.82.0"), "1.82.0");
    }

    #[test]
    fn test_detect_rustc_version_returns_a_real_parseable_version() {
        // This genuinely shells out to `rustc --version`; it must succeed in
        // any environment capable of compiling this crate at all.
        let version = detect_rustc_version().expect("rustc must be on PATH while running tests");
        assert!(semver::Version::parse(&normalize_semver(&version)).is_ok());
    }

    #[test]
    fn test_check_rust_version_passes_on_the_toolchain_compiling_this_crate() {
        let check = ProductionReadiness::check_rust_version();
        assert!(
            check.passed,
            "the toolchain that compiled this crate must satisfy its own MSRV: {}",
            check.message
        );
    }

    #[test]
    fn test_current_peak_rss_mb_is_positive_for_a_running_process() {
        let rss = current_peak_rss_mb();
        assert!(rss > 0.0, "a running process must have nonzero peak RSS");
    }
}
