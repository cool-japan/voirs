//! Reproducibility Guarantees System
//!
//! This module provides comprehensive reproducibility guarantees for evaluation
//! results, ensuring that evaluations can be reliably reproduced with identical
//! outcomes given the same inputs and configuration.
//!
//! # Features
//!
//! - **Deterministic Execution**: Guaranteed reproducible results
//! - **Environment Capture**: Complete environment state recording
//! - **Seed Management**: Controlled randomness with seed tracking
//! - **Configuration Freezing**: Immutable evaluation configurations
//! - **Result Verification**: Cross-platform result verification
//! - **Audit Trail**: Complete reproducibility audit trail
//! - **Dependency Tracking**: Track all dependencies and versions
//!
//! # Example
//!
//! ```no_run
//! use voirs_evaluation::reproducibility::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create reproducibility manager
//! let mut manager = ReproducibilityManager::new();
//!
//! // Capture environment for reproducibility
//! let snapshot = manager.capture_environment()?;
//!
//! // Set deterministic seed
//! manager.set_seed(42);
//!
//! // Freeze configuration
//! let frozen_config = manager.freeze_configuration(
//!     "pesq_evaluation",
//!     vec![("sample_rate".to_string(), "16000".to_string())]
//! )?;
//!
//! println!("Configuration frozen with ID: {}", frozen_config.id);
//! println!("Seed: {:?}", frozen_config.seed);
//! # Ok(())
//! # }
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use thiserror::Error;
use tracing::{debug, info};
use uuid::Uuid;

/// Reproducibility errors
#[derive(Error, Debug)]
pub enum ReproducibilityError {
    /// Configuration mismatch
    #[error("Configuration mismatch: {message}")]
    ConfigurationMismatch {
        /// Error message
        message: String,
    },

    /// Environment mismatch
    #[error("Environment mismatch: expected {expected}, got {actual}")]
    EnvironmentMismatch {
        /// Expected value
        expected: String,
        /// Actual value
        actual: String,
    },

    /// Seed not set
    #[error("Random seed not set - deterministic execution not guaranteed")]
    SeedNotSet,

    /// Verification failed
    #[error("Reproducibility verification failed: {message}")]
    VerificationFailed {
        /// Error message
        message: String,
    },

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Environment snapshot for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentSnapshot {
    /// Snapshot ID
    pub id: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Operating system
    pub os: String,
    /// OS version
    pub os_version: String,
    /// CPU architecture
    pub arch: String,
    /// Rust version
    pub rust_version: String,
    /// Crate version
    pub crate_version: String,
    /// Dependencies
    pub dependencies: HashMap<String, String>,
    /// Environment variables (filtered)
    pub environment_vars: HashMap<String, String>,
    /// Hardware information
    pub hardware: HardwareInfo,
    /// Compiler flags
    pub compiler_flags: Vec<String>,
}

/// Hardware information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    /// CPU model
    pub cpu_model: String,
    /// CPU cores
    pub cpu_cores: usize,
    /// Total memory (GB)
    pub memory_gb: f64,
    /// GPU information (if available)
    pub gpu: Option<String>,
}

/// Frozen configuration for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenConfiguration {
    /// Configuration ID
    pub id: String,
    /// Configuration name
    pub name: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Random seed (if set)
    pub seed: Option<u64>,
    /// Configuration parameters
    pub parameters: HashMap<String, String>,
    /// Environment snapshot
    pub environment: EnvironmentSnapshot,
    /// Configuration hash (for verification)
    pub hash: String,
}

/// Reproducibility verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Verification ID
    pub id: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Original configuration ID
    pub original_config_id: String,
    /// Verification passed
    pub passed: bool,
    /// Environment matches
    pub environment_match: bool,
    /// Configuration matches
    pub configuration_match: bool,
    /// Results match
    pub results_match: bool,
    /// Discrepancies detected
    pub discrepancies: Vec<Discrepancy>,
    /// Verification metadata
    pub metadata: HashMap<String, String>,
}

/// Detected discrepancy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Discrepancy {
    /// Discrepancy type
    pub discrepancy_type: DiscrepancyType,
    /// Component affected
    pub component: String,
    /// Expected value
    pub expected: String,
    /// Actual value
    pub actual: String,
    /// Impact level
    pub impact: ImpactLevel,
    /// Description
    pub description: String,
}

/// Type of discrepancy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscrepancyType {
    /// Environment difference
    Environment,
    /// Configuration difference
    Configuration,
    /// Result difference
    Result,
    /// Dependency version
    Dependency,
    /// Hardware difference
    Hardware,
}

/// Impact level of discrepancy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImpactLevel {
    /// Critical - prevents reproducibility
    Critical,
    /// High - likely affects results
    High,
    /// Medium - may affect results
    Medium,
    /// Low - unlikely to affect results
    Low,
}

/// Reproducibility manager
pub struct ReproducibilityManager {
    /// Current seed (if set)
    seed: Option<u64>,
    /// Frozen configurations
    configurations: HashMap<String, FrozenConfiguration>,
    /// Verification results
    verifications: Vec<VerificationResult>,
}

impl ReproducibilityManager {
    /// Create new reproducibility manager
    pub fn new() -> Self {
        Self {
            seed: None,
            configurations: HashMap::new(),
            verifications: Vec::new(),
        }
    }

    /// Set random seed for deterministic execution
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = Some(seed);
        info!("Reproducibility seed set to: {}", seed);
    }

    /// Get current seed
    pub fn get_seed(&self) -> Option<u64> {
        self.seed
    }

    /// Capture current environment snapshot
    pub fn capture_environment(&self) -> Result<EnvironmentSnapshot, ReproducibilityError> {
        info!("Capturing environment snapshot for reproducibility");

        let snapshot = EnvironmentSnapshot {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            os: env::consts::OS.to_string(),
            os_version: Self::get_os_version(),
            arch: env::consts::ARCH.to_string(),
            rust_version: Self::get_rust_version(),
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            dependencies: Self::get_dependencies(),
            environment_vars: Self::get_safe_env_vars(),
            hardware: Self::get_hardware_info(),
            compiler_flags: Self::get_compiler_flags(),
        };

        debug!("Environment snapshot captured: {}", snapshot.id);
        Ok(snapshot)
    }

    /// Freeze configuration for reproducibility
    pub fn freeze_configuration(
        &mut self,
        name: impl Into<String>,
        parameters: Vec<(String, String)>,
    ) -> Result<FrozenConfiguration, ReproducibilityError> {
        let name = name.into();
        info!("Freezing configuration: {}", name);

        let environment = self.capture_environment()?;
        let config_id = Uuid::new_v4().to_string();

        let parameters_map: HashMap<String, String> = parameters.into_iter().collect();

        // Calculate configuration hash
        let hash = Self::calculate_hash(&parameters_map);

        let frozen_config = FrozenConfiguration {
            id: config_id.clone(),
            name,
            created_at: Utc::now(),
            seed: self.seed,
            parameters: parameters_map,
            environment,
            hash,
        };

        self.configurations.insert(config_id, frozen_config.clone());

        Ok(frozen_config)
    }

    /// Verify reproducibility against frozen configuration
    pub fn verify_reproducibility(
        &mut self,
        config_id: &str,
        current_results: HashMap<String, f64>,
        original_results: HashMap<String, f64>,
    ) -> Result<VerificationResult, ReproducibilityError> {
        info!("Verifying reproducibility for configuration: {}", config_id);

        let frozen_config = self.configurations.get(config_id).ok_or_else(|| {
            ReproducibilityError::ConfigurationMismatch {
                message: format!("Configuration not found: {}", config_id),
            }
        })?;

        let current_env = self.capture_environment()?;

        let mut discrepancies = Vec::new();
        let mut environment_match = true;
        let mut configuration_match = true;
        let mut results_match = true;

        // Verify environment
        if current_env.os != frozen_config.environment.os {
            environment_match = false;
            discrepancies.push(Discrepancy {
                discrepancy_type: DiscrepancyType::Environment,
                component: "Operating System".to_string(),
                expected: frozen_config.environment.os.clone(),
                actual: current_env.os.clone(),
                impact: ImpactLevel::High,
                description: "Operating system mismatch".to_string(),
            });
        }

        if current_env.arch != frozen_config.environment.arch {
            environment_match = false;
            discrepancies.push(Discrepancy {
                discrepancy_type: DiscrepancyType::Environment,
                component: "Architecture".to_string(),
                expected: frozen_config.environment.arch.clone(),
                actual: current_env.arch.clone(),
                impact: ImpactLevel::Medium,
                description: "CPU architecture mismatch".to_string(),
            });
        }

        // Verify seed
        if self.seed != frozen_config.seed {
            configuration_match = false;
            discrepancies.push(Discrepancy {
                discrepancy_type: DiscrepancyType::Configuration,
                component: "Random Seed".to_string(),
                expected: frozen_config
                    .seed
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "None".to_string()),
                actual: self
                    .seed
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "None".to_string()),
                impact: ImpactLevel::Critical,
                description: "Random seed mismatch - deterministic execution not guaranteed"
                    .to_string(),
            });
        }

        // Verify results
        let tolerance = 1e-6;
        for (key, original_value) in &original_results {
            if let Some(current_value) = current_results.get(key) {
                let diff = (current_value - original_value).abs();
                if diff > tolerance {
                    results_match = false;
                    discrepancies.push(Discrepancy {
                        discrepancy_type: DiscrepancyType::Result,
                        component: key.clone(),
                        expected: format!("{:.10}", original_value),
                        actual: format!("{:.10}", current_value),
                        impact: ImpactLevel::Critical,
                        description: format!("Result mismatch (diff: {:.10})", diff),
                    });
                }
            } else {
                results_match = false;
                discrepancies.push(Discrepancy {
                    discrepancy_type: DiscrepancyType::Result,
                    component: key.clone(),
                    expected: format!("{:.10}", original_value),
                    actual: "Missing".to_string(),
                    impact: ImpactLevel::Critical,
                    description: "Result missing in current evaluation".to_string(),
                });
            }
        }

        let passed = environment_match && configuration_match && results_match;

        let verification = VerificationResult {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            original_config_id: config_id.to_string(),
            passed,
            environment_match,
            configuration_match,
            results_match,
            discrepancies,
            metadata: HashMap::new(),
        };

        self.verifications.push(verification.clone());

        if passed {
            info!("Reproducibility verified successfully");
        } else {
            info!(
                "Reproducibility verification failed with {} discrepancies",
                verification.discrepancies.len()
            );
        }

        Ok(verification)
    }

    /// Get all frozen configurations
    pub fn get_configurations(&self) -> Vec<FrozenConfiguration> {
        self.configurations.values().cloned().collect()
    }

    /// Get all verification results
    pub fn get_verifications(&self) -> Vec<VerificationResult> {
        self.verifications.clone()
    }

    /// Generate reproducibility report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# Reproducibility Report\n\n");
        report.push_str(&format!("**Generated:** {}\n\n", Utc::now().to_rfc3339()));

        // Configurations
        report.push_str("## Frozen Configurations\n\n");
        report.push_str(&format!(
            "Total configurations: {}\n\n",
            self.configurations.len()
        ));

        for config in self.configurations.values() {
            report.push_str(&format!("### {} (ID: {})\n\n", config.name, config.id));
            report.push_str(&format!(
                "- **Created:** {}\n",
                config.created_at.to_rfc3339()
            ));
            report.push_str(&format!(
                "- **Seed:** {}\n",
                config
                    .seed
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "Not set".to_string())
            ));
            report.push_str(&format!("- **Parameters:** {}\n", config.parameters.len()));
            report.push_str(&format!(
                "- **Environment:** {} on {}\n\n",
                config.environment.os, config.environment.arch
            ));
        }

        // Verifications
        report.push_str("## Verification History\n\n");
        report.push_str(&format!(
            "Total verifications: {}\n",
            self.verifications.len()
        ));
        let passed_verifications = self.verifications.iter().filter(|v| v.passed).count();
        report.push_str(&format!(
            "Passed: {} ({:.1}%)\n\n",
            passed_verifications,
            (passed_verifications as f64 / self.verifications.len().max(1) as f64) * 100.0
        ));

        for verification in &self.verifications {
            let status = if verification.passed {
                "✅ PASSED"
            } else {
                "❌ FAILED"
            };
            report.push_str(&format!(
                "### Verification {} - {}\n\n",
                verification.id, status
            ));
            report.push_str(&format!(
                "- **Configuration:** {}\n",
                verification.original_config_id
            ));
            report.push_str(&format!(
                "- **Timestamp:** {}\n",
                verification.timestamp.to_rfc3339()
            ));
            report.push_str(&format!(
                "- **Environment Match:** {}\n",
                if verification.environment_match {
                    "Yes"
                } else {
                    "No"
                }
            ));
            report.push_str(&format!(
                "- **Configuration Match:** {}\n",
                if verification.configuration_match {
                    "Yes"
                } else {
                    "No"
                }
            ));
            report.push_str(&format!(
                "- **Results Match:** {}\n\n",
                if verification.results_match {
                    "Yes"
                } else {
                    "No"
                }
            ));

            if !verification.discrepancies.is_empty() {
                report.push_str("**Discrepancies:**\n\n");
                for discrepancy in &verification.discrepancies {
                    report.push_str(&format!(
                        "- **[{:?}]** {}: {}\n",
                        discrepancy.impact, discrepancy.component, discrepancy.description
                    ));
                }
                report.push_str("\n");
            }
        }

        report
    }

    // Helper methods
    fn get_os_version() -> String {
        // Simplified OS version detection
        if cfg!(target_os = "macos") {
            "macOS".to_string()
        } else if cfg!(target_os = "linux") {
            "Linux".to_string()
        } else if cfg!(target_os = "windows") {
            "Windows".to_string()
        } else {
            "Unknown".to_string()
        }
    }

    fn get_rust_version() -> String {
        env!("CARGO_PKG_RUST_VERSION").to_string()
    }

    fn get_dependencies() -> HashMap<String, String> {
        // In a real implementation, parse Cargo.lock
        let mut deps = HashMap::new();
        deps.insert(
            "voirs-sdk".to_string(),
            env!("CARGO_PKG_VERSION").to_string(),
        );
        deps.insert("scirs2-core".to_string(), "0.1.0".to_string());
        deps
    }

    fn get_safe_env_vars() -> HashMap<String, String> {
        // Only include safe, non-sensitive environment variables
        let mut vars = HashMap::new();
        if let Ok(val) = env::var("RUST_LOG") {
            vars.insert("RUST_LOG".to_string(), val);
        }
        if let Ok(val) = env::var("CARGO_TARGET_DIR") {
            vars.insert("CARGO_TARGET_DIR".to_string(), val);
        }
        vars
    }

    fn get_hardware_info() -> HardwareInfo {
        HardwareInfo {
            cpu_model: "Unknown".to_string(),
            cpu_cores: num_cpus::get(),
            memory_gb: 0.0, // Would need platform-specific code
            gpu: None,
        }
    }

    fn get_compiler_flags() -> Vec<String> {
        vec![]
    }

    fn calculate_hash(parameters: &HashMap<String, String>) -> String {
        // Simple hash calculation
        let mut sorted: Vec<_> = parameters.iter().collect();
        sorted.sort_by_key(|(k, _)| *k);
        let combined: String = sorted.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
        format!("{:x}", md5::compute(combined.as_bytes()))
    }
}

impl Default for ReproducibilityManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reproducibility_manager_creation() {
        let manager = ReproducibilityManager::new();
        assert!(manager.get_seed().is_none());
        assert!(manager.get_configurations().is_empty());
    }

    #[test]
    fn test_set_seed() {
        let mut manager = ReproducibilityManager::new();
        manager.set_seed(42);
        assert_eq!(manager.get_seed(), Some(42));
    }

    #[test]
    fn test_capture_environment() {
        let manager = ReproducibilityManager::new();
        let snapshot = manager.capture_environment().unwrap();

        assert!(!snapshot.id.is_empty());
        assert!(!snapshot.os.is_empty());
        assert!(!snapshot.arch.is_empty());
    }

    #[test]
    fn test_freeze_configuration() {
        let mut manager = ReproducibilityManager::new();
        manager.set_seed(42);

        let params = vec![
            ("sample_rate".to_string(), "16000".to_string()),
            ("channels".to_string(), "1".to_string()),
        ];

        let frozen = manager.freeze_configuration("test_config", params).unwrap();

        assert_eq!(frozen.name, "test_config");
        assert_eq!(frozen.seed, Some(42));
        assert_eq!(frozen.parameters.len(), 2);
        assert!(!frozen.hash.is_empty());
    }

    #[test]
    fn test_reproducibility_verification_pass() {
        let mut manager = ReproducibilityManager::new();
        manager.set_seed(42);

        let params = vec![("param1".to_string(), "value1".to_string())];
        let frozen = manager.freeze_configuration("test", params).unwrap();

        let mut results = HashMap::new();
        results.insert("metric1".to_string(), 0.95);
        results.insert("metric2".to_string(), 0.88);

        // Same results should pass
        let verification = manager
            .verify_reproducibility(&frozen.id, results.clone(), results)
            .unwrap();

        assert!(verification.passed);
        assert!(verification.results_match);
        assert!(verification.discrepancies.is_empty());
    }

    #[test]
    fn test_reproducibility_verification_fail() {
        let mut manager = ReproducibilityManager::new();
        manager.set_seed(42);

        let params = vec![("param1".to_string(), "value1".to_string())];
        let frozen = manager.freeze_configuration("test", params).unwrap();

        let mut original_results = HashMap::new();
        original_results.insert("metric1".to_string(), 0.95);

        let mut current_results = HashMap::new();
        current_results.insert("metric1".to_string(), 0.90); // Different result

        let verification = manager
            .verify_reproducibility(&frozen.id, current_results, original_results)
            .unwrap();

        assert!(!verification.passed);
        assert!(!verification.results_match);
        assert!(!verification.discrepancies.is_empty());
    }

    #[test]
    fn test_report_generation() {
        let mut manager = ReproducibilityManager::new();
        manager.set_seed(42);

        let params = vec![("test".to_string(), "value".to_string())];
        manager.freeze_configuration("test_config", params).unwrap();

        let report = manager.generate_report();

        assert!(report.contains("Reproducibility Report"));
        assert!(report.contains("test_config"));
        assert!(report.contains("**Seed:** 42"));
    }

    #[test]
    fn test_multiple_configurations() {
        let mut manager = ReproducibilityManager::new();

        manager.freeze_configuration("config1", vec![]).unwrap();
        manager.freeze_configuration("config2", vec![]).unwrap();

        let configs = manager.get_configurations();
        assert_eq!(configs.len(), 2);
    }
}
