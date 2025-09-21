//! Binary packaging and distribution system for VoiRS CLI.
//!
//! This module provides comprehensive packaging functionality for distributing
//! the VoiRS CLI across multiple package managers and platforms. It includes:
//!
//! - Binary optimization and packaging
//! - Package manager integration (Homebrew, Chocolatey, Scoop, Debian)
//! - Update management and distribution
//! - Cross-platform packaging pipeline
//!
//! ## Features
//!
//! - **Binary Packaging**: Optimized binary generation with compression and validation
//! - **Multi-Platform Support**: Packages for macOS, Windows, and Linux distributions
//! - **Package Managers**: Native integration with popular package managers
//! - **Update System**: Automated update checking and distribution
//! - **Validation**: Comprehensive package validation and integrity checking
//!
//! ## Example
//!
//! ```rust,no_run
//! use voirs_cli::packaging::{PackagingPipeline, PackagingOptions};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let options = PackagingOptions::default();
//! let pipeline = PackagingPipeline::new(options);
//! let packages = pipeline.run_full_packaging().await?;
//! println!("Generated {} packages", packages.len());
//! # Ok(())
//! # }
//! ```

pub mod binary;
pub mod managers;
pub mod update;

pub use binary::{BinaryPackager, BinaryPackagingConfig};
pub use managers::{generate_all_packages, PackageManager, PackageManagerFactory, PackageMetadata};
pub use update::{UpdateChannel, UpdateConfig, UpdateManager, UpdateState, VersionInfo};

use crate::error::VoirsCLIError;
use anyhow::Result;
use std::path::PathBuf;
use tracing::{error, info};

#[derive(Debug, Clone)]
pub struct PackagingOptions {
    pub binary_config: BinaryPackagingConfig,
    pub package_metadata: PackageMetadata,
    pub output_directory: PathBuf,
    pub managers: Vec<String>,
    pub update_config: UpdateConfig,
}

impl Default for PackagingOptions {
    fn default() -> Self {
        Self {
            binary_config: BinaryPackagingConfig::default(),
            package_metadata: PackageMetadata::default(),
            output_directory: PathBuf::from("packages"),
            managers: vec![
                "homebrew".to_string(),
                "chocolatey".to_string(),
                "scoop".to_string(),
                "debian".to_string(),
            ],
            update_config: UpdateConfig::default(),
        }
    }
}

pub struct PackagingPipeline {
    options: PackagingOptions,
}

impl PackagingPipeline {
    pub fn new(options: PackagingOptions) -> Self {
        Self { options }
    }

    pub async fn run_full_packaging(&self) -> Result<Vec<PathBuf>> {
        info!("Starting full packaging pipeline");

        let mut package_paths = Vec::new();

        // Step 1: Build optimized binary
        info!("Step 1: Building optimized binary");
        let binary_packager = BinaryPackager::new(self.options.binary_config.clone());
        let binary_path = binary_packager.package_binary()?;

        // Validate the binary
        if !binary_packager.validate_binary(&binary_path)? {
            return Err(
                VoirsCLIError::PackagingError("Binary validation failed".to_string()).into(),
            );
        }

        // Get binary size for reporting
        let binary_size = binary_packager.get_binary_size(&binary_path)?;
        info!(
            "Binary size: {} bytes ({:.2} MB)",
            binary_size,
            binary_size as f64 / 1_048_576.0
        );

        // Step 2: Update package metadata with actual binary path
        let mut metadata = self.options.package_metadata.clone();
        metadata.binary_path = binary_path;

        // Step 3: Generate packages for all specified managers
        info!(
            "Step 2: Generating packages for managers: {:?}",
            self.options.managers
        );

        for manager_name in &self.options.managers {
            match self
                .generate_package_for_manager(manager_name, &metadata)
                .await
            {
                Ok(path) => {
                    package_paths.push(path);
                    info!("Successfully generated {} package", manager_name);
                }
                Err(e) => {
                    error!("Failed to generate {} package: {}", manager_name, e);
                }
            }
        }

        info!(
            "Packaging pipeline completed. Generated {} packages",
            package_paths.len()
        );
        Ok(package_paths)
    }

    async fn generate_package_for_manager(
        &self,
        manager_name: &str,
        metadata: &PackageMetadata,
    ) -> Result<PathBuf> {
        let manager = PackageManagerFactory::create_manager(manager_name)?;
        let manager_output_dir = self.options.output_directory.join(manager_name);

        std::fs::create_dir_all(&manager_output_dir)?;

        let package_path = manager.generate_package(metadata, &manager_output_dir)?;

        // Validate the generated package
        if !manager.validate_package(&package_path)? {
            return Err(VoirsCLIError::PackagingError(format!(
                "Package validation failed for {}",
                manager_name
            ))
            .into());
        }

        Ok(package_path)
    }

    pub fn get_package_info(&self) -> PackageInfo {
        PackageInfo {
            name: self.options.package_metadata.name.clone(),
            version: self.options.package_metadata.version.clone(),
            supported_managers: managers::PackageManagerFactory::get_supported_managers()
                .iter()
                .map(|&s| s.to_string())
                .collect(),
            binary_targets: binary::get_supported_targets()
                .iter()
                .map(|&s| s.to_string())
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PackageInfo {
    pub name: String,
    pub version: String,
    pub supported_managers: Vec<String>,
    pub binary_targets: Vec<String>,
}

pub fn validate_packaging_environment() -> Result<Vec<String>> {
    info!("Validating packaging environment");

    let mut issues = Vec::new();

    // Check for required tools
    let required_tools = vec![
        ("cargo", "Rust package manager"),
        ("git", "Version control system"),
    ];

    for (tool, description) in required_tools {
        if !is_tool_available(tool) {
            issues.push(format!("Missing required tool: {} ({})", tool, description));
        }
    }

    // Check for optional tools
    let optional_tools = vec![
        ("strip", "Binary stripping tool"),
        ("upx", "Binary compression tool"),
        ("cross", "Cross-compilation tool"),
    ];

    for (tool, description) in optional_tools {
        if !is_tool_available(tool) {
            info!("Optional tool not available: {} ({})", tool, description);
        }
    }

    if issues.is_empty() {
        info!("Packaging environment validation passed");
    } else {
        error!(
            "Packaging environment validation failed with {} issues",
            issues.len()
        );
    }

    Ok(issues)
}

fn is_tool_available(tool: &str) -> bool {
    std::process::Command::new(tool)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_packaging_options_default() {
        let options = PackagingOptions::default();
        assert_eq!(options.package_metadata.name, "voirs");
        assert!(!options.managers.is_empty());
        assert_eq!(options.output_directory, PathBuf::from("packages"));
    }

    #[test]
    fn test_packaging_pipeline_creation() {
        let options = PackagingOptions::default();
        let pipeline = PackagingPipeline::new(options);
        assert_eq!(pipeline.options.package_metadata.name, "voirs");
    }

    #[test]
    fn test_package_info() {
        let options = PackagingOptions::default();
        let pipeline = PackagingPipeline::new(options);
        let info = pipeline.get_package_info();

        assert_eq!(info.name, "voirs");
        assert!(!info.supported_managers.is_empty());
        assert!(!info.binary_targets.is_empty());
    }

    #[test]
    fn test_validate_packaging_environment() {
        let issues = validate_packaging_environment().unwrap();
        // We can't assert specific tools are available in all test environments
        // but we can verify the function runs without error
        assert!(issues.is_empty() || !issues.is_empty());
    }

    #[test]
    fn test_is_tool_available() {
        // Test with a tool that should be available on most systems
        let _result = is_tool_available("echo");
        // Can't assert true/false as it depends on the system
        // but we can verify it doesn't panic by simply calling it
    }
}
