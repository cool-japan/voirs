use crate::error::VoirsCLIError;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryPackagingConfig {
    pub target_triple: String,
    pub output_dir: PathBuf,
    pub static_linking: bool,
    pub optimize_size: bool,
    pub strip_debug: bool,
    pub compress_binary: bool,
    pub cross_compile: bool,
}

impl Default for BinaryPackagingConfig {
    fn default() -> Self {
        Self {
            target_triple: get_default_target_triple(),
            output_dir: PathBuf::from("target/release"),
            static_linking: true,
            optimize_size: true,
            strip_debug: true,
            compress_binary: false,
            cross_compile: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BinaryPackager {
    config: BinaryPackagingConfig,
}

impl BinaryPackager {
    pub fn new(config: BinaryPackagingConfig) -> Self {
        Self { config }
    }

    pub fn package_binary(&self) -> Result<PathBuf> {
        info!("Starting binary packaging process");

        // Ensure output directory exists
        fs::create_dir_all(&self.config.output_dir)?;

        // Build the binary with optimization flags
        let binary_path = self.build_optimized_binary()?;

        // Apply post-build optimizations
        let optimized_path = self.post_build_optimize(&binary_path)?;

        // Optional compression
        let final_path = if self.config.compress_binary {
            self.compress_binary(&optimized_path)?
        } else {
            optimized_path
        };

        info!("Binary packaging completed: {:?}", final_path);
        Ok(final_path)
    }

    fn build_optimized_binary(&self) -> Result<PathBuf> {
        info!(
            "Building optimized binary for target: {}",
            self.config.target_triple
        );

        let mut cmd = Command::new("cargo");
        cmd.arg("build").arg("--release").arg("--bin").arg("voirs");

        if self.config.cross_compile {
            cmd.arg("--target").arg(&self.config.target_triple);
        }

        // Add optimization flags
        if self.config.optimize_size {
            cmd.env("CARGO_PROFILE_RELEASE_OPT_LEVEL", "z");
            cmd.env("CARGO_PROFILE_RELEASE_LTO", "true");
            cmd.env("CARGO_PROFILE_RELEASE_CODEGEN_UNITS", "1");
            cmd.env("CARGO_PROFILE_RELEASE_PANIC", "abort");
        }

        if self.config.static_linking {
            cmd.env(
                "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS",
                "-C target-feature=+crt-static",
            );
        }

        let output = cmd.output()?;

        if !output.status.success() {
            return Err(VoirsCLIError::PackagingError(format!(
                "Failed to build binary: {}",
                String::from_utf8_lossy(&output.stderr)
            ))
            .into());
        }

        let binary_name = get_binary_name(&self.config.target_triple);
        let binary_path = if self.config.cross_compile {
            self.config
                .output_dir
                .join(&self.config.target_triple)
                .join(&binary_name)
        } else {
            self.config.output_dir.join(&binary_name)
        };

        debug!("Binary built at: {:?}", binary_path);
        Ok(binary_path)
    }

    fn post_build_optimize(&self, binary_path: &PathBuf) -> Result<PathBuf> {
        if self.config.strip_debug {
            info!("Stripping debug symbols from binary");
            self.strip_debug_symbols(binary_path)?;
        }

        Ok(binary_path.clone())
    }

    fn strip_debug_symbols(&self, binary_path: &PathBuf) -> Result<()> {
        let strip_cmd = if cfg!(target_os = "macos") {
            "strip"
        } else {
            "strip"
        };

        let output = Command::new(strip_cmd).arg(binary_path).output()?;

        if !output.status.success() {
            warn!(
                "Failed to strip debug symbols: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    fn compress_binary(&self, binary_path: &PathBuf) -> Result<PathBuf> {
        info!("Compressing binary using UPX");

        let compressed_path = binary_path.with_extension("compressed");

        let output = Command::new("upx")
            .arg("--best")
            .arg("--lzma")
            .arg("-o")
            .arg(&compressed_path)
            .arg(binary_path)
            .output();

        match output {
            Ok(output) if output.status.success() => {
                info!("Binary compressed successfully");
                Ok(compressed_path)
            }
            Ok(output) => {
                warn!(
                    "UPX compression failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                Ok(binary_path.clone())
            }
            Err(e) => {
                warn!("UPX not available: {}", e);
                Ok(binary_path.clone())
            }
        }
    }

    pub fn get_binary_size(&self, binary_path: &PathBuf) -> Result<u64> {
        let metadata = fs::metadata(binary_path)?;
        Ok(metadata.len())
    }

    pub fn validate_binary(&self, binary_path: &PathBuf) -> Result<bool> {
        debug!("Validating binary at {:?}", binary_path);

        // Check if file exists and is executable
        if !binary_path.exists() {
            return Err(
                VoirsCLIError::PackagingError("Binary file does not exist".to_string()).into(),
            );
        }

        // Try to execute the binary with --version flag
        let output = Command::new(binary_path).arg("--version").output()?;

        if output.status.success() {
            let version_output = String::from_utf8_lossy(&output.stdout);
            info!(
                "Binary validation successful. Version: {}",
                version_output.trim()
            );
            Ok(true)
        } else {
            Err(VoirsCLIError::PackagingError(
                "Binary validation failed - unable to execute".to_string(),
            )
            .into())
        }
    }
}

fn get_default_target_triple() -> String {
    std::env::var("TARGET").unwrap_or_else(|_| {
        if cfg!(target_os = "windows") {
            "x86_64-pc-windows-msvc".to_string()
        } else if cfg!(target_os = "macos") {
            "x86_64-apple-darwin".to_string()
        } else {
            "x86_64-unknown-linux-gnu".to_string()
        }
    })
}

fn get_binary_name(target_triple: &str) -> String {
    if target_triple.contains("windows") {
        "voirs.exe".to_string()
    } else {
        "voirs".to_string()
    }
}

pub fn get_supported_targets() -> Vec<&'static str> {
    vec![
        "x86_64-unknown-linux-gnu",
        "x86_64-unknown-linux-musl",
        "x86_64-pc-windows-msvc",
        "x86_64-apple-darwin",
        "aarch64-apple-darwin",
        "aarch64-unknown-linux-gnu",
        "armv7-unknown-linux-gnueabihf",
    ]
}

pub fn setup_cross_compilation() -> Result<()> {
    info!("Setting up cross-compilation environment");

    // Check if cross is installed
    let cross_check = Command::new("cross").arg("--version").output();

    if cross_check.is_err() {
        info!("Installing cross for cross-compilation");
        let install_output = Command::new("cargo").arg("install").arg("cross").output()?;

        if !install_output.status.success() {
            return Err(
                VoirsCLIError::PackagingError("Failed to install cross tool".to_string()).into(),
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_binary_packaging_config_default() {
        let config = BinaryPackagingConfig::default();
        assert!(!config.target_triple.is_empty());
        assert!(config.static_linking);
        assert!(config.optimize_size);
        assert!(config.strip_debug);
    }

    #[test]
    fn test_get_binary_name() {
        assert_eq!(get_binary_name("x86_64-pc-windows-msvc"), "voirs.exe");
        assert_eq!(get_binary_name("x86_64-unknown-linux-gnu"), "voirs");
        assert_eq!(get_binary_name("x86_64-apple-darwin"), "voirs");
    }

    #[test]
    fn test_supported_targets() {
        let targets = get_supported_targets();
        assert!(targets.contains(&"x86_64-unknown-linux-gnu"));
        assert!(targets.contains(&"x86_64-pc-windows-msvc"));
        assert!(targets.contains(&"x86_64-apple-darwin"));
    }

    #[test]
    fn test_binary_packager_creation() {
        let config = BinaryPackagingConfig::default();
        let packager = BinaryPackager::new(config.clone());
        assert_eq!(packager.config.target_triple, config.target_triple);
    }

    #[test]
    fn test_get_default_target_triple() {
        let target = get_default_target_triple();
        assert!(!target.is_empty());
        assert!(target.contains("x86_64") || target.contains("aarch64"));
    }
}
