use crate::error::VoirsCLIError;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub license: String,
    pub homepage: String,
    pub repository: String,
    pub author: String,
    pub maintainer: String,
    pub dependencies: Vec<String>,
    pub binary_path: PathBuf,
}

impl Default for PackageMetadata {
    fn default() -> Self {
        Self {
            name: "voirs".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: "VoiRS Speech Synthesis CLI Tool".to_string(),
            license: "MIT".to_string(),
            homepage: "https://github.com/voirs-org/voirs".to_string(),
            repository: "https://github.com/voirs-org/voirs".to_string(),
            author: "VoiRS Team".to_string(),
            maintainer: "VoiRS Team <voirs@example.com>".to_string(),
            dependencies: vec![],
            binary_path: PathBuf::from("target/release/voirs"),
        }
    }
}

pub trait PackageManager {
    fn generate_package(&self, metadata: &PackageMetadata, output_dir: &PathBuf)
        -> Result<PathBuf>;
    fn validate_package(&self, package_path: &PathBuf) -> Result<bool>;
    fn get_package_name(&self) -> &str;
    fn get_file_extension(&self) -> &str;
}

pub struct HomebrewManager {
    formula_template: String,
}

impl HomebrewManager {
    pub fn new() -> Self {
        Self {
            formula_template: include_str!("templates/homebrew.rb").to_string(),
        }
    }
}

impl PackageManager for HomebrewManager {
    fn generate_package(
        &self,
        metadata: &PackageMetadata,
        output_dir: &PathBuf,
    ) -> Result<PathBuf> {
        info!("Generating Homebrew formula");

        let formula_content = self
            .formula_template
            .replace("{{NAME}}", &metadata.name)
            .replace("{{VERSION}}", &metadata.version)
            .replace("{{DESCRIPTION}}", &metadata.description)
            .replace("{{HOMEPAGE}}", &metadata.homepage)
            .replace("{{REPOSITORY}}", &metadata.repository)
            .replace("{{LICENSE}}", &metadata.license);

        let formula_path = output_dir.join(format!("{}.rb", metadata.name));
        fs::write(&formula_path, formula_content)?;

        info!("Homebrew formula generated at: {:?}", formula_path);
        Ok(formula_path)
    }

    fn validate_package(&self, package_path: &PathBuf) -> Result<bool> {
        debug!("Validating Homebrew formula");

        if !package_path.exists() {
            return Ok(false);
        }

        let content = fs::read_to_string(package_path)?;
        Ok(content.contains("class") && content.contains("Formula"))
    }

    fn get_package_name(&self) -> &str {
        "homebrew"
    }

    fn get_file_extension(&self) -> &str {
        "rb"
    }
}

pub struct ChocolateyManager {
    nuspec_template: String,
    install_script_template: String,
}

impl ChocolateyManager {
    pub fn new() -> Self {
        Self {
            nuspec_template: include_str!("templates/chocolatey.nuspec").to_string(),
            install_script_template: include_str!("templates/chocolatey_install.ps1").to_string(),
        }
    }
}

impl PackageManager for ChocolateyManager {
    fn generate_package(
        &self,
        metadata: &PackageMetadata,
        output_dir: &PathBuf,
    ) -> Result<PathBuf> {
        info!("Generating Chocolatey package");

        let package_dir = output_dir.join(&metadata.name);
        fs::create_dir_all(&package_dir)?;

        // Generate nuspec file
        let nuspec_content = self
            .nuspec_template
            .replace("{{NAME}}", &metadata.name)
            .replace("{{VERSION}}", &metadata.version)
            .replace("{{DESCRIPTION}}", &metadata.description)
            .replace("{{AUTHOR}}", &metadata.author)
            .replace("{{LICENSE}}", &metadata.license);

        let nuspec_path = package_dir.join(format!("{}.nuspec", metadata.name));
        fs::write(&nuspec_path, nuspec_content)?;

        // Generate install script
        let tools_dir = package_dir.join("tools");
        fs::create_dir_all(&tools_dir)?;

        let install_script_content = self.install_script_template.replace(
            "{{BINARY_URL}}",
            &format!(
                "{}/releases/download/v{}/voirs-windows.exe",
                metadata.repository, metadata.version
            ),
        );

        let install_script_path = tools_dir.join("chocolateyinstall.ps1");
        fs::write(&install_script_path, install_script_content)?;

        info!("Chocolatey package generated at: {:?}", package_dir);
        Ok(package_dir)
    }

    fn validate_package(&self, package_path: &PathBuf) -> Result<bool> {
        debug!("Validating Chocolatey package");

        let nuspec_path = package_path.join("*.nuspec");
        let tools_dir = package_path.join("tools");

        Ok(tools_dir.exists()
            && fs::read_dir(package_path)?.any(|entry| {
                entry.ok().map_or(false, |e| {
                    e.path().extension().and_then(|ext| ext.to_str()) == Some("nuspec")
                })
            }))
    }

    fn get_package_name(&self) -> &str {
        "chocolatey"
    }

    fn get_file_extension(&self) -> &str {
        "nupkg"
    }
}

pub struct ScoopManager {
    manifest_template: String,
}

impl ScoopManager {
    pub fn new() -> Self {
        Self {
            manifest_template: include_str!("templates/scoop.json").to_string(),
        }
    }
}

impl PackageManager for ScoopManager {
    fn generate_package(
        &self,
        metadata: &PackageMetadata,
        output_dir: &PathBuf,
    ) -> Result<PathBuf> {
        info!("Generating Scoop manifest");

        let manifest_content = self
            .manifest_template
            .replace("{{NAME}}", &metadata.name)
            .replace("{{VERSION}}", &metadata.version)
            .replace("{{DESCRIPTION}}", &metadata.description)
            .replace("{{HOMEPAGE}}", &metadata.homepage)
            .replace("{{LICENSE}}", &metadata.license)
            .replace("{{REPOSITORY}}", &metadata.repository);

        let manifest_path = output_dir.join(format!("{}.json", metadata.name));
        fs::write(&manifest_path, manifest_content)?;

        info!("Scoop manifest generated at: {:?}", manifest_path);
        Ok(manifest_path)
    }

    fn validate_package(&self, package_path: &PathBuf) -> Result<bool> {
        debug!("Validating Scoop manifest");

        if !package_path.exists() {
            return Ok(false);
        }

        let content = fs::read_to_string(package_path)?;
        let json: serde_json::Value = serde_json::from_str(&content)?;

        Ok(json.get("version").is_some() && json.get("url").is_some())
    }

    fn get_package_name(&self) -> &str {
        "scoop"
    }

    fn get_file_extension(&self) -> &str {
        "json"
    }
}

pub struct DebianManager {
    control_template: String,
}

impl DebianManager {
    pub fn new() -> Self {
        Self {
            control_template: include_str!("templates/debian_control").to_string(),
        }
    }
}

impl PackageManager for DebianManager {
    fn generate_package(
        &self,
        metadata: &PackageMetadata,
        output_dir: &PathBuf,
    ) -> Result<PathBuf> {
        info!("Generating Debian package");

        let package_dir = output_dir.join(format!("{}-{}", metadata.name, metadata.version));
        let debian_dir = package_dir.join("DEBIAN");
        fs::create_dir_all(&debian_dir)?;

        // Generate control file
        let control_content = self
            .control_template
            .replace("{{NAME}}", &metadata.name)
            .replace("{{VERSION}}", &metadata.version)
            .replace("{{DESCRIPTION}}", &metadata.description)
            .replace("{{MAINTAINER}}", &metadata.maintainer)
            .replace("{{DEPENDENCIES}}", &metadata.dependencies.join(", "));

        let control_path = debian_dir.join("control");
        fs::write(&control_path, control_content)?;

        // Copy binary
        let bin_dir = package_dir.join("usr/bin");
        fs::create_dir_all(&bin_dir)?;

        if metadata.binary_path.exists() {
            let dest_binary = bin_dir.join(&metadata.name);
            fs::copy(&metadata.binary_path, &dest_binary)?;

            // Make binary executable
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = fs::metadata(&dest_binary)?.permissions();
                perms.set_mode(0o755);
                fs::set_permissions(&dest_binary, perms)?;
            }
        }

        info!("Debian package structure generated at: {:?}", package_dir);
        Ok(package_dir)
    }

    fn validate_package(&self, package_path: &PathBuf) -> Result<bool> {
        debug!("Validating Debian package");

        let debian_dir = package_path.join("DEBIAN");
        let control_file = debian_dir.join("control");

        Ok(debian_dir.exists() && control_file.exists())
    }

    fn get_package_name(&self) -> &str {
        "debian"
    }

    fn get_file_extension(&self) -> &str {
        "deb"
    }
}

pub struct PackageManagerFactory;

impl PackageManagerFactory {
    pub fn create_manager(manager_type: &str) -> Result<Box<dyn PackageManager>> {
        match manager_type.to_lowercase().as_str() {
            "homebrew" => Ok(Box::new(HomebrewManager::new())),
            "chocolatey" => Ok(Box::new(ChocolateyManager::new())),
            "scoop" => Ok(Box::new(ScoopManager::new())),
            "debian" | "apt" => Ok(Box::new(DebianManager::new())),
            _ => Err(VoirsCLIError::PackagingError(format!(
                "Unsupported package manager: {}",
                manager_type
            ))
            .into()),
        }
    }

    pub fn get_supported_managers() -> Vec<&'static str> {
        vec!["homebrew", "chocolatey", "scoop", "debian"]
    }
}

pub fn generate_all_packages(
    metadata: &PackageMetadata,
    output_dir: &PathBuf,
) -> Result<Vec<PathBuf>> {
    info!("Generating packages for all supported package managers");

    let mut package_paths = Vec::new();

    for manager_type in PackageManagerFactory::get_supported_managers() {
        match PackageManagerFactory::create_manager(manager_type) {
            Ok(manager) => {
                let manager_output_dir = output_dir.join(manager_type);
                fs::create_dir_all(&manager_output_dir)?;

                match manager.generate_package(metadata, &manager_output_dir) {
                    Ok(package_path) => {
                        package_paths.push(package_path);
                        info!("Successfully generated {} package", manager_type);
                    }
                    Err(e) => {
                        warn!("Failed to generate {} package: {}", manager_type, e);
                    }
                }
            }
            Err(e) => {
                warn!("Failed to create {} manager: {}", manager_type, e);
            }
        }
    }

    info!("Generated {} packages", package_paths.len());
    Ok(package_paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_package_metadata_default() {
        let metadata = PackageMetadata::default();
        assert_eq!(metadata.name, "voirs");
        assert!(!metadata.version.is_empty());
        assert!(!metadata.description.is_empty());
    }

    #[test]
    fn test_package_manager_factory() {
        let homebrew = PackageManagerFactory::create_manager("homebrew");
        assert!(homebrew.is_ok());

        let chocolatey = PackageManagerFactory::create_manager("chocolatey");
        assert!(chocolatey.is_ok());

        let invalid = PackageManagerFactory::create_manager("invalid");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_supported_managers() {
        let managers = PackageManagerFactory::get_supported_managers();
        assert!(managers.contains(&"homebrew"));
        assert!(managers.contains(&"chocolatey"));
        assert!(managers.contains(&"scoop"));
        assert!(managers.contains(&"debian"));
    }

    #[test]
    fn test_homebrew_manager_properties() {
        let manager = HomebrewManager::new();
        assert_eq!(manager.get_package_name(), "homebrew");
        assert_eq!(manager.get_file_extension(), "rb");
    }

    #[test]
    fn test_chocolatey_manager_properties() {
        let manager = ChocolateyManager::new();
        assert_eq!(manager.get_package_name(), "chocolatey");
        assert_eq!(manager.get_file_extension(), "nupkg");
    }
}
