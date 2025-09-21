//! Package Management
//!
//! This module provides comprehensive package management support for VoiRS,
//! including Debian packages (.deb), Red Hat packages (.rpm), Flatpak,
//! Snap packages, and other distribution package formats for seamless deployment.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Package management system
pub struct PackageManager {
    pub target_arch: TargetArchitecture,
    pub build_profiles: Vec<BuildProfile>,
    pub metadata: PackageMetadata,
}

/// Target architecture for packages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetArchitecture {
    X86_64,
    Aarch64,
    Armhf,
    I386,
    Universal,
}

/// Build profile configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildProfile {
    pub name: String,
    pub target_os: String,
    pub target_arch: TargetArchitecture,
    pub features: Vec<String>,
    pub dependencies: Vec<String>,
    pub runtime_deps: Vec<String>,
}

/// Package metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub maintainer: String,
    pub homepage: String,
    pub license: String,
    pub section: String,
    pub priority: String,
    pub architecture: String,
    pub depends: Vec<String>,
    pub recommends: Vec<String>,
    pub suggests: Vec<String>,
    pub conflicts: Vec<String>,
    pub replaces: Vec<String>,
    pub provides: Vec<String>,
}

/// Debian package configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebianPackage {
    pub control: DebianControl,
    pub install_files: Vec<InstallFile>,
    pub scripts: PackageScripts,
    pub changelog: Vec<ChangelogEntry>,
    pub copyright: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebianControl {
    pub package: String,
    pub version: String,
    pub section: String,
    pub priority: String,
    pub architecture: String,
    pub essential: bool,
    pub depends: Vec<String>,
    pub pre_depends: Vec<String>,
    pub recommends: Vec<String>,
    pub suggests: Vec<String>,
    pub enhances: Vec<String>,
    pub breaks: Vec<String>,
    pub conflicts: Vec<String>,
    pub maintainer: String,
    pub description: String,
    pub homepage: String,
    pub built_using: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallFile {
    pub source: String,
    pub destination: String,
    pub permissions: String,
    pub owner: String,
    pub group: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageScripts {
    pub preinst: Option<String>,
    pub postinst: Option<String>,
    pub prerm: Option<String>,
    pub postrm: Option<String>,
    pub config: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangelogEntry {
    pub version: String,
    pub distribution: String,
    pub urgency: String,
    pub changes: Vec<String>,
    pub author: String,
    pub date: String,
}

/// RPM package configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpmPackage {
    pub spec: RpmSpec,
    pub install_files: Vec<InstallFile>,
    pub scripts: RpmScripts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpmSpec {
    pub name: String,
    pub version: String,
    pub release: String,
    pub summary: String,
    pub license: String,
    pub group: String,
    pub url: String,
    pub vendor: String,
    pub packager: String,
    pub architecture: String,
    pub requires: Vec<String>,
    pub provides: Vec<String>,
    pub conflicts: Vec<String>,
    pub obsoletes: Vec<String>,
    pub build_requires: Vec<String>,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpmScripts {
    pub prep: Option<String>,
    pub build: Option<String>,
    pub install: Option<String>,
    pub clean: Option<String>,
    pub pre: Option<String>,
    pub post: Option<String>,
    pub preun: Option<String>,
    pub postun: Option<String>,
}

/// Flatpak package configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatpakPackage {
    pub manifest: FlatpakManifest,
    pub permissions: FlatpakPermissions,
    pub modules: Vec<FlatpakModule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatpakManifest {
    pub app_id: String,
    pub runtime: String,
    pub runtime_version: String,
    pub sdk: String,
    pub command: String,
    pub finish_args: Vec<String>,
    pub cleanup: Vec<String>,
    pub build_options: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatpakPermissions {
    pub shared: Vec<String>,
    pub sockets: Vec<String>,
    pub devices: Vec<String>,
    pub features: Vec<String>,
    pub filesystems: Vec<String>,
    pub persistent: Vec<String>,
    pub environment: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatpakModule {
    pub name: String,
    pub buildsystem: String,
    pub sources: Vec<FlatpakSource>,
    pub config_opts: Vec<String>,
    pub build_commands: Vec<String>,
    pub install_rule: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatpakSource {
    pub source_type: String,
    pub url: Option<String>,
    pub sha256: Option<String>,
    pub path: Option<String>,
}

/// Snap package configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapPackage {
    pub snapcraft: SnapcraftYaml,
    pub parts: HashMap<String, SnapPart>,
    pub apps: HashMap<String, SnapApp>,
    pub hooks: HashMap<String, SnapHook>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapcraftYaml {
    pub name: String,
    pub version: String,
    pub summary: String,
    pub description: String,
    pub icon: String,
    pub confinement: String,
    pub grade: String,
    pub base: String,
    pub architectures: Vec<String>,
    pub epoch: String,
    pub license: String,
    pub title: String,
    pub contact: String,
    pub issues: String,
    pub source_code: String,
    pub website: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapPart {
    pub plugin: String,
    pub source: String,
    pub source_type: String,
    pub build_packages: Vec<String>,
    pub stage_packages: Vec<String>,
    pub build_environment: HashMap<String, String>,
    pub override_build: Option<String>,
    pub override_stage: Option<String>,
    pub override_prime: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapApp {
    pub command: String,
    pub desktop: Option<String>,
    pub plugs: Vec<String>,
    pub slots: Vec<String>,
    pub environment: HashMap<String, String>,
    pub daemon: Option<String>,
    pub restart_condition: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapHook {
    pub plugs: Vec<String>,
    pub environment: HashMap<String, String>,
}

impl PackageManager {
    /// Create a new package manager
    pub fn new() -> Self {
        Self {
            target_arch: Self::detect_architecture(),
            build_profiles: Self::create_default_profiles(),
            metadata: Self::create_default_metadata(),
        }
    }

    /// Detect current system architecture
    fn detect_architecture() -> TargetArchitecture {
        match std::env::consts::ARCH {
            "x86_64" => TargetArchitecture::X86_64,
            "aarch64" => TargetArchitecture::Aarch64,
            "arm" => TargetArchitecture::Armhf,
            "x86" => TargetArchitecture::I386,
            _ => TargetArchitecture::X86_64,
        }
    }

    /// Create default build profiles
    fn create_default_profiles() -> Vec<BuildProfile> {
        vec![
            BuildProfile {
                name: "release".to_string(),
                target_os: "linux".to_string(),
                target_arch: TargetArchitecture::X86_64,
                features: vec!["default".to_string(), "performance".to_string()],
                dependencies: vec![
                    "libc6".to_string(),
                    "libgcc1".to_string(),
                    "libstdc++6".to_string(),
                ],
                runtime_deps: vec!["libasound2".to_string(), "libpulse0".to_string()],
            },
            BuildProfile {
                name: "debug".to_string(),
                target_os: "linux".to_string(),
                target_arch: TargetArchitecture::X86_64,
                features: vec!["default".to_string(), "debug".to_string()],
                dependencies: vec![
                    "libc6".to_string(),
                    "libgcc1".to_string(),
                    "libstdc++6".to_string(),
                ],
                runtime_deps: vec![
                    "libasound2".to_string(),
                    "libpulse0".to_string(),
                    "gdb".to_string(),
                ],
            },
        ]
    }

    /// Create default package metadata
    fn create_default_metadata() -> PackageMetadata {
        PackageMetadata {
            name: "voirs-ffi".to_string(),
            version: "0.1.0".to_string(),
            description: "VoiRS speech synthesis library - FFI bindings".to_string(),
            maintainer: "VoiRS Team <team@voirs.dev>".to_string(),
            homepage: "https://github.com/cool-japan/voirs".to_string(),
            license: "MIT".to_string(),
            section: "sound".to_string(),
            priority: "optional".to_string(),
            architecture: "amd64".to_string(),
            depends: vec![
                "libc6 (>= 2.31)".to_string(),
                "libgcc-s1 (>= 3.0)".to_string(),
                "libstdc++6 (>= 9)".to_string(),
            ],
            recommends: vec!["libasound2".to_string(), "libpulse0".to_string()],
            suggests: vec!["voirs-models".to_string(), "voirs-voices".to_string()],
            conflicts: vec![],
            replaces: vec![],
            provides: vec!["speech-synthesis".to_string()],
        }
    }

    /// Generate Debian package
    pub fn generate_debian_package(
        &self,
        output_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let deb_package = self.create_debian_package();
        let package_dir = output_path.join("debian");
        fs::create_dir_all(&package_dir)?;

        // Generate control file
        self.generate_debian_control(&deb_package, &package_dir)?;

        // Generate install files
        self.generate_debian_install(&deb_package, &package_dir)?;

        // Generate scripts
        self.generate_debian_scripts(&deb_package, &package_dir)?;

        // Generate changelog
        self.generate_debian_changelog(&deb_package, &package_dir)?;

        // Generate copyright
        self.generate_debian_copyright(&deb_package, &package_dir)?;

        // Generate rules file
        self.generate_debian_rules(&package_dir)?;

        println!(
            "Debian package files generated in: {}",
            package_dir.display()
        );
        Ok(())
    }

    /// Create Debian package configuration
    fn create_debian_package(&self) -> DebianPackage {
        DebianPackage {
            control: DebianControl {
                package: self.metadata.name.clone(),
                version: self.metadata.version.clone(),
                section: self.metadata.section.clone(),
                priority: self.metadata.priority.clone(),
                architecture: self.metadata.architecture.clone(),
                essential: false,
                depends: self.metadata.depends.clone(),
                pre_depends: vec![],
                recommends: self.metadata.recommends.clone(),
                suggests: self.metadata.suggests.clone(),
                enhances: vec![],
                breaks: vec![],
                conflicts: self.metadata.conflicts.clone(),
                maintainer: self.metadata.maintainer.clone(),
                description: self.metadata.description.clone(),
                homepage: self.metadata.homepage.clone(),
                built_using: "Rust".to_string(),
            },
            install_files: vec![
                InstallFile {
                    source: "target/release/libvoirs_ffi.so".to_string(),
                    destination: "/usr/lib/x86_64-linux-gnu/libvoirs_ffi.so.0.1.0".to_string(),
                    permissions: "0644".to_string(),
                    owner: "root".to_string(),
                    group: "root".to_string(),
                },
                InstallFile {
                    source: "include/voirs_ffi.h".to_string(),
                    destination: "/usr/include/voirs_ffi.h".to_string(),
                    permissions: "0644".to_string(),
                    owner: "root".to_string(),
                    group: "root".to_string(),
                },
                InstallFile {
                    source: "target/release/libvoirs_ffi.a".to_string(),
                    destination: "/usr/lib/x86_64-linux-gnu/libvoirs_ffi.a".to_string(),
                    permissions: "0644".to_string(),
                    owner: "root".to_string(),
                    group: "root".to_string(),
                },
            ],
            scripts: PackageScripts {
                preinst: None,
                postinst: Some(
                    r#"#!/bin/bash
set -e

# Create symbolic links for library versioning
cd /usr/lib/x86_64-linux-gnu
ln -sf libvoirs_ffi.so.0.1.0 libvoirs_ffi.so.0
ln -sf libvoirs_ffi.so.0 libvoirs_ffi.so

# Update library cache
ldconfig

# Create VoiRS configuration directory
mkdir -p /etc/voirs
mkdir -p /var/lib/voirs
mkdir -p /var/log/voirs

# Set proper permissions
chown root:root /etc/voirs
chown voirs:voirs /var/lib/voirs /var/log/voirs || true
chmod 755 /etc/voirs /var/lib/voirs /var/log/voirs

echo "VoiRS FFI library installed successfully"
"#
                    .to_string(),
                ),
                prerm: Some(
                    r#"#!/bin/bash
set -e

echo "Preparing to remove VoiRS FFI library"
"#
                    .to_string(),
                ),
                postrm: Some(
                    r#"#!/bin/bash
set -e

# Remove symbolic links
rm -f /usr/lib/x86_64-linux-gnu/libvoirs_ffi.so
rm -f /usr/lib/x86_64-linux-gnu/libvoirs_ffi.so.0

# Update library cache
ldconfig

# Clean up directories if empty
rmdir /etc/voirs 2>/dev/null || true
rmdir /var/lib/voirs 2>/dev/null || true
rmdir /var/log/voirs 2>/dev/null || true

echo "VoiRS FFI library removed successfully"
"#
                    .to_string(),
                ),
                config: None,
            },
            changelog: vec![ChangelogEntry {
                version: "0.1.0-1".to_string(),
                distribution: "unstable".to_string(),
                urgency: "low".to_string(),
                changes: vec![
                    "Initial release of VoiRS FFI library".to_string(),
                    "Complete C API implementation".to_string(),
                    "Memory management optimizations".to_string(),
                    "Cross-platform compatibility".to_string(),
                ],
                author: "VoiRS Team <team@voirs.dev>".to_string(),
                date: chrono::Utc::now()
                    .format("%a, %d %b %Y %H:%M:%S +0000")
                    .to_string(),
            }],
            copyright: "Copyright (c) 2025 VoiRS Team\n\nLicense: MIT".to_string(),
        }
    }

    /// Generate Debian control file
    fn generate_debian_control(
        &self,
        deb: &DebianPackage,
        package_dir: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut control_content = String::new();

        control_content.push_str(&format!("Package: {}\n", deb.control.package));
        control_content.push_str(&format!("Version: {}\n", deb.control.version));
        control_content.push_str(&format!("Section: {}\n", deb.control.section));
        control_content.push_str(&format!("Priority: {}\n", deb.control.priority));
        control_content.push_str(&format!("Architecture: {}\n", deb.control.architecture));
        control_content.push_str(&format!("Maintainer: {}\n", deb.control.maintainer));

        if !deb.control.depends.is_empty() {
            control_content.push_str(&format!("Depends: {}\n", deb.control.depends.join(", ")));
        }

        if !deb.control.recommends.is_empty() {
            control_content.push_str(&format!(
                "Recommends: {}\n",
                deb.control.recommends.join(", ")
            ));
        }

        if !deb.control.suggests.is_empty() {
            control_content.push_str(&format!("Suggests: {}\n", deb.control.suggests.join(", ")));
        }

        if !deb.control.conflicts.is_empty() {
            control_content.push_str(&format!(
                "Conflicts: {}\n",
                deb.control.conflicts.join(", ")
            ));
        }

        control_content.push_str(&format!("Homepage: {}\n", deb.control.homepage));
        control_content.push_str(&format!("Description: {}\n", deb.control.description));

        fs::write(package_dir.join("control"), control_content)?;
        Ok(())
    }

    /// Generate Debian install file
    fn generate_debian_install(
        &self,
        deb: &DebianPackage,
        package_dir: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut install_content = String::new();

        for file in &deb.install_files {
            install_content.push_str(&format!(
                "{} {}\n",
                file.source,
                file.destination.trim_start_matches('/')
            ));
        }

        fs::write(
            package_dir.join(format!("{}.install", deb.control.package)),
            install_content,
        )?;
        Ok(())
    }

    /// Generate Debian scripts
    fn generate_debian_scripts(
        &self,
        deb: &DebianPackage,
        package_dir: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(postinst) = &deb.scripts.postinst {
            fs::write(package_dir.join("postinst"), postinst)?;
            // Make executable
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = fs::metadata(package_dir.join("postinst"))?.permissions();
                perms.set_mode(0o755);
                fs::set_permissions(package_dir.join("postinst"), perms)?;
            }
        }

        if let Some(prerm) = &deb.scripts.prerm {
            fs::write(package_dir.join("prerm"), prerm)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = fs::metadata(package_dir.join("prerm"))?.permissions();
                perms.set_mode(0o755);
                fs::set_permissions(package_dir.join("prerm"), perms)?;
            }
        }

        if let Some(postrm) = &deb.scripts.postrm {
            fs::write(package_dir.join("postrm"), postrm)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = fs::metadata(package_dir.join("postrm"))?.permissions();
                perms.set_mode(0o755);
                fs::set_permissions(package_dir.join("postrm"), perms)?;
            }
        }

        Ok(())
    }

    /// Generate Debian changelog
    fn generate_debian_changelog(
        &self,
        deb: &DebianPackage,
        package_dir: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut changelog_content = String::new();

        for entry in &deb.changelog {
            changelog_content.push_str(&format!(
                "{} ({}) {}; urgency={}\n\n",
                deb.control.package, entry.version, entry.distribution, entry.urgency
            ));

            for change in &entry.changes {
                changelog_content.push_str(&format!("  * {}\n", change));
            }

            changelog_content.push_str(&format!("\n -- {}  {}\n\n", entry.author, entry.date));
        }

        fs::write(package_dir.join("changelog"), changelog_content)?;
        Ok(())
    }

    /// Generate Debian copyright
    fn generate_debian_copyright(
        &self,
        deb: &DebianPackage,
        package_dir: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        fs::write(package_dir.join("copyright"), &deb.copyright)?;
        Ok(())
    }

    /// Generate Debian rules file
    fn generate_debian_rules(&self, package_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let rules_content = r#"#!/usr/bin/make -f

%:
	dh $@

override_dh_auto_build:
	cargo build --release --features default

override_dh_auto_clean:
	cargo clean

override_dh_auto_test:
	cargo test --release

override_dh_strip:
	dh_strip --dbg-package=voirs-ffi-dbg
"#;

        fs::write(package_dir.join("rules"), rules_content)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(package_dir.join("rules"))?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(package_dir.join("rules"), perms)?;
        }

        Ok(())
    }

    /// Generate RPM package
    pub fn generate_rpm_package(
        &self,
        output_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let rpm_package = self.create_rpm_package();
        let rpm_dir = output_path.join("rpm");
        fs::create_dir_all(&rpm_dir)?;

        // Generate spec file
        self.generate_rpm_spec(&rpm_package, &rpm_dir)?;

        println!("RPM package files generated in: {}", rpm_dir.display());
        Ok(())
    }

    /// Create RPM package configuration
    fn create_rpm_package(&self) -> RpmPackage {
        RpmPackage {
            spec: RpmSpec {
                name: self.metadata.name.clone(),
                version: self.metadata.version.clone(),
                release: "1".to_string(),
                summary: "VoiRS speech synthesis library - FFI bindings".to_string(),
                license: self.metadata.license.clone(),
                group: "System Environment/Libraries".to_string(),
                url: self.metadata.homepage.clone(),
                vendor: "VoiRS Team".to_string(),
                packager: self.metadata.maintainer.clone(),
                architecture: "x86_64".to_string(),
                requires: vec![
                    "glibc >= 2.31".to_string(),
                    "libgcc >= 9.0".to_string(),
                    "libstdc++ >= 9.0".to_string(),
                ],
                provides: vec![
                    "libvoirs_ffi.so.0()(64bit)".to_string(),
                    "speech-synthesis".to_string(),
                ],
                conflicts: vec![],
                obsoletes: vec![],
                build_requires: vec![
                    "rust >= 1.70".to_string(),
                    "cargo".to_string(),
                    "gcc".to_string(),
                    "glibc-devel".to_string(),
                ],
                description: self.metadata.description.clone(),
            },
            install_files: vec![
                InstallFile {
                    source: "target/release/libvoirs_ffi.so".to_string(),
                    destination: "/usr/lib64/libvoirs_ffi.so.0.1.0".to_string(),
                    permissions: "0755".to_string(),
                    owner: "root".to_string(),
                    group: "root".to_string(),
                },
                InstallFile {
                    source: "include/voirs_ffi.h".to_string(),
                    destination: "/usr/include/voirs_ffi.h".to_string(),
                    permissions: "0644".to_string(),
                    owner: "root".to_string(),
                    group: "root".to_string(),
                },
            ],
            scripts: RpmScripts {
                prep: Some("%setup -q".to_string()),
                build: Some("cargo build --release --features default".to_string()),
                install: Some(
                    r#"mkdir -p %{buildroot}/usr/lib64
mkdir -p %{buildroot}/usr/include
cp target/release/libvoirs_ffi.so %{buildroot}/usr/lib64/libvoirs_ffi.so.0.1.0
cp include/voirs_ffi.h %{buildroot}/usr/include/"#
                        .to_string(),
                ),
                clean: Some("cargo clean".to_string()),
                pre: None,
                post: Some(
                    r#"# Create symbolic links
cd /usr/lib64
ln -sf libvoirs_ffi.so.0.1.0 libvoirs_ffi.so.0
ln -sf libvoirs_ffi.so.0 libvoirs_ffi.so
# Update library cache
/sbin/ldconfig"#
                        .to_string(),
                ),
                preun: None,
                postun: Some(
                    r#"# Remove symbolic links
rm -f /usr/lib64/libvoirs_ffi.so
rm -f /usr/lib64/libvoirs_ffi.so.0
# Update library cache
/sbin/ldconfig"#
                        .to_string(),
                ),
            },
        }
    }

    /// Generate RPM spec file
    fn generate_rpm_spec(
        &self,
        rpm: &RpmPackage,
        rpm_dir: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut spec_content = String::new();

        // Header section
        spec_content.push_str(&format!("Name:           {}\n", rpm.spec.name));
        spec_content.push_str(&format!("Version:        {}\n", rpm.spec.version));
        spec_content.push_str(&format!("Release:        {}\n", rpm.spec.release));
        spec_content.push_str(&format!("Summary:        {}\n\n", rpm.spec.summary));

        spec_content.push_str(&format!("License:        {}\n", rpm.spec.license));
        spec_content.push_str(&format!("Group:          {}\n", rpm.spec.group));
        spec_content.push_str(&format!("URL:            {}\n", rpm.spec.url));
        spec_content.push_str(&format!("Vendor:         {}\n", rpm.spec.vendor));
        spec_content.push_str(&format!("Packager:       {}\n\n", rpm.spec.packager));

        // Dependencies
        for req in &rpm.spec.requires {
            spec_content.push_str(&format!("Requires:       {}\n", req));
        }

        for prov in &rpm.spec.provides {
            spec_content.push_str(&format!("Provides:       {}\n", prov));
        }

        for build_req in &rpm.spec.build_requires {
            spec_content.push_str(&format!("BuildRequires:  {}\n", build_req));
        }

        spec_content.push_str("\n");

        // Description
        spec_content.push_str("%description\n");
        spec_content.push_str(&format!("{}\n\n", rpm.spec.description));

        // Prep section
        if let Some(prep) = &rpm.scripts.prep {
            spec_content.push_str("%prep\n");
            spec_content.push_str(&format!("{}\n\n", prep));
        }

        // Build section
        if let Some(build) = &rpm.scripts.build {
            spec_content.push_str("%build\n");
            spec_content.push_str(&format!("{}\n\n", build));
        }

        // Install section
        if let Some(install) = &rpm.scripts.install {
            spec_content.push_str("%install\n");
            spec_content.push_str(&format!("{}\n\n", install));
        }

        // Clean section
        if let Some(clean) = &rpm.scripts.clean {
            spec_content.push_str("%clean\n");
            spec_content.push_str(&format!("{}\n\n", clean));
        }

        // Files section
        spec_content.push_str("%files\n");
        for file in &rpm.install_files {
            spec_content.push_str(&format!("{}\n", file.destination));
        }
        spec_content.push_str("\n");

        // Scripts
        if let Some(post) = &rpm.scripts.post {
            spec_content.push_str("%post\n");
            spec_content.push_str(&format!("{}\n\n", post));
        }

        if let Some(postun) = &rpm.scripts.postun {
            spec_content.push_str("%postun\n");
            spec_content.push_str(&format!("{}\n\n", postun));
        }

        // Changelog
        spec_content.push_str("%changelog\n");
        spec_content.push_str(&format!(
            "* {} {} <{}> {}-{}\n",
            chrono::Utc::now().format("%a %b %d %Y"),
            "VoiRS Team",
            "team@voirs.dev",
            rpm.spec.version,
            rpm.spec.release
        ));
        spec_content.push_str("- Initial RPM release\n");

        fs::write(
            rpm_dir.join(format!("{}.spec", rpm.spec.name)),
            spec_content,
        )?;
        Ok(())
    }

    /// Generate Flatpak package
    pub fn generate_flatpak_package(
        &self,
        output_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let flatpak = self.create_flatpak_package();
        let flatpak_dir = output_path.join("flatpak");
        fs::create_dir_all(&flatpak_dir)?;

        // Generate manifest
        self.generate_flatpak_manifest(&flatpak, &flatpak_dir)?;

        println!(
            "Flatpak package files generated in: {}",
            flatpak_dir.display()
        );
        Ok(())
    }

    /// Create Flatpak package configuration
    fn create_flatpak_package(&self) -> FlatpakPackage {
        FlatpakPackage {
            manifest: FlatpakManifest {
                app_id: "dev.voirs.FFI".to_string(),
                runtime: "org.freedesktop.Platform".to_string(),
                runtime_version: "23.08".to_string(),
                sdk: "org.freedesktop.Sdk".to_string(),
                command: "voirs-demo".to_string(),
                finish_args: vec![
                    "--share=ipc".to_string(),
                    "--socket=x11".to_string(),
                    "--socket=wayland".to_string(),
                    "--socket=pulseaudio".to_string(),
                    "--device=dri".to_string(),
                    "--filesystem=home".to_string(),
                ],
                cleanup: vec![
                    "/include".to_string(),
                    "/lib/pkgconfig".to_string(),
                    "/share/pkgconfig".to_string(),
                    "/share/aclocal".to_string(),
                    "/man".to_string(),
                    "/share/man".to_string(),
                    "*.la".to_string(),
                    "*.a".to_string(),
                ],
                build_options: {
                    let mut options = HashMap::new();
                    options.insert(
                        "cflags".to_string(),
                        serde_json::Value::String("-O2 -g".to_string()),
                    );
                    options.insert(
                        "cxxflags".to_string(),
                        serde_json::Value::String("-O2 -g".to_string()),
                    );
                    options
                },
            },
            permissions: FlatpakPermissions {
                shared: vec!["ipc".to_string()],
                sockets: vec![
                    "x11".to_string(),
                    "wayland".to_string(),
                    "pulseaudio".to_string(),
                ],
                devices: vec!["dri".to_string()],
                features: vec!["devel".to_string()],
                filesystems: vec!["home".to_string()],
                persistent: vec![".voirs".to_string()],
                environment: {
                    let mut env = HashMap::new();
                    env.insert("VOIRS_DATA_DIR".to_string(), "/app/share/voirs".to_string());
                    env
                },
            },
            modules: vec![FlatpakModule {
                name: "voirs-ffi".to_string(),
                buildsystem: "simple".to_string(),
                sources: vec![FlatpakSource {
                    source_type: "archive".to_string(),
                    url: Some(
                        "https://github.com/cool-japan/voirs/archive/v0.1.0.tar.gz".to_string(),
                    ),
                    sha256: Some("placeholder_sha256".to_string()),
                    path: None,
                }],
                config_opts: vec![],
                build_commands: vec![
                    "cargo build --release --features default".to_string(),
                    "install -Dm755 target/release/libvoirs_ffi.so /app/lib/libvoirs_ffi.so"
                        .to_string(),
                    "install -Dm644 include/voirs_ffi.h /app/include/voirs_ffi.h".to_string(),
                ],
                install_rule: "install".to_string(),
            }],
        }
    }

    /// Generate Flatpak manifest
    fn generate_flatpak_manifest(
        &self,
        flatpak: &FlatpakPackage,
        flatpak_dir: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let manifest_json = serde_json::to_string_pretty(&json!({
            "app-id": flatpak.manifest.app_id,
            "runtime": flatpak.manifest.runtime,
            "runtime-version": flatpak.manifest.runtime_version,
            "sdk": flatpak.manifest.sdk,
            "command": flatpak.manifest.command,
            "finish-args": flatpak.manifest.finish_args,
            "cleanup": flatpak.manifest.cleanup,
            "build-options": flatpak.manifest.build_options,
            "modules": flatpak.modules.iter().map(|module| {
                json!({
                    "name": module.name,
                    "buildsystem": module.buildsystem,
                    "sources": module.sources.iter().map(|src| {
                        let mut source_obj = serde_json::Map::new();
                        source_obj.insert("type".to_string(), serde_json::Value::String(src.source_type.clone()));
                        if let Some(url) = &src.url {
                            source_obj.insert("url".to_string(), serde_json::Value::String(url.clone()));
                        }
                        if let Some(sha256) = &src.sha256 {
                            source_obj.insert("sha256".to_string(), serde_json::Value::String(sha256.clone()));
                        }
                        if let Some(path) = &src.path {
                            source_obj.insert("path".to_string(), serde_json::Value::String(path.clone()));
                        }
                        serde_json::Value::Object(source_obj)
                    }).collect::<Vec<_>>(),
                    "build-commands": module.build_commands
                })
            }).collect::<Vec<_>>()
        }))?;

        fs::write(
            flatpak_dir.join(format!("{}.json", flatpak.manifest.app_id)),
            manifest_json,
        )?;
        Ok(())
    }

    /// Generate Snap package
    pub fn generate_snap_package(
        &self,
        output_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let snap = self.create_snap_package();
        let snap_dir = output_path.join("snap");
        fs::create_dir_all(&snap_dir)?;

        // Generate snapcraft.yaml
        self.generate_snapcraft_yaml(&snap, &snap_dir)?;

        println!("Snap package files generated in: {}", snap_dir.display());
        Ok(())
    }

    /// Create Snap package configuration
    fn create_snap_package(&self) -> SnapPackage {
        let mut parts = HashMap::new();
        parts.insert(
            "voirs-ffi".to_string(),
            SnapPart {
                plugin: "rust".to_string(),
                source: ".".to_string(),
                source_type: "local".to_string(),
                build_packages: vec![
                    "build-essential".to_string(),
                    "libasound2-dev".to_string(),
                    "libpulse-dev".to_string(),
                ],
                stage_packages: vec!["libasound2".to_string(), "libpulse0".to_string()],
                build_environment: {
                    let mut env = HashMap::new();
                    env.insert("CARGO_BUILD_FEATURES".to_string(), "default".to_string());
                    env
                },
                override_build: Some(
                    r#"cargo build --release --features default
mkdir -p $SNAPCRAFT_PART_INSTALL/lib
mkdir -p $SNAPCRAFT_PART_INSTALL/include
cp target/release/libvoirs_ffi.so $SNAPCRAFT_PART_INSTALL/lib/
cp include/voirs_ffi.h $SNAPCRAFT_PART_INSTALL/include/"#
                        .to_string(),
                ),
                override_stage: None,
                override_prime: None,
            },
        );

        let mut apps = HashMap::new();
        apps.insert(
            "voirs-demo".to_string(),
            SnapApp {
                command: "bin/voirs-demo".to_string(),
                desktop: Some("share/applications/voirs-demo.desktop".to_string()),
                plugs: vec![
                    "home".to_string(),
                    "audio-playback".to_string(),
                    "audio-record".to_string(),
                    "desktop".to_string(),
                    "desktop-legacy".to_string(),
                    "x11".to_string(),
                    "wayland".to_string(),
                ],
                slots: vec![],
                environment: {
                    let mut env = HashMap::new();
                    env.insert(
                        "LD_LIBRARY_PATH".to_string(),
                        "$SNAP/lib:$LD_LIBRARY_PATH".to_string(),
                    );
                    env
                },
                daemon: None,
                restart_condition: None,
            },
        );

        SnapPackage {
            snapcraft: SnapcraftYaml {
                name: "voirs-ffi".to_string(),
                version: self.metadata.version.clone(),
                summary: "VoiRS speech synthesis library".to_string(),
                description: self.metadata.description.clone(),
                icon: "snap/gui/voirs.png".to_string(),
                confinement: "strict".to_string(),
                grade: "stable".to_string(),
                base: "core22".to_string(),
                architectures: vec!["amd64".to_string()],
                epoch: "0".to_string(),
                license: self.metadata.license.clone(),
                title: "VoiRS Speech Synthesis".to_string(),
                contact: self.metadata.maintainer.clone(),
                issues: "https://github.com/cool-japan/voirs/issues".to_string(),
                source_code: "https://github.com/cool-japan/voirs".to_string(),
                website: self.metadata.homepage.clone(),
            },
            parts,
            apps,
            hooks: HashMap::new(),
        }
    }

    /// Generate snapcraft.yaml
    fn generate_snapcraft_yaml(
        &self,
        snap: &SnapPackage,
        snap_dir: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let yaml_content = serde_yaml::to_string(&json!({
            "name": snap.snapcraft.name,
            "version": snap.snapcraft.version,
            "summary": snap.snapcraft.summary,
            "description": snap.snapcraft.description,
            "icon": snap.snapcraft.icon,
            "confinement": snap.snapcraft.confinement,
            "grade": snap.snapcraft.grade,
            "base": snap.snapcraft.base,
            "architectures": snap.snapcraft.architectures,
            "license": snap.snapcraft.license,
            "title": snap.snapcraft.title,
            "contact": snap.snapcraft.contact,
            "issues": snap.snapcraft.issues,
            "source-code": snap.snapcraft.source_code,
            "website": snap.snapcraft.website,
            "parts": snap.parts,
            "apps": snap.apps
        }))?;

        fs::write(snap_dir.join("snapcraft.yaml"), yaml_content)?;
        Ok(())
    }

    /// Build all package formats
    pub fn build_all_packages(&self, output_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        println!("Building all package formats...");

        self.generate_debian_package(output_path)?;
        self.generate_rpm_package(output_path)?;
        self.generate_flatpak_package(output_path)?;
        self.generate_snap_package(output_path)?;

        println!("All package formats generated successfully!");
        Ok(())
    }
}

impl Default for PackageManager {
    fn default() -> Self {
        Self::new()
    }
}

// Helper macro for JSON creation
use serde_json::json;

/// C API functions for package management
#[no_mangle]
pub extern "C" fn voirs_package_create_manager() -> *mut PackageManager {
    Box::into_raw(Box::new(PackageManager::new()))
}

#[no_mangle]
pub unsafe extern "C" fn voirs_package_destroy_manager(manager: *mut PackageManager) {
    if !manager.is_null() {
        unsafe {
            let _ = Box::from_raw(manager);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn voirs_package_build_debian(
    manager: *mut PackageManager,
    output_path: *const std::os::raw::c_char,
) -> bool {
    if manager.is_null() || output_path.is_null() {
        return false;
    }

    unsafe {
        let output_path_str = match std::ffi::CStr::from_ptr(output_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        };

        let path = Path::new(output_path_str);
        (*manager).generate_debian_package(path).is_ok()
    }
}

#[no_mangle]
pub unsafe extern "C" fn voirs_package_build_all(
    manager: *mut PackageManager,
    output_path: *const std::os::raw::c_char,
) -> bool {
    if manager.is_null() || output_path.is_null() {
        return false;
    }

    unsafe {
        let output_path_str = match std::ffi::CStr::from_ptr(output_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        };

        let path = Path::new(output_path_str);
        (*manager).build_all_packages(path).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_package_manager_creation() {
        let manager = PackageManager::new();
        assert_eq!(manager.metadata.name, "voirs-ffi");
        assert_eq!(manager.metadata.version, "0.1.0");
        assert!(!manager.build_profiles.is_empty());
    }

    #[test]
    fn test_debian_package_generation() {
        let manager = PackageManager::new();
        let temp_dir = TempDir::new().unwrap();

        let result = manager.generate_debian_package(temp_dir.path());
        assert!(result.is_ok());

        let debian_dir = temp_dir.path().join("debian");
        assert!(debian_dir.exists());
        assert!(debian_dir.join("control").exists());
        assert!(debian_dir.join("changelog").exists());
        assert!(debian_dir.join("copyright").exists());
        assert!(debian_dir.join("rules").exists());
    }

    #[test]
    fn test_rpm_package_generation() {
        let manager = PackageManager::new();
        let temp_dir = TempDir::new().unwrap();

        let result = manager.generate_rpm_package(temp_dir.path());
        assert!(result.is_ok());

        let rpm_dir = temp_dir.path().join("rpm");
        assert!(rpm_dir.exists());
        assert!(rpm_dir.join("voirs-ffi.spec").exists());
    }

    #[test]
    fn test_flatpak_package_generation() {
        let manager = PackageManager::new();
        let temp_dir = TempDir::new().unwrap();

        let result = manager.generate_flatpak_package(temp_dir.path());
        assert!(result.is_ok());

        let flatpak_dir = temp_dir.path().join("flatpak");
        assert!(flatpak_dir.exists());
        assert!(flatpak_dir.join("dev.voirs.FFI.json").exists());
    }

    #[test]
    fn test_snap_package_generation() {
        let manager = PackageManager::new();
        let temp_dir = TempDir::new().unwrap();

        let result = manager.generate_snap_package(temp_dir.path());
        assert!(result.is_ok());

        let snap_dir = temp_dir.path().join("snap");
        assert!(snap_dir.exists());
        assert!(snap_dir.join("snapcraft.yaml").exists());
    }

    #[test]
    fn test_build_all_packages() {
        let manager = PackageManager::new();
        let temp_dir = TempDir::new().unwrap();

        let result = manager.build_all_packages(temp_dir.path());
        assert!(result.is_ok());

        // Check all package directories exist
        assert!(temp_dir.path().join("debian").exists());
        assert!(temp_dir.path().join("rpm").exists());
        assert!(temp_dir.path().join("flatpak").exists());
        assert!(temp_dir.path().join("snap").exists());
    }

    #[test]
    fn test_debian_control_content() {
        let manager = PackageManager::new();
        let deb_package = manager.create_debian_package();

        assert_eq!(deb_package.control.package, "voirs-ffi");
        assert_eq!(deb_package.control.version, "0.1.0");
        assert_eq!(deb_package.control.section, "sound");
        assert!(!deb_package.control.depends.is_empty());
        assert!(!deb_package.install_files.is_empty());
    }

    #[test]
    fn test_rpm_spec_content() {
        let manager = PackageManager::new();
        let rpm_package = manager.create_rpm_package();

        assert_eq!(rpm_package.spec.name, "voirs-ffi");
        assert_eq!(rpm_package.spec.version, "0.1.0");
        assert_eq!(rpm_package.spec.license, "MIT");
        assert!(!rpm_package.spec.requires.is_empty());
        assert!(!rpm_package.install_files.is_empty());
    }
}
