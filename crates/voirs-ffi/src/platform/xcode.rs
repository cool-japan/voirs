//! Xcode Integration
//!
//! This module provides integration features for Apple Xcode,
//! including Framework packaging, CocoaPods support, Swift Package Manager,
//! project templates, and Xcode scheme configurations for seamless iOS/macOS development.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Xcode integration manager
pub struct XcodeIntegration {
    pub xcode_version: XcodeVersion,
    pub developer_dir: Option<PathBuf>,
    pub sdk_paths: SdkPaths,
    pub deployment_targets: DeploymentTargets,
}

/// Supported Xcode versions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum XcodeVersion {
    Xcode12,
    Xcode13,
    Xcode14,
    Xcode15,
    Xcode16,
}

/// SDK paths for different platforms
#[derive(Debug, Clone, Default)]
pub struct SdkPaths {
    pub macos: Option<PathBuf>,
    pub ios: Option<PathBuf>,
    pub ios_simulator: Option<PathBuf>,
    pub watchos: Option<PathBuf>,
    pub watchos_simulator: Option<PathBuf>,
    pub tvos: Option<PathBuf>,
    pub tvos_simulator: Option<PathBuf>,
}

/// Deployment target versions
#[derive(Debug, Clone)]
pub struct DeploymentTargets {
    pub macos: String,
    pub ios: String,
    pub watchos: String,
    pub tvos: String,
}

impl Default for DeploymentTargets {
    fn default() -> Self {
        Self {
            macos: "10.15".to_string(),
            ios: "13.0".to_string(),
            watchos: "6.0".to_string(),
            tvos: "13.0".to_string(),
        }
    }
}

/// Swift Package Manager manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwiftPackage {
    pub name: String,
    pub platforms: Vec<Platform>,
    pub products: Vec<Product>,
    pub targets: Vec<Target>,
    pub dependencies: Vec<Dependency>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Platform {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Product {
    pub name: String,
    pub product_type: String,
    pub targets: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Target {
    pub name: String,
    pub target_type: String,
    pub path: Option<String>,
    pub sources: Option<Vec<String>>,
    pub public_headers_path: Option<String>,
    pub c_settings: Option<Vec<CSetting>>,
    pub linker_settings: Option<Vec<LinkerSetting>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CSetting {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkerSetting {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    pub name: String,
    pub url: String,
    pub version: String,
}

/// CocoaPods podspec configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodSpec {
    pub name: String,
    pub version: String,
    pub summary: String,
    pub description: String,
    pub homepage: String,
    pub license: License,
    pub author: Author,
    pub source: Source,
    pub platforms: PodPlatforms,
    pub source_files: Vec<String>,
    pub public_header_files: Vec<String>,
    pub vendored_libraries: Vec<String>,
    pub vendored_frameworks: Vec<String>,
    pub frameworks: Vec<String>,
    pub libraries: Vec<String>,
    pub dependencies: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct License {
    pub license_type: String,
    pub file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Author {
    pub name: String,
    pub email: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub git: String,
    pub tag: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodPlatforms {
    pub ios: String,
    pub osx: String,
    pub watchos: String,
    pub tvos: String,
}

/// Xcode project configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XcodeProject {
    pub name: String,
    pub bundle_identifier: String,
    pub deployment_target: String,
    pub frameworks: Vec<String>,
    pub build_settings: std::collections::HashMap<String, String>,
    pub schemes: Vec<XcodeScheme>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XcodeScheme {
    pub name: String,
    pub build_action: BuildAction,
    pub test_action: TestAction,
    pub launch_action: LaunchAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildAction {
    pub targets: Vec<String>,
    pub parallel_buildables: bool,
    pub build_implicit_dependencies: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestAction {
    pub targets: Vec<String>,
    pub build_configuration: String,
    pub should_use_launch_scheme_args_env: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaunchAction {
    pub build_configuration: String,
    pub launch_style: String,
    pub use_custom_working_directory: bool,
    pub debug_document_versioning: bool,
}

impl XcodeIntegration {
    /// Create a new Xcode integration manager
    pub fn new() -> Self {
        let xcode_version = Self::detect_xcode_version();
        let developer_dir = Self::detect_developer_dir();
        let sdk_paths = Self::detect_sdk_paths(&developer_dir);
        let deployment_targets = DeploymentTargets::default();

        Self {
            xcode_version,
            developer_dir,
            sdk_paths,
            deployment_targets,
        }
    }

    /// Detect installed Xcode version
    fn detect_xcode_version() -> XcodeVersion {
        if let Ok(output) = Command::new("xcodebuild").arg("-version").output() {
            let version_output = String::from_utf8_lossy(&output.stdout);
            if version_output.contains("Xcode 16") {
                XcodeVersion::Xcode16
            } else if version_output.contains("Xcode 15") {
                XcodeVersion::Xcode15
            } else if version_output.contains("Xcode 14") {
                XcodeVersion::Xcode14
            } else if version_output.contains("Xcode 13") {
                XcodeVersion::Xcode13
            } else {
                XcodeVersion::Xcode12
            }
        } else {
            XcodeVersion::Xcode15 // Default fallback
        }
    }

    /// Detect Xcode developer directory
    fn detect_developer_dir() -> Option<PathBuf> {
        if let Ok(output) = Command::new("xcode-select").arg("-p").output() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            let path_str = output_str.trim();
            if !path_str.is_empty() {
                return Some(PathBuf::from(path_str));
            }
        }
        None
    }

    /// Detect SDK paths for different platforms
    fn detect_sdk_paths(developer_dir: &Option<PathBuf>) -> SdkPaths {
        let mut sdk_paths = SdkPaths::default();

        if let Some(dev_dir) = developer_dir {
            // Detect macOS SDK
            if let Ok(output) = Command::new("xcrun")
                .args(&["--sdk", "macosx", "--show-sdk-path"])
                .output()
            {
                let output_str = String::from_utf8_lossy(&output.stdout);
                let path_str = output_str.trim();
                if !path_str.is_empty() {
                    sdk_paths.macos = Some(PathBuf::from(path_str));
                }
            }

            // Detect iOS SDK
            if let Ok(output) = Command::new("xcrun")
                .args(&["--sdk", "iphoneos", "--show-sdk-path"])
                .output()
            {
                let output_str = String::from_utf8_lossy(&output.stdout);
                let path_str = output_str.trim();
                if !path_str.is_empty() {
                    sdk_paths.ios = Some(PathBuf::from(path_str));
                }
            }

            // Detect iOS Simulator SDK
            if let Ok(output) = Command::new("xcrun")
                .args(&["--sdk", "iphonesimulator", "--show-sdk-path"])
                .output()
            {
                let output_str = String::from_utf8_lossy(&output.stdout);
                let path_str = output_str.trim();
                if !path_str.is_empty() {
                    sdk_paths.ios_simulator = Some(PathBuf::from(path_str));
                }
            }

            // Detect watchOS SDK
            if let Ok(output) = Command::new("xcrun")
                .args(&["--sdk", "watchos", "--show-sdk-path"])
                .output()
            {
                let output_str = String::from_utf8_lossy(&output.stdout);
                let path_str = output_str.trim();
                if !path_str.is_empty() {
                    sdk_paths.watchos = Some(PathBuf::from(path_str));
                }
            }

            // Detect tvOS SDK
            if let Ok(output) = Command::new("xcrun")
                .args(&["--sdk", "appletvos", "--show-sdk-path"])
                .output()
            {
                let output_str = String::from_utf8_lossy(&output.stdout);
                let path_str = output_str.trim();
                if !path_str.is_empty() {
                    sdk_paths.tvos = Some(PathBuf::from(path_str));
                }
            }
        }

        sdk_paths
    }

    /// Generate Swift Package Manager manifest
    pub fn generate_swift_package(
        &self,
        output_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let package = self.create_swift_package();
        let package_swift = self.generate_package_swift(&package)?;

        fs::write(output_path.join("Package.swift"), package_swift)?;

        Ok(())
    }

    /// Create Swift Package configuration
    fn create_swift_package(&self) -> SwiftPackage {
        SwiftPackage {
            name: "VoiRS".to_string(),
            platforms: vec![
                Platform {
                    name: "macOS".to_string(),
                    version: self.deployment_targets.macos.clone(),
                },
                Platform {
                    name: "iOS".to_string(),
                    version: self.deployment_targets.ios.clone(),
                },
                Platform {
                    name: "watchOS".to_string(),
                    version: self.deployment_targets.watchos.clone(),
                },
                Platform {
                    name: "tvOS".to_string(),
                    version: self.deployment_targets.tvos.clone(),
                },
            ],
            products: vec![Product {
                name: "VoiRS".to_string(),
                product_type: "library".to_string(),
                targets: vec!["VoiRS".to_string()],
            }],
            targets: vec![
                Target {
                    name: "VoiRS".to_string(),
                    target_type: "target".to_string(),
                    path: Some("Sources/VoiRS".to_string()),
                    sources: None,
                    public_headers_path: Some("include".to_string()),
                    c_settings: Some(vec![
                        CSetting {
                            name: "headerSearchPath".to_string(),
                            value: "include".to_string(),
                        },
                        CSetting {
                            name: "define".to_string(),
                            value: "VOIRS_FFI_AVAILABLE=1".to_string(),
                        },
                    ]),
                    linker_settings: Some(vec![LinkerSetting {
                        name: "linkedLibrary".to_string(),
                        value: "voirs_ffi".to_string(),
                    }]),
                },
                Target {
                    name: "VoiRSTests".to_string(),
                    target_type: "testTarget".to_string(),
                    path: Some("Tests/VoiRSTests".to_string()),
                    sources: None,
                    public_headers_path: None,
                    c_settings: None,
                    linker_settings: None,
                },
            ],
            dependencies: vec![],
        }
    }

    /// Generate Package.swift content
    fn generate_package_swift(
        &self,
        package: &SwiftPackage,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut swift_content = String::new();

        swift_content.push_str("// swift-tools-version: 5.7\n");
        swift_content.push_str("// The swift-tools-version declares the minimum version of Swift required to build this package.\n\n");
        swift_content.push_str("import PackageDescription\n\n");
        swift_content.push_str("let package = Package(\n");
        swift_content.push_str(&format!("    name: \"{}\",\n", package.name));

        // Platforms
        swift_content.push_str("    platforms: [\n");
        for (i, platform) in package.platforms.iter().enumerate() {
            let platform_name = match platform.name.as_str() {
                "macOS" => ".macOS",
                "iOS" => ".iOS",
                "watchOS" => ".watchOS",
                "tvOS" => ".tvOS",
                _ => ".macOS",
            };
            swift_content.push_str(&format!(
                "        {}(\"{}\")",
                platform_name, platform.version
            ));
            if i < package.platforms.len() - 1 {
                swift_content.push(',');
            }
            swift_content.push('\n');
        }
        swift_content.push_str("    ],\n");

        // Products
        swift_content.push_str("    products: [\n");
        for (i, product) in package.products.iter().enumerate() {
            swift_content.push_str(&format!("        .library(\n"));
            swift_content.push_str(&format!("            name: \"{}\",\n", product.name));
            swift_content.push_str(&format!(
                "            targets: [\"{}\"]\n",
                product.targets[0]
            ));
            swift_content.push_str("        )");
            if i < package.products.len() - 1 {
                swift_content.push(',');
            }
            swift_content.push('\n');
        }
        swift_content.push_str("    ],\n");

        // Dependencies
        swift_content.push_str("    dependencies: [\n");
        for (i, dep) in package.dependencies.iter().enumerate() {
            swift_content.push_str(&format!(
                "        .package(url: \"{}\", from: \"{}\")",
                dep.url, dep.version
            ));
            if i < package.dependencies.len() - 1 {
                swift_content.push(',');
            }
            swift_content.push('\n');
        }
        swift_content.push_str("    ],\n");

        // Targets
        swift_content.push_str("    targets: [\n");
        for (i, target) in package.targets.iter().enumerate() {
            let target_type = match target.target_type.as_str() {
                "testTarget" => ".testTarget",
                _ => ".target",
            };
            swift_content.push_str(&format!("        {}(\n", target_type));
            swift_content.push_str(&format!("            name: \"{}\",\n", target.name));

            if let Some(path) = &target.path {
                swift_content.push_str(&format!("            path: \"{}\",\n", path));
            }

            if let Some(public_headers) = &target.public_headers_path {
                swift_content.push_str(&format!(
                    "            publicHeadersPath: \"{}\",\n",
                    public_headers
                ));
            }

            if let Some(c_settings) = &target.c_settings {
                swift_content.push_str("            cSettings: [\n");
                for (j, setting) in c_settings.iter().enumerate() {
                    match setting.name.as_str() {
                        "headerSearchPath" => {
                            swift_content.push_str(&format!(
                                "                .headerSearchPath(\"{}\")",
                                setting.value
                            ));
                        }
                        "define" => {
                            swift_content.push_str(&format!(
                                "                .define(\"{}\")",
                                setting.value
                            ));
                        }
                        _ => {
                            swift_content.push_str(&format!(
                                "                .unsafeFlags([\"-{}\", \"{}\"])",
                                setting.name, setting.value
                            ));
                        }
                    }
                    if j < c_settings.len() - 1 {
                        swift_content.push(',');
                    }
                    swift_content.push('\n');
                }
                swift_content.push_str("            ],\n");
            }

            if let Some(linker_settings) = &target.linker_settings {
                swift_content.push_str("            linkerSettings: [\n");
                for (j, setting) in linker_settings.iter().enumerate() {
                    match setting.name.as_str() {
                        "linkedLibrary" => {
                            swift_content.push_str(&format!(
                                "                .linkedLibrary(\"{}\")",
                                setting.value
                            ));
                        }
                        "linkedFramework" => {
                            swift_content.push_str(&format!(
                                "                .linkedFramework(\"{}\")",
                                setting.value
                            ));
                        }
                        _ => {
                            swift_content.push_str(&format!(
                                "                .unsafeFlags([\"-{}\", \"{}\"])",
                                setting.name, setting.value
                            ));
                        }
                    }
                    if j < linker_settings.len() - 1 {
                        swift_content.push(',');
                    }
                    swift_content.push('\n');
                }
                swift_content.push_str("            ]\n");
            }

            swift_content.push_str("        )");
            if i < package.targets.len() - 1 {
                swift_content.push(',');
            }
            swift_content.push('\n');
        }
        swift_content.push_str("    ]\n");
        swift_content.push_str(")\n");

        Ok(swift_content)
    }

    /// Generate CocoaPods podspec
    pub fn generate_cocoapods_spec(
        &self,
        output_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let podspec = self.create_podspec();
        let podspec_content = self.generate_podspec_content(&podspec)?;

        fs::write(output_path.join("VoiRS.podspec"), podspec_content)?;

        Ok(())
    }

    /// Create CocoaPods specification
    fn create_podspec(&self) -> PodSpec {
        PodSpec {
            name: "VoiRS".to_string(),
            version: "0.1.0".to_string(),
            summary: "High-performance speech synthesis library".to_string(),
            description: "VoiRS is a comprehensive speech synthesis library providing high-quality text-to-speech, voice cloning, and emotion control capabilities.".to_string(),
            homepage: "https://github.com/cool-japan/voirs".to_string(),
            license: License {
                license_type: "MIT".to_string(),
                file: Some("LICENSE".to_string()),
            },
            author: Author {
                name: "VoiRS Team".to_string(),
                email: "team@voirs.dev".to_string(),
            },
            source: Source {
                git: "https://github.com/cool-japan/voirs.git".to_string(),
                tag: "0.1.0".to_string(),
            },
            platforms: PodPlatforms {
                ios: self.deployment_targets.ios.clone(),
                osx: self.deployment_targets.macos.clone(),
                watchos: self.deployment_targets.watchos.clone(),
                tvos: self.deployment_targets.tvos.clone(),
            },
            source_files: vec![
                "Sources/VoiRS/**/*.{h,m,swift}".to_string(),
                "include/**/*.h".to_string(),
            ],
            public_header_files: vec![
                "include/**/*.h".to_string(),
            ],
            vendored_libraries: vec![
                "lib/libvoirs_ffi.a".to_string(),
            ],
            vendored_frameworks: vec![],
            frameworks: vec![
                "Foundation".to_string(),
                "AVFoundation".to_string(),
                "AudioToolbox".to_string(),
            ],
            libraries: vec![
                "c++".to_string(),
            ],
            dependencies: std::collections::HashMap::new(),
        }
    }

    /// Generate podspec file content
    fn generate_podspec_content(
        &self,
        podspec: &PodSpec,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut content = String::new();

        content.push_str("Pod::Spec.new do |s|\n");
        content.push_str(&format!("  s.name             = '{}'\n", podspec.name));
        content.push_str(&format!("  s.version          = '{}'\n", podspec.version));
        content.push_str(&format!("  s.summary          = '{}'\n", podspec.summary));
        content.push_str(&format!(
            "  s.description      = '{}'\n",
            podspec.description
        ));
        content.push_str(&format!("  s.homepage         = '{}'\n", podspec.homepage));
        content.push_str(&format!(
            "  s.license          = {{ :type => '{}', :file => '{}' }}\n",
            podspec.license.license_type,
            podspec
                .license
                .file
                .as_ref()
                .unwrap_or(&"LICENSE".to_string())
        ));
        content.push_str(&format!(
            "  s.author           = {{ '{}' => '{}' }}\n",
            podspec.author.name, podspec.author.email
        ));
        content.push_str(&format!(
            "  s.source           = {{ :git => '{}', :tag => '{}' }}\n",
            podspec.source.git, podspec.source.tag
        ));

        // Platforms
        content.push_str(&format!(
            "  s.ios.deployment_target = '{}'\n",
            podspec.platforms.ios
        ));
        content.push_str(&format!(
            "  s.osx.deployment_target = '{}'\n",
            podspec.platforms.osx
        ));
        content.push_str(&format!(
            "  s.watchos.deployment_target = '{}'\n",
            podspec.platforms.watchos
        ));
        content.push_str(&format!(
            "  s.tvos.deployment_target = '{}'\n",
            podspec.platforms.tvos
        ));

        // Source files
        content.push_str("  s.source_files = ");
        if podspec.source_files.len() == 1 {
            content.push_str(&format!("'{}'\n", podspec.source_files[0]));
        } else {
            content.push_str("[");
            for (i, file) in podspec.source_files.iter().enumerate() {
                content.push_str(&format!("'{}'", file));
                if i < podspec.source_files.len() - 1 {
                    content.push_str(", ");
                }
            }
            content.push_str("]\n");
        }

        // Public headers
        if !podspec.public_header_files.is_empty() {
            content.push_str("  s.public_header_files = ");
            if podspec.public_header_files.len() == 1 {
                content.push_str(&format!("'{}'\n", podspec.public_header_files[0]));
            } else {
                content.push_str("[");
                for (i, file) in podspec.public_header_files.iter().enumerate() {
                    content.push_str(&format!("'{}'", file));
                    if i < podspec.public_header_files.len() - 1 {
                        content.push_str(", ");
                    }
                }
                content.push_str("]\n");
            }
        }

        // Vendored libraries
        if !podspec.vendored_libraries.is_empty() {
            content.push_str("  s.vendored_libraries = ");
            if podspec.vendored_libraries.len() == 1 {
                content.push_str(&format!("'{}'\n", podspec.vendored_libraries[0]));
            } else {
                content.push_str("[");
                for (i, lib) in podspec.vendored_libraries.iter().enumerate() {
                    content.push_str(&format!("'{}'", lib));
                    if i < podspec.vendored_libraries.len() - 1 {
                        content.push_str(", ");
                    }
                }
                content.push_str("]\n");
            }
        }

        // Frameworks
        if !podspec.frameworks.is_empty() {
            content.push_str("  s.frameworks = ");
            if podspec.frameworks.len() == 1 {
                content.push_str(&format!("'{}'\n", podspec.frameworks[0]));
            } else {
                content.push_str("[");
                for (i, framework) in podspec.frameworks.iter().enumerate() {
                    content.push_str(&format!("'{}'", framework));
                    if i < podspec.frameworks.len() - 1 {
                        content.push_str(", ");
                    }
                }
                content.push_str("]\n");
            }
        }

        // Libraries
        if !podspec.libraries.is_empty() {
            content.push_str("  s.libraries = ");
            if podspec.libraries.len() == 1 {
                content.push_str(&format!("'{}'\n", podspec.libraries[0]));
            } else {
                content.push_str("[");
                for (i, lib) in podspec.libraries.iter().enumerate() {
                    content.push_str(&format!("'{}'", lib));
                    if i < podspec.libraries.len() - 1 {
                        content.push_str(", ");
                    }
                }
                content.push_str("]\n");
            }
        }

        content.push_str("end\n");

        Ok(content)
    }

    /// Generate Xcode project template
    pub fn generate_xcode_project_template(
        &self,
        output_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let project_dir = output_path.join("VoiRS_iOS_App.xctemplate");
        fs::create_dir_all(&project_dir)?;

        // Generate template info
        self.generate_template_info(&project_dir)?;

        // Generate source files
        self.generate_ios_app_template(&project_dir)?;

        // Generate macOS app template
        let macos_project_dir = output_path.join("VoiRS_macOS_App.xctemplate");
        fs::create_dir_all(&macos_project_dir)?;
        self.generate_macos_app_template(&macos_project_dir)?;

        Ok(())
    }

    /// Generate template info plist
    fn generate_template_info(
        &self,
        template_dir: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let info_plist = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Kind</key>
    <string>Xcode.Xcode3.ProjectTemplateUnitKind</string>
    <key>Identifier</key>
    <string>com.voirs.iosapp</string>
    <key>Concrete</key>
    <true/>
    <key>Description</key>
    <string>This template provides a starting point for an iOS application using VoiRS speech synthesis.</string>
    <key>Name</key>
    <string>VoiRS iOS App</string>
    <key>SortOrder</key>
    <integer>1</integer>
    <key>Platforms</key>
    <array>
        <string>com.apple.platform.iphoneos</string>
    </array>
    <key>AllowedTypes</key>
    <array>
        <string>public.swift-source</string>
        <string>public.objective-c-source</string>
    </array>
    <key>MainTemplateFile</key>
    <string>main.swift</string>
    <key>Options</key>
    <array>
        <dict>
            <key>Identifier</key>
            <string>productName</string>
            <key>Name</key>
            <string>Product Name</string>
            <key>Description</key>
            <string>The name of the product</string>
            <key>Type</key>
            <string>text</string>
            <key>Default</key>
            <string>VoiRSApp</string>
            <key>Required</key>
            <true/>
        </dict>
    </array>
</dict>
</plist>"#;

        fs::write(template_dir.join("TemplateInfo.plist"), info_plist)?;
        Ok(())
    }

    /// Generate iOS app template
    fn generate_ios_app_template(
        &self,
        template_dir: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Generate ContentView.swift
        let content_view = r#"//
//  ContentView.swift
//  ___PACKAGENAME___
//
//  Created by Xcode Template on ___DATE___.
//

import SwiftUI
import VoiRS

struct ContentView: View {
    @State private var textToSynthesize = "Hello from VoiRS! This is a test of speech synthesis."
    @State private var isPlaying = false
    @State private var synthesisResult: String = ""
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                Text("VoiRS Speech Synthesis")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                    .padding()
                
                TextEditor(text: $textToSynthesize)
                    .frame(height: 100)
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
                
                Button(action: {
                    synthesizeSpeech()
                }) {
                    HStack {
                        Image(systemName: isPlaying ? "stop.circle" : "play.circle")
                        Text(isPlaying ? "Stop" : "Synthesize Speech")
                    }
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(8)
                }
                .disabled(isPlaying)
                
                if !synthesisResult.isEmpty {
                    Text(synthesisResult)
                        .padding()
                        .background(Color.green.opacity(0.1))
                        .cornerRadius(8)
                }
                
                Spacer()
            }
            .padding()
            .navigationTitle("VoiRS Demo")
        }
    }
    
    private func synthesizeSpeech() {
        isPlaying = true
        synthesisResult = "Synthesizing..."
        
        // Initialize VoiRS
        let initResult = voirs_ffi_init()
        guard initResult == VOIRS_OK else {
            synthesisResult = "Failed to initialize VoiRS"
            isPlaying = false
            return
        }
        
        // Synthesize text
        var audioBuffer: UnsafeMutablePointer<VoiRSAudioBuffer>?
        let synthesisError = voirs_ffi_synthesize_text(textToSynthesize, nil, &audioBuffer)
        
        if synthesisError == VOIRS_OK, let buffer = audioBuffer {
            synthesisResult = "Synthesis successful! \\(buffer.pointee.sample_count) samples at \\(buffer.pointee.sample_rate)Hz"
            
            // Play the audio (simplified - in a real app you'd use AVAudioEngine)
            playAudioBuffer(buffer)
            
            // Clean up
            voirs_ffi_destroy_audio_buffer(buffer)
        } else {
            synthesisResult = "Synthesis failed with error: \\(synthesisError)"
        }
        
        // Cleanup VoiRS
        voirs_ffi_cleanup()
        isPlaying = false
    }
    
    private func playAudioBuffer(_ buffer: UnsafeMutablePointer<VoiRSAudioBuffer>) {
        // In a real implementation, you would use AVAudioEngine to play the audio
        // This is a placeholder for audio playback functionality
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            synthesisResult += " (Audio playback completed)"
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
"#;

        fs::write(template_dir.join("ContentView.swift"), content_view)?;

        // Generate App.swift
        let app_swift = r#"//
//  ___PACKAGENAME___App.swift
//  ___PACKAGENAME___
//
//  Created by Xcode Template on ___DATE___.
//

import SwiftUI

@main
struct ___PACKAGENAME___App: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
"#;

        fs::write(template_dir.join("___PACKAGENAME___App.swift"), app_swift)?;

        // Generate bridging header
        let bridging_header = r#"//
//  ___PACKAGENAME___-Bridging-Header.h
//  ___PACKAGENAME___
//
//  Created by Xcode Template on ___DATE___.
//

#ifndef ___PACKAGENAME____Bridging_Header_h
#define ___PACKAGENAME____Bridging_Header_h

#import <voirs_ffi.h>

#endif /* ___PACKAGENAME____Bridging_Header_h */
"#;

        fs::write(
            template_dir.join("___PACKAGENAME___-Bridging-Header.h"),
            bridging_header,
        )?;

        Ok(())
    }

    /// Generate macOS app template
    fn generate_macos_app_template(
        &self,
        template_dir: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let macos_content_view = r#"//
//  ContentView.swift
//  ___PACKAGENAME___
//
//  Created by Xcode Template on ___DATE___.
//

import SwiftUI
import VoiRS

struct ContentView: View {
    @State private var textToSynthesize = "Hello from VoiRS on macOS! This is a test of speech synthesis."
    @State private var selectedVoice = "default"
    @State private var playbackSpeed: Double = 1.0
    @State private var pitch: Double = 0.0
    @State private var isProcessing = false
    @State private var statusMessage = ""
    
    let voices = ["default", "voice1", "voice2", "voice3"]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("VoiRS Speech Synthesis for macOS")
                .font(.largeTitle)
                .fontWeight(.bold)
                .padding(.bottom)
            
            Group {
                Text("Text to Synthesize:")
                    .font(.headline)
                
                TextEditor(text: $textToSynthesize)
                    .frame(height: 120)
                    .padding(4)
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(6)
            }
            
            Group {
                Text("Voice Settings:")
                    .font(.headline)
                
                HStack {
                    Text("Voice:")
                    Picker("Voice", selection: $selectedVoice) {
                        ForEach(voices, id: \\.self) { voice in
                            Text(voice).tag(voice)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                }
                
                HStack {
                    Text("Speed:")
                    Slider(value: $playbackSpeed, in: 0.5...2.0, step: 0.1)
                    Text("\\(playbackSpeed, specifier: "%.1f")x")
                        .frame(width: 40)
                }
                
                HStack {
                    Text("Pitch:")
                    Slider(value: $pitch, in: -12.0...12.0, step: 1.0)
                    Text("\\(pitch, specifier: "%.0f") st")
                        .frame(width: 40)
                }
            }
            
            HStack {
                Button("Synthesize & Play") {
                    synthesizeAndPlay()
                }
                .disabled(isProcessing)
                
                Button("Save to File") {
                    saveToFile()
                }
                .disabled(isProcessing)
                
                if isProcessing {
                    ProgressView()
                        .scaleEffect(0.8)
                }
            }
            
            if !statusMessage.isEmpty {
                Text(statusMessage)
                    .padding()
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(6)
            }
            
            Spacer()
        }
        .padding()
        .frame(minWidth: 500, minHeight: 400)
    }
    
    private func synthesizeAndPlay() {
        isProcessing = true
        statusMessage = "Initializing VoiRS..."
        
        DispatchQueue.global(qos: .userInitiated).async {
            let result = performSynthesis()
            
            DispatchQueue.main.async {
                statusMessage = result
                isProcessing = false
            }
        }
    }
    
    private func saveToFile() {
        isProcessing = true
        statusMessage = "Saving to file..."
        
        // Implementation for saving to file
        DispatchQueue.global(qos: .userInitiated).async {
            let result = performSynthesisAndSave()
            
            DispatchQueue.main.async {
                statusMessage = result
                isProcessing = false
            }
        }
    }
    
    private func performSynthesis() -> String {
        let initResult = voirs_ffi_init()
        guard initResult == VOIRS_OK else {
            return "Failed to initialize VoiRS"
        }
        
        var audioBuffer: UnsafeMutablePointer<VoiRSAudioBuffer>?
        let voiceId = selectedVoice == "default" ? nil : selectedVoice
        let synthesisError = voirs_ffi_synthesize_text(textToSynthesize, voiceId, &audioBuffer)
        
        defer {
            if let buffer = audioBuffer {
                voirs_ffi_destroy_audio_buffer(buffer)
            }
            voirs_ffi_cleanup()
        }
        
        if synthesisError == VOIRS_OK, let buffer = audioBuffer {
            return "Synthesis successful! Generated \\(buffer.pointee.sample_count) samples at \\(buffer.pointee.sample_rate)Hz with \\(buffer.pointee.channels) channels."
        } else {
            return "Synthesis failed with error code: \\(synthesisError)"
        }
    }
    
    private func performSynthesisAndSave() -> String {
        let initResult = voirs_ffi_init()
        guard initResult == VOIRS_OK else {
            return "Failed to initialize VoiRS"
        }
        
        var audioBuffer: UnsafeMutablePointer<VoiRSAudioBuffer>?
        let voiceId = selectedVoice == "default" ? nil : selectedVoice
        let synthesisError = voirs_ffi_synthesize_text(textToSynthesize, voiceId, &audioBuffer)
        
        defer {
            if let buffer = audioBuffer {
                voirs_ffi_destroy_audio_buffer(buffer)
            }
            voirs_ffi_cleanup()
        }
        
        if synthesisError == VOIRS_OK, let buffer = audioBuffer {
            let saveError = voirs_ffi_save_audio_to_file(buffer, "output.wav")
            if saveError == VOIRS_OK {
                return "Audio saved to output.wav successfully!"
            } else {
                return "Failed to save audio file with error code: \\(saveError)"
            }
        } else {
            return "Synthesis failed with error code: \\(synthesisError)"
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
"#;

        fs::write(template_dir.join("ContentView.swift"), macos_content_view)?;

        Ok(())
    }

    /// Build universal framework for distribution
    pub fn build_universal_framework(
        &self,
        output_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("Building universal framework for VoiRS...");

        let framework_dir = output_path.join("VoiRS.framework");
        fs::create_dir_all(&framework_dir)?;

        // Create framework structure
        let versions_dir = framework_dir.join("Versions/A");
        fs::create_dir_all(&versions_dir)?;

        let headers_dir = versions_dir.join("Headers");
        fs::create_dir_all(&headers_dir)?;

        let resources_dir = versions_dir.join("Resources");
        fs::create_dir_all(&resources_dir)?;

        // Create symbolic links (Unix-style framework structure)
        #[cfg(unix)]
        {
            use std::os::unix::fs;
            let _ = fs::symlink("Versions/Current/VoiRS", framework_dir.join("VoiRS"));
            let _ = fs::symlink("Versions/Current/Headers", framework_dir.join("Headers"));
            let _ = fs::symlink(
                "Versions/Current/Resources",
                framework_dir.join("Resources"),
            );
            let _ = fs::symlink("A", framework_dir.join("Versions/Current"));
        }

        // Generate Info.plist
        self.generate_framework_info_plist(&resources_dir)?;

        // Copy headers
        self.copy_framework_headers(&headers_dir)?;

        println!(
            "Universal framework created at: {}",
            framework_dir.display()
        );

        Ok(())
    }

    /// Generate framework Info.plist
    fn generate_framework_info_plist(
        &self,
        resources_dir: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let info_plist = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>VoiRS</string>
    <key>CFBundleIdentifier</key>
    <string>com.voirs.VoiRS</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>VoiRS</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>NSPrincipalClass</key>
    <string></string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>iPhoneOS</string>
        <string>iPhoneSimulator</string>
        <string>MacOSX</string>
    </array>
    <key>MinimumOSVersion</key>
    <string>13.0</string>
</dict>
</plist>"#;

        fs::write(resources_dir.join("Info.plist"), info_plist)?;
        Ok(())
    }

    /// Copy framework headers
    fn copy_framework_headers(&self, headers_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        // Generate umbrella header
        let umbrella_header = r#"//
//  VoiRS.h
//  VoiRS Framework
//
//  Universal header for VoiRS speech synthesis framework
//

#import <Foundation/Foundation.h>

//! Project version number for VoiRS.
FOUNDATION_EXPORT double VoiRSVersionNumber;

//! Project version string for VoiRS.
FOUNDATION_EXPORT const unsigned char VoiRSVersionString[];

// Import all public headers
#import <VoiRS/voirs_ffi.h>

// Forward declarations for Swift interoperability
#ifndef VOIRS_SWIFT_INTEROP
#define VOIRS_SWIFT_INTEROP

#ifdef __cplusplus
extern "C" {
#endif

// Swift-friendly error handling
typedef NS_ENUM(NSInteger, VoiRSErrorCode) {
    VoiRSErrorCodeSuccess = 0,
    VoiRSErrorCodeInitializationFailed = 1,
    VoiRSErrorCodeInvalidInput = 2,
    VoiRSErrorCodeSynthesisFailed = 3,
    VoiRSErrorCodeAudioError = 4,
    VoiRSErrorCodeMemoryError = 5,
    VoiRSErrorCodeUnknown = 999
};

// Swift-friendly synthesis options
typedef struct {
    const char* voice_id;
    uint32_t sample_rate;
    float speed;
    float pitch;
    uint16_t channels;
} VoiRSSynthesisOptions;

// Simplified Swift API functions
VoiRSErrorCode VoiRS_Initialize(void);
VoiRSErrorCode VoiRS_Synthesize(const char* text, VoiRSSynthesisOptions* options, VoiRSAudioBuffer** result);
VoiRSErrorCode VoiRS_SaveToFile(VoiRSAudioBuffer* buffer, const char* filename);
void VoiRS_DestroyBuffer(VoiRSAudioBuffer* buffer);
void VoiRS_Cleanup(void);

#ifdef __cplusplus
}
#endif

#endif // VOIRS_SWIFT_INTEROP
"#;

        fs::write(headers_dir.join("VoiRS.h"), umbrella_header)?;

        // Generate module map
        let module_map = r#"framework module VoiRS {
    umbrella header "VoiRS.h"
    
    header "voirs_ffi.h"
    
    export *
    module * { export * }
    
    explicit module Private {
        header "voirs_ffi_private.h"
        export *
    }
}
"#;

        fs::write(headers_dir.join("module.modulemap"), module_map)?;

        Ok(())
    }

    /// Install Xcode integration
    pub fn install_integration(&self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(dev_dir) = &self.developer_dir {
            println!("Installing VoiRS Xcode integration...");

            let templates_dir = dev_dir.join("Platforms/iPhoneOS.platform/Developer/Library/Xcode/Templates/Project Templates/VoiRS");
            fs::create_dir_all(&templates_dir)?;

            // Generate integration files
            self.generate_swift_package(&templates_dir)?;
            self.generate_cocoapods_spec(&templates_dir)?;
            self.generate_xcode_project_template(&templates_dir)?;
            self.build_universal_framework(&templates_dir)?;

            println!("VoiRS Xcode integration installed successfully!");
            println!("Integration files created in: {}", templates_dir.display());
        } else {
            return Err("Xcode developer directory not found".into());
        }

        Ok(())
    }

    /// Verify installation
    pub fn verify_installation(&self) -> Result<bool, Box<dyn std::error::Error>> {
        if let Some(dev_dir) = &self.developer_dir {
            let templates_dir = dev_dir.join("Platforms/iPhoneOS.platform/Developer/Library/Xcode/Templates/Project Templates/VoiRS");

            let required_files = [
                "Package.swift",
                "VoiRS.podspec",
                "VoiRS_iOS_App.xctemplate/TemplateInfo.plist",
                "VoiRS.framework/Info.plist",
            ];

            for file in &required_files {
                if !templates_dir.join(file).exists() {
                    return Ok(false);
                }
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get integration information
    pub fn get_integration_info(&self) -> XcodeIntegrationInfo {
        XcodeIntegrationInfo {
            xcode_version: self.xcode_version,
            developer_dir: self.developer_dir.clone(),
            sdk_paths: self.sdk_paths.clone(),
            deployment_targets: self.deployment_targets.clone(),
            is_installed: self.verify_installation().unwrap_or(false),
        }
    }
}

/// Xcode integration information
#[derive(Debug, Clone)]
pub struct XcodeIntegrationInfo {
    pub xcode_version: XcodeVersion,
    pub developer_dir: Option<PathBuf>,
    pub sdk_paths: SdkPaths,
    pub deployment_targets: DeploymentTargets,
    pub is_installed: bool,
}

impl Default for XcodeIntegration {
    fn default() -> Self {
        Self::new()
    }
}

/// C API functions for Xcode integration
#[no_mangle]
pub extern "C" fn voirs_xcode_create_integration() -> *mut XcodeIntegration {
    Box::into_raw(Box::new(XcodeIntegration::new()))
}

#[no_mangle]
pub unsafe extern "C" fn voirs_xcode_destroy_integration(integration: *mut XcodeIntegration) {
    if !integration.is_null() {
        unsafe {
            let _ = Box::from_raw(integration);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn voirs_xcode_install_integration(
    integration: *mut XcodeIntegration,
) -> bool {
    if integration.is_null() {
        return false;
    }

    unsafe { (*integration).install_integration().is_ok() }
}

#[no_mangle]
pub unsafe extern "C" fn voirs_xcode_verify_installation(
    integration: *mut XcodeIntegration,
) -> bool {
    if integration.is_null() {
        return false;
    }

    unsafe { (*integration).verify_installation().unwrap_or(false) }
}

#[no_mangle]
pub unsafe extern "C" fn voirs_xcode_build_framework(
    integration: *mut XcodeIntegration,
    output_path: *const std::os::raw::c_char,
) -> bool {
    if integration.is_null() || output_path.is_null() {
        return false;
    }

    unsafe {
        let output_path_str = match std::ffi::CStr::from_ptr(output_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        };

        let path = Path::new(output_path_str);
        (*integration).build_universal_framework(path).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_xcode_integration_creation() {
        let integration = XcodeIntegration::new();
        assert!(matches!(
            integration.xcode_version,
            XcodeVersion::Xcode12
                | XcodeVersion::Xcode13
                | XcodeVersion::Xcode14
                | XcodeVersion::Xcode15
                | XcodeVersion::Xcode16
        ));
    }

    #[test]
    fn test_swift_package_generation() {
        let integration = XcodeIntegration::new();
        let temp_dir = TempDir::new().unwrap();

        let result = integration.generate_swift_package(temp_dir.path());
        assert!(result.is_ok());

        let package_swift = temp_dir.path().join("Package.swift");
        assert!(package_swift.exists());

        let content = fs::read_to_string(package_swift).unwrap();
        assert!(content.contains("swift-tools-version"));
        assert!(content.contains("VoiRS"));
    }

    #[test]
    fn test_cocoapods_spec_generation() {
        let integration = XcodeIntegration::new();
        let temp_dir = TempDir::new().unwrap();

        let result = integration.generate_cocoapods_spec(temp_dir.path());
        assert!(result.is_ok());

        let podspec = temp_dir.path().join("VoiRS.podspec");
        assert!(podspec.exists());

        let content = fs::read_to_string(podspec).unwrap();
        assert!(content.contains("Pod::Spec.new"));
        assert!(content.contains("VoiRS"));
    }

    #[test]
    fn test_framework_creation() {
        let integration = XcodeIntegration::new();
        let temp_dir = TempDir::new().unwrap();

        let result = integration.build_universal_framework(temp_dir.path());
        assert!(result.is_ok());

        let framework_dir = temp_dir.path().join("VoiRS.framework");
        assert!(framework_dir.exists());

        let info_plist = framework_dir.join("Versions/A/Resources/Info.plist");
        assert!(info_plist.exists());
    }

    #[test]
    fn test_project_template_generation() {
        let integration = XcodeIntegration::new();
        let temp_dir = TempDir::new().unwrap();

        let result = integration.generate_xcode_project_template(temp_dir.path());
        assert!(result.is_ok());

        let ios_template = temp_dir.path().join("VoiRS_iOS_App.xctemplate");
        let macos_template = temp_dir.path().join("VoiRS_macOS_App.xctemplate");

        assert!(ios_template.exists());
        assert!(macos_template.exists());

        assert!(ios_template.join("TemplateInfo.plist").exists());
        assert!(ios_template.join("ContentView.swift").exists());
    }

    #[test]
    fn test_swift_package_content() {
        let integration = XcodeIntegration::new();
        let package = integration.create_swift_package();
        let swift_content = integration.generate_package_swift(&package).unwrap();

        assert!(swift_content.contains("swift-tools-version"));
        assert!(swift_content.contains("VoiRS"));
        assert!(swift_content.contains(".library"));
        assert!(swift_content.contains(".target"));
        assert!(swift_content.contains("macOS"));
        assert!(swift_content.contains("iOS"));
    }

    #[test]
    fn test_podspec_content() {
        let integration = XcodeIntegration::new();
        let podspec = integration.create_podspec();
        let content = integration.generate_podspec_content(&podspec).unwrap();

        assert!(content.contains("Pod::Spec.new"));
        assert!(content.contains("s.name"));
        assert!(content.contains("s.version"));
        assert!(content.contains("VoiRS"));
        assert!(content.contains("ios.deployment_target"));
        assert!(content.contains("osx.deployment_target"));
    }
}
