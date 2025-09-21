//! Visual Studio Integration
//!
//! This module provides integration features for Microsoft Visual Studio,
//! including MSBuild target files, IntelliSense configuration, debug
//! visualization, and project templates for seamless development experience.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Visual Studio integration manager
pub struct VisualStudioIntegration {
    pub vs_version: VsVersion,
    pub installation_path: Option<PathBuf>,
    pub sdk_path: Option<PathBuf>,
}

/// Supported Visual Studio versions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VsVersion {
    Vs2019,
    Vs2022,
    VsBuildTools,
}

/// MSBuild target configuration for VoiRS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MSBuildTarget {
    pub target_name: String,
    pub dll_path: String,
    pub lib_path: String,
    pub include_path: String,
    pub preprocessor_definitions: Vec<String>,
    pub link_libraries: Vec<String>,
}

/// IntelliSense configuration for C/C++ development
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntelliSenseConfig {
    pub configurations: Vec<IntelliSenseConfiguration>,
    pub compiler_path: String,
    pub c_standard: String,
    pub cpp_standard: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntelliSenseConfiguration {
    pub name: String,
    pub include_path: Vec<String>,
    pub defines: Vec<String>,
    pub compiler_args: Vec<String>,
    pub browse: IntelliSenseBrowse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntelliSenseBrowse {
    pub path: Vec<String>,
    pub limit_symbols_to_included_headers: bool,
    pub database_filename: String,
}

/// Debug visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugVisualization {
    pub type_visualizers: Vec<TypeVisualizer>,
    pub custom_viewers: Vec<CustomViewer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeVisualizer {
    pub type_name: String,
    pub display_string: String,
    pub debugger_display: String,
    pub expand: Option<ExpandRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpandRule {
    pub item: Vec<ExpandItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpandItem {
    pub name: String,
    pub value: String,
    pub condition: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomViewer {
    pub viewer_name: String,
    pub dll_path: String,
    pub class_name: String,
    pub supported_types: Vec<String>,
}

/// Project template configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectTemplate {
    pub template_name: String,
    pub template_id: String,
    pub description: String,
    pub language: String,
    pub project_type: String,
    pub files: Vec<TemplateFile>,
    pub parameters: Vec<TemplateParameter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateFile {
    pub source_path: String,
    pub target_path: String,
    pub replacements: Vec<TokenReplacement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenReplacement {
    pub token: String,
    pub replacement: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateParameter {
    pub name: String,
    pub data_type: String,
    pub default_value: String,
    pub replace_parameters: bool,
}

impl VisualStudioIntegration {
    /// Create a new Visual Studio integration manager
    pub fn new() -> Self {
        let vs_version = Self::detect_vs_version();
        let installation_path = Self::detect_installation_path(vs_version);
        let sdk_path = Self::detect_sdk_path();

        Self {
            vs_version,
            installation_path,
            sdk_path,
        }
    }

    /// Detect installed Visual Studio version
    fn detect_vs_version() -> VsVersion {
        #[cfg(windows)]
        {
            use std::process::Command;

            // Try to detect VS 2022 first
            if let Ok(output) = Command::new("vswhere")
                .args(&["-version", "[17.0,18.0)", "-property", "installationPath"])
                .output()
            {
                if !output.stdout.is_empty() {
                    return VsVersion::Vs2022;
                }
            }

            // Try to detect VS 2019
            if let Ok(output) = Command::new("vswhere")
                .args(&["-version", "[16.0,17.0)", "-property", "installationPath"])
                .output()
            {
                if !output.stdout.is_empty() {
                    return VsVersion::Vs2019;
                }
            }

            // Fallback to build tools
            VsVersion::VsBuildTools
        }

        #[cfg(not(windows))]
        {
            // Default to VS 2022 for cross-platform scenarios
            VsVersion::Vs2022
        }
    }

    /// Detect Visual Studio installation path
    fn detect_installation_path(vs_version: VsVersion) -> Option<PathBuf> {
        #[cfg(windows)]
        {
            use std::process::Command;

            let version_range = match vs_version {
                VsVersion::Vs2022 => "[17.0,18.0)",
                VsVersion::Vs2019 => "[16.0,17.0)",
                VsVersion::VsBuildTools => "[15.0,)",
            };

            if let Ok(output) = Command::new("vswhere")
                .args(&["-version", version_range, "-property", "installationPath"])
                .output()
            {
                let path_str = String::from_utf8_lossy(&output.stdout).trim();
                if !path_str.is_empty() {
                    return Some(PathBuf::from(path_str));
                }
            }
        }

        None
    }

    /// Detect Windows SDK path
    fn detect_sdk_path() -> Option<PathBuf> {
        #[cfg(windows)]
        {
            use winreg::enums::*;
            use winreg::RegKey;

            let hklm = RegKey::predef(HKEY_LOCAL_MACHINE);
            if let Ok(windows_kits) =
                hklm.open_subkey("SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots")
            {
                if let Ok(sdk_path) = windows_kits.get_value::<String, _>("KitsRoot10") {
                    return Some(PathBuf::from(sdk_path));
                }
            }
        }

        None
    }

    /// Generate MSBuild target file
    pub fn generate_msbuild_target(
        &self,
        output_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let target = self.create_msbuild_target();
        let target_xml = self.generate_target_xml(&target)?;

        let target_file_path = output_path.join("VoiRS.targets");
        fs::write(&target_file_path, target_xml)?;

        // Also generate props file
        let props_xml = self.generate_props_xml(&target)?;
        let props_file_path = output_path.join("VoiRS.props");
        fs::write(&props_file_path, props_xml)?;

        Ok(())
    }

    /// Create MSBuild target configuration
    fn create_msbuild_target(&self) -> MSBuildTarget {
        MSBuildTarget {
            target_name: "VoiRS".to_string(),
            dll_path: "$(MSBuildThisFileDirectory)lib\\voirs_ffi.dll".to_string(),
            lib_path: "$(MSBuildThisFileDirectory)lib\\voirs_ffi.lib".to_string(),
            include_path: "$(MSBuildThisFileDirectory)include".to_string(),
            preprocessor_definitions: vec![
                "VOIRS_FFI_AVAILABLE".to_string(),
                "VOIRS_VERSION_MAJOR=0".to_string(),
                "VOIRS_VERSION_MINOR=1".to_string(),
            ],
            link_libraries: vec![
                "voirs_ffi.lib".to_string(),
                "user32.lib".to_string(),
                "kernel32.lib".to_string(),
                "ole32.lib".to_string(),
                "oleaut32.lib".to_string(),
            ],
        }
    }

    /// Generate MSBuild target XML
    fn generate_target_xml(
        &self,
        target: &MSBuildTarget,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut xml = String::new();
        xml.push_str(
            r#"<?xml version="1.0" encoding="utf-8"?>
<Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  
  <!-- VoiRS FFI Integration Target -->
  <Target Name="VoiRSBuildTarget" BeforeTargets="ClCompile">
    <Message Text="Configuring VoiRS FFI integration..." Importance="high" />
  </Target>

  <!-- Include directories -->
  <ItemDefinitionGroup>
    <ClCompile>
      <AdditionalIncludeDirectories>"#,
        );
        xml.push_str(&target.include_path);
        xml.push_str(
            r#";%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
      <PreprocessorDefinitions>"#,
        );

        for (i, def) in target.preprocessor_definitions.iter().enumerate() {
            if i > 0 {
                xml.push(';');
            }
            xml.push_str(def);
        }

        xml.push_str(
            r#";%(PreprocessorDefinitions)</PreprocessorDefinitions>
    </ClCompile>
    
    <Link>
      <AdditionalDependencies>"#,
        );

        for (i, lib) in target.link_libraries.iter().enumerate() {
            if i > 0 {
                xml.push(';');
            }
            xml.push_str(lib);
        }

        xml.push_str(r#";%(AdditionalDependencies)</AdditionalDependencies>
      <AdditionalLibraryDirectories>$(MSBuildThisFileDirectory)lib;%(AdditionalLibraryDirectories)</AdditionalLibraryDirectories>
    </Link>
  </ItemDefinitionGroup>

  <!-- Copy DLL to output directory -->
  <Target Name="CopyVoiRSDLL" AfterTargets="Build">
    <Copy SourceFiles=""#);
        xml.push_str(&target.dll_path);
        xml.push_str(
            r#""
          DestinationFolder="$(TargetDir)"
          ContinueOnError="false" />
  </Target>

  <!-- NuGet package support -->
  <ItemGroup>
    <None Include="$(MSBuildThisFileDirectory)lib\voirs_ffi.dll">
      <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </None>
    <None Include="$(MSBuildThisFileDirectory)include\*.h">
      <CopyToOutputDirectory>Never</CopyToOutputDirectory>
    </None>
  </ItemGroup>

</Project>"#,
        );

        Ok(xml)
    }

    /// Generate MSBuild props file
    fn generate_props_xml(
        &self,
        target: &MSBuildTarget,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let xml = format!(
            r#"<?xml version="1.0" encoding="utf-8"?>
<Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  
  <PropertyGroup>
    <VoiRSRoot>$(MSBuildThisFileDirectory)</VoiRSRoot>
    <VoiRSIncludePath>$(VoiRSRoot)include</VoiRSIncludePath>
    <VoiRSLibPath>$(VoiRSRoot)lib</VoiRSLibPath>
    <VoiRSVersion>0.1.0</VoiRSVersion>
  </PropertyGroup>

  <!-- Platform-specific configurations -->
  <PropertyGroup Condition="'$(Platform)' == 'x64'">
    <VoiRSPlatform>x64</VoiRSPlatform>
  </PropertyGroup>
  
  <PropertyGroup Condition="'$(Platform)' == 'Win32'">
    <VoiRSPlatform>x86</VoiRSPlatform>
  </PropertyGroup>
  
  <PropertyGroup Condition="'$(Platform)' == 'ARM64'">
    <VoiRSPlatform>ARM64</VoiRSPlatform>
  </PropertyGroup>

</Project>"#
        );

        Ok(xml)
    }

    /// Generate IntelliSense configuration
    pub fn generate_intellisense_config(
        &self,
        output_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = self.create_intellisense_config();
        let config_json = serde_json::to_string_pretty(&config)?;

        let config_path = output_path.join("c_cpp_properties.json");
        fs::write(&config_path, config_json)?;

        Ok(())
    }

    /// Create IntelliSense configuration
    fn create_intellisense_config(&self) -> IntelliSenseConfig {
        let configurations = vec![
            IntelliSenseConfiguration {
                name: "Win32".to_string(),
                include_path: vec![
                    "${workspaceFolder}/**".to_string(),
                    "${workspaceFolder}/include".to_string(),
                    "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/*/include".to_string(),
                    "C:/Program Files (x86)/Windows Kits/10/Include/*/ucrt".to_string(),
                    "C:/Program Files (x86)/Windows Kits/10/Include/*/um".to_string(),
                    "C:/Program Files (x86)/Windows Kits/10/Include/*/shared".to_string(),
                ],
                defines: vec![
                    "_DEBUG".to_string(),
                    "UNICODE".to_string(),
                    "_UNICODE".to_string(),
                    "VOIRS_FFI_AVAILABLE".to_string(),
                    "WIN32".to_string(),
                    "_WIN32".to_string(),
                    "_WIN64".to_string(),
                ],
                compiler_args: vec![
                    "/std:c17".to_string(),
                    "/EHsc".to_string(),
                ],
                browse: IntelliSenseBrowse {
                    path: vec![
                        "${workspaceFolder}".to_string(),
                        "${workspaceFolder}/include".to_string(),
                    ],
                    limit_symbols_to_included_headers: true,
                    database_filename: "${workspaceFolder}/.vscode/browse.vc.db".to_string(),
                },
            },
        ];

        IntelliSenseConfig {
            configurations,
            compiler_path: "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.*/bin/Hostx64/x64/cl.exe".to_string(),
            c_standard: "c17".to_string(),
            cpp_standard: "c++17".to_string(),
        }
    }

    /// Generate debug visualization files
    pub fn generate_debug_visualization(
        &self,
        output_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let natvis_content = self.generate_natvis_file()?;
        let natvis_path = output_path.join("VoiRS.natvis");
        fs::write(&natvis_path, natvis_content)?;

        // Generate autoexp.dat entries
        let autoexp_content = self.generate_autoexp_entries()?;
        let autoexp_path = output_path.join("VoiRS_autoexp.dat");
        fs::write(&autoexp_path, autoexp_content)?;

        Ok(())
    }

    /// Generate Visual Studio .natvis file for debug visualization
    fn generate_natvis_file(&self) -> Result<String, Box<dyn std::error::Error>> {
        let natvis = r#"<?xml version="1.0" encoding="utf-8"?>
<AutoVisualizer xmlns="http://schemas.microsoft.com/vstudio/debugger/natvis/2010">

  <!-- VoiRS FFI Audio Buffer -->
  <Type Name="VoiRSAudioBuffer">
    <DisplayString>VoiRS Audio Buffer ({sample_count} samples, {sample_rate}Hz, {channels} channels)</DisplayString>
    <Expand>
      <Item Name="Sample Count">sample_count</Item>
      <Item Name="Sample Rate">sample_rate</Item>
      <Item Name="Channels">channels</Item>
      <Item Name="Duration (ms)">(sample_count * 1000) / sample_rate</Item>
      <ArrayItems>
        <Size>sample_count</Size>
        <ValuePointer>samples</ValuePointer>
      </ArrayItems>
    </Expand>
  </Type>

  <!-- VoiRS FFI Synthesis Request -->
  <Type Name="VoiRSSynthesisRequest">
    <DisplayString>VoiRS Synthesis Request: "{text,s}"</DisplayString>
    <Expand>
      <Item Name="Text">text,s</Item>
      <Item Name="Voice ID">voice_id,s</Item>
      <Item Name="Sample Rate">sample_rate</Item>
      <Item Name="Speed">speed</Item>
      <Item Name="Pitch">pitch</Item>
    </Expand>
  </Type>

  <!-- VoiRS FFI Stats -->
  <Type Name="VoiRSStats">
    <DisplayString>VoiRS Stats (Calls: {total_calls}, Failures: {failed_calls})</DisplayString>
    <Expand>
      <Item Name="Total Calls">total_calls</Item>
      <Item Name="Failed Calls">failed_calls</Item>
      <Item Name="Success Rate">((total_calls - failed_calls) * 100.0) / total_calls</Item>
      <Item Name="Average Call Time (μs)">avg_call_time_ns / 1000</Item>
      <Item Name="Batch Operations">batch_operations</Item>
    </Expand>
  </Type>

  <!-- VoiRS Error -->
  <Type Name="VoiRSError">
    <DisplayString>VoiRS Error: {error_message,s} (Code: {error_code})</DisplayString>
    <Expand>
      <Item Name="Error Code">error_code</Item>
      <Item Name="Error Message">error_message,s</Item>
      <Item Name="Context">context,s</Item>
      <Item Name="Timestamp">timestamp</Item>
    </Expand>
  </Type>

  <!-- VoiRS Memory Pool -->
  <Type Name="VoiRSMemoryPool">
    <DisplayString>VoiRS Memory Pool (Chunk Size: {chunk_size}, Chunks: {chunk_count})</DisplayString>
    <Expand>
      <Item Name="Chunk Size">chunk_size</Item>
      <Item Name="Chunk Count">chunk_count</Item>
      <Item Name="Total Memory">chunk_size * chunk_count</Item>
      <Item Name="Available Chunks">available_chunks</Item>
      <Item Name="Allocated Chunks">chunk_count - available_chunks</Item>
    </Expand>
  </Type>

</AutoVisualizer>"#;

        Ok(natvis.to_string())
    }

    /// Generate autoexp.dat entries for older Visual Studio versions
    fn generate_autoexp_entries(&self) -> Result<String, Box<dyn std::error::Error>> {
        let autoexp = r#"; VoiRS FFI Debug Visualization
; Add these entries to your autoexp.dat file for older Visual Studio versions

[Visualizer]
VoiRSAudioBuffer {
    preview = #("VoiRS Audio Buffer (", $e.sample_count, " samples, ", $e.sample_rate, "Hz, ", $e.channels, " channels)")
    children = #(
        [raw members]: [$e,!],
        #array(expr: $e.samples[$i], size: $e.sample_count)
    )
}

VoiRSSynthesisRequest {
    preview = #("VoiRS Synthesis: ", $e.text)
    children = #(
        [raw members]: [$e,!],
        text: [$e.text,s],
        voice_id: [$e.voice_id,s],
        sample_rate: $e.sample_rate,
        speed: $e.speed,
        pitch: $e.pitch
    )
}

VoiRSStats {
    preview = #("VoiRS Stats (", $e.total_calls, " calls, ", $e.failed_calls, " failures)")
    children = #(
        [raw members]: [$e,!],
        total_calls: $e.total_calls,
        failed_calls: $e.failed_calls,
        success_rate: (($e.total_calls - $e.failed_calls) * 100.0) / $e.total_calls,
        avg_call_time_us: $e.avg_call_time_ns / 1000,
        batch_operations: $e.batch_operations
    )
}

VoiRSError {
    preview = #("VoiRS Error: ", $e.error_message, " (", $e.error_code, ")")
    children = #(
        [raw members]: [$e,!],
        error_code: $e.error_code,
        error_message: [$e.error_message,s],
        context: [$e.context,s],
        timestamp: $e.timestamp
    )
}
"#;

        Ok(autoexp.to_string())
    }

    /// Generate project templates
    pub fn generate_project_templates(
        &self,
        output_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let templates_dir = output_path.join("ProjectTemplates");
        fs::create_dir_all(&templates_dir)?;

        // Generate C++ console application template
        self.generate_cpp_console_template(&templates_dir)?;

        // Generate C application template
        self.generate_c_application_template(&templates_dir)?;

        // Generate DLL template
        self.generate_dll_template(&templates_dir)?;

        Ok(())
    }

    /// Generate C++ console application template
    fn generate_cpp_console_template(
        &self,
        templates_dir: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let template_dir = templates_dir.join("VoiRS_CPP_Console");
        fs::create_dir_all(&template_dir)?;

        // Generate template metadata
        let vstemplate = r#"<VSTemplate Version="3.0.0" xmlns="http://schemas.microsoft.com/developer/vstemplate/2005" Type="Project">
  <TemplateData>
    <Name>VoiRS C++ Console Application</Name>
    <Description>A C++ console application using VoiRS speech synthesis</Description>
    <ProjectType>VC</ProjectType>
    <SortOrder>1000</SortOrder>
    <CreateNewFolder>true</CreateNewFolder>
    <DefaultName>VoiRSApp</DefaultName>
    <ProvideDefaultName>true</ProvideDefaultName>
    <LocationField>Enabled</LocationField>
    <EnableLocationBrowseButton>true</EnableLocationBrowseButton>
    <Icon>__TemplateIcon.ico</Icon>
  </TemplateData>
  <TemplateContent>
    <Project TargetFileName="$projectname$.vcxproj" File="Template.vcxproj" ReplaceParameters="true">
      <ProjectItem ReplaceParameters="true" TargetFileName="$projectname$.cpp">main.cpp</ProjectItem>
      <ProjectItem ReplaceParameters="false" TargetFileName="packages.config">packages.config</ProjectItem>
    </Project>
  </TemplateContent>
</VSTemplate>"#;

        fs::write(template_dir.join("Template.vstemplate"), vstemplate)?;

        // Generate main.cpp
        let main_cpp = r#"// VoiRS C++ Console Application
// Generated from VoiRS Visual Studio Template

#include <iostream>
#include <string>
#include <voirs_ffi.h>

int main()
{
    std::cout << "VoiRS Speech Synthesis Demo\n";
    std::cout << "===========================\n\n";

    // Initialize VoiRS
    if (voirs_ffi_init() != VOIRS_OK) {
        std::cerr << "Failed to initialize VoiRS\n";
        return 1;
    }

    // Create synthesis request
    const char* text = "Hello from VoiRS! This is a test of the speech synthesis system.";
    std::cout << "Synthesizing: " << text << "\n";

    VoiRSAudioBuffer* audio_buffer = nullptr;
    VoiRSError error = voirs_ffi_synthesize_text(text, nullptr, &audio_buffer);

    if (error == VOIRS_OK && audio_buffer != nullptr) {
        std::cout << "Synthesis successful!\n";
        std::cout << "Audio length: " << audio_buffer->sample_count << " samples\n";
        std::cout << "Sample rate: " << audio_buffer->sample_rate << " Hz\n";
        std::cout << "Channels: " << audio_buffer->channels << "\n";

        // Save to file (example)
        const char* output_file = "output.wav";
        if (voirs_ffi_save_audio_to_file(audio_buffer, output_file) == VOIRS_OK) {
            std::cout << "Audio saved to " << output_file << "\n";
        }

        // Clean up
        voirs_ffi_destroy_audio_buffer(audio_buffer);
    } else {
        std::cerr << "Synthesis failed with error code: " << error << "\n";
    }

    // Cleanup VoiRS
    voirs_ffi_cleanup();

    std::cout << "\nPress Enter to exit...";
    std::cin.get();
    return 0;
}
"#;

        fs::write(template_dir.join("main.cpp"), main_cpp)?;

        // Generate vcxproj file
        let vcxproj = r#"<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup Label="ProjectConfigurations">
    <ProjectConfiguration Include="Debug|x64">
      <Configuration>Debug</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Release|x64">
      <Configuration>Release</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  
  <PropertyGroup Label="Globals">
    <VCProjectVersion>16.0</VCProjectVersion>
    <ProjectGuid>{$guid1$}</ProjectGuid>
    <Keyword>Win32Proj</Keyword>
    <RootNamespace>$safeprojectname$</RootNamespace>
    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>
  </PropertyGroup>
  
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" />
  
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <PlatformToolset>v143</PlatformToolset>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <PlatformToolset>v143</PlatformToolset>
    <WholeProgramOptimization>true</WholeProgramOptimization>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.props" />
  
  <!-- VoiRS Integration -->
  <Import Project="packages\VoiRS.FFI.0.1.0\build\VoiRS.props" Condition="exists('packages\VoiRS.FFI.0.1.0\build\VoiRS.props')" />
  
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
    <ClCompile>
      <WarningLevel>Level3</WarningLevel>
      <SDLCheck>true</SDLCheck>
      <PreprocessorDefinitions>_DEBUG;_CONSOLE;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <ConformanceMode>true</ConformanceMode>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <GenerateDebugInformation>true</GenerateDebugInformation>
    </Link>
  </ItemDefinitionGroup>
  
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <ClCompile>
      <WarningLevel>Level3</WarningLevel>
      <FunctionLevelLinking>true</FunctionLevelLinking>
      <IntrinsicFunctions>true</IntrinsicFunctions>
      <SDLCheck>true</SDLCheck>
      <PreprocessorDefinitions>NDEBUG;_CONSOLE;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <ConformanceMode>true</ConformanceMode>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <EnableCOMDATFolding>true</EnableCOMDATFolding>
      <OptimizeReferences>true</OptimizeReferences>
      <GenerateDebugInformation>true</GenerateDebugInformation>
    </Link>
  </ItemDefinitionGroup>
  
  <ItemGroup>
    <ClCompile Include="$projectname$.cpp" />
  </ItemGroup>
  
  <ItemGroup>
    <None Include="packages.config" />
  </ItemGroup>
  
  <!-- VoiRS Integration -->
  <Import Project="packages\VoiRS.FFI.0.1.0\build\VoiRS.targets" Condition="exists('packages\VoiRS.FFI.0.1.0\build\VoiRS.targets')" />
  
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.targets" />
</Project>"#;

        fs::write(template_dir.join("Template.vcxproj"), vcxproj)?;

        // Generate packages.config
        let packages_config = r#"<?xml version="1.0" encoding="utf-8"?>
<packages>
  <package id="VoiRS.FFI" version="0.1.0" targetFramework="native" />
</packages>"#;

        fs::write(template_dir.join("packages.config"), packages_config)?;

        Ok(())
    }

    /// Generate C application template
    fn generate_c_application_template(
        &self,
        templates_dir: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let template_dir = templates_dir.join("VoiRS_C_Application");
        fs::create_dir_all(&template_dir)?;

        // Similar to C++ template but with C-specific content
        let main_c = r#"/* VoiRS C Application
 * Generated from VoiRS Visual Studio Template
 */

#include <stdio.h>
#include <stdlib.h>
#include <voirs_ffi.h>

int main(void)
{
    printf("VoiRS Speech Synthesis Demo (C)\n");
    printf("===============================\n\n");

    /* Initialize VoiRS */
    if (voirs_ffi_init() != VOIRS_OK) {
        fprintf(stderr, "Failed to initialize VoiRS\n");
        return 1;
    }

    /* Create synthesis request */
    const char* text = "Hello from VoiRS C API! This is a test of the speech synthesis system.";
    printf("Synthesizing: %s\n", text);

    VoiRSAudioBuffer* audio_buffer = NULL;
    VoiRSError error = voirs_ffi_synthesize_text(text, NULL, &audio_buffer);

    if (error == VOIRS_OK && audio_buffer != NULL) {
        printf("Synthesis successful!\n");
        printf("Audio length: %u samples\n", audio_buffer->sample_count);
        printf("Sample rate: %u Hz\n", audio_buffer->sample_rate);
        printf("Channels: %u\n", audio_buffer->channels);

        /* Save to file */
        const char* output_file = "output.wav";
        if (voirs_ffi_save_audio_to_file(audio_buffer, output_file) == VOIRS_OK) {
            printf("Audio saved to %s\n", output_file);
        }

        /* Clean up */
        voirs_ffi_destroy_audio_buffer(audio_buffer);
    } else {
        fprintf(stderr, "Synthesis failed with error code: %d\n", error);
    }

    /* Cleanup VoiRS */
    voirs_ffi_cleanup();

    printf("\nPress Enter to exit...");
    getchar();
    return 0;
}
"#;

        fs::write(template_dir.join("main.c"), main_c)?;

        Ok(())
    }

    /// Generate DLL template
    fn generate_dll_template(
        &self,
        templates_dir: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let template_dir = templates_dir.join("VoiRS_DLL");
        fs::create_dir_all(&template_dir)?;

        let dll_main = r#"// VoiRS DLL Wrapper
// Generated from VoiRS Visual Studio Template

#include <windows.h>
#include <voirs_ffi.h>

// DLL Entry Point
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        // Initialize VoiRS when DLL is loaded
        return voirs_ffi_init() == VOIRS_OK;
    case DLL_THREAD_ATTACH:
        break;
    case DLL_THREAD_DETACH:
        break;
    case DLL_PROCESS_DETACH:
        // Cleanup VoiRS when DLL is unloaded
        voirs_ffi_cleanup();
        break;
    }
    return TRUE;
}

// Exported wrapper functions
__declspec(dllexport) VoiRSError __stdcall SynthesizeText(const char* text, const char* voice_id, VoiRSAudioBuffer** audio_buffer)
{
    return voirs_ffi_synthesize_text(text, voice_id, audio_buffer);
}

__declspec(dllexport) VoiRSError __stdcall SaveAudioToFile(VoiRSAudioBuffer* audio_buffer, const char* filename)
{
    return voirs_ffi_save_audio_to_file(audio_buffer, filename);
}

__declspec(dllexport) void __stdcall DestroyAudioBuffer(VoiRSAudioBuffer* audio_buffer)
{
    voirs_ffi_destroy_audio_buffer(audio_buffer);
}
"#;

        fs::write(template_dir.join("dllmain.cpp"), dll_main)?;

        Ok(())
    }

    /// Install Visual Studio integration
    pub fn install_integration(&self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(vs_path) = &self.installation_path {
            println!("Installing VoiRS Visual Studio integration...");

            let output_dir = vs_path.join("VoiRS");
            fs::create_dir_all(&output_dir)?;

            // Generate all integration files
            self.generate_msbuild_target(&output_dir)?;
            self.generate_intellisense_config(&output_dir)?;
            self.generate_debug_visualization(&output_dir)?;
            self.generate_project_templates(&output_dir)?;

            println!("VoiRS Visual Studio integration installed successfully!");
            println!("Integration files created in: {}", output_dir.display());
        } else {
            return Err("Visual Studio installation path not found".into());
        }

        Ok(())
    }

    /// Check if integration is properly installed
    pub fn verify_installation(&self) -> Result<bool, Box<dyn std::error::Error>> {
        if let Some(vs_path) = &self.installation_path {
            let integration_dir = vs_path.join("VoiRS");

            let required_files = [
                "VoiRS.targets",
                "VoiRS.props",
                "c_cpp_properties.json",
                "VoiRS.natvis",
            ];

            for file in &required_files {
                if !integration_dir.join(file).exists() {
                    return Ok(false);
                }
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get integration status information
    pub fn get_integration_info(&self) -> IntegrationInfo {
        IntegrationInfo {
            vs_version: self.vs_version,
            installation_path: self.installation_path.clone(),
            sdk_path: self.sdk_path.clone(),
            is_installed: self.verify_installation().unwrap_or(false),
        }
    }
}

/// Information about Visual Studio integration status
#[derive(Debug, Clone)]
pub struct IntegrationInfo {
    pub vs_version: VsVersion,
    pub installation_path: Option<PathBuf>,
    pub sdk_path: Option<PathBuf>,
    pub is_installed: bool,
}

impl Default for VisualStudioIntegration {
    fn default() -> Self {
        Self::new()
    }
}

/// C API functions for Visual Studio integration
#[no_mangle]
pub extern "C" fn voirs_vs_create_integration() -> *mut VisualStudioIntegration {
    Box::into_raw(Box::new(VisualStudioIntegration::new()))
}

#[no_mangle]
pub unsafe extern "C" fn voirs_vs_destroy_integration(integration: *mut VisualStudioIntegration) {
    if !integration.is_null() {
        unsafe {
            let _ = Box::from_raw(integration);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn voirs_vs_install_integration(
    integration: *mut VisualStudioIntegration,
) -> bool {
    if integration.is_null() {
        return false;
    }

    unsafe { (*integration).install_integration().is_ok() }
}

#[no_mangle]
pub unsafe extern "C" fn voirs_vs_verify_installation(
    integration: *mut VisualStudioIntegration,
) -> bool {
    if integration.is_null() {
        return false;
    }

    unsafe { (*integration).verify_installation().unwrap_or(false) }
}

#[no_mangle]
pub unsafe extern "C" fn voirs_vs_get_version(integration: *mut VisualStudioIntegration) -> u32 {
    if integration.is_null() {
        return 0;
    }

    unsafe {
        match (*integration).vs_version {
            VsVersion::Vs2022 => 2022,
            VsVersion::Vs2019 => 2019,
            VsVersion::VsBuildTools => 2017,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_vs_integration_creation() {
        let integration = VisualStudioIntegration::new();
        assert!(matches!(
            integration.vs_version,
            VsVersion::Vs2019 | VsVersion::Vs2022 | VsVersion::VsBuildTools
        ));
    }

    #[test]
    fn test_msbuild_target_generation() {
        let integration = VisualStudioIntegration::new();
        let temp_dir = TempDir::new().unwrap();

        let result = integration.generate_msbuild_target(temp_dir.path());
        assert!(result.is_ok());

        assert!(temp_dir.path().join("VoiRS.targets").exists());
        assert!(temp_dir.path().join("VoiRS.props").exists());
    }

    #[test]
    fn test_intellisense_config_generation() {
        let integration = VisualStudioIntegration::new();
        let temp_dir = TempDir::new().unwrap();

        let result = integration.generate_intellisense_config(temp_dir.path());
        assert!(result.is_ok());

        assert!(temp_dir.path().join("c_cpp_properties.json").exists());
    }

    #[test]
    fn test_debug_visualization_generation() {
        let integration = VisualStudioIntegration::new();
        let temp_dir = TempDir::new().unwrap();

        let result = integration.generate_debug_visualization(temp_dir.path());
        assert!(result.is_ok());

        assert!(temp_dir.path().join("VoiRS.natvis").exists());
        assert!(temp_dir.path().join("VoiRS_autoexp.dat").exists());
    }

    #[test]
    fn test_project_templates_generation() {
        let integration = VisualStudioIntegration::new();
        let temp_dir = TempDir::new().unwrap();

        let result = integration.generate_project_templates(temp_dir.path());
        assert!(result.is_ok());

        let templates_dir = temp_dir.path().join("ProjectTemplates");
        assert!(templates_dir.exists());
        assert!(templates_dir.join("VoiRS_CPP_Console").exists());
        assert!(templates_dir.join("VoiRS_C_Application").exists());
        assert!(templates_dir.join("VoiRS_DLL").exists());
    }

    #[test]
    fn test_natvis_content() {
        let integration = VisualStudioIntegration::new();
        let natvis_content = integration.generate_natvis_file().unwrap();

        assert!(natvis_content.contains("VoiRSAudioBuffer"));
        assert!(natvis_content.contains("VoiRSSynthesisRequest"));
        assert!(natvis_content.contains("VoiRSStats"));
        assert!(natvis_content.contains("VoiRSError"));
    }

    #[test]
    fn test_msbuild_target_content() {
        let integration = VisualStudioIntegration::new();
        let target = integration.create_msbuild_target();
        let xml = integration.generate_target_xml(&target).unwrap();

        assert!(xml.contains("VoiRS"));
        assert!(xml.contains("AdditionalIncludeDirectories"));
        assert!(xml.contains("PreprocessorDefinitions"));
        assert!(xml.contains("AdditionalDependencies"));
    }
}
