//! R Package Creation and CRAN Distribution Example
//!
//! This example demonstrates how to create a complete R package for VoiRS evaluation
//! with CRAN-compliant distribution capabilities.

#[cfg(feature = "r-integration")]
use std::path::PathBuf;
#[cfg(feature = "r-integration")]
use tempfile::TempDir;
#[cfg(feature = "r-integration")]
use voirs_evaluation::r_package_foundation::{
    create_default_voirs_package_spec, RAuthorRole, RFunctionSpec, RPackageBuilder, RParameterSpec,
    RParameterType,
};

#[cfg(feature = "r-integration")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 VoiRS R Package Creation and CRAN Distribution Example");
    println!("=========================================================");

    // Create a temporary directory for the R package
    let temp_dir = TempDir::new()?;
    let package_path = temp_dir.path().join("voirseval");

    println!("\n📦 Creating R package at: {}", package_path.display());

    // Create package specification with default VoiRS settings
    let mut spec = create_default_voirs_package_spec();

    // Customize package information if needed
    spec.version = "1.0.0".to_string();
    spec.description = "Comprehensive speech synthesis quality evaluation using the VoiRS (Voice Intelligence & Recognition System) framework. Provides objective and subjective quality metrics including PESQ, STOI, MCD, pronunciation assessment, and comparative analysis tools for speech synthesis research and development.".to_string();

    println!("📋 Package Specification:");
    println!("  Name: {}", spec.package_name);
    println!("  Version: {}", spec.version);
    println!("  Title: {}", spec.title);
    println!("  License: {}", spec.license);

    // Create package builder
    let mut builder = RPackageBuilder::new(spec, package_path.clone());

    // Add custom function if needed (in addition to defaults)
    let custom_function = RFunctionSpec {
        name: "voirs_advanced_analysis".to_string(),
        description: "Advanced speech analysis with detailed metrics".to_string(),
        parameters: vec![
            RParameterSpec {
                name: "audio_data".to_string(),
                description: "Audio data or file path".to_string(),
                param_type: RParameterType::Character,
                default: None,
                required: true,
            },
            RParameterSpec {
                name: "analysis_type".to_string(),
                description: "Type of analysis to perform".to_string(),
                param_type: RParameterType::Character,
                default: Some("'comprehensive'".to_string()),
                required: false,
            },
            RParameterSpec {
                name: "output_format".to_string(),
                description: "Output format for results".to_string(),
                param_type: RParameterType::Character,
                default: Some("'dataframe'".to_string()),
                required: false,
            },
        ],
        returns: "Detailed analysis results in specified format".to_string(),
        examples: vec![
            "# Comprehensive analysis".to_string(),
            "analysis <- voirs_advanced_analysis('audio.wav')".to_string(),
            "".to_string(),
            "# Specific analysis type".to_string(),
            "analysis <- voirs_advanced_analysis('audio.wav', 'prosody')".to_string(),
        ],
        see_also: vec!["voirs_quality_eval".to_string()],
        details: Some("Performs advanced speech analysis including prosodic features, spectral analysis, and temporal characteristics.".to_string()),
        note: Some("This function requires audio files in supported formats (WAV, FLAC, MP3).".to_string()),
        keywords: vec!["speech".to_string(), "analysis".to_string(), "advanced".to_string()],
        export: true,
    };

    builder.add_function(custom_function);

    println!("\n🔧 Building complete R package...");

    // Build the complete package with CRAN compliance
    match builder.build_complete_package().await {
        Ok(checklist) => {
            println!("✅ Package built successfully!");

            // Display CRAN compliance checklist
            println!("\n📋 CRAN Compliance Checklist:");
            println!(
                "  ✓ Valid package name: {}",
                if checklist.valid_package_name {
                    "✅"
                } else {
                    "❌"
                }
            );
            println!(
                "  ✓ Valid version: {}",
                if checklist.valid_version {
                    "✅"
                } else {
                    "❌"
                }
            );
            println!(
                "  ✓ Adequate description: {}",
                if checklist.adequate_description {
                    "✅"
                } else {
                    "❌"
                }
            );
            println!(
                "  ✓ Authors specified: {}",
                if checklist.authors_specified {
                    "✅"
                } else {
                    "❌"
                }
            );
            println!(
                "  ✓ Compatible license: {}",
                if checklist.compatible_license {
                    "✅"
                } else {
                    "❌"
                }
            );
            println!(
                "  ✓ Dependencies resolved: {}",
                if checklist.dependencies_resolved {
                    "✅"
                } else {
                    "❌"
                }
            );
            println!(
                "  ✓ Examples work: {}",
                if checklist.examples_work {
                    "✅"
                } else {
                    "❌"
                }
            );
            println!(
                "  ✓ Tests pass: {}",
                if checklist.tests_pass { "✅" } else { "❌" }
            );
            println!(
                "  ✓ Documentation complete: {}",
                if checklist.documentation_complete {
                    "✅"
                } else {
                    "❌"
                }
            );

            if !checklist.submission_notes.is_empty() {
                println!("\n📝 Submission Notes:");
                for note in &checklist.submission_notes {
                    println!("  • {}", note);
                }
            }

            // List generated files
            println!("\n📁 Generated Package Structure:");
            list_package_contents(&package_path).await?;

            // Display next steps
            println!("\n🚀 Next Steps for CRAN Submission:");
            println!(
                "1. Navigate to package directory: cd {}",
                package_path.display()
            );
            println!("2. Run R CMD check: ./check-package.sh");
            println!("3. Review generated documentation and tests");
            println!("4. Test package installation locally");
            println!("5. Prepare for CRAN submission: ./submit-to-cran.sh");
            println!("6. Follow the submission checklist created in the package");

            // Example of how to inspect generated files
            println!("\n📖 Example: Inspecting Generated Files");

            // Read and display part of the DESCRIPTION file
            if let Ok(description) =
                tokio::fs::read_to_string(package_path.join("DESCRIPTION")).await
            {
                println!("\nDESCRIPTION file preview:");
                for (i, line) in description.lines().take(10).enumerate() {
                    println!("  {}: {}", i + 1, line);
                }
                if description.lines().count() > 10 {
                    println!("  ... (truncated)");
                }
            }

            // Display part of a function file
            if let Ok(function_code) =
                tokio::fs::read_to_string(package_path.join("R").join("voirs_quality_eval.R")).await
            {
                println!("\nFunction code preview (voirs_quality_eval.R):");
                for (i, line) in function_code.lines().take(15).enumerate() {
                    println!("  {}: {}", i + 1, line);
                }
                if function_code.lines().count() > 15 {
                    println!("  ... (truncated)");
                }
            }

            println!("\n🎉 R package creation completed successfully!");
            println!("📦 Package is ready for CRAN submission process.");
        }
        Err(e) => {
            eprintln!("❌ Failed to build package: {}", e);
            return Err(e.into());
        }
    }

    println!("\n💡 Tips for CRAN Submission:");
    println!("• Ensure all examples run without errors");
    println!("• Test on multiple R versions if possible");
    println!("• Review CRAN policies: https://cran.r-project.org/web/packages/policies.html");
    println!("• Consider submitting to win-builder for pre-check");
    println!("• Respond promptly to CRAN reviewer feedback");

    Ok(())
}

/// List the contents of the generated package directory
#[cfg(feature = "r-integration")]
async fn list_package_contents(
    package_path: &std::path::PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    use tokio::fs;

    #[allow(clippy::type_complexity)]
    fn list_directory<'a>(
        path: &'a std::path::PathBuf,
        prefix: &'a str,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<(), Box<dyn std::error::Error>>> + 'a>,
    > {
        Box::pin(async move {
            let mut entries = fs::read_dir(path).await?;
            let mut items = Vec::new();

            while let Some(entry) = entries.next_entry().await? {
                let entry_path = entry.path();
                let name = entry_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("?");

                if entry_path.is_dir() {
                    items.push((name.to_string(), true));
                } else {
                    items.push((name.to_string(), false));
                }
            }

            items.sort_by(|a, b| match (a.1, b.1) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.0.cmp(&b.0),
            });

            for (name, is_dir) in items {
                if is_dir {
                    println!("{}📁 {}/", prefix, name);
                    if name != "." && name != ".." {
                        let new_prefix = format!("{}  ", prefix);
                        let _ = list_directory(&path.join(&name), &new_prefix).await;
                    }
                } else {
                    println!("{}📄 {}", prefix, name);
                }
            }

            Ok(())
        })
    }

    list_directory(package_path, "  ").await
}

#[cfg(not(feature = "r-integration"))]
fn main() {
    println!("R integration feature is not enabled. Build with --features r-integration to run this example.");
}
