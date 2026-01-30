//! Dataset Versioning Example
//!
//! Demonstrates how to use dataset versioning for reproducibility and integrity checking.
//! This example shows:
//! - Creating dataset manifests
//! - Version management
//! - Checksum verification
//! - Manifest persistence

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use tempfile::TempDir;
use voirs_dataset::versioning::{
    DatasetManifest, DatasetStatistics, DatasetVersion, DatasetVersioningBuilder, FileChecksum,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Dataset Versioning Example ===\n");

    // Create temporary directory for demonstration
    let temp_dir = TempDir::new()?;
    let dataset_path = temp_dir.path().to_path_buf();

    // Example 1: Create a dataset with versioning
    println!("--- Example 1: Creating Dataset with Version Control ---");
    create_versioned_dataset(&dataset_path)?;

    // Example 2: Load and verify dataset
    println!("\n--- Example 2: Loading and Verifying Dataset ---");
    verify_dataset_integrity(&dataset_path)?;

    // Example 3: Version compatibility
    println!("\n--- Example 3: Version Compatibility Checking ---");
    version_compatibility_example()?;

    // Example 4: Incremental updates
    println!("\n--- Example 4: Incremental Dataset Updates ---");
    incremental_update_example(&dataset_path)?;

    println!("\n=== All Examples Completed Successfully ===");
    Ok(())
}

fn create_versioned_dataset(dataset_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    // Create some sample audio files
    let audio_dir = dataset_path.join("audio");
    fs::create_dir_all(&audio_dir)?;

    println!("Creating sample dataset files...");
    for i in 1..=5 {
        let file_path = audio_dir.join(format!("sample_{:03}.wav", i));
        let mut file = File::create(&file_path)?;
        // Write dummy WAV header and data
        file.write_all(&vec![0; 1024])?;
        println!("  Created: {}", file_path.display());
    }

    // Create metadata file
    let metadata_path = dataset_path.join("metadata.json");
    let mut metadata_file = File::create(&metadata_path)?;
    metadata_file.write_all(b"{\"dataset\":\"example\"}")?;

    // Build versioned manifest
    println!("\nBuilding dataset manifest...");
    let version = DatasetVersion::new(1, 0, 0);
    let manifest =
        DatasetVersioningBuilder::new("example-dataset".to_string(), version, dataset_path)
            .creator("VoiRS Example".to_string())
            .description("Example dataset for versioning demonstration".to_string())
            .license("MIT".to_string())
            .metadata("created_by".to_string(), "voirs-dataset".to_string())
            .metadata("purpose".to_string(), "example".to_string())
            .scan_directory(None)?
            .statistics(DatasetStatistics {
                total_samples: 5,
                total_duration: 10.0,
                total_size: 5 * 1024,
                num_speakers: Some(1),
                sample_rates: vec![16000],
                languages: vec!["en-US".to_string()],
            })
            .build();

    println!("  Dataset: {}", manifest.name);
    println!("  Version: {}", manifest.version.as_string());
    println!("  Files tracked: {}", manifest.files.len());
    println!("  Total size: {} bytes", manifest.statistics.total_size);
    println!("  Manifest hash: {}", manifest.calculate_hash());

    // Save manifest
    let manifest_path = dataset_path.join("manifest.json");
    manifest.save(&manifest_path)?;
    println!("\nManifest saved to: {}", manifest_path.display());

    Ok(())
}

fn verify_dataset_integrity(dataset_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = dataset_path.join("manifest.json");

    println!("Loading manifest from: {}", manifest_path.display());
    let manifest = DatasetManifest::load(&manifest_path)?;

    println!(
        "  Dataset: {} v{}",
        manifest.name,
        manifest.version.as_string()
    );
    println!("  Creator: {:?}", manifest.creator);
    println!("  Files: {}", manifest.files.len());

    // Verify all files
    println!("\nVerifying file integrity...");
    let report = manifest.verify_all(dataset_path)?;

    println!("  Verified: {} files", report.verified_files);
    println!("  Failed: {} files", report.failed_files);
    println!("  Missing: {} files", report.missing_files);

    if report.is_valid() {
        println!("  ✓ Dataset integrity verified!");
    } else {
        println!("  ✗ Dataset has integrity issues:");
        for error in &report.errors {
            println!("    - {}", error);
        }
    }

    // Demonstrate checksum calculation
    if let Some(first_file) = manifest.files.first() {
        println!("\nFile checksum example:");
        println!("  File: {}", first_file.path.display());
        println!("  MD5: {}", first_file.md5);
        println!("  Size: {} bytes", first_file.size);
    }

    Ok(())
}

fn version_compatibility_example() -> Result<(), Box<dyn std::error::Error>> {
    let v1_0_0 = DatasetVersion::new(1, 0, 0);
    let v1_2_0 = DatasetVersion::new(1, 2, 0);
    let v2_0_0 = DatasetVersion::new(2, 0, 0);
    let v1_0_0_beta = DatasetVersion::with_pre_release(1, 0, 0, "beta.1".to_string());

    println!("Version strings:");
    println!("  v1.0.0: {}", v1_0_0.as_string());
    println!("  v1.2.0: {}", v1_2_0.as_string());
    println!("  v2.0.0: {}", v2_0_0.as_string());
    println!("  v1.0.0-beta.1: {}", v1_0_0_beta.as_string());

    println!("\nCompatibility checks:");
    println!(
        "  v1.2.0 compatible with v1.0.0? {}",
        v1_2_0.is_compatible_with(&v1_0_0)
    );
    println!(
        "  v1.0.0 compatible with v1.2.0? {}",
        v1_0_0.is_compatible_with(&v1_2_0)
    );
    println!(
        "  v2.0.0 compatible with v1.0.0? {}",
        v2_0_0.is_compatible_with(&v1_0_0)
    );

    // Parse versions from strings
    println!("\nParsing version strings:");
    let parsed = DatasetVersion::parse("1.2.3")?;
    println!("  Parsed '1.2.3': {}", parsed.as_string());

    let parsed_pre = DatasetVersion::parse("2.0.0-alpha.1")?;
    println!("  Parsed '2.0.0-alpha.1': {}", parsed_pre.as_string());

    Ok(())
}

fn incremental_update_example(dataset_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("Simulating dataset update workflow...");

    // Load existing manifest
    let manifest_path = dataset_path.join("manifest.json");
    let old_manifest = DatasetManifest::load(&manifest_path)?;
    let old_hash = old_manifest.calculate_hash();

    println!("  Old version: {}", old_manifest.version.as_string());
    println!("  Old hash: {}", old_hash);

    // Add a new file
    let audio_dir = dataset_path.join("audio");
    let new_file = audio_dir.join("sample_006.wav");
    let mut file = File::create(&new_file)?;
    file.write_all(&vec![0; 1024])?;
    println!("\n  Added new file: {}", new_file.display());

    // Create updated manifest with new version
    let new_version = DatasetVersion::new(1, 1, 0); // Bump minor version
    let new_manifest =
        DatasetVersioningBuilder::new(old_manifest.name.clone(), new_version, dataset_path)
            .creator(old_manifest.creator.clone().unwrap_or_default())
            .description(old_manifest.description.clone().unwrap_or_default())
            .license(old_manifest.license.clone().unwrap_or_default())
            .scan_directory(None)?
            .statistics(DatasetStatistics {
                total_samples: 6, // Updated count
                total_duration: 12.0,
                total_size: 6 * 1024,
                num_speakers: Some(1),
                sample_rates: vec![16000],
                languages: vec!["en-US".to_string()],
            })
            .build();

    let new_hash = new_manifest.calculate_hash();

    println!("\n  New version: {}", new_manifest.version.as_string());
    println!("  New hash: {}", new_hash);
    println!(
        "  Files: {} -> {}",
        old_manifest.files.len(),
        new_manifest.files.len()
    );
    println!("  Hash changed: {}", old_hash != new_hash);

    // Check compatibility
    println!(
        "\n  New version compatible with old? {}",
        new_manifest
            .version
            .is_compatible_with(&old_manifest.version)
    );

    // Save updated manifest
    new_manifest.save(&manifest_path)?;
    println!("\n  Updated manifest saved!");

    Ok(())
}
