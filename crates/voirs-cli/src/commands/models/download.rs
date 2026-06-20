//! Model download command implementation.

use crate::commands::models::safetensors_support::{
    check_production_requirements, SafeTensorsLoader,
};
use crate::GlobalOptions;
use indicatif::{ProgressBar, ProgressStyle};
use hex;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use voirs_sdk::config::AppConfig;
use voirs_sdk::Result;

/// Run download model command
pub async fn run_download_model(
    model_id: &str,
    force: bool,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("Downloading model: {}", model_id);
    }

    // Check if model is already installed
    if !force && is_model_installed(model_id, config).await? {
        if !global.quiet {
            println!(
                "Model '{}' is already installed. Use --force to re-download.",
                model_id
            );
        }
        return Ok(());
    }

    // Create models directory if it doesn't exist
    let models_dir = get_models_directory(config)?;
    std::fs::create_dir_all(&models_dir)?;

    // Download the model
    download_model_from_repository(model_id, &models_dir, global).await?;

    // Verify the download
    verify_model_installation(model_id, &models_dir, global).await?;

    if !global.quiet {
        println!("Model '{}' downloaded successfully!", model_id);
    }

    Ok(())
}

/// Check if model is already installed
async fn is_model_installed(model_id: &str, config: &AppConfig) -> Result<bool> {
    let models_dir = get_models_directory(config)?;
    let model_path = models_dir.join(model_id);

    Ok(model_path.exists() && model_path.is_dir())
}

/// Get the models directory path
fn get_models_directory(config: &AppConfig) -> Result<PathBuf> {
    // Use the effective cache directory from config
    let cache_dir = config.pipeline.effective_cache_dir();
    Ok(cache_dir.join("models"))
}

/// Download model from repository
async fn download_model_from_repository(
    model_id: &str,
    models_dir: &Path,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("Downloading model '{}' from HuggingFace Hub...", model_id);
    }

    // Create model directory
    let model_dir = models_dir.join(model_id);
    std::fs::create_dir_all(&model_dir)?;

    // Get model metadata first (uses the in-house pure-Rust HuggingFace downloader)
    let mut metadata = get_model_metadata(model_id).await?;

    if !global.quiet {
        println!("Model: {}", metadata.name);
        println!("Size: {:.1} MB", metadata.total_size_mb);
        println!("Files: {}", metadata.files.len());
        println!();
    }

    // Download all model files with progress tracking. The real on-disk size of each
    // successfully downloaded file is written back into `metadata` so that later
    // verification compares like-for-like.
    download_model_files(model_id, &mut metadata, &model_dir, global).await?;

    // Verify downloads
    verify_downloaded_files(&metadata, &model_dir, global).await?;

    // Create model configuration
    create_model_config(&model_dir, model_id, &metadata)?;

    if !global.quiet {
        println!("Model '{}' downloaded successfully!", model_id);
    }

    Ok(())
}

/// Model metadata structure
#[derive(Debug, Clone)]
struct ModelMetadata {
    name: String,
    description: String,
    total_size_mb: f64,
    files: Vec<ModelFile>,
}

#[derive(Debug, Clone)]
struct ModelFile {
    name: String,
    size_bytes: u64,
    sha256: Option<String>,
}

/// Estimate a file's size in bytes from its name (used when the Hub does not
/// report an exact size up-front).
fn estimate_file_size(filename: &str) -> u64 {
    match filename {
        "pytorch_model.bin" | "model.safetensors" => 100 * 1024 * 1024, // 100MB
        "config.json" => 2048,
        "tokenizer.json" => 5 * 1024 * 1024, // 5MB
        "vocab.txt" => 1024 * 1024,          // 1MB
        _ => 1024,
    }
}

/// Get model metadata from HuggingFace Hub.
///
/// Uses the in-house pure-Rust downloader: first attempts to list the repository's
/// files via the Hub model-info API, intersecting with the set of standard model
/// files we know how to consume; if listing fails (e.g. offline or a private repo
/// without a listing), falls back to a conservative default set.
async fn get_model_metadata(model_id: &str) -> Result<ModelMetadata> {
    // Standard model files to look for.
    let standard_files = [
        "config.json",
        "pytorch_model.bin",
        "model.safetensors",
        "tokenizer.json",
        "vocab.txt",
        "special_tokens_map.json",
        "tokenizer_config.json",
    ];

    let mut files = Vec::new();

    // Ask the Hub which files actually exist, then keep only the standard ones.
    if let Ok(remote_files) = voirs_acoustic::hub::list_files(model_id, None).await {
        for filename in standard_files {
            if remote_files.iter().any(|f| f == filename) {
                files.push(ModelFile {
                    name: filename.to_string(),
                    size_bytes: estimate_file_size(filename),
                    sha256: None, // HF API would provide this
                });
            }
        }
    }

    // If listing failed or yielded nothing usable, fall back to a default set.
    if files.is_empty() {
        files = vec![
            ModelFile {
                name: "config.json".to_string(),
                size_bytes: 2048,
                sha256: None,
            },
            ModelFile {
                name: "pytorch_model.bin".to_string(),
                size_bytes: 50 * 1024 * 1024, // 50MB
                sha256: None,
            },
        ];
    }

    let total_size_mb = files.iter().map(|f| f.size_bytes).sum::<u64>() as f64 / (1024.0 * 1024.0);

    Ok(ModelMetadata {
        name: model_id.to_string(),
        description: format!("HuggingFace model: {}", model_id),
        total_size_mb,
        files,
    })
}

/// Download model files with progress tracking.
///
/// Downloads each file from the HuggingFace repository `model_id` using the in-house
/// pure-Rust downloader ([`voirs_acoustic::hub::download_file`]) and copies it into
/// `model_dir`. The real on-disk size of each successfully copied file is written
/// back into `metadata.files[i].size_bytes` (the pre-download estimate is replaced)
/// so that [`verify_downloaded_files`] compares actual sizes.
///
/// If any file fails to download or copy, an error is returned and NO placeholder
/// files are created (regression guard for issue #3).
async fn download_model_files(
    model_id: &str,
    metadata: &mut ModelMetadata,
    model_dir: &Path,
    global: &GlobalOptions,
) -> Result<()> {
    let progress_bar = if !global.quiet {
        let pb = ProgressBar::new(metadata.files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
                )
                .expect("progress template is valid")
                .progress_chars("#>-"),
        );
        pb.set_message("Downloading files");
        Some(pb)
    } else {
        None
    };

    let model_name = metadata.name.clone();
    let mut failed_files: Vec<String> = Vec::new();

    for file in metadata.files.iter_mut() {
        if let Some(pb) = &progress_bar {
            pb.set_message(format!("Downloading {}", file.name));
        }

        let file_path = model_dir.join(&file.name);

        // Download the actual file from the HuggingFace Hub via the in-house
        // pure-Rust downloader, then copy it into the model directory.
        match voirs_acoustic::hub::download_file(model_id, &file.name, None).await {
            Ok(downloaded_path) => match std::fs::copy(&downloaded_path, &file_path) {
                Ok(copied_bytes) => {
                    // Replace the size estimate with the real downloaded size.
                    file.size_bytes = copied_bytes;
                }
                Err(e) => {
                    tracing::error!("Failed to copy {}: {}", file.name, e);
                    failed_files.push(format!("{}: copy failed: {}", file.name, e));
                }
            },
            Err(e) => {
                // Download failed — propagate the error; do NOT create placeholder files
                // as that masks real failures and causes false "success" reports.
                tracing::error!("Failed to download {}: {}", file.name, e);
                failed_files.push(format!("{}: {}", file.name, e));
            }
        }

        if let Some(pb) = &progress_bar {
            pb.inc(1);
        }

        // Small delay to be gentle on the API
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    if !failed_files.is_empty() {
        if let Some(pb) = &progress_bar {
            pb.abandon_with_message("Download failed");
        }
        return Err(voirs_sdk::VoirsError::config_error(format!(
            "Model download failed for '{}'. {} file(s) could not be downloaded:\n{}",
            model_name,
            failed_files.len(),
            failed_files
                .iter()
                .map(|f| format!("  - {}", f))
                .collect::<Vec<_>>()
                .join("\n")
        )));
    }

    if let Some(pb) = &progress_bar {
        pb.finish_with_message("Download complete");
    }

    Ok(())
}

/// Verify downloaded files
async fn verify_downloaded_files(
    metadata: &ModelMetadata,
    model_dir: &Path,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("Verifying downloaded files...");
    }

    for file in &metadata.files {
        let file_path = model_dir.join(&file.name);

        if !file_path.exists() {
            return Err(voirs_sdk::VoirsError::model_error(format!(
                "Downloaded file not found: {}",
                file.name
            )));
        }

        let file_metadata = std::fs::metadata(&file_path)?;
        // A size mismatch means the file was not correctly downloaded — treat as error.
        if file_metadata.len() != file.size_bytes {
            return Err(voirs_sdk::VoirsError::model_error(format!(
                "File size mismatch for {}: expected {} bytes, got {} bytes. \
                The download may have been truncated or corrupted.",
                file.name,
                file.size_bytes,
                file_metadata.len()
            )));
        }

        // Verify SHA256 checksum if available
        if let Some(expected_hash) = &file.sha256 {
            verify_file_checksum(&file_path, expected_hash).map_err(|e| {
                voirs_sdk::VoirsError::model_error(format!(
                    "Checksum verification failed for {}: {}",
                    file.name, e
                ))
            })?;
        }
    }

    if !global.quiet {
        println!("File verification complete");
    }

    Ok(())
}

/// Create model configuration file
fn create_model_config(model_dir: &Path, model_id: &str, metadata: &ModelMetadata) -> Result<()> {
    let config = serde_json::json!({
        "model_id": model_id,
        "name": metadata.name,
        "description": metadata.description,
        "total_size_mb": metadata.total_size_mb,
        "files": metadata.files.iter().map(|f| {
            serde_json::json!({
                "name": f.name,
                "size_bytes": f.size_bytes,
                "sha256": f.sha256
            })
        }).collect::<Vec<_>>(),
        "downloaded_at": chrono::Utc::now().to_rfc3339(),
        "source": "huggingface"
    });

    let config_path = model_dir.join(".voirs-model.json");
    std::fs::write(config_path, serde_json::to_string_pretty(&config)?)?;

    Ok(())
}

/// Verify model installation with enhanced SafeTensors support
async fn verify_model_installation(
    model_id: &str,
    models_dir: &Path,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("Verifying model installation...");
    }

    let model_dir = models_dir.join(model_id);

    // Check for config.json (always required)
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        return Err(voirs_sdk::VoirsError::model_error(
            "Model verification failed: missing config.json",
        ));
    }

    // Look for model files in order of preference: SafeTensors -> PyTorch -> ONNX
    let model_files = [
        ("model.safetensors", "SafeTensors"),
        ("pytorch_model.bin", "PyTorch"),
        ("model.pt", "PyTorch"),
        ("model.onnx", "ONNX"),
    ];

    let mut found_model = None;

    for (filename, format_name) in &model_files {
        let file_path = model_dir.join(filename);
        if file_path.exists() {
            found_model = Some((file_path, format_name));
            break;
        }
    }

    let (model_path, format) = found_model.ok_or_else(|| {
        voirs_sdk::VoirsError::model_error(
            "Model verification failed: no model file found (expected .safetensors, .bin, .pt, or .onnx)"
        )
    })?;

    if !global.quiet {
        println!("Found model format: {}", format);
    }

    // Enhanced validation for SafeTensors files
    if format == &"SafeTensors" {
        if !global.quiet {
            println!("Performing SafeTensors validation...");
        }

        let loader = SafeTensorsLoader::new();

        // Validate SafeTensors format
        let validation_result = loader.validate_file(&model_path)?;

        if !validation_result.is_valid {
            return Err(voirs_sdk::VoirsError::model_error(format!(
                "SafeTensors validation failed: {}",
                validation_result.validation_errors.join(", ")
            )));
        }

        if !global.quiet {
            println!("  ✅ SafeTensors format is valid");
            println!("  📊 Tensors: {}", validation_result.tensor_count);
            println!("  💾 Size: {:.1} MB", validation_result.total_size_mb);

            if !validation_result.warnings.is_empty() {
                println!("  ⚠️  Warnings:");
                for warning in &validation_result.warnings {
                    println!("    - {}", warning);
                }
            }
        }

        // Get detailed model information
        let model_info = loader.get_model_info(&model_path)?;

        if !global.quiet {
            println!(
                "  🧠 Memory efficiency: {:.1}%",
                model_info.memory_efficiency * 100.0
            );
            println!(
                "  ⏱️  Estimated load time: {} ms",
                model_info.estimated_load_time_ms
            );
        }

        // Check production readiness
        let production_report = check_production_requirements(&model_info)?;

        if !global.quiet {
            if production_report.is_production_ready {
                println!("  🚀 Production ready: ✅");
            } else {
                println!("  🚀 Production ready: ❌");
                println!("  Issues:");
                for issue in &production_report.requirements_failed {
                    println!("    - {}", issue);
                }
            }

            if !production_report.recommendations.is_empty() {
                println!("  💡 Recommendations:");
                for rec in &production_report.recommendations {
                    println!("    - {}", rec);
                }
            }

            println!(
                "  📈 Overall score: {:.1}/10",
                production_report.overall_score * 10.0
            );
        }
    } else {
        // Basic validation for other formats
        let file_size = std::fs::metadata(&model_path)?.len();
        if !global.quiet {
            println!(
                "  📊 File size: {:.1} MB",
                file_size as f64 / (1024.0 * 1024.0)
            );
            println!("  ℹ️  Enhanced validation available for SafeTensors format");
        }
    }

    if !global.quiet {
        println!("Model verification successful");
    }

    Ok(())
}

/// Verify file SHA256 checksum
fn verify_file_checksum(file_path: &PathBuf, expected_hash: &str) -> Result<()> {
    use std::io::Read;

    let mut file = std::fs::File::open(file_path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 8192];

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let result = hasher.finalize();
    let actual_hash = hex::encode(result);

    if actual_hash != expected_hash {
        return Err(voirs_sdk::VoirsError::config_error(format!(
            "Checksum mismatch: expected {}, got {}",
            expected_hash, actual_hash
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GlobalOptions;
    use std::path::PathBuf;
    use voirs_sdk::config::AppConfig;

    #[tokio::test]
    async fn test_get_models_directory() {
        let config = AppConfig::default();
        let models_dir = get_models_directory(&config).unwrap();
        assert!(models_dir.to_string_lossy().contains("models"));
    }

    fn create_placeholder_model_files(model_dir: &std::path::Path, model_name: &str) -> Result<()> {
        use std::fs;

        // Create config.json
        let config_content = serde_json::json!({
            "model_name": model_name,
            "model_type": "acoustic",
            "version": "1.0.0",
            "sample_rate": 22050,
            "channels": 1
        });
        fs::write(
            model_dir.join("config.json"),
            serde_json::to_string_pretty(&config_content)?,
        )?;

        // Create dummy model file
        fs::write(model_dir.join("model.pt"), b"dummy model data")?;

        // Create dummy tokenizer file
        let tokenizer_content = serde_json::json!({
            "version": "1.0.0",
            "vocab_size": 1000
        });
        fs::write(
            model_dir.join("tokenizer.json"),
            serde_json::to_string_pretty(&tokenizer_content)?,
        )?;

        Ok(())
    }

    #[tokio::test]
    async fn test_create_placeholder_files() {
        let temp_dir = std::env::temp_dir().join("voirs_test_model");
        std::fs::create_dir_all(&temp_dir).unwrap();

        create_placeholder_model_files(&temp_dir, "test-model").unwrap();

        assert!(temp_dir.join("config.json").exists());
        assert!(temp_dir.join("model.pt").exists());
        assert!(temp_dir.join("tokenizer.json").exists());

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).unwrap();
    }

    /// Regression test for issue #3: verify_downloaded_files must return Err when
    /// file size does not match expected size (e.g. a truncated/placeholder download).
    ///
    /// Before the fix the function silently warned on size mismatch and returned Ok,
    /// causing the CLI to print "Model downloaded successfully!" even when the
    /// actual download had failed with HTTP 401.
    #[tokio::test]
    async fn test_size_mismatch_is_an_error() {
        let temp_dir = std::env::temp_dir().join("voirs_test_size_mismatch");
        std::fs::create_dir_all(&temp_dir).unwrap();

        // Write a file whose real size (5 bytes) is smaller than the declared size.
        let file_path = temp_dir.join("config.json");
        std::fs::write(&file_path, b"hello").unwrap();

        let metadata = ModelMetadata {
            name: "test-model".to_string(),
            description: "test".to_string(),
            total_size_mb: 0.001,
            files: vec![ModelFile {
                name: "config.json".to_string(),
                size_bytes: 2048, // declared 2048, actual 5
                sha256: None,
            }],
        };

        let global = GlobalOptions {
            quiet: true,
            verbose: 0,
            config: None,
            format: None,
            voice: None,
            gpu: false,
            threads: None,
        };

        let result = verify_downloaded_files(&metadata, &temp_dir, &global).await;
        assert!(
            result.is_err(),
            "verify_downloaded_files must return Err on size mismatch (regression for issue #3)"
        );

        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("size mismatch") || msg.contains("mismatch"),
            "error message should mention size mismatch, got: {msg}"
        );

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).unwrap();
    }

    /// Regression test for issue #3: download_model_files must propagate errors
    /// rather than silently falling back to placeholder files on download failure.
    ///
    /// This test injects a metadata list that cannot be satisfied by the filesystem
    /// (no HF repo) and expects download_model_files to return Err.
    #[tokio::test]
    async fn test_failed_download_returns_error_not_placeholder() {
        let temp_dir =
            std::env::temp_dir().join(format!("voirs_test_failed_download_{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir).unwrap();

        // Use a repository id that definitely does not exist. The in-house downloader
        // will fail to resolve it (HTTP 404 when online, or a transport error when
        // offline), which must produce Err — never a placeholder file.
        let model_id = "this-org-definitely-does-not-exist/no-such-model-xxxxxxx";

        let mut metadata = ModelMetadata {
            name: "no-such-model-xxxxxxx".to_string(),
            description: "nonexistent".to_string(),
            total_size_mb: 50.0,
            files: vec![
                ModelFile {
                    name: "config.json".to_string(),
                    size_bytes: 2048,
                    sha256: None,
                },
                ModelFile {
                    name: "pytorch_model.bin".to_string(),
                    size_bytes: 50 * 1024 * 1024,
                    sha256: None,
                },
            ],
        };

        let global = GlobalOptions {
            quiet: true,
            verbose: 0,
            config: None,
            format: None,
            voice: None,
            gpu: false,
            threads: None,
        };

        let result = download_model_files(model_id, &mut metadata, &temp_dir, &global).await;
        assert!(
            result.is_err(),
            "download_model_files must return Err when the HF repo does not exist (regression for issue #3); \
             previously it silently wrote placeholder files and returned Ok"
        );

        // Crucially: no placeholder files should have been written.
        assert!(
            !temp_dir.join("config.json").exists(),
            "no placeholder config.json should be created on download failure"
        );
        assert!(
            !temp_dir.join("pytorch_model.bin").exists(),
            "no placeholder pytorch_model.bin should be created on download failure"
        );

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).unwrap();
    }
}
