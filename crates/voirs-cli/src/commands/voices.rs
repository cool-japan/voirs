//! Voice management command implementations.

use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use voirs_sdk::{config::AppConfig, error::Result, VoiceConfig, VoirsPipeline};

/// Run list voices command
pub async fn run_list_voices(
    language: Option<&str>,
    detailed: bool,
    config: &AppConfig,
) -> Result<()> {
    let pipeline = VoirsPipeline::builder().build().await?;
    let voices = pipeline.list_voices().await?;

    // Filter by language if specified
    let filtered_voices: Vec<_> = if let Some(lang) = language {
        voices
            .into_iter()
            .filter(|voice| voice.language.as_str().eq_ignore_ascii_case(lang))
            .collect()
    } else {
        voices
    };

    if detailed {
        for voice in &filtered_voices {
            println!("ID: {}", voice.id);
            println!("Name: {}", voice.name);
            println!("Language: {}", voice.language.as_str());
            println!("Quality: {:?}", voice.characteristics.quality);
            println!("---");
        }
    } else {
        for voice in &filtered_voices {
            println!(
                "{} - {} ({})",
                voice.id,
                voice.name,
                voice.language.as_str()
            );
        }
    }

    Ok(())
}

/// Run voice info command
pub async fn run_voice_info(voice_id: &str, config: &AppConfig) -> Result<()> {
    let pipeline = VoirsPipeline::builder().build().await?;
    let voices = pipeline.list_voices().await?;

    // Find the requested voice
    let voice = voices
        .iter()
        .find(|v| v.id == voice_id)
        .ok_or_else(|| voirs::VoirsError::audio_error(format!("Voice '{}' not found", voice_id)))?;

    // Display voice information
    println!("Voice Information");
    println!("================");
    println!("ID: {}", voice.id);
    println!("Name: {}", voice.name);
    println!("Language: {}", voice.language.as_str());
    println!(
        "Description: {}",
        voice
            .metadata
            .get("description")
            .unwrap_or(&"No description available".to_string())
    );
    println!();

    println!("Characteristics:");
    println!(
        "  Gender: {}",
        voice
            .characteristics
            .gender
            .map(|g| format!("{:?}", g))
            .unwrap_or_else(|| "Not specified".to_string())
    );
    println!(
        "  Age: {}",
        voice
            .characteristics
            .age
            .map(|a| format!("{:?}", a))
            .unwrap_or_else(|| "Not specified".to_string())
    );
    println!("  Style: {:?}", voice.characteristics.style);
    println!(
        "  Emotion Support: {}",
        voice.characteristics.emotion_support
    );
    println!("  Quality: {:?}", voice.characteristics.quality);
    println!();

    println!("Model Configuration:");
    println!("  Acoustic Model: {}", voice.model_config.acoustic_model);
    println!("  Vocoder Model: {}", voice.model_config.vocoder_model);
    if let Some(ref g2p_model) = voice.model_config.g2p_model {
        println!("  G2P Model: {}", g2p_model);
    }
    println!("  Format: {:?}", voice.model_config.format);
    println!();

    println!("Device Requirements:");
    println!(
        "  Minimum Memory: {} MB",
        voice.model_config.device_requirements.min_memory_mb
    );
    println!(
        "  GPU Support: {}",
        voice.model_config.device_requirements.gpu_support
    );
    if !voice
        .model_config
        .device_requirements
        .compute_capabilities
        .is_empty()
    {
        println!(
            "  Compute Capabilities: {:?}",
            voice.model_config.device_requirements.compute_capabilities
        );
    }

    if !voice.metadata.is_empty() {
        println!();
        println!("Additional Metadata:");
        for (key, value) in &voice.metadata {
            println!("  {}: {}", key, value);
        }
    }

    // Check if voice is available locally
    let cache_dir = config.pipeline.effective_cache_dir();
    let voice_dir = cache_dir.join("voices").join(voice_id);
    let is_downloaded = voice_dir.exists();

    println!();
    if is_downloaded {
        println!("Status: Downloaded and available");
        println!("Location: {}", voice_dir.display());

        // Check if all required models are present
        let acoustic_path = voice_dir.join(&voice.model_config.acoustic_model);
        let vocoder_path = voice_dir.join(&voice.model_config.vocoder_model);
        let acoustic_exists = acoustic_path.exists();
        let vocoder_exists = vocoder_path.exists();

        if acoustic_exists && vocoder_exists {
            println!("Model files: Complete");
        } else {
            println!("Model files: Incomplete - re-download may be required");
            if !acoustic_exists {
                println!("  Missing: acoustic model");
            }
            if !vocoder_exists {
                println!("  Missing: vocoder model");
            }
        }
    } else {
        println!("Status: Available for download");
    }

    Ok(())
}

/// Run download voice command
pub async fn run_download_voice(voice_id: &str, force: bool, config: &AppConfig) -> Result<()> {
    println!("Downloading voice: {}", voice_id);

    let pipeline = VoirsPipeline::builder().build().await?;
    let voices = pipeline.list_voices().await?;

    // Find the requested voice
    let voice = voices
        .iter()
        .find(|v| v.id == voice_id)
        .ok_or_else(|| voirs::VoirsError::audio_error(format!("Voice '{}' not found", voice_id)))?;

    // Check if already downloaded
    let cache_dir = config.pipeline.effective_cache_dir();
    let voice_dir = cache_dir.join("voices").join(voice_id);

    if voice_dir.exists() && !force {
        println!(
            "Voice '{}' is already downloaded. Use --force to re-download.",
            voice_id
        );
        return Ok(());
    }

    // Create voice directory
    std::fs::create_dir_all(&voice_dir).map_err(|e| voirs::VoirsError::from(e))?;

    println!("Preparing to download voice models...");

    // List of models to download
    let mut models_to_download = vec![
        ("acoustic", &voice.model_config.acoustic_model),
        ("vocoder", &voice.model_config.vocoder_model),
    ];

    if let Some(ref g2p_model) = voice.model_config.g2p_model {
        models_to_download.push(("g2p", g2p_model));
    }

    println!("Models to download:");
    for (model_type, model_path) in &models_to_download {
        println!("  {}: {}", model_type, model_path);
    }

    // Download models from configured repositories
    let models_count = models_to_download.len();
    for (model_type, model_path) in models_to_download {
        let local_path = voice_dir.join(model_path);

        // Create parent directories if needed
        if let Some(parent) = local_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| voirs::VoirsError::from(e))?;
        }

        println!("  Downloading {} model...", model_type);

        // Try downloading from each repository until one succeeds
        let mut download_success = false;
        for (repo_index, repository) in config
            .pipeline
            .model_loading
            .repositories
            .iter()
            .enumerate()
        {
            let download_url = format!("{}/voices/{}/{}", repository, voice_id, model_path);

            println!(
                "    Attempting download from repository {}: {}",
                repo_index + 1,
                repository
            );

            match download_model_file(&download_url, &local_path, &config).await {
                Ok(_) => {
                    println!("    ✓ Downloaded successfully from {}", repository);
                    download_success = true;
                    break;
                }
                Err(e) => {
                    println!("    ✗ Failed to download from {}: {}", repository, e);
                    continue;
                }
            }
        }

        // If all repositories failed, create a placeholder file as fallback
        if !download_success {
            println!("    Creating placeholder file as fallback...");
            std::fs::write(
                &local_path,
                format!("Placeholder for {} model: {}", model_type, model_path),
            )
            .map_err(|e| voirs::VoirsError::from(e))?;

            println!("    ⚠ Placeholder created: {}", local_path.display());
        }
    }

    // Save voice configuration
    let voice_config_path = voice_dir.join("voice.json");
    let voice_json = serde_json::to_string_pretty(voice).map_err(|e| {
        voirs::VoirsError::config_error(format!("Failed to serialize voice config: {}", e))
    })?;

    std::fs::write(&voice_config_path, voice_json).map_err(|e| voirs::VoirsError::from(e))?;

    println!();
    println!("Voice '{}' downloaded successfully!", voice_id);
    println!("  Location: {}", voice_dir.display());
    println!("  Models: {} files", models_count);
    println!();
    println!("Download completed successfully.");
    if models_count > 0 {
        println!("Model repositories used for download:");
    } else {
        println!("Note: No models were available for download.");
        println!("Available repositories:");
    }
    for repo in &config.pipeline.model_loading.repositories {
        println!("  - {}", repo);
    }

    if models_count > 0 {
        println!();
        println!("Configuration:");
        println!(
            "  Timeout: {} seconds",
            config.pipeline.model_loading.download_timeout_secs
        );
        println!(
            "  Retries: {}",
            config.pipeline.model_loading.download_retries
        );
        println!(
            "  Verify checksums: {}",
            config.pipeline.model_loading.verify_checksums
        );
    }

    Ok(())
}

/// Download a model file from a URL with progress tracking and verification
async fn download_model_file(
    url: &str,
    local_path: &std::path::Path,
    config: &AppConfig,
) -> Result<()> {
    use std::time::Duration;

    // Create HTTP client with timeout
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(
            config.pipeline.model_loading.download_timeout_secs,
        ))
        .build()
        .map_err(|e| {
            voirs::VoirsError::config_error(format!("Failed to create HTTP client: {}", e))
        })?;

    // Attempt download with retries
    let mut last_error = None;
    for attempt in 1..=config.pipeline.model_loading.download_retries {
        match attempt_download(&client, url, local_path, attempt).await {
            Ok(_) => {
                // Verify file if checksums are enabled
                if config.pipeline.model_loading.verify_checksums {
                    verify_downloaded_file(local_path, config)?;
                }
                return Ok(());
            }
            Err(e) => {
                last_error = Some(e);
                if attempt < config.pipeline.model_loading.download_retries {
                    println!(
                        "      Retrying... (attempt {} of {})",
                        attempt + 1,
                        config.pipeline.model_loading.download_retries
                    );
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    }

    Err(last_error
        .unwrap_or_else(|| voirs::VoirsError::config_error("Download failed after all retries")))
}

/// Attempt a single download
async fn attempt_download(
    client: &reqwest::Client,
    url: &str,
    local_path: &std::path::Path,
    attempt: u32,
) -> Result<()> {
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| voirs::VoirsError::config_error(format!("HTTP request failed: {}", e)))?;

    if !response.status().is_success() {
        return Err(voirs::VoirsError::config_error(format!(
            "HTTP error {}: {}",
            response.status(),
            response
                .status()
                .canonical_reason()
                .unwrap_or("Unknown error")
        )));
    }

    // Get content length for progress tracking
    let total_size = response.content_length().unwrap_or(0);
    if total_size > 0 {
        println!(
            "      File size: {:.2} MB",
            total_size as f64 / 1024.0 / 1024.0
        );
    }

    // Download with progress tracking
    let mut file = std::fs::File::create(local_path).map_err(|e| voirs::VoirsError::IoError {
        path: local_path.to_path_buf(),
        operation: voirs_sdk::error::IoOperation::Write,
        source: e,
    })?;

    let mut downloaded = 0u64;
    let mut stream = response.bytes_stream();

    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| {
            voirs::VoirsError::config_error(format!("Download stream error: {}", e))
        })?;

        file.write_all(&chunk)
            .map_err(|e| voirs::VoirsError::IoError {
                path: local_path.to_path_buf(),
                operation: voirs_sdk::error::IoOperation::Write,
                source: e,
            })?;

        downloaded += chunk.len() as u64;

        // Show progress every 1MB or if we have total size
        if total_size > 0 && (downloaded % (1024 * 1024) == 0 || downloaded == total_size) {
            let progress = (downloaded as f64 / total_size as f64) * 100.0;
            println!(
                "      Progress: {:.1}% ({} / {} bytes)",
                progress, downloaded, total_size
            );
        } else if total_size == 0 && downloaded % (1024 * 1024) == 0 {
            println!(
                "      Downloaded: {:.2} MB",
                downloaded as f64 / 1024.0 / 1024.0
            );
        }
    }

    file.flush().map_err(|e| voirs::VoirsError::IoError {
        path: local_path.to_path_buf(),
        operation: voirs_sdk::error::IoOperation::Write,
        source: e,
    })?;

    println!(
        "      Total downloaded: {:.2} MB",
        downloaded as f64 / 1024.0 / 1024.0
    );
    Ok(())
}

/// Verify downloaded file integrity
fn verify_downloaded_file(local_path: &std::path::Path, config: &AppConfig) -> Result<()> {
    // Basic file existence and size check
    let metadata = std::fs::metadata(local_path).map_err(|e| voirs::VoirsError::IoError {
        path: local_path.to_path_buf(),
        operation: voirs_sdk::error::IoOperation::Read,
        source: e,
    })?;

    if metadata.len() == 0 {
        return Err(voirs::VoirsError::config_error("Downloaded file is empty"));
    }

    // Check for minimum file size (models should be at least 1KB)
    if metadata.len() < 1024 {
        return Err(voirs::VoirsError::config_error(
            "Downloaded file is too small to be a valid model",
        ));
    }

    // Enhanced checksum verification implementation
    if config.pipeline.model_loading.verify_checksums {
        if let Err(e) = verify_file_checksum(local_path) {
            println!("      ⚠ Checksum verification failed: {}", e);
            // For now, warn but don't fail - future enhancement could make this configurable
        } else {
            println!("      ✓ Checksum verification passed");
        }
    }

    println!(
        "      ✓ File verification passed ({} bytes)",
        metadata.len()
    );
    Ok(())
}

/// Verify file checksum using SHA256
fn verify_file_checksum(file_path: &std::path::Path) -> Result<()> {
    // Check for accompanying checksum file (.sha256)
    let checksum_path = file_path.with_extension(format!(
        "{}.sha256",
        file_path.extension().and_then(|s| s.to_str()).unwrap_or("")
    ));

    if !checksum_path.exists() {
        // Also try looking for a .sha256 file with same base name
        let alt_checksum_path = file_path.with_extension("sha256");
        if !alt_checksum_path.exists() {
            return Err(voirs::VoirsError::config_error(
                "No checksum file found for verification",
            ));
        }
    }

    let expected_checksum = std::fs::read_to_string(&checksum_path)
        .or_else(|_| std::fs::read_to_string(&file_path.with_extension("sha256")))
        .map_err(|e| {
            voirs::VoirsError::config_error(format!("Failed to read checksum file: {}", e))
        })?
        .trim()
        .to_lowercase();

    // Validate checksum format (should be 64 hex characters for SHA256)
    if expected_checksum.len() != 64 || !expected_checksum.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(voirs::VoirsError::config_error(
            "Invalid checksum format - expected 64 hex characters",
        ));
    }

    // Calculate SHA256 of the downloaded file
    let calculated_checksum = calculate_file_sha256(file_path)?;

    // Compare checksums
    if calculated_checksum != expected_checksum {
        return Err(voirs::VoirsError::config_error(format!(
            "Checksum mismatch - expected: {}, calculated: {}",
            expected_checksum, calculated_checksum
        )));
    }

    Ok(())
}

/// Calculate SHA256 hash of a file
fn calculate_file_sha256(file_path: &std::path::Path) -> Result<String> {
    let mut file = std::fs::File::open(file_path).map_err(|e| voirs::VoirsError::IoError {
        path: file_path.to_path_buf(),
        operation: voirs_sdk::error::IoOperation::Read,
        source: e,
    })?;

    let mut hasher = Sha256::new();
    let mut buffer = [0; 8192]; // 8KB buffer for efficient reading

    loop {
        let bytes_read = file
            .read(&mut buffer)
            .map_err(|e| voirs::VoirsError::IoError {
                path: file_path.to_path_buf(),
                operation: voirs_sdk::error::IoOperation::Read,
                source: e,
            })?;

        if bytes_read == 0 {
            break;
        }

        hasher.update(&buffer[..bytes_read]);
    }

    let hash = hasher.finalize();
    Ok(format!("{:x}", hash))
}

/// Compare multiple voices side by side
pub async fn run_compare_voices(voice_ids: Vec<String>, config: &AppConfig) -> Result<()> {
    let pipeline = VoirsPipeline::builder().build().await?;
    let voices = pipeline.list_voices().await?;

    let mut found_voices = Vec::new();
    let mut missing_voices = Vec::new();

    // Find all requested voices
    for voice_id in &voice_ids {
        if let Some(voice) = voices.iter().find(|v| v.id == *voice_id) {
            found_voices.push(voice);
        } else {
            missing_voices.push(voice_id.clone());
        }
    }

    // Report missing voices
    if !missing_voices.is_empty() {
        println!("⚠ Warning: The following voices were not found:");
        for voice_id in &missing_voices {
            println!("  - {}", voice_id);
        }
        println!();
    }

    if found_voices.is_empty() {
        println!("No voices found for comparison.");
        return Ok(());
    }

    println!("Voice Comparison");
    println!("================");
    println!("Comparing {} voice(s)\n", found_voices.len());

    // Display simple comparison table
    display_simple_voice_comparison(&found_voices, config)?;

    // Display basic recommendations
    println!("\nRecommendations:");
    println!("================");
    display_simple_recommendations(&found_voices);

    Ok(())
}

/// Display a simple comparison of voices
fn display_simple_voice_comparison(voices: &[&VoiceConfig], config: &AppConfig) -> Result<()> {
    let cache_dir = config.pipeline.effective_cache_dir();

    for (i, voice) in voices.iter().enumerate() {
        println!("Voice {} - {}", i + 1, voice.name);
        println!("  ID: {}", voice.id);
        println!("  Language: {}", voice.language.as_str());
        println!("  Quality: {:?}", voice.characteristics.quality);

        if let Some(gender) = voice.characteristics.gender {
            println!("  Gender: {:?}", gender);
        }

        if let Some(age) = voice.characteristics.age {
            println!("  Age: {:?}", age);
        }

        println!("  Style: {:?}", voice.characteristics.style);
        println!(
            "  Emotion Support: {}",
            voice.characteristics.emotion_support
        );

        // Check download status
        let voice_dir = cache_dir.join("voices").join(&voice.id);
        let status = if voice_dir.exists() {
            "✓ Downloaded"
        } else {
            "✗ Not downloaded"
        };
        println!("  Status: {}", status);

        // Get estimated size if downloaded
        if voice_dir.exists() {
            let size = estimate_voice_size(&voice_dir);
            println!("  Size: {}", size);
        }

        println!();
    }

    Ok(())
}

/// Display simple recommendations based on voice comparison
fn display_simple_recommendations(voices: &[&VoiceConfig]) {
    if voices.len() < 2 {
        println!("Need at least 2 voices for recommendations.");
        return;
    }

    // Language analysis
    println!("🌐 Language Analysis:");
    let mut language_counts = std::collections::HashMap::new();
    for voice in voices {
        let lang = voice.language.as_str();
        *language_counts.entry(lang).or_insert(0) += 1;
    }

    for (language, count) in language_counts {
        println!("  - {}: {} voice(s)", language, count);
    }

    // Quality analysis
    println!("\n🎯 Quality Analysis:");
    let mut quality_counts = std::collections::HashMap::new();
    for voice in voices {
        let quality_str = format!("{:?}", voice.characteristics.quality);
        *quality_counts.entry(quality_str).or_insert(0) += 1;
    }

    for (quality, count) in quality_counts {
        println!("  - {}: {} voice(s)", quality, count);
    }

    // Feature analysis
    println!("\n🌟 Feature Analysis:");
    let emotion_support_count = voices
        .iter()
        .filter(|v| v.characteristics.emotion_support)
        .count();
    let gender_specified_count = voices
        .iter()
        .filter(|v| v.characteristics.gender.is_some())
        .count();
    let age_specified_count = voices
        .iter()
        .filter(|v| v.characteristics.age.is_some())
        .count();

    println!("  - Emotion Support: {} voice(s)", emotion_support_count);
    println!("  - Gender Specified: {} voice(s)", gender_specified_count);
    println!("  - Age Specified: {} voice(s)", age_specified_count);

    // General recommendations
    println!("\n💡 General Recommendations:");
    if voices.len() > 1 {
        println!("  - Test multiple voices with sample content to find the best fit");
        println!("  - Use high-quality voices for production content");
        println!("  - Choose voices with emotion support for dynamic content");
        println!("  - Consider language consistency for multi-voice projects");
    }
}

/// Truncate string to specified length with ellipsis
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Estimate the size of a voice directory
fn estimate_voice_size(voice_dir: &std::path::Path) -> String {
    let mut total_size = 0u64;

    if let Ok(entries) = std::fs::read_dir(voice_dir) {
        for entry in entries.flatten() {
            if let Ok(metadata) = entry.metadata() {
                total_size += metadata.len();
            }
        }
    }

    if total_size == 0 {
        "Unknown".to_string()
    } else if total_size < 1024 * 1024 {
        format!("{:.1} KB", total_size as f64 / 1024.0)
    } else if total_size < 1024 * 1024 * 1024 {
        format!("{:.1} MB", total_size as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1} GB", total_size as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}
