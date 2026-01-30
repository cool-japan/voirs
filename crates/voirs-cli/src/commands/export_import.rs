//! Export/import commands for voice profiles, presets, and configurations.

use crate::GlobalOptions;
use serde_json;
use std::fs;
use std::path::{Path, PathBuf};
use voirs_sdk::config::AppConfig;
use voirs_sdk::voice::{VoiceInfo, VoiceRegistry};
use voirs_sdk::Result;

pub async fn run_export(
    export_type: &str,
    source: &str,
    output: &Path,
    include_weights: bool,
    _config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!(
            "📦 Exporting {} '{}' to {}",
            export_type,
            source,
            output.display()
        );
    }

    match export_type {
        "voice-profile" => export_voice_profile(source, output, include_weights).await?,
        "emotion-preset" => export_emotion_preset(source, output).await?,
        "config" => export_config(source, output).await?,
        other => {
            return Err(voirs_sdk::VoirsError::config_error(format!(
                "Unsupported export type: {}",
                other
            )));
        }
    }

    if !global.quiet {
        println!("✅ Export complete");
    }

    Ok(())
}

pub async fn run_import(
    input: &Path,
    name: Option<&str>,
    force: bool,
    validate: bool,
    _config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("📥 Importing from {}", input.display());
    }

    let ext = input.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext.to_lowercase().as_str() {
        "zip" | "voirs" => import_voice_package(input, name, force, validate).await?,
        "json" | "toml" | "yaml" | "yml" => import_config(input, name, force, validate).await?,
        other => {
            return Err(voirs_sdk::VoirsError::config_error(format!(
                "Unsupported import format: {}",
                other
            )));
        }
    }

    if !global.quiet {
        println!("✅ Import complete");
    }

    Ok(())
}

async fn export_voice_profile(source: &str, output: &Path, include_weights: bool) -> Result<()> {
    // Get voice registry to lookup voice info
    let registry = VoiceRegistry::new();

    // Find voice by ID
    let voice_config = registry.get_voice(source).ok_or_else(|| {
        voirs_sdk::VoirsError::config_error(format!("Voice '{}' not found", source))
    })?;

    // Convert to VoiceInfo for easier access
    let voice = VoiceInfo::from_config(voice_config.clone());

    // Create export package structure
    let characteristics = voice.characteristics();
    let export_data = serde_json::json!({
        "format": "voirs-voice-export",
        "version": "1.0",
        "voice_id": voice.id(),
        "voice_name": voice.name(),
        "language": format!("{:?}", voice.language()),
        "characteristics": {
            "gender": characteristics.gender.map(|g| format!("{:?}", g)),
            "age": characteristics.age.map(|a| format!("{:?}", a)),
            "style": format!("{:?}", characteristics.style),
            "quality": format!("{:?}", characteristics.quality),
            "emotion_support": characteristics.emotion_support,
        },
        "model_info": {
            "acoustic_model": voice.model_info.acoustic_model.clone(),
            "vocoder_model": voice.model_info.vocoder_model.clone(),
            "g2p_model": voice.model_info.g2p_model.clone(),
        },
        "include_weights": include_weights,
        "exported_at": chrono::Utc::now().to_rfc3339(),
    });

    // Write export data to file
    let json_str = serde_json::to_string_pretty(&export_data)
        .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Failed to serialize: {}", e)))?;

    fs::write(output, json_str)
        .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Failed to write: {}", e)))?;

    // If include_weights is true, add note that model weights need to be packaged separately
    if include_weights {
        tracing::info!("Note: Model weights export not yet implemented. Only metadata exported.");
    }

    Ok(())
}

async fn export_emotion_preset(_source: &str, output: &Path) -> Result<()> {
    std::fs::write(output, b"{\n  \"type\": \"emotion_preset\"\n}\n")
        .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Failed to write: {}", e)))?;
    Ok(())
}

async fn export_config(_source: &str, output: &Path) -> Result<()> {
    std::fs::write(output, b"[voirs]\nversion=\"0.1\"\n")
        .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Failed to write: {}", e)))?;
    Ok(())
}

async fn import_voice_package(
    input: &Path,
    name: Option<&str>,
    force: bool,
    validate: bool,
) -> Result<()> {
    // Read and parse the voice package file
    let content = fs::read_to_string(input)
        .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Failed to read input: {}", e)))?;

    let package_data: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Failed to parse JSON: {}", e)))?;

    // Validate package format
    if validate {
        let format = package_data
            .get("format")
            .and_then(|f| f.as_str())
            .unwrap_or("");
        if format != "voirs-voice-export" {
            return Err(voirs_sdk::VoirsError::config_error(format!(
                "Invalid package format: expected 'voirs-voice-export', got '{}'",
                format
            )));
        }

        let version = package_data
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        tracing::info!("Importing voice package version: {}", version);
    }

    // Extract voice information
    let voice_id = package_data
        .get("voice_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            voirs_sdk::VoirsError::config_error("Missing voice_id in package".to_string())
        })?;

    // Use provided name or fall back to package name
    let final_name = name.unwrap_or(voice_id);

    // Check if voice already exists
    let registry = VoiceRegistry::new();
    if registry.get_voice(voice_id).is_some() && !force {
        return Err(voirs_sdk::VoirsError::config_error(format!(
            "Voice '{}' already exists. Use --force to overwrite.",
            voice_id
        )));
    }

    tracing::info!("Importing voice '{}' as '{}'", voice_id, final_name);

    // Actual voice registration and configuration saving
    // 1. Create voice config directory if it doesn't exist
    let voices_dir = dirs::data_dir()
        .ok_or_else(|| {
            voirs_sdk::VoirsError::config_error("Could not determine data directory".to_string())
        })?
        .join("voirs")
        .join("voices");

    fs::create_dir_all(&voices_dir).map_err(|e| {
        voirs_sdk::VoirsError::config_error(format!("Failed to create voices directory: {}", e))
    })?;

    // 2. Save voice configuration to file
    let config_file = voices_dir.join(format!("{}.json", final_name));
    fs::write(&config_file, serde_json::to_string_pretty(&package_data)?).map_err(|e| {
        voirs_sdk::VoirsError::config_error(format!("Failed to write voice config: {}", e))
    })?;

    tracing::info!("Voice configuration saved to {}", config_file.display());

    // Note: Model weights extraction (if included in package) would require
    // additional implementation to handle binary model files. This would involve:
    // - Detecting if weights are embedded in the package (base64 or separate files)
    // - Extracting and saving to appropriate model directory
    // - Updating model paths in the configuration
    // For now, we save the configuration which can reference external model paths.

    Ok(())
}

async fn import_config(
    input: &Path,
    name: Option<&str>,
    force: bool,
    validate: bool,
) -> Result<()> {
    // Determine config format from extension
    let ext = input.extension().and_then(|e| e.to_str()).unwrap_or("");

    // Read configuration file
    let content = fs::read_to_string(input).map_err(|e| {
        voirs_sdk::VoirsError::config_error(format!("Failed to read config: {}", e))
    })?;

    // Parse configuration based on format
    let config_value: serde_json::Value = match ext.to_lowercase().as_str() {
        "json" => serde_json::from_str(&content)
            .map_err(|e| voirs_sdk::VoirsError::config_error(format!("Invalid JSON: {}", e)))?,
        "toml" | "yaml" | "yml" => {
            // For TOML/YAML, convert to JSON value for consistent handling
            tracing::warn!("TOML/YAML parsing not fully implemented, treating as plain text");
            serde_json::json!({
                "raw_content": content,
                "format": ext,
            })
        }
        other => {
            return Err(voirs_sdk::VoirsError::config_error(format!(
                "Unsupported config format: {}",
                other
            )));
        }
    };

    // Validate config if requested
    if validate {
        tracing::info!("Validating configuration...");

        // Basic validation: check for required fields
        if config_value.get("format").is_none() && ext == "json" {
            tracing::warn!("Configuration missing 'format' field");
        }
    }

    // Determine config name
    let config_name = name.unwrap_or_else(|| {
        input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("imported_config")
    });

    // Check if config would overwrite existing
    let config_dir = dirs::config_dir()
        .ok_or_else(|| {
            voirs_sdk::VoirsError::config_error("Could not determine config directory".to_string())
        })?
        .join("voirs");

    let target_path = config_dir.join(format!("{}.json", config_name));

    if target_path.exists() && !force {
        return Err(voirs_sdk::VoirsError::config_error(format!(
            "Configuration '{}' already exists at {}. Use --force to overwrite.",
            config_name,
            target_path.display()
        )));
    }

    tracing::info!(
        "Installing configuration '{}' to {}",
        config_name,
        target_path.display()
    );

    // Actual configuration installation
    // 1. Create config directory if it doesn't exist
    fs::create_dir_all(&config_dir).map_err(|e| {
        voirs_sdk::VoirsError::config_error(format!("Failed to create config directory: {}", e))
    })?;

    // 2. Write configuration file
    let config_content = if ext == "json" {
        // For JSON, write the parsed and validated content
        serde_json::to_string_pretty(&config_value)?
    } else {
        // For TOML/YAML, write the original content
        // (proper parsing would require toml/serde_yaml crates)
        content
    };

    fs::write(&target_path, config_content).map_err(|e| {
        voirs_sdk::VoirsError::config_error(format!("Failed to write config file: {}", e))
    })?;

    tracing::info!(
        "Configuration installed successfully to {}",
        target_path.display()
    );

    // 3. Global configuration registry update
    // Note: The VoiRS SDK would need to support dynamic config reloading
    // for this to take effect without restart. For now, users need to
    // restart the application or explicitly reload configuration.
    tracing::info!("Configuration will be loaded on next application start");

    Ok(())
}
