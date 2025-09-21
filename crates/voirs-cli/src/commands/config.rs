//! Configuration command implementation.

use crate::config::profiles::{ConfigProfile, ProfileManager};
use std::path::Path;
use voirs::{Result, VoirsError};
use voirs_sdk::config::AppConfig;

/// Run config command
pub async fn run_config(
    show: bool,
    init: bool,
    path: Option<&Path>,
    config: &AppConfig,
) -> Result<()> {
    if show {
        // Show current configuration
        let json = serde_json::to_string_pretty(config)
            .map_err(|e| VoirsError::config_error(format!("Failed to serialize config: {}", e)))?;
        println!("{}", json);
    } else if init {
        // Initialize configuration file
        let config_path = path.unwrap_or_else(|| Path::new("voirs.json"));
        // Save configuration to file
        let config_json = serde_json::to_string_pretty(config)
            .map_err(|e| VoirsError::config_error(format!("Failed to serialize config: {}", e)))?;
        std::fs::write(config_path, config_json).map_err(|e| VoirsError::from(e))?;
        println!("Configuration initialized at: {}", config_path.display());
    } else {
        println!("Use --show to display configuration or --init to create default config file");
    }

    Ok(())
}

/// List available configuration profiles
pub async fn run_list_profiles() -> Result<()> {
    let profile_manager =
        ProfileManager::new().map_err(|e| voirs::VoirsError::config_error(e.to_string()))?;

    let profiles = profile_manager
        .list_profiles()
        .map_err(|e| voirs::VoirsError::config_error(e.to_string()))?;

    if profiles.is_empty() {
        println!("No profiles found. Use 'voirs profile create' to create a new profile.");
        return Ok(());
    }

    let current_profile = profile_manager.get_current_profile_name();

    println!("Available profiles:");
    println!();

    for profile in profiles {
        let current_indicator = if current_profile == Some(&profile.name) {
            " (current)"
        } else {
            ""
        };
        let system_indicator = if profile.system { " [system]" } else { "" };

        println!(
            "  {}{}{}",
            profile.name, current_indicator, system_indicator
        );

        if let Some(description) = &profile.description {
            println!("    Description: {}", description);
        }

        if !profile.tags.is_empty() {
            println!("    Tags: {}", profile.tags.join(", "));
        }

        println!(
            "    Created: {}",
            profile.created_at.format("%Y-%m-%d %H:%M:%S UTC")
        );
        println!();
    }

    Ok(())
}

/// Show information about a specific profile
pub async fn run_profile_info(name: &str) -> Result<()> {
    let profile_manager =
        ProfileManager::new().map_err(|e| voirs::VoirsError::config_error(e.to_string()))?;

    let profile = profile_manager
        .load_profile(name)
        .map_err(|e| voirs::VoirsError::config_error(e.to_string()))?;

    println!("Profile: {}", profile.info.name);
    println!("========{}", "=".repeat(profile.info.name.len()));

    if let Some(description) = &profile.info.description {
        println!("Description: {}", description);
    }

    println!(
        "Created: {}",
        profile.info.created_at.format("%Y-%m-%d %H:%M:%S UTC")
    );
    println!(
        "Modified: {}",
        profile.info.modified_at.format("%Y-%m-%d %H:%M:%S UTC")
    );

    if !profile.info.tags.is_empty() {
        println!("Tags: {}", profile.info.tags.join(", "));
    }

    if profile.info.system {
        println!("Type: System profile");
    } else {
        println!("Type: User profile");
    }

    println!();
    println!("Configuration:");
    println!("--------------");
    println!(
        "Default format: {}",
        profile.config.cli.default_output_format
    );
    println!("Default quality: {}", profile.config.cli.default_quality);

    if let Some(voice) = &profile.config.cli.default_voice {
        println!("Default voice: {}", voice);
    }

    println!("Colored output: {}", profile.config.cli.colored_output);
    println!("Show progress: {}", profile.config.cli.show_progress);
    println!("Auto-play: {}", profile.config.cli.auto_play);

    if let Some(output_dir) = &profile.config.cli.output_directory {
        println!("Output directory: {}", output_dir.display());
    }

    Ok(())
}

/// Switch to a different profile
pub async fn run_switch_profile(name: &str) -> Result<()> {
    let mut profile_manager =
        ProfileManager::new().map_err(|e| voirs::VoirsError::config_error(e.to_string()))?;

    profile_manager
        .switch_profile(name)
        .map_err(|e| voirs::VoirsError::config_error(e.to_string()))?;

    println!("Switched to profile: {}", name);

    Ok(())
}

/// Create a new profile
pub async fn run_create_profile(
    name: &str,
    description: Option<String>,
    copy_from: Option<String>,
) -> Result<()> {
    let mut profile_manager =
        ProfileManager::new().map_err(|e| voirs::VoirsError::config_error(e.to_string()))?;

    if let Some(source) = copy_from {
        // Copy from existing profile
        profile_manager
            .copy_profile(&source, name, description)
            .map_err(|e| voirs::VoirsError::config_error(e.to_string()))?;

        println!("Created profile '{}' copied from '{}'", name, source);
    } else {
        // Create new profile with default configuration
        let config = crate::config::CliConfig::default();
        profile_manager
            .create_profile(name, description, config)
            .map_err(|e| voirs::VoirsError::config_error(e.to_string()))?;

        println!("Created profile '{}'", name);
    }

    Ok(())
}

/// Delete a profile
pub async fn run_delete_profile(name: &str) -> Result<()> {
    let mut profile_manager =
        ProfileManager::new().map_err(|e| voirs::VoirsError::config_error(e.to_string()))?;

    profile_manager
        .delete_profile(name)
        .map_err(|e| voirs::VoirsError::config_error(e.to_string()))?;

    println!("Deleted profile: {}", name);

    Ok(())
}

/// Export a profile to a file
pub async fn run_export_profile(name: &str, path: &Path) -> Result<()> {
    let profile_manager =
        ProfileManager::new().map_err(|e| voirs::VoirsError::config_error(e.to_string()))?;

    profile_manager
        .export_profile(name, path)
        .map_err(|e| voirs::VoirsError::config_error(e.to_string()))?;

    println!("Exported profile '{}' to: {}", name, path.display());

    Ok(())
}

/// Import a profile from a file
pub async fn run_import_profile(path: &Path, name: Option<&str>) -> Result<()> {
    let mut profile_manager =
        ProfileManager::new().map_err(|e| voirs::VoirsError::config_error(e.to_string()))?;

    profile_manager
        .import_profile(path, name)
        .map_err(|e| voirs::VoirsError::config_error(e.to_string()))?;

    let imported_name = name.unwrap_or("(from file)");
    println!(
        "Imported profile '{}' from: {}",
        imported_name,
        path.display()
    );

    Ok(())
}
