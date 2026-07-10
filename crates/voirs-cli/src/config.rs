//! CLI-specific configuration utilities.

pub mod profiles;

use crate::error::{CliError, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use voirs_sdk::config::AppConfig;

/// CLI-specific configuration
///
/// `#[serde(default)]` lets this type deserialize successfully from an
/// incomplete/legacy document (e.g. one written by an older VoiRS version
/// that didn't yet have the `cli` section, or that predates a field added
/// later): any field missing from the input falls back to the corresponding
/// field of [`CliConfig::default()`] instead of failing the whole parse.
/// See [`utils::migrate_config`] for the primary consumer of this leniency.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CliConfig {
    /// Core VoiRS configuration.
    ///
    /// Serialized under its own `[core]` table (`core.pipeline`, `core.cli`,
    /// `core.server`, and `core.environment`). It is intentionally *not*
    /// `#[serde(flatten)]`ed to the top level: [`AppConfig`] has its own
    /// `cli` field (a different type than [`CliSettings`]), which under
    /// flattening would collide with this struct's own `cli` field and emit
    /// two `[cli]` tables -- producing invalid TOML that cannot be parsed
    /// back. Keeping it nested makes the whole struct round-trip cleanly.
    pub core: AppConfig,

    /// CLI-specific settings.
    ///
    /// Serialized as the single top-level `[cli]` table.
    pub cli: CliSettings,
}

/// Alias for compatibility with interactive modules
pub type Config = CliConfig;

/// CLI-specific settings
///
/// `#[serde(default)]`: see [`CliConfig`] for why partial input is accepted
/// here (each missing field falls back to [`CliSettings::default()`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CliSettings {
    /// Default output format
    pub default_output_format: String,

    /// Default voice
    pub default_voice: Option<String>,

    /// Default quality level
    pub default_quality: String,

    /// Enable colored output
    pub colored_output: bool,

    /// Show progress bars
    pub show_progress: bool,

    /// Auto-play synthesized audio
    pub auto_play: bool,

    /// Preferred output directory
    pub output_directory: Option<PathBuf>,

    /// SSML validation level
    pub ssml_validation: SsmlValidationLevel,

    /// Recent files history size
    pub history_size: usize,

    /// Voice download preferences
    pub download: DownloadSettings,
}

/// SSML validation levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SsmlValidationLevel {
    /// No validation
    None,
    /// Warn on issues
    Warn,
    /// Error on issues
    Strict,
}

/// Download settings
///
/// `#[serde(default)]`: see [`CliConfig`] for why partial input is accepted
/// here (each missing field falls back to [`DownloadSettings::default()`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DownloadSettings {
    /// Parallel downloads
    pub parallel_downloads: usize,

    /// Retry attempts
    pub retry_attempts: usize,

    /// Auto-verify checksums
    pub verify_checksums: bool,

    /// Preferred download mirrors
    pub preferred_mirrors: Vec<String>,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            core: AppConfig::default(),
            cli: CliSettings::default(),
        }
    }
}

impl Default for CliSettings {
    fn default() -> Self {
        Self {
            default_output_format: "wav".to_string(),
            default_voice: None,
            default_quality: "high".to_string(),
            colored_output: true,
            show_progress: true,
            auto_play: false,
            output_directory: None,
            ssml_validation: SsmlValidationLevel::Warn,
            history_size: 100,
            download: DownloadSettings::default(),
        }
    }
}

impl Default for DownloadSettings {
    fn default() -> Self {
        Self {
            parallel_downloads: 3,
            retry_attempts: 3,
            verify_checksums: true,
            preferred_mirrors: vec![
                "https://huggingface.co".to_string(),
                "https://github.com".to_string(),
            ],
        }
    }
}

/// Configuration manager for the CLI
pub struct ConfigManager {
    config_path: PathBuf,
    config: CliConfig,
}

impl ConfigManager {
    /// Create a new configuration manager
    pub fn new() -> Result<Self> {
        let config_path = Self::find_config_file().unwrap_or_else(Self::default_config_path);

        let config = if config_path.exists() {
            Self::load_from_file(&config_path)?
        } else {
            CliConfig::default()
        };

        Ok(Self {
            config_path,
            config,
        })
    }

    /// Create configuration manager with specific path
    pub fn with_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config_path = path.as_ref().to_path_buf();

        let config = if config_path.exists() {
            Self::load_from_file(&config_path)?
        } else {
            CliConfig::default()
        };

        Ok(Self {
            config_path,
            config,
        })
    }

    /// Get the current configuration
    pub fn config(&self) -> &CliConfig {
        &self.config
    }

    /// Get mutable reference to configuration
    pub fn config_mut(&mut self) -> &mut CliConfig {
        &mut self.config
    }

    /// Save configuration to file
    pub fn save(&self) -> Result<()> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = self.config_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                CliError::file_operation("create directory", &parent.display().to_string(), e)
            })?;
        }

        let content = toml::to_string_pretty(&self.config).map_err(CliError::from)?;

        fs::write(&self.config_path, content).map_err(|e| {
            CliError::file_operation("write", &self.config_path.display().to_string(), e)
        })?;

        Ok(())
    }

    /// Update configuration value
    pub fn set_value(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "default_output_format" => {
                self.config.cli.default_output_format = value.to_string();
            }
            "default_voice" => {
                self.config.cli.default_voice = if value.is_empty() {
                    None
                } else {
                    Some(value.to_string())
                };
            }
            "default_quality" => {
                if ["low", "medium", "high", "ultra"].contains(&value) {
                    self.config.cli.default_quality = value.to_string();
                } else {
                    return Err(CliError::invalid_parameter(
                        key,
                        "must be one of: low, medium, high, ultra",
                    ));
                }
            }
            "colored_output" => {
                self.config.cli.colored_output = value
                    .parse()
                    .map_err(|_| CliError::invalid_parameter(key, "must be true or false"))?;
            }
            "show_progress" => {
                self.config.cli.show_progress = value
                    .parse()
                    .map_err(|_| CliError::invalid_parameter(key, "must be true or false"))?;
            }
            "auto_play" => {
                self.config.cli.auto_play = value
                    .parse()
                    .map_err(|_| CliError::invalid_parameter(key, "must be true or false"))?;
            }
            "output_directory" => {
                self.config.cli.output_directory = if value.is_empty() {
                    None
                } else {
                    Some(PathBuf::from(value))
                };
            }
            _ => {
                return Err(CliError::invalid_parameter(
                    key,
                    "unknown configuration key",
                ));
            }
        }

        Ok(())
    }

    /// Get configuration value as string
    pub fn get_value(&self, key: &str) -> Option<String> {
        match key {
            "default_output_format" => Some(self.config.cli.default_output_format.clone()),
            "default_voice" => self.config.cli.default_voice.clone(),
            "default_quality" => Some(self.config.cli.default_quality.clone()),
            "colored_output" => Some(self.config.cli.colored_output.to_string()),
            "show_progress" => Some(self.config.cli.show_progress.to_string()),
            "auto_play" => Some(self.config.cli.auto_play.to_string()),
            "output_directory" => self
                .config
                .cli
                .output_directory
                .as_ref()
                .map(|p| p.display().to_string()),
            _ => None,
        }
    }

    /// Apply environment variable overrides
    pub fn apply_env_overrides(&mut self) {
        if let Ok(format) = env::var("VOIRS_OUTPUT_FORMAT") {
            self.config.cli.default_output_format = format;
        }

        if let Ok(voice) = env::var("VOIRS_DEFAULT_VOICE") {
            self.config.cli.default_voice = Some(voice);
        }

        if let Ok(quality) = env::var("VOIRS_QUALITY") {
            if ["low", "medium", "high", "ultra"].contains(&quality.as_str()) {
                self.config.cli.default_quality = quality;
            }
        }

        if let Ok(colored) = env::var("VOIRS_COLORED_OUTPUT") {
            if let Ok(value) = colored.parse() {
                self.config.cli.colored_output = value;
            }
        }

        if let Ok(progress) = env::var("VOIRS_SHOW_PROGRESS") {
            if let Ok(value) = progress.parse() {
                self.config.cli.show_progress = value;
            }
        }

        if let Ok(output_dir) = env::var("VOIRS_OUTPUT_DIR") {
            self.config.cli.output_directory = Some(PathBuf::from(output_dir));
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<Vec<String>> {
        let mut warnings = Vec::new();

        // Check if default voice exists
        if let Some(ref voice) = self.config.cli.default_voice {
            // Validate voice existence by checking if it can be resolved
            match self.validate_voice_exists(voice) {
                Ok(true) => {
                    // Voice exists, no warning needed
                }
                Ok(false) => {
                    warnings.push(format!(
                        "Default voice '{}' does not exist. Use 'voirs voices list' to see available voices.",
                        voice
                    ));
                }
                Err(_) => {
                    // Could not validate (e.g., voice system not initialized)
                    // This is not critical, just note it as a warning
                    warnings.push(format!(
                        "Could not verify existence of default voice '{}'. Voice system may not be initialized.",
                        voice
                    ));
                }
            }
        }

        // Check output directory
        if let Some(ref output_dir) = self.config.cli.output_directory {
            if !output_dir.exists() {
                warnings.push(format!(
                    "Output directory '{}' does not exist",
                    output_dir.display()
                ));
            } else if !output_dir.is_dir() {
                return Err(CliError::config(format!(
                    "Output directory '{}' is not a directory",
                    output_dir.display()
                )));
            }
        }

        // Validate download settings
        if self.config.cli.download.parallel_downloads == 0 {
            return Err(CliError::config(
                "parallel_downloads must be greater than 0",
            ));
        }

        if self.config.cli.download.parallel_downloads > 10 {
            warnings.push("parallel_downloads > 10 may cause server rate limiting".to_string());
        }

        Ok(warnings)
    }

    /// Validate if a voice exists in the system
    fn validate_voice_exists(&self, voice_id: &str) -> Result<bool> {
        // Try to check if voice exists by looking in standard voice directories
        // This is a lightweight check that doesn't require initializing the full pipeline

        // Get potential voice directories
        let voice_dirs = self.get_voice_directories();

        for voice_dir in voice_dirs {
            let voice_config_path = voice_dir.join(voice_id).join("voice.json");
            if voice_config_path.exists() {
                // Found the voice config file
                return Ok(true);
            }
        }

        // Voice not found in standard directories
        Ok(false)
    }

    /// Get list of directories where voices might be stored
    fn get_voice_directories(&self) -> Vec<PathBuf> {
        let mut dirs = Vec::new();

        // 1. Check default data directory (~/.voirs/voices)
        if let Some(home) = dirs::home_dir() {
            dirs.push(home.join(".voirs").join("voices"));
        }

        // 3. Check XDG data directory on Linux
        #[cfg(target_os = "linux")]
        {
            if let Ok(xdg_data_home) = std::env::var("XDG_DATA_HOME") {
                dirs.push(PathBuf::from(xdg_data_home).join("voirs").join("voices"));
            } else if let Some(home) = dirs::home_dir() {
                dirs.push(
                    home.join(".local")
                        .join("share")
                        .join("voirs")
                        .join("voices"),
                );
            }
        }

        // 4. Check Library directory on macOS
        #[cfg(target_os = "macos")]
        {
            if let Some(home) = dirs::home_dir() {
                dirs.push(
                    home.join("Library")
                        .join("Application Support")
                        .join("voirs")
                        .join("voices"),
                );
            }
        }

        // 5. Check AppData on Windows
        #[cfg(target_os = "windows")]
        {
            if let Ok(appdata) = std::env::var("APPDATA") {
                dirs.push(PathBuf::from(appdata).join("voirs").join("voices"));
            }
        }

        // 6. Check current directory (for development/testing)
        dirs.push(PathBuf::from("./voices"));

        dirs
    }

    /// Get configuration path
    pub fn config_path(&self) -> &Path {
        &self.config_path
    }

    /// Load configuration from file
    fn load_from_file<P: AsRef<Path>>(path: P) -> Result<CliConfig> {
        let content = fs::read_to_string(path.as_ref()).map_err(|e| {
            CliError::file_operation("read", &path.as_ref().display().to_string(), e)
        })?;

        // Try TOML first, then JSON for backward compatibility
        if let Ok(config) = toml::from_str::<CliConfig>(&content) {
            Ok(config)
        } else {
            serde_json::from_str::<CliConfig>(&content)
                .map_err(|e| CliError::config(format!("Invalid configuration format: {}", e)))
        }
    }

    /// Find configuration file in standard locations
    fn find_config_file() -> Option<PathBuf> {
        let possible_paths = [
            env::current_dir().ok().map(|d| d.join("voirs.toml")),
            env::current_dir().ok().map(|d| d.join("voirs.json")),
            Self::config_dir().map(|d| d.join("voirs.toml")),
            Self::config_dir().map(|d| d.join("voirs.json")),
            env::var("VOIRS_CONFIG").ok().map(PathBuf::from),
        ];

        possible_paths
            .into_iter()
            .flatten()
            .find(|path| path.exists())
    }

    /// Get default configuration path
    fn default_config_path() -> PathBuf {
        Self::config_dir()
            .unwrap_or_else(|| env::current_dir().expect("current dir should be accessible"))
            .join("voirs.toml")
    }

    /// Get configuration directory
    fn config_dir() -> Option<PathBuf> {
        if let Some(config_dir) = env::var_os("XDG_CONFIG_HOME") {
            Some(PathBuf::from(config_dir).join("voirs"))
        } else if let Some(home_dir) = env::var_os("HOME") {
            Some(PathBuf::from(home_dir).join(".config").join("voirs"))
        } else {
            env::var_os("APPDATA").map(|app_data| PathBuf::from(app_data).join("voirs"))
        }
    }
}

/// Configuration utilities
pub mod utils {
    use super::*;

    /// Create a default configuration file
    pub fn create_default_config<P: AsRef<Path>>(path: P) -> Result<()> {
        let config = CliConfig::default();
        let content = toml::to_string_pretty(&config).map_err(CliError::from)?;

        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent).map_err(|e| {
                CliError::file_operation("create directory", &parent.display().to_string(), e)
            })?;
        }

        fs::write(path.as_ref(), content).map_err(|e| {
            CliError::file_operation("write", &path.as_ref().display().to_string(), e)
        })?;

        Ok(())
    }

    /// Migrate old configuration format to new format
    ///
    /// The old (JSON) configuration is deserialized directly into
    /// [`CliConfig`] so that every field whose name and nesting still match
    /// the current schema survives the migration unchanged; unknown legacy
    /// top-level keys are ignored, and fields that are missing from the old
    /// document (e.g. because they were added in a later version) fall back
    /// to their [`CliConfig::default()`] value thanks to the
    /// `#[serde(default)]` attributes on [`CliConfig`]/[`CliSettings`]/
    /// [`DownloadSettings`]. The one known legacy rename -- a top-level
    /// `"output_format"` string that predates today's nested
    /// `cli.default_output_format` field -- is migrated explicitly on top of
    /// that, and always wins.
    ///
    /// Because [`CliConfig::core`] is a plain nested field (no longer
    /// `#[serde(flatten)]`ed), the whole-document deserialize succeeds for
    /// well-formed input: `core` ([`AppConfig`], from `voirs_sdk`) simply
    /// defaults when absent from an old document, and this crate's `cli`
    /// ([`CliSettings`]) parses from the old `"cli"` section. A defensive
    /// fallback still recovers the `core` and `cli` sections independently if
    /// that whole-document parse ever fails -- for instance because a
    /// hand-edited document gives a present field an incompatible type -- so
    /// one malformed field cannot discard everything else.
    pub fn migrate_config<P: AsRef<Path>>(old_path: P, new_path: P) -> Result<()> {
        let old_content = fs::read_to_string(old_path.as_ref()).map_err(|e| {
            CliError::file_operation("read", &old_path.as_ref().display().to_string(), e)
        })?;

        // Parse the raw JSON so we can also inspect legacy top-level keys
        // that no longer exist anywhere in the current `CliConfig` schema.
        let raw_old_config: serde_json::Value = serde_json::from_str(&old_content)
            .map_err(|e| CliError::config(format!("Cannot parse old config: {}", e)))?;

        // Deserialize the same document directly into `CliConfig`. Every
        // field whose name/nesting is unchanged is preserved as-is; unknown
        // legacy top-level keys are ignored and anything missing falls back
        // to `CliConfig::default()` (see the `#[serde(default)]` attributes
        // on the types involved). This replaces the old behavior of silently
        // discarding every field except `output_format`.
        let mut new_config: CliConfig = match serde_json::from_value(raw_old_config.clone()) {
            Ok(config) => config,
            Err(_) => {
                // Defensive fallback for a document the whole-struct parse
                // cannot handle (e.g. a hand-edited file where a present
                // field has an incompatible type): recover the `core` and
                // `cli` sections on their own so one bad field can't discard
                // the rest.
                let mut fallback = CliConfig::default();

                // `core` (AppConfig: pipeline/cli/server/environment) can be
                // recovered independently of the `cli` section.
                if let Ok(core) = serde_json::from_value(raw_old_config.clone()) {
                    fallback.core = core;
                }

                // Every field this function is responsible for preserving
                // lives under "cli"; migrate that subtree on its own.
                if let Some(cli_value) = raw_old_config.get("cli") {
                    if let Ok(cli_settings) =
                        serde_json::from_value::<CliSettings>(cli_value.clone())
                    {
                        fallback.cli = cli_settings;
                    }
                }

                fallback
            }
        };

        // Legacy rename: very old configs stored the output format as a
        // top-level "output_format" string instead of today's
        // "cli.default_output_format". Apply it last so it always wins over
        // whatever `cli.default_output_format` picked up above.
        if let Some(format_str) = raw_old_config
            .get("output_format")
            .and_then(|value| value.as_str())
        {
            new_config.cli.default_output_format = format_str.to_string();
        }

        // Save migrated config
        let content = toml::to_string_pretty(&new_config).map_err(CliError::from)?;

        fs::write(new_path.as_ref(), content).map_err(|e| {
            CliError::file_operation("write", &new_path.as_ref().display().to_string(), e)
        })?;

        Ok(())
    }

    /// Export configuration for sharing
    pub fn export_config<P: AsRef<Path>>(
        config: &CliConfig,
        path: P,
        format: ConfigFormat,
    ) -> Result<()> {
        let content = match format {
            ConfigFormat::Toml => toml::to_string_pretty(config)?,
            ConfigFormat::Json => serde_json::to_string_pretty(config)?,
            ConfigFormat::Yaml => serde_yaml::to_string(config)
                .map_err(|e| CliError::config(format!("YAML serialization error: {}", e)))?,
        };

        fs::write(path.as_ref(), content).map_err(|e| {
            CliError::file_operation("write", &path.as_ref().display().to_string(), e)
        })?;

        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_migrate_config_preserves_cli_fields_and_legacy_rename() {
            let dir = std::env::temp_dir().join(format!(
                "voirs_migrate_config_test_{}_{}",
                std::process::id(),
                fastrand::u64(..)
            ));
            fs::create_dir_all(&dir).expect("temp dir should be creatable");

            let old_path = dir.join("old_config.json");
            let new_path = dir.join("new_config.toml");

            // Old-format document: a legacy top-level "output_format" (the
            // one known rename) alongside a "cli" section whose field names
            // already match today's `CliSettings` schema.
            let old_json = serde_json::json!({
                "output_format": "mp3",
                "cli": {
                    "default_output_format": "wav",
                    "default_voice": "test-voice",
                    "default_quality": "ultra",
                    "colored_output": false,
                    "download": {
                        "parallel_downloads": 7
                    }
                }
            });
            fs::write(
                &old_path,
                serde_json::to_string_pretty(&old_json).expect("serialize fixture"),
            )
            .expect("write old config fixture");

            migrate_config(old_path.clone(), new_path.clone())
                .expect("migrate_config should succeed");

            let migrated_content = fs::read_to_string(&new_path).expect("read migrated config");

            // With `CliConfig::core` no longer `#[serde(flatten)]`ed, the
            // migrated document is valid, round-trippable TOML: `AppConfig`
            // now lives under its own `[core]` table (`core.pipeline`,
            // `core.cli`, `core.server`, ...) and this crate's `CliSettings`
            // is the single top-level `[cli]` table -- no more colliding
            // `[cli]` sections. First inspect the `[cli]` table on its own.
            let migrated_value: toml::Value =
                toml::from_str(&migrated_content).expect("migrated config should be valid TOML");
            let cli_table = migrated_value
                .get("cli")
                .cloned()
                .expect("migrated config should have a 'cli' table");
            // `toml::Value` implements `serde::Deserializer`, so go through
            // the trait method directly (version-robust across toml releases)
            // rather than an inherent `try_into`.
            let migrated_cli: CliSettings = CliSettings::deserialize(cli_table)
                .expect("'cli' table should deserialize into CliSettings");

            // The legacy rename wins over the (still-present) nested value.
            assert_eq!(migrated_cli.default_output_format, "mp3");
            // Fields untouched by the rename survive from the old document
            // instead of silently reverting to `CliConfig::default()`.
            assert_eq!(migrated_cli.default_voice, Some("test-voice".to_string()));
            assert_eq!(migrated_cli.default_quality, "ultra");
            assert!(!migrated_cli.colored_output);
            assert_eq!(migrated_cli.download.parallel_downloads, 7);

            // Sanity check against the previous (buggy) behavior: these
            // values must differ from a fresh default, or this test would
            // pass vacuously even if migration silently discarded everything.
            let defaults = CliSettings::default();
            assert_ne!(migrated_cli.default_voice, defaults.default_voice);
            assert_ne!(migrated_cli.default_quality, defaults.default_quality);
            assert_ne!(migrated_cli.colored_output, defaults.colored_output);
            assert_ne!(
                migrated_cli.download.parallel_downloads,
                defaults.download.parallel_downloads
            );

            // Now that the `[cli]` collision is gone, the whole migrated
            // document also deserializes straight into `CliConfig` -- proving
            // the nested `[core]` (`AppConfig`) table round-trips too, not
            // just the `[cli]` section inspected above.
            let round_tripped: CliConfig = toml::from_str(&migrated_content)
                .expect("migrated config should deserialize into CliConfig");
            assert_eq!(round_tripped.cli.default_output_format, "mp3");
            assert_eq!(
                round_tripped.cli.default_voice,
                Some("test-voice".to_string())
            );
            assert_eq!(round_tripped.cli.default_quality, "ultra");
            assert!(!round_tripped.cli.colored_output);
            assert_eq!(round_tripped.cli.download.parallel_downloads, 7);

            let _ = fs::remove_dir_all(&dir);
        }
    }
}

/// Enhanced configuration loader with performance optimizations
pub struct EnhancedConfigLoader {
    cache: Option<(PathBuf, std::time::SystemTime, CliConfig)>,
}

impl EnhancedConfigLoader {
    /// Create a new enhanced configuration loader
    pub fn new() -> Self {
        Self { cache: None }
    }

    /// Load configuration with caching and smart format detection
    pub fn load_config<P: AsRef<Path>>(&mut self, path: P) -> Result<CliConfig> {
        let path = path.as_ref();
        let start_time = Instant::now();

        // Check cache validity
        if let Some((cached_path, cached_time, ref cached_config)) = &self.cache {
            if cached_path == path {
                if let Ok(metadata) = fs::metadata(path) {
                    if let Ok(modified) = metadata.modified() {
                        if modified <= *cached_time {
                            // Cache hit - return cached config
                            return Ok(cached_config.clone());
                        }
                    }
                }
            }
        }

        // Cache miss - load from file
        let content = fs::read_to_string(path)
            .map_err(|e| CliError::file_operation("read", &path.display().to_string(), e))?;

        let config = self.parse_config_content(&content, path)?;

        // Update cache
        if let Ok(metadata) = fs::metadata(path) {
            if let Ok(modified) = metadata.modified() {
                self.cache = Some((path.to_path_buf(), modified, config.clone()));
            }
        }

        let load_time = start_time.elapsed();
        if load_time > Duration::from_millis(100) {
            eprintln!("Warning: Configuration loading took {:?}", load_time);
        }

        Ok(config)
    }

    /// Parse configuration content with smart format detection
    fn parse_config_content<P: AsRef<Path>>(&self, content: &str, path: P) -> Result<CliConfig> {
        let extension = path.as_ref().extension().and_then(|ext| ext.to_str());

        // Try format detection based on file extension first
        match extension {
            Some("toml") => {
                match toml::from_str::<CliConfig>(content) {
                    Ok(config) => return Ok(config),
                    Err(e) => {
                        // If TOML parsing fails, try fallback formats
                        eprintln!("TOML parsing failed: {}, trying fallback formats", e);
                    }
                }
            }
            Some("json") => match serde_json::from_str::<CliConfig>(content) {
                Ok(config) => return Ok(config),
                Err(e) => {
                    eprintln!("JSON parsing failed: {}, trying fallback formats", e);
                }
            },
            Some("yaml") | Some("yml") => match serde_yaml::from_str::<CliConfig>(content) {
                Ok(config) => return Ok(config),
                Err(e) => {
                    eprintln!("YAML parsing failed: {}, trying fallback formats", e);
                }
            },
            _ => {
                // No extension or unknown extension - try content-based detection
            }
        }

        // Content-based format detection with smart heuristics
        let trimmed = content.trim();

        // Try TOML first (most common format)
        if !trimmed.starts_with('{') && !trimmed.starts_with('[') {
            if let Ok(config) = toml::from_str::<CliConfig>(content) {
                return Ok(config);
            }
        }

        // Try JSON if content looks like JSON
        if trimmed.starts_with('{') && trimmed.ends_with('}') {
            if let Ok(config) = serde_json::from_str::<CliConfig>(content) {
                return Ok(config);
            }
        }

        // Try YAML as last resort
        if let Ok(config) = serde_yaml::from_str::<CliConfig>(content) {
            return Ok(config);
        }

        Err(CliError::config(format!(
            "Unable to parse configuration file '{}' - tried TOML, JSON, and YAML formats",
            path.as_ref().display()
        )))
    }

    /// Clear the configuration cache
    pub fn clear_cache(&mut self) {
        self.cache = None;
    }

    /// Check if configuration is cached
    pub fn is_cached<P: AsRef<Path>>(&self, path: P) -> bool {
        if let Some((cached_path, _, _)) = &self.cache {
            cached_path == path.as_ref()
        } else {
            false
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> Option<(PathBuf, std::time::SystemTime)> {
        self.cache
            .as_ref()
            .map(|(path, time, _)| (path.clone(), *time))
    }
}

impl Default for EnhancedConfigLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration validation utilities
pub mod validation {
    use super::*;
    use std::time::{Duration, Instant};

    /// Validate configuration with detailed reporting
    pub fn validate_config_detailed(config: &CliConfig) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();
        let start_time = Instant::now();

        // Validate CLI settings
        validate_cli_settings(&config.cli, &mut report)?;

        // Validate core configuration
        validate_core_config(&config.core, &mut report)?;

        // Performance check
        let validation_time = start_time.elapsed();
        if validation_time > Duration::from_millis(50) {
            report.add_warning(format!(
                "Configuration validation took {:?} - consider optimizing",
                validation_time
            ));
        }

        Ok(report)
    }

    /// Validate CLI-specific settings
    fn validate_cli_settings(settings: &CliSettings, report: &mut ValidationReport) -> Result<()> {
        // Validate output format
        let valid_formats = ["wav", "mp3", "flac", "ogg", "m4a"];
        if !valid_formats.contains(&settings.default_output_format.as_str()) {
            report.add_error(format!(
                "Invalid default output format '{}'. Valid formats: {}",
                settings.default_output_format,
                valid_formats.join(", ")
            ));
        }

        // Validate quality level
        let valid_qualities = ["low", "medium", "high", "ultra"];
        if !valid_qualities.contains(&settings.default_quality.as_str()) {
            report.add_error(format!(
                "Invalid default quality '{}'. Valid qualities: {}",
                settings.default_quality,
                valid_qualities.join(", ")
            ));
        }

        // Validate output directory
        if let Some(ref output_dir) = settings.output_directory {
            if !output_dir.exists() {
                report.add_warning(format!(
                    "Output directory '{}' does not exist",
                    output_dir.display()
                ));
            } else if !output_dir.is_dir() {
                report.add_error(format!(
                    "Output directory '{}' is not a directory",
                    output_dir.display()
                ));
            }
        }

        // Validate download settings
        if settings.download.parallel_downloads == 0 {
            report.add_error("parallel_downloads must be greater than 0".to_string());
        } else if settings.download.parallel_downloads > 20 {
            report.add_warning(format!(
                "parallel_downloads ({}) is very high and may cause issues",
                settings.download.parallel_downloads
            ));
        }

        if settings.download.retry_attempts > 10 {
            report.add_warning(format!(
                "retry_attempts ({}) is very high and may cause long delays",
                settings.download.retry_attempts
            ));
        }

        Ok(())
    }

    /// Validate core configuration
    fn validate_core_config(config: &AppConfig, report: &mut ValidationReport) -> Result<()> {
        // Validate device configuration
        match config.pipeline.device.as_str() {
            "cpu" => {
                report.add_info("Using CPU device - synthesis will be slower than GPU".to_string());
            }
            "gpu" | "cuda" => {
                report.add_info("GPU acceleration enabled - ensure CUDA is available".to_string());
                #[cfg(not(feature = "cuda"))]
                report.add_warning(
                    "GPU device specified but CUDA feature not enabled in build".to_string(),
                );
            }
            "metal" => {
                report.add_info("Metal acceleration enabled - macOS only".to_string());
                #[cfg(not(target_os = "macos"))]
                report.add_error("Metal device is only available on macOS".to_string());
                #[cfg(not(feature = "metal"))]
                report.add_warning(
                    "Metal device specified but metal feature not enabled in build".to_string(),
                );
            }
            other => {
                report.add_error(format!(
                    "Invalid device '{}' - must be 'cpu', 'gpu', 'cuda', or 'metal'",
                    other
                ));
            }
        }

        // Validate threads configuration
        if let Some(threads) = config.pipeline.num_threads {
            if threads == 0 {
                report.add_error("num_threads must be greater than 0".to_string());
            } else if threads > num_cpus::get() * 2 {
                report.add_warning(format!(
                    "num_threads ({}) exceeds 2x CPU count ({}) - may cause overhead",
                    threads,
                    num_cpus::get()
                ));
            }
        }

        // Validate sample rate from default synthesis config
        let sample_rate = config.pipeline.default_synthesis.sample_rate;
        match sample_rate {
            8000 | 16000 | 22050 | 24000 | 32000 | 44100 | 48000 => {
                // Standard sample rates are ok
            }
            rate if rate < 8000 => {
                report.add_error(format!("sample_rate {} is too low - minimum 8000 Hz", rate));
            }
            rate if rate > 48000 => {
                report.add_warning(format!(
                    "sample_rate {} is very high - may increase processing time",
                    rate
                ));
            }
            rate => {
                report.add_warning(format!(
                    "non-standard sample_rate {} - common rates: 16000, 22050, 44100, 48000",
                    rate
                ));
            }
        }

        // Check cache directory if specified
        if let Some(cache_dir) = &config.pipeline.cache_dir {
            if !cache_dir.exists() {
                report.add_warning(format!(
                    "cache directory does not exist: {}",
                    cache_dir.display()
                ));
            } else if !cache_dir.is_dir() {
                report.add_error(format!(
                    "cache path exists but is not a directory: {}",
                    cache_dir.display()
                ));
            }
        }

        // Validate cache size
        let max_cache_size_mb = config.pipeline.max_cache_size_mb;
        if max_cache_size_mb == 0 {
            report.add_warning("cache disabled (max_cache_size_mb = 0)".to_string());
        } else if max_cache_size_mb > 10240 {
            report.add_warning(format!(
                "very large cache size ({} MB) may consume excessive memory",
                max_cache_size_mb
            ));
        }

        // Validate GPU usage consistency
        if config.pipeline.use_gpu && config.pipeline.device == "cpu" {
            report.add_warning(
                "use_gpu is true but device is set to 'cpu' - inconsistent configuration"
                    .to_string(),
            );
        }

        Ok(())
    }

    /// Configuration validation report
    #[derive(Debug, Clone)]
    pub struct ValidationReport {
        pub errors: Vec<String>,
        pub warnings: Vec<String>,
        pub info: Vec<String>,
    }

    impl ValidationReport {
        pub fn new() -> Self {
            Self {
                errors: Vec::new(),
                warnings: Vec::new(),
                info: Vec::new(),
            }
        }

        pub fn add_error(&mut self, error: String) {
            self.errors.push(error);
        }

        pub fn add_warning(&mut self, warning: String) {
            self.warnings.push(warning);
        }

        pub fn add_info(&mut self, info: String) {
            self.info.push(info);
        }

        pub fn is_valid(&self) -> bool {
            self.errors.is_empty()
        }

        pub fn has_warnings(&self) -> bool {
            !self.warnings.is_empty()
        }

        pub fn summary(&self) -> String {
            format!(
                "Validation complete: {} errors, {} warnings, {} info messages",
                self.errors.len(),
                self.warnings.len(),
                self.info.len()
            )
        }
    }

    impl Default for ValidationReport {
        fn default() -> Self {
            Self::new()
        }
    }
}

/// Configuration export formats
pub enum ConfigFormat {
    Toml,
    Json,
    Yaml,
}
