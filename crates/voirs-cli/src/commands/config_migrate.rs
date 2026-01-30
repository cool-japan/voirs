//! Configuration migration tool for VoiRS CLI
//!
//! Handles automatic migration of configuration files between versions,
//! ensuring backward compatibility and smooth upgrades.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Configuration version
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ConfigVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl ConfigVersion {
    /// Create a new version
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Current config version
    pub const CURRENT: Self = Self::new(0, 1, 0);

    /// Parse version from string (e.g., "0.1.0")
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return None;
        }

        Some(Self {
            major: parts[0].parse().ok()?,
            minor: parts[1].parse().ok()?,
            patch: parts[2].parse().ok()?,
        })
    }

    /// Check if this version needs migration to target
    pub fn needs_migration(&self, target: &ConfigVersion) -> bool {
        self < target
    }
}

impl std::fmt::Display for ConfigVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Migration operation
#[derive(Debug, Clone)]
pub struct Migration {
    /// Source version
    pub from: ConfigVersion,

    /// Target version
    pub to: ConfigVersion,

    /// Migration function
    pub migrate: fn(&mut serde_json::Value) -> Result<()>,

    /// Description of changes
    pub description: String,
}

/// Migration manager
pub struct MigrationManager {
    migrations: Vec<Migration>,
}

impl MigrationManager {
    /// Create a new migration manager
    pub fn new() -> Self {
        let mut manager = Self {
            migrations: Vec::new(),
        };

        // Register all migrations
        manager.register_migrations();
        manager
    }

    /// Register all available migrations
    fn register_migrations(&mut self) {
        // Example: Migration from 0.0.1 to 0.1.0
        self.add_migration(
            ConfigVersion::new(0, 0, 1),
            ConfigVersion::new(0, 1, 0),
            migrate_0_0_1_to_0_1_0,
            "Add telemetry configuration and update model paths",
        );

        // Add more migrations as needed
    }

    /// Add a migration
    pub fn add_migration(
        &mut self,
        from: ConfigVersion,
        to: ConfigVersion,
        migrate_fn: fn(&mut serde_json::Value) -> Result<()>,
        description: impl Into<String>,
    ) {
        self.migrations.push(Migration {
            from,
            to,
            migrate: migrate_fn,
            description: description.into(),
        });
    }

    /// Get migration path from source to target version
    pub fn get_migration_path(
        &self,
        from: &ConfigVersion,
        to: &ConfigVersion,
    ) -> Result<Vec<&Migration>> {
        let mut path = Vec::new();
        let mut current = *from;

        while current < *to {
            // Find next migration step
            let next = self
                .migrations
                .iter()
                .filter(|m| m.from == current && m.to <= *to)
                .min_by_key(|m| m.to);

            match next {
                Some(migration) => {
                    path.push(migration);
                    current = migration.to;
                }
                None => {
                    anyhow::bail!("No migration path found from {} to {}", from, to);
                }
            }
        }

        Ok(path)
    }

    /// Migrate configuration from one version to another
    pub fn migrate(
        &self,
        config: &mut serde_json::Value,
        from: &ConfigVersion,
        to: &ConfigVersion,
    ) -> Result<()> {
        if from == to {
            return Ok(());
        }

        let path = self.get_migration_path(from, to)?;

        for migration in path {
            eprintln!(
                "Migrating from {} to {}: {}",
                migration.from, migration.to, migration.description
            );

            (migration.migrate)(config).with_context(|| {
                format!(
                    "Failed to migrate from {} to {}",
                    migration.from, migration.to
                )
            })?;
        }

        // Update version field
        if let Some(obj) = config.as_object_mut() {
            obj.insert(
                "version".to_string(),
                serde_json::Value::String(to.to_string()),
            );
        }

        Ok(())
    }
}

impl Default for MigrationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Migrate configuration file
pub async fn migrate_config_file(
    config_path: &Path,
    target_version: Option<ConfigVersion>,
    backup: bool,
    verbose: bool,
) -> Result<()> {
    let target = target_version.unwrap_or(ConfigVersion::CURRENT);

    if verbose {
        eprintln!("Reading configuration from {}...", config_path.display());
    }

    // Read existing config
    let config_str = tokio::fs::read_to_string(config_path)
        .await
        .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;

    let mut config: serde_json::Value = serde_json::from_str(&config_str)
        .with_context(|| format!("Failed to parse config file: {}", config_path.display()))?;

    // Detect current version
    let current_version = detect_version(&config)?;

    if verbose {
        eprintln!("Current config version: {}", current_version);
        eprintln!("Target config version: {}", target);
    }

    if !current_version.needs_migration(&target) {
        println!("Configuration is already up to date ({})!", current_version);
        return Ok(());
    }

    // Create backup if requested
    if backup {
        let backup_path = create_backup(config_path).await?;
        println!("Created backup: {}", backup_path.display());
    }

    // Perform migration
    let manager = MigrationManager::new();
    manager.migrate(&mut config, &current_version, &target)?;

    // Write updated config
    let new_config_str = serde_json::to_string_pretty(&config)?;
    tokio::fs::write(config_path, new_config_str)
        .await
        .with_context(|| format!("Failed to write config file: {}", config_path.display()))?;

    println!(
        "Successfully migrated configuration from {} to {}",
        current_version, target
    );

    Ok(())
}

/// Detect configuration version
fn detect_version(config: &serde_json::Value) -> Result<ConfigVersion> {
    if let Some(version_str) = config.get("version").and_then(|v| v.as_str()) {
        ConfigVersion::parse(version_str)
            .ok_or_else(|| anyhow::anyhow!("Invalid version format: {}", version_str))
    } else {
        // Assume oldest version if no version field
        Ok(ConfigVersion::new(0, 0, 1))
    }
}

/// Create backup of configuration file
async fn create_backup(config_path: &Path) -> Result<PathBuf> {
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let backup_path = config_path.with_extension(format!("toml.backup.{}", timestamp));

    tokio::fs::copy(config_path, &backup_path)
        .await
        .with_context(|| format!("Failed to create backup at {}", backup_path.display()))?;

    Ok(backup_path)
}

/// Restore configuration from backup
pub async fn restore_from_backup(
    config_path: &Path,
    backup_path: &Path,
    verbose: bool,
) -> Result<()> {
    if verbose {
        eprintln!(
            "Restoring configuration from backup: {}",
            backup_path.display()
        );
    }

    tokio::fs::copy(backup_path, config_path)
        .await
        .with_context(|| {
            format!(
                "Failed to restore from backup {} to {}",
                backup_path.display(),
                config_path.display()
            )
        })?;

    println!("Configuration restored successfully");
    Ok(())
}

/// Validate configuration file
pub async fn validate_config(config_path: &Path, verbose: bool) -> Result<()> {
    if verbose {
        eprintln!("Validating configuration: {}", config_path.display());
    }

    let config_str = tokio::fs::read_to_string(config_path)
        .await
        .with_context(|| format!("Failed to read config: {}", config_path.display()))?;

    let config: serde_json::Value =
        serde_json::from_str(&config_str).with_context(|| "Failed to parse configuration")?;

    // Detect version
    let version = detect_version(&config)?;

    if verbose {
        eprintln!("Configuration version: {}", version);
    }

    // Validate required fields
    validate_required_fields(&config)?;

    println!("Configuration is valid!");
    Ok(())
}

/// Validate required configuration fields
fn validate_required_fields(config: &serde_json::Value) -> Result<()> {
    let required_fields = vec!["version"];

    for field in required_fields {
        if config.get(field).is_none() {
            anyhow::bail!("Missing required field: {}", field);
        }
    }

    Ok(())
}

// Migration functions

/// Migrate from 0.0.1 to 0.1.0
fn migrate_0_0_1_to_0_1_0(config: &mut serde_json::Value) -> Result<()> {
    let obj = config
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("Config must be an object"))?;

    // Add telemetry configuration if missing
    if !obj.contains_key("telemetry") {
        let mut telemetry = serde_json::Map::new();
        telemetry.insert("enabled".to_string(), serde_json::Value::Bool(false));
        telemetry.insert(
            "level".to_string(),
            serde_json::Value::String("basic".to_string()),
        );
        obj.insert(
            "telemetry".to_string(),
            serde_json::Value::Object(telemetry),
        );
    }

    // Update model paths to use new structure
    if let Some(models) = obj.get_mut("models") {
        if let Some(models_obj) = models.as_object_mut() {
            // Convert old "path" to new "paths" array
            if let Some(path) = models_obj.remove("path") {
                models_obj.insert("paths".to_string(), serde_json::Value::Array(vec![path]));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parsing() {
        let v = ConfigVersion::parse("0.1.0").unwrap();
        assert_eq!(v.major, 0);
        assert_eq!(v.minor, 1);
        assert_eq!(v.patch, 0);
    }

    #[test]
    fn test_version_comparison() {
        let v1 = ConfigVersion::new(0, 0, 1);
        let v2 = ConfigVersion::new(0, 1, 0);
        assert!(v1 < v2);
        assert!(v1.needs_migration(&v2));
    }

    #[test]
    fn test_version_to_string() {
        let v = ConfigVersion::new(1, 2, 3);
        assert_eq!(v.to_string(), "1.2.3");
    }

    #[test]
    fn test_migration_path() {
        let manager = MigrationManager::new();
        let from = ConfigVersion::new(0, 0, 1);
        let to = ConfigVersion::new(0, 1, 0);

        let path = manager.get_migration_path(&from, &to).unwrap();
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].from, from);
        assert_eq!(path[0].to, to);
    }

    #[test]
    fn test_detect_version() {
        let config = serde_json::json!({
            "version": "0.1.0",
            "models": {}
        });

        let version = detect_version(&config).unwrap();
        assert_eq!(version, ConfigVersion::new(0, 1, 0));
    }

    #[test]
    fn test_detect_version_missing() {
        let config = serde_json::json!({
            "models": {}
        });

        let version = detect_version(&config).unwrap();
        assert_eq!(version, ConfigVersion::new(0, 0, 1));
    }

    #[test]
    fn test_migrate_0_0_1_to_0_1_0() {
        let mut config = serde_json::json!({
            "version": "0.0.1",
            "models": {
                "path": "/path/to/models"
            }
        });

        migrate_0_0_1_to_0_1_0(&mut config).unwrap();

        // Check telemetry was added
        assert!(config.get("telemetry").is_some());
        assert_eq!(
            config["telemetry"]["enabled"],
            serde_json::Value::Bool(false)
        );

        // Check paths conversion
        assert!(config["models"]["paths"].is_array());
        assert_eq!(
            config["models"]["paths"][0],
            serde_json::Value::String("/path/to/models".to_string())
        );
    }

    #[test]
    fn test_validate_required_fields() {
        let config = serde_json::json!({
            "version": "0.1.0"
        });

        assert!(validate_required_fields(&config).is_ok());
    }

    #[test]
    fn test_validate_required_fields_missing() {
        let config = serde_json::json!({
            "models": {}
        });

        assert!(validate_required_fields(&config).is_err());
    }

    #[test]
    fn test_migration_manager_default() {
        let manager = MigrationManager::default();
        assert!(!manager.migrations.is_empty());
    }
}
