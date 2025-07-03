//! Configuration profile management.
//!
//! Provides support for multiple configuration profiles, allowing users to
//! quickly switch between different settings for different use cases.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;
use serde::{Deserialize, Serialize};
use crate::config::{CliConfig, ConfigManager};
use crate::error::{CliError, CliResult};

/// Profile metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileInfo {
    /// Profile name
    pub name: String,
    /// Profile description
    pub description: Option<String>,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last modified timestamp
    pub modified_at: chrono::DateTime<chrono::Utc>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Whether this is a system/built-in profile
    pub system: bool,
}

/// Profile configuration container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigProfile {
    /// Profile metadata
    pub info: ProfileInfo,
    /// The actual configuration
    pub config: CliConfig,
}

/// Profile manager for handling multiple configurations
pub struct ProfileManager {
    profiles_dir: PathBuf,
    current_profile: Option<String>,
    profiles_cache: HashMap<String, ConfigProfile>,
}

impl ProfileManager {
    /// Create a new profile manager
    pub fn new() -> CliResult<Self> {
        let profiles_dir = Self::get_profiles_directory()
            .ok_or_else(|| CliError::config("Cannot determine profiles directory"))?;
        
        // Create profiles directory if it doesn't exist
        fs::create_dir_all(&profiles_dir)
            .map_err(|e| CliError::file_operation("create directory", &profiles_dir.display().to_string(), e))?;
        
        let mut manager = Self {
            profiles_dir,
            current_profile: None,
            profiles_cache: HashMap::new(),
        };
        
        // Load current profile selection
        manager.load_current_profile()?;
        
        // Initialize with default profiles if none exist
        if manager.list_profiles()?.is_empty() {
            manager.create_default_profiles()?;
        }
        
        Ok(manager)
    }
    
    /// Create a new profile
    pub fn create_profile(&mut self, name: &str, description: Option<String>, config: CliConfig) -> CliResult<()> {
        if self.profile_exists(name) {
            return Err(CliError::config(format!("Profile '{}' already exists", name)));
        }
        
        if !Self::is_valid_profile_name(name) {
            return Err(CliError::config("Profile name must contain only alphanumeric characters, hyphens, and underscores"));
        }
        
        let profile = ConfigProfile {
            info: ProfileInfo {
                name: name.to_string(),
                description,
                created_at: chrono::Utc::now(),
                modified_at: chrono::Utc::now(),
                tags: Vec::new(),
                system: false,
            },
            config,
        };
        
        self.save_profile(&profile)?;
        self.profiles_cache.insert(name.to_string(), profile);
        
        Ok(())
    }
    
    /// Update an existing profile
    pub fn update_profile(&mut self, name: &str, config: CliConfig) -> CliResult<()> {
        let mut profile = self.load_profile(name)?;
        profile.config = config;
        profile.info.modified_at = chrono::Utc::now();
        
        self.save_profile(&profile)?;
        self.profiles_cache.insert(name.to_string(), profile);
        
        Ok(())
    }
    
    /// Delete a profile
    pub fn delete_profile(&mut self, name: &str) -> CliResult<()> {
        if !self.profile_exists(name) {
            return Err(CliError::config(format!("Profile '{}' does not exist", name)));
        }
        
        // Check if it's a system profile
        if let Ok(profile) = self.load_profile(name) {
            if profile.info.system {
                return Err(CliError::config(format!("Cannot delete system profile '{}'", name)));
            }
        }
        
        // Don't allow deleting the current profile without switching first
        if self.current_profile.as_ref() == Some(&name.to_string()) {
            return Err(CliError::config("Cannot delete the currently active profile. Switch to another profile first."));
        }
        
        let profile_path = self.get_profile_path(name);
        fs::remove_file(&profile_path)
            .map_err(|e| CliError::file_operation("delete", &profile_path.display().to_string(), e))?;
        
        self.profiles_cache.remove(name);
        
        Ok(())
    }
    
    /// Switch to a different profile
    pub fn switch_profile(&mut self, name: &str) -> CliResult<()> {
        if !self.profile_exists(name) {
            return Err(CliError::config(format!("Profile '{}' does not exist", name)));
        }
        
        self.current_profile = Some(name.to_string());
        self.save_current_profile()?;
        
        Ok(())
    }
    
    /// Get the current profile
    pub fn get_current_profile(&self) -> CliResult<Option<ConfigProfile>> {
        if let Some(ref name) = self.current_profile {
            Ok(Some(self.load_profile(name)?))
        } else {
            Ok(None)
        }
    }
    
    /// Get current profile name
    pub fn get_current_profile_name(&self) -> Option<&str> {
        self.current_profile.as_deref()
    }
    
    /// Load a specific profile
    pub fn load_profile(&self, name: &str) -> CliResult<ConfigProfile> {
        if let Some(profile) = self.profiles_cache.get(name) {
            return Ok(profile.clone());
        }
        
        let profile_path = self.get_profile_path(name);
        if !profile_path.exists() {
            return Err(CliError::config(format!("Profile '{}' does not exist", name)));
        }
        
        let content = fs::read_to_string(&profile_path)
            .map_err(|e| CliError::file_operation("read", &profile_path.display().to_string(), e))?;
        
        let profile: ConfigProfile = toml::from_str(&content)
            .map_err(|e| CliError::config(format!("Invalid profile format for '{}': {}", name, e)))?;
        
        Ok(profile)
    }
    
    /// List all available profiles
    pub fn list_profiles(&self) -> CliResult<Vec<ProfileInfo>> {
        let mut profiles = Vec::new();
        
        let entries = fs::read_dir(&self.profiles_dir)
            .map_err(|e| CliError::file_operation("read directory", &self.profiles_dir.display().to_string(), e))?;
        
        for entry in entries {
            let entry = entry.map_err(|e| CliError::file_operation("read directory entry", "", e))?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("toml") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Ok(profile) = self.load_profile(name) {
                        profiles.push(profile.info);
                    }
                }
            }
        }
        
        // Sort by name
        profiles.sort_by(|a, b| a.name.cmp(&b.name));
        
        Ok(profiles)
    }
    
    /// Copy an existing profile with a new name
    pub fn copy_profile(&mut self, source: &str, target: &str, description: Option<String>) -> CliResult<()> {
        if !self.profile_exists(source) {
            return Err(CliError::config(format!("Source profile '{}' does not exist", source)));
        }
        
        if self.profile_exists(target) {
            return Err(CliError::config(format!("Target profile '{}' already exists", target)));
        }
        
        if !Self::is_valid_profile_name(target) {
            return Err(CliError::config("Profile name must contain only alphanumeric characters, hyphens, and underscores"));
        }
        
        let source_profile = self.load_profile(source)?;
        let mut new_profile = source_profile.clone();
        new_profile.info.name = target.to_string();
        new_profile.info.description = description;
        new_profile.info.created_at = chrono::Utc::now();
        new_profile.info.modified_at = chrono::Utc::now();
        new_profile.info.system = false;
        
        self.save_profile(&new_profile)?;
        self.profiles_cache.insert(target.to_string(), new_profile);
        
        Ok(())
    }
    
    /// Export a profile to a file
    pub fn export_profile(&self, name: &str, export_path: &Path) -> CliResult<()> {
        let profile = self.load_profile(name)?;
        
        let content = toml::to_string_pretty(&profile)
            .map_err(|e| CliError::config(format!("Failed to serialize profile: {}", e)))?;
        
        fs::write(export_path, content)
            .map_err(|e| CliError::file_operation("write", &export_path.display().to_string(), e))?;
        
        Ok(())
    }
    
    /// Import a profile from a file
    pub fn import_profile(&mut self, import_path: &Path, name: Option<&str>) -> CliResult<()> {
        let content = fs::read_to_string(import_path)
            .map_err(|e| CliError::file_operation("read", &import_path.display().to_string(), e))?;
        
        let mut profile: ConfigProfile = toml::from_str(&content)
            .map_err(|e| CliError::config(format!("Invalid profile format: {}", e)))?;
        
        // Use provided name or the one in the file
        let final_name = name.map(|s| s.to_string()).unwrap_or_else(|| profile.info.name.clone());
        
        if self.profile_exists(&final_name) {
            return Err(CliError::config(format!("Profile '{}' already exists", final_name)));
        }
        
        profile.info.name = final_name.clone();
        profile.info.created_at = chrono::Utc::now();
        profile.info.modified_at = chrono::Utc::now();
        profile.info.system = false;
        
        self.save_profile(&profile)?;
        self.profiles_cache.insert(final_name.clone(), profile);
        
        Ok(())
    }
    
    /// Add tags to a profile
    pub fn add_tags(&mut self, name: &str, tags: Vec<String>) -> CliResult<()> {
        let mut profile = self.load_profile(name)?;
        for tag in tags {
            if !profile.info.tags.contains(&tag) {
                profile.info.tags.push(tag);
            }
        }
        profile.info.modified_at = chrono::Utc::now();
        
        self.save_profile(&profile)?;
        self.profiles_cache.insert(name.to_string(), profile);
        
        Ok(())
    }
    
    /// Remove tags from a profile
    pub fn remove_tags(&mut self, name: &str, tags: Vec<String>) -> CliResult<()> {
        let mut profile = self.load_profile(name)?;
        profile.info.tags.retain(|tag| !tags.contains(tag));
        profile.info.modified_at = chrono::Utc::now();
        
        self.save_profile(&profile)?;
        self.profiles_cache.insert(name.to_string(), profile);
        
        Ok(())
    }
    
    /// Search profiles by tags
    pub fn find_profiles_by_tags(&self, tags: &[String]) -> CliResult<Vec<ProfileInfo>> {
        let all_profiles = self.list_profiles()?;
        let matching_profiles = all_profiles
            .into_iter()
            .filter(|profile| tags.iter().any(|tag| profile.tags.contains(tag)))
            .collect();
        
        Ok(matching_profiles)
    }
    
    // Private helper methods
    
    fn profile_exists(&self, name: &str) -> bool {
        self.get_profile_path(name).exists()
    }
    
    fn get_profile_path(&self, name: &str) -> PathBuf {
        self.profiles_dir.join(format!("{}.toml", name))
    }
    
    fn save_profile(&self, profile: &ConfigProfile) -> CliResult<()> {
        let profile_path = self.get_profile_path(&profile.info.name);
        
        let content = toml::to_string_pretty(profile)
            .map_err(|e| CliError::config(format!("Failed to serialize profile: {}", e)))?;
        
        fs::write(&profile_path, content)
            .map_err(|e| CliError::file_operation("write", &profile_path.display().to_string(), e))?;
        
        Ok(())
    }
    
    fn load_current_profile(&mut self) -> CliResult<()> {
        let current_file = self.profiles_dir.join("current");
        if current_file.exists() {
            let content = fs::read_to_string(&current_file)
                .map_err(|e| CliError::file_operation("read", &current_file.display().to_string(), e))?;
            let name = content.trim();
            if self.profile_exists(name) {
                self.current_profile = Some(name.to_string());
            }
        }
        Ok(())
    }
    
    fn save_current_profile(&self) -> CliResult<()> {
        let current_file = self.profiles_dir.join("current");
        if let Some(ref current) = self.current_profile {
            fs::write(&current_file, current)
                .map_err(|e| CliError::file_operation("write", &current_file.display().to_string(), e))?;
        } else {
            // Remove current file if no profile is selected
            if current_file.exists() {
                fs::remove_file(&current_file)
                    .map_err(|e| CliError::file_operation("delete", &current_file.display().to_string(), e))?;
            }
        }
        Ok(())
    }
    
    fn create_default_profiles(&mut self) -> CliResult<()> {
        // Default profile
        let default_profile = ConfigProfile {
            info: ProfileInfo {
                name: "default".to_string(),
                description: Some("Default VoiRS configuration".to_string()),
                created_at: chrono::Utc::now(),
                modified_at: chrono::Utc::now(),
                tags: vec!["system".to_string()],
                system: true,
            },
            config: CliConfig::default(),
        };
        self.save_profile(&default_profile)?;
        
        // High-quality profile
        let mut hq_config = CliConfig::default();
        hq_config.cli.default_quality = "ultra".to_string();
        hq_config.cli.default_output_format = "flac".to_string();
        let hq_profile = ConfigProfile {
            info: ProfileInfo {
                name: "high-quality".to_string(),
                description: Some("High-quality synthesis with FLAC output".to_string()),
                created_at: chrono::Utc::now(),
                modified_at: chrono::Utc::now(),
                tags: vec!["system".to_string(), "quality".to_string()],
                system: true,
            },
            config: hq_config,
        };
        self.save_profile(&hq_profile)?;
        
        // Fast profile
        let mut fast_config = CliConfig::default();
        fast_config.cli.default_quality = "low".to_string();
        fast_config.cli.show_progress = false;
        let fast_profile = ConfigProfile {
            info: ProfileInfo {
                name: "fast".to_string(),
                description: Some("Fast synthesis for quick testing".to_string()),
                created_at: chrono::Utc::now(),
                modified_at: chrono::Utc::now(),
                tags: vec!["system".to_string(), "speed".to_string()],
                system: true,
            },
            config: fast_config,
        };
        self.save_profile(&fast_profile)?;
        
        // Set default as current if no current profile is set
        if self.current_profile.is_none() {
            self.current_profile = Some("default".to_string());
            self.save_current_profile()?;
        }
        
        Ok(())
    }
    
    fn is_valid_profile_name(name: &str) -> bool {
        !name.is_empty() && 
        name.len() <= 50 &&
        name.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    }
    
    fn get_profiles_directory() -> Option<PathBuf> {
        if let Ok(xdg_config) = std::env::var("XDG_CONFIG_HOME") {
            Some(PathBuf::from(xdg_config).join("voirs").join("profiles"))
        } else if let Ok(home) = std::env::var("HOME") {
            Some(PathBuf::from(home).join(".config").join("voirs").join("profiles"))
        } else if let Ok(appdata) = std::env::var("APPDATA") {
            Some(PathBuf::from(appdata).join("voirs").join("profiles"))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_profile_name_validation() {
        assert!(ProfileManager::is_valid_profile_name("default"));
        assert!(ProfileManager::is_valid_profile_name("test-profile"));
        assert!(ProfileManager::is_valid_profile_name("test_profile"));
        assert!(ProfileManager::is_valid_profile_name("profile123"));
        
        assert!(!ProfileManager::is_valid_profile_name(""));
        assert!(!ProfileManager::is_valid_profile_name("profile with spaces"));
        assert!(!ProfileManager::is_valid_profile_name("profile.dot"));
        assert!(!ProfileManager::is_valid_profile_name("profile/slash"));
    }
    
    #[test]
    fn test_profile_info_creation() {
        let info = ProfileInfo {
            name: "test".to_string(),
            description: Some("Test profile".to_string()),
            created_at: chrono::Utc::now(),
            modified_at: chrono::Utc::now(),
            tags: vec!["test".to_string()],
            system: false,
        };
        
        assert_eq!(info.name, "test");
        assert_eq!(info.description, Some("Test profile".to_string()));
        assert!(!info.system);
        assert_eq!(info.tags.len(), 1);
    }
}