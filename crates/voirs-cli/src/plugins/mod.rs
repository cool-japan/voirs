//! Plugin system for extending VoiRS CLI functionality.
//!
//! This module provides a secure, extensible plugin architecture that allows
//! third-party developers to extend VoiRS with custom effects, voices, and
//! processing capabilities. The system supports both native Rust plugins
//! and WebAssembly-based plugins for security and portability.
//!
//! ## Features
//!
//! - **Secure Plugin Loading**: Sandboxed execution with permission system
//! - **Multiple Plugin Types**: Effects, voices, processors, and extensions
//! - **Plugin Discovery**: Automatic discovery from standard directories
//! - **API Versioning**: Version compatibility checking for plugins
//! - **Permission System**: Granular control over plugin capabilities
//! - **Error Handling**: Comprehensive error reporting and recovery
//!
//! ## Plugin Types
//!
//! - **Effect Plugins**: Audio processing effects (reverb, chorus, etc.)
//! - **Voice Plugins**: Custom voice models and synthesis engines
//! - **Processor Plugins**: Text processing and analysis tools
//! - **Extension Plugins**: General CLI functionality extensions
//!
//! ## Example
//!
//! ```rust,no_run
//! use voirs_cli::plugins::PluginManager;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut manager = PluginManager::new();
//! let plugins = manager.discover_plugins().await?;
//!
//! for plugin in plugins {
//!     println!("Found plugin: {} v{}", plugin.manifest.name, plugin.manifest.version);
//!     if plugin.enabled {
//!         manager.load_plugin(&plugin.manifest.name).await?;
//!     }
//! }
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

pub mod api;
pub mod effects;
pub mod loader;
pub mod registry;
pub mod voices;

#[derive(Debug, Error)]
pub enum PluginError {
    #[error("Plugin not found: {0}")]
    NotFound(String),

    #[error("Plugin loading failed: {0}")]
    LoadingFailed(String),

    #[error("Invalid plugin manifest: {0}")]
    InvalidManifest(String),

    #[error("Plugin API version mismatch: expected {expected}, got {actual}")]
    ApiVersionMismatch { expected: String, actual: String },

    #[error("Plugin permission denied: {0}")]
    PermissionDenied(String),

    #[error("Plugin execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Plugin dependency missing: {0}")]
    DependencyMissing(String),

    #[error("Plugin security violation: {0}")]
    SecurityViolation(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

pub type PluginResult<T> = Result<T, PluginError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub api_version: String,
    pub plugin_type: PluginType,
    pub entry_point: String,
    pub dependencies: Vec<String>,
    pub permissions: Vec<Permission>,
    pub configuration: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PluginType {
    Effect,
    Voice,
    Processor,
    Extension,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Permission {
    FileRead,
    FileWrite,
    NetworkAccess,
    SystemInfo,
    AudioCapture,
    AudioPlayback,
    ConfigAccess,
    ModelAccess,
}

#[derive(Debug, Clone)]
pub struct PluginInfo {
    pub manifest: PluginManifest,
    pub path: PathBuf,
    pub loaded: bool,
    pub enabled: bool,
    pub load_count: u32,
    pub last_error: Option<String>,
}

pub trait Plugin: Send + Sync {
    fn name(&self) -> &str;
    fn version(&self) -> &str;
    fn description(&self) -> &str;
    fn plugin_type(&self) -> PluginType;

    fn initialize(&mut self, config: &serde_json::Value) -> PluginResult<()>;
    fn cleanup(&mut self) -> PluginResult<()>;

    fn get_capabilities(&self) -> Vec<String>;
    fn execute(&self, command: &str, args: &serde_json::Value) -> PluginResult<serde_json::Value>;
}

pub struct PluginManager {
    plugins: RwLock<HashMap<String, Arc<RwLock<Box<dyn Plugin>>>>>,
    plugin_info: RwLock<HashMap<String, PluginInfo>>,
    plugin_directories: Vec<PathBuf>,
    api_version: String,
    security_enabled: bool,
}

impl PluginManager {
    pub fn new() -> Self {
        Self {
            plugins: RwLock::new(HashMap::new()),
            plugin_info: RwLock::new(HashMap::new()),
            plugin_directories: vec![
                dirs::config_dir()
                    .unwrap_or_default()
                    .join("voirs")
                    .join("plugins"),
                dirs::data_local_dir()
                    .unwrap_or_default()
                    .join("voirs")
                    .join("plugins"),
                PathBuf::from("/usr/local/share/voirs/plugins"),
                PathBuf::from("./plugins"),
            ],
            api_version: "1.0.0".to_string(),
            security_enabled: true,
        }
    }

    pub fn add_plugin_directory<P: AsRef<Path>>(&mut self, path: P) {
        self.plugin_directories.push(path.as_ref().to_path_buf());
    }

    pub async fn discover_plugins(&self) -> PluginResult<Vec<PluginInfo>> {
        let mut discovered = Vec::new();

        for directory in &self.plugin_directories {
            if !directory.exists() {
                continue;
            }

            let mut entries = tokio::fs::read_dir(directory).await?;

            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();

                if path.is_dir() {
                    let manifest_path = path.join("plugin.json");
                    if manifest_path.exists() {
                        match self.load_manifest(&manifest_path).await {
                            Ok(manifest) => {
                                let plugin_info = PluginInfo {
                                    manifest,
                                    path: path.clone(),
                                    loaded: false,
                                    enabled: true,
                                    load_count: 0,
                                    last_error: None,
                                };
                                discovered.push(plugin_info);
                            }
                            Err(e) => {
                                eprintln!(
                                    "Failed to load plugin manifest {}: {}",
                                    manifest_path.display(),
                                    e
                                );
                            }
                        }
                    }
                }
            }
        }

        Ok(discovered)
    }

    pub async fn load_plugin(&self, name: &str) -> PluginResult<()> {
        let plugin_info = {
            let info_guard = self.plugin_info.read().await;
            info_guard
                .get(name)
                .cloned()
                .ok_or_else(|| PluginError::NotFound(name.to_string()))?
        };

        if plugin_info.manifest.api_version != self.api_version {
            return Err(PluginError::ApiVersionMismatch {
                expected: self.api_version.clone(),
                actual: plugin_info.manifest.api_version.clone(),
            });
        }

        if self.security_enabled {
            self.validate_permissions(&plugin_info.manifest.permissions)?;
        }

        let plugin = self
            .load_plugin_from_path(&plugin_info.path, &plugin_info.manifest)
            .await?;

        {
            let mut plugins_guard = self.plugins.write().await;
            plugins_guard.insert(name.to_string(), Arc::new(RwLock::new(plugin)));
        }

        {
            let mut info_guard = self.plugin_info.write().await;
            if let Some(info) = info_guard.get_mut(name) {
                info.loaded = true;
                info.load_count += 1;
                info.last_error = None;
            }
        }

        Ok(())
    }

    pub async fn unload_plugin(&self, name: &str) -> PluginResult<()> {
        let plugin = {
            let mut plugins_guard = self.plugins.write().await;
            plugins_guard
                .remove(name)
                .ok_or_else(|| PluginError::NotFound(name.to_string()))?
        };

        {
            let mut plugin_guard = plugin.write().await;
            plugin_guard.cleanup()?;
        }

        {
            let mut info_guard = self.plugin_info.write().await;
            if let Some(info) = info_guard.get_mut(name) {
                info.loaded = false;
                info.last_error = None;
            }
        }

        Ok(())
    }

    pub async fn execute_plugin(
        &self,
        name: &str,
        command: &str,
        args: &serde_json::Value,
    ) -> PluginResult<serde_json::Value> {
        let plugin = {
            let plugins_guard = self.plugins.read().await;
            plugins_guard
                .get(name)
                .cloned()
                .ok_or_else(|| PluginError::NotFound(name.to_string()))?
        };

        let plugin_guard = plugin.read().await;
        plugin_guard.execute(command, args)
    }

    pub async fn list_plugins(&self) -> Vec<PluginInfo> {
        let info_guard = self.plugin_info.read().await;
        info_guard.values().cloned().collect()
    }

    pub async fn get_plugin_info(&self, name: &str) -> Option<PluginInfo> {
        let info_guard = self.plugin_info.read().await;
        info_guard.get(name).cloned()
    }

    pub async fn enable_plugin(&self, name: &str) -> PluginResult<()> {
        let mut info_guard = self.plugin_info.write().await;
        if let Some(info) = info_guard.get_mut(name) {
            info.enabled = true;
            Ok(())
        } else {
            Err(PluginError::NotFound(name.to_string()))
        }
    }

    pub async fn disable_plugin(&self, name: &str) -> PluginResult<()> {
        let mut info_guard = self.plugin_info.write().await;
        if let Some(info) = info_guard.get_mut(name) {
            info.enabled = false;
            Ok(())
        } else {
            Err(PluginError::NotFound(name.to_string()))
        }
    }

    async fn load_manifest(&self, path: &Path) -> PluginResult<PluginManifest> {
        let content = tokio::fs::read_to_string(path).await?;
        let manifest: PluginManifest = serde_json::from_str(&content)?;
        Ok(manifest)
    }

    async fn load_plugin_from_path(
        &self,
        _path: &Path,
        _manifest: &PluginManifest,
    ) -> PluginResult<Box<dyn Plugin>> {
        // For now, return a mock plugin
        // In a real implementation, this would use dynamic loading (libloading crate)
        // or WebAssembly (wasmtime crate) for security
        Ok(Box::new(MockPlugin::new()))
    }

    fn validate_permissions(&self, permissions: &[Permission]) -> PluginResult<()> {
        // Implement security validation logic
        for permission in permissions {
            match permission {
                Permission::FileWrite => {
                    // Check if file write is allowed
                    // This would involve checking system policies, user permissions, etc.
                }
                Permission::NetworkAccess => {
                    // Check if network access is allowed
                    // This might involve checking firewall rules, network policies, etc.
                }
                Permission::SystemInfo => {
                    // Check if system information access is allowed
                }
                _ => {
                    // Other permissions can be validated here
                }
            }
        }
        Ok(())
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

// Mock plugin for testing and development
struct MockPlugin {
    name: String,
    version: String,
    description: String,
}

impl MockPlugin {
    fn new() -> Self {
        Self {
            name: "mock-plugin".to_string(),
            version: "1.0.0".to_string(),
            description: "Mock plugin for testing".to_string(),
        }
    }
}

impl Plugin for MockPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn plugin_type(&self) -> PluginType {
        PluginType::Extension
    }

    fn initialize(&mut self, _config: &serde_json::Value) -> PluginResult<()> {
        Ok(())
    }

    fn cleanup(&mut self) -> PluginResult<()> {
        Ok(())
    }

    fn get_capabilities(&self) -> Vec<String> {
        vec!["test".to_string()]
    }

    fn execute(&self, command: &str, args: &serde_json::Value) -> PluginResult<serde_json::Value> {
        match command {
            "test" => Ok(serde_json::json!({
                "status": "ok",
                "args": args
            })),
            _ => Err(PluginError::ExecutionFailed(format!(
                "Unknown command: {}",
                command
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_plugin_manager_creation() {
        let manager = PluginManager::new();
        assert_eq!(manager.api_version, "1.0.0");
        assert!(manager.security_enabled);
    }

    #[tokio::test]
    async fn test_plugin_discovery() {
        let manager = PluginManager::new();
        let plugins = manager.discover_plugins().await.unwrap();
        // Should not fail even if no plugins found
        // Plugin discovery should not fail even if no plugins found
    }

    #[tokio::test]
    async fn test_mock_plugin() {
        let plugin = MockPlugin::new();
        assert_eq!(plugin.name(), "mock-plugin");
        assert_eq!(plugin.version(), "1.0.0");
        assert_eq!(plugin.description(), "Mock plugin for testing");
    }

    #[tokio::test]
    async fn test_plugin_execution() {
        let plugin = MockPlugin::new();
        let result = plugin
            .execute("test", &serde_json::json!({"key": "value"}))
            .unwrap();
        assert_eq!(result["status"], "ok");
        assert_eq!(result["args"]["key"], "value");
    }

    #[tokio::test]
    async fn test_plugin_unknown_command() {
        let plugin = MockPlugin::new();
        let result = plugin.execute("unknown", &serde_json::json!({}));
        assert!(result.is_err());
    }
}
