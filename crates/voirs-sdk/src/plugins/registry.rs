//! Plugin registry for discovering and loading plugins.

use crate::{
    error::{Result, VoirsError},
    traits::Plugin,
};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Plugin registry for discovering and loading plugins
pub struct PluginRegistry {
    search_paths: Vec<String>,
    plugins: HashMap<String, PluginInfo>,
}

/// Information about a discovered plugin
#[derive(Debug, Clone)]
pub struct PluginInfo {
    pub name: String,
    pub version: String,
    pub description: String,
    pub path: String,
}

impl PluginRegistry {
    /// Create new plugin registry
    pub fn new() -> Self {
        Self {
            search_paths: Vec::new(),
            plugins: HashMap::new(),
        }
    }
    
    /// Add a search path for plugins
    pub fn add_search_path<P: AsRef<Path>>(&mut self, path: P) {
        self.search_paths.push(path.as_ref().to_string_lossy().to_string());
    }
    
    /// Discover plugins in search paths
    pub fn discover_plugins(&mut self) -> Result<()> {
        // TODO: Implement plugin discovery
        // This would scan the search paths for plugin files
        tracing::info!("Plugin discovery not yet implemented");
        Ok(())
    }
    
    /// Load a plugin by name
    pub fn load_plugin(&self, name: &str) -> Result<Arc<dyn Plugin>> {
        // TODO: Implement plugin loading
        // This would load the actual plugin from file
        Err(VoirsError::plugin_error(format!("Plugin loading not yet implemented for '{}'", name)))
    }
    
    /// List discovered plugins
    pub fn list_plugins(&self) -> Vec<&PluginInfo> {
        self.plugins.values().collect()
    }
    
    /// Get plugin info by name
    pub fn get_plugin_info(&self, name: &str) -> Option<&PluginInfo> {
        self.plugins.get(name)
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}