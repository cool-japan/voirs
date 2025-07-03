//! Plugin manager for loading and managing plugins.

use crate::{
    error::{Result, VoirsError},
    traits::Plugin,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Plugin manager for loading and managing plugins
pub struct PluginManager {
    plugins: HashMap<String, Arc<dyn Plugin>>,
}

impl PluginManager {
    /// Create new plugin manager
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }
    
    /// Register a plugin
    pub fn register_plugin(&mut self, name: String, plugin: Arc<dyn Plugin>) {
        self.plugins.insert(name, plugin);
    }
    
    /// Get a plugin by name
    pub fn get_plugin(&self, name: &str) -> Option<&dyn Plugin> {
        self.plugins.get(name).map(|p| p.as_ref())
    }
    
    /// List all registered plugins
    pub fn list_plugins(&self) -> Vec<&str> {
        self.plugins.keys().map(|s| s.as_str()).collect()
    }
    
    /// Unregister a plugin
    pub fn unregister_plugin(&mut self, name: &str) -> Result<()> {
        self.plugins.remove(name);
        Ok(())
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}