//! Plugin registry for managing installed plugins.

use super::{Plugin, PluginError, PluginManager, PluginManifest, PluginResult, PluginType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryEntry {
    pub manifest: PluginManifest,
    pub install_path: PathBuf,
    pub install_date: chrono::DateTime<chrono::Utc>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
    pub enabled: bool,
    pub auto_update: bool,
    pub usage_count: u64,
    pub last_used: Option<chrono::DateTime<chrono::Utc>>,
    pub checksum: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    pub registry_path: PathBuf,
    pub auto_discovery: bool,
    pub auto_update_check: bool,
    pub update_interval_hours: u64,
    pub cleanup_unused_after_days: u64,
    pub max_registry_entries: usize,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            registry_path: dirs::config_dir()
                .unwrap_or_default()
                .join("voirs")
                .join("plugin_registry.json"),
            auto_discovery: true,
            auto_update_check: false,
            update_interval_hours: 24,
            cleanup_unused_after_days: 90,
            max_registry_entries: 1000,
        }
    }
}

pub struct PluginRegistry {
    config: RegistryConfig,
    entries: RwLock<HashMap<String, RegistryEntry>>,
    manager: Arc<PluginManager>,
}

impl PluginRegistry {
    pub fn new(config: RegistryConfig, manager: Arc<PluginManager>) -> Self {
        Self {
            config,
            entries: RwLock::new(HashMap::new()),
            manager,
        }
    }

    pub async fn load(&self) -> PluginResult<()> {
        if !self.config.registry_path.exists() {
            // Create empty registry if it doesn't exist
            self.save().await?;
            return Ok(());
        }

        let content = tokio::fs::read_to_string(&self.config.registry_path).await?;
        let entries: HashMap<String, RegistryEntry> = serde_json::from_str(&content)?;

        {
            let mut entries_guard = self.entries.write().await;
            *entries_guard = entries;
        }

        Ok(())
    }

    pub async fn save(&self) -> PluginResult<()> {
        let entries = self.entries.read().await;
        let content = serde_json::to_string_pretty(&*entries)?;

        // Ensure parent directory exists
        if let Some(parent) = self.config.registry_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        tokio::fs::write(&self.config.registry_path, content).await?;
        Ok(())
    }

    pub async fn register_plugin(
        &self,
        manifest: PluginManifest,
        install_path: PathBuf,
    ) -> PluginResult<()> {
        let now = chrono::Utc::now();
        let checksum = self.calculate_checksum(&install_path).await?;

        let entry = RegistryEntry {
            manifest: manifest.clone(),
            install_path,
            install_date: now,
            last_updated: now,
            enabled: true,
            auto_update: false,
            usage_count: 0,
            last_used: None,
            checksum,
            metadata: HashMap::new(),
        };

        {
            let mut entries_guard = self.entries.write().await;

            // Check registry size limit
            if entries_guard.len() >= self.config.max_registry_entries {
                return Err(PluginError::LoadingFailed(
                    "Registry is full. Please clean up unused plugins.".to_string(),
                ));
            }

            entries_guard.insert(manifest.name.clone(), entry);
        }

        self.save().await?;
        Ok(())
    }

    pub async fn unregister_plugin(&self, name: &str) -> PluginResult<()> {
        {
            let mut entries_guard = self.entries.write().await;
            if entries_guard.remove(name).is_none() {
                return Err(PluginError::NotFound(name.to_string()));
            }
        }

        self.save().await?;
        Ok(())
    }

    pub async fn get_plugin(&self, name: &str) -> Option<RegistryEntry> {
        let entries = self.entries.read().await;
        entries.get(name).cloned()
    }

    pub async fn list_plugins(&self) -> Vec<RegistryEntry> {
        let entries = self.entries.read().await;
        entries.values().cloned().collect()
    }

    pub async fn list_plugins_by_type(&self, plugin_type: PluginType) -> Vec<RegistryEntry> {
        let entries = self.entries.read().await;
        entries
            .values()
            .filter(|entry| entry.manifest.plugin_type == plugin_type)
            .cloned()
            .collect()
    }

    pub async fn enable_plugin(&self, name: &str) -> PluginResult<()> {
        {
            let mut entries_guard = self.entries.write().await;
            if let Some(entry) = entries_guard.get_mut(name) {
                entry.enabled = true;
                entry.last_updated = chrono::Utc::now();
            } else {
                return Err(PluginError::NotFound(name.to_string()));
            }
        }

        self.save().await?;
        Ok(())
    }

    pub async fn disable_plugin(&self, name: &str) -> PluginResult<()> {
        {
            let mut entries_guard = self.entries.write().await;
            if let Some(entry) = entries_guard.get_mut(name) {
                entry.enabled = false;
                entry.last_updated = chrono::Utc::now();
            } else {
                return Err(PluginError::NotFound(name.to_string()));
            }
        }

        self.save().await?;
        Ok(())
    }

    pub async fn record_usage(&self, name: &str) -> PluginResult<()> {
        {
            let mut entries_guard = self.entries.write().await;
            if let Some(entry) = entries_guard.get_mut(name) {
                entry.usage_count += 1;
                entry.last_used = Some(chrono::Utc::now());
            } else {
                return Err(PluginError::NotFound(name.to_string()));
            }
        }

        // Save periodically to avoid too frequent I/O
        if fastrand::f32() < 0.1 {
            self.save().await?;
        }

        Ok(())
    }

    pub async fn update_plugin_metadata(
        &self,
        name: &str,
        key: &str,
        value: serde_json::Value,
    ) -> PluginResult<()> {
        {
            let mut entries_guard = self.entries.write().await;
            if let Some(entry) = entries_guard.get_mut(name) {
                entry.metadata.insert(key.to_string(), value);
                entry.last_updated = chrono::Utc::now();
            } else {
                return Err(PluginError::NotFound(name.to_string()));
            }
        }

        self.save().await?;
        Ok(())
    }

    pub async fn search_plugins(&self, query: &str) -> Vec<RegistryEntry> {
        let entries = self.entries.read().await;
        let query_lower = query.to_lowercase();

        entries
            .values()
            .filter(|entry| {
                entry.manifest.name.to_lowercase().contains(&query_lower)
                    || entry
                        .manifest
                        .description
                        .to_lowercase()
                        .contains(&query_lower)
                    || entry.manifest.author.to_lowercase().contains(&query_lower)
            })
            .cloned()
            .collect()
    }

    pub async fn get_enabled_plugins(&self) -> Vec<RegistryEntry> {
        let entries = self.entries.read().await;
        entries
            .values()
            .filter(|entry| entry.enabled)
            .cloned()
            .collect()
    }

    pub async fn get_stats(&self) -> RegistryStats {
        let entries = self.entries.read().await;
        let total = entries.len();
        let enabled = entries.values().filter(|e| e.enabled).count();
        let by_type = entries
            .values()
            .map(|e| e.manifest.plugin_type.clone())
            .fold(HashMap::new(), |mut acc, t| {
                *acc.entry(format!("{:?}", t)).or_insert(0) += 1;
                acc
            });

        RegistryStats {
            total_plugins: total,
            enabled_plugins: enabled,
            disabled_plugins: total - enabled,
            plugins_by_type: by_type,
            total_usage: entries.values().map(|e| e.usage_count).sum(),
        }
    }

    pub async fn cleanup_unused(&self) -> PluginResult<Vec<String>> {
        let cutoff_date = chrono::Utc::now()
            - chrono::Duration::days(self.config.cleanup_unused_after_days as i64);
        let mut removed = Vec::new();

        {
            let mut entries_guard = self.entries.write().await;
            entries_guard.retain(|name, entry| {
                let should_keep = entry.enabled
                    || entry
                        .last_used
                        .map_or(false, |last_used| last_used > cutoff_date)
                    || entry.usage_count > 0;

                if !should_keep {
                    removed.push(name.clone());
                }

                should_keep
            });
        }

        if !removed.is_empty() {
            self.save().await?;
        }

        Ok(removed)
    }

    pub async fn validate_integrity(&self) -> PluginResult<Vec<String>> {
        let mut issues = Vec::new();
        let entries = self.entries.read().await;

        for (name, entry) in entries.iter() {
            // Check if plugin files still exist
            if !entry.install_path.exists() {
                issues.push(format!(
                    "Plugin '{}' file not found at {}",
                    name,
                    entry.install_path.display()
                ));
                continue;
            }

            // Check checksum
            match self.calculate_checksum(&entry.install_path).await {
                Ok(current_checksum) => {
                    if current_checksum != entry.checksum {
                        issues.push(format!(
                            "Plugin '{}' checksum mismatch (possible corruption)",
                            name
                        ));
                    }
                }
                Err(e) => {
                    issues.push(format!(
                        "Plugin '{}' checksum calculation failed: {}",
                        name, e
                    ));
                }
            }

            // Check manifest validity
            if entry.manifest.name.is_empty() || entry.manifest.version.is_empty() {
                issues.push(format!("Plugin '{}' has invalid manifest", name));
            }
        }

        Ok(issues)
    }

    async fn calculate_checksum(&self, path: &Path) -> PluginResult<String> {
        use sha2::{Digest, Sha256};

        let content = tokio::fs::read(path).await?;
        let mut hasher = Sha256::new();
        hasher.update(&content);
        let result = hasher.finalize();
        Ok(format!("{:x}", result))
    }

    pub async fn discover_and_register(&self) -> PluginResult<usize> {
        if !self.config.auto_discovery {
            return Ok(0);
        }

        let discovered = self.manager.discover_plugins().await?;
        let mut registered_count = 0;

        for plugin_info in discovered {
            if !self
                .entries
                .read()
                .await
                .contains_key(&plugin_info.manifest.name)
            {
                self.register_plugin(plugin_info.manifest, plugin_info.path)
                    .await?;
                registered_count += 1;
            }
        }

        Ok(registered_count)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryStats {
    pub total_plugins: usize,
    pub enabled_plugins: usize,
    pub disabled_plugins: usize,
    pub plugins_by_type: HashMap<String, usize>,
    pub total_usage: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn create_test_manifest() -> PluginManifest {
        PluginManifest {
            name: "test-plugin".to_string(),
            version: "1.0.0".to_string(),
            description: "Test plugin".to_string(),
            author: "Test Author".to_string(),
            api_version: "1.0.0".to_string(),
            plugin_type: PluginType::Extension,
            entry_point: "test_plugin.dll".to_string(),
            dependencies: vec![],
            permissions: vec![],
            configuration: None,
        }
    }

    #[tokio::test]
    async fn test_registry_creation() {
        let config = RegistryConfig::default();
        let manager = Arc::new(PluginManager::new());
        let registry = PluginRegistry::new(config, manager);

        let stats = registry.get_stats().await;
        assert_eq!(stats.total_plugins, 0);
    }

    #[tokio::test]
    async fn test_plugin_registration() {
        let config = RegistryConfig {
            registry_path: PathBuf::from("/tmp/test_registry.json"),
            ..Default::default()
        };
        let manager = Arc::new(PluginManager::new());
        let registry = PluginRegistry::new(config, manager);

        let manifest = create_test_manifest();
        let install_path = PathBuf::from("/tmp/test_plugin");

        // This will fail because the path doesn't exist, but that's expected in a test
        let result = registry.register_plugin(manifest, install_path).await;
        assert!(result.is_err()); // Checksum calculation will fail

        let stats = registry.get_stats().await;
        assert_eq!(stats.total_plugins, 0); // Registration failed
    }

    #[tokio::test]
    async fn test_plugin_search() {
        let config = RegistryConfig::default();
        let manager = Arc::new(PluginManager::new());
        let registry = PluginRegistry::new(config, manager);

        let results = registry.search_plugins("test").await;
        assert_eq!(results.len(), 0);
    }

    #[tokio::test]
    async fn test_registry_stats() {
        let config = RegistryConfig::default();
        let manager = Arc::new(PluginManager::new());
        let registry = PluginRegistry::new(config, manager);

        let stats = registry.get_stats().await;
        assert_eq!(stats.total_plugins, 0);
        assert_eq!(stats.enabled_plugins, 0);
        assert_eq!(stats.disabled_plugins, 0);
        assert_eq!(stats.total_usage, 0);
    }
}
