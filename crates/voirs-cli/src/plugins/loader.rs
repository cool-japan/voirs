//! Plugin loading and dynamic library management.

use super::{Plugin, PluginError, PluginManifest, PluginResult, PluginType};
use libloading::{Library, Symbol};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use wasmtime::{Engine, Instance, Module, Store, TypedFunc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoaderConfig {
    pub search_paths: Vec<PathBuf>,
    pub allowed_extensions: Vec<String>,
    pub security_enabled: bool,
    pub lazy_loading: bool,
    pub cache_manifests: bool,
    pub max_load_attempts: u32,
    pub load_timeout_ms: u64,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            search_paths: vec![
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
            allowed_extensions: vec![
                "dll".to_string(),   // Windows
                "so".to_string(),    // Linux
                "dylib".to_string(), // macOS
                "wasm".to_string(),  // WebAssembly
            ],
            security_enabled: true,
            lazy_loading: true,
            cache_manifests: true,
            max_load_attempts: 3,
            load_timeout_ms: 5000,
        }
    }
}

pub struct LoadedPlugin {
    pub manifest: PluginManifest,
    pub plugin: Arc<dyn Plugin>,
    pub load_time: std::time::Instant,
    pub load_count: u32,
    pub last_access: std::time::Instant,
    pub plugin_type: LoadedPluginType,
}

pub enum LoadedPluginType {
    Native {
        library: Arc<Library>,
    },
    WebAssembly {
        engine: Arc<Engine>,
        module: Arc<Module>,
    },
    Builtin,
}

pub struct PluginLoader {
    config: LoaderConfig,
    loaded_plugins: HashMap<String, LoadedPlugin>,
    manifest_cache: HashMap<PathBuf, PluginManifest>,
    loading_in_progress: HashMap<String, std::time::Instant>,
    wasm_engine: Arc<Engine>,
}

impl PluginLoader {
    pub fn new(config: LoaderConfig) -> PluginResult<Self> {
        let wasm_engine = Arc::new(Engine::default());

        Ok(Self {
            config,
            loaded_plugins: HashMap::new(),
            manifest_cache: HashMap::new(),
            loading_in_progress: HashMap::new(),
            wasm_engine,
        })
    }

    pub fn with_default_config() -> PluginResult<Self> {
        Self::new(LoaderConfig::default())
    }

    pub async fn discover_plugins(&mut self) -> PluginResult<Vec<PluginManifest>> {
        let mut discovered = Vec::new();
        let search_paths = self.config.search_paths.clone();

        for search_path in &search_paths {
            if !search_path.exists() {
                continue;
            }

            let plugins = self.scan_directory(search_path).await?;
            discovered.extend(plugins);
        }

        Ok(discovered)
    }

    async fn scan_directory(&mut self, dir: &Path) -> PluginResult<Vec<PluginManifest>> {
        let mut plugins = Vec::new();
        let mut entries = tokio::fs::read_dir(dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            if path.is_dir() {
                // Look for plugin.json in subdirectories
                let manifest_path = path.join("plugin.json");
                if manifest_path.exists() {
                    match self.load_manifest(&manifest_path).await {
                        Ok(manifest) => plugins.push(manifest),
                        Err(e) => {
                            eprintln!(
                                "Failed to load manifest from {}: {}",
                                manifest_path.display(),
                                e
                            );
                        }
                    }
                }
            } else if let Some(extension) = path.extension().and_then(|s| s.to_str()) {
                // Check for plugin libraries
                if self
                    .config
                    .allowed_extensions
                    .contains(&extension.to_lowercase())
                {
                    // Look for accompanying manifest
                    let manifest_path = path.with_extension("json");
                    if manifest_path.exists() {
                        match self.load_manifest(&manifest_path).await {
                            Ok(manifest) => plugins.push(manifest),
                            Err(e) => {
                                eprintln!(
                                    "Failed to load manifest from {}: {}",
                                    manifest_path.display(),
                                    e
                                );
                            }
                        }
                    }
                }
            }
        }

        Ok(plugins)
    }

    async fn load_manifest(&mut self, path: &Path) -> PluginResult<PluginManifest> {
        // Check cache first
        if self.config.cache_manifests {
            if let Some(cached) = self.manifest_cache.get(path) {
                return Ok(cached.clone());
            }
        }

        let content = tokio::fs::read_to_string(path).await?;
        let manifest: PluginManifest = serde_json::from_str(&content)?;

        // Validate manifest
        self.validate_manifest(&manifest)?;

        // Cache the manifest
        if self.config.cache_manifests {
            self.manifest_cache
                .insert(path.to_path_buf(), manifest.clone());
        }

        Ok(manifest)
    }

    fn validate_manifest(&self, manifest: &PluginManifest) -> PluginResult<()> {
        if manifest.name.is_empty() {
            return Err(PluginError::InvalidManifest(
                "Plugin name cannot be empty".to_string(),
            ));
        }

        if manifest.version.is_empty() {
            return Err(PluginError::InvalidManifest(
                "Plugin version cannot be empty".to_string(),
            ));
        }

        if manifest.entry_point.is_empty() {
            return Err(PluginError::InvalidManifest(
                "Entry point cannot be empty".to_string(),
            ));
        }

        // Validate API version compatibility
        if !self.is_api_version_compatible(&manifest.api_version) {
            return Err(PluginError::ApiVersionMismatch {
                expected: "1.0.x".to_string(),
                actual: manifest.api_version.clone(),
            });
        }

        Ok(())
    }

    fn is_api_version_compatible(&self, version: &str) -> bool {
        // Simple semver-like compatibility check
        // For now, accept 1.0.x versions
        version.starts_with("1.0.")
    }

    pub async fn load_plugin(
        &mut self,
        name: &str,
        manifest: &PluginManifest,
    ) -> PluginResult<Arc<dyn Plugin>> {
        // Check if already loading
        if self.loading_in_progress.contains_key(name) {
            return Err(PluginError::LoadingFailed(format!(
                "Plugin {} is already being loaded",
                name
            )));
        }

        // Check if already loaded
        if let Some(loaded) = self.loaded_plugins.get_mut(name) {
            loaded.last_access = std::time::Instant::now();
            loaded.load_count += 1;
            return Ok(loaded.plugin.clone());
        }

        // Mark as loading
        self.loading_in_progress
            .insert(name.to_string(), std::time::Instant::now());

        let result = self.load_plugin_impl(name, manifest).await;

        // Remove from loading queue
        self.loading_in_progress.remove(name);

        result
    }

    async fn load_plugin_impl(
        &mut self,
        name: &str,
        manifest: &PluginManifest,
    ) -> PluginResult<Arc<dyn Plugin>> {
        let start_time = std::time::Instant::now();

        // Determine plugin type based on entry point
        let entry_path = self.resolve_plugin_entry_path(manifest)?;
        let (plugin, plugin_type) = self.load_plugin_from_path(&entry_path, manifest).await?;

        let loaded_plugin = LoadedPlugin {
            manifest: manifest.clone(),
            plugin: plugin.clone(),
            load_time: start_time,
            load_count: 1,
            last_access: std::time::Instant::now(),
            plugin_type,
        };

        self.loaded_plugins.insert(name.to_string(), loaded_plugin);

        Ok(plugin)
    }

    fn resolve_plugin_entry_path(&self, manifest: &PluginManifest) -> PluginResult<PathBuf> {
        // Look for the entry point in the search paths
        for search_path in &self.config.search_paths {
            let plugin_dir = search_path.join(&manifest.name);
            let entry_path = plugin_dir.join(&manifest.entry_point);

            if entry_path.exists() {
                return Ok(entry_path);
            }

            // Also check directly in the search path
            let direct_entry = search_path.join(&manifest.entry_point);
            if direct_entry.exists() {
                return Ok(direct_entry);
            }
        }

        Err(PluginError::LoadingFailed(format!(
            "Entry point '{}' not found for plugin '{}'",
            manifest.entry_point, manifest.name
        )))
    }

    async fn load_plugin_from_path(
        &self,
        path: &Path,
        manifest: &PluginManifest,
    ) -> PluginResult<(Arc<dyn Plugin>, LoadedPluginType)> {
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| {
                PluginError::LoadingFailed("Invalid plugin file extension".to_string())
            })?;

        match extension.to_lowercase().as_str() {
            "wasm" => self.load_wasm_plugin(path, manifest).await,
            "dll" | "so" | "dylib" => self.load_native_plugin(path, manifest).await,
            _ => {
                // Fall back to builtin plugin
                let plugin = self.create_builtin_plugin(manifest)?;
                Ok((plugin, LoadedPluginType::Builtin))
            }
        }
    }

    async fn load_wasm_plugin(
        &self,
        path: &Path,
        manifest: &PluginManifest,
    ) -> PluginResult<(Arc<dyn Plugin>, LoadedPluginType)> {
        let wasm_bytes = tokio::fs::read(path)
            .await
            .map_err(|e| PluginError::LoadingFailed(format!("Failed to read WASM file: {}", e)))?;

        let module = Module::new(&*self.wasm_engine, &wasm_bytes).map_err(|e| {
            PluginError::LoadingFailed(format!("Failed to compile WASM module: {}", e))
        })?;

        let plugin = Arc::new(WasmPlugin::new(
            manifest.clone(),
            self.wasm_engine.clone(),
            Arc::new(module.clone()),
        ));

        let plugin_type = LoadedPluginType::WebAssembly {
            engine: self.wasm_engine.clone(),
            module: Arc::new(module),
        };

        Ok((plugin as Arc<dyn Plugin>, plugin_type))
    }

    async fn load_native_plugin(
        &self,
        path: &Path,
        manifest: &PluginManifest,
    ) -> PluginResult<(Arc<dyn Plugin>, LoadedPluginType)> {
        let library = unsafe {
            Library::new(path).map_err(|e| {
                PluginError::LoadingFailed(format!("Failed to load native library: {}", e))
            })?
        };

        let library = Arc::new(library);

        // Look for the plugin factory function
        let create_plugin: Symbol<unsafe extern "C" fn() -> *mut dyn Plugin> = unsafe {
            library.get(b"create_plugin").map_err(|e| {
                PluginError::LoadingFailed(format!(
                    "Plugin factory function 'create_plugin' not found: {}",
                    e
                ))
            })?
        };

        let plugin_ptr = unsafe { create_plugin() };
        if plugin_ptr.is_null() {
            return Err(PluginError::LoadingFailed(
                "Plugin factory returned null".to_string(),
            ));
        }

        let plugin = unsafe { Arc::from_raw(plugin_ptr) };

        let plugin_type = LoadedPluginType::Native {
            library: library.clone(),
        };

        Ok((plugin, plugin_type))
    }

    fn create_builtin_plugin(&self, manifest: &PluginManifest) -> PluginResult<Arc<dyn Plugin>> {
        match manifest.plugin_type {
            PluginType::Effect => Ok(Arc::new(super::effects::ReverbEffectPlugin::new())),
            PluginType::Voice => Ok(Arc::new(super::voices::DefaultVoicePlugin::new(
                &manifest.name,
            ))),
            PluginType::Processor => Ok(Arc::new(MockProcessorPlugin::new(&manifest.name))),
            PluginType::Extension => Ok(Arc::new(MockExtensionPlugin::new(&manifest.name))),
        }
    }

    pub async fn unload_plugin(&mut self, name: &str) -> PluginResult<()> {
        if let Some(loaded) = self.loaded_plugins.remove(name) {
            // In a real implementation, this would handle dynamic unloading
            drop(loaded);
            Ok(())
        } else {
            Err(PluginError::NotFound(name.to_string()))
        }
    }

    pub fn is_plugin_loaded(&self, name: &str) -> bool {
        self.loaded_plugins.contains_key(name)
    }

    pub fn get_loaded_plugins(&self) -> Vec<String> {
        self.loaded_plugins.keys().cloned().collect()
    }

    pub fn get_plugin_info(&self, name: &str) -> Option<&LoadedPlugin> {
        self.loaded_plugins.get(name)
    }

    pub fn cleanup_unused_plugins(&mut self, max_idle_time: std::time::Duration) {
        let now = std::time::Instant::now();
        self.loaded_plugins
            .retain(|_name, loaded| now.duration_since(loaded.last_access) < max_idle_time);
    }

    pub fn get_stats(&self) -> LoaderStats {
        LoaderStats {
            total_loaded: self.loaded_plugins.len(),
            total_cached_manifests: self.manifest_cache.len(),
            currently_loading: self.loading_in_progress.len(),
            search_paths: self.config.search_paths.len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoaderStats {
    pub total_loaded: usize,
    pub total_cached_manifests: usize,
    pub currently_loading: usize,
    pub search_paths: usize,
}

// WebAssembly plugin wrapper
pub struct WasmPlugin {
    manifest: PluginManifest,
    engine: Arc<Engine>,
    module: Arc<Module>,
}

impl WasmPlugin {
    pub fn new(manifest: PluginManifest, engine: Arc<Engine>, module: Arc<Module>) -> Self {
        Self {
            manifest,
            engine,
            module,
        }
    }

    fn create_store(&self) -> Store<()> {
        Store::new(&*self.engine, ())
    }

    fn call_wasm_function(
        &self,
        function_name: &str,
        args: &[wasmtime::Val],
    ) -> PluginResult<Vec<wasmtime::Val>> {
        let mut store = self.create_store();
        let instance = Instance::new(&mut store, &*self.module, &[]).map_err(|e| {
            PluginError::ExecutionFailed(format!("Failed to instantiate WASM module: {}", e))
        })?;

        let func = instance
            .get_typed_func::<(), i32>(&mut store, function_name)
            .map_err(|e| {
                PluginError::ExecutionFailed(format!(
                    "Function '{}' not found: {}",
                    function_name, e
                ))
            })?;

        let result = func.call(&mut store, ()).map_err(|e| {
            PluginError::ExecutionFailed(format!("WASM function call failed: {}", e))
        })?;

        Ok(vec![wasmtime::Val::I32(result)])
    }
}

impl Plugin for WasmPlugin {
    fn name(&self) -> &str {
        &self.manifest.name
    }

    fn version(&self) -> &str {
        &self.manifest.version
    }

    fn description(&self) -> &str {
        &self.manifest.description
    }

    fn plugin_type(&self) -> PluginType {
        self.manifest.plugin_type.clone()
    }

    fn initialize(&mut self, _config: &serde_json::Value) -> PluginResult<()> {
        // Call WASM initialization function if available
        match self.call_wasm_function("initialize", &[]) {
            Ok(_) => Ok(()),
            Err(_) => {
                // Initialize function is optional
                Ok(())
            }
        }
    }

    fn cleanup(&mut self) -> PluginResult<()> {
        // Call WASM cleanup function if available
        match self.call_wasm_function("cleanup", &[]) {
            Ok(_) => Ok(()),
            Err(_) => {
                // Cleanup function is optional
                Ok(())
            }
        }
    }

    fn get_capabilities(&self) -> Vec<String> {
        // For now, return basic capabilities
        // In a real implementation, this would query the WASM module
        vec!["execute".to_string()]
    }

    fn execute(&self, command: &str, args: &serde_json::Value) -> PluginResult<serde_json::Value> {
        // For now, return a basic response
        // In a real implementation, this would call the appropriate WASM function
        Ok(serde_json::json!({
            "status": "ok",
            "command": command,
            "args": args,
            "plugin": self.name(),
            "type": "wasm"
        }))
    }
}

// Mock plugin implementations for testing

struct MockProcessorPlugin {
    name: String,
}

impl MockProcessorPlugin {
    fn new(name: &str) -> Self {
        Self {
            name: format!("processor-{}", name),
        }
    }
}

impl Plugin for MockProcessorPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "Mock processor plugin"
    }

    fn plugin_type(&self) -> PluginType {
        PluginType::Processor
    }

    fn initialize(&mut self, _config: &serde_json::Value) -> PluginResult<()> {
        Ok(())
    }

    fn cleanup(&mut self) -> PluginResult<()> {
        Ok(())
    }

    fn get_capabilities(&self) -> Vec<String> {
        vec!["process".to_string()]
    }

    fn execute(&self, command: &str, args: &serde_json::Value) -> PluginResult<serde_json::Value> {
        match command {
            "process" => Ok(serde_json::json!({
                "status": "processed",
                "args": args
            })),
            _ => Err(PluginError::ExecutionFailed(format!(
                "Unknown command: {}",
                command
            ))),
        }
    }
}

struct MockExtensionPlugin {
    name: String,
}

impl MockExtensionPlugin {
    fn new(name: &str) -> Self {
        Self {
            name: format!("extension-{}", name),
        }
    }
}

impl Plugin for MockExtensionPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "Mock extension plugin"
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
        vec!["extend".to_string()]
    }

    fn execute(&self, command: &str, args: &serde_json::Value) -> PluginResult<serde_json::Value> {
        match command {
            "extend" => Ok(serde_json::json!({
                "status": "extended",
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

    #[test]
    fn test_loader_config_default() {
        let config = LoaderConfig::default();
        assert!(config.security_enabled);
        assert!(config.lazy_loading);
        assert!(config.cache_manifests);
        assert_eq!(config.max_load_attempts, 3);
    }

    #[tokio::test]
    async fn test_plugin_loader_creation() {
        let loader = PluginLoader::with_default_config().unwrap();
        let stats = loader.get_stats();
        assert_eq!(stats.total_loaded, 0);
        assert_eq!(stats.currently_loading, 0);
    }

    #[tokio::test]
    async fn test_plugin_discovery() {
        let mut loader = PluginLoader::with_default_config().unwrap();
        let plugins = loader.discover_plugins().await.unwrap();
        // Should not fail even if no plugins found
        // Plugin discovery should not fail even if no plugins found
    }

    #[test]
    fn test_manifest_validation() {
        let loader = PluginLoader::with_default_config().unwrap();

        let valid_manifest = PluginManifest {
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
        };

        assert!(loader.validate_manifest(&valid_manifest).is_ok());

        let invalid_manifest = PluginManifest {
            name: "".to_string(), // Empty name should fail
            ..valid_manifest
        };

        assert!(loader.validate_manifest(&invalid_manifest).is_err());
    }

    #[test]
    fn test_api_version_compatibility() {
        let loader = PluginLoader::with_default_config().unwrap();

        assert!(loader.is_api_version_compatible("1.0.0"));
        assert!(loader.is_api_version_compatible("1.0.1"));
        assert!(!loader.is_api_version_compatible("2.0.0"));
        assert!(!loader.is_api_version_compatible("0.9.0"));
    }

    #[test]
    fn test_mock_plugins() {
        let processor = MockProcessorPlugin::new("test");
        assert_eq!(processor.name(), "processor-test");
        assert_eq!(processor.plugin_type(), PluginType::Processor);

        let extension = MockExtensionPlugin::new("test");
        assert_eq!(extension.name(), "extension-test");
        assert_eq!(extension.plugin_type(), PluginType::Extension);
    }
}
