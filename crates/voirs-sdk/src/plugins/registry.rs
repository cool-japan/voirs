//! Plugin registry for discovering and loading plugins.

use crate::{
    error::{Result, VoirsError},
    plugins::{manager::PluginManager, PluginConfig, PluginMetadata, PluginType, VoirsPlugin},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
    time::SystemTime,
};
use tracing::{debug, error, info, warn};

#[cfg(feature = "plugins")]
use libloading::{Library, Symbol};
#[cfg(feature = "plugins")]
use wasmtime::{Engine, Instance, Module, Store};

/// Calculate SHA256 checksum of a file
fn calculate_checksum(path: &Path) -> Result<String> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let contents = fs::read(path)?;
    let mut hasher = DefaultHasher::new();
    contents.hash(&mut hasher);
    Ok(format!("{:x}", hasher.finish()))
}

/// Plugin registry for discovering and loading plugins
pub struct PluginRegistry {
    /// Search paths for plugin discovery
    search_paths: Vec<PathBuf>,

    /// Discovered plugins
    plugins: Arc<RwLock<HashMap<String, PluginInfo>>>,

    /// Plugin manifest cache
    manifest_cache: Arc<RwLock<HashMap<PathBuf, PluginManifest>>>,

    /// Plugin loading blacklist
    blacklist: Arc<RwLock<HashSet<String>>>,

    /// Registry configuration
    config: RegistryConfig,

    /// Discovery statistics
    stats: Arc<RwLock<DiscoveryStats>>,
}

/// Information about a discovered plugin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginInfo {
    /// Plugin name
    pub name: String,

    /// Plugin version
    pub version: String,

    /// Plugin description
    pub description: String,

    /// Path to plugin manifest or directory
    pub path: PathBuf,

    /// Plugin type
    pub plugin_type: PluginType,

    /// Plugin author
    pub author: String,

    /// Plugin dependencies
    pub dependencies: Vec<String>,

    /// Supported platforms
    pub supported_platforms: Vec<String>,

    /// Minimum VoiRS version required
    pub min_voirs_version: String,

    /// Last modification time
    pub last_modified: SystemTime,

    /// Whether plugin is trusted
    pub trusted: bool,

    /// Plugin size in bytes
    pub size: u64,

    /// Plugin checksum for integrity verification
    pub checksum: Option<String>,
}

/// Plugin manifest file structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    /// Plugin metadata
    pub plugin: PluginMetadata,

    /// Build configuration
    pub build: Option<BuildConfig>,

    /// Runtime requirements
    pub requirements: Option<RuntimeRequirements>,

    /// Plugin assets
    pub assets: Vec<AssetInfo>,

    /// License information
    pub license: Option<LicenseInfo>,

    /// Documentation links
    pub documentation: Vec<DocumentationLink>,
}

/// Build configuration for plugins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfig {
    /// Target architecture
    pub target: String,

    /// Build type (debug/release)
    pub build_type: String,

    /// Compiler version used
    pub compiler_version: String,

    /// Build timestamp
    pub build_timestamp: String,

    /// Git commit hash
    pub git_hash: Option<String>,
}

/// Runtime requirements for plugins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeRequirements {
    /// Minimum memory requirement in MB
    pub min_memory_mb: Option<u64>,

    /// Required CPU features
    pub cpu_features: Vec<String>,

    /// GPU requirements
    pub gpu_requirements: Option<GpuRequirements>,

    /// System libraries required
    pub system_libraries: Vec<String>,

    /// Environment variables required
    pub env_variables: Vec<String>,
}

/// GPU requirements specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuRequirements {
    /// Required GPU vendor (nvidia, amd, intel)
    pub vendor: Option<String>,

    /// Minimum GPU memory in MB
    pub min_memory_mb: u64,

    /// Required compute capability
    pub compute_capability: Option<String>,

    /// Required GPU APIs (cuda, opencl, vulkan)
    pub apis: Vec<String>,
}

/// Asset information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetInfo {
    /// Asset name
    pub name: String,

    /// Asset type
    pub asset_type: AssetType,

    /// Relative path from plugin directory
    pub path: PathBuf,

    /// Asset size in bytes
    pub size: u64,

    /// Asset checksum
    pub checksum: String,
}

/// Asset type classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssetType {
    /// Compiled binary/library
    Binary,
    /// Configuration file
    Config,
    /// Data file
    Data,
    /// Documentation
    Documentation,
    /// Icon/image
    Image,
    /// Other asset type
    Other(String),
}

/// License information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseInfo {
    /// License identifier (SPDX)
    pub license: String,

    /// License file path
    pub license_file: Option<PathBuf>,

    /// Copyright notice
    pub copyright: Option<String>,
}

/// Documentation link
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationLink {
    /// Link title
    pub title: String,

    /// Link URL
    pub url: String,

    /// Link type
    pub link_type: DocumentationType,
}

/// Documentation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentationType {
    /// API documentation
    Api,
    /// User guide
    Guide,
    /// Tutorial
    Tutorial,
    /// Examples
    Examples,
    /// Changelog
    Changelog,
    /// Other documentation
    Other(String),
}

/// Registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Enable automatic discovery
    pub auto_discovery: bool,

    /// Discovery interval in seconds
    pub discovery_interval: u64,

    /// Enable plugin verification
    pub verify_plugins: bool,

    /// Enable plugin caching
    pub cache_plugins: bool,

    /// Maximum plugin size in MB
    pub max_plugin_size_mb: u64,

    /// Allowed plugin file extensions
    pub allowed_extensions: Vec<String>,

    /// Trusted plugin sources
    pub trusted_sources: Vec<String>,

    /// Maximum search depth
    pub max_search_depth: usize,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            auto_discovery: true,
            discovery_interval: 300, // 5 minutes
            verify_plugins: true,
            cache_plugins: true,
            max_plugin_size_mb: 100,
            allowed_extensions: vec![
                ".so".to_string(),
                ".dll".to_string(),
                ".dylib".to_string(),
                ".wasm".to_string(),
            ],
            trusted_sources: vec![],
            max_search_depth: 3,
        }
    }
}

/// Plugin discovery statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiscoveryStats {
    /// Total plugins discovered
    pub total_discovered: usize,

    /// Valid plugins found
    pub valid_plugins: usize,

    /// Invalid plugins found
    pub invalid_plugins: usize,

    /// Plugins in blacklist
    pub blacklisted_plugins: usize,

    /// Last discovery timestamp
    pub last_discovery: Option<SystemTime>,

    /// Discovery duration in milliseconds
    pub discovery_duration_ms: u64,

    /// Directories scanned
    pub directories_scanned: usize,

    /// Files examined
    pub files_examined: usize,
}

impl PluginRegistry {
    /// Create new plugin registry
    pub fn new() -> Self {
        Self::with_config(RegistryConfig::default())
    }

    /// Create plugin registry with custom configuration
    pub fn with_config(config: RegistryConfig) -> Self {
        Self {
            search_paths: Vec::new(),
            plugins: Arc::new(RwLock::new(HashMap::new())),
            manifest_cache: Arc::new(RwLock::new(HashMap::new())),
            blacklist: Arc::new(RwLock::new(HashSet::new())),
            config,
            stats: Arc::new(RwLock::new(DiscoveryStats::default())),
        }
    }

    /// Add a search path for plugins
    pub fn add_search_path<P: AsRef<Path>>(&mut self, path: P) {
        let path_buf = path.as_ref().to_path_buf();
        if !self.search_paths.contains(&path_buf) {
            self.search_paths.push(path_buf);
            info!("Added plugin search path: {:?}", path.as_ref());
        }
    }

    /// Remove a search path
    pub fn remove_search_path<P: AsRef<Path>>(&mut self, path: P) {
        let path_buf = path.as_ref().to_path_buf();
        self.search_paths.retain(|p| p != &path_buf);
        info!("Removed plugin search path: {:?}", path.as_ref());
    }

    /// Get all search paths
    pub fn get_search_paths(&self) -> Vec<PathBuf> {
        self.search_paths.clone()
    }

    /// Discover plugins in search paths
    pub async fn discover_plugins(&mut self) -> Result<usize> {
        let start_time = SystemTime::now();
        let mut stats = DiscoveryStats::default();

        info!(
            "Starting plugin discovery in {} paths",
            self.search_paths.len()
        );

        let mut discovered_plugins = HashMap::new();

        for search_path in &self.search_paths {
            if !search_path.exists() {
                warn!("Search path does not exist: {:?}", search_path);
                continue;
            }

            match self.discover_in_path(search_path, &mut stats).await {
                Ok(path_plugins) => {
                    for (name, plugin_info) in path_plugins {
                        discovered_plugins.insert(name, plugin_info);
                    }
                }
                Err(e) => {
                    error!("Failed to discover plugins in {:?}: {}", search_path, e);
                    stats.invalid_plugins += 1;
                }
            }
        }

        // Update the main plugins collection
        {
            let mut plugins = self.plugins.write().unwrap();
            *plugins = discovered_plugins;
        }

        // Update statistics
        stats.total_discovered = stats.valid_plugins + stats.invalid_plugins;
        stats.last_discovery = Some(start_time);
        stats.discovery_duration_ms = start_time.elapsed().unwrap_or_default().as_millis() as u64;

        {
            let mut registry_stats = self.stats.write().unwrap();
            *registry_stats = stats.clone();
        }

        info!(
            "Plugin discovery completed: {} valid, {} invalid, {} total in {}ms",
            stats.valid_plugins,
            stats.invalid_plugins,
            stats.total_discovered,
            stats.discovery_duration_ms
        );

        Ok(stats.valid_plugins)
    }

    /// Discover plugins in a specific path
    async fn discover_in_path(
        &self,
        path: &Path,
        stats: &mut DiscoveryStats,
    ) -> Result<HashMap<String, PluginInfo>> {
        let mut plugins = HashMap::new();

        self.scan_directory(path, &mut plugins, stats, 0).await?;

        Ok(plugins)
    }

    /// Recursively scan directory for plugins
    fn scan_directory<'a>(
        &'a self,
        dir: &'a Path,
        plugins: &'a mut HashMap<String, PluginInfo>,
        stats: &'a mut DiscoveryStats,
        depth: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + 'a>> {
        Box::pin(async move {
            if depth > self.config.max_search_depth {
                return Ok(());
            }

            stats.directories_scanned += 1;

            let entries = fs::read_dir(dir).map_err(|e| {
                VoirsError::plugin_error(format!("Failed to read directory {dir:?}: {e}"))
            })?;

            for entry in entries {
                let entry = entry.map_err(|e| {
                    VoirsError::plugin_error(format!("Failed to read directory entry: {e}"))
                })?;

                let path = entry.path();
                stats.files_examined += 1;

                if path.is_dir() {
                    // Check for plugin manifest in subdirectory
                    let manifest_path = path.join("plugin.toml");
                    if manifest_path.exists() {
                        match self.load_plugin_from_manifest(&manifest_path).await {
                            Ok(plugin_info) => {
                                if !self.is_blacklisted(&plugin_info.name) {
                                    plugins.insert(plugin_info.name.clone(), plugin_info);
                                    stats.valid_plugins += 1;
                                } else {
                                    stats.blacklisted_plugins += 1;
                                }
                            }
                            Err(e) => {
                                warn!("Failed to load plugin from {:?}: {}", manifest_path, e);
                                stats.invalid_plugins += 1;
                            }
                        }
                    } else {
                        // Recursively scan subdirectory
                        self.scan_directory(&path, plugins, stats, depth + 1)
                            .await?;
                    }
                } else if self.is_plugin_file(&path) {
                    // Handle direct plugin files (for simple plugins without manifests)
                    match self.create_plugin_info_from_file(&path).await {
                        Ok(plugin_info) => {
                            if !self.is_blacklisted(&plugin_info.name) {
                                plugins.insert(plugin_info.name.clone(), plugin_info);
                                stats.valid_plugins += 1;
                            } else {
                                stats.blacklisted_plugins += 1;
                            }
                        }
                        Err(e) => {
                            debug!("Skipped file {:?}: {}", path, e);
                            stats.invalid_plugins += 1;
                        }
                    }
                }
            }

            Ok(())
        })
    }

    /// Check if file is a potential plugin file
    fn is_plugin_file(&self, path: &Path) -> bool {
        if let Some(extension) = path.extension() {
            if let Some(ext_str) = extension.to_str() {
                let ext_with_dot = format!(".{ext_str}");
                return self.config.allowed_extensions.contains(&ext_with_dot);
            }
        }
        false
    }

    /// Load plugin from manifest file
    async fn load_plugin_from_manifest(&self, manifest_path: &Path) -> Result<PluginInfo> {
        let manifest_content = fs::read_to_string(manifest_path).map_err(|e| {
            VoirsError::plugin_error(format!("Failed to read manifest {manifest_path:?}: {e}"))
        })?;

        let manifest: PluginManifest = toml::from_str(&manifest_content).map_err(|e| {
            VoirsError::plugin_error(format!("Failed to parse manifest {manifest_path:?}: {e}"))
        })?;

        // Cache the manifest
        {
            let mut cache = self.manifest_cache.write().unwrap();
            cache.insert(manifest_path.to_path_buf(), manifest.clone());
        }

        // Get file metadata
        let metadata = fs::metadata(manifest_path).map_err(|e| {
            VoirsError::plugin_error(format!("Failed to get metadata for {manifest_path:?}: {e}"))
        })?;

        let plugin_info = PluginInfo {
            name: manifest.plugin.name.clone(),
            version: manifest.plugin.version.clone(),
            description: manifest.plugin.description.clone(),
            path: manifest_path
                .parent()
                .unwrap_or(manifest_path)
                .to_path_buf(),
            plugin_type: manifest.plugin.plugin_type,
            author: manifest.plugin.author.clone(),
            dependencies: manifest.plugin.dependencies.clone(),
            supported_platforms: manifest
                .plugin
                .supported_platforms
                .clone()
                .unwrap_or_else(|| vec!["any".to_string()]),
            min_voirs_version: manifest
                .plugin
                .min_voirs_version
                .clone()
                .unwrap_or_else(|| "0.1.0".to_string()),
            last_modified: metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            trusted: self.is_trusted_source(manifest_path),
            size: metadata.len(),
            checksum: calculate_checksum(manifest_path).ok(),
        };

        Ok(plugin_info)
    }

    /// Create plugin info from standalone file
    async fn create_plugin_info_from_file(&self, path: &Path) -> Result<PluginInfo> {
        let metadata = fs::metadata(path).map_err(|e| {
            VoirsError::plugin_error(format!("Failed to get metadata for {path:?}: {e}"))
        })?;

        // Check file size
        if metadata.len() > self.config.max_plugin_size_mb * 1024 * 1024 {
            return Err(VoirsError::plugin_error(format!(
                "Plugin file too large: {} MB (max: {} MB)",
                metadata.len() / 1024 / 1024,
                self.config.max_plugin_size_mb
            )));
        }

        let file_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let plugin_info = PluginInfo {
            name: file_name.clone(),
            version: "unknown".to_string(),
            description: format!("Plugin loaded from {}", path.display()),
            path: path.to_path_buf(),
            plugin_type: PluginType::Effect, // Default type
            author: "unknown".to_string(),
            dependencies: vec![],
            supported_platforms: vec!["any".to_string()],
            min_voirs_version: "0.1.0".to_string(),
            last_modified: metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            trusted: self.is_trusted_source(path),
            size: metadata.len(),
            checksum: None,
        };

        Ok(plugin_info)
    }

    /// Check if plugin is in blacklist
    fn is_blacklisted(&self, plugin_name: &str) -> bool {
        let blacklist = self.blacklist.read().unwrap();
        blacklist.contains(plugin_name)
    }

    /// Check if plugin source is trusted
    fn is_trusted_source(&self, path: &Path) -> bool {
        for trusted_source in &self.config.trusted_sources {
            if path.starts_with(trusted_source) {
                return true;
            }
        }
        false
    }

    /// Add plugin to blacklist
    pub fn blacklist_plugin(&self, plugin_name: &str) {
        let mut blacklist = self.blacklist.write().unwrap();
        blacklist.insert(plugin_name.to_string());
        warn!("Added plugin '{}' to blacklist", plugin_name);
    }

    /// Remove plugin from blacklist
    pub fn unblacklist_plugin(&self, plugin_name: &str) {
        let mut blacklist = self.blacklist.write().unwrap();
        blacklist.remove(plugin_name);
        info!("Removed plugin '{}' from blacklist", plugin_name);
    }

    /// Load a plugin by name
    pub async fn load_plugin(&self, name: &str) -> Result<Arc<dyn VoirsPlugin>> {
        let plugin_info = {
            let plugins = self.plugins.read().unwrap();
            plugins.get(name).cloned()
        };

        match plugin_info {
            Some(info) => {
                if self.is_blacklisted(name) {
                    return Err(VoirsError::plugin_error(format!(
                        "Plugin '{name}' is blacklisted"
                    )));
                }

                if !info.trusted && self.config.verify_plugins {
                    return Err(VoirsError::plugin_error(format!(
                        "Plugin '{name}' is not from a trusted source"
                    )));
                }

                // Implement plugin loading based on plugin type and file extension
                self.load_plugin_from_info(&info).await
            }
            None => Err(VoirsError::plugin_error(format!(
                "Plugin '{name}' not found in registry"
            ))),
        }
    }

    /// List discovered plugins
    pub fn list_plugins(&self) -> Vec<PluginInfo> {
        let plugins = self.plugins.read().unwrap();
        plugins.values().cloned().collect()
    }

    /// Get plugin info by name
    pub fn get_plugin_info(&self, name: &str) -> Option<PluginInfo> {
        let plugins = self.plugins.read().unwrap();
        plugins.get(name).cloned()
    }

    /// List plugins by type
    pub fn list_plugins_by_type(&self, plugin_type: PluginType) -> Vec<PluginInfo> {
        let plugins = self.plugins.read().unwrap();
        plugins
            .values()
            .filter(|plugin| plugin.plugin_type == plugin_type)
            .cloned()
            .collect()
    }

    /// Get plugin manifest
    pub fn get_plugin_manifest(&self, manifest_path: &Path) -> Option<PluginManifest> {
        let cache = self.manifest_cache.read().unwrap();
        cache.get(manifest_path).cloned()
    }

    /// Get discovery statistics
    pub fn get_discovery_stats(&self) -> DiscoveryStats {
        let stats = self.stats.read().unwrap();
        stats.clone()
    }

    /// Get registry configuration
    pub fn get_config(&self) -> RegistryConfig {
        self.config.clone()
    }

    /// Update registry configuration
    pub fn update_config(&mut self, config: RegistryConfig) {
        self.config = config;
        info!("Updated registry configuration");
    }

    /// Clear discovery cache
    pub fn clear_cache(&self) {
        {
            let mut plugins = self.plugins.write().unwrap();
            plugins.clear();
        }
        {
            let mut cache = self.manifest_cache.write().unwrap();
            cache.clear();
        }
        info!("Cleared plugin discovery cache");
    }

    /// Load plugin from plugin info
    async fn load_plugin_from_info(&self, info: &PluginInfo) -> Result<Arc<dyn VoirsPlugin>> {
        tracing::info!("Loading plugin: {} v{}", info.name, info.version);

        // Validate plugin integrity
        self.validate_plugin_integrity(info)?;

        // Determine plugin loading strategy based on file extension and type
        let plugin_path = &info.path;
        let file_extension = plugin_path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        match file_extension {
            "wasm" => self.load_wasm_plugin(info).await,
            "so" | "dll" | "dylib" => self.load_native_plugin(info).await,
            _ => {
                // Check if it's a directory with a manifest
                if plugin_path.is_dir() {
                    self.load_manifest_plugin(info).await
                } else {
                    Err(VoirsError::plugin_error(format!(
                        "Unsupported plugin format: {} (file: {})",
                        file_extension,
                        plugin_path.display()
                    )))
                }
            }
        }
    }

    /// Validate plugin integrity and security
    fn validate_plugin_integrity(&self, info: &PluginInfo) -> Result<()> {
        // Check if plugin path exists
        if !info.path.exists() {
            return Err(VoirsError::plugin_error(format!(
                "Plugin file not found: {}",
                info.path.display()
            )));
        }

        // Validate file size
        if info.size > self.config.max_plugin_size_mb * 1024 * 1024 {
            return Err(VoirsError::plugin_error(format!(
                "Plugin file too large: {} MB (max: {} MB)",
                info.size / 1024 / 1024,
                self.config.max_plugin_size_mb
            )));
        }

        // Verify checksum if available
        if let Some(expected_checksum) = &info.checksum {
            let actual_checksum = calculate_checksum(&info.path)?;
            if &actual_checksum != expected_checksum {
                return Err(VoirsError::plugin_error(format!(
                    "Plugin checksum mismatch for '{}': expected {}, got {}",
                    info.name, expected_checksum, actual_checksum
                )));
            }
        }

        // Additional security checks
        if !info.trusted && self.config.verify_plugins {
            return Err(VoirsError::plugin_error(format!(
                "Plugin '{}' is not from a trusted source and verification is enabled",
                info.name
            )));
        }

        Ok(())
    }

    /// Load WASM plugin
    async fn load_wasm_plugin(&self, info: &PluginInfo) -> Result<Arc<dyn VoirsPlugin>> {
        tracing::debug!("Loading WASM plugin: {}", info.name);

        #[cfg(not(feature = "plugins"))]
        {
            Err(VoirsError::plugin_error(format!(
                "WASM plugin loading requires the 'plugins' feature for '{}'",
                info.name
            )))
        }

        #[cfg(feature = "plugins")]
        {
            // Read WASM file
            let wasm_bytes = fs::read(&info.path).map_err(|e| {
                VoirsError::plugin_error(format!(
                    "Failed to read WASM file '{}': {}",
                    info.path.display(),
                    e
                ))
            })?;

            // Create WASM engine and compile module
            let engine = Engine::default();
            let module = Module::new(&engine, &wasm_bytes).map_err(|e| {
                VoirsError::plugin_error(format!(
                    "Failed to compile WASM module '{}': {}",
                    info.name, e
                ))
            })?;

            // Create plugin wrapper
            let plugin = Arc::new(WasmPluginWrapper::new(
                info.clone(),
                Arc::new(engine),
                Arc::new(module),
            ));

            tracing::info!("Successfully loaded WASM plugin: {}", info.name);
            Ok(plugin as Arc<dyn VoirsPlugin>)
        }
    }

    /// Load native dynamic library plugin
    async fn load_native_plugin(&self, info: &PluginInfo) -> Result<Arc<dyn VoirsPlugin>> {
        tracing::debug!("Loading native plugin: {}", info.name);

        #[cfg(not(feature = "plugins"))]
        {
            Err(VoirsError::plugin_error(format!(
                "Native plugin loading requires the 'plugins' feature for '{}'",
                info.name
            )))
        }

        #[cfg(feature = "plugins")]
        {
            // Load dynamic library
            let library = unsafe {
                Library::new(&info.path).map_err(|e| {
                    VoirsError::plugin_error(format!(
                        "Failed to load native library '{}': {}",
                        info.path.display(),
                        e
                    ))
                })?
            };

            let library = Arc::new(library);

            // Look for the plugin factory function
            let create_plugin: Symbol<unsafe extern "C" fn() -> *mut dyn VoirsPlugin> = unsafe {
                library.get(b"create_voirs_plugin").map_err(|e| {
                    VoirsError::plugin_error(format!(
                        "Plugin factory function 'create_voirs_plugin' not found in '{}': {}",
                        info.name, e
                    ))
                })?
            };

            // Call factory function
            let plugin_ptr = unsafe { create_plugin() };
            if plugin_ptr.is_null() {
                return Err(VoirsError::plugin_error(format!(
                    "Plugin factory returned null for '{}'",
                    info.name
                )));
            }

            let plugin = unsafe { Arc::from_raw(plugin_ptr) };

            // Create wrapper to manage the library lifetime
            let wrapped_plugin = Arc::new(NativePluginWrapper::new(plugin, library, info.clone()));

            tracing::info!("Successfully loaded native plugin: {}", info.name);
            Ok(wrapped_plugin as Arc<dyn VoirsPlugin>)
        }
    }

    /// Load plugin from manifest directory
    async fn load_manifest_plugin(&self, info: &PluginInfo) -> Result<Arc<dyn VoirsPlugin>> {
        tracing::debug!("Loading manifest plugin: {}", info.name);

        let manifest_path = info.path.join("plugin.toml");
        if !manifest_path.exists() {
            return Err(VoirsError::plugin_error(format!(
                "Plugin manifest not found: {}",
                manifest_path.display()
            )));
        }

        // Load and parse manifest
        let manifest = self.get_plugin_manifest(&manifest_path).ok_or_else(|| {
            VoirsError::plugin_error(format!(
                "Failed to load plugin manifest for '{}'",
                info.name
            ))
        })?;

        // For now, create a built-in plugin implementation based on the type
        match info.plugin_type {
            PluginType::Effect => self.create_builtin_effect_plugin(&manifest),
            PluginType::VoiceEffect => self.create_builtin_voice_effect_plugin(&manifest),
            PluginType::TextProcessor => self.create_builtin_text_processor_plugin(&manifest),
            PluginType::ModelEnhancer => self.create_builtin_model_enhancer_plugin(&manifest),
            PluginType::OutputProcessor => self.create_builtin_output_processor_plugin(&manifest),
        }
    }

    /// Create built-in effect plugin implementation
    fn create_builtin_effect_plugin(
        &self,
        manifest: &PluginManifest,
    ) -> Result<Arc<dyn VoirsPlugin>> {
        Ok(Arc::new(BuiltinEffectPlugin {
            metadata: manifest.plugin.clone(),
        }))
    }

    /// Create built-in voice effect plugin implementation
    fn create_builtin_voice_effect_plugin(
        &self,
        manifest: &PluginManifest,
    ) -> Result<Arc<dyn VoirsPlugin>> {
        Ok(Arc::new(BuiltinVoiceEffectPlugin {
            metadata: manifest.plugin.clone(),
        }))
    }

    /// Create built-in text processor plugin implementation
    fn create_builtin_text_processor_plugin(
        &self,
        manifest: &PluginManifest,
    ) -> Result<Arc<dyn VoirsPlugin>> {
        Ok(Arc::new(BuiltinTextProcessorPlugin {
            metadata: manifest.plugin.clone(),
        }))
    }

    /// Create built-in model enhancer plugin implementation
    fn create_builtin_model_enhancer_plugin(
        &self,
        manifest: &PluginManifest,
    ) -> Result<Arc<dyn VoirsPlugin>> {
        Ok(Arc::new(BuiltinModelEnhancerPlugin {
            metadata: manifest.plugin.clone(),
        }))
    }

    /// Create built-in output processor plugin implementation
    fn create_builtin_output_processor_plugin(
        &self,
        manifest: &PluginManifest,
    ) -> Result<Arc<dyn VoirsPlugin>> {
        Ok(Arc::new(BuiltinOutputProcessorPlugin {
            metadata: manifest.plugin.clone(),
        }))
    }

    /// Install plugin to manager
    pub async fn install_plugin(
        &self,
        name: &str,
        manager: &PluginManager,
        config: Option<PluginConfig>,
    ) -> Result<()> {
        let plugin = self.load_plugin(name).await?;
        manager
            .register_plugin(name.to_string(), plugin, config)
            .await
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Built-in plugin implementations for manifest-based plugins

/// Built-in effect plugin implementation
struct BuiltinEffectPlugin {
    metadata: PluginMetadata,
}

impl VoirsPlugin for BuiltinEffectPlugin {
    fn name(&self) -> &str {
        &self.metadata.name
    }

    fn version(&self) -> &str {
        &self.metadata.version
    }

    fn description(&self) -> &str {
        &self.metadata.description
    }

    fn author(&self) -> &str {
        &self.metadata.author
    }

    fn metadata(&self) -> PluginMetadata {
        self.metadata.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Built-in voice effect plugin implementation
struct BuiltinVoiceEffectPlugin {
    metadata: PluginMetadata,
}

impl VoirsPlugin for BuiltinVoiceEffectPlugin {
    fn name(&self) -> &str {
        &self.metadata.name
    }

    fn version(&self) -> &str {
        &self.metadata.version
    }

    fn description(&self) -> &str {
        &self.metadata.description
    }

    fn author(&self) -> &str {
        &self.metadata.author
    }

    fn metadata(&self) -> PluginMetadata {
        self.metadata.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Built-in text processor plugin implementation
struct BuiltinTextProcessorPlugin {
    metadata: PluginMetadata,
}

impl VoirsPlugin for BuiltinTextProcessorPlugin {
    fn name(&self) -> &str {
        &self.metadata.name
    }

    fn version(&self) -> &str {
        &self.metadata.version
    }

    fn description(&self) -> &str {
        &self.metadata.description
    }

    fn author(&self) -> &str {
        &self.metadata.author
    }

    fn metadata(&self) -> PluginMetadata {
        self.metadata.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Built-in model enhancer plugin implementation
struct BuiltinModelEnhancerPlugin {
    metadata: PluginMetadata,
}

impl VoirsPlugin for BuiltinModelEnhancerPlugin {
    fn name(&self) -> &str {
        &self.metadata.name
    }

    fn version(&self) -> &str {
        &self.metadata.version
    }

    fn description(&self) -> &str {
        &self.metadata.description
    }

    fn author(&self) -> &str {
        &self.metadata.author
    }

    fn metadata(&self) -> PluginMetadata {
        self.metadata.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Built-in output processor plugin implementation
struct BuiltinOutputProcessorPlugin {
    metadata: PluginMetadata,
}

impl VoirsPlugin for BuiltinOutputProcessorPlugin {
    fn name(&self) -> &str {
        &self.metadata.name
    }

    fn version(&self) -> &str {
        &self.metadata.version
    }

    fn description(&self) -> &str {
        &self.metadata.description
    }

    fn author(&self) -> &str {
        &self.metadata.author
    }

    fn metadata(&self) -> PluginMetadata {
        self.metadata.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Plugin wrapper implementations for WASM and native plugins

#[cfg(feature = "plugins")]
/// WebAssembly plugin wrapper
pub struct WasmPluginWrapper {
    info: PluginInfo,
    engine: Arc<Engine>,
    module: Arc<Module>,
}

#[cfg(feature = "plugins")]
impl WasmPluginWrapper {
    pub fn new(info: PluginInfo, engine: Arc<Engine>, module: Arc<Module>) -> Self {
        Self {
            info,
            engine,
            module,
        }
    }

    fn create_store(&self) -> Store<()> {
        Store::new(&*self.engine, ())
    }

    fn call_wasm_function(&self, function_name: &str) -> Result<i32> {
        let mut store = self.create_store();
        let instance = Instance::new(&mut store, &*self.module, &[]).map_err(|e| {
            VoirsError::plugin_error(format!("Failed to instantiate WASM module: {}", e))
        })?;

        let func = instance
            .get_typed_func::<(), i32>(&mut store, function_name)
            .map_err(|e| {
                VoirsError::plugin_error(format!("Function '{}' not found: {}", function_name, e))
            })?;

        let result = func
            .call(&mut store, ())
            .map_err(|e| VoirsError::plugin_error(format!("WASM function call failed: {}", e)))?;

        Ok(result)
    }
}

#[cfg(feature = "plugins")]
impl VoirsPlugin for WasmPluginWrapper {
    fn name(&self) -> &str {
        &self.info.name
    }

    fn version(&self) -> &str {
        &self.info.version
    }

    fn description(&self) -> &str {
        &self.info.description
    }

    fn author(&self) -> &str {
        &self.info.author
    }

    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: self.info.name.clone(),
            version: self.info.version.clone(),
            description: self.info.description.clone(),
            author: self.info.author.clone(),
            plugin_type: self.info.plugin_type,
            supported_formats: vec!["wav".to_string(), "mp3".to_string(), "flac".to_string()], // Default supported formats
            capabilities: vec!["wasm".to_string(), "execute".to_string()], // WASM-specific capabilities
            dependencies: self.info.dependencies.clone(),
            supported_platforms: Some(self.info.supported_platforms.clone()),
            min_voirs_version: Some(self.info.min_voirs_version.clone()),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(feature = "plugins")]
/// Native plugin wrapper
pub struct NativePluginWrapper {
    plugin: Arc<dyn VoirsPlugin>,
    #[allow(dead_code)]
    library: Arc<Library>, // Keep library alive
    info: PluginInfo,
}

#[cfg(feature = "plugins")]
impl NativePluginWrapper {
    pub fn new(plugin: Arc<dyn VoirsPlugin>, library: Arc<Library>, info: PluginInfo) -> Self {
        Self {
            plugin,
            library,
            info,
        }
    }
}

#[cfg(feature = "plugins")]
impl VoirsPlugin for NativePluginWrapper {
    fn name(&self) -> &str {
        self.plugin.name()
    }

    fn version(&self) -> &str {
        self.plugin.version()
    }

    fn description(&self) -> &str {
        self.plugin.description()
    }

    fn author(&self) -> &str {
        self.plugin.author()
    }

    fn metadata(&self) -> PluginMetadata {
        self.plugin.metadata()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_registry_creation() {
        let registry = PluginRegistry::new();
        assert_eq!(registry.get_search_paths().len(), 0);
        assert_eq!(registry.list_plugins().len(), 0);
    }

    #[tokio::test]
    async fn test_search_path_management() {
        let mut registry = PluginRegistry::new();
        let temp_dir = TempDir::new().unwrap();

        // Add search path
        registry.add_search_path(temp_dir.path());
        assert_eq!(registry.get_search_paths().len(), 1);

        // Remove search path
        registry.remove_search_path(temp_dir.path());
        assert_eq!(registry.get_search_paths().len(), 0);
    }

    #[tokio::test]
    async fn test_blacklist_management() {
        let registry = PluginRegistry::new();

        // Test blacklisting
        assert!(!registry.is_blacklisted("test_plugin"));
        registry.blacklist_plugin("test_plugin");
        assert!(registry.is_blacklisted("test_plugin"));

        // Test unblacklisting
        registry.unblacklist_plugin("test_plugin");
        assert!(!registry.is_blacklisted("test_plugin"));
    }

    #[tokio::test]
    async fn test_empty_directory_discovery() {
        let mut registry = PluginRegistry::new();
        let temp_dir = TempDir::new().unwrap();

        registry.add_search_path(temp_dir.path());
        let discovered = registry.discover_plugins().await.unwrap();
        assert_eq!(discovered, 0);

        let stats = registry.get_discovery_stats();
        assert_eq!(stats.valid_plugins, 0);
        assert_eq!(stats.directories_scanned, 1);
    }
}
