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

/// Type alias for the plugin storage map
type PluginMap = RwLock<HashMap<String, Arc<RwLock<Box<dyn Plugin>>>>>;

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

    /// A load was requested for a plugin kind that has no real
    /// implementation yet (e.g. native `.dll`/`.so`/`.dylib` plugins,
    /// pending a versioned C ABI). Distinct from `LoadingFailed`, which
    /// means loading was attempted and failed; `NotSupported` means loading
    /// was never attempted because there is nothing real to run --
    /// callers must never receive a mock in this case.
    #[error("Plugin loading not supported: {0}")]
    NotSupported(String),

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
    plugins: PluginMap,
    plugin_info: RwLock<HashMap<String, PluginInfo>>,
    plugin_directories: Vec<PathBuf>,
    api_version: String,
    security_enabled: bool,
    /// Shared WASM engine used to compile every `.wasm` plugin this manager
    /// loads. One engine per manager (not per plugin) so compiled-code
    /// caches and configuration are shared across loads.
    wasm_engine: Arc<wasmtime::Engine>,
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
            wasm_engine: Arc::new(wasmtime::Engine::default()),
        }
    }

    pub fn add_plugin_directory<P: AsRef<Path>>(&mut self, path: P) {
        self.plugin_directories.push(path.as_ref().to_path_buf());
    }

    /// Scan `plugin_directories` for `plugin.json` manifests and register
    /// each discovered plugin into `self.plugin_info` (keyed by
    /// `manifest.name`) so subsequent `load_plugin(name)`/`get_plugin_info`/
    /// `list_plugins` calls can find it -- without this registration step
    /// `load_plugin` would always return `PluginError::NotFound` for a
    /// freshly discovered plugin, no matter what `load_plugin_from_path`
    /// does. A plugin that's already tracked (e.g. because it was
    /// previously loaded) keeps its existing `loaded`/`load_count`/
    /// `last_error` bookkeeping; re-discovery only refreshes its
    /// manifest/path.
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

        {
            let mut info_guard = self.plugin_info.write().await;
            for info in &discovered {
                let name = info.manifest.name.clone();
                info_guard
                    .entry(name)
                    .and_modify(|existing| {
                        existing.manifest = info.manifest.clone();
                        existing.path = info.path.clone();
                    })
                    .or_insert_with(|| info.clone());
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

    /// Actually load the plugin `manifest` names, dispatching on its entry
    /// point's file extension exactly like `loader::PluginLoader` does:
    /// `.wasm` is compiled and run through `wasmtime` (real, sandboxed
    /// dynamic loading), `.dll`/`.so`/`.dylib` fails closed with a typed
    /// `PluginError::NotSupported` (see `loader::native_plugin_unsupported`
    /// for why -- no unsound FFI, no mock), and anything else falls back to
    /// a real builtin implementation selected by `manifest.plugin_type`
    /// (`loader::build_builtin_plugin`). `path` is the plugin's directory
    /// (as discovered by `discover_plugins`); the entry point file itself
    /// must exist on disk for any of these branches to be attempted.
    async fn load_plugin_from_path(
        &self,
        path: &Path,
        manifest: &PluginManifest,
    ) -> PluginResult<Box<dyn Plugin>> {
        let entry_path = path.join(&manifest.entry_point);
        if !entry_path.exists() {
            return Err(PluginError::LoadingFailed(format!(
                "entry point '{}' not found for plugin '{}' (expected at '{}')",
                manifest.entry_point,
                manifest.name,
                entry_path.display()
            )));
        }

        let extension = entry_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase());

        match extension.as_deref() {
            Some("wasm") => {
                loader::build_wasm_plugin(&entry_path, manifest, &self.wasm_engine).await
            }
            Some("dll") | Some("so") | Some("dylib") => Err(loader::native_plugin_unsupported(
                &entry_path,
                &manifest.name,
            )),
            _ => Ok(loader::build_builtin_plugin(manifest)),
        }
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

// Mock plugin, for tests only. `PluginManager::load_plugin_from_path` never
// constructs this outside `#[cfg(test)]` -- a load that can't be satisfied
// for real fails closed with a typed `PluginError` instead (see
// `load_plugin_from_path` and `loader::native_plugin_unsupported`).
#[cfg(test)]
struct MockPlugin {
    name: String,
    version: String,
    description: String,
}

#[cfg(test)]
impl MockPlugin {
    fn new() -> Self {
        Self {
            name: "mock-plugin".to_string(),
            version: "1.0.0".to_string(),
            description: "Mock plugin for testing".to_string(),
        }
    }
}

#[cfg(test)]
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

    /// A tiny WebAssembly Text (WAT) module used to prove
    /// `PluginManager`/`loader::build_wasm_plugin` perform real dynamic
    /// compilation and execution: two exports return two different,
    /// genuinely wasm-computed values (`get_a` returns a raw constant,
    /// `compute` returns the result of real `i32.add`), and no
    /// `initialize`/`cleanup` export is defined (exercising the
    /// "optional export absent" path). `wasmtime::Module::new` (used by
    /// `build_wasm_plugin`) accepts WAT text directly since this crate
    /// builds wasmtime with the `wat` feature, so no separate compilation
    /// step or extra tooling is needed to produce a real `.wasm` file.
    const WAT_TEST_MODULE: &str = r#"
        (module
            (func (export "get_a") (result i32) (i32.const 111))
            (func (export "compute") (result i32) (i32.add (i32.const 40) (i32.const 2)))
        )
    "#;

    fn unique_temp_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "voirs_plugins_test_{label}_{}_{}",
            std::process::id(),
            fastrand::u64(..)
        ))
    }

    fn test_manifest(plugin_type: PluginType, entry_point: &str) -> PluginManifest {
        PluginManifest {
            name: "test-plugin".to_string(),
            version: "1.0.0".to_string(),
            description: "Plugin under test".to_string(),
            author: "test".to_string(),
            api_version: "1.0.0".to_string(),
            plugin_type,
            entry_point: entry_point.to_string(),
            dependencies: Vec::new(),
            permissions: Vec::new(),
            configuration: None,
        }
    }

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
        // Should not fail even if no plugins found. The default plugin
        // directories (e.g. `./plugins`) may or may not exist depending on
        // where tests run from, so no assumption is made about `plugins`'
        // contents here -- `test_discover_plugins_registers_into_plugin_info`
        // below exercises the registration behavior against a controlled
        // directory.
        let _ = plugins;
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

    #[tokio::test]
    async fn test_load_plugin_from_path_missing_entry_point_errors() {
        let manager = PluginManager::new();
        let dir = unique_temp_dir("missing_entry");
        std::fs::create_dir_all(&dir).unwrap();

        let manifest = test_manifest(PluginType::Extension, "does-not-exist.wasm");
        let result = manager.load_plugin_from_path(&dir, &manifest).await;

        match result {
            Err(PluginError::LoadingFailed(msg)) => {
                assert!(msg.contains("does-not-exist.wasm"), "got: {msg}");
            }
            Ok(_) => panic!("expected LoadingFailed, got Ok"),
            Err(other) => panic!("expected LoadingFailed, got {other}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_load_plugin_from_path_native_extension_fails_closed() {
        let manager = PluginManager::new();
        let dir = unique_temp_dir("native_plugin");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("plugin.so"), b"not a real shared library").unwrap();

        let manifest = test_manifest(PluginType::Extension, "plugin.so");
        let result = manager.load_plugin_from_path(&dir, &manifest).await;

        match result {
            Err(PluginError::NotSupported(msg)) => {
                assert!(
                    msg.contains("voirs_plugin_entry"),
                    "error should name the missing ABI contract, got: {msg}"
                );
            }
            Ok(_) => panic!("expected NotSupported (fail-closed, no mock), got Ok"),
            Err(other) => panic!("expected NotSupported (fail-closed, no mock), got {other}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_load_plugin_from_path_builtin_fallback_is_real_not_mock() {
        let manager = PluginManager::new();
        let dir = unique_temp_dir("builtin_plugin");
        std::fs::create_dir_all(&dir).unwrap();
        // Builtin plugins are selected by `plugin_type` alone; the entry
        // point file just has to exist (matching `loader::PluginLoader`'s
        // pre-existing convention that every manifest names a real file).
        std::fs::write(dir.join("marker.txt"), b"builtin plugin marker").unwrap();

        let manifest = test_manifest(PluginType::Processor, "marker.txt");
        let plugin = manager
            .load_plugin_from_path(&dir, &manifest)
            .await
            .expect("builtin plugin should load");

        assert_eq!(plugin.plugin_type(), PluginType::Processor);

        // The real `TextProcessorPlugin` normalizes full-width ASCII to
        // half-width via `normalize`; `MockPlugin` has no such command and
        // only understands "test", so this genuinely distinguishes the two.
        let result = plugin
            .execute(
                "normalize",
                &serde_json::json!({"text": "\u{FF21}\u{FF22}\u{FF23}"}),
            )
            .expect("real TextProcessorPlugin should handle 'normalize'");
        assert_eq!(result["normalized_text"], serde_json::json!("ABC"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_load_plugin_from_path_wasm_real_dynamic_execution() {
        let manager = PluginManager::new();
        let dir = unique_temp_dir("wasm_plugin");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("plugin.wasm"), WAT_TEST_MODULE).unwrap();

        let manifest = test_manifest(PluginType::Extension, "plugin.wasm");
        let plugin = manager
            .load_plugin_from_path(&dir, &manifest)
            .await
            .expect("wasm plugin should compile and load");

        // Two different exports must yield two different, genuinely
        // wasm-computed results -- not a canned Rust-side response.
        let result_a = plugin
            .execute("get_a", &serde_json::json!({}))
            .expect("get_a export should execute");
        assert_eq!(result_a["result"], serde_json::json!(111));

        let result_b = plugin
            .execute("compute", &serde_json::json!({}))
            .expect("compute export should execute");
        assert_eq!(result_b["result"], serde_json::json!(42));

        // A command with no matching export is a real error, not a
        // fabricated success.
        assert!(plugin
            .execute("does_not_exist", &serde_json::json!({}))
            .is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_discover_plugins_registers_into_plugin_info() {
        let mut manager = PluginManager::new();
        let root = unique_temp_dir("discover_root");
        let plugin_dir = root.join("hello-plugin");
        std::fs::create_dir_all(&plugin_dir).unwrap();
        std::fs::write(plugin_dir.join("marker.txt"), b"marker").unwrap();

        let manifest = PluginManifest {
            name: "hello-plugin".to_string(),
            version: "1.0.0".to_string(),
            description: "A discoverable plugin".to_string(),
            author: "test".to_string(),
            api_version: "1.0.0".to_string(),
            plugin_type: PluginType::Extension,
            entry_point: "marker.txt".to_string(),
            dependencies: Vec::new(),
            permissions: Vec::new(),
            configuration: None,
        };
        std::fs::write(
            plugin_dir.join("plugin.json"),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();

        manager.add_plugin_directory(&root);
        let discovered = manager.discover_plugins().await.unwrap();
        assert_eq!(discovered.len(), 1);
        assert_eq!(discovered[0].manifest.name, "hello-plugin");

        // `discover_plugins` must have registered the plugin into
        // `plugin_info` for `load_plugin`/`get_plugin_info` to find it --
        // without this, `load_plugin` always fails with `NotFound`
        // regardless of what `load_plugin_from_path` does.
        let info = manager
            .get_plugin_info("hello-plugin")
            .await
            .expect("discovered plugin should be registered");
        assert!(!info.loaded);

        manager
            .load_plugin("hello-plugin")
            .await
            .expect("load_plugin should succeed end-to-end through discover -> load");

        let info_after = manager.get_plugin_info("hello-plugin").await.unwrap();
        assert!(info_after.loaded);
        assert_eq!(info_after.load_count, 1);

        let exec_result = manager
            .execute_plugin(
                "hello-plugin",
                "safe_filename",
                &serde_json::json!({"filename": "a/b?.wav"}),
            )
            .await
            .expect("real UtilityExtensionPlugin should handle 'safe_filename'");
        assert_eq!(exec_result["safe_filename"], serde_json::json!("a-b-.wav"));

        let _ = std::fs::remove_dir_all(&root);
    }
}
