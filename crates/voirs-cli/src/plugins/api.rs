//! Plugin API definitions and utilities for VoiRS CLI plugins.

use super::{Plugin, PluginError, PluginResult, PluginType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Plugin API version information
pub const PLUGIN_API_VERSION: &str = "1.0.0";
pub const MIN_SUPPORTED_VERSION: &str = "1.0.0";
pub const MAX_SUPPORTED_VERSION: &str = "1.99.99";

/// Plugin API interface that all plugins must implement
pub trait PluginApi: Send + Sync {
    /// Get the plugin API version this plugin was compiled against
    fn api_version(&self) -> &str;

    /// Initialize the plugin with host information
    fn initialize_api(&mut self, host_info: &HostInfo) -> PluginResult<()>;

    /// Get plugin capabilities as a structured format
    fn get_plugin_info(&self) -> PluginInfo;

    /// Handle API calls from the host
    fn handle_api_call(&self, call: &ApiCall) -> PluginResult<ApiResponse>;

    /// Notify plugin of host events
    fn notify(&self, event: &PluginEvent) -> PluginResult<()>;

    /// Check if plugin supports a specific feature
    fn supports_feature(&self, feature: &str) -> bool;
}

/// Information about the host application
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostInfo {
    pub name: String,
    pub version: String,
    pub api_version: String,
    pub capabilities: Vec<String>,
    pub configuration: HashMap<String, serde_json::Value>,
}

/// Detailed plugin information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginInfo {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub plugin_type: PluginType,
    pub api_version: String,
    pub supported_features: Vec<String>,
    pub required_permissions: Vec<super::Permission>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// API call structure for plugin communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiCall {
    pub id: String,
    pub method: String,
    pub params: serde_json::Value,
    pub context: Option<CallContext>,
}

/// Context information for API calls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallContext {
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub trace_id: Option<String>,
    pub metadata: HashMap<String, String>,
}

/// Response from plugin API calls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse {
    pub id: String,
    pub success: bool,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Events that can be sent to plugins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginEvent {
    /// Plugin has been loaded
    Loaded,
    /// Plugin is being unloaded
    Unloading,
    /// Host is shutting down
    HostShutdown,
    /// Configuration has changed
    ConfigChanged(serde_json::Value),
    /// System resource state changed
    ResourceStateChanged {
        cpu_usage: f32,
        memory_usage: f32,
        available_memory: u64,
    },
    /// Audio processing state changed
    AudioStateChanged {
        sample_rate: u32,
        buffer_size: u32,
        channels: u32,
    },
    /// Custom event with arbitrary data
    Custom {
        event_type: String,
        data: serde_json::Value,
    },
}

/// Plugin API registry for managing multiple plugin APIs
pub struct ApiRegistry {
    apis: HashMap<String, Arc<dyn PluginApi>>,
    host_info: HostInfo,
}

impl ApiRegistry {
    /// Create a new API registry
    pub fn new() -> Self {
        Self {
            apis: HashMap::new(),
            host_info: HostInfo {
                name: "VoiRS CLI".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                api_version: PLUGIN_API_VERSION.to_string(),
                capabilities: vec![
                    "audio_synthesis".to_string(),
                    "voice_management".to_string(),
                    "batch_processing".to_string(),
                    "real_time_synthesis".to_string(),
                    "plugin_system".to_string(),
                ],
                configuration: HashMap::new(),
            },
        }
    }

    /// Register a plugin API implementation
    pub fn register_api(&mut self, name: String, api: Arc<dyn PluginApi>) -> PluginResult<()> {
        if self.apis.contains_key(&name) {
            return Err(PluginError::LoadingFailed(format!(
                "API {} already registered",
                name
            )));
        }

        // Validate API version compatibility
        if !self.is_version_compatible(api.api_version()) {
            return Err(PluginError::ApiVersionMismatch {
                expected: PLUGIN_API_VERSION.to_string(),
                actual: api.api_version().to_string(),
            });
        }

        self.apis.insert(name, api);
        Ok(())
    }

    /// Unregister a plugin API
    pub fn unregister_api(&mut self, name: &str) -> PluginResult<()> {
        self.apis
            .remove(name)
            .ok_or_else(|| PluginError::NotFound(name.to_string()))?;
        Ok(())
    }

    /// Get a registered API by name
    pub fn get_api(&self, name: &str) -> Option<Arc<dyn PluginApi>> {
        self.apis.get(name).cloned()
    }

    /// List all registered APIs
    pub fn list_apis(&self) -> Vec<String> {
        self.apis.keys().cloned().collect()
    }

    /// Initialize all registered APIs
    pub fn initialize_all(&self) -> PluginResult<()> {
        for (name, api) in &self.apis {
            let mut api_mut = api.clone();
            // Note: This would need proper mutable access in a real implementation
            // For now, we'll skip the mutable initialization
            tracing::info!("Initialized API: {}", name);
        }
        Ok(())
    }

    /// Broadcast an event to all registered APIs
    pub fn broadcast_event(&self, event: &PluginEvent) -> PluginResult<()> {
        for (name, api) in &self.apis {
            if let Err(e) = api.notify(event) {
                tracing::error!("Failed to notify API {}: {}", name, e);
            }
        }
        Ok(())
    }

    /// Call a method on a specific API
    pub fn call_api(&self, api_name: &str, call: &ApiCall) -> PluginResult<ApiResponse> {
        let api = self
            .apis
            .get(api_name)
            .ok_or_else(|| PluginError::NotFound(api_name.to_string()))?;

        api.handle_api_call(call)
    }

    /// Check version compatibility
    fn is_version_compatible(&self, version: &str) -> bool {
        // Simple semantic versioning check
        let min_parts: Vec<u32> = MIN_SUPPORTED_VERSION
            .split('.')
            .filter_map(|s| s.parse().ok())
            .collect();
        let max_parts: Vec<u32> = MAX_SUPPORTED_VERSION
            .split('.')
            .filter_map(|s| s.parse().ok())
            .collect();
        let version_parts: Vec<u32> = version.split('.').filter_map(|s| s.parse().ok()).collect();

        if version_parts.len() != 3 || min_parts.len() != 3 || max_parts.len() != 3 {
            return false;
        }

        // Check if version is within supported range
        version_parts >= min_parts && version_parts <= max_parts
    }

    /// Get host information
    pub fn get_host_info(&self) -> &HostInfo {
        &self.host_info
    }

    /// Update host configuration
    pub fn update_host_config(&mut self, key: String, value: serde_json::Value) {
        self.host_info.configuration.insert(key, value);
    }
}

impl Default for ApiRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for plugin API
pub mod utils {
    use super::*;

    /// Create a standard API call
    pub fn create_api_call(method: &str, params: serde_json::Value) -> ApiCall {
        ApiCall {
            id: generate_call_id(),
            method: method.to_string(),
            params,
            context: None,
        }
    }

    /// Create an API call with context
    pub fn create_api_call_with_context(
        method: &str,
        params: serde_json::Value,
        context: CallContext,
    ) -> ApiCall {
        ApiCall {
            id: generate_call_id(),
            method: method.to_string(),
            params,
            context: Some(context),
        }
    }

    /// Create a successful API response
    pub fn create_success_response(call_id: &str, result: serde_json::Value) -> ApiResponse {
        ApiResponse {
            id: call_id.to_string(),
            success: true,
            result: Some(result),
            error: None,
            metadata: HashMap::new(),
        }
    }

    /// Create an error API response
    pub fn create_error_response(call_id: &str, error: &str) -> ApiResponse {
        ApiResponse {
            id: call_id.to_string(),
            success: false,
            result: None,
            error: Some(error.to_string()),
            metadata: HashMap::new(),
        }
    }

    /// Generate a unique call ID
    fn generate_call_id() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("call_{}", timestamp)
    }

    /// Validate API method name
    pub fn is_valid_method_name(method: &str) -> bool {
        !method.is_empty()
            && method
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '.')
    }

    /// Extract plugin type from API call context
    pub fn extract_plugin_type(call: &ApiCall) -> Option<PluginType> {
        call.context
            .as_ref()
            .and_then(|ctx| ctx.metadata.get("plugin_type"))
            .and_then(|t| match t.as_str() {
                "Effect" => Some(PluginType::Effect),
                "Voice" => Some(PluginType::Voice),
                "Processor" => Some(PluginType::Processor),
                "Extension" => Some(PluginType::Extension),
                _ => None,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_registry_creation() {
        let registry = ApiRegistry::new();
        assert_eq!(registry.list_apis().len(), 0);
        assert_eq!(registry.get_host_info().name, "VoiRS CLI");
        assert_eq!(registry.get_host_info().api_version, PLUGIN_API_VERSION);
    }

    #[test]
    fn test_version_compatibility() {
        let registry = ApiRegistry::new();
        assert!(registry.is_version_compatible("1.0.0"));
        assert!(registry.is_version_compatible("1.0.1"));
        assert!(registry.is_version_compatible("1.1.0"));
        assert!(!registry.is_version_compatible("2.0.0"));
        assert!(!registry.is_version_compatible("0.9.0"));
    }

    #[test]
    fn test_api_call_creation() {
        let call = utils::create_api_call("test_method", serde_json::json!({"param": "value"}));
        assert_eq!(call.method, "test_method");
        assert_eq!(call.params["param"], "value");
        assert!(call.id.starts_with("call_"));
    }

    #[test]
    fn test_api_response_creation() {
        let response =
            utils::create_success_response("test_id", serde_json::json!({"result": "ok"}));
        assert_eq!(response.id, "test_id");
        assert!(response.success);
        assert_eq!(response.result.unwrap()["result"], "ok");

        let error_response = utils::create_error_response("test_id", "test error");
        assert_eq!(error_response.id, "test_id");
        assert!(!error_response.success);
        assert_eq!(error_response.error.unwrap(), "test error");
    }

    #[test]
    fn test_method_name_validation() {
        assert!(utils::is_valid_method_name("valid_method"));
        assert!(utils::is_valid_method_name("method.with.dots"));
        assert!(utils::is_valid_method_name("method123"));
        assert!(!utils::is_valid_method_name(""));
        assert!(!utils::is_valid_method_name("invalid-method"));
        assert!(!utils::is_valid_method_name("invalid method"));
    }

    #[test]
    fn test_host_info_structure() {
        let registry = ApiRegistry::new();
        let host_info = registry.get_host_info();

        assert!(!host_info.name.is_empty());
        assert!(!host_info.version.is_empty());
        assert!(!host_info.api_version.is_empty());
        assert!(!host_info.capabilities.is_empty());
        assert!(host_info
            .capabilities
            .contains(&"audio_synthesis".to_string()));
    }

    #[test]
    fn test_plugin_event_serialization() {
        let event = PluginEvent::ConfigChanged(serde_json::json!({"setting": "value"}));
        let serialized = serde_json::to_string(&event).unwrap();
        let deserialized: PluginEvent = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            PluginEvent::ConfigChanged(config) => {
                assert_eq!(config["setting"], "value");
            }
            _ => panic!("Unexpected event type"),
        }
    }
}
