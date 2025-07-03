//! Backend registry system for dynamic G2P backend management.

use crate::{G2p, G2pError, LanguageCode, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, warn};

/// Backend registration information
#[derive(Debug, Clone)]
pub struct BackendInfo {
    /// Backend identifier
    pub id: String,
    /// Backend name
    pub name: String,
    /// Backend description
    pub description: String,
    /// Supported languages
    pub supported_languages: Vec<LanguageCode>,
    /// Backend priority (higher values have higher priority)
    pub priority: u32,
    /// Whether this backend is enabled
    pub enabled: bool,
}

/// Backend factory function type
pub type BackendFactory = dyn Fn() -> Result<Box<dyn G2p>> + Send + Sync;

/// Backend registry for managing multiple G2P backends
pub struct BackendRegistry {
    /// Registered backends
    backends: HashMap<String, Arc<BackendFactory>>,
    /// Backend metadata
    backend_info: HashMap<String, BackendInfo>,
    /// Language to backend mapping (sorted by priority)
    language_backends: HashMap<LanguageCode, Vec<String>>,
    /// Default backend chain
    default_backends: Vec<String>,
    /// Round-robin counters for load balancing
    load_balance_counters: HashMap<String, std::sync::atomic::AtomicUsize>,
}

impl BackendRegistry {
    /// Create a new backend registry
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
            backend_info: HashMap::new(),
            language_backends: HashMap::new(),
            default_backends: Vec::new(),
            load_balance_counters: HashMap::new(),
        }
    }

    /// Register a backend factory
    pub fn register_backend<F>(&mut self, info: BackendInfo, factory: F) -> Result<()>
    where
        F: Fn() -> Result<Box<dyn G2p>> + Send + Sync + 'static,
    {
        let backend_id = info.id.clone();
        
        // Validate backend info
        if backend_id.is_empty() {
            return Err(G2pError::ConfigError("Backend ID cannot be empty".to_string()));
        }
        
        if self.backends.contains_key(&backend_id) {
            return Err(G2pError::ConfigError(format!("Backend '{}' already registered", backend_id)));
        }

        // Register the factory
        self.backends.insert(backend_id.clone(), Arc::new(factory));
        
        // Store backend info first
        self.backend_info.insert(backend_id.clone(), info.clone());
        
        // Update language mappings
        for language in &info.supported_languages {
            let backends = self.language_backends.entry(*language).or_insert_with(Vec::new);
            backends.push(backend_id.clone());
            backends.sort_by(|a, b| {
                let priority_a = self.backend_info.get(a).map(|info| info.priority).unwrap_or(0);
                let priority_b = self.backend_info.get(b).map(|info| info.priority).unwrap_or(0);
                priority_b.cmp(&priority_a) // Higher priority first
            });
        }

        // Initialize load balancing counter
        self.load_balance_counters.insert(
            backend_id.clone(),
            std::sync::atomic::AtomicUsize::new(0),
        );

        debug!("Registered backend: {}", backend_id);
        Ok(())
    }

    /// Unregister a backend
    pub fn unregister_backend(&mut self, backend_id: &str) -> Result<()> {
        if !self.backends.contains_key(backend_id) {
            return Err(G2pError::ConfigError(format!("Backend '{}' not found", backend_id)));
        }

        // Remove from backends
        self.backends.remove(backend_id);
        self.backend_info.remove(backend_id);
        self.load_balance_counters.remove(backend_id);

        // Remove from language mappings
        for backends in self.language_backends.values_mut() {
            backends.retain(|id| id != backend_id);
        }

        // Remove from default backends
        self.default_backends.retain(|id| id != backend_id);

        debug!("Unregistered backend: {}", backend_id);
        Ok(())
    }

    /// Set default backend chain
    pub fn set_default_backends(&mut self, backend_ids: Vec<String>) -> Result<()> {
        // Validate all backend IDs exist
        for backend_id in &backend_ids {
            if !self.backends.contains_key(backend_id) {
                return Err(G2pError::ConfigError(format!("Backend '{}' not found", backend_id)));
            }
        }

        self.default_backends = backend_ids;
        debug!("Set default backends: {:?}", self.default_backends);
        Ok(())
    }

    /// Get backend for a specific language with fallback chain
    pub fn get_backend_for_language(&self, language: LanguageCode) -> Result<Box<dyn G2p>> {
        // Try language-specific backends first
        if let Some(backend_ids) = self.language_backends.get(&language) {
            for backend_id in backend_ids {
                if let Some(info) = self.backend_info.get(backend_id) {
                    if info.enabled {
                        if let Some(factory) = self.backends.get(backend_id) {
                            match factory() {
                                Ok(backend) => {
                                    debug!("Using backend '{}' for language {:?}", backend_id, language);
                                    return Ok(backend);
                                }
                                Err(e) => {
                                    warn!("Failed to create backend '{}': {}", backend_id, e);
                                    continue;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Fall back to default backends
        for backend_id in &self.default_backends {
            if let Some(info) = self.backend_info.get(backend_id) {
                if info.enabled {
                    if let Some(factory) = self.backends.get(backend_id) {
                        match factory() {
                            Ok(backend) => {
                                debug!("Using default backend '{}' for language {:?}", backend_id, language);
                                return Ok(backend);
                            }
                            Err(e) => {
                                warn!("Failed to create default backend '{}': {}", backend_id, e);
                                continue;
                            }
                        }
                    }
                }
            }
        }

        Err(G2pError::ConfigError(format!("No available backend for language {:?}", language)))
    }

    /// Get backend with load balancing
    pub fn get_backend_with_load_balancing(&self, language: LanguageCode) -> Result<Box<dyn G2p>> {
        let backend_ids = if let Some(backends) = self.language_backends.get(&language) {
            backends.clone()
        } else {
            self.default_backends.clone()
        };

        if backend_ids.is_empty() {
            return Err(G2pError::ConfigError(format!("No backends available for language {:?}", language)));
        }

        // Round-robin load balancing
        let enabled_backends: Vec<String> = backend_ids
            .into_iter()
            .filter(|id| {
                self.backend_info.get(id).map(|info| info.enabled).unwrap_or(false)
            })
            .collect();

        if enabled_backends.is_empty() {
            return Err(G2pError::ConfigError(format!("No enabled backends for language {:?}", language)));
        }

        // Get next backend in round-robin fashion
        let counter = self.load_balance_counters.get(&enabled_backends[0]);
        let index = if let Some(counter) = counter {
            counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % enabled_backends.len()
        } else {
            0
        };

        let backend_id = &enabled_backends[index];
        if let Some(factory) = self.backends.get(backend_id) {
            match factory() {
                Ok(backend) => {
                    debug!("Load balanced backend '{}' for language {:?}", backend_id, language);
                    Ok(backend)
                }
                Err(e) => {
                    warn!("Failed to create load balanced backend '{}': {}", backend_id, e);
                    // Fall back to first available backend
                    self.get_backend_for_language(language)
                }
            }
        } else {
            Err(G2pError::ConfigError(format!("Backend factory '{}' not found", backend_id)))
        }
    }

    /// Get all registered backends
    pub fn list_backends(&self) -> Vec<&BackendInfo> {
        self.backend_info.values().collect()
    }

    /// Get backend info by ID
    pub fn get_backend_info(&self, backend_id: &str) -> Option<&BackendInfo> {
        self.backend_info.get(backend_id)
    }

    /// Enable/disable a backend
    pub fn set_backend_enabled(&mut self, backend_id: &str, enabled: bool) -> Result<()> {
        if let Some(info) = self.backend_info.get_mut(backend_id) {
            info.enabled = enabled;
            debug!("Backend '{}' enabled: {}", backend_id, enabled);
            Ok(())
        } else {
            Err(G2pError::ConfigError(format!("Backend '{}' not found", backend_id)))
        }
    }

    /// Get backends for a specific language
    pub fn get_backends_for_language(&self, language: LanguageCode) -> Vec<&BackendInfo> {
        if let Some(backend_ids) = self.language_backends.get(&language) {
            backend_ids
                .iter()
                .filter_map(|id| self.backend_info.get(id))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Clear all registered backends
    pub fn clear(&mut self) {
        self.backends.clear();
        self.backend_info.clear();
        self.language_backends.clear();
        self.default_backends.clear();
        self.load_balance_counters.clear();
        debug!("Cleared all registered backends");
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Registry-based G2P converter that manages multiple backends
pub struct RegistryG2p {
    registry: Arc<std::sync::RwLock<BackendRegistry>>,
    /// Whether to use load balancing
    use_load_balancing: bool,
}

impl RegistryG2p {
    /// Create a new registry-based G2P converter
    pub fn new() -> Self {
        Self {
            registry: Arc::new(std::sync::RwLock::new(BackendRegistry::new())),
            use_load_balancing: false,
        }
    }

    /// Create with load balancing enabled
    pub fn with_load_balancing() -> Self {
        Self {
            registry: Arc::new(std::sync::RwLock::new(BackendRegistry::new())),
            use_load_balancing: true,
        }
    }

    /// Get the backend registry
    pub fn registry(&self) -> Arc<std::sync::RwLock<BackendRegistry>> {
        self.registry.clone()
    }

    /// Register a backend
    pub fn register_backend<F>(&self, info: BackendInfo, factory: F) -> Result<()>
    where
        F: Fn() -> Result<Box<dyn G2p>> + Send + Sync + 'static,
    {
        let mut registry = self.registry.write().unwrap();
        registry.register_backend(info, factory)
    }

    /// Set default backends
    pub fn set_default_backends(&self, backend_ids: Vec<String>) -> Result<()> {
        let mut registry = self.registry.write().unwrap();
        registry.set_default_backends(backend_ids)
    }
}

impl Default for RegistryG2p {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl G2p for RegistryG2p {
    async fn to_phonemes(&self, text: &str, lang: Option<LanguageCode>) -> Result<Vec<crate::Phoneme>> {
        let language = lang.unwrap_or(LanguageCode::EnUs);
        
        let backend = {
            let registry = self.registry.read().unwrap();
            if self.use_load_balancing {
                registry.get_backend_with_load_balancing(language)?
            } else {
                registry.get_backend_for_language(language)?
            }
        };

        backend.to_phonemes(text, Some(language)).await
    }

    fn supported_languages(&self) -> Vec<LanguageCode> {
        let registry = self.registry.read().unwrap();
        let mut languages: Vec<LanguageCode> = registry.language_backends.keys().copied().collect();
        languages.sort();
        languages.dedup();
        languages
    }

    fn metadata(&self) -> crate::G2pMetadata {
        let registry = self.registry.read().unwrap();
        crate::G2pMetadata {
            name: "Registry G2P".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: "Registry-based G2P converter with multiple backends".to_string(),
            supported_languages: self.supported_languages(),
            accuracy_scores: std::collections::HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DummyG2p;

    #[test]
    fn test_backend_registry_creation() {
        let registry = BackendRegistry::new();
        assert!(registry.backends.is_empty());
        assert!(registry.backend_info.is_empty());
    }

    #[test]
    fn test_register_backend() {
        let mut registry = BackendRegistry::new();
        
        let info = BackendInfo {
            id: "dummy".to_string(),
            name: "Dummy Backend".to_string(),
            description: "Test backend".to_string(),
            supported_languages: vec![LanguageCode::EnUs],
            priority: 10,
            enabled: true,
        };

        let result = registry.register_backend(info, || Ok(Box::new(DummyG2p::new())));
        assert!(result.is_ok());
        
        assert!(registry.backends.contains_key("dummy"));
        assert!(registry.backend_info.contains_key("dummy"));
    }

    #[test]
    fn test_duplicate_backend_registration() {
        let mut registry = BackendRegistry::new();
        
        let info = BackendInfo {
            id: "dummy".to_string(),
            name: "Dummy Backend".to_string(),
            description: "Test backend".to_string(),
            supported_languages: vec![LanguageCode::EnUs],
            priority: 10,
            enabled: true,
        };

        // First registration should succeed
        let result1 = registry.register_backend(info.clone(), || Ok(Box::new(DummyG2p::new())));
        assert!(result1.is_ok());
        
        // Second registration should fail
        let result2 = registry.register_backend(info, || Ok(Box::new(DummyG2p::new())));
        assert!(result2.is_err());
    }

    #[test]
    fn test_backend_unregistration() {
        let mut registry = BackendRegistry::new();
        
        let info = BackendInfo {
            id: "dummy".to_string(),
            name: "Dummy Backend".to_string(),
            description: "Test backend".to_string(),
            supported_languages: vec![LanguageCode::EnUs],
            priority: 10,
            enabled: true,
        };

        registry.register_backend(info, || Ok(Box::new(DummyG2p::new()))).unwrap();
        
        let result = registry.unregister_backend("dummy");
        assert!(result.is_ok());
        
        assert!(!registry.backends.contains_key("dummy"));
        assert!(!registry.backend_info.contains_key("dummy"));
    }

    #[test]
    fn test_get_backend_for_language() {
        let mut registry = BackendRegistry::new();
        
        let info = BackendInfo {
            id: "dummy".to_string(),
            name: "Dummy Backend".to_string(),
            description: "Test backend".to_string(),
            supported_languages: vec![LanguageCode::EnUs],
            priority: 10,
            enabled: true,
        };

        registry.register_backend(info, || Ok(Box::new(DummyG2p::new()))).unwrap();
        
        let backend = registry.get_backend_for_language(LanguageCode::EnUs);
        assert!(backend.is_ok());
    }

    #[test]
    fn test_backend_priority_ordering() {
        let mut registry = BackendRegistry::new();
        
        let info1 = BackendInfo {
            id: "low_priority".to_string(),
            name: "Low Priority Backend".to_string(),
            description: "Low priority test backend".to_string(),
            supported_languages: vec![LanguageCode::EnUs],
            priority: 5,
            enabled: true,
        };

        let info2 = BackendInfo {
            id: "high_priority".to_string(),
            name: "High Priority Backend".to_string(),
            description: "High priority test backend".to_string(),
            supported_languages: vec![LanguageCode::EnUs],
            priority: 15,
            enabled: true,
        };

        registry.register_backend(info1, || Ok(Box::new(DummyG2p::new()))).unwrap();
        registry.register_backend(info2, || Ok(Box::new(DummyG2p::new()))).unwrap();
        
        // Higher priority backend should be first
        let backends = registry.get_backends_for_language(LanguageCode::EnUs);
        assert_eq!(backends.len(), 2);
        assert_eq!(backends[0].priority, 15);
        assert_eq!(backends[1].priority, 5);
    }

    #[tokio::test]
    async fn test_registry_g2p() {
        let registry_g2p = RegistryG2p::new();
        
        let info = BackendInfo {
            id: "dummy".to_string(),
            name: "Dummy Backend".to_string(),
            description: "Test backend".to_string(),
            supported_languages: vec![LanguageCode::EnUs],
            priority: 10,
            enabled: true,
        };

        registry_g2p.register_backend(info, || Ok(Box::new(DummyG2p::new()))).unwrap();
        registry_g2p.set_default_backends(vec!["dummy".to_string()]).unwrap();
        
        let phonemes = registry_g2p.to_phonemes("test", Some(LanguageCode::EnUs)).await.unwrap();
        assert_eq!(phonemes.len(), 4);
    }
}