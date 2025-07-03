//! Voice switching and management functionality.

use crate::{
    error::Result,
    traits::VoiceManager,
    types::{LanguageCode, VoiceConfig},
    VoirsError,
};
use super::discovery::{VoiceRegistry, VoiceSearchCriteria};
use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Default voice manager implementation
pub struct DefaultVoiceManager {
    registry: VoiceRegistry,
    models_dir: PathBuf,
    current_voice: Option<String>,
    download_enabled: bool,
}

impl DefaultVoiceManager {
    /// Create new voice manager
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        Self {
            registry: VoiceRegistry::new(),
            models_dir: models_dir.into(),
            current_voice: None,
            download_enabled: true,
        }
    }
    
    /// Create voice manager with custom registry
    pub fn with_registry(models_dir: impl Into<PathBuf>, registry: VoiceRegistry) -> Self {
        Self {
            registry,
            models_dir: models_dir.into(),
            current_voice: None,
            download_enabled: true,
        }
    }
    
    /// Add custom voice
    pub fn add_voice(&mut self, voice: VoiceConfig) {
        self.registry.register_voice(voice);
    }
    
    /// Search voices
    pub fn search(&self, criteria: &VoiceSearchCriteria) -> Vec<&VoiceConfig> {
        self.registry.find_voices(criteria)
    }
    
    /// Get current voice ID
    pub fn current_voice(&self) -> Option<&str> {
        self.current_voice.as_deref()
    }
    
    /// Set current voice
    pub fn set_current_voice(&mut self, voice_id: Option<String>) -> Result<()> {
        if let Some(ref id) = voice_id {
            // Validate that the voice exists
            if !self.registry.get_voice(id).is_some() {
                return Err(VoirsError::voice_not_found(
                    id.clone(),
                    self.registry.list_voices().iter().map(|v| v.id.clone()).collect()
                ));
            }
        }
        
        self.current_voice = voice_id;
        Ok(())
    }
    
    /// Switch to voice by ID
    pub async fn switch_to_voice(&mut self, voice_id: &str) -> Result<()> {
        // Check if voice exists
        if !self.registry.get_voice(voice_id).is_some() {
            return Err(VoirsError::voice_not_found(
                voice_id.to_string(),
                self.registry.list_voices().iter().map(|v| v.id.clone()).collect()
            ));
        }
        
        // Check if voice is available
        if !self.is_voice_available(voice_id) {
            if self.download_enabled {
                tracing::info!("Voice {} not available locally, attempting download", voice_id);
                self.download_voice(voice_id).await?;
            } else {
                return Err(VoirsError::voice_not_found(
                    voice_id.to_string(),
                    self.get_available_voices()
                ));
            }
        }
        
        self.current_voice = Some(voice_id.to_string());
        tracing::info!("Switched to voice: {}", voice_id);
        Ok(())
    }
    
    /// Switch to default voice for language
    pub async fn switch_to_language(&mut self, language: LanguageCode) -> Result<()> {
        if let Some(default_voice_id) = self.default_voice_for_language(language) {
            self.switch_to_voice(&default_voice_id).await
        } else {
            Err(VoirsError::voice_not_found(
                format!("default voice for {:?}", language),
                self.registry.list_voices().iter().map(|v| v.id.clone()).collect()
            ))
        }
    }
    
    /// Get available voice IDs (voices that have model files present)
    pub fn get_available_voices(&self) -> Vec<String> {
        self.registry
            .list_voices()
            .iter()
            .filter(|voice| self.is_voice_available(&voice.id))
            .map(|voice| voice.id.clone())
            .collect()
    }
    
    /// Get unavailable voice IDs (voices missing model files)
    pub fn get_unavailable_voices(&self) -> Vec<String> {
        self.registry
            .list_voices()
            .iter()
            .filter(|voice| !self.is_voice_available(&voice.id))
            .map(|voice| voice.id.clone())
            .collect()
    }
    
    /// Check if any voice is available for a language
    pub fn has_available_voice_for_language(&self, language: LanguageCode) -> bool {
        self.registry
            .voices_for_language(language)
            .iter()
            .any(|voice| self.is_voice_available(&voice.id))
    }
    
    /// Get first available voice for language
    pub fn get_available_voice_for_language(&self, language: LanguageCode) -> Option<String> {
        self.registry
            .voices_for_language(language)
            .iter()
            .find(|voice| self.is_voice_available(&voice.id))
            .map(|voice| voice.id.clone())
    }
    
    /// Enable or disable automatic downloading
    pub fn set_download_enabled(&mut self, enabled: bool) {
        self.download_enabled = enabled;
    }
    
    /// Check if automatic downloading is enabled
    pub fn is_download_enabled(&self) -> bool {
        self.download_enabled
    }
    
    /// Validate voice models exist and are accessible
    pub fn validate_voice_models(&self, voice_id: &str) -> Result<VoiceValidationResult> {
        let voice = self.registry.get_voice(voice_id)
            .ok_or_else(|| VoirsError::voice_not_found(
                voice_id.to_string(),
                self.registry.list_voices().iter().map(|v| v.id.clone()).collect()
            ))?;
        
        let mut result = VoiceValidationResult {
            voice_id: voice_id.to_string(),
            valid: true,
            missing_files: Vec::new(),
            invalid_files: Vec::new(),
            warnings: Vec::new(),
        };
        
        // Check acoustic model
        let acoustic_path = self.models_dir.join(&voice.model_config.acoustic_model);
        if !acoustic_path.exists() {
            result.missing_files.push(voice.model_config.acoustic_model.clone());
            result.valid = false;
        }
        
        // Check vocoder model
        let vocoder_path = self.models_dir.join(&voice.model_config.vocoder_model);
        if !vocoder_path.exists() {
            result.missing_files.push(voice.model_config.vocoder_model.clone());
            result.valid = false;
        }
        
        // Check G2P model if specified
        if let Some(ref g2p_model) = voice.model_config.g2p_model {
            let g2p_path = self.models_dir.join(g2p_model);
            if !g2p_path.exists() {
                result.missing_files.push(g2p_model.clone());
                result.valid = false;
            }
        }
        
        // Add warnings for potential issues
        if voice.model_config.device_requirements.min_memory_mb > 2048 {
            result.warnings.push("Voice requires more than 2GB of memory".to_string());
        }
        
        if !voice.model_config.device_requirements.compute_capabilities.contains(&"cpu".to_string()) {
            result.warnings.push("Voice may not support CPU-only inference".to_string());
        }
        
        Ok(result)
    }
    
    /// Download all models for a voice
    async fn download_voice_models(&self, voice: &VoiceConfig) -> Result<()> {
        // TODO: Implement actual model downloading
        let models = [
            &voice.model_config.acoustic_model,
            &voice.model_config.vocoder_model,
        ];
        
        let mut all_models = models.to_vec();
        if let Some(ref g2p_model) = voice.model_config.g2p_model {
            all_models.push(g2p_model);
        }
        
        for model_path in all_models {
            let full_path = self.models_dir.join(model_path);
            
            // Create parent directories if needed
            if let Some(parent) = full_path.parent() {
                tokio::fs::create_dir_all(parent).await.map_err(|e| {
                    VoirsError::IoError {
                        path: parent.to_path_buf(),
                        operation: crate::error::types::IoOperation::Create,
                        source: e,
                    }
                })?;
            }
            
            // Simulate download
            tracing::info!("Downloading model: {}", model_path);
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            
            // Create empty file to simulate download
            tokio::fs::write(&full_path, b"dummy model file").await.map_err(|e| {
                VoirsError::IoError {
                    path: full_path,
                    operation: crate::error::types::IoOperation::Write,
                    source: e,
                }
            })?;
        }
        
        Ok(())
    }
    
    /// Get voice switching history
    pub fn get_switching_history(&self) -> Vec<VoiceSwitch> {
        // TODO: Implement voice switching history tracking
        // For now, return empty vector
        Vec::new()
    }
    
    /// Clear voice switching history
    pub fn clear_switching_history(&mut self) {
        // TODO: Implement history clearing
    }
}

#[async_trait]
impl VoiceManager for DefaultVoiceManager {
    async fn list_voices(&self) -> Result<Vec<VoiceConfig>> {
        Ok(self.registry.list_voices().into_iter().cloned().collect())
    }
    
    async fn get_voice(&self, voice_id: &str) -> Result<Option<VoiceConfig>> {
        Ok(self.registry.get_voice(voice_id).cloned())
    }
    
    async fn download_voice(&self, voice_id: &str) -> Result<()> {
        let voice = self.registry.get_voice(voice_id)
            .ok_or_else(|| VoirsError::voice_not_found(
                voice_id.to_string(),
                self.registry.list_voices().iter().map(|v| v.id.clone()).collect()
            ))?;
        
        self.download_voice_models(voice).await?;
        
        tracing::info!("Voice '{}' downloaded successfully", voice_id);
        Ok(())
    }
    
    fn is_voice_available(&self, voice_id: &str) -> bool {
        if let Some(voice) = self.registry.get_voice(voice_id) {
            // Check if model files exist
            let models_exist = [
                &voice.model_config.acoustic_model,
                &voice.model_config.vocoder_model,
            ]
            .iter()
            .chain(voice.model_config.g2p_model.as_ref().iter())
            .all(|model_path| {
                let full_path = self.models_dir.join(model_path);
                full_path.exists()
            });
            
            models_exist
        } else {
            false
        }
    }
    
    fn default_voice_for_language(&self, lang: LanguageCode) -> Option<String> {
        self.registry
            .default_voice_for_language(lang)
            .map(|voice| voice.id.clone())
    }
}

/// Thread-safe voice manager wrapper
pub struct ConcurrentVoiceManager {
    inner: Arc<RwLock<DefaultVoiceManager>>,
}

impl ConcurrentVoiceManager {
    /// Create new concurrent voice manager
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(DefaultVoiceManager::new(models_dir))),
        }
    }
    
    /// Get read access to voice manager
    pub async fn read(&self) -> tokio::sync::RwLockReadGuard<'_, DefaultVoiceManager> {
        self.inner.read().await
    }
    
    /// Get write access to voice manager
    pub async fn write(&self) -> tokio::sync::RwLockWriteGuard<'_, DefaultVoiceManager> {
        self.inner.write().await
    }
    
    /// Switch voice (convenience method)
    pub async fn switch_to_voice(&self, voice_id: &str) -> Result<()> {
        self.inner.write().await.switch_to_voice(voice_id).await
    }
    
    /// Get current voice (convenience method)
    pub async fn current_voice(&self) -> Option<String> {
        self.inner.read().await.current_voice().map(|s| s.to_string())
    }
}

/// Voice validation result
#[derive(Debug, Clone)]
pub struct VoiceValidationResult {
    /// Voice ID that was validated
    pub voice_id: String,
    /// Whether the voice is valid and usable
    pub valid: bool,
    /// List of missing model files
    pub missing_files: Vec<String>,
    /// List of invalid or corrupted files
    pub invalid_files: Vec<String>,
    /// List of warnings
    pub warnings: Vec<String>,
}

/// Voice switch record for history tracking
#[derive(Debug, Clone)]
pub struct VoiceSwitch {
    /// Timestamp of the switch
    pub timestamp: std::time::SystemTime,
    /// Previous voice ID (if any)
    pub from_voice: Option<String>,
    /// New voice ID
    pub to_voice: String,
    /// Whether the switch was successful
    pub successful: bool,
    /// Error message if switch failed
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gender;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_voice_manager_creation() {
        let temp_dir = tempdir().unwrap();
        let manager = DefaultVoiceManager::new(temp_dir.path());
        
        // Should have voices from registry
        let voices = manager.list_voices().await.unwrap();
        assert!(!voices.is_empty());
    }

    #[tokio::test]
    async fn test_voice_availability() {
        let temp_dir = tempdir().unwrap();
        let manager = DefaultVoiceManager::new(temp_dir.path());
        
        // Voice should not be available (no model files)
        assert!(!manager.is_voice_available("en-US-female-calm"));
        
        // Get available voices should be empty
        let available = manager.get_available_voices();
        assert!(available.is_empty());
        
        // Get unavailable voices should not be empty
        let unavailable = manager.get_unavailable_voices();
        assert!(!unavailable.is_empty());
    }

    #[tokio::test]
    async fn test_voice_switching() {
        let temp_dir = tempdir().unwrap();
        let mut manager = DefaultVoiceManager::new(temp_dir.path());
        
        // Initially no current voice
        assert!(manager.current_voice().is_none());
        
        // Download and switch to voice
        manager.set_download_enabled(true);
        let result = manager.switch_to_voice("en-US-female-calm").await;
        assert!(result.is_ok());
        
        // Should now have current voice
        assert_eq!(manager.current_voice(), Some("en-US-female-calm"));
        
        // Voice should now be available
        assert!(manager.is_voice_available("en-US-female-calm"));
    }

    #[tokio::test]
    async fn test_language_switching() {
        let temp_dir = tempdir().unwrap();
        let mut manager = DefaultVoiceManager::new(temp_dir.path());
        manager.set_download_enabled(true);
        
        // Switch to English
        let result = manager.switch_to_language(LanguageCode::EnUs).await;
        assert!(result.is_ok());
        
        let current = manager.current_voice().unwrap();
        let voice = manager.get_voice(&current).await.unwrap().unwrap();
        assert_eq!(voice.language, LanguageCode::EnUs);
    }

    #[tokio::test]
    async fn test_voice_validation() {
        let temp_dir = tempdir().unwrap();
        let manager = DefaultVoiceManager::new(temp_dir.path());
        
        // Validate non-existent voice
        let result = manager.validate_voice_models("non-existent");
        assert!(result.is_err());
        
        // Validate existing voice (should show missing files)
        let result = manager.validate_voice_models("en-US-female-calm").unwrap();
        assert!(!result.valid);
        assert!(!result.missing_files.is_empty());
    }

    #[tokio::test]
    async fn test_concurrent_voice_manager() {
        let temp_dir = tempdir().unwrap();
        let manager = ConcurrentVoiceManager::new(temp_dir.path());
        
        // Test concurrent access
        let current = manager.current_voice().await;
        assert!(current.is_none());
        
        // Test write access
        {
            let mut write_guard = manager.write().await;
            write_guard.set_download_enabled(true);
        }
        
        // Test read access
        {
            let read_guard = manager.read().await;
            assert!(read_guard.is_download_enabled());
        }
    }

    #[tokio::test]
    async fn test_voice_search() {
        let temp_dir = tempdir().unwrap();
        let manager = DefaultVoiceManager::new(temp_dir.path());
        
        // Search for female voices
        let criteria = VoiceSearchCriteria::new().gender(Gender::Female);
        let results = manager.search(&criteria);
        assert!(!results.is_empty());
        
        for voice in results {
            assert_eq!(voice.characteristics.gender, Some(Gender::Female));
        }
    }

    #[test]
    fn test_voice_validation_result() {
        let result = VoiceValidationResult {
            voice_id: "test-voice".to_string(),
            valid: false,
            missing_files: vec!["model1.bin".to_string(), "model2.bin".to_string()],
            invalid_files: vec![],
            warnings: vec!["High memory usage".to_string()],
        };
        
        assert!(!result.valid);
        assert_eq!(result.missing_files.len(), 2);
        assert_eq!(result.warnings.len(), 1);
    }
}