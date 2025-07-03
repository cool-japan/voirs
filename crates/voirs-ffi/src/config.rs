//! Configuration management for VoiRS FFI operations.
//!
//! This module provides a comprehensive configuration system for managing
//! pipeline settings, user preferences, and runtime parameters.

use std::{
    collections::HashMap,
    fs,
    path::Path,
};
use serde::{Deserialize, Serialize};
use parking_lot::Mutex;
use once_cell::sync::Lazy;
use crate::{VoirsErrorCode, set_last_error};

/// Global configuration registry for all pipelines
static CONFIG_REGISTRY: Lazy<Mutex<HashMap<u32, PipelineConfig>>> = Lazy::new(|| {
    Mutex::new(HashMap::new())
});

/// Complete pipeline configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Synthesis configuration
    pub synthesis: SynthesisSettings,
    /// Threading configuration
    pub threading: ThreadingSettings,
    /// Audio processing configuration
    pub audio: AudioSettings,
    /// Device configuration
    pub device: DeviceSettings,
    /// Performance tuning
    pub performance: PerformanceSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisSettings {
    pub speaking_rate: f32,
    pub pitch_shift: f32,
    pub volume_gain: f32,
    pub enable_enhancement: bool,
    pub output_format: String,
    pub sample_rate: u32,
    pub quality: String,
    pub language: String,
    pub voice_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadingSettings {
    pub thread_count: u32,
    pub max_concurrent: u32,
    pub enable_thread_pool: bool,
    pub thread_priority: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSettings {
    pub buffer_size: u32,
    pub channels: u32,
    pub bit_depth: u32,
    pub enable_noise_reduction: bool,
    pub dynamic_range_compression: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceSettings {
    pub device_type: String, // cpu, cuda, metal, vulkan
    pub device_id: Option<u32>,
    pub use_gpu: bool,
    pub memory_limit_mb: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSettings {
    pub cache_size_mb: u32,
    pub enable_caching: bool,
    pub prefetch_models: bool,
    pub optimize_for_latency: bool,
    pub optimize_for_quality: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            synthesis: SynthesisSettings {
                speaking_rate: 1.0,
                pitch_shift: 0.0,
                volume_gain: 0.0,
                enable_enhancement: true,
                output_format: "wav".to_string(),
                sample_rate: 22050,
                quality: "high".to_string(),
                language: "en-US".to_string(),
                voice_id: None,
            },
            threading: ThreadingSettings {
                thread_count: num_cpus::get() as u32,
                max_concurrent: 4,
                enable_thread_pool: true,
                thread_priority: "normal".to_string(),
            },
            audio: AudioSettings {
                buffer_size: 1024,
                channels: 1,
                bit_depth: 16,
                enable_noise_reduction: false,
                dynamic_range_compression: false,
            },
            device: DeviceSettings {
                device_type: "cpu".to_string(),
                device_id: None,
                use_gpu: false,
                memory_limit_mb: None,
            },
            performance: PerformanceSettings {
                cache_size_mb: 256,
                enable_caching: true,
                prefetch_models: false,
                optimize_for_latency: false,
                optimize_for_quality: true,
            },
        }
    }
}

impl PipelineConfig {
    /// Validate configuration values
    pub fn validate(&self) -> Result<(), String> {
        // Validate synthesis settings
        if self.synthesis.speaking_rate <= 0.0 || self.synthesis.speaking_rate > 5.0 {
            return Err("Speaking rate must be between 0.0 and 5.0".to_string());
        }
        
        if self.synthesis.pitch_shift < -2.0 || self.synthesis.pitch_shift > 2.0 {
            return Err("Pitch shift must be between -2.0 and 2.0".to_string());
        }
        
        if self.synthesis.volume_gain < -20.0 || self.synthesis.volume_gain > 20.0 {
            return Err("Volume gain must be between -20.0 and 20.0 dB".to_string());
        }
        
        if !["wav", "flac", "mp3", "opus", "ogg"].contains(&self.synthesis.output_format.as_str()) {
            return Err("Invalid output format".to_string());
        }
        
        if ![8000, 16000, 22050, 44100, 48000].contains(&self.synthesis.sample_rate) {
            return Err("Invalid sample rate".to_string());
        }
        
        if !["low", "medium", "high", "ultra"].contains(&self.synthesis.quality.as_str()) {
            return Err("Invalid quality setting".to_string());
        }
        
        // Validate threading settings
        if self.threading.thread_count == 0 || self.threading.thread_count > 64 {
            return Err("Thread count must be between 1 and 64".to_string());
        }
        
        if self.threading.max_concurrent == 0 || self.threading.max_concurrent > 32 {
            return Err("Max concurrent must be between 1 and 32".to_string());
        }
        
        // Validate audio settings
        if ![256, 512, 1024, 2048, 4096].contains(&self.audio.buffer_size) {
            return Err("Buffer size must be 256, 512, 1024, 2048, or 4096".to_string());
        }
        
        if self.audio.channels == 0 || self.audio.channels > 8 {
            return Err("Channels must be between 1 and 8".to_string());
        }
        
        if ![8, 16, 24, 32].contains(&self.audio.bit_depth) {
            return Err("Bit depth must be 8, 16, 24, or 32".to_string());
        }
        
        // Validate device settings
        if !["cpu", "cuda", "metal", "vulkan"].contains(&self.device.device_type.as_str()) {
            return Err("Invalid device type".to_string());
        }
        
        // Validate performance settings
        if self.performance.cache_size_mb > 2048 {
            return Err("Cache size cannot exceed 2048 MB".to_string());
        }
        
        Ok(())
    }
    
    /// Get a configuration value by key path (e.g., "synthesis.speaking_rate")
    pub fn get_value(&self, key: &str) -> Option<String> {
        let parts: Vec<&str> = key.split('.').collect();
        if parts.len() != 2 {
            return None;
        }
        
        match (parts[0], parts[1]) {
            ("synthesis", "speaking_rate") => Some(self.synthesis.speaking_rate.to_string()),
            ("synthesis", "pitch_shift") => Some(self.synthesis.pitch_shift.to_string()),
            ("synthesis", "volume_gain") => Some(self.synthesis.volume_gain.to_string()),
            ("synthesis", "enable_enhancement") => Some(self.synthesis.enable_enhancement.to_string()),
            ("synthesis", "output_format") => Some(self.synthesis.output_format.clone()),
            ("synthesis", "sample_rate") => Some(self.synthesis.sample_rate.to_string()),
            ("synthesis", "quality") => Some(self.synthesis.quality.clone()),
            ("synthesis", "language") => Some(self.synthesis.language.clone()),
            ("synthesis", "voice_id") => self.synthesis.voice_id.clone(),
            
            ("threading", "thread_count") => Some(self.threading.thread_count.to_string()),
            ("threading", "max_concurrent") => Some(self.threading.max_concurrent.to_string()),
            ("threading", "enable_thread_pool") => Some(self.threading.enable_thread_pool.to_string()),
            ("threading", "thread_priority") => Some(self.threading.thread_priority.clone()),
            
            ("audio", "buffer_size") => Some(self.audio.buffer_size.to_string()),
            ("audio", "channels") => Some(self.audio.channels.to_string()),
            ("audio", "bit_depth") => Some(self.audio.bit_depth.to_string()),
            ("audio", "enable_noise_reduction") => Some(self.audio.enable_noise_reduction.to_string()),
            ("audio", "dynamic_range_compression") => Some(self.audio.dynamic_range_compression.to_string()),
            
            ("device", "device_type") => Some(self.device.device_type.clone()),
            ("device", "device_id") => self.device.device_id.map(|id| id.to_string()),
            ("device", "use_gpu") => Some(self.device.use_gpu.to_string()),
            ("device", "memory_limit_mb") => self.device.memory_limit_mb.map(|mb| mb.to_string()),
            
            ("performance", "cache_size_mb") => Some(self.performance.cache_size_mb.to_string()),
            ("performance", "enable_caching") => Some(self.performance.enable_caching.to_string()),
            ("performance", "prefetch_models") => Some(self.performance.prefetch_models.to_string()),
            ("performance", "optimize_for_latency") => Some(self.performance.optimize_for_latency.to_string()),
            ("performance", "optimize_for_quality") => Some(self.performance.optimize_for_quality.to_string()),
            
            _ => None,
        }
    }
    
    /// Set a configuration value by key path
    pub fn set_value(&mut self, key: &str, value: &str) -> Result<(), String> {
        let parts: Vec<&str> = key.split('.').collect();
        if parts.len() != 2 {
            return Err("Invalid key format. Use 'section.key'".to_string());
        }
        
        match (parts[0], parts[1]) {
            ("synthesis", "speaking_rate") => {
                self.synthesis.speaking_rate = value.parse()
                    .map_err(|_| "Invalid speaking rate value")?;
            },
            ("synthesis", "pitch_shift") => {
                self.synthesis.pitch_shift = value.parse()
                    .map_err(|_| "Invalid pitch shift value")?;
            },
            ("synthesis", "volume_gain") => {
                self.synthesis.volume_gain = value.parse()
                    .map_err(|_| "Invalid volume gain value")?;
            },
            ("synthesis", "enable_enhancement") => {
                self.synthesis.enable_enhancement = value.parse()
                    .map_err(|_| "Invalid enhancement flag value")?;
            },
            ("synthesis", "output_format") => {
                self.synthesis.output_format = value.to_string();
            },
            ("synthesis", "sample_rate") => {
                self.synthesis.sample_rate = value.parse()
                    .map_err(|_| "Invalid sample rate value")?;
            },
            ("synthesis", "quality") => {
                self.synthesis.quality = value.to_string();
            },
            ("synthesis", "language") => {
                self.synthesis.language = value.to_string();
            },
            ("synthesis", "voice_id") => {
                self.synthesis.voice_id = if value.is_empty() { None } else { Some(value.to_string()) };
            },
            
            ("threading", "thread_count") => {
                self.threading.thread_count = value.parse()
                    .map_err(|_| "Invalid thread count value")?;
            },
            ("threading", "max_concurrent") => {
                self.threading.max_concurrent = value.parse()
                    .map_err(|_| "Invalid max concurrent value")?;
            },
            ("threading", "enable_thread_pool") => {
                self.threading.enable_thread_pool = value.parse()
                    .map_err(|_| "Invalid thread pool flag value")?;
            },
            ("threading", "thread_priority") => {
                self.threading.thread_priority = value.to_string();
            },
            
            ("audio", "buffer_size") => {
                self.audio.buffer_size = value.parse()
                    .map_err(|_| "Invalid buffer size value")?;
            },
            ("audio", "channels") => {
                self.audio.channels = value.parse()
                    .map_err(|_| "Invalid channels value")?;
            },
            ("audio", "bit_depth") => {
                self.audio.bit_depth = value.parse()
                    .map_err(|_| "Invalid bit depth value")?;
            },
            ("audio", "enable_noise_reduction") => {
                self.audio.enable_noise_reduction = value.parse()
                    .map_err(|_| "Invalid noise reduction flag value")?;
            },
            ("audio", "dynamic_range_compression") => {
                self.audio.dynamic_range_compression = value.parse()
                    .map_err(|_| "Invalid dynamic range compression flag value")?;
            },
            
            ("device", "device_type") => {
                self.device.device_type = value.to_string();
            },
            ("device", "device_id") => {
                self.device.device_id = if value.is_empty() { 
                    None 
                } else { 
                    Some(value.parse().map_err(|_| "Invalid device ID value")?) 
                };
            },
            ("device", "use_gpu") => {
                self.device.use_gpu = value.parse()
                    .map_err(|_| "Invalid GPU flag value")?;
            },
            ("device", "memory_limit_mb") => {
                self.device.memory_limit_mb = if value.is_empty() { 
                    None 
                } else { 
                    Some(value.parse().map_err(|_| "Invalid memory limit value")?) 
                };
            },
            
            ("performance", "cache_size_mb") => {
                self.performance.cache_size_mb = value.parse()
                    .map_err(|_| "Invalid cache size value")?;
            },
            ("performance", "enable_caching") => {
                self.performance.enable_caching = value.parse()
                    .map_err(|_| "Invalid caching flag value")?;
            },
            ("performance", "prefetch_models") => {
                self.performance.prefetch_models = value.parse()
                    .map_err(|_| "Invalid prefetch flag value")?;
            },
            ("performance", "optimize_for_latency") => {
                self.performance.optimize_for_latency = value.parse()
                    .map_err(|_| "Invalid latency optimization flag value")?;
            },
            ("performance", "optimize_for_quality") => {
                self.performance.optimize_for_quality = value.parse()
                    .map_err(|_| "Invalid quality optimization flag value")?;
            },
            
            _ => return Err(format!("Unknown configuration key: {}", key)),
        }
        
        // Validate after setting
        self.validate().map_err(|e| format!("Validation failed: {}", e))?;
        
        Ok(())
    }
    
    /// Load configuration from JSON file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file: {}", e))?;
        
        let config: PipelineConfig = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse config file: {}", e))?;
        
        config.validate()?;
        Ok(config)
    }
    
    /// Save configuration to JSON file
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        self.validate()?;
        
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize config: {}", e))?;
        
        fs::write(path, content)
            .map_err(|e| format!("Failed to write config file: {}", e))?;
        
        Ok(())
    }
}

/// Register a configuration for a pipeline
pub fn register_config(pipeline_id: u32, config: PipelineConfig) {
    let mut registry = CONFIG_REGISTRY.lock();
    registry.insert(pipeline_id, config);
}

/// Get configuration for a pipeline
pub fn get_config(pipeline_id: u32) -> Option<PipelineConfig> {
    let registry = CONFIG_REGISTRY.lock();
    registry.get(&pipeline_id).cloned()
}

/// Update configuration for a pipeline
pub fn update_config(pipeline_id: u32, config: PipelineConfig) -> VoirsErrorCode {
    if let Err(e) = config.validate() {
        set_last_error(e);
        return VoirsErrorCode::InvalidParameter;
    }
    
    let mut registry = CONFIG_REGISTRY.lock();
    if registry.contains_key(&pipeline_id) {
        registry.insert(pipeline_id, config);
        VoirsErrorCode::Success
    } else {
        set_last_error("Pipeline not found".to_string());
        VoirsErrorCode::InvalidParameter
    }
}

/// Remove configuration for a pipeline
pub fn remove_config(pipeline_id: u32) -> bool {
    let mut registry = CONFIG_REGISTRY.lock();
    registry.remove(&pipeline_id).is_some()
}

/// Get configuration value for a pipeline
pub fn get_config_value(pipeline_id: u32, key: &str) -> Option<String> {
    let registry = CONFIG_REGISTRY.lock();
    registry.get(&pipeline_id)?.get_value(key)
}

/// Set configuration value for a pipeline
pub fn set_config_value(pipeline_id: u32, key: &str, value: &str) -> VoirsErrorCode {
    let mut registry = CONFIG_REGISTRY.lock();
    
    if let Some(config) = registry.get_mut(&pipeline_id) {
        match config.set_value(key, value) {
            Ok(_) => VoirsErrorCode::Success,
            Err(e) => {
                set_last_error(e);
                VoirsErrorCode::InvalidParameter
            }
        }
    } else {
        set_last_error("Pipeline not found".to_string());
        VoirsErrorCode::InvalidParameter
    }
}

/// Load configuration from file for a pipeline
pub fn load_config_from_file(pipeline_id: u32, file_path: &str) -> VoirsErrorCode {
    match PipelineConfig::from_file(file_path) {
        Ok(config) => update_config(pipeline_id, config),
        Err(e) => {
            set_last_error(e);
            VoirsErrorCode::IoError
        }
    }
}

/// Save configuration to file for a pipeline
pub fn save_config_to_file(pipeline_id: u32, file_path: &str) -> VoirsErrorCode {
    let config = match get_config(pipeline_id) {
        Some(config) => config,
        None => {
            set_last_error("Pipeline not found".to_string());
            return VoirsErrorCode::InvalidParameter;
        }
    };
    
    match config.to_file(file_path) {
        Ok(_) => VoirsErrorCode::Success,
        Err(e) => {
            set_last_error(e);
            VoirsErrorCode::IoError
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_default_config() {
        let config = PipelineConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.synthesis.speaking_rate, 1.0);
        assert_eq!(config.threading.thread_count, num_cpus::get() as u32);
    }

    #[test]
    fn test_config_validation() {
        let mut config = PipelineConfig::default();
        
        // Test invalid speaking rate
        config.synthesis.speaking_rate = 10.0;
        assert!(config.validate().is_err());
        
        // Reset and test invalid thread count
        config = PipelineConfig::default();
        config.threading.thread_count = 100;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_get_set_value() {
        let mut config = PipelineConfig::default();
        
        // Test getting value
        assert_eq!(config.get_value("synthesis.speaking_rate"), Some("1".to_string()));
        
        // Test setting value
        assert!(config.set_value("synthesis.speaking_rate", "1.5").is_ok());
        assert_eq!(config.get_value("synthesis.speaking_rate"), Some("1.5".to_string()));
        
        // Test invalid key
        assert!(config.set_value("invalid.key", "value").is_err());
        
        // Test invalid value
        assert!(config.set_value("synthesis.speaking_rate", "invalid").is_err());
    }

    #[test]
    fn test_file_operations() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_config.json");
        
        let config = PipelineConfig::default();
        
        // Test saving
        assert!(config.to_file(&file_path).is_ok());
        assert!(file_path.exists());
        
        // Test loading
        let loaded_config = PipelineConfig::from_file(&file_path).unwrap();
        assert_eq!(config.synthesis.speaking_rate, loaded_config.synthesis.speaking_rate);
        assert_eq!(config.threading.thread_count, loaded_config.threading.thread_count);
    }

    #[test]
    fn test_config_registry() {
        let config = PipelineConfig::default();
        let pipeline_id = 1;
        
        // Test register
        register_config(pipeline_id, config.clone());
        assert!(get_config(pipeline_id).is_some());
        
        // Test update
        let mut updated_config = config.clone();
        updated_config.synthesis.speaking_rate = 1.5;
        assert_eq!(update_config(pipeline_id, updated_config), VoirsErrorCode::Success);
        
        let retrieved = get_config(pipeline_id).unwrap();
        assert_eq!(retrieved.synthesis.speaking_rate, 1.5);
        
        // Test remove
        assert!(remove_config(pipeline_id));
        assert!(get_config(pipeline_id).is_none());
    }

    #[test]
    fn test_config_value_operations() {
        let config = PipelineConfig::default();
        let pipeline_id = 1;
        register_config(pipeline_id, config);
        
        // Test get value
        assert_eq!(get_config_value(pipeline_id, "synthesis.speaking_rate"), Some("1".to_string()));
        
        // Test set value
        assert_eq!(set_config_value(pipeline_id, "synthesis.speaking_rate", "1.2"), VoirsErrorCode::Success);
        assert_eq!(get_config_value(pipeline_id, "synthesis.speaking_rate"), Some("1.2".to_string()));
        
        // Test invalid pipeline
        assert_eq!(set_config_value(999, "synthesis.speaking_rate", "1.0"), VoirsErrorCode::InvalidParameter);
    }

    #[test]
    fn test_file_config_operations() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_config.json");
        
        let config = PipelineConfig::default();
        let pipeline_id = 1;
        register_config(pipeline_id, config.clone());
        
        // Test save to file
        assert_eq!(save_config_to_file(pipeline_id, file_path.to_str().unwrap()), VoirsErrorCode::Success);
        assert!(file_path.exists());
        
        // Modify config
        assert_eq!(set_config_value(pipeline_id, "synthesis.speaking_rate", "1.8"), VoirsErrorCode::Success);
        
        // Test load from file (should restore original value)
        assert_eq!(load_config_from_file(pipeline_id, file_path.to_str().unwrap()), VoirsErrorCode::Success);
        assert_eq!(get_config_value(pipeline_id, "synthesis.speaking_rate"), Some("1".to_string()));
    }
}