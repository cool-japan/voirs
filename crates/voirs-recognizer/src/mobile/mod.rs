//! Mobile platform optimizations for VoiRS recognizer
//!
//! This module provides optimizations specifically designed for mobile devices
//! including iOS and Android platforms. These optimizations focus on:
//!
//! - **Battery Efficiency**: Power-aware processing modes
//! - **Memory Management**: Aggressive memory optimization for limited RAM
//! - **Background Processing**: Efficient handling of app lifecycle events
//! - **Network Awareness**: Adaptive behavior based on connectivity
//! - **Thermal Management**: CPU throttling to prevent overheating
//!
//! # Platform Support
//!
//! - iOS (ARM64)
//! - Android (ARM, ARM64, x86_64)
//! - Optimized for mobile-specific constraints
//!
//! # Example
//!
//! ```rust,no_run
//! use voirs_recognizer::mobile::{MobileOptimizer, MobileConfig, PowerMode};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create mobile-optimized configuration
//!     let config = MobileConfig {
//!         power_mode: PowerMode::Balanced,
//!         max_memory_mb: 256,
//!         enable_background_processing: true,
//!         thermal_throttle: true,
//!         ..Default::default()
//!     };
//!
//!     // Initialize mobile optimizer
//!     let optimizer = MobileOptimizer::new(config).await?;
//!
//!     // Use optimizer to manage ASR processing
//!     optimizer.optimize_for_battery().await?;
//!
//!     Ok(())
//! }
//! ```

use crate::RecognitionError;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

pub mod battery;
pub mod lifecycle;
pub mod memory;
pub mod network;
pub mod thermal;

// Re-export commonly used types
pub use battery::{BatteryManager, ProcessingIntensity};
pub use lifecycle::{AppState, LifecycleEvent, LifecycleManager};
pub use memory::MobileMemoryManager;
pub use network::NetworkManager;
pub use thermal::ThermalManager;

/// Mobile platform optimizer for efficient recognition on mobile devices
pub struct MobileOptimizer {
    /// Configuration settings
    config: MobileConfig,
    /// Battery manager for power-aware processing
    battery_manager: Arc<RwLock<battery::BatteryManager>>,
    /// Memory manager for mobile constraints
    memory_manager: Arc<RwLock<memory::MobileMemoryManager>>,
    /// Thermal manager for CPU throttling
    thermal_manager: Arc<RwLock<thermal::ThermalManager>>,
    /// Network manager for connectivity awareness
    network_manager: Arc<RwLock<network::NetworkManager>>,
    /// Lifecycle manager for app state handling
    lifecycle_manager: Arc<RwLock<lifecycle::LifecycleManager>>,
}

/// Configuration for mobile platform optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileConfig {
    /// Power mode for battery optimization
    pub power_mode: PowerMode,
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    /// Enable background processing when app is inactive
    pub enable_background_processing: bool,
    /// Enable thermal throttling to prevent overheating
    pub thermal_throttle: bool,
    /// Network awareness for model downloads
    pub network_aware: bool,
    /// Minimum battery percentage to run heavy tasks
    pub min_battery_percent: f32,
    /// Maximum CPU temperature before throttling (Celsius)
    pub max_cpu_temp: f32,
    /// Prefer on-device processing over cloud
    pub prefer_ondevice: bool,
}

/// Power mode for battery optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerMode {
    /// Maximum performance, highest battery drain
    Performance,
    /// Balanced mode for typical usage
    Balanced,
    /// Low power mode for battery saving
    LowPower,
    /// Ultra low power for critical battery levels
    UltraLowPower,
}

/// Platform-specific capabilities
#[derive(Debug, Clone)]
pub struct PlatformCapabilities {
    /// CPU core count
    pub cpu_cores: usize,
    /// Total RAM in MB
    pub total_ram_mb: usize,
    /// Available RAM in MB
    pub available_ram_mb: usize,
    /// GPU availability
    pub has_gpu: bool,
    /// Neural engine availability (iOS)
    pub has_neural_engine: bool,
    /// NNAPI availability (Android)
    pub has_nnapi: bool,
    /// Current battery level (0.0 to 1.0)
    pub battery_level: f32,
    /// Is device charging
    pub is_charging: bool,
    /// Network connectivity type
    pub network_type: NetworkType,
}

/// Network connectivity type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkType {
    /// No network connectivity
    None,
    /// WiFi connection
    Wifi,
    /// Cellular 3G
    Cellular3G,
    /// Cellular 4G/LTE
    Cellular4G,
    /// Cellular 5G
    Cellular5G,
}

impl Default for MobileConfig {
    fn default() -> Self {
        Self {
            power_mode: PowerMode::Balanced,
            max_memory_mb: 512,
            enable_background_processing: false,
            thermal_throttle: true,
            network_aware: true,
            min_battery_percent: 20.0,
            max_cpu_temp: 80.0,
            prefer_ondevice: true,
        }
    }
}

impl MobileOptimizer {
    /// Create a new mobile optimizer with the given configuration
    ///
    /// # Errors
    ///
    /// Returns an error if platform capabilities cannot be detected
    pub async fn new(config: MobileConfig) -> Result<Self, RecognitionError> {
        let battery_manager = Arc::new(RwLock::new(battery::BatteryManager::new(
            config.min_battery_percent,
        )?));
        let memory_manager = Arc::new(RwLock::new(memory::MobileMemoryManager::new(
            config.max_memory_mb,
        )?));
        let thermal_manager = Arc::new(RwLock::new(thermal::ThermalManager::new(
            config.max_cpu_temp,
        )?));
        let network_manager = Arc::new(RwLock::new(network::NetworkManager::new()?));
        let lifecycle_manager = Arc::new(RwLock::new(lifecycle::LifecycleManager::new()));

        Ok(Self {
            config,
            battery_manager,
            memory_manager,
            thermal_manager,
            network_manager,
            lifecycle_manager,
        })
    }

    /// Optimize processing for battery efficiency
    ///
    /// # Errors
    ///
    /// Returns an error if optimization fails
    pub async fn optimize_for_battery(&self) -> Result<(), RecognitionError> {
        let mut battery_mgr = self.battery_manager.write().await;
        battery_mgr.set_power_mode(self.config.power_mode);
        battery_mgr.update_battery_status().await?;

        if !battery_mgr.can_run_heavy_tasks() {
            return Err(RecognitionError::ResourceError {
                message: "Battery level too low for heavy processing".to_string(),
                source: None,
            });
        }

        Ok(())
    }

    /// Check if processing should be throttled due to thermal constraints
    ///
    /// # Errors
    ///
    /// Returns an error if thermal status cannot be checked
    pub async fn should_throttle(&self) -> Result<bool, RecognitionError> {
        if !self.config.thermal_throttle {
            return Ok(false);
        }

        let thermal_mgr = self.thermal_manager.read().await;
        thermal_mgr.should_throttle()
    }

    /// Get current platform capabilities
    ///
    /// # Errors
    ///
    /// Returns an error if platform capabilities cannot be detected
    pub async fn get_platform_capabilities(
        &self,
    ) -> Result<PlatformCapabilities, RecognitionError> {
        let memory_mgr = self.memory_manager.read().await;
        let battery_mgr = self.battery_manager.read().await;
        let network_mgr = self.network_manager.read().await;

        Ok(PlatformCapabilities {
            cpu_cores: num_cpus::get(),
            total_ram_mb: memory_mgr.total_memory_mb(),
            available_ram_mb: memory_mgr.available_memory_mb()?,
            has_gpu: self.detect_gpu(),
            has_neural_engine: self.detect_neural_engine(),
            has_nnapi: self.detect_nnapi(),
            battery_level: battery_mgr.battery_level(),
            is_charging: battery_mgr.is_charging(),
            network_type: network_mgr.network_type(),
        })
    }

    /// Handle app lifecycle transition
    ///
    /// # Errors
    ///
    /// Returns an error if lifecycle transition fails
    pub async fn handle_lifecycle_event(
        &self,
        event: lifecycle::LifecycleEvent,
    ) -> Result<(), RecognitionError> {
        let mut lifecycle_mgr = self.lifecycle_manager.write().await;
        lifecycle_mgr.handle_event(event).await
    }

    /// Optimize memory usage for current conditions
    ///
    /// # Errors
    ///
    /// Returns an error if memory optimization fails
    pub async fn optimize_memory(&self) -> Result<(), RecognitionError> {
        let mut memory_mgr = self.memory_manager.write().await;
        memory_mgr.optimize().await
    }

    /// Check if network conditions allow model download
    ///
    /// # Errors
    ///
    /// Returns an error if network status cannot be checked
    pub async fn can_download_models(&self) -> Result<bool, RecognitionError> {
        if !self.config.network_aware {
            return Ok(true);
        }

        let network_mgr = self.network_manager.read().await;
        Ok(network_mgr.is_suitable_for_downloads())
    }

    /// Detect GPU availability (platform-specific)
    fn detect_gpu(&self) -> bool {
        // Platform-specific GPU detection
        #[cfg(target_os = "ios")]
        {
            true // iOS devices have Metal GPUs
        }
        #[cfg(target_os = "android")]
        {
            // Android GPU detection would require JNI calls
            false
        }
        #[cfg(not(any(target_os = "ios", target_os = "android")))]
        {
            false
        }
    }

    /// Detect Neural Engine availability (iOS-specific)
    fn detect_neural_engine(&self) -> bool {
        #[cfg(target_os = "ios")]
        {
            // Neural Engine available on A11 Bionic and later
            // Would require platform-specific detection
            true
        }
        #[cfg(not(target_os = "ios"))]
        {
            false
        }
    }

    /// Detect NNAPI availability (Android-specific)
    fn detect_nnapi(&self) -> bool {
        #[cfg(target_os = "android")]
        {
            // NNAPI available on Android 8.1+
            // Would require JNI calls to check
            true
        }
        #[cfg(not(target_os = "android"))]
        {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mobile_optimizer_creation() {
        let config = MobileConfig::default();
        let optimizer = MobileOptimizer::new(config).await;
        assert!(optimizer.is_ok());
    }

    #[tokio::test]
    async fn test_power_modes() {
        let modes = vec![
            PowerMode::Performance,
            PowerMode::Balanced,
            PowerMode::LowPower,
            PowerMode::UltraLowPower,
        ];

        for mode in modes {
            let config = MobileConfig {
                power_mode: mode,
                ..Default::default()
            };
            let optimizer = MobileOptimizer::new(config).await.unwrap();
            assert_eq!(optimizer.config.power_mode, mode);
        }
    }

    #[tokio::test]
    async fn test_platform_capabilities() {
        let config = MobileConfig::default();
        let optimizer = MobileOptimizer::new(config).await.unwrap();
        let capabilities = optimizer.get_platform_capabilities().await;
        assert!(capabilities.is_ok());

        let caps = capabilities.unwrap();
        assert!(caps.cpu_cores > 0);
        assert!(caps.total_ram_mb > 0);
    }
}
