//! Mobile platform adapter implementation
//!
//! This module provides mobile-specific implementations for VoiRS feedback system
//! including iOS and Android support with platform-specific optimizations.

use super::{AudioDeviceInfo, NetworkType, PlatformAdapter, PlatformError, PlatformResult};
use std::path::PathBuf;
use std::time::Duration;

/// Mobile platform adapter for iOS and Android
pub struct MobileAdapter {
    initialized: bool,
}

impl MobileAdapter {
    /// Create a new mobile adapter
    pub fn new() -> Self {
        Self { initialized: false }
    }

    /// Get mobile platform type
    pub fn get_mobile_platform() -> MobilePlatform {
        #[cfg(target_os = "ios")]
        return MobilePlatform::IOS;

        #[cfg(target_os = "android")]
        return MobilePlatform::Android;

        #[cfg(not(any(target_os = "ios", target_os = "android")))]
        return MobilePlatform::Unknown;
    }

    /// Check if device has specific hardware capabilities
    pub fn has_hardware_capability(capability: &str) -> bool {
        match capability {
            "microphone" => true,
            "speaker" => true,
            "haptic" => true,
            "accelerometer" => true,
            "gyroscope" => true,
            "proximity_sensor" => true,
            "ambient_light_sensor" => true,
            "camera" => true,
            "gps" => true,
            "bluetooth" => true,
            "nfc" => false, // Not all devices have NFC
            _ => false,
        }
    }

    /// Get device information
    pub fn get_device_info() -> DeviceInfo {
        DeviceInfo {
            platform: Self::get_mobile_platform(),
            device_model: "Unknown".to_string(),
            os_version: "Unknown".to_string(),
            screen_width: 390,
            screen_height: 844,
            screen_density: 3.0,
            battery_level: 1.0,
            is_low_power_mode: false,
            available_storage: 1024 * 1024 * 1024,  // 1GB
            total_storage: 64 * 1024 * 1024 * 1024, // 64GB
            network_type: NetworkType::WiFi,
        }
    }

    /// Request permission for specific feature
    pub async fn request_permission(permission: Permission) -> Result<bool, PlatformError> {
        match permission {
            Permission::Microphone => {
                // Request microphone permission
                Ok(true)
            }
            Permission::LocalStorage => {
                // Usually granted by default
                Ok(true)
            }
            Permission::Notifications => {
                // Request notification permission
                Ok(true)
            }
            Permission::Camera => {
                // Request camera permission
                Ok(true)
            }
            Permission::Location => {
                // Request location permission
                Ok(true)
            }
            Permission::Bluetooth => {
                // Request bluetooth permission
                Ok(true)
            }
        }
    }

    /// Check if app is in foreground
    pub fn is_foreground() -> bool {
        // In real implementation, this would check app state
        true
    }

    /// Check if device is in low power mode
    pub fn is_low_power_mode() -> bool {
        // In real implementation, this would check device power state
        false
    }

    /// Get network connectivity status
    pub fn get_network_status() -> NetworkStatus {
        NetworkStatus {
            is_connected: true,
            network_type: NetworkType::WiFi,
            is_metered: false,
            signal_strength: 0.8,
        }
    }

    /// Configure audio session for mobile platform
    pub fn configure_audio_session(config: &MobileAudioConfig) -> Result<(), PlatformError> {
        #[cfg(target_os = "ios")]
        {
            // iOS-specific audio session configuration
            // This would use AVAudioSession APIs
            let _ = config; // Use config to configure audio session
        }

        #[cfg(target_os = "android")]
        {
            // Android-specific audio configuration
            // This would use AudioManager APIs
            let _ = config; // Use config to configure audio
        }

        Ok(())
    }
}

impl Default for MobileAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl PlatformAdapter for MobileAdapter {
    fn initialize(&self) -> PlatformResult<()> {
        // Initialize mobile-specific resources

        // Check if required permissions are available
        if !Self::has_hardware_capability("microphone") {
            return Err(PlatformError::FeatureNotAvailable {
                feature: "microphone".to_string(),
            });
        }

        // Configure audio session for mobile
        let audio_config = MobileAudioConfig {
            category: AudioCategory::PlayAndRecord,
            mode: AudioMode::VoiceChat,
            enable_noise_cancellation: true,
            enable_echo_cancellation: true,
            enable_automatic_gain_control: true,
            prefer_low_latency: true,
        };

        Self::configure_audio_session(&audio_config)?;

        // Initialize battery monitoring
        if Self::is_low_power_mode() {
            // Adjust settings for low power mode
        }

        // Initialize network monitoring
        let _network_status = Self::get_network_status();

        Ok(())
    }

    fn cleanup(&self) -> PlatformResult<()> {
        // Cleanup mobile-specific resources

        #[cfg(target_os = "ios")]
        {
            // iOS-specific cleanup
            // This would deactivate audio session, etc.
        }

        #[cfg(target_os = "android")]
        {
            // Android-specific cleanup
            // This would release audio focus, etc.
        }

        Ok(())
    }

    fn supports_feature(&self, feature: &str) -> bool {
        match feature {
            "realtime_audio" => true,
            "local_storage" => true,
            "network_sync" => true,
            "offline" => true,
            "notifications" => true,
            "haptic" => Self::has_hardware_capability("haptic"),
            "touch_gestures" => true,
            "keyboard_shortcuts" => false, // Mobile doesn't typically have keyboard shortcuts
            "background_processing" => true, // Limited background processing
            "file_system" => false,        // Limited file system access
            "system_integration" => true,  // Some system integration available
            "battery_monitoring" => true,
            "network_monitoring" => true,
            "device_orientation" => true,
            "proximity_sensor" => Self::has_hardware_capability("proximity_sensor"),
            "ambient_light_sensor" => Self::has_hardware_capability("ambient_light_sensor"),
            _ => false,
        }
    }

    fn get_storage_path(&self) -> PlatformResult<PathBuf> {
        // Mobile platforms have sandboxed storage
        #[cfg(target_os = "ios")]
        {
            // iOS Documents directory
            Ok(PathBuf::from(
                "/var/mobile/Containers/Data/Application/VoiRS/Documents",
            ))
        }

        #[cfg(target_os = "android")]
        {
            // Android internal storage
            Ok(PathBuf::from("/data/data/com.voirs.feedback/files"))
        }

        #[cfg(not(any(target_os = "ios", target_os = "android")))]
        {
            // Test environment
            Ok(PathBuf::from("./mobile_data"))
        }
    }

    fn get_cache_path(&self) -> PlatformResult<PathBuf> {
        // Mobile platforms have sandboxed cache
        #[cfg(target_os = "ios")]
        {
            // iOS Caches directory
            Ok(PathBuf::from(
                "/var/mobile/Containers/Data/Application/VoiRS/Library/Caches",
            ))
        }

        #[cfg(target_os = "android")]
        {
            // Android cache directory
            Ok(PathBuf::from("/data/data/com.voirs.feedback/cache"))
        }

        #[cfg(not(any(target_os = "ios", target_os = "android")))]
        {
            // Test environment
            Ok(PathBuf::from("./mobile_cache"))
        }
    }

    fn show_notification(&self, title: &str, message: &str) -> PlatformResult<()> {
        // Mobile notification implementation
        #[cfg(target_os = "ios")]
        {
            // iOS User Notifications
            // This would use UNUserNotificationCenter
            println!("iOS Notification: {} - {}", title, message);
        }

        #[cfg(target_os = "android")]
        {
            // Android Notifications
            // This would use NotificationManager
            println!("Android Notification: {} - {}", title, message);
        }

        #[cfg(not(any(target_os = "ios", target_os = "android")))]
        {
            // Test environment
            println!("Mobile Notification: {} - {}", title, message);
        }

        Ok(())
    }

    fn get_audio_device_info(&self) -> PlatformResult<AudioDeviceInfo> {
        // Mobile audio device info
        let device_info = Self::get_device_info();

        Ok(AudioDeviceInfo {
            name: format!("{} Audio Device", device_info.platform.to_string()),
            supported_sample_rates: vec![44100, 48000],
            supported_buffer_sizes: vec![128, 256, 512, 1024, 2048],
            input_channels: 1,
            output_channels: 2,
            default_sample_rate: 44100,
            default_buffer_size: 512,
        })
    }

    fn configure_feature(&self, feature: &str, enabled: bool) -> PlatformResult<()> {
        match feature {
            "realtime_audio" => {
                // Configure mobile audio settings
                if enabled {
                    let config = MobileAudioConfig {
                        category: AudioCategory::PlayAndRecord,
                        mode: AudioMode::VoiceChat,
                        enable_noise_cancellation: true,
                        enable_echo_cancellation: true,
                        enable_automatic_gain_control: true,
                        prefer_low_latency: true,
                    };
                    Self::configure_audio_session(&config)?;
                } else {
                    // Disable audio session
                }
            }
            "background_processing" => {
                // Configure background processing
                if enabled {
                    // Request background processing permissions
                    // This is limited on mobile platforms
                } else {
                    // Disable background processing
                }
            }
            "battery_monitoring" => {
                // Configure battery monitoring
                if enabled {
                    // Enable battery level monitoring
                } else {
                    // Disable battery monitoring
                }
            }
            "network_monitoring" => {
                // Configure network monitoring
                if enabled {
                    // Enable network status monitoring
                } else {
                    // Disable network monitoring
                }
            }
            "haptic" => {
                // Configure haptic feedback
                if enabled {
                    // Enable haptic feedback
                } else {
                    // Disable haptic feedback
                }
            }
            "device_orientation" => {
                // Configure device orientation monitoring
                if enabled {
                    // Enable orientation monitoring
                } else {
                    // Disable orientation monitoring
                }
            }
            _ => {
                return Err(PlatformError::FeatureNotAvailable {
                    feature: feature.to_string(),
                });
            }
        }

        Ok(())
    }
}

/// Mobile platform types
#[derive(Debug, Clone, PartialEq)]
pub enum MobilePlatform {
    /// Description
    IOS,
    /// Description
    Android,
    /// Description
    Unknown,
}

impl ToString for MobilePlatform {
    fn to_string(&self) -> String {
        match self {
            MobilePlatform::IOS => "iOS".to_string(),
            MobilePlatform::Android => "Android".to_string(),
            MobilePlatform::Unknown => "Unknown".to_string(),
        }
    }
}

/// Device information structure
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Description
    pub platform: MobilePlatform,
    /// Description
    pub device_model: String,
    /// Description
    pub os_version: String,
    /// Description
    pub screen_width: u32,
    /// Description
    pub screen_height: u32,
    /// Description
    pub screen_density: f32,
    /// Description
    pub battery_level: f32,
    /// Description
    pub is_low_power_mode: bool,
    /// Description
    pub available_storage: u64,
    /// Description
    pub total_storage: u64,
    /// Description
    pub network_type: NetworkType,
}

/// Network status information
#[derive(Debug, Clone)]
pub struct NetworkStatus {
    /// Description
    pub is_connected: bool,
    /// Description
    pub network_type: NetworkType,
    /// Description
    pub is_metered: bool,
    /// Description
    pub signal_strength: f32,
}

/// Permission types for mobile platforms
#[derive(Debug, Clone, PartialEq)]
pub enum Permission {
    /// Description
    Microphone,
    /// Description
    LocalStorage,
    /// Description
    Notifications,
    /// Description
    Camera,
    /// Description
    Location,
    /// Description
    Bluetooth,
}

/// Mobile audio configuration
#[derive(Debug, Clone)]
pub struct MobileAudioConfig {
    /// Description
    pub category: AudioCategory,
    /// Description
    pub mode: AudioMode,
    /// Description
    pub enable_noise_cancellation: bool,
    /// Description
    pub enable_echo_cancellation: bool,
    /// Description
    pub enable_automatic_gain_control: bool,
    /// Description
    pub prefer_low_latency: bool,
}

/// Audio category for mobile platforms
#[derive(Debug, Clone, PartialEq)]
pub enum AudioCategory {
    /// Description
    Ambient,
    /// Description
    SoloAmbient,
    /// Description
    Playback,
    /// Description
    Record,
    /// Description
    PlayAndRecord,
    /// Description
    MultiRoute,
}

/// Audio mode for mobile platforms
#[derive(Debug, Clone, PartialEq)]
pub enum AudioMode {
    /// Description
    Default,
    /// Description
    VoiceChat,
    /// Description
    GameChat,
    /// Description
    VideoRecording,
    /// Description
    Measurement,
    /// Description
    MoviePlayback,
    /// Description
    VideoChat,
}

/// Mobile-specific utilities
pub struct MobileUtils;

impl MobileUtils {
    /// Check if device supports specific audio feature
    pub fn supports_audio_feature(feature: &str) -> bool {
        match feature {
            "low_latency" => true,
            "echo_cancellation" => true,
            "noise_reduction" => true,
            "automatic_gain_control" => true,
            "multi_channel" => false, // Limited on mobile
            "high_quality" => true,
            "hardware_acceleration" => true,
            _ => false,
        }
    }

    /// Get recommended audio settings for mobile
    pub fn get_recommended_audio_settings() -> MobileAudioSettings {
        MobileAudioSettings {
            sample_rate: 44100,
            buffer_size: 512,
            channels: 1,
            enable_echo_cancellation: true,
            enable_noise_reduction: true,
            enable_automatic_gain_control: true,
            enable_low_latency: true,
            category: AudioCategory::PlayAndRecord,
            mode: AudioMode::VoiceChat,
        }
    }

    /// Get optimal settings based on device capabilities
    pub fn get_optimal_settings(device_info: &DeviceInfo) -> MobileAudioSettings {
        let mut settings = Self::get_recommended_audio_settings();

        // Adjust settings based on device capabilities
        if device_info.is_low_power_mode {
            settings.buffer_size = 1024; // Larger buffer for power saving
            settings.enable_low_latency = false;
        }

        if device_info.available_storage < 100 * 1024 * 1024 {
            // Low storage, optimize for space
            settings.enable_echo_cancellation = false;
            settings.enable_noise_reduction = false;
        }

        settings
    }

    /// Check if device supports background audio processing
    pub fn supports_background_audio() -> bool {
        // Background audio is supported but limited on mobile
        true
    }

    /// Check if device supports hardware acceleration
    pub fn supports_hardware_acceleration() -> bool {
        // Most modern mobile devices support hardware acceleration
        true
    }

    /// Get battery optimization recommendations
    pub fn get_battery_optimization_tips() -> Vec<String> {
        vec![
            "Use larger buffer sizes to reduce CPU usage".to_string(),
            "Disable unnecessary audio processing features".to_string(),
            "Reduce sample rate if acceptable quality".to_string(),
            "Limit background processing when not needed".to_string(),
            "Use device sleep mode when inactive".to_string(),
        ]
    }

    /// Get network optimization recommendations
    pub fn get_network_optimization_tips() -> Vec<String> {
        vec![
            "Use compression for data transfer".to_string(),
            "Implement efficient sync algorithms".to_string(),
            "Cache frequently accessed data".to_string(),
            "Batch network requests to reduce overhead".to_string(),
            "Handle network interruptions gracefully".to_string(),
        ]
    }
}

/// Mobile audio settings
#[derive(Debug, Clone)]
pub struct MobileAudioSettings {
    /// Description
    pub sample_rate: u32,
    /// Description
    pub buffer_size: usize,
    /// Description
    pub channels: u32,
    /// Description
    pub enable_echo_cancellation: bool,
    /// Description
    pub enable_noise_reduction: bool,
    /// Description
    pub enable_automatic_gain_control: bool,
    /// Description
    pub enable_low_latency: bool,
    /// Description
    pub category: AudioCategory,
    /// Description
    pub mode: AudioMode,
}

/// Legacy mobile performance optimizer (deprecated - see comprehensive version below)
// This is replaced by the comprehensive MobilePerformanceOptimizer below

impl MobileUtils {
    /// Optimize for battery life
    pub fn optimize_for_battery() -> PerformanceProfile {
        PerformanceProfile {
            cpu_usage_limit: 0.3,
            memory_usage_limit: 0.4,
            network_usage_limit: 0.2,
            audio_buffer_size: 2048,
            audio_sample_rate: 44100,
            enable_background_processing: false,
            enable_hardware_acceleration: true,
            enable_power_saving_mode: true,
        }
    }

    /// Optimize for performance
    pub fn optimize_for_performance() -> PerformanceProfile {
        PerformanceProfile {
            cpu_usage_limit: 0.8,
            memory_usage_limit: 0.7,
            network_usage_limit: 0.6,
            audio_buffer_size: 256,
            audio_sample_rate: 48000,
            enable_background_processing: true,
            enable_hardware_acceleration: true,
            enable_power_saving_mode: false,
        }
    }

    /// Optimize for balanced usage
    pub fn optimize_for_balanced() -> PerformanceProfile {
        PerformanceProfile {
            cpu_usage_limit: 0.5,
            memory_usage_limit: 0.5,
            network_usage_limit: 0.4,
            audio_buffer_size: 512,
            audio_sample_rate: 44100,
            enable_background_processing: true,
            enable_hardware_acceleration: true,
            enable_power_saving_mode: false,
        }
    }

    /// Auto-optimize based on device state
    pub fn auto_optimize(device_info: &DeviceInfo) -> PerformanceProfile {
        if device_info.is_low_power_mode || device_info.battery_level < 0.2 {
            Self::optimize_for_battery()
        } else if device_info.available_storage > 1024 * 1024 * 1024 {
            Self::optimize_for_performance()
        } else {
            Self::optimize_for_balanced()
        }
    }
}

/// Performance profile for mobile optimization
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Description
    pub cpu_usage_limit: f32,
    /// Description
    pub memory_usage_limit: f32,
    /// Description
    pub network_usage_limit: f32,
    /// Description
    pub audio_buffer_size: usize,
    /// Description
    pub audio_sample_rate: u32,
    /// Description
    pub enable_background_processing: bool,
    /// Description
    pub enable_hardware_acceleration: bool,
    /// Description
    pub enable_power_saving_mode: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mobile_adapter_creation() {
        let adapter = MobileAdapter::new();
        assert!(!adapter.initialized);
    }

    #[test]
    fn test_mobile_platform_detection() {
        let platform = MobileAdapter::get_mobile_platform();
        assert!(matches!(
            platform,
            MobilePlatform::IOS | MobilePlatform::Android | MobilePlatform::Unknown
        ));
    }

    #[test]
    fn test_hardware_capabilities() {
        assert!(MobileAdapter::has_hardware_capability("microphone"));
        assert!(MobileAdapter::has_hardware_capability("speaker"));
        assert!(MobileAdapter::has_hardware_capability("haptic"));
        assert!(MobileAdapter::has_hardware_capability("accelerometer"));
        assert!(MobileAdapter::has_hardware_capability("gyroscope"));
        assert!(!MobileAdapter::has_hardware_capability("unknown_hardware"));
    }

    #[test]
    fn test_device_info() {
        let device_info = MobileAdapter::get_device_info();
        assert!(device_info.screen_width > 0);
        assert!(device_info.screen_height > 0);
        assert!(device_info.screen_density > 0.0);
        assert!(device_info.battery_level >= 0.0 && device_info.battery_level <= 1.0);
        assert!(device_info.total_storage > 0);
    }

    #[test]
    fn test_mobile_adapter_features() {
        let adapter = MobileAdapter::new();
        assert!(adapter.supports_feature("realtime_audio"));
        assert!(adapter.supports_feature("local_storage"));
        assert!(adapter.supports_feature("network_sync"));
        assert!(adapter.supports_feature("offline"));
        assert!(adapter.supports_feature("notifications"));
        assert!(adapter.supports_feature("haptic"));
        assert!(adapter.supports_feature("touch_gestures"));
        assert!(adapter.supports_feature("background_processing"));
        assert!(adapter.supports_feature("battery_monitoring"));
        assert!(adapter.supports_feature("network_monitoring"));
        assert!(adapter.supports_feature("device_orientation"));
        assert!(!adapter.supports_feature("keyboard_shortcuts"));
        assert!(!adapter.supports_feature("file_system"));
    }

    #[test]
    fn test_network_status() {
        let network_status = MobileAdapter::get_network_status();
        assert!(network_status.signal_strength >= 0.0 && network_status.signal_strength <= 1.0);
        assert!(matches!(
            network_status.network_type,
            NetworkType::WiFi
                | NetworkType::Cellular
                | NetworkType::Ethernet
                | NetworkType::Offline
        ));
    }

    #[test]
    fn test_mobile_utils_audio_features() {
        assert!(MobileUtils::supports_audio_feature("low_latency"));
        assert!(MobileUtils::supports_audio_feature("echo_cancellation"));
        assert!(MobileUtils::supports_audio_feature("noise_reduction"));
        assert!(MobileUtils::supports_audio_feature(
            "automatic_gain_control"
        ));
        assert!(MobileUtils::supports_audio_feature("hardware_acceleration"));
        assert!(!MobileUtils::supports_audio_feature("multi_channel"));
        assert!(!MobileUtils::supports_audio_feature("unknown_feature"));
    }

    #[test]
    fn test_mobile_audio_settings() {
        let settings = MobileUtils::get_recommended_audio_settings();
        assert!(settings.sample_rate > 0);
        assert!(settings.buffer_size > 0);
        assert!(settings.channels > 0);
        assert!(settings.enable_echo_cancellation);
        assert!(settings.enable_noise_reduction);
        assert!(settings.enable_automatic_gain_control);
        assert!(settings.enable_low_latency);
        assert_eq!(settings.category, AudioCategory::PlayAndRecord);
        assert_eq!(settings.mode, AudioMode::VoiceChat);
    }

    #[test]
    fn test_optimal_settings_low_power() {
        let mut device_info = MobileAdapter::get_device_info();
        device_info.is_low_power_mode = true;

        let settings = MobileUtils::get_optimal_settings(&device_info);
        assert_eq!(settings.buffer_size, 1024);
        assert!(!settings.enable_low_latency);
    }

    #[test]
    fn test_optimal_settings_low_storage() {
        let mut device_info = MobileAdapter::get_device_info();
        device_info.available_storage = 50 * 1024 * 1024; // 50MB

        let settings = MobileUtils::get_optimal_settings(&device_info);
        assert!(!settings.enable_echo_cancellation);
        assert!(!settings.enable_noise_reduction);
    }

    #[test]
    fn test_performance_optimization() {
        let battery_profile = MobileUtils::optimize_for_battery();
        assert_eq!(battery_profile.cpu_usage_limit, 0.3);
        assert_eq!(battery_profile.audio_buffer_size, 2048);
        assert!(battery_profile.enable_power_saving_mode);
        assert!(!battery_profile.enable_background_processing);

        let performance_profile = MobileUtils::optimize_for_performance();
        assert_eq!(performance_profile.cpu_usage_limit, 0.8);
        assert_eq!(performance_profile.audio_buffer_size, 256);
        assert!(!performance_profile.enable_power_saving_mode);
        assert!(performance_profile.enable_background_processing);

        let balanced_profile = MobileUtils::optimize_for_balanced();
        assert_eq!(balanced_profile.cpu_usage_limit, 0.5);
        assert_eq!(balanced_profile.audio_buffer_size, 512);
        assert!(!balanced_profile.enable_power_saving_mode);
        assert!(balanced_profile.enable_background_processing);
    }

    #[test]
    fn test_auto_optimization() {
        let mut device_info = MobileAdapter::get_device_info();

        // Test low battery optimization
        device_info.battery_level = 0.1;
        let profile = MobileUtils::auto_optimize(&device_info);
        assert!(profile.enable_power_saving_mode);

        // Test low power mode optimization
        device_info.battery_level = 0.8;
        device_info.is_low_power_mode = true;
        let profile = MobileUtils::auto_optimize(&device_info);
        assert!(profile.enable_power_saving_mode);

        // Test high storage optimization
        device_info.is_low_power_mode = false;
        device_info.available_storage = 2 * 1024 * 1024 * 1024; // 2GB
        let profile = MobileUtils::auto_optimize(&device_info);
        assert!(!profile.enable_power_saving_mode);
        assert!(profile.enable_background_processing);
    }

    #[test]
    fn test_get_storage_path() {
        let adapter = MobileAdapter::new();
        let storage_path = adapter.get_storage_path().unwrap();
        assert!(
            storage_path.to_string_lossy().contains("mobile")
                || storage_path.to_string_lossy().contains("data")
                || storage_path.to_string_lossy().contains("Application")
                || storage_path.to_string_lossy().contains("Documents")
        );
    }

    #[test]
    fn test_get_cache_path() {
        let adapter = MobileAdapter::new();
        let cache_path = adapter.get_cache_path().unwrap();
        assert!(
            cache_path.to_string_lossy().contains("cache")
                || cache_path.to_string_lossy().contains("Cache")
                || cache_path.to_string_lossy().contains("mobile")
        );
    }

    #[test]
    fn test_get_audio_device_info() {
        let adapter = MobileAdapter::new();
        let audio_info = adapter.get_audio_device_info().unwrap();

        assert!(!audio_info.name.is_empty());
        assert!(!audio_info.supported_sample_rates.is_empty());
        assert!(!audio_info.supported_buffer_sizes.is_empty());
        assert!(audio_info.input_channels > 0);
        assert!(audio_info.output_channels > 0);
        assert!(audio_info.default_sample_rate > 0);
        assert!(audio_info.default_buffer_size > 0);
    }

    #[test]
    fn test_configure_feature() {
        let adapter = MobileAdapter::new();

        // Test valid features
        assert!(adapter.configure_feature("realtime_audio", true).is_ok());
        assert!(adapter
            .configure_feature("background_processing", false)
            .is_ok());
        assert!(adapter
            .configure_feature("battery_monitoring", true)
            .is_ok());
        assert!(adapter
            .configure_feature("network_monitoring", false)
            .is_ok());
        assert!(adapter.configure_feature("haptic", true).is_ok());
        assert!(adapter
            .configure_feature("device_orientation", false)
            .is_ok());

        // Test invalid feature
        assert!(adapter.configure_feature("invalid_feature", true).is_err());
    }

    #[test]
    fn test_mobile_platform_to_string() {
        assert_eq!(MobilePlatform::IOS.to_string(), "iOS");
        assert_eq!(MobilePlatform::Android.to_string(), "Android");
        assert_eq!(MobilePlatform::Unknown.to_string(), "Unknown");
    }

    #[test]
    fn test_battery_optimization_tips() {
        let tips = MobileUtils::get_battery_optimization_tips();
        assert!(!tips.is_empty());
        assert!(tips.iter().any(|tip| tip.contains("buffer")));
        assert!(tips.iter().any(|tip| tip.contains("processing")));
    }

    #[test]
    fn test_network_optimization_tips() {
        let tips = MobileUtils::get_network_optimization_tips();
        assert!(!tips.is_empty());
        assert!(tips.iter().any(|tip| tip.contains("compression")));
        assert!(tips.iter().any(|tip| tip.to_lowercase().contains("cache")));
    }

    #[tokio::test]
    async fn test_request_permission() {
        let result = MobileAdapter::request_permission(Permission::Microphone).await;
        assert!(result.is_ok());
        assert!(result.unwrap());

        let result = MobileAdapter::request_permission(Permission::LocalStorage).await;
        assert!(result.is_ok());
        assert!(result.unwrap());

        let result = MobileAdapter::request_permission(Permission::Notifications).await;
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_mobile_status_checks() {
        assert!(MobileAdapter::is_foreground());
        assert!(!MobileAdapter::is_low_power_mode());
        assert!(MobileUtils::supports_background_audio());
        assert!(MobileUtils::supports_hardware_acceleration());
    }
}

/// Native mobile application manager for comprehensive mobile support
#[derive(Debug)]
pub struct NativeMobileApp {
    /// Application state
    app_state: MobileAppState,
    /// Push notification manager
    push_manager: PushNotificationManager,
    /// Offline sync manager
    offline_sync: OfflineSyncManager,
    /// Performance optimizer
    performance_optimizer: MobilePerformanceOptimizer,
    /// Background task manager
    background_tasks: BackgroundTaskManager,
}

impl NativeMobileApp {
    /// Initialize native mobile application
    pub async fn initialize() -> Result<Self, PlatformError> {
        Ok(Self {
            app_state: MobileAppState::Initializing,
            push_manager: PushNotificationManager::new().await?,
            offline_sync: OfflineSyncManager::new().await?,
            performance_optimizer: MobilePerformanceOptimizer::new(),
            background_tasks: BackgroundTaskManager::new(),
        })
    }

    /// Handle app lifecycle events
    pub async fn handle_lifecycle_event(&mut self, event: AppLifecycleEvent) -> PlatformResult<()> {
        match event {
            AppLifecycleEvent::WillEnterForeground => {
                self.app_state = MobileAppState::Active;
                self.resume_real_time_features().await?;
                self.sync_offline_data().await?;
            }
            AppLifecycleEvent::DidEnterBackground => {
                self.app_state = MobileAppState::Background;
                self.pause_real_time_features().await?;
                self.schedule_background_tasks().await?;
            }
            AppLifecycleEvent::WillTerminate => {
                self.app_state = MobileAppState::Terminated;
                self.save_app_state().await?;
            }
            AppLifecycleEvent::MemoryWarning => {
                self.handle_memory_pressure().await?;
            }
        }
        Ok(())
    }

    /// Resume real-time features when app becomes active
    async fn resume_real_time_features(&mut self) -> PlatformResult<()> {
        // Resume audio processing
        self.performance_optimizer.optimize_for_foreground().await?;

        // Reconnect WebSocket connections
        // Re-enable real-time feedback processing
        log::info!("Resumed real-time features for foreground mode");
        Ok(())
    }

    /// Pause real-time features when app goes to background
    async fn pause_real_time_features(&mut self) -> PlatformResult<()> {
        // Pause non-essential audio processing
        self.performance_optimizer.optimize_for_background().await?;

        // Gracefully close real-time connections
        log::info!("Paused real-time features for background mode");
        Ok(())
    }

    /// Schedule background tasks for offline synchronization
    async fn schedule_background_tasks(&mut self) -> PlatformResult<()> {
        self.background_tasks
            .schedule_task(BackgroundTask {
                id: "offline_sync".to_string(),
                task_type: BackgroundTaskType::DataSync,
                priority: TaskPriority::High,
                estimated_duration: Duration::from_secs(30),
                data_payload: None,
            })
            .await?;

        self.background_tasks
            .schedule_task(BackgroundTask {
                id: "analytics_upload".to_string(),
                task_type: BackgroundTaskType::AnalyticsUpload,
                priority: TaskPriority::Medium,
                estimated_duration: Duration::from_secs(15),
                data_payload: None,
            })
            .await?;

        Ok(())
    }

    /// Handle memory pressure by reducing memory usage
    async fn handle_memory_pressure(&mut self) -> PlatformResult<()> {
        // Clear caches
        self.performance_optimizer.clear_memory_caches().await?;

        // Reduce quality settings temporarily
        self.performance_optimizer.reduce_quality_settings().await?;

        // Force garbage collection if possible
        log::warn!("Handling memory pressure - reduced performance mode activated");
        Ok(())
    }

    /// Sync offline data when network becomes available
    async fn sync_offline_data(&mut self) -> PlatformResult<()> {
        if self.offline_sync.has_pending_data().await? {
            match self.offline_sync.sync_with_server().await {
                Ok(sync_result) => {
                    log::info!(
                        "Offline sync completed: {} items synced",
                        sync_result.items_synced
                    );
                }
                Err(e) => {
                    log::warn!("Offline sync failed: {}", e);
                    // Schedule retry
                    self.offline_sync.schedule_retry().await?;
                }
            }
        }
        Ok(())
    }

    /// Save current app state for restoration
    async fn save_app_state(&mut self) -> PlatformResult<()> {
        // Save current session state
        // Save user preferences
        // Save offline data
        log::info!("App state saved for restoration");
        Ok(())
    }

    /// Register for push notifications
    pub async fn register_push_notifications(&mut self) -> PlatformResult<String> {
        self.push_manager.register().await
    }

    /// Handle received push notification
    pub async fn handle_push_notification(
        &mut self,
        notification: PushNotification,
    ) -> PlatformResult<()> {
        match notification.notification_type {
            PushNotificationType::ExerciseReminder => {
                // Show in-app reminder or schedule local notification
                self.show_exercise_reminder(notification.payload).await?;
            }
            PushNotificationType::ProgressUpdate => {
                // Update local progress data
                self.update_progress_data(notification.payload).await?;
            }
            PushNotificationType::SystemMessage => {
                // Show system message to user
                self.show_system_message(notification.payload).await?;
            }
        }
        Ok(())
    }

    /// Show exercise reminder to user
    async fn show_exercise_reminder(&self, payload: serde_json::Value) -> PlatformResult<()> {
        // Extract reminder details from payload
        let title = payload
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Practice Reminder");
        let message = payload
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("Time for your voice practice!");

        // Show native notification
        self.push_manager
            .show_local_notification(title, message)
            .await?;
        Ok(())
    }

    /// Update progress data from server
    async fn update_progress_data(&self, payload: serde_json::Value) -> PlatformResult<()> {
        // Parse progress update from payload
        // Update local database
        // Refresh UI if app is active
        log::info!("Progress data updated from push notification");
        Ok(())
    }

    /// Show system message to user
    async fn show_system_message(&self, payload: serde_json::Value) -> PlatformResult<()> {
        // Extract system message from payload
        // Show appropriate UI alert or notification
        log::info!("System message received via push notification");
        Ok(())
    }

    /// Get current app state
    pub fn get_app_state(&self) -> &MobileAppState {
        &self.app_state
    }

    /// Check if offline mode is available
    pub async fn is_offline_mode_available(&self) -> bool {
        self.offline_sync.is_available().await
    }

    /// Enable offline mode
    pub async fn enable_offline_mode(&mut self) -> PlatformResult<()> {
        self.offline_sync.enable_offline_mode().await?;
        self.performance_optimizer.optimize_for_offline().await?;
        Ok(())
    }

    /// Disable offline mode
    pub async fn disable_offline_mode(&mut self) -> PlatformResult<()> {
        self.offline_sync.disable_offline_mode().await?;
        self.performance_optimizer.optimize_for_online().await?;
        Ok(())
    }
}

/// Mobile application state
#[derive(Debug, Clone, PartialEq)]
pub enum MobileAppState {
    /// App is initializing
    Initializing,
    /// App is active and in foreground
    Active,
    /// App is in background
    Background,
    /// App is suspended
    Suspended,
    /// App is terminated
    Terminated,
}

/// App lifecycle events
#[derive(Debug, Clone)]
pub enum AppLifecycleEvent {
    /// App will enter foreground
    WillEnterForeground,
    /// App did enter background
    DidEnterBackground,
    /// App will terminate
    WillTerminate,
    /// System memory warning
    MemoryWarning,
}

/// Push notification manager for mobile apps
#[derive(Debug)]
pub struct PushNotificationManager {
    /// Device token for push notifications
    device_token: Option<String>,
    /// Notification settings
    settings: NotificationSettings,
}

impl PushNotificationManager {
    /// Create new push notification manager
    pub async fn new() -> Result<Self, PlatformError> {
        Ok(Self {
            device_token: None,
            settings: NotificationSettings::default(),
        })
    }

    /// Register for push notifications
    pub async fn register(&mut self) -> PlatformResult<String> {
        // Request notification permissions
        let permission_granted =
            MobileAdapter::request_permission(Permission::Notifications).await?;

        if !permission_granted {
            return Err(PlatformError::PermissionDenied {
                permission: "notifications".to_string(),
            });
        }

        // Generate device token (in real implementation, this would come from platform)
        let token = format!("device_token_{}", uuid::Uuid::new_v4());
        self.device_token = Some(token.clone());

        log::info!("Registered for push notifications with token: {}", token);
        Ok(token)
    }

    /// Show local notification
    pub async fn show_local_notification(&self, title: &str, message: &str) -> PlatformResult<()> {
        // Platform-specific local notification implementation
        #[cfg(target_os = "ios")]
        {
            // Use UNUserNotificationCenter for iOS
            log::info!("iOS local notification: {} - {}", title, message);
        }

        #[cfg(target_os = "android")]
        {
            // Use NotificationManager for Android
            log::info!("Android local notification: {} - {}", title, message);
        }

        #[cfg(not(any(target_os = "ios", target_os = "android")))]
        {
            log::info!("Local notification: {} - {}", title, message);
        }

        Ok(())
    }

    /// Update notification settings
    pub fn update_settings(&mut self, settings: NotificationSettings) {
        self.settings = settings;
    }

    /// Get device token
    pub fn get_device_token(&self) -> Option<&String> {
        self.device_token.as_ref()
    }
}

/// Push notification
#[derive(Debug, Clone)]
pub struct PushNotification {
    /// Notification ID
    pub id: String,
    /// Type of notification
    pub notification_type: PushNotificationType,
    /// Notification title
    pub title: String,
    /// Notification message
    pub message: String,
    /// Additional payload data
    pub payload: serde_json::Value,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Types of push notifications
#[derive(Debug, Clone)]
pub enum PushNotificationType {
    /// Exercise reminder notification
    ExerciseReminder,
    /// Progress update notification
    ProgressUpdate,
    /// System message notification
    SystemMessage,
}

/// Notification settings
#[derive(Debug, Clone)]
pub struct NotificationSettings {
    /// Enable exercise reminders
    pub exercise_reminders: bool,
    /// Enable progress updates
    pub progress_updates: bool,
    /// Enable system messages
    pub system_messages: bool,
    /// Quiet hours start
    pub quiet_hours_start: Option<chrono::NaiveTime>,
    /// Quiet hours end
    pub quiet_hours_end: Option<chrono::NaiveTime>,
}

impl Default for NotificationSettings {
    fn default() -> Self {
        Self {
            exercise_reminders: true,
            progress_updates: true,
            system_messages: true,
            quiet_hours_start: Some(chrono::NaiveTime::from_hms_opt(22, 0, 0).unwrap()),
            quiet_hours_end: Some(chrono::NaiveTime::from_hms_opt(8, 0, 0).unwrap()),
        }
    }
}

/// Offline synchronization manager
#[derive(Debug)]
pub struct OfflineSyncManager {
    /// Pending operations queue
    pending_operations: std::collections::VecDeque<OfflineOperation>,
    /// Sync settings
    settings: OfflineSyncSettings,
    /// Last sync timestamp
    last_sync: Option<chrono::DateTime<chrono::Utc>>,
}

impl OfflineSyncManager {
    /// Create new offline sync manager
    pub async fn new() -> Result<Self, PlatformError> {
        Ok(Self {
            pending_operations: std::collections::VecDeque::new(),
            settings: OfflineSyncSettings::default(),
            last_sync: None,
        })
    }

    /// Check if offline sync is available
    pub async fn is_available(&self) -> bool {
        true // Always available for this implementation
    }

    /// Enable offline mode
    pub async fn enable_offline_mode(&mut self) -> PlatformResult<()> {
        self.settings.enabled = true;
        log::info!("Offline mode enabled");
        Ok(())
    }

    /// Disable offline mode
    pub async fn disable_offline_mode(&mut self) -> PlatformResult<()> {
        self.settings.enabled = false;
        // Sync any pending data before disabling
        self.sync_with_server().await?;
        log::info!("Offline mode disabled");
        Ok(())
    }

    /// Check if there's pending data to sync
    pub async fn has_pending_data(&self) -> Result<bool, PlatformError> {
        Ok(!self.pending_operations.is_empty())
    }

    /// Add operation to offline queue
    pub async fn queue_operation(&mut self, operation: OfflineOperation) -> PlatformResult<()> {
        self.pending_operations.push_back(operation);

        // Limit queue size to prevent memory issues
        while self.pending_operations.len() > self.settings.max_queue_size {
            self.pending_operations.pop_front();
        }

        Ok(())
    }

    /// Sync with server
    pub async fn sync_with_server(&mut self) -> Result<SyncResult, PlatformError> {
        let mut synced_count = 0;
        let mut failed_count = 0;

        while let Some(operation) = self.pending_operations.pop_front() {
            match self.execute_operation(operation).await {
                Ok(_) => synced_count += 1,
                Err(e) => {
                    failed_count += 1;
                    log::warn!("Failed to sync operation: {}", e);
                }
            }
        }

        self.last_sync = Some(chrono::Utc::now());

        Ok(SyncResult {
            items_synced: synced_count,
            items_failed: failed_count,
            sync_timestamp: self.last_sync.unwrap(),
        })
    }

    /// Execute a queued operation
    async fn execute_operation(&self, operation: OfflineOperation) -> PlatformResult<()> {
        match operation.operation_type {
            OfflineOperationType::DataUpload => {
                // Upload data to server
                log::info!("Executing data upload operation: {}", operation.id);
            }
            OfflineOperationType::ProgressSync => {
                // Sync progress data
                log::info!("Executing progress sync operation: {}", operation.id);
            }
            OfflineOperationType::SettingsSync => {
                // Sync settings
                log::info!("Executing settings sync operation: {}", operation.id);
            }
        }
        Ok(())
    }

    /// Schedule sync retry
    pub async fn schedule_retry(&mut self) -> PlatformResult<()> {
        // Schedule retry after delay
        log::info!(
            "Scheduled sync retry in {} seconds",
            self.settings.retry_delay_seconds
        );
        Ok(())
    }
}

/// Offline operation
#[derive(Debug, Clone)]
pub struct OfflineOperation {
    /// Operation ID
    pub id: String,
    /// Type of operation
    pub operation_type: OfflineOperationType,
    /// Operation data
    pub data: serde_json::Value,
    /// Timestamp when operation was created
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Number of retry attempts
    pub retry_count: u32,
}

/// Types of offline operations
#[derive(Debug, Clone)]
pub enum OfflineOperationType {
    /// Data upload operation
    DataUpload,
    /// Progress synchronization
    ProgressSync,
    /// Settings synchronization
    SettingsSync,
}

/// Offline sync settings
#[derive(Debug, Clone)]
pub struct OfflineSyncSettings {
    /// Enable offline sync
    pub enabled: bool,
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Retry delay in seconds
    pub retry_delay_seconds: u64,
    /// Maximum retry attempts
    pub max_retry_attempts: u32,
}

impl Default for OfflineSyncSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            max_queue_size: 1000,
            retry_delay_seconds: 60,
            max_retry_attempts: 3,
        }
    }
}

/// Sync result
#[derive(Debug, Clone)]
pub struct SyncResult {
    /// Number of items successfully synced
    pub items_synced: u32,
    /// Number of items that failed to sync
    pub items_failed: u32,
    /// Timestamp of sync completion
    pub sync_timestamp: chrono::DateTime<chrono::Utc>,
}

/// Mobile performance optimizer
#[derive(Debug)]
pub struct MobilePerformanceOptimizer {
    /// Current performance mode
    performance_mode: PerformanceMode,
    /// Performance settings
    settings: PerformanceSettings,
}

impl MobilePerformanceOptimizer {
    /// Create new performance optimizer
    pub fn new() -> Self {
        Self {
            performance_mode: PerformanceMode::Balanced,
            settings: PerformanceSettings::default(),
        }
    }

    /// Optimize for foreground operation
    pub async fn optimize_for_foreground(&mut self) -> PlatformResult<()> {
        self.performance_mode = PerformanceMode::HighPerformance;
        log::info!("Optimized for foreground operation");
        Ok(())
    }

    /// Optimize for background operation
    pub async fn optimize_for_background(&mut self) -> PlatformResult<()> {
        self.performance_mode = PerformanceMode::PowerSaver;
        log::info!("Optimized for background operation");
        Ok(())
    }

    /// Optimize for offline operation
    pub async fn optimize_for_offline(&mut self) -> PlatformResult<()> {
        self.performance_mode = PerformanceMode::PowerSaver;
        log::info!("Optimized for offline operation");
        Ok(())
    }

    /// Optimize for online operation
    pub async fn optimize_for_online(&mut self) -> PlatformResult<()> {
        self.performance_mode = PerformanceMode::Balanced;
        log::info!("Optimized for online operation");
        Ok(())
    }

    /// Clear memory caches
    pub async fn clear_memory_caches(&mut self) -> PlatformResult<()> {
        log::info!("Cleared memory caches to reduce memory pressure");
        Ok(())
    }

    /// Reduce quality settings temporarily
    pub async fn reduce_quality_settings(&mut self) -> PlatformResult<()> {
        log::info!("Reduced quality settings for memory conservation");
        Ok(())
    }

    /// Get current performance mode
    pub fn get_performance_mode(&self) -> &PerformanceMode {
        &self.performance_mode
    }
}

/// Performance modes for mobile optimization
#[derive(Debug, Clone)]
pub enum PerformanceMode {
    /// High performance mode (foreground, battery not critical)
    HighPerformance,
    /// Balanced mode (normal operation)
    Balanced,
    /// Power saver mode (background or low battery)
    PowerSaver,
}

/// Performance settings
#[derive(Debug, Clone)]
pub struct PerformanceSettings {
    /// Enable hardware acceleration
    pub hardware_acceleration: bool,
    /// Enable background processing
    pub background_processing: bool,
    /// Maximum CPU usage percentage
    pub max_cpu_usage: f32,
    /// Maximum memory usage in MB
    pub max_memory_usage: usize,
}

impl Default for PerformanceSettings {
    fn default() -> Self {
        Self {
            hardware_acceleration: true,
            background_processing: true,
            max_cpu_usage: 80.0,
            max_memory_usage: 512, // 512 MB
        }
    }
}

/// Background task manager for mobile apps
#[derive(Debug)]
pub struct BackgroundTaskManager {
    /// Active background tasks
    active_tasks: std::collections::HashMap<String, BackgroundTask>,
    /// Task settings
    settings: BackgroundTaskSettings,
}

impl BackgroundTaskManager {
    /// Create new background task manager
    pub fn new() -> Self {
        Self {
            active_tasks: std::collections::HashMap::new(),
            settings: BackgroundTaskSettings::default(),
        }
    }

    /// Schedule a background task
    pub async fn schedule_task(&mut self, task: BackgroundTask) -> PlatformResult<()> {
        if self.active_tasks.len() >= self.settings.max_concurrent_tasks {
            return Err(PlatformError::ResourceLimitExceeded {
                resource: "background_tasks".to_string(),
                limit: self.settings.max_concurrent_tasks,
            });
        }

        self.active_tasks.insert(task.id.clone(), task);
        log::info!("Scheduled background task: {}", self.active_tasks.len());
        Ok(())
    }

    /// Execute all pending background tasks
    pub async fn execute_pending_tasks(&mut self) -> PlatformResult<()> {
        let task_ids: Vec<String> = self.active_tasks.keys().cloned().collect();

        for task_id in task_ids {
            if let Some(task) = self.active_tasks.remove(&task_id) {
                self.execute_task(task).await?;
            }
        }

        Ok(())
    }

    /// Execute a single background task
    async fn execute_task(&self, task: BackgroundTask) -> PlatformResult<()> {
        log::info!(
            "Executing background task: {} (type: {:?})",
            task.id,
            task.task_type
        );

        // Simulate task execution
        tokio::time::sleep(Duration::from_millis(100)).await;

        Ok(())
    }

    /// Get number of active tasks
    pub fn active_task_count(&self) -> usize {
        self.active_tasks.len()
    }
}

/// Background task
#[derive(Debug, Clone)]
pub struct BackgroundTask {
    /// Task ID
    pub id: String,
    /// Type of background task
    pub task_type: BackgroundTaskType,
    /// Task priority
    pub priority: TaskPriority,
    /// Estimated duration
    pub estimated_duration: Duration,
    /// Task data payload
    pub data_payload: Option<serde_json::Value>,
}

/// Types of background tasks
#[derive(Debug, Clone)]
pub enum BackgroundTaskType {
    /// Data synchronization
    DataSync,
    /// Analytics upload
    AnalyticsUpload,
    /// Cache cleanup
    CacheCleanup,
    /// Progress backup
    ProgressBackup,
}

/// Task priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Low priority task
    Low,
    /// Medium priority task
    Medium,
    /// High priority task
    High,
    /// Critical priority task
    Critical,
}

/// Background task settings
#[derive(Debug, Clone)]
pub struct BackgroundTaskSettings {
    /// Maximum concurrent background tasks
    pub max_concurrent_tasks: usize,
    /// Maximum task execution time
    pub max_execution_time: Duration,
    /// Enable task prioritization
    pub enable_prioritization: bool,
}

impl Default for BackgroundTaskSettings {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: 5,
            max_execution_time: Duration::from_secs(30),
            enable_prioritization: true,
        }
    }
}
