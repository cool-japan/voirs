//! Multi-platform compatibility support for VoiRS feedback system
//!
//! This module provides abstractions and implementations for different platforms
//! including desktop applications, web browsers, mobile apps, and cross-platform
//! synchronization capabilities.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

pub mod desktop;
pub mod mobile;
pub mod notification_reliability;
pub mod notifications;
pub mod offline;
pub mod reliable_notifications;
pub mod sync;
pub mod web;

/// Supported platforms for VoiRS feedback system
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Platform {
    /// Desktop application (Windows, macOS, Linux)
    Desktop,
    /// Web browser application
    Web,
    /// Mobile application (iOS, Android)
    Mobile,
    /// Embedded system
    Embedded,
}

/// Platform-specific capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCapabilities {
    /// Platform type
    pub platform: Platform,
    /// Supports real-time audio processing
    pub supports_realtime_audio: bool,
    /// Supports local file storage
    pub supports_local_storage: bool,
    /// Supports network synchronization
    pub supports_network_sync: bool,
    /// Supports offline operation
    pub supports_offline: bool,
    /// Supports haptic feedback
    pub supports_haptic: bool,
    /// Supports system notifications
    pub supports_notifications: bool,
    /// Maximum audio buffer size
    pub max_audio_buffer_size: usize,
    /// Preferred audio sample rate
    pub preferred_sample_rate: u32,
}

impl PlatformCapabilities {
    /// Get default capabilities for desktop platform
    pub fn desktop() -> Self {
        Self {
            platform: Platform::Desktop,
            supports_realtime_audio: true,
            supports_local_storage: true,
            supports_network_sync: true,
            supports_offline: true,
            supports_haptic: false,
            supports_notifications: true,
            max_audio_buffer_size: 8192,
            preferred_sample_rate: 44100,
        }
    }

    /// Get default capabilities for web platform
    pub fn web() -> Self {
        Self {
            platform: Platform::Web,
            supports_realtime_audio: true,
            supports_local_storage: true,
            supports_network_sync: true,
            supports_offline: true,
            supports_haptic: false,
            supports_notifications: true,
            max_audio_buffer_size: 4096,
            preferred_sample_rate: 44100,
        }
    }

    /// Get default capabilities for mobile platform
    pub fn mobile() -> Self {
        Self {
            platform: Platform::Mobile,
            supports_realtime_audio: true,
            supports_local_storage: true,
            supports_network_sync: true,
            supports_offline: true,
            supports_haptic: true,
            supports_notifications: true,
            max_audio_buffer_size: 2048,
            preferred_sample_rate: 44100,
        }
    }

    /// Get default capabilities for embedded platform
    pub fn embedded() -> Self {
        Self {
            platform: Platform::Embedded,
            supports_realtime_audio: true,
            supports_local_storage: false,
            supports_network_sync: false,
            supports_offline: true,
            supports_haptic: false,
            supports_notifications: false,
            max_audio_buffer_size: 1024,
            preferred_sample_rate: 16000,
        }
    }
}

/// Platform-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformConfig {
    /// Target platform
    pub platform: Platform,
    /// Platform capabilities
    pub capabilities: PlatformCapabilities,
    /// Storage configuration
    pub storage: StorageConfig,
    /// Network configuration
    pub network: NetworkConfig,
    /// Audio configuration
    pub audio: AudioConfig,
    /// UI configuration
    pub ui: UIConfig,
}

impl PlatformConfig {
    /// Create configuration for desktop platform
    pub fn desktop() -> Self {
        Self {
            platform: Platform::Desktop,
            capabilities: PlatformCapabilities::desktop(),
            storage: StorageConfig::desktop(),
            network: NetworkConfig::default(),
            audio: AudioConfig::desktop(),
            ui: UIConfig::desktop(),
        }
    }

    /// Create configuration for web platform
    pub fn web() -> Self {
        Self {
            platform: Platform::Web,
            capabilities: PlatformCapabilities::web(),
            storage: StorageConfig::web(),
            network: NetworkConfig::default(),
            audio: AudioConfig::web(),
            ui: UIConfig::web(),
        }
    }

    /// Create configuration for mobile platform
    pub fn mobile() -> Self {
        Self {
            platform: Platform::Mobile,
            capabilities: PlatformCapabilities::mobile(),
            storage: StorageConfig::mobile(),
            network: NetworkConfig::default(),
            audio: AudioConfig::mobile(),
            ui: UIConfig::mobile(),
        }
    }
}

/// Storage configuration for different platforms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Base directory for data storage
    pub base_dir: PathBuf,
    /// Cache directory
    pub cache_dir: PathBuf,
    /// Maximum storage size in bytes
    pub max_storage_size: u64,
    /// Enable encryption at rest
    pub encrypt_at_rest: bool,
    /// Enable automatic cleanup
    pub auto_cleanup: bool,
    /// Backup configuration
    pub backup: BackupConfig,
}

impl StorageConfig {
    /// Desktop storage configuration
    pub fn desktop() -> Self {
        Self {
            base_dir: PathBuf::from("./data"),
            cache_dir: PathBuf::from("./cache"),
            max_storage_size: 10 * 1024 * 1024 * 1024, // 10GB
            encrypt_at_rest: true,
            auto_cleanup: true,
            backup: BackupConfig::desktop(),
        }
    }

    /// Web storage configuration
    pub fn web() -> Self {
        Self {
            base_dir: PathBuf::from("./web_data"),
            cache_dir: PathBuf::from("./web_cache"),
            max_storage_size: 100 * 1024 * 1024, // 100MB
            encrypt_at_rest: false,
            auto_cleanup: true,
            backup: BackupConfig::web(),
        }
    }

    /// Mobile storage configuration
    pub fn mobile() -> Self {
        Self {
            base_dir: PathBuf::from("./mobile_data"),
            cache_dir: PathBuf::from("./mobile_cache"),
            max_storage_size: 500 * 1024 * 1024, // 500MB
            encrypt_at_rest: true,
            auto_cleanup: true,
            backup: BackupConfig::mobile(),
        }
    }
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Enable automatic backups
    pub enable_backup: bool,
    /// Backup interval in hours
    pub backup_interval_hours: u32,
    /// Maximum number of backups to keep
    pub max_backups: u32,
    /// Backup compression
    pub compress_backups: bool,
    /// Remote backup URL
    pub remote_backup_url: Option<String>,
}

impl BackupConfig {
    /// Desktop backup configuration
    pub fn desktop() -> Self {
        Self {
            enable_backup: true,
            backup_interval_hours: 24,
            max_backups: 30,
            compress_backups: true,
            remote_backup_url: None,
        }
    }

    /// Web backup configuration
    pub fn web() -> Self {
        Self {
            enable_backup: true,
            backup_interval_hours: 6,
            max_backups: 10,
            compress_backups: true,
            remote_backup_url: Some("https://api.voirs.com/backup".to_string()),
        }
    }

    /// Mobile backup configuration
    pub fn mobile() -> Self {
        Self {
            enable_backup: true,
            backup_interval_hours: 12,
            max_backups: 20,
            compress_backups: true,
            remote_backup_url: Some("https://api.voirs.com/backup".to_string()),
        }
    }
}

/// Network configuration for different platforms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Enable network synchronization
    pub enable_sync: bool,
    /// Sync server URL
    pub sync_server_url: String,
    /// Connection timeout in seconds
    pub connection_timeout: u64,
    /// Request timeout in seconds
    pub request_timeout: u64,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,
    /// Enable offline mode
    pub enable_offline_mode: bool,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            enable_sync: true,
            sync_server_url: "https://api.voirs.com/sync".to_string(),
            connection_timeout: 30,
            request_timeout: 60,
            max_retries: 3,
            retry_delay_ms: 1000,
            enable_offline_mode: true,
        }
    }
}

/// Audio configuration for different platforms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Sample rate for audio processing
    pub sample_rate: u32,
    /// Audio buffer size
    pub buffer_size: usize,
    /// Number of audio channels
    pub channels: u32,
    /// Audio format bit depth
    pub bit_depth: u32,
    /// Enable audio compression
    pub enable_compression: bool,
    /// Audio quality level (0.0 to 1.0)
    pub quality_level: f32,
    /// Enable noise reduction
    pub enable_noise_reduction: bool,
    /// Enable echo cancellation
    pub enable_echo_cancellation: bool,
}

impl AudioConfig {
    /// Desktop audio configuration
    pub fn desktop() -> Self {
        Self {
            sample_rate: 44100,
            buffer_size: 8192,
            channels: 1,
            bit_depth: 16,
            enable_compression: false,
            quality_level: 1.0,
            enable_noise_reduction: true,
            enable_echo_cancellation: true,
        }
    }

    /// Web audio configuration
    pub fn web() -> Self {
        Self {
            sample_rate: 44100,
            buffer_size: 4096,
            channels: 1,
            bit_depth: 16,
            enable_compression: true,
            quality_level: 0.8,
            enable_noise_reduction: true,
            enable_echo_cancellation: true,
        }
    }

    /// Mobile audio configuration
    pub fn mobile() -> Self {
        Self {
            sample_rate: 44100,
            buffer_size: 2048,
            channels: 1,
            bit_depth: 16,
            enable_compression: true,
            quality_level: 0.7,
            enable_noise_reduction: true,
            enable_echo_cancellation: true,
        }
    }
}

/// UI configuration for different platforms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIConfig {
    /// UI theme
    pub theme: String,
    /// Font size
    pub font_size: u32,
    /// Enable animations
    pub enable_animations: bool,
    /// Enable touch gestures
    pub enable_touch_gestures: bool,
    /// Enable keyboard shortcuts
    pub enable_keyboard_shortcuts: bool,
    /// Screen orientation (for mobile)
    pub screen_orientation: ScreenOrientation,
    /// UI density
    pub ui_density: UIDensity,
}

impl UIConfig {
    /// Desktop UI configuration
    pub fn desktop() -> Self {
        Self {
            theme: "light".to_string(),
            font_size: 14,
            enable_animations: true,
            enable_touch_gestures: false,
            enable_keyboard_shortcuts: true,
            screen_orientation: ScreenOrientation::Landscape,
            ui_density: UIDensity::Standard,
        }
    }

    /// Web UI configuration
    pub fn web() -> Self {
        Self {
            theme: "auto".to_string(),
            font_size: 16,
            enable_animations: true,
            enable_touch_gestures: true,
            enable_keyboard_shortcuts: true,
            screen_orientation: ScreenOrientation::Auto,
            ui_density: UIDensity::Standard,
        }
    }

    /// Mobile UI configuration
    pub fn mobile() -> Self {
        Self {
            theme: "auto".to_string(),
            font_size: 18,
            enable_animations: true,
            enable_touch_gestures: true,
            enable_keyboard_shortcuts: false,
            screen_orientation: ScreenOrientation::Auto,
            ui_density: UIDensity::Compact,
        }
    }
}

/// Screen orientation options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScreenOrientation {
    /// Portrait orientation
    Portrait,
    /// Landscape orientation
    Landscape,
    /// Auto-rotate based on device
    Auto,
}

/// UI density options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UIDensity {
    /// Compact UI for small screens
    Compact,
    /// Standard UI density
    Standard,
    /// Comfortable UI for large screens
    Comfortable,
}

/// Extended platform information including system details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformInfo {
    /// Platform type
    pub platform: Platform,
    /// Operating system name
    pub os_name: String,
    /// Operating system version
    pub os_version: String,
    /// System architecture
    pub architecture: String,
    /// Total system memory in bytes
    pub total_memory: u64,
    /// Available system memory in bytes
    pub available_memory: u64,
    /// Number of CPU cores
    pub cpu_count: u32,
    /// Supports multicore processing
    pub supports_multicore: bool,
    /// Battery level (0.0 to 1.0, or -1.0 if not available)
    pub battery_level: f32,
    /// Network connection type
    pub network_type: NetworkType,
}

/// Network connection type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkType {
    /// WiFi connection
    WiFi,
    /// Cellular connection
    Cellular,
    /// Ethernet connection
    Ethernet,
    /// Bluetooth connection
    Bluetooth,
    /// Offline/No connection
    Offline,
    /// Unknown connection type
    Unknown,
}

/// Platform performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformPerformanceMetrics {
    /// CPU usage percentage (0.0 to 1.0)
    pub cpu_usage: f32,
    /// Memory usage percentage (0.0 to 1.0)
    pub memory_usage: f32,
    /// Battery usage rate (positive for draining, negative for charging)
    pub battery_usage: f32,
    /// Network usage statistics
    pub network_usage: NetworkUsage,
    /// Audio latency in milliseconds
    pub audio_latency: f32,
    /// Render FPS
    pub render_fps: f32,
}

/// Network usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkUsage {
    /// Bytes sent
    pub bytes_sent: u64,
    /// Bytes received
    pub bytes_received: u64,
    /// Packets sent
    pub packets_sent: u64,
    /// Packets received
    pub packets_received: u64,
}

/// Feature support information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSupport {
    /// Whether the feature is supported
    pub supported: bool,
    /// Feature version if available
    pub version: Option<String>,
    /// Known limitations
    pub limitations: Vec<String>,
    /// Whether a fallback is available
    pub fallback_available: bool,
}

/// Platform resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformResourceLimits {
    /// Maximum memory usage in bytes
    pub max_memory_usage: u64,
    /// Maximum CPU usage percentage
    pub max_cpu_usage: f32,
    /// Maximum storage usage in bytes
    pub max_storage_usage: u64,
    /// Maximum network bandwidth in bytes/second
    pub max_network_bandwidth: u64,
    /// Maximum concurrent connections
    pub max_concurrent_connections: u32,
    /// Maximum audio buffer size
    pub max_audio_buffer_size: usize,
}

impl PlatformResourceLimits {
    /// Get default resource limits for platform
    pub fn for_platform(platform: Platform) -> Self {
        match platform {
            Platform::Desktop => Self {
                max_memory_usage: 16 * 1024 * 1024 * 1024,   // 16GB
                max_cpu_usage: 0.8,                          // 80%
                max_storage_usage: 100 * 1024 * 1024 * 1024, // 100GB
                max_network_bandwidth: 1024 * 1024 * 1024,   // 1GB/s
                max_concurrent_connections: 1000,
                max_audio_buffer_size: 16384,
            },
            Platform::Web => Self {
                max_memory_usage: 2 * 1024 * 1024 * 1024, // 2GB
                max_cpu_usage: 0.6,                       // 60%
                max_storage_usage: 1024 * 1024 * 1024,    // 1GB
                max_network_bandwidth: 100 * 1024 * 1024, // 100MB/s
                max_concurrent_connections: 50,
                max_audio_buffer_size: 8192,
            },
            Platform::Mobile => Self {
                max_memory_usage: 1024 * 1024 * 1024,    // 1GB
                max_cpu_usage: 0.4,                      // 40%
                max_storage_usage: 512 * 1024 * 1024,    // 512MB
                max_network_bandwidth: 50 * 1024 * 1024, // 50MB/s
                max_concurrent_connections: 20,
                max_audio_buffer_size: 4096,
            },
            Platform::Embedded => Self {
                max_memory_usage: 256 * 1024 * 1024,     // 256MB
                max_cpu_usage: 0.3,                      // 30%
                max_storage_usage: 128 * 1024 * 1024,    // 128MB
                max_network_bandwidth: 10 * 1024 * 1024, // 10MB/s
                max_concurrent_connections: 5,
                max_audio_buffer_size: 2048,
            },
        }
    }
}

/// Platform detection and management
pub struct PlatformManager {
    /// Current platform configuration
    config: PlatformConfig,
    /// Platform-specific adapters
    adapters: HashMap<Platform, Box<dyn PlatformAdapter>>,
}

impl PlatformManager {
    /// Create a new platform manager
    pub fn new(config: PlatformConfig) -> Self {
        let mut adapters: HashMap<Platform, Box<dyn PlatformAdapter>> = HashMap::new();

        // Register platform adapters
        adapters.insert(Platform::Desktop, Box::new(desktop::DesktopAdapter::new()));
        adapters.insert(Platform::Web, Box::new(web::WebAdapter::new()));
        adapters.insert(Platform::Mobile, Box::new(mobile::MobileAdapter::new()));

        Self { config, adapters }
    }

    /// Enhanced platform detection with system information
    pub fn detect_platform_with_info() -> PlatformInfo {
        let platform = Self::detect_platform();

        PlatformInfo {
            platform,
            os_name: Self::get_os_name(),
            os_version: Self::get_os_version(),
            architecture: Self::get_architecture(),
            total_memory: Self::get_total_memory(),
            available_memory: Self::get_available_memory(),
            cpu_count: Self::get_cpu_count(),
            supports_multicore: Self::supports_multicore(),
            battery_level: Self::get_battery_level(),
            network_type: Self::get_network_type(),
        }
    }

    /// Get operating system name
    fn get_os_name() -> String {
        #[cfg(target_os = "windows")]
        return "Windows".to_string();

        #[cfg(target_os = "macos")]
        return "macOS".to_string();

        #[cfg(target_os = "linux")]
        return "Linux".to_string();

        #[cfg(target_os = "ios")]
        return "iOS".to_string();

        #[cfg(target_os = "android")]
        return "Android".to_string();

        #[cfg(target_arch = "wasm32")]
        return "Web".to_string();

        "Unknown".to_string()
    }

    /// Get operating system version
    fn get_os_version() -> String {
        // In a real implementation, this would query the OS for version info
        "Unknown".to_string()
    }

    /// Get system architecture
    fn get_architecture() -> String {
        #[cfg(target_arch = "x86_64")]
        return "x86_64".to_string();

        #[cfg(target_arch = "x86")]
        return "x86".to_string();

        #[cfg(target_arch = "aarch64")]
        return "aarch64".to_string();

        #[cfg(target_arch = "arm")]
        return "arm".to_string();

        #[cfg(target_arch = "wasm32")]
        return "wasm32".to_string();

        "Unknown".to_string()
    }

    /// Get total system memory in bytes
    fn get_total_memory() -> u64 {
        // This would use platform-specific APIs to get actual memory
        // For now, return a default value
        8 * 1024 * 1024 * 1024 // 8GB default
    }

    /// Get available system memory in bytes
    fn get_available_memory() -> u64 {
        // This would use platform-specific APIs to get available memory
        // For now, return a default value
        4 * 1024 * 1024 * 1024 // 4GB default
    }

    /// Get number of CPU cores
    fn get_cpu_count() -> u32 {
        std::thread::available_parallelism()
            .map(|p| p.get() as u32)
            .unwrap_or(1)
    }

    /// Check if system supports multicore processing
    fn supports_multicore() -> bool {
        Self::get_cpu_count() > 1
    }

    /// Get battery level (0.0 to 1.0, or -1.0 if not available)
    fn get_battery_level() -> f32 {
        // This would use platform-specific APIs to get battery level
        // For now, return -1.0 to indicate not available
        -1.0
    }

    /// Get network connection type
    fn get_network_type() -> NetworkType {
        // This would detect actual network type
        NetworkType::WiFi
    }

    /// Initialize platform-specific resources
    pub fn initialize_resources(&mut self) -> Result<(), PlatformError> {
        if let Some(adapter) = self.get_adapter() {
            adapter.initialize()?;
        }

        // Initialize platform-specific optimizations
        self.initialize_performance_optimizations()?;

        Ok(())
    }

    /// Initialize platform-specific performance optimizations
    fn initialize_performance_optimizations(&self) -> Result<(), PlatformError> {
        match self.config.platform {
            Platform::Desktop => {
                // Enable desktop-specific optimizations
                self.enable_desktop_optimizations()?;
            }
            Platform::Web => {
                // Enable web-specific optimizations
                self.enable_web_optimizations()?;
            }
            Platform::Mobile => {
                // Enable mobile-specific optimizations
                self.enable_mobile_optimizations()?;
            }
            Platform::Embedded => {
                // Enable embedded-specific optimizations
                self.enable_embedded_optimizations()?;
            }
        }

        Ok(())
    }

    /// Enable desktop-specific optimizations
    fn enable_desktop_optimizations(&self) -> Result<(), PlatformError> {
        // Set high performance power plan
        // Enable hardware acceleration
        // Optimize thread pool size
        Ok(())
    }

    /// Enable web-specific optimizations
    fn enable_web_optimizations(&self) -> Result<(), PlatformError> {
        // Enable service worker caching
        // Optimize web worker usage
        // Set up IndexedDB for offline storage
        Ok(())
    }

    /// Enable mobile-specific optimizations
    fn enable_mobile_optimizations(&self) -> Result<(), PlatformError> {
        // Enable battery optimization
        // Reduce background processing
        // Optimize for lower memory usage
        Ok(())
    }

    /// Enable embedded-specific optimizations
    fn enable_embedded_optimizations(&self) -> Result<(), PlatformError> {
        // Minimize memory usage
        // Disable non-essential features
        // Optimize for real-time processing
        Ok(())
    }

    /// Get platform performance metrics
    pub fn get_performance_metrics(&self) -> PlatformPerformanceMetrics {
        PlatformPerformanceMetrics {
            cpu_usage: self.get_cpu_usage(),
            memory_usage: self.get_memory_usage(),
            battery_usage: self.get_battery_usage(),
            network_usage: self.get_network_usage(),
            audio_latency: self.get_audio_latency(),
            render_fps: self.get_render_fps(),
        }
    }

    /// Get current CPU usage percentage
    fn get_cpu_usage(&self) -> f32 {
        // This would use platform-specific APIs to get CPU usage
        0.0
    }

    /// Get current memory usage percentage
    fn get_memory_usage(&self) -> f32 {
        // This would use platform-specific APIs to get memory usage
        0.0
    }

    /// Get current battery usage rate
    fn get_battery_usage(&self) -> f32 {
        // This would use platform-specific APIs to get battery usage
        0.0
    }

    /// Get current network usage
    fn get_network_usage(&self) -> NetworkUsage {
        NetworkUsage {
            bytes_sent: 0,
            bytes_received: 0,
            packets_sent: 0,
            packets_received: 0,
        }
    }

    /// Get current audio latency in milliseconds
    fn get_audio_latency(&self) -> f32 {
        // This would measure actual audio latency
        10.0 // Default 10ms
    }

    /// Get current render FPS
    fn get_render_fps(&self) -> f32 {
        // This would measure actual render FPS
        60.0 // Default 60 FPS
    }

    /// Check if platform supports specific feature with detailed information
    pub fn check_feature_support(&self, feature: &str) -> FeatureSupport {
        let capabilities = &self.config.capabilities;

        match feature {
            "realtime_audio" => FeatureSupport {
                supported: capabilities.supports_realtime_audio,
                version: Some("1.0".to_string()),
                limitations: if capabilities.supports_realtime_audio {
                    vec![]
                } else {
                    vec!["Audio API not available".to_string()]
                },
                fallback_available: false,
            },
            "local_storage" => FeatureSupport {
                supported: capabilities.supports_local_storage,
                version: Some("1.0".to_string()),
                limitations: vec![],
                fallback_available: true,
            },
            "network_sync" => FeatureSupport {
                supported: capabilities.supports_network_sync,
                version: Some("1.0".to_string()),
                limitations: vec![],
                fallback_available: true,
            },
            "offline" => FeatureSupport {
                supported: capabilities.supports_offline,
                version: Some("1.0".to_string()),
                limitations: vec![],
                fallback_available: false,
            },
            "haptic" => FeatureSupport {
                supported: capabilities.supports_haptic,
                version: Some("1.0".to_string()),
                limitations: if !capabilities.supports_haptic {
                    vec!["Haptic hardware not available".to_string()]
                } else {
                    vec![]
                },
                fallback_available: true,
            },
            "notifications" => FeatureSupport {
                supported: capabilities.supports_notifications,
                version: Some("1.0".to_string()),
                limitations: vec![],
                fallback_available: false,
            },
            _ => FeatureSupport {
                supported: false,
                version: None,
                limitations: vec!["Feature not recognized".to_string()],
                fallback_available: false,
            },
        }
    }

    /// Get current platform configuration
    pub fn get_config(&self) -> &PlatformConfig {
        &self.config
    }

    /// Update platform configuration
    pub fn update_config(&mut self, config: PlatformConfig) {
        self.config = config;
    }

    /// Get platform adapter for current platform
    pub fn get_adapter(&self) -> Option<&dyn PlatformAdapter> {
        self.adapters
            .get(&self.config.platform)
            .map(|adapter| adapter.as_ref())
    }

    /// Detect current platform automatically
    pub fn detect_platform() -> Platform {
        #[cfg(target_os = "windows")]
        return Platform::Desktop;

        #[cfg(target_os = "macos")]
        return Platform::Desktop;

        #[cfg(target_os = "linux")]
        return Platform::Desktop;

        #[cfg(target_arch = "wasm32")]
        return Platform::Web;

        #[cfg(target_os = "ios")]
        return Platform::Mobile;

        #[cfg(target_os = "android")]
        return Platform::Mobile;

        #[cfg(not(any(
            target_os = "windows",
            target_os = "macos",
            target_os = "linux",
            target_arch = "wasm32",
            target_os = "ios",
            target_os = "android"
        )))]
        return Platform::Embedded;
    }

    /// Create platform manager with auto-detected platform
    pub fn auto_detect() -> Self {
        let platform = Self::detect_platform();
        let config = match platform {
            Platform::Desktop => PlatformConfig::desktop(),
            Platform::Web => PlatformConfig::web(),
            Platform::Mobile => PlatformConfig::mobile(),
            Platform::Embedded => PlatformConfig {
                platform: Platform::Embedded,
                capabilities: PlatformCapabilities::embedded(),
                storage: StorageConfig::mobile(), // Similar to mobile
                network: NetworkConfig::default(),
                audio: AudioConfig::mobile(),
                ui: UIConfig::mobile(),
            },
        };

        Self::new(config)
    }

    /// Get platform resource limits
    pub fn get_resource_limits(&self) -> PlatformResourceLimits {
        PlatformResourceLimits::for_platform(self.config.platform.clone())
    }

    /// Check if resource usage is within limits
    pub fn check_resource_usage(&self) -> ResourceUsageStatus {
        let limits = self.get_resource_limits();
        let metrics = self.get_performance_metrics();

        ResourceUsageStatus {
            memory_status: if metrics.memory_usage > limits.max_cpu_usage {
                ResourceStatus::Exceeded
            } else if metrics.memory_usage > limits.max_cpu_usage * 0.8 {
                ResourceStatus::Warning
            } else {
                ResourceStatus::Normal
            },
            cpu_status: if metrics.cpu_usage > limits.max_cpu_usage {
                ResourceStatus::Exceeded
            } else if metrics.cpu_usage > limits.max_cpu_usage * 0.8 {
                ResourceStatus::Warning
            } else {
                ResourceStatus::Normal
            },
            storage_status: ResourceStatus::Normal, // Would need actual storage usage
            network_status: ResourceStatus::Normal, // Would need actual network usage
            overall_status: ResourceStatus::Normal, // Would be computed based on all statuses
        }
    }

    /// Optimize platform configuration based on current conditions
    pub fn optimize_for_conditions(&mut self) -> Result<(), PlatformError> {
        let metrics = self.get_performance_metrics();
        let limits = self.get_resource_limits();

        // Adjust audio buffer size based on performance
        if metrics.cpu_usage > limits.max_cpu_usage * 0.8 {
            // Increase buffer size to reduce CPU load
            self.config.audio.buffer_size = std::cmp::min(
                self.config.audio.buffer_size * 2,
                limits.max_audio_buffer_size,
            );
        } else if metrics.cpu_usage < limits.max_cpu_usage * 0.4 {
            // Decrease buffer size to reduce latency
            self.config.audio.buffer_size = std::cmp::max(
                self.config.audio.buffer_size / 2,
                512, // Minimum buffer size
            );
        }

        // Adjust quality settings based on performance
        if metrics.memory_usage > limits.max_cpu_usage * 0.8 {
            // Reduce quality to save memory
            self.config.audio.quality_level = (self.config.audio.quality_level * 0.8).min(1.0);
        }

        Ok(())
    }

    /// Get platform-specific recommendations
    pub fn get_recommendations(&self) -> Vec<PlatformRecommendation> {
        let mut recommendations = Vec::new();
        let metrics = self.get_performance_metrics();
        let limits = self.get_resource_limits();

        // CPU usage recommendations
        if metrics.cpu_usage > limits.max_cpu_usage * 0.8 {
            recommendations.push(PlatformRecommendation {
                category: RecommendationCategory::Performance,
                severity: RecommendationSeverity::High,
                title: "High CPU Usage".to_string(),
                description: "CPU usage is high, consider reducing quality settings".to_string(),
                action: "Reduce audio quality or increase buffer size".to_string(),
            });
        }

        // Memory usage recommendations
        if metrics.memory_usage > limits.max_cpu_usage * 0.8 {
            recommendations.push(PlatformRecommendation {
                category: RecommendationCategory::Performance,
                severity: RecommendationSeverity::High,
                title: "High Memory Usage".to_string(),
                description: "Memory usage is high, consider optimizing settings".to_string(),
                action: "Clear cache or reduce concurrent operations".to_string(),
            });
        }

        // Battery recommendations for mobile
        if self.config.platform == Platform::Mobile && metrics.battery_usage > 0.1 {
            recommendations.push(PlatformRecommendation {
                category: RecommendationCategory::Battery,
                severity: RecommendationSeverity::Medium,
                title: "High Battery Usage".to_string(),
                description: "Battery usage is high, consider power-saving mode".to_string(),
                action: "Enable power-saving features".to_string(),
            });
        }

        recommendations
    }
}

/// Platform adapter trait for platform-specific implementations
pub trait PlatformAdapter: Send + Sync {
    /// Initialize platform-specific resources
    fn initialize(&self) -> Result<(), PlatformError>;

    /// Cleanup platform-specific resources
    fn cleanup(&self) -> Result<(), PlatformError>;

    /// Check if platform supports specific feature
    fn supports_feature(&self, feature: &str) -> bool;

    /// Get platform-specific storage path
    fn get_storage_path(&self) -> Result<PathBuf, PlatformError>;

    /// Get platform-specific cache path
    fn get_cache_path(&self) -> Result<PathBuf, PlatformError>;

    /// Show platform-specific notification
    fn show_notification(&self, title: &str, message: &str) -> Result<(), PlatformError>;

    /// Get platform-specific audio device info
    fn get_audio_device_info(&self) -> Result<AudioDeviceInfo, PlatformError>;

    /// Enable/disable platform-specific features
    fn configure_feature(&self, feature: &str, enabled: bool) -> Result<(), PlatformError>;
}

/// Audio device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioDeviceInfo {
    /// Device name
    pub name: String,
    /// Supported sample rates
    pub supported_sample_rates: Vec<u32>,
    /// Supported buffer sizes
    pub supported_buffer_sizes: Vec<usize>,
    /// Number of input channels
    pub input_channels: u32,
    /// Number of output channels
    pub output_channels: u32,
    /// Default sample rate
    pub default_sample_rate: u32,
    /// Default buffer size
    pub default_buffer_size: usize,
}

impl Default for AudioDeviceInfo {
    fn default() -> Self {
        Self {
            name: "Default Audio Device".to_string(),
            supported_sample_rates: vec![44100, 48000],
            supported_buffer_sizes: vec![512, 1024, 2048, 4096],
            input_channels: 1,
            output_channels: 2,
            default_sample_rate: 44100,
            default_buffer_size: 2048,
        }
    }
}

/// Platform-specific error types
#[derive(Debug, thiserror::Error)]
pub enum PlatformError {
    #[error("Platform not supported: {platform:?}")]
    UnsupportedPlatform { platform: Platform },

    #[error("Feature not available: {feature}")]
    FeatureNotAvailable { feature: String },

    #[error("Storage error: {message}")]
    StorageError { message: String },

    #[error("Audio device error: {message}")]
    AudioDeviceError { message: String },

    #[error("Network error: {message}")]
    NetworkError { message: String },

    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },

    #[error("Initialization error: {message}")]
    InitializationError { message: String },

    #[error("Capacity exceeded: current {current}, max {max}")]
    CapacityExceeded { current: usize, max: usize },

    #[error("Operation timed out")]
    Timeout { message: String },

    #[error("Rate limited: {reason}")]
    RateLimited { reason: String },

    #[error("Permission denied: {permission}")]
    PermissionDenied { permission: String },

    #[error("Resource limit exceeded: {resource} limit {limit}")]
    ResourceLimitExceeded { resource: String, limit: usize },
}

/// Platform-specific result type
pub type PlatformResult<T> = Result<T, PlatformError>;

/// Resource usage status for different system resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageStatus {
    /// Memory usage status
    pub memory_status: ResourceStatus,
    /// CPU usage status  
    pub cpu_status: ResourceStatus,
    /// Storage usage status
    pub storage_status: ResourceStatus,
    /// Network usage status
    pub network_status: ResourceStatus,
    /// Overall system status
    pub overall_status: ResourceStatus,
}

/// Resource status levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResourceStatus {
    /// Normal usage levels
    Normal,
    /// Warning levels - approaching limits
    Warning,
    /// Exceeded limits - action required
    Exceeded,
}

/// Platform-specific recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformRecommendation {
    /// Category of recommendation
    pub category: RecommendationCategory,
    /// Severity level
    pub severity: RecommendationSeverity,
    /// Recommendation title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Recommended action
    pub action: String,
}

/// Recommendation categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationCategory {
    /// Performance optimization
    Performance,
    /// Battery optimization
    Battery,
    /// Network optimization
    Network,
    /// Storage optimization
    Storage,
    /// Security recommendation
    Security,
    /// User experience improvement
    UserExperience,
}

/// Recommendation severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationSeverity {
    /// Low severity - optional improvement
    Low,
    /// Medium severity - recommended action
    Medium,
    /// High severity - urgent action required
    High,
    /// Critical severity - immediate action required
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_detection() {
        let platform = PlatformManager::detect_platform();
        assert!(matches!(
            platform,
            Platform::Desktop | Platform::Web | Platform::Mobile | Platform::Embedded
        ));
    }

    #[test]
    fn test_platform_capabilities() {
        let desktop_caps = PlatformCapabilities::desktop();
        assert!(desktop_caps.supports_realtime_audio);
        assert!(desktop_caps.supports_local_storage);
        assert!(desktop_caps.supports_network_sync);
        assert!(desktop_caps.supports_offline);

        let web_caps = PlatformCapabilities::web();
        assert!(web_caps.supports_realtime_audio);
        assert!(web_caps.supports_local_storage);
        assert!(web_caps.supports_network_sync);

        let mobile_caps = PlatformCapabilities::mobile();
        assert!(mobile_caps.supports_realtime_audio);
        assert!(mobile_caps.supports_haptic);
        assert!(mobile_caps.supports_notifications);
    }

    #[test]
    fn test_platform_config_creation() {
        let desktop_config = PlatformConfig::desktop();
        assert_eq!(desktop_config.platform, Platform::Desktop);
        assert!(desktop_config.capabilities.supports_realtime_audio);

        let web_config = PlatformConfig::web();
        assert_eq!(web_config.platform, Platform::Web);
        assert!(web_config.capabilities.supports_network_sync);

        let mobile_config = PlatformConfig::mobile();
        assert_eq!(mobile_config.platform, Platform::Mobile);
        assert!(mobile_config.capabilities.supports_haptic);
    }

    #[test]
    fn test_storage_config() {
        let desktop_storage = StorageConfig::desktop();
        assert!(desktop_storage.encrypt_at_rest);
        assert!(desktop_storage.auto_cleanup);
        assert!(desktop_storage.max_storage_size > 0);

        let web_storage = StorageConfig::web();
        assert!(!web_storage.encrypt_at_rest);
        assert!(web_storage.max_storage_size < desktop_storage.max_storage_size);
    }

    #[test]
    fn test_audio_config() {
        let desktop_audio = AudioConfig::desktop();
        assert_eq!(desktop_audio.sample_rate, 44100);
        assert!(desktop_audio.buffer_size > 0);
        assert_eq!(desktop_audio.channels, 1);

        let mobile_audio = AudioConfig::mobile();
        assert!(mobile_audio.enable_compression);
        assert!(mobile_audio.quality_level < 1.0);
    }

    #[test]
    fn test_ui_config() {
        let desktop_ui = UIConfig::desktop();
        assert!(desktop_ui.enable_keyboard_shortcuts);
        assert!(!desktop_ui.enable_touch_gestures);

        let mobile_ui = UIConfig::mobile();
        assert!(mobile_ui.enable_touch_gestures);
        assert!(!mobile_ui.enable_keyboard_shortcuts);
    }

    #[test]
    fn test_platform_manager_creation() {
        let config = PlatformConfig::desktop();
        let manager = PlatformManager::new(config);
        assert_eq!(manager.get_config().platform, Platform::Desktop);
    }

    #[test]
    fn test_auto_detection() {
        let manager = PlatformManager::auto_detect();
        let config = manager.get_config();
        assert!(matches!(
            config.platform,
            Platform::Desktop | Platform::Web | Platform::Mobile | Platform::Embedded
        ));
    }

    #[test]
    fn test_serialization() {
        let config = PlatformConfig::desktop();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: PlatformConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(config.platform, deserialized.platform);
    }

    #[test]
    fn test_platform_info_detection() {
        let info = PlatformManager::detect_platform_with_info();
        assert!(matches!(
            info.platform,
            Platform::Desktop | Platform::Web | Platform::Mobile | Platform::Embedded
        ));
        assert!(!info.os_name.is_empty());
        assert!(!info.architecture.is_empty());
        assert!(info.total_memory > 0);
        assert!(info.available_memory > 0);
        assert!(info.cpu_count > 0);
        assert!(info.battery_level >= -1.0 && info.battery_level <= 1.0);
        assert!(matches!(
            info.network_type,
            NetworkType::WiFi
                | NetworkType::Cellular
                | NetworkType::Ethernet
                | NetworkType::Bluetooth
                | NetworkType::Offline
                | NetworkType::Unknown
        ));
    }

    #[test]
    fn test_resource_limits() {
        let desktop_limits = PlatformResourceLimits::for_platform(Platform::Desktop);
        let mobile_limits = PlatformResourceLimits::for_platform(Platform::Mobile);

        assert!(desktop_limits.max_memory_usage > mobile_limits.max_memory_usage);
        assert!(desktop_limits.max_cpu_usage > mobile_limits.max_cpu_usage);
        assert!(desktop_limits.max_storage_usage > mobile_limits.max_storage_usage);
        assert!(desktop_limits.max_network_bandwidth > mobile_limits.max_network_bandwidth);
        assert!(
            desktop_limits.max_concurrent_connections > mobile_limits.max_concurrent_connections
        );
        assert!(desktop_limits.max_audio_buffer_size > mobile_limits.max_audio_buffer_size);
    }

    #[test]
    fn test_platform_manager_initialization() {
        let config = PlatformConfig::desktop();
        let mut manager = PlatformManager::new(config);

        assert!(manager.initialize_resources().is_ok());

        let metrics = manager.get_performance_metrics();
        assert!(metrics.cpu_usage >= 0.0);
        assert!(metrics.memory_usage >= 0.0);
        assert!(metrics.audio_latency >= 0.0);
        assert!(metrics.render_fps >= 0.0);
    }

    #[test]
    fn test_feature_support_checking() {
        let config = PlatformConfig::desktop();
        let manager = PlatformManager::new(config);

        let audio_support = manager.check_feature_support("realtime_audio");
        assert!(audio_support.supported);
        assert!(audio_support.version.is_some());
        assert!(audio_support.limitations.is_empty());

        let haptic_support = manager.check_feature_support("haptic");
        assert!(!haptic_support.supported); // Desktop doesn't support haptic
        assert!(!haptic_support.limitations.is_empty());

        let unknown_support = manager.check_feature_support("unknown_feature");
        assert!(!unknown_support.supported);
        assert!(!unknown_support.limitations.is_empty());
    }

    #[test]
    fn test_resource_usage_checking() {
        let config = PlatformConfig::desktop();
        let manager = PlatformManager::new(config);

        let usage_status = manager.check_resource_usage();
        assert!(matches!(
            usage_status.memory_status,
            ResourceStatus::Normal | ResourceStatus::Warning | ResourceStatus::Exceeded
        ));
        assert!(matches!(
            usage_status.cpu_status,
            ResourceStatus::Normal | ResourceStatus::Warning | ResourceStatus::Exceeded
        ));
        assert!(matches!(
            usage_status.storage_status,
            ResourceStatus::Normal | ResourceStatus::Warning | ResourceStatus::Exceeded
        ));
        assert!(matches!(
            usage_status.network_status,
            ResourceStatus::Normal | ResourceStatus::Warning | ResourceStatus::Exceeded
        ));
        assert!(matches!(
            usage_status.overall_status,
            ResourceStatus::Normal | ResourceStatus::Warning | ResourceStatus::Exceeded
        ));
    }

    #[test]
    fn test_platform_optimization() {
        let config = PlatformConfig::desktop();
        let mut manager = PlatformManager::new(config);

        let original_buffer_size = manager.get_config().audio.buffer_size;

        assert!(manager.optimize_for_conditions().is_ok());

        // Buffer size may have changed based on simulated conditions
        let new_buffer_size = manager.get_config().audio.buffer_size;
        assert!(new_buffer_size >= 512); // Minimum buffer size
    }

    #[test]
    fn test_platform_recommendations() {
        let config = PlatformConfig::mobile();
        let manager = PlatformManager::new(config);

        let recommendations = manager.get_recommendations();
        // May have recommendations based on simulated conditions
        for recommendation in recommendations {
            assert!(!recommendation.title.is_empty());
            assert!(!recommendation.description.is_empty());
            assert!(!recommendation.action.is_empty());
            assert!(matches!(
                recommendation.category,
                RecommendationCategory::Performance
                    | RecommendationCategory::Battery
                    | RecommendationCategory::Network
                    | RecommendationCategory::Storage
                    | RecommendationCategory::Security
                    | RecommendationCategory::UserExperience
            ));
            assert!(matches!(
                recommendation.severity,
                RecommendationSeverity::Low
                    | RecommendationSeverity::Medium
                    | RecommendationSeverity::High
                    | RecommendationSeverity::Critical
            ));
        }
    }

    #[test]
    fn test_network_type_serialization() {
        let network_types = vec![
            NetworkType::WiFi,
            NetworkType::Cellular,
            NetworkType::Ethernet,
            NetworkType::Bluetooth,
            NetworkType::Offline,
            NetworkType::Unknown,
        ];

        for network_type in network_types {
            let serialized = serde_json::to_string(&network_type).unwrap();
            let deserialized: NetworkType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(
                std::mem::discriminant(&network_type),
                std::mem::discriminant(&deserialized)
            );
        }
    }

    #[test]
    fn test_platform_info_serialization() {
        let info = PlatformManager::detect_platform_with_info();
        let serialized = serde_json::to_string(&info).unwrap();
        let deserialized: PlatformInfo = serde_json::from_str(&serialized).unwrap();
        assert_eq!(info.platform, deserialized.platform);
        assert_eq!(info.os_name, deserialized.os_name);
        assert_eq!(info.architecture, deserialized.architecture);
    }

    #[test]
    fn test_resource_status_comparison() {
        assert!(ResourceStatus::Normal != ResourceStatus::Warning);
        assert!(ResourceStatus::Warning != ResourceStatus::Exceeded);
        assert!(ResourceStatus::Normal != ResourceStatus::Exceeded);

        let normal_status = ResourceStatus::Normal;
        let warning_status = ResourceStatus::Warning;
        let exceeded_status = ResourceStatus::Exceeded;

        assert_eq!(normal_status, ResourceStatus::Normal);
        assert_eq!(warning_status, ResourceStatus::Warning);
        assert_eq!(exceeded_status, ResourceStatus::Exceeded);
    }
}
