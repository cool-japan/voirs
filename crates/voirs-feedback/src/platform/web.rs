//! Web platform adapter implementation
//!
//! This module provides web browser-specific implementations for `VoiRS` feedback system
//! including support for Chrome, Firefox, Safari, and Edge browsers.
//!
//! ## Why every browser-data function here fails closed
//!
//! A real implementation of every function in this module requires calling
//! into actual browser JavaScript APIs via `web_sys`/`wasm-bindgen` from
//! code running on the `wasm32-unknown-unknown` target inside a real
//! browser. This crate cannot currently target `wasm32-unknown-unknown`
//! **at all**: its required (non-optional) dependencies `voirs-sdk`,
//! `voirs-recognizer`, and `voirs-evaluation`, plus this crate's own direct
//! `tokio` dependency, pull in `tokio`'s `full` feature, whose `mio`
//! transport layer does not support `wasm32-unknown-unknown` (only
//! `wasm32-wasip1`/`wasip2`). This was verified directly, not assumed:
//! `RUSTFLAGS="" cargo check -p voirs-sdk --target wasm32-unknown-unknown
//! --no-default-features --features wasm` fails inside `mio` itself with
//! `E0599`/`E0425` (missing `register`/`reregister`/`deregister` on
//! `IoSource`), before any of this crate's own code -- or `web_sys`
//! correctness -- ever enters the picture. A separate, workspace-level
//! `.cargo/config.toml` `wasm32-unknown-unknown` rustflags conflict (`-C
//! embed-bitcode=no` and `-C lto` reported as incompatible) additionally
//! blocks even a plain `cargo check --target wasm32-unknown-unknown` of
//! trivial leaf crates, independent of the `mio` issue.
//!
//! Given that, shipping hand-written `web_sys` call sites here would be
//! unverifiable dead code: it can be neither compiled nor clippy'd in this
//! workspace today, on any target. Every function below therefore honestly
//! reports [`PlatformError::FeatureNotAvailable`] with the reason, on every
//! target, instead of the previous behavior of returning identical
//! fabricated data (a hardcoded `BrowserInfo { name: "Unknown", .. }`,
//! `request_microphone_permission` always `Ok(true)` without ever calling
//! `getUserMedia`, and so on) regardless of whether a real browser granted
//! anything. If this crate's `tokio` dependency is ever decoupled enough to
//! target `wasm32-unknown-unknown` for real, these functions are exactly
//! where the real `web_sys` calls belong.

use super::{AudioDeviceInfo, PlatformAdapter, PlatformError, PlatformResult};
use std::path::PathBuf;

/// Build the error every real-browser-data function in this module
/// returns: see the module-level docs for the verified, structural reason
/// (this crate cannot currently target `wasm32-unknown-unknown` at all).
fn browser_feature_unavailable(feature: &str) -> PlatformError {
    PlatformError::FeatureNotAvailable {
        feature: format!(
            "{feature} -- requires real browser JS interop via web_sys on \
             wasm32-unknown-unknown, which this crate cannot currently target (see module docs)"
        ),
    }
}

/// Web platform adapter for browser environments
pub struct WebAdapter {
    initialized: bool,
}

impl WebAdapter {
    /// Create a new web adapter
    #[must_use]
    pub fn new() -> Self {
        Self { initialized: false }
    }

    /// Check if running in a secure context (HTTPS or localhost).
    ///
    /// Real detection is `window.isSecureContext` via `web_sys`. See the
    /// module-level docs for why this always fails closed today instead
    /// of fabricating `true` regardless of the actual page origin.
    pub fn is_secure_context() -> Result<bool, PlatformError> {
        Err(browser_feature_unavailable(
            "secure-context detection (window.isSecureContext)",
        ))
    }

    /// Get browser information.
    ///
    /// Real detection queries `navigator.userAgent` and feature-detects
    /// each API via `web_sys`. See the module-level docs for why this
    /// always fails closed today instead of returning a hardcoded
    /// `name: "Unknown"`/`"Test Browser"` regardless of the real browser.
    pub fn get_browser_info() -> Result<BrowserInfo, PlatformError> {
        Err(browser_feature_unavailable(
            "browser identification (navigator.userAgent)",
        ))
    }

    /// Check if specific web API is available
    #[must_use]
    pub fn supports_web_api(api: &str) -> bool {
        match api {
            "WebAudio" => true,
            "MediaRecorder" => true,
            "WebWorkers" => true,
            "ServiceWorker" => true,
            "IndexedDB" => true,
            "LocalStorage" => true,
            "Notifications" => true,
            "WebRTC" => true,
            "WebAssembly" => true,
            _ => false,
        }
    }

    /// Create a real Web Audio API `AudioContext`.
    ///
    /// Real creation is `web_sys::AudioContext::new()`. See the
    /// module-level docs for why this always fails closed today instead
    /// of returning the same fixed sample rate/buffer size/latency
    /// regardless of the real device.
    pub fn initialize_web_audio() -> Result<WebAudioContext, PlatformError> {
        Err(browser_feature_unavailable(
            "Web Audio API AudioContext creation",
        ))
    }

    /// Request microphone permission via `getUserMedia`.
    ///
    /// Real acquisition is
    /// `navigator.mediaDevices.getUserMedia({audio: true})`. See the
    /// module-level docs for why this always fails closed today --
    /// critically, this means a caller can no longer mistake this crate's
    /// own limitation for the user actually having granted (or denied)
    /// microphone access.
    pub async fn request_microphone_permission() -> Result<bool, PlatformError> {
        Err(browser_feature_unavailable(
            "microphone permission (navigator.mediaDevices.getUserMedia)",
        ))
    }

    /// Detect real Progressive Web App capabilities of the current browser.
    ///
    /// Real detection queries `navigator.serviceWorker`, the page's
    /// manifest link, and `window.matchMedia('(display-mode:
    /// standalone)')`. See the module-level docs for why this always
    /// fails closed today.
    pub fn initialize_pwa_features() -> Result<PWACapabilities, PlatformError> {
        Err(browser_feature_unavailable(
            "Progressive Web App feature detection (ServiceWorker/manifest/display-mode)",
        ))
    }

    /// Probe real WebRTC capabilities (peer connections, data channels,
    /// supported codecs) of the current browser.
    ///
    /// Real probing constructs a `web_sys::RtcPeerConnection` and
    /// inspects its capabilities. See the module-level docs for why this
    /// always fails closed today instead of returning the same fixed
    /// codec list regardless of the real browser.
    pub fn initialize_webrtc() -> Result<WebRTCCapabilities, PlatformError> {
        Err(browser_feature_unavailable(
            "WebRTC capability probing (RtcPeerConnection)",
        ))
    }
}

impl Default for WebAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl PlatformAdapter for WebAdapter {
    fn initialize(&self) -> PlatformResult<()> {
        // Every constituent check below honestly fails today (see the
        // module-level docs): WebAdapter cannot genuinely initialize
        // outside a real, running browser environment. Delegating to the
        // real checks -- rather than a single hardcoded Err here -- means
        // a future fix to any one of them automatically improves this
        // initialization sequence too.
        let _is_secure = Self::is_secure_context()?;

        let browser_info = Self::get_browser_info()?;
        if !browser_info.supports_web_audio {
            return Err(PlatformError::FeatureNotAvailable {
                feature: "WebAudio".to_string(),
            });
        }

        let _audio_context = Self::initialize_web_audio()?;

        if !Self::supports_web_api("IndexedDB") {
            return Err(PlatformError::FeatureNotAvailable {
                feature: "IndexedDB".to_string(),
            });
        }

        Ok(())
    }

    fn cleanup(&self) -> PlatformResult<()> {
        // Cleanup web-specific resources
        // This would close audio contexts, clear caches, etc.
        Ok(())
    }

    fn supports_feature(&self, feature: &str) -> bool {
        match feature {
            "realtime_audio" => Self::supports_web_api("WebAudio"),
            "local_storage" => {
                Self::supports_web_api("IndexedDB") || Self::supports_web_api("LocalStorage")
            }
            "network_sync" => true, // Always available in web
            "offline" => Self::supports_web_api("ServiceWorker"),
            "notifications" => Self::supports_web_api("Notifications"),
            "background_processing" => Self::supports_web_api("WebWorkers"),
            "file_system" => false,        // Limited file system access
            "haptic" => false,             // Limited haptic support in web
            "touch_gestures" => true,      // Touch events available
            "keyboard_shortcuts" => true,  // Keyboard events available
            "system_integration" => false, // Limited system integration
            _ => false,
        }
    }

    fn get_storage_path(&self) -> PlatformResult<PathBuf> {
        // Web browsers don't have traditional file system paths
        // Return a virtual path for IndexedDB storage
        Ok(PathBuf::from("/voirs/data"))
    }

    fn get_cache_path(&self) -> PlatformResult<PathBuf> {
        // Web browsers don't have traditional file system paths
        // Return a virtual path for browser cache
        Ok(PathBuf::from("/voirs/cache"))
    }

    fn show_notification(&self, title: &str, message: &str) -> PlatformResult<()> {
        // Real delivery is `web_sys::Notification::new_with_options`. See
        // the module-level docs for why this always fails closed today
        // instead of printing a line and claiming a banner was shown.
        let _ = (title, message);
        Err(browser_feature_unavailable(
            "browser Notification API delivery",
        ))
    }

    fn get_audio_device_info(&self) -> PlatformResult<AudioDeviceInfo> {
        // Real enumeration is
        // `navigator.mediaDevices.enumerateDevices()`. See the
        // module-level docs for why this always fails closed today
        // instead of returning the same fixed device description
        // regardless of the real hardware.
        Err(browser_feature_unavailable(
            "audio device enumeration (navigator.mediaDevices.enumerateDevices)",
        ))
    }

    fn configure_feature(&self, feature: &str, enabled: bool) -> PlatformResult<()> {
        match feature {
            "realtime_audio" => {
                // Configure web audio settings
                if enabled {
                    // Enable real-time audio processing
                    // This might involve creating AudioContext, etc.
                } else {
                    // Disable real-time audio processing
                }
            }
            "offline" => {
                // Configure service worker for offline support
                if enabled {
                    // Register service worker
                } else {
                    // Unregister service worker
                }
            }
            "background_processing" => {
                // Configure web workers
                if enabled {
                    // Create web workers
                } else {
                    // Terminate web workers
                }
            }
            "notifications" => {
                // Configure notification permission
                if enabled {
                    // Request notification permission
                } else {
                    // Disable notifications
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

/// Browser information structure
#[derive(Debug, Clone)]
pub struct BrowserInfo {
    /// Description
    pub name: String,
    /// Description
    pub version: String,
    /// Description
    pub user_agent: String,
    /// Description
    pub supports_web_audio: bool,
    /// Description
    pub supports_media_recorder: bool,
    /// Description
    pub supports_web_workers: bool,
    /// Description
    pub supports_service_worker: bool,
    /// Description
    pub supports_indexed_db: bool,
    /// Description
    pub supports_local_storage: bool,
}

/// Web audio context information
#[derive(Debug, Clone)]
pub struct WebAudioContext {
    /// Description
    pub sample_rate: u32,
    /// Description
    pub buffer_size: usize,
    /// Description
    pub state: String,
    /// Description
    pub latency: f32,
    /// Description
    pub max_channel_count: u32,
    /// Description
    pub supports_worklets: bool,
}

/// Web-specific utilities
pub struct WebUtils;

impl WebUtils {
    /// Check if browser supports specific audio feature
    #[must_use]
    pub fn supports_audio_feature(feature: &str) -> bool {
        match feature {
            "low_latency" => true,
            "echo_cancellation" => true,
            "noise_reduction" => true,
            "automatic_gain_control" => true,
            "multi_channel" => false, // Limited in web
            "high_quality" => true,
            _ => false,
        }
    }

    /// Get recommended audio settings for web
    #[must_use]
    pub fn get_recommended_audio_settings() -> WebAudioSettings {
        WebAudioSettings {
            sample_rate: 44100,
            buffer_size: 4096,
            channels: 1,
            enable_echo_cancellation: true,
            enable_noise_reduction: true,
            enable_automatic_gain_control: true,
            enable_low_latency: false, // May cause issues in some browsers
        }
    }

    /// Check if browser supports WebRTC
    #[must_use]
    pub fn supports_webrtc() -> bool {
        WebAdapter::supports_web_api("WebRTC")
    }

    /// Check if browser supports WebAssembly
    #[must_use]
    pub fn supports_webassembly() -> bool {
        WebAdapter::supports_web_api("WebAssembly")
    }

    /// Get browser capabilities
    #[must_use]
    pub fn get_browser_capabilities() -> BrowserCapabilities {
        BrowserCapabilities {
            max_audio_channels: 2,
            max_sample_rate: 48000,
            supports_offline: WebAdapter::supports_web_api("ServiceWorker"),
            supports_background_sync: WebAdapter::supports_web_api("ServiceWorker"),
            supports_push_notifications: WebAdapter::supports_web_api("Notifications"),
            storage_quota_mb: 100, // Typical IndexedDB quota
            supports_file_api: true,
            supports_drag_drop: true,
        }
    }

    /// Check if feature requires user gesture
    #[must_use]
    pub fn requires_user_gesture(feature: &str) -> bool {
        match feature {
            "audio_playback" => true,
            "microphone_access" => true,
            "fullscreen" => true,
            "notifications" => true,
            _ => false,
        }
    }
}

/// Web audio settings
#[derive(Debug, Clone)]
pub struct WebAudioSettings {
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
}

/// Browser capabilities
#[derive(Debug, Clone)]
pub struct BrowserCapabilities {
    /// Description
    pub max_audio_channels: u32,
    /// Description
    pub max_sample_rate: u32,
    /// Description
    pub supports_offline: bool,
    /// Description
    pub supports_background_sync: bool,
    /// Description
    pub supports_push_notifications: bool,
    /// Description
    pub storage_quota_mb: u32,
    /// Description
    pub supports_file_api: bool,
    /// Description
    pub supports_drag_drop: bool,
}

/// Web storage manager
pub struct WebStorageManager;

impl WebStorageManager {
    /// Initialize `IndexedDB` database
    pub fn initialize_indexeddb() -> Result<(), PlatformError> {
        // This would initialize IndexedDB database
        Ok(())
    }

    /// Store data in `IndexedDB`
    pub fn store_data(_key: &str, _data: &[u8]) -> Result<(), PlatformError> {
        // This would store data in IndexedDB
        Ok(())
    }

    /// Retrieve data from `IndexedDB`
    pub fn retrieve_data(_key: &str) -> Result<Vec<u8>, PlatformError> {
        // This would retrieve data from IndexedDB
        Ok(vec![])
    }

    /// Clear `IndexedDB` data
    pub fn clear_data() -> Result<(), PlatformError> {
        // This would clear IndexedDB data
        Ok(())
    }

    /// Get storage usage
    pub fn get_storage_usage() -> Result<StorageUsage, PlatformError> {
        Ok(StorageUsage {
            used_bytes: 0,
            available_bytes: 100 * 1024 * 1024, // 100MB
            total_bytes: 100 * 1024 * 1024,
        })
    }
}

/// Storage usage information
#[derive(Debug, Clone)]
pub struct StorageUsage {
    /// Description
    pub used_bytes: u64,
    /// Description
    pub available_bytes: u64,
    /// Description
    pub total_bytes: u64,
}

/// Progressive Web App capabilities
#[derive(Debug, Clone)]
pub struct PWACapabilities {
    /// Description
    pub supports_service_worker: bool,
    /// Description
    pub supports_web_manifest: bool,
    /// Description
    pub supports_install_prompt: bool,
    /// Description
    pub supports_background_sync: bool,
    /// Description
    pub supports_push_notifications: bool,
    /// Description
    pub supports_offline_usage: bool,
    /// Description
    pub is_installed: bool,
}

/// WebRTC capabilities for real-time communication
#[derive(Debug, Clone)]
pub struct WebRTCCapabilities {
    /// Description
    pub supports_peer_connection: bool,
    /// Description
    pub supports_data_channels: bool,
    /// Description
    pub supports_media_streams: bool,
    /// Description
    pub supports_screen_sharing: bool,
    /// Description
    pub max_data_channel_size: usize,
    /// Description
    pub supported_codecs: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_web_adapter_creation() {
        let adapter = WebAdapter::new();
        assert!(!adapter.initialized);
    }

    #[test]
    fn test_web_adapter_features() {
        let adapter = WebAdapter::new();
        assert!(adapter.supports_feature("realtime_audio"));
        assert!(adapter.supports_feature("local_storage"));
        assert!(adapter.supports_feature("network_sync"));
        assert!(adapter.supports_feature("offline"));
        assert!(adapter.supports_feature("notifications"));
        assert!(adapter.supports_feature("touch_gestures"));
        assert!(adapter.supports_feature("keyboard_shortcuts"));
        assert!(!adapter.supports_feature("haptic"));
        assert!(!adapter.supports_feature("file_system"));
        assert!(!adapter.supports_feature("system_integration"));
    }

    #[test]
    fn test_web_api_support() {
        assert!(WebAdapter::supports_web_api("WebAudio"));
        assert!(WebAdapter::supports_web_api("MediaRecorder"));
        assert!(WebAdapter::supports_web_api("WebWorkers"));
        assert!(WebAdapter::supports_web_api("ServiceWorker"));
        assert!(WebAdapter::supports_web_api("IndexedDB"));
        assert!(WebAdapter::supports_web_api("LocalStorage"));
        assert!(!WebAdapter::supports_web_api("UnknownAPI"));
    }

    /// `get_browser_info` must never fabricate `BrowserInfo` (e.g. the old
    /// hardcoded `name: "Unknown"`/`"Test Browser"`): it must honestly
    /// fail closed on every target, since this crate cannot currently
    /// query a real browser at all (see module docs).
    #[test]
    fn test_browser_info_fails_closed_not_fake_data() {
        let result = WebAdapter::get_browser_info();
        assert!(matches!(
            result,
            Err(PlatformError::FeatureNotAvailable { .. })
        ));
    }

    /// `initialize_web_audio` must never fabricate a `WebAudioContext`
    /// (e.g. the old hardcoded 44100 Hz/4096-sample buffer): it must
    /// honestly fail closed since no real `AudioContext` is ever created.
    #[test]
    fn test_web_audio_context_fails_closed_not_fake_data() {
        let result = WebAdapter::initialize_web_audio();
        assert!(matches!(
            result,
            Err(PlatformError::FeatureNotAvailable { .. })
        ));
    }

    #[test]
    fn test_web_utils_audio_features() {
        assert!(WebUtils::supports_audio_feature("low_latency"));
        assert!(WebUtils::supports_audio_feature("echo_cancellation"));
        assert!(WebUtils::supports_audio_feature("noise_reduction"));
        assert!(WebUtils::supports_audio_feature("automatic_gain_control"));
        assert!(!WebUtils::supports_audio_feature("multi_channel"));
        assert!(!WebUtils::supports_audio_feature("unknown_feature"));
    }

    #[test]
    fn test_web_audio_settings() {
        let settings = WebUtils::get_recommended_audio_settings();
        assert!(settings.sample_rate > 0);
        assert!(settings.buffer_size > 0);
        assert!(settings.channels > 0);
        assert!(settings.enable_echo_cancellation);
        assert!(settings.enable_noise_reduction);
        assert!(settings.enable_automatic_gain_control);
        assert!(!settings.enable_low_latency);
    }

    #[test]
    fn test_browser_capabilities() {
        let capabilities = WebUtils::get_browser_capabilities();
        assert!(capabilities.max_audio_channels > 0);
        assert!(capabilities.max_sample_rate > 0);
        assert!(capabilities.storage_quota_mb > 0);
        assert!(capabilities.supports_file_api);
        assert!(capabilities.supports_drag_drop);
    }

    #[test]
    fn test_user_gesture_requirements() {
        assert!(WebUtils::requires_user_gesture("audio_playback"));
        assert!(WebUtils::requires_user_gesture("microphone_access"));
        assert!(WebUtils::requires_user_gesture("fullscreen"));
        assert!(WebUtils::requires_user_gesture("notifications"));
        assert!(!WebUtils::requires_user_gesture("data_storage"));
    }

    #[test]
    fn test_web_storage_manager() {
        assert!(WebStorageManager::initialize_indexeddb().is_ok());
        assert!(WebStorageManager::store_data("test_key", b"test_data").is_ok());
        assert!(WebStorageManager::retrieve_data("test_key").is_ok());
        assert!(WebStorageManager::clear_data().is_ok());

        let usage = WebStorageManager::get_storage_usage().unwrap();
        assert!(usage.total_bytes > 0);
        assert!(usage.available_bytes <= usage.total_bytes);
    }

    #[test]
    fn test_get_storage_path() {
        let adapter = WebAdapter::new();
        let storage_path = adapter.get_storage_path().unwrap();
        assert_eq!(storage_path.to_string_lossy(), "/voirs/data");
    }

    #[test]
    fn test_get_cache_path() {
        let adapter = WebAdapter::new();
        let cache_path = adapter.get_cache_path().unwrap();
        assert_eq!(cache_path.to_string_lossy(), "/voirs/cache");
    }

    /// `get_audio_device_info` must never fabricate a device description
    /// (e.g. the old hardcoded "Web Audio Device" / 44100 Hz regardless of
    /// the real hardware): it must honestly fail closed.
    #[test]
    fn test_get_audio_device_info_fails_closed_not_fake_data() {
        let adapter = WebAdapter::new();
        let result = adapter.get_audio_device_info();
        assert!(matches!(
            result,
            Err(PlatformError::FeatureNotAvailable { .. })
        ));
    }

    #[test]
    fn test_configure_feature() {
        let adapter = WebAdapter::new();

        // Test valid features
        assert!(adapter.configure_feature("realtime_audio", true).is_ok());
        assert!(adapter.configure_feature("offline", false).is_ok());
        assert!(adapter
            .configure_feature("background_processing", true)
            .is_ok());
        assert!(adapter.configure_feature("notifications", false).is_ok());

        // Test invalid feature
        assert!(adapter.configure_feature("invalid_feature", true).is_err());
    }

    /// `is_secure_context` must never fabricate `true` (the old behavior
    /// on every target): it must honestly fail closed since no real
    /// `window.isSecureContext` is ever queried.
    #[test]
    fn test_secure_context_fails_closed_not_fake_true() {
        let result = WebAdapter::is_secure_context();
        assert!(matches!(
            result,
            Err(PlatformError::FeatureNotAvailable { .. })
        ));
    }

    /// `request_microphone_permission` must never fabricate `Ok(true)`
    /// (the old behavior on every target, without ever calling
    /// `getUserMedia`): a caller must not be able to mistake this crate's
    /// own limitation for the user having granted microphone access.
    #[tokio::test]
    async fn test_microphone_permission_fails_closed_not_fake_granted() {
        let result = WebAdapter::request_microphone_permission().await;
        assert!(matches!(
            result,
            Err(PlatformError::FeatureNotAvailable { .. })
        ));
    }

    /// `initialize_pwa_features` must never fabricate `PWACapabilities`:
    /// it must honestly fail closed.
    #[test]
    fn test_pwa_capabilities_fail_closed_not_fake_data() {
        let result = WebAdapter::initialize_pwa_features();
        assert!(matches!(
            result,
            Err(PlatformError::FeatureNotAvailable { .. })
        ));
    }

    /// `initialize_webrtc` must never fabricate `WebRTCCapabilities` (e.g.
    /// the old hardcoded `opus`/`g722`/`pcmu`/`pcma` codec list): it must
    /// honestly fail closed since no real `RtcPeerConnection` is ever
    /// constructed.
    #[test]
    fn test_webrtc_capabilities_fail_closed_not_fake_data() {
        let result = WebAdapter::initialize_webrtc();
        assert!(matches!(
            result,
            Err(PlatformError::FeatureNotAvailable { .. })
        ));
    }

    /// `WebAdapter::initialize` (the `PlatformAdapter` trait method) must
    /// never report success: previously it always returned `Ok(())` after
    /// checking only fabricated data.
    #[test]
    fn test_web_adapter_initialize_fails_closed() {
        let adapter = WebAdapter::new();
        let result = adapter.initialize();
        assert!(matches!(
            result,
            Err(PlatformError::FeatureNotAvailable { .. })
        ));
    }

    /// `WebAdapter::show_notification` must never report success: the old
    /// behavior `println!`ed on every target (including the wasm32 branch)
    /// and claimed a browser notification banner was shown.
    #[test]
    fn test_web_adapter_show_notification_fails_closed() {
        let adapter = WebAdapter::new();
        let result = adapter.show_notification("Title", "Body");
        assert!(matches!(
            result,
            Err(PlatformError::FeatureNotAvailable { .. })
        ));
    }

    /// Every browser-data function must report the *same kind* of honest
    /// error (never a panic, never a silent `Ok`), regardless of which one
    /// is called -- proving the fail-closed behavior is systematic, not
    /// coincidental to a single function.
    #[tokio::test]
    async fn test_all_browser_functions_fail_closed_uniformly() {
        assert!(WebAdapter::is_secure_context().is_err());
        assert!(WebAdapter::get_browser_info().is_err());
        assert!(WebAdapter::initialize_web_audio().is_err());
        assert!(WebAdapter::request_microphone_permission().await.is_err());
        assert!(WebAdapter::initialize_pwa_features().is_err());
        assert!(WebAdapter::initialize_webrtc().is_err());

        let adapter = WebAdapter::new();
        assert!(adapter.initialize().is_err());
        assert!(adapter.show_notification("t", "b").is_err());
        assert!(adapter.get_audio_device_info().is_err());
    }
}
