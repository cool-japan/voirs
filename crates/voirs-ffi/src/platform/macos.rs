//! macOS-specific platform integration for VoiRS FFI
//!
//! This module provides macOS-specific functionality including:
//! - Core Audio framework integration
//! - AVFoundation support
//! - Objective-C runtime bindings
//! - macOS performance monitoring

use crate::error::VoirsFFIError;
use std::ffi::{CStr, CString};
use std::process::Command;
use std::ptr;
use std::sync::OnceLock;

use super::parsers;

#[cfg(feature = "macos-platform")]
use cpal::{
    self,
    traits::{DeviceTrait, HostTrait},
};

// macOS Core Foundation and Core Audio types (placeholders for non-macOS builds)
#[cfg(target_os = "macos")]
type CFStringRef = *const std::ffi::c_void;
#[cfg(target_os = "macos")]
type AudioDeviceID = u32;
#[cfg(target_os = "macos")]
type OSStatus = i32;

#[cfg(not(target_os = "macos"))]
type CFStringRef = *const std::ffi::c_void;
#[cfg(not(target_os = "macos"))]
type AudioDeviceID = u32;
#[cfg(not(target_os = "macos"))]
type OSStatus = i32;

/// macOS Core Audio integration
pub struct MacOSCoreAudio {
    initialized: bool,
    default_output_device: AudioDeviceID,
}

impl MacOSCoreAudio {
    /// Initialize Core Audio system
    pub fn new() -> Result<Self, VoirsFFIError> {
        #[cfg(target_os = "macos")]
        {
            // Implementation would use Core Audio APIs
            // For now, return a working placeholder
            Ok(MacOSCoreAudio {
                initialized: true,
                default_output_device: 0,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(VoirsFFIError::PlatformError(
                "Core Audio not available on non-macOS platforms".to_string(),
            ))
        }
    }

    /// Get available audio devices.
    ///
    /// With the `macos-platform` feature enabled, this enumerates devices via
    /// `cpal` (richer: exposes cpal's own default-config heuristics). In the
    /// Pure-Rust default build (no `macos-platform`), and as a fallback if
    /// cpal's enumeration comes back empty, devices are parsed from
    /// `system_profiler SPAudioDataType -json` -- a real OS query, never a
    /// fabricated device list.
    pub fn get_audio_devices(&self) -> Result<Vec<AudioDevice>, VoirsFFIError> {
        #[cfg(target_os = "macos")]
        {
            if !self.initialized {
                return Err(VoirsFFIError::PlatformError(
                    "Core Audio not initialized".to_string(),
                ));
            }

            #[cfg(feature = "macos-platform")]
            {
                let mut devices = Vec::new();
                let host = cpal::default_host();
                // Get output devices
                if let Ok(output_devices) = host.output_devices() {
                    for (index, device) in output_devices.enumerate() {
                        // cpal 0.18 removed `DeviceTrait::name()`; the device
                        // name is now obtained via `Display` (`to_string()`),
                        // which is infallible (see cpal's `DeviceTrait::description`
                        // doc comment for the migration guidance).
                        let device_name = device.to_string();
                        let sample_rate = device
                            .default_output_config()
                            .map(|config| config.sample_rate() as f64)
                            .unwrap_or(44100.0);

                        let channels = device
                            .default_output_config()
                            .map(|config| config.channels() as u32)
                            .unwrap_or(2);

                        devices.push(AudioDevice {
                            id: index as u32 + 1,
                            name: device_name,
                            is_default: index == 0,
                            sample_rate,
                            channels,
                            is_input: false,
                        });
                    }
                }

                // Get input devices
                if let Ok(input_devices) = host.input_devices() {
                    for (index, device) in input_devices.enumerate() {
                        // See the matching comment in the output-devices loop
                        // above: cpal 0.18 requires `to_string()` (via `Display`)
                        // instead of the removed `DeviceTrait::name()`.
                        let device_name = device.to_string();
                        let sample_rate = device
                            .default_input_config()
                            .map(|config| config.sample_rate() as f64)
                            .unwrap_or(44100.0);

                        let channels = device
                            .default_input_config()
                            .map(|config| config.channels() as u32)
                            .unwrap_or(1);

                        devices.push(AudioDevice {
                            id: (index + 1000) as u32, // Offset input device IDs
                            name: device_name,
                            is_default: index == 0,
                            sample_rate,
                            channels,
                            is_input: true,
                        });
                    }
                }

                if !devices.is_empty() {
                    return Ok(devices);
                }
                // cpal enumeration came back empty (unusual, e.g. sandboxed CI
                // without audio hardware) -- fall through to the pure
                // `system_profiler` query below as a secondary *real* source
                // rather than fabricating a device list.
            }

            Self::query_system_profiler_audio_devices()
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(VoirsFFIError::PlatformError(
                "Core Audio not available".to_string(),
            ))
        }
    }

    /// Query `system_profiler SPAudioDataType -json` for the real audio
    /// device list. The result is cached process-wide after the first
    /// (relatively slow, ~1s) call.
    #[cfg(target_os = "macos")]
    fn query_system_profiler_audio_devices() -> Result<Vec<AudioDevice>, VoirsFFIError> {
        static CACHE: OnceLock<Vec<AudioDevice>> = OnceLock::new();

        if let Some(cached) = CACHE.get() {
            return Ok(cached.clone());
        }

        let output = Command::new("system_profiler")
            .args(["SPAudioDataType", "-json"])
            .output()
            .map_err(|e| {
                VoirsFFIError::PlatformError(format!("Failed to run system_profiler: {e}"))
            })?;

        if !output.status.success() {
            return Err(VoirsFFIError::PlatformError(
                "system_profiler exited with a non-zero status".to_string(),
            ));
        }

        let json_str = String::from_utf8_lossy(&output.stdout);
        let parsed = parsers::parse_system_profiler_audio(&json_str);
        if parsed.is_empty() {
            return Err(VoirsFFIError::PlatformError(
                "system_profiler reported no audio devices".to_string(),
            ));
        }

        let devices: Vec<AudioDevice> = parsed
            .into_iter()
            .map(|d| AudioDevice {
                id: d.index + 1,
                name: d.name,
                is_default: d.is_default,
                sample_rate: d.sample_rate,
                channels: d.channels,
                is_input: d.is_input,
            })
            .collect();

        // Best-effort cache: if another thread raced us and populated it
        // first, keep that value rather than erroring.
        let _ = CACHE.set(devices.clone());
        Ok(devices)
    }

    /// Set audio device sample rate
    pub fn set_device_sample_rate(
        &mut self,
        device_id: AudioDeviceID,
        sample_rate: f64,
    ) -> Result<(), VoirsFFIError> {
        #[cfg(target_os = "macos")]
        {
            if !self.initialized {
                return Err(VoirsFFIError::PlatformError(
                    "Core Audio not initialized".to_string(),
                ));
            }

            if sample_rate < 8000.0 || sample_rate > 192000.0 {
                return Err(VoirsFFIError::InvalidParameter(
                    "Invalid sample rate".to_string(),
                ));
            }

            // Implementation would use AudioDeviceSetProperty
            // For now, return success as placeholder
            let _ = device_id;
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (device_id, sample_rate);
            Err(VoirsFFIError::PlatformError(
                "Core Audio not available".to_string(),
            ))
        }
    }

    /// Get system volume using Core Audio (via `osascript`, a Pure-Rust
    /// shell-out in the same spirit as this crate's `sysctl`/`pmset` usage
    /// elsewhere in the `platform` module).
    pub fn get_system_volume(&self) -> Result<f32, VoirsFFIError> {
        #[cfg(target_os = "macos")]
        {
            if !self.initialized {
                return Err(VoirsFFIError::PlatformError(
                    "Core Audio not initialized".to_string(),
                ));
            }

            let output = Command::new("osascript")
                .args(["-e", "output volume of (get volume settings)"])
                .output()
                .map_err(|e| {
                    VoirsFFIError::PlatformError(format!("Failed to run osascript: {e}"))
                })?;

            if !output.status.success() {
                return Err(VoirsFFIError::PlatformError(
                    "osascript failed to read the system volume".to_string(),
                ));
            }

            let stdout = String::from_utf8_lossy(&output.stdout);
            Ok(parsers::parse_volume_output(&stdout))
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(VoirsFFIError::PlatformError(
                "Core Audio not available".to_string(),
            ))
        }
    }
}

/// Audio device information
#[derive(Debug, Clone)]
pub struct AudioDevice {
    pub id: AudioDeviceID,
    pub name: String,
    pub is_default: bool,
    pub sample_rate: f64,
    pub channels: u32,
    pub is_input: bool,
}

/// macOS AVFoundation integration
pub struct MacOSAVFoundation {
    initialized: bool,
}

impl MacOSAVFoundation {
    /// Initialize AVFoundation
    pub fn new() -> Result<Self, VoirsFFIError> {
        #[cfg(target_os = "macos")]
        {
            // Implementation would initialize AVFoundation
            Ok(MacOSAVFoundation { initialized: true })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(VoirsFFIError::PlatformError(
                "AVFoundation not available on non-macOS platforms".to_string(),
            ))
        }
    }

    /// Request microphone permission
    pub fn request_microphone_permission(&self) -> Result<bool, VoirsFFIError> {
        #[cfg(target_os = "macos")]
        {
            if !self.initialized {
                return Err(VoirsFFIError::PlatformError(
                    "AVFoundation not initialized".to_string(),
                ));
            }

            // Implementation would use AVAudioSession.requestRecordPermission
            // For now, return granted as placeholder
            Ok(true)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(VoirsFFIError::PlatformError(
                "AVFoundation not available".to_string(),
            ))
        }
    }

    /// Check if microphone permission is granted
    pub fn has_microphone_permission(&self) -> Result<bool, VoirsFFIError> {
        #[cfg(target_os = "macos")]
        {
            if !self.initialized {
                return Err(VoirsFFIError::PlatformError(
                    "AVFoundation not initialized".to_string(),
                ));
            }

            // Implementation would check AVAudioSession.recordPermission
            // For now, return granted as placeholder
            Ok(true)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(VoirsFFIError::PlatformError(
                "AVFoundation not available".to_string(),
            ))
        }
    }

    /// Configure audio session for speech synthesis
    pub fn configure_synthesis_session(&mut self) -> Result<(), VoirsFFIError> {
        #[cfg(target_os = "macos")]
        {
            if !self.initialized {
                return Err(VoirsFFIError::PlatformError(
                    "AVFoundation not initialized".to_string(),
                ));
            }

            // Implementation would configure AVAudioSession
            // Categories: AVAudioSessionCategoryPlayback for synthesis
            // Options: AVAudioSessionCategoryOptionDuckOthers, etc.
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(VoirsFFIError::PlatformError(
                "AVFoundation not available".to_string(),
            ))
        }
    }
}

/// macOS Objective-C runtime utilities
pub struct MacOSObjectiveC;

impl MacOSObjectiveC {
    /// Get system language preference via `defaults read -g AppleLocale`
    /// (equivalent in effect to `NSLocale.current.identifier`, without
    /// requiring an Objective-C runtime bridge).
    pub fn get_system_language() -> Result<String, VoirsFFIError> {
        #[cfg(target_os = "macos")]
        {
            let output = Command::new("defaults")
                .args(["read", "-g", "AppleLocale"])
                .output()
                .map_err(|e| {
                    VoirsFFIError::PlatformError(format!("Failed to run defaults: {e}"))
                })?;

            if !output.status.success() {
                return Err(VoirsFFIError::PlatformError(
                    "defaults read -g AppleLocale failed".to_string(),
                ));
            }

            let stdout = String::from_utf8_lossy(&output.stdout);
            parsers::parse_locale(&stdout).ok_or_else(|| {
                VoirsFFIError::PlatformError("AppleLocale value was empty".to_string())
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(VoirsFFIError::PlatformError(
                "Objective-C runtime not available".to_string(),
            ))
        }
    }

    /// Get system appearance (light/dark mode) via
    /// `defaults read -g AppleInterfaceStyle`. macOS represents "Light mode"
    /// by *leaving this key unset* (so the command exits non-zero), and
    /// "Dark mode" by setting it to `"Dark"` -- so a failed command here is
    /// the normal, expected signal for light mode, not an error.
    pub fn get_system_appearance() -> Result<String, VoirsFFIError> {
        #[cfg(target_os = "macos")]
        {
            let output = Command::new("defaults")
                .args(["read", "-g", "AppleInterfaceStyle"])
                .output()
                .map_err(|e| {
                    VoirsFFIError::PlatformError(format!("Failed to run defaults: {e}"))
                })?;

            let stdout = String::from_utf8_lossy(&output.stdout);
            let raw = output.status.success().then_some(stdout.as_ref());

            Ok(parsers::parse_appearance(raw).to_string())
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(VoirsFFIError::PlatformError(
                "Objective-C runtime not available".to_string(),
            ))
        }
    }

    /// Show native notification
    pub fn show_notification(title: &str, message: &str) -> Result<(), VoirsFFIError> {
        #[cfg(target_os = "macos")]
        {
            // Implementation would use NSUserNotification
            // For now, return success as placeholder
            let _ = (title, message);
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (title, message);
            Err(VoirsFFIError::PlatformError(
                "Objective-C runtime not available".to_string(),
            ))
        }
    }
}

/// macOS performance monitoring using system APIs
pub struct MacOSPerformanceMonitor;

impl MacOSPerformanceMonitor {
    /// Get macOS-specific performance metrics.
    ///
    /// Sources (all real OS queries, zero fabricated values):
    /// - `cpu_usage`: sum of `ps -A -o %cpu=` normalized by core count.
    /// - `memory_pressure`: `sysctl kern.memorystatus_vm_pressure_level`,
    ///   normalized from XNU's 1/2/4 (normal/warn/critical) levels to 0.0-1.0.
    /// - `thermal_state` / `power_state`: `pmset -g therm` / `pmset -g batt`.
    /// - `audio_latency_ms` / `core_audio_overruns`: macOS exposes no
    ///   portable, always-available query for per-process Core Audio
    ///   latency/xrun counts outside of an active `AudioUnit` render
    ///   callback, so these honestly report `0.0`/`0` rather than a
    ///   fabricated reading.
    pub fn get_metrics() -> Result<MacOSMetrics, VoirsFFIError> {
        #[cfg(target_os = "macos")]
        {
            let cpu_output = Command::new("ps")
                .args(["-A", "-o", "%cpu="])
                .output()
                .map_err(|e| VoirsFFIError::PlatformError(format!("Failed to run ps: {e}")))?;
            let cpu_str = String::from_utf8_lossy(&cpu_output.stdout);
            let core_count = num_cpus::get().max(1) as f32;
            let cpu_usage = (parsers::parse_ps_cpu_output(&cpu_str) / core_count).clamp(0.0, 100.0);

            let pressure_output = Command::new("sysctl")
                .args(["-n", "kern.memorystatus_vm_pressure_level"])
                .output()
                .map_err(|e| VoirsFFIError::PlatformError(format!("Failed to run sysctl: {e}")))?;
            // Default to "1" (normal) only when the sysctl's own output is
            // unparseable -- this is a parse fallback, not a fabricated
            // pressure reading; the sysctl call itself is still real.
            let pressure_raw: u32 = String::from_utf8_lossy(&pressure_output.stdout)
                .trim()
                .parse()
                .unwrap_or(1);
            let memory_pressure = parsers::normalize_memory_pressure_level(pressure_raw);

            let therm_output = Command::new("pmset")
                .args(["-g", "therm"])
                .output()
                .map_err(|e| VoirsFFIError::PlatformError(format!("Failed to run pmset: {e}")))?;
            let thermal_state =
                parsers::parse_thermal_state(&String::from_utf8_lossy(&therm_output.stdout));

            let batt_output = Command::new("pmset")
                .args(["-g", "batt"])
                .output()
                .map_err(|e| VoirsFFIError::PlatformError(format!("Failed to run pmset: {e}")))?;
            let power_state =
                parsers::parse_power_state(&String::from_utf8_lossy(&batt_output.stdout));

            Ok(MacOSMetrics {
                cpu_usage,
                memory_pressure,
                audio_latency_ms: 0.0,  // no portable query available
                core_audio_overruns: 0, // no portable query available
                thermal_state,
                power_state,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(VoirsFFIError::PlatformError(
                "macOS performance monitoring not available".to_string(),
            ))
        }
    }

    /// Enable low-latency audio mode
    pub fn enable_low_latency_mode() -> Result<(), VoirsFFIError> {
        #[cfg(target_os = "macos")]
        {
            // Implementation would configure Core Audio for low latency
            // Adjust buffer sizes, disable energy efficiency, etc.
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(VoirsFFIError::PlatformError(
                "macOS performance controls not available".to_string(),
            ))
        }
    }
}

/// macOS-specific performance metrics
#[derive(Debug, Clone)]
pub struct MacOSMetrics {
    pub cpu_usage: f32,
    pub memory_pressure: f32,
    pub audio_latency_ms: f32,
    pub core_audio_overruns: u32,
    pub thermal_state: String,
    pub power_state: String,
}

/// C API for macOS integration
#[no_mangle]
pub extern "C" fn voirs_macos_init_core_audio() -> *mut MacOSCoreAudio {
    match MacOSCoreAudio::new() {
        Ok(core_audio) => Box::into_raw(Box::new(core_audio)),
        Err(_) => ptr::null_mut(),
    }
}

/// Destroy a macOS Core Audio instance
///
/// # Safety
/// The `core_audio` pointer must be a valid handle previously returned by `voirs_macos_init_core_audio`.
/// After calling this function, the handle becomes invalid and must not be used.
#[no_mangle]
pub unsafe extern "C" fn voirs_macos_destroy_core_audio(core_audio: *mut MacOSCoreAudio) {
    if !core_audio.is_null() {
        let _ = Box::from_raw(core_audio);
    }
}

/// Get macOS system volume
///
/// # Safety
/// The `core_audio` pointer must be a valid handle previously returned by `voirs_macos_init_core_audio`.
#[no_mangle]
pub unsafe extern "C" fn voirs_macos_get_system_volume(core_audio: *mut MacOSCoreAudio) -> f32 {
    if core_audio.is_null() {
        return -1.0;
    }

    (*core_audio).get_system_volume().unwrap_or(-1.0)
}

#[no_mangle]
pub extern "C" fn voirs_macos_init_avfoundation() -> *mut MacOSAVFoundation {
    match MacOSAVFoundation::new() {
        Ok(av_foundation) => Box::into_raw(Box::new(av_foundation)),
        Err(_) => ptr::null_mut(),
    }
}

/// Destroy a macOS AVFoundation instance
///
/// # Safety
/// The `av_foundation` pointer must be a valid handle previously returned by `voirs_macos_init_avfoundation`.
/// After calling this function, the handle becomes invalid and must not be used.
#[no_mangle]
pub unsafe extern "C" fn voirs_macos_destroy_avfoundation(av_foundation: *mut MacOSAVFoundation) {
    if !av_foundation.is_null() {
        let _ = Box::from_raw(av_foundation);
    }
}

/// Request microphone permission on macOS
///
/// # Safety
/// The `av_foundation` pointer must be a valid handle previously returned by `voirs_macos_init_avfoundation`.
#[no_mangle]
pub unsafe extern "C" fn voirs_macos_request_microphone_permission(
    av_foundation: *mut MacOSAVFoundation,
) -> bool {
    if av_foundation.is_null() {
        return false;
    }

    (*av_foundation)
        .request_microphone_permission()
        .unwrap_or(false)
}

#[no_mangle]
pub extern "C" fn voirs_macos_get_system_language() -> *mut std::os::raw::c_char {
    match MacOSObjectiveC::get_system_language() {
        Ok(language) => match CString::new(language) {
            Ok(c_string) => c_string.into_raw(),
            Err(_) => ptr::null_mut(),
        },
        Err(_) => ptr::null_mut(),
    }
}

/// Show a macOS notification
///
/// # Safety
/// Both `title` and `message` pointers must be valid and point to null-terminated C strings.
#[no_mangle]
pub unsafe extern "C" fn voirs_macos_show_notification(
    title: *const std::os::raw::c_char,
    message: *const std::os::raw::c_char,
) -> bool {
    if title.is_null() || message.is_null() {
        return false;
    }

    let title_str = match CStr::from_ptr(title).to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    let message_str = match CStr::from_ptr(message).to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    MacOSObjectiveC::show_notification(title_str, message_str).is_ok()
}

#[no_mangle]
pub extern "C" fn voirs_macos_enable_low_latency_mode() -> bool {
    MacOSPerformanceMonitor::enable_low_latency_mode().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(feature = "macos-platform", ignore = "Requires audio hardware access")]
    fn test_core_audio_creation() {
        let core_audio = MacOSCoreAudio::new();

        #[cfg(target_os = "macos")]
        {
            assert!(core_audio.is_ok());
            // Device enumeration testing skipped - requires actual audio hardware
            // and may segfault in test environments without audio devices
        }

        #[cfg(not(target_os = "macos"))]
        {
            assert!(core_audio.is_err());
        }
    }

    #[test]
    fn test_avfoundation_creation() {
        let av_foundation = MacOSAVFoundation::new();

        #[cfg(target_os = "macos")]
        {
            assert!(av_foundation.is_ok());
            if let Ok(av) = av_foundation {
                let has_permission = av.has_microphone_permission();
                assert!(has_permission.is_ok());
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            assert!(av_foundation.is_err());
        }
    }

    #[test]
    fn test_objective_c_utilities() {
        let language = MacOSObjectiveC::get_system_language();
        let appearance = MacOSObjectiveC::get_system_appearance();

        #[cfg(target_os = "macos")]
        {
            // `defaults read -g AppleLocale` is a fundamental system default
            // set during initial macOS setup, so it is expected to succeed on
            // any real host (this is a stronger check than the old fabricated
            // "en-US" constant, which would trivially pass any such test).
            assert!(language.is_ok());
            assert!(appearance.is_ok());

            if let Ok(lang) = language {
                assert!(!lang.is_empty());
                // Locale-shaped: ASCII letters and hyphens only (e.g. "ja-JP",
                // "en-US"), and normalized to hyphens (never a stray "_").
                assert!(lang.chars().all(|c| c.is_ascii_alphabetic() || c == '-'));
                assert!(!lang.contains('_'));
            }

            if let Ok(app) = appearance {
                // Contract is exactly {"light", "dark"} -- no more fabricated
                // "auto" fallback.
                assert!(app == "light" || app == "dark");
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            assert!(language.is_err());
            assert!(appearance.is_err());
        }
    }

    #[test]
    fn test_performance_monitoring() {
        let metrics = MacOSPerformanceMonitor::get_metrics();

        #[cfg(target_os = "macos")]
        {
            assert!(metrics.is_ok());
            if let Ok(metrics) = metrics {
                assert!(metrics.cpu_usage >= 0.0);
                assert!(metrics.cpu_usage <= 100.0);
                assert!((0.0..=1.0).contains(&metrics.memory_pressure));
                assert!(metrics.audio_latency_ms >= 0.0);
                assert!(
                    matches!(
                        metrics.thermal_state.as_str(),
                        "nominal" | "throttled" | "critical"
                    ),
                    "unexpected thermal_state: {}",
                    metrics.thermal_state
                );
                assert!(
                    matches!(
                        metrics.power_state.as_str(),
                        "ac_power" | "battery" | "unknown"
                    ),
                    "unexpected power_state: {}",
                    metrics.power_state
                );
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            assert!(metrics.is_err());
        }
    }

    #[test]
    fn test_audio_device_validation() {
        let core_audio = MacOSCoreAudio::new();

        #[cfg(target_os = "macos")]
        if let Ok(mut ca) = core_audio {
            // Test invalid sample rate
            let result = ca.set_device_sample_rate(1, 1000.0); // Too low
            assert!(result.is_err());

            let result = ca.set_device_sample_rate(1, 300000.0); // Too high
            assert!(result.is_err());

            // Test valid sample rate
            let result = ca.set_device_sample_rate(1, 44100.0);
            assert!(result.is_ok());
        }
    }

    /// The `osascript`-based volume query is tolerant of headless/CI
    /// environments that may lack an active Core Audio session: a failure is
    /// an acceptable, honest outcome, but a *successful* result must be a
    /// real 0.0-1.0 fraction (never the old hardcoded `0.8`).
    #[test]
    fn test_get_system_volume_live() {
        let core_audio = MacOSCoreAudio::new();

        #[cfg(target_os = "macos")]
        if let Ok(ca) = core_audio {
            match ca.get_system_volume() {
                Ok(volume) => assert!((0.0..=1.0).contains(&volume)),
                Err(_) => {
                    // Acceptable in a sandboxed/headless test environment.
                }
            }
        }
    }

    /// `get_audio_devices()` must return either a real (non-empty) device
    /// list or an honest error -- never the old hardcoded
    /// "Built-in Output"/"Built-in Microphone" pair produced unconditionally.
    ///
    /// Ignored under `macos-platform` for the same reason as
    /// `test_core_audio_creation`: with that feature on, this method drives
    /// `cpal`'s device enumeration, which can segfault in headless test
    /// environments lacking audio hardware. In the Pure-Rust default build
    /// (the path this batch adds) it drives the safe `system_profiler`
    /// query, which is exactly what we want to exercise here.
    #[test]
    #[cfg_attr(feature = "macos-platform", ignore = "Requires audio hardware access")]
    fn test_get_audio_devices_live() {
        let core_audio = MacOSCoreAudio::new();

        #[cfg(target_os = "macos")]
        if let Ok(ca) = core_audio {
            match ca.get_audio_devices() {
                Ok(devices) => assert!(
                    !devices.is_empty(),
                    "Ok(..) result must contain at least one real device"
                ),
                Err(_) => {
                    // Acceptable if system_profiler is unavailable in this
                    // environment.
                }
            }
        }
    }
}
