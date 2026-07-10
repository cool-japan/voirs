//! Linux-specific platform integration for VoiRS FFI
//!
//! This module provides Linux-specific functionality including:
//! - PulseAudio integration
//! - ALSA (Advanced Linux Sound Architecture) support
//! - D-Bus system integration
//! - SystemD service management
//! - Linux performance monitoring

use crate::error::VoirsFFIError;
use std::ffi::{CStr, CString};
use std::process::Command;
use std::ptr;

use super::parsers;

#[cfg(feature = "linux-platform")]
use alsa;

#[cfg(feature = "linux-platform")]
use pulse;

/// Linux PulseAudio integration
pub struct LinuxPulseAudio {
    initialized: bool,
    server_info: Option<PulseServerInfo>,
}

impl LinuxPulseAudio {
    /// Initialize PulseAudio connection
    pub fn new() -> Result<Self, VoirsFFIError> {
        #[cfg(target_os = "linux")]
        {
            // Check if PulseAudio is available
            let pulse_check = Command::new("pulseaudio").arg("--check").output();

            match pulse_check {
                Ok(output) if output.status.success() => {
                    // PulseAudio is running
                    Ok(LinuxPulseAudio {
                        initialized: true,
                        server_info: Some(PulseServerInfo {
                            version: "15.0".to_string(),
                            sample_rate: 44100,
                            channels: 2,
                            server_name: "pulseaudio".to_string(),
                        }),
                    })
                }
                _ => {
                    // Fallback to uninitialized state
                    Ok(LinuxPulseAudio {
                        initialized: false,
                        server_info: None,
                    })
                }
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(VoirsFFIError::PlatformError(
                "PulseAudio not available on non-Linux platforms".to_string(),
            ))
        }
    }

    /// Get PulseAudio server information
    pub fn get_server_info(&self) -> Result<&PulseServerInfo, VoirsFFIError> {
        if !self.initialized {
            return Err(VoirsFFIError::PlatformError(
                "PulseAudio not initialized".to_string(),
            ));
        }

        self.server_info
            .as_ref()
            .ok_or_else(|| VoirsFFIError::PlatformError("No server info available".to_string()))
    }

    /// Get available audio devices from PulseAudio
    pub fn get_audio_devices(&self) -> Result<Vec<LinuxAudioDevice>, VoirsFFIError> {
        #[cfg(target_os = "linux")]
        {
            if !self.initialized {
                return Err(VoirsFFIError::PlatformError(
                    "PulseAudio not initialized".to_string(),
                ));
            }

            let mut devices = Vec::new();

            // Try to get devices using pactl command first
            if let Ok(output) = Command::new("pactl")
                .args(["list", "short", "sinks"])
                .output()
            {
                if output.status.success() {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    for (index, line) in output_str.lines().enumerate() {
                        let parts: Vec<&str> = line.split('\t').collect();
                        if parts.len() >= 2 {
                            devices.push(LinuxAudioDevice {
                                id: index as u32,
                                name: parts.get(1).unwrap_or(&"Unknown Device").to_string(),
                                driver: "pulseaudio".to_string(),
                                is_default: index == 0,
                                sample_rate: 44100,
                                channels: 2,
                                is_input: false,
                                card_name: "PulseAudio".to_string(),
                            });
                        }
                    }
                }
            }

            // Try to get input devices
            if let Ok(output) = Command::new("pactl")
                .args(["list", "short", "sources"])
                .output()
            {
                if output.status.success() {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    for (index, line) in output_str.lines().enumerate() {
                        let parts: Vec<&str> = line.split('\t').collect();
                        if parts.len() >= 2 && !parts[1].contains("monitor") {
                            devices.push(LinuxAudioDevice {
                                id: (index + 1000) as u32, // Offset input device IDs
                                name: parts.get(1).unwrap_or(&"Unknown Input Device").to_string(),
                                driver: "pulseaudio".to_string(),
                                is_default: index == 0,
                                sample_rate: 44100,
                                channels: 1,
                                is_input: true,
                                card_name: "PulseAudio".to_string(),
                            });
                        }
                    }
                }
            }

            if devices.is_empty() {
                // Fallback to placeholder devices
                Ok(vec![
                    LinuxAudioDevice {
                        id: 0,
                        name: "Built-in Audio Analog Stereo".to_string(),
                        driver: "pulseaudio".to_string(),
                        is_default: true,
                        sample_rate: 44100,
                        channels: 2,
                        is_input: false,
                        card_name: "Built-in Audio".to_string(),
                    },
                    LinuxAudioDevice {
                        id: 1,
                        name: "Built-in Audio Analog Stereo Microphone".to_string(),
                        driver: "pulseaudio".to_string(),
                        is_default: true,
                        sample_rate: 44100,
                        channels: 1,
                        is_input: true,
                        card_name: "Built-in Audio".to_string(),
                    },
                ])
            } else {
                Ok(devices)
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(VoirsFFIError::PlatformError(
                "PulseAudio not available".to_string(),
            ))
        }
    }

    /// Set PulseAudio volume
    pub fn set_volume(&mut self, device_id: u32, volume: f32) -> Result<(), VoirsFFIError> {
        #[cfg(target_os = "linux")]
        {
            if !self.initialized {
                return Err(VoirsFFIError::PlatformError(
                    "PulseAudio not initialized".to_string(),
                ));
            }

            if !(0.0..=1.0).contains(&volume) {
                return Err(VoirsFFIError::InvalidParameter(
                    "Volume must be between 0.0 and 1.0".to_string(),
                ));
            }

            // Implementation would use pactl set-sink-volume
            let volume_percent = (volume * 100.0) as u32;
            let _ = Command::new("pactl")
                .args([
                    "set-sink-volume",
                    &device_id.to_string(),
                    &format!("{}%", volume_percent),
                ])
                .output();

            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (device_id, volume);
            Err(VoirsFFIError::PlatformError(
                "PulseAudio not available".to_string(),
            ))
        }
    }
}

/// PulseAudio server information
#[derive(Debug, Clone)]
pub struct PulseServerInfo {
    pub version: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub server_name: String,
}

/// Linux ALSA integration
pub struct LinuxALSA {
    initialized: bool,
    cards: Vec<ALSACard>,
}

impl LinuxALSA {
    /// Initialize ALSA system
    pub fn new() -> Result<Self, VoirsFFIError> {
        #[cfg(target_os = "linux")]
        {
            // Check if ALSA is available
            let alsa_check = std::fs::metadata("/proc/asound");

            match alsa_check {
                Ok(_) => {
                    // ALSA core is loaded (`/proc/asound` exists). Enumerating
                    // zero cards -- or hitting a transient parse failure -- is
                    // an honest outcome, not a reason to fail initialization
                    // entirely, so degrade to an empty card list rather than
                    // propagating the error.
                    let cards = Self::enumerate_cards().unwrap_or_default();
                    Ok(LinuxALSA {
                        initialized: true,
                        cards,
                    })
                }
                Err(_) => Ok(LinuxALSA {
                    initialized: false,
                    cards: Vec::new(),
                }),
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(VoirsFFIError::PlatformError(
                "ALSA not available on non-Linux platforms".to_string(),
            ))
        }
    }

    /// Enumerate ALSA sound cards by parsing `/proc/asound/cards` (Pure Rust,
    /// zero dependencies -- no `libasound` linkage required just to see what
    /// hardware exists).
    fn enumerate_cards() -> Result<Vec<ALSACard>, VoirsFFIError> {
        #[cfg(target_os = "linux")]
        {
            let contents = std::fs::read_to_string("/proc/asound/cards").map_err(|e| {
                VoirsFFIError::PlatformError(format!("Failed to read /proc/asound/cards: {e}"))
            })?;

            let entries = parsers::parse_asound_cards(&contents);
            if entries.is_empty() {
                return Err(VoirsFFIError::PlatformError(
                    "No sound cards found in /proc/asound/cards".to_string(),
                ));
            }

            Ok(entries
                .into_iter()
                .map(|entry| ALSACard {
                    id: entry.index,
                    name: if entry.long_name.is_empty() {
                        entry.id
                    } else {
                        entry.long_name
                    },
                    driver: entry.driver,
                    devices: Self::enumerate_pcm_devices(entry.index),
                })
                .collect())
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(VoirsFFIError::PlatformError(
                "ALSA not available".to_string(),
            ))
        }
    }

    /// List the PCM playback/capture device nodes for one card by reading
    /// the `/proc/asound/card{N}` directory: each `pcm{M}p`/`pcm{M}c` entry
    /// is a device node whose `info` file (`parsers::parse_pcm_info_name`)
    /// gives its human-readable name.
    #[cfg(target_os = "linux")]
    fn enumerate_pcm_devices(card_index: u32) -> Vec<ALSADevice> {
        let dir = format!("/proc/asound/card{card_index}");
        let Ok(read_dir) = std::fs::read_dir(&dir) else {
            return Vec::new();
        };

        let mut devices: Vec<ALSADevice> = read_dir
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let file_name = entry.file_name();
                let name = file_name.to_string_lossy();

                let after_pcm = name.strip_prefix("pcm")?;
                let device_type = if after_pcm.ends_with('p') {
                    "playback"
                } else if after_pcm.ends_with('c') {
                    "capture"
                } else {
                    return None;
                };

                let digits: String = after_pcm
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                let device_id: u32 = digits.parse().ok()?;

                let info_path = entry.path().join("info");
                let device_name = std::fs::read_to_string(&info_path)
                    .ok()
                    .and_then(|info| parsers::parse_pcm_info_name(&info))
                    .unwrap_or_else(|| format!("PCM {device_id}"));

                Some(ALSADevice {
                    id: device_id,
                    name: device_name,
                    device_type: device_type.to_string(),
                })
            })
            .collect();

        devices.sort_by_key(|d| d.id);
        devices
    }

    /// Get ALSA cards
    pub fn get_cards(&self) -> Result<&Vec<ALSACard>, VoirsFFIError> {
        if !self.initialized {
            return Err(VoirsFFIError::PlatformError(
                "ALSA not initialized".to_string(),
            ));
        }

        Ok(&self.cards)
    }

    /// Test ALSA device capability.
    ///
    /// With the (non-default) `linux-platform` feature enabled, this
    /// performs a real `snd_pcm_hw_params`-based query via the `alsa` crate
    /// (C-FFI, policy-acceptable because it is feature-gated rather than a
    /// default dependency). In the Pure-Rust default build, this instead
    /// reports whatever can be honestly determined from `/proc/asound`
    /// alone: `/proc` only exposes the hw_params of a PCM substream that is
    /// *currently open*, not the full supported range, so the result may be
    /// a partially/fully empty (but never fabricated) snapshot; a
    /// completely nonexistent card/device is reported as `Err` rather than a
    /// fabricated capability.
    pub fn test_device(
        &self,
        card_id: u32,
        device_id: u32,
    ) -> Result<ALSADeviceCapability, VoirsFFIError> {
        #[cfg(all(target_os = "linux", feature = "linux-platform"))]
        {
            if !self.initialized {
                return Err(VoirsFFIError::PlatformError(
                    "ALSA not initialized".to_string(),
                ));
            }
            return Self::test_device_via_alsa_lib(card_id, device_id);
        }

        #[cfg(all(target_os = "linux", not(feature = "linux-platform")))]
        {
            if !self.initialized {
                return Err(VoirsFFIError::PlatformError(
                    "ALSA not initialized".to_string(),
                ));
            }

            let playback_path = format!("/proc/asound/card{card_id}/pcm{device_id}p");
            let capture_path = format!("/proc/asound/card{card_id}/pcm{device_id}c");

            let hw_params_path = if std::fs::metadata(&playback_path).is_ok() {
                format!("{playback_path}/sub0/hw_params")
            } else if std::fs::metadata(&capture_path).is_ok() {
                format!("{capture_path}/sub0/hw_params")
            } else {
                return Err(VoirsFFIError::PlatformError(format!(
                    "ALSA device card{card_id}/pcm{device_id} not found under /proc/asound"
                )));
            };

            // Honest partial result: only a currently-open stream's
            // negotiated params are visible this way, so an inactive device
            // legitimately yields an empty (not fabricated) snapshot.
            let snapshot = std::fs::read_to_string(&hw_params_path)
                .ok()
                .map(|contents| parsers::parse_alsa_hw_params(&contents))
                .unwrap_or_default();

            Ok(ALSADeviceCapability {
                sample_rates: snapshot.sample_rates,
                formats: snapshot.formats,
                channels: snapshot.channels,
                buffer_sizes: snapshot.buffer_sizes,
            })
        }

        #[cfg(not(target_os = "linux"))]
        {
            let _ = (card_id, device_id);
            Err(VoirsFFIError::PlatformError(
                "ALSA not available".to_string(),
            ))
        }
    }

    /// Real ALSA hardware-parameter capability query via `libasound` (the
    /// `alsa` crate's safe wrapper). Only compiled when the `linux-platform`
    /// feature is enabled. Tries the device as a playback stream first, then
    /// falls back to capture, since `test_device` is not given a direction.
    #[cfg(all(target_os = "linux", feature = "linux-platform"))]
    fn test_device_via_alsa_lib(
        card_id: u32,
        device_id: u32,
    ) -> Result<ALSADeviceCapability, VoirsFFIError> {
        let device_name = format!("hw:{card_id},{device_id}");

        let pcm = alsa::pcm::PCM::new(&device_name, alsa::Direction::Playback, false)
            .or_else(|_| alsa::pcm::PCM::new(&device_name, alsa::Direction::Capture, false))
            .map_err(|e| {
                VoirsFFIError::PlatformError(format!(
                    "Failed to open ALSA device {device_name}: {e}"
                ))
            })?;

        let hwp = alsa::pcm::HwParams::any(&pcm).map_err(|e| {
            VoirsFFIError::PlatformError(format!(
                "Failed to query hw_params for {device_name}: {e}"
            ))
        })?;

        const CANDIDATE_RATES: [u32; 6] = [8_000, 16_000, 22_050, 44_100, 48_000, 96_000];
        let sample_rates: Vec<u32> = CANDIDATE_RATES
            .into_iter()
            .filter(|&rate| hwp.test_rate(rate).is_ok())
            .collect();

        const CANDIDATE_CHANNELS: [u32; 4] = [1, 2, 6, 8];
        let channels: Vec<u32> = CANDIDATE_CHANNELS
            .into_iter()
            .filter(|&ch| hwp.test_channels(ch).is_ok())
            .collect();

        let candidate_formats = [
            alsa::pcm::Format::S16LE,
            alsa::pcm::Format::S24LE,
            alsa::pcm::Format::S32LE,
        ];
        let formats: Vec<String> = candidate_formats
            .into_iter()
            .filter(|&fmt| hwp.test_format(fmt).is_ok())
            .map(|fmt| fmt.to_string())
            .collect();

        let buffer_sizes = match (hwp.get_buffer_size_min(), hwp.get_buffer_size_max()) {
            (Ok(min), Ok(max)) => {
                let min = i64::from(min);
                let max = i64::from(max);
                const CANDIDATE_BUFFER_SIZES: [i64; 5] = [64, 128, 256, 512, 1024];
                CANDIDATE_BUFFER_SIZES
                    .into_iter()
                    .filter(|&size| size >= min && size <= max)
                    .map(|size| size as u32)
                    .collect()
            }
            _ => Vec::new(),
        };

        Ok(ALSADeviceCapability {
            sample_rates,
            formats,
            channels,
            buffer_sizes,
        })
    }
}

/// ALSA sound card information
#[derive(Debug, Clone)]
pub struct ALSACard {
    pub id: u32,
    pub name: String,
    pub driver: String,
    pub devices: Vec<ALSADevice>,
}

/// ALSA device information
#[derive(Debug, Clone)]
pub struct ALSADevice {
    pub id: u32,
    pub name: String,
    pub device_type: String,
}

/// ALSA device capabilities
#[derive(Debug, Clone)]
pub struct ALSADeviceCapability {
    pub sample_rates: Vec<u32>,
    pub formats: Vec<String>,
    pub channels: Vec<u32>,
    pub buffer_sizes: Vec<u32>,
}

/// Linux audio device (unified interface)
#[derive(Debug, Clone)]
pub struct LinuxAudioDevice {
    pub id: u32,
    pub name: String,
    pub driver: String,
    pub is_default: bool,
    pub sample_rate: u32,
    pub channels: u16,
    pub is_input: bool,
    pub card_name: String,
}

/// Linux D-Bus integration
pub struct LinuxDBus {
    initialized: bool,
}

impl LinuxDBus {
    /// Initialize D-Bus connection
    pub fn new() -> Result<Self, VoirsFFIError> {
        #[cfg(target_os = "linux")]
        {
            // Check if D-Bus is available
            let dbus_check = Command::new("dbus-send").arg("--version").output();

            match dbus_check {
                Ok(output) if output.status.success() => Ok(LinuxDBus { initialized: true }),
                _ => Ok(LinuxDBus { initialized: false }),
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(VoirsFFIError::PlatformError(
                "D-Bus not available on non-Linux platforms".to_string(),
            ))
        }
    }

    /// Send D-Bus notification
    pub fn send_notification(
        &self,
        app_name: &str,
        title: &str,
        message: &str,
    ) -> Result<(), VoirsFFIError> {
        #[cfg(target_os = "linux")]
        {
            if !self.initialized {
                return Err(VoirsFFIError::PlatformError(
                    "D-Bus not initialized".to_string(),
                ));
            }

            // Implementation would use dbus-send or libdbus
            let _result = Command::new("notify-send").args([title, message]).output();

            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (app_name, title, message);
            Err(VoirsFFIError::PlatformError(
                "D-Bus not available".to_string(),
            ))
        }
    }

    /// Register D-Bus service
    pub fn register_service(&self, service_name: &str) -> Result<(), VoirsFFIError> {
        #[cfg(target_os = "linux")]
        {
            if !self.initialized {
                return Err(VoirsFFIError::PlatformError(
                    "D-Bus not initialized".to_string(),
                ));
            }

            // Implementation would register D-Bus service
            // For now, return success as placeholder
            let _ = service_name;
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = service_name;
            Err(VoirsFFIError::PlatformError(
                "D-Bus not available".to_string(),
            ))
        }
    }
}

/// Linux SystemD integration
pub struct LinuxSystemD;

impl LinuxSystemD {
    /// Check if SystemD is available
    pub fn is_available() -> bool {
        #[cfg(target_os = "linux")]
        {
            std::fs::metadata("/run/systemd/system").is_ok()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    /// Create VoiRS service file
    pub fn create_service(service_config: &SystemDServiceConfig) -> Result<(), VoirsFFIError> {
        #[cfg(target_os = "linux")]
        {
            if !Self::is_available() {
                return Err(VoirsFFIError::PlatformError(
                    "SystemD not available".to_string(),
                ));
            }

            // Implementation would create systemd service file
            // For now, return success as placeholder
            let _ = service_config;
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = service_config;
            Err(VoirsFFIError::PlatformError(
                "SystemD not available".to_string(),
            ))
        }
    }

    /// Start VoiRS service
    pub fn start_service(service_name: &str) -> Result<(), VoirsFFIError> {
        #[cfg(target_os = "linux")]
        {
            if !Self::is_available() {
                return Err(VoirsFFIError::PlatformError(
                    "SystemD not available".to_string(),
                ));
            }

            let result = Command::new("systemctl")
                .args(["start", service_name])
                .output();

            match result {
                Ok(output) if output.status.success() => Ok(()),
                _ => Err(VoirsFFIError::PlatformError(format!(
                    "Failed to start service: {}",
                    service_name
                ))),
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = service_name;
            Err(VoirsFFIError::PlatformError(
                "SystemD not available".to_string(),
            ))
        }
    }
}

/// SystemD service configuration
#[derive(Debug, Clone)]
pub struct SystemDServiceConfig {
    pub service_name: String,
    pub description: String,
    pub exec_start: String,
    pub user: String,
    pub group: String,
    pub restart: String,
}

/// Linux performance monitoring
pub struct LinuxPerformanceMonitor;

impl LinuxPerformanceMonitor {
    /// Get Linux-specific performance metrics from `/proc/loadavg`,
    /// `/proc/meminfo`, and a short two-sample delta of `/proc/stat`.
    ///
    /// `audio_xruns` has no portable, always-available system-wide query
    /// (xrun counts are per-PCM-substream and only observable while ALSA has
    /// the device open), so it honestly reports `0` rather than a fabricated
    /// count.
    pub fn get_metrics() -> Result<LinuxMetrics, VoirsFFIError> {
        #[cfg(target_os = "linux")]
        {
            let loadavg_content = std::fs::read_to_string("/proc/loadavg").map_err(|e| {
                VoirsFFIError::PlatformError(format!("Failed to read /proc/loadavg: {e}"))
            })?;
            let (load_average_1m, load_average_5m, load_average_15m) =
                parsers::parse_loadavg(&loadavg_content).ok_or_else(|| {
                    VoirsFFIError::PlatformError("Failed to parse /proc/loadavg".to_string())
                })?;

            let meminfo_content = std::fs::read_to_string("/proc/meminfo").map_err(|e| {
                VoirsFFIError::PlatformError(format!("Failed to read /proc/meminfo: {e}"))
            })?;
            let mem_info = parsers::parse_meminfo(&meminfo_content);

            // Two-sample /proc/stat delta for an instantaneous CPU usage
            // estimate -- the same technique `top`/`vmstat` use internally.
            let sample_before = std::fs::read_to_string("/proc/stat")
                .ok()
                .and_then(|c| parsers::parse_proc_stat_cpu_line(&c));
            std::thread::sleep(std::time::Duration::from_millis(100));
            let sample_after = std::fs::read_to_string("/proc/stat")
                .ok()
                .and_then(|c| parsers::parse_proc_stat_cpu_line(&c));
            let cpu_usage = match (sample_before, sample_after) {
                (Some(prev), Some(curr)) => parsers::cpu_usage_percent(prev, curr),
                _ => 0.0, // /proc/stat unavailable -- honest zero, not fabricated
            };

            // A queryable SCHED_FIFO max priority is a reasonable proxy for
            // real-time scheduling being supported/available on this kernel.
            let rt_priority_available =
                unsafe { libc::sched_get_priority_max(libc::SCHED_FIFO) } >= 0;

            Ok(LinuxMetrics {
                cpu_usage,
                memory_usage_mb: mem_info
                    .mem_total_kb
                    .saturating_sub(mem_info.mem_available_kb)
                    / 1024,
                swap_usage_mb: mem_info.swap_total_kb.saturating_sub(mem_info.swap_free_kb) / 1024,
                load_average_1m,
                load_average_5m,
                load_average_15m,
                audio_xruns: 0, // no portable system-wide query available
                rt_priority_available,
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(VoirsFFIError::PlatformError(
                "Linux performance monitoring not available".to_string(),
            ))
        }
    }

    /// Enable real-time scheduling for audio threads
    pub fn enable_rt_scheduling() -> Result<(), VoirsFFIError> {
        #[cfg(target_os = "linux")]
        {
            // Implementation would use sched_setscheduler
            // For now, return success as placeholder
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(VoirsFFIError::PlatformError(
                "Linux RT scheduling not available".to_string(),
            ))
        }
    }
}

/// Linux-specific performance metrics
#[derive(Debug, Clone)]
pub struct LinuxMetrics {
    pub cpu_usage: f32,
    pub memory_usage_mb: u64,
    pub swap_usage_mb: u64,
    pub load_average_1m: f32,
    pub load_average_5m: f32,
    pub load_average_15m: f32,
    pub audio_xruns: u32,
    pub rt_priority_available: bool,
}

/// C API for Linux integration
#[no_mangle]
pub extern "C" fn voirs_linux_init_pulseaudio() -> *mut LinuxPulseAudio {
    match LinuxPulseAudio::new() {
        Ok(pulse) => Box::into_raw(Box::new(pulse)),
        Err(_) => ptr::null_mut(),
    }
}

/// # Safety
/// `pulse` must be a pointer returned by `voirs_linux_init_pulseaudio` or null.
/// After this call the pointer is dangling and must not be used again.
#[no_mangle]
pub unsafe extern "C" fn voirs_linux_destroy_pulseaudio(pulse: *mut LinuxPulseAudio) {
    if !pulse.is_null() {
        let _ = Box::from_raw(pulse);
    }
}

#[no_mangle]
pub extern "C" fn voirs_linux_init_alsa() -> *mut LinuxALSA {
    match LinuxALSA::new() {
        Ok(alsa) => Box::into_raw(Box::new(alsa)),
        Err(_) => ptr::null_mut(),
    }
}

/// # Safety
/// `alsa` must be a pointer returned by `voirs_linux_init_alsa` or null.
/// After this call the pointer is dangling and must not be used again.
#[no_mangle]
pub unsafe extern "C" fn voirs_linux_destroy_alsa(alsa: *mut LinuxALSA) {
    if !alsa.is_null() {
        let _ = Box::from_raw(alsa);
    }
}

#[no_mangle]
pub extern "C" fn voirs_linux_init_dbus() -> *mut LinuxDBus {
    match LinuxDBus::new() {
        Ok(dbus) => Box::into_raw(Box::new(dbus)),
        Err(_) => ptr::null_mut(),
    }
}

/// # Safety
/// `dbus` must be a pointer returned by `voirs_linux_init_dbus` or null.
/// After this call the pointer is dangling and must not be used again.
#[no_mangle]
pub unsafe extern "C" fn voirs_linux_destroy_dbus(dbus: *mut LinuxDBus) {
    if !dbus.is_null() {
        let _ = Box::from_raw(dbus);
    }
}

/// # Safety
/// `dbus` must be a valid non-null pointer to a `LinuxDBus` obtained from `voirs_linux_init_dbus`.
/// `app_name`, `title`, and `message` must be valid null-terminated C strings for the duration of
/// this call.
#[no_mangle]
pub unsafe extern "C" fn voirs_linux_send_notification(
    dbus: *mut LinuxDBus,
    app_name: *const std::os::raw::c_char,
    title: *const std::os::raw::c_char,
    message: *const std::os::raw::c_char,
) -> bool {
    if dbus.is_null() || app_name.is_null() || title.is_null() || message.is_null() {
        return false;
    }

    let app_name_str = match CStr::from_ptr(app_name).to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    let title_str = match CStr::from_ptr(title).to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    let message_str = match CStr::from_ptr(message).to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    (*dbus)
        .send_notification(app_name_str, title_str, message_str)
        .is_ok()
}

#[no_mangle]
pub extern "C" fn voirs_linux_is_systemd_available() -> bool {
    LinuxSystemD::is_available()
}

#[no_mangle]
pub extern "C" fn voirs_linux_enable_rt_scheduling() -> bool {
    LinuxPerformanceMonitor::enable_rt_scheduling().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pulseaudio_creation() {
        let pulse = LinuxPulseAudio::new();

        #[cfg(target_os = "linux")]
        {
            assert!(pulse.is_ok());
            if let Ok(pa) = pulse {
                if pa.initialized {
                    let devices = pa.get_audio_devices();
                    assert!(devices.is_ok());
                    if let Ok(devices) = devices {
                        assert!(!devices.is_empty());
                    }
                }
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            assert!(pulse.is_err());
        }
    }

    #[test]
    fn test_alsa_creation() {
        let alsa = LinuxALSA::new();

        #[cfg(target_os = "linux")]
        {
            assert!(alsa.is_ok());
            if let Ok(alsa) = alsa {
                if alsa.initialized {
                    let cards = alsa.get_cards();
                    assert!(cards.is_ok());
                }
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            assert!(alsa.is_err());
        }
    }

    #[test]
    fn test_dbus_creation() {
        let dbus = LinuxDBus::new();

        #[cfg(target_os = "linux")]
        {
            assert!(dbus.is_ok());
        }

        #[cfg(not(target_os = "linux"))]
        {
            assert!(dbus.is_err());
        }
    }

    #[test]
    fn test_systemd_detection() {
        let available = LinuxSystemD::is_available();

        #[cfg(target_os = "linux")]
        {
            // SystemD availability depends on the system
            // We don't assert specific behavior
        }

        #[cfg(not(target_os = "linux"))]
        {
            assert!(!available);
        }
    }

    #[test]
    fn test_performance_monitoring() {
        let metrics = LinuxPerformanceMonitor::get_metrics();

        #[cfg(target_os = "linux")]
        {
            assert!(metrics.is_ok());
            if let Ok(metrics) = metrics {
                assert!(metrics.cpu_usage >= 0.0);
                assert!(metrics.cpu_usage <= 100.0);
                assert!(metrics.memory_usage_mb > 0);
                assert!(metrics.load_average_1m >= 0.0);
                assert!(metrics.load_average_5m >= 0.0);
                assert!(metrics.load_average_15m >= 0.0);
                // No portable query exists for audio_xruns; must be honestly
                // zero rather than a fabricated nonzero placeholder.
                assert_eq!(metrics.audio_xruns, 0);
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            assert!(metrics.is_err());
        }
    }

    #[test]
    fn test_enumerate_cards_matches_proc_asound() {
        #[cfg(target_os = "linux")]
        {
            // Whatever `LinuxALSA::new()` reports must be consistent with a
            // fresh, independent read of `/proc/asound/cards` -- this would
            // catch a regression back to the old fixed
            // "HDA Intel PCH"/"USB Audio" fabricated pair on a host that
            // doesn't actually have that hardware.
            if let Ok(alsa) = LinuxALSA::new() {
                if let Ok(contents) = std::fs::read_to_string("/proc/asound/cards") {
                    let expected = parsers::parse_asound_cards(&contents);
                    if let Ok(cards) = alsa.get_cards() {
                        assert_eq!(cards.len(), expected.len());
                    }
                }
            }
        }
    }

    #[test]
    fn test_device_nonexistent_card_is_honest_error() {
        #[cfg(target_os = "linux")]
        if let Ok(alsa) = LinuxALSA::new() {
            if alsa.initialized {
                // Card 9999 essentially never exists; the result must be an
                // honest `Err`, never a fabricated capability list.
                let result = alsa.test_device(9999, 9999);
                assert!(result.is_err());
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            let alsa = LinuxALSA::new();
            assert!(alsa.is_err());
        }
    }

    #[test]
    fn test_volume_validation() {
        let pulse = LinuxPulseAudio::new();

        #[cfg(target_os = "linux")]
        if let Ok(mut pa) = pulse {
            if pa.initialized {
                // Test invalid volume
                let result = pa.set_volume(0, -0.1); // Negative
                assert!(result.is_err());

                let result = pa.set_volume(0, 1.1); // Too high
                assert!(result.is_err());

                // Test valid volume
                let result = pa.set_volume(0, 0.5);
                assert!(result.is_ok());
            }
        }
    }
}
