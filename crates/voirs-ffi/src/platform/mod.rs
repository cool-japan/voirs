//! Platform-specific functionality for VoiRS FFI
//!
//! This module provides platform-specific optimizations and integrations
//! for Windows, macOS, Linux, and other platforms.

use std::sync::OnceLock;

#[cfg(target_os = "linux")]
pub mod linux;
#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(target_os = "windows")]
pub mod windows;

// Pure, platform-independent parsing helpers shared by the OS-specific
// modules above. Unlike those modules, this one is *not* gated by
// `target_os`, so its parsers can be unit-tested against fixed sample
// strings on any host (see `parsers::tests` for Linux/macOS/Windows
// fixtures that all run regardless of which OS is executing the suite).
pub mod parsers;

// Mobile platform support
#[cfg(target_os = "android")]
pub mod android;
#[cfg(target_os = "ios")]
pub mod ios;

// IDE and development environment integrations
pub mod packages;
pub mod vs; // Visual Studio integration
pub mod xcode; // Xcode integration // Package management (deb, rpm, flatpak, snap)

// Re-export platform-specific functionality
#[cfg(target_os = "android")]
pub use android::*;
#[cfg(target_os = "ios")]
pub use ios::*;
#[cfg(target_os = "linux")]
pub use linux::*;
#[cfg(target_os = "macos")]
pub use macos::*;
#[cfg(target_os = "windows")]
pub use windows::*;

/// Platform information and capabilities
#[derive(Debug, Clone)]
pub struct PlatformInfo {
    pub os: String,
    pub arch: String,
    pub cpu_cores: usize,
    pub total_memory: u64,
    pub has_avx2: bool,
    pub has_sse2: bool,
    pub has_neon: bool,
    pub audio_backend: String,
}

impl PlatformInfo {
    /// Get current platform information
    pub fn current() -> &'static Self {
        static PLATFORM_INFO: OnceLock<PlatformInfo> = OnceLock::new();

        PLATFORM_INFO.get_or_init(|| Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu_cores: num_cpus::get(),
            total_memory: Self::get_total_memory(),
            has_avx2: Self::has_cpu_feature("avx2"),
            has_sse2: Self::has_cpu_feature("sse2"),
            has_neon: Self::has_cpu_feature("neon"),
            audio_backend: Self::detect_audio_backend(),
        })
    }

    /// Detect the best audio backend for this platform
    fn detect_audio_backend() -> String {
        #[cfg(target_os = "windows")]
        return "wasapi".to_string();

        #[cfg(target_os = "macos")]
        return "coreaudio".to_string();

        #[cfg(target_os = "linux")]
        {
            // Check for available audio systems in order of preference
            if std::process::Command::new("pulseaudio")
                .arg("--check")
                .output()
                .is_ok()
            {
                return "pulseaudio".to_string();
            }
            if std::fs::metadata("/dev/snd").is_ok() {
                return "alsa".to_string();
            }
            "dummy".to_string()
        }

        #[cfg(target_os = "ios")]
        return "coreaudio".to_string();

        #[cfg(target_os = "android")]
        return "oboe".to_string(); // Android high-performance audio

        #[cfg(not(any(
            target_os = "windows",
            target_os = "macos",
            target_os = "linux",
            target_os = "ios",
            target_os = "android"
        )))]
        return "unknown".to_string();
    }

    /// Get total system memory in bytes
    fn get_total_memory() -> u64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                for line in meminfo.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return kb * 1024; // Convert KB to bytes
                            }
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            if let Ok(output) = Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
                if let Ok(mem_str) = String::from_utf8(output.stdout) {
                    if let Ok(mem_bytes) = mem_str.trim().parse::<u64>() {
                        return mem_bytes;
                    }
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            // Query `GlobalMemoryStatusEx` (kernel32) directly via a hand-rolled
            // `extern "system"` declaration. This is a zero-dependency approach
            // (mirrors `voirs-cli/src/performance/profiler.rs`) that keeps the
            // *default* build Pure-Rust regardless of which optional
            // FFI-related features happen to be enabled: `GlobalMemoryStatusEx`
            // has been part of the stable Win32 ABI since Windows 2000 and is
            // always linkable via kernel32 (the standard library already links
            // against it), so no `windows`/`winapi` crate dependency is needed.
            #[repr(C)]
            struct MemoryStatusEx {
                dw_length: u32,
                dw_memory_load: u32,
                ull_total_phys: u64,
                ull_avail_phys: u64,
                ull_total_page_file: u64,
                ull_avail_page_file: u64,
                ull_total_virtual: u64,
                ull_avail_virtual: u64,
                ull_avail_extended_virtual: u64,
            }

            extern "system" {
                fn GlobalMemoryStatusEx(lpbuffer: *mut MemoryStatusEx) -> i32;
            }

            let mut memory_status = MemoryStatusEx {
                dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
                dw_memory_load: 0,
                ull_total_phys: 0,
                ull_avail_phys: 0,
                ull_total_page_file: 0,
                ull_avail_page_file: 0,
                ull_total_virtual: 0,
                ull_avail_virtual: 0,
                ull_avail_extended_virtual: 0,
            };

            let succeeded = unsafe { GlobalMemoryStatusEx(&mut memory_status) } != 0;
            if succeeded && memory_status.ull_total_phys > 0 {
                return memory_status.ull_total_phys;
            }
            // Extremely rare: the syscall itself failed. Fall through to the
            // documented fallback below rather than fabricating a Windows-only
            // constant here.
        }

        // Default fallback: only reached if every platform-specific query
        // above failed to produce a real value (or on platforms with no
        // implemented query at all).
        4 * 1024 * 1024 * 1024 // 4 GB default
    }

    /// Check if a specific CPU feature is available
    fn has_cpu_feature(feature: &str) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            match feature {
                "avx2" => is_x86_feature_detected!("avx2"),
                "sse2" => is_x86_feature_detected!("sse2"),
                _ => false,
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            matches!(feature, "neon") && cfg!(target_feature = "neon")
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let _ = feature;
            false
        }
    }

    /// Get optimal number of threads for audio processing
    pub fn optimal_threads(&self) -> usize {
        // Use 75% of available cores, minimum 1, maximum 8
        (self.cpu_cores * 3 / 4).clamp(1, 8)
    }

    /// Get optimal buffer size for audio processing
    pub fn optimal_buffer_size(&self) -> usize {
        match self.os.as_str() {
            "windows" => 512, // WASAPI typically prefers 512 samples
            "macos" => 256,   // Core Audio prefers smaller buffers
            "linux" => 1024,  // ALSA/PulseAudio can handle larger buffers
            _ => 512,         // Conservative default
        }
    }

    /// Check whether the current CPU exposes SIMD-based acceleration.
    ///
    /// **Contract**: this reports CPU SIMD feature availability (AVX2/SSE2 on
    /// x86_64, NEON on aarch64) via `has_avx2 || has_sse2 || has_neon` --
    /// computed uniformly the same way on every platform. It does **not**
    /// probe for GPU/ML-accelerator availability (Metal, DirectML, CUDA);
    /// earlier versions of this function unconditionally returned `true` on
    /// macOS/Windows regardless of the actual hardware, which was a
    /// fabricated result. Callers that need GPU acceleration detection
    /// should consult the `gpu` feature / `voirs-acoustic`'s device-hub
    /// detection instead.
    pub fn supports_hardware_acceleration(&self) -> bool {
        self.has_avx2 || self.has_sse2 || self.has_neon
    }
}

/// Platform-specific audio configuration
#[derive(Debug, Clone)]
pub struct AudioConfig {
    pub backend: String,
    pub sample_rate: u32,
    pub buffer_size: usize,
    pub channels: u16,
    pub use_exclusive_mode: bool,
}

impl AudioConfig {
    /// Get optimal audio configuration for current platform
    pub fn optimal() -> Self {
        let platform = PlatformInfo::current();

        Self {
            backend: platform.audio_backend.clone(),
            sample_rate: 44100, // Standard sample rate
            buffer_size: platform.optimal_buffer_size(),
            channels: 2,               // Stereo
            use_exclusive_mode: false, // Conservative default
        }
    }

    /// Get low-latency audio configuration
    pub fn low_latency() -> Self {
        let platform = PlatformInfo::current();

        Self {
            backend: platform.audio_backend.clone(),
            sample_rate: 48000, // Higher sample rate for low latency
            buffer_size: platform.optimal_buffer_size() / 2, // Smaller buffer
            channels: 2,
            use_exclusive_mode: true, // Exclusive mode for lower latency
        }
    }
}

/// C API functions for platform information
#[no_mangle]
pub extern "C" fn voirs_get_platform_info() -> *const PlatformInfo {
    PlatformInfo::current() as *const PlatformInfo
}

#[no_mangle]
pub extern "C" fn voirs_get_optimal_threads() -> u32 {
    PlatformInfo::current().optimal_threads() as u32
}

#[no_mangle]
pub extern "C" fn voirs_get_optimal_buffer_size() -> u32 {
    PlatformInfo::current().optimal_buffer_size() as u32
}

#[no_mangle]
pub extern "C" fn voirs_supports_hardware_acceleration() -> bool {
    PlatformInfo::current().supports_hardware_acceleration()
}

// Note: AudioConfig contains String which is not FFI-safe
// These functions are for internal use only
#[allow(improper_ctypes_definitions)]
#[no_mangle]
pub extern "C" fn voirs_get_audio_config_optimal() -> AudioConfig {
    AudioConfig::optimal()
}

#[allow(improper_ctypes_definitions)]
#[no_mangle]
pub extern "C" fn voirs_get_audio_config_low_latency() -> AudioConfig {
    AudioConfig::low_latency()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_info() {
        let info = PlatformInfo::current();
        assert!(!info.os.is_empty());
        assert!(!info.arch.is_empty());
        assert!(info.cpu_cores > 0);
        assert!(info.total_memory > 0);
    }

    #[test]
    fn test_optimal_threads() {
        let info = PlatformInfo::current();
        let threads = info.optimal_threads();
        assert!(threads > 0);
        assert!(threads <= 8);
        assert!(threads <= info.cpu_cores);
    }

    #[test]
    fn test_audio_config() {
        let config = AudioConfig::optimal();
        assert!(!config.backend.is_empty());
        assert!(config.sample_rate > 0);
        assert!(config.buffer_size > 0);
        assert!(config.channels > 0);
    }

    #[test]
    fn test_low_latency_config() {
        let config = AudioConfig::low_latency();
        let optimal = AudioConfig::optimal();

        assert!(config.buffer_size <= optimal.buffer_size);
        assert!(config.sample_rate >= optimal.sample_rate);
    }

    /// Cross-check `get_total_memory()` against a direct `sysctl` call so a
    /// regression back to a hardcoded constant (e.g. the old Windows "8 GB
    /// default") would be caught: a fabricated constant would not match this
    /// host's actual physical memory.
    #[test]
    fn test_total_memory_matches_sysctl_on_macos() {
        #[cfg(target_os = "macos")]
        {
            let Ok(output) = std::process::Command::new("sysctl")
                .args(["-n", "hw.memsize"])
                .output()
            else {
                return; // sysctl unavailable in this environment; nothing to compare.
            };
            let Ok(expected) = String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse::<u64>()
            else {
                return;
            };
            assert_eq!(PlatformInfo::current().total_memory, expected);
        }
    }

    /// `supports_hardware_acceleration()` must be exactly the SIMD-flag OR --
    /// uniformly on every platform -- never a fabricated `true` regardless of
    /// hardware (the previous macOS/Windows behavior).
    #[test]
    fn test_supports_hardware_acceleration_matches_simd_flags() {
        let info = PlatformInfo::current();
        assert_eq!(
            info.supports_hardware_acceleration(),
            info.has_avx2 || info.has_sse2 || info.has_neon
        );
    }
}
