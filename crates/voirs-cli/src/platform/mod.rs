//! Cross-platform compatibility and system integration
//!
//! This module provides platform-specific optimizations and features for Windows, macOS, and Linux.

use std::path::PathBuf;

pub mod hardware;
pub mod integration;

/// Platform-specific information and capabilities
#[derive(Debug, Clone)]
pub struct PlatformInfo {
    /// Operating system name
    pub os: String,
    /// OS version
    pub version: String,
    /// System architecture
    pub architecture: String,
    /// Available CPU cores
    pub cpu_cores: usize,
    /// Total system memory in bytes
    pub total_memory: u64,
    /// Available memory in bytes
    pub available_memory: u64,
}

/// Get current platform information
pub fn get_platform_info() -> PlatformInfo {
    let os = std::env::consts::OS.to_string();
    let architecture = std::env::consts::ARCH.to_string();
    let cpu_cores = num_cpus::get();

    // Get system memory information
    let (total_memory, available_memory) = get_memory_info();

    PlatformInfo {
        os: os.clone(),
        version: get_os_version(),
        architecture,
        cpu_cores,
        total_memory,
        available_memory,
    }
}

/// Get OS version string
fn get_os_version() -> String {
    #[cfg(target_os = "windows")]
    {
        // Windows version detection
        "Windows".to_string()
    }
    #[cfg(target_os = "macos")]
    {
        // macOS version detection
        use std::process::Command;
        match Command::new("sw_vers").arg("-productVersion").output() {
            Ok(output) => String::from_utf8_lossy(&output.stdout).trim().to_string(),
            Err(_) => "macOS".to_string(),
        }
    }
    #[cfg(target_os = "linux")]
    {
        // Linux version detection
        use std::fs;
        if let Ok(content) = fs::read_to_string("/etc/os-release") {
            for line in content.lines() {
                if line.starts_with("PRETTY_NAME=") {
                    return line
                        .trim_start_matches("PRETTY_NAME=")
                        .trim_matches('"')
                        .to_string();
                }
            }
        }
        "Linux".to_string()
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        "Unknown".to_string()
    }
}

/// Get system memory information (total, available)
fn get_memory_info() -> (u64, u64) {
    #[cfg(target_os = "windows")]
    {
        // Windows memory detection using Windows API
        (8_000_000_000, 4_000_000_000) // Placeholder values
    }
    #[cfg(target_os = "macos")]
    {
        // macOS memory detection
        use std::process::Command;
        let total = Command::new("sysctl")
            .arg("-n")
            .arg("hw.memsize")
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(8_000_000_000);

        // For available memory, use vm_stat
        let available = total / 2; // Simplified estimation
        (total, available)
    }
    #[cfg(target_os = "linux")]
    {
        // Linux memory detection from /proc/meminfo
        use std::fs;
        let mut total = 8_000_000_000u64;
        let mut available = 4_000_000_000u64;

        if let Ok(content) = fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(kb) = line.split_whitespace().nth(1) {
                        if let Ok(kb_val) = kb.parse::<u64>() {
                            total = kb_val * 1024; // Convert KB to bytes
                        }
                    }
                } else if line.starts_with("MemAvailable:") {
                    if let Some(kb) = line.split_whitespace().nth(1) {
                        if let Ok(kb_val) = kb.parse::<u64>() {
                            available = kb_val * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
        }
        (total, available)
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        (8_000_000_000, 4_000_000_000) // Default fallback
    }
}

/// Get platform-specific configuration directory
pub fn get_config_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        // Windows: %APPDATA%\VoiRS
        std::env::var("APPDATA")
            .ok()
            .map(|appdata| PathBuf::from(appdata).join("VoiRS"))
    }
    #[cfg(target_os = "macos")]
    {
        // macOS: ~/Library/Application Support/VoiRS
        dirs::home_dir().map(|home| {
            home.join("Library")
                .join("Application Support")
                .join("VoiRS")
        })
    }
    #[cfg(target_os = "linux")]
    {
        // Linux: ~/.config/voirs or $XDG_CONFIG_HOME/voirs
        std::env::var("XDG_CONFIG_HOME")
            .ok()
            .map(PathBuf::from)
            .or_else(|| dirs::home_dir().map(|home| home.join(".config")))
            .map(|config_dir| config_dir.join("voirs"))
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        dirs::home_dir().map(|home| home.join(".voirs"))
    }
}

/// Get platform-specific cache directory
pub fn get_cache_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        // Windows: %LOCALAPPDATA%\VoiRS\Cache
        std::env::var("LOCALAPPDATA")
            .ok()
            .map(|localappdata| PathBuf::from(localappdata).join("VoiRS").join("Cache"))
    }
    #[cfg(target_os = "macos")]
    {
        // macOS: ~/Library/Caches/VoiRS
        dirs::home_dir().map(|home| home.join("Library").join("Caches").join("VoiRS"))
    }
    #[cfg(target_os = "linux")]
    {
        // Linux: ~/.cache/voirs or $XDG_CACHE_HOME/voirs
        std::env::var("XDG_CACHE_HOME")
            .ok()
            .map(PathBuf::from)
            .or_else(|| dirs::home_dir().map(|home| home.join(".cache")))
            .map(|cache_dir| cache_dir.join("voirs"))
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        dirs::home_dir().map(|home| home.join(".voirs").join("cache"))
    }
}

/// Get platform-specific data directory  
pub fn get_data_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        // Windows: %LOCALAPPDATA%\VoiRS\Data
        std::env::var("LOCALAPPDATA")
            .ok()
            .map(|localappdata| PathBuf::from(localappdata).join("VoiRS").join("Data"))
    }
    #[cfg(target_os = "macos")]
    {
        // macOS: ~/Library/Application Support/VoiRS
        dirs::home_dir().map(|home| {
            home.join("Library")
                .join("Application Support")
                .join("VoiRS")
        })
    }
    #[cfg(target_os = "linux")]
    {
        // Linux: ~/.local/share/voirs or $XDG_DATA_HOME/voirs
        std::env::var("XDG_DATA_HOME")
            .ok()
            .map(PathBuf::from)
            .or_else(|| dirs::home_dir().map(|home| home.join(".local").join("share")))
            .map(|data_dir| data_dir.join("voirs"))
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        dirs::home_dir().map(|home| home.join(".voirs").join("data"))
    }
}

/// Ensure platform directories exist
pub fn ensure_platform_dirs() -> Result<(), Box<dyn std::error::Error>> {
    if let Some(config_dir) = get_config_dir() {
        std::fs::create_dir_all(&config_dir)?;
    }

    if let Some(cache_dir) = get_cache_dir() {
        std::fs::create_dir_all(&cache_dir)?;
    }

    if let Some(data_dir) = get_data_dir() {
        std::fs::create_dir_all(&data_dir)?;
    }

    Ok(())
}

/// Check if running with administrator/root privileges
pub fn is_elevated() -> bool {
    #[cfg(target_os = "windows")]
    {
        // Windows: Check if running as administrator
        false // Placeholder - would need Windows API calls
    }
    #[cfg(unix)]
    {
        // Unix: Check if running as root
        unsafe { libc::geteuid() == 0 }
    }
    #[cfg(not(any(target_os = "windows", unix)))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_info() {
        let info = get_platform_info();
        assert!(!info.os.is_empty());
        assert!(!info.architecture.is_empty());
        assert!(info.cpu_cores > 0);
        assert!(info.total_memory > 0);
    }

    #[test]
    fn test_platform_directories() {
        assert!(get_config_dir().is_some());
        assert!(get_cache_dir().is_some());
        assert!(get_data_dir().is_some());
    }

    #[test]
    fn test_ensure_directories() {
        // This test should not fail
        assert!(ensure_platform_dirs().is_ok());
    }
}
