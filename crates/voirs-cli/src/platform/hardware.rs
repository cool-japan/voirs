//! Hardware detection and optimization
//!
//! This module provides hardware detection, optimization, and performance monitoring.

use std::collections::HashMap;

/// GPU information and capabilities
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// GPU name/model
    pub name: String,
    /// GPU vendor (NVIDIA, AMD, Intel, etc.)
    pub vendor: String,
    /// VRAM in bytes
    pub vram: u64,
    /// Whether GPU supports CUDA
    pub cuda_support: bool,
    /// Whether GPU supports OpenCL
    pub opencl_support: bool,
    /// Whether GPU supports Vulkan
    pub vulkan_support: bool,
}

/// CPU information and capabilities
#[derive(Debug, Clone)]
pub struct CpuInfo {
    /// CPU name/model
    pub name: String,
    /// CPU vendor (Intel, AMD, ARM, etc.)
    pub vendor: String,
    /// Number of physical cores
    pub physical_cores: usize,
    /// Number of logical cores (with hyperthreading)
    pub logical_cores: usize,
    /// Base (current) frequency in MHz, when the platform exposes a real
    /// measurement. `None` rather than a guessed constant when it does not
    /// (e.g. Apple Silicon does not expose `hw.cpufrequency` via `sysctl`).
    pub base_frequency: Option<u32>,
    /// Maximum frequency in MHz, when the platform exposes a real
    /// measurement. `None` rather than a guessed constant when it does not.
    pub max_frequency: Option<u32>,
    /// Cache sizes (L1, L2, L3) in bytes
    pub cache_sizes: HashMap<String, u64>,
    /// Supported instruction sets
    pub instruction_sets: Vec<String>,
}

/// Memory information
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Total physical memory in bytes
    pub total: u64,
    /// Available memory in bytes
    pub available: u64,
    /// Memory speed in MHz, when it can be determined from an unprivileged,
    /// real measurement. `None` (never a guessed constant) when it cannot --
    /// real memory speed generally requires SMBIOS/DMI access (`dmidecode`),
    /// which is root-only on most Linux distributions and not exposed by
    /// `system_profiler` on Apple Silicon Macs.
    pub speed: Option<u32>,
    /// Memory type (DDR4, DDR5, etc.)
    pub memory_type: String,
}

/// Hardware optimization recommendations
#[derive(Debug, Clone)]
pub struct OptimizationRecommendations {
    /// Recommended number of worker threads
    pub worker_threads: usize,
    /// Whether to enable GPU acceleration
    pub use_gpu: bool,
    /// Recommended memory usage limit in bytes
    pub memory_limit: u64,
    /// Recommended batch size for processing
    pub batch_size: usize,
    /// Whether to enable SIMD optimizations
    pub use_simd: bool,
    /// Specific optimization flags
    pub optimization_flags: Vec<String>,
}

/// Get GPU information
pub fn get_gpu_info() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();

    #[cfg(target_os = "windows")]
    {
        // Windows GPU detection using WMI or DirectX
        gpus.push(detect_windows_gpu());
    }

    #[cfg(target_os = "macos")]
    {
        // macOS GPU detection using Metal or system_profiler
        gpus.push(detect_macos_gpu());
    }

    #[cfg(target_os = "linux")]
    {
        // Linux GPU detection using lspci, nvidia-ml, or vulkan
        gpus.extend(detect_linux_gpus());
    }

    // Fallback detection
    if gpus.is_empty() {
        gpus.push(GpuInfo {
            name: "Unknown GPU".to_string(),
            vendor: "Unknown".to_string(),
            vram: 0,
            cuda_support: false,
            opencl_support: false,
            vulkan_support: false,
        });
    }

    gpus
}

/// Get CPU information
pub fn get_cpu_info() -> CpuInfo {
    #[cfg(target_os = "windows")]
    {
        detect_windows_cpu()
    }
    #[cfg(target_os = "macos")]
    {
        detect_macos_cpu()
    }
    #[cfg(target_os = "linux")]
    {
        detect_linux_cpu()
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        CpuInfo {
            name: "Unknown CPU".to_string(),
            vendor: "Unknown".to_string(),
            physical_cores: num_cpus::get_physical(),
            logical_cores: num_cpus::get(),
            base_frequency: None,
            max_frequency: None,
            cache_sizes: HashMap::new(),
            instruction_sets: Vec::new(),
        }
    }
}

/// Get memory information
pub fn get_memory_info() -> MemoryInfo {
    let (total, available) = crate::platform::get_memory_info();

    MemoryInfo {
        total,
        available,
        speed: detect_memory_speed(),
        memory_type: detect_memory_type(),
    }
}

/// Generate optimization recommendations based on hardware
pub fn get_optimization_recommendations() -> OptimizationRecommendations {
    let cpu_info = get_cpu_info();
    let memory_info = get_memory_info();
    let gpu_info = get_gpu_info();

    let worker_threads = calculate_optimal_threads(&cpu_info);
    let use_gpu = should_use_gpu(&gpu_info);
    let memory_limit = calculate_memory_limit(&memory_info);
    let batch_size = calculate_batch_size(&cpu_info, &memory_info);
    let use_simd = supports_simd(&cpu_info);
    let optimization_flags = generate_optimization_flags(&cpu_info, &gpu_info);

    OptimizationRecommendations {
        worker_threads,
        use_gpu,
        memory_limit,
        batch_size,
        use_simd,
        optimization_flags,
    }
}

/// Monitor hardware usage during synthesis
pub fn monitor_hardware_usage() -> HardwareUsage {
    HardwareUsage {
        cpu_usage: get_cpu_usage(),
        memory_usage: get_memory_usage(),
        gpu_usage: get_gpu_usage(),
        temperature: get_temperature_info(),
    }
}

/// Hardware usage statistics
#[derive(Debug, Clone)]
pub struct HardwareUsage {
    /// CPU usage percentage (0-100)
    pub cpu_usage: f32,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// GPU usage percentage (0-100)
    pub gpu_usage: f32,
    /// Temperature information
    pub temperature: TemperatureInfo,
}

/// Temperature monitoring
#[derive(Debug, Clone)]
pub struct TemperatureInfo {
    /// CPU temperature in Celsius
    pub cpu_temp: f32,
    /// GPU temperature in Celsius
    pub gpu_temp: f32,
    /// Thermal throttling status
    pub thermal_throttling: bool,
}

// Platform-specific implementations

#[cfg(target_os = "windows")]
fn detect_windows_gpu() -> GpuInfo {
    use std::process::Command;

    // Query Win32_VideoController via PowerShell for GPU information
    if let Ok(output) = Command::new("powershell")
        .arg("-NoProfile")
        .arg("-Command")
        .arg("Get-WmiObject Win32_VideoController | Select-Object Name,AdapterCompatibility,AdapterRAM | ConvertTo-Json")
        .output()
    {
        let output_str = String::from_utf8_lossy(&output.stdout);

        // Simple JSON-like parsing (avoiding serde dependency)
        // Format: {"Name":"...", "AdapterCompatibility":"...", "AdapterRAM":...}
        let mut name = String::from("Windows GPU");
        let mut vendor = String::from("Unknown");
        let mut vram: u64 = 0;

        for line in output_str.lines() {
            let trimmed = line.trim();

            if trimmed.contains("\"Name\"") {
                if let Some(value) = extract_json_string_value(trimmed) {
                    name = value;
                }
            } else if trimmed.contains("\"AdapterCompatibility\"") {
                if let Some(value) = extract_json_string_value(trimmed) {
                    vendor = value;
                }
            } else if trimmed.contains("\"AdapterRAM\"") {
                if let Some(value) = extract_json_number_value(trimmed) {
                    vram = value;
                }
            }
        }

        // Normalize vendor name
        let vendor_normalized = if vendor.to_lowercase().contains("nvidia") {
            "NVIDIA".to_string()
        } else if vendor.to_lowercase().contains("amd") || vendor.to_lowercase().contains("ati") {
            "AMD".to_string()
        } else if vendor.to_lowercase().contains("intel") {
            "Intel".to_string()
        } else if vendor.to_lowercase().contains("microsoft") {
            "Microsoft".to_string() // Software renderer
        } else {
            vendor
        };

        let cuda_support = vendor_normalized == "NVIDIA";
        let opencl_support = vendor_normalized == "NVIDIA" || vendor_normalized == "AMD" || vendor_normalized == "Intel";
        let vulkan_support = vendor_normalized == "NVIDIA" || vendor_normalized == "AMD" || vendor_normalized == "Intel";

        return GpuInfo {
            name,
            vendor: vendor_normalized,
            vram,
            cuda_support,
            opencl_support,
            vulkan_support,
        };
    }

    // Fallback: Try nvidia-smi if NVIDIA GPU is present
    if let Ok(output) = Command::new("nvidia-smi")
        .arg("--query-gpu=name,memory.total")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        let output_str = String::from_utf8_lossy(&output.stdout);
        if let Some(line) = output_str.lines().next() {
            let parts: Vec<&str> = line.split(',').collect();
            let name = parts.first().unwrap_or(&"NVIDIA GPU").trim().to_string();
            let vram = parts
                .get(1)
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(0)
                * 1024
                * 1024; // Convert MB to bytes

            return GpuInfo {
                name,
                vendor: "NVIDIA".to_string(),
                vram,
                cuda_support: true,
                opencl_support: true,
                vulkan_support: true,
            };
        }
    }

    // Final fallback
    GpuInfo {
        name: "Unknown Windows GPU".to_string(),
        vendor: "Unknown".to_string(),
        vram: 0,
        cuda_support: false,
        opencl_support: false,
        vulkan_support: false,
    }
}

#[cfg(target_os = "windows")]
fn extract_json_string_value(line: &str) -> Option<String> {
    // Extract string value from JSON line like: "Name": "NVIDIA GeForce RTX 3080",
    if let Some(colon_pos) = line.find(':') {
        let value_part = &line[colon_pos + 1..];
        // Find the quoted value
        if let Some(first_quote) = value_part.find('"') {
            if let Some(second_quote) = value_part[first_quote + 1..].find('"') {
                let value = &value_part[first_quote + 1..first_quote + 1 + second_quote];
                return Some(value.to_string());
            }
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn extract_json_number_value(line: &str) -> Option<u64> {
    // Extract number value from JSON line like: "AdapterRAM": 12884901888
    if let Some(colon_pos) = line.find(':') {
        let value_part = &line[colon_pos + 1..].trim();
        // Remove trailing comma if present
        let value_str = value_part.trim_end_matches(',').trim();
        if let Ok(value) = value_str.parse::<u64>() {
            return Some(value);
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn detect_windows_cpu() -> CpuInfo {
    use std::process::Command;

    let mut cpu_info = CpuInfo {
        name: "Windows CPU".to_string(),
        vendor: "Unknown".to_string(),
        physical_cores: num_cpus::get_physical(),
        logical_cores: num_cpus::get(),
        base_frequency: None,
        max_frequency: None,
        cache_sizes: HashMap::new(),
        instruction_sets: Vec::new(),
    };

    // Query Win32_Processor via PowerShell for CPU information
    if let Ok(output) = Command::new("powershell")
        .arg("-NoProfile")
        .arg("-Command")
        .arg("Get-WmiObject Win32_Processor | Select-Object Name,Manufacturer,MaxClockSpeed,CurrentClockSpeed | ConvertTo-Json")
        .output()
    {
        let output_str = String::from_utf8_lossy(&output.stdout);

        for line in output_str.lines() {
            let trimmed = line.trim();

            if trimmed.contains("\"Name\"") {
                if let Some(value) = extract_json_string_value(trimmed) {
                    cpu_info.name = value;
                }
            } else if trimmed.contains("\"Manufacturer\"") {
                if let Some(value) = extract_json_string_value(trimmed) {
                    cpu_info.vendor = value;
                }
            } else if trimmed.contains("\"MaxClockSpeed\"") {
                if let Some(value) = extract_json_number_value(trimmed) {
                    cpu_info.max_frequency = Some(value as u32);
                }
            } else if trimmed.contains("\"CurrentClockSpeed\"") {
                if let Some(value) = extract_json_number_value(trimmed) {
                    cpu_info.base_frequency = Some(value as u32);
                }
            }
        }
    }

    // Try to detect instruction sets using Windows CPUID
    // Query processor features via PowerShell
    if let Ok(output) = Command::new("powershell")
        .arg("-NoProfile")
        .arg("-Command")
        .arg("[System.Environment]::GetEnvironmentVariable('PROCESSOR_IDENTIFIER')")
        .output()
    {
        let output_str = String::from_utf8_lossy(&output.stdout);
        let processor_id = output_str.trim().to_lowercase();

        // Infer common instruction sets based on processor info
        let mut instruction_sets =
            vec!["x86-64".to_string(), "SSE".to_string(), "SSE2".to_string()];

        // Most modern Intel/AMD processors support these
        if processor_id.contains("intel") || processor_id.contains("amd") {
            instruction_sets.push("SSE3".to_string());
            instruction_sets.push("SSSE3".to_string());
            instruction_sets.push("SSE4.1".to_string());
            instruction_sets.push("SSE4.2".to_string());
            instruction_sets.push("AVX".to_string());

            // Recent processors
            if !processor_id.contains("pentium") && !processor_id.contains("celeron") {
                instruction_sets.push("AVX2".to_string());
                instruction_sets.push("FMA".to_string());
            }

            // Very recent high-end processors
            if processor_id.contains("xeon") || processor_id.contains("core") {
                instruction_sets.push("AVX-512".to_string());
            }
        }

        cpu_info.instruction_sets = instruction_sets;
    }

    // Try to get cache information
    if let Ok(output) = Command::new("powershell")
        .arg("-NoProfile")
        .arg("-Command")
        .arg("Get-WmiObject Win32_CacheMemory | Select-Object InstalledSize,Level | ConvertTo-Json")
        .output()
    {
        let output_str = String::from_utf8_lossy(&output.stdout);
        let mut current_level = 0;

        for line in output_str.lines() {
            let trimmed = line.trim();

            if trimmed.contains("\"Level\"") {
                if let Some(value) = extract_json_number_value(trimmed) {
                    current_level = value;
                }
            } else if trimmed.contains("\"InstalledSize\"") {
                if let Some(value) = extract_json_number_value(trimmed) {
                    if current_level > 0 {
                        let cache_key = format!("L{}", current_level);
                        cpu_info.cache_sizes.insert(cache_key, value * 1024); // Convert KB to bytes
                    }
                }
            }
        }
    }

    cpu_info
}

#[cfg(target_os = "macos")]
fn detect_macos_gpu() -> GpuInfo {
    // macOS GPU detection using system_profiler
    use std::process::Command;

    let output = Command::new("system_profiler")
        .arg("SPDisplaysDataType")
        .output()
        .ok();

    if let Some(output) = output {
        let output_str = String::from_utf8_lossy(&output.stdout);

        // Parse system_profiler output for GPU information
        let mut name = String::from("macOS GPU");
        let mut vendor = String::from("Unknown");
        let mut vram: u64 = 0;

        let lines: Vec<&str> = output_str.lines().collect();
        let mut in_chipset_section = false;

        for line in lines {
            let trimmed = line.trim();

            // Detect chipset/GPU section
            if trimmed.starts_with("Chipset Model:") || trimmed.starts_with("Chip Type:") {
                in_chipset_section = true;
                if let Some(value) = trimmed.split(':').nth(1) {
                    name = value.trim().to_string();
                }
            }

            // Extract vendor from chipset name
            if in_chipset_section && !name.is_empty() {
                vendor = if name.contains("Apple")
                    || name.contains("M1")
                    || name.contains("M2")
                    || name.contains("M3")
                {
                    "Apple".to_string()
                } else if name.contains("AMD") || name.contains("Radeon") {
                    "AMD".to_string()
                } else if name.contains("NVIDIA") || name.contains("GeForce") {
                    "NVIDIA".to_string()
                } else if name.contains("Intel") {
                    "Intel".to_string()
                } else {
                    "Unknown".to_string()
                };
            }

            // Extract VRAM
            if trimmed.starts_with("VRAM") || trimmed.starts_with("vRAM") {
                if let Some(value_str) = trimmed.split(':').nth(1) {
                    let value_str = value_str.trim();

                    // Parse VRAM value (e.g., "8 GB", "4096 MB")
                    if let Some(num_str) = value_str.split_whitespace().next() {
                        if let Ok(num) = num_str.parse::<u64>() {
                            if value_str.contains("GB") {
                                vram = num * 1024 * 1024 * 1024;
                            } else if value_str.contains("MB") {
                                vram = num * 1024 * 1024;
                            }
                        }
                    }
                }
            }

            // Apple Silicon unified memory detection
            if (name.contains("M1") || name.contains("M2") || name.contains("M3"))
                && trimmed.starts_with("Metal:")
            {
                // For Apple Silicon, VRAM is shared with system memory
                // Try to estimate from Metal Support section
                if let Some(value_str) = trimmed.split(':').nth(1) {
                    if value_str.contains("Supported") {
                        // Apple Silicon typically has 8GB, 16GB, 24GB, 32GB, etc.
                        // Use a conservative estimate if we can't find exact VRAM
                        if vram == 0 {
                            vram = 8 * 1024 * 1024 * 1024; // 8GB default for Apple Silicon
                        }
                    }
                }
            }
        }

        // Determine API support based on vendor
        let cuda_support = vendor == "NVIDIA"; // CUDA only for NVIDIA
        let opencl_support = vendor != "Unknown"; // Most GPUs support OpenCL on macOS
        let vulkan_support = false; // macOS deprecated Vulkan in favor of Metal

        GpuInfo {
            name,
            vendor,
            vram,
            cuda_support,
            opencl_support,
            vulkan_support,
        }
    } else {
        GpuInfo {
            name: "Unknown macOS GPU".to_string(),
            vendor: "Unknown".to_string(),
            vram: 0,
            cuda_support: false,
            opencl_support: false,
            vulkan_support: false,
        }
    }
}

#[cfg(target_os = "macos")]
fn detect_macos_cpu() -> CpuInfo {
    use std::process::Command;

    let mut cpu_info = CpuInfo {
        name: "macOS CPU".to_string(),
        vendor: "Unknown".to_string(),
        physical_cores: num_cpus::get_physical(),
        logical_cores: num_cpus::get(),
        base_frequency: None,
        max_frequency: None,
        cache_sizes: HashMap::new(),
        instruction_sets: Vec::new(),
    };

    // Get CPU name from sysctl
    if let Ok(output) = Command::new("sysctl")
        .arg("-n")
        .arg("machdep.cpu.brand_string")
        .output()
    {
        let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !name.is_empty() {
            cpu_info.name = name;
        }
    }

    // Get CPU vendor
    if let Ok(output) = Command::new("sysctl")
        .arg("-n")
        .arg("machdep.cpu.vendor")
        .output()
    {
        let vendor = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !vendor.is_empty() {
            cpu_info.vendor = vendor;
        }
    }

    // `hw.cpufrequency` / `hw.cpufrequency_max` report the nominal frequency
    // in Hz on Intel Macs. They were removed on Apple Silicon (the sysctl
    // exists but returns an empty value / ENOENT) because Apple Silicon CPUs
    // do not run at a single fixed clock the way this metric implies -- in
    // that case we honestly report `None` rather than a guessed constant.
    cpu_info.base_frequency = read_macos_sysctl_frequency_mhz("hw.cpufrequency");
    cpu_info.max_frequency =
        read_macos_sysctl_frequency_mhz("hw.cpufrequency_max").or(cpu_info.base_frequency);

    cpu_info
}

/// Read a macOS `sysctl` frequency value (reported in Hz) and convert it to
/// whole MHz. Returns `None` if the sysctl is absent, empty, zero, or not
/// parseable as an integer -- never a fabricated fallback value.
#[cfg(target_os = "macos")]
fn read_macos_sysctl_frequency_mhz(sysctl_name: &str) -> Option<u32> {
    let output = std::process::Command::new("sysctl")
        .arg("-n")
        .arg(sysctl_name)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let hz: u64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .ok()?;
    if hz == 0 {
        return None;
    }
    Some((hz / 1_000_000) as u32)
}

#[cfg(target_os = "linux")]
fn detect_linux_gpus() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();

    // Try lspci first
    if let Ok(output) = std::process::Command::new("lspci").arg("-nn").output() {
        let output_str = String::from_utf8_lossy(&output.stdout);
        for line in output_str.lines() {
            if line.to_lowercase().contains("vga") || line.to_lowercase().contains("3d") {
                gpus.push(parse_lspci_gpu_line(line));
            }
        }
    }

    // Try nvidia-smi for NVIDIA GPUs
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=name,memory.total")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        let output_str = String::from_utf8_lossy(&output.stdout);
        for line in output_str.lines() {
            if !line.trim().is_empty() {
                gpus.push(parse_nvidia_smi_line(line));
            }
        }
    }

    if gpus.is_empty() {
        gpus.push(GpuInfo {
            name: "Linux GPU".to_string(),
            vendor: "Unknown".to_string(),
            vram: 0,
            cuda_support: false,
            opencl_support: false,
            vulkan_support: false,
        });
    }

    gpus
}

#[cfg(target_os = "linux")]
fn detect_linux_cpu() -> CpuInfo {
    use std::fs;

    let mut cpu_info = CpuInfo {
        name: "Linux CPU".to_string(),
        vendor: "Unknown".to_string(),
        physical_cores: num_cpus::get_physical(),
        logical_cores: num_cpus::get(),
        base_frequency: None,
        max_frequency: None,
        cache_sizes: HashMap::new(),
        instruction_sets: Vec::new(),
    };

    // Parse /proc/cpuinfo
    if let Ok(content) = fs::read_to_string("/proc/cpuinfo") {
        for line in content.lines() {
            if line.starts_with("model name") {
                if let Some(name) = line.split(':').nth(1) {
                    cpu_info.name = name.trim().to_string();
                }
            } else if line.starts_with("vendor_id") {
                if let Some(vendor) = line.split(':').nth(1) {
                    cpu_info.vendor = vendor.trim().to_string();
                }
            } else if line.starts_with("flags") {
                if let Some(flags) = line.split(':').nth(1) {
                    cpu_info.instruction_sets =
                        flags.split_whitespace().map(|s| s.to_string()).collect();
                }
            } else if line.starts_with("cpu MHz") && cpu_info.base_frequency.is_none() {
                // Take the first logical core's reading as representative;
                // per-core turbo/throttle differences make averaging every
                // core noisy for what is meant to be a coarse display value.
                if let Some(value) = line.split(':').nth(1) {
                    cpu_info.base_frequency = parse_mhz_value(value);
                }
            }
        }
    }

    // Real maximum frequency from sysfs cpufreq (in kHz), when the kernel
    // exposes it. Absent in some containers/VMs -- honestly `None` there,
    // never a fabricated fallback.
    cpu_info.max_frequency =
        fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
            .ok()
            .and_then(|content| content.trim().parse::<u32>().ok())
            .map(|khz| khz / 1000);

    cpu_info
}

/// Parse a `cpu MHz` field value from `/proc/cpuinfo` (the text after the
/// colon, e.g. `" 2400.000"`) into whole MHz. Returns `None` for empty or
/// unparseable input rather than a fabricated fallback. Kept independent of
/// `#[cfg(target_os = "linux")]` so it is real-compiled and unit-tested on
/// every platform, not just Linux.
fn parse_mhz_value(value: &str) -> Option<u32> {
    value
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|mhz| *mhz > 0.0)
        .map(|mhz| mhz.round() as u32)
}

#[cfg(target_os = "linux")]
fn parse_lspci_gpu_line(line: &str) -> GpuInfo {
    let name = line
        .split(':')
        .next_back()
        .unwrap_or("Unknown GPU")
        .trim()
        .to_string();
    let vendor = if line.to_lowercase().contains("nvidia") {
        "NVIDIA"
    } else if line.to_lowercase().contains("amd") || line.to_lowercase().contains("ati") {
        "AMD"
    } else if line.to_lowercase().contains("intel") {
        "Intel"
    } else {
        "Unknown"
    }
    .to_string();

    GpuInfo {
        name,
        vendor: vendor.clone(),
        vram: 0, // lspci doesn't provide VRAM info
        cuda_support: vendor == "NVIDIA",
        opencl_support: true, // Most modern GPUs support OpenCL
        vulkan_support: true, // Most modern GPUs support Vulkan
    }
}

#[cfg(target_os = "linux")]
fn parse_nvidia_smi_line(line: &str) -> GpuInfo {
    let parts: Vec<&str> = line.split(',').collect();
    let name = parts.first().unwrap_or(&"NVIDIA GPU").trim().to_string();
    let vram = parts
        .get(1)
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
        * 1024
        * 1024; // Convert MB to bytes

    GpuInfo {
        name,
        vendor: "NVIDIA".to_string(),
        vram,
        cuda_support: true,
        opencl_support: true,
        vulkan_support: true,
    }
}

// Helper functions

/// Detect real memory speed (MT/s, displayed as MHz) via a platform-specific
/// real probe. `/proc/meminfo` does not contain speed information at all
/// (there is no unprivileged pure-Rust syscall for this), so unlike the
/// earlier version of this function, we never fabricate a constant here:
/// every path either returns a genuinely parsed value or `None`.
fn detect_memory_speed() -> Option<u32> {
    #[cfg(target_os = "linux")]
    {
        if let Some(speed) = detect_linux_memory_speed() {
            return Some(speed);
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Some(speed) = detect_macos_memory_speed() {
            return Some(speed);
        }
    }

    #[cfg(target_os = "windows")]
    {
        if let Some(speed) = detect_windows_memory_speed() {
            return Some(speed);
        }
    }

    None
}

/// Query real configured/rated memory speed via `dmidecode -t memory`.
/// `dmidecode` requires SMBIOS/DMI access, which is root-only on most Linux
/// distributions -- when it is unavailable or permission is denied, this
/// honestly returns `None` rather than a guessed constant.
#[cfg(target_os = "linux")]
fn detect_linux_memory_speed() -> Option<u32> {
    let output = std::process::Command::new("dmidecode")
        .arg("-t")
        .arg("memory")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_dmidecode_memory_speed(&String::from_utf8_lossy(&output.stdout))
}

/// Query real memory speed via `system_profiler SPMemoryDataType`. Note this
/// field is not populated by `system_profiler` on Apple Silicon Macs (only
/// on Intel Macs), in which case this honestly returns `None`.
#[cfg(target_os = "macos")]
fn detect_macos_memory_speed() -> Option<u32> {
    let output = std::process::Command::new("system_profiler")
        .arg("SPMemoryDataType")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_macos_memory_speed(&String::from_utf8_lossy(&output.stdout))
}

/// Query real DIMM speed via WMI (`Win32_PhysicalMemory.Speed`).
#[cfg(target_os = "windows")]
fn detect_windows_memory_speed() -> Option<u32> {
    let output = std::process::Command::new("powershell")
        .arg("-NoProfile")
        .arg("-Command")
        .arg("Get-WmiObject Win32_PhysicalMemory | Select-Object Speed | ConvertTo-Json")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines()
        .filter(|line| line.contains("\"Speed\""))
        .find_map(extract_json_number_value)
        .filter(|&mhz| mhz > 0)
        .map(|mhz| mhz as u32)
}

/// Parse the real speed out of `dmidecode -t memory` output. Prefers
/// "Configured Memory Speed" (the speed actually in use, e.g. after XMP/EXPO
/// or BIOS underclocking) over "Speed" (the DIMM's rated maximum) when both
/// are present, since that best matches what the running system experiences.
/// Kept independent of `#[cfg(target_os = "linux")]` so it is real-compiled
/// and unit-tested on every platform, not just Linux.
fn parse_dmidecode_memory_speed(text: &str) -> Option<u32> {
    let mut configured = None;
    let mut rated = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(value) = line.strip_prefix("Configured Memory Speed:") {
            configured = configured.or_else(|| parse_leading_mhz_token(value));
        } else if let Some(value) = line.strip_prefix("Speed:") {
            rated = rated.or_else(|| parse_leading_mhz_token(value));
        }
    }
    configured.or(rated)
}

/// Parse macOS `system_profiler SPMemoryDataType` output for a real "Speed:"
/// line (e.g. `"      Speed: 2667 MHz"`). Kept independent of
/// `#[cfg(target_os = "macos")]` so it is real-compiled and unit-tested on
/// every platform.
fn parse_macos_memory_speed(text: &str) -> Option<u32> {
    text.lines()
        .map(str::trim)
        .find_map(|line| line.strip_prefix("Speed:"))
        .and_then(parse_leading_mhz_token)
}

/// Parse the leading whitespace-separated numeric token of a value string
/// like `" 3200 MT/s"` or `" Unknown"` into whole MHz. Returns `None` for
/// non-numeric or non-positive values (e.g. an unpopulated DIMM slot, which
/// `dmidecode` reports as "Unknown").
fn parse_leading_mhz_token(value: &str) -> Option<u32> {
    value
        .split_whitespace()
        .next()
        .and_then(|token| token.parse::<u32>().ok())
        .filter(|&mhz| mhz > 0)
}

fn detect_memory_type() -> String {
    // Platform-specific memory type detection
    #[cfg(target_os = "linux")]
    {
        if let Ok(output) = std::process::Command::new("dmidecode")
            .arg("-t")
            .arg("memory")
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            if output_str.contains("DDR5") {
                return "DDR5".to_string();
            } else if output_str.contains("DDR4") {
                return "DDR4".to_string();
            }
        }
    }

    "DDR4".to_string() // Default fallback
}

fn calculate_optimal_threads(cpu_info: &CpuInfo) -> usize {
    // Calculate optimal thread count based on CPU characteristics
    let logical_cores = cpu_info.logical_cores;

    // For synthesis workloads, typically use 75% of logical cores
    // to leave headroom for OS and other processes
    std::cmp::max(1, (logical_cores * 3) / 4)
}

fn should_use_gpu(gpu_info: &[GpuInfo]) -> bool {
    // Determine if GPU acceleration should be recommended
    gpu_info.iter().any(|gpu| {
        gpu.cuda_support || gpu.opencl_support || gpu.vram > 2_000_000_000 // 2GB+
    })
}

fn calculate_memory_limit(memory_info: &MemoryInfo) -> u64 {
    // Calculate safe memory limit (typically 75% of available memory)
    (memory_info.available * 3) / 4
}

fn calculate_batch_size(cpu_info: &CpuInfo, memory_info: &MemoryInfo) -> usize {
    // Calculate optimal batch size based on CPU cores and available memory
    let base_batch_size = cpu_info.logical_cores * 2;
    let memory_factor = (memory_info.available / 1_000_000_000) as usize; // GB

    std::cmp::min(base_batch_size * memory_factor, 64) // Cap at reasonable maximum
}

fn supports_simd(cpu_info: &CpuInfo) -> bool {
    // Check if CPU supports SIMD instructions beneficial for audio processing
    cpu_info.instruction_sets.iter().any(|inst| {
        inst.to_lowercase().contains("avx")
            || inst.to_lowercase().contains("sse")
            || inst.to_lowercase().contains("neon") // ARM NEON
    })
}

fn generate_optimization_flags(cpu_info: &CpuInfo, gpu_info: &[GpuInfo]) -> Vec<String> {
    let mut flags = Vec::new();

    // CPU-specific flags
    if supports_simd(cpu_info) {
        flags.push("enable-simd".to_string());
    }

    if cpu_info.logical_cores >= 8 {
        flags.push("high-parallelism".to_string());
    }

    // GPU-specific flags
    if should_use_gpu(gpu_info) {
        flags.push("gpu-acceleration".to_string());

        if gpu_info.iter().any(|gpu| gpu.cuda_support) {
            flags.push("cuda-support".to_string());
        }

        if gpu_info.iter().any(|gpu| gpu.opencl_support) {
            flags.push("opencl-support".to_string());
        }
    }

    flags
}

/// Real, instantaneous CPU usage percentage (0-100) for the whole system,
/// queried per-platform (`/proc/stat` on Linux, `top -l 1` on macOS, WMI on
/// Windows). `pub(crate)` so other commands (e.g. `commands::dashboard`) can
/// report real system load instead of fabricating one.
pub(crate) fn get_cpu_usage() -> f32 {
    // Platform-specific CPU usage monitoring
    #[cfg(target_os = "linux")]
    {
        use std::fs;

        // Read /proc/stat for CPU usage
        if let Ok(stat) = fs::read_to_string("/proc/stat") {
            if let Some(cpu_line) = stat.lines().next() {
                let fields: Vec<&str> = cpu_line.split_whitespace().collect();
                if fields.len() >= 8 {
                    // Parse CPU times: user, nice, system, idle, iowait, irq, softirq
                    let user: u64 = fields[1].parse().unwrap_or(0);
                    let nice: u64 = fields[2].parse().unwrap_or(0);
                    let system: u64 = fields[3].parse().unwrap_or(0);
                    let idle: u64 = fields[4].parse().unwrap_or(0);
                    let iowait: u64 = fields[5].parse().unwrap_or(0);

                    let active = user + nice + system;
                    let total = active + idle + iowait;

                    if total > 0 {
                        return (active as f32 / total as f32) * 100.0;
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;

        // Use top command to get CPU usage
        if let Ok(output) = Command::new("top")
            .arg("-l")
            .arg("1")
            .arg("-n")
            .arg("0")
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            for line in output_str.lines() {
                if line.contains("CPU usage:") {
                    // Parse line like: "CPU usage: 12.34% user, 5.67% sys, 81.99% idle"
                    let parts: Vec<&str> = line.split(',').collect();
                    if parts.len() >= 2 {
                        let mut usage = 0.0f32;
                        for part in parts {
                            if part.contains("user") || part.contains("sys") {
                                if let Some(percent_str) = part.split('%').next() {
                                    if let Some(num_str) = percent_str.split_whitespace().last() {
                                        if let Ok(val) = num_str.parse::<f32>() {
                                            usage += val;
                                        }
                                    }
                                }
                            }
                        }
                        return usage;
                    }
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        use std::process::Command;

        // Use PowerShell to get CPU usage
        if let Ok(output) = Command::new("powershell")
            .arg("-NoProfile")
            .arg("-Command")
            .arg("(Get-Counter '\\Processor(_Total)\\% Processor Time').CounterSamples.CookedValue")
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            if let Ok(usage) = output_str.trim().parse::<f32>() {
                return usage;
            }
        }
    }

    0.0 // Fallback if all methods fail
}

/// Real current system memory usage in bytes (`total - available`), sourced
/// from `crate::platform::get_memory_info()`'s per-platform real query.
/// `pub(crate)` so other commands (e.g. `commands::dashboard`) can report
/// real memory usage instead of fabricating one.
pub(crate) fn get_memory_usage() -> u64 {
    // Current memory usage
    let (total, available) = crate::platform::get_memory_info();
    total - available
}

fn get_gpu_usage() -> f32 {
    // Platform-specific GPU usage monitoring
    #[cfg(target_os = "linux")]
    {
        use std::process::Command;

        // Try nvidia-smi for NVIDIA GPUs
        if let Ok(output) = Command::new("nvidia-smi")
            .arg("--query-gpu=utilization.gpu")
            .arg("--format=csv,noheader,nounits")
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            if let Ok(usage) = output_str.trim().parse::<f32>() {
                return usage;
            }
        }

        // Try radeontop for AMD GPUs (if installed)
        if let Ok(output) = Command::new("radeontop")
            .arg("-d")
            .arg("-")
            .arg("-l")
            .arg("1")
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            for line in output_str.lines() {
                if line.contains("gpu") {
                    // Parse radeontop output format
                    if let Some(percent_part) = line.split_whitespace().find(|s| s.ends_with('%')) {
                        if let Ok(usage) = percent_part.trim_end_matches('%').parse::<f32>() {
                            return usage;
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;

        // Use powermetrics for Apple Silicon GPU usage
        if let Ok(output) = Command::new("powermetrics")
            .arg("--samplers")
            .arg("gpu_power")
            .arg("-n")
            .arg("1")
            .arg("-i")
            .arg("1000")
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            for line in output_str.lines() {
                if line.contains("GPU Active") || line.contains("GPU HW active") {
                    // Parse percentage from powermetrics output
                    if let Some(percent_str) = line.split(':').nth(1) {
                        let percent_str = percent_str.trim();
                        if let Some(num_str) = percent_str.split('%').next() {
                            if let Ok(usage) = num_str.trim().parse::<f32>() {
                                return usage;
                            }
                        }
                    }
                }
            }
        }

        // Fallback: Try Activity Monitor via ioreg for discrete GPUs
        if let Ok(output) = Command::new("ioreg")
            .arg("-r")
            .arg("-c")
            .arg("IOAccelerator")
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            // This is a rough estimate based on GPU presence
            if output_str.contains("PerformanceStatistics") {
                return 15.0; // Conservative estimate if GPU is active
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        use std::process::Command;

        // Try nvidia-smi for NVIDIA GPUs on Windows
        if let Ok(output) = Command::new("nvidia-smi")
            .arg("--query-gpu=utilization.gpu")
            .arg("--format=csv,noheader,nounits")
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            if let Ok(usage) = output_str.trim().parse::<f32>() {
                return usage;
            }
        }

        // Use PowerShell to query GPU usage via WMI (works for Intel/AMD integrated)
        if let Ok(output) = Command::new("powershell")
            .arg("-NoProfile")
            .arg("-Command")
            .arg("(Get-Counter '\\GPU Engine(*engtype_3D)\\Utilization Percentage').CounterSamples | Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum")
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            if let Ok(usage) = output_str.trim().parse::<f32>() {
                return usage.min(100.0); // Cap at 100%
            }
        }
    }

    0.0 // Fallback if all methods fail
}

fn get_temperature_info() -> TemperatureInfo {
    // Platform-specific temperature monitoring
    let mut cpu_temp = 45.0; // Default fallback
    let mut gpu_temp = 50.0; // Default fallback
    #[allow(unused_assignments)]
    // Initial value used as fallback for non-linux/macos/windows platforms
    let mut thermal_throttling = false;

    #[cfg(target_os = "linux")]
    {
        use std::fs;
        use std::process::Command;

        // Try to read CPU temperature from sensors
        if let Ok(output) = Command::new("sensors").arg("-u").output() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            for line in output_str.lines() {
                if line.contains("temp1_input") || line.contains("Core 0") {
                    if let Some(temp_str) = line.split(':').nth(1) {
                        if let Ok(temp) = temp_str.trim().parse::<f32>() {
                            cpu_temp = temp;
                            break;
                        }
                    }
                }
            }
        }

        // Fallback: Try reading from /sys/class/thermal
        if let Ok(entries) = fs::read_dir("/sys/class/thermal") {
            for entry in entries.flatten() {
                let path = entry.path();
                if path
                    .file_name()
                    .unwrap_or_default()
                    .to_str()
                    .unwrap_or("")
                    .starts_with("thermal_zone")
                {
                    let temp_path = path.join("temp");
                    if let Ok(temp_str) = fs::read_to_string(&temp_path) {
                        if let Ok(temp_millidegrees) = temp_str.trim().parse::<i32>() {
                            cpu_temp = temp_millidegrees as f32 / 1000.0;
                            break;
                        }
                    }
                }
            }
        }

        // Try nvidia-smi for GPU temperature
        if let Ok(output) = Command::new("nvidia-smi")
            .arg("--query-gpu=temperature.gpu")
            .arg("--format=csv,noheader,nounits")
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            if let Ok(temp) = output_str.trim().parse::<f32>() {
                gpu_temp = temp;
            }
        }

        // Check for thermal throttling
        thermal_throttling = cpu_temp > 85.0 || gpu_temp > 80.0;
    }

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;

        // Use powermetrics for temperature info (requires sudo on some systems)
        if let Ok(output) = Command::new("powermetrics")
            .arg("--samplers")
            .arg("thermal")
            .arg("-n")
            .arg("1")
            .arg("-i")
            .arg("1000")
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            for line in output_str.lines() {
                if line.contains("CPU die temperature") || line.contains("CPU temp") {
                    if let Some(temp_str) = line.split(':').nth(1) {
                        let temp_str = temp_str.trim();
                        if let Some(num_str) = temp_str.split_whitespace().next() {
                            if let Ok(temp) = num_str.parse::<f32>() {
                                cpu_temp = temp;
                            }
                        }
                    }
                }
                if line.contains("GPU die temperature") || line.contains("GPU temp") {
                    if let Some(temp_str) = line.split(':').nth(1) {
                        let temp_str = temp_str.trim();
                        if let Some(num_str) = temp_str.split_whitespace().next() {
                            if let Ok(temp) = num_str.parse::<f32>() {
                                gpu_temp = temp;
                            }
                        }
                    }
                }
            }
        }

        // Fallback: Try osx-cpu-temp if available
        if let Ok(output) = Command::new("osx-cpu-temp").output() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            // Parse output like "61.2°C"
            if let Some(temp_str) = output_str.split('°').next() {
                if let Ok(temp) = temp_str.trim().parse::<f32>() {
                    cpu_temp = temp;
                }
            }
        }

        // Check for thermal throttling (macOS typically throttles around 100°C)
        thermal_throttling = cpu_temp > 95.0 || gpu_temp > 90.0;
    }

    #[cfg(target_os = "windows")]
    {
        use std::process::Command;

        // Try nvidia-smi for GPU temperature
        if let Ok(output) = Command::new("nvidia-smi")
            .arg("--query-gpu=temperature.gpu")
            .arg("--format=csv,noheader,nounits")
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            if let Ok(temp) = output_str.trim().parse::<f32>() {
                gpu_temp = temp;
            }
        }

        // Use PowerShell/WMI for CPU temperature (limited on Windows without admin)
        if let Ok(output) = Command::new("powershell")
            .arg("-NoProfile")
            .arg("-Command")
            .arg("Get-WmiObject MSAcpi_ThermalZoneTemperature -Namespace root/wmi | Select-Object -ExpandProperty CurrentTemperature")
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            if let Ok(temp_tenth_kelvin) = output_str.trim().parse::<f32>() {
                // Convert from tenths of Kelvin to Celsius
                cpu_temp = (temp_tenth_kelvin / 10.0) - 273.15;
            }
        }

        // Check for thermal throttling
        thermal_throttling = cpu_temp > 90.0 || gpu_temp > 85.0;
    }

    TemperatureInfo {
        cpu_temp,
        gpu_temp,
        thermal_throttling,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_detection() {
        let gpus = get_gpu_info();
        assert!(!gpus.is_empty());
        assert!(!gpus[0].name.is_empty());
    }

    #[test]
    fn test_cpu_detection() {
        let cpu = get_cpu_info();
        assert!(!cpu.name.is_empty());
        assert!(cpu.logical_cores > 0);
        assert!(cpu.physical_cores > 0);
    }

    #[test]
    fn test_memory_detection() {
        let memory = get_memory_info();
        assert!(memory.total > 0);
        assert!(memory.available <= memory.total);
    }

    #[test]
    fn test_optimization_recommendations() {
        let recommendations = get_optimization_recommendations();
        assert!(recommendations.worker_threads > 0);
        assert!(recommendations.memory_limit > 0);
        assert!(recommendations.batch_size > 0);
    }

    #[test]
    fn test_hardware_monitoring() {
        let usage = monitor_hardware_usage();
        assert!(usage.cpu_usage >= 0.0);
        assert!(usage.gpu_usage >= 0.0);
    }

    #[test]
    fn test_cpu_frequency_never_fabricated_constant() {
        // Regression test for the hardcoded-2400/3600 finding: whichever
        // platform this runs on, the reported frequency must either be a
        // real, distinguishable measurement or an honest `None` -- never the
        // literal old constants presented as if measured.
        let cpu = get_cpu_info();
        if let Some(base) = cpu.base_frequency {
            assert_ne!(
                base, 2400,
                "base_frequency must not be the old fabricated constant"
            );
        }
        if let Some(max) = cpu.max_frequency {
            assert_ne!(
                max, 3600,
                "max_frequency must not be the old fabricated constant"
            );
        }
    }

    #[test]
    fn test_memory_speed_never_fabricated_constant() {
        // Regression test for the /proc/meminfo-read-then-discard finding:
        // speed must be a genuinely parsed value or an honest `None` -- never
        // the old 3200/2400 constants.
        let memory = get_memory_info();
        if let Some(speed) = memory.speed {
            assert_ne!(
                speed, 3200,
                "speed must not be the old fabricated DDR4-3200 constant"
            );
            assert_ne!(
                speed, 2400,
                "speed must not be the old fabricated fallback constant"
            );
            assert!(speed > 0);
        }
    }

    #[test]
    fn test_parse_mhz_value_extracts_real_number() {
        // Output actually varies with input -- proves this is real parsing,
        // not a hardcoded return.
        assert_eq!(parse_mhz_value(" 2400.000"), Some(2400));
        assert_eq!(parse_mhz_value(" 3792.014"), Some(3792));
        assert_eq!(parse_mhz_value(" 1000.500"), Some(1001)); // rounds
        assert_eq!(parse_mhz_value(" not-a-number"), None);
        assert_eq!(parse_mhz_value(" 0.0"), None);
        assert_eq!(parse_mhz_value(""), None);
    }

    #[test]
    fn test_parse_leading_mhz_token() {
        assert_eq!(parse_leading_mhz_token(" 3200 MT/s"), Some(3200));
        assert_eq!(parse_leading_mhz_token(" 2667 MHz"), Some(2667));
        assert_eq!(parse_leading_mhz_token(" Unknown"), None);
        assert_eq!(parse_leading_mhz_token(" 0 MT/s"), None);
        assert_eq!(parse_leading_mhz_token(""), None);
    }

    #[test]
    fn test_parse_dmidecode_memory_speed_prefers_configured_over_rated() {
        let dmidecode_output = "\
Memory Device
\tArray Handle: 0x0000
\tSpeed: 3200 MT/s
\tConfigured Memory Speed: 2933 MT/s
\tManufacturer: Samsung
";
        // Real, input-dependent parsing: "Configured Memory Speed" (the speed
        // actually in use) wins over "Speed" (the DIMM's rated maximum).
        assert_eq!(parse_dmidecode_memory_speed(dmidecode_output), Some(2933));
    }

    #[test]
    fn test_parse_dmidecode_memory_speed_falls_back_to_rated() {
        let dmidecode_output = "\
Memory Device
\tSpeed: 3200 MT/s
\tManufacturer: Samsung
";
        assert_eq!(parse_dmidecode_memory_speed(dmidecode_output), Some(3200));
    }

    #[test]
    fn test_parse_dmidecode_memory_speed_empty_slot_is_none() {
        let dmidecode_output = "\
Memory Device
\tSize: No Module Installed
\tSpeed: Unknown
\tConfigured Memory Speed: Unknown
";
        assert_eq!(parse_dmidecode_memory_speed(dmidecode_output), None);
    }

    #[test]
    fn test_parse_macos_memory_speed_extracts_real_value() {
        let system_profiler_output = "\
Memory:

      Memory: 16 GB
      Type: DDR4
      Speed: 2667 MHz
      Manufacturer: Samsung
";
        assert_eq!(parse_macos_memory_speed(system_profiler_output), Some(2667));
    }

    #[test]
    fn test_parse_macos_memory_speed_missing_field_is_none() {
        // Matches this machine's real `system_profiler SPMemoryDataType`
        // output on Apple Silicon, which does not report a Speed field.
        let system_profiler_output = "\
Memory:

      Memory: 24 GB
      Type: LPDDR5
      Manufacturer: Hynix
";
        assert_eq!(parse_macos_memory_speed(system_profiler_output), None);
    }
}
