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
    /// Base frequency in MHz
    pub base_frequency: u32,
    /// Maximum frequency in MHz
    pub max_frequency: u32,
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
    /// Memory speed in MHz
    pub speed: u32,
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
            base_frequency: 2400,
            max_frequency: 3600,
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
    // Windows GPU detection implementation
    GpuInfo {
        name: "Windows GPU".to_string(),
        vendor: "Unknown".to_string(),
        vram: 4_000_000_000, // 4GB placeholder
        cuda_support: false,
        opencl_support: false,
        vulkan_support: false,
    }
}

#[cfg(target_os = "windows")]
fn detect_windows_cpu() -> CpuInfo {
    // Windows CPU detection implementation
    CpuInfo {
        name: "Windows CPU".to_string(),
        vendor: "Unknown".to_string(),
        physical_cores: num_cpus::get_physical(),
        logical_cores: num_cpus::get(),
        base_frequency: 2400,
        max_frequency: 3600,
        cache_sizes: HashMap::new(),
        instruction_sets: vec!["SSE".to_string(), "AVX".to_string()],
    }
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
        // Parse system_profiler output for GPU info
        // This is a simplified implementation
        GpuInfo {
            name: "macOS GPU".to_string(),
            vendor: "Apple/AMD/Intel".to_string(),
            vram: 4_000_000_000,
            cuda_support: false,
            opencl_support: true,  // macOS supports OpenCL
            vulkan_support: false, // macOS deprecated OpenGL/Vulkan in favor of Metal
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
        base_frequency: 2400,
        max_frequency: 3600,
        cache_sizes: HashMap::new(),
        instruction_sets: Vec::new(),
    };

    // Get CPU name from sysctl
    if let Ok(output) = Command::new("sysctl")
        .arg("-n")
        .arg("machdep.cpu.brand_string")
        .output()
    {
        cpu_info.name = String::from_utf8_lossy(&output.stdout).trim().to_string();
    }

    // Get CPU vendor
    if let Ok(output) = Command::new("sysctl")
        .arg("-n")
        .arg("machdep.cpu.vendor")
        .output()
    {
        cpu_info.vendor = String::from_utf8_lossy(&output.stdout).trim().to_string();
    }

    cpu_info
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
        base_frequency: 2400,
        max_frequency: 3600,
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
            }
        }
    }

    cpu_info
}

#[cfg(target_os = "linux")]
fn parse_lspci_gpu_line(line: &str) -> GpuInfo {
    let name = line
        .split(':')
        .last()
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
    let name = parts.get(0).unwrap_or(&"NVIDIA GPU").trim().to_string();
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

fn detect_memory_speed() -> u32 {
    // Platform-specific memory speed detection
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(content) = fs::read_to_string("/proc/meminfo") {
            // Try to parse memory speed from dmidecode or other sources
            // This is a simplified implementation
            return 3200; // DDR4-3200 as default
        }
    }

    2400 // Default fallback
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

fn get_cpu_usage() -> f32 {
    // Platform-specific CPU usage monitoring
    // This is a simplified implementation
    0.0 // Placeholder
}

fn get_memory_usage() -> u64 {
    // Current memory usage
    let (total, available) = crate::platform::get_memory_info();
    total - available
}

fn get_gpu_usage() -> f32 {
    // Platform-specific GPU usage monitoring
    // This is a simplified implementation
    0.0 // Placeholder
}

fn get_temperature_info() -> TemperatureInfo {
    // Platform-specific temperature monitoring
    TemperatureInfo {
        cpu_temp: 45.0, // Placeholder
        gpu_temp: 50.0, // Placeholder
        thermal_throttling: false,
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
}
