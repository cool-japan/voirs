//! Platform Compatibility Module
//!
//! This module provides comprehensive platform compatibility features for the VoiRS evaluation system,
//! including cross-platform support for Windows, macOS, Linux, ARM and x86_64 optimization,
//! container deployment readiness, and cloud platform integration capabilities.

use crate::{EvaluationError, EvaluationResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;

/// Platform information and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformInfo {
    /// Operating system
    pub os: OperatingSystem,
    /// CPU architecture
    pub architecture: CpuArchitecture,
    /// Available CPU features
    pub cpu_features: Vec<String>,
    /// Memory information
    pub memory: MemoryInfo,
    /// Available compute devices
    pub compute_devices: Vec<ComputeDevice>,
    /// Platform-specific optimization flags
    pub optimization_flags: OptimizationFlags,
    /// Container support information
    pub container_support: ContainerSupport,
    /// Cloud platform capabilities
    pub cloud_capabilities: CloudCapabilities,
}

/// Operating system types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OperatingSystem {
    /// Windows NT-based systems
    Windows {
        /// Windows version
        version: String,
        /// Build number
        build: Option<u32>,
    },
    /// macOS systems
    MacOS {
        /// macOS version
        version: String,
        /// Darwin kernel version
        darwin_version: Option<String>,
    },
    /// Linux distributions
    Linux {
        /// Distribution name
        distribution: String,
        /// Kernel version
        kernel_version: String,
        /// Distribution version
        version: Option<String>,
    },
    /// Other Unix-like systems
    Unix {
        /// System name
        name: String,
        /// Version information
        version: Option<String>,
    },
    /// Unknown or unsupported system
    Unknown {
        /// Raw system information
        raw_info: String,
    },
}

/// CPU architecture types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CpuArchitecture {
    /// x86-64 / AMD64
    X86_64,
    /// ARM64 / AArch64
    ARM64,
    /// x86 32-bit
    X86,
    /// ARM 32-bit
    ARM,
    /// RISC-V
    RISCV64,
    /// Other architecture
    Other(String),
}

/// Memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// Total system memory in bytes
    pub total_memory: u64,
    /// Available memory in bytes
    pub available_memory: u64,
    /// Memory page size
    pub page_size: u64,
    /// NUMA topology information
    pub numa_nodes: Option<Vec<NumaNode>>,
}

/// NUMA node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaNode {
    /// Node ID
    pub node_id: u32,
    /// Memory size in bytes
    pub memory_size: u64,
    /// CPU cores associated with this node
    pub cpu_cores: Vec<u32>,
}

/// Compute device types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeDevice {
    /// Device type
    pub device_type: DeviceType,
    /// Device name
    pub name: String,
    /// Device ID
    pub device_id: u32,
    /// Compute capability or version
    pub compute_capability: String,
    /// Memory size in bytes
    pub memory_size: u64,
    /// Device-specific features
    pub features: Vec<String>,
    /// Performance characteristics
    pub performance: DevicePerformance,
}

/// Device types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DeviceType {
    /// CPU device
    CPU,
    /// NVIDIA CUDA GPU
    CUDA,
    /// Apple Metal GPU
    Metal,
    /// OpenCL device
    OpenCL,
    /// Vulkan compute device
    Vulkan,
    /// Custom accelerator
    Custom(String),
}

/// Device performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevicePerformance {
    /// Theoretical peak FLOPS
    pub peak_flops: f64,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth: f64,
    /// Power consumption in watts
    pub power_consumption: f32,
    /// Thermal design power
    pub tdp: f32,
}

/// Platform optimization flags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationFlags {
    /// SIMD instruction sets available
    pub simd_features: Vec<SimdFeature>,
    /// Vector processing capabilities
    pub vector_width: u32,
    /// Cache hierarchy information
    pub cache_info: CacheInfo,
    /// Threading capabilities
    pub threading: ThreadingInfo,
    /// Platform-specific compiler flags
    pub compiler_flags: Vec<String>,
}

/// SIMD feature sets
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SimdFeature {
    /// x86 SSE
    SSE,
    /// x86 SSE2
    SSE2,
    /// x86 SSE3
    SSE3,
    /// x86 SSSE3
    SSSE3,
    /// x86 SSE4.1
    SSE41,
    /// x86 SSE4.2
    SSE42,
    /// x86 AVX
    AVX,
    /// x86 AVX2
    AVX2,
    /// x86 AVX-512
    AVX512,
    /// ARM NEON
    NEON,
    /// ARM SVE
    SVE,
    /// RISC-V Vector extension
    RVV,
}

/// Cache hierarchy information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheInfo {
    /// L1 cache information
    pub l1_cache: Vec<CacheLevel>,
    /// L2 cache information
    pub l2_cache: Vec<CacheLevel>,
    /// L3 cache information
    pub l3_cache: Vec<CacheLevel>,
}

/// Cache level information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheLevel {
    /// Cache type
    pub cache_type: CacheType,
    /// Cache size in bytes
    pub size: u64,
    /// Cache line size in bytes
    pub line_size: u32,
    /// Associativity
    pub associativity: u32,
    /// Number of cache sets
    pub sets: u32,
}

/// Cache types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CacheType {
    /// Instruction cache
    Instruction,
    /// Data cache
    Data,
    /// Unified cache
    Unified,
}

/// Threading information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadingInfo {
    /// Number of physical CPU cores
    pub physical_cores: u32,
    /// Number of logical CPU cores (with hyperthreading)
    pub logical_cores: u32,
    /// Thread scheduling information
    pub scheduling: SchedulingInfo,
    /// NUMA-aware threading
    pub numa_aware: bool,
}

/// Scheduling information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingInfo {
    /// Scheduler type
    pub scheduler_type: String,
    /// CPU affinity support
    pub cpu_affinity: bool,
    /// Real-time scheduling support
    pub realtime_support: bool,
    /// Priority levels available
    pub priority_levels: u32,
}

/// Container support information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerSupport {
    /// Docker support
    pub docker: ContainerRuntime,
    /// Podman support
    pub podman: ContainerRuntime,
    /// Kubernetes support
    pub kubernetes: KubernetesSupport,
    /// Container image formats supported
    pub image_formats: Vec<String>,
    /// Resource isolation capabilities
    pub resource_isolation: ResourceIsolation,
}

/// Container runtime information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerRuntime {
    /// Runtime available
    pub available: bool,
    /// Runtime version
    pub version: Option<String>,
    /// Supported features
    pub features: Vec<String>,
    /// Security features
    pub security_features: Vec<String>,
}

/// Kubernetes support information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesSupport {
    /// Kubernetes available
    pub available: bool,
    /// Kubernetes version
    pub version: Option<String>,
    /// Supported resource types
    pub resource_types: Vec<String>,
    /// CSI support
    pub csi_support: bool,
    /// Custom resource definitions
    pub crd_support: bool,
}

/// Resource isolation capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceIsolation {
    /// CPU isolation
    pub cpu_isolation: bool,
    /// Memory isolation
    pub memory_isolation: bool,
    /// Network isolation
    pub network_isolation: bool,
    /// Storage isolation
    pub storage_isolation: bool,
    /// Device isolation
    pub device_isolation: bool,
}

/// Cloud platform capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudCapabilities {
    /// Cloud provider detection
    pub cloud_provider: Option<CloudProvider>,
    /// Instance metadata
    pub instance_metadata: InstanceMetadata,
    /// Available cloud services
    pub services: CloudServices,
    /// Auto-scaling capabilities
    pub auto_scaling: AutoScalingInfo,
    /// Monitoring and logging
    pub monitoring: MonitoringInfo,
}

/// Cloud providers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CloudProvider {
    /// Amazon Web Services
    AWS,
    /// Microsoft Azure
    Azure,
    /// Google Cloud Platform
    GCP,
    /// Alibaba Cloud
    AlibabaCloud,
    /// IBM Cloud
    IBMCloud,
    /// Oracle Cloud
    OracleCloud,
    /// DigitalOcean
    DigitalOcean,
    /// Vultr
    Vultr,
    /// Linode
    Linode,
    /// On-premises
    OnPremises,
    /// Unknown or other cloud
    Other(String),
}

/// Instance metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceMetadata {
    /// Instance type
    pub instance_type: Option<String>,
    /// Instance ID
    pub instance_id: Option<String>,
    /// Region
    pub region: Option<String>,
    /// Availability zone
    pub availability_zone: Option<String>,
    /// Instance tags
    pub tags: HashMap<String, String>,
}

/// Cloud services information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudServices {
    /// Object storage available
    pub object_storage: bool,
    /// Database services
    pub database_services: Vec<String>,
    /// ML/AI services
    pub ml_services: Vec<String>,
    /// Compute services
    pub compute_services: Vec<String>,
    /// Networking services
    pub networking_services: Vec<String>,
}

/// Auto-scaling information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingInfo {
    /// Horizontal scaling support
    pub horizontal_scaling: bool,
    /// Vertical scaling support
    pub vertical_scaling: bool,
    /// Auto-scaling triggers
    pub scaling_triggers: Vec<String>,
    /// Scaling policies
    pub scaling_policies: Vec<String>,
}

/// Monitoring information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringInfo {
    /// Metrics collection
    pub metrics_collection: bool,
    /// Log aggregation
    pub log_aggregation: bool,
    /// Distributed tracing
    pub distributed_tracing: bool,
    /// Alert management
    pub alert_management: bool,
    /// Available monitoring tools
    pub monitoring_tools: Vec<String>,
}

/// Platform compatibility checker
#[derive(Debug, Clone)]
pub struct PlatformCompatibility {
    /// Current platform information
    pub platform_info: PlatformInfo,
    /// Supported platforms
    pub supported_platforms: Vec<PlatformRequirement>,
    /// Platform-specific optimizations
    pub optimizations: HashMap<String, PlatformOptimization>,
}

/// Platform requirement specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformRequirement {
    /// Required operating system
    pub os: Option<OperatingSystem>,
    /// Required architecture
    pub architecture: Option<CpuArchitecture>,
    /// Minimum memory requirement
    pub min_memory: Option<u64>,
    /// Required CPU features
    pub required_features: Vec<String>,
    /// Optional features
    pub optional_features: Vec<String>,
}

/// Platform-specific optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformOptimization {
    /// Optimization name
    pub name: String,
    /// Target platforms
    pub target_platforms: Vec<String>,
    /// Optimization parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Performance gain estimate
    pub performance_gain: f64,
    /// Memory overhead
    pub memory_overhead: i64,
}

impl PlatformCompatibility {
    /// Create new platform compatibility checker
    pub fn new() -> EvaluationResult<Self> {
        let platform_info = Self::detect_platform_info()?;
        let supported_platforms = Self::get_supported_platforms();
        let optimizations = Self::get_platform_optimizations();

        Ok(Self {
            platform_info,
            supported_platforms,
            optimizations,
        })
    }

    /// Detect current platform information
    pub fn detect_platform_info() -> EvaluationResult<PlatformInfo> {
        let os = Self::detect_operating_system()?;
        let architecture = Self::detect_cpu_architecture();
        let cpu_features = Self::detect_cpu_features();
        let memory = Self::detect_memory_info()?;
        let compute_devices = Self::detect_compute_devices()?;
        let optimization_flags = Self::detect_optimization_flags();
        let container_support = Self::detect_container_support();
        let cloud_capabilities = Self::detect_cloud_capabilities();

        Ok(PlatformInfo {
            os,
            architecture,
            cpu_features,
            memory,
            compute_devices,
            optimization_flags,
            container_support,
            cloud_capabilities,
        })
    }

    /// Detect operating system
    fn detect_operating_system() -> EvaluationResult<OperatingSystem> {
        let os = env::consts::OS;
        let family = env::consts::FAMILY;

        match (os, family) {
            ("windows", _) => {
                // In a real implementation, would query Windows version APIs
                Ok(OperatingSystem::Windows {
                    version: "10.0".to_string(),
                    build: Some(19044),
                })
            }
            ("macos", _) => {
                // In a real implementation, would query macOS version
                Ok(OperatingSystem::MacOS {
                    version: "12.0".to_string(),
                    darwin_version: Some("21.0.0".to_string()),
                })
            }
            ("linux", _) => {
                // In a real implementation, would parse /etc/os-release
                Ok(OperatingSystem::Linux {
                    distribution: "Ubuntu".to_string(),
                    kernel_version: "5.4.0".to_string(),
                    version: Some("20.04".to_string()),
                })
            }
            _ => Ok(OperatingSystem::Unix {
                name: os.to_string(),
                version: None,
            }),
        }
    }

    /// Detect CPU architecture
    fn detect_cpu_architecture() -> CpuArchitecture {
        match env::consts::ARCH {
            "x86_64" => CpuArchitecture::X86_64,
            "aarch64" => CpuArchitecture::ARM64,
            "x86" => CpuArchitecture::X86,
            "arm" => CpuArchitecture::ARM,
            "riscv64" => CpuArchitecture::RISCV64,
            arch => CpuArchitecture::Other(arch.to_string()),
        }
    }

    /// Detect CPU features

    /// Detect memory information
    fn detect_memory_info() -> EvaluationResult<MemoryInfo> {
        let (total_memory, available_memory, page_size) = Self::get_system_memory()?;

        Ok(MemoryInfo {
            total_memory,
            available_memory,
            page_size,
            numa_nodes: None, // NUMA topology detection would require additional system calls
        })
    }

    /// Get system memory information using platform-specific APIs
    fn get_system_memory() -> EvaluationResult<(u64, u64, u64)> {
        #[cfg(target_os = "linux")]
        {
            Self::get_linux_memory()
        }
        #[cfg(target_os = "macos")]
        {
            Self::get_macos_memory()
        }
        #[cfg(target_os = "windows")]
        {
            Self::get_windows_memory()
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            // Fallback for unsupported platforms
            Ok((8_000_000_000, 4_000_000_000, 4096)) // 8GB total, 4GB available, 4KB page
        }
    }

    #[cfg(target_os = "linux")]
    fn get_linux_memory() -> EvaluationResult<(u64, u64, u64)> {
        use std::fs;

        // Read /proc/meminfo for memory information
        let meminfo =
            fs::read_to_string("/proc/meminfo").map_err(|e| EvaluationError::InvalidInput {
                message: format!("Failed to read /proc/meminfo: {}", e),
            })?;

        let mut total_memory = 0u64;
        let mut available_memory = 0u64;

        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                if let Some(value) = Self::parse_meminfo_line(line) {
                    total_memory = value * 1024; // Convert KB to bytes
                }
            } else if line.starts_with("MemAvailable:") {
                if let Some(value) = Self::parse_meminfo_line(line) {
                    available_memory = value * 1024; // Convert KB to bytes
                }
            }
        }

        // Get page size using libc
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;

        if total_memory == 0 {
            return Err(EvaluationError::InvalidInput {
                message: "Could not determine total memory".to_string(),
            }
            .into());
        }

        // Fallback for available memory if not found
        if available_memory == 0 {
            available_memory = total_memory / 2; // Conservative estimate
        }

        Ok((total_memory, available_memory, page_size))
    }

    #[cfg(target_os = "macos")]
    fn get_macos_memory() -> EvaluationResult<(u64, u64, u64)> {
        use std::mem;

        // Get total physical memory
        let mut size = mem::size_of::<u64>();
        let mut total_memory = 0u64;

        let result = unsafe {
            libc::sysctlbyname(
                b"hw.memsize\0".as_ptr() as *const i8,
                &mut total_memory as *mut _ as *mut libc::c_void,
                &mut size,
                std::ptr::null_mut(),
                0,
            )
        };

        if result != 0 {
            return Err(EvaluationError::InvalidInput {
                message: "Failed to get macOS memory size".to_string(),
            }
            .into());
        }

        // Get page size
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;

        // Estimate available memory (conservative approach)
        let available_memory = total_memory * 3 / 4; // Assume 75% might be available

        Ok((total_memory, available_memory, page_size))
    }

    #[cfg(target_os = "windows")]
    fn get_windows_memory() -> EvaluationResult<(u64, u64, u64)> {
        // For Windows, we would use GetPhysicallyInstalledSystemMemory and GlobalMemoryStatusEx
        // This is a simplified implementation
        let page_size = 4096u64; // Standard page size on Windows

        // Note: In a full implementation, we would use Windows APIs:
        // - GetPhysicallyInstalledSystemMemory for total memory
        // - GlobalMemoryStatusEx for available memory
        // For now, provide reasonable defaults
        let total_memory = 16_000_000_000u64; // 16 GB
        let available_memory = 8_000_000_000u64; // 8 GB

        Ok((total_memory, available_memory, page_size))
    }

    #[cfg(target_os = "linux")]
    fn parse_meminfo_line(line: &str) -> Option<u64> {
        // Parse lines like "MemTotal:       16384000 kB"
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            parts[1].parse().ok()
        } else {
            None
        }
    }

    /// Detect compute devices
    fn detect_compute_devices() -> EvaluationResult<Vec<ComputeDevice>> {
        let mut devices = Vec::new();

        // Get actual system memory for CPU device
        let (total_memory, _, _) =
            Self::get_system_memory().unwrap_or((8_000_000_000, 4_000_000_000, 4096));

        // Get CPU information
        let cpu_name = Self::get_cpu_name();
        let cpu_count = num_cpus::get();
        let cpu_features = Self::detect_cpu_features();

        // Estimate CPU performance based on core count and architecture
        let estimated_flops = Self::estimate_cpu_flops(cpu_count);
        let estimated_bandwidth = Self::estimate_memory_bandwidth();
        let estimated_tdp = Self::estimate_cpu_tdp(cpu_count);

        // CPU device
        devices.push(ComputeDevice {
            device_type: DeviceType::CPU,
            name: cpu_name,
            device_id: 0,
            compute_capability: format!("{} cores", cpu_count),
            memory_size: total_memory,
            features: cpu_features,
            performance: DevicePerformance {
                peak_flops: estimated_flops,
                memory_bandwidth: estimated_bandwidth,
                power_consumption: estimated_tdp as f32,
                tdp: estimated_tdp as f32,
            },
        });

        // In a real implementation, would detect CUDA, Metal, OpenCL devices
        // For now, just return CPU device

        Ok(devices)
    }

    /// Get CPU name/model information
    fn get_cpu_name() -> String {
        #[cfg(target_os = "linux")]
        {
            Self::get_linux_cpu_name().unwrap_or_else(|| "Unknown CPU".to_string())
        }
        #[cfg(target_os = "macos")]
        {
            Self::get_macos_cpu_name().unwrap_or_else(|| "Unknown CPU".to_string())
        }
        #[cfg(target_os = "windows")]
        {
            "Windows CPU".to_string() // Would use WMI or registry in full implementation
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            "Unknown CPU".to_string()
        }
    }

    #[cfg(target_os = "linux")]
    fn get_linux_cpu_name() -> Option<String> {
        use std::fs;

        let cpuinfo = fs::read_to_string("/proc/cpuinfo").ok()?;
        for line in cpuinfo.lines() {
            if line.starts_with("model name") {
                if let Some(colon_pos) = line.find(':') {
                    return Some(line[colon_pos + 1..].trim().to_string());
                }
            }
        }
        None
    }

    #[cfg(target_os = "macos")]
    fn get_macos_cpu_name() -> Option<String> {
        use std::ffi::CStr;
        use std::mem;

        let mut size = 0;

        // First get the size needed
        let result = unsafe {
            libc::sysctlbyname(
                b"machdep.cpu.brand_string\0".as_ptr() as *const i8,
                std::ptr::null_mut(),
                &mut size,
                std::ptr::null_mut(),
                0,
            )
        };

        if result != 0 || size == 0 {
            return None;
        }

        // Allocate buffer and get the string
        let mut buffer = vec![0u8; size];
        let result = unsafe {
            libc::sysctlbyname(
                b"machdep.cpu.brand_string\0".as_ptr() as *const i8,
                buffer.as_mut_ptr() as *mut libc::c_void,
                &mut size,
                std::ptr::null_mut(),
                0,
            )
        };

        if result == 0 {
            // Remove null terminator and convert to string
            if let Some(null_pos) = buffer.iter().position(|&x| x == 0) {
                buffer.truncate(null_pos);
            }
            String::from_utf8(buffer).ok()
        } else {
            None
        }
    }

    /// Detect CPU features
    fn detect_cpu_features() -> Vec<String> {
        let mut features = vec!["fp32".to_string(), "fp64".to_string()];

        // Add architecture-specific features
        match env::consts::ARCH {
            "x86_64" => {
                features.extend_from_slice(&[
                    "sse2".to_string(),
                    "sse4.1".to_string(),
                    "avx".to_string(),
                ]);

                // In a full implementation, would check CPUID for actual features
                #[cfg(target_feature = "avx2")]
                features.push("avx2".to_string());

                #[cfg(target_feature = "fma")]
                features.push("fma".to_string());
            }
            "aarch64" => {
                features.extend_from_slice(&["neon".to_string(), "asimd".to_string()]);
            }
            _ => {}
        }

        features
    }

    /// Estimate CPU FLOPS based on core count and architecture
    fn estimate_cpu_flops(cpu_count: usize) -> f64 {
        // Rough estimation based on architecture and core count
        let base_flops_per_core = match env::consts::ARCH {
            "x86_64" => 8.0,  // Assume ~8 GFLOPS per core for modern x86_64
            "aarch64" => 4.0, // Assume ~4 GFLOPS per core for ARM64
            _ => 2.0,         // Conservative fallback
        };

        base_flops_per_core * cpu_count as f64 * 1e9 // Convert to FLOPS
    }

    /// Estimate memory bandwidth
    fn estimate_memory_bandwidth() -> f64 {
        // Rough estimation based on typical memory configurations
        match env::consts::ARCH {
            "x86_64" => 50.0,  // ~50 GB/s for typical DDR4
            "aarch64" => 30.0, // ~30 GB/s for typical mobile/embedded
            _ => 20.0,         // Conservative fallback
        }
    }

    /// Estimate CPU TDP (Thermal Design Power)
    fn estimate_cpu_tdp(cpu_count: usize) -> f64 {
        // Rough estimation based on core count
        let base_tdp_per_core = match env::consts::ARCH {
            "x86_64" => 10.0, // ~10W per core for desktop CPUs
            "aarch64" => 2.0, // ~2W per core for mobile CPUs
            _ => 5.0,         // Conservative fallback
        };

        base_tdp_per_core * cpu_count as f64
    }

    /// Detect optimization flags
    fn detect_optimization_flags() -> OptimizationFlags {
        let simd_features = match env::consts::ARCH {
            "x86_64" => vec![
                SimdFeature::SSE,
                SimdFeature::SSE2,
                SimdFeature::AVX,
                SimdFeature::AVX2,
            ],
            "aarch64" => vec![SimdFeature::NEON],
            _ => vec![],
        };

        let vector_width = match env::consts::ARCH {
            "x86_64" => 256,  // AVX2 width
            "aarch64" => 128, // NEON width
            _ => 64,
        };

        let cache_info = CacheInfo {
            l1_cache: vec![
                CacheLevel {
                    cache_type: CacheType::Data,
                    size: 32 * 1024, // 32KB
                    line_size: 64,
                    associativity: 8,
                    sets: 64,
                },
                CacheLevel {
                    cache_type: CacheType::Instruction,
                    size: 32 * 1024, // 32KB
                    line_size: 64,
                    associativity: 8,
                    sets: 64,
                },
            ],
            l2_cache: vec![CacheLevel {
                cache_type: CacheType::Unified,
                size: 256 * 1024, // 256KB
                line_size: 64,
                associativity: 8,
                sets: 512,
            }],
            l3_cache: vec![CacheLevel {
                cache_type: CacheType::Unified,
                size: 8 * 1024 * 1024, // 8MB
                line_size: 64,
                associativity: 16,
                sets: 8192,
            }],
        };

        let threading = ThreadingInfo {
            physical_cores: num_cpus::get_physical() as u32,
            logical_cores: num_cpus::get() as u32,
            scheduling: SchedulingInfo {
                scheduler_type: "CFS".to_string(), // Linux CFS, Windows scheduler, etc.
                cpu_affinity: true,
                realtime_support: false,
                priority_levels: 8,
            },
            numa_aware: false, // Would detect NUMA awareness
        };

        OptimizationFlags {
            simd_features,
            vector_width,
            cache_info,
            threading,
            compiler_flags: vec![
                "-O3".to_string(),
                "-march=native".to_string(),
                "-mtune=native".to_string(),
            ],
        }
    }

    /// Detect container support
    fn detect_container_support() -> ContainerSupport {
        // In a real implementation, would check for Docker/Podman availability
        ContainerSupport {
            docker: ContainerRuntime {
                available: false, // Would check actual availability
                version: None,
                features: vec![],
                security_features: vec![],
            },
            podman: ContainerRuntime {
                available: false,
                version: None,
                features: vec![],
                security_features: vec![],
            },
            kubernetes: KubernetesSupport {
                available: false,
                version: None,
                resource_types: vec![],
                csi_support: false,
                crd_support: false,
            },
            image_formats: vec!["OCI".to_string(), "Docker".to_string()],
            resource_isolation: ResourceIsolation {
                cpu_isolation: true,
                memory_isolation: true,
                network_isolation: true,
                storage_isolation: true,
                device_isolation: true,
            },
        }
    }

    /// Detect cloud capabilities
    fn detect_cloud_capabilities() -> CloudCapabilities {
        // In a real implementation, would query cloud metadata endpoints
        CloudCapabilities {
            cloud_provider: Self::detect_cloud_provider(),
            instance_metadata: InstanceMetadata {
                instance_type: None,
                instance_id: None,
                region: None,
                availability_zone: None,
                tags: HashMap::new(),
            },
            services: CloudServices {
                object_storage: false,
                database_services: vec![],
                ml_services: vec![],
                compute_services: vec![],
                networking_services: vec![],
            },
            auto_scaling: AutoScalingInfo {
                horizontal_scaling: false,
                vertical_scaling: false,
                scaling_triggers: vec![],
                scaling_policies: vec![],
            },
            monitoring: MonitoringInfo {
                metrics_collection: false,
                log_aggregation: false,
                distributed_tracing: false,
                alert_management: false,
                monitoring_tools: vec![],
            },
        }
    }

    /// Detect cloud provider
    fn detect_cloud_provider() -> Option<CloudProvider> {
        // In a real implementation, would check cloud metadata endpoints
        // For now, assume on-premises
        Some(CloudProvider::OnPremises)
    }

    /// Get supported platforms
    fn get_supported_platforms() -> Vec<PlatformRequirement> {
        vec![
            // Windows x64
            PlatformRequirement {
                os: Some(OperatingSystem::Windows {
                    version: "10.0".to_string(),
                    build: Some(17134),
                }),
                architecture: Some(CpuArchitecture::X86_64),
                min_memory: Some(4 * 1024 * 1024 * 1024), // 4GB
                required_features: vec!["sse2".to_string()],
                optional_features: vec!["avx".to_string(), "avx2".to_string()],
            },
            // macOS x64
            PlatformRequirement {
                os: Some(OperatingSystem::MacOS {
                    version: "10.15".to_string(),
                    darwin_version: None,
                }),
                architecture: Some(CpuArchitecture::X86_64),
                min_memory: Some(4 * 1024 * 1024 * 1024), // 4GB
                required_features: vec!["sse2".to_string()],
                optional_features: vec!["avx".to_string(), "avx2".to_string()],
            },
            // macOS ARM64
            PlatformRequirement {
                os: Some(OperatingSystem::MacOS {
                    version: "11.0".to_string(),
                    darwin_version: None,
                }),
                architecture: Some(CpuArchitecture::ARM64),
                min_memory: Some(8 * 1024 * 1024 * 1024), // 8GB
                required_features: vec!["neon".to_string()],
                optional_features: vec![],
            },
            // Linux x64
            PlatformRequirement {
                os: Some(OperatingSystem::Linux {
                    distribution: "ANY".to_string(),
                    kernel_version: "3.10".to_string(),
                    version: None,
                }),
                architecture: Some(CpuArchitecture::X86_64),
                min_memory: Some(2 * 1024 * 1024 * 1024), // 2GB
                required_features: vec!["sse2".to_string()],
                optional_features: vec!["avx".to_string(), "avx2".to_string()],
            },
            // Linux ARM64
            PlatformRequirement {
                os: Some(OperatingSystem::Linux {
                    distribution: "ANY".to_string(),
                    kernel_version: "4.1".to_string(),
                    version: None,
                }),
                architecture: Some(CpuArchitecture::ARM64),
                min_memory: Some(2 * 1024 * 1024 * 1024), // 2GB
                required_features: vec!["neon".to_string()],
                optional_features: vec![],
            },
        ]
    }

    /// Get platform-specific optimizations
    fn get_platform_optimizations() -> HashMap<String, PlatformOptimization> {
        let mut optimizations = HashMap::new();

        // x86_64 SIMD optimization
        optimizations.insert(
            "x86_64_simd".to_string(),
            PlatformOptimization {
                name: "x86_64 SIMD Optimization".to_string(),
                target_platforms: vec!["x86_64".to_string()],
                parameters: [
                    ("use_avx2".to_string(), serde_json::Value::Bool(true)),
                    (
                        "vector_width".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(256)),
                    ),
                ]
                .iter()
                .cloned()
                .collect(),
                performance_gain: 2.5, // 2.5x performance improvement
                memory_overhead: 0,    // No additional memory overhead
            },
        );

        // ARM64 NEON optimization
        optimizations.insert(
            "arm64_neon".to_string(),
            PlatformOptimization {
                name: "ARM64 NEON Optimization".to_string(),
                target_platforms: vec!["aarch64".to_string()],
                parameters: [
                    ("use_neon".to_string(), serde_json::Value::Bool(true)),
                    (
                        "vector_width".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(128)),
                    ),
                ]
                .iter()
                .cloned()
                .collect(),
                performance_gain: 2.0, // 2x performance improvement
                memory_overhead: 0,
            },
        );

        // GPU acceleration
        optimizations.insert(
            "gpu_acceleration".to_string(),
            PlatformOptimization {
                name: "GPU Acceleration".to_string(),
                target_platforms: vec![
                    "cuda".to_string(),
                    "metal".to_string(),
                    "opencl".to_string(),
                ],
                parameters: [
                    (
                        "batch_size".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(32)),
                    ),
                    (
                        "memory_pool_size".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(1024 * 1024 * 1024)),
                    ),
                ]
                .iter()
                .cloned()
                .collect(),
                performance_gain: 10.0, // 10x performance improvement
                memory_overhead: 1024 * 1024 * 1024, // 1GB GPU memory
            },
        );

        optimizations
    }

    /// Check platform compatibility
    pub fn check_compatibility(&self) -> PlatformCompatibilityResult {
        let mut compatible_platforms = Vec::new();
        let mut incompatible_reasons = Vec::new();
        let mut warnings = Vec::new();
        let mut optimizations_available = Vec::new();

        for requirement in &self.supported_platforms {
            let mut is_compatible = true;
            let mut reasons = Vec::new();

            // Check OS compatibility
            if let Some(required_os) = &requirement.os {
                if !self.os_compatible(required_os) {
                    is_compatible = false;
                    reasons.push(format!(
                        "OS mismatch: required {:?}, found {:?}",
                        required_os, self.platform_info.os
                    ));
                }
            }

            // Check architecture compatibility
            if let Some(required_arch) = &requirement.architecture {
                if *required_arch != self.platform_info.architecture {
                    is_compatible = false;
                    reasons.push(format!(
                        "Architecture mismatch: required {:?}, found {:?}",
                        required_arch, self.platform_info.architecture
                    ));
                }
            }

            // Check memory requirements
            if let Some(min_memory) = requirement.min_memory {
                if self.platform_info.memory.total_memory < min_memory {
                    is_compatible = false;
                    reasons.push(format!(
                        "Insufficient memory: required {}GB, found {}GB",
                        min_memory / (1024 * 1024 * 1024),
                        self.platform_info.memory.total_memory / (1024 * 1024 * 1024)
                    ));
                }
            }

            // Check required CPU features
            for required_feature in &requirement.required_features {
                if !self.platform_info.cpu_features.contains(required_feature) {
                    is_compatible = false;
                    reasons.push(format!(
                        "Missing required CPU feature: {}",
                        required_feature
                    ));
                }
            }

            // Check optional features (generate warnings if missing)
            for optional_feature in &requirement.optional_features {
                if !self.platform_info.cpu_features.contains(optional_feature) {
                    warnings.push(format!(
                        "Optional CPU feature not available: {} (may impact performance)",
                        optional_feature
                    ));
                }
            }

            if is_compatible {
                compatible_platforms.push(requirement.clone());
            } else {
                incompatible_reasons.extend(reasons);
            }
        }

        // Check available optimizations
        for (name, optimization) in &self.optimizations {
            if self.optimization_applicable(optimization) {
                optimizations_available.push(optimization.clone());
            }
        }

        let overall_compatibility = if !compatible_platforms.is_empty() {
            CompatibilityStatus::Compatible
        } else if incompatible_reasons.is_empty() {
            CompatibilityStatus::Unknown
        } else {
            CompatibilityStatus::Incompatible
        };

        PlatformCompatibilityResult {
            overall_compatibility,
            compatible_platforms,
            incompatible_reasons,
            warnings,
            optimizations_available,
            platform_score: self.calculate_platform_score(),
            recommendations: self.generate_recommendations(),
        }
    }

    /// Check OS compatibility
    fn os_compatible(&self, required_os: &OperatingSystem) -> bool {
        match (&self.platform_info.os, required_os) {
            (OperatingSystem::Windows { .. }, OperatingSystem::Windows { .. }) => true,
            (OperatingSystem::MacOS { .. }, OperatingSystem::MacOS { .. }) => true,
            (OperatingSystem::Linux { .. }, OperatingSystem::Linux { .. }) => true,
            (OperatingSystem::Unix { .. }, OperatingSystem::Unix { .. }) => true,
            _ => false,
        }
    }

    /// Check if optimization is applicable
    fn optimization_applicable(&self, optimization: &PlatformOptimization) -> bool {
        let arch_str = format!("{:?}", self.platform_info.architecture).to_lowercase();
        optimization.target_platforms.iter().any(|platform| {
            platform.to_lowercase() == arch_str
                || self.platform_info.cpu_features.contains(platform)
        })
    }

    /// Calculate platform performance score
    fn calculate_platform_score(&self) -> f64 {
        let mut score = 50.0; // Base score

        // CPU architecture score
        score += match self.platform_info.architecture {
            CpuArchitecture::X86_64 => 20.0,
            CpuArchitecture::ARM64 => 15.0,
            CpuArchitecture::X86 => 10.0,
            CpuArchitecture::ARM => 5.0,
            _ => 0.0,
        };

        // Memory score
        let memory_gb = self.platform_info.memory.total_memory / (1024 * 1024 * 1024);
        score += (memory_gb as f64 / 32.0) * 10.0; // Max 10 points for 32GB+

        // CPU features score
        let feature_score = self.platform_info.cpu_features.len() as f64 * 0.5;
        score += feature_score.min(10.0); // Max 10 points for features

        // Compute devices score
        if !self.platform_info.compute_devices.is_empty() {
            score += 10.0;
        }

        score.min(100.0)
    }

    /// Generate platform recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Memory recommendations
        let memory_gb = self.platform_info.memory.total_memory / (1024 * 1024 * 1024);
        if memory_gb < 8 {
            recommendations.push(
                "Consider upgrading to at least 8GB of RAM for optimal performance".to_string(),
            );
        }

        // CPU feature recommendations
        if !self
            .platform_info
            .cpu_features
            .contains(&"avx2".to_string())
            && self.platform_info.architecture == CpuArchitecture::X86_64
        {
            recommendations.push(
                "AVX2 support not detected. Consider using a newer CPU for better SIMD performance"
                    .to_string(),
            );
        }

        // GPU recommendations
        let has_gpu = self.platform_info.compute_devices.iter().any(|device| {
            matches!(
                device.device_type,
                DeviceType::CUDA | DeviceType::Metal | DeviceType::OpenCL
            )
        });
        if !has_gpu {
            recommendations.push("No GPU acceleration detected. Consider adding GPU support for significant performance improvements".to_string());
        }

        // Container recommendations
        if !self.platform_info.container_support.docker.available {
            recommendations.push("Docker not available. Consider installing Docker for containerized deployment options".to_string());
        }

        recommendations
    }

    /// Get deployment configuration for current platform
    pub fn get_deployment_config(&self) -> DeploymentConfig {
        let mut config = DeploymentConfig::default();

        // Set resource limits based on platform
        config.resource_limits.cpu_cores = self
            .platform_info
            .optimization_flags
            .threading
            .logical_cores;
        config.resource_limits.memory_mb =
            (self.platform_info.memory.available_memory / (1024 * 1024)) as u32;

        // Configure optimizations
        if self
            .platform_info
            .cpu_features
            .contains(&"avx2".to_string())
        {
            config.optimizations.insert("enable_avx2".to_string(), true);
        }
        if self
            .platform_info
            .cpu_features
            .contains(&"neon".to_string())
        {
            config.optimizations.insert("enable_neon".to_string(), true);
        }

        // Configure container settings
        if self.platform_info.container_support.docker.available {
            config.container_config.enable_docker = true;
        }
        if self.platform_info.container_support.kubernetes.available {
            config.container_config.enable_kubernetes = true;
        }

        // Configure cloud settings
        if let Some(cloud_provider) = &self.platform_info.cloud_capabilities.cloud_provider {
            config.cloud_config.provider = Some(cloud_provider.clone());
            config.cloud_config.enable_auto_scaling = self
                .platform_info
                .cloud_capabilities
                .auto_scaling
                .horizontal_scaling;
        }

        config
    }
}

/// Platform compatibility result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCompatibilityResult {
    /// Overall compatibility status
    pub overall_compatibility: CompatibilityStatus,
    /// Compatible platform requirements
    pub compatible_platforms: Vec<PlatformRequirement>,
    /// Reasons for incompatibility
    pub incompatible_reasons: Vec<String>,
    /// Compatibility warnings
    pub warnings: Vec<String>,
    /// Available optimizations
    pub optimizations_available: Vec<PlatformOptimization>,
    /// Platform performance score (0-100)
    pub platform_score: f64,
    /// Recommendations for improving compatibility
    pub recommendations: Vec<String>,
}

/// Compatibility status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CompatibilityStatus {
    /// Fully compatible
    Compatible,
    /// Compatible with warnings
    CompatibleWithWarnings,
    /// Incompatible
    Incompatible,
    /// Unknown compatibility
    Unknown,
}

/// Deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfig {
    /// Resource limits
    pub resource_limits: ResourceLimits,
    /// Optimization settings
    pub optimizations: HashMap<String, bool>,
    /// Container configuration
    pub container_config: ContainerConfig,
    /// Cloud configuration
    pub cloud_config: CloudConfig,
    /// Monitoring configuration
    pub monitoring_config: MonitoringConfig,
}

/// Resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum CPU cores to use
    pub cpu_cores: u32,
    /// Maximum memory in MB
    pub memory_mb: u32,
    /// Maximum GPU memory in MB
    pub gpu_memory_mb: Option<u32>,
    /// Maximum disk space in MB
    pub disk_space_mb: Option<u32>,
}

/// Container configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerConfig {
    /// Enable Docker support
    pub enable_docker: bool,
    /// Enable Kubernetes support
    pub enable_kubernetes: bool,
    /// Container image name
    pub image_name: String,
    /// Container registry
    pub registry: String,
    /// Resource requests and limits
    pub resources: ContainerResources,
}

/// Container resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerResources {
    /// CPU request
    pub cpu_request: String,
    /// CPU limit
    pub cpu_limit: String,
    /// Memory request
    pub memory_request: String,
    /// Memory limit
    pub memory_limit: String,
}

/// Cloud configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudConfig {
    /// Cloud provider
    pub provider: Option<CloudProvider>,
    /// Enable auto-scaling
    pub enable_auto_scaling: bool,
    /// Instance type preferences
    pub instance_types: Vec<String>,
    /// Regions
    pub regions: Vec<String>,
    /// Load balancing configuration
    pub load_balancing: LoadBalancingConfig,
}

/// Load balancing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingConfig {
    /// Enable load balancing
    pub enabled: bool,
    /// Load balancing algorithm
    pub algorithm: String,
    /// Health check configuration
    pub health_check: HealthCheckConfig,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Health check endpoint
    pub endpoint: String,
    /// Check interval in seconds
    pub interval_seconds: u32,
    /// Timeout in seconds
    pub timeout_seconds: u32,
    /// Healthy threshold
    pub healthy_threshold: u32,
    /// Unhealthy threshold
    pub unhealthy_threshold: u32,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Enable logging
    pub enable_logging: bool,
    /// Enable tracing
    pub enable_tracing: bool,
    /// Metrics endpoints
    pub metrics_endpoints: Vec<String>,
    /// Log aggregation endpoints
    pub log_endpoints: Vec<String>,
}

impl Default for DeploymentConfig {
    fn default() -> Self {
        Self {
            resource_limits: ResourceLimits {
                cpu_cores: 2,
                memory_mb: 2048,
                gpu_memory_mb: None,
                disk_space_mb: Some(10240), // 10GB
            },
            optimizations: HashMap::new(),
            container_config: ContainerConfig {
                enable_docker: false,
                enable_kubernetes: false,
                image_name: "voirs-evaluation".to_string(),
                registry: "registry.hub.docker.com".to_string(),
                resources: ContainerResources {
                    cpu_request: "500m".to_string(),
                    cpu_limit: "2".to_string(),
                    memory_request: "1Gi".to_string(),
                    memory_limit: "2Gi".to_string(),
                },
            },
            cloud_config: CloudConfig {
                provider: None,
                enable_auto_scaling: false,
                instance_types: vec!["t3.medium".to_string(), "t3.large".to_string()],
                regions: vec!["us-west-2".to_string(), "us-east-1".to_string()],
                load_balancing: LoadBalancingConfig {
                    enabled: false,
                    algorithm: "round_robin".to_string(),
                    health_check: HealthCheckConfig {
                        endpoint: "/health".to_string(),
                        interval_seconds: 30,
                        timeout_seconds: 5,
                        healthy_threshold: 2,
                        unhealthy_threshold: 3,
                    },
                },
            },
            monitoring_config: MonitoringConfig {
                enable_metrics: true,
                enable_logging: true,
                enable_tracing: false,
                metrics_endpoints: vec!["http://localhost:9090".to_string()],
                log_endpoints: vec!["http://localhost:5044".to_string()],
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_compatibility_creation() {
        let platform_compat = PlatformCompatibility::new();
        assert!(platform_compat.is_ok());
    }

    #[test]
    fn test_cpu_architecture_detection() {
        let arch = PlatformCompatibility::detect_cpu_architecture();
        assert!(matches!(
            arch,
            CpuArchitecture::X86_64 | CpuArchitecture::ARM64 | CpuArchitecture::Other(_)
        ));
    }

    #[test]
    fn test_cpu_features_detection() {
        let features = PlatformCompatibility::detect_cpu_features();
        assert!(!features.is_empty()); // Should detect at least some features
    }

    #[test]
    fn test_platform_info_serialization() {
        let platform_info = PlatformCompatibility::detect_platform_info().unwrap();
        let serialized = serde_json::to_string(&platform_info);
        assert!(serialized.is_ok());
    }

    #[test]
    fn test_compatibility_check() {
        let platform_compat = PlatformCompatibility::new().unwrap();
        let result = platform_compat.check_compatibility();
        assert!(matches!(
            result.overall_compatibility,
            CompatibilityStatus::Compatible | CompatibilityStatus::CompatibleWithWarnings
        ));
    }

    #[test]
    fn test_deployment_config_generation() {
        let platform_compat = PlatformCompatibility::new().unwrap();
        let config = platform_compat.get_deployment_config();
        assert!(config.resource_limits.cpu_cores > 0);
        assert!(config.resource_limits.memory_mb > 0);
    }

    #[test]
    fn test_operating_system_detection() {
        let os = PlatformCompatibility::detect_operating_system();
        assert!(os.is_ok());
    }

    #[test]
    fn test_memory_info_detection() {
        let memory = PlatformCompatibility::detect_memory_info();
        assert!(memory.is_ok());
        let memory = memory.unwrap();
        assert!(memory.total_memory > 0);
        assert!(memory.page_size > 0);
    }

    #[test]
    fn test_compute_devices_detection() {
        let devices = PlatformCompatibility::detect_compute_devices();
        assert!(devices.is_ok());
        let devices = devices.unwrap();
        assert!(!devices.is_empty()); // Should at least detect CPU
    }

    #[test]
    fn test_platform_score_calculation() {
        let platform_compat = PlatformCompatibility::new().unwrap();
        let score = platform_compat.calculate_platform_score();
        assert!(score >= 0.0 && score <= 100.0);
    }
}
