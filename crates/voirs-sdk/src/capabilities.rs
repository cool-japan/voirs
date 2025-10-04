//! Feature capability detection and negotiation system for VoiRS SDK.
//!
//! This module provides comprehensive capability detection for hardware, software features,
//! and model support to enable intelligent feature negotiation and resource management.

use crate::types::{
    AdvancedFeature, CapabilityNegotiation, CapabilityRequest, FallbackStrategy, FeaturePriority,
    HardwareCapabilities, HardwareRequirements, ModelCapabilities, PerformanceProfile,
    ResourceLimits, ResourceUsage, SystemCapabilities,
};
use crate::{VoirsError, VoirsResult};
use std::collections::HashMap;

/// Capability detection and negotiation manager
pub struct CapabilityManager {
    /// Cached system capabilities
    system_caps: SystemCapabilities,
    /// Feature detectors
    feature_detectors: HashMap<AdvancedFeature, Box<dyn FeatureDetector>>,
}

/// Trait for detecting specific feature availability
pub trait FeatureDetector: Send + Sync {
    /// Check if the feature is available in the current environment
    fn is_available(&self) -> bool;

    /// Get hardware requirements for this feature
    fn hardware_requirements(&self) -> HardwareRequirements;

    /// Get performance characteristics
    fn performance_profile(&self) -> PerformanceProfile;

    /// Get alternative implementations if primary is unavailable
    fn alternatives(&self) -> Vec<String>;
}

impl CapabilityManager {
    /// Create a new capability manager with system detection
    pub fn new() -> VoirsResult<Self> {
        let system_caps = Self::detect_system_capabilities()?;
        let feature_detectors = Self::create_feature_detectors();

        Ok(Self {
            system_caps,
            feature_detectors,
        })
    }

    /// Detect current system capabilities
    pub fn detect_system_capabilities() -> VoirsResult<SystemCapabilities> {
        let hardware = Self::detect_hardware_capabilities()?;
        let available_features = Self::detect_available_features(&hardware)?;
        let resource_limits = Self::detect_resource_limits(&hardware);

        Ok(SystemCapabilities {
            available_features,
            hardware,
            resource_limits,
            model_capabilities: Self::detect_model_capabilities()?,
        })
    }

    /// Detect hardware capabilities
    fn detect_hardware_capabilities() -> VoirsResult<HardwareCapabilities> {
        let cpu_cores = num_cpus::get() as u32;
        let system_memory_mb = Self::detect_system_memory()?;
        let (gpu_available, gpu_memory_mb) = Self::detect_gpu_capabilities();
        let fast_storage = Self::detect_storage_type();

        Ok(HardwareCapabilities {
            gpu_available,
            gpu_memory_mb,
            cpu_cores,
            system_memory_mb,
            fast_storage,
        })
    }

    /// Detect available memory in MB
    fn detect_system_memory() -> VoirsResult<u64> {
        // Platform-specific memory detection
        #[cfg(target_os = "linux")]
        {
            if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                for line in meminfo.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return Ok(kb / 1024); // Convert KB to MB
                            }
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            if let Ok(output) = Command::new("sysctl").arg("-n").arg("hw.memsize").output() {
                if let Ok(size_str) = String::from_utf8(output.stdout) {
                    if let Ok(bytes) = size_str.trim().parse::<u64>() {
                        return Ok(bytes / (1024 * 1024)); // Convert bytes to MB
                    }
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            // Conservative default for Windows - could be improved with Windows API calls
            return Ok(8192); // 8GB default
        }

        // Conservative fallback
        Ok(4096) // 4GB default
    }

    /// Detect GPU capabilities
    fn detect_gpu_capabilities() -> (bool, Option<u64>) {
        // Check for common GPU indicators
        let gpu_available = Self::check_gpu_available();
        let gpu_memory = if gpu_available {
            Self::detect_gpu_memory()
        } else {
            None
        };

        (gpu_available, gpu_memory)
    }

    /// Check if GPU is available
    fn check_gpu_available() -> bool {
        // Check for common GPU directories/files
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/dev/nvidia0").exists()
                || std::path::Path::new("/dev/dri/card0").exists()
        }

        #[cfg(target_os = "macos")]
        {
            // Metal is available on all modern Macs
            true
        }

        #[cfg(target_os = "windows")]
        {
            // Assume GPU is available on Windows
            true
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            false
        }
    }

    /// Detect GPU memory
    fn detect_gpu_memory() -> Option<u64> {
        // Platform-specific GPU memory detection
        #[cfg(target_os = "linux")]
        {
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args(&["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
                .output()
            {
                if let Ok(memory_str) = String::from_utf8(output.stdout) {
                    if let Ok(memory_mb) = memory_str.trim().parse::<u64>() {
                        return Some(memory_mb);
                    }
                }
            }
        }

        // Conservative default for GPU memory
        Some(2048) // 2GB default
    }

    /// Detect storage type
    fn detect_storage_type() -> bool {
        // Simple heuristic: assume SSD if system is modern enough
        #[cfg(target_os = "linux")]
        {
            if let Ok(output) = std::process::Command::new("lsblk")
                .args(&["-d", "-o", "ROTA"])
                .output()
            {
                if let Ok(rota_str) = String::from_utf8(output.stdout) {
                    // If any drive shows '0' (non-rotating), assume SSD
                    return rota_str.contains('0');
                }
            }
        }

        // Default to assuming fast storage
        true
    }

    /// Detect available features based on hardware and compilation features
    fn detect_available_features(
        hardware: &HardwareCapabilities,
    ) -> VoirsResult<Vec<AdvancedFeature>> {
        let mut features = Vec::new();

        // Always available features
        features.push(AdvancedFeature::StreamingSynthesis);
        features.push(AdvancedFeature::RealtimeProcessing);

        // Conditional features based on compilation flags
        #[cfg(feature = "emotion")]
        features.push(AdvancedFeature::EmotionControl);

        #[cfg(feature = "cloning")]
        features.push(AdvancedFeature::VoiceCloning);

        #[cfg(feature = "conversion")]
        features.push(AdvancedFeature::VoiceConversion);

        #[cfg(feature = "singing")]
        features.push(AdvancedFeature::SingingSynthesis);

        #[cfg(feature = "spatial")]
        features.push(AdvancedFeature::SpatialAudio);

        #[cfg(feature = "wasm")]
        features.push(AdvancedFeature::WasmSupport);

        #[cfg(feature = "cloud")]
        features.push(AdvancedFeature::CloudProcessing);

        // Hardware-dependent features
        if hardware.gpu_available {
            features.push(AdvancedFeature::GpuAcceleration);
        }

        if hardware.system_memory_mb >= 4096 {
            features.push(AdvancedFeature::HighQualityVocoding);
        }

        Ok(features)
    }

    /// Detect resource limits based on hardware
    fn detect_resource_limits(hardware: &HardwareCapabilities) -> ResourceLimits {
        let max_memory_mb = (hardware.system_memory_mb * 3) / 4; // Use 75% of available memory
        let max_cpu_percent = if hardware.cpu_cores >= 4 { 80 } else { 60 };
        let max_latency_ms = if hardware.fast_storage { 200 } else { 500 };

        ResourceLimits {
            max_memory_mb,
            max_cpu_percent,
            max_latency_ms,
            battery_optimization: false, // Default to performance mode
        }
    }

    /// Detect model capabilities
    fn detect_model_capabilities() -> VoirsResult<HashMap<String, ModelCapabilities>> {
        let mut capabilities = HashMap::new();

        // Default voice model capabilities
        capabilities.insert(
            "default".to_string(),
            ModelCapabilities {
                supported_features: vec![
                    AdvancedFeature::StreamingSynthesis,
                    AdvancedFeature::RealtimeProcessing,
                ],
                hardware_requirements: HardwareRequirements {
                    min_memory_mb: 512,
                    min_gpu_memory_mb: None,
                    requires_gpu: false,
                    min_cpu_cores: 1,
                },
                performance_profile: PerformanceProfile {
                    init_latency_ms: 1000,
                    synthesis_latency_ms_per_sec: 200,
                    synthesis_memory_mb: 256,
                    quality_score: 80, // 0.8 as u8
                },
            },
        );

        // High-quality model capabilities
        capabilities.insert(
            "high_quality".to_string(),
            ModelCapabilities {
                supported_features: vec![
                    AdvancedFeature::StreamingSynthesis,
                    AdvancedFeature::RealtimeProcessing,
                    AdvancedFeature::HighQualityVocoding,
                    AdvancedFeature::EmotionControl,
                ],
                hardware_requirements: HardwareRequirements {
                    min_memory_mb: 2048,
                    min_gpu_memory_mb: Some(1024),
                    requires_gpu: false,
                    min_cpu_cores: 2,
                },
                performance_profile: PerformanceProfile {
                    init_latency_ms: 3000,
                    synthesis_latency_ms_per_sec: 500,
                    synthesis_memory_mb: 1024,
                    quality_score: 95, // 0.95 as u8
                },
            },
        );

        Ok(capabilities)
    }

    /// Create feature detectors for all supported features
    fn create_feature_detectors() -> HashMap<AdvancedFeature, Box<dyn FeatureDetector>> {
        let mut detectors: HashMap<AdvancedFeature, Box<dyn FeatureDetector>> = HashMap::new();

        detectors.insert(
            AdvancedFeature::EmotionControl,
            Box::new(EmotionFeatureDetector),
        );
        detectors.insert(
            AdvancedFeature::VoiceCloning,
            Box::new(CloningFeatureDetector),
        );
        detectors.insert(
            AdvancedFeature::VoiceConversion,
            Box::new(ConversionFeatureDetector),
        );
        detectors.insert(
            AdvancedFeature::SingingSynthesis,
            Box::new(SingingFeatureDetector),
        );
        detectors.insert(
            AdvancedFeature::SpatialAudio,
            Box::new(SpatialFeatureDetector),
        );
        detectors.insert(
            AdvancedFeature::StreamingSynthesis,
            Box::new(StreamingFeatureDetector),
        );
        detectors.insert(
            AdvancedFeature::GpuAcceleration,
            Box::new(GpuFeatureDetector),
        );
        detectors.insert(
            AdvancedFeature::HighQualityVocoding,
            Box::new(HighQualityFeatureDetector),
        );
        detectors.insert(
            AdvancedFeature::RealtimeProcessing,
            Box::new(RealtimeFeatureDetector),
        );

        detectors
    }

    /// Negotiate capabilities based on a request
    pub fn negotiate_capabilities(
        &self,
        request: &CapabilityRequest,
    ) -> VoirsResult<CapabilityNegotiation> {
        let mut enabled_features = Vec::new();
        let mut unavailable_features = Vec::new();
        let mut warnings = Vec::new();
        let mut estimated_memory = 0u64;
        let mut estimated_init_time = 0u32;
        let mut estimated_latency = 0u32;
        let mut estimated_cpu = 0u8;

        // Check each requested feature
        for (feature, priority) in request
            .desired_features
            .iter()
            .zip(request.feature_priorities.iter())
        {
            if self.system_caps.available_features.contains(feature) {
                if let Some(detector) = self.feature_detectors.get(feature) {
                    let hw_req = detector.hardware_requirements();
                    let perf = detector.performance_profile();

                    // Check if hardware requirements are met
                    if self.check_hardware_requirements(&hw_req) {
                        enabled_features.push(*feature);
                        estimated_memory += perf.synthesis_memory_mb;
                        estimated_init_time = estimated_init_time.max(perf.init_latency_ms);
                        estimated_latency += perf.synthesis_latency_ms_per_sec;
                        estimated_cpu = (estimated_cpu + 20).min(100); // Rough estimate
                    } else {
                        unavailable_features.push(*feature);
                        if matches!(
                            priority,
                            FeaturePriority::Required | FeaturePriority::Critical
                        ) {
                            match request.fallback_strategy {
                                FallbackStrategy::FailFast => {
                                    return Err(VoirsError::FeatureUnavailable {
                                        feature: format!("{:?}", feature),
                                        reason: "Hardware requirements not met".to_string(),
                                    });
                                }
                                _ => {
                                    warnings.push(format!("Required feature {:?} unavailable due to hardware constraints", feature));
                                }
                            }
                        }
                    }
                } else {
                    unavailable_features.push(*feature);
                }
            } else {
                unavailable_features.push(*feature);
                if matches!(priority, FeaturePriority::Critical) {
                    return Err(VoirsError::FeatureUnavailable {
                        feature: format!("{:?}", feature),
                        reason: "Feature not compiled or available".to_string(),
                    });
                }
            }
        }

        // Check resource constraints
        if estimated_memory > request.constraints.max_memory_mb {
            warnings.push(format!(
                "Estimated memory usage ({} MB) exceeds limit ({} MB)",
                estimated_memory, request.constraints.max_memory_mb
            ));
        }

        if estimated_latency > request.constraints.max_latency_ms {
            warnings.push(format!(
                "Estimated latency ({} ms) exceeds tolerance ({} ms)",
                estimated_latency, request.constraints.max_latency_ms
            ));
        }

        // Select appropriate models based on enabled features
        let selected_models = self.select_models(&enabled_features)?;

        Ok(CapabilityNegotiation {
            enabled_features,
            unavailable_features,
            warnings,
            selected_models,
            estimated_usage: ResourceUsage {
                memory_mb: estimated_memory,
                init_time_ms: estimated_init_time,
                processing_latency_ms: estimated_latency,
                cpu_usage_percent: estimated_cpu,
            },
        })
    }

    /// Check if hardware requirements are met
    fn check_hardware_requirements(&self, requirements: &HardwareRequirements) -> bool {
        let hw = &self.system_caps.hardware;

        if hw.system_memory_mb < requirements.min_memory_mb {
            return false;
        }

        if hw.cpu_cores < requirements.min_cpu_cores {
            return false;
        }

        if requirements.requires_gpu && !hw.gpu_available {
            return false;
        }

        if let Some(min_gpu_mem) = requirements.min_gpu_memory_mb {
            if let Some(gpu_mem) = hw.gpu_memory_mb {
                if gpu_mem < min_gpu_mem {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }

    /// Select appropriate models based on enabled features
    fn select_models(
        &self,
        enabled_features: &[AdvancedFeature],
    ) -> VoirsResult<HashMap<String, String>> {
        let mut models = HashMap::new();

        // Select models based on feature requirements and hardware capabilities
        if enabled_features.contains(&AdvancedFeature::HighQualityVocoding)
            && self.system_caps.hardware.system_memory_mb >= 2048
        {
            models.insert("voice".to_string(), "high_quality".to_string());
        } else {
            models.insert("voice".to_string(), "default".to_string());
        }

        if enabled_features.contains(&AdvancedFeature::GpuAcceleration)
            && self.system_caps.hardware.gpu_available
        {
            models.insert("vocoder".to_string(), "gpu_optimized".to_string());
        } else {
            models.insert("vocoder".to_string(), "cpu_optimized".to_string());
        }

        Ok(models)
    }

    /// Get current system capabilities
    pub fn system_capabilities(&self) -> &SystemCapabilities {
        &self.system_caps
    }

    /// Refresh system capabilities
    pub fn refresh_capabilities(&mut self) -> VoirsResult<()> {
        self.system_caps = Self::detect_system_capabilities()?;
        Ok(())
    }
}

impl Default for CapabilityManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            system_caps: SystemCapabilities::default(),
            feature_detectors: Self::create_feature_detectors(),
        })
    }
}

// Feature detector implementations
struct EmotionFeatureDetector;
impl FeatureDetector for EmotionFeatureDetector {
    fn is_available(&self) -> bool {
        cfg!(feature = "emotion")
    }

    fn hardware_requirements(&self) -> HardwareRequirements {
        HardwareRequirements {
            min_memory_mb: 256,
            min_gpu_memory_mb: None,
            requires_gpu: false,
            min_cpu_cores: 1,
        }
    }

    fn performance_profile(&self) -> PerformanceProfile {
        PerformanceProfile {
            init_latency_ms: 100,
            synthesis_latency_ms_per_sec: 50,
            synthesis_memory_mb: 128,
            quality_score: 85,
        }
    }

    fn alternatives(&self) -> Vec<String> {
        vec!["basic_prosody_control".to_string()]
    }
}

struct CloningFeatureDetector;
impl FeatureDetector for CloningFeatureDetector {
    fn is_available(&self) -> bool {
        cfg!(feature = "cloning")
    }

    fn hardware_requirements(&self) -> HardwareRequirements {
        HardwareRequirements {
            min_memory_mb: 1024,
            min_gpu_memory_mb: Some(512),
            requires_gpu: false,
            min_cpu_cores: 2,
        }
    }

    fn performance_profile(&self) -> PerformanceProfile {
        PerformanceProfile {
            init_latency_ms: 2000,
            synthesis_latency_ms_per_sec: 400,
            synthesis_memory_mb: 512,
            quality_score: 90,
        }
    }

    fn alternatives(&self) -> Vec<String> {
        vec!["voice_selection".to_string()]
    }
}

struct ConversionFeatureDetector;
impl FeatureDetector for ConversionFeatureDetector {
    fn is_available(&self) -> bool {
        cfg!(feature = "conversion")
    }

    fn hardware_requirements(&self) -> HardwareRequirements {
        HardwareRequirements {
            min_memory_mb: 512,
            min_gpu_memory_mb: None,
            requires_gpu: false,
            min_cpu_cores: 2,
        }
    }

    fn performance_profile(&self) -> PerformanceProfile {
        PerformanceProfile {
            init_latency_ms: 500,
            synthesis_latency_ms_per_sec: 200,
            synthesis_memory_mb: 256,
            quality_score: 80,
        }
    }

    fn alternatives(&self) -> Vec<String> {
        vec!["pitch_shifting".to_string()]
    }
}

struct SingingFeatureDetector;
impl FeatureDetector for SingingFeatureDetector {
    fn is_available(&self) -> bool {
        cfg!(feature = "singing")
    }

    fn hardware_requirements(&self) -> HardwareRequirements {
        HardwareRequirements {
            min_memory_mb: 768,
            min_gpu_memory_mb: None,
            requires_gpu: false,
            min_cpu_cores: 2,
        }
    }

    fn performance_profile(&self) -> PerformanceProfile {
        PerformanceProfile {
            init_latency_ms: 1500,
            synthesis_latency_ms_per_sec: 600,
            synthesis_memory_mb: 384,
            quality_score: 85,
        }
    }

    fn alternatives(&self) -> Vec<String> {
        vec!["prosody_control".to_string()]
    }
}

struct SpatialFeatureDetector;
impl FeatureDetector for SpatialFeatureDetector {
    fn is_available(&self) -> bool {
        cfg!(feature = "spatial")
    }

    fn hardware_requirements(&self) -> HardwareRequirements {
        HardwareRequirements {
            min_memory_mb: 512,
            min_gpu_memory_mb: None,
            requires_gpu: false,
            min_cpu_cores: 2,
        }
    }

    fn performance_profile(&self) -> PerformanceProfile {
        PerformanceProfile {
            init_latency_ms: 300,
            synthesis_latency_ms_per_sec: 150,
            synthesis_memory_mb: 256,
            quality_score: 80,
        }
    }

    fn alternatives(&self) -> Vec<String> {
        vec!["stereo_panning".to_string()]
    }
}

struct StreamingFeatureDetector;
impl FeatureDetector for StreamingFeatureDetector {
    fn is_available(&self) -> bool {
        true // Always available
    }

    fn hardware_requirements(&self) -> HardwareRequirements {
        HardwareRequirements {
            min_memory_mb: 128,
            min_gpu_memory_mb: None,
            requires_gpu: false,
            min_cpu_cores: 1,
        }
    }

    fn performance_profile(&self) -> PerformanceProfile {
        PerformanceProfile {
            init_latency_ms: 50,
            synthesis_latency_ms_per_sec: 100,
            synthesis_memory_mb: 64,
            quality_score: 75,
        }
    }

    fn alternatives(&self) -> Vec<String> {
        vec!["batch_processing".to_string()]
    }
}

struct GpuFeatureDetector;
impl FeatureDetector for GpuFeatureDetector {
    fn is_available(&self) -> bool {
        CapabilityManager::check_gpu_available()
    }

    fn hardware_requirements(&self) -> HardwareRequirements {
        HardwareRequirements {
            min_memory_mb: 1024,
            min_gpu_memory_mb: Some(1024),
            requires_gpu: true,
            min_cpu_cores: 2,
        }
    }

    fn performance_profile(&self) -> PerformanceProfile {
        PerformanceProfile {
            init_latency_ms: 1000,
            synthesis_latency_ms_per_sec: 100,
            synthesis_memory_mb: 512,
            quality_score: 90,
        }
    }

    fn alternatives(&self) -> Vec<String> {
        vec!["cpu_optimized".to_string()]
    }
}

struct HighQualityFeatureDetector;
impl FeatureDetector for HighQualityFeatureDetector {
    fn is_available(&self) -> bool {
        true // Available if system has enough memory
    }

    fn hardware_requirements(&self) -> HardwareRequirements {
        HardwareRequirements {
            min_memory_mb: 2048,
            min_gpu_memory_mb: None,
            requires_gpu: false,
            min_cpu_cores: 2,
        }
    }

    fn performance_profile(&self) -> PerformanceProfile {
        PerformanceProfile {
            init_latency_ms: 2000,
            synthesis_latency_ms_per_sec: 800,
            synthesis_memory_mb: 1024,
            quality_score: 95,
        }
    }

    fn alternatives(&self) -> Vec<String> {
        vec!["medium_quality".to_string(), "standard_quality".to_string()]
    }
}

struct RealtimeFeatureDetector;
impl FeatureDetector for RealtimeFeatureDetector {
    fn is_available(&self) -> bool {
        true // Always available
    }

    fn hardware_requirements(&self) -> HardwareRequirements {
        HardwareRequirements {
            min_memory_mb: 256,
            min_gpu_memory_mb: None,
            requires_gpu: false,
            min_cpu_cores: 1,
        }
    }

    fn performance_profile(&self) -> PerformanceProfile {
        PerformanceProfile {
            init_latency_ms: 100,
            synthesis_latency_ms_per_sec: 150,
            synthesis_memory_mb: 128,
            quality_score: 80,
        }
    }

    fn alternatives(&self) -> Vec<String> {
        vec!["batch_processing".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_manager_creation() {
        let manager = CapabilityManager::new();
        assert!(manager.is_ok());
    }

    #[test]
    fn test_system_capabilities_detection() {
        let caps = CapabilityManager::detect_system_capabilities();
        assert!(caps.is_ok());

        let caps = caps.unwrap();
        assert!(!caps.available_features.is_empty());
        assert!(caps.hardware.cpu_cores > 0);
        assert!(caps.hardware.system_memory_mb > 0);
    }

    #[test]
    fn test_capability_negotiation() {
        let manager = CapabilityManager::new().unwrap();

        let request = CapabilityRequest {
            desired_features: vec![
                AdvancedFeature::StreamingSynthesis,
                AdvancedFeature::RealtimeProcessing,
            ],
            feature_priorities: vec![FeaturePriority::Required, FeaturePriority::Preferred],
            constraints: ResourceLimits::default(),
            fallback_strategy: FallbackStrategy::GracefulDegradation,
        };

        let negotiation = manager.negotiate_capabilities(&request);
        assert!(negotiation.is_ok());

        let negotiation = negotiation.unwrap();
        assert!(!negotiation.enabled_features.is_empty());
    }

    #[test]
    fn test_hardware_requirements_checking() {
        let manager = CapabilityManager::new().unwrap();

        let minimal_req = HardwareRequirements {
            min_memory_mb: 100,
            min_gpu_memory_mb: None,
            requires_gpu: false,
            min_cpu_cores: 1,
        };

        assert!(manager.check_hardware_requirements(&minimal_req));

        let excessive_req = HardwareRequirements {
            min_memory_mb: 1_000_000, // 1TB
            min_gpu_memory_mb: None,
            requires_gpu: false,
            min_cpu_cores: 1,
        };

        assert!(!manager.check_hardware_requirements(&excessive_req));
    }
}
