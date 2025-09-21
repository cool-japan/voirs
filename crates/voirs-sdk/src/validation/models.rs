use crate::error::{ModelType, VoirsError};
use crate::types::VoirsResult;
use serde::{Deserialize, Serialize};
#[cfg(feature = "cloud")]
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelValidationConfig {
    pub check_integrity: bool,
    pub check_version_compatibility: bool,
    pub check_hardware_requirements: bool,
    pub check_quality_metrics: bool,
    pub strict_mode: bool,
    pub allowed_model_types: Option<Vec<ModelType>>,
    pub minimum_quality_threshold: f64,
    pub trusted_sources: Vec<String>,
}

impl Default for ModelValidationConfig {
    fn default() -> Self {
        Self {
            check_integrity: true,
            check_version_compatibility: true,
            check_hardware_requirements: true,
            check_quality_metrics: false,
            strict_mode: false,
            allowed_model_types: None,
            minimum_quality_threshold: 0.7,
            trusted_sources: vec!["huggingface.co".to_string(), "github.com".to_string()],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub model_type: ModelType,
    pub architecture: String,
    pub checksum: Option<String>,
    pub size_bytes: u64,
    pub created_at: Option<String>,
    pub source: Option<String>,
    pub license: Option<String>,
    pub description: Option<String>,
    pub requirements: ModelRequirements,
    pub quality_metrics: Option<QualityMetrics>,
    pub compatibility: CompatibilityInfo,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRequirements {
    pub minimum_memory_mb: u64,
    pub recommended_memory_mb: u64,
    pub requires_gpu: bool,
    pub minimum_gpu_memory_mb: Option<u64>,
    pub supported_devices: Vec<String>,
    pub minimum_cpu_cores: Option<u32>,
    pub required_frameworks: Vec<String>,
    pub python_version: Option<String>,
    pub operating_systems: Vec<String>,
}

impl Default for ModelRequirements {
    fn default() -> Self {
        Self {
            minimum_memory_mb: 512,
            recommended_memory_mb: 1024,
            requires_gpu: false,
            minimum_gpu_memory_mb: None,
            supported_devices: vec!["cpu".to_string()],
            minimum_cpu_cores: None,
            required_frameworks: vec![],
            python_version: None,
            operating_systems: vec![
                "linux".to_string(),
                "windows".to_string(),
                "macos".to_string(),
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub accuracy: Option<f64>,
    pub latency_ms: Option<f64>,
    pub model_size_mb: f64,
    pub inference_time_ms: Option<f64>,
    pub quality_score: Option<f64>,
    pub benchmark_results: HashMap<String, f64>,
    pub test_dataset: Option<String>,
    pub evaluation_date: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityInfo {
    pub sdk_version: String,
    pub api_version: String,
    pub backward_compatible_versions: Vec<String>,
    pub breaking_changes: Vec<String>,
    pub migration_guide: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelValidationResult {
    pub is_valid: bool,
    pub model_path: PathBuf,
    pub metadata: Option<ModelMetadata>,
    pub integrity_check: Option<IntegrityCheckResult>,
    pub version_compatibility: Option<VersionCompatibilityResult>,
    pub hardware_compatibility: Option<HardwareCompatibilityResult>,
    pub quality_validation: Option<QualityValidationResult>,
    pub errors: Vec<ModelValidationError>,
    pub warnings: Vec<ModelValidationWarning>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct IntegrityCheckResult {
    pub passed: bool,
    pub expected_checksum: Option<String>,
    pub actual_checksum: String,
    pub file_size_bytes: u64,
    pub corrupted_files: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct VersionCompatibilityResult {
    pub is_compatible: bool,
    pub model_version: String,
    pub sdk_version: String,
    pub compatibility_level: CompatibilityLevel,
    pub required_migration: bool,
    pub migration_steps: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum CompatibilityLevel {
    FullyCompatible,
    BackwardCompatible,
    RequiresMigration,
    Incompatible,
}

#[derive(Debug, Clone)]
pub struct HardwareCompatibilityResult {
    pub is_compatible: bool,
    pub memory_sufficient: bool,
    pub gpu_compatible: bool,
    pub device_supported: bool,
    pub missing_requirements: Vec<String>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct QualityValidationResult {
    pub meets_threshold: bool,
    pub quality_score: f64,
    pub threshold: f64,
    pub benchmark_results: HashMap<String, f64>,
    pub quality_issues: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ModelValidationError {
    pub error_type: ModelErrorType,
    pub message: String,
    pub severity: ValidationSeverity,
    pub affected_component: Option<String>,
    pub resolution: Option<String>,
}

#[derive(Debug, Clone)]
pub enum ModelErrorType {
    IntegrityFailure,
    VersionIncompatible,
    HardwareIncompatible,
    QualityBelowThreshold,
    MetadataMissing,
    UnsupportedFormat,
    SourceUntrusted,
    RequirementsMissing,
}

#[derive(Debug, Clone)]
pub struct ModelValidationWarning {
    pub message: String,
    pub recommendation: Option<String>,
}

use super::config::ValidationSeverity;

pub struct ModelValidator {
    config: ModelValidationConfig,
    system_info: SystemInfo,
}

#[derive(Debug, Clone)]
struct SystemInfo {
    available_memory_mb: u64,
    available_gpu_memory_mb: Option<u64>,
    cpu_cores: u32,
    supported_devices: Vec<String>,
    #[allow(dead_code)]
    operating_system: String,
    #[allow(dead_code)]
    frameworks: Vec<String>,
}

impl ModelValidator {
    pub fn new(config: ModelValidationConfig) -> Self {
        Self {
            config,
            system_info: Self::detect_system_info(),
        }
    }

    pub fn validate_model<P: AsRef<Path>>(
        &self,
        model_path: P,
    ) -> VoirsResult<ModelValidationResult> {
        let model_path = model_path.as_ref().to_path_buf();

        let mut result = ModelValidationResult {
            is_valid: true,
            model_path: model_path.clone(),
            metadata: None,
            integrity_check: None,
            version_compatibility: None,
            hardware_compatibility: None,
            quality_validation: None,
            errors: Vec::new(),
            warnings: Vec::new(),
            recommendations: Vec::new(),
        };

        // Load model metadata
        match self.load_model_metadata(&model_path) {
            Ok(metadata) => {
                result.metadata = Some(metadata.clone());

                // Perform all validations
                if self.config.check_integrity {
                    result.integrity_check = Some(self.validate_integrity(&model_path, &metadata)?);
                }

                if self.config.check_version_compatibility {
                    result.version_compatibility =
                        Some(self.validate_version_compatibility(&metadata)?);
                }

                if self.config.check_hardware_requirements {
                    result.hardware_compatibility =
                        Some(self.validate_hardware_compatibility(&metadata)?);
                }

                if self.config.check_quality_metrics {
                    result.quality_validation = self.validate_quality(&metadata)?;
                }

                // Validate model type
                if let Some(ref allowed_types) = self.config.allowed_model_types {
                    if !allowed_types.contains(&metadata.model_type) {
                        result.errors.push(ModelValidationError {
                            error_type: ModelErrorType::UnsupportedFormat,
                            message: format!("Model type {:?} is not allowed", metadata.model_type),
                            severity: ValidationSeverity::Error,
                            affected_component: Some("model_type".to_string()),
                            resolution: Some(format!("Use one of: {allowed_types:?}")),
                        });
                    }
                }

                // Validate source
                if let Some(ref source) = metadata.source {
                    if !self.is_trusted_source(source) {
                        result.warnings.push(ModelValidationWarning {
                            message: format!(
                                "Model source '{source}' is not in trusted sources list"
                            ),
                            recommendation: Some("Verify model integrity manually".to_string()),
                        });
                    }
                }
            }
            Err(e) => {
                result.errors.push(ModelValidationError {
                    error_type: ModelErrorType::MetadataMissing,
                    message: format!("Failed to load model metadata: {e}"),
                    severity: ValidationSeverity::Error,
                    affected_component: Some("metadata".to_string()),
                    resolution: Some("Ensure model has valid metadata file".to_string()),
                });
            }
        }

        // Check if model file exists
        if !model_path.exists() {
            result.errors.push(ModelValidationError {
                error_type: ModelErrorType::RequirementsMissing,
                message: format!("Model file not found: {}", model_path.display()),
                severity: ValidationSeverity::Critical,
                affected_component: None,
                resolution: Some("Check file path and permissions".to_string()),
            });
        }

        // Aggregate validation results
        result.is_valid = self.determine_validity(&result);

        // Generate recommendations
        result.recommendations = self.generate_recommendations(&result);

        Ok(result)
    }

    fn load_model_metadata(&self, model_path: &Path) -> VoirsResult<ModelMetadata> {
        // Try to find metadata file
        let metadata_paths = vec![
            model_path.join("model.json"),
            model_path.join("config.json"),
            model_path.join("metadata.json"),
            model_path.with_extension("json"),
        ];

        for metadata_path in metadata_paths {
            if metadata_path.exists() {
                let content = std::fs::read_to_string(&metadata_path).map_err(|e| {
                    VoirsError::io_error(metadata_path.clone(), crate::error::IoOperation::Read, e)
                })?;

                let metadata: ModelMetadata = serde_json::from_str(&content).map_err(|e| {
                    VoirsError::serialization("JSON", format!("Invalid metadata format: {e}"))
                })?;

                return Ok(metadata);
            }
        }

        // If no metadata file found, create basic metadata from file info
        let file_size = std::fs::metadata(model_path)
            .map_err(|e| VoirsError::io_error(model_path, crate::error::IoOperation::Metadata, e))?
            .len();

        Ok(ModelMetadata {
            name: model_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
            version: "unknown".to_string(),
            model_type: self.infer_model_type(model_path),
            architecture: "unknown".to_string(),
            checksum: None,
            size_bytes: file_size,
            created_at: None,
            source: None,
            license: None,
            description: None,
            requirements: ModelRequirements::default(),
            quality_metrics: None,
            compatibility: CompatibilityInfo {
                sdk_version: "unknown".to_string(),
                api_version: "unknown".to_string(),
                backward_compatible_versions: vec![],
                breaking_changes: vec![],
                migration_guide: None,
            },
            tags: vec![],
        })
    }

    fn validate_integrity(
        &self,
        model_path: &Path,
        metadata: &ModelMetadata,
    ) -> VoirsResult<IntegrityCheckResult> {
        let actual_checksum = self.calculate_checksum(model_path)?;
        let file_size = std::fs::metadata(model_path)
            .map_err(|e| VoirsError::io_error(model_path, crate::error::IoOperation::Metadata, e))?
            .len();

        let passed = if let Some(ref expected) = metadata.checksum {
            *expected == actual_checksum
        } else {
            true // No checksum to verify against
        };

        Ok(IntegrityCheckResult {
            passed,
            expected_checksum: metadata.checksum.clone(),
            actual_checksum,
            file_size_bytes: file_size,
            corrupted_files: if passed {
                vec![]
            } else {
                vec![model_path.display().to_string()]
            },
        })
    }

    fn validate_version_compatibility(
        &self,
        metadata: &ModelMetadata,
    ) -> VoirsResult<VersionCompatibilityResult> {
        let sdk_version = env!("CARGO_PKG_VERSION");
        let compatibility_level =
            self.check_compatibility_level(&metadata.compatibility, sdk_version);

        let is_compatible = !matches!(compatibility_level, CompatibilityLevel::Incompatible);
        let required_migration =
            matches!(compatibility_level, CompatibilityLevel::RequiresMigration);

        Ok(VersionCompatibilityResult {
            is_compatible,
            model_version: metadata.version.clone(),
            sdk_version: sdk_version.to_string(),
            compatibility_level,
            required_migration,
            migration_steps: if required_migration {
                vec!["Update model to latest version".to_string()]
            } else {
                vec![]
            },
        })
    }

    fn validate_hardware_compatibility(
        &self,
        metadata: &ModelMetadata,
    ) -> VoirsResult<HardwareCompatibilityResult> {
        let requirements = &metadata.requirements;
        let mut missing_requirements = Vec::new();
        let mut recommendations = Vec::new();

        // Check memory requirements
        let memory_sufficient =
            self.system_info.available_memory_mb >= requirements.minimum_memory_mb;
        if !memory_sufficient {
            missing_requirements.push(format!(
                "Insufficient memory: {} MB required, {} MB available",
                requirements.minimum_memory_mb, self.system_info.available_memory_mb
            ));
        }

        if self.system_info.available_memory_mb < requirements.recommended_memory_mb {
            recommendations.push(format!(
                "Recommended memory: {} MB (current: {} MB)",
                requirements.recommended_memory_mb, self.system_info.available_memory_mb
            ));
        }

        // Check GPU requirements
        let gpu_compatible = if requirements.requires_gpu {
            if let Some(available_gpu_memory) = self.system_info.available_gpu_memory_mb {
                if let Some(required_gpu_memory) = requirements.minimum_gpu_memory_mb {
                    available_gpu_memory >= required_gpu_memory
                } else {
                    true // GPU available but no specific memory requirement
                }
            } else {
                missing_requirements.push("GPU required but not available".to_string());
                false
            }
        } else {
            true // GPU not required
        };

        // Check device support
        let device_supported = requirements
            .supported_devices
            .iter()
            .any(|device| self.system_info.supported_devices.contains(device));

        if !device_supported {
            missing_requirements.push(format!(
                "No supported device found. Required: {:?}, Available: {:?}",
                requirements.supported_devices, self.system_info.supported_devices
            ));
        }

        // Check CPU cores
        if let Some(min_cores) = requirements.minimum_cpu_cores {
            if self.system_info.cpu_cores < min_cores {
                missing_requirements.push(format!(
                    "Insufficient CPU cores: {} required, {} available",
                    min_cores, self.system_info.cpu_cores
                ));
            }
        }

        let is_compatible = missing_requirements.is_empty();

        Ok(HardwareCompatibilityResult {
            is_compatible,
            memory_sufficient,
            gpu_compatible,
            device_supported,
            missing_requirements,
            recommendations,
        })
    }

    fn validate_quality(
        &self,
        metadata: &ModelMetadata,
    ) -> VoirsResult<Option<QualityValidationResult>> {
        if let Some(ref quality_metrics) = metadata.quality_metrics {
            let quality_score = quality_metrics.quality_score.unwrap_or(0.0);
            let meets_threshold = quality_score >= self.config.minimum_quality_threshold;

            let mut quality_issues = Vec::new();
            if !meets_threshold {
                quality_issues.push(format!(
                    "Quality score {} below threshold {}",
                    quality_score, self.config.minimum_quality_threshold
                ));
            }

            Ok(Some(QualityValidationResult {
                meets_threshold,
                quality_score,
                threshold: self.config.minimum_quality_threshold,
                benchmark_results: quality_metrics.benchmark_results.clone(),
                quality_issues,
            }))
        } else {
            Ok(None)
        }
    }

    fn calculate_checksum(&self, file_path: &Path) -> VoirsResult<String> {
        #[cfg(feature = "cloud")]
        {
            let content = std::fs::read(file_path)
                .map_err(|e| VoirsError::io_error(file_path, crate::error::IoOperation::Read, e))?;

            let mut hasher = Sha256::new();
            hasher.update(&content);
            let result = hasher.finalize();

            Ok(format!("{:x}", result))
        }
        #[cfg(not(feature = "cloud"))]
        {
            // When cloud feature is not enabled, return a simple hash based on file metadata
            let metadata = std::fs::metadata(file_path).map_err(|e| {
                VoirsError::io_error(file_path, crate::error::IoOperation::Metadata, e)
            })?;

            Ok(format!(
                "{:x}",
                metadata.len()
                    ^ metadata
                        .modified()
                        .unwrap_or(std::time::UNIX_EPOCH)
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs()
            ))
        }
    }

    fn infer_model_type(&self, model_path: &Path) -> ModelType {
        if let Some(extension) = model_path.extension().and_then(|s| s.to_str()) {
            match extension.to_lowercase().as_str() {
                "onnx" | "ort" => ModelType::Acoustic, // Default to acoustic for ONNX
                "pt" | "pth" => ModelType::Acoustic,   // PyTorch models
                "bin" | "safetensors" => ModelType::Acoustic, // Hugging Face models
                _ => ModelType::Acoustic,              // Default
            }
        } else {
            ModelType::Acoustic
        }
    }

    fn check_compatibility_level(
        &self,
        compatibility: &CompatibilityInfo,
        sdk_version: &str,
    ) -> CompatibilityLevel {
        if compatibility.sdk_version == sdk_version {
            CompatibilityLevel::FullyCompatible
        } else if compatibility
            .backward_compatible_versions
            .contains(&sdk_version.to_string())
        {
            CompatibilityLevel::BackwardCompatible
        } else if !compatibility.breaking_changes.is_empty() {
            CompatibilityLevel::RequiresMigration
        } else {
            CompatibilityLevel::Incompatible
        }
    }

    fn is_trusted_source(&self, source: &str) -> bool {
        self.config
            .trusted_sources
            .iter()
            .any(|trusted| source.contains(trusted))
    }

    fn determine_validity(&self, result: &ModelValidationResult) -> bool {
        if self.config.strict_mode {
            result.errors.is_empty()
        } else {
            !result.errors.iter().any(|e| {
                matches!(
                    e.severity,
                    ValidationSeverity::Critical | ValidationSeverity::Error
                )
            })
        }
    }

    fn generate_recommendations(&self, result: &ModelValidationResult) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Hardware recommendations
        if let Some(ref hw_compat) = result.hardware_compatibility {
            recommendations.extend(hw_compat.recommendations.clone());
        }

        // Version compatibility recommendations
        if let Some(ref version_compat) = result.version_compatibility {
            if version_compat.required_migration {
                recommendations.extend(version_compat.migration_steps.clone());
            }
        }

        // Quality recommendations
        if let Some(ref quality) = result.quality_validation {
            if !quality.meets_threshold {
                recommendations.push("Consider using a higher quality model".to_string());
            }
        }

        // Integrity recommendations
        if let Some(ref integrity) = result.integrity_check {
            if !integrity.passed {
                recommendations.push("Re-download the model to ensure integrity".to_string());
            }
        }

        recommendations
    }

    fn detect_system_info() -> SystemInfo {
        // Simplified system detection - in production this would use proper system APIs
        SystemInfo {
            available_memory_mb: 8192,           // 8GB default
            available_gpu_memory_mb: Some(4096), // 4GB default
            cpu_cores: 8,
            supported_devices: vec!["cpu".to_string(), "cuda".to_string()],
            operating_system: std::env::consts::OS.to_string(),
            frameworks: vec!["onnx".to_string(), "pytorch".to_string()],
        }
    }
}

pub fn validate_model_basic<P: AsRef<Path>>(model_path: P) -> VoirsResult<bool> {
    let validator = ModelValidator::new(ModelValidationConfig::default());
    let result = validator.validate_model(model_path)?;
    Ok(result.is_valid)
}

pub fn validate_model_with_config<P: AsRef<Path>>(
    model_path: P,
    config: ModelValidationConfig,
) -> VoirsResult<ModelValidationResult> {
    let validator = ModelValidator::new(config);
    validator.validate_model(model_path)
}
