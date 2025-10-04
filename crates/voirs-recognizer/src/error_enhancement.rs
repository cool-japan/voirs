//! Enhanced error messages and solutions for VoiRS Recognizer
//!
//! This module provides comprehensive error enhancement functionality that adds
//! detailed context, recovery suggestions, and actionable solutions to errors.

use crate::RecognitionError;
use std::collections::HashMap;
use std::fmt;

/// Enhanced error information with context and solutions
#[derive(Debug, Clone)]
pub struct ErrorEnhancement {
    /// Original error message
    pub original_message: String,
    /// Error category for better classification
    pub category: ErrorCategory,
    /// Severity level of the error
    pub severity: ErrorSeverity,
    /// Detailed context about when and how the error occurred
    pub context: ErrorContext,
    /// Suggested solutions and recovery actions
    pub solutions: Vec<Solution>,
    /// Related documentation links
    pub documentation_links: Vec<String>,
    /// Troubleshooting steps
    pub troubleshooting_steps: Vec<String>,
}

/// Error category for better classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// Configuration or setup issues
    Configuration,
    /// Resource availability issues (memory, disk, network)
    Resources,
    /// Audio format or quality issues
    AudioFormat,
    /// Model loading or inference issues
    ModelIssues,
    /// Performance or timeout issues
    Performance,
    /// Input validation issues
    InputValidation,
    /// System integration issues
    Integration,
    /// Feature availability issues
    FeatureSupport,
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum ErrorSeverity {
    /// Critical errors that prevent core functionality
    Critical,
    /// High priority errors that significantly impact functionality
    High,
    /// Medium priority errors that may cause degraded experience
    Medium,
    /// Low priority errors that have minimal impact
    Low,
    /// Informational messages
    Info,
}

/// Error context information
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Component or module where the error occurred
    pub component: String,
    /// Operation being performed when error occurred
    pub operation: String,
    /// Input parameters or configuration relevant to the error
    pub input_summary: String,
    /// System state when error occurred
    pub system_state: String,
    /// Timestamp when error occurred
    pub timestamp: std::time::SystemTime,
    /// Additional context information
    pub additional_info: std::collections::HashMap<String, String>,
}

/// Solution with actionable steps
#[derive(Debug, Clone)]
pub struct Solution {
    /// Solution title
    pub title: String,
    /// Detailed description of the solution
    pub description: String,
    /// Priority of this solution (1 = highest)
    pub priority: u8,
    /// Estimated time to implement
    pub estimated_time: String,
    /// Difficulty level
    pub difficulty: SolutionDifficulty,
    /// Step-by-step instructions
    pub steps: Vec<String>,
    /// Code example if applicable
    pub code_example: Option<String>,
    /// Success indicators
    pub success_indicators: Vec<String>,
}

/// Solution difficulty levels
#[derive(Debug, Clone, PartialEq)]
pub enum SolutionDifficulty {
    /// Easy to implement
    Easy,
    /// Moderate complexity
    Moderate,
    /// Advanced solution requiring expertise
    Advanced,
}

/// Error enhancement trait
pub trait ErrorEnhancer {
    /// Enhance an error with detailed context and solutions
    fn enhance_error(&self) -> ErrorEnhancement;

    /// Get formatted error message with solutions
    fn get_enhanced_message(&self) -> String;

    /// Get quick fix suggestions
    fn get_quick_fixes(&self) -> Vec<String>;

    /// Check if error is recoverable
    fn is_recoverable(&self) -> bool;

    /// Get context-aware error message based on system state
    fn get_contextual_message(&self, system_info: &SystemInfo) -> String;

    /// Get environment-specific solutions
    fn get_environment_solutions(&self, env: &EnvironmentInfo) -> Vec<Solution>;
}

/// System information for context-aware error messages
#[derive(Debug, Clone)]
pub struct SystemInfo {
    /// Operating system
    pub os: String,
    /// Architecture (x86_64, aarch64, etc.)
    pub arch: String,
    /// Available memory in MB
    pub available_memory_mb: u64,
    /// CPU count
    pub cpu_count: usize,
    /// Available disk space in MB
    pub available_disk_mb: u64,
    /// GPU availability
    pub has_gpu: bool,
    /// Network connectivity
    pub has_network: bool,
}

/// Environment information for targeted solutions
#[derive(Debug, Clone)]
pub struct EnvironmentInfo {
    /// Development vs production environment
    pub environment_type: EnvironmentType,
    /// Container environment (Docker, Kubernetes, etc.)
    pub container_type: Option<String>,
    /// Cloud provider (AWS, GCP, Azure, etc.)
    pub cloud_provider: Option<String>,
    /// Programming language integration (Python, JavaScript, etc.)
    pub language_binding: Option<String>,
    /// Framework integration (Flask, FastAPI, etc.)
    pub framework: Option<String>,
}

/// Environment type classification
#[derive(Debug, Clone, PartialEq)]
pub enum EnvironmentType {
    /// Development environment
    Development,
    /// Testing environment
    Testing,
    /// Staging environment
    Staging,
    /// Production environment
    Production,
    /// CI/CD environment
    CI,
}

impl Default for SystemInfo {
    fn default() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            available_memory_mb: 8192, // Default estimate
            cpu_count: num_cpus::get(),
            available_disk_mb: 10240, // Default estimate
            has_gpu: false,           // Conservative default
            has_network: true,        // Optimistic default
        }
    }
}

impl Default for EnvironmentInfo {
    fn default() -> Self {
        Self {
            environment_type: EnvironmentType::Development,
            container_type: None,
            cloud_provider: None,
            language_binding: None,
            framework: None,
        }
    }
}

impl ErrorEnhancer for RecognitionError {
    fn enhance_error(&self) -> ErrorEnhancement {
        match self {
            RecognitionError::ModelLoadError { message, .. } => {
                create_model_load_enhancement(message)
            }
            RecognitionError::ModelError { message, .. } => create_model_error_enhancement(message),
            RecognitionError::AudioProcessingError { message, .. } => {
                create_audio_processing_enhancement(message)
            }
            RecognitionError::TranscriptionError { message, .. } => {
                create_transcription_enhancement(message)
            }
            RecognitionError::PhonemeRecognitionError { message, .. } => {
                create_phoneme_recognition_enhancement(message)
            }
            RecognitionError::AudioAnalysisError { message, .. } => {
                create_audio_analysis_enhancement(message)
            }
            RecognitionError::ConfigurationError { message, .. } => {
                create_configuration_enhancement(message)
            }
            RecognitionError::InsufficientMemory {
                required_mb,
                available_mb,
                ..
            } => create_memory_error_enhancement(&format!(
                "Insufficient memory: {required_mb} MB required, {available_mb} MB available"
            )),
            RecognitionError::MemoryError { message, .. } => {
                create_memory_error_enhancement(message)
            }
            RecognitionError::UnsupportedFormat(format) => create_format_error_enhancement(format),
            RecognitionError::FeatureNotSupported { feature, .. } => {
                create_feature_not_supported_enhancement(feature)
            }
            RecognitionError::InvalidInput { message, .. } => {
                create_invalid_input_enhancement(message)
            }
            RecognitionError::ResourceError { message, .. } => {
                create_resource_error_enhancement(message)
            }
            RecognitionError::InvalidFormat(format) => create_invalid_format_enhancement(format),
            RecognitionError::ModelNotFound {
                model,
                available,
                suggestions,
            } => create_model_not_found_enhancement(model, available, suggestions),
            RecognitionError::LanguageNotSupported {
                language,
                supported,
                suggestions,
            } => create_language_not_supported_enhancement(language, supported, suggestions),
            RecognitionError::DeviceNotAvailable {
                device,
                reason,
                fallback,
            } => create_device_not_available_enhancement(device, reason, fallback),
            RecognitionError::RecognitionTimeout {
                timeout_ms,
                audio_duration_ms,
                suggestion,
            } => {
                create_recognition_timeout_enhancement(*timeout_ms, *audio_duration_ms, suggestion)
            }
            RecognitionError::TrainingError { message, .. } => {
                create_training_error_enhancement(message)
            }
        }
    }

    fn get_enhanced_message(&self) -> String {
        let enhancement = self.enhance_error();
        format_enhanced_error(&enhancement)
    }

    fn get_quick_fixes(&self) -> Vec<String> {
        let enhancement = self.enhance_error();
        enhancement
            .solutions
            .iter()
            .filter(|s| s.priority <= 2 && s.difficulty == SolutionDifficulty::Easy)
            .map(|s| s.title.clone())
            .collect()
    }

    fn is_recoverable(&self) -> bool {
        match self {
            RecognitionError::ModelLoadError { .. } => true,
            RecognitionError::AudioProcessingError { .. } => true,
            RecognitionError::ConfigurationError { .. } => true,
            RecognitionError::InvalidInput { .. } => true,
            RecognitionError::ResourceError { .. } => true,
            RecognitionError::UnsupportedFormat(_) => true,
            RecognitionError::InvalidFormat(_) => true,
            RecognitionError::ModelNotFound { .. } => true,
            RecognitionError::LanguageNotSupported { .. } => true,
            RecognitionError::DeviceNotAvailable { .. } => true,
            RecognitionError::RecognitionTimeout { .. } => true,
            RecognitionError::InsufficientMemory { .. } => false,
            RecognitionError::MemoryError { .. } => false,
            _ => false,
        }
    }

    fn get_contextual_message(&self, system_info: &SystemInfo) -> String {
        let base_enhancement = self.enhance_error();
        let mut contextual_message = format!(
            "Error: {message}\n",
            message = base_enhancement.original_message
        );

        // Add system context
        contextual_message.push_str(&format!(
            "System Context: {} {} ({} cores, {}MB RAM available)\n",
            system_info.os,
            system_info.arch,
            system_info.cpu_count,
            system_info.available_memory_mb
        ));

        // Add specific context based on error type
        match self {
            RecognitionError::InsufficientMemory {
                required_mb,
                available_mb,
                ..
            } => {
                contextual_message.push_str(&format!(
                    "Memory Requirements: {}MB required vs {}MB available (system reports {}MB)\n",
                    required_mb, available_mb, system_info.available_memory_mb
                ));

                if system_info.available_memory_mb < *required_mb {
                    contextual_message
                        .push_str("⚠️  System memory may be insufficient for this model size.\n");
                }
            }
            RecognitionError::DeviceNotAvailable { device, .. } => {
                if device.to_lowercase().contains("gpu") && !system_info.has_gpu {
                    contextual_message.push_str(
                        "ℹ️  No GPU detected on this system. Consider using CPU-only mode.\n",
                    );
                }
            }
            RecognitionError::ModelLoadError { .. } => {
                if system_info.available_disk_mb < 1000 {
                    contextual_message.push_str("⚠️  Low disk space detected. Consider freeing up space for model storage.\n");
                }
                if !system_info.has_network {
                    contextual_message.push_str(
                        "⚠️  No network connectivity detected. Model download may fail.\n",
                    );
                }
            }
            _ => {}
        }

        // Add prioritized solutions
        let solutions = base_enhancement.solutions;
        if !solutions.is_empty() {
            contextual_message.push_str("\nRecommended Solutions:\n");
            for (i, solution) in solutions.iter().take(3).enumerate() {
                contextual_message.push_str(&format!(
                    "{}. {} ({})\n   {}\n",
                    i + 1,
                    solution.title,
                    solution.estimated_time,
                    solution.description
                ));
            }
        }

        contextual_message
    }

    fn get_environment_solutions(&self, env: &EnvironmentInfo) -> Vec<Solution> {
        let mut solutions = Vec::new();

        match self {
            RecognitionError::ModelLoadError { .. } => {
                match env.environment_type {
                    EnvironmentType::Production => {
                        solutions.push(Solution {
                            title: "Production model deployment check".to_string(),
                            description: "Verify model deployment in production environment"
                                .to_string(),
                            priority: 1,
                            estimated_time: "5-10 minutes".to_string(),
                            difficulty: SolutionDifficulty::Moderate,
                            steps: vec![
                                "Check if model is included in production build".to_string(),
                                "Verify model path configuration for production".to_string(),
                                "Check production filesystem permissions".to_string(),
                                "Validate model integrity in deployment".to_string(),
                            ],
                            code_example: Some(
                                r#"
# Check model in production environment
if [ ! -f "/app/models/whisper-base.bin" ]; then
    echo "Model missing in production"
    exit 1
fi

# Verify permissions
ls -la /app/models/whisper-base.bin
"#
                                .to_string(),
                            ),
                            success_indicators: vec![
                                "Model file exists in production path".to_string(),
                                "Correct permissions set".to_string(),
                            ],
                        });
                    }
                    EnvironmentType::Development => {
                        solutions.push(Solution {
                            title: "Development environment setup".to_string(),
                            description: "Set up model for local development".to_string(),
                            priority: 1,
                            estimated_time: "2-5 minutes".to_string(),
                            difficulty: SolutionDifficulty::Easy,
                            steps: vec![
                                "Download model to local development directory".to_string(),
                                "Set up development configuration".to_string(),
                                "Add model path to environment variables".to_string(),
                            ],
                            code_example: Some(
                                r#"
# Set up development environment
mkdir -p ./models
export VOIRS_MODEL_PATH="./models/whisper-base.bin"

# Download model (example)
curl -L "https://example.com/model.bin" -o ./models/whisper-base.bin
"#
                                .to_string(),
                            ),
                            success_indicators: vec![
                                "Model downloaded to development directory".to_string(),
                                "Environment variables configured".to_string(),
                            ],
                        });
                    }
                    EnvironmentType::CI => {
                        solutions.push(Solution {
                            title: "CI/CD model handling".to_string(),
                            description: "Configure model access in CI environment".to_string(),
                            priority: 1,
                            estimated_time: "10-15 minutes".to_string(),
                            difficulty: SolutionDifficulty::Moderate,
                            steps: vec![
                                "Add model to CI artifacts or cache".to_string(),
                                "Configure CI environment variables".to_string(),
                                "Set up model download in CI pipeline".to_string(),
                                "Add model validation step".to_string(),
                            ],
                            code_example: Some(
                                r#"
# CI configuration (GitHub Actions example)
- name: Setup models
  run: |
    mkdir -p models
    if [ ! -f models/whisper-base.bin ]; then
      curl -L "$MODEL_URL" -o models/whisper-base.bin
    fi
  env:
    MODEL_URL: ${{ secrets.MODEL_URL }}
"#
                                .to_string(),
                            ),
                            success_indicators: vec![
                                "Models cached in CI".to_string(),
                                "Pipeline completes successfully".to_string(),
                            ],
                        });
                    }
                    _ => {}
                }

                // Add container-specific solutions
                if let Some(container_type) = &env.container_type {
                    match container_type.as_str() {
                        "docker" => {
                            solutions.push(Solution {
                                title: "Docker container model setup".to_string(),
                                description: "Configure model access in Docker container"
                                    .to_string(),
                                priority: 2,
                                estimated_time: "5-10 minutes".to_string(),
                                difficulty: SolutionDifficulty::Moderate,
                                steps: vec![
                                    "Add model to Docker image or volume mount".to_string(),
                                    "Set correct file permissions in container".to_string(),
                                    "Configure model path environment variables".to_string(),
                                ],
                                code_example: Some(
                                    r"
# Dockerfile
COPY models/ /app/models/
RUN chmod -R 644 /app/models/

# docker-compose.yml
volumes:
  - ./models:/app/models:ro

# Environment
ENV VOIRS_MODEL_PATH=/app/models/whisper-base.bin
"
                                    .to_string(),
                                ),
                                success_indicators: vec![
                                    "Model accessible in container".to_string(),
                                    "Permissions correctly set".to_string(),
                                ],
                            });
                        }
                        "kubernetes" => {
                            solutions.push(Solution {
                                title: "Kubernetes model deployment".to_string(),
                                description: "Deploy model in Kubernetes environment".to_string(),
                                priority: 2,
                                estimated_time: "15-20 minutes".to_string(),
                                difficulty: SolutionDifficulty::Advanced,
                                steps: vec![
                                    "Create ConfigMap or Secret for model".to_string(),
                                    "Mount model in Pod specification".to_string(),
                                    "Set up persistent volume if needed".to_string(),
                                    "Configure resource limits".to_string(),
                                ],
                                code_example: Some(
                                    r#"
# model-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: voirs-models
data:
  model-path: "/models/whisper-base.bin"

# deployment.yaml
spec:
  template:
    spec:
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: model-pvc
      containers:
      - name: voirs
        volumeMounts:
        - name: model-volume
          mountPath: /models
"#
                                    .to_string(),
                                ),
                                success_indicators: vec![
                                    "Model accessible in all pods".to_string(),
                                    "Persistent storage configured".to_string(),
                                ],
                            });
                        }
                        _ => {}
                    }
                }
            }
            _ => {
                // Add general environment-specific solutions for other error types
                solutions.extend(self.enhance_error().solutions);
            }
        }

        solutions
    }
}

// Helper functions for creating specific error enhancements

fn create_model_load_enhancement(message: &str) -> ErrorEnhancement {
    ErrorEnhancement {
        original_message: message.to_string(),
        category: ErrorCategory::ModelIssues,
        severity: ErrorSeverity::Critical,
        context: ErrorContext {
            component: "Model Loader".to_string(),
            operation: "Loading recognition model".to_string(),
            input_summary: "Model path and configuration".to_string(),
            system_state: "Model initialization".to_string(),
            timestamp: std::time::SystemTime::now(),
            additional_info: std::collections::HashMap::new(),
        },
        solutions: vec![
            Solution {
                title: "Check model file existence and permissions".to_string(),
                description: "Verify that the model file exists and is readable".to_string(),
                priority: 1,
                estimated_time: "1-2 minutes".to_string(),
                difficulty: SolutionDifficulty::Easy,
                steps: vec![
                    "Check if the model file exists at the specified path".to_string(),
                    "Verify file permissions (should be readable)".to_string(),
                    "Check available disk space".to_string(),
                    "Ensure model file is not corrupted".to_string(),
                ],
                code_example: Some(
                    r#"
use std::fs;
use std::path::Path;

// Check model file
if !Path::new("path/to/model.bin").exists() {
    println!("Model file not found!");
}

// Check permissions
if let Ok(metadata) = fs::metadata("path/to/model.bin") {
    if metadata.permissions().readonly() {
        println!("Model file is read-only");
    }
}
"#
                    .to_string(),
                ),
                success_indicators: vec![
                    "Model file exists and is readable".to_string(),
                    "No permission errors".to_string(),
                ],
            },
            Solution {
                title: "Download or regenerate model".to_string(),
                description: "Download a fresh copy of the model or regenerate it".to_string(),
                priority: 2,
                estimated_time: "5-10 minutes".to_string(),
                difficulty: SolutionDifficulty::Easy,
                steps: vec![
                    "Download model from official source".to_string(),
                    "Verify model checksum".to_string(),
                    "Place model in correct directory".to_string(),
                    "Update model path in configuration".to_string(),
                ],
                code_example: Some(
                    r#"
// Re-download model
let model_url = "https://example.com/model.bin";
let response = reqwest::get(model_url).await?;
let model_data = response.bytes().await?;
fs::write("models/model.bin", model_data)?;
"#
                    .to_string(),
                ),
                success_indicators: vec![
                    "Model downloads successfully".to_string(),
                    "Checksum verification passes".to_string(),
                ],
            },
        ],
        documentation_links: vec![
            "https://docs.voirs.ai/models/loading".to_string(),
            "https://docs.voirs.ai/troubleshooting/model-errors".to_string(),
        ],
        troubleshooting_steps: vec![
            "Check system logs for more details".to_string(),
            "Verify model format compatibility".to_string(),
            "Test with a smaller test model first".to_string(),
            "Check memory availability".to_string(),
        ],
    }
}

fn create_audio_processing_enhancement(message: &str) -> ErrorEnhancement {
    ErrorEnhancement {
        original_message: message.to_string(),
        category: ErrorCategory::AudioFormat,
        severity: ErrorSeverity::High,
        context: ErrorContext {
            component: "Audio Processor".to_string(),
            operation: "Processing audio input".to_string(),
            input_summary: "Audio file or stream".to_string(),
            system_state: "Audio processing pipeline".to_string(),
            timestamp: std::time::SystemTime::now(),
            additional_info: std::collections::HashMap::new(),
        },
        solutions: vec![
            Solution {
                title: "Verify audio format compatibility".to_string(),
                description: "Check if the audio format is supported".to_string(),
                priority: 1,
                estimated_time: "1-2 minutes".to_string(),
                difficulty: SolutionDifficulty::Easy,
                steps: vec![
                    "Check audio file format (WAV, MP3, FLAC, etc.)".to_string(),
                    "Verify sample rate (16kHz recommended)".to_string(),
                    "Check bit depth (16-bit recommended)".to_string(),
                    "Ensure audio is not corrupted".to_string(),
                ],
                code_example: Some(
                    r#"
use voirs_recognizer::audio_formats::{load_audio, AudioFormat};

// Load and inspect audio
match load_audio("path/to/audio.wav") {
    Ok(audio) => {
        println!("Sample rate: {}", audio.sample_rate);
        println!("Channels: {}", audio.channels);
        println!("Duration: {:.2}s", audio.duration_seconds);
    }
    Err(e) => println!("Audio loading error: {}", e),
}
"#
                    .to_string(),
                ),
                success_indicators: vec![
                    "Audio format is supported".to_string(),
                    "Sample rate is compatible".to_string(),
                    "Audio loads without errors".to_string(),
                ],
            },
            Solution {
                title: "Convert audio to supported format".to_string(),
                description: "Convert audio to a supported format and sample rate".to_string(),
                priority: 2,
                estimated_time: "2-5 minutes".to_string(),
                difficulty: SolutionDifficulty::Moderate,
                steps: vec![
                    "Convert to WAV format".to_string(),
                    "Resample to 16kHz".to_string(),
                    "Convert to mono if needed".to_string(),
                    "Ensure 16-bit depth".to_string(),
                ],
                code_example: Some(
                    r#"
use voirs_recognizer::audio_formats::load_audio_with_sample_rate;

// Load with automatic resampling
let audio = load_audio_with_sample_rate("input.mp3", 16000)?;
"#
                    .to_string(),
                ),
                success_indicators: vec![
                    "Audio converts successfully".to_string(),
                    "Sample rate is 16kHz".to_string(),
                    "Audio quality is preserved".to_string(),
                ],
            },
        ],
        documentation_links: vec![
            "https://docs.voirs.ai/audio/formats".to_string(),
            "https://docs.voirs.ai/audio/preprocessing".to_string(),
        ],
        troubleshooting_steps: vec![
            "Test with a simple WAV file".to_string(),
            "Check audio file integrity".to_string(),
            "Verify codec support".to_string(),
            "Test with different sample rates".to_string(),
        ],
    }
}

fn create_configuration_enhancement(message: &str) -> ErrorEnhancement {
    ErrorEnhancement {
        original_message: message.to_string(),
        category: ErrorCategory::Configuration,
        severity: ErrorSeverity::Medium,
        context: ErrorContext {
            component: "Configuration Manager".to_string(),
            operation: "Loading configuration".to_string(),
            input_summary: "Configuration file or parameters".to_string(),
            system_state: "System initialization".to_string(),
            timestamp: std::time::SystemTime::now(),
            additional_info: std::collections::HashMap::new(),
        },
        solutions: vec![
            Solution {
                title: "Check configuration file syntax".to_string(),
                description: "Verify configuration file format and syntax".to_string(),
                priority: 1,
                estimated_time: "1-2 minutes".to_string(),
                difficulty: SolutionDifficulty::Easy,
                steps: vec![
                    "Check JSON/YAML syntax".to_string(),
                    "Verify required fields are present".to_string(),
                    "Check data types match expected values".to_string(),
                    "Ensure no invalid characters".to_string(),
                ],
                code_example: Some(
                    r#"
use serde_json;
use std::fs;

// Validate JSON configuration
let config_content = fs::read_to_string("config.json")?;
match serde_json::from_str::<serde_json::Value>(&config_content) {
    Ok(_) => println!("Configuration is valid JSON"),
    Err(e) => println!("JSON error: {}", e),
}
"#
                    .to_string(),
                ),
                success_indicators: vec![
                    "Configuration parses successfully".to_string(),
                    "All required fields are present".to_string(),
                    "No syntax errors".to_string(),
                ],
            },
            Solution {
                title: "Use default configuration".to_string(),
                description: "Generate and use a default configuration".to_string(),
                priority: 2,
                estimated_time: "1 minute".to_string(),
                difficulty: SolutionDifficulty::Easy,
                steps: vec![
                    "Generate default configuration".to_string(),
                    "Save to configuration file".to_string(),
                    "Customize as needed".to_string(),
                    "Test with default settings".to_string(),
                ],
                code_example: Some(
                    r#"
use voirs_recognizer::ASRConfig;

// Create default configuration
let config = ASRConfig::default();
let config_json = serde_json::to_string_pretty(&config)?;
std::fs::write("config.json", config_json)?;
"#
                    .to_string(),
                ),
                success_indicators: vec![
                    "Default configuration loads successfully".to_string(),
                    "System initializes with default settings".to_string(),
                ],
            },
        ],
        documentation_links: vec![
            "https://docs.voirs.ai/configuration".to_string(),
            "https://docs.voirs.ai/configuration/examples".to_string(),
        ],
        troubleshooting_steps: vec![
            "Check configuration file permissions".to_string(),
            "Validate against schema".to_string(),
            "Test with minimal configuration".to_string(),
            "Compare with working examples".to_string(),
        ],
    }
}

fn create_memory_error_enhancement(message: &str) -> ErrorEnhancement {
    ErrorEnhancement {
        original_message: message.to_string(),
        category: ErrorCategory::Resources,
        severity: ErrorSeverity::Critical,
        context: ErrorContext {
            component: "Memory Manager".to_string(),
            operation: "Memory allocation for processing".to_string(),
            input_summary: "Large audio files or models".to_string(),
            system_state: "High memory usage".to_string(),
            timestamp: std::time::SystemTime::now(),
            additional_info: std::collections::HashMap::new(),
        },
        solutions: vec![
            Solution {
                title: "Reduce batch size and enable streaming".to_string(),
                description: "Process audio in smaller chunks to reduce memory usage".to_string(),
                priority: 1,
                estimated_time: "1-2 minutes".to_string(),
                difficulty: SolutionDifficulty::Easy,
                steps: vec![
                    "Enable streaming mode in configuration".to_string(),
                    "Reduce batch size to 1-4 samples".to_string(),
                    "Use memory-efficient model variants".to_string(),
                    "Clear cache between processing".to_string(),
                ],
                code_example: Some(
                    r"
use voirs_recognizer::{ASRConfig, StreamingConfig};

let mut config = ASRConfig::default();
config.streaming = Some(StreamingConfig {
    chunk_length_ms: 30000,  // 30 second chunks
    overlap_ms: 3000,        // 3 second overlap
    ..Default::default()
});
config.batch_size = 1;  // Process one sample at a time
"
                    .to_string(),
                ),
                success_indicators: vec![
                    "Memory usage stays below 2GB".to_string(),
                    "Processing completes without OOM errors".to_string(),
                    "Audio quality is preserved".to_string(),
                ],
            },
            Solution {
                title: "Use quantized models".to_string(),
                description: "Switch to quantized model variants that use less memory".to_string(),
                priority: 2,
                estimated_time: "2-3 minutes".to_string(),
                difficulty: SolutionDifficulty::Moderate,
                steps: vec![
                    "Download quantized model variant".to_string(),
                    "Update model path in configuration".to_string(),
                    "Enable quantization in runtime config".to_string(),
                    "Test accuracy on sample data".to_string(),
                ],
                code_example: Some(
                    r#"
let mut config = ASRConfig::default();
config.model_path = "models/whisper-base-q8.bin".to_string();
config.quantization_enabled = true;
"#
                    .to_string(),
                ),
                success_indicators: vec![
                    "Model loads with reduced memory footprint".to_string(),
                    "Inference speed is maintained or improved".to_string(),
                    "Acceptable accuracy for use case".to_string(),
                ],
            },
        ],
        documentation_links: vec![
            "https://docs.voirs.ai/memory/optimization".to_string(),
            "https://docs.voirs.ai/models/quantization".to_string(),
        ],
        troubleshooting_steps: vec![
            "Monitor system memory usage".to_string(),
            "Profile memory allocation patterns".to_string(),
            "Test with progressively smaller inputs".to_string(),
            "Check for memory leaks".to_string(),
        ],
    }
}

fn create_device_error_enhancement(message: &str) -> ErrorEnhancement {
    ErrorEnhancement {
        original_message: message.to_string(),
        category: ErrorCategory::Resources,
        severity: ErrorSeverity::High,
        context: ErrorContext {
            component: "Device Manager".to_string(),
            operation: "Device initialization and selection".to_string(),
            input_summary: "GPU/CPU device configuration".to_string(),
            system_state: "Device enumeration".to_string(),
            timestamp: std::time::SystemTime::now(),
            additional_info: std::collections::HashMap::new(),
        },
        solutions: vec![
            Solution {
                title: "Fallback to CPU processing".to_string(),
                description: "Use CPU when GPU is unavailable or incompatible".to_string(),
                priority: 1,
                estimated_time: "30 seconds".to_string(),
                difficulty: SolutionDifficulty::Easy,
                steps: vec![
                    "Set device type to CPU in configuration".to_string(),
                    "Disable GPU acceleration".to_string(),
                    "Verify CPU meets minimum requirements".to_string(),
                    "Test with CPU-optimized settings".to_string(),
                ],
                code_example: Some(
                    r"
let mut config = ASRConfig::default();
config.device_type = DeviceType::Cpu;
config.gpu_enabled = false;
config.num_threads = std::thread::available_parallelism()?.get();
"
                    .to_string(),
                ),
                success_indicators: vec![
                    "Processing works on CPU".to_string(),
                    "No device-related errors".to_string(),
                    "Reasonable processing speed".to_string(),
                ],
            },
            Solution {
                title: "Update GPU drivers and check compatibility".to_string(),
                description: "Ensure GPU drivers are up-to-date and compatible".to_string(),
                priority: 2,
                estimated_time: "10-15 minutes".to_string(),
                difficulty: SolutionDifficulty::Moderate,
                steps: vec![
                    "Check GPU compatibility list".to_string(),
                    "Update GPU drivers to latest version".to_string(),
                    "Verify CUDA/OpenCL installation".to_string(),
                    "Test GPU functionality".to_string(),
                ],
                code_example: Some(
                    r#"
// Check device availability
use candle_core::Device;

match Device::cuda_if_available(0) {
    Ok(device) => println!("GPU available: {:?}", device),
    Err(e) => println!("GPU not available: {}", e),
}
"#
                    .to_string(),
                ),
                success_indicators: vec![
                    "GPU device is detected".to_string(),
                    "No driver compatibility issues".to_string(),
                    "GPU memory is accessible".to_string(),
                ],
            },
        ],
        documentation_links: vec![
            "https://docs.voirs.ai/setup/gpu".to_string(),
            "https://docs.voirs.ai/troubleshooting/devices".to_string(),
        ],
        troubleshooting_steps: vec![
            "Check device manager for GPU".to_string(),
            "Test with different device indices".to_string(),
            "Verify compute capability".to_string(),
            "Check for conflicting GPU processes".to_string(),
        ],
    }
}

fn create_format_error_enhancement(format: &str) -> ErrorEnhancement {
    ErrorEnhancement {
        original_message: format!("Unsupported audio format: {format}"),
        category: ErrorCategory::AudioFormat,
        severity: ErrorSeverity::Medium,
        context: ErrorContext {
            component: "Audio Format Detector".to_string(),
            operation: "Audio format validation".to_string(),
            input_summary: format!("Audio file with format: {format}"),
            system_state: "Format detection".to_string(),
            timestamp: std::time::SystemTime::now(),
            additional_info: std::collections::HashMap::new(),
        },
        solutions: vec![Solution {
            title: "Convert to supported format".to_string(),
            description: "Convert audio to WAV, FLAC, MP3, or OGG format".to_string(),
            priority: 1,
            estimated_time: "1-3 minutes".to_string(),
            difficulty: SolutionDifficulty::Easy,
            steps: vec![
                "Use FFmpeg or similar tool to convert".to_string(),
                "Convert to WAV format (recommended)".to_string(),
                "Ensure 16kHz sample rate".to_string(),
                "Use mono channel if possible".to_string(),
            ],
            code_example: Some(format!(
                r#"
// Convert using FFmpeg command line
// ffmpeg -i input.{} -ar 16000 -ac 1 output.wav

// Or use the built-in audio loader with conversion
use voirs_recognizer::audio_formats::load_audio;

let audio = load_audio("input.wav")?;  // Will auto-convert
"#,
                format
            )),
            success_indicators: vec![
                "Audio converts without quality loss".to_string(),
                "Converted file loads successfully".to_string(),
                "Recognition works with converted audio".to_string(),
            ],
        }],
        documentation_links: vec![
            "https://docs.voirs.ai/audio/supported-formats".to_string(),
            "https://docs.voirs.ai/audio/conversion".to_string(),
        ],
        troubleshooting_steps: vec![
            "Check file extension matches content".to_string(),
            "Verify file is not corrupted".to_string(),
            "Test with simple WAV file first".to_string(),
            "Check codec requirements".to_string(),
        ],
    }
}

// Additional helper functions for other error types...

fn create_model_error_enhancement(message: &str) -> ErrorEnhancement {
    // Implementation for model errors
    ErrorEnhancement {
        original_message: message.to_string(),
        category: ErrorCategory::ModelIssues,
        severity: ErrorSeverity::High,
        context: ErrorContext {
            component: "Model Runtime".to_string(),
            operation: "Model inference".to_string(),
            input_summary: "Model input data".to_string(),
            system_state: "Model execution".to_string(),
            timestamp: std::time::SystemTime::now(),
            additional_info: std::collections::HashMap::new(),
        },
        solutions: vec![Solution {
            title: "Check model compatibility".to_string(),
            description: "Verify model is compatible with current version".to_string(),
            priority: 1,
            estimated_time: "2-3 minutes".to_string(),
            difficulty: SolutionDifficulty::Easy,
            steps: vec![
                "Check model version".to_string(),
                "Verify system requirements".to_string(),
                "Update to compatible model".to_string(),
            ],
            code_example: None,
            success_indicators: vec![
                "Model version is compatible".to_string(),
                "System requirements met".to_string(),
            ],
        }],
        documentation_links: vec!["https://docs.voirs.ai/models/compatibility".to_string()],
        troubleshooting_steps: vec![
            "Test with different model".to_string(),
            "Check system resources".to_string(),
        ],
    }
}

// Implement the remaining helper functions...
fn create_transcription_enhancement(message: &str) -> ErrorEnhancement {
    // Similar structure for transcription errors
    ErrorEnhancement {
        original_message: message.to_string(),
        category: ErrorCategory::ModelIssues,
        severity: ErrorSeverity::Medium,
        context: ErrorContext {
            component: "Transcription Engine".to_string(),
            operation: "Speech transcription".to_string(),
            input_summary: "Audio for transcription".to_string(),
            system_state: "Transcription processing".to_string(),
            timestamp: std::time::SystemTime::now(),
            additional_info: std::collections::HashMap::new(),
        },
        solutions: vec![Solution {
            title: "Check audio quality".to_string(),
            description: "Ensure audio quality is sufficient for transcription".to_string(),
            priority: 1,
            estimated_time: "1-2 minutes".to_string(),
            difficulty: SolutionDifficulty::Easy,
            steps: vec![
                "Check for background noise".to_string(),
                "Verify speech clarity".to_string(),
                "Ensure adequate volume".to_string(),
            ],
            code_example: None,
            success_indicators: vec![
                "Audio is clear and audible".to_string(),
                "Minimal background noise".to_string(),
            ],
        }],
        documentation_links: vec!["https://docs.voirs.ai/transcription/quality".to_string()],
        troubleshooting_steps: vec![
            "Test with high-quality audio".to_string(),
            "Check microphone settings".to_string(),
        ],
    }
}

// Implement remaining helper functions with similar structure...
fn create_phoneme_recognition_enhancement(message: &str) -> ErrorEnhancement {
    create_default_enhancement(message, ErrorCategory::ModelIssues, "Phoneme Recognition")
}

fn create_audio_analysis_enhancement(message: &str) -> ErrorEnhancement {
    create_default_enhancement(message, ErrorCategory::AudioFormat, "Audio Analysis")
}

fn create_feature_not_supported_enhancement(feature: &str) -> ErrorEnhancement {
    create_default_enhancement(feature, ErrorCategory::FeatureSupport, "Feature Support")
}

fn create_invalid_input_enhancement(message: &str) -> ErrorEnhancement {
    create_default_enhancement(message, ErrorCategory::InputValidation, "Input Validation")
}

fn create_io_error_enhancement(message: &str) -> ErrorEnhancement {
    create_default_enhancement(message, ErrorCategory::Resources, "I/O Operations")
}

fn create_network_error_enhancement(message: &str) -> ErrorEnhancement {
    create_default_enhancement(message, ErrorCategory::Resources, "Network Operations")
}

fn create_timeout_error_enhancement(message: &str) -> ErrorEnhancement {
    create_default_enhancement(message, ErrorCategory::Performance, "Timeout Handling")
}

fn create_concurrency_error_enhancement(message: &str) -> ErrorEnhancement {
    create_default_enhancement(message, ErrorCategory::Integration, "Concurrency")
}

fn create_other_error_enhancement(message: &str) -> ErrorEnhancement {
    create_default_enhancement(message, ErrorCategory::Integration, "General")
}

fn create_resource_error_enhancement(message: &str) -> ErrorEnhancement {
    create_default_enhancement(message, ErrorCategory::Resources, "Resource Management")
}

fn create_unsupported_format_enhancement(format: &str) -> ErrorEnhancement {
    ErrorEnhancement {
        original_message: format!("Unsupported audio format: {format}"),
        category: ErrorCategory::AudioFormat,
        severity: ErrorSeverity::Medium,
        context: ErrorContext {
            component: "Audio Format Handler".to_string(),
            operation: "Audio format detection".to_string(),
            input_summary: format!("Audio format: {format}"),
            system_state: "Format processing".to_string(),
            timestamp: std::time::SystemTime::now(),
            additional_info: std::collections::HashMap::new(),
        },
        solutions: vec![Solution {
            title: "Convert to supported format".to_string(),
            description: "Convert audio to a supported format like WAV or FLAC".to_string(),
            priority: 1,
            estimated_time: "2-5 minutes".to_string(),
            difficulty: SolutionDifficulty::Easy,
            steps: vec![
                "Convert to WAV format".to_string(),
                "Use 16kHz sample rate".to_string(),
                "Ensure 16-bit depth".to_string(),
                "Convert to mono if needed".to_string(),
            ],
            code_example: Some(
                r#"
// Convert using ffmpeg command line
ffmpeg -i input.{format} -ar 16000 -ac 1 -c:a pcm_s16le output.wav

// Or use VoiRS built-in conversion
use voirs_recognizer::audio_formats::load_audio_with_sample_rate;
let audio = load_audio_with_sample_rate("input.{format}", 16000)?;
"#
                .replace("{format}", format),
            ),
            success_indicators: vec![
                "Audio converts successfully".to_string(),
                "VoiRS can process the converted audio".to_string(),
            ],
        }],
        documentation_links: vec![
            "https://docs.voirs.ai/audio/formats".to_string(),
            "https://docs.voirs.ai/audio/conversion".to_string(),
        ],
        troubleshooting_steps: vec![
            "Check supported formats list".to_string(),
            "Verify audio file integrity".to_string(),
            "Test with minimal audio sample".to_string(),
        ],
    }
}

fn create_invalid_format_enhancement(format: &str) -> ErrorEnhancement {
    create_default_enhancement(
        &format!("Invalid audio format: {format}"),
        ErrorCategory::AudioFormat,
        "Audio Format Validation",
    )
}

fn create_model_not_found_enhancement(
    model: &str,
    available: &[String],
    suggestions: &[String],
) -> ErrorEnhancement {
    ErrorEnhancement {
        original_message: format!("Model '{model}' not found. Available models: {available:?}"),
        category: ErrorCategory::ModelIssues,
        severity: ErrorSeverity::High,
        context: ErrorContext {
            component: "Model Manager".to_string(),
            operation: "Model loading".to_string(),
            input_summary: format!("Requested model: {model}"),
            system_state: "Model discovery".to_string(),
            timestamp: std::time::SystemTime::now(),
            additional_info: std::collections::HashMap::new(),
        },
        solutions: vec![Solution {
            title: "Use available model".to_string(),
            description: "Select from available models".to_string(),
            priority: 1,
            estimated_time: "1 minute".to_string(),
            difficulty: SolutionDifficulty::Easy,
            steps: vec![
                format!("Available models: {available:?}"),
                format!("Suggested alternatives: {:?}", suggestions),
                "Update configuration to use available model".to_string(),
            ],
            code_example: Some(format!(
                r#"
// Use available model instead
let config = ASRConfig::default().with_model("{}");
"#,
                available.first().unwrap_or(&"base".to_string())
            )),
            success_indicators: vec![
                "Model loads successfully".to_string(),
                "System initializes properly".to_string(),
            ],
        }],
        documentation_links: vec!["https://docs.voirs.ai/models/available".to_string()],
        troubleshooting_steps: vec![
            "List available models".to_string(),
            "Check model installation".to_string(),
        ],
    }
}

fn create_language_not_supported_enhancement(
    language: &str,
    supported: &[String],
    suggestions: &[String],
) -> ErrorEnhancement {
    ErrorEnhancement {
        original_message: format!(
            "Language '{}' not supported. Supported languages: {:?}",
            language, supported
        ),
        category: ErrorCategory::FeatureSupport,
        severity: ErrorSeverity::Medium,
        context: ErrorContext {
            component: "Language Support".to_string(),
            operation: "Language validation".to_string(),
            input_summary: format!("Requested language: {}", language),
            system_state: "Language processing".to_string(),
            timestamp: std::time::SystemTime::now(),
            additional_info: std::collections::HashMap::new(),
        },
        solutions: vec![Solution {
            title: "Use supported language".to_string(),
            description: "Select from supported languages".to_string(),
            priority: 1,
            estimated_time: "1 minute".to_string(),
            difficulty: SolutionDifficulty::Easy,
            steps: vec![
                format!("Supported languages: {:?}", supported),
                format!("Suggested alternatives: {:?}", suggestions),
                "Update configuration to use supported language".to_string(),
            ],
            code_example: Some(format!(
                r#"
// Use supported language instead
let config = ASRConfig::default().with_language("{}");
"#,
                supported.first().unwrap_or(&"en".to_string())
            )),
            success_indicators: vec![
                "Language is recognized".to_string(),
                "Processing continues successfully".to_string(),
            ],
        }],
        documentation_links: vec!["https://docs.voirs.ai/languages/supported".to_string()],
        troubleshooting_steps: vec![
            "Check language code format".to_string(),
            "Verify language pack installation".to_string(),
        ],
    }
}

fn create_device_not_available_enhancement(
    device: &str,
    reason: &str,
    fallback: &str,
) -> ErrorEnhancement {
    ErrorEnhancement {
        original_message: format!(
            "Device '{}' not available: {}. Fallback: {}",
            device, reason, fallback
        ),
        category: ErrorCategory::Resources,
        severity: ErrorSeverity::Medium,
        context: ErrorContext {
            component: "Device Manager".to_string(),
            operation: "Device initialization".to_string(),
            input_summary: format!("Requested device: {}", device),
            system_state: "Device discovery".to_string(),
            timestamp: std::time::SystemTime::now(),
            additional_info: std::collections::HashMap::new(),
        },
        solutions: vec![Solution {
            title: "Use fallback device".to_string(),
            description: format!("Use fallback device: {}", fallback),
            priority: 1,
            estimated_time: "Immediate".to_string(),
            difficulty: SolutionDifficulty::Easy,
            steps: vec![
                format!("Fallback device: {}", fallback),
                "System will automatically use fallback".to_string(),
                "Performance may be reduced".to_string(),
            ],
            code_example: Some(format!(
                r#"
// Configure fallback device
let config = ASRConfig::default().with_device("{}");
"#,
                fallback
            )),
            success_indicators: vec![
                "Processing continues with fallback".to_string(),
                "No device errors occur".to_string(),
            ],
        }],
        documentation_links: vec!["https://docs.voirs.ai/devices/configuration".to_string()],
        troubleshooting_steps: vec![
            "Check device availability".to_string(),
            "Verify device drivers".to_string(),
            "Test with CPU fallback".to_string(),
        ],
    }
}

fn create_insufficient_memory_enhancement(
    required_mb: u64,
    available_mb: u64,
    recommendation: &str,
) -> ErrorEnhancement {
    ErrorEnhancement {
        original_message: format!(
            "Insufficient memory: need {}MB, have {}MB. Recommendation: {}",
            required_mb, available_mb, recommendation
        ),
        category: ErrorCategory::Resources,
        severity: ErrorSeverity::High,
        context: ErrorContext {
            component: "Memory Manager".to_string(),
            operation: "Memory allocation".to_string(),
            input_summary: format!("Required: {}MB, Available: {}MB", required_mb, available_mb),
            system_state: "Memory allocation".to_string(),
            timestamp: std::time::SystemTime::now(),
            additional_info: std::collections::HashMap::new(),
        },
        solutions: vec![Solution {
            title: "Free up memory".to_string(),
            description: "Close other applications and free up memory".to_string(),
            priority: 1,
            estimated_time: "2-5 minutes".to_string(),
            difficulty: SolutionDifficulty::Easy,
            steps: vec![
                "Close unnecessary applications".to_string(),
                "Clear system cache".to_string(),
                "Restart the application".to_string(),
                recommendation.to_string(),
            ],
            code_example: Some(
                r#"
// Use smaller model to reduce memory usage
let config = ASRConfig::default().with_model_size("small");
"#
                .to_string(),
            ),
            success_indicators: vec![
                "Sufficient memory available".to_string(),
                "Model loads successfully".to_string(),
            ],
        }],
        documentation_links: vec!["https://docs.voirs.ai/performance/memory".to_string()],
        troubleshooting_steps: vec![
            "Check system memory usage".to_string(),
            "Use memory profiler".to_string(),
            "Consider smaller model".to_string(),
        ],
    }
}

fn create_recognition_timeout_enhancement(
    timeout_ms: u64,
    audio_duration_ms: u64,
    suggestion: &str,
) -> ErrorEnhancement {
    ErrorEnhancement {
        original_message: format!(
            "Recognition timed out after {}ms. Audio duration: {}ms. Suggestion: {}",
            timeout_ms, audio_duration_ms, suggestion
        ),
        category: ErrorCategory::Performance,
        severity: ErrorSeverity::Medium,
        context: ErrorContext {
            component: "Recognition Engine".to_string(),
            operation: "Speech recognition".to_string(),
            input_summary: format!("Timeout: {}ms, Audio: {}ms", timeout_ms, audio_duration_ms),
            system_state: "Recognition processing".to_string(),
            timestamp: std::time::SystemTime::now(),
            additional_info: std::collections::HashMap::new(),
        },
        solutions: vec![Solution {
            title: "Increase timeout".to_string(),
            description: "Increase recognition timeout for longer audio".to_string(),
            priority: 1,
            estimated_time: "1 minute".to_string(),
            difficulty: SolutionDifficulty::Easy,
            steps: vec![
                "Increase timeout setting".to_string(),
                suggestion.to_string(),
                "Test with new timeout".to_string(),
            ],
            code_example: Some(
                r"
// Increase timeout
let config = ASRConfig::default().with_timeout_ms(30000); // 30 seconds
"
                .to_string(),
            ),
            success_indicators: vec![
                "Recognition completes within timeout".to_string(),
                "No timeout errors occur".to_string(),
            ],
        }],
        documentation_links: vec!["https://docs.voirs.ai/performance/timeouts".to_string()],
        troubleshooting_steps: vec![
            "Check audio length".to_string(),
            "Monitor processing time".to_string(),
            "Test with shorter audio".to_string(),
        ],
    }
}

fn create_default_enhancement(
    message: &str,
    category: ErrorCategory,
    component: &str,
) -> ErrorEnhancement {
    ErrorEnhancement {
        original_message: message.to_string(),
        category,
        severity: ErrorSeverity::Medium,
        context: ErrorContext {
            component: component.to_string(),
            operation: "Processing".to_string(),
            input_summary: "User input".to_string(),
            system_state: "Runtime".to_string(),
            timestamp: std::time::SystemTime::now(),
            additional_info: std::collections::HashMap::new(),
        },
        solutions: vec![Solution {
            title: "Check system logs".to_string(),
            description: "Review system logs for more details".to_string(),
            priority: 1,
            estimated_time: "1-2 minutes".to_string(),
            difficulty: SolutionDifficulty::Easy,
            steps: vec![
                "Check application logs".to_string(),
                "Look for related error messages".to_string(),
                "Check system resource usage".to_string(),
            ],
            code_example: None,
            success_indicators: vec![
                "Logs provide additional context".to_string(),
                "Root cause identified".to_string(),
            ],
        }],
        documentation_links: vec!["https://docs.voirs.ai/troubleshooting".to_string()],
        troubleshooting_steps: vec![
            "Check recent system changes".to_string(),
            "Test with minimal configuration".to_string(),
            "Verify system requirements".to_string(),
        ],
    }
}

fn format_enhanced_error(enhancement: &ErrorEnhancement) -> String {
    let mut output = String::new();

    output.push_str(&format!(
        "🚨 {} Error: {}\n",
        match enhancement.severity {
            ErrorSeverity::Critical => "CRITICAL",
            ErrorSeverity::High => "HIGH",
            ErrorSeverity::Medium => "MEDIUM",
            ErrorSeverity::Low => "LOW",
            ErrorSeverity::Info => "INFO",
        },
        enhancement.original_message
    ));

    output.push_str(&format!("📂 Category: {:?}\n", enhancement.category));
    output.push_str(&format!(
        "⚙️  Component: {}\n",
        enhancement.context.component
    ));
    output.push_str(&format!(
        "🔧 Operation: {}\n",
        enhancement.context.operation
    ));

    if !enhancement.solutions.is_empty() {
        output.push_str("\n💡 Suggested Solutions:\n");
        for (i, solution) in enhancement.solutions.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {} ({} - {})\n",
                i + 1,
                solution.title,
                solution.estimated_time,
                match solution.difficulty {
                    SolutionDifficulty::Easy => "Easy",
                    SolutionDifficulty::Moderate => "Moderate",
                    SolutionDifficulty::Advanced => "Advanced",
                }
            ));
            output.push_str(&format!("     {}\n", solution.description));

            if !solution.steps.is_empty() {
                output.push_str("     Steps:\n");
                for step in &solution.steps {
                    output.push_str(&format!("     - {}\n", step));
                }
            }

            if let Some(code) = &solution.code_example {
                output.push_str("     Example code:\n");
                output.push_str(&format!("     ```rust{}\n     ```\n", code));
            }
        }
    }

    if !enhancement.documentation_links.is_empty() {
        output.push_str("\n📚 Documentation:\n");
        for link in &enhancement.documentation_links {
            output.push_str(&format!("  - {}\n", link));
        }
    }

    if !enhancement.troubleshooting_steps.is_empty() {
        output.push_str("\n🔍 Troubleshooting:\n");
        for step in &enhancement.troubleshooting_steps {
            output.push_str(&format!("  - {}\n", step));
        }
    }

    output
}

/// Convenient function to enhance any RecognitionError
pub fn enhance_recognition_error(error: &RecognitionError) -> String {
    error.get_enhanced_message()
}

/// Get quick fixes for an error
pub fn get_quick_fixes(error: &RecognitionError) -> Vec<String> {
    error.get_quick_fixes()
}

/// Check if an error is recoverable
pub fn is_error_recoverable(error: &RecognitionError) -> bool {
    error.is_recoverable()
}

impl fmt::Display for ErrorEnhancement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format_enhanced_error(self))
    }
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCategory::Configuration => write!(f, "Configuration"),
            ErrorCategory::Resources => write!(f, "Resources"),
            ErrorCategory::AudioFormat => write!(f, "Audio Format"),
            ErrorCategory::ModelIssues => write!(f, "Model Issues"),
            ErrorCategory::Performance => write!(f, "Performance"),
            ErrorCategory::InputValidation => write!(f, "Input Validation"),
            ErrorCategory::Integration => write!(f, "Integration"),
            ErrorCategory::FeatureSupport => write!(f, "Feature Support"),
        }
    }
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Critical => write!(f, "Critical"),
            ErrorSeverity::High => write!(f, "High"),
            ErrorSeverity::Medium => write!(f, "Medium"),
            ErrorSeverity::Low => write!(f, "Low"),
            ErrorSeverity::Info => write!(f, "Info"),
        }
    }
}

impl fmt::Display for SolutionDifficulty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolutionDifficulty::Easy => write!(f, "Easy"),
            SolutionDifficulty::Moderate => write!(f, "Moderate"),
            SolutionDifficulty::Advanced => write!(f, "Advanced"),
        }
    }
}

/// Create enhancement for training errors
fn create_training_error_enhancement(message: &str) -> ErrorEnhancement {
    ErrorEnhancement {
        original_message: message.to_string(),
        category: ErrorCategory::ModelIssues,
        severity: ErrorSeverity::High,
        context: ErrorContext {
            operation: "Training".to_string(),
            component: "Training Manager".to_string(),
            input_summary: "training module".to_string(),
            system_state: "Training in progress".to_string(),
            additional_info: HashMap::new(),
            timestamp: std::time::SystemTime::now(),
        },
        solutions: vec![
            Solution {
                title: "Check training data quality".to_string(),
                description: "Verify training data format and quality".to_string(),
                priority: 1,
                estimated_time: "5-10 minutes".to_string(),
                difficulty: SolutionDifficulty::Easy,
                steps: vec![
                    "Verify audio file formats are supported".to_string(),
                    "Check transcription accuracy and formatting".to_string(),
                    "Ensure data splits are properly balanced".to_string(),
                ],
                code_example: Some(
                    r#"
# Validate training data
cargo run --example validate_training_data --path ./data/train
"#
                    .to_string(),
                ),
                success_indicators: vec![
                    "All audio files load successfully".to_string(),
                    "No missing transcriptions".to_string(),
                ],
            },
            Solution {
                title: "Verify model configuration".to_string(),
                description: "Check training hyperparameters and model settings".to_string(),
                priority: 2,
                estimated_time: "2-5 minutes".to_string(),
                difficulty: SolutionDifficulty::Easy,
                steps: vec![
                    "Review learning rate settings".to_string(),
                    "Check batch size configuration".to_string(),
                    "Verify model architecture parameters".to_string(),
                ],
                code_example: Some(
                    r#"
// Check training configuration
let config = TrainingConfig::default();
println!("Learning rate: {}", config.learning_rate);
println!("Batch size: {}", config.batch_size);
"#
                    .to_string(),
                ),
                success_indicators: vec![
                    "Configuration parameters are within valid ranges".to_string()
                ],
            },
        ],
        documentation_links: vec![
            "https://docs.voirs.ai/training/getting-started".to_string(),
            "https://docs.voirs.ai/training/configuration".to_string(),
        ],
        troubleshooting_steps: vec![
            "Check training logs for detailed error information".to_string(),
            "Verify system has sufficient disk space for training artifacts".to_string(),
            "Ensure GPU memory is sufficient for model size".to_string(),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_load_enhancement() {
        let error = RecognitionError::ModelLoadError {
            message: "Model file not found".to_string(),
            source: None,
        };

        let enhancement = error.enhance_error();
        assert_eq!(enhancement.category, ErrorCategory::ModelIssues);
        assert_eq!(enhancement.severity, ErrorSeverity::Critical);
        assert!(!enhancement.solutions.is_empty());
        assert!(enhancement.solutions[0].title.contains("Check model file"));
    }

    #[test]
    fn test_audio_processing_enhancement() {
        let error = RecognitionError::AudioProcessingError {
            message: "Unsupported audio format".to_string(),
            source: None,
        };

        let enhancement = error.enhance_error();
        assert_eq!(enhancement.category, ErrorCategory::AudioFormat);
        assert_eq!(enhancement.severity, ErrorSeverity::High);
        assert!(!enhancement.solutions.is_empty());
    }

    #[test]
    fn test_enhanced_message_formatting() {
        let error = RecognitionError::ConfigurationError {
            message: "Invalid configuration".to_string(),
        };

        let message = error.get_enhanced_message();
        assert!(message.contains("🚨"));
        assert!(message.contains("💡 Suggested Solutions:"));
        assert!(message.contains("📚 Documentation:"));
    }

    #[test]
    fn test_quick_fixes() {
        let error = RecognitionError::ConfigurationError {
            message: "Invalid configuration".to_string(),
        };

        let fixes = error.get_quick_fixes();
        assert!(!fixes.is_empty());
        assert!(fixes.iter().any(|f| f.contains("configuration")));
    }

    #[test]
    fn test_recoverable_errors() {
        let recoverable_error = RecognitionError::ConfigurationError {
            message: "Invalid config".to_string(),
        };
        assert!(recoverable_error.is_recoverable());

        let non_recoverable_error = RecognitionError::MemoryError {
            message: "Out of memory".to_string(),
            source: None,
        };
        assert!(!non_recoverable_error.is_recoverable());
    }
}
