//! Error handling system for the CLI.
//!
//! This module provides comprehensive error handling capabilities including
//! basic CLI errors, advanced error handling with recovery, and formatting utilities.

use std::fmt;

use thiserror::Error;

pub mod advanced_handler;

/// CLI-specific error types
#[derive(Error, Debug)]
pub enum CliError {
    /// Configuration error
    #[error("Configuration error: {message}")]
    Config { message: String },

    /// File operation error
    #[error("File operation failed: {operation} on {path}: {source}")]
    File {
        operation: String,
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// Audio format error
    #[error("Unsupported audio format: {format}. Supported formats: {supported}")]
    AudioFormat { format: String, supported: String },

    /// Voice not found error
    #[error("Voice '{voice_id}' not found. Available voices: {available}")]
    VoiceNotFound { voice_id: String, available: String },

    /// Invalid parameter error
    #[error("Invalid parameter '{parameter}': {message}")]
    InvalidParameter { parameter: String, message: String },

    /// SDK error wrapper
    #[error("VoiRS SDK error: {0}")]
    Sdk(#[from] voirs_sdk::VoirsError),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// TOML serialization error
    #[error("TOML error: {0}")]
    Toml(#[from] toml::ser::Error),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Voice error for interactive mode
    #[error("Voice error: {0}")]
    VoiceError(String),

    /// Audio error for interactive mode
    #[error("Audio error: {0}")]
    AudioError(String),

    /// Synthesis error for interactive mode
    #[error("Synthesis error: {0}")]
    SynthesisError(String),

    /// Validation error
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// IO error with message
    #[error("IO error: {0}")]
    IoError(String),

    /// Serialization error with message
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Invalid argument error
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Not implemented error
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Interactive error
    #[error("Interactive error: {0}")]
    InteractiveError(String),

    /// Packaging error
    #[error("Packaging error: {0}")]
    PackagingError(String),

    /// Update error
    #[error("Update error: {0}")]
    UpdateError(String),

    /// Emotion control error
    #[error("Emotion control error: {0}")]
    EmotionError(String),

    /// Voice cloning error
    #[error("Voice cloning error: {0}")]
    CloningError(String),

    /// Voice conversion error
    #[error("Voice conversion error: {0}")]
    ConversionError(String),

    /// Singing synthesis error
    #[error("Singing synthesis error: {0}")]
    SingingError(String),

    /// Spatial audio error
    #[error("Spatial audio error: {0}")]
    SpatialError(String),

    /// Model loading error
    #[error("Model loading error: {0}")]
    ModelLoadingError(String),

    /// Feature not available error
    #[error("Feature not available: {0}")]
    FeatureNotAvailable(String),

    /// Dependency missing error
    #[error("Missing dependency: {0}")]
    MissingDependency(String),

    /// Hardware requirement error
    #[error("Hardware requirement not met: {0}")]
    HardwareRequirement(String),

    /// Network error
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Performance warning
    #[error("Performance warning: {0}")]
    PerformanceWarning(String),

    /// Advanced error with rich context
    #[error("Advanced error: {0}")]
    Advanced(#[from] advanced_handler::AdvancedError),
}

impl CliError {
    /// Create a configuration error
    pub fn config<S: Into<String>>(message: S) -> Self {
        Self::Config {
            message: message.into(),
        }
    }

    /// Create a file operation error
    pub fn file_operation<S: Into<String>>(operation: S, path: S, source: std::io::Error) -> Self {
        Self::File {
            operation: operation.into(),
            path: path.into(),
            source,
        }
    }

    /// Create an audio format error
    pub fn audio_format<S: Into<String>>(format: S) -> Self {
        Self::AudioFormat {
            format: format.into(),
            supported: "wav, flac, mp3, opus, ogg".to_string(),
        }
    }

    /// Create a voice not found error
    pub fn voice_not_found<S: Into<String>>(voice_id: S, available: Vec<String>) -> Self {
        Self::VoiceNotFound {
            voice_id: voice_id.into(),
            available: available.join(", "),
        }
    }

    /// Create an invalid parameter error
    pub fn invalid_parameter<S: Into<String>>(parameter: S, message: S) -> Self {
        Self::InvalidParameter {
            parameter: parameter.into(),
            message: message.into(),
        }
    }

    /// Create an emotion control error
    pub fn emotion_error<S: Into<String>>(message: S) -> Self {
        Self::EmotionError(message.into())
    }

    /// Create a voice cloning error
    pub fn cloning_error<S: Into<String>>(message: S) -> Self {
        Self::CloningError(message.into())
    }

    /// Create a voice conversion error
    pub fn conversion_error<S: Into<String>>(message: S) -> Self {
        Self::ConversionError(message.into())
    }

    /// Create a singing synthesis error
    pub fn singing_error<S: Into<String>>(message: S) -> Self {
        Self::SingingError(message.into())
    }

    /// Create a spatial audio error
    pub fn spatial_error<S: Into<String>>(message: S) -> Self {
        Self::SpatialError(message.into())
    }

    /// Create a model loading error
    pub fn model_loading_error<S: Into<String>>(message: S) -> Self {
        Self::ModelLoadingError(message.into())
    }

    /// Create a feature not available error
    pub fn feature_not_available<S: Into<String>>(message: S) -> Self {
        Self::FeatureNotAvailable(message.into())
    }

    /// Create a missing dependency error
    pub fn missing_dependency<S: Into<String>>(message: S) -> Self {
        Self::MissingDependency(message.into())
    }

    /// Create a hardware requirement error
    pub fn hardware_requirement<S: Into<String>>(message: S) -> Self {
        Self::HardwareRequirement(message.into())
    }

    /// Create a network error
    pub fn network_error<S: Into<String>>(message: S) -> Self {
        Self::NetworkError(message.into())
    }

    /// Create a performance warning
    pub fn performance_warning<S: Into<String>>(message: S) -> Self {
        Self::PerformanceWarning(message.into())
    }

    /// Convert to an advanced error with additional context
    pub fn to_advanced_error(&self) -> advanced_handler::AdvancedError {
        use advanced_handler::*;
        use std::collections::HashMap;
        use std::time::{SystemTime, UNIX_EPOCH};

        let (category, severity) = match self {
            CliError::Config { .. } => (ErrorCategory::Configuration, ErrorSeverity::Error),
            CliError::File { .. } => (ErrorCategory::FileSystem, ErrorSeverity::Error),
            CliError::AudioFormat { .. } => (ErrorCategory::UserInput, ErrorSeverity::Error),
            CliError::VoiceNotFound { .. } => (ErrorCategory::UserInput, ErrorSeverity::Error),
            CliError::InvalidParameter { .. } => (ErrorCategory::UserInput, ErrorSeverity::Error),
            CliError::NetworkError(_) => (ErrorCategory::Network, ErrorSeverity::Error),
            CliError::ModelLoadingError(_) => (ErrorCategory::ModelLoading, ErrorSeverity::Error),
            CliError::AudioError(_) => (ErrorCategory::AudioProcessing, ErrorSeverity::Error),
            CliError::SynthesisError(_) => (ErrorCategory::Synthesis, ErrorSeverity::Error),
            CliError::MissingDependency(_) => (ErrorCategory::Dependency, ErrorSeverity::Error),
            CliError::HardwareRequirement(_) => (ErrorCategory::Hardware, ErrorSeverity::Error),
            CliError::PerformanceWarning(_) => {
                (ErrorCategory::ResourceExhaustion, ErrorSeverity::Warning)
            }
            _ => (ErrorCategory::Internal, ErrorSeverity::Error),
        };

        let context = ErrorContext {
            operation: "cli_operation".to_string(),
            user: std::env::var("USER").ok(),
            session_id: None,
            request_id: None,
            component: "voirs-cli".to_string(),
            function: None,
            location: None,
            parameters: HashMap::new(),
            system_state: SystemState {
                available_memory_bytes: 0, // Would be populated from system info
                cpu_usage_percent: 0.0,
                active_operations: 0,
                queue_depth: 0,
                last_success_time: None,
                uptime_seconds: 0,
            },
            performance_metrics: None,
        };

        let recovery_suggestions = self.generate_recovery_suggestions();

        AdvancedError {
            category,
            severity,
            message: self.to_string(),
            technical_details: format!("CLI Error: {:?}", self),
            context,
            recovery_suggestions,
            related_errors: vec![],
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            error_id: format!("cli_error_{}", fastrand::u64(..)),
            recoverable: self.is_recoverable(),
            retry_info: if self.is_recoverable() {
                Some(RetryInfo {
                    attempt: 0,
                    max_attempts: 3,
                    retry_delay: std::time::Duration::from_secs(1),
                    backoff_strategy: BackoffStrategy::Exponential,
                    last_retry: None,
                    success_history: vec![],
                })
            } else {
                None
            },
        }
    }

    /// Generate recovery suggestions for this error
    fn generate_recovery_suggestions(&self) -> Vec<advanced_handler::RecoverySuggestion> {
        use advanced_handler::*;
        use std::collections::HashMap;

        match self {
            CliError::NetworkError(_) => vec![RecoverySuggestion {
                category: RecoveryCategory::NetworkTroubleshooting,
                priority: 8,
                suggestion: "Check internet connection and retry".to_string(),
                steps: vec![
                    "Check internet connectivity".to_string(),
                    "Verify DNS resolution".to_string(),
                    "Retry the operation".to_string(),
                ],
                estimated_time: std::time::Duration::from_secs(30),
                difficulty: DifficultyLevel::Easy,
                success_probability: 0.8,
                requires_user_action: true,
                automated_actions: vec![AutomatedAction {
                    action_type: ActionType::RetryOperation,
                    description: "Retry with exponential backoff".to_string(),
                    parameters: HashMap::new(),
                    safe_to_automate: true,
                    execution_time: std::time::Duration::from_secs(10),
                    dependencies: vec![],
                }],
            }],
            CliError::Config { .. } => vec![RecoverySuggestion {
                category: RecoveryCategory::ConfigurationFix,
                priority: 7,
                suggestion: "Initialize default configuration".to_string(),
                steps: vec![
                    "Run 'voirs config --init'".to_string(),
                    "Verify configuration with 'voirs config --show'".to_string(),
                ],
                estimated_time: std::time::Duration::from_secs(10),
                difficulty: DifficultyLevel::Easy,
                success_probability: 0.9,
                requires_user_action: true,
                automated_actions: vec![],
            }],
            CliError::ModelLoadingError(_) => vec![RecoverySuggestion {
                category: RecoveryCategory::ResourceOptimization,
                priority: 6,
                suggestion: "Validate and re-download model".to_string(),
                steps: vec![
                    "Check available models with 'voirs list-models'".to_string(),
                    "Download required model".to_string(),
                    "Verify model integrity".to_string(),
                ],
                estimated_time: std::time::Duration::from_secs(300),
                difficulty: DifficultyLevel::Medium,
                success_probability: 0.7,
                requires_user_action: true,
                automated_actions: vec![AutomatedAction {
                    action_type: ActionType::ValidateConfiguration,
                    description: "Validate model files".to_string(),
                    parameters: HashMap::new(),
                    safe_to_automate: true,
                    execution_time: std::time::Duration::from_secs(30),
                    dependencies: vec![],
                }],
            }],
            _ => vec![RecoverySuggestion {
                category: RecoveryCategory::RetryOptimization,
                priority: 5,
                suggestion: "Retry the operation".to_string(),
                steps: vec![
                    "Wait a moment".to_string(),
                    "Retry the same command".to_string(),
                ],
                estimated_time: std::time::Duration::from_secs(5),
                difficulty: DifficultyLevel::Trivial,
                success_probability: 0.5,
                requires_user_action: true,
                automated_actions: vec![],
            }],
        }
    }

    /// Check if this error type is recoverable
    fn is_recoverable(&self) -> bool {
        match self {
            CliError::NetworkError(_) => true,
            CliError::Config { .. } => true,
            CliError::File { .. } => true,
            CliError::ModelLoadingError(_) => true,
            CliError::MissingDependency(_) => true,
            CliError::InvalidParameter { .. } => false,
            CliError::AudioFormat { .. } => false,
            CliError::VoiceNotFound { .. } => false,
            _ => true,
        }
    }

    /// Get user-friendly error message with suggestions
    pub fn user_message(&self) -> String {
        match self {
            CliError::Config { message } => {
                format!("Configuration error: {}\n\nTry running 'voirs config --init' to create a default configuration.", message)
            }
            CliError::File {
                operation,
                path,
                source,
            } => {
                format!("Failed to {} '{}': {}\n\nPlease check that the path exists and you have the necessary permissions.", operation, path, source)
            }
            CliError::AudioFormat { format, supported } => {
                format!("Unsupported audio format: '{}'\n\nSupported formats: {}\n\nExample: voirs synthesize \"Hello\" --output audio.wav", format, supported)
            }
            CliError::VoiceNotFound {
                voice_id,
                available,
            } => {
                format!("Voice '{}' not found.\n\nAvailable voices: {}\n\nUse 'voirs voices list' to see all available voices.", voice_id, available)
            }
            CliError::InvalidParameter { parameter, message } => {
                format!(
                    "Invalid parameter '{}': {}\n\nUse 'voirs --help' for usage information.",
                    parameter, message
                )
            }
            CliError::Sdk(e) => {
                format!("VoiRS synthesis error: {}\n\nThis might be a model loading or synthesis issue. Try checking your voice installation with 'voirs voices list'.", e)
            }
            CliError::Serialization(e) => {
                format!("Data serialization error: {}\n\nThis might indicate a configuration file corruption.", e)
            }
            CliError::Toml(e) => {
                format!(
                    "TOML configuration error: {}\n\nPlease check your configuration file syntax.",
                    e
                )
            }
            CliError::Io(e) => {
                format!("File system error: {}\n\nPlease check file permissions and available disk space.", e)
            }
            CliError::VoiceError(e) => {
                format!(
                    "Voice error: {}\n\nPlease check available voices with 'voirs voices list'.",
                    e
                )
            }
            CliError::AudioError(e) => {
                format!(
                    "Audio system error: {}\n\nPlease check your audio device configuration.",
                    e
                )
            }
            CliError::SynthesisError(e) => {
                format!(
                    "Synthesis error: {}\n\nThis might be a model or configuration issue.",
                    e
                )
            }
            CliError::ValidationError(e) => {
                format!("Validation error: {}\n\nPlease check your input format.", e)
            }
            CliError::IoError(e) => {
                format!(
                    "I/O error: {}\n\nPlease check file paths and permissions.",
                    e
                )
            }
            CliError::SerializationError(e) => {
                format!(
                    "Serialization error: {}\n\nThis might indicate corrupted data.",
                    e
                )
            }
            CliError::InvalidArgument(e) => {
                format!(
                    "Invalid argument: {}\n\nPlease check command usage with --help.",
                    e
                )
            }
            CliError::NotImplemented(e) => {
                format!(
                    "Feature not implemented: {}\n\nThis feature is coming in a future update.",
                    e
                )
            }
            CliError::InteractiveError(e) => {
                format!(
                    "Interactive mode error: {}\n\nTry restarting the interactive session.",
                    e
                )
            }
            CliError::PackagingError(e) => {
                format!(
                    "Packaging error: {}\n\nPlease check your packaging configuration and dependencies.",
                    e
                )
            }
            CliError::UpdateError(e) => {
                format!(
                    "Update error: {}\n\nPlease check your network connection and try again.",
                    e
                )
            }
            CliError::EmotionError(e) => {
                format!(
                    "Emotion control error: {}\n\nSuggestions:\n- Check if emotion models are installed\n- Verify emotion intensity (0.0-1.0)\n- Use 'voirs capabilities check emotion' to verify feature availability\n- Use 'voirs emotion list' to see available emotions",
                    e
                )
            }
            CliError::CloningError(e) => {
                format!(
                    "Voice cloning error: {}\n\nSuggestions:\n- Ensure reference audio is high quality (16kHz+, clear speech)\n- Use at least 30 seconds of reference audio\n- Check if voice cloning models are installed\n- Use 'voirs capabilities check cloning' to verify feature availability\n- Use 'voirs clone validate' to check reference audio quality",
                    e
                )
            }
            CliError::ConversionError(e) => {
                format!(
                    "Voice conversion error: {}\n\nSuggestions:\n- Check if voice conversion models are installed\n- Verify input audio quality and format\n- Use 'voirs capabilities check conversion' to verify feature availability\n- Try reducing conversion intensity for better results",
                    e
                )
            }
            CliError::SingingError(e) => {
                format!(
                    "Singing synthesis error: {}\n\nSuggestions:\n- Check if singing models are installed\n- Verify musical score format (MusicXML, MIDI)\n- Ensure lyrics are properly formatted\n- Use 'voirs capabilities check singing' to verify feature availability\n- Use 'voirs sing validate' to check score format",
                    e
                )
            }
            CliError::SpatialError(e) => {
                format!(
                    "Spatial audio error: {}\n\nSuggestions:\n- Check if spatial audio models are installed\n- Verify HRTF dataset is available\n- Use 'voirs capabilities check spatial' to verify feature availability\n- Ensure position coordinates are valid (x,y,z format)",
                    e
                )
            }
            CliError::ModelLoadingError(e) => {
                format!(
                    "Model loading error: {}\n\nSuggestions:\n- Check if the required model files are present\n- Verify model file integrity\n- Use 'voirs list-models' to see available models\n- Try downloading the model again with 'voirs download-model'\n- Check available disk space and memory",
                    e
                )
            }
            CliError::FeatureNotAvailable(e) => {
                format!(
                    "Feature not available: {}\n\nSuggestions:\n- Check if the feature is enabled in your configuration\n- Verify feature dependencies are installed\n- Use 'voirs capabilities list' to see all available features\n- Check if your VoiRS version supports this feature",
                    e
                )
            }
            CliError::MissingDependency(e) => {
                format!(
                    "Missing dependency: {}\n\nSuggestions:\n- Install the required dependency\n- Use 'voirs capabilities requirements' to see all requirements\n- Check the installation documentation\n- Verify your system meets all prerequisites",
                    e
                )
            }
            CliError::HardwareRequirement(e) => {
                format!(
                    "Hardware requirement not met: {}\n\nSuggestions:\n- Check if your hardware meets the minimum requirements\n- Try using CPU-only mode if GPU is not available\n- Reduce quality settings for better performance\n- Use 'voirs capabilities list' to see system requirements",
                    e
                )
            }
            CliError::NetworkError(e) => {
                format!(
                    "Network error: {}\n\nSuggestions:\n- Check your internet connection\n- Verify proxy settings if applicable\n- Try again later if the server is temporarily unavailable\n- Check firewall settings",
                    e
                )
            }
            CliError::PerformanceWarning(e) => {
                format!(
                    "Performance warning: {}\n\nSuggestions:\n- Consider using GPU acceleration with --gpu flag\n- Reduce quality settings for faster processing\n- Close other resource-intensive applications\n- Use 'voirs capabilities list' to check system capabilities",
                    e
                )
            }
            CliError::Advanced(advanced_error) => {
                // For advanced errors, we'd typically use the advanced handler's user report
                format!("Advanced error: {}", advanced_error.message)
            }
        }
    }

    /// Get exit code for this error
    pub fn exit_code(&self) -> i32 {
        match self {
            CliError::Config { .. } => 1,
            CliError::File { .. } => 2,
            CliError::AudioFormat { .. } => 3,
            CliError::VoiceNotFound { .. } => 4,
            CliError::InvalidParameter { .. } => 5,
            CliError::Sdk(_) => 10,
            CliError::Serialization(_) => 11,
            CliError::Toml(_) => 11,
            CliError::Io(_) => 12,
            CliError::VoiceError(_) => 13,
            CliError::AudioError(_) => 14,
            CliError::SynthesisError(_) => 15,
            CliError::ValidationError(_) => 16,
            CliError::IoError(_) => 17,
            CliError::SerializationError(_) => 17,
            CliError::InvalidArgument(_) => 18,
            CliError::NotImplemented(_) => 19,
            CliError::InteractiveError(_) => 20,
            CliError::PackagingError(_) => 21,
            CliError::UpdateError(_) => 22,
            CliError::EmotionError(_) => 30,
            CliError::CloningError(_) => 31,
            CliError::ConversionError(_) => 32,
            CliError::SingingError(_) => 33,
            CliError::SpatialError(_) => 34,
            CliError::ModelLoadingError(_) => 35,
            CliError::FeatureNotAvailable(_) => 36,
            CliError::MissingDependency(_) => 37,
            CliError::HardwareRequirement(_) => 38,
            CliError::NetworkError(_) => 39,
            CliError::PerformanceWarning(_) => 40,
            CliError::Advanced(_) => 50,
        }
    }
}

/// Result type for CLI operations
pub type Result<T> = std::result::Result<T, CliError>;
pub type VoirsCliError = CliError;
pub type VoirsCLIError = CliError;

/// Convert CliError to VoirsError for compatibility
impl From<CliError> for voirs_sdk::VoirsError {
    fn from(err: CliError) -> Self {
        match err {
            CliError::Sdk(voirs_err) => voirs_err,
            other => voirs_sdk::VoirsError::InternalError {
                component: "voirs-cli".to_string(),
                message: other.to_string(),
            },
        }
    }
}

/// Helper trait for adding context to errors
pub trait ErrorContext<T> {
    fn with_context<F, S>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> S,
        S: Into<String>;
}

impl<T> ErrorContext<T> for std::result::Result<T, std::io::Error> {
    fn with_context<F, S>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> S,
        S: Into<String>,
    {
        self.map_err(|e| CliError::config(f().into()))
    }
}

/// Error message formatting utilities
pub mod formatting {
    use super::CliError;
    use crate::output::get_formatter;

    /// Print error message with proper formatting
    pub fn print_error(error: &CliError) {
        let formatter = get_formatter();
        formatter.error(&error.user_message());

        // Print suggestions based on error type
        match error {
            CliError::VoiceNotFound { .. } => {
                formatter.info("Suggested commands:");
                formatter.list_item("voirs voices list", 1);
                formatter.list_item("voirs voices download <voice-id>", 1);
            }
            CliError::AudioFormat { .. } => {
                formatter.info("Example usage:");
                formatter.list_item("voirs synthesize \"Hello\" -o output.wav", 1);
                formatter.list_item("voirs synthesize \"Hello\" -o output.mp3", 1);
            }
            CliError::Config { .. } => {
                formatter.info("Configuration commands:");
                formatter.list_item("voirs config --init", 1);
                formatter.list_item("voirs config --show", 1);
            }
            CliError::EmotionError(_) => {
                formatter.info("Emotion control commands:");
                formatter.list_item("voirs emotion list", 1);
                formatter.list_item("voirs capabilities check emotion", 1);
                formatter.list_item("voirs guide emotion", 1);
            }
            CliError::CloningError(_) => {
                formatter.info("Voice cloning commands:");
                formatter.list_item("voirs clone validate --reference-files samples/*.wav", 1);
                formatter.list_item("voirs capabilities check cloning", 1);
                formatter.list_item("voirs guide clone", 1);
            }
            CliError::ConversionError(_) => {
                formatter.info("Voice conversion commands:");
                formatter.list_item("voirs convert list-models", 1);
                formatter.list_item("voirs capabilities check conversion", 1);
                formatter.list_item("voirs guide convert", 1);
            }
            CliError::SingingError(_) => {
                formatter.info("Singing synthesis commands:");
                formatter.list_item("voirs sing validate --score score.xml", 1);
                formatter.list_item("voirs capabilities check singing", 1);
                formatter.list_item("voirs guide sing", 1);
            }
            CliError::SpatialError(_) => {
                formatter.info("Spatial audio commands:");
                formatter.list_item("voirs spatial validate --test-audio test.wav", 1);
                formatter.list_item("voirs capabilities check spatial", 1);
                formatter.list_item("voirs guide spatial", 1);
            }
            CliError::ModelLoadingError(_) => {
                formatter.info("Model management commands:");
                formatter.list_item("voirs list-models", 1);
                formatter.list_item("voirs download-model <model-id>", 1);
                formatter.list_item("voirs capabilities requirements", 1);
            }
            CliError::FeatureNotAvailable(_) => {
                formatter.info("Feature diagnosis commands:");
                formatter.list_item("voirs capabilities list", 1);
                formatter.list_item("voirs capabilities requirements", 1);
                formatter.list_item("voirs config --show", 1);
            }
            CliError::MissingDependency(_) => {
                formatter.info("Dependency check commands:");
                formatter.list_item("voirs capabilities requirements", 1);
                formatter.list_item("voirs test", 1);
            }
            CliError::HardwareRequirement(_) => {
                formatter.info("Hardware check commands:");
                formatter.list_item("voirs capabilities list", 1);
                formatter.list_item("voirs test --gpu", 1);
            }
            CliError::NetworkError(_) => {
                formatter.info("Network troubleshooting:");
                formatter.list_item("Check internet connectivity", 1);
                formatter.list_item("Verify proxy settings", 1);
                formatter.list_item("Try again later", 1);
            }
            CliError::PerformanceWarning(_) => {
                formatter.info("Performance optimization:");
                formatter.list_item("Use --gpu flag for acceleration", 1);
                formatter.list_item("Reduce quality settings", 1);
                formatter.list_item("Close other applications", 1);
            }
            CliError::Advanced(advanced_error) => {
                // For advanced errors, we could display more detailed recovery suggestions
                formatter.info("Advanced error detected - recovery suggestions available");
                for suggestion in &advanced_error.recovery_suggestions {
                    formatter.list_item(&suggestion.suggestion, 1);
                }
            }
            _ => {}
        }
    }

    /// Print warning message
    pub fn print_warning(message: &str) {
        get_formatter().warning(message);
    }

    /// Print info message
    pub fn print_info(message: &str) {
        get_formatter().info(message);
    }
}
