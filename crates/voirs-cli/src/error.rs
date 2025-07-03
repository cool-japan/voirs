//! Error handling system for the CLI.

use std::fmt;

use thiserror::Error;

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
    AudioFormat {
        format: String,
        supported: String,
    },
    
    /// Voice not found error
    #[error("Voice '{voice_id}' not found. Available voices: {available}")]
    VoiceNotFound {
        voice_id: String,
        available: String,
    },
    
    /// Invalid parameter error
    #[error("Invalid parameter '{parameter}': {message}")]
    InvalidParameter {
        parameter: String,
        message: String,
    },
    
    /// SDK error wrapper
    #[error("VoiRS SDK error: {0}")]
    Sdk(#[from] voirs::VoirsError),
    
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
    
    /// Get user-friendly error message with suggestions
    pub fn user_message(&self) -> String {
        match self {
            CliError::Config { message } => {
                format!("Configuration error: {}\n\nTry running 'voirs config --init' to create a default configuration.", message)
            }
            CliError::File { operation, path, source } => {
                format!("Failed to {} '{}': {}\n\nPlease check that the path exists and you have the necessary permissions.", operation, path, source)
            }
            CliError::AudioFormat { format, supported } => {
                format!("Unsupported audio format: '{}'\n\nSupported formats: {}\n\nExample: voirs synthesize \"Hello\" --output audio.wav", format, supported)
            }
            CliError::VoiceNotFound { voice_id, available } => {
                format!("Voice '{}' not found.\n\nAvailable voices: {}\n\nUse 'voirs voices list' to see all available voices.", voice_id, available)
            }
            CliError::InvalidParameter { parameter, message } => {
                format!("Invalid parameter '{}': {}\n\nUse 'voirs --help' for usage information.", parameter, message)
            }
            CliError::Sdk(e) => {
                format!("VoiRS synthesis error: {}\n\nThis might be a model loading or synthesis issue. Try checking your voice installation with 'voirs voices list'.", e)
            }
            CliError::Serialization(e) => {
                format!("Data serialization error: {}\n\nThis might indicate a configuration file corruption.", e)
            }
            CliError::Toml(e) => {
                format!("TOML configuration error: {}\n\nPlease check your configuration file syntax.", e)
            }
            CliError::Io(e) => {
                format!("File system error: {}\n\nPlease check file permissions and available disk space.", e)
            }
            CliError::VoiceError(e) => {
                format!("Voice error: {}\n\nPlease check available voices with 'voirs voices list'.", e)
            }
            CliError::AudioError(e) => {
                format!("Audio system error: {}\n\nPlease check your audio device configuration.", e)
            }
            CliError::SynthesisError(e) => {
                format!("Synthesis error: {}\n\nThis might be a model or configuration issue.", e)
            }
            CliError::IoError(e) => {
                format!("I/O error: {}\n\nPlease check file paths and permissions.", e)
            }
            CliError::SerializationError(e) => {
                format!("Serialization error: {}\n\nThis might indicate corrupted data.", e)
            }
            CliError::InvalidArgument(e) => {
                format!("Invalid argument: {}\n\nPlease check command usage with --help.", e)
            }
            CliError::NotImplemented(e) => {
                format!("Feature not implemented: {}\n\nThis feature is coming in a future update.", e)
            }
            CliError::InteractiveError(e) => {
                format!("Interactive mode error: {}\n\nTry restarting the interactive session.", e)
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
            CliError::IoError(_) => 16,
            CliError::SerializationError(_) => 17,
            CliError::InvalidArgument(_) => 18,
            CliError::NotImplemented(_) => 19,
            CliError::InteractiveError(_) => 20,
        }
    }
}

/// Result type for CLI operations
pub type CliResult<T> = Result<T, CliError>;

/// Alias for compatibility with interactive modules
pub type Result<T> = CliResult<T>;
pub type VoirsCliError = CliError;

/// Helper trait for adding context to errors
pub trait ErrorContext<T> {
    fn with_context<F, S>(self, f: F) -> CliResult<T>
    where
        F: FnOnce() -> S,
        S: Into<String>;
}

impl<T> ErrorContext<T> for Result<T, std::io::Error> {
    fn with_context<F, S>(self, f: F) -> CliResult<T>
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