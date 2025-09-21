pub mod comprehensive_tests;
pub mod i18n;
pub mod recovery;
pub mod structured;

pub use i18n::*;
pub use recovery::*;
pub use structured::*;

/// Main FFI error type
#[derive(Debug, Clone)]
pub enum VoirsFFIError {
    /// Platform-specific error
    PlatformError(String),
    /// Invalid parameter error
    InvalidParameter(String),
    /// Initialization failed
    InitializationFailed(String),
    /// I/O operation failed
    IoError(String),
    /// Out of memory
    OutOfMemory(String),
    /// Internal error
    InternalError(String),
}

impl std::fmt::Display for VoirsFFIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VoirsFFIError::PlatformError(msg) => write!(f, "Platform error: {}", msg),
            VoirsFFIError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            VoirsFFIError::InitializationFailed(msg) => write!(f, "Initialization failed: {}", msg),
            VoirsFFIError::IoError(msg) => write!(f, "I/O error: {}", msg),
            VoirsFFIError::OutOfMemory(msg) => write!(f, "Out of memory: {}", msg),
            VoirsFFIError::InternalError(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for VoirsFFIError {}

impl From<VoirsStructuredError> for VoirsFFIError {
    fn from(error: VoirsStructuredError) -> Self {
        match error.subcode {
            VoirsErrorSubcode::InvalidParameter
            | VoirsErrorSubcode::NullPointer
            | VoirsErrorSubcode::InvalidFormat
            | VoirsErrorSubcode::InvalidRange => {
                VoirsFFIError::InvalidParameter(error.context.message)
            }
            VoirsErrorSubcode::ConfigurationMissing
            | VoirsErrorSubcode::ConfigurationInvalid
            | VoirsErrorSubcode::ConfigurationConflict => {
                VoirsFFIError::InitializationFailed(error.context.message)
            }
            VoirsErrorSubcode::OutOfMemory => VoirsFFIError::OutOfMemory(error.context.message),
            VoirsErrorSubcode::IoError => VoirsFFIError::IoError(error.context.message),
            _ => VoirsFFIError::InternalError(error.context.message),
        }
    }
}
