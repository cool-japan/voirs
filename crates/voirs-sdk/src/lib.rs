//! # VoiRS SDK
//! 
//! Unified SDK and public API for VoiRS speech synthesis framework.

pub mod audio;
pub mod builder;
pub mod cache;
pub mod config;
pub mod error;
pub mod logging;
pub mod pipeline;
pub mod plugins;
pub mod prelude;
pub mod streaming;
pub mod traits;
pub mod types;
pub mod voice;

// Re-export core types and traits
pub use audio::AudioBuffer;
pub use builder::VoirsPipelineBuilder;
pub use error::VoirsError;
pub use pipeline::VoirsPipeline;
pub use traits::{AcousticModel, G2p, Vocoder};
pub use types::*;

/// Result type alias for VoiRS operations
pub type Result<T> = std::result::Result<T, VoirsError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_types() {
        // Basic compilation test
        let _result: Result<()> = Ok(());
    }
}