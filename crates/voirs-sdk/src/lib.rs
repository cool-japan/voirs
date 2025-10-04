//! # VoiRS SDK
//!
//! Unified SDK and public API for VoiRS speech synthesis framework.
//!
//! VoiRS SDK provides a comprehensive, high-level interface for neural speech synthesis,
//! abstracting the complexity of G2P (Grapheme-to-Phoneme), acoustic modeling, and vocoding
//! into a simple, efficient API.
//!
//! ## Quick Start
//!
//! ```no_run
//! use voirs_sdk::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Create a pipeline with default settings
//!     let pipeline = VoirsPipelineBuilder::new()
//!         .with_quality(QualityLevel::High)
//!         .with_voice("default")
//!         .build()
//!         .await?;
//!
//!     // Synthesize speech
//!     let audio = pipeline.synthesize("Hello, world!").await?;
//!     
//!     // Save to file
//!     audio.save_wav("output.wav")?;
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Key Features
//!
//! - **Simple API**: High-level interface for speech synthesis
//! - **Async/Concurrent**: Built for modern async Rust applications
//! - **Streaming**: Real-time synthesis with low latency
//! - **Plugin System**: Extensible audio effects and processing
//! - **Caching**: Intelligent model and result caching
//! - **Quality Control**: Comprehensive audio quality validation
//! - **Performance**: Optimized for both speed and memory efficiency
//!
//! ## Architecture
//!
//! The VoiRS SDK consists of several key components:
//!
//! - [`VoirsPipeline`]: Main synthesis pipeline
//! - [`VoirsPipelineBuilder`]: Fluent API for pipeline configuration  
//! - [`AudioBuffer`]: Audio data management and processing
//! - [`Streaming`]: Real-time synthesis capabilities
//! - [`Plugins`]: Extensible effects system
//! - [`Cache`]: Intelligent caching system
//!
//! ## Examples
//!
//! ### Basic Synthesis
//!
//! ```no_run
//! use voirs_sdk::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let pipeline = VoirsPipelineBuilder::new().build().await?;
//!     let audio = pipeline.synthesize("Hello, world!").await?;
//!     audio.save_wav("hello.wav")?;
//!     Ok(())
//! }
//! ```
//!
//! ### Streaming Synthesis
//!
//! ```no_run
//! use voirs_sdk::prelude::*;
//! use futures::StreamExt;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let pipeline = Arc::new(VoirsPipelineBuilder::new().build().await?);
//!     
//!     let mut stream = pipeline.synthesize_stream(
//!         "This is a longer text that will be synthesized in real-time."
//!     ).await?;
//!     
//!     while let Some(chunk) = stream.next().await {
//!         let audio_chunk = chunk?;
//!         // Process audio chunk in real-time
//!         println!("Received {} samples", audio_chunk.len());
//!     }
//!     
//!     Ok(())
//! }
//! ```
//!
//! ### Voice Management
//!
//! ```no_run
//! use voirs_sdk::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let pipeline = VoirsPipelineBuilder::new()
//!         .with_voice("female_voice")
//!         .build()
//!         .await?;
//!     
//!     // List available voices
//!     let voices = pipeline.list_voices().await?;
//!     for voice in voices {
//!         println!("Available voice: {} ({})", voice.name, voice.language);
//!     }
//!     
//!     // Switch voice at runtime
//!     pipeline.set_voice("male_voice").await?;
//!     let audio = pipeline.synthesize("Speaking with a different voice").await?;
//!     
//!     Ok(())
//! }
//! ```
//!
//! ### Advanced Configuration
//!
//! ```no_run
//! use voirs_sdk::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let pipeline = VoirsPipelineBuilder::new()
//!         .with_quality(QualityLevel::High)
//!         .with_gpu_acceleration(true)
//!         .with_threads(4)
//!         .build()
//!         .await?;
//!     
//!     let audio = pipeline.synthesize("High quality synthesis!").await?;
//!     audio.save_wav("quality_output.wav")?;
//!     
//!     Ok(())
//! }
//! ```
//!
//! ### Configuration and Quality Control
//!
//! ```no_run
//! use voirs_sdk::prelude::*;
//! use std::path::PathBuf;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let pipeline = VoirsPipelineBuilder::new()
//!         .with_quality(QualityLevel::High)
//!         .with_threads(4)
//!         .with_cache_dir(PathBuf::from("/tmp/voirs-cache"))
//!         .build()
//!         .await?;
//!     
//!     let audio = pipeline.synthesize("High quality synthesis").await?;
//!     
//!     // Access audio properties
//!     println!("Sample rate: {} Hz", audio.sample_rate());
//!     println!("Duration: {:.2} seconds", audio.duration());
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Performance
//!
//! The VoiRS SDK is designed for high performance:
//!
//! - **Initialization**: ≤ 2 seconds (cold start with model download)
//! - **Synthesis Latency**: ≤ 100ms overhead per synthesis
//! - **Memory Usage**: ≤ 50MB SDK overhead
//! - **Real-time Factor**: ≤ 0.5 (synthesis faster than playback)
//! - **Concurrent Operations**: 100+ simultaneous operations supported
//!
//! ## Error Handling
//!
//! All operations return [`Result<T, VoirsError>`](VoirsError) for comprehensive error handling:
//!
//! ```no_run
//! use voirs_sdk::prelude::*;
//!
//! #[tokio::main]
//! async fn main() {
//!     match VoirsPipelineBuilder::new().build().await {
//!         Ok(pipeline) => {
//!             match pipeline.synthesize("Hello!").await {
//!                 Ok(audio) => println!("Success! {} samples", audio.len()),
//!                 Err(e) => eprintln!("Synthesis error: {}", e),
//!             }
//!         }
//!         Err(e) => eprintln!("Pipeline creation error: {}", e),
//!     }
//! }
//! ```
//!
//! ## Feature Flags
//!
//! - `gpu`: Enable GPU acceleration for models
//! - `onnx`: Enable ONNX runtime support
//! - `default`: Standard CPU-based processing
//!
//! ## Platform Support
//!
//! - **Operating Systems**: Linux, macOS, Windows
//! - **Architectures**: x86_64, ARM64
//! - **Runtimes**: Tokio async runtime required

pub mod adapters;
pub mod r#async;
pub mod audio;
pub mod builder;
pub mod cache;
pub mod capabilities;
pub mod config;
pub mod error;
pub mod logging;
pub mod memory;
pub mod performance;
pub mod pipeline;
pub mod plugins;
pub mod prelude;
pub mod streaming;
pub mod traits;
pub mod types;
pub mod validation;
pub mod versioning;
pub mod voice;

// Advanced voice features
#[cfg(feature = "cloning")]
pub mod cloning;
#[cfg(feature = "conversion")]
pub mod conversion;
#[cfg(feature = "emotion")]
pub mod emotion;
#[cfg(feature = "singing")]
pub mod singing;
#[cfg(feature = "spatial")]
pub mod spatial;

// Web Integration modules
#[cfg(feature = "http")]
pub mod http;
#[cfg(feature = "wasm")]
pub mod wasm;

// Cloud Integration modules
#[cfg(feature = "cloud")]
pub mod cloud;

// Re-export core types and traits
pub use audio::AudioBuffer;
pub use builder::VoirsPipelineBuilder;
pub use capabilities::CapabilityManager;
pub use error::VoirsError;
pub use performance::PerformanceMonitor;
pub use pipeline::VoirsPipeline;
pub use traits::{AcousticModel, G2p, Vocoder};
pub use types::*;

// Advanced voice features re-exports
#[cfg(feature = "cloning")]
pub use cloning::{CloningConfig, VoiceCloner};
#[cfg(feature = "conversion")]
pub use conversion::{ConversionConfig, VoiceConverter};
#[cfg(feature = "emotion")]
pub use emotion::{EmotionConfig, EmotionController};
#[cfg(feature = "singing")]
pub use singing::{SingingConfig, SingingController};
#[cfg(feature = "spatial")]
pub use spatial::{SpatialAudioConfig, SpatialAudioController};

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
