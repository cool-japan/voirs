//! Python bindings for VoiRS using PyO3.
//!
//! This module provides comprehensive Python bindings for the VoiRS speech synthesis library,
//! including support for text-to-speech synthesis, audio processing, voice configuration,
//! and optional features like NumPy integration, speech recognition, and advanced audio analysis.
//!
//! # Features
//!
//! - **Core TTS**: Text-to-speech synthesis with voice selection and configuration
//! - **Audio Processing**: Advanced audio buffer management with optional NumPy integration
//! - **Voice Management**: Voice discovery, configuration, and customization
//! - **Streaming**: Real-time streaming synthesis for low-latency applications
//! - **Recognition**: Optional speech recognition integration (with `recognition` feature)
//! - **NumPy Integration**: Seamless NumPy array support (with `numpy` feature)
//!
//! # Quick Start
//!
//! ```python
//! import voirs
//!
//! # Create a pipeline
//! pipeline = voirs.VoirsPipeline()
//!
//! # Synthesize speech
//! result = pipeline.synthesize("Hello, world!")
//! audio = result.audio
//!
//! # Get synthesis metrics
//! metrics = result.metrics
//! print(f"Real-time factor: {metrics.real_time_factor}")
//! ```

//! Python bindings implementation modules
//!
//! This module is organized into logical sub-modules for maintainability:
//! - `error`: Error types and exception handling
//! - `metrics`: Performance metrics and synthesis results
//! - `pipeline`: Main TTS pipeline wrapper
//! - `audio_buffer`: Audio buffer management and NumPy integration
//! - `voice`: Voice information and management
//! - `streaming`: Real-time streaming audio processing
//! - `analyzer`: Advanced audio analysis tools
//! - `config`: Synthesis configuration
//! - `recognition`: Speech recognition (optional, feature-gated)

#[cfg(feature = "python")]
pub mod analyzer;
#[cfg(feature = "python")]
pub mod audio_buffer;
#[cfg(feature = "python")]
mod common;
#[cfg(feature = "python")]
pub mod config;
#[cfg(feature = "python")]
pub mod error;
#[cfg(feature = "python")]
pub mod metrics;
#[cfg(feature = "python")]
pub mod pipeline;
#[cfg(feature = "python")]
pub mod streaming;
#[cfg(feature = "python")]
pub mod voice;

#[cfg(all(feature = "python", feature = "recognition"))]
pub mod recognition;

// Public re-exports for Python module
#[cfg(all(feature = "python", feature = "numpy"))]
pub use analyzer::PyAudioAnalyzer;
#[cfg(feature = "python")]
pub use audio_buffer::PyAudioBuffer;
#[cfg(feature = "python")]
pub use config::PySynthesisConfig;
#[cfg(feature = "python")]
pub use error::VoirsErrorInfo;
#[cfg(feature = "python")]
pub use error::VoirsException;
#[cfg(feature = "python")]
pub use metrics::{SynthesisMetrics, SynthesisResult};
#[cfg(feature = "python")]
pub use pipeline::VoirsPipeline;
#[cfg(all(feature = "python", feature = "numpy"))]
pub use streaming::PyStreamingProcessor;
#[cfg(feature = "python")]
pub use voice::PyVoiceInfo;

#[cfg(all(feature = "python", feature = "recognition"))]
pub use recognition::{
    PyASRModel as RecognitionASRModel, PyAudioAnalysis as RecognitionAudioAnalysis,
    PyAudioAnalyzer as RecognitionAudioAnalyzer, PyPerformanceMetrics, PyPhonemeAlignment,
    PyPhonemeRecognizer, PyRecognitionResult, PyTranscript,
};

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python module entry point
///
/// This function registers all VoiRS Python classes and constants.
/// It is called automatically when the module is imported in Python.
#[cfg(feature = "python")]
#[pymodule]
fn voirs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core classes
    m.add_class::<VoirsPipeline>()?;
    m.add_class::<PyAudioBuffer>()?;
    m.add_class::<PyVoiceInfo>()?;
    m.add_class::<PySynthesisConfig>()?;

    // Enhanced error handling and metrics
    m.add_class::<VoirsErrorInfo>()?;
    m.add("VoirsException", m.py().get_type::<VoirsException>())?;
    m.add_class::<SynthesisMetrics>()?;
    m.add_class::<SynthesisResult>()?;

    // Advanced NumPy integration classes (when numpy feature is enabled)
    #[cfg(feature = "numpy")]
    {
        m.add_class::<PyStreamingProcessor>()?;
        m.add_class::<PyAudioAnalyzer>()?;
    }

    // Add version constant
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Add feature flags for runtime detection
    #[cfg(feature = "numpy")]
    m.add("HAS_NUMPY", true)?;
    #[cfg(not(feature = "numpy"))]
    m.add("HAS_NUMPY", false)?;

    #[cfg(feature = "gpu")]
    m.add("HAS_GPU", true)?;
    #[cfg(not(feature = "gpu"))]
    m.add("HAS_GPU", false)?;

    // Recognition classes (when voirs-recognizer feature is enabled)
    #[cfg(feature = "recognition")]
    {
        m.add_class::<RecognitionASRModel>()?;
        m.add_class::<RecognitionAudioAnalyzer>()?;
        m.add_class::<PyPhonemeRecognizer>()?;
        m.add_class::<PyRecognitionResult>()?;
        m.add_class::<PyTranscript>()?;
        m.add_class::<RecognitionAudioAnalysis>()?;
        m.add_class::<PyPerformanceMetrics>()?;
    }

    // Add metrics constants
    m.add("DEFAULT_PROCESSING_TIMEOUT_MS", 30000)?;
    m.add("DEFAULT_CACHE_SIZE_MB", 512)?;
    m.add("DEFAULT_BATCH_SIZE", 8)?;

    // Add recognition feature flag
    #[cfg(feature = "recognition")]
    m.add("HAS_RECOGNITION", true)?;
    #[cfg(not(feature = "recognition"))]
    m.add("HAS_RECOGNITION", false)?;

    Ok(())
}
