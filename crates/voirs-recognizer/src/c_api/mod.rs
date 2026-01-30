//! C/C++ Foreign Function Interface (FFI) for speech recognition.
//!
//! This module provides a stable C-compatible API for integrating speech recognition
//! into C, C++, and other languages that support C FFI. It includes comprehensive
//! error handling, memory management utilities, and both synchronous and streaming
//! recognition interfaces.
//!
//! # Features
//!
//! - **C-Compatible ABI**: Stable FFI interface for cross-language integration
//! - **Memory Safety**: RAII wrappers and explicit ownership semantics
//! - **Error Handling**: Comprehensive error codes with detailed messages
//! - **Streaming Support**: Real-time audio processing APIs
//! - **Thread Safety**: Safe concurrent access from multiple threads
//! - **Zero-Copy Operations**: Efficient audio buffer handling
//!
//! # C Header Generation
//!
//! Use `cbindgen` to generate C/C++ headers:
//!
//! ```bash
//! cbindgen --config cbindgen.toml --output voirs_recognizer.h
//! ```
//!
//! # Example Usage (C)
//!
//! ```c
//! #include "voirs_recognizer.h"
//!
//! // Initialize recognizer
//! VoirsRecognizer* recognizer = voirs_recognizer_new();
//!
//! // Load audio
//! VoirsAudioBuffer* audio = voirs_audio_load("audio.wav");
//!
//! // Recognize speech
//! VoirsResult* result = voirs_recognize(recognizer, audio);
//! printf("Text: %s\n", voirs_result_text(result));
//!
//! // Cleanup
//! voirs_result_free(result);
//! voirs_audio_free(audio);
//! voirs_recognizer_free(recognizer);
//! ```
//!
//! # Memory Management
//!
//! All heap-allocated objects must be explicitly freed using the corresponding
//! `*_free()` functions to prevent memory leaks.

#[cfg(feature = "c-api")]
mod core;

#[cfg(feature = "c-api")]
mod types;

#[cfg(feature = "c-api")]
mod recognition;

#[cfg(feature = "c-api")]
mod streaming;

#[cfg(feature = "c-api")]
mod error;

#[cfg(feature = "c-api")]
mod memory;

#[cfg(feature = "c-api")]
pub use core::*;

#[cfg(feature = "c-api")]
pub use types::*;

#[cfg(feature = "c-api")]
pub use recognition::*;

#[cfg(feature = "c-api")]
pub use streaming::*;

#[cfg(feature = "c-api")]
pub use error::*;

#[cfg(feature = "c-api")]
pub use memory::*;
