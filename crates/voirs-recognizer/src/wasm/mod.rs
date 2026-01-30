//! WebAssembly bindings for speech recognition.
//!
//! This module provides WebAssembly (WASM) bindings that enable running speech recognition
//! directly in web browsers and Node.js environments. It supports both real-time streaming
//! and batch processing with efficient memory management.
//!
//! # Features
//!
//! - **Browser Compatibility**: Run ASR models directly in modern web browsers
//! - **Node.js Support**: Server-side JavaScript integration
//! - **Web Worker Support**: Off-main-thread processing for better UI responsiveness
//! - **Streaming Recognition**: Real-time audio processing with low latency
//! - **Efficient Memory**: Optimized for WASM's limited memory constraints
//!
//! # Examples
//!
//! ```javascript
//! // Initialize recognizer in browser
//! const recognizer = await VoirsRecognizer.new();
//!
//! // Process audio buffer
//! const result = await recognizer.recognize(audioBuffer);
//! console.log(result.text);
//! ```
//!
//! # Platform Support
//!
//! - Web browsers with WASM support
//! - Node.js 14+ with WASM support
//! - Cloudflare Workers and edge environments

// Allow unused async for WASM API consistency and future compatibility
#![allow(clippy::unused_async)]

#[cfg(feature = "wasm")]
mod recognizer;

#[cfg(feature = "wasm")]
mod worker;

#[cfg(feature = "wasm")]
mod streaming;

#[cfg(feature = "wasm")]
mod utils;

#[cfg(feature = "wasm")]
pub use recognizer::*;

#[cfg(feature = "wasm")]
pub use worker::*;

#[cfg(feature = "wasm")]
pub use streaming::*;

#[cfg(feature = "wasm")]
pub use utils::*;
