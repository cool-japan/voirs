//! CLI command implementations for VoiRS.
//!
//! This module contains all the command-line interface commands available in VoiRS CLI.
//! Each submodule implements a specific command or group of related commands, providing
//! the core functionality that users interact with.
//!
//! ## Core Commands
//!
//! - [`synthesize`]: Text-to-speech synthesis commands
//! - [`voices`]: Voice management and discovery
//! - [`models`]: Model download, installation, and management
//! - [`config`]: Configuration management
//! - [`capabilities`]: System capability checking and validation
//!
//! ## Advanced Features
//!
//! - [`cloning`]: Voice cloning from reference audio (feature-gated)
//! - [`emotion`]: Emotion control and expression (feature-gated)
//! - [`conversion`]: Voice conversion between speakers (feature-gated)
//! - [`singing`]: Singing voice synthesis (feature-gated)
//! - [`spatial`]: Spatial audio processing (feature-gated)
//!
//! ## Processing Commands
//!
//! - [`batch`]: Batch processing of multiple files
//! - [`interactive`]: Interactive synthesis shell
//! - [`server`]: HTTP server for API access
//! - [`monitoring`]: Performance monitoring and analytics
//! - [`accuracy`]: Accuracy testing and evaluation
//!
//! ## Utility Commands
//!
//! - [`mod@test`]: System testing and diagnostics
//! - [`cloud`]: Cloud integration and distributed processing
//! - [`dataset`]: Dataset management and validation
//! - [`performance`]: Performance benchmarking
//! - [`voice_search`]: Voice discovery and search
//! - [`cross_lang_test`]: Cross-language testing utilities

pub mod accuracy;
pub mod batch;
pub mod capabilities;
#[cfg(feature = "cloning")]
pub mod cloning;
pub mod cloud;
pub mod config;
#[cfg(feature = "conversion")]
pub mod conversion;
pub mod cross_lang_test;
pub mod dataset;
#[cfg(feature = "emotion")]
pub mod emotion;
pub mod interactive;
pub mod models;
pub mod monitoring;
pub mod performance;
pub mod server;
#[cfg(feature = "singing")]
pub mod singing;
#[cfg(feature = "spatial")]
pub mod spatial;
pub mod synthesize;
pub mod test;
pub mod voice_search;
pub mod voices;
