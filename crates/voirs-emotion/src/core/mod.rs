//! Core emotion processing functionality
//!
//! This module provides the main emotion processing system, including:
//! - Emotion processor with state management and transitions
//! - Audio processing with emotion-aware effects
//! - SIMD-optimized operations for performance (basic and advanced)
//! - Advanced spectral processing using scirs2-fft
//! - Caching and buffer pooling for efficiency
//! - Builder pattern for flexible configuration

mod audio_processing;
mod builder;
mod cache;
mod processor;
mod simd;
pub mod simd_advanced;
pub mod spectral_advanced;

// Re-export public types
pub use builder::EmotionProcessorBuilder;
pub use processor::EmotionProcessor;
