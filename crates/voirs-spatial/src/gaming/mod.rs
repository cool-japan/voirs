//! Gaming Engine Integration for VoiRS Spatial Audio
//!
//! This module provides integration with popular gaming engines including Unity, Unreal Engine,
//! Godot, and custom game engines. It offers C-compatible APIs, real-time audio processing,
//! and game-specific optimizations for immersive spatial audio experiences.

#![allow(unsafe_code)] // Allow unsafe code for C API

// Module declarations
pub mod audio;
pub mod c_api;
pub mod config;
pub mod hardware;
pub mod types;
pub mod unity;
pub mod unreal;

// Backward compatibility: re-export hardware as console
pub use hardware as console;

// Re-export main types for convenience
pub use audio::GamingAudioManager;
pub use config::GamingConfig;
pub use types::{
    AttenuationCurve, AttenuationSettings, AudioCategory, GameAudioSource, GameEngine,
    GamingMetrics,
};

// Re-export C API functions
pub use c_api::*;

// Re-export console hardware types
pub use hardware::*;
