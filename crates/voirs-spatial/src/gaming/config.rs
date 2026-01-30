//! Configuration types for gaming audio integration

use super::types::GameEngine;
use serde::{Deserialize, Serialize};

/// Gaming audio configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamingConfig {
    /// Target engine
    pub engine: GameEngine,
    /// Target frame rate (FPS)
    pub target_fps: u32,
    /// Audio latency target (ms)
    pub target_latency_ms: f32,
    /// Maximum concurrent audio sources
    pub max_sources: usize,
    /// Enable real-time processing optimizations
    pub realtime_optimizations: bool,
    /// Enable GPU acceleration
    pub gpu_acceleration: bool,
    /// Audio quality level (0.0 = lowest, 1.0 = highest)
    pub quality_level: f32,
    /// Memory budget (MB)
    pub memory_budget_mb: u32,
    /// Enable debug mode
    pub debug_mode: bool,
}

impl Default for GamingConfig {
    fn default() -> Self {
        Self {
            engine: GameEngine::Custom,
            target_fps: 60,
            target_latency_ms: 33.0, // ~2 frames at 60 FPS
            max_sources: 32,
            realtime_optimizations: true,
            gpu_acceleration: true,
            quality_level: 0.8,
            memory_budget_mb: 256,
            debug_mode: false,
        }
    }
}
