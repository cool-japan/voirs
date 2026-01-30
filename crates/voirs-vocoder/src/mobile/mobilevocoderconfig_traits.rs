//! # MobileVocoderConfig - Trait Implementations
//!
//! This module contains trait implementations for `MobileVocoderConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::MobileVocoderConfig;

impl Default for MobileVocoderConfig {
    fn default() -> Self {
        Self {
            enable_neon: cfg!(target_arch = "aarch64"),
            enable_quantization: true,
            enable_power_management: true,
            enable_thermal_management: true,
            enable_memory_optimization: true,
            target_memory_mb: 100.0,
            max_cpu_temperature: 75.0,
            min_battery_percent: 15.0,
            enable_adaptive_quality: true,
            max_concurrent_synthesis: 2,
            quantization_bits: 16,
            use_arm_optimized_models: cfg!(target_arch = "aarch64"),
            enable_model_caching: true,
            cache_size_mb: 50.0,
        }
    }
}

