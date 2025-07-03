//! Streaming configuration module.
//!
//! This module provides configuration options for streaming audio processing,
//! latency optimization, and real-time constraints.

use serde::{Deserialize, Serialize};
use super::{PerformanceProfile, ValidationResult};
use std::time::Duration;

/// Buffer strategy for streaming
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BufferStrategy {
    /// Fixed-size buffers
    Fixed,
    /// Dynamic buffer sizing based on processing speed
    Dynamic,
    /// Circular buffer with overlap
    Circular,
    /// Lock-free ring buffer
    LockFree,
}

impl Default for BufferStrategy {
    fn default() -> Self {
        Self::Dynamic
    }
}

/// Latency optimization modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LatencyMode {
    /// Minimize latency at the cost of quality
    UltraLow,
    /// Low latency with good quality balance
    Low,
    /// Balanced latency and quality
    Balanced,
    /// Higher latency for maximum quality
    Quality,
}

impl Default for LatencyMode {
    fn default() -> Self {
        Self::Balanced
    }
}

/// Real-time processing constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RealtimeMode {
    /// No real-time constraints
    None,
    /// Soft real-time (best effort)
    Soft,
    /// Hard real-time (strict deadlines)
    Hard,
}

impl Default for RealtimeMode {
    fn default() -> Self {
        Self::None
    }
}

/// Streaming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Buffer strategy for streaming
    pub buffer_strategy: BufferStrategy,
    /// Latency optimization mode
    pub latency_mode: LatencyMode,
    /// Real-time processing mode
    pub realtime_mode: RealtimeMode,
    /// Performance profile
    pub profile: PerformanceProfile,
    /// Chunk size in samples
    pub chunk_size: usize,
    /// Overlap between chunks in samples
    pub overlap_samples: usize,
    /// Number of buffers to pre-allocate
    pub buffer_count: usize,
    /// Maximum buffer size in samples
    pub max_buffer_size: usize,
    /// Target latency in milliseconds
    pub target_latency_ms: f32,
    /// Maximum acceptable latency in milliseconds
    pub max_latency_ms: f32,
    /// Enable look-ahead processing
    pub enable_lookahead: bool,
    /// Look-ahead window size in samples
    pub lookahead_samples: usize,
    /// Enable predictive processing
    pub enable_prediction: bool,
    /// Number of processing threads
    pub thread_count: usize,
    /// Thread priority (0-99, higher = more priority)
    pub thread_priority: u8,
    /// Enable memory pinning for GPU processing
    pub enable_memory_pinning: bool,
    /// Enable NUMA-aware processing
    pub enable_numa_aware: bool,
    /// Timeout for processing operations
    pub processing_timeout_ms: u64,
    /// Enable adaptive chunk sizing
    pub enable_adaptive_chunking: bool,
    /// Minimum chunk size for adaptive mode
    pub min_chunk_size: usize,
    /// Maximum chunk size for adaptive mode
    pub max_chunk_size: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_strategy: BufferStrategy::Dynamic,
            latency_mode: LatencyMode::Balanced,
            realtime_mode: RealtimeMode::None,
            profile: PerformanceProfile::Balanced,
            chunk_size: 1024,
            overlap_samples: 256,
            buffer_count: 4,
            max_buffer_size: 8192,
            target_latency_ms: 50.0,
            max_latency_ms: 200.0,
            enable_lookahead: true,
            lookahead_samples: 512,
            enable_prediction: false,
            thread_count: 0, // 0 = auto-detect
            thread_priority: 50,
            enable_memory_pinning: false,
            enable_numa_aware: false,
            processing_timeout_ms: 1000,
            enable_adaptive_chunking: true,
            min_chunk_size: 256,
            max_chunk_size: 4096,
        }
    }
}

impl StreamingConfig {
    /// Create configuration for high-quality streaming
    pub fn high_quality() -> Self {
        Self {
            buffer_strategy: BufferStrategy::Circular,
            latency_mode: LatencyMode::Quality,
            realtime_mode: RealtimeMode::Soft,
            profile: PerformanceProfile::Quality,
            chunk_size: 2048,
            overlap_samples: 512,
            buffer_count: 8,
            max_buffer_size: 16384,
            target_latency_ms: 100.0,
            max_latency_ms: 500.0,
            enable_lookahead: true,
            lookahead_samples: 1024,
            enable_prediction: true,
            thread_count: 0,
            thread_priority: 70,
            enable_memory_pinning: true,
            enable_numa_aware: true,
            processing_timeout_ms: 2000,
            enable_adaptive_chunking: false,
            min_chunk_size: 1024,
            max_chunk_size: 8192,
        }
    }

    /// Create configuration for real-time streaming
    pub fn realtime() -> Self {
        Self {
            buffer_strategy: BufferStrategy::LockFree,
            latency_mode: LatencyMode::UltraLow,
            realtime_mode: RealtimeMode::Hard,
            profile: PerformanceProfile::Realtime,
            chunk_size: 256,
            overlap_samples: 64,
            buffer_count: 2,
            max_buffer_size: 1024,
            target_latency_ms: 10.0,
            max_latency_ms: 30.0,
            enable_lookahead: false,
            lookahead_samples: 0,
            enable_prediction: false,
            thread_count: 1,
            thread_priority: 99,
            enable_memory_pinning: true,
            enable_numa_aware: false,
            processing_timeout_ms: 50,
            enable_adaptive_chunking: false,
            min_chunk_size: 128,
            max_chunk_size: 512,
        }
    }

    /// Create configuration for low-resource environments
    pub fn low_resource() -> Self {
        Self {
            buffer_strategy: BufferStrategy::Fixed,
            latency_mode: LatencyMode::Low,
            realtime_mode: RealtimeMode::None,
            profile: PerformanceProfile::Speed,
            chunk_size: 512,
            overlap_samples: 128,
            buffer_count: 2,
            max_buffer_size: 2048,
            target_latency_ms: 100.0,
            max_latency_ms: 300.0,
            enable_lookahead: false,
            lookahead_samples: 0,
            enable_prediction: false,
            thread_count: 1,
            thread_priority: 30,
            enable_memory_pinning: false,
            enable_numa_aware: false,
            processing_timeout_ms: 500,
            enable_adaptive_chunking: true,
            min_chunk_size: 256,
            max_chunk_size: 1024,
        }
    }

    /// Validate the streaming configuration
    pub fn validate(&self) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Validate chunk size
        if self.chunk_size == 0 {
            errors.push("Chunk size must be greater than 0".to_string());
        } else if self.chunk_size > 16384 {
            warnings.push("Large chunk sizes may increase latency".to_string());
        }

        // Validate overlap
        if self.overlap_samples >= self.chunk_size {
            errors.push("Overlap samples must be less than chunk size".to_string());
        }

        // Validate buffer count
        if self.buffer_count == 0 {
            errors.push("Buffer count must be at least 1".to_string());
        } else if self.buffer_count > 32 {
            warnings.push("High buffer count may increase memory usage".to_string());
        }

        // Validate max buffer size
        if self.max_buffer_size < self.chunk_size {
            errors.push("Max buffer size must be at least chunk size".to_string());
        }

        // Validate latency settings
        if self.target_latency_ms <= 0.0 {
            errors.push("Target latency must be positive".to_string());
        }
        if self.max_latency_ms <= self.target_latency_ms {
            errors.push("Max latency must be greater than target latency".to_string());
        }

        // Validate lookahead settings
        if self.enable_lookahead && self.lookahead_samples == 0 {
            warnings.push("Lookahead enabled but lookahead samples is 0".to_string());
        }
        if self.lookahead_samples > self.chunk_size {
            warnings.push("Lookahead samples should not exceed chunk size".to_string());
        }

        // Validate thread settings
        if self.thread_count > 64 {
            warnings.push("High thread count may cause contention".to_string());
        }
        if self.thread_priority > 99 {
            errors.push("Thread priority must be between 0 and 99".to_string());
        }

        // Validate adaptive chunking
        if self.enable_adaptive_chunking {
            if self.min_chunk_size == 0 {
                errors.push("Minimum chunk size must be greater than 0".to_string());
            }
            if self.max_chunk_size < self.min_chunk_size {
                errors.push("Maximum chunk size must be greater than minimum".to_string());
            }
            if self.chunk_size < self.min_chunk_size || self.chunk_size > self.max_chunk_size {
                warnings.push("Initial chunk size should be within adaptive range".to_string());
            }
        }

        // Real-time mode specific validations
        match self.realtime_mode {
            RealtimeMode::Hard => {
                if self.target_latency_ms > 50.0 {
                    warnings.push("High target latency may not be suitable for hard real-time".to_string());
                }
                if self.enable_lookahead {
                    warnings.push("Lookahead may add latency in hard real-time mode".to_string());
                }
                if self.buffer_count > 4 {
                    warnings.push("High buffer count may increase latency".to_string());
                }
            }
            RealtimeMode::Soft => {
                if self.processing_timeout_ms < 100 {
                    warnings.push("Low timeout may cause processing failures".to_string());
                }
            }
            RealtimeMode::None => {}
        }

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
        }
    }

    /// Get estimated memory usage in MB
    pub fn estimated_memory_mb(&self) -> f32 {
        let buffer_memory = (self.buffer_count * self.max_buffer_size * 4) as f32; // Float values
        let lookahead_memory = if self.enable_lookahead {
            (self.lookahead_samples * 4) as f32
        } else {
            0.0
        };
        
        (buffer_memory + lookahead_memory) / (1024.0 * 1024.0)
    }

    /// Get estimated latency in milliseconds
    pub fn estimated_latency_ms(&self, sample_rate: u32) -> f32 {
        let chunk_latency = (self.chunk_size as f32 / sample_rate as f32) * 1000.0;
        let buffer_latency = (self.buffer_count as f32 * chunk_latency) / 2.0;
        let lookahead_latency = if self.enable_lookahead {
            (self.lookahead_samples as f32 / sample_rate as f32) * 1000.0
        } else {
            0.0
        };
        
        chunk_latency + buffer_latency + lookahead_latency
    }

    /// Get processing timeout as Duration
    pub fn processing_timeout(&self) -> Duration {
        Duration::from_millis(self.processing_timeout_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_streaming_config() {
        let config = StreamingConfig::default();
        assert_eq!(config.chunk_size, 1024);
        assert_eq!(config.buffer_count, 4);
        assert!(config.enable_adaptive_chunking);
    }

    #[test]
    fn test_high_quality_config() {
        let config = StreamingConfig::high_quality();
        assert_eq!(config.latency_mode, LatencyMode::Quality);
        assert_eq!(config.chunk_size, 2048);
        assert!(config.enable_prediction);
    }

    #[test]
    fn test_realtime_config() {
        let config = StreamingConfig::realtime();
        assert_eq!(config.realtime_mode, RealtimeMode::Hard);
        assert_eq!(config.latency_mode, LatencyMode::UltraLow);
        assert!(!config.enable_lookahead);
    }

    #[test]
    fn test_config_validation() {
        let mut config = StreamingConfig::default();
        
        // Valid configuration
        let result = config.validate();
        assert!(result.is_valid);

        // Invalid chunk size
        config.chunk_size = 0;
        let result = config.validate();
        assert!(!result.is_valid);

        // Invalid overlap
        config.chunk_size = 1024;
        config.overlap_samples = 2048;
        let result = config.validate();
        assert!(!result.is_valid);
    }

    #[test]
    fn test_memory_estimation() {
        let config = StreamingConfig::default();
        let memory_mb = config.estimated_memory_mb();
        assert!(memory_mb > 0.0);
        assert!(memory_mb < 100.0); // Reasonable upper bound
    }

    #[test]
    fn test_latency_estimation() {
        let config = StreamingConfig::default();
        let latency_ms = config.estimated_latency_ms(22050);
        assert!(latency_ms > 0.0);
        assert!(latency_ms < 1000.0); // Reasonable upper bound
    }

    #[test]
    fn test_processing_timeout() {
        let config = StreamingConfig::default();
        let timeout = config.processing_timeout();
        assert_eq!(timeout.as_millis(), 1000);
    }
}