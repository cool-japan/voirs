//! Real-time streaming architecture for neural vocoders
//!
//! This module provides a complete streaming pipeline for real-time audio generation
//! with latency optimization, chunk-based processing, and adaptive buffering.

pub mod buffer;
pub mod chunk_processor;
pub mod interrupt_processor;
pub mod latency;
pub mod latency_optimizer;
pub mod memory_buffer;
pub mod pipeline;
pub mod realtime_scheduler;

pub use buffer::{BufferManager, RingBuffer, StreamingBuffer};
pub use chunk_processor::{
    AdvancedChunkConfig, AdvancedChunkProcessor, AdvancedChunkStats, WindowType,
};
pub use interrupt_processor::{
    BufferEventType, InterruptContext, InterruptController, InterruptData, InterruptPriority,
    InterruptResponse, SystemCommand,
};
pub use latency::{LatencyOptimizer, PredictiveProcessor};
pub use latency_optimizer::{EnhancedLatencyOptimizer, EnhancedLatencyStats};
pub use memory_buffer::{
    AllocationStrategy, LockFreeCircularBuffer, MemoryEfficientBufferManager, MemoryStats,
};
pub use pipeline::{ChunkProcessor, StreamingPipeline};
pub use realtime_scheduler::{
    EnhancedRtScheduler, LoadBalancingStrategy, RtPriority, RtTask, SchedulerConfig, SchedulerStats,
};

// Re-export from config for convenience
pub use crate::config::StreamingConfig;
use crate::{AudioBuffer, MelSpectrogram, Result, VocoderError};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// Streaming vocoder trait for real-time audio generation
#[async_trait]
pub trait StreamingVocoder: Send + Sync {
    /// Initialize the streaming vocoder with configuration
    async fn initialize(&mut self, config: StreamingConfig) -> Result<()>;

    /// Start streaming processing
    async fn start_stream(&self) -> Result<StreamHandle>;

    /// Process a mel spectrogram chunk
    async fn process_chunk(&self, mel_chunk: MelSpectrogram) -> Result<AudioBuffer>;

    /// Process multiple chunks in parallel
    async fn process_batch(&self, mel_chunks: Vec<MelSpectrogram>) -> Result<Vec<AudioBuffer>>;

    /// Stop streaming and cleanup resources
    async fn stop_stream(&self) -> Result<()>;

    /// Get current streaming statistics
    fn get_stats(&self) -> StreamingStats;
}

/// Handle for managing an active stream
pub struct StreamHandle {
    /// Input channel for mel spectrograms
    pub input_tx: mpsc::Sender<MelSpectrogram>,

    /// Output channel for audio buffers
    pub output_rx: mpsc::Receiver<AudioBuffer>,

    /// Control channel for stream management
    pub control_tx: mpsc::Sender<StreamCommand>,

    /// Stream ID for tracking
    pub stream_id: u64,
}

/// Commands for controlling the streaming process
#[derive(Debug, Clone)]
pub enum StreamCommand {
    /// Pause streaming
    Pause,
    /// Resume streaming
    Resume,
    /// Flush buffers
    Flush,
    /// Update configuration
    UpdateConfig(StreamingConfig),
    /// Get current statistics
    GetStats,
    /// Shutdown stream
    Shutdown,
}

/// Streaming performance statistics
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    /// Total processed chunks
    pub chunks_processed: u64,

    /// Average processing time per chunk (ms)
    pub avg_processing_time_ms: f32,

    /// Current latency (ms)
    pub current_latency_ms: f32,

    /// Peak latency (ms)
    pub peak_latency_ms: f32,

    /// Buffer underruns
    pub buffer_underruns: u64,

    /// Buffer overruns
    pub buffer_overruns: u64,

    /// Real-time factor (1.0 = real-time)
    pub real_time_factor: f32,

    /// CPU usage percentage
    pub cpu_usage: f32,

    /// Memory usage (MB)
    pub memory_usage_mb: f32,

    /// Active stream count
    pub active_streams: u32,

    /// Error count
    pub error_count: u64,
}

impl StreamingStats {
    /// Update real-time factor calculation
    pub fn update_rtf(&mut self, audio_duration_ms: f32, processing_time_ms: f32) {
        if audio_duration_ms > 0.0 {
            self.real_time_factor = processing_time_ms / audio_duration_ms;
        }
    }

    /// Check if streaming is meeting real-time requirements
    pub fn is_real_time(&self) -> bool {
        self.real_time_factor <= 1.0
    }

    /// Get quality score (0.0-1.0, higher is better)
    pub fn quality_score(&self) -> f32 {
        let rtf_score = if self.real_time_factor <= 1.0 {
            1.0
        } else {
            (2.0 - self.real_time_factor).max(0.0)
        };

        let latency_score = if self.current_latency_ms <= 50.0 {
            1.0
        } else if self.current_latency_ms <= 200.0 {
            1.0 - (self.current_latency_ms - 50.0) / 150.0
        } else {
            0.0
        };

        let error_score = if self.error_count == 0 {
            1.0
        } else {
            (1.0 / (1.0 + self.error_count as f32)).max(0.1)
        };

        (rtf_score + latency_score + error_score) / 3.0
    }
}

/// Streaming error types
#[derive(Debug, Clone)]
pub enum StreamingError {
    /// Buffer overflow
    BufferOverflow,
    /// Buffer underflow
    BufferUnderflow,
    /// Latency exceeded threshold
    LatencyExceeded(f32),
    /// Processing timeout
    ProcessingTimeout,
    /// Stream not initialized
    NotInitialized,
    /// Invalid chunk size
    InvalidChunkSize(usize),
    /// Configuration error
    ConfigurationError(String),
}

impl std::fmt::Display for StreamingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamingError::BufferOverflow => write!(f, "Buffer overflow"),
            StreamingError::BufferUnderflow => write!(f, "Buffer underflow"),
            StreamingError::LatencyExceeded(latency) => {
                write!(f, "Latency exceeded: {latency:.2}ms")
            }
            StreamingError::ProcessingTimeout => write!(f, "Processing timeout"),
            StreamingError::NotInitialized => write!(f, "Stream not initialized"),
            StreamingError::InvalidChunkSize(size) => write!(f, "Invalid chunk size: {size}"),
            StreamingError::ConfigurationError(msg) => write!(f, "Configuration error: {msg}"),
        }
    }
}

impl std::error::Error for StreamingError {}

impl From<StreamingError> for VocoderError {
    fn from(error: StreamingError) -> Self {
        VocoderError::StreamingError(error.to_string())
    }
}

/// Simple buffer pool for common buffer sizes to reduce allocation overhead
pub struct BufferPool {
    pools: Arc<Mutex<HashMap<usize, Vec<Vec<f32>>>>>,
    max_buffers_per_size: usize,
}

impl BufferPool {
    /// Create a new buffer pool
    pub fn new() -> Self {
        Self {
            pools: Arc::new(Mutex::new(HashMap::new())),
            max_buffers_per_size: 16,
        }
    }

    /// Get a buffer of the specified size
    pub fn get_buffer(&self, size: usize) -> Vec<f32> {
        let mut pools = self.pools.lock().unwrap();

        if let Some(pool) = pools.get_mut(&size) {
            if let Some(buffer) = pool.pop() {
                return buffer;
            }
        }

        // Create new buffer if pool is empty
        vec![0.0; size]
    }

    /// Return a buffer to the pool
    pub fn return_buffer(&self, mut buffer: Vec<f32>) {
        let size = buffer.len();
        buffer.clear();
        buffer.resize(size, 0.0);

        let mut pools = self.pools.lock().unwrap();
        let pool = pools.entry(size).or_default();

        if pool.len() < self.max_buffers_per_size {
            pool.push(buffer);
        }
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Global buffer pool instance
use once_cell::sync::Lazy;
static GLOBAL_BUFFER_POOL: Lazy<BufferPool> = Lazy::new(BufferPool::new);

/// Convenience function to get a buffer from the global pool
pub fn get_pooled_buffer(size: usize) -> Vec<f32> {
    GLOBAL_BUFFER_POOL.get_buffer(size)
}

/// Convenience function to return a buffer to the global pool
pub fn return_pooled_buffer(buffer: Vec<f32>) {
    GLOBAL_BUFFER_POOL.return_buffer(buffer);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_stats_rtf_calculation() {
        let mut stats = StreamingStats::default();

        // Test real-time performance
        stats.update_rtf(100.0, 50.0); // 0.5x real-time
        assert!(stats.is_real_time());
        assert_eq!(stats.real_time_factor, 0.5);

        // Test slower than real-time
        stats.update_rtf(100.0, 150.0); // 1.5x real-time
        assert!(!stats.is_real_time());
        assert_eq!(stats.real_time_factor, 1.5);
    }

    #[test]
    fn test_quality_score_calculation() {
        // Perfect quality
        let mut stats = StreamingStats {
            real_time_factor: 0.8,
            current_latency_ms: 30.0,
            error_count: 0,
            ..Default::default()
        };

        let score = stats.quality_score();
        assert!(score > 0.9);

        // Poor quality
        stats.real_time_factor = 2.0;
        stats.current_latency_ms = 500.0;
        stats.error_count = 10;

        let score = stats.quality_score();
        assert!(score < 0.5);
    }

    #[test]
    fn test_streaming_error_display() {
        let error = StreamingError::LatencyExceeded(123.45);
        assert_eq!(error.to_string(), "Latency exceeded: 123.45ms");

        let error = StreamingError::InvalidChunkSize(2048);
        assert_eq!(error.to_string(), "Invalid chunk size: 2048");
    }

    #[test]
    fn test_buffer_pool_optimization() {
        let pool = BufferPool::new();

        // Test allocation and deallocation
        let buffer1 = pool.get_buffer(1024);
        assert_eq!(buffer1.len(), 1024);

        pool.return_buffer(buffer1);

        // Test reuse
        let buffer2 = pool.get_buffer(1024);
        assert_eq!(buffer2.len(), 1024);

        pool.return_buffer(buffer2);
    }
}
