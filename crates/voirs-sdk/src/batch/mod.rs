//! Advanced batch processing optimization for VoiRS SDK.
//!
//! This module provides high-performance batch synthesis capabilities with:
//! - Automatic parallel processing and load balancing
//! - Memory-efficient streaming batch processing
//! - Progress tracking and cancellation support
//! - Adaptive resource allocation
//! - Result aggregation and error handling
//!
//! # Examples
//!
//! ```no_run
//! use voirs_sdk::prelude::*;
//! use voirs_sdk::batch::{BatchProcessor, BatchConfig, BatchRequest};
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let pipeline = VoirsPipelineBuilder::new().build().await?;
//!
//!     let batch_processor = BatchProcessor::new(
//!         Arc::new(pipeline),
//!         BatchConfig::default()
//!     );
//!
//!     let requests = vec![
//!         BatchRequest::new("Hello, world!", None),
//!         BatchRequest::new("How are you?", Some("voice-1")),
//!         BatchRequest::new("Goodbye!", Some("voice-2")),
//!     ];
//!
//!     let results = batch_processor.process(requests).await?;
//!
//!     for (idx, result) in results.iter().enumerate() {
//!         if let Ok(audio) = &result.result {
//!             println!("Request {}: {} samples", idx, audio.samples().len());
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```

use crate::{AudioBuffer, VoirsError, VoirsPipeline, VoirsResult};
use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, info, warn};

mod optimization;
mod processor;
mod scheduler;
mod statistics;

pub use optimization::{
    BatchOptimizer, NormalizationStrategy, OptimizationConfig, OptimizationStats,
};
pub use processor::BatchProcessor;
pub use scheduler::{BatchScheduler, SchedulingStrategy};
pub use statistics::{BatchStatistics, ProcessingMetrics};

/// Configuration for batch processing operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum number of concurrent synthesis operations
    pub max_concurrency: usize,

    /// Maximum batch size before splitting
    pub max_batch_size: usize,

    /// Memory limit per batch in megabytes
    pub memory_limit_mb: usize,

    /// Timeout for individual synthesis operations
    pub synthesis_timeout: Duration,

    /// Enable adaptive resource allocation
    pub adaptive_resources: bool,

    /// Enable progress tracking
    pub track_progress: bool,

    /// Retry failed operations
    pub retry_failed: bool,

    /// Maximum retry attempts
    pub max_retries: usize,

    /// Scheduling strategy
    pub scheduling_strategy: SchedulingStrategy,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_concurrency: num_cpus::get(),
            max_batch_size: 100,
            memory_limit_mb: 2048,
            synthesis_timeout: Duration::from_secs(30),
            adaptive_resources: true,
            track_progress: true,
            retry_failed: true,
            max_retries: 3,
            scheduling_strategy: SchedulingStrategy::LoadBalanced,
        }
    }
}

/// A single batch synthesis request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequest {
    /// Unique request identifier
    pub id: String,

    /// Text to synthesize
    pub text: String,

    /// Optional voice override
    pub voice: Option<String>,

    /// Optional synthesis speed
    pub speed: Option<f32>,

    /// Optional pitch adjustment
    pub pitch: Option<f32>,

    /// Optional priority (higher = processed first)
    pub priority: i32,

    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
}

impl BatchRequest {
    /// Create a new batch request with text.
    pub fn new(text: impl Into<String>, voice: Option<&str>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            text: text.into(),
            voice: voice.map(|v| v.to_string()),
            speed: None,
            pitch: None,
            priority: 0,
            metadata: None,
        }
    }

    /// Set request priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Set synthesis speed.
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = Some(speed);
        self
    }

    /// Set pitch adjustment.
    pub fn with_pitch(mut self, pitch: f32) -> Self {
        self.pitch = Some(pitch);
        self
    }

    /// Set metadata.
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Result of a batch synthesis operation.
#[derive(Debug)]
pub struct BatchResult {
    /// Request ID
    pub request_id: String,

    /// Synthesis result
    pub result: VoirsResult<AudioBuffer>,

    /// Processing time
    pub processing_time: Duration,

    /// Retry count
    pub retry_count: usize,

    /// Worker ID that processed this request
    pub worker_id: Option<usize>,
}

impl BatchResult {
    /// Check if synthesis was successful.
    pub fn is_success(&self) -> bool {
        self.result.is_ok()
    }

    /// Get audio buffer if successful.
    pub fn audio(&self) -> Option<&AudioBuffer> {
        self.result.as_ref().ok()
    }

    /// Get error if failed.
    pub fn error(&self) -> Option<&VoirsError> {
        self.result.as_ref().err()
    }
}

/// Progress callback for batch processing.
pub type ProgressCallback = Arc<dyn Fn(usize, usize) + Send + Sync>;

/// Batch processing context.
#[derive(Clone)]
pub struct BatchContext {
    /// Configuration
    pub config: BatchConfig,

    /// Pipeline instance
    pub pipeline: Arc<VoirsPipeline>,

    /// Concurrency semaphore
    pub semaphore: Arc<Semaphore>,

    /// Statistics
    pub statistics: Arc<RwLock<BatchStatistics>>,

    /// Progress callback
    pub progress_callback: Option<ProgressCallback>,
}

impl BatchContext {
    /// Create new batch context.
    pub fn new(pipeline: Arc<VoirsPipeline>, config: BatchConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrency));
        let statistics = Arc::new(RwLock::new(BatchStatistics::new()));

        Self {
            config,
            pipeline,
            semaphore,
            statistics,
            progress_callback: None,
        }
    }

    /// Set progress callback.
    pub fn with_progress_callback(mut self, callback: ProgressCallback) -> Self {
        self.progress_callback = Some(callback);
        self
    }
}

/// Trait for batch processing strategies.
#[async_trait]
pub trait BatchStrategy: Send + Sync {
    /// Process a batch of requests.
    async fn process_batch(
        &self,
        context: &BatchContext,
        requests: Vec<BatchRequest>,
    ) -> VoirsResult<Vec<BatchResult>>;

    /// Estimate processing time.
    fn estimate_time(&self, requests: &[BatchRequest]) -> Duration;

    /// Estimate memory usage.
    fn estimate_memory(&self, requests: &[BatchRequest]) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert!(config.max_concurrency > 0);
        assert!(config.max_batch_size > 0);
        assert!(config.memory_limit_mb > 0);
    }

    #[test]
    fn test_batch_request_creation() {
        let request = BatchRequest::new("Hello, world!", Some("voice-1"));
        assert_eq!(request.text, "Hello, world!");
        assert_eq!(request.voice.as_deref(), Some("voice-1"));
        assert_eq!(request.priority, 0);
    }

    #[test]
    fn test_batch_request_builder() {
        let request = BatchRequest::new("Test", None)
            .with_priority(10)
            .with_speed(1.2)
            .with_pitch(0.8);

        assert_eq!(request.priority, 10);
        assert_eq!(request.speed, Some(1.2));
        assert_eq!(request.pitch, Some(0.8));
    }

    #[test]
    fn test_batch_result() {
        let result = BatchResult {
            request_id: "test-id".to_string(),
            result: Err(VoirsError::audio_error("test error")),
            processing_time: Duration::from_millis(100),
            retry_count: 2,
            worker_id: Some(1),
        };

        assert!(!result.is_success());
        assert!(result.audio().is_none());
        assert!(result.error().is_some());
    }
}
