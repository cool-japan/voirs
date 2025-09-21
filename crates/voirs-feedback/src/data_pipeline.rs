//! Efficient data pipeline for real-time and batch data processing
//!
//! This module provides comprehensive data pipeline functionality for processing,
//! transforming, and moving data efficiently across the VoiRS feedback system.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::{broadcast, mpsc, RwLock};
use uuid::Uuid;

use crate::traits::{FeedbackResponse, UserProgress};

/// Data pipeline errors
#[derive(Error, Debug)]
pub enum DataPipelineError {
    /// Processing failed
    #[error("Data processing failed: {message}")]
    ProcessingFailed {
        /// Error message
        message: String,
    },

    /// Pipeline congestion
    #[error("Pipeline is congested, dropping data")]
    PipelineCongested,

    /// Transformation error
    #[error("Data transformation failed: {details}")]
    TransformationError {
        /// Error details
        details: String,
    },

    /// Sink error
    #[error("Data sink error: {sink_name} - {message}")]
    SinkError {
        /// Sink name
        sink_name: String,
        /// Error message
        message: String,
    },

    /// Configuration error
    #[error("Pipeline configuration error: {message}")]
    ConfigurationError {
        /// Error message
        message: String,
    },
}

/// Result type for data pipeline operations
pub type DataPipelineResult<T> = Result<T, DataPipelineError>;

/// Pipeline data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    /// User feedback data
    Feedback(FeedbackResponse),
    /// User progress data
    Progress(UserProgress),
    /// Analytics events
    AnalyticsEvent(AnalyticsEvent),
    /// Real-time metrics
    Metrics(MetricsData),
    /// Raw audio data
    AudioData(AudioEvent),
    /// Custom data type
    Custom {
        /// Type name
        type_name: String,
        /// Serialized data
        data: Vec<u8>,
    },
}

/// Analytics event data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsEvent {
    /// Event ID
    pub id: Uuid,
    /// User ID
    pub user_id: String,
    /// Event type
    pub event_type: String,
    /// Event properties
    pub properties: HashMap<String, String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Metrics data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsData {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Tags
    pub tags: HashMap<String, String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Audio event data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEvent {
    /// Event ID
    pub id: Uuid,
    /// User ID
    pub user_id: String,
    /// Audio samples
    pub samples: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Data processing stage trait
#[async_trait]
pub trait DataProcessor: Send + Sync {
    /// Process data
    async fn process(&self, data: DataType) -> DataPipelineResult<Vec<DataType>>;

    /// Get processor name
    fn name(&self) -> &str;

    /// Check if processor can handle this data type
    fn can_process(&self, data: &DataType) -> bool;
}

/// Data sink trait for outputting processed data
#[async_trait]
pub trait DataSink: Send + Sync {
    /// Write data to sink
    async fn write(&self, data: Vec<DataType>) -> DataPipelineResult<()>;

    /// Get sink name
    fn name(&self) -> &str;

    /// Flush any buffered data
    async fn flush(&self) -> DataPipelineResult<()>;
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum buffer size
    pub max_buffer_size: usize,
    /// Batch size for processing
    pub batch_size: usize,
    /// Processing timeout
    pub processing_timeout: Duration,
    /// Enable parallel processing
    pub enable_parallel_processing: bool,
    /// Maximum concurrent processors
    pub max_concurrent_processors: usize,
    /// Pipeline health check interval
    pub health_check_interval: Duration,
    /// Enable data compression
    pub enable_compression: bool,
    /// Enable data encryption in transit
    pub enable_encryption: bool,
    /// Retry configuration
    pub retry_config: RetryConfig,
}

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_attempts: usize,
    /// Initial retry delay
    pub initial_delay: Duration,
    /// Maximum retry delay
    pub max_delay: Duration,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 10000,
            batch_size: 100,
            processing_timeout: Duration::from_secs(30),
            enable_parallel_processing: true,
            max_concurrent_processors: 4,
            health_check_interval: Duration::from_secs(30),
            enable_compression: true,
            enable_encryption: false,
            retry_config: RetryConfig::default(),
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(60),
            backoff_multiplier: 2.0,
        }
    }
}

/// Pipeline statistics
#[derive(Debug, Default, Clone)]
pub struct PipelineStats {
    /// Total items processed
    pub items_processed: u64,
    /// Items per second throughput
    pub throughput: f64,
    /// Processing errors
    pub processing_errors: u64,
    /// Average processing time
    pub avg_processing_time: Duration,
    /// Current buffer size
    pub current_buffer_size: usize,
    /// Last processing time
    pub last_processing_time: Option<Instant>,
}

/// Batch processing statistics
#[derive(Debug, Default, Clone)]
pub struct BatchProcessingStats {
    /// Number of batches processed
    pub batches_processed: usize,
    /// Total items processed in this operation
    pub items_processed: usize,
    /// Adaptive batch size used
    pub adaptive_batch_size: usize,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Individual batch processing times
    pub batch_processing_times: Vec<Duration>,
    /// Throughput (items per second)
    pub throughput: f64,
    /// Compression ratio achieved (if enabled)
    pub compression_ratio: f64,
}

/// Main data pipeline orchestrator
pub struct DataPipeline {
    /// Pipeline configuration
    config: PipelineConfig,
    /// Data processors in order
    processors: Vec<Arc<dyn DataProcessor>>,
    /// Data sinks
    sinks: Vec<Arc<dyn DataSink>>,
    /// Input data buffer
    buffer: Arc<RwLock<VecDeque<DataType>>>,
    /// Pipeline statistics
    stats: Arc<RwLock<PipelineStats>>,
    /// Processing channel
    tx: mpsc::Sender<DataType>,
    /// Pipeline health status
    is_healthy: Arc<RwLock<bool>>,
}

impl DataPipeline {
    /// Create a new data pipeline
    pub fn new(config: PipelineConfig) -> Self {
        let (tx, _rx) = mpsc::channel(config.max_buffer_size);

        Self {
            config,
            processors: Vec::new(),
            sinks: Vec::new(),
            buffer: Arc::new(RwLock::new(VecDeque::new())),
            stats: Arc::new(RwLock::new(PipelineStats::default())),
            tx,
            is_healthy: Arc::new(RwLock::new(true)),
        }
    }

    /// Add a data processor to the pipeline
    pub fn add_processor(&mut self, processor: Arc<dyn DataProcessor>) {
        self.processors.push(processor);
    }

    /// Add a data sink to the pipeline
    pub fn add_sink(&mut self, sink: Arc<dyn DataSink>) {
        self.sinks.push(sink);
    }

    /// Start the pipeline processing
    pub async fn start(&self) -> DataPipelineResult<()> {
        let (tx, mut rx) = mpsc::channel::<DataType>(self.config.max_buffer_size);

        // Start processing task
        let processors = self.processors.clone();
        let sinks = self.sinks.clone();
        let config = self.config.clone();
        let stats = self.stats.clone();
        let is_healthy = self.is_healthy.clone();
        let buffer = self.buffer.clone();

        tokio::spawn(async move {
            while let Some(data) = rx.recv().await {
                let start_time = Instant::now();

                // Add to buffer
                {
                    let mut buf = buffer.write().await;
                    buf.push_back(data.clone());

                    // Check buffer size
                    if buf.len() > config.max_buffer_size {
                        buf.pop_front(); // Drop oldest data
                    }
                }

                // Process data through all processors
                let mut processed_data = vec![data];
                for processor in &processors {
                    let mut next_batch = Vec::new();
                    for item in processed_data {
                        if processor.can_process(&item) {
                            match processor.process(item).await {
                                Ok(mut results) => next_batch.append(&mut results),
                                Err(e) => {
                                    eprintln!("Processing error in {}: {}", processor.name(), e);
                                    let mut stats = stats.write().await;
                                    stats.processing_errors += 1;
                                    *is_healthy.write().await = false;
                                }
                            }
                        } else {
                            next_batch.push(item);
                        }
                    }
                    processed_data = next_batch;
                }

                // Send to sinks
                if !processed_data.is_empty() {
                    for sink in &sinks {
                        if let Err(e) = sink.write(processed_data.clone()).await {
                            eprintln!("Sink error in {}: {}", sink.name(), e);
                            let mut stats = stats.write().await;
                            stats.processing_errors += 1;
                        }
                    }
                }

                // Update statistics
                {
                    let mut stats = stats.write().await;
                    stats.items_processed += 1;
                    let processing_time = start_time.elapsed();
                    stats.avg_processing_time = if stats.items_processed == 1 {
                        processing_time
                    } else {
                        Duration::from_nanos(
                            ((stats.avg_processing_time.as_nanos() as u64
                                * (stats.items_processed - 1))
                                + processing_time.as_nanos() as u64)
                                / stats.items_processed,
                        )
                    };
                    stats.last_processing_time = Some(start_time);
                    stats.current_buffer_size = buffer.read().await.len();

                    // Calculate throughput
                    if let Some(last_time) = stats.last_processing_time {
                        let elapsed = last_time.elapsed().as_secs_f64();
                        if elapsed > 0.0 {
                            stats.throughput = stats.items_processed as f64 / elapsed;
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Send data to the pipeline
    pub async fn send(&self, data: DataType) -> DataPipelineResult<()> {
        self.tx
            .send(data)
            .await
            .map_err(|_| DataPipelineError::PipelineCongested)
    }

    /// Get pipeline statistics
    pub async fn get_stats(&self) -> PipelineStats {
        self.stats.read().await.clone()
    }

    /// Check pipeline health
    pub async fn is_healthy(&self) -> bool {
        *self.is_healthy.read().await
    }

    /// Flush all sinks
    pub async fn flush(&self) -> DataPipelineResult<()> {
        for sink in &self.sinks {
            sink.flush().await?;
        }
        Ok(())
    }

    /// Advanced batch processing with intelligent sizing
    pub async fn process_batch_intelligent(&self) -> DataPipelineResult<BatchProcessingStats> {
        let start_time = Instant::now();
        let mut batch_stats = BatchProcessingStats::default();

        // Get current buffer state
        let buffer_snapshot = {
            let buffer = self.buffer.read().await;
            buffer.clone()
        };

        if buffer_snapshot.is_empty() {
            return Ok(batch_stats);
        }

        // Convert VecDeque to Vec for chunking
        let buffer_vec: Vec<DataType> = buffer_snapshot.into_iter().collect();

        // Adaptive batch sizing based on data complexity and system load
        let adaptive_batch_size = self.calculate_adaptive_batch_size_vec(&buffer_vec).await;
        batch_stats.adaptive_batch_size = adaptive_batch_size;

        // Process in intelligent batches
        let mut processed_items = 0;
        for batch_chunk in buffer_vec.chunks(adaptive_batch_size) {
            let batch_start = Instant::now();

            // Parallel batch processing if enabled
            let processed_batch = if self.config.enable_parallel_processing {
                self.process_batch_parallel(batch_chunk.to_vec()).await?
            } else {
                self.process_batch_sequential(batch_chunk.to_vec()).await?
            };

            // Compress batch if enabled
            let final_batch = if self.config.enable_compression {
                self.compress_batch_data(processed_batch).await?
            } else {
                processed_batch
            };

            // Send to sinks with retry logic
            for sink in &self.sinks {
                self.write_to_sink_with_retry(sink.clone(), final_batch.clone())
                    .await?;
            }

            processed_items += batch_chunk.len();
            batch_stats.batches_processed += 1;
            batch_stats.items_processed += batch_chunk.len();
            batch_stats
                .batch_processing_times
                .push(batch_start.elapsed());
        }

        // Update pipeline statistics
        {
            let mut stats = self.stats.write().await;
            stats.items_processed += processed_items as u64;
            let total_time = start_time.elapsed();
            batch_stats.total_processing_time = total_time;

            // Calculate throughput
            batch_stats.throughput = processed_items as f64 / total_time.as_secs_f64();
            stats.throughput = batch_stats.throughput;
        }

        // Clear processed items from buffer
        {
            let mut buffer = self.buffer.write().await;
            let buffer_len = buffer.len();
            let drain_count = processed_items.min(buffer_len);
            buffer.drain(..drain_count);
        }

        Ok(batch_stats)
    }

    /// Calculate adaptive batch size based on data complexity and system load
    async fn calculate_adaptive_batch_size_vec(&self, buffer: &[DataType]) -> usize {
        let base_batch_size = self.config.batch_size;
        let buffer_size = buffer.len();

        // Factor 1: Buffer pressure (more items = larger batches)
        let buffer_factor = if buffer_size > self.config.max_buffer_size / 2 {
            1.5 // Increase batch size when buffer is getting full
        } else if buffer_size < self.config.max_buffer_size / 10 {
            0.7 // Decrease batch size when buffer is mostly empty
        } else {
            1.0
        };

        // Factor 2: Data complexity (larger data = smaller batches)
        let complexity_factor = self.analyze_data_complexity_vec(buffer).await;

        // Factor 3: Recent processing performance
        let performance_factor = {
            let stats = self.stats.read().await;
            if stats.avg_processing_time > Duration::from_millis(1000) {
                0.8 // Slower processing = smaller batches
            } else if stats.avg_processing_time < Duration::from_millis(100) {
                1.2 // Faster processing = larger batches
            } else {
                1.0
            }
        };

        let adaptive_size =
            (base_batch_size as f64 * buffer_factor * complexity_factor * performance_factor)
                as usize;
        adaptive_size.max(1).min(self.config.max_buffer_size / 2)
    }

    /// Analyze data complexity to optimize batch sizing
    async fn analyze_data_complexity_vec(&self, buffer: &[DataType]) -> f64 {
        if buffer.is_empty() {
            return 1.0;
        }

        let sample_size = buffer.len().min(50); // Analyze up to 50 items
        let mut complexity_score = 0.0;

        for (i, item) in buffer.iter().enumerate() {
            if i >= sample_size {
                break;
            }

            let item_complexity = match item {
                DataType::AudioData(audio) => {
                    // Large audio data = high complexity
                    (audio.samples.len() as f64 / 44100.0).min(5.0) // Normalize by 1 second of audio
                }
                DataType::Feedback(_) => 1.0,
                DataType::Progress(_) => 0.8,
                DataType::AnalyticsEvent(_) => 0.5,
                DataType::Metrics(_) => 0.3,
                DataType::Custom { data, .. } => {
                    // Custom data complexity based on size
                    (data.len() as f64 / 1024.0).min(3.0) // Normalize by 1KB
                }
            };

            complexity_score += item_complexity;
        }

        let avg_complexity = complexity_score / sample_size as f64;

        // Return factor to adjust batch size (higher complexity = smaller batches)
        if avg_complexity > 2.0 {
            0.6 // High complexity data
        } else if avg_complexity > 1.0 {
            0.8 // Medium complexity data
        } else {
            1.0 // Low complexity data
        }
    }

    /// Process batch in parallel using multiple workers
    async fn process_batch_parallel(
        &self,
        batch: Vec<DataType>,
    ) -> DataPipelineResult<Vec<DataType>> {
        use futures::future::try_join_all;

        let chunk_size = (batch.len() / self.config.max_concurrent_processors).max(1);
        let mut futures = Vec::new();

        for chunk in batch.chunks(chunk_size) {
            let chunk_vec = chunk.to_vec();
            let processors = self.processors.clone();

            let future = tokio::spawn(async move {
                Self::process_chunk_through_processors(chunk_vec, processors).await
            });

            futures.push(future);
        }

        let results =
            try_join_all(futures)
                .await
                .map_err(|e| DataPipelineError::ProcessingFailed {
                    message: format!("Parallel processing failed: {}", e),
                })?;

        let mut final_result = Vec::new();
        for result in results {
            match result {
                Ok(processed_data) => final_result.extend(processed_data),
                Err(e) => return Err(e),
            }
        }

        Ok(final_result)
    }

    /// Process batch sequentially
    async fn process_batch_sequential(
        &self,
        batch: Vec<DataType>,
    ) -> DataPipelineResult<Vec<DataType>> {
        Self::process_chunk_through_processors(batch, self.processors.clone()).await
    }

    /// Process a chunk of data through all processors
    async fn process_chunk_through_processors(
        mut data: Vec<DataType>,
        processors: Vec<Arc<dyn DataProcessor>>,
    ) -> DataPipelineResult<Vec<DataType>> {
        for processor in &processors {
            let mut next_batch = Vec::new();
            for item in data {
                if processor.can_process(&item) {
                    match processor.process(item).await {
                        Ok(mut results) => next_batch.append(&mut results),
                        Err(e) => {
                            return Err(DataPipelineError::ProcessingFailed {
                                message: format!("Processor {} failed: {}", processor.name(), e),
                            });
                        }
                    }
                } else {
                    next_batch.push(item);
                }
            }
            data = next_batch;
        }
        Ok(data)
    }

    /// Compress batch data for efficient storage/transmission
    async fn compress_batch_data(&self, batch: Vec<DataType>) -> DataPipelineResult<Vec<DataType>> {
        // For large batches, compress similar data types together
        if batch.len() < 10 {
            return Ok(batch); // Skip compression for small batches
        }

        let mut compressed_batch = Vec::new();
        let mut audio_events = Vec::new();
        let mut analytics_events = Vec::new();

        // Group similar data types for better compression
        for item in batch {
            match item {
                DataType::AudioData(audio) => audio_events.push(audio),
                DataType::AnalyticsEvent(event) => analytics_events.push(event),
                other => compressed_batch.push(other), // Pass through other types
            }
        }

        // Compress audio events if we have multiple
        if audio_events.len() > 1 {
            let compressed_audio = self.compress_audio_events(audio_events).await?;
            compressed_batch.extend(compressed_audio.into_iter().map(DataType::AudioData));
        } else {
            compressed_batch.extend(audio_events.into_iter().map(DataType::AudioData));
        }

        // Compress analytics events if we have multiple
        if analytics_events.len() > 1 {
            let compressed_analytics = self.compress_analytics_events(analytics_events).await?;
            compressed_batch.extend(
                compressed_analytics
                    .into_iter()
                    .map(DataType::AnalyticsEvent),
            );
        } else {
            compressed_batch.extend(analytics_events.into_iter().map(DataType::AnalyticsEvent));
        }

        Ok(compressed_batch)
    }

    /// Compress multiple audio events
    async fn compress_audio_events(
        &self,
        events: Vec<AudioEvent>,
    ) -> DataPipelineResult<Vec<AudioEvent>> {
        // Simple compression: merge audio events from same user within short time window
        let mut compressed = Vec::new();
        let mut current_group: Option<AudioEvent> = None;

        for event in events {
            match &mut current_group {
                Some(group)
                    if group.user_id == event.user_id
                        && group.sample_rate == event.sample_rate
                        && (event.timestamp - group.timestamp).num_seconds() < 5 =>
                {
                    // Merge into current group
                    group.samples.extend(event.samples);
                }
                _ => {
                    // Start new group
                    if let Some(group) = current_group.take() {
                        compressed.push(group);
                    }
                    current_group = Some(event);
                }
            }
        }

        if let Some(group) = current_group {
            compressed.push(group);
        }

        Ok(compressed)
    }

    /// Compress multiple analytics events
    async fn compress_analytics_events(
        &self,
        events: Vec<AnalyticsEvent>,
    ) -> DataPipelineResult<Vec<AnalyticsEvent>> {
        // Simple compression: combine events of same type from same user
        let mut event_map: HashMap<(String, String), AnalyticsEvent> = HashMap::new();

        for event in events {
            let key = (event.user_id.clone(), event.event_type.clone());

            match event_map.get_mut(&key) {
                Some(existing) => {
                    // Merge properties and update timestamp to latest
                    existing.properties.extend(event.properties);
                    if event.timestamp > existing.timestamp {
                        existing.timestamp = event.timestamp;
                    }
                }
                None => {
                    event_map.insert(key, event);
                }
            }
        }

        Ok(event_map.into_values().collect())
    }

    /// Write to sink with retry logic
    async fn write_to_sink_with_retry(
        &self,
        sink: Arc<dyn DataSink>,
        data: Vec<DataType>,
    ) -> DataPipelineResult<()> {
        let mut attempts = 0;
        let mut delay = self.config.retry_config.initial_delay;

        while attempts < self.config.retry_config.max_attempts {
            match sink.write(data.clone()).await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    attempts += 1;
                    if attempts >= self.config.retry_config.max_attempts {
                        return Err(DataPipelineError::SinkError {
                            sink_name: sink.name().to_string(),
                            message: format!("Failed after {} attempts: {}", attempts, e),
                        });
                    }

                    // Exponential backoff
                    tokio::time::sleep(delay).await;
                    delay = Duration::from_millis(
                        (delay.as_millis() as f64 * self.config.retry_config.backoff_multiplier)
                            as u64,
                    )
                    .min(self.config.retry_config.max_delay);
                }
            }
        }

        Err(DataPipelineError::SinkError {
            sink_name: sink.name().to_string(),
            message: "Max retry attempts exceeded".to_string(),
        })
    }

    /// Schedule batch processing based on optimal timing
    pub async fn schedule_batch_processing(&self) -> DataPipelineResult<()> {
        let buffer_size = self.buffer.read().await.len();
        let stats = self.stats.read().await.clone();

        // Determine optimal processing trigger
        let should_process = self
            .should_trigger_batch_processing(buffer_size, &stats)
            .await;

        if should_process {
            self.process_batch_intelligent().await?;
        }

        Ok(())
    }

    /// Determine if batch processing should be triggered
    async fn should_trigger_batch_processing(
        &self,
        buffer_size: usize,
        stats: &PipelineStats,
    ) -> bool {
        // Trigger conditions:

        // 1. Buffer is getting full
        if buffer_size >= self.config.max_buffer_size * 3 / 4 {
            return true;
        }

        // 2. Minimum batch size reached and some time has passed
        if buffer_size >= self.config.batch_size {
            if let Some(last_processing) = stats.last_processing_time {
                if last_processing.elapsed() > Duration::from_secs(5) {
                    return true;
                }
            }
        }

        // 3. Long time since last processing (avoid data staleness)
        if let Some(last_processing) = stats.last_processing_time {
            if last_processing.elapsed() > Duration::from_secs(30) && buffer_size > 0 {
                return true;
            }
        }

        // 4. Initial processing (no previous processing time)
        if stats.last_processing_time.is_none() && buffer_size > 0 {
            return true;
        }

        false
    }
}

/// Example feedback processor
pub struct FeedbackProcessor {
    name: String,
}

impl FeedbackProcessor {
    /// Create a new feedback processor
    pub fn new() -> Self {
        Self {
            name: "FeedbackProcessor".to_string(),
        }
    }
}

#[async_trait]
impl DataProcessor for FeedbackProcessor {
    async fn process(&self, data: DataType) -> DataPipelineResult<Vec<DataType>> {
        match data {
            DataType::Feedback(mut feedback) => {
                // Example processing: enrich feedback items with metadata
                for item in &mut feedback.feedback_items {
                    item.metadata
                        .insert("processed_by".to_string(), "FeedbackProcessor".to_string());
                    item.metadata
                        .insert("processed_at".to_string(), Utc::now().to_rfc3339());
                }

                Ok(vec![DataType::Feedback(feedback)])
            }
            other => Ok(vec![other]), // Pass through other data types
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn can_process(&self, data: &DataType) -> bool {
        matches!(data, DataType::Feedback(_))
    }
}

/// Example analytics sink
pub struct AnalyticsSink {
    name: String,
    buffer: Arc<RwLock<Vec<DataType>>>,
}

impl AnalyticsSink {
    /// Create a new analytics sink
    pub fn new() -> Self {
        Self {
            name: "AnalyticsSink".to_string(),
            buffer: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

#[async_trait]
impl DataSink for AnalyticsSink {
    async fn write(&self, data: Vec<DataType>) -> DataPipelineResult<()> {
        let mut buffer = self.buffer.write().await;
        buffer.extend(data);

        // Simulate analytics processing
        if buffer.len() > 100 {
            println!("Analytics: Processing {} items", buffer.len());
            buffer.clear();
        }

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn flush(&self) -> DataPipelineResult<()> {
        let mut buffer = self.buffer.write().await;
        if !buffer.is_empty() {
            println!("Analytics: Flushing {} items", buffer.len());
            buffer.clear();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_creation() {
        let config = PipelineConfig::default();
        let pipeline = DataPipeline::new(config);
        assert!(pipeline.is_healthy().await);
    }

    #[tokio::test]
    async fn test_feedback_processor() {
        let processor = FeedbackProcessor::new();
        let mut feedback_item = crate::traits::UserFeedback {
            message: "Test feedback".to_string(),
            suggestion: Some("Improve pronunciation".to_string()),
            confidence: 0.95,
            score: 0.85,
            priority: 0.7,
            metadata: HashMap::new(),
        };

        let feedback = FeedbackResponse {
            feedback_items: vec![feedback_item],
            overall_score: 0.85,
            immediate_actions: vec!["Practice pronunciation".to_string()],
            long_term_goals: vec!["Improve fluency".to_string()],
            progress_indicators: crate::traits::ProgressIndicators::default(),
            timestamp: Utc::now(),
            processing_time: std::time::Duration::from_millis(100),
            feedback_type: crate::traits::FeedbackType::Pronunciation,
        };

        let data = DataType::Feedback(feedback);
        assert!(processor.can_process(&data));

        let result = processor.process(data).await.unwrap();
        assert_eq!(result.len(), 1);

        if let DataType::Feedback(processed_feedback) = &result[0] {
            assert!(!processed_feedback.feedback_items.is_empty());
            assert!(processed_feedback.feedback_items[0]
                .metadata
                .contains_key("processed_by"));
        }
    }

    #[tokio::test]
    async fn test_analytics_sink() {
        let sink = AnalyticsSink::new();
        let feedback_item = crate::traits::UserFeedback {
            message: "Test feedback".to_string(),
            suggestion: Some("Improve pronunciation".to_string()),
            confidence: 0.95,
            score: 0.85,
            priority: 0.7,
            metadata: HashMap::new(),
        };

        let feedback = FeedbackResponse {
            feedback_items: vec![feedback_item],
            overall_score: 0.85,
            immediate_actions: vec!["Practice pronunciation".to_string()],
            long_term_goals: vec!["Improve fluency".to_string()],
            progress_indicators: crate::traits::ProgressIndicators::default(),
            timestamp: Utc::now(),
            processing_time: std::time::Duration::from_millis(100),
            feedback_type: crate::traits::FeedbackType::Pronunciation,
        };

        let data = vec![DataType::Feedback(feedback)];
        let result = sink.write(data).await;
        assert!(result.is_ok());

        let flush_result = sink.flush().await;
        assert!(flush_result.is_ok());
    }

    #[tokio::test]
    async fn test_pipeline_stats() {
        let config = PipelineConfig::default();
        let pipeline = DataPipeline::new(config);

        let stats = pipeline.get_stats().await;
        assert_eq!(stats.items_processed, 0);
        assert_eq!(stats.processing_errors, 0);
    }

    #[tokio::test]
    async fn test_data_types() {
        let event = AnalyticsEvent {
            id: Uuid::new_v4(),
            user_id: "test_user".to_string(),
            event_type: "click".to_string(),
            properties: HashMap::new(),
            timestamp: Utc::now(),
        };

        let data = DataType::AnalyticsEvent(event);
        match data {
            DataType::AnalyticsEvent(_) => assert!(true),
            _ => assert!(false, "Unexpected data type"),
        }
    }

    #[tokio::test]
    async fn test_config_defaults() {
        let config = PipelineConfig::default();
        assert_eq!(config.max_buffer_size, 10000);
        assert_eq!(config.batch_size, 100);
        assert!(config.enable_parallel_processing);
        assert_eq!(config.max_concurrent_processors, 4);

        let retry_config = config.retry_config;
        assert_eq!(retry_config.max_attempts, 3);
        assert_eq!(retry_config.backoff_multiplier, 2.0);
    }
}
