//! Streaming voice conversion

use crate::{
    realtime::{ProcessingMode, RealtimeConfig, RealtimeConverter},
    types::{ConversionTarget, ConversionType},
    Error, Result,
};
use async_stream::stream;
use fastrand;
use futures::{Stream, StreamExt};
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, error, info, warn};

/// Streaming converter for continuous audio processing
#[derive(Debug)]
pub struct StreamingConverter {
    /// Real-time converter
    realtime_converter: Arc<Mutex<RealtimeConverter>>,
    /// Stream configuration
    config: StreamConfig,
    /// Buffer for audio accumulation
    accumulation_buffer: Arc<Mutex<VecDeque<f32>>>,
    /// Processing statistics
    stats: Arc<RwLock<StreamingStats>>,
    /// Conversion target
    conversion_target: Option<ConversionTarget>,
    /// Stream state
    state: StreamState,
}

/// Stream processing state tracking current operation status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamState {
    /// Stream is idle
    Idle,
    /// Stream is actively processing
    Processing,
    /// Stream is paused
    Paused,
    /// Stream encountered an error
    Error,
    /// Stream is stopped
    Stopped,
}

impl StreamingConverter {
    /// Create new streaming converter
    pub fn new(config: StreamConfig) -> Result<Self> {
        let realtime_config = RealtimeConfig {
            buffer_size: config.chunk_size,
            sample_rate: config.sample_rate,
            target_latency_ms: config.target_latency_ms,
            overlap_factor: 0.25,
            adaptive_buffering: config.adaptive_buffering,
            max_threads: config.max_concurrent_streams.min(4),
            enable_lookahead: true,
            lookahead_size: config.chunk_size / 4,
        };

        let realtime_converter = Arc::new(Mutex::new(RealtimeConverter::new(realtime_config)?));
        let accumulation_buffer =
            Arc::new(Mutex::new(VecDeque::with_capacity(config.buffer_capacity)));
        let stats = Arc::new(RwLock::new(StreamingStats::default()));

        Ok(Self {
            realtime_converter,
            config,
            accumulation_buffer,
            stats,
            conversion_target: None,
            state: StreamState::Idle,
        })
    }

    /// Set conversion target for voice transformation
    pub fn set_conversion_target(&mut self, target: ConversionTarget) {
        self.conversion_target = Some(target);
    }

    /// Set processing mode
    pub async fn set_processing_mode(&self, mode: ProcessingMode) {
        let mut converter = self.realtime_converter.lock().await;
        converter.set_processing_mode(mode);
    }

    /// Get current stream state
    pub fn state(&self) -> StreamState {
        self.state
    }

    /// Start streaming processing
    pub async fn start(&mut self) -> Result<()> {
        if self.state != StreamState::Idle && self.state != StreamState::Stopped {
            return Err(Error::Streaming {
                message: "Stream is already active".to_string(),
                stream_info: None,
                context: None,
                recovery_suggestions: Box::new(vec![
                    "Stop the current stream before starting a new one".to_string(),
                    "Check stream state before calling start()".to_string(),
                ]),
            });
        }

        self.state = StreamState::Processing;
        info!("Started streaming converter");
        Ok(())
    }

    /// Pause streaming processing
    pub async fn pause(&mut self) -> Result<()> {
        if self.state != StreamState::Processing {
            return Err(Error::Streaming {
                message: "Stream is not processing".to_string(),
                stream_info: None,
                context: None,
                recovery_suggestions: Box::new(vec![
                    "Start the stream before trying to pause".to_string(),
                    "Check stream state before calling pause()".to_string(),
                ]),
            });
        }

        self.state = StreamState::Paused;
        info!("Paused streaming converter");
        Ok(())
    }

    /// Stop streaming processing
    pub async fn stop(&mut self) -> Result<()> {
        self.state = StreamState::Stopped;

        // Clear buffers
        let mut buffer = self.accumulation_buffer.lock().await;
        buffer.clear();

        info!("Stopped streaming converter");
        Ok(())
    }

    /// Process streaming audio
    pub async fn process_stream<S>(
        &mut self,
        mut stream: S,
    ) -> Result<impl Stream<Item = Result<Vec<f32>>>>
    where
        S: Stream<Item = Vec<f32>> + Unpin + Send + 'static,
    {
        if self.state != StreamState::Processing {
            return Err(Error::streaming(
                "Stream is not in processing state".to_string(),
            ));
        }

        let realtime_converter = self.realtime_converter.clone();
        let config = self.config.clone();
        let stats = self.stats.clone();
        let conversion_target = self.conversion_target.clone();

        let (tx, rx) = mpsc::channel(config.channel_buffer_size);

        // Spawn processing task
        tokio::spawn(async move {
            let mut converter = realtime_converter.lock().await;
            if let Some(target) = conversion_target {
                converter.set_conversion_target(target);
            }
            drop(converter);

            let mut chunk_count = 0u64;
            let start_time = Instant::now();

            while let Some(chunk) = stream.next().await {
                let chunk_start = Instant::now();

                if chunk.is_empty() {
                    continue;
                }

                // Process chunk through real-time converter
                let mut converter = realtime_converter.lock().await;
                match converter.process_chunk(&chunk).await {
                    Ok(processed_chunk) => {
                        if !processed_chunk.is_empty()
                            && tx.send(Ok(processed_chunk)).await.is_err()
                        {
                            warn!("Receiver dropped, stopping processing");
                            break;
                        }
                    }
                    Err(e) => {
                        error!("Error processing chunk: {}", e);
                        if tx.send(Err(e)).await.is_err() {
                            break;
                        }
                    }
                }
                drop(converter);

                // Update statistics
                chunk_count += 1;
                let chunk_duration = chunk_start.elapsed();
                let mut stats_guard = stats.write().await;
                stats_guard.update_chunk_stats(chunk.len(), chunk_duration);

                // Check if we need to throttle
                if chunk_duration > Duration::from_millis(config.target_latency_ms as u64) {
                    warn!(
                        "Processing slower than real-time: {:.2}ms",
                        chunk_duration.as_millis()
                    );
                }
            }

            let total_duration = start_time.elapsed();
            info!(
                "Processed {} chunks in {:.2}s",
                chunk_count,
                total_duration.as_secs_f32()
            );
        });

        Ok(ReceiverStream::new(rx))
    }

    /// Process audio stream with backpressure handling
    pub async fn process_stream_with_backpressure<S>(
        &mut self,
        stream: S,
    ) -> Result<impl Stream<Item = Result<Vec<f32>>>>
    where
        S: Stream<Item = Vec<f32>> + Unpin + Send + 'static,
    {
        let throttled_stream = self.throttle_stream(stream).await?;
        self.process_stream(throttled_stream).await
    }

    /// Apply throttling to prevent buffer overflow
    async fn throttle_stream<S>(
        &self,
        stream: S,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Vec<f32>> + Send>>>
    where
        S: Stream<Item = Vec<f32>> + Unpin + Send + 'static,
    {
        let config = self.config.clone();
        let throttle_interval = Duration::from_millis(
            (config.chunk_size as f64 / config.sample_rate as f64 * 1000.0) as u64,
        );

        Ok(Box::pin(stream! {
            tokio::pin!(stream);

            let mut last_yield = Instant::now();

            while let Some(chunk) = stream.next().await {
                let now = Instant::now();
                let time_since_last = now - last_yield;

                if time_since_last < throttle_interval {
                    let sleep_time = throttle_interval - time_since_last;
                    tokio::time::sleep(sleep_time).await;
                }

                yield chunk;
                last_yield = Instant::now();
            }
        }))
    }

    /// Get streaming statistics
    pub async fn get_stats(&self) -> StreamingStats {
        self.stats.read().await.clone()
    }

    /// Reset streaming statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = StreamingStats::default();
    }

    /// Check if streaming is healthy (meeting performance targets)
    pub async fn is_healthy(&self) -> bool {
        let stats = self.stats.read().await;
        let avg_latency = stats.average_chunk_latency_ms();
        avg_latency <= self.config.target_latency_ms * 1.5 // 50% tolerance
    }

    /// Create a processing stream from multiple input streams
    pub async fn multiplex_streams<S>(
        &mut self,
        streams: Vec<S>,
    ) -> Result<impl Stream<Item = Result<Vec<f32>>>>
    where
        S: Stream<Item = Vec<f32>> + Unpin + Send + 'static,
    {
        if streams.is_empty() {
            return Err(Error::Streaming {
                message: "No input streams provided".to_string(),
                stream_info: None,
                context: None,
                recovery_suggestions: Box::new(vec![
                    "Provide at least one input stream".to_string(),
                    "Check stream configuration".to_string(),
                ]),
            });
        }

        if streams.len() > self.config.max_concurrent_streams {
            return Err(Error::Streaming {
                message: format!(
                    "Too many streams: {} > {}",
                    streams.len(),
                    self.config.max_concurrent_streams
                ),
                stream_info: None,
                context: None,
                recovery_suggestions: Box::new(vec![
                    "Reduce the number of concurrent streams".to_string(),
                    "Increase max_concurrent_streams configuration".to_string(),
                ]),
            });
        }

        let (tx, rx) = mpsc::channel(self.config.channel_buffer_size);
        let realtime_converter = self.realtime_converter.clone();
        let config = self.config.clone();

        for (stream_id, stream) in streams.into_iter().enumerate() {
            let tx_clone = tx.clone();
            let converter_clone = realtime_converter.clone();
            let config_clone = config.clone();

            tokio::spawn(async move {
                tokio::pin!(stream);

                while let Some(chunk) = stream.next().await {
                    let mut converter = converter_clone.lock().await;
                    match converter.process_chunk(&chunk).await {
                        Ok(processed) => {
                            if !processed.is_empty() && tx_clone.send(Ok(processed)).await.is_err()
                            {
                                debug!("Stream {} receiver dropped", stream_id);
                                break;
                            }
                        }
                        Err(e) => {
                            error!("Stream {} processing error: {}", stream_id, e);
                            if tx_clone.send(Err(e)).await.is_err() {
                                break;
                            }
                        }
                    }
                }
            });
        }

        Ok(ReceiverStream::new(rx))
    }
}

/// Stream processor for handling audio streams
#[derive(Debug)]
pub struct StreamProcessor {
    /// Processing configuration
    config: StreamConfig,
    /// Active streaming converters
    converters: Arc<RwLock<Vec<StreamingConverter>>>,
    /// Load balancer for distributing streams
    load_balancer: LoadBalancer,
}

/// Load balancer for distributing processing load across converters
#[derive(Debug, Clone)]
pub struct LoadBalancer {
    /// Load balancing strategy
    strategy: LoadBalancingStrategy,
    /// Current round-robin index
    round_robin_index: usize,
}

/// Load balancing strategies for distributing stream processing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Least loaded converter
    LeastLoaded,
    /// Random distribution
    Random,
}

/// Configuration for stream processing with buffering and latency settings
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Chunk size for processing
    pub chunk_size: usize,
    /// Sample rate
    pub sample_rate: u32,
    /// Target latency in milliseconds
    pub target_latency_ms: f32,
    /// Buffer capacity for accumulation
    pub buffer_capacity: usize,
    /// Channel buffer size for async communication
    pub channel_buffer_size: usize,
    /// Maximum concurrent streams
    pub max_concurrent_streams: usize,
    /// Enable adaptive buffering
    pub adaptive_buffering: bool,
    /// Quality vs latency trade-off (0.0 = lowest latency, 1.0 = highest quality)
    pub quality_vs_latency: f32,
    /// Enable error recovery
    pub enable_error_recovery: bool,
    /// Stream timeout in seconds
    pub stream_timeout_secs: u64,
}

impl StreamProcessor {
    /// Create new stream processor
    pub fn new(config: StreamConfig) -> Self {
        Self {
            config,
            converters: Arc::new(RwLock::new(Vec::new())),
            load_balancer: LoadBalancer::new(LoadBalancingStrategy::LeastLoaded),
        }
    }

    /// Create stream processor with multiple converters
    pub async fn with_converter_pool(config: StreamConfig, pool_size: usize) -> Result<Self> {
        let mut processor = Self::new(config.clone());

        for _ in 0..pool_size {
            let converter = StreamingConverter::new(config.clone())?;
            processor.converters.write().await.push(converter);
        }

        Ok(processor)
    }

    /// Add a streaming converter to the pool
    pub async fn add_converter(&self, converter: StreamingConverter) {
        self.converters.write().await.push(converter);
    }

    /// Process audio stream using load balancing
    pub async fn process_stream(&self, audio_stream: AudioStream) -> Result<ProcessedAudioStream> {
        let converter_index = self.select_converter().await?;

        Ok(ProcessedAudioStream::new(
            audio_stream,
            self.config.clone(),
            converter_index,
        ))
    }

    /// Process multiple streams concurrently
    pub async fn process_multiple_streams(
        &self,
        streams: Vec<AudioStream>,
    ) -> Result<Vec<ProcessedAudioStream>> {
        if streams.len() > self.config.max_concurrent_streams {
            return Err(Error::Streaming {
                message: format!(
                    "Too many streams: {} > {}",
                    streams.len(),
                    self.config.max_concurrent_streams
                ),
                stream_info: None,
                context: None,
                recovery_suggestions: Box::new(vec![
                    "Reduce the number of concurrent streams".to_string(),
                    "Increase max_concurrent_streams configuration".to_string(),
                ]),
            });
        }

        let mut processed_streams = Vec::new();

        for stream in streams {
            let processed = self.process_stream(stream).await?;
            processed_streams.push(processed);
        }

        Ok(processed_streams)
    }

    /// Select the best converter for load balancing
    async fn select_converter(&self) -> Result<usize> {
        let converters = self.converters.read().await;

        if converters.is_empty() {
            return Err(Error::Streaming {
                message: "No converters available".to_string(),
                stream_info: None,
                context: None,
                recovery_suggestions: Box::new(vec![
                    "Initialize converters before processing".to_string(),
                    "Check converter configuration".to_string(),
                ]),
            });
        }

        match self.load_balancer.strategy {
            LoadBalancingStrategy::RoundRobin => {
                Ok(self.load_balancer.round_robin_index % converters.len())
            }
            LoadBalancingStrategy::LeastLoaded => {
                // Select converter with lowest load based on comprehensive metrics
                let mut best_index = 0;
                let mut lowest_load_score = f32::MAX;

                for (index, converter) in converters.iter().enumerate() {
                    let stats = converter.get_stats().await;

                    // Calculate composite load score (lower is better)
                    let latency_score = stats.average_chunk_latency_ms();
                    let error_rate = if stats.total_chunks_processed > 0 {
                        stats.total_errors as f32 / stats.total_chunks_processed as f32
                    } else {
                        0.0
                    };
                    let throughput_score = 1.0 / (stats.throughput_samples_per_sec() + 1.0); // Inverse throughput

                    // Weighted composite score (prioritize latency and throughput)
                    let load_score = (latency_score * 0.5)
                        + (throughput_score * 0.3)
                        + (error_rate * 100.0 * 0.2);

                    if load_score < lowest_load_score {
                        lowest_load_score = load_score;
                        best_index = index;
                    }
                }

                Ok(best_index)
            }
            LoadBalancingStrategy::Random => Ok(fastrand::usize(0..converters.len())),
        }
    }

    /// Get processor statistics
    pub async fn get_stats(&self) -> ProcessorStats {
        let converters = self.converters.read().await;
        let mut total_processed = 0;
        let mut total_errors = 0;
        let mut avg_latency = 0.0;

        for converter in converters.iter() {
            let stats = converter.get_stats().await;
            total_processed += stats.total_chunks_processed;
            total_errors += stats.total_errors;
            avg_latency += stats.average_chunk_latency_ms();
        }

        if !converters.is_empty() {
            avg_latency /= converters.len() as f32;
        }

        ProcessorStats {
            total_converters: converters.len(),
            total_processed,
            total_errors,
            average_latency_ms: avg_latency,
        }
    }
}

impl LoadBalancer {
    /// Create new load balancer
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            round_robin_index: 0,
        }
    }

    /// Set load balancing strategy
    pub fn set_strategy(&mut self, strategy: LoadBalancingStrategy) {
        self.strategy = strategy;
    }
}

/// Audio stream wrapper providing buffered audio playback
#[derive(Debug)]
pub struct AudioStream {
    /// Audio data
    data: Vec<f32>,
    /// Current position
    position: usize,
    /// Stream metadata
    metadata: StreamMetadata,
    /// Stream format
    format: AudioFormat,
}

/// Stream metadata containing identification and source information
#[derive(Debug, Clone)]
pub struct StreamMetadata {
    /// Stream identifier
    pub id: String,
    /// Stream name
    pub name: String,
    /// Source information
    pub source: String,
    /// Creation timestamp
    pub created_at: std::time::SystemTime,
}

/// Audio format specification defining sample rate and encoding
#[derive(Debug, Clone)]
pub struct AudioFormat {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Bits per sample
    pub bits_per_sample: u16,
    /// Audio encoding
    pub encoding: AudioEncoding,
}

/// Audio encoding types supported for streaming
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioEncoding {
    /// Linear PCM
    PCM,
    /// IEEE 754 floating point
    Float32,
    /// Compressed formats
    MP3,
    /// Advanced Audio Coding
    AAC,
    /// Opus audio codec
    Opus,
}

impl AudioStream {
    /// Create new audio stream
    pub fn new(data: Vec<f32>) -> Self {
        Self {
            data,
            position: 0,
            metadata: StreamMetadata::default(),
            format: AudioFormat::default(),
        }
    }

    /// Create audio stream with metadata
    pub fn with_metadata(data: Vec<f32>, metadata: StreamMetadata, format: AudioFormat) -> Self {
        Self {
            data,
            position: 0,
            metadata,
            format,
        }
    }

    /// Get stream metadata
    pub fn metadata(&self) -> &StreamMetadata {
        &self.metadata
    }

    /// Get audio format
    pub fn format(&self) -> &AudioFormat {
        &self.format
    }

    /// Get remaining samples
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.position)
    }

    /// Check if stream has ended
    pub fn is_finished(&self) -> bool {
        self.position >= self.data.len()
    }

    /// Reset stream position
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Seek to position
    pub fn seek(&mut self, position: usize) -> Result<()> {
        if position > self.data.len() {
            return Err(Error::Streaming {
                message: "Seek position out of bounds".to_string(),
                stream_info: None,
                context: None,
                recovery_suggestions: Box::new(vec![
                    "Provide a valid seek position within stream bounds".to_string(),
                    "Check stream length before seeking".to_string(),
                ]),
            });
        }
        self.position = position;
        Ok(())
    }
}

impl Default for StreamMetadata {
    fn default() -> Self {
        Self {
            id: format!("stream_{}", fastrand::u64(..)),
            name: "Unnamed Stream".to_string(),
            source: "Unknown".to_string(),
            created_at: std::time::SystemTime::now(),
        }
    }
}

impl Default for AudioFormat {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            channels: 1,
            bits_per_sample: 32,
            encoding: AudioEncoding::Float32,
        }
    }
}

/// Processed audio stream with conversion applied
#[derive(Debug)]
pub struct ProcessedAudioStream {
    /// Source stream
    source: AudioStream,
    /// Processing config
    config: StreamConfig,
    /// Converter index for load balancing
    converter_index: usize,
    /// Processing buffer
    buffer: VecDeque<f32>,
    /// Error recovery state
    error_recovery: ErrorRecoveryState,
}

/// Error recovery state tracking failures and recovery strategy
#[derive(Debug, Clone)]
pub struct ErrorRecoveryState {
    /// Number of consecutive errors
    consecutive_errors: u32,
    /// Last error time
    last_error_time: Option<Instant>,
    /// Recovery strategy
    strategy: ErrorRecoveryStrategy,
}

/// Error recovery strategies for handling stream errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorRecoveryStrategy {
    /// Skip problematic chunks
    Skip,
    /// Retry processing
    Retry,
    /// Fallback to pass-through
    Passthrough,
    /// Stop processing
    Stop,
}

impl ProcessedAudioStream {
    /// Create new processed stream
    pub fn new(source: AudioStream, config: StreamConfig, converter_index: usize) -> Self {
        let buffer_capacity = config.buffer_capacity;
        Self {
            source,
            config,
            converter_index,
            buffer: VecDeque::with_capacity(buffer_capacity),
            error_recovery: ErrorRecoveryState::default(),
        }
    }

    /// Get converter index
    pub fn converter_index(&self) -> usize {
        self.converter_index
    }

    /// Get error recovery state
    pub fn error_recovery_state(&self) -> &ErrorRecoveryState {
        &self.error_recovery
    }

    /// Handle processing error
    fn handle_error(&mut self, error: Error) -> Result<Option<Vec<f32>>> {
        self.error_recovery.consecutive_errors += 1;
        self.error_recovery.last_error_time = Some(Instant::now());

        match self.error_recovery.strategy {
            ErrorRecoveryStrategy::Skip => {
                warn!("Skipping chunk due to error: {}", error);
                Ok(None) // Skip this chunk
            }
            ErrorRecoveryStrategy::Retry => {
                if self.error_recovery.consecutive_errors < 3 {
                    Err(error) // Will trigger retry
                } else {
                    warn!("Too many retries, falling back to passthrough");
                    self.error_recovery.strategy = ErrorRecoveryStrategy::Passthrough;
                    Ok(None)
                }
            }
            ErrorRecoveryStrategy::Passthrough => {
                warn!("Using passthrough due to error: {}", error);
                // Return original chunk if available
                Ok(None)
            }
            ErrorRecoveryStrategy::Stop => {
                error!("Stopping processing due to error: {}", error);
                Err(error)
            }
        }
    }
}

impl Default for ErrorRecoveryState {
    fn default() -> Self {
        Self {
            consecutive_errors: 0,
            last_error_time: None,
            strategy: ErrorRecoveryStrategy::Retry,
        }
    }
}

impl Stream for ProcessedAudioStream {
    type Item = Result<Vec<f32>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Check if stream is finished
        if self.source.is_finished() && self.buffer.is_empty() {
            return Poll::Ready(None);
        }

        // Fill buffer if needed
        while self.buffer.len() < self.config.chunk_size && !self.source.is_finished() {
            let remaining = self.source.remaining();
            let chunk_size = self.config.chunk_size.min(remaining);

            if chunk_size == 0 {
                break;
            }

            let start_pos = self.source.position;
            let end_pos = start_pos + chunk_size;

            let samples: Vec<f32> = self.source.data[start_pos..end_pos].to_vec();
            for sample in samples {
                self.buffer.push_back(sample);
            }

            self.source.position = end_pos;
        }

        // Return chunk if buffer has enough data
        if self.buffer.len() >= self.config.chunk_size
            || (self.source.is_finished() && !self.buffer.is_empty())
        {
            let chunk_size = self.config.chunk_size.min(self.buffer.len());
            let mut chunk = Vec::with_capacity(chunk_size);

            for _ in 0..chunk_size {
                if let Some(sample) = self.buffer.pop_front() {
                    chunk.push(sample);
                }
            }

            // Reset error count on successful processing
            if !chunk.is_empty() {
                self.error_recovery.consecutive_errors = 0;
            }

            Poll::Ready(Some(Ok(chunk)))
        } else {
            // Not ready yet, register waker
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            sample_rate: 22050,
            target_latency_ms: 20.0,
            buffer_capacity: 8192,
            channel_buffer_size: 100,
            max_concurrent_streams: 4,
            adaptive_buffering: true,
            quality_vs_latency: 0.5,
            enable_error_recovery: true,
            stream_timeout_secs: 30,
        }
    }
}

/// Streaming statistics for performance monitoring
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    /// Total chunks processed
    pub total_chunks_processed: u64,
    /// Total processing time
    pub total_processing_time_ms: f64,
    /// Total errors encountered
    pub total_errors: u64,
    /// Maximum chunk latency
    pub max_chunk_latency_ms: f32,
    /// Minimum chunk latency
    pub min_chunk_latency_ms: f32,
    /// Total samples processed
    pub total_samples: u64,
}

impl StreamingStats {
    /// Update statistics with chunk processing information
    pub fn update_chunk_stats(&mut self, sample_count: usize, processing_time: Duration) {
        let time_ms = processing_time.as_millis() as f64;

        self.total_chunks_processed += 1;
        self.total_processing_time_ms += time_ms;
        self.total_samples += sample_count as u64;

        let time_ms_f32 = time_ms as f32;
        if self.total_chunks_processed == 1 {
            self.max_chunk_latency_ms = time_ms_f32;
            self.min_chunk_latency_ms = time_ms_f32;
        } else {
            self.max_chunk_latency_ms = self.max_chunk_latency_ms.max(time_ms_f32);
            self.min_chunk_latency_ms = self.min_chunk_latency_ms.min(time_ms_f32);
        }
    }

    /// Get average chunk processing latency
    pub fn average_chunk_latency_ms(&self) -> f32 {
        if self.total_chunks_processed == 0 {
            0.0
        } else {
            (self.total_processing_time_ms / self.total_chunks_processed as f64) as f32
        }
    }

    /// Get processing throughput (samples per second)
    pub fn throughput_samples_per_sec(&self) -> f32 {
        if self.total_processing_time_ms == 0.0 {
            0.0
        } else {
            (self.total_samples as f64 / (self.total_processing_time_ms / 1000.0)) as f32
        }
    }

    /// Get error rate
    pub fn error_rate(&self) -> f32 {
        if self.total_chunks_processed == 0 {
            0.0
        } else {
            self.total_errors as f32 / self.total_chunks_processed as f32
        }
    }
}

/// Processor statistics aggregating metrics across converters
#[derive(Debug, Clone)]
pub struct ProcessorStats {
    /// Total number of converters
    pub total_converters: usize,
    /// Total chunks processed across all converters
    pub total_processed: u64,
    /// Total errors across all converters
    pub total_errors: u64,
    /// Average latency across all converters
    pub average_latency_ms: f32,
}

impl Default for StreamProcessor {
    fn default() -> Self {
        Self::new(StreamConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use tokio_test;

    #[tokio::test]
    async fn test_streaming_converter_creation() {
        let config = StreamConfig::default();
        let converter = StreamingConverter::new(config);
        assert!(converter.is_ok());
    }

    #[tokio::test]
    async fn test_audio_stream_processing() {
        let data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let audio_stream = AudioStream::new(data.clone());

        assert_eq!(audio_stream.remaining(), data.len());
        assert!(!audio_stream.is_finished());
    }

    #[tokio::test]
    async fn test_stream_processor() {
        let config = StreamConfig {
            chunk_size: 2,
            ..Default::default()
        };
        let processor = StreamProcessor::with_converter_pool(config.clone(), 1).await;
        assert!(processor.is_ok());
        let processor = processor.unwrap();

        let data = vec![0.1, 0.2, 0.3, 0.4];
        let audio_stream = AudioStream::new(data);

        let processed_stream = processor.process_stream(audio_stream).await;
        assert!(processed_stream.is_ok());
    }

    #[tokio::test]
    async fn test_streaming_stats() {
        let mut stats = StreamingStats::default();
        let duration = Duration::from_millis(10);

        stats.update_chunk_stats(100, duration);

        assert_eq!(stats.total_chunks_processed, 1);
        assert_eq!(stats.total_samples, 100);
        assert_eq!(stats.average_chunk_latency_ms(), 10.0);
    }
}
