//! Streaming pipeline for real-time audio processing
//!
//! Provides a complete streaming pipeline that coordinates chunk processing,
//! buffer management, and latency optimization for real-time audio synthesis.

use super::{
    buffer::{BufferManager, StreamingBuffer},
    latency::{LatencyOptimizer, PredictiveProcessor},
    StreamCommand, StreamHandle, StreamingStats, StreamingVocoder,
};
use crate::config::StreamingConfig;
use crate::{AudioBuffer, MelSpectrogram, Result, Vocoder, VocoderError};
use async_trait::async_trait;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tokio::sync::{mpsc, Mutex as AsyncMutex};
use tokio::time::{sleep, Duration};

/// Main streaming pipeline implementation
pub struct StreamingPipeline {
    /// Underlying vocoder
    vocoder: Arc<dyn Vocoder>,

    /// Configuration
    config: Arc<RwLock<StreamingConfig>>,

    /// Buffer manager
    buffer_manager: Arc<BufferManager>,

    /// Latency optimizer
    latency_optimizer: Arc<LatencyOptimizer>,

    /// Predictive processor
    predictive_processor: Arc<PredictiveProcessor>,

    /// Active streams
    active_streams: Arc<AsyncMutex<Vec<StreamContext>>>,

    /// Global statistics
    stats: Arc<RwLock<StreamingStats>>,

    /// Next stream ID
    next_stream_id: Arc<AsyncMutex<u64>>,

    /// Pipeline state
    is_running: Arc<RwLock<bool>>,
}

/// Context for an active stream
struct StreamContext {
    /// Stream ID
    id: u64,

    /// Input buffer
    #[allow(dead_code)]
    input_buffer: Arc<dyn StreamingBuffer>,

    /// Output buffer
    #[allow(dead_code)]
    output_buffer: Arc<dyn StreamingBuffer>,

    /// Stream statistics
    #[allow(dead_code)]
    stats: StreamingStats,

    /// Last activity time
    last_activity: Instant,

    /// Processing task handle
    task_handle: Option<tokio::task::JoinHandle<()>>,
}

impl StreamingPipeline {
    /// Create new streaming pipeline
    pub fn new(vocoder: Arc<dyn Vocoder>) -> Self {
        let config = StreamingConfig::default();
        let buffer_manager = Arc::new(BufferManager::new(config.clone()));
        let latency_optimizer = Arc::new(LatencyOptimizer::new(config.clone()));
        let predictive_processor = Arc::new(PredictiveProcessor::new(config.clone()));

        Self {
            vocoder,
            config: Arc::new(RwLock::new(config)),
            buffer_manager,
            latency_optimizer,
            predictive_processor,
            active_streams: Arc::new(AsyncMutex::new(Vec::new())),
            stats: Arc::new(RwLock::new(StreamingStats::default())),
            next_stream_id: Arc::new(AsyncMutex::new(1)),
            is_running: Arc::new(RwLock::new(false)),
        }
    }

    /// Update configuration
    pub async fn update_config(&self, new_config: StreamingConfig) -> Result<()> {
        // Validate configuration
        let validation = new_config.validate();
        if !validation.is_valid {
            return Err(VocoderError::ConfigurationError(format!(
                "Invalid streaming configuration: {:?}",
                validation.errors
            )));
        }

        // Update configuration
        if let Ok(mut config) = self.config.write() {
            *config = new_config;
        }

        tracing::info!("Updated streaming configuration");
        Ok(())
    }

    /// Process a single chunk of mel spectrogram
    async fn process_chunk_internal(
        &self,
        mel_chunk: MelSpectrogram,
        stream_id: u64,
    ) -> Result<AudioBuffer> {
        let start_time = Instant::now();

        // Add to predictive processor for future optimization
        self.predictive_processor.add_lookahead(mel_chunk.clone());

        // Get optimal chunk size from latency optimizer
        let optimal_size = self.latency_optimizer.get_optimal_chunk_size();

        // Adjust chunk if needed (simplified - in practice would involve resampling)
        let processed_mel = if mel_chunk.n_frames != optimal_size {
            // For now, just use the original chunk
            mel_chunk
        } else {
            mel_chunk
        };

        // Perform vocoding
        let audio_result = self.vocoder.vocode(&processed_mel, None).await;

        let processing_time = start_time.elapsed().as_secs_f32() * 1000.0;

        match audio_result {
            Ok(audio) => {
                // Record performance metrics
                self.latency_optimizer
                    .record_processing_time(processing_time);

                let audio_duration_ms = audio.duration() * 1000.0;
                let latency_ms = processing_time; // Simplified latency calculation

                self.latency_optimizer.record_latency(latency_ms);

                // Update global statistics
                if let Ok(mut stats) = self.stats.write() {
                    stats.chunks_processed += 1;
                    stats.avg_processing_time_ms =
                        (stats.avg_processing_time_ms + processing_time) / 2.0;
                    stats.current_latency_ms = latency_ms;
                    if latency_ms > stats.peak_latency_ms {
                        stats.peak_latency_ms = latency_ms;
                    }
                    stats.update_rtf(audio_duration_ms, processing_time);
                }

                tracing::debug!(
                    "Processed chunk for stream {} in {:.2}ms (RTF: {:.3})",
                    stream_id,
                    processing_time,
                    processing_time / audio_duration_ms
                );

                Ok(audio)
            }
            Err(e) => {
                if let Ok(mut stats) = self.stats.write() {
                    stats.error_count += 1;
                }

                tracing::error!("Chunk processing failed for stream {}: {}", stream_id, e);
                Err(e)
            }
        }
    }

    /// Background task for processing a stream
    async fn stream_processing_task(
        &self,
        stream_id: u64,
        mut input_rx: mpsc::Receiver<MelSpectrogram>,
        output_tx: mpsc::Sender<AudioBuffer>,
        mut control_rx: mpsc::Receiver<StreamCommand>,
    ) {
        let mut is_paused = false;

        loop {
            tokio::select! {
                // Process input mel spectrograms
                mel_opt = input_rx.recv(), if !is_paused => {
                    match mel_opt {
                        Some(mel) => {
                            match self.process_chunk_internal(mel, stream_id).await {
                                Ok(audio) => {
                                    if output_tx.send(audio).await.is_err() {
                                        tracing::warn!("Output channel closed for stream {}", stream_id);
                                        break;
                                    }
                                }
                                Err(e) => {
                                    tracing::error!("Processing error in stream {}: {}", stream_id, e);
                                    // Continue processing despite errors
                                }
                            }
                        }
                        None => {
                            tracing::info!("Input channel closed for stream {}", stream_id);
                            break;
                        }
                    }
                }

                // Handle control commands
                cmd_opt = control_rx.recv() => {
                    match cmd_opt {
                        Some(cmd) => {
                            match cmd {
                                StreamCommand::Pause => {
                                    is_paused = true;
                                    tracing::debug!("Stream {} paused", stream_id);
                                }
                                StreamCommand::Resume => {
                                    is_paused = false;
                                    tracing::debug!("Stream {} resumed", stream_id);
                                }
                                StreamCommand::Flush => {
                                    // Flush any internal buffers
                                    tracing::debug!("Stream {} flushed", stream_id);
                                }
                                StreamCommand::UpdateConfig(config) => {
                                    if let Err(e) = self.update_config(config).await {
                                        tracing::error!("Failed to update config for stream {}: {}", stream_id, e);
                                    }
                                }
                                StreamCommand::Shutdown => {
                                    tracing::info!("Shutting down stream {}", stream_id);
                                    break;
                                }
                                StreamCommand::GetStats => {
                                    // Stats are available through the main interface
                                }
                            }
                        }
                        None => {
                            tracing::info!("Control channel closed for stream {}", stream_id);
                            break;
                        }
                    }
                }

                // Periodic maintenance
                _ = sleep(Duration::from_millis(100)) => {
                    // Perform background maintenance tasks
                    self.buffer_manager.cleanup();
                }
            }
        }

        tracing::info!("Stream {} processing task ended", stream_id);
    }

    /// Cleanup inactive streams
    async fn cleanup_inactive_streams(&self) {
        let mut streams = self.active_streams.lock().await;
        let now = Instant::now();
        let timeout = Duration::from_secs(300); // 5 minutes timeout

        streams.retain(|stream| {
            if now.duration_since(stream.last_activity) > timeout {
                tracing::info!("Cleaning up inactive stream {}", stream.id);
                false
            } else {
                true
            }
        });

        // Update active stream count
        if let Ok(mut stats) = self.stats.write() {
            stats.active_streams = streams.len() as u32;
        }
    }

    /// Background maintenance task
    async fn maintenance_task(&self) {
        let mut cleanup_interval = tokio::time::interval(Duration::from_secs(60));

        while *self.is_running.read().unwrap() {
            cleanup_interval.tick().await;

            // Cleanup inactive streams
            self.cleanup_inactive_streams().await;

            // Update aggregated statistics
            let buffer_stats = self.buffer_manager.get_aggregated_stats();
            let latency_stats = self.latency_optimizer.get_stats();

            if let Ok(mut stats) = self.stats.write() {
                stats.buffer_underruns += buffer_stats.buffer_underruns;
                stats.buffer_overruns += buffer_stats.buffer_overruns;
                stats.memory_usage_mb = buffer_stats.memory_usage_mb;
                stats.current_latency_ms = latency_stats.avg_latency;
                stats.peak_latency_ms = latency_stats.peak_latency.max(stats.peak_latency_ms);
            }
        }
    }

    /// Start the pipeline
    pub async fn start(&self) -> Result<()> {
        if *self.is_running.read().unwrap() {
            return Ok(());
        }

        *self.is_running.write().unwrap() = true;

        // Start maintenance task
        let pipeline = self.clone();
        tokio::spawn(async move {
            pipeline.maintenance_task().await;
        });

        tracing::info!("Streaming pipeline started");
        Ok(())
    }

    /// Stop the pipeline
    pub async fn stop(&self) -> Result<()> {
        *self.is_running.write().unwrap() = false;

        // Stop all active streams
        let mut streams = self.active_streams.lock().await;
        for stream in streams.drain(..) {
            if let Some(handle) = stream.task_handle {
                handle.abort();
            }
        }

        tracing::info!("Streaming pipeline stopped");
        Ok(())
    }
}

// Manual Clone implementation to handle the complex types
impl Clone for StreamingPipeline {
    fn clone(&self) -> Self {
        Self {
            vocoder: self.vocoder.clone(),
            config: self.config.clone(),
            buffer_manager: self.buffer_manager.clone(),
            latency_optimizer: self.latency_optimizer.clone(),
            predictive_processor: self.predictive_processor.clone(),
            active_streams: self.active_streams.clone(),
            stats: self.stats.clone(),
            next_stream_id: self.next_stream_id.clone(),
            is_running: self.is_running.clone(),
        }
    }
}

#[async_trait]
impl StreamingVocoder for StreamingPipeline {
    async fn initialize(&mut self, config: StreamingConfig) -> Result<()> {
        self.update_config(config).await?;
        self.start().await?;
        Ok(())
    }

    async fn start_stream(&self) -> Result<StreamHandle> {
        let stream_id = {
            let mut next_id = self.next_stream_id.lock().await;
            let id = *next_id;
            *next_id += 1;
            id
        };

        // Create channels
        let (input_tx, input_rx) = mpsc::channel::<MelSpectrogram>(32);
        let (output_tx, output_rx) = mpsc::channel::<AudioBuffer>(32);
        let (control_tx, control_rx) = mpsc::channel::<StreamCommand>(16);

        // Create buffers
        let input_buffer = self.buffer_manager.create_buffer();
        let output_buffer = self.buffer_manager.create_buffer();

        // Start processing task
        let pipeline = self.clone();
        let task_handle = tokio::spawn(async move {
            pipeline
                .stream_processing_task(stream_id, input_rx, output_tx, control_rx)
                .await;
        });

        // Create stream context
        let stream_context = StreamContext {
            id: stream_id,
            input_buffer,
            output_buffer,
            stats: StreamingStats::default(),
            last_activity: Instant::now(),
            task_handle: Some(task_handle),
        };

        // Add to active streams
        self.active_streams.lock().await.push(stream_context);

        // Update active stream count
        if let Ok(mut stats) = self.stats.write() {
            stats.active_streams += 1;
        }

        tracing::info!("Started stream {}", stream_id);

        Ok(StreamHandle {
            input_tx,
            output_rx,
            control_tx,
            stream_id,
        })
    }

    async fn process_chunk(&self, mel_chunk: MelSpectrogram) -> Result<AudioBuffer> {
        self.process_chunk_internal(mel_chunk, 0).await
    }

    async fn process_batch(&self, mel_chunks: Vec<MelSpectrogram>) -> Result<Vec<AudioBuffer>> {
        let mut results = Vec::new();

        // Process chunks in parallel for better throughput
        let tasks: Vec<_> = mel_chunks
            .into_iter()
            .enumerate()
            .map(|(i, mel)| {
                let pipeline = self.clone();
                tokio::spawn(async move { pipeline.process_chunk_internal(mel, i as u64).await })
            })
            .collect();

        // Collect results
        for task in tasks {
            match task.await {
                Ok(Ok(audio)) => results.push(audio),
                Ok(Err(e)) => return Err(e),
                Err(e) => {
                    return Err(VocoderError::StreamingError(format!(
                        "Task join error: {e}"
                    )))
                }
            }
        }

        Ok(results)
    }

    async fn stop_stream(&self) -> Result<()> {
        self.stop().await
    }

    fn get_stats(&self) -> StreamingStats {
        self.stats.read().unwrap().clone()
    }
}

/// Chunk processor for handling individual mel spectrogram chunks
pub struct ChunkProcessor {
    /// Vocoder for processing
    vocoder: Arc<dyn Vocoder>,

    /// Processing configuration
    #[allow(dead_code)]
    config: StreamingConfig,

    /// Chunk statistics
    stats: Arc<RwLock<ChunkStats>>,
}

/// Statistics for chunk processing
#[derive(Debug, Clone, Default)]
pub struct ChunkStats {
    /// Chunks processed
    pub chunks_processed: u64,

    /// Average processing time (ms)
    pub avg_processing_time: f32,

    /// Processing errors
    pub processing_errors: u64,

    /// Total processing time (ms)
    pub total_processing_time: f32,
}

impl ChunkProcessor {
    /// Create new chunk processor
    pub fn new(vocoder: Arc<dyn Vocoder>, config: StreamingConfig) -> Self {
        Self {
            vocoder,
            config,
            stats: Arc::new(RwLock::new(ChunkStats::default())),
        }
    }

    /// Process a single chunk
    pub async fn process(&self, mel_chunk: MelSpectrogram) -> Result<AudioBuffer> {
        let start_time = Instant::now();

        let result = self.vocoder.vocode(&mel_chunk, None).await;

        let processing_time = start_time.elapsed().as_secs_f32() * 1000.0;

        // Update statistics
        if let Ok(mut stats) = self.stats.write() {
            stats.chunks_processed += 1;
            stats.total_processing_time += processing_time;
            stats.avg_processing_time = stats.total_processing_time / stats.chunks_processed as f32;

            if result.is_err() {
                stats.processing_errors += 1;
            }
        }

        result
    }

    /// Get processing statistics
    pub fn get_stats(&self) -> ChunkStats {
        self.stats.read().unwrap().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        if let Ok(mut stats) = self.stats.write() {
            *stats = ChunkStats::default();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DummyVocoder, MelSpectrogram};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_streaming_pipeline_creation() {
        let vocoder = Arc::new(DummyVocoder::new());
        let pipeline = StreamingPipeline::new(vocoder);

        let stats = pipeline.get_stats();
        assert_eq!(stats.active_streams, 0);
        assert_eq!(stats.chunks_processed, 0);
    }

    #[tokio::test]
    #[ignore] // Ignore slow test
    async fn test_chunk_processing() {
        let vocoder = Arc::new(DummyVocoder::new());
        let mut pipeline = StreamingPipeline::new(vocoder);

        let config = StreamingConfig::default();
        pipeline.initialize(config).await.unwrap();

        // Create test mel spectrogram
        let mel_data = vec![vec![0.5; 80]; 100];
        let mel = MelSpectrogram::new(mel_data, 22050, 256);

        let result = pipeline.process_chunk(mel).await;
        assert!(result.is_ok());

        let stats = pipeline.get_stats();
        assert_eq!(stats.chunks_processed, 1);
        assert!(stats.avg_processing_time_ms > 0.0);
    }

    #[tokio::test]
    #[ignore] // Ignore slow test
    async fn test_batch_processing() {
        let vocoder = Arc::new(DummyVocoder::new());
        let mut pipeline = StreamingPipeline::new(vocoder);

        let config = StreamingConfig::default();
        pipeline.initialize(config).await.unwrap();

        // Create test mel spectrograms
        let mel_data = vec![vec![0.5; 80]; 100];
        let mels = vec![
            MelSpectrogram::new(mel_data.clone(), 22050, 256),
            MelSpectrogram::new(mel_data.clone(), 22050, 256),
            MelSpectrogram::new(mel_data, 22050, 256),
        ];

        let results = pipeline.process_batch(mels).await;
        assert!(results.is_ok());

        let audio_buffers = results.unwrap();
        assert_eq!(audio_buffers.len(), 3);
    }

    #[tokio::test]
    async fn test_chunk_processor() {
        let vocoder = Arc::new(DummyVocoder::new());
        let config = StreamingConfig::default();
        let processor = ChunkProcessor::new(vocoder, config);

        let mel_data = vec![vec![0.5; 80]; 100];
        let mel = MelSpectrogram::new(mel_data, 22050, 256);

        let result = processor.process(mel).await;
        assert!(result.is_ok());

        let stats = processor.get_stats();
        assert_eq!(stats.chunks_processed, 1);
        assert!(stats.avg_processing_time > 0.0);
    }
}
