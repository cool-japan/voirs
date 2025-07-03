//! Stream management, ordering, and state tracking functionality.

use crate::{
    audio::AudioBuffer,
    error::Result,
    types::AudioFormat,
    VoirsError,
};
use futures::{stream::BoxStream, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, VecDeque},
    pin::Pin,
    task::{Context, Poll},
    time::{Duration, Instant, SystemTime},
};
use tokio::sync::RwLock;

/// Configuration for streaming synthesis
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Maximum characters per chunk
    pub max_chunk_chars: usize,
    
    /// Minimum characters before triggering real-time synthesis
    pub min_chunk_chars: usize,
    
    /// Maximum number of chunks to process concurrently
    pub max_concurrent_chunks: usize,
    
    /// Overlap frames for smooth concatenation
    pub overlap_frames: usize,
    
    /// Maximum latency for real-time synthesis
    pub max_latency: Duration,
    
    /// Buffer size for real-time processing
    pub realtime_buffer_size: usize,
    
    /// Enable chunk reordering (maintain order even with concurrent processing)
    pub maintain_order: bool,
    
    /// Maximum buffer size before forcing synthesis
    pub max_buffer_size: usize,
    
    /// Timeout for urgent synthesis operations
    pub urgent_timeout: Duration,
    
    /// Timeout for synthesis tasks before dropping
    pub task_timeout: Duration,
    
    /// Quality vs latency trade-off (0.0 = fastest, 1.0 = best quality)
    pub quality_vs_latency: f32,
    
    /// Enable adaptive chunk sizing based on performance
    pub adaptive_chunking: bool,
    
    /// Target real-time factor (processing_time / audio_duration)
    pub target_rtf: f32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_chunk_chars: 200,
            min_chunk_chars: 50,
            max_concurrent_chunks: 4,
            overlap_frames: 8,
            max_latency: Duration::from_millis(500),
            realtime_buffer_size: 32,
            maintain_order: true,
            max_buffer_size: 2000,
            urgent_timeout: Duration::from_millis(100),
            task_timeout: Duration::from_millis(1000),
            quality_vs_latency: 0.7,
            adaptive_chunking: true,
            target_rtf: 0.3,
        }
    }
}

impl StreamingConfig {
    /// Create config optimized for low latency
    pub fn low_latency() -> Self {
        Self {
            max_chunk_chars: 100,
            min_chunk_chars: 25,
            max_latency: Duration::from_millis(200),
            urgent_timeout: Duration::from_millis(50),
            quality_vs_latency: 0.3,
            target_rtf: 0.2,
            ..Default::default()
        }
    }
    
    /// Create config optimized for high quality
    pub fn high_quality() -> Self {
        Self {
            max_chunk_chars: 400,
            min_chunk_chars: 100,
            max_latency: Duration::from_millis(1000),
            quality_vs_latency: 1.0,
            target_rtf: 0.8,
            overlap_frames: 16,
            ..Default::default()
        }
    }
    
    /// Create config optimized for batch processing
    pub fn batch_processing() -> Self {
        Self {
            max_chunk_chars: 500,
            min_chunk_chars: 200,
            max_concurrent_chunks: 8,
            max_latency: Duration::from_millis(5000),
            quality_vs_latency: 0.9,
            target_rtf: 1.0,
            maintain_order: true,
            ..Default::default()
        }
    }
    
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.min_chunk_chars >= self.max_chunk_chars {
            return Err(VoirsError::invalid_config(
                "chunk_chars",
                format!("min={}, max={}", self.min_chunk_chars, self.max_chunk_chars),
                "min_chunk_chars must be less than max_chunk_chars"
            ));
        }
        
        if self.max_concurrent_chunks == 0 {
            return Err(VoirsError::invalid_config(
                "max_concurrent_chunks",
                "0",
                "must be greater than 0"
            ));
        }
        
        if self.quality_vs_latency < 0.0 || self.quality_vs_latency > 1.0 {
            return Err(VoirsError::invalid_config(
                "quality_vs_latency",
                self.quality_vs_latency.to_string(),
                "must be between 0.0 and 1.0"
            ));
        }
        
        Ok(())
    }
    
    /// Adapt configuration based on performance metrics
    pub fn adapt_for_performance(&mut self, rtf: f32, latency: Duration) {
        if !self.adaptive_chunking {
            return;
        }
        
        // If we're too slow (RTF > target), reduce chunk size
        if rtf > self.target_rtf * 1.2 {
            self.max_chunk_chars = (self.max_chunk_chars * 9 / 10).max(self.min_chunk_chars + 10);
            self.min_chunk_chars = (self.min_chunk_chars * 9 / 10).max(20);
        }
        // If we're fast enough and latency is low, increase chunk size
        else if rtf < self.target_rtf * 0.8 && latency < self.max_latency / 2 {
            self.max_chunk_chars = (self.max_chunk_chars * 11 / 10).min(1000);
            self.min_chunk_chars = (self.min_chunk_chars * 11 / 10).min(self.max_chunk_chars / 2);
        }
    }
}

/// State tracking for streaming synthesis
#[derive(Debug, Clone)]
pub struct StreamingState {
    /// Number of chunks processed
    pub chunks_processed: usize,
    
    /// Total audio duration generated
    pub total_duration: f32,
    
    /// Average processing time per chunk
    pub avg_processing_time: Duration,
    
    /// Current synthesis quality metrics
    pub quality_metrics: QualityMetrics,
    
    /// Processing start time
    pub processing_start: Option<Instant>,
    
    /// Total text characters processed
    pub total_chars_processed: usize,
    
    /// Throughput metrics
    pub throughput: ThroughputMetrics,
    
    /// Error tracking
    pub error_count: usize,
    pub last_error: Option<String>,
}

impl Default for StreamingState {
    fn default() -> Self {
        Self {
            chunks_processed: 0,
            total_duration: 0.0,
            avg_processing_time: Duration::ZERO,
            quality_metrics: QualityMetrics::default(),
            processing_start: None,
            total_chars_processed: 0,
            throughput: ThroughputMetrics::default(),
            error_count: 0,
            last_error: None,
        }
    }
}

impl StreamingState {
    /// Reset state for new synthesis session
    pub fn reset_for_new_synthesis(&mut self) {
        *self = Self::default();
        self.processing_start = Some(Instant::now());
    }
    
    /// Update state with processed chunk
    pub fn update_with_chunk(&mut self, chunk: &AudioChunk) {
        self.chunks_processed += 1;
        self.total_duration += chunk.audio.duration();
        self.total_chars_processed += chunk.text.len();
        
        // Update average processing time
        let total_time = self.avg_processing_time.as_nanos() as u64 * (self.chunks_processed - 1) as u64
            + chunk.processing_time.as_nanos() as u64;
        self.avg_processing_time = Duration::from_nanos(total_time / self.chunks_processed as u64);
        
        // Update quality metrics
        self.quality_metrics.update_with_chunk(chunk);
        
        // Update throughput
        if let Some(start) = self.processing_start {
            let elapsed = start.elapsed();
            self.throughput.update(self.total_chars_processed, elapsed);
        }
    }
    
    /// Record error
    pub fn record_error(&mut self, error: &str) {
        self.error_count += 1;
        self.last_error = Some(error.to_string());
    }
    
    /// Get overall real-time factor
    pub fn overall_rtf(&self) -> f32 {
        if self.total_duration <= 0.0 {
            return 0.0;
        }
        
        if let Some(start) = self.processing_start {
            let processing_time = start.elapsed().as_secs_f32();
            processing_time / self.total_duration
        } else {
            self.avg_processing_time.as_secs_f32() / (self.total_duration / self.chunks_processed as f32)
        }
    }
    
    /// Check if synthesis is meeting real-time requirements
    pub fn is_realtime(&self) -> bool {
        self.overall_rtf() <= 1.0
    }
    
    /// Get processing efficiency (0.0 to 1.0, higher is better)
    pub fn efficiency(&self) -> f32 {
        let rtf = self.overall_rtf();
        if rtf <= 1.0 {
            1.0 - rtf * 0.5 // Reward sub-real-time performance
        } else {
            1.0 / rtf // Penalize super-real-time
        }
    }
}

/// Quality metrics for streaming synthesis
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Real-time factor (processing_time / audio_duration)
    pub real_time_factor: f32,
    
    /// Average latency
    pub avg_latency: Duration,
    
    /// Buffer underruns
    pub underruns: usize,
    
    /// Chunks dropped due to timing
    pub dropped_chunks: usize,
    
    /// Peak RTF observed
    pub peak_rtf: f32,
    
    /// Latency distribution
    pub latency_percentiles: LatencyPercentiles,
    
    /// Quality consistency score (0.0 to 1.0)
    pub consistency_score: f32,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            real_time_factor: 0.0,
            avg_latency: Duration::ZERO,
            underruns: 0,
            dropped_chunks: 0,
            peak_rtf: 0.0,
            latency_percentiles: LatencyPercentiles::default(),
            consistency_score: 1.0,
        }
    }
}

impl QualityMetrics {
    /// Update metrics with new chunk data
    pub fn update_with_chunk(&mut self, chunk: &AudioChunk) {
        let chunk_rtf = chunk.real_time_factor();
        
        // Update average RTF
        self.real_time_factor = if self.real_time_factor == 0.0 {
            chunk_rtf
        } else {
            self.real_time_factor * 0.9 + chunk_rtf * 0.1 // Exponential moving average
        };
        
        // Update peak RTF
        if chunk_rtf > self.peak_rtf {
            self.peak_rtf = chunk_rtf;
        }
        
        // Update latency percentiles
        self.latency_percentiles.add_sample(chunk.processing_time);
        
        // Update consistency score based on RTF variance
        let rtf_variance = (chunk_rtf - self.real_time_factor).abs();
        self.consistency_score = self.consistency_score * 0.95 + (1.0 - rtf_variance.min(1.0)) * 0.05;
    }
    
    /// Check if quality is degrading
    pub fn is_quality_degrading(&self) -> bool {
        self.real_time_factor > 1.0 || 
        self.peak_rtf > 1.0 ||  // Check peak RTF as well 
        self.consistency_score < 0.7 ||
        self.dropped_chunks > 0
    }
}

/// Latency distribution tracking
#[derive(Debug, Clone)]
pub struct LatencyPercentiles {
    samples: VecDeque<Duration>,
    max_samples: usize,
}

impl Default for LatencyPercentiles {
    fn default() -> Self {
        Self {
            samples: VecDeque::new(),
            max_samples: 100,
        }
    }
}

impl LatencyPercentiles {
    fn add_sample(&mut self, latency: Duration) {
        self.samples.push_back(latency);
        if self.samples.len() > self.max_samples {
            self.samples.pop_front();
        }
    }
    
    pub fn percentile(&self, p: f32) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        
        let mut sorted: Vec<_> = self.samples.iter().collect();
        sorted.sort();
        
        let index = ((sorted.len() - 1) as f32 * p / 100.0) as usize;
        *sorted[index]
    }
    
    pub fn p50(&self) -> Duration { self.percentile(50.0) }
    pub fn p95(&self) -> Duration { self.percentile(95.0) }
    pub fn p99(&self) -> Duration { self.percentile(99.0) }
}

/// Throughput metrics
#[derive(Debug, Clone, Default)]
pub struct ThroughputMetrics {
    /// Characters per second
    pub chars_per_second: f32,
    
    /// Audio seconds per wall-clock second
    pub audio_per_second: f32,
    
    /// Peak throughput observed
    pub peak_chars_per_second: f32,
}

impl ThroughputMetrics {
    fn update(&mut self, total_chars: usize, elapsed: Duration) {
        if elapsed.as_secs_f32() > 0.0 {
            self.chars_per_second = total_chars as f32 / elapsed.as_secs_f32();
            
            if self.chars_per_second > self.peak_chars_per_second {
                self.peak_chars_per_second = self.chars_per_second;
            }
        }
    }
}

/// Latency statistics for real-time processing
#[derive(Debug, Clone)]
pub struct LatencyStats {
    /// Total number of samples
    pub sample_count: usize,
    
    /// Average latency
    pub average_latency: Duration,
    
    /// 95th percentile latency
    pub p95_latency: Duration,
    
    /// Maximum latency observed
    pub max_latency: Duration,
    
    /// Number of urgent syntheses
    pub urgent_count: usize,
    
    /// Recent latency samples
    pub recent_samples: VecDeque<Duration>,
}

impl Default for LatencyStats {
    fn default() -> Self {
        Self {
            sample_count: 0,
            average_latency: Duration::ZERO,
            p95_latency: Duration::ZERO,
            max_latency: Duration::ZERO,
            urgent_count: 0,
            recent_samples: VecDeque::new(),
        }
    }
}

impl LatencyStats {
    const MAX_RECENT_SAMPLES: usize = 100;
    
    /// Update statistics with new latency sample
    pub fn update(&mut self, latency: Duration) {
        self.sample_count += 1;
        
        // Update average latency
        let total_nanos = self.average_latency.as_nanos() as u64 * (self.sample_count - 1) as u64
            + latency.as_nanos() as u64;
        self.average_latency = Duration::from_nanos(total_nanos / self.sample_count as u64);
        
        // Update max latency
        if latency > self.max_latency {
            self.max_latency = latency;
        }
        
        // Add to recent samples
        self.recent_samples.push_back(latency);
        if self.recent_samples.len() > Self::MAX_RECENT_SAMPLES {
            self.recent_samples.pop_front();
        }
        
        // Update percentiles
        self.update_percentiles();
    }
    
    /// Update statistics for urgent synthesis
    pub fn update_urgent(&mut self, latency: Duration) {
        self.urgent_count += 1;
        self.update(latency);
    }
    
    fn update_percentiles(&mut self) {
        if self.recent_samples.is_empty() {
            return;
        }
        
        let mut sorted: Vec<_> = self.recent_samples.iter().collect();
        sorted.sort();
        
        let p95_index = (sorted.len() as f32 * 0.95) as usize;
        if p95_index < sorted.len() {
            self.p95_latency = *sorted[p95_index];
        }
    }
}

/// Audio chunk produced by streaming synthesis
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Chunk identifier for ordering
    pub chunk_id: usize,
    
    /// Generated audio
    pub audio: AudioBuffer,
    
    /// Original text for this chunk
    pub text: String,
    
    /// Time taken to process this chunk
    pub processing_time: Duration,
    
    /// Additional metadata
    pub metadata: ChunkMetadata,
}

impl AudioChunk {
    /// Get real-time factor for this chunk
    pub fn real_time_factor(&self) -> f32 {
        let audio_duration = Duration::from_secs_f32(self.audio.duration());
        if audio_duration.as_secs_f32() > 0.0 {
            self.processing_time.as_secs_f32() / audio_duration.as_secs_f32()
        } else {
            f32::INFINITY
        }
    }

    /// Check if this chunk was processed in real-time
    pub fn is_realtime(&self) -> bool {
        self.real_time_factor() <= 1.0
    }
    
    /// Get processing efficiency score (higher is better, capped at 1.0)
    pub fn efficiency_score(&self) -> f32 {
        let rtf = self.real_time_factor();
        if rtf <= 1.0 {
            // Cap efficiency at 1.0 for very fast processing
            (1.0 / (rtf + 0.1)).min(1.0)
        } else {
            1.0 / rtf.powf(2.0) // Heavily penalize slower than real-time
        }
    }
    
    /// Export chunk metadata as JSON
    pub fn export_metadata(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.metadata)
            .map_err(|e| VoirsError::serialization(
                "JSON", 
                format!("Failed to serialize chunk metadata: {}", e)
            ))
    }
    
    /// Create chunk from components
    pub fn new(
        chunk_id: usize,
        audio: AudioBuffer,
        text: String,
        processing_time: Duration,
        phoneme_count: usize,
        mel_frames: usize,
    ) -> Self {
        let metadata = ChunkMetadata {
            phoneme_count,
            mel_frames,
            is_sentence_boundary: text.trim_end().ends_with(['.', '!', '?']),
            is_paragraph_boundary: text.trim_end().ends_with('\n'),
            real_time_factor: Some(processing_time.as_secs_f32() / audio.duration()),
            confidence_score: 1.0, // TODO: Calculate actual confidence
        };
        
        Self {
            chunk_id,
            audio,
            text,
            processing_time,
            metadata,
        }
    }
}

/// Metadata for audio chunks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Number of phonemes in this chunk
    pub phoneme_count: usize,
    
    /// Number of mel spectrogram frames
    pub mel_frames: usize,
    
    /// Whether this chunk ends at a sentence boundary
    pub is_sentence_boundary: bool,
    
    /// Whether this chunk ends at a paragraph boundary
    pub is_paragraph_boundary: bool,
    
    /// Real-time factor for this chunk
    pub real_time_factor: Option<f32>,
    
    /// Confidence score for synthesis quality (0.0 to 1.0)
    pub confidence_score: f32,
}

/// Stream for ordered chunk processing
pub struct OrderedChunkStream {
    chunks: BoxStream<'static, Result<AudioChunk>>,
    next_expected_id: usize,
    buffer: BTreeMap<usize, AudioChunk>,
    max_buffer_size: usize,
    stats: StreamStats,
}

impl OrderedChunkStream {
    /// Create new ordered stream
    pub fn new(
        chunks: BoxStream<'static, Result<AudioChunk>>,
        max_buffer_size: usize,
    ) -> Self {
        Self {
            chunks,
            next_expected_id: 0,
            buffer: BTreeMap::new(),
            max_buffer_size,
            stats: StreamStats::default(),
        }
    }
    
    /// Get stream statistics
    pub fn stats(&self) -> &StreamStats {
        &self.stats
    }
    
    /// Reset stream statistics
    pub fn reset_stats(&mut self) {
        self.stats = StreamStats::default();
    }
}

impl Stream for OrderedChunkStream {
    type Item = Result<AudioChunk>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // First check if we have the next expected chunk in buffer
        let expected_id = self.next_expected_id;
        if let Some(chunk) = self.buffer.remove(&expected_id) {
            self.next_expected_id += 1;
            self.stats.chunks_delivered += 1;
            return Poll::Ready(Some(Ok(chunk)));
        }

        // Poll for new chunks
        loop {
            match self.chunks.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    self.stats.chunks_received += 1;
                    
                    if chunk.chunk_id == self.next_expected_id {
                        // This is the next expected chunk
                        self.next_expected_id += 1;
                        self.stats.chunks_delivered += 1;
                        return Poll::Ready(Some(Ok(chunk)));
                    } else if chunk.chunk_id > self.next_expected_id {
                        // Future chunk, buffer it
                        if self.buffer.len() < self.max_buffer_size {
                            self.buffer.insert(chunk.chunk_id, chunk);
                            self.stats.chunks_buffered += 1;
                        } else {
                            // Buffer full, emit error
                            self.stats.buffer_overflows += 1;
                            return Poll::Ready(Some(Err(VoirsError::internal(
                                "streaming", 
                                "Chunk ordering buffer overflow"
                            ))));
                        }
                    } else {
                        // Old chunk, drop it
                        self.stats.chunks_dropped += 1;
                        tracing::warn!("Dropping old chunk {} (expected {})", chunk.chunk_id, self.next_expected_id);
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    self.stats.errors += 1;
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(None) => {
                    // No more chunks, emit any remaining buffered chunks in order
                    if let Some((&id, _)) = self.buffer.iter().next() {
                        if let Some(chunk) = self.buffer.remove(&id) {
                            self.stats.chunks_delivered += 1;
                            return Poll::Ready(Some(Ok(chunk)));
                        }
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Statistics for ordered chunk streams
#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    /// Total chunks received
    pub chunks_received: usize,
    
    /// Total chunks delivered in order
    pub chunks_delivered: usize,
    
    /// Chunks currently buffered
    pub chunks_buffered: usize,
    
    /// Chunks dropped due to being out of order
    pub chunks_dropped: usize,
    
    /// Buffer overflow events
    pub buffer_overflows: usize,
    
    /// Processing errors
    pub errors: usize,
}

impl StreamStats {
    /// Calculate ordering efficiency (0.0 to 1.0)
    pub fn ordering_efficiency(&self) -> f32 {
        if self.chunks_received == 0 {
            return 1.0;
        }
        
        self.chunks_delivered as f32 / self.chunks_received as f32
    }
    
    /// Calculate drop rate
    pub fn drop_rate(&self) -> f32 {
        if self.chunks_received == 0 {
            return 0.0;
        }
        
        self.chunks_dropped as f32 / self.chunks_received as f32
    }
    
    /// Check if stream is healthy
    pub fn is_healthy(&self) -> bool {
        self.drop_rate() < 0.01 && // Less than 1% drop rate
        self.buffer_overflows == 0 &&
        self.ordering_efficiency() > 0.95
    }
}

/// Stream combiner for merging multiple audio streams
pub struct StreamCombiner {
    streams: Vec<BoxStream<'static, Result<AudioChunk>>>,
    output_format: AudioFormat,
    combination_strategy: CombinationStrategy,
}

impl StreamCombiner {
    /// Create new stream combiner
    pub fn new(
        streams: Vec<BoxStream<'static, Result<AudioChunk>>>,
        output_format: AudioFormat,
        strategy: CombinationStrategy,
    ) -> Self {
        Self {
            streams,
            output_format,
            combination_strategy: strategy,
        }
    }
    
    /// Combine streams into single output stream
    pub async fn combine(mut self) -> Result<impl Stream<Item = Result<AudioChunk>>> {
        match self.combination_strategy {
            CombinationStrategy::Concatenate => {
                // Concatenate streams sequentially
                let combined = futures::stream::iter(self.streams)
                    .then(|stream| async move { stream.collect::<Vec<_>>().await })
                    .map(|chunks| futures::stream::iter(chunks))
                    .flatten();
                
                Ok(Box::pin(combined) as BoxStream<'static, Result<AudioChunk>>)
            }
            CombinationStrategy::Interleave => {
                // Interleave chunks from multiple streams
                // TODO: Implement proper interleaving
                Err(VoirsError::internal("streaming", "Interleaving not yet implemented"))
            }
            CombinationStrategy::Mix => {
                // Mix audio from multiple streams
                // TODO: Implement audio mixing
                Err(VoirsError::internal("streaming", "Audio mixing not yet implemented"))
            }
        }
    }
}

/// Strategy for combining multiple streams
#[derive(Debug, Clone, Copy)]
pub enum CombinationStrategy {
    /// Concatenate streams sequentially
    Concatenate,
    /// Interleave chunks from streams
    Interleave,
    /// Mix audio from streams
    Mix,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioBuffer;

    #[test]
    fn test_streaming_config_validation() {
        let mut config = StreamingConfig::default();
        assert!(config.validate().is_ok());
        
        // Invalid: min >= max
        config.min_chunk_chars = 200;
        config.max_chunk_chars = 200;
        assert!(config.validate().is_err());
        
        // Invalid: quality_vs_latency out of range
        config = StreamingConfig::default();
        config.quality_vs_latency = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_presets() {
        let low_latency = StreamingConfig::low_latency();
        assert!(low_latency.max_latency < StreamingConfig::default().max_latency);
        assert!(low_latency.quality_vs_latency < 0.5);
        
        let high_quality = StreamingConfig::high_quality();
        assert!(high_quality.quality_vs_latency > 0.9);
        assert!(high_quality.overlap_frames > StreamingConfig::default().overlap_frames);
    }

    #[test]
    fn test_adaptive_config() {
        let mut config = StreamingConfig::default();
        config.adaptive_chunking = true;
        
        let original_max = config.max_chunk_chars;
        
        // Simulate poor performance
        config.adapt_for_performance(1.5, Duration::from_millis(800));
        assert!(config.max_chunk_chars < original_max);
        
        // Simulate good performance
        config.adapt_for_performance(0.1, Duration::from_millis(50));
        assert!(config.max_chunk_chars >= original_max * 9 / 10);
    }

    #[test]
    fn test_streaming_state() {
        let mut state = StreamingState::default();
        assert_eq!(state.chunks_processed, 0);
        assert!(!state.is_realtime()); // No data yet
        
        // Create test chunk
        let audio = AudioBuffer::sine_wave(440.0, 1.0, 22050, 0.5);
        let chunk = AudioChunk::new(
            0,
            audio,
            "Test text".to_string(),
            Duration::from_millis(100),
            10,
            100,
        );
        
        state.update_with_chunk(&chunk);
        assert_eq!(state.chunks_processed, 1);
        assert!(state.is_realtime()); // 100ms for 1s audio is real-time
    }

    #[test]
    fn test_audio_chunk_metrics() {
        let audio = AudioBuffer::sine_wave(440.0, 1.0, 22050, 0.5);
        let chunk = AudioChunk::new(
            0,
            audio,
            "Test".to_string(),
            Duration::from_millis(100),
            4,
            100,
        );

        let rtf = chunk.real_time_factor();
        assert!(rtf > 0.0);
        assert!(rtf < 1.0); // 100ms processing for 1s audio
        
        assert!(chunk.is_realtime());
        assert!(chunk.efficiency_score() > 0.5);
    }

    #[test]
    fn test_latency_percentiles() {
        let mut percentiles = LatencyPercentiles::default();
        
        // Add some samples
        for i in 1..=100 {
            percentiles.add_sample(Duration::from_millis(i));
        }
        
        assert_eq!(percentiles.p50(), Duration::from_millis(50));
        assert_eq!(percentiles.p95(), Duration::from_millis(95));
        assert_eq!(percentiles.p99(), Duration::from_millis(99));
    }

    #[test]
    fn test_latency_stats() {
        let mut stats = LatencyStats::default();
        
        stats.update(Duration::from_millis(100));
        stats.update(Duration::from_millis(200));
        
        assert_eq!(stats.sample_count, 2);
        assert_eq!(stats.average_latency, Duration::from_millis(150));
        assert_eq!(stats.max_latency, Duration::from_millis(200));
    }

    #[test]
    fn test_quality_metrics() {
        let mut metrics = QualityMetrics::default();
        
        let audio = AudioBuffer::sine_wave(440.0, 1.0, 22050, 0.5);
        let chunk = AudioChunk::new(
            0,
            audio,
            "Test".to_string(),
            Duration::from_millis(500), // Slower than real-time
            10,
            100,
        );
        
        metrics.update_with_chunk(&chunk);
        
        assert!(metrics.real_time_factor > 0.0);
        assert!(metrics.peak_rtf > 0.0);
        assert!(metrics.consistency_score <= 1.0);
    }

    #[test]
    fn test_stream_stats() {
        let mut stats = StreamStats::default();
        
        stats.chunks_received = 100;
        stats.chunks_delivered = 95;
        stats.chunks_dropped = 5;
        
        assert_eq!(stats.ordering_efficiency(), 0.95);
        assert_eq!(stats.drop_rate(), 0.05);
        assert!(!stats.is_healthy()); // Drop rate too high
    }

    #[test]
    fn test_chunk_metadata_serialization() {
        let audio = AudioBuffer::sine_wave(440.0, 1.0, 22050, 0.5);
        let chunk = AudioChunk::new(
            0,
            audio,
            "Test".to_string(),
            Duration::from_millis(100),
            10,
            100,
        );
        
        let json = chunk.export_metadata().unwrap();
        assert!(json.contains("phoneme_count"));
        assert!(json.contains("mel_frames"));
        
        // Should be able to deserialize
        let metadata: ChunkMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(metadata.phoneme_count, 10);
        assert_eq!(metadata.mel_frames, 100);
    }

    #[tokio::test]
    async fn test_ordered_chunk_stream() {
        // Create test chunks in random order
        let chunks = vec![
            Ok(AudioChunk::new(2, AudioBuffer::sine_wave(440.0, 0.1, 22050, 0.5), "Third".to_string(), Duration::from_millis(10), 5, 20)),
            Ok(AudioChunk::new(0, AudioBuffer::sine_wave(440.0, 0.1, 22050, 0.5), "First".to_string(), Duration::from_millis(10), 5, 20)),
            Ok(AudioChunk::new(1, AudioBuffer::sine_wave(440.0, 0.1, 22050, 0.5), "Second".to_string(), Duration::from_millis(10), 5, 20)),
        ];
        
        let stream = futures::stream::iter(chunks);
        let mut ordered_stream = OrderedChunkStream::new(Box::pin(stream), 10);
        
        // Should receive chunks in order
        let chunk0 = ordered_stream.next().await.unwrap().unwrap();
        assert_eq!(chunk0.chunk_id, 0);
        assert_eq!(chunk0.text, "First");
        
        let chunk1 = ordered_stream.next().await.unwrap().unwrap();
        assert_eq!(chunk1.chunk_id, 1);
        assert_eq!(chunk1.text, "Second");
        
        let chunk2 = ordered_stream.next().await.unwrap().unwrap();
        assert_eq!(chunk2.chunk_id, 2);
        assert_eq!(chunk2.text, "Third");
        
        // Check stats
        let stats = ordered_stream.stats();
        assert_eq!(stats.chunks_received, 3);
        assert_eq!(stats.chunks_delivered, 3);
        assert!(stats.is_healthy());
    }
}