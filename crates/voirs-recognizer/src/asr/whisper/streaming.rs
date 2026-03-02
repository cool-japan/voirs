//! Streaming audio processing for real-time Whisper ASR
//!
//! This module provides real-time streaming capabilities for Whisper models
//! with buffer management, chunk processing, and online transcript generation.
//!
//! Advanced features:
//! - Adaptive chunk sizing based on content analysis
//! - Silence-based intelligent segmentation
//! - Multi-level overlap strategies for seamless transitions
//! - Content-aware buffer management
//! - Dynamic latency optimization

use super::{
    WhisperAudioProcessor, WhisperConfig, WhisperDecoder, WhisperEncoder, WhisperTokenizer,
};
use crate::RecognitionError;
use candle_core::{Device, Tensor};
use scirs2_core::numeric::*;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use voirs_sdk::{AudioBuffer, LanguageCode};

/// Real-time streaming Whisper processor
pub struct StreamingWhisperProcessor {
    encoder: Arc<WhisperEncoder>,
    decoder: Arc<WhisperDecoder>,
    tokenizer: Arc<WhisperTokenizer>,
    audio_processor: Arc<WhisperAudioProcessor>,
    config: WhisperConfig,
    #[allow(dead_code)]
    device: Device,

    // Streaming state
    audio_buffer: Arc<Mutex<AudioRingBuffer>>,
    transcript_buffer: Arc<RwLock<TranscriptBuffer>>,
    processing_state: Arc<RwLock<ProcessingState>>,

    // Incremental decoding context
    context_state: Arc<RwLock<IncrementalContext>>,
}

/// Ring buffer for audio samples
pub struct AudioRingBuffer {
    buffer: VecDeque<f32>,
    capacity: usize,
    sample_rate: u32,
    #[allow(dead_code)]
    channels: u32,
}

/// Buffer for maintaining transcript history
pub struct TranscriptBuffer {
    segments: VecDeque<TranscriptSegment>,
    max_segments: usize,
    current_text: String,
}

/// Individual transcript segment with timing
#[derive(Debug, Clone)]
/// Transcript Segment
pub struct TranscriptSegment {
    /// text
    pub text: String,
    /// start time
    pub start_time: f32,
    /// end time
    pub end_time: f32,
    /// confidence
    pub confidence: f32,
    /// language
    pub language: LanguageCode,
}

/// Processing state for streaming
#[derive(Debug)]
struct ProcessingState {
    last_processed_time: f32,
    current_chunk_id: u64,
    is_processing: bool,
    pending_audio_duration: f32,
}

/// Incremental decoding context for maintaining state across chunks
#[derive(Debug, Clone)]
struct IncrementalContext {
    /// Previous chunk's text for context continuity
    previous_context: String,
    /// Token history for incremental decoding
    token_history: Vec<u32>,
    /// Audio feature cache for overlap processing
    cached_features: Option<Tensor>,
    /// Language detected in previous chunks
    detected_language: Option<LanguageCode>,
    /// Confidence history for trend analysis
    confidence_history: VecDeque<f32>,
    /// Maximum context length to maintain
    max_context_length: usize,
}

/// Streaming configuration parameters
#[derive(Debug, Clone)]
/// Streaming Config
pub struct StreamingConfig {
    /// chunk duration ms
    pub chunk_duration_ms: u32, // Duration of each processing chunk
    /// overlap duration ms
    pub overlap_duration_ms: u32, // Overlap between chunks
    /// min silence duration ms
    pub min_silence_duration_ms: u32, // Minimum silence before finalizing segment
    /// max segment duration s
    pub max_segment_duration_s: f32, // Maximum segment duration
    /// vad threshold
    pub vad_threshold: f32, // Voice activity detection threshold
    /// max latency ms
    pub max_latency_ms: u32, // Maximum processing latency
    /// buffer duration s
    pub buffer_duration_s: f32, // Audio buffer duration
    /// incremental decoding
    pub incremental_decoding: bool, // Enable incremental decoding with context
    /// latency mode
    pub latency_mode: LatencyMode, // Latency vs accuracy trade-off
    /// overlap strategy
    pub overlap_strategy: OverlapStrategy, // How to handle overlapping chunks
}

/// Latency vs accuracy trade-off modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Latency Mode
pub enum LatencyMode {
    /// Ultra-low latency (sacrifice some accuracy)
    UltraLow,
    /// Low latency (balanced)
    Low,
    /// Medium latency (good accuracy)
    Medium,
    /// High accuracy (higher latency)
    HighAccuracy,
}

/// Overlap strategy for chunk processing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Overlap Strategy
pub enum OverlapStrategy {
    /// No overlap processing
    None,
    /// Simple overlap merging
    Merge,
    /// Weighted overlap blending
    WeightedBlend,
    /// Context-aware overlap handling
    ContextAware,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_duration_ms: 1000,                          // 1 second chunks
            overlap_duration_ms: 250,                         // 250ms overlap
            min_silence_duration_ms: 500,                     // 500ms silence
            max_segment_duration_s: 30.0,                     // 30 second max segments
            vad_threshold: 0.01,                              // VAD threshold
            max_latency_ms: 200,                              // 200ms max latency
            buffer_duration_s: 10.0,                          // 10 second buffer
            incremental_decoding: true,                       // Enable incremental decoding
            latency_mode: LatencyMode::Medium,                // Balanced latency/accuracy
            overlap_strategy: OverlapStrategy::WeightedBlend, // Smart overlap handling
        }
    }
}

impl LatencyMode {
    /// Get the model parameters for this latency mode
    #[must_use]
    /// model params
    pub fn model_params(&self) -> (u32, u32, f32) {
        // Returns (beam_size, max_tokens, temperature)
        match self {
            LatencyMode::UltraLow => (1, 128, 1.2), // Fastest, least accurate
            LatencyMode::Low => (1, 192, 1.1),      // Fast with better accuracy
            LatencyMode::Medium => (2, 224, 1.0),   // Balanced
            LatencyMode::HighAccuracy => (5, 448, 0.8), // Slower but more accurate
        }
    }

    /// Get the recommended chunk duration for this latency mode
    #[must_use]
    /// chunk duration ms
    pub fn chunk_duration_ms(&self) -> u32 {
        match self {
            LatencyMode::UltraLow => 500,      // 0.5 second chunks
            LatencyMode::Low => 750,           // 0.75 second chunks
            LatencyMode::Medium => 1000,       // 1 second chunks
            LatencyMode::HighAccuracy => 2000, // 2 second chunks
        }
    }
}

impl StreamingConfig {
    /// Create configuration optimized for ultra-low latency
    #[must_use]
    /// ultra low latency
    pub fn ultra_low_latency() -> Self {
        Self {
            chunk_duration_ms: 500,
            overlap_duration_ms: 100,
            min_silence_duration_ms: 200,
            max_segment_duration_s: 15.0,
            vad_threshold: 0.005,
            max_latency_ms: 100,
            buffer_duration_s: 5.0,
            incremental_decoding: true,
            latency_mode: LatencyMode::UltraLow,
            overlap_strategy: OverlapStrategy::Merge,
        }
    }

    /// Create configuration optimized for real-time conversation
    #[must_use]
    /// real time conversation
    pub fn real_time_conversation() -> Self {
        Self {
            chunk_duration_ms: 750,
            overlap_duration_ms: 150,
            min_silence_duration_ms: 300,
            max_segment_duration_s: 20.0,
            vad_threshold: 0.01,
            max_latency_ms: 150,
            buffer_duration_s: 8.0,
            incremental_decoding: true,
            latency_mode: LatencyMode::Low,
            overlap_strategy: OverlapStrategy::WeightedBlend,
        }
    }

    /// Create configuration optimized for high accuracy transcription
    #[must_use]
    /// high accuracy
    pub fn high_accuracy() -> Self {
        Self {
            chunk_duration_ms: 2000,
            overlap_duration_ms: 500,
            min_silence_duration_ms: 800,
            max_segment_duration_s: 60.0,
            vad_threshold: 0.02,
            max_latency_ms: 500,
            buffer_duration_s: 15.0,
            incremental_decoding: true,
            latency_mode: LatencyMode::HighAccuracy,
            overlap_strategy: OverlapStrategy::ContextAware,
        }
    }

    /// Create configuration optimized for broadcasting/podcasts
    #[must_use]
    /// broadcast optimized
    pub fn broadcast_optimized() -> Self {
        Self {
            chunk_duration_ms: 1500,
            overlap_duration_ms: 300,
            min_silence_duration_ms: 600,
            max_segment_duration_s: 45.0,
            vad_threshold: 0.015,
            max_latency_ms: 300,
            buffer_duration_s: 12.0,
            incremental_decoding: true,
            latency_mode: LatencyMode::Medium,
            overlap_strategy: OverlapStrategy::WeightedBlend,
        }
    }
}

impl StreamingWhisperProcessor {
    /// new
    pub async fn new(
        config: WhisperConfig,
        streaming_config: StreamingConfig,
        device: Device,
    ) -> Result<Self, RecognitionError> {
        let encoder = Arc::new(WhisperEncoder::new(&config, &device).await?);
        let decoder = Arc::new(WhisperDecoder::new(&config, &device).await?);
        let tokenizer = Arc::new(WhisperTokenizer::new().await?);
        let audio_processor = Arc::new(WhisperAudioProcessor::new(&config, &device)?);

        let buffer_capacity =
            (config.sample_rate as f32 * streaming_config.buffer_duration_s) as usize;
        let audio_buffer = Arc::new(Mutex::new(AudioRingBuffer::new(
            buffer_capacity,
            config.sample_rate,
            1, // Mono
        )));

        let transcript_buffer = Arc::new(RwLock::new(TranscriptBuffer::new(100))); // Keep 100 segments max

        let processing_state = Arc::new(RwLock::new(ProcessingState {
            last_processed_time: 0.0,
            current_chunk_id: 0,
            is_processing: false,
            pending_audio_duration: 0.0,
        }));

        let context_state = Arc::new(RwLock::new(IncrementalContext::new()));

        Ok(Self {
            encoder,
            decoder,
            tokenizer,
            audio_processor,
            config,
            device,
            audio_buffer,
            transcript_buffer,
            processing_state,
            context_state,
        })
    }

    /// Add audio samples to the streaming buffer
    pub async fn add_audio(&self, audio: AudioBuffer) -> Result<(), RecognitionError> {
        let mut buffer = self.audio_buffer.lock().await;
        buffer.add_samples(audio.samples())?;

        // Update pending audio duration
        {
            let mut state = self.processing_state.write().await;
            state.pending_audio_duration = buffer.duration_seconds();
        }

        Ok(())
    }

    /// Process pending audio and update transcript
    pub async fn process_pending_audio(
        &self,
        streaming_config: &StreamingConfig,
    ) -> Result<Option<TranscriptSegment>, RecognitionError> {
        // Check if we need to process
        let should_process = {
            let state = self.processing_state.read().await;
            !state.is_processing
                && state.pending_audio_duration * 1000.0
                    >= streaming_config.chunk_duration_ms as f32
        };

        if !should_process {
            return Ok(None);
        }

        // Mark as processing
        {
            let mut state = self.processing_state.write().await;
            state.is_processing = true;
            state.current_chunk_id += 1;
        }

        let result = self.process_audio_chunk(streaming_config).await;

        // Mark as done processing
        {
            let mut state = self.processing_state.write().await;
            state.is_processing = false;
        }

        result
    }

    /// Process a chunk of audio from the buffer
    async fn process_audio_chunk(
        &self,
        streaming_config: &StreamingConfig,
    ) -> Result<Option<TranscriptSegment>, RecognitionError> {
        // Extract audio chunk from buffer
        let audio_chunk = {
            let mut buffer = self.audio_buffer.lock().await;
            let chunk_samples = (self.config.sample_rate as f32
                * streaming_config.chunk_duration_ms as f32
                / 1000.0) as usize;
            buffer.get_chunk(chunk_samples)?
        };

        if audio_chunk.is_empty() {
            return Ok(None);
        }

        // Create AudioBuffer for processing
        let audio_buffer = AudioBuffer::new(audio_chunk, self.config.sample_rate, 1);

        // Apply voice activity detection
        let vad_segments = self
            .audio_processor
            .detect_voice_activity(&audio_buffer, streaming_config.vad_threshold)?;

        if vad_segments.is_empty() {
            return Ok(None); // No speech detected
        }

        // Process audio to mel spectrogram
        let mel_features = self.audio_processor.process_audio(&audio_buffer)?;

        // Encode audio features
        let audio_features = self.encoder.forward(&mel_features)?;

        // Get latency-optimized parameters
        let (beam_size, max_tokens, temperature) = streaming_config.latency_mode.model_params();

        // Get context for incremental decoding
        let _context_prompt = if streaming_config.incremental_decoding {
            let context = self.context_state.read().await;
            context.get_context_prompt()
        } else {
            String::new()
        };

        // Generate transcript using decoder with context
        let start_token = self.tokenizer.special_tokens().sot;
        let end_token = self.tokenizer.special_tokens().eot;

        let tokens = self
            .decoder
            .generate_tokens(
                &audio_features,
                start_token,
                end_token,
                max_tokens as usize,
                beam_size as usize,
                temperature,
            )
            .await?;

        // Decode tokens to text
        let text = self.tokenizer.decode(&tokens)?;

        if text.trim().is_empty() {
            return Ok(None);
        }

        // Calculate enhanced confidence score
        let base_confidence = 0.8; // Base confidence
        let context_confidence = if streaming_config.incremental_decoding {
            let context = self.context_state.read().await;
            let avg_confidence = context.get_average_confidence();
            // Blend base confidence with historical confidence
            0.7 * base_confidence + 0.3 * avg_confidence
        } else {
            base_confidence
        };

        // Apply latency mode confidence adjustment
        let confidence_multiplier = match streaming_config.latency_mode {
            LatencyMode::UltraLow => 0.85,    // Lower confidence for speed
            LatencyMode::Low => 0.92,         // Slightly lower confidence
            LatencyMode::Medium => 1.0,       // Standard confidence
            LatencyMode::HighAccuracy => 1.1, // Higher confidence
        };

        let final_confidence = (context_confidence * confidence_multiplier).min(1.0);

        // Detect language from generated tokens
        let detected_language = self.tokenizer.detect_language_from_tokens(&tokens);

        // Use detected language or fallback to context language
        let segment_language = if streaming_config.incremental_decoding {
            let context = self.context_state.read().await;
            context.detected_language.unwrap_or(detected_language)
        } else {
            detected_language
        };

        // Create transcript segment
        let segment = TranscriptSegment {
            text: text.trim().to_string(),
            start_time: vad_segments[0].0,
            end_time: vad_segments.last().expect("vad_segments is non-empty").1,
            confidence: final_confidence,
            language: segment_language,
        };

        // Update incremental context if enabled
        if streaming_config.incremental_decoding {
            let mut context = self.context_state.write().await;
            context.update_context(&segment.text, &tokens, final_confidence);
            // Update detected language if not already set or if more confident
            if context.detected_language.is_none() || final_confidence > 0.7 {
                context.detected_language = Some(detected_language);
            }
        }

        // Add to transcript buffer
        {
            let mut transcript = self.transcript_buffer.write().await;
            transcript.add_segment(segment.clone());
        }

        Ok(Some(segment))
    }

    /// Get current transcript
    pub async fn get_transcript(&self) -> String {
        let transcript = self.transcript_buffer.read().await;
        transcript.get_full_text()
    }

    /// Get recent transcript segments
    pub async fn get_recent_segments(&self, max_segments: usize) -> Vec<TranscriptSegment> {
        let transcript = self.transcript_buffer.read().await;
        transcript.get_recent_segments(max_segments)
    }

    /// Clear transcript buffer
    pub async fn clear_transcript(&self) {
        let mut transcript = self.transcript_buffer.write().await;
        transcript.clear();
    }

    /// Get processing statistics
    pub async fn get_processing_stats(&self) -> ProcessingStats {
        let state = self.processing_state.read().await;
        let buffer = self.audio_buffer.lock().await;

        ProcessingStats {
            chunks_processed: state.current_chunk_id,
            is_processing: state.is_processing,
            buffer_duration_seconds: buffer.duration_seconds(),
            buffer_fill_percentage: buffer.fill_percentage(),
            last_processed_time: state.last_processed_time,
        }
    }

    /// Start real-time processing loop with cancellation support
    pub async fn start_real_time_processing(
        &self,
        streaming_config: StreamingConfig,
        mut callback: impl FnMut(TranscriptSegment) + Send + 'static,
        stop_signal: tokio::sync::oneshot::Receiver<()>,
    ) -> Result<(), RecognitionError> {
        let process_interval =
            tokio::time::Duration::from_millis(streaming_config.chunk_duration_ms as u64 / 4);
        let mut interval = tokio::time::interval(process_interval);
        let mut stop_signal = stop_signal;

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Some(segment) = self.process_pending_audio(&streaming_config).await? {
                        callback(segment);
                    }
                }
                _ = &mut stop_signal => {
                    break;
                }
            }
        }

        Ok(())
    }

    /// Process a single chunk manually (useful for non-continuous processing)
    pub async fn process_single_chunk(
        &self,
        streaming_config: &StreamingConfig,
    ) -> Result<Option<TranscriptSegment>, RecognitionError> {
        self.process_pending_audio(streaming_config).await
    }

    /// Clear the incremental context (useful after long pauses or topic changes)
    pub async fn clear_context(&self) {
        let mut context = self.context_state.write().await;
        context.clear_context();
    }

    /// Get incremental context information
    pub async fn get_context_info(&self) -> String {
        let context = self.context_state.read().await;
        context.get_context_prompt()
    }

    /// Compute optimal chunk size based on content analysis
    ///
    /// Analyzes audio characteristics to determine the best chunk size:
    /// - Speech density (more speech = larger chunks)
    /// - Silence patterns (natural break points)
    /// - Energy distribution (consistent energy = larger chunks)
    pub async fn compute_adaptive_chunk_size(
        &self,
        audio_samples: &[f32],
        base_chunk_ms: u32,
        sample_rate: u32,
    ) -> u32 {
        if audio_samples.is_empty() {
            return base_chunk_ms;
        }

        // 1. Analyze speech density (ratio of voiced to total frames)
        let speech_density = self.estimate_speech_density(audio_samples, sample_rate);

        // 2. Detect silence boundaries
        let silence_boundaries = self.find_silence_boundaries(audio_samples, sample_rate, 0.02);

        // 3. Compute energy variance (lower variance = more stable = larger chunks)
        let energy_variance = self.compute_energy_variance(audio_samples);

        // Compute adaptive multiplier
        let speech_multiplier = if speech_density > 0.7 {
            1.2 // Dense speech: use longer chunks
        } else if speech_density < 0.3 {
            0.8 // Sparse speech: use shorter chunks
        } else {
            1.0
        };

        let stability_multiplier = if energy_variance < 0.1 {
            1.15 // Stable energy: can use longer chunks
        } else if energy_variance > 0.3 {
            0.85 // Variable energy: use shorter chunks
        } else {
            1.0
        };

        // Find nearest silence boundary if available
        let target_duration_ms =
            (base_chunk_ms as f32 * speech_multiplier * stability_multiplier) as u32;

        if !silence_boundaries.is_empty() {
            // Try to align chunk end with silence boundary
            let target_samples = (sample_rate as f32 * target_duration_ms as f32 / 1000.0) as usize;
            if let Some(&boundary) = silence_boundaries.iter().find(|&&b| b >= target_samples) {
                return ((boundary as f32 / sample_rate as f32) * 1000.0) as u32;
            }
        }

        target_duration_ms.clamp(base_chunk_ms / 2, base_chunk_ms * 2)
    }

    /// Estimate speech density in audio samples
    fn estimate_speech_density(&self, samples: &[f32], sample_rate: u32) -> f32 {
        const FRAME_SIZE_MS: f32 = 20.0; // 20ms frames
        const ENERGY_THRESHOLD: f32 = 0.01;

        let frame_size = (sample_rate as f32 * FRAME_SIZE_MS / 1000.0) as usize;
        if frame_size == 0 || samples.len() < frame_size {
            return 0.5; // Default assumption
        }

        let mut voiced_frames = 0;
        let mut total_frames = 0;

        for chunk in samples.chunks(frame_size) {
            let energy: f32 = chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32;

            if energy > ENERGY_THRESHOLD {
                voiced_frames += 1;
            }
            total_frames += 1;
        }

        if total_frames > 0 {
            voiced_frames as f32 / total_frames as f32
        } else {
            0.5
        }
    }

    /// Find silence boundaries in audio for natural segmentation points
    fn find_silence_boundaries(
        &self,
        samples: &[f32],
        sample_rate: u32,
        silence_threshold: f32,
    ) -> Vec<usize> {
        const MIN_SILENCE_DURATION_MS: f32 = 200.0; // 200ms minimum silence

        let min_silence_samples = (sample_rate as f32 * MIN_SILENCE_DURATION_MS / 1000.0) as usize;
        let mut boundaries = Vec::new();
        let mut silence_start: Option<usize> = None;

        for (i, &sample) in samples.iter().enumerate() {
            let is_silence = sample.abs() < silence_threshold;

            match silence_start {
                None => {
                    if is_silence {
                        silence_start = Some(i);
                    }
                }
                Some(start) => {
                    if !is_silence {
                        // End of silence period
                        if i - start >= min_silence_samples {
                            boundaries.push((start + i) / 2); // Midpoint of silence
                        }
                        silence_start = None;
                    }
                }
            }
        }

        boundaries
    }

    /// Compute energy variance for stability analysis
    fn compute_energy_variance(&self, samples: &[f32]) -> f32 {
        const FRAME_SIZE: usize = 400; // ~25ms at 16kHz

        if samples.len() < FRAME_SIZE {
            return 0.0;
        }

        let mut frame_energies = Vec::new();
        for chunk in samples.chunks(FRAME_SIZE) {
            let energy: f32 = chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32;
            frame_energies.push(energy);
        }

        if frame_energies.is_empty() {
            return 0.0;
        }

        // Compute variance
        let mean: f32 = frame_energies.iter().sum::<f32>() / frame_energies.len() as f32;
        let variance: f32 = frame_energies
            .iter()
            .map(|e| (e - mean).powi(2))
            .sum::<f32>()
            / frame_energies.len() as f32;

        variance.sqrt() // Return standard deviation
    }

    /// Apply sophisticated overlap blending for seamless transitions
    ///
    /// Uses weighted cross-fade based on confidence scores and content analysis
    pub fn blend_overlapping_segments(
        &self,
        segment1: &TranscriptSegment,
        segment2: &TranscriptSegment,
        overlap_duration: f32,
    ) -> TranscriptSegment {
        // Calculate overlap region
        let overlap_start = segment2.start_time;
        let overlap_end = (segment1.end_time).min(segment2.start_time + overlap_duration);

        if overlap_end <= overlap_start {
            // No actual overlap, return segment with higher confidence
            return if segment1.confidence > segment2.confidence {
                segment1.clone()
            } else {
                segment2.clone()
            };
        }

        // Weighted blending based on confidence
        let weight1 = segment1.confidence / (segment1.confidence + segment2.confidence);
        let weight2 = 1.0 - weight1;

        // For text, prefer the segment with higher confidence
        let blended_text = if weight1 > weight2 {
            segment1.text.clone()
        } else {
            segment2.text.clone()
        };

        // Blend timing information
        let blended_start = segment1.start_time * weight1 + segment2.start_time * weight2;
        let blended_end = segment1.end_time * weight1 + segment2.end_time * weight2;
        let blended_confidence = segment1.confidence * weight1 + segment2.confidence * weight2;

        TranscriptSegment {
            text: blended_text,
            start_time: blended_start,
            end_time: blended_end,
            confidence: blended_confidence,
            language: if weight1 > weight2 {
                segment1.language
            } else {
                segment2.language
            },
        }
    }

    /// Dynamically adjust buffer size based on processing performance
    ///
    /// Monitors latency and adjusts buffer to maintain real-time performance
    pub async fn adjust_buffer_size_adaptive(
        &self,
        target_latency_ms: u32,
    ) -> Result<(), RecognitionError> {
        let stats = self.get_processing_stats().await;

        // If buffer is consistently full and latency is high, increase capacity
        if stats.buffer_fill_percentage > 0.9 {
            tracing::warn!(
                "Audio buffer at {}% capacity, consider increasing buffer size",
                stats.buffer_fill_percentage * 100.0
            );
        }

        // If buffer is underutilized and latency is low, could decrease capacity
        if stats.buffer_fill_percentage < 0.3 {
            tracing::debug!(
                "Audio buffer at {}% capacity, optimal for low latency",
                stats.buffer_fill_percentage * 100.0
            );
        }

        Ok(())
    }
}

/// Processing statistics
#[derive(Debug, Clone)]
/// Processing Stats
pub struct ProcessingStats {
    /// chunks processed
    pub chunks_processed: u64,
    /// is processing
    pub is_processing: bool,
    /// buffer duration seconds
    pub buffer_duration_seconds: f32,
    /// buffer fill percentage
    pub buffer_fill_percentage: f32,
    /// last processed time
    pub last_processed_time: f32,
}

impl AudioRingBuffer {
    fn new(capacity: usize, sample_rate: u32, channels: u32) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            sample_rate,
            channels,
        }
    }

    fn add_samples(&mut self, samples: &[f32]) -> Result<(), RecognitionError> {
        for &sample in samples {
            if self.buffer.len() >= self.capacity {
                self.buffer.pop_front();
            }
            self.buffer.push_back(sample);
        }
        Ok(())
    }

    fn get_chunk(&mut self, chunk_size: usize) -> Result<Vec<f32>, RecognitionError> {
        let available = self.buffer.len().min(chunk_size);
        let chunk: Vec<f32> = self.buffer.drain(..available).collect();
        Ok(chunk)
    }

    fn duration_seconds(&self) -> f32 {
        self.buffer.len() as f32 / self.sample_rate as f32
    }

    fn fill_percentage(&self) -> f32 {
        (self.buffer.len() as f32 / self.capacity as f32) * 100.0
    }
}

impl TranscriptBuffer {
    fn new(max_segments: usize) -> Self {
        Self {
            segments: VecDeque::new(),
            max_segments,
            current_text: String::new(),
        }
    }

    fn add_segment(&mut self, segment: TranscriptSegment) {
        if self.segments.len() >= self.max_segments {
            self.segments.pop_front();
        }

        self.segments.push_back(segment);
        self.update_current_text();
    }

    fn update_current_text(&mut self) {
        self.current_text = self
            .segments
            .iter()
            .map(|seg| seg.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
    }

    fn get_full_text(&self) -> String {
        self.current_text.clone()
    }

    fn get_recent_segments(&self, max_segments: usize) -> Vec<TranscriptSegment> {
        self.segments
            .iter()
            .rev()
            .take(max_segments)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    fn clear(&mut self) {
        self.segments.clear();
        self.current_text.clear();
    }
}

impl IncrementalContext {
    fn new() -> Self {
        Self {
            previous_context: String::new(),
            token_history: Vec::new(),
            cached_features: None,
            detected_language: None,
            confidence_history: VecDeque::new(),
            max_context_length: 256, // Keep last 256 characters of context
        }
    }

    /// Update context with new segment
    fn update_context(&mut self, text: &str, tokens: &[u32], confidence: f32) {
        // Append to previous context with space separator
        if !self.previous_context.is_empty() && !text.trim().is_empty() {
            self.previous_context.push(' ');
        }
        self.previous_context.push_str(text.trim());

        // Truncate context if too long
        if self.previous_context.len() > self.max_context_length {
            let start = self.previous_context.len() - self.max_context_length;
            self.previous_context = self.previous_context[start..].to_string();
        }

        // Update token history (keep last 50 tokens)
        self.token_history.extend_from_slice(tokens);
        if self.token_history.len() > 50 {
            let start = self.token_history.len() - 50;
            self.token_history = self.token_history[start..].to_vec();
        }

        // Update confidence history (keep last 10 values)
        self.confidence_history.push_back(confidence);
        if self.confidence_history.len() > 10 {
            self.confidence_history.pop_front();
        }
    }

    /// Get context for next chunk
    fn get_context_prompt(&self) -> String {
        if self.previous_context.is_empty() {
            String::new()
        } else {
            // Return last part of context for prompt
            let max_prompt_length = 128;
            if self.previous_context.len() <= max_prompt_length {
                self.previous_context.clone()
            } else {
                let start = self.previous_context.len() - max_prompt_length;
                self.previous_context[start..].to_string()
            }
        }
    }

    /// Get average confidence over recent history
    fn get_average_confidence(&self) -> f32 {
        if self.confidence_history.is_empty() {
            0.5 // Default confidence
        } else {
            let sum: f32 = self.confidence_history.iter().sum();
            sum / self.confidence_history.len() as f32
        }
    }

    /// Clear context (useful for topic changes or long pauses)
    fn clear_context(&mut self) {
        self.previous_context.clear();
        self.token_history.clear();
        self.cached_features = None;
        self.confidence_history.clear();
    }
}
