//! Batch processing implementation for Whisper models
//!
//! This module provides efficient batch processing capabilities for multiple
//! audio files with optimized memory usage and parallel processing.

use super::{
    WhisperAudioProcessor, WhisperConfig, WhisperDecoder, WhisperEncoder, WhisperTokenizer,
};
use crate::traits::{SentenceBoundary, Transcript, WordTimestamp};
use crate::RecognitionError;
use candle_core::Device;
use futures::stream::{FuturesUnordered, StreamExt};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use voirs_sdk::{AudioBuffer, LanguageCode};

/// Configuration for batch processing
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum concurrent processing
    pub max_concurrent: usize,
    /// Use dynamic batching
    pub dynamic_batching: bool,
    /// Memory limit in MB
    pub memory_limit_mb: f32,
    /// Enable parallel processing
    pub parallel_processing: bool,
    /// Timeout per batch (seconds)
    pub timeout_seconds: u64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_concurrent: 4,
            dynamic_batching: true,
            memory_limit_mb: 2048.0,
            parallel_processing: true,
            timeout_seconds: 300, // 5 minutes
        }
    }
}

/// Batch processing statistics
#[derive(Debug, Clone)]
pub struct BatchStats {
    /// Total batches processed
    pub batches_processed: usize,
    /// Total audio files processed
    pub files_processed: usize,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Average batch processing time
    pub average_batch_time: Duration,
    /// Average Real-Time Factor
    pub average_rtf: f32,
    /// Peak memory usage (MB)
    pub peak_memory_mb: f32,
    /// Throughput (files per second)
    pub throughput_fps: f32,
}

impl Default for BatchStats {
    fn default() -> Self {
        Self {
            batches_processed: 0,
            files_processed: 0,
            total_processing_time: Duration::ZERO,
            average_batch_time: Duration::ZERO,
            average_rtf: 0.0,
            peak_memory_mb: 0.0,
            throughput_fps: 0.0,
        }
    }
}

/// Batch input for processing
#[derive(Debug, Clone)]
pub struct BatchInput {
    /// Audio buffer
    pub audio: AudioBuffer,
    /// Optional language hint
    pub language: Option<LanguageCode>,
    /// Input identifier
    pub id: String,
}

/// Batch output result
#[derive(Debug)]
pub struct BatchOutput {
    /// Input identifier
    pub id: String,
    /// Transcription result
    pub transcript: Result<Transcript, RecognitionError>,
    /// Processing time for this item
    pub processing_time: Duration,
}

/// Whisper batch processor
pub struct WhisperBatchProcessor {
    /// Whisper encoder
    encoder: Arc<WhisperEncoder>,
    /// Whisper decoder
    decoder: Arc<WhisperDecoder>,
    /// Tokenizer
    tokenizer: Arc<WhisperTokenizer>,
    /// Audio processor
    audio_processor: Arc<WhisperAudioProcessor>,
    /// Configuration
    config: WhisperConfig,
    /// Batch configuration
    batch_config: BatchConfig,
    /// Device
    device: Device,
    /// Statistics
    stats: Arc<RwLock<BatchStats>>,
    /// Semaphore for concurrency control
    semaphore: Arc<Semaphore>,
}

impl WhisperBatchProcessor {
    /// Create a new batch processor
    ///
    /// # Errors
    ///
    /// Returns an error if the Whisper model components (encoder, decoder, tokenizer) fail to initialize.
    pub async fn new(
        config: WhisperConfig,
        batch_config: BatchConfig,
        device: Device,
    ) -> Result<Self, RecognitionError> {
        let encoder = Arc::new(WhisperEncoder::new(&config, &device).await?);
        let decoder = Arc::new(WhisperDecoder::new(&config, &device).await?);
        let tokenizer = Arc::new(WhisperTokenizer::new().await?);
        let audio_processor = Arc::new(WhisperAudioProcessor::new(&config, &device)?);

        let semaphore = Arc::new(Semaphore::new(batch_config.max_concurrent));
        let stats = Arc::new(RwLock::new(BatchStats::default()));

        Ok(Self {
            encoder,
            decoder,
            tokenizer,
            audio_processor,
            config,
            batch_config,
            device,
            stats,
            semaphore,
        })
    }

    /// Process a batch of audio inputs
    pub async fn process_batch(&self, inputs: Vec<BatchInput>) -> Vec<BatchOutput> {
        let start_time = Instant::now();

        if inputs.is_empty() {
            return Vec::new();
        }

        // Determine optimal batch size
        let batch_size = self.determine_optimal_batch_size(&inputs).await;

        // Process in chunks if needed
        let mut all_outputs = Vec::new();

        for chunk in inputs.chunks(batch_size) {
            let chunk_outputs = if self.batch_config.parallel_processing {
                self.process_batch_parallel(chunk.to_vec()).await
            } else {
                self.process_batch_sequential(chunk.to_vec()).await
            };

            all_outputs.extend(chunk_outputs);
        }

        // Update statistics
        self.update_batch_stats(inputs.len(), start_time.elapsed())
            .await;

        all_outputs
    }

    /// Process batch in parallel
    async fn process_batch_parallel(&self, inputs: Vec<BatchInput>) -> Vec<BatchOutput> {
        let mut futures = FuturesUnordered::new();

        for input in inputs {
            let permit = self.semaphore.clone().acquire_owned().await.unwrap();
            let processor = self.clone();

            futures.push(tokio::spawn(async move {
                let _permit = permit; // Keep permit alive
                processor.process_single_item(input).await
            }));
        }

        let mut outputs = Vec::new();
        while let Some(result) = futures.next().await {
            match result {
                Ok(output) => outputs.push(output),
                Err(e) => {
                    tracing::error!("Batch processing task failed: {}", e);
                }
            }
        }

        outputs
    }

    /// Process batch sequentially
    async fn process_batch_sequential(&self, inputs: Vec<BatchInput>) -> Vec<BatchOutput> {
        let mut outputs = Vec::new();

        for input in inputs {
            let output = self.process_single_item(input).await;
            outputs.push(output);
        }

        outputs
    }

    /// Process a single item
    async fn process_single_item(&self, input: BatchInput) -> BatchOutput {
        let start_time = Instant::now();
        let id = input.id.clone();

        let result = self.transcribe_single(&input.audio, input.language).await;
        let processing_time = start_time.elapsed();

        BatchOutput {
            id,
            transcript: result,
            processing_time,
        }
    }

    /// Transcribe a single audio buffer
    async fn transcribe_single(
        &self,
        audio: &AudioBuffer,
        language: Option<LanguageCode>,
    ) -> Result<Transcript, RecognitionError> {
        // Process audio to mel spectrogram
        let mel_features = self.audio_processor.process_audio(audio)?;

        // Encode audio features with batch dimension
        let audio_features = self.encoder.forward(&mel_features)?;

        // Generate tokens
        let start_token = self.tokenizer.special_tokens().sot;
        let end_token = self.tokenizer.special_tokens().eot;

        let tokens = self
            .decoder
            .generate_tokens(
                &audio_features,
                start_token,
                end_token,
                448, // max tokens
                5,   // beam size
                1.0, // temperature
            )
            .await?;

        // Decode tokens to text
        let text = self.tokenizer.decode(&tokens)?;

        // Extract word timestamps from tokens
        let word_timestamps = self.extract_word_timestamps(&tokens, &text).await?;

        // Extract sentence boundaries from text and timestamps
        let sentence_boundaries = Self::extract_sentence_boundaries(&text, &word_timestamps);

        // Create transcript with metadata
        Ok(Transcript {
            text,
            language: language.unwrap_or(LanguageCode::EnUs),
            confidence: 0.9, // Placeholder confidence
            word_timestamps,
            sentence_boundaries,
            processing_duration: None,
        })
    }

    /// Determine optimal batch size based on available memory and input characteristics
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    async fn determine_optimal_batch_size(&self, inputs: &[BatchInput]) -> usize {
        if !self.batch_config.dynamic_batching {
            return self.batch_config.max_batch_size;
        }

        // Estimate memory usage per input
        let avg_duration = inputs
            .iter()
            .map(|input| input.audio.duration())
            .sum::<f32>()
            / inputs.len() as f32;

        // Rough estimate: 1MB per second of audio
        let estimated_memory_per_input = avg_duration * 1.0; // MB

        let max_by_memory =
            (self.batch_config.memory_limit_mb / estimated_memory_per_input) as usize;

        std::cmp::min(
            self.batch_config.max_batch_size,
            std::cmp::max(1, max_by_memory),
        )
    }

    /// Update batch processing statistics
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    async fn update_batch_stats(&self, processed_count: usize, batch_time: Duration) {
        let mut stats = self.stats.write().await;

        stats.batches_processed += 1;
        stats.files_processed += processed_count;
        stats.total_processing_time += batch_time;
        stats.average_batch_time =
            stats.total_processing_time / stats.batches_processed.max(1) as u32;

        // Calculate throughput
        if stats.total_processing_time.as_secs() > 0 {
            stats.throughput_fps =
                stats.files_processed as f32 / stats.total_processing_time.as_secs_f32();
        }

        // Update peak memory (placeholder - would need actual memory measurement)
        stats.peak_memory_mb = stats.peak_memory_mb.max(512.0);

        // Calculate average RTF (placeholder)
        stats.average_rtf = 0.8; // Placeholder RTF value
    }

    /// Get current batch processing statistics
    pub async fn get_stats(&self) -> BatchStats {
        self.stats.read().await.clone()
    }

    /// Reset statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = BatchStats::default();
    }

    /// Process audio files from filesystem
    pub async fn process_files(&self, file_paths: Vec<String>) -> Vec<BatchOutput> {
        let mut batch_inputs = Vec::new();

        for (idx, path) in file_paths.iter().enumerate() {
            // Load audio file (placeholder - would need actual file loading)
            match self.load_audio_file(path).await {
                Ok(audio) => {
                    batch_inputs.push(BatchInput {
                        audio,
                        language: None,
                        id: format!("file_{idx}"),
                    });
                }
                Err(e) => {
                    tracing::error!("Failed to load audio file {}: {}", path, e);
                }
            }
        }

        self.process_batch(batch_inputs).await
    }

    /// Load audio file (placeholder implementation)
    async fn load_audio_file(&self, _path: &str) -> Result<AudioBuffer, RecognitionError> {
        // Placeholder implementation - would need actual file loading
        Ok(AudioBuffer::new(vec![0.0; 16000], 16000, 1))
    }

    /// Extract word timestamps from tokens and align with text
    #[allow(clippy::cast_precision_loss)]
    async fn extract_word_timestamps(
        &self,
        tokens: &[u32],
        text: &str,
    ) -> Result<Vec<WordTimestamp>, RecognitionError> {
        let mut word_timestamps = Vec::new();
        let mut current_time = 0.0;
        let words: Vec<&str> = text.split_whitespace().collect();

        // Track timestamps from tokens
        let mut timestamp_queue = Vec::new();
        for &token in tokens {
            if self.tokenizer.is_timestamp_token(token) {
                if let Some(time) = self.tokenizer.timestamp_to_seconds(token) {
                    timestamp_queue.push(time);
                }
            }
        }

        // If we have no timestamps, estimate based on audio duration
        if timestamp_queue.is_empty() && !words.is_empty() {
            // Estimate 2 words per second (rough approximation)
            let estimated_duration = words.len() as f32 * 0.5;
            for (i, word) in words.iter().enumerate() {
                let start_time = (i as f32 * estimated_duration) / words.len() as f32;
                let end_time = ((i + 1) as f32 * estimated_duration) / words.len() as f32;

                word_timestamps.push(WordTimestamp {
                    word: (*word).to_string(),
                    start_time,
                    end_time,
                    confidence: 0.8, // Estimated confidence
                });
            }
            return Ok(word_timestamps);
        }

        // Process timestamps and align with words
        let mut timestamp_iter = timestamp_queue.iter().peekable();

        for word in words {
            let word_start_time = current_time;

            // Look for next timestamp
            if let Some(&&next_timestamp) = timestamp_iter.peek() {
                current_time = next_timestamp;
                timestamp_iter.next();
            } else {
                // Estimate word duration if no more timestamps
                current_time += 0.5; // 500ms per word estimate
            }

            word_timestamps.push(WordTimestamp {
                word: word.to_string(),
                start_time: word_start_time,
                end_time: current_time,
                confidence: 0.9, // High confidence for timestamp-aligned words
            });
        }

        Ok(word_timestamps)
    }

    /// Extract sentence boundaries from text and word timestamps
    fn extract_sentence_boundaries(
        text: &str,
        word_timestamps: &[WordTimestamp],
    ) -> Vec<SentenceBoundary> {
        let mut sentence_boundaries = Vec::new();

        // Split text into sentences using basic punctuation
        let sentence_endings = ['.', '!', '?'];
        let mut current_sentence = String::new();
        let mut sentence_start_time = 0.0;
        for (word_index, word) in text.split_whitespace().enumerate() {
            // Get timing for this word
            let word_timestamp = word_timestamps.get(word_index);

            if current_sentence.is_empty() {
                // Start of new sentence
                sentence_start_time = word_timestamp.map_or(0.0, |wt| wt.start_time);
            }

            if !current_sentence.is_empty() {
                current_sentence.push(' ');
            }
            current_sentence.push_str(word);

            // Check if this word ends a sentence
            let ends_sentence = word
                .chars()
                .last()
                .is_some_and(|ch| sentence_endings.contains(&ch));

            if ends_sentence {
                let sentence_end_time =
                    word_timestamp.map_or(sentence_start_time + 1.0, |wt| wt.end_time);

                sentence_boundaries.push(SentenceBoundary {
                    start_time: sentence_start_time,
                    end_time: sentence_end_time,
                    text: current_sentence.clone(),
                    confidence: 0.85, // Reasonable confidence for sentence detection
                });

                current_sentence.clear();
            }
        }

        // Handle any remaining text as a sentence
        if !current_sentence.is_empty() {
            let final_end_time = word_timestamps
                .last()
                .map_or(sentence_start_time + 1.0, |wt| wt.end_time);

            sentence_boundaries.push(SentenceBoundary {
                start_time: sentence_start_time,
                end_time: final_end_time,
                text: current_sentence,
                confidence: 0.85,
            });
        }

        sentence_boundaries
    }
}

impl Clone for WhisperBatchProcessor {
    fn clone(&self) -> Self {
        Self {
            encoder: self.encoder.clone(),
            decoder: self.decoder.clone(),
            tokenizer: self.tokenizer.clone(),
            audio_processor: self.audio_processor.clone(),
            config: self.config.clone(),
            batch_config: self.batch_config.clone(),
            device: self.device.clone(),
            stats: self.stats.clone(),
            semaphore: self.semaphore.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voirs_sdk::AudioBuffer;

    #[tokio::test]
    async fn test_batch_processor_creation() {
        let config = WhisperConfig::default();
        let batch_config = BatchConfig::default();
        let device = Device::Cpu;

        let processor = WhisperBatchProcessor::new(config, batch_config, device).await;
        assert!(processor.is_ok());
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let config = WhisperConfig::default();
        let batch_config = BatchConfig {
            max_batch_size: 2,
            max_concurrent: 1,
            ..Default::default()
        };
        let device = Device::Cpu;

        let processor = WhisperBatchProcessor::new(config, batch_config, device)
            .await
            .unwrap();

        let inputs = vec![
            BatchInput {
                audio: AudioBuffer::new(vec![0.1; 16000], 16000, 1),
                language: Some(LanguageCode::EnUs),
                id: "test1".to_string(),
            },
            BatchInput {
                audio: AudioBuffer::new(vec![0.2; 16000], 16000, 1),
                language: Some(LanguageCode::EnUs),
                id: "test2".to_string(),
            },
        ];

        let outputs = processor.process_batch(inputs).await;
        assert_eq!(outputs.len(), 2);

        let stats = processor.get_stats().await;
        assert_eq!(stats.files_processed, 2);
        assert_eq!(stats.batches_processed, 1);
    }

    #[tokio::test]
    async fn test_optimal_batch_size_calculation() {
        let config = WhisperConfig::default();
        let batch_config = BatchConfig::default();
        let device = Device::Cpu;

        let processor = WhisperBatchProcessor::new(config, batch_config, device)
            .await
            .unwrap();

        let inputs = vec![
            BatchInput {
                audio: AudioBuffer::new(vec![0.1; 16000], 16000, 1), // 1 second
                language: None,
                id: "test1".to_string(),
            };
            10
        ];

        let batch_size = processor.determine_optimal_batch_size(&inputs).await;
        assert!(batch_size > 0);
        assert!(batch_size <= processor.batch_config.max_batch_size);
    }
}
