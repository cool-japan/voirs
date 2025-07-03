//! Core streaming pipeline functionality and chunk processing.

use crate::{
    audio::AudioBuffer,
    error::Result,
    traits::{AcousticModel, G2p, Vocoder},
    types::{LanguageCode, MelSpectrogram, Phoneme, SynthesisConfig},
    VoirsError,
};
use super::{
    management::{StreamingConfig, StreamingState, AudioChunk, ChunkMetadata},
    realtime::RealtimeProcessor,
};
use futures::{stream::BoxStream, Stream, StreamExt, TryStreamExt};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    sync::{mpsc, RwLock},
    time::sleep,
};

/// Streaming synthesis pipeline for real-time processing
pub struct StreamingPipeline {
    /// G2P component
    pub(super) g2p: Arc<dyn G2p>,
    
    /// Acoustic model component  
    pub(super) acoustic: Arc<dyn AcousticModel>,
    
    /// Vocoder component
    pub(super) vocoder: Arc<dyn Vocoder>,
    
    /// Streaming configuration
    pub(super) config: StreamingConfig,
    
    /// Current synthesis state
    pub(super) state: Arc<RwLock<StreamingState>>,
}

impl StreamingPipeline {
    /// Create new streaming pipeline
    pub fn new(
        g2p: Arc<dyn G2p>,
        acoustic: Arc<dyn AcousticModel>,
        vocoder: Arc<dyn Vocoder>,
        config: StreamingConfig,
    ) -> Self {
        Self {
            g2p,
            acoustic,
            vocoder,
            config,
            state: Arc::new(RwLock::new(StreamingState::default())),
        }
    }

    /// Start streaming synthesis for long text
    pub async fn synthesize_stream(
        &self,
        text: &str,
    ) -> Result<impl Stream<Item = Result<AudioChunk>> + Send + Unpin> {
        self.synthesize_stream_with_config(text, &SynthesisConfig::default()).await
    }

    /// Start streaming synthesis with custom configuration
    pub async fn synthesize_stream_with_config(
        &self,
        text: &str,
        synthesis_config: &SynthesisConfig,
    ) -> Result<impl Stream<Item = Result<AudioChunk>> + Send + Unpin> {
        tracing::info!("Starting streaming synthesis for {} characters", text.len());

        // Update state
        {
            let mut state = self.state.write().await;
            state.reset_for_new_synthesis();
        }

        // Split text into chunks
        let text_chunks = self.split_text_for_streaming(text);
        
        // Create processing pipeline
        let stream = futures::stream::iter(text_chunks)
            .enumerate()
            .map({
                let g2p = Arc::clone(&self.g2p);
                let acoustic = Arc::clone(&self.acoustic);
                let vocoder = Arc::clone(&self.vocoder);
                let config = synthesis_config.clone();
                let streaming_config = self.config.clone();
                let state = Arc::clone(&self.state);
                
                move |(chunk_id, text_chunk)| {
                    let g2p = Arc::clone(&g2p);
                    let acoustic = Arc::clone(&acoustic);
                    let vocoder = Arc::clone(&vocoder);
                    let config = config.clone();
                    let streaming_config = streaming_config.clone();
                    let state = Arc::clone(&state);
                    
                    async move {
                        let result = Self::process_text_chunk(
                            chunk_id,
                            text_chunk,
                            g2p,
                            acoustic,
                            vocoder,
                            &config,
                            &streaming_config,
                        ).await;
                        
                        // Update state after processing
                        if let Ok(ref chunk) = result {
                            let mut state_guard = state.write().await;
                            state_guard.update_with_chunk(chunk);
                        }
                        
                        result
                    }
                }
            })
            .buffer_unordered(self.config.max_concurrent_chunks)
            .map(|result| {
                match result {
                    Ok(chunk) => Ok(chunk),
                    Err(e) => Err(e),
                }
            });

        Ok(Box::pin(stream))
    }

    /// Start real-time synthesis with minimal latency
    pub async fn synthesize_realtime(
        &self,
        text_stream: impl Stream<Item = String> + Send + Unpin + 'static,
    ) -> Result<impl Stream<Item = Result<AudioChunk>> + Send + Unpin> {
        let (tx, rx) = mpsc::channel(self.config.realtime_buffer_size);

        // Spawn background processor
        let processor = RealtimeProcessor::new(
            Arc::clone(&self.g2p),
            Arc::clone(&self.acoustic),
            Arc::clone(&self.vocoder),
            self.config.clone(),
        );

        tokio::spawn(async move {
            if let Err(e) = processor.process_stream(text_stream, tx).await {
                tracing::error!("Real-time processing error: {}", e);
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx)
            .map(Ok);

        Ok(Box::pin(stream))
    }

    /// Process a single text chunk
    pub(super) async fn process_text_chunk(
        chunk_id: usize,
        text: String,
        g2p: Arc<dyn G2p>,
        acoustic: Arc<dyn AcousticModel>,
        vocoder: Arc<dyn Vocoder>,
        synthesis_config: &SynthesisConfig,
        streaming_config: &StreamingConfig,
    ) -> Result<AudioChunk> {
        let start_time = Instant::now();
        
        tracing::debug!("Processing chunk {}: '{}'", chunk_id, text);

        // Step 1: Text to phonemes
        let phonemes = g2p.to_phonemes(&text, None).await
            .map_err(|e| VoirsError::synthesis_failed(&text, e))?;

        // Step 2: Phonemes to mel spectrogram
        let mel = acoustic.synthesize(&phonemes, Some(synthesis_config)).await
            .map_err(|e| VoirsError::synthesis_failed(&text, e))?;

        // Step 3: Apply overlap-add windowing for smooth concatenation
        let windowed_mel = Self::apply_windowing(&mel, streaming_config);

        // Step 4: Mel to audio
        let audio = vocoder.vocode(&windowed_mel, Some(synthesis_config)).await
            .map_err(|e| VoirsError::synthesis_failed(&text, e))?;

        let processing_time = start_time.elapsed();
        
        let chunk = AudioChunk {
            chunk_id,
            audio,
            text: text.clone(),
            processing_time,
            metadata: ChunkMetadata {
                phoneme_count: phonemes.len(),
                mel_frames: mel.n_frames as usize,
                is_sentence_boundary: text.trim_end().ends_with(['.', '!', '?']),
                is_paragraph_boundary: text.trim_end().ends_with('\n'),
                real_time_factor: None, // Will be calculated by chunk
                confidence_score: 1.0, // TODO: Calculate actual confidence
            },
        };

        tracing::debug!(
            "Chunk {} processed in {:.2}ms: {:.2}s audio",
            chunk_id,
            processing_time.as_millis(),
            chunk.audio.duration()
        );

        Ok(chunk)
    }

    /// Apply overlap-add windowing for smooth concatenation
    pub(super) fn apply_windowing(mel: &MelSpectrogram, config: &StreamingConfig) -> MelSpectrogram {
        if config.overlap_frames == 0 {
            return mel.clone();
        }

        // Create windowed version of mel spectrogram
        let mut windowed_data = mel.data.clone();
        let overlap_frames = config.overlap_frames.min(mel.n_frames as usize);
        
        if overlap_frames > 0 {
            // Apply fade-in to beginning frames
            for frame_idx in 0..overlap_frames {
                let fade_factor = frame_idx as f32 / overlap_frames as f32;
                for mel_idx in 0..windowed_data.len() {
                    windowed_data[mel_idx][frame_idx] *= fade_factor;
                }
            }
            
            // Apply fade-out to ending frames
            let total_frames = mel.n_frames as usize;
            let fade_start = total_frames.saturating_sub(overlap_frames);
            for frame_idx in fade_start..total_frames {
                let fade_factor = (total_frames - frame_idx) as f32 / overlap_frames as f32;
                for mel_idx in 0..windowed_data.len() {
                    windowed_data[mel_idx][frame_idx] *= fade_factor;
                }
            }
        }

        MelSpectrogram::new(windowed_data, mel.sample_rate, mel.hop_length)
    }

    /// Split text into chunks suitable for streaming
    pub fn split_text_for_streaming(&self, text: &str) -> Vec<String> {
        let max_chunk_size = self.config.max_chunk_chars;
        let mut chunks = Vec::new();
        
        // Split by sentences first for better naturalness
        let sentences = self.split_into_sentences(text);
        
        // If we have multiple substantial sentences, split them for better streaming
        // even if the total length is under max_chunk_size
        if sentences.len() > 1 && sentences.iter().any(|s| s.trim().len() > 20) {
            for sentence in sentences {
                let trimmed = sentence.trim();
                if !trimmed.is_empty() {
                    // If this single sentence is too long, split it further
                    if trimmed.len() > max_chunk_size {
                        let sub_chunks = self.split_long_text_intelligently(trimmed, max_chunk_size);
                        chunks.extend(sub_chunks);
                    } else {
                        chunks.push(trimmed.to_string());
                    }
                }
            }
        } 
        // Handle single very long sentence that should be split
        else if sentences.len() == 1 && sentences[0].trim().len() > max_chunk_size {
            let sub_chunks = self.split_long_text_intelligently(sentences[0].trim(), max_chunk_size);
            chunks.extend(sub_chunks);
        } else {
            // For single sentence or very short sentences, use original logic
            let mut current_chunk = String::new();
            
            for sentence in sentences {
                // If adding this sentence would exceed chunk size and we have content
                if !current_chunk.is_empty() && 
                   current_chunk.len() + sentence.len() > max_chunk_size {
                    chunks.push(current_chunk.trim().to_string());
                    current_chunk = String::new();
                }
                
                current_chunk.push_str(&sentence);
                current_chunk.push(' ');
                
                // If this single sentence is too long, split it by phrases/words
                if current_chunk.len() > max_chunk_size {
                    let sub_chunks = self.split_long_text_intelligently(&current_chunk, max_chunk_size);
                    chunks.extend(sub_chunks);
                    current_chunk = String::new();
                }
            }
            
            // Add remaining text
            if !current_chunk.trim().is_empty() {
                chunks.push(current_chunk.trim().to_string());
            }
        }
        
        // Ensure no empty chunks
        chunks.retain(|chunk| !chunk.trim().is_empty());
        
        chunks
    }

    /// Split text into sentences with improved detection
    pub fn split_into_sentences(&self, text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();
        let mut in_quotes = false;
        let mut prev_char = ' ';
        
        for ch in text.chars() {
            current.push(ch);
            
            // Track quote state
            if ch == '"' || ch == '\'' {
                in_quotes = !in_quotes;
            }
            
            // Check for sentence endings
            if !in_quotes && matches!(ch, '.' | '!' | '?') {
                // Look ahead to avoid splitting on abbreviations
                let next_chars: String = text.chars()
                    .skip_while(|&c| c != ch)
                    .skip(1)
                    .take(2)
                    .collect();
                
                // Only split if followed by whitespace and capital letter
                if next_chars.starts_with(' ') && 
                   next_chars.chars().nth(1).map_or(false, |c| c.is_uppercase()) {
                    sentences.push(current.trim().to_string());
                    current = String::new();
                }
            }
            
            prev_char = ch;
        }
        
        if !current.trim().is_empty() {
            sentences.push(current.trim().to_string());
        }
        
        sentences
    }

    /// Split long text intelligently by phrases and clauses
    fn split_long_text_intelligently(&self, text: &str, max_size: usize) -> Vec<String> {
        // First try splitting by phrases (commas, semicolons)
        let phrase_chunks = self.split_by_phrases(text, max_size);
        
        if phrase_chunks.iter().all(|chunk| chunk.len() <= max_size) {
            return phrase_chunks;
        }
        
        // Fall back to word-based splitting for very long phrases
        self.split_long_text_by_words(text, max_size)
    }

    /// Split text by phrases (commas, semicolons, etc.)
    fn split_by_phrases(&self, text: &str, max_size: usize) -> Vec<String> {
        let phrase_separators = [',', ';', ':', '-', '—'];
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut words = text.split_whitespace();
        
        for word in words {
            // Check if adding this word would exceed max size
            if !current_chunk.is_empty() && 
               current_chunk.len() + word.len() + 1 > max_size {
                chunks.push(current_chunk.trim().to_string());
                current_chunk = String::new();
            }
            
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(word);
            
            // Check if word ends with phrase separator
            if word.chars().last().map_or(false, |c| phrase_separators.contains(&c)) {
                chunks.push(current_chunk.trim().to_string());
                current_chunk = String::new();
            }
        }
        
        if !current_chunk.trim().is_empty() {
            chunks.push(current_chunk.trim().to_string());
        }
        
        chunks
    }

    /// Split long text by words as fallback
    fn split_long_text_by_words(&self, text: &str, max_size: usize) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        
        for word in words {
            if !current_chunk.is_empty() && 
               current_chunk.len() + word.len() + 1 > max_size {
                chunks.push(current_chunk.trim().to_string());
                current_chunk = String::new();
            }
            
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(word);
        }
        
        if !current_chunk.trim().is_empty() {
            chunks.push(current_chunk.trim().to_string());
        }
        
        chunks
    }

    /// Get current pipeline state
    pub async fn get_state(&self) -> StreamingState {
        self.state.read().await.clone()
    }

    /// Reset pipeline state
    pub async fn reset_state(&self) {
        let mut state = self.state.write().await;
        *state = StreamingState::default();
    }

    /// Estimate processing time for given text
    pub fn estimate_processing_time(&self, text: &str) -> Duration {
        let chunks = self.split_text_for_streaming(text);
        let avg_chunk_time = Duration::from_millis(200); // Rough estimate
        Duration::from_millis(avg_chunk_time.as_millis() as u64 * chunks.len() as u64)
    }

    /// Calculate optimal chunk size for target latency
    pub fn calculate_optimal_chunk_size(&self, target_latency: Duration) -> usize {
        // Rough estimate: 1 character takes about 10ms to process
        let chars_per_ms = 0.1;
        let target_ms = target_latency.as_millis() as f32;
        let optimal_chars = (target_ms * chars_per_ms) as usize;
        
        // Clamp to reasonable bounds
        optimal_chars.clamp(self.config.min_chunk_chars, self.config.max_chunk_chars)
    }

    /// Get components for external access
    pub fn components(&self) -> (&Arc<dyn G2p>, &Arc<dyn AcousticModel>, &Arc<dyn Vocoder>) {
        (&self.g2p, &self.acoustic, &self.vocoder)
    }

    /// Get streaming configuration
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }

    /// Update streaming configuration
    pub fn update_config(&mut self, new_config: StreamingConfig) {
        self.config = new_config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{DummyAcoustic, DummyG2p, DummyVocoder};

    fn create_test_pipeline() -> StreamingPipeline {
        StreamingPipeline::new(
            Arc::new(DummyG2p::new()),
            Arc::new(DummyAcoustic::new()),
            Arc::new(DummyVocoder::new()),
            StreamingConfig::default(),
        )
    }

    #[tokio::test]
    async fn test_text_chunking() {
        let pipeline = create_test_pipeline();

        let text = "This is the first sentence. This is the second sentence! And this is a question?";
        let chunks = pipeline.split_text_for_streaming(text);
        
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(!chunk.trim().is_empty());
            assert!(chunk.len() <= 200); // Default max_chunk_chars
        }
    }

    #[tokio::test]
    async fn test_sentence_splitting() {
        let pipeline = create_test_pipeline();
        
        let text = "Hello world. This is a test! How are you? I'm fine.";
        let sentences = pipeline.split_into_sentences(text);
        
        assert_eq!(sentences.len(), 4);
        assert_eq!(sentences[0], "Hello world.");
        assert_eq!(sentences[1], "This is a test!");
        assert_eq!(sentences[2], "How are you?");
        assert_eq!(sentences[3], "I'm fine.");
    }

    #[tokio::test]
    async fn test_phrase_splitting() {
        let pipeline = create_test_pipeline();
        
        let long_text = "This is a very long sentence with many clauses, separated by commas; and some semicolons: and even colons - plus some dashes.";
        let chunks = pipeline.split_by_phrases(long_text, 50);
        
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(!chunk.trim().is_empty());
        }
    }

    #[tokio::test]
    async fn test_chunk_processing() {
        let result = StreamingPipeline::process_text_chunk(
            0,
            "Hello, world!".to_string(),
            Arc::new(DummyG2p::new()),
            Arc::new(DummyAcoustic::new()),
            Arc::new(DummyVocoder::new()),
            &SynthesisConfig::default(),
            &StreamingConfig::default(),
        ).await;

        assert!(result.is_ok());
        let chunk = result.unwrap();
        assert_eq!(chunk.chunk_id, 0);
        assert_eq!(chunk.text, "Hello, world!");
        assert!(chunk.audio.duration() > 0.0);
        assert!(chunk.metadata.phoneme_count > 0);
    }

    #[tokio::test]
    async fn test_windowing() {
        let mel_data = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 3.0, 4.0, 5.0, 6.0],
        ];
        let mel = MelSpectrogram::new(mel_data, 22050, 512);
        
        let config = StreamingConfig {
            overlap_frames: 2,
            ..Default::default()
        };
        
        let windowed = StreamingPipeline::apply_windowing(&mel, &config);
        
        // Check that fade-in and fade-out were applied
        assert!(windowed.data[0][0] < mel.data[0][0]); // First frame should be faded
        assert!(windowed.data[0][4] < mel.data[0][4]); // Last frame should be faded
    }

    #[tokio::test]
    async fn test_streaming_synthesis() {
        let pipeline = create_test_pipeline();

        let text = "This is a longer text that should be split into multiple chunks for streaming synthesis.";
        let mut stream = pipeline.synthesize_stream(text).await.unwrap();

        let mut chunk_count = 0;
        while let Some(result) = stream.next().await {
            assert!(result.is_ok());
            let chunk = result.unwrap();
            assert!(chunk.audio.duration() > 0.0);
            assert!(!chunk.text.is_empty());
            chunk_count += 1;
        }

        assert!(chunk_count > 0);
    }

    #[tokio::test]
    async fn test_state_management() {
        let pipeline = create_test_pipeline();
        
        // Initial state should be default
        let state = pipeline.get_state().await;
        assert_eq!(state.chunks_processed, 0);
        
        // Reset state
        pipeline.reset_state().await;
        let state = pipeline.get_state().await;
        assert_eq!(state.chunks_processed, 0);
    }

    #[test]
    fn test_processing_time_estimation() {
        let pipeline = create_test_pipeline();
        
        let short_text = "Hello";
        let long_text = "This is a much longer text that should take more time to process.";
        
        let short_time = pipeline.estimate_processing_time(short_text);
        let long_time = pipeline.estimate_processing_time(long_text);
        
        assert!(long_time > short_time);
    }

    #[test]
    fn test_optimal_chunk_size_calculation() {
        let pipeline = create_test_pipeline();
        
        let target_latency = Duration::from_millis(100);
        let chunk_size = pipeline.calculate_optimal_chunk_size(target_latency);
        
        assert!(chunk_size >= pipeline.config.min_chunk_chars);
        assert!(chunk_size <= pipeline.config.max_chunk_chars);
    }

    #[test]
    fn test_complex_sentence_splitting() {
        let pipeline = create_test_pipeline();
        
        // Test with abbreviations and quotes
        let text = "Dr. Smith said \"Hello world.\" He continued, \"How are you?\" It was 3 p.m.";
        let sentences = pipeline.split_into_sentences(text);
        
        // Should handle abbreviations and quotes correctly
        assert!(sentences.len() >= 2);
    }
}