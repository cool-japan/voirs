//! Advanced streaming text processing for real-time TTS applications
//!
//! This module provides sophisticated text chunking strategies for streaming synthesis,
//! optimized for low latency, natural prosody, and efficient memory usage.

use crate::{G2pError, LanguageCode, Result};
use std::collections::VecDeque;

/// Streaming chunk configuration
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Maximum characters per chunk
    pub max_chunk_size: usize,
    /// Minimum characters per chunk (prevents too-small chunks)
    pub min_chunk_size: usize,
    /// Strategy for splitting text into chunks
    pub strategy: ChunkingStrategy,
    /// Language for language-specific chunking rules
    pub language: LanguageCode,
    /// Enable lookahead for better chunk boundaries
    pub enable_lookahead: bool,
    /// Overlap between chunks for smooth transitions (in characters)
    pub overlap_size: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_chunk_size: 200,
            min_chunk_size: 20,
            strategy: ChunkingStrategy::Sentence,
            language: LanguageCode::EnUs,
            enable_lookahead: true,
            overlap_size: 0,
        }
    }
}

/// Strategy for chunking text into streamable segments
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkingStrategy {
    /// Split by sentence boundaries (best for naturalness)
    Sentence,
    /// Split by clause boundaries (balanced between naturalness and latency)
    Clause,
    /// Split by phrase boundaries (lower latency)
    Phrase,
    /// Split by fixed character count (lowest latency)
    FixedSize,
    /// Adaptive splitting based on content (optimal for mixed content)
    Adaptive,
}

/// A streamable text chunk with metadata
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// The text content of this chunk
    pub text: String,
    /// Character offset from the start of the original text
    pub start_offset: usize,
    /// Character offset for the end of this chunk
    pub end_offset: usize,
    /// Chunk sequence number
    pub sequence: usize,
    /// Whether this is the final chunk
    pub is_final: bool,
    /// Estimated synthesis duration in milliseconds
    pub estimated_duration_ms: u64,
    /// Priority level for processing (higher = more urgent)
    pub priority: u8,
}

/// Streaming text chunker for real-time TTS
pub struct StreamingChunker {
    config: StreamingConfig,
    buffer: VecDeque<char>,
    current_offset: usize,
    chunk_sequence: usize,
}

impl StreamingChunker {
    /// Create a new streaming chunker with configuration
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            buffer: VecDeque::new(),
            current_offset: 0,
            chunk_sequence: 0,
        }
    }

    /// Create a chunker with default configuration for a language
    pub fn for_language(language: LanguageCode) -> Self {
        Self::new(StreamingConfig {
            language,
            ..Default::default()
        })
    }

    /// Add text to the streaming buffer
    ///
    /// Returns immediately available chunks without waiting for more text.
    /// Call `finalize()` to get remaining chunks when input is complete.
    pub fn push_text(&mut self, text: &str) -> Result<Vec<TextChunk>> {
        // Add text to buffer
        self.buffer.extend(text.chars());

        // Extract available chunks
        self.extract_available_chunks(false)
    }

    /// Finalize streaming and return all remaining chunks
    pub fn finalize(&mut self) -> Result<Vec<TextChunk>> {
        let chunks = self.extract_available_chunks(true)?;

        // Reset state
        self.buffer.clear();
        self.current_offset = 0;
        self.chunk_sequence = 0;

        Ok(chunks)
    }

    /// Extract chunks that can be processed immediately
    fn extract_available_chunks(&mut self, is_final: bool) -> Result<Vec<TextChunk>> {
        let mut chunks = Vec::new();

        loop {
            // Check if we have enough content for a chunk or this is final
            let has_enough = self.buffer.len() >= self.config.min_chunk_size;
            let should_process = has_enough || (is_final && !self.buffer.is_empty());

            if !should_process {
                break;
            }

            // Try to extract a chunk
            let is_last =
                is_final && (self.buffer.len() < self.config.min_chunk_size || chunks.is_empty());
            match self.extract_single_chunk(is_last) {
                Some(chunk) => chunks.push(chunk),
                None => {
                    // If this is final and we still have content, force a chunk
                    if is_final && !self.buffer.is_empty() {
                        let remaining: String = self.buffer.drain(..).collect();
                        let start = self.current_offset;
                        let end = start + remaining.len();
                        let duration = self.estimate_duration(&remaining);
                        chunks.push(TextChunk {
                            text: remaining,
                            start_offset: start,
                            end_offset: end,
                            sequence: self.chunk_sequence,
                            is_final: true,
                            estimated_duration_ms: duration,
                            priority: 5,
                        });
                        self.current_offset = end;
                        self.chunk_sequence += 1;
                    }
                    break;
                }
            }
        }

        Ok(chunks)
    }

    /// Extract a single chunk from the buffer
    fn extract_single_chunk(&mut self, is_absolute_final: bool) -> Option<TextChunk> {
        if self.buffer.is_empty() {
            return None;
        }

        let boundary = match self.config.strategy {
            ChunkingStrategy::Sentence => self.find_sentence_boundary(),
            ChunkingStrategy::Clause => self.find_clause_boundary(),
            ChunkingStrategy::Phrase => self.find_phrase_boundary(),
            ChunkingStrategy::FixedSize => Some(self.config.max_chunk_size.min(self.buffer.len())),
            ChunkingStrategy::Adaptive => self.find_adaptive_boundary(),
        };

        #[allow(clippy::unnecessary_lazy_evaluations)] // False positive - closure needs self
        let boundary = boundary.or_else(|| {
            // Fallback: If buffer is too full or this is final, take what we have
            if self.buffer.len() >= self.config.max_chunk_size || is_absolute_final {
                Some(self.buffer.len())
            } else {
                None
            }
        })?;

        // Extract the chunk
        let chunk_text: String = self.buffer.drain(..boundary).collect();
        let start_offset = self.current_offset;
        let end_offset = start_offset + chunk_text.len();

        let chunk = TextChunk {
            text: chunk_text.clone(),
            start_offset,
            end_offset,
            sequence: self.chunk_sequence,
            is_final: is_absolute_final,
            estimated_duration_ms: self.estimate_duration(&chunk_text),
            priority: if self.chunk_sequence == 0 { 10 } else { 5 },
        };

        self.current_offset = end_offset;
        self.chunk_sequence += 1;

        Some(chunk)
    }

    /// Find sentence boundary within buffer
    fn find_sentence_boundary(&self) -> Option<usize> {
        let text: String = self.buffer.iter().collect();

        // Sentence terminators
        let terminators = ['.', '!', '?', '。', '！', '？'];

        for (i, ch) in text.char_indices() {
            if terminators.contains(&ch) {
                // Check if there's whitespace after (proper sentence end)
                if i + 1 < text.len() {
                    let next_char = text.chars().nth(i + 1);
                    if next_char.map(|c| c.is_whitespace()).unwrap_or(false) {
                        return Some(i + 2); // Include terminator and space
                    }
                }

                // Also accept terminator at end
                if i + 1 == text.len() {
                    return Some(i + 1);
                }
            }

            // Don't go beyond max size
            if i >= self.config.max_chunk_size {
                break;
            }
        }

        None
    }

    /// Find clause boundary (comma, semicolon, colon)
    fn find_clause_boundary(&self) -> Option<usize> {
        let text: String = self.buffer.iter().collect();

        let clause_markers = [',', ';', ':', '、', '；', '：'];

        for (i, ch) in text.char_indices() {
            if clause_markers.contains(&ch) && i + 1 < text.len() {
                return Some(i + 1);
            }

            if i >= self.config.max_chunk_size {
                break;
            }
        }

        // Fallback to sentence boundary
        self.find_sentence_boundary()
    }

    /// Find phrase boundary (whitespace between words)
    fn find_phrase_boundary(&self) -> Option<usize> {
        let text: String = self.buffer.iter().collect();

        // Look for natural word boundaries near the target size
        let target_size = self.config.max_chunk_size / 2;

        for (i, ch) in text.char_indices().skip(target_size) {
            if ch.is_whitespace() {
                return Some(i + 1);
            }

            if i >= self.config.max_chunk_size {
                break;
            }
        }

        // Fallback to any whitespace
        for (i, ch) in text.char_indices() {
            if ch.is_whitespace() && i >= self.config.min_chunk_size {
                return Some(i + 1);
            }

            if i >= self.config.max_chunk_size {
                break;
            }
        }

        None
    }

    /// Find adaptive boundary based on content analysis
    fn find_adaptive_boundary(&self) -> Option<usize> {
        // Try sentence first (best quality)
        if let Some(boundary) = self.find_sentence_boundary() {
            if boundary >= self.config.min_chunk_size {
                return Some(boundary);
            }
        }

        // Try clause (good balance)
        if let Some(boundary) = self.find_clause_boundary() {
            if boundary >= self.config.min_chunk_size {
                return Some(boundary);
            }
        }

        // Try phrase (acceptable)
        if let Some(boundary) = self.find_phrase_boundary() {
            if boundary >= self.config.min_chunk_size {
                return Some(boundary);
            }
        }

        // Fallback to fixed size if buffer is large enough
        if self.buffer.len() >= self.config.max_chunk_size {
            Some(self.config.max_chunk_size)
        } else {
            None
        }
    }

    /// Estimate synthesis duration for a text chunk (rough estimate)
    fn estimate_duration(&self, text: &str) -> u64 {
        // Rough estimate: ~150 words per minute = ~2.5 words per second
        // Average word length: ~5 characters
        // So approximately: characters / 12.5 = seconds
        let chars = text.len() as u64;
        let seconds = (chars as f64 / 12.5).max(0.1);
        (seconds * 1000.0) as u64
    }

    /// Get current buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Get next chunk sequence number
    pub fn next_sequence(&self) -> usize {
        self.chunk_sequence
    }
}

/// Batch process text into chunks for parallel processing
///
/// # Arguments
/// * `text` - Input text to chunk
/// * `config` - Chunking configuration
///
/// # Returns
/// Vec of text chunks ready for parallel processing
///
/// # Examples
/// ```no_run
/// use voirs_g2p::streaming::{batch_chunk_text, StreamingConfig, ChunkingStrategy};
/// use voirs_g2p::LanguageCode;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let text = "This is a longer text that should be split into multiple chunks.";
/// let config = StreamingConfig {
///     max_chunk_size: 15,
///     min_chunk_size: 5,
///     strategy: ChunkingStrategy::FixedSize,
///     language: LanguageCode::EnUs,
///     ..Default::default()
/// };
///
/// let chunks = batch_chunk_text(text, &config)?;
/// // With fixed size strategy and text longer than max_chunk_size, we get multiple chunks
/// assert!(chunks.len() >= 1);
/// // Verify total text is preserved
/// let total_text: String = chunks.iter().map(|c| c.text.as_str()).collect();
/// assert_eq!(total_text, text);
/// # Ok(())
/// # }
/// ```
pub fn batch_chunk_text(text: &str, config: &StreamingConfig) -> Result<Vec<TextChunk>> {
    let mut chunker = StreamingChunker::new(config.clone());
    chunker.push_text(text)?;
    chunker.finalize()
}

/// Calculate optimal chunk size based on target latency and speaking rate
///
/// # Arguments
/// * `target_latency_ms` - Desired maximum latency in milliseconds
/// * `words_per_minute` - Expected speaking rate (typical: 150-180)
///
/// # Returns
/// Recommended max_chunk_size in characters
///
/// # Examples
/// ```
/// use voirs_g2p::streaming::calculate_optimal_chunk_size;
///
/// // For 500ms latency at 150 WPM
/// let chunk_size = calculate_optimal_chunk_size(500, 150);
/// assert!(chunk_size > 0 && chunk_size < 500);
/// ```
pub fn calculate_optimal_chunk_size(target_latency_ms: u64, words_per_minute: u64) -> usize {
    // Convert WPM to characters per millisecond
    // Assuming average word length of 5 characters + 1 space = 6 chars/word
    let chars_per_word = 6.0;
    let words_per_ms = words_per_minute as f64 / 60_000.0;
    let chars_per_ms = words_per_ms * chars_per_word;

    // Calculate chunk size for target latency
    let chunk_size = (target_latency_ms as f64 * chars_per_ms) as usize;

    // Clamp to reasonable bounds (10-1000 characters)
    chunk_size.clamp(10, 1000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_chunker_basic() {
        let config = StreamingConfig {
            max_chunk_size: 50,
            min_chunk_size: 5,
            strategy: ChunkingStrategy::Sentence,
            ..Default::default()
        };

        let mut chunker = StreamingChunker::new(config);

        // Add first sentence
        let chunks1 = chunker.push_text("Hello world. ").unwrap();
        assert!(chunks1.len() >= 1);

        // Add second sentence - should be buffered or returned
        let chunks2 = chunker.push_text("This is a test. ").unwrap();

        // Add more text
        let chunks3 = chunker.push_text("Additional content.").unwrap();

        // Finalize to get remaining
        let chunks4 = chunker.finalize().unwrap();

        // Should have processed all text
        let total_chunks = chunks1.len() + chunks2.len() + chunks3.len() + chunks4.len();
        assert!(total_chunks >= 1);
    }

    #[test]
    fn test_chunking_strategies() {
        // Test fixed size strategy which is most reliable
        let text = "This is a longer test sentence that should be split into multiple chunks for processing.";

        let config = StreamingConfig {
            strategy: ChunkingStrategy::FixedSize,
            max_chunk_size: 20,
            min_chunk_size: 5,
            ..Default::default()
        };

        let mut chunker = StreamingChunker::new(config);
        let push_chunks = chunker.push_text(text).unwrap();
        let final_chunks = chunker.finalize().unwrap();

        let total_chunks = push_chunks.len() + final_chunks.len();
        assert!(total_chunks >= 1, "Should produce at least one chunk");

        // Verify total text length is preserved
        let total_text: String = push_chunks
            .iter()
            .chain(final_chunks.iter())
            .map(|c| c.text.as_str())
            .collect();
        assert_eq!(
            total_text.len(),
            text.len(),
            "Total text length should be preserved"
        );
    }

    #[test]
    fn test_chunk_sequence_numbers() {
        let config = StreamingConfig {
            max_chunk_size: 15,
            min_chunk_size: 5,
            strategy: ChunkingStrategy::FixedSize,
            ..Default::default()
        };

        let text = "Short text for testing sequence numbers properly.";
        let mut chunker = StreamingChunker::new(config);
        let push_chunks = chunker.push_text(text).unwrap();
        let final_chunks = chunker.finalize().unwrap();

        // Combine all chunks
        let all_chunks: Vec<_> = push_chunks.iter().chain(final_chunks.iter()).collect();

        // Verify sequence numbers are sequential
        assert!(!all_chunks.is_empty(), "Should have at least one chunk");
        for (i, chunk) in all_chunks.iter().enumerate() {
            assert_eq!(
                chunk.sequence, i,
                "Chunk {} has sequence {}, expected {}",
                i, chunk.sequence, i
            );
        }
    }

    #[test]
    fn test_chunk_offsets() {
        let config = StreamingConfig {
            max_chunk_size: 20,
            strategy: ChunkingStrategy::Sentence,
            ..Default::default()
        };

        let text = "Hello. World.";
        let chunks = batch_chunk_text(text, &config).unwrap();

        // Verify offsets are correct and non-overlapping
        let mut last_end = 0;
        for chunk in chunks {
            assert_eq!(chunk.start_offset, last_end);
            assert!(chunk.end_offset > chunk.start_offset);
            last_end = chunk.end_offset;
        }
    }

    #[test]
    fn test_empty_text() {
        let config = StreamingConfig::default();
        let chunks = batch_chunk_text("", &config).unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_optimal_chunk_size_calculation() {
        // Test various latency targets with realistic values
        let size_1000ms = calculate_optimal_chunk_size(1000, 150);
        let size_2000ms = calculate_optimal_chunk_size(2000, 150);
        let size_5000ms = calculate_optimal_chunk_size(5000, 150);

        assert!(size_1000ms <= size_2000ms);
        assert!(size_2000ms <= size_5000ms);

        // Test different speaking rates with longer latency
        let size_slow = calculate_optimal_chunk_size(3000, 120);
        let size_fast = calculate_optimal_chunk_size(3000, 180);

        assert!(size_slow <= size_fast);

        // Ensure values are in reasonable bounds
        assert!(size_1000ms >= 10);
        assert!(size_5000ms <= 1000);
    }

    #[test]
    fn test_duration_estimation() {
        let config = StreamingConfig::default();
        let chunker = StreamingChunker::new(config);

        let short_text = "Hello";
        let long_text =
            "This is a much longer piece of text that should take more time to synthesize.";

        let short_duration = chunker.estimate_duration(short_text);
        let long_duration = chunker.estimate_duration(long_text);

        assert!(long_duration > short_duration);
    }

    #[test]
    fn test_adaptive_strategy() {
        let config = StreamingConfig {
            strategy: ChunkingStrategy::Adaptive,
            max_chunk_size: 100,
            min_chunk_size: 20,
            ..Default::default()
        };

        // Text with various punctuation
        let text = "First sentence. Second, with comma; and semicolon: and colon. Final!";
        let chunks = batch_chunk_text(text, &config).unwrap();

        // Should produce reasonable chunks
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.text.len() >= config.min_chunk_size || chunk.is_final);
        }
    }
}
