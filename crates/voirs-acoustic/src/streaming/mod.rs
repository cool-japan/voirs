//! Streaming synthesis infrastructure for real-time text-to-speech
//!
//! This module provides comprehensive chunk-based processing capabilities for low-latency
//! real-time synthesis with overlap-add windowing, adaptive buffering, predictive synthesis,
//! voice activity detection, and multi-threaded processing for optimal performance.

pub mod buffer;
pub mod latency;

// Re-export latency types
pub use latency::{
    LatencyOptimizer, LatencyOptimizerConfig, LatencyStats, LatencyStrategy,
    PerformanceMeasurement, PerformancePredictor,
};

use crate::Phoneme;
use crate::{AcousticError, AcousticModel, MelSpectrogram, Result, SynthesisConfig};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Type alias for output callback function
type OutputCallback = Box<dyn Fn(&[f32]) + Send + Sync>;

/// Configuration for streaming synthesis
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Chunk size in frames for processing
    pub chunk_frames: usize,
    /// Overlap between chunks in frames
    pub overlap_frames: usize,
    /// Maximum latency in milliseconds
    pub max_latency_ms: u32,
    /// Quality vs latency trade-off (0.0 = fastest, 1.0 = highest quality)
    pub quality_factor: f32,
    /// Buffer size in chunks
    pub buffer_size: usize,
    /// Enable predictive synthesis
    pub enable_prediction: bool,
    /// Adaptive chunk sizing
    pub adaptive_chunking: bool,
    /// Enable voice activity detection
    pub enable_vad: bool,
    /// Lookahead frames for better quality
    pub lookahead_frames: usize,
    /// Number of processing threads
    pub num_threads: usize,
    /// Enable real-time audio streaming
    pub enable_realtime_streaming: bool,
    /// Minimum chunk size for adaptive processing
    pub min_chunk_frames: usize,
    /// Maximum chunk size for adaptive processing
    pub max_chunk_frames: usize,
    /// Predictive synthesis lookahead in phonemes
    pub prediction_lookahead: usize,
    /// Voice activity detection threshold
    pub vad_threshold: f32,
    /// Enable cross-fade between chunks
    pub enable_crossfade: bool,
    /// Cross-fade duration in frames
    pub crossfade_frames: usize,
}

/// Voice Activity Detection configuration
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Energy threshold for voice detection
    pub energy_threshold: f32,
    /// Spectral centroid threshold
    pub spectral_threshold: f32,
    /// Minimum voice duration in frames
    pub min_voice_frames: usize,
    /// Maximum silence duration in frames
    pub max_silence_frames: usize,
    /// Smoothing factor for VAD decisions
    pub smoothing_factor: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            energy_threshold: 0.01,
            spectral_threshold: 1000.0,
            min_voice_frames: 10,
            max_silence_frames: 50,
            smoothing_factor: 0.8,
        }
    }
}

/// Predictive synthesis configuration
#[derive(Debug, Clone)]
pub struct PredictiveConfig {
    /// Enable lookahead processing
    pub enable_lookahead: bool,
    /// Lookahead window size in phonemes
    pub lookahead_window: usize,
    /// Confidence threshold for predictions
    pub confidence_threshold: f32,
    /// Maximum prediction horizon in frames
    pub max_prediction_frames: usize,
    /// Enable context-aware prediction
    pub context_aware: bool,
}

impl Default for PredictiveConfig {
    fn default() -> Self {
        Self {
            enable_lookahead: true,
            lookahead_window: 5,
            confidence_threshold: 0.7,
            max_prediction_frames: 512,
            context_aware: true,
        }
    }
}

/// Real-time streaming configuration
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    /// Target sample rate for audio output
    pub sample_rate: u32,
    /// Audio buffer size in samples
    pub audio_buffer_size: usize,
    /// Number of audio channels
    pub channels: usize,
    /// Enable low-latency mode
    pub low_latency_mode: bool,
    /// Audio format bit depth
    pub bit_depth: u16,
    /// Maximum audio dropout tolerance in ms
    pub dropout_tolerance_ms: u32,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            audio_buffer_size: 1024,
            channels: 1,
            low_latency_mode: true,
            bit_depth: 16,
            dropout_tolerance_ms: 100,
        }
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(2)
            .min(4); // Cap at 4 threads for streaming

        Self {
            chunk_frames: 256,   // ~11ms at 22kHz hop length
            overlap_frames: 64,  // 25% overlap
            max_latency_ms: 50,  // 50ms target latency
            quality_factor: 0.7, // Good balance
            buffer_size: 8,      // 8 chunks buffered
            enable_prediction: true,
            adaptive_chunking: true,
            enable_vad: true,
            lookahead_frames: 128,
            num_threads,
            enable_realtime_streaming: true,
            min_chunk_frames: 64,
            max_chunk_frames: 512,
            prediction_lookahead: 10,
            vad_threshold: 0.01,
            enable_crossfade: true,
            crossfade_frames: 32,
        }
    }
}

impl StreamingConfig {
    /// Validate streaming configuration
    pub fn validate(&self) -> Result<()> {
        if self.chunk_frames == 0 {
            return Err(AcousticError::InvalidConfiguration(
                "Chunk frames must be greater than 0".to_string(),
            ));
        }

        if self.overlap_frames >= self.chunk_frames {
            return Err(AcousticError::InvalidConfiguration(
                "Overlap frames must be less than chunk frames".to_string(),
            ));
        }

        if self.max_latency_ms == 0 {
            return Err(AcousticError::InvalidConfiguration(
                "Max latency must be greater than 0".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&self.quality_factor) {
            return Err(AcousticError::InvalidConfiguration(
                "Quality factor must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.buffer_size == 0 {
            return Err(AcousticError::InvalidConfiguration(
                "Buffer size must be greater than 0".to_string(),
            ));
        }

        if self.min_chunk_frames == 0 {
            return Err(AcousticError::InvalidConfiguration(
                "Minimum chunk frames must be greater than 0".to_string(),
            ));
        }

        if self.max_chunk_frames <= self.min_chunk_frames {
            return Err(AcousticError::InvalidConfiguration(
                "Maximum chunk frames must be greater than minimum chunk frames".to_string(),
            ));
        }

        if self.chunk_frames < self.min_chunk_frames || self.chunk_frames > self.max_chunk_frames {
            return Err(AcousticError::InvalidConfiguration(
                "Chunk frames must be between min and max chunk frames".to_string(),
            ));
        }

        if self.num_threads == 0 {
            return Err(AcousticError::InvalidConfiguration(
                "Number of threads must be greater than 0".to_string(),
            ));
        }

        if self.num_threads > 16 {
            return Err(AcousticError::InvalidConfiguration(
                "Number of threads should not exceed 16 for optimal performance".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&self.vad_threshold) {
            return Err(AcousticError::InvalidConfiguration(
                "VAD threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.crossfade_frames >= self.overlap_frames {
            return Err(AcousticError::InvalidConfiguration(
                "Crossfade frames must be less than overlap frames".to_string(),
            ));
        }

        if self.prediction_lookahead == 0 {
            return Err(AcousticError::InvalidConfiguration(
                "Prediction lookahead must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Get effective chunk size considering overlap
    pub fn effective_chunk_frames(&self) -> usize {
        self.chunk_frames - self.overlap_frames
    }

    /// Calculate latency in milliseconds for given hop length
    pub fn estimated_latency_ms(&self, hop_length: u32, sample_rate: u32) -> f32 {
        let frames_per_ms = sample_rate as f32 / (hop_length as f32 * 1000.0);
        self.chunk_frames as f32 / frames_per_ms
    }
}

/// Enhanced streaming synthesis processor with advanced real-time features
pub struct StreamingSynthesizer<M: AcousticModel> {
    /// Underlying acoustic model
    model: Arc<M>,
    /// Streaming configuration
    config: StreamingConfig,
    /// Circular buffer for input phonemes
    input_buffer: buffer::CircularBuffer<Phoneme>,
    /// Output buffer for mel spectrograms
    output_buffer: buffer::CircularBuffer<MelSpectrogram>,
    /// Windowing function for overlap-add
    _window: Vec<f32>,
    /// Overlap buffer for smooth transitions
    overlap_buffer: Option<MelSpectrogram>,
    /// Processing state
    state: StreamingState,
    /// Performance metrics
    metrics: StreamingMetrics,
    /// Voice activity detector
    vad: Option<VoiceActivityDetector>,
    /// Predictive synthesizer
    predictor: Option<PredictiveSynthesizer<M>>,
    /// Real-time audio streamer
    audio_streamer: Option<RealtimeAudioStreamer>,
    /// Thread pool for parallel processing
    _thread_pool: Option<scirs2_core::parallel_ops::ThreadPool>,
    /// Latency optimizer
    latency_optimizer: LatencyOptimizer,
    /// Processing task handles for async operations
    _processing_handles: Vec<tokio::task::JoinHandle<Result<()>>>,
    /// Cross-fade buffer for smooth transitions
    crossfade_buffer: Option<MelSpectrogram>,
}

/// Streaming processing state
#[derive(Debug, Clone)]
pub enum StreamingState {
    Idle,
    Processing,
    Flushing,
    Predicting,
    Buffering,
    Error(String),
}

/// Voice Activity Detector for efficient processing
pub struct VoiceActivityDetector {
    config: VadConfig,
    energy_history: std::collections::VecDeque<f32>,
    voice_state: bool,
    voice_frame_count: usize,
    silence_frame_count: usize,
    smoothed_decision: f32,
}

impl VoiceActivityDetector {
    /// Create new VAD instance
    pub fn new(config: VadConfig) -> Self {
        Self {
            config,
            energy_history: std::collections::VecDeque::with_capacity(100),
            voice_state: false,
            voice_frame_count: 0,
            silence_frame_count: 0,
            smoothed_decision: 0.0,
        }
    }

    /// Detect voice activity in phoneme sequence
    pub fn detect_voice_activity(&mut self, phonemes: &[Phoneme]) -> bool {
        if phonemes.is_empty() {
            return false;
        }

        // Simple heuristic: estimate energy from phoneme characteristics
        let estimated_energy = self.estimate_phoneme_energy(phonemes);
        self.energy_history.push_back(estimated_energy);

        if self.energy_history.len() > 20 {
            self.energy_history.pop_front();
        }

        // Apply energy threshold
        let voice_detected = estimated_energy > self.config.energy_threshold;

        // Apply smoothing
        let target = if voice_detected { 1.0 } else { 0.0 };
        self.smoothed_decision = self.config.smoothing_factor * self.smoothed_decision
            + (1.0 - self.config.smoothing_factor) * target;

        let smooth_voice = self.smoothed_decision > 0.5;

        // Apply duration constraints
        if smooth_voice {
            self.voice_frame_count += 1;
            self.silence_frame_count = 0;

            if self.voice_frame_count >= self.config.min_voice_frames {
                self.voice_state = true;
            }
        } else {
            self.silence_frame_count += 1;
            self.voice_frame_count = 0;

            if self.silence_frame_count >= self.config.max_silence_frames {
                self.voice_state = false;
            }
        }

        self.voice_state
    }

    /// Reset VAD state
    pub fn reset(&mut self) {
        self.energy_history.clear();
        self.voice_state = false;
        self.voice_frame_count = 0;
        self.silence_frame_count = 0;
        self.smoothed_decision = 0.0;
    }

    fn estimate_phoneme_energy(&self, phonemes: &[Phoneme]) -> f32 {
        // Simple energy estimation based on phoneme characteristics
        let total_energy: f32 = phonemes
            .iter()
            .map(|p| {
                // Classify phonemes based on their symbol
                let symbol = &p.symbol.to_lowercase();
                if self.is_vowel_symbol(symbol) {
                    0.8 // Vowels are typically higher energy
                } else if self.is_silence_symbol(symbol) {
                    0.0 // Silence has no energy
                } else {
                    0.3 // Consonants and other phonemes are lower energy
                }
            })
            .sum();

        if phonemes.is_empty() {
            0.0
        } else {
            total_energy / phonemes.len() as f32
        }
    }

    fn is_vowel_symbol(&self, symbol: &str) -> bool {
        // Common vowel symbols in IPA and various languages
        matches!(
            symbol,
            "a" | "e"
                | "i"
                | "o"
                | "u"
                | "ɑ"
                | "ɛ"
                | "ɪ"
                | "ɔ"
                | "ʊ"
                | "ə"
                | "ɯ"
                | "ɤ"
                | "y"
                | "ø"
                | "æ"
                | "ɨ"
                | "ʉ"
                | "ɘ"
                | "ɵ"
                | "ɐ"
                | "ɶ"
                | "ɒ"
                | "ɞ"
                | "ɜ"
                | "ɝ"
                | "ɚ"
                | "ɹ"
                | "ɻ"
                | "ɺ"
                | "ɭ"
                | "ɽ"
        )
    }

    fn is_silence_symbol(&self, symbol: &str) -> bool {
        // Common silence symbols
        matches!(
            symbol,
            "" | "sil" | "SIL" | "_" | " " | "<silence>" | "<sil>"
        )
    }
}

/// Predictive synthesis processor
pub struct PredictiveSynthesizer<M: AcousticModel> {
    model: Arc<M>,
    config: PredictiveConfig,
    phoneme_buffer: std::collections::VecDeque<Phoneme>,
    prediction_cache: std::collections::HashMap<Vec<Phoneme>, MelSpectrogram>,
    confidence_scores: std::collections::VecDeque<f32>,
}

impl<M: AcousticModel> PredictiveSynthesizer<M> {
    /// Create new predictive synthesizer
    pub fn new(model: Arc<M>, config: PredictiveConfig) -> Self {
        let lookahead_capacity = config.lookahead_window * 2;
        Self {
            model,
            config,
            phoneme_buffer: std::collections::VecDeque::with_capacity(lookahead_capacity),
            prediction_cache: std::collections::HashMap::new(),
            confidence_scores: std::collections::VecDeque::with_capacity(100),
        }
    }

    /// Add phonemes to prediction buffer
    pub fn add_phonemes(&mut self, phonemes: &[Phoneme]) {
        for phoneme in phonemes {
            self.phoneme_buffer.push_back(phoneme.clone());
        }

        // Limit buffer size
        while self.phoneme_buffer.len() > self.config.lookahead_window * 2 {
            self.phoneme_buffer.pop_front();
        }
    }

    /// Predict next mel spectrogram chunk
    pub async fn predict_next_chunk(
        &mut self,
        synthesis_config: Option<&SynthesisConfig>,
    ) -> Result<Option<MelSpectrogram>> {
        if !self.config.enable_lookahead || self.phoneme_buffer.len() < self.config.lookahead_window
        {
            return Ok(None);
        }

        // Extract prediction window
        let prediction_phonemes: Vec<Phoneme> = self
            .phoneme_buffer
            .iter()
            .take(self.config.lookahead_window)
            .cloned()
            .collect();

        // Check cache first
        if let Some(cached_mel) = self.prediction_cache.get(&prediction_phonemes) {
            return Ok(Some(cached_mel.clone()));
        }

        // Estimate confidence for this prediction
        let confidence = self.estimate_prediction_confidence(&prediction_phonemes);

        if confidence < self.config.confidence_threshold {
            return Ok(None);
        }

        // Perform prediction
        let predicted_mel = self
            .model
            .synthesize(&prediction_phonemes, synthesis_config)
            .await?;

        // Cache the result
        self.prediction_cache
            .insert(prediction_phonemes, predicted_mel.clone());

        // Limit cache size
        if self.prediction_cache.len() > 100 {
            self.prediction_cache.clear();
        }

        // Update confidence tracking
        self.confidence_scores.push_back(confidence);
        if self.confidence_scores.len() > 100 {
            self.confidence_scores.pop_front();
        }

        Ok(Some(predicted_mel))
    }

    /// Get average prediction confidence
    pub fn get_average_confidence(&self) -> f32 {
        if self.confidence_scores.is_empty() {
            0.0
        } else {
            self.confidence_scores.iter().sum::<f32>() / self.confidence_scores.len() as f32
        }
    }

    /// Clear prediction cache
    pub fn clear_cache(&mut self) {
        self.prediction_cache.clear();
        self.confidence_scores.clear();
    }

    fn estimate_prediction_confidence(&self, phonemes: &[Phoneme]) -> f32 {
        // Simple confidence estimation based on phoneme sequence characteristics
        if phonemes.is_empty() {
            return 0.0;
        }

        let mut confidence = 0.5; // Base confidence

        // Higher confidence for common phoneme patterns
        let vowel_count = phonemes
            .iter()
            .filter(|p| self.is_vowel_symbol(&p.symbol.to_lowercase()))
            .count();
        let _consonant_count = phonemes
            .iter()
            .filter(|p| {
                !self.is_vowel_symbol(&p.symbol.to_lowercase())
                    && !self.is_silence_symbol(&p.symbol.to_lowercase())
            })
            .count();

        let vowel_ratio = vowel_count as f32 / phonemes.len() as f32;

        // Balanced vowel/consonant ratio increases confidence
        if (0.2..=0.6).contains(&vowel_ratio) {
            confidence += 0.2;
        }

        // Longer sequences with patterns increase confidence
        if phonemes.len() >= 3 {
            confidence += 0.1;
        }

        // Context-aware adjustments
        if self.config.context_aware {
            let avg_historical_confidence = self.get_average_confidence();
            if avg_historical_confidence > 0.0 {
                confidence = (confidence + avg_historical_confidence) / 2.0;
            }
        }

        confidence.clamp(0.0, 1.0)
    }

    fn is_vowel_symbol(&self, symbol: &str) -> bool {
        // Common vowel symbols in IPA and various languages
        matches!(
            symbol,
            "a" | "e"
                | "i"
                | "o"
                | "u"
                | "ɑ"
                | "ɛ"
                | "ɪ"
                | "ɔ"
                | "ʊ"
                | "ə"
                | "ɯ"
                | "ɤ"
                | "y"
                | "ø"
                | "æ"
                | "ɨ"
                | "ʉ"
                | "ɘ"
                | "ɵ"
                | "ɐ"
                | "ɶ"
                | "ɒ"
                | "ɞ"
                | "ɜ"
                | "ɝ"
                | "ɚ"
                | "ɹ"
                | "ɻ"
                | "ɺ"
                | "ɭ"
                | "ɽ"
        )
    }

    fn is_silence_symbol(&self, symbol: &str) -> bool {
        // Common silence symbols
        matches!(
            symbol,
            "" | "sil" | "SIL" | "_" | " " | "<silence>" | "<sil>"
        )
    }
}

/// Real-time audio streaming processor
pub struct RealtimeAudioStreamer {
    config: RealtimeConfig,
    audio_buffer: std::collections::VecDeque<f32>,
    output_callback: Option<OutputCallback>,
    streaming_active: bool,
    dropout_count: u32,
    total_samples_processed: u64,
}

impl RealtimeAudioStreamer {
    /// Create new real-time audio streamer
    pub fn new(config: RealtimeConfig) -> Self {
        let buffer_capacity = config.audio_buffer_size * 4;
        Self {
            config,
            audio_buffer: std::collections::VecDeque::with_capacity(buffer_capacity),
            output_callback: None,
            streaming_active: false,
            dropout_count: 0,
            total_samples_processed: 0,
        }
    }

    /// Set audio output callback
    pub fn set_output_callback<F>(&mut self, callback: F)
    where
        F: Fn(&[f32]) + Send + Sync + 'static,
    {
        self.output_callback = Some(Box::new(callback));
    }

    /// Add mel spectrogram data for audio conversion and streaming
    pub fn add_mel_data(&mut self, mel: &MelSpectrogram) -> Result<()> {
        // Convert mel spectrogram to audio samples (this would use a vocoder in practice)
        let audio_samples = self.mel_to_audio(mel)?;

        for sample in audio_samples {
            self.audio_buffer.push_back(sample);
        }

        // Process buffered audio
        self.process_audio_buffer()?;

        Ok(())
    }

    /// Start real-time streaming
    pub fn start_streaming(&mut self) -> Result<()> {
        self.streaming_active = true;
        Ok(())
    }

    /// Stop real-time streaming
    pub fn stop_streaming(&mut self) {
        self.streaming_active = false;
    }

    /// Get streaming statistics
    pub fn get_stats(&self) -> RealtimeStats {
        RealtimeStats {
            buffer_size: self.audio_buffer.len(),
            dropout_count: self.dropout_count,
            total_samples: self.total_samples_processed,
            is_active: self.streaming_active,
        }
    }

    fn mel_to_audio(&self, mel: &MelSpectrogram) -> Result<Vec<f32>> {
        // Placeholder implementation - in practice this would use a vocoder
        // For now, generate simple audio based on mel energy
        let mut audio = Vec::new();
        let samples_per_frame = 256; // Hop length

        for frame in 0..mel.n_frames {
            // Calculate frame energy
            let mut energy = 0.0f32;
            for mel_bin in 0..mel.n_mels {
                if frame < mel.data[mel_bin].len() {
                    energy += mel.data[mel_bin][frame];
                }
            }
            energy /= mel.n_mels as f32;

            // Generate simple sine wave audio
            for sample_idx in 0..samples_per_frame {
                let time = (frame * samples_per_frame + sample_idx) as f32
                    / self.config.sample_rate as f32;
                let frequency = 440.0 + energy * 200.0; // Simple frequency modulation
                let amplitude = (energy * 0.1).min(0.1); // Limit amplitude
                let sample = amplitude * (2.0 * std::f32::consts::PI * frequency * time).sin();
                audio.push(sample);
            }
        }

        Ok(audio)
    }

    fn process_audio_buffer(&mut self) -> Result<()> {
        if !self.streaming_active {
            return Ok(());
        }

        while self.audio_buffer.len() >= self.config.audio_buffer_size {
            let mut chunk = Vec::with_capacity(self.config.audio_buffer_size);

            for _ in 0..self.config.audio_buffer_size {
                if let Some(sample) = self.audio_buffer.pop_front() {
                    chunk.push(sample);
                } else {
                    break;
                }
            }

            if chunk.len() == self.config.audio_buffer_size {
                // Call output callback
                if let Some(ref callback) = self.output_callback {
                    callback(&chunk);
                }
                self.total_samples_processed += chunk.len() as u64;
            } else {
                // Handle underrun
                self.dropout_count += 1;
                break;
            }
        }

        Ok(())
    }
}

/// Real-time streaming statistics
#[derive(Debug, Clone)]
pub struct RealtimeStats {
    pub buffer_size: usize,
    pub dropout_count: u32,
    pub total_samples: u64,
    pub is_active: bool,
}

/// Performance metrics for streaming synthesis
#[derive(Debug, Clone)]
pub struct StreamingMetrics {
    /// Total chunks processed
    pub chunks_processed: u64,
    /// Average processing time per chunk
    pub avg_processing_time_ms: f32,
    /// Current latency in milliseconds
    pub current_latency_ms: f32,
    /// Buffer utilization (0.0 to 1.0)
    pub buffer_utilization: f32,
    /// Quality metric (higher is better)
    pub quality_score: f32,
}

impl Default for StreamingMetrics {
    fn default() -> Self {
        Self {
            chunks_processed: 0,
            avg_processing_time_ms: 0.0,
            current_latency_ms: 0.0,
            buffer_utilization: 0.0,
            quality_score: 1.0,
        }
    }
}

impl<M: AcousticModel> StreamingSynthesizer<M> {
    /// Create new streaming synthesizer
    pub fn new(model: Arc<M>, config: StreamingConfig) -> Result<Self> {
        config.validate()?;

        let input_buffer = buffer::CircularBuffer::new(config.buffer_size * config.chunk_frames)?;
        let output_buffer = buffer::CircularBuffer::new(config.buffer_size)?;

        // Create Hann window for overlap-add
        let window = create_hann_window(config.overlap_frames);

        // Initialize Voice Activity Detector if enabled
        let vad = if config.enable_vad {
            Some(VoiceActivityDetector::new(VadConfig::default()))
        } else {
            None
        };

        // Initialize Predictive Synthesizer if enabled
        let predictor = if config.enable_prediction {
            Some(PredictiveSynthesizer::new(
                model.clone(),
                PredictiveConfig::default(),
            ))
        } else {
            None
        };

        // Initialize Real-time Audio Streamer if enabled
        let audio_streamer = if config.enable_realtime_streaming {
            Some(RealtimeAudioStreamer::new(RealtimeConfig::default()))
        } else {
            None
        };

        // Initialize thread pool for parallel processing
        let thread_pool = if config.num_threads > 1 {
            Some(
                scirs2_core::parallel_ops::ThreadPoolBuilder::new()
                    .num_threads(config.num_threads)
                    .build()
                    .map_err(|e| {
                        AcousticError::Processing(format!("Failed to create thread pool: {e}"))
                    })?,
            )
        } else {
            None
        };

        // Initialize latency optimizer
        let latency_optimizer_config = LatencyOptimizerConfig {
            target_latency_ms: config.max_latency_ms as f32 * 0.8, // Target 80% of max latency
            max_latency_ms: config.max_latency_ms as f32,
            enable_prediction: config.enable_prediction,
            adaptive_chunks: config.adaptive_chunking,
            ..Default::default()
        };
        let latency_optimizer = LatencyOptimizer::new(latency_optimizer_config);

        Ok(Self {
            model,
            config,
            input_buffer,
            output_buffer,
            _window: window,
            overlap_buffer: None,
            state: StreamingState::Idle,
            metrics: StreamingMetrics::default(),
            vad,
            predictor,
            audio_streamer,
            _thread_pool: thread_pool,
            latency_optimizer,
            _processing_handles: Vec::new(),
            crossfade_buffer: None,
        })
    }

    /// Add phonemes to input buffer
    pub fn push_phonemes(&mut self, phonemes: &[Phoneme]) -> Result<()> {
        for phoneme in phonemes {
            self.input_buffer.push(phoneme.clone())?;
        }
        Ok(())
    }

    /// Process available chunks with advanced real-time features
    pub async fn process_chunks(
        &mut self,
        synthesis_config: Option<&SynthesisConfig>,
    ) -> Result<()> {
        self.state = StreamingState::Processing;

        while self.input_buffer.len() >= self.config.chunk_frames {
            let start_time = Instant::now();

            // Extract chunk with overlap
            let chunk = self.extract_chunk()?;

            // Apply Voice Activity Detection if enabled
            let should_process = if let Some(ref mut vad) = self.vad {
                vad.detect_voice_activity(&chunk)
            } else {
                true // Process all chunks if VAD is disabled
            };

            if !should_process {
                // Skip silent chunks to reduce processing load
                continue;
            }

            // Adaptive chunk sizing and latency optimization
            let effective_config = self.adapt_synthesis_config(synthesis_config);
            let input_complexity = self.estimate_chunk_complexity(&chunk);

            // Try predictive synthesis first if enabled
            let mel_chunk = if let Some(ref mut predictor) = self.predictor {
                // Add phonemes to predictor buffer
                predictor.add_phonemes(&chunk);

                // Check if we have a cached prediction
                if let Some(predicted_mel) = predictor
                    .predict_next_chunk(effective_config.as_ref())
                    .await?
                {
                    predicted_mel
                } else {
                    // Fallback to regular synthesis
                    self.model
                        .synthesize(&chunk, effective_config.as_ref())
                        .await?
                }
            } else {
                // Regular synthesis without prediction
                self.model
                    .synthesize(&chunk, effective_config.as_ref())
                    .await?
            };

            // Apply cross-fade and overlap-add windowing
            let windowed_chunk = if self.config.enable_crossfade {
                self.apply_crossfade(mel_chunk)?
            } else {
                self.apply_overlap_add(mel_chunk)?
            };

            // Add to output buffer
            self.output_buffer.push(windowed_chunk.clone())?;

            // Stream audio in real-time if enabled
            if let Some(ref mut audio_streamer) = self.audio_streamer {
                audio_streamer.add_mel_data(&windowed_chunk)?;
            }

            // Update performance metrics and latency optimizer
            let processing_time = start_time.elapsed();
            let processing_time_ms = processing_time.as_secs_f32() * 1000.0;

            // Update latency optimizer with performance data
            self.latency_optimizer.add_measurement(
                processing_time_ms,
                input_complexity,
                self.estimate_quality_score(&windowed_chunk),
                self.config.chunk_frames,
                effective_config
                    .as_ref()
                    .unwrap_or(&SynthesisConfig::default()),
            );

            self.update_metrics(processing_time);

            // Apply latency optimization if needed
            if !self.latency_optimizer.is_meeting_latency_target() {
                self.optimize_for_latency()?;
            }
        }

        self.state = StreamingState::Idle;
        Ok(())
    }

    /// Get processed mel spectrogram chunks
    pub fn pop_mel_chunk(&mut self) -> Option<MelSpectrogram> {
        self.output_buffer.pop()
    }

    /// Flush remaining input and get final output
    pub async fn flush(
        &mut self,
        synthesis_config: Option<&SynthesisConfig>,
    ) -> Result<Vec<MelSpectrogram>> {
        self.state = StreamingState::Flushing;
        let mut results = Vec::new();

        // Process remaining phonemes in smaller chunks
        while !self.input_buffer.is_empty() {
            let remaining = self.input_buffer.len();
            let chunk_size = remaining.min(self.config.chunk_frames);

            let chunk = self.input_buffer.drain(chunk_size);
            if !chunk.is_empty() {
                let mel_chunk = self.model.synthesize(&chunk, synthesis_config).await?;
                results.push(mel_chunk);
            }
        }

        self.state = StreamingState::Idle;
        Ok(results)
    }

    /// Get current streaming metrics
    pub fn metrics(&self) -> &StreamingMetrics {
        &self.metrics
    }

    /// Reset streaming state
    pub fn reset(&mut self) -> Result<()> {
        self.input_buffer.clear();
        self.output_buffer.clear();
        self.overlap_buffer = None;
        self.crossfade_buffer = None;
        self.state = StreamingState::Idle;
        self.metrics = StreamingMetrics::default();

        // Reset advanced features
        if let Some(ref mut vad) = self.vad {
            vad.reset();
        }
        if let Some(ref mut predictor) = self.predictor {
            predictor.clear_cache();
        }
        if let Some(ref mut audio_streamer) = self.audio_streamer {
            audio_streamer.stop_streaming();
        }

        Ok(())
    }

    /// Start real-time audio streaming if enabled
    pub fn start_audio_streaming(&mut self) -> Result<()> {
        if let Some(ref mut audio_streamer) = self.audio_streamer {
            audio_streamer.start_streaming()?;
        }
        Ok(())
    }

    /// Stop real-time audio streaming
    pub fn stop_audio_streaming(&mut self) {
        if let Some(ref mut audio_streamer) = self.audio_streamer {
            audio_streamer.stop_streaming();
        }
    }

    /// Set audio output callback for real-time streaming
    pub fn set_audio_callback<F>(&mut self, callback: F) -> Result<()>
    where
        F: Fn(&[f32]) + Send + Sync + 'static,
    {
        if let Some(ref mut audio_streamer) = self.audio_streamer {
            audio_streamer.set_output_callback(callback);
            Ok(())
        } else {
            Err(AcousticError::InvalidConfiguration(
                "Real-time audio streaming is not enabled".to_string(),
            ))
        }
    }

    /// Get real-time audio streaming statistics
    pub fn get_audio_stats(&self) -> Option<RealtimeStats> {
        self.audio_streamer
            .as_ref()
            .map(|streamer| streamer.get_stats())
    }

    /// Get latency optimization statistics
    pub fn get_latency_stats(&self) -> crate::streaming::latency::LatencyStats {
        self.latency_optimizer.get_stats()
    }

    /// Get predictive synthesis confidence if enabled
    pub fn get_prediction_confidence(&self) -> Option<f32> {
        self.predictor.as_ref().map(|p| p.get_average_confidence())
    }

    /// Check if voice activity is currently detected
    pub fn is_voice_active(&self) -> bool {
        self.vad.as_ref().map_or(true, |vad| vad.voice_state)
    }

    /// Get current streaming state
    pub fn get_state(&self) -> &StreamingState {
        &self.state
    }

    /// Enable or disable adaptive chunking at runtime
    pub fn set_adaptive_chunking(&mut self, _enabled: bool) {
        // Note: This requires config to be mutable, which would need structural changes
        // For now, this is a placeholder for future enhancement
    }

    /// Update configuration parameters at runtime
    pub fn update_config(&mut self, new_config: StreamingConfig) -> Result<()> {
        new_config.validate()?;

        // Only update safe-to-change parameters during runtime
        // Buffer sizes and major structural changes require restart
        self.config.max_latency_ms = new_config.max_latency_ms;
        self.config.quality_factor = new_config.quality_factor;
        self.config.enable_prediction = new_config.enable_prediction;
        self.config.adaptive_chunking = new_config.adaptive_chunking;
        self.config.vad_threshold = new_config.vad_threshold;
        self.config.enable_crossfade = new_config.enable_crossfade;

        Ok(())
    }

    // Private helper methods

    fn extract_chunk(&mut self) -> Result<Vec<Phoneme>> {
        Ok(self
            .input_buffer
            .drain(self.config.effective_chunk_frames()))
    }

    fn adapt_synthesis_config(
        &self,
        base_config: Option<&SynthesisConfig>,
    ) -> Option<SynthesisConfig> {
        if !self.config.adaptive_chunking {
            return base_config.cloned();
        }

        let mut config = base_config.cloned().unwrap_or_default();

        // Adjust processing parameters based on latency constraints
        if self.metrics.current_latency_ms > self.config.max_latency_ms as f32 * 0.8 {
            // Increase speed for better latency
            config.speed = (config.speed * 1.1).min(2.0);
        }

        Some(config)
    }

    fn apply_overlap_add(&mut self, mel_chunk: MelSpectrogram) -> Result<MelSpectrogram> {
        if let Some(overlap) = &self.overlap_buffer {
            // Apply overlap-add with previous chunk
            let mut result = mel_chunk.clone();
            let overlap_frames = self.config.overlap_frames.min(overlap.n_frames);

            // Simple overlap-add implementation
            for i in 0..overlap_frames {
                let weight = i as f32 / overlap_frames as f32;
                for mel_bin in 0..result.n_mels {
                    if i < overlap.n_frames && i < result.n_frames {
                        result.data[mel_bin][i] = overlap.data[mel_bin]
                            [overlap.n_frames - overlap_frames + i]
                            * (1.0 - weight)
                            + result.data[mel_bin][i] * weight;
                    }
                }
            }

            // Store tail for next overlap
            if mel_chunk.n_frames > self.config.overlap_frames {
                self.overlap_buffer = Some(mel_chunk);
            }

            Ok(result)
        } else {
            // First chunk, no overlap
            self.overlap_buffer = Some(mel_chunk.clone());
            Ok(mel_chunk)
        }
    }

    fn update_metrics(&mut self, processing_time: Duration) {
        self.metrics.chunks_processed += 1;

        let processing_ms = processing_time.as_secs_f32() * 1000.0;

        // Exponential moving average for processing time
        let alpha = 0.1;
        self.metrics.avg_processing_time_ms =
            alpha * processing_ms + (1.0 - alpha) * self.metrics.avg_processing_time_ms;

        // Update current latency estimate
        self.metrics.current_latency_ms = self.metrics.avg_processing_time_ms
            + (self.output_buffer.len() as f32 * self.config.estimated_latency_ms(256, 22050));

        // Update buffer utilization
        self.metrics.buffer_utilization =
            self.input_buffer.len() as f32 / self.input_buffer.capacity() as f32;
    }

    fn optimize_for_latency(&mut self) -> Result<()> {
        // Dynamic optimization strategies
        if self.config.adaptive_chunking {
            // Could implement dynamic chunk size adjustment here
            // For now, we'll rely on synthesis config adaptation
        }
        Ok(())
    }

    /// Estimate complexity of phoneme chunk for adaptive processing
    fn estimate_chunk_complexity(&self, chunk: &[Phoneme]) -> f32 {
        if chunk.is_empty() {
            return 0.0;
        }

        let mut complexity = 0.0;
        let mut vowel_count = 0;
        let mut consonant_count = 0;
        let mut transition_count = 0;

        for (i, phoneme) in chunk.iter().enumerate() {
            let symbol = &phoneme.symbol.to_lowercase();
            if self.is_vowel_symbol(symbol) {
                vowel_count += 1;
                complexity += 0.3; // Vowels are moderately complex
            } else if self.is_silence_symbol(symbol) {
                complexity += 0.1; // Silence is simple
            } else {
                consonant_count += 1;
                complexity += 0.5; // Consonants can be more complex
            }

            // Count phoneme transitions (higher complexity)
            if i > 0 {
                let prev = &chunk[i - 1];
                let prev_symbol = &prev.symbol.to_lowercase();
                if self.is_vowel_symbol(prev_symbol) != self.is_vowel_symbol(symbol) {
                    transition_count += 1;
                    complexity += 0.1; // Transitions add complexity
                }
            }
        }

        // Normalize by chunk length
        complexity /= chunk.len() as f32;

        // Add complexity based on phoneme diversity
        let total_phonemes = chunk.len() as f32;
        let vowel_ratio = vowel_count as f32 / total_phonemes;
        let consonant_ratio = consonant_count as f32 / total_phonemes;
        let transition_ratio = transition_count as f32 / total_phonemes.max(1.0);

        // Balanced phoneme distributions are more complex to synthesize
        if (0.2..=0.7).contains(&vowel_ratio) && (0.3..=0.8).contains(&consonant_ratio) {
            complexity += 0.2;
        }

        // High transition rates increase complexity
        if transition_ratio > 0.5 {
            complexity += transition_ratio * 0.3;
        }

        complexity.clamp(0.0, 1.0)
    }

    /// Apply cross-fade between chunks for smooth transitions
    fn apply_crossfade(&mut self, mel_chunk: MelSpectrogram) -> Result<MelSpectrogram> {
        if let Some(ref crossfade_buffer) = self.crossfade_buffer {
            let mut result = mel_chunk.clone();
            let crossfade_frames = self
                .config
                .crossfade_frames
                .min(crossfade_buffer.n_frames)
                .min(result.n_frames);

            if crossfade_frames > 0 {
                // Apply cross-fade in the overlap region
                for i in 0..crossfade_frames {
                    let fade_in_weight = i as f32 / crossfade_frames as f32;
                    let fade_out_weight = 1.0 - fade_in_weight;

                    for mel_bin in 0..result.n_mels {
                        if i < crossfade_buffer.n_frames && i < result.n_frames {
                            let fade_out_idx = crossfade_buffer.n_frames - crossfade_frames + i;
                            if fade_out_idx < crossfade_buffer.data[mel_bin].len()
                                && i < result.data[mel_bin].len()
                            {
                                result.data[mel_bin][i] =
                                    crossfade_buffer.data[mel_bin][fade_out_idx] * fade_out_weight
                                        + result.data[mel_bin][i] * fade_in_weight;
                            }
                        }
                    }
                }
            }

            // Update crossfade buffer with current chunk's tail
            self.crossfade_buffer = Some(mel_chunk);
            Ok(result)
        } else {
            // First chunk, no crossfade needed
            self.crossfade_buffer = Some(mel_chunk.clone());
            Ok(mel_chunk)
        }
    }

    /// Estimate quality score of synthesized mel spectrogram
    fn estimate_quality_score(&self, mel: &MelSpectrogram) -> f32 {
        if mel.n_frames == 0 || mel.n_mels == 0 {
            return 0.0;
        }

        let mut total_energy = 0.0;
        let mut spectral_flatness = 0.0;
        let mut frame_count = 0;

        // Analyze spectral characteristics
        for frame_idx in 0..mel.n_frames {
            let mut frame_energy = 0.0;
            let mut geometric_mean = 0.0;
            let mut arithmetic_mean = 0.0;
            let mut valid_bins = 0;

            for mel_bin in 0..mel.n_mels {
                if frame_idx < mel.data[mel_bin].len() {
                    let magnitude = mel.data[mel_bin][frame_idx].abs();
                    frame_energy += magnitude;

                    if magnitude > 0.0 {
                        geometric_mean += magnitude.ln();
                        arithmetic_mean += magnitude;
                        valid_bins += 1;
                    }
                }
            }

            if valid_bins > 0 {
                frame_energy /= valid_bins as f32;
                total_energy += frame_energy;

                // Calculate spectral flatness (measure of spectral shape quality)
                geometric_mean = (geometric_mean / valid_bins as f32).exp();
                arithmetic_mean /= valid_bins as f32;

                if arithmetic_mean > 0.0 {
                    spectral_flatness += geometric_mean / arithmetic_mean;
                }

                frame_count += 1;
            }
        }

        if frame_count == 0 {
            return 0.0;
        }

        let avg_energy = total_energy / frame_count as f32;
        let avg_spectral_flatness = spectral_flatness / frame_count as f32;

        // Quality score based on energy distribution and spectral characteristics
        let energy_score = (avg_energy * 10.0).min(1.0); // Normalize energy
        let flatness_score = 1.0 - avg_spectral_flatness.min(1.0); // Higher flatness = lower quality
        let length_score = (mel.n_frames as f32 / 100.0).min(1.0); // Longer sequences generally better

        // Weighted combination
        let quality_score = 0.4 * energy_score + 0.4 * flatness_score + 0.2 * length_score;
        quality_score.clamp(0.0, 1.0)
    }

    fn is_vowel_symbol(&self, symbol: &str) -> bool {
        // Common vowel symbols in IPA and various languages
        matches!(
            symbol,
            "a" | "e"
                | "i"
                | "o"
                | "u"
                | "ɑ"
                | "ɛ"
                | "ɪ"
                | "ɔ"
                | "ʊ"
                | "ə"
                | "ɯ"
                | "ɤ"
                | "y"
                | "ø"
                | "æ"
                | "ɨ"
                | "ʉ"
                | "ɘ"
                | "ɵ"
                | "ɐ"
                | "ɶ"
                | "ɒ"
                | "ɞ"
                | "ɜ"
                | "ɝ"
                | "ɚ"
                | "ɹ"
                | "ɻ"
                | "ɺ"
                | "ɭ"
                | "ɽ"
        )
    }

    fn is_silence_symbol(&self, symbol: &str) -> bool {
        // Common silence symbols
        matches!(
            symbol,
            "" | "sil" | "SIL" | "_" | " " | "<silence>" | "<sil>"
        )
    }
}

/// Create Hann window for overlap-add processing
fn create_hann_window(length: usize) -> Vec<f32> {
    if length == 0 {
        return vec![];
    }

    (0..length)
        .map(|i| {
            let phase = 2.0 * std::f32::consts::PI * i as f32 / (length - 1) as f32;
            0.5 * (1.0 - phase.cos())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config_validation() {
        let config = StreamingConfig::default();
        assert!(config.validate().is_ok());

        let invalid_config = StreamingConfig {
            chunk_frames: 0,
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_streaming_config_effective_chunk() {
        let config = StreamingConfig {
            chunk_frames: 256,
            overlap_frames: 64,
            ..Default::default()
        };
        assert_eq!(config.effective_chunk_frames(), 192);
    }

    #[test]
    fn test_hann_window_creation() {
        let window = create_hann_window(4);
        assert_eq!(window.len(), 4);
        assert!((window[0] - 0.0).abs() < 1e-6);
        assert!((window[3] - 0.0).abs() < 1e-6);
        assert!(window[1] > 0.0 && window[1] < 1.0);
        assert!(window[2] > 0.0 && window[2] < 1.0);
    }

    #[test]
    fn test_streaming_metrics_default() {
        let metrics = StreamingMetrics::default();
        assert_eq!(metrics.chunks_processed, 0);
        assert_eq!(metrics.avg_processing_time_ms, 0.0);
        assert_eq!(metrics.quality_score, 1.0);
    }
}
