//! Recognition Integration - ASR-guided conversion for intelligent voice processing
//!
//! This module provides intelligent voice conversion capabilities guided by
//! Automatic Speech Recognition (ASR) to improve conversion quality and accuracy.
//!
//! ## Features
//!
//! - **Speech-Guided Conversion**: Use ASR transcription to guide voice conversion
//! - **Phoneme-Level Processing**: Fine-grained control based on speech units
//! - **Intelligibility Enhancement**: Maintain speech clarity during conversion
//! - **Context-Aware Processing**: Adapt conversion based on speech content
//! - **Real-time ASR Integration**: Low-latency speech recognition for live conversion
//!
//! ## Example
//!
//! ```rust
//! use voirs_conversion::recognition::{RecognitionGuidedConverter, ASRConfig, ASREngine};
//!
//! let asr_config = ASRConfig::default()
//!     .with_language("en-US")
//!     .with_realtime_threshold(0.3);
//!
//! let mut converter = RecognitionGuidedConverter::new(asr_config)?;
//!
//! let audio_samples = vec![0.1, 0.2, -0.1, 0.05]; // Input audio
//! let result = converter.convert_with_guidance(&audio_samples, "target_speaker")?;
//!
//! println!("Transcription: {}", result.transcription);
//! println!("Quality score: {:.2}", result.quality_score);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::{
    core::QualityMetrics,
    types::{
        ConversionRequest, ConversionResult, ConversionTarget, ConversionType, VoiceCharacteristics,
    },
    Error,
};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Module-level audio analysis helpers
// ---------------------------------------------------------------------------

/// Root-mean-square energy of a sample slice.
fn rms_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Zero-crossing rate: fraction of adjacent-sample sign changes.
fn zero_crossing_rate(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let crossings = samples
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();
    crossings as f32 / samples.len() as f32
}

/// Spectral centroid using a Hann-windowed real DFT via scirs2_fft::rfft.
fn spectral_centroid_simple(samples: &[f32], sample_rate: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let n = samples.len();
    // Apply Hann window
    let windowed: Vec<f64> = samples
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let w =
                (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1).max(1) as f64).cos()) * 0.5;
            s as f64 * w
        })
        .collect();

    let spectrum = match scirs2_fft::rfft(&windowed, None) {
        Ok(s) => s,
        Err(_) => return 0.0,
    };

    let bin_hz = sample_rate as f64 / n as f64;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (k, c) in spectrum.iter().enumerate() {
        let mag = (c.re * c.re + c.im * c.im).sqrt();
        let freq = k as f64 * bin_hz;
        num += freq * mag;
        den += mag;
    }
    if den < 1e-10 {
        0.0
    } else {
        (num / den) as f32
    }
}

/// Classify a speech frame into a broad phoneme category using three acoustic
/// features: zero-crossing rate (ZCR), RMS energy, and spectral centroid.
fn classify_phoneme(zcr: f32, energy: f32, centroid: f32) -> &'static str {
    match (zcr > 0.15, energy > 0.05, centroid > 3000.0) {
        (true, false, true) => "s", // Fricative: high ZCR, low energy, high centroid
        (true, true, true) => "sh", // Fricative with energy
        (false, true, false) => "a", // Vowel: low ZCR, high energy, low centroid
        (false, true, true) => "e", // Front vowel
        (true, true, false) => "n", // Nasal: high ZCR, energy, mid centroid
        (false, false, true) => "t", // Stop burst
        (false, false, false) => "sil", // Silence
        _ => "h",                   // Aspirated / other
    }
}

/// ASR engine types supported for speech recognition
#[derive(Debug, Clone, PartialEq)]
pub enum ASREngine {
    /// Whisper-based ASR (OpenAI)
    Whisper,
    /// DeepSpeech-based ASR (Mozilla)
    DeepSpeech,
    /// Wav2Vec2-based ASR (Facebook)
    Wav2Vec2,
    /// Custom ASR implementation
    Custom(String),
}

/// Configuration for ASR-guided voice conversion
#[derive(Debug, Clone)]
pub struct ASRConfig {
    /// ASR engine to use for speech recognition
    pub engine: ASREngine,
    /// Language code for ASR (e.g., "en-US", "es-ES")
    pub language: String,
    /// Confidence threshold for ASR transcription (0.0-1.0)
    pub confidence_threshold: f32,
    /// Real-time processing threshold in seconds
    pub realtime_threshold: f32,
    /// Enable phoneme-level alignment
    pub phoneme_alignment: bool,
    /// Enable word-level timestamps
    pub word_timestamps: bool,
    /// Maximum audio duration for processing (seconds)
    pub max_audio_duration: f32,
    /// Chunk size for streaming ASR (samples)
    pub chunk_size: usize,
}

impl Default for ASRConfig {
    fn default() -> Self {
        Self {
            engine: ASREngine::Whisper,
            language: "en-US".to_string(),
            confidence_threshold: 0.7,
            realtime_threshold: 0.5,
            phoneme_alignment: true,
            word_timestamps: true,
            max_audio_duration: 30.0,
            chunk_size: 16000, // 1 second at 16kHz
        }
    }
}

impl ASRConfig {
    /// Set the language for ASR processing
    pub fn with_language(mut self, language: &str) -> Self {
        self.language = language.to_string();
        self
    }

    /// Set the confidence threshold for ASR
    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the real-time processing threshold
    pub fn with_realtime_threshold(mut self, threshold: f32) -> Self {
        self.realtime_threshold = threshold.max(0.0);
        self
    }

    /// Enable or disable phoneme alignment
    pub fn with_phoneme_alignment(mut self, enable: bool) -> Self {
        self.phoneme_alignment = enable;
        self
    }
}

/// Word-level timestamp information from ASR
#[derive(Debug, Clone)]
pub struct WordTimestamp {
    /// The recognized word
    pub word: String,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// ASR confidence for this word (0.0-1.0)
    pub confidence: f32,
}

/// Phoneme-level alignment information
#[derive(Debug, Clone)]
pub struct PhonemeAlignment {
    /// Phoneme symbol (IPA or ARPA)
    pub phoneme: String,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Pronunciation confidence (0.0-1.0)
    pub confidence: f32,
}

/// ASR transcription result with timing information
#[derive(Debug, Clone)]
pub struct ASRTranscription {
    /// Full transcription text
    pub text: String,
    /// Overall confidence score (0.0-1.0)
    pub confidence: f32,
    /// Word-level timestamps
    pub word_timestamps: Vec<WordTimestamp>,
    /// Phoneme-level alignment (if enabled)
    pub phoneme_alignment: Vec<PhonemeAlignment>,
    /// Audio duration in seconds
    pub audio_duration: f32,
}

impl std::fmt::Display for ASRTranscription {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (confidence: {:.2})", self.text, self.confidence)
    }
}

/// Speech-guided conversion parameters
#[derive(Debug, Clone)]
pub struct SpeechGuidedParams {
    /// Emphasis on vowel preservation (0.0-1.0)
    pub vowel_emphasis: f32,
    /// Emphasis on consonant clarity (0.0-1.0)
    pub consonant_clarity: f32,
    /// Prosody preservation weight (0.0-1.0)
    pub prosody_weight: f32,
    /// Intelligibility boost factor (0.0-2.0)
    pub intelligibility_boost: f32,
    /// Use phoneme-specific processing
    pub phoneme_specific: bool,
}

impl Default for SpeechGuidedParams {
    fn default() -> Self {
        Self {
            vowel_emphasis: 0.8,
            consonant_clarity: 0.9,
            prosody_weight: 0.7,
            intelligibility_boost: 1.2,
            phoneme_specific: true,
        }
    }
}

/// Result of recognition-guided voice conversion
#[derive(Debug, Clone)]
pub struct RecognitionGuidedResult {
    /// Converted audio samples
    pub audio: Vec<f32>,
    /// ASR transcription of input
    pub transcription: ASRTranscription,
    /// Quality metrics for the conversion
    pub quality_metrics: QualityMetrics,
    /// Processing statistics
    pub stats: RecognitionStats,
    /// Overall quality score (0.0-1.0)
    pub quality_score: f32,
}

/// Statistics for recognition-guided conversion
#[derive(Debug, Clone, Default)]
pub struct RecognitionStats {
    /// ASR processing time (seconds)
    pub asr_processing_time: f32,
    /// Voice conversion time (seconds)
    pub conversion_time: f32,
    /// Total processing time (seconds)
    pub total_time: f32,
    /// Number of phonemes processed
    pub phonemes_processed: usize,
    /// Number of words recognized
    pub words_recognized: usize,
    /// Average word confidence
    pub avg_word_confidence: f32,
    /// Intelligibility preservation score (0.0-1.0)
    pub intelligibility_score: f32,
}

/// Main recognition-guided voice converter
pub struct RecognitionGuidedConverter {
    /// ASR configuration
    config: ASRConfig,
    /// ASR transcription cache
    transcription_cache: HashMap<String, ASRTranscription>,
    /// Speech-guided parameters
    guided_params: SpeechGuidedParams,
    /// Statistics tracking
    stats: RecognitionStats,
}

impl RecognitionGuidedConverter {
    /// Create a new recognition-guided converter
    pub fn new(config: ASRConfig) -> Result<Self, Error> {
        Ok(Self {
            config,
            transcription_cache: HashMap::new(),
            guided_params: SpeechGuidedParams::default(),
            stats: RecognitionStats::default(),
        })
    }

    /// Create converter with custom guided parameters
    pub fn with_guided_params(mut self, params: SpeechGuidedParams) -> Self {
        self.guided_params = params;
        self
    }

    /// Convert audio with ASR guidance
    pub fn convert_with_guidance(
        &mut self,
        audio: &[f32],
        target: &str,
    ) -> Result<RecognitionGuidedResult, Error> {
        let start_time = std::time::Instant::now();

        // Step 1: Perform ASR transcription
        let asr_start = std::time::Instant::now();
        let transcription = self.transcribe_audio(audio)?;
        let asr_time = asr_start.elapsed().as_secs_f32();

        // Step 2: Apply speech-guided voice conversion
        let conversion_start = std::time::Instant::now();
        let conversion_result = self.convert_with_speech_guidance(audio, target, &transcription)?;
        let conversion_time = conversion_start.elapsed().as_secs_f32();

        let total_time = start_time.elapsed().as_secs_f32();

        // Step 3: Calculate quality metrics
        let quality_metrics = self.calculate_guided_quality_metrics(
            &conversion_result.converted_audio,
            audio,
            &transcription,
        )?;
        let quality_score = self.calculate_overall_quality_score(&quality_metrics, &transcription);

        // Step 4: Update statistics
        self.update_stats(
            &transcription,
            asr_time,
            conversion_time,
            total_time,
            &quality_metrics,
        );

        Ok(RecognitionGuidedResult {
            audio: conversion_result.converted_audio,
            transcription,
            quality_metrics,
            stats: self.stats.clone(),
            quality_score,
        })
    }

    /// Perform streaming conversion with ASR guidance
    pub fn convert_streaming(
        &mut self,
        audio_chunk: &[f32],
        target: &str,
        is_final: bool,
    ) -> Result<Vec<f32>, Error> {
        // For streaming, we accumulate chunks and process when we have enough context
        if audio_chunk.len() < self.config.chunk_size && !is_final {
            return Ok(Vec::new()); // Need more audio
        }

        // Quick ASR for streaming (reduced quality for speed)
        let streaming_config = ASRConfig {
            confidence_threshold: self.config.confidence_threshold * 0.8, // Lower threshold for streaming
            phoneme_alignment: false,                                     // Disable for speed
            word_timestamps: false,                                       // Disable for speed
            ..self.config.clone()
        };

        // Perform lightweight transcription
        let transcription = self.transcribe_with_config(audio_chunk, &streaming_config)?;

        // Apply guided conversion with reduced processing
        let guided_params = SpeechGuidedParams {
            phoneme_specific: false, // Disable for speed
            ..self.guided_params.clone()
        };

        let result = self.convert_with_speech_guidance_params(
            audio_chunk,
            target,
            &transcription,
            &guided_params,
        )?;
        Ok(result.converted_audio)
    }

    /// Transcribe audio using configured ASR engine
    fn transcribe_audio(&mut self, audio: &[f32]) -> Result<ASRTranscription, Error> {
        self.transcribe_with_config(audio, &self.config.clone())
    }

    /// Transcribe audio with specific configuration
    fn transcribe_with_config(
        &mut self,
        audio: &[f32],
        config: &ASRConfig,
    ) -> Result<ASRTranscription, Error> {
        // Check cache first
        let audio_hash = self.calculate_audio_hash(audio);
        if let Some(cached) = self.transcription_cache.get(&audio_hash) {
            return Ok(cached.clone());
        }

        // Check audio duration
        let duration = audio.len() as f32 / 16000.0; // Assuming 16kHz
        if duration > config.max_audio_duration {
            return Err(Error::validation(format!(
                "Audio duration {:.2}s exceeds maximum {:.2}s",
                duration, config.max_audio_duration
            )));
        }

        // Perform ASR transcription (simulated for now - in real implementation would use actual ASR)
        let transcription = self.simulate_asr_transcription(audio, config, duration)?;

        // Cache the result
        self.transcription_cache
            .insert(audio_hash, transcription.clone());

        Ok(transcription)
    }

    /// Simulate ASR transcription (placeholder for actual ASR engine integration)
    fn simulate_asr_transcription(
        &self,
        audio: &[f32],
        config: &ASRConfig,
        duration: f32,
    ) -> Result<ASRTranscription, Error> {
        // This is a simulation - in real implementation, this would interface with actual ASR engines

        // Simulate transcription based on audio characteristics
        let avg_energy = audio.iter().map(|&s| s.abs()).sum::<f32>() / audio.len() as f32;
        let confidence = (avg_energy * 2.0).min(1.0).max(config.confidence_threshold);

        // Generate simulated text based on audio energy patterns
        let words = self.generate_simulated_words(audio, duration);
        let text = words
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        // Generate phoneme alignment if enabled
        let phoneme_alignment = if config.phoneme_alignment {
            self.generate_simulated_phonemes(&words)
        } else {
            Vec::new()
        };

        Ok(ASRTranscription {
            text,
            confidence,
            word_timestamps: words,
            phoneme_alignment,
            audio_duration: duration,
        })
    }

    /// Segment audio into voiced word regions using energy-based voice activity
    /// detection, then classify each region with acoustic features.
    fn generate_simulated_words(&self, audio: &[f32], duration: f32) -> Vec<WordTimestamp> {
        const SAMPLE_RATE: f32 = 16000.0;
        const FRAME_LEN: usize = 320; // 20 ms at 16 kHz
        const VOICED_ENERGY_THRESH: f32 = 0.01;
        const SILENCE_FRAMES_TO_SPLIT: usize = 3; // 60 ms silence → new word

        if audio.len() < FRAME_LEN {
            return Vec::new();
        }

        // Per-frame RMS energy
        let frame_energies: Vec<f32> = audio.chunks(FRAME_LEN).map(rms_energy).collect();

        // Group consecutive voiced frames into word segments
        let mut words: Vec<WordTimestamp> = Vec::new();
        let mut seg_start: Option<usize> = None;
        let mut silence_run = 0usize;

        for (i, &energy) in frame_energies.iter().enumerate() {
            if energy >= VOICED_ENERGY_THRESH {
                if seg_start.is_none() {
                    seg_start = Some(i);
                }
                silence_run = 0;
            } else {
                silence_run += 1;
                if silence_run >= SILENCE_FRAMES_TO_SPLIT {
                    if let Some(start) = seg_start.take() {
                        let end = i.saturating_sub(silence_run - 1);
                        if end > start {
                            let start_time = start as f32 * FRAME_LEN as f32 / SAMPLE_RATE;
                            let end_time =
                                (end as f32 * FRAME_LEN as f32 / SAMPLE_RATE).min(duration);
                            let seg_start_sample = (start * FRAME_LEN).min(audio.len());
                            let seg_end_sample = (end * FRAME_LEN).min(audio.len());
                            let seg_samples = &audio[seg_start_sample..seg_end_sample];
                            let avg_e = rms_energy(seg_samples);
                            let zcr = zero_crossing_rate(seg_samples);
                            let centroid = spectral_centroid_simple(seg_samples, SAMPLE_RATE);
                            let phoneme = classify_phoneme(zcr, avg_e, centroid);
                            let word = format!("{}-{}", phoneme, words.len() + 1);
                            let confidence = (avg_e * 20.0).clamp(0.4, 1.0);
                            words.push(WordTimestamp {
                                word,
                                start_time,
                                end_time,
                                confidence,
                            });
                        }
                    }
                    silence_run = 0;
                }
            }
        }

        // Flush final voiced segment
        if let Some(start) = seg_start {
            let end = frame_energies.len();
            let start_time = start as f32 * FRAME_LEN as f32 / SAMPLE_RATE;
            let end_time = (end as f32 * FRAME_LEN as f32 / SAMPLE_RATE).min(duration);
            let seg_samples = &audio[(start * FRAME_LEN).min(audio.len())..];
            if !seg_samples.is_empty() {
                let avg_e = rms_energy(seg_samples);
                let zcr = zero_crossing_rate(seg_samples);
                let centroid = spectral_centroid_simple(seg_samples, SAMPLE_RATE);
                let phoneme = classify_phoneme(zcr, avg_e, centroid);
                let word = format!("{}-{}", phoneme, words.len() + 1);
                let confidence = (avg_e * 20.0).clamp(0.4, 1.0);
                words.push(WordTimestamp {
                    word,
                    start_time,
                    end_time,
                    confidence,
                });
            }
        }

        words
    }

    /// Subdivide each word segment into sub-phoneme units using the word's
    /// classified base phoneme. Duration is evenly split at ~60 ms per unit with
    /// alternating phoneme variants to reflect coarticulation.
    fn generate_simulated_phonemes(&self, words: &[WordTimestamp]) -> Vec<PhonemeAlignment> {
        const PHONEME_DURATION: f32 = 0.06; // ~60 ms per phoneme unit
        const MAX_PHONEMES_PER_WORD: usize = 8;

        let mut phonemes = Vec::new();

        for word in words {
            let word_duration = (word.end_time - word.start_time).max(PHONEME_DURATION);
            let n_phonemes = ((word_duration / PHONEME_DURATION).round() as usize)
                .max(1)
                .min(MAX_PHONEMES_PER_WORD);
            let phoneme_dur = word_duration / n_phonemes as f32;

            // Derive the base phoneme class from the word label (chars before '-')
            let base_phoneme = word.word.split('-').next().unwrap_or("a");

            for i in 0..n_phonemes {
                let start_time = word.start_time + i as f32 * phoneme_dur;
                let end_time = (start_time + phoneme_dur).min(word.end_time);

                // Cycle through coarticulation variants to avoid identical labels
                let variant = match i % 3 {
                    0 => base_phoneme,
                    1 => {
                        if base_phoneme == "sil" {
                            "sil"
                        } else {
                            "ə"
                        }
                    }
                    _ => {
                        if base_phoneme == "sil" {
                            "sil"
                        } else {
                            "ɪ"
                        }
                    }
                };

                let confidence = (word.confidence * (1.0 - i as f32 * 0.03)).max(0.5);
                phonemes.push(PhonemeAlignment {
                    phoneme: variant.to_string(),
                    start_time,
                    end_time,
                    confidence,
                });
            }
        }

        phonemes
    }

    /// Apply speech-guided voice conversion
    fn convert_with_speech_guidance(
        &self,
        audio: &[f32],
        target: &str,
        transcription: &ASRTranscription,
    ) -> Result<ConversionResult, Error> {
        self.convert_with_speech_guidance_params(audio, target, transcription, &self.guided_params)
    }

    /// Apply speech-guided voice conversion with custom parameters
    fn convert_with_speech_guidance_params(
        &self,
        audio: &[f32],
        target: &str,
        transcription: &ASRTranscription,
        params: &SpeechGuidedParams,
    ) -> Result<ConversionResult, Error> {
        // Create conversion request with speech guidance
        let target_characteristics = VoiceCharacteristics::new();
        let conversion_target =
            ConversionTarget::new(target_characteristics).with_speaker_id(target.to_string());

        let mut request = ConversionRequest::new(
            format!("asr_guided_{target}"),
            audio.to_vec(),
            16000, // Assuming 16kHz sample rate
            ConversionType::SpeakerConversion,
            conversion_target,
        );

        // Apply speech-guided modifications to the conversion request
        if params.phoneme_specific && !transcription.phoneme_alignment.is_empty() {
            request = self.apply_phoneme_guidance(request, transcription, params);
        }

        // Apply intelligibility enhancements
        request = self.apply_intelligibility_enhancements(request, transcription, params);

        // Perform the conversion (simulated for ASR-guided demonstration)
        Ok(self.simulate_speech_guided_conversion(&request, transcription, params))
    }

    /// Apply phoneme-specific guidance to conversion
    fn apply_phoneme_guidance(
        &self,
        mut request: ConversionRequest,
        transcription: &ASRTranscription,
        params: &SpeechGuidedParams,
    ) -> ConversionRequest {
        // Analyze phonemes and adjust conversion parameters
        let vowel_phonemes = transcription
            .phoneme_alignment
            .iter()
            .filter(|p| self.is_vowel_phoneme(&p.phoneme))
            .count();

        let consonant_phonemes = transcription.phoneme_alignment.len() - vowel_phonemes;

        // Adjust conversion strength based on phoneme distribution
        if vowel_phonemes > consonant_phonemes {
            // More vowels - emphasize vowel preservation
            request = request.with_parameter(
                "conversion_strength".to_string(),
                0.8 * params.vowel_emphasis,
            );
        } else {
            // More consonants - emphasize consonant clarity
            request = request.with_parameter(
                "conversion_strength".to_string(),
                0.9 * params.consonant_clarity,
            );
        }

        request
    }

    /// Apply intelligibility enhancements
    fn apply_intelligibility_enhancements(
        &self,
        mut request: ConversionRequest,
        transcription: &ASRTranscription,
        params: &SpeechGuidedParams,
    ) -> ConversionRequest {
        // Boost intelligibility based on transcription confidence
        let avg_confidence = if transcription.word_timestamps.is_empty() {
            transcription.confidence
        } else {
            transcription
                .word_timestamps
                .iter()
                .map(|w| w.confidence)
                .sum::<f32>()
                / transcription.word_timestamps.len() as f32
        };

        // Lower confidence means we need more intelligibility preservation
        let intelligibility_factor = (2.0 - avg_confidence) * params.intelligibility_boost;
        let preservation = (0.9 * intelligibility_factor).min(1.0);

        request = request.with_parameter("source_preservation".to_string(), preservation);
        request
    }

    /// Simulate speech-guided voice conversion
    fn simulate_speech_guided_conversion(
        &self,
        request: &ConversionRequest,
        transcription: &ASRTranscription,
        params: &SpeechGuidedParams,
    ) -> ConversionResult {
        use std::time::SystemTime;

        let _t0 = std::time::Instant::now();

        // Simulate the conversion process with speech guidance
        let processed_audio =
            self.apply_speech_guided_processing(&request.source_audio, transcription, params);

        // Compute a rough SNR-based quality metric from the processed audio
        let signal_rms = (processed_audio.iter().map(|&s| s * s).sum::<f32>()
            / processed_audio.len().max(1) as f32)
            .sqrt();
        let noise_est = processed_audio
            .iter()
            .map(|&s| s.abs())
            .fold(f32::MAX, f32::min)
            .max(1e-8);
        let overall_quality = (signal_rms / noise_est).min(100.0).log10() / 2.0;
        let overall_quality = overall_quality.clamp(0.0, 1.0);

        // Create conversion result
        ConversionResult {
            request_id: request.id.clone(),
            converted_audio: processed_audio,
            output_sample_rate: request.source_sample_rate,
            quality_metrics: HashMap::from([
                ("intelligibility".to_string(), transcription.confidence),
                ("prosody_preservation".to_string(), params.prosody_weight),
                ("overall_quality".to_string(), overall_quality),
            ]),
            artifacts: None,
            objective_quality: None,
            processing_time: _t0.elapsed(),
            conversion_type: request.conversion_type.clone(),
            success: true,
            error_message: None,
            timestamp: SystemTime::now(),
        }
    }

    /// Apply speech-guided processing to audio using FFT-domain per-phoneme spectral shaping.
    ///
    /// For each phoneme segment the audio is processed in overlapping frames of 512 samples
    /// (50 % hop), shaped in the frequency domain according to the phoneme type (vowel vs.
    /// consonant), and overlap-added back.  A global intelligibility clamp pass is applied
    /// at the end.
    fn apply_speech_guided_processing(
        &self,
        audio: &[f32],
        transcription: &ASRTranscription,
        params: &SpeechGuidedParams,
    ) -> Vec<f32> {
        use scirs2_core::numeric::Complex64;

        const SAMPLE_RATE: f64 = 16000.0;
        const FRAME_LEN: usize = 512;
        const HOP: usize = FRAME_LEN / 2;

        let n_audio = audio.len();
        // Accumulation buffer for overlap-add, initialised from the source
        let mut processed: Vec<f32> = audio.to_vec();

        // Per-phoneme spectral shaping
        for phoneme in &transcription.phoneme_alignment {
            let start_sample = ((phoneme.start_time * SAMPLE_RATE as f32) as usize).min(n_audio);
            let end_sample = ((phoneme.end_time * SAMPLE_RATE as f32) as usize).min(n_audio);

            if start_sample >= end_sample {
                continue;
            }

            let is_vowel = self.is_vowel_phoneme(&phoneme.phoneme);

            // Determine spectral boost for this phoneme class
            let (boost_lo_hz, boost_hi_hz, boost_gain) = if is_vowel {
                (
                    400.0_f64,
                    2000.0_f64,
                    1.0 + params.vowel_emphasis as f64 * 0.4,
                )
            } else {
                (
                    3000.0_f64,
                    8000.0_f64,
                    1.0 + params.consonant_clarity as f64 * 0.3,
                )
            };

            let region_len = end_sample - start_sample;

            if region_len < FRAME_LEN {
                // Single-frame path: FFT on the whole region
                let frame_f64: Vec<f64> = audio[start_sample..end_sample]
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| {
                        let w = Self::hann_coeff(i, region_len);
                        s as f64 * w
                    })
                    .collect();

                if let Ok(mut spectrum) = scirs2_fft::rfft(&frame_f64, None) {
                    // Apply frequency-domain boost
                    let bin_hz = SAMPLE_RATE / region_len as f64;
                    for (k, c) in spectrum.iter_mut().enumerate() {
                        let freq = k as f64 * bin_hz;
                        if freq >= boost_lo_hz && freq <= boost_hi_hz {
                            c.re *= boost_gain;
                            c.im *= boost_gain;
                        }
                    }

                    if let Ok(time_out) = scirs2_fft::irfft(spectrum.as_slice(), Some(region_len)) {
                        // Synthesis window + overlap-add
                        let norm = region_len as f64;
                        for (i, &s) in time_out.iter().enumerate() {
                            let w = Self::hann_coeff(i, region_len);
                            let idx = start_sample + i;
                            if idx < n_audio {
                                processed[idx] = (s * w / norm) as f32;
                            }
                        }
                    }
                }
            } else {
                // Overlap-add path over the phoneme region
                // We accumulate into a local buffer then write back
                let mut ola_buf = vec![0.0_f64; region_len];
                let mut norm_buf = vec![0.0_f64; region_len];

                let mut frame_start = 0usize;
                while frame_start < region_len {
                    let frame_end = (frame_start + FRAME_LEN).min(region_len);
                    let actual_len = frame_end - frame_start;

                    let frame_f64: Vec<f64> = audio
                        [start_sample + frame_start..start_sample + frame_end]
                        .iter()
                        .enumerate()
                        .map(|(i, &s)| {
                            let w = Self::hann_coeff(i, actual_len);
                            s as f64 * w
                        })
                        .collect();

                    let fft_size = if actual_len == FRAME_LEN {
                        None // use the natural length
                    } else {
                        Some(actual_len)
                    };

                    if let Ok(mut spectrum) = scirs2_fft::rfft(&frame_f64, fft_size) {
                        let fft_len = fft_size.unwrap_or(actual_len);
                        let bin_hz = SAMPLE_RATE / fft_len as f64;
                        for (k, c) in spectrum.iter_mut().enumerate() {
                            let freq = k as f64 * bin_hz;
                            if freq >= boost_lo_hz && freq <= boost_hi_hz {
                                c.re *= boost_gain;
                                c.im *= boost_gain;
                            }
                        }

                        if let Ok(time_out) =
                            scirs2_fft::irfft(spectrum.as_slice(), Some(actual_len))
                        {
                            let norm_scale = actual_len as f64;
                            for (i, &s) in time_out.iter().enumerate() {
                                let w = Self::hann_coeff(i, actual_len);
                                let out_idx = frame_start + i;
                                if out_idx < region_len {
                                    ola_buf[out_idx] += s * w / norm_scale;
                                    norm_buf[out_idx] += w * w;
                                }
                            }
                        }
                    }

                    if frame_start + HOP >= region_len {
                        break;
                    }
                    frame_start += HOP;
                }

                // Write OLA result back to processed buffer
                for i in 0..region_len {
                    let norm = if norm_buf[i] > 1e-12 {
                        norm_buf[i]
                    } else {
                        1.0
                    };
                    let idx = start_sample + i;
                    if idx < n_audio {
                        processed[idx] = (ola_buf[i] / norm) as f32;
                    }
                }
                // suppress unused import warning
                let _: Option<Complex64> = None;
            }
        }

        // Apply intelligibility boost: global clamp pass
        if params.intelligibility_boost > 1.0 {
            let boost_factor = (params.intelligibility_boost - 1.0) * 0.1;
            for sample in &mut processed {
                if sample.abs() > 0.01 {
                    *sample *= 1.0 + boost_factor;
                    *sample = sample.clamp(-1.0, 1.0);
                }
            }
        }

        processed
    }

    /// Hann window coefficient for sample index `i` in a frame of length `n`.
    #[inline]
    fn hann_coeff(i: usize, n: usize) -> f64 {
        if n <= 1 {
            return 1.0;
        }
        0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())
    }

    /// Check if a phoneme is a vowel
    fn is_vowel_phoneme(&self, phoneme: &str) -> bool {
        // Simplified vowel detection - in real implementation would use proper phoneme classification
        let vowel_patterns = [
            "a", "e", "i", "o", "u", "aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey", "ih",
            "iy", "ow", "oy", "uh", "uw",
        ];
        vowel_patterns
            .iter()
            .any(|&v| phoneme.to_lowercase().contains(v))
    }

    /// Calculate quality metrics for guided conversion
    fn calculate_guided_quality_metrics(
        &self,
        converted_audio: &[f32],
        original_audio: &[f32],
        transcription: &ASRTranscription,
    ) -> Result<QualityMetrics, Error> {
        // Calculate standard quality metrics
        let mut metrics = QualityMetrics {
            similarity: 0.8,
            naturalness: 0.8,
            conversion_strength: 0.8,
        };

        // Speech-specific quality assessments
        let intelligibility_score =
            self.calculate_intelligibility_score(converted_audio, transcription);
        let prosody_preservation =
            self.calculate_prosody_preservation(converted_audio, original_audio, transcription);
        let phoneme_clarity = self.calculate_phoneme_clarity(converted_audio, transcription);

        // Update metrics with speech-specific scores
        metrics.similarity = (intelligibility_score + prosody_preservation) / 2.0;
        metrics.naturalness = phoneme_clarity;
        metrics.conversion_strength =
            (intelligibility_score + prosody_preservation + phoneme_clarity) / 3.0;

        Ok(metrics)
    }

    /// Calculate intelligibility preservation score
    fn calculate_intelligibility_score(
        &self,
        audio: &[f32],
        transcription: &ASRTranscription,
    ) -> f32 {
        // Simulate intelligibility measurement based on audio clarity and transcription confidence
        let audio_clarity = self.calculate_audio_clarity(audio);
        let transcription_confidence = transcription.confidence;

        // Combine clarity and confidence
        (audio_clarity + transcription_confidence) / 2.0
    }

    /// Calculate prosody preservation score
    fn calculate_prosody_preservation(
        &self,
        converted: &[f32],
        original: &[f32],
        transcription: &ASRTranscription,
    ) -> f32 {
        // Analyze prosodic features preservation
        let original_rhythm = self.calculate_rhythm_pattern(original, transcription);
        let converted_rhythm = self.calculate_rhythm_pattern(converted, transcription);

        // Calculate rhythm similarity
        self.calculate_pattern_correlation(&original_rhythm, &converted_rhythm)
    }

    /// Calculate phoneme clarity score
    fn calculate_phoneme_clarity(&self, audio: &[f32], transcription: &ASRTranscription) -> f32 {
        if transcription.phoneme_alignment.is_empty() {
            return 0.8; // Default score when no phoneme data
        }

        // Analyze clarity for each phoneme
        let mut total_clarity = 0.0;
        for phoneme in &transcription.phoneme_alignment {
            let clarity = self.calculate_phoneme_specific_clarity(audio, phoneme);
            total_clarity += clarity;
        }

        total_clarity / transcription.phoneme_alignment.len() as f32
    }

    /// Calculate clarity for a specific phoneme
    fn calculate_phoneme_specific_clarity(&self, audio: &[f32], phoneme: &PhonemeAlignment) -> f32 {
        // Get audio segment for this phoneme
        let sample_rate = 16000.0;
        let start_sample = (phoneme.start_time * sample_rate) as usize;
        let end_sample = (phoneme.end_time * sample_rate) as usize;

        if start_sample >= audio.len() || end_sample > audio.len() {
            return 0.5; // Default for out-of-bounds
        }

        let phoneme_audio = &audio[start_sample..end_sample];

        // Calculate clarity based on energy and spectral characteristics
        let energy = phoneme_audio.iter().map(|&s| s * s).sum::<f32>() / phoneme_audio.len() as f32;
        let clarity = if self.is_vowel_phoneme(&phoneme.phoneme) {
            // Vowels should have sustained energy
            energy.sqrt() * 2.0
        } else {
            // Consonants should have clear transitions
            let energy_variance = self.calculate_energy_variance(phoneme_audio);
            energy_variance * 1.5
        };

        clarity.min(1.0)
    }

    /// Calculate overall quality score
    fn calculate_overall_quality_score(
        &self,
        metrics: &QualityMetrics,
        transcription: &ASRTranscription,
    ) -> f32 {
        // Weight factors for overall quality
        let intelligibility_weight = 0.4;
        let similarity_weight = 0.3;
        let naturalness_weight = 0.2;
        let confidence_weight = 0.1;

        intelligibility_weight * metrics.similarity
            + similarity_weight * metrics.similarity
            + naturalness_weight * metrics.naturalness
            + confidence_weight * transcription.confidence
    }

    /// Update processing statistics
    fn update_stats(
        &mut self,
        transcription: &ASRTranscription,
        asr_time: f32,
        conversion_time: f32,
        total_time: f32,
        metrics: &QualityMetrics,
    ) {
        self.stats.asr_processing_time = asr_time;
        self.stats.conversion_time = conversion_time;
        self.stats.total_time = total_time;
        self.stats.phonemes_processed = transcription.phoneme_alignment.len();
        self.stats.words_recognized = transcription.word_timestamps.len();
        self.stats.avg_word_confidence = if transcription.word_timestamps.is_empty() {
            transcription.confidence
        } else {
            transcription
                .word_timestamps
                .iter()
                .map(|w| w.confidence)
                .sum::<f32>()
                / transcription.word_timestamps.len() as f32
        };
        self.stats.intelligibility_score = metrics.similarity;
    }

    /// Calculate audio hash for caching
    fn calculate_audio_hash(&self, audio: &[f32]) -> String {
        // Simple hash based on audio characteristics
        let energy = audio.iter().map(|&s| s.abs()).sum::<f32>();
        let length = audio.len();
        format!("audio_{}_{}", length, (energy * 1000.0) as u32)
    }

    /// Calculate audio clarity measure
    fn calculate_audio_clarity(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        // Calculate signal-to-noise ratio estimation
        let signal_energy = audio.iter().map(|&s| s * s).sum::<f32>() / audio.len() as f32;
        let noise_floor = 0.01; // Estimated noise floor
        let snr = (signal_energy / noise_floor).log10() * 10.0;

        // Convert SNR to clarity score (0.0-1.0)
        (snr / 40.0).clamp(0.0, 1.0)
    }

    /// Calculate rhythm pattern from audio and transcription
    fn calculate_rhythm_pattern(
        &self,
        audio: &[f32],
        transcription: &ASRTranscription,
    ) -> Vec<f32> {
        let mut rhythm_pattern = Vec::new();

        for word in &transcription.word_timestamps {
            let duration = word.end_time - word.start_time;
            rhythm_pattern.push(duration);
        }

        rhythm_pattern
    }

    /// Calculate correlation between two patterns
    fn calculate_pattern_correlation(&self, pattern1: &[f32], pattern2: &[f32]) -> f32 {
        if pattern1.is_empty() || pattern2.is_empty() || pattern1.len() != pattern2.len() {
            return 0.5; // Default correlation
        }

        let mean1 = pattern1.iter().sum::<f32>() / pattern1.len() as f32;
        let mean2 = pattern2.iter().sum::<f32>() / pattern2.len() as f32;

        let mut numerator = 0.0;
        let mut sum_sq1 = 0.0;
        let mut sum_sq2 = 0.0;

        for i in 0..pattern1.len() {
            let diff1 = pattern1[i] - mean1;
            let diff2 = pattern2[i] - mean2;
            numerator += diff1 * diff2;
            sum_sq1 += diff1 * diff1;
            sum_sq2 += diff2 * diff2;
        }

        if sum_sq1 == 0.0 || sum_sq2 == 0.0 {
            return 0.5;
        }

        let correlation = numerator / (sum_sq1 * sum_sq2).sqrt();
        (correlation + 1.0) / 2.0 // Convert from [-1,1] to [0,1]
    }

    /// Calculate energy variance for phoneme clarity
    fn calculate_energy_variance(&self, audio: &[f32]) -> f32 {
        if audio.len() < 2 {
            return 0.5;
        }

        let energies: Vec<f32> = audio
            .windows(100)
            .map(|window| window.iter().map(|&s| s * s).sum::<f32>() / window.len() as f32)
            .collect();

        if energies.is_empty() {
            return 0.5;
        }

        let mean_energy = energies.iter().sum::<f32>() / energies.len() as f32;
        let variance = energies
            .iter()
            .map(|&e| (e - mean_energy).powi(2))
            .sum::<f32>()
            / energies.len() as f32;

        variance.sqrt() * 10.0 // Scale to reasonable range
    }

    /// Get current conversion statistics
    pub fn get_stats(&self) -> &RecognitionStats {
        &self.stats
    }

    /// Clear transcription cache
    pub fn clear_cache(&mut self) {
        self.transcription_cache.clear();
    }

    /// Get number of cached transcriptions
    pub fn cache_size(&self) -> usize {
        self.transcription_cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asr_config_creation() {
        let config = ASRConfig::default();
        assert_eq!(config.engine, ASREngine::Whisper);
        assert_eq!(config.language, "en-US");
        assert_eq!(config.confidence_threshold, 0.7);
        assert!(config.phoneme_alignment);
    }

    #[test]
    fn test_asr_config_builder() {
        let config = ASRConfig::default()
            .with_language("es-ES")
            .with_confidence_threshold(0.8)
            .with_realtime_threshold(0.3)
            .with_phoneme_alignment(false);

        assert_eq!(config.language, "es-ES");
        assert_eq!(config.confidence_threshold, 0.8);
        assert_eq!(config.realtime_threshold, 0.3);
        assert!(!config.phoneme_alignment);
    }

    #[test]
    fn test_recognition_guided_converter_creation() {
        let config = ASRConfig::default();
        let converter = RecognitionGuidedConverter::new(config);
        assert!(converter.is_ok());
    }

    #[test]
    fn test_speech_guided_params_default() {
        let params = SpeechGuidedParams::default();
        assert_eq!(params.vowel_emphasis, 0.8);
        assert_eq!(params.consonant_clarity, 0.9);
        assert_eq!(params.prosody_weight, 0.7);
        assert_eq!(params.intelligibility_boost, 1.2);
        assert!(params.phoneme_specific);
    }

    #[test]
    fn test_asr_engine_types() {
        let engines = vec![
            ASREngine::Whisper,
            ASREngine::DeepSpeech,
            ASREngine::Wav2Vec2,
            ASREngine::Custom("CustomASR".to_string()),
        ];

        assert_eq!(engines.len(), 4);
        assert_eq!(engines[3], ASREngine::Custom("CustomASR".to_string()));
    }

    #[test]
    fn test_word_timestamp_creation() {
        let word = WordTimestamp {
            word: "hello".to_string(),
            start_time: 0.0,
            end_time: 0.5,
            confidence: 0.95,
        };

        assert_eq!(word.word, "hello");
        assert_eq!(word.start_time, 0.0);
        assert_eq!(word.end_time, 0.5);
        assert_eq!(word.confidence, 0.95);
    }

    #[test]
    fn test_phoneme_alignment_creation() {
        let phoneme = PhonemeAlignment {
            phoneme: "h".to_string(),
            start_time: 0.0,
            end_time: 0.1,
            confidence: 0.9,
        };

        assert_eq!(phoneme.phoneme, "h");
        assert_eq!(phoneme.start_time, 0.0);
        assert_eq!(phoneme.end_time, 0.1);
        assert_eq!(phoneme.confidence, 0.9);
    }

    #[test]
    fn test_recognition_stats_default() {
        let stats = RecognitionStats::default();
        assert_eq!(stats.asr_processing_time, 0.0);
        assert_eq!(stats.conversion_time, 0.0);
        assert_eq!(stats.total_time, 0.0);
        assert_eq!(stats.phonemes_processed, 0);
        assert_eq!(stats.words_recognized, 0);
        assert_eq!(stats.avg_word_confidence, 0.0);
        assert_eq!(stats.intelligibility_score, 0.0);
    }

    #[test]
    fn test_vowel_phoneme_detection() {
        let config = ASRConfig::default();
        let converter = RecognitionGuidedConverter::new(config).unwrap();

        assert!(converter.is_vowel_phoneme("aa"));
        assert!(converter.is_vowel_phoneme("eh"));
        assert!(converter.is_vowel_phoneme("iy"));
        assert!(!converter.is_vowel_phoneme("b"));
        assert!(!converter.is_vowel_phoneme("k"));
        assert!(!converter.is_vowel_phoneme("s"));
    }

    #[test]
    fn test_audio_hash_calculation() {
        let config = ASRConfig::default();
        let converter = RecognitionGuidedConverter::new(config).unwrap();

        let audio1 = vec![0.1, 0.2, 0.3];
        let audio2 = vec![0.1, 0.2, 0.3];
        let audio3 = vec![0.4, 0.5, 0.6];

        let hash1 = converter.calculate_audio_hash(&audio1);
        let hash2 = converter.calculate_audio_hash(&audio2);
        let hash3 = converter.calculate_audio_hash(&audio3);

        assert_eq!(hash1, hash2); // Same audio should have same hash
        assert_ne!(hash1, hash3); // Different audio should have different hash
    }

    #[test]
    fn test_simulated_transcription() {
        let config = ASRConfig::default();
        let mut converter = RecognitionGuidedConverter::new(config).unwrap();

        let audio = vec![0.1; 16000]; // 1 second of audio at 16kHz
        let result = converter.transcribe_audio(&audio);

        assert!(result.is_ok());
        let transcription = result.unwrap();
        assert!(!transcription.text.is_empty());
        assert!(transcription.confidence > 0.0);
        assert!(!transcription.word_timestamps.is_empty());
    }

    #[test]
    fn test_cache_functionality() {
        let config = ASRConfig::default();
        let mut converter = RecognitionGuidedConverter::new(config).unwrap();

        assert_eq!(converter.cache_size(), 0);

        let audio = vec![0.1; 1000];
        let _ = converter.transcribe_audio(&audio);

        assert_eq!(converter.cache_size(), 1);

        converter.clear_cache();
        assert_eq!(converter.cache_size(), 0);
    }

    // -----------------------------------------------------------------
    // FFT-domain spectral shaping + real-timing + quality tests
    // -----------------------------------------------------------------

    #[test]
    fn test_spectral_shaping_output_length() {
        // sine wave → output same length as input
        let sr = 16000_u32;
        let samples: Vec<f32> = (0..sr)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();
        let asr_config = ASRConfig::default();
        let converter = RecognitionGuidedConverter::new(asr_config).unwrap();
        // Use empty transcription — should be a no-op through phoneme loop
        let transcription = ASRTranscription {
            confidence: 0.9,
            text: "".to_string(),
            word_timestamps: vec![],
            phoneme_alignment: vec![],
            audio_duration: 1.0,
        };
        let params = SpeechGuidedParams::default();
        let out = converter.apply_speech_guided_processing(&samples, &transcription, &params);
        assert_eq!(out.len(), samples.len());
    }

    #[test]
    fn test_real_timing_not_fixed() {
        // processing_time must not be exactly 50 ms
        let asr_config = ASRConfig::default();
        let converter = RecognitionGuidedConverter::new(asr_config).unwrap();
        let audio = vec![0.0f32; 1600];
        let transcription = ASRTranscription {
            confidence: 0.9,
            text: "".to_string(),
            word_timestamps: vec![],
            phoneme_alignment: vec![],
            audio_duration: 0.1,
        };
        let params = SpeechGuidedParams::default();
        let request = ConversionRequest::new(
            "test".to_string(),
            audio,
            16000,
            ConversionType::SpeakerConversion,
            ConversionTarget::new(VoiceCharacteristics::new()),
        );
        let result = converter.simulate_speech_guided_conversion(&request, &transcription, &params);
        assert_ne!(result.processing_time, std::time::Duration::from_millis(50));
    }

    #[test]
    fn test_overall_quality_in_range() {
        let asr_config = ASRConfig::default();
        let converter = RecognitionGuidedConverter::new(asr_config).unwrap();
        let sr = 16000_u32;
        let audio: Vec<f32> = (0..sr)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();
        let transcription = ASRTranscription {
            confidence: 0.9,
            text: "test".to_string(),
            word_timestamps: vec![],
            phoneme_alignment: vec![],
            audio_duration: 1.0,
        };
        let params = SpeechGuidedParams::default();
        let request = ConversionRequest::new(
            "test".to_string(),
            audio,
            16000,
            ConversionType::SpeakerConversion,
            ConversionTarget::new(VoiceCharacteristics::new()),
        );
        let result = converter.simulate_speech_guided_conversion(&request, &transcription, &params);
        let q = result.quality_metrics["overall_quality"];
        assert!(q >= 0.0 && q <= 1.0, "quality {q} out of [0,1]");
    }

    // -----------------------------------------------------------------
    // Real audio-analysis tests
    // -----------------------------------------------------------------

    #[test]
    fn test_generate_words_silence_returns_empty() {
        // Pure silence must yield no voiced segments
        let converter = RecognitionGuidedConverter::new(ASRConfig::default()).unwrap();
        let silence = vec![0.0f32; 16000];
        let words = converter.generate_simulated_words(&silence, 1.0);
        assert!(
            words.is_empty(),
            "Expected no words for silence, got {}",
            words.len()
        );
    }

    #[test]
    fn test_generate_words_tone_detected() {
        let converter = RecognitionGuidedConverter::new(ASRConfig::default()).unwrap();
        // 440 Hz sine at 16 kHz, 0.5 s — must produce at least one voiced segment
        let samples: Vec<f32> = (0..8000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let words = converter.generate_simulated_words(&samples, 0.5);
        assert!(
            !words.is_empty(),
            "Expected voiced segments for a 440 Hz tone"
        );
        for w in &words {
            assert!(
                w.start_time < w.end_time,
                "start_time must precede end_time"
            );
            assert!(
                (0.4..=1.0).contains(&w.confidence),
                "confidence out of range: {}",
                w.confidence
            );
        }
    }

    #[test]
    fn test_phoneme_alignment_covers_word_span() {
        let converter = RecognitionGuidedConverter::new(ASRConfig::default()).unwrap();
        let words = vec![WordTimestamp {
            word: "a-1".to_string(),
            start_time: 0.0,
            end_time: 0.3,
            confidence: 0.8,
        }];
        let phonemes = converter.generate_simulated_phonemes(&words);
        assert!(!phonemes.is_empty(), "Expected at least one phoneme unit");
        let span_start = phonemes.first().unwrap().start_time;
        let span_end = phonemes.last().unwrap().end_time;
        assert!(
            (span_start - 0.0).abs() < 0.01,
            "Phoneme span must start at word start"
        );
        assert!(
            (span_end - 0.3).abs() < 0.1,
            "Phoneme span must end near word end, got {}",
            span_end
        );
    }
}
