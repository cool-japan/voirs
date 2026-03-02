//! Zero-copy streaming pipeline for ultra-low latency audio synthesis
//!
//! Combines lock-free ring buffers, predictive synthesis, and adaptive quality scaling
//! to achieve <10ms latency in real-time singing synthesis.

use super::{
    adaptive_quality::{AdaptiveQualityScaler, QualityLevel},
    cpu_monitor::CpuStats,
    lock_free_ring_buffer::LockFreeRingBuffer,
    predictive_synthesis::PredictiveSynthesisEngine,
    AdvancedStreamingConfig, StreamingMetrics,
};
use crate::{Error, MusicalNote, MusicalScore, VoiceCharacteristics};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Zero-copy streaming pipeline
///
/// Ultra-low latency streaming synthesis using:
/// - Lock-free ring buffer for zero-copy audio transfer
/// - Predictive synthesis for upcoming notes
/// - Adaptive quality scaling based on CPU load
pub struct ZeroCopyStreamingPipeline {
    /// Configuration
    config: AdvancedStreamingConfig,

    /// Lock-free audio ring buffer
    ring_buffer: LockFreeRingBuffer,

    /// Predictive synthesis engine
    predictive_engine: PredictiveSynthesisEngine,

    /// Adaptive quality scaler
    quality_scaler: AdaptiveQualityScaler,

    /// Current playback position (seconds)
    playback_position: Arc<std::sync::Mutex<f32>>,

    /// Streaming metrics
    metrics: Arc<std::sync::Mutex<StreamingMetrics>>,

    /// Musical score being streamed
    current_score: Arc<std::sync::Mutex<Option<MusicalScore>>>,

    /// Voice characteristics
    current_voice: Arc<std::sync::Mutex<Option<VoiceCharacteristics>>>,

    /// Voice identifier
    voice_id: Arc<std::sync::Mutex<String>>,

    /// Pipeline state
    state: Arc<std::sync::Mutex<PipelineState>>,
}

/// Pipeline state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineState {
    /// Pipeline is idle
    Idle,

    /// Pipeline is buffering
    Buffering,

    /// Pipeline is actively streaming
    Streaming,

    /// Pipeline is paused
    Paused,

    /// Pipeline encountered an error
    Error,
}

impl ZeroCopyStreamingPipeline {
    /// Create a new zero-copy streaming pipeline
    ///
    /// # Arguments
    /// * `config` - Streaming configuration
    pub fn new(config: AdvancedStreamingConfig) -> Self {
        let ring_buffer = LockFreeRingBuffer::new(config.ring_buffer_size);

        let predictive_engine =
            PredictiveSynthesisEngine::new(config.lookahead_time, config.chunk_count * 2);

        let quality_scaler = AdaptiveQualityScaler::new(
            config.cpu_threshold,
            QualityLevel::Minimum,
            QualityLevel::Maximum,
        );

        Self {
            config,
            ring_buffer,
            predictive_engine,
            quality_scaler,
            playback_position: Arc::new(std::sync::Mutex::new(0.0)),
            metrics: Arc::new(std::sync::Mutex::new(StreamingMetrics::default())),
            current_score: Arc::new(std::sync::Mutex::new(None)),
            current_voice: Arc::new(std::sync::Mutex::new(None)),
            voice_id: Arc::new(std::sync::Mutex::new("default".to_string())),
            state: Arc::new(std::sync::Mutex::new(PipelineState::Idle)),
        }
    }

    /// Start streaming a musical score
    ///
    /// # Arguments
    /// * `score` - Musical score to stream
    /// * `voice` - Voice characteristics to use
    /// * `voice_id` - Voice identifier
    pub fn start_streaming(
        &self,
        score: MusicalScore,
        voice: VoiceCharacteristics,
        voice_id: String,
    ) -> Result<(), Error> {
        // Update state
        *self.state.lock().expect("lock should not be poisoned") = PipelineState::Buffering;

        // Store score and voice
        *self
            .current_score
            .lock()
            .expect("lock should not be poisoned") = Some(score.clone());
        *self
            .current_voice
            .lock()
            .expect("lock should not be poisoned") = Some(voice.clone());
        *self.voice_id.lock().expect("lock should not be poisoned") = voice_id.clone();

        // Reset playback position
        *self
            .playback_position
            .lock()
            .expect("lock should not be poisoned") = 0.0;

        // Clear ring buffer
        self.ring_buffer.clear();

        // Pre-warm predictive cache
        if self.config.enable_predictive {
            let upcoming_notes = self.predictive_engine.analyze_upcoming_notes(&score, 0.0);
            self.predictive_engine
                .prewarm_cache(&upcoming_notes, &voice_id, &voice);
        }

        // Synthesize initial buffer
        self.fill_initial_buffer()?;

        // Update state
        *self.state.lock().expect("lock should not be poisoned") = PipelineState::Streaming;

        Ok(())
    }

    /// Fill initial buffer before playback starts
    fn fill_initial_buffer(&self) -> Result<(), Error> {
        let target_fill = self.config.chunk_size * 4; // Pre-buffer 4 chunks

        while self.ring_buffer.available_samples() < target_fill {
            self.synthesize_next_chunk()?;
        }

        Ok(())
    }

    /// Synthesize next audio chunk
    fn synthesize_next_chunk(&self) -> Result<(), Error> {
        let start_time = Instant::now();

        let playback_pos = *self
            .playback_position
            .lock()
            .expect("lock should not be poisoned");
        let score = self
            .current_score
            .lock()
            .expect("lock should not be poisoned")
            .clone();
        let voice = self
            .current_voice
            .lock()
            .expect("lock should not be poisoned")
            .clone();
        let voice_id = self
            .voice_id
            .lock()
            .expect("lock should not be poisoned")
            .clone();

        if score.is_none() || voice.is_none() {
            return Err(Error::Processing(
                "No score or voice set for streaming".to_string(),
            ));
        }

        let score = score.expect("operation should succeed");
        let voice = voice.expect("operation should succeed");

        // Find notes in current chunk
        let chunk_duration = self.config.chunk_size as f32 / self.config.sample_rate as f32;
        let chunk_end = playback_pos + chunk_duration;

        let notes_in_chunk: Vec<MusicalNote> = score
            .notes
            .iter()
            .filter(|note| note.start_time >= playback_pos && note.start_time < chunk_end)
            .cloned()
            .collect();

        // Synthesize chunk
        let mut chunk_samples = vec![0.0; self.config.chunk_size];

        for note in &notes_in_chunk {
            // Try to use cached note
            let note_samples = if self.config.enable_predictive {
                self.predictive_engine.get_cached_note(note, &voice_id)
            } else {
                None
            };

            let note_samples = if let Some(cached) = note_samples {
                cached
            } else {
                // Synthesize note (stub implementation)
                self.synthesize_note(note, &voice)
            };

            // Mix note into chunk
            let note_offset =
                ((note.start_time - playback_pos) * self.config.sample_rate as f32) as usize;
            for (i, &sample) in note_samples.iter().enumerate() {
                let chunk_idx = note_offset + i;
                if chunk_idx < chunk_samples.len() {
                    chunk_samples[chunk_idx] += sample;
                }
            }
        }

        // Write to ring buffer
        let written = self.ring_buffer.write(&chunk_samples);
        if written < chunk_samples.len() {
            // Buffer overrun
            if let Ok(mut metrics) = self.metrics.lock() {
                metrics.buffer_underruns += 1;
            }
        }

        // Update metrics
        let synthesis_time = start_time.elapsed();
        let frame_duration = Duration::from_secs_f32(chunk_duration);

        self.update_metrics(synthesis_time, frame_duration);

        // Update playback position
        *self
            .playback_position
            .lock()
            .expect("lock should not be poisoned") += chunk_duration;

        // Adaptive quality scaling
        if self.config.enable_adaptive_quality {
            let (quality, adjusted) = self.quality_scaler.update(synthesis_time, frame_duration);
            if adjusted {
                if let Ok(mut metrics) = self.metrics.lock() {
                    metrics.quality_adjustments += 1;
                    metrics.quality_level = quality.quality_factor();
                }
            }
        }

        // Predictive synthesis for upcoming notes
        if self.config.enable_predictive {
            let playback_pos = *self
                .playback_position
                .lock()
                .expect("lock should not be poisoned");
            let upcoming = self
                .predictive_engine
                .analyze_upcoming_notes(&score, playback_pos);
            self.predictive_engine
                .prewarm_cache(&upcoming, &voice_id, &voice);
        }

        Ok(())
    }

    /// Stub synthesis (replace with actual synthesis in production)
    fn synthesize_note(&self, note: &MusicalNote, _voice: &VoiceCharacteristics) -> Vec<f32> {
        let sample_count = (note.duration * self.config.sample_rate as f32) as usize;
        let dynamics_value = self.dynamics_to_f32(&note.dynamics);
        let samples: Vec<f32> = (0..sample_count)
            .map(|i| {
                let t = i as f32 / self.config.sample_rate as f32;
                let phase = 2.0 * std::f32::consts::PI * note.event.frequency * t;
                phase.sin() * 0.3 * dynamics_value
            })
            .collect();
        samples
    }

    /// Convert Dynamics enum to f32 amplitude
    fn dynamics_to_f32(&self, dynamics: &crate::types::Dynamics) -> f32 {
        use crate::types::Dynamics;
        match dynamics {
            Dynamics::Pianissimo => 0.2,
            Dynamics::Piano => 0.4,
            Dynamics::MezzoPiano => 0.6,
            Dynamics::MezzoForte => 0.7,
            Dynamics::Forte => 0.85,
            Dynamics::Fortissimo => 1.0,
            Dynamics::Crescendo => 0.8,
            Dynamics::Diminuendo => 0.5,
        }
    }

    /// Read audio samples from the pipeline
    ///
    /// # Arguments
    /// * `output` - Buffer to fill with audio samples
    ///
    /// # Returns
    /// Number of samples read
    pub fn read_samples(&self, output: &mut [f32]) -> Result<usize, Error> {
        let read = self.ring_buffer.read(output);

        // Check if we need to synthesize more
        if self.ring_buffer.available_samples() < self.config.chunk_size {
            if let PipelineState::Streaming =
                *self.state.lock().expect("lock should not be poisoned")
            {
                self.synthesize_next_chunk()?;
            }
        }

        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.frames_synthesized += 1;
            metrics.total_samples += read as u64;
        }

        Ok(read)
    }

    /// Update streaming metrics
    fn update_metrics(&self, synthesis_time: Duration, frame_duration: Duration) {
        if let Ok(mut metrics) = self.metrics.lock() {
            let latency_ms = synthesis_time.as_secs_f32() * 1000.0;

            metrics.current_latency_ms = latency_ms;

            // Update average latency (exponential moving average)
            if metrics.avg_latency_ms == 0.0 {
                metrics.avg_latency_ms = latency_ms;
            } else {
                metrics.avg_latency_ms = metrics.avg_latency_ms * 0.9 + latency_ms * 0.1;
            }

            // Update peak latency
            if latency_ms > metrics.peak_latency_ms {
                metrics.peak_latency_ms = latency_ms;
            }

            // Calculate jitter (simplified)
            let jitter = (latency_ms - metrics.avg_latency_ms).abs();
            metrics.jitter_ms = metrics.jitter_ms * 0.9 + jitter * 0.1;

            // Update CPU usage
            let cpu_usage =
                (synthesis_time.as_secs_f32() / frame_duration.as_secs_f32()).clamp(0.0, 1.0);
            metrics.cpu_usage = cpu_usage;

            // Update predictive stats
            let pred_stats = self.predictive_engine.get_stats();
            metrics.predictive_hits = pred_stats.cache_hits;
            metrics.predictive_misses = pred_stats.cache_misses;
        }
    }

    /// Pause streaming
    pub fn pause(&self) {
        *self.state.lock().expect("lock should not be poisoned") = PipelineState::Paused;
    }

    /// Resume streaming
    pub fn resume(&self) -> Result<(), Error> {
        let state = *self.state.lock().expect("lock should not be poisoned");
        if state == PipelineState::Paused {
            *self.state.lock().expect("lock should not be poisoned") = PipelineState::Streaming;
            Ok(())
        } else {
            Err(Error::Processing(
                "Cannot resume from current state".to_string(),
            ))
        }
    }

    /// Stop streaming
    pub fn stop(&self) {
        *self.state.lock().expect("lock should not be poisoned") = PipelineState::Idle;
        self.ring_buffer.clear();
        self.predictive_engine.clear_cache();
    }

    /// Get current streaming metrics
    pub fn get_metrics(&self) -> StreamingMetrics {
        self.metrics
            .lock()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Get current pipeline state
    pub fn get_state(&self) -> PipelineState {
        *self.state.lock().expect("lock should not be poisoned")
    }

    /// Get buffer fill percentage
    pub fn buffer_fill_percentage(&self) -> f32 {
        self.ring_buffer.fill_percentage()
    }

    /// Get CPU statistics
    pub fn get_cpu_stats(&self) -> CpuStats {
        self.quality_scaler.get_cpu_stats()
    }

    /// Get current quality level
    pub fn get_quality_level(&self) -> QualityLevel {
        self.quality_scaler.get_quality()
    }

    /// Manually set quality level
    pub fn set_quality_level(&self, quality: QualityLevel) {
        self.quality_scaler.set_quality(quality);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_score() -> MusicalScore {
        use crate::types::{Articulation, Dynamics, NoteEvent};

        let mut score = MusicalScore::new("Test".to_string(), "Composer".to_string());

        score.notes.push(MusicalNote {
            event: NoteEvent {
                note: "A".to_string(),
                octave: 4,
                frequency: 440.0,
                duration: 0.5,
                velocity: 0.8,
                vibrato: 0.5,
                lyric: Some("a".to_string()),
                phonemes: vec!["a".to_string()],
                expression: crate::types::Expression::Neutral,
                timing_offset: 0.0,
                breath_before: 0.0,
                legato: false,
                articulation: Articulation::Normal,
            },
            start_time: 0.0,
            duration: 0.5,
            pitch_bend: None,
            articulation: Articulation::Normal,
            dynamics: Dynamics::MezzoForte,
            tie_next: false,
            tie_prev: false,
            tuplet: None,
            ornaments: Vec::new(),
            chord: None,
        });

        score.notes.push(MusicalNote {
            event: NoteEvent {
                note: "B".to_string(),
                octave: 4,
                frequency: 494.0,
                duration: 0.5,
                velocity: 0.8,
                vibrato: 0.5,
                lyric: Some("e".to_string()),
                phonemes: vec!["e".to_string()],
                expression: crate::types::Expression::Neutral,
                timing_offset: 0.0,
                breath_before: 0.0,
                legato: false,
                articulation: Articulation::Normal,
            },
            start_time: 0.5,
            duration: 0.5,
            pitch_bend: None,
            articulation: Articulation::Normal,
            dynamics: Dynamics::MezzoForte,
            tie_next: false,
            tie_prev: false,
            tuplet: None,
            ornaments: Vec::new(),
            chord: None,
        });

        score
    }

    #[test]
    fn test_pipeline_creation() {
        let config = AdvancedStreamingConfig::default();
        let pipeline = ZeroCopyStreamingPipeline::new(config);
        assert_eq!(pipeline.get_state(), PipelineState::Idle);
    }

    #[test]
    fn test_start_streaming() {
        let config = AdvancedStreamingConfig::default();
        let pipeline = ZeroCopyStreamingPipeline::new(config);

        let score = create_test_score();
        let voice = VoiceCharacteristics::default();

        let result = pipeline.start_streaming(score, voice, "voice1".to_string());
        assert!(result.is_ok());
        assert_eq!(pipeline.get_state(), PipelineState::Streaming);
    }

    #[test]
    fn test_read_samples() {
        let config = AdvancedStreamingConfig::default();
        let pipeline = ZeroCopyStreamingPipeline::new(config);

        let score = create_test_score();
        let voice = VoiceCharacteristics::default();

        pipeline
            .start_streaming(score, voice, "voice1".to_string())
            .unwrap();

        let mut output = vec![0.0; 1024];
        let read = pipeline.read_samples(&mut output);
        assert!(read.is_ok());
        assert!(read.unwrap() > 0);
    }

    #[test]
    fn test_pause_resume() {
        let config = AdvancedStreamingConfig::default();
        let pipeline = ZeroCopyStreamingPipeline::new(config);

        let score = create_test_score();
        let voice = VoiceCharacteristics::default();

        pipeline
            .start_streaming(score, voice, "voice1".to_string())
            .unwrap();

        pipeline.pause();
        assert_eq!(pipeline.get_state(), PipelineState::Paused);

        pipeline.resume().unwrap();
        assert_eq!(pipeline.get_state(), PipelineState::Streaming);
    }

    #[test]
    fn test_stop_streaming() {
        let config = AdvancedStreamingConfig::default();
        let pipeline = ZeroCopyStreamingPipeline::new(config);

        let score = create_test_score();
        let voice = VoiceCharacteristics::default();

        pipeline
            .start_streaming(score, voice, "voice1".to_string())
            .unwrap();
        pipeline.stop();

        assert_eq!(pipeline.get_state(), PipelineState::Idle);
        assert_eq!(pipeline.buffer_fill_percentage(), 0.0);
    }

    #[test]
    fn test_metrics_collection() {
        let config = AdvancedStreamingConfig::default();
        let pipeline = ZeroCopyStreamingPipeline::new(config);

        let score = create_test_score();
        let voice = VoiceCharacteristics::default();

        pipeline
            .start_streaming(score, voice, "voice1".to_string())
            .unwrap();

        let mut output = vec![0.0; 1024];
        pipeline.read_samples(&mut output).unwrap();

        let metrics = pipeline.get_metrics();
        assert!(metrics.total_samples > 0);
        assert!(metrics.frames_synthesized > 0);
    }

    #[test]
    fn test_quality_level_control() {
        let config = AdvancedStreamingConfig::default();
        let pipeline = ZeroCopyStreamingPipeline::new(config);

        pipeline.set_quality_level(QualityLevel::Medium);
        assert_eq!(pipeline.get_quality_level(), QualityLevel::Medium);
    }

    #[test]
    fn test_buffer_fill_percentage() {
        let config = AdvancedStreamingConfig::default();
        let pipeline = ZeroCopyStreamingPipeline::new(config);

        let score = create_test_score();
        let voice = VoiceCharacteristics::default();

        pipeline
            .start_streaming(score, voice, "voice1".to_string())
            .unwrap();

        let fill = pipeline.buffer_fill_percentage();
        assert!(fill >= 0.0 && fill <= 1.0);
    }

    #[test]
    fn test_predictive_synthesis_enabled() {
        let mut config = AdvancedStreamingConfig::default();
        config.enable_predictive = true;

        let pipeline = ZeroCopyStreamingPipeline::new(config);

        let score = create_test_score();
        let voice = VoiceCharacteristics::default();

        pipeline
            .start_streaming(score, voice, "voice1".to_string())
            .unwrap();

        let metrics = pipeline.get_metrics();
        // Should have some predictive hits after pre-warming
        assert!(metrics.predictive_hits > 0 || metrics.predictive_misses > 0);
    }

    #[test]
    fn test_adaptive_quality_enabled() {
        let mut config = AdvancedStreamingConfig::default();
        config.enable_adaptive_quality = true;

        let pipeline = ZeroCopyStreamingPipeline::new(config);

        let score = create_test_score();
        let voice = VoiceCharacteristics::default();

        pipeline
            .start_streaming(score, voice, "voice1".to_string())
            .unwrap();

        // Read some samples to trigger quality adjustments
        for _ in 0..10 {
            let mut output = vec![0.0; 1024];
            let _ = pipeline.read_samples(&mut output);
        }

        let quality = pipeline.get_quality_level();
        assert!(quality as u8 >= QualityLevel::Minimum as u8);
        assert!(quality as u8 <= QualityLevel::Maximum as u8);
    }
}
