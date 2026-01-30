//! Streaming emotion control system for real-time synthesis
//!
//! This module provides streaming-based emotion control with:
//! - Multi-session streaming support with concurrent session management
//! - Real-time audio chunk processing with emotion adaptation
//! - Low-latency emotion updates optimized for streaming scenarios
//! - Session-based emotion state tracking and interpolation
//! - Performance metrics and adaptive quality control

use crate::{
    interpolation::EmotionInterpolator,
    types::{EmotionParameters, EmotionVector},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, trace};

// Import from parent realtime module (realtime.rs)
// These types are defined in the parent realtime module and used by the streaming controller
use super::{EmotionSignal, RealtimeEmotionAdapter, RealtimeEmotionConfig, StreamingConfig};

/// Streaming emotion control system for real-time synthesis
pub struct StreamingEmotionController {
    /// Real-time emotion adapter
    adapter: Arc<RwLock<RealtimeEmotionAdapter>>,
    /// Audio buffer for processing
    audio_buffer: VecDeque<f32>,
    /// Streaming configuration
    config: StreamingConfig,
    /// Emotion signal sender for external control
    emotion_sender: Option<mpsc::UnboundedSender<EmotionSignal>>,
    /// Active streaming sessions
    active_sessions: Arc<RwLock<HashMap<String, StreamingSession>>>,
    /// Performance metrics
    stream_metrics: StreamingMetrics,
}

/// Individual streaming session
#[derive(Debug, Clone)]
pub struct StreamingSession {
    /// Session identifier
    pub session_id: String,
    /// Current emotion state for this session
    pub current_emotion: EmotionParameters,
    /// Target emotion for this session
    pub target_emotion: EmotionParameters,
    /// Session start time
    pub start_time: Instant,
    /// Last activity time
    pub last_activity: Instant,
    /// Audio chunks processed
    pub chunks_processed: u64,
    /// Session-specific configuration overrides
    pub config_overrides: Option<StreamingConfig>,
    /// Session priority (higher values get more processing resources)
    pub priority: u8,
}

/// Streaming performance metrics
#[derive(Debug, Clone)]
pub struct StreamingMetrics {
    /// Total audio chunks processed
    pub chunks_processed: u64,
    /// Average processing latency per chunk
    pub avg_processing_latency_ms: f32,
    /// Active session count
    pub active_sessions: u32,
    /// Total sessions created
    pub total_sessions_created: u64,
    /// Emotion updates per second
    pub emotion_updates_per_sec: f32,
    /// Buffer underruns count
    pub buffer_underruns: u64,
    /// Last metrics update time
    pub last_update: Instant,
}

impl Default for StreamingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingMetrics {
    /// Create a new `StreamingMetrics` instance with default values
    ///
    /// # Returns
    /// A new metrics instance with all counters set to zero and current timestamp
    pub fn new() -> Self {
        Self {
            chunks_processed: 0,
            avg_processing_latency_ms: 0.0,
            active_sessions: 0,
            total_sessions_created: 0,
            emotion_updates_per_sec: 0.0,
            buffer_underruns: 0,
            last_update: Instant::now(),
        }
    }
}

impl StreamingEmotionController {
    /// Create new streaming emotion controller
    pub async fn new(
        realtime_config: RealtimeEmotionConfig,
        streaming_config: StreamingConfig,
    ) -> Result<Self> {
        let (adapter, emotion_sender) =
            RealtimeEmotionAdapter::new(realtime_config)?.with_signal_input();

        Ok(Self {
            adapter: Arc::new(RwLock::new(adapter)),
            audio_buffer: VecDeque::with_capacity(streaming_config.buffer_size),
            config: streaming_config,
            emotion_sender: Some(emotion_sender),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            stream_metrics: StreamingMetrics::new(),
        })
    }

    /// Start a new streaming session
    pub async fn start_session(&mut self, session_id: String) -> Result<()> {
        let mut sessions = self.active_sessions.write().await;

        if sessions.len() >= self.config.max_concurrent_sessions {
            return Err(Error::Processing(format!(
                "Maximum concurrent sessions ({}) reached",
                self.config.max_concurrent_sessions
            )));
        }

        let session = StreamingSession {
            session_id: session_id.clone(),
            current_emotion: EmotionParameters::neutral(),
            target_emotion: EmotionParameters::neutral(),
            start_time: Instant::now(),
            last_activity: Instant::now(),
            chunks_processed: 0,
            config_overrides: None,
            priority: 0,
        };

        sessions.insert(session_id, session);
        self.stream_metrics.total_sessions_created += 1;
        self.stream_metrics.active_sessions = sessions.len() as u32;

        debug!("Started streaming session: {}", sessions.len());
        Ok(())
    }

    /// Stop a streaming session
    pub async fn stop_session(&mut self, session_id: &str) -> Result<()> {
        let mut sessions = self.active_sessions.write().await;

        if sessions.remove(session_id).is_some() {
            self.stream_metrics.active_sessions = sessions.len() as u32;
            debug!("Stopped streaming session: {}", session_id);
            Ok(())
        } else {
            Err(Error::Processing(format!(
                "Session not found: {}",
                session_id
            )))
        }
    }

    /// Update emotion for a specific session
    pub async fn update_emotion(
        &mut self,
        session_id: &str,
        target_emotion: EmotionParameters,
    ) -> Result<()> {
        let mut sessions = self.active_sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.target_emotion = target_emotion;
            session.last_activity = Instant::now();
            Ok(())
        } else {
            Err(Error::Processing(format!(
                "Session not found: {}",
                session_id
            )))
        }
    }

    /// Get current emotion parameters for the adapter
    pub async fn get_current_params(&self) -> Result<EmotionParameters> {
        let adapter = self.adapter.read().await;
        Ok(adapter.get_current_emotion())
    }

    /// Process streaming audio chunk with emotion control
    pub async fn process_audio_chunk(
        &mut self,
        session_id: &str,
        audio_chunk: &[f32],
        target_emotion: Option<EmotionParameters>,
    ) -> Result<Vec<f32>> {
        let start_time = Instant::now();

        // Update session activity and emotion
        {
            let mut sessions = self.active_sessions.write().await;
            if let Some(session) = sessions.get_mut(session_id) {
                session.last_activity = Instant::now();
                session.chunks_processed += 1;

                if let Some(emotion) = target_emotion {
                    session.target_emotion = emotion;
                }
            } else {
                return Err(Error::Processing(format!(
                    "Session not found: {}",
                    session_id
                )));
            }
        }

        // Process audio with emotion adaptation
        let mut processed_audio = audio_chunk.to_vec();

        // Update emotion state based on audio characteristics
        {
            let mut adapter = self.adapter.write().await;
            adapter.update_from_audio(audio_chunk, self.config.sample_rate)?;

            // Get current emotion parameters
            let current_emotion = adapter.get_current_emotion();

            // Apply emotion-based audio processing
            self.apply_streaming_emotion_effects(&mut processed_audio, &current_emotion)
                .await?;
        }

        // Update session metrics
        {
            let mut sessions = self.active_sessions.write().await;
            if let Some(session) = sessions.get_mut(session_id) {
                // Apply session-specific emotion interpolation if enabled
                if self.config.enable_chunk_interpolation {
                    self.apply_chunk_interpolation(&mut processed_audio, session)
                        .await?;
                }
            }
        }

        // Update performance metrics
        let processing_time = start_time.elapsed();
        self.stream_metrics.chunks_processed += 1;
        self.stream_metrics.avg_processing_latency_ms =
            (self.stream_metrics.avg_processing_latency_ms * 0.9)
                + (processing_time.as_secs_f32() * 1000.0 * 0.1);

        trace!(
            "Processed audio chunk for session {} in {:?}",
            session_id,
            processing_time
        );
        Ok(processed_audio)
    }

    /// Apply streaming emotion effects to audio
    async fn apply_streaming_emotion_effects(
        &self,
        audio: &mut [f32],
        emotion_params: &EmotionParameters,
    ) -> Result<()> {
        // Apply pitch modulation for emotion
        if (emotion_params.pitch_shift - 1.0).abs() > 0.01 {
            self.apply_streaming_pitch_shift(audio, emotion_params.pitch_shift)?;
        }

        // Apply energy scaling for emotion
        if (emotion_params.energy_scale - 1.0).abs() > 0.01 {
            for sample in audio.iter_mut() {
                *sample *= emotion_params.energy_scale;
            }
        }

        // Apply voice quality effects for emotion
        if emotion_params.breathiness > 0.1 {
            self.apply_streaming_breathiness(audio, emotion_params.breathiness)?;
        }

        if emotion_params.roughness > 0.1 {
            self.apply_streaming_roughness(audio, emotion_params.roughness)?;
        }

        Ok(())
    }

    /// Apply chunk-level emotion interpolation
    async fn apply_chunk_interpolation(
        &self,
        audio: &mut [f32],
        session: &mut StreamingSession,
    ) -> Result<()> {
        // Interpolate between current and target emotion over the chunk duration
        let chunk_duration_ms = (audio.len() as f32 / self.config.sample_rate * 1000.0) as u64;

        // Calculate interpolation progress
        let time_since_target = session.last_activity.elapsed().as_millis() as f32;
        let interpolation_progress = (time_since_target / chunk_duration_ms as f32).clamp(0.0, 1.0);

        // Apply gradual transition
        if interpolation_progress < 1.0 {
            let samples_per_step = audio.len() / 10; // 10 interpolation steps per chunk

            for (step, chunk) in audio.chunks_mut(samples_per_step).enumerate() {
                let step_progress = (step as f32 / 10.0).clamp(0.0, 1.0);

                // Simple linear interpolation factor
                let interp_factor = interpolation_progress * step_progress;

                // Apply progressive emotion blending
                for sample in chunk.iter_mut() {
                    *sample *= 1.0 + (interp_factor * 0.1); // Gradual amplitude change
                }
            }
        }

        Ok(())
    }

    /// Apply streaming-optimized pitch shifting
    fn apply_streaming_pitch_shift(&self, audio: &mut [f32], pitch_shift: f32) -> Result<()> {
        if audio.len() < 2 {
            return Ok(());
        }

        // Simple time-domain pitch shifting for streaming (optimized for low latency)
        let shift_samples = ((pitch_shift - 1.0) * 10.0) as isize;

        if shift_samples != 0 {
            let mut shifted = vec![0.0; audio.len()];

            for (i, &sample) in audio.iter().enumerate() {
                let target_idx = (i as isize + shift_samples) as usize;
                if target_idx < shifted.len() {
                    shifted[target_idx] = sample;
                }
            }

            // Copy back with overlap-add for smoothness
            for (i, &shifted_sample) in shifted.iter().enumerate() {
                if i < audio.len() {
                    audio[i] = audio[i] * 0.3 + shifted_sample * 0.7;
                }
            }
        }

        Ok(())
    }

    /// Apply streaming breathiness effect
    fn apply_streaming_breathiness(&self, audio: &mut [f32], breathiness: f32) -> Result<()> {
        let noise_level = breathiness * 0.05; // Reduced for streaming quality

        for sample in audio.iter_mut() {
            let noise = (scirs2_core::random::random::<f32>() - 0.5) * noise_level;
            *sample = *sample * (1.0 - breathiness * 0.2) + noise;
        }

        Ok(())
    }

    /// Apply streaming roughness effect
    fn apply_streaming_roughness(&self, audio: &mut [f32], roughness: f32) -> Result<()> {
        // Light harmonic distortion for streaming
        for sample in audio.iter_mut() {
            if sample.abs() > 0.01 {
                let distortion = sample.signum() * (sample.abs().powf(1.0 - roughness * 0.2));
                *sample = *sample * (1.0 - roughness * 0.3) + distortion * roughness * 0.3;
            }
        }

        Ok(())
    }

    /// Send emotion signal to specific session
    pub async fn send_emotion_signal(&self, session_id: &str, signal: EmotionSignal) -> Result<()> {
        // Check if session exists
        {
            let sessions = self.active_sessions.read().await;
            if !sessions.contains_key(session_id) {
                return Err(Error::Processing(format!(
                    "Session not found: {}",
                    session_id
                )));
            }
        }

        // Send signal to emotion adapter
        if let Some(sender) = &self.emotion_sender {
            sender
                .send(signal)
                .map_err(|e| Error::Processing(format!("Failed to send emotion signal: {}", e)))?;
        }

        Ok(())
    }

    /// Get streaming metrics
    pub fn get_metrics(&self) -> StreamingMetrics {
        self.stream_metrics.clone()
    }

    /// Get active session count
    pub async fn get_active_session_count(&self) -> usize {
        self.active_sessions.read().await.len()
    }

    /// Get session information
    pub async fn get_session_info(&self, session_id: &str) -> Result<StreamingSession> {
        let sessions = self.active_sessions.read().await;
        sessions
            .get(session_id)
            .cloned()
            .ok_or_else(|| Error::Processing(format!("Session not found: {}", session_id)))
    }

    /// Update streaming configuration
    pub fn update_config(&mut self, new_config: StreamingConfig) {
        self.config = new_config;
        // Resize buffer if needed
        if self.audio_buffer.capacity() != self.config.buffer_size {
            self.audio_buffer = VecDeque::with_capacity(self.config.buffer_size);
        }
    }

    /// Cleanup inactive sessions
    pub async fn cleanup_inactive_sessions(&mut self, timeout_duration: Duration) -> Result<usize> {
        let mut sessions = self.active_sessions.write().await;
        let now = Instant::now();
        let initial_count = sessions.len();

        sessions.retain(|_, session| now.duration_since(session.last_activity) < timeout_duration);

        let cleaned_up = initial_count - sessions.len();
        self.stream_metrics.active_sessions = sessions.len() as u32;

        if cleaned_up > 0 {
            debug!("Cleaned up {} inactive sessions", cleaned_up);
        }

        Ok(cleaned_up)
    }
}
