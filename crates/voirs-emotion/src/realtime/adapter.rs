//! Real-time emotion adapter implementation
//!
//! This module provides the core `RealtimeEmotionAdapter` for real-time emotion adaptation.
//! The adapter can respond to audio input characteristics, external emotion signals, and
//! provides low-latency emotion updates with smooth transitions.
//!
//! # Features
//!
//! - Audio-driven emotion adaptation based on acoustic characteristics
//! - External emotion signal input with priority and duration control
//! - Configurable smoothing and rate limiting
//! - Emotion history tracking
//! - Performance metrics collection
//!
//! # Example
//!
//! ```no_run
//! use voirs_emotion::realtime::{RealtimeEmotionAdapter, RealtimeEmotionConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = RealtimeEmotionConfig::default();
//! let mut adapter = RealtimeEmotionAdapter::new(config)?;
//!
//! // Update emotion state
//! let current_emotion = adapter.update()?;
//! # Ok(())
//! # }
//! ```

use super::{AudioCharacteristics, EmotionSignal, RealtimeEmotionConfig};
use crate::{
    interpolation::{EmotionInterpolator, InterpolationConfig},
    types::{Emotion, EmotionDimensions, EmotionIntensity, EmotionParameters, EmotionVector},
    Error, Result,
};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, trace};

/// Real-time emotion adaptation system
pub struct RealtimeEmotionAdapter {
    /// Configuration
    config: RealtimeEmotionConfig,
    /// Emotion interpolator
    interpolator: EmotionInterpolator,
    /// Current emotion state
    current_emotion: EmotionParameters,
    /// Target emotion state
    target_emotion: EmotionParameters,
    /// Emotion history buffer
    pub(super) emotion_history: VecDeque<(Instant, EmotionParameters)>,
    /// External signal receiver
    signal_receiver: Option<mpsc::UnboundedReceiver<EmotionSignal>>,
    /// Active emotion signals
    active_signals: Vec<EmotionSignal>,
    /// Last update time
    last_update: Instant,
    /// Performance metrics
    pub(super) metrics: AdaptationMetrics,
}

impl RealtimeEmotionAdapter {
    /// Create new real-time emotion adapter
    pub fn new(config: RealtimeEmotionConfig) -> Result<Self> {
        config.validate()?;

        let interpolator = EmotionInterpolator::new(config.interpolation.clone());
        let neutral = EmotionParameters::neutral();

        Ok(Self {
            config,
            interpolator,
            current_emotion: neutral.clone(),
            target_emotion: neutral,
            emotion_history: VecDeque::new(),
            signal_receiver: None,
            active_signals: Vec::new(),
            last_update: Instant::now(),
            metrics: AdaptationMetrics::new(),
        })
    }

    /// Create with external signal input
    pub fn with_signal_input(mut self) -> (Self, mpsc::UnboundedSender<EmotionSignal>) {
        let (sender, receiver) = mpsc::unbounded_channel();
        self.signal_receiver = Some(receiver);
        (self, sender)
    }

    /// Update emotion state based on audio characteristics
    pub fn update_from_audio(&mut self, audio: &[f32], sample_rate: f32) -> Result<()> {
        if !self.config.enable_audio_adaptation {
            return Ok(());
        }

        let now = Instant::now();
        let delta_time = now.duration_since(self.last_update);

        // Check if we should update based on frequency
        let update_interval = Duration::from_secs_f32(1.0 / self.config.update_frequency);
        if delta_time < update_interval {
            return Ok(());
        }

        let characteristics = AudioCharacteristics::from_audio(audio, sample_rate);
        let emotion_dims = characteristics.to_emotion_dimensions();

        // Convert dimensions to emotion vector
        let emotion_vector = self.dimensions_to_emotion_vector(emotion_dims);
        let new_target = EmotionParameters::new(emotion_vector);

        // Apply smoothing and rate limiting
        let smoothed_target = self.apply_smoothing(new_target, delta_time)?;

        self.set_target_emotion(smoothed_target)?;
        self.last_update = now;

        trace!(
            "Updated emotion from audio: arousal={:.2}, valence={:.2}",
            emotion_dims.arousal,
            emotion_dims.valence
        );

        Ok(())
    }

    /// Set target emotion directly
    pub fn set_target_emotion(&mut self, target: EmotionParameters) -> Result<()> {
        // Check rate limiting
        if !self.check_rate_limit(&target) {
            debug!("Emotion change rate limited");
            return Ok(());
        }

        // Start transition to new target
        self.interpolator.start_transition(
            self.current_emotion.clone(),
            target.clone(),
            Some(self.config.interpolation.transition_duration_ms),
        )?;

        self.target_emotion = target;

        // Add to history
        self.add_to_history(self.current_emotion.clone());

        Ok(())
    }

    /// Get current emotion parameters
    pub fn get_current_emotion(&self) -> EmotionParameters {
        self.current_emotion.clone()
    }

    /// Update interpolation and get current emotion parameters
    pub fn update(&mut self) -> Result<EmotionParameters> {
        let now = Instant::now();

        // Process external signals
        self.process_external_signals()?;

        // Update interpolation
        if let Some(interpolated) = self.interpolator.update_transitions()? {
            self.current_emotion = interpolated;
        }

        // Update metrics
        self.metrics.update(now);

        Ok(self.current_emotion.clone())
    }

    /// Process external emotion signals
    pub(super) fn process_external_signals(&mut self) -> Result<()> {
        if !self.config.enable_external_input {
            return Ok(());
        }

        // Receive new signals
        if let Some(receiver) = &mut self.signal_receiver {
            while let Ok(signal) = receiver.try_recv() {
                self.active_signals.push(signal);
            }
        }

        // Remove expired signals
        self.active_signals.retain(|signal| !signal.is_expired());

        // Find highest priority signal
        if let Some(signal) = self.active_signals.iter().max_by_key(|s| s.priority) {
            // Blend signal with current target based on strength
            let blended = self.interpolator.interpolate(
                &self.target_emotion,
                &signal.target,
                signal.strength,
            )?;

            // Only update if significantly different
            if self.is_significant_change(&blended) {
                self.target_emotion = blended;
            }
        }

        Ok(())
    }

    /// Convert emotion dimensions to emotion vector
    pub(super) fn dimensions_to_emotion_vector(&self, dims: EmotionDimensions) -> EmotionVector {
        let mut vector = EmotionVector::new();
        vector.dimensions = dims;

        // Map dimensions to basic emotions
        // This is a simplified mapping - could be more sophisticated
        if dims.valence > 0.3 && dims.arousal > 0.3 {
            vector.add_emotion(
                Emotion::Happy,
                EmotionIntensity::new(dims.valence * dims.arousal),
            );
        }
        if dims.valence < -0.3 && dims.arousal > 0.3 {
            vector.add_emotion(
                Emotion::Angry,
                EmotionIntensity::new((-dims.valence) * dims.arousal),
            );
        }
        if dims.valence < -0.3 && dims.arousal < -0.3 {
            vector.add_emotion(
                Emotion::Sad,
                EmotionIntensity::new((-dims.valence) * (-dims.arousal)),
            );
        }
        if dims.valence > 0.3 && dims.arousal < -0.3 {
            vector.add_emotion(
                Emotion::Calm,
                EmotionIntensity::new(dims.valence * (-dims.arousal)),
            );
        }

        vector
    }

    /// Apply smoothing to emotion changes
    fn apply_smoothing(
        &self,
        new_target: EmotionParameters,
        delta_time: Duration,
    ) -> Result<EmotionParameters> {
        let smoothing = self.config.smoothing_factor;
        let delta_seconds = delta_time.as_secs_f32();
        let adaptive_smoothing = smoothing * (1.0 - delta_seconds.min(1.0));

        self.interpolator
            .interpolate(&self.target_emotion, &new_target, 1.0 - adaptive_smoothing)
    }

    /// Check if emotion change respects rate limiting
    fn check_rate_limit(&self, new_target: &EmotionParameters) -> bool {
        let now = Instant::now();
        let min_interval = Duration::from_millis(self.config.min_change_interval_ms);

        if now.duration_since(self.last_update) < min_interval {
            return false;
        }

        // Calculate change magnitude
        let change_magnitude = self.calculate_change_magnitude(&self.target_emotion, new_target);
        let max_change =
            self.config.max_change_rate * now.duration_since(self.last_update).as_secs_f32();

        change_magnitude <= max_change
    }

    /// Calculate magnitude of emotion change
    fn calculate_change_magnitude(&self, from: &EmotionParameters, to: &EmotionParameters) -> f32 {
        let dims_from = &from.emotion_vector.dimensions;
        let dims_to = &to.emotion_vector.dimensions;

        let valence_diff = (dims_to.valence - dims_from.valence).abs();
        let arousal_diff = (dims_to.arousal - dims_from.arousal).abs();
        let dominance_diff = (dims_to.dominance - dims_from.dominance).abs();

        (valence_diff + arousal_diff + dominance_diff) / 3.0
    }

    /// Check if emotion change is significant enough to process
    pub(super) fn is_significant_change(&self, new_emotion: &EmotionParameters) -> bool {
        let change_magnitude = self.calculate_change_magnitude(&self.current_emotion, new_emotion);
        change_magnitude > self.config.interpolation.change_threshold
    }

    /// Add emotion state to history
    fn add_to_history(&mut self, emotion: EmotionParameters) {
        let now = Instant::now();
        self.emotion_history.push_back((now, emotion));

        // Maintain buffer size
        while self.emotion_history.len() > self.config.history_buffer_size {
            self.emotion_history.pop_front();
        }
    }

    /// Get emotion history
    pub fn get_history(&self) -> &VecDeque<(Instant, EmotionParameters)> {
        &self.emotion_history
    }

    /// Get adaptation metrics
    pub fn get_metrics(&self) -> &AdaptationMetrics {
        &self.metrics
    }

    /// Reset to neutral emotion
    pub fn reset_to_neutral(&mut self) -> Result<()> {
        let neutral = EmotionParameters::neutral();
        self.set_target_emotion(neutral)?;
        self.emotion_history.clear();
        self.active_signals.clear();
        Ok(())
    }
}

/// Performance metrics for emotion adaptation
#[derive(Debug, Clone)]
pub struct AdaptationMetrics {
    /// Number of emotion updates
    pub update_count: u64,
    /// Average update frequency
    pub avg_update_frequency: f32,
    /// Number of transitions started
    pub transition_count: u64,
    /// Average transition duration
    pub avg_transition_duration: f32,
    /// Last update time
    pub last_update: Option<Instant>,
}

impl AdaptationMetrics {
    /// Create a new `AdaptationMetrics` instance with default values
    ///
    /// # Returns
    /// A new metrics instance with all counters set to zero
    pub fn new() -> Self {
        Self {
            update_count: 0,
            avg_update_frequency: 0.0,
            transition_count: 0,
            avg_transition_duration: 0.0,
            last_update: None,
        }
    }

    /// Update metrics with a new timestamp
    ///
    /// # Arguments
    /// * `now` - Current timestamp for computing update frequency
    pub fn update(&mut self, now: Instant) {
        self.update_count += 1;

        if let Some(last) = self.last_update {
            let delta = now.duration_since(last).as_secs_f32();
            let frequency = 1.0 / delta;

            // Exponential moving average
            let alpha = 0.1;
            self.avg_update_frequency =
                alpha * frequency + (1.0 - alpha) * self.avg_update_frequency;
        }

        self.last_update = Some(now);
    }
}

impl Default for AdaptationMetrics {
    fn default() -> Self {
        Self::new()
    }
}
