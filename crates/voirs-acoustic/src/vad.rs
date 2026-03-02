//! Voice Activity Detection (VAD) Integration
//!
//! This module provides Voice Activity Detection capabilities for improved
//! silence handling, pause detection, and speech segmentation in TTS synthesis.
//!
//! # Features
//! - Real-time voice activity detection
//! - Configurable energy and spectral thresholds
//! - Adaptive threshold adjustment
//! - Speech/silence segmentation
//! - Pause duration optimization

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::{AcousticError, Result};

/// Voice Activity Detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadConfig {
    /// Energy threshold for voice activity (dB)
    pub energy_threshold_db: f32,
    /// Zero-crossing rate threshold
    pub zcr_threshold: f32,
    /// Minimum speech duration (ms)
    pub min_speech_duration_ms: f32,
    /// Minimum silence duration (ms)
    pub min_silence_duration_ms: f32,
    /// Frame size for analysis (samples)
    pub frame_size: usize,
    /// Hop size for frame advance (samples)
    pub hop_size: usize,
    /// Sample rate (Hz)
    pub sample_rate: usize,
    /// Enable adaptive threshold adjustment
    pub adaptive_threshold: bool,
    /// Smoothing window size for decisions
    pub smoothing_window: usize,
    /// Spectral flux threshold for onset detection
    pub spectral_flux_threshold: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            energy_threshold_db: -40.0, // -40 dB threshold
            zcr_threshold: 0.2,         // 20% crossing rate
            min_speech_duration_ms: 100.0,
            min_silence_duration_ms: 200.0,
            frame_size: 512,
            hop_size: 256,
            sample_rate: 22050,
            adaptive_threshold: true,
            smoothing_window: 5, // Smooth over 5 frames
            spectral_flux_threshold: 0.1,
        }
    }
}

impl VadConfig {
    /// Create configuration optimized for conversational speech
    pub fn conversational() -> Self {
        Self {
            energy_threshold_db: -35.0,
            min_speech_duration_ms: 150.0,
            min_silence_duration_ms: 250.0,
            adaptive_threshold: true,
            ..Default::default()
        }
    }

    /// Create configuration optimized for clean studio recording
    pub fn studio() -> Self {
        Self {
            energy_threshold_db: -50.0,
            min_speech_duration_ms: 80.0,
            min_silence_duration_ms: 150.0,
            adaptive_threshold: false,
            ..Default::default()
        }
    }

    /// Create configuration for noisy environment
    pub fn noisy() -> Self {
        Self {
            energy_threshold_db: -25.0,
            min_speech_duration_ms: 200.0,
            min_silence_duration_ms: 300.0,
            adaptive_threshold: true,
            smoothing_window: 7,
            ..Default::default()
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.energy_threshold_db > 0.0 || self.energy_threshold_db < -100.0 {
            return Err(AcousticError::ConfigError {
                message: format!(
                    "Invalid energy threshold: {} (must be -100 to 0 dB)",
                    self.energy_threshold_db
                ),
            });
        }

        if !(0.0..=1.0).contains(&self.zcr_threshold) {
            return Err(AcousticError::ConfigError {
                message: format!(
                    "Invalid ZCR threshold: {} (must be 0.0-1.0)",
                    self.zcr_threshold
                ),
            });
        }

        if self.frame_size == 0 || self.hop_size == 0 {
            return Err(AcousticError::ConfigError {
                message: "Frame size and hop size must be positive".to_string(),
            });
        }

        Ok(())
    }

    /// Calculate number of frames per second
    pub fn frames_per_second(&self) -> f32 {
        self.sample_rate as f32 / self.hop_size as f32
    }

    /// Convert milliseconds to number of frames
    pub fn ms_to_frames(&self, ms: f32) -> usize {
        (ms * self.frames_per_second() / 1000.0) as usize
    }
}

/// Voice activity state for a frame
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoiceActivity {
    /// Speech is present
    Speech,
    /// Silence detected
    Silence,
    /// Uncertain/transition state
    Uncertain,
}

/// Voice activity detection result for a segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadSegment {
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Activity type
    pub activity: VoiceActivity,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Average energy in segment (dB)
    pub avg_energy_db: f32,
}

impl VadSegment {
    /// Get duration of segment in seconds
    pub fn duration(&self) -> f32 {
        self.end_time - self.start_time
    }

    /// Get duration in milliseconds
    pub fn duration_ms(&self) -> f32 {
        self.duration() * 1000.0
    }

    /// Check if this is a speech segment
    pub fn is_speech(&self) -> bool {
        self.activity == VoiceActivity::Speech
    }

    /// Check if this is a silence segment
    pub fn is_silence(&self) -> bool {
        self.activity == VoiceActivity::Silence
    }
}

/// Voice Activity Detector
pub struct VoiceActivityDetector {
    /// Configuration
    config: VadConfig,
    /// Adaptive energy threshold
    adaptive_energy_threshold: f32,
    /// Recent energy history for adaptation
    energy_history: VecDeque<f32>,
    /// Smoothing buffer for activity decisions
    decision_buffer: VecDeque<VoiceActivity>,
    /// Current segment being built
    current_segment: Option<VadSegment>,
    /// Frame counter
    frame_count: usize,
}

impl VoiceActivityDetector {
    /// Create new VAD with configuration
    pub fn new(config: VadConfig) -> Result<Self> {
        config.validate()?;

        Ok(Self {
            adaptive_energy_threshold: config.energy_threshold_db,
            energy_history: VecDeque::with_capacity(100),
            decision_buffer: VecDeque::with_capacity(config.smoothing_window),
            current_segment: None,
            frame_count: 0,
            config,
        })
    }

    /// Create VAD with default configuration
    pub fn with_defaults() -> Result<Self> {
        Self::new(VadConfig::default())
    }

    /// Process audio frame and detect voice activity
    pub fn process_frame(&mut self, frame: &[f32]) -> VoiceActivity {
        if frame.len() != self.config.frame_size {
            return VoiceActivity::Uncertain;
        }

        // Calculate frame energy
        let energy_db = Self::calculate_energy_db(frame);

        // Calculate zero-crossing rate
        let zcr = Self::calculate_zcr(frame);

        // Update adaptive threshold if enabled
        if self.config.adaptive_threshold {
            self.update_adaptive_threshold(energy_db);
        }

        // Make decision based on features
        let activity = self.classify_frame(energy_db, zcr);

        // Apply smoothing
        let smoothed = self.apply_smoothing(activity);

        self.frame_count += 1;

        smoothed
    }

    /// Process audio buffer and return segments
    pub fn process_buffer(&mut self, audio: &[f32]) -> Vec<VadSegment> {
        let mut segments = Vec::new();
        let num_frames = (audio.len() - self.config.frame_size) / self.config.hop_size + 1;

        for i in 0..num_frames {
            let start_idx = i * self.config.hop_size;
            let end_idx = start_idx + self.config.frame_size;

            if end_idx <= audio.len() {
                let frame = &audio[start_idx..end_idx];
                let activity = self.process_frame(frame);

                // Build segments
                self.update_segments(activity, i, &mut segments);
            }
        }

        // Finalize any remaining segment
        if let Some(segment) = self.current_segment.take() {
            if Self::is_valid_segment(&segment, &self.config) {
                segments.push(segment);
            }
        }

        segments
    }

    /// Calculate energy in dB
    pub fn calculate_energy_db(frame: &[f32]) -> f32 {
        let energy: f32 = frame.iter().map(|x| x * x).sum();
        let rms = (energy / frame.len() as f32).sqrt();

        if rms > 0.0 {
            20.0 * rms.log10()
        } else {
            -100.0 // Very low energy
        }
    }

    /// Calculate zero-crossing rate
    pub fn calculate_zcr(frame: &[f32]) -> f32 {
        let mut crossings = 0;

        for i in 1..frame.len() {
            if (frame[i] >= 0.0) != (frame[i - 1] >= 0.0) {
                crossings += 1;
            }
        }

        crossings as f32 / (frame.len() - 1) as f32
    }

    /// Classify frame based on features
    fn classify_frame(&self, energy_db: f32, zcr: f32) -> VoiceActivity {
        let energy_threshold = self.adaptive_energy_threshold;

        // Speech if energy above threshold and ZCR in reasonable range
        if energy_db > energy_threshold && zcr < self.config.zcr_threshold {
            VoiceActivity::Speech
        } else if energy_db < energy_threshold - 10.0 {
            // Clear silence
            VoiceActivity::Silence
        } else {
            // Uncertain
            VoiceActivity::Uncertain
        }
    }

    /// Apply temporal smoothing to decisions
    fn apply_smoothing(&mut self, activity: VoiceActivity) -> VoiceActivity {
        self.decision_buffer.push_back(activity);

        if self.decision_buffer.len() > self.config.smoothing_window {
            self.decision_buffer.pop_front();
        }

        // Majority vote
        let speech_count = self
            .decision_buffer
            .iter()
            .filter(|&a| *a == VoiceActivity::Speech)
            .count();
        let silence_count = self
            .decision_buffer
            .iter()
            .filter(|&a| *a == VoiceActivity::Silence)
            .count();

        if speech_count > silence_count {
            VoiceActivity::Speech
        } else if silence_count > speech_count {
            VoiceActivity::Silence
        } else {
            VoiceActivity::Uncertain
        }
    }

    /// Update adaptive energy threshold
    fn update_adaptive_threshold(&mut self, energy_db: f32) {
        self.energy_history.push_back(energy_db);

        if self.energy_history.len() > 100 {
            self.energy_history.pop_front();
        }

        // Update threshold based on recent energy distribution
        if self.energy_history.len() >= 10 {
            let mut sorted: Vec<f32> = self.energy_history.iter().copied().collect();
            // Sort with NaN handling for noise floor estimation
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Use 20th percentile as noise floor estimate
            let noise_floor_idx = sorted.len() / 5;
            let noise_floor = sorted[noise_floor_idx];

            // Set threshold 15dB above noise floor
            self.adaptive_energy_threshold = noise_floor + 15.0;
        }
    }

    /// Update segments based on current activity
    fn update_segments(
        &mut self,
        activity: VoiceActivity,
        frame_idx: usize,
        segments: &mut Vec<VadSegment>,
    ) {
        let time = frame_idx as f32 * self.config.hop_size as f32 / self.config.sample_rate as f32;

        if let Some(ref mut segment) = self.current_segment {
            if segment.activity == activity {
                // Continue current segment
                segment.end_time = time;
            } else {
                // End current segment and start new one
                let completed = self
                    .current_segment
                    .take()
                    .expect("checked Some in if-let above");

                if Self::is_valid_segment(&completed, &self.config) {
                    segments.push(completed);
                }

                self.current_segment = Some(VadSegment {
                    start_time: time,
                    end_time: time,
                    activity,
                    confidence: 0.8,
                    avg_energy_db: self.adaptive_energy_threshold,
                });
            }
        } else {
            // Start new segment
            self.current_segment = Some(VadSegment {
                start_time: time,
                end_time: time,
                activity,
                confidence: 0.8,
                avg_energy_db: self.adaptive_energy_threshold,
            });
        }
    }

    /// Check if segment meets minimum duration requirements
    fn is_valid_segment(segment: &VadSegment, config: &VadConfig) -> bool {
        let duration_ms = segment.duration_ms();

        match segment.activity {
            VoiceActivity::Speech => duration_ms >= config.min_speech_duration_ms,
            VoiceActivity::Silence => duration_ms >= config.min_silence_duration_ms,
            VoiceActivity::Uncertain => false,
        }
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        self.energy_history.clear();
        self.decision_buffer.clear();
        self.current_segment = None;
        self.frame_count = 0;
        self.adaptive_energy_threshold = self.config.energy_threshold_db;
    }

    /// Get current configuration
    pub fn config(&self) -> &VadConfig {
        &self.config
    }

    /// Get current adaptive threshold
    pub fn adaptive_threshold(&self) -> f32 {
        self.adaptive_energy_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_config_validation() {
        let valid = VadConfig::default();
        assert!(valid.validate().is_ok());

        let invalid_energy = VadConfig {
            energy_threshold_db: 10.0,
            ..Default::default()
        };
        assert!(invalid_energy.validate().is_err());

        let invalid_zcr = VadConfig {
            zcr_threshold: 1.5,
            ..Default::default()
        };
        assert!(invalid_zcr.validate().is_err());
    }

    #[test]
    fn test_vad_config_presets() {
        let conv = VadConfig::conversational();
        assert_eq!(conv.energy_threshold_db, -35.0);

        let studio = VadConfig::studio();
        assert_eq!(studio.energy_threshold_db, -50.0);

        let noisy = VadConfig::noisy();
        assert_eq!(noisy.energy_threshold_db, -25.0);
    }

    #[test]
    fn test_vad_config_frame_conversion() {
        let config = VadConfig::default();
        let fps = config.frames_per_second();
        assert!(fps > 0.0);

        let frames = config.ms_to_frames(100.0);
        assert!(frames > 0);
    }

    #[test]
    fn test_vad_segment() {
        let segment = VadSegment {
            start_time: 0.0,
            end_time: 1.5,
            activity: VoiceActivity::Speech,
            confidence: 0.9,
            avg_energy_db: -30.0,
        };

        assert_eq!(segment.duration(), 1.5);
        assert_eq!(segment.duration_ms(), 1500.0);
        assert!(segment.is_speech());
        assert!(!segment.is_silence());
    }

    #[test]
    fn test_vad_creation() {
        let config = VadConfig::default();
        let vad = VoiceActivityDetector::new(config);
        assert!(vad.is_ok());
    }

    #[test]
    fn test_energy_calculation() {
        // High energy frame (speech-like)
        let high_energy: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let energy_db = VoiceActivityDetector::calculate_energy_db(&high_energy);
        assert!(energy_db > -50.0);

        // Low energy frame (silence-like)
        let low_energy = vec![0.001; 512];
        let energy_db = VoiceActivityDetector::calculate_energy_db(&low_energy);
        assert!(energy_db < -50.0);
    }

    #[test]
    fn test_zcr_calculation() {
        // High ZCR (noise-like)
        let high_zcr: Vec<f32> = (0..512)
            .map(|i| if i % 2 == 0 { 0.1 } else { -0.1 })
            .collect();
        let zcr = VoiceActivityDetector::calculate_zcr(&high_zcr);
        assert!(zcr > 0.9);

        // Low ZCR (tone-like)
        let low_zcr: Vec<f32> = vec![0.1; 512];
        let zcr = VoiceActivityDetector::calculate_zcr(&low_zcr);
        assert!(zcr < 0.01);
    }

    #[test]
    fn test_vad_frame_processing() {
        let config = VadConfig::default();
        let mut vad = VoiceActivityDetector::new(config).unwrap();

        // Speech-like frame
        let speech_frame: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin() * 0.3).collect();
        let activity = vad.process_frame(&speech_frame);
        // Due to smoothing, might not immediately classify as speech
        assert!(activity == VoiceActivity::Speech || activity == VoiceActivity::Uncertain);

        // Silence frame
        let silence_frame = vec![0.001; 512];
        let activity = vad.process_frame(&silence_frame);
        // Should eventually classify as silence
        for _ in 0..10 {
            vad.process_frame(&silence_frame);
        }
        let activity = vad.process_frame(&silence_frame);
        assert_eq!(activity, VoiceActivity::Silence);
    }

    #[test]
    fn test_vad_buffer_processing() {
        let config = VadConfig {
            min_speech_duration_ms: 50.0,
            min_silence_duration_ms: 50.0,
            ..Default::default()
        };
        let mut vad = VoiceActivityDetector::new(config).unwrap();

        // Create audio with speech and silence
        let mut audio = Vec::new();

        // Speech segment (1 second)
        for i in 0..22050 {
            audio.push((i as f32 * 0.01).sin() * 0.3);
        }

        // Silence segment (0.5 second)
        audio.resize(audio.len() + 11025, 0.001);

        let segments = vad.process_buffer(&audio);

        // Should have at least one speech segment
        let speech_segments: Vec<_> = segments.iter().filter(|s| s.is_speech()).collect();
        assert!(!speech_segments.is_empty());
    }

    #[test]
    fn test_vad_reset() {
        let config = VadConfig::default();
        let mut vad = VoiceActivityDetector::new(config).unwrap();

        // Process some frames
        let frame = vec![0.1; 512];
        for _ in 0..10 {
            vad.process_frame(&frame);
        }

        // Reset
        vad.reset();

        // State should be cleared
        assert_eq!(vad.frame_count, 0);
        assert_eq!(vad.energy_history.len(), 0);
    }

    #[test]
    fn test_adaptive_threshold() {
        let config = VadConfig {
            adaptive_threshold: true,
            ..Default::default()
        };
        let mut vad = VoiceActivityDetector::new(config).unwrap();

        let initial_threshold = vad.adaptive_threshold();

        // Process frames with varying energy
        for _ in 0..50 {
            let energy = fastrand::f32() * 0.5;
            let frame: Vec<f32> = vec![energy; 512];
            vad.process_frame(&frame);
        }

        // Threshold should have adapted
        // (might increase or decrease depending on the random data)
        let adapted_threshold = vad.adaptive_threshold();
        assert_ne!(initial_threshold, adapted_threshold);
    }
}
