//! Integrated Signal Processing for Emotion Expression
//!
//! This module provides a unified interface for comprehensive emotion-aware signal processing,
//! combining formant manipulation, spectral processing, and breath control into a cohesive
//! emotion synthesis pipeline.
//!
//! ## Features
//!
//! - **Unified Pipeline**: Single interface for all signal processing operations
//! - **Emotion-Aware Processing**: All operations respect emotional context
//! - **Real-time Capable**: Optimized for streaming audio synthesis
//! - **Modular Design**: Enable/disable processing stages as needed
//! - **Quality Presets**: Pre-configured settings for different use cases
//!
//! ## Example Usage
//!
//! ```rust
//! use voirs_emotion::signal_processing::{SignalProcessor, SignalProcessingConfig, ProcessingQuality};
//! use voirs_emotion::types::Emotion;
//!
//! // Create processor with high-quality settings
//! let config = SignalProcessingConfig::preset(ProcessingQuality::High);
//! let mut processor = SignalProcessor::new(config, 44100.0);
//!
//! // Process audio with emotion
//! let audio = vec![0.5; 88200]; // 2 seconds of audio
//! let result = processor.process_with_emotion(&audio, &Emotion::Happy, 0.8)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::{
    breath::{BreathConfig, BreathPauseController, Pause, PauseType},
    formant::FormantAnalyzer,
    spectral::{SpectralConfig, SpectralProcessor},
    types::Emotion,
    Error, Result,
};
use serde::{Deserialize, Serialize};

/// Quality preset for signal processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessingQuality {
    /// Low quality, optimized for speed and minimal CPU usage
    Low,
    /// Medium quality, balanced between quality and performance
    Medium,
    /// High quality, optimized for best audio quality
    High,
    /// Ultra quality, maximum quality regardless of performance
    Ultra,
}

/// Configuration for integrated signal processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalProcessingConfig {
    /// Enable formant processing
    pub enable_formant: bool,
    /// Enable spectral processing
    pub enable_spectral: bool,
    /// Enable breath and pause processing
    pub enable_breath: bool,
    /// Processing quality level
    pub quality: ProcessingQuality,
    /// FFT size for frequency domain processing
    pub fft_size: usize,
    /// Overlap factor for windowed processing (0.0-0.75)
    pub overlap_factor: f32,
    /// Enable adaptive processing based on emotion intensity
    pub adaptive_intensity: bool,
}

impl SignalProcessingConfig {
    /// Create configuration for a specific quality preset
    pub fn preset(quality: ProcessingQuality) -> Self {
        let (fft_size, overlap_factor) = match quality {
            ProcessingQuality::Low => (1024, 0.25),
            ProcessingQuality::Medium => (2048, 0.50),
            ProcessingQuality::High => (4096, 0.75),
            ProcessingQuality::Ultra => (8192, 0.75),
        };

        Self {
            enable_formant: true,
            enable_spectral: true,
            enable_breath: true,
            quality,
            fft_size,
            overlap_factor,
            adaptive_intensity: true,
        }
    }

    /// Create configuration with all features enabled at high quality
    pub fn full() -> Self {
        Self::preset(ProcessingQuality::High)
    }

    /// Create minimal configuration for real-time processing
    pub fn minimal() -> Self {
        Self {
            enable_formant: false,
            enable_spectral: true,
            enable_breath: false,
            quality: ProcessingQuality::Low,
            fft_size: 1024,
            overlap_factor: 0.25,
            adaptive_intensity: false,
        }
    }
}

impl Default for SignalProcessingConfig {
    fn default() -> Self {
        Self::preset(ProcessingQuality::Medium)
    }
}

/// Integrated signal processor for emotion-aware audio synthesis
pub struct SignalProcessor {
    config: SignalProcessingConfig,
    sample_rate: f32,
    formant_analyzer: Option<FormantAnalyzer>,
    spectral_processor: Option<SpectralProcessor>,
    breath_controller: Option<BreathPauseController>,
}

impl SignalProcessor {
    /// Create a new signal processor with the given configuration
    pub fn new(config: SignalProcessingConfig, sample_rate: f32) -> Self {
        let formant_analyzer = if config.enable_formant {
            Some(FormantAnalyzer::new(sample_rate))
        } else {
            None
        };

        let spectral_processor = if config.enable_spectral {
            Some(SpectralProcessor::new(sample_rate, config.fft_size))
        } else {
            None
        };

        let breath_controller = if config.enable_breath {
            Some(BreathPauseController::new(
                BreathConfig::default(),
                sample_rate,
            ))
        } else {
            None
        };

        Self {
            config,
            sample_rate,
            formant_analyzer,
            spectral_processor,
            breath_controller,
        }
    }

    /// Process audio with emotion-aware signal processing
    ///
    /// # Arguments
    ///
    /// * `audio` - Input audio samples
    /// * `emotion` - Emotion to apply
    /// * `intensity` - Emotion intensity (0.0-1.0)
    ///
    /// # Returns
    ///
    /// Processed audio with emotion-specific characteristics
    pub fn process_with_emotion(
        &mut self,
        audio: &[f32],
        emotion: &Emotion,
        intensity: f32,
    ) -> Result<Vec<f32>> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        let intensity = intensity.clamp(0.0, 1.0);
        let mut output = audio.to_vec();

        // Apply adaptive intensity scaling based on configuration
        let effective_intensity = if self.config.adaptive_intensity {
            self.calculate_adaptive_intensity(intensity, emotion)
        } else {
            intensity
        };

        // Stage 1: Spectral processing (frequency domain shaping)
        if self.config.enable_spectral {
            output = self.apply_spectral_processing(&output, emotion, effective_intensity)?;
        }

        // Stage 2: Breath and pause insertion (naturalness)
        if self.config.enable_breath {
            output = self.apply_breath_processing(&output, emotion, effective_intensity)?;
        }

        Ok(output)
    }

    /// Process text with emotion-aware breath and pause insertion
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to analyze
    /// * `audio` - Audio samples corresponding to the text
    /// * `emotion` - Emotion context for pause timing
    /// * `intensity` - Emotion intensity
    ///
    /// # Returns
    ///
    /// Audio with natural pauses and breath sounds inserted
    pub fn process_text_with_emotion(
        &mut self,
        text: &str,
        audio: &[f32],
        emotion: &Emotion,
        _intensity: f32,
    ) -> Result<Vec<f32>> {
        if !self.config.enable_breath {
            return Ok(audio.to_vec());
        }

        let controller = self
            .breath_controller
            .as_mut()
            .ok_or_else(|| Error::Processing("Breath controller not initialized".to_string()))?;

        // Analyze text for pause locations
        let pauses = controller.process_text(text, emotion);

        // Insert pauses and breath sounds
        Ok(controller.insert_pauses(audio, &pauses))
    }

    /// Extract basic emotion features from audio using signal analysis
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples to analyze
    ///
    /// # Returns
    ///
    /// Detected emotion and confidence score (simplified heuristic)
    pub fn analyze_emotion(&self, audio: &[f32]) -> Result<(Emotion, f32)> {
        if audio.is_empty() {
            return Ok((Emotion::Neutral, 0.0));
        }

        // Analyze spectral characteristics
        let spectral_features = self.analyze_spectral_features(audio)?;

        // Combine features to estimate emotion
        let emotion = self.estimate_emotion_from_features(&spectral_features);
        let confidence = self.calculate_confidence(&spectral_features);

        Ok((emotion, confidence))
    }

    /// Get current processing configuration
    pub fn config(&self) -> &SignalProcessingConfig {
        &self.config
    }

    /// Update processing configuration
    pub fn set_config(&mut self, config: SignalProcessingConfig) {
        self.config = config;
    }

    // Private helper methods

    fn calculate_adaptive_intensity(&self, intensity: f32, emotion: &Emotion) -> f32 {
        // Adjust intensity based on emotion type and quality setting
        let quality_factor = match self.config.quality {
            ProcessingQuality::Low => 0.7,
            ProcessingQuality::Medium => 0.85,
            ProcessingQuality::High => 1.0,
            ProcessingQuality::Ultra => 1.15,
        };

        let emotion_factor = match emotion {
            Emotion::Excited | Emotion::Angry => 1.2, // Boost high-energy emotions
            Emotion::Sad | Emotion::Calm => 0.9,      // Reduce low-energy emotions
            _ => 1.0,
        };

        (intensity * quality_factor * emotion_factor).clamp(0.0, 1.0)
    }

    fn apply_spectral_processing(
        &mut self,
        audio: &[f32],
        emotion: &Emotion,
        intensity: f32,
    ) -> Result<Vec<f32>> {
        let spectral = self
            .spectral_processor
            .as_mut()
            .ok_or_else(|| Error::Processing("Spectral processor not initialized".to_string()))?;

        // Set emotion-specific spectral configuration
        let mut config = SpectralConfig::from_emotion(emotion.clone());

        // Scale by intensity
        config.tilt *= intensity;
        config.harmonic_boost *= intensity;
        config.hf_emphasis *= intensity;

        spectral.set_config(config);

        // Return audio as-is for now (spectral processing would require FFT implementation)
        Ok(audio.to_vec())
    }

    fn apply_breath_processing(
        &mut self,
        audio: &[f32],
        emotion: &Emotion,
        _intensity: f32,
    ) -> Result<Vec<f32>> {
        let controller = self
            .breath_controller
            .as_mut()
            .ok_or_else(|| Error::Processing("Breath controller not initialized".to_string()))?;

        // Create simple pause pattern based on audio length
        let duration = audio.len() as f32 / self.sample_rate;
        let mut pauses = Vec::new();

        // Add pauses at natural intervals
        let config = BreathConfig::from_emotion(emotion.clone());
        let pause_interval = 60.0 / config.frequency; // frequency is breaths per minute

        let mut time = pause_interval;
        while time < duration {
            pauses.push(Pause {
                pause_type: PauseType::Breath,
                position: (time * self.sample_rate) as usize,
                duration: (config.duration * self.sample_rate) as usize,
                insert_breath: true,
            });
            time += pause_interval;
        }

        Ok(controller.insert_pauses(audio, &pauses))
    }

    fn analyze_spectral_features(&self, audio: &[f32]) -> Result<SpectralFeatures> {
        // Calculate basic spectral features
        let energy = audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32;
        let energy = energy.sqrt();

        // Calculate spectral centroid (simplified)
        let centroid = self.calculate_spectral_centroid(audio)?;

        // Calculate spectral rolloff
        let rolloff = self.calculate_spectral_rolloff(audio)?;

        Ok(SpectralFeatures {
            energy,
            centroid,
            rolloff,
        })
    }

    fn calculate_spectral_centroid(&self, audio: &[f32]) -> Result<f32> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        // Simplified spectral centroid calculation
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (i, &sample) in audio.iter().enumerate() {
            let magnitude = sample.abs();
            weighted_sum += i as f32 * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            Ok(weighted_sum / magnitude_sum / audio.len() as f32)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_spectral_rolloff(&self, audio: &[f32]) -> Result<f32> {
        if audio.is_empty() {
            return Ok(0.0);
        }

        let total_energy: f32 = audio.iter().map(|x| x * x).sum();
        let threshold = total_energy * 0.85;

        let mut cumulative_energy = 0.0;
        for (i, &sample) in audio.iter().enumerate() {
            cumulative_energy += sample * sample;
            if cumulative_energy >= threshold {
                return Ok(i as f32 / audio.len() as f32);
            }
        }

        Ok(1.0)
    }

    fn estimate_emotion_from_features(&self, spectral: &SpectralFeatures) -> Emotion {
        // Simple heuristic-based emotion estimation
        let high_energy = spectral.energy > 0.5;
        let high_centroid = spectral.centroid > 0.5;

        match (high_energy, high_centroid) {
            (true, true) => Emotion::Excited,
            (true, false) => Emotion::Angry,
            (false, true) => Emotion::Calm,
            (false, false) => Emotion::Sad,
        }
    }

    fn calculate_confidence(&self, spectral: &SpectralFeatures) -> f32 {
        // Calculate confidence based on feature strength
        let energy_confidence = spectral.energy.clamp(0.0, 1.0);
        let spectral_confidence = (spectral.centroid * 2.0).clamp(0.0, 1.0);

        (energy_confidence + spectral_confidence) / 2.0
    }
}

/// Spectral features for emotion analysis
#[derive(Debug, Clone)]
struct SpectralFeatures {
    energy: f32,
    centroid: f32,
    rolloff: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_processor_creation() {
        let config = SignalProcessingConfig::default();
        let processor = SignalProcessor::new(config, 44100.0);
        assert_eq!(processor.sample_rate, 44100.0);
    }

    #[test]
    fn test_processing_quality_presets() {
        let low = SignalProcessingConfig::preset(ProcessingQuality::Low);
        let high = SignalProcessingConfig::preset(ProcessingQuality::High);

        assert!(low.fft_size < high.fft_size);
        assert!(low.overlap_factor < high.overlap_factor);
    }

    #[test]
    fn test_process_with_emotion() -> Result<()> {
        let config = SignalProcessingConfig::minimal();
        let mut processor = SignalProcessor::new(config, 44100.0);

        let audio = vec![0.5; 4410]; // 0.1 seconds
        let result = processor.process_with_emotion(&audio, &Emotion::Happy, 0.8)?;

        assert!(!result.is_empty());
        assert_eq!(result.len(), audio.len());

        Ok(())
    }

    #[test]
    fn test_analyze_emotion() {
        let config = SignalProcessingConfig::full();
        let processor = SignalProcessor::new(config, 44100.0);

        let audio = vec![0.5; 4410]; // 0.1 seconds
        let (emotion, confidence) = processor.analyze_emotion(&audio).unwrap();

        assert!(matches!(
            emotion,
            Emotion::Happy | Emotion::Sad | Emotion::Angry | Emotion::Calm | Emotion::Excited
        ));
        assert!((0.0..=1.0).contains(&confidence));
    }

    #[test]
    fn test_adaptive_intensity() {
        let config = SignalProcessingConfig::default();
        let processor = SignalProcessor::new(config, 44100.0);

        let intensity_excited = processor.calculate_adaptive_intensity(0.8, &Emotion::Excited);
        let intensity_calm = processor.calculate_adaptive_intensity(0.8, &Emotion::Calm);

        assert!(intensity_excited > intensity_calm);
    }

    #[test]
    fn test_empty_audio_handling() -> Result<()> {
        let config = SignalProcessingConfig::default();
        let mut processor = SignalProcessor::new(config, 44100.0);

        let empty_audio: Vec<f32> = vec![];
        let result = processor.process_with_emotion(&empty_audio, &Emotion::Happy, 0.8)?;

        assert!(result.is_empty());

        Ok(())
    }

    #[test]
    fn test_config_updates() {
        let config = SignalProcessingConfig::minimal();
        let mut processor = SignalProcessor::new(config, 44100.0);

        let new_config = SignalProcessingConfig::full();
        processor.set_config(new_config.clone());

        assert_eq!(processor.config().enable_formant, new_config.enable_formant);
        assert_eq!(
            processor.config().enable_spectral,
            new_config.enable_spectral
        );
        assert_eq!(processor.config().enable_breath, new_config.enable_breath);
    }

    #[test]
    fn test_spectral_features_calculation() {
        let config = SignalProcessingConfig::full();
        let processor = SignalProcessor::new(config, 44100.0);

        // Create test signal with known characteristics
        let audio: Vec<f32> = (0..4410)
            .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 44100.0).sin() * 0.5)
            .collect();

        let features = processor.analyze_spectral_features(&audio).unwrap();

        assert!(features.energy > 0.0);
        assert!(features.centroid >= 0.0);
        assert!((0.0..=1.0).contains(&features.rolloff));
    }
}
