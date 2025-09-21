//! Emotion control integration for voice conversion
//!
//! This module provides integration with the voirs-emotion crate to enable
//! emotional transformation during voice conversion processes.

#[cfg(feature = "emotion-integration")]
use voirs_emotion;

use crate::{Error, Result};

/// Emotion integration adapter for voice conversion
#[cfg(feature = "emotion-integration")]
#[derive(Debug, Clone)]
pub struct EmotionConversionAdapter {
    /// Emotion model configuration
    config: Option<voirs_emotion::config::EmotionConfig>,
    /// Currently active emotion state
    current_emotion: Option<voirs_emotion::types::EmotionState>,
    /// Emotion transfer configuration
    transfer_config: EmotionTransferConfig,
}

#[cfg(feature = "emotion-integration")]
impl EmotionConversionAdapter {
    /// Create new emotion adapter
    pub fn new() -> Self {
        Self {
            config: None,
            current_emotion: None,
            transfer_config: EmotionTransferConfig::default(),
        }
    }

    /// Create adapter with specific emotion configuration
    pub fn with_config(config: voirs_emotion::config::EmotionConfig) -> Self {
        Self {
            config: Some(config),
            current_emotion: None,
            transfer_config: EmotionTransferConfig::default(),
        }
    }

    /// Create adapter with emotion transfer configuration
    pub fn with_transfer_config(
        emotion_config: voirs_emotion::config::EmotionConfig,
        transfer_config: EmotionTransferConfig,
    ) -> Self {
        let adapter = Self {
            config: Some(emotion_config),
            current_emotion: None,
            transfer_config,
        };
        adapter
    }

    /// Set target emotion for conversion
    pub fn set_target_emotion(&mut self, emotion: voirs_emotion::types::EmotionState) {
        self.current_emotion = Some(emotion);
    }

    /// Convert voice with emotional transformation
    pub async fn convert_with_emotion(
        &self,
        input_audio: &[f32],
        target_emotion: &voirs_emotion::types::EmotionState,
        intensity: f32,
    ) -> Result<Vec<f32>> {
        if intensity < 0.0 || intensity > 1.0 {
            return Err(Error::validation(
                "Emotion intensity must be between 0.0 and 1.0".to_string(),
            ));
        }

        if input_audio.is_empty() {
            return Err(Error::audio("Input audio cannot be empty".to_string()));
        }

        // Extract emotion parameters for conversion
        let emotion_params = self.extract_emotion_parameters(target_emotion, intensity);

        // Apply comprehensive emotional transformation
        let mut output = input_audio.to_vec();

        // Apply pitch modification based on arousal
        if emotion_params.pitch_modification.abs() > 0.01 {
            self.apply_pitch_modulation(&mut output, emotion_params.pitch_modification)?;
        }

        // Apply formant shifting based on valence
        if emotion_params.formant_shift.abs() > 0.01 {
            self.apply_formant_modification(&mut output, emotion_params.formant_shift)?;
        }

        // Apply rhythm adjustment based on dominance
        if emotion_params.rhythm_adjustment.abs() > 0.01 {
            self.apply_rhythm_modification(&mut output, emotion_params.rhythm_adjustment)?;
        }

        // Apply spectral tilt for emotional coloring
        if emotion_params.spectral_tilt.abs() > 0.01 {
            self.apply_spectral_tilt(&mut output, emotion_params.spectral_tilt)?;
        }

        // Apply overall emotional intensity scaling
        self.apply_intensity_scaling(&mut output, intensity)?;

        Ok(output)
    }

    /// Convert voice preserving source emotion while changing speaker
    pub async fn convert_preserving_emotion(
        &self,
        input_audio: &[f32],
        target_characteristics: &crate::types::VoiceCharacteristics,
    ) -> Result<Vec<f32>> {
        // Extract emotion from source
        let source_emotion = self.detect_source_emotion(input_audio)?;

        // Apply speaker conversion while preserving emotional state
        let converted_audio = self.apply_speaker_conversion_with_emotion_preservation(
            input_audio,
            target_characteristics,
            &source_emotion,
        )?;

        Ok(converted_audio)
    }

    /// Detect emotional state from audio
    pub fn detect_source_emotion(
        &self,
        audio: &[f32],
    ) -> Result<voirs_emotion::types::EmotionState> {
        if audio.is_empty() {
            return Err(Error::audio("Input audio cannot be empty".to_string()));
        }

        // Analyze audio characteristics to detect emotion
        let emotion_features = self.extract_emotion_features(audio)?;

        // Convert features to emotion state
        let emotion_state = self.features_to_emotion_state(emotion_features)?;

        Ok(emotion_state)
    }

    /// Transfer emotion from source to target voice while preserving target speaker identity
    pub async fn transfer_emotion_between_speakers(
        &self,
        source_audio: &[f32],
        target_audio: &[f32],
        transfer_intensity: f32,
    ) -> Result<Vec<f32>> {
        if source_audio.is_empty() || target_audio.is_empty() {
            return Err(Error::audio(
                "Both source and target audio must be non-empty".to_string(),
            ));
        }

        if transfer_intensity < 0.0 || transfer_intensity > 1.0 {
            return Err(Error::validation(
                "Transfer intensity must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Detect emotion from source
        let source_emotion = self.detect_source_emotion(source_audio)?;

        // Detect baseline emotion from target
        let target_emotion = self.detect_source_emotion(target_audio)?;

        // Blend emotions based on transfer intensity
        let blended_emotion =
            self.blend_emotions(&source_emotion, &target_emotion, transfer_intensity)?;

        // Apply blended emotion to target audio
        self.convert_with_emotion(target_audio, &blended_emotion, transfer_intensity)
            .await
    }

    /// Blend two emotional states with specified mixing ratio
    pub fn blend_emotions(
        &self,
        emotion_a: &voirs_emotion::types::EmotionState,
        emotion_b: &voirs_emotion::types::EmotionState,
        blend_ratio: f32, // 0.0 = all A, 1.0 = all B
    ) -> Result<voirs_emotion::types::EmotionState> {
        if blend_ratio < 0.0 || blend_ratio > 1.0 {
            return Err(Error::validation(
                "Blend ratio must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Create blended emotion state
        let mut blended_state = emotion_a.clone();

        // Blend dimensional values
        let a_dims = &emotion_a.current.emotion_vector.dimensions;
        let b_dims = &emotion_b.current.emotion_vector.dimensions;

        blended_state.current.emotion_vector.dimensions.arousal =
            a_dims.arousal * (1.0 - blend_ratio) + b_dims.arousal * blend_ratio;
        blended_state.current.emotion_vector.dimensions.valence =
            a_dims.valence * (1.0 - blend_ratio) + b_dims.valence * blend_ratio;
        blended_state.current.emotion_vector.dimensions.dominance =
            a_dims.dominance * (1.0 - blend_ratio) + b_dims.dominance * blend_ratio;

        // Update intensity based on blend
        // Calculate intensity from emotion vector dimensions
        let intensity_a =
            (a_dims.arousal.powi(2) + a_dims.valence.powi(2) + a_dims.dominance.powi(2)).sqrt();
        let intensity_b =
            (b_dims.arousal.powi(2) + b_dims.valence.powi(2) + b_dims.dominance.powi(2)).sqrt();
        let _blended_intensity = intensity_a * (1.0 - blend_ratio) + intensity_b * blend_ratio;

        Ok(blended_state)
    }

    /// Apply gradual emotion transition over time
    pub async fn apply_emotion_transition(
        &self,
        input_audio: &[f32],
        start_emotion: &voirs_emotion::types::EmotionState,
        end_emotion: &voirs_emotion::types::EmotionState,
        transition_points: usize,
    ) -> Result<Vec<f32>> {
        if input_audio.is_empty() {
            return Err(Error::audio("Input audio cannot be empty".to_string()));
        }

        if transition_points == 0 {
            return Err(Error::validation(
                "Transition points must be greater than 0".to_string(),
            ));
        }

        let chunk_size = input_audio.len() / transition_points;
        if chunk_size == 0 {
            return Err(Error::audio(
                "Audio too short for specified transition points".to_string(),
            ));
        }

        let mut output = Vec::with_capacity(input_audio.len());

        for i in 0..transition_points {
            let start_idx = i * chunk_size;
            let end_idx = if i == transition_points - 1 {
                input_audio.len()
            } else {
                (i + 1) * chunk_size
            };

            let chunk = &input_audio[start_idx..end_idx];
            let blend_ratio = i as f32 / (transition_points - 1) as f32;

            // Blend emotions for this chunk
            let current_emotion = self.blend_emotions(start_emotion, end_emotion, blend_ratio)?;

            // Apply emotion to chunk
            let processed_chunk = self
                .convert_with_emotion(chunk, &current_emotion, 0.8)
                .await?;
            output.extend(processed_chunk);
        }

        Ok(output)
    }

    // Private helper methods
    fn extract_emotion_parameters(
        &self,
        emotion: &voirs_emotion::types::EmotionState,
        intensity: f32,
    ) -> EmotionParameters {
        // Extract parameters from the actual EmotionState structure
        let arousal = emotion.current.emotion_vector.dimensions.arousal;
        let valence = emotion.current.emotion_vector.dimensions.valence;
        let dominance = emotion.current.emotion_vector.dimensions.dominance;

        // Scale parameters based on emotion intensity and user-specified intensity
        let emotion_intensity = (arousal.powi(2) + valence.powi(2) + dominance.powi(2)).sqrt();
        let effective_intensity = intensity * emotion_intensity;

        EmotionParameters {
            pitch_modification: arousal * effective_intensity * 0.25,
            formant_shift: valence * effective_intensity * 0.15,
            rhythm_adjustment: dominance * effective_intensity * 0.20,
            spectral_tilt: (arousal - 0.5) * effective_intensity * 0.35,
        }
    }

    fn extract_emotion_features(&self, audio: &[f32]) -> Result<EmotionFeatures> {
        // Analyze audio for emotional characteristics
        let mean_energy = audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32;
        let peak_energy = audio.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        let energy_variance = {
            let mean_square = mean_energy;
            let variance_sum: f32 = audio.iter().map(|&x| (x * x - mean_square).powi(2)).sum();
            variance_sum / audio.len() as f32
        };

        // Estimate pitch variation (simplified)
        let pitch_variation = self.estimate_pitch_variation(audio)?;

        // Estimate spectral characteristics
        let spectral_centroid = self.estimate_spectral_centroid(audio)?;
        let spectral_rolloff = self.estimate_spectral_rolloff(audio)?;

        Ok(EmotionFeatures {
            energy_level: mean_energy.sqrt(),
            energy_variance,
            pitch_variation,
            spectral_centroid,
            spectral_rolloff,
            peak_energy,
        })
    }

    fn estimate_pitch_variation(&self, audio: &[f32]) -> Result<f32> {
        // Simple pitch variation estimation using zero-crossing rate
        let mut zero_crossings = 0;
        for i in 1..audio.len() {
            if (audio[i] >= 0.0) != (audio[i - 1] >= 0.0) {
                zero_crossings += 1;
            }
        }

        let zcr = zero_crossings as f32 / audio.len() as f32;
        Ok(zcr.clamp(0.0, 1.0))
    }

    fn estimate_spectral_centroid(&self, audio: &[f32]) -> Result<f32> {
        // Simplified spectral centroid estimation
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for (i, &sample) in audio.iter().enumerate() {
            let magnitude = sample.abs();
            weighted_sum += i as f32 * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 0.0 {
            Ok((weighted_sum / magnitude_sum) / audio.len() as f32)
        } else {
            Ok(0.5) // Default to middle frequency
        }
    }

    fn estimate_spectral_rolloff(&self, audio: &[f32]) -> Result<f32> {
        // Simplified spectral rolloff estimation (85% of energy)
        let mut cumulative_energy = 0.0;
        let total_energy: f32 = audio.iter().map(|&x| x * x).sum();

        if total_energy == 0.0 {
            return Ok(0.5);
        }

        let target_energy = total_energy * 0.85;

        for (i, &sample) in audio.iter().enumerate() {
            cumulative_energy += sample * sample;
            if cumulative_energy >= target_energy {
                return Ok(i as f32 / audio.len() as f32);
            }
        }

        Ok(1.0)
    }

    fn features_to_emotion_state(
        &self,
        features: EmotionFeatures,
    ) -> Result<voirs_emotion::types::EmotionState> {
        // Convert audio features to emotion dimensions

        // Arousal correlates with energy and pitch variation
        let arousal =
            (features.energy_level * 0.6 + features.pitch_variation * 0.4).clamp(0.0, 1.0);

        // Valence correlates with spectral characteristics and energy stability
        let energy_stability = 1.0 - features.energy_variance.min(1.0);
        let valence = (features.spectral_centroid * 0.5 + energy_stability * 0.5).clamp(0.0, 1.0);

        // Dominance correlates with overall energy and spectral rolloff
        let dominance =
            (features.peak_energy * 0.7 + features.spectral_rolloff * 0.3).clamp(0.0, 1.0);

        // Create emotion state (simplified)
        let mut emotion_state = voirs_emotion::types::EmotionState::default();
        emotion_state.current.emotion_vector.dimensions.arousal = arousal;
        emotion_state.current.emotion_vector.dimensions.valence = valence;
        emotion_state.current.emotion_vector.dimensions.dominance = dominance;
        // Note: intensity is calculated from emotion vector magnitude when needed

        Ok(emotion_state)
    }

    fn apply_pitch_modulation(&self, audio: &mut [f32], pitch_factor: f32) -> Result<()> {
        // Apply pitch shifting using a simple time-domain approach
        // In a real implementation, this would use more sophisticated algorithms like PSOLA
        let shift_amount = (pitch_factor * 0.2).clamp(-0.5, 0.5);

        if shift_amount.abs() < 0.001 {
            return Ok(());
        }

        // Simple pitch shifting by sample rate manipulation simulation
        for (i, sample) in audio.iter_mut().enumerate() {
            let phase_shift = (i as f32 * shift_amount * 0.001).sin();
            *sample *= 1.0 + phase_shift * 0.1;
        }

        Ok(())
    }

    fn apply_formant_modification(&self, audio: &mut [f32], formant_factor: f32) -> Result<()> {
        // Apply formant shifting to modify vocal tract characteristics
        let formant_shift = (formant_factor * 0.15).clamp(-0.3, 0.3);

        if formant_shift.abs() < 0.001 {
            return Ok(());
        }

        // Simple formant shifting using spectral envelope modification
        for chunk in audio.chunks_mut(512) {
            let chunk_len = chunk.len();
            for (i, sample) in chunk.iter_mut().enumerate() {
                let freq_factor = 1.0 + formant_shift * (i as f32 / chunk_len as f32);
                *sample *= freq_factor;
            }
        }

        Ok(())
    }

    fn apply_rhythm_modification(&self, audio: &mut [f32], rhythm_factor: f32) -> Result<()> {
        // Apply rhythm/tempo adjustments based on dominance
        let tempo_change = (rhythm_factor * 0.1).clamp(-0.2, 0.2);

        if tempo_change.abs() < 0.001 {
            return Ok(());
        }

        // Simple tempo modification by selective sample amplification
        let window_size = 1024;
        for chunk in audio.chunks_mut(window_size) {
            let energy_factor = 1.0 + tempo_change;
            for sample in chunk.iter_mut() {
                *sample *= energy_factor;
            }
        }

        Ok(())
    }

    fn apply_spectral_tilt(&self, audio: &mut [f32], tilt_factor: f32) -> Result<()> {
        // Apply spectral tilt for emotional coloring
        let tilt = (tilt_factor * 0.25).clamp(-0.5, 0.5);

        if tilt.abs() < 0.001 {
            return Ok(());
        }

        // Apply high-frequency emphasis or de-emphasis
        let audio_len = audio.len();
        for (i, sample) in audio.iter_mut().enumerate() {
            let freq_weight = (i as f32 / audio_len as f32) * tilt;
            *sample *= 1.0 + freq_weight;
        }

        Ok(())
    }

    fn apply_intensity_scaling(&self, audio: &mut [f32], intensity: f32) -> Result<()> {
        // Apply overall intensity scaling based on emotional strength
        let scale_factor = 1.0 + (intensity - 0.5) * 0.3; // Scale around neutral

        for sample in audio.iter_mut() {
            *sample *= scale_factor;
            // Prevent clipping
            *sample = sample.clamp(-1.0, 1.0);
        }

        Ok(())
    }

    fn apply_speaker_conversion_with_emotion_preservation(
        &self,
        input_audio: &[f32],
        target_characteristics: &crate::types::VoiceCharacteristics,
        source_emotion: &voirs_emotion::types::EmotionState,
    ) -> Result<Vec<f32>> {
        if input_audio.is_empty() {
            return Err(Error::audio("Input audio cannot be empty".to_string()));
        }

        let mut converted_audio = input_audio.to_vec();

        // Apply speaker characteristics transformation while preserving emotion
        self.apply_speaker_characteristics(&mut converted_audio, target_characteristics)?;

        // Re-apply the source emotion to the converted voice
        let emotion_params = self.extract_emotion_parameters(source_emotion, 0.8);

        // Restore emotional characteristics
        self.apply_pitch_modulation(&mut converted_audio, emotion_params.pitch_modification)?;
        self.apply_formant_modification(&mut converted_audio, emotion_params.formant_shift)?;
        self.apply_rhythm_modification(&mut converted_audio, emotion_params.rhythm_adjustment)?;
        self.apply_spectral_tilt(&mut converted_audio, emotion_params.spectral_tilt)?;

        Ok(converted_audio)
    }

    fn apply_speaker_characteristics(
        &self,
        audio: &mut [f32],
        characteristics: &crate::types::VoiceCharacteristics,
    ) -> Result<()> {
        // Apply speaker-specific transformations
        // This would typically involve more sophisticated voice conversion algorithms

        // Apply gender-based modifications
        if let Some(gender) = &characteristics.gender {
            match gender {
                crate::types::Gender::Male => {
                    // Lower pitch and formants for male voice
                    self.apply_pitch_modulation(audio, -0.2)?;
                    self.apply_formant_modification(audio, -0.15)?;
                }
                crate::types::Gender::Female => {
                    // Higher pitch and formants for female voice
                    self.apply_pitch_modulation(audio, 0.2)?;
                    self.apply_formant_modification(audio, 0.15)?;
                }
                crate::types::Gender::NonBinary => {
                    // Neutral characteristics
                    self.apply_pitch_modulation(audio, 0.0)?;
                }
                crate::types::Gender::Other | crate::types::Gender::Unknown => {
                    // Neutral characteristics for unknown/other genders
                    self.apply_pitch_modulation(audio, 0.0)?;
                }
            }
        }

        // Apply age-based modifications
        if let Some(age_group) = &characteristics.age_group {
            match age_group {
                crate::types::AgeGroup::Child => {
                    // Higher pitch, more energy in high frequencies
                    self.apply_pitch_modulation(audio, 0.4)?;
                    self.apply_spectral_tilt(audio, 0.3)?;
                }
                crate::types::AgeGroup::Teen => {
                    // Slightly higher pitch than young adult
                    self.apply_pitch_modulation(audio, 0.2)?;
                    self.apply_spectral_tilt(audio, 0.1)?;
                }
                crate::types::AgeGroup::YoungAdult => {
                    // Moderate characteristics
                    self.apply_pitch_modulation(audio, 0.1)?;
                }
                crate::types::AgeGroup::Adult => {
                    // Baseline characteristics (no modification)
                }
                crate::types::AgeGroup::MiddleAged => {
                    // Slightly lower pitch than adult
                    self.apply_pitch_modulation(audio, -0.05)?;
                }
                crate::types::AgeGroup::Senior => {
                    // Lower pitch, more breathiness
                    self.apply_pitch_modulation(audio, -0.15)?;
                    self.apply_spectral_tilt(audio, -0.2)?;
                }
                crate::types::AgeGroup::Unknown => {
                    // No modification for unknown age
                }
            }
        }

        Ok(())
    }
}

#[cfg(feature = "emotion-integration")]
impl Default for EmotionConversionAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Parameters for emotional voice modification
#[cfg(feature = "emotion-integration")]
#[derive(Debug, Clone)]
pub struct EmotionParameters {
    /// Pitch modification factor (-1.0 to 1.0)
    pub pitch_modification: f32,
    /// Formant shift factor (-1.0 to 1.0)
    pub formant_shift: f32,
    /// Rhythm adjustment factor (-1.0 to 1.0)
    pub rhythm_adjustment: f32,
    /// Spectral tilt modification (-1.0 to 1.0)
    pub spectral_tilt: f32,
}

/// Audio features used for emotion detection
#[cfg(feature = "emotion-integration")]
#[derive(Debug, Clone)]
pub struct EmotionFeatures {
    /// Average energy level
    pub energy_level: f32,
    /// Energy variance (stability)
    pub energy_variance: f32,
    /// Pitch variation measure
    pub pitch_variation: f32,
    /// Spectral centroid (brightness)
    pub spectral_centroid: f32,
    /// Spectral rolloff point
    pub spectral_rolloff: f32,
    /// Peak energy level
    pub peak_energy: f32,
}

/// Emotion transfer configuration
#[cfg(feature = "emotion-integration")]
#[derive(Debug, Clone)]
pub struct EmotionTransferConfig {
    /// Enable pitch modification
    pub enable_pitch_mod: bool,
    /// Enable formant shifting
    pub enable_formant_shift: bool,
    /// Enable rhythm adjustment
    pub enable_rhythm_adj: bool,
    /// Enable spectral tilt
    pub enable_spectral_tilt: bool,
    /// Overall intensity scaling factor
    pub intensity_scale: f32,
    /// Minimum intensity threshold
    pub min_intensity: f32,
    /// Maximum intensity threshold
    pub max_intensity: f32,
}

#[cfg(feature = "emotion-integration")]
impl Default for EmotionTransferConfig {
    fn default() -> Self {
        Self {
            enable_pitch_mod: true,
            enable_formant_shift: true,
            enable_rhythm_adj: true,
            enable_spectral_tilt: true,
            intensity_scale: 1.0,
            min_intensity: 0.1,
            max_intensity: 1.0,
        }
    }
}

// Stub implementation when emotion integration is disabled
#[cfg(not(feature = "emotion-integration"))]
#[derive(Debug, Clone)]
pub struct EmotionConversionAdapter;

#[cfg(not(feature = "emotion-integration"))]
impl EmotionConversionAdapter {
    pub fn new() -> Self {
        Self
    }

    pub async fn convert_with_emotion(
        &self,
        _input_audio: &[f32],
        _target_emotion: &EmotionState,
        _intensity: f32,
    ) -> Result<Vec<f32>> {
        Err(Error::config(
            "Emotion integration not enabled. Enable with 'emotion-integration' feature."
                .to_string(),
        ))
    }

    pub async fn convert_preserving_emotion(
        &self,
        _input_audio: &[f32],
        _target_characteristics: &crate::types::VoiceCharacteristics,
    ) -> Result<Vec<f32>> {
        Err(Error::config(
            "Emotion integration not enabled. Enable with 'emotion-integration' feature."
                .to_string(),
        ))
    }
}

#[cfg(not(feature = "emotion-integration"))]
impl Default for EmotionConversionAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Stub emotion state when emotion integration is disabled
#[cfg(not(feature = "emotion-integration"))]
#[derive(Debug, Clone)]
pub struct EmotionState {
    pub valence: f32,
    pub arousal: f32,
    pub dominance: f32,
    pub emotion_label: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_adapter_creation() {
        let adapter = EmotionConversionAdapter::new();
        assert!(matches!(adapter, EmotionConversionAdapter { .. }));
    }

    #[cfg(feature = "emotion-integration")]
    #[tokio::test]
    async fn test_emotion_conversion_validation() {
        let adapter = EmotionConversionAdapter::new();
        let audio = vec![0.1, 0.2, 0.3, 0.4];
        let emotion = voirs_emotion::types::EmotionState::default();

        // Test invalid intensity
        let result = adapter.convert_with_emotion(&audio, &emotion, 1.5).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("intensity must be between"));

        // Test empty audio
        let result = adapter.convert_with_emotion(&[], &emotion, 0.5).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));

        // Test valid conversion
        let result = adapter.convert_with_emotion(&audio, &emotion, 0.5).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), audio.len());
    }

    #[cfg(feature = "emotion-integration")]
    #[tokio::test]
    async fn test_emotion_transfer_between_speakers() {
        let adapter = EmotionConversionAdapter::new();
        let source_audio = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let target_audio = vec![0.2, 0.3, 0.4, 0.5, 0.6];

        let result = adapter
            .transfer_emotion_between_speakers(&source_audio, &target_audio, 0.7)
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), target_audio.len());

        // Test validation errors
        let result = adapter
            .transfer_emotion_between_speakers(&[], &target_audio, 0.7)
            .await;
        assert!(result.is_err());

        let result = adapter
            .transfer_emotion_between_speakers(&source_audio, &target_audio, 1.5)
            .await;
        assert!(result.is_err());
    }

    #[cfg(feature = "emotion-integration")]
    #[test]
    fn test_emotion_blending() {
        let adapter = EmotionConversionAdapter::new();
        let emotion_a = voirs_emotion::types::EmotionState::default();
        let emotion_b = voirs_emotion::types::EmotionState::default();

        let result = adapter.blend_emotions(&emotion_a, &emotion_b, 0.5);
        assert!(result.is_ok());

        // Test invalid blend ratio
        let result = adapter.blend_emotions(&emotion_a, &emotion_b, 1.5);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Blend ratio must be between"));
    }

    #[cfg(feature = "emotion-integration")]
    #[tokio::test]
    async fn test_emotion_transition() {
        let adapter = EmotionConversionAdapter::new();
        let audio = vec![0.1; 1000]; // Create longer audio for transition
        let start_emotion = voirs_emotion::types::EmotionState::default();
        let end_emotion = voirs_emotion::types::EmotionState::default();

        let result = adapter
            .apply_emotion_transition(&audio, &start_emotion, &end_emotion, 4)
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), audio.len());

        // Test validation errors
        let result = adapter
            .apply_emotion_transition(&[], &start_emotion, &end_emotion, 4)
            .await;
        assert!(result.is_err());

        let result = adapter
            .apply_emotion_transition(&audio, &start_emotion, &end_emotion, 0)
            .await;
        assert!(result.is_err());
    }

    #[cfg(feature = "emotion-integration")]
    #[test]
    fn test_emotion_features_extraction() {
        let adapter = EmotionConversionAdapter::new();
        let audio = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6];

        let features = adapter.extract_emotion_features(&audio);
        assert!(features.is_ok());

        let features = features.unwrap();
        assert!(features.energy_level >= 0.0);
        assert!(features.pitch_variation >= 0.0 && features.pitch_variation <= 1.0);
        assert!(features.spectral_centroid >= 0.0 && features.spectral_centroid <= 1.0);
    }

    #[cfg(feature = "emotion-integration")]
    #[test]
    fn test_emotion_transfer_config() {
        let config = EmotionTransferConfig::default();
        assert!(config.enable_pitch_mod);
        assert!(config.enable_formant_shift);
        assert!(config.enable_rhythm_adj);
        assert!(config.enable_spectral_tilt);
        assert_eq!(config.intensity_scale, 1.0);
        assert_eq!(config.min_intensity, 0.1);
        assert_eq!(config.max_intensity, 1.0);
    }

    #[cfg(feature = "emotion-integration")]
    #[test]
    fn test_emotion_detection_from_audio() {
        let adapter = EmotionConversionAdapter::new();
        let audio = vec![0.1, -0.1, 0.2, -0.2, 0.15, -0.15];

        let result = adapter.detect_source_emotion(&audio);
        assert!(result.is_ok());

        let emotion = result.unwrap();
        assert!(emotion.current.emotion_vector.dimensions.arousal >= 0.0);
        assert!(emotion.current.emotion_vector.dimensions.arousal <= 1.0);
        assert!(emotion.current.emotion_vector.dimensions.valence >= 0.0);
        assert!(emotion.current.emotion_vector.dimensions.valence <= 1.0);
        assert!(emotion.current.emotion_vector.dimensions.dominance >= 0.0);
        assert!(emotion.current.emotion_vector.dimensions.dominance <= 1.0);

        // Test empty audio error
        let result = adapter.detect_source_emotion(&[]);
        assert!(result.is_err());
    }

    #[cfg(not(feature = "emotion-integration"))]
    #[tokio::test]
    async fn test_emotion_integration_disabled() {
        let adapter = EmotionConversionAdapter::new();
        let audio = vec![0.1, 0.2, 0.3, 0.4];
        let emotion = EmotionState {
            valence: 0.8,
            arousal: 0.6,
            dominance: 0.7,
            emotion_label: "happy".to_string(),
        };

        let result = adapter.convert_with_emotion(&audio, &emotion, 0.5).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not enabled"));
    }
}
