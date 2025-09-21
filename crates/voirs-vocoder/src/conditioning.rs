//! Unified conditioning interface for advanced vocoder features.
//!
//! This module provides a unified API for configuring and applying multiple
//! vocoder conditioning features such as emotion, voice conversion, and other
//! advanced voice characteristics.

use crate::{conversion::VoiceConversionConfig, hifigan::EmotionConfig, AudioBuffer, Result};
use serde::{Deserialize, Serialize};

/// Unified conditioning configuration for advanced vocoder features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocoderConditioningConfig {
    /// Emotion-based conditioning
    pub emotion: Option<EmotionConfig>,
    /// Voice conversion settings
    pub voice_conversion: Option<VoiceConversionConfig>,
    /// Speaker characteristics
    pub speaker: Option<SpeakerConfig>,
    /// Prosodic modifications
    pub prosody: Option<ProsodyConfig>,
    /// Audio enhancement settings
    pub enhancement: Option<EnhancementConfig>,
    /// Global conditioning strength (0.0 to 1.0)
    pub global_strength: f32,
    /// Feature priority weights for conflict resolution
    pub feature_weights: FeatureWeights,
}

impl Default for VocoderConditioningConfig {
    fn default() -> Self {
        Self {
            emotion: None,
            voice_conversion: None,
            speaker: None,
            prosody: None,
            enhancement: None,
            global_strength: 1.0,
            feature_weights: FeatureWeights::default(),
        }
    }
}

/// Speaker characteristics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerConfig {
    /// Speaker ID for multi-speaker models
    pub speaker_id: Option<u32>,
    /// Speaker embedding vector
    pub speaker_embedding: Option<Vec<f32>>,
    /// Voice characteristics
    pub voice_characteristics: VoiceCharacteristics,
}

/// Voice characteristics settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCharacteristics {
    /// Fundamental frequency adjustment (-1.0 to 1.0)
    pub f0_adjustment: f32,
    /// Vocal tract length adjustment (-1.0 to 1.0)
    pub vtl_adjustment: f32,
    /// Voice quality settings
    pub quality: VoiceQuality,
}

impl Default for VoiceCharacteristics {
    fn default() -> Self {
        Self {
            f0_adjustment: 0.0,
            vtl_adjustment: 0.0,
            quality: VoiceQuality::default(),
        }
    }
}

/// Voice quality settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceQuality {
    /// Breathiness (0.0 to 1.0)
    pub breathiness: f32,
    /// Roughness (0.0 to 1.0)  
    pub roughness: f32,
    /// Tenseness (0.0 to 1.0)
    pub tenseness: f32,
    /// Creakiness (0.0 to 1.0)
    pub creakiness: f32,
}

impl Default for VoiceQuality {
    fn default() -> Self {
        Self {
            breathiness: 0.0,
            roughness: 0.0,
            tenseness: 0.5,
            creakiness: 0.0,
        }
    }
}

/// Prosodic modification settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodyConfig {
    /// Speaking rate multiplier (0.5 to 2.0)
    pub speaking_rate: f32,
    /// Pitch range scaling (0.5 to 2.0)
    pub pitch_range: f32,
    /// Rhythm modifications
    pub rhythm: RhythmConfig,
    /// Intonation adjustments
    pub intonation: IntonationConfig,
}

impl Default for ProsodyConfig {
    fn default() -> Self {
        Self {
            speaking_rate: 1.0,
            pitch_range: 1.0,
            rhythm: RhythmConfig::default(),
            intonation: IntonationConfig::default(),
        }
    }
}

/// Rhythm modification settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmConfig {
    /// Stress emphasis (0.0 to 1.0)
    pub stress_emphasis: f32,
    /// Pause insertion probability (0.0 to 1.0)
    pub pause_insertion: f32,
    /// Syllable timing variation (0.0 to 1.0)
    pub timing_variation: f32,
}

impl Default for RhythmConfig {
    fn default() -> Self {
        Self {
            stress_emphasis: 0.5,
            pause_insertion: 0.0,
            timing_variation: 0.0,
        }
    }
}

/// Intonation adjustment settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntonationConfig {
    /// Question intonation strength (0.0 to 1.0)
    pub question_strength: f32,
    /// Exclamation intonation strength (0.0 to 1.0)
    pub exclamation_strength: f32,
    /// Statement intonation adjustment (-1.0 to 1.0)
    pub statement_adjustment: f32,
}

impl Default for IntonationConfig {
    fn default() -> Self {
        Self {
            question_strength: 0.0,
            exclamation_strength: 0.0,
            statement_adjustment: 0.0,
        }
    }
}

/// Audio enhancement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancementConfig {
    /// Noise reduction strength (0.0 to 1.0)
    pub noise_reduction: f32,
    /// Dynamic range compression (0.0 to 1.0)
    pub compression: f32,
    /// Spectral enhancement (0.0 to 1.0)
    pub spectral_enhancement: f32,
    /// Stereo widening (0.0 to 1.0)
    pub stereo_widening: f32,
    /// Reverb settings
    pub reverb: Option<ReverbConfig>,
}

impl Default for EnhancementConfig {
    fn default() -> Self {
        Self {
            noise_reduction: 0.0,
            compression: 0.0,
            spectral_enhancement: 0.0,
            stereo_widening: 0.0,
            reverb: None,
        }
    }
}

/// Reverb configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReverbConfig {
    /// Room size (0.0 to 1.0)
    pub room_size: f32,
    /// Damping factor (0.0 to 1.0)
    pub damping: f32,
    /// Wet/dry mix (0.0 to 1.0)
    pub wet_level: f32,
    /// Pre-delay in milliseconds
    pub pre_delay_ms: f32,
}

/// Feature weights for resolving conflicts between multiple conditioning features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureWeights {
    /// Weight for emotion-based modifications
    pub emotion_weight: f32,
    /// Weight for voice conversion modifications
    pub voice_conversion_weight: f32,
    /// Weight for speaker-specific modifications
    pub speaker_weight: f32,
    /// Weight for prosodic modifications
    pub prosody_weight: f32,
    /// Weight for audio enhancement
    pub enhancement_weight: f32,
}

impl Default for FeatureWeights {
    fn default() -> Self {
        Self {
            emotion_weight: 1.0,
            voice_conversion_weight: 1.0,
            speaker_weight: 1.0,
            prosody_weight: 1.0,
            enhancement_weight: 0.8, // Slightly lower to preserve voice characteristics
        }
    }
}

/// Unified conditioning processor
pub struct VocoderConditioner {
    config: VocoderConditioningConfig,
    sample_rate: u32,
    // Processing state
    prosody_state: ProsodyProcessingState,
    enhancement_state: EnhancementProcessingState,
}

/// State for prosody processing
#[derive(Debug)]
struct ProsodyProcessingState {
    speaking_rate_buffer: Vec<f32>,
    pitch_buffer: Vec<f32>,
    timing_state: TimingState,
}

/// Timing state for prosodic modifications
#[derive(Debug)]
struct TimingState {
    current_position: f32,
    target_position: f32,
    rate_adjustment: f32,
}

/// State for audio enhancement processing  
#[derive(Debug)]
struct EnhancementProcessingState {
    noise_reduction_state: NoiseReductionState,
    compressor_state: CompressorState,
    reverb_state: Option<ReverbState>,
}

/// Noise reduction processing state
#[derive(Debug)]
struct NoiseReductionState {
    noise_floor: f32,
    adaptation_rate: f32,
    history_buffer: Vec<f32>,
}

/// Compressor processing state
#[derive(Debug)]
struct CompressorState {
    envelope: f32,
    attack_coeff: f32,
    release_coeff: f32,
}

/// Reverb processing state
#[derive(Debug)]
struct ReverbState {
    delay_lines: Vec<Vec<f32>>,
    delay_positions: Vec<usize>,
    feedback_gains: Vec<f32>,
}

impl VocoderConditioner {
    /// Create a new conditioning processor
    pub fn new(config: VocoderConditioningConfig, sample_rate: u32) -> Self {
        let prosody_state = ProsodyProcessingState {
            speaking_rate_buffer: vec![0.0; (sample_rate as f32 * 0.1) as usize], // 100ms buffer
            pitch_buffer: vec![0.0; (sample_rate as f32 * 0.05) as usize],        // 50ms buffer
            timing_state: TimingState {
                current_position: 0.0,
                target_position: 0.0,
                rate_adjustment: config.prosody.as_ref().map_or(1.0, |p| p.speaking_rate),
            },
        };

        let enhancement_state = EnhancementProcessingState {
            noise_reduction_state: NoiseReductionState {
                noise_floor: -40.0, // -40 dB default noise floor
                adaptation_rate: 0.01,
                history_buffer: vec![0.0; 512],
            },
            compressor_state: CompressorState {
                envelope: 0.0,
                attack_coeff: 0.003,   // ~1ms attack at 44.1kHz
                release_coeff: 0.0001, // ~100ms release at 44.1kHz
            },
            reverb_state: config
                .enhancement
                .as_ref()
                .and_then(|e| e.reverb.as_ref())
                .map(|_| ReverbState {
                    delay_lines: vec![vec![0.0; 1024]; 4], // 4 delay lines
                    delay_positions: vec![0; 4],
                    feedback_gains: vec![0.7, 0.6, 0.5, 0.4],
                }),
        };

        Self {
            config,
            sample_rate,
            prosody_state,
            enhancement_state,
        }
    }

    /// Update conditioning configuration
    pub fn update_config(&mut self, config: VocoderConditioningConfig) {
        self.config = config;

        // Update prosody state if needed
        if let Some(prosody) = &self.config.prosody {
            self.prosody_state.timing_state.rate_adjustment = prosody.speaking_rate;
        }

        // Update reverb state if needed
        if let Some(enhancement) = &self.config.enhancement {
            if enhancement.reverb.is_some() && self.enhancement_state.reverb_state.is_none() {
                self.enhancement_state.reverb_state = Some(ReverbState {
                    delay_lines: vec![vec![0.0; 1024]; 4],
                    delay_positions: vec![0; 4],
                    feedback_gains: vec![0.7, 0.6, 0.5, 0.4],
                });
            } else if enhancement.reverb.is_none() {
                self.enhancement_state.reverb_state = None;
            }
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &VocoderConditioningConfig {
        &self.config
    }

    /// Apply all conditioning features to audio buffer
    pub fn process(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        if audio.is_empty() {
            return Ok(());
        }

        // Apply conditioning features in priority order based on weights
        let mut processing_order = vec![
            ("speaker", self.config.feature_weights.speaker_weight),
            ("emotion", self.config.feature_weights.emotion_weight),
            (
                "voice_conversion",
                self.config.feature_weights.voice_conversion_weight,
            ),
            ("prosody", self.config.feature_weights.prosody_weight),
            (
                "enhancement",
                self.config.feature_weights.enhancement_weight,
            ),
        ];

        // Sort by weight (highest first)
        processing_order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (feature, weight) in processing_order {
            if weight > 0.01 {
                // Only process if weight is significant
                match feature {
                    "speaker" => self.apply_speaker_conditioning(audio)?,
                    "emotion" => self.apply_emotion_conditioning(audio)?,
                    "voice_conversion" => self.apply_voice_conversion_conditioning(audio)?,
                    "prosody" => self.apply_prosody_conditioning(audio)?,
                    "enhancement" => self.apply_enhancement_conditioning(audio)?,
                    _ => {}
                }
            }
        }

        Ok(())
    }

    /// Apply speaker-specific conditioning
    fn apply_speaker_conditioning(&self, audio: &mut AudioBuffer) -> Result<()> {
        if let Some(ref speaker_config) = self.config.speaker {
            let intensity =
                self.config.global_strength * self.config.feature_weights.speaker_weight;

            // Apply voice characteristics
            let characteristics = &speaker_config.voice_characteristics;

            // F0 adjustment
            if characteristics.f0_adjustment.abs() > 0.01 {
                self.apply_f0_adjustment(audio, characteristics.f0_adjustment * intensity)?;
            }

            // VTL adjustment (simplified formant scaling)
            if characteristics.vtl_adjustment.abs() > 0.01 {
                self.apply_vtl_adjustment(audio, characteristics.vtl_adjustment * intensity)?;
            }

            // Voice quality adjustments
            let quality = &characteristics.quality;
            self.apply_voice_quality(audio, quality, intensity)?;
        }
        Ok(())
    }

    /// Apply emotion-based conditioning with advanced audio processing
    fn apply_emotion_conditioning(&self, audio: &mut AudioBuffer) -> Result<()> {
        if let Some(ref emotion_config) = self.config.emotion {
            let intensity = emotion_config.intensity.clamp(0.0, 1.0);
            let emotion_weight = self.config.feature_weights.emotion_weight;
            let effective_intensity = intensity * emotion_weight;

            // Apply emotion-specific audio processing
            match emotion_config.emotion_type.to_lowercase().as_str() {
                "happy" | "joy" | "excited" => {
                    self.apply_happy_emotion_processing(audio, effective_intensity)?;
                }
                "sad" | "melancholy" | "depressed" => {
                    self.apply_sad_emotion_processing(audio, effective_intensity)?;
                }
                "angry" | "rage" | "frustrated" => {
                    self.apply_angry_emotion_processing(audio, effective_intensity)?;
                }
                "calm" | "peaceful" | "relaxed" => {
                    self.apply_calm_emotion_processing(audio, effective_intensity)?;
                }
                "surprised" | "shocked" | "amazed" => {
                    self.apply_surprised_emotion_processing(audio, effective_intensity)?;
                }
                "fearful" | "scared" | "anxious" => {
                    self.apply_fearful_emotion_processing(audio, effective_intensity)?;
                }
                "neutral" | "normal" => {
                    // No processing for neutral emotion
                }
                _ => {
                    // Generic emotion processing for unknown emotions
                    self.apply_generic_emotion_processing(audio, effective_intensity)?;
                }
            }

            // Apply emotion vector if provided
            if let Some(ref emotion_vector) = emotion_config.emotion_vector {
                self.apply_emotion_vector_processing(audio, emotion_vector, effective_intensity)?;
            }
        }

        Ok(())
    }

    /// Apply voice conversion conditioning
    fn apply_voice_conversion_conditioning(&self, audio: &mut AudioBuffer) -> Result<()> {
        if let Some(ref config) = self.config.voice_conversion {
            let samples = &mut audio.samples;
            let sample_rate = audio.sample_rate as f32;

            // Apply formant scaling using spectral envelope modification
            if (config.formant_scaling - 1.0).abs() > 0.01 {
                self.apply_formant_scaling(samples, sample_rate, config.formant_scaling)?;
            }

            // Apply pitch modification if enabled
            if config.pitch_shift.abs() > 0.01 {
                self.apply_pitch_modification(samples, sample_rate, config.pitch_shift)?;
            }

            // Apply voice characteristics (brightness, warmth, etc.)
            self.apply_voice_characteristics_from_config(samples, sample_rate, config)?;

            // Apply age and gender transformations
            if config.age_shift.abs() > 0.01 || config.gender_shift.abs() > 0.01 {
                self.apply_age_gender_transformation(
                    samples,
                    sample_rate,
                    config.age_shift,
                    config.gender_shift,
                )?;
            }

            // Apply breathiness and roughness effects
            if config.breathiness > 0.01 || config.roughness > 0.01 {
                self.apply_voice_texture(
                    samples,
                    sample_rate,
                    config.breathiness,
                    config.roughness,
                )?;
            }
        }
        Ok(())
    }

    /// Apply formant scaling through spectral envelope modification
    fn apply_formant_scaling(
        &self,
        samples: &mut [f32],
        _sample_rate: f32,
        scaling_factor: f32,
    ) -> Result<()> {
        if samples.is_empty() || (scaling_factor - 1.0).abs() < 0.01 {
            return Ok(());
        }

        let frame_size = 1024;
        let hop_size = frame_size / 4;

        // Process in overlapping frames
        for i in (0..samples.len()).step_by(hop_size) {
            let end = (i + frame_size).min(samples.len());
            if end - i < frame_size / 2 {
                break;
            }

            let frame = &mut samples[i..end];
            self.process_frame_formant_scaling(frame, scaling_factor)?;
        }

        Ok(())
    }

    /// Process a single frame for formant scaling
    fn process_frame_formant_scaling(&self, frame: &mut [f32], scaling_factor: f32) -> Result<()> {
        let n = frame.len();
        if n < 64 {
            return Ok(());
        }

        // Apply Hann window
        for (i, sample) in frame.iter_mut().enumerate() {
            let window =
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos());
            *sample *= window;
        }

        // Apply formant scaling through frequency-selective processing
        // Formant frequencies typically at 800Hz, 1200Hz, 2600Hz for neutral vowels
        let formant_frequencies = [800.0, 1200.0, 2600.0];

        for &formant_freq in &formant_frequencies {
            let scaled_freq = formant_freq * scaling_factor;
            let normalized_freq = scaled_freq / (44100.0 / 2.0); // Normalize to Nyquist

            if normalized_freq < 1.0 {
                let freq_bin = (normalized_freq * n as f32 / 2.0) as usize;
                if freq_bin < n / 2 {
                    // Apply frequency-selective emphasis/de-emphasis
                    let gain = if scaling_factor > 1.0 { 1.3 } else { 0.7 };
                    let bandwidth = (20.0 * scaling_factor) as usize; // Scale bandwidth with formant shift

                    let start = freq_bin.saturating_sub(bandwidth);
                    let end = (freq_bin + bandwidth).min(n - 1) + 1;
                    for element in frame.iter_mut().skip(start).take(end - start) {
                        *element *= gain;
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply pitch modification
    fn apply_pitch_modification(
        &self,
        samples: &mut [f32],
        _sample_rate: f32,
        shift_semitones: f32,
    ) -> Result<()> {
        if samples.is_empty() || shift_semitones.abs() < 0.01 {
            return Ok(());
        }

        // Simple pitch shift simulation using time-domain stretching
        let pitch_factor = 2.0_f32.powf(shift_semitones / 12.0);
        let stretch_factor = 1.0 / pitch_factor;

        // Apply time-stretching effect
        let original_samples = samples.to_vec();
        for (i, sample) in samples.iter_mut().enumerate() {
            let source_index = (i as f32 * stretch_factor) as usize;
            if source_index < original_samples.len() {
                *sample = original_samples[source_index];
            } else {
                *sample = 0.0;
            }
        }

        Ok(())
    }

    /// Apply voice characteristics from configuration
    fn apply_voice_characteristics_from_config(
        &self,
        samples: &mut [f32],
        _sample_rate: f32,
        config: &crate::conversion::VoiceConversionConfig,
    ) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }

        // Apply brightness adjustment
        if config.brightness.abs() > 0.01 {
            let brightness_gain = 1.0 + config.brightness * 0.3;
            for sample in samples.iter_mut() {
                *sample *= brightness_gain;
            }
        }

        // Apply warmth adjustment (low-frequency emphasis)
        if config.warmth.abs() > 0.01 {
            let warmth_factor = 1.0 + config.warmth * 0.2;
            // Simple low-pass filtering effect
            let mut prev_sample = 0.0;
            for sample in samples.iter_mut() {
                let filtered = prev_sample * 0.1 + *sample * 0.9;
                *sample = *sample + (filtered - *sample) * warmth_factor * 0.3;
                prev_sample = *sample;
            }
        }

        Ok(())
    }

    /// Apply age and gender transformation
    fn apply_age_gender_transformation(
        &self,
        samples: &mut [f32],
        _sample_rate: f32,
        age_shift: f32,
        gender_shift: f32,
    ) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }

        // Age transformation affects formant frequencies and voice brightness
        if age_shift.abs() > 0.01 {
            let age_factor = 1.0 + age_shift * 0.2; // Younger = higher formants, older = lower
            for sample in samples.iter_mut() {
                *sample *= age_factor;
            }
        }

        // Gender transformation affects fundamental frequency and formant relationships
        if gender_shift.abs() > 0.01 {
            let gender_factor = 1.0 + gender_shift * 0.15; // Masculine = lower, feminine = higher
            let samples_len = samples.len() as f32;
            // Apply frequency-dependent scaling
            for (i, sample) in samples.iter_mut().enumerate() {
                let freq_weight = (i as f32 / samples_len).sqrt(); // Emphasize higher frequencies
                let scaling = 1.0 + gender_shift * 0.1 * freq_weight;
                *sample *= gender_factor * scaling;
            }
        }

        Ok(())
    }

    /// Apply voice texture effects (breathiness and roughness)
    fn apply_voice_texture(
        &self,
        samples: &mut [f32],
        _sample_rate: f32,
        breathiness: f32,
        roughness: f32,
    ) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }

        // Breathiness adds high-frequency noise and reduces harmonic content
        if breathiness > 0.01 {
            for (i, sample) in samples.iter_mut().enumerate() {
                // Add controlled noise for breathiness effect
                let noise = (i as f32 * 0.1).sin() * 0.05 * breathiness;
                *sample = *sample * (1.0 - breathiness * 0.3) + noise;
            }
        }

        // Roughness adds irregular amplitude modulation
        if roughness > 0.01 {
            for (i, sample) in samples.iter_mut().enumerate() {
                // Add roughness through amplitude modulation
                let modulation = 1.0 + roughness * 0.2 * (i as f32 * 0.03).sin();
                *sample *= modulation;
            }
        }

        Ok(())
    }

    /// Apply prosodic conditioning
    fn apply_prosody_conditioning(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        // Extract values to avoid borrow conflicts
        let (speaking_rate, pitch_range, intensity) =
            if let Some(ref prosody_config) = self.config.prosody {
                let intensity =
                    self.config.global_strength * self.config.feature_weights.prosody_weight;
                (
                    prosody_config.speaking_rate,
                    prosody_config.pitch_range,
                    intensity,
                )
            } else {
                return Ok(());
            };

        // Apply speaking rate changes
        if (speaking_rate - 1.0).abs() > 0.01 {
            self.apply_speaking_rate_change(audio, speaking_rate, intensity)?;
        }

        // Apply pitch range scaling
        if (pitch_range - 1.0).abs() > 0.01 {
            self.apply_pitch_range_scaling(audio, pitch_range, intensity)?;
        }

        Ok(())
    }

    /// Apply audio enhancement conditioning
    fn apply_enhancement_conditioning(&mut self, audio: &mut AudioBuffer) -> Result<()> {
        // Extract values to avoid borrow conflicts
        let (noise_reduction, compression, spectral_enhancement, reverb_config, intensity) =
            if let Some(ref enhancement_config) = self.config.enhancement {
                let intensity =
                    self.config.global_strength * self.config.feature_weights.enhancement_weight;
                (
                    enhancement_config.noise_reduction,
                    enhancement_config.compression,
                    enhancement_config.spectral_enhancement,
                    enhancement_config.reverb.clone(),
                    intensity,
                )
            } else {
                return Ok(());
            };

        // Apply noise reduction
        if noise_reduction > 0.01 {
            self.apply_noise_reduction(audio, noise_reduction * intensity)?;
        }

        // Apply compression
        if compression > 0.01 {
            self.apply_compression(audio, compression * intensity)?;
        }

        // Apply spectral enhancement
        if spectral_enhancement > 0.01 {
            self.apply_spectral_enhancement(audio, spectral_enhancement * intensity)?;
        }

        // Apply reverb
        if let Some(reverb_config) = reverb_config {
            self.apply_reverb(audio, &reverb_config, intensity)?;
        }

        Ok(())
    }

    /// Apply F0 (fundamental frequency) adjustment
    fn apply_f0_adjustment(&self, audio: &mut AudioBuffer, adjustment: f32) -> Result<()> {
        // Simplified pitch shifting for F0 adjustment
        let pitch_shift_semitones = adjustment * 12.0; // Convert to semitones
        let pitch_ratio = 2.0_f32.powf(pitch_shift_semitones / 12.0);

        let samples = audio.samples().to_vec();
        let mut processed_samples = Vec::with_capacity(samples.len());

        for (i, _) in samples.iter().enumerate() {
            let source_pos = (i as f32) * pitch_ratio;
            let source_idx = source_pos as usize;
            let source_frac = source_pos - source_idx as f32;

            if source_idx + 1 < samples.len() {
                let interpolated = samples[source_idx] * (1.0 - source_frac)
                    + samples[source_idx + 1] * source_frac;
                processed_samples.push(interpolated);
            } else {
                processed_samples.push(samples.get(source_idx).copied().unwrap_or(0.0));
            }
        }

        *audio = AudioBuffer::new(processed_samples, audio.sample_rate(), audio.channels());
        Ok(())
    }

    /// Apply vocal tract length adjustment
    fn apply_vtl_adjustment(&self, audio: &mut AudioBuffer, adjustment: f32) -> Result<()> {
        // Simplified formant scaling
        let formant_scale = 1.0 + adjustment * 0.2; // ±20% formant scaling

        // Apply simple spectral emphasis/de-emphasis
        let samples = audio.samples_mut();
        let alpha = (formant_scale - 1.0) * 0.1;

        for i in 1..samples.len() {
            let high_freq = samples[i] - samples[i - 1];
            samples[i] = (samples[i] + alpha * high_freq).clamp(-1.0, 1.0);
        }

        Ok(())
    }

    /// Apply voice quality modifications
    fn apply_voice_quality(
        &self,
        audio: &mut AudioBuffer,
        quality: &VoiceQuality,
        intensity: f32,
    ) -> Result<()> {
        let samples = audio.samples_mut();

        // Apply breathiness
        if quality.breathiness > 0.01 {
            let noise_level = quality.breathiness * intensity * 0.03;
            for (i, sample) in samples.iter_mut().enumerate() {
                let t = i as f32 * 0.001;
                let noise = (t * 1847.0).sin() * 0.3 + (t * 3271.0).sin() * 0.2;
                let breath_noise = noise * noise_level * sample.abs().sqrt();
                *sample = (*sample + breath_noise).clamp(-1.0, 1.0);
            }
        }

        // Apply roughness
        if quality.roughness > 0.01 {
            let distortion_amount = quality.roughness * intensity;
            for sample in samples.iter_mut() {
                if sample.abs() > 0.1 {
                    let sign = sample.signum();
                    let distortion_factor = 1.0 - distortion_amount * 0.2;
                    let distorted = sample.abs().powf(distortion_factor);
                    *sample = (sign * distorted).clamp(-1.0, 1.0);
                }
            }
        }

        Ok(())
    }

    /// Apply speaking rate changes
    fn apply_speaking_rate_change(
        &mut self,
        audio: &mut AudioBuffer,
        rate: f32,
        intensity: f32,
    ) -> Result<()> {
        let effective_rate = 1.0 + (rate - 1.0) * intensity;

        // Simple time-stretching implementation
        let samples = audio.samples().to_vec();
        let mut processed_samples = Vec::new();

        let mut read_pos = 0.0;
        while (read_pos as usize) < samples.len() {
            let idx = read_pos as usize;
            let frac = read_pos - idx as f32;

            if idx + 1 < samples.len() {
                let interpolated = samples[idx] * (1.0 - frac) + samples[idx + 1] * frac;
                processed_samples.push(interpolated);
            } else if idx < samples.len() {
                processed_samples.push(samples[idx]);
            }

            read_pos += effective_rate;
        }

        *audio = AudioBuffer::new(processed_samples, audio.sample_rate(), audio.channels());
        Ok(())
    }

    /// Apply pitch range scaling
    fn apply_pitch_range_scaling(
        &self,
        audio: &mut AudioBuffer,
        range_scale: f32,
        intensity: f32,
    ) -> Result<()> {
        // Simplified pitch range modification through spectral emphasis
        let effective_scale = 1.0 + (range_scale - 1.0) * intensity;
        let emphasis = (effective_scale - 1.0) * 0.1;

        let samples = audio.samples_mut();
        for i in 1..samples.len() {
            let pitch_component = samples[i] - samples[i - 1];
            samples[i] = (samples[i] + emphasis * pitch_component).clamp(-1.0, 1.0);
        }

        Ok(())
    }

    /// Apply adaptive noise reduction
    fn apply_noise_reduction(&mut self, audio: &mut AudioBuffer, strength: f32) -> Result<()> {
        let samples = audio.samples_mut();
        let noise_state = &mut self.enhancement_state.noise_reduction_state;

        // Adaptive noise floor estimation
        let mut signal_energy = 0.0;
        let mut noise_samples = 0;

        // First pass: estimate current noise floor
        for sample in samples.iter() {
            let abs_sample = sample.abs();
            if abs_sample < 0.1 {
                // Likely noise
                signal_energy += abs_sample * abs_sample;
                noise_samples += 1;
            }
        }

        if noise_samples > 0 {
            let current_noise_floor = (signal_energy / noise_samples as f32).sqrt();
            let current_noise_db = 20.0 * current_noise_floor.max(1e-10).log10();

            // Adaptive update of noise floor
            noise_state.noise_floor = noise_state.noise_floor * (1.0 - noise_state.adaptation_rate)
                + current_noise_db * noise_state.adaptation_rate;
        }

        // Apply noise reduction with sample rate compensation
        let sample_rate_factor = self.sample_rate as f32 / 44100.0;
        let adjusted_strength = strength * sample_rate_factor.sqrt();

        let noise_threshold = 10.0_f32.powf(noise_state.noise_floor / 20.0);
        let gate_threshold = noise_threshold * (1.0 + adjusted_strength);

        // Update history buffer and apply spectral gate
        for (i, sample) in samples.iter_mut().enumerate() {
            let abs_sample = sample.abs();

            // Update history buffer
            let hist_idx = i % noise_state.history_buffer.len();
            noise_state.history_buffer[hist_idx] = abs_sample;

            // Apply noise gate with smoothing
            if abs_sample < gate_threshold {
                let reduction_factor = 1.0 - adjusted_strength * 0.8;
                *sample *= reduction_factor;
            }
        }

        Ok(())
    }

    /// Apply dynamic range compression with attack/release
    fn apply_compression(&mut self, audio: &mut AudioBuffer, ratio: f32) -> Result<()> {
        let samples = audio.samples_mut();
        let compressor_state = &mut self.enhancement_state.compressor_state;

        // Dynamic threshold and ratio based on sample rate
        let sample_rate_factor = self.sample_rate as f32 / 44100.0;
        let threshold = 0.7; // -3dB threshold
        let effective_ratio = 1.0 + ratio * 3.0; // 1:1 to 4:1 compression

        // Adjust attack/release coefficients for sample rate
        let attack_coeff = compressor_state.attack_coeff * sample_rate_factor;
        let release_coeff = compressor_state.release_coeff * sample_rate_factor;

        for sample in samples.iter_mut() {
            let abs_sample = sample.abs();

            // Update envelope with attack/release characteristics
            if abs_sample > compressor_state.envelope {
                // Attack phase - fast response to signal increases
                compressor_state.envelope =
                    compressor_state.envelope * (1.0 - attack_coeff) + abs_sample * attack_coeff;
            } else {
                // Release phase - slow response to signal decreases
                compressor_state.envelope =
                    compressor_state.envelope * (1.0 - release_coeff) + abs_sample * release_coeff;
            }

            // Apply compression based on envelope
            if compressor_state.envelope > threshold {
                let excess = compressor_state.envelope - threshold;
                let compressed_excess = excess / effective_ratio;
                let gain_reduction =
                    (threshold + compressed_excess) / compressor_state.envelope.max(1e-10);

                *sample *= gain_reduction;
            }
        }

        Ok(())
    }

    /// Apply spectral enhancement
    fn apply_spectral_enhancement(&self, audio: &mut AudioBuffer, strength: f32) -> Result<()> {
        let samples = audio.samples_mut();

        // High-frequency emphasis
        for i in 1..samples.len() {
            let high_freq = samples[i] - samples[i - 1];
            samples[i] = (samples[i] + strength * 0.1 * high_freq).clamp(-1.0, 1.0);
        }

        Ok(())
    }

    /// Apply reverb effect
    fn apply_reverb(
        &mut self,
        audio: &mut AudioBuffer,
        reverb_config: &ReverbConfig,
        intensity: f32,
    ) -> Result<()> {
        if let Some(ref mut reverb_state) = self.enhancement_state.reverb_state {
            let samples = audio.samples_mut();
            let wet_level = reverb_config.wet_level * intensity;

            for sample in samples.iter_mut() {
                let mut reverb_sum = 0.0;

                // Process each delay line
                for (i, delay_line) in reverb_state.delay_lines.iter_mut().enumerate() {
                    let delay_pos = reverb_state.delay_positions[i];
                    let delayed_sample = delay_line[delay_pos];

                    // Add input with feedback
                    delay_line[delay_pos] = *sample
                        + delayed_sample * reverb_state.feedback_gains[i] * reverb_config.damping;

                    reverb_sum += delayed_sample;

                    // Update delay position
                    reverb_state.delay_positions[i] = (delay_pos + 1) % delay_line.len();
                }

                // Mix with original signal
                *sample = sample.mul_add(1.0 - wet_level, reverb_sum * wet_level * 0.25);
            }
        }

        Ok(())
    }

    /// Apply happy emotion processing - brighter, more energetic sound
    fn apply_happy_emotion_processing(
        &self,
        audio: &mut AudioBuffer,
        intensity: f32,
    ) -> Result<()> {
        let samples = audio.samples_mut();

        // Add brightness and energy
        let brightness_gain = 1.0 + intensity * 0.2;
        let dynamic_range = 1.0 + intensity * 0.1;

        for sample in samples.iter_mut() {
            // Increase amplitude and add slight harmonic enhancement
            *sample = (*sample * brightness_gain * dynamic_range).clamp(-1.0, 1.0);
        }

        // Add high-frequency emphasis for sparkle
        for i in 1..samples.len() {
            let high_freq = samples[i] - samples[i - 1];
            samples[i] = (samples[i] + intensity * 0.05 * high_freq).clamp(-1.0, 1.0);
        }

        Ok(())
    }

    /// Apply sad emotion processing - warmer, softer, reduced dynamics
    fn apply_sad_emotion_processing(&self, audio: &mut AudioBuffer, intensity: f32) -> Result<()> {
        let samples = audio.samples_mut();

        // Reduce dynamics and brightness
        let dampening = 1.0 - intensity * 0.15;
        let warmth_filter = intensity * 0.1;

        let mut prev_sample = 0.0;
        for sample in samples.iter_mut() {
            // Apply low-pass filtering for warmth
            let filtered = prev_sample + warmth_filter * (*sample - prev_sample);
            *sample = (filtered * dampening).clamp(-1.0, 1.0);
            prev_sample = *sample;
        }

        Ok(())
    }

    /// Apply angry emotion processing - more aggressive, distorted
    fn apply_angry_emotion_processing(
        &self,
        audio: &mut AudioBuffer,
        intensity: f32,
    ) -> Result<()> {
        let samples = audio.samples_mut();

        // Add controlled distortion and compression
        let distortion_amount = intensity * 0.3;
        let compression_ratio = 1.0 + intensity * 0.5;

        for sample in samples.iter_mut() {
            // Apply soft clipping distortion
            let distorted = if sample.abs() > 0.5 {
                sample.signum() * (0.5 + (sample.abs() - 0.5) * distortion_amount)
            } else {
                *sample
            };

            // Apply compression
            *sample = (distorted * compression_ratio).clamp(-1.0, 1.0);
        }

        Ok(())
    }

    /// Apply calm emotion processing - smooth, relaxed sound
    fn apply_calm_emotion_processing(&self, audio: &mut AudioBuffer, intensity: f32) -> Result<()> {
        let samples = audio.samples_mut();

        // Smooth dynamics and reduce harsh frequencies
        let smoothing_factor = intensity * 0.2;
        let gentleness = 1.0 - intensity * 0.05;

        // Apply smoothing filter
        for i in 1..samples.len() {
            let smoothed = samples[i - 1] + smoothing_factor * (samples[i] - samples[i - 1]);
            samples[i] = (smoothed * gentleness).clamp(-1.0, 1.0);
        }

        Ok(())
    }

    /// Apply surprised emotion processing - sudden dynamics, emphasis
    fn apply_surprised_emotion_processing(
        &self,
        audio: &mut AudioBuffer,
        intensity: f32,
    ) -> Result<()> {
        let samples = audio.samples_mut();

        // Add sudden dynamic changes and transient emphasis
        let transient_boost = 1.0 + intensity * 0.25;
        let dynamic_variation = intensity * 0.15;

        for i in 1..samples.len() {
            // Emphasize transients (sudden changes)
            let transient = (samples[i] - samples[i - 1]).abs();
            if transient > 0.1 {
                samples[i] = (samples[i] * transient_boost).clamp(-1.0, 1.0);
            }

            // Add slight random variation
            let variation = (i as f32 * 0.1).sin() * dynamic_variation * 0.1;
            samples[i] = (samples[i] + variation).clamp(-1.0, 1.0);
        }

        Ok(())
    }

    /// Apply fearful emotion processing - trembling, uncertain sound
    fn apply_fearful_emotion_processing(
        &self,
        audio: &mut AudioBuffer,
        intensity: f32,
    ) -> Result<()> {
        let samples = audio.samples_mut();

        // Add tremolo and reduce confidence
        let tremolo_rate = 6.0; // Hz
        let tremolo_depth = intensity * 0.1;
        let uncertainty_reduction = 1.0 - intensity * 0.1;

        for (i, sample) in samples.iter_mut().enumerate() {
            // Apply tremolo effect
            let tremolo_phase =
                (i as f32 / self.sample_rate as f32) * tremolo_rate * 2.0 * std::f32::consts::PI;
            let tremolo_mod = 1.0 + tremolo_depth * tremolo_phase.sin();

            *sample = (*sample * tremolo_mod * uncertainty_reduction).clamp(-1.0, 1.0);
        }

        Ok(())
    }

    /// Apply generic emotion processing for unknown emotions
    fn apply_generic_emotion_processing(
        &self,
        audio: &mut AudioBuffer,
        intensity: f32,
    ) -> Result<()> {
        let samples = audio.samples_mut();

        // Apply gentle enhancement based on intensity
        let enhancement_factor = 1.0 + intensity * 0.05;

        for sample in samples.iter_mut() {
            *sample = (*sample * enhancement_factor).clamp(-1.0, 1.0);
        }

        Ok(())
    }

    /// Apply emotion vector processing for multidimensional emotion control
    fn apply_emotion_vector_processing(
        &self,
        audio: &mut AudioBuffer,
        emotion_vector: &[f32],
        intensity: f32,
    ) -> Result<()> {
        if emotion_vector.is_empty() {
            return Ok(());
        }

        let samples = audio.samples_mut();
        let samples_len = samples.len();

        // Apply weighted combination of emotion dimensions
        for (i, sample) in samples.iter_mut().enumerate() {
            let mut processed_sample = *sample;

            // Apply each emotion dimension
            for (dim_idx, &emotion_weight) in emotion_vector.iter().enumerate() {
                if emotion_weight.abs() > 0.01 {
                    let dim_effect = match dim_idx % 6 {
                        0 => emotion_weight * intensity * 0.1,  // Valence
                        1 => emotion_weight * intensity * 0.08, // Arousal
                        2 => emotion_weight * intensity * 0.06, // Dominance
                        3 => emotion_weight * intensity * 0.05, // Tension
                        4 => emotion_weight * intensity * 0.04, // Energy
                        _ => emotion_weight * intensity * 0.03, // Other dimensions
                    };

                    // Apply modulation based on position in audio
                    let phase = (i as f32 / samples_len as f32) * 2.0 * std::f32::consts::PI;
                    let modulation = 1.0 + dim_effect * phase.sin();
                    processed_sample *= modulation;
                }
            }

            *sample = processed_sample.clamp(-1.0, 1.0);
        }

        Ok(())
    }

    /// Reset all processing state
    pub fn reset(&mut self) {
        self.prosody_state.speaking_rate_buffer.fill(0.0);
        self.prosody_state.pitch_buffer.fill(0.0);
        self.prosody_state.timing_state.current_position = 0.0;
        self.prosody_state.timing_state.target_position = 0.0;

        self.enhancement_state
            .noise_reduction_state
            .history_buffer
            .fill(0.0);
        self.enhancement_state.compressor_state.envelope = 0.0;

        if let Some(ref mut reverb_state) = self.enhancement_state.reverb_state {
            for delay_line in &mut reverb_state.delay_lines {
                delay_line.fill(0.0);
            }
            reverb_state.delay_positions.fill(0);
        }
    }
}

/// Builder for creating conditioning configurations
pub struct ConditioningConfigBuilder {
    config: VocoderConditioningConfig,
}

impl ConditioningConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: VocoderConditioningConfig::default(),
        }
    }

    /// Set emotion configuration
    pub fn emotion(mut self, emotion: EmotionConfig) -> Self {
        self.config.emotion = Some(emotion);
        self
    }

    /// Set voice conversion configuration
    pub fn voice_conversion(mut self, conversion: VoiceConversionConfig) -> Self {
        self.config.voice_conversion = Some(conversion);
        self
    }

    /// Set speaker configuration
    pub fn speaker(mut self, speaker: SpeakerConfig) -> Self {
        self.config.speaker = Some(speaker);
        self
    }

    /// Set prosody configuration
    pub fn prosody(mut self, prosody: ProsodyConfig) -> Self {
        self.config.prosody = Some(prosody);
        self
    }

    /// Set enhancement configuration
    pub fn enhancement(mut self, enhancement: EnhancementConfig) -> Self {
        self.config.enhancement = Some(enhancement);
        self
    }

    /// Set global conditioning strength
    pub fn global_strength(mut self, strength: f32) -> Self {
        self.config.global_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set feature weights
    pub fn feature_weights(mut self, weights: FeatureWeights) -> Self {
        self.config.feature_weights = weights;
        self
    }

    /// Build the configuration
    pub fn build(self) -> VocoderConditioningConfig {
        self.config
    }
}

impl Default for ConditioningConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversion::VoiceConversionConfig;
    use crate::hifigan::EmotionConfig;

    #[test]
    fn test_conditioning_config_builder() {
        let emotion = EmotionConfig::new("happy".to_string(), 0.7);
        let voice_conversion = VoiceConversionConfig::feminine(0.5);

        let config = ConditioningConfigBuilder::new()
            .emotion(emotion)
            .voice_conversion(voice_conversion)
            .global_strength(0.8)
            .build();

        assert!(config.emotion.is_some());
        assert!(config.voice_conversion.is_some());
        assert_eq!(config.global_strength, 0.8);
    }

    #[test]
    fn test_voice_characteristics_default() {
        let characteristics = VoiceCharacteristics::default();
        assert_eq!(characteristics.f0_adjustment, 0.0);
        assert_eq!(characteristics.vtl_adjustment, 0.0);
        assert_eq!(characteristics.quality.tenseness, 0.5);
    }

    #[test]
    fn test_feature_weights_default() {
        let weights = FeatureWeights::default();
        assert_eq!(weights.emotion_weight, 1.0);
        assert_eq!(weights.voice_conversion_weight, 1.0);
        assert_eq!(weights.enhancement_weight, 0.8); // Should be lower
    }

    #[test]
    fn test_conditioning_processor_creation() {
        let config = VocoderConditioningConfig::default();
        let processor = VocoderConditioner::new(config, 22050);
        assert_eq!(processor.sample_rate, 22050);
    }
}
