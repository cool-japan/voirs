//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::conditioning::{VocoderConditioner, VocoderConditioningConfig};
use crate::conversion::{VoiceConversionConfig, VoiceConverter};
use crate::effects::{EffectChain, EffectPresets};
#[cfg(feature = "candle")]
use crate::models::hifigan::generator::HiFiGanGenerator;
#[cfg(feature = "candle")]
use crate::models::hifigan::inference::HiFiGanInference;
use crate::models::hifigan::{HiFiGanConfig, HiFiGanVariant, HiFiGanVariants};
use crate::{
    AudioBuffer, MelSpectrogram, Result, SynthesisConfig, Vocoder, VocoderError, VocoderFeature,
    VocoderMetadata,
};
use async_trait::async_trait;
use std::collections::VecDeque;

/// Performance metrics for streaming vocoder
#[derive(Debug, Clone)]
pub struct StreamingMetrics {
    pub avg_processing_time_ms: f32,
    pub max_processing_time_ms: f32,
    pub total_latency_ms: f32,
    pub processed_frames: usize,
    pub buffer_utilization: f32,
    pub chunk_size: usize,
    pub overlap_size: usize,
}
/// Emotion configuration for vocoder conditioning
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmotionConfig {
    /// Emotion type
    pub emotion_type: String,
    /// Emotion intensity (0.0 to 1.0)
    pub intensity: f32,
    /// Emotion vector (alternative to type + intensity)
    pub emotion_vector: Option<Vec<f32>>,
}
impl EmotionConfig {
    /// Create new emotion configuration
    pub fn new(emotion_type: String, intensity: f32) -> Self {
        Self {
            emotion_type,
            intensity,
            emotion_vector: None,
        }
    }
    /// Create configuration with emotion vector
    pub fn with_vector(emotion_vector: Vec<f32>) -> Self {
        Self {
            emotion_type: "custom".to_string(),
            intensity: 1.0,
            emotion_vector: Some(emotion_vector),
        }
    }
}
/// Emotion-specific vocoding parameters
#[derive(Debug, Clone)]
pub struct EmotionVocodingParams {
    /// Spectral filtering adjustments
    pub spectral_tilt: f32,
    /// Harmonic enhancement
    pub harmonic_boost: f32,
    /// Formant shifting
    pub formant_shift: f32,
    /// Brightness adjustment
    pub brightness: f32,
    /// Breathiness simulation
    pub breathiness: f32,
    /// Voice quality roughness
    pub roughness: f32,
}
impl EmotionVocodingParams {
    /// Create parameters for specific emotion
    pub fn for_emotion(emotion_type: &str, intensity: f32) -> Self {
        let base_params = match emotion_type {
            "happy" => Self {
                spectral_tilt: 0.1,
                harmonic_boost: 0.2,
                formant_shift: 0.05,
                brightness: 0.15,
                breathiness: -0.05,
                roughness: -0.1,
            },
            "sad" => Self {
                spectral_tilt: -0.1,
                harmonic_boost: -0.1,
                formant_shift: -0.03,
                brightness: -0.2,
                breathiness: 0.1,
                roughness: 0.05,
            },
            "angry" => Self {
                spectral_tilt: 0.15,
                harmonic_boost: 0.3,
                formant_shift: 0.1,
                brightness: 0.25,
                breathiness: -0.1,
                roughness: 0.2,
            },
            "calm" => Self {
                spectral_tilt: -0.05,
                harmonic_boost: 0.05,
                formant_shift: 0.0,
                brightness: -0.05,
                breathiness: 0.05,
                roughness: -0.15,
            },
            _ => Self::neutral(),
        };
        Self {
            spectral_tilt: base_params.spectral_tilt * intensity,
            harmonic_boost: base_params.harmonic_boost * intensity,
            formant_shift: base_params.formant_shift * intensity,
            brightness: base_params.brightness * intensity,
            breathiness: base_params.breathiness * intensity,
            roughness: base_params.roughness * intensity,
        }
    }
    /// Create neutral parameters (no modification)
    pub fn neutral() -> Self {
        Self {
            spectral_tilt: 0.0,
            harmonic_boost: 0.0,
            formant_shift: 0.0,
            brightness: 0.0,
            breathiness: 0.0,
            roughness: 0.0,
        }
    }
}
/// HiFi-GAN vocoder implementation
pub struct HiFiGanVocoder {
    /// Configuration
    pub(super) config: HiFiGanConfig,
    /// Inference engine
    #[cfg(feature = "candle")]
    pub(super) inference: Option<HiFiGanInference>,
    /// Vocoder metadata
    pub(super) metadata: VocoderMetadata,
    /// Audio post-processing effect chain
    pub(super) effect_chain: EffectChain,
    /// Current emotion configuration for conditioning
    pub(super) current_emotion: Option<EmotionConfig>,
    /// Emotion-specific processing parameters
    pub(super) emotion_params: Option<EmotionVocodingParams>,
    /// Voice conversion processor
    pub(super) voice_converter: Option<VoiceConverter>,
    /// Current voice conversion configuration
    pub(super) voice_conversion_config: Option<VoiceConversionConfig>,
    /// Unified conditioning processor
    pub(super) unified_conditioner: Option<VocoderConditioner>,
}
impl HiFiGanVocoder {
    /// Create new HiFi-GAN vocoder with default V1 configuration
    pub fn new() -> Self {
        let config = HiFiGanVariants::v1();
        Self::with_config(config)
    }
    /// Create HiFi-GAN vocoder with specific configuration
    pub fn with_config(config: HiFiGanConfig) -> Self {
        let metadata = VocoderMetadata {
            name: config.variant.name().to_string(),
            version: "1.0.0".to_string(),
            architecture: "HiFi-GAN".to_string(),
            sample_rate: config.sample_rate,
            mel_channels: config.mel_channels,
            latency_ms: match config.variant {
                HiFiGanVariant::V1 => 8.0,
                HiFiGanVariant::V2 => 6.0,
                HiFiGanVariant::V3 => 4.0,
            },
            quality_score: match config.variant {
                HiFiGanVariant::V1 => 4.5,
                HiFiGanVariant::V2 => 4.0,
                HiFiGanVariant::V3 => 3.5,
            },
        };
        Self {
            #[cfg(feature = "candle")]
            inference: None,
            metadata,
            effect_chain: EffectPresets::speech_enhancement(config.sample_rate),
            config,
            current_emotion: None,
            emotion_params: None,
            voice_converter: None,
            voice_conversion_config: None,
            unified_conditioner: None,
        }
    }
    /// Create HiFi-GAN vocoder with specific variant
    pub fn with_variant(variant: HiFiGanVariant) -> Self {
        let config = HiFiGanVariants::get_variant(variant);
        Self::with_config(config)
    }
    /// Load HiFi-GAN model from file
    #[cfg(feature = "candle")]
    pub fn load_from_file(path: &str) -> Result<Self> {
        use crate::backends::loader::ModelLoader;
        use std::path::Path;
        let path = Path::new(path);
        if !path.exists() {
            return Err(crate::VocoderError::ModelError(format!(
                "Model file not found: {path}",
                path = path.display()
            )));
        }
        let mut loader = ModelLoader::new();
        match tokio::runtime::Runtime::new()
            .expect("tokio runtime creation should succeed")
            .block_on(loader.load_from_file(path))
        {
            Ok(model_info) => {
                let config = Self::config_from_model_info(&model_info)?;
                Ok(Self::with_config(config))
            }
            Err(e) => {
                tracing::warn!("Could not load model: {e}. Using default config.");
                Ok(Self::new())
            }
        }
    }
    /// Extract HiFiGAN configuration from model info
    #[cfg(feature = "candle")]
    fn config_from_model_info(
        model_info: &crate::backends::loader::ModelInfo,
    ) -> Result<HiFiGanConfig> {
        let model_name = model_info.metadata.name.to_lowercase();
        let config = if model_name.contains("v1") {
            HiFiGanVariants::v1()
        } else if model_name.contains("v2") {
            HiFiGanVariants::v2()
        } else if model_name.contains("v3") {
            HiFiGanVariants::v3()
        } else {
            HiFiGanVariants::v1()
        };
        Ok(config)
    }
    /// Load HiFi-GAN model from file (without Candle)
    #[cfg(not(feature = "candle"))]
    pub fn load_from_file(_path: &str) -> Result<Self> {
        Ok(Self::new())
    }
    /// Initialize the inference engine
    #[cfg(feature = "candle")]
    pub fn initialize_inference(&mut self, vb: candle_nn::VarBuilder) -> Result<()> {
        let generator = HiFiGanGenerator::new(self.config.clone(), vb)?;
        let inference = HiFiGanInference::new(generator, self.config.clone())?;
        self.inference = Some(inference);
        Ok(())
    }
    /// Initialize inference for testing (creates dummy weights)
    #[cfg(feature = "candle")]
    pub fn initialize_inference_for_testing(&mut self) -> Result<()> {
        use candle_core::Device;
        use candle_nn::VarMap;
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        self.initialize_inference(vb)
    }
    /// Initialize inference for testing (without Candle)
    #[cfg(not(feature = "candle"))]
    pub fn initialize_inference_for_testing(&mut self) -> Result<()> {
        Ok(())
    }
    /// Get configuration
    pub fn config(&self) -> &HiFiGanConfig {
        &self.config
    }
    /// Check if inference is initialized
    #[cfg(feature = "candle")]
    pub fn is_initialized(&self) -> bool {
        self.inference.is_some()
    }
    /// Check if inference is initialized (always false without Candle)
    #[cfg(not(feature = "candle"))]
    pub fn is_initialized(&self) -> bool {
        false
    }
    /// Get mutable reference to effect chain for configuration
    pub fn effect_chain_mut(&mut self) -> &mut EffectChain {
        &mut self.effect_chain
    }
    /// Get reference to effect chain
    pub fn effect_chain(&self) -> &EffectChain {
        &self.effect_chain
    }
    /// Set effect chain preset
    pub fn set_effect_preset(&mut self, preset_name: &str) {
        self.effect_chain = match preset_name {
            "speech" => EffectPresets::speech_enhancement(self.config.sample_rate),
            "warmth" => EffectPresets::voice_warmth(self.config.sample_rate),
            "broadcast" => EffectPresets::broadcast_quality(self.config.sample_rate),
            "minimal" => EffectPresets::minimal_enhancement(self.config.sample_rate),
            _ => EffectPresets::speech_enhancement(self.config.sample_rate),
        };
    }
    /// Enable or disable audio post-processing
    pub fn set_post_processing_enabled(&mut self, enabled: bool) {
        self.effect_chain.set_bypass(!enabled);
    }
    /// Apply basic post-processing without mutable state
    pub(super) fn apply_basic_post_processing(&self, audio: &mut AudioBuffer) {
        let samples = audio.samples_mut();
        self.apply_dc_removal(samples);
        self.apply_normalization(samples);
        self.apply_soft_limiting(samples);
    }
    fn apply_dc_removal(&self, samples: &mut [f32]) {
        if samples.len() < 2 {
            return;
        }
        let alpha = 0.995;
        let mut prev_input = samples[0];
        let mut prev_output = samples[0];
        #[allow(clippy::needless_range_loop)]
        for i in 1..samples.len() {
            let current_input = samples[i];
            let output = alpha * (prev_output + current_input - prev_input);
            samples[i] = output;
            prev_input = current_input;
            prev_output = output;
        }
    }
    fn apply_normalization(&self, samples: &mut [f32]) {
        let peak = samples.iter().map(|x| x.abs()).fold(0.0, f32::max);
        if peak > 1e-10 && peak < 0.95 {
            let target_peak = 0.7;
            let scale = target_peak / peak;
            for sample in samples {
                *sample *= scale;
            }
        }
    }
    fn apply_soft_limiting(&self, samples: &mut [f32]) {
        let threshold = 0.95;
        for sample in samples {
            if sample.abs() > threshold {
                let sign = if *sample > 0.0 { 1.0 } else { -1.0 };
                let normalized = sample.abs() / threshold;
                *sample = sign * threshold * normalized.tanh();
            }
        }
    }
    /// Set emotion for vocoding
    pub fn set_emotion(&mut self, emotion_config: EmotionConfig) {
        let emotion_params = EmotionVocodingParams::for_emotion(
            &emotion_config.emotion_type,
            emotion_config.intensity,
        );
        self.current_emotion = Some(emotion_config);
        self.emotion_params = Some(emotion_params);
    }
    /// Get current emotion configuration
    pub fn get_emotion(&self) -> Option<&EmotionConfig> {
        self.current_emotion.as_ref()
    }
    /// Clear emotion configuration (return to neutral)
    pub fn clear_emotion(&mut self) {
        self.current_emotion = None;
        self.emotion_params = None;
    }
    /// Set voice conversion configuration
    pub fn set_voice_conversion(&mut self, config: VoiceConversionConfig) {
        let converter = VoiceConverter::new(config.clone(), self.config.sample_rate);
        self.voice_converter = Some(converter);
        self.voice_conversion_config = Some(config);
    }
    /// Get current voice conversion configuration
    pub fn get_voice_conversion(&self) -> Option<&VoiceConversionConfig> {
        self.voice_conversion_config.as_ref()
    }
    /// Clear voice conversion configuration (disable voice conversion)
    pub fn clear_voice_conversion(&mut self) {
        self.voice_converter = None;
        self.voice_conversion_config = None;
    }
    /// Update voice conversion configuration
    pub fn update_voice_conversion(&mut self, config: VoiceConversionConfig) {
        if let Some(ref mut converter) = self.voice_converter {
            converter.update_config(config.clone());
            self.voice_conversion_config = Some(config);
        } else {
            self.set_voice_conversion(config);
        }
    }
    /// Set unified conditioning configuration
    pub fn set_unified_conditioning(&mut self, config: VocoderConditioningConfig) {
        let conditioner = VocoderConditioner::new(config, self.config.sample_rate);
        self.unified_conditioner = Some(conditioner);
    }
    /// Get current unified conditioning configuration
    pub fn get_unified_conditioning(&self) -> Option<&VocoderConditioningConfig> {
        self.unified_conditioner.as_ref().map(|c| c.config())
    }
    /// Clear unified conditioning configuration
    pub fn clear_unified_conditioning(&mut self) {
        self.unified_conditioner = None;
    }
    /// Update unified conditioning configuration
    pub fn update_unified_conditioning(&mut self, config: VocoderConditioningConfig) {
        if let Some(ref mut conditioner) = self.unified_conditioner {
            conditioner.update_config(config);
        } else {
            self.set_unified_conditioning(config);
        }
    }
    /// Apply emotion-specific processing to audio
    pub(super) fn apply_emotion_processing(&self, audio: &mut AudioBuffer) {
        if let Some(ref params) = self.emotion_params {
            if params.spectral_tilt.abs() > 0.001 {
                self.apply_spectral_tilt(audio, params.spectral_tilt);
            }
            if params.harmonic_boost.abs() > 0.001 {
                self.apply_harmonic_boost(audio, params.harmonic_boost);
            }
            if params.formant_shift.abs() > 0.001 {
                self.apply_formant_shift(audio, params.formant_shift);
            }
            if params.brightness.abs() > 0.001 {
                self.apply_brightness_adjustment(audio, params.brightness);
            }
            if params.breathiness.abs() > 0.001 {
                self.apply_breathiness(audio, params.breathiness);
            }
            if params.roughness.abs() > 0.001 {
                self.apply_roughness(audio, params.roughness);
            }
        }
    }
    /// Apply spectral tilt to audio
    fn apply_spectral_tilt(&self, audio: &mut AudioBuffer, tilt: f32) {
        let mut prev_sample = 0.0;
        for sample in audio.samples_mut() {
            let filtered = *sample + tilt * (*sample - prev_sample);
            prev_sample = *sample;
            *sample = filtered.clamp(-1.0, 1.0);
        }
    }
    /// Apply harmonic boost
    fn apply_harmonic_boost(&self, audio: &mut AudioBuffer, boost: f32) {
        for sample in audio.samples_mut() {
            let enhanced = *sample + boost * (*sample * *sample * *sample);
            *sample = enhanced.clamp(-1.0, 1.0);
        }
    }
    /// Apply formant shifting using spectral envelope modification
    fn apply_formant_shift(&self, audio: &mut AudioBuffer, shift: f32) {
        if shift.abs() < 0.01 {
            return;
        }
        use scirs2_core::Complex;
        let samples = audio.samples_mut();
        let frame_size = 1024;
        let hop_size = frame_size / 4;
        let _overlap = frame_size - hop_size;
        if samples.len() < frame_size {
            return;
        }
        let mut output_samples = vec![0.0f32; samples.len()];
        let mut window = vec![0.0f32; frame_size];
        for (i, w) in window.iter_mut().enumerate().take(frame_size) {
            *w = 0.5
                * (1.0 - ((2.0 * std::f32::consts::PI * i as f32) / (frame_size - 1) as f32).cos());
        }
        let shift_factor = 2.0f32.powf(shift);
        let mut pos = 0;
        while pos + frame_size <= samples.len() {
            let mut input_buffer = vec![0.0f32; frame_size];
            for i in 0..frame_size {
                input_buffer[i] = samples[pos + i] * window[i];
            }
            let spectrum = match scirs2_fft::rfft(&input_buffer, None) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let mut shifted_spectrum = vec![Complex::new(0.0, 0.0); spectrum.len()];
            for (i, &value) in spectrum.iter().enumerate() {
                let shifted_bin = ((i as f32) * shift_factor) as usize;
                if shifted_bin < shifted_spectrum.len() {
                    shifted_spectrum[shifted_bin] = value;
                }
            }
            for i in 1..shifted_spectrum.len() - 1 {
                if shifted_spectrum[i].norm() == 0.0 {
                    let prev = shifted_spectrum[i - 1];
                    let next = shifted_spectrum[i + 1];
                    shifted_spectrum[i] =
                        Complex::new((prev.re + next.re) * 0.5, (prev.im + next.im) * 0.5);
                }
            }
            let time_output = match scirs2_fft::irfft(&shifted_spectrum, Some(frame_size)) {
                Ok(t) => t,
                Err(_) => continue,
            };
            for i in 0..frame_size {
                if pos + i < output_samples.len() && i < time_output.len() {
                    output_samples[pos + i] +=
                        (time_output[i] as f32) * window[i] / frame_size as f32;
                }
            }
            pos += hop_size;
        }
        let gain_compensation = 1.0 / shift_factor.sqrt();
        for (original, processed) in samples.iter_mut().zip(output_samples.iter()) {
            *original = (processed * gain_compensation).clamp(-1.0, 1.0);
        }
    }
    /// Apply brightness adjustment
    fn apply_brightness_adjustment(&self, audio: &mut AudioBuffer, brightness: f32) {
        let alpha = (brightness * 0.5).clamp(-0.5, 0.5);
        let mut prev_sample = 0.0;
        for sample in audio.samples_mut() {
            let filtered = *sample + alpha * (*sample - prev_sample);
            prev_sample = *sample;
            *sample = filtered.clamp(-1.0, 1.0);
        }
    }
    /// Apply breathiness effect
    fn apply_breathiness(&self, audio: &mut AudioBuffer, breathiness: f32) {
        let noise_level = breathiness.abs() * 0.05;
        for (i, sample) in audio.samples_mut().iter_mut().enumerate() {
            let t = i as f32 * 0.001;
            let noise =
                (t * 1234.0).sin() * 0.3 + (t * 2345.0).sin() * 0.2 + (t * 3456.0).sin() * 0.1;
            let scaled_noise = noise * noise_level;
            *sample = (*sample + scaled_noise).clamp(-1.0, 1.0);
        }
    }
    /// Apply roughness effect
    fn apply_roughness(&self, audio: &mut AudioBuffer, roughness: f32) {
        let distortion = roughness.abs() * 0.3;
        for sample in audio.samples_mut() {
            if sample.abs() > 0.1 {
                let sign = sample.signum();
                let distorted = sample.abs().powf(1.0 - distortion);
                *sample = (sign * distorted).clamp(-1.0, 1.0);
            }
        }
    }
    /// Apply voice conversion processing to audio
    pub(super) fn apply_voice_conversion_processing(&self, audio: &mut AudioBuffer) -> Result<()> {
        if let Some(ref config) = self.voice_conversion_config {
            self.apply_simplified_voice_conversion(audio, config)?;
        }
        Ok(())
    }
    /// Simplified voice conversion implementation
    fn apply_simplified_voice_conversion(
        &self,
        audio: &mut AudioBuffer,
        config: &VoiceConversionConfig,
    ) -> Result<()> {
        let samples = audio.samples().to_vec();
        let mut processed_samples = samples;
        if config.pitch_shift.abs() > 0.01 {
            self.apply_simple_pitch_shift(
                &mut processed_samples,
                config.pitch_shift,
                config.conversion_strength,
            );
        }
        if config.age_shift.abs() > 0.01 || config.gender_shift.abs() > 0.01 {
            self.apply_spectral_voice_modifications(&mut processed_samples, config);
        }
        if config.breathiness > 0.01 {
            self.apply_voice_breathiness(
                &mut processed_samples,
                config.breathiness,
                config.conversion_strength,
            );
        }
        if config.roughness > 0.01 {
            self.apply_voice_roughness(
                &mut processed_samples,
                config.roughness,
                config.conversion_strength,
            );
        }
        if config.brightness.abs() > 0.01 {
            self.apply_voice_brightness(
                &mut processed_samples,
                config.brightness,
                config.conversion_strength,
            );
        }
        if config.warmth.abs() > 0.01 {
            self.apply_voice_warmth(
                &mut processed_samples,
                config.warmth,
                config.conversion_strength,
            );
        }
        *audio = AudioBuffer::new(processed_samples, audio.sample_rate(), audio.channels());
        Ok(())
    }
    /// Simple pitch shifting implementation
    fn apply_simple_pitch_shift(&self, samples: &mut [f32], pitch_shift: f32, intensity: f32) {
        let pitch_ratio = 2.0_f32.powf(pitch_shift / 12.0);
        let temp_samples = samples.to_vec();
        for (i, sample) in samples.iter_mut().enumerate() {
            let source_pos = (i as f32) * pitch_ratio;
            let source_idx = source_pos as usize;
            let source_frac = source_pos - source_idx as f32;
            if source_idx + 1 < temp_samples.len() {
                let interpolated = temp_samples[source_idx] * (1.0 - source_frac)
                    + temp_samples[source_idx + 1] * source_frac;
                *sample = sample.mul_add(1.0 - intensity, interpolated * intensity);
            }
        }
    }
    /// Apply spectral modifications for age and gender transformation
    fn apply_spectral_voice_modifications(
        &self,
        samples: &mut [f32],
        config: &VoiceConversionConfig,
    ) {
        let intensity = config.conversion_strength;
        if config.age_shift.abs() > 0.01 {
            let spectral_tilt = -config.age_shift * 0.15;
            for i in 1..samples.len() {
                let high_freq_emphasis = samples[i] - samples[i - 1];
                samples[i] =
                    (samples[i] + spectral_tilt * high_freq_emphasis * intensity).clamp(-1.0, 1.0);
            }
        }
        if config.gender_shift.abs() > 0.01 {
            let formant_emphasis = config.gender_shift * 0.1;
            for i in 2..samples.len() {
                let formant_diff = samples[i] - 0.5 * (samples[i - 1] + samples[i - 2]);
                samples[i] =
                    (samples[i] + formant_emphasis * formant_diff * intensity).clamp(-1.0, 1.0);
            }
        }
    }
    /// Apply breathiness effect for voice conversion
    fn apply_voice_breathiness(&self, samples: &mut [f32], breathiness: f32, intensity: f32) {
        let noise_level = breathiness * intensity * 0.02;
        for (i, sample) in samples.iter_mut().enumerate() {
            let t = i as f32 * 0.001;
            let noise = (t * 1847.0).sin() * 0.3 + (t * 3271.0).sin() * 0.2;
            let breath_noise = noise * noise_level * sample.abs().sqrt();
            *sample = (*sample + breath_noise).clamp(-1.0, 1.0);
        }
    }
    /// Apply roughness effect for voice conversion
    fn apply_voice_roughness(&self, samples: &mut [f32], roughness: f32, intensity: f32) {
        let distortion_amount = roughness * intensity;
        for sample in samples.iter_mut() {
            if sample.abs() > 0.1 {
                let sign = sample.signum();
                let distortion_factor = 1.0 - distortion_amount * 0.2;
                let distorted = sample.abs().powf(distortion_factor);
                *sample = (sign * distorted).clamp(-1.0, 1.0);
            }
        }
    }
    /// Apply brightness adjustment for voice conversion
    fn apply_voice_brightness(&self, samples: &mut [f32], brightness: f32, intensity: f32) {
        let brightness_factor = brightness * intensity;
        for i in 1..samples.len() {
            let high_freq = samples[i] - samples[i - 1];
            samples[i] = (samples[i] + brightness_factor * 0.15 * high_freq).clamp(-1.0, 1.0);
        }
    }
    /// Apply warmth adjustment for voice conversion
    fn apply_voice_warmth(&self, samples: &mut [f32], warmth: f32, intensity: f32) {
        let warmth_factor = warmth * intensity;
        let alpha = (1.0 - warmth_factor * 0.2).clamp(0.1, 0.9);
        for i in 1..samples.len() {
            samples[i] = alpha * samples[i] + (1.0 - alpha) * samples[i - 1];
        }
    }
    /// Apply unified conditioning processing to audio
    /// This is a simplified version that works with &self
    pub(super) fn apply_unified_conditioning_processing(
        &self,
        audio: &mut AudioBuffer,
    ) -> Result<()> {
        if let Some(ref conditioner) = self.unified_conditioner {
            let config = conditioner.config();
            let samples = audio.samples_mut();
            let global_strength = config.global_strength;
            if let Some(ref emotion_config) = config.emotion {
                self.apply_emotion_adjustments(samples, emotion_config, global_strength)?;
            }
            if let Some(ref speaker_config) = config.speaker {
                self.apply_speaker_adjustments(samples, speaker_config, global_strength)?;
            }
            if let Some(ref prosody_config) = config.prosody {
                self.apply_prosody_adjustments(samples, prosody_config, global_strength)?;
            }
            if let Some(ref enhancement_config) = config.enhancement {
                self.apply_enhancement_adjustments(samples, enhancement_config, global_strength)?;
            }
            tracing::debug!(
                "Applied unified conditioning with {} features active",
                [
                    config.emotion.is_some(),
                    config.speaker.is_some(),
                    config.prosody.is_some(),
                    config.enhancement.is_some()
                ]
                .iter()
                .filter(|&&x| x)
                .count()
            );
        }
        Ok(())
    }
    /// Apply emotion-based adjustments to audio samples (stateless)
    fn apply_emotion_adjustments(
        &self,
        samples: &mut [f32],
        emotion_config: &EmotionConfig,
        global_strength: f32,
    ) -> Result<()> {
        let intensity = emotion_config.intensity.clamp(0.0, 1.0) * global_strength;
        match emotion_config.emotion_type.to_lowercase().as_str() {
            "happy" | "joy" | "excited" => {
                for sample in samples.iter_mut() {
                    *sample *= 1.0 + (intensity * 0.1);
                    *sample = sample.clamp(-1.0, 1.0);
                }
            }
            "sad" | "melancholy" | "depressed" => {
                for sample in samples.iter_mut() {
                    *sample *= 1.0 - (intensity * 0.15);
                    *sample *= 0.95 + (intensity * 0.05);
                }
            }
            "angry" | "rage" | "frustrated" => {
                for sample in samples.iter_mut() {
                    *sample *= 1.0 + (intensity * 0.2);
                    *sample = (*sample).tanh();
                }
            }
            "calm" | "peaceful" | "relaxed" => {
                for sample in samples.iter_mut() {
                    *sample *= 0.98 + (intensity * 0.02);
                }
            }
            _ => {
                if let Some(ref emotion_vector) = emotion_config.emotion_vector {
                    let modulation =
                        emotion_vector.iter().sum::<f32>() / emotion_vector.len() as f32;
                    for sample in samples.iter_mut() {
                        *sample *= 1.0 + (modulation * intensity * 0.1);
                        *sample = sample.clamp(-1.0, 1.0);
                    }
                }
            }
        }
        Ok(())
    }
    /// Apply speaker-based adjustments to audio samples (stateless)
    fn apply_speaker_adjustments(
        &self,
        samples: &mut [f32],
        speaker_config: &crate::conditioning::SpeakerConfig,
        global_strength: f32,
    ) -> Result<()> {
        let characteristics = &speaker_config.voice_characteristics;
        let intensity = global_strength;
        if characteristics.f0_adjustment.abs() > 0.01 {
            let pitch_factor = 1.0 + (characteristics.f0_adjustment * intensity * 0.1);
            for sample in samples.iter_mut() {
                *sample *= pitch_factor;
                *sample = sample.clamp(-1.0, 1.0);
            }
        }
        let quality = &characteristics.quality;
        if quality.breathiness > 0.01 {
            for (i, sample) in samples.iter_mut().enumerate() {
                let noise_val = (((i * 12345) % 65536) as f32 / 32768.0 - 1.0) * 0.5;
                let noise = noise_val * quality.breathiness * intensity * 0.02;
                *sample += noise;
                *sample = sample.clamp(-1.0, 1.0);
            }
        }
        let tenseness_factor = 1.0 + (quality.tenseness - 0.5) * intensity * 0.1;
        for sample in samples.iter_mut() {
            *sample *= tenseness_factor;
            *sample = sample.clamp(-1.0, 1.0);
        }
        Ok(())
    }
    /// Apply prosodic adjustments to audio samples (stateless)
    fn apply_prosody_adjustments(
        &self,
        samples: &mut [f32],
        prosody_config: &crate::conditioning::ProsodyConfig,
        global_strength: f32,
    ) -> Result<()> {
        let intensity = global_strength;
        if (prosody_config.speaking_rate - 1.0).abs() > 0.01 {
            let rate_factor = prosody_config.speaking_rate * intensity;
            for sample in samples.iter_mut() {
                *sample *= rate_factor;
                *sample = sample.clamp(-1.0, 1.0);
            }
        }
        if (prosody_config.pitch_range - 1.0).abs() > 0.01 {
            let pitch_factor = prosody_config.pitch_range * intensity;
            for sample in samples.iter_mut() {
                *sample *= pitch_factor;
                *sample = sample.clamp(-1.0, 1.0);
            }
        }
        Ok(())
    }
    /// Apply enhancement adjustments to audio samples (stateless)
    fn apply_enhancement_adjustments(
        &self,
        samples: &mut [f32],
        enhancement_config: &crate::conditioning::EnhancementConfig,
        global_strength: f32,
    ) -> Result<()> {
        let intensity = global_strength;
        if enhancement_config.noise_reduction > 0.01 {
            let threshold = 0.01;
            let reduction_factor = 1.0 - (enhancement_config.noise_reduction * intensity * 0.5);
            for sample in samples.iter_mut() {
                if sample.abs() < threshold {
                    *sample *= reduction_factor;
                }
            }
        }
        if enhancement_config.compression > 0.01 {
            let threshold = 0.7;
            let ratio = 2.0 + enhancement_config.compression * 2.0;
            for sample in samples.iter_mut() {
                if sample.abs() > threshold {
                    let excess = sample.abs() - threshold;
                    let compressed_excess = excess / ratio;
                    *sample = (*sample).signum() * (threshold + compressed_excess);
                }
            }
        }
        Ok(())
    }
    /// Synthesize audio from mel spectrogram using basic vocoder fallback
    /// This is used when Candle feature is not available
    #[cfg(not(feature = "candle"))]
    pub(super) fn synthesize_from_mel_fallback(
        &self,
        mel: &MelSpectrogram,
        config: Option<&SynthesisConfig>,
    ) -> Result<AudioBuffer> {
        let sample_rate = self.config.sample_rate;
        let hop_length = mel.hop_length;
        let num_samples = (mel.n_frames as u32 * hop_length) as usize;
        let mut samples = vec![0.0f32; num_samples];
        for frame_idx in 0..mel.n_frames {
            let frame_start_sample = frame_idx * hop_length as usize;
            let mut frame_energy = 0.0f32;
            let mut dominant_frequency = 0.0f32;
            let mut spectral_centroid = 0.0f32;
            let mut total_weight = 0.0f32;
            for (mel_bin, mel_channel) in mel.data.iter().enumerate() {
                if frame_idx < mel_channel.len() {
                    let mel_value = mel_channel[frame_idx];
                    frame_energy += mel_value;
                    let freq = self.mel_bin_to_frequency(mel_bin, mel.n_mels);
                    spectral_centroid += freq * mel_value;
                    total_weight += mel_value;
                    if mel_value > dominant_frequency {
                        dominant_frequency = freq;
                    }
                }
            }
            if total_weight > 0.0 {
                spectral_centroid /= total_weight;
            } else {
                spectral_centroid = 200.0;
            }
            spectral_centroid = spectral_centroid.clamp(80.0, 1000.0);
            frame_energy = frame_energy.clamp(0.0, 10.0);
            self.synthesize_frame_additive(
                &mut samples,
                frame_start_sample,
                hop_length as usize,
                spectral_centroid,
                frame_energy,
                sample_rate,
            );
        }
        if let Some(config) = config {
            self.apply_synthesis_config(&mut samples, config, sample_rate);
        }
        let mut audio = AudioBuffer::from_samples(samples, sample_rate);
        self.normalize_audio(&mut audio);
        Ok(audio)
    }
    /// Convert mel bin index to approximate frequency
    #[cfg(not(feature = "candle"))]
    fn mel_bin_to_frequency(&self, mel_bin: usize, n_mels: usize) -> f32 {
        let mel_max = 2595.0 * (1.0 + 8000.0 / 700.0).log10();
        let mel_value = (mel_bin as f32 / n_mels as f32) * mel_max;
        700.0 * ((mel_value / 2595.0).exp() - 1.0)
    }
    /// Synthesize audio frame using additive synthesis
    #[cfg(not(feature = "candle"))]
    fn synthesize_frame_additive(
        &self,
        samples: &mut [f32],
        start_sample: usize,
        frame_length: usize,
        fundamental_freq: f32,
        energy: f32,
        sample_rate: u32,
    ) {
        let end_sample = (start_sample + frame_length).min(samples.len());
        let normalized_energy = (energy * 0.1).clamp(0.0, 0.5);
        let harmonics = [
            (1.0, 1.0),
            (2.0, 0.5),
            (3.0, 0.25),
            (4.0, 0.125),
            (5.0, 0.0625),
        ];
        for sample_idx in start_sample..end_sample {
            let relative_idx = sample_idx - start_sample;
            let t = relative_idx as f32 / sample_rate as f32;
            let window = if frame_length > 1 {
                0.5 * (1.0
                    - ((2.0 * std::f32::consts::PI * relative_idx as f32)
                        / (frame_length - 1) as f32)
                        .cos())
            } else {
                1.0
            };
            let mut harmonic_sum = 0.0f32;
            for (harmonic_ratio, amplitude) in &harmonics {
                let freq = fundamental_freq * harmonic_ratio;
                if freq < sample_rate as f32 / 2.0 {
                    let phase = 2.0 * std::f32::consts::PI * freq * t;
                    harmonic_sum += amplitude * phase.sin();
                }
            }
            let noise_val = (((sample_idx * 54321) % 65536) as f32 / 32768.0 - 1.0) * 0.5;
            let noise = noise_val * 0.05 * normalized_energy;
            let sample_value = harmonic_sum * normalized_energy * window + noise;
            samples[sample_idx] += sample_value;
        }
    }
    /// Apply synthesis configuration to audio samples
    #[cfg(not(feature = "candle"))]
    fn apply_synthesis_config(
        &self,
        samples: &mut [f32],
        config: &SynthesisConfig,
        sample_rate: u32,
    ) {
        if (config.energy - 1.0).abs() > 0.01 {
            for sample in samples.iter_mut() {
                *sample *= config.energy;
            }
        }
        if config.pitch_shift.abs() > 0.01 {
            let pitch_factor = 2.0_f32.powf(config.pitch_shift / 12.0);
            for sample in samples.iter_mut() {
                *sample *= pitch_factor.clamp(0.5, 2.0);
            }
        }
        if (config.speed - 1.0).abs() > 0.01 {
            let speed_factor = config.speed.clamp(0.5, 2.0);
            for sample in samples.iter_mut() {
                *sample *= speed_factor;
            }
        }
    }
    /// Normalize audio to prevent clipping and ensure reasonable levels
    #[cfg(not(feature = "candle"))]
    fn normalize_audio(&self, audio: &mut AudioBuffer) {
        let samples = audio.samples_mut();
        let peak = samples
            .iter()
            .map(|&x| x.abs())
            .fold(0.0f32, |acc, x| acc.max(x));
        if peak > 0.8 {
            let scale = 0.8 / peak;
            for sample in samples {
                *sample *= scale;
            }
        }
        for sample in samples {
            *sample = sample.clamp(-1.0, 1.0);
        }
    }
}
/// Streaming vocoder for chunk-based processing with overlap-add windowing
pub struct StreamingVocoder {
    vocoder: HiFiGanVocoder,
    config: Option<SynthesisConfig>,
    mel_buffer: std::collections::VecDeque<Vec<f32>>,
    mel_buffer_size: usize,
    audio_buffer: std::collections::VecDeque<f32>,
    overlap_buffer: Vec<f32>,
    chunk_size: usize,
    overlap_size: usize,
    hop_length: usize,
    sample_rate: u32,
    processed_frames: usize,
    total_latency_samples: usize,
    lookahead_frames: usize,
    processing_times: std::collections::VecDeque<std::time::Duration>,
    max_metrics_history: usize,
}
impl StreamingVocoder {
    pub fn new(vocoder: HiFiGanVocoder, config: Option<SynthesisConfig>) -> Result<Self> {
        let (chunk_size, overlap_size, lookahead_frames) = match vocoder.config().variant {
            HiFiGanVariant::V1 => (256, 64, 128),
            HiFiGanVariant::V2 => (192, 48, 96),
            HiFiGanVariant::V3 => (128, 32, 64),
        };
        let hop_length = 256;
        let sample_rate = vocoder.config().sample_rate;
        let mel_buffer_size = lookahead_frames + chunk_size + overlap_size;
        let processing_latency = chunk_size * hop_length;
        let buffering_latency = lookahead_frames * hop_length;
        let total_latency_samples = processing_latency + buffering_latency;
        Ok(Self {
            vocoder,
            config,
            mel_buffer: std::collections::VecDeque::new(),
            mel_buffer_size,
            audio_buffer: std::collections::VecDeque::new(),
            overlap_buffer: Vec::new(),
            chunk_size,
            overlap_size,
            hop_length,
            sample_rate,
            processed_frames: 0,
            total_latency_samples,
            lookahead_frames,
            processing_times: std::collections::VecDeque::new(),
            max_metrics_history: 100,
        })
    }
    /// Add mel frames to the buffer and process when enough data is available
    pub async fn process_chunk(&mut self, mel: &MelSpectrogram) -> Result<Option<AudioBuffer>> {
        let start_time = std::time::Instant::now();
        let mel_frames = self.extract_mel_frames(mel)?;
        for frame in mel_frames {
            self.mel_buffer.push_back(frame);
            while self.mel_buffer.len() > self.mel_buffer_size {
                self.mel_buffer.pop_front();
            }
        }
        if self.mel_buffer.len() >= self.lookahead_frames + self.chunk_size {
            let audio_chunk = self.process_buffered_chunk().await?;
            let processing_time = start_time.elapsed();
            self.processing_times.push_back(processing_time);
            while self.processing_times.len() > self.max_metrics_history {
                self.processing_times.pop_front();
            }
            Ok(Some(audio_chunk))
        } else {
            Ok(None)
        }
    }
    /// Flush remaining buffered audio
    pub async fn flush(&mut self) -> Result<Option<AudioBuffer>> {
        if self.mel_buffer.len() >= self.chunk_size {
            let audio_chunk = self.process_buffered_chunk().await?;
            Ok(Some(audio_chunk))
        } else if !self.audio_buffer.is_empty() {
            let samples: Vec<f32> = self.audio_buffer.drain(..).collect();
            let audio = AudioBuffer::new(samples, self.sample_rate, 1);
            Ok(Some(audio))
        } else {
            Ok(None)
        }
    }
    /// Extract mel frames from MelSpectrogram for buffering
    fn extract_mel_frames(&self, mel: &MelSpectrogram) -> Result<Vec<Vec<f32>>> {
        let num_frames = mel.n_frames;
        let mel_channels = mel.n_mels;
        if mel.data.len() != mel_channels {
            return Err(VocoderError::InvalidMelSpectrogram(format!(
                "Data length {} doesn't match n_mels {}",
                mel.data.len(),
                mel_channels
            )));
        }
        let mut frames = Vec::with_capacity(num_frames);
        for frame_idx in 0..num_frames {
            let mut frame = Vec::with_capacity(mel_channels);
            for mel_channel in &mel.data {
                if frame_idx < mel_channel.len() {
                    frame.push(mel_channel[frame_idx]);
                } else {
                    frame.push(0.0);
                }
            }
            frames.push(frame);
        }
        Ok(frames)
    }
    /// Process a chunk of buffered mel frames
    async fn process_buffered_chunk(&mut self) -> Result<AudioBuffer> {
        let start_idx = self.lookahead_frames.min(self.mel_buffer.len());
        let end_idx = (start_idx + self.chunk_size).min(self.mel_buffer.len());
        if start_idx >= end_idx {
            return Err(crate::VocoderError::VocodingError(
                "Insufficient frames for processing".to_string(),
            ));
        }
        let chunk_frames = self
            .mel_buffer
            .range(start_idx..end_idx)
            .cloned()
            .collect::<Vec<_>>();
        let chunk_mel = self.create_mel_from_frames(&chunk_frames)?;
        let mut audio = self
            .vocoder
            .vocode(&chunk_mel, self.config.as_ref())
            .await?;
        if !self.overlap_buffer.is_empty() {
            let samples = audio.samples_mut();
            self.apply_overlap_add(samples);
        }
        self.update_overlap_buffer(&audio);
        self.processed_frames += end_idx - start_idx;
        let frames_to_remove =
            (end_idx - start_idx).min(self.mel_buffer.len().saturating_sub(self.lookahead_frames));
        for _ in 0..frames_to_remove {
            self.mel_buffer.pop_front();
        }
        Ok(audio)
    }
    /// Create MelSpectrogram from buffered frames
    fn create_mel_from_frames(&self, frames: &[Vec<f32>]) -> Result<MelSpectrogram> {
        if frames.is_empty() {
            return Err(crate::VocoderError::VocodingError(
                "Cannot create mel from empty frames".to_string(),
            ));
        }
        let num_frames = frames.len();
        let num_channels = frames[0].len();
        let mut data = vec![vec![0.0; num_frames]; num_channels];
        for (frame_idx, frame) in frames.iter().enumerate() {
            for (channel_idx, &value) in frame.iter().enumerate() {
                data[channel_idx][frame_idx] = value;
            }
        }
        Ok(MelSpectrogram::new(
            data,
            self.sample_rate,
            self.hop_length as u32,
        ))
    }
    /// Apply overlap-add windowing with the previous chunk
    fn apply_overlap_add(&mut self, current_samples: &mut [f32]) {
        let overlap_len = self.overlap_buffer.len().min(current_samples.len());
        #[allow(clippy::needless_range_loop)]
        for i in 0..overlap_len {
            let fade_factor = i as f32 / overlap_len as f32;
            current_samples[i] =
                current_samples[i] * fade_factor + self.overlap_buffer[i] * (1.0 - fade_factor);
        }
    }
    /// Update overlap buffer for next chunk
    fn update_overlap_buffer(&mut self, audio: &AudioBuffer) {
        let samples = audio.samples();
        let overlap_samples = self.overlap_size * self.hop_length;
        if samples.len() >= overlap_samples {
            let start_idx = samples.len() - overlap_samples;
            self.overlap_buffer = samples[start_idx..].to_vec();
        } else {
            self.overlap_buffer = samples.to_vec();
        }
    }
    #[allow(dead_code)]
    fn apply_windowing(&self, audio: &AudioBuffer) -> Result<AudioBuffer> {
        let samples = audio.samples().to_vec();
        let windowed_samples = self.apply_hann_window(&samples);
        Ok(AudioBuffer::new(
            windowed_samples,
            audio.sample_rate(),
            audio.channels(),
        ))
    }
    #[allow(dead_code)]
    fn apply_hann_window(&self, samples: &[f32]) -> Vec<f32> {
        let len = samples.len();
        let mut windowed = Vec::with_capacity(len);
        for (i, &sample) in samples.iter().enumerate() {
            let window_val =
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (len - 1) as f32).cos());
            windowed.push(sample * window_val);
        }
        windowed
    }
    /// Add overlap-add processing for smooth transitions
    #[allow(dead_code)]
    fn overlap_add(&self, new_audio: &[f32], overlap_samples: &[f32]) -> Vec<f32> {
        let mut result = new_audio.to_vec();
        let overlap_len = overlap_samples.len().min(result.len());
        for i in 0..overlap_len {
            let fade_factor = i as f32 / overlap_len as f32;
            result[i] = new_audio[i] * fade_factor + overlap_samples[i] * (1.0 - fade_factor);
        }
        result
    }
    /// Get streaming performance metrics
    pub fn get_performance_metrics(&self) -> StreamingMetrics {
        let avg_processing_time = if self.processing_times.is_empty() {
            std::time::Duration::from_secs(0)
        } else {
            let total: std::time::Duration = self.processing_times.iter().sum();
            total / self.processing_times.len() as u32
        };
        let max_processing_time = self
            .processing_times
            .iter()
            .max()
            .copied()
            .unwrap_or_else(|| std::time::Duration::from_secs(0));
        StreamingMetrics {
            avg_processing_time_ms: avg_processing_time.as_millis() as f32,
            max_processing_time_ms: max_processing_time.as_millis() as f32,
            total_latency_ms: (self.total_latency_samples as f32 / self.sample_rate as f32)
                * 1000.0,
            processed_frames: self.processed_frames,
            buffer_utilization: (self.mel_buffer.len() as f32 / self.mel_buffer_size as f32)
                * 100.0,
            chunk_size: self.chunk_size,
            overlap_size: self.overlap_size,
        }
    }
    /// Configure adaptive streaming parameters based on performance
    pub fn configure_for_performance(&mut self, target_latency_ms: f32) -> Result<()> {
        let target_latency_samples =
            (target_latency_ms / 1000.0 * self.sample_rate as f32) as usize;
        let max_chunk_latency = target_latency_samples / 2;
        let max_chunk_frames = max_chunk_latency / self.hop_length;
        if max_chunk_frames > 32 {
            self.chunk_size = max_chunk_frames.min(512);
            self.overlap_size = self.chunk_size / 4;
            self.lookahead_frames = self.chunk_size / 2;
            self.mel_buffer_size = self.lookahead_frames + self.chunk_size + self.overlap_size;
            self.total_latency_samples =
                self.chunk_size * self.hop_length + self.lookahead_frames * self.hop_length;
            Ok(())
        } else {
            Err(crate::VocoderError::ConfigError(
                "Target latency too low for stable streaming".to_string(),
            ))
        }
    }
    /// Reset the streaming state
    pub fn reset(&mut self) {
        self.mel_buffer.clear();
        self.audio_buffer.clear();
        self.overlap_buffer.clear();
        self.processed_frames = 0;
        self.processing_times.clear();
    }
    /// Get estimated latency in milliseconds
    pub fn get_latency_ms(&self) -> f32 {
        (self.total_latency_samples as f32 / self.sample_rate as f32) * 1000.0
    }
    /// Check if the vocoder can process in real-time
    pub fn is_realtime_capable(&self) -> bool {
        let avg_metrics = self.get_performance_metrics();
        let processing_budget_ms = (self.chunk_size as f32 / self.sample_rate as f32) * 1000.0;
        avg_metrics.avg_processing_time_ms < processing_budget_ms * 0.8
    }
}
