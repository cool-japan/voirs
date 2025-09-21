//! Configuration for singing synthesis

use crate::types::{Expression, QualitySettings, VoiceCharacteristics, VoiceType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Main configuration for singing synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingingConfig {
    /// Enable singing synthesis
    pub enabled: bool,
    /// Default voice characteristics
    pub default_voice: VoiceCharacteristics,
    /// Default expression
    pub default_expression: Expression,
    /// Quality settings
    pub quality: QualitySettings,
    /// Audio settings
    pub audio: AudioSettings,
    /// Model settings
    pub model: ModelSettings,
    /// Effect settings
    pub effects: EffectSettings,
    /// Performance settings
    pub performance: PerformanceSettings,
    /// Voice bank directory
    pub voice_bank_dir: Option<PathBuf>,
    /// Model cache directory
    pub model_cache_dir: Option<PathBuf>,
    /// Custom presets
    pub custom_presets: HashMap<String, PresetConfig>,
}

/// Audio settings for singing synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSettings {
    /// Sample rate (Hz)
    pub sample_rate: u32,
    /// Channels (1 for mono, 2 for stereo)
    pub channels: u8,
    /// Bit depth
    pub bit_depth: u16,
    /// Buffer size for real-time processing
    pub buffer_size: usize,
    /// Windowing function for spectral processing
    pub window_function: WindowFunction,
    /// Overlap ratio for spectral processing
    pub overlap_ratio: f32,
    /// Enable real-time processing
    pub real_time: bool,
}

/// Model settings for singing synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSettings {
    /// Primary synthesis model
    pub synthesis_model: SynthesisModel,
    /// Vocoder model
    pub vocoder_model: VocoderModel,
    /// Pitch model
    pub pitch_model: PitchModel,
    /// Duration model
    pub duration_model: DurationModel,
    /// Model precision
    pub precision: ModelPrecision,
    /// Enable model caching
    pub cache_models: bool,
    /// Maximum cache size in MB
    pub max_cache_size: usize,
}

/// Effect settings for singing synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSettings {
    /// Enable effects processing
    pub enabled: bool,
    /// Default effect chain
    pub default_chain: Vec<String>,
    /// Effect parameters
    pub parameters: HashMap<String, EffectParams>,
    /// Enable parallel processing
    pub parallel_processing: bool,
    /// Maximum effect chain length
    pub max_chain_length: usize,
}

/// Performance settings for singing synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSettings {
    /// Number of worker threads
    pub num_threads: Option<usize>,
    /// Enable GPU acceleration
    pub gpu_acceleration: bool,
    /// GPU device ID
    pub gpu_device: Option<u32>,
    /// Memory limit in MB
    pub memory_limit: Option<usize>,
    /// Enable performance monitoring
    pub monitoring: bool,
    /// Timeout for synthesis in seconds
    pub timeout: Option<u64>,
}

/// Windowing functions for spectral processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WindowFunction {
    /// Hamming window
    Hamming,
    /// Hanning window
    Hanning,
    /// Blackman window
    Blackman,
    /// Kaiser window
    Kaiser,
    /// Gaussian window
    Gaussian,
    /// Rectangular window
    Rectangular,
}

/// Synthesis model types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SynthesisModel {
    /// Neural singing synthesis
    Neural,
    /// Concatenative synthesis
    Concatenative,
    /// Parametric synthesis
    Parametric,
    /// Hybrid synthesis
    Hybrid,
}

/// Vocoder model types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VocoderModel {
    /// HiFi-GAN vocoder
    HiFiGAN,
    /// WaveGlow vocoder
    WaveGlow,
    /// WaveNet vocoder
    WaveNet,
    /// MelGAN vocoder
    MelGAN,
    /// Neural vocoder
    Neural,
}

/// Pitch model types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PitchModel {
    /// Rule-based pitch
    RuleBased,
    /// Statistical pitch
    Statistical,
    /// Neural pitch
    Neural,
    /// Hybrid pitch
    Hybrid,
}

/// Duration model types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DurationModel {
    /// Rule-based duration
    RuleBased,
    /// Statistical duration
    Statistical,
    /// Neural duration
    Neural,
    /// Hybrid duration
    Hybrid,
}

/// Model precision levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelPrecision {
    /// 16-bit precision
    Float16,
    /// 32-bit precision
    Float32,
    /// 64-bit precision
    Float64,
    /// Mixed precision
    Mixed,
}

/// Effect parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectParams {
    /// Effect name
    pub name: String,
    /// Parameter values
    pub parameters: HashMap<String, f32>,
    /// Enable/disable effect
    pub enabled: bool,
    /// Processing order
    pub order: u32,
}

/// Preset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetConfig {
    /// Preset name
    pub name: String,
    /// Preset description
    pub description: String,
    /// Voice characteristics
    pub voice: VoiceCharacteristics,
    /// Singing technique
    pub technique: crate::techniques::SingingTechnique,
    /// Effect chain
    pub effects: Vec<String>,
    /// Quality settings
    pub quality: QualitySettings,
    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Builder for singing configuration
#[derive(Debug, Clone)]
pub struct SingingConfigBuilder {
    config: SingingConfig,
}

impl SingingConfigBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: SingingConfig::default(),
        }
    }

    /// Enable or disable singing synthesis
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Set default voice characteristics
    pub fn default_voice(mut self, voice: VoiceCharacteristics) -> Self {
        self.config.default_voice = voice;
        self
    }

    /// Set default expression
    pub fn default_expression(mut self, expression: Expression) -> Self {
        self.config.default_expression = expression;
        self
    }

    /// Set quality settings
    pub fn quality(mut self, quality: QualitySettings) -> Self {
        self.config.quality = quality;
        self
    }

    /// Set audio settings
    pub fn audio(mut self, audio: AudioSettings) -> Self {
        self.config.audio = audio;
        self
    }

    /// Set model settings
    pub fn model(mut self, model: ModelSettings) -> Self {
        self.config.model = model;
        self
    }

    /// Set effect settings
    pub fn effects(mut self, effects: EffectSettings) -> Self {
        self.config.effects = effects;
        self
    }

    /// Set performance settings
    pub fn performance(mut self, performance: PerformanceSettings) -> Self {
        self.config.performance = performance;
        self
    }

    /// Set voice bank directory
    pub fn voice_bank_dir(mut self, path: PathBuf) -> Self {
        self.config.voice_bank_dir = Some(path);
        self
    }

    /// Set model cache directory
    pub fn model_cache_dir(mut self, path: PathBuf) -> Self {
        self.config.model_cache_dir = Some(path);
        self
    }

    /// Add custom preset
    pub fn add_preset(mut self, name: String, preset: PresetConfig) -> Self {
        self.config.custom_presets.insert(name, preset);
        self
    }

    /// Build the configuration
    pub fn build(self) -> SingingConfig {
        self.config
    }
}

impl Default for SingingConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SingingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_voice: VoiceCharacteristics::default(),
            default_expression: Expression::Neutral,
            quality: QualitySettings::default(),
            audio: AudioSettings::default(),
            model: ModelSettings::default(),
            effects: EffectSettings::default(),
            performance: PerformanceSettings::default(),
            voice_bank_dir: None,
            model_cache_dir: None,
            custom_presets: HashMap::new(),
        }
    }
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            channels: 1,
            bit_depth: 16,
            buffer_size: 1024,
            window_function: WindowFunction::Hamming,
            overlap_ratio: 0.5,
            real_time: false,
        }
    }
}

impl Default for ModelSettings {
    fn default() -> Self {
        Self {
            synthesis_model: SynthesisModel::Neural,
            vocoder_model: VocoderModel::HiFiGAN,
            pitch_model: PitchModel::Neural,
            duration_model: DurationModel::Neural,
            precision: ModelPrecision::Float32,
            cache_models: true,
            max_cache_size: 512,
        }
    }
}

impl Default for EffectSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            default_chain: vec![
                String::from("vibrato"),
                String::from("breath"),
                String::from("reverb"),
            ],
            parameters: HashMap::new(),
            parallel_processing: true,
            max_chain_length: 10,
        }
    }
}

impl Default for PerformanceSettings {
    fn default() -> Self {
        Self {
            num_threads: None, // Use system default
            gpu_acceleration: false,
            gpu_device: None,
            memory_limit: None,
            monitoring: false,
            timeout: Some(300), // 5 minutes
        }
    }
}

impl WindowFunction {
    /// Get window coefficients for given size
    pub fn coefficients(&self, size: usize) -> Vec<f32> {
        match self {
            WindowFunction::Hamming => (0..size)
                .map(|n| {
                    0.54 - 0.46 * (2.0 * std::f32::consts::PI * n as f32 / (size - 1) as f32).cos()
                })
                .collect(),
            WindowFunction::Hanning => (0..size)
                .map(|n| {
                    0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / (size - 1) as f32).cos())
                })
                .collect(),
            WindowFunction::Blackman => (0..size)
                .map(|n| {
                    let angle = 2.0 * std::f32::consts::PI * n as f32 / (size - 1) as f32;
                    0.42 - 0.5 * angle.cos() + 0.08 * (2.0 * angle).cos()
                })
                .collect(),
            WindowFunction::Kaiser => {
                // Simplified Kaiser window with beta=8.6
                let beta = 8.6;
                let alpha = (size - 1) as f32 / 2.0;
                (0..size)
                    .map(|n| {
                        let x = (n as f32 - alpha) / alpha;
                        let arg = beta * (1.0 - x * x).sqrt();
                        Self::bessel_i0(arg) / Self::bessel_i0(beta)
                    })
                    .collect()
            }
            WindowFunction::Gaussian => {
                // Simplified Gaussian window with sigma=0.4
                let sigma = 0.4;
                let alpha = (size - 1) as f32 / 2.0;
                (0..size)
                    .map(|n| {
                        let x = (n as f32 - alpha) / alpha;
                        (-0.5 * (x / sigma).powi(2)).exp()
                    })
                    .collect()
            }
            WindowFunction::Rectangular => {
                vec![1.0; size]
            }
        }
    }

    /// Bessel function I0 for Kaiser window
    fn bessel_i0(x: f32) -> f32 {
        let mut result = 1.0;
        let mut term = 1.0;
        let mut k = 1.0;

        while term > 1e-12 {
            term *= (x / (2.0 * k)).powi(2);
            result += term;
            k += 1.0;
        }

        result
    }
}

impl SingingConfig {
    /// Create configuration for specific voice type
    pub fn for_voice_type(voice_type: VoiceType) -> Self {
        let mut config = Self::default();
        config.default_voice.voice_type = voice_type;
        config.default_voice.range = voice_type.frequency_range();
        config.default_voice.f0_mean = voice_type.f0_mean();
        config
    }

    /// Create high-quality configuration
    pub fn high_quality() -> Self {
        let mut config = Self::default();
        config.quality.quality_level = 10;
        config.quality.high_quality_pitch = true;
        config.quality.advanced_vibrato = true;
        config.quality.breath_modeling = true;
        config.quality.formant_modeling = true;
        config.quality.fft_size = 4096;
        config.quality.hop_size = 256;
        config.audio.sample_rate = 48000;
        config.audio.bit_depth = 24;
        config.model.precision = ModelPrecision::Float64;
        config
    }

    /// Create real-time configuration
    pub fn real_time() -> Self {
        let mut config = Self::default();
        config.audio.real_time = true;
        config.audio.buffer_size = 512;
        config.quality.quality_level = 5;
        config.quality.fft_size = 1024;
        config.quality.hop_size = 256;
        config.performance.gpu_acceleration = true;
        config.effects.parallel_processing = true;
        config
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.audio.sample_rate < 8000 || self.audio.sample_rate > 192000 {
            return Err(String::from(
                "Invalid sample rate. Must be between 8000 and 192000 Hz",
            ));
        }

        if self.audio.channels == 0 || self.audio.channels > 2 {
            return Err(String::from("Invalid channel count. Must be 1 or 2"));
        }

        if self.quality.quality_level > 10 {
            return Err(String::from(
                "Invalid quality level. Must be between 0 and 10",
            ));
        }

        if self.quality.fft_size < 256 || self.quality.fft_size > 8192 {
            return Err(String::from(
                "Invalid FFT size. Must be between 256 and 8192",
            ));
        }

        if self.quality.hop_size > self.quality.fft_size {
            return Err(String::from(
                "Hop size must be less than or equal to FFT size",
            ));
        }

        if self.audio.overlap_ratio < 0.0 || self.audio.overlap_ratio > 1.0 {
            return Err("Invalid overlap ratio. Must be between 0.0 and 1.0".to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = SingingConfigBuilder::new()
            .enabled(true)
            .default_expression(Expression::Happy)
            .build();

        assert!(config.enabled);
        assert_eq!(config.default_expression, Expression::Happy);
    }

    #[test]
    fn test_config_validation() {
        let config = SingingConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid_config = config.clone();
        invalid_config.audio.sample_rate = 1000;
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_window_functions() {
        let size = 256;
        let hamming = WindowFunction::Hamming.coefficients(size);
        let hanning = WindowFunction::Hanning.coefficients(size);
        let rectangular = WindowFunction::Rectangular.coefficients(size);

        assert_eq!(hamming.len(), size);
        assert_eq!(hanning.len(), size);
        assert_eq!(rectangular.len(), size);

        // Rectangular window should be all ones
        assert!(rectangular.iter().all(|&x| (x - 1.0).abs() < 1e-6));

        // Hamming and Hanning should be symmetric
        assert!((hamming[0] - hamming[size - 1]).abs() < 1e-6);
        assert!((hanning[0] - hanning[size - 1]).abs() < 1e-6);
    }

    #[test]
    fn test_voice_type_config() {
        let config = SingingConfig::for_voice_type(VoiceType::Soprano);
        assert_eq!(config.default_voice.voice_type, VoiceType::Soprano);
        assert_eq!(
            config.default_voice.range,
            VoiceType::Soprano.frequency_range()
        );
    }

    #[test]
    fn test_quality_configs() {
        let hq_config = SingingConfig::high_quality();
        let rt_config = SingingConfig::real_time();

        assert_eq!(hq_config.quality.quality_level, 10);
        assert_eq!(rt_config.quality.quality_level, 5);
        assert!(rt_config.audio.real_time);
        assert!(!hq_config.audio.real_time);
    }
}
