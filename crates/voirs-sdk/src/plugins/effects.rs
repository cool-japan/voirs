//! Built-in audio effects plugins.

use crate::{
    audio::AudioBuffer,
    error::Result,
    plugins::{AudioEffect, ParameterDefinition, ParameterType, ParameterValue, VoirsPlugin},
    VoirsError,
};
use async_trait::async_trait;
use std::{collections::HashMap, sync::RwLock};

/// Reverb effect plugin
pub struct ReverbEffect {
    /// Wet/dry mix (0.0 = dry, 1.0 = wet)
    pub mix: RwLock<f32>,
    
    /// Room size (0.0 - 1.0)
    pub room_size: RwLock<f32>,
    
    /// Damping factor (0.0 - 1.0)
    pub damping: RwLock<f32>,
    
    /// Decay time in seconds
    pub decay_time: RwLock<f32>,
}

impl ReverbEffect {
    pub fn new() -> Self {
        Self {
            mix: RwLock::new(0.3),
            room_size: RwLock::new(0.5),
            damping: RwLock::new(0.5),
            decay_time: RwLock::new(2.0),
        }
    }
}

impl Default for ReverbEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl VoirsPlugin for ReverbEffect {
    fn name(&self) -> &str {
        "Reverb"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "High-quality reverb effect for spatial audio enhancement"
    }

    fn author(&self) -> &str {
        "VoiRS Team"
    }
}

#[async_trait]
impl AudioEffect for ReverbEffect {
    async fn process_audio(&self, audio: &AudioBuffer) -> Result<AudioBuffer> {
        // TODO: Implement proper reverb algorithm
        // For now, implement a simple echo effect as placeholder
        let mut processed = audio.clone();
        let samples = processed.samples_mut();
        let delay_samples = (0.1 * audio.sample_rate() as f32) as usize; // 100ms delay
        let mix = *self.mix.read().unwrap();
        
        if samples.len() > delay_samples {
            for i in delay_samples..samples.len() {
                let delayed_sample = samples[i - delay_samples] * mix * 0.5;
                samples[i] = samples[i] * (1.0 - mix) + delayed_sample;
            }
        }
        
        Ok(processed)
    }

    fn get_parameters(&self) -> HashMap<String, ParameterValue> {
        let mut params = HashMap::new();
        params.insert("mix".to_string(), ParameterValue::Float(*self.mix.read().unwrap()));
        params.insert("room_size".to_string(), ParameterValue::Float(*self.room_size.read().unwrap()));
        params.insert("damping".to_string(), ParameterValue::Float(*self.damping.read().unwrap()));
        params.insert("decay_time".to_string(), ParameterValue::Float(*self.decay_time.read().unwrap()));
        params
    }

    fn set_parameter(&self, name: &str, value: ParameterValue) -> Result<()> {
        match name {
            "mix" => {
                if let Some(v) = value.as_f32() {
                    *self.mix.write().unwrap() = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal("plugins", "Invalid mix parameter type"))
                }
            }
            "room_size" => {
                if let Some(v) = value.as_f32() {
                    *self.room_size.write().unwrap() = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal("plugins", "Invalid room_size parameter type"))
                }
            }
            "damping" => {
                if let Some(v) = value.as_f32() {
                    *self.damping.write().unwrap() = v.clamp(0.0, 1.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal("plugins", "Invalid damping parameter type"))
                }
            }
            "decay_time" => {
                if let Some(v) = value.as_f32() {
                    *self.decay_time.write().unwrap() = v.clamp(0.1, 10.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal("plugins", "Invalid decay_time parameter type"))
                }
            }
            _ => Err(VoirsError::internal("plugins", format!("Unknown parameter: {}", name))),
        }
    }

    fn get_parameter_definition(&self, name: &str) -> Option<ParameterDefinition> {
        match name {
            "mix" => Some(ParameterDefinition {
                name: "mix".to_string(),
                description: "Wet/dry mix level".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.3),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(1.0)),
                step_size: Some(0.01),
                realtime_safe: true,
            }),
            "room_size" => Some(ParameterDefinition {
                name: "room_size".to_string(),
                description: "Virtual room size".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.5),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(1.0)),
                step_size: Some(0.01),
                realtime_safe: false,
            }),
            "damping" => Some(ParameterDefinition {
                name: "damping".to_string(),
                description: "High frequency damping".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.5),
                min_value: Some(ParameterValue::Float(0.0)),
                max_value: Some(ParameterValue::Float(1.0)),
                step_size: Some(0.01),
                realtime_safe: true,
            }),
            "decay_time" => Some(ParameterDefinition {
                name: "decay_time".to_string(),
                description: "Reverb decay time in seconds".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(2.0),
                min_value: Some(ParameterValue::Float(0.1)),
                max_value: Some(ParameterValue::Float(10.0)),
                step_size: Some(0.1),
                realtime_safe: false,
            }),
            _ => None,
        }
    }

    fn get_latency_samples(&self) -> usize {
        // Reverb typically adds some latency
        512
    }
}

/// Equalizer effect plugin
pub struct EqualizerEffect {
    /// Low frequency gain (dB)
    pub low_gain: RwLock<f32>,
    
    /// Mid frequency gain (dB)
    pub mid_gain: RwLock<f32>,
    
    /// High frequency gain (dB)
    pub high_gain: RwLock<f32>,
    
    /// Low frequency cutoff (Hz)
    pub low_freq: RwLock<f32>,
    
    /// High frequency cutoff (Hz)
    pub high_freq: RwLock<f32>,
}

impl EqualizerEffect {
    pub fn new() -> Self {
        Self {
            low_gain: RwLock::new(0.0),
            mid_gain: RwLock::new(0.0),
            high_gain: RwLock::new(0.0),
            low_freq: RwLock::new(200.0),
            high_freq: RwLock::new(2000.0),
        }
    }
}

impl Default for EqualizerEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl VoirsPlugin for EqualizerEffect {
    fn name(&self) -> &str {
        "Equalizer"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "3-band equalizer for frequency shaping"
    }

    fn author(&self) -> &str {
        "VoiRS Team"
    }
}

#[async_trait]
impl AudioEffect for EqualizerEffect {
    async fn process_audio(&self, audio: &AudioBuffer) -> Result<AudioBuffer> {
        // TODO: Implement proper EQ filtering
        // For now, just apply simple gain to demonstrate
        let mut processed = audio.clone();
        let gain_factor = 1.0 + (*self.low_gain.read().unwrap() + *self.mid_gain.read().unwrap() + *self.high_gain.read().unwrap()) / 60.0; // Rough approximation
        
        for sample in processed.samples_mut() {
            *sample *= gain_factor;
            *sample = sample.clamp(-1.0, 1.0);
        }
        
        Ok(processed)
    }

    fn get_parameters(&self) -> HashMap<String, ParameterValue> {
        let mut params = HashMap::new();
        params.insert("low_gain".to_string(), ParameterValue::Float(*self.low_gain.read().unwrap()));
        params.insert("mid_gain".to_string(), ParameterValue::Float(*self.mid_gain.read().unwrap()));
        params.insert("high_gain".to_string(), ParameterValue::Float(*self.high_gain.read().unwrap()));
        params.insert("low_freq".to_string(), ParameterValue::Float(*self.low_freq.read().unwrap()));
        params.insert("high_freq".to_string(), ParameterValue::Float(*self.high_freq.read().unwrap()));
        params
    }

    fn set_parameter(&self, name: &str, value: ParameterValue) -> Result<()> {
        match name {
            "low_gain" => {
                if let Some(v) = value.as_f32() {
                    *self.low_gain.write().unwrap() = v.clamp(-20.0, 20.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal("plugins", "Invalid low_gain parameter type"))
                }
            }
            "mid_gain" => {
                if let Some(v) = value.as_f32() {
                    *self.mid_gain.write().unwrap() = v.clamp(-20.0, 20.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal("plugins", "Invalid mid_gain parameter type"))
                }
            }
            "high_gain" => {
                if let Some(v) = value.as_f32() {
                    *self.high_gain.write().unwrap() = v.clamp(-20.0, 20.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal("plugins", "Invalid high_gain parameter type"))
                }
            }
            "low_freq" => {
                if let Some(v) = value.as_f32() {
                    *self.low_freq.write().unwrap() = v.clamp(20.0, 20000.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal("plugins", "Invalid low_freq parameter type"))
                }
            }
            "high_freq" => {
                if let Some(v) = value.as_f32() {
                    *self.high_freq.write().unwrap() = v.clamp(20.0, 20000.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal("plugins", "Invalid high_freq parameter type"))
                }
            }
            _ => Err(VoirsError::internal("plugins", format!("Unknown parameter: {}", name))),
        }
    }

    fn get_parameter_definition(&self, name: &str) -> Option<ParameterDefinition> {
        match name {
            "low_gain" => Some(ParameterDefinition {
                name: "low_gain".to_string(),
                description: "Low frequency gain in dB".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.0),
                min_value: Some(ParameterValue::Float(-20.0)),
                max_value: Some(ParameterValue::Float(20.0)),
                step_size: Some(0.1),
                realtime_safe: true,
            }),
            "mid_gain" => Some(ParameterDefinition {
                name: "mid_gain".to_string(),
                description: "Mid frequency gain in dB".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.0),
                min_value: Some(ParameterValue::Float(-20.0)),
                max_value: Some(ParameterValue::Float(20.0)),
                step_size: Some(0.1),
                realtime_safe: true,
            }),
            "high_gain" => Some(ParameterDefinition {
                name: "high_gain".to_string(),
                description: "High frequency gain in dB".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.0),
                min_value: Some(ParameterValue::Float(-20.0)),
                max_value: Some(ParameterValue::Float(20.0)),
                step_size: Some(0.1),
                realtime_safe: true,
            }),
            "low_freq" => Some(ParameterDefinition {
                name: "low_freq".to_string(),
                description: "Low/mid crossover frequency in Hz".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(200.0),
                min_value: Some(ParameterValue::Float(20.0)),
                max_value: Some(ParameterValue::Float(20000.0)),
                step_size: Some(10.0),
                realtime_safe: false,
            }),
            "high_freq" => Some(ParameterDefinition {
                name: "high_freq".to_string(),
                description: "Mid/high crossover frequency in Hz".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(2000.0),
                min_value: Some(ParameterValue::Float(20.0)),
                max_value: Some(ParameterValue::Float(20000.0)),
                step_size: Some(10.0),
                realtime_safe: false,
            }),
            _ => None,
        }
    }
}

/// Compressor effect plugin
pub struct CompressorEffect {
    /// Threshold in dB
    pub threshold: RwLock<f32>,
    
    /// Compression ratio
    pub ratio: RwLock<f32>,
    
    /// Attack time in milliseconds
    pub attack_ms: RwLock<f32>,
    
    /// Release time in milliseconds
    pub release_ms: RwLock<f32>,
    
    /// Makeup gain in dB
    pub makeup_gain: RwLock<f32>,
}

impl CompressorEffect {
    pub fn new() -> Self {
        Self {
            threshold: RwLock::new(-12.0),
            ratio: RwLock::new(4.0),
            attack_ms: RwLock::new(10.0),
            release_ms: RwLock::new(100.0),
            makeup_gain: RwLock::new(0.0),
        }
    }
}

impl Default for CompressorEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl VoirsPlugin for CompressorEffect {
    fn name(&self) -> &str {
        "Compressor"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "Dynamic range compressor for level control"
    }

    fn author(&self) -> &str {
        "VoiRS Team"
    }
}

#[async_trait]
impl AudioEffect for CompressorEffect {
    async fn process_audio(&self, audio: &AudioBuffer) -> Result<AudioBuffer> {
        // TODO: Implement proper compressor algorithm
        // For now, implement simple limiting as placeholder
        let threshold_linear = 10.0_f32.powf(*self.threshold.read().unwrap() / 20.0);
        let makeup_linear = 10.0_f32.powf(*self.makeup_gain.read().unwrap() / 20.0);
        
        let mut processed = audio.clone();
        
        for sample in processed.samples_mut() {
            let abs_sample = sample.abs();
            if abs_sample > threshold_linear {
                let reduction = 1.0 / *self.ratio.read().unwrap();
                let excess = abs_sample - threshold_linear;
                let new_sample = threshold_linear + excess * reduction;
                *sample = new_sample * sample.signum() * makeup_linear;
            } else {
                *sample *= makeup_linear;
            }
            *sample = sample.clamp(-1.0, 1.0);
        }
        
        Ok(processed)
    }

    fn get_parameters(&self) -> HashMap<String, ParameterValue> {
        let mut params = HashMap::new();
        params.insert("threshold".to_string(), ParameterValue::Float(*self.threshold.read().unwrap()));
        params.insert("ratio".to_string(), ParameterValue::Float(*self.ratio.read().unwrap()));
        params.insert("attack_ms".to_string(), ParameterValue::Float(*self.attack_ms.read().unwrap()));
        params.insert("release_ms".to_string(), ParameterValue::Float(*self.release_ms.read().unwrap()));
        params.insert("makeup_gain".to_string(), ParameterValue::Float(*self.makeup_gain.read().unwrap()));
        params
    }

    fn set_parameter(&self, name: &str, value: ParameterValue) -> Result<()> {
        match name {
            "threshold" => {
                if let Some(v) = value.as_f32() {
                    *self.threshold.write().unwrap() = v.clamp(-60.0, 0.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal("plugins", "Invalid threshold parameter type"))
                }
            }
            "ratio" => {
                if let Some(v) = value.as_f32() {
                    *self.ratio.write().unwrap() = v.clamp(1.0, 20.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal("plugins", "Invalid ratio parameter type"))
                }
            }
            "attack_ms" => {
                if let Some(v) = value.as_f32() {
                    *self.attack_ms.write().unwrap() = v.clamp(0.1, 1000.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal("plugins", "Invalid attack_ms parameter type"))
                }
            }
            "release_ms" => {
                if let Some(v) = value.as_f32() {
                    *self.release_ms.write().unwrap() = v.clamp(1.0, 5000.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal("plugins", "Invalid release_ms parameter type"))
                }
            }
            "makeup_gain" => {
                if let Some(v) = value.as_f32() {
                    *self.makeup_gain.write().unwrap() = v.clamp(-20.0, 20.0);
                    Ok(())
                } else {
                    Err(VoirsError::internal("plugins", "Invalid makeup_gain parameter type"))
                }
            }
            _ => Err(VoirsError::internal("plugins", format!("Unknown parameter: {}", name))),
        }
    }

    fn get_parameter_definition(&self, name: &str) -> Option<ParameterDefinition> {
        match name {
            "threshold" => Some(ParameterDefinition {
                name: "threshold".to_string(),
                description: "Compression threshold in dB".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(-12.0),
                min_value: Some(ParameterValue::Float(-60.0)),
                max_value: Some(ParameterValue::Float(0.0)),
                step_size: Some(0.1),
                realtime_safe: true,
            }),
            "ratio" => Some(ParameterDefinition {
                name: "ratio".to_string(),
                description: "Compression ratio".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(4.0),
                min_value: Some(ParameterValue::Float(1.0)),
                max_value: Some(ParameterValue::Float(20.0)),
                step_size: Some(0.1),
                realtime_safe: true,
            }),
            "attack_ms" => Some(ParameterDefinition {
                name: "attack_ms".to_string(),
                description: "Attack time in milliseconds".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(10.0),
                min_value: Some(ParameterValue::Float(0.1)),
                max_value: Some(ParameterValue::Float(1000.0)),
                step_size: Some(0.1),
                realtime_safe: false,
            }),
            "release_ms" => Some(ParameterDefinition {
                name: "release_ms".to_string(),
                description: "Release time in milliseconds".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(100.0),
                min_value: Some(ParameterValue::Float(1.0)),
                max_value: Some(ParameterValue::Float(5000.0)),
                step_size: Some(1.0),
                realtime_safe: false,
            }),
            "makeup_gain" => Some(ParameterDefinition {
                name: "makeup_gain".to_string(),
                description: "Makeup gain in dB".to_string(),
                parameter_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.0),
                min_value: Some(ParameterValue::Float(-20.0)),
                max_value: Some(ParameterValue::Float(20.0)),
                step_size: Some(0.1),
                realtime_safe: true,
            }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_reverb_effect() {
        let mut reverb = ReverbEffect::new();
        
        // Test parameter setting
        reverb.set_parameter("mix", ParameterValue::Float(0.5)).unwrap();
        assert_eq!(*reverb.mix.read().unwrap(), 0.5);
        
        // Test audio processing
        let audio = crate::AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);
        let processed = reverb.process_audio(&audio).await.unwrap();
        
        assert_eq!(processed.len(), audio.len());
        assert_eq!(processed.sample_rate(), audio.sample_rate());
    }

    #[tokio::test]
    async fn test_equalizer_effect() {
        let mut eq = EqualizerEffect::new();
        
        // Test parameter setting
        eq.set_parameter("low_gain", ParameterValue::Float(3.0)).unwrap();
        eq.set_parameter("mid_gain", ParameterValue::Float(-2.0)).unwrap();
        eq.set_parameter("high_gain", ParameterValue::Float(1.0)).unwrap();
        
        assert_eq!(*eq.low_gain.read().unwrap(), 3.0);
        assert_eq!(*eq.mid_gain.read().unwrap(), -2.0);
        assert_eq!(*eq.high_gain.read().unwrap(), 1.0);
        
        // Test audio processing
        let audio = crate::AudioBuffer::sine_wave(1000.0, 0.5, 44100, 0.3);
        let processed = eq.process_audio(&audio).await.unwrap();
        
        assert_eq!(processed.len(), audio.len());
    }

    #[tokio::test]
    async fn test_compressor_effect() {
        let mut comp = CompressorEffect::new();
        
        // Test parameter setting
        comp.set_parameter("threshold", ParameterValue::Float(-18.0)).unwrap();
        comp.set_parameter("ratio", ParameterValue::Float(6.0)).unwrap();
        
        assert_eq!(*comp.threshold.read().unwrap(), -18.0);
        assert_eq!(*comp.ratio.read().unwrap(), 6.0);
        
        // Test audio processing with loud signal
        let audio = crate::AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.9); // Loud signal
        let processed = comp.process_audio(&audio).await.unwrap();
        
        // Should have reduced the peaks
        let original_peak = audio.samples().iter().map(|&s| s.abs()).fold(0.0, f32::max);
        let processed_peak = processed.samples().iter().map(|&s| s.abs()).fold(0.0, f32::max);
        
        assert!(processed_peak <= original_peak);
    }

    #[test]
    fn test_parameter_definitions() {
        let reverb = ReverbEffect::new();
        let mix_def = reverb.get_parameter_definition("mix").unwrap();
        
        assert_eq!(mix_def.name, "mix");
        assert_eq!(mix_def.parameter_type, ParameterType::Float);
        assert!(mix_def.realtime_safe);
        
        let room_def = reverb.get_parameter_definition("room_size").unwrap();
        assert!(!room_def.realtime_safe); // Room size changes aren't real-time safe
    }

    #[test]
    fn test_plugin_metadata() {
        let reverb = ReverbEffect::new();
        assert_eq!(reverb.name(), "Reverb");
        assert_eq!(reverb.version(), "1.0.0");
        assert_eq!(reverb.author(), "VoiRS Team");
        
        let eq = EqualizerEffect::new();
        assert_eq!(eq.name(), "Equalizer");
        
        let comp = CompressorEffect::new();
        assert_eq!(comp.name(), "Compressor");
    }
}