//! Built-in audio effects plugins.
//!
//! This module provides a collection of high-quality audio effects:
//! - **Reverb**: Freeverb-style reverb with room size and damping control
//! - **Equalizer**: 3-band parametric equalizer for frequency shaping
//! - **Compressor**: Dynamic range compressor with envelope following
//! - **Delay**: Delay effect with feedback and high-frequency damping
//! - **Spatial**: 3D spatial audio positioning effect

mod compressor;
mod delay;
mod equalizer;
mod filters;
mod reverb;
mod spatial;

// Re-export all public effects
pub use compressor::CompressorEffect;
pub use delay::DelayEffect;
pub use equalizer::EqualizerEffect;
pub use reverb::ReverbEffect;
pub use spatial::SpatialAudioEffect;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugins::{AudioEffect, ParameterType, ParameterValue};

    #[tokio::test]
    async fn test_reverb_effect() {
        let reverb = ReverbEffect::new();

        // Test parameter setting
        reverb
            .set_parameter("mix", ParameterValue::Float(0.5))
            .unwrap();
        assert_eq!(*reverb.mix.read().unwrap(), 0.5);

        // Test audio processing
        let audio = crate::AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);
        let processed = reverb.process_audio(&audio).await.unwrap();

        assert_eq!(processed.len(), audio.len());
        assert_eq!(processed.sample_rate(), audio.sample_rate());
    }

    #[tokio::test]
    async fn test_equalizer_effect() {
        let eq = EqualizerEffect::new();

        // Test parameter setting
        eq.set_parameter("low_gain", ParameterValue::Float(3.0))
            .unwrap();
        eq.set_parameter("mid_gain", ParameterValue::Float(-2.0))
            .unwrap();
        eq.set_parameter("high_gain", ParameterValue::Float(1.0))
            .unwrap();

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
        let comp = CompressorEffect::new();

        // Test parameter setting
        comp.set_parameter("threshold", ParameterValue::Float(-18.0))
            .unwrap();
        comp.set_parameter("ratio", ParameterValue::Float(6.0))
            .unwrap();

        assert_eq!(*comp.threshold.read().unwrap(), -18.0);
        assert_eq!(*comp.ratio.read().unwrap(), 6.0);

        // Test audio processing with loud signal
        let audio = crate::AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.9); // Loud signal
        let processed = comp.process_audio(&audio).await.unwrap();

        // Should have reduced the peaks
        let original_peak = audio.samples().iter().map(|&s| s.abs()).fold(0.0, f32::max);
        let processed_peak = processed
            .samples()
            .iter()
            .map(|&s| s.abs())
            .fold(0.0, f32::max);

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

    #[tokio::test]
    async fn test_delay_effect() {
        let delay = DelayEffect::new();

        // Test parameter setting
        delay
            .set_parameter("delay_ms", ParameterValue::Float(500.0))
            .unwrap();
        delay
            .set_parameter("feedback", ParameterValue::Float(0.6))
            .unwrap();

        assert_eq!(*delay.delay_ms.read().unwrap(), 500.0);
        assert_eq!(*delay.feedback.read().unwrap(), 0.6);

        // Test audio processing
        let audio = crate::AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.5);
        let processed = delay.process_audio(&audio).await.unwrap();

        assert_eq!(processed.len(), audio.len());
        assert_eq!(processed.sample_rate(), audio.sample_rate());
    }

    #[tokio::test]
    async fn test_spatial_audio_effect() {
        let spatial = SpatialAudioEffect::new();

        // Test parameter setting
        spatial
            .set_parameter("azimuth", ParameterValue::Float(45.0))
            .unwrap();
        spatial
            .set_parameter("distance", ParameterValue::Float(2.0))
            .unwrap();

        assert_eq!(*spatial.azimuth.read().unwrap(), 45.0);
        assert_eq!(*spatial.distance.read().unwrap(), 2.0);

        // Test audio processing
        let mono_audio = crate::AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.5);
        let processed = spatial.process_audio(&mono_audio).await.unwrap();

        // Verify processing occurred
        assert_eq!(processed.sample_rate(), mono_audio.sample_rate());
    }

    #[test]
    fn test_plugin_metadata() {
        let reverb = ReverbEffect::new();
        use crate::plugins::VoirsPlugin;
        assert_eq!(reverb.name(), "Reverb");
        assert_eq!(reverb.version(), "1.0.0");
        assert_eq!(reverb.author(), "VoiRS Team");

        let eq = EqualizerEffect::new();
        assert_eq!(eq.name(), "Equalizer");

        let comp = CompressorEffect::new();
        assert_eq!(comp.name(), "Compressor");

        let delay = DelayEffect::new();
        assert_eq!(delay.name(), "Delay");

        let spatial = SpatialAudioEffect::new();
        assert_eq!(spatial.name(), "Spatial Audio");
    }
}
