//! Tests for acoustic emotion adapter

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use super::super::super::integration::{EmotionSpeakerMapping, VoiceQualityMapping};
    use super::super::core::AcousticEmotionAdapter;
    use crate::types::{Emotion, EmotionIntensity, EmotionParameters, EmotionVector};

    #[test]
    fn test_acoustic_adapter_creation() {
        let adapter = AcousticEmotionAdapter::new();

        #[cfg(feature = "acoustic-integration")]
        {
            assert!(adapter.get_speaker_mappings().is_empty());
        }

        // Test should compile regardless of feature flag
        let _adapter = adapter;
    }

    #[cfg(feature = "acoustic-integration")]
    #[test]
    fn test_speaker_mapping() {
        let mapping = EmotionSpeakerMapping::new()
            .with_speaker_id("happy_speaker".to_string())
            .with_param("pitch_shift".to_string(), 1.2);

        assert_eq!(mapping.speaker_id, Some("happy_speaker".to_string()));
        assert_eq!(mapping.speaker_params.get("pitch_shift"), Some(&1.2));
    }

    #[cfg(feature = "acoustic-integration")]
    #[test]
    fn test_voice_quality_mapping() {
        let quality = VoiceQualityMapping::neutral()
            .with_breathiness(0.3)
            .with_roughness(0.1);

        assert_eq!(quality.breathiness, 0.3);
        assert_eq!(quality.roughness, 0.1);
    }

    #[test]
    fn test_disabled_features() {
        let adapter = AcousticEmotionAdapter::new();

        #[cfg(not(feature = "acoustic-integration"))]
        {
            let emotion_params = EmotionParameters::neutral();
            let result = adapter.synthesize_with_emotion("test", &emotion_params);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_rms_energy_calculation() {
        let adapter = AcousticEmotionAdapter::new();

        // Test empty audio
        let empty_audio = vec![];
        assert_eq!(adapter.calculate_rms_energy(&empty_audio), 0.0);

        // Test silent audio
        let silent_audio = vec![0.0; 1000];
        assert_eq!(adapter.calculate_rms_energy(&silent_audio), 0.0);

        // Test audio with constant amplitude
        let constant_audio = vec![0.5; 1000];
        let rms = adapter.calculate_rms_energy(&constant_audio);
        assert!((rms - 0.5).abs() < 0.001);

        // Test sine wave (approximate RMS should be amplitude / sqrt(2))
        let mut sine_wave = vec![0.0; 1000];
        for (i, sample) in sine_wave.iter_mut().enumerate() {
            *sample = (2.0 * std::f32::consts::PI * i as f32 / 1000.0).sin();
        }
        let rms = adapter.calculate_rms_energy(&sine_wave);
        assert!(rms > 0.6 && rms < 0.8); // Should be close to 1/sqrt(2) ≈ 0.707
    }

    #[test]
    fn test_zero_crossing_rate_calculation() {
        let adapter = AcousticEmotionAdapter::new();

        // Test empty audio
        let empty_audio = vec![];
        assert_eq!(adapter.calculate_zero_crossing_rate(&empty_audio), 0.0);

        // Test single sample
        let single_sample = vec![0.5];
        assert_eq!(adapter.calculate_zero_crossing_rate(&single_sample), 0.0);

        // Test constant positive signal (no crossings)
        let constant_audio = vec![0.5; 1000];
        assert_eq!(adapter.calculate_zero_crossing_rate(&constant_audio), 0.0);

        // Test alternating signal (maximum crossings)
        let alternating_audio: Vec<f32> = (0..1000)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let zcr = adapter.calculate_zero_crossing_rate(&alternating_audio);
        assert!(zcr > 0.9); // Should be close to 1.0 (maximum ZCR)

        // Test sine wave
        let mut sine_wave = vec![0.0; 1000];
        for (i, sample) in sine_wave.iter_mut().enumerate() {
            *sample = (2.0 * std::f32::consts::PI * i as f32 / 100.0).sin();
        }
        let zcr = adapter.calculate_zero_crossing_rate(&sine_wave);
        assert!(zcr > 0.01 && zcr < 0.1); // Should have some crossings but not too many
    }

    #[test]
    fn test_spectral_centroid_calculation() {
        let adapter = AcousticEmotionAdapter::new();
        let sample_rate = 44100.0;

        // Test short audio (below minimum window size)
        let short_audio = vec![0.5; 100];
        let result = adapter.calculate_spectral_centroid(&short_audio, sample_rate);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), sample_rate / 4.0); // Should return default value

        // Test longer audio
        let long_audio = vec![0.1; 1024];
        let result = adapter.calculate_spectral_centroid(&long_audio, sample_rate);
        assert!(result.is_ok());
        let centroid = result.unwrap();
        assert!(centroid > 0.0 && centroid < sample_rate / 2.0);

        // Test that the function handles different inputs without crashing
        let mut high_freq_audio = vec![0.0; 1024];
        for (i, sample) in high_freq_audio.iter_mut().enumerate() {
            *sample = (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sample_rate).sin();
        }
        let high_centroid = adapter
            .calculate_spectral_centroid(&high_freq_audio, sample_rate)
            .unwrap();
        assert!(high_centroid > 0.0);

        let mut low_freq_audio = vec![0.0; 1024];
        for (i, sample) in low_freq_audio.iter_mut().enumerate() {
            *sample = (2.0 * std::f32::consts::PI * 100.0 * i as f32 / sample_rate).sin();
        }
        let low_centroid = adapter
            .calculate_spectral_centroid(&low_freq_audio, sample_rate)
            .unwrap();
        assert!(low_centroid > 0.0);

        // Test that both calculations return reasonable values
        assert!(high_centroid < sample_rate / 2.0);
        assert!(low_centroid < sample_rate / 2.0);
    }

    #[test]
    fn test_basic_emotion_feature_extraction() {
        let adapter = AcousticEmotionAdapter::new();

        // Test empty audio
        let empty_audio = vec![];
        let result = adapter
            .extract_basic_emotion_features(&empty_audio, 44100)
            .unwrap();
        assert_eq!(result.dimensions.valence, 0.0);
        assert_eq!(result.dimensions.arousal, 0.0);
        assert_eq!(result.dimensions.dominance, 0.0);

        // Test audio with high energy
        let high_energy_audio = vec![0.8; 1024];
        let result = adapter
            .extract_basic_emotion_features(&high_energy_audio, 44100)
            .unwrap();
        assert!(result.dimensions.dominance > 0.0);

        // Test quiet audio
        let quiet_audio = vec![0.1; 1024];
        let quiet_result = adapter
            .extract_basic_emotion_features(&quiet_audio, 44100)
            .unwrap();

        assert!(result.dimensions.dominance > quiet_result.dimensions.dominance);

        // Test that dimensions are properly clamped
        assert!(result.dimensions.dominance <= 1.0 && result.dimensions.dominance >= -1.0);
        assert!(
            quiet_result.dimensions.dominance <= 1.0 && quiet_result.dimensions.dominance >= -1.0
        );
    }

    #[test]
    fn test_emotion_feature_extraction_api() {
        let adapter = AcousticEmotionAdapter::new();

        // Test the public API
        let test_audio = vec![0.3; 1024];
        let result = adapter.extract_emotion_features(&test_audio, 44100);
        assert!(result.is_ok());

        let emotion_vector = result.unwrap();
        assert!(
            emotion_vector.dimensions.valence >= -1.0 && emotion_vector.dimensions.valence <= 1.0
        );
        assert!(
            emotion_vector.dimensions.arousal >= -1.0 && emotion_vector.dimensions.arousal <= 1.0
        );
        assert!(
            emotion_vector.dimensions.dominance >= -1.0
                && emotion_vector.dimensions.dominance <= 1.0
        );
    }

    #[cfg(feature = "acoustic-integration")]
    #[tokio::test]
    async fn test_synthesize_with_emotion_basic() {
        let adapter = AcousticEmotionAdapter::new();

        // Test with neutral emotion
        let neutral_params = EmotionParameters::neutral();
        let result = adapter
            .synthesize_with_emotion("hello", &neutral_params)
            .await;
        match &result {
            Ok(audio) => {
                assert!(!audio.is_empty());
                assert!(audio.len() >= 16000);
            }
            Err(e) => {
                println!("Synthesis failed as expected: {:?}", e);
                assert!(
                    e.to_string().contains("Placeholder")
                        || e.to_string().contains("not implemented")
                        || e.to_string().contains("No base acoustic configuration")
                        || e.to_string()
                            .contains("No valid acoustic configuration set")
                );
            }
        }
    }

    #[cfg(feature = "acoustic-integration")]
    #[tokio::test]
    async fn test_synthesize_with_different_emotions() {
        let adapter = AcousticEmotionAdapter::new();

        let mut happy_vector = EmotionVector::new();
        happy_vector.add_emotion(Emotion::Happy, EmotionIntensity::HIGH);
        let happy_params = EmotionParameters::new(happy_vector).with_prosody(1.2, 1.1, 1.3);

        let mut sad_vector = EmotionVector::new();
        sad_vector.add_emotion(Emotion::Sad, EmotionIntensity::HIGH);
        let sad_params = EmotionParameters::new(sad_vector).with_prosody(0.8, 0.9, 0.7);

        let happy_result = adapter.synthesize_with_emotion("test", &happy_params).await;
        let sad_result = adapter.synthesize_with_emotion("test", &sad_params).await;

        match (&happy_result, &sad_result) {
            (Ok(happy_audio), Ok(sad_audio)) => {
                assert!(!happy_audio.is_empty());
                assert!(!sad_audio.is_empty());
                assert_eq!(happy_audio.len(), sad_audio.len());
            }
            (Err(e1), Err(e2)) => {
                let is_acceptable_error = |e: &crate::Error| {
                    let err_str = e.to_string();
                    err_str.contains("Placeholder")
                        || err_str.contains("not implemented")
                        || err_str.contains("No base acoustic configuration")
                        || err_str.contains("No valid acoustic configuration set")
                };
                assert!(is_acceptable_error(e1));
                assert!(is_acceptable_error(e2));
            }
            _ => {
                panic!(
                    "Mixed results: happy={:?}, sad={:?}",
                    happy_result, sad_result
                );
            }
        }
    }

    #[cfg(feature = "acoustic-integration")]
    #[test]
    fn test_vocoder_emotion_config() {
        let adapter = AcousticEmotionAdapter::new();

        let mut excited_vector = EmotionVector::new();
        excited_vector.add_emotion(Emotion::Excited, EmotionIntensity::HIGH);
        let excited_params = EmotionParameters::new(excited_vector).with_prosody(1.3, 1.2, 1.4);

        let vocoder_config = adapter.apply_emotion_to_vocoder(&excited_params, &());
        assert!(vocoder_config.is_ok());

        let config = vocoder_config.unwrap();
        assert_eq!(config.pitch_shift, excited_params.pitch_shift);
        assert_eq!(config.energy_scale, excited_params.energy_scale);
        assert!(config.formant_shift > 1.0);
    }

    #[cfg(feature = "acoustic-integration")]
    #[tokio::test]
    async fn test_vocode_with_emotion() {
        let adapter = AcousticEmotionAdapter::new();

        let input_audio = vec![0.1; 1024];
        let emotion_params = EmotionParameters::neutral();
        let base_config = ();

        let result = adapter
            .vocode_with_emotion(&input_audio, &emotion_params, &base_config)
            .await;
        assert!(result.is_ok());

        let output_audio = result.unwrap();
        assert!(!output_audio.is_empty());
        assert_eq!(output_audio.len(), input_audio.len());
    }

    #[test]
    fn test_speaker_mappings_management() {
        let mut adapter = AcousticEmotionAdapter::new();

        assert!(adapter.get_speaker_mappings().is_empty());

        #[cfg(feature = "acoustic-integration")]
        {
            let mapping = EmotionSpeakerMapping::new()
                .with_speaker_id("test_speaker".to_string())
                .with_param("pitch".to_string(), 1.5);

            adapter.add_speaker_mapping("happy".to_string(), mapping);

            assert_eq!(adapter.get_speaker_mappings().len(), 1);
            assert!(adapter.get_speaker_mappings().contains_key("happy"));
        }
    }

    #[cfg(feature = "acoustic-integration")]
    #[tokio::test]
    async fn test_base_config_integration() {
        use voirs_acoustic::config::synthesis::SynthesisConfig;

        let base_config = SynthesisConfig::default();
        let adapter = AcousticEmotionAdapter::new().with_base_synthesis_config(base_config);

        let emotion_params = EmotionParameters::neutral();
        let result = adapter
            .synthesize_with_emotion("test with config", &emotion_params)
            .await;
        // `synthesize_with_emotion` always fails closed: this adapter has no
        // real voirs_acoustic::AcousticModel/vocoder wired in, so even with a
        // valid base config it must not fabricate an audio-effects tone and
        // present that as synthesized speech. A configured base config still
        // changes *which* error is returned (it gets past the "no config"
        // check and into the real-but-unimplemented-synthesis error).
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not implemented"),
            "expected the honest not-implemented error, got: {err}"
        );
    }

    #[test]
    fn test_emotion_dimension_clamping() {
        let adapter = AcousticEmotionAdapter::new();

        let extreme_audio = vec![10.0; 1024];
        let result = adapter
            .extract_basic_emotion_features(&extreme_audio, 44100)
            .unwrap();

        assert!(result.dimensions.valence >= -1.0 && result.dimensions.valence <= 1.0);
        assert!(result.dimensions.arousal >= -1.0 && result.dimensions.arousal <= 1.0);
        assert!(result.dimensions.dominance >= -1.0 && result.dimensions.dominance <= 1.0);
    }

    #[cfg(feature = "acoustic-integration")]
    #[tokio::test]
    async fn test_synthesize_with_enhanced_emotion_fails_closed_not_fabricated_tone() {
        let adapter = AcousticEmotionAdapter::new();
        let emotion_params = EmotionParameters::neutral();

        // Must never fabricate an additive-harmonic tone from text.len() and
        // present it as "enhanced emotion-aware acoustic synthesis" - there
        // is no real acoustic model/vocoder wired into this adapter.
        let result = adapter
            .synthesize_with_enhanced_emotion("hello world", &emotion_params)
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not implemented"),
            "expected the honest not-implemented error, got: {err}"
        );
    }

    #[test]
    fn test_audio_processing_edge_cases() {
        let adapter = AcousticEmotionAdapter::new();

        let short_audio = vec![0.5; 10];
        let result = adapter.calculate_spectral_centroid(&short_audio, 44100.0);
        assert!(result.is_ok());

        let mut nan_audio = vec![0.5; 1024];
        nan_audio[100] = f32::NAN;
        let rms = adapter.calculate_rms_energy(&nan_audio);
        assert!(rms.is_nan() || rms >= 0.0);
    }
}
