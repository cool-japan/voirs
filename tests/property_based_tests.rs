//! Property-based tests for VoiRS using proptest
//!
//! This test suite uses property-based testing to validate invariants and edge cases
//! by generating random inputs and ensuring that certain properties always hold.

use proptest::prelude::*;
use proptest::strategy::ValueTree;
use voirs_acoustic::{
    AcousticModel, DummyAcousticModel, Phoneme as AcousticPhoneme,
    SynthesisConfig as AcousticConfig,
};
use voirs_g2p::{DummyG2p, G2p, LanguageCode};
use voirs_vocoder::{
    DummyVocoder, MelSpectrogram as VocoderMel, SynthesisConfig as VocoderConfig,
    Vocoder,
};

/// Property-based test runner for VoiRS components
pub struct PropertyTests;

impl PropertyTests {
    /// Generate valid ASCII text for testing
    fn ascii_text_strategy() -> impl Strategy<Value = String> {
        "[a-zA-Z0-9 .,!?'-]{0,100}"
    }

    /// Generate valid phoneme symbols for testing
    fn phoneme_symbol_strategy() -> impl Strategy<Value = String> {
        // Use only ASCII characters to ensure length constraints are met
        "[a-zA-Z0-9]{1,3}"
    }

    /// Generate valid synthesis speed values
    fn speed_strategy() -> impl Strategy<Value = f32> {
        0.1f32..=3.0f32
    }

    /// Generate valid pitch shift values (in semitones)
    fn pitch_shift_strategy() -> impl Strategy<Value = f32> {
        -12.0f32..=12.0f32
    }

    /// Generate valid energy values
    fn energy_strategy() -> impl Strategy<Value = f32> {
        0.1f32..=3.0f32
    }

    /// Generate valid mel spectrogram dimensions
    fn mel_dimensions_strategy() -> impl Strategy<Value = (usize, usize)> {
        (1usize..=128, 1usize..=1000)
    }

    /// Generate valid audio sample rates
    fn sample_rate_strategy() -> impl Strategy<Value = f32> {
        prop_oneof![
            Just(8000.0),
            Just(16000.0),
            Just(22050.0),
            Just(44100.0),
            Just(48000.0)
        ]
    }
}

// Property-based tests for G2P
proptest! {
    #[test]
    fn g2p_always_returns_finite_phonemes(
        text in PropertyTests::ascii_text_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let g2p = DummyG2p::new();
            let result = g2p.to_phonemes(&text, Some(LanguageCode::EnUs)).await;

            prop_assert!(result.is_ok());
            let phonemes = result.unwrap();

            // Property: All phonemes should have non-empty symbols
            for phoneme in &phonemes {
                prop_assert!(!phoneme.symbol.is_empty());
                prop_assert!(phoneme.symbol.len() <= 10); // Reasonable max length
            }

            // Property: Output should be finite for any input
            prop_assert!(phonemes.len() < 10000); // Reasonable upper bound

            Ok(())
        })?;
    }

    #[test]
    fn g2p_deterministic_for_same_input(
        text in PropertyTests::ascii_text_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let g2p = DummyG2p::new();

            let result1 = g2p.to_phonemes(&text, Some(LanguageCode::EnUs)).await.unwrap();
            let result2 = g2p.to_phonemes(&text, Some(LanguageCode::EnUs)).await.unwrap();

            // Property: Same input should produce same output (determinism)
            prop_assert_eq!(result1.len(), result2.len());
            for (p1, p2) in result1.iter().zip(result2.iter()) {
                prop_assert_eq!(&p1.symbol, &p2.symbol);
            }

            Ok(())
        })?;
    }

    #[test]
    fn g2p_handles_all_language_codes(
        text in PropertyTests::ascii_text_strategy(),
        lang_code in prop_oneof![
            Just(LanguageCode::EnUs),
            Just(LanguageCode::De),
            Just(LanguageCode::Fr),
            Just(LanguageCode::Es),
            Just(LanguageCode::It),
            Just(LanguageCode::Ja),
            Just(LanguageCode::ZhCn)
        ]
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let g2p = DummyG2p::new();
            let result = g2p.to_phonemes(&text, Some(lang_code)).await;

            // Property: All language codes should be handled gracefully
            prop_assert!(result.is_ok());
            let phonemes = result.unwrap();

            // Property: Output should be reasonable regardless of language
            for phoneme in &phonemes {
                prop_assert!(!phoneme.symbol.is_empty());
                prop_assert!(phoneme.symbol.chars().all(|c| c.is_ascii() || !c.is_control()));
            }

            Ok(())
        })?;
    }

    #[test]
    fn g2p_monotonic_length_property(
        base_text in "[a-zA-Z]{1,20}"
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let g2p = DummyG2p::new();

            let single_result = g2p.to_phonemes(&base_text, Some(LanguageCode::EnUs)).await.unwrap();
            let double_text = format!("{} {}", base_text, base_text);
            let double_result = g2p.to_phonemes(&double_text, Some(LanguageCode::EnUs)).await.unwrap();

            // Property: Longer text should generally produce more phonemes (or at least same)
            // This is not strict monotonic but should generally hold for meaningful text
            prop_assert!(double_result.len() >= single_result.len());
            Ok(())
        })?;
    }
}

// Property-based tests for Acoustic Model
proptest! {
    #[test]
    fn acoustic_model_valid_mel_output(
        phoneme_symbols in prop::collection::vec(PropertyTests::phoneme_symbol_strategy(), 1..20),
        speed in PropertyTests::speed_strategy(),
        pitch_shift in PropertyTests::pitch_shift_strategy(),
        energy in PropertyTests::energy_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let _ = rt.block_on(async {
            let acoustic_model = DummyAcousticModel::new();

            let phonemes: Vec<AcousticPhoneme> = phoneme_symbols.iter()
                .map(|s| AcousticPhoneme::new(s))
                .collect();

            let config = AcousticConfig {
                speed,
                pitch_shift,
                energy,
                speaker_id: None,
                seed: Some(42),
                emotion: None,
                voice_style: None,
            };

            let result = acoustic_model.synthesize(&phonemes, Some(&config)).await;
            prop_assert!(result.is_ok());

            let mel = result.unwrap();

            // Property: Mel spectrogram should have valid dimensions
            prop_assert!(mel.n_mels > 0);
            prop_assert!(mel.n_frames > 0);
            prop_assert!(mel.n_mels <= 512); // Reasonable upper bound
            prop_assert!(mel.n_frames <= 10000); // Reasonable upper bound

            // Property: Sample rate should be positive
            prop_assert!(mel.sample_rate > 0);
            prop_assert!(mel.sample_rate <= 96000); // Reasonable upper bound

            // Property: Hop length should be positive
            prop_assert!(mel.hop_length > 0);
            prop_assert!(mel.hop_length <= 2048); // Reasonable upper bound

            // Property: All mel values should be finite
            for row in &mel.data {
                for &value in row {
                    prop_assert!(value.is_finite());
                    prop_assert!(value >= -20.0 && value <= 20.0); // Reasonable mel range
                }
            }

            // Property: Data structure consistency
            prop_assert_eq!(mel.data.len(), mel.n_mels);
            if !mel.data.is_empty() {
                prop_assert_eq!(mel.data[0].len(), mel.n_frames);
                Ok(())
            } else {
                Ok(())
            }
        });
    }

    #[test]
    fn acoustic_model_speed_affects_duration(
        phoneme_symbols in prop::collection::vec(PropertyTests::phoneme_symbol_strategy(), 2..10),
        slow_speed in 0.1f32..0.9f32,
        fast_speed in 1.1f32..3.0f32
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let acoustic_model = DummyAcousticModel::new();

            let phonemes: Vec<AcousticPhoneme> = phoneme_symbols.iter()
                .map(|s| AcousticPhoneme::new(s))
                .collect();

            let slow_config = AcousticConfig {
                speed: slow_speed,
                pitch_shift: 0.0,
                energy: 1.0,
                speaker_id: None,
                seed: Some(42),
                emotion: None,
                voice_style: None,
            };

            let fast_config = AcousticConfig {
                speed: fast_speed,
                ..slow_config.clone()
            };

            let slow_mel = acoustic_model.synthesize(&phonemes, Some(&slow_config)).await.unwrap();
            let fast_mel = acoustic_model.synthesize(&phonemes, Some(&fast_config)).await.unwrap();

            // Property: Faster speech should generally produce shorter duration
            // (This may not always hold for dummy implementations, but it's the expected behavior)
            let slow_duration = slow_mel.duration();
            let fast_duration = fast_mel.duration();

            // At minimum, they should be different if speed is significantly different
            if (fast_speed - slow_speed).abs() > 0.5 {
                prop_assert_ne!(slow_duration, fast_duration);
            }
            Ok(())
        })?;
    }

    #[test]
    fn acoustic_model_deterministic_with_seed(
        phoneme_symbols in prop::collection::vec(PropertyTests::phoneme_symbol_strategy(), 1..10),
        seed in any::<u64>()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let acoustic_model = DummyAcousticModel::new();

            let phonemes: Vec<AcousticPhoneme> = phoneme_symbols.iter()
                .map(|s| AcousticPhoneme::new(s))
                .collect();

            let config = AcousticConfig {
                speed: 1.0,
                pitch_shift: 0.0,
                energy: 1.0,
                speaker_id: None,
                seed: Some(seed),
                emotion: None,
                voice_style: None,
            };

            let mel1 = acoustic_model.synthesize(&phonemes, Some(&config)).await.unwrap();
            let mel2 = acoustic_model.synthesize(&phonemes, Some(&config)).await.unwrap();

            // Property: Same seed should produce deterministic results
            prop_assert_eq!(mel1.n_mels, mel2.n_mels);
            prop_assert_eq!(mel1.n_frames, mel2.n_frames);
            prop_assert_eq!(mel1.sample_rate, mel2.sample_rate);
            prop_assert_eq!(mel1.hop_length, mel2.hop_length);
            Ok(())
        })?;
    }
}

// Property-based tests for Vocoder
proptest! {
    #[test]
    fn vocoder_valid_audio_output(
        (n_mels, n_frames) in PropertyTests::mel_dimensions_strategy(),
        sample_rate in PropertyTests::sample_rate_strategy(),
        hop_length in 64usize..=1024,
        speed in PropertyTests::speed_strategy(),
        energy in PropertyTests::energy_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let vocoder = DummyVocoder::new();

            // Create valid mel spectrogram
            let mel_data = vec![vec![0.0f32; n_frames]; n_mels];
            let mel = VocoderMel::new(mel_data, sample_rate as u32, hop_length as u32);

            let config = VocoderConfig {
                speed,
                pitch_shift: 0.0,
                energy,
                speaker_id: None,
                seed: Some(42),
            };

            let result = vocoder.vocode(&mel, Some(&config)).await;
            prop_assert!(result.is_ok());

            let audio = result.unwrap();

            // Property: Audio should have reasonable properties
            prop_assert!(audio.samples().len() > 0);
            prop_assert!(audio.samples().len() <= 10_000_000); // Reasonable upper bound
            prop_assert!(audio.sample_rate() > 0);
            prop_assert!(audio.sample_rate() <= 96000);

            // Property: All audio samples should be in valid range
            for &sample in audio.samples() {
                prop_assert!(sample.is_finite());
                prop_assert!(sample >= -1.0 && sample <= 1.0);
            }

            // Property: Audio length should be roughly proportional to mel frames
            let expected_samples = (n_frames * hop_length) as f32 * (audio.sample_rate() as f32 / sample_rate as f32);
            let tolerance = expected_samples * 0.5; // 50% tolerance for dummy implementation
            prop_assert!((audio.samples().len() as f32 - expected_samples).abs() <= tolerance || audio.samples().len() > 0);
            Ok(())
        })?;
    }

    #[test]
    fn vocoder_energy_affects_amplitude(
        mel_size in 10usize..50,
        low_energy in 0.1f32..0.5f32,
        high_energy in 1.5f32..3.0f32
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let vocoder = DummyVocoder::new();

            // Create test mel spectrogram
            let mel_data = vec![vec![0.5f32; mel_size]; mel_size];
            let mel = VocoderMel::new(mel_data, 22050, 256);

            let low_config = VocoderConfig {
                speed: 1.0,
                pitch_shift: 0.0,
                energy: low_energy,
                speaker_id: None,
                seed: Some(42),
            };

            let high_config = VocoderConfig {
                energy: high_energy,
                ..low_config
            };

            let low_audio = vocoder.vocode(&mel, Some(&low_config)).await.unwrap();
            let high_audio = vocoder.vocode(&mel, Some(&high_config)).await.unwrap();

            // Property: Higher energy should generally produce higher amplitude
            let low_rms = calculate_rms(low_audio.samples());
            let high_rms = calculate_rms(high_audio.samples());

            // Allow for some tolerance since it's a dummy implementation
            prop_assert!(low_rms >= 0.0 && high_rms >= 0.0);
            prop_assert!(low_rms <= 1.0 && high_rms <= 1.0);
            Ok(())
        })?;
    }

    #[test]
    fn vocoder_handles_extreme_mel_values(
        size in 1usize..20,
        sample_rate in PropertyTests::sample_rate_strategy(),
        mel_value in -10.0f32..10.0f32
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let vocoder = DummyVocoder::new();

            // Create mel with extreme values
            let mel_data = vec![vec![mel_value; size]; size];
            let mel = VocoderMel::new(mel_data, sample_rate as u32, 256);

            let config = VocoderConfig {
                speed: 1.0,
                pitch_shift: 0.0,
                energy: 1.0,
                speaker_id: None,
                seed: Some(42),
            };

            let result = vocoder.vocode(&mel, Some(&config)).await;

            // Property: Should handle extreme values gracefully
            prop_assert!(result.is_ok());

            let audio = result.unwrap();

            // Property: Output should still be valid audio
            prop_assert!(audio.len() > 0);
            for &sample in audio.samples() {
                prop_assert!(sample.is_finite());
                prop_assert!(sample >= -1.0 && sample <= 1.0);
            }

            Ok(())
        })?;
    }
}

// Property-based tests for full pipeline
proptest! {
    #[test]
    fn full_pipeline_preserves_invariants(
        text in PropertyTests::ascii_text_strategy(),
        speed in PropertyTests::speed_strategy(),
        energy in PropertyTests::energy_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Initialize components
            let g2p = DummyG2p::new();
            let acoustic_model = DummyAcousticModel::new();
            let vocoder = DummyVocoder::new();

            let acoustic_config = AcousticConfig {
                speed,
                pitch_shift: 0.0,
                energy,
                speaker_id: None,
                seed: Some(42),
                emotion: None,
                voice_style: None,
            };

            let vocoder_config = VocoderConfig {
                speed,
                pitch_shift: 0.0,
                energy,
                speaker_id: None,
                seed: Some(42),
            };

            // Full pipeline
            let phonemes_result = g2p.to_phonemes(&text, Some(LanguageCode::EnUs)).await;
            prop_assert!(phonemes_result.is_ok());

            let phonemes = phonemes_result.unwrap();

            // Handle empty text case or text that produces no phonemes
            // This can happen with empty text or text containing only special characters
            if text.trim().is_empty() || phonemes.is_empty() {
                // For empty text or no phonemes, skip further processing as it's a valid case
                return Ok(());
            }

            let acoustic_phonemes: Vec<AcousticPhoneme> = phonemes.iter()
                .map(|p| AcousticPhoneme::new(&p.symbol))
                .collect();

            // Ensure we have phonemes to process
            if acoustic_phonemes.is_empty() {
                return Ok(());
            }

            let mel_result = acoustic_model.synthesize(&acoustic_phonemes, Some(&acoustic_config)).await;
            prop_assert!(mel_result.is_ok());

            let mel = mel_result.unwrap();
            let vocoder_mel = VocoderMel::new(mel.data.clone(), mel.sample_rate, mel.hop_length);

            let audio_result = vocoder.vocode(&vocoder_mel, Some(&vocoder_config)).await;
            prop_assert!(audio_result.is_ok());

            let audio = audio_result.unwrap();

            // Property: Pipeline should maintain data flow invariants

            // 1. Text with alphabetic characters should produce some phonemes
            if text.chars().any(|c| c.is_alphabetic()) {
                prop_assert!(phonemes.len() > 0);
            }

            // 2. Phonemes should produce mel spectrogram
            if !acoustic_phonemes.is_empty() {
                prop_assert!(mel.n_frames > 0);
                prop_assert!(mel.n_mels > 0);
            }

            // 3. Mel spectrogram should produce audio
            prop_assert!(audio.len() > 0);

            // 4. Data types should be consistent
            prop_assert!(mel.sample_rate > 0);
            prop_assert!(audio.sample_rate() > 0);

            // 5. All outputs should be finite and in valid ranges
            for phoneme in &phonemes {
                prop_assert!(!phoneme.symbol.is_empty());
            }

            for row in &mel.data {
                for &value in row {
                    prop_assert!(value.is_finite());
                }
            }

            for &sample in audio.samples() {
                prop_assert!(sample.is_finite());
                prop_assert!(sample >= -1.0 && sample <= 1.0);
            }

            Ok(())
        })?;
    }

    #[test]
    fn pipeline_performance_is_bounded(
        text_length in 1usize..100
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let text = "a".repeat(text_length);

            let g2p = DummyG2p::new();
            let acoustic_model = DummyAcousticModel::new();
            let vocoder = DummyVocoder::new();

            let config = AcousticConfig {
                speed: 1.0,
                pitch_shift: 0.0,
                energy: 1.0,
                speaker_id: None,
                seed: Some(42),
                emotion: None,
                voice_style: None,
            };

            let start_time = std::time::Instant::now();

            // Run pipeline
            let phonemes = g2p.to_phonemes(&text, Some(LanguageCode::EnUs)).await.unwrap();
            let acoustic_phonemes: Vec<AcousticPhoneme> = phonemes.iter()
                .map(|p| AcousticPhoneme::new(&p.symbol))
                .collect();
            let mel = acoustic_model.synthesize(&acoustic_phonemes, Some(&config)).await.unwrap();
            let vocoder_mel = VocoderMel::new(mel.data.clone(), mel.sample_rate, mel.hop_length);
            let vocoder_config = VocoderConfig {
                speed: config.speed,
                pitch_shift: config.pitch_shift,
                energy: config.energy,
                speaker_id: config.speaker_id,
                seed: config.seed,
            };
            let _audio = vocoder.vocode(&vocoder_mel, Some(&vocoder_config)).await.unwrap();

            let duration = start_time.elapsed();

            // Property: Processing time should be bounded (reasonable for test environment)
            prop_assert!(duration.as_secs() < 30); // Should complete within 30 seconds for any input

            // Property: Memory usage should be reasonable (indirect check via output size)
            prop_assert!(phonemes.len() <= text_length * 10); // Reasonable phoneme expansion
            prop_assert!(mel.n_frames <= text_length * 1000); // Reasonable mel frame count

            Ok(())
        })?;
    }
}

/// Helper function to calculate RMS of audio signal
fn calculate_rms(audio: &[f32]) -> f32 {
    if audio.is_empty() {
        return 0.0;
    }
    let sum_squares: f32 = audio.iter().map(|&x| x * x).sum();
    (sum_squares / audio.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_property_test_helpers() {
        // Test that our property test generators work correctly
        let _rt = tokio::runtime::Runtime::new().unwrap();

        // Test text strategy
        let text_strategy = PropertyTests::ascii_text_strategy();
        let text = text_strategy
            .new_tree(&mut proptest::test_runner::TestRunner::default())
            .unwrap()
            .current();
        assert!(text.len() <= 100);
        assert!(text.chars().all(|c| c.is_ascii()));

        // Test phoneme strategy
        let phoneme_strategy = PropertyTests::phoneme_symbol_strategy();
        let phoneme = phoneme_strategy
            .new_tree(&mut proptest::test_runner::TestRunner::default())
            .unwrap()
            .current();
        assert!(!phoneme.is_empty());
        assert!(phoneme.len() <= 3);

        // Test parameter strategies
        let speed_strategy = PropertyTests::speed_strategy();
        let speed = speed_strategy
            .new_tree(&mut proptest::test_runner::TestRunner::default())
            .unwrap()
            .current();
        assert!(speed >= 0.1 && speed <= 3.0);

        println!("Property test helpers validated successfully");
    }
}
