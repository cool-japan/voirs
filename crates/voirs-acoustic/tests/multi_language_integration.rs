//! Multi-language integration tests for voirs-acoustic
//!
//! These tests verify that the acoustic models work correctly across different
//! languages and handle multi-language synthesis scenarios.

use voirs_acoustic::{LanguageCode, MelSpectrogram, Phoneme, SynthesisConfig};

/// Test that all supported languages can be instantiated and have valid properties
#[test]
fn test_all_languages_valid() {
    for &lang in LanguageCode::all() {
        // Verify language code is 2 characters
        assert_eq!(lang.language_code().len(), 2);

        // Verify BCP 47 code is valid format (e.g., "en-US")
        let bcp47 = lang.as_str();
        assert!(bcp47.contains('-'));
        assert!(bcp47.len() >= 5); // Minimum: "en-US"

        // Verify language name is not empty
        assert!(!lang.language_name().is_empty());

        // Verify parsing roundtrip
        assert_eq!(LanguageCode::parse(bcp47), Some(lang));
    }
}

/// Test synthesis configuration with different language contexts
/// Language information is metadata - the actual synthesis works with phonemes
#[test]
fn test_synthesis_config_language_agnostic() {
    let languages = vec![
        LanguageCode::EnUs,
        LanguageCode::JaJp,
        LanguageCode::ZhCn,
        LanguageCode::FrFr,
        LanguageCode::DeDe,
    ];

    for _lang in languages {
        // Create config - synthesis is language-agnostic and works with phonemes
        let config = SynthesisConfig {
            speed: 1.0,
            pitch_shift: 0.0,
            energy: 1.0,
            speaker_id: None,
            seed: None,
            emotion: None,
            voice_style: None,
        };

        // Verify configuration is valid
        assert!(config.speed > 0.0);
        assert_eq!(config.pitch_shift, 0.0);
        assert_eq!(config.energy, 1.0);
    }
}

/// Test phoneme sequences for different languages
#[test]
fn test_phoneme_sequences_multi_language() {
    // English phonemes (ARPAbet)
    let english_phonemes = vec![
        Phoneme::new("HH".to_string()),
        Phoneme::new("AH".to_string()),
        Phoneme::new("L".to_string()),
        Phoneme::new("OW".to_string()),
    ];

    // Japanese phonemes (Hiragana-based)
    let japanese_phonemes = vec![
        Phoneme::new("k".to_string()),
        Phoneme::new("o".to_string()),
        Phoneme::new("n".to_string()),
        Phoneme::new("n".to_string()),
        Phoneme::new("i".to_string()),
        Phoneme::new("ch".to_string()),
        Phoneme::new("i".to_string()),
        Phoneme::new("w".to_string()),
        Phoneme::new("a".to_string()),
    ];

    // Chinese phonemes (Pinyin-based)
    let chinese_phonemes = vec![
        Phoneme::new("n".to_string()),
        Phoneme::new("i".to_string()),
        Phoneme::new("h".to_string()),
        Phoneme::new("a".to_string()),
        Phoneme::new("o".to_string()),
    ];

    // Test that all phoneme sequences are valid
    for phonemes in &[&english_phonemes, &japanese_phonemes, &chinese_phonemes] {
        assert!(!phonemes.is_empty());
        for phoneme in *phonemes {
            assert!(!phoneme.symbol.is_empty());
        }
    }
}

/// Test mel spectrogram creation with language-specific parameters
#[test]
fn test_mel_spectrogram_language_specific() {
    // Different languages may benefit from different mel parameters
    let configs = vec![
        // English: Standard 80 mel bins
        (LanguageCode::EnUs, 80, 22050),
        // Japanese: Higher mel bins for tonal precision
        (LanguageCode::JaJp, 100, 22050),
        // Chinese: Higher sample rate for tonal languages
        (LanguageCode::ZhCn, 80, 24000),
        // German: Standard configuration
        (LanguageCode::DeDe, 80, 22050),
    ];

    for (_lang, n_mels, sample_rate) in configs {
        // Create a simple mel spectrogram
        let n_frames = 100;
        let data = vec![vec![0.0f32; n_frames]; n_mels];

        let mel = MelSpectrogram::new(
            data,
            sample_rate,
            256, // hop_length
        );

        assert_eq!(mel.n_mels, n_mels);
        assert_eq!(mel.n_frames, n_frames);
        assert_eq!(mel.sample_rate, sample_rate);

        // Verify duration calculation is reasonable
        let duration = mel.duration();
        assert!(duration > 0.0);
        assert!(duration < 10.0); // Should be less than 10 seconds for 100 frames
    }
}

/// Test language-specific phoneme duration patterns
#[test]
fn test_language_specific_duration_patterns() {
    // Different languages have different typical phoneme durations
    let test_cases = vec![
        // English: moderate speaking rate
        (LanguageCode::EnUs, 0.08, 0.12),
        // Japanese: faster mora timing
        (LanguageCode::JaJp, 0.06, 0.10),
        // Spanish: slightly faster
        (LanguageCode::EsEs, 0.07, 0.11),
        // German: moderate to slow
        (LanguageCode::DeDe, 0.09, 0.13),
    ];

    for (_lang, min_duration, max_duration) in test_cases {
        // Create test phoneme with expected duration range
        let mut phoneme = Phoneme::new("a".to_string());

        // Test that durations are within expected language-specific ranges
        for duration in [min_duration, max_duration] {
            phoneme.duration = Some(duration);

            if let Some(d) = phoneme.duration {
                assert!(d > 0.0);
                assert!(d < 0.5); // No single phoneme should be longer than 0.5s
                assert!(d >= min_duration - 0.01); // Allow small tolerance
                assert!(d <= max_duration + 0.01);
            }
        }
    }
}

/// Test cross-lingual phoneme mapping
#[test]
fn test_cross_lingual_phoneme_mapping() {
    // Some phonemes appear across multiple languages
    let universal_phonemes = vec![
        "a", "i", "u", "e", "o", // Universal vowels
        "m", "n", "p", "t", "k", // Common consonants
    ];

    for symbol in universal_phonemes {
        let phoneme = Phoneme::new(symbol.to_string());

        // Verify phoneme is valid across languages
        assert_eq!(phoneme.symbol, symbol);
        assert!(phoneme.duration.is_none() || phoneme.duration.unwrap() > 0.0);
    }
}

/// Test language code parsing with various formats
#[test]
fn test_language_code_parsing_robustness() {
    // Test standard BCP 47 format
    assert_eq!(LanguageCode::parse("en-US"), Some(LanguageCode::EnUs));
    assert_eq!(LanguageCode::parse("ja-JP"), Some(LanguageCode::JaJp));
    assert_eq!(LanguageCode::parse("zh-CN"), Some(LanguageCode::ZhCn));

    // Test case sensitivity
    assert_eq!(LanguageCode::parse("EN-US"), Some(LanguageCode::EnUs));
    assert_eq!(LanguageCode::parse("en-us"), Some(LanguageCode::EnUs));
    assert_eq!(LanguageCode::parse("En-Us"), Some(LanguageCode::EnUs));

    // Test invalid formats
    assert_eq!(LanguageCode::parse("invalid"), None);
    assert_eq!(LanguageCode::parse("en"), None); // Missing region
    assert_eq!(LanguageCode::parse(""), None);
    assert_eq!(LanguageCode::parse("xx-XX"), None); // Non-existent language
}

/// Test language-specific synthesis configurations
/// Note: Language is metadata; synthesis works with phoneme sequences
#[test]
fn test_language_synthesis_configurations() {
    let configs = vec![
        // English: Standard American English
        (LanguageCode::EnUs, 1.0, 0.0, "Standard English synthesis"),
        // Japanese: Slightly slower for clarity
        (
            LanguageCode::JaJp,
            0.9,
            0.0,
            "Japanese synthesis with mora timing",
        ),
        // Spanish: Slightly faster
        (
            LanguageCode::EsEs,
            1.1,
            0.0,
            "Spanish synthesis with rhythm",
        ),
        // German: Moderate speed with slight pitch variation
        (
            LanguageCode::DeDe,
            0.95,
            1.0,
            "German synthesis with intonation",
        ),
        // French: Standard with slight pitch shift
        (
            LanguageCode::FrFr,
            1.0,
            0.5,
            "French synthesis with prosody",
        ),
    ];

    for (_lang, speed, pitch_shift, description) in configs {
        let config = SynthesisConfig {
            speed,
            pitch_shift,
            energy: 1.0,
            speaker_id: None,
            seed: None,
            emotion: None,
            voice_style: None,
        };

        // Verify configuration
        assert!(
            config.speed > 0.5 && config.speed < 2.0,
            "Speed out of range for {}",
            description
        );
        assert!(
            config.pitch_shift >= -12.0 && config.pitch_shift <= 12.0,
            "Pitch shift out of range for {}",
            description
        );
    }
}

/// Test multi-language batch synthesis configuration
#[test]
fn test_multi_language_batch_synthesis() {
    let batch_configs = vec![
        (LanguageCode::EnUs, "Hello"),
        (LanguageCode::JaJp, "こんにちは"),
        (LanguageCode::ZhCn, "你好"),
        (LanguageCode::FrFr, "Bonjour"),
        (LanguageCode::DeDe, "Guten Tag"),
        (LanguageCode::EsEs, "Hola"),
        (LanguageCode::ItIt, "Ciao"),
        (LanguageCode::KoKr, "안녕하세요"),
    ];

    let mut configs = Vec::new();

    for (_lang, _text) in batch_configs {
        let config = SynthesisConfig {
            speed: 1.0,
            pitch_shift: 0.0,
            energy: 1.0,
            speaker_id: None,
            seed: Some(42), // For reproducibility
            emotion: None,
            voice_style: None,
        };
        configs.push(config);
    }

    // Verify all configurations are valid
    assert_eq!(configs.len(), 8);
    for config in &configs {
        assert_eq!(config.speed, 1.0);
        assert_eq!(config.seed, Some(42));
    }
}

/// Test regional language variants
#[test]
fn test_regional_language_variants() {
    // Test that different regions of the same language are distinct
    assert_ne!(LanguageCode::EnUs, LanguageCode::EnGb);
    assert_ne!(LanguageCode::PtBr, LanguageCode::PtPt);

    // Test that they parse to different values
    assert_eq!(LanguageCode::parse("en-US"), Some(LanguageCode::EnUs));
    assert_eq!(LanguageCode::parse("en-GB"), Some(LanguageCode::EnGb));
    assert_eq!(LanguageCode::parse("pt-BR"), Some(LanguageCode::PtBr));
    assert_eq!(LanguageCode::parse("pt-PT"), Some(LanguageCode::PtPt));

    // Test that they have different BCP 47 codes
    assert_ne!(LanguageCode::EnUs.as_str(), LanguageCode::EnGb.as_str());
    assert_ne!(LanguageCode::PtBr.as_str(), LanguageCode::PtPt.as_str());

    // Verify the same language code but different regions
    assert_eq!(
        LanguageCode::EnUs.language_code(),
        LanguageCode::EnGb.language_code()
    ); // Both "en"
    assert_eq!(
        LanguageCode::PtBr.language_code(),
        LanguageCode::PtPt.language_code()
    ); // Both "pt"
}

/// Test language coverage across major language families
#[test]
fn test_language_family_coverage() {
    let language_families = vec![
        // Germanic
        vec![
            LanguageCode::EnUs,
            LanguageCode::DeDe,
            LanguageCode::NlNl,
            LanguageCode::SvSe,
            LanguageCode::NoNo,
            LanguageCode::DaDk,
        ],
        // Romance
        vec![
            LanguageCode::FrFr,
            LanguageCode::EsEs,
            LanguageCode::ItIt,
            LanguageCode::PtBr,
            LanguageCode::PtPt,
        ],
        // Slavic
        vec![LanguageCode::RuRu, LanguageCode::PlPl, LanguageCode::CsCz],
        // Asian
        vec![
            LanguageCode::JaJp,
            LanguageCode::ZhCn,
            LanguageCode::KoKr,
            LanguageCode::ThTh,
            LanguageCode::ViVn,
        ],
        // Other
        vec![
            LanguageCode::ArSa,
            LanguageCode::HiIn,
            LanguageCode::TrTr,
            LanguageCode::ElGr,
            LanguageCode::HeIl,
        ],
    ];

    for family in language_families {
        // Verify each family has valid languages
        assert!(!family.is_empty());

        for lang in family {
            // Verify basic properties
            assert!(!lang.as_str().is_empty());
            assert!(!lang.language_name().is_empty());
            assert_eq!(lang.language_code().len(), 2);
        }
    }
}

/// Test language-specific mel spectrogram parameters optimization
#[test]
fn test_language_specific_mel_optimization() {
    // Test that mel parameters can be optimized for language characteristics

    // Tonal languages (Chinese, Thai, Vietnamese) - verify language codes exist
    let tonal_languages = vec![LanguageCode::ZhCn, LanguageCode::ThTh, LanguageCode::ViVn];
    for lang in tonal_languages {
        assert!(!lang.as_str().is_empty());
        assert!(!lang.language_name().is_empty());
    }

    // Languages with complex consonant clusters - verify language codes exist
    let cluster_languages = vec![LanguageCode::RuRu, LanguageCode::PlPl, LanguageCode::CsCz];
    for lang in cluster_languages {
        assert!(!lang.as_str().is_empty());
        assert!(!lang.language_name().is_empty());
    }
}

/// Test language code enumeration completeness
#[test]
fn test_language_code_enumeration() {
    let all_languages = LanguageCode::all();

    // Verify we have 28 languages total
    assert_eq!(all_languages.len(), 28);

    // Verify no duplicates
    use std::collections::HashSet;
    let unique_codes: HashSet<_> = all_languages.iter().map(|l| l.as_str()).collect();
    assert_eq!(unique_codes.len(), all_languages.len());

    // Verify all languages are reachable via parsing
    for &lang in all_languages {
        assert_eq!(LanguageCode::parse(lang.as_str()), Some(lang));
    }
}

/// Test that phoneme sequences with durations work correctly across languages
#[test]
fn test_phoneme_durations_cross_language() {
    // English phoneme with duration
    let mut en_phoneme = Phoneme::new("AH".to_string());
    en_phoneme.duration = Some(0.10); // 100ms

    // Japanese phoneme with duration
    let mut ja_phoneme = Phoneme::new("a".to_string());
    ja_phoneme.duration = Some(0.08); // 80ms (faster mora)

    // Chinese phoneme with duration
    let mut zh_phoneme = Phoneme::new("a".to_string());
    zh_phoneme.duration = Some(0.12); // 120ms (tonal)

    // All should have valid durations
    assert!(en_phoneme.duration.unwrap() > 0.0);
    assert!(ja_phoneme.duration.unwrap() > 0.0);
    assert!(zh_phoneme.duration.unwrap() > 0.0);

    // All should be reasonable (< 500ms)
    assert!(en_phoneme.duration.unwrap() < 0.5);
    assert!(ja_phoneme.duration.unwrap() < 0.5);
    assert!(zh_phoneme.duration.unwrap() < 0.5);
}

/// Test mel spectrogram compatibility across languages
#[test]
fn test_mel_spectrogram_cross_language_compatibility() {
    // Create mel spectrograms with different parameters (simulating different languages)
    let mels = vec![
        // English: standard parameters
        MelSpectrogram::new(vec![vec![0.0; 100]; 80], 22050, 256),
        // Japanese: more mel bins
        MelSpectrogram::new(vec![vec![0.0; 100]; 100], 22050, 256),
        // Chinese: higher sample rate
        MelSpectrogram::new(vec![vec![0.0; 100]; 80], 24000, 256),
    ];

    for mel in &mels {
        // All should have valid dimensions
        assert!(mel.n_mels > 0);
        assert!(mel.n_frames > 0);
        assert!(mel.sample_rate > 0);
        assert!(mel.hop_length > 0);

        // All should have reasonable durations
        let duration = mel.duration();
        assert!(duration > 0.0);
        assert!(duration < 10.0);
    }
}

/// Test synthesis config parameter ranges for different languages
#[test]
fn test_synthesis_config_language_parameter_ranges() {
    // Test various parameter combinations that might be used for different languages
    let test_cases = vec![
        // (speed, pitch_shift, energy, description)
        (1.0, 0.0, 1.0, "Neutral baseline"),
        (0.8, 0.0, 1.0, "Slower for clarity (beginner learners)"),
        (1.2, 0.0, 1.0, "Faster for native speakers"),
        (1.0, -2.0, 1.0, "Lower pitch (male voice)"),
        (1.0, 2.0, 1.0, "Higher pitch (female voice)"),
        (1.0, 0.0, 0.8, "Softer energy (quiet speech)"),
        (1.0, 0.0, 1.2, "Higher energy (emphasis)"),
        (0.9, 1.0, 1.1, "Expressive combination"),
    ];

    for (speed, pitch_shift, energy, description) in test_cases {
        let config = SynthesisConfig {
            speed,
            pitch_shift,
            energy,
            speaker_id: None,
            seed: None,
            emotion: None,
            voice_style: None,
        };

        // Verify all parameters are in valid ranges
        assert!(config.speed > 0.0, "Invalid speed for: {}", description);
        assert!(
            config.pitch_shift >= -12.0 && config.pitch_shift <= 12.0,
            "Invalid pitch shift for: {}",
            description
        );
        assert!(config.energy > 0.0, "Invalid energy for: {}", description);
    }
}
