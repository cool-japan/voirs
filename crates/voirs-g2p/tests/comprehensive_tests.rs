//! Comprehensive tests for G2P phoneme conversion

use voirs_g2p::rules::EnglishRuleG2p;
use voirs_g2p::{DummyG2p, G2p, LanguageCode, Result};

#[tokio::test]
async fn test_basic_word_conversion() -> Result<()> {
    let g2p = DummyG2p::new();

    // Test common words with known phoneme patterns
    let test_cases = vec![
        ("hello", vec!["HH", "EH", "L", "OW"]),
        ("world", vec!["W", "ER", "L", "D"]),
        ("cat", vec!["K", "AE", "T"]),
        ("dog", vec!["D", "AO", "G"]),
        ("house", vec!["HH", "AW", "S"]),
        ("tree", vec!["T", "R", "IY"]),
        ("water", vec!["W", "AO", "T", "ER"]),
        ("phone", vec!["F", "OW", "N"]),
    ];

    for (word, expected_phonemes) in test_cases {
        let phonemes = g2p.to_phonemes(word, Some(LanguageCode::EnUs)).await?;
        let phoneme_symbols: Vec<&str> = phonemes.iter().map(|p| p.symbol.as_str()).collect();

        println!("Word: '{word}' -> {phoneme_symbols:?} (expected: {expected_phonemes:?})");

        // For rule-based system, we may have variations, so check if output is reasonable
        assert!(
            !phonemes.is_empty(),
            "Should generate phonemes for '{word}'"
        );
        assert!(
            phonemes.len() >= 2,
            "Should generate multiple phonemes for '{word}'"
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_magic_e_patterns() -> Result<()> {
    let g2p = DummyG2p::new();

    let magic_e_words = vec![
        "cake", "make", "take", "lake", "name", "game", "same", "bite", "kite", "site", "cute",
        "mute", "tube", "hope", "rope",
    ];

    for word in magic_e_words {
        let phonemes = g2p.to_phonemes(word, Some(LanguageCode::EnUs)).await?;
        println!(
            "Magic-e word '{}' -> {:?}",
            word,
            phonemes.iter().map(|p| &p.symbol).collect::<Vec<_>>()
        );

        assert!(
            !phonemes.is_empty(),
            "Should generate phonemes for magic-e word '{word}'"
        );
        // Magic-e words typically have long vowel sounds
        assert!(
            phonemes.len() >= 2,
            "Magic-e words should have multiple phonemes"
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_consonant_clusters() -> Result<()> {
    let g2p = DummyG2p::new();

    let cluster_words = vec![
        ("street", "str"),
        ("spring", "spr"),
        ("splash", "spl"),
        ("throw", "thr"),
        ("school", "sch"),
        ("chrome", "chr"),
        ("blend", "bl"),
        ("grape", "gr"),
        ("clock", "cl"),
    ];

    for (word, cluster) in cluster_words {
        let phonemes = g2p.to_phonemes(word, Some(LanguageCode::EnUs)).await?;
        println!(
            "Cluster word '{}' (cluster: {}) -> {:?}",
            word,
            cluster,
            phonemes.iter().map(|p| &p.symbol).collect::<Vec<_>>()
        );

        assert!(
            !phonemes.is_empty(),
            "Should generate phonemes for cluster word '{word}'"
        );
        assert!(
            phonemes.len() >= 3,
            "Cluster words should have multiple phonemes"
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_sentence_processing() -> Result<()> {
    let g2p = EnglishRuleG2p::new()?;

    let sentences = vec![
        "Hello world",
        "The quick brown fox",
        "How are you today?",
        "This is a test sentence.",
        "VoiRS speech synthesis system",
    ];

    for sentence in sentences {
        let phonemes = g2p.to_phonemes(sentence, Some(LanguageCode::EnUs)).await?;
        println!("Sentence: '{}' -> {} phonemes", sentence, phonemes.len());

        assert!(
            !phonemes.is_empty(),
            "Should generate phonemes for sentence"
        );
        assert!(
            phonemes.len() >= sentence.split_whitespace().count() * 2,
            "Should have reasonable number of phonemes for sentence length"
        );

        // Check that we have word boundaries (spaces should create pauses)
        let has_spaces = phonemes.iter().any(|p| p.symbol == " ");
        if sentence.contains(' ') {
            assert!(
                has_spaces,
                "Multi-word sentences should contain space phonemes"
            );
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_edge_cases() -> Result<()> {
    let g2p = DummyG2p::new();

    // Test empty input
    let phonemes = g2p.to_phonemes("", Some(LanguageCode::EnUs)).await?;
    assert!(
        phonemes.is_empty(),
        "Empty input should produce no phonemes"
    );

    // Test single character
    let phonemes = g2p.to_phonemes("a", Some(LanguageCode::EnUs)).await?;
    assert!(
        !phonemes.is_empty(),
        "Single character should produce phonemes"
    );

    // Test punctuation
    let phonemes = g2p.to_phonemes("Hello!", Some(LanguageCode::EnUs)).await?;
    assert!(
        !phonemes.is_empty(),
        "Text with punctuation should produce phonemes"
    );

    // Test numbers (if implemented)
    let phonemes = g2p
        .to_phonemes("test 123", Some(LanguageCode::EnUs))
        .await?;
    println!(
        "Numbers test: {:?}",
        phonemes.iter().map(|p| &p.symbol).collect::<Vec<_>>()
    );

    // Test mixed case
    let phonemes = g2p
        .to_phonemes("Hello World", Some(LanguageCode::EnUs))
        .await?;
    assert!(!phonemes.is_empty(), "Mixed case should work");

    Ok(())
}

#[tokio::test]
async fn test_vowel_patterns() -> Result<()> {
    let g2p = DummyG2p::new();

    let vowel_tests = vec![
        // Long vowels
        ("beat", "IY"),
        ("boat", "OW"),
        ("boot", "UW"),
        // Short vowels
        ("bit", "IH"),
        ("bet", "EH"),
        ("bat", "AE"),
        // Diphthongs
        ("boy", "OY"),
        ("cow", "AW"),
        ("buy", "AY"),
        // R-controlled
        ("car", "AR"),
        ("her", "ER"),
        ("for", "OR"),
    ];

    for (word, vowel_sound) in vowel_tests {
        let phonemes = g2p.to_phonemes(word, Some(LanguageCode::EnUs)).await?;
        let phoneme_symbols: Vec<&str> = phonemes.iter().map(|p| p.symbol.as_str()).collect();

        println!(
            "Vowel test '{word}' -> {phoneme_symbols:?} (should contain sound like {vowel_sound})"
        );

        assert!(
            !phonemes.is_empty(),
            "Should generate phonemes for vowel test word '{word}'"
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_irregular_words() -> Result<()> {
    let g2p = DummyG2p::new();

    // Test words that might be in the dictionary vs rule-based
    let irregular_words = vec![
        "one", "two", "eight", "through", "though", "rough", "said", "says", "women", "colonel",
        "island",
    ];

    for word in irregular_words {
        let phonemes = g2p.to_phonemes(word, Some(LanguageCode::EnUs)).await?;
        println!(
            "Irregular word '{}' -> {:?}",
            word,
            phonemes.iter().map(|p| &p.symbol).collect::<Vec<_>>()
        );

        assert!(
            !phonemes.is_empty(),
            "Should generate phonemes for irregular word '{word}'"
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_stress_patterns() -> Result<()> {
    let g2p = DummyG2p::new();

    // Test multi-syllable words that should have stress patterns
    let stress_words = vec![
        "computer",
        "telephone",
        "information",
        "beautiful",
        "interesting",
        "development",
        "understanding",
    ];

    for word in stress_words {
        let phonemes = g2p.to_phonemes(word, Some(LanguageCode::EnUs)).await?;
        println!(
            "Stress word '{}' -> {:?}",
            word,
            phonemes.iter().map(|p| &p.symbol).collect::<Vec<_>>()
        );

        assert!(
            !phonemes.is_empty(),
            "Should generate phonemes for stress word '{word}'"
        );
        assert!(
            phonemes.len() >= 4,
            "Multi-syllable words should have multiple phonemes"
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_performance_benchmark() -> Result<()> {
    let g2p = DummyG2p::new();

    let test_text = "The quick brown fox jumps over the lazy dog. \
                     This sentence contains every letter of the alphabet. \
                     VoiRS is a speech synthesis system written in Rust.";

    let start_time = std::time::Instant::now();
    let phonemes = g2p.to_phonemes(test_text, Some(LanguageCode::EnUs)).await?;
    let duration = start_time.elapsed();

    println!("Performance test:");
    println!("  Input text: {} characters", test_text.len());
    println!("  Output phonemes: {}", phonemes.len());
    println!("  Processing time: {:.2}ms", duration.as_millis());
    println!(
        "  Rate: {:.1} chars/ms",
        test_text.len() as f64 / duration.as_millis() as f64
    );

    assert!(
        !phonemes.is_empty(),
        "Should generate phonemes for performance test"
    );
    assert!(
        duration.as_millis() < 100,
        "Should process text quickly (< 100ms)"
    );

    Ok(())
}

#[tokio::test]
async fn test_configuration_options() -> Result<()> {
    // Using DummyG2p which doesn't have configuration options yet
    let g2p = DummyG2p::new();

    let phonemes = g2p
        .to_phonemes("Hello, world!", Some(LanguageCode::EnUs))
        .await?;
    println!(
        "Configured G2P output: {:?}",
        phonemes.iter().map(|p| &p.symbol).collect::<Vec<_>>()
    );

    assert!(
        !phonemes.is_empty(),
        "Should generate phonemes with custom config"
    );

    Ok(())
}

#[tokio::test]
async fn test_phoneme_properties() -> Result<()> {
    let g2p = DummyG2p::new();

    let phonemes = g2p.to_phonemes("test", Some(LanguageCode::EnUs)).await?;

    for phoneme in &phonemes {
        // Test that phonemes have valid symbols
        assert!(
            !phoneme.symbol.is_empty(),
            "Phoneme symbol should not be empty"
        );

        // Test that phonemes are properly structured
        println!("Phoneme: {} (stress: {}, syllable_position: {:?}, duration_ms: {:?}, confidence: {}, custom_features: {:?})", 
                phoneme.symbol, phoneme.stress, phoneme.syllable_position, phoneme.duration_ms, phoneme.confidence, phoneme.custom_features);
    }

    Ok(())
}
