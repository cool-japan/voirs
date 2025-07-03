//! End-to-end integration tests for G2P conversion

use voirs_g2p::{
    G2p, G2pConverter, LanguageCode, DummyG2p, Result,
    rules::EnglishRuleG2p,
    backends::{
        rule_based::RuleBasedG2p,
        hybrid::HybridG2p,
        registry::BackendRegistry,
    },
    config::ConfigManager,
    preprocessing::{TextPreprocessor, PreprocessingConfig},
    detection::{LanguageDetector, DetectionMethod},
};

#[tokio::test]
async fn test_complete_text_pipeline() -> Result<()> {
    // Test the complete pipeline from raw text to phonemes
    let g2p = EnglishRuleG2p::new()?;
    
    let test_texts = vec![
        "Hello, world! This is a test.",
        "Dr. Smith lives at 123 Main St.",
        "The price is $19.99 USD.",
        "Call me at 3:30 PM today.",
        "Visit https://example.com for more info.",
    ];
    
    for text in test_texts {
        let phonemes = g2p.to_phonemes(text, Some(LanguageCode::EnUs)).await?;
        
        println!("Text: '{}' -> {} phonemes", text, phonemes.len());
        for phoneme in &phonemes {
            print!("{} ", phoneme.symbol);
        }
        println!();
        
        assert!(!phonemes.is_empty(), "Should generate phonemes for: '{}'", text);
        
        // Verify phonemes are well-formed
        for phoneme in &phonemes {
            assert!(!phoneme.symbol.is_empty(), "Phoneme symbol should not be empty");
            assert!(phoneme.confidence >= 0.0 && phoneme.confidence <= 1.0, 
                   "Confidence should be between 0.0 and 1.0");
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_multi_backend_fallback() -> Result<()> {
    // Test multi-backend system with fallback behavior
    let mut hybrid = HybridG2p::new()?;
    
    // Add backends with different priorities
    let rule_based = Box::new(RuleBasedG2p::new()?);
    let dummy = Box::new(DummyG2p::new());
    
    hybrid.add_backend("rule_based".to_string(), rule_based, 1.0)?;
    hybrid.add_backend("dummy".to_string(), dummy, 0.5)?;
    
    let test_text = "Hello world";
    let phonemes = hybrid.to_phonemes(test_text, Some(LanguageCode::EnUs)).await?;
    
    assert!(!phonemes.is_empty(), "Hybrid backend should generate phonemes");
    println!("Hybrid result: {} phonemes", phonemes.len());
    
    Ok(())
}

#[tokio::test]
async fn test_language_detection_integration() -> Result<()> {
    // Test integrated language detection with G2P conversion
    let detector = LanguageDetector::new(DetectionMethod::Mixed);
    let g2p = RuleBasedG2p::new()?;
    
    let texts = vec![
        ("Hello, how are you?", LanguageCode::EnUs),
        ("Hallo, wie geht es dir?", LanguageCode::De),
        ("Bonjour, comment allez-vous?", LanguageCode::Fr),
        ("Hola, ¿cómo estás?", LanguageCode::Es),
    ];
    
    for (text, expected_lang) in texts {
        // Detect language
        let detected_langs = detector.detect(text);
        println!("Text: '{}' -> detected: {:?}", text, detected_langs);
        
        // Use detected language for G2P
        let lang = if detected_langs.is_empty() {
            Some(LanguageCode::EnUs) // fallback
        } else {
            Some(detected_langs[0].0)
        };
        
        let phonemes = g2p.to_phonemes(text, lang).await?;
        assert!(!phonemes.is_empty(), "Should generate phonemes for detected language");
        
        println!("  -> {} phonemes generated", phonemes.len());
    }
    
    Ok(())
}

#[tokio::test]
async fn test_configuration_integration() -> Result<()> {
    // Test configuration system integration
    let mut config_manager = ConfigManager::new();
    
    // Create test configuration
    let config_content = r#"
[preprocessing]
normalize_unicode = true
expand_numbers = true
expand_abbreviations = true

[backends.rule_based]
enabled = true
priority = 1.0

[backends.dummy]
enabled = true
priority = 0.5

[detection]
method = "Mixed"
confidence_threshold = 0.7
"#;
    
    // Write config to temp file
    let temp_file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(temp_file.path(), config_content).unwrap();
    
    // Load configuration
    config_manager.load_from_file(temp_file.path().to_str().unwrap())?;
    let config = config_manager.get_config();
    
    // Test configuration values
    assert!(config.preprocessing.normalize_unicode);
    assert!(config.preprocessing.expand_numbers);
    assert!(config.preprocessing.expand_abbreviations);
    
    // Test preprocessing with configuration
    let preprocessor = TextPreprocessor::new(config.preprocessing.clone());
    let result = preprocessor.process("Dr. Smith has $5.99", Some(LanguageCode::EnUs)).await?;
    
    println!("Preprocessed: '{}'", result);
    assert!(result.contains("Doctor"), "Should expand abbreviations");
    assert!(result.contains("dollar"), "Should expand currency");
    
    Ok(())
}

#[tokio::test]
async fn test_g2p_converter_integration() -> Result<()> {
    // Test the main G2pConverter with multiple backends
    let mut converter = G2pConverter::new();
    
    // Add backends for different languages
    converter.add_backend(LanguageCode::EnUs, Box::new(EnglishRuleG2p::new()?))?;
    converter.add_backend(LanguageCode::De, Box::new(RuleBasedG2p::new()?))?;
    
    // Test conversion for different languages
    let test_cases = vec![
        ("Hello world", LanguageCode::EnUs),
        ("Hallo Welt", LanguageCode::De),
    ];
    
    for (text, lang) in test_cases {
        let phonemes = converter.to_phonemes(text, Some(lang)).await?;
        assert!(!phonemes.is_empty(), "Should convert '{}' in {:?}", text, lang);
        
        println!("'{}' ({:?}) -> {} phonemes", text, lang, phonemes.len());
    }
    
    Ok(())
}

#[tokio::test]
async fn test_backend_registry_integration() -> Result<()> {
    // Test backend registry system
    let mut registry = BackendRegistry::new();
    
    // Register backends with different priorities
    registry.register("english", Box::new(EnglishRuleG2p::new()?), 1.0, vec![LanguageCode::EnUs])?;
    registry.register("rule_based", Box::new(RuleBasedG2p::new()?), 0.8, vec![LanguageCode::De, LanguageCode::Fr])?;
    registry.register("dummy", Box::new(DummyG2p::new()), 0.1, vec![LanguageCode::EnUs, LanguageCode::De])?;
    
    // Test backend selection
    let backend = registry.get_backend_for_language(LanguageCode::EnUs);
    assert!(backend.is_some(), "Should find backend for English");
    
    let backend = registry.get_backend_for_language(LanguageCode::De);
    assert!(backend.is_some(), "Should find backend for German");
    
    // Test conversion through registry
    let phonemes = registry.to_phonemes("Hello", Some(LanguageCode::EnUs)).await?;
    assert!(!phonemes.is_empty(), "Registry should convert text");
    
    println!("Registry conversion: {} phonemes", phonemes.len());
    
    Ok(())
}

#[tokio::test]
async fn test_error_handling_integration() -> Result<()> {
    // Test error handling across the pipeline
    let g2p = DummyG2p::new();
    
    // Test empty input
    let result = g2p.to_phonemes("", Some(LanguageCode::EnUs)).await;
    assert!(result.is_ok(), "Empty input should not error");
    
    // Test very long input
    let long_text = "word ".repeat(1000);
    let result = g2p.to_phonemes(&long_text, Some(LanguageCode::EnUs)).await;
    assert!(result.is_ok(), "Long input should not error");
    
    // Test special characters
    let special_text = "Hello! @#$%^&*()_+ 世界 🌍";
    let result = g2p.to_phonemes(special_text, Some(LanguageCode::EnUs)).await;
    assert!(result.is_ok(), "Special characters should not error");
    
    println!("Error handling tests passed");
    
    Ok(())
}

#[tokio::test]
async fn test_performance_integration() -> Result<()> {
    // Test performance characteristics
    let g2p = EnglishRuleG2p::new()?;
    
    let test_text = "The quick brown fox jumps over the lazy dog. ".repeat(10);
    
    let start = std::time::Instant::now();
    let phonemes = g2p.to_phonemes(&test_text, Some(LanguageCode::EnUs)).await?;
    let duration = start.elapsed();
    
    println!("Performance test:");
    println!("  Input: {} characters", test_text.len());
    println!("  Output: {} phonemes", phonemes.len());
    println!("  Duration: {:.2}ms", duration.as_millis());
    println!("  Rate: {:.1} chars/ms", test_text.len() as f64 / duration.as_millis() as f64);
    
    // Performance requirements (relaxed for testing)
    assert!(duration.as_millis() < 1000, "Should process text within 1 second");
    assert!(!phonemes.is_empty(), "Should generate phonemes");
    
    Ok(())
}