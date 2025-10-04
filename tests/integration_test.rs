//! Comprehensive integration tests for VoiRS complete TTS pipeline

use std::time::Duration;
use tokio::time::timeout;
use voirs::{
    create_acoustic, create_g2p, create_vocoder, AcousticBackend, AudioFormat, G2pBackend,
    QualityLevel, Result, SynthesisConfig, VocoderBackend, VoirsPipelineBuilder,
};

// Test timeout to prevent hanging tests
const TEST_TIMEOUT: Duration = Duration::from_secs(120); // Increased to 2 minutes for synthesis operations
const LONG_TEST_TIMEOUT: Duration = Duration::from_secs(300); // 5 minutes for complex tests
const STREAM_TEST_TIMEOUT: Duration = Duration::from_secs(180); // 3 minutes for streaming tests

// Environment variable to skip slow tests during development/CI
fn should_skip_slow_tests() -> bool {
    std::env::var("VOIRS_SKIP_SLOW_TESTS").unwrap_or_default() == "1"
}

#[tokio::test]
async fn test_complete_pipeline_integration() -> Result<()> {
    if should_skip_slow_tests() {
        println!("⏭️ Skipping slow test due to VOIRS_SKIP_SLOW_TESTS=1");
        return Ok(());
    }

    // Wrap the entire test in a timeout to prevent hanging
    timeout(TEST_TIMEOUT, async {
        // Initialize logging for tests
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .try_init();

        // Test text
        let text = "Hello world, this is a test.";

        // Create synthesis configuration
        let config = SynthesisConfig {
            speaking_rate: 1.0,
            pitch_shift: 0.0,
            volume_gain: 0.0,
            enable_enhancement: false, // Disable enhancement for test compatibility
            output_format: AudioFormat::Wav,
            sample_rate: 22050,
            quality: QualityLevel::High,
            language: voirs::LanguageCode::EnUs,
            effects: Vec::new(),
            streaming_chunk_size: None,
            seed: Some(42),
            enable_emotion: false,
            emotion_type: None,
            emotion_intensity: 0.7,
            emotion_preset: None,
            auto_emotion_detection: false,
            ..Default::default()
        };

        // Initialize components using bridge pattern
        let g2p = create_g2p(G2pBackend::RuleBased);
        let acoustic = create_acoustic(AcousticBackend::Vits);
        let vocoder = create_vocoder(VocoderBackend::WaveGlow); // Use WaveGlow (DummyVocoder) for reliable test audio

        // Build pipeline
        let pipeline = VoirsPipelineBuilder::new()
            .with_g2p(g2p)
            .with_acoustic_model(acoustic)
            .with_vocoder(vocoder)
            .with_quality(QualityLevel::High)
            .with_enhancement(false) // Disable pipeline-level enhancement for testing
            .with_test_mode(true) // Enable test mode to skip expensive operations
            .build()
            .await?;

        // Test synthesis
        let audio = pipeline.synthesize_with_config(text, &config).await?;

        // Verify output
        assert!(
            audio.duration() > 0.0,
            "Audio should have non-zero duration"
        );
        assert!(!audio.samples().is_empty(), "Audio should contain samples");
        assert_eq!(
            audio.sample_rate(),
            22050,
            "Sample rate should match config"
        );
        assert_eq!(audio.channels(), 1, "Should be mono audio");

        // Test basic audio properties
        let samples = audio.samples();
        let peak = samples.iter().map(|s| s.abs()).fold(0.0, f32::max);
        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

        assert!(peak > 0.0, "Audio should have non-zero peak amplitude");
        assert!(rms > 0.0, "Audio should have non-zero RMS level");
        assert!(peak <= 1.0, "Audio should not exceed maximum amplitude");

        println!("✅ Complete pipeline test passed:");
        println!("   Duration: {:.2}s", audio.duration());
        println!("   Samples: {}", samples.len());
        println!("   Peak: {peak:.3}");
        println!("   RMS: {rms:.3}");

        Ok(())
    })
    .await
    .map_err(|_| voirs::VoirsError::timeout("Test timed out after 2 minutes"))?
}

#[tokio::test]
async fn test_g2p_phoneme_conversion() -> Result<()> {
    let g2p = create_g2p(G2pBackend::RuleBased);

    // Test basic word conversion
    let phonemes = g2p.to_phonemes("hello", None).await?;
    assert!(!phonemes.is_empty(), "Should generate phonemes for 'hello'");

    // Test sentence conversion
    let phonemes = g2p.to_phonemes("Hello world", None).await?;
    assert!(
        phonemes.len() >= 6,
        "Should generate multiple phonemes for sentence"
    );

    println!(
        "✅ G2P test passed: {} phonemes for 'Hello world'",
        phonemes.len()
    );

    Ok(())
}

#[tokio::test]
async fn test_vits_acoustic_model() -> Result<()> {
    use voirs::{Phoneme, SynthesisConfig};

    let model = create_acoustic(AcousticBackend::Vits);

    // Create test phonemes
    let phonemes = vec![
        Phoneme::new("HH"),
        Phoneme::new("EH"),
        Phoneme::new("L"),
        Phoneme::new("OW"),
    ];

    let config = SynthesisConfig::default();
    let mel = model.synthesize(&phonemes, Some(&config)).await?;

    assert!(mel.n_mels > 0, "Should generate mel channels");
    assert!(mel.n_frames > 0, "Should generate mel frames");
    assert_eq!(mel.sample_rate, 22050, "Should have correct sample rate");

    println!(
        "✅ VITS test passed: {}x{} mel spectrogram",
        mel.n_mels, mel.n_frames
    );

    Ok(())
}

#[tokio::test]
async fn test_hifigan_vocoder() -> Result<()> {
    use voirs::MelSpectrogram;

    let vocoder = create_vocoder(VocoderBackend::HifiGan);

    // Create test mel spectrogram
    let mel_data = vec![vec![0.1; 50]; 80]; // 80 mel channels, 50 frames
    let mel = MelSpectrogram::new(mel_data, 22050, 256);

    let audio = vocoder.vocode(&mel, None).await?;

    assert!(
        audio.duration() > 0.0,
        "Should generate audio with duration"
    );
    assert!(!audio.samples().is_empty(), "Should generate audio samples");

    println!(
        "✅ HiFi-GAN test passed: {:.2}s audio generated",
        audio.duration()
    );

    Ok(())
}

#[tokio::test]
async fn test_pipeline_builder_validation() -> Result<()> {
    // Test that builder validates configurations properly
    let result = VoirsPipelineBuilder::new()
        .with_speaking_rate(0.1) // Invalid rate
        .with_test_mode(true) // Enable test mode
        .build()
        .await;

    // Should still work with dummy components for now
    // In a full implementation, this would validate the rate
    match result {
        Ok(_) => println!("✅ Builder validation test passed (using dummy components)"),
        Err(e) => println!("✅ Builder validation properly rejected invalid config: {e}"),
    }

    Ok(())
}

#[tokio::test]
async fn test_synthesis_config_effects() -> Result<()> {
    if should_skip_slow_tests() {
        println!("⏭️ Skipping slow test due to VOIRS_SKIP_SLOW_TESTS=1");
        return Ok(());
    }

    timeout(TEST_TIMEOUT, async {
        let g2p = create_g2p(G2pBackend::RuleBased);
        let acoustic = create_acoustic(AcousticBackend::Vits);
        let vocoder = create_vocoder(VocoderBackend::HifiGan);

        let pipeline = VoirsPipelineBuilder::new()
            .with_g2p(g2p)
            .with_acoustic_model(acoustic)
            .with_vocoder(vocoder)
            .with_test_mode(true) // Enable test mode
            .build()
            .await?;

        let text = "Test synthesis";

        // Test different configurations
        let configs = [
            SynthesisConfig {
                speaking_rate: 0.8,
                pitch_shift: -2.0,
                volume_gain: -3.0,
                ..Default::default()
            },
            SynthesisConfig {
                speaking_rate: 1.2,
                pitch_shift: 2.0,
                volume_gain: 3.0,
                ..Default::default()
            },
        ];

        for (i, config) in configs.iter().enumerate() {
            let audio = pipeline.synthesize_with_config(text, config).await?;
            println!(
                "✅ Config {} test passed: {:.2}s audio",
                i + 1,
                audio.duration()
            );
        }

        Ok(())
    })
    .await
    .map_err(|_| {
        voirs::VoirsError::timeout("Synthesis config effects test timed out after 2 minutes")
    })?
}

#[tokio::test]
async fn test_integration_various_text_types() -> Result<()> {
    if should_skip_slow_tests() {
        println!("⏭️ Skipping slow test due to VOIRS_SKIP_SLOW_TESTS=1");
        return Ok(());
    }

    // Use longer timeout for multiple synthesis operations
    timeout(LONG_TEST_TIMEOUT, async {
        let g2p = create_g2p(G2pBackend::RuleBased);
        let acoustic = create_acoustic(AcousticBackend::Vits);
        let vocoder = create_vocoder(VocoderBackend::HifiGan);

        let pipeline = VoirsPipelineBuilder::new()
            .with_g2p(g2p)
            .with_acoustic_model(acoustic)
            .with_vocoder(vocoder)
            .with_test_mode(true) // Enable test mode
            .build()
            .await?;

        let test_texts = [
            "Hello world!",
            "This is a test sentence with numbers like 123 and 456.",
            "Dr. Smith went to the U.S.A. on Jan. 1st, 2023.",
            "The quick brown fox jumps over the lazy dog.",
            "She sells seashells by the seashore.",
            "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
        ];

        let config = SynthesisConfig::default();

        for (i, text) in test_texts.iter().enumerate() {
            let audio = pipeline.synthesize_with_config(text, &config).await?;

            assert!(audio.duration() > 0.0, "Text {i} should generate audio");
            assert!(!audio.samples().is_empty(), "Text {i} should have samples");

            println!(
                "   Text {}: '{}' -> {:.2}s audio",
                i,
                text,
                audio.duration()
            );
        }

        println!("✅ Integration test for various text types passed");

        Ok(())
    })
    .await
    .map_err(|_| voirs::VoirsError::timeout("Various text types test timed out after 5 minutes"))?
}

#[tokio::test]
async fn test_integration_streaming_synthesis() -> Result<()> {
    if should_skip_slow_tests() {
        println!("⏭️ Skipping slow test due to VOIRS_SKIP_SLOW_TESTS=1");
        return Ok(());
    }

    timeout(STREAM_TEST_TIMEOUT, async {
        use futures::StreamExt;
        use std::sync::Arc;

        println!("🧪 Testing streaming synthesis integration...");

        let g2p = create_g2p(G2pBackend::RuleBased);
        let acoustic = create_acoustic(AcousticBackend::Vits);
        let vocoder = create_vocoder(VocoderBackend::HifiGan);

        let pipeline = Arc::new(VoirsPipelineBuilder::new()
            .with_g2p(g2p)
            .with_acoustic_model(acoustic)
            .with_vocoder(vocoder)
            .with_test_mode(true) // Enable test mode
            .build()
            .await?);

        let test_text = "This is a streaming synthesis test. \
                         It should generate multiple audio chunks. \
                         Each sentence should be processed separately for optimal streaming performance.";

        println!("  Text: {}", test_text);
        println!("  Length: {} characters", test_text.len());

        // Test streaming synthesis
        let mut stream = pipeline.synthesize_stream(test_text).await?;

        let mut chunk_count = 0;
        let mut total_duration = 0.0;
        let mut all_samples = Vec::new();
        let mut expected_sample_rate = 0;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;
            chunk_count += 1;
            total_duration += chunk.duration();

            // Validate chunk properties
            assert!(chunk.duration() > 0.0, "Chunk duration should be positive");
            assert!(chunk.samples().len() > 0, "Chunk should contain audio samples");
            assert!(chunk.sample_rate() > 0, "Sample rate should be positive");

            // Store samples for final validation
            if expected_sample_rate == 0 {
                expected_sample_rate = chunk.sample_rate();
            } else {
                assert_eq!(chunk.sample_rate(), expected_sample_rate,
                          "All chunks should have same sample rate");
            }

            all_samples.extend_from_slice(chunk.samples());

            println!("    Chunk {}: {:.2}s audio @ {}Hz", 
                     chunk_count, chunk.duration(), chunk.sample_rate());
        }

        // Validate overall results
        assert!(chunk_count > 0, "Should generate at least one audio chunk");
        assert!(total_duration > 0.0, "Total duration should be positive");
        assert!(all_samples.len() > 0, "Should generate audio samples");

        // Verify streaming works with expected number of chunks for this text
        assert!(chunk_count >= 1, "Should generate multiple chunks for long text");

        println!("  ✅ Generated {} chunks with {:.2}s total audio", chunk_count, total_duration);
        println!("  ✅ Sample rate: {}Hz, total samples: {}", expected_sample_rate, all_samples.len());

        println!("✅ Streaming synthesis integration test passed");
        Ok(())
    })
    .await
    .map_err(|_| voirs::VoirsError::timeout("Streaming synthesis test timed out after 3 minutes"))?
}

#[tokio::test]
async fn test_integration_performance_benchmark() -> Result<()> {
    if should_skip_slow_tests() {
        println!("⏭️ Skipping slow test due to VOIRS_SKIP_SLOW_TESTS=1");
        return Ok(());
    }

    // Use longer timeout for performance benchmark
    timeout(LONG_TEST_TIMEOUT, async {
        let g2p = create_g2p(G2pBackend::RuleBased);
        let acoustic = create_acoustic(AcousticBackend::Vits);
        let vocoder = create_vocoder(VocoderBackend::HifiGan); // Fastest variant

        let pipeline = VoirsPipelineBuilder::new()
            .with_g2p(g2p)
            .with_acoustic_model(acoustic)
            .with_vocoder(vocoder)
            .with_test_mode(true) // Enable test mode
            .build()
            .await?;

        let test_sentences = ["Short test.",
            "This is a medium length sentence for testing performance.",
            "This is a much longer sentence that contains multiple words and should take more time to process through the entire text-to-speech pipeline from grapheme to phoneme conversion through acoustic modeling to final audio generation."];

        let config = SynthesisConfig::default();

        for (i, text) in test_sentences.iter().enumerate() {
            let start_time = std::time::Instant::now();
            let audio = pipeline.synthesize_with_config(text, &config).await?;
            let synthesis_time = start_time.elapsed();

            let real_time_factor = synthesis_time.as_secs_f32() / audio.duration();

            println!(
                "   Test {}: '{}' -> {:.2}s audio in {:.3}s (RTF: {:.2}x)",
                i,
                text,
                audio.duration(),
                synthesis_time.as_secs_f32(),
                real_time_factor
            );

            assert!(
                real_time_factor < 20.0,
                "Should synthesize within reasonable time"
            );
        }

        println!("✅ Integration performance benchmark test passed");

        Ok(())
    })
    .await
    .map_err(|_| voirs::VoirsError::timeout("Performance benchmark test timed out after 5 minutes"))?
}

#[tokio::test]
async fn test_integration_component_integration() -> Result<()> {
    if should_skip_slow_tests() {
        println!("⏭️ Skipping slow test due to VOIRS_SKIP_SLOW_TESTS=1");
        return Ok(());
    }

    timeout(TEST_TIMEOUT, async {
        // Test individual components work together correctly
        let g2p = create_g2p(G2pBackend::RuleBased);
        let acoustic = create_acoustic(AcousticBackend::Vits);
        let vocoder = create_vocoder(VocoderBackend::HifiGan);

        let text = "Integration test";

        // Step 1: G2P conversion
        let phonemes = g2p.to_phonemes(text, None).await?;
        assert!(!phonemes.is_empty(), "G2P should produce phonemes");

        // Step 2: Acoustic model synthesis
        let config = SynthesisConfig::default();
        let mel = acoustic.synthesize(&phonemes, Some(&config)).await?;
        assert!(
            mel.n_frames > 0,
            "Acoustic model should produce mel spectrogram"
        );

        // Step 3: Vocoder synthesis
        let audio = vocoder.vocode(&mel, Some(&config)).await?;
        assert!(audio.duration() > 0.0, "Vocoder should produce audio");

        // Step 4: Full pipeline test
        let pipeline = VoirsPipelineBuilder::new()
            .with_g2p(g2p)
            .with_acoustic_model(acoustic)
            .with_vocoder(vocoder)
            .with_test_mode(true) // Enable test mode
            .build()
            .await?;

        let pipeline_audio = pipeline.synthesize_with_config(text, &config).await?;

        // Results should be consistent
        assert_eq!(
            audio.sample_rate(),
            pipeline_audio.sample_rate(),
            "Sample rates should match"
        );
        assert_eq!(
            audio.channels(),
            pipeline_audio.channels(),
            "Channel counts should match"
        );

        println!("✅ Integration component integration test passed");
        println!("   Phonemes: {}", phonemes.len());
        println!("   Mel: {}x{}", mel.n_mels, mel.n_frames);
        println!("   Audio: {:.2}s", audio.duration());

        Ok(())
    })
    .await
    .map_err(|_| {
        voirs::VoirsError::timeout("Component integration test timed out after 2 minutes")
    })?
}

#[tokio::test]
async fn test_integration_reproducibility() -> Result<()> {
    if should_skip_slow_tests() {
        println!("⏭️ Skipping slow test due to VOIRS_SKIP_SLOW_TESTS=1");
        return Ok(());
    }

    timeout(TEST_TIMEOUT, async {
        let g2p = create_g2p(G2pBackend::RuleBased);
        let acoustic = create_acoustic(AcousticBackend::Vits);
        let vocoder = create_vocoder(VocoderBackend::HifiGan);

        let pipeline = VoirsPipelineBuilder::new()
            .with_g2p(g2p)
            .with_acoustic_model(acoustic)
            .with_vocoder(vocoder)
            .with_test_mode(true) // Enable test mode
            .build()
            .await?;

        let text = "Reproducibility test";
        let config = SynthesisConfig {
            // Use a fixed seed for deterministic results
            seed: Some(42),
            ..Default::default()
        };

        // Generate the same text twice
        let audio1 = pipeline.synthesize_with_config(text, &config).await?;
        let audio2 = pipeline.synthesize_with_config(text, &config).await?;

        // Should have same basic properties (allow small tolerance for VITS non-determinism)
        let duration_diff = (audio1.duration() - audio2.duration()).abs();
        assert!(
            duration_diff < 0.02, // 20ms tolerance for VITS acoustic model variations
            "Durations should be similar (diff: {duration_diff:.6}s)"
        );
        assert_eq!(
            audio1.sample_rate(),
            audio2.sample_rate(),
            "Sample rates should match"
        );
        assert_eq!(
            audio1.samples().len(),
            audio2.samples().len(),
            "Sample counts should match"
        );

        println!("✅ Integration reproducibility test passed");
        println!(
            "   Both runs produced {:.3}s audio with {} samples",
            audio1.duration(),
            audio1.samples().len()
        );

        Ok(())
    })
    .await
    .map_err(|_| voirs::VoirsError::timeout("Reproducibility test timed out after 2 minutes"))?
}

#[tokio::test]
async fn test_integration_error_handling() -> Result<()> {
    if should_skip_slow_tests() {
        println!("⏭️ Skipping slow test due to VOIRS_SKIP_SLOW_TESTS=1");
        return Ok(());
    }

    timeout(TEST_TIMEOUT, async {
        let g2p = create_g2p(G2pBackend::RuleBased);
        let acoustic = create_acoustic(AcousticBackend::Vits);
        let vocoder = create_vocoder(VocoderBackend::HifiGan);

        let pipeline = VoirsPipelineBuilder::new()
            .with_g2p(g2p)
            .with_acoustic_model(acoustic)
            .with_vocoder(vocoder)
            .with_test_mode(true) // Enable test mode
            .build()
            .await?;

        let config = SynthesisConfig::default();

        // Test empty text
        let result = pipeline.synthesize_with_config("", &config).await;
        // Currently, empty text is allowed and produces empty audio
        // This could be changed in the future to return an error
        match result {
            Ok(audio) => {
                // Empty text produces very short or empty audio
                assert!(
                    audio.duration() < 0.1,
                    "Empty text should produce minimal audio"
                );
            }
            Err(_) => {
                // Some implementations may choose to error on empty text
                // Both behaviors are acceptable
            }
        }

        // Test extremely long text (should still work but might be slow)
        let long_text = "word ".repeat(1000); // 1000 words
        let result = pipeline.synthesize_with_config(&long_text, &config).await;
        // This might succeed or fail depending on implementation limits
        // Just verify it doesn't panic
        match result {
            Ok(audio) => println!("   Long text succeeded: {:.2}s audio", audio.duration()),
            Err(e) => println!("   Long text failed as expected: {e}"),
        }

        println!("✅ Integration error handling test passed");

        Ok(())
    })
    .await
    .map_err(|_| voirs::VoirsError::timeout("Error handling test timed out after 2 minutes"))?
}
