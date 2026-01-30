//! # Zero-shot Voice Conversion Example
//!
//! This example demonstrates zero-shot voice conversion - converting a voice
//! to match a target speaker using reference characteristics.
//!
//! ## Features Demonstrated
//! - Reference voice database creation
//! - Zero-shot converter configuration
//! - Voice conversion methods
//! - Performance tracking
//!
//! Note: This is a simplified demonstration. In a real application, you would:
//! - Load pre-trained universal voice models
//! - Use actual high-quality audio recordings
//! - Configure based on your specific requirements

use std::collections::HashMap;
use voirs_conversion::{
    types::{Gender, VoiceCharacteristics},
    zero_shot::{
        AudioSample, PhoneticAnalysis, ProsodicFeatures, QualityScores, ReferenceVoice,
        ReferenceVoiceDatabase, SpeakerEmbedding, VoiceMetadata, ZeroShotConfig, ZeroShotConverter,
        ZeroShotMethod,
    },
    Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Zero-shot Voice Conversion Example ===\n");

    // Step 1: Create reference voice database
    println!("1. Creating reference voice database...");
    let mut database = ReferenceVoiceDatabase::new();

    // Add reference voices (in production, these would be real audio samples)
    let reference_audio_1 = generate_sample_audio(200.0, 16000); // 200 Hz fundamental
    let reference_audio_2 = generate_sample_audio(150.0, 16000); // 150 Hz fundamental
    let reference_audio_3 = generate_sample_audio(250.0, 16000); // 250 Hz fundamental

    // Create reference voice entries
    let voice1 = create_reference_voice("speaker_1", "Speaker 1", reference_audio_1.clone());
    let voice2 = create_reference_voice("speaker_2", "Speaker 2", reference_audio_2.clone());
    let voice3 = create_reference_voice("speaker_3", "Speaker 3", reference_audio_3.clone());

    database.add_voice(voice1)?;
    database.add_voice(voice2)?;
    database.add_voice(voice3)?;

    println!(
        "   Added {} reference voices",
        database.metadata().total_voices
    );

    // Step 2: Configure zero-shot converter
    println!("\n2. Configuring zero-shot converter...");
    let mut config = ZeroShotConfig::default();
    config.conversion_method = ZeroShotMethod::Hybrid;
    config.similarity_threshold = 0.7;
    config.quality_threshold = 0.75;

    let mut converter = ZeroShotConverter::new(config);
    println!("   Converter initialized with hybrid method");
    println!("   Similarity threshold: 0.7");
    println!("   Quality threshold: 0.75");

    // Step 3: Demonstrate zero-shot conversion methods
    println!("\n3. Available conversion methods:");
    let methods = vec![
        ZeroShotMethod::EmbeddingInterpolation,
        ZeroShotMethod::StyleTransfer,
        ZeroShotMethod::NeuralAdaptation,
        ZeroShotMethod::Hybrid,
        ZeroShotMethod::DirectSynthesis,
    ];

    for method in methods {
        println!("   - {:?}", method);
    }

    // Step 4: Add reference voices to converter
    println!("\n4. Adding reference voices to converter...");
    let ref_voice1 =
        create_reference_voice("ref_1", "Reference 1", generate_sample_audio(180.0, 8000));
    converter.add_reference_voice(ref_voice1)?;
    println!("   Reference voice added successfully");

    // Step 5: Prepare source and target characteristics
    println!("\n5. Preparing voice conversion...");
    let source_audio = generate_sample_audio(180.0, 16000);
    let target_characteristics = VoiceCharacteristics::for_gender(Gender::Female);

    println!("   Source audio: {} samples", source_audio.len());
    println!("   Target characteristics: Female voice");

    // Step 6: Perform zero-shot conversion
    println!("\n6. Performing zero-shot conversion...");
    let start = std::time::Instant::now();

    let converted_audio = converter.convert_voice(&source_audio, &target_characteristics, 16000)?;

    let duration = start.elapsed();
    println!("   Conversion completed in {:?}", duration);
    println!("   Output: {} samples", converted_audio.len());

    // Step 7: Configuration examples
    println!("\n7. Configuration examples:");

    println!("   High Quality Configuration:");
    let mut hq_config = ZeroShotConfig::default();
    hq_config.quality_threshold = 0.9;
    hq_config.quality_preservation.min_quality_threshold = 0.85;
    hq_config.quality_preservation.quality_speed_tradeoff = 0.9;
    println!("      Quality threshold: {}", hq_config.quality_threshold);
    println!(
        "      Min quality: {}",
        hq_config.quality_preservation.min_quality_threshold
    );
    println!(
        "      Quality/speed tradeoff: {}",
        hq_config.quality_preservation.quality_speed_tradeoff
    );

    println!("\n   Real-time Configuration:");
    let mut rt_config = ZeroShotConfig::default();
    rt_config.performance_constraints.target_rtf = 0.1;
    rt_config.performance_constraints.max_processing_time = 100.0;
    rt_config.performance_constraints.gpu_acceleration = true;
    println!(
        "      Target RTF: {}",
        rt_config.performance_constraints.target_rtf
    );
    println!(
        "      Max processing time: {}ms",
        rt_config.performance_constraints.max_processing_time
    );
    println!(
        "      GPU acceleration: {}",
        rt_config.performance_constraints.gpu_acceleration
    );

    println!("\n   Balanced Configuration:");
    let mut balanced_config = ZeroShotConfig::default();
    balanced_config.quality_threshold = 0.75;
    balanced_config.similarity_threshold = 0.7;
    balanced_config.adaptation_settings.learning_rate = 0.001;
    balanced_config.adaptation_settings.adaptation_steps = 100;
    println!(
        "      Quality threshold: {}",
        balanced_config.quality_threshold
    );
    println!(
        "      Similarity threshold: {}",
        balanced_config.similarity_threshold
    );
    println!(
        "      Learning rate: {}",
        balanced_config.adaptation_settings.learning_rate
    );
    println!(
        "      Adaptation steps: {}",
        balanced_config.adaptation_settings.adaptation_steps
    );

    // Step 8: Find similar voices in database
    println!("\n8. Finding similar voices in database...");
    let similar_voices = database.find_similar_voices(&target_characteristics, 3)?;

    println!("   Top {} similar voices:", similar_voices.len());
    for (i, voice) in similar_voices.iter().enumerate() {
        println!(
            "   {}. {} (quality: {:.2})",
            i + 1,
            voice.speaker_id,
            voice.quality_scores.overall
        );
    }

    // Step 9: Performance summary
    println!("\n9. Performance Summary:");
    println!("   - Conversion latency: {:?}", duration);
    println!(
        "   - Database size: {} voices",
        database.metadata().total_voices
    );
    println!("   - Sample rate: 16000 Hz");
    println!(
        "   - Audio length: {:.2}s",
        source_audio.len() as f32 / 16000.0
    );

    println!("\n=== Example completed successfully ===");
    println!("\nNote: This is a demonstration of the API structure.");
    println!("For full functionality, you would need to:");
    println!("  1. Load pre-trained universal voice models");
    println!("  2. Use high-quality reference recordings");
    println!("  3. Configure quality assessment metrics");
    println!("  4. Implement custom style extractors if needed");
    println!("\nSee the test files for complete working examples.");

    Ok(())
}

/// Generate sample audio with specific fundamental frequency
fn generate_sample_audio(f0_hz: f32, sample_count: usize) -> Vec<f32> {
    use std::f32::consts::PI;

    (0..sample_count)
        .map(|i| {
            let t = i as f32 / 16000.0; // 16kHz sample rate
                                        // Generate harmonics for more realistic voice
            let fundamental = (2.0 * PI * f0_hz * t).sin() * 0.5;
            let harmonic2 = (2.0 * PI * f0_hz * 2.0 * t).sin() * 0.25;
            let harmonic3 = (2.0 * PI * f0_hz * 3.0 * t).sin() * 0.15;
            let harmonic4 = (2.0 * PI * f0_hz * 4.0 * t).sin() * 0.1;

            fundamental + harmonic2 + harmonic3 + harmonic4
        })
        .collect()
}

/// Create a reference voice entry for the database
fn create_reference_voice(speaker_id: &str, name: &str, audio: Vec<f32>) -> ReferenceVoice {
    let duration = audio.len() as f32 / 16000.0;

    ReferenceVoice {
        speaker_id: speaker_id.to_string(),
        name: name.to_string(),
        audio_samples: vec![AudioSample {
            id: format!("{}_sample_1", speaker_id),
            audio_data: audio,
            sample_rate: 16000,
            duration,
            transcription: None,
            quality_score: 0.9,
            phonetic_content: PhoneticAnalysis {
                phoneme_distribution: HashMap::new(),
                diversity_score: 0.75,
                vowel_consonant_ratio: 0.6,
                prosodic_features: ProsodicFeatures {
                    mean_f0: 200.0,
                    f0_range: (150.0, 300.0),
                    speaking_rate: 4.5,
                    pause_patterns: vec![0.2, 0.3, 0.15],
                    stress_patterns: vec![0.8, 0.6, 0.9],
                },
            },
        }],
        embedding: SpeakerEmbedding {
            data: vec![0.1; 256], // Placeholder embedding
            confidence: 0.9,
        },
        characteristics: VoiceCharacteristics::default(),
        quality_scores: QualityScores {
            overall: 0.9,
            clarity: 0.85,
            naturalness: 0.9,
            consistency: 0.88,
            recording_quality: 0.9,
            prosody_quality: 0.87,
        },
        metadata: VoiceMetadata {
            language: "en".to_string(),
            accent: Some("neutral".to_string()),
            gender: Some("neutral".to_string()),
            age_group: Some("adult".to_string()),
            recording_environment: Some("studio".to_string()),
            tags: vec!["reference".to_string(), "high_quality".to_string()],
            created: std::time::Instant::now(),
            modified: None,
        },
        last_used: None,
    }
}
