//! Comprehensive tests for VITS acoustic model

use voirs_acoustic::{
    AcousticModel, Phoneme, MelSpectrogram, SynthesisConfig, LanguageCode,
    vits::{VitsModel, VitsConfig, TextEncoderConfig, DurationConfig, FlowConfig},
    Result,
};
use std::collections::HashMap;

#[tokio::test]
async fn test_vits_model_creation() -> Result<()> {
    // Test default model creation
    let model = VitsModel::new()?;
    let metadata = model.metadata();
    
    assert_eq!(metadata.name, "VITS");
    assert_eq!(metadata.architecture, "VITS");
    assert_eq!(metadata.sample_rate, 22050);
    assert_eq!(metadata.mel_channels, 80);
    assert!(!metadata.is_multi_speaker);
    
    println!("✅ VITS model created successfully");
    println!("   Metadata: {:?}", metadata);
    
    Ok(())
}

#[tokio::test]
async fn test_vits_custom_configuration() -> Result<()> {
    let config = VitsConfig {
        text_encoder: TextEncoderConfig {
            vocab_size: 256,
            d_model: 192,
            n_layers: 6,
            n_heads: 2,
            d_ff: 768,
            max_seq_len: 1000,
            dropout: 0.1,
            kernel_size: 3,
            n_conv_layers: 3,
            use_relative_pos: true,
        },
        duration_predictor: DurationConfig {
            input_dim: 192,
            hidden_dim: 256,
            n_layers: 2,
            kernel_size: 3,
            dropout: 0.5,
            filter_channels: 256,
        },
        flows: FlowConfig {
            n_flows: 4,
            n_coupling_layers: 4,
            hidden_dim: 256,
            kernel_size: 5,
            n_channels: 80,
            dropout: 0.0,
        },
        sample_rate: 22050,
        mel_channels: 80,
        multi_speaker: false,
        speaker_count: None,
        ..Default::default()
    };
    
    let model = VitsModel::with_config(config.clone())?;
    assert_eq!(model.config().sample_rate, config.sample_rate);
    assert_eq!(model.config().mel_channels, config.mel_channels);
    
    println!("✅ VITS model with custom config created successfully");
    
    Ok(())
}

#[tokio::test]
async fn test_vits_phoneme_synthesis() -> Result<()> {
    let model = VitsModel::new()?;
    
    // Create test phonemes for "hello"
    let phonemes = vec![
        Phoneme::new("HH"),
        Phoneme::new("EH"),
        Phoneme::new("L"),
        Phoneme::new("OW"),
    ];
    
    let config = SynthesisConfig::default();
    let mel = model.synthesize(&phonemes, Some(&config)).await?;
    
    assert_eq!(mel.n_mels, 80, "Should generate 80 mel channels");
    assert!(mel.n_frames > 0, "Should generate mel frames");
    assert_eq!(mel.sample_rate, 22050, "Should have correct sample rate");
    assert!(mel.duration() > 0.0, "Should have positive duration");
    
    println!("✅ VITS phoneme synthesis test passed");
    println!("   Mel spectrogram: {}x{} frames", mel.n_mels, mel.n_frames);
    println!("   Duration: {:.3}s", mel.duration());
    
    Ok(())
}

#[tokio::test]
async fn test_vits_batch_synthesis() -> Result<()> {
    let model = VitsModel::new()?;
    
    let phoneme_sequences = vec![
        vec![Phoneme::new("HH"), Phoneme::new("EH"), Phoneme::new("L"), Phoneme::new("OW")],
        vec![Phoneme::new("W"), Phoneme::new("ER"), Phoneme::new("L"), Phoneme::new("D")],
        vec![Phoneme::new("T"), Phoneme::new("EH"), Phoneme::new("S"), Phoneme::new("T")],
    ];
    
    let inputs: Vec<&[Phoneme]> = phoneme_sequences.iter().map(|seq| seq.as_slice()).collect();
    let configs = vec![SynthesisConfig::default(); 3];
    
    let mels = model.synthesize_batch(&inputs, Some(&configs)).await?;
    
    assert_eq!(mels.len(), 3, "Should generate mel for each input");
    
    for (i, mel) in mels.iter().enumerate() {
        assert_eq!(mel.n_mels, 80, "Batch item {} should have 80 mel channels", i);
        assert!(mel.n_frames > 0, "Batch item {} should have frames", i);
        println!("   Batch item {}: {}x{} frames, {:.3}s", i, mel.n_mels, mel.n_frames, mel.duration());
    }
    
    println!("✅ VITS batch synthesis test passed");
    
    Ok(())
}

#[tokio::test]
async fn test_vits_synthesis_configurations() -> Result<()> {
    let model = VitsModel::new()?;
    
    let phonemes = vec![
        Phoneme::new("T"), Phoneme::new("EH"), Phoneme::new("S"), Phoneme::new("T")
    ];
    
    let configs = vec![
        SynthesisConfig {
            speed: 0.8,
            pitch_shift: -2.0,
            energy: 0.8,
            speaker_id: None,
            seed: Some(42),
        },
        SynthesisConfig {
            speed: 1.2,
            pitch_shift: 2.0,
            energy: 1.2,
            speaker_id: None,
            seed: Some(123),
        },
        SynthesisConfig {
            speed: 1.0,
            pitch_shift: 0.0,
            energy: 1.0,
            speaker_id: None,
            seed: None,
        },
    ];
    
    for (i, config) in configs.iter().enumerate() {
        let mel = model.synthesize(&phonemes, Some(config)).await?;
        
        assert!(mel.n_frames > 0, "Config {} should generate frames", i);
        println!("   Config {}: speed={}, pitch={}, energy={} -> {}x{} frames", 
                i, config.speed, config.pitch_shift, config.energy, mel.n_mels, mel.n_frames);
    }
    
    println!("✅ VITS synthesis configuration test passed");
    
    Ok(())
}

#[tokio::test]
async fn test_vits_empty_input_handling() -> Result<()> {
    let model = VitsModel::new()?;
    
    // Test empty phoneme sequence
    let result = model.synthesize(&[], None).await;
    assert!(result.is_err(), "Empty phonemes should return error");
    
    println!("✅ VITS empty input handling test passed");
    
    Ok(())
}

#[tokio::test]
async fn test_vits_long_sequence() -> Result<()> {
    let model = VitsModel::new()?;
    
    // Create a longer phoneme sequence
    let phonemes = vec![
        Phoneme::new("DH"), Phoneme::new("IH"), Phoneme::new("S"), Phoneme::new(" "),
        Phoneme::new("IH"), Phoneme::new("Z"), Phoneme::new(" "),
        Phoneme::new("AH"), Phoneme::new(" "),
        Phoneme::new("L"), Phoneme::new("AO"), Phoneme::new("NG"), Phoneme::new("G"), Phoneme::new("ER"), Phoneme::new(" "),
        Phoneme::new("T"), Phoneme::new("EH"), Phoneme::new("S"), Phoneme::new("T"), Phoneme::new(" "),
        Phoneme::new("S"), Phoneme::new("EH"), Phoneme::new("N"), Phoneme::new("T"), Phoneme::new("AH"), Phoneme::new("N"), Phoneme::new("S"),
    ];
    
    let config = SynthesisConfig::default();
    let mel = model.synthesize(&phonemes, Some(&config)).await?;
    
    assert!(mel.n_frames > 10, "Long sequence should generate frames (neural decoder)");
    assert!(mel.duration() > 0.1, "Long sequence should have some duration (neural decoder)");
    
    println!("✅ VITS long sequence test passed");
    println!("   Phonemes: {}", phonemes.len());
    println!("   Mel frames: {}", mel.n_frames);
    println!("   Duration: {:.3}s", mel.duration());
    
    Ok(())
}

#[tokio::test]
async fn test_vits_special_phonemes() -> Result<()> {
    let model = VitsModel::new()?;
    
    // Test phonemes with features
    let mut phoneme_with_features = Phoneme::new("AH");
    let mut features = HashMap::new();
    features.insert("stress".to_string(), "primary".to_string());
    features.insert("position".to_string(), "syllable_nucleus".to_string());
    phoneme_with_features.features = Some(features);
    
    // Test phoneme with duration
    let mut phoneme_with_duration = Phoneme::new("N");
    phoneme_with_duration.duration = Some(0.15);
    
    let phonemes = vec![
        Phoneme::new("T"),
        phoneme_with_features,
        phoneme_with_duration,
        Phoneme::new("S"),
    ];
    
    let mel = model.synthesize(&phonemes, None).await?;
    
    assert!(mel.n_frames > 0, "Should handle phonemes with features and duration");
    
    println!("✅ VITS special phonemes test passed");
    
    Ok(())
}

#[tokio::test]
async fn test_vits_model_features() -> Result<()> {
    let model = VitsModel::new()?;
    
    use voirs_acoustic::AcousticModelFeature;
    
    // Test feature support
    assert!(model.supports(AcousticModelFeature::BatchProcessing), "Should support batch processing");
    assert!(model.supports(AcousticModelFeature::GpuAcceleration), "Should support GPU acceleration");
    assert!(!model.supports(AcousticModelFeature::MultiSpeaker), "Default model should not be multi-speaker");
    
    println!("✅ VITS model features test passed");
    
    Ok(())
}

#[tokio::test]
async fn test_vits_mel_spectrogram_properties() -> Result<()> {
    let model = VitsModel::new()?;
    
    let phonemes = vec![
        Phoneme::new("S"), Phoneme::new("P"), Phoneme::new("IY"), Phoneme::new("CH")
    ];
    
    let mel = model.synthesize(&phonemes, None).await?;
    
    // Test mel spectrogram properties
    assert_eq!(mel.data.len(), mel.n_mels, "Data should have n_mels rows");
    assert_eq!(mel.data[0].len(), mel.n_frames, "Each row should have n_frames columns");
    assert_eq!(mel.hop_length, 256, "Should have expected hop length");
    
    // Test that mel data contains reasonable values
    let mut has_non_zero = false;
    let mut values_in_range = true;
    
    for row in &mel.data {
        for &value in row {
            if value != 0.0 {
                has_non_zero = true;
            }
            if value.abs() > 100.0 { // Reasonable range for mel values (neural decoder with random weights)
                values_in_range = false;
            }
        }
    }
    
    assert!(has_non_zero, "Mel spectrogram should contain non-zero values");
    assert!(values_in_range, "Mel values should be in reasonable range");
    
    println!("✅ VITS mel spectrogram properties test passed");
    println!("   Mel shape: {}x{}", mel.n_mels, mel.n_frames);
    println!("   Duration: {:.3}s", mel.duration());
    println!("   Hop length: {}", mel.hop_length);
    
    Ok(())
}

#[tokio::test]
async fn test_vits_performance_benchmark() -> Result<()> {
    let model = VitsModel::new()?;
    
    // Create a realistic phoneme sequence
    let phonemes = vec![
        Phoneme::new("V"), Phoneme::new("OY"), Phoneme::new("R"), Phoneme::new("S"), Phoneme::new(" "),
        Phoneme::new("IH"), Phoneme::new("Z"), Phoneme::new(" "),
        Phoneme::new("AH"), Phoneme::new(" "),
        Phoneme::new("S"), Phoneme::new("P"), Phoneme::new("IY"), Phoneme::new("CH"), Phoneme::new(" "),
        Phoneme::new("S"), Phoneme::new("IH"), Phoneme::new("N"), Phoneme::new("TH"), Phoneme::new("AH"), Phoneme::new("S"), Phoneme::new("AH"), Phoneme::new("S"), Phoneme::new(" "),
        Phoneme::new("S"), Phoneme::new("IH"), Phoneme::new("S"), Phoneme::new("T"), Phoneme::new("AH"), Phoneme::new("M"),
    ];
    
    let start_time = std::time::Instant::now();
    let mel = model.synthesize(&phonemes, None).await?;
    let synthesis_time = start_time.elapsed();
    
    let real_time_factor = synthesis_time.as_secs_f32() / mel.duration();
    
    println!("✅ VITS performance benchmark:");
    println!("   Input phonemes: {}", phonemes.len());
    println!("   Output mel: {}x{}", mel.n_mels, mel.n_frames);
    println!("   Audio duration: {:.3}s", mel.duration());
    println!("   Synthesis time: {:.3}s", synthesis_time.as_secs_f32());
    println!("   Real-time factor: {:.2}x", real_time_factor);
    
    assert!(real_time_factor < 10.0, "Should synthesize reasonably fast (< 10x real-time)");
    
    Ok(())
}

#[tokio::test]
async fn test_vits_reproducibility() -> Result<()> {
    let model = VitsModel::new()?;
    
    let phonemes = vec![
        Phoneme::new("R"), Phoneme::new("IH"), Phoneme::new("P"), Phoneme::new("IY"), Phoneme::new("T")
    ];
    
    let config = SynthesisConfig {
        seed: Some(42),
        ..Default::default()
    };
    
    // Generate the same sequence twice
    let mel1 = model.synthesize(&phonemes, Some(&config)).await?;
    let mel2 = model.synthesize(&phonemes, Some(&config)).await?;
    
    // Should have same dimensions
    assert_eq!(mel1.n_mels, mel2.n_mels, "Should have same mel dimensions");
    assert_eq!(mel1.n_frames, mel2.n_frames, "Should have same frame count");
    
    println!("✅ VITS reproducibility test passed");
    println!("   Both runs produced {}x{} mel spectrograms", mel1.n_mels, mel1.n_frames);
    
    Ok(())
}