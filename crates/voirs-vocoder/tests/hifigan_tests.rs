//! Comprehensive tests for HiFi-GAN vocoder

use voirs_vocoder::{
    Vocoder, VocoderFeature, MelSpectrogram, AudioBuffer, SynthesisConfig,
    hifigan::{HiFiGanVocoder, HiFiGanVariants, HiFiGanVariant, HiFiGanConfig},
    models::hifigan::{
        generator::{HiFiGanGenerator, GeneratorStats},
        mrf::{MultiReceptiveField, MRFConfig, MRFStats},
        variants::{VariantModifications, VariantComparison},
    },
    Result,
};

#[tokio::test]
async fn test_hifigan_vocoder_creation() -> Result<()> {
    // Test default creation
    let vocoder = HiFiGanVocoder::new();
    let metadata = vocoder.metadata();
    
    assert_eq!(metadata.architecture, "HiFi-GAN");
    assert_eq!(metadata.sample_rate, 22050);
    assert_eq!(metadata.mel_channels, 80);
    assert!(metadata.quality_score > 4.0);
    
    println!("✅ HiFi-GAN vocoder created successfully");
    println!("   Metadata: {:?}", metadata);
    
    Ok(())
}

#[tokio::test]
async fn test_hifigan_variants() -> Result<()> {
    let variants = vec![
        (HiFiGanVariant::V1, "highest quality"),
        (HiFiGanVariant::V2, "balanced"),
        (HiFiGanVariant::V3, "fastest"),
    ];
    
    for (variant, description) in variants {
        let vocoder = HiFiGanVocoder::with_variant(variant);
        let metadata = vocoder.metadata();
        
        assert_eq!(metadata.name, variant.name());
        println!("   {} ({}): quality={:.1}, latency={:.1}ms", 
                variant.name(), description, metadata.quality_score, metadata.latency_ms);
    }
    
    println!("✅ HiFi-GAN variants test passed");
    
    Ok(())
}

#[tokio::test]
async fn test_hifigan_mel_to_audio() -> Result<()> {
    let mut vocoder = HiFiGanVocoder::with_variant(HiFiGanVariant::V3); // Use fastest for testing
    vocoder.initialize_inference_for_testing()?;
    
    // Create test mel spectrogram
    let mel_data = create_test_mel_spectrogram(80, 100); // 80 mels, 100 frames
    // Use hop_length that matches the V3 upsampling factor (1024 = 8*8*8*2)
    let mel = MelSpectrogram::new(mel_data, 22050, 1024);
    
    let audio = vocoder.vocode(&mel, None).await?;
    
    assert!(audio.duration() > 0.0, "Should generate audio with duration");
    assert!(!audio.samples().is_empty(), "Should generate audio samples");
    assert_eq!(audio.sample_rate(), 22050, "Should have correct sample rate");
    assert_eq!(audio.channels(), 1, "Should be mono audio");
    
    // Calculate expected duration based on mel frames and hop length
    let expected_duration = (mel.n_frames * mel.hop_length as usize) as f32 / mel.sample_rate as f32;
    let actual_duration = audio.duration();
    let duration_ratio = actual_duration / expected_duration;
    
    assert!(duration_ratio > 0.5 && duration_ratio < 2.0, 
           "Audio duration should be reasonable relative to mel spectrogram");
    
    println!("✅ HiFi-GAN mel-to-audio test passed");
    println!("   Mel: {}x{} frames", mel.n_mels, mel.n_frames);
    println!("   Audio: {:.3}s, {} samples", audio.duration(), audio.samples().len());
    println!("   Duration ratio: {:.2}", duration_ratio);
    
    Ok(())
}

#[tokio::test]
async fn test_hifigan_batch_processing() -> Result<()> {
    let mut vocoder = HiFiGanVocoder::with_variant(HiFiGanVariant::V3);
    vocoder.initialize_inference_for_testing()?;
    
    let mels = vec![
        MelSpectrogram::new(create_test_mel_spectrogram(80, 50), 22050, 1024), // V3 upsampling factor
        MelSpectrogram::new(create_test_mel_spectrogram(80, 75), 22050, 1024),
        MelSpectrogram::new(create_test_mel_spectrogram(80, 100), 22050, 1024),
    ];
    
    let configs = vec![SynthesisConfig::default(); 3];
    let audio_buffers = vocoder.vocode_batch(&mels, Some(&configs)).await?;
    
    assert_eq!(audio_buffers.len(), 3, "Should generate audio for each mel");
    
    for (i, audio) in audio_buffers.iter().enumerate() {
        assert!(audio.duration() > 0.0, "Batch item {} should have duration", i);
        assert!(!audio.samples().is_empty(), "Batch item {} should have samples", i);
        println!("   Batch item {}: {:.3}s, {} samples", i, audio.duration(), audio.samples().len());
    }
    
    println!("✅ HiFi-GAN batch processing test passed");
    
    Ok(())
}

#[tokio::test]
async fn test_hifigan_synthesis_config() -> Result<()> {
    let mut vocoder = HiFiGanVocoder::with_variant(HiFiGanVariant::V2);
    vocoder.initialize_inference_for_testing()?;
    let mel = MelSpectrogram::new(create_test_mel_spectrogram(80, 80), 22050, 512); // V2 upsampling factor: 8*8*4*2=512
    
    let configs = vec![
        SynthesisConfig {
            speed: 0.8,
            pitch_shift: -2.0,
            energy: 0.7,
            speaker_id: None,
            seed: Some(42),
        },
        SynthesisConfig {
            speed: 1.2,
            pitch_shift: 3.0,
            energy: 1.3,
            speaker_id: None,
            seed: Some(123),
        },
        SynthesisConfig::default(),
    ];
    
    for (i, config) in configs.iter().enumerate() {
        let audio = vocoder.vocode(&mel, Some(config)).await?;
        
        assert!(audio.duration() > 0.0, "Config {} should generate audio", i);
        
        // Test audio properties
        let samples = audio.samples();
        let peak = samples.iter().map(|s| s.abs()).fold(0.0, f32::max);
        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        
        assert!(peak <= 1.0, "Audio should not clip");
        assert!(rms > 0.0, "Audio should have energy");
        
        println!("   Config {}: speed={}, pitch={}, energy={} -> {:.3}s, peak={:.3}, rms={:.3}", 
                i, config.speed, config.pitch_shift, config.energy, audio.duration(), peak, rms);
    }
    
    println!("✅ HiFi-GAN synthesis config test passed");
    
    Ok(())
}

#[tokio::test]
async fn test_hifigan_vocoder_features() -> Result<()> {
    let vocoder = HiFiGanVocoder::default();
    
    // Test feature support
    assert!(vocoder.supports(VocoderFeature::StreamingInference), "Should support streaming");
    assert!(vocoder.supports(VocoderFeature::BatchProcessing), "Should support batch processing");
    assert!(vocoder.supports(VocoderFeature::HighQuality), "Should support high quality");
    
    let v3_vocoder = HiFiGanVocoder::with_variant(HiFiGanVariant::V3);
    assert!(v3_vocoder.supports(VocoderFeature::RealtimeProcessing), "V3 should support realtime");
    
    println!("✅ HiFi-GAN vocoder features test passed");
    
    Ok(())
}

#[tokio::test]
async fn test_hifigan_variant_analysis() -> Result<()> {
    let comparisons = HiFiGanVariants::compare_variants();
    
    assert_eq!(comparisons.len(), 3, "Should analyze all 3 variants");
    
    for comparison in &comparisons {
        assert!(comparison.parameters > 0, "Should have parameters");
        assert!(comparison.model_size_mb > 0.0, "Should have model size");
        assert!(comparison.upsampling_factor > 0, "Should have upsampling factor");
        assert!(comparison.quality_score > 0.0, "Should have quality score");
        assert!(comparison.speed_score > 0.0, "Should have speed score");
        
        println!("   {}: {:.1}M params, {:.1}MB, quality={:.1}, speed={:.1}", 
                comparison.name, comparison.parameters as f32 / 1_000_000.0, 
                comparison.model_size_mb, comparison.quality_score, comparison.speed_score);
    }
    
    // V1 should have more parameters than V2, V2 more than V3
    let v1 = comparisons.iter().find(|c| c.name.contains("V1")).unwrap();
    let v2 = comparisons.iter().find(|c| c.name.contains("V2")).unwrap();
    let v3 = comparisons.iter().find(|c| c.name.contains("V3")).unwrap();
    
    assert!(v1.parameters > v2.parameters, "V1 should have more parameters than V2");
    assert!(v2.parameters > v3.parameters, "V2 should have more parameters than V3");
    assert!(v1.quality_score > v3.quality_score, "V1 should have higher quality than V3");
    assert!(v3.speed_score > v1.speed_score, "V3 should be faster than V1");
    
    println!("✅ HiFi-GAN variant analysis test passed");
    
    Ok(())
}

#[tokio::test]
async fn test_hifigan_custom_variant() -> Result<()> {
    let modifications = VariantModifications::new()
        .sample_rate(44100)
        .mel_channels(128)
        .initial_channels(1024);
    
    let custom_config = HiFiGanVariants::custom(HiFiGanVariant::V2, modifications);
    let vocoder = HiFiGanVocoder::with_config(custom_config.clone());
    
    assert_eq!(custom_config.sample_rate, 44100);
    assert_eq!(custom_config.mel_channels, 128);
    assert_eq!(custom_config.initial_channels, 1024);
    
    // Should retain other V2 characteristics
    assert_eq!(custom_config.upsample_rates, vec![8, 8, 4, 2]);
    
    println!("✅ HiFi-GAN custom variant test passed");
    
    Ok(())
}

#[tokio::test]
async fn test_hifigan_generator_stats() -> Result<()> {
    let config = HiFiGanVariants::v2();
    
    #[cfg(not(feature = "candle"))]
    {
        let generator = HiFiGanGenerator::new(config)?;
        let stats = generator.stats();
        
        assert!(stats.num_parameters > 0, "Should have parameters");
        assert!(stats.model_size_bytes > 0, "Should have model size");
        assert_eq!(stats.upsampling_factor, 512, "V2 should have 512x upsampling"); // 8*8*4*2
        assert!(stats.receptive_field_size > 0, "Should have receptive field");
        
        println!("✅ HiFi-GAN generator stats test passed");
        println!("   Parameters: {:.1}M", stats.num_parameters as f32 / 1_000_000.0);
        println!("   Model size: {:.1} MB", stats.model_size_bytes as f32 / 1_000_000.0);
        println!("   Upsampling: {}x", stats.upsampling_factor);
        println!("   Receptive field: {}", stats.receptive_field_size);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_mrf_configuration() -> Result<()> {
    let config = MRFConfig::default();
    
    assert_eq!(config.channels, 512);
    assert_eq!(config.kernel_sizes, vec![3, 7, 11]);
    assert_eq!(config.dilation_sizes.len(), 3);
    assert_eq!(config.leaky_relu_slope, 0.1);
    
    #[cfg(not(feature = "candle"))]
    {
        let mrf = MultiReceptiveField::new(
            config.channels,
            &config.kernel_sizes,
            &config.dilation_sizes,
            config.leaky_relu_slope,
        )?;
        
        let stats = mrf.stats();
        assert_eq!(stats.num_blocks, 3);
        assert!(stats.num_parameters > 0);
        assert_eq!(stats.receptive_field_size, 51); // Max receptive field for kernel=11, dilation=5
        assert_eq!(stats.avg_kernel_size, 7.0); // (3+7+11)/3
        
        println!("✅ MRF configuration test passed");
        println!("   Blocks: {}", stats.num_blocks);
        println!("   Parameters: {:.1}M", stats.num_parameters as f32 / 1_000_000.0);
        println!("   Receptive field: {}", stats.receptive_field_size);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_hifigan_audio_quality() -> Result<()> {
    let mut vocoder = HiFiGanVocoder::with_variant(HiFiGanVariant::V1); // Highest quality
    vocoder.initialize_inference_for_testing()?;
    
    // Create realistic mel spectrogram with formant-like patterns
    let mel = create_realistic_mel_spectrogram();
    let audio = vocoder.vocode(&mel, None).await?;
    
    let samples = audio.samples();
    
    // Test basic audio quality metrics
    let peak = samples.iter().map(|s| s.abs()).fold(0.0, f32::max);
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    let dynamic_range = if rms > 0.0 { 20.0 * (peak / rms).log10() } else { 0.0 };
    
    // Test for clipping
    let clipped_samples = samples.iter().filter(|&&s| s.abs() >= 0.99).count();
    let clipping_rate = clipped_samples as f32 / samples.len() as f32;
    
    // Test for silence
    let silent_samples = samples.iter().filter(|&&s| s.abs() < 0.001).count();
    let silence_rate = silent_samples as f32 / samples.len() as f32;
    
    assert!(peak > 0.01, "Audio should have reasonable amplitude");
    assert!(peak <= 1.0, "Audio should not exceed maximum amplitude");
    assert!(rms > 0.001, "Audio should have measurable energy");
    assert!(clipping_rate < 0.01, "Should have minimal clipping (< 1%)");
    assert!(silence_rate < 0.9, "Should not be mostly silent");
    assert!(dynamic_range > 10.0, "Should have reasonable dynamic range");
    
    println!("✅ HiFi-GAN audio quality test passed");
    println!("   Peak: {:.3}", peak);
    println!("   RMS: {:.3}", rms);
    println!("   Dynamic range: {:.1} dB", dynamic_range);
    println!("   Clipping rate: {:.2}%", clipping_rate * 100.0);
    println!("   Silence rate: {:.2}%", silence_rate * 100.0);
    
    Ok(())
}

#[tokio::test]
async fn test_hifigan_performance_benchmark() -> Result<()> {
    let mut vocoder = HiFiGanVocoder::with_variant(HiFiGanVariant::V3); // Fastest variant
    vocoder.initialize_inference_for_testing()?;
    
    let mel = MelSpectrogram::new(create_test_mel_spectrogram(80, 200), 22050, 1024); // V3 upsampling factor
    
    let start_time = std::time::Instant::now();
    let audio = vocoder.vocode(&mel, None).await?;
    let vocoding_time = start_time.elapsed();
    
    let real_time_factor = vocoding_time.as_secs_f32() / audio.duration();
    
    println!("✅ HiFi-GAN performance benchmark:");
    println!("   Mel: {}x{} frames", mel.n_mels, mel.n_frames);
    println!("   Audio: {:.3}s, {} samples", audio.duration(), audio.samples().len());
    println!("   Vocoding time: {:.3}s", vocoding_time.as_secs_f32());
    println!("   Real-time factor: {:.2}x", real_time_factor);
    
    // V3 should be reasonably fast
    assert!(real_time_factor < 5.0, "V3 variant should vocode reasonably fast (< 5x real-time)");
    
    Ok(())
}

// Helper functions

fn create_test_mel_spectrogram(n_mels: usize, n_frames: usize) -> Vec<Vec<f32>> {
    let mut data = vec![vec![0.0; n_frames]; n_mels];
    
    // Generate simple patterns for testing
    for (mel_idx, row) in data.iter_mut().enumerate() {
        for (frame_idx, value) in row.iter_mut().enumerate() {
            // Create a simple sinusoidal pattern with some formant-like structure
            let freq_factor = mel_idx as f32 / n_mels as f32;
            let time_factor = frame_idx as f32 / n_frames as f32;
            
            *value = 0.1 * (freq_factor * 10.0 + time_factor * 5.0).sin();
            
            // Add some formant-like peaks
            if mel_idx > 10 && mel_idx < 30 {
                *value += 0.3 * (time_factor * std::f32::consts::PI).sin();
            }
        }
    }
    
    data
}

fn create_realistic_mel_spectrogram() -> MelSpectrogram {
    let n_mels = 80;
    let n_frames = 150;
    let mut data = vec![vec![0.0; n_frames]; n_mels];
    
    // Create more realistic mel spectrogram with formant structure
    for (mel_idx, row) in data.iter_mut().enumerate() {
        let mel_freq = mel_idx as f32 / n_mels as f32;
        
        for (frame_idx, value) in row.iter_mut().enumerate() {
            let time = frame_idx as f32 / n_frames as f32;
            
            // Base energy
            *value = -2.0;
            
            // Add formant peaks
            let formants = [0.1, 0.25, 0.4, 0.65]; // Normalized formant positions
            for &formant in &formants {
                let distance = (mel_freq - formant).abs();
                let formant_energy = 2.0 * (-distance * 20.0).exp();
                *value += formant_energy;
            }
            
            // Add temporal variation
            *value += 0.5 * (time * std::f32::consts::TAU).sin();
            
            // Add some noise
            *value += 0.1 * (fastrand::f32() - 0.5);
            
            // Ensure reasonable range
            *value = value.max(-4.0).min(2.0);
        }
    }
    
    MelSpectrogram::new(data, 22050, 256)
}