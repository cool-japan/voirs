//! End-to-end integration tests
//!
//! Tests complete workflows from mel spectrograms to audio output
//! across different vocoders and configurations.

use voirs_vocoder::{
    AudioBuffer, MelSpectrogram, DummyVocoder, Vocoder, SynthesisConfig,
    config::{VocodingConfig, StreamingConfig, ModelConfig, QualityLevel},
    effects::{EffectChain, Compressor, ParametricEQ},
    streaming::{StreamingPipeline, StreamingBuffer},
    VocoderError
};
use std::sync::Arc;
use tempfile::NamedTempFile;
use std::io::Write;

#[tokio::test]
async fn test_complete_vocoding_workflow() {
    // Create a complete workflow: mel -> vocoder -> effects -> output
    let vocoder = Arc::new(DummyVocoder::new());
    
    // Create test mel spectrogram
    let mel_data = create_test_mel_spectrogram(100, 80);
    let mel = MelSpectrogram::new(mel_data, 22050, 256);
    
    // Step 1: Vocode mel to audio
    let raw_audio = vocoder.vocode(&mel, None).await.unwrap();
    assert!(raw_audio.samples().len() > 0);
    assert_eq!(raw_audio.sample_rate(), 22050);
    assert_eq!(raw_audio.channels(), 1);
    
    // Step 2: Apply effects
    let mut effect_chain = EffectChain::new();
    let compressor = Box::new(Compressor::new(4.0, -12.0, 10.0, 50.0).unwrap());
    let eq = Box::new(ParametricEQ::new(2000.0, 3.0, 1.0, 22050).unwrap());
    
    effect_chain.add_effect(compressor);
    effect_chain.add_effect(eq);
    
    let mut processed_audio = raw_audio.clone();
    effect_chain.process(&mut processed_audio).unwrap();
    
    // Step 3: Validate output quality
    assert!(processed_audio.peak() <= 1.0);
    assert!(processed_audio.rms() > 0.0);
    assert_eq!(processed_audio.sample_rate(), raw_audio.sample_rate());
    assert_eq!(processed_audio.channels(), raw_audio.channels());
    
    // Step 4: Export to file
    let mut temp_file = NamedTempFile::new().unwrap();
    processed_audio.write_wav(temp_file.path()).unwrap();
    
    // Verify file was created and has content
    let metadata = std::fs::metadata(temp_file.path()).unwrap();
    assert!(metadata.len() > 44); // At least WAV header size
}

#[tokio::test]
async fn test_batch_processing_workflow() {
    let vocoder = Arc::new(DummyVocoder::new());
    
    // Create multiple mel spectrograms
    let mel_data1 = create_test_mel_spectrogram(50, 80);
    let mel_data2 = create_test_mel_spectrogram(75, 80);
    let mel_data3 = create_test_mel_spectrogram(100, 80);
    
    let mels = vec![
        MelSpectrogram::new(mel_data1, 22050, 256),
        MelSpectrogram::new(mel_data2, 22050, 256),
        MelSpectrogram::new(mel_data3, 22050, 256),
    ];
    
    // Process batch
    let audio_buffers = vocoder.vocode_batch(&mels, None).await.unwrap();
    
    assert_eq!(audio_buffers.len(), 3);
    
    // Verify each output
    for (i, audio) in audio_buffers.iter().enumerate() {
        assert!(audio.samples().len() > 0);
        assert_eq!(audio.sample_rate(), 22050);
        assert_eq!(audio.channels(), 1);
        
        // Different input sizes should produce different output sizes
        if i > 0 {
            assert_ne!(audio.samples().len(), audio_buffers[0].samples().len());
        }
    }
    
    // Test batch with different configurations
    let configs = vec![
        SynthesisConfig { speed: 1.0, pitch: 1.0, energy: 1.0, ..Default::default() },
        SynthesisConfig { speed: 1.2, pitch: 0.9, energy: 1.1, ..Default::default() },
        SynthesisConfig { speed: 0.8, pitch: 1.1, energy: 0.9, ..Default::default() },
    ];
    
    let configured_audio = vocoder.vocode_batch(&mels, Some(&configs)).await.unwrap();
    
    assert_eq!(configured_audio.len(), 3);
    
    // Different configs should produce different outputs
    for i in 1..configured_audio.len() {
        assert_ne!(configured_audio[i].peak(), configured_audio[0].peak());
    }
}

#[tokio::test]
async fn test_streaming_workflow() {
    let vocoder = Arc::new(DummyVocoder::new());
    let mut pipeline = StreamingPipeline::new(vocoder);
    
    let config = StreamingConfig::default();
    pipeline.initialize(config).await.unwrap();
    
    // Create streaming mel data
    let mel_chunks = create_streaming_mel_data(10, 50, 80);
    
    // Process streaming data
    let mut output_chunks = Vec::new();
    
    for mel_chunk in mel_chunks {
        let audio_chunk = pipeline.process_chunk(mel_chunk).await.unwrap();
        output_chunks.push(audio_chunk);
    }
    
    assert_eq!(output_chunks.len(), 10);
    
    // Verify streaming output
    for chunk in &output_chunks {
        assert!(chunk.samples().len() > 0);
        assert_eq!(chunk.sample_rate(), 22050);
        assert_eq!(chunk.channels(), 1);
        assert!(chunk.peak() <= 1.0);
    }
    
    // Concatenate chunks to form complete audio
    let mut complete_audio = output_chunks[0].clone();
    for chunk in &output_chunks[1..] {
        complete_audio = complete_audio.concatenate(chunk).unwrap();
    }
    
    assert!(complete_audio.duration() > 1.0); // Should be substantial duration
    assert!(complete_audio.samples().len() > 22050); // > 1 second of audio
}

#[tokio::test]
async fn test_quality_level_comparison() {
    let vocoder = Arc::new(DummyVocoder::new());
    
    // Create test mel spectrogram
    let mel_data = create_test_mel_spectrogram(100, 80);
    let mel = MelSpectrogram::new(mel_data, 22050, 256);
    
    // Test different quality levels
    let quality_levels = vec![
        QualityLevel::Low,
        QualityLevel::Medium,
        QualityLevel::High,
        QualityLevel::Ultra,
    ];
    
    let mut results = Vec::new();
    
    for quality in quality_levels {
        let config = SynthesisConfig {
            quality_level: Some(quality),
            ..Default::default()
        };
        
        let start_time = std::time::Instant::now();
        let audio = vocoder.vocode(&mel, Some(&config)).await.unwrap();
        let processing_time = start_time.elapsed();
        
        results.push((quality, audio, processing_time));
    }
    
    // Verify quality progression
    for (quality, audio, time) in &results {
        assert!(audio.samples().len() > 0);
        assert!(audio.peak() <= 1.0);
        assert!(audio.rms() > 0.0);
        
        // Higher quality might take longer (for real vocoders)
        // For dummy vocoder, time should be consistent
        assert!(time.as_millis() < 100);
    }
}

#[tokio::test]
async fn test_error_handling_workflow() {
    let vocoder = Arc::new(DummyVocoder::new());
    
    // Test with invalid mel spectrogram
    let empty_mel = MelSpectrogram::new(vec![], 22050, 256);
    let result = vocoder.vocode(&empty_mel, None).await;
    
    // Dummy vocoder should handle this gracefully
    assert!(result.is_ok());
    
    // Test with mismatched sample rates
    let mel_data = create_test_mel_spectrogram(50, 80);
    let weird_mel = MelSpectrogram::new(mel_data, 8000, 256); // Unusual sample rate
    
    let result = vocoder.vocode(&weird_mel, None).await;
    assert!(result.is_ok()); // Dummy vocoder should adapt
    
    // Test with extreme synthesis parameters
    let extreme_config = SynthesisConfig {
        speed: 10.0,  // Very fast
        pitch: 0.1,   // Very low
        energy: 5.0,  // Very high
        ..Default::default()
    };
    
    let normal_mel = MelSpectrogram::new(create_test_mel_spectrogram(50, 80), 22050, 256);
    let result = vocoder.vocode(&normal_mel, Some(&extreme_config)).await;
    
    assert!(result.is_ok());
    let audio = result.unwrap();
    
    // Output should still be valid
    assert!(audio.samples().len() > 0);
    assert!(audio.peak() <= 1.0);
    
    // Check for stability
    for &sample in audio.samples() {
        assert!(sample.is_finite());
        assert!(!sample.is_nan());
    }
}

#[tokio::test]
async fn test_memory_management_workflow() {
    let vocoder = Arc::new(DummyVocoder::new());
    
    // Process many small chunks to test memory management
    for i in 0..100 {
        let mel_data = create_test_mel_spectrogram(20, 80);
        let mel = MelSpectrogram::new(mel_data, 22050, 256);
        
        let audio = vocoder.vocode(&mel, None).await.unwrap();
        
        // Verify each result is valid
        assert!(audio.samples().len() > 0);
        assert_eq!(audio.sample_rate(), 22050);
        
        // Memory usage shouldn't grow unboundedly
        // (This is more relevant for real implementations with GPU memory)
        if i % 20 == 0 {
            // Simulate periodic cleanup
            std::hint::black_box(&audio); // Prevent optimization
        }
    }
}

#[tokio::test]
async fn test_concurrent_processing() {
    let vocoder = Arc::new(DummyVocoder::new());
    
    // Create multiple processing tasks
    let mut handles = Vec::new();
    
    for i in 0..5 {
        let vocoder_clone = vocoder.clone();
        let handle = tokio::spawn(async move {
            let mel_data = create_test_mel_spectrogram(50 + i * 10, 80);
            let mel = MelSpectrogram::new(mel_data, 22050, 256);
            
            let config = SynthesisConfig {
                speed: 1.0 + i as f32 * 0.1,
                ..Default::default()
            };
            
            let audio = vocoder_clone.vocode(&mel, Some(&config)).await.unwrap();
            (i, audio)
        });
        
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    let mut results = Vec::new();
    for handle in handles {
        let result = handle.await.unwrap();
        results.push(result);
    }
    
    // Verify all results are valid and different
    assert_eq!(results.len(), 5);
    
    for (i, audio) in results {
        assert!(audio.samples().len() > 0);
        assert_eq!(audio.sample_rate(), 22050);
        assert!(audio.peak() <= 1.0);
    }
}

#[tokio::test]
async fn test_format_conversion_workflow() {
    let vocoder = Arc::new(DummyVocoder::new());
    
    // Generate audio
    let mel_data = create_test_mel_spectrogram(100, 80);
    let mel = MelSpectrogram::new(mel_data, 22050, 256);
    let audio = vocoder.vocode(&mel, None).await.unwrap();
    
    // Test different format conversions
    let i16_data = audio.to_i16();
    assert_eq!(i16_data.len(), audio.samples().len());
    
    let i24_data = audio.to_i24();
    assert_eq!(i24_data.len(), audio.samples().len());
    
    let i32_data = audio.to_i32();
    assert_eq!(i32_data.len(), audio.samples().len());
    
    // Test round-trip conversion
    let recovered_audio = AudioBuffer::from_i16(&i16_data, audio.sample_rate(), audio.channels()).unwrap();
    assert_eq!(recovered_audio.samples().len(), audio.samples().len());
    assert_eq!(recovered_audio.sample_rate(), audio.sample_rate());
    
    // Verify conversion accuracy (allowing for quantization)
    for (original, recovered) in audio.samples().iter().zip(recovered_audio.samples().iter()) {
        let error = (original - recovered).abs();
        assert!(error < 0.01); // Within quantization error
    }
}

#[tokio::test]
async fn test_sample_rate_conversion_workflow() {
    let vocoder = Arc::new(DummyVocoder::new());
    
    // Generate audio at 22050 Hz
    let mel_data = create_test_mel_spectrogram(100, 80);
    let mel = MelSpectrogram::new(mel_data, 22050, 256);
    let audio_22k = vocoder.vocode(&mel, None).await.unwrap();
    
    // Convert to different sample rates
    let audio_44k = audio_22k.resample(44100).unwrap();
    let audio_48k = audio_22k.resample(48000).unwrap();
    let audio_16k = audio_22k.resample(16000).unwrap();
    
    // Verify sample rate conversions
    assert_eq!(audio_44k.sample_rate(), 44100);
    assert_eq!(audio_48k.sample_rate(), 48000);
    assert_eq!(audio_16k.sample_rate(), 16000);
    
    // Duration should be preserved
    let duration_tolerance = 0.01; // 10ms tolerance
    assert!((audio_44k.duration() - audio_22k.duration()).abs() < duration_tolerance);
    assert!((audio_48k.duration() - audio_22k.duration()).abs() < duration_tolerance);
    assert!((audio_16k.duration() - audio_22k.duration()).abs() < duration_tolerance);
    
    // Sample counts should be proportional to sample rates
    let ratio_44k = audio_44k.samples().len() as f32 / audio_22k.samples().len() as f32;
    let ratio_48k = audio_48k.samples().len() as f32 / audio_22k.samples().len() as f32;
    let ratio_16k = audio_16k.samples().len() as f32 / audio_22k.samples().len() as f32;
    
    assert!((ratio_44k - 2.0).abs() < 0.1); // 44100/22050 ≈ 2.0
    assert!((ratio_48k - 2.18).abs() < 0.1); // 48000/22050 ≈ 2.18
    assert!((ratio_16k - 0.73).abs() < 0.1); // 16000/22050 ≈ 0.73
}

#[tokio::test]
async fn test_channel_conversion_workflow() {
    let vocoder = Arc::new(DummyVocoder::new());
    
    // Generate mono audio
    let mel_data = create_test_mel_spectrogram(100, 80);
    let mel = MelSpectrogram::new(mel_data, 22050, 256);
    let mono_audio = vocoder.vocode(&mel, None).await.unwrap();
    
    assert_eq!(mono_audio.channels(), 1);
    
    // Convert to stereo
    let stereo_audio = mono_audio.to_stereo().unwrap();
    assert_eq!(stereo_audio.channels(), 2);
    assert_eq!(stereo_audio.samples().len(), mono_audio.samples().len() * 2);
    assert_eq!(stereo_audio.sample_rate(), mono_audio.sample_rate());
    
    // Convert back to mono
    let back_to_mono = stereo_audio.to_mono().unwrap();
    assert_eq!(back_to_mono.channels(), 1);
    assert_eq!(back_to_mono.samples().len(), mono_audio.samples().len());
    assert_eq!(back_to_mono.sample_rate(), mono_audio.sample_rate());
    
    // Content should be similar (allowing for conversion artifacts)
    for (original, converted) in mono_audio.samples().iter().zip(back_to_mono.samples().iter()) {
        let error = (original - converted).abs();
        assert!(error < 0.01);
    }
}

// Helper functions

fn create_test_mel_spectrogram(frames: usize, mel_bins: usize) -> Vec<Vec<f32>> {
    let mut mel_data = Vec::new();
    
    for frame in 0..frames {
        let mut mel_frame = Vec::new();
        
        for bin in 0..mel_bins {
            // Create some realistic mel spectrogram values
            let freq_component = (bin as f32 / mel_bins as f32) * 2.0 - 1.0; // -1 to 1
            let time_component = (frame as f32 / frames as f32) * 2.0 * std::f32::consts::PI;
            
            // Simulate mel spectrogram with frequency and temporal structure
            let value = -5.0 + 3.0 * (freq_component * time_component.sin()).abs();
            mel_frame.push(value);
        }
        
        mel_data.push(mel_frame);
    }
    
    mel_data
}

fn create_streaming_mel_data(chunks: usize, frames_per_chunk: usize, mel_bins: usize) -> Vec<MelSpectrogram> {
    let mut mel_chunks = Vec::new();
    
    for chunk_idx in 0..chunks {
        let mel_data = create_test_mel_spectrogram(frames_per_chunk, mel_bins);
        let mel = MelSpectrogram::new(mel_data, 22050, 256);
        mel_chunks.push(mel);
    }
    
    mel_chunks
}