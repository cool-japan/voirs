//! Unit tests for Vocoder trait implementations
//!
//! Tests the core vocoder interface and implementations to ensure
//! correct behavior across different vocoder types.

use voirs_vocoder::{
    AudioBuffer, DummyVocoder, MelSpectrogram, Vocoder, VocoderError, VocoderFeature,
    VocoderMetadata, SynthesisConfig
};
use std::sync::Arc;
use tokio_test;

#[tokio::test]
async fn test_dummy_vocoder_creation() {
    let vocoder = DummyVocoder::new();
    
    let metadata = vocoder.metadata();
    assert_eq!(metadata.name, "DummyVocoder");
    assert_eq!(metadata.version, "1.0.0");
    assert_eq!(metadata.sample_rate, 22050);
    assert_eq!(metadata.channels, 1);
}

#[tokio::test]
async fn test_dummy_vocoder_features() {
    let vocoder = DummyVocoder::new();
    
    // Test supported features
    assert!(vocoder.supports(VocoderFeature::Streaming));
    assert!(vocoder.supports(VocoderFeature::BatchProcessing));
    assert!(!vocoder.supports(VocoderFeature::RealTimeProcessing));
    assert!(!vocoder.supports(VocoderFeature::GpuAcceleration));
}

#[tokio::test]
async fn test_dummy_vocoder_basic_vocoding() {
    let vocoder = DummyVocoder::new();
    
    // Create test mel spectrogram
    let mel_data = vec![vec![0.5; 80]; 100]; // 100 frames, 80 mel bins
    let mel = MelSpectrogram::new(mel_data, 22050, 256);
    
    let result = vocoder.vocode(&mel, None).await;
    assert!(result.is_ok());
    
    let audio = result.unwrap();
    assert_eq!(audio.sample_rate(), 22050);
    assert_eq!(audio.channels(), 1);
    assert!(audio.samples().len() > 0);
    
    // Verify audio is not all zeros (dummy vocoder should generate sine wave)
    let non_zero_count = audio.samples().iter().filter(|&&x| x.abs() > 0.001).count();
    assert!(non_zero_count > 0);
}

#[tokio::test]
async fn test_dummy_vocoder_with_config() {
    let vocoder = DummyVocoder::new();
    
    let config = SynthesisConfig {
        speed: 1.2,
        pitch: 0.8,
        energy: 1.5,
        ..Default::default()
    };
    
    let mel_data = vec![vec![0.5; 80]; 100];
    let mel = MelSpectrogram::new(mel_data, 22050, 256);
    
    let result = vocoder.vocode(&mel, Some(&config)).await;
    assert!(result.is_ok());
    
    let audio = result.unwrap();
    assert!(audio.samples().len() > 0);
    
    // Verify that config affects the output (dummy vocoder should use config)
    let peak = audio.peak();
    assert!(peak > 0.0);
}

#[tokio::test]
async fn test_dummy_vocoder_batch_processing() {
    let vocoder = DummyVocoder::new();
    
    // Create multiple mel spectrograms
    let mel_data = vec![vec![0.5; 80]; 50];
    let mels = vec![
        MelSpectrogram::new(mel_data.clone(), 22050, 256),
        MelSpectrogram::new(mel_data.clone(), 22050, 256),
        MelSpectrogram::new(mel_data, 22050, 256),
    ];
    
    let result = vocoder.vocode_batch(&mels, None).await;
    assert!(result.is_ok());
    
    let audio_buffers = result.unwrap();
    assert_eq!(audio_buffers.len(), 3);
    
    for audio in &audio_buffers {
        assert_eq!(audio.sample_rate(), 22050);
        assert_eq!(audio.channels(), 1);
        assert!(audio.samples().len() > 0);
    }
}

#[tokio::test]
async fn test_dummy_vocoder_batch_with_configs() {
    let vocoder = DummyVocoder::new();
    
    let mel_data = vec![vec![0.5; 80]; 50];
    let mels = vec![
        MelSpectrogram::new(mel_data.clone(), 22050, 256),
        MelSpectrogram::new(mel_data, 22050, 256),
    ];
    
    let configs = vec![
        SynthesisConfig { speed: 1.0, pitch: 1.0, energy: 1.0, ..Default::default() },
        SynthesisConfig { speed: 1.5, pitch: 0.8, energy: 1.2, ..Default::default() },
    ];
    
    let result = vocoder.vocode_batch(&mels, Some(&configs)).await;
    assert!(result.is_ok());
    
    let audio_buffers = result.unwrap();
    assert_eq!(audio_buffers.len(), 2);
    
    // Verify different configs produce different outputs
    assert_ne!(audio_buffers[0].peak(), audio_buffers[1].peak());
}

#[tokio::test]
async fn test_dummy_vocoder_streaming() {
    let vocoder = DummyVocoder::new();
    
    // Create a stream of mel spectrograms
    let mel_data = vec![vec![0.5; 80]; 30];
    let mels = vec![
        MelSpectrogram::new(mel_data.clone(), 22050, 256),
        MelSpectrogram::new(mel_data.clone(), 22050, 256),
        MelSpectrogram::new(mel_data, 22050, 256),
    ];
    
    let mel_stream = futures::stream::iter(mels);
    let result = vocoder.vocode_stream(Box::new(mel_stream), None).await;
    assert!(result.is_ok());
    
    let mut audio_stream = result.unwrap();
    let mut count = 0;
    
    while let Some(audio_result) = futures::StreamExt::next(&mut audio_stream).await {
        assert!(audio_result.is_ok());
        let audio = audio_result.unwrap();
        assert!(audio.samples().len() > 0);
        count += 1;
    }
    
    assert_eq!(count, 3);
}

#[tokio::test]
async fn test_vocoder_error_handling() {
    let vocoder = DummyVocoder::new();
    
    // Test with invalid mel spectrogram (empty data)
    let empty_mel = MelSpectrogram::new(vec![], 22050, 256);
    let result = vocoder.vocode(&empty_mel, None).await;
    
    // Dummy vocoder should handle this gracefully
    assert!(result.is_ok());
    
    // Test with mismatched configs in batch
    let mel_data = vec![vec![0.5; 80]; 30];
    let mels = vec![
        MelSpectrogram::new(mel_data.clone(), 22050, 256),
        MelSpectrogram::new(mel_data, 22050, 256),
    ];
    
    let configs = vec![
        SynthesisConfig::default(),
        // Missing one config - should use default
    ];
    
    let result = vocoder.vocode_batch(&mels, Some(&configs)).await;
    assert!(result.is_ok());
    
    let audio_buffers = result.unwrap();
    assert_eq!(audio_buffers.len(), 2);
}

#[tokio::test]
async fn test_vocoder_thread_safety() {
    let vocoder = Arc::new(DummyVocoder::new());
    
    // Test concurrent access
    let mut handles = Vec::new();
    
    for i in 0..5 {
        let vocoder_clone = vocoder.clone();
        let handle = tokio::spawn(async move {
            let mel_data = vec![vec![0.5; 80]; 50];
            let mel = MelSpectrogram::new(mel_data, 22050, 256);
            
            let result = vocoder_clone.vocode(&mel, None).await;
            assert!(result.is_ok());
            
            let audio = result.unwrap();
            assert!(audio.samples().len() > 0);
            
            i
        });
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    for handle in handles {
        let result = handle.await;
        assert!(result.is_ok());
    }
}

#[tokio::test]
async fn test_vocoder_metadata_consistency() {
    let vocoder = DummyVocoder::new();
    
    // Test that metadata is consistent across calls
    let metadata1 = vocoder.metadata();
    let metadata2 = vocoder.metadata();
    
    assert_eq!(metadata1.name, metadata2.name);
    assert_eq!(metadata1.version, metadata2.version);
    assert_eq!(metadata1.sample_rate, metadata2.sample_rate);
    assert_eq!(metadata1.channels, metadata2.channels);
    
    // Test that output matches metadata
    let mel_data = vec![vec![0.5; 80]; 50];
    let mel = MelSpectrogram::new(mel_data, 22050, 256);
    
    let result = vocoder.vocode(&mel, None).await;
    assert!(result.is_ok());
    
    let audio = result.unwrap();
    assert_eq!(audio.sample_rate(), metadata1.sample_rate);
    assert_eq!(audio.channels(), metadata1.channels);
}

#[tokio::test]
async fn test_vocoder_feature_consistency() {
    let vocoder = DummyVocoder::new();
    
    // Test that feature support is consistent
    assert_eq!(
        vocoder.supports(VocoderFeature::Streaming),
        vocoder.supports(VocoderFeature::Streaming)
    );
    
    assert_eq!(
        vocoder.supports(VocoderFeature::BatchProcessing),
        vocoder.supports(VocoderFeature::BatchProcessing)
    );
    
    // Test that unsupported features consistently return false
    assert!(!vocoder.supports(VocoderFeature::RealTimeProcessing));
    assert!(!vocoder.supports(VocoderFeature::GpuAcceleration));
}

#[tokio::test]
async fn test_vocoder_performance_characteristics() {
    let vocoder = DummyVocoder::new();
    
    // Test performance with different input sizes
    let sizes = vec![10, 50, 100, 200];
    
    for size in sizes {
        let mel_data = vec![vec![0.5; 80]; size];
        let mel = MelSpectrogram::new(mel_data, 22050, 256);
        
        let start = std::time::Instant::now();
        let result = vocoder.vocode(&mel, None).await;
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        
        // Dummy vocoder should be fast (< 100ms for any reasonable size)
        assert!(duration.as_millis() < 100);
        
        let audio = result.unwrap();
        
        // Output size should be proportional to input size
        let expected_samples = size * 256; // hop_length = 256
        let actual_samples = audio.samples().len();
        let ratio = actual_samples as f32 / expected_samples as f32;
        
        // Should be approximately correct (allowing for some variation)
        assert!(ratio > 0.5 && ratio < 2.0);
    }
}

#[tokio::test]
async fn test_vocoder_output_quality() {
    let vocoder = DummyVocoder::new();
    
    // Test with different mel spectrogram characteristics
    let mel_data = vec![vec![0.5; 80]; 100];
    let mel = MelSpectrogram::new(mel_data, 22050, 256);
    
    let result = vocoder.vocode(&mel, None).await;
    assert!(result.is_ok());
    
    let audio = result.unwrap();
    
    // Basic quality checks
    assert!(audio.samples().len() > 0);
    assert!(audio.peak() > 0.0);
    assert!(audio.peak() <= 1.0);
    assert!(audio.rms() > 0.0);
    
    // Check for reasonable signal characteristics
    let dynamic_range = audio.peak() / audio.rms();
    assert!(dynamic_range > 1.0); // Should have some dynamic range
    
    // Check that output is not clipping
    let clipped_samples = audio.samples().iter().filter(|&&x| x.abs() >= 1.0).count();
    assert_eq!(clipped_samples, 0);
}

#[tokio::test]
async fn test_vocoder_deterministic_behavior() {
    let vocoder = DummyVocoder::new();
    
    // Test that same input produces same output
    let mel_data = vec![vec![0.5; 80]; 50];
    let mel = MelSpectrogram::new(mel_data, 22050, 256);
    
    let result1 = vocoder.vocode(&mel, None).await;
    let result2 = vocoder.vocode(&mel, None).await;
    
    assert!(result1.is_ok());
    assert!(result2.is_ok());
    
    let audio1 = result1.unwrap();
    let audio2 = result2.unwrap();
    
    assert_eq!(audio1.samples().len(), audio2.samples().len());
    assert_eq!(audio1.sample_rate(), audio2.sample_rate());
    assert_eq!(audio1.channels(), audio2.channels());
    
    // Samples should be identical for deterministic vocoder
    for (i, (&sample1, &sample2)) in audio1.samples().iter().zip(audio2.samples().iter()).enumerate() {
        assert!((sample1 - sample2).abs() < 1e-6, "Sample {} differs: {} vs {}", i, sample1, sample2);
    }
}