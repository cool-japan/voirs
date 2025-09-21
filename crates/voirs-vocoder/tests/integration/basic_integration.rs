//! Basic integration tests
//!
//! Simplified integration tests that work with the actual API.

use std::sync::Arc;
use voirs_vocoder::{AudioBuffer, DummyVocoder, MelSpectrogram, SynthesisConfig, Vocoder};

#[tokio::test]
async fn test_complete_workflow() {
    let vocoder = Arc::new(DummyVocoder::new());

    // Create test mel spectrogram
    let mel_data = vec![vec![0.5; 80]; 100];
    let mel = MelSpectrogram::new(mel_data, 22050, 256);

    // Process mel to audio
    let audio = vocoder.vocode(&mel, None).await.unwrap();

    assert!(!audio.samples().is_empty());
    assert_eq!(audio.sample_rate(), 22050);
    assert_eq!(audio.channels(), 1);
    assert!(audio.duration() > 0.0);
}

#[tokio::test]
async fn test_batch_workflow() {
    let vocoder = Arc::new(DummyVocoder::new());

    let mel_data = vec![vec![0.5; 80]; 50];
    let mels = vec![
        MelSpectrogram::new(mel_data.clone(), 22050, 256),
        MelSpectrogram::new(mel_data.clone(), 22050, 256),
        MelSpectrogram::new(mel_data, 22050, 256),
    ];

    let results = vocoder.vocode_batch(&mels, None).await.unwrap();

    assert_eq!(results.len(), 3);
    for audio in results {
        assert!(!audio.samples().is_empty());
        assert_eq!(audio.sample_rate(), 22050);
    }
}

#[tokio::test]
async fn test_concurrent_processing() {
    let vocoder = Arc::new(DummyVocoder::new());

    let mut handles = Vec::new();

    for i in 0..3 {
        let vocoder_clone = vocoder.clone();
        let handle = tokio::spawn(async move {
            let mel_data = vec![vec![0.5; 80]; 50 + i * 10];
            let mel = MelSpectrogram::new(mel_data, 22050, 256);

            vocoder_clone.vocode(&mel, None).await
        });
        handles.push(handle);
    }

    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
        let audio = result.unwrap();
        assert!(!audio.samples().is_empty());
    }
}

#[tokio::test]
async fn test_different_configurations() {
    let vocoder = Arc::new(DummyVocoder::new());
    let mel_data = vec![vec![0.5; 80]; 50];
    let mel = MelSpectrogram::new(mel_data, 22050, 256);

    let configs = vec![
        SynthesisConfig {
            speed: 1.0,
            ..Default::default()
        },
        SynthesisConfig {
            speed: 1.2,
            ..Default::default()
        },
        SynthesisConfig {
            speed: 0.8,
            ..Default::default()
        },
    ];

    for config in configs {
        let audio = vocoder.vocode(&mel, Some(&config)).await.unwrap();
        assert!(!audio.samples().is_empty());
        assert_eq!(audio.sample_rate(), 22050);
    }
}

#[tokio::test]
async fn test_audio_buffer_operations() {
    // Test different audio operations
    let buffer1 = AudioBuffer::sine_wave(440.0, 0.5, 44100, 0.5);
    let buffer2 = AudioBuffer::sine_wave(880.0, 0.5, 44100, 0.3);

    assert_eq!(buffer1.sample_rate(), 44100);
    assert_eq!(buffer2.sample_rate(), 44100);
    assert!(buffer1.duration() > 0.49 && buffer1.duration() < 0.51);
    assert!(buffer2.duration() > 0.49 && buffer2.duration() < 0.51);

    // Test silence
    let silence = AudioBuffer::silence(1.0, 22050, 1);
    assert_eq!(silence.sample_rate(), 22050);
    assert!(silence.duration() > 0.99 && silence.duration() < 1.01);

    for &sample in silence.samples() {
        assert_eq!(sample, 0.0);
    }
}

#[tokio::test]
async fn test_mel_spectrogram_properties() {
    let mel_data = vec![vec![1.0, 2.0, 3.0, 4.0]; 80];
    let mel = MelSpectrogram::new(mel_data, 22050, 256);

    assert_eq!(mel.n_mels, 80);
    assert_eq!(mel.n_frames, 4);
    assert_eq!(mel.sample_rate, 22050);
    assert_eq!(mel.hop_length, 256);

    let expected_duration = (4 * 256) as f32 / 22050.0;
    assert!((mel.duration() - expected_duration).abs() < 0.001);
}

#[tokio::test]
async fn test_vocoder_metadata() {
    let vocoder = DummyVocoder::new();
    let metadata = vocoder.metadata();

    assert_eq!(metadata.name, "Dummy Vocoder");
    assert_eq!(metadata.version, "0.1.0");
    assert_eq!(metadata.architecture, "Sine Wave");
    assert_eq!(metadata.sample_rate, 22050);
    assert_eq!(metadata.mel_channels, 80);
    assert_eq!(metadata.quality_score, 2.0);
}

#[tokio::test]
async fn test_error_handling() {
    let vocoder = DummyVocoder::new();

    // Test with empty mel spectrogram
    let empty_mel = MelSpectrogram::new(vec![], 22050, 256);
    let result = vocoder.vocode(&empty_mel, None).await;

    // Dummy vocoder should handle this gracefully
    assert!(result.is_ok());

    // Test with unusual configurations
    let unusual_config = SynthesisConfig {
        speed: 10.0,
        pitch_shift: 50.0,
        energy: 0.0,
        ..Default::default()
    };

    let mel_data = vec![vec![0.5; 80]; 50];
    let mel = MelSpectrogram::new(mel_data, 22050, 256);

    let result = vocoder.vocode(&mel, Some(&unusual_config)).await;
    assert!(result.is_ok());

    let audio = result.unwrap();
    assert!(!audio.samples().is_empty());

    // Check for stability - no NaN or infinite values
    for &sample in audio.samples() {
        assert!(sample.is_finite());
        assert!(!sample.is_nan());
    }
}
