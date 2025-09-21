//! Basic unit tests for voirs-vocoder
//!
//! Simplified tests that work with the actual API.

use voirs_vocoder::{
    AudioBuffer, DummyVocoder, MelSpectrogram, SynthesisConfig, Vocoder, VocoderFeature,
};

#[tokio::test]
async fn test_dummy_vocoder_basic() {
    let vocoder = DummyVocoder::new();

    // Test metadata
    let metadata = vocoder.metadata();
    assert_eq!(metadata.name, "Dummy Vocoder");
    assert_eq!(metadata.sample_rate, 22050);
    assert_eq!(metadata.mel_channels, 80);

    // Test feature support
    assert!(vocoder.supports(VocoderFeature::BatchProcessing));
    assert!(!vocoder.supports(VocoderFeature::StreamingInference));
}

#[tokio::test]
async fn test_dummy_vocoder_vocoding() {
    let vocoder = DummyVocoder::new();

    // Create test mel spectrogram
    let mel_data = vec![vec![0.5; 80]; 100]; // 100 frames, 80 mel bins
    let mel = MelSpectrogram::new(mel_data, 22050, 256);

    let audio = vocoder.vocode(&mel, None).await.unwrap();

    assert_eq!(audio.sample_rate(), 22050);
    assert_eq!(audio.channels(), 1);
    assert!(!audio.samples().is_empty());
    assert!(!audio.is_empty());
}

#[tokio::test]
async fn test_dummy_vocoder_with_config() {
    let vocoder = DummyVocoder::new();

    let config = SynthesisConfig {
        speed: 1.2,
        pitch_shift: 0.5,
        energy: 1.1,
        ..Default::default()
    };

    let mel_data = vec![vec![0.5; 80]; 100];
    let mel = MelSpectrogram::new(mel_data, 22050, 256);

    let audio = vocoder.vocode(&mel, Some(&config)).await.unwrap();

    assert!(!audio.samples().is_empty());
    assert_eq!(audio.sample_rate(), 22050);
}

#[tokio::test]
async fn test_audio_buffer_creation() {
    // Test basic audio buffer creation
    let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let buffer = AudioBuffer::new(samples.clone(), 44100, 1);

    assert_eq!(buffer.samples(), &samples);
    assert_eq!(buffer.sample_rate(), 44100);
    assert_eq!(buffer.channels(), 1);
    assert!(!buffer.is_empty());
}

#[tokio::test]
async fn test_audio_buffer_sine_wave() {
    let buffer = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);

    assert_eq!(buffer.sample_rate(), 44100);
    assert_eq!(buffer.channels(), 1);
    assert!(buffer.duration() > 0.9 && buffer.duration() < 1.1);
    assert!(buffer.samples().len() > 44000);
}

#[tokio::test]
async fn test_audio_buffer_silence() {
    let buffer = AudioBuffer::silence(0.5, 48000, 2);

    assert_eq!(buffer.sample_rate(), 48000);
    assert_eq!(buffer.channels(), 2);
    assert!(buffer.duration() > 0.49 && buffer.duration() < 0.51);

    // All samples should be zero
    for &sample in buffer.samples() {
        assert_eq!(sample, 0.0);
    }
}

#[tokio::test]
async fn test_mel_spectrogram_creation() {
    let mel_data = vec![vec![1.0, 2.0, 3.0]; 80]; // 80 mel bins, 3 frames
    let mel = MelSpectrogram::new(mel_data.clone(), 22050, 256);

    assert_eq!(mel.data, mel_data);
    assert_eq!(mel.n_mels, 80);
    assert_eq!(mel.n_frames, 3);
    assert_eq!(mel.sample_rate, 22050);
    assert_eq!(mel.hop_length, 256);
    assert!(mel.duration() > 0.0);
}

#[test]
fn test_synthesis_config_defaults() {
    let config = SynthesisConfig::default();

    assert_eq!(config.speed, 1.0);
    assert_eq!(config.pitch_shift, 0.0);
    assert_eq!(config.energy, 1.0);
    assert_eq!(config.speaker_id, None);
    assert_eq!(config.seed, None);
}

#[tokio::test]
async fn test_batch_processing() {
    let vocoder = DummyVocoder::new();

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
