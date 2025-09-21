//! Basic quality tests for voirs-vocoder
//!
//! Simplified quality tests that work with the actual API.

use std::sync::Arc;
use voirs_vocoder::{AudioBuffer, DummyVocoder, MelSpectrogram, SynthesisConfig, Vocoder};

/// Basic quality metrics
#[derive(Debug)]
pub struct BasicQualityMetrics {
    /// Peak level
    pub peak: f32,
    /// RMS level
    pub rms: f32,
    /// Dynamic range (peak/rms)
    pub dynamic_range: f32,
    /// Zero crossing rate
    pub zero_crossing_rate: f32,
    /// Has clipping
    pub has_clipping: bool,
    /// Sample count
    pub sample_count: usize,
}

impl BasicQualityMetrics {
    pub fn analyze(audio: &AudioBuffer) -> Self {
        let samples = audio.samples();

        // Calculate peak
        let peak = samples.iter().map(|x| x.abs()).fold(0.0, f32::max);

        // Calculate RMS
        let rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt();

        // Calculate dynamic range
        let dynamic_range = if rms > 0.0 { peak / rms } else { 0.0 };

        // Calculate zero crossing rate
        let mut zero_crossings = 0;
        for i in 1..samples.len() {
            if (samples[i - 1] >= 0.0) != (samples[i] >= 0.0) {
                zero_crossings += 1;
            }
        }
        let zero_crossing_rate = if samples.len() > 1 {
            zero_crossings as f32 / (samples.len() - 1) as f32
        } else {
            0.0
        };

        // Check for clipping
        let has_clipping = samples.iter().any(|&x| x.abs() >= 0.99);

        Self {
            peak,
            rms,
            dynamic_range,
            zero_crossing_rate,
            has_clipping,
            sample_count: samples.len(),
        }
    }
}

#[tokio::test]
async fn test_basic_audio_quality() {
    let vocoder = Arc::new(DummyVocoder::new());

    let mel_data = vec![vec![0.5; 80]; 100];
    let mel = MelSpectrogram::new(mel_data, 22050, 256);

    let audio = vocoder.vocode(&mel, None).await.unwrap();
    let metrics = BasicQualityMetrics::analyze(&audio);

    // Basic quality checks
    assert!(metrics.peak > 0.0);
    assert!(metrics.peak <= 1.0);
    assert!(metrics.rms > 0.0);
    assert!(metrics.rms <= metrics.peak);
    assert!(metrics.dynamic_range > 0.0);
    assert!(metrics.zero_crossing_rate >= 0.0);
    assert!(metrics.zero_crossing_rate <= 1.0);
    assert!(metrics.sample_count > 0);

    // Should not be clipping
    assert!(!metrics.has_clipping);
}

#[tokio::test]
async fn test_sine_wave_quality() {
    // Test known good audio (sine wave)
    let sine_wave = AudioBuffer::sine_wave(440.0, 1.0, 44100, 0.5);
    let metrics = BasicQualityMetrics::analyze(&sine_wave);

    // Sine wave should have specific characteristics
    assert!((metrics.peak - 0.5).abs() < 0.01); // Should be close to amplitude
    assert!(metrics.rms > 0.3 && metrics.rms < 0.4); // RMS of sine wave ≈ amplitude/√2
    assert!(metrics.dynamic_range > 1.0); // Should have some dynamic range
    assert!(metrics.zero_crossing_rate > 0.0); // Sine wave crosses zero regularly
    assert!(!metrics.has_clipping); // Should not clip at 0.5 amplitude
}

#[tokio::test]
async fn test_silence_quality() {
    let silence = AudioBuffer::silence(1.0, 44100, 1);
    let metrics = BasicQualityMetrics::analyze(&silence);

    // Silence should have zero characteristics
    assert_eq!(metrics.peak, 0.0);
    assert_eq!(metrics.rms, 0.0);
    assert_eq!(metrics.dynamic_range, 0.0);
    assert_eq!(metrics.zero_crossing_rate, 0.0);
    assert!(!metrics.has_clipping);
}

#[tokio::test]
async fn test_quality_consistency() {
    let vocoder = Arc::new(DummyVocoder::new());

    let mel_data = vec![vec![0.5; 80]; 50];
    let mel = MelSpectrogram::new(mel_data, 22050, 256);

    // Generate the same audio multiple times
    let audio1 = vocoder.vocode(&mel, None).await.unwrap();
    let audio2 = vocoder.vocode(&mel, None).await.unwrap();

    let metrics1 = BasicQualityMetrics::analyze(&audio1);
    let metrics2 = BasicQualityMetrics::analyze(&audio2);

    // Results should be identical for dummy vocoder
    assert_eq!(metrics1.peak, metrics2.peak);
    assert_eq!(metrics1.rms, metrics2.rms);
    assert_eq!(metrics1.sample_count, metrics2.sample_count);
}

#[tokio::test]
async fn test_quality_across_configurations() {
    let vocoder = Arc::new(DummyVocoder::new());
    let mel_data = vec![vec![0.5; 80]; 100];
    let mel = MelSpectrogram::new(mel_data, 22050, 256);

    let configs = vec![
        SynthesisConfig {
            speed: 1.0,
            ..Default::default()
        },
        SynthesisConfig {
            speed: 1.5,
            ..Default::default()
        },
        SynthesisConfig {
            speed: 0.8,
            ..Default::default()
        },
    ];

    let mut all_metrics = Vec::new();

    for config in configs {
        let audio = vocoder.vocode(&mel, Some(&config)).await.unwrap();
        let metrics = BasicQualityMetrics::analyze(&audio);

        // All should produce valid audio
        assert!(metrics.peak > 0.0);
        assert!(metrics.rms > 0.0);
        assert!(!metrics.has_clipping);
        assert!(metrics.sample_count > 0);

        all_metrics.push(metrics);
    }

    // All should be reasonable quality
    for metrics in all_metrics {
        assert!(metrics.peak <= 1.0);
        assert!(metrics.dynamic_range > 0.0);
    }
}

#[tokio::test]
async fn test_clipping_detection() {
    // Create audio that should clip
    let loud_samples = vec![1.5f32; 1000]; // Above clipping threshold
    let clipped_audio = AudioBuffer::new(
        loud_samples.iter().map(|&x| x.clamp(-1.0, 1.0)).collect(),
        44100,
        1,
    );

    let metrics = BasicQualityMetrics::analyze(&clipped_audio);

    // Should detect clipping
    assert!(metrics.has_clipping);
    assert_eq!(metrics.peak, 1.0);
}

#[tokio::test]
async fn test_batch_quality_consistency() {
    let vocoder = Arc::new(DummyVocoder::new());

    let mel_data = vec![vec![0.5; 80]; 50];
    let mels = vec![
        MelSpectrogram::new(mel_data.clone(), 22050, 256),
        MelSpectrogram::new(mel_data.clone(), 22050, 256),
        MelSpectrogram::new(mel_data, 22050, 256),
    ];

    let results = vocoder.vocode_batch(&mels, None).await.unwrap();

    assert_eq!(results.len(), 3);

    let mut all_metrics = Vec::new();
    for audio in results {
        let metrics = BasicQualityMetrics::analyze(&audio);
        all_metrics.push(metrics);
    }

    // All batch results should have identical quality (for dummy vocoder)
    for i in 1..all_metrics.len() {
        assert_eq!(all_metrics[i].peak, all_metrics[0].peak);
        assert_eq!(all_metrics[i].rms, all_metrics[0].rms);
        assert_eq!(all_metrics[i].sample_count, all_metrics[0].sample_count);
    }
}

#[tokio::test]
async fn test_stability_over_time() {
    let vocoder = Arc::new(DummyVocoder::new());

    // Process many audio clips to test stability
    for i in 0..10 {
        let mel_data = vec![vec![0.5; 80]; 20 + i * 5];
        let mel = MelSpectrogram::new(mel_data, 22050, 256);

        let audio = vocoder.vocode(&mel, None).await.unwrap();
        let metrics = BasicQualityMetrics::analyze(&audio);

        // Should remain stable
        assert!(metrics.peak > 0.0);
        assert!(metrics.peak <= 1.0);
        assert!(metrics.rms > 0.0);
        assert!(!metrics.has_clipping);

        // Check for any problematic values
        for &sample in audio.samples() {
            assert!(sample.is_finite());
            assert!(!sample.is_nan());
        }
    }
}

#[tokio::test]
async fn test_different_mel_sizes() {
    let vocoder = Arc::new(DummyVocoder::new());

    // Test different mel spectrogram sizes
    let sizes = vec![10, 50, 100, 200];

    for size in sizes {
        let mel_data = vec![vec![0.5; 80]; size];
        let mel = MelSpectrogram::new(mel_data, 22050, 256);

        let audio = vocoder.vocode(&mel, None).await.unwrap();
        let metrics = BasicQualityMetrics::analyze(&audio);

        // Should produce valid audio regardless of size
        assert!(metrics.peak > 0.0);
        assert!(metrics.rms > 0.0);
        assert!(!metrics.has_clipping);

        // Audio length should be proportional to mel time frames (80 frames for all sizes)
        let expected_duration = (80 * 256) as f32 / 22050.0; // 80 time frames, hop_length=256
        let actual_duration = audio.duration();
        assert!(
            (actual_duration - expected_duration).abs() < 0.01,
            "Duration mismatch: expected {expected_duration:.3}s, got {actual_duration:.3}s",
        );
    }
}
